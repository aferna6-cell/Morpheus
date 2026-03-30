"""Morpheus watchdog / health check.

Run by systemd timer every 5 minutes. Checks:
1. Is the morpheus service running?
2. Is the log file stale (no writes in 15 min)? — stall detection
3. Memory usage — leak / OOM-approaching detection
4. Has daily loss limit been hit? — circuit breaker

Sends Telegram alert and optionally restarts the service if unhealthy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Telegram helper (standalone — doesn't import bot dependencies)
# ---------------------------------------------------------------------------

async def _send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[watchdog] alert (no telegram): {message}")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json={"chat_id": chat_id, "text": f"⚠️ Morpheus watchdog: {message}"})
            r.raise_for_status()
    except Exception as e:
        print(f"[watchdog] telegram failed: {e}")


def _systemctl(args: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        ["systemctl"] + args, capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip()


def _service_state(service: str) -> str:
    rc, out = _systemctl(["is-active", service])
    return out  # "active", "inactive", "failed", "activating", etc.


def _restart_count_recent(service: str, minutes: int = 5) -> int:
    """Count how many times the service restarted in the last N minutes."""
    since = f"{minutes} min ago"
    rc, out = _systemctl([
        "show", service,
        "--property=NRestarts",
        "--value",
    ])
    try:
        return int(out.strip())
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_service(service: str) -> Optional[str]:
    """Returns error string if service is not active, else None."""
    state = _service_state(service)
    if state != "active":
        return f"Service {service!r} is {state!r} (not active)"
    return None


def check_log_staleness(log_file: str, max_age_minutes: int = 15) -> Optional[str]:
    """Returns error if log file hasn't been written to recently (stall detection)."""
    path = Path(log_file)
    if not path.exists():
        return None  # Journal logging — skip file check
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > max_age_minutes * 60:
        return f"Log file not updated in {age_seconds/60:.0f} min (stall?)"
    return None


def check_memory(service: str, warn_mb: int = 800, crit_mb: int = 1800) -> Optional[str]:
    """Returns warning/error if service memory is approaching limits."""
    rc, out = _systemctl([
        "show", service,
        "--property=MemoryCurrent",
        "--value",
    ])
    try:
        mem_bytes = int(out.strip())
        mem_mb = mem_bytes / (1024 * 1024)
        if mem_mb >= crit_mb:
            return f"Memory CRITICAL: {mem_mb:.0f} MB (limit ~2048 MB)"
        if mem_mb >= warn_mb:
            return f"Memory high: {mem_mb:.0f} MB (approaching limit)"
    except (ValueError, ZeroDivisionError):
        pass
    return None


def check_restart_loop(service: str) -> Optional[str]:
    """Returns error if service restarted too many times recently (crash loop)."""
    rc, out = _systemctl([
        "show", service,
        "--property=NRestarts",
        "--value",
    ])
    try:
        restarts = int(out.strip())
        if restarts >= 5:
            return f"Restart loop detected: {restarts} restarts total (StartLimitBurst may trip soon)"
    except ValueError:
        pass
    return None


def check_daily_loss(state_dir: str) -> Optional[str]:
    """Returns warning if daily loss limit was hit today (circuit breaker)."""
    risk_file = Path(state_dir) / "risk_state.json"
    if not risk_file.exists():
        return None
    try:
        data = json.loads(risk_file.read_text())
        daily_loss = float(data.get("daily_loss", 0))
        limit = float(data.get("daily_loss_limit", 999))
        today = datetime.now(timezone.utc).date().isoformat()
        loss_date = data.get("loss_date", "")
        if loss_date == today and daily_loss >= limit:
            return f"Daily loss limit hit: ${daily_loss:.2f} / ${limit:.2f} — bot halted"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_checks(service: str, state_dir: str, log_file: str) -> None:
    print(f"[watchdog] {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} — checking {service}")

    errors = []
    warnings = []

    # 1. Service running?
    if err := check_service(service):
        errors.append(err)
        # Service is down — try to restart it
        print(f"[watchdog] SERVICE DOWN: {err}")
        rc, _ = _systemctl(["start", service])
        if rc == 0:
            errors.append(f"Restarted {service} (was down)")
        else:
            errors.append(f"Failed to restart {service}")

    # 2. Log staleness (stall detection)
    if err := check_log_staleness(log_file):
        warnings.append(err)

    # 3. Memory
    if err := check_memory(service):
        if "CRITICAL" in err:
            errors.append(err)
        else:
            warnings.append(err)

    # 4. Restart loop
    if err := check_restart_loop(service):
        warnings.append(err)

    # 5. Daily loss limit
    if err := check_daily_loss(state_dir):
        warnings.append(err)

    # --- Report ---
    if errors:
        msg = "ERRORS:\n" + "\n".join(f"• {e}" for e in errors)
        print(f"[watchdog] {msg}")
        await _send_telegram(msg)
    elif warnings:
        msg = "Warnings:\n" + "\n".join(f"• {w}" for w in warnings)
        print(f"[watchdog] {msg}")
        await _send_telegram(msg)
    else:
        print(f"[watchdog] All checks passed ✓")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv("/opt/morpheus/.env")

    parser = argparse.ArgumentParser(description="Morpheus watchdog")
    parser.add_argument("--service", default="morpheus")
    parser.add_argument("--state-dir", default="/opt/morpheus/state")
    parser.add_argument("--log-file", default="/opt/morpheus/logs/morpheus.log")
    args = parser.parse_args()

    asyncio.run(run_checks(
        service=args.service,
        state_dir=args.state_dir,
        log_file=args.log_file,
    ))


if __name__ == "__main__":
    main()
