"""Telegram ↔ Claude bridge.

Lets you chat with Claude from Telegram without a laptop.
Also provides bot status commands:

  /status  — check if Morpheus is running + cost summary
  /balance — check Kalshi account balances
  /logs    — last 20 lines of bot logs

All other messages are forwarded to Claude and the response sent back.

Runs as a separate systemd service (telegram-claude.service).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
STATE_DIR = Path("/root/Morpheus/state")
LOG_FILE = Path("/root/Morpheus/morpheus.log")

# Conversation history (in-memory, resets on restart)
conversation: list[dict[str, str]] = []
MAX_HISTORY = 20


async def send_telegram(text: str, chat_id: str = CHAT_ID) -> None:
    """Send a message via Telegram, splitting if too long."""
    async with httpx.AsyncClient() as client:
        # Telegram max message length is 4096
        for i in range(0, len(text), 4000):
            chunk = text[i:i + 4000]
            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"},
                timeout=30,
            )


async def handle_status() -> str:
    """Return bot status and cost summary."""
    # Systemd status
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "morpheus.service"],
            capture_output=True, text=True, timeout=5,
        )
        service_status = result.stdout.strip()
    except Exception:
        service_status = "unknown"

    # Cost tracker
    cost_file = STATE_DIR / "cost_tracker.json"
    cost_info = "No cost data yet"
    if cost_file.exists():
        try:
            data = json.loads(cost_file.read_text())
            cost_info = (
                f"Month: {data.get('month')}\n"
                f"Total spend: ${data.get('total_spend', 0):.2f}\n"
                f"Daily spend: ${data.get('daily_spend', 0):.2f}\n"
                f"Budget remaining: ${data.get('monthly_budget', 100) - data.get('total_spend', 0):.2f}\n"
                f"Calls: {data.get('call_count', 0)}"
            )
        except Exception:
            pass

    return f"*Morpheus Status*\nService: `{service_status}`\n\n{cost_info}"


async def handle_balance() -> str:
    """Check Kalshi balances."""
    try:
        from kalshi_python import Configuration, KalshiClient
        from kalshi_python.api.portfolio_api import PortfolioApi

        lines = []
        for label, key_env, path_env in [
            ("Primary", "KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH"),
            ("Secondary", "KALSHI_API_KEY_ID_2", "KALSHI_PRIVATE_KEY_PATH_2"),
        ]:
            key_id = os.getenv(key_env, "")
            key_path = os.getenv(path_env, "")
            if not key_id or not key_path:
                lines.append(f"{label}: not configured")
                continue
            cfg = Configuration(host="https://api.elections.kalshi.com/trade-api/v2")
            cfg.api_key_id = key_id
            with open(key_path) as f:
                cfg.private_key_pem = f.read()
            api = KalshiClient(cfg)
            portfolio = PortfolioApi(api)
            bal = portfolio.get_balance()
            usd = bal.balance / 100.0 if hasattr(bal, "balance") else 0.0
            lines.append(f"{label}: ${usd:.2f}")

        return "*Kalshi Balances*\n" + "\n".join(lines)
    except Exception as e:
        return f"Balance check failed: {e}"


async def handle_logs() -> str:
    """Return last 20 lines of bot logs."""
    if not LOG_FILE.exists():
        return "No log file found"
    try:
        result = subprocess.run(
            ["tail", "-20", str(LOG_FILE)],
            capture_output=True, text=True, timeout=5,
        )
        # Strip ANSI escape codes for readability
        import re
        clean = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
        return f"```\n{clean[-3500:]}\n```"
    except Exception as e:
        return f"Error reading logs: {e}"


async def ask_claude(user_message: str) -> str:
    """Forward message to Claude and return response."""
    client = AsyncAnthropic(api_key=ANTHROPIC_KEY)

    conversation.append({"role": "user", "content": user_message})
    # Trim history
    while len(conversation) > MAX_HISTORY:
        conversation.pop(0)

    try:
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1000,
            system=(
                "You are a helpful assistant accessible via Telegram. "
                "Keep responses concise (under 500 words) since they're read on a phone. "
                "You help the user manage their Morpheus prediction market trading bot. "
                "You have knowledge of Python, trading, Kalshi, and Polymarket."
            ),
            messages=conversation,
        )
        reply = response.content[0].text if response.content else "No response"
        conversation.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"Claude error: {e}"


async def poll_updates() -> None:
    """Long-poll Telegram for new messages."""
    offset = 0
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            try:
                resp = await client.get(
                    f"{TELEGRAM_API}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                )
                data = resp.json()
                if not data.get("ok"):
                    await asyncio.sleep(5)
                    continue

                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    # Only respond to authorized chat
                    if chat_id != CHAT_ID:
                        continue

                    if not text:
                        continue

                    # Handle commands
                    if text.strip().lower() == "/status":
                        reply = await handle_status()
                    elif text.strip().lower() == "/balance":
                        reply = await handle_balance()
                    elif text.strip().lower() == "/logs":
                        reply = await handle_logs()
                    elif text.strip().lower() == "/help":
                        reply = (
                            "*Commands:*\n"
                            "/status — Bot status + cost summary\n"
                            "/balance — Kalshi account balances\n"
                            "/logs — Last 20 lines of bot logs\n"
                            "/help — This message\n\n"
                            "Any other message → chat with Claude"
                        )
                    else:
                        reply = await ask_claude(text)

                    await send_telegram(reply, chat_id)

            except httpx.TimeoutException:
                continue
            except Exception as e:
                print(f"Poll error: {e}")
                await asyncio.sleep(5)


def main() -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID or not ANTHROPIC_KEY:
        print("Missing TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, or ANTHROPIC_API_KEY")
        return
    print("Telegram-Claude bridge started. Listening for messages...")
    asyncio.run(poll_updates())


if __name__ == "__main__":
    main()
