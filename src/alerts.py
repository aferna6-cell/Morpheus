"""Telegram alerts stub.

Sends alerts via Telegram Bot API if configured, otherwise logs.
Wire into main.py for trade execution, daily P&L, and error alerts.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
import structlog

from .utils import BotConfig

logger = structlog.get_logger()


async def send_alert(message: str, config: Optional[BotConfig] = None) -> None:
    """Send an alert message.

    Checks for telegram config in env vars (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
    Falls back to logging if not configured.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.info("alert", message=message)
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
    except Exception as e:
        logger.warning("telegram_alert_failed", error=str(e), message=message[:200])
