"""Optional notification hooks.

Currently supports Telegram if env vars are set:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

If not configured, calls are no-ops.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
import structlog


class TelegramNotifier:
    def __init__(self) -> None:
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.logger = structlog.get_logger()

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(self, text: str) -> None:
        if not self.enabled:
            self.logger.debug("telegram_disabled")
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
        except Exception as e:
            self.logger.warning("telegram_send_failed", error=str(e))
