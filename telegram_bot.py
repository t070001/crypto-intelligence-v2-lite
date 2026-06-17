from __future__ import annotations

import os

import requests

from config import TELEGRAM_BOT_TOKEN as CFG_BOT_TOKEN
from config import TELEGRAM_CHAT_ID as CFG_CHAT_ID


def _get_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or CFG_BOT_TOKEN
    if not token or token == "YOUR_BOT_TOKEN_HERE":
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    return token


def _get_chat_id() -> str:
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or CFG_CHAT_ID
    if not chat_id or chat_id == "YOUR_CHAT_ID_HERE":
        raise RuntimeError("Missing TELEGRAM_CHAT_ID")
    return chat_id


def send_message(text: str) -> None:
    """Send a Telegram message. Splits into chunks if exceeds Telegram's 4096 char limit."""
    token = _get_token()
    chat_id = _get_chat_id()

    # Split if message is too long (Telegram limit ~4096 chars)
    max_len = 4000
    if len(text) <= max_len:
        chunks = [text]
    else:
        # Split at double newlines first (between recommendations)
        lines = text.split("\n\n")
        chunks = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 2 > max_len:
                chunks.append(current.strip())
                current = line
            else:
                current = (current + "\n\n" + line).strip() if current else line
        if current.strip():
            chunks.append(current.strip())

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
