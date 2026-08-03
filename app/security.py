from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Optional


def password_hash(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"{base64.urlsafe_b64encode(salt).decode()}.{base64.urlsafe_b64encode(digest).decode()}"


def password_ok(password: str, encoded: str) -> bool:
    try:
        salt64, expected64 = encoded.split(".", 1)
        salt = base64.urlsafe_b64decode(salt64)
        expected = base64.urlsafe_b64decode(expected64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def load_secret(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(os.urandom(32))
        path.chmod(0o600)
    return path.read_bytes()


def make_token(user_id: int, secret: bytes, days: int) -> str:
    payload = {"uid": user_id, "exp": int(time.time()) + days * 86400}
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=")
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return raw.decode() + "." + base64.urlsafe_b64encode(sig).rstrip(b"=").decode()


def read_token(token: str, secret: bytes) -> Optional[int]:
    try:
        raw64, sig64 = token.split(".", 1)
        raw = raw64.encode()
        sig = base64.urlsafe_b64decode(sig64 + "=" * (-len(sig64) % 4))
        if not hmac.compare_digest(sig, hmac.new(secret, raw, hashlib.sha256).digest()):
            return None
        payload = json.loads(base64.urlsafe_b64decode(raw64 + "=" * (-len(raw64) % 4)))
        if int(payload["exp"]) < time.time():
            return None
        return int(payload["uid"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
