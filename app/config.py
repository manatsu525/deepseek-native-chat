from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "/opt/deepseek-native-chat/data"))
    session_days: int = int(os.getenv("SESSION_DAYS", "60"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "1200"))
    tls_cert_file: str = os.getenv("TLS_CERT_FILE", "")
    tls_key_file: str = os.getenv("TLS_KEY_FILE", "")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chat.db"

    @property
    def secret_path(self) -> Path:
        return self.data_dir / ".session_secret"


settings = Settings()
