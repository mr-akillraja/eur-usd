"""One orchestrator per OS user, across project copies and asset aliases."""
from contextlib import AbstractContextManager
import fcntl
import os
from pathlib import Path
import tempfile

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "config" / ".env", override=False)


def _allow_multiple_sessions() -> bool:
    for key in ("ALLOW_MULTIPLE_SESSIONS", "ALLOW_MULTIPLE_SESSION"):
        value = os.getenv(key, "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
    return False


class ProcessLock(AbstractContextManager):
    def __init__(self, path=None):
        # Global across copies: the browser session must have only one owner,
        # unless local testing explicitly opts into multiple concurrent sessions.
        self.path = Path(path or (Path(tempfile.gettempdir()) /
                                 f"tg-algo-orchestrator-{os.getuid()}.lock"))
        self._file = None

    def __enter__(self):
        if _allow_multiple_sessions():
            return self

        self._file = self.path.open("a+")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file.close()
            self._file = None
            raise RuntimeError("Another unified orchestrator owns the browser session")
        self._file.seek(0)
        self._file.truncate()
        self._file.write(str(os.getpid()))
        self._file.flush()
        return self

    def __exit__(self, *_):
        if self._file:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None
