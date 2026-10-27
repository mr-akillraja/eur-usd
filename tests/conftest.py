"""Keep every test's journal, execution state, and learned models disposable."""
import os
from pathlib import Path

import pytest

from config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_persistence(monkeypatch, tmp_path):
    """Protect account state even when a test omits a more specific fixture."""
    directories = {
        "LOG_DIR": tmp_path / "logs",
        "LOG_ARCHIVE_DIR": tmp_path / "logs" / "archive",
        "JOURNAL_ARCHIVE_DIR": tmp_path / "logs" / "archive",
        "MODEL_DIR": tmp_path / "models",
    }
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(settings, name, str(path))
        monkeypatch.setenv(name, str(path))

    # Older helpers use relative logs/models paths rather than settings.
    monkeypatch.chdir(tmp_path)
    pythonpath = [str(PROJECT_ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(pythonpath))

    # These paths are cached when browser.py is imported during collection.
    from bot import browser

    screenshots = directories["LOG_DIR"] / "screenshots"
    screenshots.mkdir(exist_ok=True)
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    monkeypatch.setattr(browser, "SCREENSHOT_DIR", str(screenshots))
    monkeypatch.setattr(browser, "SESSION_FILE", str(data / "session.json"))
