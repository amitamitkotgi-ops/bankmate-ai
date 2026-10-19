"""
Start the backend (FastAPI) and the frontend (Streamlit) together.

    python run_all.py

Press Ctrl+C to stop both. Backend: http://127.0.0.1:8000/docs   Frontend: http://localhost:8501
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    py = sys.executable
    api = subprocess.Popen([py, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"], cwd=ROOT)
    time.sleep(2)
    ui = subprocess.Popen([py, "-m", "streamlit", "run", "app.py", "--server.port", "8501"], cwd=ROOT)
    try:
        api.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (ui, api):
            p.terminate()


if __name__ == "__main__":
    main()
