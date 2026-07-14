"""
CineScene application launcher.

Run:
    python app.py

Then open:
    http://127.0.0.1:8000
"""

from __future__ import annotations


def main():
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is not installed. Install requirements first:\n"
            "python -m pip install -r requirements.txt"
        ) from exc

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
