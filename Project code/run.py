"""Root launcher — avoids ModuleNotFoundError when running as a script.

Use:  python run.py
"""
from app.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
