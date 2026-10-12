"""Local Flask entry point, independent of historical model experiments."""
from acos_app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
