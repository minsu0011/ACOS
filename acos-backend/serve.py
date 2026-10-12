import argparse
import os

from waitress import serve

from app import app


def main():
    parser = argparse.ArgumentParser(description="Serve the local ACOS API.")
    parser.add_argument("--port", type=int, default=int(os.getenv("ACOS_BACKEND_PORT", "5001")))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    serve(app, host="127.0.0.1", port=args.port, threads=4,
          # Waitress rejects sizes >= this setting; Flask's limit is inclusive.
          max_request_body_size=app.config["MAX_CONTENT_LENGTH"] + 1)


if __name__ == "__main__":
    main()
