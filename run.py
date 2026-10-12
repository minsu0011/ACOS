import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


ROOT = Path(__file__).resolve().parent
ENV_KEYS = {
    "ACOS_ANALYSIS_MODE", "ACOS_SEED_DEMO", "ACOS_DATABASE_PATH",
    "ACOS_BACKEND_PORT", "ACOS_FRONTEND_PORT", "ACOS_API_URL",
    "ACOS_PREDICTOR_FACTORY", "ACOS_ALLOWED_ORIGINS", "ACOS_STARTUP_TIMEOUT",
}


def load_environment(path, environment):
    result = dict(environment)
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in ENV_KEYS:
                result.setdefault(key, value.strip().strip("\"'"))
    return result


def ensure_port_available(port):
    if not 1 <= port <= 65535:
        raise ValueError("Ports must be between 1 and 65535.")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_service(process, url, service, timeout=120, dependencies=()):
    """Wait for HTTP readiness while ensuring every required child is alive."""
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"The {service} exited before becoming ready.")
        if any(dependency.poll() is not None for dependency in dependencies):
            raise RuntimeError(f"A required service exited while the {service} was starting.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"The {service} did not become ready within {timeout} seconds.")
        try:
            with urllib.request.urlopen(url, timeout=min(1, remaining)) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))


def main(argv=None):
    environment = load_environment(ROOT / ".env", os.environ)
    parser = argparse.ArgumentParser(description="Start the ACOS backend and Streamlit frontend.")
    parser.add_argument("--backend-port", type=int, default=environment.get("ACOS_BACKEND_PORT", "5001"))
    parser.add_argument("--frontend-port", type=int, default=environment.get("ACOS_FRONTEND_PORT", "8501"))
    parser.add_argument("--startup-timeout", type=int,
                        default=environment.get("ACOS_STARTUP_TIMEOUT", "120"),
                        help="Readiness deadline per service in seconds, including model loading (1-600; default 120).")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    children = []
    try:
        if not 1 <= args.startup_timeout <= 600:
            raise ValueError("Startup timeout must be between 1 and 600 seconds.")
        if args.backend_port == args.frontend_port:
            raise ValueError("Backend and frontend ports must differ.")
        ensure_port_available(args.backend_port)
        ensure_port_available(args.frontend_port)
        environment["ACOS_API_URL"] = f"http://127.0.0.1:{args.backend_port}"
        environment["ACOS_BACKEND_PORT"] = str(args.backend_port)
        environment["ACOS_FRONTEND_PORT"] = str(args.frontend_port)
        environment["ACOS_STARTUP_TIMEOUT"] = str(args.startup_timeout)
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        backend = subprocess.Popen(
            [sys.executable, "-B", str(ROOT / "acos-backend" / "serve.py"), "--port", str(args.backend_port)],
            cwd=ROOT, env=environment,
        )
        children.append(backend)
        wait_for_service(backend, environment["ACOS_API_URL"] + "/api/health",
                         "backend", args.startup_timeout)
        frontend = subprocess.Popen(
            [sys.executable, "-B", "-m", "streamlit", "run",
             str(ROOT / "acos-frontend" / "acos_reco" / "streamlit_app.py"),
             "--server.port", str(args.frontend_port), "--server.address", "127.0.0.1",
             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
            cwd=ROOT, env=environment,
        )
        children.append(frontend)
        ui_url = f"http://127.0.0.1:{args.frontend_port}"
        wait_for_service(frontend, ui_url + "/_stcore/health", "frontend",
                         args.startup_timeout, dependencies=(backend,))
        print(f"API: {environment['ACOS_API_URL']}", flush=True)
        print(f"UI:  {ui_url}", flush=True)
        if args.open_browser:
            webbrowser.open(ui_url)
        while all(child.poll() is None for child in children):
            time.sleep(0.3)
        raise RuntimeError("A service exited. Both services have been stopped.")
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for child in reversed(children):
            stop_process(child)


if __name__ == "__main__":
    raise SystemExit(main())
