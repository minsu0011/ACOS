import importlib.util
from pathlib import Path
import socket
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acos_launcher", ROOT / "run.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_environment_respects_existing_values_and_ignores_unknown_keys(tmp_path):
    config = tmp_path / ".env"
    config.write_text("ACOS_BACKEND_PORT=6000\nACOS_ANALYSIS_MODE=heuristic\nPATH=unsafe\n", encoding="utf-8")
    result = launcher.load_environment(config, {"ACOS_BACKEND_PORT": "7000", "PATH": "safe"})
    assert result["ACOS_BACKEND_PORT"] == "7000"
    assert result["ACOS_ANALYSIS_MODE"] == "heuristic"
    assert result["PATH"] == "safe"


def test_port_validation_does_not_take_over_an_existing_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(OSError):
            launcher.ensure_port_available(listener.getsockname()[1])
    with pytest.raises(ValueError):
        launcher.ensure_port_available(0)


def test_same_ports_fail_before_starting_children(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No child should start.")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    assert launcher.main(["--backend-port", "9000", "--frontend-port", "9000"]) == 1


class Process:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class Clock:
    def __init__(self):
        self.value = 0

    def monotonic(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


def fake_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(launcher.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(launcher.time, "sleep", clock.sleep)
    return clock


def test_readiness_retries_http_failure_until_service_is_available(monkeypatch):
    clock = fake_clock(monkeypatch)
    replies = iter([launcher.urllib.error.URLError("starting"), 503, 200])
    requested = []
    def open_url(url, timeout):
        requested.append((url, timeout))
        response = next(replies)
        if isinstance(response, Exception):
            raise response
        return nullcontext(SimpleNamespace(status=response))
    monkeypatch.setattr(launcher.urllib.request, "urlopen", open_url)
    launcher.wait_for_service(Process(), "http://127.0.0.1:8501/_stcore/health", "frontend", 5)
    assert len(requested) == 3
    assert clock.value == pytest.approx(0.4)
    assert all(timeout <= 1 for _, timeout in requested)


def test_readiness_timeout_uses_configured_duration_and_service_name(monkeypatch):
    clock = fake_clock(monkeypatch)
    def unavailable(*args, **kwargs):
        raise launcher.urllib.error.URLError("starting")
    monkeypatch.setattr(launcher.urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="backend did not become ready within 3 seconds"):
        launcher.wait_for_service(Process(), "http://127.0.0.1:5001/api/health", "backend", 3)
    assert clock.value == pytest.approx(3)


def test_readiness_stops_if_primary_or_dependency_exits(monkeypatch):
    fake_clock(monkeypatch)
    process = Process()
    process.returncode = 1
    with pytest.raises(RuntimeError, match="frontend exited before becoming ready"):
        launcher.wait_for_service(process, "http://127.0.0.1:8501/_stcore/health", "frontend")
    backend = Process()
    backend.returncode = 1
    with pytest.raises(RuntimeError, match="required service exited"):
        launcher.wait_for_service(Process(), "http://127.0.0.1:8501/_stcore/health", "frontend", dependencies=(backend,))


def test_browser_opens_only_after_both_services_are_ready(monkeypatch, capsys):
    events = []
    processes = [Process(), Process()]
    starts = iter(processes)
    monkeypatch.setattr(launcher, "load_environment", lambda *args: {"ACOS_STARTUP_TIMEOUT": "47"})
    monkeypatch.setattr(launcher, "ensure_port_available", lambda port: None)
    def start(command, **kwargs):
        events.append(("start", kwargs["env"]["ACOS_STARTUP_TIMEOUT"]))
        assert kwargs["cwd"] == ROOT
        return next(starts)
    def ready(process, url, service, timeout, dependencies=()):
        assert "UI:" not in capsys.readouterr().out
        events.append(("ready", service, timeout))
        if service == "frontend":
            assert url.endswith("/_stcore/health")
            assert dependencies == (processes[0],)
    def open_browser(url):
        events.append(("browser", url))
    def interrupt(_duration):
        raise KeyboardInterrupt
    monkeypatch.setattr(launcher.subprocess, "Popen", start)
    monkeypatch.setattr(launcher, "wait_for_service", ready)
    monkeypatch.setattr(launcher.webbrowser, "open", open_browser)
    monkeypatch.setattr(launcher.time, "sleep", interrupt)
    assert launcher.main(["--backend-port", "6001", "--frontend-port", "8502", "--open-browser"]) == 0
    assert events == [("start", "47"), ("ready", "backend", 47),
                      ("start", "47"), ("ready", "frontend", 47),
                      ("browser", "http://127.0.0.1:8502")]
    assert all(process.terminated for process in processes)


@pytest.mark.parametrize("timeout", [0, -1, 601])
def test_invalid_startup_timeout_fails_before_starting_children(monkeypatch, timeout):
    def forbidden(*args, **kwargs):
        raise AssertionError("No child should start.")
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    assert launcher.main(["--startup-timeout", str(timeout)]) == 1


def test_startup_failure_stops_started_children_without_opening_browser(monkeypatch):
    backend = Process()
    monkeypatch.setattr(launcher, "load_environment", lambda *args: {})
    monkeypatch.setattr(launcher, "ensure_port_available", lambda port: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: backend)
    def fail(*args, **kwargs):
        raise RuntimeError("The backend did not become ready within 120 seconds.")
    def forbidden_browser(*args):
        raise AssertionError("No browser should open before readiness.")
    monkeypatch.setattr(launcher, "wait_for_service", fail)
    monkeypatch.setattr(launcher.webbrowser, "open", forbidden_browser)
    assert launcher.main(["--open-browser"]) == 1
    assert backend.terminated


def test_stop_process_kills_after_grace_period(monkeypatch):
    process = Process()
    waits = []
    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("child", timeout)
        return process.returncode
    process.wait = wait
    launcher.stop_process(process)
    assert process.terminated and process.killed
    assert waits == [5, 5]


def test_waitress_cap_preserves_flask_inclusive_body_limit(monkeypatch):
    import waitress
    calls = []
    fake_app = SimpleNamespace(config={"MAX_CONTENT_LENGTH": 2_000_000})
    monkeypatch.setitem(sys.modules, "app", SimpleNamespace(app=fake_app))
    monkeypatch.setattr(waitress, "serve", lambda app, **kwargs: calls.append((app, kwargs)))
    spec = importlib.util.spec_from_file_location("acos_serve_test", ROOT / "acos-backend" / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["serve.py", "--port", "6001"])
    module.main()
    assert calls == [(fake_app, {"host": "127.0.0.1", "port": 6001, "threads": 4,
                               "max_request_body_size": 2_000_001})]
