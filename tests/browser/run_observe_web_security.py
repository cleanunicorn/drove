"""Run real-browser security assertions against an isolated packaged observe app."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", action="store_true", help="Allow the old CDN for reproduction"
    )
    args = parser.parse_args()
    session = f"drove-observe-security-{os.getpid()}"
    with tempfile.TemporaryDirectory(prefix="drove-observe-security-") as temporary:
        temp = Path(temporary)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(temp / "dist")],
            cwd=ROOT,
            check=True,
            timeout=120,
        )
        wheel = next((temp / "dist").glob("*.whl"))
        package = temp / "package"
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(package)
        source_static = ROOT / "src/drove/observe_web_static"
        for source in source_static.rglob("*"):
            if source.is_file():
                installed = package / "drove/observe_web_static" / source.relative_to(source_static)
                assert installed.read_bytes() == source.read_bytes(), source
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = os.environ | {"PYTHONPATH": str(package), "PYTHONDONTWRITEBYTECODE": "1"}
        with (temp / "server.log").open("w+") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests/browser/observe_web_fixture.py"),
                    str(temp / "records"),
                    str(port),
                    *(["--baseline"] if args.baseline else []),
                ],
                cwd=temp,
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                deadline = time.monotonic() + 15
                while True:
                    try:
                        with urllib.request.urlopen(url, timeout=1):
                            break
                    except OSError:
                        if server.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0)
                            raise RuntimeError(log.read() or "Observe test server did not start")
                        time.sleep(0.05)
                code = (
                    f"const BASE_URL = {json.dumps(url)};\n"
                    f"const BASELINE = {json.dumps(args.baseline)};\n"
                    + (ROOT / "tests/browser/observe_web_security.js").read_text()
                )
                result = subprocess.run(
                    ["betterwright", "run", "--session", session, "-"],
                    input=code,
                    text=True,
                    capture_output=True,
                    cwd=ROOT,
                    timeout=90,
                )
                print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                report = json.loads(result.stdout)
                return 0 if result.returncode == 0 and report.get("ok") else 1
            finally:
                subprocess.run(["betterwright", "close", session], capture_output=True, timeout=20)
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    raise SystemExit(main())
