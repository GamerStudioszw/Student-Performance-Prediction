"""Run the release checks; any failed check produces a nonzero exit status."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    destination = ROOT / ".local" / "verification" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True)
    checks = [
        ("dependencies", ["-m", "pip", "check"], 120),
        ("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"], 900),
        ("advisories", ["-m", "pip_audit", "--strict", "--progress-spinner", "off",
                        "--format", "json", "--output", str(destination / "dependency-audit.json")], 300),
    ]
    results = []
    for name, arguments, timeout in checks:
        print(f"Running {name}...", flush=True)
        log = destination / f"{name}.log"
        try:
            with log.open("w", encoding="utf-8") as output:
                result = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=timeout, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        results.append({"check": name, "exit_code": code, "log": str(log.relative_to(ROOT))})
        print(f"{name}: {'PASS' if code == 0 else 'FAIL'} ({log})", flush=True)
        if code:
            print(log.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
    (destination / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if any(result["exit_code"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
