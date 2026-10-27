"""Run `python lightweight_app.py` for a browser app with no third-party packages."""

import argparse
import json
import math
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from download_data import download
from lite_ml import ROOT, features, predict, train


class AppHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def trusted_request(self):
        expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        # Prevent DNS rebinding and cross-origin browser access to this local demo.
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return (host in expected and (origin is None or origin in {f"http://{item}" for item in expected})
                and self.headers.get("Sec-Fetch-Site") != "cross-site")

    def send_content(self, content, content_type, status=200):
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if not self.trusted_request():
            self.send_content("Forbidden", "text/plain", 403)
            return
        if self.path == "/":
            self.send_content((ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/metrics":
            self.send_content(json.dumps(self.server.report), "application/json")
        elif self.path == "/test_predictions.csv":
            self.send_content((ROOT / "artifacts" / "lite_test_predictions.csv").read_bytes(), "text/csv; charset=utf-8")
        elif self.path == "/health":
            self.send_content("ok", "text/plain")
        else:
            self.send_content("Not found", "text/plain", 404)

    def do_POST(self):
        if not self.trusted_request():
            self.send_content("Forbidden", "text/plain", 403)
            return
        if self.path != "/api/predict":
            self.send_content("Not found", "text/plain", 404)
            return
        try:
            if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
                raise ValueError("Provide one Content-Length header.")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Use application/json.")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("Provide one student record.")
            def unique_fields(pairs):
                values = {}
                for key, value in pairs:
                    if key in values:
                        raise ValueError("Duplicate JSON fields are not allowed.")
                    values[key] = value
                return values
            record = json.loads(self.rfile.read(size), object_pairs_hook=unique_fields)
            if not isinstance(record, dict):
                raise ValueError("Provide a JSON object containing the student inputs.")
            grade = predict(self.server.model, features(record))
            if not math.isfinite(grade):
                raise ValueError("The predictor returned an invalid grade.")
            self.send_content(json.dumps({"grade": grade, "mae": self.server.report["mae"]}), "application/json")
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError) as error:
            self.send_content(json.dumps({"error": str(error)}), "application/json", 400)
        except socket.timeout:
            self.send_content(json.dumps({"error": "Request timed out."}), "application/json", 408)


def create_server(port=8501):
    data = ROOT / "data" / "student-mat.csv"
    if not data.exists():
        download(data)
    model, report = train(data)
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    server.model, server.report = model, report
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    server = create_server(args.port)
    print(f"Student Performance Prediction: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"{server.report['selected_model']} | test MAE {server.report['mae']:.3f} / 20", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
