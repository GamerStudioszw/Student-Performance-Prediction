"""Run `python lightweight_app.py` for a browser app with no third-party packages."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from download_data import download
from lite_ml import ROOT, features, predict, train


class AppHandler(BaseHTTPRequestHandler):
    def send_content(self, content, content_type, status=200):
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
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
        if self.path != "/api/predict":
            self.send_content("Not found", "text/plain", 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("Provide one student record.")
            record = json.loads(self.rfile.read(size))
            if not isinstance(record, dict):
                raise ValueError("Provide a JSON object containing the student inputs.")
            grade = predict(self.server.model, features(record))
            self.send_content(json.dumps({"grade": grade, "mae": self.server.report["mae"]}), "application/json")
        except (ValueError, TypeError, UnicodeDecodeError) as error:
            self.send_content(json.dumps({"error": str(error)}), "application/json", 400)


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
