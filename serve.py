"""Open the results dashboard locally:  python serve.py   (then it opens http://localhost:8000/web/)"""
import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
root = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(root), **k)


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as srv:
        url = f"http://localhost:{PORT}/web/index.html"
        print(f"Serving {root} -> {url}   (Ctrl+C to stop)")
        webbrowser.open(url)
        srv.serve_forever()
