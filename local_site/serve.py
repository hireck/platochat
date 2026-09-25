"""Serve local_site/ for chatbot development.

    python3 serve.py [port]        # default 8090; `make dev` and `make web` run it

This is `python3 -m http.server` with one addition. The chatbot pages link to
each other by the URLs they will have on PLATO-Pub (/about/chatbot.php,
/about/chatbot_about.php), so that their markup can go to the site unchanged;
here those URLs are answered from the static files standing in for them. Any
other .php URL is one of PLATO-Pub's own pages, which are not copied here, so
it is sent on to the live site -- where the link would take you there too.

Unlike `python3 -m http.server`, it listens on this machine only, as the API
does.
"""

import functools
import http.server
import sys
from pathlib import Path
from urllib.parse import urlsplit

LIVE_SITE = "https://platopub.phys.au.dk"

# URL on PLATO-Pub -> the file here that stands in for it.
PAGES = {
    "/about/chatbot.php": "/chatbot.html",
    "/about/chatbot_about.php": "/chatbot_about.html",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if not self.sent_to_live_site():
            super().do_GET()

    def do_HEAD(self):
        if not self.sent_to_live_site():
            super().do_HEAD()

    def sent_to_live_site(self):
        path = urlsplit(self.path).path
        if not path.endswith(".php") or path in PAGES:
            return False
        self.send_response(302)
        self.send_header("Location", LIVE_SITE + self.path)
        self.end_headers()
        return True

    def translate_path(self, path):
        return super().translate_path(PAGES.get(urlsplit(path).path, path))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    root = Path(__file__).resolve().parent
    handler = functools.partial(Handler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Serving {root} on http://localhost:{port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
