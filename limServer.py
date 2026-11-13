#!/usr/bin/env python3
import asyncio
import os
import re
import secrets
import socket
from pathlib import Path
from aiohttp import web

# The FIFO and inter-process marker files live in LIM_CONFIG_DIR
# (user-owned, under the user's home) so other local users cannot
# pre-create them (issue 8).
LIM_CONFIG_DIR = os.environ.get('LIM_CONFIG_DIR',
                                os.path.join(os.environ.get('HOME', '/tmp'), '.config', 'lim'))
os.makedirs(LIM_CONFIG_DIR, exist_ok=True)

FIFO_PATH = os.path.join(LIM_CONFIG_DIR, "lim.fifo")

# Configurable port via environment variable (default: 8765)
LIM_PORT = int(os.environ.get('LIM_PORT', '8765'))

# Random token: delivered in the page URL's fragment (#token=...), which the
# browser never sends to the server, and required for the /ws upgrade.  It is
# written to LIM_CONFIG_DIR/lim.viewer_url (mode 0640, readable only by lim
# and the sharing user); lim reads it to print the URL and the VS Code
# extension reads it to open the page.  The served page and JS stay token-free,
# so other local users cannot obtain it and cannot connect to /ws (issue 8).
WS_TOKEN = secrets.token_hex(16)
VIEWER_URL_PATH = os.path.join(LIM_CONFIG_DIR, "lim.viewer_url")

def allowed_ws_origins():
    """Origins the viewer page itself can have when it connects to /ws:
    the loopback names and this machine's own name, all on LIM_PORT, plus
    the origin of LIM_VIEWER_URL when set (a custom tunnel local port
    changes the page's origin; lim and the server read the same env var).
    viewer.html is served by this server and connects back to the same
    host (window.location.host), so no other origin is legitimate.
    WebSockets are not covered by CORS, so without this check JavaScript
    on any website the user has open could subscribe to the stream
    through an SSH tunnel."""
    hosts = {'localhost', '127.0.0.1'}
    try:
        name = socket.gethostname()
        hosts.add(name)
        hosts.add(name.split('.')[0])
    except OSError:
        pass
    origins = {'http://' + h + ':' + str(LIM_PORT) for h in hosts}
    m = re.match(r'https?://[^/]+', os.environ.get('LIM_VIEWER_URL', ''))
    if m:
        origins.add(m.group(0))
    return origins

# Keep track of connected browsers
clients = set()

# Buffer the last SEG_SPEED payload for late-connecting browsers.
_last_speed = [""]

# Marker file for browser connection signaling.

BROWSER_READY_PATH = os.path.join(LIM_CONFIG_DIR, "lim.browser_ready")

def _update_browser_ready():
    """Write/remove the browser-ready marker file."""
    try:
        if clients:
            with open(BROWSER_READY_PATH, 'w') as f:
                f.write(str(len(clients)))
        else:
            os.unlink(BROWSER_READY_PATH)
    except FileNotFoundError:
        pass  # File already gone
    except Exception:
        pass  # Best effort; don't crash the server

async def broadcast_llm_stream():
    """Reads from the FIFO and broadcasts to all connected WebSockets.

    Uses loop.add_reader() (epoll on Linux) so we wake up only when
    data is actually available -- no polling, no wasted cycles.
    """
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH, 0o660)

    # Open non-blocking; add_reader will notify us when readable.
    # O_RDWR: the server itself always holds a write end, so the FIFO can never
    # reach EOF when lim exits -- an at-EOF FIFO is permanently readable and
    # would spin the reader.  Between lim sessions the fd is simply not
    # readable; the next lim's first write wakes it with real data.
    fd = os.open(FIFO_PATH, os.O_RDWR | os.O_NONBLOCK)
    # Best effort (EPERM once aishare has re-owned the node to the sharing
    # user): fix nodes created with a looser mode.  0660 = owner + group,
    # no others.
    try:
        os.fchmod(fd, 0o660)
    except PermissionError:
        pass
    loop = asyncio.get_running_loop()

    def _on_fifo_readable():
        """Called by the event loop whenever the FIFO has data."""
        try:
            # Drain everything currently buffered in the pipe.
            buf = b""
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buf += chunk
        except BlockingIOError:
            pass  # No more data right now

        if buf:
            text = buf.decode('utf-8', errors='ignore')
            # Always buffer the latest SEG_SPEED (\x05) payload so reconnecting
            # browsers receive the current context position.
            i = text.rfind('\x05')
            if i >= 0:
                e = len(text)
                for s in '\x02\x03\x04\x05\x06\x07':
                    p = text.find(s, i + 1)
                    if 0 <= p < e:
                        e = p
                _last_speed[0] = text[i:e]

            if clients:
                for client in list(clients):
                    try:
                        asyncio.ensure_future(client.send_str(text))
                    except Exception as e:
                        print(f"[WARNING] Send failed for a client: {e}")

        # Re-arm -- data wakes the reader again as long as any writer (lim, or
        # the server's own O_RDWR fd) holds the FIFO.
        loop.add_reader(fd, _on_fifo_readable)

    loop.add_reader(fd, _on_fifo_readable)

    # Block forever; the server lives as long as the FIFO is open.
    await asyncio.Event().wait()

async def websocket_handler(request):
    """Handles WebSocket connections for LLM streaming.

    Two layers of authentication:
    1. Origin check: a present Origin must be the viewer's own; a website
       the user has open presents a foreign one and is rejected (the browser
       sets Origin automatically, so a page cannot bypass this).  A missing
       Origin is accepted: it only happens off a normal web page (embedded
       browsers, raw clients), and the token below is the barrier there.
    2. Token check: the ?token= query parameter must match the server's
       random token.  The token reaches the browser only in the page URL's
       fragment (#token=...), which the browser never sends to the server,
       and in lim.viewer_url (mode 0640) for lim to print and the VS Code
       extension to open -- so other local users cannot obtain it and
       cannot connect (issue 8).
    """
    origin = request.headers.get('Origin')
    if origin is not None and origin not in allowed_ws_origins():
        print(f"[WARNING] Rejected WebSocket connection from disallowed origin: {origin}")
        return web.Response(status=403, text="Origin not allowed")

    # Require the token: only the viewer page (served with ?token=) knows it.
    token = request.query.get('token', '')
    if not token or token != WS_TOKEN:
        print(f"[WARNING] Rejected WebSocket connection with missing/invalid token from {request.remote}")
        return web.Response(status=403, text="Invalid or missing token")

    ws = web.WebSocketResponse(heartbeat=10)  # Enable ping/pong every 10 seconds to prevent timeout
    await ws.prepare(request)
    clients.add(ws)
    _update_browser_ready()

    # Replay last speed message so late-connecting browsers see context info.
    if _last_speed[0]:
        try:
            await ws.send_str(_last_speed[0])
        except Exception:
            pass

    try:
        async for msg in ws:
            pass  # Browser doesn't send messages, just receives
    except Exception as e:
        print(f"[ERROR] WebSocket handler exception: {e}")
        raise
    finally:
        clients.discard(ws)
        _update_browser_ready()

    return ws

async def status_handler(request):
    """HTTP endpoint to check if browser is connected (for main.cc to query)."""
    connected = len(clients) > 0
    return web.json_response({"connected": connected, "clients": len(clients)})

async def serve_viewer(request):
    """Serve viewer.html from the script directory.

    The LIM_INLINE_LATEX flag now lives in viewer.js (see serve_viewer_js),
    but the auto-render <script> line still has to be removed here when LaTeX
    is disabled so it doesn't mangle $HOME etc.
    """
    script_path = Path(__file__).resolve()
    viewer_html_path = script_path.parent / 'viewer.html'

    if not viewer_html_path.exists():
        return web.Response(status=404, text="viewer.html not found")

    with open(viewer_html_path, 'r') as f:
        content = f.read()

    # When LaTeX is disabled, remove auto-render so it doesn't mangle $HOME etc.
    if os.environ.get('LIM_INLINE_LATEX', '1') == '0':
        content = content.replace(
            '    <script defer src="libs/auto-render.min.js"></script>\n',
            '',
        )

    return web.Response(
        text=content,
        content_type='text/html',
        headers={'Cache-Control': 'no-store'},
    )

async def serve_viewer_js(request):
    """Serve viewer.js, injecting the LIM_INLINE_LATEX setting.

    The app script was moved out of viewer.html so the page's script-src can
    be 'self' (no 'unsafe-inline').  The LIM_INLINE_LATEX_ENABLED default in
    the file is overridden here, exactly as serve_viewer used to do for HTML.
    The /ws token is deliberately not injected here: it rides in the page
    URL's fragment, so the served JS stays token-free (issue 8).
    """
    script_path = Path(__file__).resolve()
    viewer_js_path = script_path.parent / 'viewer.js'

    if not viewer_js_path.exists():
        return web.Response(status=404, text="viewer.js not found")

    with open(viewer_js_path, 'r') as f:
        content = f.read()

    # Override the default LIM_INLINE_LATEX_ENABLED value based on the env var.
    latex_val = os.environ.get('LIM_INLINE_LATEX', '1')
    content = content.replace(
        "var LIM_INLINE_LATEX_ENABLED = 1;",
        "var LIM_INLINE_LATEX_ENABLED = " + latex_val + ";",
    )

    return web.Response(
        text=content,
        content_type='application/javascript',
        headers={'Cache-Control': 'no-store'},
    )

async def main():
    # Single aiohttp server handling both HTTP and WebSocket
    app = web.Application()
    script_path = Path(__file__).resolve()
    app.router.add_get('/viewer.html', serve_viewer)
    app.router.add_get('/viewer.js', serve_viewer_js)
    app.router.add_static('/libs/', str(script_path.parent / 'libs'))
    app.router.add_get('/status', status_handler)
    app.router.add_get('/ws', websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    # Loopback only: the stream carries everything the viewer shows (file
    # contents, shell output, secrets in them).  Remote access goes through
    # an explicit SSH tunnel (ssh -L <port>:localhost:<port> ...), which
    # arrives here as a localhost connection; LAN hosts cannot reach it.
    site = web.TCPSite(runner, '127.0.0.1', LIM_PORT)
    await site.start()

    try:
        # The tokenized viewer URL, written before the ready marker so a
        # reader that waits for the marker (lim) always sees the finished
        # file.  Mode 0640 via chmod (umask-independent): owner + group read,
        # no others -- lim (the ai user) and the sharing user can read it,
        # other local users cannot (issue 8).
        base = os.environ.get('LIM_VIEWER_URL', '')
        if not base:
            base = 'http://localhost:%d/viewer.html' % LIM_PORT
        url_tmp = VIEWER_URL_PATH + '.tmp'
        # Create the temp file at 0640 from the start so it is never
        # world-readable, even for the instant before the chmod below
        # (umask here is 0002, so O_CREAT already yields 0640; the explicit
        # chmod keeps the final mode umask-independent either way).
        fd = os.open(url_tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
        with os.fdopen(fd, 'w') as f:
            f.write(base + '#token=' + WS_TOKEN + '\n')
        os.chmod(url_tmp, 0o640)
        os.replace(url_tmp, VIEWER_URL_PATH)
        with open(os.path.join(LIM_CONFIG_DIR, "lim.server_ready"), 'w') as f:
            f.write(str(LIM_PORT))
        with open(os.path.join(LIM_CONFIG_DIR, "lim.server.pid"), 'w') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    # Run the pipe reader concurrently
    await broadcast_llm_stream()

if __name__ == "__main__":
    asyncio.run(main())
