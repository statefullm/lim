#!/usr/bin/env python3
import asyncio
import os
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

# Random token: only the viewer page served by this server knows it.
# Prevents other local users from connecting to /ws (issue 8).
WS_TOKEN = secrets.token_hex(16)

def allowed_ws_origins():
    """Origins the viewer page itself can have when it connects to /ws:
    the loopback names and this machine's own name, all on LIM_PORT.
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
    return {'http://' + h + ':' + str(LIM_PORT) for h in hosts}

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
    1. Origin check: browsers must present a matching Origin header.
       (Non-browser clients are rejected -- they cannot know the token.)
    2. Token check: the ?token= query parameter must match the server's
       random token.  Only the viewer page served by this server knows it,
       so other local users cannot connect (issue 8).
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

    # Inject the WebSocket token so the viewer can authenticate to /ws.
    content = content.replace(
        'var LIM_WS_TOKEN = "";',
        'var LIM_WS_TOKEN = "' + WS_TOKEN + '";',
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
