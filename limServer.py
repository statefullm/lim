#!/usr/bin/env python3
import asyncio
import os
import socket
from pathlib import Path
from aiohttp import web

FIFO_PATH = "/tmp/lim.fifo"

# Configurable port via environment variable (default: 8765)
LIM_PORT = int(os.environ.get('LIM_PORT', '8765'))

# Keep track of connected browsers
clients = set()

# Buffer the last SEG_SPEED payload for late-connecting browsers.
_last_speed = [""]

# Marker file for browser connection signaling.

BROWSER_READY_PATH = "/tmp/lim.browser_ready"

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
        os.mkfifo(FIFO_PATH)

    # Open non-blocking; add_reader will notify us when readable.
    fd = os.open(FIFO_PATH, os.O_RDONLY | os.O_NONBLOCK)
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

        # Re-arm -- the writer keeps its end open, so we'll be notified again.
        loop.add_reader(fd, _on_fifo_readable)

    loop.add_reader(fd, _on_fifo_readable)

    # Block forever; the server lives as long as the FIFO is open.
    await asyncio.Event().wait()

async def websocket_handler(request):
    """Handles WebSocket connections for LLM streaming."""
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
    """Serve viewer.html from the script directory, injecting LIM_INLINE_LATEX setting."""
    script_path = Path(__file__).resolve()
    viewer_html_path = script_path.parent / 'viewer.html'

    if not viewer_html_path.exists():
        return web.Response(status=404, text="viewer.html not found")

    with open(viewer_html_path, 'r') as f:
        content = f.read()

    # Override the default LIM_INLINE_LATEX_ENABLED value based on the env var.
    latex_val = os.environ.get('LIM_INLINE_LATEX', '1')
    content = content.replace(
        "var LIM_INLINE_LATEX_ENABLED = 1;",
        "var LIM_INLINE_LATEX_ENABLED = " + latex_val + ";",
    )

    # When LaTeX is disabled, also remove auto-render so it doesn't mangle $HOME etc.
    if latex_val == '0':
        content = content.replace(
            '    <script defer src="libs/auto-render.min.js"></script>\n',
            '',
        )

    return web.Response(text=content, content_type='text/html')

async def main():
    # Single aiohttp server handling both HTTP and WebSocket
    app = web.Application()
    script_path = Path(__file__).resolve()
    app.router.add_get('/viewer.html', serve_viewer)
    app.router.add_static('/libs/', str(script_path.parent / 'libs'))
    app.router.add_get('/status', status_handler)
    app.router.add_get('/ws', websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', LIM_PORT)
    await site.start()

    try:
        with open("/tmp/lim.server_ready", 'w') as f:
            f.write(str(LIM_PORT))
        with open("/tmp/lim.server.pid", 'w') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    # Run the pipe reader concurrently
    await broadcast_llm_stream()

if __name__ == "__main__":
    asyncio.run(main())
