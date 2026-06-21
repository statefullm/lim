#!/usr/bin/env python3
import asyncio
import os
import socket
from pathlib import Path
from aiohttp import web

FIFO_PATH = "/tmp/lllm.fifo"

# Configurable port via environment variable (default: 8765)
LLLM_PORT = int(os.environ.get('LLLM_PORT', '8765'))

# Keep track of connected browsers
clients = set()

# Marker file for browser connection signaling.
BROWSER_READY_PATH = "/tmp/lllm.browser_ready"

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
    """Reads from the FIFO and broadcasts to all connected WebSockets."""
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH)

#    print(f"Listening to {FIFO_PATH}...")

    # Open non-blocking
    fd = os.open(FIFO_PATH, os.O_RDONLY | os.O_NONBLOCK)

    with os.fdopen(fd, 'rb', buffering=0) as pipe:
        while True:
            try:
                chunk = pipe.read(65536)
                if chunk:
                    text = chunk.decode('utf-8', errors='ignore')
                    if clients:
                        # Send to all connected browser tabs
                        for client in list(clients):
                            try:
                                await client.send_str(text)
                            except Exception as e:
                                print(f"[WARNING] Send failed for a client: {e}")
                                import traceback
                                traceback.print_exc()
                                # Client will be removed when websocket_handler detects the actual disconnection
                else:
                    await asyncio.sleep(0.01)
            except BlockingIOError:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Pipe error: {e}")
                await asyncio.sleep(1)

async def websocket_handler(request):
    """Handles WebSocket connections for LLM streaming."""
    ws = web.WebSocketResponse(heartbeat=10)  # Enable ping/pong every 10 seconds to prevent timeout
    await ws.prepare(request)
    clients.add(ws)
    _update_browser_ready()

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
    """Serve viewer.html from the script directory."""
    script_path = Path(__file__).resolve()
    viewer_html_path = script_path.parent / 'viewer.html'

    if not viewer_html_path.exists():
        return web.Response(status=404, text="viewer.html not found")

    with open(viewer_html_path, 'r') as f:
        content = f.read()
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
    site = web.TCPSite(runner, '0.0.0.0', LLLM_PORT)
    await site.start()

    try:
        with open("/tmp/lllm.server_ready", 'w') as f:
            f.write(str(LLLM_PORT))
        with open("/tmp/lllm.server.pid", 'w') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    # Run the pipe reader concurrently
    await broadcast_llm_stream()

if __name__ == "__main__":
    asyncio.run(main())
