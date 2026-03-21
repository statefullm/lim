#!/usr/bin/env python3
import asyncio
import websockets
import os

FIFO_PATH = "/tmp/lllm.fifo"

# Keep track of connected browsers
clients = set()

async def broadcast_llm_stream():
    """Reads from the FIFO and broadcasts to all connected WebSockets."""
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH)

    print(f"Listening to {FIFO_PATH}...")

    # Open non-blocking
    fd = os.open(FIFO_PATH, os.O_RDONLY | os.O_NONBLOCK)

    with os.fdopen(fd, 'rb', buffering=0) as pipe:
        while True:
            try:
                chunk = pipe.read(4096)
                if chunk:
                    text = chunk.decode('utf-8', errors='ignore')
                    if clients:
                        # Send to all connected browser tabs
                        await asyncio.gather(*(client.send(text) for client in clients))
                else:
                    await asyncio.sleep(0.01)
            except BlockingIOError:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Pipe error: {e}")
                await asyncio.sleep(1)

async def handler(websocket):
    """Handles new browser connections."""
    clients.add(websocket)
    print("Browser connected!")
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)
        print("Browser disconnected.")

async def main():
    # Start the websocket server on port 8765
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    print("WebSocket Server running on ws://localhost:8765")

    # Run the pipe reader concurrently
    await broadcast_llm_stream()

if __name__ == "__main__":
    asyncio.run(main())
