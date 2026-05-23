import asyncio
import json
import httpx
from httpx_sse import aconnect_sse
from client import Qwen3TTSClient
import signal

tts = Qwen3TTSClient(server_url="http://10.0.0.17:8123/tts")
tts.start()

SERVER_URL = "http://localhost:8000/chat/stream"
USER_ID = "local_user"

async def process_stream_for_tts(client, request_data):
    """Connects to FastAPI, reads the SSE stream, and buffers text for TTS."""
    buffer = ""
    punctuation_marks = {'.', '?', '!'}
    session_id = request_data.get("session_id")

    async with aconnect_sse(client, "POST", SERVER_URL, json=request_data) as event_source:
        async for sse in event_source.aiter_sse():
            # Capture the session_id emitted by the server to use in the next prompt
            if sse.event == "session_id":
                session_id = sse.data
                continue
                
            if sse.data:
                data = json.loads(sse.data)
                text_chunk = data.get("text", "")
                buffer += text_chunk

                stripped_buffer = buffer.strip()
                if stripped_buffer and stripped_buffer[-1] in punctuation_marks:
                    tts.speak(stripped_buffer)
                    buffer = ""

    if buffer.strip():
        tts.speak(buffer.strip())
        
    print(session_id)
    return session_id

async def main():
    print("API Client is ready. Type your request. Type 'exit' to quit.\n")
    current_session_id = None
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                user_input = input("\n[You (STT)]: ")
            except KeyboardInterrupt:
                if hasattr(tts, 'interrupt'): tts.interrupt()
                print()
                continue

            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            if hasattr(tts, 'interrupt'):
                tts.interrupt()

            request_data = {
                "user_id": USER_ID,
                "text": user_input,
                "session_id": current_session_id
            }

            # === RESTORED SIGNAL HANDLING MAGIC ===
            main_task = asyncio.current_task()
            loop = asyncio.get_running_loop()

            def _cancel_stream():
                if main_task:
                    main_task.cancel()

            loop.add_signal_handler(signal.SIGINT, _cancel_stream)

            try:
                # Process the stream and update the session ID for the next loop
                current_session_id = await process_stream_for_tts(client, request_data)
            except asyncio.CancelledError:
                print("\n[System] 🛑 Stream interrupted!")
                if hasattr(tts, 'interrupt'): tts.interrupt()
            except Exception as e:
                print(f"\nAn error occurred connecting to server: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        tts.close()
