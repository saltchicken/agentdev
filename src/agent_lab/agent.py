import asyncio
import os
import signal

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from client import Qwen3TTSClient

tts = Qwen3TTSClient(server_url="http://10.0.0.17:8123/tts")
tts.start()


# 1. Define your action tools
def turn_on_lights(room: str) -> str:
    """
    Turns on lights in the specified room.
    """
    print(f"\n[Tool Execution] Turning on {room} lights...")
    return f"The {room} lights are now on."


# 2. Initialize the Agent (The Brain)
voice_brain = Agent(
    model=LiteLlm(model="ollama_chat/devstral-small-2"),
    name="local_voice_brain",
    instruction=
    "You are a helpful home assistant. Keep responses short and conversational, as they will be spoken out loud. Use your tools when asked.",
    tools=[turn_on_lights])


# 3. The Mouth (TTS Server Callback)
def send_to_local_tts_server(sentence: str):
    """
    Simulates sending text to a local TTS server.
    """
    tts.speak(sentence)


# 4. The Buffer Logic
async def process_stream_for_tts(event_stream, tts_callback):
    buffer = ""
    punctuation_marks = {'.', '?', '!'}

    async for event in event_stream:

        if event.partial and event.content and event.content.parts:
            part = event.content.parts[0]
            text_chunk = part.text if hasattr(part, 'text') else str(part)
            buffer += text_chunk

            stripped_buffer = buffer.strip()
            if stripped_buffer and stripped_buffer[-1] in punctuation_marks:
                tts_callback(stripped_buffer)
                buffer = ""

    if buffer.strip():
        tts_callback(buffer.strip())


# 5. Main Execution Loop
async def main():
    print(
        "Agent is ready. Type your request (simulating STT). Type 'exit' to quit.\n"
        "[Tip: Press Ctrl+C while the agent is speaking to interrupt it.]"
    )

    app_name = "local_voice_app"
    user_id = "local_user"

    session_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///agent_sessions.db")

    existing_sessions_response = await session_service.list_sessions(
        app_name=app_name, 
        user_id=user_id
    )

    if existing_sessions_response and existing_sessions_response.sessions:
        session = existing_sessions_response.sessions[-1]
        print(f"[System] Resuming previous session: {session.id}")
    else:
        session = await session_service.create_session(
            app_name=app_name, 
            user_id=user_id
        )
        print(f"[System] Started new persistent session: {session.id}")

    runner = Runner(agent=voice_brain,
                    app_name=app_name,
                    session_service=session_service)

    run_config = RunConfig(streaming_mode=StreamingMode.SSE)

    while True:
        # This catches standard Ctrl+C to exit the program while waiting for user input
        try:
            user_input = input("\n[You (STT)]: ")
        except KeyboardInterrupt:
            if hasattr(tts, 'interrupt'):
                tts.interrupt()
            print()
            continue

        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input.strip():
            continue

        if hasattr(tts, 'interrupt'):
            tts.interrupt()

        user_message = types.Content(role="user",
                                     parts=[types.Part(text=user_input)])

        event_stream = runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=user_message,
            run_config=run_config)

        # === SIGNAL HANDLING MAGIC ===
        # We temporarily hijack the SIGINT (Ctrl+C) signal so it cancels the task 
        # instead of killing the event loop.
        main_task = asyncio.current_task()
        loop = asyncio.get_running_loop()

        def _cancel_stream():
            if main_task:
                main_task.cancel()

        loop.add_signal_handler(signal.SIGINT, _cancel_stream)

        try:
            await process_stream_for_tts(event_stream, send_to_local_tts_server)
        except asyncio.CancelledError:
            print("\n[System] 🛑 Agent interrupted! Ready for your next request.")
            # Safely close the OpenTelemetry generator to prevent context leaks
            if hasattr(tts, 'interrupt'):
                tts.interrupt()
            else:
                print("didn't work")
            try:
                if hasattr(event_stream, 'aclose'):
                    await event_stream.aclose()
            except Exception:
                pass 
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            # Always restore normal Ctrl+C behavior for the input() prompt
            loop.remove_signal_handler(signal.SIGINT)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass # Handle cleanly if triggered right at startup
    finally:
        loop.close()
        tts.close()
