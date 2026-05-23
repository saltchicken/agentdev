import asyncio
import os

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types


# 1. Define your action tools
def turn_on_lights(room: str) -> str:
    """
    Turns on lights in the specified room.
    
    Args:
        room: The name of the room.
        
    Returns:
        A confirmation string.
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
    Simulates sending text to a local TTS server (e.g., Kokoro or Piper).
    """
    print(f"[TTS Audio Playing]: {sentence}")


# 4. The Buffer Logic
async def process_stream_for_tts(event_stream, tts_callback):
    """
    Buffers the ADK async event stream and triggers the TTS callback 
    whenever a full sentence is formed.
    """
    buffer = ""
    punctuation_marks = {'.', '?', '!'}

    # The Runner yields ADK 'Event' objects
    async for event in event_stream:
        # We only care about partial events (streaming chunks) that contain text
        if event.partial and event.content and event.content.parts:
            # Extract text safely
            part = event.content.parts[0]
            text_chunk = part.text if hasattr(part, 'text') else str(part)
            buffer += text_chunk

            # Strip trailing whitespace to check the last actual character
            stripped_buffer = buffer.strip()
            if stripped_buffer and stripped_buffer[-1] in punctuation_marks:
                tts_callback(stripped_buffer)
                buffer = ""

    # Flush any remaining text that didn't end in punctuation
    if buffer.strip():
        tts_callback(buffer.strip())


# 5. Main Execution Loop
async def main():
    print(
        "Agent is ready. Type your request (simulating STT). Type 'exit' to quit."
    )

    app_name = "local_voice_app"
    user_id = "local_user"

    # 1. Initialize the Session Service
    session_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///agent_sessions.db")

    # 2. Explicitly create the session BEFORE running the agent
    # This registers the session in memory and generates a valid UUID for it.
    session = await session_service.create_session(app_name=app_name,
                                                   user_id=user_id)

    # 3. Use the base Runner and pass our configured session_service
    runner = Runner(agent=voice_brain,
                    app_name=app_name,
                    session_service=session_service)

    # Force the runner into Server-Sent Events (SSE) streaming mode
    run_config = RunConfig(streaming_mode=StreamingMode.SSE)

    while True:
        try:
            user_input = input("\n[You (STT)]: ")
            if user_input.lower() in ['exit', 'quit']:
                break

            # Format the input as a strongly typed Content object
            user_message = types.Content(role="user",
                                         parts=[types.Part(text=user_input)])

            # 4. Start the streaming execution using the dynamic session.id
            event_stream = runner.run_async(
                user_id=user_id,
                session_id=session.id,  # Using the valid, registered ID
                new_message=user_message,
                run_config=run_config)

            # Pipe the async event stream through the sentence buffer
            await process_stream_for_tts(event_stream, send_to_local_tts_server)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
