import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

app = FastAPI()

# 1. Define tools (These now execute on the server)
def turn_on_lights(room: str) -> str:
    print(f"\n[Server Tool Execution] Turning on {room} lights...")
    return f"The {room} lights are now on."

# 2. Initialize the Agent
voice_brain = Agent(
    model=LiteLlm(model="ollama_chat/devstral-small-2"),
    name="local_voice_brain",
    instruction="You are a helpful home assistant. Keep responses short.",
    tools=[turn_on_lights]
)

# 3. Setup Runner and Database
app_name = "local_voice_app"
session_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///agent_sessions.db")
runner = Runner(agent=voice_brain, app_name=app_name, session_service=session_service)

# Request Schema
class ChatRequest(BaseModel):
    user_id: str
    text: str
    session_id: str | None = None

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    user_message = types.Content(role="user", parts=[types.Part(text=request.text)])
    run_config = RunConfig(streaming_mode=StreamingMode.SSE)

    # Resolve or create the session
    session_id = request.session_id
    if not session_id:
        session = await session_service.create_session(
            app_name=app_name, 
            user_id=request.user_id
        )
        session_id = session.id

    async def event_generator():
        # Send the session ID to the client first so it can maintain history
        yield f"event: session_id\ndata: {session_id}\n\n"

        event_stream = runner.run_async(
            user_id=request.user_id,
            session_id=session_id,
            new_message=user_message,
            run_config=run_config
        )

        async for event in event_stream:
            if event.partial and event.content and event.content.parts:
                part = event.content.parts[0]
                text_chunk = part.text if hasattr(part, 'text') else str(part)
                
                # Yield text chunks as JSON to safely handle newlines in SSE
                payload = json.dumps({"text": text_chunk})
                yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
