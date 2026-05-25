"""
Domain-specific expert subagents.
"""

from pydantic import BaseModel, Field
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm

# DEFAULT_MODEL = "ollama_chat/devstral-small-2"
DEFAULT_MODEL = "ollama_chat/gemma4:e4b"
llm_client = LiteLlm(model=DEFAULT_MODEL)


class ExpertResponse(BaseModel):
    internal_reasoning: str = Field(
        description="Your step-by-step logic, calculations, or technical markdown."
    )
    final_answer: str = Field(
        description="A concise, 1-2 sentence conversational summary optimized for text-to-speech."
    )


tech_expert = Agent(
    model=llm_client,
    name="tech_expert",
    output_schema=ExpertResponse,
    instruction=(
        "You are a senior software engineer. Answer the technical question.\n"
        "Put all your code blocks and deep technical details into 'internal_reasoning'.\n"
        "Put a conversational, 1-2 sentence spoken summary into 'final_answer'.\n"
        "Here is the conversation history:\n{history?}"
    ),
)

math_expert = Agent(
    model=llm_client,
    name="math_expert",
    output_schema=ExpertResponse,
    instruction=(
        "You are a mathematician. Solve the user's math question.\n"
        "Put all your complex equations and step-by-step work into 'internal_reasoning'.\n"
        "Put a highly concise, spoken summary into 'final_answer'.\n"
        "Here is the conversation history:\n{history?}"
    ),
)

general_expert = Agent(
    model=llm_client,
    name="general_expert",
    output_schema=ExpertResponse,
    instruction=(
        "You are a helpful AI assistant. Answer the general question.\n"
        "Put any detailed explanations or lists into 'internal_reasoning'.\n"
        "Put a conversational, 1-2 sentence spoken summary into 'final_answer'.\n"
        "Here is the conversation history:\n{history?}"
    ),
)
