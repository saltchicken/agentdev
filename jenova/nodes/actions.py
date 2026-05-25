"""
Nodes that perform tasks or handle conversational fallbacks.
"""

from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm

from jenova.tools.lights import turn_off_lights
from jenova.tools.lights import turn_on_lights
from jenova.nodes.experts import ExpertResponse

# DEFAULT_MODEL = "ollama_chat/devstral-small-2"
DEFAULT_MODEL = "ollama_chat/gemma4:e4b"
llm_client = LiteLlm(model=DEFAULT_MODEL)


take_action = Agent(
    model=llm_client,
    name="take_action",
    output_schema=ExpertResponse,
    instruction=
    ("Figure out what the user wants you to do and take that action based on this input: {input}\n"
     "IMPORTANT: Once the tool has been executed and you receive the result, you MUST stop calling tools. "
     "Put any internal tool-calling logic or notes into 'internal_reasoning'.\n"
     "Put a natural, conversational message confirming the action into 'final_answer'."
    ),
    tools=[turn_on_lights, turn_off_lights])


handle_other = Agent(
    model=llm_client,
    name="handle_other",
    output_schema=ExpertResponse,
    instruction=
    ("You are the Jenova AI assistant. The user just said something that isn't a direct question or action command.\n"
     "Here is the conversation history:\n{history?}\n\n"
     "Respond naturally to the user's input: {input}\n"
     "Put any internal thoughts about intent into 'internal_reasoning'.\n"
     "Put your natural conversational reply into 'final_answer'.\n"))
