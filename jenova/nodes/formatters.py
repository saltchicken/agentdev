from google.adk import Event
from jenova.nodes.experts import ExpertResponse

def format_expert_response(response: ExpertResponse) -> Event:
    """
    Converts the structured Pydantic response into an Event payload 
    with distinct 'thought' and 'spoken' text parts.
    """
    return Event(
        content={
            "parts": [
                {
                    "text": response.internal_reasoning,
                    "thought": True
                },
                {
                    "text": response.final_answer
                }
            ]
        }
    )
