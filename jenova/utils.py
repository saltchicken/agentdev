"""
Utility functions for the agent.
"""

from google.adk import Context


def get_full_history(ctx: Context, max_messages: int = 10) -> list[dict]:
    """
    Extracts the conversation history, ignoring internal 'thought' parts,
    and trims it to the last `max_messages` to fit the context window.
    """
    history = []

    for event in ctx.session.events:
        author = event.author or ""

        if event.content and event.content.parts:
            # Rebuild the text using ONLY parts that are NOT marked as thoughts
            spoken_text = "".join(
                part.text 
                for part in event.content.parts 
                if part.text and not getattr(part, 'thought', False)
            )

            if spoken_text:
                role = "user" if author == "user" else "assistant"
                history.append({"role": role, "content": spoken_text})

    if max_messages > 0:
        return history[-max_messages:]

    return history
