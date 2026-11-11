"""Run the ADK pipeline and collect the outputs exposed by its agents."""

from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Callable

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


STAGE_OUTPUTS = {
    "Extractor": "atoms",
    "Simplifier": "draft",
    "Verifier": "verdict",
    "Refiner": "draft",
    "Readability": "final",
}


@dataclass
class PipelineResult:
    outputs: dict[str, list[Any]] = field(default_factory=dict)

    def record(self, author: str, state_delta: dict[str, Any]) -> bool:
        key = STAGE_OUTPUTS.get(author)
        if key is None or key not in state_delta:
            return False
        self.outputs.setdefault(author, []).append(state_delta[key])
        return True


async def run_pipeline(
    note: str,
    result: PipelineResult,
    on_update: Callable[[PipelineResult], None],
    agent: BaseAgent | None = None,
) -> None:
    """Run one isolated session, recording only outputs actually emitted by ADK."""
    if not note.strip():
        raise ValueError("Enter a medical note before running the pipeline.")

    if agent is None:
        from faithfulmed.agent import root_agent

        agent = root_agent

    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="faithfulmed_ui", user_id="local_user")
    runner = Runner(app_name="faithfulmed_ui", agent=agent, session_service=sessions)
    try:
        message = types.Content(role="user", parts=[types.Part.from_text(text=note)])
        async with aclosing(
            runner.run_async(user_id="local_user", session_id=session.id, new_message=message)
        ) as events:
            async for event in events:
                if event.error_code or event.error_message:
                    raise RuntimeError(
                        f"ADK reported an error ({event.error_code or 'unknown'}): "
                        f"{event.error_message or 'No details provided'}"
                    )
                if result.record(event.author, event.actions.state_delta):
                    on_update(result)
    finally:
        await runner.close()

    final = result.outputs.get("Readability", [])
    if not final or not isinstance(final[-1], str) or not final[-1].strip():
        raise RuntimeError("The pipeline finished without a final readability output.")
