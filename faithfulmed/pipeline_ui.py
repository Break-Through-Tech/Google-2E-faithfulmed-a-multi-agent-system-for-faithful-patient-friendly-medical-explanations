"""Run the ADK pipeline and collect the outputs exposed by its agents."""

from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.plugins import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


STAGE_OUTPUTS = {
    "Extractor": "atoms",
    "Simplifier": "draft",
    "readability": "text",
    "Verifier": "verdict",
    "Refiner": "text",
}
STAGE_LABELS = {"readability": "Readability"}


@dataclass
class PipelineResult:
    outputs: dict[str, list[Any]] = field(default_factory=dict)
    current_text: str | None = None
    last_verified_text: str | None = None
    phase: Literal["idle", "running", "completed", "failed"] = "running"
    active_stage: str | None = None

    def record(self, author: str, state_delta: dict[str, Any]) -> bool:
        key = STAGE_OUTPUTS.get(author)
        if key is None or key not in state_delta:
            return False
        if author in ("readability", "Refiner"):
            text = state_delta[key]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{STAGE_LABELS.get(author, author)} returned no usable text.")
            self.current_text = text
        elif author == "Verifier":
            self.last_verified_text = self.current_text
        self.outputs.setdefault(author, []).append(state_delta[key])
        return True


class StageProgressPlugin(BasePlugin):
    def __init__(self, result: PipelineResult, on_update: Callable[[PipelineResult], None]):
        super().__init__(name="stage_progress")
        self.result = result
        self.on_update = on_update

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        if agent.name in STAGE_OUTPUTS:
            self.result.active_stage = agent.name
            self.on_update(self.result)
        return None


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
    runner = Runner(
        app=App(
            name="faithfulmed_ui",
            root_agent=agent,
            plugins=[StageProgressPlugin(result, on_update)],
        ),
        session_service=sessions,
    )
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

    if result.current_text is None:
        raise RuntimeError("The pipeline finished without a final explanation.")
    if not result.outputs.get("Verifier"):
        raise RuntimeError("The pipeline finished without a verifier report.")
