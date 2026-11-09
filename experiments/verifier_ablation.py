import os
import getpass

os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
import asyncio
import time
import uuid
import pandas as pd
import textstat

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


from google.adk.agents import SequentialAgent, LlmAgent

from faithfulmed.adk_pipeline import (
    faithfulmed,
    ExtractAtomsStep,
    SimplifyStep,
    readability_agent,
    MODEL,
)

# Build the ablation without changing the original pipeline
readability_ablation = LlmAgent(
    name="ReadabilityAblation",
    model=MODEL,
    description=readability_agent.description,
    instruction=readability_agent.instruction,
    output_key="text",
)

faithfulmed_no_verifier = SequentialAgent(
    name="FaithfulMedNoVerifier",
    sub_agents=[
        ExtractAtomsStep(name="ExtractorAblation"),
        SimplifyStep(name="SimplifierAblation"),
        readability_ablation,
    ],
)



async def run_pipeline(agent, medical_text):
    session_service = InMemorySessionService()
    app_name = "faithfulmed_ablation"
    user_id = "researcher"
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=medical_text)],
    )

    start = time.perf_counter()

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.error_code:
            raise RuntimeError(
                event.error_message or event.error_code
            )

    latency = time.perf_counter() - start

    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    return {
        "output": session.state.get("text", ""),
        "latency_seconds": latency,
    }


async def main():
    medical_text = (
        "The patient has type 2 diabetes. "
        "Blood glucose was 180 mg/dL. "
        "Metformin 500 mg was prescribed twice daily."
    )

    for name, agent in [
        ("full", faithfulmed),
        ("no_verification_loop", faithfulmed_no_verifier),
    ]:
        result = await run_pipeline(agent, medical_text)

        print(f"\n--- {name} ---")
        print(result["output"])
        print("Latency:", result["latency_seconds"])
        print(
            "Reading grade:",
            textstat.flesch_kincaid_grade(result["output"])
        )


if __name__ == "__main__":
    asyncio.run(main())
