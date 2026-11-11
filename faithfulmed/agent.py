"""Google ADK entry point for the FaithfulMed pipeline."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from faithfulmed.adk_pipeline import faithfulmed as root_agent