import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext

from langchain_community.utilities import WikipediaAPIWrapper

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")

# --- State Management Tool ---

def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's initial prompt to the state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] Added to PROMPT: {prompt}")
    return {"status": "success"}

# --- External Knowledge Tool (Wikipedia) ---

wikipedia_api = WikipediaAPIWrapper()

def search_wikipedia(query: str) -> str:
    """Searches Wikipedia for historical background, cultural significance, and attractions."""
    return wikipedia_api.run(query)

# --- 1. Researcher Agent ---

comprehensive_researcher = Agent(
    name="comprehensive_researcher",
    model=model_name,
    description="A travel research assistant that gathers detailed information about sightseeing locations, landmarks, and attractions using Wikipedia.",
    instruction="""
    You are an expert travel research assistant. Your goal is to fully answer the user's PROMPT about sightseeing, travel destinations, landmarks, or places of interest.

    You have access to:
    1. A Wikipedia search tool for general and detailed knowledge about places, history, culture, and attractions.

    Instructions:
    - Carefully analyze the user's PROMPT.
    - Identify the location, landmark, or type of attraction being asked about.
    - Use the Wikipedia tool to gather relevant information such as:
        * Historical background
        * Cultural significance
        * Key highlights or attractions
        * Visitor tips (if available)
    - If the query is broad (e.g., "places to visit in X"), gather structured data covering multiple key attractions.
    - Synthesize all retrieved information into a structured research summary.

    PROMPT:
    { PROMPT }
    """,
    tools=[search_wikipedia],
    output_key="research_data"
)

# --- 2. Response Formatter Agent ---

response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Formats travel and sightseeing information into a friendly, engaging guide-style response.",
    instruction="""
    You are a friendly and knowledgeable sightseeing guide.

    Your task is to take the RESEARCH_DATA and present it as a helpful and engaging travel response.

    Guidelines:
    - Start with a brief introduction to the place or attraction.
    - Highlight the most important or interesting aspects (history, significance, must-see spots).
    - If multiple places are involved, organize them clearly (bullet points or sections).
    - Include interesting facts or tips where relevant.
    - Keep the tone conversational and helpful, like a real tour guide.
    - If some details are missing, present what is available without mentioning gaps.

    RESEARCH_DATA:
    { research_data }
    """
)

# --- Workflow (Sequential Execution) ---

tour_guide_workflow = SequentialAgent(
    name="tour_guide_workflow",
    description="Main workflow for handling user requests related to sightseeing and travel guidance.",
    sub_agents=[
        comprehensive_researcher,  # Step 1: Gather travel data
        response_formatter,        # Step 2: Format response
    ]
)

# --- Root Agent (Entry Point) ---

root_agent = Agent(
    name="greeter",
    model=model_name,
    description="The main entry point for the Sightseeing Guide Agent.",
    instruction="""
    - Greet the user and let them know you can help them explore sightseeing destinations, landmarks, and travel ideas.
    - Ask what place or destination they are interested in.
    - When the user responds, use the 'add_prompt_to_state' tool to save their query.
    - After saving the prompt, transfer control to the 'tour_guide_workflow' agent.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[tour_guide_workflow]
)