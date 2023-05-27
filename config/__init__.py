"""General configuration stuff."""
import os
from config.helper_functions import load_from_env_file
from config.init_logger import logger

# Load the environment variables if working in weird environments
load_from_env_file()

# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")