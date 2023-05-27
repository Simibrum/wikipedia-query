"""Configuration helper functions."""
import os
import tiktoken


# Load the environment variables if working in weird environments
def load_from_env_file():
    """Load environment variables from a file."""
    # Load environment variables from project root .env file
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory of the current file's directory
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    # Set path of .env file
    env_path = os.path.join(parent_dir, ".env")
    try:
        # Load environment variables from .env file
        load_env_vars(env_path)
    except FileNotFoundError:
        pass


def load_env_vars(path):
    """Load environment variables from a file."""
    with open(path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                # Split the line into the variable name and value
                var, value = line.split("=")

                # Strip leading and trailing whitespace from the variable name and value
                var = var.strip()
                value = value.strip()

                if var and value:
                    # Set the environment variable
                    os.environ[var] = value


# OpenAI method for counting tokens
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
