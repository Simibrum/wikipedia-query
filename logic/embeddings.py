"""Code to provide embeddings for text using the OpenAI Embedding API."""
from typing import List
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import openai
import random
import time
from config import logger, openai_api_key


def convert_array_to_string(array):
    """Convert a numpy array to a string representation.

    Args:
        array: The numpy array to convert.

    Returns:
        A string representation of the array.
    """
    return np.array2string(array, separator=",")[1:-1]


def get_embedding(
        text: List[str],
        model: str = "text-embedding-ada-002",
        api_key: str = openai_api_key
) -> List[List[float]]:
    """
    Get the embedding for a given text using the OpenAI Embedding API.

    Args:
        text: The input text as a list of strings.
        model: The model name or ID to use for embeddings. Default is "text-embedding-ada-002".
        api_key: The OpenAI API key to use for embeddings. Default is the value of the OPENAI_API_KEY environment
            variable.

    Returns:
        A list of embeddings corresponding to the input text, each embedding is a list of floats.
    """
    return api_request(text, model, api_key)


def add_embeddings_to_dataframe(
        df: pd.DataFrame,
        text_column: str = "clipping_text",
        embedding_column: str = "embedding",
        model: str = "text-embedding-ada-002",
        api_key: str = openai_api_key,
        progress_bar: bool = True,
        save_interval: int = 100,
        save_path: str = "progress_temp.pkl",
        retry_on_error: bool = True,
        batch_size: int = 10
) -> pd.DataFrame:
    """
    Add embeddings to a dataframe for a specified text column.

    Args:
        df: The input dataframe.
        text_column: The name of the column containing the text.
        embedding_column: The name of the column to store the embeddings.
        model: The model name or ID to use for embeddings. Default is "text-embedding-ada-002".
        api_key: The OpenAI API key to use for embeddings. Default is the value of the OPENAI_API_KEY environment
            variable.
        progress_bar: Whether to display a progress bar. Default is True.
        save_interval: The number of rows to save progress after. Default is 100.
        save_path: The path to the temporary file for saving progress. Default is "progress_temp.csv".
        retry_on_error: Whether to retry processing rows that throw exceptions at the end. Default is True.
        batch_size: The number of rows to process in each batch. Default is 10.

    Returns:
        The modified dataframe with the embeddings added.
    """
    # Create an iterator for the text column
    if progress_bar:
        iterator = tqdm(df[text_column], desc="Adding Embeddings", total=len(df), dynamic_ncols=True)
    else:
        iterator = df[text_column]

    # Track progress
    progress_count = 0

    # Track error cases
    error_rows = []

    # Track processed rows
    processed_rows = []

    # Add the embedding column to the DataFrame
    df[embedding_column] = None

    # Iterate over embeddings and build text
    for i, text in enumerate(iterator):
        try:
            processed_rows.append(i)

            # Create batches of text
            if len(processed_rows) % batch_size == 0 or i == len(df) - 1:
                offset = min(batch_size, len(processed_rows))
                batch_text = df[text_column].iloc[processed_rows[-offset:]].tolist()
                batch_embeddings = get_embedding(batch_text, model, api_key)

                for j in range(len(batch_embeddings)):
                    index = processed_rows[-offset] + j
                    df.at[index, embedding_column] = batch_embeddings[j]

                # Save progress
                if len(processed_rows) % save_interval == 0 and len(processed_rows) > 0:
                    df.to_pickle(save_path)
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            error_rows.append(i)

    # Save final progress
    df.to_pickle(save_path)

    # Retry error rows
    # Retry processing rows that threw exceptions
    if retry_on_error and error_rows:
        for row in error_rows:
            try:
                text = df.loc[row, text_column]
                embeddings = get_embedding([text], model, api_key)
                df.at[row, embedding_column] = embeddings[0]
            except Exception as e:
                # Handle the error, e.g., print an error message or perform error-specific actions
                pass

    return df


def api_request(
        text: List[str], model: str = "text-embedding-ada-002", api_key: str = openai_api_key
) -> List[List[float]]:
    """Make a request to the openai api."""
    max_tries = 5
    initial_delay = 1
    backoff_factor = 2
    max_delay = 16
    jitter_range = (1, 3)

    # Get rid of newlines
    text = [t.replace("\n", " ") for t in text]

    for attempt in range(1, max_tries + 1):
        try:
            response = openai.Embedding.create(input=text, model=model, api_key=api_key)
            embeddings = [r['embedding'] for r in response['data']]
            return embeddings
        except Exception as e:
            if attempt == max_tries:
                logger.error(f"API request failed after {attempt} attempts with final error {e}.")
                return []

            delay = min(initial_delay * (backoff_factor ** (attempt - 1)), max_delay)
            jitter = random.uniform(jitter_range[0], jitter_range[1])
            sleep_time = delay + jitter
            logger.error(f"API request failed with error: {e}. Retrying in {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
