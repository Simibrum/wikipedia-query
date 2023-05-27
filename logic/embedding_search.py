"""Functions for queries on embeddings."""

from openai.embeddings_utils import cosine_similarity
import pandas as pd
from logic.embeddings import get_embedding, openai_api_key


def run_similarity_query(
        df: pd.DataFrame,
        query_text: str,
        embedding_column: str,
        n: int = 10,
        pprint: bool = True,
        api_key: str = openai_api_key
) -> pd.DataFrame:
    """
    Runs a similarity query on a dataframe using a query text.

    Args:
        df: The input dataframe.
        query_text: The query text to search for similarity.
        embedding_column: The name of the column containing the embeddings in the dataframe.
        n: The number of top results to return. Default is 3.
        pprint: Whether to pretty print the resulting dataframe. Default is True.
        api_key: The OpenAI API key to use for embeddings. Default is the value of the OPENAI_API_KEY environment.

    Returns:
        The top matching rows from the dataframe based on cosine similarity.
    """
    # Get the embedding for the query text
    query_embedding = get_embedding([query_text], api_key=api_key)[0]

    # Calculate cosine similarity between embeddings
    df['similarities'] = df[embedding_column].apply(lambda x: cosine_similarity(x, query_embedding))

    # Sort the dataframe based on similarities
    df_sorted = df.sort_values('similarities', ascending=False).head(n)

    if pprint:
        # Pretty print the resulting dataframe
        print(df_sorted.to_string(index=False))

    return df_sorted
