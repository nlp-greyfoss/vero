import os
from tavily import TavilyClient

from vero.tool import tool
from vero.config import settings


@tool
def google_search(query: str, max_results: int = 3) -> str:
    """
    Tool: Used to search for relevant context on Google.

    Usage:
    - The input should be a query question from the user, such as 'How to add numbers in Clojure?'
    - If the question is complex or too long, it is recommended to split the question into multiple sub-queries and then call this tool to improve search accuracy.

    Parameters:
        query (str): The query question to search for.
        max_results (int, optional): Maximum number of search results to retrieve. Defaults to 3.


    Returns:
        str: The context related to the question, which the LLM can further extract the answer from.
    """
    # Check if TAVILY_API_KEY exists
    api_key = settings.TAVILY_API_KEY
    if not api_key:
        raise ValueError("Error: TAVILY_API_KEY is missing or not set.")

    # If API key exists, proceed with search
    tavily_client = TavilyClient(api_key=api_key)

    return tavily_client.qna_search(query, max_results=max_results)
