import os
import json
import requests

from vero.tool import tool
from vero.config import settings

search_url = "https://api.bocha.cn/v1/web-search"


@tool
def bocha_search(query: str, max_results: int = 2) -> str:
    """
    Used to search the web and retrieve the most accurate and up-to-date information.

    Instructions:
    - The input should be the user's query question, such as 'Why is the sky blue?'
    - This tool will automatically perform a web search and return a summarized, concise answer.

    Parameters:
        query (str): The query question.
        max_results (int): Maximum number of results to return (default is 2). If you feel the results are inaccurate, you can try increasing this number.

    Example:
        search("What is the latest market value of Microsoft?", 2)

    Returns:
        str: The webpage results found, which the LLM can extract an answer from.
    """
    api_key = settings.BOCHA_API_KEY
    if not api_key:
        raise ValueError("Error: BOCHA_API_KEY is missing or not set.")

    payload = {
        "query": query,
        "summary": True,  # Return summarized text
        "count": max_results,  # Number of search results
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(search_url, headers=headers, data=json.dumps(payload))
        data = resp.json()
    except Exception as e:
        return f"Bocha API request failed: {e}"

    pages = data.get("data", {}).get("webPages", {}).get("value", [])
    if not pages:
        return "No relevant content found."

    results = []
    for page in pages[:max_results]:
        summary = page.get("summary") or page.get("snippet")
        if summary:
            results.append(summary.strip())

    if results:
        return "\n".join(results)
    else:
        return "No valid text found in the search results."
