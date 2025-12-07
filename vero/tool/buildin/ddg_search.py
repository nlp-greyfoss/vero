from ddgs import DDGS
from vero.tool import tool

@tool
def duckduckgo_search(query: str, max_results: int = 3) -> str:
    """
    Perform a web search and return formatted results.

    Args:
        query (str): The search query string.
        max_results (int, optional): Maximum number of search results to retrieve. Defaults to 3.

    Returns:
        str: A formatted string containing titles, URLs and snippets of the search results,
             or an error/fallback message if the search fails or yields no results.
    """
    try:
        # Use a context manager to ensure the DDGS session is properly closed
        with DDGS() as ddgs:
            # Execute the search
            results = ddgs.text(query, max_results=max_results)

        if not results:
            return "No search results found."

        output_lines = []
        for r in results:
            # r is a dict containing 'title', 'href', 'body'
            title = r.get("title", "No title")
            href = r.get("href", "")
            body = r.get("body", "")
            output_lines.append(f"Title: {title}\nLink: {href}\nSnippet: {body}\n")

        # Join and return the output
        return "\n".join(output_lines)

    except Exception as e:
        # Catch all exceptions to prevent the agent from crashing
        return f"DuckDuckGo search failed: {e}"
