import json
from typing import Optional, List, Dict

import pytest


from vero.tool import tool


@tool
def search(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Search documents using a query string.

    Args:
        query: The search query.
        top_k: Number of results to return.
        filters: Optional key-value filters.

    Returns:
        A list of document IDs.
    """
    return [f"doc_{i}" for i in range(top_k)]


@pytest.fixture
def search_tool():
    """
    Fixture that provides the wrapped search tool.
    """
    return search


def test_tool_is_callable(search_tool):
    """
    The wrapped tool should remain callable as a normal Python function.
    """
    result = search_tool("hello", top_k=3)
    assert result == ["doc_0", "doc_1", "doc_2"]


def test_openai_schema_top_level(search_tool):
    """
    Validate top-level OpenAI function schema structure.
    """
    schema = search_tool.to_openai_schema()

    assert schema["type"] == "function"
    assert "function" in schema

    fn = schema["function"]
    assert fn["name"] == "search"
    assert isinstance(fn["description"], str)


def test_openai_schema_parameters_object(search_tool):
    """
    Validate parameters object structure.
    """
    params = search_tool.to_openai_schema()["function"]["parameters"]

    assert params["type"] == "object"
    assert isinstance(params["properties"], dict)
    assert isinstance(params["required"], list)


def test_required_parameters(search_tool):
    """
    Parameters without default values should be required.
    """
    params = search_tool.to_openai_schema()["function"]["parameters"]

    assert "query" in params["required"]
    assert "top_k" not in params["required"]
    assert "filters" not in params["required"]


def test_parameter_type_mapping(search_tool):
    """
    Python type annotations should be mapped to correct JSON Schema types.
    """
    props = search_tool.to_openai_schema()["function"]["parameters"]["properties"]

    assert props["query"]["type"] == "string"
    assert props["top_k"]["type"] == "integer"


def test_optional_dict_schema(search_tool):
    """
    Optional[Dict[str, str]] should be represented as anyOf[object, null].
    """
    filters_schema = (
        search_tool.to_openai_schema()
        ["function"]["parameters"]["properties"]["filters"]
    )

    assert "anyOf" in filters_schema

    object_schema = next(
        s for s in filters_schema["anyOf"] if s.get("type") == "object"
    )

    assert "additionalProperties" in object_schema
    assert object_schema["additionalProperties"]["type"] == "string"


def test_schema_is_json_serializable(search_tool):
    """
    The generated schema must be JSON-serializable for OpenAI API usage.
    """
    schema = search_tool.to_openai_schema()
    json.dumps(schema)  # should not raise
