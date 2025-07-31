from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(search_results: List[Dict[str, Any]], id: int) -> Dict[str, str]:
    """
    Create a map of Tavily search result URLs to short URLs with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://search-result.com/id/"
    urls = [result.get("url", "") for result in search_results if result.get("url")]

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                               contains 'start_index', 'end_index', and
                               'segments' (the citation information).
                               Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    # Sort citations by end_index in descending order.
    # If end_index is the same, secondary sort by start_index descending.
    # This ensures that insertions at the end of the string don't affect
    # the indices of earlier parts of the string that still need to be processed.
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # These indices refer to positions in the *original* text,
        # but since we iterate from the end, they remain valid for insertion
        # relative to the parts of the string already processed.
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


def get_citations_from_tavily_response(tavily_response: Dict[str, Any], resolved_urls_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Extracts and formats citation information from a Tavily search response.

    This function processes the Tavily search results to construct citation objects.
    Since Tavily doesn't provide grounding metadata like Google Search, we create
    citations based on the search results themselves.

    Args:
        tavily_response: The response object from Tavily search API.
        resolved_urls_map: A mapping of original URLs to shortened URLs.

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): The starting character index of the cited
                                     segment in the original text.
              - "end_index" (int): The character index immediately after the
                                   end of the cited segment (exclusive).
              - "segments" (list): A list of individual citation segments with
                                   label, short_url, and value.
    """
    citations = []
    
    # Ensure response and results are present
    if not tavily_response or "results" not in tavily_response:
        return citations

    search_results = tavily_response.get("results", [])
    
    # For Tavily, we'll create citations based on the search results
    # Since we don't have grounding metadata, we'll create a simple citation
    # that can be used to reference the sources
    for idx, result in enumerate(search_results):
        url = result.get("url", "")
        if url and url in resolved_urls_map:
            # Create a citation that spans the entire text
            citation = {
                "start_index": 0,  # Default to beginning of text
                "end_index": len(result.get("content", "")),  # Use content length
                "segments": [{
                    "label": result.get("title", f"Source {idx + 1}"),
                    "short_url": resolved_urls_map[url],
                    "value": url
                }]
            }
            citations.append(citation)
    
    return citations


def get_citations(response, resolved_urls_map):
    """
    Extracts and formats citation information from search results.
    This is a wrapper function that can handle different types of search responses.

    Args:
        response: The search response object (can be Tavily or other search providers).
        resolved_urls_map: A mapping of URLs to resolved URLs.

    Returns:
        list: A list of citation dictionaries with source information.
    """
    # Check if this is a Tavily response
    if isinstance(response, dict) and "results" in response:
        return get_citations_from_tavily_response(response, resolved_urls_map)
    
    # For other response types, return empty list
    # This maintains backward compatibility
    return []
