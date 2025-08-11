#!/usr/bin/env python3
"""
Custom LangGraph agent optimized for HotPotQA.

This module provides a modified version of the LangGraph agent that generates
concise, direct answers suitable for HotPotQA evaluation.
"""

import os
import sys
from pathlib import Path

# Add the backend src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from datetime import datetime
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from openai import OpenAI
from tavily import TavilyClient
from langchain_openai import ChatOpenAI

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.tools_and_schemas import SearchQueryList, Reflection
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

# HotPotQA-optimized prompts
HOTPOT_QUERY_INSTRUCTIONS = """Generate search queries to find specific facts for a HotPotQA question.

Instructions:
- Generate 2-3 focused search queries to find the specific facts needed
- Focus on finding concrete, factual information
- Target specific entities, dates, locations, or relationships mentioned
- Avoid broad, general queries

Research Topic: {research_topic}
Number of queries: {number_queries}

Format your response as a JSON object with these exact keys:
- "rationale": Brief explanation of why these queries are relevant
- "query": A list of search queries"""

HOTPOT_WEB_SEARCH_INSTRUCTIONS = """Search for specific factual information to answer a HotPotQA question.

Instructions:
- Focus on finding concrete facts, dates, names, locations
- Look for authoritative sources (Wikipedia, official websites, etc.)
- Extract specific information, not general background
- Current date: {current_date}

Research Topic: {research_topic}"""

HOTPOT_REFLECTION_INSTRUCTIONS = """Analyze if you have enough specific facts to answer the HotPotQA question.

Instructions:
- Check if you have the specific fact(s) needed to answer the question
- If missing specific facts, generate follow-up queries
- Focus on concrete information, not general background

Research Topic: {research_topic}

Summaries:
{summaries}

Output Format:
- Format your response as a JSON object with these exact keys:
  - "is_sufficient": true or false
  - "knowledge_gap": Describe what specific fact is missing
  - "follow_up_queries": List of specific queries to find missing facts"""

HOTPOT_ANSWER_INSTRUCTIONS = """Answer the HotPotQA question with a SHORT, DIRECT response.

Instructions:
- Provide a CONCISE answer (1-5 words maximum)
- Do NOT include explanations, citations, or detailed reasoning
- Focus on the specific fact requested
- For yes/no questions, answer only "yes" or "no"
- For location questions, provide just the location name
- For person/entity questions, provide just the name
- For date/time questions, provide just the date/time
- For comparison questions, provide just the comparison result

Examples:
- Question: "Were Scott Derrickson and Ed Wood of the same nationality?"
- Answer: "yes"

- Question: "What government position was held by the woman who portrayed Corliss Archer?"
- Answer: "Chief of Protocol"

- Question: "What science fantasy young adult series has companion books about enslaved worlds?"
- Answer: "Animorphs"

Research Topic: {research_topic}

Summaries:
{summaries}

Answer:"""


def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """Generate search queries optimized for HotPotQA."""
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = HOTPOT_QUERY_INSTRUCTIONS.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """Send search queries to web research node."""
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Conduct web research optimized for HotPotQA."""
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Perform search
    search_results = tavily_client.search(
        query=state["search_query"],
        search_depth="advanced",
        include_domains=["wikipedia.org", "britannica.com", "imdb.com", "official websites"],
        max_results=5,
    )
    
    # Process results
    processed_results = []
    sources_gathered = []
    
    for result in search_results.get("results", []):
        content = result.get("content", "")
        url = result.get("url", "")
        
        if content and url:
            processed_results.append(f"Source: {url}\nContent: {content}")
            sources_gathered.append({
                "value": url,
                "short_url": f"[{len(sources_gathered) + 1}]",
            })
    
    return {
        "web_research_result": ["\n\n".join(processed_results)],
        "sources_gathered": sources_gathered,
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """Reflect on research results for HotPotQA."""
    configurable = Configuration.from_runnable_config(config)
    
    llm = ChatOpenAI(
        model=configurable.reflection_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(Reflection)
    
    current_date = get_current_date()
    formatted_prompt = HOTPOT_REFLECTION_INSTRUCTIONS.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    
    result = structured_llm.invoke(formatted_prompt)
    
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state.get("research_loop_count", 0) + 1,
        "number_of_ran_queries": len(state.get("search_query", [])),
    }


def evaluate_research(state: ReflectionState, config: RunnableConfig) -> OverallState:
    """Evaluate if more research is needed."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """Generate final answer optimized for HotPotQA."""
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    current_date = get_current_date()
    formatted_prompt = HOTPOT_ANSWER_INSTRUCTIONS.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Clean up sources
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create HotPotQA-optimized graph
def create_hotpot_graph():
    """Create a LangGraph optimized for HotPotQA."""
    builder = StateGraph(OverallState, config_schema=Configuration)

    # Define nodes
    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("reflection", reflection)
    builder.add_node("finalize_answer", finalize_answer)

    # Set entrypoint
    builder.add_edge(START, "generate_query")
    
    # Add conditional edges
    builder.add_conditional_edges(
        "generate_query", continue_to_web_research, ["web_research"]
    )
    builder.add_edge("web_research", "reflection")
    builder.add_conditional_edges(
        "reflection", evaluate_research, ["web_research", "finalize_answer"]
    )
    builder.add_edge("finalize_answer", END)

    return builder.compile(name="hotpot-qa-agent")


# Create the graph instance
hotpot_graph = create_hotpot_graph() 