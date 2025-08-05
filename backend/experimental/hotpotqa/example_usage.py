#!/usr/bin/env python3
"""
Example usage of LangGraph HotPotQA integration.

This script demonstrates how to use the HotPotQARunner class to evaluate
the LangGraph agent on HotPotQA questions with detailed traces including system prompts.
"""

import sys
import json
from pathlib import Path

# Add the current directory to path to import the runner
sys.path.append(str(Path(__file__).parent))

from langgraph_hotpotqa import HotPotQARunner


def example_single_question():
    """Example: Run on a single question."""
    print("=== Single Question Example ===")
    
    # Initialize runner (assuming data is in ../ReAct/data/)
    runner = HotPotQARunner(data_dir="../ReAct/data", split="dev")
    
    # Run on question index 0
    trace = runner.run_single_question(0)
    
    print(f"Question: {trace['question']}")
    print(f"Ground Truth: {trace['ground_truth']}")
    print(f"Prediction: {trace.get('final_answer', 'No answer')}")
    print(f"Exact Match: {trace['evaluation']['exact_match']}")
    print(f"F1 Score: {trace['evaluation']['f1']:.3f}")
    print(f"Execution Time: {trace['execution_time']:.2f}s")
    
    # Show detailed agent trace
    print("\nDetailed Agent Trace:")
    for i, agent_step in enumerate(trace.get('agent_trace', []), 1):
        print(f"{i}. {agent_step['agent']}:")
        print(f"   System Prompt: {agent_step.get('system_prompt', 'N/A')[:100]}...")
        print(f"   Input: {agent_step['input']}")
        print(f"   Output: {agent_step['output']}")
        print()


def example_batch_run():
    """Example: Run on a batch of questions."""
    print("\n=== Batch Run Example ===")
    
    # Initialize runner
    runner = HotPotQARunner(data_dir="../ReAct/data", split="dev")
    
    # Run on first 3 questions
    results = runner.run_batch(
        start_idx=0,
        num_questions=3,
        output_file="hotpotqa_results.json"
    )
    
    # Print summary
    valid_results = [r for r in results if "evaluation" in r]
    if valid_results:
        total_em = sum(1 for r in valid_results if r["evaluation"]["exact_match"])
        total_f1 = sum(r["evaluation"]["f1"] for r in valid_results)
        avg_em = total_em / len(valid_results)
        avg_f1 = total_f1 / len(valid_results)
        
        print(f"\nBatch Results Summary:")
        print(f"Questions processed: {len(valid_results)}")
        print(f"Exact Match: {avg_em:.3f} ({total_em}/{len(valid_results)})")
        print(f"Average F1: {avg_f1:.3f}")
        
        # Show trace summary for first result
        if valid_results:
            first_result = valid_results[0]
            print(f"\nSample Agent Trace (Question {first_result['question_idx']}):")
            for i, agent_step in enumerate(first_result.get('agent_trace', []), 1):
                print(f"{i}. {agent_step['agent']}:")
                print(f"   System Prompt Length: {len(str(agent_step.get('system_prompt', '')))} chars")
                print(f"   Output Length: {len(str(agent_step['output']))} chars")


def example_custom_config():
    """Example: Run with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    runner = HotPotQARunner(data_dir="../ReAct/data", split="dev")
    
    # Custom configuration
    config = {
        "initial_search_query_count": 5,  # Generate more search queries
        "max_research_loops": 3,         # Allow more research loops
        "reasoning_model": "gpt-4o-mini"  # Use different model
    }
    
    trace = runner.run_single_question(1, config=config)
    
    print(f"Custom config results:")
    print(f"Research loops used: {trace.get('research_loop_count', 0)}")
    print(f"Search queries generated: {len(trace.get('search_queries', []))}")
    print(f"F1 Score: {trace['evaluation']['f1']:.3f}")
    
    # Show agent trace
    print(f"\nAgent Trace:")
    for i, agent_step in enumerate(trace.get('agent_trace', []), 1):
        print(f"{i}. {agent_step['agent']}:")
        if isinstance(agent_step['output'], list):
            print(f"   Output: {len(agent_step['output'])} items")
        else:
            print(f"   Output: {str(agent_step['output'])[:100]}...")


def example_trace_analysis():
    """Example: Analyze agent traces."""
    print("\n=== Trace Analysis Example ===")
    
    runner = HotPotQARunner(data_dir="../ReAct/data", split="dev")
    
    # Run a few questions and analyze traces
    results = runner.run_batch(start_idx=0, num_questions=2)
    
    print("Agent Trace Analysis:")
    for result in results:
        if "agent_trace" in result:
            print(f"\nQuestion {result['question_idx']}: {result['question'][:50]}...")
            print(f"Answer: {result['final_answer']}")
            print(f"EM: {result['evaluation']['exact_match']}")
            
            # Count agent steps
            agent_counts = {}
            for step in result['agent_trace']:
                agent = step['agent']
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            print(f"Agent steps: {agent_counts}")
            
            # Analyze system prompts
            total_prompt_length = 0
            for step in result['agent_trace']:
                if 'system_prompt' in step:
                    total_prompt_length += len(step['system_prompt'])
            print(f"Total system prompt length: {total_prompt_length} chars")


def example_system_prompt_analysis():
    """Example: Analyze system prompts used by agents."""
    print("\n=== System Prompt Analysis Example ===")
    
    runner = HotPotQARunner(data_dir="../ReAct/data", split="dev")
    
    # Run a single question and analyze system prompts
    trace = runner.run_single_question(0)
    
    print("System Prompt Analysis:")
    for i, agent_step in enumerate(trace.get('agent_trace', []), 1):
        print(f"\n{i}. {agent_step['agent']}:")
        
        if 'system_prompt' in agent_step:
            prompt = agent_step['system_prompt']
            print(f"   Prompt Length: {len(prompt)} chars")
            print(f"   First 200 chars: {prompt[:200]}...")
            
            # Count key instructions
            key_phrases = ['Instructions:', 'Format:', 'Example:', 'Research Topic:']
            for phrase in key_phrases:
                if phrase in prompt:
                    print(f"   Contains: {phrase}")
        else:
            print("   No system prompt recorded")


if __name__ == "__main__":
    print("LangGraph HotPotQA Integration Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_single_question()
        example_batch_run()
        example_custom_config()
        example_trace_analysis()
        example_system_prompt_analysis()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the HotPotQA data files are available in ../ReAct/data/")
        print("You can download them from the original ReAct repository.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure your environment is properly set up with:")
        print("- OPENAI_API_KEY environment variable")
        print("- TAVILY_API_KEY environment variable")
        print("- All required dependencies installed") 