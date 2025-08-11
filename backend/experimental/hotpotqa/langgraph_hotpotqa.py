#!/usr/bin/env python3
"""
LangGraph HotPotQA Integration

This script runs the LangGraph research agent on HotPotQA questions following the ReAct approach.
It records actual agent inputs and outputs including system prompts.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the backend src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from langchain_core.messages import HumanMessage, AIMessage
from agent.state import OverallState

# Import the HotPotQA-optimized graph
try:
    from hotpot_graph import hotpot_graph
    GRAPH = hotpot_graph
except ImportError:
    # Fallback to original graph if hotpot_graph is not available
    from agent.graph import graph as GRAPH


class HotPotQARunner:
    """Runner for HotPotQA questions using LangGraph agent."""
    
    def __init__(self, data_dir: str = "data", split: str = "dev"):
        """
        Initialize the HotPotQA runner.
        
        Args:
            data_dir: Directory containing HotPotQA data files
            split: Dataset split to use ('train', 'dev', 'test')
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load HotPotQA data from JSON file."""
        split_files = {
            "train": "hotpot_train_v1.1_simplified.json",
            "dev": "hotpot_dev_v1_simplified.json", 
            "test": "hotpot_test_v1_simplified.json"
        }
        
        data_file = self.data_dir / split_files[self.split]
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Extract question and answer pairs
        return [(item['question'], item['answer']) for item in data]
    
    def normalize_answer(self, s: str) -> str:
        """Normalize answer for comparison."""
        import re
        import string
        
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        
        def white_space_fix(text):
            return " ".join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def evaluate_answer(self, prediction: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate prediction against ground truth."""
        pred_norm = self.normalize_answer(prediction)
        gt_norm = self.normalize_answer(ground_truth)
        
        em = (pred_norm == gt_norm)
        
        # Simple F1 calculation
        pred_tokens = pred_norm.split()
        gt_tokens = gt_norm.split()
        
        if not pred_tokens or not gt_tokens:
            f1 = 0.0
        else:
            common = set(pred_tokens) & set(gt_tokens)
            if not common:
                f1 = 0.0
            else:
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(gt_tokens)
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'exact_match': em,
            'f1': f1,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'prediction_normalized': pred_norm,
            'ground_truth_normalized': gt_norm
        }
    
    def run_single_question(self, idx: int, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run LangGraph agent on a single HotPotQA question.
        
        Args:
            idx: Index of the question in the dataset
            config: Optional configuration for the agent
            
        Returns:
            Dictionary containing results and detailed trace
        """
        if idx >= len(self.data):
            raise ValueError(f"Index {idx} out of range. Dataset has {len(self.data)} questions.")
        
        question, ground_truth = self.data[idx]
        
        # Initialize state
        state: OverallState = {
            "messages": [HumanMessage(content=question)],
            "search_query": [],
            "web_research_result": [],
            "sources_gathered": [],
            "initial_search_query_count": 3,
            "max_research_loops": 2,
            "research_loop_count": 0,
            "reasoning_model": "gpt-4o",
        }
        
        # Update with config if provided
        if config:
            state.update(config)
        
        print(f"Question {idx}: {question}")
        print(f"Ground truth: {ground_truth}")
        print("-" * 80)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the LangGraph agent with trace recording
            result = GRAPH.invoke(state)
            
            # Extract final answer from the last message
            messages = result.get("messages", [])
            final_answer = None
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    final_answer = last_message.content
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Evaluate the answer
            evaluation = self.evaluate_answer(final_answer or "", ground_truth)
            
            # Create detailed trace with actual agent inputs/outputs
            trace = {
                "question_idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "final_answer": final_answer,
                "execution_time": execution_time,
                "evaluation": evaluation,
                "agent_trace": self._extract_detailed_agent_trace(result, state)
            }
            
            print(f"Final Answer: {final_answer}")
            print(f"Exact Match: {evaluation['exact_match']}")
            print(f"F1 Score: {evaluation['f1']:.3f}")
            print(f"Execution Time: {execution_time:.2f}s")
            print("=" * 80)
            
            return trace
            
        except Exception as e:
            print(f"Error processing question {idx}: {e}")
            return {
                "question_idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _extract_detailed_agent_trace(self, result: Dict, initial_state: Dict) -> List[Dict[str, Any]]:
        """
        Extract detailed trace of actual agent inputs and outputs including system prompts.
        
        Args:
            result: The result from the LangGraph agent
            initial_state: The initial state passed to the agent
            
        Returns:
            List of detailed agent trace entries
        """
        trace = []
        
        # Extract search queries (generate_query agent)
        search_queries = result.get("search_query", [])
        if search_queries:
            # Get the actual prompt used for query generation
            from agent.prompts import query_writer_instructions
            from agent.utils import get_research_topic
            from datetime import datetime
            
            current_date = datetime.now().strftime("%B %d, %Y")
            formatted_prompt = query_writer_instructions.format(
                current_date=current_date,
                research_topic=get_research_topic(initial_state["messages"]),
                number_queries=initial_state.get("initial_search_query_count", 3),
            )
            
            trace.append({
                "agent": "generate_query",
                "system_prompt": formatted_prompt,
                "input": {
                    "messages": [msg.dict() for msg in initial_state["messages"]],
                    "initial_search_query_count": initial_state.get("initial_search_query_count", 3)
                },
                "output": search_queries
            })
        
        # Extract web research results (web_research agent)
        web_results = result.get("web_research_result", [])
        if web_results:
            # Get the actual prompt used for web research
            from agent.prompts import web_searcher_instructions
            
            web_prompts = []
            for query in search_queries:
                formatted_prompt = web_searcher_instructions.format(
                    current_date=current_date,
                    research_topic=query,
                )
                web_prompts.append(formatted_prompt)
            
            trace.append({
                "agent": "web_research",
                "system_prompts": web_prompts,
                "input": {
                    "search_queries": search_queries,
                    "tavily_search_results": "Raw search results from Tavily API"
                },
                "output": web_results
            })
        
        # Extract reflection results (reflection agent)
        research_loop_count = result.get("research_loop_count", 0)
        if research_loop_count > 0:
            # Get the actual prompt used for reflection
            from agent.prompts import reflection_instructions
            
            reflection_prompt = reflection_instructions.format(
                current_date=current_date,
                research_topic=get_research_topic(initial_state["messages"]),
                summaries="\n---\n\n".join(web_results),
            )
            
            trace.append({
                "agent": "reflection",
                "system_prompt": reflection_prompt,
                "input": {
                    "web_research_results": web_results,
                    "research_loop_count": research_loop_count
                },
                "output": {
                    "is_sufficient": result.get("is_sufficient", False),
                    "knowledge_gap": result.get("knowledge_gap", ""),
                    "follow_up_queries": result.get("follow_up_queries", [])
                }
            })
        
        # Extract final answer (finalize_answer agent)
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                # Get the actual prompt used for final answer
                from agent.prompts import answer_instructions
                
                answer_prompt = answer_instructions.format(
                    current_date=current_date,
                    research_topic=get_research_topic(initial_state["messages"]),
                    summaries="\n---\n\n".join(web_results),
                )
                
                trace.append({
                    "agent": "finalize_answer",
                    "system_prompt": answer_prompt,
                    "input": {
                        "web_research_results": web_results,
                        "sources_gathered": result.get("sources_gathered", []),
                        "messages": [msg.dict() for msg in messages[:-1]]  # All messages except the final answer
                    },
                    "output": last_message.content
                })
        
        return trace
    
    def run_batch(self, start_idx: int = 0, num_questions: int = 10, 
                  output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run LangGraph agent on a batch of HotPotQA questions.
        
        Args:
            start_idx: Starting index for questions
            num_questions: Number of questions to process
            output_file: Optional file to save results
            
        Returns:
            List of trace dictionaries
        """
        results = []
        correct_answers = 0
        total_f1 = 0.0
        
        for i in range(start_idx, min(start_idx + num_questions, len(self.data))):
            print(f"\nProcessing question {i}/{len(self.data)}")
            
            trace = self.run_single_question(i)
            results.append(trace)
            
            if "evaluation" in trace:
                if trace["evaluation"]["exact_match"]:
                    correct_answers += 1
                total_f1 += trace["evaluation"]["f1"]
            
            # Print running statistics
            processed = len(results)
            if processed > 0:
                em_score = correct_answers / processed
                avg_f1 = total_f1 / processed
                print(f"Running EM: {em_score:.3f} ({correct_answers}/{processed})")
                print(f"Running F1: {avg_f1:.3f}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return results


def main():
    """Main function to run HotPotQA evaluation."""
    parser = argparse.ArgumentParser(description="Run LangGraph agent on HotPotQA")
    parser.add_argument("--data-dir", default="data", help="Directory containing HotPotQA data")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"], 
                       help="Dataset split to use")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting question index")
    parser.add_argument("--num-questions", type=int, default=10, help="Number of questions to process")
    parser.add_argument("--output-file", help="File to save results")
    parser.add_argument("--single-question", type=int, help="Run on single question by index")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = HotPotQARunner(data_dir=args.data_dir, split=args.split)
    
    if args.single_question is not None:
        # Run single question
        trace = runner.run_single_question(args.single_question)
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(trace, f, indent=2)
    else:
        # Run batch
        results = runner.run_batch(
            start_idx=args.start_idx,
            num_questions=args.num_questions,
            output_file=args.output_file
        )
        
        # Print final statistics
        if results:
            valid_results = [r for r in results if "evaluation" in r]
            if valid_results:
                total_em = sum(1 for r in valid_results if r["evaluation"]["exact_match"])
                total_f1 = sum(r["evaluation"]["f1"] for r in valid_results)
                avg_em = total_em / len(valid_results)
                avg_f1 = total_f1 / len(valid_results)
                
                print(f"\nFinal Results:")
                print(f"Exact Match: {avg_em:.3f} ({total_em}/{len(valid_results)})")
                print(f"Average F1: {avg_f1:.3f}")


if __name__ == "__main__":
    main() 