# LangGraph HotPotQA Integration

This directory contains integration code to run the LangGraph research agent on HotPotQA questions following the ReAct approach. The implementation records all traces including inputs and outputs from each agent node.

## Files

- `langgraph_hotpotqa.py`: Main integration script that runs LangGraph agent on HotPotQA with detailed trace recording
- `hotpot_graph.py`: HotPotQA-optimized LangGraph agent with concise answer generation
- `example_usage.py`: Example scripts showing how to use the integration with detailed traces
- `README.md`: This documentation file
- `IMPROVEMENTS.md`: Documentation of prompt optimization improvements

## Setup

1. **Install Dependencies**: Make sure you have all the required dependencies from the main project:
   ```bash
   cd backend
   pip install -e .
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export TAVILY_API_KEY="your-tavily-api-key"
   ```

3. **Download HotPotQA Data**: You need the HotPotQA dataset files. You can get them from the original ReAct repository or download them directly:
   ```bash
   # Create data directory
   mkdir -p ../ReAct/data
   
   # Download the simplified HotPotQA dev set
   wget https://raw.githubusercontent.com/ysu1989/ReAct/main/data/hotpot_dev_v1_simplified.json -O ../ReAct/data/hotpot_dev_v1_simplified.json
   ```

## Usage

### Command Line Interface

The main script can be run from the command line:

```bash
# Run on a single question
python langgraph_hotpotqa.py --single-question 0 --output-file result.json

# Run on a batch of questions
python langgraph_hotpotqa.py --start-idx 0 --num-questions 10 --output-file batch_results.json

# Use different dataset split
python langgraph_hotpotqa.py --split dev --num-questions 5
```

### Programmatic Usage

```python
from langgraph_hotpotqa import HotPotQARunner

# Initialize runner
runner = HotPotQARunner(data_dir="../ReAct/data", split="dev")

# Run single question
trace = runner.run_single_question(0)
print(f"Exact Match: {trace['evaluation']['exact_match']}")
print(f"F1 Score: {trace['evaluation']['f1']}")

# Run batch
results = runner.run_batch(start_idx=0, num_questions=10, output_file="results.json")
```

### Example Script

Run the example script to see different usage patterns:

```bash
python example_usage.py
```

## Features

### HotPotQA-Optimized Answer Generation

The integration uses specially optimized prompts that generate concise, direct answers suitable for HotPotQA evaluation:

- **Concise Answers**: 1-5 words maximum, no explanations
- **Direct Responses**: "yes"/"no" for yes/no questions
- **Factual Focus**: Names, locations, dates without elaboration
- **Proper Formatting**: Matches HotPotQA answer expectations

### Detailed Trace Recording with System Prompts

The integration records comprehensive traces with **actual system prompts** and **real agent inputs/outputs**:

- **Question and Ground Truth**: Original question and expected answer
- **Final Answer**: The agent's final prediction
- **Agent Trace**: Detailed list of agent steps with:
  - **System Prompts**: Actual prompts used by each agent (e.g., `query_writer_instructions`, `web_searcher_instructions`)
  - **Inputs**: Real inputs received by each agent (messages, search queries, research results)
  - **Outputs**: Actual outputs produced by each agent (search queries, research summaries, final answers)
  - **Agent Types**: `generate_query`, `web_research`, `reflection`, `finalize_answer`
- **Execution Time**: Time taken for each question
- **Evaluation Metrics**: Exact match and F1 scores

This provides **complete visibility** into agent behavior for debugging and analysis.

### Benefits of Detailed Traces

- **Complete Visibility**: See exactly what instructions each agent received
- **Better Analysis**: Analyze system prompt effectiveness across different questions
- **Reproducible Debugging**: Full trace allows exact reproduction of agent behavior
- **Prompt Engineering**: Test and optimize prompts based on trace analysis
- **Agent Behavior Understanding**: Understand why agents make specific decisions

### Evaluation Metrics

The integration provides the same evaluation metrics as the original ReAct implementation:

- **Exact Match (EM)**: Exact string match after normalization
- **F1 Score**: Token-level F1 score between prediction and ground truth

### Configuration Options

You can customize the agent behavior:

```python
config = {
    "initial_search_query_count": 5,  # Number of initial search queries
    "max_research_loops": 3,         # Maximum research iterations
    "reasoning_model": "gpt-4o-mini"  # Model for final reasoning
}
```

## Output Format

The trace output is a detailed JSON object with **actual system prompts** and **real agent inputs/outputs**:

```json
{
  "question_idx": 0,
  "question": "What is the question?",
  "ground_truth": "Expected answer",
  "final_answer": "Agent's answer",
  "execution_time": 45.2,
  "evaluation": {
    "exact_match": true,
    "f1": 0.95,
    "prediction": "Agent's answer",
    "ground_truth": "Expected answer",
    "prediction_normalized": "agents answer",
    "ground_truth_normalized": "expected answer"
  },
  "agent_trace": [
    {
      "agent": "generate_query",
      "system_prompt": "Your goal is to generate sophisticated and diverse web search queries...",
      "input": {
        "messages": [{"type": "human", "content": "What is the question?"}],
        "initial_search_query_count": 3
      },
      "output": ["search query 1", "search query 2"]
    },
    {
      "agent": "web_research",
      "system_prompts": ["Conduct targeted Google Searches to gather the most recent..."],
      "input": {
        "search_queries": ["search query 1", "search query 2"],
        "tavily_search_results": "Raw search results from Tavily API"
      },
      "output": ["research result 1", "research result 2"]
    },
    {
      "agent": "finalize_answer",
      "system_prompt": "Generate a high-quality answer to the user's question...",
      "input": {
        "web_research_results": ["research result 1", "research result 2"],
        "sources_gathered": [...],
        "messages": [...]
      },
      "output": "yes"
    }
  ]
}
```

## Comparison with Original ReAct

This integration follows the same approach as the original `hotpotqa.ipynb` but uses the LangGraph agent instead of the custom ReAct implementation:

| Aspect | Original ReAct | LangGraph Integration |
|--------|----------------|---------------------|
| Agent Architecture | Custom ReAct loop | LangGraph nodes |
| Search | Wikipedia API | Tavily Search API |
| Reasoning | Manual prompt engineering | Structured LangGraph nodes |
| Trace Recording | Manual logging | **Detailed traces with system prompts** |
| Evaluation | Same metrics | Same metrics |
| Answer Format | Concise facts | Concise facts (optimized) |

## Troubleshooting

1. **Data File Not Found**: Make sure the HotPotQA data files are in the correct location (`../ReAct/data/`)

2. **API Key Errors**: Ensure both `OPENAI_API_KEY` and `TAVILY_API_KEY` are set

3. **Import Errors**: Make sure you're running from the correct directory and the backend src is in your Python path

4. **Memory Issues**: For large batches, consider processing questions in smaller chunks

5. **Answer Format Issues**: If answers are still too verbose, check that `hotpot_graph.py` is being used instead of the default graph

6. **Trace Analysis**: Use the detailed traces to analyze agent behavior and system prompt effectiveness

## License

This integration follows the same license as the main project. 