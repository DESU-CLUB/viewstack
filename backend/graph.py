import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict, cast, Annotated, Callable
import json
import datetime
from pydantic import BaseModel, Field
import torch
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import Messages
import operator
from nodes import LlamaNode, ModelRecommendation, StreamUpdate
from scrape import HuggingFaceScraper
import re

# Load environment variables
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()  # Try current directory

class GraphState(TypedDict):
    """State for the LangGraph pipeline"""
    query: str
    models: List[Dict[str, Any]]
    analysis: Optional[str]
    task_complete: bool
    graph_nodes: List[str]
    graph_edges: List[Dict[str, str]]
    current_iteration: int
    max_iterations: int
    recommendation: Optional[Dict[str, Any]]

def initialize_state(query: str, max_iterations: int = 3) -> GraphState:
    """Initialize the graph state with a user query"""
    return {
        "query": query,
        "models": [],
        "analysis": None,
        "task_complete": False,
        "graph_nodes": ["Start"],
        "graph_edges": [],
        "current_iteration": 0,
        "max_iterations": max_iterations,
        "recommendation": None
    }

def run_pipeline(query: str, max_iterations: int = 0, stream_callback: Optional[Callable[[StreamUpdate], None]] = None) -> Dict[str, Any]:
    """Run the model search pipeline using the streaming LlamaNode
    
    Args:
        query: The initial search query
        max_iterations: Maximum number of iterations (0 for unlimited)
        stream_callback: Optional callback for streaming updates
    
    Returns:
        Dictionary containing the results and mermaid diagram
    """
    # Initialize an instance of LlamaNode with the provided stream callback
    llama_node = LlamaNode(
        use_cpu_friendly_model=not torch.cuda.is_available(),
        stream_callback=stream_callback
    )
    
    # Process the query directly using the LlamaNode
    result = llama_node.process_query(query)
    
    # Generate Mermaid diagram if not provided in the result
    if "mermaid_diagram" not in result and result.get("status") == "success":
        mermaid_diagram = draw_mermaid(result)
        result["mermaid_diagram"] = mermaid_diagram
    
    return result

def draw_mermaid(result: Dict[str, Any]) -> str:
    """Generate a Mermaid diagram from the result data"""
    # Start with mermaid initialization for better styling
    mermaid_code = """%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
"""
    
    # Add start and analysis nodes
    mermaid_code += "    Start([Start]):::first\n"
    mermaid_code += "    Analysis([Analysis]):::analysis\n"
    mermaid_code += "    Start --> Analysis;\n"
    
    # Get the recommended model if available
    if result.get("recommendation") and "model_name" in result["recommendation"]:
        recommended_model = result["recommendation"]["model_name"]
        model_id = recommended_model.replace(" ", "_")
        mermaid_code += f"    {model_id}([{recommended_model}]):::recommended\n"
        mermaid_code += f"    Analysis -->|recommends| {model_id};\n"
    
    # Add other models
    if "raw_results" in result:
        for i, model in enumerate(result["raw_results"][:5]):  # Limit to 5 models
            model_name = model.get("name", f"Model {i+1}")
            if result.get("recommendation") and model_name == result["recommendation"].get("model_name"):
                continue  # Skip the recommended model as it's already added
                
            model_id = model_name.replace(" ", "_")
            mermaid_code += f"    {model_id}([{model_name}]):::model\n"
            mermaid_code += f"    Start -->|discovers| {model_id};\n"
    
    # Add styling classes
    mermaid_code += """    classDef first fill:#ffdfba,stroke:#ff9a00,color:black
    classDef analysis fill:#bae1ff,stroke:#0077ff,color:black
    classDef recommended fill:#baffc9,stroke:#00b050,color:black,stroke-width:2px
    classDef model fill:#f2f0ff,stroke:#9c88ff,color:black
"""
    
    return mermaid_code

def main():
    """Main function to run the pipeline"""
    print("🚀 Starting Model Search Pipeline")
    
    # Define a simple stream callback for testing
    def stream_callback(update: StreamUpdate):
        print(f"\n=== Stream Update [{update.type}] ===")
        if update.type == "reasoning_update":
            print(f"Title: {update.data['title']}")
            print(f"Content: {update.data['content']}")
        elif update.type == "graph_update":
            print("Graph updated")
        elif update.type == "complete":
            print("Processing complete")
        elif update.type == "error":
            print(f"Error: {update.data['message']}")
    
    # Example query
    query = "I need models for image classification of fire hazards"
    max_iterations = 0  # 0 means run until the model decides it's done
    
    print(f"Query: {query}")
    print(f"Max iterations: {'Unlimited' if max_iterations == 0 else max_iterations}")
    
    # Run the pipeline with streaming
    result = run_pipeline(query, max_iterations, stream_callback)
    
    # Print results
    print("\n=== PIPELINE RESULTS ===")
    
    # Print analysis if available
    if result.get("analysis"):
        print("\n=== ANALYSIS ===")
        print(result["analysis"])
    
    # Print recommendation if available
    if result.get("recommendation"):
        print("\n=== RECOMMENDATION ===")
        print(f"Model: {result['recommendation']['model_name']}")
        print(f"URL: {result['recommendation']['model_url']}")
        print(f"Reason: {result['recommendation']['reason']}")
    
    # Print Mermaid diagram
    if result.get("mermaid_diagram"):
        print("\n=== MERMAID DIAGRAM ===")
        print(result["mermaid_diagram"])
        
        # Save diagram to file
        with open("model_search_graph.md", "w") as f:
            f.write("```mermaid\n")
            f.write(result["mermaid_diagram"])
            f.write("\n```")
        
        print("\nMermaid diagram saved to model_search_graph.md")
    
    # Also save HTML version for easy viewing
    if result.get("mermaid_diagram"):
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Search Graph</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{ startOnLoad: true }});
            </script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }}
                .mermaid {{
                    display: flex;
                    justify-content: center;
                    margin: 30px 0;
                }}
                h1, h2 {{
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .recommendation {{
                    border: 1px solid #ddd;
                    padding: 15px;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Search Results</h1>
                
                <h2>Query</h2>
                <p>{query}</p>
                
                <h2>Visualization</h2>
                <div class="mermaid">
{result['mermaid_diagram']}
                </div>
                
                <h2>Recommendation</h2>
                <div class="recommendation">
                    <p><strong>Model:</strong> {result['recommendation']['model_name']}</p>
                    <p><strong>URL:</strong> <a href="{result['recommendation']['model_url']}" target="_blank">{result['recommendation']['model_url']}</a></p>
                    <p><strong>Reason:</strong> {result['recommendation']['reason']}</p>
                </div>
                
                <h2>Analysis</h2>
                <p>{result['analysis'].replace(chr(10), '<br>')}</p>
                
                <p><small>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </div>
        </body>
        </html>
        """
        
        with open("model_search_graph.html", "w") as f:
            f.write(html_content)
        
        print("HTML visualization saved to model_search_graph.html")

if __name__ == "__main__":
    main()