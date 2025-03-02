import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Callable, Union
import json
from copy import deepcopy
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import torch
from scrape import HuggingFaceScraper
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()  # Try current directory

class SearchQuery(BaseModel):
    """Structure for search queries extracted from user prompts"""
    keyword: str
    max_pages: int = Field(default=1)

class ModelRecommendation(BaseModel):
    """Structure for model recommendations"""
    model_name: str
    model_url: str
    reason: str

class StreamUpdate(BaseModel):
    """Structure for streaming updates to the client"""
    type: str  # 'reasoning_update', 'graph_update', 'complete', 'error'
    data: Dict[str, Any]

class MermaidDiagramBuilder:
    """Helper class to manage incremental Mermaid diagram building"""
    
    def __init__(self, direction='TD'):
        self.direction = direction
        self.nodes = {}  # Dict of node_id -> node_data
        self.edges = []  # List of edge dictionaries
        self.current_diagram = None
        self.changed = True  # Flag to track if diagram needs updating
    
    def add_node(self, node_id, label, node_class=None):
        """Add a node to the diagram if it doesn't exist"""
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                'label': label,
                'class': node_class
            }
            self.changed = True
    
    def add_edge(self, from_id, to_id, label=None):
        """Add an edge between nodes if it doesn't exist"""
        edge = {'from': from_id, 'to': to_id, 'label': label}
        
        # Check if this edge already exists
        edge_exists = False
        for existing_edge in self.edges:
            if (existing_edge['from'] == from_id and 
                existing_edge['to'] == to_id and 
                existing_edge['label'] == label):
                edge_exists = True
                break
        
        if not edge_exists:
            self.edges.append(edge)
            self.changed = True
    
    def get_diagram(self, force_update=False):
        """Get the current Mermaid diagram, regenerating only if needed"""
        if self.changed or force_update or self.current_diagram is None:
            self.current_diagram = self._generate_diagram()
            self.changed = False
        
        return self.current_diagram
    
    def _generate_diagram(self):
        """Generate a Mermaid diagram from the current nodes and edges"""
        # Start with mermaid initialization for better styling
        diagram = f"""%%{{init: {{'flowchart': {{'curve': 'linear'}}}}}}%%
graph {self.direction};
"""
        
        # Add nodes
        for node_id, node_data in self.nodes.items():
            safe_id = self._make_safe_id(node_id)
            diagram += f"    {safe_id}([{node_data['label']}])"
            if node_data['class']:
                diagram += f":::{node_data['class']}"
            diagram += "\n"
        
        # Add edges
        for edge in self.edges:
            from_id = self._make_safe_id(edge['from'])
            to_id = self._make_safe_id(edge['to'])
            
            if edge['label']:
                diagram += f"    {from_id} -->|{edge['label']}| {to_id};\n"
            else:
                diagram += f"    {from_id} --> {to_id};\n"
        
        # Add styling classes
        diagram += """    classDef first fill:#ffdfba,stroke:#ff9a00,color:black
    classDef analysis fill:#bae1ff,stroke:#0077ff,color:black
    classDef recommended fill:#baffc9,stroke:#00b050,color:black,stroke-width:2px
    classDef model fill:#f2f0ff,stroke:#9c88ff,color:black
"""
        
        return diagram
    
    @staticmethod
    def _make_safe_id(node_id):
        """Convert node ID to safe Mermaid ID"""
        return node_id.replace(" ", "_").replace("-", "_")

class LlamaNode:
    """Node for processing user queries with Llama 3.2 and performing web searches with streaming updates"""
    
    def __init__(self, use_cpu_friendly_model=False, stream_callback: Optional[Callable[[StreamUpdate], None]] = None):
        # Default model
        self.model_id = "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8"
        
        # If CPU-friendly model is requested, use a smaller model
        if use_cpu_friendly_model:
            # Alternative smaller models that work better on CPU
            self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
            logger.info(f"Using CPU-friendly model: {self.model_id}")
        
        self.scraper = HuggingFaceScraper()
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        # Stream callback function
        self.stream_callback = stream_callback
        
        # Initialize the Mermaid diagram builder
        self.diagram_builder = MermaidDiagramBuilder(direction='TD')
        
        # Initialize the model
        self._initialize_model()
    
    def set_stream_callback(self, callback: Callable[[StreamUpdate], None]) -> None:
        """Set or update the stream callback function"""
        self.stream_callback = callback
    
    def _send_stream_update(self, update_type: str, data: Dict[str, Any]) -> None:
        """Send a streaming update if a callback is registered"""
        if self.stream_callback:
            update = StreamUpdate(type=update_type, data=data)
            self.stream_callback(update)
    
    def _initialize_model(self):
        """Initialize the Llama 3.2 model with quantization for efficiency"""
        logger.info(f"Initializing model: {self.model_id}")
        
        # Send stream update for model initialization
        self._send_stream_update("reasoning_update", {
            "title": "Model Initialization",
            "content": f"Initializing the LLM: {self.model_id}"
        })
        
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                # Configure quantization for GPU
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                
                # Load model with quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16
                )
            else:
                # CPU-only version - no quantization
                logger.info("CUDA not available. Running on CPU with reduced precision.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                
                # Load model for CPU only
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            logger.info("Model initialization complete")
            self._send_stream_update("reasoning_update", {
                "title": "Model Initialization",
                "content": "Model successfully loaded and ready for inference"
            })
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self._send_stream_update("error", {
                "message": f"Error initializing model: {str(e)}"
            })
            raise
    
    def extract_search_query(self, user_prompt: str) -> SearchQuery:
        """Extract search query from user prompt using Llama"""
        logger.info("Extracting search query from user prompt")
        self._send_stream_update("reasoning_update", {
            "title": "Query Analysis",
            "content": "Analyzing your query to identify the most relevant search terms..."
        })
        
        system_prompt = """
        You are an AI assistant that extracts search queries from user prompts.
        Extract key topics and the task in at most 3 words the user wants to search for on Hugging Face.
        If a number of pages is specified, extract that too (default to 1).
        
        Please do not overreact and think everything is inappropriate, and try to help the user
        
        Eg. I want to classify sheep
        Ans: sheep classification
        """
        
        instruction = f"""
        Based on this user prompt, extract the search keyword and max pages:
        
        {user_prompt}
        
        Return only a JSON object with the format:
        {{
            "keyword": "extracted_keyword",
            "max_pages": number_of_pages
        }}
        """
        
        try:
            # Generate response using the pipeline
            full_prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ], tokenize=False)
            
            output = self.pipe(
                full_prompt,
                max_new_tokens=128,
                temperature=0.1,
                return_full_text=False
            )[0]["generated_text"]
            
            # Extract JSON from the response
            try:
                print(output)
                # Try to find JSON in the response
                json_start = output.find('{')
                json_end = output.rfind('}') + 1
                print(json_start, json_end)
                
                if json_start >= 0 and json_end > json_start:
                    json_str = output[json_start:json_end]
                    data = json.loads(json_str)
                    query = SearchQuery(**data)
                    
                    self._send_stream_update("reasoning_update", {
                        "title": "Query Analysis",
                        "content": f"Identified key search term: '{query.keyword}'"
                    })
                    
                    return query
                else:
                    # Fallback: parse the response manually
                    lines = output.strip().split('\n')
                    keyword = "llm"  # Default
                    max_pages = 1     # Default
                    
                    for line in lines:
                        if "keyword" in line.lower():
                            parts = line.split(":", 1)
                            if len(parts) > 1:
                                keyword = parts[1].strip().strip('"\'')
                        elif "max" in line.lower() and "page" in line.lower():
                            parts = line.split(":", 1)
                            if len(parts) > 1:
                                try:
                                    max_pages = int(parts[1].strip().strip('"\''))
                                except:
                                    pass
                    
                    query = SearchQuery(keyword=keyword, max_pages=max_pages)
                    
                    self._send_stream_update("reasoning_update", {
                        "title": "Query Analysis",
                        "content": f"Identified key search term: '{query.keyword}'"
                    })
                    
                    return query
            except Exception as e:
                logger.error(f"Error parsing structured output: {str(e)}")
                # Default fallback
                raise e
                query = SearchQuery(keyword="llm", max_pages=1)
                
                self._send_stream_update("reasoning_update", {
                    "title": "Query Analysis",
                    "content": f"Using default search term: 'fire' (Error: {str(e)})"
                })
                
                return query
                
        except Exception as e:
            logger.error(f"Error extracting search query: {str(e)}")
            self._send_stream_update("error", {
                "message": f"Error extracting search query: {str(e)}"
            })
            raise
    
    def _update_diagram_with_models(self, model_details: List[Dict[str, Any]]):
        """Update the diagram with new models and send update if changed"""
        # Add Start node if not present
        self.diagram_builder.add_node("Start", "Start", "first")
        
        # Add models (limit to 5 to avoid cluttering)
        model_count = 0
        for model in model_details:
            if model_count >= 5:  # Limit to 5 models to avoid clutter
                break
                
            model_name = model.get("name", f"Model {model_count+1}")
            # Add model node
            self.diagram_builder.add_node(model_name, model_name, "model")
            # Add discovery edge
            self.diagram_builder.add_edge("Start", model_name, "discovers")
            model_count += 1
        
        # Check if diagram changed, and if so, send update
        mermaid_diagram = self.diagram_builder.get_diagram()
        self._send_stream_update("graph_update", {
            "mermaid_diagram": mermaid_diagram,
            "partial_results": {
                "models": model_details
            }
        })

    def _update_diagram_with_analysis(self):
        """Update the diagram with analysis node and send update if changed"""
        # Add Analysis node
        self.diagram_builder.add_node("Analysis", "Analysis", "analysis")
        # Connect Start to Analysis
        self.diagram_builder.add_edge("Start", "Analysis")
        
        # Check if diagram changed, and if so, send update
        mermaid_diagram = self.diagram_builder.get_diagram()
        self._send_stream_update("graph_update", {
            "mermaid_diagram": mermaid_diagram
        })

    def _update_diagram_with_recommendation(self, recommendation: ModelRecommendation):
        """Update the diagram with recommendation and send update if changed"""
        # Add the recommended model with special styling
        self.diagram_builder.add_node(
            recommendation.model_name, 
            recommendation.model_name, 
            "recommended"
        )
        
        # Add recommendation edge
        self.diagram_builder.add_edge(
            "Analysis", 
            recommendation.model_name, 
            "recommends"
        )
        
        # Check if diagram changed, and if so, send update
        mermaid_diagram = self.diagram_builder.get_diagram()
        self._send_stream_update("graph_update", {
            "mermaid_diagram": mermaid_diagram,
            "partial_results": {
                "recommendation": recommendation.__dict__
            }
        })
    
    def search_and_analyze(self, user_prompt: str) -> Dict[str, Any]:
        """
        Process user prompt, extract search query, perform search, and analyze results
        with incremental diagram updates
        """
        # Extract search query from user prompt
        search_query = self.extract_search_query(user_prompt)
        logger.info(f"Extracted search query: {search_query.keyword}, max pages: {search_query.max_pages}")
        
        # Send stream update for search
        self._send_stream_update("reasoning_update", {
            "title": "Model Search",
            "content": f"Searching for models with keyword: '{search_query.keyword}'"
        })
        
        # Perform search using the scraper
        try:
            model_urls = self.scraper.search_models(search_query.keyword, search_query.max_pages)
            
            if not model_urls:
                self._send_stream_update("reasoning_update", {
                    "title": "Model Search",
                    "content": f"No models found for keyword '{search_query.keyword}'"
                })
                
                return {
                    "status": "error",
                    "message": f"No models found for keyword '{search_query.keyword}'",
                    "search_query": search_query.__dict__,
                    "results": []
                }
            
            # Update with number of models found
            self._send_stream_update("reasoning_update", {
                "title": "Model Search",
                "content": f"Found {len(model_urls)} models for keyword '{search_query.keyword}'"
            })
            
            # Get details for each model
            self._send_stream_update("reasoning_update", {
                "title": "Model Analysis",
                "content": "Gathering details about each model..."
            })
            
            model_details = self.scraper.scrape_model_details(model_urls)
            
            # Update diagram with initial models and send stream update
            self._update_diagram_with_models(model_details)
            
            # Analyze results using Llama
            self._send_stream_update("reasoning_update", {
                "title": "Model Analysis",
                "content": "Analyzing model properties and suitability for your needs..."
            })
            
            # Update diagram with analysis node
            self._update_diagram_with_analysis()
            
            analysis = self._analyze_search_results(user_prompt, model_details)
            
            # Stream partial results with analysis
            self._send_stream_update("graph_update", {
                "mermaid_diagram": self.diagram_builder.get_diagram(),
                "partial_results": {
                    "analysis": analysis
                }
            })
            
            return {
                "status": "success",
                "search_query": search_query.__dict__,
                "raw_results": model_details,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error in search and analyze: {str(e)}")
            self._send_stream_update("error", {
                "message": f"Error searching for models: {str(e)}"
            })
            raise
    
    def _analyze_search_results(self, user_prompt: str, model_details: List[Dict[str, Any]]) -> str:
        """
        Analyze search results using Llama 3.2 to provide insights with streaming updates
        """
        logger.info("Analyzing search results")
        self._send_stream_update("reasoning_update", {
            "title": "Model Analysis",
            "content": "Performing in-depth analysis of model capabilities, downloads, and community metrics..."
        })
        
        system_prompt = """
        You are an AI assistant that analyzes Hugging Face model search results.
        Provide a concise summary of the search results and highlight the most relevant models.
        """
        
        # Format model details for the prompt
        models_text = json.dumps(model_details, indent=2)
        
        instruction = f"""
        User prompt: {user_prompt}
        
        Here are the search results from Hugging Face:
        {models_text}
        
        Please analyze these results and provide:
        1. A brief summary of what types of models were found
        2. Which models seem most relevant to the user's query
        3. Any notable characteristics (high downloads, likes, etc.)
        4. Any recommendations based on the search results
        """
        
        try:
            # Generate analysis
            outputs = self.pipe(
                self.tokenizer.apply_chat_template([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction}
                ], tokenize=False),
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True
            )
            
            # Extract the generated text
            analysis = outputs[0]["generated_text"]
            
            # Clean up the response to remove any template
            analysis = analysis.split("assistant")[-1].strip()
            if analysis.startswith(":"):
                analysis = analysis[1:].strip()
            
            # Update analysis step
            self._send_stream_update("reasoning_update", {
                "title": "Model Analysis",
                "content": "Analysis complete. Generating recommendation..."
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing search results: {str(e)}")
            self._send_stream_update("error", {
                "message": f"Error analyzing search results: {str(e)}"
            })
            raise
    
    def get_best_model_recommendation(self, user_prompt: str, model_details: List[Dict[str, Any]], analysis: str) -> ModelRecommendation:
        """
        Use Llama to recommend the best model from search results using structured output
        """
        logger.info("Generating model recommendation")
        self._send_stream_update("reasoning_update", {
            "title": "Recommendation",
            "content": "Evaluating models to determine the best match for your needs..."
        })
        
        system_prompt = """
        You are an AI assistant that analyzes Hugging Face model search results.
        Your task is to recommend the SINGLE BEST model based on the user's needs and search results.
        You must choose exactly one model and provide a clear reason for your choice.
        """
        
        # Format model details for the prompt
        models_text = json.dumps(model_details, indent=2)
        
        instruction = f"""
        User Prompt: {user_prompt}
        Search results from HuggingFace: {models_text}
        Analysis: {analysis}

        Based on the above information, recommend the single best model for the user's needs.
        Choose exactly one model from the search results.
        
        Return only a JSON object with the format:
        {{
            "model_name": "name of the best model",
            "model_url": "URL of the best model",
            "reason": "clear reason why this model is the best choice"
        }}
        """
        
        try:
            # Generate response using the pipeline
            full_prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ], tokenize=False)
            
            output = self.pipe(
                full_prompt,
                max_new_tokens=256,
                temperature=0.3,
                return_full_text=False
            )[0]["generated_text"]
            
            # Extract JSON from the response
            try:
                # Try to find JSON in the response
                json_start = output.find('{')
                json_end = output.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = output[json_start:json_end]
                    data = json.loads(json_str)
                    recommendation = ModelRecommendation(**data)
                    
                    # Update diagram with recommendation
                    self._update_diagram_with_recommendation(recommendation)
                    
                    # Update recommendation step
                    self._send_stream_update("reasoning_update", {
                        "title": "Recommendation",
                        "content": f"Recommended model: {recommendation.model_name}"
                    })
                    
                    return recommendation
                else:
                    # If no valid JSON found, try to extract structured information manually
                    if model_details and len(model_details) > 0:
                        # Use the first model as fallback
                        first_model = model_details[0]
                        recommendation = ModelRecommendation(
                            model_name=first_model.get("name", "Unknown"),
                            model_url=first_model.get("url", "https://huggingface.co"),
                            reason="This appears to be the most relevant model based on the search results."
                        )
                        
                        # Update diagram with recommendation
                        self._update_diagram_with_recommendation(recommendation)
                        
                        return recommendation
                    else:
                        # Default fallback if no models available
                        recommendation = ModelRecommendation(
                            model_name="No suitable model found",
                            model_url="https://huggingface.co",
                            reason="No models were found that match your query."
                        )
                        
                        self._send_stream_update("reasoning_update", {
                            "title": "Recommendation",
                            "content": "No suitable model found for your query."
                        })
                        
                        return recommendation
            except Exception as e:
                logger.error(f"Error parsing structured output for recommendation: {str(e)}")
                # Default fallback
                if model_details and len(model_details) > 0:
                    first_model = model_details[0]
                    recommendation = ModelRecommendation(
                        model_name=first_model.get("name", "Unknown"),
                        model_url=first_model.get("url", "https://huggingface.co"),
                        reason="This appears to be the most relevant model based on the search results."
                    )
                    
                    # Send error in reasoning step
                    self._send_stream_update("reasoning_update", {
                        "title": "Recommendation",
                        "content": f"Error in recommendation process: {str(e)}. Using fallback recommendation."
                    })
                    
                    return recommendation
                else:
                    recommendation = ModelRecommendation(
                        model_name="Error processing recommendation",
                        model_url="https://huggingface.co",
                        reason=f"An error occurred: {str(e)}"
                    )
                    
                    self._send_stream_update("reasoning_update", {
                        "title": "Recommendation",
                        "content": f"Error: {str(e)}"
                    })
                    
                    return recommendation
                    
        except Exception as e:
            logger.error(f"Error getting model recommendation: {str(e)}")
            self._send_stream_update("error", {
                "message": f"Error generating recommendation: {str(e)}"
            })
            raise
    
    def process_query(self, user_prompt: str) -> Dict[str, Any]:
        """
        Main method to process a user query - searches and provides AI analysis and recommendation
        with streaming updates
        """
        try:
            logger.info(f"Processing query: {user_prompt}")
            
            # Start time for performance tracking
            start_time = time.time()
            
            # Get search results and analysis
            result = self.search_and_analyze(user_prompt)
            
            # If search was successful, add a specific recommendation
            if result["status"] == "success" and result["raw_results"]:
                recommendation = self.get_best_model_recommendation(
                    user_prompt, 
                    result["raw_results"], 
                    result["analysis"]
                )
                result["recommendation"] = recommendation.__dict__
            
            # Add processing time
            processing_time = time.time() - start_time
            result["processing_time"] = f"{processing_time:.2f} seconds"
            
            # Add the final mermaid diagram
            result["mermaid_diagram"] = self.diagram_builder.get_diagram(force_update=True)
            
            # Send completion update
            self._send_stream_update("complete", result)
            
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"Error processing query: {str(e)}",
                "user_prompt": user_prompt
            }
            
            # Send error update
            self._send_stream_update("error", error_result)
            
            logger.error(f"Error in process_query: {str(e)}")
            return error_result

def main():
    """Test function for the node"""
    # Define a simple stream callback
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
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU-friendly model.")
        node = LlamaNode(use_cpu_friendly_model=True, stream_callback=stream_callback)
    else:
        node = LlamaNode(stream_callback=stream_callback)
    
    # Test prompt
    test_prompt = "I need models for image classification of fire hazards. Show me 2 pages of results."
    
    # Process the query
    result = node.process_query(test_prompt)
    
    # Print the result
    print(json.dumps(result, indent=2))
    
    # Specifically print the recommendation if available
    if result.get("status") == "success" and result.get("recommendation"):
        print("\n=== BEST MODEL RECOMMENDATION ===")
        print(f"Model: {result['recommendation']['model_name']}")
        print(f"URL: {result['recommendation']['model_url']}")
        print(f"Reason: {result['recommendation']['reason']}")

if __name__ == "__main__":
    main()