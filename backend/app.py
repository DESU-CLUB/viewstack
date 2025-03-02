from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import logging
import time
import queue
import threading
import torch
from nodes import LlamaNode, StreamUpdate, MermaidDiagramBuilder

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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Set up static folder for saved visualizations
os.makedirs('static', exist_ok=True)

# Initialize the global LlamaNode (used as a template)
global_llama_node = LlamaNode(use_cpu_friendly_model=not torch.cuda.is_available())

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": time.time()
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/api/search', methods=['POST'])
def search():
    """
    Endpoint to perform model search and analysis with streaming response
    
    Expected JSON payload:
    {
        "query": "Search query text",
        "max_iterations": 0,  # 0 for unlimited, or specify a number
        "direction": "TD" or "LR"  # TD for top-down, LR for left-right
    }
    """
    try:
        # Get data from request
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Missing required field: query"}), 400
        
        query = data.get('query')
        direction = data.get('direction', 'LR')  # Default to left-right
        
        logger.info(f"Starting search for query: '{query}', direction: {direction}")
        
        # Set up queue for stream updates
        update_queue = queue.Queue()
        
        # Create a new LlamaNode instance with the specified direction
        # This ensures we don't modify the global instance's diagram builder
        search_llama_node = LlamaNode(
            use_cpu_friendly_model=not torch.cuda.is_available(),
            stream_callback=None  # We'll set this below
        )
        
        # Initialize the diagram builder with the specified direction
        search_llama_node.diagram_builder = MermaidDiagramBuilder(direction=direction)
        
        # Define the stream callback function
        def stream_callback(update: StreamUpdate):
            # Put the update in the queue
            update_queue.put(update)
        
        # Set the stream callback on the llama node
        search_llama_node.set_stream_callback(stream_callback)
        
        # Define a generator function for the streaming response
        def generate():
            # Start processing in a background thread
            def process_query():
                try:
                    # Process the query - this will trigger stream updates via the callback
                    search_llama_node.process_query(query)
                    # Mark the end of the stream
                    update_queue.put(None)
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}", exc_info=True)
                    # Send error update
                    error_update = StreamUpdate(
                        type="error", 
                        data={"message": f"Error processing query: {str(e)}"}
                    )
                    update_queue.put(error_update)
                    update_queue.put(None)
            
            # Start processing thread
            threading.Thread(target=process_query).start()
            
            # Stream updates as they become available
            while True:
                update = update_queue.get()
                
                # None means end of stream
                if update is None:
                    break
                
                # Convert the update to JSON and yield
                yield json.dumps(update.__dict__) + "\n"
                
                # If this is a completion update, we're done
                if update.type == "complete":
                    break
        
        # Return streaming response
        return Response(
            stream_with_context(generate()),
            mimetype='application/x-ndjson'
        )
        
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/reasoning-steps', methods=['GET'])
def reasoning_steps():
    """
    Get the detailed reasoning steps for the sidebar
    """
    try:
        # Return predefined reasoning steps structure
        return jsonify({
            "status": "success",
            "steps": [
                {
                    "title": "Query Analysis",
                    "description": "Processing and extracting keywords from user query"
                },
                {
                    "title": "Model Search",
                    "description": "Searching HuggingFace for relevant models"
                },
                {
                    "title": "Model Analysis",
                    "description": "Analyzing model properties and determining relevance"
                },
                {
                    "title": "Recommendation",
                    "description": "Selecting the most appropriate model based on analysis"
                }
            ]
        })
    except Exception as e:
        logger.error(f"Error getting reasoning steps: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/save-visualization', methods=['POST'])
def save_visualization():
    """
    Save the search results and visualization as static files
    
    Expected JSON payload:
    {
        "query": "Search query text",
        "mermaid_diagram": "Mermaid diagram code",
        "recommendation": {...},  # Recommendation data
        "analysis": "Analysis text"
    }
    """
    try:
        data = request.json
        if not data or 'mermaid_diagram' not in data:
            return jsonify({"error": "Missing required data"}), 400
        
        # Generate unique file names based on timestamp
        timestamp = int(time.time())
        md_filename = f"search_result_{timestamp}.md"
        html_filename = f"search_result_{timestamp}.html"
        
        # Save Mermaid diagram
        md_path = os.path.join('static', md_filename)
        with open(md_path, "w") as f:
            f.write("```mermaid\n")
            f.write(data["mermaid_diagram"])
            f.write("\n```")
        
        # Create HTML visualization
        query = data.get('query', 'Unknown query')
        recommendation = data.get('recommendation', {})
        analysis = data.get('analysis', 'No analysis available')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Search Results</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{ 
                    startOnLoad: true,
                    flowchart: {{
                        useMaxWidth: false,
                        htmlLabels: true
                    }}
                }});
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
                /* Make diagram fit horizontally */
                .mermaid svg {{
                    width: 100% !important;
                    max-width: none !important;
                    height: auto !important;
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
{data['mermaid_diagram']}
                </div>
                
                <h2>Recommendation</h2>
                <div class="recommendation">
                    <p><strong>Model:</strong> {recommendation.get('model_name', 'N/A')}</p>
                    <p><strong>URL:</strong> <a href="{recommendation.get('model_url', '#')}" target="_blank">{recommendation.get('model_url', 'N/A')}</a></p>
                    <p><strong>Reason:</strong> {recommendation.get('reason', 'N/A')}</p>
                </div>
                
                <h2>Analysis</h2>
                <p>{analysis.replace(chr(10), '<br>')}</p>
                
                <p><small>Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </div>
        </body>
        </html>
        """
        
        html_path = os.path.join('static', html_filename)
        with open(html_path, "w") as f:
            f.write(html_content)
        
        # Return file URLs
        return jsonify({
            "status": "success",
            "visualization_urls": {
                "markdown": f"/static/{md_filename}",
                "html": f"/static/{html_filename}"
            }
        })
        
    except Exception as e:
        logger.error(f"Error saving visualization: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    logger.info(f"Starting server on port {port}, debug mode: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)