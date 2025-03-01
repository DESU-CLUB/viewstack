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
from nodes import LlamaNode, StreamUpdate
import torch

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

# Initialize the LlamaNode globally
llama_node = LlamaNode(use_cpu_friendly_model=not torch.cuda.is_available())

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

def modify_mermaid_for_horizontal(mermaid_code):
    """Modify mermaid code to use LR (Left to Right) direction"""
    if "graph TD;" in mermaid_code:
        return mermaid_code.replace("graph TD;", "graph LR;")
    return mermaid_code

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
        
        # Define the stream callback function
        def stream_callback(update: StreamUpdate):
            # If it's a graph update and direction is specified, modify the mermaid diagram
            if update.type == "graph_update" and direction == 'LR':
                if 'mermaid_diagram' in update.data:
                    update.data['mermaid_diagram'] = modify_mermaid_for_horizontal(update.data['mermaid_diagram'])
            
            # Put the update in the queue
            update_queue.put(update)
        
        # Set the stream callback on the llama node
        llama_node.set_stream_callback(stream_callback)
        
        # Define a generator function for the streaming response
        def generate():
            # Start processing in a background thread
            def process_query():
                try:
                    # Process the query - this will trigger stream updates via the callback
                    llama_node.process_query(query)
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    logger.info(f"Starting server on port {port}, debug mode: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)