// src/services/api.ts
import { ReadableStream } from 'stream/web';

/**
 * Search query parameters
 */
export interface SearchParams {
  query: string;
  max_iterations?: number;
  direction?: 'TD' | 'LR';
}

/**
 * Model recommendation
 */
export interface ModelRecommendation {
  model_name: string;
  model_url: string;
  reason: string;
}

/**
 * Model details
 */
export interface ModelDetail {
  name: string;
  url: string;
  author: string;
  downloads: number;
  likes: number;
  description: string;
}

/**
 * Complete search results
 */
export interface SearchResults {
  status: string;
  query: string;
  models: ModelDetail[];
  analysis: string;
  recommendation: ModelRecommendation;
  mermaid_diagram: string;
  processing_time?: string;
}

/**
 * Stream update types
 */
export type StreamUpdate = 
  | { type: 'reasoning_update', data: { title: string, content: string } }
  | { type: 'graph_update', data: { mermaid_diagram: string, partial_results?: Partial<SearchResults> } }
  | { type: 'complete', data: SearchResults }
  | { type: 'error', data: { message: string } };

/**
 * API base URL from environment or default
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

/**
 * Stream search results from the API with real-time updates
 * 
 * @param params Search parameters
 * @param updateHandler Callback function to handle stream updates
 * @returns A function that can be called to abort the stream
 */
export function streamSearchModels(
  params: SearchParams,
  updateHandler: (update: StreamUpdate) => void
): () => void {
  // Create an AbortController to allow cancellation
  const controller = new AbortController();
  const { signal } = controller;

  // Start the streaming request
  (async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
        signal,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      // Process the stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          // Process any remaining data in the buffer
          if (buffer.trim()) {
            try {
              const update = JSON.parse(buffer.trim()) as StreamUpdate;
              updateHandler(update);
            } catch (e) {
              console.error('Error parsing final chunk:', e);
            }
          }
          break;
        }

        // Add new data to the buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Split by newlines, as each line should be a complete JSON object
        const lines = buffer.split('\n');
        
        // Process all complete lines except the last one (which might be incomplete)
        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i].trim();
          if (line) {
            try {
              const update = JSON.parse(line) as StreamUpdate;
              updateHandler(update);
            } catch (e) {
              console.error('Error parsing update:', e, 'Line:', line);
            }
          }
        }
        
        // Keep the last (potentially incomplete) line in the buffer
        buffer = lines[lines.length - 1];
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        // Request was aborted, this is expected when canceling
        console.log('Stream aborted by user');
      } else {
        // Unexpected error, notify handler
        console.error('Stream error:', error);
        updateHandler({
          type: 'error',
          data: { message: error instanceof Error ? error.message : 'Unknown error' }
        });
      }
    }
  })();

  // Return a function to abort the stream
  return () => controller.abort();
}

/**
 * Get reasoning steps information
 * @returns Promise with reasoning steps
 */
export async function getReasoningSteps(): Promise<{ title: string; description: string }[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/reasoning-steps`);
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    return data.steps;
  } catch (error) {
    console.error('Error fetching reasoning steps:', error);
    return [];
  }
}