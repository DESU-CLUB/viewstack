// src/components/DiagramDisplay.tsx
import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

// Initialize mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  flowchart: {
    curve: 'linear',
    htmlLabels: true,
    useMaxWidth: false, // Important for horizontal layout
  },
  securityLevel: 'loose' // Needed for some diagram features
});

// Define props interface
interface DiagramDisplayProps {
  loading: boolean;
  error: string;
  mermaidDiagram: string;
}

const DiagramDisplay: React.FC<DiagramDisplayProps> = ({
  loading,
  error,
  mermaidDiagram
}) => {
  // Reference to mermaid diagram container
  const mermaidContainer = useRef<HTMLDivElement>(null);

  // Render mermaid diagram when it changes
  useEffect(() => {
    if (mermaidDiagram && mermaidContainer.current) {
      // Clear previous diagram
      mermaidContainer.current.innerHTML = '';
      
      // Create a div for mermaid
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = mermaidDiagram;
      mermaidContainer.current.appendChild(div);
      
      // Render the diagram
      try {
        mermaid.contentLoaded();
      } catch (err) {
        console.error('Error rendering mermaid diagram:', err);
        
        // Fallback to displaying the diagram as text
        const fallbackDiv = document.createElement('div');
        fallbackDiv.className = 'mermaid-fallback';
        fallbackDiv.textContent = 'Error rendering diagram. Please check the console for details.';
        mermaidContainer.current.appendChild(fallbackDiv);
      }
    }
  }, [mermaidDiagram]);

  return (
    <div className="diagram-container">
      {loading && !mermaidDiagram && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <p>Initiating search and analysis...</p>
        </div>
      )}
      
      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}
      
      <div 
        ref={mermaidContainer} 
        className={`mermaid-diagram ${loading && mermaidDiagram ? 'updating' : ''}`}
      >
        {/* Mermaid diagram will be rendered here */}
      </div>
      
      {loading && mermaidDiagram && (
        <div className="stream-indicator">
          <div className="dot-pulse"></div>
          <p>Receiving real-time updates...</p>
        </div>
      )}
    </div>
  );
};

export default DiagramDisplay;