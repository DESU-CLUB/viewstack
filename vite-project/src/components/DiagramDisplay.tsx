// DiagramDisplay.tsx
import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: false,
  theme: 'default',
  flowchart: {
    curve: 'linear',
    htmlLabels: true,
    useMaxWidth: true
  },
  securityLevel: 'loose'
});

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
  const mermaidContainer = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mermaidDiagram || !mermaidContainer.current) return;

    // Clear old content
    mermaidContainer.current.innerHTML = '';

    // Sanitize to remove weird characters
    const sanitizedDiagram = mermaidDiagram
      .normalize('NFD')
      .replace(/[^\x20-\x7E\n\r]+/g, '');

    const renderId = 'mermaid-diagram-' + Date.now();

    mermaid
      .render(renderId, sanitizedDiagram)
      .then(({ svg }) => {
        mermaidContainer.current!.innerHTML = svg;
      })
      .catch((err) => {
        console.error('Mermaid render error:', err);
        const fallbackDiv = document.createElement('div');
        fallbackDiv.className = 'mermaid-fallback';
        fallbackDiv.textContent = 'Error rendering diagram. See console for details.';
        mermaidContainer.current!.appendChild(fallbackDiv);
      });
  }, [mermaidDiagram]);

  return (
    <div className="diagram-container">
      {loading && !mermaidDiagram && (
        <div className="loading-indicator">
          <div className="spinner" />
          <p>Initiating search and analysis...</p>
        </div>
      )}

      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {/* WRAP the mermaid diagram container in .diag-scale-wrap */}
      <div className="diag-scale-wrap">
        <div
          ref={mermaidContainer}
          className={`mermaid-diagram ${loading && mermaidDiagram ? 'updating' : ''}`}
        />
      </div>

      {loading && mermaidDiagram && (
        <div className="stream-indicator">
          <div className="dot-pulse" />
          <p>Receiving real-time updates...</p>
        </div>
      )}
    </div>
  );
};

export default DiagramDisplay;
