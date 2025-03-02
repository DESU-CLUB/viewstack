import React, { useState, useRef, useEffect, useCallback } from 'react';
import { SearchResults } from '../services/api';

interface ReasoningStep {
  title: string;
  content: string;
}

interface SidebarProps {
  collapsed: boolean;
  toggleSidebar: () => void;
  searchResults: SearchResults | null;
  reasoningSteps: ReasoningStep[];
  loading: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  toggleSidebar,
  searchResults,
  reasoningSteps,
  loading
}) => {
  const MIN_WIDTH = 200;  // You can adjust as needed
  const MAX_WIDTH = 600;  // You can adjust as needed

  // Track the width of the sidebar. Default to 300px (example).
  const [sidebarWidth, setSidebarWidth] = useState<number>(300);

  // Refs to control resizing
  const isResizingRef = useRef<boolean>(false);
  const lastDownXRef = useRef<number>(0);

  // Handle mousedown on the resize handle
  const onMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    isResizingRef.current = true;
    lastDownXRef.current = e.clientX;
  }, []);

  // Global mousemove event listener
  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizingRef.current) return;

      const offsetRight = lastDownXRef.current - e.clientX;
      let newWidth = sidebarWidth - offsetRight;

      // Enforce minimum and maximum widths
      if (newWidth < MIN_WIDTH) newWidth = MIN_WIDTH;
      if (newWidth > MAX_WIDTH) newWidth = MAX_WIDTH;

      setSidebarWidth(newWidth);
      lastDownXRef.current = e.clientX;
    },
    [sidebarWidth, MIN_WIDTH, MAX_WIDTH]
  );

  // Global mouseup event listener
  const onMouseUp = useCallback(() => {
    isResizingRef.current = false;
  }, []);

  // Attach/detach mouse event listeners
  useEffect(() => {
    // Attach
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);

    // Cleanup
    return () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  // Helper to format analysis text into paragraphs
  const formatAnalysisText = (): string[] => {
    if (!searchResults?.analysis) return [];
    return searchResults.analysis.split('\n').filter(line => line.trim());
  };

  return (
    <aside
      className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''}`}
      // Only apply a fixed inline width if not collapsed
      style={!collapsed ? { width: sidebarWidth } : {}}
    >
      {/* Toggle button */}
      <div className="sidebar-toggle" onClick={toggleSidebar}>
        {collapsed ? '›' : '‹'}
      </div>

      {/* The draggable handle (only show if not collapsed) */}
      {!collapsed && (
        <div 
          className="sidebar-resize-handle"
          onMouseDown={onMouseDown}
        />
      )}

      {/* Sidebar content (hidden or shown based on collapsed state) */}
      {!collapsed && (
        <div className="sidebar-content">
          <h2>Chain of Thought and Reasoning</h2>

          {(reasoningSteps.length > 0 || searchResults) ? (
            <div className="reasoning-chain">
              {/* Live reasoning steps */}
              {reasoningSteps.map((step, index) => (
                <div key={`step-${index}`} className="reasoning-step">
                  <h3>{step.title}</h3>
                  <div className="step-content">
                    {step.content}
                    {index === reasoningSteps.length - 1 && loading && (
                      <span className="typing-indicator">
                        <span className="dot"></span>
                        <span className="dot"></span>
                        <span className="dot"></span>
                      </span>
                    )}
                  </div>
                </div>
              ))}

              {/* Recommendation - shown only when available */}
              {searchResults?.recommendation && (
                <div className="reasoning-step">
                  <h3>Recommendation</h3>
                  <div className="recommendation">
                    <p className="model-name">{searchResults.recommendation.model_name}</p>
                    <a 
                      href={searchResults.recommendation.model_url} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="model-url"
                    >
                      {searchResults.recommendation.model_url}
                    </a>
                    <p className="reason">{searchResults.recommendation.reason}</p>
                  </div>
                </div>
              )}

              {/* Analysis section - shown only when available */}
              {searchResults?.analysis && (
                <div className="reasoning-step">
                  <h3>Analysis</h3>
                  <div className="analysis-text">
                    {formatAnalysisText().map((paragraph, idx) => (
                      <p key={idx}>{paragraph}</p>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state">
              <p>Enter a query to start searching for HuggingFace models</p>
            </div>
          )}
        </div>
      )}
    </aside>
  );
};

export default Sidebar;
