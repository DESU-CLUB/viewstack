// src/components/Sidebar.tsx
import React from 'react';
import { SearchResults } from '../services/api';

// Define reasoning step interface
interface ReasoningStep {
  title: string;
  content: string;
}

// Define props interface
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
  // Format analysis text into paragraphs
  const formatAnalysisText = (): string[] => {
    if (!searchResults?.analysis) return [];
    return searchResults.analysis.split('\n').filter(line => line.trim());
  };

  return (
    <aside className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''}`}>
      <div className="sidebar-toggle" onClick={toggleSidebar}>
        {collapsed ? '›' : '‹'}
      </div>
      
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