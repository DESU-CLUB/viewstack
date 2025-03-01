// src/App.tsx
import React, { useState, useEffect, useRef } from 'react';
import { streamSearchModels, SearchResults, StreamUpdate } from './services/api';
import SearchBar from './components/SearchBar';
import DiagramDisplay from './components/DiagramDisplay';
import Sidebar from './components/Sidebar';
import './App.css';

const App: React.FC = () => {
  // Application state
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [searchResults, setSearchResults] = useState<Partial<SearchResults> | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [currentMermaidDiagram, setCurrentMermaidDiagram] = useState<string>('');
  const [reasoningSteps, setReasoningSteps] = useState<Array<{ title: string; content: string }>>([]);
  
  // Reference to the abort controller function
  const abortStreamRef = useRef<(() => void) | null>(null);
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (abortStreamRef.current) {
        abortStreamRef.current();
      }
    };
  }, []);
  
  // Handle search form submission
  const performSearch = async (): Promise<void> => {
    if (!searchQuery || loading) return;
    
    // Cancel any existing stream
    if (abortStreamRef.current) {
      abortStreamRef.current();
      abortStreamRef.current = null;
    }
    
    setLoading(true);
    setError('');
    setSearchResults(null);
    setCurrentMermaidDiagram('');
    setReasoningSteps([]);
    
    // Automatically open sidebar when starting a new search
    setSidebarCollapsed(false);
    
    // Set up the update handler
    const handleStreamUpdate = (update: StreamUpdate) => {
      switch (update.type) {
        case 'graph_update':
          // Update the Mermaid diagram
          setCurrentMermaidDiagram(update.data.mermaid_diagram);
          
          // Update partial results if available
          if (update.data.partial_results) {
            setSearchResults(prevResults => ({
              ...prevResults,
              ...update.data.partial_results
            }));
          }
          break;
          
        case 'reasoning_update':
          // Add a new reasoning step or update an existing one
          setReasoningSteps(prevSteps => {
            const { title, content } = update.data;
            const stepIndex = prevSteps.findIndex(step => step.title === title);
            
            if (stepIndex >= 0) {
              // Update existing step
              const updatedSteps = [...prevSteps];
              updatedSteps[stepIndex] = { title, content };
              return updatedSteps;
            } else {
              // Add new step
              return [...prevSteps, { title, content }];
            }
          });
          break;
          
        case 'complete':
          // Final update with complete results
          setSearchResults(update.data);
          setLoading(false);
          abortStreamRef.current = null;
          break;
          
        case 'error':
          setError(update.data.message || 'An error occurred');
          setLoading(false);
          abortStreamRef.current = null;
          break;
      }
    };
    
    // Start the streaming search
    try {
      abortStreamRef.current = streamSearchModels(
        {
          query: searchQuery,
          max_iterations: 0,
          direction: 'LR'
        },
        handleStreamUpdate
      );
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred');
      }
      setLoading(false);
    }
  };
  
  // Toggle sidebar open/closed
  const toggleSidebar = (): void => {
    setSidebarCollapsed(!sidebarCollapsed);
  };
  
  // Cancel search if in progress
  const cancelSearch = (): void => {
    if (abortStreamRef.current) {
      abortStreamRef.current();
      abortStreamRef.current = null;
      setLoading(false);
      setError('Search cancelled');
    }
  };
  
  return (
    <div className="app-container">
      <main className="main-content">
        {/* Search Bar Component */}
        <SearchBar
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          performSearch={performSearch}
          cancelSearch={cancelSearch}
          loading={loading}
        />
        
        {/* Diagram Display Component */}
        <DiagramDisplay
          loading={loading}
          error={error}
          mermaidDiagram={currentMermaidDiagram}
        />
      </main>
      
      {/* Sidebar Component */}
      <Sidebar
        collapsed={sidebarCollapsed}
        toggleSidebar={toggleSidebar}
        searchResults={searchResults as SearchResults | null}
        reasoningSteps={reasoningSteps}
        loading={loading}
      />
    </div>
  );
};

export default App;