// src/components/SearchBar.tsx
import React from 'react';

// Define props interface
interface SearchBarProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  performSearch: () => Promise<void>;
  cancelSearch: () => void;
  loading: boolean;
}

const SearchBar: React.FC<SearchBarProps> = ({
  searchQuery,
  setSearchQuery,
  performSearch,
  cancelSearch,
  loading
}) => {
  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) {
      cancelSearch();
    } else {
      performSearch();
    }
  };

  return (
    <form className="prompt-container" onSubmit={handleSubmit}>
      <input 
        value={searchQuery}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
        type="text" 
        placeholder="What kind of model are you looking for?" 
        className="prompt-input"
        disabled={loading}
      />
      <button 
        type="submit"
        className={`send-button ${loading ? 'cancel-button' : ''}`}
        disabled={!loading && !searchQuery}
      >
        {loading ? 'Cancel' : 'Send'}
      </button>
    </form>
  );
};

export default SearchBar;