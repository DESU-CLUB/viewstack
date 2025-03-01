// src/main.tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// TypeScript checks that getElementById returns a non-null value
const rootElement = document.getElementById('root')

// If root element is not found, throw an error
if (!rootElement) {
  throw new Error('Root element not found in the document')
}

// Create root and render app
ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)