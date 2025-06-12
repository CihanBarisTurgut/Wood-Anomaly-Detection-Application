import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import ModelPage from './pages/ModelPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/model/:modelId" element={<ModelPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;