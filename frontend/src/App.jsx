import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e) => {
    setImage(e.target.files[0]);
  };

  const handlePredict = async () => {
    if (!image) {
      alert('Please select an image');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error:', error);
      alert(`Error uploading image: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>MNIST Digit Predictor</h1>
      <div className="container">
        <input 
          type="file" 
          onChange={handleImageUpload} 
          accept="image/*"
        />
        <button onClick={handlePredict} disabled={loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
        
        {prediction && (
          <div className="result">
            <h2>Prediction: {prediction.prediction}</h2>
            <p>Confidence: {prediction.confidence}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
