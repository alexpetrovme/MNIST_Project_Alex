import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImage(file);
    
    // Create preview
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRefresh = () => {
    setImage(null);
    setImagePreview(null);
    setPrediction(null);
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
        <label className="file-input-label">
          📁 Choose Image
          <input 
            type="file" 
            onChange={handleImageUpload} 
            accept="image/*"
          />
        </label>
        
        {imagePreview && (
          <div className="preview">
            <img src={imagePreview} alt="preview" />
          </div>
        )}
        
        <button onClick={handlePredict} disabled={loading || !image}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
        
        {prediction && (
          <div className="result">
            <h2>Prediction: {prediction.prediction}</h2>
            <p>Confidence: {prediction.confidence}</p>
            <button className="refresh-btn" onClick={handleRefresh}>
              🔄 Try Another
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
