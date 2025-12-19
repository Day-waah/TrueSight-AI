import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    handleFile(file);
  };

  const handleFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(file);
      setResult(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/predict-image', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: 'Failed to get prediction. Make sure the backend is running.' });
    } finally {
      setLoading(false);
    }
  };

  const getConfidencePercentage = () => {
    if (!result || result.error) return 0;
    return Math.round((1 - result.final_score) * 100);
  };

  return (
    <div className="App">
      <div className="masthead">
        <h1>The Deepfake Detector</h1>
        <p>Uncovering Digital Deception in the Age of AI</p>
      </div>

      <div className="main-content">
        <div className="left-column">
          <div className="section">
            <h2>Breaking News: Image Authentication</h2>
            <div className="upload-section">
              <div
                className={`upload-area ${dragOver ? 'dragover' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={handleClick}
              >
                <div className="upload-icon">📰</div>
                <div className="upload-text">
                  {selectedFile ? `Selected: ${selectedFile.name}` : 'Drop your image here or click to browse'}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="file-input"
                />
              </div>

              {preview && (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="preview" />
                </div>
              )}

              <div className="button-section">
                <button onClick={handleSubmit} disabled={!selectedFile || loading}>
                  {loading ? (
                    <>
                      Analyzing Image...
                      <div className="loading"></div>
                    </>
                  ) : (
                    'Verify Authenticity'
                  )}
                </button>
              </div>
            </div>
          </div>

          {result && (
            <div className="section">
              <h2>Investigation Results</h2>
              <div className="result-section">
                <div className={`result ${result.prediction === 'REAL' ? 'real' : 'fake'}`}>
                  {result.error ? (
                    <div>
                      <h3>Error in Investigation</h3>
                      <p>{result.error}</p>
                    </div>
                  ) : (
                    <div>
                      <h3>Headline: {result.prediction === 'REAL' ? 'Authentic Image Confirmed' : 'Deepfake Detected!'}</h3>
                      <p><strong>Confidence Level:</strong> {getConfidencePercentage()}%</p>
                      <div className="confidence-bar">
                        <div
                          className={`confidence-fill ${result.prediction === 'FAKE' ? 'fake' : ''}`}
                          style={{ width: `${getConfidencePercentage()}%` }}
                        ></div>
                      </div>
                      <p><em>This analysis is based on our prototype detection system.</em></p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="right-column">
          <div className="sidebar">
            <h3>About Our Investigation</h3>
            <p>Our advanced AI system uses cutting-edge machine learning techniques to detect manipulated images and videos. We combine convolutional neural networks with anomaly detection to provide reliable authenticity assessments.</p>

            <h3>How It Works</h3>
            <p>The system first detects faces in the image, then analyzes pixel patterns and reconstruction errors to determine if the content has been artificially generated or altered.</p>

            <div className="disclaimer">
              <p><strong>Disclaimer:</strong> This is a research prototype. Results should not be used for legal or professional purposes without further validation.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;