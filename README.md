# The Deepfake Detector

A newspaper-style web application for detecting deepfake images using machine learning. This project demonstrates a full-stack implementation with React frontend, FastAPI backend, and PyTorch models, presented in a classic newspaper layout.

## Features

- Newspaper-inspired UI design
- Drag-and-drop image upload
- Real-time deepfake detection
- Confidence scoring with visual indicators
- Responsive layout for desktop and mobile
- RESTful API for predictions

## Architecture

### Frontend (React)
- Newspaper-style layout with masthead and columns
- File upload with drag-and-drop support
- Real-time preview and result display
- Responsive design

### Backend (FastAPI)
- REST API endpoints: `/predict-image`, `/predict-video`
- Secure file upload handling
- ML model inference pipeline

### Machine Learning Models
- **CNN Classifier**: MobileNetV2 fine-tuned for binary classification (real/fake)
- **Autoencoder**: For anomaly detection based on reconstruction error
- Combined scoring for final prediction

## Model Files

**Note:** Due to GitHub's file size limits, the trained model files are not included in this repository. To run the application, you need to download or train the models separately.

### Required Model Files:
- `models/ae.pt` - Autoencoder model (~300MB)
- `models/cnn_efficientnet_b0.pt` - CNN classifier (~50MB) 
- `models/cnn_mobilenetv2.pt` - Alternative CNN model (~50MB)

### Option 1: Download Pre-trained Models
If you have access to the pre-trained models, place them in the `models/` directory.

### Option 2: Train Your Own Models
1. Prepare training data in `data/` directory (cropped faces from real/fake images)
2. Run training scripts:
   ```bash
   python src/train_ae.py      # Train autoencoder
   python src/train_cnn.py     # Train CNN classifiers
   ```

### Option 3: Use Demo Mode
The application will show an error if models are missing. For demonstration purposes, you can modify `backend/app.py` to return mock predictions.

## Limitations

- Prototype model with limited accuracy
- Video detection not fully implemented (placeholder)
- Requires face detection in images
- Runs on CPU only

## Future Improvements

- Train models on larger datasets for better accuracy
- Implement full video frame analysis
- Add more advanced models (e.g., transformers)
- Deploy to cloud for better performance

## Quick Start

- Prototype model with limited accuracy
- Video detection not fully implemented (placeholder)
- Requires face detection in images
- Runs on CPU only

## Future Improvements

- Train models on larger datasets for better accuracy
- Implement full video frame analysis
- Add more advanced models (e.g., transformers)
- Deploy to cloud for better performance

## Quick Start

### Run the Complete Application
Double-click `run.bat` (Windows) or run `bash run.sh` (Linux/Mac) to start both frontend and backend.

**Single Access Link:** http://localhost:3000

The frontend automatically connects to the backend API through a proxy, so you only need to visit one URL.

### Manual Setup (if needed)

#### Backend Setup
```bash
# Activate virtual environment
source .venv/Scripts/activate

# Navigate to backend
cd backend

# Install dependencies (if not done)
pip install -r requirements.txt

# Run server
uvicorn app:app --reload
```

#### Frontend Setup
```bash
# Navigate to frontend
cd frontend

# Install dependencies (if not done)
npm install

# Run development server
npm start
```

## Usage
- Open http://localhost:3000 in your browser
- Drag and drop or click to upload an image (JPG/PNG)
- Click "Check Authenticity"
- View the prediction result with confidence score

## Model Training

To retrain the models:

1. Prepare your dataset in the `data/` directory
2. Run training scripts:
   ```bash
   python src/train_cnn.py
   python src/train_ae.py
   ```
3. Update model paths in `backend/app.py` if needed

## API Documentation

- `GET /`: Health check
- `POST /predict-image`: Upload image file, returns prediction JSON
- `POST /predict-video`: Placeholder for video prediction

## Technologies Used

- Frontend: React, JavaScript
- Backend: FastAPI, Python
- ML: PyTorch, Torchvision
- Face Detection: MTCNN (via facenet-pytorch)