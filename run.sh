#!/bin/bash

echo "Starting The Deepfake Detector..."

# Function to cleanup background processes
cleanup() {
    echo "Stopping servers..."
    kill $backend_pid $frontend_pid 2>/dev/null
    exit
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Start backend in background
cd backend
source ../venv/Scripts/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
backend_pid=$!

echo "Backend started on port 8000"

# Wait a moment for backend to start
sleep 3

# Start frontend in background
cd ../frontend
npm start &
frontend_pid=$!

echo "Frontend started on port 3000"
echo ""
echo "🎉 The Deepfake Detector is now running!"
echo "📱 Access the application at: http://localhost:3000"
echo "🔗 Backend API available at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait