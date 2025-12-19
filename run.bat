@echo off
echo Starting The Deepfake Detector...

REM Start backend in background
start "Backend" cmd /c "cd /d %~dp0backend && call ..\venv\Scripts\activate.bat && uvicorn app:app --reload --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend
start "Frontend" cmd /c "cd /d %~dp0frontend && npm start"

echo.
echo 🎉 The Deepfake Detector is now running!
echo 📱 Access the application at: http://localhost:3000
echo 🔗 Backend API available at: http://localhost:8000
echo.
echo Press any key to stop...
pause > nul