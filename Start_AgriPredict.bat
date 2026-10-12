@echo off
cd /d "C:\Users\Shreya.G.B\agriculture-ml-internship"
start "Tomato Disease API" cmd /k "cd /d C:\Users\Shreya.G.B\agriculture-ml-internship && .venv\Scripts\python.exe disease_detection\src\disease_api.py"
timeout /t 5 /nobreak >nul
start "Crop Yield Website" cmd /k "cd /d C:\Users\Shreya.G.B\agriculture-ml-internship && .venv\Scripts\python.exe server.py"
start "" http://127.0.0.1:8000