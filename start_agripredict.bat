```bat
@echo off
cd /d "C:\Users\Shreya.G.B\agriculture-ml-internship\disease_detection"

start "AgriPredict API" /min cmd /k "cd /d C:\Users\Shreya.G.B\agriculture-ml-internship\disease_detection && venv\Scripts\python.exe src\disease_api.py"

timeout /t 10 /nobreak >nul

start "" "C:\Users\Shreya.G.B\agriculture-ml-internship\disease_detection\index.html"
```