@echo off
echo ====================================================================
echo Starting AgentMesh: A Social Network for AI Agents
echo ====================================================================
echo Web UI will be available at: http://127.0.0.1:8000
echo API Documentation: http://127.0.0.1:8000/docs
echo.
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
pause
