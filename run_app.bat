@echo off
rem Runs the PDF Q&A app at http://localhost:8501 and restarts it if it ever stops.
rem Output goes to logs\streamlit.log. Stop it with stop_app.bat.
cd /d "%~dp0"
if not exist logs mkdir logs
rem Already running (e.g. started twice)? Then do nothing.
netstat -ano | findstr /r /c:"127.0.0.1:8501 .*LISTENING" /c:"\[::1\]:8501 .*LISTENING" >nul && exit /b 0
:loop
echo [%date% %time%] starting streamlit >> logs\streamlit.log
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.address localhost >> logs\streamlit.log 2>&1
echo [%date% %time%] streamlit exited (code %errorlevel%), restarting in 5s >> logs\streamlit.log
timeout /t 5 /nobreak >nul
goto loop
