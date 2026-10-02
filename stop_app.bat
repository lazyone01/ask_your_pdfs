@echo off
rem Stops the app started by run_app.bat / the "PDF QA app" scheduled task.
rem Kills the restart loop first (so it can't relaunch), then Streamlit.
rem Matches on process name too, so this script's own PowerShell is never killed.
powershell -NoProfile -Command ^
  "$all = Get-CimInstance Win32_Process;" ^
  "$all | Where-Object { $_.Name -eq 'cmd.exe' -and $_.CommandLine -like '*run_app.bat*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue };" ^
  "$all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*streamlit run app.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue };" ^
  "Write-Host 'PDF Q&A app stopped.'"
