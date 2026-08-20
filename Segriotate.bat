@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"
set "APP=%ROOT%\desktop_app.py"
set "ICO=%ROOT%\Segriotate.ico"

set "PY="
set "PYW="
if exist "%ROOT%\venv\Scripts\python.exe" (
  set "PY=%ROOT%\venv\Scripts\python.exe"
  set "PYW=%ROOT%\venv\Scripts\pythonw.exe"
) else if exist "%ROOT%\.venv\Scripts\python.exe" (
  set "PY=%ROOT%\.venv\Scripts\python.exe"
  set "PYW=%ROOT%\.venv\Scripts\pythonw.exe"
)

if not defined PY (
  powershell -NoProfile -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('Python venv not found. In Command Prompt run:\r\n\r\ncd /d \"%ROOT%\"\r\npython -m venv venv\r\nvenv\Scripts\activate\r\npip install -r requirements.txt','Segriotate',0,'Error')"
  exit /b 1
)

set "LAUNCH=%PY%"
if exist "%PYW%" set "LAUNCH=%PYW%"

if exist "%ROOT%\scripts\install_windows_shortcut.ps1" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\install_windows_shortcut.ps1" -Root "%ROOT%" -Python "%LAUNCH%" -App "%APP%" -Icon "%ICO%" >nul 2>&1
)

start "Segriotate" /D "%ROOT%" "%LAUNCH%" "%APP%"
exit /b 0
