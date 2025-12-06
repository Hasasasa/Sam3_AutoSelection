@echo off
setlocal

REM One-click virtual env setup for SAM3 Auto Selection
REM Creates .\venv, activates it and installs project dependencies.

echo [setup] Using Python from PATH...
python --version || (
  echo [setup] ERROR: python not found in PATH.
  echo Please install Python 3.10+ and try again.
  goto :EOF
)

echo [setup] Creating virtual environment in .\venv ...
python -m venv venv
if errorlevel 1 (
  echo [setup] ERROR: Failed to create virtual environment.
  goto :EOF
)

echo [setup] Activating virtual environment ...
call venv\Scripts\activate.bat
if errorlevel 1 (
  echo [setup] ERROR: Failed to activate virtual environment.
  goto :EOF
)

echo [setup] Upgrading pip ...
python -m pip install --upgrade pip

echo [setup] Installing project dependencies from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
  echo [setup] ERROR: Dependency installation failed.
  goto :EOF
)

echo.
echo [setup] Done. Virtual environment is ready.
echo [setup] To activate it later, run:
echo     venv\Scripts\activate
echo.

endlocal

