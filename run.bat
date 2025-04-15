@echo off
setlocal enabledelayedexpansion

REM Check if at least one argument is provided
if "%~1"=="" (
    echo Error: No arguments provided.
    echo Usage: run_test.bat [python_file] [arg1] [arg2] ...
    exit /b 1
)

REM Store the first argument as the Python file to run
set PYTHON_FILE=%~1

REM Shift the arguments to remove the first one
shift

REM Build the command line arguments string
set ARGS=
:loop
if "%~1"=="" goto :endloop
set ARGS=!ARGS! %1
shift
goto :loop
:endloop

REM Run the Python file with the arguments
echo Running %PYTHON_FILE% with arguments:%ARGS%
python %PYTHON_FILE%%ARGS%

REM Check if the Python script executed successfully
if %ERRORLEVEL% neq 0 (
    echo Error: Python script failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Script completed successfully.
exit /b 0 