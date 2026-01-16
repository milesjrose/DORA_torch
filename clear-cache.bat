@echo off

:: Remove __pycache__ directories
for /d /r %%d in (__pycache__) do (
    if exist "%%d" (
        REM echo Deleting folder %%d
        rd /s /q "%%d"
    )
)

:: Remove .pyc and .pyo files
for /r %%f in (*.pyc *.pyo) do (
    REM echo Deleting file %%f
    del /f /q "%%f"
)

