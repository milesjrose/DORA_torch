@echo off
echo Cleaning Python cache files...

:: Remove __pycache__ directories
for /d /r %%d in (__pycache__) do (
    if exist "%%d" (
        echo Deleting folder %%d
        rd /s /q "%%d"
    )
)

:: Remove .pyc and .pyo files
for /r %%f in (*.pyc *.pyo) do (
    echo Deleting file %%f
    del /f /q "%%f"
)

echo Done.
pause
