@echo off
echo ============================================
echo   VideoToASCII — Build Standalone .exe
echo ============================================
echo.
echo [1/2] Installing dependencies...
pip install -r requirements.txt
echo.
echo [2/2] Building .exe with PyInstaller...
if exist icon.ico (
    pyinstaller --onefile --windowed --name VideoToASCII --icon=icon.ico main.py
) else (
    echo No icon.ico found — building without icon.
    echo Run: python generate_icon.py   to create one first.
    pyinstaller --onefile --windowed --name VideoToASCII main.py
)
echo.
echo Build complete! Check the dist\ folder.
pause
