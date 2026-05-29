@echo off
echo ============================================================
echo   Compilation ForestDL_Installer.exe avec PyInstaller
echo   IMPORTANT : Executer dans CMD (pas PowerShell)
echo   Si vous etes dans PowerShell, tapez : cmd /c build_exe.bat
echo ============================================================
echo.

:: Verifier Python
echo [1/4] Verification de Python...
python --version >nul 2>&1
if %errorLevel% NEQ 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    echo Telechargez Python sur : https://www.python.org/downloads/
    echo IMPORTANT : Cochez "Add Python to PATH" lors de l'installation.
    pause
    exit /b 1
)
echo [OK] Python trouve :
python --version

:: Verifier/installer pip
echo.
echo [2/4] Mise a jour de pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip a jour

:: Installer PyInstaller
echo.
echo [3/4] Installation de PyInstaller...
pip install pyinstaller --quiet
if %errorLevel% NEQ 0 (
    echo [ERREUR] Impossible d'installer PyInstaller.
    echo Essayez manuellement : pip install pyinstaller
    pause
    exit /b 1
)
echo [OK] PyInstaller installe

:: Compiler l'installateur
echo.
echo [4/4] Compilation de ForestDL_Installer.exe...
echo (peut prendre 1-3 minutes, ne pas fermer)
echo.

set "ICON_PATH=..\icons\icon.png"
if not exist "%ICON_PATH%" set "ICON_PATH="

if defined ICON_PATH (
    pyinstaller --onefile --windowed --name "ForestDL_Installer" ^
        --icon "%ICON_PATH%" ^
        --add-data "%ICON_PATH%;." ^
        --hidden-import winreg ^
        --hidden-import tkinter ^
        --hidden-import tkinter.ttk ^
        --hidden-import tkinter.messagebox ^
        --hidden-import tkinter.filedialog ^
        --clean ForestDL_Installer.py
) else (
    pyinstaller --onefile --windowed --name "ForestDL_Installer" ^
        --hidden-import winreg ^
        --hidden-import tkinter ^
        --hidden-import tkinter.ttk ^
        --hidden-import tkinter.messagebox ^
        --hidden-import tkinter.filedialog ^
        --clean ForestDL_Installer.py
)

if %errorLevel% NEQ 0 (
    echo.
    echo [ERREUR] Compilation echouee.
    echo Verifiez les messages ci-dessus.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   [OK]  dist\ForestDL_Installer.exe cree avec succes !
echo ============================================================
echo.
echo   Ce fichier .exe est autonome (aucun Python requis).
echo   Vous pouvez le copier et l'executer sur n'importe quel PC.
echo ============================================================
echo.

explorer dist
pause
