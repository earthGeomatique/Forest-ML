@echo off
chcp 65001 >nul
echo ============================================================
echo   Compilation ForestDL_Installer.exe avec PyInstaller
echo ============================================================
echo.

:: ── Vérifier Python ─────────────────────────────────────────────────────
python --version >nul 2>&1
if %errorLevel% NEQ 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    echo Telechargez Python sur : https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python :
python --version

:: ── Installer PyInstaller ────────────────────────────────────────────────
echo.
echo Installation de PyInstaller...
pip install pyinstaller --quiet
if %errorLevel% NEQ 0 (
    echo [ERREUR] Impossible d'installer PyInstaller.
    pause
    exit /b 1
)
echo [OK] PyInstaller installe

:: ── Compiler l'installateur ──────────────────────────────────────────────
echo.
echo Compilation de ForestDL_Installer.exe...
echo (peut prendre 1-2 minutes)
echo.

pyinstaller ^
    --onefile ^
    --windowed ^
    --name "ForestDL_Installer" ^
    --icon "..\icons\icon.png" ^
    --add-data "..\icons\icon.png;." ^
    --hidden-import winreg ^
    --hidden-import tkinter ^
    --hidden-import tkinter.ttk ^
    --hidden-import tkinter.messagebox ^
    --hidden-import tkinter.filedialog ^
    --clean ^
    ForestDL_Installer.py

if %errorLevel% NEQ 0 (
    echo.
    echo [ERREUR] Compilation echouee. Verifiez les messages ci-dessus.
    pause
    exit /b 1
)

:: ── Résultat ─────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   [OK]  ForestDL_Installer.exe cree avec succes !
echo ============================================================
echo.
echo   Fichier : dist\ForestDL_Installer.exe
echo.
echo   Vous pouvez distribuer ce .exe directement.
echo   Il ne necessite aucune installation Python.
echo ============================================================
echo.

:: Ouvrir le dossier dist
explorer dist
pause
