@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo   ForestDL - Installation automatique NodeODM (Windows)
echo ============================================================
echo.

:: Verifier les droits administrateur
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo [ERREUR] Ce script doit etre execute en tant qu'Administrateur.
    echo Clic droit sur le fichier, "Executer en tant qu'administrateur"
    pause
    exit /b 1
)

:: Verifier si Docker est installe
echo [1/5] Verification de Docker...
docker --version >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Docker n'est pas installe. Telechargement en cours...
    echo.
    echo Ouverture de la page de telechargement Docker Desktop...
    start https://www.docker.com/products/docker-desktop/
    echo.
    echo INSTRUCTIONS :
    echo  1. Telechargez et installez Docker Desktop
    echo  2. Redemarrez votre ordinateur
    echo  3. Relancez ce script
    echo.
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('docker --version') do echo    [OK] %%i
)

:: Verifier que Docker tourne
echo.
echo [2/5] Verification que Docker est en cours d'execution...
docker info >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Docker n'est pas demarre. Lancement de Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Attente du demarrage de Docker (30 secondes)...
    timeout /t 30 /nobreak >nul
    docker info >nul 2>&1
    if %errorLevel% NEQ 0 (
        echo [ERREUR] Docker ne repond pas. Verifiez que Docker Desktop est demarre.
        pause
        exit /b 1
    )
)
echo    [OK] Docker est en marche

:: Telecharger l'image NodeODM
echo.
echo [3/5] Telechargement de l'image NodeODM (peut prendre 2-5 minutes)...
docker pull opendronemap/nodeodm:latest
if %errorLevel% NEQ 0 (
    echo [ERREUR] Impossible de telecharger l'image. Verifiez votre connexion internet.
    pause
    exit /b 1
)
echo    [OK] Image NodeODM telechargee

:: Creer le dossier de donnees
echo.
echo [4/5] Creation du dossier de donnees...
set "DATA_DIR=%USERPROFILE%\ForestDL\nodeodm_data"
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
echo    [OK] Dossier cree : %DATA_DIR%

:: Creer les scripts de demarrage
echo.
echo [5/5] Creation des scripts de gestion...

set "START_SCRIPT=%USERPROFILE%\ForestDL\demarrer_nodeodm.bat"
(
    echo @echo off
    echo echo Demarrage de NodeODM...
    echo docker stop nodeodm_forestdl 2^>nul
    echo docker rm nodeodm_forestdl 2^>nul
    echo docker run -d --name nodeodm_forestdl ^
    echo   -p 3000:3000 ^
    echo   -v "%USERPROFILE%\ForestDL\nodeodm_data":/var/www/data ^
    echo   --restart unless-stopped ^
    echo   opendronemap/nodeodm:latest
    echo echo.
    echo echo [OK] NodeODM demarre sur http://localhost:3000
    echo pause
) > "%START_SCRIPT%"

set "STOP_SCRIPT=%USERPROFILE%\ForestDL\arreter_nodeodm.bat"
(
    echo @echo off
    echo echo Arret de NodeODM...
    echo docker stop nodeodm_forestdl
    echo docker rm nodeodm_forestdl
    echo echo [OK] NodeODM arrete.
    echo pause
) > "%STOP_SCRIPT%"

set "STATUS_SCRIPT=%USERPROFILE%\ForestDL\statut_nodeodm.bat"
(
    echo @echo off
    echo echo === Statut NodeODM ===
    echo docker ps --filter name=nodeodm_forestdl
    echo echo.
    echo echo Test API :
    echo curl -s http://localhost:3000/info
    echo echo.
    echo pause
) > "%STATUS_SCRIPT%"

echo    [OK] Scripts crees dans %USERPROFILE%\ForestDL\

:: Demarrer NodeODM maintenant
echo.
echo ============================================================
echo   Demarrage de NodeODM...
echo ============================================================
docker stop nodeodm_forestdl >nul 2>&1
docker rm nodeodm_forestdl >nul 2>&1
docker run -d --name nodeodm_forestdl ^
    -p 3000:3000 ^
    -v "%DATA_DIR%":/var/www/data ^
    --restart unless-stopped ^
    opendronemap/nodeodm:latest

if %errorLevel% NEQ 0 (
    echo [ERREUR] Impossible de demarrer NodeODM.
    pause
    exit /b 1
)

:: Attendre et tester
echo Attente du demarrage (20 secondes)...
timeout /t 20 /nobreak >nul

echo.
echo Test de connexion...
curl -s http://localhost:3000/info >nul 2>&1
if %errorLevel% EQU 0 (
    echo.
    echo ============================================================
    echo   [OK]  NodeODM est installe et demarre avec succes !
    echo ============================================================
    echo.
    echo   URL a utiliser dans ForestDL : http://localhost:3000
    echo   Token                        : (laisser vide)
    echo.
    echo   Scripts de gestion dans : %USERPROFILE%\ForestDL\
    echo     - demarrer_nodeodm.bat
    echo     - arreter_nodeodm.bat
    echo     - statut_nodeodm.bat
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo   [OK] NodeODM installe mais pas encore pret.
    echo ============================================================
    echo   Attendez 30 secondes puis ouvrez dans votre navigateur :
    echo   http://localhost:3000/info
    echo.
    echo   Si ca repond, tout fonctionne.
    echo   URL pour ForestDL : http://localhost:3000
    echo ============================================================
)

echo.
pause
