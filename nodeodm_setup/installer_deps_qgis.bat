@echo off
echo ============================================================
echo   Installation des dependances ForestDL dans QGIS Python
echo ============================================================
echo.

:: Chercher Python de QGIS dans les chemins habituels
set "QGIS_PYTHON="

for %%V in (3.36 3.34 3.32 3.30 3.28 3.26 3.22) do (
    if exist "C:\Program Files\QGIS %%V\bin\python3.exe" (
        set "QGIS_PYTHON=C:\Program Files\QGIS %%V\bin\python3.exe"
        echo [OK] QGIS %%V trouve : !QGIS_PYTHON!
        goto :found
    )
)

:: Chercher OSGeo4W
if exist "C:\OSGeo4W\bin\python3.exe" (
    set "QGIS_PYTHON=C:\OSGeo4W\bin\python3.exe"
    echo [OK] OSGeo4W Python trouve : %QGIS_PYTHON%
    goto :found
)

:: Chercher dans Program Files avec wildcard
for /d %%D in ("C:\Program Files\QGIS*") do (
    if exist "%%D\bin\python3.exe" (
        set "QGIS_PYTHON=%%D\bin\python3.exe"
        echo [OK] QGIS trouve : !QGIS_PYTHON!
        goto :found
    )
)

echo [ERREUR] Python QGIS non trouve automatiquement.
echo.
echo Solutions :
echo  1. Ouvrez "OSGeo4W Shell" depuis le menu Demarrer
echo  2. Tapez : pip install ultralytics Pillow shapely
echo.
set /p QGIS_PYTHON="Entrez le chemin complet vers python3.exe de QGIS : "
if not exist "%QGIS_PYTHON%" (
    echo [ERREUR] Fichier introuvable. Abandon.
    pause
    exit /b 1
)

:found
echo.
echo Python QGIS : %QGIS_PYTHON%
echo.

:: Mettre a jour pip
echo [1/4] Mise a jour de pip...
"%QGIS_PYTHON%" -m pip install --upgrade pip --quiet
echo    [OK] pip a jour

:: Installer ultralytics (YOLO)
echo.
echo [2/4] Installation de ultralytics (YOLOv8)...
echo    (peut prendre 2-5 minutes selon votre connexion)
"%QGIS_PYTHON%" -m pip install ultralytics --quiet
if %errorLevel% NEQ 0 (
    echo [ERREUR] Echec ultralytics. Essayez dans OSGeo4W Shell :
    echo    pip install ultralytics
) else (
    echo    [OK] ultralytics installe
)

:: Installer Pillow
echo.
echo [3/4] Installation de Pillow...
"%QGIS_PYTHON%" -m pip install Pillow --quiet
if %errorLevel% NEQ 0 (
    echo    [AVERT] Echec Pillow (generalement deja present)
) else (
    echo    [OK] Pillow installe
)

:: Installer shapely (NMS geographique)
echo.
echo [4/4] Installation de shapely...
"%QGIS_PYTHON%" -m pip install shapely --quiet
if %errorLevel% NEQ 0 (
    echo    [AVERT] Echec shapely (NMS basique sera utilise)
) else (
    echo    [OK] shapely installe
)

:: Verification finale
echo.
echo ============================================================
echo   Verification des installations
echo ============================================================
"%QGIS_PYTHON%" -c "import ultralytics; print('[OK] ultralytics', ultralytics.__version__)"
"%QGIS_PYTHON%" -c "from PIL import Image; print('[OK] Pillow OK')"
"%QGIS_PYTHON%" -c "from osgeo import gdal; print('[OK] GDAL', gdal.__version__)"
"%QGIS_PYTHON%" -c "import numpy; print('[OK] numpy', numpy.__version__)"

echo.
echo ============================================================
echo   Installation terminee !
echo   Relancez QGIS et utilisez ForestDL.
echo ============================================================
echo.
pause
