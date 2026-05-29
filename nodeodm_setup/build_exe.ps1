# ForestDL - Compilation ForestDL_Installer.exe
# Executer dans PowerShell : .\build_exe.ps1
# Si erreur politique, tapez d'abord : Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Compilation ForestDL_Installer.exe avec PyInstaller" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Se placer dans le dossier du script
Set-Location $PSScriptRoot

# [1/4] Verifier Python
Write-Host "[1/4] Verification de Python..." -ForegroundColor Yellow
try {
    $pyver = python --version 2>&1
    Write-Host "  [OK] $pyver" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] Python non trouve." -ForegroundColor Red
    Write-Host "Telechargez Python sur : https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "IMPORTANT : Cochez 'Add Python to PATH' lors de l'installation." -ForegroundColor Yellow
    Read-Host "Appuyez sur Entree pour quitter"
    exit 1
}

# [2/4] Mettre a jour pip
Write-Host ""
Write-Host "[2/4] Mise a jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "  [OK] pip a jour" -ForegroundColor Green

# [3/4] Installer PyInstaller
Write-Host ""
Write-Host "[3/4] Installation de PyInstaller..." -ForegroundColor Yellow
pip install pyinstaller --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERREUR] Echec installation PyInstaller." -ForegroundColor Red
    Read-Host "Appuyez sur Entree pour quitter"
    exit 1
}
Write-Host "  [OK] PyInstaller installe" -ForegroundColor Green

# [4/4] Compiler
Write-Host ""
Write-Host "[4/4] Compilation en cours (1-3 minutes)..." -ForegroundColor Yellow
Write-Host ""

$iconPath = "..\icons\icon.png"
$args_list = @(
    "--onefile",
    "--windowed",
    "--name", "ForestDL_Installer",
    "--hidden-import", "winreg",
    "--hidden-import", "tkinter",
    "--hidden-import", "tkinter.ttk",
    "--hidden-import", "tkinter.messagebox",
    "--hidden-import", "tkinter.filedialog",
    "--clean",
    "ForestDL_Installer.py"
)

if (Test-Path $iconPath) {
    $args_list = @("--icon", $iconPath, "--add-data", "${iconPath};.") + $args_list
}

& pyinstaller @args_list

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERREUR] Compilation echouee." -ForegroundColor Red
    Read-Host "Appuyez sur Entree pour quitter"
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  [OK]  dist\ForestDL_Installer.exe cree avec succes !" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Ce fichier .exe est autonome (aucun Python requis)." -ForegroundColor White
Write-Host "  Copiez-le et executez-le sur n'importe quel PC Windows." -ForegroundColor White
Write-Host ""

# Ouvrir le dossier dist
if (Test-Path "dist") {
    Start-Process explorer "dist"
}

Read-Host "Appuyez sur Entree pour quitter"
