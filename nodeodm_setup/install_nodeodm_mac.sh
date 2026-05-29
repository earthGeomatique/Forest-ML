#!/usr/bin/env bash
# ForestDL — Installation automatique NodeODM (macOS)
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERREUR]${NC} $*"; exit 1; }

echo -e "${BOLD}"
echo "============================================================"
echo "  ForestDL — Installation automatique NodeODM (macOS)"
echo "============================================================"
echo -e "${NC}"

# ── Vérifier macOS ─────────────────────────────────────────────────────
MACOS_VER=$(sw_vers -productVersion)
info "macOS version : $MACOS_VER"

# ── Installer Homebrew si absent ────────────────────────────────────────
info "[1/5] Vérification de Homebrew..."
if ! command -v brew &>/dev/null; then
    warn "Homebrew non trouvé. Installation..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Ajouter brew au PATH selon architecture
    if [[ "$(uname -m)" == "arm64" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    fi
fi
success "Homebrew disponible : $(brew --version | head -1)"

# ── Installer Docker Desktop si absent ─────────────────────────────────
info "[2/5] Vérification de Docker..."
if ! command -v docker &>/dev/null; then
    warn "Docker non trouvé. Installation via Homebrew Cask..."
    brew install --cask docker
    open /Applications/Docker.app
    echo ""
    warn "Docker Desktop s'ouvre. Acceptez les conditions et attendez qu'il démarre."
    echo "Appuyez sur ENTRÉE une fois Docker démarré (icône baleine dans la barre de menu)..."
    read -r
fi
success "Docker installé : $(docker --version)"

# ── Démarrer Docker si nécessaire ──────────────────────────────────────
info "[3/5] Vérification que Docker est en marche..."
if ! docker info &>/dev/null; then
    open /Applications/Docker.app
    info "Attente du démarrage de Docker (30 secondes)..."
    sleep 30
    docker info &>/dev/null || error "Docker ne répond pas. Ouvrez Docker Desktop manuellement."
fi
success "Docker est en marche"

# ── Télécharger NodeODM ─────────────────────────────────────────────────
info "[4/5] Téléchargement image NodeODM..."
docker pull opendronemap/nodeodm:latest
success "Image NodeODM téléchargée"

# ── Créer les scripts et dossiers ───────────────────────────────────────
info "[5/5] Création des dossiers et scripts..."
DATA_DIR="$HOME/ForestDL/nodeodm_data"
SCRIPTS_DIR="$HOME/ForestDL"
mkdir -p "$DATA_DIR"

cat > "$SCRIPTS_DIR/demarrer_nodeodm.sh" << 'SCRIPT'
#!/usr/bin/env bash
echo "Démarrage de NodeODM..."
docker stop nodeodm_forestdl 2>/dev/null || true
docker rm   nodeodm_forestdl 2>/dev/null || true
docker run -d --name nodeodm_forestdl \
    -p 3000:3000 \
    -v "$HOME/ForestDL/nodeodm_data":/var/www/data \
    --restart unless-stopped \
    opendronemap/nodeodm:latest
echo "NodeODM démarré : http://localhost:3000"
SCRIPT
chmod +x "$SCRIPTS_DIR/demarrer_nodeodm.sh"

cat > "$SCRIPTS_DIR/arreter_nodeodm.sh" << 'SCRIPT'
#!/usr/bin/env bash
docker stop nodeodm_forestdl && docker rm nodeodm_forestdl
echo "NodeODM arrêté."
SCRIPT
chmod +x "$SCRIPTS_DIR/arreter_nodeodm.sh"

# LaunchAgent pour démarrage automatique
PLIST_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$PLIST_DIR"
cat > "$PLIST_DIR/com.forestdl.nodeodm.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.forestdl.nodeodm</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/docker</string>
        <string>start</string>
        <string>nodeodm_forestdl</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$HOME/ForestDL/nodeodm.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/ForestDL/nodeodm_error.log</string>
</dict>
</plist>
PLIST
launchctl load "$PLIST_DIR/com.forestdl.nodeodm.plist" 2>/dev/null || true
success "LaunchAgent créé (démarrage automatique au login)"

# ── Démarrer NodeODM ────────────────────────────────────────────────────
echo ""
info "Démarrage de NodeODM..."
docker stop nodeodm_forestdl 2>/dev/null || true
docker rm   nodeodm_forestdl 2>/dev/null || true
docker run -d --name nodeodm_forestdl \
    -p 3000:3000 \
    -v "$DATA_DIR":/var/www/data \
    --restart unless-stopped \
    opendronemap/nodeodm:latest

info "Attente du démarrage (15 secondes)..."
sleep 15

if curl -sf http://localhost:3000/info > /tmp/nodeodm_info.json 2>/dev/null; then
    VERSION=$(python3 -c "import json; d=json.load(open('/tmp/nodeodm_info.json')); print(d.get('version','?'))" 2>/dev/null || echo "?")
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "============================================================"
    echo "  ✓  NodeODM installé et démarré avec succès !"
    echo "============================================================"
    echo -e "${NC}"
    echo -e "  Version          : ${BOLD}$VERSION${NC}"
    echo -e "  URL pour ForestDL: ${BOLD}http://localhost:3000${NC}"
    echo -e "  Token            : ${BOLD}(laisser vide)${NC}"
    echo ""
    echo "  Scripts dans : $SCRIPTS_DIR"
    echo "  Démarrage auto au login : activé"
    echo "============================================================"
else
    warn "Vérifiez dans 30s : curl http://localhost:3000/info"
fi
