#!/usr/bin/env bash
# ForestDL — Installation automatique NodeODM (Linux : Ubuntu/Debian/Fedora/Arch)
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERREUR]${NC} $*"; exit 1; }

echo -e "${BOLD}"
echo "============================================================"
echo "  ForestDL — Installation automatique NodeODM (Linux)"
echo "============================================================"
echo -e "${NC}"

# ── Détecter la distribution ────────────────────────────────────────────
if   [ -f /etc/debian_version ]; then DISTRO="debian"
elif [ -f /etc/fedora-release  ]; then DISTRO="fedora"
elif [ -f /etc/arch-release    ]; then DISTRO="arch"
else DISTRO="unknown"; fi
info "Distribution détectée : $DISTRO"

# ── Installer Docker si absent ──────────────────────────────────────────
info "[1/5] Vérification de Docker..."
if ! command -v docker &>/dev/null; then
    warn "Docker non trouvé. Installation en cours..."

    case $DISTRO in
        debian)
            sudo apt-get update -y
            sudo apt-get install -y ca-certificates curl gnupg lsb-release
            sudo install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
                | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            sudo chmod a+r /etc/apt/keyrings/docker.gpg
            echo "deb [arch=$(dpkg --print-architecture) \
                signed-by=/etc/apt/keyrings/docker.gpg] \
                https://download.docker.com/linux/ubuntu \
                $(lsb_release -cs) stable" \
                | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo apt-get update -y
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            ;;
        fedora)
            sudo dnf -y install dnf-plugins-core
            sudo dnf config-manager --add-repo \
                https://download.docker.com/linux/fedora/docker-ce.repo
            sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            ;;
        arch)
            sudo pacman -Sy --noconfirm docker docker-compose
            ;;
        *)
            error "Distribution non reconnue. Installez Docker manuellement : https://docs.docker.com/engine/install/"
            ;;
    esac

    # Ajouter l'utilisateur au groupe docker
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
    success "Docker installé"
    warn "Vous devrez peut-être vous déconnecter/reconnecter pour utiliser Docker sans sudo."
else
    success "Docker déjà installé : $(docker --version)"
fi

# ── Démarrer Docker si arrêté ───────────────────────────────────────────
info "[2/5] Vérification que Docker est en cours d'exécution..."
if ! docker info &>/dev/null; then
    sudo systemctl start docker
    sleep 3
    docker info &>/dev/null || error "Docker ne répond pas. Vérifiez avec : sudo systemctl status docker"
fi
success "Docker est en marche"

# ── Télécharger l'image NodeODM ─────────────────────────────────────────
info "[3/5] Téléchargement de l'image NodeODM (2-5 min selon connexion)..."
docker pull opendronemap/nodeodm:latest
success "Image NodeODM téléchargée"

# ── Créer le dossier de données ─────────────────────────────────────────
info "[4/5] Création des dossiers de données..."
DATA_DIR="$HOME/ForestDL/nodeodm_data"
mkdir -p "$DATA_DIR"
success "Dossier créé : $DATA_DIR"

# ── Créer les scripts de gestion ────────────────────────────────────────
info "[5/5] Création des scripts de gestion..."
SCRIPTS_DIR="$HOME/ForestDL"

# Script démarrer
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
echo ""
echo "NodeODM démarré sur http://localhost:3000"
SCRIPT
chmod +x "$SCRIPTS_DIR/demarrer_nodeodm.sh"

# Script arrêter
cat > "$SCRIPTS_DIR/arreter_nodeodm.sh" << 'SCRIPT'
#!/usr/bin/env bash
echo "Arrêt de NodeODM..."
docker stop nodeodm_forestdl
docker rm   nodeodm_forestdl
echo "NodeODM arrêté."
SCRIPT
chmod +x "$SCRIPTS_DIR/arreter_nodeodm.sh"

# Script statut
cat > "$SCRIPTS_DIR/statut_nodeodm.sh" << 'SCRIPT'
#!/usr/bin/env bash
echo "=== Statut NodeODM ==="
docker ps --filter name=nodeodm_forestdl --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "Test API :"
curl -s http://localhost:3000/info | python3 -m json.tool 2>/dev/null || echo "Serveur non accessible"
SCRIPT
chmod +x "$SCRIPTS_DIR/statut_nodeodm.sh"

# Service systemd (démarrage automatique au boot)
if systemctl --user is-active --quiet > /dev/null 2>&1 || true; then
    SYSTEMD_DIR="$HOME/.config/systemd/user"
    mkdir -p "$SYSTEMD_DIR"
    cat > "$SYSTEMD_DIR/nodeodm-forestdl.service" << UNIT
[Unit]
Description=NodeODM ForestDL
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/docker start nodeodm_forestdl
ExecStop=/usr/bin/docker stop nodeodm_forestdl

[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user enable nodeodm-forestdl.service 2>/dev/null || true
fi

success "Scripts créés dans $SCRIPTS_DIR"

# ── Démarrer NodeODM ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${NC}"
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

# ── Test de connexion ───────────────────────────────────────────────────
echo ""
if curl -sf http://localhost:3000/info > /tmp/nodeodm_info.json 2>/dev/null; then
    VERSION=$(python3 -c "import json,sys; d=json.load(open('/tmp/nodeodm_info.json')); print(d.get('version','?'))" 2>/dev/null || echo "?")
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
    echo "  Pour démarrer après redémarrage :"
    echo "    $SCRIPTS_DIR/demarrer_nodeodm.sh"
    echo ""
    echo "  Pour voir le statut :"
    echo "    $SCRIPTS_DIR/statut_nodeodm.sh"
    echo ""
    echo "  Pour arrêter :"
    echo "    $SCRIPTS_DIR/arreter_nodeodm.sh"
    echo "============================================================"
else
    warn "NodeODM démarre mais pas encore prêt."
    echo "  Attendez 30 secondes puis testez :"
    echo "  curl http://localhost:3000/info"
fi
