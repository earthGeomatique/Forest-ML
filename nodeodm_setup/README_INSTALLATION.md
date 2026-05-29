# Installation NodeODM — ForestDL

## Votre système d'exploitation ?

---

### Windows

1. Téléchargez le script : `install_nodeodm_windows.bat`
2. Clic droit → **"Exécuter en tant qu'administrateur"**
3. Le script installe Docker + NodeODM automatiquement
4. À la fin : `URL = http://localhost:3000`

---

### Linux (Ubuntu / Debian / Fedora / Arch)

```bash
# Dans un terminal :
chmod +x install_nodeodm_linux.sh
sudo ./install_nodeodm_linux.sh
```

---

### macOS

```bash
# Dans un terminal :
chmod +x install_nodeodm_mac.sh
./install_nodeodm_mac.sh
```

---

## Après installation

Dans ForestDL (onglet Traitement drone) :

| Champ | Valeur |
|-------|--------|
| URL serveur | `http://localhost:3000` |
| Token | *(laisser vide)* |

Cliquez **"Tester connexion"** pour vérifier.

---

## Commandes utiles

```bash
# Voir si NodeODM tourne
docker ps --filter name=nodeodm_forestdl

# Voir les logs
docker logs nodeodm_forestdl

# Arrêter
docker stop nodeodm_forestdl

# Redémarrer
docker start nodeodm_forestdl

# Tester l'API
curl http://localhost:3000/info
```

---

## Problèmes courants

| Problème | Solution |
|----------|----------|
| Port 3000 occupé | Changer en `-p 3001:3000` et mettre `http://localhost:3001` dans le plugin |
| Docker ne démarre pas | Vérifier que la virtualisation est activée dans le BIOS |
| "Permission denied" Linux | Exécuter avec `sudo` ou ajouter votre user au groupe `docker` |
| Mémoire insuffisante | NodeODM nécessite minimum 4 Go de RAM libre |
