"""
ForestDL Installer — Interface graphique complète
Installe : Docker, NodeODM, dépendances Python, plugin QGIS
Peut être compilé en .exe avec PyInstaller
"""

import os
import sys
import subprocess
import threading
import json
import urllib.request
import shutil
import platform
import winreg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

APP_TITLE   = "ForestDL Installer v2.0"
NODEODM_IMG = "opendronemap/nodeodm:latest"
NODEODM_PORT = 3000
CONTAINER_NAME = "nodeodm_forestdl"

DOCKER_DOWNLOAD_URL = (
    "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
)
DOCKER_INSTALLER_PATH = Path.home() / "Downloads" / "DockerDesktopInstaller.exe"

PYTHON_DEPS = [
    "requests>=2.28.0",
    "numpy>=1.21.0",
    "Pillow>=9.0.0",
    "rasterio>=1.3.0",
    "shapely>=2.0.0",
    "ultralytics>=8.0.0",
    "onnxruntime>=1.15.0",
    "pyproj>=3.4.0",
]

# Dossier données NodeODM
DATA_DIR = Path.home() / "ForestDL" / "nodeodm_data"

# Chemins plugin QGIS possibles
QGIS_PLUGIN_DIRS = [
    Path(os.environ.get("APPDATA", "")) / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins",
    Path.home() / "AppData" / "Roaming" / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins",
]


# ══════════════════════════════════════════════════════════════════════════
# Logique d'installation (thread séparé)
# ══════════════════════════════════════════════════════════════════════════

class Installer:
    def __init__(self, log_fn, progress_fn, step_fn):
        self.log      = log_fn
        self.progress = progress_fn
        self.set_step = step_fn
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run_all(self, install_docker, install_nodeodm,
                install_deps, install_plugin, plugin_src_dir):
        try:
            total = sum([install_docker, install_nodeodm,
                         install_deps, install_plugin])
            done  = 0

            if install_docker:
                self.set_step("Étape 1 — Docker")
                self._install_docker()
                done += 1
                self.progress(int(done / total * 100))

            if self._cancelled:
                return

            if install_nodeodm:
                self.set_step("Étape 2 — NodeODM")
                self._install_nodeodm()
                done += 1
                self.progress(int(done / total * 100))

            if self._cancelled:
                return

            if install_deps:
                self.set_step("Étape 3 — Dépendances Python")
                self._install_python_deps()
                done += 1
                self.progress(int(done / total * 100))

            if self._cancelled:
                return

            if install_plugin:
                self.set_step("Étape 4 — Plugin QGIS")
                self._install_qgis_plugin(plugin_src_dir)
                done += 1
                self.progress(int(done / total * 100))

            if not self._cancelled:
                self.set_step("✅  Installation terminée !")
                self.progress(100)
                self.log("\n🎉  Tout est installé avec succès !\n")
                self._print_summary()

        except Exception as exc:
            self.log(f"\n❌  ERREUR : {exc}\n")
            self.set_step("❌  Erreur — voir les logs")

    # ── Docker ──────────────────────────────────────────────────────────

    def _install_docker(self):
        self.log("=== Docker ===")
        if self._docker_available():
            self.log("✓ Docker déjà installé : " + self._run("docker --version"))
            if not self._docker_running():
                self.log("  Démarrage de Docker Desktop...")
                self._start_docker_desktop()
            return

        self.log("  Docker non trouvé. Téléchargement de Docker Desktop...")
        self._download_file(
            DOCKER_DOWNLOAD_URL,
            DOCKER_INSTALLER_PATH,
            "Docker Desktop Installer"
        )
        self.log("  Installation de Docker Desktop (fenêtre UAC requise)...")
        subprocess.run(
            [str(DOCKER_INSTALLER_PATH), "install", "--quiet",
             "--accept-license", "--backend=wsl-2"],
            check=True
        )
        self.log("  Démarrage de Docker Desktop...")
        self._start_docker_desktop()
        self.log("✓ Docker installé et démarré.")

    def _docker_available(self):
        try:
            subprocess.run(["docker", "--version"],
                           capture_output=True, timeout=10, check=True)
            return True
        except Exception:
            return False

    def _docker_running(self):
        try:
            subprocess.run(["docker", "info"],
                           capture_output=True, timeout=15, check=True)
            return True
        except Exception:
            return False

    def _start_docker_desktop(self):
        exe = Path("C:/Program Files/Docker/Docker/Docker Desktop.exe")
        if exe.exists():
            subprocess.Popen([str(exe)])
        # Attendre jusqu'à 60 secondes
        import time
        for i in range(12):
            time.sleep(5)
            self.log(f"  Attente Docker... ({(i+1)*5}s)")
            if self._docker_running():
                self.log("✓ Docker est en marche.")
                return
        raise RuntimeError(
            "Docker ne répond pas après 60 secondes.\n"
            "Ouvrez Docker Desktop manuellement puis relancez."
        )

    # ── NodeODM ─────────────────────────────────────────────────────────

    def _install_nodeodm(self):
        self.log("\n=== NodeODM ===")
        if not self._docker_running():
            self._start_docker_desktop()

        self.log("  Téléchargement image NodeODM (2-5 min)...")
        self._run_stream(["docker", "pull", NODEODM_IMG])

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.log(f"  Dossier données : {DATA_DIR}")

        # Arrêter l'ancien conteneur si existant
        subprocess.run(["docker", "stop", CONTAINER_NAME],
                       capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME],
                       capture_output=True)

        self.log("  Démarrage du conteneur NodeODM...")
        subprocess.run([
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", f"{NODEODM_PORT}:3000",
            "-v", f"{DATA_DIR}:/var/www/data",
            "--restart", "unless-stopped",
            NODEODM_IMG,
        ], check=True)

        # Tester la connexion
        import time
        self.log("  Attente du démarrage (20 secondes)...")
        time.sleep(20)
        try:
            with urllib.request.urlopen(
                f"http://localhost:{NODEODM_PORT}/info", timeout=10
            ) as resp:
                info = json.loads(resp.read())
                version = info.get("version", "?")
                self.log(f"✓ NodeODM v{version} opérationnel sur "
                         f"http://localhost:{NODEODM_PORT}")
        except Exception:
            self.log(f"⚠  NodeODM démarré mais pas encore prêt.")
            self.log(f"  Testez dans 30s : http://localhost:{NODEODM_PORT}/info")

        # Créer les scripts de gestion
        self._create_management_scripts()

    def _create_management_scripts(self):
        scripts_dir = Path.home() / "ForestDL"
        scripts_dir.mkdir(exist_ok=True)

        start = scripts_dir / "demarrer_nodeodm.bat"
        start.write_text(
            "@echo off\n"
            "echo Démarrage de NodeODM...\n"
            f"docker start {CONTAINER_NAME}\n"
            f"echo NodeODM sur http://localhost:{NODEODM_PORT}\n"
            "pause\n"
        )
        stop = scripts_dir / "arreter_nodeodm.bat"
        stop.write_text(
            "@echo off\n"
            "echo Arrêt de NodeODM...\n"
            f"docker stop {CONTAINER_NAME}\n"
            "echo Arrêté.\npause\n"
        )
        self.log(f"✓ Scripts créés dans {scripts_dir}")

    # ── Dépendances Python ───────────────────────────────────────────────

    def _install_python_deps(self):
        self.log("\n=== Dépendances Python ===")
        # Trouver Python de QGIS ou système
        python_exe = self._find_python()
        self.log(f"  Python : {python_exe}")

        # pip upgrade
        self._run_stream([python_exe, "-m", "pip", "install",
                          "--upgrade", "pip"])

        for dep in PYTHON_DEPS:
            if self._cancelled:
                return
            self.log(f"  Installation : {dep}")
            try:
                self._run_stream([python_exe, "-m", "pip", "install", dep])
                self.log(f"  ✓ {dep}")
            except Exception as e:
                self.log(f"  ⚠ Échec {dep}: {e}")

        self.log("✓ Dépendances installées.")

    def _find_python(self):
        # 1. Python de QGIS (le plus important)
        qgis_pythons = [
            r"C:\Program Files\QGIS 3.28\bin\python3.exe",
            r"C:\Program Files\QGIS 3.34\bin\python3.exe",
            r"C:\Program Files\QGIS 3.36\bin\python3.exe",
            r"C:\OSGeo4W\bin\python3.exe",
        ]
        for p in qgis_pythons:
            if Path(p).exists():
                return p

        # 2. Python système
        for exe in ["python3", "python"]:
            try:
                result = subprocess.run(
                    [exe, "--version"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return exe
            except Exception:
                continue

        # 3. Python dans PATH
        py = shutil.which("python3") or shutil.which("python")
        if py:
            return py

        raise RuntimeError(
            "Python non trouvé.\n"
            "Installez QGIS (il inclut Python) ou Python 3.x depuis python.org"
        )

    # ── Plugin QGIS ─────────────────────────────────────────────────────

    def _install_qgis_plugin(self, src_dir):
        self.log("\n=== Plugin QGIS ForestDL ===")
        if not src_dir or not Path(src_dir).exists():
            self.log("⚠  Dossier source du plugin non spécifié — étape ignorée.")
            return

        dest = None
        for candidate in QGIS_PLUGIN_DIRS:
            if candidate.parent.parent.exists():  # profil QGIS existe
                dest = candidate / "ForestDL"
                break

        if dest is None:
            # Chercher dans le registre Windows
            dest = self._find_qgis_plugin_dir()

        if dest is None:
            self.log("⚠  Dossier plugins QGIS introuvable.")
            self.log("   Copiez manuellement le dossier dans :")
            for d in QGIS_PLUGIN_DIRS:
                self.log(f"   {d / 'ForestDL'}")
            return

        self.log(f"  Destination : {dest}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src_dir, dest)
        self.log(f"✓ Plugin copié dans {dest}")
        self.log("  → Dans QGIS : Extensions → Gérer → ForestDL → Activer")

    def _find_qgis_plugin_dir(self):
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\QGIS\QGIS3\profiles\default"
            )
            profile_dir = winreg.QueryValue(key, "")
            if profile_dir:
                d = Path(profile_dir) / "python" / "plugins"
                d.mkdir(parents=True, exist_ok=True)
                return d / "ForestDL"
        except Exception:
            pass
        # Chercher dans AppData
        appdata = Path(os.environ.get("APPDATA", ""))
        qgis_dir = appdata / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"
        if qgis_dir.parent.parent.parent.parent.exists():
            qgis_dir.mkdir(parents=True, exist_ok=True)
            return qgis_dir / "ForestDL"
        return None

    # ── Utilitaires ──────────────────────────────────────────────────────

    def _run(self, cmd):
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()

    def _run_stream(self, cmd):
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace"
        )
        for line in process.stdout:
            line = line.rstrip()
            if line:
                self.log(f"    {line}")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Commande échouée: {' '.join(cmd)}")

    def _download_file(self, url, dest, name):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.log(f"  Téléchargement {name}...")

        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(count * block_size / total_size * 100))
                self.log(f"    {pct}%", end="\r")

        urllib.request.urlretrieve(url, dest, reporthook)
        self.log(f"✓ Téléchargement terminé : {dest}")

    def _print_summary(self):
        self.log("=" * 55)
        self.log("  RÉSUMÉ")
        self.log("=" * 55)
        self.log(f"  NodeODM URL : http://localhost:{NODEODM_PORT}")
        self.log("  Token       : (laisser vide dans ForestDL)")
        self.log("")
        self.log("  Dans QGIS :")
        self.log("  1. Extensions → Gérer et installer")
        self.log("  2. Onglet 'Installé' → ForestDL → Activer")
        self.log("  3. La barre d'outils ForestDL apparaît")
        self.log("=" * 55)


# ══════════════════════════════════════════════════════════════════════════
# Interface graphique
# ══════════════════════════════════════════════════════════════════════════

class InstallerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("720x620")
        self.root.resizable(True, True)
        self.root.configure(bg="#1a1a2e")

        try:
            self.root.iconbitmap("icon.ico")
        except Exception:
            pass

        self._installer = None
        self._thread    = None
        self._build_ui()

    # ── Construction de l'UI ─────────────────────────────────────────────

    def _build_ui(self):
        # ── En-tête ──────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#16213e", pady=12)
        header.pack(fill="x")

        tk.Label(
            header, text="🌿  ForestDL Installer",
            font=("Segoe UI", 20, "bold"),
            fg="#00d4aa", bg="#16213e"
        ).pack()
        tk.Label(
            header,
            text="Installation automatique : NodeODM · Python · Plugin QGIS",
            font=("Segoe UI", 10),
            fg="#a0aec0", bg="#16213e"
        ).pack()

        # ── Étapes à installer ────────────────────────────────────────────
        opts_frame = tk.LabelFrame(
            self.root, text="  Composants à installer  ",
            font=("Segoe UI", 10, "bold"),
            fg="#00d4aa", bg="#1a1a2e",
            bd=2, relief="groove"
        )
        opts_frame.pack(fill="x", padx=15, pady=(10, 5))

        self.var_docker  = tk.BooleanVar(value=True)
        self.var_nodeodm = tk.BooleanVar(value=True)
        self.var_deps    = tk.BooleanVar(value=True)
        self.var_plugin  = tk.BooleanVar(value=True)

        checkboxes = [
            (self.var_docker,  "🐋 Docker Desktop",
             "Moteur de virtualisation pour NodeODM"),
            (self.var_nodeodm, "🛩 NodeODM",
             "Serveur de traitement photogrammétrique (ODM)"),
            (self.var_deps,    "🐍 Dépendances Python",
             "rasterio, ultralytics (YOLO), onnxruntime, shapely…"),
            (self.var_plugin,  "🗺 Plugin QGIS ForestDL",
             "Copie le plugin dans le dossier QGIS"),
        ]

        for var, label, desc in checkboxes:
            row = tk.Frame(opts_frame, bg="#1a1a2e")
            row.pack(fill="x", padx=10, pady=3)
            tk.Checkbutton(
                row, variable=var, text=label,
                font=("Segoe UI", 10, "bold"),
                fg="white", bg="#1a1a2e",
                selectcolor="#16213e",
                activebackground="#1a1a2e",
                activeforeground="#00d4aa",
            ).pack(side="left")
            tk.Label(
                row, text=desc,
                font=("Segoe UI", 9), fg="#718096", bg="#1a1a2e"
            ).pack(side="left", padx=(5, 0))

        # ── Source plugin ─────────────────────────────────────────────────
        src_frame = tk.Frame(opts_frame, bg="#1a1a2e")
        src_frame.pack(fill="x", padx=10, pady=(5, 8))
        tk.Label(
            src_frame, text="Dossier plugin :",
            font=("Segoe UI", 9), fg="#a0aec0", bg="#1a1a2e"
        ).pack(side="left")

        self.plugin_src_var = tk.StringVar()
        # Auto-détecter si on est lancé depuis le dossier du plugin
        auto_src = Path(sys.executable).parent.parent
        if (auto_src / "__init__.py").exists():
            self.plugin_src_var.set(str(auto_src))
        tk.Entry(
            src_frame, textvariable=self.plugin_src_var,
            font=("Segoe UI", 9), bg="#16213e", fg="white",
            insertbackground="white", width=40
        ).pack(side="left", padx=5, fill="x", expand=True)
        tk.Button(
            src_frame, text="Parcourir",
            font=("Segoe UI", 9), bg="#2d3748", fg="white",
            activebackground="#4a5568",
            command=self._browse_plugin_src
        ).pack(side="left")

        # ── Barre de progression ──────────────────────────────────────────
        progress_frame = tk.Frame(self.root, bg="#1a1a2e")
        progress_frame.pack(fill="x", padx=15, pady=5)

        self.step_label = tk.Label(
            progress_frame, text="En attente…",
            font=("Segoe UI", 10, "bold"),
            fg="#00d4aa", bg="#1a1a2e", anchor="w"
        )
        self.step_label.pack(fill="x")

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "green.Horizontal.TProgressbar",
            troughcolor="#16213e", background="#00d4aa",
            thickness=20
        )
        self.progress_bar = ttk.Progressbar(
            progress_frame, style="green.Horizontal.TProgressbar",
            orient="horizontal", length=400, mode="determinate"
        )
        self.progress_bar.pack(fill="x", pady=5)

        self.pct_label = tk.Label(
            progress_frame, text="0%",
            font=("Segoe UI", 9), fg="#a0aec0", bg="#1a1a2e"
        )
        self.pct_label.pack()

        # ── Logs ─────────────────────────────────────────────────────────
        log_frame = tk.LabelFrame(
            self.root, text="  Journal d'installation  ",
            font=("Segoe UI", 9, "bold"),
            fg="#00d4aa", bg="#1a1a2e", bd=2, relief="groove"
        )
        log_frame.pack(fill="both", expand=True, padx=15, pady=(0, 5))

        self.log_text = tk.Text(
            log_frame,
            font=("Consolas", 9),
            bg="#0d1117", fg="#c9d1d9",
            insertbackground="white",
            state="disabled", wrap="word",
            height=12
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Boutons ───────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg="#1a1a2e")
        btn_frame.pack(fill="x", padx=15, pady=(0, 12))

        self.btn_install = tk.Button(
            btn_frame,
            text="▶  INSTALLER",
            font=("Segoe UI", 12, "bold"),
            bg="#00d4aa", fg="#1a1a2e",
            activebackground="#00b894",
            padx=20, pady=8,
            command=self._start_install
        )
        self.btn_install.pack(side="left")

        self.btn_cancel = tk.Button(
            btn_frame,
            text="■  Annuler",
            font=("Segoe UI", 10),
            bg="#e53e3e", fg="white",
            activebackground="#c53030",
            padx=10, pady=8,
            state="disabled",
            command=self._cancel
        )
        self.btn_cancel.pack(side="left", padx=8)

        tk.Button(
            btn_frame,
            text="🌐  Tester NodeODM",
            font=("Segoe UI", 10),
            bg="#2d3748", fg="#00d4aa",
            activebackground="#4a5568",
            padx=10, pady=8,
            command=self._test_nodeodm
        ).pack(side="right")

    def _browse_plugin_src(self):
        folder = filedialog.askdirectory(
            title="Sélectionner le dossier ForestDL (contient __init__.py)"
        )
        if folder:
            self.plugin_src_var.set(folder)

    # ── Démarrage de l'installation ───────────────────────────────────────

    def _start_install(self):
        self.btn_install.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self._log_clear()

        installer = Installer(
            log_fn=self._log,
            progress_fn=self._set_progress,
            step_fn=self._set_step,
        )
        self._installer = installer

        kwargs = dict(
            install_docker  = self.var_docker.get(),
            install_nodeodm = self.var_nodeodm.get(),
            install_deps    = self.var_deps.get(),
            install_plugin  = self.var_plugin.get(),
            plugin_src_dir  = self.plugin_src_var.get(),
        )

        self._thread = threading.Thread(
            target=installer.run_all, kwargs=kwargs, daemon=True
        )
        self._thread.start()
        self.root.after(300, self._check_thread)

    def _check_thread(self):
        if self._thread and self._thread.is_alive():
            self.root.after(300, self._check_thread)
        else:
            self.btn_install.config(state="normal")
            self.btn_cancel.config(state="disabled")

    def _cancel(self):
        if self._installer:
            self._installer.cancel()
        self._set_step("Annulation en cours…")

    # ── Test NodeODM ─────────────────────────────────────────────────────

    def _test_nodeodm(self):
        try:
            with urllib.request.urlopen(
                f"http://localhost:{NODEODM_PORT}/info", timeout=8
            ) as resp:
                info = json.loads(resp.read())
                v = info.get("version", "?")
                messagebox.showinfo(
                    "NodeODM OK ✓",
                    f"Serveur NodeODM connecté !\n\n"
                    f"Version : {v}\n"
                    f"URL ForestDL : http://localhost:{NODEODM_PORT}\n"
                    f"Token : (laisser vide)"
                )
        except Exception as e:
            messagebox.showerror(
                "NodeODM non accessible",
                f"Impossible de contacter NodeODM :\n{e}\n\n"
                f"Vérifiez que Docker est démarré."
            )

    # ── Utilitaires UI ────────────────────────────────────────────────────

    def _log(self, msg, end="\n"):
        def _do():
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg + end)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(0, _do)

    def _log_clear(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _set_progress(self, value):
        def _do():
            self.progress_bar["value"] = value
            self.pct_label.config(text=f"{value}%")
        self.root.after(0, _do)

    def _set_step(self, text):
        self.root.after(0, lambda: self.step_label.config(text=text))

    # ── Lancement ────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if platform.system() != "Windows":
        print("Cet installateur est conçu pour Windows.")
        print("Utilisez install_nodeodm_linux.sh ou install_nodeodm_mac.sh")
        sys.exit(1)

    app = InstallerGUI()
    app.run()
