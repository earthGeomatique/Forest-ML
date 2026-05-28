"""
ODMProcessor — communicates with a NodeODM server via its REST API.

No PyODM dependency: uses the requests library directly.

NodeODM REST API used:
  GET  /info                          → server info
  POST /task/new/init                 → create task, returns {uuid}
  POST /task/new/upload/{uuid}        → upload image files (multipart/form-data)
  POST /task/new/commit/{uuid}        → start processing
  GET  /task/{uuid}/info              → task status {status:{code}, progress}
  GET  /task/{uuid}/download/orthophoto.tif  → download result

Status codes:
  10 = QUEUED
  20 = RUNNING
  30 = FAILED
  40 = COMPLETED
  50 = CANCELLED
"""

import os
import time
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from qgis.PyQt.QtCore import QObject, pyqtSignal
except ImportError:
    from PyQt5.QtCore import QObject, pyqtSignal


# ---------------------------------------------------------------------------
# Status code constants
# ---------------------------------------------------------------------------
STATUS_QUEUED = 10
STATUS_RUNNING = 20
STATUS_FAILED = 30
STATUS_COMPLETED = 40
STATUS_CANCELLED = 50

STATUS_LABELS = {
    STATUS_QUEUED: "En attente",
    STATUS_RUNNING: "En cours",
    STATUS_FAILED: "Échoué",
    STATUS_COMPLETED: "Terminé",
    STATUS_CANCELLED: "Annulé",
}


class ODMProcessor(QObject):
    """
    Handles all communication with a NodeODM server.

    This object is meant to be used inside a QThread:
        worker = ODMWorker(...)
        worker.start()

    Signals are emitted to report progress back to the UI thread.
    """

    progress_changed = pyqtSignal(int)    # 0-100
    status_changed = pyqtSignal(str)      # human-readable status message
    task_finished = pyqtSignal(str)       # path to downloaded orthophoto
    task_failed = pyqtSignal(str)         # error message

    def __init__(self, server_url: str, token: str = "", parent=None):
        super().__init__(parent)
        self.server_url = server_url.rstrip("/")
        self.token = token.strip()
        self._cancelled = False
        self._task_uuid = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, image_paths: list, options: dict, output_dir: str):
        """
        Main entry point. Call this from the worker thread.

        :param image_paths: list of absolute paths to drone images
        :param options: dict of NodeODM processing options
        :param output_dir: directory where orthophoto.tif will be saved
        """
        if not REQUESTS_AVAILABLE:
            self.task_failed.emit(
                "Le module 'requests' n'est pas installé. "
                "Exécutez: pip install requests"
            )
            return

        try:
            self._run_internal(image_paths, options, output_dir)
        except Exception as exc:
            if not self._cancelled:
                self.task_failed.emit(str(exc))

    def cancel(self):
        """Signal cancellation. The polling loop will exit on the next iteration."""
        self._cancelled = True
        if self._task_uuid:
            try:
                self._delete_task(self._task_uuid)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _headers(self):
        h = {"Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _check_server(self):
        """Verify the server is reachable and return server info."""
        self.status_changed.emit("Vérification du serveur NodeODM…")
        resp = requests.get(
            f"{self.server_url}/info",
            headers=self._headers(),
            timeout=15,
        )
        resp.raise_for_status()
        info = resp.json()
        version = info.get("version", "inconnue")
        self.status_changed.emit(f"Serveur NodeODM v{version} connecté.")
        return info

    def _init_task(self, options: dict) -> str:
        """Initialize a new task and return its UUID."""
        self.status_changed.emit("Initialisation de la tâche ODM…")
        payload = {
            "options": json.dumps(self._build_options_list(options)),
            "name": f"ForestDL_{int(time.time())}",
        }
        resp = requests.post(
            f"{self.server_url}/task/new/init",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        uuid = data.get("uuid")
        if not uuid:
            raise RuntimeError(f"Réponse inattendue lors de l'init: {data}")
        self.status_changed.emit(f"Tâche créée: {uuid}")
        return uuid

    def _upload_images(self, uuid: str, image_paths: list):
        """Upload all images to the task."""
        total = len(image_paths)
        for idx, path in enumerate(image_paths):
            if self._cancelled:
                return
            fname = os.path.basename(path)
            self.status_changed.emit(f"Envoi image {idx + 1}/{total}: {fname}")
            with open(path, "rb") as f:
                files = {"images": (fname, f, "image/jpeg")}
                resp = requests.post(
                    f"{self.server_url}/task/new/upload/{uuid}",
                    headers=self._headers(),
                    files=files,
                    timeout=120,
                )
                resp.raise_for_status()
            # Report upload progress (first 30%)
            upload_pct = int((idx + 1) / total * 30)
            self.progress_changed.emit(upload_pct)

    def _commit_task(self, uuid: str):
        """Start processing the uploaded images."""
        self.status_changed.emit("Démarrage du traitement…")
        resp = requests.post(
            f"{self.server_url}/task/new/commit/{uuid}",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()

    def _poll_task(self, uuid: str) -> dict:
        """
        Poll task status every 5 seconds until completion or failure.
        Returns the final task info dict.
        """
        self.status_changed.emit("Traitement en cours (veuillez patienter)…")
        while not self._cancelled:
            time.sleep(5)
            try:
                resp = requests.get(
                    f"{self.server_url}/task/{uuid}/info",
                    headers=self._headers(),
                    timeout=15,
                )
                resp.raise_for_status()
                info = resp.json()
            except Exception as exc:
                self.status_changed.emit(f"Erreur de polling (nouvelle tentative): {exc}")
                continue

            status_obj = info.get("status", {})
            code = status_obj.get("code", 0)
            odm_progress = info.get("progress", 0)
            label = STATUS_LABELS.get(code, f"Code {code}")

            # Map ODM progress (0-100) to our 30-95% range
            panel_pct = 30 + int(odm_progress * 0.65)
            self.progress_changed.emit(min(panel_pct, 95))
            self.status_changed.emit(
                f"Statut: {label} — Progression ODM: {odm_progress:.1f}%"
            )

            if code == STATUS_COMPLETED:
                return info
            elif code in (STATUS_FAILED, STATUS_CANCELLED):
                error_msg = status_obj.get("errorMessage", "Erreur inconnue")
                raise RuntimeError(f"Tâche {label}: {error_msg}")

        raise RuntimeError("Traitement annulé par l'utilisateur.")

    def _download_orthophoto(self, uuid: str, output_dir: str) -> str:
        """Download the orthophoto and return the local file path."""
        self.status_changed.emit("Téléchargement de l'orthophoto…")
        self.progress_changed.emit(96)

        output_path = os.path.join(output_dir, "orthophoto.tif")
        url = f"{self.server_url}/task/{uuid}/download/orthophoto.tif"

        with requests.get(
            url,
            headers=self._headers(),
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            total_bytes = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if self._cancelled:
                        raise RuntimeError("Téléchargement annulé.")
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes > 0:
                            dl_pct = 96 + int(downloaded / total_bytes * 4)
                            self.progress_changed.emit(min(dl_pct, 99))

        self.status_changed.emit(f"Orthophoto sauvegardée: {output_path}")
        self.progress_changed.emit(100)
        return output_path

    def _delete_task(self, uuid: str):
        """Delete the task from the server (cleanup on cancel)."""
        try:
            requests.delete(
                f"{self.server_url}/task/{uuid}",
                headers=self._headers(),
                timeout=10,
            )
        except Exception:
            pass

    def _run_internal(self, image_paths: list, options: dict, output_dir: str):
        """Main processing pipeline."""
        # 1. Check server
        self._check_server()
        if self._cancelled:
            return

        # 2. Init task
        uuid = self._init_task(options)
        self._task_uuid = uuid
        if self._cancelled:
            self._delete_task(uuid)
            return

        # 3. Upload images
        self._upload_images(uuid, image_paths)
        if self._cancelled:
            self._delete_task(uuid)
            return

        # 4. Commit (start processing)
        self._commit_task(uuid)
        if self._cancelled:
            self._delete_task(uuid)
            return

        # 5. Poll until done
        self._poll_task(uuid)
        if self._cancelled:
            return

        # 6. Download result
        output_path = self._download_orthophoto(uuid, output_dir)

        # 7. Signal completion
        self.task_finished.emit(output_path)

    @staticmethod
    def _build_options_list(options: dict) -> list:
        """
        Convert an options dict to NodeODM's expected list format:
          [{"name": "key", "value": "val"}, ...]
        """
        result = []
        for key, value in options.items():
            if isinstance(value, bool):
                if value:
                    result.append({"name": key, "value": "true"})
                # If False, simply omit (NodeODM treats absence as False for flags)
            else:
                result.append({"name": key, "value": str(value)})
        return result
