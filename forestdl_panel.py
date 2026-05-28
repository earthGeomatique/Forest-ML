"""
ForestDL Panel - Main dockable panel with ODM and YOLO tabs.
"""
import os

from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QSlider, QProgressBar, QFileDialog,
    QGroupBox, QFormLayout, QSizePolicy, QSpacerItem, QAbstractItemView,
    QMessageBox, QDoubleSpinBox, QSpinBox
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from qgis.PyQt.QtGui import QFont

try:
    from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsMapLayerProxyModel
    from qgis.gui import QgsMapLayerComboBox
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False

from .odm.odm_processor import ODMProcessor
from .yolo.detector import YOLODetector


# ---------------------------------------------------------------------------
# Worker thread wrappers
# ---------------------------------------------------------------------------

class ODMWorker(QThread):
    """Runs ODMProcessor in a background thread."""
    progress_changed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    task_finished = pyqtSignal(str)
    task_failed = pyqtSignal(str)

    def __init__(self, image_paths, server_url, token, options, output_dir, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.server_url = server_url
        self.token = token
        self.options = options
        self.output_dir = output_dir
        self._processor = None

    def run(self):
        self._processor = ODMProcessor(
            server_url=self.server_url,
            token=self.token,
        )
        self._processor.progress_changed.connect(self.progress_changed)
        self._processor.status_changed.connect(self.status_changed)
        self._processor.task_finished.connect(self.task_finished)
        self._processor.task_failed.connect(self.task_failed)
        self._processor.run(
            image_paths=self.image_paths,
            options=self.options,
            output_dir=self.output_dir,
        )

    def stop(self):
        if self._processor:
            self._processor.cancel()


class YOLOWorker(QThread):
    """Runs YOLODetector in a background thread."""
    progress_changed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    detection_finished = pyqtSignal(str)
    detection_failed = pyqtSignal(str)

    def __init__(self, raster_path, model_path, classes, conf, iou, tile_size, output_path, parent=None):
        super().__init__(parent)
        self.raster_path = raster_path
        self.model_path = model_path
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.tile_size = tile_size
        self.output_path = output_path
        self._detector = None

    def run(self):
        self._detector = YOLODetector()
        self._detector.progress_changed.connect(self.progress_changed)
        self._detector.status_changed.connect(self.status_changed)
        self._detector.detection_finished.connect(self.detection_finished)
        self._detector.detection_failed.connect(self.detection_failed)
        self._detector.run(
            raster_path=self.raster_path,
            model_path=self.model_path,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            tile_size=self.tile_size,
            output_path=self.output_path,
        )

    def stop(self):
        if self._detector:
            self._detector.cancel()


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class ForestDLPanel(QDockWidget):
    """Dockable side panel for ForestDL plugin."""

    def __init__(self, iface, parent=None):
        super().__init__("ForestDL", parent)
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)

        # Worker references
        self._odm_worker = None
        self._yolo_worker = None

        # ODM state
        self._odm_result_path = None

        # YOLO state
        self._yolo_result_path = None

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(360)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        container = QWidget()
        self.setWidget(container)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Title
        title = QLabel("<b>ForestDL</b> — Orthophoto &amp; Détection")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(11)
        title.setFont(title_font)
        main_layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_odm_tab(), "🛩 ODM")
        self.tabs.addTab(self._build_yolo_tab(), "🔍 YOLO")

    # ------------------------------------------------------------------
    # ODM Tab
    # ------------------------------------------------------------------

    def _build_odm_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # ---- Images group ----
        grp_images = QGroupBox("Images drone")
        grp_layout = QVBoxLayout(grp_images)

        self.odm_image_list = QListWidget()
        self.odm_image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.odm_image_list.setMinimumHeight(100)
        grp_layout.addWidget(self.odm_image_list)

        btn_row = QHBoxLayout()
        self.btn_add_images = QPushButton("Ajouter images")
        self.btn_add_images.clicked.connect(self._odm_add_images)
        self.btn_remove_images = QPushButton("Supprimer sélection")
        self.btn_remove_images.clicked.connect(self._odm_remove_images)
        self.btn_clear_images = QPushButton("Tout effacer")
        self.btn_clear_images.clicked.connect(self._odm_clear_images)
        btn_row.addWidget(self.btn_add_images)
        btn_row.addWidget(self.btn_remove_images)
        btn_row.addWidget(self.btn_clear_images)
        grp_layout.addLayout(btn_row)

        layout.addWidget(grp_images)

        # ---- Server group ----
        grp_server = QGroupBox("Serveur NodeODM")
        form_server = QFormLayout(grp_server)

        self.odm_url = QLineEdit("http://localhost:3000")
        form_server.addRow("URL serveur:", self.odm_url)

        self.odm_token = QLineEdit()
        self.odm_token.setPlaceholderText("(laisser vide si non requis)")
        self.odm_token.setEchoMode(QLineEdit.Password)
        form_server.addRow("Token:", self.odm_token)

        self.btn_test_server = QPushButton("Tester connexion")
        self.btn_test_server.clicked.connect(self._odm_test_server)
        form_server.addRow("", self.btn_test_server)

        layout.addWidget(grp_server)

        # ---- Options group ----
        grp_opts = QGroupBox("Options de traitement")
        form_opts = QFormLayout(grp_opts)

        self.odm_quality = QComboBox()
        self.odm_quality.addItems(["Basse (lowest)", "Normale (low)", "Moyenne (medium)", "Haute (high)", "Ultra (ultra)"])
        self.odm_quality.setCurrentIndex(2)
        form_opts.addRow("Qualité:", self.odm_quality)

        self.odm_gsd = QDoubleSpinBox()
        self.odm_gsd.setRange(0.1, 100.0)
        self.odm_gsd.setValue(5.0)
        self.odm_gsd.setSuffix(" cm/px")
        self.odm_gsd.setDecimals(1)
        form_opts.addRow("Résolution (GSD):", self.odm_gsd)

        self.odm_pc = QCheckBox("Générer nuage de points")
        self.odm_pc.setChecked(True)
        form_opts.addRow("", self.odm_pc)

        self.odm_dtm = QCheckBox("Générer DTM")
        self.odm_dtm.setChecked(False)
        form_opts.addRow("", self.odm_dtm)

        self.odm_dsm = QCheckBox("Générer DSM")
        self.odm_dsm.setChecked(True)
        form_opts.addRow("", self.odm_dsm)

        layout.addWidget(grp_opts)

        # ---- Output directory ----
        grp_out = QGroupBox("Dossier de sortie")
        out_layout = QHBoxLayout(grp_out)
        self.odm_output_dir = QLineEdit()
        self.odm_output_dir.setPlaceholderText("Sélectionner un dossier…")
        btn_browse_out = QPushButton("Parcourir")
        btn_browse_out.clicked.connect(self._odm_browse_output)
        out_layout.addWidget(self.odm_output_dir)
        out_layout.addWidget(btn_browse_out)
        layout.addWidget(grp_out)

        # ---- Progress & status ----
        self.odm_progress = QProgressBar()
        self.odm_progress.setRange(0, 100)
        self.odm_progress.setValue(0)
        layout.addWidget(self.odm_progress)

        self.odm_status = QLabel("Prêt.")
        self.odm_status.setWordWrap(True)
        layout.addWidget(self.odm_status)

        # ---- Launch / Load buttons ----
        btn_launch_row = QHBoxLayout()
        self.btn_odm_launch = QPushButton("Lancer le traitement ODM")
        self.btn_odm_launch.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_odm_launch.clicked.connect(self._odm_launch)
        btn_launch_row.addWidget(self.btn_odm_launch)

        self.btn_odm_stop = QPushButton("Arrêter")
        self.btn_odm_stop.setEnabled(False)
        self.btn_odm_stop.clicked.connect(self._odm_stop)
        btn_launch_row.addWidget(self.btn_odm_stop)

        layout.addLayout(btn_launch_row)

        self.btn_odm_load = QPushButton("Charger l'orthophoto dans QGIS")
        self.btn_odm_load.setEnabled(False)
        self.btn_odm_load.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_odm_load.clicked.connect(self._odm_load_result)
        layout.addWidget(self.btn_odm_load)

        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return widget

    # ------------------------------------------------------------------
    # YOLO Tab
    # ------------------------------------------------------------------

    def _build_yolo_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # ---- Layer selection ----
        grp_layer = QGroupBox("Couche raster source")
        layer_layout = QVBoxLayout(grp_layer)

        if QGIS_AVAILABLE:
            self.yolo_layer_combo = QgsMapLayerComboBox()
            self.yolo_layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
            layer_layout.addWidget(self.yolo_layer_combo)
        else:
            self.yolo_layer_combo = QComboBox()
            layer_layout.addWidget(self.yolo_layer_combo)

        self.btn_refresh_layers = QPushButton("Actualiser les couches")
        self.btn_refresh_layers.clicked.connect(self._yolo_refresh_layers)
        layer_layout.addWidget(self.btn_refresh_layers)

        # Allow manual file path as fallback
        manual_row = QHBoxLayout()
        self.yolo_raster_path = QLineEdit()
        self.yolo_raster_path.setPlaceholderText("Ou chemin vers fichier raster…")
        btn_browse_raster = QPushButton("Parcourir")
        btn_browse_raster.clicked.connect(self._yolo_browse_raster)
        manual_row.addWidget(self.yolo_raster_path)
        manual_row.addWidget(btn_browse_raster)
        layer_layout.addLayout(manual_row)

        layout.addWidget(grp_layer)

        # ---- Detection classes ----
        grp_classes = QGroupBox("Classes à détecter")
        classes_layout = QVBoxLayout(grp_classes)

        self.chk_mangroves = QCheckBox("Mangroves")
        self.chk_mangroves.setChecked(True)
        self.chk_trees = QCheckBox("Arbres fruitiers")
        self.chk_trees.setChecked(True)
        self.chk_vehicles = QCheckBox("Véhicules (voitures, camions, bus)")
        self.chk_vehicles.setChecked(True)
        self.chk_buildings = QCheckBox("Bâtiments")
        self.chk_buildings.setChecked(True)

        classes_layout.addWidget(self.chk_mangroves)
        classes_layout.addWidget(self.chk_trees)
        classes_layout.addWidget(self.chk_vehicles)
        classes_layout.addWidget(self.chk_buildings)

        layout.addWidget(grp_classes)

        # ---- Model selection ----
        grp_model = QGroupBox("Modèle YOLO")
        model_layout = QFormLayout(grp_model)

        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.addItems([
            "YOLOv8n (nano, rapide)",
            "YOLOv8s (small)",
            "YOLOv8m (medium, recommandé)",
            "YOLOv8l (large)",
            "YOLOv8x (extra-large, lent)",
            "Modèle personnalisé…",
        ])
        self.yolo_model_combo.setCurrentIndex(2)
        self.yolo_model_combo.currentIndexChanged.connect(self._yolo_model_changed)
        model_layout.addRow("Modèle:", self.yolo_model_combo)

        custom_row = QHBoxLayout()
        self.yolo_custom_path = QLineEdit()
        self.yolo_custom_path.setPlaceholderText("Chemin vers .pt personnalisé…")
        self.yolo_custom_path.setEnabled(False)
        self.btn_browse_model = QPushButton("Parcourir")
        self.btn_browse_model.setEnabled(False)
        self.btn_browse_model.clicked.connect(self._yolo_browse_model)
        custom_row.addWidget(self.yolo_custom_path)
        custom_row.addWidget(self.btn_browse_model)
        model_layout.addRow("Modèle .pt:", custom_row)

        layout.addWidget(grp_model)

        # ---- Detection parameters ----
        grp_params = QGroupBox("Paramètres de détection")
        params_layout = QFormLayout(grp_params)

        conf_row = QHBoxLayout()
        self.yolo_conf_slider = QSlider(Qt.Horizontal)
        self.yolo_conf_slider.setRange(5, 95)
        self.yolo_conf_slider.setValue(25)
        self.yolo_conf_slider.setTickInterval(5)
        self.yolo_conf_label = QLabel("0.25")
        self.yolo_conf_label.setMinimumWidth(35)
        self.yolo_conf_slider.valueChanged.connect(
            lambda v: self.yolo_conf_label.setText(f"{v/100:.2f}")
        )
        conf_row.addWidget(self.yolo_conf_slider)
        conf_row.addWidget(self.yolo_conf_label)
        params_layout.addRow("Confiance:", conf_row)

        iou_row = QHBoxLayout()
        self.yolo_iou_slider = QSlider(Qt.Horizontal)
        self.yolo_iou_slider.setRange(10, 90)
        self.yolo_iou_slider.setValue(45)
        self.yolo_iou_slider.setTickInterval(5)
        self.yolo_iou_label = QLabel("0.45")
        self.yolo_iou_label.setMinimumWidth(35)
        self.yolo_iou_slider.valueChanged.connect(
            lambda v: self.yolo_iou_label.setText(f"{v/100:.2f}")
        )
        iou_row.addWidget(self.yolo_iou_slider)
        iou_row.addWidget(self.yolo_iou_label)
        params_layout.addRow("Seuil IoU:", iou_row)

        self.yolo_tile_combo = QComboBox()
        self.yolo_tile_combo.addItems(["512", "1024", "2048"])
        self.yolo_tile_combo.setCurrentIndex(1)
        params_layout.addRow("Taille tuile (px):", self.yolo_tile_combo)

        layout.addWidget(grp_params)

        # ---- Output ----
        grp_yolo_out = QGroupBox("Fichier de sortie")
        yolo_out_layout = QHBoxLayout(grp_yolo_out)
        self.yolo_output_path = QLineEdit()
        self.yolo_output_path.setPlaceholderText("Chemin du GeoJSON de sortie…")
        btn_browse_yolo_out = QPushButton("Parcourir")
        btn_browse_yolo_out.clicked.connect(self._yolo_browse_output)
        yolo_out_layout.addWidget(self.yolo_output_path)
        yolo_out_layout.addWidget(btn_browse_yolo_out)
        layout.addWidget(grp_yolo_out)

        # ---- Progress & status ----
        self.yolo_progress = QProgressBar()
        self.yolo_progress.setRange(0, 100)
        self.yolo_progress.setValue(0)
        layout.addWidget(self.yolo_progress)

        self.yolo_status = QLabel("Prêt.")
        self.yolo_status.setWordWrap(True)
        layout.addWidget(self.yolo_status)

        # ---- Launch / Load buttons ----
        btn_yolo_row = QHBoxLayout()
        self.btn_yolo_launch = QPushButton("Lancer la détection YOLO")
        self.btn_yolo_launch.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_yolo_launch.clicked.connect(self._yolo_launch)
        btn_yolo_row.addWidget(self.btn_yolo_launch)

        self.btn_yolo_stop = QPushButton("Arrêter")
        self.btn_yolo_stop.setEnabled(False)
        self.btn_yolo_stop.clicked.connect(self._yolo_stop)
        btn_yolo_row.addWidget(self.btn_yolo_stop)

        layout.addLayout(btn_yolo_row)

        self.btn_yolo_load = QPushButton("Charger les résultats dans QGIS")
        self.btn_yolo_load.setEnabled(False)
        self.btn_yolo_load.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_yolo_load.clicked.connect(self._yolo_load_result)
        layout.addWidget(self.btn_yolo_load)

        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return widget

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def switch_to_tab(self, index):
        self.tabs.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # ODM slots
    # ------------------------------------------------------------------

    def _odm_add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Sélectionner des images drone",
            "", "Images (*.jpg *.jpeg *.JPG *.JPEG *.png *.PNG *.tif *.tiff *.TIF *.TIFF)"
        )
        existing = [self.odm_image_list.item(i).text()
                    for i in range(self.odm_image_list.count())]
        for p in paths:
            if p not in existing:
                self.odm_image_list.addItem(QListWidgetItem(p))
        self.odm_status.setText(f"{self.odm_image_list.count()} image(s) dans la liste.")

    def _odm_remove_images(self):
        for item in self.odm_image_list.selectedItems():
            self.odm_image_list.takeItem(self.odm_image_list.row(item))

    def _odm_clear_images(self):
        self.odm_image_list.clear()
        self.odm_status.setText("Liste d'images effacée.")

    def _odm_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie")
        if folder:
            self.odm_output_dir.setText(folder)

    def _odm_test_server(self):
        import requests
        url = self.odm_url.text().rstrip("/")
        token = self.odm_token.text().strip()
        try:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(f"{url}/info", headers=headers, timeout=10)
            if resp.status_code == 200:
                info = resp.json()
                version = info.get("version", "?")
                QMessageBox.information(self, "Connexion réussie",
                    f"Serveur NodeODM connecté.\nVersion: {version}")
            else:
                QMessageBox.warning(self, "Erreur serveur",
                    f"Code HTTP: {resp.status_code}\n{resp.text[:200]}")
        except Exception as e:
            QMessageBox.critical(self, "Connexion impossible",
                f"Impossible de joindre le serveur:\n{e}")

    def _odm_launch(self):
        # Validation
        image_count = self.odm_image_list.count()
        if image_count == 0:
            QMessageBox.warning(self, "Aucune image", "Veuillez ajouter des images drone.")
            return

        server_url = self.odm_url.text().strip()
        if not server_url:
            QMessageBox.warning(self, "URL manquante", "Veuillez entrer l'URL du serveur NodeODM.")
            return

        output_dir = self.odm_output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Dossier manquant", "Veuillez choisir un dossier de sortie.")
            return

        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de créer le dossier:\n{e}")
                return

        image_paths = [self.odm_image_list.item(i).text()
                       for i in range(self.odm_image_list.count())]

        # Build options
        quality_map = {
            0: "lowest", 1: "low", 2: "medium", 3: "high", 4: "ultra"
        }
        quality = quality_map.get(self.odm_quality.currentIndex(), "medium")
        gsd = self.odm_gsd.value()

        options = {
            "orthophoto-resolution": gsd,
            "quality": quality,
            "pc-quality": quality,
            "feature-quality": quality,
            "min-num-features": 8000,
            "mesh-size": 200000,
            "dsm": self.odm_dsm.isChecked(),
            "dtm": self.odm_dtm.isChecked(),
            "pc-classify": self.odm_pc.isChecked(),
        }

        token = self.odm_token.text().strip()

        # Start worker
        self._odm_result_path = None
        self.btn_odm_load.setEnabled(False)
        self.btn_odm_launch.setEnabled(False)
        self.btn_odm_stop.setEnabled(True)
        self.odm_progress.setValue(0)
        self.odm_status.setText("Démarrage du traitement ODM…")

        self._odm_worker = ODMWorker(
            image_paths=image_paths,
            server_url=server_url,
            token=token,
            options=options,
            output_dir=output_dir,
        )
        self._odm_worker.progress_changed.connect(self._odm_on_progress)
        self._odm_worker.status_changed.connect(self._odm_on_status)
        self._odm_worker.task_finished.connect(self._odm_on_finished)
        self._odm_worker.task_failed.connect(self._odm_on_failed)
        self._odm_worker.start()

    def _odm_stop(self):
        if self._odm_worker and self._odm_worker.isRunning():
            self._odm_worker.stop()
            self._odm_worker.wait(3000)
            self.odm_status.setText("Traitement arrêté par l'utilisateur.")
        self._odm_set_idle()

    @pyqtSlot(int)
    def _odm_on_progress(self, value):
        self.odm_progress.setValue(value)

    @pyqtSlot(str)
    def _odm_on_status(self, message):
        self.odm_status.setText(message)

    @pyqtSlot(str)
    def _odm_on_finished(self, path):
        self._odm_result_path = path
        self.odm_status.setText(f"Traitement terminé ! Orthophoto: {path}")
        self.odm_progress.setValue(100)
        self.btn_odm_load.setEnabled(True)
        self._odm_set_idle()

    @pyqtSlot(str)
    def _odm_on_failed(self, error):
        self.odm_status.setText(f"Erreur: {error}")
        QMessageBox.critical(self, "Erreur ODM", f"Le traitement a échoué:\n{error}")
        self._odm_set_idle()

    def _odm_set_idle(self):
        self.btn_odm_launch.setEnabled(True)
        self.btn_odm_stop.setEnabled(False)

    def _odm_load_result(self):
        if not self._odm_result_path or not os.path.exists(self._odm_result_path):
            QMessageBox.warning(self, "Fichier introuvable",
                "L'orthophoto n'a pas été trouvée. Vérifiez le dossier de sortie.")
            return
        if QGIS_AVAILABLE:
            layer_name = os.path.splitext(os.path.basename(self._odm_result_path))[0]
            layer = QgsRasterLayer(self._odm_result_path, layer_name)
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.odm_status.setText(f"Couche '{layer_name}' chargée dans QGIS.")
            else:
                QMessageBox.critical(self, "Erreur QGIS", "Impossible de charger la couche raster.")
        else:
            QMessageBox.information(self, "QGIS non disponible",
                f"Fichier: {self._odm_result_path}")

    # ------------------------------------------------------------------
    # YOLO slots
    # ------------------------------------------------------------------

    def _yolo_refresh_layers(self):
        if not QGIS_AVAILABLE:
            return
        # QgsMapLayerComboBox refreshes automatically; this is a manual trigger
        if hasattr(self, 'yolo_layer_combo') and isinstance(self.yolo_layer_combo, QComboBox):
            self.yolo_layer_combo.clear()
            for layer in QgsProject.instance().mapLayers().values():
                if hasattr(layer, 'bandCount'):
                    self.yolo_layer_combo.addItem(layer.name(), layer.id())

    def _yolo_browse_raster(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un raster",
            "", "Rasters (*.tif *.tiff *.TIF *.TIFF *.img *.vrt *.jp2 *.png)"
        )
        if path:
            self.yolo_raster_path.setText(path)

    def _yolo_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un modèle YOLO",
            "", "Modèle YOLO (*.pt *.onnx)"
        )
        if path:
            self.yolo_custom_path.setText(path)

    def _yolo_browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer les détections",
            "detections.geojson", "GeoJSON (*.geojson);;Shapefile (*.shp)"
        )
        if path:
            self.yolo_output_path.setText(path)

    def _yolo_model_changed(self, index):
        is_custom = (index == self.yolo_model_combo.count() - 1)
        self.yolo_custom_path.setEnabled(is_custom)
        self.btn_browse_model.setEnabled(is_custom)

    def _get_yolo_model_path(self):
        """Return the model path string based on current combo selection."""
        idx = self.yolo_model_combo.currentIndex()
        model_names = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        if idx < len(model_names):
            return model_names[idx]
        # Custom
        return self.yolo_custom_path.text().strip()

    def _get_raster_path(self):
        """Return the raster path from the layer combo or manual input."""
        manual = self.yolo_raster_path.text().strip()
        if manual and os.path.exists(manual):
            return manual

        if not QGIS_AVAILABLE:
            return ""

        if isinstance(self.yolo_layer_combo, QComboBox):
            # Fallback combo
            layer_id = self.yolo_layer_combo.currentData()
            if layer_id:
                layer = QgsProject.instance().mapLayer(layer_id)
                if layer:
                    return layer.source()
        else:
            # QgsMapLayerComboBox
            layer = self.yolo_layer_combo.currentLayer()
            if layer:
                return layer.source()
        return ""

    def _yolo_launch(self):
        raster_path = self._get_raster_path()
        if not raster_path:
            QMessageBox.warning(self, "Raster manquant",
                "Veuillez sélectionner une couche raster ou un fichier.")
            return

        model_path = self._get_yolo_model_path()
        if not model_path:
            QMessageBox.warning(self, "Modèle manquant",
                "Veuillez sélectionner un modèle YOLO.")
            return

        output_path = self.yolo_output_path.text().strip()
        if not output_path:
            # Default to same directory as raster
            base = os.path.splitext(raster_path)[0]
            output_path = base + "_detections.geojson"
            self.yolo_output_path.setText(output_path)

        # Build classes list
        classes = []
        if self.chk_mangroves.isChecked():
            classes.append("mangrove")
        if self.chk_trees.isChecked():
            classes.append("arbre_fruitier")
        if self.chk_vehicles.isChecked():
            classes.append("vehicule")
        if self.chk_buildings.isChecked():
            classes.append("batiment")

        if not classes:
            QMessageBox.warning(self, "Aucune classe",
                "Veuillez sélectionner au moins une classe à détecter.")
            return

        conf = self.yolo_conf_slider.value() / 100.0
        iou = self.yolo_iou_slider.value() / 100.0
        tile_size = int(self.yolo_tile_combo.currentText())

        # Start worker
        self._yolo_result_path = None
        self.btn_yolo_load.setEnabled(False)
        self.btn_yolo_launch.setEnabled(False)
        self.btn_yolo_stop.setEnabled(True)
        self.yolo_progress.setValue(0)
        self.yolo_status.setText("Démarrage de la détection YOLO…")

        self._yolo_worker = YOLOWorker(
            raster_path=raster_path,
            model_path=model_path,
            classes=classes,
            conf=conf,
            iou=iou,
            tile_size=tile_size,
            output_path=output_path,
        )
        self._yolo_worker.progress_changed.connect(self._yolo_on_progress)
        self._yolo_worker.status_changed.connect(self._yolo_on_status)
        self._yolo_worker.detection_finished.connect(self._yolo_on_finished)
        self._yolo_worker.detection_failed.connect(self._yolo_on_failed)
        self._yolo_worker.start()

    def _yolo_stop(self):
        if self._yolo_worker and self._yolo_worker.isRunning():
            self._yolo_worker.stop()
            self._yolo_worker.wait(3000)
            self.yolo_status.setText("Détection arrêtée par l'utilisateur.")
        self._yolo_set_idle()

    @pyqtSlot(int)
    def _yolo_on_progress(self, value):
        self.yolo_progress.setValue(value)

    @pyqtSlot(str)
    def _yolo_on_status(self, message):
        self.yolo_status.setText(message)

    @pyqtSlot(str)
    def _yolo_on_finished(self, path):
        self._yolo_result_path = path
        self.yolo_status.setText(f"Détection terminée ! Résultats: {path}")
        self.yolo_progress.setValue(100)
        self.btn_yolo_load.setEnabled(True)
        self._yolo_set_idle()

    @pyqtSlot(str)
    def _yolo_on_failed(self, error):
        self.yolo_status.setText(f"Erreur: {error}")
        QMessageBox.critical(self, "Erreur YOLO", f"La détection a échoué:\n{error}")
        self._yolo_set_idle()

    def _yolo_set_idle(self):
        self.btn_yolo_launch.setEnabled(True)
        self.btn_yolo_stop.setEnabled(False)

    def _yolo_load_result(self):
        if not self._yolo_result_path or not os.path.exists(self._yolo_result_path):
            QMessageBox.warning(self, "Fichier introuvable",
                "Le fichier de résultats est introuvable.")
            return
        if QGIS_AVAILABLE:
            layer_name = os.path.splitext(os.path.basename(self._yolo_result_path))[0]
            layer = QgsVectorLayer(self._yolo_result_path, layer_name, "ogr")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.yolo_status.setText(f"Couche '{layer_name}' chargée dans QGIS.")
            else:
                QMessageBox.critical(self, "Erreur QGIS",
                    "Impossible de charger la couche vectorielle de résultats.")
        else:
            QMessageBox.information(self, "QGIS non disponible",
                f"Fichier: {self._yolo_result_path}")
