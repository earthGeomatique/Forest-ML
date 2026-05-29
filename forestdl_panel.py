"""
ForestDL Panel — Panneau ancrable principal (QDockWidget).

Onglets :
  0  🛩  Traitement drone  — alignement photos + densification + ortho-mosaïque (ODM)
  1  🔍  Détection YOLO    — YOLOv8 / ONNX sur orthophotos
  2  ⚙️  Configuration     — Clés API + préférences
"""

import os

from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QSlider, QProgressBar, QFileDialog,
    QGroupBox, QFormLayout, QSizePolicy, QSpacerItem, QAbstractItemView,
    QMessageBox, QDoubleSpinBox, QSpinBox, QScrollArea, QFrame,
    QToolButton,
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from qgis.PyQt.QtGui import QFont

try:
    from qgis.core import (
        QgsProject, QgsRasterLayer, QgsVectorLayer, QgsMapLayerProxyModel,
    )
    from qgis.gui import QgsMapLayerComboBox
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False

from .odm.odm_processor import ODMProcessor
from .yolo.detector import YOLODetector
from .yolo.onnx_detector import ONNXDetector, list_zoo_models, download_zoo_model
from .config.api_config import load_config, save_config


# ══════════════════════════════════════════════════════════════════════════════
# Worker threads
# ══════════════════════════════════════════════════════════════════════════════

class ODMWorker(QThread):
    progress_changed = pyqtSignal(int)
    status_changed   = pyqtSignal(str)
    task_finished    = pyqtSignal(str)
    task_failed      = pyqtSignal(str)

    def __init__(self, image_paths, server_url, token, options,
                 output_dir, gcp_path=None, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.server_url  = server_url
        self.token       = token
        self.options     = options
        self.output_dir  = output_dir
        self.gcp_path    = gcp_path
        self._processor  = None

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
            gcp_path=self.gcp_path,
        )

    def stop(self):
        if self._processor:
            self._processor.cancel()


class YOLOWorker(QThread):
    progress_changed   = pyqtSignal(int)
    status_changed     = pyqtSignal(str)
    detection_finished = pyqtSignal(str)
    detection_failed   = pyqtSignal(str)

    def __init__(self, raster_path, model_path, model_backend,
                 model_type, classes, conf, iou,
                 tile_size, output_path, parent=None):
        super().__init__(parent)
        self.raster_path   = raster_path
        self.model_path    = model_path
        self.model_backend = model_backend   # "ultralytics" | "onnxruntime"
        self.model_type    = model_type       # "yolov3" | "ssd" | etc.
        self.classes       = classes
        self.conf          = conf
        self.iou           = iou
        self.tile_size     = tile_size
        self.output_path   = output_path
        self._detector     = None

    def run(self):
        if self.model_backend == "onnxruntime":
            self._detector = ONNXDetector()
        else:
            self._detector = YOLODetector()

        self._detector.progress_changed.connect(self.progress_changed)
        self._detector.status_changed.connect(self.status_changed)
        self._detector.detection_finished.connect(self.detection_finished)
        self._detector.detection_failed.connect(self.detection_failed)

        if self.model_backend == "onnxruntime":
            self._detector.run(
                raster_path=self.raster_path,
                model_path=self.model_path,
                model_type=self.model_type,
                classes=self.classes,
                conf=self.conf,
                iou=self.iou,
                tile_size=self.tile_size,
                output_path=self.output_path,
            )
        else:
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


class ONNXDownloadWorker(QThread):
    progress_changed = pyqtSignal(int)
    status_changed   = pyqtSignal(str)
    finished_path    = pyqtSignal(str)
    failed           = pyqtSignal(str)

    def __init__(self, model_name, dest_dir, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.dest_dir   = dest_dir

    def run(self):
        try:
            self.status_changed.emit(f"Téléchargement {self.model_name}…")
            path = download_zoo_model(
                self.model_name, self.dest_dir,
                progress_callback=self.progress_changed.emit,
            )
            self.status_changed.emit(f"Modèle téléchargé: {path}")
            self.finished_path.emit(path)
        except Exception as exc:
            self.failed.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Panneau principal
# ══════════════════════════════════════════════════════════════════════════════

class ForestDLPanel(QDockWidget):

    def __init__(self, iface, parent=None):
        super().__init__("ForestDL", parent)
        self.iface      = iface
        self.plugin_dir = os.path.dirname(__file__)

        self._odm_worker        = None
        self._yolo_worker       = None
        self._onnx_dl_worker    = None

        self._odm_result_path   = None
        self._yolo_result_path  = None

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(380)

        self._build_ui()
        self._load_config_to_ui()

    # ─────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        container = QWidget()
        self.setWidget(container)

        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Titre
        title = QLabel("<b>ForestDL</b> — Drone · Orthophoto · Détection")
        title.setAlignment(Qt.AlignCenter)
        tf = QFont()
        tf.setPointSize(10)
        tf.setBold(True)
        title.setFont(tf)
        main_layout.addWidget(title)

        # Onglets
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tabs.addTab(self._wrap_scroll(self._build_odm_tab()),   "🛩 Traitement drone")
        self.tabs.addTab(self._wrap_scroll(self._build_yolo_tab()),  "🔍 Détection YOLO")
        self.tabs.addTab(self._wrap_scroll(self._build_config_tab()), "⚙️ Configuration")

    def _wrap_scroll(self, widget):
        """Enveloppe un widget dans un QScrollArea pour les petits écrans."""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    # ──────────────────────────────────────────────────────────────────────
    # Onglet 0 — Traitement drone (ODM)
    # ──────────────────────────────────────────────────────────────────────

    def _build_odm_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Images ──────────────────────────────────────────────────────
        grp_images = QGroupBox("📁 Images drone")
        gl = QVBoxLayout(grp_images)

        self.odm_image_list = QListWidget()
        self.odm_image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.odm_image_list.setMinimumHeight(90)
        gl.addWidget(self.odm_image_list)

        br = QHBoxLayout()
        self.btn_add_images = QPushButton("Ajouter")
        self.btn_add_images.clicked.connect(self._odm_add_images)
        self.btn_remove_images = QPushButton("Supprimer")
        self.btn_remove_images.clicked.connect(self._odm_remove_images)
        self.btn_clear_images = QPushButton("Tout effacer")
        self.btn_clear_images.clicked.connect(self._odm_clear_images)
        br.addWidget(self.btn_add_images)
        br.addWidget(self.btn_remove_images)
        br.addWidget(self.btn_clear_images)
        gl.addLayout(br)
        layout.addWidget(grp_images)

        # ── Serveur NodeODM ─────────────────────────────────────────────
        grp_server = QGroupBox("🌐 Serveur NodeODM")
        fs = QFormLayout(grp_server)

        self.odm_url = QLineEdit("http://localhost:3000")
        fs.addRow("URL serveur:", self.odm_url)

        self.odm_token = QLineEdit()
        self.odm_token.setPlaceholderText("(vide si non requis)")
        self.odm_token.setEchoMode(QLineEdit.Password)
        fs.addRow("Token:", self.odm_token)

        self.btn_test_server = QPushButton("Tester connexion")
        self.btn_test_server.clicked.connect(self._odm_test_server)
        fs.addRow("", self.btn_test_server)
        layout.addWidget(grp_server)

        # ── Photogrammétrie ─────────────────────────────────────────────
        grp_photo = QGroupBox("📐 Photogrammétrie & Alignement")
        fp = QFormLayout(grp_photo)

        self.odm_feature_type = QComboBox()
        self.odm_feature_type.addItems([
            "SIFT (recommandé)", "ORB (rapide)", "HAHOG (robuste)",
            "DSP-SIFT (haute précision)",
        ])
        fp.addRow("Extraction traits:", self.odm_feature_type)

        self.odm_matcher = QComboBox()
        self.odm_matcher.addItems([
            "FLANN (recommandé)", "BFMatcher", "Bow (vocabulaire)",
        ])
        fp.addRow("Correspondance:", self.odm_matcher)

        self.odm_min_features = QSpinBox()
        self.odm_min_features.setRange(1000, 50000)
        self.odm_min_features.setValue(8000)
        self.odm_min_features.setSingleStep(1000)
        fp.addRow("Min traits/image:", self.odm_min_features)

        self.odm_dense_method = QComboBox()
        self.odm_dense_method.addItems([
            "MVS (rapide, recommandé)", "OpenMVS Dense", "Poisson",
        ])
        fp.addRow("Densification pts:", self.odm_dense_method)

        layout.addWidget(grp_photo)

        # ── Options générales ────────────────────────────────────────────
        grp_opts = QGroupBox("⚙️ Options de traitement")
        fo = QFormLayout(grp_opts)

        self.odm_quality = QComboBox()
        self.odm_quality.addItems([
            "Basse (lowest)", "Normale (low)", "Moyenne (medium)",
            "Haute (high)", "Ultra (ultra)",
        ])
        self.odm_quality.setCurrentIndex(2)
        fo.addRow("Qualité globale:", self.odm_quality)

        self.odm_gsd = QDoubleSpinBox()
        self.odm_gsd.setRange(0.5, 100.0)
        self.odm_gsd.setValue(5.0)
        self.odm_gsd.setSuffix(" cm/px")
        self.odm_gsd.setDecimals(1)
        fo.addRow("Résolution (GSD):", self.odm_gsd)

        self.odm_pc = QCheckBox("Générer nuage de points")
        self.odm_pc.setChecked(True)
        fo.addRow("", self.odm_pc)

        self.odm_dtm = QCheckBox("Générer DTM (Modèle Terrain Numérique)")
        self.odm_dtm.setChecked(False)
        fo.addRow("", self.odm_dtm)

        self.odm_dsm = QCheckBox("Générer DSM (Modèle Surface Numérique)")
        self.odm_dsm.setChecked(True)
        fo.addRow("", self.odm_dsm)

        self.odm_use_3dmesh = QCheckBox("Maillage 3D (meilleure orthorectification)")
        self.odm_use_3dmesh.setChecked(False)
        fo.addRow("", self.odm_use_3dmesh)

        layout.addWidget(grp_opts)

        # ── Orthorectification précise ───────────────────────────────────
        grp_ortho = QGroupBox("🎯 Orthorectification de précision")
        fo2 = QFormLayout(grp_ortho)

        # MNT de référence
        mnt_row = QHBoxLayout()
        self.odm_dem_path = QLineEdit()
        self.odm_dem_path.setPlaceholderText("MNT .tif (optionnel, améliore la précision)")
        btn_browse_dem = QPushButton("Parcourir")
        btn_browse_dem.clicked.connect(self._odm_browse_dem)
        mnt_row.addWidget(self.odm_dem_path)
        mnt_row.addWidget(btn_browse_dem)
        fo2.addRow("MNT référence:", mnt_row)

        # Fichier GCP
        gcp_row = QHBoxLayout()
        self.odm_gcp_path = QLineEdit()
        self.odm_gcp_path.setPlaceholderText("Fichier GCP .txt (optionnel, précision < 3 cm)")
        btn_browse_gcp = QPushButton("Parcourir")
        btn_browse_gcp.clicked.connect(self._odm_browse_gcp)
        gcp_row.addWidget(self.odm_gcp_path)
        gcp_row.addWidget(btn_browse_gcp)
        fo2.addRow("Fichier GCP:", gcp_row)

        # Préréglages précision
        preset_row = QHBoxLayout()
        btn_preset_3cm = QPushButton("⚡ Préréglage < 3 cm")
        btn_preset_3cm.setStyleSheet("background-color: #1565C0; color: white;")
        btn_preset_3cm.clicked.connect(self._odm_preset_3cm)
        btn_preset_6cm = QPushButton("⚡ Préréglage < 6 cm")
        btn_preset_6cm.setStyleSheet("background-color: #0277BD; color: white;")
        btn_preset_6cm.clicked.connect(self._odm_preset_6cm)
        preset_row.addWidget(btn_preset_3cm)
        preset_row.addWidget(btn_preset_6cm)
        fo2.addRow("Précision cible:", preset_row)

        # Indicateur de précision attendue
        self.odm_precision_label = QLabel("Résolution actuelle: 5.0 cm/px")
        self.odm_precision_label.setStyleSheet("color: #555; font-style: italic;")
        fo2.addRow("", self.odm_precision_label)
        self.odm_gsd.valueChanged.connect(self._odm_update_precision_label)

        layout.addWidget(grp_ortho)

        # ── Dossier de sortie ─────────────────────────────────────────────
        grp_out = QGroupBox("📂 Dossier de sortie")
        out_layout = QHBoxLayout(grp_out)
        self.odm_output_dir = QLineEdit()
        self.odm_output_dir.setPlaceholderText("Sélectionner un dossier…")
        btn_browse_out = QPushButton("Parcourir")
        btn_browse_out.clicked.connect(self._odm_browse_output)
        out_layout.addWidget(self.odm_output_dir)
        out_layout.addWidget(btn_browse_out)
        layout.addWidget(grp_out)

        # ── Progression ───────────────────────────────────────────────────
        self.odm_progress = QProgressBar()
        self.odm_progress.setRange(0, 100)
        layout.addWidget(self.odm_progress)

        self.odm_status = QLabel("Prêt.")
        self.odm_status.setWordWrap(True)
        layout.addWidget(self.odm_status)

        # ── Boutons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_odm_launch = QPushButton("▶ Lancer le traitement ODM")
        self.btn_odm_launch.setStyleSheet(
            "background-color: #1976D2; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_odm_launch.clicked.connect(self._odm_launch)
        btn_row.addWidget(self.btn_odm_launch)

        self.btn_odm_stop = QPushButton("■ Arrêter")
        self.btn_odm_stop.setEnabled(False)
        self.btn_odm_stop.clicked.connect(self._odm_stop)
        btn_row.addWidget(self.btn_odm_stop)
        layout.addLayout(btn_row)

        self.btn_odm_load = QPushButton("📥 Charger l'orthophoto dans QGIS")
        self.btn_odm_load.setEnabled(False)
        self.btn_odm_load.setStyleSheet(
            "background-color: #388E3C; color: white; padding: 5px;"
        )
        self.btn_odm_load.clicked.connect(self._odm_load_result)
        layout.addWidget(self.btn_odm_load)

        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return widget

    # ──────────────────────────────────────────────────────────────────────
    # Onglet 1 — Détection YOLO / ONNX
    # ──────────────────────────────────────────────────────────────────────

    def _build_yolo_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Couche raster ──────────────────────────────────────────────
        grp_layer = QGroupBox("🗺️ Couche raster source")
        ll = QVBoxLayout(grp_layer)

        if QGIS_AVAILABLE:
            self.yolo_layer_combo = QgsMapLayerComboBox()
            self.yolo_layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
            ll.addWidget(self.yolo_layer_combo)
        else:
            self.yolo_layer_combo = QComboBox()
            ll.addWidget(self.yolo_layer_combo)

        self.btn_refresh_layers = QPushButton("Actualiser les couches")
        self.btn_refresh_layers.clicked.connect(self._yolo_refresh_layers)
        ll.addWidget(self.btn_refresh_layers)

        manual_row = QHBoxLayout()
        self.yolo_raster_path = QLineEdit()
        self.yolo_raster_path.setPlaceholderText("Ou chemin fichier raster…")
        btn_browse_raster = QPushButton("Parcourir")
        btn_browse_raster.clicked.connect(self._yolo_browse_raster)
        manual_row.addWidget(self.yolo_raster_path)
        manual_row.addWidget(btn_browse_raster)
        ll.addLayout(manual_row)
        layout.addWidget(grp_layer)

        # ── Classes à détecter ─────────────────────────────────────────
        grp_classes = QGroupBox("🎯 Classes à détecter")
        cl = QVBoxLayout(grp_classes)

        self.chk_mangroves = QCheckBox("🌿 Mangroves")
        self.chk_mangroves.setChecked(True)
        self.chk_trees     = QCheckBox("🌳 Arbres fruitiers")
        self.chk_trees.setChecked(True)
        self.chk_vehicles  = QCheckBox("🚗 Véhicules (voitures, camions, bus)")
        self.chk_vehicles.setChecked(True)
        self.chk_buildings = QCheckBox("🏠 Bâtiments")
        self.chk_buildings.setChecked(True)

        cl.addWidget(self.chk_mangroves)
        cl.addWidget(self.chk_trees)
        cl.addWidget(self.chk_vehicles)
        cl.addWidget(self.chk_buildings)
        layout.addWidget(grp_classes)

        # ── Sélection du modèle ────────────────────────────────────────
        grp_model = QGroupBox("🤖 Modèle de détection")
        ml = QFormLayout(grp_model)

        self.yolo_backend_combo = QComboBox()
        self.yolo_backend_combo.addItems([
            "Ultralytics (YOLOv8 / .pt / .onnx ultralytics)",
            "ONNX Runtime (modèles ONNX Zoo)",
        ])
        self.yolo_backend_combo.currentIndexChanged.connect(self._yolo_backend_changed)
        ml.addRow("Moteur:", self.yolo_backend_combo)

        # ── Sous-widget Ultralytics ──
        self.yolo_ultralytics_widget = QWidget()
        ul_layout = QFormLayout(self.yolo_ultralytics_widget)
        ul_layout.setContentsMargins(0, 0, 0, 0)

        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.addItems([
            "YOLOv8n (nano, rapide)",
            "YOLOv8s (small)",
            "YOLOv8m (medium, recommandé)",
            "YOLOv8l (large)",
            "YOLOv8x (extra-large)",
            "Modèle .pt personnalisé…",
            "Modèle .onnx (export ultralytics)…",
        ])
        self.yolo_model_combo.setCurrentIndex(2)
        self.yolo_model_combo.currentIndexChanged.connect(self._yolo_model_changed)
        ul_layout.addRow("Modèle YOLOv8:", self.yolo_model_combo)

        custom_row = QHBoxLayout()
        self.yolo_custom_path = QLineEdit()
        self.yolo_custom_path.setPlaceholderText("Chemin .pt ou .onnx personnalisé…")
        self.yolo_custom_path.setEnabled(False)
        self.btn_browse_model = QPushButton("Parcourir")
        self.btn_browse_model.setEnabled(False)
        self.btn_browse_model.clicked.connect(self._yolo_browse_model)
        custom_row.addWidget(self.yolo_custom_path)
        custom_row.addWidget(self.btn_browse_model)
        ul_layout.addRow("Fichier modèle:", custom_row)

        # ── Sous-widget ONNX Runtime ──
        self.yolo_onnx_widget = QWidget()
        onnx_layout = QFormLayout(self.yolo_onnx_widget)
        onnx_layout.setContentsMargins(0, 0, 0, 0)

        self.onnx_zoo_combo = QComboBox()
        self.onnx_zoo_combo.addItem("— Sélectionner un modèle zoo —")
        for name in list_zoo_models():
            self.onnx_zoo_combo.addItem(name)
        self.onnx_zoo_combo.addItem("Modèle ONNX local…")
        self.onnx_zoo_combo.currentIndexChanged.connect(self._onnx_zoo_changed)
        onnx_layout.addRow("Modèle ONNX:", self.onnx_zoo_combo)

        onnx_file_row = QHBoxLayout()
        self.onnx_model_path = QLineEdit()
        self.onnx_model_path.setPlaceholderText("Chemin fichier .onnx…")
        btn_browse_onnx = QPushButton("Parcourir")
        btn_browse_onnx.clicked.connect(self._onnx_browse_model)
        onnx_file_row.addWidget(self.onnx_model_path)
        onnx_file_row.addWidget(btn_browse_onnx)
        onnx_layout.addRow("Fichier .onnx:", onnx_file_row)

        self.onnx_type_combo = QComboBox()
        self.onnx_type_combo.addItems([
            "yolov3", "yolov3_tiny", "ssd", "faster_rcnn", "generic",
        ])
        onnx_layout.addRow("Type modèle:", self.onnx_type_combo)

        onnx_dl_row = QHBoxLayout()
        self.btn_onnx_download = QPushButton("⬇ Télécharger modèle sélectionné")
        self.btn_onnx_download.setStyleSheet(
            "background-color: #5C6BC0; color: white;"
        )
        self.btn_onnx_download.clicked.connect(self._onnx_download_model)
        onnx_dl_row.addWidget(self.btn_onnx_download)
        onnx_layout.addRow("", onnx_dl_row)

        self.onnx_dl_progress = QProgressBar()
        self.onnx_dl_progress.setRange(0, 100)
        self.onnx_dl_progress.setValue(0)
        self.onnx_dl_progress.setVisible(False)
        onnx_layout.addRow("Téléch.:", self.onnx_dl_progress)

        ml.addRow("", self.yolo_ultralytics_widget)
        ml.addRow("", self.yolo_onnx_widget)
        self.yolo_onnx_widget.setVisible(False)

        layout.addWidget(grp_model)

        # ── Paramètres ─────────────────────────────────────────────────
        grp_params = QGroupBox("🔧 Paramètres de détection")
        pp = QFormLayout(grp_params)

        conf_row = QHBoxLayout()
        self.yolo_conf_slider = QSlider(Qt.Horizontal)
        self.yolo_conf_slider.setRange(5, 95)
        self.yolo_conf_slider.setValue(25)
        self.yolo_conf_label = QLabel("0.25")
        self.yolo_conf_label.setMinimumWidth(35)
        self.yolo_conf_slider.valueChanged.connect(
            lambda v: self.yolo_conf_label.setText(f"{v/100:.2f}")
        )
        conf_row.addWidget(self.yolo_conf_slider)
        conf_row.addWidget(self.yolo_conf_label)
        pp.addRow("Confiance min.:", conf_row)

        iou_row = QHBoxLayout()
        self.yolo_iou_slider = QSlider(Qt.Horizontal)
        self.yolo_iou_slider.setRange(10, 90)
        self.yolo_iou_slider.setValue(45)
        self.yolo_iou_label = QLabel("0.45")
        self.yolo_iou_label.setMinimumWidth(35)
        self.yolo_iou_slider.valueChanged.connect(
            lambda v: self.yolo_iou_label.setText(f"{v/100:.2f}")
        )
        iou_row.addWidget(self.yolo_iou_slider)
        iou_row.addWidget(self.yolo_iou_label)
        pp.addRow("Seuil IoU (NMS):", iou_row)

        self.yolo_tile_combo = QComboBox()
        self.yolo_tile_combo.addItems(["512", "1024", "2048"])
        self.yolo_tile_combo.setCurrentIndex(1)
        pp.addRow("Taille tuile (px):", self.yolo_tile_combo)

        layout.addWidget(grp_params)

        # ── Sortie ─────────────────────────────────────────────────────
        grp_yolo_out = QGroupBox("📄 Fichier de sortie")
        yo = QHBoxLayout(grp_yolo_out)
        self.yolo_output_path = QLineEdit()
        self.yolo_output_path.setPlaceholderText("GeoJSON de sortie…")
        btn_browse_yolo_out = QPushButton("Parcourir")
        btn_browse_yolo_out.clicked.connect(self._yolo_browse_output)
        yo.addWidget(self.yolo_output_path)
        yo.addWidget(btn_browse_yolo_out)
        layout.addWidget(grp_yolo_out)

        # ── Progression ────────────────────────────────────────────────
        self.yolo_progress = QProgressBar()
        self.yolo_progress.setRange(0, 100)
        layout.addWidget(self.yolo_progress)

        self.yolo_status = QLabel("Prêt.")
        self.yolo_status.setWordWrap(True)
        layout.addWidget(self.yolo_status)

        # ── Boutons ────────────────────────────────────────────────────
        btn_yolo_row = QHBoxLayout()
        self.btn_yolo_launch = QPushButton("▶ Lancer la détection")
        self.btn_yolo_launch.setStyleSheet(
            "background-color: #E65100; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_yolo_launch.clicked.connect(self._yolo_launch)
        btn_yolo_row.addWidget(self.btn_yolo_launch)

        self.btn_yolo_stop = QPushButton("■ Arrêter")
        self.btn_yolo_stop.setEnabled(False)
        self.btn_yolo_stop.clicked.connect(self._yolo_stop)
        btn_yolo_row.addWidget(self.btn_yolo_stop)
        layout.addLayout(btn_yolo_row)

        self.btn_yolo_load = QPushButton("📥 Charger les résultats dans QGIS")
        self.btn_yolo_load.setEnabled(False)
        self.btn_yolo_load.setStyleSheet(
            "background-color: #388E3C; color: white; padding: 5px;"
        )
        self.btn_yolo_load.clicked.connect(self._yolo_load_result)
        layout.addWidget(self.btn_yolo_load)

        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return widget

    # ──────────────────────────────────────────────────────────────────────
    # Onglet 2 — Configuration API
    # ──────────────────────────────────────────────────────────────────────

    def _build_config_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Clés API ──────────────────────────────────────────────────
        grp_api = QGroupBox("🔑 Clés API (Intelligence Artificielle)")
        fa = QFormLayout(grp_api)

        providers = [
            ("openai",    "OpenAI (GPT-4)",    "sk-…"),
            ("gemini",    "Google Gemini",      "AIza…"),
            ("mistral",   "Mistral AI",         "Clé Mistral…"),
            ("deepseek",  "Deepseek",           "Clé Deepseek…"),
        ]
        self._api_fields = {}
        for pid, label, placeholder in providers:
            row = QHBoxLayout()
            field = QLineEdit()
            field.setPlaceholderText(placeholder)
            field.setEchoMode(QLineEdit.Password)

            btn_show = QToolButton()
            btn_show.setText("👁")
            btn_show.setCheckable(True)
            btn_show.toggled.connect(
                lambda checked, f=field: f.setEchoMode(
                    QLineEdit.Normal if checked else QLineEdit.Password
                )
            )

            btn_test = QPushButton("Tester")
            btn_test.setFixedWidth(60)
            btn_test.clicked.connect(
                lambda _, p=pid, f=field: self._test_api_key(p, f.text())
            )

            row.addWidget(field)
            row.addWidget(btn_show)
            row.addWidget(btn_test)
            fa.addRow(label + ":", row)
            self._api_fields[pid] = field

        layout.addWidget(grp_api)

        # ── Configuration ODM par défaut ──────────────────────────────
        grp_odm_cfg = QGroupBox("🌐 Serveur ODM par défaut")
        fo = QFormLayout(grp_odm_cfg)

        self.cfg_odm_url = QLineEdit()
        self.cfg_odm_url.setPlaceholderText("http://localhost:3000")
        fo.addRow("URL NodeODM:", self.cfg_odm_url)

        self.cfg_odm_token = QLineEdit()
        self.cfg_odm_token.setEchoMode(QLineEdit.Password)
        self.cfg_odm_token.setPlaceholderText("Token (optionnel)")
        fo.addRow("Token:", self.cfg_odm_token)
        layout.addWidget(grp_odm_cfg)

        # ── Préférences ───────────────────────────────────────────────
        grp_pref = QGroupBox("📂 Préférences générales")
        fpr = QFormLayout(grp_pref)

        out_dir_row = QHBoxLayout()
        self.cfg_output_dir = QLineEdit()
        self.cfg_output_dir.setPlaceholderText("Dossier de sortie par défaut…")
        btn_cfg_browse = QPushButton("Parcourir")
        btn_cfg_browse.clicked.connect(self._cfg_browse_output)
        out_dir_row.addWidget(self.cfg_output_dir)
        out_dir_row.addWidget(btn_cfg_browse)
        fpr.addRow("Dossier sortie:", out_dir_row)

        self.cfg_format_combo = QComboBox()
        self.cfg_format_combo.addItems(["GeoJSON", "Shapefile", "GeoPackage"])
        fpr.addRow("Format vecteur:", self.cfg_format_combo)
        layout.addWidget(grp_pref)

        # ── Boutons save/reset ────────────────────────────────────────
        btn_cfg_row = QHBoxLayout()
        btn_save = QPushButton("💾 Sauvegarder la configuration")
        btn_save.setStyleSheet(
            "background-color: #1976D2; color: white; padding: 5px;"
        )
        btn_save.clicked.connect(self._cfg_save)

        btn_reset = QPushButton("🔄 Réinitialiser")
        btn_reset.clicked.connect(self._cfg_reset)

        btn_cfg_row.addWidget(btn_save)
        btn_cfg_row.addWidget(btn_reset)
        layout.addLayout(btn_cfg_row)

        self.cfg_status = QLabel("")
        self.cfg_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cfg_status)

        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return widget

    # ─────────────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────────────

    def switch_to_tab(self, index):
        self.tabs.setCurrentIndex(index)

    # ─────────────────────────────────────────────────────────────────────
    # ODM slots
    # ─────────────────────────────────────────────────────────────────────

    def _odm_add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Sélectionner des images drone", "",
            "Images (*.jpg *.jpeg *.JPG *.JPEG *.png *.PNG *.tif *.tiff *.TIF *.TIFF)"
        )
        existing = {self.odm_image_list.item(i).text()
                    for i in range(self.odm_image_list.count())}
        for p in paths:
            if p not in existing:
                self.odm_image_list.addItem(QListWidgetItem(p))
        self.odm_status.setText(f"{self.odm_image_list.count()} image(s) chargée(s).")

    def _odm_remove_images(self):
        for item in self.odm_image_list.selectedItems():
            self.odm_image_list.takeItem(self.odm_image_list.row(item))

    def _odm_clear_images(self):
        self.odm_image_list.clear()
        self.odm_status.setText("Liste effacée.")

    def _odm_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Dossier de sortie")
        if folder:
            self.odm_output_dir.setText(folder)

    def _odm_browse_dem(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "MNT de référence", "",
            "Raster (*.tif *.tiff *.img *.vrt *.asc)"
        )
        if path:
            self.odm_dem_path.setText(path)

    def _odm_browse_gcp(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Fichier GCP", "",
            "Fichiers GCP (*.txt *.csv *.gcp);;Tous (*)"
        )
        if path:
            self.odm_gcp_path.setText(path)

    def _odm_update_precision_label(self, value):
        self.odm_precision_label.setText(
            f"Résolution actuelle: {value:.1f} cm/px"
        )

    def _odm_preset_3cm(self):
        """Précision < 3 cm : ultra, 2 cm/px, 10 000 traits, maillage 3D."""
        self.odm_quality.setCurrentIndex(4)       # Ultra
        self.odm_gsd.setValue(2.0)
        self.odm_min_features.setValue(10000)
        self.odm_feature_type.setCurrentIndex(3)  # DSP-SIFT
        self.odm_matcher.setCurrentIndex(0)        # FLANN
        self.odm_use_3dmesh.setChecked(True)
        self.odm_dsm.setChecked(True)
        self.odm_dtm.setChecked(True)
        self.odm_status.setText(
            "Préréglage < 3 cm appliqué (GSD=2 cm, Ultra, DSP-SIFT, maillage 3D)."
        )

    def _odm_preset_6cm(self):
        """Précision < 6 cm : haute, 5 cm/px, 8 000 traits."""
        self.odm_quality.setCurrentIndex(3)       # Haute
        self.odm_gsd.setValue(5.0)
        self.odm_min_features.setValue(8000)
        self.odm_feature_type.setCurrentIndex(0)  # SIFT
        self.odm_matcher.setCurrentIndex(0)        # FLANN
        self.odm_use_3dmesh.setChecked(False)
        self.odm_dsm.setChecked(True)
        self.odm_status.setText(
            "Préréglage < 6 cm appliqué (GSD=5 cm, Haute, SIFT)."
        )

    def _odm_test_server(self):
        import requests
        url   = self.odm_url.text().rstrip("/")
        token = self.odm_token.text().strip()
        try:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(f"{url}/info", headers=headers, timeout=10)
            if resp.status_code == 200:
                info    = resp.json()
                version = info.get("version", "?")
                QMessageBox.information(
                    self, "Connexion réussie",
                    f"Serveur NodeODM connecté ✓\nVersion: {version}"
                )
            else:
                QMessageBox.warning(
                    self, "Erreur serveur",
                    f"Code HTTP: {resp.status_code}\n{resp.text[:200]}"
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Connexion impossible",
                f"Impossible de joindre le serveur:\n{e}"
            )

    def _odm_build_options(self) -> dict:
        quality_map = {0: "lowest", 1: "low", 2: "medium", 3: "high", 4: "ultra"}
        quality = quality_map.get(self.odm_quality.currentIndex(), "medium")

        feature_type_map = {0: "sift", 1: "orb", 2: "hahog", 3: "dspsift"}
        feature_type = feature_type_map.get(self.odm_feature_type.currentIndex(), "sift")

        matcher_map = {0: "flann", 1: "bruteforce", 2: "bow"}
        matcher = matcher_map.get(self.odm_matcher.currentIndex(), "flann")

        dense_map = {0: "mvs", 1: "openMVS", 2: "poisson"}
        # ODM doesn't have a direct option for this, but we use pc-quality
        _ = dense_map.get(self.odm_dense_method.currentIndex(), "mvs")

        opts = {
            "orthophoto-resolution": self.odm_gsd.value(),
            "quality":               quality,
            "pc-quality":            quality,
            "feature-quality":       quality,
            "feature-type":          feature_type,
            "matcher-type":          matcher,
            "min-num-features":      self.odm_min_features.value(),
            "mesh-size":             200000,
            "dsm":                   self.odm_dsm.isChecked(),
            "dtm":                   self.odm_dtm.isChecked(),
            "pc-classify":           self.odm_pc.isChecked(),
        }
        if self.odm_use_3dmesh.isChecked():
            opts["use-3dmesh"] = True
        return opts

    def _odm_launch(self):
        if self.odm_image_list.count() == 0:
            QMessageBox.warning(self, "Aucune image",
                                "Veuillez ajouter des images drone.")
            return

        server_url = self.odm_url.text().strip()
        if not server_url:
            QMessageBox.warning(self, "URL manquante",
                                "Entrez l'URL du serveur NodeODM.")
            return

        output_dir = self.odm_output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Dossier manquant",
                                "Choisissez un dossier de sortie.")
            return

        os.makedirs(output_dir, exist_ok=True)

        image_paths = [self.odm_image_list.item(i).text()
                       for i in range(self.odm_image_list.count())]

        gcp_path = self.odm_gcp_path.text().strip() or None

        self._odm_result_path = None
        self.btn_odm_load.setEnabled(False)
        self.btn_odm_launch.setEnabled(False)
        self.btn_odm_stop.setEnabled(True)
        self.odm_progress.setValue(0)
        self.odm_status.setText("Démarrage du traitement ODM…")

        self._odm_worker = ODMWorker(
            image_paths=image_paths,
            server_url=server_url,
            token=self.odm_token.text().strip(),
            options=self._odm_build_options(),
            output_dir=output_dir,
            gcp_path=gcp_path,
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
    def _odm_on_progress(self, v):
        self.odm_progress.setValue(v)

    @pyqtSlot(str)
    def _odm_on_status(self, msg):
        self.odm_status.setText(msg)

    @pyqtSlot(str)
    def _odm_on_finished(self, path):
        self._odm_result_path = path
        self.odm_status.setText(f"Terminé ✓ — Orthophoto: {path}")
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
        path = self._odm_result_path
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Fichier introuvable",
                                "L'orthophoto n'a pas été trouvée.")
            return
        if QGIS_AVAILABLE:
            name  = os.path.splitext(os.path.basename(path))[0]
            layer = QgsRasterLayer(path, name)
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.odm_status.setText(f"Couche '{name}' chargée.")
            else:
                QMessageBox.critical(self, "Erreur",
                                     "Impossible de charger la couche raster.")

    # ─────────────────────────────────────────────────────────────────────
    # YOLO / ONNX slots
    # ─────────────────────────────────────────────────────────────────────

    def _yolo_backend_changed(self, index):
        use_onnx = (index == 1)
        self.yolo_ultralytics_widget.setVisible(not use_onnx)
        self.yolo_onnx_widget.setVisible(use_onnx)

    def _yolo_refresh_layers(self):
        if not QGIS_AVAILABLE:
            return
        if isinstance(self.yolo_layer_combo, QComboBox):
            self.yolo_layer_combo.clear()
            for layer in QgsProject.instance().mapLayers().values():
                if hasattr(layer, "bandCount"):
                    self.yolo_layer_combo.addItem(layer.name(), layer.id())

    def _yolo_browse_raster(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un raster", "",
            "Rasters (*.tif *.tiff *.TIF *.TIFF *.img *.vrt *.jp2 *.png)"
        )
        if path:
            self.yolo_raster_path.setText(path)

    def _yolo_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Modèle YOLO", "",
            "Modèle (*.pt *.onnx)"
        )
        if path:
            self.yolo_custom_path.setText(path)

    def _yolo_browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer les détections",
            "detections.geojson",
            "GeoJSON (*.geojson);;Shapefile (*.shp)"
        )
        if path:
            self.yolo_output_path.setText(path)

    def _yolo_model_changed(self, index):
        custom_indices = {5, 6}  # .pt personnalisé, .onnx personnalisé
        is_custom = index in custom_indices
        self.yolo_custom_path.setEnabled(is_custom)
        self.btn_browse_model.setEnabled(is_custom)

    def _onnx_zoo_changed(self, index):
        names = list(list_zoo_models().keys())
        # index 0 = placeholder, 1..len(names) = zoo models, last = local
        if 1 <= index <= len(names):
            name = names[index - 1]
            # Auto-set type
            model_info = list_zoo_models()[name]
            mtype = model_info.get("type", "generic")
            idx = {"yolov3": 0, "yolov3_tiny": 1, "ssd": 2,
                   "faster_rcnn": 3, "generic": 4}.get(mtype, 4)
            self.onnx_type_combo.setCurrentIndex(idx)

    def _onnx_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Modèle ONNX", "", "ONNX (*.onnx)"
        )
        if path:
            self.onnx_model_path.setText(path)

    def _onnx_download_model(self):
        index = self.onnx_zoo_combo.currentIndex()
        names = list(list_zoo_models().keys())
        if index < 1 or index > len(names):
            QMessageBox.warning(self, "Sélection",
                                "Sélectionnez d'abord un modèle zoo.")
            return

        model_name = names[index - 1]
        dest_dir = os.path.join(self.plugin_dir, "yolo", "models")
        os.makedirs(dest_dir, exist_ok=True)

        self.onnx_dl_progress.setVisible(True)
        self.onnx_dl_progress.setValue(0)
        self.btn_onnx_download.setEnabled(False)

        self._onnx_dl_worker = ONNXDownloadWorker(model_name, dest_dir)
        self._onnx_dl_worker.progress_changed.connect(self.onnx_dl_progress.setValue)
        self._onnx_dl_worker.status_changed.connect(self.yolo_status.setText)
        self._onnx_dl_worker.finished_path.connect(self._onnx_on_dl_done)
        self._onnx_dl_worker.failed.connect(self._onnx_on_dl_failed)
        self._onnx_dl_worker.start()

    @pyqtSlot(str)
    def _onnx_on_dl_done(self, path):
        self.onnx_model_path.setText(path)
        self.onnx_dl_progress.setVisible(False)
        self.btn_onnx_download.setEnabled(True)
        QMessageBox.information(self, "Téléchargement terminé",
                                f"Modèle sauvegardé:\n{path}")

    @pyqtSlot(str)
    def _onnx_on_dl_failed(self, error):
        self.onnx_dl_progress.setVisible(False)
        self.btn_onnx_download.setEnabled(True)
        QMessageBox.critical(self, "Erreur téléchargement", error)

    def _get_yolo_model_path(self):
        idx = self.yolo_model_combo.currentIndex()
        model_names = [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
            "yolov8l.pt", "yolov8x.pt",
        ]
        if idx < len(model_names):
            return model_names[idx]
        return self.yolo_custom_path.text().strip()

    def _get_raster_path(self):
        manual = self.yolo_raster_path.text().strip()
        if manual and os.path.exists(manual):
            return manual
        if not QGIS_AVAILABLE:
            return ""
        if isinstance(self.yolo_layer_combo, QComboBox):
            lid = self.yolo_layer_combo.currentData()
            if lid:
                layer = QgsProject.instance().mapLayer(lid)
                if layer:
                    return layer.source()
        else:
            layer = self.yolo_layer_combo.currentLayer()
            if layer:
                return layer.source()
        return ""

    def _yolo_launch(self):
        raster_path = self._get_raster_path()
        if not raster_path:
            QMessageBox.warning(self, "Raster manquant",
                                "Sélectionnez une couche raster ou un fichier.")
            return

        # Determine backend
        use_onnx = (self.yolo_backend_combo.currentIndex() == 1)

        if use_onnx:
            model_path = self.onnx_model_path.text().strip()
            if not model_path:
                QMessageBox.warning(self, "Modèle manquant",
                                    "Sélectionnez ou téléchargez un modèle ONNX.")
                return
            model_backend = "onnxruntime"
            model_type = self.onnx_type_combo.currentText()
        else:
            model_path = self._get_yolo_model_path()
            if not model_path:
                QMessageBox.warning(self, "Modèle manquant",
                                    "Sélectionnez un modèle YOLO.")
                return
            model_backend = "ultralytics"
            model_type = "yolov8"

        output_path = self.yolo_output_path.text().strip()
        if not output_path:
            base = os.path.splitext(raster_path)[0]
            output_path = base + "_detections.geojson"
            self.yolo_output_path.setText(output_path)

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
                                "Sélectionnez au moins une classe à détecter.")
            return

        self._yolo_result_path = None
        self.btn_yolo_load.setEnabled(False)
        self.btn_yolo_launch.setEnabled(False)
        self.btn_yolo_stop.setEnabled(True)
        self.yolo_progress.setValue(0)
        self.yolo_status.setText("Démarrage de la détection…")

        self._yolo_worker = YOLOWorker(
            raster_path=raster_path,
            model_path=model_path,
            model_backend=model_backend,
            model_type=model_type,
            classes=classes,
            conf=self.yolo_conf_slider.value() / 100.0,
            iou=self.yolo_iou_slider.value() / 100.0,
            tile_size=int(self.yolo_tile_combo.currentText()),
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
            self.yolo_status.setText("Détection arrêtée.")
        self._yolo_set_idle()

    @pyqtSlot(int)
    def _yolo_on_progress(self, v):
        self.yolo_progress.setValue(v)

    @pyqtSlot(str)
    def _yolo_on_status(self, msg):
        self.yolo_status.setText(msg)

    @pyqtSlot(str)
    def _yolo_on_finished(self, path):
        self._yolo_result_path = path
        self.yolo_status.setText(f"Détection terminée ✓ — {path}")
        self.yolo_progress.setValue(100)
        self.btn_yolo_load.setEnabled(True)
        self._yolo_set_idle()

    @pyqtSlot(str)
    def _yolo_on_failed(self, error):
        self.yolo_status.setText(f"Erreur: {error}")
        QMessageBox.critical(self, "Erreur détection", f"{error}")
        self._yolo_set_idle()

    def _yolo_set_idle(self):
        self.btn_yolo_launch.setEnabled(True)
        self.btn_yolo_stop.setEnabled(False)

    def _yolo_load_result(self):
        path = self._yolo_result_path
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Fichier introuvable",
                                "Le fichier de résultats est introuvable.")
            return
        if QGIS_AVAILABLE:
            name  = os.path.splitext(os.path.basename(path))[0]
            layer = QgsVectorLayer(path, name, "ogr")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.yolo_status.setText(f"Couche '{name}' chargée.")
            else:
                QMessageBox.critical(self, "Erreur",
                                     "Impossible de charger la couche de résultats.")

    # ─────────────────────────────────────────────────────────────────────
    # Config slots
    # ─────────────────────────────────────────────────────────────────────

    def _load_config_to_ui(self):
        cfg = load_config(self.plugin_dir)
        for pid, field in self._api_fields.items():
            field.setText(cfg.get(f"{pid}_key", ""))
        self.cfg_odm_url.setText(cfg.get("odm_url", ""))
        self.cfg_odm_token.setText(cfg.get("odm_token", ""))
        self.cfg_output_dir.setText(cfg.get("output_dir", ""))
        fmt_map = {"geojson": 0, "shapefile": 1, "geopackage": 2}
        self.cfg_format_combo.setCurrentIndex(
            fmt_map.get(cfg.get("output_format", "geojson"), 0)
        )
        # Apply defaults to ODM tab
        if cfg.get("odm_url"):
            self.odm_url.setText(cfg["odm_url"])
        if cfg.get("odm_token"):
            self.odm_token.setText(cfg["odm_token"])

    def _cfg_save(self):
        fmt_map = {0: "geojson", 1: "shapefile", 2: "geopackage"}
        cfg = {
            "odm_url":       self.cfg_odm_url.text(),
            "odm_token":     self.cfg_odm_token.text(),
            "output_dir":    self.cfg_output_dir.text(),
            "output_format": fmt_map.get(self.cfg_format_combo.currentIndex(), "geojson"),
        }
        for pid, field in self._api_fields.items():
            cfg[f"{pid}_key"] = field.text()
        try:
            save_config(self.plugin_dir, cfg)
            self.cfg_status.setText("✓ Configuration sauvegardée.")
            self.cfg_status.setStyleSheet("color: green;")
            # Sync ODM tab
            if cfg["odm_url"]:
                self.odm_url.setText(cfg["odm_url"])
            if cfg["odm_token"]:
                self.odm_token.setText(cfg["odm_token"])
        except Exception as e:
            self.cfg_status.setText(f"Erreur: {e}")
            self.cfg_status.setStyleSheet("color: red;")

    def _cfg_reset(self):
        for field in self._api_fields.values():
            field.clear()
        self.cfg_odm_url.setText("http://localhost:3000")
        self.cfg_odm_token.clear()
        self.cfg_output_dir.clear()
        self.cfg_format_combo.setCurrentIndex(0)
        self.cfg_status.setText("Configuration réinitialisée.")
        self.cfg_status.setStyleSheet("color: #555;")

    def _cfg_browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Dossier de sortie par défaut"
        )
        if folder:
            self.cfg_output_dir.setText(folder)

    def _test_api_key(self, provider: str, key: str):
        if not key:
            QMessageBox.warning(self, "Clé vide",
                                f"Entrez d'abord la clé {provider}.")
            return
        # Simple test calls per provider
        try:
            if provider == "openai":
                import urllib.request, json as _json
                req = urllib.request.Request(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = _json.loads(resp.read())
                    n = len(data.get("data", []))
                    QMessageBox.information(
                        self, "OpenAI OK",
                        f"Connexion réussie ✓\n{n} modèles disponibles."
                    )

            elif provider == "gemini":
                import urllib.request, json as _json
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = _json.loads(resp.read())
                    n = len(data.get("models", []))
                    QMessageBox.information(
                        self, "Gemini OK",
                        f"Connexion réussie ✓\n{n} modèles disponibles."
                    )
            else:
                QMessageBox.information(
                    self, "Test non implémenté",
                    f"Test automatique non disponible pour {provider}.\n"
                    "La clé a été enregistrée, elle sera validée lors de la première utilisation."
                )
        except Exception as e:
            QMessageBox.critical(self, f"Erreur {provider}",
                                 f"Test échoué:\n{e}")
