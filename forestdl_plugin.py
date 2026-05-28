import os
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt

from .forestdl_panel import ForestDLPanel


class ForestDLPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.panel = None
        self.toolbar = None
        self.actions = []
        self.menu_name = "ForestDL"
        self.plugin_dir = os.path.dirname(__file__)

    def initGui(self):
        # Create toolbar
        self.toolbar = self.iface.addToolBar("ForestDL")
        self.toolbar.setObjectName("ForestDLToolBar")

        # Main action: open/close panel
        icon_path = os.path.join(self.plugin_dir, "icons", "icon.png")
        self.action_panel = QAction(
            QIcon(icon_path),
            "ForestDL - Orthophoto & Détection",
            self.iface.mainWindow()
        )
        self.action_panel.setCheckable(True)
        self.action_panel.triggered.connect(self.toggle_panel)
        self.toolbar.addAction(self.action_panel)
        self.iface.addPluginToMenu(self.menu_name, self.action_panel)
        self.actions.append(self.action_panel)

        # ODM action shortcut
        odm_icon = os.path.join(self.plugin_dir, "icons", "odm.png")
        self.action_odm = QAction(
            QIcon(odm_icon),
            "ODM - Créer Orthophoto",
            self.iface.mainWindow()
        )
        self.action_odm.triggered.connect(self.open_odm_tab)
        self.toolbar.addAction(self.action_odm)
        self.iface.addPluginToMenu(self.menu_name, self.action_odm)
        self.actions.append(self.action_odm)

        # YOLO action shortcut
        yolo_icon = os.path.join(self.plugin_dir, "icons", "yolo.png")
        self.action_yolo = QAction(
            QIcon(yolo_icon),
            "YOLO - Détecter Objets",
            self.iface.mainWindow()
        )
        self.action_yolo.triggered.connect(self.open_yolo_tab)
        self.toolbar.addAction(self.action_yolo)
        self.iface.addPluginToMenu(self.menu_name, self.action_yolo)
        self.actions.append(self.action_yolo)

        # Create dockable panel
        self.panel = ForestDLPanel(self.iface)
        self.iface.mainWindow().addDockWidget(Qt.RightDockWidgetArea, self.panel)
        self.panel.hide()
        self.panel.visibilityChanged.connect(self._panel_visibility_changed)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.menu_name, action)
            self.iface.removeToolBarIcon(action)
        if self.toolbar:
            del self.toolbar
        if self.panel:
            self.iface.mainWindow().removeDockWidget(self.panel)
            self.panel.deleteLater()

    def toggle_panel(self, checked):
        if self.panel:
            self.panel.setVisible(checked)

    def open_odm_tab(self):
        if self.panel:
            self.panel.show()
            self.panel.switch_to_tab(0)
            self.action_panel.setChecked(True)

    def open_yolo_tab(self):
        if self.panel:
            self.panel.show()
            self.panel.switch_to_tab(1)
            self.action_panel.setChecked(True)

    def _panel_visibility_changed(self, visible):
        self.action_panel.setChecked(visible)
