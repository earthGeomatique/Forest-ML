import os
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt

from .forestdl_panel import ForestDLPanel


class ForestDLPlugin:
    def __init__(self, iface):
        self.iface      = iface
        self.panel      = None
        self.toolbar    = None
        self.actions    = []
        self.menu_name  = "ForestDL"
        self.plugin_dir = os.path.dirname(__file__)

    def initGui(self):
        self.toolbar = self.iface.addToolBar("ForestDL")
        self.toolbar.setObjectName("ForestDLToolBar")

        def _action(icon_file, label, slot, checkable=False):
            icon_path = os.path.join(self.plugin_dir, "icons", icon_file)
            act = QAction(QIcon(icon_path), label, self.iface.mainWindow())
            act.setCheckable(checkable)
            act.triggered.connect(slot)
            self.toolbar.addAction(act)
            self.iface.addPluginToMenu(self.menu_name, act)
            self.actions.append(act)
            return act

        self.action_panel = _action(
            "icon.png",
            "ForestDL — Ouvrir le panneau",
            self.toggle_panel,
            checkable=True,
        )
        self.action_odm = _action(
            "odm.png",
            "ODM — Traitement drone / Orthophoto",
            self.open_odm_tab,
        )
        self.action_yolo = _action(
            "yolo.png",
            "YOLO — Détection d'objets",
            self.open_yolo_tab,
        )
        self.action_config = _action(
            "icon.png",
            "Configuration API (clés OpenAI, Gemini, Mistral, Deepseek)",
            self.open_config_tab,
        )

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

    def open_config_tab(self):
        if self.panel:
            self.panel.show()
            self.panel.switch_to_tab(2)
            self.action_panel.setChecked(True)

    def _panel_visibility_changed(self, visible):
        self.action_panel.setChecked(visible)
