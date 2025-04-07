# Structure du plugin ForestDL pour QGIS

## 1. Organisation des fichiers

```
ForestDL/
├── __init__.py                      # Point d'entrée du plugin
├── metadata.txt                     # Métadonnées du plugin
├── forest_dl.py                     # Classe principale du plugin
├── forest_dl_dialog_base.ui         # Interface principale (Qt Designer)
├── resources.qrc                    # Fichier de ressources (icônes, etc.)
├── resources.py                     # Fichier de ressources compilé
├── README.md                        # Documentation
├── LICENSE                          # Licence du plugin
├── icon.png                         # Icône du plugin
│
├── core/                            # Module principal
│   ├── __init__.py
│   ├── config.py                    # Gestion de la configuration
│   ├── data_manager.py              # Gestion des données
│   └── utils.py                     # Fonctions utilitaires
│
├── ui/                              # Interfaces utilisateur
│   ├── __init__.py
│   ├── api_config_dialog.py         # Configuration des API
│   ├── api_config_dialog_base.ui
│   ├── processing_dialog.py         # Traitement d'images
│   ├── processing_dialog_base.ui
│   ├── deeplearning_dialog.py       # Fonctionnalités de deep learning
│   ├── deeplearning_dialog_base.ui
│   ├── allometry_dialog.py          # Calculs allométriques
│   ├── allometry_dialog_base.ui
│   ├── prompt_dialog.py             # Console de prompts
│   └── prompt_dialog_base.ui
│
├── processing/                      # Modules de traitement d'images
│   ├── __init__.py
│   ├── preprocessing.py             # Prétraitement d'images
│   ├── segmentation.py              # Segmentation d'arbres
│   ├── classification.py            # Classification d'espèces
│   └── feature_extraction.py        # Extraction de caractéristiques
│
├── deeplearning/                    # Modules de deep learning
│   ├── __init__.py
│   ├── api_connector.py             # Connecteurs API pour LLM
│   ├── detection_models.py          # Modèles de détection
│   ├── estimation_models.py         # Modèles d'estimation
│   └── model_manager.py             # Gestion des modèles
│
├── allometry/                       # Modules de calculs forestiers
│   ├── __init__.py
│   ├── biomass.py                   # Calculs de biomasse
│   ├── carbon.py                    # Calculs de stock de carbone
│   ├── density.py                   # Analyse de densité
│   └── equations.py                 # Équations allométriques
│
├── output/                          # Modules de génération de résultats
│   ├── __init__.py
│   ├── csv_export.py                # Export CSV
│   ├── shapefile_export.py          # Export Shapefile
│   ├── raster_export.py             # Export Raster
│   └── report_generator.py          # Génération de rapports
│
├── test/                            # Tests unitaires et d'intégration
│   ├── __init__.py
│   ├── test_processing.py
│   ├── test_deeplearning.py
│   └── test_allometry.py
│
└── icons/                           # Icônes et ressources graphiques
    ├── forestdl.png
    ├── processing.png
    ├── deeplearning.png
    ├── allometry.png
    └── api.png
```

## 2. Fichiers principaux

### 2.1 __init__.py

```python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ForestDL
                                 A QGIS plugin
 Deep Learning pour l'inventaire forestier et l'analyse des mangroves
                             -------------------
        begin                : 2025-03-20
        copyright            : (C) 2025
        email                : info@forestdl.org
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load ForestDL class from file forest_dl.py.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .forest_dl import ForestDL
    return ForestDL(iface)
```

### 2.2 metadata.txt

```ini
[general]
name=ForestDL
qgisMinimumVersion=3.28
description=Deep Learning pour l'inventaire forestier et l'analyse des mangroves
version=0.1
author=ForestDL Team
email=info@forestdl.org

about=Plugin QGIS pour l'analyse des forêts et mangroves par deep learning et photogrammétrie. Combine les fonctionnalités du Semi-Automatic Classification Plugin (SCP) et de l'Orfeo ToolBox (OTB), tout en ajoutant des capacités spécifiques pour l'inventaire forestier et l'analyse des mangroves.

tracker=https://github.com/forestdl/forestdl/issues
repository=https://github.com/forestdl/forestdl
homepage=https://forestdl.org

category=Raster
icon=icon.png

experimental=True
deprecated=False

tags=forest,deep learning,remote sensing,mangrove,biomass,carbon,allometry,segmentation,classification
```

### 2.3 forest_dl.py

```python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ForestDL
                                 A QGIS plugin
 Deep Learning pour l'inventaire forestier et l'analyse des mangroves
                              -------------------
        begin                : 2025-03-20
        git sha              : $Format:%H$
        copyright            : (C) 2025
        email                : info@forestdl.org
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu

# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the dialog
from .ui.api_config_dialog import ApiConfigDialog
from .ui.processing_dialog import ProcessingDialog
from .ui.deeplearning_dialog import DeepLearningDialog
from .ui.allometry_dialog import AllometryDialog
from .ui.prompt_dialog import PromptDialog

import os.path


class ForestDL:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'ForestDL_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&ForestDL')
        self.toolbar = self.iface.addToolBar(u'ForestDL')
        self.toolbar.setObjectName(u'ForestDL')

        # Initialize dialogs
        self.api_config_dialog = None
        self.processing_dialog = None
        self.deeplearning_dialog = None
        self.allometry_dialog = None
        self.prompt_dialog = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('ForestDL', message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        
        # Main menu
        self.main_menu = QMenu(self.tr("ForestDL"))
        self.iface.mainWindow().menuBar().insertMenu(
            self.iface.firstRightStandardMenu().menuAction(), self.main_menu)
        
        # Add actions to the toolbar
        self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'forestdl.png'),
            text=self.tr(u'ForestDL'),
            callback=self.run,
            parent=self.iface.mainWindow())
            
        # Add submenu for processing
        self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'processing.png'),
            text=self.tr(u'Traitement d\'images'),
            callback=self.run_processing,
            parent=self.iface.mainWindow())
            
        # Add submenu for deep learning
        self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'deeplearning.png'),
            text=self.tr(u'Deep Learning'),
            callback=self.run_deeplearning,
            parent=self.iface.mainWindow())
            
        # Add submenu for allometry
        self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'allometry.png'),
            text=self.tr(u'Calculs allométriques'),
            callback=self.run_allometry,
            parent=self.iface.mainWindow())
            
        # Add submenu for API configuration
        self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'api.png'),
            text=self.tr(u'Configuration API'),
            callback=self.run_api_config,
            parent=self.iface.mainWindow())
            
        # Add submenu for prompt console
        self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'prompt.png'),
            text=self.tr(u'Console de prompts'),
            callback=self.run_prompt,
            parent=self.iface.mainWindow())

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&ForestDL'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def run(self):
        """Run method that performs all the real work"""
        # Create the dialog with elements (after translation) and keep reference
        # Only create dialogs if they don't exist
        if self.processing_dialog is None:
            self.processing_dialog = ProcessingDialog(self.iface)
        
        # show the dialog
        self.processing_dialog.show()
        # Run the dialog event loop
        result = self.processing_dialog.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass
    
    def run_processing(self):
        """Run the processing dialog"""
        if self.processing_dialog is None:
            self.processing_dialog = ProcessingDialog(self.iface)
        self.processing_dialog.show()
        result = self.processing_dialog.exec_()
        if result:
            pass
    
    def run_deeplearning(self):
        """Run the deep learning dialog"""
        if self.deeplearning_dialog is None:
            self.deeplearning_dialog = DeepLearningDialog(self.iface)
        self.deeplearning_dialog.show()
        result = self.deeplearning_dialog.exec_()
        if result:
            pass
    
    def run_allometry(self):
        """Run the allometry dialog"""
        if self.allometry_dialog is None:
            self.allometry_dialog = AllometryDialog(self.iface)
        self.allometry_dialog.show()
        result = self.allometry_dialog.exec_()
        if result:
            pass
    
    def run_api_config(self):
        """Run the API configuration dialog"""
        if self.api_config_dialog is None:
            self.api_config_dialog = ApiConfigDialog(self.iface)
        self.api_config_dialog.show()
        result = self.api_config_dialog.exec_()
        if result:
            pass
    
    def run_prompt(self):
        """Run the prompt console dialog"""
        if self.prompt_dialog is None:
            self.prompt_dialog = PromptDialog(self.iface)
        self.prompt_dialog.show()
        result = self.prompt_dialog.exec_()
        if result:
            pass
```

### 2.4 resources.qrc

```xml
<RCC>
    <qresource prefix="/plugins/forestdl" >
        <file>icons/forestdl.png</file>
        <file>icons/processing.png</file>
        <file>icons/deeplearning.png</file>
        <file>icons/allometry.png</file>
        <file>icons/api.png</file>
        <file>icons/prompt.png</file>
    </qresource>
</RCC>
```

## 3. Modules principaux

### 3.1 core/config.py

```python
# -*- coding: utf-8 -*-
"""
Configuration manager for ForestDL plugin.
"""

from qgis.core import QgsSettings
import os
import json

class ConfigManager:
    """Class to manage plugin configuration."""
    
    def __init__(self):
        """Constructor."""
        self.settings = QgsSettings()
        self.settings_prefix = "ForestDL/"
        
        # Default configuration
        self.default_config = {
            "api_keys": {
                "openai": "",
                "gemini": "",
                "mistral": "",
                "deepseek": ""
            },
            "models": {
                "openai": "gpt-4",
                "gemini": "gemini-pro",
                "mistral": "mistral-large",
                "deepseek": "deepseek-coder"
            },
            "processing": {
                "segmentation_method": "watershed",
                "classification_method": "random_forest",
                "feature_extraction_method": "spectral"
            },
            "allometry": {
                "biomass_equation": "chave2014",
                "carbon_factor": 0.47
            }
        }
        
        # Load configuration
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from QgsSettings."""
        config = {}
        
        # Try to load from settings
        config_str = self.settings.value(self.settings_prefix + "config", "")
        if config_str:
            try:
                config = json.loads(config_str)
            except json.JSONDecodeError:
                config = self.default_config
        else:
            config = self.default_config
            
        return config
    
    def save_config(self):
        """Save configuration to QgsSettings."""
        config_str = json.dumps(self.config)
        self.settings.setValue(self.settings_prefix + "config", config_str)
    
    def get_api_key(self, api_name):
        """Get API key for the specified API."""
        return self.config["api_keys"].get(api_name, "")
    
    def set_api_key(self, api_name, api_key):
        """Set API key for the specified API."""
        self.config["api_keys"][api_name] = api_key
        self.save_config()
    
    def get_model(self, api_name):
        """Get model for the specified API."""
        return self.config["models"].get(api_name, "")
    
    def set_model(self, api_name, model):
        """Set model for the specified API."""
        self.config["models"][api_name] = model
        self.save_config()
    
    def get_processing_param(self, param_name):
        """Get processing parameter."""
        return self.config["processing"].get(param_name, "")
    
    def set_processing_param(self, param_name, value):
        """Set processing parameter."""
        self.config["processing"][param_name] = value
        self.save_config()
    
    def get_allometry_param(self, param_name):
        """Get allometry parameter."""
        return self.config["allometry"].get(param_name, "")
    
    def set_allometry_param(self, param_name, value):
        """Set allometry parameter."""
        self.config["allometry"][param_name] = value
        self.save_config()
```

### 3.2 core/data_manager.py

```python
# -*- coding: utf-8 -*-
"""
Data manager for ForestDL plugin.
"""

from qgis.core import (QgsProject, QgsRasterLayer, QgsVectorLayer, 
                      QgsCoordinateReferenceSystem, QgsCoordinateTransform)
import os

class DataManager:
    """Class to manage data for ForestDL plugin."""
    
    def __init__(self, iface):
        """Constructor.
        
        :param iface: QGIS interface
        :type iface: QgsInterface
        """
        self.iface = iface
        self.project = QgsProject.instance()
        
    def load_raster(self, file_path, layer_name=None):
        """Load raster layer from file.
        
        :param file_path: Path to raster file
        :type file_path: str
        :param layer_name: Name for the layer (optional)
        :type layer_name: str
        
        :returns: Loaded raster layer
        :rtype: QgsRasterLayer
        """
        if not os.path.exists(file_path):
            return None
            
        if layer_name is None:
            layer_name = os.path.basename(file_path)
            
        layer = QgsRasterLayer(file_path, layer_name)
        
        if not layer.isValid():
            return None
            
        self.project.addMapLayer(layer)
        return layer
    
    def load_vector(self, file_path, layer_name=None):
        """Load vector layer from file.
        
        :param file_path: Path to vector file
        :type file_path: str
        :param layer_name: Name for the layer (optional)
        :type layer_name: str
        
        :returns: Loaded vector layer
        :rtype: QgsVectorLayer
        """
        if not os.path.exists(file_path):
            return None
            
        if layer_name is None:
            layer_name = os.path.basename(file_path)
            
        layer = QgsVectorLayer(file_path, layer_name, "ogr")
        
        if not layer.isValid():
            return None
            
        self.project.addMapLayer(layer)
        return layer
    
    def get_active_layer(self):
        """Get active layer.
        
        :returns: Active layer
        :rtype: QgsMapLayer
        """
        return self.iface.activeLayer()
    
    def get_layers_by_type(self, layer_type):
        """Get layers by type.
        
        :param layer_type: Layer type (0: vector, 1: raster)
        :type layer_type: int
        
        :returns: List of layers
        :rtype: list
        """
        layers = []
        for layer in self.project.mapLayers().values():
            if layer_type == 0 and layer.type() == 0:  # Vector
                layers.append(layer)
            elif layer_type == 1 and layer.type() == 1:  # Raster
                layers.append(layer)
        return layers
    
    def create_temp_layer(self, layer_type, name, crs=None):
        """Create temporary layer.
        
        :param layer_type: Layer type (point, line, polygon)
        :type layer_type: str
        :param name: Layer name
        :type name: str
        :param crs: Coordinate reference system (optional)
        :type crs: QgsCoordinateReferenceSystem
        
        :returns: Created layer
        :rtype: QgsVectorLayer
        """
        if crs is None:
            crs = QgsCoordinateReferenceSystem("EPSG:4326")
            
        layer = QgsVectorLayer(f"{layer_type}?crs={crs.authid()}", name, "memory")
        
        if not layer.isValid():
            return None
            
        self.project.addMapLayer(layer)
        return layer
    
    def save_layer(self, layer, file_path):
        """Save layer to file.
        
        :param layer: Layer to save
        :type layer: QgsMapLayer
        :param file_path: Path to save file
        :type file_path: str
        
        :returns: Success
        :rtype: bool
        """
        if layer.type() == 0:  # Vector
            error = QgsVectorFileWriter.writeAsVectorFormat(
                layer, file_path, "UTF-8", layer.crs(), "ESRI Shapefile")
            return error[0] == QgsVectorFileWriter.NoError
        elif layer.type() == 1:  # Raster
            # For raster, we need to use GDAL directly
            # This is a simplified version
            provider = layer.dataProvider()
            file_writer = QgsRasterFileWriter(file_path)
            pipe = QgsRasterPipe()
            pipe.set(provider.clone())
            file_writer.writeRaster(
                pipe, provider.xSize(), provider.ySize(), provider.extent(), provider.crs())
            return True
        return False
```

### 3.3 core/utils.py

```python
# -*- coding: utf-8 -*-
"""
Utility functions for ForestDL plugin.
"""

import os
import numpy as np
from qgis.core import (QgsMessageLog, QgsProject, QgsRasterLayer, 
                      QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY)
from qgis.PyQt.QtWidgets import QMessageBox

def log_message(message, level=0, tag="ForestDL"):
    """Log message to QGIS log.
    
    :param message: Message to log
    :type message: str
    :param level: Log level (0: info, 1: warning, 2: critical)
    :type level: int
    :param tag: Log tag
    :type tag: str
    """
    QgsMessageLog.logMessage(message, tag, level)

def show_message(title, message, level=0):
    """Show message box.
    
    :param title: Message box title
    :type title: str
    :param message: Message to show
    :type message: str
    :param level: Message level (0: info, 1: warning, 2: critical)
    :type level: int
    """
    if level == 0:
        QMessageBox.information(None, title, message)
    elif level == 1:
        QMessageBox.warning(None, title, message)
    elif level == 2:
        QMessageBox.critical(None, title, message)

def get_plugin_path():
    """Get plugin path.
    
    :returns: Plugin path
    :rtype: str
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_temp_path():
    """Get temporary path.
    
    :returns: Temporary path
    :rtype: str
    """
    temp_path = os.path.join(get_plugin_path(), "temp")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    return temp_path

def raster_to_array(raster_layer):
    """Convert raster layer to numpy array.
    
    :param raster_layer: Raster layer
    :type raster_layer: QgsRasterLayer
    
    :returns: Numpy array
    :rtype: numpy.ndarray
    """
    provider = raster_layer.dataProvider()
    block = provider.block(1, raster_layer.extent(), raster_layer.width(), raster_layer.height())
    
    array = np.zeros((raster_layer.height(), raster_layer.width()))
    for row in range(raster_layer.height()):
        for col in range(raster_layer.width()):
            array[row, col] = block.value(row, col)
    
    return array

def array_to_raster(array, extent, crs, output_path):
    """Convert numpy array to raster file.
    
    :param array: Numpy array
    :type array: numpy.ndarray
    :param extent: Extent
    :type extent: QgsRectangle
    :param crs: Coordinate reference system
    :type crs: QgsCoordinateReferenceSystem
    :param output_path: Output path
    :type output_path: str
    
    :returns: Success
    :rtype: bool
    """
    from osgeo import gdal, osr
    
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = array.shape
    
    # Create output raster
    out_raster = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    
    # Set extent
    x_min = extent.xMinimum()
    y_max = extent.yMaximum()
    x_res = extent.width() / cols
    y_res = extent.height() / rows
    out_raster.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    
    # Set CRS
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs.toWkt())
    out_raster.SetProjection(srs.ExportToWkt())
    
    # Write data
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()
    
    # Close dataset
    out_raster = None
    
    return os.path.exists(output_path)

def polygons_to_layer(polygons, crs, output_path=None):
    """Convert polygons to vector layer.
    
    :param polygons: List of polygons (list of list of QgsPointXY)
    :type polygons: list
    :param crs: Coordinate reference system
    :type crs: QgsCoordinateReferenceSystem
    :param output_path: Output path (optional)
    :type output_path: str
    
    :returns: Vector layer
    :rtype: QgsVectorLayer
    """
    if output_path:
        layer = QgsVectorLayer(f"Polygon?crs={crs.authid()}", "Polygons", "memory")
    else:
        layer = QgsVectorLayer(f"Polygon?crs={crs.authid()}", os.path.basename(output_path), "memory")
    
    provider = layer.dataProvider()
    
    # Add features
    features = []
    for polygon in polygons:
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPolygonXY([polygon]))
        features.append(feature)
    
    provider.addFeatures(features)
    layer.updateExtents()
    
    # Save to file if output_path is provided
    if output_path:
        error = QgsVectorFileWriter.writeAsVectorFormat(
            layer, output_path, "UTF-8", crs, "ESRI Shapefile")
        if error[0] == QgsVectorFileWriter.NoError:
            return QgsVectorLayer(output_path, os.path.basename(output_path), "ogr")
    
    return layer
```

## 4. Interfaces utilisateur de base

### 4.1 ui/api_config_dialog_base.ui

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>ApiConfigDialogBase</class>
 <widget class="QDialog" name="ApiConfigDialogBase">
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>400</width>
    <height>400</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>Configuration API</string>
  </property>
  <layout class="QVBoxLayout" name="verticalLayout">
   <item>
    <widget class="QTabWidget" name="tabWidget">
     <property name="currentIndex">
      <number>0</number>
     </property>
     <widget class="QWidget" name="tabOpenAI">
      <attribute name="title">
       <string>OpenAI</string>
      </attribute>
      <layout class="QVBoxLayout" name="verticalLayout_2">
       <item>
        <widget class="QLabel" name="labelOpenAIKey">
         <property name="text">
          <string>Clé API:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QLineEdit" name="lineEditOpenAIKey">
         <property name="echoMode">
          <enum>QLineEdit::Password</enum>
         </property>
         <property name="placeholderText">
          <string>Entrez votre clé API OpenAI</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QLabel" name="labelOpenAIModel">
         <property name="text">
          <string>Modèle:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QComboBox" name="comboBoxOpenAIModel">
         <item>
          <property name="text">
           <string>gpt-4</string>
          </property>
         </item>
         <item>
          <property name="text">
           <string>gpt-4-turbo</string>
          </property>
         </item>
         <item>
          <property name="text">
           <string>gpt-3.5-turbo</string>
          </property>
         </item>
        </widget>
       </item>
       <item>
        <spacer name="verticalSpacer">
         <property name="orientation">
          <enum>Qt::Vertical</enum>
         </property>
         <property name="sizeHint" stdset="0">
          <size>
           <width>20</width>
           <height>40</height>
          </size>
         </property>
        </spacer>
       </item>
      </layout>
     </widget>
     <widget class="QWidget" name="tabGemini">
      <attribute name="title">
       <string>Gemini</string>
      </attribute>
      <layout class="QVBoxLayout" name="verticalLayout_3">
       <item>
        <widget class="QLabel" name="labelGeminiKey">
         <property name="text">
          <string>Clé API:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QLineEdit" name="lineEditGeminiKey">
         <property name="echoMode">
          <enum>QLineEdit::Password</enum>
         </property>
         <property name="placeholderText">
          <string>Entrez votre clé API Gemini</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QLabel" name="labelGeminiModel">
         <property name="text">
          <string>Modèle:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QComboBox" name="comboBoxGeminiModel">
         <item>
          <property name="text">
           <string>gemini-pro</string>
          </property>
         </item>
         <item>
          <property name="text">
           <string>gemini-pro-vision</string>
          </property>
         </item>
        </widget>
       </item>
       <item>
        <spacer name="verticalSpacer_2">
         <property name="orientation">
          <enum>Qt::Vertical</enum>
         </property>
         <property name="sizeHint" stdset="0">
          <size>
           <width>20</width>
           <height>40</height>
          </size>
         </property>
        </spacer>
       </item>
      </layout>
     </widget>
     <widget class="QWidget" name="tabMistral">
      <attribute name="title">
       <string>Mistral</string>
      </attribute>
      <layout class="QVBoxLayout" name="verticalLayout_4">
       <item>
        <widget class="QLabel" name="labelMistralKey">
         <property name="text">
          <string>Clé API:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QLineEdit" name="lineEditMistralKey">
         <property name="echoMode">
          <enum>QLineEdit::Password</enum>
         </property>
         <property name="placeholderText">
          <string>Entrez votre clé API Mistral</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QLabel" name="labelMistralModel">
         <property name="text">
          <string>Modèle:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QComboBox" name="comboBoxMistralModel">
         <item>
          <property name="text">
           <string>mistral-large</string>
          </property>
         </item>
         <item>
          <property name="text">
           <string>mistral-medium</string>
          </property>
         </item>
         <item>
          <property name="text">
           <string>mistral-small</string>
          </property>
         </item>
        </widget>
       </item>
       <item>
        <spacer name="verticalSpacer_3">
         <property name="orientation">
          <enum>Qt::Vertical</enum>
         </property>
         <property name="sizeHint" stdset="0">
          <size>
           <width>20</width>
           <height>40</height>
          </size>
         </property>
        </spacer>
       </item>
      </layout>
     </widget>
     <widget class="QWidget" name="tabDeepseek">
      <attribute name="title">
       <string>Deepseek</string>
      </attribute>
      <layout class="QVBoxLayout" name="verticalLayout_5">
       <item>
        <widget class="QLabel" name="labelDeepseekKey">
         <property name="text">
          <string>Clé API:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QLineEdit" name="lineEditDeepseekKey">
         <property name="echoMode">
          <enum>QLineEdit::Password</enum>
         </property>
         <property name="placeholderText">
          <string>Entrez votre clé API Deepseek</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QLabel" name="labelDeepseekModel">
         <property name="text">
          <string>Modèle:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QComboBox" name="comboBoxDeepseekModel">
         <item>
          <property name="text">
           <string>deepseek-coder</string>
          </property>
         </item>
         <item>
          <property name="text">
           <string>deepseek-chat</string>
          </property>
         </item>
        </widget>
       </item>
       <item>
        <spacer name="verticalSpacer_4">
         <property name="orientation">
          <enum>Qt::Vertical</enum>
         </property>
         <property name="sizeHint" stdset="0">
          <size>
           <width>20</width>
           <height>40</height>
          </size>
         </property>
        </spacer>
       </item>
      </layout>
     </widget>
    </widget>
   </item>
   <item>
    <widget class="QDialogButtonBox" name="buttonBox">
     <property name="orientation">
      <enum>Qt::Horizontal</enum>
     </property>
     <property name="standardButtons">
      <set>QDialogButtonBox::Cancel|QDialogButtonBox::Ok</set>
     </property>
    </widget>
   </item>
  </layout>
 </widget>
 <resources/>
 <connections>
  <connection>
   <sender>buttonBox</sender>
   <signal>accepted()</signal>
   <receiver>ApiConfigDialogBase</receiver>
   <slot>accept()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>20</x>
     <y>20</y>
    </hint>
    <hint type="destinationlabel">
     <x>20</x>
     <y>20</y>
    </hint>
   </hints>
  </connection>
  <connection>
   <sender>buttonBox</sender>
   <signal>rejected()</signal>
   <receiver>ApiConfigDialogBase</receiver>
   <slot>reject()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>20</x>
     <y>20</y>
    </hint>
    <hint type="destinationlabel">
     <x>20</x>
     <y>20</y>
    </hint>
   </hints>
  </connection>
 </connections>
</ui>
```

### 4.2 ui/prompt_dialog_base.ui

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>PromptDialogBase</class>
 <widget class="QDialog" name="PromptDialogBase">
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>600</width>
    <height>500</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>Console de prompts</string>
  </property>
  <layout class="QVBoxLayout" name="verticalLayout">
   <item>
    <widget class="QSplitter" name="splitter">
     <property name="orientation">
      <enum>Qt::Vertical</enum>
     </property>
     <widget class="QWidget" name="layoutWidget">
      <layout class="QVBoxLayout" name="verticalLayout_2">
       <item>
        <widget class="QLabel" name="labelPrompt">
         <property name="text">
          <string>Entrez votre prompt:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QPlainTextEdit" name="plainTextEditPrompt">
         <property name="placeholderText">
          <string>Décrivez la tâche à effectuer...</string>
         </property>
        </widget>
       </item>
       <item>
        <layout class="QHBoxLayout" name="horizontalLayout">
         <item>
          <widget class="QLabel" name="labelModel">
           <property name="text">
            <string>Modèle:</string>
           </property>
          </widget>
         </item>
         <item>
          <widget class="QComboBox" name="comboBoxModel">
           <item>
            <property name="text">
             <string>OpenAI</string>
            </property>
           </item>
           <item>
            <property name="text">
             <string>Gemini</string>
            </property>
           </item>
           <item>
            <property name="text">
             <string>Mistral</string>
            </property>
           </item>
           <item>
            <property name="text">
             <string>Deepseek</string>
            </property>
           </item>
          </widget>
         </item>
         <item>
          <spacer name="horizontalSpacer">
           <property name="orientation">
            <enum>Qt::Horizontal</enum>
           </property>
           <property name="sizeHint" stdset="0">
            <size>
             <width>40</width>
             <height>20</height>
            </size>
           </property>
          </spacer>
         </item>
         <item>
          <widget class="QPushButton" name="pushButtonSend">
           <property name="text">
            <string>Envoyer</string>
           </property>
          </widget>
         </item>
        </layout>
       </item>
      </layout>
     </widget>
     <widget class="QWidget" name="layoutWidget1">
      <layout class="QVBoxLayout" name="verticalLayout_3">
       <item>
        <widget class="QLabel" name="labelResponse">
         <property name="text">
          <string>Réponse:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QTextEdit" name="textEditResponse">
         <property name="readOnly">
          <bool>true</bool>
         </property>
        </widget>
       </item>
      </layout>
     </widget>
    </widget>
   </item>
   <item>
    <widget class="QGroupBox" name="groupBoxTasks">
     <property name="title">
      <string>Tâches prédéfinies</string>
     </property>
     <layout class="QHBoxLayout" name="horizontalLayout_2">
      <item>
       <widget class="QPushButton" name="pushButtonSymbology">
        <property name="text">
         <string>Modifier symbologie</string>
        </property>
       </widget>
      </item>
      <item>
       <widget class="QPushButton" name="pushButtonClassify">
        <property name="text">
         <string>Classifier objets</string>
        </property>
       </widget>
      </item>
      <item>
       <widget class="QPushButton" name="pushButtonDigitize">
        <property name="text">
         <string>Numériser plan</string>
        </property>
       </widget>
      </item>
      <item>
       <widget class="QPushButton" name="pushButtonLayout">
        <property name="text">
         <string>Créer mise en page</string>
        </property>
       </widget>
      </item>
     </layout>
    </widget>
   </item>
   <item>
    <widget class="QDialogButtonBox" name="buttonBox">
     <property name="orientation">
      <enum>Qt::Horizontal</enum>
     </property>
     <property name="standardButtons">
      <set>QDialogButtonBox::Close|QDialogButtonBox::Help</set>
     </property>
    </widget>
   </item>
  </layout>
 </widget>
 <resources/>
 <connections>
  <connection>
   <sender>buttonBox</sender>
   <signal>accepted()</signal>
   <receiver>PromptDialogBase</receiver>
   <slot>accept()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>248</x>
     <y>254</y>
    </hint>
    <hint type="destinationlabel">
     <x>157</x>
     <y>274</y>
    </hint>
   </hints>
  </connection>
  <connection>
   <sender>buttonBox</sender>
   <signal>rejected()</signal>
   <receiver>PromptDialogBase</receiver>
   <slot>reject()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>316</x>
     <y>260</y>
    </hint>
    <hint type="destinationlabel">
     <x>286</x>
     <y>274</y>
    </hint>
   </hints>
  </connection>
 </connections>
</ui>
```

## 5. Compilation des ressources

Pour compiler le fichier resources.qrc en resources.py, il faut utiliser la commande suivante :

```
pyrcc5 -o resources.py resources.qrc
```

Cette commande doit être exécutée dans le répertoire du plugin.

## 6. Structure des dossiers à créer

Pour mettre en place la structure du plugin, il faut créer les dossiers suivants :

```
ForestDL/
├── core/
├── ui/
├── processing/
├── deeplearning/
├── allometry/
├── output/
├── test/
└── icons/
```

Chaque dossier doit contenir un fichier `__init__.py` vide pour être reconnu comme un package Python.
