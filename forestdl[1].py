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
        email                : contact@forestdl.org
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
import os.path

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.core import QgsProject, QgsMessageLog

# Initialisation des ressources Qt
from .resources import *

# Import des classes de dialogue
from .forestdl_dialog import ForestDLDialog

# Import des modules de traitement
from .processing.preprocessing import Preprocessing
from .processing.segmentation import Segmentation
from .processing.classification import Classification
from .processing.feature_extraction import FeatureExtraction

# Import des modules de deep learning
from .deeplearning.llm_connector import LLMConnector
from .deeplearning.object_detection import ObjectDetection
from .deeplearning.semantic_segmentation import SemanticSegmentation
from .deeplearning.classification import DeepLearningClassification

# Import des modules d'allométrie (à implémenter)
from .allometry.biomass import BiomassCalculator
from .allometry.crown_metrics import CrownMetrics
from .allometry.height_metrics import HeightMetrics
from .allometry.carbon_stock import CarbonStock


class ForestDL:
    """Plugin QGIS pour l'inventaire forestier et l'analyse des mangroves par deep learning."""

    def __init__(self, iface):
        """Constructeur.
        
        :param iface: Interface QGIS
        :type iface: QgsInterface
        """
        # Sauvegarde de la référence à l'interface QGIS
        self.iface = iface
        
        # Initialisation du plugin_dir
        self.plugin_dir = os.path.dirname(__file__)
        
        # Initialisation du traducteur
        self.locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'ForestDL_{}.qm'.format(self.locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Déclaration des attributs
        self.actions = []
        self.menu = self.tr('&ForestDL')
        self.toolbar = self.iface.addToolBar('ForestDL')
        self.toolbar.setObjectName('ForestDL')
        
        # Initialisation des modules
        self.preprocessing = Preprocessing(self.iface)
        self.segmentation = Segmentation(self.iface)
        self.classification = Classification(self.iface)
        self.feature_extraction = FeatureExtraction(self.iface)
        
        # Initialisation des modules de deep learning
        self.llm_connector = LLMConnector()
        self.object_detection = ObjectDetection(self.iface)
        self.semantic_segmentation = SemanticSegmentation(self.iface)
        self.dl_classification = DeepLearningClassification(self.iface)
        
        # Initialisation des modules d'allométrie
        self.biomass_calculator = BiomassCalculator(self.iface)
        self.crown_metrics = CrownMetrics(self.iface)
        self.height_metrics = HeightMetrics(self.iface)
        self.carbon_stock = CarbonStock(self.iface)
        
        # Initialisation du dialogue
        self.dlg = None

    def tr(self, message):
        """Méthode de traduction.
        
        :param message: Message à traduire
        :type message: str
        
        :returns: Message traduit
        :rtype: str
        """
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
        """Ajouter une action à l'interface.
        
        :param icon_path: Chemin vers l'icône
        :type icon_path: str
        :param text: Texte de l'action
        :type text: str
        :param callback: Fonction à appeler
        :type callback: function
        :param enabled_flag: Activer l'action
        :type enabled_flag: bool
        :param add_to_menu: Ajouter au menu
        :type add_to_menu: bool
        :param add_to_toolbar: Ajouter à la barre d'outils
        :type add_to_toolbar: bool
        :param status_tip: Texte d'aide
        :type status_tip: str
        :param whats_this: Texte d'aide contextuelle
        :type whats_this: str
        :param parent: Widget parent
        :type parent: QWidget
        
        :returns: Action créée
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
        """Créer les éléments d'interface graphique du plugin."""
        icon_path = ':/plugins/forestdl/icon.png'
        self.add_action(
            icon_path,
            text=self.tr('ForestDL'),
            callback=self.run,
            parent=self.iface.mainWindow())

    def unload(self):
        """Supprimer les éléments d'interface graphique du plugin."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr('&ForestDL'),
                action)
            self.iface.removeToolBarIcon(action)
        # Supprimer la barre d'outils
        del self.toolbar

    def run(self):
        """Exécuter le plugin."""
        # Créer et afficher le dialogue
        if self.dlg is None:
            self.dlg = ForestDLDialog(self.iface)
            
            # Connecter les signaux
            self.connect_signals()
            
            # Initialiser les widgets
            self.initialize_widgets()
        
        # Afficher le dialogue
        self.dlg.show()
        
        # Exécuter le dialogue
        result = self.dlg.exec_()
        
        # Traiter le résultat
        if result:
            pass
    
    def connect_signals(self):
        """Connecter les signaux du dialogue."""
        # Onglet Accueil
        self.dlg.pushButtonRunWorkflow.clicked.connect(self.run_workflow)
        self.dlg.pushButtonPreprocessing.clicked.connect(lambda: self.dlg.tabWidget.setCurrentIndex(1))
        self.dlg.pushButtonSegmentation.clicked.connect(lambda: self.dlg.tabWidget.setCurrentIndex(2))
        self.dlg.pushButtonClassification.clicked.connect(lambda: self.dlg.tabWidget.setCurrentIndex(3))
        self.dlg.pushButtonFeatureExtraction.clicked.connect(lambda: self.dlg.tabWidget.setCurrentIndex(1))
        self.dlg.pushButtonDeepLearning.clicked.connect(lambda: self.dlg.tabWidget.setCurrentIndex(3))
        self.dlg.pushButtonAllometry.clicked.connect(lambda: self.dlg.tabWidget.setCurrentIndex(4))
        
        # Onglet Prétraitement
        self.dlg.pushButtonRunPreprocessing.clicked.connect(self.run_preprocessing)
        
        # Onglet Segmentation
        self.dlg.pushButtonRunSegmentation.clicked.connect(self.run_segmentation)
        
        # Onglet Deep Learning
        self.dlg.pushButtonRunDL.clicked.connect(self.run_deep_learning)
        self.dlg.radioButtonLLMQuery.toggled.connect(self.toggle_llm_parameters)
        
        # Onglet Allométrie
        self.dlg.pushButtonRunAllometry.clicked.connect(self.run_allometry)
        
        # Onglet Configuration API
        self.dlg.pushButtonSaveAPIConfig.clicked.connect(self.save_api_config)
    
    def initialize_widgets(self):
        """Initialiser les widgets du dialogue."""
        # Charger les configurations API sauvegardées
        self.load_api_config()
        
        # Initialiser les sélecteurs de couches
        self.update_layer_comboboxes()
        
        # Connecter le signal de changement de projet
        QgsProject.instance().layersAdded.connect(self.update_layer_comboboxes)
        QgsProject.instance().layersRemoved.connect(self.update_layer_comboboxes)
    
    def update_layer_comboboxes(self):
        """Mettre à jour les combobox de sélection de couches."""
        # Cette méthode est appelée automatiquement lorsque des couches sont ajoutées ou supprimées
        pass
    
    def run_workflow(self):
        """Exécuter le flux de travail sélectionné."""
        workflow = self.dlg.comboBoxWorkflow.currentText()
        
        # Récupérer les couches d'entrée
        orthophoto_layer = self.dlg.mMapLayerComboBoxOrthophoto.currentLayer()
        mne_layer = self.dlg.mMapLayerComboBoxMNE.currentLayer()
        
        if not orthophoto_layer or not mne_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr("Veuillez sélectionner les couches d'entrée."))
            return
        
        # Exécuter le flux de travail en fonction de la sélection
        if workflow == self.tr("Détection d'arbres individuels"):
            self.run_tree_detection_workflow(orthophoto_layer, mne_layer)
        elif workflow == self.tr('Segmentation des types de forêts'):
            self.run_forest_segmentation_workflow(orthophoto_layer, mne_layer)
        elif workflow == self.tr('Classification des espèces'):
            self.run_species_classification_workflow(orthophoto_layer, mne_layer)
        elif workflow == self.tr('Inventaire forestier complet'):
            self.run_complete_inventory_workflow(orthophoto_layer, mne_layer)
        elif workflow == self.tr('Analyse des mangroves'):
            self.run_mangrove_analysis_workflow(orthophoto_layer, mne_layer)
    
    def run_tree_detection_workflow(self, orthophoto_layer, mne_layer):
        """Exécuter le flux de travail de détection d'arbres individuels."""
        QMessageBox.information(self.dlg, self.tr('Information'), 
                               self.tr("Démarrage du flux de travail de détection d'arbres individuels..."))
        
        # Créer un MNH
        mnh_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'mnh.tif')
        mnh_layer = self.preprocessing.create_chm(mne_layer, None, mnh_path)
        
        if not mnh_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la création du MNH.'))
            return
        
        # Détecter les arbres
        trees_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'trees.shp')
        trees_layer = self.segmentation.detect_trees(mnh_layer, trees_path)
        
        if not trees_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la détection des arbres.'))
            return
        
        # Segmenter les couronnes
        crowns_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'crowns.shp')
        crowns_layer = self.segmentation.segment_crowns(mnh_layer, trees_layer, crowns_path)
        
        if not crowns_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la segmentation des couronnes.'))
            return
        
        QMessageBox.information(self.dlg, self.tr('Succès'), 
                               self.tr("Flux de travail de détection d'arbres individuels terminé avec succès."))
    
    def run_forest_segmentation_workflow(self, orthophoto_layer, mne_layer):
        """Exécuter le flux de travail de segmentation des types de forêts."""
        QMessageBox.information(self.dlg, self.tr('Information'), 
                               self.tr('Démarrage du flux de travail de segmentation des types de forêts...'))
        
        # Vérifier si le modèle de deep learning est disponible
        if not self.check_dl_dependencies():
            return
        
        # Segmenter les types de forêts
        forest_types_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'forest_types.tif')
        forest_types_layer = self.semantic_segmentation.segment_forest_types(orthophoto_layer, forest_types_path)
        
        if not forest_types_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la segmentation des types de forêts.'))
            return
        
        # Convertir en vecteur
        forest_types_vector_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'forest_types.shp')
        forest_types_vector_layer = self.semantic_segmentation.segment_to_vector(forest_types_layer, forest_types_vector_path)
        
        if not forest_types_vector_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la conversion en vecteur.'))
            return
        
        QMessageBox.information(self.dlg, self.tr('Succès'), 
                               self.tr('Flux de travail de segmentation des types de forêts terminé avec succès.'))
    
    def run_species_classification_workflow(self, orthophoto_layer, mne_layer):
        """Exécuter le flux de travail de classification des espèces."""
        QMessageBox.information(self.dlg, self.tr('Information'), 
                               self.tr('Démarrage du flux de travail de classification des espèces...'))
        
        # Vérifier si le modèle de deep learning est disponible
        if not self.check_dl_dependencies():
            return
        
        # Créer un MNH
        mnh_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'mnh.tif')
        mnh_layer = self.preprocessing.create_chm(mne_layer, None, mnh_path)
        
        if not mnh_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la création du MNH.'))
            return
        
        # Détecter les arbres
        trees_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'trees.shp')
        trees_layer = self.segmentation.detect_trees(mnh_layer, trees_path)
        
        if not trees_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la détection des arbres.'))
            return
        
        # Segmenter les couronnes
        crowns_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'crowns.shp')
        crowns_layer = self.segmentation.segment_crowns(mnh_layer, trees_layer, crowns_path)
        
        if not crowns_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la segmentation des couronnes.'))
            return
        
        # Classifier les espèces
        species_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'species.shp')
        species_layer = self.dl_classification.classify_crowns(crowns_layer, [orthophoto_layer], species_path)
        
        if not species_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la classification des espèces.'))
            return
        
        QMessageBox.information(self.dlg, self.tr('Succès'), 
                               self.tr('Flux de travail de classification des espèces terminé avec succès.'))
    
    def run_complete_inventory_workflow(self, orthophoto_layer, mne_layer):
        """Exécuter le flux de travail d'inventaire forestier complet."""
        QMessageBox.information(self.dlg, self.tr('Information'), 
                               self.tr("Démarrage du flux de travail d'inventaire forestier complet..."))
        
        # Vérifier si le modèle de deep learning est disponible
        if not self.check_dl_dependencies():
            return
        
        # Créer un MNH
        mnh_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'mnh.tif')
        mnh_layer = self.preprocessing.create_chm(mne_layer, None, mnh_path)
        
        if not mnh_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la création du MNH.'))
            return
        
        # Détecter les arbres
        trees_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'trees.shp')
        trees_layer = self.segmentation.detect_trees(mnh_layer, trees_path)
        
        if not trees_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la détection des arbres.'))
            return
        
        # Segmenter les couronnes
        crowns_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'crowns.shp')
        crowns_layer = self.segmentation.segment_crowns(mnh_layer, trees_layer, crowns_path)
        
        if not crowns_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la segmentation des couronnes.'))
            return
        
        # Classifier les espèces
        species_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'species.shp')
        species_layer = self.dl_classification.classify_crowns(crowns_layer, [orthophoto_layer], species_path)
        
        if not species_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la classification des espèces.'))
            return
        
        # Calculer les métriques de hauteur
        height_metrics_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'height_metrics.shp')
        height_metrics_layer = self.height_metrics.extract_height_metrics(species_layer, mnh_layer, height_metrics_path)
        
        if not height_metrics_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec du calcul des métriques de hauteur.'))
            return
        
        # Calculer les métriques de couronne
        crown_metrics_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'crown_metrics.shp')
        crown_metrics_layer = self.crown_metrics.calculate_crown_metrics(height_metrics_layer, crown_metrics_path)
        
        if not crown_metrics_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec du calcul des métriques de couronne.'))
            return
        
        # Calculer la biomasse
        biomass_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'biomass.shp')
        biomass_layer = self.biomass_calculator.calculate_biomass(crown_metrics_layer, biomass_path)
        
        if not biomass_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec du calcul de la biomasse.'))
            return
        
        # Calculer le stock de carbone
        carbon_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'carbon.shp')
        carbon_layer = self.carbon_stock.calculate_carbon_stock(biomass_layer, carbon_path)
        
        if not carbon_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec du calcul du stock de carbone.'))
            return
        
        # Exporter les résultats en CSV
        csv_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'inventory_results.csv')
        self.export_to_csv(carbon_layer, csv_path)
        
        QMessageBox.information(self.dlg, self.tr('Succès'), 
                               self.tr("Flux de travail d'inventaire forestier complet terminé avec succès."))
    
    def run_mangrove_analysis_workflow(self, orthophoto_layer, mne_layer):
        """Exécuter le flux de travail d'analyse des mangroves."""
        QMessageBox.information(self.dlg, self.tr('Information'), 
                               self.tr("Démarrage du flux de travail d'analyse des mangroves..."))
        
        # Vérifier si le modèle de deep learning est disponible
        if not self.check_dl_dependencies():
            return
        
        # Segmenter les mangroves
        mangroves_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'mangroves.tif')
        mangroves_vector_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'mangroves.shp')
        mangroves_layer = self.semantic_segmentation.segment_mangroves(orthophoto_layer, mangroves_path, mangroves_vector_path)
        
        if not mangroves_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la segmentation des mangroves.'))
            return
        
        # Créer un MNH
        mnh_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'mnh.tif')
        mnh_layer = self.preprocessing.create_chm(mne_layer, None, mnh_path)
        
        if not mnh_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la création du MNH.'))
            return
        
        # Calculer la biomasse des mangroves
        biomass_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'mangrove_biomass.shp')
        biomass_layer = self.biomass_calculator.calculate_mangrove_biomass(mangroves_vector_path, mnh_layer, biomass_path)
        
        if not biomass_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec du calcul de la biomasse des mangroves.'))
            return
        
        # Calculer le stock de carbone
        carbon_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'mangrove_carbon.shp')
        carbon_layer = self.carbon_stock.calculate_carbon_stock(biomass_layer, carbon_path)
        
        if not carbon_layer:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec du calcul du stock de carbone.'))
            return
        
        # Exporter les résultats en CSV
        csv_path = os.path.join(os.path.dirname(orthophoto_layer.source()), 'mangrove_results.csv')
        self.export_to_csv(carbon_layer, csv_path)
        
        QMessageBox.information(self.dlg, self.tr('Succès'), 
                               self.tr("Flux de travail d'analyse des mangroves terminé avec succès."))
    
    def run_preprocessing(self):
        """Exécuter l'opération de prétraitement sélectionnée."""
        # Récupérer les paramètres
        input_layer = self.dlg.mMapLayerComboBoxInputRaster.currentLayer()
        output_path = self.dlg.mQgsFileWidgetOutputPath.filePath()
        
        if not input_layer or not output_path:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr("Veuillez sélectionner une couche d'entrée et un chemin de sortie."))
            return
        
        # Déterminer l'opération sélectionnée
        if self.dlg.radioButtonRadiometricCorrection.isChecked():
            method = self.dlg.comboBoxMethod.currentText()
            result = self.preprocessing.radiometric_correction(input_layer, output_path, method)
        elif self.dlg.radioButtonIndices.isChecked():
            result = self.preprocessing.calculate_indices(input_layer, output_path)
        elif self.dlg.radioButtonFilter.isChecked():
            kernel_size = self.dlg.spinBoxKernelSize.value()
            result = self.preprocessing.filter_image(input_layer, output_path, kernel_size)
        elif self.dlg.radioButtonMNH.isChecked():
            result = self.preprocessing.create_chm(input_layer, None, output_path)
        elif self.dlg.radioButtonMask.isChecked():
            result = self.preprocessing.create_mask(input_layer, output_path)
        
        if result:
            QMessageBox.information(self.dlg, self.tr('Succès'), self.tr('Prétraitement terminé avec succès.'))
        else:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec du prétraitement.'))
    
    def run_segmentation(self):
        """Exécuter l'opération de segmentation sélectionnée."""
        # Récupérer les paramètres
        input_layer = self.dlg.mMapLayerComboBoxMNH.currentLayer()
        output_path = self.dlg.mQgsFileWidgetSegmentationOutput.filePath()
        
        if not input_layer or not output_path:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr("Veuillez sélectionner une couche d'entrée et un chemin de sortie."))
            return
        
        # Récupérer les paramètres
        min_height = self.dlg.doubleSpinBoxMinHeight.value()
        min_distance = self.dlg.spinBoxMinDistance.value()
        min_area = self.dlg.doubleSpinBoxMinArea.value()
        
        # Déterminer l'opération sélectionnée
        if self.dlg.radioButtonDetectTrees.isChecked():
            result = self.segmentation.detect_trees(input_layer, output_path, min_height, min_distance)
        elif self.dlg.radioButtonSegmentCrownsWatershed.isChecked():
            result = self.segmentation.segment_crowns_watershed(input_layer, None, output_path, min_height)
        elif self.dlg.radioButtonSegmentCrownsRegionGrowing.isChecked():
            result = self.segmentation.segment_crowns_region_growing(input_layer, None, output_path, min_height)
        elif self.dlg.radioButtonPostProcessCrowns.isChecked():
            result = self.segmentation.post_process_crowns(input_layer, output_path, min_area)
        
        if result:
            QMessageBox.information(self.dlg, self.tr('Succès'), self.tr('Segmentation terminée avec succès.'))
        else:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de la segmentation.'))
    
    def run_deep_learning(self):
        """Exécuter l'opération de deep learning sélectionnée."""
        # Récupérer les paramètres
        input_layer = self.dlg.mMapLayerComboBoxDLInput.currentLayer()
        output_path = self.dlg.mQgsFileWidgetDLOutput.filePath()
        
        if not input_layer or not output_path:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr("Veuillez sélectionner une couche d'entrée et un chemin de sortie."))
            return
        
        # Vérifier si le modèle de deep learning est disponible
        if not self.check_dl_dependencies():
            return
        
        # Récupérer les paramètres
        model_path = self.dlg.mQgsFileWidgetModelPath.filePath()
        framework = self.dlg.comboBoxFramework.currentText()
        confidence_threshold = self.dlg.doubleSpinBoxConfidenceThreshold.value()
        
        # Déterminer l'opération sélectionnée
        if self.dlg.radioButtonObjectDetection.isChecked():
            if framework == 'TensorFlow':
                self.object_detection.load_model_tf(model_path)
            else:
                self.object_detection.load_model_torch(model_path)
            
            result = self.object_detection.detect_objects(input_layer, output_path, confidence_threshold)
        
        elif self.dlg.radioButtonSemanticSegmentation.isChecked():
            if framework == 'TensorFlow':
                self.semantic_segmentation.load_model_tf(model_path)
            else:
                self.semantic_segmentation.load_model_torch(model_path)
            
            result = self.semantic_segmentation.segment_image(input_layer, output_path)
        
        elif self.dlg.radioButtonClassification.isChecked():
            if framework == 'TensorFlow':
                self.dl_classification.load_model_tf(model_path)
            else:
                self.dl_classification.load_model_torch(model_path)
            
            result = self.dl_classification.classify_image(input_layer, output_path)
        
        elif self.dlg.radioButtonLLMQuery.isChecked():
            # Récupérer les paramètres LLM
            provider = self.dlg.comboBoxLLMProvider.currentText()
            prompt = self.dlg.plainTextEditPrompt.toPlainText()
            
            # Récupérer la clé API
            api_key = self.get_api_key(provider)
            
            if not api_key:
                QMessageBox.warning(self.dlg, self.tr('Erreur'), 
                                   self.tr('Clé API non configurée pour {}. Veuillez configurer la clé API dans l'onglet Configuration API.').format(provider))
                return
            
            # Exécuter la requête LLM
            result = self.llm_connector.query(provider, prompt, api_key, input_layer, output_path)
        
        if result:
            QMessageBox.information(self.dlg, self.tr('Succès'), self.tr('Opération de deep learning terminée avec succès.'))
        else:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec de l'opération de deep learning.'))
    
    def run_allometry(self):
        """Exécuter l'opération allométrique sélectionnée."""
        # Récupérer les paramètres
        crown_layer = self.dlg.mMapLayerComboBoxCrowns.currentLayer()
        mnh_layer = self.dlg.mMapLayerComboBoxMNHAllometry.currentLayer()
        output_path = self.dlg.mQgsFileWidgetAllometryOutput.filePath()
        
        if not crown_layer or not mnh_layer or not output_path:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr("Veuillez sélectionner les couches d'entrée et un chemin de sortie."))
            return
        
        # Récupérer les paramètres
        species_field = self.dlg.lineEditSpeciesField.text()
        height_field = self.dlg.lineEditHeightField.text()
        method = self.dlg.comboBoxAllometricMethod.currentText()
        
        # Déterminer l'opération sélectionnée
        if self.dlg.radioButtonHeightMetrics.isChecked():
            result = self.height_metrics.extract_height_metrics(crown_layer, mnh_layer, output_path, height_field)
        elif self.dlg.radioButtonEstimateDiameter.isChecked():
            result = self.height_metrics.estimate_diameter(crown_layer, mnh_layer, output_path, height_field, species_field)
        elif self.dlg.radioButtonCrownMetrics.isChecked():
            result = self.crown_metrics.calculate_crown_metrics(crown_layer, output_path)
        elif self.dlg.radioButtonBiomassCalculation.isChecked():
            result = self.biomass_calculator.calculate_biomass(crown_layer, output_path, method, species_field, height_field)
        elif self.dlg.radioButtonCarbonStock.isChecked():
            result = self.carbon_stock.calculate_carbon_stock(crown_layer, output_path, method, species_field)
        
        if result:
            QMessageBox.information(self.dlg, self.tr('Succès'), self.tr('Calcul allométrique terminé avec succès.'))
        else:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), self.tr('Échec du calcul allométrique.'))
    
    def toggle_llm_parameters(self, checked):
        """Afficher ou masquer les paramètres LLM."""
        self.dlg.groupBoxLLMParameters.setVisible(checked)
    
    def save_api_config(self):
        """Sauvegarder la configuration des API."""
        # Récupérer les clés API
        openai_key = self.dlg.lineEditOpenAIKey.text()
        gemini_key = self.dlg.lineEditGeminiKey.text()
        mistral_key = self.dlg.lineEditMistralKey.text()
        deepseek_key = self.dlg.lineEditDeepseekKey.text()
        
        # Récupérer les modèles
        openai_model = self.dlg.comboBoxOpenAIModel.currentText()
        gemini_model = self.dlg.comboBoxGeminiModel.currentText()
        mistral_model = self.dlg.comboBoxMistralModel.currentText()
        deepseek_model = self.dlg.comboBoxDeepseekModel.currentText()
        
        # Sauvegarder dans les paramètres QGIS
        s = QSettings()
        s.beginGroup('ForestDL')
        
        if openai_key:
            s.setValue('openai_key', openai_key)
            s.setValue('openai_model', openai_model)
        
        if gemini_key:
            s.setValue('gemini_key', gemini_key)
            s.setValue('gemini_model', gemini_model)
        
        if mistral_key:
            s.setValue('mistral_key', mistral_key)
            s.setValue('mistral_model', mistral_model)
        
        if deepseek_key:
            s.setValue('deepseek_key', deepseek_key)
            s.setValue('deepseek_model', deepseek_model)
        
        s.endGroup()
        
        QMessageBox.information(self.dlg, self.tr('Succès'), self.tr('Configuration des API sauvegardée avec succès.'))
    
    def load_api_config(self):
        """Charger la configuration des API."""
        s = QSettings()
        s.beginGroup('ForestDL')
        
        # Charger les clés API
        openai_key = s.value('openai_key', '')
        gemini_key = s.value('gemini_key', '')
        mistral_key = s.value('mistral_key', '')
        deepseek_key = s.value('deepseek_key', '')
        
        # Charger les modèles
        openai_model = s.value('openai_model', 'gpt-4o')
        gemini_model = s.value('gemini_model', 'gemini-1.5-pro')
        mistral_model = s.value('mistral_model', 'mistral-large-latest')
        deepseek_model = s.value('deepseek_model', 'deepseek-chat')
        
        s.endGroup()
        
        # Mettre à jour les widgets
        self.dlg.lineEditOpenAIKey.setText(openai_key)
        self.dlg.lineEditGeminiKey.setText(gemini_key)
        self.dlg.lineEditMistralKey.setText(mistral_key)
        self.dlg.lineEditDeepseekKey.setText(deepseek_key)
        
        # Mettre à jour les combobox
        index = self.dlg.comboBoxOpenAIModel.findText(openai_model)
        if index >= 0:
            self.dlg.comboBoxOpenAIModel.setCurrentIndex(index)
        
        index = self.dlg.comboBoxGeminiModel.findText(gemini_model)
        if index >= 0:
            self.dlg.comboBoxGeminiModel.setCurrentIndex(index)
        
        index = self.dlg.comboBoxMistralModel.findText(mistral_model)
        if index >= 0:
            self.dlg.comboBoxMistralModel.setCurrentIndex(index)
        
        index = self.dlg.comboBoxDeepseekModel.findText(deepseek_model)
        if index >= 0:
            self.dlg.comboBoxDeepseekModel.setCurrentIndex(index)
    
    def get_api_key(self, provider):
        """Récupérer la clé API pour un fournisseur donné.
        
        :param provider: Nom du fournisseur
        :type provider: str
        
        :returns: Clé API
        :rtype: str
        """
        s = QSettings()
        s.beginGroup('ForestDL')
        
        if provider == 'OpenAI':
            key = s.value('openai_key', '')
        elif provider == 'Gemini':
            key = s.value('gemini_key', '')
        elif provider == 'Mistral':
            key = s.value('mistral_key', '')
        elif provider == 'Deepseek':
            key = s.value('deepseek_key', '')
        else:
            key = ''
        
        s.endGroup()
        
        return key
    
    def check_dl_dependencies(self):
        """Vérifier les dépendances pour le deep learning.
        
        :returns: True si les dépendances sont disponibles, False sinon
        :rtype: bool
        """
        # Vérifier les frameworks disponibles
        frameworks = self.object_detection.check_dependencies()
        
        if not frameworks['tensorflow'] and not frameworks['pytorch']:
            QMessageBox.warning(self.dlg, self.tr('Erreur'), 
                               self.tr('Aucun framework de deep learning disponible. Veuillez installer TensorFlow ou PyTorch.'))
            return False
        
        # Vérifier si un modèle est sélectionné
        model_path = self.dlg.mQgsFileWidgetModelPath.filePath()
        
        if not model_path and not self.dlg.radioButtonLLMQuery.isChecked():
            QMessageBox.warning(self.dlg, self.tr('Erreur'), 
                               self.tr('Veuillez sélectionner un modèle de deep learning.'))
            return False
        
        return True
    
    def export_to_csv(self, layer, output_path):
        """Exporter une couche en CSV.
        
        :param layer: Couche à exporter
        :type layer: QgsVectorLayer
        :param output_path: Chemin du fichier de sortie
        :type output_path: str
        
        :returns: True si l'exportation a réussi, False sinon
        :rtype: bool
        """
        try:
            from qgis.core import QgsVectorFileWriter
            
            # Exporter la couche en CSV
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = 'CSV'
            options.layerName = os.path.basename(output_path)
            
            error = QgsVectorFileWriter.writeAsVectorFormat(layer, output_path, options)
            
            return error[0] == QgsVectorFileWriter.NoError
        
        except Exception as e:
            QgsMessageLog.logMessage(f"Erreur lors de l'exportation en CSV: {str(e)}", 'ForestDL', level=2)
            return False
