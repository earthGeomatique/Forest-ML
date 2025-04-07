# -*- coding: utf-8 -*-
"""
Module de test pour le plugin ForestDL.
"""

import os
import sys
import unittest
from qgis.core import QgsApplication, QgsVectorLayer, QgsRasterLayer, QgsProject

# Ajouter le chemin du plugin au PYTHONPATH
plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if plugin_path not in sys.path:
    sys.path.append(plugin_path)

# Importer les modules du plugin
from ForestDL.allometry.height_metrics import HeightMetrics
from ForestDL.allometry.crown_metrics import CrownMetrics
from ForestDL.allometry.biomass import BiomassCalculator
from ForestDL.allometry.carbon_stock import CarbonStock
from ForestDL.processing.preprocessing import Preprocessing
from ForestDL.processing.segmentation import Segmentation
from ForestDL.processing.classification import Classification
from ForestDL.processing.feature_extraction import FeatureExtraction
from ForestDL.deeplearning.llm_connector import LLMConnector
from ForestDL.deeplearning.object_detection import ObjectDetection
from ForestDL.deeplearning.semantic_segmentation import SemanticSegmentation
from ForestDL.deeplearning.classification import DeepClassification


class TestForestDL(unittest.TestCase):
    """Classe de test pour le plugin ForestDL."""
    
    def setUp(self):
        """Configuration des tests."""
        # Initialiser QGIS
        self.qgs = QgsApplication([], False)
        self.qgs.initQgis()
        
        # Initialiser le projet
        self.project = QgsProject.instance()
        
        # Créer une interface factice
        self.iface = MockInterface()
    
    def tearDown(self):
        """Nettoyage après les tests."""
        # Nettoyer le projet
        self.project.clear()
        
        # Fermer QGIS
        self.qgs.exitQgis()
    
    def test_height_metrics(self):
        """Tester le module de métriques de hauteur."""
        # Créer une instance de HeightMetrics
        height_metrics = HeightMetrics(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(height_metrics)
        
        # Tester l'estimation du DBH à partir de la hauteur
        dbh, dbh_std = height_metrics.estimate_dbh_from_height(20.0, "mangrove")
        self.assertGreater(dbh, 0)
        self.assertGreater(dbh_std, 0)
    
    def test_crown_metrics(self):
        """Tester le module de métriques de couronne."""
        # Créer une instance de CrownMetrics
        crown_metrics = CrownMetrics(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(crown_metrics)
        
        # Tester l'estimation du diamètre de la couronne à partir du DBH
        crown_diameter = crown_metrics.estimate_crown_diameter_from_dbh(30.0, "mangrove")
        self.assertGreater(crown_diameter, 0)
        
        # Tester l'estimation de la surface de la couronne à partir du diamètre
        crown_area = crown_metrics.estimate_crown_area_from_diameter(crown_diameter)
        self.assertGreater(crown_area, 0)
    
    def test_biomass_calculator(self):
        """Tester le module de calcul de la biomasse."""
        # Créer une instance de BiomassCalculator
        biomass_calculator = BiomassCalculator(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(biomass_calculator)
        
        # Tester le calcul de la biomasse des mangroves
        agb, bgb, wood_volume = biomass_calculator.calculate_mangrove_biomass(30.0, 20.0, "rhizophora")
        self.assertGreater(agb, 0)
        self.assertGreater(bgb, 0)
        self.assertGreater(wood_volume, 0)
        
        # Tester l'obtention de la densité du bois
        wood_density = biomass_calculator.get_wood_density("rhizophora")
        self.assertGreater(wood_density, 0)
    
    def test_carbon_stock(self):
        """Tester le module de calcul du stock de carbone."""
        # Créer une instance de CarbonStock
        carbon_stock = CarbonStock(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(carbon_stock)
        
        # Tester le calcul du stock de carbone des mangroves
        agc, bgc, total_carbon, co2_equivalent, cl_lower, cl_upper = carbon_stock.calculate_mangrove_carbon(1000.0, 500.0)
        self.assertGreater(agc, 0)
        self.assertGreater(bgc, 0)
        self.assertGreater(total_carbon, 0)
        self.assertGreater(co2_equivalent, 0)
        self.assertGreater(cl_lower, 0)
        self.assertGreater(cl_upper, cl_lower)
    
    def test_preprocessing(self):
        """Tester le module de prétraitement."""
        # Créer une instance de Preprocessing
        preprocessing = Preprocessing(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(preprocessing)
    
    def test_segmentation(self):
        """Tester le module de segmentation."""
        # Créer une instance de Segmentation
        segmentation = Segmentation(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(segmentation)
    
    def test_classification(self):
        """Tester le module de classification."""
        # Créer une instance de Classification
        classification = Classification(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(classification)
    
    def test_feature_extraction(self):
        """Tester le module d'extraction de caractéristiques."""
        # Créer une instance de FeatureExtraction
        feature_extraction = FeatureExtraction(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(feature_extraction)
    
    def test_llm_connector(self):
        """Tester le module de connexion aux LLM."""
        # Créer une instance de LLMConnector
        llm_connector = LLMConnector()
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(llm_connector)
        
        # Vérifier les modèles disponibles
        models = llm_connector.get_available_models()
        self.assertIsNotNone(models)
        self.assertGreater(len(models), 0)
    
    def test_object_detection(self):
        """Tester le module de détection d'objets."""
        # Créer une instance de ObjectDetection
        object_detection = ObjectDetection(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(object_detection)
    
    def test_semantic_segmentation(self):
        """Tester le module de segmentation sémantique."""
        # Créer une instance de SemanticSegmentation
        semantic_segmentation = SemanticSegmentation(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(semantic_segmentation)
    
    def test_deep_classification(self):
        """Tester le module de classification profonde."""
        # Créer une instance de DeepClassification
        deep_classification = DeepClassification(self.iface)
        
        # Vérifier que l'instance a été créée correctement
        self.assertIsNotNone(deep_classification)


class MockInterface:
    """Interface factice pour les tests."""
    
    def __init__(self):
        """Constructeur."""
        pass
    
    def messageBar(self):
        """Simuler la barre de message."""
        return MockMessageBar()


class MockMessageBar:
    """Barre de message factice pour les tests."""
    
    def __init__(self):
        """Constructeur."""
        pass
    
    def pushMessage(self, title, text, level, duration):
        """Simuler l'affichage d'un message."""
        pass


if __name__ == '__main__':
    unittest.main()
