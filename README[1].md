# Documentation d'installation et d'utilisation du plugin ForestDL

## Présentation

ForestDL est un plugin QGIS spécialisé dans le deep learning pour la photogrammétrie et l'interprétation d'images, avec un focus particulier sur l'inventaire des mangroves et des forêts. Ce plugin combine les fonctionnalités du Semi-Automatic Classification Plugin (SCP) et de l'Orfeo ToolBox (OTB), tout en ajoutant des capacités d'apprentissage profond et des calculs allométriques et dendrométriques spécifiques.

## Fonctionnalités principales

- **Traitement d'images** : prétraitement, segmentation, classification et extraction de caractéristiques
- **Deep Learning** : détection d'objets, segmentation sémantique et classification d'images
- **Calculs allométriques et dendrométriques** : hauteur, diamètre, surface de couronne, biomasse et stock de carbone
- **Intégration de modèles LLM** : connexion aux API OpenAI, Gemini, Mistral et Deepseek
- **Automatisation** : tâches prédéfinies pour l'inventaire forestier et des mangroves

## Prérequis

- QGIS 3.28 ou supérieur
- Python 3.9 ou supérieur
- Accès à Internet pour les fonctionnalités d'API

## Installation

### Méthode 1 : Installation depuis le répertoire ZIP

1. Téléchargez le fichier ZIP du plugin ForestDL
2. Ouvrez QGIS
3. Allez dans le menu **Extensions > Installer/Gérer les extensions**
4. Cliquez sur **Installer depuis un ZIP**
5. Sélectionnez le fichier ZIP téléchargé
6. Cliquez sur **Installer l'extension**

### Méthode 2 : Installation manuelle

1. Téléchargez et décompressez le fichier ZIP du plugin ForestDL
2. Copiez le dossier `ForestDL` dans le répertoire des extensions QGIS :
   - Windows : `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins`
   - Linux : `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins`
   - macOS : `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins`
3. Redémarrez QGIS
4. Activez l'extension dans le menu **Extensions > Gérer les extensions**

## Configuration initiale

Après l'installation, vous devez configurer les API des modèles LLM que vous souhaitez utiliser :

1. Ouvrez QGIS
2. Allez dans le menu **Extensions > ForestDL > Configuration API**
3. Entrez vos clés API pour les modèles que vous souhaitez utiliser (OpenAI, Gemini, Mistral, Deepseek)
4. Cliquez sur **Enregistrer**

## Structure des dossiers

```
ForestDL/
├── __init__.py
├── forestdl.py
├── forestdl_dialog_base.ui
├── metadata.txt
├── README.md
├── resources.py
├── allometry/
│   ├── __init__.py
│   ├── biomass.py
│   ├── carbon_stock.py
│   ├── crown_metrics.py
│   └── height_metrics.py
├── deeplearning/
│   ├── __init__.py
│   ├── classification.py
│   ├── llm_connector.py
│   ├── object_detection.py
│   └── semantic_segmentation.py
├── icons/
│   └── [icônes du plugin]
└── processing/
    ├── __init__.py
    ├── classification.py
    ├── feature_extraction.py
    ├── preprocessing.py
    └── segmentation.py
```

## Guide d'utilisation

### Interface principale

L'interface du plugin ForestDL est accessible via le menu **Extensions > ForestDL > ForestDL** ou en cliquant sur l'icône ForestDL dans la barre d'outils.

L'interface principale est divisée en plusieurs onglets :

1. **Prétraitement** : outils de préparation des images
2. **Segmentation** : outils de segmentation d'images
3. **Classification** : outils de classification d'images
4. **Deep Learning** : outils d'apprentissage profond
5. **Allométrie** : outils de calculs allométriques et dendrométriques
6. **Prompts** : interface pour les requêtes aux modèles LLM

### Flux de travail typique

#### Inventaire forestier à partir d'orthophotos et MNE

1. **Prétraitement des données** :
   - Chargez vos orthophotos et MNE dans QGIS
   - Utilisez l'onglet **Prétraitement** pour préparer vos images

2. **Détection des arbres** :
   - Utilisez l'onglet **Deep Learning > Détection d'objets** pour détecter les arbres
   - Ou utilisez l'onglet **Segmentation > Segmentation des couronnes** pour segmenter les couronnes d'arbres

3. **Extraction des métriques de hauteur** :
   - Utilisez l'onglet **Allométrie > Métriques de hauteur** pour extraire les hauteurs des arbres à partir du MNE

4. **Calcul des métriques de couronne** :
   - Utilisez l'onglet **Allométrie > Métriques de couronne** pour calculer les métriques des couronnes

5. **Estimation du diamètre** :
   - Utilisez l'onglet **Allométrie > Estimation du diamètre** pour estimer le diamètre des arbres à partir de leur hauteur

6. **Calcul de la biomasse** :
   - Utilisez l'onglet **Allométrie > Calcul de biomasse** pour calculer la biomasse aérienne et souterraine

7. **Calcul du stock de carbone** :
   - Utilisez l'onglet **Allométrie > Stock de carbone** pour calculer le stock de carbone et générer les rapports

### Utilisation des modèles LLM

L'onglet **Prompts** vous permet d'interagir avec les modèles LLM pour automatiser certaines tâches :

1. Sélectionnez le modèle LLM à utiliser
2. Choisissez une tâche prédéfinie ou entrez un prompt personnalisé
3. Cliquez sur **Exécuter** pour envoyer la requête au modèle
4. Les résultats seront affichés et peuvent être appliqués directement aux couches QGIS

Exemples de tâches prédéfinies :
- Classification d'objets (bâtiments, voitures, parcelles agricoles)
- Numérisation automatique de plans cadastraux
- Mise en page cartographique automatique
- Recherche et importation de fichiers shapefile
- Jointures attributaires et spatiales
- Téléchargement de données OSM

## Formats de données

### Entrées
- Orthophotos (GeoTIFF, JPEG, PNG)
- Modèles numériques d'élévation (GeoTIFF)
- Couches vectorielles (Shapefile, GeoJSON)

### Sorties
- CSV des paramètres mesurés avec le nom des orthophotos
- CSV des paramètres de la biomasse avec le nom des orthophotos
- Fichiers shapefile avec les résultats des calculs

## Paramètres calculés

Le plugin ForestDL permet de calculer les paramètres suivants :

- Nombre d'espèces
- Diamètre moyen
- Volume total de biomasse
- Types d'espèces
- Niveau de précision
- Hauteur estimée
- Diamètre estimé
- Densité de biomasse
- Diamètre moyen
- Rayon maximal
- Surface de la canopée
- Volume de bois gisant
- Diamètre des arbres
- Stock de carbone (CL 2.5 et CL 97.5)

## Dépannage

### Problèmes courants

1. **Le plugin ne se charge pas**
   - Vérifiez que vous avez QGIS 3.28 ou supérieur
   - Vérifiez que vous avez Python 3.9 ou supérieur
   - Vérifiez que tous les fichiers du plugin sont présents

2. **Erreur lors de l'utilisation des API LLM**
   - Vérifiez que vos clés API sont correctement configurées
   - Vérifiez votre connexion Internet
   - Vérifiez que vous avez des crédits disponibles pour l'API utilisée

3. **Erreur lors du traitement des images**
   - Vérifiez que vos images sont correctement géoréférencées
   - Vérifiez que vous avez suffisamment de mémoire disponible
   - Essayez de traiter des images plus petites ou de réduire la résolution

### Support

Pour obtenir de l'aide ou signaler un bug, veuillez contacter le développeur ou ouvrir une issue sur le dépôt GitHub du plugin.

## Licence

Ce plugin est distribué sous licence GPL v3.

## Remerciements

Ce plugin s'inspire et utilise des composants des projets suivants :
- Semi-Automatic Classification Plugin (SCP)
- Orfeo ToolBox (OTB)
- OTBTF (module de deep learning pour OTB)

## Versions

- **1.0.0** : Version initiale
