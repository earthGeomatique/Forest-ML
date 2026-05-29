"""
geo_utils.py — Geospatial utility functions for ForestDL.

Provides helpers for:
  - Converting pixel bounding boxes to geographic coordinates
  - Loading raster/vector layers into QGIS
  - Listing raster layers from the QGIS project
"""

import os
from typing import Tuple, List, Optional

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from qgis.core import (
        QgsProject,
        QgsRasterLayer,
        QgsVectorLayer,
        QgsMapLayer,
    )
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def pixel_bbox_to_geo(
    transform,
    x1_px: float,
    y1_px: float,
    x2_px: float,
    y2_px: float,
) -> Tuple[float, float, float, float]:
    """
    Convert pixel bounding box coordinates to geographic coordinates using
    a rasterio affine transform.

    The rasterio Affine transform maps (col, row) → (x, y) in the raster CRS.

    :param transform: rasterio.transform.Affine object.
    :param x1_px: Left column (pixel) of bounding box.
    :param y1_px: Top row (pixel) of bounding box.
    :param x2_px: Right column (pixel) of bounding box.
    :param y2_px: Bottom row (pixel) of bounding box.
    :return: (geo_minx, geo_miny, geo_maxx, geo_maxy) in the raster CRS units.
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio est requis pour la conversion de coordonnées.")

    # rasterio.transform.xy(transform, row, col) → (x, y)
    geo_x1, geo_y1 = rasterio.transform.xy(transform, y1_px, x1_px, offset="ul")
    geo_x2, geo_y2 = rasterio.transform.xy(transform, y2_px, x2_px, offset="ul")

    geo_minx = min(geo_x1, geo_x2)
    geo_maxx = max(geo_x1, geo_x2)
    geo_miny = min(geo_y1, geo_y2)
    geo_maxy = max(geo_y1, geo_y2)

    return geo_minx, geo_miny, geo_maxx, geo_maxy


def pixel_point_to_geo(
    transform,
    col: float,
    row: float,
) -> Tuple[float, float]:
    """
    Convert a single pixel (col, row) to geographic (x, y).

    :param transform: rasterio.transform.Affine object.
    :param col: Column (pixel x coordinate).
    :param row: Row (pixel y coordinate).
    :return: (geo_x, geo_y) in the raster CRS units.
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio est requis pour la conversion de coordonnées.")
    x, y = rasterio.transform.xy(transform, row, col, offset="center")
    return float(x), float(y)


def geo_to_pixel(
    transform,
    geo_x: float,
    geo_y: float,
) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel (row, col) using rasterio.

    :param transform: rasterio.transform.Affine object.
    :param geo_x: Geographic X coordinate.
    :param geo_y: Geographic Y coordinate.
    :return: (row, col) as integers.
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio est requis pour la conversion de coordonnées.")
    row, col = rasterio.transform.rowcol(transform, geo_x, geo_y)
    return int(row), int(col)


def get_raster_info(raster_path: str) -> dict:
    """
    Return a dictionary with basic raster metadata.

    :param raster_path: Path to a raster file.
    :return: dict with keys: width, height, crs, transform, band_count, dtype, bounds.
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio est requis pour lire les métadonnées raster.")
    with rasterio.open(raster_path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": src.transform,
            "band_count": src.count,
            "dtype": src.dtypes[0],
            "bounds": src.bounds,
            "nodata": src.nodata,
        }


# ---------------------------------------------------------------------------
# QGIS layer loading
# ---------------------------------------------------------------------------

def load_raster_layer(path: str, layer_name: Optional[str] = None) -> Optional["QgsRasterLayer"]:
    """
    Load a raster file as a QGIS raster layer and add it to the current project.

    :param path: Absolute path to the raster file.
    :param layer_name: Name for the layer in QGIS. Defaults to filename without extension.
    :return: QgsRasterLayer if successful, None otherwise.
    :raises RuntimeError: If QGIS is not available or the layer is invalid.
    """
    if not QGIS_AVAILABLE:
        raise RuntimeError("QGIS n'est pas disponible dans cet environnement.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier raster introuvable: {path}")

    if layer_name is None:
        layer_name = os.path.splitext(os.path.basename(path))[0]

    layer = QgsRasterLayer(path, layer_name)
    if not layer.isValid():
        raise RuntimeError(
            f"Impossible de charger le raster '{path}'. "
            "Vérifiez que le fichier est un format raster valide."
        )

    QgsProject.instance().addMapLayer(layer)
    return layer


def load_vector_layer(
    path: str,
    layer_name: Optional[str] = None,
    provider: str = "ogr",
) -> Optional["QgsVectorLayer"]:
    """
    Load a vector file as a QGIS vector layer and add it to the current project.

    :param path: Absolute path to the vector file (GeoJSON, SHP, GPKG, etc.).
    :param layer_name: Name for the layer in QGIS. Defaults to filename without extension.
    :param provider: QGIS data provider (default: "ogr").
    :return: QgsVectorLayer if successful, None otherwise.
    :raises RuntimeError: If QGIS is not available or the layer is invalid.
    """
    if not QGIS_AVAILABLE:
        raise RuntimeError("QGIS n'est pas disponible dans cet environnement.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier vecteur introuvable: {path}")

    if layer_name is None:
        layer_name = os.path.splitext(os.path.basename(path))[0]

    layer = QgsVectorLayer(path, layer_name, provider)
    if not layer.isValid():
        raise RuntimeError(
            f"Impossible de charger le vecteur '{path}'. "
            "Vérifiez que le fichier est un format vectoriel valide."
        )

    QgsProject.instance().addMapLayer(layer)
    return layer


# ---------------------------------------------------------------------------
# QGIS project helpers
# ---------------------------------------------------------------------------

def get_raster_layers() -> List["QgsRasterLayer"]:
    """
    Return a list of all raster layers currently loaded in the QGIS project.

    :return: List of QgsRasterLayer objects (may be empty).
    """
    if not QGIS_AVAILABLE:
        return []

    raster_layers = []
    for layer in QgsProject.instance().mapLayers().values():
        if isinstance(layer, QgsRasterLayer) or (
            hasattr(layer, 'type') and layer.type() == QgsMapLayer.RasterLayer
        ):
            raster_layers.append(layer)
    return raster_layers


def get_raster_layer_names() -> List[str]:
    """
    Return a list of names of all raster layers in the QGIS project.

    :return: List of layer name strings.
    """
    return [layer.name() for layer in get_raster_layers()]


def get_raster_layer_by_name(name: str) -> Optional["QgsRasterLayer"]:
    """
    Find a raster layer by its name in the current QGIS project.

    :param name: Layer name to search for.
    :return: Matching QgsRasterLayer, or None if not found.
    """
    for layer in get_raster_layers():
        if layer.name() == name:
            return layer
    return None


def get_vector_layers() -> list:
    """
    Return a list of all vector layers currently loaded in the QGIS project.

    :return: List of QgsVectorLayer objects.
    """
    if not QGIS_AVAILABLE:
        return []

    vector_layers = []
    for layer in QgsProject.instance().mapLayers().values():
        if isinstance(layer, QgsVectorLayer) or (
            hasattr(layer, 'type') and layer.type() == QgsMapLayer.VectorLayer
        ):
            vector_layers.append(layer)
    return vector_layers


def remove_layer_by_name(name: str) -> bool:
    """
    Remove a layer from the QGIS project by name.

    :param name: Name of the layer to remove.
    :return: True if a layer was removed, False if not found.
    """
    if not QGIS_AVAILABLE:
        return False

    for layer_id, layer in QgsProject.instance().mapLayers().items():
        if layer.name() == name:
            QgsProject.instance().removeMapLayer(layer_id)
            return True
    return False


def layer_source_path(layer) -> str:
    """
    Extract the file path from a QGIS map layer source URI.

    :param layer: A QgsMapLayer (raster or vector).
    :return: File path string, or the raw source string if parsing fails.
    """
    source = layer.source()
    # For simple file-based layers the source IS the path.
    # For more complex URIs (e.g. WMS, PostGIS), return the raw string.
    if os.path.exists(source):
        return source
    # Try to strip any query parameters (e.g. "path/to/file.tif|layerid=0")
    if "|" in source:
        candidate = source.split("|")[0]
        if os.path.exists(candidate):
            return candidate
    return source
