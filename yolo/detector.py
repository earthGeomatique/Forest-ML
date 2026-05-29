"""
YOLODetector — detection YOLOv8 sur grandes orthophotos GeoTIFF.

Lecteur raster : GDAL en priorite (toujours disponible dans QGIS),
                 rasterio en option si installe.
Modele         : ultralytics YOLOv8 (.pt) ou ONNX ultralytics (.onnx).
Sortie         : GeoJSON avec boites en coordonnees geographiques.
"""

import os
import sys
import json
import subprocess

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# GDAL — toujours present dans QGIS
try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

# rasterio — optionnel (fallback si GDAL absent, non utilise si GDAL present)
try:
    import rasterio
    from rasterio.windows import Window as RasterioWindow
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from shapely.geometry import box as shapely_box
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    from qgis.PyQt.QtCore import QObject, pyqtSignal
except ImportError:
    from PyQt5.QtCore import QObject, pyqtSignal


COCO_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}
TILE_OVERLAP = 64


# ══════════════════════════════════════════════════════════════════════════
# Utilitaires GDAL (remplacement de rasterio)
# ══════════════════════════════════════════════════════════════════════════

class GDALRaster:
    """Wrapper minimal autour d'un dataset GDAL pour lire les tuiles."""

    def __init__(self, path: str):
        self._ds = gdal.Open(path, gdal.GA_ReadOnly)
        if self._ds is None:
            raise RuntimeError(f"GDAL ne peut pas ouvrir : {path}")
        self.width      = self._ds.RasterXSize
        self.height     = self._ds.RasterYSize
        self.band_count = self._ds.RasterCount
        self._gt        = self._ds.GetGeoTransform()  # (x0, dx, rx, y0, ry, dy)
        self._proj      = self._ds.GetProjection()

    def epsg(self) -> int | None:
        try:
            srs = osr.SpatialReference(wkt=self._proj)
            code = srs.GetAttrValue("AUTHORITY", 1)
            return int(code) if code else None
        except Exception:
            return None

    def crs_label(self) -> str:
        try:
            srs = osr.SpatialReference(wkt=self._proj)
            return srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS") or "?"
        except Exception:
            return "?"

    def read_tile(self, col_off: int, row_off: int,
                  tile_w: int, tile_h: int) -> "np.ndarray":
        """Retourne un tableau (bands, h, w) en float32."""
        data = self._ds.ReadAsArray(col_off, row_off, tile_w, tile_h)
        if data is None:
            raise RuntimeError(
                f"GDAL ReadAsArray echoue pour tuile ({col_off},{row_off})"
            )
        if data.ndim == 2:          # bande unique -> (1, h, w)
            data = data[np.newaxis, :]
        return data.astype(np.float32)

    def pixel_to_geo(self, col: float, row: float):
        """Convertit (col, row) en (X_geo, Y_geo) via la geotransformation."""
        gt = self._gt
        x = gt[0] + col * gt[1] + row * gt[2]
        y = gt[3] + col * gt[4] + row * gt[5]
        return x, y

    def bbox_to_geo(self, x1: float, y1: float,
                    x2: float, y2: float):
        """Retourne (minx, miny, maxx, maxy) en coordonnees geographiques."""
        corners = [
            self.pixel_to_geo(x1, y1),
            self.pixel_to_geo(x2, y1),
            self.pixel_to_geo(x1, y2),
            self.pixel_to_geo(x2, y2),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def close(self):
        self._ds = None


# ══════════════════════════════════════════════════════════════════════════
# Installeur automatique de dependances
# ══════════════════════════════════════════════════════════════════════════

def _pip_install(package: str) -> bool:
    """Installe un paquet pip dans le Python courant. Retourne True si succes."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            timeout=300,
        )
        return True
    except Exception:
        return False


def ensure_ultralytics(status_fn=None) -> bool:
    """Verifie que ultralytics est installe ; tente de l'installer sinon."""
    global ULTRALYTICS_AVAILABLE, YOLO
    if ULTRALYTICS_AVAILABLE:
        return True
    if status_fn:
        status_fn("Installation automatique de ultralytics (YOLO)...")
    ok = _pip_install("ultralytics")
    if ok:
        try:
            from ultralytics import YOLO as _YOLO
            YOLO = _YOLO
            ULTRALYTICS_AVAILABLE = True
            if status_fn:
                status_fn("ultralytics installe avec succes.")
            return True
        except ImportError:
            pass
    return False


def ensure_pillow(status_fn=None) -> bool:
    """Verifie que Pillow est installe ; tente de l'installer sinon."""
    global PIL_AVAILABLE, Image
    if PIL_AVAILABLE:
        return True
    if status_fn:
        status_fn("Installation automatique de Pillow...")
    ok = _pip_install("Pillow")
    if ok:
        try:
            from PIL import Image as _Image
            Image = _Image
            PIL_AVAILABLE = True
            return True
        except ImportError:
            pass
    return False


# ══════════════════════════════════════════════════════════════════════════
# Detecteur principal
# ══════════════════════════════════════════════════════════════════════════

class YOLODetector(QObject):

    progress_changed   = pyqtSignal(int)
    status_changed     = pyqtSignal(str)
    detection_finished = pyqtSignal(str)
    detection_failed   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._model     = None

    def run(self, raster_path: str, model_path: str, classes: list,
            conf: float, iou: float, tile_size: int, output_path: str):

        # ── Verifications ───────────────────────────────────────────────
        if not NUMPY_AVAILABLE:
            self.detection_failed.emit(
                "numpy n'est pas installe.\n"
                "Dans OSGeo4W Shell : pip install numpy"
            )
            return

        if not GDAL_AVAILABLE and not RASTERIO_AVAILABLE:
            self.detection_failed.emit(
                "Ni GDAL ni rasterio ne sont disponibles.\n"
                "GDAL devrait etre present dans QGIS — verifiez votre installation."
            )
            return

        # Tenter d'installer ultralytics si absent
        if not ensure_ultralytics(self.status_changed.emit):
            self.detection_failed.emit(
                "ultralytics (YOLO) n'est pas installe.\n\n"
                "Solution : ouvrez OSGeo4W Shell (Menu Demarrer -> QGIS -> OSGeo4W Shell)\n"
                "puis tapez :\n"
                "  pip install ultralytics\n\n"
                "Relancez ensuite QGIS et reessayez."
            )
            return

        # Tenter d'installer Pillow si absent
        if not ensure_pillow(self.status_changed.emit):
            self.detection_failed.emit(
                "Pillow n'est pas installe.\n"
                "Dans OSGeo4W Shell : pip install Pillow"
            )
            return

        try:
            self._run_internal(raster_path, model_path, classes,
                               conf, iou, tile_size, output_path)
        except Exception as exc:
            if not self._cancelled:
                self.detection_failed.emit(str(exc))

    def cancel(self):
        self._cancelled = True

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline interne
    # ─────────────────────────────────────────────────────────────────────

    def _run_internal(self, raster_path, model_path, classes,
                      conf, iou, tile_size, output_path):

        # 1. Charger le modele
        self.status_changed.emit(
            f"Chargement du modele : {os.path.basename(model_path)}"
        )
        self._model = YOLO(model_path)
        self.progress_changed.emit(5)

        # 2. Ouvrir le raster (GDAL en priorite, rasterio en fallback)
        self.status_changed.emit(
            f"Ouverture du raster : {os.path.basename(raster_path)}"
        )

        if GDAL_AVAILABLE:
            self._run_with_gdal(
                raster_path, classes, conf, iou, tile_size, output_path
            )
        else:
            self._run_with_rasterio(
                raster_path, classes, conf, iou, tile_size, output_path
            )

    # ── Chemin GDAL ──────────────────────────────────────────────────────

    def _run_with_gdal(self, raster_path, classes, conf, iou,
                       tile_size, output_path):

        raster = GDALRaster(raster_path)
        width      = raster.width
        height     = raster.height
        band_count = raster.band_count
        crs_epsg   = raster.epsg()

        self.status_changed.emit(
            f"Raster : {width}x{height} px, {band_count} bandes, "
            f"CRS : {raster.crs_label()}"
        )
        self.progress_changed.emit(8)

        tiles       = self._compute_tiles(width, height, tile_size)
        total_tiles = len(tiles)
        self.status_changed.emit(
            f"Traitement de {total_tiles} tuiles ({tile_size}x{tile_size} px)..."
        )

        all_detections = []

        for tile_idx, (col_off, row_off, tile_w, tile_h) in enumerate(tiles):
            if self._cancelled:
                raise RuntimeError("Detection annulee.")

            pct = 8 + int((tile_idx / total_tiles) * 85)
            self.progress_changed.emit(pct)
            self.status_changed.emit(
                f"Tuile {tile_idx + 1}/{total_tiles}  "
                f"({col_off},{row_off}) {tile_w}x{tile_h} px"
            )

            try:
                tile_data = raster.read_tile(col_off, row_off, tile_w, tile_h)
            except Exception as e:
                self.status_changed.emit(f"  Lecture tuile ignoree : {e}")
                continue

            pil_img = self._to_pil_rgb(tile_data, band_count)
            if pil_img is None:
                continue

            results = self._model.predict(
                source=pil_img, conf=conf, iou=iou, verbose=False
            )

            for result in results:
                if result.boxes is None:
                    continue
                for box_data in result.boxes:
                    cls_id = int(box_data.cls[0].item())
                    score  = float(box_data.conf[0].item())
                    xyxy   = box_data.xyxy[0].tolist()

                    label = self._resolve_label(cls_id, classes)
                    if label is None:
                        continue

                    x1_r = col_off + xyxy[0]
                    y1_r = row_off + xyxy[1]
                    x2_r = col_off + xyxy[2]
                    y2_r = row_off + xyxy[3]

                    bbox_geo = raster.bbox_to_geo(x1_r, y1_r, x2_r, y2_r)

                    all_detections.append({
                        "label":    label,
                        "conf":     score,
                        "cls_id":   cls_id,
                        "bbox_geo": list(bbox_geo),
                        "bbox_px":  [x1_r, y1_r, x2_r, y2_r],
                    })

        raster.close()
        self._finalize(all_detections, iou, output_path, crs_epsg)

    # ── Chemin rasterio (fallback) ────────────────────────────────────────

    def _run_with_rasterio(self, raster_path, classes, conf, iou,
                           tile_size, output_path):
        import rasterio as _rio
        from rasterio.windows import Window

        with _rio.open(raster_path) as src:
            width      = src.width
            height     = src.height
            band_count = src.count
            transform  = src.transform
            try:
                crs_epsg = src.crs.to_epsg()
            except Exception:
                crs_epsg = None

            self.status_changed.emit(
                f"Raster : {width}x{height} px, {band_count} bandes"
            )
            self.progress_changed.emit(8)

            tiles       = self._compute_tiles(width, height, tile_size)
            total_tiles = len(tiles)
            all_detections = []

            for tile_idx, (col_off, row_off, tile_w, tile_h) in enumerate(tiles):
                if self._cancelled:
                    raise RuntimeError("Detection annulee.")

                pct = 8 + int((tile_idx / total_tiles) * 85)
                self.progress_changed.emit(pct)
                self.status_changed.emit(
                    f"Tuile {tile_idx + 1}/{total_tiles}..."
                )

                window    = Window(col_off, row_off, tile_w, tile_h)
                tile_data = src.read(window=window).astype(np.float32)
                pil_img   = self._to_pil_rgb(tile_data, band_count)
                if pil_img is None:
                    continue

                results = self._model.predict(
                    source=pil_img, conf=conf, iou=iou, verbose=False
                )

                for result in results:
                    if result.boxes is None:
                        continue
                    for box_data in result.boxes:
                        cls_id = int(box_data.cls[0].item())
                        score  = float(box_data.conf[0].item())
                        xyxy   = box_data.xyxy[0].tolist()

                        label = self._resolve_label(cls_id, classes)
                        if label is None:
                            continue

                        x1_r = col_off + xyxy[0]
                        y1_r = row_off + xyxy[1]
                        x2_r = col_off + xyxy[2]
                        y2_r = row_off + xyxy[3]

                        import rasterio.transform as rt
                        geo_minx, geo_maxy = rt.xy(transform, y1_r, x1_r, offset="ul")
                        geo_maxx, geo_miny = rt.xy(transform, y2_r, x2_r, offset="ul")

                        all_detections.append({
                            "label":    label,
                            "conf":     score,
                            "cls_id":   cls_id,
                            "bbox_geo": [geo_minx, geo_miny, geo_maxx, geo_maxy],
                            "bbox_px":  [x1_r, y1_r, x2_r, y2_r],
                        })

        self._finalize(all_detections, iou, output_path, crs_epsg)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers communs
    # ─────────────────────────────────────────────────────────────────────

    def _finalize(self, all_detections, iou, output_path, crs_epsg):
        self.status_changed.emit(
            f"{len(all_detections)} detections brutes — NMS en cours..."
        )
        self.progress_changed.emit(94)
        merged = self._apply_geo_nms(all_detections, iou_threshold=iou)
        self.status_changed.emit(
            f"{len(merged)} detections apres NMS."
        )
        self.progress_changed.emit(96)
        self._save_geojson(merged, output_path, crs_epsg)
        self.progress_changed.emit(100)
        self.detection_finished.emit(output_path)

    def _compute_tiles(self, width, height, tile_size):
        tiles = []
        step  = max(1, tile_size - TILE_OVERLAP)
        col   = 0
        while col < width:
            row = 0
            while row < height:
                tw = min(tile_size, width - col)
                th = min(tile_size, height - row)
                tiles.append((col, row, tw, th))
                row += step
            col += step
        return tiles

    def _to_pil_rgb(self, tile_data, band_count):
        if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
            return None
        if band_count >= 3:
            r = tile_data[0]
            g = tile_data[1]
            b = tile_data[2]
        else:
            r = g = b = tile_data[0]

        def norm(a):
            mn, mx = float(a.min()), float(a.max())
            if mx > mn:
                return ((a - mn) / (mx - mn) * 255).astype(np.uint8)
            return np.zeros(a.shape, dtype=np.uint8)

        rgb = np.stack([norm(r), norm(g), norm(b)], axis=-1)
        return Image.fromarray(rgb, "RGB")

    def _resolve_label(self, cls_id: int, requested_classes: list):
        """Mappe un cls_id COCO/custom vers un label ForestDL."""
        # Vehicules COCO standard
        if cls_id in COCO_VEHICLE_CLASSES and "vehicule" in requested_classes:
            return "Vehicule"

        # Modele custom ForestDL (classes 0-3)
        custom_map = {
            0: ("mangrove",       "Mangrove"),
            1: ("arbre_fruitier", "Arbre fruitier"),
            2: ("vehicule",       "Vehicule"),
            3: ("batiment",       "Batiment"),
        }
        if cls_id in custom_map:
            key, display = custom_map[cls_id]
            if key in requested_classes:
                return display

        return None

    def _apply_geo_nms(self, detections, iou_threshold):
        if not detections:
            return []
        if not SHAPELY_AVAILABLE:
            return detections

        by_label = {}
        for det in detections:
            by_label.setdefault(det["label"], []).append(det)

        result = []
        for label, dets in by_label.items():
            dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)
            kept        = []
            suppressed  = set()
            for i, di in enumerate(dets_sorted):
                if i in suppressed:
                    continue
                kept.append(di)
                bi = shapely_box(*di["bbox_geo"])
                ai = bi.area
                for j in range(i + 1, len(dets_sorted)):
                    if j in suppressed:
                        continue
                    bj    = shapely_box(*dets_sorted[j]["bbox_geo"])
                    inter = bi.intersection(bj).area
                    union = ai + bj.area - inter
                    if union > 0 and inter / union >= iou_threshold:
                        suppressed.add(j)
            result.extend(kept)
        return result

    def _save_geojson(self, detections, output_path, crs_epsg):
        features = []
        for det in detections:
            minx, miny, maxx, maxy = det["bbox_geo"]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[minx, maxy], [maxx, maxy],
                                     [maxx, miny], [minx, miny],
                                     [minx, maxy]]],
                },
                "properties": {
                    "label":      det["label"],
                    "confidence": round(det["conf"], 4),
                    "class_id":   det["cls_id"],
                },
            })

        geojson = {"type": "FeatureCollection", "features": features}
        if crs_epsg and crs_epsg != 4326:
            geojson["crs"] = {
                "type":       "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_epsg}"},
            }

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        self.status_changed.emit(
            f"GeoJSON sauvegarde : {output_path} ({len(features)} objets)"
        )
