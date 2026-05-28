"""
YOLODetector — runs YOLOv8 object detection on large GeoTIFF orthophotos.

Strategy:
  1. Read raster metadata (CRS, affine transform) via rasterio.
  2. Tile the raster into overlapping tiles of configurable size.
  3. Run YOLO on each tile.
  4. Convert pixel-space bounding boxes to geographic coordinates.
  5. Merge all detections and apply NMS to remove duplicates from tile overlaps.
  6. Save output as GeoJSON.

Class mapping:
  - "vehicule"      → COCO classes 2 (car), 5 (bus), 7 (truck)
  - "mangrove"      → custom model class 0  (or user-supplied model)
  - "arbre_fruitier"→ custom model class 0  (or user-supplied model)
  - "batiment"      → custom model class 0  (or user-supplied model)
  When a standard COCO model is used, only vehicles can be detected
  reliably; other classes require a custom model.
"""

import os
import json
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import rasterio
    from rasterio.windows import Window
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
    from shapely.geometry import box as shapely_box, mapping as shapely_mapping
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    from qgis.PyQt.QtCore import QObject, pyqtSignal
except ImportError:
    from PyQt5.QtCore import QObject, pyqtSignal


# COCO class indices for vehicles
COCO_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# Label mapping: our class name → COCO class ids (empty = custom model needed)
CLASS_COCO_MAPPING = {
    "vehicule": [2, 5, 7],
    "mangrove": [],        # requires custom model
    "arbre_fruitier": [],  # requires custom model
    "batiment": [],        # requires custom model
}

# Overlap between tiles (pixels) to avoid missing objects at edges
TILE_OVERLAP = 64


class YOLODetector(QObject):
    """
    Performs tiled YOLO inference on a GeoTIFF and outputs a GeoJSON file.

    Designed to run inside a QThread.
    """

    progress_changed = pyqtSignal(int)    # 0-100
    status_changed = pyqtSignal(str)
    detection_finished = pyqtSignal(str)  # path to output GeoJSON
    detection_failed = pyqtSignal(str)    # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, raster_path: str, model_path: str, classes: list,
            conf: float, iou: float, tile_size: int, output_path: str):
        """
        Entry point called from the worker thread.

        :param raster_path:  Path to input GeoTIFF.
        :param model_path:   YOLO model filename (e.g. 'yolov8m.pt') or full path.
        :param classes:      List of class names to detect (from CLASS_COCO_MAPPING keys).
        :param conf:         Confidence threshold (0-1).
        :param iou:          IoU threshold for NMS (0-1).
        :param tile_size:    Tile size in pixels (e.g. 1024).
        :param output_path:  Output GeoJSON file path.
        """
        if not NUMPY_AVAILABLE:
            self.detection_failed.emit("numpy n'est pas installé.")
            return
        if not RASTERIO_AVAILABLE:
            self.detection_failed.emit(
                "rasterio n'est pas installé. Exécutez: pip install rasterio"
            )
            return
        if not ULTRALYTICS_AVAILABLE:
            self.detection_failed.emit(
                "ultralytics n'est pas installé. Exécutez: pip install ultralytics"
            )
            return
        if not PIL_AVAILABLE:
            self.detection_failed.emit(
                "Pillow n'est pas installé. Exécutez: pip install Pillow"
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

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _run_internal(self, raster_path, model_path, classes,
                      conf, iou, tile_size, output_path):
        # --- 1. Load model ---
        self.status_changed.emit(f"Chargement du modèle: {os.path.basename(model_path)}")
        self._model = YOLO(model_path)
        self.progress_changed.emit(5)

        # --- 2. Open raster ---
        self.status_changed.emit(f"Ouverture du raster: {os.path.basename(raster_path)}")
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            band_count = src.count

            self.status_changed.emit(
                f"Raster: {width}×{height} px, {band_count} bandes, CRS: {raster_crs}"
            )
            self.progress_changed.emit(8)

            # Determine CRS EPSG for GeoJSON output
            try:
                crs_epsg = raster_crs.to_epsg()
            except Exception:
                crs_epsg = None

            # --- 3. Tile the raster ---
            tiles = self._compute_tiles(width, height, tile_size)
            total_tiles = len(tiles)
            self.status_changed.emit(
                f"Traitement de {total_tiles} tuiles ({tile_size}×{tile_size} px)…"
            )

            all_detections = []  # list of dicts with geo bbox + label + conf

            for tile_idx, (col_off, row_off, tile_w, tile_h) in enumerate(tiles):
                if self._cancelled:
                    raise RuntimeError("Détection annulée.")

                pct = 8 + int((tile_idx / total_tiles) * 85)
                self.progress_changed.emit(pct)
                self.status_changed.emit(
                    f"Tuile {tile_idx + 1}/{total_tiles} "
                    f"(col={col_off}, row={row_off})…"
                )

                # Read tile from raster
                window = Window(col_off, row_off, tile_w, tile_h)
                tile_data = src.read(window=window)  # shape: (bands, h, w)

                # Convert to RGB PIL image
                pil_img = self._to_pil_rgb(tile_data, band_count)
                if pil_img is None:
                    continue

                # Run YOLO
                results = self._model.predict(
                    source=pil_img,
                    conf=conf,
                    iou=iou,
                    verbose=False,
                )

                # Parse detections
                tile_transform = rasterio.transform.rowcol  # for reference
                for result in results:
                    if result.boxes is None:
                        continue
                    for box_data in result.boxes:
                        cls_id = int(box_data.cls[0].item())
                        score = float(box_data.conf[0].item())
                        xyxy = box_data.xyxy[0].tolist()  # [x1, y1, x2, y2] in tile pixels

                        label = self._resolve_label(cls_id, classes, model_path)
                        if label is None:
                            continue

                        # Convert tile-pixel coords to raster-pixel coords
                        x1_raster = col_off + xyxy[0]
                        y1_raster = row_off + xyxy[1]
                        x2_raster = col_off + xyxy[2]
                        y2_raster = row_off + xyxy[3]

                        # Convert raster-pixel coords to geographic coords
                        geo_minx, geo_maxy = rasterio.transform.xy(
                            transform, y1_raster, x1_raster, offset="ul"
                        )
                        geo_maxx, geo_miny = rasterio.transform.xy(
                            transform, y2_raster, x2_raster, offset="ul"
                        )

                        all_detections.append({
                            "label": label,
                            "conf": score,
                            "cls_id": cls_id,
                            "bbox_geo": [geo_minx, geo_miny, geo_maxx, geo_maxy],
                            "bbox_px": [x1_raster, y1_raster, x2_raster, y2_raster],
                        })

        # --- 4. NMS across tile boundaries ---
        self.status_changed.emit(f"{len(all_detections)} détections brutes. Application NMS…")
        self.progress_changed.emit(94)
        merged = self._apply_geo_nms(all_detections, iou_threshold=iou)
        self.status_changed.emit(f"{len(merged)} détections après NMS.")

        # --- 5. Save GeoJSON ---
        self.progress_changed.emit(96)
        self._save_geojson(merged, output_path, crs_epsg)
        self.progress_changed.emit(100)
        self.detection_finished.emit(output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_tiles(self, width: int, height: int, tile_size: int) -> list:
        """Return list of (col_off, row_off, tile_w, tile_h) tuples."""
        tiles = []
        step = tile_size - TILE_OVERLAP
        col = 0
        while col < width:
            row = 0
            while row < height:
                tw = min(tile_size, width - col)
                th = min(tile_size, height - row)
                tiles.append((col, row, tw, th))
                row += step
            col += step
        return tiles

    def _to_pil_rgb(self, tile_data: "np.ndarray", band_count: int) -> "Image.Image":
        """Convert rasterio band array (bands, H, W) to PIL RGB image."""
        if band_count >= 3:
            r = tile_data[0].astype(np.float32)
            g = tile_data[1].astype(np.float32)
            b = tile_data[2].astype(np.float32)
        elif band_count == 1:
            gray = tile_data[0].astype(np.float32)
            r = g = b = gray
        else:
            r = tile_data[0].astype(np.float32)
            g = tile_data[1].astype(np.float32) if band_count > 1 else r
            b = r

        def normalize(arr):
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
            return np.zeros_like(arr, dtype=np.uint8)

        rgb = np.stack([normalize(r), normalize(g), normalize(b)], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    def _resolve_label(self, cls_id: int, requested_classes: list, model_path: str) -> str | None:
        """
        Map a YOLO class ID to one of our detection labels.
        Returns None if the detection is not relevant to the requested classes.
        """
        # For standard models (yolov8*.pt without custom path), use COCO mapping
        is_custom = os.path.isabs(model_path) and os.path.exists(model_path)

        if not is_custom:
            # Standard COCO model
            if "vehicule" in requested_classes and cls_id in COCO_VEHICLE_CLASSES:
                return "Vehicule"
            # Other classes need custom model; skip
            return None
        else:
            # Custom model: assume class 0 = mangrove, 1 = arbre_fruitier,
            # 2 = vehicule, 3 = batiment (user must train accordingly)
            # But we also respect COCO vehicle IDs if they appear
            label_map = {
                0: ("mangrove", "Mangrove"),
                1: ("arbre_fruitier", "Arbre fruitier"),
                2: ("vehicule", "Vehicule"),
                3: ("batiment", "Batiment"),
            }
            if cls_id in label_map:
                key, display = label_map[cls_id]
                if key in requested_classes:
                    return display
            # Fallback: check COCO vehicles
            if "vehicule" in requested_classes and cls_id in COCO_VEHICLE_CLASSES:
                return "Vehicule"
            return None

    def _apply_geo_nms(self, detections: list, iou_threshold: float) -> list:
        """
        Non-Maximum Suppression on geographic bounding boxes.
        Groups detections by label and applies NMS within each group.
        """
        if not detections:
            return []

        if not SHAPELY_AVAILABLE:
            # Without shapely, just return all detections
            return detections

        # Group by label
        by_label = {}
        for det in detections:
            lbl = det["label"]
            by_label.setdefault(lbl, []).append(det)

        result = []
        for label, dets in by_label.items():
            kept = self._nms_for_group(dets, iou_threshold)
            result.extend(kept)

        return result

    @staticmethod
    def _nms_for_group(dets: list, iou_threshold: float) -> list:
        """Apply NMS to a list of detections of the same label."""
        if len(dets) <= 1:
            return dets

        # Sort by confidence descending
        dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)
        kept = []
        suppressed = set()

        for i, det_i in enumerate(dets_sorted):
            if i in suppressed:
                continue
            kept.append(det_i)
            box_i = shapely_box(*det_i["bbox_geo"])
            area_i = box_i.area

            for j in range(i + 1, len(dets_sorted)):
                if j in suppressed:
                    continue
                box_j = shapely_box(*dets_sorted[j]["bbox_geo"])
                intersection = box_i.intersection(box_j).area
                union = area_i + box_j.area - intersection
                if union > 0 and (intersection / union) >= iou_threshold:
                    suppressed.add(j)

        return kept

    def _save_geojson(self, detections: list, output_path: str, crs_epsg):
        """Save detections as a GeoJSON FeatureCollection."""
        features = []
        for det in detections:
            minx, miny, maxx, maxy = det["bbox_geo"]
            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [minx, maxy],
                    [maxx, maxy],
                    [maxx, miny],
                    [minx, miny],
                    [minx, maxy],
                ]]
            }
            properties = {
                "label": det["label"],
                "confidence": round(det["conf"], 4),
                "class_id": det["cls_id"],
            }
            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }

        # Add CRS if known and not EPSG:4326 (GeoJSON default)
        if crs_epsg and crs_epsg != 4326:
            geojson["crs"] = {
                "type": "name",
                "properties": {
                    "name": f"urn:ogc:def:crs:EPSG::{crs_epsg}"
                }
            }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        self.status_changed.emit(
            f"GeoJSON sauvegardé: {output_path} ({len(features)} objets)"
        )
