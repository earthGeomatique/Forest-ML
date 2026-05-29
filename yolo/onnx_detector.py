"""
ONNXDetector — inférence sur modèles ONNX du zoo ONNX (non-ultralytics).

Supporte :
  - YOLOv3 / YOLOv3-Tiny  (onnx/models)
  - YOLOv4
  - SSD MobileNet
  - Faster-RCNN
  - Tout modèle ONNX avec entrée image NCHW

Pour les modèles exportés depuis ultralytics (.onnx),
utiliser directement YOLODetector (ultralytics gère nativement).
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
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

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

TILE_OVERLAP = 64

# ── Classes COCO pour véhicules ────────────────────────────────────────────
COCO_VEHICLE_IDS = {2, 5, 7}  # car, bus, truck
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat",
    9: "traffic light", 14: "bird", 15: "cat", 16: "dog",
    17: "horse", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 62: "tv", 63: "laptop",
}

# ── Modèles ONNX zoo prédéfinis ────────────────────────────────────────────
ONNX_ZOO_MODELS = {
    "YOLOv3 (ONNX Zoo)": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx",
        "type": "yolov3",
        "input_size": (416, 416),
    },
    "YOLOv3-Tiny (ONNX Zoo)": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx",
        "type": "yolov3_tiny",
        "input_size": (416, 416),
    },
    "SSD MobileNet (ONNX Zoo)": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx",
        "type": "ssd",
        "input_size": (1200, 1200),
    },
    "Faster-RCNN ResNet50 (ONNX Zoo)": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx",
        "type": "faster_rcnn",
        "input_size": (800, 800),
    },
}


class ONNXDetector(QObject):
    """
    Inférence ONNX générique sur grandes orthophotos GeoTIFF.
    Tuilage + conversion coordonnées pixel → géographiques.
    """

    progress_changed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    detection_finished = pyqtSignal(str)
    detection_failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._session = None
        self._model_type = "generic"

    def run(self, raster_path: str, model_path: str, model_type: str,
            classes: list, conf: float, iou: float,
            tile_size: int, output_path: str):
        if not NUMPY_AVAILABLE:
            self.detection_failed.emit("numpy n'est pas installé.")
            return
        if not ORT_AVAILABLE:
            self.detection_failed.emit(
                "onnxruntime n'est pas installé. Exécutez: pip install onnxruntime"
            )
            return
        if not RASTERIO_AVAILABLE:
            self.detection_failed.emit(
                "rasterio n'est pas installé. Exécutez: pip install rasterio"
            )
            return
        if not PIL_AVAILABLE:
            self.detection_failed.emit("Pillow n'est pas installé.")
            return

        self._model_type = model_type
        try:
            self._run_internal(raster_path, model_path, classes,
                               conf, iou, tile_size, output_path)
        except Exception as exc:
            if not self._cancelled:
                self.detection_failed.emit(str(exc))

    def cancel(self):
        self._cancelled = True

    # ─────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────

    def _run_internal(self, raster_path, model_path, classes,
                      conf, iou, tile_size, output_path):
        # 1. Load ONNX session
        self.status_changed.emit(f"Chargement modèle ONNX: {os.path.basename(model_path)}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # suppress verbose logs
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._session = ort.InferenceSession(model_path, opts, providers=providers)
        except Exception:
            self._session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

        input_name = self._session.get_inputs()[0].name
        input_shape = self._session.get_inputs()[0].shape  # [batch, C, H, W] or [batch, H, W, C]
        self.progress_changed.emit(5)

        # Determine model input size
        try:
            model_h = int(input_shape[2]) if input_shape[2] else tile_size
            model_w = int(input_shape[3]) if input_shape[3] else tile_size
        except Exception:
            model_h = model_w = 640

        # 2. Open raster
        self.status_changed.emit(f"Ouverture raster: {os.path.basename(raster_path)}")
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            band_count = src.count

            try:
                crs_epsg = raster_crs.to_epsg()
            except Exception:
                crs_epsg = None

            tiles = self._compute_tiles(width, height, tile_size)
            total_tiles = len(tiles)
            self.status_changed.emit(f"Traitement {total_tiles} tuiles…")
            self.progress_changed.emit(8)

            all_detections = []

            for tile_idx, (col_off, row_off, tile_w, tile_h) in enumerate(tiles):
                if self._cancelled:
                    raise RuntimeError("Annulé.")

                pct = 8 + int((tile_idx / total_tiles) * 85)
                self.progress_changed.emit(pct)
                self.status_changed.emit(
                    f"Tuile {tile_idx + 1}/{total_tiles}…"
                )

                window = Window(col_off, row_off, tile_w, tile_h)
                tile_data = src.read(window=window)
                pil_img = self._to_pil_rgb(tile_data, band_count)
                if pil_img is None:
                    continue

                # Resize to model input size
                pil_resized = pil_img.resize((model_w, model_h), Image.BILINEAR)
                scale_x = tile_w / model_w
                scale_y = tile_h / model_h

                # Preprocess
                img_array = np.array(pil_resized).astype(np.float32) / 255.0
                img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
                img_array = np.expand_dims(img_array, axis=0)   # NCHW

                # Inference
                try:
                    outputs = self._session.run(None, {input_name: img_array})
                except Exception as exc:
                    self.status_changed.emit(f"Erreur tuile {tile_idx}: {exc}")
                    continue

                # Parse outputs based on model type
                dets = self._parse_outputs(
                    outputs, classes, conf,
                    col_off, row_off, scale_x, scale_y,
                    transform
                )
                all_detections.extend(dets)

        # 3. NMS
        self.status_changed.emit(f"{len(all_detections)} détections brutes. NMS…")
        self.progress_changed.emit(94)
        merged = self._apply_nms(all_detections, iou)

        # 4. Save
        self.progress_changed.emit(96)
        self._save_geojson(merged, output_path, crs_epsg)
        self.progress_changed.emit(100)
        self.detection_finished.emit(output_path)

    def _parse_outputs(self, outputs, classes, conf_thresh,
                       col_off, row_off, scale_x, scale_y, transform):
        """Parse ONNX outputs generically. Returns list of detection dicts."""
        detections = []

        # Try to handle common output formats
        if self._model_type in ("yolov3", "yolov3_tiny"):
            detections = self._parse_yolov3(outputs, classes, conf_thresh,
                                             col_off, row_off, scale_x, scale_y, transform)
        elif self._model_type == "ssd":
            detections = self._parse_ssd(outputs, classes, conf_thresh,
                                          col_off, row_off, scale_x, scale_y, transform)
        elif self._model_type == "faster_rcnn":
            detections = self._parse_faster_rcnn(outputs, classes, conf_thresh,
                                                   col_off, row_off, scale_x, scale_y, transform)
        else:
            # Generic: assume output[0] has shape [N, 6] = [x1,y1,x2,y2,conf,cls]
            detections = self._parse_generic(outputs, classes, conf_thresh,
                                              col_off, row_off, scale_x, scale_y, transform)

        return detections

    def _parse_yolov3(self, outputs, classes, conf_thresh,
                      col_off, row_off, scale_x, scale_y, transform):
        """Parse YOLOv3 outputs: [boxes(N,4), scores(N,80), indices(K,3)]"""
        detections = []
        if len(outputs) < 3:
            return detections
        try:
            boxes = outputs[0][0]    # (N, 4) in yxyx or xywh
            scores = outputs[1][0]   # (N, 80)
            indices = outputs[2]     # (K, 3): [batch, class, box_idx]

            if indices is None or len(indices) == 0:
                return detections

            for idx in indices:
                batch_id, cls_id, box_id = int(idx[0]), int(idx[1]), int(idx[2])
                score = float(scores[box_id][cls_id])
                if score < conf_thresh:
                    continue

                label = self._coco_to_label(cls_id, classes)
                if label is None:
                    continue

                # YOLOv3 box format: [y1, x1, y2, x2] normalized
                y1, x1, y2, x2 = boxes[box_id]
                x1_px = x1 * scale_x + col_off
                y1_px = y1 * scale_y + row_off
                x2_px = x2 * scale_x + col_off
                y2_px = y2 * scale_y + row_off

                geo_minx, geo_maxy = rasterio.transform.xy(transform, y1_px, x1_px, offset="ul")
                geo_maxx, geo_miny = rasterio.transform.xy(transform, y2_px, x2_px, offset="ul")

                detections.append({
                    "label": label,
                    "conf": score,
                    "cls_id": cls_id,
                    "bbox_geo": [geo_minx, geo_miny, geo_maxx, geo_maxy],
                    "bbox_px": [x1_px, y1_px, x2_px, y2_px],
                })
        except Exception:
            pass
        return detections

    def _parse_ssd(self, outputs, classes, conf_thresh,
                   col_off, row_off, scale_x, scale_y, transform):
        """Parse SSD outputs: [boxes(1,N,4), labels(1,N), scores(1,N)]"""
        detections = []
        if len(outputs) < 3:
            return detections
        try:
            boxes = outputs[0][0]    # (N, 4) normalized [y1,x1,y2,x2]
            labels = outputs[1][0]   # (N,) class ids
            scores = outputs[2][0]   # (N,) confidence

            for i in range(len(scores)):
                score = float(scores[i])
                if score < conf_thresh:
                    continue
                cls_id = int(labels[i])
                label = self._coco_to_label(cls_id, classes)
                if label is None:
                    continue

                y1, x1, y2, x2 = boxes[i]
                x1_px = float(x1) * scale_x + col_off
                y1_px = float(y1) * scale_y + row_off
                x2_px = float(x2) * scale_x + col_off
                y2_px = float(y2) * scale_y + row_off

                geo_minx, geo_maxy = rasterio.transform.xy(transform, y1_px, x1_px, offset="ul")
                geo_maxx, geo_miny = rasterio.transform.xy(transform, y2_px, x2_px, offset="ul")

                detections.append({
                    "label": label,
                    "conf": score,
                    "cls_id": cls_id,
                    "bbox_geo": [geo_minx, geo_miny, geo_maxx, geo_maxy],
                    "bbox_px": [x1_px, y1_px, x2_px, y2_px],
                })
        except Exception:
            pass
        return detections

    def _parse_faster_rcnn(self, outputs, classes, conf_thresh,
                            col_off, row_off, scale_x, scale_y, transform):
        """Parse Faster-RCNN outputs: [boxes(N,4), labels(N), scores(N)]"""
        detections = []
        if len(outputs) < 3:
            return detections
        try:
            boxes = outputs[0]    # (N, 4) [x1,y1,x2,y2]
            labels = outputs[1]   # (N,)
            scores = outputs[2]   # (N,)

            for i in range(len(scores)):
                score = float(scores[i])
                if score < conf_thresh:
                    continue
                cls_id = int(labels[i])
                label = self._coco_to_label(cls_id, classes)
                if label is None:
                    continue

                x1, y1, x2, y2 = boxes[i]
                x1_px = float(x1) * scale_x + col_off
                y1_px = float(y1) * scale_y + row_off
                x2_px = float(x2) * scale_x + col_off
                y2_px = float(y2) * scale_y + row_off

                geo_minx, geo_maxy = rasterio.transform.xy(transform, y1_px, x1_px, offset="ul")
                geo_maxx, geo_miny = rasterio.transform.xy(transform, y2_px, x2_px, offset="ul")

                detections.append({
                    "label": label,
                    "conf": score,
                    "cls_id": cls_id,
                    "bbox_geo": [geo_minx, geo_miny, geo_maxx, geo_maxy],
                    "bbox_px": [x1_px, y1_px, x2_px, y2_px],
                })
        except Exception:
            pass
        return detections

    def _parse_generic(self, outputs, classes, conf_thresh,
                       col_off, row_off, scale_x, scale_y, transform):
        """Fallback: assume output[0] has shape [N, >=6] = [x1,y1,x2,y2,conf,cls,...]"""
        detections = []
        try:
            data = outputs[0]
            if data.ndim == 3:
                data = data[0]
            for row in data:
                if len(row) < 6:
                    continue
                x1, y1, x2, y2, score, cls_id = row[:6]
                score = float(score)
                if score < conf_thresh:
                    continue
                cls_id = int(cls_id)
                label = self._coco_to_label(cls_id, classes)
                if label is None:
                    continue

                x1_px = float(x1) * scale_x + col_off
                y1_px = float(y1) * scale_y + row_off
                x2_px = float(x2) * scale_x + col_off
                y2_px = float(y2) * scale_y + row_off

                geo_minx, geo_maxy = rasterio.transform.xy(transform, y1_px, x1_px, offset="ul")
                geo_maxx, geo_miny = rasterio.transform.xy(transform, y2_px, x2_px, offset="ul")

                detections.append({
                    "label": label,
                    "conf": score,
                    "cls_id": cls_id,
                    "bbox_geo": [geo_minx, geo_miny, geo_maxx, geo_maxy],
                    "bbox_px": [x1_px, y1_px, x2_px, y2_px],
                })
        except Exception:
            pass
        return detections

    def _coco_to_label(self, cls_id: int, requested_classes: list):
        """Map COCO class id to ForestDL label, or None if not requested."""
        if cls_id in COCO_VEHICLE_IDS and "vehicule" in requested_classes:
            return "Vehicule"
        return None

    def _compute_tiles(self, width, height, tile_size):
        tiles = []
        step = max(1, tile_size - TILE_OVERLAP)
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

    def _to_pil_rgb(self, tile_data, band_count):
        if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
            return None
        if band_count >= 3:
            r = tile_data[0].astype(np.float32)
            g = tile_data[1].astype(np.float32)
            b = tile_data[2].astype(np.float32)
        else:
            r = g = b = tile_data[0].astype(np.float32)

        def norm(a):
            mn, mx = a.min(), a.max()
            if mx > mn:
                return ((a - mn) / (mx - mn) * 255).astype(np.uint8)
            return np.zeros_like(a, dtype=np.uint8)

        rgb = np.stack([norm(r), norm(g), norm(b)], axis=-1)
        return Image.fromarray(rgb, "RGB")

    def _apply_nms(self, detections, iou_threshold):
        if not detections or not SHAPELY_AVAILABLE:
            return detections
        by_label = {}
        for det in detections:
            by_label.setdefault(det["label"], []).append(det)

        result = []
        for label, dets in by_label.items():
            dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)
            kept = []
            suppressed = set()
            for i, di in enumerate(dets_sorted):
                if i in suppressed:
                    continue
                kept.append(di)
                bi = shapely_box(*di["bbox_geo"])
                ai = bi.area
                for j in range(i + 1, len(dets_sorted)):
                    if j in suppressed:
                        continue
                    bj = shapely_box(*dets_sorted[j]["bbox_geo"])
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
            geom = {
                "type": "Polygon",
                "coordinates": [[[minx, maxy], [maxx, maxy],
                                  [maxx, miny], [minx, miny], [minx, maxy]]]
            }
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "label": det["label"],
                    "confidence": round(det["conf"], 4),
                    "class_id": det["cls_id"],
                },
            })
        geojson = {"type": "FeatureCollection", "features": features}
        if crs_epsg and crs_epsg != 4326:
            geojson["crs"] = {"type": "name",
                              "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_epsg}"}}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        self.status_changed.emit(f"GeoJSON sauvegardé: {output_path} ({len(features)} objets)")


def list_zoo_models():
    """Return dict of available ONNX zoo models."""
    return dict(ONNX_ZOO_MODELS)


def download_zoo_model(model_name: str, dest_dir: str, progress_callback=None) -> str:
    """
    Download an ONNX zoo model to dest_dir.
    Returns the local file path.
    """
    import urllib.request

    if model_name not in ONNX_ZOO_MODELS:
        raise ValueError(f"Modèle inconnu: {model_name}")

    url = ONNX_ZOO_MODELS[model_name]["url"]
    filename = os.path.basename(url)
    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path):
        return dest_path

    os.makedirs(dest_dir, exist_ok=True)

    def reporthook(block_num, block_size, total_size):
        if progress_callback and total_size > 0:
            pct = min(100, int(block_num * block_size / total_size * 100))
            progress_callback(pct)

    urllib.request.urlretrieve(url, dest_path, reporthook)
    return dest_path
