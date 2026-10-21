import os
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def extract_apex_roi_patches(
    image: np.ndarray,
    teeth_data: List[Dict[str, Any]],
    patch_size: int = 256,
    margin_ratio: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    2-Stage Periapical ROI Extractor:
    Crops high-resolution patches centered around the anatomical root apex of each tooth
    (superior border ty1 for maxillary teeth FDI 11-28; inferior border ty2 for mandibular teeth FDI 31-48)
    with CLAHE contrast enhancement for 2-Stage periapical lesion detection.
    """
    if image is None or image.size == 0 or not teeth_data:
        return []

    img_h, img_w = image.shape[:2]
    patches: List[Dict[str, Any]] = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for tooth in teeth_data:
        fdi = tooth.get("fdi")
        if fdi is None:
            continue
        try:
            fdi_num = int(str(fdi).lstrip("#"))
        except (ValueError, TypeError):
            continue

        bbox = tooth.get("bbox")
        contour = tooth.get("contour")
        if bbox is not None and len(bbox) == 4:
            tx1, ty1, tx2, ty2 = [float(c) for c in bbox]
        elif contour is not None and len(contour) > 0:
            pts = np.asarray(contour).reshape(-1, 2)
            tx1, ty1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
            tx2, ty2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
        else:
            continue

        tw = max(16.0, tx2 - tx1)
        th = max(16.0, ty2 - ty1)
        tcx = (tx1 + tx2) / 2.0

        is_maxillary = 11 <= fdi_num <= 28
        apex_y = ty1 if is_maxillary else ty2

        half_w = max(tw * (0.5 + margin_ratio), 32.0)
        half_h = max(th * margin_ratio, 32.0)

        cx1 = int(max(0, round(tcx - half_w)))
        cx2 = int(min(img_w, round(tcx + half_w)))
        if is_maxillary:
            cy1 = int(max(0, round(apex_y - half_h * 1.2)))
            cy2 = int(min(img_h, round(apex_y + half_h * 0.8)))
        else:
            cy1 = int(max(0, round(apex_y - half_h * 0.8)))
            cy2 = int(min(img_h, round(apex_y + half_h * 1.2)))

        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            continue

        crop = image[cy1:cy2, cx1:cx2].copy()
        if crop.ndim == 3 and crop.shape[2] == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            lab = cv2.merge((clahe.apply(l_ch), a_ch, b_ch))
            crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        elif crop.ndim == 2:
            crop = clahe.apply(crop)

        resized_patch = cv2.resize(
            crop, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC
        )
        patches.append(
            {
                "fdi": fdi_num,
                "patch": resized_patch,
                "crop_coords": (cx1, cy1, cx2, cy2),
                "patch_size": patch_size,
            }
        )

    return patches


def project_patch_bbox_to_global(
    patch_bbox: List[float],
    crop_coords: Tuple[int, int, int, int],
    patch_size: int = 256,
) -> List[int]:
    """
    Projects a bounding box [rx1, ry1, rx2, ry2] predicted on a resized square apex patch
    back to the original panoramic image global coordinates [gx1, gy1, gx2, gy2].
    """
    cx1, cy1, cx2, cy2 = crop_coords
    pw = max(1, cx2 - cx1)
    ph = max(1, cy2 - cy1)
    rx1, ry1, rx2, ry2 = [float(v) for v in patch_bbox[:4]]

    gx1 = int(round(cx1 + (rx1 / float(patch_size)) * pw))
    gy1 = int(round(cy1 + (ry1 / float(patch_size)) * ph))
    gx2 = int(round(cx1 + (rx2 / float(patch_size)) * pw))
    gy2 = int(round(cy1 + (ry2 / float(patch_size)) * ph))
    return [gx1, gy1, gx2, gy2]


def match_lesions_to_teeth(
    lesions: List[Dict[str, Any]],
    teeth_data: List[Dict[str, Any]],
    anatomical_guard: bool = True,
) -> List[Dict[str, Any]]:
    """
    Matches detected periapical lesions to the nearest anatomical tooth root apex.
    - Maxillary teeth (FDI 11-28): root apex is near the superior border (ty1);
      molars (6, 7, 8) use multi-rooted mesial/central/distal apex candidate anchors.
    - Mandibular teeth (FDI 31-48): root apex is near the inferior border (ty2);
      molars (6, 7, 8) use multi-rooted mesial/central/distal apex candidate anchors.
    - Unmatched radiolucencies receive isolated_anatomical_suspicion=True when
      anatomical_guard is enabled (protecting against mental foramen / maxillary sinus FPs).
    """
    for lesion in lesions:
        lx1, ly1, lx2, ly2 = [float(v) for v in lesion["bbox"][:4]]
        cx = (lx1 + lx2) / 2.0
        cy = (ly1 + ly2) / 2.0

        best_fdi: Optional[int] = None
        min_dist = float("inf")

        for tooth in teeth_data:
            fdi = tooth.get("fdi")
            if fdi is None:
                continue
            try:
                fdi_num = int(str(fdi).lstrip("#"))
            except (ValueError, TypeError):
                continue

            contour = tooth.get("contour")
            bbox = tooth.get("bbox")

            if bbox is not None and len(bbox) == 4:
                tx1, ty1, tx2, ty2 = [float(c) for c in bbox]
            elif contour is not None and len(contour) > 0:
                pts = np.asarray(contour).reshape(-1, 2)
                tx1, ty1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                tx2, ty2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
            else:
                continue

            tooth_cx = (tx1 + tx2) / 2.0
            tooth_w = max(10.0, tx2 - tx1)
            tooth_h = max(10.0, ty2 - ty1)

            is_maxillary = 11 <= fdi_num <= 28
            is_mandibular = 31 <= fdi_num <= 48
            is_molar = (fdi_num % 10) in (6, 7, 8)

            if is_maxillary:
                apex_y = ty1
                if is_molar:
                    apex_candidates = [
                        tx1 + tooth_w * 0.20,
                        tooth_cx,
                        tx2 - tooth_w * 0.20,
                    ]
                    apex_x = min(apex_candidates, key=lambda ax: abs(cx - ax))
                else:
                    apex_x = tooth_cx
                if cy > (ty1 + tooth_h * 0.7):
                    continue
            elif is_mandibular:
                apex_y = ty2
                if is_molar:
                    apex_candidates = [
                        tx1 + tooth_w * 0.15,
                        tooth_cx,
                        tx2 - tooth_w * 0.15,
                    ]
                    apex_x = min(apex_candidates, key=lambda ax: abs(cx - ax))
                else:
                    apex_x = tooth_cx
                if cy < (ty2 - tooth_h * 0.7):
                    continue
            else:
                apex_x = tooth_cx
                apex_y = (ty1 + ty2) / 2.0

            dx = abs(cx - apex_x)
            dy = abs(cy - apex_y)

            if dx > (tooth_w * 1.5):
                continue

            dist = float(np.sqrt((dx * 1.3) ** 2 + (dy * 0.9) ** 2))

            if fdi_num % 10 == 8:
                dist += 40.0
            if (fdi_num % 10 == 6) and dist < 100.0:
                dist -= 40.0

            if dist < min_dist:
                min_dist = dist
                best_fdi = fdi_num

        if best_fdi is not None and min_dist < 180.0:
            lesion["fdi"] = best_fdi
            lesion["apex_distance"] = round(float(min_dist), 1)
            lesion["isolated_anatomical_suspicion"] = False
        else:
            lesion["fdi"] = None
            if anatomical_guard:
                lesion["isolated_anatomical_suspicion"] = True

    return lesions


class PeriapicalPredictor:
    """
    Standalone Periapical Lesion Predictor (YOLO11s + Multi-Rooted Apex FDI Attribution
    + 2-Stage Apex Patch Inference Support).
    """

    def __init__(self, model_path: str = "models/best.onnx"):
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        if self.model is None and YOLO is not None and os.path.exists(self.model_path):
            self.model = YOLO(self.model_path, task="detect")

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    def predict(
        self,
        image: np.ndarray,
        teeth_data: Optional[List[Dict[str, Any]]] = None,
        threshold_definite: float = 0.50,
        threshold_review: float = 0.25,
        anatomical_guard: bool = True,
    ) -> Dict[str, Any]:
        self.load_model()
        if self.model is None:
            return {
                "module_name": "Dental_012_periapical",
                "error": "Model not loaded",
                "lesions": [],
            }

        results = self.model(image, conf=threshold_review, verbose=False)
        lesions: List[Dict[str, Any]] = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                status = (
                    "DEFINITE" if conf >= threshold_definite else "REVIEW_REQUIRED"
                )
                lesions.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf, 2),
                        "clinical_status": status,
                        "fdi": None,
                        "isolated_anatomical_suspicion": False,
                    }
                )

        if teeth_data and lesions:
            match_lesions_to_teeth(
                lesions, teeth_data, anatomical_guard=anatomical_guard
            )

        return {"module_name": "Dental_012_periapical", "lesions": lesions}

    def predict_two_stage_apex_patches(
        self,
        image: np.ndarray,
        teeth_data: List[Dict[str, Any]],
        patch_size: int = 256,
        threshold_definite: float = 0.50,
        threshold_review: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Executes 2-Stage periapical lesion detection on high-resolution root apex patches
        and projects detections back to global panoramic coordinates.
        """
        self.load_model()
        if self.model is None:
            return {
                "module_name": "Dental_012_periapical_2stage",
                "error": "Model not loaded",
                "lesions": [],
            }

        apex_patches = extract_apex_roi_patches(
            image, teeth_data, patch_size=patch_size
        )
        lesions: List[Dict[str, Any]] = []
        for item in apex_patches:
            fdi_num = item["fdi"]
            patch_img = item["patch"]
            crop_coords = item["crop_coords"]
            results = self.model(patch_img, conf=threshold_review, verbose=False)
            if len(results) > 0:
                for box in results[0].boxes:
                    raw_xyxy = box.xyxy[0].tolist()
                    global_bbox = project_patch_bbox_to_global(
                        raw_xyxy, crop_coords, patch_size=patch_size
                    )
                    conf = float(box.conf[0].item())
                    status = (
                        "DEFINITE"
                        if conf >= threshold_definite
                        else "REVIEW_REQUIRED"
                    )
                    lesions.append(
                        {
                            "bbox": global_bbox,
                            "confidence": round(conf, 2),
                            "clinical_status": status,
                            "fdi": fdi_num,
                            "isolated_anatomical_suspicion": False,
                        }
                    )

        return {"module_name": "Dental_012_periapical_2stage", "lesions": lesions}
