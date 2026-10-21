import os
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


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
                    apex_candidates = [tx1 + tooth_w * 0.20, tooth_cx, tx2 - tooth_w * 0.20]
                    apex_x = min(apex_candidates, key=lambda ax: abs(cx - ax))
                else:
                    apex_x = tooth_cx
                if cy > (ty1 + tooth_h * 0.7):
                    continue
            elif is_mandibular:
                apex_y = ty2
                if is_molar:
                    apex_candidates = [tx1 + tooth_w * 0.15, tooth_cx, tx2 - tooth_w * 0.15]
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
    Standalone Periapical Lesion Predictor (YOLO11s + Multi-Rooted Apex FDI Attribution).
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
            return {"module_name": "Dental_012_periapical", "error": "Model not loaded", "lesions": []}

        results = self.model(image, conf=threshold_review, verbose=False)
        lesions: List[Dict[str, Any]] = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                status = "DEFINITE" if conf >= threshold_definite else "REVIEW_REQUIRED"
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
            match_lesions_to_teeth(lesions, teeth_data, anatomical_guard=anatomical_guard)

        return {"module_name": "Dental_012_periapical", "lesions": lesions}
