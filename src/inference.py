import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import cv2
import numpy as np

from dental_core.schemas.panorama_contract import (
    LesionAnnotation,
    PanoramaHiTLDocument,
    ToothAnnotation,
    compute_tooth_axis_and_apex,
)

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

        bbox = tooth.get("bbox_xyxy") or tooth.get("bbox")
        contour = tooth.get("polygon_px") or tooth.get("contour")
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

        is_maxillary = (fdi_num // 10) in (1, 2, 5, 6)
        apex_y = ty1 if is_maxillary else ty2
        if contour is not None and len(contour) >= 4:
            axis_info = compute_tooth_axis_and_apex(fdi_num, [tx1, ty1, tx2, ty2], contour)
            tcx, apex_y = axis_info["apex_xy"]

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
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            lab = cv2.merge((clahe.apply(l_ch), a_ch, b_ch))
            crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
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


def _collect_rct_set(
    teeth_data: Sequence[Dict[str, Any]],
    rct_fdis: Optional[Sequence[int]] = None,
) -> Set[int]:
    rct_set: Set[int] = set(int(f) for f in (rct_fdis or []))
    for tooth in teeth_data:
        fdi = tooth.get("fdi")
        if fdi is None:
            continue
        try:
            fdi_int = int(str(fdi).lstrip("#"))
        except (TypeError, ValueError):
            continue
        tags = tooth.get("restoration_tags") or []
        if any("endodontic" in str(t).lower() or "rct" in str(t).lower() for t in tags):
            rct_set.add(fdi_int)
    return rct_set


def match_lesions_to_teeth(
    lesions: List[Dict[str, Any]],
    teeth_data: List[Dict[str, Any]],
    anatomical_guard: bool = True,
    rct_fdis: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Matches detected periapical lesions to the nearest anatomical tooth root apex,
    incorporating Dental_008 polygon principal-axis apices and Dental_013 Endodontic (RCT)
    tie-breaking & post-RCT periapical subtype classification.
    """
    rct_set = _collect_rct_set(teeth_data, rct_fdis=rct_fdis)

    for lesion in lesions:
        raw_box = lesion.get("bbox") or lesion.get("bbox_xyxy")
        lx1, ly1, lx2, ly2 = [float(v) for v in raw_box[:4]]
        cx = (lx1 + lx2) / 2.0
        cy = (ly1 + ly2) / 2.0

        best_fdi: Optional[int] = None
        min_dist = float("inf")
        raw_geom_dist = float("inf")

        for tooth in teeth_data:
            fdi = tooth.get("fdi")
            if fdi is None:
                continue
            try:
                fdi_num = int(str(fdi).lstrip("#"))
            except (ValueError, TypeError):
                continue

            contour = tooth.get("polygon_px") or tooth.get("contour")
            bbox = tooth.get("bbox_xyxy") or tooth.get("bbox")

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

            quadrant = fdi_num // 10
            is_maxillary = quadrant in (1, 2, 5, 6)
            is_mandibular = quadrant in (3, 4, 7, 8)
            is_molar = (fdi_num % 10) in (6, 7, 8)

            axis_apex_xy: Optional[List[float]] = None
            if contour is not None and len(contour) >= 4:
                pts_2d = np.asarray(contour, dtype=np.float64).reshape(-1, 2).tolist()
                axis_info = compute_tooth_axis_and_apex(fdi_num, [tx1, ty1, tx2, ty2], pts_2d)
                axis_apex_xy = axis_info["apex_xy"]

            if is_maxillary:
                apex_y = axis_apex_xy[1] if axis_apex_xy else ty1
                if is_molar:
                    apex_candidates = [
                        tx1 + tooth_w * 0.20,
                        axis_apex_xy[0] if axis_apex_xy else tooth_cx,
                        tx2 - tooth_w * 0.20,
                    ]
                    apex_x = min(apex_candidates, key=lambda ax: abs(cx - ax))
                else:
                    apex_x = axis_apex_xy[0] if axis_apex_xy else tooth_cx
                if cy > (ty1 + tooth_h * 0.7):
                    continue
            elif is_mandibular:
                apex_y = axis_apex_xy[1] if axis_apex_xy else ty2
                if is_molar:
                    apex_candidates = [
                        tx1 + tooth_w * 0.15,
                        axis_apex_xy[0] if axis_apex_xy else tooth_cx,
                        tx2 - tooth_w * 0.15,
                    ]
                    apex_x = min(apex_candidates, key=lambda ax: abs(cx - ax))
                else:
                    apex_x = axis_apex_xy[0] if axis_apex_xy else tooth_cx
                if cy < (ty2 - tooth_h * 0.7):
                    continue
            else:
                apex_x = tooth_cx
                apex_y = (ty1 + ty2) / 2.0

            dx = abs(cx - apex_x)
            dy = abs(cy - apex_y)

            if dx > (tooth_w * 1.5):
                continue

            geom_dist = float(np.sqrt((dx * 1.3) ** 2 + (dy * 0.9) ** 2))
            eff_dist = geom_dist

            if fdi_num % 10 == 8:
                eff_dist += 40.0
            if (fdi_num % 10 == 6) and geom_dist < 100.0:
                eff_dist -= 40.0

            # Clinical tie-breaker: prioritize RCT-treated root apex when radiolucency lies between neighbors
            if fdi_num in rct_set and geom_dist < 150.0:
                eff_dist -= 25.0

            if eff_dist < min_dist:
                min_dist = eff_dist
                raw_geom_dist = geom_dist
                best_fdi = fdi_num

        if best_fdi is not None and min_dist < 180.0:
            is_rct = best_fdi in rct_set
            is_deciduous_or_super = ((best_fdi // 10) in (5, 6, 7, 8)) or ((best_fdi % 10) == 9)
            lesion["fdi"] = best_fdi
            lesion["apex_distance"] = round(float(max(0.0, raw_geom_dist)), 1)
            lesion["isolated_anatomical_suspicion"] = False
            lesion["rct_associated"] = is_rct
            lesion["lesion_subtype"] = (
                "post_rct_periapical_lesion" if is_rct else "untreated_periapical_lesion"
            )
            conf_val = float(lesion.get("confidence", 0.5))
            if is_deciduous_or_super:
                # Clinician policy: deciduous (#51-#85) and supernumerary (#19/#29/#39/#49)
                # periapical findings always require clinician review.
                lesion["clinical_status"] = "REVIEW_REQUIRED"
                lesion["rct_confidence_boosted"] = False
            elif (
                is_rct
                and lesion.get("clinical_status") == "REVIEW_REQUIRED"
                and conf_val >= 0.35
                and raw_geom_dist <= 95.0
            ):
                lesion["clinical_status"] = "DEFINITE"
                lesion["rct_confidence_boosted"] = True
            else:
                lesion.setdefault("rct_confidence_boosted", False)
        else:
            lesion["fdi"] = None
            lesion["rct_associated"] = None
            lesion["lesion_subtype"] = "unmatched_radiolucency"
            lesion["rct_confidence_boosted"] = False
            if anatomical_guard:
                lesion["isolated_anatomical_suspicion"] = True

    return lesions


def link_periapical_with_008_and_013(
    lesions: List[Dict[str, Any]],
    doc_or_teeth: Union[PanoramaHiTLDocument, Dict[Any, Any], Sequence[Any]],
    rct_fdis: Optional[Sequence[int]] = None,
    anatomical_guard: bool = True,
) -> List[Dict[str, Any]]:
    """
    Integrates Dental_012 periapical detections with Dental_008 FDI segmentation
    and Dental_013 Endodontic (RCT) classifications, updating `PanoramaHiTLDocument.lesions`
    while preserving clinician-edited HiTL entries.
    """
    teeth_dicts: List[Dict[str, Any]] = []
    if isinstance(doc_or_teeth, PanoramaHiTLDocument):
        for t in doc_or_teeth.teeth:
            teeth_dicts.append(
                {
                    "fdi": t.fdi,
                    "bbox": list(t.bbox_xyxy),
                    "contour": list(t.polygon_px) if t.polygon_px else None,
                    "restoration_tags": list(t.restoration_tags),
                }
            )
    elif isinstance(doc_or_teeth, dict):
        for k, val in doc_or_teeth.items():
            if isinstance(val, dict):
                entry = dict(val)
                entry.setdefault("fdi", k)
                teeth_dicts.append(entry)
    else:
        for item in doc_or_teeth:
            if isinstance(item, ToothAnnotation):
                teeth_dicts.append(
                    {
                        "fdi": item.fdi,
                        "bbox": list(item.bbox_xyxy),
                        "contour": list(item.polygon_px) if item.polygon_px else None,
                        "restoration_tags": list(item.restoration_tags),
                    }
                )
            elif isinstance(item, dict):
                teeth_dicts.append(dict(item))

    matched = match_lesions_to_teeth(
        lesions,
        teeth_dicts,
        anatomical_guard=anatomical_guard,
        rct_fdis=rct_fdis,
    )

    if isinstance(doc_or_teeth, PanoramaHiTLDocument):
        preserved = [
            l
            for l in doc_or_teeth.lesions
            if l.category != "periapical" or l.clinician_corrected
        ]
        corrected_periapical = [
            l for l in preserved if l.category == "periapical" and l.clinician_corrected
        ]

        def _box_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
            a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
            union = a1 + a2 - inter
            return inter / union if union > 0 else 0.0

        for item in matched:
            raw_box = [float(v) for v in (item.get("bbox") or item.get("bbox_xyxy"))[:4]]
            item_fdi = item.get("fdi")
            is_dup = any(
                (c.fdi == item_fdi and item_fdi is not None) or _box_iou(c.bbox_xyxy, raw_box) >= 0.3
                for c in corrected_periapical
            )
            if is_dup:
                continue
            preserved.append(
                LesionAnnotation(
                    category="periapical",
                    label=str(item.get("lesion_subtype") or "Periapical Lesion"),
                    bbox_xyxy=raw_box,
                    fdi=item_fdi,
                    confidence=float(item.get("confidence", 0.5)),
                    clinical_status=str(item.get("clinical_status", "DEFINITE")),
                    rct_associated=item.get("rct_associated"),
                    lesion_subtype=item.get("lesion_subtype"),
                    isolated_anatomical_suspicion=item.get("isolated_anatomical_suspicion"),
                    clinician_corrected=False,
                )
            )
        doc_or_teeth.lesions = preserved

    return matched


class PeriapicalPredictor:
    """
    Standalone Periapical Lesion Predictor (YOLO11s + Multi-Rooted Apex FDI Attribution
    + Dental_013 RCT Correlation + 2-Stage Apex Patch Inference Support).
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
        rct_fdis: Optional[Sequence[int]] = None,
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
                lesions,
                teeth_data,
                anatomical_guard=anatomical_guard,
                rct_fdis=rct_fdis,
            )

        return {"module_name": "Dental_012_periapical", "lesions": lesions}

    def predict_two_stage_apex_patches(
        self,
        image: np.ndarray,
        teeth_data: List[Dict[str, Any]],
        patch_size: int = 256,
        threshold_definite: float = 0.50,
        threshold_review: float = 0.25,
        rct_fdis: Optional[Sequence[int]] = None,
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

        rct_set = _collect_rct_set(teeth_data, rct_fdis=rct_fdis)
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
                    is_rct = fdi_num in rct_set
                    status = (
                        "DEFINITE"
                        if (conf >= threshold_definite or (is_rct and conf >= 0.35))
                        else "REVIEW_REQUIRED"
                    )
                    lesions.append(
                        {
                            "bbox": global_bbox,
                            "confidence": round(conf, 2),
                            "clinical_status": status,
                            "fdi": fdi_num,
                            "rct_associated": is_rct,
                            "lesion_subtype": (
                                "post_rct_periapical_lesion"
                                if is_rct
                                else "untreated_periapical_lesion"
                            ),
                            "isolated_anatomical_suspicion": False,
                        }
                    )

        return {"module_name": "Dental_012_periapical_2stage", "lesions": lesions}
