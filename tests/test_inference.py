import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dental_core.schemas.panorama_contract import (
    PanoramaHiTLDocument,
    PanoramaImageMeta,
    ToothAnnotation,
)
from src.inference import (
    PeriapicalPredictor,
    extract_apex_roi_patches,
    link_periapical_with_008_and_013,
    match_lesions_to_teeth,
    project_patch_bbox_to_global,
)


def test_predictor_initialization_without_weights():
    predictor = PeriapicalPredictor(model_path="nonexistent_model.onnx")
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    res = predictor.predict(dummy_img)
    assert res["module_name"] == "Dental_012_periapical"
    assert res["error"] == "Model not loaded"


def test_multi_rooted_molar_apex_attribution_and_guard():
    teeth_data = [
        {"fdi": 16, "bbox": [200, 150, 280, 280], "contour": None},  # Maxillary 1st molar
        {"fdi": 36, "bbox": [700, 400, 780, 530], "contour": None},  # Mandibular 1st molar
        {"fdi": 38, "bbox": [820, 410, 890, 525], "contour": None},  # Mandibular 3rd molar
    ]
    lesions = [
        {
            "bbox": [205, 130, 235, 165],  # Near mesial/buccal root apex of maxillary #16
            "confidence": 0.71,
            "clinical_status": "DEFINITE",
            "fdi": None,
            "isolated_anatomical_suspicion": False,
        },
        {
            "bbox": [750, 515, 785, 550],  # Near distal root apex of mandibular #36
            "confidence": 0.64,
            "clinical_status": "DEFINITE",
            "fdi": None,
            "isolated_anatomical_suspicion": False,
        },
        {
            "bbox": [40, 720, 80, 760],  # Isolated radiolucency (e.g. mental foramen) far from roots
            "confidence": 0.32,
            "clinical_status": "REVIEW_REQUIRED",
            "fdi": None,
            "isolated_anatomical_suspicion": False,
        },
    ]

    matched = match_lesions_to_teeth(lesions, teeth_data, anatomical_guard=True)
    assert matched[0]["fdi"] == 16
    assert matched[0]["isolated_anatomical_suspicion"] is False
    assert matched[1]["fdi"] == 36
    assert matched[1]["isolated_anatomical_suspicion"] is False
    assert matched[2]["fdi"] is None
    assert matched[2]["isolated_anatomical_suspicion"] is True


def test_two_stage_apex_patch_extraction_and_inverse_projection():
    pano = np.full((600, 1000, 3), 120, dtype=np.uint8)
    teeth_data = [
        {"fdi": 16, "bbox": [200, 150, 280, 280]},  # Maxillary: apex near y=150
        {"fdi": 36, "bbox": [700, 350, 780, 480]},  # Mandibular: apex near y=480
    ]
    patches = extract_apex_roi_patches(pano, teeth_data, patch_size=256)
    assert len(patches) == 2
    assert patches[0]["fdi"] == 16
    assert patches[0]["patch"].shape == (256, 256, 3)
    assert patches[1]["fdi"] == 36
    assert patches[1]["patch"].shape == (256, 256, 3)

    # Verify inverse projection round-trip accuracy (< 1 px)
    crop_coords = (100, 200, 300, 400)  # 200x200 region
    global_bbox = project_patch_bbox_to_global([64.0, 64.0, 192.0, 192.0], crop_coords, patch_size=256)
    assert global_bbox == [150, 250, 250, 350]


def test_rct_tie_breaking_and_hitl_document_linking():
    # Two adjacent mandibular premolars (#44 at cx=220, #45 at cx=270), lesion centered at cx=246 (slightly closer to #44)
    doc = PanoramaHiTLDocument(
        image_meta=PanoramaImageMeta(filename="rct_periapical.png", width=1000, height=600),
        teeth=[
            ToothAnnotation(fdi=44, bbox_xyxy=[200.0, 350.0, 240.0, 480.0], restoration_tags=[]),
            ToothAnnotation(
                fdi=45,
                bbox_xyxy=[250.0, 350.0, 290.0, 480.0],
                restoration_tags=["Endodontic (RCT)", "Crown/Bridge"],
            ),
            ToothAnnotation(fdi=34, bbox_xyxy=[650.0, 350.0, 690.0, 480.0], restoration_tags=[]),
        ],
    )

    lesions = [
        {
            "bbox": [231.0, 470.0, 261.0, 500.0],  # cx=246, between #44(220) and #45(270); #45 has RCT
            "confidence": 0.41,
            "clinical_status": "REVIEW_REQUIRED",
        },
        {
            "bbox": [655.0, 472.0, 685.0, 505.0],  # Untreated #34 apex
            "confidence": 0.68,
            "clinical_status": "DEFINITE",
        },
    ]

    matched = link_periapical_with_008_and_013(lesions, doc)
    assert matched[0]["fdi"] == 45
    assert matched[0]["rct_associated"] is True
    assert matched[0]["lesion_subtype"] == "post_rct_periapical_lesion"
    assert matched[0]["clinical_status"] == "DEFINITE"
    assert matched[0]["rct_confidence_boosted"] is True

    assert matched[1]["fdi"] == 34
    assert matched[1]["rct_associated"] is False
    assert matched[1]["lesion_subtype"] == "untreated_periapical_lesion"

    assert len(doc.lesions) == 2
    assert doc.lesions[0].category == "periapical"
    assert doc.lesions[0].fdi == 45
    assert doc.lesions[0].rct_associated is True
