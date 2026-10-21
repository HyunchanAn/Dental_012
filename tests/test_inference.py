import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import PeriapicalPredictor, match_lesions_to_teeth


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
