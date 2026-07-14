import sys
import os
import cv2
import numpy as np

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(BASE_DIR))

# Dummy placeholder for Dental_Panoramic_Reader structure
from Dental_Panoramic_Reader.modules.periapical_predictor import PeriapicalPredictorWrapper

def test_matching():
    # 1. Create a dummy PeriapicalPredictorWrapper
    print("Initializing PeriapicalPredictorWrapper...")
    # Using a dummy model path since we don't have weights yet
    predictor = PeriapicalPredictorWrapper(model_path="dummy.pt")
    
    # Let's override the predict method temporarily for testing since we have no model
    def dummy_predict(image, **kwargs):
        # Fake lesions detected by YOLO
        lesions = [
            {"bbox": [100, 400, 150, 450], "confidence": 0.85, "fdi": None}, # Lesion 1 (Lower left, around FDI 36)
            {"bbox": [800, 100, 850, 150], "confidence": 0.92, "fdi": None}  # Lesion 2 (Upper right, around FDI 16)
        ]
        
        teeth_data = kwargs.get("teeth_data", None)
        if teeth_data:
            predictor._match_fdi(lesions, teeth_data)
            
        return {"module_name": "Dental_012_periapical", "lesions": lesions}

    predictor.predict = dummy_predict
    
    # 2. Create Dummy teeth_data (from Dental_008)
    print("Creating dummy teeth masks from Dental_008...")
    # Tooth 36 contour (Lower left)
    contour_36 = np.array([[[90, 350]], [[160, 350]], [[160, 395]], [[90, 395]]], dtype=np.int32)
    # Tooth 16 contour (Upper right)
    contour_16 = np.array([[[790, 160]], [[860, 160]], [[860, 220]], [[790, 220]]], dtype=np.int32)
    
    teeth_data = [
        {"fdi": 36, "confidence": 0.99, "bbox": [90, 350, 160, 395], "contour": contour_36},
        {"fdi": 16, "confidence": 0.95, "bbox": [790, 160, 860, 220], "contour": contour_16}
    ]
    
    # 3. Run prediction with FDI matching
    print("Running E2E FDI matching logic...")
    dummy_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = predictor.predict(dummy_image, teeth_data=teeth_data)
    
    # 4. Verify results
    print("\n--- Matching Results ---")
    for idx, lesion in enumerate(result['lesions']):
        matched_fdi = lesion['fdi']
        print(f"Lesion {idx+1} Bbox: {lesion['bbox']}")
        print(f" -> Matched FDI: {matched_fdi} (Confidence: {lesion['confidence']})")
        
        if idx == 0 and matched_fdi == 36:
            print(" -> [PASS] Correctly matched to FDI 36!")
        elif idx == 1 and matched_fdi == 16:
            print(" -> [PASS] Correctly matched to FDI 16!")
        else:
            print(" -> [FAIL] Matching logic returned unexpected result.")
            
if __name__ == "__main__":
    test_matching()
