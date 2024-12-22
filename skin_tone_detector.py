"""
Skin Tone Detector node for ComfyUI
Detects skin tone from input image and matches to emoji skin tone palette
"""

import mediapipe as mp
import numpy as np
from PIL import Image
from skimage import color


class SkinToneDetector:
    def __init__(self):
        # Emoji skin tone values in LAB color space
        self.SKIN_TONES = {
            "NOT_DETECTED": 0,
            "LIGHT": 1,
            "MEDIUM_LIGHT": 2, 
            "MEDIUM": 3,
            "MEDIUM_DARK": 4,
            "DARK": 5
        }
        
        # Reference LAB values for emoji skin tones
        self.REFERENCE_TONES_LAB = {
            "LIGHT": [85.0, 5.0, 15.0],        # 🏻
            "MEDIUM_LIGHT": [75.0, 10.0, 25.0], # 🏼
            "MEDIUM": [65.0, 15.0, 30.0],       # 🏽
            "MEDIUM_DARK": [45.0, 20.0, 35.0],  # 🏾
            "DARK": [30.0, 15.0, 30.0]          # 🏿
        }
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def calculate_lab_distance(self, lab1, lab2):
        return np.sqrt(np.sum((np.array(lab1) - np.array(lab2)) ** 2))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Skin Tone",)
    FUNCTION = "detect_skin_tone"
    CATEGORY = "image/analysis"
    OUTPUT_NODE = True

    def detect_skin_tone(self, image):
        result = None
        if isinstance(image, torch.Tensor):
            image = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
        
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        results = self.face_detection.process(image_np)
        
        if results.detections:
            primary_detection = None
            max_score = float('-inf')
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                face_size = bbox.width * bbox.height
                face_center_x = bbox.xmin + (bbox.width/2)
                face_center_y = bbox.ymin + (bbox.height/2)
                center_dist = ((face_center_x - 0.5)**2 + (face_center_y - 0.5)**2)**0.5
                score = face_size * (1 - center_dist)
                
                if score > max_score:
                    max_score = score
                    primary_detection = detection
            
            # Extract face region
            bbox = primary_detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            face_region = image_np[y:y+height, x:x+width]
            
            # Convert to LAB color space
            face_lab = color.rgb2lab(face_region)
            
            # Calculate average LAB values of face region
            avg_lab = np.mean(face_lab, axis=(0, 1))
            
            # Find closest matching skin tone
            min_distance = float('inf')
            closest_tone = None
            
            for tone, lab_values in self.REFERENCE_TONES_LAB.items():
                distance = self.calculate_lab_distance(avg_lab, lab_values)
                if distance < min_distance:
                    min_distance = distance
                    closest_tone = tone
            
            # Print the result
            print(f"Detected skin tone: {closest_tone}")
            result = closest_tone
        else:
            print("No person detected in image")
            result = "NOT_DETECTED"
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "SkinToneDetector": SkinToneDetector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinToneDetector": "Skin Tone Detector"
}
