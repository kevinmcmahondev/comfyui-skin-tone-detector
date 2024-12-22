"""
Skin Tone Detector node for ComfyUI
Detects skin tone from input image and matches to emoji skin tone palette
"""

import mediapipe as mp
import numpy as np
import torch
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
            # Convert tensor to NumPy array
            image_np = image.cpu().numpy()

            # Remove batch dimension if present
            if image_np.ndim == 4:
                image_np = np.squeeze(image_np, axis=0)  # Removes batch dimension

            # Check if channels are first dimension and transpose if necessary
            # Expected shape after squeeze: (C, H, W) or (H, W, C)
            if image_np.shape[0] in [1, 3]:
                # Transpose from (C, H, W) to (H, W, C)
                image_np = np.transpose(image_np, (1, 2, 0))

            # Convert image data from [0, 1] to [0, 255] and ensure uint8 type
            image_np = np.clip(255 * image_np, 0, 255).astype(np.uint8)

            # Handle grayscale images by repeating the single channel to make it RGB
            if image_np.shape[2] == 1:
                image_np = np.repeat(image_np, 3, axis=2)

            # Convert to PIL Image
            image = Image.fromarray(image_np)

        elif isinstance(image, Image.Image):
            # Ensure image is in RGB mode
            image = image.convert('RGB')
        else:
            raise TypeError("Unsupported image type")

        # Proceed with the rest of your existing code
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        print(f"Image shape: {h}x{w}")

        results = self.face_detection.process(image_np)

        if results.detections:
            primary_detection = self._get_primary_detection(results.detections)
            face_region = self._extract_face_region(image_np, primary_detection)
            result = self._determine_skin_tone(face_region)
        else:
            print("No person detected in image")
            result = "NOT_DETECTED"

        return (result,)

    def _process_image(self, image):
        image_np = self._prepare_image(image)
        results = self.face_detection.process(image_np)
        
        if not results.detections:
            return None

        primary_detection = self._get_primary_detection(results.detections)
        face_region = self._extract_face_region(image_np, primary_detection)
        return self._determine_skin_tone(face_region)

    def _prepare_image(self, image):
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)
            if image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))
            image = Image.fromarray(np.clip(255. * image_np, 0, 255).astype(np.uint8))
        return np.array(image)
    def _get_primary_detection(self, detections):
        return max(detections, key=self._calculate_detection_score)

    def _calculate_detection_score(self, detection):
        bbox = detection.location_data.relative_bounding_box
        face_size = bbox.width * bbox.height
        face_center_x = bbox.xmin + (bbox.width/2)
        face_center_y = bbox.ymin + (bbox.height/2)
        center_dist = ((face_center_x - 0.5)**2 + (face_center_y - 0.5)**2)**0.5
        return face_size * (1 - center_dist)

    def _extract_face_region(self, image_np, detection):
        h, w = image_np.shape[:2]
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        return image_np[y:y+height, x:x+width]

    def _determine_skin_tone(self, face_region):
        face_lab = color.rgb2lab(face_region)
        avg_lab = np.mean(face_lab, axis=(0, 1))
        closest_tone = min(self.REFERENCE_TONES_LAB.items(),
                           key=lambda x: self.calculate_lab_distance(avg_lab, x[1]))[0]
        print(f"Detected skin tone: {closest_tone}")
        return closest_tone

    def _handle_no_detection(self):
        print("No person detected in image")
        return ("NOT_DETECTED",)
    
NODE_CLASS_MAPPINGS = {
    "SkinToneDetector": SkinToneDetector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinToneDetector": "Skin Tone Detector"
}
