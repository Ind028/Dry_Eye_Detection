# faceMeshModule.py - UPDATED FOR YOUR CODE
import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=1, minDetectionConf=0.5, minTrackingConf=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        
        # Initialize with new API
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=maxFaces,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        results = self.detector.detect(mp_image)
        
        faces = []
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                h, w, _ = img.shape
                face = []
                for id, lm in enumerate(face_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append((cx, cy))  # Store only coordinates for your code
                
                # Your code expects face[22], face[23], etc.
                faces.append(face)
                
                # Draw face mesh if requested
                if draw and hasattr(mp.solutions, 'face_mesh'):
                    for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        if start_idx < len(face) and end_idx < len(face):
                            cv2.line(img, face[start_idx], face[end_idx], (0, 255, 0), 1)
        
        return img, faces
    
    def findDistance(self, p1, p2):
        """Calculate distance between two points - for your ratio calculation"""
        x1, y1 = p1
        x2, y2 = p2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance, []

# Keep the old class name for compatibility
faceMeshDetection = FaceMeshDetector