import cv2
import numpy as np
from scipy.spatial.distance import cosine
import glob
import os

def resize_for_reid(image, target_size=(256, 128)):
    """Resize image to ReID model's expected size while maintaining aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factors
    scale_w = target_w / w
    scale_h = target_h / h
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image maintaining aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create blank canvas of target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate padding
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def get_person_embeddings(image_directory, extractor):
    """Get embeddings for all target person images"""
    images = []
    for image_path in glob.glob(os.path.join(image_directory, '*')):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_for_reid(image)
            images.append(image)
    
    if not images:
        return []
    
    features = extractor(images)
    return list(features.cpu().numpy())

def compare_embeddings(embedding, target_embeddings, threshold=0.7):
    """Compare person embeddings using average cosine similarity"""
    if embedding is None or not target_embeddings:
        return False, 0.0
    
    similarities = [1 - cosine(embedding, target_embedding) for target_embedding in target_embeddings]
    max_similarity = max(similarities)
    return max_similarity > threshold, max_similarity

def find_person_in_frame(frame, target_embeddings, threshold, person_detector, extractor, frame_face_dets):
    """Find target person and their face in frame"""
    results = person_detector(frame, verbose=False)[0]
    person_boxes = []
    
    for result in results.boxes.data:
        if result[5] == 0 and result[4] >= 0.15:  # Class 0 is person
            person_boxes.append(result[:4].cpu().numpy())
    
    if not person_boxes:
        return None, None, 0, None
    
    person_crops = []
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box)
        person_crop = frame[y1:y2, x1:x2]
        person_crop = resize_for_reid(person_crop)
        person_crops.append(person_crop)
    
    person_embeddings = extractor(person_crops).cpu().numpy()
    
    for idx, person_embedding in enumerate(person_embeddings):
        is_match, score = compare_embeddings(person_embedding, target_embeddings)
        
        if is_match:
            person_box = person_boxes[idx]
            x1, y1, x2, y2 = map(int, person_box)
            person_img = frame[y1:y2, x1:x2]
            
            valid_faces = []
            for det in frame_face_dets:
                # Convert face detection to absolute coordinates
                face_x1, face_y1 = det[0], det[1]
                face_x2, face_y2 = det[2], det[3]
                
                # Check if face intersects with person box
                if (face_x1 < x2 and face_x2 > x1 and 
                    face_y1 < y2 and face_y2 > y1):
                    # Calculate face center y-coordinate relative to person box
                    face_center_y = (face_y1 + face_y2) / 2 - y1
                    person_height = y2 - y1
                    
                    # Check if face center is in top half of person
                    if face_center_y < (person_height / 3):
                        valid_faces.append(det)
            
            if valid_faces:
                face_det = max(valid_faces, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
                fx1, fy1, fx2, fy2 = map(int, face_det[:4])
                face_box = (fx1, fy1, fx2 - fx1, fy2 - fy1)
                face_img = frame[fy1:fy2, fx1:fx2]
                return face_box, face_img, score, person_img
    
    return None, None, 0, None