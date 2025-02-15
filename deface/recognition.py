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

def compare_embeddings(embedding, target_embeddings, threshold=0.70):
    """Compare person embeddings using average cosine similarity"""
    if embedding is None or not target_embeddings:
        return False, 0.0
    
    similarities = [1 - cosine(embedding, target_embedding) for target_embedding in target_embeddings]
    max_similarity = max(similarities)
    return max_similarity > threshold, max_similarity

def find_person_in_frame(frame, target_embeddings, threshold, person_detection_results, extractor, frame_face_dets):
    """Find target person and their face in frame"""
    results = person_detection_results
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
    
    # Track best match across all persons
    best_match = {
        'score': 0,
        'person_box': None,
        'face_box': None,
        'face_img': None,
        'person_img': None
    }
    
    for idx, person_embedding in enumerate(person_embeddings):
        is_match, score = compare_embeddings(person_embedding, target_embeddings, threshold)
        
        if is_match and score > best_match['score']:
            person_box = person_boxes[idx]
            x1, y1, x2, y2 = map(int, person_box)
            person_img = frame[y1:y2, x1:x2]
            person_area = (x2 - x1) * (y2 - y1)
            
            valid_faces = []
            for det in frame_face_dets:
                # Convert face detection to absolute coordinates
                face_x1, face_y1 = det[0], det[1]
                face_x2, face_y2 = det[2], det[3]
                face_area = (face_x2 - face_x1) * (face_y2 - face_y1)
                
                # Calculate intersection area
                intersection_x1 = max(x1, face_x1)
                intersection_y1 = max(y1, face_y1)
                intersection_x2 = min(x2, face_x2)
                intersection_y2 = min(y2, face_y2)
                
                if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    containment_ratio = intersection_area / face_area
                    
                    # Check face position relative to person top using face height
                    face_height = face_y2 - face_y1
                    face_top = face_y1  # Changed from using face center to face top
                    person_top = y1
                    
                    if (containment_ratio >= 0.9 and  # 90% containment threshold
                        abs(face_top - person_top) <= face_height):  # Compare face top to person top
                        valid_faces.append(det)
            
            if valid_faces:
                face_det = max(valid_faces, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
                fx1, fy1, fx2, fy2 = map(int, face_det[:4])
                face_box = (fx1, fy1, fx2 - fx1, fy2 - fy1)
                face_img = frame[fy1:fy2, fx1:fx2]
                
                best_match.update({
                    'score': score,
                    'person_box': face_box,
                    'face_img': face_img,
                    'person_img': person_img
                })
    
    if best_match['score'] > 0:
        return (best_match['person_box'], 
                best_match['face_img'], 
                best_match['score'], 
                best_match['person_img'])
    
    return None, None, 0, None    