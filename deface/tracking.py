import cv2
import numpy as np

def init_face_tracker(frame, bbox):
    """Initialize CSRT tracker with a face detection"""
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    
    # Use the larger dimension to create a square
    side_length = max(w, h) * 2.5  # Scale factor of 2.5
    
    # Calculate new coordinates to maintain center
    new_x = center_x - side_length / 2
    new_y = center_y - side_length / 2
    
    # Ensure the new bounding box stays within the frame
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    side_length = min(side_length, frame.shape[1] - new_x)  # Constrain by width
    side_length = min(side_length, frame.shape[0] - new_y)  # Constrain by height
    
    new_bbox = (int(new_x), int(new_y), int(side_length), int(side_length))
    
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, new_bbox)
    return tracker, new_bbox

def update_face_tracker(frame, tracker, prev_bbox):
    """Update tracker and validate tracking results"""
    success, bbox = tracker.update(frame)
    if success:
        bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        new_w, new_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        prev_w, prev_h = prev_bbox[2] - prev_bbox[0], prev_bbox[3] - prev_bbox[1]

        # Calculate distance moved
        distance_moved = np.sqrt(
            (bbox[0] - prev_bbox[0])**2 + 
            (bbox[1] - prev_bbox[1])**2
        )
        
        # Calculate maximum allowed movement
        max_movement = prev_bbox[2] * 0.6
        
        if distance_moved > max_movement:
            return None  # Consider tracking lost if moved too far
        
        max_change = 0
        
        if prev_w > 0 and prev_h > 0:
            w_change = abs(new_w - prev_w) / prev_w
            h_change = abs(new_h - prev_h) / prev_h
            
            if w_change > max_change or h_change > max_change:
                center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                bbox = (
                    center_x - prev_w / 2,
                    center_y - prev_h / 2,
                    center_x + prev_w / 2,
                    center_y + prev_h / 2
                )

        return bbox
    return None

def recover_tracking(frame, prev_bbox, dets, debugging=False):
    """Attempt to recover lost tracking using face detections"""
    prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
    prev_center_x = (prev_x1 + prev_x2) / 2
    prev_center_y = (prev_y1 + prev_y2) / 2
    prev_width = prev_x2 - prev_x1
    prev_height = prev_y2 - prev_y1
    
    # Define search region
    search_width = prev_width * 2
    search_height = prev_height * 1
    max_vertical_shift = prev_height * 0.2
    
    search_x1 = prev_center_x - search_width/2
    search_y1 = prev_center_y - search_height/2
    search_x2 = search_x1 + search_width
    search_y2 = search_y1 + search_height
    
    best_det = None
    min_distance = float('inf')
    
    for det in dets:
        det_center_x = (det[0] + det[2]) / 2
        det_center_y = (det[1] + det[3]) / 2
        vertical_shift = abs(det_center_y - prev_center_y)
        
        if vertical_shift > max_vertical_shift:
            continue
        
        if (search_x1 <= det_center_x <= search_x2 and 
            search_y1 <= det_center_y <= search_y2):
            
            distance = np.sqrt(
                (det_center_x - prev_center_x)**2 + 
                (det_center_y - prev_center_y)**2
            )
            
            if distance < min_distance:
                min_distance = distance
                best_det = (
                    det[0], det[1],
                    det[2] - det[0],
                    det[3] - det[1]
                )
    
    return best_det