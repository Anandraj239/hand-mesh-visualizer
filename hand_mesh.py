import cv2
import colorsys
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Standard skeletal connections for a single hand
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index Finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle Finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky Finger
]

def download_model():
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        print("Downloading MediaPipe hand landmarker model...")
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_path)
        print("Downloaded.")
    return model_path

def get_hsv_color(index, max_index=21):
    # Generates a vibrant color along the rainbow/HSV spectrum
    hue = index / float(max_index)
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    # OpenCV uses BGR
    return (int(b * 255), int(g * 255), int(r * 255))

def main():
    model_path = download_model()

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
        
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print("\nStarting Webcam...\nPress 'q' or 'ESC' to exit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            h, w, _ = image.shape
            
            # Extract keypoints
            all_hands_points = []
            for hand_landmarks in detection_result.hand_landmarks:
                pts = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pts.append((cx, cy))
                all_hands_points.append(pts)

            # 1. Intra-hand Connections
            # Draw standard skeleton instead of a massive web
            for pts in all_hands_points:
                for idx_1, idx_2 in HAND_CONNECTIONS:
                    if idx_1 < len(pts) and idx_2 < len(pts):
                        p1, p2 = pts[idx_1], pts[idx_2]
                        cv2.line(image, p1, p2, (255, 100, 0), 2) # Blue skeletal lines

            # 2. Inter-hand Connections
            # Instead of fully connecting all points, just connect corresponding points (Thumb to Thumb, Index to Index, etc.)
            if len(all_hands_points) == 2:
                pts1 = all_hands_points[0]
                pts2 = all_hands_points[1]
                
                for idx in range(min(len(pts1), len(pts2))):
                    # Each line connecting the hands gets a unique rainbow color
                    color = get_hsv_color(idx, 21)
                    cv2.line(image, pts1[idx], pts2[idx], color, 2)

            # 3. Draw Keypoints
            for pts in all_hands_points:
                for p in pts:
                    cv2.circle(image, p, 4, (0, 255, 0), -1)

        cv2.imshow('Connected Hands', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
