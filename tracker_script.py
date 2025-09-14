import cv2
import mediapipe as mp
from ultralytics import YOLO
import os


# Load your 1080p video
input_video ="input/vid.mov" #paste the path of your video here
output_video = "output/vid.mov" #will be saved in the same directory as the 
USE_MEDIAPIPE = True
USE_YOLO = True


# Initialize MediaPipe Pose with custom configuration
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Use the most accurate model
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) if USE_MEDIAPIPE else None


# Load YOLOv8 Pose Model
yolo_model = YOLO("model/yolov8x.pt") if USE_YOLO else None

# Open input video
cap = cv2.VideoCapture(input_video)

# Get input video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Setup output video writer
out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📹 Processing {frame_count} frames from {input_video}...")

# Frame loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLOv8 Detection
    if yolo_model:
        yolo_results = yolo_model(
            frame,
            classes=[0,2,9],
            verbose=False
            )[0]
        
        ##frame = yolo_results[0].plot()
        
        for box in yolo_results.boxes:
            # Coordonnées
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confiance + classe
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            #Style
            color = (0, 255, 0)  # vert
            thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Dessin du rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Texte (label + score)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5),
                        font, 0.4, color, 1, cv2.LINE_AA)

    # 2. MediaPipe Pose Detection
    if pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            # Custom drawing specifications - all white
            landmark_spec = mp_draw.DrawingSpec(
                color=(255, 255, 255),  # White color
                thickness=2,
                circle_radius=2
            )
            connection_spec = mp_draw.DrawingSpec(
                color=(255, 255, 255),  # White color
                thickness=2
            )
            
            # Draw the pose landmarks
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec
            )
            
            # Add extra pose connections for more detail
            landmarks = results.pose_landmarks.landmark
            def draw_extra_connection(p1, p2):
                x1 = int(landmarks[p1].x * frame_width)
                y1 = int(landmarks[p1].y * frame_height)
                x2 = int(landmarks[p2].x * frame_width)
                y2 = int(landmarks[p2].y * frame_height)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Additional connections
            extra_connections = [
                (0, 1),    # Nose to inner eye
                (0, 4),    # Nose to outer eye
                (9, 10),   # Shoulders connection
                (11, 13),  # Upper arm connections
                (12, 14),
                (13, 15),  # Lower arm connections
                (14, 16),
                (23, 24),  # Hip connection
                (11, 23),  # Shoulder to hip
                (12, 24),
                (23, 25),  # Upper leg connections
                (24, 26),
                (25, 27),  # Lower leg connections
                (26, 28),
                (27, 31),  # Foot connections
                (28, 32)
            ]
            for connection in extra_connections:
                draw_extra_connection(*connection)

    # Save the processed frame
    out.write(frame)

# Cleanup
cap.release()
out.release()
print(f"✅ Saved processed video to: {output_video}")

