# MediaPipe Pose Detection Demo with CSV Export
import cv2
import mediapipe as mp
import csv
from google.colab.patches import cv2_imshow

# Load video from Google Drive
video_path = "/content/drive/MyDrive/Sign_Language_Translation/night.MOV"
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Prepare CSV file to save coordinates
csv_filename = 'pose_coordinates.csv'
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Landmark_ID", "X", "Y"])  # Header

frame_count = 0

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(image_rgb)

    # If pose landmarks are detected
    if results.pose_landmarks:
        # Get image dimensions
        h, w, _ = frame.shape

        # Draw landmarks and export coordinates
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save landmark to CSV
            csv_writer.writerow([frame_count, id, cx, cy])

    # Show the processed frame
    cv2_imshow(frame)

    # Optional: press 'q' to quit early (won't work in Colab, mostly for local use)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"Pose coordinates saved to: {csv_filename}")
