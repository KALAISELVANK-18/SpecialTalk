import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import io
from moviepy.editor import VideoFileClip
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import google.generativeai as genai
import os


app = FastAPI()

def custom_lstm(*args, **kwargs):
    kwargs.pop('time_major', None)  # Remove time_major if it exists
    return LSTM(*args, **kwargs)


# Load the trained model
custom_objects = {'LSTM': custom_lstm}
model = load_model('cropped_POC.keras', custom_objects=custom_objects)


# Initialize Label Encoder
label_encoder = LabelEncoder()


label_encoder.fit(['Minute', 'Morning', 'cheap', 'Month', 'flat', 'Blind', 'Monday', 'Week', 'happy', 'he', 'tight', 'Nice', 'loose',
                   'Mean', 'sad', 'Today', 'loud', 'she', 'Tomorrow', 'Friday', 'expensive', 'Ugly', 'it', 'Second', 'curved', 'I', 'we',
                   'poor', 'thick', 'Yesterday', 'you (plural)', 'quiet', 'Time', 'Tuesday', 'Sunday', 'Deaf', 'they', 'Hour', 'Year',
                   'thin', 'rich', 'Beautiful', 'Thursday', 'male', 'Saturday', 'you', 'Afternoon', 'Night', 'Wednesday', 'Evening', 'female'])


#label_encoder.fit(['Morning', 'Blind', 'happy', 'he', 'she', 'it', 'I', 'Deaf', 'combined', 'Night'])


API_KEY="AIzaSyCuWMeLjjC_Ta2WI2dQqIyOO4uniczU528"
genai.configure(api_key=API_KEY)
models = genai.GenerativeModel("gemini-1.5-flash")



def get_bounding_box(landmarks, image_width, image_height):
    x_coords = [landmark[0] * image_width for landmark in landmarks]
    y_coords = [landmark[1] * image_height for landmark in landmarks]
    x_min = int(min(x_coords))
    x_max = int(max(x_coords))
    y_min = int(min(y_coords))
    y_max = int(max(y_coords))
    padding = 20
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, image_width)
    y_max = min(y_max + padding, image_height)
    width = x_max - x_min
    height = y_max - y_min
    x_cor = [(land - x_min) / width for land in x_coords]
    y_cor = [(land - y_min) / height for land in y_coords]
    coordinates = [list(pair) for pair in zip(x_cor, y_cor)]
    return coordinates



def extract_hand_face_landmarks(frame, holistic):
    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = rgb_frame.shape
    # Process the frame with holistic
    results = holistic.process(rgb_frame)
    landmarks = []
    # Extract left hand landmarks if available
    if results.left_hand_landmarks:
        left_hand_coords = [[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]
        # landmarks.extend(left_hand_coords)

        # Get bounding box and crop left hand
        left_hand_bbox = get_bounding_box(left_hand_coords, image_width, image_height)
        landmarks.extend(left_hand_bbox)

    # Extract right hand landmarks if available
    if results.right_hand_landmarks:
        right_hand_coords = [[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]

        # Get bounding box and crop right hand
        right_hand_bbox = get_bounding_box(right_hand_coords, image_width, image_height)
        landmarks.extend(right_hand_bbox)

    if len(landmarks) == 42:
        # print(landmarks,"\n")
        return np.array(landmarks), frame
    else:
        return None, frame



def process_new_video(video_path, pose):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, frames = extract_hand_face_landmarks(frame, pose)

        if landmarks is not None:
            landmarks_list.append(landmarks)

    cap.release()

    # Convert to numpy array
    landmarks_array = np.array(landmarks_list)
    return landmarks_array



def predict_sign_for_each_frame(landmarks):
    predicted_signs = []
    for i, frame_landmarks in enumerate(landmarks):
        frame_landmarks = np.expand_dims(frame_landmarks, axis=0)
        prediction = model.predict(frame_landmarks)
        predicted_label = np.argmax(prediction, axis=-1)[0]
        accuracy = np.max(prediction, axis=-1)[0]
        predicted_sign = label_encoder.inverse_transform([predicted_label])[0]
        predicted_signs.append([predicted_sign,accuracy])
        print(f"Frame {i + 1}: Predicted Sign - {predicted_sign}")
        
    return predicted_signs



def segment_video_into_windows(predicted_signs, fps, window_duration=2):
    window_size = int(fps * window_duration)
    segmented_signs = []
    for i in range(0, len(predicted_signs), window_size):
        window = predicted_signs[i:i + window_size]
        windows=[]
        for j in range(len(window)):
            if(window[j][1]>=0.65):
                windows.append(window[j][0])
        print(len(windows))
        most_common_sign = Counter(windows).most_common(1)[0][0]
        if(len(windows)>=40):
            segmented_signs.append(most_common_sign)
    return segmented_signs



def remove_consecutive_duplicates(signs):
    filtered_signs = []
    previous_sign = None
    for sign in signs:
        if sign != previous_sign:
            filtered_signs.append(sign)
            previous_sign = sign
    return filtered_signs



@app.post("/predict_signs/")
async def predict_signs(file: UploadFile = File(...)):
    # Load pose detection model
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Process video and extract landmarks
    video_bytes = await file.read()

    # Save the received video file
    with open("received_video.mp4", "wb") as f:
        f.write(video_bytes)

    # Process the video using the saved file path
    landmarks = process_new_video("received_video.mp4", holistic)

    # Predict sign for each frame
    predicted_signs = predict_sign_for_each_frame(landmarks)

    # Get the video FPS
    video_clip = VideoFileClip("received_video.mp4")
    fps = video_clip.fps

    # Segment video and get the most common sign for each segment
    segmented_signs = segment_video_into_windows(predicted_signs, fps, window_duration=2)

    # Remove consecutive duplicates
    final_output = remove_consecutive_duplicates(segmented_signs)
    if(len(final_output)>1):
        request = "consider the below key words and return me a single sentence which is most probable one  ["+" ".join(final_output)+"] . you must give only one sentence"
        print(request)
        response = models.generate_content(request)
        print(response.text)
        return JSONResponse(content={"predicted_signs": response.text})
    if(len(final_output)==0):
        return JSONResponse(content={"predicted_signs": []})
    return JSONResponse(content={"predicted_signs": final_output[0]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)