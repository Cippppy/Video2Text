"""
## Summary

"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Standard library imports
import os

# Third-party imports
import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import KMeans
from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def extract_frame_features(frame):
    """
    Extracts a color histogram feature vector from a given video frame.

    This function converts the input frame from BGR to HSV color space 
    for better representation of color information. It computes a 3D 
    histogram across the hue, saturation, and value channels. The histogram 
    is then normalized and flattened into a 1D feature vector suitable for 
    use in clustering algorithms like K-means.

    Parameters:
    -----------
    frame : numpy.ndarray
        A single video frame in BGR format, typically obtained from 
        OpenCV's `cv2.VideoCapture`.

    Returns:
    --------
    numpy.ndarray
        A 1D feature vector representing the normalized color histogram 
        of the input frame, with 8 bins per channel (512 total features).
    """
    # Convert to HSV color space for better clustering
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    # Normalize the histogram and flatten it into a feature vector
    return cv2.normalize(hist, hist).flatten()

def extract_keyframes_with_kmeans(model, video_path, output_folder, num_clusters=5, threshold=0.5, device="cuda:0"):
    """
    Extracts keyframes from a video using YOLO object detection and K-means clustering.

    Parameters:
    -----------
    model : torch.nn.Module
        The pre-trained YOLO model used for object detection.
    
    video_path : str
        Path to the input video file.

    output_folder : str
        Path to the folder where the extracted keyframes will be saved.

    num_clusters : int, optional (default=5)
        The number of clusters for K-means to group similar frames. 

    threshold : float, optional (default=0.5)
        Confidence threshold for YOLO object detection. 

    Returns:
    --------
    list of str
        A list of file paths for the saved keyframes.
    """
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    frames = []
    frame_id = 0

    # Step 1: Extract features from frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO on the frame
        results = model.track(frame, persist=True, device=device, verbose=False, tracker="bytetrack.yaml")
        if len(results[0].boxes.xywh.cpu()) > 0:
            feature_vector = extract_frame_features(frame)
            frame_features.append(feature_vector)
            frames.append((frame_id, frame))  # Store frame for saving later

        frame_id += 1

    cap.release()

    # Step 2: Apply K-means clustering on frame features
    frame_features = np.array(frame_features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(frame_features)
    labels = kmeans.labels_

    # Step 3: Save keyframes and return their paths
    last_label = -1
    keyframe_paths = []

    for i, (frame_id, frame) in enumerate(frames):
        if labels[i] != last_label:
            frame_name = f"keyframe_{frame_id}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            print(f"Saved {frame_name}")
            keyframe_paths.append(frame_path)  # Store the saved path
            last_label = labels[i]

    return keyframe_paths  # Return list of keyframe paths

def generate_caption(frame, model="Salesforce/blip-image-captioning-base", device=0):
    """Generate"""
    captioner = pipeline("image-to-text", model=model, device=device, max_new_tokens=100)
    caption = captioner(frame)
    return caption[0]['generated_text']

def generate_summary(model, tokenizer, prompt):
    """Generate a text summary based on the given prompt."""
    inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Example usage
    det_model = YOLO('yolov8n.pt')  # Load an official Detect model
    # Load GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    video_path = 'videos/Saint.mp4'
    keyframes = extract_keyframes_with_kmeans(det_model, video_path, 'frames/', num_clusters=5)
    
    captions = []
    for keyframe in keyframes:
        captions.append(generate_caption(keyframe))
    print(captions)
    summary = generate_summary(model, tokenizer, str(captions))
    print(summary)