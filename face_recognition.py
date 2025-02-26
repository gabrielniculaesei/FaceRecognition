import cv2
import face_recognition
import numpy as np
from sklearn.datasets import fetch_lfw_people
import time

print("Hi and welcome, please select your desired threshold.")
print("(Keep in mind, a higher threshold results in a higher FNMR (False Non-Match Rate),")
print("and vice versa for FAR (False Acceptance Rate).)")
threshold = int(input("Enter matching threshold as % (e.g., 50): "))

print("Initializing face recognition system...")

print("Loading the dataset...")
ds = fetch_lfw_people(min_faces_per_person=50, resize=0.5)

# Extract images, targets, and labels
ds_images = ds.images
ds_targets = ds.target
ds_names = ds.target_names

known_encodings = []
known_names = []

print("Processing reference faces...")
n = enumerate(ds_images)
for i, img in n:
    if i % 100 == 0:
        print(f"Processing image {i}/{len(ds_images)}...")
        
    # Convert grayscale image to RGB (3 channels)
    img_rgb = np.stack((img,)*3, axis=-1)
    
    # Convert image from float32 to uint8
    img_uint8 = (img_rgb * 255).astype(np.uint8)
    
    try:
        # Find face locations first to make sure we have a face
        face_locations = face_recognition.face_locations(img_uint8)
        if len(face_locations) > 0:
            encoding = face_recognition.face_encodings(img_uint8, face_locations)[0]
            known_encodings.append(encoding)
            known_names.append(ds_names[ds_targets[i]])
    except Exception as e:
        # If processing fails, skip the image
        continue  

print(f"Loaded {len(known_encodings)} faces from LFW dataset.")

# List available cameras
def list_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Camera {i} is available")
            cap.release()
    return available_cameras

# Camera initialization with camera selection
def initialize_camera(camera_index=0, max_attempts=3):
    for attempt in range(max_attempts):
        print(f"Attempting to open camera {camera_index} (attempt {attempt+1}/{max_attempts})...")
        video_capture = cv2.VideoCapture(camera_index)
        
        # Check if camera opened successfully
        if video_capture.isOpened():
            # Read a test frame to ensure camera is working
            ret, frame = video_capture.read()
            if ret:
                print(f"Camera {camera_index} initialized successfully!")
                return video_capture
        
        # Release and try again
        video_capture.release()
        print("Camera initialization failed. Retrying in 2 seconds...")
        time.sleep(2)
    
    print(f"Failed to initialize camera {camera_index} after multiple attempts.")
    return None

# Start by listing available cameras
print("Checking available cameras...")
available_cameras = list_cameras()
if not available_cameras:
    print("No cameras found!")
    exit(1)

# Since I use an apple ecosystem, i need to implement this condition to prevent
# the program from selecting my iPhone camera first

camera_index = 0 
if len(available_cameras) > 1:
    camera_index = 1
    print(f"Multiple cameras found. Will try camera index {camera_index}")
else:
    print(f"Only one camera found. Will use camera index {camera_index}")

video_capture = initialize_camera(camera_index)

# If the first camera choice failed, try another one
if video_capture is None and len(available_cameras) > 1:
    print("First camera choice failed. Trying alternative camera...")
    alt_camera_index = 0 if camera_index != 0 else 1
    video_capture = initialize_camera(alt_camera_index)
    if video_capture is not None:
        camera_index = alt_camera_index

if video_capture is None:
    print("Could not open any camera...")
    exit(1)

print(f"Successfully connected to camera {camera_index}")
print("Starting face recognition. Press 'q' to quit.")

frame_count = 0
try:
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            print("Failed to grab frame. Retrying...")
            # Try to reinitialize camera if frame grab fails
            video_capture.release()
            video_capture = initialize_camera(camera_index)
            if video_capture is None:
                print("Could not reconnect to camera.")
                break
            continue
            
        frame_count += 1
        if frame_count % 10 == 0:  # Log every 10 frames to show the program is still running
            print(f"Processing frame {frame_count}")
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) > 0:
            print(f"Detected {len(face_locations)} faces in frame")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Skip processing if no known encodings 
            if len(known_encodings) == 0:
                cv2.putText(frame, "No reference faces loaded", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                  
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = known_names[best_match_index]
                    
                    #  match percentage
                    match_percentage = (1 - face_distances[best_match_index]) * 100
                    
                    # If match percentage is below the threshold, label as "Unknown" and use a red rectangle
                    if match_percentage < threshold:
                        name = "Unknown"
                        rect_color = (0, 0, 255)  
                    else:
                        name = known_names[best_match_index]
                        rect_color = (0, 255, 0)  
                    
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), rect_color, 2)
                    label = f"{name} ({match_percentage:.1f}%)"
                    cv2.putText(frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)
        
        
        cv2.putText(frame, f"Faces in database: {len(known_encodings)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
       
        cv2.imshow('Live Face Recognition', frame)
        
        # Break loop on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit. Closing...")
            break

except Exception as e:
    print(f"An error occurred: {e}")
    
finally:
    # Here I make sure I release the webcam and close windows
    print("Cleaning up resources...")
    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()
    print("Application closed.")
