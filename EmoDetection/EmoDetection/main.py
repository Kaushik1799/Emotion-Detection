from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import openai
import sys

# Set UTF-8 encoding for stdout (useful for Windows systems to avoid UnicodeEncodeError)
sys.stdout.reconfigure(encoding='utf-8')

# Set your OpenAI API key directly (not recommended for production)

# Chatbot function to interact with OpenAI's GPT
def chatbot(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error with chatbot: {e}")
        return "Sorry, I couldn't process your request at the moment."

# Load face detection and emotion recognition models
face_classifier = cv2.CascadeClassifier(r'C:\Users\CyberianSK\Desktop\OpenCV\Projects Mine\EmoDetection\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\CyberianSK\Desktop\OpenCV\Projects Mine\EmoDetection\model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Counter for detecting prolonged sadness
sad_frame_count = 0

try:
    # Main loop for emotion detection and chatbot interaction
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  # Detect faces

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Draw a rectangle around the face
            roi_gray = gray[y:y + h, x:x + w]  # Region of interest for the face
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)  # Resize to (48, 48)

            if roi_gray.size != 0:
                roi = roi_gray.astype('float32') / 255.0  # Normalize the image data
                roi = img_to_array(roi)  # Convert to array
                roi = np.expand_dims(roi, axis=0)  # Expand dimensions to match model input shape

                # Predict the emotion
                prediction = classifier.predict(roi, verbose=0)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y - 10)  # Display label slightly above the rectangle

                # Display the emotion label on the screen
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # If the emotion is "Sad", increase the counter
                if label == 'Sad':
                    sad_frame_count += 1
                else:
                    sad_frame_count = 0  # Reset counter if another emotion is detected

                # If the user is sad for 10 consecutive frames, activate chatbot
                if sad_frame_count == 10:
                    chatbot_response = chatbot("I am feeling sad. Can you talk with me like a friend?")
                    print(f"Chatbot: {chatbot_response}")
                    sad_frame_count = 0  # Reset after chatbot interaction

            else:
                cv2.putText(frame, 'No Faces Detected', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Emotion Detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
