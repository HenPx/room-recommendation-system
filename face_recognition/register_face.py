import streamlit as st
import cv2
import numpy as np

def capture_images(username):
    # video capture = 0(kamera laptop biasa), 1(obs klo gk salah) coba aja

    cap = cv2.VideoCapture(2)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_images = []
    num_images = 50
    count = 0

    stframe = st.empty()
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))  # Resize face_roi to a consistent size
            face_images.append(face_roi_resized)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Tampilkan jumlah sampel yang sudah diambil
            cv2.putText(frame, f'Sampel: {count}/{num_images}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        stframe.image(frame, channels="BGR")
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()

    face_images = np.array(face_images)
    return face_images