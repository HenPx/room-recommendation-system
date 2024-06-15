import streamlit as st
import cv2
import pickle
import time

def login_with_face(model_file, username_input):
    # video capture = 0(kamera laptop biasa), 1(obs klo gk salah) coba aja
    cap = cv2.VideoCapture(2)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    loaded_model = pickle.load(open(model_file, 'rb'))
    stframe = st.empty()
    
    login_success = False
    
    start_time = time.time()  # Start the timer

    while not login_success:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time > 15:
            st.error("Waktu habis. Login tidak valid.")
            return 1
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))
            
            face = face_roi_resized.flatten().reshape(1, -1)
            predicted_label = loaded_model.predict(face)[0]
            confidence = loaded_model.decision_function(face)[0]
            
            
            if predicted_label == username_input:
                login_success = True
                st.success(f"Login berhasil! Selamat datang, {username_input}.")
                return 0
        
        stframe.image(frame, channels="BGR")
                    
    cap.release()
    cv2.destroyAllWindows()