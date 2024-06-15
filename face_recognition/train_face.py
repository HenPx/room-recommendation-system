import streamlit as st
import cv2
import numpy as np
from sklearn.svm import SVC
import pickle

# Fungsi untuk membuat data wajah dummy
def generate_dummy_faces(num_images, image_shape):
    dummy_faces = np.random.randint(0, 256, (num_images, *image_shape), dtype=np.uint8)
    return dummy_faces

# Fungsi untuk training SVM dan menyimpan model
def train_and_save_model(username, face_images):
    dummy_faces = generate_dummy_faces(len(face_images), face_images[0].shape)
    
    labels = np.array([username] * len(face_images) + ['dummy'] * len(dummy_faces))
    face_images_flatten = np.concatenate([face_images, dummy_faces]).reshape((len(face_images) + len(dummy_faces), -1))
    
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(face_images_flatten, labels)
    
    model_filename = f'user_model/{username}_face_recognition_model.pkl'
    pickle.dump(svm_model, open(model_filename, 'wb'))
    
    return model_filename