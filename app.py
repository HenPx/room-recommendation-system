import streamlit as st
import pymysql
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from database.connection import create_connection
from database.table_create import create_table
from user.login import login_user
from user.register import add_user, hash_password
import os
from face_recognition.login_face import login_with_face
from face_recognition.register_face import capture_images
from face_recognition.train_face import train_and_save_model
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("""
<style>
    .highlight {
        font-weight :bold;
        text-decoration: underline;
    }
    [data-testid=stSidebar] {
        background-color: #D6E4FF;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1.5rem;
    }
    .menu-title {
        font-size: 30px;
        font-weight: bold;
        margin : 0;
    }
    .header-login{
        font-size: 20px;
    }
    
    .st-emotion-cache-16idsys p {
        word-break: break-word;
        margin-bottom: 0px;
        font-size: 18px;
    }
    .menu-item {
        font-size: 18px;
    }
    .st-emotion-cache-1inwz65{
        visibility: hidden;
    }
    .st-emotion-cache-10y5sf6{
        visibility: hidden;

    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
        padding-right: 10px;
        padding-left: 4px;
        padding-bottom: 3px;
        margin: 4px;
    }
    .p{
        font-size: 1.5 rem;
    }
            
</style>
""", unsafe_allow_html=True)
# Koneksi ke database MySQL
create_connection()

# Membuat tabel pengguna jika belum ada
create_table()

# Streamlit UI
st.title("Room Recommendation System")

menu = ["Home", "Login and Check", "SignUp"]
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #555;
        }
    </style>
    <div class="footer">
        © 2024 SIMRATEK
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-wrapper">', unsafe_allow_html=True)
    st.image("asset/logo2.png", width=300)
    st.markdown('<div class="menu-title">Menu</div>', unsafe_allow_html=True)
    choice = st.radio("", menu, index=0, key="menu_radio")
    st.markdown('</div>', unsafe_allow_html=True)

if choice == "Home":
    st.subheader("Welcome to the Room Recommendation System")
    # Load dataset
    name = ['Keamanan', 'Luas', 'INDUS', 'CHAS', 'NOX', 'Jumlah Blok Ruangan', 'AGE', 'Jarak Dari Pusat Kota', 'Jarak Dari Jalan Raya', 'Properti', 'PTRATIO', 'B', 'Kepadatan', 'Rerata Harga']
    filepath = 'asset/housing.csv'
    df = pd.read_csv(filepath, delim_whitespace=True, names=name)

    # Add some space between sections
    st.write("\n")
    
    # Line chart example
    st.write("### Grafik Garis Distribusi ruangan:")
    chart_data = pd.DataFrame(
        df[['Rerata Harga', 'Jumlah Blok Ruangan', 'Kepadatan']].head(20)
    )
    st.line_chart(chart_data)
    
    # Distribution plots
    st.write("### Grafik Distribusi Ruangan:")
    features = ['Rerata Harga', 'Jumlah Blok Ruangan', 'Kepadatan']
    for feature in features:
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        ax.set_title(f'Distribusi {feature}', fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frekuensi', fontsize=12)
        st.pyplot(fig)

elif choice == "Login and Check":
    st.markdown('<div class="header-login">Silahkan masukan username dan password untuk mendapatkan rekomendasi sistem.</div>', unsafe_allow_html=True)
    username = st.text_input("**Username**")
    password = st.text_input("**Password**", type='password')
    name = ['Keamanan', 'Luas', 'INDUS', 'CHAS', 'NOX', 'Jumlah Blok Ruangan', 'AGE', 'Jarak Dari Pusat Kota', 'Jarak Dari Jalan Raya', 'Properti', 'PTRATIO', 'B', 'Kepadatan', 'Harga*($1000)']
    filepath = 'asset/housing.csv'
    df = pd.read_csv(filepath, delim_whitespace=True, names=name)
    df.head()

    data = df  

    features = data.drop(columns=['CHAS', 'NOX', 'AGE', 'INDUS', 'PTRATIO', 'B'])

    # 3. Normalisasi fitur
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    # Fungsi untuk menerima input dari pengguna di Streamlit
    def get_user_input():
        user_input = {}
        for col in features.columns:
            min_val = float(features[col].min())
            max_val = float(features[col].max())
            user_input[col] = st.slider(f"Nilai bobot {col} yang diinginkan", min_val, max_val)
        return user_input

    # 4. Menerima input dari pengguna melalui Streamlit
    user_input = get_user_input()
    if st.button("Login"):
        hashed_password = hash_password(password)
        result = login_user(username, hashed_password)

        model_file = f'user_model/{username}_face_recognition_model.pkl'

        if result and os.path.exists(model_file):
            st.write("Mendeteksi wajah...")
            hasil_scan = login_with_face(model_file, username)
            if hasil_scan != 1:
                # 5. Normalisasi input pengguna
                user_input_df = pd.DataFrame(user_input, index=[0])
                normalized_user_input = scaler.transform(user_input_df)

                # Debugging: Periksa input pengguna setelah normalisasi
                st.write("Normalized User Input:")
                st.write(normalized_user_input)

                # 6. Menghitung kemiripan menggunakan cosine similarity
                similarity_scores = cosine_similarity(normalized_user_input, normalized_features)

                # Debugging: Periksa skor kemiripan
                st.write("Similarity Scores:")
                st.write(similarity_scores)

                # 7. Membuat fungsi rekomendasi
                def recommend_houses(similarity_scores, top_n=5):
                    similarity_scores = list(enumerate(similarity_scores[0]))
                    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
                    similar_houses = [i[0] for i in similarity_scores[:top_n]]
                    recommended_data = data.iloc[similar_houses].copy()  # Make a copy to avoid modifying original data
                    # Drop columns that were excluded from features
                    recommended_data.drop(columns=['Keamanan', 'Luas', 'INDUS', 'CHAS', 'NOX', 'Jumlah Blok Ruangan', 'AGE', 'Jarak Dari Pusat Kota', 'Jarak Dari Jalan Raya', 'Properti', 'PTRATIO', 'B', 'Kepadatan'], inplace=True)
                    return recommended_data

                # Display recommendations
                recommendations = recommend_houses(similarity_scores)
                st.write("Recommendations:")
                st.write(recommendations)
                st.button("Restart")
            else : 
                st.button("Restart")
                st.markdown('<p class="highlight">Dont have account? SignUp first.</p>', unsafe_allow_html=True)
        else:
            st.button("Restart")
            st.error("Incorrect Username or Password")
            st.markdown('<p class="highlight">Dont have account? SignUp first.</p>', unsafe_allow_html=True)
            
elif choice == "SignUp":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    k=0
    new_password = st.text_input("Password", type='password')
    if st.button("Signup"):
        hashed_new_password = hash_password(new_password)
        try:
            add_user(new_user, hashed_new_password)
            k=1
        except pymysql.err.IntegrityError:
            st.warning("Username already exists")
    
    if k ==1:
        st.write("Mengambil gambar wajah...")
        face_images = capture_images(new_user)
        st.write("Face recognition...")
        model_file = train_and_save_model(new_user, face_images)
        st.success("Buat akun berhasil! Data anda telah disimpan.")
        st.button("Selesai")