import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes = pd.read_csv('diabetes.csv')

# Memisahkan data dan label
X = diabetes.drop(columns='Outcome', axis=1)
Y = diabetes['Outcome']

# Standarisasi Data
scaler = StandardScaler()
scaler.fit(X)
standarized_data = scaler.transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(standarized_data, Y, test_size=0.2, stratify=Y, random_state=2)

# Create and train the SVM model
svc = svm.SVC(kernel='linear')
svc.fit(X_train, Y_train)

# Streamlit App
st.title('Deteksi Diabetes')

# Input untuk atribut
st.header('Masukkan Atribut Pasien')
nama = st.text_input('Nama Pasien', '')
pregnancies = st.number_input('Jumlah Kehamilan', min_value=0, max_value=17, value=0)
glucose = st.number_input('Kadar Glukosa', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Tekanan Darah', min_value=0, max_value=100, value=0)
skin_thickness = st.number_input('Ketebalan Kulit', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=0.0)
diabetes_pred = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=5.0, value=0.0)
age = st.number_input('Umur', min_value=0, max_value=100, value=0)

# Tambahkan input untuk atribut lainnya sesuai kebutuhan

input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pred, age])  # Isi atribut lainnya

# Button untuk prediksi
if st.button('Prediksi'):
    input_data_reshape = input_data.reshape(1, -1)
    std_data = scaler.transform(input_data_reshape)
    prediction = svc.predict(std_data)

    if prediction[0] == 0:
        st.write(f"{nama} tidak terkena diabetes.")
    else:
        st.write(f"{nama} terkena diabetes.")

# Menampilkan dataset jika diinginkan
if st.checkbox('Tampilkan Dataset'):
    st.write(diabetes)  # Menampilkan dataset
