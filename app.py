import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Judul aplikasi
st.title('🏠 House Price Prediction App')
st.write("""
Aplikasi ini memprediksi harga rumah berdasarkan fitur-fitur properti.
Model menggunakan algoritma Random Forest yang telah dilatih sebelumnya.
""")

# Load model dan scaler
@st.cache_resource
def load_model():
    model = joblib.load('final_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Daftar fitur yang dibutuhkan model
features = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', '1stFlrSF',
    '2ndFlrSF', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
    'MasVnrType_BrkCmn', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone',
    'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin',
    'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl',
    'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA',
    'BsmtExposure_Av', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_No',
    'BsmtFinType1_ALQ', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ',
    'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'Fireplaces_0', 'Fireplaces_1',
    'Fireplaces_2', 'Fireplaces_3', 'GarageType_Attchd', 'GarageType_Detchd',
    'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_None',
    'GarageFinish_Fin', 'GarageFinish_RFn', 'GarageFinish_Unf',
    'SaleType_WD', 'SaleType_New', 'SaleType_COD', 'SaleType_CWD',
    'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleCondition_Abnorml'
]

# Fungsi untuk preprocessing input
def preprocess_input(input_df):
    # Buat DataFrame dengan semua fitur, diisi 0
    processed = pd.DataFrame(0, index=[0], columns=features)
    
    # Isi nilai untuk fitur numerik
    for num_col in ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 
                   '1stFlrSF', '2ndFlrSF', 'GarageCars', 'GarageArea', 
                   'WoodDeckSF', 'OpenPorchSF']:
        processed[num_col] = input_df[num_col]
    
    # Set nilai 1 untuk fitur kategori yang dipilih
    for cat_col in input_df['categorical_features']:
        if cat_col in processed.columns:
            processed[cat_col] = 1
    
    return processed

# Input form
with st.form("input_form"):
    st.header("Masukkan Detail Properti")
    
    # Input numerik
    col1, col2 = st.columns(2)
    with col1:
        lot_frontage = st.number_input('Lebar Depan Properti (feet)', min_value=0, value=70)
        lot_area = st.number_input('Luas Tanah (sqft)', min_value=0, value=9600)
        mas_vnr_area = st.number_input('Luas Veneer Batu (sqft)', min_value=0, value=0)
        bsmt_fin_sf1 = st.number_input('Luas Ruang Bawah Tanah (sqft)', min_value=0, value=0)
    
    with col2:
        first_flr_sf = st.number_input('Luas Lantai 1 (sqft)', min_value=0, value=1464)
        second_flr_sf = st.number_input('Luas Lantai 2 (sqft)', min_value=0, value=0)
        garage_cars = st.number_input('Kapasitas Garasi (jumlah mobil)', min_value=0, max_value=4, value=2)
        garage_area = st.number_input('Luas Garasi (sqft)', min_value=0, value=480)
    
    wood_deck_sf = st.number_input('Luas Dek Kayu (sqft)', min_value=0, value=0)
    open_porch_sf = st.number_input('Luas Beranda Terbuka (sqft)', min_value=0, value=0)
    
    # Input kategori
    st.subheader("Fitur Kategori")
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        mas_vnr_type = st.selectbox('Jenis Veneer Batu', ['BrkFace', 'Stone', 'BrkCmn', 'None'])
        house_style = st.selectbox('Gaya Rumah', ['1Story', '2Story', '1.5Fin', '1.5Unf', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'])
        bsmt_qual = st.selectbox('Kualitas Ruang Bawah Tanah', ['Ex', 'Gd', 'TA', 'Fa'])
        bsmt_exposure = st.selectbox('Paparan Ruang Bawah Tanah', ['Av', 'Gd', 'Mn', 'No'])
    
    with cat_col2:
        bsmt_fin_type1 = st.selectbox('Tipe Penyelesaian Ruang Bawah Tanah', ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf'])
        fireplaces = st.selectbox('Jumlah Perapian', [0, 1, 2, 3])
        garage_type = st.selectbox('Tipe Garasi', ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'None'])
        garage_finish = st.selectbox('Penyelesaian Garasi', ['Fin', 'RFn', 'Unf'])
    
    sale_type = st.selectbox('Tipe Penjualan', ['WD', 'New', 'COD', 'CWD'])
    sale_condition = st.selectbox('Kondisi Penjualan', ['Normal', 'Partial', 'Abnorml'])
    
    submit_button = st.form_submit_button("Prediksi Harga")

# Ketika tombol submit diklik
if submit_button:
    # Buat dictionary dari input
    input_data = {
        'LotFrontage': lot_frontage,
        'LotArea': lot_area,
        'MasVnrArea': mas_vnr_area,
        'BsmtFinSF1': bsmt_fin_sf1,
        '1stFlrSF': first_flr_sf,
        '2ndFlrSF': second_flr_sf,
        'GarageCars': garage_cars,
        'GarageArea': garage_area,
        'WoodDeckSF': wood_deck_sf,
        'OpenPorchSF': open_porch_sf,
        'categorical_features': [
            f"MasVnrType_{mas_vnr_type}",
            f"HouseStyle_{house_style}",
            f"BsmtQual_{bsmt_qual}",
            f"BsmtExposure_{bsmt_exposure}",
            f"BsmtFinType1_{bsmt_fin_type1}",
            f"Fireplaces_{fireplaces}",
            f"GarageType_{garage_type}",
            f"GarageFinish_{garage_finish}",
            f"SaleType_{sale_type}",
            f"SaleCondition_{sale_condition}"
        ]
    }
    
    # Preprocess input
    input_df = pd.DataFrame(input_data, index=[0])
    processed_input = preprocess_input(input_df)
    
    # Scaling
    scaled_input = scaler.transform(processed_input)
    
    # Prediksi
    prediction = model.predict(scaled_input)
    
    # Tampilkan hasil
    st.success(f"Prediksi Harga Rumah: ${prediction[0]:,.2f}")
    
    # Tampilkan detail input
    with st.expander("Lihat Detail Input"):
        st.write("**Input Numerik:**")
        st.write(input_df.iloc[0, :10])
        st.write("**Input Kategori:**")
        st.write(input_data['categorical_features'])
