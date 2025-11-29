import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from statsmodels.tools.sm_exceptions import HessianInversionWarning

# Mengabaikan peringatan tertentu dari statsmodels saat loading model
import warnings
warnings.filterwarnings("ignore", category=HessianInversionWarning)

# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Forecasting Penjualan Instax",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PATH FILE (Sesuaikan jika diperlukan) ---
DATA_FILE = "instax_sales_transaction_data.csv"
SARIMA_MODEL_FILE = "model_sarima.pkl"
HOLTWINTERS_MODEL_FILE = "model_holtwinters.pkl"

# --- FUNGSI LOAD DATA & MODEL ---

@st.cache_data
def load_data(file_path):
    """Memuat dan membersihkan data penjualan."""
    try:
        df = pd.read_csv(file_path)
        
        # Mengubah kolom Tanggal menjadi datetime
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        
        # Mengagregasi data penjualan harian (Quantity)
        daily_sales = df.groupby('Tanggal')['Qty'].sum().reset_index()
        daily_sales = daily_sales.set_index('Tanggal')['Qty'].resample('D').sum() # Mengisi hari yang kosong dengan 0
        
        # Mengagregasi data penjualan bulanan
        monthly_sales = daily_sales.resample('M').sum()
        monthly_sales.index = monthly_sales.index.to_period('M')
        
        return df, daily_sales, monthly_sales
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan: {file_path}")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
        return None, None, None

@st.cache_resource
def load_model(file_path):
    """Memuat model yang sudah dilatih (pickle)."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan: {file_path}")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model {file_path}: {e}")
        return None

# Muat data dan model
df_raw, df_daily, df_monthly = load_data(DATA_FILE)
sarima_model = load_model(SARIMA_MODEL_FILE)
hw_model = load_model(HOLTWINTERS_MODEL_FILE)

# --- FUNGSI HALAMAN (VIEWS) ---

def home_page():
    """Halaman utama: Penjelasan Aplikasi."""
    st.title("Aplikasi Peramalan Penjualan Instax")
    st.markdown("---")

    st.header("Selamat Datang!")
    st.markdown(
        """
        Aplikasi ini dikembangkan untuk melakukan peramalan (forecasting) terhadap 
        kuantitas penjualan produk Instax berdasarkan data transaksi historis. 
        Tujuan utamanya adalah membantu manajemen dalam pengambilan keputusan terkait 
        inventaris, strategi pemasaran, dan perencanaan produksi di masa mendatang.
        """
    )

    st.subheader("Data yang Digunakan")
    st.markdown(f"Data yang digunakan bersumber dari file `{DATA_FILE}`, yang berisi data transaksi harian dari Mei 2022 hingga April 2025.")
    
    st.subheader("Metodologi Peramalan")
    st.markdown(
        """
        Kami membandingkan dua model deret waktu (Time Series) yang populer untuk peramalan penjualan bulanan:
        
        1. **SARIMA (Seasonal AutoRegressive Integrated Moving Average):** Model statistik yang kuat untuk data deret waktu yang menunjukkan tren, musiman, dan ketergantungan waktu.
        2. **Holt-Winters' Additive Method (Triple Exponential Smoothing):** Metode perataan eksponensial yang mencakup komponen level, tren, dan musiman.
        
        Analisis dan pelatihan kedua model ini dilakukan di Google Colab.
        """
    )
    st.subheader("Navigasi")
    st.markdown("Silakan jelajahi menu di sidebar untuk melihat performa model dan melakukan uji coba peramalan.")

def model_analysis_page():
    """Halaman Analisis Model: Menampilkan hasil fitting dan perbandingan metrik."""
    st.title("Analisis & Perbandingan Model")
    st.markdown("---")

    if sarima_model is None or hw_model is None or df_monthly is None:
        st.warning("Model atau Data Bulanan belum termuat sepenuhnya. Silakan cek status di log.")
        return

    # 1. Metrik Perbandingan (Asumsi metrik dihitung saat pelatihan)
    st.header("Perbandingan Metrik Evaluasi Model")

    # Ambil metrik dari hasil training (Jika tersedia)
    try:
        sarima_aic = sarima_model.aic
        sarima_bic = sarima_model.bic
        sarima_llf = sarima_model.llf
    except:
        sarima_aic, sarima_bic, sarima_llf = "N/A", "N/A", "N/A"
    
    try:
        hw_aic = hw_model.aic
        hw_bic = hw_model.bic
        hw_sse = hw_model.sse # Sum of Squared Errors
    except:
        hw_aic, hw_bic, hw_sse = "N/A", "N/A", "N/A"

    col_metric1, col_metric2 = st.columns(2)

    with col_metric1:
        st.subheader("SARIMA Model Summary")
        st.markdown(f"**Best Order:** {sarima_model.order} x {sarima_model.seasonal_order}")
        st.metric(label="Akaike Info Criterion (AIC)", value=f"{sarima_aic:.2f}" if isinstance(sarima_aic, float) else sarima_aic)
        st.metric(label="Bayesian Info Criterion (BIC)", value=f"{sarima_bic:.2f}" if isinstance(sarima_bic, float) else sarima_bic)
    
    with col_metric2:
        st.subheader("Holt-Winters Model Summary")
        st.markdown(f"**Seasonal Method:** {hw_model.model.seasonal}")
        st.metric(label="Akaike Info Criterion (AIC)", value=f"{hw_aic:.2f}" if isinstance(hw_aic, float) else hw_aic)
        st.metric(label="Bayesian Info Criterion (BIC)", value=f"{hw_bic:.2f}" if isinstance(hw_bic, float) else hw_bic)
        st.metric(label="Sum of Squared Errors (SSE)", value=f"{hw_sse:.2f}" if isinstance(hw_sse, float) else hw_sse)


    # 2. Plotting Hasil Fitting (In-Sample Prediction)
    st.header("Visualisasi Kinerja Model (Data Historis)")
    
    # Ambil data aktual (Monthly Sales)
    actual_data = df_monthly.to_timestamp()
    
    # In-Sample Prediction SARIMA
    sarima_fitted_values = sarima_model.fittedvalues
    # Ubah index SARIMA ke datetime
    sarima_fitted_values.index = pd.to_datetime(sarima_fitted_values.index.astype(str))
    
    # In-Sample Prediction Holt-Winters
    hw_fitted_values = hw_model.fittedvalues
    # Ubah index Holt-Winters ke datetime
    hw_fitted_values.index = pd.to_datetime(hw_fitted_values.index.to_period('M').to_timestamp())


    # Visualisasi
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_data, label='Penjualan Aktual (Qty)', color='black', linewidth=2)
    ax.plot(sarima_fitted_values, label='SARIMA Fitted', linestyle='--', color='red')
    ax.plot(hw_fitted_values, label='Holt-Winters Fitted', linestyle='-.', color='blue')
    
    ax.set_title('Perbandingan Kinerja In-Sample Model (Penjualan Bulanan)', fontsize=16)
    ax.set_xlabel('Tanggal', fontsize=12)
    ax.set_ylabel('Kuantitas Penjualan (Qty)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    st.pyplot(fig)
    
    st.markdown(
        """
        Grafik di atas menunjukkan seberapa baik masing-masing model dapat menyesuaikan diri 
        (fitting) dengan pola data penjualan historis (garis hitam). Model yang baik akan 
        memiliki garis fitted (merah dan biru) yang sangat dekat dengan garis Aktual.
        """
    )

def forecast_page():
    """Halaman Uji Coba: Melakukan peramalan untuk periode tertentu."""
    st.title("Uji Coba Peramalan Penjualan")
    st.markdown("---")

    if sarima_model is None or hw_model is None or df_monthly is None:
        st.warning("Model atau Data Bulanan belum termuat sepenuhnya. Silakan coba lagi atau pastikan file model tersedia.")
        return

    st.header("Lakukan Peramalan Bulanan")
    
    # Input dari pengguna
    periods = st.slider(
        "Pilih jumlah bulan ke depan yang akan diramal:",
        min_value=1, 
        max_value=12, 
        value=3, 
        step=1
    )

    if st.button("Jalankan Peramalan"):
        st.subheader(f"Hasil Peramalan untuk {periods} Bulan ke Depan")

        # --- 1. Peramalan SARIMA ---
        try:
            # Menggunakan .get_forecast() untuk mendapatkan interval kepercayaan
            sarima_forecast_result = sarima_model.get_forecast(steps=periods)
            sarima_forecast = sarima_forecast_result.predicted_mean
            sarima_conf_int = sarima_forecast_result.conf_int()
            
            # Ubah index periode menjadi datetime
            sarima_forecast.index = sarima_forecast.index.to_timestamp()
            sarima_conf_int.index = sarima_conf_int.index.to_timestamp()

            sarima_results = pd.DataFrame({
                'Bulan': sarima_forecast.index.strftime('%Y-%m'),
                'SARIMA Prediksi (Qty)': sarima_forecast.round(0).astype(int),
                'SARIMA Batas Bawah': sarima_conf_int.iloc[:, 0].round(0).astype(int),
                'SARIMA Batas Atas': sarima_conf_int.iloc[:, 1].round(0).astype(int)
            }).set_index('Bulan')
            
        except Exception as e:
            st.error(f"Gagal melakukan peramalan SARIMA: {e}")
            sarima_results = None
        
        # --- 2. Peramalan Holt-Winters ---
        try:
            # Menggunakan .forecast()
            hw_forecast = hw_model.forecast(steps=periods)
            
            # Ubah index periode menjadi datetime
            hw_forecast.index = hw_forecast.index.to_timestamp()

            hw_results = pd.DataFrame({
                'Holt-Winters Prediksi (Qty)': hw_forecast.round(0).astype(int)
            }, index=hw_forecast.index.strftime('%Y-%m'))
            
        except Exception as e:
            st.error(f"Gagal melakukan peramalan Holt-Winters: {e}")
            hw_results = None
        
        # --- 3. Menggabungkan Hasil ---
        if sarima_results is not None and hw_results is not None:
            # Gabungkan kedua hasil berdasarkan index Bulan
            combined_results = sarima_results.join(hw_results)
            st.dataframe(combined_results, use_container_width=True)

            # --- 4. Plotting Hasil Peramalan ---
            # Data historis (Monthly)
            historical_data = df_monthly.to_timestamp()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            # Plot Historis
            ax.plot(historical_data.index, historical_data.values, label='Penjualan Aktual', color='black')
            
            # Plot SARIMA Forecast
            ax.plot(sarima_forecast.index, sarima_forecast.values, label='SARIMA Forecast', color='red', linestyle='--')
            # Plot SARIMA Confidence Interval
            ax.fill_between(sarima_conf_int.index, 
                            sarima_conf_int.iloc[:, 0], 
                            sarima_conf_int.iloc[:, 1], 
                            color='red', alpha=0.1, label='SARIMA 95% Confidence Interval')
            
            # Plot Holt-Winters Forecast
            ax.plot(hw_forecast.index, hw_forecast.values, label='Holt-Winters Forecast', color='blue', linestyle='-.')
            
            ax.set_title(f'Peramalan Penjualan {periods} Bulan ke Depan', fontsize=16)
            ax.set_xlabel('Tanggal', fontsize=12)
            ax.set_ylabel('Kuantitas Penjualan (Qty)', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Tampilkan Plot
            st.pyplot(fig)

    st.markdown("---")
    st.header("Navigasi ke Colab untuk Pelatihan")
    st.markdown(
        """
        Jika Anda ingin melakukan pelatihan ulang (retrain) model dengan data baru, 
        mengubah parameter model, atau melakukan analisis data lebih mendalam, 
        silakan gunakan notebook Google Colab yang telah disediakan.
        
        Notebook Colab (`Copy_of_Forecasting_Instax_Sales.ipynb`) berisi semua langkah
        preprocessing data, EDA (Exploratory Data Analysis), pemilihan model, pelatihan, 
        dan evaluasi model SARIMA dan Holt-Winters.
        """
    )
    # Ganti URL ini dengan link langsung ke Colab Notebook Anda jika sudah di-host
    colab_url = "https://colab.research.google.com/github/YOUR_GITHUB_USER/YOUR_REPO/blob/main/Copy_of_Forecasting_Instax_Sales.ipynb" # Sesuaikan!
    st.markdown(f"[**Klik di sini untuk membuka Notebook Google Colab**]({colab_url})")
    st.caption("Catatan: Anda mungkin perlu menyesuaikan URL Colab di kode `app.py`.")


# --- STRUKTUR APLIKASI STREAMLIT ---

# Sidebar Navigasi
st.sidebar.title("Navigasi Aplikasi")
selection = st.sidebar.radio(
    "Pilih Halaman:",
    ["Penjelasan Aplikasi", "Analisis Model", "Uji Coba & Pelatihan"]
)

# Menampilkan Halaman yang Dipilih
if selection == "Penjelasan Aplikasi":
    home_page()
elif selection == "Analisis Model":
    model_analysis_page()
elif selection == "Uji Coba & Pelatihan":
    forecast_page()
    
st.sidebar.markdown("---")
st.sidebar.subheader("Info Data")
if df_raw is not None:
    st.sidebar.write(f"Data Transaksi: {len(df_raw):,} baris")
    st.sidebar.write(f"Periode Data: {df_raw['Tanggal'].min().strftime('%Y-%m')} s/d {df_raw['Tanggal'].max().strftime('%Y-%m')}")
