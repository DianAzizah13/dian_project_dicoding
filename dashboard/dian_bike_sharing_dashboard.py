import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Function to display Time Series Decomposition
def time_series_decomposition(df):
    df.set_index('dteday', inplace=True)
    tsd_data = df['cnt']
    
    # Time series decomposition
    decomposition = seasonal_decompose(tsd_data, model='additive', period=30)
    
    # Plot the components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 6))

    ax1.plot(decomposition.observed, label='Observed')
    ax1.set_title('Cnt (Total Rentals)')
    ax1.grid(True)

    ax2.plot(decomposition.trend, label='Trend')
    ax2.set_title('Trend')
    ax2.grid(True)

    ax3.plot(decomposition.seasonal, label='Seasonal')
    ax3.set_title('Seasonal')
    ax3.grid(True)

    ax4.plot(decomposition.resid, label='Residual')
    ax4.set_title('Residual')
    ax4.grid(True)

    # Add label in x and y
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Date')
        ax.set_ylabel('Count')

    plt.tight_layout()
    st.pyplot(fig)

# Fungsi untuk menampilkan pola pengguna umum vs pengguna terdaftar
def plot_user_type(df):
    fig, ax = plt.subplots(figsize=(10, 6))  # Buat figure dan axes baru
    sns.lineplot(data=df, x='dteday', y='casual', label='Casual Users', color='blue', ax=ax)
    sns.lineplot(data=df, x='dteday', y='registered', label='Registered Users', color='green', ax=ax)
    ax.set_title('Usage Pattern: Casual vs Registered Users')
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    st.pyplot(fig)  # Berikan argumen figure

# Fungsi untuk menampilkan pola hari kerja vs akhir pekan
def plot_weekday_vs_weekend(df):
    workingday_data = df.groupby('workingday')[['casual', 'registered']].mean().reset_index()
    workingday_data['day_type'] = workingday_data['workingday'].map({0: 'Weekend/Holiday', 1: 'Working Day'})

    # Positions for bars
    x = np.arange(len(workingday_data))  # Bars position
    width = 0.30  # Width of each bar

    # Creating the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting casual data
    bars1 = ax.bar(x - width / 2, workingday_data['casual'], width, color='blue', label='Casual')

    # Plotting registered data
    bars2 = ax.bar(x + width / 2, workingday_data['registered'], width, color='orange', label='Registered')

    # Adding title and labels
    ax.set_title('Average Rentals: Working Day vs Weekend/Holiday')
    ax.set_ylabel('Number of Rentals')
    ax.set_xlabel('Day Type') 

    ax.set_xticks(x)
    ax.set_xticklabels(workingday_data['day_type'])
    ax.legend()

    # Adding bar labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}', ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}', ha='center', va='bottom')

    # Show the plot in Streamlit
    st.pyplot(fig)

def plot_highest_day(df):
    daily_data = df.groupby('weekday')[['casual', 'registered']].mean().reset_index()

    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting casual and registered data
    sns.lineplot(data=daily_data, x='weekday', y='casual', label='Casual', color='blue', ax=ax)
    sns.lineplot(data=daily_data, x='weekday', y='registered', label='Registered', color='orange', ax=ax)

    # Adding title and labels
    ax.set_title('Daily Rentals: Casual vs Registered')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Number of Rentals')
    ax.legend()

    # Customizing ticks and grid
    ax.set_xticks(range(0, 7))
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

# Fungsi untuk menampilkan pola penggunaan sepeda tertinggi per jam
def plot_hourly_pattern(df):
    hourly_data = df.groupby('hr')[['casual', 'registered']].mean().reset_index()

    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting casual and registered data
    sns.lineplot(data=hourly_data, x='hr', y='casual', label='Casual', color='blue', ax=ax)
    sns.lineplot(data=hourly_data, x='hr', y='registered', label='Registered', color='orange', ax=ax)

    # Adding title and labels
    ax.set_title('Hourly Rentals: Casual vs Registered')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Number of Rentals')
    ax.legend()

    # Customizing ticks and grid
    ax.set_xticks(range(0, 24, 2))  # Set ticks every 2 hours
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

def plot_heatmap_hour(df):
    heatmap_data = df.groupby(['weekday', 'hr'])[['casual', 'registered']].mean().unstack()

    # Create subplots for casual and registered rentals
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Heatmap for casual rentals
    sns.heatmap(heatmap_data['casual'], cmap='Blues', ax=axes[0], annot=False, cbar=True)
    axes[0].set_title('Casual Rentals Heatmap (Weekday vs Hour)')
    axes[0].set_ylabel('Weekday')
    axes[0].set_xlabel('Hour of Day')

    # Heatmap for registered rentals
    sns.heatmap(heatmap_data['registered'], cmap='Oranges', ax=axes[1], annot=False, cbar=True)
    axes[1].set_title('Registered Rentals Heatmap (Weekday vs Hour)')
    axes[1].set_ylabel('')  # Remove y-label for the second plot for cleaner look
    axes[1].set_xlabel('Hour of Day')

    # Adjust layout for a cleaner look
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# Sidebar navigasi
st.sidebar.title("Bike Rental Data Analysis 🚴‍♂️")
option = st.sidebar.selectbox("Pilih visualisasi:", 
                              ["Time Series Decomposition", 
                               "Pola Pengguna Umum & Terdaftar", 
                               "Pola Penggunaan dalam Hari",
                               "Pola Penggunaan dalam Jam"])

all_data = pd.read_csv('dashboard/all_data.csv')

# Menampilkan visualisasi berdasarkan pilihan di sidebar
if option == "Pola Pengguna Umum & Terdaftar":
    st.title("1. Pola Penyewaan Pengguna Umum & Terdaftar")
    plot_user_type(all_data)
    st.markdown("### Insight:")
    st.markdown("""
        - Penyewa terdaftar (registered) memiliki jumlah penyewaan yang jauh lebih tinggi daripada penyewa umum (casual). Tren penyewa terdaftar menunjukkan pola yang cukup konsisten dengan fluktuasi yang tinggi dan berulang, terutama di musim puncak.
        - Penyewa umum memiliki jumlah penyewaan yang jauh lebih rendah, namun mengalami sedikit peningkatan secara bertahap dalam periode waktu yang sama, terutama di bulan-bulan tertentu.
        - Fluktuasi yang lebih tinggi terlihat pada data penyewa terdaftar dibandingkan dengan penyewa umum. Hal ini dapat mengindikasikan adanya pola musiman atau hari-hari puncak dalam penyewaan sepeda oleh penyewa terdaftar.
    """)

elif option == "Pola Penggunaan dalam Hari":
    st.title("2. Perbedaan Penggunaan Sepeda antara Hari Kerja dan Akhir Minggu")
    plot_weekday_vs_weekend(all_data)
    st.markdown("### Insight:")
    st.markdown("""
        - Penyewa terdaftar jauh lebih banyak daripada penyewa umum, baik pada hari kerja maupun akhir pekan/liburan.
        - Penyewa terdaftar cenderung lebih sering menggunakan sepeda pada hari kerja, yang mengindikasikan bahwa mereka menggunakan sepeda sebagai moda transportasi reguler seperti untuk pergi ke kantor atau sekolah.
        - Penyewa umum, di sisi lain, cenderung lebih sering menyewa pada akhir pekan atau hari libur, yang mengindikasikan bahwa mereka lebih banyak menggunakan sepeda untuk rekreasi atau kegiatan santai.
    """)
    plot_highest_day(all_data)
    st.markdown("### Insight:")
    st.markdown("""
        - Jumlah penyewa sepeda relatif stabil selama hari kerja (dari 0 hingga 6 pada sumbu x). Jumlah rata-rata penyewaan berkisar antara 20 hingga 30 sepeda.
        - Tidak ada fluktuasi yang signifikan dalam perilaku penyewaan penyewa umum. Hal ini menunjukkan bahwa penyewa umum dapat menggunakan sepeda secara konsisten tanpa terpengaruh oleh hari kerja atau hari libur.
        - Penyewa terdaftar menunjukkan tren yang jelas selama seminggu, dengan peningkatan yang signifikan pada awal minggu (hari ke-0) yang kemungkinan besar terjadi pada hari Senin.
        - Penyewaan mencapai puncaknya pada pertengahan hingga akhir minggu (sekitar hari ke-4 atau ke-5), dan menurun tajam pada hari terakhir (hari ke-6), yang kemungkinan besar adalah hari Minggu. 
        - Hal ini menunjukkan bahwa penyewa yang terdaftar cenderung lebih aktif menggunakan sepeda selama hari kerja, terutama di pertengahan hingga akhir minggu. Penurunan pada hari terakhir menunjukkan bahwa penyewa terdaftar mungkin lebih jarang menggunakan sepeda pada hari libur atau akhir pekan.
    """)

elif option == "Pola Penggunaan dalam Jam":
    st.title("3. Pola Penggunaan Sepeda dalam Jam")
    plot_hourly_pattern(all_data)
    st.markdown("### Insight:")
    st.markdown("""
        - Penyewa umum memiliki pola penyewaan yang relatif stabil sepanjang hari. Tidak ada lonjakan jumlah penyewaan yang signifikan di mana jumlah penyewaan perlahan meningkat mulai dari pagi hari (jam 8 pagi), mencapai puncaknya antara jam 11 pagi dan jam 4 sore, kemudian perlahan menurun setelah jam 6 sore.
        - Penyewa terdaftar menunjukkan pola yang lebih jelas dengan dua puncak aktivitas. Puncak pertama terjadi sekitar pukul 8 pagi, kemungkinan besar terkait dengan perjalanan ke tempat kerja. Setelah puncak pagi, jumlah penyewaan menurun, kemudian meningkat lagi di sore hari dengan puncak kedua sekitar pukul 17-18, kemungkinan besar terkait dengan perjalanan pulang kerja. Setelah pukul 18.00, jumlah penyewaan sepeda menurun drastis hingga malam hari.
    """)
    plot_heatmap_hour(all_data)
    st.markdown("### Insight:")
    st.markdown("**Heatmap Penyewa Umum (Kiri):**")
    st.markdown("""
        - Aktivitas penyewaan oleh penyewa umum lebih tinggi antara pukul 10.00 hingga 18.00 di semua hari dalam seminggu.
        - Pola ini relatif konsisten baik pada hari kerja maupun akhir pekan, dengan sedikit peningkatan pada akhir pekan (jam 0 dan 6) di siang hari.
        - Hal ini menunjukkan bahwa penyewa umum lebih cenderung menyewa sepeda pada sore hingga malam hari, kemungkinan besar untuk rekreasi atau kegiatan yang lebih santai.
    """)
    st.markdown("**Heatmap Penyewa Terdaftar (Kanan):**")
    st.markdown("""
        - Aktivitas penyewaan oleh penyewa terdaftar menunjukkan pola yang sangat berbeda, dengan dua lonjakan yang jelas: satu di pagi hari antara pukul 07.00 dan 09.00, dan satu lagi di sore hari antara pukul 17.00 dan 19.00, terutama pada hari kerja (hari ke-1 dan ke-2).
        - Di akhir pekan (hari ke-0 dan ke-6), pola ini hampir tidak terlihat, dengan jumlah penyewaan yang lebih rendah sepanjang hari.
        - Hal ini menegaskan bahwa penyewa yang terdaftar cenderung menggunakan sepeda sebagai bagian dari rutinitas perjalanan harian mereka, terutama untuk pergi dan pulang kerja pada hari kerja.
    """)

elif option == "Time Series Decomposition":
    st.title("Time Series Decomposition")
    time_series_decomposition(all_data)
    st.markdown("### Insight:")
    st.markdown("""
        **1. Trend**
        - Tren menunjukkan pola peningkatan penggunaan sepeda dari waktu ke waktu.
        - Terjadi peningkatan jumlah penyewaan sepeda dari awal 2011 hingga pertengahan 2012, yang kemudian diikuti oleh fluktuasi. Meskipun terjadi pasang surut, tren secara umum menunjukkan peningkatan selama periode analisis.
        - Hal ini dapat mengindikasikan bahwa popularitas berbagi sepeda meningkat selama periode tersebut, mungkin karena faktor-faktor seperti kesadaran lingkungan atau kenyamanan.
        **2. Seasonal**
        - Komponen Seasonal terlihat datar, yang dapat mengindikasikan bahwa periode musiman yang digunakan dalam dekomposisi tidak sesuai atau data tidak memiliki pola musiman yang signifikan.
        **3. Residual**
        - Residual yang cukup besar dan terlihat mengandung banyak variasi, mengindikasikan adanya variabel eksternal atau outlier yang mempengaruhi jumlah penyewaan sepeda.
    """)

