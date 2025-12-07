# 1. Taban İmaj: Python 3.10'un hafif versiyonunu (Linux tabanlı) kullan
FROM python:3.10-slim

# 2. Konteyner içinde çalışacağımız klasörü oluştur
WORKDIR /app

# 3. Önce sadece gereksinim listesini kopyala (Docker Cache avantajı için)
COPY requirements.txt .

# 4. Kütüphaneleri yükle (Önbellek kullanma ki imaj şişmesin)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Şimdi geri kalan tüm proje dosyalarını (kodlar, klasörler) içeri kopyala
COPY . .

# 6. Streamlit'in kullandığı portu (8501) dış dünyaya aç
EXPOSE 8501

# 7. Konteyner başladığında çalışacak komut
# API Key'i kullanıcıdan bekleyeceğimiz için burada tanımlamıyoruz
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]