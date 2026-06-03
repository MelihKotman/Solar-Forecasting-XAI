import requests
import pandas as pd
from io import StringIO
from typing import Optional
import time

import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
import pvlib

def fetch_nrel_data(api_key: str, lat: float, lon: float, year: int = 2023) -> Optional[pd.DataFrame]:
    """
    NREL NSRDB API'sinden belirtilen konum ve yıl için güneş/meteroloji verilerini indirir.
    url = "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
    Args:
        api_key (str): NREL geliştirici API anahtarı.
        lat (float): Veri alınacak noktanın enlem değeri.
        lon (float): Veri alınacak noktanın boylam değeri.
        year (int): Verinin alınacağı yıl. İstek içinde `names` alanına aktarılır.

    Returns:
        Optional[pd.DataFrame]: API'den dönen ham veriyi tablo olarak içerir.
        İstek veya parse başarısız olursa None döner.
    """

    url = "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"

    # PAYLOAD OLUŞTURMA
    # Dokümantasyondaki zorunlu alanları buraya ekle.

    payload = {
        'api_key' : api_key,
        'names' : str(year), # Yıl bilgisi burada sağlanır, örneğin 2020
        'leap_day' : 'false', # Artık yıl verisi dahil edilmez
        'interval' : 30, #30 dakikalık veriler
        'utc' : 'false', # Veriler yerel saat diliminde sağlanır
        'full_name' : 'Melih Yiğit Kotman', # Geliştirici adı
        'email' : 'mykotman@icloud.com', # Geliştirici e-posta adresi
        'affiliation' : 'BAIBU', # Geliştirici kuruluşu
        'attributes': 'ghi,dhi,dni,wind_speed,air_temperature,cloud_type,dew_point,relative_humidity,solar_zenith_angle',# İstenen veri türleri
        'wkt' : f'POINT({lon} {lat})' # Verinin alınacağı nokta, WKT formatında
    }

    headers = {
        'cache-control': "no-cache",
        'content-type': "application/x-www-form-urlencoded"
    }

    # HTTP İSTEĞİ GÖNDERME
    try:
        # requests.get() fonksiyonu ile API'ye GET isteği gönderilir.
        response = requests.get(url, params=payload, headers=headers, timeout=10) # timeout ekleyerek uzun süren isteklerde hata alınmasını önleriz.
        response.raise_for_status()
        
        # VERİYİ PANDAS DATAFRAME'INE DÖNÜŞTÜRME

        # response.text içindeki veriyi StringIO ile sarmalayıp Pandas'a ver.
        # skiprows = 2 ile iki satır atlanır, çünkü ilk iki satır genellikle meta veri içerir.

        df = pd.read_csv(StringIO(response.text), skiprows=2)

        return df

    except pd.errors.ParserError as e:
        print(f"CSV parse hatası: {e}")
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"API isteği sırasında hata oluştu: {e}")
        return None  # Hata durumunda boş bir DataFrame döndürülebilir veya None döndürülebilir

def build_multicity_dataset(api_key: str, cities: dict[str, tuple], years: list[int]) -> Optional[pd.DataFrame]:
    """
    Birden fazla şehir ve yıl için NREL verisini çeker, birleştirir ve 'City' kolonu ekler.
    """
    frames = []
    
    # Her şehir ve yıl kombinasyonu için veriyi çek ve işaretle
    for city_name, coords in cities.items():
        lat, lon = coords
        print(f"\n {city_name} bölgesi için veri hasadı başlıyor...")
        
        for y in years:
            print(f"   -> {y} verisi çekiliyor...")
            df_year = fetch_nrel_data(api_key, lat, lon, y)
            
            if df_year is not None:
                df_year["Requested Year"] = y
                df_year["City"] = city_name  # En kritik dokunuş: Hangi şehir olduğunu etiketliyoruz!
                frames.append(df_year)
            else:
                print(f"   HATA: {city_name} {y} için veri çekilemedi.")

    if not frames:
        return None

    # Bütün parçaları tek bir devasa tabloda birleştir
    print("\n Bütün veriler başarıyla indirildi, birleştiriliyor...")
    final_df = pd.concat(frames, ignore_index=True)
    
    return final_df


def fetch_global_ood_data(start_date="2023-01-01", end_date="2023-12-31"):
    """
    Modelin OOD (Out-of-Distribution) genelleme yeteneğini test etmek için
    farklı kıtalardan verileri çeker ve NREL formatına hizalar.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    locations = {
    # ─── 0. ORİJİNAL LİSTE (Sıcak, Kurak ve Yüksek GHI) ──────────────────────
    "Sevilla_ES":      {"lat": 37.39,  "lon": -5.98},    # Güneybatı Avrupa, kurak yaz
    "Almeria_ES":      {"lat": 36.83,  "lon": -2.46},    # Akdeniz kurak
    "Tucson_AZ":       {"lat": 32.22,  "lon": -110.97},  # Çöl, yüksek güneşlenme
    "LasVegas_NV":     {"lat": 36.17,  "lon": -115.14},  # Çöl, yüksek GHI
    "Muscat_OM":       {"lat": 23.59,  "lon": 58.41},    # Subtropikal kurak
    
    # ─── 1. ZORLU SENARYO (Sürekli Kapalı, Yağışlı ve Düşük GHI) ─────────────
    "London_UK":       {"lat": 51.51,  "lon": -0.13},    # Okyanusal, kapalı ve düşük ışınım
    "Seattle_WA":      {"lat": 47.61,  "lon": -122.33},  # Günlerce süren yağmur/bulut
    "Berlin_DE":       {"lat": 52.52,  "lon": 13.40},    # Karasal Avrupa, kışın koyu gri gökyüzü
    
    # ─── 2. KAOTİK SENARYO (Tropikal, Muson ve Ani Kırılmalar) ───────────────
    "Singapore_SG":    {"lat": 1.35,   "lon": 103.82},   # Ekvatoral, aniden bastıran sağanaklar
    "Mumbai_IN":       {"lat": 19.08,  "lon": 72.88},    # Muson iklimi, aşırı bulutlanma geçişleri
    "Miami_FL":        {"lat": 25.76,  "lon": -80.19},   # Subtropikal, yüksek nem ve okyanus fırtınaları
    
    # ─── 3. FİZİKSEL SINIRLAR (Ekstrem Zenith Açısı ve Kar Yansıması/Albedo) ─
    "Oslo_NO":         {"lat": 59.91,  "lon": 10.75},    # Yüksek enlem, çok dar açılı kış güneşi
    "Anchorage_AK":    {"lat": 61.22,  "lon": -149.90},  # Çok yüksek enlem, dondurucu soğuk ve kar yansıması
    "Toronto_CA":      {"lat": 43.65,  "lon": -79.38},   # Karasal soğuk, kışın karlı kapalı günler
    
    # ─── 4. TERS DİNAMİKLER (Güney Yarımküre - Mevsimsel Ezber Sınaması) ─────
    "AliceSprings_AU": {"lat": -23.70, "lon": 133.88},   # Güney Y.K. (Ocak-Mart arası kavurucu yazdır)
    "Calama_CL":       {"lat": -22.45, "lon": -68.93},   # Atacama Çölü, dünyadaki en yüksek GHI noktalarından
    "CapeTown_ZA":     {"lat": -33.92, "lon": 18.42},    # Akdeniz iklimi ama mevsimler ters
    
    # ─── 5. MEKANSAL EZBER KONTROLÜ (Çok Yakın Coğrafyalar) ──────────────────
    "Cordoba_ES":      {"lat": 37.89,  "lon": -4.78},    # Sevilla'ya komşu (Bölgeyi mi ezberledi, fiziği mi?)
    "Phoenix_AZ":      {"lat": 33.45,  "lon": -112.07},  # Tucson'a çok yakın devasa çöl
    "Dubai_AE":        {"lat": 25.20,  "lon": 55.27}     # Muscat komşusu, kum fırtınası dinamikleri
}

    all_city_data = []
    
    for city, coords in locations.items():
        print(f"🌍 {city} için OOD test verisi çekiliyor...")
        time.sleep(2)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coords['lat'],
            "longitude": coords['lon'],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover", "shortwave_radiation"],
            "wind_speed_unit": "ms",
            "timezone" : "auto"  
        }
        try:
            response = openmeteo.weather_api(url, params=params, timeout = 10)[0]
        except Exception as e:
            print(f"❌ {city} için veri çekilemedi: {e}")
            continue
        hourly = response.Hourly()
        
        df = pd.DataFrame({"datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )})
        
        df["Temperature"] = hourly.Variables(0).ValuesAsNumpy()
        df["Relative Humidity"] = hourly.Variables(1).ValuesAsNumpy()
        df["Wind Speed"] = hourly.Variables(2).ValuesAsNumpy()
        df["Cloud Type"] = np.round((hourly.Variables(3).ValuesAsNumpy() / 100) * 9) # 0-9 NREL Formatı
        df["GHI_Filtered"] = hourly.Variables(4).ValuesAsNumpy()
        df = df.set_index('datetime')
        
        # Fiziksel ve Astronomik Hesaplamalar
        solpos = pvlib.solarposition.get_solarposition(df.index, coords['lat'], coords['lon'])
        df['Solar Zenith Angle'] = solpos['zenith'].values
        dni_extra = pvlib.irradiance.get_extra_radiation(df.index).values
        dni_extra = np.where(dni_extra == 0, 1, dni_extra)
        df['Clearness_Index'] = np.where(df['GHI_Filtered'] > 0, df['GHI_Filtered'] / dni_extra, 0)
        
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        
        expected_cols = ['GHI_Filtered', 'Solar Zenith Angle', 'Temperature', 'Relative Humidity', 
                         'Wind Speed', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'Cloud Type', 'Clearness_Index']
        
        df = df.reset_index()[['datetime'] + expected_cols]
        df['City'] = city
        all_city_data.append(df)

    return pd.concat(all_city_data, ignore_index=True)