import requests
import pandas as pd
from io import StringIO
from typing import Optional

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