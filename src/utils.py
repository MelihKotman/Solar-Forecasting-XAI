import pandas as pd
import numpy as np

def chronogical_split(df, datetime_col = "datetime"):
    """
    Zaman serisi verisini gelecekten kopya çekeyi (Data Leakage) önlemek için kronolojik olarak böler.

    Train : 2020 - 2022
    Val : 2023
    Test : 2024
    """

    temp_df = df.copy() # Orijinal DataFrame'i değiştirmemek için kopyasını oluştur

    # GÜVENLİK ÖNLEMİ: Eğer 'datetime' hem index'te hem sütundaysa çakışmayı önle
    if datetime_col in temp_df.columns and temp_df.index.name == datetime_col:
        temp_df = temp_df.reset_index(drop=True) # İndeksi at, sütunu kullan
    elif temp_df.index.name == datetime_col:
        temp_df = temp_df.reset_index() # Sadece indekste varsa sütuna al
    
    temp_df[datetime_col] = pd.to_datetime(temp_df[datetime_col]) # Datetime formatına çevirme

    # Tarihe göre kesin olarak sıralandığından emin ol
    temp_df = temp_df.sort_values(by=datetime_col).reset_index(drop=True)

    # Tarih aralıklarını belirle
    train_end = pd.Timestamp("2022-12-31 23:59:59")
    val_end = pd.Timestamp("2023-12-31 23:59:59")

    # Maskelerle veriyi böl
    train_mask = temp_df[datetime_col] <= train_end
    val_mask = (temp_df[datetime_col] > train_end) & (temp_df[datetime_col] <= val_end)
    test_mask = temp_df[datetime_col] > val_end

    # Bölünmüş veri setlerini oluştur
    train_df = temp_df[train_mask].reset_index(drop=True)
    val_df = temp_df[val_mask].reset_index(drop=True)
    test_df = temp_df[test_mask].reset_index(drop=True)

    return train_df, val_df, test_df

def create_sliding_features(df, target_col, feature_cols, look_back=96, horizon=24):
    """
    Her bir şehir için kayan pencereler (sliding windows) oluşturarak özellikler ve hedef değişkenler üretir.
    Şehirlerin sınırlarında birbirine karışmasını (Teleportation Bug) engellenir.

    Args:
        df (pd.DataFrame): Orijinal veri seti.
        target_col (str): Tahmin edilmek istenen hedef sütun adı.
        feature_cols (list): Özellik olarak kullanılacak sütun adları.
        look_back (int): Geçmiş kaç zaman adımının özellik olarak kullanılacağı.
        horizon (int): Gelecekte kaç zaman adımının tahmin edileceği.
    Returns:
        X: [Örnek Sayısı, Look Back (Geçmiş Zaman Adımları), Özellik Sayısı] -> 3D Tensör
        y: [Örnek Sayısı, Horizon (Gelecek Zaman Adımları)] -> 2D Tensör
    """
    import numpy as np

    X_list, y_list = [], []

    # Şehir bazında işlem yaparak teleportation bug'ını önle
    for city, group in df.groupby('City'):

        # NumPy dizilerine çevir
        features = group[feature_cols].values
        target = group[target_col].values

        # Kayan pencere oluşturma 
        for i in range(len(group) - look_back - horizon + 1):

            X_window = features[i:i+look_back]  # Geçmiş özellikler
            y_window = target[i+look_back:i+look_back+horizon]  # Gelecek hedefler

            X_list.append(X_window)
            y_list.append(y_window)

    # Listeleri NumPy dizilerine dönüştür
    X = np.array(X_list, dtype = np.float32)
    y = np.array(y_list, dtype = np.float32)
    return X, y