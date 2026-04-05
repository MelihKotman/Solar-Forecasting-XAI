import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import pywt

def _as_2d_array(X):
    """
    Girdiyi (DataFrame veya DataFrame Olmayan) her duruma göre 2D float NumPy dizisine cevirir ve
    Orijinal tur bilgisini dondurur.
    """
    # Eğer DataFrame ise bu bloğa girer
    if isinstance(X, pd.DataFrame):
        data = X.values.astype(float) # Tüm değerlerimiz Float olur.
        # Orijinal veri tipini geri kurabilmek için kullanılan sözlük
        meta = {
            "is_df": True,       # DataFrame (True)
            "index": X.index,    # Satır indeksleri saklanır
            "columns": X.columns,# Sütun adları saklanır.
            "is_1d": False,      # Tek boyutlu değildir.
        }
        return data, meta

    # Eğer DataFrame değilse
    arr = np.asarray(X, dtype=float)  # Float tipinde DataFrame yap.
    if arr.ndim == 1:   # Tek boyutluysa
        arr = arr.reshape(-1, 1) # Tek boyutluya getir
        is_1d = True
    elif arr.ndim == 2: # Çift boyutluysa
        is_1d = False # Tek boyutlu değil işaretle
    else:
        raise ValueError("X 1D veya 2D olmalidir.")

    meta = {
        "is_df": False, # DataFrame değildi.
        "index": None,  # Satırları yoktur.
        "columns": None,# Sütunları yoktur.
        "is_1d": is_1d, # Boyuta göre
    }
    return arr, meta


def _restore_type(filtered, meta):
    """
    Filtrelenmis 2D diziyi orijinal tipe geri donusturur.
    """
    if meta["is_df"]:
        return pd.DataFrame(filtered, index=meta["index"], columns=meta["columns"])
    if meta["is_1d"]:
        return filtered.ravel()
    return filtered


def sav_gol_filter(X, window_length=9, polyorder=3):
    """
    Savitzky-Golay filtresi uygular (gunesli gunler icin ideal).

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        1D veya 2D sinyal.
    window_length : int, default=9
        Tek sayi olmali ve sinyal uzunlugundan buyuk olmamalidir.
    polyorder : int, default=3
        Polinom derecesi; window_length'ten kucuk olmalidir.
    """
    data, meta = _as_2d_array(X)
    rows = data.shape[0]

    # Eğer satır sayısı 3'ten az ise, filtreleme yapılamaz, orijinal veriyi döndürür.
    if rows < 3:
        return _restore_type(data.copy(), meta)

    # window_length tek sayı olmalı ve sinyal uzunluğundan büyük olmamalıdır.
    wl = min(window_length, rows if rows % 2 == 1 else rows - 1)
    wl = max(wl, 3)
    if wl % 2 == 0:
        wl -= 1

    # polyorder window_length'ten küçük olmalıdır.
    po = min(polyorder, wl - 1)
    if po < 1:
        po = 1

    filtered = savgol_filter(data, window_length=wl, polyorder=po, axis=0)
    return _restore_type(filtered, meta)


def wavelet_filter(X, wavelet="db4", level=2, threshold_scale=0.8, mode="soft"):
    """
    Wavelet tabanli gurultu azaltma uygular (kaotik gunler icin ideal).

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        1D veya 2D sinyal.
    wavelet : str, default="db4"
        PyWavelets dalgacik adi.
    level : int, default=2
        Ayrisma seviyesi.
    threshold_scale : float, default=0.8
        Evrensel esik carpanı (sigma*sqrt(2*log(n))*threshold_scale).
    mode : str, default="soft"
        Esikleme modu: "soft" veya "hard".
    """
    data, meta = _as_2d_array(X)
    rows, cols = data.shape
    filtered = data.copy()

    for i in range(cols):
        signal = data[:, i]
        max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
        use_level = max(1, min(level, max_level))

        coeffs = pywt.wavedec(signal, wavelet, level=use_level)

        detail_coeffs = coeffs[1:]
        if detail_coeffs:
            sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745
        else:
            sigma = 0.0

        uthresh = threshold_scale * sigma * np.sqrt(2.0 * np.log(max(len(signal), 2)))

        coeffs_thresh = [coeffs[0]]
        for c in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(c, value=uthresh, mode=mode))

        reconstructed = pywt.waverec(coeffs_thresh, wavelet)

        if len(reconstructed) > rows:
            reconstructed = reconstructed[:rows]
        elif len(reconstructed) < rows:
            reconstructed = np.pad(reconstructed, (0, rows - len(reconstructed)), mode="edge")

        filtered[:, i] = reconstructed

    return _restore_type(filtered, meta)


def mov_avg_filter(X, window=5, center=True):
    """
    Hareketli ortalama filtresi uygular (kapali gunler icin ideal).

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        1D veya 2D sinyal.
    window : int, default=5
        Pencere uzunlugu.
    center : bool, default=True
        Ortalama penceresini merkeze alir; lag etkisini azaltir.
    """
    data, meta = _as_2d_array(X)
    if window < 1:
        raise ValueError("window en az 1 olmalidir.")

    temp_df = pd.DataFrame(data)
    filtered = temp_df.rolling(window=window, center=center, min_periods=1).mean().values
    return _restore_type(filtered, meta)


def apply_filter_by_label(
    signal,
    label,
    savgol_window=9,
    savgol_polyorder=3,
    wavelet_name="db4",
    wavelet_level=2,
    wavelet_threshold_scale=0.8,
    moving_window=5,
):
    """
    Gun etiketine gore uygun filtreyi secip uygular.

    label degerleri: "Gunesli", "Kaotik", "Kapali".
    signal 1D/2D NumPy veya DataFrame olabilir.
    """
    normalized = str(label).strip().lower()

    if normalized == "gunesli":
        return sav_gol_filter(signal, window_length=savgol_window, polyorder=savgol_polyorder)
    if normalized == "kaotik":
        return wavelet_filter(
            signal,
            wavelet=wavelet_name,
            level=wavelet_level,
            threshold_scale=wavelet_threshold_scale,
        )
    if normalized == "kapali":
        return mov_avg_filter(signal, window=moving_window, center=True)

    raise ValueError(
        "Bilinmeyen label. Beklenen degerler: 'Gunesli', 'Kaotik', 'Kapali'."
    )