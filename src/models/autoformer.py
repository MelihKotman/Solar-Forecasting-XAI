import torch
import torch.nn as nn

class MovingAverage(nn.Module):
    """
    Zaman serisindeki 'Trend'i (Genel gidişatı) bulmak için Hareketli Ortalama bloğu.
    Autoformer modelinde, bu blok, zaman serisindeki genel eğilimi yakalamak ve kısa vadeli dalgalanmalardan arındırmak için kullanılır.
    """
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size # Hareketli ortalama hesaplamasında kullanılan pencere boyutu
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) # 1D ortalama havuzlama katmanı

    def forward(self, x):
        # Zamanın başından ve sonunda veri kaybetmemek için Padding (doldurma) işlemi yapıyoruz ki
        # hareketli ortalama hesaplaması sırasında veri kaybı yaşanmasın..
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) # Zaman serisinin başına padding ekleme
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1) # Zaman serisinin sonuna padding ekleme
        x = torch.cat([front, x, end], dim=1) # Padding eklenmiş zaman serisi

        # Boyutları AvgPool1d'ye uygun hale getir, ortalamayı al ve eski haline getir.
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1) # Hareketli ortalama hesaplama

        return x

class SeriesDecomposition(nn.Module):
    """
    Karmaşık sinyali -> Trend (Yavaş değişen genel gidişat) ve Seasonal (Hızlı değişen dalgalanma) bileşenlerine ayırmak için kullanılan seri ayrıştırma bloğu.
    """

    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size=kernel_size, stride=1) # Hareketli Ortalama bloğu

    def forward(self, x):
        trend = self.moving_avg(x) # Trend bileşenini hesapla
        seasonal = x - trend # Orijinal sinyalden trendi çıkararak seasonal bileşenini elde et bu şekilde geriye anlık dalgalanmayı yakalayabilirsin.
        return seasonal, trend

class CustomAutoFormer(nn.Module):
    """
    Autoformer için optimize edilmiş Solar Forecasting modeli.
    Bu model, zaman serisi verilerindeki trend ve sezonluk bileşenleri ayrıştırarak, gelecekteki değerleri tahmin etmek için tasarlanmıştır.
    """

    def __init__(self, seq_len = 96, pred_len = 24, enc_in = 11, d_model = 64, kernel_size = 25):
        super(CustomAutoFormer, self).__init__()
        self.seq_len = seq_len # Giriş dizisinin uzunluğu (örneğin, geçmiş 96 zaman adımı)
        self.pred_len = pred_len # Tahmin edilecek zaman adımlarının sayısı (örneğin, gelecek 24 zaman adımı)

        # Ayrıştırma bloğu
        self.decomp = SeriesDecomposition(kernel_size=kernel_size) # Seri ayrıştırma bloğu

        # Lineer İzdüşüm Katmanları (Trend ve Seasonal bileşenlerini modellemek için)
        # Amaç 96 adımlık geçmişi 24 adımlık geleceğe dönüştürmek, bu yüzden seq_len -> pred_len dönüşümü yapıyoruz.
        self.trend_projection = nn.Linear(seq_len, pred_len) # Trend bileşeni için lineer izdüşüm
        self.seasonal_projection = nn.Linear(seq_len, pred_len) # Seasonal bileşeni için lineer izdüşüm

        # Özellikleri birleştirip tek bir GHI tahmini yapmak için son bir lineer katman
        self.outlayer = nn.Linear(enc_in, 1) # Son tahmin için lineer katman


    def forward(self, x):
        # x'in boyutu: [Batch Size, Sequence Length, Feature Count] -> [B, 96, 11]

        # Sinyali Parçala
        seasonal_init, trend_init = self.decomp(x) # Giriş sinyalini seasonal ve trend bileşenlerine ayır

        # Geçmişi Geleceğe Dönüştür (Boyut değişimi: [B, 96, D] -> [B, 24, D])
        trend_part = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1) # Trend bileşenini geleceğe dönüştür
        seasonal_part = self.seasonal_projection(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1) # Seasonal bileşenini geleceğe dönüştür

        # İki yapboz parçasını ( Gelecek Trend ve Gelecek Seasonal) birleştir
        # Boyut: [Batch, 24, 11]
        forecast = trend_part + seasonal_part # Trend ve Seasonal bileşenlerini toplayarak birleşik bir temsil oluştur

        # Son tahmin için özellikleri birleştir ve tek bir GHI tahmini yap
        # Boyut: [Batch, 24, 1]
        final_output = self.outlayer(forecast) # Son tahmin katmanından geçirme

        # Çıktı boyutunu [Batch, 24] yaparak geri döndür
        return final_output.squeeze(-1) # Son boyutu kaldırarak [Batch, 24] boyutunda çıktı döndür

        