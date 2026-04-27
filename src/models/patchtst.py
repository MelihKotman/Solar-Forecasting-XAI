import torch
import torch.nn as nn

class CustomPatchTST(nn.Module):
    """
    Solar Forecasting için Kanal Bağımsız (Channel Independent) PatchTST modeli.
     - Kanal bağımsız: Her bir kanal için ayrı bir model eğitilir.
     - Patch tabanlı: Zaman serisi verisi, belirli uzunlukta parçalara (patch) bölünerek işlenir.
     - Transformer tabanlı: Verilerin uzun vadeli bağımlılıklarını yakalamak için transformer mimarisi kullanılır.
     - Esnek giriş boyutu: Model, farklı uzunluklarda zaman serisi verisi ile çalışabilir.
     - Çoklu çıktı: Her bir kanal için ayrı bir tahmin yapılır, bu da modelin esnekliğini artırır.
     - Düzenleme teknikleri: Overfitting'i önlemek için dropout ve erken durdurma gibi teknikler kullanılabilir.
     - Performans optimizasyonu: Modelin eğitim süresini azaltmak için GPU hızlandırması ve mini-batch eğitim gibi yöntemler uygulanabilir.
     - Değerlendirme metrikleri: Modelin performansını değerlendirmek için RMSE, MAE gibi metrikler kullanılabilir.
    """
    def __init__(self, seq_len = 96, pred_len = 24, enc_in = 11, patch_len = 16, stride = 8, d_model = 64, nhead = 4, num_layers = 2):
        super(CustomPatchTST, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride

        # Kaç adet (Yama) Patch çıkacağını matematiksel olarak hesaplama
        # Patch sayısı = (Toplam uzunluk - Yama uzunluğu) / Kayma + 1
        # 96 adımı, 16'lık yamalara 8'er adım kaydırarak bölüyoruz
        self.patch_num = int((seq_len - patch_len) / stride + 1)

        # Yamaları Transformer'ın anlayacağı d_model boyutuna dönüştürmek için lineer katman
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Transformer'ın kalbi olan encoder katmanları (decoder yok, çünkü sadece tahmin yapacağız)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # Çıktı Başlığı: Yamaları düzleştirip gelecekteki 24 adımı tahmin etmek için lineer katman
        self.flatten = nn.Flatten(start_dim = -2)
        self.linear_head = nn.Linear(self.patch_num * d_model, pred_len)

        # GHI Hedef Süzgeci: 11 farklı özelliği tek bir hedef tahmine dönüştürmek için lineer katman
        self.out_layer = nn.Linear(enc_in, 1)

    def forward(self, x):
        # x'in ilk boyutu 
        B, L, M = x.shape  # B: Batch size, L: Sequence length, M: Number of features (channels)

        # Kanal Bağımsızlığı ve Yamalama

        # Özellikleri baştan alıp her birini bağımsız bir zaman serisi olarak işleyelim
        x = x.permute(0, 2, 1)  # (B, M, L) -> (B, L, M)

        # Unfold ile 96 adımı 'patch_len' boyutunda yamalara bölüyoruz,
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Boyutu [Batch, 11, patch_num, patch_len] olur

        x = x.reshape(B * M, self.patch_num, self.patch_len)  # (Batch*11, patch_num, patch_len)

        # Transformer İşlemi
        x = self.patch_embedding(x)  # (Batch*11, patch_num, d_model)
        x = self.encoder(x)  # (Batch*11, patch_num, d_model)

        # Geleceği Tahmin Etme ve Birleştirme
        x = self.flatten(x) # (Batch*11, patch_num * d_model)
        x = self.linear_head(x)  # (Batch*11, 24)

        # Matrisi tekrar eski düzenine getiriyoruz
        x = x.reshape(B, M, self.pred_len)  # (Batch, 11, 24)
        x = x.permute(0, 2, 1)  # (Batch, 24, 11)

        # Sadece GHI tahminini almak için son katman
        out = self.out_layer(x)  # (Batch, 24, 1)

        return out.squeeze(-1)  # (Batch, 24) -> GHI tahminleri
        