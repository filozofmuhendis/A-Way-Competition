# Exoplanet Detection System

Bu proje, K2 veri setinden öznitelik çıkarımı yapan bulanık mantık sistemi ve desteklemeli öğrenme (PPO) algoritması kullanarak exoplanet tespiti yapmaktadır.

## 🌟 Özellikler

### Bulanık Mantık Sistemi
- **SNR_transit**: Sinyal gücü analizi
- **β faktörü**: Veri kalitesi/gürültü tipi değerlendirmesi
- **Odd-Even farkı**: Düzenlilik/EB ayrımı
- **Duty Cycle (q)**: Geometrik tutarlılık
- **Depth Consistency**: Sinyalin zaman içindeki kararlılığı

### Desteklemeli Öğrenme
- **PPO Algoritması**: Politika tabanlı öğrenme
- **Üç Ajan Tipi**:
  - **Simple**: Basit veri setleri için (3 öznitelik)
  - **Intermediate**: Orta karmaşıklık (6 öznitelik)
  - **Complex**: Karmaşık veri setleri (10 öznitelik)

## 📁 Dosya Yapısı

```
├── feature_extracted__k2_dataset_clean.csv  # Ana veri seti
├── fuzzy_logic_system.py                    # Bulanık mantık sistemi
├── rl_environment.py                        # Desteklemeli öğrenme ortamı
├── ppo_agent.py                            # PPO algoritması
├── integrated_system.py                    # Entegre sistem
├── demo.py                                 # Hızlı demo
├── test_system.py                          # Sistem testleri
├── requirements.txt                        # Gerekli kütüphaneler
└── README.md                              # Bu dosya
```

## 🚀 Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Eksik kütüphaneler için:
```bash
pip install scikit-fuzzy gym
```

## 💻 Kullanım

### Hızlı Demo
```bash
python demo.py
```

### Tam Analiz
```bash
python integrated_system.py
```

### Sistem Testleri
```bash
python test_system.py
```

## 📊 Sistem Bileşenleri

### 1. Bulanık Mantık Sistemi (`fuzzy_logic_system.py`)
- Beş öznitelik için üyelik fonksiyonları
- 27 bulanık kural
- Exoplanet olasılık hesaplama

### 2. Desteklemeli Öğrenme Ortamı (`rl_environment.py`)
- Üç farklı karmaşıklık seviyesi
- Dinamik ödül sistemi
- Multi-agent destek

### 3. PPO Ajanı (`ppo_agent.py`)
- Actor-Critic ağ yapısı
- Adaptif karmaşıklık
- Deneyim tekrarı

### 4. Entegre Sistem (`integrated_system.py`)
- Bulanık mantık + RL entegrasyonu
- Ensemble tahminleri
- Performans değerlendirmesi
- Görselleştirme

## 📈 Performans Metrikleri

Sistem aşağıdaki metrikleri kullanarak değerlendirilir:
- **Doğruluk (Accuracy)**
- **Kesinlik (Precision)**
- **Duyarlılık (Recall)**
- **F1-Score**
- **Ortalama Ödül**

## 🔧 Yapılandırma

### Bulanık Mantık Parametreleri
- Üyelik fonksiyonları: Üçgen (trimf)
- Çıkarım yöntemi: Mamdani
- Durulaştırma: Centroid

### PPO Parametreleri
- Learning Rate: 0.0003
- Gamma: 0.99
- GAE Lambda: 0.95
- Clip Epsilon: 0.2

## 📋 Veri Seti

`feature_extracted__k2_dataset_clean.csv` dosyası:
- 161 örnek
- 22 öznitelik
- K2 misyonu verileri
- Önceden işlenmiş öznitelikler

## 🎯 Sonuçlar

Sistem üç farklı ajan tipi ile test edilir:
- **Simple Ajan**: Yüksek doğruluk, düşük karmaşıklık
- **Intermediate Ajan**: Dengeli performans
- **Complex Ajan**: Detaylı analiz, yüksek karmaşıklık

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

**Not**: Bu sistem araştırma amaçlı geliştirilmiştir ve sürekli geliştirilmektedir.