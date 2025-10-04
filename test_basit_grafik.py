"""
Basit test scripti - CSV grafikleyici sistemini test eder
"""

from csv_grafikleyici import CSVGrafikleyici
import os

def test_basit():
    """Basit test fonksiyonu"""
    print("🧪 CSV Grafikleyici Test")
    print("=" * 30)
    
    # En küçük CSV dosyasını test et
    test_dosya = "feature_extracted__k2_dataset_clean.csv"
    
    if not os.path.exists(test_dosya):
        print(f"❌ Test dosyası bulunamadı: {test_dosya}")
        return
    
    print(f"📁 Test dosyası: {test_dosya}")
    
    # Grafikleyici oluştur
    grafikleyici = CSVGrafikleyici(test_dosya)
    
    # CSV yükle
    if not grafikleyici.csv_yukle():
        print("❌ CSV yükleme başarısız!")
        return
    
    print("✅ CSV başarıyla yüklendi!")
    
    # Özet rapor
    print(grafikleyici.ozet_rapor())
    
    # İlk sayısal sütun için grafik oluştur
    sayisal_sutunlar = [sutun for sutun, bilgi in grafikleyici.sutun_bilgileri.items() if bilgi['sayisal']]
    
    if sayisal_sutunlar:
        test_sutun = sayisal_sutunlar[0]
        print(f"\n📈 Test grafiği oluşturuluyor: {test_sutun}")
        
        basarili = grafikleyici.tek_sutun_grafik(test_sutun, kaydet=True, goster=False)
        
        if basarili:
            print("✅ Test grafiği başarıyla oluşturuldu!")
        else:
            print("❌ Test grafiği oluşturulamadı!")
    else:
        print("❌ Sayısal sütun bulunamadı!")

if __name__ == "__main__":
    test_basit()