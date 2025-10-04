"""
CSV Grafikleyici Demo - Sistemin kullanımını gösteren demo scripti
Bu script, klasördeki CSV dosyalarını analiz eder ve grafik oluşturma örnekleri sunar.
"""

import os
from pathlib import Path
from csv_grafikleyici import CSVGrafikleyici
import pandas as pd

def klasordeki_csv_dosyalari_listele(klasor_yolu: str = ".") -> list:
    """
    Belirtilen klasördeki tüm CSV dosyalarını listeler
    
    Args:
        klasor_yolu (str): Aranacak klasör yolu
        
    Returns:
        list: CSV dosya yollarının listesi
    """
    klasor = Path(klasor_yolu)
    csv_dosyalari = list(klasor.glob("*.csv"))
    
    print(f"📁 {klasor.absolute()} klasöründeki CSV dosyaları:")
    print("-" * 60)
    
    if not csv_dosyalari:
        print("❌ CSV dosyası bulunamadı!")
        return []
    
    for i, dosya in enumerate(csv_dosyalari, 1):
        try:
            # Dosya boyutunu al
            boyut = dosya.stat().st_size
            boyut_mb = boyut / (1024 * 1024)
            
            # Satır sayısını tahmin et (ilk birkaç satırı okuyarak)
            try:
                df_sample = pd.read_csv(dosya, nrows=5)
                sutun_sayisi = len(df_sample.columns)
                
                # Toplam satır sayısını tahmin et
                with open(dosya, 'r', encoding='utf-8') as f:
                    satir_sayisi = sum(1 for _ in f) - 1  # Header hariç
                    
            except Exception:
                sutun_sayisi = "?"
                satir_sayisi = "?"
            
            print(f"{i:2}. {dosya.name:40} | {boyut_mb:6.2f} MB | {satir_sayisi:>8} satır | {sutun_sayisi:>3} sütun")
            
        except Exception as e:
            print(f"{i:2}. {dosya.name:40} | Hata: {e}")
    
    return csv_dosyalari

def csv_dosyasi_sec(csv_dosyalari: list) -> Path:
    """
    Kullanıcıdan CSV dosyası seçmesini ister
    
    Args:
        csv_dosyalari (list): Seçilebilir CSV dosyaları
        
    Returns:
        Path: Seçilen CSV dosyası yolu
    """
    if not csv_dosyalari:
        return None
    
    print(f"\n🎯 Hangi CSV dosyasını işlemek istiyorsunuz?")
    print("Dosya numarasını girin (1-{}) veya 'q' ile çıkın: ".format(len(csv_dosyalari)), end="")
    
    while True:
        secim = input().strip().lower()
        
        if secim == 'q':
            return None
        
        try:
            secim_num = int(secim)
            if 1 <= secim_num <= len(csv_dosyalari):
                return csv_dosyalari[secim_num - 1]
            else:
                print(f"❌ Geçersiz numara! 1-{len(csv_dosyalari)} arası bir sayı girin: ", end="")
        except ValueError:
            print("❌ Geçersiz giriş! Sayı girin veya 'q' ile çıkın: ", end="")

def interaktif_grafik_olustur(grafikleyici: CSVGrafikleyici):
    """
    Kullanıcıyla interaktif olarak grafik oluşturma seçenekleri sunar
    
    Args:
        grafikleyici (CSVGrafikleyici): Grafik oluşturucu nesne
    """
    while True:
        print(f"\n🎨 Grafik Oluşturma Seçenekleri:")
        print("1. Tüm sayısal sütunlar için grafik oluştur")
        print("2. Belirli bir sütun için grafik oluştur")
        print("3. Sütun listesini görüntüle")
        print("4. Özet raporu göster")
        print("5. Çıkış")
        
        secim = input("\nSeçiminizi yapın (1-5): ").strip()
        
        if secim == '1':
            print("\n🚀 Tüm sayısal sütunlar için grafik oluşturuluyor...")
            max_sutun = input("Maksimum kaç sütun işlensin? (varsayılan: 50): ").strip()
            try:
                max_sutun = int(max_sutun) if max_sutun else 50
            except ValueError:
                max_sutun = 50
            
            sonuclar = grafikleyici.tum_sutunlar_grafik(max_sutun=max_sutun)
            
            basarili = sum(sonuclar.values())
            toplam = len(sonuclar)
            print(f"\n✅ İşlem tamamlandı! {basarili}/{toplam} grafik oluşturuldu.")
            
        elif secim == '2':
            print("\n📊 Mevcut sütunlar:")
            sayisal_sutunlar = []
            for i, (sutun, bilgi) in enumerate(grafikleyici.sutun_bilgileri.items(), 1):
                if bilgi['sayisal']:
                    sayisal_sutunlar.append(sutun)
                    print(f"{i:2}. {sutun}")
            
            if not sayisal_sutunlar:
                print("❌ Sayısal sütun bulunamadı!")
                continue
            
            sutun_secim = input(f"\nSütun numarası (1-{len(sayisal_sutunlar)}) veya sütun adı: ").strip()
            
            try:
                if sutun_secim.isdigit():
                    sutun_idx = int(sutun_secim) - 1
                    if 0 <= sutun_idx < len(sayisal_sutunlar):
                        secilen_sutun = sayisal_sutunlar[sutun_idx]
                    else:
                        print("❌ Geçersiz numara!")
                        continue
                else:
                    secilen_sutun = sutun_secim
                    if secilen_sutun not in sayisal_sutunlar:
                        print("❌ Sütun bulunamadı!")
                        continue
                
                print(f"📈 {secilen_sutun} sütunu için grafik oluşturuluyor...")
                basarili = grafikleyici.tek_sutun_grafik(secilen_sutun, kaydet=True, goster=False)
                
                if basarili:
                    print("✅ Grafik başarıyla oluşturuldu!")
                else:
                    print("❌ Grafik oluşturulamadı!")
                    
            except Exception as e:
                print(f"❌ Hata: {e}")
        
        elif secim == '3':
            print("\n📋 Sütun Listesi:")
            print("-" * 80)
            for sutun, bilgi in grafikleyici.sutun_bilgileri.items():
                durum = "✓ Sayısal" if bilgi['sayisal'] else "○ Sayısal değil"
                print(f"{sutun:30} | {bilgi['veri_tipi']:15} | {durum}")
        
        elif secim == '4':
            print(grafikleyici.ozet_rapor())
        
        elif secim == '5':
            print("👋 Çıkılıyor...")
            break
        
        else:
            print("❌ Geçersiz seçim! 1-5 arası bir sayı girin.")

def main():
    """Ana demo fonksiyonu"""
    print("🎯 CSV Grafikleyici Demo")
    print("=" * 50)
    
    # Mevcut klasördeki CSV dosyalarını listele
    csv_dosyalari = klasordeki_csv_dosyalari_listele(".")
    
    if not csv_dosyalari:
        print("\n❌ İşlenecek CSV dosyası bulunamadı!")
        return
    
    # Kullanıcıdan dosya seçmesini iste
    secilen_dosya = csv_dosyasi_sec(csv_dosyalari)
    
    if not secilen_dosya:
        print("👋 İşlem iptal edildi.")
        return
    
    print(f"\n🎯 Seçilen dosya: {secilen_dosya.name}")
    print("=" * 50)
    
    # Grafikleyici oluştur
    grafikleyici = CSVGrafikleyici(str(secilen_dosya))
    
    # CSV dosyasını yükle
    if not grafikleyici.csv_yukle():
        print("❌ CSV dosyası yüklenemedi!")
        return
    
    # Özet raporu göster
    print(grafikleyici.ozet_rapor())
    
    # İnteraktif grafik oluşturma
    interaktif_grafik_olustur(grafikleyici)

if __name__ == "__main__":
    main()