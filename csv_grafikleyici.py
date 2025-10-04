"""
CSV Grafikleyici - CSV dosyalarından değer-zaman grafikleri oluşturan sistem
Bu modül, belirtilen CSV dosyasının her sütunu için değer-zaman grafiklerini çıkarır.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings

# Matplotlib Türkçe karakter desteği
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

class CSVGrafikleyici:
    """CSV dosyalarından değer-zaman grafikleri oluşturan sınıf"""
    
    def __init__(self, csv_dosya_yolu: str):
        """
        CSV Grafikleyici başlatıcı
        
        Args:
            csv_dosya_yolu (str): İşlenecek CSV dosyasının yolu
        """
        self.csv_dosya_yolu = Path(csv_dosya_yolu)
        self.veri = None
        self.sutun_bilgileri = {}
        self.zaman_sutunu = None
        
        # Çıktı klasörü oluştur
        self.cikti_klasoru = self.csv_dosya_yolu.parent / f"grafikler_{self.csv_dosya_yolu.stem}"
        self.cikti_klasoru.mkdir(exist_ok=True)
        
    def csv_yukle(self) -> bool:
        """
        CSV dosyasını yükler ve analiz eder
        
        Returns:
            bool: Yükleme başarılı ise True
        """
        try:
            print(f"CSV dosyası yükleniyor: {self.csv_dosya_yolu}")
            self.veri = pd.read_csv(self.csv_dosya_yolu)
            print(f"✓ {len(self.veri)} satır, {len(self.veri.columns)} sütun yüklendi")
            
            # Sütun bilgilerini analiz et
            self._sutun_analizi()
            
            # Zaman sütununu tespit et
            self._zaman_sutunu_tespit()
            
            return True
            
        except Exception as e:
            print(f"❌ CSV yükleme hatası: {e}")
            return False
    
    def _sutun_analizi(self):
        """Sütunları analiz eder ve veri tiplerini belirler"""
        print("\n📊 Sütun Analizi:")
        print("-" * 50)
        
        for sutun in self.veri.columns:
            veri_tipi = self.veri[sutun].dtype
            bos_deger_sayisi = self.veri[sutun].isnull().sum()
            benzersiz_deger_sayisi = self.veri[sutun].nunique()
            
            # Sayısal sütun mu kontrol et
            sayisal_mi = pd.api.types.is_numeric_dtype(self.veri[sutun])
            
            self.sutun_bilgileri[sutun] = {
                'veri_tipi': str(veri_tipi),
                'sayisal': sayisal_mi,
                'bos_deger': bos_deger_sayisi,
                'benzersiz_deger': benzersiz_deger_sayisi,
                'min_deger': self.veri[sutun].min() if sayisal_mi else None,
                'max_deger': self.veri[sutun].max() if sayisal_mi else None,
                'ortalama': self.veri[sutun].mean() if sayisal_mi else None
            }
            
            print(f"{sutun:30} | {str(veri_tipi):15} | Sayısal: {sayisal_mi:5} | Boş: {bos_deger_sayisi:6} | Benzersiz: {benzersiz_deger_sayisi}")
    
    def _zaman_sutunu_tespit(self):
        """Zaman sütununu otomatik tespit eder"""
        zaman_anahtar_kelimeler = ['time', 'zaman', 'tarih', 'date', 'timestamp', 'epoch', 'cadenceno']
        
        for sutun in self.veri.columns:
            sutun_kucuk = sutun.lower()
            if any(anahtar in sutun_kucuk for anahtar in zaman_anahtar_kelimeler):
                if pd.api.types.is_numeric_dtype(self.veri[sutun]):
                    self.zaman_sutunu = sutun
                    print(f"\n⏰ Zaman sütunu tespit edildi: {sutun}")
                    break
        
        if not self.zaman_sutunu:
            # İlk sayısal sütunu zaman olarak kullan
            for sutun in self.veri.columns:
                if pd.api.types.is_numeric_dtype(self.veri[sutun]):
                    self.zaman_sutunu = sutun
                    print(f"\n⏰ Varsayılan zaman sütunu: {sutun}")
                    break
    
    def tek_sutun_grafik(self, sutun_adi: str, kaydet: bool = True, goster: bool = False) -> Optional[plt.Figure]:
        """
        Belirtilen sütun için değer-zaman grafiği oluşturur
        
        Args:
            sutun_adi (str): Grafik oluşturulacak sütun adı
            kaydet (bool): Grafiği kaydet
            goster (bool): Grafiği göster
            
        Returns:
            plt.Figure: Oluşturulan grafik figürü
        """
        if sutun_adi not in self.veri.columns:
            print(f"❌ Sütun bulunamadı: {sutun_adi}")
            return None
            
        if not self.sutun_bilgileri[sutun_adi]['sayisal']:
            print(f"⚠️  Sayısal olmayan sütun atlandı: {sutun_adi}")
            return None
        
        try:
            # Grafik oluştur
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Zaman ekseni
            if self.zaman_sutunu and self.zaman_sutunu != sutun_adi:
                x_veri = self.veri[self.zaman_sutunu]
                x_label = self.zaman_sutunu
            else:
                x_veri = range(len(self.veri))
                x_label = "İndeks"
            
            y_veri = self.veri[sutun_adi]
            
            # Çizgi grafiği
            ax.plot(x_veri, y_veri, linewidth=1, alpha=0.8, color='steelblue')
            
            # Grafik düzenlemeleri
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(sutun_adi, fontsize=12)
            ax.set_title(f'{sutun_adi} - Değer-Zaman Grafiği', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # İstatistikler ekle
            stats_text = f"""İstatistikler:
Min: {y_veri.min():.4f}
Max: {y_veri.max():.4f}
Ortalama: {y_veri.mean():.4f}
Std: {y_veri.std():.4f}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Kaydet
            if kaydet:
                temiz_sutun_adi = sutun_adi.replace('/', '_').replace('\\', '_')
                dosya_adi = f"{temiz_sutun_adi}_grafik.png"
                kayit_yolu = self.cikti_klasoru / dosya_adi
                plt.savefig(kayit_yolu, dpi=300, bbox_inches='tight')
                print(f"✓ Grafik kaydedildi: {kayit_yolu}")
            
            # Göster
            if goster:
                plt.show()
            else:
                plt.close()
                
            return fig
            
        except Exception as e:
            print(f"❌ Grafik oluşturma hatası ({sutun_adi}): {e}")
            return None
    
    def tum_sutunlar_grafik(self, sadece_sayisal: bool = True, max_sutun: int = 50) -> Dict[str, bool]:
        """
        Tüm sütunlar için grafik oluşturur
        
        Args:
            sadece_sayisal (bool): Sadece sayısal sütunları işle
            max_sutun (int): Maksimum işlenecek sütun sayısı
            
        Returns:
            Dict[str, bool]: Sütun adı ve başarı durumu
        """
        print(f"\n🎨 Tüm sütunlar için grafik oluşturuluyor...")
        print(f"Çıktı klasörü: {self.cikti_klasoru}")
        print("-" * 60)
        
        sonuclar = {}
        islenen_sutun = 0
        
        for sutun in self.veri.columns:
            if islenen_sutun >= max_sutun:
                print(f"⚠️  Maksimum sütun sayısına ulaşıldı ({max_sutun})")
                break
                
            if sadece_sayisal and not self.sutun_bilgileri[sutun]['sayisal']:
                sonuclar[sutun] = False
                continue
            
            print(f"📈 İşleniyor: {sutun}")
            basarili = self.tek_sutun_grafik(sutun, kaydet=True, goster=False) is not None
            sonuclar[sutun] = basarili
            
            if basarili:
                islenen_sutun += 1
        
        # Özet rapor
        basarili_sayisi = sum(sonuclar.values())
        toplam_sutun = len(sonuclar)
        
        print(f"\n📊 Grafik Oluşturma Özeti:")
        print(f"Toplam sütun: {toplam_sutun}")
        print(f"Başarılı: {basarili_sayisi}")
        print(f"Başarısız: {toplam_sutun - basarili_sayisi}")
        print(f"Çıktı klasörü: {self.cikti_klasoru}")
        
        return sonuclar
    
    def ozet_rapor(self) -> str:
        """
        CSV dosyası ve işlem özeti raporu oluşturur
        
        Returns:
            str: Rapor metni
        """
        if self.veri is None:
            return "❌ Veri yüklenmemiş"
        
        rapor = f"""
📋 CSV Grafik Analiz Raporu
{'='*50}

📁 Dosya: {self.csv_dosya_yolu.name}
📊 Boyut: {len(self.veri)} satır × {len(self.veri.columns)} sütun
⏰ Zaman sütunu: {self.zaman_sutunu or 'Tespit edilemedi'}
📂 Çıktı klasörü: {self.cikti_klasoru}

🔢 Sütun Detayları:
{'-'*50}
"""
        
        sayisal_sutunlar = 0
        for sutun, bilgi in self.sutun_bilgileri.items():
            if bilgi['sayisal']:
                sayisal_sutunlar += 1
                min_val = bilgi['min_deger'] if bilgi['min_deger'] is not None else 0
                max_val = bilgi['max_deger'] if bilgi['max_deger'] is not None else 0
                rapor += f"✓ {sutun:30} | {bilgi['veri_tipi']:15} | Min: {min_val:10.4f} | Max: {max_val:10.4f}\n"
            else:
                rapor += f"○ {sutun:30} | {bilgi['veri_tipi']:15} | (Sayısal değil)\n"
        
        rapor += f"\n📈 Grafik Oluşturulabilir Sütun Sayısı: {sayisal_sutunlar}"
        
        return rapor


def main():
    """Ana fonksiyon - Komut satırından kullanım için"""
    import sys
    
    if len(sys.argv) < 2:
        print("Kullanım: python csv_grafikleyici.py <csv_dosya_yolu>")
        print("Örnek: python csv_grafikleyici.py data.csv")
        return
    
    csv_dosya = sys.argv[1]
    
    if not os.path.exists(csv_dosya):
        print(f"❌ Dosya bulunamadı: {csv_dosya}")
        return
    
    # Grafikleyici oluştur
    grafikleyici = CSVGrafikleyici(csv_dosya)
    
    # CSV yükle
    if not grafikleyici.csv_yukle():
        return
    
    # Özet rapor göster
    print(grafikleyici.ozet_rapor())
    
    # Kullanıcıdan onay al
    cevap = input("\nTüm sayısal sütunlar için grafik oluşturulsun mu? (e/h): ").lower()
    
    if cevap in ['e', 'evet', 'y', 'yes']:
        grafikleyici.tum_sutunlar_grafik()
    else:
        print("İşlem iptal edildi.")


if __name__ == "__main__":
    main()