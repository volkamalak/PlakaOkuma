# Plaka Tanıma Sistemi (License Plate Recognition System)

Bu proje, Python kullanılarak geliştirilmiş, görüntü işleme ve yapay zeka tabanlı bir plaka tanıma sistemidir. Sistem, tır veya kamyon gibi araçların plakalarını tespit etmek ve okumak için tasarlanmıştır.

## Proje Yapısı

- `main.py`: Uygulamanın ana giriş noktasıdır. Resim üzerindeki işlemi bu dosya üzerinden başlatırsınız.
- `src/detector.py`: YOLOv8 kullanarak araç veya plaka tespiti yapan modüldür.
- `src/ocr.py`: EasyOCR kullanarak tespit edilen alandaki yazıyı (plakayı) okuyan modüldür.
- `src/train.py`: Kendi veri setinizle özel bir YOLO modeli eğitmek için kullanılan betiktir.
- `requirements.txt`: Projenin çalışması için gerekli kütüphaneleri listeler.

## Kurulum

Projeyi çalıştırmadan önce gerekli kütüphaneleri yüklemeniz gerekmektedir:

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Resim Üzerinde Plaka Tanıma

Sistemi bir resim üzerinde çalıştırmak için `main.py` dosyasını kullanabilirsiniz:

```bash
python main.py --image "resim_yolu/ornek_tir.jpg"
```

Sonuçlar `data/output` klasörüne kaydedilecektir.

İsteğe bağlı parametreler:
- `--model`: Kullanılacak YOLO modelinin yolu (varsayılan: `yolov8n.pt`).
- `--output`: Sonuçların kaydedileceği klasör (varsayılan: `data/output`).

Örnek:
```bash
python main.py --image "test.jpg" --model "best.pt"
```

### 2. Model Eğitimi

Kendi plaka veri setinizle daha hassas bir model eğitmek isterseniz `src/train.py` içindeki fonksiyonu kullanabilirsiniz.

Eğitim için verilerinizin `data.yaml` formatında hazır olması gerekir.

```python
from src.train import train_model

# Eğitimi başlat
train_model('veri_seti/data.yaml', epochs=100)
```

## Nasıl Çalışır?

1. **Tespit (Detection):** `PlateDetector` sınıfı, YOLOv8 modelini kullanarak resimdeki ilgili nesneleri (başlangıçta araçları, eğitildiğinde plakaları) bulur ve koordinatlarını (Bounding Box) çıkarır.
2. **Kırpma (Cropping):** Tespit edilen alan orijinal resimden kırpılır.
3. **Okuma (OCR):** `PlateReader` sınıfı, kırpılan bu görüntüyü alır. Önce gri tonlamaya çevirerek işler, ardından EasyOCR kütüphanesi ile üzerindeki yazıyı okur.
4. **Sonuç:** Okunan plaka metni ve çizilen kutucuk resim üzerine işlenir ve kaydedilir.
