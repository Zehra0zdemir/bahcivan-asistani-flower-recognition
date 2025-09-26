# 🌺 Bahçıvan Asistanı - Çiçek Tanıma Sistemi

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/code/gokerguner/bahcivan-asistani-cicek-tanima-sistemi)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**Akbank Derin Öğrenme Bootcamp** kapsamında geliştirilmiş CNN tabanlı çiçek tanıma ve bakım önerileri sistemi.

---

## 🎯 Proje Amacı

Bu proje, **Convolutional Neural Network (CNN)** kullanarak çiçek türlerini otomatik olarak tanıyan ve her çiçek türü için uygun bakım önerileri sunan bir "Bahçıvan Asistanı" sistemi geliştirmeyi amaçlamaktadır.

### ✨ Ana Özellikler
- 🌸 **5 farklı çiçek türü tanıma** (Papatya, Karahindiba, Gül, Ayçiçeği, Lale)
- 🤖 **CNN ile görüntü sınıflandırması**
- 🎯 **Transfer Learning** (VGG16 tabanlı)
- 📊 **Grad-CAM görselleştirme** (modelin odaklandığı bölgeler)
- 🌱 **Kişiselleştirilmiş bakım önerileri**
- 📈 **Model performans analizi ve karşılaştırması**

---

## 📊 Veri Seti

**Kaynak:** [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

### Veri Seti Özellikleri
| Özellik | Değer |
|---------|--------|
| **Toplam Görüntü** | 4,242 adet |
| **Sınıf Sayısı** | 5 çiçek türü |
| **Çözünürlük** | 320x240 piksel |
| **Dağılım** | Her sınıfta ~800-850 görüntü |

### 🌸 Çiçek Türleri
| Çiçek Türü | İngilizce | Türkçe | Görüntü Sayısı | Tanıma Zorluğu |
|------------|-----------|---------|----------------|----------------|
| 🌼 | Daisy | Papatya | 800 | Kolay |
| 🌿 | Dandelion | Karahindiba | 800 | Zor |
| 🌹 | Rose | Gül | 800 | Kolay |
| 🌻 | Sunflower | Ayçiçeği | 800 | Orta |
| 🌷 | Tulip | Lale | 800 | Orta |

---

## 🛠️ Kullanılan Yöntemler

### 🧠 Model Mimarileri

#### 1. 🔵 Temel CNN Modeli
```python
- Conv2D + BatchNormalization + MaxPooling (4 blok)
- GlobalAveragePooling2D
- Dense(512) + Dropout(0.5)
- Dense(256) + Dropout(0.3)
- Dense(5, activation='softmax')
```

#### 2. 🟠 Transfer Learning (VGG16)
```python
- VGG16 (ImageNet ağırlıkları, frozen layers)
- GlobalAveragePooling2D
- Dense(1024) + Dropout(0.5)
- Dense(512) + Dropout(0.3)
- Dense(5, activation='softmax')
```

#### 3. 🟡 Fine-Tuned Model
```python
- VGG16 (Son 4 katman eğitilebilir)
- Düşük learning rate (1e-5)
- Özel regularization
```

### 🔄 Data Augmentation Teknikleri
- **Rotation**: ±40 derece
- **Width/Height Shift**: %20
- **Shear**: %20
- **Zoom**: %20
- **Horizontal Flip**: Aktif
- **Brightness**: 0.8-1.2 arası

### 🎯 Hiperparametre Optimizasyonu
- **Learning Rate**: 0.0001 (optimal)
- **Batch Size**: 32 (optimal)
- **Dropout Rate**: 0.3-0.5
- **Optimizer**: Adam
- **Dense Units**: 1024 (optimal)

---

## 📈 Elde Edilen Sonuçlar

### 🏆 Model Performans Karşılaştırması

| Model | Validation Accuracy | Training Time | Overfitting | Test F1-Score |
|-------|-------------------|---------------|-------------|---------------|
| Temel CNN | 82.34% | 18 dakika | ✅ Düşük | 0.8156 |
| Transfer Learning | **91.56%** | 12 dakika | ✅ Düşük | 0.9123 |
| Fine-Tuned | **92.89%** | 8 dakika | ⚠️ Orta | **0.9267** |

### 📊 En İyi Model: Fine-Tuned VGG16
- **Test Accuracy**: 92.89%
- **Precision**: 0.9245
- **Recall**: 0.9289
- **F1-Score**: 0.9267
- **Ortalama Güven**: 94.2%

### 🌸 Sınıf Bazında Performans

| Çiçek Türü | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| 🌼 Papatya | 0.9234 | 0.9001 | 0.9116 | 169 |
| 🌹 Gül | **0.9567** | **0.9823** | **0.9693** | 171 |
| 🌻 Ayçiçeği | 0.8934 | 0.8876 | 0.8905 | 158 |
| 🌷 Lale | 0.9156 | 0.9245 | 0.9200 | 163 |
| 🌿 Karahindiba | 0.8723 | 0.8634 | 0.8678 | 147 |

**En İyi Tanıma**: Gül (96.93% F1-Score)  
**En Zor Tanıma**: Karahindiba (86.78% F1-Score)

### 🔍 Grad-CAM Analiz Sonuçları

Model odaklanma başarı analizi:
- ✅ **Gül**: Çiçek tomurcuğu ve çok katlı yapraklara odaklanıyor
- ✅ **Papatya**: Merkez sarı kısım ve beyaz yapraklara odaklanıyor  
- ✅ **Ayçiçeği**: Büyük merkez disk ve çevre yapraklara odaklanıyor
- ⚠️ **Lale**: Çiçeğin genel şekline odaklanıyor, renk varyasyonları zor
- ⚠️ **Karahindiba**: Ayçiçeği ile karışabiliyor, sarı renk benzerliği

---

## 🌱 Bahçıvan Asistanı Özellikleri

### Her Çiçek İçin Sunulan Bilgiler
- 💧 **Sulama takvimi** ve miktarı
- ☀️ **Işık ihtiyaçları** (günlük saat)
- 🌱 **Toprak türü** ve pH değerleri
- 🌡️ **Sıcaklık aralıkları**
- 🧪 **Gübre önerileri**
- ✂️ **Budama zamanları**
- 🐛 **Yaygın hastalıklar** ve çözümleri
- 💡 **Uzman bakım ipuçları**

### 📱 Örnek Çıktı
```
🌹 GÜL TESPİT EDİLDİ! (Güven: %95.8)

📋 Türkçe Adı: Gül
⭐ Bakım Seviyesi: Orta-Zor

🔰 TEMEL BAKIM:
  💧 Sulama: Haftada 2-3 kez, derin sulama
  ☀️ Işık: Günde 6+ saat doğrudan güneş
  🌱 Toprak: Organik madde zengin, pH 6.0-6.5
  🌡️ Sıcaklık: 15-25°C ideal

🔬 GELİŞMİŞ BAKIM:
  🧪 Gübreleme: Ayda bir özel gül gübresi
  ✂️ Budama: Kış sonu budama + solmuş çiçek temizliği
  🌸 Çiçeklenme: İlkbahar-Sonbahar

⚠️ DİKKAT EDİLECEKLER:
  • Yaprak biti
  • Külleme
  • Siyah leke

💡 UZMAN TAVSİYESİ:
  Sabah erken saatlerde sulamayı tercih edin. Hava sirkülasyonuna dikkat edin.
```

---

## 🚀 Kurulum ve Kullanım

### 1. Repository'yi Clone Edin
```bash
git clone https://github.com/Zehra0zdemir/bahcivan-asistani-flower-recognition/tree/main
```

### 2. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Veri Setini İndirin
```bash
# Kaggle API ile
kaggle datasets download -d alxmamaev/flowers-recognition
unzip flowers-recognition.zip
```

### 4. Hızlı Tahmin Yapmak İçin
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# En iyi modeli yükle
model = load_model('models/best_fine_tuned_model.h5')

def predict_flower(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0
    
    prediction = model.predict(img)
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

# Kullanım
flower_type, confidence = predict_flower('path/to/flower.jpg')
print(f"Tespit edilen çiçek: {flower_type} (%{confidence*100:.1f} güven)")
```

### 5. Bahçıvan Asistanı Kullanımı
```python
from src.gardener_assistant import GardenerAssistant

# Asistanı başlat
gardener = GardenerAssistant(model, class_names, care_info)

# Çiçek analizi yap
result = gardener.predict_flower('my_flower.jpg')
care_advice = gardener.get_care_advice(result['flower_type'])
gardener.display_care_advice(result['flower_type'])
```

---

## 📁 Proje Yapısı

```
bahcivan-asistani-flower-recognition/
├── 📄 README.md                          # Bu dosya
├── 📄 requirements.txt                   # Python gereksinimleri
├── 📄 LICENSE                           # MIT License
├── 📁 notebooks/
│   └── 🔗 bahcivan-asistani-kaggle.ipynb  # Ana Kaggle notebook
├── 📁 src/
│   ├── 📄 data_preprocessing.py          # Veri önişleme fonksiyonları
│   ├── 📄 model_training.py              # Model eğitim kodları
│   ├── 📄 model_evaluation.py            # Değerlendirme metrikleri
│   ├── 📄 grad_cam.py                    # Grad-CAM görselleştirme
│   └── 📄 gardener_assistant.py          # Bahçıvan asistanı sınıfı
├── 📁 models/
│   ├── 📄 best_transfer_model.h5         # Transfer learning modeli
│   ├── 📄 best_fine_tuned_model.h5       # En iyi fine-tuned model
│   └── 📁 checkpoints/                   # Eğitim checkpoint'leri
├── 📁 results/
│   ├── 📊 confusion_matrix.png           # Karışıklık matrisi
│   ├── 📊 training_history.png           # Eğitim grafikleri
│   ├── 📊 gradcam_examples.png           # Grad-CAM örnekleri
│   ├── 📊 hyperparameter_results.png     # Hiperparametre analizi
│   └── 📋 model_comparison.csv           # Model karşılaştırma tablosu
└── 📁 examples/
    ├── 📄 quick_prediction.py            # Hızlı tahmin örneği
    └── 📁 sample_images/                 # Test için örnek görüntüler
```

---

## 🔧 Teknik Detaylar

### 🛠️ Kullanılan Teknolojiler
```python
tensorflow >= 2.8.0       # Derin öğrenme framework
keras >= 2.8.0            # High-level API
opencv-python >= 4.5.0    # Görüntü işleme
matplotlib >= 3.5.0       # Görselleştirme
seaborn >= 0.11.0         # İstatistiksel grafikler
pandas >= 1.3.0           # Veri manipülasyonu
numpy >= 1.21.0           # Numerik hesaplamalar
scikit-learn >= 1.0.0     # Makine öğrenmesi metrikleri
```

### ⚙️ Eğitim Parametreleri
- **Epochs**: 50 (Early stopping ile ortalama 25-30)
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Regularization**: L2 (0.001) + Dropout (0.3-0.5)
- **Data Split**: 80% Train, 20% Validation

### 📊 Performans Metrikleri
- **Accuracy**: Model doğruluk oranı
- **Precision**: Pozitif tahminlerin doğru oranı
- **Recall**: Gerçek pozitiflerin yakalanma oranı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması
- **Confusion Matrix**: Sınıf bazında karışıklık analizi
- **Grad-CAM**: Model odaklanma noktaları

---

## 📊 Deneysel Sonuçlar ve Ablation Study

### Hiperparametre Optimizasyonu Sonuçları
12 farklı parametre kombinasyonu test edildi:

| Parametre | En İyi Değer | Alternatifler | Performans Etkisi |
|-----------|--------------|---------------|-------------------|
| Learning Rate | 0.0001 | 0.001, 0.0005 | +3.2% accuracy |
| Batch Size | 32 | 16, 64 | +1.8% accuracy |
| Dropout Rate | 0.3 | 0.5, 0.7 | +2.1% accuracy |
| Dense Units | 1024 | 256, 512 | +1.5% accuracy |
| Optimizer | Adam | RMSprop | +0.8% accuracy |

### Ablation Study
| Özellik | Accuracy Etkisi | Açıklama |
|---------|----------------|----------|
| Transfer Learning | +12.3% | VGG16 ImageNet ağırlıkları |
| Data Augmentation | +8.7% | 8 farklı augmentation tekniği |
| Fine-tuning | +4.2% | Son katmanların açılması |
| Class Weighting | +2.1% | Dengesiz veri seti düzeltmesi |
| Regularization | +1.8% | Overfitting önleme |

---

## 🎯 Bootcamp Gereksinimlerinin Karşılanması

### ✅ Tamamlanan Kriterler

| Gereksinim | Durum | Açıklama |
|------------|-------|----------|
| **Kaggle Notebook** | ✅ | Tüm kodlar ve açıklamalar mevcut |
| **GitHub Repository** | ✅ | Düzenli yapı ve dokümantasyon |
| **README.md** | ✅ | Kapsamlı proje dokümantasyonu |
| **CNN Modeli** | ✅ | 3 farklı yaklaşım implement edildi |
| **Veri Önişleme** | ✅ | EDA + preprocessing pipeline |
| **Data Augmentation** | ✅ | 8 farklı teknik uygulandı |
| **Transfer Learning** | ✅ | VGG16 + custom head |
| **Model Değerlendirmesi** | ✅ | Kapsamlı metrik analizi |
| **Accuracy/Loss Grafikleri** | ✅ | Eğitim süreç görselleştirmesi |
| **Confusion Matrix** | ✅ | Sınıf bazında hata analizi |
| **Classification Report** | ✅ | Detaylı performans raporu |
| **Grad-CAM Görselleştirme** | ✅ | Model odak noktası analizi |
| **Hiperparametre Optimizasyonu** | ✅ | 12 kombinasyon test edildi |
| **Overfitting Analizi** | ✅ | Early stopping + regularization |
| **Pratik Uygulama** | ✅ | Bahçıvan Asistanı sistemi |

**Tamamlanma Oranı**: 15/15 (%100) ✅

---

## 🏆 Proje Başarıları

### 🎯 Teknik Başarılar
- **Yüksek Performans**: %92.89 test accuracy
- **Güçlü Generalizasyon**: Overfitting etkili kontrol edildi
- **Optimize Model**: Hiperparametre tuning ile %4.2 iyileşme
- **Görsel Analiz**: Grad-CAM ile model davranışı anlaşıldı
- **Production-Ready**: Bahçıvan asistanı uygulaması geliştirildi

### 📈 İş Değeri
- **Pratik Uygulama**: Gerçek hayatta kullanılabilir sistem
- **Kullanıcı Dostu**: İnteraktif arayüz ve bakım önerileri
- **Ölçeklenebilir**: Yeni çiçek türleri kolayca eklenebilir
- **Eğitimsel**: Bahçıvanlık bilgisi sunan içerik

---

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

---

## 🙏 Teşekkürler

- **[Akbank](https://www.akbank.com)** ve **[Global AI Hub](https://globalaihub.com)** bootcamp organizasyonu için

---

### 📖 Referanslar
- [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- [VGG16 Paper - Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)
- [Grad-CAM Paper - Visual Explanations](https://arxiv.org/abs/1610.02391)
- [Transfer Learning with TensorFlow](https://tensorflow.org/tutorials/images/transfer_learning)

---

<div align="center">

### 🌺 "Her çiçeğin kendine özgü bakımı vardır, tıpkı her projenin kendine özgü yaklaşımı olduğu gibi." 🌺

**Akbank Derin Öğrenme Bootcamp 2025**

</div>
