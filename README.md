<div align="center">
  
# 🧠 EEG Analiz ve Yönetim Sistemi

Bu proje, derin öğrenme algoritmaları kullanarak ham **EEG (Elektroensefalografi)** sinyallerini analiz eden, görselleştiren ve tıbbi/akademik yorumlamalar sunan yenilikçi, profesyonel bir web platformudur.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blueviolet?style=for-the-badge&logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)]()
[![Flask](https://img.shields.io/badge/Flask-3.1.2-000000?style=for-the-badge&logo=flask&logoColor=white)]()
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)]()

</div>

<br>

<p align="center">
  <img src="https://via.placeholder.com/900x450?text=EEG+Analiz+Dashboard+ve+Analiz+Grafikleri" alt="EEG Analiz Sistemi Ekran Görüntüsü">
</p>

## 🚀 Proje Hakkında

Bu sistem, nörolojik verilerin hızlı bir şekilde yorumlanmasına yardımcı olmak için tasarlanmıştır. Kullanıcıların `.txt` formatındaki zaman aralıklı EEG veri dosyalarını sisteme yüklemesine olanak tanır. Yüklenen sinyaller arka planda **ön işlemeden (Z-score normalizasyonu, hedef uzunluk kırpma/doldurma)** geçirilir ve önceden eğitilmiş bir **Keras** modeline beslenerek 5 farklı kategoriden birine sınıflandırılır:

1. **`Z` Sınıfı:** Anormal (Gözler Kapalı)
2. **`O` Sınıfı:** Normal (Gözler Kapalı)
3. **`N` Sınıfı:** Anormal (Gözler Açık)
4. **`F` Sınıfı:** Anormal (Fokal Odaklı - Epileptik Eğilim)
5. **`S` Sınıfı:** Anormal (Nöbet Anı - Aktif Epilepsi)

## ✨ Öne Çıkan Özellikler

- **Gelişmiş Veri Görselleştirme:** Ham sinyalin *Zaman Domeni* (Time Domain) grafiği ve Welch metodu ile hesaplanmış *Güç Spektral Yoğunluğu (PSD - Power Spectral Density)* frekans grafiği Chart.js üzerinden akıcı bir şekilde sunulur.
- **Kişiselleştirilmiş Karşılaştırma Modülü:** İki farklı EEG analiz kaydını aynı grafik üzerinde frekans gücü (Delta, Theta, Alfa, Beta) ve dalga genliği üzerinden sayısal istatistiklerle kıyaslama imkanı.
- **Tek Tıkla PDF Raporlama:** Analiz sonuçlarını ve kıyaslamaları saniyeler içerisinde modern stiliyle (Karanlık Tema, PDF formatı) dışarı aktarabilme. Yüksek çözünürlüklü Canvas -> PNG yöntemiyle mükemmel netlik.
- **Yapay Zeka Destekli Chatbot:** Kullanıcıların sistemin işleyişi hakkında (örn: *PSD nedir?*) hızlıca bilgi alabileceği doğal dil destekli yardımcı asistan.
- **Gelişmiş Kullanıcı Yönetimi:** Werkzeug `scrypt` hashing mimarisi ile korunan üyelik sistemi, kullanıcı özelinde gizli analiz geçmişi (`userData`).
- **Esnek Hata Raporlama ve Backend İletişimi:** Otomatik JS hatalarını takip mekanizması (Global error catcher).

## 🛠️ Teknoloji Yığını (Tech Stack)

### **Backend:**
- **Python:** Çekirdek programlama dili.
- **Flask:** Web yönlendirmeleri, oturum yönetimi ve API servisleri.
- **TensorFlow / Keras:** Derin öğrenme iş akışları, sınıflandırma modellerinin çalıştırılması.
- **NumPy & SciPy (Welch İşlemcisi):** Gelişmiş matematiksel sinyal işleme matriksleri.
- **Werkzeug:** Şifreleme algoritmaları (Güvenlik).

### **Frontend:**
- **HTML5 & Jinja2:** Sunucu taraflı şablonlama motoru.
- **Tailwind CSS v3:** Esnek, utility-first UI ve duyarlı tasarım.
- **Chart.js:** İnteraktif, pürüzsüz EEG zaman/frekans domeni grafikleri.
- **Vanilla JavaScript (ES6+):** Chatbot ve DOM manipülasyonu.

---

## 🏗️ Kurulum ve Çalıştırma

Aşağıdaki adımları takip ederek projeyi kendi yerel makinenizde saniyeler içinde ayağa kaldırabilirsiniz. (Python **3.10+** yüklü olmalıdır. TensorFlow son sürümü için önerilen Python sürümü 3.12 veya 3.13)

### 1️⃣ Repoyu Klonlayın
```bash
git clone https://github.com/Mers4596/EEG-Project.git
cd proje-ana-dizini
```

### 2️⃣ Sanal Ortam (Virtual Environment) Oluşturun
Bu adım paket çakışmalarını önlemek adına şiddetle tavsiye edilir.
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Gerekli Python Kütüphanelerini Kurun
Sistemin ihtiyaç duyduğu tüm bağımlılıklar titizlikle listelenmiştir:
```bash
pip install -r requirements.txt
```

### 4️⃣ Yerel Geliştirme Sunucusunu Başlatın
Tüm kurumlar tamamlandıktan sonra ana uygulamayı çalıştırın:
```bash
python app.py
```
Herhangi bir modül noksanlığı yoksa terminalinizde `Running on http://127.0.0.1:5000` mesajını göreceksiniz. Tarayıcınızdan bu adrese giderek sisteme erişebilirsiniz.

---

## 📂 Proje Dosya Dizini Mimarisi
```plaintext
📁 EEG-Project
 ├── 🧠 models/                      # Yapay Zeka Derin Öğrenme Modelleri
 │   ├── 📉 final_model.keras          # Önceden eğitilmiş ana model
 │   └── 📑 best_model.keras           # Yedek yüksek başarım modeli
 ├── 🎨 templates/                   # Frontend Görünümleri (HTML)
 │   ├── 📋 layout.html                # Ana omurga, Tailwind yükleyici, Chatbot
 │   ├── 🏠 index.html                 # Dashboard, yükleme ve geçmiş
 │   ├── 📊 analiz.html                # Grafikli Detay Görüntüleme sayfası
 │   ├── ⚖️ karsilastirma.html         # İstatistiksel kıyaslama alanı
 │   └── ...                           # Diğer sayfalar (Login/Register)
 ├── ⬆️ uploads/                     # Geçici işleme dosyaları 
 │   └── (gitignore)
 ├── 💾 userData/                    # Yerel veritabanı JSON dosyaları
 │   ├── 🕒 history/                   # Kullanıcı ID bazlı geçmiş veriler
 │   ├── 👤 users.json                 # Identity yönetimi (Şifrelenmiş)
 │   └── 🤖 chatbot_knowledge.json     # NLP Kuralları ve cevaplar
 ├── ⚙️ app.py                       # Ana sunucu mimarisi, controller fonksiyonları
 ├── 📦 requirements.txt             # Proje Python Bağımlılıkları (Paket listesi)
 ├── 🛡️ .gitignore                   # Hassas veri gizleme konfigürasyonu
 └── 📖 README.md                    # Proje dökümü
```

## ⚠️ Uyarı ve Feragatname (Disclaimer)
**EEG Sinyal Analizi Sistemi** tamamen akademik araştırma, veri işleme öğrenimi ve yazılım mühendisliği vaka analizleri (*Proof of Concept*) çerçevesinde geliştirilmiştir. Sistem **hiçbir şekilde profesyonel tıbbi, diagnostik veya klinik onaylı karar mekanizması olarak kullanılamaz**. Elde edilen EEG yorumlamaları bir doktor teşhisinin yerine geçmez; kritik ve şüpheli durumlarda mutlaka bir nöroloji, psikiyatriyatrik klinik uzmanına başvurulmalıdır.

## 🤝 Katkıda Bulunma, Lisans ve İletişim
Proje, açık geliştirmeye açıktır. Repoyu 'Fork' ederek pull-request gönderebilirsiniz. 
**Sorun Bildirimi:** Sistem içerisinde "Hata Bildirme" veya "Chatbot" üzerinden `yanlış, hata vb` keywordleri girildiğinde sistem loglarına düşürebilir, ya da Github Issues altından konu açabilirsiniz. 

---
*Developed with ❤️ by **Mehmet Ersolak***
