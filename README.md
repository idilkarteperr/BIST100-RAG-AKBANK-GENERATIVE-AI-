# 📊 BIST100 RAG – Retrieval-Augmented Financial Assistant

Bu proje, **Akbank Generative AI Bootcamp** kapsamında geliştirilmiş bir **Retrieval-Augmented Generation (RAG)** modelidir.  
Amaç, **Borsa İstanbul (BIST100)** şirketleri hakkında hızlı, güvenilir ve doğal dilde yanıtlar üretmektir.

🔗 **Canlı Uygulama:** [https://bist100-rag-vlagvhyz2hewypzgmsosy5.streamlit.app](https://bist100-rag-vlagvhyz2hewypzgmsosy5.streamlit.app)

---

## 🚀 Özellikler

- 💬 Türkçe doğal dilde sorulara yanıt verebilir.  
- 📈 Şirketlerin **faaliyet alanlarını**, **sektörlerini**, **temettü oranlarını** ve **finansal metriklerini** açıklar.  
- 🔍 RAG mimarisiyle **belge tabanlı arama (retrieval)** ve **LLM tabanlı metin üretimini (generation)** birleştirir.  
- 🧠 Finansal verileri kullanarak anlamlı özetler ve açıklamalar oluşturur.  
- ⚙️ Streamlit arayüzü üzerinden etkileşimli kullanım sağlar.

---

## 🧩 Mimari

Proje üç ana bileşenden oluşur:

1. **Retriever** → BIST100 şirket verilerini içeren CSV veya DataFrame üzerinde **TF-IDF / Embedding tabanlı arama** yapar.  
2. **Generator** → Kullanıcının sorusuna göre en ilgili dokümanları seçip **LLM tabanlı yanıt** üretir.  
3. **UI (Streamlit)** → Kullanıcının metin girişi yapabildiği, model yanıtlarını görebildiği sade bir arayüz sağlar.

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Açıklama |
|------------|-----------|
| 🐍 **Python** | Ana geliştirme dili |
| 💬 **Streamlit** | Web arayüzü |
| 🔎 **Haystack / FAISS / TF-IDF** | Bilgi getirici (retriever) katmanı |
| 🤖 **Hugging Face Transformers** | LLM tabanlı metin üretimi |
| 📊 **Pandas / NumPy** | Finansal veri işleme |
| 🧮 **Scikit-learn** | TF-IDF, cosine similarity hesaplamaları |
| 🧠 **SentenceTransformers** | Embedding üretimi |
| 💾 **CSV / JSON** | Veri depolama formatları |

---

## 💰 Veri Kaynağı

Projedeki BIST100 verileri aşağıdaki kaynaklardan derlenmiştir:

- [Borsa İstanbul Resmî Sitesi (bist.com.tr)](https://www.borsaistanbul.com)
- [Finnhub.io API](https://finnhub.io)
- [Yahoo Finance](https://finance.yahoo.com)
- [Kaggle - BIST100 Dataset](https://www.kaggle.com)

Veriler manuel olarak doğrulanmış ve CSV formatında sisteme eklenmiştir.  
Her satırda şirketin kısa kodu, faaliyet alanı, sektör bilgisi ve son temettü oranı yer almaktadır.

---

## 📂 Proje Yapısı

📁 bist100-rag
├── app.py # Streamlit uygulaması
├── retriever.py # RAG için veri getirici modül
├── generator.py # LLM tabanlı yanıt üretici
├── data/
│ └── bist100_companies.csv
├── requirements.txt # Bağımlılıklar
└── README.md # Proje açıklaması (bu dosya)


---

## ▶️ Çalıştırma

⃣ Gerekli paketleri yükle
```bash
pip install -r requirements.txt

streamlit run app.py

http://localhost:8501

| Kullanıcı Sorusu                                         | Model Yanıtı (örnek)                                                        |
| -------------------------------------------------------- | --------------------------------------------------------------------------- |
| “ASELSAN hangi alanda faaliyet gösteriyor?”              | ASELSAN, savunma sanayi ve elektronik sistemler alanında faaliyet gösterir. |
| “THYAO’nun 2023 temettü oranı nedir?”                    | Türk Hava Yolları, 2023 yılında %12,5 temettü dağıtmıştır.                  |
| “BIST100’de enerji sektöründe hangi şirketler var?”      | Enerjisa, Aksa Enerji, Zorlu Enerji gibi şirketler listelenir.              |
| “Banka hisseleri arasında en yüksek kar marjı kime ait?” | Akbank ve Garanti BBVA öne çıkmaktadır.                                     |

⚙️ Model Detayları
Embedding Modeli: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Retriever: TF-IDF + Cosine Similarity veya FAISS tabanlı vektör arama
Generator: GPT tabanlı LLM (OpenAI veya HuggingFace pipeline)
Prompt Şablonu:

Kullanıcının sorusu: {question}
En alakalı belgeler: {context}
Cevap: 

🧾 Lisans
Bu proje MIT Lisansı altında paylaşılmıştır.
Eğitim ve araştırma amaçlı serbestçe kullanılabilir

👩‍💻 Geliştirici
İdil Karteper
GitHub: @idilkarteperr
Deploy: https://bist100-rag-vlagvhyz2hewypzgmsosy5.streamlit.app

🌟 Katkıda Bulunmak
Pull request’ler ve issue’lar memnuniyetle karşılanır.
Projeyi fork’layıp kendi verinizle test edebilirsiniz.


