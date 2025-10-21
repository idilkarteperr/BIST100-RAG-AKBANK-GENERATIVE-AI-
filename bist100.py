# -*- coding: utf-8 -*-

# =================================================================================

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset

# Haystack ve Gerekli Bileşenleri İçe Aktarma
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator 
from haystack.utils import Secret
# --- 1. Adım: Ortam Değişkenlerini ve API Anahtarını Yükleme ---
#.env dosyasını yükleyerek API anahtarını güvenli bir şekilde alıyoruz.
# Hugging Face Spaces'e deploy ederken bu anahtarı "Secrets" bölümüne eklemelisiniz.
try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY bulunamadı. Lütfen.env dosyanızı veya Streamlit secrets ayarlarınızı kontrol edin.")
        st.stop()
except Exception as e:
    st.error(f"Ortam değişkenleri yüklenirken bir hata oluştu: {e}")
    st.stop()

# --- 2. Adım: Veri Yükleme ve Hazırlama ---
# Bu fonksiyon, Hugging Face'ten veri setini indirir, işler ve
# Haystack'in kullanabileceği Document formatına dönüştürür.
# Streamlit'in cache mekanizması sayesinde bu işlem sadece bir kez yapılır.
@st.cache_resource
def load_and_prepare_data():
    
    with st.spinner("BIST100 veri seti yükleniyor..."):
        try:
            csv_file_path = "bist100verilercsv.csv" # Dosya ayracının (delimiter) ';' olduğundan emin olun 
            df = pd.read_csv(csv_file_path, sep=";")
            #dataset = load_dataset("bist100verilercsv.csv", split="train", token=HF_TOKEN)
            #df = dataset.to_pandas()

            # Yalnızca Türkçe özetleri olan ve boş olmayan tezleri al
            df_turkish = df[df['rag-text'].notna() & (df['rag-text']!= '')].copy()
            df_turkish.reset_index(drop=True, inplace=True)
            df_turkish = df_turkish.head(100) #zaten yüz

            # Haystack Document nesneleri oluştur
            # Haystack Document nesneleri oluştur
            documents = list()
            for _, row in df.iterrows(): # df_turkish yerine yeni yüklenen DataFrame'iniz olan 'df' kullanılmalı
                
                # 1. content (Ana Aranacak Metin) Eşleştirme
                # 'rag-text' alanı, RAG için en zengin bilgiyi içerir ve ana aranacak metin olmalıdır.
                content = str(row['rag-text'])
                
                # 2. meta (Ek Bilgiler) Eşleştirme
                meta = {
                    # 'title_tr' -> 'sembol'
                    # Document.content'ı 'rag-text' olarak kullandığımız için, sembolü (ASELS, GARAN vb.)
                    # promplarda kolayca referans verebilmek için 'title' olarak saklamak mantıklıdır.
                    'title': str(row['sembol']), 
                    
                    # 'author' -> 'sirket'
                    # Şirket adını (uzun halini) okunaklı bir etiket olarak saklamak için 'author' yerine 
                    # doğrudan kendi adıyla saklamak en iyisidir.
                    'sirket': str(row['sirket']),
                    
                    # Diğer önemli alanları metadata olarak saklayalım:
                    'marketcap': str(row['marketcap']),
                    'kur': str(row['kur']),
                    'temettu': str(row['temettue']),
                    'endeks': str(row['endeks'])
                    
                    # NOT: 'author', 'year', 'subject' gibi eski alan adlarını KULLANMAYIN.
                    # Bunlar anlamsız olacaktır. Bunun yerine verinizdeki gerçek alan adlarını kullanın.
                }
                
                # Document nesnesini listeye ekle
                documents.append(Document(content=content, meta=meta))

# Buradan sonra DocumentSplitter çalıştırılmalıdır (mevcut kodunuzdaki gibi)
            
            # Belgeleri parçalara ayır
            splitter = DocumentSplitter(split_by="word", split_length=700, split_overlap=0)
            split_docs = splitter.run(documents)
            
            return split_docs['documents']
        except Exception as e:
            st.error(f"Veri seti yüklenirken veya işlenirken hata oluştu: {e}")
            return None

# --- 3. Adım: FAISS Vektör Veritabanı Oluşturma ---
# Bu fonksiyon, hazırlanan belgeleri alır, gömme (embedding) modelini kullanarak
# vektörlere dönüştürür ve bir FAISS veritabanı oluşturur.
# Bu işlem de cache'lenir, böylece uygulama her yeniden çalıştığında tekrarlanmaz.
@st.cache_resource
def create_faiss_index(_split_docs):
    """
    Verilen belgeler için bir InMemory DocumentStore oluşturur ve doldurur.
    """
    if not _split_docs:
        return None
        
    with st.spinner("Vektör veritabanı oluşturuluyor ve belgeler işleniyor..."):
        try:
            document_store = InMemoryDocumentStore()
            
            doc_embedder = SentenceTransformersDocumentEmbedder(
                model="trmteb/turkish-embedding-model"
            )

            # Belgeleri ve gömme vektörlerini deposuna yazmak için bir boru hattı
            indexing_pipeline = Pipeline()
            indexing_pipeline.add_component("embedder", doc_embedder)
            indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
            indexing_pipeline.connect("embedder.documents", "writer.documents")

            # Boru hattını çalıştırarak indeksi oluştur
            indexing_pipeline.run({"embedder": {"documents": _split_docs}})
            
            return document_store
        except Exception as e:
            st.error(f"Vektör indeksi oluşturulurken hata oluştu: {e}")
            return None

# --- 4. Adım: RAG Pipeline Kurma ---
# Bu fonksiyon, RAG sisteminin tüm bileşenlerini (retriever, prompt, generator)
# bir araya getirerek sorgulanabilir bir Haystack Pipeline'ı oluşturur.
@st.cache_resource
def build_rag_pipeline(_document_store):
    """
    Verilen document_store'u kullanarak tam bir RAG Pipeline oluşturur.
    """
    if not _document_store:
        return None
        
    try:
        # 1. Geri Getirici (Retriever)
        retriever = InMemoryEmbeddingRetriever(document_store=_document_store, top_k=3)
        
        # 2. Prompt Şablonu
        template = """
        {% message role="system" %}
        Sağlanan BIST100 şirket belgelerine dayanarak soruyu yanıtlayın.
        Eğer belgeler soruyu yanıtlamak için yeterli bilgi içermiyorsa, 'Belgelerde bu konu hakkında yeterli bilgi bulamadım.' deyin.
        Yanıtınızı yalnızca sağlanan belgelere dayandırın ve kendi bilginizi eklemeyin.
        Yanıtınızın sonunda, kullandığınız şirketlerin kısa kodlarını (sembollerini) 'Hisse sembolü:' başlığı altında listeleyin.

        Belgeler:
        {% for doc in documents %}
          Şirket: {{ doc.meta['sirket'] }} (Sembol: {{ doc.meta['title'] }}) 
          Piyasa Değeri: {{ doc.meta['marketcap'] }} TRY
          Ortalama Kur (Ekim 2025): {{ doc.meta['kur'] }} TL
          Temettü Getirisi: {{ doc.meta['temettue'] }}
          İçerik: {{ doc.content }}
        {% endfor %}
        {% endmessage %}

        {% message role="user" %}
        Soru: {{question}}
        Yanıt:
        {% endmessage %} 
        """
        prompt_builder = ChatPromptBuilder(
            template=template, 
            required_variables=["documents", "question"]
            )

        # 3. Üretici (Generator)
        #message_converter = ChatMessageConverter()
        generator = GoogleGenAIChatGenerator(model="gemini-2.5-flash", api_key=Secret.from_token(GOOGLE_API_KEY))
        # Sorgu için metin gömme modeli
        text_embedder = SentenceTransformersTextEmbedder(model="trmteb/turkish-embedding-model")

        # 4. RAG Boru Hattını Oluşturma
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("generator", generator)

        # Bileşenleri birbirine bağla
        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "generator.messages") 
        #rag_pipeline.connect("message_converter.messages", "generator.messages") 
        return rag_pipeline
    except Exception as e:
        st.error(f"RAG boru hattı oluşturulurken hata oluştu: {e}")
        return None

# --- 5. Adım: Streamlit Web Arayüzü ---
def main():
    st.set_page_config(page_title="BİST100-İDİLKARTEPER CHATBOT", page_icon="$")
    
    st.title("BIST100 Araştırma Asistanı")
    st.caption("BIST100 hisseleri hakkında sorular sorun. (Veri Seti: `idilkarteper`)")

    # Gerekli bileşenleri yükle ve cache'le
    split_documents = load_and_prepare_data()
    if split_documents:
        document_store = create_faiss_index(split_documents)
        if document_store:
            rag_pipeline = build_rag_pipeline(document_store)
        else:
            rag_pipeline = None
    else:
        rag_pipeline = None

    if not rag_pipeline:
        st.warning("Uygulama başlatılamadı. Lütfen hata mesajlarını kontrol edin.")
        st.stop()

    # Sohbet geçmişini saklamak için session state kullan
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmiş mesajları göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan girdi al
    if prompt := st.chat_input("Örn: TCELL hakkında bilgi veriniz"):
        # Kullanıcının mesajını sohbet geçmişine ekle ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG boru hattını çalıştır ve yanıt al
        with st.spinner("BIST100 taranıyor..."):
            try:
                result = rag_pipeline.run({
                    "text_embedder": {"text": prompt},
                    "prompt_builder": {"question": prompt}
                })
                
                response = "Bir hata oluştu veya yanıt alınamadı."
                if result and "generator" in result and result["generator"]["replies"]:
                    chat_message = result["generator"]["replies"][0]
                    #response = chat_message.content[0].text
                    #response = result["generator"]["replies"]
                    response = chat_message.text # <-- KORRIGIERT: Zugriff direkt über .text

            except Exception as e:
                response = f"Sorgu işlenirken bir hata oluştu: {e}"

        # Asistanın yanıtını sohbet geçmişine ekle ve göster
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
