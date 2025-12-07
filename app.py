import streamlit as st
import nest_asyncio
import chromadb
import os
import Stemmer
from llama_index.core import StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Yerel Hukuk Asistanı", layout="wide", page_icon="⚖️")
st.title("⚖️ Hukuk Asistanı (Gemini 2.5 Flash)")

# Notebook hatası önleyici (Localde de bazen gerekir)
nest_asyncio.apply()

# 2. SİSTEMİ YÜKLEME (Cache ile hızlandırılmış)
@st.cache_resource
def load_system():
    # --- A. MODELLER ---
    # Embedding: CPU'da çalışsın (Hafif ve güvenli)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cpu")
    
    # LLM: Gemini 2.5 Flash
    # BURAYA API ANAHTARINI YAPIŞTIR
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        st.error("⚠️ API Key Bulunamadı! Lütfen GOOGLE_API_KEY ortam değişkenini ayarlayın.")
        st.stop()

    llm = Gemini(model="models/gemini-2.5-flash", api_key=api_key)
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    # --- B. VERİTABANI BAĞLANTISI (YEREL) ---
    # Klasör yapına göre yollar: './chroma_db' ve './storage'
    base_path = "." 
    
    if not os.path.exists(f"{base_path}/chroma_db"):
        st.error("HATA: 'chroma_db' klasörü bulunamadı!")
        st.stop()

    db = chromadb.PersistentClient(path=f"{base_path}/chroma_db")
    chroma_collection = db.get_or_create_collection("hukuk_verileri")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(
        persist_dir=f"{base_path}/storage", 
        vector_store=vector_store
    )
    index = load_index_from_storage(storage_context)
    
    # --- C. RETRIEVER KURULUMU ---
    nodes = list(storage_context.docstore.docs.values())
    
    # Similarity Top K = 20 (Geniş Tarama - Gemini için ideal)
    base_retriever = index.as_retriever(similarity_top_k=20)
    
    auto_merging_retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context=storage_context
    )
    
    stemmer = Stemmer.Stemmer("turkish")
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, 
        similarity_top_k=20, 
        stemmer=stemmer, 
        language="turkish"
    )
    
    # --- D. FUSION (Voltran) ---
    retriever_prompt = PromptTemplate(
        "Kullanıcının sorusunu veritabanında aramak için en iyi Türkçe arama cümlesini yaz.\n"
        "Soru: {query}\nArama Cümlesi:"
    )
    
    fusion_retriever = QueryFusionRetriever(
        [auto_merging_retriever, bm25_retriever],
        similarity_top_k=20,
        num_queries=3, # 3 farklı açıdan arasın
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
        query_gen_prompt=retriever_prompt
    )
    
    # --- E. CEVAP MOTORU (Avukat Prompt) ---
    qa_prompt = PromptTemplate(
        "Sen uzman bir Türk hukukçususun. Aşağıdaki yasal metinleri analiz et.\n"
        "---------------------\n{context_str}\n---------------------\n"
        "KURALLAR:\n"
        "1. Sadece yukarıdaki metne sadık kal.\n"
        "2. Asla kafandan başlık veya içerik uydurma.\n"
        "3. İlgili maddeyi, fıkraları ve bentleri eksiksiz ve olduğu gibi aktar.\n"
        "4. Eğer madde parçalara ayrılmışsa, hepsini birleştir ve bütün halini sun.\n"
        "Soru: {query_str}\n"
        "Cevap:"
    )
    
    return RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        llm=llm,
        text_qa_template=qa_prompt
    )

# 3. BAŞLATMA
with st.spinner("Sistem Hazırlanıyor... (İlk açılış birkaç saniye sürebilir)"):
    try:
        query_engine = load_system()
        st.success("Sistem Hazır! 🚀")
    except Exception as e:
        st.error(f"Sistem başlatılamadı: {e}")
        st.stop()

# 4. SOHBET ARAYÜZÜ
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana yaz
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Yeni soru girişi
if prompt := st.chat_input("Hukuki sorunuzu buraya yazın..."):
    # Kullanıcı mesajını göster ve kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan cevabını üret
    with st.chat_message("assistant"):
        with st.spinner("Mevzuat taranıyor..."):
            try:
                response = query_engine.query(prompt)
                st.markdown(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
                
                # Kaynakları göster (Opsiyonel)
                with st.expander("📚 Kaynak Belgeleri İncele"):
                    # Skor sırasına göre ilk 5 kaynağı gösterelim
                    for node in response.source_nodes[:5]:
                        dosya_adi = node.metadata.get('file_name', 'Bilinmiyor')
                        skor = node.score
                        st.write(f"**Dosya:** {dosya_adi} (Alaka: {skor:.2f})")
                        st.caption(node.text[:300] + "...") # İlk 300 karakter
                        st.divider()
                        
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")