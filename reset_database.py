import shutil
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_parse import LlamaParse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys

# --- GÜVENLİK KONTROLÜ ---
# Yanlışlıkla çalıştırmayı önlemek için kullanıcıdan onay isteyelim
onay = input("⚠️ DİKKAT: Bu işlem mevcut veritabanını SİLİP sıfırdan oluşturacak. Devam edilsin mi? (e/h): ")
if onay.lower() != 'e':
    print("İşlem iptal edildi.")
    sys.exit()

# API Key Kontrolü
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
if not LLAMA_CLOUD_API_KEY:
    print("❌ HATA: 'LLAMA_CLOUD_API_KEY' bulunamadı. Lütfen ortam değişkeni olarak ekleyin.")
    sys.exit(1)

# --- 1. MODELLER (CPU) ---
print("⚙️ Modeller hazırlanıyor...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cpu")
Settings.embed_model = embed_model

# --- 2. TEMİZLİK ---
print("🧹 Eski veritabanı temizleniyor...")
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
if os.path.exists("./storage"):
    shutil.rmtree("./storage")

# --- 3. OKUMA (Tüm Klasör) ---
print("📚 'data' klasöründeki tüm PDF'ler okunuyor...")

os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY
parser = LlamaParse(result_type="markdown", language="tr")
file_extractor = {".pdf": parser}

# Klasör kontrolü
if not os.path.exists("./data"):
    os.makedirs("./data")
    print("❌ HATA: './data' klasörü yoktu, oluşturuldu. Lütfen içine PDF atıp tekrar deneyin.")
    sys.exit(1)

try:
    documents = SimpleDirectoryReader(
        input_dir="./data", 
        file_extractor=file_extractor
    ).load_data()
    
    if not documents:
        print("❌ HATA: './data' klasörü boş veya PDF bulunamadı.")
        sys.exit(1)

    # --- 4. PARÇALAMA ---
    print(f"🧩 {len(documents)} parça belge işleniyor...")
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 512, 256],
        chunk_overlap=50
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    # --- 5. KAYDETME ---
    print("💾 Veritabanı oluşturuluyor...")
    
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("hukuk_verileri")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store
    )

    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    index.storage_context.persist(persist_dir="./storage")
    
    print("✅ KURULUM TAMAMLANDI! Veritabanı sıfırlandı ve yeniden kuruldu.")

except Exception as e:
    print(f"❌ BEKLENMEYEN HATA: {e}")