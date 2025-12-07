import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import sys

# --- GÜVENLİK VE AYAR KONTROLÜ ---
# API Key'i kodun içine gömmüyoruz, ortam değişkeninden çekiyoruz
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not LLAMA_CLOUD_API_KEY:
    print("❌ HATA: 'LLAMA_CLOUD_API_KEY' bulunamadı!")
    print("Lütfen terminalde veya Docker'da bu değişkeni tanımlayın.")
    sys.exit(1)

# Kullanıcıdan dosya ismini alalım
if len(sys.argv) < 2:
    print("❌ HATA: Lütfen eklenecek dosya adını belirtin.")
    print("Kullanım: python add_new_file.py <dosya_adi.pdf>")
    sys.exit(1)

dosya_adi = sys.argv[1]
yeni_dosya_yolu = f"./data/{dosya_adi}"

if not os.path.exists(yeni_dosya_yolu):
    print(f"❌ HATA: '{yeni_dosya_yolu}' dosyası bulunamadı.")
    print("Lütfen dosyayı 'data' klasörüne attığınızdan emin olun.")
    sys.exit(1)

# --- 0. MODELLER ---
print("⚙️ Modeller hazırlanıyor (CPU)...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cpu")
Settings.embed_model = embed_model

# --- 1. SİSTEMİ YÜKLE ---
print("💾 Veritabanı diskten yükleniyor...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("hukuk_verileri")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    persist_dir="./storage", 
    vector_store=vector_store
)
index = load_index_from_storage(storage_context)

# --- 2. DOSYAYI OKU VE PARÇALA ---
print(f"📄 '{dosya_adi}' LlamaParse ile işleniyor...")

os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY
parser = LlamaParse(result_type="markdown", language="tr")
file_extractor = {".pdf": parser}

try:
    new_documents = SimpleDirectoryReader(
        input_files=[yeni_dosya_yolu], 
        file_extractor=file_extractor
    ).load_data()

    print("🧩 Dosya hiyerarşik olarak parçalanıyor...")
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 512, 256],
        chunk_overlap=50
    )

    new_nodes = node_parser.get_nodes_from_documents(new_documents)
    new_leaf_nodes = get_leaf_nodes(new_nodes)

    # --- 3. KAYDET ---
    print("➕ Veriler veritabanına ekleniyor...")
    index.docstore.add_documents(new_nodes)
    index.insert_nodes(new_leaf_nodes)
    index.storage_context.persist(persist_dir="./storage")

    print(f"✅ BAŞARILI! '{dosya_adi}' sisteme eklendi ve kaydedildi.")

except Exception as e:
    print(f"❌ BEKLENMEYEN HATA: {e}")