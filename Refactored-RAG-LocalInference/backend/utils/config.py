import os
from dotenv import load_dotenv

load_dotenv()

# ----------------- PATHS -----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
HISTORY_DB_DIR = os.path.join(BASE_DIR, "history_db")
USER_DATA_DB_DIR = os.path.join(BASE_DIR, "user_data_db")
USER_FILES_ROOT = os.path.join(BASE_DIR, "user_files")
DB_PATH = os.path.join(BASE_DIR, "chat_history.db")
KNOWLEDGE_BASE_FOLDER = r"D:/PyCharm/GenAI/Chatbot/RAG-pipeline/ReveNanoBooks/Books"
CACHE_DIR = r"D:\huggingface_cache"

# Ensure cache directory exists and set environment variables
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR
print(f"Using shared cache: {CACHE_DIR}")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ----------------- MODELS -----------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
GROQ_MODEL = "llama-3.1-8b-instant"

# ----------------- RAG CONFIG -----------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_HISTORY = 4
RETRIEVAL_K = 15
FETCH_K = 30
RERANK_TOP_N = 5
COLLECTION_NAME = "rag_collection"
HISTORY_THRESHOLD = 0.20
HISTORY_TOKEN_LIMIT = 1500
HISTORY_CHUNK_SIZE = 500
HISTORY_CHUNK_OVERLAP = 50

# ----------------- PATTERNS -----------------
REMOVE_PATTERNS = [
    r"^\d+$",                      # page numbers
    r"^Table\s+\d+",                # Table references
    r"^Figure\s+\d+",               # Figure references
    r"INTRODUCTION",
    r"WEATHERING AND SOIL FORMATION",
    r"SOIL WATER",
    r"SOIL AIR AND SOIL TEMPERATURE",
    r"TILLAGE",
    r"WATER MANAGEMENT",
    r"SOIL EROSION AND SOIL CONSERVATION",
    r"FUNDAMENTALS OF SOIL SCIENCE",
    r"SOIL COLLOIDS AND ION EXCHANGE IN SOIL",
    r"SOIL SURVEY AND MAPPING",
    r"SOIL ACIDITY",
    r"SOIL SALINITY AND ALKALINITY",
    r"MINERAL NUTRITION OF PLANTS",
    r"SOIL CLASSIFICATION",
    r"NITROGEN",
    r"PHOSPHORUS",
    r"POTASSIUM",
    r"SECONDARY NUTRIENTS",
    r"MICRONUTRIENTS",
    r"ANALYSIS OF SOIL, PLANT AND FERTILIZER FOR PLANT NU1RIENTS",
    r"SOIL FERTILITY EVALUATION",
    r"SOIL BIOLOGY AND BIOCHEMISTRY",
    r"SOIL ORGANIC MATTER",
    r"FERTILIZERS, MANURES AND BIOFERTILIZERS",
    r"SOIL FERTILITY MANAGEMENT",
    r"SOIL AND WATER QUALITY",
    r"SOIL POLLUTION AND ITS CONTROL",
    r"SOIL MANAGEMENT FOR SUSTAINABLE FARMING",
    r"CHEMICAL COMPOSITION OF SOILS",
    r"PHYSICAL PROPERTIES OF SOILS"
]

SKIP_SECTIONS = [
    r"^PREFACE",
    r"^REFERENCES",
    r"^CONTRIBUTORS"
]
