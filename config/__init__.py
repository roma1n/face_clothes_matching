import os

# Lib constants
EMBEDDING_SIZE = 2048
EMBEDDER_INPUT_SHAPE = (224, 224, 3)
MATCHING_IGNORE_LOGIT_VALUE = -1e5

# DeepFace models
# Avaliable: ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
DEEPFACE_MODEL = 'Facenet512'

# Paths
PROJECT_DIR = os.environ['PROJECT_DIR']
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TOKENS_DIR = os.path.join(PROJECT_DIR, 'tokens')

# Lamoda dataset
LAMODA_DIR = os.path.join(DATA_DIR, 'lamoda')
LAMODA_IMG_DIR = os.path.join(LAMODA_DIR, 'img')
LAMODA_TRANSFORMED_IMG_DIR = os.path.join(LAMODA_DIR, 'transformed')
LAMODA_IMG_WITH_SEGMENTATION_DIR = os.path.join(LAMODA_DIR, 'segmentation')
LAMODA_FASHION_ITEM_EMBDDEINGS_PATH = os.path.join(LAMODA_DIR, 'fashion_item_embeddings.json')

# API
API_HOST = '178.154.250.149'
API_PORT = 8443

SSL_CERT_PATH = os.path.join(TOKENS_DIR, 'cert.pem')
SSL_PKEY_PATH = os.path.join(TOKENS_DIR, 'pkey.pem')

# Telegram bots
BOT_HOST = API_HOST
BOT_PORT = API_PORT
DEMO_BOT_TOKEN_PATH = os.path.join(TOKENS_DIR, 'demo_bot.token')
BOT_DATA_DIR = os.path.join(DATA_DIR, 'tg_bot')
