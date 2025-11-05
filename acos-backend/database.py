import sqlite3
import torch
import os
from bert_utils.tokenization import BertTokenizer
from modeling import BertForQuadABSA, CategorySentiClassification
from tqdm import tqdm

# --- Configuration ---
# Path to the trained models (should match the output_dir in run.sh)
STEP1_MODEL_DIR = './output/Extract-Classify-QUAD/rest16_1st/'
STEP2_MODEL_DIR = './output/Extract-Classify-QUAD/rest16_2nd/'
DB_FILE = 'products.db'

# --- Dummy product and review data ---
# In a real project, this part should be modified to read data from a DB or files.
DUMMY_PRODUCTS = [
    {"id": "P001", "name": "Super Soft Cotton T-Shirt", "image_url": "http://example.com/image1.jpg"},
    {"id": "P005", "name": "Fluffy Teddy Bear", "image_url": "http://example.com/image5.jpg"},
    {"id": "N007", "name": "Gaming Laptop", "image_url": "http://example.com/image7.jpg"}
]

DUMMY_REVIEWS = {
    "P001": [
        "This shirt is really soft, I want to wear it every day. The color is nice too.",
        "The texture is soft, but the downside is that it feels a bit heavy."
    ],
    "P005": [
        "The fur is incredibly soft. My kid absolutely loves it.",
        "The doll was harder than I expected, which was a bit disappointing."
    ],
    "N007": [
        "The screen is truly beautiful, but the battery life is too short.",
        "The keyboard is really comfortable for typing. The design is great too."
    ]
}

class ACOS_Processor:
    def __init__(self, model_dir_step1, model_dir_step2):
        print("Initializing ACOS Processor...")
        self.device = torch.device("cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_dir_step1, do_lower_case=True)
        self.model_step1 = BertForQuadABSA.from_pretrained(model_dir_step1)
        self.model_step1.to(self.device)
        self.model_step1.eval()
        self.model_step2 = CategorySentiClassification.from_pretrained(model_dir_step2)
        self.model_step2.to(self.device)
        self.model_step2.eval()
        print("Models loaded successfully.")

    def analyze(self, sentence):
        # TODO: The actual ACOS model analysis logic needs to be implemented here.
        #       Step 1 -> Extract Aspect-Opinion pairs
        #       Step 2 -> Classify Category-Sentiment
        # For now, it returns dummy analysis results.
        print(f"Analyzing: \"{sentence}\"")
        if "soft" in sentence:
            return [{"aspect": "texture", "category": "TEXTURE", "opinion": "soft", "sentiment": "Positive"}]
        if "hard" in sentence:
            return [{"aspect": "texture", "category": "TEXTURE", "opinion": "hard", "sentiment": "Negative"}]
        if "screen" in sentence or "design" in sentence:
            return [{"aspect": "screen", "category": "DESIGN", "opinion": "beautiful", "sentiment": "Positive"}]
        if "battery" in sentence:
            return [{"aspect": "battery life", "category": "BATTERY", "opinion": "short", "sentiment": "Negative"}]
        return []

def setup_database():
    # If the DB file already exists, delete it and start over.
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # --- Create Tables ---
    print("Creating tables...")
    # Product information table
    cursor.execute('''
        CREATE TABLE products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            image_url TEXT
        )
    ''')
    # Original reviews table
    cursor.execute('''
        CREATE TABLE reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products (id)
        )
    ''')
    # ACOS analysis results table
    cursor.execute('''
        CREATE TABLE acos_attributes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER NOT NULL,
            aspect TEXT,
            category TEXT NOT NULL,
            opinion TEXT,
            sentiment TEXT NOT NULL,
            FOREIGN KEY (review_id) REFERENCES reviews (id)
        )
    ''')
    print("Tables created successfully.")
    conn.commit()
    return conn, cursor

def process_and_store_data(conn, cursor, processor):
    print("Processing data and storing to DB...")
    # 1. Store product information
    for product in DUMMY_PRODUCTS:
        cursor.execute("INSERT INTO products VALUES (?, ?, ?)", (product['id'], product['name'], product['image_url']))

    # 2. Analyze each review and store the results
    for product_id, reviews in tqdm(DUMMY_REVIEWS.items(), desc="Processing Products"):
        for review_content in reviews:
            # Store the original review
            cursor.execute("INSERT INTO reviews (product_id, content) VALUES (?, ?)", (product_id, review_content))
            review_id = cursor.lastrowid # Get the ID of the review just saved

            # Analyze with ACOS model
            quads = processor.analyze(review_content)

            # Store the analysis results
            for quad in quads:
                cursor.execute("INSERT INTO acos_attributes (review_id, aspect, category, opinion, sentiment) VALUES (?, ?, ?, ?, ?)",
                               (review_id, quad['aspect'], quad['category'], quad['opinion'], quad['sentiment']))
    
    conn.commit()
    print("Data processing and storage complete.")


if __name__ == '__main__':
    # Load ACOS model processor
    # Note: A DummyProcessor can be used for testing until the actual analysis logic is implemented.
    # processor = ACOS_Processor(STEP1_MODEL_DIR, STEP2_MODEL_DIR) # For loading the actual model
    class DummyProcessor: # Dummy processor for testing
        def analyze(self, s): return ACOS_Processor.analyze(self, s)
    processor = DummyProcessor()

    # Set up DB and run data processing
    conn, cursor = setup_database()
    process_and_store_data(conn, cursor, processor)
    conn.close()
    print(f"\nTask complete! '{DB_FILE}' has been created.")