import sys
import os
import importlib
# ACOS CODE IMPORT
ACOS_CODE_DIR = os.path.abspath('./ACOS-main/Extract-Classify-ACOS')
sys.path.insert(0, ACOS_CODE_DIR)
import run_classifier_dataset_utils
importlib.reload(run_classifier_dataset_utils)
print("="*50)
print("DEBUG: Loaded 'run_classifier_dataset_utils' from:")
print(run_classifier_dataset_utils.__file__)
print("="*50)

import pandas as pd
import sqlite3
import argparse
import json
import shutil
import logging
from typing import List, Dict, Any


# Helper function to build argv from argparse.Namespace
def build_argv_from_args(args_namespace, script_name="script.py"):
    argv = [script_name]
    for key, value in vars(args_namespace).items():
        if isinstance(value, bool):
            if value:
                argv.append(f'--{key}')
        elif value is not None:
            argv.append(f'--{key}')
            argv.append(str(value))
    return argv


try:
    import run_step1
    import run_step2
    from tokenized_data import get_1st_pairs
except ImportError as e:
    print(f"Error: Cannot import ACOS Code. ACOS_CODE_DIR: {ACOS_CODE_DIR}")
    print(f"Error details: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# BERT MODEL
BERT_BASE_DIR = './ACOS-main/Extract-Classify-ACOS/bert-base-uncased'

# AMAZON REVIEW DATASET PATH
NEW_REVIEW_FILE = './Data/Appliances_trimmed.jsonl'

# SQLLITE DB PATH
FLASK_DB_PATH = './db1.db'

# ACOS TRAINED MODEL PATH
TRAINED_MODEL_STEP1 = './Trained/rest16_1st'
TRAINED_MODEL_STEP2 = './Trained/rest16_2nd'

# TEMP PATH
TEMP_DATA_DIR = '/tmp/acos_pipeline'

# TEMP STEP1 INPUT PATH
STEP1_INPUT_DIR = os.path.join(TEMP_DATA_DIR, 'step1_input')

# TEMP STEP1 OUTPUT PATH
STEP1_OUTPUT_DIR = os.path.join(TEMP_DATA_DIR, 'step1_output')

# TEMP STEP2 INPUT PATH
STEP2_INPUT_DIR = os.path.join(TEMP_DATA_DIR, 'step2_input')

# TEMP STEP2 OUTPUT PATH
STEP2_OUTPUT_DIR = os.path.join(TEMP_DATA_DIR, 'step2_output')

DOMAIN_NAME = 'predict'
STEP1_INPUT_FILE_TSV = f"{DOMAIN_NAME}_quad_bert.tsv"
STEP2_INPUT_FILE_TSV = f"{DOMAIN_NAME}_pair_1st.tsv"


# TEMP DIRECTORY SETUP
def setup_directories():
    logging.info(f"Setting up temporary directories at {TEMP_DATA_DIR}")
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
    os.makedirs(STEP1_INPUT_DIR, exist_ok=True)
    os.makedirs(STEP1_OUTPUT_DIR, exist_ok=True)
    os.makedirs(STEP2_INPUT_DIR, exist_ok=True)
    os.makedirs(STEP2_OUTPUT_DIR, exist_ok=True)


# JSONL TO STEP1 INPUT TSV
def prepare_step1_input(source_jsonl_path: str, output_tsv_path: str):
    logging.info(f"'{source_jsonl_path}' to STEP1 input TSV at '{output_tsv_path}'")
    dummy_label = '-1,-1 -1,-1 0 -1,-1'

    line_count = 0
    error_count = 0

    try:
        with open(source_jsonl_path, 'r', encoding='utf-8') as fin, \
             open(output_tsv_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    product_id = data.get("asin")
                    text = data.get("text")
                    
                    if not product_id or not text or text.isspace():
                        error_count += 1
                        continue
                        
                    product_id = str(product_id).strip()
                    text = str(text).strip().replace('\n', ' ').replace('\t', ' ')
                        
                    fout.write(f"{product_id} @@@ {text}\t{dummy_label}\n")
                    line_count += 1
                    
                except json.JSONDecodeError:
                    error_count += 1
                except Exception:
                    error_count += 1

    except FileNotFoundError:
        logging.error(f"Error: File not Found: '{source_jsonl_path}")
        sys.exit(1)
        
    if line_count == 0:
        logging.error(f"Error: No valid reviews.")
        sys.exit(1)

    logging.info(f"Step 1 Complete! Total {line_count} reviews (Error: {error_count})")


# RUN STEP1
def execute_step1():
    logging.info("Executing ACOS Step 1...")

    args = argparse.Namespace(
        bert_model=TRAINED_MODEL_STEP1,
        data_dir=STEP1_INPUT_DIR,
        task_name='quad',
        output_dir=STEP1_OUTPUT_DIR,
        do_train=False,
        do_eval=True,
        eval_batch_size=32,
        max_seq_length=128,
        model_type='quad',
        domain_type=DOMAIN_NAME,
        do_lower_case=True,

        train_batch_size=32,
        num_train_epochs=10.0,
        warmup_proportion=0.1,
        learning_rate=2e-5,
        no_cuda=False,
        seed=42,
        gradient_accumulation_steps=1,
        fp16=False,
        loss_scale=0,
        local_rank=-1
    )

    original_argv = sys.argv
    sys.argv = build_argv_from_args(args, "run_step1.py")

    try:
        run_step1.main()
        logging.info("ACOS Step 1 Completed Successfully.")
    except Exception as e:
        logging.error(f"Error during ACOS Step 1: {e}")
        logging.error(f"   >Check BERT_BASE_DIR: {BERT_BASE_DIR} or TRAINED_MODEL_STEP1: {TRAINED_MODEL_STEP1}")
        sys.exit(1)
    finally:
        sys.argv = original_argv


# STEP1 OUTPUT TO STEP2 INPUT
def execute_get_pairs():
    logging.info("Preparing STEP2 input from STEP1 output...")

    args = argparse.Namespace(
        pred_data_dir=STEP1_OUTPUT_DIR,
        data_dir=STEP2_INPUT_DIR,
        domain=DOMAIN_NAME
    )

    try:
        get_1st_pairs.main(args)
        logging.info("STEP2 input preparation completed successfully.")
    except Exception as e:
        logging.error(f"Error during STEP2 input preparation: {e}")
        logging.error(f"   >Check STEP1_OUTPUT_DIR: {STEP1_OUTPUT_DIR}")
        sys.exit(1)


# RUN STEP2
def execute_step2():
    logging.info("Executing ACOS Step 2...")

    args = argparse.Namespace(
        bert_model=TRAINED_MODEL_STEP2,          
        data_dir=STEP2_INPUT_DIR,
        task_name='categorysenti',
        output_dir=STEP2_OUTPUT_DIR,
        do_train=False,
        do_eval=True,
        eval_batch_size=32,
        max_seq_length=128,
        model_type='categorysenti',
        domain_type=DOMAIN_NAME,
        do_lower_case=True,

        train_batch_size=32,
        num_train_epochs=10.0,
        warmup_proportion=0.1,
        learning_rate=2e-5,
        no_cuda=False,
        seed=42,
        gradient_accumulation_steps=1,
        fp16=False,
        local_rank=-1
    )

    original_argv = sys.argv
    sys.argv = build_argv_from_args(args, "run_step2.py")

    try:
        run_step2.main()
        logging.info("ACOS Step 2 Completed Successfully.")
    except Exception as e:
        logging.error(f"Error during ACOS Step 2: {e}")
        logging.error(f"   >Check TRAINED_MODEL_STEP2: {TRAINED_MODEL_STEP2}")
        sys.exit(1)
    finally:
        sys.argv = original_argv


# PARSE RESULTS AND LOAD TO DB
def parse_results_and_load_db(results_dir: str, db_path: str):
    logging.info(f"DB Loading Started... DB Path: {db_path}")

    results_file = os.path.join(results_dir, 'predict_results.json')

    if not os.path.exists(results_file):
        logging.error(f"Error: Results file not found at '{results_file}'")
        return
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f) 
    except json.JSONDecodeError as e:
        logging.error(f"Error: '{results_file}' Parsing JSON failed. {e}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS acos_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL,
        review_text TEXT,
        aspect TEXT,
        opinion TEXT,
        category TEXT,
        sentiment INTEGER
    )
    ''')

    sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2, "negative": 0, "neutral": 1, "positive": 2}

    processed_count = 0
    
    for item in predictions:
        try:
            full_text, aspect, opinion, category, sentiment_str = item
            
            if ' @@@ ' not in full_text:
                logging.warning(f"Warning: no ' @@@ ' exists. Skip: {full_text}")
                continue
                
            product_id, review_text = full_text.split(' @@@ ', 1)
            
            sentiment_int = sentiment_map.get(sentiment_str, -1)
            
            cursor.execute(
                "INSERT INTO acos_results (product_id, review_text, aspect, opinion, category, sentiment) VALUES (?, ?, ?, ?, ?, ?)",
                (product_id, review_text, aspect, opinion, category, sentiment_int)
            )
            processed_count += 1
            
        except ValueError as e:
            logging.warning(f"Warning: Parsing Error. Skip. item: {item}, error: {e}")
        except Exception as e:
            logging.error(f"error: {e}")

    conn.commit()
    conn.close()
    
    logging.info(f"--- DB Saving Complete: Total {processed_count} ACOS Quadruples Saved")


def main_pipeline():

    logging.info("======================================")
    logging.info("   ACOS PIPELINE AND DB LOADING START   ")
    logging.info(f"   DATASET: {NEW_REVIEW_FILE}    ")
    logging.info("======================================")

    setup_directories()

    step1_input_path = os.path.join(STEP1_INPUT_DIR, STEP1_INPUT_FILE_TSV)
    prepare_step1_input(NEW_REVIEW_FILE, step1_input_path)

    execute_step1()

    execute_get_pairs()

    execute_step2()

    parse_results_and_load_db(STEP2_OUTPUT_DIR, FLASK_DB_PATH)

    logging.info("======================================")
    logging.info(f"    Pipeline Completed. Results are saved in '{FLASK_DB_PATH}'  ")
    logging.info("======================================")


if __name__ == "__main__":
    main_pipeline()

