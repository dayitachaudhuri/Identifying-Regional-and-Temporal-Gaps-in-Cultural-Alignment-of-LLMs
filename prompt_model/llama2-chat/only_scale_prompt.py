import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import re
import json
import warnings
import logging
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==============================
# CONFIG
# ==============================

MODE = 'country'
COUNTRY_NAME = 'russia'
STATE = 'maharashtra'
LANGUAGE_CODE = 'en'
YEAR = '2022'
# Files
PERSONA_FILE = f"data/{COUNTRY_NAME}/{YEAR}/{YEAR}_{COUNTRY_NAME}_persona_groups_cleaned_{LANGUAGE_CODE}.csv"
EXISTING_CSV = f"responses/llama_responses/survey_answers_allstates_{COUNTRY_NAME}_{LANGUAGE_CODE}.csv"
CHOOSEN_COLS_FILE = "data/chosen_cols_updated.json"
QUESTIONS_FILE = f"data/translated_questions/questions_{LANGUAGE_CODE}.json"

MODEL_PATH = "/assets/models/meta-llama-2-chat-13b"

# Persona columns in the original CSV
persona_cols = {
    "country": 'B_COUNTRY: ISO 3166-1 numeric country code',
    "region": 'N_REGION_ISO: Region ISO 3166-2',
    "urban_rural": 'H_URBRURAL: Urban-Rural',
    "age": 'X003R: Age recoded (6 intervals)',
    "gender": 'Q260: Sex',
    "language": 'Q272: Language at home',
    "marital_status": 'Q273: Marital status',
    "education_level": 'Q275R: Highest educational level: Respondent (recoded into 3 groups)',
    "social_class": 'Q287: Social class (subjective)'
}

# ==============================
# LOAD DATA
# ==============================
df_existing = pd.read_csv(EXISTING_CSV)
df_persona = pd.read_csv(PERSONA_FILE)

with open(CHOOSEN_COLS_FILE, "r") as f:
    chosen_cols = json.load(f)

with open(QUESTIONS_FILE, "r") as f:
    all_questions = json.load(f)

# Filter only chosen & scale questions
scale_qsns = {
    qsn: all_questions[qsn]
    for qsn in all_questions
    if chosen_cols['chosen_cols'].get(qsn, False) and all_questions[qsn].get('scale', False)
}

print(f"Found {len(scale_qsns)} scale questions to update")

# ==============================
# MODEL INIT
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.float16
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==============================
# PROMPTS
# ==============================
general_prompts = {
    'en': '''
    Imagine you are a {language}-speaking {marital_status} {gender} from {urban_rural} {region}, India.
    You are in {age} years of age category and have completed {education_level} education level.
    You consider yourself part of the {social_class}. Answer the following question from this perspective.
    Select exactly one option from 1-10. Answer ONLY with the number corresponding to your choice.
    '''
}

user_prompts = {
    'en': "\nAnswer ONLY with numbers in this format: Q1: <option_number>, Q2: <option_number>, ... Do NOT repeat the questions or any other text."
}

joining_prompts = {
    'en': "Question {idx}: {q_text}\nOptions: {opts_text}\n"
}

# ==============================
# UPDATE SCALE QUESTIONS
# ==============================
import math

def update_scale_questions(df_existing, df_persona, scale_qsns, tokenizer, model, language_code=LANGUAGE_CODE, chunk_size=5):
    updated_rows = []

    for idx, row in tqdm(df_existing.iterrows(), total=len(df_existing), desc="Updating scale questions"):
        persona_row = df_persona.iloc[idx]

        general_context = {k: persona_row[v] for k, v in persona_cols.items()}
        respondent_answers = row.to_dict()

        # Prepare all questions
        all_questions_list = []
        for qsn_key, q_data in scale_qsns.items():
            for qsn_instance, q_text in enumerate(q_data['questions']):
                options_text = " ".join([f"{i}. {i}" for i in range(1, 11)])
                all_questions_list.append((qsn_key, q_text, options_text, qsn_instance))

        # Split into chunks of chunk_size
        num_chunks = math.ceil(len(all_questions_list) / chunk_size)
        for chunk_idx in range(num_chunks):
            chunk_questions = all_questions_list[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]

            # Build prompts for this chunk
            user_prompt = ""
            for q_idx, (qsn_key, q_text, opts_text, _) in enumerate(chunk_questions, start=1):
                user_prompt += f"Question {q_idx}: {q_text}\nOptions: {opts_text}\n"
            user_prompt += user_prompts[language_code]

            general_prompt = general_prompts[language_code].format(**general_context)
            messages = [
                {"role": "system", "content": general_prompt},
                {"role": "user", "content": user_prompt}
            ]

            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # increase if needed
                temperature=0.0,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Extract numbers only
            batch_answers = re.findall(r'Q\d+\s*[:\-]?\s*(\d+)', answer_text)

            for j, (qsn_key, _, _, qsn_instance) in enumerate(chunk_questions):
                if j < len(batch_answers):
                    ans_value = int(batch_answers[j])
                    respondent_answers[f"{qsn_key} - {qsn_instance}"] = ans_value
                else:
                    respondent_answers[f"{qsn_key} - {qsn_instance}"] = None

        updated_rows.append(respondent_answers)

    return pd.DataFrame(updated_rows)


# ==============================
# MAIN
# ==============================
df_updated = update_scale_questions(df_existing, df_persona, scale_qsns, tokenizer, model)
df_updated.to_csv(EXISTING_CSV, index=False)
print(f"Updated CSV saved at {EXISTING_CSV}")