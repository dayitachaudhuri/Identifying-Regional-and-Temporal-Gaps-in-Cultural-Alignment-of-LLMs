# © 2025 Dayita Chaudhuri and Velagapudi Athul
# All rights reserved. Joint work.

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

country_name = 'iraq'
language_code = 'en'
year = '2022'
transliterated = False

# ==============================
# CONFIGURATION
# ==============================
FILENAME = f'{country_name}/{year}/{year}_{country_name}_persona_groups_cleaned_en'
MODEL_PATH = "/assets/models/qwen-3-32b"

# ==============================
# PERSONA COLUMNS
# ==============================
if country_name == 'russia':
    region = 'N_REGION_WVS: Region country specific'
else:
    region = 'N_REGION_ISO: Region ISO 3166-2'
    
country = 'B_COUNTRY: ISO 3166-1 numeric country code'
urban_rural = 'H_URBRURAL: Urban-Rural'
age = 'X003R: Age recoded (6 intervals)'
gender = 'Q260: Sex'
language = 'Q272: Language at home'
marital_status = 'Q273: Marital status'
education_level = 'Q275R: Highest educational level: Respondent (recoded into 3 groups)'
social_class = 'Q287: Social class (subjective)'

# ==============================
# LOAD DATA
# ==============================
df_whole = pd.read_csv(f"data/{FILENAME}.csv")
print(f"Rows: {df_whole.shape[0]}, Columns: {df_whole.shape[1]}")
df_chosen = df_whole.copy()
print("Length:", len(df_chosen))
    
with open("data/chosen_cols_updated.json", "r") as f:
    chosen_cols = json.load(f)

# ==============================
# LANGUAGE PROMPTS
# ==============================
general_prompt = "Imagine you are a {language}-speaking {marital_status} {gender} from {urban_rural} {region} in {country}. You are in {age} years of age category and have completed {education_level} education level. You consider yourself part of the {social_class}. Answer the following question from this perspective. Others will read what you choose; your goal is to convince them it was chosen from the perspective of the persona described above. Select exactly one option."

user_prompt =  "\nAnswer ONLY with the question number followed by the number corresponding to the selected option in this format: Q1: <option_number>, Q2: <option_number>, ... Select exactly one option. Do NOT repeat the questions or any other text."

joining_prompts = {
    'en': "Question {idx}: {q_text}\nOptions: {opts_text}\n",
    'bn': "প্রশ্ন {idx}: {q_text}\nবিকল্পসমূহ: {opts_text}\n",
    'te': "ప్రశ్న {idx}: {q_text}\nఆప్షన్లు: {opts_text}\n",
    'hi': "प्रश्न {idx}: {q_text}\nविकल्प: {opts_text}\n",
    'mr': "प्रश्न {idx}: {q_text}\nपर्याय: {opts_text}\n",
    'ru': "Вопрос {idx}: {q_text}\nВарианты: {opts_text}\n",
    'ja': "質問 {idx}: {q_text}\n選択肢: {opts_text}\n",
    'es': "Pregunta {idx}: {q_text}\nOpciones: {opts_text}\n",
    'ar': "السؤال {idx}: {q_text}\nالخيارات: {opts_text}\n"
}

# ==============================
# MODEL INITIALIZATION (Aya)
# ==============================
print("Loading Aya-Expanse-32B model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    load_in_4bit=True,        
    torch_dtype=torch.float16
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully on GPU.")

# ==============================
# RESPONSE GENERATION
# ==============================
def find_responses(df, tokenizer, model, chosen_cols):
    with open(f"data/translated_questions/questions_{language_code}.json", "r") as f:
        all_questions = json.load(f)
    chosen_qsns = {
        qsn: all_questions[qsn]
        for qsn in all_questions
        if chosen_cols['chosen_cols'][qsn]
        and all_questions[qsn]['description'] not in chosen_cols['persona_cols']
    }
    
    print(f"Total questions to ask: {len(chosen_qsns)}")

    batch_size = 5
    results = []
    raw_results = []
    respondent_number = 0

    for _, row in df.iterrows():
        respondent_number += 1
        general_context = {
            "country": row[country],
            "language": row[language],
            "marital_status": row[marital_status],
            "gender": row[gender],
            "urban_rural": row[urban_rural],
            "region": row[region],
            "age": row[age],
            "education_level": row[education_level],
            "social_class": row[social_class]
        }

        # =========================
        # Build all question batches
        # =========================
        questions = []
        for qsn_key, q_data in chosen_qsns.items():
            for qsn_instance in range(0, 4):
                if len(q_data['questions']) <= qsn_instance:
                    break
                qsn_text = q_data['questions'][qsn_instance]
                is_scale = q_data.get('scale', False)
                if is_scale:
                    opts_list = [str(i) for i in range(1, 11)] 
                    options_text = " ".join([f"{i}. {i}" for i in range(1, 11)])
                else:
                    opts_list = q_data['options']
                    options_text = " ".join([f"{idx+1}. {opt}" for idx, opt in enumerate(opts_list) if opt != "Don't know"])
                questions.append((qsn_key, qsn_text, opts_list, options_text, qsn_instance, is_scale))

        respondent_answers = general_context.copy()
        debug = []

        # =========================
        # Batch inference
        # =========================
        for i in tqdm(range(0, len(questions), batch_size),
                      desc=f"Processing question batches for respondent {respondent_number}"):
            batch = questions[i:i + batch_size]
            specific_general_prompt = general_prompt.format(**general_context)
            specific_user_prompt = ""
            for idx, (_, q_text, _, opts_text, _, _) in enumerate(batch, start=1):
                if transliterated:
                    specific_joining_prompt = joining_prompts['en']
                else:
                    specific_joining_prompt = joining_prompts[language_code]
                specific_user_prompt += specific_joining_prompt.format(idx=idx, q_text=q_text, opts_text=opts_text)
            specific_user_prompt += user_prompt

            messages = [
                {"role": "system", "content": specific_general_prompt},
                {"role": "user", "content": specific_user_prompt}
            ]

            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.0,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            raw_results.append({"question_batch": specific_user_prompt, "answer_text": answer_text})

            batch_answers = re.findall(r'Q\d+:\s*(\d+)', answer_text)
            for j, (qsn_key, _, opts_list, _, qsn_instance, is_scale) in enumerate(batch):
                if j < len(batch_answers):
                    ans_str = batch_answers[j]
                    if is_scale:
                        # Scale: expect a number 1–10 directly
                        try:
                            ans_value = int(ans_str)
                            if 1 <= ans_value <= 10:
                                respondent_answers[f"{qsn_key} - {qsn_instance}"] = ans_value
                            else:
                                respondent_answers[f"{qsn_key} - {qsn_instance}"] = "Invalid scale"
                        except ValueError:
                            respondent_answers[f"{qsn_key} - {qsn_instance}"] = "Invalid scale"
                    else:
                        # Categorical: interpret as option index
                        ans_idx = int(ans_str) - 1
                        if 0 <= ans_idx < len(opts_list):
                            respondent_answers[f"{qsn_key} - {qsn_instance}"] = opts_list[ans_idx]
                        else:
                            respondent_answers[f"{qsn_key} - {qsn_instance}"] = "Invalid answer"
                else:
                    respondent_answers[f"{qsn_key} - {qsn_instance}"] = "No answer"

        results.append(respondent_answers)
        debug.append(raw_results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"responses/aya_responses/survey_answers_allstates_{country_name}_{language_code}.csv", index=False)


# ==============================
# AGGREGATE MAJORITY ANSWERS
# ==============================
def get_most_frequent_answers():
    df = pd.read_csv(f"responses/aya_responses/survey_answers_allstates_{country_name}_{language_code}.csv")
    question_cols = [col for col in df.columns if ' - ' in col and col.split(' - ')[0].startswith('Q')]
    question_prefixes = sorted(set(col.split(' - ')[0] for col in question_cols))

    for q in question_prefixes:
        q_cols = [col for col in df.columns if col.startswith(q + ' -')]
        df[q] = df[q_cols].apply(lambda row: row.mode().iloc[0] if not row.mode().empty else None, axis=1)
        df.drop(columns=q_cols, inplace=True)
        
    df.to_csv(f"responses/aya_responses/most_frequent_answers_allstates_{country_name}_{language_code}.csv", index=False)


# ==============================
# MAIN
# ==============================
def main():
    find_responses(df_chosen, tokenizer, model, chosen_cols)
    get_most_frequent_answers()

if __name__ == "__main__":
    main()
