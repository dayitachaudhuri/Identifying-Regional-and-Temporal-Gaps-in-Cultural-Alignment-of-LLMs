import os
import json
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
country = "russia"
year = "2006"
INPUT_DIRECTORY = f"data/{country}/{year}"
INPUT_FILE_NAME = f"{year}_{country}_majority_answers_by_persona"
OUTPUT_DIRECTORY = "data/translated_data"
LANG_CODES = ["ru"]  # Russian
transliterate = True

# === SETUP ===
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

with open(f"data/translated_questions/questions_en.json", "r", encoding="utf-8") as f:
    questions_en = json.load(f)
    
for LANG_CODE in LANG_CODES:
    if transliterate:
        with open(f"data/translated_questions/questions_{LANG_CODE}_transliterated.json", "r", encoding="utf-8") as f:
            questions_bn = json.load(f)
    else:
        with open(f"data/translated_questions/questions_{LANG_CODE}.json", "r", encoding="utf-8") as f:
            questions_bn = json.load(f)

    input_path = os.path.join(INPUT_DIRECTORY, f"{INPUT_FILE_NAME}_en.csv")
    df = pd.read_csv(input_path)

    # Identify columns that correspond to questions
    question_cols = [col for col in df.columns if col.split(":")[0] in questions_bn]
    print(f"Columns to translate ({len(question_cols)}): {question_cols}")

    # Translation logic
    def translate_answer(qcode, answer):
        """Map English option to Bengali equivalent or keep unchanged if scale question."""
        if not isinstance(answer, str) or not answer.strip():
            return answer

        q_en = questions_en.get(qcode)
        q_bn = questions_bn.get(qcode)

        if not q_en or not q_bn:
            return answer

        # If scale-based question, keep answer unchanged
        if q_en.get("scale", False):
            return answer

        # Find index of answer in English options
        try:
            idx = q_en["options"].index(answer.strip())
            return q_bn["options"][idx]
        except (ValueError, IndexError):
            return answer


    # Apply translation
    for col in tqdm(question_cols, desc="Translating columns"):
        qcode = col.split(":")[0]
        df[col] = df[col].apply(lambda x: translate_answer(qcode, x))

    # Save translated output
    if transliterate:
        output_path = os.path.join(OUTPUT_DIRECTORY, f"{INPUT_FILE_NAME}_{LANG_CODE}_transliterated.csv")
    else:
        output_path = os.path.join(OUTPUT_DIRECTORY, f"{INPUT_FILE_NAME}_{LANG_CODE}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {LANG_CODE} CSV: {output_path}")
