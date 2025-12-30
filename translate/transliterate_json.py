# © 2025 Dayita Chaudhuri and Velagapudi Athul
# All rights reserved. Joint work.

import os
import json
from tqdm import tqdm
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from arabic2latin import arabic_to_latin
import pykakasi

INPUT_DIRECTORY = "data/translated_questions"
OUTPUT_DIRECTORY = "data/translated_questions"
LANGUAGES = ["ar"]

MAPPING = {
    "en": sanscript.ITRANS,
    "mr": sanscript.DEVANAGARI,
    "pa": sanscript.GURMUKHI,
    "hi": sanscript.DEVANAGARI,
    "te": sanscript.TELUGU,
    "bn": sanscript.BENGALI
}

kks = pykakasi.kakasi()

def count_strings(obj):
    """Count total number of strings in JSON-like object"""
    if isinstance(obj, dict):
        return sum(count_strings(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_strings(elem) for elem in obj)
    elif isinstance(obj, str):
        return 1
    else:
        return 0

def translate_json(obj, target_lang, pbar=None):
    """Recursively translate all string fields in JSON-like object"""
    if isinstance(obj, dict):
        return {k: translate_json(v, target_lang, pbar) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [translate_json(elem, target_lang, pbar) for elem in obj]
    elif isinstance(obj, str):
        if pbar:
            pbar.update(1)
        if target_lang in MAPPING.keys():
            response = transliterate(obj, MAPPING.get(target_lang), sanscript.ITRANS)
        elif target_lang == "ar":
            response = arabic_to_latin(obj)
        elif target_lang == "ja":
            result = kks.convert(obj)
            response = " ".join([item['hepburn'] for item in result])
        else:
            response = obj
        return response
    else:
        return obj
    
def main():
    for lang in LANGUAGES:
        with open(f"{INPUT_DIRECTORY}/questions_{lang}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        total = count_strings(data)
        with tqdm(total=total, desc="Translating") as pbar:
            translated_data = translate_json(data, lang, pbar)
        with open(os.path.join(OUTPUT_DIRECTORY, f"questions_{lang}_transliterated.json"), "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
    