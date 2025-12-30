# © 2025 Dayita Chaudhuri and Velagapudi Athul
# All rights reserved. Joint work.

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Union, Optional

from demographic_mappings import (
    TOWNSIZE_MAP_2012,
    TOWNSIZE_MAP_2006,
    JAPAN_REGION_TO_PREFECTURE_2012,
    JAPAN_REGION_TO_PREFECTURE_2006,
    EGYPT_REGION_TO_REGION_2012,
    US_REGION_TO_STATE_2006,
    RUSSIA_SOCIAL_CLASS_2006,
    COLOMBIA_REGIONS_2006,
    RUSSIA_REGIONS_GEMMA
)


# Constants
REPLACEMENTS_FILE = 'replacements.json'
QUESTIONS_FILE_TEMPLATE = '../data/translated_questions/questions_{language}.json'
QUESTIONS_FILE_DEFAULT = '../data/questions.json'
CONFIG_FILE = '../data/chosen_cols_updated.json'
MAPPING_FILE = '../data/qsns_mapping.json'
THEMES_FILE = '../data/themes.json'

NON_SCALE_QUESTIONS = ["Q111", "Q151", "Q152", "Q153", "Q154", "Q155", "Q156", "Q157"]

DEFAULT_DEMOGRAPHIC_VALUES = {
    'region': 'default_region',
    'urban_rural': 'default_rural',
    'age': 'default_age',
    'gender': 'default_gender',
    'marital_status': 'default_unmarried',
    'education_level': 'default_education',
    'social_class': 'default_class'
}

AGE_BINS = [0, 15, 24, 34, 44, 54, 64, 100]
AGE_LABELS = ['0-15', '16-24', '25-34', '35-44', '45-54', '55-64', '65+']

# =======================================
# Data preprocessing functions
# =======================================

def normalize_text(s):
    """Normalize text by applying basic normalization and replacements."""
    if not isinstance(s, str):
        return s
    
    # Basic normalization
    s = s.strip().lower()
    s = s.replace("'", "'").replace("´", "'").replace("`", "'")
    s = s.rstrip('.').rstrip('"').lstrip('"')
    
    # Apply replacements from file
    with open(REPLACEMENTS_FILE, 'r') as f:
        replacements = json.load(f)['replacements']
    
    return replacements.get(s, s)

def _prepare_dataframes_for_analysis(wvs_df, model_df, year, country, model):
    """Common preprocessing for WVS and model dataframes."""
    # Rename columns with ':' in WVS data
    rename_map = {col: col.split(':')[0].strip() for col in wvs_df.columns if ':' in col}
    wvs_df.rename(columns=rename_map, inplace=True)
    
    # Get and apply demographic mappings
    demographic_mapping_wvs = get_demographic_mapping(year=year, country=country)
    demographic_mapping_model = get_demographic_mapping(country=country)
    wvs_df.rename(columns=demographic_mapping_wvs, inplace=True)
    model_df.rename(columns=demographic_mapping_model, inplace=True)
    
    # Map demographics (explode, normalize)
    wvs_df = map_demographics(wvs_df, country, year, model)
    
    return wvs_df, model_df, demographic_mapping_wvs


def _apply_year_specific_mapping(df, year, demographic_mapping_values):
    """ Apply year-specific question mappings for non-2022 data."""
    if year == '2022':
        return df
    
    with open(MAPPING_FILE, 'r') as f:
        qsns_mapping_data = json.load(f)
    
    qsns_mapping = qsns_mapping_data.get(str(year), {})
    valid_columns = set()
    rename_map_v_to_q = {}
    for col in df.columns:
        if col in qsns_mapping and qsns_mapping[col] is not None:
            rename_map_v_to_q[col] = qsns_mapping[col]
            valid_columns.add(col)
        elif col in demographic_mapping_values:
            valid_columns.add(col)
    valid_columns.add('urban_rural')
    df = df[list(valid_columns)]
    df.rename(columns=rename_map_v_to_q, inplace=True)
    
    return df


def _get_selected_questions(wvs_df, model_df):
    """ Get list of selected questions present in both dataframes."""
    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)
    chosen_questions = [q for q, k in data['chosen_cols'].items() if k == True]
    selected_questions = [q for q in chosen_questions if q in wvs_df.columns and q in model_df.columns]
    return selected_questions


def load_themes(filepath=None):
    """Load themes and convert ranges into question lists."""
    if filepath is None:
        filepath = THEMES_FILE
        
    with open(filepath, 'r') as f:
        themes_data = json.load(f)
    
    themes_map = {}
    for key, theme_name in themes_data.items():
        start, end = map(int, key.split('-'))
        themes_map[theme_name] = [f"Q{i}" for i in range(start, end + 1)]
    
    return themes_map

# =======================================
# Metric calculations
# =======================================

def hard_metric(survey_answers, model_answers):
    """Calculate the hard metric (exact match)."""
    scores = (survey_answers == model_answers).astype(int)
    return scores


def soft_metric(survey_answers, model_answers, num_options):
    """Calculate the soft metric (normalized error)."""
    error = np.abs(survey_answers - model_answers)
    normalized_error = error / (num_options - 1)
    scores = 1 - normalized_error
    return scores


def _compute_scores_for_question(df, question, num_options_map):
    """Compute hard and soft scores for a single question."""
    survey_col = f"{question}_x"
    model_col = f"{question}_y"
    
    if survey_col not in df.columns or model_col not in df.columns:
        return [], []
    
    survey_answers = pd.to_numeric(df[survey_col], errors='coerce')
    model_answers = pd.to_numeric(df[model_col], errors='coerce')
    valid_idx = (survey_answers.notna()) & (model_answers.notna()) & (survey_answers >= 0)
    
    if not valid_idx.any():
        return [0], [0]
    
    survey_answers = survey_answers[valid_idx]
    model_answers = model_answers[valid_idx]
    num_options = num_options_map.get(question)
    
    if not num_options or num_options <= 1:
        return [], []
    
    survey_answers = np.clip(survey_answers, 1, num_options)
    model_answers = np.clip(model_answers, 1, num_options)
    
    hard_scores = hard_metric(survey_answers, model_answers)
    
    if question not in NON_SCALE_QUESTIONS:
        soft_scores = soft_metric(survey_answers, model_answers, num_options)
    else:
        soft_scores = hard_metric(survey_answers, model_answers)
    
    return list(hard_scores), list(soft_scores)


def compute_metrics(merged_df, selected_questions, num_options_map, region_wise=False, verbose=True):
    """Compute hard and soft metrics for the selected questions."""
    if region_wise:
        results_by_region = {}
        unique_regions = merged_df['region'].unique()
        
        for region in unique_regions:
            region_df = merged_df[merged_df['region'] == region]
            hard_scores, soft_scores = [], []
            
            for question in selected_questions:
                h_scores, s_scores = _compute_scores_for_question(region_df, question, num_options_map)
                hard_scores.extend(h_scores)
                soft_scores.extend(s_scores)
            
            results_by_region[region] = {
                'hard_metric': np.mean(hard_scores) if hard_scores else 0,
                'soft_metric': np.mean(soft_scores) if soft_scores else 0
            }
        
        return results_by_region
    else:
        hard_scores, soft_scores = [], []
        
        for question in selected_questions:
            h_scores, s_scores = _compute_scores_for_question(merged_df, question, num_options_map)
            hard_scores.extend(h_scores)
            soft_scores.extend(s_scores)
        
        results = {}
        if hard_scores:
            results['hard_metric'] = np.mean(hard_scores)
        if soft_scores:
            results['soft_metric'] = np.mean(soft_scores)
        
        return results

# =======================================
#  ANSWER PROCESSING
# =======================================

def process_questions_config(filepath=None):
    """Process the questions configuration file to create answer mappings and option counts."""
    if filepath is None:
        filepath = QUESTIONS_FILE_DEFAULT
        
    with open(filepath, 'r') as f:
        questions_data = json.load(f)
    answer_mappings = {}
    num_options_map = {}
    for qid, details in questions_data.items():
        if details.get("scale", False):
            num_options_map[qid] = 10
            answer_mappings[qid] = {
                details["options"][0]: 1,
                "don't know": 5,
                details["options"][-1]: 10
            }
        else:
            valid_options = details["options"]
            num_options_map[qid] = len(valid_options)
            answer_mappings[qid] = {option: i + 1 for i, option in enumerate(valid_options)}
    return answer_mappings, num_options_map


def load_and_normalize_csv(filepath):
    """ Load a CSV file and normalize its text entries. """
    df = pd.read_csv(filepath)
    for col in df.columns:
        df[col] = df[col].apply(normalize_text)
    return df


def replace_answers(df, selected_questions, flat_answer_mapping):
    """Replace text answers with numeric codes based on the provided mapping."""
    for col in selected_questions:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: flat_answer_mapping.get(' '.join(str(x).split()), x) 
                if isinstance(x, str) else x
            )
    return df


def _to_list(value):
    """Convert value to list if it isn't already."""
    return value if isinstance(value, list) else [value]


def _get_townsize_map_2012(country):
    """Get townsize mapping for 2012 data."""
    return TOWNSIZE_MAP_2012.get(country, TOWNSIZE_MAP_2012['all'])


def _get_townsize_map_2006(country):
    """Get townsize mapping for 2006 data."""
    return TOWNSIZE_MAP_2006.get(country, TOWNSIZE_MAP_2006['all'])


def _apply_urban_rural_mapping(df, mapping):
    """Apply urban/rural mapping and explode results."""
    if 'urban_rural' in df.columns:
        df['urban_rural'] = df['urban_rural'].map(lambda x: _to_list(mapping.get(x, x)))
        df = df.explode('urban_rural')
        df['urban_rural'] = df['urban_rural'].astype(str)
    return df


def _set_default_urban_rural(df):
    """Set default urban/rural values (both urban and rural)."""
    df['urban_rural'] = [['urban', 'rural']] * len(df)
    df = df.explode('urban_rural')
    df['urban_rural'] = df['urban_rural'].astype(str)
    return df


def map_demographics(wvs_df, country, year, model):
    """Map demographic columns to standardized values based on country and year.""" 

    # =================
    # 2012 mappings
    # =================
    if str(year) in ['2012']:
        townsize_map = _get_townsize_map_2012(country)
        wvs_df = _apply_urban_rural_mapping(wvs_df, townsize_map)

    if str(year) in ['2012'] and country == 'japan':
        if 'region' in wvs_df.columns:
            wvs_df['region'] = wvs_df['region'].map(
                lambda x: _to_list(JAPAN_REGION_TO_PREFECTURE_2012.get(x, [x]))
            )
            wvs_df = wvs_df.explode('region')
            wvs_df['region'] = wvs_df['region'].astype(str)
    
    if str(year) in ['2012'] and country in ['US', 'egypt', 'russia']:
        wvs_df = _set_default_urban_rural(wvs_df)
        
    if str(year) == '2012' and 'age' in wvs_df.columns:
        wvs_df['age'] = pd.to_numeric(wvs_df['age'], errors='coerce')
        wvs_df['age'] = pd.cut(wvs_df['age'], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        
    if str(year) == '2012' and country == 'egypt':
        if 'region' in wvs_df.columns:
            wvs_df['region'] = wvs_df['region'].map(EGYPT_REGION_TO_REGION_2012)
            wvs_df['region'] = wvs_df['region'].astype(str)
        
    # =================
    # 2006 mappings
    # =================
            
    if str(year) in ['2006']:
        townsize_map = _get_townsize_map_2006(country)
        wvs_df = _apply_urban_rural_mapping(wvs_df, townsize_map)
            
    if str(year) in ['2006'] and country == 'japan':
        if 'region' in wvs_df.columns:
            wvs_df['region'] = wvs_df['region'].map(
                lambda x: _to_list(JAPAN_REGION_TO_PREFECTURE_2006.get(x, [x]))
            )
            wvs_df = wvs_df.explode('region')
            wvs_df['region'] = wvs_df['region'].astype(str)
            
    if str(year) in ['2006'] and country == 'US':
        if 'region' in wvs_df.columns:
            wvs_df['region'] = wvs_df['region'].map(
                lambda x: _to_list(US_REGION_TO_STATE_2006.get(x, [x]))
            )
            wvs_df = wvs_df.explode('region')
            wvs_df['region'] = wvs_df['region'].astype(str)
            
    if str(year) in ['2006'] and country == 'russia':
        if 'social_class' in wvs_df.columns:
                wvs_df['social_class'] = wvs_df['social_class'].map(
                    lambda x: _to_list(RUSSIA_SOCIAL_CLASS_2006.get(x, [x]))
                )
                wvs_df = wvs_df.explode('social_class')
                wvs_df['social_class'] = wvs_df['social_class'].astype(str)

    if str(year) in ['2006'] and country in ['US', 'colombia']:
        wvs_df = _set_default_urban_rural(wvs_df)
        
    if str(year) in ['2006'] and country in ['colombia']:
        # Set default regions for Colombia in 2006
        wvs_df['region'] = [COLOMBIA_REGIONS_2006] * len(wvs_df)
        wvs_df = wvs_df.explode('region')
        wvs_df['region'] = wvs_df['region'].astype(str)
        wvs_df = _set_default_urban_rural(wvs_df)
            
    if model == 'gemma' and country == 'russia' and year in ['2006', '2012']:
        # Assign the list of regions to each row, then explode
        wvs_df['region'] = [RUSSIA_REGIONS_GEMMA] * len(wvs_df)
        wvs_df = wvs_df.explode('region')
        wvs_df['region'] = wvs_df['region'].astype(str)
        wvs_df = wvs_df.reset_index(drop=True)

    return wvs_df
    
    
def get_demographic_mapping(year='2022', country='all'):
    """Get demographic column mapping for a given year and country."""
    with open(CONFIG_FILE, "r") as f:
        persona_cols_json = json.load(f)
    if year not in persona_cols_json['persona_cols']:
        raise ValueError(f"No demographic mapping found for year {year}")

    year_mapping = persona_cols_json['persona_cols'][year].get(
        country, 
        persona_cols_json['persona_cols'][year]['all']
    )
    
    demographic_mapping = {}
    for key, val in year_mapping.items():
        col_name = val.split(":")[0].strip()
        if key == 'sex':
            final_key = 'gender'
        elif key == 'education':
            final_key = 'education_level'
        else:
            final_key = key    
        demographic_mapping[col_name] = final_key
    return demographic_mapping

# =======================================
#  MAIN ANALYSIS FUNCTIONS
# =======================================

def analyze_survey_alignment(year='2022', country='india', language='en', model='llama', region_wise=False, verbose=True):
    """Analyze survey alignment between WVS data and model responses."""
    if verbose:
        print(f"Analyzing survey alignment for {country}, year {year}, language {language}")

    # Construct file paths
    wvs_filepath = f'../data/{country}/{year}/{year}_{country}_majority_answers_by_persona_{language}.csv'
    model_filepath = f'../{model}_responses/most_frequent_answers_allstates_{country}_{language}.csv' 
    questions_filepath = QUESTIONS_FILE_TEMPLATE.format(language=language)
    
    # Process question configuration
    answer_mappings_by_q, num_options_map = process_questions_config(questions_filepath)
    flat_answer_mapping = {normalize_text(k): v 
                          for q_map in answer_mappings_by_q.values() 
                          for k, v in q_map.items()}
    
    # Load and normalize data
    wvs_df = load_and_normalize_csv(wvs_filepath)
    model_df = load_and_normalize_csv(model_filepath)
    
    # Prepare dataframes (rename, map demographics)
    wvs_df, model_df, demographic_mapping_wvs = _prepare_dataframes_for_analysis(
        wvs_df, model_df, year, country, model
    )
    
    # Apply year-specific question mappings
    wvs_df = _apply_year_specific_mapping(wvs_df, year, set(demographic_mapping_wvs.values()))
    
    # Get selected questions and replace answers with numeric codes
    selected_questions = _get_selected_questions(wvs_df, model_df)
    wvs_df = replace_answers(wvs_df, selected_questions, flat_answer_mapping)
    model_df = replace_answers(model_df, selected_questions, flat_answer_mapping)
    
    # Merge datasets
    merge_columns = list(DEFAULT_DEMOGRAPHIC_VALUES.keys())
    if verbose:
        print("Unique values of merge_columns in wvs_df:")
        for col in merge_columns:
            if col in wvs_df.columns:
                print(f"{col}: {wvs_df[col].unique()}")
        print("\nUnique values of merge_columns in model_df:")
        for col in merge_columns:
            if col in model_df.columns:
                print(f"{col}: {model_df[col].unique()}")
    merged_df = pd.merge(wvs_df, model_df, on=merge_columns, how='inner')
    
    if merged_df.empty:
        return {}
    
    if verbose:
        print(f"Number of merged rows: {len(merged_df)}")
    
    # Compute and return metrics
    return compute_metrics(merged_df, selected_questions, num_options_map, 
                          region_wise=region_wise, verbose=verbose)


def compute_similarity_per_theme(year='2022', country='india', language='en', metric_type='soft', model='llama',  region_wise=False, verbose=True):
    """Compute similarity scores per question and aggregate by theme."""
    # Construct file paths
    wvs_filepath = f'../data/{country}/{year}/{year}_{country}_majority_answers_by_persona_{language}.csv'
    model_filepath = f'../{model}_responses/most_frequent_answers_allstates_{country}_{language}.csv'
    questions_filepath = QUESTIONS_FILE_TEMPLATE.format(language=language)

    # Load mappings and theme definitions
    answer_mappings_by_q, num_options_map = process_questions_config(questions_filepath)
    flat_answer_mapping = {normalize_text(k): v 
                          for q_map in answer_mappings_by_q.values() 
                          for k, v in q_map.items()}
    themes_map = load_themes()

    # Load and normalize data
    wvs_df = load_and_normalize_csv(wvs_filepath)
    model_df = load_and_normalize_csv(model_filepath)

    # Prepare dataframes
    wvs_df, model_df, demographic_mapping_wvs = _prepare_dataframes_for_analysis(
        wvs_df, model_df, year, country, model
    )

    # Apply year-specific mappings
    wvs_df = _apply_year_specific_mapping(wvs_df, year, set(demographic_mapping_wvs.values()))

    # Get selected questions and replace answers
    selected_questions = _get_selected_questions(wvs_df, model_df)

    if verbose:
        print("Selected questions (present in both):", selected_questions)
        print("WVS columns (sample):", list(wvs_df.columns)[:30])
        print("Model columns (sample):", list(model_df.columns)[:30])

    wvs_df = replace_answers(wvs_df, selected_questions, flat_answer_mapping)
    model_df = replace_answers(model_df, selected_questions, flat_answer_mapping)

    # Merge datasets
    merge_columns = list(DEFAULT_DEMOGRAPHIC_VALUES.keys())

    if verbose:
        print("Merge columns expected:", merge_columns)
        print("Columns present in WVS for merge:", [c for c in merge_columns if c in wvs_df.columns])
        print("Columns present in model for merge:", [c for c in merge_columns if c in model_df.columns])
    
    merged_df = pd.merge(wvs_df, model_df, on=merge_columns, how='inner')
    
    if merged_df.empty:
        if verbose:
            print("Merged dataframe is empty — no overlapping rows on merge columns.")
        return {}
    
    if verbose:
        print(f"Number of merged rows: {len(merged_df)}")

    # Region-wise computation if requested
    if region_wise:
        results_by_region = {}
        unique_regions = merged_df['region'].unique()
        
        for region in unique_regions:
            if verbose:
                print(f"\nProcessing region: {region}")
            
            region_df = merged_df[merged_df['region'] == region]
            
            # Compute per-question similarity for this region
            per_question_scores = {}
            
            for question in selected_questions:
                survey_col = f"{question}_x"
                model_col = f"{question}_y"
                
                if survey_col not in region_df.columns or model_col not in region_df.columns:
                    continue
                
                survey_answers = pd.to_numeric(region_df[survey_col], errors='coerce')
                model_answers = pd.to_numeric(region_df[model_col], errors='coerce')
                valid_idx = survey_answers.notna() & model_answers.notna() & (survey_answers >= 0)
                
                if not valid_idx.any():
                    continue
                
                survey_answers = survey_answers[valid_idx]
                model_answers = model_answers[valid_idx]

                num_options = num_options_map.get(question)
                if not num_options or num_options <= 1:
                    continue

                survey_answers = np.clip(survey_answers, 1, num_options)
                model_answers = np.clip(model_answers, 1, num_options)

                if metric_type == 'hard' or question in NON_SCALE_QUESTIONS:
                    scores = hard_metric(survey_answers, model_answers)
                else:
                    scores = soft_metric(survey_answers, model_answers, num_options)

                per_question_scores[question] = float(np.mean(scores))

            # Compute per-theme similarity for this region
            per_theme_scores = {}
            per_theme_counts = {}

            for theme_name, questions in themes_map.items():
                scores = [per_question_scores[q] for q in questions if q in per_question_scores]
                per_theme_counts[theme_name] = len(scores)
                per_theme_scores[theme_name] = float(np.mean(scores)) if scores else np.nan

            results_by_region[region] = {
                'per_question': per_question_scores,
                'per_theme': per_theme_scores,
                'per_theme_counts': per_theme_counts
            }
        
        return results_by_region
    
    else:
        # Compute per-question similarity (overall)
        per_question_scores = {}

        for question in selected_questions:
            survey_col = f"{question}_x"
            model_col = f"{question}_y"
            
            if survey_col not in merged_df.columns or model_col not in merged_df.columns:
                continue
            
            survey_answers = pd.to_numeric(merged_df[survey_col], errors='coerce')
            model_answers = pd.to_numeric(merged_df[model_col], errors='coerce')
            valid_idx = survey_answers.notna() & model_answers.notna() & (survey_answers >= 0)
            
            if not valid_idx.any():
                continue
            
            survey_answers = survey_answers[valid_idx]
            model_answers = model_answers[valid_idx]

            num_options = num_options_map.get(question)
            if not num_options or num_options <= 1:
                continue

            survey_answers = np.clip(survey_answers, 1, num_options)
            model_answers = np.clip(model_answers, 1, num_options)

            if metric_type == 'hard' or question in NON_SCALE_QUESTIONS:
                scores = hard_metric(survey_answers, model_answers)
            else:
                scores = soft_metric(survey_answers, model_answers, num_options)

            per_question_scores[question] = float(np.mean(scores))

        # Compute per-theme similarity (overall)
        per_theme_scores = {}
        per_theme_counts = {}

        for theme_name, questions in themes_map.items():
            scores = [per_question_scores[q] for q in questions if q in per_question_scores]
            per_theme_counts[theme_name] = len(scores)
            per_theme_scores[theme_name] = float(np.mean(scores)) if scores else np.nan

        return {
            'per_question': per_question_scores,
            'per_theme': per_theme_scores,
            'per_theme_counts': per_theme_counts
        }
