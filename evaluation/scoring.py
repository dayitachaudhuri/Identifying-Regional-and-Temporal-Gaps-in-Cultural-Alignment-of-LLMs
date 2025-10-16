import pandas as pd
import numpy as np
import json


def normalize_text(s):
    """
    Normalize text by applying basic normalization and replacements.
    """
    if not isinstance(s, str):
        return s
    # Step 1: basic normalization
    s = s.strip().lower()
    s = s.replace("’", "'").replace("´", "'").replace("`", "'")
    s = s.rstrip('.')
    s = s.rstrip('"')
    s = s.lstrip('"')
    # Step 2: apply replacements
    with open('replacements.json', 'r') as f:
        replacements = json.load(f)['replacements']
    s = replacements.get(s, s)
    return s


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


def compute_metrics(merged_df, selected_questions, num_options_map, region_wise=False, verbose=True):
    """
    Compute hard and soft metrics for the selected questions.
    If region_wise is True, compute metrics for each region separately.
    """
    not_scale_questions = ["Q111", "Q151", "Q152", "Q153", "Q154", "Q155", "Q156", "Q157"]
    if region_wise:
        results_by_region = {}
        unique_regions = merged_df['region'].unique()
        for region in unique_regions:
            region_df = merged_df[merged_df['region'] == region]
            hard_scores, soft_scores = [], []
            for q in selected_questions:
                survey_col, _col = f"{q}_x", f"{q}_y"
                if survey_col not in region_df.columns or _col not in region_df.columns:
                    continue
                survey_answers = pd.to_numeric(region_df[survey_col], errors='coerce')
                model_answers = pd.to_numeric(region_df[_col], errors='coerce')
                valid_idx = (survey_answers.notna()) & (model_answers.notna()) & (survey_answers >= 0)
                if not valid_idx.any():
                    hard_scores.extend([0])
                    soft_scores.extend([0])
                    continue
                survey_answers, model_answers = survey_answers[valid_idx], model_answers[valid_idx]
                num_options = num_options_map.get(q)
                survey_answers, model_answers = np.clip(survey_answers, 1, num_options), np.clip(model_answers, 1, num_options)
                hard_scores.extend(hard_metric(survey_answers, model_answers))
                if q not in not_scale_questions:
                    soft_scores.extend(soft_metric(survey_answers, model_answers, num_options))
                else:
                    soft_scores.extend(hard_metric(survey_answers, model_answers))
            results_by_region[region] = {'hard_metric': np.mean(hard_scores), 'soft_metric': np.mean(soft_scores)}
        return results_by_region
    else:
        hard_scores, soft_scores = [], []
        for q in selected_questions:
            survey_col, _col = f"{q}_x", f"{q}_y"
            survey_answers = pd.to_numeric(merged_df[survey_col], errors='coerce')
            model_answers = pd.to_numeric(merged_df[_col], errors='coerce')
            valid_idx = (survey_answers.notna()) & (model_answers.notna()) & (survey_answers >= 0)
            if not valid_idx.any():
                hard_scores.extend([0])
                soft_scores.extend([0])
                continue
            survey_answers, model_answers = survey_answers[valid_idx], model_answers[valid_idx]
            num_options = num_options_map.get(q)
            survey_answers, model_answers = np.clip(survey_answers, 1, num_options), np.clip(model_answers, 1, num_options)
            if not num_options or num_options <= 1:
                continue
            hard_scores.extend(hard_metric(survey_answers, model_answers))
            if q not in not_scale_questions:
                soft_scores.extend(soft_metric(survey_answers, model_answers, num_options))
            else:
                soft_scores.extend(hard_metric(survey_answers, model_answers))
        results = {}
        if hard_scores: results['hard_metric'] = np.mean(hard_scores)
        if soft_scores: results['soft_metric'] = np.mean(soft_scores)
        return results


def process_questions_config(filepath='../data/questions.json'):
    """
    Process the questions configuration file to create answer mappings and option counts.
    """
    with open(filepath, 'r') as f:
        questions_data = json.load(f)
    answer_mappings = {}
    num_options_map = {}
    for qid, details in questions_data.items():
        if details.get("scale", False):
            num_options_map[qid] = 10
            answer_mappings[qid] = {details["options"][0]: 1, "don't know": 5, details["options"][-1]: 10}
        else:
            valid_options = details["options"]
            num_options_map[qid] = len(valid_options)
            answer_mappings[qid] = {option: i + 1 for i, option in enumerate(valid_options)}
    return answer_mappings, num_options_map


def load_and_normalize_csv(filepath):
    """
    Load a CSV file and normalize its text entries.
    """
    df = pd.read_csv(filepath)
    for col in df.columns:
        df[col] = df[col].apply(normalize_text)
    return df


def replace_answers(df, selected_questions, flat_answer_mapping):
    """
    Replace answers in the dataframe based on the provided flat answer mapping.
    """
    for col in selected_questions:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: flat_answer_mapping.get(' '.join(str(x).split()), x) if isinstance(x, str) else x)
    return df


def map_demographics(wvs_df, country, year, model):
    """
    Map demographic columns to standardized values.
    """
    def to_list(v):
        if isinstance(v, list):
            return v
        else:
            return [v]
        
    n_rows_original = wvs_df.shape[0] 

    if year in ['2012']:
        townsize_map = {
            "all": {
                "under 5,000": ["rural"],
                "5000-20000": ["rural", "urban"],
                "20000-100000": ["rural", "urban"],
                "100000-500000": ["rural", "urban"],
                "500000 and more": ["urban"]
            },
            "japan": {
                "jp: rural districts": "rural",
                "jp:cities with populations less than 100,000": "rural",
                "jp:cities with populations from 100,000 to under 200,000": "urban",
                "jp:cities with populations of 200,000 or more": "urban",
                "jp:18 major large cities": "urban"
            }
        }
        if 'urban_rural' in wvs_df.columns:
            wvs_df['urban_rural'] = wvs_df['urban_rural'].map(
                lambda x: to_list(townsize_map.get(country, townsize_map['all']).get(x, x))
            )
            wvs_df = wvs_df.explode('urban_rural')
            wvs_df['urban_rural'] = wvs_df['urban_rural'].astype(str)
            
    if year in ['2006']:
        townsize_map = {
            "all": {
                "under 5,000": ["rural"],
                "5000-20000": ["rural", "urban"],
                "20000-100000": ["rural", "urban"],
                "100000-500000": ["rural", "urban"],
                "500000 and more": ["urban"]
            },
            "japan": {
                "jp: rural districts": "rural",
                "jp: up to 50,000 residents cities": "rural",
                "jp: 50,000 to 150,000  residents cities": "urban",
                "jp: 150,000 more residents cities": "urban",
                "jp: 12 major large cities(i.e.tokyo,osaka,etc.)": "urban"
            },
            "russia": {
                "ru: rural population": ["rural"],
                "ru: pgt (rural township)": ["rural"],
                "ru: less than 50 tsd": ["rural", "urban"],
                "ru: 50-99,9 tsd": ["rural", "urban"],
                "ru: 100-249,9 tsd": ["rural", "urban"],
                "ru: 250-499,9 tsd": ["rural", "urban"],
                'ru: 500-999,9 tsd': ["rural", "urban"],
                "ru: 1mln. and more": ["urban"]
            }
        }
        if 'urban_rural' in wvs_df.columns:
            wvs_df['urban_rural'] = wvs_df['urban_rural'].map(
                lambda x: to_list(townsize_map.get(country, townsize_map['all']).get(x, x))
            )
            wvs_df = wvs_df.explode('urban_rural')
            wvs_df['urban_rural'] = wvs_df['urban_rural'].astype(str)

    if year in ['2012'] and country == 'japan':
        region_to_prefecture_mapping = {
            'jp: hokkaido region': ['jp-01 hokkaido'],
            'jp: tohoku region': ['jp-02 aomori', 'jp-03 iwate', 'jp-04 miyagi', 'jp-05 akita',
                                'jp-06 yamagata', 'jp-07 fukushima'],
            'jp: kita-kanto region': ['jp-08 ibaraki', 'jp-09 tochigi', 'jp-10 gunma', 'jp-11 saitama'],
            'jp: minami-kanto region': ['jp-12 chiba', 'jp-13 tokyo', 'jp-14 kanagawa'],
            'jp: tokai region': ['jp-22 shizuoka', 'jp-23 aichi', 'jp-24 mie', 'jp-21 gifu'],
            'jp: kinki region': ['jp-25 shiga', 'jp-26 kyoto', 'jp-27 osaka', 'jp-28 hyogo', 'jp-29 nara', 'jp-30 wakayama'],
            'jp: hokuriku, shinetsu region': ['jp-15 niigata', 'jp-16 toyama', 'jp-17 ishikawa', 'jp-18 fukui', 'jp-20 nagano'],
            'jp: shikoku region': ['jp-31 tokushima', 'jp-32 kagawa', 'jp-33 ehime', 'jp-34 kochi'],
            'jp: kyushu region': ['jp-40 fukuoka', 'jp-41 saga', 'jp-42 nagasaki', 'jp-43 kumamoto', 
                                'jp-44 oita', 'jp-45 miyazaki', 'jp-46 kagoshima', 'jp-47 okinawa']
        }
        if 'region' in wvs_df.columns:
            wvs_df['region'] = wvs_df['region'].map(
                lambda x: to_list(region_to_prefecture_mapping.get(x, [x]))
            )
            wvs_df = wvs_df.explode('region')
            wvs_df['region'] = wvs_df['region'].astype(str)
            
    if year in ['2006'] and country == 'japan':
        region_to_prefecture_mapping = {
            'jp: hokkaido/tohoku': [
                'jp-01 hokkaido', 'jp-02 aomori', 'jp-03 iwate', 'jp-04 miyagi', 
                'jp-05 akita', 'jp-06 yamagata', 'jp-07 fukushima'
            ],
            'jp: kanto': [
                'jp-08 ibaraki', 'jp-09 tochigi', 'jp-10 gunma', 'jp-11 saitama', 
                'jp-12 chiba', 'jp-13 tokyo', 'jp-14 kanagawa'
            ],
            'jp: chubu,hokuriku': [
                'jp-21 gifu', 'jp-22 shizuoka', 'jp-23 aichi', 'jp-24 mie', 
                'jp-15 niigata', 'jp-16 toyama', 'jp-17 ishikawa', 'jp-18 fukui', 'jp-20 nagano'
            ],
            'jp: kinki': [
                'jp-25 shiga', 'jp-26 kyoto', 'jp-27 osaka', 'jp-28 hyogo', 
                'jp-29 nara', 'jp-30 wakayama'
            ],
            'jp: chugoku,shikoku,kyushu,okinawa': [
                'jp-31 tottori', 'jp-32 shimane', 'jp-33 okayama', 'jp-34 hiroshima', 'jp-35 yamaguchi',
                'jp-31 tokushima', 'jp-32 kagawa', 'jp-33 ehime', 'jp-34 kochi',
                'jp-40 fukuoka', 'jp-41 saga', 'jp-42 nagasaki', 'jp-43 kumamoto', 'jp-44 oita', 
                'jp-45 miyazaki', 'jp-46 kagoshima', 'jp-47 okinawa' 
            ]
        }
        if 'region' in wvs_df.columns:
            wvs_df['region'] = wvs_df['region'].map(
                lambda x: to_list(region_to_prefecture_mapping.get(x, [x]))
            )
            wvs_df = wvs_df.explode('region')
            wvs_df['region'] = wvs_df['region'].astype(str)
            
    if year in ['2006'] and country == 'US':
        us_region_to_state_mapping = {
            'us: new england': [
                'us-me maine', 'us-nh new hampshire', 'us-vt vermont', 'us-ma massachusetts', 
                'us-ri rhode island', 'us-ct connecticut'
            ],
            'us: middle atlantic states': [
                'us-ny new york', 'us-nj new jersey', 'us-pa pennsylvania'
            ],
            'us: east north central': [
                'us-oh ohio', 'us-in indiana', 'us-il illinois', 'us-mi michigan', 'us-wi wisconsin'
            ],
            'us: west north central': [
                'us-mn minnesota', 'us-ia iowa', 'us-mo missouri', 'us-nd north dakota', 
                'us-sd south dakota', 'us-ne nebraska', 'us-ks kansas'
            ],
            'us: south atlantic': [
                'us-de delaware', 'us-md maryland', 'us-dc district of columbia', 'us-va virginia',
                'us-wv west virginia', 'us-nc north carolina', 'us-sc south carolina', 'us-ga georgia', 'us-fl florida'
            ],
            'us: east south central': [
                'us-ky kentucky', 'us-tn tennessee', 'us-ms mississippi', 'us-al alabama'
            ],
            'us: west south central': [
                'us-ok oklahoma', 'us-tx texas', 'us-ar arkansas', 'us-la louisiana'
            ],
            'us: mountain': [
                'us-id idaho', 'us-mt montana', 'us-wy wyoming', 'us-nv nevada', 
                'us-ut utah', 'us-co colorado', 'us-az arizona', 'us-nm new mexico'
            ],
            'us: pacific': [
                'us-wa washington', 'us-or oregon', 'us-ca california', 'us-ak alaska', 'us-hi hawaii'
            ]
        }
        if 'region' in wvs_df.columns:
            wvs_df['region'] = wvs_df['region'].map(
                lambda x: to_list(us_region_to_state_mapping.get(x, [x]))
            )
            wvs_df = wvs_df.explode('region')
            wvs_df['region'] = wvs_df['region'].astype(str)
            
    if year in ['2006'] and country == 'russia':
        social_class_map = {
            "low": ["lower class", "lower middle class", "middle class", "working class"],
            "middle": ["lower class", "lower middle class", "middle class", "upper middle class", "working class"],
            "high": ["middle class", "upper middle class", "upper class"]
        }
        if 'social_class' in wvs_df.columns:
                wvs_df['social_class'] = wvs_df['social_class'].map(
                    lambda x: to_list(social_class_map.get(x, [x]))
                )
                wvs_df = wvs_df.explode('social_class')
                wvs_df['social_class'] = wvs_df['social_class'].astype(str)

    if year in ['2006', '2012'] and country == 'US':
        wvs_df['urban_rural'] = [['urban', 'rural']] * len(wvs_df)
        wvs_df = wvs_df.explode('urban_rural')
        wvs_df['urban_rural'] = wvs_df['urban_rural'].astype(str)

    if year == '2012' and 'age' in wvs_df.columns:
        age_bins = [0, 15, 24, 34, 44, 54, 64, 100]
        age_labels = ['0-15', '16-24', '25-34', '35-44', '45-54', '55-64', '65+']
        wvs_df['age'] = pd.to_numeric(wvs_df['age'], errors='coerce')
        wvs_df['age'] = pd.cut(wvs_df['age'], bins=age_bins, labels=age_labels, right=True)
    
    if model == 'gemma' and country == 'russia' and year in ['2006', '2012']:
        regions = ['ru-bel belgorodskaya area', 'ru-mow moskva autonomous city',
                   'ru-pri primorskiy kray', 'ru-ros rostovskaya area', 'ru-sve sverdlovskaya area']
        
        # Assign the list of regions to each row, then explode
        wvs_df['region'] = [regions] * len(wvs_df)
        wvs_df = wvs_df.explode('region')
        wvs_df['region'] = wvs_df['region'].astype(str)
        wvs_df = wvs_df.reset_index(drop=True)

    return wvs_df
    
    
def get_demographic_mapping(year='2022', country='all'):
    """
    Get demographic column mapping for a given year.
    """
    with open("../data/chosen_cols_updated.json", "r") as f:
        persona_cols_json = json.load(f)
    if year not in persona_cols_json['persona_cols']:
        raise ValueError(f"No demographic mapping found for year {year}")
    year_mapping = persona_cols_json['persona_cols'][year].get(country, persona_cols_json['persona_cols'][year]['all'])
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


def analyze_survey_alignment(year='2022', mode='state', country='india', state='bengal', language='en', model='llama', region_wise=False, verbose=True):
    """
    Analyze survey alignment between WVS data and model responses.
    1. Load and normalize data
    2. Process WVS data
    3. Replace answers with numeric codes
    4. Map demographics
    5. Ensure merge columns exist
    6. Merge datasets
    7. Compute metrics 
    """
    
    if verbose:
        print(f"Analyzing survey alignment for {country}, year {year}, mode {mode}, state {state}, language {language}")
        
    wvs_filepath=f'../data/{country}/{year}/{year}_{country}_majority_answers_by_persona_{language}.csv'
    if mode == 'state':
        filepath=f'../{model}_responses/most_frequent_answers_{state}_{language}.csv'
    else:
        filepath = f'../{model}_responses/most_frequent_answers_allstates_{country}_{language}.csv'
    questions_filepath=f'../data/translated_questions/questions_{language}.json'
    config_file='../data/chosen_cols_updated.json'
    mapping_file = '../data/qsns_mapping.json'
    
    answer_mappings_by_q, num_options_map = process_questions_config(questions_filepath)
    flat_answer_mapping = {normalize_text(k): v for q_map in answer_mappings_by_q.values() for k,v in q_map.items()}
    
    wvs_df = load_and_normalize_csv(wvs_filepath)
    model_df = load_and_normalize_csv(filepath)
    
    # Rename column names
    rename_map = {col: col.split(':')[0].strip() for col in wvs_df.columns if ':' in col}
    wvs_df.rename(columns=rename_map, inplace=True)
    
    # Get demographic mappings
    demographic_mapping_wvs = get_demographic_mapping(year=year, country=country)
    demographic_mapping_responses = get_demographic_mapping(country=country)
    model_df.rename(columns=demographic_mapping_responses, inplace=True)
    wvs_df.rename(columns=demographic_mapping_wvs, inplace=True)
    
    # Specific mapping of demographics
    wvs_df = map_demographics(wvs_df, country, year, model)
    
    # Year specific processing
    if year != '2022':
        with open(mapping_file, 'r') as f:
            qsns_mapping_data = json.load(f)
        qsns_mapping = qsns_mapping_data.get(str(year), {})
        valid_columns = set()
        rename_map_v_to_q = {}
        for col in wvs_df.columns:
            if col in qsns_mapping and qsns_mapping[col] is not None:
                rename_map_v_to_q[col] = qsns_mapping[col]
                valid_columns.add(col)
            elif col in demographic_mapping_wvs.values():
                valid_columns.add(col)
        wvs_df = wvs_df[list(valid_columns)]
        wvs_df.rename(columns=rename_map_v_to_q, inplace=True)
    
    # Get chosen questions
    with open(config_file, "r") as f:
        data = json.load(f)
    chosen_questions = [q for q, k in data['chosen_cols'].items() if k == True]
    selected_questions = [q for q in chosen_questions if q in wvs_df.columns and q in model_df.columns]
    
    # Replace answers with numeric codes
    wvs_df = replace_answers(wvs_df, selected_questions, flat_answer_mapping)
    model_df = replace_answers(model_df, selected_questions, flat_answer_mapping)
    
    # Merge
    default_values = {
                      'region':'default_region',
                      'urban_rural':'default_rural',
                      'age':'default_age','gender':'default_gender',
                      'marital_status':'default_unmarried',
                      'education_level':'default_education',
                      'social_class':'default_class'
                    }
    merge_columns = list(default_values.keys())
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
    if merged_df.empty: return {}
    if verbose:
        print(f"Number of merged rows: {len(merged_df)}")
    
    # Compute metrics
    return compute_metrics(merged_df, selected_questions, num_options_map, region_wise=region_wise, verbose=verbose)


def analyze_data_alignment(file1, file2, year1, year2, mode='state', country='india', state='bengal', language='en', model='llama', region_wise=False, verbose=True): 

    questions_filepath=f'../data/translated_questions/questions_{language}.json'
    config_file='../data/chosen_cols_updated.json'
    mapping_file = '../data/qsns_mapping.json'
    
    answer_mappings_by_q, num_options_map = process_questions_config(questions_filepath)
    flat_answer_mapping = {normalize_text(k): v for q_map in answer_mappings_by_q.values() for k,v in q_map.items()}
    
    df1 = load_and_normalize_csv(file1)
    df2 = load_and_normalize_csv(file2)
    
    # Rename column names
    rename_map = {col: col.split(':')[0].strip() for col in df1.columns if ':' in col}
    df1.rename(columns=rename_map, inplace=True)  
    rename_map = {col: col.split(':')[0].strip() for col in df2.columns if ':' in col}
    df2.rename(columns=rename_map, inplace=True)

    # Get demographic mappings
    demographic_mapping_1 = get_demographic_mapping(year=year1, country=country)
    df1.rename(columns=demographic_mapping_1, inplace=True)
    demographic_mapping_2 = get_demographic_mapping(year=year2, country=country)
    df2.rename(columns=demographic_mapping_2, inplace=True)
    
    # Specific mapping of demographics
    df1 = map_demographics(df1, country, year1, model)
    df2 = map_demographics(df2, country, year2, model)
    
    # Year specific processing
    if year1 != '2022':
        with open(mapping_file, 'r') as f:
            qsns_mapping_data = json.load(f)
        qsns_mapping = qsns_mapping_data.get(str(year1), {})
        valid_columns = set()
        rename_map_v_to_q = {}
        for col in df1.columns:
            if col in qsns_mapping and qsns_mapping[col] is not None:
                rename_map_v_to_q[col] = qsns_mapping[col]
                valid_columns.add(col)
            elif col in demographic_mapping_1.values():
                valid_columns.add(col)
        df1 = df1[list(valid_columns)]
        df1.rename(columns=rename_map_v_to_q, inplace=True)
    if year2 != '2022':
        with open(mapping_file, 'r') as f:
            qsns_mapping_data = json.load(f)
        qsns_mapping = qsns_mapping_data.get(str(year2), {})
        valid_columns = set()
        rename_map_v_to_q = {}
        for col in df2.columns:
            if col in qsns_mapping and qsns_mapping[col] is not None:
                rename_map_v_to_q[col] = qsns_mapping[col]
                valid_columns.add(col)
            elif col in demographic_mapping_1.values():
                valid_columns.add(col)
        df2 = df2[list(valid_columns)]
        df2.rename(columns=rename_map_v_to_q, inplace=True)
    
    # Get chosen questions
    with open(config_file, "r") as f:
        data = json.load(f)
    chosen_questions = [q for q, k in data['chosen_cols'].items() if k == True]
    selected_questions = [q for q in chosen_questions if q in df1.columns and q in df2.columns]
    
    # Replace answers with numeric codes
    df1 = replace_answers(df1, selected_questions, flat_answer_mapping)
    df2 = replace_answers(df2, selected_questions, flat_answer_mapping)
    
    # Merge
    default_values = {
                      'region':'default_region',
                      'urban_rural':'default_rural',
                      'age':'default_age','gender':'default_gender',
                      'marital_status':'default_unmarried',
                      'education_level':'default_education',
                      'social_class':'default_class'
                    }
    merge_columns = list(default_values.keys())
    if verbose:
        print("Unique values of merge_columns in df1:")
        for col in merge_columns:
            if col in df1.columns:
                print(f"{col}: {df1[col].unique()}")
        print("\nUnique values of merge_columns in df2:")
        for col in merge_columns:
            if col in df2.columns:
                print(f"{col}: {df2[col].unique()}")
                
    merged_df = pd.merge(df1, df2, on=merge_columns, how='inner')
    if merged_df.empty: return {}
    if verbose:
        print(f"Number of merged rows: {len(merged_df)}")
    
    # Compute metrics
    return compute_metrics(merged_df, selected_questions, num_options_map, region_wise=region_wise, verbose=verbose)


def load_themes(filepath='../data/themes.json'):
    """
    Load themes and convert ranges into question lists.
    """
    with open(filepath, 'r') as f:
        themes_data = json.load(f)
    themes_map = {}
    for key, theme_name in themes_data.items():
        start, end = map(int, key.split('-'))
        themes_map[theme_name] = [f"Q{i}" for i in range(start, end + 1)]
    return themes_map


def compute_similarity_per_theme(
    year='2022',
    mode='state',
    country='india',
    state='bengal',
    language='en',
    metric_type='soft',
    model='llama',
    region_wise=False,
    verbose=True
):
    """
    Compute the similarity per theme (cleaning and merging aligned with analyze_survey_alignment)
    """
    # File paths (use translated questions like analyze_survey_alignment)
    wvs_filepath=f'../data/{country}/{year}/{year}_{country}_majority_answers_by_persona_{language}.csv'
    if mode == 'state':
        model_filepath=f'../{model}_responses/most_frequent_answers_{state}_{language}.csv'
    else:
        model_filepath=f'../{model}_responses/most_frequent_answers_allstates_{country}_{language}.csv'

    questions_filepath=f'../data/translated_questions/questions_{language}.json'
    config_filepath='../data/chosen_cols_updated.json'
    mapping_file = '../data/qsns_mapping.json'
    themes_filepath='../data/themes.json'

    # Load mappings and theme definitions
    answer_mappings_by_q, num_options_map = process_questions_config(questions_filepath)
    flat_answer_mapping = {normalize_text(k): v for q_map in answer_mappings_by_q.values() for k, v in q_map.items()}
    themes_map = load_themes(themes_filepath)

    # Load and normalize CSVs
    wvs_df = load_and_normalize_csv(wvs_filepath)
    model_df = load_and_normalize_csv(model_filepath)

    # Rename columns with ':' in WVS only (keep model columns as-is to match analyze_survey_alignment)
    rename_map = {col: col.split(':')[0].strip() for col in wvs_df.columns if ':' in col}
    wvs_df.rename(columns=rename_map, inplace=True)
    # NOTE: do NOT strip ':' from model_df column names here — preserve original model_df column names

    # Year/country specific demographic column mapping (mirror analyze_survey_alignment)
    demographic_mapping_wvs = get_demographic_mapping(year=year, country=country)
    demographic_mapping_model = get_demographic_mapping(country=country)
    wvs_df.rename(columns=demographic_mapping_wvs, inplace=True)
    model_df.rename(columns=demographic_mapping_model, inplace=True)

    # Map demographics in WVS (explode / normalize)
    wvs_df = map_demographics(wvs_df, country, year, model)

    # For non-2022 years, apply qsns mapping to WVS columns (same logic as analyze_survey_alignment)
    if year != '2022':
        with open(mapping_file, 'r') as f:
            qsns_mapping_data = json.load(f)
        qsns_mapping = qsns_mapping_data.get(str(year), {})
        valid_columns = set()
        rename_map_v_to_q = {}
        for col in wvs_df.columns:
            if col in qsns_mapping and qsns_mapping[col] is not None:
                rename_map_v_to_q[col] = qsns_mapping[col]
                valid_columns.add(col)
            elif col in demographic_mapping_wvs.values():
                valid_columns.add(col)
        # keep only valid columns and rename question columns
        wvs_df = wvs_df[list(valid_columns)]
        wvs_df.rename(columns=rename_map_v_to_q, inplace=True)

    # ---------------------------
    # Prepare selected questions and replace answers
    # ---------------------------
    with open(config_filepath, "r") as f:
        data = json.load(f)
    chosen_questions = [q for q, k in data['chosen_cols'].items() if k == True]
    selected_questions = [q for q in chosen_questions if q in wvs_df.columns and q in model_df.columns]

    if verbose:
        print("Selected questions (present in both):", selected_questions)
        print("WVS columns (sample):", list(wvs_df.columns)[:30])
        print("Model columns (sample):", list(model_df.columns)[:30])

    # Replace answers with numeric codes
    wvs_df = replace_answers(wvs_df, selected_questions, flat_answer_mapping)
    model_df = replace_answers(model_df, selected_questions, flat_answer_mapping)

    # Merge on demographics (same defaults as analyze_survey_alignment)
    default_values = {
        'region':'default_region',
        'urban_rural':'default_rural',
        'age':'default_age','gender':'default_gender',
        'marital_status':'default_unmarried',
        'education_level':'default_education',
        'social_class':'default_class'
    }
    merge_columns = list(default_values.keys())

    if verbose:
        print("Merge columns expected:", merge_columns)
        print("Columns present in WVS for merge:", [c for c in merge_columns if c in wvs_df.columns])
        print("Columns present in model for merge:", [c for c in merge_columns if c in model_df.columns])
    merged_df = pd.merge(wvs_df, model_df, on=merge_columns, how='inner')
    if merged_df.empty:
        if verbose:
            print("Merged dataframe is empty — no overlapping rows on merge columns. Check above diagnostics.")
        return {}
    if verbose:
        print(f"Number of merged rows: {len(merged_df)}")

    # ---------------------------
    # Compute per-question similarity
    # ---------------------------
    not_scale_questions = ["Q111", "Q151", "Q152", "Q153", "Q154", "Q155", "Q156", "Q157"]
    per_question_scores = {}

    for q in selected_questions:
        survey_col = f"{q}_x"
        model_col = f"{q}_y"
        if survey_col not in merged_df.columns or model_col not in merged_df.columns:
            continue
        survey_answers = pd.to_numeric(merged_df[survey_col], errors='coerce')
        model_answers = pd.to_numeric(merged_df[model_col], errors='coerce')
        valid_idx = survey_answers.notna() & model_answers.notna() & (survey_answers >= 0)
        if not valid_idx.any():
            continue
        survey_answers = survey_answers[valid_idx]
        model_answers = model_answers[valid_idx]

        num_options = num_options_map.get(q)
        if not num_options or num_options <= 1:
            continue

        survey_answers = np.clip(survey_answers, 1, num_options)
        model_answers = np.clip(model_answers, 1, num_options)

        if metric_type == 'hard' or q in not_scale_questions:
            scores = hard_metric(survey_answers, model_answers)
        else:
            scores = soft_metric(survey_answers, model_answers, num_options)

        per_question_scores[q] = float(np.mean(scores))

    # ---------------------------
    # Compute per-theme similarity
    # ---------------------------
    per_theme_scores = {}
    per_theme_counts = {}

    for theme_name, questions in themes_map.items():
        scores = [per_question_scores[q] for q in questions if q in per_question_scores]
        per_theme_counts[theme_name] = len(scores)
        if scores:
            per_theme_scores[theme_name] = float(np.mean(scores))
        else:
            per_theme_scores[theme_name] = np.nan

    return {
        'per_question': per_question_scores,
        'per_theme': per_theme_scores,
        'per_theme_counts': per_theme_counts
    }
