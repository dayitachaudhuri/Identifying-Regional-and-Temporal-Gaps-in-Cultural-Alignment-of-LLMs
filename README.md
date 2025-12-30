# Cultural Alignment of LLMs with World Values Survey Data

A research project investigating how well Large Language Models (LLMs) align with cultural values across different countries, languages, and demographic groups using the World Values Survey (WVS) dataset.

## 📄 License and Usage

© 2025 Dayita Chaudhuri and Velagapudi Athul. All rights reserved.

This repository contains original joint research observations. Reuse, redistribution, or derivative work is not permitted without prior written consent from both authors.

This project uses World Values Survey data, which has its own terms of use. Please refer to the [WVS website](https://www.worldvaluessurvey.org/) for data usage guidelines.

## 📋 Project Overview

This project evaluates whether LLMs can accurately represent diverse cultural perspectives by comparing their responses to questions from the World Values Survey with actual human responses. The study examines multiple dimensions:

- **Cross-cultural alignment**: How well do LLMs represent values from different countries?
- **Temporal alignment:** Do LLMs align better with Contemporary or past values?
- **Multilingual capabilities**: Does multilingual prompting improve cultural alignment compared to English-only prompts?

## 🗂️ Project Structure

```
.
├── data/                           # World Values Survey data
│   ├── bangladesh/, colombia/, egypt/, india/, japan/, russia/, US/
│   ├── config_data/               # Configuration templates
│   ├── translated_data/           # Translated survey responses
│   └── translated_questions/      # Questions in multiple languages
├── prompt_model/                  # LLM prompting implementations
│   ├── aya/                       # Aya-Expanse-32B model
│   ├── gemma/                     # Google Gemma model
│   ├── gemma-chat/                # Gemma chat variant
│   ├── llama2-chat/               # Meta Llama-2-Chat-13B model
│   └── qwen/                      # Qwen model
├── responses/                     # Model-generated responses
│   ├── aya_responses/
│   ├── gemma_responses/
│   ├── llama_responses/
│   └── llama3_responses/
├── evaluation/                    # Evaluation scripts and notebooks
│   ├── scoring.py                # Main evaluation metrics
│   ├── distribution_comparison.py
│   ├── themes_and_temporal.ipynb # Theme-based analysis
│   └── languages.ipynb           # Language comparison
├── translate/                     # Translation utilities
└── clean_data_wvs.ipynb          # Data preprocessing notebook
```

## 🌍 Datasets

### Countries Covered

- **Asia**: India, Japan, Bangladesh
- **Middle East**: Egypt
- **Americas**: United States, Colombia
- **Europe**: Russia

### Surveys Used

World Values Survey data from multiple waves (2000-2022), covering:

- **Themes**: Social values, happiness, trust, economic values, corruption, migration, security, science & technology, religious values, ethical values, political values
- **Demographics**: Region, urban/rural, age, gender, marital status, education level, social class, language

### Languages

English (en), Bengali (bn), Hindi (hi), Marathi (mr), Telugu (te), Punjabi (pa), Japanese (ja), Russian (ru), Arabic (ar), Spanish (es)

## 🤖 Models Evaluated

1. **Aya-Expanse-32B** (CohereLabs)
2. **Gemma-3-IT-12B** (Google)
3. **Llama-2-Chat-13B** (Meta)
4. **Qwen-3-32B** (Alibaba)

All models were evaluated using:

- 4-bit quantization for efficient inference
- Persona-based prompting with demographic information
- Multiple language prompts for cross-lingual evaluation

## 🔬 Methodology

### Persona-Based Prompting

Each LLM was prompted to respond as specific personas characterized by:

```python
{
    "region": "West Bengal",
    "urban_rural": "Urban",
    "age": "35-44",
    "gender": "Male",
    "marital_status": "Married",
    "education_level": "Secondary",
    "social_class": "Middle class",
    "language": "Bengali"
}
```

### Evaluation Metrics

#### 1. Hard Metric (Exact Match)

```
accuracy = (correct_matches / total_responses) * 100
```

#### 2. Soft Metric (Scaled Distance)

For ordinal scale questions (1-10):

```
score = 1 - |survey_response - model_response| / (num_options - 1)
```

#### 3. Distribution-Based Metrics

- Jensen-Shannon Divergence
- Kullback-Leibler Divergence
- Chi-square test for distribution alignment

## 📊 Key Findings

### 1. Cross-Cultural Insights

- Models show stronger alignment with Western cultural contexts
- Regional and demographic factors significantly impact model performance
- Cultural nuances remain challenging for current LLMs

### 2. Temporal Insights: Cultural Evolution Over Time

- LLMs demonstrate **stronger alignment with cultural values from the preceding decade** (2012) rather than present-day values (2022) in **developing nations** (India, Colombia, Egypt). This temporal misalignment suggests that models' pretraining data reflects earlier cultural distributions, potentially from 2012-2019 period.
- This differential response reflects fundamental differences in:
  - **Cultural dynamics**: Developing nations undergoing rapid social transformation
  - **Digital representation**: Developing countries' values evolving faster than their online representation
  - **Socioeconomic development**: Developed nations show cultural stagnation with minimal change over two decades

### 3. Thematic Change Patterns

Analysis of value changes across WVS waves reveals two distinct patterns:

1. **Fragmented Change**: Questions where 25%+ of personas changed their majority response. Indicates demographic groups shifting opinions at varying rates. Common in themes like "Social Values" and "Social Capital"
2. **Uniform Change**: Questions where 75%+ of personas changed their majority response. Represents widespread societal consensus shifts. Most consistent in "Economic Values" and "Science & Technology" themes

### 4. Multilingual Evaluation

**High-Resource vs Low-Resource Languages:**

- **High alignment**: English, Hindi, Marathi (higher-resource, standardized scripts)
- **Low alignment**: Bengali, Telugu (lower-resource languages)
- Marathi benefits from using Devanagari script (shared with Hindi)

**Regional Language Prompting Results:**

- **Developed countries** (USA, Japan, Russia): Better alignment with native language prompts (e.g., Japanese)
- **India & Egypt**: Mixed results, with better alignment in regional languages for some states
- **Colombia**: Improved alignment with Spanish prompts

**Script Impact:**

- English and regional-script prompts consistently outperform romanized prompts
- Romanized data is less common in pretraining corpora
- **Aya-32B**: Better on regional-script prompts (larger model captures script-specific cues)
- **Gemma-3-12B**: Better on English prompts (smaller model generalizes better with standardized data)

#### Implications

These temporal patterns highlight that LLMs may perpetuate outdated cultural representations, particularly for rapidly developing societies. This raises concerns about using LLMs for cultural understanding or decision-making in dynamic social contexts without temporal awareness.

## 🛠️ Installation & Setup

### Prerequisites

```bash
# Python 3.8+
pip install pandas numpy torch transformers tqdm scikit-learn
```

### Data Preparation

1. Download World Values Survey data for target countries
2. Place data files in respective country folders under `data/`
3. Run preprocessing: `clean_data_wvs.ipynb`

### Running Model Inference

```bash
# Example: Running Aya model for India in English
cd prompt_model/aya
python prompt.py
# Configure: country_name='india', language_code='en', year='2022'
```

### Running Evaluation

```python
from evaluation.scoring import analyze_survey_alignment, compute_similarity_per_theme

# Analyze alignment
results = analyze_survey_alignment(
    year='2022',
    country='india',
    language='en',
    model='aya',
    verbose=True
)

# Theme-based comparison
theme_scores = compute_similarity_per_theme(
    year='2022',
    country='india',
    language='en',
    metric_type='soft',
    model='aya'
)
```

## 📈 Evaluation Notebooks

- [distribution.ipynb](evaluation/distribution.ipynb): Response distribution analysis
- [themes_and_temporal.ipynb](evaluation/themes_and_temporal.ipynb): Theme and temporal analysis
- [languages.ipynb](evaluation/languages.ipynb): Cross-lingual comparison

## 🔍 Example Results

```python
# Sample output from evaluation
{
    'overall_accuracy': 0.632,
    'theme_scores': {
        'SOCIAL VALUES': 0.587,
        'POLITICAL VALUES': 0.701,
        'RELIGIOUS VALUES': 0.543
    },
    'demographic_breakdown': {
        'region': {...},
        'age': {...},
        'gender': {...}
    }
}
```

## 📝 Citation

If you use this work, please cite:

```
Cultural Alignment of Large Language Models: A World Values Survey Analysis
Authors: Athul & Dayita
Institute: Indian Institute of Science
Date: December 2025
```

## 🔗 Resources

- [World Values Survey](https://www.worldvaluessurvey.org/)
- [Hugging Face Models](https://huggingface.co/)

## 👥 Authors

**Dayita & Athul**
DS 307: Ethics in Data Science
Indian Institute of Science
December 2025
