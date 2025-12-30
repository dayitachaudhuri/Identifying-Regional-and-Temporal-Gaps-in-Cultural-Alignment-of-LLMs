# © 2025 Dayita Chaudhuri and Velagapudi Athul
# All rights reserved. Joint work.

import pandas as pd
from scipy.spatial.distance import jensenshannon

# Read CSVs
csv1 = pd.read_csv("csv1.csv")
csv2 = pd.read_csv("csv2.csv")

# Identify persona columns and question columns
persona_cols = ['age', 'gender', 'region']
question_cols = [c for c in csv1.columns if c.startswith('q')]

# Melt csv1 so each row = one (persona, question, answer)
csv1_melted = csv1.melt(id_vars=persona_cols, value_vars=question_cols,
                        var_name='question', value_name='answer')

# Melt csv2 similarly but account for q1-0, q1-1...
csv2_melted = csv2.melt(id_vars=persona_cols, var_name='q_variant', value_name='answer')
csv2_melted['question'] = csv2_melted['q_variant'].str.extract(r'(q\d+)')

# Aggregate distributions
def get_distribution(df):
    return (
        df.groupby(persona_cols + ['question', 'answer'])
          .size()
          .groupby(level=persona_cols + ['question'])
          .apply(lambda x: x / x.sum())  # Normalize to probs
          .reset_index(name='prob')
    )

dist1 = get_distribution(csv1_melted)
dist2 = get_distribution(csv2_melted)

# Merge distributions for comparison
merged = pd.merge(dist1, dist2, on=persona_cols + ['question', 'answer'], 
                  how='outer', suffixes=('_csv1', '_csv2')).fillna(0)

# Compute Jensen-Shannon divergence per persona+question
results = (
    merged.groupby(persona_cols + ['question'])
    .apply(lambda g: jensenshannon(g['prob_csv1'], g['prob_csv2']))
    .reset_index(name='js_distance')
)

print(results.head())
