import pandas as pd
df_labels = pd.read_csv('labels.csv', header=None, encoding='utf-8', delimiter=',', names=['text', 'filename'])
df_labels.head()
lengths = df_labels['text'].str.len()
length_distribution = lengths.value_counts().sort_index()
unique_characters = pd.Series(list(''.join(df_labels['text']))).value_counts()

print(length_distribution, unique_characters)

char_counts = df_labels['text'].apply(lambda x: pd.Series(list(x))).unstack().value_counts()
total_chars = char_counts.sum()
char_weights = char_counts.map(lambda x: total_chars / (len(char_counts) * x))
most_common_chars = char_counts.head(10)
least_common_chars = char_counts.tail(10)
label_length_counts = df_labels['text'].str.len().value_counts(normalize=True)

print(char_counts, char_weights, most_common_chars, least_common_chars, label_length_counts)
