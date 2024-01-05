import pandas as pd

# Загружаем данные из CSV-файла
df_labels = pd.read_csv('labels_new_big.csv', header=None, encoding='utf-8', delimiter=',', names=['text', 'filename'])

# Выводим первые несколько строк для проверки
df_labels.head()

# Анализируем распределение длин текстов в лейблах
lengths = df_labels['text'].str.len()
length_distribution = lengths.value_counts().sort_index()

# Подсчет количества уникальных символов и их частоты
unique_characters = pd.Series(list(''.join(df_labels['text']))).value_counts()

print(length_distribution, unique_characters)

# Подготовка данных для более детального анализа распределения символов и длин
char_counts = df_labels['text'].apply(lambda x: pd.Series(list(x))).unstack().value_counts()

# Рассчитаем веса для каждого символа на основе их частоты встречаемости
total_chars = char_counts.sum()
char_weights = char_counts.map(lambda x: total_chars / (len(char_counts) * x))

# Определим, какие символы встречаются наиболее и наименее часто
most_common_chars = char_counts.head(10)
least_common_chars = char_counts.tail(10)

# Анализ длины лейблов
label_length_counts = df_labels['text'].str.len().value_counts(normalize=True)

print(char_counts, char_weights, most_common_chars, least_common_chars, label_length_counts)


