import pandas as pd
import ast

# Пути
dataset_path = "data/dataset.csv"
output_empty = "data/complaints_empty.csv"
output_nonempty = "data/complaints_nonempty.csv"

# Загружаем
df = pd.read_csv(dataset_path)

# Условие: score < 5 и complaint == True
mask = (df["score"] < 5) & (df["complaint"] == True)
df_filtered = df[mask].copy()

# complaint_types — это строка вроде "[]" или "[{...}]"
# Парсим строку в объект Python
def parse_types(x):
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except Exception:
        return []

# df_filtered["complaint_types_parsed"] = df_filtered["complaint_types"].apply(parse_types)

# Разделяем
df_empty = df_filtered[df_filtered["complaint_types"].apply(lambda x: len(x) < 5)]
df_nonempty = df_filtered[df_filtered["complaint_types"].apply(lambda x: len(x) > 5)]

# Сохраняем
df_empty.to_csv(output_empty, index=False)
df_nonempty.to_csv(output_nonempty, index=False)

print(f"Сохранено: {len(df_empty)} строк в {output_empty}")
print(f"Сохранено: {len(df_nonempty)} строк в {output_nonempty}")
