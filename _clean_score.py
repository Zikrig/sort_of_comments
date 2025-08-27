import pandas as pd

# Загружаем датасет с предсказанными значениями
df = pd.read_csv("data/dataset_with_predictions.csv")

# # Перемещаем 'score' рядом с 'pred_score'
# cols = df.columns.tolist()
# cols.remove("score")
# cols.insert(cols.index("pred_score")+1, "score")
# df = df[cols]

# # Добавляем столбец с разницей
# df["delta"] = abs(df["score"] - df["pred_score"])

# # Сортируем по delta (по убыванию — ошибки наверху)
# df = df.sort_values("delta", ascending=False)

# # Выводим базовую статистику
# print("Статистика разницы предсказаний:")
# print(df["delta"].describe())
# print("\nКоличество идеально предсказанных оценок:", (df["delta"]==0).sum())
# print("Процент идеально предсказанных оценок:", (df["delta"]==0).mean()*100)

# # Сохраняем в файл
# df.to_csv("data/dataset_with_predictions_sorted.csv", index=False, encoding="utf-8")
# print("\nСохранено в 'data/dataset_with_predictions_sorted.csv'")

# import pandas as pd

# # Путь к исходному CSV
# dataset_path = "data/dataset.csv"
output_path = "data/dataset_no_score.csv"

# Загружаем датасет
# df = pd.read_csv(dataset_path)

# Удаляем колонку score, если она есть
if "score" in df.columns:
    df = df.drop(columns=["score"])

# Сохраняем новый файл
df.to_csv(output_path, index=False)

print(f"Новый датасет сохранён: {output_path}")
