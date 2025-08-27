import pandas as pd

from services.score_model_trainer import ScoreModelTrainer  # предположим, что класс в score_trainer.py
from services.score_model_predictor import ScorePredictor

# Путь к CSV с данными
dataset_path = "data/dataset.csv"

# Создаём тренера
# trainer = ScoreModelTrainer(dataset_path=dataset_path, output_dir="models")

# # Обучаем модель
# save_path = trainer.train(
#     text_column="text",   # колонка с текстом
#     target_column="score", # колонка с целевой переменной (число от 0 до 5)
#     test_size=0.2         # 20% на тест
# )

# print(f"Модель сохранена в {save_path}")

predictor = ScorePredictor(
    model_path="models/score/model.pkl",
    vectorizer_path="models/score/vectorizer.pkl"
)
# df = pd.read_csv(dataset_path)
# df_result = predictor.predict_dataframe(df, text_column="text")

# output_path = "data/dataset_with_predictions.csv"
# df_result.to_csv(output_path, index=False, encoding="utf-8")
# print(f"Результат сохранён в {output_path}")

text = '''Отзыв: гид Галина Николаевна полное безобразие: 4 часа всю дорогу громко в микрофон трындела не по делу: про ее дочь в Китае, про мужа, сестру, про её друзей, про ее важные знакомства, про ее навыки фотографа, что она -лучший гид, такая популярная, фантазировала на свои личные темы и выдавала очень много ненужной информации, говорила где что купить-рекламировала конкретные места, ноль информации по делу про историю: зато обругала учителей, что детей ничему не учат и тп(а турист-преподаватель)
Гид Вела себя как будто на междусобойчике. 
В конце спросила об их мнении о ней и просила оставить отзыв.
Город, экскурсия и туроператор - хорошие, но гид не квалифицированный. 
Отзыв от Татьяны: Бывалый наш турист, ездила сама и со своими учениками. 
Настойчиво просит замену гида.'''
print(predictor.predict_text(text))