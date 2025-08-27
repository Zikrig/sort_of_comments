from services.predictor import UnifiedPredictor


# Инициализация predictor
predictor = UnifiedPredictor(
    complain_model_path="models/complain/model.pkl",
    multilabel_model_path="models/multilabel/model.pkl",
    score_model_path="models/score/model.pkl",
    complain_vectorizer_path="models/complain/vectorizer.pkl",
    multilabel_vectorizer_path="models/multilabel/vectorizer.pkl",
    score_vectorizer_path="models/score/vectorizer.pkl"
)

# Обработка одного отзыва
single_result = predictor.predict_single({
    "review_id": 175,
    "text": '''"Оценка - 3. Еда не понравилась
Свободного времени не было
Устала сидеть в автобусе"''',
    "order_id": 331629
})

print(single_result)

# Обработка пакета отзывов с сохранением в файл
# batch_result = predictor.predict_batch(
#     data=your_data,
#     save_path="results.json"
# )