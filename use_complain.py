from services.complain_model_trainer import TextModelTrainer
from services.complain_model_predictor import TextModelPredictor


trainer = TextModelTrainer("data/dataset.csv")

# 1. Обучение на поле complaint (True/False)
trainer.train(field_name="complaint", classes=[True])


# predictor = TextModelPredictor(
#     model_path="models/complaint_types_20250827_101010/model.pkl",
#     vectorizer_path="models/complaint_types_20250827_101010/vectorizer.pkl"
# )

# single_pred = predictor.predict_text("Экскурсия была ужасная, гид не понравился")
# proba = predictor.predict_proba("Экскурсия была ужасная, гид не понравился")

# df_with_preds = predictor.predict_dataframe(df, text_field="text")