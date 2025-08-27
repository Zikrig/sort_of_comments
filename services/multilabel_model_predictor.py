import pandas as pd
import joblib
import os
import string
import re
from typing import List, Dict

# Препроцессор
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text

CLASS_MAP = {
    # 1: "Автобус",
    # 2: "Питание",
    # 3: "Отель",
    # 4: "Гид",
    # 5: "Экскурсовод на месте",
    # 6: "Программа тура",
    # 7: "Сервис",
    # 8: "Логистика",
    # 9: "Другое",
    
    'id_1': "Автобус",
    'id_2': "Питание",
    'id_3': "Отель",
    'id_4': "Гид",
    'id_5': "Экскурсовод на месте",
    'id_6': "Программа тура",
    'id_7': "Сервис",
    'id_8': "Логистика",
    'id_9': "Другое",
}

class MultiLabelPredictor:
    def __init__(self, models_path: str, vectorizer_path: str):
        self.models = joblib.load(models_path)  # dict: {class_id: model}
        self.vectorizer = joblib.load(vectorizer_path)
        self.class_ids = list(self.models.keys())

    def predict_text(self, text: str) -> Dict[int, int]:
        """Возвращает словарь {class_id: 0/1}"""
        vec = self.vectorizer.transform([text])
        preds = {cls: int(self.models[cls].predict(vec)[0]) for cls in self.class_ids}
        return preds

    def predict_text_as_list(self, text: str) -> List[Dict[str, str]]:
        """Возвращает список словарей [{'id': 1, 'name': 'Автобус'}, ...]"""
        preds = self.predict_text(text)
        result = [{"id": cls, "name": CLASS_MAP[cls]} for cls, val in preds.items() if val == 1]
        return result

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Добавляет колонки классов и их предсказаний, строки с несовпадениями наверх"""
        df = df.copy()
        X = df[text_column].astype(str).tolist()
        vec = self.vectorizer.transform(X)

        for cls in self.class_ids:
            pred_col = f"id_{cls}_pred"
            df[pred_col] = self.models[cls].predict(vec).astype(int)

        # Если реальные колонки есть, считаем несовпадения
        mismatch_cols = []
        for cls in self.class_ids:
            true_col = f"id_{cls}"
            pred_col = f"id_{cls}_pred"
            if true_col in df.columns:
                mismatch_col = f"{true_col}_mismatch"
                df[mismatch_col] = df[true_col] != df[pred_col]
                mismatch_cols.append(mismatch_col)

        if mismatch_cols:
            df["_total_mismatch"] = df[mismatch_cols].sum(axis=1)
            df = df.sort_values("_total_mismatch", ascending=False).drop(columns=mismatch_cols + ["_total_mismatch"])

        df = df.reset_index(drop=True)
        return df
