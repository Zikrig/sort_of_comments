import pandas as pd
import joblib
import os
from datetime import datetime
from typing import List, Union, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd
import joblib
import os
import string
from datetime import datetime
from typing import List, Union, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import re


# Текстовый препроцессор
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # убираем пунктуацию
    # text = " ".join([w for w in text.split() if w not in ENGLISH_STOP_WORDS])
    return text
    
class TextModelTrainer:
    def __init__(self, dataset_path: str, output_dir: str = "models"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_timestamped_path(self, field_name: str) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"{field_name}_{now}")

    def _extract_target(
        self, df: pd.DataFrame, field_name: str, classes: List[Any]
    ) -> List[int]:
        """
        Преобразует столбец field_name в метки классов
        """
        if field_name == "complaint_types":
            # complaint_types хранится в виде строки со списком словарей
            return [
                1 if str(row).strip() not in ("[]", "", "nan") else 0
                for row in df[field_name]
            ]
        else:
            # для булевых и обычных значений
            return [1 if row in classes else 0 for row in df[field_name]]

    def train(
        self,
        field_name: str,
        classes: List[Union[str, int, bool]],
        test_size: float = 0.2,
    ):
        # Загружаем датасет
        df = pd.read_csv(self.dataset_path)

        # Тексты и целевая переменная
        X = df["text"].astype(str).tolist()
        y = self._extract_target(df, field_name, classes)

        # Тренировочная и тестовая выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=45
        )

        # Векторизация
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), preprocessor=clean_text)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Модель
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X_train_vec, y_train)

        # Оценка
        y_pred = model.predict(X_test_vec)
        print(classification_report(y_test, y_pred))

        # Сохранение
        save_path = self._get_timestamped_path(field_name)
        os.makedirs(save_path, exist_ok=True)

        joblib.dump(model, os.path.join(save_path, "model.pkl"))
        joblib.dump(vectorizer, os.path.join(save_path, "vectorizer.pkl"))

        print(f"Модель сохранена в {save_path}")
        self.show_mistakes(model, vectorizer, X_test, y_test, 10)
        self.save_dataset_with_errors_full(model, vectorizer, df, field_name, output_path=os.path.join(save_path, "dataset_with_errors.csv"))
        return save_path

    def show_mistakes(self, model, vectorizer, X_test, y_test, max_examples: int = 20):
        """
        Показывает примеры, где модель ошиблась
        """
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
        
        mistakes = []
        for text, true_label, pred_label in zip(X_test, y_test, y_pred):
            if true_label != pred_label:
                mistakes.append({
                    "text": text,
                    "true": true_label,
                    "pred": pred_label
                })
                if len(mistakes) >= max_examples:
                    break
        
        print(f"Всего ошибок: {len([1 for t, p in zip(y_test, y_pred) if t != p])}")
        for i, m in enumerate(mistakes, 1):
            print(f"\nОшибка #{i}")
            print(f"True: {m['true']}, Pred: {m['pred']}")
            print(f"Text: {m['text'][:500]}")  # обрезаем длинные тексты

        return mistakes
    
    def save_dataset_with_errors(
        self, model, vectorizer, df: pd.DataFrame, field_name: str, output_path: str = "dataset_with_errors.csv"
    ):
        """
        Создаёт новый датасет:
        - ошибки модели идут в начале
        - целевое поле дополнительно помечается постфиксом '_p' у ошибок
        """
        X = df["text"].astype(str).tolist()
        y_true = self._extract_target(df, field_name, classes=[1])  # используем класс 1 для примера

        X_vec = vectorizer.transform(X)
        y_pred = model.predict(X_vec)

        rows_with_error = []
        rows_correct = []

        for idx, (text, true_label, pred_label) in enumerate(zip(X, y_true, y_pred)):
            row = df.iloc[idx].copy()
            if true_label != pred_label:
                # ошибка — помечаем поле
                row[field_name] = f"{row[field_name]}_p"
                rows_with_error.append(row)
            else:
                rows_correct.append(row)

        # объединяем: ошибки в начале
        new_df = pd.DataFrame(rows_with_error + rows_correct)
        new_df.to_csv(output_path, index=False)
        print(f"Новый датасет сохранён в {output_path}, ошибок: {len(rows_with_error)}")
        return new_df

    def save_dataset_with_errors_full(
        self, model, vectorizer, df: pd.DataFrame, field_name: str, output_path: str = "dataset_with_errors.csv"
    ):
        """
        Создаёт новый датасет по всему df:
        - ошибки модели идут в начале
        - целевое поле дополнительно помечается постфиксом '_p' у ошибок
        """
        # Векторизуем все тексты
        X = df["text"].astype(str).tolist()
        X_vec = vectorizer.transform(X)

        # Предсказания модели
        y_pred = model.predict(X_vec)

        # Истинные значения
        y_true = self._extract_target(df, field_name, classes=[1])  # берем 1 как целевой класс

        rows_with_error = []
        rows_correct = []

        for idx, (text, true_label, pred_label) in enumerate(zip(X, y_true, y_pred)):
            row = df.iloc[idx].copy()
            if true_label != pred_label:
                row[field_name] = f"{row[field_name]}_p"
                rows_with_error.append(row)
            else:
                rows_correct.append(row)

        # объединяем: ошибки в начале
        new_df = pd.DataFrame(rows_with_error + rows_correct)
        new_df.to_csv(output_path, index=False)
        print(f"Новый датасет сохранён в {output_path}, ошибок: {len(rows_with_error)}")
        return new_df
