import re
import json
from typing import Dict, List, Union
from services.text_model_predictor import TextModelPredictor
from services.multilabel_model_predictor import MultiLabelPredictor
from services.score_model_predictor import ScorePredictor

class UnifiedPredictor:
    def __init__(self, 
                 complain_model_path: str,
                 multilabel_model_path: str,
                 score_model_path: str,
                 complain_vectorizer_path: str,
                 multilabel_vectorizer_path: str,
                 score_vectorizer_path: str):
        
        self.complain_predictor = TextModelPredictor(complain_model_path, complain_vectorizer_path)
        self.multilabel_predictor = MultiLabelPredictor(multilabel_model_path, multilabel_vectorizer_path)
        self.score_predictor = ScorePredictor(score_model_path, score_vectorizer_path)

    def _should_process_text(self, text: str) -> bool:
        """Проверяет, нужно ли обрабатывать текст моделями"""
        text_lower = text.lower()
        return len(text) > 30 and "нет" not in text_lower

    def predict_single(self, data: Dict) -> Dict:
        """Обрабатывает один отзыв"""
        text = data.get("text", "")
        
        if self._should_process_text(text):
            is_complaint = bool(self.complain_predictor.predict_text(text))
            score = self.score_predictor.predict_text(text)
        else:
            is_complaint = False
            score = 0

        # Предсказываем причины если есть жалоба или низкая оценка
        if is_complaint or score < 5:
            reasons = self.multilabel_predictor.predict_text_as_list(text)
            reason_names = [r["name"] for r in reasons]
            reason = ", ".join(reason_names)
        else:
            reason = ""

        return {
            **data,
            "is_complaint": is_complaint,
            "score": score,
            "reason": reason
        }

    def predict_batch(self, data: Dict[str, Dict], save_path: Union[str, None] = None) -> Dict[str, Dict]:
        """Обрабатывает пакет отзывов"""
        results = {}
        for key, item in data.items():
            results[key] = self.predict_single(item)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results