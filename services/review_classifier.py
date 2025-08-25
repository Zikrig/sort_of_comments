import asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging
from tqdm.asyncio import tqdm_asyncio

from db.models import Base, RRating, RReview, RComplaintType, RRatingToComplaintType

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from db.database_manager import DatabaseManager
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Настройка модели
model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Few-shot примеры для классификации
FEW_SHOT_EXAMPLES = [
    """
Текст: "Отличный тур, всё организовано на высшем уровне, обязательно поеду ещё!"
Анализ:
- Жалоба: нет
- Оценка: 5
- Причина жалобы: 
- Поедет во второй раз: да
""",
    """
Текст: "Грязь в номере, еда невкусная, персонал грубый, больше не поеду."
Анализ:
- Жалоба: да
- Оценка: 1
- Причина жалобы: грязь в номере, невкусная еда, грубый персонал
- Поедет во второй раз: нет
""",
    """
Текст: "В целом нормально, но экскурсии были скучными. Оценка 3."
Анализ:
- Жалоба: да
- Оценка: 3
- Причина жалобы: скучные экскурсии
- Поедет во второй раз: не указано
""",
    """
Текст: "Всё прошло гладко, но ничего выдающегося."
Анализ:
- Жалоба: нет
- Оценка: None
- Причина жалобы: 
- Поедет во второй раз: не указано
""",
    """
Текст: "Ужасный сервис, потеряли багаж, оценка 2, не рекомендую!"
Анализ:
- Жалоба: да
- Оценка: 2
- Причина жалобы: ужасный сервис, потеря багажа
- Поедет во второй раз: нет
""",
    """
Текст: "Очень понравилось, гид был замечательный, ставлю 5!"
Анализ:
- Жалоба: нет
- Оценка: 5
- Причина жалобы: 
- Поедет во второй раз: не указано
"""
]

class ReviewClassifier:
    def __init__(self, db_manager: DatabaseManager, batch_size: int = 10):
        self.db_manager = db_manager
        self.batch_size = batch_size

    async def get_all_reviews(self, offset: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получить отзывы из таблицы r_review с пагинацией.
        """
        async with self.db_manager.async_session() as session:
            stmt = select(RReview).options(
                selectinload(RReview.rating)
            ).offset(offset).limit(limit)
            result = await session.execute(stmt)
            reviews = result.scalars().all()
            
            return [
                {
                    "review_id": review.id,
                    "text": review.text,
                    "rating_id": review.rating_id,
                    "order_id": review.rating.orderId if review.rating else None,
                    "current_score": review.rating.score if review.rating else None,
                    "current_complaint": review.rating.complaint if review.rating else None,
                    "current_plan_next": review.rating.planNext if review.rating else None
                }
                for review in reviews if review.text  # Только непустые тексты
            ]

    def classify_reviews_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Классифицировать батч текстов с использованием few-shot промпта.
        """
        results = []
        for text in texts:
            prompt = "\n".join(FEW_SHOT_EXAMPLES) + f'\nТекст: "{text}"\nАнализ:\n'
            inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=False
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis_part = generated_text[len(prompt):].strip()
            parsed = self.parse_analysis(analysis_part)
            
            results.append({
                "text": text,
                "is_complaint": parsed.get("complaint", False),
                "score": parsed.get("score", None),
                "reason": parsed.get("reason", ""),
                "plan_next": parsed.get("plan_next", None)
            })
        
        return results

    def parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """
        Парсер для извлечения полей из сгенерированного текста.
        """
        lines = analysis_text.split("\n")
        parsed = {}
        for line in lines:
            if line.startswith("- Жалоба:"):
                parsed["complaint"] = line.split(":")[1].strip().lower() == "да"
            elif line.startswith("- Оценка:"):
                score_str = line.split(":")[1].strip()
                parsed["score"] = int(score_str) if score_str.isdigit() else None
            elif line.startswith("- Причина жалобы:"):
                parsed["reason"] = line.split(":")[1].strip()
            elif line.startswith("- Поедет во второй раз:"):
                val = line.split(":")[1].strip().lower()
                parsed["plan_next"] = True if val == "да" else False if val == "нет" else None
        return parsed

    async def process_all_reviews(self, update_db: bool = False) -> List[Dict[str, Any]]:
        """
        Обработать все отзывы с пагинацией и батчевой классификацией.
        """
        total_reviews = 40
        offset = 0
        all_results = []
        
        while offset < total_reviews:
            logger.info(f"Обработка отзывов с {offset} по {offset + self.batch_size}")
            reviews = await self.get_all_reviews(offset=offset, limit=self.batch_size)
            if not reviews:
                break
                
            texts = [review["text"] for review in reviews]
            classifications = self.classify_reviews_batch(texts)
            
            for review, classification in zip(reviews, classifications):
                result = {
                    "review_id": review["review_id"],
                    "text": review["text"],
                    "is_complaint": classification["is_complaint"],
                    "score": classification["score"],
                    "reason": classification["reason"],
                    "plan_next": classification["plan_next"],
                    "order_id": review["order_id"]
                }
                all_results.append(result)
                
                if update_db and review["order_id"]:
                    try:
                        await self.db_manager.create_or_update_order(
                            order_id=review["order_id"],
                            score=classification["score"] or review["current_score"],
                            complaint=classification["is_complaint"] or review["current_complaint"],
                            plan_next=classification["plan_next"] or review["current_plan_next"],
                            review_text=review["text"]
                        )
                    except Exception as e:
                        logger.error(f"Ошибка при обновлении заказа {review['order_id']}: {e}")
            
            offset += self.batch_size
        
        return all_results

