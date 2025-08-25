import asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging
import json
from tqdm.asyncio import tqdm_asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from db.database_manager import DatabaseManager

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
    def __init__(self, db_manager: DatabaseManager, batch_size: int = 100):
        self.db_manager = db_manager
        self.batch_size = batch_size

    def classify_reviews_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Классифицировать батч текстов с использованием few-shot промпта.
        """
        results = []
        prompts = [("\n".join(FEW_SHOT_EXAMPLES) + f'\nТекст: "{text}"\nАнализ:\n') for text in texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                return_dict_in_generate=True
            )
        
        for i, output in enumerate(outputs.sequences):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            analysis_part = generated_text[len(prompts[i]):].strip()
            parsed = self.parse_analysis(analysis_part)
            results.append({
                "text": texts[i],
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

    async def map_reason_to_complaint_types(self, reason: str) -> List[int]:
        """
        Сопоставить причину жалобы с типами жалоб из r_complaint_type.
        """
        complaint_type_mapping = {
            "грязь": 1,  # Пример ID для "грязь в номере"
            "еда": 2,    # Пример ID для "невкусная еда"
            "сервис": 3, # Пример ID для "плохой сервис"
            "гид": 4,    # Пример ID для "проблемы с гидом"
            "экскурсии": 5  # Пример ID для "скучные экскурсии"
        }
        return [complaint_type_mapping[word] for word in complaint_type_mapping if word.lower() in reason.lower()]

    async def process_all_reviews(self, update_db: bool = False, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Обработать все отзывы с пагинацией и батчевой классификацией.
        """
        total_reviews = 40000  # Замените на точное число из r_review
        offset = 0
        all_results = []
        
        while offset < total_reviews:
            logger.info(f"Обработка отзывов с {offset} по {offset + self.batch_size}")
            reviews = await self.db_manager.get_all_reviews(offset=offset, limit=self.batch_size)
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
                    "order_id": review["order_id"],
                    "current_score": review["current_score"],
                    "current_complaint": review["current_complaint"],
                    "current_plan_next": review["current_plan_next"]
                }
                all_results.append(result)
                
                if update_db and review["order_id"]:
                    try:
                        complaint_types = await self.map_reason_to_complaint_types(classification["reason"])
                        await self.db_manager.create_or_update_order(
                            order_id=review["order_id"],
                            score=classification["score"] or review["current_score"],
                            complaint=classification["is_complaint"] or review["current_complaint"],
                            plan_next=classification["plan_next"] or review["current_plan_next"],
                            review_text=review["text"],
                            complaint_types=complaint_types
                        )
                    except Exception as e:
                        logger.error(f"Ошибка при обновлении заказа {review['order_id']}: {e}")
            
            offset += self.batch_size
        
        # Сохранение результатов в файл, если указан
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        return all_results

async def main():
    db_url = os.getenv('DB_LINK')
    db_manager = DatabaseManager(db_url)
    await db_manager.init_models()
    
    classifier = ReviewClassifier(db_manager, batch_size=100)
    results = await classifier.process_all_reviews(update_db=False, output_file="classification_results.json")
    
    for res in results[:10]:  # Вывод первых 10 результатов
        print(f"ID отзыва: {res['review_id']}\n"
              f"ID заказа: {res['order_id']}\n"
              f"Текст: '{res['text']}'\n"
              f"Жалоба: {res['is_complaint']}\n"
              f"Оценка: {res['score']}\n"
              f"Причина: {res['reason']}\n"
              f"Поедет еще: {res['plan_next']}\n"
              f"Текущие данные: score={res['current_score']}, complaint={res['current_complaint']}, plan_next={res['current_plan_next']}\n"
              f"{'-'*70}")

if __name__ == "__main__":
    asyncio.run(main())