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
tokenizer.pad_token_id = tokenizer.eos_token_id  # Устанавливаем pad_token_id
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Few-shot промпты (определены выше)
from prompts import *

class ReviewClassifier:
    def __init__(self, db_manager: DatabaseManager, batch_size: int = 100):
        self.db_manager = db_manager
        self.batch_size = batch_size

    def classify_reviews_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Классифицировать батч текстов с использованием отдельных few-shot промптов для каждого вопроса.
        """
        
        if len(texts) == 0:
            return []
        
        results = []
        
        # Подготовка промптов для каждого вопроса
        complaint_prompts = [("\n".join(IS_COMPLAINT_PROMPT) + f'\nТекст: "{text}"\nЯвляется ли текст жалобой? ') for text in texts]
        score_prompts = [("\n".join(SCORE_PROMPT) + f'\nТекст: "{text}"\nОценка поездки: ') for text in texts]
        reason_prompts = [("\n".join(REASON_PROMPT) + f'\nТекст: "{text}"\nПричина жалобы: ') for text in texts]
        # plan_next_prompts = [(("\n".join(PLAN_NEXT_PROMPT) + f'\nТекст: "{text}"\nПоедет во второй раз: ') for text in texts)]
        
        # Обработка каждого вопроса отдельно
        for prompt_list, key, max_new_tokens in [
            (complaint_prompts, "is_complaint", 10),
            (score_prompts, "score", 10),
            (reason_prompts, "reason", 150),
            # (plan_next_prompts, "plan_next", 10)
        ]:

            
        # Дополнительная отладочная информация
            if not prompt_list:
                return []
            
            inputs = tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True
                )
            
            for i, output in enumerate(outputs.sequences):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                answer = generated_text[len(prompt_list[i]):].strip()
                
                if not results:
                    results = [{} for _ in texts]
                
                if key == "is_complaint":
                    results[i]["is_complaint"] = answer.lower() == "да"
                elif key == "score":
                    results[i]["score"] = int(answer) if answer.isdigit() else None
                elif key == "reason":
                    results[i]["reason"] = answer
                # elif key == "plan_next":
                #     results[i]["plan_next"] = True if answer.lower() == "да" else False if answer.lower() == "нет" else None
                results[i]["text"] = texts[i]
        
        return results

    async def map_reason_to_complaint_types(self, reason: str) -> List[int]:
        """
        Сопоставить причину жалобы с типами жалоб из r_complaint_type.
        """
        complaint_type_mapping = {
            "автобус": 1,
            "питание": 2,
            "отель": 3,
            "гид": 4,
            "экскурсовод на месте": 5,
            "программа тура": 6,
            "сервис": 7,
            "логистика": 8,
            "другое": 9
        }
        return [complaint_type_mapping[word] for word in complaint_type_mapping if word.lower() in reason.lower()]

    async def process_all_reviews(self, update_db: bool = False, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Обработать все отзывы с пагинацией и батчевой классификацией.
        """
        total_reviews = 200
        offset = 0
        all_results = []
        
        # Загрузка существующих результатов для возобновления
        if output_file and os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            offset = len(all_results)
            logger.info(f"Возобновление с offset={offset}")

        while offset < total_reviews:
            logger.info(f"Обработка отзывов с {offset} по {offset + self.batch_size}")
            reviews = await self.db_manager.get_all_reviews(offset=offset, limit=self.batch_size)
            if not reviews:
                break
                
            texts = [str(review["text"]) for review in reviews if review["text"] is not None and review["text"].lower() != "нет отзыва"]

            if not texts:
                offset += self.batch_size
                continue

            classifications = self.classify_reviews_batch(texts)
            
            text_index = 0
            for review in reviews:
                if review["text"].lower() == "нет отзыва":
                    result = {
                        "review_id": review["review_id"],
                        "text": review["text"],
                        "is_complaint": False,
                        "score": 0,
                        "reason": "нет",
                        # "plan_next": None,
                        "order_id": review["order_id"],
                        "current_score": review["current_score"],
                        "current_complaint": review["current_complaint"],
                        "current_plan_next": review["current_plan_next"]
                    }
                else:
                    classification = classifications[text_index]
                    result = {
                        "review_id": review["review_id"],
                        "text": review["text"],
                        "is_complaint": classification["is_complaint"],
                        "score": classification["score"],
                        "reason": classification["reason"],
                        # "plan_next": classification["plan_next"],
                        "order_id": review["order_id"],
                        "current_score": review["current_score"],
                        "current_complaint": review["current_complaint"],
                        "current_plan_next": review["current_plan_next"]
                    }
                    text_index += 1
                
                all_results.append(result)
                
                if update_db and review["order_id"]:
                    try:
                        complaint_types = await self.map_reason_to_complaint_types(result["reason"])
                        await self.db_manager.create_or_update_order(
                            order_id=review["order_id"],
                            score=result["score"] or review["current_score"],
                            complaint=result["is_complaint"] or review["current_complaint"],
                            # plan_next=result["plan_next"] or review["current_plan_next"],
                            review_text=review["text"],
                            complaint_types=complaint_types
                        )
                    except Exception as e:
                        logger.error(f"Ошибка при обновлении заказа {review['order_id']}: {e}")
            
            # Сохранение промежуточных результатов
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            offset += self.batch_size
        
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
            #   f"Поедет еще: {res['plan_next']}\n"
              f"Текущие данные: score={res['current_score']}, complaint={res['current_complaint']}, plan_next={res['current_plan_next']}\n"
              f"{'-'*70}")

if __name__ == "__main__":
    asyncio.run(main())