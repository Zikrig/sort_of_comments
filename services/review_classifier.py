import asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
import shutil
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from time import sleep

from db.database_manager import DatabaseManager
from prompts import *

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Настройка модели
model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

class ReviewClassifier:
    def __init__(self, db_manager: DatabaseManager, batch_size: int = 100):
        self.db_manager = db_manager
        self.batch_size = batch_size
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)  # Создаем папку data если не существует

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
        
        # Обработка каждого вопроса отдельно
        for prompt_list, key, max_new_tokens, requiest_sentence in [
            (complaint_prompts, "is_complaint", 10, 'Является ли текст жалобой? '),
            (score_prompts, "score", 10, 'Оценка поездки: '),
            (reason_prompts, "reason", 150, 'Причина жалобы: '),
        ]:
            if not prompt_list:
                return []
                        
            max_ctx = getattr(model.config, "max_position_embeddings", 2048)
            tokenizer_max = min( max_ctx, 2048 )

            inputs = tokenizer(
                prompt_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer_max
            ).to(model.device)

            # посчитаем реальные длины входов (чтобы потом отрезать prompt из полного sequence)
            input_lengths = inputs['attention_mask'].sum(dim=1).tolist()

            with torch.no_grad():
                outputs = model.generate(
                    **{k: inputs[k] for k in ("input_ids", "attention_mask")},
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True
                )

            for i, seq in enumerate(outputs.sequences):
                # отрежем именно сгенерированные токены
                in_len = int(input_lengths[i])
                gen_ids = seq[in_len:].to('cpu')
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                print(gen_text)

                # fallback: если вдруг пусто — декодируем всю последовательность и вычитаем prompt
                if not gen_text:
                    full = tokenizer.decode(seq, skip_special_tokens=True)
                    gen_text = full.replace(prompt_list[i], "").strip()
                    print('AAALLL')

                # более устойчивый парсинг
                if not results:
                    results = [{} for _ in texts]

                if key == "is_complaint":
                    # ищем отдельное слово "да" или "нет"
                    if re.search(r'\bда\b', gen_text, re.I):
                        results[i]["is_complaint"] = True
                    elif re.search(r'\bнет\b', gen_text, re.I):
                        results[i]["is_complaint"] = False
                    else:
                        # неопределённо — можно поставить None или сделать эвристику
                        results[i]["is_complaint"] = False

                elif key == "score":
                    m = re.search(r'(-?\d+)', gen_text)
                    results[i]["score"] = int(m.group(1)) if m else None

                elif key == "reason":
                    # не трогаем — просто текст причины
                    results[i]["reason"] = gen_text

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

    def _save_batch_results(self, batch_results: List[Dict[str, Any]], batch_number: int):
        """
        Сохранить результаты батча в файл с временной меткой.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"batch_{batch_number:04d}_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Сохранен батч {batch_number} в файл: {filename}")

    async def process_all_reviews(self, save_to_file: bool = False, update_db: bool = False) -> List[Dict[str, Any]]:
        """
        Обработать все отзывы с пагинацией и батчевой классификацией.
        
        Args:
            save_to_file: Если True, сохранять каждый батч в папку data
            update_db: Если True, обновлять данные в базе
        """
        total_reviews = 200
        offset = 80
        # all_results = []
        batch_number = 0
        
        # Проверяем существующие файлы для возобновления
        # if save_to_file and self.data_dir.exists():
        #     existing_files = list(self.data_dir.glob("batch_*.json"))
        #     if existing_files:
        #         # Сортируем по номеру батча и берем последний
        #         existing_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        #         last_file = existing_files[-1]
        #         with open(last_file, "r", encoding="utf-8") as f:
        #             last_batch = json.load(f)
        #         offset = len(last_batch)  # Обновляем offset на основе загруженных данных
        #         logger.info(f"Найдены существующие файлы. Возобновление с offset={offset}")

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
            batch_results = []
            
            text_index = 0
            for i in range(len(reviews)):
                review = reviews[i]
                if review["text"].lower() == "нет отзыва":
                    result = {
                        "review_id": review["review_id"],
                        "text": review["text"],
                        "is_complaint": False,
                        "score": 0,
                        "reason": "",
                        "order_id": review["order_id"],
                        "current_score": review["current_score"],
                        "current_complaint": review["current_complaint"],
                        "current_plan_next": review["current_plan_next"]
                    }
                else:
                    classification = classifications[text_index]
                    # print(classification)
                    result = {
                        "review_id": review["review_id"],
                        "text": review["text"],
                        "is_complaint": classification["is_complaint"],
                        "score": classification["score"],
                        "reason": classification["reason"],
                        "order_id": review["order_id"],
                        "current_score": review["current_score"],
                        "current_complaint": review["current_complaint"],
                        "current_plan_next": review["current_plan_next"]
                    }
                    text_index += 1
                
                batch_results.append(result)
                # all_results.append(result)
                
                if update_db and review["order_id"]:
                    try:
                        complaint_types = await self.map_reason_to_complaint_types(result["reason"])
                        await self.db_manager.create_or_update_order(
                            order_id=review["order_id"],
                            score=result["score"] or review["current_score"],
                            complaint=result["is_complaint"] or review["current_complaint"],
                            review_text=review["text"],
                            complaint_types=complaint_types
                        )
                    except Exception as e:
                        logger.error(f"Ошибка при обновлении заказа {review['order_id']}: {e}")
            
            # Сохранение батча в файл если нужно
            
            offset += self.batch_size
            batch_number += 1
            
            if save_to_file:
                self._save_batch_results(batch_results, batch_number - 1)
        
        return True

# async def main():
#     db_url = os.getenv('DB_LINK')
#     db_manager = DatabaseManager(db_url)
#     await db_manager.init_models()
    
#     classifier = ReviewClassifier(db_manager, batch_size=100)
#     results = await classifier.process_all_reviews(update_db=False, output_file="classification_results.json")
    
#     for res in results[:10]:  # Вывод первых 10 результатов
#         print(f"ID отзыва: {res['review_id']}\n"
#               f"ID заказа: {res['order_id']}\n"
#               f"Текст: '{res['text']}'\n"
#               f"Жалоба: {res['is_complaint']}\n"
#               f"Оценка: {res['score']}\n"
#               f"Причина: {res['reason']}\n"
#             #   f"Поедет еще: {res['plan_next']}\n"
#               f"Текущие данные: score={res['current_score']}, complaint={res['current_complaint']}, plan_next={res['current_plan_next']}\n"
#               f"{'-'*70}")

# if __name__ == "__main__":
#     asyncio.run(main())