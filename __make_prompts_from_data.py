import asyncio
from dotenv import load_dotenv
import os

from db.database_manager import DatabaseManager
from services.batch_processor import BatchProcessor

load_dotenv()

async def main():
    db_url = os.getenv('DB_LINK')
    db_manager = DatabaseManager(db_url)
    batch_processor = BatchProcessor(db_manager)
    
    total_count = await batch_processor.get_total_reviews_count()
    print(f"Всего отзывов: {total_count}")

    # Открываем файлы для записи промптов
    with open('prompts/complaint_prompts.py', 'w', encoding='utf-8') as f_comp, \
         open('prompts/score_prompts.py', 'w', encoding='utf-8') as f_score, \
         open('prompts/reason_prompts.py', 'w', encoding='utf-8') as f_reason, \
         open('prompts/plan_next_prompts.py', 'w', encoding='utf-8') as f_plan:

        # Записываем заголовки для каждого файла
        f_comp.write('IS_COMPLAINT_PROMPT = [\n')
        f_score.write('SCORE_PROMPT = [\n')
        f_reason.write('REASON_PROMPT = [\n')
        f_plan.write('PLAN_NEXT_PROMPT = [\n')

        batch_size = 200
        for offset in range(0, total_count, batch_size):
            batch = await batch_processor.get_reviews_batch(batch_size=batch_size, offset=offset)
            
            for review in batch:
                text = review['text'].replace('\n', ' ').replace('"', '\\"')
                
                # Формируем промпты
                complaint_prompt = f'    """\nТекст: "{text}"\nЯвляется ли текст жалобой? {"Да" if review["complaint"] else "Нет"}\n""",'
                score_prompt = f'    """\nТекст: "{text}"\nОценка поездки: {review["score"]}\n""",'
                
                reason = ", ".join([ct["name"] for ct in review["complaint_types"]]) if review["complaint"] else ""
                reason_prompt = f'    """\nТекст: "{text}"\nПричина жалобы: {reason}\n""",'
                
                plan_next = "не указано"
                if review["plan_next"] is True:
                    plan_next = "да"
                elif review["plan_next"] is False:
                    plan_next = "нет"
                plan_prompt = f'    """\nТекст: "{text}"\nПоедет во второй раз: {plan_next}\n""",'

                # Записываем в файлы
                f_comp.write(complaint_prompt + '\n')
                f_score.write(score_prompt + '\n')
                f_reason.write(reason_prompt + '\n')
                f_plan.write(plan_prompt + '\n')

            print(f"Обработано {min(offset + batch_size, total_count)} из {total_count} отзывов")

        # Закрываем списки промптов
        f_comp.write(']\n')
        f_score.write(']\n')
        f_reason.write(']\n')
        f_plan.write(']\n')

    print("Промпты успешно сохранены в файлы!")

if __name__ == "__main__":
    asyncio.run(main())