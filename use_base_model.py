import os
import asyncio
from dotenv import load_dotenv

from db.database_manager import DatabaseManager
from services.review_classifier import ReviewClassifier

load_dotenv()

async def main():
    db_url = os.getenv('DB_LINK')
    db_manager = DatabaseManager(db_url)
    await db_manager.init_models()
    
    classifier = ReviewClassifier(db_manager, batch_size=100)
    results = await classifier.process_all_reviews(update_db=False)
    
    for res in results[:10]:  # Выводим первые 10 результатов для примера
        print(f"ID отзыва: {res['review_id']}\n"
              f"Текст: '{res['text']}'\n"
              f"Жалоба: {res['is_complaint']}\n"
              f"Оценка: {res['score']}\n"
              f"Причина: {res['reason']}\n"
              f"Поедет еще: {res['plan_next']}\n"
              f"{'-'*70}")

if __name__ == "__main__":
    asyncio.run(main())