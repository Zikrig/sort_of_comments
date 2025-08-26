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
    
    classifier = ReviewClassifier(db_manager, batch_size=30)
    results = await classifier.process_all_reviews(save_to_file=True)
    

if __name__ == "__main__":
    asyncio.run(main())