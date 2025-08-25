import asyncio
from dotenv import load_dotenv
import os

from db.database_manager import DatabaseManager

load_dotenv()


# Пример использования
async def main():
    # Подключение к базе данных
    # Формат строки подключения: mysql+asyncmy://user:password@host/database
    db_url = os.getenv('DB_LINK')
    db_manager = DatabaseManager(db_url)
    
    # Инициализация таблиц (если они не существуют)
    await db_manager.init_models()
    
    # Пример: получение данных заказа
    order_data = await db_manager.get_order_data(315185)
    print("Данные заказа:", order_data)
    
    # Пример: добавление нового заказа
    # result = await db_manager.add_new_order(
    #     order_id=123456,
    #     score=5,
    #     complaint=True,
    #     plan_next=False,
    #     review_text="Отличный тур!",
    #     complaint_types=[1, 3]  # ID типов жалоб из r_complaint_type
    # )
    # print("Результат добавления заказа:", result)
    
    # # Пример: обновление данных заказа
    # result = await db_manager.create_or_update_order(
    #     order_id=123456,
    #     score=4,
    #     complaint=False,
    #     plan_next=True,
    #     review_text="Обновленный комментарий",
    #     complaint_types=[2, 5]
    # )
    # print("Результат обновления заказа:", result)
    
    # # Пример: обновление данных заказа без комментария
    # result = await db_manager.create_or_update_order_without_review(
    #     order_id=123456,
    #     score=3,
    #     complaint=True,
    #     plan_next=False,
    #     complaint_types=[1, 4, 6]
    # )
    # print("Результат обновления без комментария:", result)

if __name__ == "__main__":
    asyncio.run(main())