import asyncio
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import select, update, insert
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import selectinload

from db.models import Base, RRating, RReview, RComplaintType, RRatingToComplaintType

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_async_engine(
            database_url,
            echo=False,  # Установите True для отладки SQL-запросов
            future=True
        )
        self.async_session = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )
    
    async def init_models(self):
        """Создание таблиц (если они не существуют)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_all_reviews(self, offset: int = 0, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Получить все отзывы из таблицы r_review с непустым текстом, с пагинацией.
        """
        async with self.async_session() as session:
            stmt = select(RReview).options(
                selectinload(RReview.rating)
            ).where(RReview.text.isnot(None)).offset(offset).limit(limit)
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
                for review in reviews
            ]

    async def get_order_data(self, order_id: int) -> Optional[Dict[str, Any]]:
        """Получить все данные по конкретному заказу"""
        async with self.async_session() as session:
            try:
                # Находим рейтинг по orderId с подгрузкой связанных данных
                stmt = select(RRating).where(RRating.orderId == order_id).options(
                    selectinload(RRating.review),
                    selectinload(RRating.complaint_types).selectinload(RRatingToComplaintType.complaint_type)
                )
                result = await session.execute(stmt)
                rating = result.scalar_one_or_none()
                
                if not rating:
                    return None
                
                # Получаем связанные данные
                order_data = {
                    "rating": {
                        "id": rating.id,
                        "orderId": rating.orderId,
                        "score": rating.score,
                        "complaint": rating.complaint,
                        "planNext": rating.planNext
                    },
                    "review": None,
                    "complaint_types": []
                }
                
                # Если есть отзыв
                if rating.review:
                    order_data["review"] = {
                        "id": rating.review.id,
                        "text": rating.review.text
                    }
                
                # Если есть типы жалоб
                for comp_type in rating.complaint_types:
                    order_data["complaint_types"].append({
                        "id": comp_type.complaint_type.id,
                        "name": comp_type.complaint_type.name
                    })
                
                return order_data
            except Exception as e:
                await session.rollback()
                raise e

    async def create_or_update_order(
        self, 
        order_id: int, 
        score: int, 
        complaint: bool, 
        plan_next: bool, 
        review_text: Optional[str] = None,
        complaint_types: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Создать или обновить данные заказа"""
        async with self.async_session() as session:
            try:
                # Проверяем, существует ли уже запись
                stmt = select(RRating).where(RRating.orderId == order_id)
                result = await session.execute(stmt)
                rating = result.scalar_one_or_none()
                
                # Если записи нет, создаем новую
                if not rating:
                    rating = RRating(
                        orderId=order_id,
                        score=score,
                        complaint=complaint,
                        planNext=plan_next
                    )
                    session.add(rating)
                    await session.flush()  # Получаем ID новой записи
                
                # Обновляем данные рейтинга
                rating.score = score
                rating.complaint = complaint
                rating.planNext = plan_next
                
                # Обрабатываем отзыв
                if review_text is not None:
                    if rating.review:
                        rating.review.text = review_text
                    else:
                        rating.review = RReview(text=review_text)
                
                # Обрабатываем типы жалоб
                if complaint_types is not None:
                    # Удаляем существующие связи
                    if rating.complaint_types:
                        for comp_type in rating.complaint_types:
                            await session.delete(comp_type)
                    
                    # Добавляем новые связи
                    for type_id in complaint_types:
                        comp_type = RRatingToComplaintType(
                            rating_id=rating.id,
                            type_id=type_id
                        )
                        session.add(comp_type)
                
                await session.commit()
                
                return {"status": "success", "rating_id": rating.id}
            except Exception as e:
                await session.rollback()
                raise e
    
    async def create_or_update_order_without_review(
        self, 
        order_id: int, 
        score: int, 
        complaint: bool, 
        plan_next: bool,
        complaint_types: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Создать или обновить данные заказа без комментария"""
        return await self.create_or_update_order(
            order_id, score, complaint, plan_next, None, complaint_types
        )
    
    async def add_new_order(
        self,
        order_id: int,
        score: int,
        complaint: bool,
        plan_next: bool,
        review_text: Optional[str] = None,
        complaint_types: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Добавить данные нового заказа (только если заказа с таким orderId еще нет)"""
        async with self.async_session() as session:
            try:
                # Проверяем, существует ли уже запись
                stmt = select(RRating).where(RRating.orderId == order_id)
                result = await session.execute(stmt)
                rating = result.scalar_one_or_none()
                
                if rating:
                    return {"status": "error", "message": "Order already exists"}
                
                # Создаем новую запись
                rating = RRating(
                    orderId=order_id,
                    score=score,
                    complaint=complaint,
                    planNext=plan_next
                )
                session.add(rating)
                await session.flush()  # Получаем ID новой записи
                
                # Добавляем отзыв, если указан
                if review_text:
                    review = RReview(
                        rating_id=rating.id,
                        text=review_text
                    )
                    session.add(review)
                
                # Добавляем типы жалоб, если указаны
                if complaint_types:
                    for type_id in complaint_types:
                        comp_type = RRatingToComplaintType(
                            rating_id=rating.id,
                            type_id=type_id
                        )
                        session.add(comp_type)
                
                await session.commit()
                
                return {"status": "success", "rating_id": rating.id}
            except Exception as e:
                await session.rollback()
                raise e