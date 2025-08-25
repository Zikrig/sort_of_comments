import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from db.models import RReview, RRating, RRatingToComplaintType, RComplaintType

class BatchProcessor:
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    async def get_reviews_batch(
        self, 
        batch_size: int = 1000, 
        offset: int = 0,
        with_rating_info: bool = True,
        with_complaint_types: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Получить пачку отзывов с дополнительной информацией
        
        Args:
            batch_size: Размер пачки
            offset: Смещение
            with_rating_info: Включать информацию о рейтинге
            with_complaint_types: Включать информацию о типах жалоб
        
        Returns:
            Список словарей с данными отзывов
        """
        async with self.db_manager.async_session() as session:
            # Базовый запрос
            stmt = select(RReview).where(RReview.text.isnot(None))
            
            # Добавляем подгрузку связанных данных при необходимости
            if with_rating_info:
                stmt = stmt.options(selectinload(RReview.rating))
            
            if with_complaint_types and with_rating_info:
                stmt = stmt.options(
                    selectinload(RReview.rating).selectinload(
                        RRating.complaint_types
                    ).selectinload(RRatingToComplaintType.complaint_type)
                )
            
            # Добавляем пагинацию
            stmt = stmt.offset(offset).limit(batch_size)
            
            # Выполняем запрос
            result = await session.execute(stmt)
            reviews = result.scalars().all()
            
            # Преобразуем в список словарей
            return await self._format_reviews(reviews, with_rating_info, with_complaint_types)
    
    async def _format_reviews(
        self, 
        reviews: List[RReview], 
        with_rating_info: bool,
        with_complaint_types: bool
    ) -> List[Dict[str, Any]]:
        """Форматирование отзывов в словари"""
        formatted_reviews = []
        
        for review in reviews:
            review_data = {
                "review_id": review.id,
                "text": review.text,
                "rating_id": review.rating_id,
            }
            
            # Добавляем информацию о рейтинге, если нужно
            if with_rating_info and review.rating:
                review_data.update({
                    "order_id": review.rating.orderId,
                    "score": review.rating.score,
                    "complaint": review.rating.complaint,
                    "plan_next": review.rating.planNext,
                })
                
                # Добавляем информацию о типах жалоб, если нужно
                if with_complaint_types:
                    complaint_types = []
                    for comp_type in review.rating.complaint_types:
                        complaint_types.append({
                            "id": comp_type.complaint_type.id,
                            "name": comp_type.complaint_type.name
                        })
                    review_data["complaint_types"] = complaint_types
            
            formatted_reviews.append(review_data)
        
        return formatted_reviews
    
    async def stream_reviews(
        self, 
        batch_size: int = 1000,
        with_rating_info: bool = True,
        with_complaint_types: bool = True
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        Асинхронный генератор для потоковой обработки отзывов
        
        Args:
            batch_size: Размер пачки
            with_rating_info: Включать информацию о рейтинге
            with_complaint_types: Включать информацию о типах жалоб
        
        Yields:
            Пачки отформатированных отзывов
        """
        offset = 0
        
        while True:
            batch = await self.get_reviews_batch(
                batch_size, offset, with_rating_info, with_complaint_types
            )
            
            if not batch:
                break
                
            yield batch
            offset += batch_size
    
    async def get_total_reviews_count(self) -> int:
        """Получить общее количество отзывов с текстом"""
        async with self.db_manager.async_session() as session:
            stmt = select(RReview).where(RReview.text.isnot(None))
            result = await session.execute(stmt)
            return len(result.scalars().all())