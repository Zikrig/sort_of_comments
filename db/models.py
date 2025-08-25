import asyncio
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import select, update, insert
from typing import Optional, List, Dict, Any

# Базовый класс для моделей
Base = declarative_base()

# Модели данных
class RComplaintType(Base):
    __tablename__ = 'r_complaint_type'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)

class RRating(Base):
    __tablename__ = 'r_rating'
    
    id = Column(Integer, primary_key=True)
    orderId = Column(Integer, nullable=False)
    score = Column(Integer)
    complaint = Column(Boolean)
    planNext = Column(Boolean)
    
    # Связи с другими таблицами
    review = relationship("RReview", back_populates="rating", uselist=False)
    complaint_types = relationship("RRatingToComplaintType", back_populates="rating")

class RReview(Base):
    __tablename__ = 'r_review'
    
    id = Column(Integer, primary_key=True)
    rating_id = Column(Integer, ForeignKey('r_rating.id'), name='rating')
    text = Column(Text)
    
    # Связь с рейтингом
    rating = relationship("RRating", back_populates="review")

class RRatingToComplaintType(Base):
    __tablename__ = 'r_rating_to_complaint_type'
    
    id = Column(Integer, primary_key=True)
    rating_id = Column(Integer, ForeignKey('r_rating.id'), name='rating')
    type_id = Column(Integer, ForeignKey('r_complaint_type.id'), name='type')
    
    # Связи с другими таблицами
    rating = relationship("RRating", back_populates="complaint_types")
    complaint_type = relationship("RComplaintType")

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