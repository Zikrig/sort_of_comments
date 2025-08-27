import pandas as pd
import random
import re
from pathlib import Path
from dotenv import load_dotenv
import os

from db.database_manager import DatabaseManager
from services.batch_processor import BatchProcessor

load_dotenv()

# Подключение к базе данных
# Формат строки подключения: mysql+asyncmy://user:password@host/database
db_url = os.getenv('DB_LINK')
db_manager = DatabaseManager(db_url)


async def prepare_dataset(batch_processor, out_path="dataset.csv"):
    data = []

    # Загружаем все отзывы батчами
    async for batch in batch_processor.stream_reviews(batch_size=5000):
        data.extend(batch)

    df = pd.DataFrame(data)

    # --- Очистка ---
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]

    # --- Фильтрация нерелевантных текстов ---
    ignore_patterns = [
        r"нет\s+отзыва",
        r"нет\s+оценки",
        r"не\s+ответил",
        r"нет\s+комментария",
        r"отзыва\s+нет",
        r"оценки\s+нет",
    ]
    # составляем регулярку через ИЛИ
    combined_pattern = "(" + "|".join(ignore_patterns) + ")"
    mask_ignore = df["text"].str.lower().str.contains(combined_pattern, regex=True)

    df_ignore = df[mask_ignore]
    df_other = df[~mask_ignore]

    # случайно берём только 5% таких строк
    df_ignore_sampled = df_ignore.sample(frac=0.05, random_state=42) if not df_ignore.empty else pd.DataFrame()

    # объединяем
    df = pd.concat([df_other, df_ignore_sampled], ignore_index=True)

    # --- Перемешивание чанками по 500 ---
    chunks = []
    for i in range(0, len(df), 500):
        chunk = df.iloc[i:i+500].sample(frac=1, random_state=random.randint(0, 10_000))
        chunks.append(chunk)

    df_final = pd.concat(chunks, ignore_index=True)

    # --- Сохранение ---
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Сохранено {len(df_final)} строк в {out_path}")


dp = BatchProcessor(db_manager)

if __name__ == "__main__":
    import asyncio
    asyncio.run(prepare_dataset(dp, out_path="data/dataset.csv"))