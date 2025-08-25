import os
from dotenv import load_dotenv
from huggingface_hub import login

# Загружаем переменные из .env файла
load_dotenv()

# Получаем токен из переменной окружения
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)
else:
    print("Предупреждение: HF_TOKEN не найден в .env файле")