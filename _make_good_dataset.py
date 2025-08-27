import re

from prompts import IS_COMPLAINT_PROMPT

# твой старый список
raw_prompts = IS_COMPLAINT_PROMPT

dataset = []

for item in raw_prompts:
    # объединяем многострочные элементы в одну строку
    text = " ".join(item.splitlines()).strip()
    
    # ищем метку Да/Нет
    match = re.search(r'Является ли текст жалобой\?\s*(Да|Нет)', text, re.I)
    if not match:
        continue
    
    label = 1 if match.group(1).lower() == "да" else 0
    
    # убираем все служебные части, оставляем только отзыв
    text = re.sub(r'Текст:\s*"', '', text)
    text = re.sub(r'Является ли текст жалобой\?.*', '', text)
    text = text.strip().strip('"')
    
    dataset.append({"text": text, "label": label})

# проверим
for d in dataset[:5]:
    print(d)
