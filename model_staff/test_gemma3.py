from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Загрузка модели и токенизатора
model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем pad_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

print("Чат-бот запущен! Введите ваше сообщение (или 'quit' для выхода):")
print("-" * 50)

while True:
    # Получаем ввод пользователя
    user_input = input("Вы: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
        
    if not user_input:
        continue
    
    # Формируем промпт в формате, который ожидает модель
    prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    
    # Токенизируем входные данные
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Генерируем ответ
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Декодируем ответ, удаляя промпт
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    
    # Выводим ответ модели
    print(f"Бот: {response}")
    print("-" * 50)

print("Диалог завершен. До свидания!")