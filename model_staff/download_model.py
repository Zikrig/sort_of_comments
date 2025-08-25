from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-3-270m-it"  # версия для чата

# 1. Явно указываем CPU для всех компонентов
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

# 2. Убираем .to("cuda"), используем CPU
input_text = "Как развернуть Gemma 3 270M?"
inputs = tokenizer(input_text, return_tensors="pt") 

# 3. Генерация на CPU
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))