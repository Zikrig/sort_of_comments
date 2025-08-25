from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from train_data import examples, examples_to_teach, test_texts, neutral_but_risky_phrases

model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Узнаем ID токенов для "1" и "0"
token_id_1 = tokenizer.convert_tokens_to_ids("1")  # 297
token_id_0 = tokenizer.convert_tokens_to_ids("0")  # 296

def classify_text(texts: list) -> list:
    
    results = []
    for text in texts:
        # Формируем промпт с примерами + текущий текст
        prompt = '\n'.join(examples) + f"\nТекст: \"{text}\" Ответ:"
        
        # Токенизируем ВСЕ примеры + текущий текст
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
        
        # Получаем логиты для ПОСЛЕДНЕГО токена (после "Ответ:")
        with torch.no_grad():
            outputs = model(**inputs)
            last_token_logits = outputs.logits[0, -1, :]
        
        # Вычисляем вероятности только для "0" и "1"
        probs = torch.softmax(
            last_token_logits[[token_id_0, token_id_1]], 
            dim=-1
        )
        
        # Вероятности классов
        prob_0 = probs[0].item()  # P(0)
        prob_1 = probs[1].item()  # P(1)
        
        # Определяем класс (порог 0.5)
        class_id = 1 if prob_1 > 0.5 else 0
        
        results.append({
            "text": text,
            "class": class_id,
            "prob_anger": prob_1,
            "prob_normal": prob_0
        })
    
    return results

# ===== ТЕСТ НА 40 ТЕКСТАХ =====


# Классификация всех 40 текстов
classifications = classify_text(test_texts)
# classifications = classify_text(neutral_but_risky_phrases)

# Вывод результатов
for res in classifications:
    print(f"Текст: '{res['text']}'\n"
          f"Класс: {res['class']} ({'злость' if res['class'] == 1 else 'норма'}), "
          f"P(злость)={res['prob_anger']:.4f}, "
          f"P(норма)={res['prob_normal']:.4f}\n"
          f"{'-'*70}")