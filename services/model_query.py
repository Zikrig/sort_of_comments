import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

class ModelQuery:
    def __init__(
        self,
        model_name: str,
        shots: List[str],
        max_shots: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        model_name: имя модели HuggingFace
        shots: список few-shot примеров (строки)
        max_shots: сколько шотов использовать
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

        self.shots = shots[:max_shots]

    def build_prompt(self, question: str) -> str:
        """Собираем промпт из шотов и вопроса"""
        base_prompt = "\n".join(self.shots) if self.shots else ""
        return f"{base_prompt}\n{question}" if base_prompt else question

    def query(self, question: str, max_new_tokens: int = 10) -> str:
        """
        Возвращает текстовый ответ модели
        """
        prompt = self.build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        # отрезаем prompt, оставляем только ответ
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        return answer
