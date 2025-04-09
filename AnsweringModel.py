from openai import OpenAI
from anthropic import Anthropic
import os


class AnsweringModel:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        api_key = os.getenv("AIML_API_KEY")
        if "claude" in model_name:
            base_url = "https://api.aimlapi.com/"
            self.api = Anthropic(auth_token=api_key, base_url=base_url)
        else:
            base_url = "https://api.aimlapi.com/v1"
            self.api = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature

    def generate_openai_response(self, user_prompt: str, system_prompt: str) -> str | None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self.api.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def generate_anthropic_response(self, user_prompt: str, system_prompt: str) -> str | None:
        try:
            response = self.api.messages.create(
                model=self.model_name,
                max_tokens=500,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def generate_response(self, user_prompt: str, system_prompt: str) -> str | None:
        if type(self.api) == Anthropic:
            return self.generate_anthropic_response(user_prompt=user_prompt, system_prompt=system_prompt)
        elif type(self.api) == OpenAI:
            return self.generate_openai_response(user_prompt=user_prompt, system_prompt=system_prompt)
        else:
            raise ValueError("Unsupported model type. Please use either OpenAI or Anthropic.")

    @staticmethod
    def parse_response(model_answer: str, task: dict):
        if model_answer in ["A", "B", "C", "D"]:
            parsed_answer = ["A", "B", "C", "D"].index(model_answer)
            correct_answer = task["answer"]
            correctness = parsed_answer == correct_answer
        else:
            first_token = model_answer[0]
            if first_token in ["A", "B", "C", "D"] and (model_answer[1] == "." or model_answer[1] == ")" or model_answer[1] == "\n"):
                parsed_answer = ["A", "B", "C", "D"].index(first_token)
                correct_answer = task["answer"]
                correctness = parsed_answer == correct_answer
            else:
                correctness = None
                parsed_answer = model_answer

        return parsed_answer, correctness


    @staticmethod
    def create_prompt(mmlu_task: dict, persona: str) -> tuple[str, str]:
        article = "an" if persona[0].lower() in "aeiou" else "a"
        system_prompt = (
            f"You are {article} {persona.strip()}. Your task is to answer a multiple-choice question"
            f" about {mmlu_task['subject'].replace('_', ' ').strip()}. Your response must include ONLY"
            f" the letter of the correct answer: A, B, C, or D. "
            f"Do not write any other text."
        )

        user_prompt = (
            f"{mmlu_task['question'].strip()}\nA. {mmlu_task['choices'][0].strip()}\nB. {mmlu_task['choices'][1].strip()}\nC. {mmlu_task['choices'][2].strip()}\nD. {mmlu_task['choices'][3].strip()}\nAnswer: "
        )
        return system_prompt, user_prompt


    def generate_full_response(self, mmlu_task: dict, persona: str):
        system_prompt, user_prompt = self.create_prompt(mmlu_task, persona)
        model_answer = self.generate_response(user_prompt=user_prompt, system_prompt=system_prompt)
        if model_answer is None or model_answer == "":
            return {}
        parsed_answer, correctness = self.parse_response(model_answer, mmlu_task)

        return {
            "task": mmlu_task,
            "model": self.model_name,
            "persona": persona,
            "model_answer": model_answer,
            "parsed_answer": parsed_answer,
            "correctness": correctness,
        }
