from pathlib import Path

import ollama


class Agent:
    def __init__(self) -> None:
        self.history = []
        self.buffer = []

        prompt_path = Path(__file__).parent / "sys_prompt.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

        self.history.append({"role": "system", "content": system_prompt})

    def response(self):
        text = " ".join(self.buffer)
        self.buffer = []
        
            
        self.history.append({"role": "user", "content": text})
        response = ollama.chat(model="llama2", messages=self.history)
        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def add_to_buffer(self, text):
        self.buffer.append(text)
