from pathlib import Path

import ollama


class Agent:
    def __init__(self, model="llama3.2:3b-instruct-q4_K_M") -> None:
        self.model = model
        system_prompt = (Path(__file__).parent / "sys_prompt.txt").read_text(encoding="utf-8")
        self.history = [{"role": "system", "content": system_prompt}]

    def turns(self):
        """The spoken dialog so far, without the system prompt"""
        return [message["content"] for message in self.history[1:]]

    def response(self, text):
        self.history.append({"role": "user", "content": text})
        reply = ollama.chat(model=self.model, messages=self.history)["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply
