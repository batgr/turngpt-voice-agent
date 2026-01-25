import ollama


class Agent:
    def __init__(self) -> None:
        self.history = []

    def response(self, text):
        self.history.append({"role": "user", "content": text})
        response = ollama.chat(model="llama2", messages=self.history)
        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply
