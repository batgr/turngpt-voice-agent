import ollama


class Agent:
    def __init__(self) -> None:
        self.history = []
        self.buffer = []

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
