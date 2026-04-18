from collections import deque

class SequenceBuffer:
    def __init__(self, size=30):
        self.buffer = deque(maxlen=size)

    def add(self, features):
        if features is not None:
            self.buffer.append(features)

    def get_all(self):
        return list(self.buffer)