class Metric:
    """Lớp cơ sở cho mọi độ đo đánh giá"""
    def __init__(self):
        self.reset()

    def update(self, preds, labels):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def reset(self):
        self.preds = []
        self.labels = []