import torch
from torch.nn.functional import softmax
from abc import ABC, abstractmethod
from .utils import subsequent_mask, scaled_softmax


def generate(model, x, iterations, temperature=None, top_p=None, top_k=None):
    if temperature is None:
        generator = Greedy(model)
    else:
        generator = Sampler(model)
    return generator(x, iterations, temperature=temperature, top_p=top_p, top_k=top_k)


class _Generator(ABC):
    def __init__(self, model) -> None:
        super(ABC).__init__()
        self.model = model

    @abstractmethod
    def __call__(self, x, iterations, temperature=None, top_p=None, top_k=None):
        pass

    @staticmethod
    def get_updated_x(x, next_token):
        return torch.cat([x, torch.zeros(1, 1).to(torch.int64).fill_(next_token)], dim=1)


class Greedy(_Generator):
    def __init__(self, model) -> None:
        super().__init__(model)

    def generate_probs(self, x):
        return self.model(x, subsequent_mask(x.size(1)))[1]

    def __call__(self, x, iterations, temperature=None, top_p=None, top_k=None):
        for _ in range(iterations):
            probs = self.generate_probs(x)
            _, next_word = torch.max(probs[0, -1], dim=0)
            x = _Generator.get_updated_x(x, next_word)
        return x


class Sampler(_Generator):
    def __init__(self, model) -> None:
        super().__init__(model)

    def generate_logits(self, x):
        return self.model(x, subsequent_mask(x.size(1)))[0]

    def __call__(self, x, iterations, temperature=1, top_p=None, top_k=None):
        fill = -1e9
        for _ in range(iterations):
            logits = self.generate_logits(x)
            logits = logits[0, -1]
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True)
                cumulative_probs = torch.cumsum(
                    softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = fill
            if top_k is not None:
                logits[logits < torch.sort(
                    logits)[0][top_k].unsqueeze(dim=-1)] = fill
            probs = scaled_softmax(logits, temperature)
            next_token = torch.multinomial(probs, num_samples=1)[0]
            x = _Generator.get_updated_x(x, next_token)
        return x
