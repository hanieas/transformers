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
            logits = logits / temperature
            assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
            top_k = min(top_k, logits.size(-1))  # Safety check
            if top_k > 0:
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = fill

            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = fill
            probs = scaled_softmax(logits, temperature)
            next_token = torch.multinomial(probs, num_samples=1)[0]
            x = _Generator.get_updated_x(x, next_token)
        return x
