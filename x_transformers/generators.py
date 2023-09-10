import torch
from torch.nn.functional import softmax
from abc import ABC, abstractmethod
from .utils import subsequent_mask, scaled_softmax


def generate(model, x, iterations, stop_token=None, temperature=None, top_p=None, top_k=None):
    if temperature is None:
        generator = Greedy(model, stop_token, temperature, top_p, top_k)
    else:
        generator = Sampler(model, stop_token, temperature, top_p, top_k)
    return generator(x, iterations)


class _Generator(ABC):
    def __init__(self, model, stop_token, temperature, top_p, top_k) -> None:
        super(ABC).__init__()
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop_token = stop_token

    def __call__(self, x, iterations):
        all_logits = []
        for _ in range(iterations):
            logits, probs = self.model(x, subsequent_mask(x.size(1)))
            logits, probs = logits[0, -1], probs[0, -1]
            next_word = self.choose_next_word(logits, probs)
            x = torch.cat([x, torch.zeros(1, 1).to(
                torch.int64).fill_(next_word)], dim=1)
            all_logits.append(logits.unsqueeze(0))
            if next_word == self.stop_token:
                break
        return torch.cat(all_logits), x

    @abstractmethod
    def choose_next_word(self, logits, probs):
        pass


class Greedy(_Generator):
    def __init__(self, model, stop_token, temperature, top_p, top_k) -> None:
        super().__init__(model, stop_token, temperature, top_p, top_k)

    def choose_next_word(self, _, probs):
        return torch.max(probs, dim=0)[1]


class Sampler(_Generator):
    fill = -1e9

    def __init__(self, model, stop_token, temperature, top_p, top_k) -> None:
        super().__init__(model, stop_token, temperature, top_p, top_k)

    def choose_next_word(self, logits, _):
        logits = logits / self.temperature

        if self.top_k is not None and self.top_k > 0:
            self.top_k = min(self.top_k, logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, self.top_k)[
                0][..., -1, None]
            logits[indices_to_remove] = self.fill

        if self.top_p is not None and self.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=True)
            cumulative_probs = torch.cumsum(
                softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[...,
                                     1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = self.fill
        probs = scaled_softmax(logits, self.temperature)
        return torch.multinomial(probs, num_samples=1)[0]
