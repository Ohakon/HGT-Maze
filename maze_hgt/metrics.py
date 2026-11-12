"""Aggregate metrics with sample and positive counts."""
from collections import defaultdict
import torch
from torch import Tensor


class Metrics:
    def __init__(self):
        self.means = defaultdict(lambda: [0., 0])
        self.confusions = defaultdict(lambda: [0, 0, 0, 0])
        self.errors = defaultdict(lambda: [0., 0., 0])

    def mean(self, name: str, value: float, weight: int = 1) -> None:
        self.means[name][0] += value * weight
        self.means[name][1] += weight

    def binary(self, name: str, prediction: Tensor, label: Tensor) -> None:
        predicted, actual = prediction.detach().bool().reshape(-1), label.detach().bool().reshape(-1)
        if not len(actual):
            return
        table = torch.bincount(actual.long() * 2 + predicted.long(), minlength=4).cpu().tolist()
        for index, n in enumerate(table):
            self.confusions[name][index] += n

    def error(self, name: str, prediction: Tensor, label: Tensor) -> None:
        delta = (prediction.detach() - label.detach()).reshape(-1)
        if len(delta):
            self.errors[name][0] += delta.abs().sum().item()
            self.errors[name][1] += delta.square().sum().item()
            self.errors[name][2] += len(delta)

    def flat(self) -> dict[str, float | int]:
        result = {key: sum_value / n for key, (sum_value, n) in self.means.items() if n}
        for key, (tn, fp, fn, tp) in self.confusions.items():
            n = tn + fp + fn + tp
            result.update({f"{key}.accuracy": (tp + tn) / n,
                           f"{key}.f1": 2 * tp / max(1, 2 * tp + fp + fn),
                           f"{key}.precision": tp / max(1, tp + fp),
                           f"{key}.recall": tp / max(1, tp + fn),
                           f"{key}.positives": tp + fn, f"{key}.n": n})
            if tn + fp:
                result[f"{key}.specificity"] = tn / (tn + fp)
                if tp + fn:
                    result[f"{key}.balanced_accuracy"] = .5 * (tp / (tp + fn) + tn / (tn + fp))
        for key, (absolute, square, n) in self.errors.items():
            result.update({f"{key}.mae": absolute / n, f"{key}.mse": square / n, f"{key}.n": n})
        return result


def weight_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
