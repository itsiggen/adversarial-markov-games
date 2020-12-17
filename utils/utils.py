from foolbox.criteria import Criterion
from abc import ABC, abstractmethod
from typing import TypeVar, Any
import eagerpy as ep
import numpy as np
import torch


T = TypeVar("T")

class skl_callable():
    """Wraps a scikit learn model into a callable
    """
    def __init__(self, model):
        self.model = model
    
    def __call__(self, sample):
        return self.model.predict_proba(sample.reshape(1, -1))

class MisdirectedMisclassification(Criterion):
    """Targeted misclassification with a twist, the returned class
    is the second highest.
    """

    def __init__(self, target_classes: Any):
        super().__init__()
        self.target_classes: ep.Tensor = ep.astensor(target_classes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs
        
        outs = outputs_.squeeze().numpy()
        rank = np.argsort(outs)
        classes = torch.tensor([rank[-2]])
        assert classes.shape == self.target_classes.shape
        is_adv = classes == self.target_classes
        return restore_type(is_adv)
    
class iskl_adversarial(Criterion):
    """Targeted misclassification for sklearn models
    """

    def __init__(self, target_classes, model):
        super().__init__()
        self.target_classes = target_classes
        self.model = model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, perturbed):
        perturbed = perturbed.reshape(1, -1)
        print(perturbed)
        classes = self.model.predict(perturbed.numpy())
        print(classes)
        print(self.target_classes)
        is_adv = bool(classes == self.target_classes)
        return is_adv

def flatten(x: ep.Tensor, keep: int = 1) -> ep.Tensor:
    return x.flatten(start=keep)


def atleast_kd(x: ep.Tensor, k: int) -> ep.Tensor:
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)

class Distance(ABC):
    @abstractmethod
    def __call__(self, reference: T, perturbed: T) -> T:
        ...

    @abstractmethod
    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T:
        ...


class LpDistance(Distance):
    def __init__(self, p: float):
        self.p = p

    def __repr__(self) -> str:
        return f"LpDistance({self.p})"

    def __str__(self) -> str:
        return f"L{self.p} distance"

    def __call__(self, references: T, perturbed: T) -> T:
        """Calculates the distances from references to perturbed using the Lp norm.

        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.

        Returns:
            A 1D tensor with the distances from references to perturbed.
        """
        (x, y), restore_type = ep.astensors_(references, perturbed)
        norms = ep.norms.lp(flatten(y - x), self.p, axis=-1)
        return restore_type(norms)

    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T:
        """Clips the perturbations to epsilon and returns the new perturbed

        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.

        Returns:
            A tenosr like perturbed but with the perturbation clipped to epsilon.
        """
        (x, y), restore_type = ep.astensors_(references, perturbed)
        p = y - x
        if self.p == ep.inf:
            clipped_perturbation = ep.clip(p, -epsilon, epsilon)
            return restore_type(x + clipped_perturbation)
        norms = ep.norms.lp(flatten(p), self.p, axis=-1)
        norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
        factor = epsilon / norms
        factor = ep.minimum(1, factor)  # clipping -> decreasing but not increasing
        if self.p == 0:
            if (factor == 1).all():
                return perturbed
            raise NotImplementedError("reducing L0 norms not yet supported")
        factor = atleast_kd(factor, x.ndim)
        clipped_perturbation = factor * p
        return restore_type(x + clipped_perturbation)


l0 = LpDistance(0)
l1 = LpDistance(1)
l2 = LpDistance(2)
linf = LpDistance(ep.inf)
