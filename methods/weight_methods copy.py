# import torch.optim as optim
# import torch.nn as nn
import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, least_squares

from methods.min_norm_solvers import MinNormSolver, gradient_normalizers
from methods.abstract_weighting import AbsWeighting

EPS = 1e-8  # for numerical stability


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm=1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter],
                              torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class PMGD(WeightMethod):  # Just use sol in the backward, it should be used for our theory
    def __init__(
        self, n_tasks, device: torch.device, params="shared", normalization="none", pmgd_x=8, sol=None, alpha=0.9
    ):
        super().__init__(n_tasks, device=device)
        self.solver = MinNormSolver()
        assert params in ["shared", "last", "rep"]
        self.params = params
        assert normalization in ["norm", "loss", "loss+", "none"]
        self.normalization = normalization
        self.pmgd_x = pmgd_x
        self.backward_counter = 0
        self.sol = sol
        self.prev_weights = None
        self.alpha = alpha

    @staticmethod
    def _flattening(grad):
        return torch.cat([g.reshape(-1) for g in grad], dim=0)

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter],
                                 torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter],
                                        torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter],
                                      torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter],
                              torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        if self.backward_counter % self.pmgd_x == 0:
            loss, extra_outputs = self.get_weighted_loss0(
                losses=losses,
                shared_parameters=shared_parameters,
                task_specific_parameters=task_specific_parameters,
                last_shared_parameters=last_shared_parameters,
                representation=representation,
                **kwargs,
            )
        else:
            loss, extra_outputs = self.get_weighted_loss1(
                losses=losses,
                **kwargs,
            )

        self.backward_counter += 1

        # Clip gradients for all parameters
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        if 'weights' in extra_outputs and isinstance(extra_outputs['weights'], torch.Tensor):
            if self.prev_weights is None:
                self.prev_weights = extra_outputs['weights'].detach().clone()
            else:
                current_weights = extra_outputs['weights'].detach().clone()
                extra_outputs['weights'] = self.alpha * current_weights + \
                    (1 - self.alpha) * self.prev_weights
                self.prev_weights = current_weights

        loss.backward()
        return loss, extra_outputs

    def get_weighted_loss0(
        self,
        losses,
        shared_parameters=None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ):
        params = dict(rep=representation, shared=shared_parameters,
                      last=last_shared_parameters)[self.params]
        grads = {i: [torch.flatten(grad) for grad in torch.autograd.grad(
            loss, params, retain_graph=True)] for i, loss in enumerate(losses)}

        gn = gradient_normalizers(grads, losses, self.normalization)
        grads = {t: [g / gn[t] for g in grads[t]] for t in range(self.n_tasks)}

        self.sol, min_norm = self.solver.find_min_norm_element(
            [grads[t] for t in range(len(grads))])
        self.sol = self.sol * self.n_tasks
        weighted_loss = torch.sum(torch.stack(
            [losses[i] * self.sol[i] for i in range(len(self.sol))]))

        return weighted_loss, dict(weights=torch.from_numpy(self.sol.astype(np.float32)))

    def get_weighted_loss1(self, losses, **kwargs):
        weighted_loss = torch.sum(torch.stack(
            [losses[i] * self.sol[i] for i in range(len(self.sol))]))
        return weighted_loss, dict(weights=torch.from_numpy(self.sol.astype(np.float32)))



class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


METHODS = dict(
    pmgd=PMGD
)
