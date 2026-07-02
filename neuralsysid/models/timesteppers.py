"""This module implements classes for different time-stepping methods. This
includes discrete system descriptions as well as continuous system descriptions
discretized using different numerical integration methods."""

import abc

import torch
from torch import Tensor, nn
import torchdiffeq as tdeq
from torchdiffeq._impl.adjoint import find_parameters

from .statedynamics import DynamicsFunction


class TimeStepper(nn.Module, abc.ABC):
    """Abstract base class for time-stepping methods. Used to output the state
    at time t_k+1 given the state and control input at time t_k. At this point,
    time-steps are fixed for every instance of the time-stepper ad should not be
    altered after initialization.

    Args:
        dynamics (DynamicsFunction): The dynamics function to be used for the
            time-stepping method.
    """

    def __init__(self, dynamics: DynamicsFunction):
        super().__init__()
        self.dynamics = dynamics

    @abc.abstractmethod
    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the time-stepper.

        Args:
            x (Tensor): State at time t_k.
            u (Tensor): Control input at time t_k.

        Returns:
            Tensor: The output (state) at the t_{k+1}.
        """


class Discrete(TimeStepper):
    """Discrete time-stepper. Here, the dynamics function represents a discrete
    model x(t_{k+1}) = f(x(t_k), u(t_k)).

    Args:
        dynamics (DynamicsFunction): The dynamics function to be used for the
            time-stepping method.
        delta_t (float, optional): Time step size. Only used for descriptive
            purpose this class. Defaults to 1.0.
    """

    def __init__(self, dynamics: DynamicsFunction, delta_t: float = 1.0):
        super().__init__(dynamics)
        self.delta_t = delta_t

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the time-stepper.

        Args:
            x (Tensor): State at time t_k.
            u (Tensor): Control input at time t_k.

        Returns:
            Tensor: The output (state) at the t_{k+1}.
        """
        out = self.dynamics(x, u)
        return out


class Residual(TimeStepper):
    """Discrete time-stepper. Here, the dynamics function represents a discrete
    model with residual formulation x(t_{k+1}) = x(t_k) + f(x(t_k), u(t_k)).

    Args:
        dynamics (DynamicsFunction): The dynamics function to be used for the
            time-stepping method.
        delta_t (float, optional): Time step size. Only used for descriptive
            purpose this class. Defaults to 1.0.
    """

    def __init__(self, dynamics: DynamicsFunction, delta_t: float = None):
        super().__init__(dynamics)
        self.delta_t = delta_t

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the time-stepper.

        Args:
            x (Tensor): State at time t_k.
            u (Tensor): Control input at time t_k.

        Returns:
            Tensor: The output (state) at the t_{k+1}.
        """
        out = x + self.dynamics(x, u)
        return out


class Euler(TimeStepper):
    """Euler time-stepper. Here, the dynamics function represents a Euler-
    discretized model x(t_{k+1}) = x(t_k) + delta_t * f(x(t_k), u(t_k)). It
    makes use of the torchdiffeq library to perform the integration.

    Args:
        dynamics (DynamicsFunction): The dynamics function to be used for the
            time-stepping method.
        delta_t (float, optional): Time step size. Defaults to 1.0.
        adjoint (bool, optional): Whether to use the adjoint method for
            backpropagation. Defaults to False.
    """

    def __init__(
        self,
        dynamics: DynamicsFunction,
        delta_t: float = 1.0,
        adjoint: bool = False,
    ):
        super().__init__(dynamics)
        self.delta_t = delta_t
        self.adjoint = adjoint

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the time-stepper.

        Args:
            x (Tensor): State at time t_k.
            u (Tensor): Control input at time t_k.
        Returns:
            Tensor: The output (state) at the t_{k+1}.
        """

        def dyn_fct(t, x):
            return self.dynamics(x, u)

        time = torch.tensor([0, self.delta_t], dtype=torch.float32)
        if self.adjoint:
            out = tdeq.odeint_adjoint(
                dyn_fct,
                x,
                method="euler",
                t=time,
                adjoint_params=find_parameters(self.dynamics),
            )
        else:
            out = tdeq.odeint(
                dyn_fct,
                x,
                method="euler",
                t=time,
            )
        return out[-1]


class RungeKutta4(TimeStepper):
    """RK4 time-stepper. Here, the dynamics function is discretized during
    numerical integration with a Runge-Kutta 4 with 3/8 rule (fixed step,
    explicite). It makes use of the torchdiffeq library to perform the
    integration.

    Args:
        dynamics (DynamicsFunction): The dynamics function to be used for the
            time-stepping method.
        delta_t (float, optional): Time step size. Defaults to 1.0.
        adjoint (bool, optional): Whether to use the adjoint method for
            backpropagation. Defaults to False.
    """

    def __init__(
        self,
        dynamics: DynamicsFunction,
        delta_t: float = 1.0,
        adjoint: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dynamics)
        self.delta_t = delta_t
        self.adjoint = adjoint
        self.device = device

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the time-stepper.

        Args:
            x (Tensor): State at time t_k.
            u (Tensor): Control input at time t_k.

        Returns:
            Tensor: The output (state) at the t_{k+1}.
        """

        def dyn_fct(t, x):
            return self.dynamics(x, u)

        time = torch.tensor(
            [0, self.delta_t], dtype=torch.float32, device=self.device
        )

        if self.adjoint:
            out = tdeq.odeint_adjoint(
                dyn_fct,
                x,
                method="rk4",
                t=time,
                adjoint_params=find_parameters(self.dynamics),
            )
        else:
            out = tdeq.odeint(
                dyn_fct,
                x,
                method="rk4",
                t=time,
            )
        return out[-1]


class Dopri5(TimeStepper):
    """Dopri5 time-stepper. Here, the dynamics function is discretized during
    numerical integration with a Dormand-Prince-Shampine (adaptive step,
    explicite). It makes use of the torchdiffeq library to perform the
    integration.

    Args:
        dynamics (DynamicsFunction): The dynamics function to be used for the
            time-stepping method.
        delta_t (float, optional): Time step size. Defaults to 1.0.
        adjoint (bool, optional): Whether to use the adjoint method for
            backpropagation. Defaults to True.
    """

    def __init__(
        self,
        dynamics: DynamicsFunction,
        delta_t: float = 1.0,
        adjoint: bool = True,
        rtol=1e-4,
        atol=1e-7,
    ):
        super().__init__(dynamics)
        self.delta_t = delta_t
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the time-stepper.

        Args:
            x (Tensor): State at time t_k.
            u (Tensor): Control input at time t_k.

        Returns:
            Tensor: The output (state) at the t_{k+1}.
        """

        def dyn_fct(t, x):
            return self.dynamics(x, u)

        time = torch.tensor([0, self.delta_t], dtype=torch.float32)
        if self.adjoint:
            out = tdeq.odeint_adjoint(
                dyn_fct,
                x,
                method="dopri5",
                t=time,
                adjoint_params=find_parameters(self.dynamics),
                rtol=self.rtol,
                atol=self.atol,
            )
        else:
            out = tdeq.odeint(
                dyn_fct,
                x,
                method="dopri5",
                t=time,
                rtol=self.rtol,
                atol=self.atol,
            )
        return out[-1]


class Dopri8(TimeStepper):
    """Dopri5 time-stepper. Here, the dynamics function is discretized during
    numerical integration with a Dormand-Prince-Shampine (adaptive step,
    explicite). It makes use of the torchdiffeq library to perform the
    integration.

    Args:
        dynamics (DynamicsFunction): The dynamics function to be used for the
            time-stepping method.
        delta_t (float, optional): Time step size. Defaults to 1.0.
        adjoint (bool, optional): Whether to use the adjoint method for
            backpropagation. Defaults to True.
    """

    def __init__(
        self,
        dynamics: DynamicsFunction,
        delta_t: float = 1.0,
        adjoint: bool = True,
        rtol=1e-4,
        atol=1e-7,
    ):
        super().__init__(dynamics)
        self.delta_t = delta_t
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the time-stepper.

        Args:
            x (Tensor): State at time t_k.
            u (Tensor): Control input at time t_k.

        Returns:
            Tensor: The output (state) at the t_{k+1}.
        """

        def dyn_fct(t, x):
            return self.dynamics(x, u)

        time = torch.tensor([0, self.delta_t], dtype=torch.float32)
        if self.adjoint:
            out = tdeq.odeint_adjoint(
                dyn_fct,
                x,
                method="dopri8",
                t=time,
                adjoint_params=find_parameters(self.dynamics),
                rtol=self.rtol,
                atol=self.atol,
            )
        else:
            out = tdeq.odeint(
                dyn_fct,
                x,
                method="dopri8",
                t=time,
                rtol=self.rtol,
                atol=self.atol,
            )
        return out[-1]
