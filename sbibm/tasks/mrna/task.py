from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pyro
import torch
from pyro import distributions as pdist

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task

from .frohlich_sde import *

class mrna(Task):
    def __init__(self):
        """mrna"""

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000011,  # observation 1
        ]

        super().__init__(
            dim_parameters=8,
            dim_data=60,
            name=Path(__file__).parent.name,
            name_display="mrna",
            num_observations=1,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        self.prior_params = {
            "loc": torch.tensor([-0.694, -3, 0.027, 5.704, 0.751, -0.164, 2.079, -2]),
            "covariance_matrix": 0.5*torch.eye(self.dim_parameters),
        }

        self.prior_dist = pdist.MultivariateNormal(**self.prior_params)
        self.prior_dist.set_default_validate_args(False)

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """        
        def simulator(parameters):
            numpy_array = batch_simulator(param_samples=parameters.numpy(), n_obs=self.dim_data, with_noise=True)
            tensor = torch.from_numpy(numpy_array)
            return(tensor)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

   

if __name__ == "__main__":
    task = mrna()
    task._setup()
