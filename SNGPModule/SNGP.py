"""
Implementation of a Spectral-normalized Gaussian Process transformer as presented in
`Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.
"""

# STD
import math
from typing import Tuple, Optional, Dict, Any, Type
import warnings

# EXT
from einops import rearrange
import torch
from torch import nn as nn
from torch.nn import functional as F
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun


class SNGP(nn.Module):
    """
    Spectral-normalized Gaussian Process output layer, as presented in
    `Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_. Requires underlying model to contain residual
    connections in order to maintain bi-Lipschitz constraint.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        ridge_factor: float,
        scaling_coefficient: float,
        beta_length_scale: float,
        kernel_amplitude: float,
        num_predictions: int,
        device: Device,
        **build_params,
    ):
        """
        Initialize a SNGPModule output layer.

        Parameters
        ----------
        hidden_size: int
            Hidden size of last regular network layer.
        last_layer_size: int
            Size of last layer before output layer. Called D_L in the original paper.
        output_size: int
            Size of output layer, so number of classes.
        ridge_factor: float
            Factor that identity sigma hat matrices of the SNGPModule layer are multiplied by.
        scaling_coefficient: float
            Momentum factor that is used when updating the sigma hat matrix of the SNGPModule layer during the last training
            epoch.
        beta_length_scale: float
            Factor for the variance parameter of the normal distribution all beta parameters of the SNGPModule layer are
            initialized from.
        kernel_amplitude: float
            Kernel amplitude used when computing GP features.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGPModule layer to come to the final prediction.
        device: Device
            Device the replication is performed on.
        """
        super().__init__()
        self.device = device

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ridge_factor = ridge_factor
        self.scaling_coefficient = scaling_coefficient
        self.beta_length_scale = beta_length_scale
        self.kernel_amplitude = kernel_amplitude
        self.num_predictions = num_predictions

        # ### Init parameters

        # Random, frozen output layer
        self.output = nn.Linear(self.hidden_size, self.output_size)
        # Change init of weights and biases following Liu et al. (2020)
        self.output.weight.data.normal_(0, 0.05)
        self.output.bias.data.uniform_(0, 2 * math.pi)

        # This layer is frozen right after init
        self.output.weight.requires_grad = False
        self.output.bias.requires_grad = False

        # Bundle all beta_k vectors into a matrix
        self.Beta = nn.Linear(output_size, output_size)
        self.Beta.weight.data.normal_(0, beta_length_scale)
        self.Beta.bias.data = torch.zeros(output_size)

        # Initialize inverse of sigma hat, one matrix in total to save memory
        self.sigma_hat_inv = torch.eye(output_size, device=self.device) * self.beta_length_scale
        self.sigma_hat = torch.zeros(output_size, output_size, device=device)
        self.inversed_sigma = False

        # Multivariate normal distributions that beta columns are sampled from after inverting sigma hat
        self.beta_dists = [None for _ in range(output_size)]

    def _get_features(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get posterior mean / logits and Phi feature matrix given an input.

        Parameters
        ----------
        x: torch.FloatTensor
            Last hidden state of underlying model.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor]
            Tensors of posterior mean and Phi matrix.
        """
        Phi = math.sqrt(2 * self.kernel_amplitude ** 2 / self.output_size) * torch.cos(
            self.output(-x)
        )  # batch_size x last_layer_size
        # Logits: batch_size x last_layer_size @ last_layer_size x output_size -> batch_size x output_size
        post_mean = self.Beta(Phi)

        return post_mean, Phi

    def forward(
        self, x: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Forward pass for SNGPModule layer.

        Parameters
        ----------
        x: torch.FloatTensor
            Last hidden state of underlying model.

        Returns
        -------
        torch.FloatTensor
            Logits for the current batch.
        """
        logits, Phi = self._get_features(x)

        if self.training:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)  # batch_size x seq_len x output_size
                max_probs = torch.max(probs, dim=-1)[0]  # batch_size x seq_len
                max_probs = max_probs * (1 - max_probs)

                Phi = Phi.unsqueeze(-1)  # Make it batch_size x last_layer_size x 1

                # Vectorized version of eq. 9
                # b: batch size
                # s: sequence length
                # o, p: output size
                # z: singleton dimension
                # Integrate multiplication with max_probs into einsum to avoid producing another 4D tensor
                PhiPhi = torch.einsum(
                    "bsoz,bszp->op",
                    Phi * max_probs.unsqueeze(-1).unsqueeze(-1),
                    torch.transpose(Phi, 2, 3)
                )
                self.sigma_hat_inv *= self.scaling_coefficient
                self.sigma_hat_inv += (1 - self.scaling_coefficient) * PhiPhi

        return logits

    def predict(self, x: torch.FloatTensor, num_predictions: Optional[int] = None):
        """
        Get predictions for the current batch.

        Parameters
        ----------
        x: torch.FloatTensor
            Last hidden state of underlying model.
        num_predictions: Optional[int]
            Number of predictions sampled from the GP in the SNGPModule layer to come to the final prediction. If None, number
            specified during initialization is used.

        Returns
        -------
        torch.FloatTensor
            Class probabilities for current batch.
        """

        logits = self.get_logits(x, num_predictions)

        out = F.softmax(logits, dim=-1).mean(dim=1)

        return out

    def get_logits(self, x: torch.FloatTensor, num_predictions: Optional[int] = None):
        """
        Get the logits for an input. Results in a tensor of size batch_size x num_predictions x seq_len x output_size
        depending on the model type.

        Parameters
        ----------
        x: torch.LongTensor
            Input to the model, containing all token IDs of the current batch.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGPModule layer to come to the final prediction. If None, number
            specified during initialization is used.

        Returns
        -------
        torch.FloatTensor
            Logits for the current batch.
        """

        if num_predictions is None:
            num_predictions = self.num_predictions

        # In case the Sigma matrix wasn't inverted yet, make sure that it is here.
        # Also set inversed_sigma to False again in case this is called during validation so that the matrix will be
        # updated over the training time again.
        if not self.inversed_sigma:
            self.invert_sigma_hat()
            self.inversed_sigma = False

        if num_predictions is None:
            num_predictions = self.num_predictions

        _, Phi = self._get_features(x)

        # Sample num_predictions Beta matrices
        beta_samples = torch.stack(
            [
                dist.rsample((num_predictions, ))
                for dist in self.beta_dists
            ],
            dim=-1
        )

        # Because we just stacked num_predictions samples for every column of the beta matrix, we now have to switch
        # the last two dimensions to obtain a num_predictions Beta matrices
        beta_samples = rearrange(beta_samples, "n c r -> n r c")

        # Now compute different logits using betas sampled from the Laplace posterior. This operation is a batch
        # instance-wise time step-wise multiplication of features with one Beta matrix per num_predictions.
        # b: batch
        # s: sequence length
        # k: number of classes
        # n: number of predictions
        logits = torch.einsum("bsk,nkk->bnsk", Phi, beta_samples)

        return logits

    def invert_sigma_hat(self) -> None:
        """
        Invert the sigma hat matrix.
        """
        try:
            self.sigma_hat = torch.inverse(self.sigma_hat_inv)

        except RuntimeError:
            warnings.warn(f"Matrix could not be inverted, compute pseudo-inverse instead.")
            self.sigma_hat = torch.linalg.pinv(self.sigma_hat_inv)

        self.inversed_sigma = True

        # Create multivariate normal distributions to sample columns from Beta matrix from. Updated every time
        # sigma_hat is updated since it sigma_hat is the covariance matrix used.
        self.beta_dists = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                self.Beta.weight.data[:, k], covariance_matrix=self.sigma_hat,
            )
            for k in range(self.output_size)
        ]