import torch
from torch import Tensor

class DiffusionForwardProcess:
    """
    Implements the forward process of the diffusion model.
    """
    def __init__(self,
                 num_time_steps: int =1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02
                 ):
        """
        Initializes the DiffusionForwardProcess with the given parameters.

        :param num_time_steps: Number of time steps in the diffusion process.
        :param beta_start: Starting value of beta.
        :param beta_end: Ending value of beta.
        """

        self.betas = torch.linspace(beta_start, beta_end, num_time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def add_noise(self, original: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """
        Adds noise to the original image at the given time step t.
        :param original: Input Image
        :param noise: Random Noise Tensor sampled from Normal Dist
        :param t: timestep
        :return: Noisy image tensor
        """

        sqrt_alpha_bar_t = self.sqrt_alpha_bars.to(original.device)[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.to(original.device)[t]

        # Broadcast to multiply with the original image.
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]

        # Return
        return (sqrt_alpha_bar_t * original) \
            + \
            (sqrt_one_minus_alpha_bar_t * noise)


class DiffusionReverseProcess:
    """
    Implements the reverse process of the diffusion model.
    """

    def __init__(self,
                 num_time_steps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02
                 ):

        """
        Initializes the DiffusionReverseProcess with the given parameters.
        :param num_time_steps: Number of time steps in the diffusion process.
        :param beta_start: Starting value of beta.
        :param beta_end: Ending value of beta.
        """

        # Precomputing beta, alpha, and alpha_bar for all t's.
        self.b = torch.linspace(beta_start, beta_end, num_time_steps)  # b -> beta
        self.a = 1 - self.b  # a -> alpha
        self.a_bar = torch.cumprod(self.a, dim=0)  # a_bar = alpha_bar

    def sample_prev_timestep(self, xt: Tensor, noise_pred: Tensor, t) -> (Tensor, Tensor):
        """
        Samples the previous timestep image given the current timestep image and noise prediction.
        :param xt: Image tensor at timestep t of shape -> B x C x H x W
        :param noise_pred: Noise tensor predicted by the model at timestep t of shape -> B x C x H x W
        :param t: timestep
        :return: predicted x_t-1 and x0
        """

        # Original Image Prediction at timestep t
        x0 = xt - (torch.sqrt(1 - self.a_bar.to(xt.device)[t]) * noise_pred)
        x0 = x0 / torch.sqrt(self.a_bar.to(xt.device)[t])
        x0 = torch.clamp(x0, -1., 1.)

        # mean of x_(t-1)
        mean = (xt - ((1 - self.a.to(xt.device)[t]) * noise_pred) / (torch.sqrt(1 - self.a_bar.to(xt.device)[t])))
        mean = mean / (torch.sqrt(self.a.to(xt.device)[t]))

        # only return mean
        if t == 0:
            return mean, x0

        else:
            variance = (1 - self.a_bar.to(xt.device)[t - 1]) / (1 - self.a_bar.to(xt.device)[t])
            variance = variance * self.b.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

            return mean + sigma * z, x0