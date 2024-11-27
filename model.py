import torch
from torch import Tensor, nn

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


def get_time_embedding(time_steps: torch.Tensor, t_emb_dim: int) -> torch.Tensor:
    """
    Transform a scalar time-step into a vector representation of size t_emb_dim.
    :param time_steps: 1D tensor of size -> (Batch, )
    :param t_emb_dim: Embedding Dimension -> for ex: 128 (scalar value)
    :return tensor of size -> (B, t_emb_dim)
    """

    assert t_emb_dim % 2 == 0, "time embedding must be divisible by 2."

    factor = 2 * torch.arange(start=0,
                              end=t_emb_dim // 2,
                              dtype=torch.float32,
                              device=time_steps.device
                              ) / t_emb_dim

    factor = 10000 ** factor

    t_emb = time_steps[:, None]  # B -> (B, 1)
    t_emb = t_emb / factor  # (B, 1) -> (B, t_emb_dim//2)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)  # (B , t_emb_dim)

    return t_emb


class NormActConv(nn.Module):
    """
    Perform GroupNorm, Activation, and Convolution operations.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int = 8,
                 kernel_size: int = 3,
                 norm: bool = True,
                 act: bool = True
                 ):
        super().__init__()

        # GroupNorm
        self.g_norm = nn.GroupNorm(
            num_groups,
            in_channels
        ) if norm is True else nn.Identity()

        # Activation
        self.act = nn.SiLU() if act is True else nn.Identity()

        # Convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class TimeEmbedding(nn.Module):
    """
    Maps the Time Embedding to the Required output Dimension.
    """

    def __init__(self,
                 n_out: int,  # Output Dimension
                 t_emb_dim: int = 128  # Time Embedding Dimension
                 ):
        super().__init__()

        # Time Embedding Block
        self.te_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_out)
        )

    def forward(self, x):
        return self.te_block(x)



class SelfAttentionBlock(nn.Module):
    """
    Perform GroupNorm and Multi-headed Self Attention operation.
    """

    def __init__(self,
                 num_channels: int,
                 num_groups: int = 8,
                 num_heads: int = 4,
                 norm: bool = True
                 ):
        super().__init__()

        # GroupNorm
        self.g_norm = nn.GroupNorm(
            num_groups,
            num_channels
        ) if norm is True else nn.Identity()

        # Self-Attention
        self.attn = nn.MultiheadAttention(
            num_channels,
            num_heads,
            batch_first=True
        )

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h * w)
        x = self.g_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x


class Downsample(nn.Module):
    """
    Perform Down sampling by the factor of k across Height and Width.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int = 2,  # Downsampling factor
                 use_conv: bool = True,  # If Downsampling using conv-block
                 use_mpool: bool = True  # If Downsampling using max-pool
                 ):
        super(Downsample, self).__init__()

        self.use_conv = use_conv
        self.use_mpool = use_mpool

        # Downsampling using Convolution
        self.cv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(
                in_channels,
                out_channels // 2 if use_mpool else out_channels,
                kernel_size=4,
                stride=k,
                padding=1
            )
        ) if use_conv else nn.Identity()

        # Downsampling using Maxpool
        self.mpool = nn.Sequential(
            nn.MaxPool2d(k, k),
            nn.Conv2d(
                in_channels,
                out_channels // 2 if use_conv else out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if use_mpool else nn.Identity()

    def forward(self, x):

        if not self.use_conv:
            return self.mpool(x)

        if not self.use_mpool:
            return self.cv(x)

        return torch.cat([self.cv(x), self.mpool(x)], dim=1)


class Upsample(nn.Module):
    """
    Perform Upsampling by the factor of k across Height and Width
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int = 2,  # Upsampling factor
                 use_conv: bool = True,  # Upsampling using conv-block
                 use_upsample: bool = True  # Upsampling using nn.upsample
                 ):
        super().__init__()

        self.use_conv = use_conv
        self.use_upsample = use_upsample

        # Upsampling using conv
        self.cv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels // 2 if use_upsample else out_channels,
                kernel_size=4,
                stride=k,
                padding=1
            ),
            nn.Conv2d(
                out_channels // 2 if use_upsample else out_channels,
                out_channels // 2 if use_upsample else out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if use_conv else nn.Identity()

        # Upsamling using nn.Upsample
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=k,
                mode='bilinear',
                align_corners=False
            ),
            nn.Conv2d(
                in_channels,
                out_channels // 2 if use_conv else out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if use_upsample else nn.Identity()

    def forward(self, x):

        if not self.use_conv:
            return self.up(x)

        if not self.use_upsample:
            return self.cv(x)

        return torch.cat([self.cv(x), self.up(x)], dim=1)


class DownC(nn.Module):
    """
    Perform Down-convolution on the input using following approach.
    1. Conv + TimeEmbedding
    2. Conv
    3. Skip-connection from input x.
    4. Self-Attention
    5. Skip-Connection from 3.
    6. Downsampling
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_emb_dim: int = 128,  # Time Embedding Dimension
                 num_layers: int = 2,
                 down_sample: bool = True  # True for Downsampling
                 ):

        super().__init__()

        self.num_layers = num_layers

        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i == 0 else out_channels,
                        out_channels
                        ) for i in range(num_layers)
        ])

        self.conv2 = nn.ModuleList([
            NormActConv(out_channels,
                        out_channels
                        ) for _ in range(num_layers)
        ])

        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
        ])

        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])

        self.down_block = Downsample(out_channels, out_channels) if down_sample else nn.Identity()

        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=1
            ) for i in range(num_layers)
        ])

    def forward(self, x, t_emb):
        out = x

        for i in range(self.num_layers):
            resnet_input = out

            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn

        # Downsampling
        out = self.down_block(out)

        return out


class MidC(nn.Module):
    """
    Refine the features obtained from the DownC block.
    It refines the features using following operations:

    1. Resnet Block with Time Embedding
    2. A Series of Self-Attention + Resnet Block with Time-Embedding
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_emb_dim: int = 128,
                 num_layers: int = 2
                 ):
        super(MidC, self).__init__()

        self.num_layers = num_layers

        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i == 0 else out_channels,
                        out_channels
                        ) for i in range(num_layers + 1)
        ])

        self.conv2 = nn.ModuleList([
            NormActConv(out_channels,
                        out_channels
                        ) for _ in range(num_layers + 1)
        ])

        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers + 1)
        ])

        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])

        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=1
            ) for i in range(num_layers + 1)
        ])

    def forward(self, x, t_emb):
        out = x

        # First-Resnet Block
        resnet_input = out
        out = self.conv1[0](out)
        out = out + self.te_block[0](t_emb)[:, :, None, None]
        out = self.conv2[0](out)
        out = out + self.res_block[0](resnet_input)

        # Sequence of Self-Attention + Resnet Blocks
        for i in range(self.num_layers):
            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn

            # Resnet Block
            resnet_input = out
            out = self.conv1[i + 1](out)
            out = out + self.te_block[i + 1](t_emb)[:, :, None, None]
            out = self.conv2[i + 1](out)
            out = out + self.res_block[i + 1](resnet_input)

        return out


class UpC(nn.Module):
    """
    Perform Up-convolution on the input using following approach.
    1. Upsampling
    2. Conv + TimeEmbedding
    3. Conv
    4. Skip-connection from 1.
    5. Self-Attention
    6. Skip-Connection from 3.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_emb_dim: int = 128,  # Time Embedding Dimension
                 num_layers: int = 2,
                 up_sample: bool = True  # True for Upsampling
                 ):
        super().__init__()

        self.num_layers = num_layers

        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i == 0 else out_channels,
                        out_channels
                        ) for i in range(num_layers)
        ])

        self.conv2 = nn.ModuleList([
            NormActConv(out_channels,
                        out_channels
                        ) for _ in range(num_layers)
        ])

        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
        ])

        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])

        self.up_block = Upsample(in_channels, in_channels // 2) if up_sample else nn.Identity()

        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=1
            ) for i in range(num_layers)
        ])

    def forward(self, x, down_out, t_emb):
        # Upsampling
        x = self.up_block(x)
        x = torch.cat([x, down_out], dim=1)

        out = x
        for i in range(self.num_layers):
            resnet_input = out

            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn

        return out