# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import typing as tp

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import remove_weight_norm

from hilcodec.modules.conv import NormConv1d
from hilcodec.modules.seanet import SEANetDecoder, SEANetEncoder
from hilcodec.vector_quantize import ResidualVQ

Array = tp.Union[np.ndarray, list]


class HILCodec(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """

    def __init__(
        self,
        sample_rate: int,
        channels_audio: int = 1,
        channels_enc: int = 32,
        channels_dec: int = 32,
        n_fft_base: int = 64,
        n_residual_enc: int = 1,
        n_residual_dec: int = 1,
        res_scale_enc: tp.Optional[float] = None,
        res_scale_dec: tp.Optional[float] = None,
        strides: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_kwargs: dict = {"alpha": 1.0},
        norm: str = "weight_norm",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 5,
        last_kernel_size: int = 5,
        residual_kernel_size: int = 5,
        dilation_base: int = 1,
        skip: str = "identity",
        final_activation: tp.Optional[str] = "Tanh",
        vq: str = "ResidualVQ",  # "" / "ResidualVQ" / "ResidualGainShapeVQ" / "ResidualGainResidualShapeVQ"
        vq_kwargs: tp.Dict[str, tp.Any] = {},
        act_all: bool = False,
        expansion: int = 1,
        groups: int = -1,
        encoder_l2norm: bool = True,
        bias: bool = True,
        spec: str = "stft",  # dct or stft
        spec_compression: str = "",  # "" or "log" or float(0~1)
        spec_learnable: bool = False,
        pad_mode: str = "constant",
        causal: bool = True,
        zero_init: bool = True,
        inout_norm: bool = True,
    ):
        assert spec in ["stft", ""]
        assert skip in ["1x1", "scale", "channelwise_scale", "identity"]
        if expansion != 1 and groups != -1:
            raise RuntimeError(
                f"Both expansion({expansion}) and groups({groups}) are set. "
                f"Either set expansion=1 or set groups=-1"
            )
        if encoder_l2norm and vq != "ResidualVQ":
            print(f"Warning: encoder_l2norm is used with vq {vq}")

        super().__init__()
        self.norm = norm
        channels_vq = vq_kwargs["dim"]
        self.encoder = SEANetEncoder(
            channels_audio,
            channels_vq,
            channels_enc,
            n_fft_base,
            n_residual_enc,
            strides,
            activation,
            activation_kwargs,
            norm,
            norm_kwargs,
            kernel_size,
            last_kernel_size,
            residual_kernel_size,
            dilation_base,
            skip,
            causal=causal,
            act_all=act_all,
            expansion=expansion,
            groups=groups,
            l2norm=encoder_l2norm,
            bias=bias,
            spec=spec,
            spec_compression=spec_compression,
            res_scale=res_scale_enc,
            pad_mode=pad_mode,
            spec_learnable=spec_learnable,
            zero_init=zero_init,
            inout_norm=inout_norm,
        )
        self.decoder = SEANetDecoder(
            channels_audio,
            channels_vq,
            channels_dec,
            n_residual_dec,
            strides,
            activation,
            activation_kwargs,
            norm,
            norm_kwargs,
            kernel_size,
            last_kernel_size,
            residual_kernel_size,
            dilation_base,
            skip,
            causal=causal,
            final_activation=final_activation,
            act_all=act_all,
            expansion=expansion,
            groups=groups,
            bias=bias,
            res_scale=res_scale_dec,
            pad_mode=pad_mode,
            zero_init=zero_init,
            inout_norm=inout_norm,
        )
        if vq == "ResidualVQ":
            self.quantizer = ResidualVQ(channel_last=False, **vq_kwargs)
        elif vq == "":
            self.quantizer = None
        else:
            raise ValueError(f"Unknown vq: {vq}")

        self.sample_rate = sample_rate
        self.channels = channels_audio

    def forward(
        self, x: Tensor, n: tp.Optional[int] = None
    ) -> tp.Tuple[Tensor, Array, Tensor]:
        x = self.encoder(x)
        if self.quantizer is not None:
            x, num_replaces, loss_vq = self.quantizer(x, n)
        else:
            num_replaces, loss_vq = [], torch.zeros(
                1, dtype=torch.float32, device=x.device
            )
        x = self.decoder(x)
        return x.float(), num_replaces, loss_vq

    def remove_weight_reparameterizations(self):
        if self.norm == "weight_norm":
            for module in self.modules():
                if isinstance(module, NormConv1d):
                    module.conv = remove_weight_norm(module.conv)

    @classmethod
    def from_pretrained(cls, model_name):
        import os
        import gdown
        import yaml
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

        model_info = {
            "hilcodec_speech": {
                "repo_id": "hilcodec/hilcodec-speech",
                "files": {
                    "config.yaml": "1ntnV6fQ8U-VYc_DYDhTengQJHZAEYEU2",
                    "00150.pth": "16-G1LaPsEclIvqUmaqQo6VTid65y5_tF",
                },
            },
            "hilcodec_music": {
                "repo_id": "hilcodec/hilcodec-music",
                "files": {
                    "config.yaml": "17Vhy3M32Azl9M0CjrBPIQKsmkVHILf5g",
                    "00150.pth": "1fQpuIab8HYWulWaC-GJ7VK21862m-sri",
                },
            },
        }

        if model_name not in model_info:
            raise NotImplementedError(f"Unknown model name: {model_name}")

        repo_id = model_info[model_name]["repo_id"]
        files = model_info[model_name]["files"]

        # Resolve Hugging Face cache directory
        cache_dir = os.path.join(HUGGINGFACE_HUB_CACHE, repo_id.replace("/", "--"))
        os.makedirs(cache_dir, exist_ok=True)

        # Download files if not already cached
        for filename, file_id in files.items():
            path = os.path.join(cache_dir, filename)
            if not os.path.exists(path):
                #print(f"Downloading {filename} to cache...")
                gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
            #else:
            #    print(f"Using cached file: {path}")

        # Load config
        config_path = os.path.join(cache_dir, "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Instantiate model
        model = cls(
            sample_rate=config["data"]["sampling_rate"],
            channels_audio=config["data"]["channels"],
            **config["model_kwargs"]
        )

        checkpoint_path = os.path.join(cache_dir, "00150.pth")
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
        model.load_state_dict(state_dict)

        model.remove_weight_reparameterizations()
        return model.eval()


if __name__ == "__main__":
    codec = HILCodec.from_pretrained("hilcodec_speech").eval()
    x = torch.randn(2, 1, 24000)
    x = codec.encoder(x)
    x, _, _, indices = codec.quantizer(x, n=8)
    y = codec.quantizer.decode(indices)
    x = codec.decoder(x)
    print(codec)
