#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict, make_non_pad_mask
from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        idim,
        selfattention_layer_type="selfattn",
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length=11,
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        padding_idx=-1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self.attention_dim = attention_dim
        self._register_load_state_dict_pre_hook(_pre_hook)
        self.input_layer = input_layer
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate)
        elif input_layer == "conv2d-scaled-pos-enc":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(idim, attention_dim, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(idim, attention_dim, dropout_rate)
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "hs-conv":
            self.embed = torch.nn.Sequential(
                pos_enc_class(idim, positional_dropout_rate),
            )

            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 2), stride=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(3, 2), stride=(2, 2)),
                nn.ReLU()
            )

            self.conv_group_2 = nn.Sequential(
                nn.Conv2d(4, 4, kernel_size=1, stride=1),
                nn.ReLU()
            )

            self.conv_group_3 = nn.Sequential(
                nn.Conv2d(6, 6, kernel_size=1, stride=1),
                nn.ReLU()
            )

            self.conv_group_4 = nn.Sequential(
                nn.Conv2d(7, 7, kernel_size=1, stride=1),
                nn.ReLU()
            )
        elif input_layer == "channel-transformer":
            self.embed = torch.nn.Sequential()
            # 1. two conv layer to extract multi-channel feature maps
            self.conv = torch.nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 8), stride=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(3, 7), stride=(2, 2)),
                nn.ReLU()
            )

            # 2. use transformer to joint feature maps
            # just a self-attention layer
            self.cd_model = self.attention_dim // 16
            self.ctlayers = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    self.cd_model,
                    MultiHeadedAttention(
                        attention_heads, self.cd_model, attention_dropout_rate
                    ),
                    positionwise_layer(self.cd_model, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

            # 3. transformer layers with positional encoding
            self.pe = pos_enc_class(attention_dim, positional_dropout_rate)

        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            logging.info("self.embed:{}".format(self.embed))
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before

        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads, attention_dim, attention_dropout_rate
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

        elif selfattention_layer_type == "lightconv":
            logging.info("encoder self-attention layer type = lightweight convolution")
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    LightweightConvolution(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

        elif selfattention_layer_type == "lightconv2d":
            logging.info(
                "encoder self-attention layer "
                "type = lightweight convolution 2-dimentional"
            )
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    LightweightConvolution2D(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "dynamicconv":
            logging.info("encoder self-attention layer type = dynamic convolution")
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    DynamicConvolution(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "dynamicconv2d":
            logging.info(
                "encoder self-attention layer type = dynamic convolution 2-dimentional"
            )
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    DynamicConvolution2D(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def get_positionwise_layer(
        self,
        positionwise_layer_type="linear",
        attention_dim=256,
        linear_units=2048,
        dropout_rate=0.1,
        positionwise_conv_kernel_size=1,
    ):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args

    def forward(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            # linear input layer
            xs = self.embed(xs)
        if self.input_layer == "hs-conv":
            batchsize = xs.size(0)
            inputs = xs.unsqueeze(1)
            logging.info("inputs:{}".format(inputs.shape))
            inputs_length = []

            logging.info("masks:{}".format(masks))

            if masks is not None:
                for mask in masks.tolist():
                    inputs_length.append(mask[0].count(True))

                for i in range(batchsize):

                    inputs_s = inputs[i].unsqueeze(0)[:, :, 0:inputs_length[i], :]

                    core_out = self.conv1(inputs_s)
                    inputs_length[i] = core_out.size(2)

                inputs_length = torch.as_tensor(inputs_length)

            else:

                core_out = self.conv1(inputs)

                inputs_length = core_out.size(2)
                inputs_length = torch.as_tensor(inputs_length)
                logging.info("inputs_length:{}".format(inputs_length))

            # block 1
            # the inputs shape of Conv2d is 4-dim of (bsz * c * l * w)
            inputs = self.conv1(inputs)

            # group 1 stay                   4 channel
            # group 2 conv + split           2 channel
            # group 3 concate + conv + split 3 channel
            # group 4 concate + conv         7 channel
            group = [item for item in enumerate(torch.chunk(inputs, 4, 1))]

            group[0] = group[0][1]

            group[1] = self.conv_group_2(group[1][1])
            group[1] = [item[1] for item in enumerate(torch.chunk(group[1], 2, 1))]

            group[2] = torch.cat((group[1][1], group[2][1]), 1)
            group[2] = self.conv_group_3(group[2])
            group[2] = [item[1] for item in enumerate(torch.chunk(group[2], 2, 1))]

            group[3] = torch.cat((group[2][1], group[3][1]), 1)
            group[3] = self.conv_group_4(group[3])

            core_out_12 = torch.cat((group[0], group[1][0]), 1)
            core_out_34 = torch.cat((group[2][0], group[3]), 1)
            core_out = torch.cat((core_out_12, core_out_34), 1)

            # ResNet
            core_out = core_out + inputs

            xs = core_out.view(core_out.size(0), core_out.size(2), -1)

            if masks is not None:
                masks = make_non_pad_mask(inputs_length.tolist()).unsqueeze(-2)
            else:
                masks = make_non_pad_mask([inputs_length]).unsqueeze(-2)

        elif self.input_layer == "channel-transformer":
            batchsize = xs.size(0)
            inputs = xs.unsqueeze(1)
            logging.info("inputs:{}".format(inputs.shape))
            inputs_length = []

            logging.info("masks:{}".format(masks))


            if masks is not None:
                for mask in masks.tolist():
                    inputs_length.append(mask[0].count(True))

                for i in range(batchsize):

                    inputs_s = inputs[i].unsqueeze(0)[:, :, 0:inputs_length[i], :]

                    core_out = self.conv(inputs_s)
                    inputs_length[i] = core_out.size(2)

                inputs_length = torch.as_tensor(inputs_length)

            else:

                core_out = self.conv(inputs)

                inputs_length = core_out.size(2)
                inputs_length = torch.as_tensor(inputs_length)
                logging.info("inputs_length:{}".format(inputs_length))

            # block 1
            # the inputs shape of Conv2d is 4-dim of (bsz * c * l * w)
            # the inputs shape of Conv1d is 3-dim of (bsz * c * l)
            # the inputs shape of transformer is 3-dim of (l * bsz * c)
            # conv output format: (bsz * c * t * d)
            inputs = self.conv(inputs)

            # we can get a batch of 16 channels feature maps in all time steps
            # merge 16 channels of one timestep to create one self-attention input (batch, 16, dim)
            inputs = inputs.permute(2, 0, 1, 3)
            logging.info("inputs:{}".format(inputs.shape))
            merge = torch.zeros(inputs.size(0), batchsize, self.attention_dim)

            for t in range(inputs.size(0)):
                merge[t] = self.ctlayers(inputs[t], None)[0].reshape(batchsize, self.attention_dim)

            xs = merge.permute(1, 0, 2)
            xs = xs + self.pe(xs)

            if inputs_length.dim() == 0:
                masks = make_non_pad_mask([inputs_length]).unsqueeze(-2)
            else:
                masks = make_non_pad_mask(inputs_length.tolist()).unsqueeze(-2)

        xs, masks = self.encoders(xs, masks)

        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
