import torch
import torch.nn as nn
import numpy as np
import math

def zero_pad(input, length_after_padding, dim=0, device="cpu"):
    missing = length_after_padding - input.shape[dim]
    if dim == 0:
        zeros = torch.zeros((missing,) + input.shape[dim:]).to(device)
    if dim == 1:
        zeros = torch.zeros((input.shape[0], missing,) + input.shape[dim+1:]).to(device)

    return torch.cat((zeros, input), dim=dim)

def relative_position_embedding(seq_length, out_dim, repeat_pos_encoding=1):
    """Creates a [seq_length x out_dim] matrix for rel. pos encoding.
    Denoted as Phi in [2] and [3]. Phi is the standard sinusoid encoding
    matrix.
    Args:
        seq_length (int): The max. sequence length (time axis).
        out_dim (int): The number of nodes to go into the first Tranformer
            layer with. Should be a power of 2.
    Returns:
        torch.Tensor: The encoding matrix Phi.
    """
    rest = seq_length % repeat_pos_encoding
    nr_of_timesteps = math.floor(seq_length / repeat_pos_encoding)
    nr_of_steps_with_repition = nr_of_timesteps * repeat_pos_encoding

    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

    inverse_freq = 1 / (10000**(torch.arange(0, out_dim, 2.0) / out_dim))
    pos_offsets = torch.arange(seq_length - 1, -1, -1)
    inputs = pos_offsets[:, None] * inverse_freq[None, :]
    return torch.cat((torch.cat((torch.sin(inputs), torch.cos(inputs)), dim=-1)[:rest],
        tile(torch.cat((torch.sin(inputs)[-nr_of_timesteps:], torch.cos(inputs)[-nr_of_timesteps:]), dim=-1), 0, repeat_pos_encoding)), dim=0)

class FullyConnected(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size,
                 out_size,
                 initializer=None,
                 activation_fn=None,
                 use_bias=True,
                 bias_init=0.0):
        super(FullyConnected, self).__init__()
        layers = []
        # Actual Conv2D layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer:
            initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class GRUGate(nn.Module):
    """Implements a gated recurrent unit for use in AttentionNet"""

    def __init__(self, dim, init_bias=0., device="cpu", **kwargs):
        """
        input_shape (torch.Tensor): dimension of the input
        init_bias (int): Bias added to every input to stabilize training
        """
        super().__init__(**kwargs)
        self._init_bias = init_bias

        # Xavier initialization of torch tensors
        self._w_r = torch.zeros(dim, dim).to(device)
        self._w_z = torch.zeros(dim, dim).to(device)
        self._w_h = torch.zeros(dim, dim).to(device)

        self._u_r = torch.zeros(dim, dim).to(device)
        self._u_z = torch.zeros(dim, dim).to(device)
        self._u_h = torch.zeros(dim, dim).to(device)

        nn.init.xavier_uniform_(self._w_r)
        nn.init.xavier_uniform_(self._w_z)
        nn.init.xavier_uniform_(self._w_h)

        nn.init.xavier_uniform_(self._u_r)
        nn.init.xavier_uniform_(self._u_z)
        nn.init.xavier_uniform_(self._u_h)

        self._bias_z = torch.zeros(dim, ).fill_(self._init_bias).to(device)

    def forward(self, inputs, **kwargs):
        # Pass in internal state first.
        h, X = inputs

        r = torch.tensordot(X, self._w_r, dims=1) + \
            torch.tensordot(h, self._u_r, dims=1)
        r = torch.sigmoid(r)

        z = torch.tensordot(X, self._w_z, dims=1) + \
            torch.tensordot(h, self._u_z, dims=1) - self._bias_z
        z = torch.sigmoid(z)

        h_next = torch.tensordot(X, self._w_h, dims=1) + \
            torch.tensordot((h * r), self._u_h, dims=1)
        h_next = torch.tanh(h_next)

        return (1 - z) * h + z * h_next


class MultiHeadAttention(nn.Module):
    """A multi-head attention layer described in [1]."""

    def __init__(self, in_dim, out_dim, num_heads, head_dim, input_layernorm=False, output_activation=None, device="auto",
                 **kwargs):
        """
        in_dim (int): Dimension of input
        out_dim (int): Dimension of output
        num_heads (int): Number of attention heads
        input_layernorm (bool): Whether to prepend a LayerNorm before
            everything else. Should be True for building a GTrXL.
        output_activation (Optional[tf.nn.activation]): Optional tf.nn
            activation function. Should be relu for GTrXL.
        head_dim (int): Output dimension of each attention head
        """
        super().__init__(**kwargs)

        # No bias or non-linearity.
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._qkv_layer = FullyConnected(
            in_size=in_dim, out_size=3 * num_heads * head_dim, use_bias=False, activation_fn=output_activation)

        self._linear_layer = FullyConnected(
            in_size=num_heads * head_dim, out_size=out_dim, use_bias=False)

        self._input_layernorm = None

        if input_layernorm:
            self._input_layernorm = torch.nn.LayerNorm(in_dim)

        self.device = device

    def forward(self, inputs):
        L = list(inputs.size())[1]  # length of segment
        H = self._num_heads  # number of attention heads
        D = self._head_dim  # attention head dimension

        if self._input_layernorm is not None:
            inputs = self._input_layernorm(inputs)

        qkv = self._qkv_layer(inputs)

        queries, keys, values = torch.chunk(input=qkv, chunks=3, dim=-1)
        queries = queries[:, -L:]  # only query based on the segment

        queries = torch.reshape(queries, [-1, L, H, D])
        keys = torch.reshape(keys, [-1, L, H, D])
        values = torch.reshape(values, [-1, L, H, D])

        score = torch.einsum("bihd,bjhd->bijh", queries, keys)
        score = score / D**0.5

        # causal mask of the same length as the sequence
        mask = sequence_mask(torch.arange(1, L + 1), dtype=score.dtype)
        mask = mask[None, :, :, None]
        mask = mask.float().to(self.device)

        masked_score = score * mask + 1e30 * (mask - 1.)
        wmat = nn.functional.softmax(masked_score, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", wmat, values)
        shape = list(out.size())[:2] + [H * D]
        #        temp = torch.cat(temp2, [H * D], dim=0)
        out = torch.reshape(out, shape)
        return self._linear_layer(out)


class SkipConnection(nn.Module):
    """
    Skip connection layer.

    Adds the original input to the output (regular residual layer) OR uses input
    as hidden state input to a given fan_in_layer
    """

    def __init__(self, layer, fan_in_layer=None, add_memory=False, **kwargs):
        """
        Initialize a SkipConnection nn Module object.

        Args:
            layer (nn.Module): Any layer processing inputs
            in_layer (Optional[nn.Module]): An optional layer taking two inputs: The original input and the output
            of 'layer'
        """
        super().__init__(**kwargs)
        self._layer = layer
        self._fan_in_layer = fan_in_layer

    def forward(self, inputs, **kwargs):

        outputs = self._layer(inputs, **kwargs)
        # Residual case, just add inputs to outputs
        if self._fan_in_layer is None:
            outputs = outputs + inputs
        # Fan-in e.g. RNN: Call fan-in with 'inputs' and 'outputs'.
        else:
            outputs = self._fan_in_layer((inputs, outputs))

        return outputs

def sequence_mask(lengths, maxlen=None, dtype=None, time_major=False):
    """Offers same behavior as tf.sequence_mask for torch.
    Thanks to Dimitris Papatheodorou
    (https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/
    39036).
    """
    if maxlen is None:
        maxlen = int(lengths.max())

    mask = ~(torch.ones(
        (len(lengths), maxlen)).to(lengths.device).cumsum(dim=1).t() > lengths)
    if not time_major:
        mask = mask.t()
    mask.type(dtype or torch.bool)

    return mask