import torch


def initialize_weights(node_out, node_in, window):
    return torch.nn.Parameter(
        torch.randn(node_out, node_in, window, window)
        * (2 / (window * window * node_in)) ** 0.5
    )


def initialize_bias(node_in):
    return torch.nn.Parameter(torch.zeros(node_in))


class Unet(torch.nn.Module):
    """A from-scratch U-Net.

    Note ``forward`` takes a SINGLE image of shape ``(C, H, W)`` -- not a
    batch. Callers that need batching stack the per-image results themselves
    (see ``forward_batch`` in ``train.py``).
    """

    def __init__(self, base=16):
        super().__init__()
        self.down = torch.nn.ParameterList()
        self.up = torch.nn.ParameterList()

        channels = [1, base, base * 2, base * 4, base * 8, base * 16]
        for i in range(5):
            in_c = channels[i]
            out_c = channels[i + 1]
            self.down.append(initialize_weights(out_c, in_c, 3))
            self.down.append(initialize_bias(out_c))
            self.down.append(initialize_weights(out_c, out_c, 3))
            self.down.append(initialize_bias(out_c))

        for i in range(4, 0, -1):
            in_c = channels[i + 1]
            out_c = channels[i]
            if i != 0:
                self.up.append(initialize_weights(out_c, in_c, 2))
                self.up.append(initialize_bias(out_c))
            self.up.append(initialize_weights(out_c, out_c * 2, 3))
            self.up.append(initialize_bias(out_c))
            self.up.append(initialize_weights(out_c, out_c, 3))
            self.up.append(initialize_bias(out_c))

        # The 1x1 head emits 2 channels against a 1-element bias, which
        # broadcasts. `forward` therefore returns (2, H, W) and every caller
        # slices [:1] for the binary logit. This is not a typo -- the released
        # checkpoint stores exactly these shapes, so changing them to 1 channel
        # breaks `load_state_dict`.
        self.up.append(initialize_weights(2, base, 1))
        self.up.append(initialize_bias(1))

    @staticmethod
    def conv_3(block, W, b):
        """Valid 3x3 convolution + ReLU.

        block : (n_c_in, n_w, n_h)
        W     : (n_c_out, n_c_in, 3, 3)
        b     : (n_c_out,)
        out   : (n_c_out, n_w - 2, n_h - 2)
        """
        window = block.unfold(1, 3, 1).unfold(2, 3, 1)
        return torch.relu(torch.einsum('cijmn,kcmn->kij', window, W) + b[:, None, None])

    @staticmethod
    def conv_1(block, W, b):
        """1x1 convolution, no activation (returns logits).

        block : (n_c_in, n_w, n_h)
        W     : (n_c_out, n_c_in, 1, 1)
        b     : (n_c_out,) or (1,), broadcast
        out   : (n_c_out, n_w, n_h)
        """
        window = block.unfold(1, 1, 1).unfold(2, 1, 1)
        return torch.einsum('cijmn,kcmn->kij', window, W) + b[:, None, None]

    @staticmethod
    def max_pool2(block):
        """2x2 max pool, stride 2.

        block : (n_c, n_w, n_h)
        out   : (n_c, n_w // 2, n_h // 2)
        """
        window = block.unfold(1, 2, 2).unfold(2, 2, 2)
        return window.amax(dim=(3, 4))

    @staticmethod
    def trans_up_conv2(block, W, b):
        """2x2 transposed convolution, stride 2, + ReLU.

        block : (n_c_in, n_w, n_h)
        W     : (n_c_out, n_c_in, 2, 2)
        b     : (n_c_out,)
        out   : (n_c_out, n_w * 2, n_h * 2)
        """
        # scatter each input pixel into a 2x2 tile -> (n_c_out, n_w, n_h, 2, 2)
        initial_output = torch.einsum('ijk,limn->ljkmn', block, W)
        # interleave the tiles into the spatial grid: (n_c_out, n_w, 2, n_h, 2)
        # before reshape, because reshape flattens the last axes first.
        initial_output = initial_output.permute(0, 1, 3, 2, 4).reshape(
            W.shape[0], block.shape[1] * 2, block.shape[2] * 2
        )
        return torch.relu(initial_output + b[:, None, None])

    @staticmethod
    def copy_crop(block, last_block):
        """Centre-crop the encoder feature map and concatenate it on channels.

        block      : (n_c, n_w, n_h)          -- upsampled decoder features
        last_block : (n_c, n_w', n_h')        -- larger encoder features
        out        : (n_c * 2, n_w, n_h)
        """
        pre_crop = last_block
        shape_different = (pre_crop.shape[-1] - block.shape[-1]) // 2
        crop = pre_crop[
            :,
            shape_different:pre_crop.shape[-1] - shape_different,
            shape_different:pre_crop.shape[-1] - shape_different,
        ]
        return torch.concatenate((block, crop), axis=0)

    def forward(self, input):
        """input : (C, H, W) -- a single image, NOT a batch.

        Returns (2, H - 184, W - 184) logits; slice [:1] for the binary channel.
        """
        d = self.down
        u = self.up
        A = input
        last_block = []

        for i in range(5):
            A = self.conv_3(A, d[i * 4], d[i * 4 + 1])
            A = self.conv_3(A, d[i * 4 + 2], d[i * 4 + 3])
            if i != 4:
                last_block.append(A)
                A = self.max_pool2(A)

        for i in range(4):
            A = self.trans_up_conv2(A, u[i * 6], u[i * 6 + 1])
            A = self.copy_crop(A, last_block.pop())
            A = self.conv_3(A, u[i * 6 + 2], u[i * 6 + 3])
            A = self.conv_3(A, u[i * 6 + 4], u[i * 6 + 5])

        A = self.conv_1(A, u[24], u[25])
        return A


if __name__ == "__main__":
    IN_SIZE, OUT_SIZE = 684, 500
    model = Unet(base=16)
    with torch.no_grad():
        got = model(torch.zeros(1, IN_SIZE, IN_SIZE)).shape
    assert got[-1] == OUT_SIZE, f"geometry broken: input {IN_SIZE} gave {got[-1]}"
    n = sum(p.numel() for p in model.parameters())
    print(f"input {IN_SIZE} -> output {tuple(got)}  ({n:,} parameters)")
