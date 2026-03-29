import torch
import torch.nn as nn


class SpeakerEncoder(nn.Module):
    """Causal/non-causal speaker encoder: Conv1d front-end -> LSTM -> Attention pooling -> FC.

    Context is configurable via left_context_frames and right_context_frames.
    Kernel sizes and padding are auto-computed to match the requested context.
    """

    def __init__(self, n_mels=80, embedding_dim=256,
                 left_context_frames=10, right_context_frames=0,
                 conv_channels=512, conv_strides=None):
        super().__init__()

        if conv_strides is None:
            conv_strides = [1, 2, 2]

        self.conv_strides = conv_strides
        n_layers = len(conv_strides)

        # Cumulative stride product BEFORE each layer
        # e.g. strides [1, 2, 2] -> stride_products [1, 1, 2]
        stride_products = []
        sp = 1
        for s in conv_strides:
            stride_products.append(sp)
            sp *= s

        # Distribute context frames across layers
        left_pads = self._distribute_context(left_context_frames, stride_products)
        right_pads = self._distribute_context(right_context_frames, stride_products)

        # Ensure minimum kernel size of 3 per layer
        for i in range(n_layers):
            while left_pads[i] + right_pads[i] + 1 < 3:
                left_pads[i] += 1

        kernels = [lp + rp + 1 for lp, rp in zip(left_pads, right_pads)]

        # Verify actual context matches requested
        actual_left = sum(lp * sp for lp, sp in zip(left_pads, stride_products))
        actual_right = sum(rp * sp for rp, sp in zip(right_pads, stride_products))

        self.left_context_frames = actual_left
        self.right_context_frames = actual_right
        self.causal_pad = list(zip(left_pads, right_pads))

        # Print architecture
        print(f"  Conv architecture ({n_layers} layers):")
        for i, (k, s, lp, rp) in enumerate(zip(kernels, conv_strides, left_pads, right_pads)):
            ch_in = n_mels if i == 0 else conv_channels
            print(f"    Layer {i}: Conv1d({ch_in}->{conv_channels}, kernel={k}, stride={s}, "
                  f"pad_left={lp}, pad_right={rp})")
        print(f"  Receptive field: {actual_left + 1 + actual_right} frames "
              f"(left={actual_left}, right={actual_right})")
        total_stride = 1
        for s in conv_strides:
            total_stride *= s
        print(f"  Temporal downsampling: {total_stride}x")

        # Build conv layers
        self.conv = nn.ModuleList()
        for i in range(n_layers):
            ch_in = n_mels if i == 0 else conv_channels
            self.conv.append(nn.Sequential(
                nn.Conv1d(ch_in, conv_channels, kernel_size=kernels[i],
                          stride=conv_strides[i], padding=0),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
            ))

        # LSTM (causal by nature)
        self.lstm = nn.LSTM(
            input_size=conv_channels, hidden_size=conv_channels,
            num_layers=2, batch_first=True, dropout=0.1,
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(conv_channels, conv_channels // 2),
            nn.Tanh(),
            nn.Linear(conv_channels // 2, 1),
        )

        # Final projection
        self.fc = nn.Linear(conv_channels, embedding_dim, bias=False)

    @staticmethod
    def _distribute_context(total_frames, stride_products):
        """Distribute context frames across conv layers, accounting for stride multipliers.

        Each layer's padding contributes (pad * stride_product) input frames.
        Layers with smaller stride_product are cheaper, so get more padding.
        """
        n = len(stride_products)
        pads = [0] * n

        if total_frames == 0:
            return pads

        remaining = total_frames
        for i in range(n):
            sp = stride_products[i]
            if i < n - 1:
                # Give a fair share to this layer
                share = remaining // (n - i)
                pads[i] = share // sp
                remaining -= pads[i] * sp
            else:
                # Last layer gets the rest
                pads[i] = remaining // sp
                remaining -= pads[i] * sp

        # Leftover (rounding with large stride) goes to first layer (cheapest)
        if remaining > 0:
            pads[0] += remaining

        return pads

    def forward(self, x):
        # x: (B, n_mels, T)

        # Conv with configured padding
        for conv, (lp, rp) in zip(self.conv, self.causal_pad):
            x = nn.functional.pad(x, (lp, rp))
            x = conv(x)
        # x: (B, channels, T')

        # LSTM expects (B, T, C)
        x = x.transpose(1, 2)  # (B, T', channels)
        x, _ = self.lstm(x)    # (B, T', channels)

        # Attention pooling
        w = self.attention(x)          # (B, T', 1)
        w = torch.softmax(w, dim=1)    # (B, T', 1)
        x = (x * w).sum(dim=1)         # (B, channels)

        # Project to embedding
        x = self.fc(x)    # (B, embedding_dim)
        return x
