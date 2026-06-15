import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class MultimodalNetwork(nn.Module):
    """Bi-LSTM + 1x1 Conv affinity matrix network.

    Batched LSTM: processes all cameras/phones simultaneously instead of for-loop.
    """

    def __init__(self, config):
        super().__init__()
        self.Nm_phone = config.Nm_phone
        self.Nm_camera = config.Nm_camera
        self.lstm_feat_dim = config.lstm_hidden
        self.false_constant = config.false_constant
        self.camera_feat_dim = config.camera_feat_dim
        self.phone_feat_dim = config.phone_feat_dim

        self.lstm_v = nn.LSTM(
            config.camera_feat_dim, self.lstm_feat_dim,
            num_layers=2, batch_first=True, bidirectional=True
        )
        self.lstm_f = nn.LSTM(
            config.phone_feat_dim, self.lstm_feat_dim,
            num_layers=2, batch_first=True, bidirectional=True
        )

        self.compress = nn.Sequential(
            nn.Conv2d(self.lstm_feat_dim * 2, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def _encode_batched(self, lstm, x, x_len, Nm, B):
        """Batch-encode all Nm entities with a single LSTM call.

        Args:
            lstm: Bi-LSTM module
            x: (B, Nm, k, D) features
            x_len: (B, Nm) valid sequence lengths
            Nm: number of entities
            B: batch size

        Returns:
            (B, Nm, lstm_feat_dim) encoded features
        """
        # Flatten batch and entity dims: (B * Nm, k, D)
        x_flat = x.reshape(B * Nm, x.size(2), x.size(3))
        len_flat = x_len.reshape(-1).long().clamp(min=1)

        # Single batched LSTM call
        h0 = torch.zeros(4, B * Nm, self.lstm_feat_dim, device=x.device)
        c0 = torch.zeros(4, B * Nm, self.lstm_feat_dim, device=x.device)
        packed = pack_padded_sequence(x_flat, len_flat.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = lstm(packed, (h0, c0))
        # hn: (4, B*Nm, 32), take last layer backward direction → (B*Nm, 32)
        encoded = hn[-1].reshape(B, Nm, self.lstm_feat_dim)
        return encoded

    def forward(self, x_v, x_f, x_v_mask, x_f_mask):
        B = x_v.size(0)

        x_v_len = x_v[:, :, -1]   # (B, Nc)
        x_f_len = x_f[:, :, -1]   # (B, Np)
        x_v = x_v[:, :, :-1]      # (B, Nc, k*3)
        x_f = x_f[:, :, :-1]      # (B, Np, k*11)

        x_v = x_v.view(B, self.Nm_camera, -1, self.camera_feat_dim)
        x_f = x_f.view(B, self.Nm_phone, -1, self.phone_feat_dim)

        # Batched Bi-LSTM encoding
        output_v = self._encode_batched(self.lstm_v, x_v, x_v_len, self.Nm_camera, B)
        output_f = self._encode_batched(self.lstm_f, x_f, x_f_len, self.Nm_phone, B)

        # Apply masks (zero out dummy entities)
        output_v = output_v * x_v_mask[:, :-1].unsqueeze(2).float()
        output_f = output_f * x_f_mask[:, :-1].unsqueeze(2).float()

        # Feature ensemble: Cartesian product → (B, 64, Np, Nc)
        out_f = output_f.unsqueeze(2).repeat(1, 1, self.Nm_camera, 1).permute(0, 3, 1, 2)
        out_v = output_v.unsqueeze(1).repeat(1, self.Nm_phone, 1, 1).permute(0, 3, 1, 2)
        feat_cube = torch.cat([out_f, out_v], dim=1)

        affinity = self.compress(feat_cube)
        affinity = self._add_unmatched_dim(affinity)
        return affinity

    def _add_unmatched_dim(self, x):
        B = x.size(0)
        false_col = torch.ones(B, x.size(1), x.size(2), 1, device=x.device) * self.false_constant
        x = torch.cat([x, false_col], dim=3)
        false_row = torch.ones(B, x.size(1), 1, x.size(3), device=x.device) * self.false_constant
        x = torch.cat([x, false_row], dim=2)
        return x
