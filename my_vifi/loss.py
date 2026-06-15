import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


class AffinityLoss(nn.Module):
    """4-term loss for affinity matrix learning.

    L = (L_pc + L_cp + L_cons + L_aff) / 4
    """

    def __init__(self, Nm_phone=5, Nm_camera=15):
        super().__init__()
        self.Nm_phone = Nm_phone
        self.Nm_camera = Nm_camera

    def forward(self, pred, target, mask_phone, mask_camera):
        """
        Args:
            pred: (B, 1, Np+1, Nc+1) predicted affinity
            target: (B, Np+1, Nc+1) ground truth affinity
            mask_phone: (B, Np+1) valid phone mask
            mask_camera: (B, Nc+1) valid camera mask

        Returns:
            (loss_pre, loss_next, loss_similarity, loss_total, acc_pre, acc_next, acc, pred_indices)
        """
        B = pred.size(0)

        # Reshape masks for broadcasting
        mask_p = mask_phone.unsqueeze(1)    # (B, 1, Np+1)
        mask_c = mask_camera.unsqueeze(1)   # (B, 1, Nc+1)
        target = target.unsqueeze(1)        # (B, 1, Np+1, Nc+1)

        # Expand masks to 4D
        mask_p_4d = mask_p.unsqueeze(3).repeat(1, 1, 1, self.Nm_camera + 1)  # (B, 1, Np+1, Nc+1)
        mask_c_4d = mask_c.unsqueeze(2).repeat(1, 1, self.Nm_phone + 1, 1)   # (B, 1, Np+1, Nc+1)
        mask_valid = (mask_p_4d * mask_c_4d).float()  # valid positions

        # Row-wise softmax (phone → camera)
        mask_pc = mask_valid.clone()
        mask_pc[:, :, self.Nm_phone, :] = 0  # exclude extra row
        pred_pc = mask_pc * pred
        pred_pc = nn.Softmax(dim=3)(pred_pc)  # softmax along camera dim

        # Col-wise softmax (camera → phone)
        mask_cp = mask_valid.clone()
        mask_cp[:, :, :, self.Nm_camera] = 0  # exclude extra col
        pred_cp = mask_cp * pred
        pred_cp = nn.Softmax(dim=2)(pred_cp)  # softmax along phone dim

        # Union prediction: max of both directions
        mask_union = mask_pc * mask_cp
        pred_all = pred_pc.clone()
        pred_all[:, :, :self.Nm_phone, :self.Nm_camera] = torch.max(
            pred_pc, pred_cp
        )[:, :, :self.Nm_phone, :self.Nm_camera]

        # Ground truth splitting
        target = target.float()
        target_pc = mask_pc * target
        target_cp = mask_cp * target
        target_union = mask_union * target

        n_pc = target_pc.sum()
        n_cp = target_cp.sum()
        n_union = target_union.sum()
        n_total = target.sum()

        # L_pc: phone → camera cross-entropy
        if n_pc > 0:
            loss_pc = -(target_pc * torch.log(pred_pc + 1e-8)).sum() / n_pc
        else:
            loss_pc = torch.tensor(0.0, device=pred.device)

        # L_cp: camera → phone cross-entropy
        if n_cp > 0:
            loss_cp = -(target_cp * torch.log(pred_cp + 1e-8)).sum() / n_cp
        else:
            loss_cp = torch.tensor(0.0, device=pred.device)

        # L_aff: joint direction cross-entropy
        if n_union > 0:
            loss_aff = -(target_union * torch.log(pred_all + 1e-8)).sum() / n_union
        else:
            loss_aff = torch.tensor(0.0, device=pred.device)

        # L_cons: consistency between bidirectional predictions
        if n_union > 0:
            loss_cons = (target_union * torch.abs(pred_pc - pred_cp)).sum() / n_total
        else:
            loss_cons = torch.tensor(0.0, device=pred.device)

        loss_total = (loss_pc + loss_cp + loss_aff + loss_cons) / 4.0

        # Accuracy: phone → camera direction
        _, target_idx = target_pc.max(dim=3)
        target_idx = target_idx[:, :, :-1]  # exclude extra col
        _, pred_idx = pred_pc.max(dim=3)
        pred_idx = pred_idx[:, :, :-1]
        mask_p_rows = mask_p[:, :, :-1]
        n_valid_p = mask_p_rows.sum()
        if n_valid_p > 0:
            acc_pc = (pred_idx[mask_p_rows.bool()] == target_idx[mask_p_rows.bool()]).float().sum() / n_valid_p
        else:
            acc_pc = torch.tensor(1.0, device=pred.device)

        # Accuracy: camera → phone direction
        _, target_idx = target_cp.max(dim=2)
        target_idx = target_idx[:, :, :-1]  # exclude extra row
        _, pred_idx = pred_cp.max(dim=2)
        pred_idx = pred_idx[:, :, :-1]
        mask_c_cols = mask_c[:, :, :-1]
        n_valid_c = mask_c_cols.sum()
        if n_valid_c > 0:
            acc_cp = (pred_idx[mask_c_cols.bool()] == target_idx[mask_c_cols.bool()]).float().sum() / n_valid_c
        else:
            acc_cp = torch.tensor(1.0, device=pred.device)

        acc = (acc_pc + acc_cp) / 2.0

        return loss_pc, loss_cp, loss_cons, loss_total, acc_pc, acc_cp, acc, pred_idx


class AffinityEvaluator:
    """Evaluation with Hungarian bipartite matching for 1:1 association."""

    def __init__(self, Nm_phone=5, Nm_camera=15):
        self.Nm_phone = Nm_phone
        self.Nm_camera = Nm_camera

    def evaluate(self, pred, target, mask_phone, mask_camera):
        """
        Returns:
            dict with fc_association, cf_association, accuracies
        """
        B = pred.size(0)
        device = pred.device

        mask_p = mask_phone.unsqueeze(1)    # (B, 1, Np+1)
        mask_c = mask_camera.unsqueeze(1)   # (B, 1, Nc+1)
        target = target.unsqueeze(1)        # (B, 1, Np+1, Nc+1)

        mask_p_4d = mask_p.unsqueeze(3).repeat(1, 1, 1, self.Nm_camera + 1)
        mask_c_4d = mask_c.unsqueeze(2).repeat(1, 1, self.Nm_phone + 1, 1)
        mask_valid = (mask_p_4d * mask_c_4d).float()

        # Row-wise (phone → camera)
        mask_pc = mask_valid.clone()
        mask_pc[:, :, self.Nm_phone, :] = 0
        pred_pc = mask_pc * pred
        pred_pc = nn.Softmax(dim=3)(pred_pc)

        # Col-wise (camera → phone)
        mask_cp = mask_valid.clone()
        mask_cp[:, :, :, self.Nm_camera] = 0
        pred_cp = mask_cp * pred
        pred_cp = nn.Softmax(dim=2)(pred_cp)

        # Average both directions
        avg = (pred_pc[:, :, :self.Nm_phone, :self.Nm_camera] +
               pred_cp[:, :, :self.Nm_phone, :self.Nm_camera]) / 2.0
        pred_pc[:, :, :self.Nm_phone, :self.Nm_camera] = avg
        pred_cp[:, :, :self.Nm_phone, :self.Nm_camera] = avg

        # Ground truth indices for phone→camera
        target = target.float()
        target_pc = mask_pc * target
        _, target_idx_pc = target_pc.max(dim=3)
        target_idx_pc = target_idx_pc[:, :, :-1]

        _, target_idx_cp = (mask_cp * target).max(dim=2)
        target_idx_cp = target_idx_cp[:, :, :-1]

        # Simple max-based prediction
        _, pred_idx_pc = pred_pc.max(dim=3)
        pred_idx_pc = pred_idx_pc[:, :, :-1]
        _, pred_idx_cp = pred_cp.max(dim=2)
        pred_idx_cp = pred_idx_cp[:, :, :-1]

        # Hungarian bipartite matching
        mask_p_rows = mask_p[:, :, :-1].bool()
        mask_c_cols = mask_c[:, :, :-1].bool()

        correct_cf = 0
        total_cf = 0
        for b in range(B):
            valid_p = mask_p_rows[b].nonzero(as_tuple=False).squeeze(-1)
            valid_c = mask_c_cols[b].nonzero(as_tuple=False).squeeze(-1)

            if len(valid_p) == 0 or len(valid_c) == 0:
                continue

            aff = pred_cp[b, 0][valid_p][:, valid_c].cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(-aff)

            pred_assigned = torch.tensor([valid_c[c].item() for c in col_ind])
            target_assigned = target_idx_cp[b, 0][valid_p]

            correct_cf += (pred_assigned == target_assigned.cpu()).sum().item()
            total_cf += len(pred_assigned)

        acc_bp = correct_cf / total_cf if total_cf > 0 else 0.0

        return {
            "acc_pc": (pred_idx_pc[mask_p_rows] == target_idx_pc[mask_p_rows]).float().mean().item(),
            "acc_cp": (pred_idx_cp[mask_c_cols] == target_idx_cp[mask_c_cols]).float().mean().item(),
            "acc_bp": acc_bp,
            "pred_pc": pred_idx_pc,
            "pred_cp": pred_idx_cp,
        }
