import torch
from clrnet.models.losses.lineiou_loss import line_iou


def distance_cost(predictions, targets, img_w):
    """
    repeat predictions and targets to generate all combinations
    use the abs distance as the new distance cost
    """
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    predictions = torch.repeat_interleave(
        predictions, num_targets, dim=0
    )[...,
      6:]  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b] ((np + nt) * 78)

    targets = torch.cat(
        num_priors *
        [targets])[...,
                   6:]  # applying this 2 times on [c, d] gives [c, d, c, d]

    invalid_masks = (targets < 0) | (targets >= img_w)
    lengths = (~invalid_masks).sum(dim=1)
    distances = torch.abs((targets - predictions))
    distances[invalid_masks] = 0.
    distances = distances.sum(dim=1) / (lengths.float() + 1e-9)
    distances = distances.view(num_priors, num_targets)

    return distances


def focal_cost(cls_pred, gt_labels, alpha=0.25, gamma=2, eps=1e-12):
    """
    Args:
        cls_pred (Tensor): Predicted classification logits, shape
            [num_query, num_class].
        gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

    Returns:
        torch.Tensor: cls_cost value
    """
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost


def dynamic_k_assign(cost, pair_wise_ious):
    """
    Assign grouth truths with priors dynamically.

    Args:
        cost: the assign cost.
        pair_wise_ious: iou of grouth truth and priors.

    Returns:
        prior_idx: the index of assigned prior.
        gt_idx: the corresponding ground truth index.
    """
    matching_matrix = torch.zeros_like(cost)
    ious_matrix = pair_wise_ious
    ious_matrix[ious_matrix < 0] = 0.
    n_candidate_k = 4
    topk_ious, _ = torch.topk(ious_matrix, n_candidate_k, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
    num_gt = cost.shape[1]
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[:, gt_idx],
                                k=dynamic_ks[gt_idx].item(),
                                largest=False)
        matching_matrix[pos_idx, gt_idx] = 1.0
    del topk_ious, dynamic_ks, pos_idx

    matched_gt = matching_matrix.sum(1)
    if (matched_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
        matching_matrix[matched_gt > 1, 0] *= 0.0
        matching_matrix[matched_gt > 1, cost_argmin] = 1.0

    prior_idx = matching_matrix.sum(1).nonzero()
    gt_idx = matching_matrix[prior_idx].argmax(-1)
    return prior_idx.flatten(), gt_idx.flatten()


def assign(
    predictions,
    targets,
    img_w,
    img_h,
    distance_cost_weight=3.,
    cls_cost_weight=1.,
):
    '''
    computes dynamicly matching based on the cost, including cls cost and lane similarity cost
    Args:
        predictions (Tensor): predictions predicted by each stage, shape: (num_priors, 78)
        targets (Tensor): lane targets, shape: (num_targets, 78)
    return:
        matched_row_inds (Tensor): matched predictions, shape: (num_targets)
        matched_col_inds (Tensor): matched targets, shape: (num_targets)
    '''
    predictions = predictions.detach().clone()
    predictions[:, 3] *= (img_w - 1)
    predictions[:, 6:] *= (img_w - 1)
    targets = targets.detach().clone()

    # distances cost
    distances_score = distance_cost(predictions, targets, img_w)
    distances_score = 1 - (distances_score / torch.max(distances_score)
                           ) + 1e-2  # normalize the distance

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]  # num_targets, 2
    target_start_xys[..., 0] *= (img_h - 1)
    prediction_start_xys = predictions[:, 2:4]
    prediction_start_xys[..., 0] *= (img_h - 1)

    start_xys_score = torch.cdist(prediction_start_xys, target_start_xys,
                                  p=2).reshape(num_priors, num_targets)
    start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

    target_thetas = targets[:, 4].unsqueeze(-1)
    theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
                              target_thetas,
                              p=1).reshape(num_priors, num_targets) * 180
    theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

    cost = -(distances_score * start_xys_score * theta_score
             )**2 * distance_cost_weight + cls_score * cls_cost_weight

    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, aligned=False)
    matched_row_inds, matched_col_inds = dynamic_k_assign(cost, iou)

    return matched_row_inds, matched_col_inds
