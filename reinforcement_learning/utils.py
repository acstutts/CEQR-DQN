
import random
import torch.nn.functional as F
import numpy as np
import torch

def set_global_seed(seed, env):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def quantile_huber_loss(x,y, device, kappa=1):

    batch_size = x.shape[0] 
    num_quant = x.shape[1]

    #Get x and y to repeat here
    x = x.unsqueeze(2).repeat(1,1,num_quant)
    y = y.unsqueeze(2).repeat(1,1,num_quant).transpose(1,2)

    tau_hat = torch.linspace(0.0, 1.0 - 1. / num_quant, num_quant) + 0.5 / num_quant
    tau_hat = tau_hat.to(device)
    tau_hat = tau_hat.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1,num_quant)
    
    diff = y-x

    if kappa == 0:
        huber_loss = diff.abs()
    else:
        huber_loss = 0.5 * diff.abs().clamp(min=0.0, max=kappa).pow(2)
        huber_loss += kappa * (diff.abs() - diff.abs().clamp(min=0.0, max=kappa))

    quantile_loss = (tau_hat - (diff < 0).float()).abs() * huber_loss

    return quantile_loss.mean(2).mean(0).sum()

def intervalscore(label, upper, lower):

    s = (upper - lower)
    
    label_lt_lower = (label < lower).float()
    label_gt_upper = (label > upper).float()

    s += label_lt_lower * (2 / 0.95) * (lower - label) + label_gt_upper * (2 / 0.95) * (label - upper)
    s += label_lt_lower * (2 / 0.05) * (lower - label) + label_gt_upper * (2 / 0.05) * (label - upper)

    s_mean = s.mean(dim=0).mean()

    return s_mean

def combinedcalibrationloss(label, upper, lower, lamda):

    marginalcov = ((lower <= label) & (label <= upper)).float().mean(dim=0)
    
    marginalcov_lt_09 = marginalcov < 0.9
    marginalcov_gt_09 = marginalcov > 0.9

    calibration_upper = torch.where(marginalcov_lt_09, torch.clamp(label - upper, min=0), torch.clamp(upper - label, min=0))
    calibration_lower = torch.where(marginalcov_lt_09, torch.clamp(label - lower, min=0), torch.clamp(lower - label, min=0))

    calibration_upper = calibration_upper.mean(dim=0).mean()
    calibration_lower = calibration_lower.mean(dim=0).mean()
    calibration = calibration_upper + calibration_lower

    sharpness = upper - lower
    sharpness = torch.where(marginalcov_gt_09.unsqueeze(0), sharpness, torch.zeros_like(sharpness))
    sharpness = sharpness.mean(dim=0).mean()

    combinedcalloss = (1 - lamda) * calibration + lamda * sharpness

    return combinedcalloss

def gamma_cal_loss(gamma,y,lamda,device):

    gamma_lowq = gamma[:,0]
    gamma_upq = gamma[:,1]
    if gamma_lowq.dim() == 1:
        gamma_lowq = gamma_lowq.unsqueeze(-1)
        gamma_upq = gamma_upq.unsqueeze(-1)
    interval_loss = intervalscore(y,gamma_upq,gamma_lowq) 
    calibrationloss = combinedcalibrationloss(y,gamma_upq,gamma_lowq,lamda) # for evidentical quantile calibration loss
    evi_cal_loss = interval_loss + calibrationloss

    return evi_cal_loss

def loss_fn(x,y,lamda,kappa,device):
    
    quant_loss = quantile_huber_loss(x,y,device,kappa=kappa) # for distributional RL quantile loss
    cal_loss = batch_cali_loss(y, x, device, lamda)
    
    return quant_loss + cal_loss

def batch_cali_loss(y, x, device, lamda):
    """
    Batched calibration loss function for batch of quantiles using precomputed predictions.

    Args:
        y (torch.Tensor): The target values with shape [batch_size, num_q].
        x (torch.Tensor): The predicted quantiles with shape [batch_size, num_q].
        device (torch.device): The device on which to perform calculations.
        lamda (float): balance between sharpness and coverage
    """

    batch_size, num_q = y.shape
    # Create quantile levels tensor
    q_list = torch.linspace(0.0, 1.0, steps=num_q+2,device=device)[1:-1].unsqueeze(0)
    q_list = q_list.expand(batch_size, -1)  # shape (batch_size, num_q)

    # Calculate coverage and indicator matrices
    idx_under = (y <= x)
    idx_over = ~idx_under
    coverage = torch.mean(idx_under.float(), dim=1)  # shape (batch_size,)

    # Calculate mean differences where predictions are under or over the targets
    mean_diff_under = torch.mean((y - x) * idx_under.float(), dim=1)
    mean_diff_over = torch.mean((y - x) * idx_over.float(), dim=1)

    # Determine whether each prediction falls under or over the corresponding quantile
    cov_under = coverage.unsqueeze(1) < q_list
    cov_over = ~cov_under
    loss_list = (cov_under * mean_diff_over.unsqueeze(1)) + (cov_over * mean_diff_under.unsqueeze(1))

    # handle scaling
    with torch.no_grad():
        cov_diff = torch.abs(coverage.unsqueeze(1) - q_list)
    loss_list = cov_diff * loss_list
    loss = torch.mean(loss_list)

    # handle sharpness penalty
    # Assume x already contains the predicted quantiles
    # Hence, for opposite quantiles, we simply use 1 - x
    opp_pred_y = 1.0 - x

    # Check if each quantile is below or above the median (0.5)
    with torch.no_grad():
        below_med = q_list <= 0.5
        above_med = ~below_med

    # Calculate sharpness penalty based on whether the quantile is below or above the median
    sharp_penalty = below_med * (opp_pred_y - x) + above_med * (x - opp_pred_y)
    with torch.no_grad():
        width_positive = sharp_penalty > 0.0

    # Penalize sharpness only if centered interval observation proportions are too high
    # Calculate expected and observed interval proportions
    with torch.no_grad():
        exp_interval_props = torch.abs((2 * q_list) - 1)
        interval_lower = torch.min(x, opp_pred_y)
        interval_upper = torch.max(x, opp_pred_y)

        # Check if y falls within the predicted interval
        within_interval = (interval_lower <= y) & (y <= interval_upper)
        obs_interval_props = torch.mean(within_interval.float(), dim=1)
        obs_interval_props = obs_interval_props.unsqueeze(1).expand(-1, num_q)

        obs_over_exp = obs_interval_props > exp_interval_props

        # Reshape tensors to match the required dimensions for the penalty calculation
        #obs_over_exp = obs_over_exp.unsqueeze(1).expand(-1, num_q)
        width_positive = width_positive.expand(-1, num_q)

    # Apply sharpness penalty based on whether observed interval proportions are too high
    sharp_penalty = obs_over_exp * width_positive * sharp_penalty

    loss = ((1 - lamda) * loss) + (
        (lamda) * torch.mean(sharp_penalty)
    )

    return loss