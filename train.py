import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from augmentations import SequenceAugmenter
from torch import amp

from dataset import RAMDataset
from model import GestureModel
from config import *
import torch

def detect_peaks(probs):
    left = probs.roll(1, dims=1)
    right = probs.roll(-1, dims=1)
    peaks = (probs > left) & (probs > right) & (probs > 0.3)
    return peaks.long()


def gaussian_smooth(labels, sigma=1.5):
    labels = labels.float()
    B, T = labels.shape
    pad_mask = labels == -1
    labels = labels.clamp(min=0)
    x = torch.arange(T, device=labels.device).float()
    smoothed = torch.zeros_like(labels)

    for i in range(T):
        mask = labels[:, i] == 1
        if mask.any():
            smoothed[mask] += torch.exp(-((x - i) ** 2) / (2 * sigma ** 2))

    smoothed = torch.clamp(smoothed, 0, 1)
    smoothed[pad_mask] = 0
    return smoothed


def compute_weights(dataset, num_gestures):
    start_all, end_all = [], []
    gesture_all = []

    for _, gesture, start, end in dataset:
        start_all.append(start.view(-1))
        end_all.append(end.view(-1))
        gesture_all.append(gesture.view(-1))

    start_all = torch.cat(start_all)
    end_all = torch.cat(end_all)
    gesture_all = torch.cat(gesture_all).view(-1).long()
    gesture_all = gesture_all[gesture_all >= 0]
    assert gesture_all.numel() > 0, "No valid gesture labels found"

    # --- Start / End (binary) ---
    start_weight = min(float((start_all == 0).sum()) / max(1, float((start_all == 1).sum())), 10)
    end_weight = min(float((end_all == 0).sum()) / max(1, float((end_all == 1).sum())), 10)

    # --- Gesture (multi-class) ---
    counts = torch.bincount(gesture_all, minlength=num_gestures-1).float()
    counts += 1  # prevent zero counts
    weights = torch.log1p(counts.sum() / counts)
    weights /= weights.mean()       
    weights = torch.cat([weights, torch.tensor([1.0], device=weights.device)]) # 1.0 weight for NO_GESTURE class

    return start_weight, end_weight, weights


import matplotlib.pyplot as plt
def boundary_error_analysis(true_s, pred_s, true_e, pred_e, max_offset=10):
    # Convert to tensors
    true_s = torch.cat(true_s)
    pred_s = torch.cat(pred_s)
    true_e = torch.cat(true_e)
    pred_e = torch.cat(pred_e)

    def get_offsets(true, pred, tol=3):
        true_idx = (true==1).nonzero(as_tuple=True)[0]
        pred_idx = (pred==1).nonzero(as_tuple=True)[0]

        offsets = []

        for t in true_idx:
            if len(pred_idx) == 0:
                offsets.append(None)  # missed entirely
            else:
                diff = pred_idx - t
                abs_diff, min_idx = diff.abs().min(0)
                if abs_diff.item() <= tol:
                    offsets.append(diff[min_idx].item())  # matched
                else:
                    offsets.append(None)  # too far, count as missed
        return offsets

    start_offsets = get_offsets(true_s, pred_s, tol=3)
    end_offsets   = get_offsets(true_e, pred_e, tol=3)

    # remove None (missed predictions) for histogram
    start_offsets_plot = [o for o in start_offsets if o is not None]
    end_offsets_plot   = [o for o in end_offsets if o is not None]

    print(f"Start boundaries: {len(start_offsets_plot)}/{len(start_offsets)} matched, mean offset: {torch.tensor(start_offsets_plot).float().abs().mean():.2f} frames")
    print(f"End boundaries:   {len(end_offsets_plot)}/{len(end_offsets)} matched, mean offset: {torch.tensor(end_offsets_plot).float().abs().mean():.2f} frames")

    # histogram of offsets
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.hist(start_offsets_plot, bins=range(-max_offset,max_offset+1), color='green', alpha=0.7)
    plt.title("Start boundary offsets")
    plt.xlabel("Frames from true boundary")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    plt.hist(end_offsets_plot, bins=range(-max_offset,max_offset+1), color='red', alpha=0.7)
    plt.title("End boundary offsets")
    plt.xlabel("Frames from true boundary")
    plt.ylabel("Count")
    plt.show()


import time
import subprocess

def get_gpu_utilization():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
            encoding='utf-8'
        )
        return int(output.strip().split("\n")[0])
    except Exception as e:
        return None
    
def run_epoch(model, loader, augmenter=None, train=True):
    model.train(train)
    total_loss, correct_g, total_g = 0, 0, 0
    true_s, pred_s, true_e, pred_e = [], [], [], []

    total_tokens = 0
    total_examples = 0
    start_time = time.time()
    gloss, sloss, eloss = 0,0,0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seqs, g_lbls, s_lbls, e_lbls in loader:
            batch_size = g_lbls.size(0)
            seq_len = g_lbls.size(1)
            total_tokens += batch_size * seq_len
            total_examples += batch_size

            seqs = {k: v.to(DEVICE) for k, v in seqs.items()}
            orig_s = s_lbls.clone()
            orig_e = e_lbls.clone()
            if train:
                s_lbls = gaussian_smooth(s_lbls, sigma=5)
                e_lbls = gaussian_smooth(e_lbls, sigma=5)
            g_lbls, s_lbls, e_lbls = g_lbls.to(DEVICE), s_lbls.to(DEVICE), e_lbls.to(DEVICE)
            orig_s, orig_e = orig_s.to(DEVICE), orig_e.to(DEVICE)

            if train and augmenter: seqs = augmenter(seqs)

            with amp.autocast(device_type="cuda", dtype=torch.float16):
                g, s, e = model(seqs)

                gloss = 2*gesture_loss_fn(g.view(-1, g.size(-1)), g_lbls.view(-1))
                sloss = boundary_loss_fn_start(s.view(-1), s_lbls.view(-1).float())
                eloss = boundary_loss_fn_end(e.view(-1), e_lbls.view(-1).float())
            
            loss = gloss+sloss+eloss

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() == prev_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            preds_g = g.argmax(-1)
            mask_g = g_lbls != -1
            correct_g += (preds_g[mask_g] == g_lbls[mask_g]).sum().item()
            total_g += mask_g.sum().item()

            preds_s, preds_e = detect_peaks(torch.sigmoid(s)), detect_peaks(torch.sigmoid(e))
            true_s.append(orig_s.view(-1))
            pred_s.append(preds_s.view(-1))
            true_e.append(orig_e.view(-1))
            pred_e.append(preds_e.view(-1))

        if not train:
            # --- PRINT STATS PER EPOCH ---
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            examples_per_sec = total_examples / elapsed
            mem_alloc = torch.cuda.memory_allocated(DEVICE) / 1e9
            mem_max   = torch.cuda.max_memory_allocated(DEVICE) / 1e9
            util      = get_gpu_utilization()

            print("losses: ", gloss.item(), ", ", sloss.item(), ", ", eloss.item())
            print(f"[Epoch stats] tokens/sec: {tokens_per_sec:.1f}, "
                    f"examples/sec: {examples_per_sec:.1f}, "
                    f"GPU mem: {mem_alloc:.2f}/{mem_max:.2f} GB, "
                    f"GPU util: {util}%")

    return (total_loss/len(loader), 
            correct_g/total_g if total_g>0 else 0,
            (true_s, pred_s, true_e, pred_e))


if __name__ == "__main__":

    # ---------------- Dataset ----------------
    train_loader = DataLoader(
        RAMDataset("train_dataset.pt"),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    test_loader = DataLoader(
        RAMDataset("test_dataset.pt"),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    in_dim = train_loader.dataset[0][0]["raw"].shape[1]

    # ---------------- Model ----------------
    model = GestureModel(modelConfig, in_dim).to(DEVICE)

    global gesture_loss_fn, boundary_loss_fn_start, boundary_loss_fn_end
    
    start_weight, end_weight, gestures_weight = compute_weights(train_loader.dataset, num_gestures=modelConfig.num_gestures)
    print("gesutres weights: ", gestures_weight)

    gesture_loss_fn = nn.CrossEntropyLoss(weight=gestures_weight.clone().detach().to(DEVICE), ignore_index=-1, label_smoothing=0.2)

    boundary_loss_fn_start = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(start_weight).to(DEVICE))
    boundary_loss_fn_end   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(end_weight).to(DEVICE))

    global optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.04)
    scaler = amp.GradScaler()
    torch.backends.cuda.matmul.allow_tf32 = True  
    torch.backends.cudnn.allow_tf32 = True  

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=num_epochs * len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=50,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model.print_num_params()
    if any(p.numel() > 0 for p in model.parameters()):
        ddevice = next(model.parameters()).device
        print("Model device:", ddevice)
    else:
        print("Model has no parameters!")

    # Augmentations
    augmenter = SequenceAugmenter(jitter_std=0.1, scale_range=(0.9,1.1), max_shift=25, prob=0.5)

    # ---------------- Training ----------------
    for epoch in range(num_epochs):
        train_loss, train_gesture_acc, train_data = run_epoch(model, train_loader, augmenter, train=True)
        test_loss, test_gesture_acc, test_data = run_epoch(model, test_loader, augmenter, train=False)

        train_acc_str = f"gesture_acc {train_gesture_acc:.3f}"
        test_acc_str  = f"gesture_acc {test_gesture_acc:.3f}"

        print(f"Epoch {epoch+1:3d} | train loss {train_loss:.4f}, {train_acc_str} | "
            f"test loss {test_loss:.4f}, {test_acc_str}\n")

        if epoch == num_epochs-1:
            true_s, pred_s, true_e, pred_e = test_data
            boundary_error_analysis(true_s, pred_s, true_e, pred_e)