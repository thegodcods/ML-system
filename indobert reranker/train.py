# hello, disini untuk training loop
import json
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# External dependencies

from global_ml_sys_config import LR, MODEL_NAME, BATCH_SIZE, DEVICE, EPOCHS, CHECKPOINT_DIR, WEIGHT_DECAY
from quickhire_model import IndoBERTRanker
from dataset import build_dataloader

# Utilities

train_seed = 42

torch.manual_seed(train_seed)
torch.cuda.manual_seed(train_seed)
torch.cuda.manual_seed_all(train_seed)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)


def move_to_device(batch, device):
    """
    Supports:
        - tuple/list of tensors
        - dict of tensors
        - single tensor
    """

    if isinstance(batch, (list, tuple)):
        return [x.to(device) for x in batch]

    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}

    return batch.to(device)

# Train / Validation Steps


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    running_loss = 0.0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        batch = move_to_device(batch, device)

        # ----------------------------------------------------
        # Assumes:
        #   batch = (inputs, targets)
        # Adjust if your structurizer returns something else
        # ----------------------------------------------------

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        if "labels" in batch:
            labels = batch["labels"].float()
            loss = criterion(outputs, labels)
        else:
            assert "labels" in batch

        if loss is not None:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / len(loader)

    return epoch_loss


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    for batch in progress_bar:
        batch = move_to_device(batch, device)

        outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
        )

        if "labels" in batch:
            labels = batch["labels"].float()
            loss = criterion(outputs, labels)
        else:
            continue

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)

    return epoch_loss


# Main Training Function


def main():

    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    # loading
    with open("train_dataset.jsonl", "r", encoding="utf-8") as f:
        complete_data = [json.loads(line) for line in f]

    # splitter
    split_idx = int(len(complete_data) * 0.8)

    train_samples = complete_data[:split_idx]
    val_samples = complete_data[split_idx:]

    # loader
    train_loader = build_dataloader(train_samples,
                                    model_name=MODEL_NAME,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True
                                    )
    val_loader = build_dataloader(val_samples,
                                  model_name=MODEL_NAME,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False
                                  )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = IndoBERTRanker().to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Checkpoint setup
    # --------------------------------------------------------
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    # Training Loop

    for epoch in range(EPOCHS):
        start_time = time.time()

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        elapsed = time.time() - start_time

        print(
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"time={elapsed:.2f}s"
        )

        # Save latest checkpoint

        latest_ckpt = checkpoint_dir / "latest.pt"

        save_checkpoint(
            path=latest_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
        )

        # Save best checkpoint

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_ckpt = checkpoint_dir / "best.pt"

            save_checkpoint(
                path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )

            print(f"Saved new best checkpoint -> {best_ckpt}")

    print("\nTraining complete.")

# Entry


if __name__ == "__main__":
    main()
