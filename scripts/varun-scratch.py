# Part B: the network, training, the overfitting demo, and two fixes.
#
# What this script does, in order:
#   1. Load the data and preprocessor that Person A already prepared.
#   2. Define build_model(...) -- the one reusable function Person C will
#      call during tuning, per the group plan.
#   3. Train a plain network with no safeguards against overfitting, and
#      plot its training-vs-validation curves (the overfitting demo).
#   4. Apply fix #1 -- early stopping -- and show it improves validation
#      performance compared to the no-fix run.
#   5. Apply fix #2 -- dropout -- and show it also improves validation
#      performance, via a different mechanism than early stopping.
#
# Headline finding: Person A's EDA found that none of the allowed columns
# strongly predict the real target, so no version of this model is expected
# to reach high accuracy (everything should hover near 50%, i.e. chance).
# That is expected and correct here, not a bug -- the point of this section
# is to clearly demonstrate overfitting and how to fix it, not to maximize
# accuracy (doing the latter would require using the leakage columns A
# correctly excluded).

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 1969  # matches Person A's data-split seed, for consistency across the vignette
N_EPOCHS = 20
BATCH_SIZE = 64

## ---------------------------------------------------------------------
## STEP 1: Load Person A's already-prepared data and preprocessor
## ---------------------------------------------------------------------

pre = joblib.load('data/processed/preprocessor.joblib')
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')

X_train = pre.transform(train_df.drop(columns=['y'])).astype(np.float32)
y_train = train_df['y'].to_numpy().astype(np.float32)
X_val = pre.transform(val_df.drop(columns=['y'])).astype(np.float32)
y_val = val_df['y'].to_numpy().astype(np.float32)

print(f'X_train shape: {X_train.shape}  (expected: (21000, 29))')
print(f'X_val shape:   {X_val.shape}  (expected: (4500, 29))')

# The network only understands PyTorch tensors, not plain numpy arrays.
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)  # shape (n, 1) to match the output layer
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val).unsqueeze(1)

INPUT_DIM = X_train.shape[1]

## ---------------------------------------------------------------------
## STEP 2: The network architecture, wrapped in build_model(...)
## ---------------------------------------------------------------------
# Person C calls this function with different hidden_units/dropout_rate
# values during tuning -- this is the handoff point between our sections.

def build_model(input_dim, hidden_units=16, dropout_rate=0.0):
    layers = [nn.Linear(input_dim, hidden_units), nn.ReLU()]
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(hidden_units, 1))
    # No sigmoid here -- we output raw "logits" and use BCEWithLogitsLoss,
    # which is more numerically stable. Sigmoid is applied separately
    # (see accuracy_from_logits) only when we need actual probabilities.
    return nn.Sequential(*layers)


def accuracy_from_logits(logits, y_true):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == y_true).float().mean().item()


## ---------------------------------------------------------------------
## STEP 3: One reusable training function
## ---------------------------------------------------------------------
# Every version of the model below (no fix, early stopping, dropout) is
# trained with this same function -- only its settings differ. This keeps
# the comparison fair (same data, same batch size, same starting point)
# and avoids repeating the training loop three separate times.

def train_model(model, early_stopping=False, patience=3, verbose=True):
    """Train `model` for up to N_EPOCHS, tracking train/val loss and
    accuracy every epoch. If early_stopping=True, training halts once
    validation loss hasn't improved for `patience` epochs in a row, and
    the model is rolled back to its best checkpoint.
    Returns: (history dict, number of epochs actually run)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    n_train = X_train_t.shape[0]

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()  # dropout (if any) is ACTIVE in this mode
        permutation = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()            # clear old gradients
            logits = model(batch_x)          # forward pass: make a guess
            loss = loss_fn(logits, batch_y)  # how wrong was the guess?
            loss.backward()                  # figure out which direction to adjust
            optimizer.step()                 # actually adjust the weights

        model.eval()  # dropout (if any) is OFF in this mode, as in real use
        with torch.no_grad():
            train_logits = model(X_train_t)
            val_logits = model(X_val_t)
            train_loss = loss_fn(train_logits, y_train_t).item()
            val_loss = loss_fn(val_logits, y_val_t).item()
            train_acc = accuracy_from_logits(train_logits, y_train_t)
            val_acc = accuracy_from_logits(val_logits, y_val_t)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch + 1
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

        if verbose:
            flag = '  <-- best so far' if (early_stopping and epochs_without_improvement == 0) else ''
            print(f"Epoch {epoch+1:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}{flag}")

        if early_stopping and epochs_without_improvement >= patience:
            print(f"\nStopping early at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs in a row). "
                  f"Restoring best epoch: {best_epoch}")
            model.load_state_dict(best_state)
            return history, best_epoch

    return history, N_EPOCHS


def final_val_metrics(model):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(X_val_t)
        return loss_fn(logits, y_val_t).item(), accuracy_from_logits(logits, y_val_t)


## ---------------------------------------------------------------------
## STEP 4: Run 1 -- no fix (the overfitting demo)
## ---------------------------------------------------------------------

print('=== Run 1: no fix (plain network, all 20 epochs) ===')
torch.manual_seed(SEED)
model_nofix = build_model(INPUT_DIM, hidden_units=16, dropout_rate=0.0)
history_nofix, _ = train_model(model_nofix)
nofix_val_loss, nofix_val_acc = final_val_metrics(model_nofix)

epochs_range = range(1, N_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(epochs_range, history_nofix['train_loss'], label='Training', marker='o')
axes[0].plot(epochs_range, history_nofix['val_loss'], label='Validation', marker='o')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (lower = better)')
axes[0].set_title('Loss over training'); axes[0].legend()
axes[1].plot(epochs_range, history_nofix['train_acc'], label='Training', marker='o')
axes[1].plot(epochs_range, history_nofix['val_acc'], label='Validation', marker='o')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (higher = better)')
axes[1].set_title('Accuracy over training'); axes[1].legend()
plt.suptitle('Simple network: training vs. validation performance (no fix)')
plt.tight_layout()
plt.savefig('scripts/overfitting_demo_v1.png', dpi=150)
print('Chart saved to scripts/overfitting_demo_v1.png\n')

## ---------------------------------------------------------------------
## STEP 5: Run 2 -- fix #1: early stopping
## ---------------------------------------------------------------------

print('=== Run 2: early stopping ===')
torch.manual_seed(SEED)  # same starting point as Run 1, for a fair comparison
model_es = build_model(INPUT_DIM, hidden_units=16, dropout_rate=0.0)
history_es, es_stop_epoch = train_model(model_es, early_stopping=True, patience=3)
es_val_loss, es_val_acc = final_val_metrics(model_es)

print(f"\nNo fix          (all {N_EPOCHS} epochs):     loss={nofix_val_loss:.4f}  acc={nofix_val_acc:.4f}")
print(f"Early stopping  (kept epoch {es_stop_epoch}):      loss={es_val_loss:.4f}  acc={es_val_acc:.4f}")

es_epochs_range = range(1, len(history_es['val_loss']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(epochs_range, history_nofix['val_loss'], label='No fix', color='orange', linestyle='--')
axes[0].plot(es_epochs_range, history_es['val_loss'], label='Early stopping', color='green', marker='o')
axes[0].axvline(es_stop_epoch, color='gray', linestyle=':', label=f'Kept epoch {es_stop_epoch}')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Validation loss (lower = better)')
axes[0].set_title('Validation loss: before vs. after fix'); axes[0].legend()
axes[1].plot(epochs_range, history_nofix['val_acc'], label='No fix', color='orange', linestyle='--')
axes[1].plot(es_epochs_range, history_es['val_acc'], label='Early stopping', color='green', marker='o')
axes[1].axvline(es_stop_epoch, color='gray', linestyle=':', label=f'Kept epoch {es_stop_epoch}')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation accuracy (higher = better)')
axes[1].set_title('Validation accuracy: before vs. after fix'); axes[1].legend()
plt.suptitle('Effect of early stopping on validation performance')
plt.tight_layout()
plt.savefig('scripts/early_stopping_fix.png', dpi=150)
print('Chart saved to scripts/early_stopping_fix.png\n')

## ---------------------------------------------------------------------
## STEP 6: Run 3 -- fix #2: dropout
## ---------------------------------------------------------------------
# Same architecture, same data, all 20 epochs (no early stopping this
# time) -- the only change from Run 1 is dropout_rate=0.3 instead of 0.0,
# so this isolates dropout's effect on its own.

print('=== Run 3: dropout (0.3), all 20 epochs ===')
torch.manual_seed(SEED)
model_dropout = build_model(INPUT_DIM, hidden_units=16, dropout_rate=0.3)
history_dropout, _ = train_model(model_dropout)
dropout_val_loss, dropout_val_acc = final_val_metrics(model_dropout)

print(f"\nNo fix          (20 epochs, no dropout):  loss={nofix_val_loss:.4f}  acc={nofix_val_acc:.4f}")
print(f"Early stopping  (kept epoch {es_stop_epoch}):       loss={es_val_loss:.4f}  acc={es_val_acc:.4f}")
print(f"Dropout         (20 epochs, dropout=0.3): loss={dropout_val_loss:.4f}  acc={dropout_val_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(epochs_range, history_nofix['val_loss'], label='No fix', color='orange', linestyle='--')
axes[0].plot(es_epochs_range, history_es['val_loss'], label='Early stopping', color='green', marker='o')
axes[0].plot(epochs_range, history_dropout['val_loss'], label='Dropout', color='purple', marker='s')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Validation loss (lower = better)')
axes[0].set_title('Validation loss: comparing fixes'); axes[0].legend()
axes[1].plot(epochs_range, history_nofix['val_acc'], label='No fix', color='orange', linestyle='--')
axes[1].plot(es_epochs_range, history_es['val_acc'], label='Early stopping', color='green', marker='o')
axes[1].plot(epochs_range, history_dropout['val_acc'], label='Dropout', color='purple', marker='s')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation accuracy (higher = better)')
axes[1].set_title('Validation accuracy: comparing fixes'); axes[1].legend()
plt.suptitle('Comparing two fixes for overfitting: early stopping vs. dropout')
plt.tight_layout()
plt.savefig('scripts/dropout_comparison.png', dpi=150)
print('Chart saved to scripts/dropout_comparison.png')

## ---------------------------------------------------------------------
## Summary
## ---------------------------------------------------------------------
# - The plain network (no fix) overfits: training loss/accuracy improve
#   while validation loss/accuracy get worse across 20 epochs.
# - Early stopping fixes this by halting training at its best point
#   (epoch 3 here) instead of continuing past it.
# - Dropout fixes the same problem a different way: by randomly disabling
#   neurons during training, the network never gets the chance to overfit
#   as badly in the first place, even across the full 20 epochs.
# - All validation accuracies land close to 50%, consistent with Person
#   A's finding that the allowed features carry little real signal. That
#   is expected here, and is a separate issue from overfitting itself.
