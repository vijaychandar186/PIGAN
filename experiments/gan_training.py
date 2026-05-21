import copy
import time
import torch
import torch.nn as nn

from models.pigan import PIGANGenerator, PIGANDiscriminator
from utils.metrics import evaluate_model
from utils.visualization import format_time


def pretrain_generator(generator, x_train, epochs, lr, batch_size):
    """Teacher-forcing MSE pre-training on parent (4 steps) -> child (5th step)."""
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = x_train[:4]
    target = x_train[4]
    N = x_train.shape[1]
    num_batches = max(1, N // batch_size)

    for epoch in range(epochs):
        generator.train()
        total_loss = 0.0
        perm = torch.randperm(N, device=x_train.device)
        for start in range(0, N - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            generated = generator(history[:, idx, :])
            loss = criterion(generated, target[idx, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Pretrain epoch {epoch + 1:3d}/{epochs} | loss {total_loss / num_batches:.6f}")


def train_pigan(generator, discriminator, x_train, y_train, x_test, y_test,
                epochs, g_lr, d_lr, batch_size):
    """Adversarial training. D: cross-entropy on real pairs. G: adversarial + MSE anchor."""
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    MSE_WEIGHT = 0.3
    D_STEPS = 2

    N = x_train.shape[1]
    history = x_train[:4]
    real_child = x_train[4]

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    start_time = time.time()

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        perm = torch.randperm(N, device=x_train.device)

        for s in range(0, N - batch_size + 1, batch_size):
            idx = perm[s:s + batch_size]
            x_batch = x_train[:, idx, :]
            y_batch = y_train[idx]
            h_batch = history[:, idx, :]

            for _ in range(D_STEPS):
                d_scores, _ = discriminator(x_batch)
                d_loss = criterion(d_scores, y_batch)
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            gen_child = generator(h_batch)
            fake_seq = torch.cat([h_batch, gen_child.unsqueeze(0)], dim=0)
            target_labels = torch.ones(idx.shape[0], dtype=torch.long, device=x_train.device)
            g_scores, _ = discriminator(fake_seq)
            g_loss = criterion(g_scores, target_labels) + MSE_WEIGHT * mse(gen_child, real_child[idx])
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        discriminator.eval()
        with torch.no_grad():
            test_scores, _ = discriminator(x_test)
            preds = test_scores.argmax(dim=1)
            pre, rec, f1, mcc, acc = evaluate_model(y_test.cpu().numpy(), preds.cpu().numpy())

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(discriminator.state_dict())
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0:
            elapsed = format_time(time.time() - start_time)
            print(f"Epoch {epoch + 1:3d} [{elapsed}] | "
                  f"acc {acc:.3f} pre {pre:.3f} rec {rec:.3f} f1 {f1:.3f} mcc {mcc:.3f}")

    discriminator.load_state_dict(best_state)
    return best_epoch


def run_pigan_experiment(x_train, y_train, x_test, y_test,
                          input_dim=100, hidden_size=256, num_layers=2, dropout_p=1e-4,
                          pretrain_epochs=100, gan_epochs=30,
                          g_lr=1e-3, d_lr=1e-3, batch_size=256,
                          seed=9, threshold=0.48):
    """PIGAN pipeline: pretrain G -> transfer G->D encoder weights -> GAN training -> threshold-calibrated prediction."""
    device = x_train.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = PIGANGenerator(input_dim, hidden_size, num_layers, dropout_p).to(device)
    discriminator = PIGANDiscriminator(input_dim, hidden_size, num_layers, dropout_p).to(device)

    print("Pre-training generator (teacher forcing)...")
    pretrain_generator(generator, x_train, pretrain_epochs, g_lr, batch_size)

    # Discriminator inherits phylogenetic representations from the pretrained generator.
    discriminator.transformer_encoder.load_state_dict(generator.transformer_encoder.state_dict())
    print("  Transferred G->D transformer encoder weights.")

    print("\nGAN adversarial training...")
    best_epoch = train_pigan(generator, discriminator, x_train, y_train, x_test, y_test,
                              gan_epochs, g_lr, d_lr, batch_size)

    discriminator.eval()
    with torch.no_grad():
        scores, _ = discriminator(x_test)
        probs = torch.softmax(scores, dim=1)
    preds = (probs[:, 1] > threshold).long()
    pre, rec, f1, mcc, acc = evaluate_model(y_test.cpu().numpy(), preds.cpu().numpy())

    print(f"\nPIGAN final (epoch {best_epoch}, threshold {threshold}):")
    print(f"  V_acc {acc:.3f}  V_pre {pre:.3f}  V_rec {rec:.3f}  V_f1 {f1:.3f}  V_mcc {mcc:.3f}")
    return {"val_acc": acc, "val_precision": pre, "val_recall": rec, "val_fscore": f1, "val_mcc": mcc}
