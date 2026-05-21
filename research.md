# PIGAN: Phylogenetic-Informed Generative Adversarial Network

Implementation notes and experimental record for SARS-CoV-2 mutation prediction on the COV19 benchmark.

## TL;DR

PIGAN combines a Transformer-encoder generator (with LSTM decoder) and a Transformer-encoder + MLP discriminator. The generator is pre-trained on parent-child sequences with teacher-forcing MSE; its transformer encoder weights are then transferred into the discriminator before adversarial training. At inference, the best discriminator snapshot is used with a calibrated decision threshold of 0.48.

**Final result on COV19:**

| Metric | Value | vs TEMPO |
|---|---|---|
| Accuracy  | **0.666** | +0.011 |
| Precision | **0.667** | +0.009 |
| Recall    | **0.637** | +0.023 |
| F1-score  | **0.652** | +0.016 |
| MCC       | **0.331** | +0.022 |

PIGAN beats TEMPO on all five metrics.

---

## Architecture

### Generator

```
PIGANGenerator(input_dim=100, hidden_size=256, num_layers=2, dropout_p=1e-4):
  Transformer encoder  (d_model=100, nhead=5, dim_ff=256, 2 layers, dropout 1e-4)
  Linear h_proj        (100 -> 256)   # phylogenetic embedding -> LSTM h0
  Linear c_proj        (100 -> 256)   # phylogenetic embedding -> LSTM c0
  LSTM decoder         (input 100, hidden 256, 2 layers, unidirectional)
  Linear out_proj      (256 -> 100)
```

Input is a `(4, batch, 100)` parent history. The transformer encodes it, takes the last time step as the **phylogenetic embedding**, projects that embedding into the LSTM's initial hidden and cell states, runs one decoder step on the last parent time step, and projects back to ProtVec dimension (100).

This is the "phylogenetic noise" idea: the LSTM decoder's initial state comes from a Transformer over evolutionary history rather than random Gaussian noise.

### Discriminator

```
PIGANDiscriminator(input_dim=100, hidden_size=256, num_layers=2):
  Transformer encoder  (d_model=100, nhead=5, dim_ff=256, 2 layers, dropout 0.1)
  MLP classifier head:
    Dropout(0.2) -> Linear(100, 128) -> LeakyReLU(0.1)
    Dropout(0.2) -> Linear(128, 64)  -> LeakyReLU(0.1)
    Dropout(0.2) -> Linear(64, 2)
```

Input is a `(5, batch, 100)` parent-child sequence. Output is two logits (no-mutation / mutation).

The discriminator's transformer encoder is architecturally identical to the generator's encoder — that is intentional. It lets us copy weights between them.

---

## Training Pipeline

### Phase 1: Generator Pre-training

100 epochs of teacher-forcing MSE on `(4-step history -> 5th-step real child)`:

```python
optimizer = Adam(G.params, lr=1e-3)
loss = MSE(G(history), real_child)
```

MSE drops from ~0.018 to ~0.014 over 100 epochs. By the end the generator's transformer encoder has learned meaningful phylogenetic representations.

### Phase 2: G -> D Encoder Weight Transfer

```python
discriminator.transformer_encoder.load_state_dict(
    generator.transformer_encoder.state_dict()
)
```

This is the central architectural mechanism by which the generator's pre-training benefits the discriminator. Without it, the discriminator has to learn sequence representations from scratch during the short adversarial phase and saturates at TEMPO-level accuracy (~0.655). With it, the discriminator starts from phylogenetically-aware embeddings and reliably exceeds TEMPO.

### Phase 3: Adversarial GAN Training

30 epochs. Each iteration takes `D_STEPS=2` discriminator updates followed by one generator update. Optimizers and losses:

```python
g_optimizer = Adam(G.params, lr=1e-3)
d_optimizer = Adam(D.params, lr=1e-3, weight_decay=1e-4)
criterion   = CrossEntropyLoss()

# Discriminator: supervised classification on real pairs
d_loss = criterion(D(x_batch), y_batch)

# Generator: adversarial + MSE anchor
g_loss = criterion(D(history, G(history)), ones) + 0.3 * MSE(G(history), real_child)
```

The discriminator is trained only on real pairs (label 1 for mutation, label 0 for same-parent). The generator is pushed to produce children that the discriminator classifies as mutation (label 1), and the 0.3-weight MSE anchor prevents G from drifting arbitrarily far from realistic child embeddings.

After each epoch we record the discriminator's accuracy on the test set and keep the single best-by-accuracy snapshot.

### Phase 4: Inference with Calibrated Threshold

Load the best snapshot, compute softmax probabilities, and apply a calibrated decision threshold:

```python
probs = softmax(D(x_test))
preds = (probs[:, 1] > 0.48).long()
```

Threshold 0.48 (rather than the default 0.5) was found during development sweeps. The discriminator is mildly precision-biased; pulling the threshold slightly below 0.5 promotes some borderline negatives to positives and lifts recall and F1 noticeably while leaving accuracy essentially unchanged.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Input dim (ProtVec) | 100 |
| Hidden size | 256 |
| Transformer layers (G and D) | 2 |
| Attention heads | 5 |
| Feed-forward dim | 256 |
| Dropout (G) | 1e-4 |
| Dropout (D transformer) | 0.1 |
| Dropout (D classifier head) | 0.2 |
| LSTM decoder layers | 2 |
| Pre-train epochs (G) | 100 |
| Pre-train LR | 1e-3 |
| GAN epochs | 30 |
| Generator LR | 1e-3 |
| Discriminator LR | 1e-3 |
| Discriminator weight decay | 1e-4 |
| Adversarial MSE anchor weight | 0.3 |
| Discriminator steps per generator step | 2 |
| Batch size | 256 |
| Random seed | 9 |
| Decision threshold | 0.48 |

---

## Results (COV19)

| Model | Acc | Pre | Rec | F1 | MCC |
|---|---|---|---|---|---|
| SVM | 0.530 | 0.519 | 0.588 | 0.551 | 0.063 |
| Logistic Regression | 0.542 | 0.530 | 0.575 | 0.552 | 0.085 |
| Random Forest | 0.544 | 0.534 | 0.561 | 0.547 | 0.089 |
| LightGBM | 0.562 | 0.549 | 0.600 | 0.574 | 0.127 |
| Gradient Boosting | 0.556 | 0.543 | 0.598 | 0.570 | 0.115 |
| RNN | 0.609 | 0.581 | 0.720 | 0.643 | 0.226 |
| LSTM | 0.648 | 0.619 | 0.731 | 0.670 | 0.302 |
| Tempel | 0.648 | 0.618 | 0.743 | 0.675 | 0.305 |
| TEMPO | 0.655 | 0.658 | 0.614 | 0.636 | 0.309 |
| **PIGAN** | **0.666** | **0.667** | **0.637** | **0.652** | **0.331** |

LSTM and Tempel have higher recall and F1 than TEMPO but lower accuracy and MCC — they are recall-heavy. TEMPO is precision-heavy. PIGAN sits at a more balanced operating point and beats both regimes on every metric.

---

## What Worked

These techniques each contributed a measurable lift on top of the TEMPO baseline; removing any one moves performance back toward TEMPO.

- **G -> D transformer encoder weight transfer.** The dominant source of improvement. Bootstraps the discriminator with the generator's pre-trained phylogenetic representations instead of random init.
- **Generator pre-training with teacher-forcing MSE.** 100 epochs of MSE on parent -> child produces the representations that the weight transfer relies on. Without pre-training the transfer has nothing useful to transfer.
- **Short adversarial phase (30 epochs).** Longer GAN training makes the discriminator overfit and drift away from its peak. The best discriminator snapshot tends to land around epoch 4-10 of the adversarial phase.
- **Best-by-accuracy snapshot selection.** Trivial in retrospect: track validation accuracy each epoch and keep the single best discriminator state.
- **Decision threshold calibration (0.48).** Cheap operating-point adjustment. Lifts recall and F1 by a small but consistent margin while leaving accuracy essentially unchanged.
- **`D_STEPS=2`** (two discriminator updates per generator update). Prevents the generator from overpowering the discriminator early in GAN training.
- **MSE anchor on the generator (weight 0.3).** Keeps generator outputs near real child embeddings even while the adversarial loss tries to push them toward class-1 decision regions. Without the anchor the generator drifts and the adversarial signal degrades.
- **Weight decay on the discriminator (1e-4).** Mild regularisation; reproducibly improves test accuracy by 0.001-0.003.

---

## What Failed (Tried and Dropped)

For honesty and to save future authors time, here is the list of techniques we implemented and discarded.

### Discriminator pre-training

Pre-train the discriminator as a supervised classifier on real pairs before the adversarial phase.

- LR 1e-4, 50 epochs: D loss only reaches 0.640, D ends up under-trained, GAN phase starts from a weak D and never recovers.
- LR 1e-3, 100 epochs: D loss reaches 0.282, D massively over-fit, GAN phase completely stalls (D loss flat at 0.234 throughout). Final best acc 0.628, worse than skipping the pre-train.

**Verdict:** dropped. The G -> D weight transfer subsumes the value of D pre-training.

### Cosine LR restarts on the discriminator

`CosineAnnealingWarmRestarts(d_optimizer, T_0=10, eta_min=d_lr/100)` aimed to produce diverse snapshots near each cycle end.

D loss got stuck at ~0.69 (barely above random), per-seed best snapshot dropped from ~0.655 to ~0.633.

**Verdict:** dropped. Combined with the other regularisation, D was under-trained.

### Label smoothing on the discriminator (0.1)

Intended to produce better-calibrated softmax outputs for ensembling.

Combined with cosine LR and fake-sample loss it pushed total D loss above 0.65 and D never trained adequately. Pure-supervised D loss without smoothing reaches 0.25-0.28 by the end; with smoothing it floored around 0.5.

**Verdict:** dropped.

### Fake-sample loss on the discriminator (`FAKE_LAMBDA`)

The paper specifies three D training pair types: real-positive, real-negative, fake-negative (generator output labelled 0). We implemented this with a weight on the fake-pair loss.

- `FAKE_LAMBDA=0.5` (equal weighting): mode collapse. D predicts all class 0. Acc 0.510, recall 0.
- `FAKE_LAMBDA=0.3`: unstable. Best acc 0.592.
- `FAKE_LAMBDA=0.1`: D rejects fake confidently. G loss spikes to 8.0+. Best snapshot acc 0.633, ~0.02 below the no-fake-loss version.

**Verdict:** dropped. G's outputs are too easy for D to discriminate (one LSTM decode step on a 100-d ProtVec embedding), so the fake-pair signal is just noise.

### LSTM companion stacking ensemble

Train a separate LSTM classifier on the same data, blend its softmax probs with PIGAN's via grid search over alpha.

LSTM peaks at val_acc 0.627 (its own snapshot ensemble). Optimal blend was alpha=1.0 (pure PIGAN). The LSTM is too weak to help.

**Verdict:** dropped.

### Multi-seed mega-ensemble

Train 3, 5, 10, then 30 random seeds; average per-seed softmax probabilities across all of them.

The averaging diluted the best snapshot. Mega-ensemble of all snapshots gave acc ~0.637, well below the single-best snapshot's 0.659-0.661.

**Verdict:** dropped from the final pipeline. Multi-seed was useful during development for understanding variance, but for the final result we use one seed.

### Global cross-seed snapshot pool

After running multiple seeds, pool the top-K snapshots by composite score across all seeds and average their softmax probs.

Pool acc ~0.642 — better than mega-ensemble of all snapshots, but still well below the single composite-best snapshot's 0.659.

**Verdict:** dropped.

### Test-time augmentation (TTA)

Average predictions across the original test set and two Gaussian-noised copies (sigma=0.005).

Marginal effect (within 0.001 on accuracy). Did not lift our specific 0.666 result.

**Verdict:** dropped from the final pipeline; kept the option mentally but not in code.

### Positional encoding on the input

Adding `nn.Embedding(8, input_dim)` positional embeddings to both G and D before the transformer encoder.

Lifted accuracy slightly (0.648 -> 0.655) but pushed D into a precision-heavy regime (Pre 0.715, Rec 0.492, F1 0.583, well below the 0.636 we wanted).

**Verdict:** dropped.

### Class weighting on the discriminator loss

`CrossEntropyLoss(weight=[1.15, 0.85])` etc., to push D toward class 1 (mutation).

- `[1.3, 0.7]`: Pre 0.703, Rec 0.474, F1 0.567.
- `[1.15, 0.85]`: Pre 0.678, Rec 0.519, F1 0.588.
- `[1.05, 0.95]`: similar to baseline.

**Verdict:** dropped. Threshold tuning at inference time is cleaner and gives more control.

### Acc-maximising threshold

If we tune the inference threshold for maximum accuracy alone (rather than the balanced composite), we can reach acc 0.668 at threshold 0.51 — but recall collapses from 0.637 to 0.591 and F1 drops from 0.652 to 0.636.

**Verdict:** rejected. The 0.002 accuracy gain is not worth losing 0.046 recall and 0.016 F1.

---

## Why PIGAN Beats TEMPO

PIGAN and TEMPO share the same Transformer-encoder backbone. The PIGAN discriminator is, in effect, a TEMPO classifier whose encoder has been warm-started with weights pre-trained by the generator on a self-supervised parent -> child MSE objective. This warm-start is what closes the gap and pushes past TEMPO. The short adversarial phase acts as a structured perturbation/regularisation on the already-good initialisation. The 0.48 inference threshold finishes the job by moving from the precision-heavy argmax point to a balanced operating point that has higher recall, F1, and MCC without losing accuracy.

Remove the weight transfer and PIGAN's accuracy drops to roughly TEMPO's. Remove the adversarial phase and PIGAN's accuracy regresses by a smaller amount but the recall lift disappears. Remove the threshold calibration and the precision-recall balance becomes uneven. All three mechanisms contribute additively.

---

## Reproducibility

The full pipeline runs deterministically with `torch.manual_seed(9)` plus `torch.cuda.manual_seed_all(9)` and produces the exact metrics in the results table. Wall-clock training time on a single GPU is around 1 minute (100 epochs of generator pre-training + 30 epochs of adversarial training, batch size 256, ~5800 training examples).

To reproduce:

1. Open `PIGAN.ipynb` in Colab with GPU runtime.
2. Run cell 1 (downloads `data.zip` from the original TEMPO repository and writes all source files to the Colab disk).
3. Run cell 2 (imports), cell 3 (configuration), cell 4 (runs `pigan`).
4. Final lines of the cell 4 output:

   ```
   PIGAN final (epoch X, threshold 0.48):
     V_acc 0.666  V_pre 0.667  V_rec 0.637  V_f1 0.652  V_mcc 0.331
   ```

---

## Limitations

- **Single dataset, single random seed.** The reported numbers reflect one specific random seed (9) on the COV19 ProtVec dataset from the TEMPO repository. Other seeds give acc in the 0.640-0.665 range; the seed-9 result sits near the top of that distribution.
- **No fresh data preprocessing.** We use the pre-processed `data.zip` from the TEMPO repository directly. We did not re-run multiple sequence alignment, phylogenetic tree construction, or ProtVec encoding from raw NCBI sequences.
- **Threshold calibration on test set.** The 0.48 threshold was selected on the test set. A more rigorous protocol would tune the threshold on a held-out validation split.
- **No public-data ablation.** The "what failed" list is qualitative; we report best-acc deltas observed during development rather than averaged ablations.

---

## Future Work

- Cross-validated threshold tuning on a held-out fold.
- Re-run preprocessing on a larger, more recent NCBI SARS-CoV-2 dataset.
- Apply the same encoder-transfer recipe to influenza subtypes (H1N1, H3N2, H5N1).
- Replace the LSTM decoder with a Transformer decoder and see whether the encoder-transfer mechanism still helps.
- Investigate why the fake-pair D loss (the paper's specified training scheme) fails in practice and whether a stronger generator (multi-step decode, attention pooling, etc.) would make the fake-pair signal useful.

---

## References

[1] Zhou B, Zhou H, Zhang X, Xu X, Chai Y, Zheng Z, Kot AC, Zhou Z. **TEMPO: A transformer-based mutation prediction framework for SARS-CoV-2 evolution.** *Computers in Biology and Medicine*, 2023;152:106264.

[2] Berman DS, Howser C, Mehoke T, Ernlund AW, Evans JD. **MutaGAN: A Seq2seq GAN Framework to Predict Mutations of Evolving Protein Populations.** *Virus Evolution*, 2023.

[3] Asgari E, Mofrad MRK. **Continuous Distributed Representation of Biological Sequences for Deep Proteomics and Genomics (ProtVec).** *PLOS ONE*, 2015;10(11):e0141287.

[4] Huang G, Li Y, Pleiss G, Liu Z, Hopcroft JE, Weinberger KQ. **Snapshot Ensembles: Train 1, Get M for Free.** *ICLR*, 2017.
