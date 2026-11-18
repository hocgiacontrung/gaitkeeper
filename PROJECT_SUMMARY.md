# GaitKeeper — Disentangling Identity from Disease in 3D Parkinsonian Gait

*Biometrics project · Ngoc Minh Le, Jack Markham*

Models that score Parkinson's severity from walking might partly be recognizing **who** is walking. We want to **measure** that identity leak and **remove** it.

## Abstract

Disentangling identity from disease in 3D Parkinsonian gait. Gait is both a biometric (it can identify who a person is) and a clinical signal (it changes with Parkinson’s disease). Models that estimate Parkinson’s severity from walking may therefore also absorb identity information, which can hurt generalization to new patients and hospitals and creates a privacy risk. Using CARE-PD, the largest public multi-site dataset of 3D Parkinsonian gait (9 cohorts, 8 clinical sites, ~360 participants, in SMPL body-model format with UPDRS-gait labels), we will (1) measure how much identity information is present in the features of standard severity-estimation models, (2) reduce it with an adversarial “identity detective” trained alongside the severity model, and (3) evaluate whether this improves performance on CARE-PD’s cross-site benchmarks while lowering re-identification risk. As a complementary analysis, we will convert joint trajectories into spectrograms to study which walking rhythms carry identity information and which carry disease information. Healthy gait data in the same format (e.g., AMASS, Gait3D) can support pretraining and identity modeling. The project builds on CARE-PD’s open-source baselines and runs on modest hardware. To our knowledge, identity disentanglement has not yet been studied for Parkinsonian gait, although the underlying technique is established in other domains such as speech and face analysis.

## Background

- **Clinical task:** a doctor rates MDS-UPDRS item 3.10 (gait) on a 0–4 scale by watching the patient walk. It's subjective and only happens at clinic visits.
- **Goal of the field:** predict that score automatically from motion data.
- **Dataset: CARE-PD** (NeurIPS 2025). It has 9 cohorts from 8 centres in 6 countries, all converted to anonymized SMPL body meshes. 4 cohorts have UPDRS-gait labels.
- **Current best (MoCha 2026 winner):** a frozen MotionAGFormer-S encoder with a linear probe, macro-F1 0.694. The gains came from subject-level averaging and test-time calibration, not from better features. So subject and site structure matter a lot.


## Method: adversarial training

```
3D walk (SMPL) → Encoder → Features ─┬─→ Severity head  (predicts UPDRS-gait)
                                     └─→ [Gradient Reversal] → Identity detective (guesses who)

Total loss = L_severity − λ · L_identity
```

The detective trains normally, but the gradient flowing back into the encoder is flipped, so the encoder learns to fool it. λ sets how hard we push.

## Evaluation

| Question | Metric | Protocol | Goal |
|---|---|---|---|
| Is severity still right? | Macro-F1, per-class recall | LOSO (within dataset) | Same or higher |
| Does it transfer? | Macro-F1 | LODO (leave-one-dataset-out) | Better than baseline |
| How much identity leaks? | Re-ID accuracy of a fresh probe on held-out walks | Frozen features | Down to chance (1/N) |

**Variants:** no adversary · site adversary · identity adversary · both
**Main figure:** privacy vs accuracy curve (re-ID vs macro-F1, one point per λ)

## Expected outcomes

1. Identity leaks into standard severity features.
2. The adversary pushes re-ID down toward chance.
3. Severity at unseen sites holds or improves. Outcomes 1 and 2 are still a result even if 3 doesn't happen.

## What the data looks like

Each `<Dataset>.pkl` is `dict[subject_id][walk_id] → walk`, where each walk has:

| Key | Shape | Meaning |
|---|---|---|
| `pose` | (T, 72) | SMPL joint rotations: 24 joints × 3 (axis-angle) |
| `trans` | (T, 3) | Root position over time |
| `beta` | (1, 10) | **Body shape**. This is basically an identity fingerprint |
| `fps` | int | Frame rate |
| `UPDRS_GAIT` | int / None | Severity label, 0–4 |
| `medication` | — | ON/OFF where available |

The `folds/` files give the official train/eval subject splits. `Canonicalized_SMPL_pickles/` has the same data normalized (facing direction, starting position).
