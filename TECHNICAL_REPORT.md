# Technical Report: ChestX-ray14 Diagnosis Assistant

**Author:** jausaff12
**Dataset:** NIH ChestX-ray14 (224×224 resized)
**Model:** DenseNet121 (transfer learning)

---

## 1. Introduction

Chest X-rays are one of the most common diagnostic imaging tools in medicine,
but interpreting them accurately requires trained radiologists, who are in
limited supply relative to imaging volume in many healthcare settings. This
project builds a multi-label deep learning classifier that flags 14 possible
disease findings in a chest X-ray, intended as a decision-support aid — a
second set of eyes that highlights regions and findings worth a closer look,
not a replacement for clinical judgment.

The task has three properties that shape every design decision in this
report: it is **multi-label** (an image can show several findings
simultaneously), the classes are **severely imbalanced** (common findings
like Infiltration vastly outnumber rare ones like Hernia), and the dataset
has **repeated patients** (many individuals have multiple scans), which
creates a data-leakage risk that is easy to miss.

## 2. Dataset & Preprocessing

### 2.1 Dataset

NIH ChestX-ray14, accessed via a 224×224-resized Kaggle mirror
(`khanfashee/nih-chest-x-ray-14-224x224-resized`). ~112,000 images, 14
disease labels plus a "No Finding" category, sourced from labels extracted
via NLP from the original radiology reports (not verified per-image by a
radiologist — see Limitations).

### 2.2 Splitting Strategy

The official `train_val_list_NIH.txt` / `test_list_NIH.txt` files define the
test set. The `train_val` portion is further split 80/20 into train/val —
**by Patient ID, not by image row**. Several patients in this dataset have
multiple scans; a naive random split can place the same patient's images in
both train and validation, letting the model partially learn to recognize
the *patient* rather than the *disease pattern*, which inflates validation
performance in a way that doesn't reflect real generalization. Patient-level
splitting was verified programmatically (hard assertion on the train/val
split, which is fully under the pipeline's control; a soft warning on
train/val-vs-test, since that depends on properties of the official NIH
files rather than this pipeline's own logic).

### 2.3 Preprocessing Pipeline

Built with MONAI transforms:
- Grayscale images converted to 3-channel (required for the ImageNet-pretrained
  backbone, even though X-rays are inherently single-channel)
- Resized to 224×224, scaled to [0,1], then normalized with ImageNet
  mean/std (required for transfer learning to work well)
- Training augmentation: horizontal flip, small rotation (~10°), contrast
  jitter, mild Gaussian noise — deliberately **no vertical flip**, since that
  would invert anatomy in a way that never occurs in real scans, and no large
  rotations, since patients are never imaged at extreme angles

## 3. Model Architecture

**DenseNet121**, pretrained on ImageNet, classifier head replaced with a
14-way linear layer producing raw logits (sigmoid applied inside the loss
function for numerical stability, not as a separate model layer).

DenseNet was chosen over alternatives (e.g. ResNet) for two reasons: its
dense connectivity pattern helps gradient flow and encourages feature reuse,
which is useful for the subtle texture-level patterns in radiographic
findings; and it is the architecture used in the original CheXNet work on
this same dataset, which makes results here more directly comparable to a
known baseline.

**Two-phase transfer learning:**
- **Phase 1** — backbone frozen, only the new classifier head trains. This
  lets the randomly-initialized head adapt without immediately corrupting
  the pretrained backbone's useful features with large early gradients.
- **Phase 2** — full network unfrozen, fine-tuned at a substantially lower
  learning rate (1e-5 vs. 1e-3 in phase 1), since the backbone weights are
  already close to a good solution.

A fresh optimizer and LR scheduler are built for phase 2 rather than reusing
phase 1's — `ReduceLROnPlateau` is bound to a specific optimizer instance and
cannot be repointed at a new learning rate directly.

## 4. Handling Class Imbalance

Rather than oversampling minority classes — ambiguous in a multi-label
setting, since a single image can belong to several classes at once, so
"oversampling the rare class" isn't a well-defined operation — this project
uses a **weighted loss** (`BCEWithLogitsLoss` with per-class `pos_weight`,
computed as `num_negative / num_positive` on the training split, capped at
10.0). The cap exists because a handful of very rare classes would otherwise
produce weights in the hundreds, causing loss spikes and unstable gradients
— a practical stability/correctness tradeoff.

## 5. Training Methodology

- **Mixed precision** (`torch.amp`, float16 forward pass + `GradScaler`) for
  memory efficiency and speed on GPU
- **AdamW** optimizer (decoupled weight decay, more predictable
  regularization across the two very different learning rates used in each
  phase than plain Adam)
- **ReduceLROnPlateau**, monitoring **validation macro-AUC**, not loss — BCE
  loss can keep improving on majority classes while macro-AUC, the metric
  that actually matters for this task, plateaus or regresses under class
  imbalance
- **Early stopping** on macro-AUC, with a single best-checkpoint tracker that
  spans both training phases (so `best_model.pth` reflects the best epoch
  from either phase, not just the last one run)
- **TensorBoard logging** and a `last_model.pth` saved every epoch, so a run
  can be inspected or resumed even if interrupted

## 6. Development Process & Issues Encountered

This section documents real problems hit during development and how they
were diagnosed and resolved — included because the debugging process is as
much a part of the engineering as the final design.

**Dependency conflict (numpy vs. MONAI):** an early version pinned
`numpy<2.0` to satisfy an older MONAI version. This actively fought Colab's
default environment, where numpy 2.x is standard and many other preinstalled
packages depend on it — pinning downward corrupted already-loaded C
extensions (`numpy.dtype size changed` errors). Fix: upgrade MONAI
(`>=1.5.0`, which added numpy 2.0 support) instead of downgrading numpy —
work with the platform's defaults rather than against them.

**Nested dataset folder structure:** one Kaggle mirror of this dataset
extracted to `images-224/images-224/*.png` instead of the expected
`images-224/*.png`, due to how the original zip was packaged. Fixed by
having the config auto-detect which level actually contains image files,
rather than assuming a fixed layout.

**Grad-CAM silently failing on MONAI tensors:** `pytorch-grad-cam`'s
backward hooks returned `None` gradients when given a MONAI `MetaTensor`
(the type MONAI transforms produce) instead of a plain `torch.Tensor`.
Diagnosed by isolating the exact input type causing the failure; fixed with
`.as_tensor()` before passing images into the CAM extractor.

**Undefined threshold edge case:** `sklearn.roc_curve` prepends an artificial
`+inf` threshold to every ROC curve (so the curve start point is well
defined). Without excluding it, the Youden's-J-optimal-threshold search
could select this unusable threshold for a class, silently making that
class always predict negative. Fixed by filtering non-finite thresholds
before the argmax.

**Training throughput on Google Drive:** reading ~112k individual image
files directly from a mounted Google Drive during training was extremely
slow (each file read is a network round trip). Fixed by staging the image
folder onto Colab's local disk once per session before training begins.

## 7. Results

| Metric | Value |
|---|---|
| Validation macro-AUC-ROC | 0.770 |
| Test macro-AUC-ROC | 0.778 |
| Test macro F1 (default 0.5 threshold) | 0.263 |
| Test macro F1 (optimal / Youden's J threshold) | 0.234 |

**Per-class test AUC-ROC ranged from 0.684 (Pneumonia) to 0.882 (Hernia)** —
consistently in a reasonable range across all 14 classes, in line with
published baselines for DenseNet-based approaches on this dataset.

**F1-scores are notably lower and more variable than AUC** (e.g. Hernia: AUC
0.882 but F1 0.022). This gap is a direct consequence of class imbalance,
not a modeling failure: AUC measures ranking quality (does the model assign
higher scores to positives than negatives, independent of any specific
cutoff) and stays informative even with very few positive examples. F1
depends on a chosen decision threshold, and with only a handful of positive
examples in the test set for the rarest classes, F1 becomes a fragile,
high-variance statistic — a single flipped prediction can swing it
substantially. Reporting both metrics, rather than only the more flattering
one, is a deliberate choice.

Notably, the **optimal (Youden's J) threshold produced a slightly lower
macro F1 (0.234) than the default 0.5 threshold (0.263)**. This is expected,
not a bug: Youden's J maximizes sensitivity + specificity, not F1 directly,
so there is no guarantee it also maximizes F1 — the two objectives can
disagree, particularly under heavy imbalance.

## 8. Explainability

Grad-CAM heatmaps (target layer: `features.norm5`, the final batch-norm
layer before global pooling) were generated for sample test images and
overlaid on the originals. Qualitatively, activated regions consistently
fall within lung/chest anatomy rather than image borders or artifacts — a
basic but meaningful sanity check that the model is attending to clinically
relevant regions rather than exploiting some unrelated shortcut in the data.
Where ground-truth bounding box annotations were available
(`BBox_List_2017_Official_NIH.csv`), they were overlaid alongside the
heatmap for direct visual comparison. This is a qualitative check only — no
quantitative localization metric (e.g. IoU) was computed, since the modeling
task here is classification, not detection.

## 9. Limitations

- **Not a diagnostic device.** A research/decision-support prototype only;
  not validated or cleared for clinical use.
- **Label noise.** Labels were extracted via NLP from radiology reports, not
  verified per-image by a radiologist — some fraction are known to be
  inaccurate in the source dataset.
- **Single-institution data.** All data originates from one hospital system;
  generalization to other patient populations, scanners, and imaging
  protocols is untested. No external validation was performed.
- **No cross-validation.** A single patient-disjoint train/val/test split
  was used rather than k-fold cross-validation, primarily for training-time
  practicality at this dataset scale.
- **No deployment benchmarking.** Inference latency and model size were not
  measured against any specific deployment target.
- **Grad-CAM is a debugging aid, not proof of correctness.** A plausible
  heatmap indicates the model attended to a sensible region; it does not
  confirm the model's reasoning matches a clinician's.

## 10. Ethical Considerations

The dataset is de-identified by NIH; this pipeline introduces no additional
PII. Any real-world deployment on patient data would require appropriate
regulatory and institutional data-handling compliance (e.g. HIPAA in the
US). Demographic fields (`Patient Age`, `Patient Gender`) are present in the
source data but were not used as model inputs or audited for subgroup
performance disparities here — a necessary step before any real-world
consideration of this model, and an explicit gap in the current work.

## 11. Conclusion & Future Work

The model achieves a test macro-AUC of 0.778, consistent with published
DenseNet-based baselines on this dataset, with the class-imbalance,
patient-leakage, and threshold-selection issues specific to this task
addressed explicitly rather than left as implicit assumptions. The gap
between strong AUC and weaker/more variable F1 is reported honestly as a
consequence of severe class imbalance, not concealed by only reporting the
more flattering metric.

Natural next steps, not pursued here due to scope: external validation on a
second dataset, quantitative Grad-CAM/bounding-box overlap (IoU) as a
localization metric, subgroup fairness analysis across age/gender, and
inference latency/model-size benchmarking against a concrete deployment
target.
