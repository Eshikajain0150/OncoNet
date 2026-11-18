# OncoNet

**Attention-Guided Segmentation and Multi-Resolution Transformer Learning for Blood Cancer Detection**

Pipeline:

1. **Segmentation**: UNet++ with Attention Gates and multi-scale feature fusion.
2. **Feature extraction**: multi-resolution DeiT-Small/16 on 128×128 and 256×256 cell regions, giving a 768-d embedding.
3. **Classification**: Attention-Enhanced MLP (AE-MLP) for AML / ALL / Healthy, with error-guided fine-tuning.

Baselines: U-Net + ResNet-50 and U-Net + ViT-Small/16.

## Data

Kaggle dataset: [Blood Cell Images for Cancer Detection](https://www.kaggle.com/datasets/sumithsingh/blood-cell-images-for-cancer-detection). Extract it to `data/raw/`.

| Class | Source folder(s) | Images |
|---|---|---|
| AML | `myeloblast` | 200 |
| ALL | `erythroblast` | 175 |
| Healthy | `basophil`, `monocyte`, `seg_neutrophil` | 125 |

The split is stratified 70/15/15 (350 / 75 / 75) with seed 42. Five-fold CV runs on the 425 training + validation images.

## Requirements

Python 3.10, PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.1, and `requirements.txt`.

## Usage

Run the scripts from the repository root, in order. All settings are in `configs/onconet.yaml`.

| Step | Script | Stage |
|---|---|---|
| 1 | `01_prepare_data.py` | subset, splits, preprocessing |
| 2 | `02_make_masks.py` | reference masks |
| 3 | `03_train_segmentation.py` | segmentation models (`--variant`, `--all`, `--no-stain-norm`) |
| 4 | `04_extract_rois.py` | cell regions (`--seg-variant`) |
| 5 | `05_train_proposed.py` | DeiT + AE-MLP + error-guided fine-tuning |
| 6 | `06_train_baselines.py` | baselines (`--model resnet50` / `vit_small`) |
| 7 | `07_cross_validation.py` | five-fold cross-validation |
| 8 | `08_interpretability.py` | Grad-CAM, Attention Focus Score |
| 9 | `09_robustness.py` | robustness tests |
| 10 | `10_benchmark.py` | parameters, MACs, runtime |
| 11 | `11_make_report.py` | result tables and figures |

Outputs and logs go to `outputs/`. Results may vary slightly with hardware and library versions.

## Structure

```
configs/    settings
onconet/    data, models, losses, metrics, training, interpretability
scripts/    experiment stages
```

## License

MIT
