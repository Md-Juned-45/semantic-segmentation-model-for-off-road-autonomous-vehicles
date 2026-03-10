# Duality AI Offroad Autonomy Segmentation Report

## 1. Title and Summary

**Team Name:** Team Chaos  
**Project Name:** Offroad Autonomy Semantic Segmentation

This project focuses on semantic segmentation for off-road autonomous navigation using synthetic data provided by Duality AI. Our objective was to classify each pixel in an off-road scene into one of 11 semantic classes, including trees, bushes, rocks, landscape, sky, logs, and flowers. We built a complete pipeline covering training code, evaluation, visualization, and an interactive demo interface for qualitative testing.

The actual hackathon training run was performed in a **Kaggle Notebook GPU environment**, and the resulting model checkpoint was exported for local inference and demo usage in this repository.

## 2. Methodology

### Model Architecture

We selected **DeepLabV3+** with a **MiT-B2** backbone from `segmentation_models_pytorch`.

This choice gave us a strong balance between segmentation quality and inference efficiency:
- DeepLabV3+ captures multi-scale context effectively through atrous spatial pyramid pooling.
- MiT-B2 provides a lightweight but expressive encoder well-suited for dense prediction tasks.
- The architecture performed well for irregular natural structures common in off-road environments.

### Dataset Handling and Preprocessing

The dataset contains synthetic off-road scenes with paired segmentation masks. We organized the data into training, validation, and test splits using the following structure:

```text
data/
  train/
    Color_Images/
    Segmentation/
  val/
    Color_Images/
    Segmentation/
  testImages/
    Color_Images/
    Segmentation/
```

We converted raw mask values into contiguous training class IDs to support multi-class segmentation training.

### Data Augmentation

To reduce overfitting and improve generalization to unseen environments, we applied augmentations using `albumentations`:
- horizontal flip
- vertical flip
- random rotation
- shift, scale, and rotate
- grid distortion and elastic transform
- color jitter
- gaussian blur
- grayscale conversion
- normalization

These transformations were especially helpful in preventing the model from memorizing synthetic textures or lighting patterns.

### Loss Function and Optimization

A major challenge in the dataset was class imbalance. Large classes such as landscape and sky dominate the scene, while classes like flowers, logs, and dry grass occupy relatively small regions.

To address this, we used:
- **weighted cross-entropy loss** to emphasize underrepresented classes
- **Dice loss** to improve overlap-based segmentation quality

The final training objective was a combined weighted CE + Dice loss.

For optimization, we used:
- **AdamW** optimizer
- **CosineAnnealingWarmRestarts** learning rate scheduling
- differential learning rates for encoder and decoder components
- mixed precision training for improved speed and memory efficiency on GPU

### Training Environment

The main experiment and best-performing checkpoint were produced in **Kaggle Notebook** using GPU acceleration. This repository preserves the code used for training and provides the exported model checkpoint for local batch inference and Flask-based demo testing.

## 3. Results and Performance

Our best validation performance reached:

- **Best Validation mIoU:** `0.5257`
- **Average Inference Speed:** typically under `50 ms` per image in the intended setup

In addition to quantitative metrics, we generated colorized prediction masks for qualitative review and built a Flask-based web interface for interactive testing.

## 4. Challenges and Solutions

### Challenge 1: Class Imbalance

**Issue:** Rare classes such as flowers, logs, and dry grass were harder to detect because they covered fewer pixels and were often visually similar to nearby regions.

**Solution:** We introduced manually tuned class weights in cross-entropy loss and combined it with Dice loss. This helped the model pay more attention to smaller semantic regions during training.

### Challenge 2: Overfitting to Synthetic Appearance

**Issue:** Since the dataset is synthetic, the model can easily overfit to texture, color distribution, and scene-specific rendering artifacts.

**Solution:** We used stronger spatial and color augmentations to push the model toward learning structural features instead of memorizing exact synthetic patterns.

### Challenge 3: Thin and Ambiguous Structures

**Issue:** Classes such as logs, clutter, and edge regions are harder to segment accurately because they are thin, irregular, and sometimes partially occluded.

**Solution:** The DeepLabV3+ architecture and Dice-based overlap objective helped improve mask consistency for fine structures, though these classes remain challenging.

## 5. Failure Case Analysis

While the model performed well overall, a few common failure patterns remained:

### Shadows and Dark Regions

Dark or heavily occluded areas were sometimes confused with rocks or ground clutter instead of background or terrain.

### Rare-Class Confusion

Small flower patches and logs were occasionally absorbed into nearby dry bushes or clutter when they occupied very few pixels.

### Boundary Errors

Transitions between landscape, bushes, and clutter sometimes produced soft or imprecise boundaries.

These observations suggest that additional boundary-aware losses, stronger rare-class sampling, or post-processing could further improve results.

## 6. Conclusion

We developed a complete semantic segmentation system for off-road autonomy scenes using PyTorch and DeepLabV3+ with a MiT-B2 encoder. The project includes training code, metric evaluation, batch inference, and an interactive browser demo for visual testing. The best hackathon model was trained in Kaggle Notebook and then exported for local usage. Our final system achieved a validation mIoU of `0.5257` while maintaining efficient inference performance.

## 7. Future Work

Potential next steps include:
- domain adaptation from synthetic to real-world off-road imagery
- self-supervised pretraining on additional unlabeled data
- model ensembling for improved mIoU
- test-time augmentation for more stable predictions
- improved handling of rare and thin classes through better sampling or auxiliary losses
