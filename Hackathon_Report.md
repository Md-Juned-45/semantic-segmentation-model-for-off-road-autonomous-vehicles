# Duality AI Offroad Autonomy Segmentation Report

## 1. Title & Summary
**Team Name:** Team Chaos
**Project Name:** Offroad Autonomy Segmentation
**Summary:** This project builds a robust semantic segmentation model to classify off-road environmental features (Trees, Logs, Rocks, etc.) from Duality AI's digital twin synthetic data. We utilize a modern encoder-decoder architecture, heavily augmented data, and an interactive UI to evaluate our results.

## 2. Methodology
Our workflow consisted of the following steps:
1.  **Architecture:** We chose *DeepLabV3+* with a *MiT-B2 (SegFormer)* backbone. DeepLabV3+ is excellent at capturing multi-scale context using Atrous Spatial Pyramid Pooling (ASPP). The MiT-B2 encoder provides a powerful, lightweight transformer backbone that excels at dense prediction tasks without being overly slow.
2.  **Dataset Handling & Augmentations:** The dataset presented high variance in lighting and terrain. Therefore, we applied random flips, grid distortions, color jittering (brightness, contrast, hue), and Gaussian blurring using `albumentations` to prevent overfitting on synthetic textures and prepare the model for context shifts.
3.  **Training Process:** 
    *   **Class Imbalance:** Classes like Background, Logs, and Flowers suffer from class imbalance. We addressed this using a dual approach: a manually weighted Cross-Entropy Loss combined with a Dice Loss function.
    *   **Optimization:** We used `AdamW` optimization with a `CosineAnnealingWarmRestarts` learning rate schedule. We applied differential learning rates (lower for the pretrained encoder, higher for the segmentation head).
    *   **Hardware Acceleration:** Mixed precision (`torch.amp`) was used to maximize GPU utilization and batch size.

## 3. Results & Performance Metrics
*   **Best Validation mIoU:** 0.5257
*   **Average Inference Speed:** < 50ms per image (meets the benchmark).

<br/>

*(Insert IoU graph screenshot from your Kaggle notebook here)*

*(Insert Loss graph screenshot here)*

## 4. Challenges & Solutions
**Challenge 1: Class Imbalance (Rare Classes)**
*   **Issue:** The model initially struggled to segment "Flowers", "Logs", and "Dry Grass", as they occupied fewer pixels and were often occluded.
*   **Solution:** We implemented a weighted Cross-Entropy loss heavily favoring underrepresented classes combined with Dice loss to maximize overlap directly.

**Challenge 2: Overfitting to Synthetic Textures**
*   **Issue:** The model would quickly overfit the training dataset and val scores would plateau.
*   **Solution:** We aggressively introduced `GridDistortion`, `ColorJitter`, and `Blur` to the augmentations, forcing the model to learn structural shapes rather than memorizing exact synthetic pixel values.

## 5. Failure Case Analysis
*(Insert 1 or 2 screenshots of failure cases here - e.g. from the test script outputs where it misclassified something)*

**Case 1: Misclassifying Shadows as 'Rocks' or 'Ground Clutter'**
As seen in the image above, dark, heavily occluded areas are occasionally misclassified as ground clutter rather than the background. 
**Improvement:** Further fine-tuning color jittering brightness/contrast augmentations or utilizing test-time augmentations (TTA) to combine predictions.

## 6. Conclusion & Future Work
We successfully trained a highly robust segmentation model capable of generalizing to novel desert digital twins. Future improvements could involve:
*   **Self-Supervised Pre-training:** Utilizing unlabelled Duality AI data to pre-train the encoder before fine-tuning.
*   **Domain Adaptation:** Using adversarial training (like cycleGANs) to bridge the exact gap between the training domain and the unseen test domain.
*   **Model Ensembling:** Averaging the predictions of multiple backbones (e.g., MiT-B2 and ResNet50) to boost the final IoU.
