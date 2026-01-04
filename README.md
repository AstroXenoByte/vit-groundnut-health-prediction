# Groundnut Leaf Health Prediction using Vision Transformers (ViT)

## Overview
This project applies **Vision Transformer (ViT)** models to predict the health status of groundnut (peanut) plants based on leaf images.  
The goal is to assist early detection of unhealthy crops using deep learning and computer vision.

The model achieved **~97% accuracy** on a hybrid dataset.

---

## Dataset
- **Hybrid dataset**
  - Custom groundnut leaf images collected by the project team
  - Additional publicly available datasets sourced online
- Dataset split into:
  - Training
  - Validation
  - Testing
- Images organized using `ImageFolder` structure

---

## Technologies Used
- Python
- PyTorch
- Torchvision
- Hugging Face Transformers
- Vision Transformers (ViT / DeiT)
- TQDM

---

## Model Architecture
- Pretrained model: **facebook/deit-tiny-patch16-224**
- Modified for multi-class classification
- Fine-tuned using transfer learning
- Optimized for **CPU-friendly deployment**

---

## Training Configuration
- Image Size: 224 × 224
- Batch Size: 4
- Epochs: 10
- Learning Rate: 2e-5
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Gradient Accumulation used for memory efficiency

---

## Performance
- **Accuracy:** ~97% on validation/testing data
- Stable convergence across epochs
- Effective generalization despite mixed data sources

---

## Workflow
1. Dataset preparation and augmentation
2. Transfer learning with pretrained ViT
3. Model fine-tuning
4. Validation and accuracy evaluation
5. Model export for deployment

---

## Saved Model
- `vit_groundnut_cpu_friendly.pth`  
This file contains the trained model weights and can be loaded for inference or deployment.

---

## Applications
- Smart agriculture
- Crop health monitoring
- Precision farming
- Early disease detection in plants

---

## Future Improvements
- Mobile or edge deployment
- Larger dataset collection
- Explainability (Grad-CAM / attention visualization)
- Integration with IoT or drone imagery

---

## Author
Senzokuhle Mokoena
