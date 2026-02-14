# HARE Skin Lesion Classifier - User Guide

## Overview

The **HARE Skin Lesion Classifier** is an AI-powered web application that performs **binary classification** of skin lesions: Melanoma (Malignant) vs Melanocytic Nevus (Benign). It provides high-accuracy predictions, visual explanations via Grad-CAM with attention concentration scores, and demonstrates model robustness through FGSM adversarial attack simulation.

**Key Features:**
- ✅ Binary skin lesion classification (Melanoma vs Nevus)
- 🔥 Grad-CAM visualization with **attention dispersion score** showing model focus
- 🛡️ FGSM adversarial attack simulation to test robustness (95.0% PGD robustness)
- ⚠️ Clinical risk banners (Benign, Malignant)

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Install Dependencies

Open PowerShell and navigate to the project directory:

```powershell
cd C:\Users\MaleeshaRodrigo\Documents\University\FYP\Prototype
```

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install required packages:

```powershell
pip install -r requirements.txt
```

The `requirements.txt` includes:
- streamlit (web UI)
- torch, torchvision, timm (deep learning)
- Pillow, numpy, matplotlib (image processing)

### Step 2: Place Model Weights

Place your **binary-trained model weights** in the same directory as `app.py`. The app will look for:
- `hare_final.pth` (preferred - GA-optimized binary model)
- `hare_best.pth` (backup - best checkpoint)
- `best_hare_model.pth` (legacy filename)

**Directory Structure:**
```
C:\Users\MaleeshaRodrigo\Documents\University\FYP\Prototype\
├── app.py
├── hare_model.py
├── utils.py
├── requirements.txt
├── hare_final.pth  (← place binary weights here)
├── .venv/
└── USER_GUIDE.md
```

### Step 3: Run the Application

```powershell
streamlit run app.py
```

The app will launch at `http://localhost:8501` in your default browser.

---

## How to Use

### 1. Upload a Skin Lesion Image

- Click **"Upload a skin lesion image"** button
- Select a PNG, JPG, or JPEG file from your device
- The image will appear on the left side of the screen

**Supported Image Formats:**
- `.png`, `.jpg`, `.jpeg`
- Recommended resolution: 224×224 pixels or larger
- Recommended file size: < 5 MB

### 2. View the Prediction

After uploading, the right side displays:

**Specific Diagnosis:**
- Shows the predicted skin lesion type: "Melanocytic Nevus (Benign)" or "Melanoma (Malignant)"
- Displayed as a metric card

**Confidence Score:**
- A progress bar (0-100%) showing model confidence
- Example: 87% confidence

**Risk Banner:**
- **Red Banner: "🔴 MALIGNANT - High Risk: Immediate Consultation Recommended"**
  - Predicted: Melanoma
  - Action: Consult a dermatologist urgently

- **Green Banner: "🟢 BENIGN - Low Risk: Routine Check Recommended"**
  - Predicted: Melanocytic Nevus
  - Action: Standard follow-up care

**Class Probabilities:**
- Probability distribution for both classes (Nevus, Melanoma)

---

### 3. Inspect the Grad-CAM Heatmap

The **Grad-CAM Heatmap** section visualizes which regions of the image the model focused on for its decision.

**Two visualizations are shown:**

**Left: Heatmap Overlay**
- Shows the attention heatmap overlaid on the original image
- Red = high importance, Blue = low importance
- Helps understand model reasoning

**Right: Raw Heatmap**
- Pure grayscale heatmap (normalized 0-1)
- Easier to analyze exact attention regions

**Attention Concentration Score (Dispersion Score):**
- **0.0-0.5 (Dispersed):** Model attends to many regions across the image. May indicate background influence.
- **0.5-1.0 (Focused):** Model concentrates on specific lesion regions. Demonstrates interpretable decision-making.

**How to Interpret:**
- Bright/red regions indicate pixels the model used most for classification
- Dark/blue regions indicate pixels that had less influence
- Focused attention patterns = higher interpretability and trustworthiness
- Dispersed attention = potential bias toward background or irrelevant features

---

### 4. Simulate an Adversarial Attack

Click the **"🎯 Simulate FGSM Attack"** button to test model robustness.

**What Happens:**
1. The app applies FGSM (Fast Gradient Sign Method) noise to the image
2. The perturbation is imperceptible to humans but optimized to fool the model
3. Displays the perturbed image side-by-side with the original
4. Shows the new prediction after attack
5. Compares probability distributions before/after

**Interpretation:**
- **Prediction unchanged?** → Model is **robust** to this adversarial noise ✅
  - This indicates successful adversarial training (OATGA optimization)
  - Consistent with 95.0% PGD robustness from research
- **Prediction changed?** → Model is **vulnerable** to FGSM attacks ⚠️
  - Shows potential area for improvement in adversarial training
  - May indicate insufficient exposure to adversarial examples during training

**FGSM Parameters (Fixed):**
- Epsilon (ε) = 0.03 (in normalized image space, ~7.6×10⁻³ per channel)
- The noise is imperceptible to the human eye but specifically crafted to maximize misclassification

---

## Understanding the Classes

| Class | Disease | Risk Level | Characteristics |
|-------|---------|-----------|-----------------|
| **MEL** | Melanoma | 🔴 MALIGNANT | Most dangerous skin cancer; asymmetric, irregular borders, varied color |
| **NV** | Melanocytic Nevus | 🟢 BENIGN | Common mole; symmetric, uniform color, stable over time |

---

## About HARE & OATGA

### HARE (Hybrid Attention ResNet Ensemble)

HARE combines two powerful deep learning architectures for robust melanoma detection:

1. **ResNet50** - Convolutional Neural Network (CNN)
   - Excellent at extracting texture, edges, and local features
   - Fast and computationally efficient
   - Learns color patterns, rough texture, irregular borders
   - Outputs 2048-dimensional feature vectors

2. **Vision Transformer (ViT)** - Transformer-based model
   - Captures long-range dependencies and global structure
   - Understands holistic lesion characteristics (shape, symmetry)
   - Learns contextual information from the entire image
   - Outputs 768-dimensional feature vectors

3. **GA-Optimized Fusion Layer**
   - Concatenates 2048 CNN features + 768 ViT features = 2816 dimensions
   - Intelligently fuses and reduces to 512-dimensional latent representation
   - Parameters evolved via Genetic Algorithm (OATGA) for adversarial robustness
   - Final layer outputs 2 class logits (Nevus vs Melanoma)

**Why Hybrid?**
- CNN catches fine details: color, texture, edges
- ViT understands overall shape and structure: symmetry, borders, distribution
- Together: More accurate, robust, and interpretable predictions
- Achieves **84.9% clean accuracy** and **95.0% adversarial robustness** (PGD ε=8/255)

---

### OATGA (Optimization-Augmented Training with Genetic Algorithm)

After training HARE on clean images, OATGA fine-tunes the model to be adversarially robust:

1. **Adversarial Training Phase (Phase A)**
   - Train HARE on both clean and perturbed images
   - Minimize loss on adversarial examples (FGSM, PGD attacks)
   - Increases natural robustness to perturbations
   - Produces checkpoint: `hare_best.pth`

2. **Genetic Algorithm Optimization Phase (Phase B)**
   - Evolves the fusion layer parameters for better robustness
   - Search space: Fusion layer weights (2816 → 512)
   - Fitness function: 0.3 × Clean_Accuracy + 0.7 × Adversarial_Accuracy
   - Population-based search with crossover, mutation, and selection operations
   - Produces final model: `hare_final.pth` (GA-optimized)

**Result:** A model that maintains high diagnostic accuracy while resisting adversarial attacks
- **Clean Accuracy:** 84.9% on test set
- **PGD Robustness:** 95.0% (ε=8/255) - 3.3× more robust than ResNet50-only baseline (28.4%)
- **CAR Score:** 0.7295 (Clean-Adversarial Robustness ratio)

---

## Troubleshooting

### Issue: "Binary weights not found" Error

**Solution:**
- Ensure model weights file is in the same directory as `app.py`
- File should be named `hare_final.pth`, `hare_best.pth`, or `best_hare_model.pth`
- Check file extension is exactly `.pth` (not `.pth.1` or `.pth~`)
- Verify weights are from **binary-trained model** (2 classes: Nevus, Melanoma)

### Issue: App Crashes on Image Upload

**Solution:**
- Ensure image is in PNG, JPG, or JPEG format
- Image file size should be < 5 MB
- Try re-uploading with a different image
- Ensure image is not corrupted or invalid format

### Issue: Grad-CAM Not Displaying or Showing "Grad-CAM generation failed"

**Solution:**
- Model must be in eval mode (automatically set in app.py)
- Ensure hooks are properly registered on ResNet50 layer4[-1]
- Try refreshing the page and re-uploading the image
- Check that weights loaded successfully (green ✅ indicator)

### Issue: "No module named 'timm'" Error

**Solution:**
- Install timm library explicitly:
  ```powershell
  pip install timm
  ```
- Or reinstall all dependencies:
  ```powershell
  pip install -r requirements.txt
  ```

### Issue: CUDA/GPU Not Detected

**Solution:**
- App automatically falls back to CPU (slower but functional)
- CPU inference will be ~5-10x slower than GPU
- To use GPU: ensure PyTorch is installed with CUDA support
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- Verify GPU is detected:
  ```powershell
  python -c "import torch; print(torch.cuda.is_available())"
  ```

### Issue: Weight Loading with "strict=False" Message

**Solution:**
- This is **normal and expected** behavior
- Occurs when loading 8-class weights into binary (2-class) model
- The classifier layer is skipped; CNN and ViT features still loaded
- App will still work but predictions may be less reliable
- **Recommended:** Use binary-trained weights (`hare_final.pth` from binary training)

---

## Performance Tips

### For Faster Inference:
1. **Use GPU** - If available, CUDA will speed up predictions by 5-10x
2. **Reduce image size** - Pre-process images to 224×224 before uploading
3. **Close other apps** - Free up RAM for smoother operation

### For Accurate Results:
1. **Ensure good lighting** in the skin lesion photograph
2. **Capture full lesion** - Don't crop or cut off edges
3. **Avoid shadows and reflections** on the lesion
4. **Use clinical photographs** - Professional/dermatologist-quality images work best
5. **Clear background** - Solid, uniform background is ideal

### Improving Robustness:
- The model's **95.0% PGD robustness** demonstrates strong adversarial resistance
- Robust predictions are less sensitive to small image variations
- If model is vulnerable to FGSM (ε=0.03), this indicates the image may have quality issues

---

## Limitations & Disclaimers

⚠️ **CRITICAL: This app is for research and educational purposes only.**

- **Not a medical device** - Do not use as primary diagnostic tool
- **Not FDA-approved** - For demonstration purposes only
- **Always consult a dermatologist** for skin lesion diagnosis
- **Binary classification only** - Designed for Melanoma vs Nevus detection (not other skin conditions)
- **Accuracy varies** - Model trained on ISIC 2019 dataset; performance depends on image quality
- **No liability** - Authors are not responsible for misdiagnosis or incorrect predictions
- **Research context** - Results (84.9% clean accuracy, 95.0% PGD robustness) are from controlled A100 training
- **Real-world performance** - May differ from reported metrics due to deployment conditions

---

## Next Steps

After using the app:

1. **For Research:** Document predictions and compare with dermatologist assessment
2. **For Real Diagnosis:** Consult a qualified dermatologist
3. **For Feedback:** Report robustness findings (whether FGSM attacks succeeded or failed)
4. **For Deployment:** Model can be fine-tuned on custom datasets for specific use cases
5. **For Publication:** Use robustness metrics (95.0% PGD) and Grad-CAM interpretability evidence

---

## Support & Questions

If you encounter issues:

1. Check this USER_GUIDE for common problems
2. Review console output for error messages
3. Ensure all dependencies are installed:
   ```powershell
   pip list | findstr streamlit
   ```
4. Verify weights file exists and is readable:
   ```powershell
   Get-Item hare_final.pth
   ```

---

## Glossary

| Term | Definition |
|------|-----------|
| **FGSM** | Fast Gradient Sign Method - adversarial attack technique |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping - visualization technique |
| **Adversarial** | Input designed to fool neural networks (imperceptible noise) |
| **Robustness** | Model's ability to maintain accuracy under adversarial attacks |
| **Ensemble** | Combination of multiple models to improve performance |
| **Fusion** | Combining features from multiple models (CNN + ViT) |
| **PGD** | Projected Gradient Descent - stronger adversarial attack |
| **CAR Score** | Clean-Adversarial Robustness ratio (0.3×clean + 0.7×robust) |
| **Dispersion Score** | Metric measuring attention concentration in Grad-CAM |

---

**Version:** 2.0 (Binary HARE)  
**Last Updated:** February 2026  
**Model:** HARE + OATGA (ResNet50 + ViT Hybrid Ensemble, Binary Classification)
