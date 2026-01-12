# DeitFake: Deepfake Detection using Vision Transformers

A deepfake detection system built on Vision Transformers (ViT) using Facebook's Data-efficient Image Transformers (DeiT) architecture, fine-tuned for binary classification of real and fake images.

## 🎯 Overview

This repository contains the implementation and validation of a deepfake detection model based on the DeiT (Data-efficient Image Transformer) architecture. The model achieves state-of-the-art performance in distinguishing between real and synthetically generated (deepfake) images across multiple benchmark datasets.

**Key Features:**
- Fine-tuned `facebook/deit-base-patch16-224` Vision Transformer
- Multi-GPU and TPU training support
- Validated on multiple benchmark datasets (CelebDF-v2, FaceForensics++, OpenForensics)
- High accuracy (98.71%) and AUROC (99.93%) on test data
- Face detection and alignment preprocessing pipeline

## 📊 Model Performance

### Training Dataset Performance
The model was trained on the [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) dataset:

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.71% |
| **Macro F1-Score** | 98.71% |
| **AUROC** | 99.93% |

### Validation on Benchmark Datasets

The model has been extensively validated on multiple standard deepfake detection benchmarks:

1. **CelebDF-v2**: Industry-standard celebrity deepfake dataset
2. **FaceForensics++**: Large-scale forensics dataset with multiple manipulation techniques
3. **OpenForensics**: Open-world deepfake detection benchmark

Detailed validation results and metrics are available in the respective validation notebooks.

## 🏗️ Model Architecture

**Base Model:** `facebook/deit-base-patch16-224`

DeiT (Data-efficient Image Transformer) is a Vision Transformer architecture that:
- Uses a transformer encoder architecture adapted for image classification
- Processes 224×224 images with 16×16 patches
- Employs distillation techniques for efficient training
- Contains approximately 86M parameters

**Classification Head:** Binary classifier (Real vs. Fake)

## 🔧 Installation

### Prerequisites
```bash
pip install torch torchvision
pip install transformers datasets
pip install scikit-learn imbalanced-learn
pip install facenet-pytorch opencv-python
pip install tqdm pandas numpy pillow
```

### For TPU Support (Optional)
```bash
pip install torch_xla[tpu]
```

## 🚀 Usage

### Quick Start - Inference

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

# Load the model and processor
model = AutoModelForImageClassification.from_pretrained("sakshamkr1/deepfake-fb-deit-vit-224")
processor = AutoImageProcessor.from_pretrained("sakshamkr1/deepfake-fb-deit-vit-224")

# Load and preprocess image
image = Image.open("path/to/image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Get prediction
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()

# 0: Fake, 1: Real
print(f"Prediction: {'Real' if predicted_class == 1 else 'Fake'}")
```

### Training from Scratch

Use the `DeitFake_complete.ipynb` notebook which includes:
1. Data loading and preparation
2. Data augmentation and preprocessing
3. Model definition and training configuration
4. Training with mixed precision
5. Evaluation and metrics visualization

### Retraining on Custom Data

Use the `DeitFake_retrain.ipynb` notebook to fine-tune the model on your own dataset.

### Validation on Benchmark Datasets

Three validation notebooks are provided:
- `DeitFake_Validation1_celebdf-v2.ipynb` - CelebDF-v2 validation
- `DeitFake_Validation2_FF++.ipynb` - FaceForensics++ validation
- `DeitFake_Validation3_OpenForensics_Official_TestDev.ipynb` - OpenForensics validation

## 📁 Repository Structure

```
DeitFake/
├── DeitFake_complete.ipynb                    # Complete training pipeline
├── DeitFake_retrain.ipynb                     # Retraining on custom data
├── DeitFake_Validation1_celebdf-v2.ipynb      # CelebDF-v2 validation
├── DeitFake_Validation2_FF++.ipynb            # FaceForensics++ validation
├── DeitFake_Validation3_OpenForensics_Official_TestDev.ipynb  # OpenForensics validation
└── README.md                                   # This file
```

## 🔬 Training Details

### Hyperparameters
- **Epochs:** 5
- **Learning Rate:** 2e-5
- **Batch Size:** Configured for multi-GPU/TPU
- **Weight Decay:** 0.01
- **Mixed Precision:** Enabled (fp16)
- **Optimizer:** AdamW

### Data Augmentation
- Random horizontal flipping
- Color jittering
- Random rotation
- Normalization with ImageNet statistics

### Class Balancing
- Random over-sampling applied to balance real and fake classes during training

## 🎓 Citation

If you use this work in your research, please reference the paper:

```bibtex
@article{deitfake2025,
  title={DeitFake: Deepfake Detection using Vision Transformers},
  year={2025},
  note={arXiv:2511.12048}
}
```

## 🙏 Acknowledgments

- **Model:** Based on Facebook AI's [DeiT](https://github.com/facebookresearch/deit) (Data-efficient Image Transformers)
- **Framework:** Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- **Training Dataset:** [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) on Kaggle
- **Validation Datasets:**
  - CelebDF-v2
  - FaceForensics++
  - OpenForensics

## 📝 Model Card

The trained model is available on Hugging Face Hub: [sakshamkr1/deepfake-fb-deit-vit-224](https://huggingface.co/sakshamkr1/deepfake-fb-deit-vit-224)

### Intended Uses
- Research purposes related to deepfake detection
- Binary classification of images as Real or Fake
- Benchmarking deepfake detection algorithms

### Limitations
- Designed for face images; may not generalize to other types of deepfakes
- Performance may vary on deepfakes generated by methods not represented in training data
- Requires face detection preprocessing for optimal results

### Ethical Considerations
This model should be used responsibly and ethically:
- Not for surveillance without proper consent
- Not for discriminatory purposes
- Consider privacy implications when processing personal images
- Be aware of potential biases in training data

## 🔒 License

Please refer to the individual licenses of the datasets and base models used in this project.

## 📧 Contact

For questions or issues, please open an issue in this repository.

---

**Note:** This is a research project. Performance may vary depending on the nature and quality of input images and the specific deepfake generation techniques used.
