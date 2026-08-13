# garbage_classification_agent
# ♻️ Garbage Classification & Smart Recycling Recommendation Agent

An AI-powered system that classifies garbage into 6 categories and provides 
recycling recommendations, built by comparing three deep learning architectures.

## Problem Statement
AI Agent for Garbage Classification and Smart Recycling Recommendation, 
comparing EfficientNet-B3, ResNet50, and MobileNetV3 on the Kaggle Garbage 
Classification dataset.

## Models Compared
| Model | Test Accuracy |
|---|---|
| EfficientNetB3 | 96.61% (Best) |
| ResNet50 | 93.49% |
| MobileNetV3 | 96.35% |



## Dataset
[Kaggle Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- 6 classes: cardboard, glass, metal, paper, plastic, trash
- Train/Validation/Test split: 70/15/15

## Approach
1. Data cleaning (removed corrupted/non-image files)
2. Transfer learning with ImageNet pretrained weights
3. Class-weighted training to handle class imbalance
4. 3-stage progressive fine-tuning (frozen → partial unfreeze → full unfreeze)
5. Evaluation on held-out test set + real-world images
6. Best model (EfficientNetB3) deployed as a Streamlit web app

## Files
- `Garbage_Classification_Project.ipynb` — full training, evaluation, and comparison notebook
- `app.py` — Streamlit web application for live classification

## How to Run the App
```bash
pip install streamlit tensorflow pillow numpy
streamlit run app.py
```
Note: requires the trained EfficientNetB3 model file (`.keras`), hosted separately due to file size.

## Results
- EfficientNetB3 achieved the highest test set accuracy and also generalized 
  best on real-world (out-of-dataset) images, making it the selected model 
  for the final recycling recommendation agent.
- Real-world testing on 6 new images showed 83.33% accuracy, with the main 
  confusion being between visually similar paper and trash items.

## Tech Stack
Python, TensorFlow/Keras, Streamlit, scikit-learn, Google Colab
