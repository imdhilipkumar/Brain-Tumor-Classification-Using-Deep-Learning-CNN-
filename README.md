# 🧠 Brain Tumor Classification Using Deep Learning (CNN)

This project implements a **deep learning-based solution** to detect and classify brain tumors in MRI images using **image processing techniques and Convolutional Neural Networks (CNNs)**. Developed as part of my Master's thesis, this work focuses on improving early diagnosis and classification accuracy to support clinical decision-making.

---

## 📌 Project Highlights

- 🧠 **Domain**: Medical Imaging (Brain MRI)
- 🤖 **Techniques Used**: Image Processing, CNN, Data Augmentation
- 🧮 **Tools & Libraries**: Python, OpenCV, Keras, TensorFlow, NumPy, Matplotlib
- 📊 **Dataset**: Brain MRI dataset from [Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- 🎯 **Best Accuracy Achieved**: **91%** on validation data
- 📂 **Models Compared**: Two CNN architectures with and without Dropout layers
- 📈 **Key Metrics**: Accuracy, Loss, F1 Score

---

## 🚀 Tech Stack & Skills Demonstrated

- **Deep Learning**: CNN (Convolutional Neural Networks)
- **Image Processing**: Thresholding, Morphological Operations, Bilateral Filtering
- **Data Engineering**: Preprocessing, Data Augmentation, Cropping & Resizing
- **Model Evaluation**: Accuracy, F1 Score, Loss plots
- **Libraries**: `TensorFlow`, `Keras`, `OpenCV`, `NumPy`, `Matplotlib`, `scikit-learn`
- **Tools**: Google Colab, Jupyter Notebook

---


---


## 🔬 Methodology

1. **Data Collection**: MRI images categorized into 'Tumor' and 'No Tumor'
2. **Image Preprocessing**:
   - Grayscale conversion
   - Noise removal (bilateral filter)
   - Image enhancement (Gaussian blur)
   - Resizing to uniform shape
3. **Segmentation**:
   - Binary thresholding
   - Erosion and dilation
4. **CNN Model**:
   - Two different architectures implemented
   - Layers: Conv2D → MaxPooling → Flatten → Dense → Dropout
   - Optimizer: Adam | Loss: Binary Crossentropy | Activation: ReLU, Sigmoid
5. **Training**:
   - Data Augmentation applied
   - 70% Training, 15% Validation, 15% Testing
6. **Evaluation**:
   - Accuracy: **91%**
   - F1 Score: **0.88**
   - Loss curve analysis

---

## 📈 Results

| Model | Training Accuracy | Test Accuracy | F1 Score | Params |
|-------|-------------------|---------------|----------|--------|
| CNN-1 | 94%               | 85%           | 0.85     | 1.84M  |
| CNN-2 | 96%               | **91%**       | **0.88** | 1.86M  |

---

## 🔮 Future Scope

- Add more labeled data for multi-class tumor type detection
- Deploy as a web app for radiologists
- Explore other architectures like ResNet, VGG for improved results

---

## 👨‍💻 Author

**Dhilipkumar M**  
M.Tech (Software Engineering), VIT University  
[GitHub](https://github.com/imdhilipkumar) | [LinkedIn](https://www.linkedin.com/in/dhilipkumar20/)

---

## 📜 License

This project is for academic and educational purposes. Attribution required for commercial use.
