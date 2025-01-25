# Arabic Handwriting Recognition using CNNs and Transfer Learning

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Approach](#approach)
  - [Task 1: Custom CNN](#task-1-custom-cnn)
  - [Task 2: Data Augmentation](#task-2-data-augmentation)
  - [Task 3: AlexNet](#task-3-alexnet)
  - [Task 4: ResNet50 with Transfer Learning](#task-4-resnet50-with-transfer-learning)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## Introduction
This project focuses on Arabic handwriting recognition using Convolutional Neural Networks (CNNs) and transfer learning techniques. It involves building, training, and evaluating multiple architectures to classify Arabic handwriting images. The main objectives are:
1. To create a robust system capable of classifying handwriting from multiple users.
2. To explore data augmentation and advanced architectures for improving generalization.
3. To implement transfer learning using pre-trained ResNet50 for state-of-the-art performance.

The project consists of four major tasks:
1. Building a custom CNN model.
2. Utilizing data augmentation to improve performance.
3. Training AlexNet from scratch on the augmented dataset.
4. Fine-tuning a pre-trained ResNet50 model for transfer learning.

---


## Dataset
The dataset consists of grayscale images of Arabic handwriting, with each folder representing a unique user (or class). Images are resized and normalized before being fed into the models. The augmented dataset includes additional variations through random transformations, such as flipping and rotation, to improve model generalization.

Key features:
- Images are resized to 256x128 pixels for custom CNNs or 224x224 pixels for AlexNet and ResNet50.
- Data augmentation techniques include random horizontal flipping, rotation, and synthetic dataset expansion.

---

## Approach

### Task 1: Custom CNN
Two CNN architectures, `SimpleCNN` and `DeeperCNN`, were developed and trained. These models served as baseline architectures, with varying depths and complexity. Hyperparameter tuning was performed to optimize training.

### Task 2: Data Augmentation
Data augmentation techniques were applied to create a more diverse dataset. Transformations included:
- Random horizontal flipping
- Random rotations between -15° and +15°

These transformations improved model robustness and generalization.

### Task 3: AlexNet
AlexNet, a well-known CNN architecture, was implemented and trained from scratch. Adjustments included modifying the final fully connected layer to match the number of classes and employing a learning rate scheduler (`CosineAnnealingLR`) for better optimization.

### Task 4: ResNet50 with Transfer Learning
ResNet50, pre-trained on the ImageNet dataset, was fine-tuned for Arabic handwriting recognition. Its fully connected layer was replaced to match the number of classes. Transfer learning enabled faster convergence and significantly improved accuracy.

---

## Results

| Task                                | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|-------------------------------------|------------|----------------|-----------------|----------------------|
| Task 1: DeeperCNN (No Augmentation) | 1.0563     | 66.41%         | 1.4310          | 60.32%              |
| Task 2: DeeperCNN (With Augmentation) | 0.3784     | 87.19%         | 1.1345          | 70.36%              |
| Task 3: AlexNet                     | 0.0169     | 99.52%         | 1.0663          | 80.47%              |
| Task 4: ResNet50 (Transfer Learning) | 0.0018     | 100.00%        | 0.0177          | 99.53%              |

### Observations:
1. Data augmentation significantly improved the model’s ability to generalize to new data.
2. AlexNet showed better accuracy than DeeperCNN, demonstrating the importance of deeper architectures.
3. ResNet50 with transfer learning achieved the highest validation accuracy of 99.53%, showcasing the power of pre-trained models.

---

## Conclusion
The project demonstrated the effectiveness of CNNs and transfer learning for Arabic handwriting recognition. The custom CNNs provided a baseline, while AlexNet and ResNet50 achieved superior performance, especially with data augmentation and pre-trained weights. ResNet50 with transfer learning outperformed all models, achieving a validation accuracy of 99.53%.

---

## Future Work
To further enhance the robustness and scalability of the system, the following steps are proposed:
1. **Advanced Augmentation**: Use techniques like synthetic data generation to increase dataset diversity.
2. **Lightweight Architectures**: Experiment with models like MobileNet or EfficientNet for real-time applications.
3. **Dataset Expansion**: Include additional handwriting styles and users for improved generalization.
4. **Deployment**: Integrate the trained model into a mobile or web application for real-time handwriting recognition.

---

## Contributors

- [Mohammad AbuShams](https://github.com/MohammadAbuShams)
- [Joud Hijaz](https://github.com/JoudHijaz)
