# Weather Image Classification
 
Classifying photos into **11 weather types** using unsupervised clustering, probabilistic classifiers, a deep neural network, and a convolutional autoencoder with learned embeddings. The project compares how each approach handles noisy, real-world, unconstrained images.
 
## Dataset
 
[Weather Image Recognition (Kaggle)](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/data): 6,862 images across 11 classes: dew, fogsmog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, snow. Classes are imbalanced (rime has 1,160 images, rainbow has 232).
 
## Approach
 
1. **Preprocessing:** built an image-path/label dataframe, made a stratified 70/30 split (4,803 train / 2,059 test), resized every image to 100x100 RGB, scaled pixels to [0, 1], and label-encoded classes.
2. **K-Means (k=22) + PCA:** clustered flattened 30,000-dimensional pixel vectors, visualized with PCA, and inspected cluster composition and silhouette score.
3. **Naive Bayes baselines:** Gaussian NB on raw pixels, on k-means centroid images only, and on PCA components.
4. **Deep neural network (Keras):** Flatten, Dense(128, ReLU), Dense(512, tanh), Dense(11, softmax) with Adam and sparse categorical cross-entropy, trained 10 epochs (about 3.9M parameters).
5. **Convolutional autoencoder:** Conv/MaxPool encoder to a 64-dimension latent vector, mirrored decoder, MSE loss, 15 epochs. The latent embeddings were then used for k-means and for a logistic regression classifier.
## Results
 
| Method | Test accuracy |
|---|---|
| Gaussian NB, centroid images only (22 clusters) | 12% |
| Gaussian NB, PCA (2 components) | 32% |
| Gaussian NB, raw pixels | 48% |
| Dense neural network | 50.4% |
| **Conv. autoencoder embeddings + logistic regression** | **54.7%** |
 
Chance level for 11 classes is roughly 9%.
 
- Raw-pixel k-means clusters overlap heavily (silhouette 0.051). Autoencoder embeddings were only slightly better (silhouette 0.099, adjusted Rand index 0.094, NMI 0.205).
- The convolutional features beat the fully connected network, supporting the value of spatial structure for images.
- Accuracy is modest. The images are unconstrained (varied scenes and backgrounds), and several classes are visually similar (for example frost, rime, and glaze).

## Tech Stack
 
Python, TensorFlow/Keras, scikit-learn, NumPy, pandas, PIL, matplotlib, seaborn
 
## Running It
 
```bash
pip install tensorflow scikit-learn pandas numpy pillow matplotlib seaborn
# download the Kaggle dataset and set data_dir in the notebook to the folder containing the class subfolders
jupyter notebook proj2.ipynb
```
 
> Note: `data_dir` in the notebook is a local Windows path. Update it before running.
 
