# Music Genre Classification: A Comparative Study (GTZAN)

## 🎯 Project Overview
This project implements a complete pipeline for automated Music Genre Classification (MGC). The goal was to compare the performance of classic machine learning models on a high-dimensional audio feature set, with a focus on rigorous data preprocessing.

The final pipeline uses the **Support Vector Machine (SVM)** model, achieving a song-level accuracy of **82.38%** on the GTZAN test set.

## ⚙️ Methodology & Technologies
The project follows a best-practice machine learning workflow, focusing on preventing common errors in time-series (audio) data.

### Key Preprocessing Techniques:
1.  **Data Leakage Prevention:** Implemented a **song-based Train-Test Split** to ensure no 3-second segments from the same song appeared in both sets, guaranteeing a fair evaluation.
2.  **Dimensionality Reduction:** Used **Principal Component Analysis (PCA)** (`n_components=0.95`) to reduce the initial **57 features** to **39 components**, which increases training speed and reduces noise.
3.  **Standardization:** Applied `StandardScaler` to ensure the features (MFCCs, Chroma, etc.) are all on a similar scale, which is critical for the SVM model's performance.

### Technologies Used:
* **Python 3.x**
* **Data Science:** Pandas, NumPy, Scikit-learn
* **Audio Processing:** Librosa
* **Modeling:** Support Vector Machine (SVC)
* **Serialization:** Joblib (for saving the model)

## 📁 Repository Structure
. ├── main_classifier.py # The main Python script with the full pipeline. ├── README.md # This file. ├── gtzan_model.pkl # [Generated after running the script] The trained SVM model. ├── gtzan_scaler.pkl # [Generated after running the script] The fitted StandardScaler object. └── features_3_sec.csv # REQUIRED: The GTZAN dataset file.


## 🚀 How to Run the Project

### Prerequisites:
1.  **Install Libraries:**
    ```bash
    pip install pandas numpy scikit-learn xgboost librosa joblib seaborn matplotlib
    ```
    *(Note: The full GTZAN audio dataset is not required; only the pre-extracted `features_3_sec.csv` file is needed.)*

### Execution:
1.  **Download:** Place `main_classifier.py` and `features_3_sec.csv` into the same folder.
2.  **Execute:** Run the script from your terminal:
    ```bash
    python main_classifier.py
    ```

### Output:
The script will output the following:
1.  The final **Accuracy Score** (should be approximately **82.38%**).
2.  A detailed **Classification Report** showing precision, recall, and F1-scores for all 10 genres.
3.  A visualization of the **Confusion Matrix**.
