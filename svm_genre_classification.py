# --- Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Dataset ---
df = pd.read_csv('features_3_sec.csv')
df = df.dropna()

X = df.drop(['filename', 'label', 'length'], axis=1)
y_text = df['label']
filenames = df['filename']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

# --- Train-Test Splits ---
test_sizes = [0.4, 0.3, 0.2]
for test_size in test_sizes:
    print(f"\n{'='*20} SVM | Split: {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test {'='*20}")

    unique_files = df['filename'].unique()
    file_labels = df.groupby('filename')['label'].first()
    train_files, test_files = train_test_split(unique_files, test_size=test_size, random_state=42, stratify=file_labels)

    train_indices = df[df['filename'].isin(train_files)].index
    test_indices = df[df['filename'].isin(test_files)].index

    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA applied. Features reduced from {X_train.shape[1]} to {X_train_pca.shape[1]}.")

    model = SVC(random_state=42)
    model.fit(X_train_pca, y_train)

    segment_preds = model.predict(X_test_pca)
    results_df = pd.DataFrame({'filename': filenames.loc[test_indices], 'prediction': segment_preds})
    final_preds = results_df.groupby('filename')['prediction'].apply(lambda x: mode(x)[0])

    true_labels_df = df.loc[test_indices].groupby('filename')['label'].first()
    true_labels_encoded = label_encoder.transform(true_labels_df)
    aligned_preds = final_preds.loc[true_labels_df.index].values

    accuracy = accuracy_score(true_labels_encoded, aligned_preds)
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(classification_report(true_labels_encoded, aligned_preds, target_names=label_encoder.classes_))

    cm = confusion_matrix(true_labels_encoded, aligned_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'SVM Confusion Matrix ({int((1-test_size)*100)}/{int(test_size*100)} Split)')
    plt.ylabel('Actual Genre')
    plt.xlabel('Predicted Genre')
    plt.show()
