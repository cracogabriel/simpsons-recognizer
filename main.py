import os
import re
import warnings
from collections import Counter

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# config
TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"
IMG_SIZE = (128, 128)
RANDOM_STATE = 42
PCA_VARIANCE = 0.90

# HOG params
HOG_PIXELS = (8, 8)
HOG_CELLS = (2, 2)
HOG_ORIENT = 9

# LBP params
LBP_P = 24
LBP_R = 3
LBP_BINS = LBP_P + 2

HSV_BINS = (8, 8, 8)


# ------------------------------
# Label extraction
# ------------------------------

def extract_label(fname: str) -> str:
    base = os.path.basename(fname).lower()
    m = re.match(r"([a-z]+)", base)
    if m:
        return m.group(1)
    return base.split(".")[0]


# ------------------------------
# Image loader with augmentations
# ------------------------------

def augment_image(img):
    aug = []
    aug.append(cv2.flip(img, 1))

    M = cv2.getRotationMatrix2D((64, 64), 8, 1.0)
    aug.append(cv2.warpAffine(img, M, (128, 128)))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
    aug.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

    return aug


def load_images_and_labels(folder: str, size=(128, 128), do_augment=False):
    imgs, labels = [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".bmp"):
            continue

        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Square crop
        h, w = img.shape[:2]
        s = min(h, w)
        img = img[(h - s) // 2:(h + s) // 2,
                  (w - s) // 2:(w + s) // 2]

        img = cv2.resize(img, size)

        label = extract_label(fname)

        imgs.append(img)
        labels.append(label)

        if do_augment:
            for a in augment_image(img):
                imgs.append(a)
                labels.append(label)

    return np.array(imgs), np.array(labels)


# ------------------------------
# Feature extractors
# ------------------------------

def extract_hog(images):
    feats = []
    for img in images:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feats.append(hog(
            g, orientations=HOG_ORIENT,
            pixels_per_cell=HOG_PIXELS,
            cells_per_block=HOG_CELLS,
            block_norm="L2-Hys",
            feature_vector=True
        ))
    return np.array(feats)


def extract_lbp(images):
    feats = []
    for img in images:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(g, LBP_P, LBP_R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS))
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)
    return np.array(feats)


# ---------- NEW: Dominant Colors ----------
def extract_dominant_colors(images, k=3):
    feats = []
    for img in images:
        data = img.reshape(-1, 3)
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        km.fit(data)
        centers = km.cluster_centers_.flatten() / 255.0  # normalize
        feats.append(centers)
    return np.array(feats)


# ---------- NEW: HSV MEAN ----------
def extract_color_mean(images):
    feats = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mean = hsv.mean(axis=(0, 1)) / 255.0
        feats.append(mean)
    return np.array(feats)


# ---------- Regular HSV histogram ----------
def extract_color_hist(images):
    feats = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            HSV_BINS,
            [0, 180, 0, 256, 0, 256]
        ).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)
    return np.array(feats)


# ------------------------------
# Models
# ------------------------------

def build_classifiers():
    clfs = []

    # kNN
    for k in [1, 3, 5, 7]:
        clfs.append(("knn_" + str(k), KNeighborsClassifier(n_neighbors=k)))

    # DecisionTree variations
    for d in [5, 10, 20]:
        clfs.append(("dt_" + str(d), DecisionTreeClassifier(max_depth=d)))

    # SVMs
    clfs.append(("svm_lin", SVC(kernel="linear", C=1)))
    clfs.append(("svm_rbf1", SVC(kernel="rbf", C=1)))
    clfs.append(("svm_rbf5", SVC(kernel="rbf", C=5)))
    clfs.append(("svm_rbf10", SVC(kernel="rbf", C=10)))
    clfs.append(("svm_best", SVC(kernel="rbf", C=8, gamma=0.01)))

    # MLP
    clfs.append(("mlp1", MLPClassifier(hidden_layer_sizes=(64,), max_iter=600)))
    clfs.append(("mlp2", MLPClassifier(hidden_layer_sizes=(128,), max_iter=600)))

    # RandomForest
    for n in [50, 100, 200]:
        clfs.append(("rf_" + str(n),
                     RandomForestClassifier(n_estimators=n)))

    # Ensure ≥ 20 models
    return clfs[:20]


def majority_vote(preds):
    preds = np.array(preds)
    final = []
    for i in range(preds.shape[1]):
        vote = Counter(preds[:, i]).most_common(1)[0][0]
        final.append(vote)
    return np.array(final)


# ------------------------------
# Main
# ------------------------------

def main():
    print("\nCarregando imagens...")
    X_train_img, y_train = load_images_and_labels(TRAIN_DIR, do_augment=True)
    X_valid_img, y_valid = load_images_and_labels(VALID_DIR, do_augment=False)

    print(f"Treino expandido para {len(X_train_img)} imagens")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_valid_enc = le.transform(y_valid)

    print("\nExtraindo features PRINCIPAIS (cores)...")
    f_dom_train = extract_dominant_colors(X_train_img)
    f_dom_valid = extract_dominant_colors(X_valid_img)

    f_mean_train = extract_color_mean(X_train_img)
    f_mean_valid = extract_color_mean(X_valid_img)

    f_hist_train = extract_color_hist(X_train_img)
    f_hist_valid = extract_color_hist(X_valid_img)

    print("Extraindo features auxiliares (texturas)...")
    f_hog_train = extract_hog(X_train_img)
    f_hog_valid = extract_hog(X_valid_img)

    f_lbp_train = extract_lbp(X_train_img)
    f_lbp_valid = extract_lbp(X_valid_img)

    # FEAT FINAL = cores (prioridade) + texturas
    X_train = np.hstack([
        f_dom_train * 2.0,      # reforça importância da cor
        f_mean_train * 2.0,
        f_hist_train * 1.2,
        f_hog_train,
        f_lbp_train
    ])

    X_valid = np.hstack([
        f_dom_valid * 2.0,
        f_mean_valid * 2.0,
        f_hist_valid * 1.2,
        f_hog_valid,
        f_lbp_valid
    ])

    print("\nPadronizando...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    print("Aplicando PCA...")
    pca = PCA(n_components=PCA_VARIANCE)
    X_train = pca.fit_transform(X_train)
    X_valid = pca.transform(X_valid)

    print("Balanceando com SMOTE...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train_enc = sm.fit_resample(X_train, y_train_enc)

    print("\nTreinando 20 classificadores...")
    clfs = build_classifiers()

    preds_valid = []
    for name, clf in clfs:
        clf.fit(X_train, y_train_enc)
        preds_valid.append(clf.predict(X_valid))

    final_pred = majority_vote(preds_valid)

    print("\n=== RESULTADOS ===")
    print("Acurácia:", accuracy_score(y_valid_enc, final_pred))
    print("\nRelatório:")
    print(classification_report(y_valid_enc, final_pred, target_names=le.classes_))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
