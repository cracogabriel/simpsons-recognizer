import os
import re
import warnings
from collections import Counter

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAW_TRAIN = "dataset/train"
RAW_VALID = "dataset/valid"

CROPPED_TRAIN = "cropped/train"
CROPPED_VALID = "cropped/valid"

IMG_SIZE = (128, 128)
RANDOM_STATE = 42
PCA_VARIANCE = 0.90

# Melhor HOG
HOG_PIXELS = (8, 8)
HOG_CELLS = (2, 2)
HOG_ORIENT = 9

# LBP
LBP_P = 24
LBP_R = 3
LBP_BINS = LBP_P + 2

HSV_BINS = (8, 8, 8)

# ============================================================
# DETECÇÃO DE FACE (SIMPOSNIZED SKIN DETECTOR)
# ============================================================

def crop_simpson_face(img):
    """
    Detecta face dos Simpsons via segmentação de pele amarela.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Pele amarela típica
    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morfologia (limpa ruídos)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    if w < 40 or h < 40:
        return None

    # Expande bounding box
    pad = int(0.20 * max(w, h))
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    return img[y:y+h, x:x+w]

# ============================================================
# CRIAÇÃO DO NOVO DATASET (CROPPED)
# ============================================================

def generate_cropped_dataset(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

    for fname in sorted(os.listdir(src_folder)):
        if not fname.lower().endswith(".bmp"):
            continue

        img_path = os.path.join(src_folder, fname)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cropped = crop_simpson_face(img)
        if cropped is None:
            continue

        out_path = os.path.join(dst_folder, fname)
        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, cropped_bgr)


# ============================================================
# EXTRAÇÃO DO RÓTULO
# ============================================================

def extract_label(fname: str):
    base = os.path.basename(fname).lower()
    m = re.match(r"([a-z]+)", base)
    if m:
        return m.group(1)
    return base.split(".")[0]


# ============================================================
# AUGMENTATION
# ============================================================

def augment_image(img):
    aug = []

    aug.append(cv2.flip(img, 1))

    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 10, 1.0)
    aug.append(cv2.warpAffine(img, M, IMG_SIZE))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.25, 0, 255)
    aug.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

    return aug


# ============================================================
# LOADING DAS IMAGENS
# ============================================================

def load_images_and_labels(folder, do_augment=False):
    imgs, labels = [], []

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".bmp"):
            continue

        path = os.path.join(folder, fname)
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)

        label = extract_label(fname)

        imgs.append(img)
        labels.append(label)

        if do_augment:
            for aug in augment_image(img):
                imgs.append(aug)
                labels.append(label)

    return np.array(imgs), np.array(labels)


# ============================================================
# FEATURES: HOG / LBP / COLOR
# ============================================================

def extract_hog(imgs):
    feats = []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feat = hog(gray,
                   orientations=HOG_ORIENT,
                   pixels_per_cell=HOG_PIXELS,
                   cells_per_block=HOG_CELLS,
                   block_norm="L2-Hys",
                   feature_vector=True)
        feats.append(feat)
    return np.array(feats)


def extract_lbp(imgs):
    feats = []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS))
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)
    return np.array(feats)


def extract_color(imgs):
    feats = []
    for img in imgs:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, HSV_BINS,
                            [0,180, 0,256, 0,256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)
    return np.array(feats)


# ============================================================
# CLASSIFICADORES
# ============================================================

def build_classifiers():
    return [
        ("knn_3", KNeighborsClassifier(3)),
        ("knn_5", KNeighborsClassifier(5)),
        ("svm_linear", SVC(kernel="linear", C=1)),
        ("svm_rbf", SVC(kernel="rbf", C=5, gamma=0.01)),
        ("rf_100", RandomForestClassifier(100)),
        ("dt_10", DecisionTreeClassifier(max_depth=10)),
    ]


def majority_vote(preds):
    preds = np.array(preds)
    final = []
    for i in range(preds.shape[1]):
        c = Counter(preds[:, i])
        final.append(c.most_common(1)[0][0])
    return np.array(final)


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")

    print("\n=== GERANDO NOVO DATASET (CROPPED) ===")

    generate_cropped_dataset(RAW_TRAIN, CROPPED_TRAIN)
    generate_cropped_dataset(RAW_VALID, CROPPED_VALID)

    print("Dataset de faces gerado com sucesso.\n")

    print("=== Carregando imagens ===")
    X_train_img, y_train = load_images_and_labels(CROPPED_TRAIN, do_augment=True)
    X_valid_img, y_valid = load_images_and_labels(CROPPED_VALID)

    print(f"Train size após augment: {len(X_train_img)}")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_valid_enc = le.transform(y_valid)

    print("\n=== Extraindo features ===")
    X_train = np.hstack([
        extract_color(X_train_img),
        extract_hog(X_train_img),
        extract_lbp(X_train_img)
    ])
    X_valid = np.hstack([
        extract_color(X_valid_img),
        extract_hog(X_valid_img),
        extract_lbp(X_valid_img)
    ])

    print("Escalonando...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    print("Aplicando PCA...")
    pca = PCA(n_components=PCA_VARIANCE)
    X_train = pca.fit_transform(X_train)
    X_valid = pca.transform(X_valid)

    print("Balanceando SMOTE...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train_enc = sm.fit_resample(X_train, y_train_enc)

    print("\n=== Treinando classificadores ===")
    preds = []
    clfs = build_classifiers()

    for name, clf in clfs:
        print("Treinando:", name)
        clf.fit(X_train, y_train_enc)
        preds.append(clf.predict(X_valid))

    final_pred = majority_vote(preds)

    print("\n=== RESULTADOS FINAIS ===")
    print("Acurácia:", accuracy_score(y_valid_enc, final_pred))
    print("\nClassification Report:")
    print(classification_report(y_valid_enc, final_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
