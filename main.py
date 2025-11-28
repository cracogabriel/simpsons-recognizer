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

# configurações de tamanho, pca e geração aleatória.
TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"
IMG_SIZE = (128, 128)
RANDOM_STATE = 42
PCA_VARIANCE = 0.95

# parâmetros para extração de características hog.
HOG_PIXELS = (16, 16)
HOG_CELLS = (2, 2)
HOG_ORIENT = 9

# parâmetros para extração de características lbp.
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"
LBP_BINS = 59

# parâmetros para histograma de cores no espaço hsv.
HSV_BINS = (8, 8, 8)


def extract_label(fname: str) -> str:
  """extrai o rótulo do nome do arquivo, removendo números e extensão."""
  base = os.path.basename(fname).lower()
  m = re.match(r"([a-z]+)", base)
  if m:
    return m.group(1)
  return base.split(".")[0]


def load_images_and_labels(folder: str, size=(128, 128)):
  """carrega imagens realizando crop quadrado central e redimensionamento."""
  imgs = []
  labels = []
  fnames = sorted(os.listdir(folder))
  for fname in fnames:
    if not fname.lower().endswith(".bmp"):
      continue
    path = os.path.join(folder, fname)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
      print(f"[WARN] couldn't read {path}, skipping")
      continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # crop quadrado central para evitar distorções no redimensionamento.
    min_side = min(h, w)
    y0 = (h - min_side) // 2
    x0 = (w - min_side) // 2
    crop = img[y0:y0 + min_side, x0:x0 + min_side]
    resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    imgs.append(resized)
    labels.append(extract_label(fname))
  return np.array(imgs), np.array(labels)


def extract_hog_features(images):
  """extrai características hog convertendo para escala de cinza."""
  hog_feats = []
  for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = hog(
      gray,
      orientations=HOG_ORIENT,
      pixels_per_cell=HOG_PIXELS,
      cells_per_block=HOG_CELLS,
      block_norm="L2-Hys",
      feature_vector=True,
    )
    hog_feats.append(f)
  return np.array(hog_feats)


def extract_lbp_features(images):
  """extrai histograma normalizado de padrões binários locais."""
  lbp_feats = []
  for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS))
    hist = hist.astype(float)
    if hist.sum() > 0:
      hist /= hist.sum()
    lbp_feats.append(hist)
  return np.array(lbp_feats)


def extract_color_hist_features(images):
  """extrai histograma 3d normalizado no espaço de cores hsv."""
  feats = []
  for img in images:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS, [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(float)
    if hist.sum() > 0:
      hist /= hist.sum()
    feats.append(hist)
  return np.array(feats)


def build_classifiers():
  """constrói 20 classificadores diversos para garantir variedade no ensemble."""
  clfs = []

  clfs.append(("knn_1", KNeighborsClassifier(n_neighbors=1)))
  clfs.append(("knn_3", KNeighborsClassifier(n_neighbors=3)))
  clfs.append(("knn_5", KNeighborsClassifier(n_neighbors=5)))
  clfs.append(("knn_7", KNeighborsClassifier(n_neighbors=7)))

  clfs.append(("dt_depth5", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)))
  clfs.append(("dt_depth10", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)))
  clfs.append(("dt_full", DecisionTreeClassifier(random_state=RANDOM_STATE)))

  clfs.append(("svm_linear_C1", SVC(kernel="linear", C=1.0, probability=False, random_state=RANDOM_STATE)))
  clfs.append(("svm_rbf_C1", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=RANDOM_STATE)))
  clfs.append(("svm_rbf_C10", SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=RANDOM_STATE)))
  clfs.append(("svm_poly_C1", SVC(kernel="poly", degree=2, C=1.0, probability=False, random_state=RANDOM_STATE)))

  clfs.append(("rf_50", RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)))
  clfs.append(("rf_100", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)))
  clfs.append(("rf_200", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)))

  clfs.append(("mlp_64", MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, early_stopping=True, random_state=RANDOM_STATE)))
  clfs.append(("mlp_128", MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, early_stopping=True, random_state=RANDOM_STATE)))
  clfs.append(("mlp_256", MLPClassifier(hidden_layer_sizes=(256,), max_iter=1500, early_stopping=True, random_state=RANDOM_STATE)))

  clfs.append(("knn_9", KNeighborsClassifier(n_neighbors=9)))
  clfs.append(("rf_300", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)))
  clfs.append(("dt_depth20", DecisionTreeClassifier(max_depth=20, random_state=RANDOM_STATE)))

  if len(clfs) < 20:
    i = 0
    while len(clfs) < 20:
      clfs.append((f"extra_svm_{i}", SVC(kernel="rbf", C=1.0 + i, gamma="scale", probability=False, random_state=RANDOM_STATE)))
      i += 1

  return clfs[:20]


def majority_vote(predictions):
  """combina predições de múltiplos classificadores por voto majoritário."""
  preds = np.array(predictions)
  n_models, n_samples = preds.shape
  final = []
  for j in range(n_samples):
    col = preds[:, j]
    cnt = Counter(col)
    most_common = cnt.most_common()
    final.append(most_common[0][0])
  return np.array(final)


def main():
  warnings.filterwarnings("ignore", category=FutureWarning)
  warnings.filterwarnings("ignore", category=UserWarning)

  print("Carregando TRAIN...")
  X_train_img, y_train = load_images_and_labels(TRAIN_DIR, size=IMG_SIZE)

  print("Carregando VALID...")
  X_valid_img, y_valid = load_images_and_labels(VALID_DIR, size=IMG_SIZE)

  print(f"Total treino: {len(X_train_img)} imagens")
  print(f"Total valid: {len(X_valid_img)} imagens")
  if len(X_train_img) == 0:
    raise SystemExit("Nenhuma imagem no TRAIN (verifique o caminho).")
  if len(X_valid_img) == 0:
    print("[WARN] nenhuma imagem no VALID - nada será avaliado no final.")

  # codifica labels apenas com base no conjunto de treino.
  le = LabelEncoder()
  y_train_enc = le.fit_transform(y_train)

  # remove exemplos de validação com classes não vistas no treino.
  valid_mask = np.isin(y_valid, le.classes_)
  if not np.all(valid_mask):
    unseen = np.unique(y_valid[~valid_mask])
    print(f"[WARN] valid contém labels não vistas no train: {unseen}. Esses exemplos serão ignorados.")
    X_valid_img = X_valid_img[valid_mask]
    y_valid = y_valid[valid_mask]
  y_valid_enc = le.transform(y_valid) if len(y_valid) > 0 else np.array([])

  print("\nExtraindo HOG (train)...")
  X_hog_train = extract_hog_features(X_train_img)
  print("Extraindo LBP (train)...")
  X_lbp_train = extract_lbp_features(X_train_img)
  print("Extraindo ColorHist (train)...")
  X_col_train = extract_color_hist_features(X_train_img)

  print("\nExtraindo HOG (valid)...")
  X_hog_valid = extract_hog_features(X_valid_img) if len(X_valid_img) > 0 else np.zeros((0, X_hog_train.shape[1]))
  print("Extraindo LBP (valid)...")
  X_lbp_valid = extract_lbp_features(X_valid_img) if len(X_valid_img) > 0 else np.zeros((0, X_lbp_train.shape[1] if 'X_lbp_train' in locals() else LBP_BINS))
  print("Extraindo ColorHist (valid)...")
  X_col_valid = extract_color_hist_features(X_valid_img) if len(X_valid_img) > 0 else np.zeros((0, np.prod(HSV_BINS)))

  # concatena todas as características extraídas.
  X_train_concat = np.hstack([X_hog_train, X_lbp_train, X_col_train])
  X_valid_concat = np.hstack([X_hog_valid, X_lbp_valid, X_col_valid]) if len(X_valid_img) > 0 else np.zeros((0, X_train_concat.shape[1]))

  # padroniza características antes da pca para melhorar desempenho.
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train_concat)
  X_valid_scaled = scaler.transform(X_valid_concat) if len(X_valid_concat) > 0 else np.zeros((0, X_train_scaled.shape[1]))

  # reduz dimensionalidade mantendo 95% da variância para melhorar eficiência.
  print("\nAplicando PCA (retendo {:.0%} variância)...".format(PCA_VARIANCE))
  pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=RANDOM_STATE)
  X_train_pca = pca.fit_transform(X_train_scaled)
  X_valid_pca = pca.transform(X_valid_scaled) if len(X_valid_scaled) > 0 else np.zeros((0, X_train_pca.shape[1]))
  print("Dimensão após PCA:", X_train_pca.shape)

  clfs = build_classifiers()
  print("\nForam construídos {} classificadores.".format(len(clfs)))

  # valida com 10 folds estratificados para garantir distribuição balanceada de classes.
  print("\n=== Stratified 10-fold CV (TRAIN only) ===")
  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
  fold_scores = []

  for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_train_pca, y_train_enc), 1):
    print(f"\n--- Fold {fold_idx} ---")
    Xtr, Xte = X_train_pca[tr_idx], X_train_pca[te_idx]
    ytr, yte = y_train_enc[tr_idx], y_train_enc[te_idx]

    fold_preds = []
    for name, clf in clfs:
      model = clf
      model.fit(Xtr, ytr)
      pred = model.predict(Xte)
      fold_preds.append(pred)

    # combina predições dos classificadores por voto majoritário.
    fold_pred_major = majority_vote(fold_preds)
    acc = accuracy_score(yte, fold_pred_major)
    print("Fold accuracy (majority vote): {:.4f}".format(acc))
    fold_scores.append(acc)

  print("\nCV mean accuracy: {:.4f}, std: {:.4f}".format(np.mean(fold_scores), np.std(fold_scores)))

  # treina modelos finais com todo conjunto de treino para avaliação final.
  print("\nTraining final classifiers on full TRAIN...")
  final_preds = []
  trained_models = []
  for name, clf in clfs:
    model = clf
    model.fit(X_train_pca, y_train_enc)
    trained_models.append((name, model))
    if len(X_valid_pca) > 0:
      final_preds.append(model.predict(X_valid_pca))

  if len(final_preds) == 0:
    print("[WARN] Sem exemplos de VALID para avaliar.")
    return

  y_valid_major = majority_vote(final_preds)
  final_acc = accuracy_score(y_valid_enc, y_valid_major)
  print("\n=== Final evaluation on VALID ===")
  print("Accuracy (majority vote over {} classifiers): {:.4f}".format(len(final_preds), final_acc))
  print("\nClassification report (VALID):")
  print(classification_report(y_valid_enc, y_valid_major, target_names=le.classes_))

  print("\nPer-classifier accuracy on VALID (optional):")
  for (name, model), preds in zip(trained_models, final_preds):
    a = accuracy_score(y_valid_enc, preds)
    print(f" {name:15s} -> {a:.4f}")

  print("\nDONE.")

if __name__ == "__main__":
  main()
