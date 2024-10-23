import os
import cv2
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print(os.getcwd())

# 画像フォルダのパス
positive_path = './data/positives/'
negative_path = './data/negatives/'

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, (64, 64))  # サイズ変更
            images.append(resized.flatten())  # 1次元配列に変換
    return np.array(images)

# ポジティブとネガティブの画像を読み込み
X_pos = load_images_from_folder(positive_path)
X_neg = load_images_from_folder(negative_path)

# ラベルの設定（ポジティブ: 1、ネガティブ: 0）
y_pos = np.ones(len(X_pos))
y_neg = np.zeros(len(X_neg))

# データとラベルの結合
X = np.vstack((X_pos, X_neg))
y = np.hstack((y_pos, y_neg))

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# SVMモデルの作成
model = SVC(kernel='linear', probability=True)

# モデルの訓練
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# 精度の確認
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# モデルの保存
with open('svm_image_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

