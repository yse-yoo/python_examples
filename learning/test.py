import pickle
import cv2
import matplotlib.pyplot as plt

def predict_image(model, image_path):
    # 画像の読み込みと前処理
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: '{image_path}' could not be loaded. Check the path and try again.")
        return

    resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)

    # 予測
    prediction = model.predict(resized)
    probability = model.predict_proba(resized)[0][int(prediction)]
    
    label = 'Positive' if prediction == 1 else 'Negative'
    print(f'Prediction: {label}, Probability: {probability:.2f}')

def show_prediction(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: '{image_path}' could not be loaded. Check the path and try again.")
        return

    resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)

    prediction = model.predict(resized)
    label = 'Positive' if prediction == 1 else 'Negative'

    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {label}')
    plt.axis('off')
    plt.show()

# モデルの読み込み
with open('svm_image_classifier.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 標準入力から画像ファイル名を取得
image_name = input("Enter the path to the image file: ").strip()
image_path = "./data/test/" + image_name + ".jpg"

# 入力された画像で予測と表示を実行
predict_image(loaded_model, image_path)
show_prediction(image_path, loaded_model)
