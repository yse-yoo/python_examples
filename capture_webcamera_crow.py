import cv2
import pickle

# scikit-learnモデルの読み込み
with open('svm_image_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# カメラの映像を取得
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        # グレースケール変換とリサイズ
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)).flatten().reshape(1, -1)

        # 予測の実行
        prediction = model.predict(resized)
        label = 'Positive' if prediction == 1 else 'Negative'

        # 結果を表示
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Prediction', frame)

        # ウィンドウが閉じられた場合の確認
        if cv2.getWindowProperty('Real-time Prediction', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting...")
            break

        # ESCキーで終了
        if cv2.waitKey(1) == 27:  # 27はESCキーのASCIIコード
            print("ESC pressed. Exiting...")
            break

except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Exiting gracefully...")

finally:
    # カメラとウィンドウの解放
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released successfully.")
