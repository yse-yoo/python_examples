import os
import cv2
import pickle

# scikit-learnモデルの読み込み
with open('svm_image_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# 動画ファイルのパスを指定
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "videos", "crow.mp4")
cap = cv2.VideoCapture(video_path)

try:
    while cap.isOpened():  # 動画が正常に読み込まれている間
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read the frame. Exiting...")
            break

        # グレースケール変換とリサイズ
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)).flatten().reshape(1, -1)

        # 予測の実行
        prediction = model.predict(resized)
        label = 'Positive' if prediction == 1 else 'Negative'

        # 結果を表示
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video Prediction', frame)

        # ウィンドウが閉じられた場合の確認
        if cv2.getWindowProperty('Video Prediction', cv2.WND_PROP_VISIBLE) < 1:
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
