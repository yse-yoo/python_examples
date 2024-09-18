import cv2 as cv
import os

# 現在のフォルダ
current_dir = os.path.dirname(os.path.abspath(__file__))

# モデルファイルパス (適切なパスに変更)
model_path = os.path.join(current_dir, "models/haarcascade_frontalface_alt.xml")

# カスケード分類器の読み込み
face_cascade = cv.CascadeClassifier(model_path)

# カメラからの入力を開始
cap = cv.VideoCapture(0)

if not face_cascade.load(model_path):
    print("カスケードファイルが読み込めませんでした")
    exit()

while True:
    # フレームを読み込む
    ret, frame = cap.read()

    # フレームが正常に読み込まれなかった場合はスキップ
    if not ret:
        continue

    # グレースケールに変換
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 顔を検出
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 検出した顔を処理
    for (x, y, w, h) in faces:
        RATE = 1.2
        # 検出された顔のサイズを拡大
        enlarged_w = int(w * RATE)
        enlarged_h = int(h * RATE)

        # 顔の位置を調整
        x = max(0, x - (enlarged_w - w) // 2)
        y = max(0, y - (enlarged_h - h) // 2)

        # 顔の位置と拡大されたサイズがフレームの範囲内に収まるように調整
        end_x = min(x + enlarged_w, frame.shape[1])
        end_y = min(y + enlarged_h, frame.shape[0])
        start_x = max(0, end_x - enlarged_w)
        start_y = max(0, end_y - enlarged_h)

        # 検出した顔の周りに矩形を描画
        cv.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    # 結果を表示
    cv.imshow('Face Replacement', frame)

    # escキーで終了
    if cv.waitKey(20) & 0xFF == 27:
        break

# リソースを解放
cap.release()
cv.destroyAllWindows()