import cv2 as cv

# Webカメラをキャプチャ (デフォルトのカメラはインデックス0)
capture = cv.VideoCapture(0)

# カメラが開かれているか確認
if not capture.isOpened():
    print("カメラが開けませんでした。")
    exit()

# Webカメラから映像をキャプチャして表示
while True:
    isTrue, frame = capture.read()

    # フレームが正しく取得できたか確認
    if not isTrue:
        print("フレームの読み込みに失敗しました。")
        break

    # フレームを表示
    cv.imshow("Webcam", frame)

    # 'Esc'キーで終了 (ASCIIコード27)
    if cv.waitKey(1) & 0xFF == 27:
        break

# リソースを解放してウィンドウを閉じる
capture.release()
cv.destroyAllWindows()