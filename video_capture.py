import cv2 as cv
import os

MAX_SIZE = 1000
def resizeFrame(frame):
    scale = 1.0
    width = frame.shape[1]
    height = frame.shape[0]
    
    if ((width > height) and width > MAX_SIZE):
        scale = MAX_SIZE / width
    elif ((height > width) and height > MAX_SIZE):
        scale = MAX_SIZE / height

    width = int(width * scale)
    height = int(height * scale)

    resize = (width, height)
    return cv.resize(frame, resize, interpolation=cv.INTER_AREA)

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "videos", "python_capture_video.mp4")
capture = cv.VideoCapture(image_path)

if capture is None:
    print("ファイルを読み込めませんでした。")
else:
    while True:
        isTrue, frame = capture.read()
        cv.imshow("Video", resizeFrame(frame))
        if cv.waitKey(20) & 0xFF == 27:
            break

capture.release()
cv.destroyAllWindows()