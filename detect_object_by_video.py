import numpy as np
import cv2 as cv
import os

prob_threshold = 0.35
iou_threshold = 0.6

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install the latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from nanodet import NanoDet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()
    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv.BORDER_CONSTANT, value=0)
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
    else:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)
    letterbox_scale = [top, left, newh, neww]
    return img, letterbox_scale

def vis(preds, res_img, letterbox_scale, fps=None):
    ret = res_img.copy()

    # ラベルを左上に表示
    label_y_position = 50  # 最初のラベルのY位置
    label_spacing = 30     # ラベル間のスペース

    for pred in preds:
        conf = pred[-2]
        if conf >= 0.6:  # 60%以上の場合のみラベルを表示
            classid = pred[-1].astype(np.int32)
            label = "{:s}: {:.2f}".format(classes[classid], conf)
            
            # 左上に順番にラベルを表示
            cv.putText(ret, label, (18, label_y_position), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
            label_y_position += label_spacing  # 次のラベルのY位置を更新

    return ret


if __name__ == '__main__':
    video_path = 'videos/ng2.mp4'
    model_path = 'object_detection_nanodet_2022nov.onnx'

    backend_id = cv.dnn.DNN_BACKEND_OPENCV  # CPUを使用
    target_id = cv.dnn.DNN_TARGET_CPU

    model = NanoDet(modelPath=model_path,
                    prob_threshold=prob_threshold,
                    iou_threshold=iou_threshold,
                    backend_id=backend_id,
                    target_id=target_id)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        exit(1)

    tm = cv.TickMeter()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 前処理
        input_blob, letterbox_scale = letterbox(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        # 推論
        tm.start()
        preds = model.infer(input_blob)
        tm.stop()

        # 結果の可視化
        img = vis(preds, frame, letterbox_scale, fps=tm.getFPS())
        tm.reset()

        # 表示
        cv.imshow("Video Detection", img)

        # q キーで終了
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
