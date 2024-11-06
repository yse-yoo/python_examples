import numpy as np
import cv2 as cv
import os

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install the latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from nanodet import NanoDet

# 固定のディレクトリパスとモデルパス
input_dir = "learning/data/test/"
output_dir = "learning/data/output/"
modelPath = "object_detection_nanodet_2022nov.onnx"

# 有効な画像拡張子
valid_extensions = ('.jpg', '.jpeg', '.png')

# 有効なクラスラベル
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

def vis(preds, res_img, letterbox_scale):
    ret = res_img.copy()
    for pred in preds:
        bbox = pred[:4]
        conf = pred[-2]
        classid = pred[-1].astype(np.int32)
        xmin, ymin, xmax, ymax = bbox.astype(int)
        cv.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
        label = "{:s}: {:.2f}".format(classes[classid], conf)
        cv.putText(ret, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    return ret

# モデルの初期化
backend_id = cv.dnn.DNN_BACKEND_OPENCV  # CPUを使用
target_id = cv.dnn.DNN_TARGET_CPU
prob_threshold = 0.35
iou_threshold = 0.6

model = NanoDet(modelPath=modelPath,
                prob_threshold=prob_threshold,
                iou_threshold=iou_threshold,
                backend_id=backend_id,
                target_id=target_id)

# 出力ディレクトリの作成
os.makedirs(output_dir, exist_ok=True)

# すべての画像ファイルを処理
for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_extensions):
        image_path = os.path.join(input_dir, filename)
        image = cv.imread(image_path)
        if image is None:
            print(f"Could not load image {image_path}")
            continue

        # 前処理
        input_blob, letterbox_scale = letterbox(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # 推論実行
        preds = model.infer(input_blob)

        # 結果の可視化
        img = vis(preds, image, letterbox_scale)
        
        # 結果の保存
        output_path = os.path.join(output_dir, f"result_{filename}")
        cv.imwrite(output_path, img)
        print(f'Results saved to {output_path}')
