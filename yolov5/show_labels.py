import cv2
import os

# Should change
img_h = 900
img_w = 1200

# Все это нужно поменять с учетом расположения ваших файлов
images_path = "/home/artem/datasets/tatarstan/train/test/images"
labels_path = "/home/artem/datasets/tatarstan/train/test/labels"

bad_labeled_ds = "C:/Users/Pupil/transport/bad_labeled"
good_labeles_ds = "C:/Users/Pupil/transport/good_labeled"

WINDOW_NAME = "Amanda"

# Classes param
classes_list = ["car", "head", "face", "human", "carplate"]
colors_list = [
    (200, 100, 100),
    (250, 200, 200),
    (100, 100, 100),
    (0, 0, 255),
    (255, 0, 0),
    (100, 100, 250),
    (0, 150, 250),
    (0, 250, 0),
]

# draw param
fontScale = 1
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, img_w, img_h)
imgs_name = filter(lambda s: s.endswith(".jpg"), os.listdir(images_path))


class Replace2Good:
    def execute(self, img_name, lbl_name):
        try:
            # replace image
            old_path = os.path.join(images_path, img_name)
            new_path = os.path.join(good_labeles_ds, img_name)
            os.replace(old_path, new_path)
            # replace label
            old_path = os.path.join(labels_path, lbl_name)
            new_path = os.path.join(good_labeles_ds, lbl_name)
            os.replace(old_path, new_path)
        except:
            pass


class Replace2Bad:
    def execute(self, img_name, lbl_name):
        try:
            # replace image
            old_path = os.path.join(images_path, img_name)
            new_path = os.path.join(bad_labeled_ds, img_name)
            os.replace(old_path, new_path)
            # replace label
            old_path = os.path.join(labels_path, lbl_name)
            new_path = os.path.join(bad_labeled_ds, lbl_name)
            os.replace(old_path, new_path)
        except:
            pass


def yolo2pixels(line, img_shape):

    cls_index, x_center, y_center, width, height, _ = [float(x) for x in line.split()]
    img_h, img_w = img_shape[:2]
    x1 = int((x_center - (width / 2)) * img_w)
    x2 = int((x_center + (width / 2)) * img_w)
    y1 = int((y_center - (height / 2)) * img_h)
    y2 = int((y_center + (height / 2)) * img_h)
    return {
        "text": classes_list[int(cls_index)],
        "start_point": (x1, y1),
        "end_point": (x2, y2),
        "color": colors_list[int(cls_index)],
    }


def read_data(lbl_name, img_shape):
    lbl_path = os.path.join(labels_path, lbl_name)
    print(lbl_path)
    if not os.path.isfile(lbl_path):
        return None
    data = []
    with open(lbl_path, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            data.append(yolo2pixels(line, img_shape))
    return data


def draw_label(img, datas):
    for data in datas:
        start_point = data["start_point"]
        end_point = data["end_point"]
        color = data["color"]
        text = data["text"]
        img_h = img.shape[0]
        thichness = img_h // 400 if img_h > 400 else 1
        img = cv2.rectangle(img, start_point, end_point, color, thichness)
        img = cv2.putText(
            img, text, start_point, font, fontScale, color, thichness, cv2.LINE_AA
        )
    return img


for img_name in imgs_name:

    img = cv2.imread(os.path.join(images_path, img_name))

    img_shape = img.shape
    lbl_name = img_name[:-4] + ".txt"
    print(lbl_name)
    datas = read_data(lbl_name, img_shape)
    print(datas)
    if datas is not None:
        img = draw_label(img, datas)
    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
