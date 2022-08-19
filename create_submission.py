from pathlib import Path
import csv


labels_dir = Path("yolov5/runs/detect/ensemble/labels")
save_path = Path("submission.csv")


def yolobbox2bbox(x, y, w, h):
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)

    xmin, ymin = x - w / 2, y - h / 2
    xmax, ymax = x + w / 2, y + h / 2
    return xmin, xmax, ymin, ymax


fieldnames = ["ImageID", "LabelName", "Conf", "XMin", "XMax", "YMin", "YMax"]
with save_path.open("w") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()

    for labels_file in labels_dir.glob("*"):
        with labels_file.open(mode="r") as f:
            labels = f.read().splitlines()

            for label in labels:
                image_name = labels_file.stem + ".jpg"
                class_id, x, y, w, h, conf = label.split()
                xmin, xmax, ymin, ymax = yolobbox2bbox(x, y, w, h)

                writer.writerow(
                    {
                        "ImageID": image_name,
                        "LabelName": class_id,
                        "Conf": conf,
                        "XMin": xmin,
                        "XMax": xmax,
                        "YMin": ymin,
                        "YMax": ymax,
                    }
                )
                