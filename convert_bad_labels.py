from pathlib import Path
from tqdm import tqdm

labels_dir = Path("/home/artem/datasets/tatarstan/train/labels")
save_dir = Path("/home/artem/datasets/tatarstan/train/new_labels")
save_dir.mkdir(exist_ok=True, parents=True)

classes_map = {"car": 0, "head": 1, "face": 2, "human": 3, "carplate": 4}

for label_file in tqdm(list(labels_dir.glob("*"))):
    s = label_file.stem.rfind("_")
    class_name = str(label_file.stem)[s + 1 : -1]
    new_file_name = str(label_file.stem)[:s] + ".txt"
    new_label_file = save_dir.joinpath(new_file_name)
    with new_label_file.open(mode="a") as nf:
        with label_file.open(mode="r") as of:
            labels = of.readlines()
            for label in labels:
                new_class_id = classes_map.get(class_name)
                new_label = str(new_class_id) + " " + " ".join(label.split()[1:]) + "\n"
                nf.write(new_label)
