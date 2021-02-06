# Autogen with
with open("../datasets/lvis/lvis_v1_train.json", "r") as f:
    a = json.load(f)
c = a["categories"]
all_image_count = len(set([_["image_id"] for _ in a["annotations"]]))
for x in c:
    x["image_count_frequency"] = x["image_count"] / all_image_count
LVIS_CATEGORIES = repr(c) + "  # noqa"

NUM_CLASSES = 1230


def get_image_count_frequency():
    image_count_frequency = [None] * NUM_CLASSES
    for c in LVIS_CATEGORIES:
        category_id = c["id"] - 1  # conver to 0-based index
        image_count_frequency[category_id] = c["image_count_frequency"]
    return image_count_frequency
