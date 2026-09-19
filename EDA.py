import json
from collections import Counter


def count_instances(coco_path):
    with open(coco_path) as f:
        data = json.load(f)

    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    counts = Counter(ann['category_id'] for ann in data['annotations'])

    print(f"\n{coco_path}")
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"Total images: {len(data['images'])}")
    for cat_id, count in counts.most_common():
        print(f"  {id_to_name[cat_id]}: {count}")

count_instances('data/images_rgb_train/coco.json')
count_instances('data/images_rgb_val/coco.json')
count_instances('data/images_thermal_train/coco.json')
count_instances('data/images_thermal_val/coco.json')