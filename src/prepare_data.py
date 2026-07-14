import os
import glob
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw", "Panoramic radiographs with periapical lesions Dataset", "Periapical Dataset", "Periapical Lesions")
    
    img_dir = os.path.join(raw_dir, "Original JPG Images")
    ann_dir = os.path.join(raw_dir, "Image Annots")
    
    if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
        print(f"Error: Could not find data at {raw_dir}")
        print("Please ensure you extracted the Mendeley zip into data/raw/")
        return

    # Setup output directories
    out_dir = os.path.join(base_dir, "data", "yolo_dataset")
    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    xml_files = glob.glob(os.path.join(ann_dir, "*.xml"))
    print(f"Found {len(xml_files)} annotation files.")

    # We use the stem of the XML to find the image, ignoring the <filename> tag 
    # since it is incorrect in this dataset.
    valid_pairs = []
    for xml_file in xml_files:
        stem = os.path.splitext(os.path.basename(xml_file))[0]
        img_path = os.path.join(img_dir, f"{stem}.jpg")
        if os.path.exists(img_path):
            valid_pairs.append((xml_file, img_path))
    
    print(f"Found {len(valid_pairs)} matching image-annotation pairs.")
    
    # Train-val split
    train_pairs, val_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    
    def process_split(pairs, split_name):
        print(f"Processing {split_name} split...")
        for xml_file, img_file in tqdm(pairs):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            yolo_labels = []
            for obj in root.iter('object'):
                # We map all lesions to class 0
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_to_yolo((w, h), b)
                yolo_labels.append(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
                
            if yolo_labels:
                # Copy image
                stem = os.path.splitext(os.path.basename(img_file))[0]
                dst_img = os.path.join(out_dir, "images", split_name, f"{stem}.jpg")
                shutil.copy2(img_file, dst_img)
                
                # Write label
                dst_lbl = os.path.join(out_dir, "labels", split_name, f"{stem}.txt")
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(yolo_labels) + '\n')

    process_split(train_pairs, 'train')
    process_split(val_pairs, 'val')
    
    # Create data.yaml
    yaml_content = f"""path: {out_dir}
train: images/train
val: images/val

names:
  0: Periapical Lesion
"""
    yaml_path = os.path.join(base_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"Data preparation complete. data.yaml created at {yaml_path}")

if __name__ == "__main__":
    main()
