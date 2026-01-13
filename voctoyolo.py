import os
import xml.etree.ElementTree as ET
import random
import shutil

# --- Configuration ---
# NOTE: Using absolute paths as requested. Ensure your path is correct.
# Windows paths use double backslashes (\\) for Python to interpret them correctly.
PROJECT_ROOT = 'C:\\Users\\mohid\\Desktop\\pothole'

XML_DIR = os.path.join(PROJECT_ROOT, 'annotations') # Full path to your XML folder
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'images')    # Full path to your image folder
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'pothole_yolo') # Full path for the output structure

# List of all classes in your dataset. The index in this list is the class ID 0.
CLASS_NAMES = ['pothole'] 

# Dataset split ratios. These must sum to 1.0.
SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15} 

# --- Helper Function for Coordinate Conversion ---
def convert_to_yolo_format(size, box):
    """
    Converts PASCAL VOC (xmin, xmax, ymin, ymax) to 
    YOLO (x_center, y_center, width, height) normalized coordinates.
    """
    img_w, img_h = size[0], size[1]
    xmin, xmax, ymin, ymax = box[0], box[1], box[2], box[3]
    
    # Calculate center, width, and height in normalized values (0.0 to 1.0)
    x_center = (xmin + xmax) / (2.0 * img_w)
    y_center = (ymin + ymax) / (2.0 * img_h)
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    
    return (x_center, y_center, width, height)

# --- Core Processing Function ---
def process_xml_file(xml_path, classes):
    """
    Parses a single PASCAL VOC XML file and returns a list of 
    YOLO-formatted label strings for that image.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print(f"Error: Could not parse XML file at {xml_path}. Skipping.")
        return []

    root = tree.getroot()
    
    size_node = root.find('size')
    try:
        img_w = int(size_node.find('width').text)
        img_h = int(size_node.find('height').text)
    except AttributeError:
        print(f"Error: Could not find image size in XML file {xml_path}. Skipping.")
        return []
    
    yolo_lines = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        try:
            class_id = classes.index(class_name)
        except ValueError:
            print(f"Warning: Class '{class_name}' in XML not defined in CLASS_NAMES. Skipping.")
            continue
            
        bndbox = obj.find('bndbox')
        try:
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
        except AttributeError:
            print(f"Warning: Missing bounding box coordinates in XML file {xml_path}. Skipping object.")
            continue
        
        yolo_box = convert_to_yolo_format((img_w, img_h), (xmin, xmax, ymin, ymax))
        
        line = f"{class_id} {' '.join([f'{f:.6f}' for f in yolo_box])}\n"
        yolo_lines.append(line)
        
    return yolo_lines

# --- Main Execution ---
def main():
    print(f"Starting data preparation from project root: {PROJECT_ROOT}")
    
    # 1. Setup output directories
    for subset in SPLIT_RATIOS:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', subset), exist_ok=True)
        print(f"Created directories for {subset}.")

    # 2. Get all XML file names and shuffle them for random splitting
    try:
        xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    except FileNotFoundError:
        print(f"\nFATAL ERROR: XML folder not found at {XML_DIR}. Check your path configuration.")
        return

    if not xml_files:
        print(f"\nFATAL ERROR: No XML files found in the '{XML_DIR}' directory.")
        return

    random.shuffle(xml_files)
    num_files = len(xml_files)
    
    # Calculate split index points
    train_end = int(SPLIT_RATIOS['train'] * num_files)
    val_end = train_end + int(SPLIT_RATIOS['val'] * num_files)
    
    data_splits = {
        'train': xml_files[:train_end],
        'val': xml_files[train_end:val_end],
        'test': xml_files[val_end:]
    }

    # 3. Report split sizes
    print(f"\n--- Data Split Summary ---")
    print(f"Total files: {num_files}")
    print(f"Train Set: {len(data_splits['train'])} files ({SPLIT_RATIOS['train']*100:.0f}%)")
    print(f"Validation Set: {len(data_splits['val'])} files ({SPLIT_RATIOS['val']*100:.0f}%)")
    print(f"Test Set: {len(data_splits['test'])} files ({SPLIT_RATIOS['test']*100:.0f}%)")
    print("--------------------------")

    # 4. Process each split: Convert labels and copy images
    image_extensions = ['.jpg', '.jpeg', '.png']

    for subset, files in data_splits.items():
        print(f"\nProcessing {subset} set...")
        for xml_file in files:
            base_name = os.path.splitext(xml_file)[0]
            xml_path = os.path.join(XML_DIR, xml_file)
            
            # 4a. Conversion (XML -> YOLO TXT)
            yolo_lines = process_xml_file(xml_path, CLASS_NAMES)
            
            # Save the YOLO label file
            label_output_path = os.path.join(OUTPUT_DIR, 'labels', subset, f"{base_name}.txt")
            with open(label_output_path, 'w') as f:
                f.writelines(yolo_lines)
            
            # 4b. Copy the image file
            src_image_path = None
            dst_image_path = None
            
            for ext in image_extensions:
                temp_path = os.path.join(IMAGE_DIR, base_name + ext)
                if os.path.exists(temp_path):
                    src_image_path = temp_path
                    dst_image_path = os.path.join(OUTPUT_DIR, 'images', subset, base_name + ext)
                    break 

            if src_image_path:
                shutil.copyfile(src_image_path, dst_image_path)
            else:
                print(f"Warning: Image file not found for {base_name}. Skipping image copy.")

    print("\n✅ Data preparation complete! The data is now ready for YOLO training in the 'pothole_yolo' folder.")

if __name__ == '__main__':
    main()