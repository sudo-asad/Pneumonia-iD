# Functions related to  dataset statistics and preparation for training, testing and validation
import random
from os import path, walk, makedirs
from PIL import Image
import time
import statistics

# Relative path of directory where the images are.
DATASET_PATH = "dataset"

# Output dataset path. It will be created if necessary. If it already exists, then the user will be asked about deletion of previous files.
OUTPUT_DATASET_PATH = "shuffled_dataset"

# File extension filter
FILE_EXTENSION = ["jpg", "jpeg"]

# Config variables
# We don't care about the original split directory names, like val, train etc. We only care about the different class name structure
# so we can shuffle them all for each class, and then create our own split structure
CLASS_FOLDER_NAMES = {"normal": "healthy", "opacity": "pneumonia"}

TARGET_TRAINING_PERCENT = 0.7
TARGET_VALIDATION_PERCENT = 0.15
TARGET_TESTING_PERCENT = 0.15
TARGET_RESOLUTION_W = 256
TARGET_RESOLUTION_H = 256

def get_dataset_stats():
    class_count = {k: 0 for k in CLASS_FOLDER_NAMES.values()}
    image_sizes = {}
    file_list_dict = {k: [] for k in CLASS_FOLDER_NAMES.values()}
    
    for root, dir, file_list in walk(DATASET_PATH):
        for class_name, class_new_name in CLASS_FOLDER_NAMES.items():
            if class_name in root.lower():
                filtered_list = [path.abspath(path.join(root, x)) for x in file_list if x.rsplit(".", 1)[1].lower() in FILE_EXTENSION]
                file_list_dict[class_new_name].extend(filtered_list)
                class_count[class_new_name] += len(filtered_list)
            
            for f in file_list:
                if f.rsplit(".", 1)[1].lower() in FILE_EXTENSION:
                    full_path = path.abspath(path.join(root, f))
                    im = Image.open(full_path)
                    w, h = im.size
                    image_sizes[full_path] = (w, h)
    
    image_sizes_w = [s[0] for s in image_sizes.values()]
    image_sizes_h = [s[1] for s in image_sizes.values()]
    
    print("Median of widths: {}".format(statistics.median(image_sizes_w)))
    print("Median of heights: {}".format(statistics.median(image_sizes_h)))
    
    print("Quantiles-Width: {}".format(statistics.quantiles(image_sizes_w)))
    print("Quantiles-Height: {}".format(statistics.quantiles(image_sizes_h)))
    
    print("Minimum width: {}".format(min(image_sizes_w)))
    print("Minimum height: {}".format(min(image_sizes_h)))
    
    return file_list_dict


def resize_image(image_path, output_size, output_path):
    try:
        image = Image.open(image_path)
        image = image.resize(output_size)
        image.save(output_path)
        return True
    
    except Exception as e:
        print("Exception in converting image: {}".format(e.name))
        return False
       

def convert_and_shuffle_dataset(input_file_dict, output_path, output_width, output_height, sets_percentages):
    successfully_converted = 0
    failed_converted = []
    sets_dir_names = ["training", "validation", "testing"]
    
    # Create directories for class names
    for class_name in input_file_dict.keys():
        dir_paths = [path.join(output_path, x, class_name) for x in sets_dir_names]
        
        for dir_path in dir_paths:
            if not path.exists(dir_path):
                makedirs(dir_path)
        
    for class_name, file_list in input_file_dict.items():
        #Shuffle list first
        random.shuffle(file_list)
        sets_dir_target_count = [round(len(file_list) * x) for x in sets_percentages]
        for i in range(len(sets_dir_target_count)):
            print("For the class '{}', the set '{}' will have {} images".format(class_name, sets_dir_names[i], sets_dir_target_count[i]))
            
        sets_dir_idx = 0
        sets_current_count = 0
        for current_image in file_list:
            filename = path.basename(current_image)
            dir_path = path.join(output_path, sets_dir_names[sets_dir_idx], class_name)
            
            if resize_image(current_image, (output_width, output_height), path.join(dir_path, filename)):
                successfully_converted += 1
                sets_current_count += 1
                if sets_current_count >= sets_dir_target_count[sets_dir_idx]:
                    sets_current_count = 0
                    sets_dir_idx += 1
                    if sets_dir_idx >= len(sets_dir_names):
                        sets_dir_idx = 0
                        
            else:
                failed_converted.append(current_image)
    
    print("Successfully converted {} images".format(successfully_converted))
    print("Failed images: {}".format(failed_converted))


dataset_files = get_dataset_stats()
for k, v in dataset_files.items():
    print("For class type '{}', there are {} images".format(k, len(v)))
    
print("Shuffling and converting images now...")
convert_and_shuffle_dataset(dataset_files, path.abspath(OUTPUT_DATASET_PATH), TARGET_RESOLUTION_W, TARGET_RESOLUTION_H, [TARGET_TRAINING_PERCENT, TARGET_VALIDATION_PERCENT, TARGET_TESTING_PERCENT])
print("Done!")


