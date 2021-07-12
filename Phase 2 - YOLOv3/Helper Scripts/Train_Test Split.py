import os
from shutil import copy

# Where to copy files from.
ORIGINAL_PATH    = "..\\Face Mask Dataset\\"
IMAGE_PATH       = ORIGINAL_PATH + "images\\"                   # "Phase 2 - YOLOv3\\Face Mask Dataset\\images"

# Where to paste files in.
TRAIN_DEST = ORIGINAL_PATH + "Train\\"
TEST_DEST  = ORIGINAL_PATH + "Test\\"
#--- Directories creation
os.makedirs(os.path.dirname(TRAIN_DEST), exist_ok=True)
os.makedirs(os.path.dirname(TEST_DEST) , exist_ok=True)


image_list = os.listdir(IMAGE_PATH) # Cpature the files names in the main directory
image_list.sort()                   # Sort to ensure that every image is followed by its annoations file.

train_list = image_list[:767*2]     # First 90% of the dataset, 767 Image + 767 annotation
test_list = image_list[767*2:]      # Final 10% of the dataset

#Train split
for i in range(0,len(train_list),2):
    img_src_file   = IMAGE_PATH + train_list[i]      # Image
    label_src_file = IMAGE_PATH + train_list[i + 1]  # Annotations

    # Copy image & annotation
    copy(img_src_file, TRAIN_DEST)
    copy(label_src_file, TRAIN_DEST)

#Test split
for i in range(0,len(test_list),2):
    img_src_file   = IMAGE_PATH + test_list[i]      # Image
    label_src_file = IMAGE_PATH + test_list[i + 1]  # Annotations

   # Copy image & annotation
    copy(img_src_file, TEST_DEST)
    copy(label_src_file, TEST_DEST)

print('Total Count :', len(image_list), "(853 * 2)")
print('Train       :', len(train_list), "(767 * 2)")
print('Test        :', len(test_list) , "(86 * 2)")