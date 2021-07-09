import cv2
import glob         # pathnames pattern matching library
import os
import argparse     # Console arguments library


#---------------- Prepare the command line -----------------------------
def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description="Display Bounding-Boxes on png images with yolo format (<label> <x_center> <y_center> <width> <height>).")

    parser.add_argument("--input", type=str, default="..\\Face Mask Dataset\\images\\",
                    help="The path of the folder that contains the png images with their corresponding txt annotations.")


    return parser.parse_args()

#----------------------- CHECK FOR ARGUMENT ERRORS ---------------------
def check_arguments_errors(args):
    if not os.path.exists(args.input):
        raise(ValueError("Invalid input folder path: {}".format(os.path.abspath(args.input))))


#---------------- BOUNDING BOX DRAWER ------- --------------------------
def show_bounding_boxes(dir_path):
    """ Auxiliary function that displays the bounding boxes of images in folder <dir_path>.
    """

    # Loop through the images & its annotations files to display the bounding boxes.
    for image_name in glob.glob(dir_path + '/*.png'):

        image = cv2.imread(image_name)  # Read the image
        height, width, _ = image.shape  # Capture its size


        # Open the correspondence image annotations file to use its values to display the bounding boxes.
        image_annotation = "..\\Face Mask Dataset\\images\\" + image_name.split("\\")[-1][0:-4] + '.txt'
        with open(image_annotation, 'r') as reader:

            annotations = reader.readlines()

            for annotation in annotations:
                annotation = annotation.split()
                
                # Calculation of top left point and bottom right point of the bounding box           
                x1, y1 = (int( ( float(annotation[1]) - float(annotation[3] ) / 2 ) * width ),
                          int( ( float(annotation[2]) - float(annotation[4] ) / 2 ) * height))

                x2, y2 = (int( ( float(annotation[1]) + float(annotation[3]) / 2 ) * width),
                          int( ( float(annotation[2]) + float(annotation[4]) / 2 ) * height))
                
                # BGR color format
                if annotation[0] == '0':
                    color = (0, 255, 0)  # Mask is worn correctly (Green color)
                    label = 'OK'
                elif annotation[0] == '1':
                    color = (0, 0, 255)  # Mask is not worn correctly (Yellow color)
                    label = 'Bad'
                else:
                    color = (0, 255, 255)   #Mask is not worn at all (Red color)
                    label = 'Wrong'
                
                cv2.putText(image,
                        label, 
                        (x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5, 
                        color=color,
                        thickness=1) 
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)
        
        k = cv2.waitKey(0) & 0xFF
        cv2.imshow(image_name.split("sss")[-1], image)

        if k == 27:
            cv2.destroyAllWindows()
            break


def main():
    args = get_input_args()
    check_arguments_errors(args) 

    show_bounding_boxes(args.input)

if __name__ == "__main__":
    main()