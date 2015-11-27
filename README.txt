# USAGE

The Python file main.py can process an image as specified on the command line
 e.g. $ python main.py path_to_image.png 3 5 500 path_to_write_filtered_image_to.png
 the first number (3) is the size of the filter to be applied (in this case 3x3)
 the second number (5) is the standard deviation for the distance Gaussian function
 the third number (500) is the standard deviation for the intensity difference Gaussian function

The script can be run from within an interactive shell by
$ python -i main.py
