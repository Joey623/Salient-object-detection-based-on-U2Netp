import os
import cv2 as cv
import glob

image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images' + os.sep)
mask_dir = os.path.join(os.getcwd(), 'test_data', 'unetlight_results' + os.sep)
mapping_dir = os.path.join(os.getcwd(), 'test_data', 'mapping' + os.sep)

image_list = glob.glob(image_dir + '*')
image_len = len(image_list)
mask_ext = '.png'
for image_path in image_list:
    test_image = cv.imread(image_path)
    test_name = image_path.split(os.sep)[-1]
    aaa = test_name.split(".")
    img_name = aaa[0:-1]
    # mapping_name = str(img_name[0])
    img_name = str(img_name[0]) + mask_ext
    mask_path = mask_dir + img_name
    mask_image = cv.imread(mask_path)
    _, mask_image = cv.threshold(mask_image, 128, 255, cv.THRESH_BINARY_INV)
    mapping_result = cv.add(test_image, mask_image)
    cv.imwrite(mapping_dir + img_name, mapping_result)

