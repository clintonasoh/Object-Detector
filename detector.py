import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Read in the image
path = r'fruits.jpeg'
img = cv2.imread(path, -1)

bbox, label, conf = cv.detect_common_objects(img)
output_image = draw_bbox(img, bbox, label, conf)

# Visualize result with boxed detections.
plt.imshow(output_image)
plt.show()