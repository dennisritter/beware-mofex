import cv2
import numpy as np

h = 227
w = 227
img = np.full((h, w, 3), 123, dtype=np.uint8)
img[:, 0:114] = (15, 99, 201)
cv2.imshow('Cool Image.', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Yow')
