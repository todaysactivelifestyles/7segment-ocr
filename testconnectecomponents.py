import cv2

threshold = cv2.imread("thresh.jpg")
color = threshold.copy()
threshold = cv2.cvtColor(threshold, cv2.COLOR_RGB2GRAY)
_, threshold = cv2.threshold(threshold, 1, 255, cv2.THRESH_BINARY)

retval, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold)
for c in stats:
    area = c[4]
    pt1 = (c[0], c[1])
    pt2 = (c[0]+c[2], c[1]+c[3])
    cv2.rectangle(color, pt1, pt2, color=(0,255,0))

cv2.imwrite("color.jpg", color)