import cv2
import numpy as np
import sys
import time

DIGITS_LOOKUP = {
    #   1
    #  2 3
    #   4
    #  5 6
    #   7

    (1, 1, 1, 0, 1, 1, 1): '0',
    (0, 1, 0, 0, 1, 0, 0): '1',
    (1, 0, 1, 1, 1, 0, 1): '2',
    (1, 0, 1, 1, 0, 1, 1): '3',
    (0, 1, 1, 1, 0, 1, 0): '4',
    (1, 1, 0, 1, 0, 1, 1): '5',
    (1, 1, 0, 1, 1, 1, 1): '6',
    (1, 0, 1, 0, 0, 1, 0): '7',
    (1, 1, 1, 1, 1, 1, 1): '8',
    (1, 1, 1, 1, 0, 1, 1): '9',
    (0, 1, 1, 1, 1, 1, 0): 'H'
}

def sort_4pts(pts):
    sorted_y = pts[np.argsort(pts[:, 1]), :]
    upper = sorted_y[:2]
    lower = sorted_y[2:]
    (upper_l, upper_r) = upper[np.argsort(upper[:, 0]), :]
    (lower_r, lower_l) = lower[np.argsort(-lower[:, 0]), :]
    print(np.array([upper_l, upper_r, lower_r, lower_l]))
    return np.array([upper_l, upper_r, lower_r, lower_l], dtype=np.float32)

def four_point_transform(image, pts):
    pts = sort_4pts(pts)
    dst = np.array([(0,0), (310, 0), (310, 220), (0, 220)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (310, 220))

    # return the warped image
    return warped


def searchdigits(image):
    (h, w) = image.shape[:2]
    dst_height = 500
    r = dst_height / float(h)
    dim = (int(w * r), dst_height)
    image = cv2.resize(image, dsize=dim)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    display_contour = None
    # loop over the contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if the contour has four vertices, then we have found the thermostat display
        if len(approx) == 4:
            display_contour = approx
            break

    warped = four_point_transform(gray, display_contour.reshape(4, 2))

    _, threshold = cv2.threshold(warped, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    threshold = cv2.dilate(threshold, np.ones((7, 1), np.uint8), iterations=1)

    threshold = threshold[20:115, 10:140]

    # find contours in the thresholded image, then initialize the digit contours lists
    contours, _ = cv2.findContours(
        threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = []
    # loop over the digit area candidates
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 10 and (h > 50):
            digitCnts.append(cv2.boundingRect(c))

    digitCnts.sort(key=lambda x: x[0])
    digits = []
    for c in digitCnts:
        x, y, w, h = c
        digits.append(threshold[y:y+h, x:x+w])

    return digits


def read_digit(image):

    height, width = image.shape

    if width < 20:
        return 1

    else:
        dW = int(width * 0.35)
        dH = int(height * 0.15)
        dHC = int(dH / 2)

        segments = [
            ((0, 0), (width, dH)),  # top
            ((0, 0), (dW, height // 2)),  # top-left
            ((width - dW, 0), (width, height // 2)),  # top-right
            ((0, (height // 2) - dHC), (width, (height // 2) + dHC)),  # center
            ((0, height // 2), (dW, height)),  # bottom-left
            ((width - dW, height // 2), (width, height)),  # bottom-right
            ((0, height - dH), (width, height))  # bottom
        ]

        display = [0] * 7    # on/off

        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            roi = image[yA:yB, xA:xB]
            count = cv2.countNonZero(roi)
            area = (xB - xA) * (yB - yA)
            if count / float(area) > 0.5:
                display[i] = 1

        digit = DIGITS_LOOKUP[tuple(display)]
        return digit


def main():

    cap = cv2.VideoCapture(2)
    try:
        while True:
            _, image = cap.read()
            cv2.imwrite("a.jpg", image)
            digits = searchdigits(image)
            temp = ""
            for i in digits:
                try:
                    predict = read_digit(i)
                    temp = temp + str(predict)
                except KeyError:
                    temp = ""
                    break

            if temp == "":
                time.sleep(1)
                continue
            else:
                print(temp)
                time.sleep(1)


    except KeyboardInterrupt:
        sys.exit()


if __name__ == "__main__":
    main()
