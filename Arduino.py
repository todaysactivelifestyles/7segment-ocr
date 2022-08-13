import cv2
import numpy as np
from imutils.perspective import four_point_transform
import imutils
import serial
import time
import datetime
import sys
import argparse
import csv

DIGITS_LOOKUP = {
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

ser = serial.Serial("/dev/ttyACM0", 9600)

def sort_4pts_clockwise(pts):
    sorted_y = pts[np.argsort(pts[:, 1]), :]
    upper = sorted_y[2:]
    lower = sorted_y[:2]
    upper = upper[np.argsort(upper[:, 0]), :]
    lower = lower[np.argsort(-lower[:, 0]), :]
    return np.vstack([upper, lower])

def searchdigits(image):
    image = imutils.resize(image, height=500)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.temp:
        target_temp = args.temp
    else:
        target_temp = 30

    if args.log:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        yyyyMMddHHmmss = now.strftime("%Y%m%d%H%M%S")
        f = open(f"./log_{yyyyMMddHHmmss}.csv")
        writer = csv.writer(f)
        time_init = time.time()


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
                if int(temp) < target_temp:
                    ser.write(b'1')
                else:
                    ser.write(b'0')
                
                if args.log:
                    time_elapsed = time.time() - time_init              
                    writer.writerow([time_elapsed, temp])

    except KeyboardInterrupt:
        ser.close()

        if args.log:
            f.close()
            
        sys.exit()


if __name__ == "__main__":
    main()
