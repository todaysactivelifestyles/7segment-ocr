import cv2
from Relay_abstract import NOrelay
import toaster_control
import serial
import sys
import argparse

class NOrelay_over_Arduino(NOrelay):

    def __init__(self, ser):
        self.ser = ser

    def on(self):
         self.ser.write(b'1')

    def off(self):
        self.ser.write(b'0')

    def close(self):
        self.ser.close()


def main():

    ser = serial.Serial("/dev/ttyACM0", 9600)
    SSR = NOrelay_over_Arduino(ser)
    cap = cv2.VideoCapture(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.temp:
        target_temp = int(args.temp)
    else:
        target_temp = 30

    toaster_control.mainloop(cap, SSR, target_temp, args.log)

    SSR.close()
    sys.exit()

if __name__ == "__main__":
    main()
