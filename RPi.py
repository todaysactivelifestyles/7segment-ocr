import cv2
from Relay_abstract import NOrelay
import toaster_control
import RPi.GPIO as GPIO
import sys
import argparse

class NOrelay_over_RPi(NOrelay):

    def __init__(self, gpio_relay):
        self.gpio = gpio_relay

    def on(self):
        GPIO.output(self.gpio, 1)

    def off(self):
        GPIO.output(self.gpio, 0)


def main():

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(gpio_relay, GPIO.OUT)
    gpio_relay = 21
    SSR = NOrelay_over_RPi(gpio_relay)

    cap = cv2.VideoCapture(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--temp")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    if args.temp:
        target_temp = int(args.temp)
    else:
        target_temp = 30

    toaster_control.mainloop(cap, SSR, target_temp, args.log)
    
    GPIO.cleanup()
    sys.exit()

if __name__ == "__main__":
    main()
