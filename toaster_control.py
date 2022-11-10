import cv2
from relay_abstract import NOrelay
import thermometer_OCR
import time
import datetime
import csv

def mainloop(cap: cv2.VideoCapture, Relay: NOrelay, target_temp: int, logging: bool):

    if logging:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        yyyyMMddHHmmss = now.strftime("%Y%m%d%H%M%S")
        f = open(f"./log_{yyyyMMddHHmmss}.csv")
        writer = csv.writer(f)
        time_init = time.time()


    try:
        while True:
            _, image = cap.read()
            digits = thermometer_OCR.searchdigits(image)
            temp = ""
            for i in digits:
                try:
                    predict = thermometer_OCR.read_digit(i)
                    temp = temp + str(predict)
                    
                except KeyError:
                    temp = ""
                    break

            if temp == "":
                time.sleep(1)
                continue
            else:
                print(temp)
                if int(temp) < target_temp:
                    Relay.on()
                else:
                    Relay.off()

                if logging:
                    time_elapsed = time.time() - time_init              
                    writer.writerow([time_elapsed, temp])

                time.sleep(1)
                
    except KeyboardInterrupt:

        if logging:
            f.close()
        
