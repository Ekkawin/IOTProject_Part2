import tensorflow as tf
from tensorflow import keras
import numpy as np
import picamera #Importing the library of picamera
from time import sleep
import cv2
import RPi.GPIO as GPIO
import serial
import microgear.client as microgear

camera = picamera.PiCamera()
camera.capture('/home/pi/Desktop/Picture/img.jpg', resize=(28, 28))
print('Done')
img = cv2.imread('/home/pi/Desktop/Picture/img.jpg')
cv2.imshow('Original Image',img)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert BGR to HSV
cv2.imshow('hsv color', img_hsv)


Green = [20, 174, 64]
Yellow = [0, 255, 255]
Red = [136, 148, 142]

BGR_Fillter = Red
hsv_Fillter = cv2.cvtColor( np.uint8([[BGR_Fillter]] ), cv2.COLOR_BGR2HSV)[0][0]
thresh = 25

minHSV = np.array([hsv_Fillter[0] - thresh, hsv_Fillter[1] - thresh, hsv_Fillter[2] - thresh])
maxHSV = np.array([hsv_Fillter[0] + thresh, hsv_Fillter[1] + thresh, hsv_Fillter[2] + thresh])

maskColor = cv2.inRange(img_hsv, minHSV, maxHSV)

cv2.imshow('Detect Color', maskColor)
cv2.imwrite('/home/pi/Desktop/Projpic.jpg', maskColor)
Testing_img_path = '/home/pi/Desktop/Projpic.jpg'
input1 = cv2.imread(Testing_img_path)
#cv2.imshow('image1', input1)

#print (input_image.shape)

# Preprae data
input_image = cv2.imread(Testing_img_path,cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image,(28, 28))
cv2.imshow('image', input_image)
input_image = cv2.bitwise_not(input_image) # invert Black to White like data input when trained (Line is White)
input_image = np.expand_dims(input_image, axis=0) # Change the shape of image array like input image when trained

# ***load model***
loadpath = "num_reader2.model"
model = tf.keras.models.load_model(loadpath)

# Make predictions
predictions = model.predict(input_image)
#print(np.argmax(predictions))
if(np.argmax(predictions)==5):
    print("1")
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(40,GPIO.OUT)
    GPIO.output(40, GPIO.HIGH)
    sleep(3)
    GPIO.output(40, GPIO.LOW)
else:
    print("2")
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(40,GPIO.OUT)
    GPIO.setup(38,GPIO.OUT)
    GPIO.output(40, GPIO.HIGH)
    GPIO.output(38, GPIO.HIGH)
    sleep(3)
    GPIO.output(38, GPIO.LOW)
    GPIO.output(40, GPIO.LOW)
"""
GPIO.setmode(GPIO.BOARD)
GPIO.setup(40,GPIO.OUT)
GPIO.output(40, GPIO.HIGH)
sleep(3)
GPIO.output(40, GPIO.LOW)
"""
appid = "EkkawinV"
gearkey = "UX0O5uKTBZuOjEw"
gearsecret =  "xN2sMvgEp9Z3y1RB75J76LpSN"

ALIAS = "RaspberryPi"
thing = "FreeBoardTESR"

FEEDID = "EkkawinFeedTraining"
APIKEY = "0C8mmFd3USPXz3UPS287xgDaAcVk5YQJ"

microgear.create(gearkey,gearsecret,appid)
def connection():
    print("Now I am connected with netpie")

def subscription(topic,message):
    print(topic+" "+message)
    if "ON" in message :
        microgear.chat(thing,"ON," + ("%.1f" %netpie))
        GPIO.output(LED,GPIO.HIGH)
    elif "OFF" in message :
        microgear.chat(thing,"OFF," + ("%.1f" %netpie))
        GPIO.output(LED,GPIO.LOW)

def disconnect():
    print("disconnect is work")
    
microgear.setalias(ALIAS)
microgear.on_connect = connection
microgear.on_message = subscription
microgear.on_disconnect = disconnect
#microgear.subscribe("/mails")
microgear.connect(False)


ser = serial.Serial('/dev/ttyS0',115200, timeout = 1)
ser.close()
ser.open()
i=0
while 1:
    sleep(0.2)
    #ser.write("Hellowworld.\r\n".encode())
    line = (ser.readline())
    if (i<10):
        i = i+1
    else:
        if len(line) >0:
         netpiestring = line[6:9]
         print(line)
         print(netpiestring)
         netpie = float(netpiestring)
         #netpie = 50
         print("Writing Feed")
         print(netpie)
         data2feed = {"Range":("%1f" %netpie)}
         microgear.writeFeed(FEEDID, data2feed, APIKEY)
         i=0
    
       
    
    



    

# Visualize, check the answer
Original_input_image_testing = cv2.imread(Testing_img_path)
cv2.imshow("This is what your computer has seen.",Original_input_image_testing)
cv2.waitKey(0)
cv2.destroyAllWindows()
