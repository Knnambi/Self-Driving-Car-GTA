from keras.models import load_model
import pyautogui
import time
import cv2
from pynput.keyboard import Key, Controller,Listener
import numpy as np
time.sleep(5)
i=1
keyboard=Controller()
model = load_model('model_15.h5')
while i<10:
  im = pyautogui.screenshot(region=(0,200,800,400))
  filename = "Drive/file_%d.jpg"%i
  im.save(filename)
  read_original = cv2.imread(filename)
  read_resize = cv2.resize(read_original, (320,160))
  image = np.asarray(read_resize)  
  image = np.array([image]) 
  steering_angle = model.predict(image, batch_size=1)
  print(steering_angle)
 #  steering_angle=np.floor(steering_angle)
 #  print(steering_angle) 
  if (steering_angle < 1):
        output = 'w'
  elif (steering_angle < 2 ):
        output = 'a'
  elif (steering_angle < 3 ):
         output = 'd'
#   elif (steering_angle < 4 ):
#         output = 'd'
  else:
        output = 'g'
  print(output)
  keyboard.press(output)
  time.sleep(1)
  keyboard.release(output)
  if (steering_angle > 2 ):
     keyboard.press('w')
     time.sleep(1)
     keyboard.release('w')
  elif(steering_angle < 3):
     keyboard.press('w')
     time.sleep(1)
     keyboard.release('w')
  else:
    	output = 'g'
