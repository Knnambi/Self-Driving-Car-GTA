from pynput.keyboard import Key, Controller,Listener
import time
import pyautogui
import numpy as np
import click
import cv2
import pyscreenshot as ImageGrab
import keyboard as keyy
from PIL import Image
import pickle

# ############################# keyboard ctrl ############################
keyboard = Controller()
trainingdataset=[]
image_list=[]
time.sleep(5)
i=1
  im = pyautogui.screenshot(region=(0,200,800,400))
  #im.show()
#            # run a color convert:
#  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  filename = "Training/file_%d.jpg"%i
  im.save(filename)
  
  read_original = cv2.imread(filename)
  read_resize = cv2.resize(read_original, (320,160))

  image_list.append(read_resize)
  X=np.array(image_list)
  keys = keyy.read_key()
  if 'w' in keys:
        output = 1
  elif 'a' in keys:
        output = 2
  elif 'd' in keys:
        output = 3
#  elif 'd' in keys:
#        output = 4
  else:
        output = 0
  print(output)
 #recorded = key.record()
  trainingdataset.append([output])
  Y=np.array(trainingdataset)
 #keyboard.press('w')
  i=i+1
  file_name = 'E:/Hypo/GTA/Behavoiral Cloning/training_data.npy'
  with open('my_dataset.pickle', 'wb') as output:
   pickle.dump(trainingdataset, output)
  with open('my_datasetimg.pickle', 'wb') as output:
   pickle.dump(image_list, output)
 #state = pygame.key.get_pressed()
  np.save(file_name,trainingdataset)

 
