import cv2
import os
from google.cloud import vision
import re
import glob
credientials_path = 'credentials.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=credientials_path
import pandas as pd
folder_path = ''

def detect_text(img, client = ''):
  # print(img.shape,'ssss')
  content = cv2.imencode('.jpg', img)[1].tobytes()

  image = vision.Image(content=content)
  response = client.text_detection(image=image)
  texts = response.text_annotations
  english_text = []
  number_text = []
  if len(texts)>0:
      text = texts[0].description
  else:
      english_text = " ".join( english_text)
      number_text = " ".join( number_text)
      return english_text, number_text
  text = re.sub('\n',' ',text)
  english_text = []
  number_text = []
  for word in text.split():
    if word.isdigit():
      english_text.append(word)
      number_text.append(word)
    elif any(alpha in word.lower() for alpha in 'abcdefghijklmnopqrstuvwxyz'):
      # print(word)
      english_text.append(word)
    else:
      number_text.append(word)
  english_text = " ".join( english_text)
  number_text = " ".join( number_text)
  return english_text, number_text

types = ('*.jpg', '*.png', '*.jpeg') # the tuple of file types
img_files = []
license_numbers = []
imgs_names = []
for files in types:
    img_files.extend(glob.glob(folder_path))
    
for img in img_files:
    client = vision.ImageAnnotatorClient()
    english_text, number_text = detect_text(img,client = client)
    if english_text == '' and number_text == '':
        continue
    license_numbers.append(english_text+' '+ number_text)
    imgs_names.append(os.path.basename(img))
    
df = pd.DataFrame(list(zip(imgs_names,license_numbers))
             , columns=['Image','License number'])
df.to_csv('Extracted Numbers.csv')