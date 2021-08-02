import pandas as pd
import cv2
import os


df = pd.read_csv('metadata.csv', usecols= ['RT_PCR_positive','filename'])
covid_data_filenames = []
for index, row in df.iterrows():
    if (row['RT_PCR_positive'] == 'Y'):
        covid_data_filenames.append(row['filename'])

os.makedirs("covid_positive")

for subdir, dirs, files in os.walk("images"):
    for image_name in files:
        if (image_name[0] != '.' and image_name in covid_data_filenames):
            image = cv2.imread("images/"f"{image_name}")
            cv2.imwrite("covid_positive/"f"{image_name}", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

