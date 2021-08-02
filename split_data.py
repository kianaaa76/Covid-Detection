import random
import os
import cv2

random_indexes_for_test_in_positive = []
random_indexes_for_test_in_negative = []
random_indexes_for_valid_in_positive = []
random_indexes_for_valid_in_negative = []

valid_size_per_class = 240
test_size_per_class = 120
total_size_per_class = 1200

# select random indexes within negative class for validation
iteration = 0
while(iteration < valid_size_per_class):
    random_index = random.randint(0, total_size_per_class)
    if (random_index not in random_indexes_for_valid_in_negative):
        random_indexes_for_valid_in_negative.append(random_index)
        iteration += 1

# select random indexes within positive class for validation
iteration = 0
while(iteration < valid_size_per_class):
    random_index = random.randint(0, total_size_per_class)
    if (random_index not in random_indexes_for_valid_in_positive):
        random_indexes_for_valid_in_positive.append(random_index)
        iteration += 1


# select random indexes within negative class for test
iteration = 0
while(iteration < test_size_per_class):
    random_index = random.randint(0, total_size_per_class)
    if (random_index not in random_indexes_for_test_in_negative):
        random_indexes_for_test_in_negative.append(random_index)
        iteration += 1

# select random indexes within positive class for test
iteration = 0
while(iteration < test_size_per_class):
    random_index = random.randint(0, total_size_per_class)
    if (random_index not in random_indexes_for_test_in_positive):
        random_indexes_for_test_in_positive.append(random_index)
        iteration += 1

# split normal (non-covid) images into train, validation and test
index = 0
for subdir, dirs, files in os.walk("Non-COVID-19"):
    for image_name in files:
        if (image_name[0] != '.'):
            image = cv2.imread("Non-COVID-19/"f"{image_name}") 
            if (index in random_indexes_for_test_in_negative): 
                cv2.imwrite("test/negative/"f"{image_name}", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  
            elif (index in random_indexes_for_valid_in_negative):
                cv2.imwrite("validation/negative/"f"{image_name}", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            else:
                cv2.imwrite("train/negative/"f"{image_name}", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            index += 1

# split covid images into train, validation and test
index = 0
for subdir, dirs, files in os.walk("COVID-19"):
    for image_name in files:
        if (image_name[0] != '.'):
            image = cv2.imread("COVID-19/"f"{image_name}") 
            if (index in random_indexes_for_test_in_positive): 
                cv2.imwrite("test/positive/"f"{image_name}", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  
            elif (index in random_indexes_for_valid_in_positive):
                cv2.imwrite("validation/positive/"f"{image_name}", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            else:
                cv2.imwrite("train/positive/"f"{image_name}", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            index += 1


        
