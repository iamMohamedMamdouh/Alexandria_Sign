#Take Photo For Dataset #
import cv2
import os
import time
import uuid

images_path ='Images/collect_images'

labels = ['Al_Mansheya-Raml_station-Sayidi_Bashar','Bahari-Almandara-Miami','Victoria','Mawqif','Forty_Five','Twenty_one','Alsaaea','Abu_Talata']
number_imgs = 15

for label in labels:
    os.makedirs(os.path.join(images_path, label), exist_ok=True)
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        imagename = os.path.join(images_path, label , label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename ,frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()