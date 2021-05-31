import numpy as np
import cv2
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from local_functions import detect_np
from os.path import splitext
from motion_detection.singlemotiondetector import SingleMotionDetector
import datetime

import threading

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

# Load model architecture, weight and labels
json_file = open('MobileNets_char_rec_18052021.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("character_recognition_18052021.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('character_classes.npy')
print("[INFO] Labels loaded successfully...")

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image, Dmax=608, Dmin=608):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    ratio = float(max(image.shape[:2])) / min(image.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_np(wpod_net, image, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
    return cnts

# pre-processing input images and pedict with model
def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,) * 3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction

##############################  Clear the opencv bufer:
import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name, format):
    self.cap = cv2.VideoCapture(name, format)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

  def release(self):
      self.cap.release()

def camera_in():
    total = 0
    totalFrames = 0
    previosNumber = ''
    cap = VideoCapture("rtsp://admin:admin123@192.168.1.243", cv2.CAP_FFMPEG)
    md = SingleMotionDetector(accumWeight=0.1)
    k = 0
    minX, minY, maxX, maxY = 0, 0, 0, 0
    #width = cap.get(3)
    #height = cap.get(4)

    while True: #(cap.isOpened()):
        # Capture frame-by-frame
        frame = cap.read()

        height, width = frame.shape[:2]

        crop = frame[int(height / 3):int(height), int(width / 3): int(width)]

        ht, wh = crop.shape[:2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        timestamp = datetime.datetime.now()
        cv2.putText(crop, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, crop.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if total > 10:
            # detect motion in the image
            motion = md.detect(gray)
            # check to see if motion was found in the frame
            if motion is not None:
                k = 1
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                #cv2.rectangle(crop, (minX, minY), (maxX, maxY),
                #              (0, 0, 255), 2)
            else:
                k = 0
        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

        cv2.line(crop, (0, int(ht / 2)), (int(wh/5), int(ht / 2)), (100, 0, 0), 3)

        cv2.imshow('Region of Interest', crop)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if (maxY <= int(ht / 2)) or (k == 0):
            continue


        if totalFrames % 1 == 0:
            # Obtain plate image and its coordinates from an image

            try:
                LpImg, cor = get_plate(crop)
                # print("Detect %i plate(s) in" % len(LpImg))
                # print("Coordinate of plate(s) in image: \n", cor)
            except:
                totalFrames += 1
                continue

            if (len(LpImg)):  # check if there is at least one license image
                # Scales, calculates absolute values, and converts the result to 8-bit.
                plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

                # convert to grayscale and blur the image
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)

                # Applied inversed thresh_binary
                binary = cv2.threshold(blur, 180, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

            kernel = np.ones((6, 6), np.uint8)
            erosion = cv2.erode(thre_mor, kernel, iterations=1)

            h, w = erosion.shape
            erosion[0:int(0.1 * h), 0:w] = 0
            erosion[int(0.9 * h):h, 0:w] = 0
            erosion[0:h, 0:int(0.04 * w)] = 0
            erosion[0:h, int(0.96 * w):w] = 0

            cont, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # creat a copy version "test_roi" of plat_image to draw bounding box
            test_roi = plate_image.copy()

            # Initialize a list which will be used to append character image
            crop_characters = []

            # define standard width and height of character
            digit_w, digit_h = 30, 60

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h / w
                if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
                    if h / plate_image.shape[0] >= 0.35:  # Select contour which has the height larger than 50% of the plate
                        # Draw bounding box arroung digit number
                        cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Sperate number and gibe prediction
                        curr_num = erosion[y:y + h, x:x + w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)

            if len(crop_characters) != 8:
                totalFrames += 1
                continue
            # print("Detect {} letters...".format(len(crop_characters)))

            final_string = ''
            for i, character in enumerate(crop_characters):
                title = np.array2string(predict_from_model(character, model, labels))
                final_string += title.strip("'[]")

            if 'G' in final_string[:2]:
                final_string = final_string[:2].replace('G', '0') + final_string[2:8]

            if final_string != previosNumber:
                print('Plate Number: ', final_string)


                previosNumber = final_string

            totalFrames += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def camera_out():
    total = 0
    totalFrames = 0
    previosNumber = ''
    cap = VideoCapture("rtsp://admin:admin123@192.168.1.243", cv2.CAP_FFMPEG)
    md = SingleMotionDetector(accumWeight=0.1)
    k = 0
    minX, minY, maxX, maxY = 0, 0, 0, 0
    #width = cap.get(3)
    #height = cap.get(4)

    while True: #(cap.isOpened()):
        # Capture frame-by-frame
        frame = cap.read()

        height, width = frame.shape[:2]

        crop = frame[int(height / 3):int(height), int(width / 3): int(width)]

        ht, wh = crop.shape[:2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        timestamp = datetime.datetime.now()
        cv2.putText(crop, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, crop.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if total > 10:
            # detect motion in the image
            motion = md.detect(gray)
            # check to see if motion was found in the frame
            if motion is not None:
                k = 1
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                #cv2.rectangle(crop, (minX, minY), (maxX, maxY),
                #              (0, 0, 255), 2)
            else:
                k = 0
        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

        cv2.line(crop, (0, int(ht / 2)), (int(wh/5), int(ht / 2)), (100, 0, 0), 3)

        cv2.imshow('Region of Interest1', crop)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if (maxY <= int(ht / 2)) or (k == 0):
            continue


        if totalFrames % 1 == 0:
            # Obtain plate image and its coordinates from an image

            try:
                LpImg, cor = get_plate(crop)
                # print("Detect %i plate(s) in" % len(LpImg))
                # print("Coordinate of plate(s) in image: \n", cor)
            except:
                totalFrames += 1
                continue

            if (len(LpImg)):  # check if there is at least one license image
                # Scales, calculates absolute values, and converts the result to 8-bit.
                plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

                # convert to grayscale and blur the image
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)

                # Applied inversed thresh_binary
                binary = cv2.threshold(blur, 180, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

            kernel = np.ones((6, 6), np.uint8)
            erosion = cv2.erode(thre_mor, kernel, iterations=1)

            h, w = erosion.shape
            erosion[0:int(0.1 * h), 0:w] = 0
            erosion[int(0.9 * h):h, 0:w] = 0
            erosion[0:h, 0:int(0.04 * w)] = 0
            erosion[0:h, int(0.96 * w):w] = 0

            cont, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # creat a copy version "test_roi" of plat_image to draw bounding box
            test_roi = plate_image.copy()

            # Initialize a list which will be used to append character image
            crop_characters = []

            # define standard width and height of character
            digit_w, digit_h = 30, 60

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h / w
                if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
                    if h / plate_image.shape[0] >= 0.35:  # Select contour which has the height larger than 50% of the plate
                        # Draw bounding box arroung digit number
                        cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Sperate number and gibe prediction
                        curr_num = erosion[y:y + h, x:x + w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)

            if len(crop_characters) != 8:
                totalFrames += 1
                continue
            # print("Detect {} letters...".format(len(crop_characters)))

            final_string = ''
            for i, character in enumerate(crop_characters):
                title = np.array2string(predict_from_model(character, model, labels))
                final_string += title.strip("'[]")

            if 'G' in final_string[:2]:
                final_string = final_string[:2].replace('G', '0') + final_string[2:8]

            if final_string != previosNumber:
                print('Plate Number: ', final_string)


                previosNumber = final_string

            totalFrames += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

threading.Thread(target=camera_in).start()
threading.Thread(target=camera_out).start()
