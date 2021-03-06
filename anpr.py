import cv2
import numpy as np
from os.path import splitext, basename
from keras.models import model_from_json
import glob
from sklearn.preprocessing import LabelEncoder
from local_functions import detect_np

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

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

# Create a list of image paths
image_paths = glob.glob("car_image_source/image_5.jpg")
print("Found %i images..."%(len(image_paths)))

def get_plate(image_path, Dmax=608, Dmin=608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_np(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# Obtain plate image and its coordinates from an image
image = image_paths[0]
LpImg,cor = get_plate(image)
print("Detect %i plate(s) in"%len(LpImg),splitext(basename(image))[0])
print("Coordinate of plate(s) in image: \n", cor)

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


# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


cont, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

# define standard width and height of character
digit_w, digit_h = 30, 60

for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h / w
    if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
        if h / plate_image.shape[0] >= 0.4:  # Select contour which has the height larger than 50% of the plate
            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Sperate number and gibe prediction
            curr_num = erosion[y:y + h, x:x + w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)

print("Detect {} letters...".format(len(crop_characters)))
print(len(cont))

# Load model architecture, weight and labels
json_file = open('MobileNets_char_rec.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("character_recognition.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('character_classes.npy')
print("[INFO] Labels loaded successfully...")

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

final_string = ''
for i,character in enumerate(crop_characters):
    title = np.array2string(predict_from_model(character, model, labels))
    final_string += title.strip("'[]")

print('Plate Number: ', final_string)

