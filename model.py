
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Load the pre-trained fire detection model
# classifier = load_model('classifier.h5')

# def preprocess_image(img):
#     # Convert the image to grayscale if it has more than 1 channel
#     if len(img.shape) > 2:
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     else:
#         gray = img.copy()
    
#     # Normalize the pixel values to [0, 1]
#     gray = gray / 255.0
    
#     return gray



# def predict_part(inp_arr_image):
#     # Preprocess the input image
#     test_image = preprocess_image(inp_arr_image)
#     test_image = np.expand_dims(test_image, axis=0)
#     test_image = np.expand_dims(test_image, axis=-1)  # add channel dimension
    
#     # Make a prediction using the pre-trained model
#     result = classifier.predict(test_image)
#     if result[0][0] < 0.5:
#         # The model predicts that the input image contains fire
#         return True
#     else:
#         # The model predicts that the input image does not contain fire
#         return False

# def get_cells_img(np_arr_img, n=64):
#     # Split the input image into small cells of size n x n
#     sub_imgs = []
#     for row in range((np_arr_img.shape[0] // n) + 1):
#         for col in range((np_arr_img.shape[1] // n) + 1):
#             c_0 = col * n
#             c_1 = min((c_0 + n), np_arr_img.shape[1])
#             r_0 = row * n
#             r_1 = min((r_0 + n), np_arr_img.shape[0])
#             sub_imgs.append(np_arr_img[r_0:r_1, c_0:c_1, :])
#     return sub_imgs

# def predict(img_path):
#     # Load the input image
#     # inp_img = cv2.imread(img_path)

#     # Split the input image into cells and make predictions on each cell
#     fire_pred = [predict_part(img) for img in get_cells_img(img_path, n=128)]

#     # Count the number of cells predicted to contain fire
#     fire_cnt = sum(fire_pred)

#     # Make a final prediction based on the number of cells predicted to contain fire
#     if fire_cnt > 10:
#         # If more than 10 cells contain fire, return True (fire detected)
#         return True
#     else:
#         # Otherwise, return False (no fire detected)
#         return False

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained fire detection model
classifier = load_model('classifier.h5')

def get_img_array(path):
    # Load the input image and convert it to a NumPy array
    img = image.load_img(path, target_size=(500, 750))
    img_array = image.img_to_array(img)
    return img_array

def predict_part(inp_arr_image):
    # Preprocess the input image
    test_image = cv2.resize(inp_arr_image, (64, 64))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # normalize pixel values

    # Make a prediction using the pre-trained model
    result = classifier.predict(test_image)
    if result[0][0] < 0.5:
        # The model predicts that the input image contains fire
        return True
    else:
        # The model predicts that the input image does not contain fire
        return False

def get_cells_img(np_arr_img, n=64):
    # Split the input image into small cells of size n x n
    sub_imgs = []
    for row in range((np_arr_img.shape[0] // n) + 1):
        for col in range((np_arr_img.shape[1] // n) + 1):
            c_0 = col * n
            c_1 = min((c_0 + n), np_arr_img.shape[1])
            r_0 = row * n
            r_1 = min((r_0 + n), np_arr_img.shape[0])
            sub_imgs.append(np_arr_img[r_0:r_1, c_0:c_1, :])
    return sub_imgs

def predict(img_path):
    # Load the input image
    inp_img = get_img_array(img_path)
    img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB) # Convert the color space to RGB
    img = cv2.resize(img, (500, 750)) # Resize the image
    img = img / 255.0 # Normalize the pixel values


    # Split the input image into cells and make predictions on each cell
    fire_pred = [predict_part(img) for img in get_cells_img(inp_img, n=128)]

    # Count the number of cells predicted to contain fire
    fire_cnt = sum(fire_pred)

    print(fire_cnt)
    # Make a final prediction based on the number of cells predicted to contain fire
    if fire_cnt > 10:
        # If more than 10 cells contain fire, return True (fire detected)
        return True
    else:
        # Otherwise, return False (no fire detected)
        return False
