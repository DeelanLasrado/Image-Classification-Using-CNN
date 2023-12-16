#PREDICTION

# Load the trained model
from keras.models import load_model
model = load_model("cats_vs_dogs_model.h5")

# Load and preprocess an image for prediction
from keras.preprocessing import image
import numpy as np
# Defining the Image Properties
Image_Width=128
Image_Height=128
Image_Size=(Image_Width,Image_Height)

img_path = "C:\\Users\\deela\\Downloads\\train\\train\\cat.565.jpg"  # Replace with the path to your image
img = image.load_img(img_path, target_size=Image_Size)
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0  # Rescale the image as done during training


# Make a prediction
prediction = model.predict(img)
print(prediction)
if prediction[0][1] > 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")
#0-cat
#1-dog