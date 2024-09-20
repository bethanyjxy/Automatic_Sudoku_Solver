import numpy as np
import cv2
import pickle

pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

# Read the image file
image_path = "Numbers\Sample4\img005-00003.png"  # Replace with your image file path
imgOriginal = cv2.imread(image_path)

if imgOriginal is None:
    print("Failed to read image")
else:
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)

    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img)
    classIndex = int(np.argmax(predictions))
    probVal = np.amax(predictions)

    print(f"Predicted Class: {classIndex}, Probability: {probVal}")

    if probVal > 0.8:
        cv2.putText(imgOriginal, f"{classIndex} {probVal:.2f}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("Original Image", imgOriginal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()