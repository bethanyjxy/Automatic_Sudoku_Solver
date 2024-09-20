
from utils import * 
import sudokuSolver

# user to enter the image path
ImagePath = get_valid_image_path()
print(f"The provided image path is: {ImagePath}")

heightImg = 405
widthImg = 405
model = initializePredictionModel()
print(model.input_shape)

#### Prepare image
img = cv2.imread(ImagePath)
img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThreshold = preProcess(img)

#### Find contours
imgContours = img.copy() 
imgBigContour = img.copy() 
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find all contours
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

#### Find the biggest contour
biggest, maxArea = biggestContour(contours) # Find the biggest contour
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    #### Split the image and predict the digits
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    numbers = getPrediction(boxes, model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)

    #### Find Solution
    board = np.array_split(numbers, 9)
    print(board)
    try:
        sudokuSolver.solve(board)
    except:
        pass
    print(board)

    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList * posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

    #### Overlay the digits
    pts2 = np.float32(biggest)
    pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([imgThreshold, imgContours, imgBigContour],
                [imgDetectedDigits, imgSolvedDigits, inv_perspective])
    stackedImages = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImages)

else:
    print("No Sudoku Found")

cv2.waitKey(0)






