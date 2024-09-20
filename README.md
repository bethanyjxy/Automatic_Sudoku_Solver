# Automatic Sudoku Solver

## Project Description

A Sudoku is a popular logic-based number puzzle that requires the player to complete a 9x9 number grid by filling in the missing values while adhering to a set of constraints. The goal of this project is to develop an automatic Sudoku puzzle solver using image recognition and artificial intelligence techniques. The program captures and processes an image of a Sudoku puzzle, recognizes the digits, reconstructs the puzzle, and solves it.

By leveraging a Convolutional Neural Network (CNN) model for digit recognition, the program can effectively handle Sudoku puzzles from various images, making it a versatile tool for solving puzzles from different sources.

## Dataset

The model is trained using the **[Chars74K dataset](https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/)**, which provides a variety of handwritten characters for training the CNN model to recognize digits in Sudoku puzzles.

## Constraints

To solve the Sudoku puzzle, the following constraints must be met:

- **Row constraints:** Each row in the 9x9 grid must contain all the numbers from 1 to 9, without repetition.
- **Column constraints:** Each column must also contain all the numbers from 1 to 9, without repetition.
- **Subgrid constraints:** The 9x9 grid is divided into nine 3x3 subgrids, and each subgrid must contain the numbers 1 to 9, without repetition.

## Convolutional Neural Network (CNN) Model

The CNN model used in this project consists of several layers:

1. **Convolutional Layers:** 
   - Two initial layers with 60 filters of size 5x5, detecting basic features like edges and simple shapes.
   - Two additional layers with 30 filters of size 3x3, capturing more complex details.
   - Activation function: ReLU to introduce non-linearity.

2. **Pooling Layer:** 
   - Reduces spatial dimensions of the feature maps to lower computational load and prevent overfitting.

3. **Flatten Layer:**
   - Converts multi-dimensional tensors to 1D vectors for the fully connected layers.

4. **Fully Connected (Dense) Layers:** 
   - First dense layer with 500 neurons.
   - Second dense layer combines features and outputs class probabilities using the softmax activation function.

5. **Dropout Layer:**
   - A regularization technique that randomly sets a fraction of input units to 0 during training to prevent overfitting.

6. **Compilation and Training:**
   - Learning rate: 0.001.
   - Epochs: 10 (data shuffled before each epoch).
   - Optimizer: Adam.

## Image Preprocessing

When the program is run, it first preprocesses the input Sudoku image using the following steps:

1. **Convert to Grayscale:** The image is converted from its RGB format to grayscale for easier processing.
2. **Apply Gaussian Blur:** A kernel size of 5x5 and sigma value of 1 are used to reduce noise and help with edge detection.
3. **Adaptive Thresholding:** The image is converted into a binary image (black and white), simplifying the contour detection process.
4. **Find Biggest Contour:** Contour detection is used to find the Sudoku grid in the image, which is then extracted using perspective transformation.
5. **Warping the Grid:** The Sudoku grid is warped into a perfect square.
6. **Split into 81 Small Images:** The grid is divided into 81 individual images, each representing a cell in the puzzle.
7. **Image Recognition:** Each small image is resized to 32x32 and normalized for input to the CNN model, which predicts the digits. If the highest probability class has confidence greater than 0.8, it is added to the Sudoku puzzle. Otherwise, it's marked as empty.
  
## Running the Project

To run the Sudoku solver:

1. Clone this repository.
2. Download the **[Chars74K dataset](https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/)** and ensure it's correctly set up for model training.
3. Train the CNN model using the dataset.
4. Run the solver with the following command:

   ```bash
   python SudokuMain.py

5. You will be prompted to enter the file path of the Sudoku puzzle image. For example:
   
   ```bash
   PuzzleImages/easy-sudoku.png

## Future Improvements

- Further optimize the CNN model for more efficient image recognition.
- Enhance the solver's ability to handle images with different types of noise and distortions.


