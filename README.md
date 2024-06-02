
# GUI Image Processing Application

## Overview

This application provides a graphical user interface (GUI) for performing various image processing tasks using the Tkinter library in Python. The application allows users to load images, apply edge detection algorithms (Sobel and Canny), and perform image segmentation using thresholding and K-Means clustering. Additionally, the application can integrate webcam or video streaming for real-time processing.

## Features

1. **Edge Detection with Sobel Operator:**
   - Load an image and apply the Sobel operator to detect edges.
   - Display the original image and the detected edges side by side.

2. **Edge Detection with Canny Edge Detector:**
   - Apply the Canny edge detection algorithm with adjustable threshold values using sliders.
   - Display the original image and the detected edges side by side.

3. **Image Segmentation with Thresholding:**
   - Perform image segmentation using global and adaptive thresholding techniques.
   - Adjust threshold values interactively using sliders.
   - Display the segmented regions alongside the original image.

4. **Image Segmentation with K-Means Clustering:**
   - Segment the image using the K-Means clustering algorithm.
   - Allow users to specify the number of clusters and other relevant parameters.
   - Display the segmented regions alongside the original image.

5. **Webcam/Video Streaming Integration:**
   - Integrate webcam or video streaming functionality.
   - Perform real-time edge detection and segmentation on video input.
   - Display the processed video stream with detected edges or segmented regions in real time.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/gui-image-processing.git
   cd gui-image-processing
   ```

2. **Install Dependencies:**
   Ensure you have Python 3 installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

   **Requirements file (`requirements.txt`):**
   ```
   opencv-python
   opencv-python-headless
   pillow
   numpy
   tkinter
   ```

3. **Run the Application:**
   ```bash
   python gui_image_processing.py
   ```

## Usage

1. **Load an Image:**
   - Click the "Load Image" button to select an image file from your computer.

2. **Apply Edge Detection:**
   - Switch to the "Sobel Edge Detection" or "Canny Edge Detection" tab.
   - Adjust the parameters if needed and click the corresponding button to apply the algorithm.
   - View the original and processed images side by side.

3. **Perform Image Segmentation:**
   - Switch to the "Thresholding" or "K-Means Clustering" tab.
   - Adjust the parameters if needed and click the corresponding button to apply the algorithm.
   - View the original and segmented images side by side.

4. **Real-Time Processing:**
   - Integrate webcam or video streaming functionality for real-time edge detection and segmentation (feature to be implemented).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to improve the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```