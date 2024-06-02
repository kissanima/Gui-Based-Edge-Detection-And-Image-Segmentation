# File: gui_image_processing.py

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.geometry("1200x800")
        self.image = None
        self.original_image = None  # To store the original image
        self.processed_image = None
        self.video_stream = False
        self.cap = None
        self.create_widgets()

    def create_widgets(self):
        # Frame for the buttons and canvases
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Button to load image
        load_button = ttk.Button(frame, text="Load Image", command=self.load_image)
        load_button.pack(pady=10)

        # Canvas to display images
        self.original_canvas = tk.Canvas(frame, width=400, height=400, bg='gray')
        self.original_canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.processed_canvas = tk.Canvas(frame, width=400, height=400, bg='gray')
        self.processed_canvas.pack(side=tk.RIGHT, padx=10, pady=10)

        # Add tabs for different processing techniques
        tab_control = ttk.Notebook(self.root)

        # Sobel Edge Detection Tab
        sobel_tab = ttk.Frame(tab_control)
        tab_control.add(sobel_tab, text='Sobel Edge Detection')
        self.create_sobel_tab(sobel_tab)

        # Canny Edge Detection Tab
        canny_tab = ttk.Frame(tab_control)
        tab_control.add(canny_tab, text='Canny Edge Detection')
        self.create_canny_tab(canny_tab)

        # Thresholding Tab
        thresholding_tab = ttk.Frame(tab_control)
        tab_control.add(thresholding_tab, text='Thresholding')
        self.create_thresholding_tab(thresholding_tab)

        # K-Means Clustering Tab
        kmeans_tab = ttk.Frame(tab_control)
        tab_control.add(kmeans_tab, text='K-Means Clustering')
        self.create_kmeans_tab(kmeans_tab)

        # Video Stream Tab
        video_tab = ttk.Frame(tab_control)
        tab_control.add(video_tab, text='Video Stream')
        self.create_video_tab(video_tab)

        tab_control.pack(expand=1, fill='both')

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.original_image = self.image.copy()  # Store the original image
            self.display_image(self.image, self.original_canvas)
            self.display_image(self.image, self.processed_canvas)

    def display_image(self, image, canvas):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=image)
        canvas.image = image

    def create_sobel_tab(self, parent):
        sobel_apply_button = ttk.Button(parent, text="Apply Sobel Operator", command=self.apply_sobel)
        sobel_apply_button.pack(pady=10)
        
        sobel_remove_button = ttk.Button(parent, text="Remove Sobel Operator", command=self.remove_sobel)
        sobel_remove_button.pack(pady=10)

    def apply_sobel(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded")
            return

        sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(sobel)
        self.processed_image = sobel  # Store the processed image
        self.display_image(sobel, self.processed_canvas)

    def remove_sobel(self):
        if self.original_image is not None:
            self.display_image(self.original_image, self.processed_canvas)
            self.processed_image = self.original_image.copy()  # Reset to original image

    def create_canny_tab(self, parent):
        self.canny_threshold1 = tk.IntVar(value=100)
        self.canny_threshold2 = tk.IntVar(value=200)

        ttk.Label(parent, text="Threshold1").pack(pady=5)
        threshold1_slider = ttk.Scale(parent, from_=0, to=500, orient=tk.HORIZONTAL, variable=self.canny_threshold1)
        threshold1_slider.pack(pady=5)

        ttk.Label(parent, text="Threshold2").pack(pady=5)
        threshold2_slider = ttk.Scale(parent, from_=0, to=500, orient=tk.HORIZONTAL, variable=self.canny_threshold2)
        threshold2_slider.pack(pady=5)

        canny_apply_button = ttk.Button(parent, text="Apply Canny Edge Detector", command=self.apply_canny)
        canny_apply_button.pack(pady=10)
        
        canny_remove_button = ttk.Button(parent, text="Remove Canny Edge Detector", command=self.remove_canny)
        canny_remove_button.pack(pady=10)

    def apply_canny(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded")
            return

        threshold1 = self.canny_threshold1.get()
        threshold2 = self.canny_threshold2.get()
        canny = cv2.Canny(self.image, threshold1, threshold2)
        self.processed_image = canny  # Store the processed image
        self.display_image(canny, self.processed_canvas)

    def remove_canny(self):
        if self.original_image is not None:
            self.display_image(self.original_image, self.processed_canvas)
            self.processed_image = self.original_image.copy()  # Reset to original image

    def create_thresholding_tab(self, parent):
        self.threshold_value = tk.IntVar(value=127)

        ttk.Label(parent, text="Threshold Value").pack(pady=5)
        threshold_slider = ttk.Scale(parent, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.threshold_value)
        threshold_slider.pack(pady=5)

        global_thresh_apply_button = ttk.Button(parent, text="Apply Global Thresholding", command=self.apply_global_threshold)
        global_thresh_apply_button.pack(pady=10)
        
        global_thresh_remove_button = ttk.Button(parent, text="Remove Global Thresholding", command=self.remove_threshold)
        global_thresh_remove_button.pack(pady=10)

        adaptive_thresh_apply_button = ttk.Button(parent, text="Apply Adaptive Thresholding", command=self.apply_adaptive_threshold)
        adaptive_thresh_apply_button.pack(pady=10)
        
        adaptive_thresh_remove_button = ttk.Button(parent, text="Remove Adaptive Thresholding", command=self.remove_threshold)
        adaptive_thresh_remove_button.pack(pady=10)

    def apply_global_threshold(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded")
            return

        _, thresh = cv2.threshold(self.image, self.threshold_value.get(), 255, cv2.THRESH_BINARY)
        self.processed_image = thresh  # Store the processed image
        self.display_image(thresh, self.processed_canvas)

    def apply_adaptive_threshold(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded")
            return

        thresh = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.processed_image = thresh  # Store the processed image
        self.display_image(thresh, self.processed_canvas)

    def remove_threshold(self):
        if self.original_image is not None:
            self.display_image(self.original_image, self.processed_canvas)
            self.processed_image = self.original_image.copy()  # Reset to original image

    def create_kmeans_tab(self, parent):
        self.kmeans_clusters = tk.IntVar(value=2)

        ttk.Label(parent, text="Number of Clusters").pack(pady=5)
        clusters_slider = ttk.Scale(parent, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.kmeans_clusters)
        clusters_slider.pack(pady=5)

        kmeans_apply_button = ttk.Button(parent, text="Apply K-Means Clustering", command=self.apply_kmeans)
        kmeans_apply_button.pack(pady=10)
        
        kmeans_remove_button = ttk.Button(parent, text="Remove K-Means Clustering", command=self.remove_kmeans)
        kmeans_remove_button.pack(pady=10)

    def apply_kmeans(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded")
            return

        Z = self.image.reshape((-1, 1))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = self.kmeans_clusters.get()
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        segmented_image = res.reshape((self.image.shape))
        self.processed_image = segmented_image  # Store the processed image
        self.display_image(segmented_image, self.processed_canvas)

    def remove_kmeans(self):
        if self.original_image is not None:
            self.display_image(self.original_image, self.processed_canvas)
            self.processed_image = self.original_image.copy()  # Reset to original image

    def create_video_tab(self, parent):
        start_button = ttk.Button(parent, text="Start Video Stream", command=self.start_video_stream)
        start_button.pack(pady=10)

        stop_button = ttk.Button(parent, text="Stop Video Stream", command=self.stop_video_stream)
        stop_button.pack(pady=10)

        self.video_canvas = tk.Canvas(parent, width=400, height=400, bg='gray')
        self.video_canvas.pack(pady=10)

        self.video_processing_mode = tk.StringVar(value="None")
        ttk.Radiobutton(parent, text="None", variable=self.video_processing_mode, value="None").pack(pady=5)
        ttk.Radiobutton(parent, text="Sobel Edge Detection", variable=self.video_processing_mode, value="Sobel").pack(pady=5)
        ttk.Radiobutton(parent, text="Canny Edge Detection", variable=self.video_processing_mode, value="Canny").pack(pady=5)
        ttk.Radiobutton(parent, text="K-Means Clustering", variable=self.video_processing_mode, value="KMeans").pack(pady=5)

    def start_video_stream(self):
        self.video_stream = True
        self.cap = cv2.VideoCapture(0)
        self.update_video_stream()

    def stop_video_stream(self):
        self.video_stream = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_canvas.delete("all")

    def update_video_stream(self):
        if not self.video_stream:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video_stream()
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.video_processing_mode.get() == "Sobel":
            sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
            processed_frame = cv2.magnitude(sobelx, sobely)
            processed_frame = np.uint8(processed_frame)
        elif self.video_processing_mode.get() == "Canny":
            processed_frame = cv2.Canny(gray_frame, 100, 200)
        elif self.video_processing_mode.get() == "KMeans":
            Z = gray_frame.reshape((-1, 1))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 2  # Example cluster count
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            res = centers[labels.flatten()]
            processed_frame = res.reshape((gray_frame.shape))
        else:
            processed_frame = gray_frame

        self.display_video_frame(processed_frame, self.video_canvas)
        self.root.after(10, self.update_video_stream)

    def display_video_frame(self, frame, canvas):
        # Clear the existing content on the canvas
        canvas.delete("all")

        # Convert frame to PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = Image.fromarray(frame)

        # Resize the frame to fit the canvas
        frame = frame.resize((400, 400), Image.ANTIALIAS)

        # Convert PIL Image to Tkinter PhotoImage
        frame_tk = ImageTk.PhotoImage(frame)

        # Display the frame on the canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
        canvas.image = frame_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
