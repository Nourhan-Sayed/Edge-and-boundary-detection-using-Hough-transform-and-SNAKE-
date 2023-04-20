import numpy as np
import cv2
import streamlit as st


def canny_edge_detection(image, threshold1, threshold2):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1, threshold2)
    return edges

def findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    # Apply Canny edge detection
    edges = canny_edge_detection(image, threshold1=100, threshold2=200)

    # Find contours in the image
    contours = []
    hierarchy = None
    rows, cols = edges.shape
    visited = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if edges[i][j] == 255 and visited[i][j] == 0:
                contour = []
                stack = [(i, j)]
                while len(stack) > 0:
                    y, x = stack.pop()
                    visited[y][x] = 1
                    contour.append((x, y))
                    neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                    for ny, nx in neighbors:
                        if ny >= 0 and ny < rows and nx >= 0 and nx < cols and edges[ny][nx] == 255 and visited[ny][nx] == 0:
                            stack.append((ny, nx))
                if len(contour) > 0:
                    contours.append(np.array(contour))

    return contours, hierarchy

def contour_detection(image, threshold1=100, threshold2=200):
    # Apply Canny edge detection
    # edges = canny_edge_detection(image, threshold1, threshold2)
    # Find contours in the image
    contours, hierarchy = findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on the image
    contours_image = np.zeros_like(image)
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    return contours_image


#############################################################
st.title("Contour Detection")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    # Convert to grayscale
    # gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    # Perform contour detection
    result = contour_detection(image)
    # Display original image and contour detection result
    col1, col2 = st.columns(2)
    col1.header("Original Image")
    col1.image(uploaded_file, use_column_width=True)
    col2.header("Contour Detection")
    col2.image(result, use_column_width=True)
