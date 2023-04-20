import cv2
import numpy as np
from helpers import convolve, round_angle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



def sobel_filter(img):
    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gx_flipped = np.flip(Gx)
    Gy_flipped = np.flip(Gy)

    sobel_x = convolve(img, Gx_flipped)
    sobel_y = convolve(img, Gy_flipped)


    sobel_total = np.sqrt(sobel_x**2 + sobel_y**2)
    # sobel_total = np.abs(sobel_x) + np.abs(sobel_y)

    angles = np.arctan2(sobel_y, sobel_x)
    plt.imsave("static/images/output.jpg", sobel_total, cmap="gray")

    return img, angles, sobel_x, sobel_y, sobel_total, "Sobel"

def supression(img, D):
    row, col = img.shape
    z = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            # find neighbour pixels to visit from the gradient directions
            loc = round_angle(D[i, j])
            try:
                if loc == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        z[i, j] = img[i, j]
                elif loc == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        z[i, j] = img[i, j]
                elif loc == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (
                        img[i, j] >= img[i + 1, j + 1]
                    ):
                        z[i, j] = img[i, j]
                elif loc == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (
                        img[i, j] >= img[i + 1, j - 1]
                    ):
                        z[i, j] = img[i, j]
            except IndexError as e:
                pass
    return z


def threshold(img, t, T):
    cf = {
        "WEAK": 100,
        "STRONG": 255,
    }

    strong_i, strong_j = np.where(img > T)

    weak_i, weak_j = np.where((img >= t) & (img <= T))

    zero_i, zero_j = np.where(img < t)

    img[strong_i, strong_j] = cf.get("STRONG")
    img[weak_i, weak_j] = cf.get("WEAK")
    img[zero_i, zero_j] = 0

    return (img, cf.get("WEAK"))


def tracking(img, weak, strong=255):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j] == weak:
                try:
                    if (
                        (img[i + 1, j] == strong)
                        or (img[i - 1, j] == strong)
                        or (img[i, j + 1] == strong)
                        or (img[i, j - 1] == strong)
                        or (img[i + 1, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong)
                
                    ):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def canny_edge_detector(image, t, T):
    print("CANNY STARTED")
    img = gaussian_filter(image,sigma=1.4)
    img, D, sobel_x, sobel_y, sobel_total, name = sobel_filter(img)
    img = supression(sobel_total, D)
    img, weak = threshold(img, t, T)
    img = tracking(img, weak)
    plt.imsave("static/images/output.jpg", img, cmap="gray")

    return img


# img = cv2.imread("static/images/lena.png",0)
# canny = canny_edge_detector(img, 20 , 40)
# cv2.imshow('canny', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
