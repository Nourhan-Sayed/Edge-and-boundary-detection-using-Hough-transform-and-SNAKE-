import numpy as np
import matplotlib.pyplot as plt
import cv2
import canny
import matplotlib.lines as mlines
import helpers

def hough_lines(img, threshold = 100):
    h,w = img.shape
    
    canny_img = canny.canny_edge_detector(img,100,200)
    diagonal_dist = int(np.round(np.sqrt(h**2 + w ** 2)))
    thetas = np.deg2rad(np.arange(-90, 90))

    radii = np.linspace(-diagonal_dist, diagonal_dist, 2*diagonal_dist)
    accumulator = np.zeros((2 * diagonal_dist, len(thetas)))
    figure = plt.figure(figsize=(12, 12))
    plot = figure.add_subplot(1, 2, 2)  
    plot2=figure.add_subplot(1,2,1)
    plot.imshow(img, cmap='gray')
    plot2.imshow(img,cmap='gray')
    for i in range(h):
        for j in range(w):
          if canny_img[i][j] > 0:
            for k in range(len(thetas)):
                r = j*np.cos(thetas[k]) + i * np.sin(thetas[k])
                accumulator[int(r) + diagonal_dist,k] += 1
        for i in range(accumulator.shape[0]):
            for j in range(accumulator.shape[1]):
                if accumulator[i][j] > threshold:
                    r = radii[i]
                    theta = thetas[j]
                    a = np.cos((theta))
                    b = np.sin((theta))
                    x0 = (a * r) 
                    y0 = (b * r) 
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000* (a))
              
                    plot.add_line(mlines.Line2D([x0, x1], [y0, y1], color='red'))
    plt.axis('off')
    plt.savefig("static/images/output.jpg",bbox_inches='tight')
    
    return 


import numpy as np
import matplotlib.pyplot as plt
import cv2
import canny
from collections import defaultdict

def hough_circles(path, r_min, r_max, delta_r, thetas_num, bin_threshold):
#   figure = plt.figure(figsize=(12, 12))
#   plot = figure.add_subplot(1, 1, 1)  
#   plot.imshow(image, cmap='gray')
  image = cv2.imread(path,0)

  cv2.imwrite("static/images/output.jpg",image)

  edge_image = canny.canny_edge_detector(image,20,40)
  h, w = edge_image.shape
  theta_step = int(360 / thetas_num)
  thetas = np.arange(0, 360, step=theta_step)
  radii = np.arange(r_min, r_max, step=delta_r)
  circles = []
  for r in radii:
    for t in range(thetas_num):
      circles.append((r, int(r * np.cos(np.deg2rad(thetas))[t]), int(r * np.sin(np.deg2rad(thetas))[t])))
  accumulator = defaultdict(int)
#   accumulator = np.zeros((2 * radii, len(thetas)))
  for y in range(h):
    for x in range(w):
      if edge_image[y][x] != 0:
        for r, rcos_t, rsin_t in circles:
          x0 = x - rcos_t
          y0 = y - rsin_t
          accumulator[(x0, y0, r)] += 1 

  out_circles = []

  for circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
    x, y, r = circle
    votes = votes / thetas_num
    if votes >= bin_threshold: 

      out_circles.append((x, y, r, votes))

   #  threshold = 5
   #  strong_circles = []
   #  for x, y, r, v in out_circles:

   #    if all(abs(x - xc) > threshold or abs(y - yc) > threshold or abs(r - rc) > threshold for xc, yc, rc, v in strong_circles):
   #      strong_circles.append((x, y, r, v))
   #  out_circles = strong_circles
  
  image = cv2.imread(path)
  
  for x, y, r, v in out_circles:
    output_img = cv2.circle(image, (x,y), r, (0,0,255), 2)
    cv2.imwrite("static/images/output.jpg",output_img)
  path = "static/images/output.jpg"
  return path

# def main():
    
    
#    img_path = "static/images/coins.jpg"
   
#    # r_min = 10
#    # r_max = 200
#    # delta_r = 1
#    # thetas_num = 100
#    # bin_threshold = 0.4

#    input_img = cv2.imread(img_path)
    
#    width = int(input_img.shape[1] * 100 / 100)
#    height = int(input_img.shape[0] * 100 / 100)
#    dim = (width, height)
#    resized = cv2.resize(input_img, dim)

#    cv2.imwrite("static/images/resized.jpg",resized)
#    img1 = cv2.imread("static/images/resized.jpg")

#    print ("started")
#    hough_circles("static/images/resized.jpg", 10, 300, 10, 100, 0.4)
#    circle_img = cv2.imread("static/images/output.jpg")
#    cv2.imshow("Uploaded Image", img1)
#    cv2.imshow("Output Image",circle_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    #  plt.axis('off')
#    #  plt.savefig("static/images/output.jpg",bbox_inches='tight')
#    #  plt.show()
#    #  cv2.imshow('Detected Circles', circle_img)
#    #  cv2.waitKey(0)

#    #  if circle_img is not None:
#    #    cv2.imwrite("static/images/output.jpg", circle_img)
   
#    print ("ended")



# if __name__ == "__main__":
#     main()


# img = cv2.imread('static/images/lines.jpg',0)
# print("shapee", img.shape)
# scale_percent = 100 # percent of original size
# if (img.shape[0]>=500 or img.shape[1]>=600):
#    scale_percent = 40
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# # resize image
# resized = cv2.resize(img, dim)
# # resized = imutils.resize(img,width=200)
# plt.imsave("static/images/resized.jpg",resized,cmap='gray')
# img1 = cv2.imread("static/images/resized.jpg",0)
# print("resized",img1.shape)
# hough_lines(img1)


# plt.axis('off')
# plt.savefig("static/images/output.jpg",bbox_inches='tight')
# img2 = cv2.imread("static/images/output.jpg")
# plt.show()
# # cv2.imshow("Output Image",img2)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

