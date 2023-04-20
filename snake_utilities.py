import numpy as np

def ConvertToGaryscale(ColoredImage):
    
    Red, Green, Blue = ColoredImage[:,:,0], ColoredImage[:,:,1], ColoredImage[:,:,2]
    GrayscaleImage = 0.299 * Red + 0.587 * Green + 0.114 * Blue

    return GrayscaleImage

def Convolve(Image,Gx,Gy=np.zeros((3,3))):
    if len(Image.shape) == 3:
        Image = ConvertToGaryscale(Image)

    ImageNumberOfRows,ImageNumberOfColumns=Image.shape
    
    KernalNumberOfRows, KernalNumberOfColumns = Gx.shape
    PaddedHight = int((KernalNumberOfRows - 1) / 2)
    PaddedWidth = int((KernalNumberOfColumns - 1) / 2)
    PaddedImage = np.zeros((ImageNumberOfRows + (2 * PaddedHight), ImageNumberOfColumns + (2 * PaddedWidth)))
    PaddedImage[ PaddedHight : PaddedImage.shape[0] - PaddedHight,PaddedWidth : PaddedImage.shape[1] - PaddedWidth,] = Image
    ResultantImage = np.ones([ImageNumberOfRows,ImageNumberOfColumns]) 

    for row in range(ImageNumberOfRows):
        for column in range(ImageNumberOfColumns):

            if(Gy.any() == 0):
                ResultantImage[row,column]=np.sum(np.multiply(Gx,PaddedImage[row:row+KernalNumberOfRows,column:column+KernalNumberOfColumns]))  
            else: 
                PixelValueX=np.sum(np.multiply(Gx,PaddedImage[row:row+KernalNumberOfRows,column:column+KernalNumberOfColumns])) 
                PixelValueY=np.sum(np.multiply(Gy,PaddedImage[row:row+KernalNumberOfRows,column:column+KernalNumberOfColumns]))
                ResultantImage[row,column]=np.sqrt(PixelValueX**2+PixelValueY**2)
              
    return ResultantImage
    



def Sobel(image):
    SobelKernalX=np.array([[-1,0,1],
                           [-2,0,2],
                           [-1,0,1]])                           
    SobelKernalY=np.flip(SobelKernalX.transpose())
    SobelImageX=Convolve(image,SobelKernalX)
    SobelImageY= Convolve(image,SobelKernalY)
    return SobelImageX,SobelImageY

def local_thresh(input_img,T):

    h, w = input_img.shape##rows and columns

    S = w/8
    s2 = S/2
    #integral img
    #we use 32 bit because
    #The values in the integral image get very 
    #large because they are the sums of the pixels above and to the left
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row,col] = input_img[0:row,0:col].sum()

    #output img
    out_img = np.zeros_like(input_img)    

    for col in range(w):
        for row in range(h):
            #SxS region
            y0 = int(max(row-s2, 0))
            y1 = int(min(row+s2, h-1))
            x0 = int(max(col-s2, 0))
            x1 = int(min(col+s2, w-1))

            count = int((y1-y0)*(x1-x0))

            sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]

            if input_img[row, col]*count < sum_*(100.-T)/100.:
                out_img[row,col] = 0
            else:
                out_img[row,col] = 255
    ##the result
    return out_img
