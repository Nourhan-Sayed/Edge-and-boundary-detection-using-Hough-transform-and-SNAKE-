import matplotlib.pyplot as plt
import snake
import numpy as np
import cv2
import numpy as np
import numpy as np

def chain_code (contourPoints):
    chainCode=[]
    code=0
    size=int(np.size(contourPoints)/2)
    for i in range(0,int (size)):
        x=contourPoints[i][0]
        y=contourPoints[i][1]
        x_1=contourPoints[(i+1)%size][0]
        y_1=contourPoints[(i+1)%size][1]
        dx=x_1-x
        dy=y_1-y
        if (dx==0):
            if dy>0:
                code=2
            else :
                code=6
        else:
            slope=dy/dx
            angle=np.degrees(np.arctan(slope))
            print (angle)
            if angle>=-22.5 and angle<22.5:
                code =0
            elif angle>=22.5 and angle <67.5:
                code =1
            elif angle >=112.5 and angle <157.5:
                code =3
            elif angle >157.5 and angle <-157.5:
                code =4
            elif angle >-157.5 and angle <-112.5:
                code =5
            elif angle >=-112.5 and angle <-67.5:
                code =6
            else :
                code =7
            chainCode.append (code)


            # slope=dy/dx
            # if slope>-0.5 and slope<0.5:
            #     code=0
            # elif slope

    return chainCode


def snakes(image,x,y,z):
    img = snake.Snake( image, closed = True )
    img.set_alpha(x)
    img.set_beta(y)
    img.set_gamma(z)
    for i in range(500):
        snakeImg = img.visualize()
        x = []
        y = []
        for i in range(len(img.points)):
            x.append(img.points[i][0])
            y.append(img.points[i][1])
        area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
        area=np.abs(area)
        perimeter = img.get_length()
        snake_changed = img.step()
    plt.imsave("static/images/output.jpg",snakeImg)
    # plt.show()
    print("area=",area)
    print("............")
    print("perimeter=",perimeter)
    print("size",np.size(img.points))
    # print("end of function")
    return img.points,area,perimeter
cv2.destroyAllWindows()
