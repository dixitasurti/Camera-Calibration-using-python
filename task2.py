###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
    def calibrateMethod(CHECKERBOARD,worldCoordinates):
        imageCoordinates=[]
        image = imread(imgname)
        gray = cv2.cvtColor(image,COLOR_BGR2GRAY)
        cornersFound, corners = findChessboardCorners(gray,CHECKERBOARD,None)
        if cornersFound==True :
            corners = corners.reshape(-1,2)
            imageCoordinates = corners
            cvImg = cv2.cvtColor(image,COLOR_BGR2GRAY)
            img = cv2.drawChessboardCorners(cvImg, CHECKERBOARD, corners, cornersFound)
            n = len(imageCoordinates)
            A = np.zeros((2*n,12), dtype=np.float32)
            for i in range(n):
                X, Y, Z = worldCoordinates[i]
                x, y = imageCoordinates[i]
                row_1 = np.array([ X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
                row_2 = np.array([ 0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
                A[2*i] = row_1
                A[(2*i)+1] = row_2
            u, s, vt = np.linalg.svd(A)  
            x_vector = vt[-1]
            x_vector = x_vector.reshape(3,4)
            norm = np.linalg.norm(x_vector[-1][:3])
            lambda_scalar = 1/norm
            m = lambda_scalar * x_vector
            m1 = np.asmatrix(m[0][:3]).T
            m2 = np.asmatrix(m[1][:3]).T
            m3 = np.asmatrix(m[2][:3]).T
            Ox = np.dot(m1.T,m3)
            Oy = np.dot(m2.T,m3)
            fx = np.sqrt(np.dot(m1.T,m1)-Ox*Ox)
            fy = np.sqrt(np.dot(m2.T,m2)-Oy*Oy)
            intrinsic = np.array([fx,fy,Ox,Oy])
            intrinsic =  intrinsic.reshape(1,4)
        return intrinsic
    CHECKERBOARD = (4,9)
    worldCoordinates = np.array([[40,0,40], [40, 0, 30], [40, 0,20],[40,0,10],[30,0,40],[30,0,30],[30,0,20],[30,0,10],[20,0,40],
                                    [20,0,30],[20,0,20],[20,0,10],[10,0,40],[10,0,30],[10,0,20],[10,0,10],[0,0,40],[0,0,30],
                                    [0,0,20],[0,0,10],[0,10,40],[0,10,30],[0,10,20],[1,10,10],[0,20,40],[0,20,30],[0,20,20],
                                    [0,20,10],[0,30,40],[0,30,30],[0,30,20],[0,30,10],[0,40,40],[0,40,30],[0,40,20],[0,40,10]])
    intrinsic = calibrateMethod(CHECKERBOARD,worldCoordinates)
    return intrinsic,True

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)