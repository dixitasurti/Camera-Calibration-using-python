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
            #print("imageCoordinates",imageCoordinates)
            #print("imageCoordinates",imageCoordinates.shape)
            cvImg = cv2.cvtColor(image,COLOR_BGR2GRAY)
            img = cv2.drawChessboardCorners(cvImg, CHECKERBOARD, corners, cornersFound)
            cv2.imwrite("new_img.jpg", image)
            cv2.imshow('image',image)
            n = len(imageCoordinates)
            A = np.zeros((2*n,12), dtype=np.float32)
            #print("Shape of Matrix A : ",n, A.shape)
            for i in range(n):
                X, Y, Z = worldCoordinates[i]
                x, y = imageCoordinates[i]
                row_1 = np.array([ X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
                row_2 = np.array([ 0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
                A[2*i] = row_1
                A[(2*i)+1] = row_2
            u, s, vt = np.linalg.svd(A)  
            #print("vt rotation ",vt)
            #print("u : ",u.shape,"s : ",s.shape," vt : ",vt.shape)
            x_vector = vt[-1]
            #print("45678rtyu",np.dot(A,x_vector))
            x_vector = x_vector.reshape(3,4)
            #print("reshaped x_vector",x_vector)
            norm = np.linalg.norm(x_vector[-1][:3])
            #print(x_vector[-1][:3])
            #print("norm",norm)
            lambda_scalar = 1/norm
            #print("lambda_scalar",lambda_scalar)
            m = lambda_scalar * x_vector
            #print("verify",np.dot(m,worldCoordinates))
            #print("m",m)
            m1 = np.asmatrix(m[0][:3]).T
            m2 = np.asmatrix(m[1][:3]).T
            m3 = np.asmatrix(m[2][:3]).T
            Ox = np.dot(m1.T,m3)
            Oy = np.dot(m2.T,m3)
            #print("Ox,Oy",Ox,Oy)
            fx = np.sqrt(np.dot(m1.T,m1)-Ox*Ox)
            fy = np.sqrt(np.dot(m2.T,m2)-Oy*Oy)
            #print("fx",fx)
            #print("fy",fy)
            intrinsic = np.array([fx,fy,Ox,Oy])
            intrinsic =  intrinsic.reshape(1,4)
            #print(intrinsic)
            #u :  (72, 72) s :  (1,12)  vt :  (12, 12)
        return intrinsic
    CHECKERBOARD = (4,9)
    worldCoordinates = np.array([[40,0,40], [40, 0, 30], [40, 0,20],[40,0,10],[30,0,40],[30,0,30],[30,0,20],[30,0,10],[20,0,40],
                                    [20,0,30],[20,0,20],[20,0,10],[10,0,40],[10,0,30],[10,0,20],[10,0,10],[0,0,40],[0,0,30],
                                    [0,0,20],[0,0,10],[0,10,40],[0,10,30],[0,10,20],[1,10,10],[0,20,40],[0,20,30],[0,20,20],
                                    [0,20,10],[0,30,40],[0,30,30],[0,30,20],[0,30,10],[0,40,40],[0,40,30],[0,40,20],[0,40,10]])
    intrinsic = calibrateMethod(CHECKERBOARD,worldCoordinates)
    worldCoordinates_2 = np.array([[40,0,40], [40, 0, 30], [40, 0,20],[40,0,10],[30,0,40],[30,0,30],[30,0,20],[30,0,10],[20,0,40],
                                    [20,0,30],[20,0,20],[20,0,10],[10,0,40],[10,0,30],[10,0,20],[10,0,10],[0,0,40],[0,0,30],
                                    [0,0,20],[0,0,10],[0,10,40],[0,10,30],[0,10,20],[1,10,10],[0,20,40],[0,20,30],[0,20,20],
                                    [0,20,10],[0,30,40],[0,30,30],[0,30,20],[0,30,10]])
    intrinsic_2 = calibrateMethod((4,8),worldCoordinates_2)
    is_constant = True
    for i in range(1):
        for j in range(4):
            if(is_constant):
                if(intrinsic[i][j]!=intrinsic_2[i][j]):
                    is_constant= False
                    break
            
    return intrinsic,is_constant

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)