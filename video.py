import numpy as np
import cv2 as cv2
import glob
import pafy
import operator

DIMX=8
DIMY=5
square_size=28

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((DIMY*DIMX,3), np.float32)
objp[:,:2] = np.mgrid[0:DIMX,0:DIMY].T.reshape(-1,2)
objp *= square_size
#print("se crea los puntos origen note la escala que es 1 en este caso aca se multiplica por el tamanio del rectangulo en milimetros")
#print(objp)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (DIMX,DIMY),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #print(corners2)
        imgpoints.append(corners2)
    

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (DIMX,DIMY), corners2,ret)
        cv2.imshow('img',img)
        cv2.imwrite('cal.png',img)
        cv2.waitKey(3)


cv2.destroyAllWindows()
cv2.waitKey(2)
#print(imgpoints)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#print(mtx)
#print(type(mtx))
#print(dist)
#print(type(dist))
# termina la calibración



img2 = cv2.imread('objeto_20.jpg')
h,  w = img2.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))

# undistort
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
#print(dst)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult1.png',dst)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img2,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult2.png',dst)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print ("total error: ", mean_error/len(objpoints))


#print ("matriz inversa ")
mtx_inv=np.linalg.inv(mtx)

cap = cv2.VideoCapture(0)


url = "https://www.youtube.com/watch?v=doLMt10ytHY"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

#captura = cv2.VideoCapture('video.mp4')                 # Video File
#captura = cv2.VideoCapture(0)                           # Webcam 1,2,3
#captura = cv2.VideoCapture('http://192.168.1.68:8081/') # IPCamara

captura = cv2.VideoCapture() #Youtube
captura.open(best.url)


while True:
    ret, frame = cap.read()
    retV, vid = captura.read()
    vid = cv2.resize(vid, (640,480))

    h,  w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    #print(dst)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (DIMX,DIMY),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        #print(corners2)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        #vid = cv2.resize(vid,corners2, interpolation=cv2.INTER_LINEAR)
        # Draw and display the corner
        #frame = cv2.drawChessboardCorners(frame, (DIMX,DIMY), corners2,ret) ### Cambiar
        cv2.waitKey(1)
        img2 = cv2.imread('objeto_20.jpg')
        #print(corners2)
        min = np.amin(corners2, axis=0)
        #print(min)
        max = np.amax(corners2, axis=0)
        #print(max)
        minx = min[0,0]
        miny = min[0,1]
        maxx = max[0,0]
        maxy = max[0,1]
        #print(minx)
        #print(miny)
        #print(maxx)
        #print(maxy)
        minimox = np.where(corners2 == minx)
        minimoy = np.where(corners2 == miny)
        maximox = np.where(corners2 == maxx)
        maximoy = np.where(corners2 == maxy)
        """print(minimox)
        print(minimoy)
        print(maximox)
        print(maximoy)
        print(corners2[minimox[0],0,0])
        print(corners2[minimoy[0],0,0])
        print(corners2[maximox[0],0,0])
        print(corners2[maximoy[0],0,0])
        print(corners2[minimox[0],0,1])
        print(corners2[minimoy[0],0,1])
        print(corners2[maximox[0],0,1])
        print(corners2[maximoy[0],0,1])"""
        pts2 = np.float32([[(corners2[minimoy[0],0,0]), (corners2[minimoy[0],0,1])], [(corners2[maximox[0],0,0]), (corners2[maximox[0],0,1])], [(corners2[minimox[0],0,0]),(corners2[minimox[0],0,1])], [(corners2[maximoy[0],0,0]), (corners2[maximoy[0],0,1])]])
        pts1 = np.float32([[0,0],[640,0],[0,480],[640,480]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        M2 = cv2.getPerspectiveTransform(pts2,pts2)
        mask = np.ones((480,640,3),dtype=np.uint8)*255
        A = cv2.warpPerspective(vid, M, (640, 480)) #Cuadro por dentro
        A2 = cv2.warpPerspective(frame, M2, (640, 480)) 
        A3 = cv2.warpPerspective(mask, M, (640, 480))
        A4 = cv2.bitwise_not(A3)
        A5 = cv2.bitwise_and(A2,A4)
        A6 = A5 + A
        #cv2.imwrite('A.png',A)
        #cv2.imwrite('A2.png',A2)
        #cv2.imwrite('A3.png',A3)
        #cv2.imwrite('A4.png',A4)
        #cv2.imwrite('A5.png',A5)
        #cv2.imwrite('A6.png',A6)


    
    cv2.imshow('frame', A6)

    if cv2.waitKey(24) == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()