#Steps
#1 Read and show video stream,capture images
#2 Detect faces and show bounding box
#3 Flatten the largest face image and save in numpy array
#4 Repeat the above for multiple people to generate training data
#5 map the predicted id to name of the user
#6 Display the predictions on the screen -bounding the box and name


import cv2
import numpy as np

#Initilize Camera

cap=cv2.VideoCapture(0)

#Face detection
face_cascade=cv2.CascadeClassifier("harrcascade_frontalface_alt.xml")

skip=0
face_data=[]
dataset_path='./data/'


file_name=input("Enter name of person: ")
while True:
    ret,frame=cap.read()
    #If frame not found then capture again
    if ret==False:
        continue

    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    

    #faces is a list of tuples of the form x,y,w,h x and y are starting coordinates of image and w and h are width and height
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])  #We are sorting the images according to the width * height ie., the total, area of the image

    #Pick the last face which is the largest since array is sorted by lambda function
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #Extract (Crop the region of interest) :region of interest
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Frame",frame)
    cv2.imshow("Face Section",face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):  #Check if pressed key is q
        break

#Convert our face list array into a numpy array

face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save data into file system

np.save(dataset_path+file_name+'.npy',face_data)
print("Data successfully saved in "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()