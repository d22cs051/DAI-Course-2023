# Importing all necessary libraries
import cv2
import os

# Read the video from specified path

def extract_frames(src:str,num:int):
    cam = cv2.VideoCapture(src)

    try:
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')

    # frame
    currentframe = 0

    while(True):
        
        # reading from frame
        ret,frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = f'./data/gen_vid_frames/vid_{num}_frame{str(currentframe)}.jpg'
            print ('Creating...' + name)
            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

for i in range(1,65):
    try:
        extract_frames(f"data/gen_vid/output_video{i}.mp4",i)
    except:
        print(f"Video Not Exists: video{i}.mp4")