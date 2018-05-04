import numpy as np
import dlib
import cv2
import get_points
import argparse as ap

def run(source=0, dispLoc=False):
    #This will load the camera object that we want for tracking
    cam = cv2.VideoCapture(source)
    
    if not cam.isOpened():
        print "Video device or file couldn't be opened"
        exit()
    
    print "Press key `p` to pause the video to start tracking"
    while True:
        retval, img = cam.read()
        if not retval:
            print "Cannot capture frame device"
            exit()
        if(cv2.waitKey(10)==ord('p')):
            break
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
    cv2.destroyWindow("Image")
    #The points of the moving rectangle are stored in points and gotten thru a method get_points
    points = get_points.run(img, multi=True)
    if not points:
        print "ERROR: No object to be tracked."
        exit()
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    # Initial co-ordinates of the object to be tracked
    tracker = [dlib.correlation_tracker() for _ in xrange(len(points))]
    # Provide the tracker the initial position of the object
    [tracker[i].start_track(img, dlib.rectangle(*rect)) for i, rect in enumerate(points)]

    while True:
    #Read frame from device or file
    # retval, img = cam.read()
    #  if not retval:
    # print "Cannot capture frame device | CODE TERMINATION :( "
    #  exit()
        # Update the tracker
        for i in xrange(len(tracker)):
            tracker[i].update(img)
            # Get the position of th object, draw a
            # bounding box around it and display it.
            rect = tracker[i].get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
            # Continue until the user presses ESC key
        if cv2.waitKey(1) == 27:
            break

        # Relase the VideoCapture object
        #cam.release()

        #HEREcap = cv2.VideoCapture('tennis.mp4')

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                              qualityLevel = 0.3,
                              minDistance = 7,
                              blockSize = 7 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                         maxLevel = 2,
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        # Take first frame and find corners in it
        ret, old_frame = cam.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while True:
            ret,frame = cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)

            for i in xrange(len(tracker)):
                tracker[i].update(img)
                # Get the position of th object, draw a
                # bounding box around it and display it.
                rect = tracker[i].get_position()
                pt1 = (int(rect.left()), int(rect.top()))
                pt2 = (int(rect.right()), int(rect.bottom()))
                cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", img)

            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cam.release()

if __name__ == "__main__":
    # Parse command line arguments
    parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', "--deviceID", help="Device ID")
    group.add_argument('-v', "--videoFile", help="Path to Video File")
    parser.add_argument('-l', "--dispLoc", dest="dispLoc", action="store_true")
    args = vars(parser.parse_args())
    
    # Get the source of video
    if args["videoFile"]:
        source = args["videoFile"]
    else:
        source = int(args["deviceID"])
    run(source, args["dispLoc"])
