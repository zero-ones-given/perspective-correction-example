import cv2
import numpy as np

RED = (0, 0, 255) # red in BGR colorspace
WINDOW_NAME = 'Video'
# original coordinates before perspective transformation
originalArucos = {}

# transformed coordinates
arucos = {}

def getAveragePoint(points):
    xSum = ySum = 0
    numberOfPoints = len(points)
    for point in points:
        x, y = point
        xSum += x
        ySum += y
    return np.float32([xSum / numberOfPoints, ySum / numberOfPoints])

def main():
    cv2.namedWindow(WINDOW_NAME)
    # read from webcam
    # capture = cv2.VideoCapture(0)

    # read from network stream
    capture = cv2.VideoCapture('http://localhost:8080')

    # prepare aruco detector
    arucoDictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detectorParameters =  cv2.aruco.DetectorParameters()
    arucoDetector = cv2.aruco.ArucoDetector(arucoDictionary, detectorParameters)

    if capture.isOpened():
        returnValue, frame = capture.read()
    else:
        returnValue = False

    while returnValue:
        # detect aruco markers
        arucoCorners, ids, rejected = arucoDetector.detectMarkers(frame)

        # update aruco positions to a dictionary
        for index, cornerPoints in enumerate(arucoCorners):
            originalArucos[ids[index][0]] = cornerPoints[0]

        # perspective correction
        if (46 in originalArucos and 47 in originalArucos and 48 in originalArucos and 49 in originalArucos):
            # square output
            height = frame.shape[0]
            width = height
            margin = 5
            detectedArenaCorners = [
                originalArucos[46], # top left
                originalArucos[47], # top right
                originalArucos[48], # bottom left
                originalArucos[49] # bottom right
            ]
            sourceMatrix = np.float32(list(map(getAveragePoint, detectedArenaCorners)))
            targetMatrix = np.float32([
                [margin, margin], # top left
                [width - margin, margin], # top right
                [margin, height - margin], # bottom left
                [width - margin, height - margin] # bottom right
            ])

            # apply perspective correction to the frame
            perspectiveMatrix = cv2.getPerspectiveTransform(sourceMatrix, targetMatrix)
            frame = cv2.warpPerspective(frame, perspectiveMatrix, (width, height), cv2.INTER_NEAREST)

            # apply perspective correction to the detected markers
            for id in originalArucos:
                arucos[id] = cv2.perspectiveTransform(np.array([originalArucos[id]]), perspectiveMatrix)[0]

        # draw a circle on each aruco
        for aruco in arucos.values():
            centerPoint = getAveragePoint(aruco)
            x, y = centerPoint
            cv2.circle(frame, [round(x), round(y)], 5, RED, 2)

        # display frame
        cv2.imshow(WINDOW_NAME, frame)
        returnValue, frame = capture.read()

        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow(WINDOW_NAME)
    capture.release()

if __name__ == "__main__": main()