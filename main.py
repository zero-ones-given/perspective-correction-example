import cv2
import numpy as np

RED = (0, 0, 255) # red in BGR colorspace
WINDOW_NAME = 'Video'

def getAveragePoint(points):
    xSum = ySum = 0
    numberOfPoints = len(points)
    for point in points:
        x, y = point
        xSum += x
        ySum += y
    return np.float32([xSum / numberOfPoints, ySum / numberOfPoints])

def drawCircleOnArucos(frame, arucos):
    for aruco in arucos.values():
        centerPoint = getAveragePoint(aruco)
        x, y = centerPoint
        cv2.circle(frame, [round(x), round(y)], 5, RED, 2)

def applyPerspectiveCorrection(frame, arucos, topLeftId, topRightId, bottomLeftId, bottomRightId):
    if (topLeftId not in arucos or topRightId not in arucos or bottomLeftId not in arucos or bottomRightId not in arucos):
        return (frame, arucos)

    # square output
    height = frame.shape[0]
    width = height
    margin = 5
    detectedArenaCorners = [
        arucos[topLeftId],
        arucos[topRightId],
        arucos[bottomLeftId],
        arucos[bottomRightId]
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
    transformedFrame = cv2.warpPerspective(frame, perspectiveMatrix, (width, height), cv2.INTER_NEAREST)

    # apply perspective correction to the detected markers
    transformedArucos = {}
    for id in arucos:
        transformedArucos[id] = cv2.perspectiveTransform(np.array([arucos[id]]), perspectiveMatrix)[0]

    return (transformedFrame, transformedArucos)

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

    # original coordinates before perspective transformation get persisted in this dictionary.
    # If some of them flicker, the transformation stays consistent this way.
    originalArucos = {}

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
        frame, arucos = applyPerspectiveCorrection(frame, originalArucos, 46, 47, 48, 49)

        drawCircleOnArucos(frame, arucos)

        # display frame
        cv2.imshow(WINDOW_NAME, frame)
        returnValue, frame = capture.read()

        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow(WINDOW_NAME)
    capture.release()

if __name__ == "__main__": main()