import cv2
import numpy as np
from scipy.ndimage.filters import sobel
from skimage.feature import canny
from skimage import io

def grad_directions(image):
    dx = sobel(image, axis=1)  # axis = 1 => x-axis
    dy = sobel(image, axis=0)  # axis = 0 => y-axis
    grad_directions = np.arctan2(dy, dx) + np.pi
    grad_magnitude = np.sqrt(dx**2 + dy**2)
    return grad_directions, grad_magnitude

def detect_lines():
    low_threshold = 100
    high_threshold = 350
    FilterSize = 5
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=FilterSize, L2gradient=True)

        lines = cv2.HoughLinesP(edges, rho = 1, theta = 1*np.pi/180, threshold = 150, minLineLength = 100, maxLineGap = 100)
        #lines = cv2.HoughLines(edges, rho = 1, theta = 1*np.pi/180, threshold = 150)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('Line Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_lines()

