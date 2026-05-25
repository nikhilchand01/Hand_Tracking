import cv2 as cv
import time
from hand_detector import HandDetector


def main():
    cap = cv.VideoCapture(0)

    pTime = 0
    detector = HandDetector()

    while True:
        success, img = cap.read()

        if not success:
            break

        # Detect hands
        img = detector.findHands(img)

        # Find landmark positions
        position = detector.findPosition(img)

        if len(position) != 0:
            print(position[4])   # thumb tip

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(
            img,
            f"FPS: {int(fps)}",
            (15, 30),
            cv.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2
        )

        cv.imshow("Hand Tracking", img)

        # Press d to exit
        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()