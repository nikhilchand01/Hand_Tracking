import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self,
                 mode=False,
                 maxHands=2,
                 modelComplexity=1,
                 detectionConf=0.5,
                 trackingConf=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackingConf
        )

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(RGBimg)

        if self.results.multi_hand_landmarks:

            for handLms in self.results.multi_hand_landmarks:

                if draw:

                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(
                            color=(255, 0, 0),   # Blue landmark points
                            thickness=2,
                            circle_radius=4
                        )
                    )

        return img

    def findPosition(self, img, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:

            for hand in self.results.multi_hand_landmarks:

                for id, lm in enumerate(hand.landmark):

                    h, w, c = img.shape
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    lmList.append([id, cx, cy])

                    if draw:
                        cv2.circle(
                            img,
                            (cx, cy),
                            8,
                            (255, 0, 0),   
                            cv2.FILLED
                        )

        return lmList