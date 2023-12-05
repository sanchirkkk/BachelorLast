import cv2
from src.FaceLandmark import *
from src.PoseEstimation import *
from notebooks.turshilt3.FaceNormalization import *
import time


class Camera:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.pTime = 0
        self.frameR = 100

        wCam, hCam = 720, 640

    def capture(self):
        return self.cap


if "__name__" == "__name__":
    camera = Camera()

    landmark1 = FaceLandmark()

    capture = camera.capture()
    landmark = FaceLandmark()
    normal = Normalizatoin()
    pose = PoseEstimation()
    # img_swap = cv2.imread("../data/me5.jpg")

    # results_1 = landmark1.process(img_swap)
    # img_swap = normal.geneate_face_oval_mesh(
    #     results_1.multi_face_landmarks[0].landmark, img_swap
    # )

    while True:
        ret, img = capture.read()
        img = cv2.flip(img, 1)
        if not ret:
            print("lai vee")
            break
        results = landmark.process(img)
        face_found = bool(results.multi_face_landmarks)
        if face_found:
            # img = normal.geneate_face_oval_mesh(
            #     results.multi_face_landmarks[0].landmark, img
            # )
            # image2_resized = cv2.resize(img_swap, (img.shape[1], img.shape[0]))
            # img = landmark.draw_landmarks_on_image(img, results)

            face_features = pose.extract_features(img, results)
            normalized = pose.normalize(face_features)
            pitch_pred, yaw_pred, roll_pred = pose.model.predict(normalized).ravel()
            text = "x: {},y: {}, x: {}".format(pitch_pred, yaw_pred, roll_pred)
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (10, 50)
            font_scale = 1
            font_color = (255, 255, 255)  # White color in BGR
            thickness = 2

            cv2.putText(img, text, position, font, font_scale, font_color, thickness)

            # img = pose.draw_axes(img, pitch_pred, yaw_pred, roll_pred, face_features)

        cv2.imshow("cam", img)
        if cv2.waitKey(1) == ord("x"):
            break
