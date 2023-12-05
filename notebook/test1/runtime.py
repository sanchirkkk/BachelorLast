import cv2
from FaceLandmark import *

# from PoseEstimation import *

from Transform import *
import time
from PoseEstimation import *
from Filter import *


class Camera:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.pTime = 0
        self.frameR = 100

        wCam, hCam = 720, 640

    def capture(self):
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        return self.cap


if "__name__" == "__name__":
    ###
    camera = Camera()
    capture = camera.capture()
    ##
    #
    #
    #
    ##

    normal = "mean"

    first_time = True
    # latest = np.zeros((800, 600, 3), dtype=np.uint8)
    landmark = FaceLandmark()
    landmark1 = FaceLandmark()

    transform = Transform()
    pose = PoseEstimation()
    swapping = MakeFilter()

    lists = [
        "l1",
        "l2",
        "l3",
        "l4",
        "l5",
        "l6",
        "l7",
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
        "r6",
        "r7",
        "n",
    ]
    all_means = []
    for i in lists:
        save_data_dir = "../../result/mean_1/mean_{}.npy".format(i)
        all_means.append(np.array(np.load(save_data_dir)))
    while True:
        ret, img = capture.read()
        img = cv2.flip(img, 1)
        if not ret:
            print("lai vee")
            break

        results = landmark.process(img)

        face_found = bool(results.multi_face_landmarks)
        if face_found:
            processed_image = landmark.process(img)
            # Толгойн чиглэлийг тооцох
            face_features = pose.extract_features(img, processed_image)
            normalized = pose.normalize(face_features)
            pitch_pred, yaw_pred, roll_pred = pose.model.predict(normalized).ravel()
            ind = pose.pose_calculate_y(yaw_pred)
            face_mean_image = cv2.cvtColor(all_means[ind], cv2.COLOR_BGR2RGB)

            image1_shape = face_mean_image.shape
            image2_shape = (600, 800, 3)

            image1 = face_mean_image
            image2 = np.zeros(image2_shape, dtype=np.uint8)

            # Calculate the center coordinates for placing image1 onto image2
            center_x = (image2_shape[1] - image1_shape[1]) // 2
            center_y = (image2_shape[0] - image1_shape[0]) // 2

            # Place image1 onto image2 at the center
            image2[
                center_y : center_y + image1_shape[0],
                center_x : center_x + image1_shape[1],
                :,
            ] = face_mean_image
            processed_image_face = landmark1.process(image2)
            (
                a,
                b,
                c,
            ) = swapping.process(img, results, image2, processed_image_face)

            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (10, 50)
            font_scale = 1
            font_color = (255, 255, 255)  # White color in BGR
            thickness = 2

            cv2.putText(a, str(ind), position, font, font_scale, font_color, thickness)
            cv2.imshow("cam1", a)

        # cv2.imshow("cam_face", c)

        if cv2.waitKey(1) == ord("x"):
            break
