import cv2
from matplotlib import pyplot as plt
import pickle
import glob
import pandas as pd
import numpy as np
import sys


class PoseEstimation:
    def __init__(self) -> None:
        try:
            self.model = pickle.load(open("src/models/model.pkl", "rb"))

        except:
            print("Model unshihad aldaa garlaa")
            sys.exit(1)

    def column(self):
        cols = []
        for pos in [
            "nose_",
            "forehead_",
            "left_eye_",
            "mouth_left_",
            "chin_",
            "right_eye_",
            "mouth_right_",
        ]:
            for dim in ("x", "y"):
                cols.append(pos + dim)

        return cols

    def extract_features(self, img, result):
        NOSE = 1
        FOREHEAD = 10
        LEFT_EYE = 33
        MOUTH_LEFT = 61
        CHIN = 199
        RIGHT_EYE = 263
        MOUTH_RIGHT = 291

        face_features = []

        if result.multi_face_landmarks != None:
            face_landmarks = result.multi_face_landmarks[0]
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [
                    FOREHEAD,
                    NOSE,
                    MOUTH_LEFT,
                    MOUTH_RIGHT,
                    CHIN,
                    LEFT_EYE,
                    RIGHT_EYE,
                ]:
                    face_features.append(lm.x)
                    face_features.append(lm.y)

        return face_features

    def draw_axes(self, img, pitch, yaw, roll, extrected, size=50):
        img_h, img_w, img_c = img.shape
        face_features_df = pd.DataFrame([extrected], columns=self.column())
        nose_x = face_features_df["nose_x"].values * img_w
        nose_y = face_features_df["nose_y"].values * img_h

        yaw = -yaw
        rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(
            np.float64
        )
        axes_points = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64
        )
        axes_points = rotation_matrix @ axes_points
        axes_points = (axes_points[:2, :] * size).astype(int)
        axes_points[0, :] = axes_points[0, :] + nose_x
        axes_points[1, :] = axes_points[1, :] + nose_y

        new_img = img.copy()
        cv2.line(
            new_img,
            tuple(axes_points[:, 3].ravel()),
            tuple(axes_points[:, 0].ravel()),
            (255, 0, 0),
            3,
        )
        cv2.line(
            new_img,
            tuple(axes_points[:, 3].ravel()),
            tuple(axes_points[:, 1].ravel()),
            (0, 255, 0),
            3,
        )
        cv2.line(
            new_img,
            tuple(axes_points[:, 3].ravel()),
            tuple(axes_points[:, 2].ravel()),
            (0, 0, 255),
            3,
        )
        return new_img

    def normalize(self, face_features):
        poses_df = pd.DataFrame([face_features], columns=self.column())
        normalized_df = poses_df.copy()

        for dim in ["x", "y"]:
            # Centerning around the nose
            for feature in [
                "forehead_" + dim,
                "nose_" + dim,
                "mouth_left_" + dim,
                "mouth_right_" + dim,
                "left_eye_" + dim,
                "chin_" + dim,
                "right_eye_" + dim,
            ]:
                normalized_df[feature] = poses_df[feature] - poses_df["nose_" + dim]

            # Scaling
            diff = (
                normalized_df["mouth_right_" + dim] - normalized_df["left_eye_" + dim]
            )
            for feature in [
                "forehead_" + dim,
                "nose_" + dim,
                "mouth_left_" + dim,
                "mouth_right_" + dim,
                "left_eye_" + dim,
                "chin_" + dim,
                "right_eye_" + dim,
            ]:
                normalized_df[feature] = normalized_df[feature] / diff

        return normalized_df
