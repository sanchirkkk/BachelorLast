from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

import pandas as pd


class Normalizatoin:
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_oval = self.mp_face_mesh.FACEMESH_FACE_OVAL
        self.landmark_connecter = pd.DataFrame(
            list(self.face_oval), columns=["p1", "p2"]
        )
        self.left_eye = [
            [226, 247],
            [247, 30],
            [30, 29],
            [29, 27],
            [27, 28],
            [28, 56],
            [56, 190],
            [190, 244],
            [244, 112],
            [112, 26],
            [26, 22],
            [22, 23],
            [23, 24],
            [24, 110],
            [110, 25],
            [25, 226],
        ]
        self.right_eye = [
            [463, 414],
            [414, 286],
            [286, 258],
            [258, 257],
            [257, 259],
            [259, 260],
            [260, 467],
            [467, 446],
            [446, 255],
            [255, 339],
            [339, 254],
            [254, 253],
            [253, 252],
            [252, 256],
            [256, 341],
            [341, 463],
        ]

        self.face_oval_matrix = self.face_oval_point_connecter()

        print(self.face_oval_matrix)

    def face_oval_point_connecter(self):
        routes_idx = []
        p1 = self.landmark_connecter.iloc[0]["p1"]
        p2 = self.landmark_connecter.iloc[0]["p2"]

        for i in range(0, self.landmark_connecter.shape[0]):
            obj = self.landmark_connecter[self.landmark_connecter["p1"] == p2]
            p1 = obj["p1"].values[0]
            p2 = obj["p2"].values[0]
            route_idx = []
            route_idx.append(p1)
            route_idx.append(p2)
            routes_idx.append(route_idx)
        return routes_idx

    def find_oval(self, point_connection, landmarked_face, w_img, h_img):
        routes = []
        # print(landmarked_face)
        for source_idx, target_idx in point_connection:
            source = landmarked_face[source_idx]
            target = landmarked_face[target_idx]

            relative_source = (
                int(w_img * source.x),
                int(h_img * source.y),
            )
            relative_target = (
                int(w_img * target.x),
                int(h_img * target.y),
            )
            routes.append(relative_source)
            routes.append(relative_target)
        return routes

    def face_image_mesh(self, coordinat, landmark, width, height, img):
        mask = np.zeros((height, width))
        c_with_mesh = self.find_oval(coordinat, landmark, width, height)
        mask = cv2.fillConvexPoly(mask, np.array(c_with_mesh), 1)
        mask = mask.astype(bool)
        out_full = np.zeros_like(img)
        out_full[mask] = img[mask]
        return out_full[:, :, :]

    def geneate_face_oval_mesh(self, landmarked_face, img):
        height, width, img_c = img.shape
        # width = img.shape[1]

        face = self.face_image_mesh(
            self.face_oval_matrix, landmarked_face, width, height, img
        )
        l_eye = self.face_image_mesh(self.left_eye, landmarked_face, width, height, img)
        r_eye = self.face_image_mesh(
            self.right_eye, landmarked_face, width, height, img
        )
        # return face - l_eye - r_eye
        return face

    # def change(self,swap,cam_im):
