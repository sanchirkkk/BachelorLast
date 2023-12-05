from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


class FaceLandmark:
    # Init
    def __init__(self) -> None:
        self.base_options = python.BaseOptions(
            model_asset_path="src/data/face_landmarker_v2_with_blendshapes.task"
        )

        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            # output_face_blendshapes=True,
        )
        self.face_oval = self.mp_face_mesh.FACEMESH_FACE_OVAL

    def face_oval_(self):
        return self.face_oval

    def process(self, img):
        return self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def land_marks(self, img):
        processed = self.process(img)
        return processed.multi_face_landmarks[0]

    def faceMesh(self, img):
        mesh = mp.solutions.face_mesh
        face_mesh = mesh.FaceMesh(static_image_mode=False)
        result = face_mesh.process(img)

        # print(result.)

    # Test draw
    def draw_landmarks_on_image(self, img, detection_result):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.mp_drawing.draw_landmarks(
            image=img,
            landmark_list=detection_result.multi_face_landmarks[0],
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

        return img

    def plot_face_blendshapes_bar_graph(face_blendshapes):
        face_blendshapes_names = [
            face_blendshapes_category.category_name
            for face_blendshapes_category in face_blendshapes
        ]
        face_blendshapes_scores = [
            face_blendshapes_category.score
            for face_blendshapes_category in face_blendshapes
        ]
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(
            face_blendshapes_ranks,
            face_blendshapes_scores,
            label=[str(x) for x in face_blendshapes_ranks],
        )
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # # Label each bar with values
        # for score, patch in zip(face_blendshapes_scores, bar.patches):
        #     plt.text(
        #         patch.get_x() + patch.get_width(),
        #         patch.get_y(),
        #         f"{score:.4f}",
        #         va="top",
        #     )

        # ax.set_xlabel("Score")
        # ax.set_title("Face Blendshapes")
        # plt.tight_layout()
        # plt.show()
