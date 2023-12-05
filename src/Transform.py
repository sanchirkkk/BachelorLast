import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
from src.image_warp import *
import math
import skimage.exposure


class Transform:
    def __init__(self) -> None:
        self.mesh_list = self.mesh_latest()

    def mesh_latest(self):
        left_eye = [
            226,
            247,
            247,
            30,
            30,
            29,
            29,
            27,
            27,
            28,
            28,
            56,
            56,
            190,
            190,
            244,
            244,
            112,
            112,
            26,
            26,
            22,
            22,
            23,
            23,
            24,
            24,
            110,
            110,
            25,
            25,
            226,
        ]
        right_eye = [
            463,
            414,
            414,
            286,
            286,
            258,
            258,
            257,
            257,
            259,
            259,
            260,
            260,
            467,
            467,
            446,
            446,
            255,
            255,
            339,
            339,
            254,
            254,
            253,
            253,
            252,
            252,
            256,
            256,
            341,
        ]
        full_mesh_connection = mp.solutions.face_mesh.FACEMESH_TESSELATION

        r_eye_mesh_connection = mp.solutions.face_mesh.FACEMESH_LEFT_EYE
        l_eye_mesh_connection = mp.solutions.face_mesh.FACEMESH_RIGHT_EYE

        result = full_mesh_connection - r_eye_mesh_connection - l_eye_mesh_connection
        list_landmarks = []
        for i in result:
            list_landmarks.append(i[0])
            list_landmarks.append(i[1])
        only_landmark_1 = list(set(list_landmarks))
        only_landmark = []
        eyes_ = list(set(right_eye))
        eyes_r = list(set(left_eye))
        for i in only_landmark_1:
            if i in eyes_ or i in eyes_r:
                pass
            else:
                only_landmark.append(i)

        return only_landmark

    def get_coordinates(self, processed_image):
        landmark = self.mesh_list
        x_lan = []
        y_lan = []
        for lan in landmark:
            x_lan.append(processed_image.multi_face_landmarks[0].landmark[lan].x)
            y_lan.append(processed_image.multi_face_landmarks[0].landmark[lan].y)

        x_values = np.array(x_lan)
        y_values = np.array(y_lan)
        return x_values, y_values

    def normal_image(self, dstIm, proccessed_1):
        width = dstIm.shape[1]
        height = dstIm.shape[0]
        dstIm = Image.fromarray(dstIm)
        dst_pts = []
        x_values, y_values = self.get_coordinates(proccessed_1)
        for x, y in zip(x_values, y_values):
            dst_pts.append((width * x, height * y))
        return dstIm, dst_pts

    def normalize_shape(
        self, input_image, processed_image, mean_image, mean_image_processed
    ):
        srcIm = Image.fromarray(input_image)
        dstIm, dst_pts = self.normal_image(mean_image, mean_image_processed)
        src_pts = []
        x_values, y_values = self.get_coordinates(processed_image)
        for x, y in zip(x_values, y_values):
            src_pts.append((input_image.shape[1] * x, input_image.shape[0] * y))

        mean_array = np.array(dstIm.convert("L"))

        ymin, ymax, xmin, xmax, img_mask = PiecewiseAffineTransform(
            srcIm, src_pts, dstIm, dst_pts
        )
        shape_normalized_img = np.array(dstIm)
        mean_array = mean_array
        try:
            shape_normalized_img = (shape_normalized_img * (img_mask != 0))[
                xmin:xmax, ymin:ymax
            ]
        except:
            shape_normalized_img = shape_normalized_img[xmin:xmax, ymin:ymax] * (
                img_mask != 0
            )

        return shape_normalized_img

    def cut_head(self, mean_image):
        image1_shape = mean_image.shape
        image2_shape = (1200, 800, 3)

        image1 = mean_image
        image2 = np.zeros(image2_shape, dtype=np.uint8)

        center_x = (image2_shape[1] - image1_shape[1]) // 2
        center_y = (image2_shape[0] - image1_shape[0]) // 2

        image2[
            center_y : center_y + image1_shape[0],
            center_x : center_x + image1_shape[1],
            :,
        ] = image1

        return image2

    def bigger_eyes(self, img, cx, cy):
        radius = 20
        cx = int(cx)
        cy = int(cy)
        # set distortion gain
        gain = 1.1

        # crop image
        crop = img[cy - radius : cy + radius, cx - radius : cx + radius]

        # get dimensions
        ht, wd = crop.shape[:2]
        xcent = wd / 2
        ycent = ht / 2
        rad = min(xcent, ycent)

        # set up the x and y maps as float32
        map_x = np.zeros((ht, wd), np.float32)
        map_y = np.zeros((ht, wd), np.float32)
        mask = np.zeros((ht, wd), np.uint8)
        for y in range(ht):
            Y = (y - ycent) / ycent
            for x in range(wd):
                X = (x - xcent) / xcent
                R = math.hypot(X, Y)
                if R == 0:
                    map_x[y, x] = x
                    map_y[y, x] = y
                    mask[y, x] = 255
                elif R >= 0.90:  # avoid extreme blurring near R = 1
                    map_x[y, x] = x
                    map_y[y, x] = y
                    mask[y, x] = 0
                elif gain >= 0:
                    map_x[y, x] = (
                        xcent * X * math.pow((2 / math.pi) * (math.asin(R) / R), gain)
                        + xcent
                    )
                    map_y[y, x] = (
                        ycent * Y * math.pow((2 / math.pi) * (math.asin(R) / R), gain)
                        + ycent
                    )
                    mask[y, x] = 255
                elif gain < 0:
                    gain2 = -gain
                    map_x[y, x] = (
                        xcent * X * math.pow((math.sin(math.pi * R / 2) / R), gain2)
                        + xcent
                    )
                    map_y[y, x] = (
                        ycent * Y * math.pow((math.sin(math.pi * R / 2) / R), gain2)
                        + ycent
                    )
                    mask[y, x] = 255

        # remap using map_x and map_y
        bump = cv2.remap(
            crop,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # antialias edge of mask
        # (pad so blur does not extend to edges of image, then crop later)
        blur = 7
        mask = cv2.copyMakeBorder(
            mask, blur, blur, blur, blur, borderType=cv2.BORDER_CONSTANT, value=(0)
        )
        mask = cv2.GaussianBlur(
            mask, (0, 0), sigmaX=blur, sigmaY=blur, borderType=cv2.BORDER_DEFAULT
        )
        h, w = mask.shape
        mask = mask[blur : h - blur, blur : w - blur]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = skimage.exposure.rescale_intensity(
            mask, in_range=(127.5, 255), out_range=(0, 1)
        )

        # merge bump with crop using grayscale (not binary) mask
        bumped = (bump * mask + crop * (1 - mask)).clip(0, 255).astype(np.uint8)

        # insert bumped image into original
        result = img.copy()
        result[cy - radius : cy + radius, cx - radius : cx + radius] = bumped
        return result
