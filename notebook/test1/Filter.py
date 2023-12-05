import cv2
import mediapipe as mp
import triangulation_media_pipe as tmp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


class MakeFilter:
    def __init__(self) -> None:
        self.TRIANGULATION = tmp.TRIANGULATION

    def transform_landmarks_from_tf_to_ocv(self, keypoints, face_width, face_height):
        landmark_list = []
        if keypoints.multi_face_landmarks != None:
            for face_landmarks in keypoints.multi_face_landmarks:
                for l in face_landmarks.landmark:
                    pt = mp_drawing._normalized_to_pixel_coordinates(
                        l.x, l.y, face_width, face_height
                    )
                    landmark_list.append(pt)
        return landmark_list

    def process_base_face_mesh(self, results_face, image_file, income_result):
        base_face_handler = {"img": image_file, "landmarks": income_result}
        base_input_image = base_face_handler["img"].copy()
        image_rows, image_cols, _ = base_face_handler["img"].shape
        landmark_base_ocv = self.transform_landmarks_from_tf_to_ocv(
            base_face_handler["landmarks"], image_cols, image_rows
        )
        return landmark_base_ocv, base_input_image

    def process(self, webcam_img, results_face, income_image, income_result):
        (
            landmark_base_ocv,
            base_input_image,
        ) = self.process_base_face_mesh(results_face, income_image, income_result)

        webcam_img_ = webcam_img.copy()

        image_rows, image_cols, _ = webcam_img.shape
        webcam_img.flags.writeable = False
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        landmark_target_ocv = self.transform_landmarks_from_tf_to_ocv(
            results_face, image_cols, image_rows
        )
        # Draw the face mesh annotations on the image.
        webcam_img.flags.writeable = True
        image = webcam_img.copy()
        seam_clone = image.copy()
        out_image = webcam_img.copy()
        img2_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(image)
        seamlessclone = webcam_img.copy()

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=out_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

                if len(landmark_target_ocv) > 0:
                    points2 = np.array(landmark_target_ocv, np.int32)
                    convexhull2 = cv2.convexHull(points2)
                    process = True
                    if process == True:
                        for i in range(0, int(len(tmp.TRIANGULATION) / 3)):
                            triangle_index = [
                                self.TRIANGULATION[i * 3],
                                self.TRIANGULATION[i * 3 + 1],
                                self.TRIANGULATION[i * 3 + 2],
                            ]
                            tbas1 = landmark_base_ocv[triangle_index[0]]
                            tbas2 = landmark_base_ocv[triangle_index[1]]
                            tbas3 = landmark_base_ocv[triangle_index[2]]
                            triangle1 = np.array([tbas1, tbas2, tbas3], np.int32)

                            rect1 = cv2.boundingRect(triangle1)
                            (x, y, w, h) = rect1
                            cropped_triangle = base_input_image[y : y + h, x : x + w]
                            cropped_tr1_mask = np.zeros((h, w), np.uint8)

                            points = np.array(
                                [
                                    [tbas1[0] - x, tbas1[1] - y],
                                    [tbas2[0] - x, tbas2[1] - y],
                                    [tbas3[0] - x, tbas3[1] - y],
                                ],
                                np.int32,
                            )

                            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
                            ttar1 = landmark_target_ocv[triangle_index[0]]
                            ttar2 = landmark_target_ocv[triangle_index[1]]
                            ttar3 = landmark_target_ocv[triangle_index[2]]

                            triangle2 = np.array([ttar1, ttar2, ttar3], np.int32)

                            rect2 = cv2.boundingRect(triangle2)
                            (x, y, w, h) = rect2

                            cropped_tr2_mask = np.zeros((h, w), np.uint8)

                            points2 = np.array(
                                [
                                    [ttar1[0] - x, ttar1[1] - y],
                                    [ttar2[0] - x, ttar2[1] - y],
                                    [ttar3[0] - x, ttar3[1] - y],
                                ],
                                np.int32,
                            )

                            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
                            # Warp triangles
                            points = np.float32(points)
                            points2 = np.float32(points2)
                            M = cv2.getAffineTransform(points, points2)
                            warped_triangle = cv2.warpAffine(
                                cropped_triangle,
                                M,
                                (w, h),
                                borderMode=cv2.BORDER_REPLICATE,
                            )
                            warped_triangle = cv2.bitwise_and(
                                warped_triangle,
                                warped_triangle,
                                mask=cropped_tr2_mask,
                            )

                            # Reconstructing destination face
                            img2_new_face_rect_area = img2_new_face[
                                y : y + h, x : x + w
                            ]
                            img2_new_face_rect_area_gray = cv2.cvtColor(
                                img2_new_face_rect_area, cv2.COLOR_BGR2GRAY
                            )
                            _, mask_triangles_designed = cv2.threshold(
                                img2_new_face_rect_area_gray,
                                1,
                                255,
                                cv2.THRESH_BINARY_INV,
                            )
                            warped_triangle = cv2.bitwise_and(
                                warped_triangle,
                                warped_triangle,
                                mask=mask_triangles_designed,
                            )

                            img2_new_face_rect_area = cv2.add(
                                img2_new_face_rect_area, warped_triangle
                            )
                            img2_new_face[
                                y : y + h, x : x + w
                            ] = img2_new_face_rect_area

                        img2_face_mask = np.zeros_like(img2_gray)
                        img2_head_mask = cv2.fillConvexPoly(
                            img2_face_mask, convexhull2, 255
                        )
                        img2_face_mask = cv2.bitwise_not(img2_head_mask)

                        img2_head_noface = cv2.bitwise_and(
                            seam_clone, seam_clone, mask=img2_face_mask
                        )
                        result_image = cv2.add(img2_head_noface, img2_new_face)

                        (x, y, w, h) = cv2.boundingRect(convexhull2)
                        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
                        seamlessclone = cv2.seamlessClone(
                            result_image,
                            seam_clone,
                            img2_head_mask,
                            center_face2,
                            cv2.MIXED_CLONE,
                        )

        return seamlessclone, result_image, webcam_img_
