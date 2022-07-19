# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                              (module) faceAuth.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                           https://keepo.io/rp11ai ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import cv2
import mediapipe as mp
import os
import colorama as col
from infos_log import info_faceAuth
# ---------------------------------------------------------------------------------------------------------------------|


class ExceptionPass(object):
    pass


class SecondaryDetection(object):
    def __init__(self, path: str = 'face_bbox') -> None:
        """
        It reads the png files from the indicated directory and loads the images into RAM for encoding. Contains more
        rigorous face detection.
        :param path: Directory containing the png files.
        """
        self.path = path
        self.img_id = []
        self.images = []
        self.len_path = None
        try:
            self.my_list = os.listdir(self.path)
            info_faceAuth(log='loading_path') if self.my_list else exit()
            self.len_path = len(self.my_list)
            info_faceAuth(log='number_img', number_img=int(self.len_path))
        except ExceptionPass():
            info_faceAuth(log='os.listdir_error')
            exit()

        for img_name in self.my_list:
            info_faceAuth(log='reading process', img_read=img_name)
            self.img_id.append(img_name)
            self.images.append(cv2.imread(f'{path}/{img_name}'))
            print(col.Fore.GREEN + " |---------------| COMPLETE!" + col.Style.RESET_ALL)

        self.mp_face_detection = mp.solutions.face_detection

    def FaceAuth(self, model_s: int = 1, min_detection_conf: float = 0.5) -> None:
        """
        Detect faces in png files in the directory indicated in the __init__ function. The purpose of the function is to
        filter out unnecessary images, making AI coding easier. Recommended to keep the default settings.

        :param model_s: 0 or 1. 0 to select a short-range model that works best for faces within 2 meters from the
        camera, and 1 for a full-range model best for faces within 5 meters. See details in:
        https://solutions.mediapipe.dev/face_detection#model_selection.
        :param min_detection_conf: Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.
        See details in: https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
        """

        with self.mp_face_detection.FaceDetection(min_detection_confidence=min_detection_conf,
                                                  model_selection=model_s) as face_detection:
            for id_x, file in enumerate(self.images):
                info_faceAuth(log='processing_detector', img_read=self.img_id[id_x])
                results = face_detection.process(cv2.cvtColor(file, cv2.COLOR_BGR2RGB))
                if not results.detections:
                    print(col.Fore.RED + "EMPTY" + col.Style.RESET_ALL, end=" ")
                    try:
                        os.remove(f'{self.path}/{self.img_id[id_x]}')
                        print(col.Fore.YELLOW + "||| removed" + col.Style.RESET_ALL)
                    except OSError as e:
                        print(f'Error:{e.strerror}')
                    continue
                print(col.Fore.GREEN + "DETECTION" + col.Style.RESET_ALL)

        try:
            self.my_list = os.listdir(self.path)
            info_faceAuth(log='removed_archives', number_img=int(self.len_path - len(self.my_list)))
        except ExceptionPass():
            info_faceAuth(log='os.listdir_error')
            exit()
