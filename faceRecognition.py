# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                       (module) faceRecognition.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                           https://keepo.io/rp11ai ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import cv2
import os
import colorama as col
import face_recognition
import shutil
from infos_log import info_face_recognition
# ---------------------------------------------------------------------------------------------------------------------|


class FaceRecognition(object):
    def __init__(self, path: str = 'face_bbox') -> None:
        """
        Defines the directory needed to read the data and defines a list (self.img_id) with the id of each png image.
        :param path: Directory containing png files.
        """
        self.path = path
        self.img_id = []
        try:
            self.img_id = os.listdir(self.path)
            info_face_recognition(log='loading_path')
        except FileNotFoundError:
            info_face_recognition(log='os.listdir_error'), exit()

    def Engine(self, tolerance_encode: float = 0.5) -> None:
        """
        It encodes the faces in the png files informed by the function above and organizes them in folders containing
        the same person.
        :param tolerance_encode: Detection tolerance. The smaller, the more rigorous the coding.
        """
        for n, b_img in enumerate(self.img_id):
            info_face_recognition(log='new_person', person_img_id=b_img)
            try:
                img_base = face_recognition.load_image_file(file=f'{self.path}/{b_img}')
                img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
                try:
                    info_face_recognition(log='encoding_new_person')
                    img_base_encode = face_recognition.face_encodings(face_image=img_base)[0]
                    info_face_recognition(log='complete_new_person')
                    os.makedirs(f'{self.path}/person{n}'), info_face_recognition(log='new_path')
                    shutil.move(f'{self.path}/{b_img}', f'{self.path}/person{n}')
                    for t_img in self.img_id:
                        if t_img != b_img:
                            try:
                                img_test = face_recognition.load_image_file(file=f'{self.path}/{t_img}')
                                img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
                                try:
                                    info_face_recognition(log='search_data', person_img_id=t_img)
                                    img_test_encode = face_recognition.face_encodings(face_image=img_test)[0]
                                    result = face_recognition.compare_faces([img_base_encode],
                                                                            img_test_encode, tolerance=tolerance_encode)
                                    info_face_recognition(log='complete_search')
                                    print(col.Fore.GREEN if result[0] else col.Fore.RED, end='')
                                    print('[identify]' if result[0] else '[not found]')
                                    if result[0]:
                                        shutil.move(f'{self.path}/{t_img}', f'{self.path}/person{n}')
                                except IndexError:
                                    info_face_recognition(log='face_not_found')
                                    os.remove(f'{self.path}/{t_img}')
                            except FileNotFoundError:
                                pass
                except IndexError:
                    info_face_recognition(log='face_not_found')
                    os.remove(f'{self.path}/{b_img}')
            except FileNotFoundError:
                info_face_recognition(log='face_not_found')
