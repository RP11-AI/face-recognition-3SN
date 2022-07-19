# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                               (module) decoder.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                           https://keepo.io/rp11ai ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import os
import face_recognition
import numpy as np
import csv
from infos_log import info_decoder
# ---------------------------------------------------------------------------------------------------------------------|


class EncodeMaster(object):
    def __init__(self, path: str = 'face_bbox'):
        """
        It generates lists for the future execution of the class and defines the directory with the folders of each
        recognized person.
        :param path: Directory containing the folders of each coded person.
        """
        self.path = path
        self.decoder = {}
        self.sub_path = os.listdir(self.path)
        self.img_list = []
        self.temp_encoder = []

    @staticmethod
    def mean_encode(encode_data: list) -> list:
        """
        Internal class function that generates an arithmetic average of the general encodings of the person in focus.
        :param encode_data: List containing all encodings of all images containing the same person
        :return: List containing the master encodings
        """
        encode_master = list(range(len(encode_data[0])))
        for encode in encode_data:
            for i in range(len(encode_data[0])):
                encode_master[i] = (float((encode_data[0][i] + encode[i]) / 2))
        return encode_master

    def CodingReview(self):
        """
        It encodes the images again in order to generate the master encoding. Generates a csv file in each person's
        folder containing the master encoding data.
        """
        for sub_dir in self.sub_path:
            info_decoder('path_reading', path=f'{self.path}/{sub_dir}')
            self.img_list = os.listdir(f'{self.path}/{sub_dir}')
            for img in self.img_list:
                load_image = face_recognition.load_image_file(file=f'{self.path}/{sub_dir}/{img}')
                info_decoder(log='img_encode', img=f'{img}')
                encode = face_recognition.face_encodings(face_image=load_image)
                self.temp_encoder.append(encode[0])
                info_decoder(log='complete')

            encode_master = self.mean_encode(self.temp_encoder)
            with open(f'{self.path}/{sub_dir}/encode.csv', 'w') as csv_file:
                info_decoder(log='csv_file', dir_csv=f'{self.path}/{sub_dir}/encode.csv')
                csv.writer(csv_file, delimiter=',').writerow(encode_master)

            self.img_list = []
            self.temp_encoder = []

    def Decode(self):
        """
        Decodes csv files containing master encodings.
        :return: Dictionary containing lists of data for each person.
        """
        for sub_dir in self.sub_path:
            with open(f'{self.path}/{sub_dir}/encode.csv') as csv_file:
                info_decoder(log='decoder', person=f'{sub_dir}')
                reader = csv.reader(csv_file)
                for n, x in enumerate(reader):
                    if n == 0:
                        self.decoder[f'{sub_dir}'] = np.array(x)
                        info_decoder(log='complete_decoder')

        return self.decoder
