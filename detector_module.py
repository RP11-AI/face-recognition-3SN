# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                       (module) detector_module.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                           https://keepo.io/rp11ai ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import os
from typing import Any
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
# ---------------------------------------------------------------------------------------------------------------------|


class Detector(object):
    def __init__(self, dir_video, pixel_box: int = 100, box_per_line: int = 6,
                 output_grid_video_name: str = 'output_grid_video',
                 output_video_name: str = 'output_video',
                 resolution: tuple = (1280, 720), record_video: bool = False, record_grid: bool = False) -> None:
        """
        Initial settings to use face detection and search module. Remembering, that the 'resolution' and 'pixel_box'
        parameters can significantly increase the disk space used. Depending on the density of detections on the video -
        camera, it is not difficult to reach 10GB of storage in 20 - 30 processing seconds.

        :param dir_video: Video or camera directory. To use your webcam or other attached video device, the inputs are
                          [0: main video], [1, 2, 3 and so on for other devices]
        :param pixel_box: Pixel face detection box size for face grid construction. Default is 100x100px
        :param box_per_line: How many face detection boxes should be in the face detection grid.
        :param output_grid_video_name: Name of the file where the grid video will be recorded
        :param output_video_name: Name of the file where the video/camera will be recorded
        :param resolution: Resolution of the video to be recorded
        """
        self.image, self.machine, self.results, self.face_img, self.stack = None, None, None, None, None
        self.img_array_master, self.upper_line, self.lower_line = [], [], []
        self.cap = cv2.VideoCapture(dir_video)
        self.mp_face_detector = mp.solutions.face_detection
        self.pixel_box = pixel_box
        self.dir_video = dir_video
        self.resolution = resolution
        self.box_per_line = box_per_line
        self.record_video = record_video
        self.record_grid = record_grid
        self.black_img = np.zeros((pixel_box, pixel_box, 3), dtype=np.uint8)
        self.grid_resolution = (int(pixel_box * box_per_line), int(pixel_box * 2))
        self.modVideo = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        if self.record_grid:
            self.output_grid = cv2.VideoWriter(f'{output_grid_video_name}.avi', self.modVideo, 20, self.grid_resolution)
        if self.record_video:
            self.output_video = cv2.VideoWriter(f'{output_video_name}.avi', self.modVideo, 20, resolution)

    @staticmethod
    def stack_images(scale: float, img_array) -> Any:
        """
            Def to generate a generic image stack. Within the Detector object, it has the function of generation a grid
            [2 Lines, X Columns] where X is the 'box_per_line' parameter.

            :param scale: Percent resizing of image resolution.
            :param img_array: List or array containing images to be stacked.
            :return: Image grid.
            """
        rows, cols = len(img_array), len(img_array[0])
        width, height = img_array[0][0].shape[1], img_array[0][0].shape[0]
        if isinstance(img_array[0], list):
            for x in range(0, rows):
                for y in range(0, cols):
                    if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                        img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                    else:
                        img_array[x][y] = cv2.resize(img_array[x][y],
                                                     (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                     None, scale, scale)
                    if len(img_array[x][y].shape) == 2:
                        img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
            hor = [np.zeros((height, width, 3), np.uint8)] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(img_array[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if img_array[x].shape[:2] == img_array[0].shape[:2]:
                    img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
                else:
                    img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None,
                                              scale, scale)
                    if len(img_array[x].shape) == 2:
                        img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
            ver = np.hstack(img_array)
        return ver

    def Capture(self, machine) -> None:
        """
        The function is intended to process the image with the specified machine.

        :param machine: Face detection machine (mediapipe)
        """
        self.machine = machine
        _, self.image = self.cap.read()
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.results = self.machine.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def Bbox(self, print_bbox: bool = True, save_faces: bool = False) -> None:
        """
        Purpose to extract and save the clippings of the original video with the bounding boxes.

        :param print_bbox: Whether to display bounding boxes in the main video.
        :param save_faces: Save the bounding box clippings in the (automatically created) folder 'face_bbox'
        """
        try:
            os.mkdir('face_bbox')
        except FileExistsError:
            pass
        if self.results.detections:
            for id_face, face in enumerate(self.results.detections):
                height_shape, width_shape, _ = self.image.shape
                bboxC = face.location_data.relative_bounding_box
                bbox = [int(bboxC.xmin * width_shape), int(bboxC.ymin * height_shape),
                        int(bboxC.width * width_shape), int(bboxC.height * height_shape)]

                if print_bbox:
                    cv2.rectangle(img=self.image,
                                  pt1=(bbox[0], bbox[1]), pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  color=(0, 255, 0), thickness=1)

                self.face_img = self.image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                try:
                    self.face_img = cv2.resize(self.face_img, (self.pixel_box, self.pixel_box))
                except:
                    self.face_img = self.black_img
                self.img_array_master.append(self.face_img)

                if save_faces:
                    cod = str("face_bbox" + "\\" + datetime.now().strftime('%d%m%Y-%H%M%S%f') + '.png')
                    cv2.imwrite(filename=cod, img=self.face_img)

    def Grid(self) -> None:
        """
        Generate an image grid simultaneously with the video.
        """
        if len(self.img_array_master) != 0:
            if len(self.img_array_master) <= self.box_per_line:
                for i in range(0, len(self.img_array_master)):
                    self.upper_line.append(self.img_array_master[i])
                for i in range(0, (self.box_per_line - len(self.upper_line))):
                    self.upper_line.append(self.black_img)
                for i in range(0, self.box_per_line):
                    self.lower_line.append(self.black_img)

            elif self.box_per_line < len(self.img_array_master) <= 2 * self.box_per_line:
                for i in range(0, self.box_per_line):
                    self.upper_line.append(self.img_array_master[i])
                for i in range(self.box_per_line, len(self.img_array_master)):
                    self.lower_line.append(self.img_array_master[i])
                for i in range(0, (self.box_per_line - len(self.lower_line))):
                    self.lower_line.append(self.black_img)

            else:
                for i in range(0, self.box_per_line):
                    self.upper_line.append(self.img_array_master[i])
                for i in range(self.box_per_line, self.box_per_line * 2):
                    self.lower_line.append(self.img_array_master[i])
        else:
            for i in range(0, self.box_per_line):
                self.upper_line.append(self.black_img)
                self.lower_line.append(self.black_img)

        self.stack = Detector.stack_images(1, (self.upper_line, self.lower_line))
        self.stack = cv2.resize(self.stack, dsize=self.grid_resolution)

        self.upper_line, self.lower_line, self.img_array_master = [], [], []

    def VideoOutput(self, print_video: bool = True, print_grid: bool = True) -> None:
        """
        Display window containing processed videos

        :param print_video: show original video
        :param print_grid: show face detection grid
        """
        cv2.imshow(f'Video: {self.dir_video}', self.image) if print_video else None
        cv2.imshow(f'Face Grid [Video: {self.dir_video}]', self.stack) if print_grid else None

    def RecordVideo(self) -> None:
        """
        Records videos processed with face detection boxes.
        """
        self.output_grid.write(self.stack) if self.record_grid else None
        if self.record_video:
            self.image = cv2.resize(self.image, self.resolution)
            self.output_video.write(self.image)

    def ReleaseCap(self) -> None:
        """
        Release the camera/video used and destroy all windows generated by opencv.
        """
        self.cap.release()
        cv2.destroyAllWindows()
