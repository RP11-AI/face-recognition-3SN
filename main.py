# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                                  (module) main.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                           https://keepo.io/rp11ai ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import detector_module as dtc
import faceRecognition as FR
import faceAuth
import decoder
import cv2
# ---------------------------------------------------------------------------------------------------------------------|

if __name__ == '__main__':
    detection_system = dtc.Detector(dir_video='videos/oxford-street-view.mp4')
    with detection_system.mp_face_detector.FaceDetection(min_detection_confidence=0.2,
                                                         model_selection=1) as machine_face_detector:
        while detection_system.cap.isOpened():
            detection_system.Capture(machine=machine_face_detector)
            detection_system.Bbox(save_faces=True, print_bbox=False)
            detection_system.Grid()
            detection_system.VideoOutput(print_video=False)
            detection_system.RecordVideo()

            if cv2.waitKey(5) & 0xFF == 27:
                break

    detection_system.ReleaseCap()
    faceAuth.SecondaryDetection().FaceAuth(model_s=0, min_detection_conf=0.8)
    FR.FaceRecognition().Engine(tolerance_encode=0.5)
    decoder.EncodeMaster().CodingReview()

    encode_dict = decoder.EncodeMaster().Decode()
