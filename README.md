### Introduction
The project aims to extract as many faces as possible from the informed video/camera. The first module, [detector_module.py](https://github.com/RP11-AI/face-search-by-recognition/blob/main/py/detector_module.py) aims to perform a soft scan to extract as many face detections as possible. This module, in addition to generating a soft scan, gives the user freedom to record the video played, display the bounding boxes and manipulate other inputs for optimization or testing.

![Alt Text](https://github.com/RP11-AI/face-search-by-recognition/blob/main/readme-data/output_video.gif?raw=true)

In the same module, a stack of images is also generated that will be saved in the automatically generated folder ***faces_bbox* **for future treatment with the other modules.

![Alt Text](https://github.com/RP11-AI/face-search-by-recognition/blob/main/readme-data/output_grid_video.gif?raw=true)

### Face Auth
In the second process, a more robust face detection process is performed. In [faceAuth.py](https://github.com/RP11-AI/face-search-by-recognition/blob/main/py/faceAuth.py) the detection will be performed by the png images. As the images are focused on possible faces, a more robust detection is more efficient. Images in which a face is not identified the system will delete.

### Face Recognition
In the module [faceRecognition.py](https://github.com/RP11-AI/face-search-by-recognition/blob/main/py/faceRecognition.py) the images that are still in the folder will be encoded. While the images are being encoded, they will be compared with themselves in a search engine to organize identical faces in the same specific directory.

![All Text](https://github.com/RP11-AI/face-search-by-recognition/blob/main/readme-data/2022-07-19%20120423.png?raw=true)

After the process is finished, all the images will be organized in folders.

![All Text](https://github.com/RP11-AI/face-search-by-recognition/blob/main/readme-data/2022-07-19%20120214.png?raw=true)

### Decoder
With the detections already organized, the module [decoder.py](https://github.com/RP11-AI/face-search-by-recognition/blob/main/py/decoder.py) will encode all the images in the folder and generate a master encoding of the identified person. Master encoding will be exported in csv.

![All text](https://github.com/RP11-AI/face-search-by-recognition/blob/main/readme-data/2022-07-19%20120319.png?raw=true)

In this module it will be possible to decode the csv file for future use by the identified people.

![All text](https://github.com/RP11-AI/face-search-by-recognition/blob/main/readme-data/2022-07-19%20120503.png?raw=true)
