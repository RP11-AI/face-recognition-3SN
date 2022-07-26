![](https://img.shields.io/github/stars/pandao/editor.md.svg) ![](https://img.shields.io/github/forks/pandao/editor.md.svg) ![](https://img.shields.io/github/tag/pandao/editor.md.svg) ![](https://img.shields.io/github/release/pandao/editor.md.svg) ![](https://img.shields.io/github/issues/pandao/editor.md.svg) ![](https://img.shields.io/bower/v/editor.md.svg)

![](https://github.com/RP11-AI/personal-data/blob/main/general/header.png?raw=true)

### Introduction
The project aims to detect and recognize faces. The final objective is to code the identified faces and export the data for application in other projects, such as searching social networks by facial recognition, searching in databases and applications in Marketing, how to qualify Leads to prospect possible customers who frequent your establishment.

The code is divided into four modules, making the system easier to read and understand. We ask that, before using these modules, you fully understand how each one communicates with the other.

### Modules
In this topic we will explain the basics of how all modules work to facilitate the use and reproduction of the code. [Opencv-python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) and [Mediapipe](https://google.github.io/mediapipe/) are essential for understanding. We recommend that you have a base of usage for each extension.

#### [main.py](https://github.com/RP11-AI/face-recognition-3SN/blob/main/py/main.py)
We start by activating the Detector class of the [detector_module.py](https://github.com/RP11-AI/face-recognition-3SN/blob/main/py/detector_module.py) module. In this segmentation are most of the system properties, including whether you will proceed with the treatment and encoding of the exported images. Pay attention to the method:

<pre><code>with detection_system.mp_face_detector.FaceDetection(min_detection_confidence=0.2,
                                                     model_selection=1) as machine_face_detector:
</code></pre>

We recommend that you read the [Mediapipe Framework Concepts](https://google.github.io/mediapipe/framework_concepts/gpu.html) documentation to understand how it works and the need to reduce processing costs. You will better understand how the [detector_module.py](https://github.com/RP11-AI/face-recognition-3SN/blob/main/py/detector_module.py) module works. In the loop are the system specs and the video/camera frame refresh feature. Afterwards, the treatment of the exported images and the coding of the faces.

#### [detector_module.py](https://github.com/RP11-AI/face-recognition-3SN/blob/main/py/detector_module.py)
In order, we will continue to explain the construction of the `Detector()`. Pay attention to the __init__ function and its parameters. Some internal processing parameters such as `mp.solutions.face_detection` should have more of your attention.

| Function name  | Description                    |
| -------------  | ------------------------------ |
| `Detector()`   | Initial Parameters             |
| `stack_image()`| Generate a generic image stack |
| `Capture()`    | Process the image with the specified machine |
| `Bbox()`       | Extract and save the clippings of the original video with the bounding boxes |
| `Grid()`       | Generate an image grid simultaneously with the video |
| `VideoOutput()`| Display window containing processed videos |
| `RecordVideo()`| Records videos |
| `ReleaseCap()` | Release the camera/video used and destroy all windows generated by opencv |

In the static methods, we have the `stack_image()` function which is a generalization of ways of stacking images. It will serve to show the user the soft search being performed on the video/camera. In the `Capture()`, we have the machine parameter that will be used an argument of the class itself `self.mp_face_detection` and after executed in the [main.py](https://github.com/RP11-AI/face-recognition-3SN/blob/main/py/main.py) file in the with operator. In addition, we have the treatment of each frame, based on what is recommended by Mediapipe and the execution of the face search.

![Face Detection with Mediapipe](https://github.com/RP11-AI/personal-data/blob/main/face-recognition-3SN/face-recognition-3SN-output-video.gif?raw=true)

In `Bbox()` there is a method of showing the bounding boxes in detections to the user. Also, and more importantly to proceed with processing, extracting each bounding box into a png file. These files will be the basis for the development of other modules. The `Grid()` will generate the stacking of images for simultaneous viewing. `VideoOutput()` is the function that will generate a window to display the videos.

![Grid Video with face detections](https://github.com/RP11-AI/personal-data/blob/main/face-recognition-3SN/face-recognition-3SN-output-grid-video.gif?raw=true)

#### [faceAuth.py](https://github.com/RP11-AI/face-recognition-3SN/blob/main/py/faceAuth.py)
The module will scan the images generated by the previous module. The scan method is predestined in the *faces_bbox* folder. The class is extremely simple, but very useful. The detection pattern is more critical, to filter out unnecessary images and make [faceRecognition.py](https://github.com/RP11-AI/face-recognition-3SN/blob/main/py/faceRecognition.py) easier to work with.

|Function name          | Description                    |
| --------------------- | ------------------------------ |
| `SecondaryDetection()`| It reads the png files from the indicated directory and loads the images into RAM for encoding. Contains more rigorous face detection. |
| `FaceAuth()`          | Detect faces in png files in the directory indicated in the __init__ function. The purpose of the function is to filter out unnecessary images, making AI coding easier. Recommended to keep the default settings. |

#### [faceRecognition.py](https://github.com/RP11-AI/face-recognition-3SN/blob/main/py/faceRecognition.py)
In this class we will identify each face and organize them by folders. For the module to work properly, you must [install Visual Studio Community 2022](https://visualstudio.microsoft.com/pt-br/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false) and in the run lobby, install the C++ features. The encoding of images considers the module [dlib==18.19.0](http://dlib.net) for performing the encoding. The image search algorithm is simple but efficient. When running `Engine()` it will read the first image from the directory and encode it. Afterwards, it will encode and compare with the second image. If there is any similarity in the faces found, the analyzed images will be exported to a __person(x)__ folder. The algorithm will continue the loop until no image remains.

![Folders](https://github.com/RP11-AI/personal-data/blob/main/face-recognition-3SN/face-recognition-3SN-pathes.png?raw=true)

|Function name          | Description                    |
| --------------------- | ------------------------------ |
| `FaceRecognition()`   | Defines the directory needed to read the data and defines a list (self.img_id) with the id of each png image |
| `Engine()`            |  It encodes the faces in the png files informed by the function above and organizes them in folders containing the same person |

![Algorithm faceRecognition.py](https://github.com/RP11-AI/personal-data/blob/main/face-recognition-3SN/face-recognition-3SN-authFace.png?raw=true)

#### [decoder.py](https://github.com/RP11-AI/face-recognition-3SN/blob/main/py/decoder.py)
The system aims to read each __person(x)__ folder and reanalyze each included image. Each folder will have n images that are supposed to be identical faces. In this reanalysis, the encoding will be reserved in RAM to generate a master encoding of the identified face. When re-parsing all the images in the directory, the static `mean_encode()` method will kick in. It will generate an arithmetic average of the face coding data in the analyzed folder. Therefore, we have an encoding with more relevant data in a smaller disk space.

![CodingReview() by decoder.py](https://github.com/RP11-AI/personal-data/blob/main/face-recognition-3SN/face-recognition-3SN-recognition.png?raw=true)

|Function name          | Description                    |
| --------------------- | ------------------------------ |
| `EncodeMaster()`      | It generates lists for the future execution of the class and defines the directory with the folders of each recognized person |
| `mean_encode()`       | Internal class function that generates an arithmetic average of the general encodings of the person in focus |
| `CodingReview()`      | It encodes the images again in order to generate the master encoding. Generates a csv file in each person's folder containing the master encoding data |
| `Decode()`            | Decodes csv files containing master encodings |

When the csv file is generated, it will be exported to the parsed directory with the name __encode.csv__. The intention of generating such a file is to facilitate future use in other projects, as mentioned in the introduction.

![encode.csv](https://github.com/RP11-AI/personal-data/blob/main/face-recognition-3SN/face-recognition-3SN-person-list.png?raw=true)

To use the csv file, we have the `Decode()` function. It will read all the folders in the directory and look for the __encode.csv__ file. Upon reading, a dictionary will be generated containing [numpy.array files](https://numpy.org/doc/stable/reference/generated/numpy.array.html), standard for using the [face-recognition](https://pypi.org/project/face-recognition/) module.

![](https://github.com/RP11-AI/personal-data/blob/main/general/baseboard.png?raw=true)
