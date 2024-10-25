### README
**About**

This Python script utilizes the Picamera2 library to capture images from a Raspberry Pi camera, process these images using a TensorFlow Lite model, and detect people in real time. It reads labels from a file and uses OpenCV to draw bounding rectangles around detected objects. The InferenceTensorFlow function performs object detection on grayscale images, resizing them to match the model's input size, and checks the detection scores against a threshold. If a "person" is detected with a score above 0.60, it triggers the capture_video function, which records a 5-second video of the scene. The script continuously captures frames from the camera, processes them for object detection, and draws rectangles on the detected objects in real time.
