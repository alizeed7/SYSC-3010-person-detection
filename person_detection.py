import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from picamera2 import MappedArray, Picamera2, Preview

rectangles = []
camera = Picamera2()

def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    dict = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        dict[int(pair[0])] = pair[1].strip()
    return dict


def DrawRectangles(request):
    with MappedArray(request, "main") as m:
        for rect in rectangles:
            rect_start = (int(rect[0] * 2) - 5, int(rect[1] * 2) - 5)
            rect_end = (int(rect[2] * 2) + 5, int(rect[3] * 2) + 5)
            cv2.rectangle(m.array, rect_start, rect_end, (0, 255, 0, 0))


def InferenceTensorFlow(image, model, label, video_out_location):
    global rectangles

    if label:
        labels = ReadLabelFile(label)
    else:
        labels = None

    interpreter = tflite.Interpreter(model_path=model, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = False
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    initial_h, initial_w, channels = rgb.shape

    picture = cv2.resize(rgb, (width, height))

    input_data = np.expand_dims(picture, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])

    rectangles = []
    for i in range(int(num_boxes)):
        top, left, bottom, right = detected_boxes[0][i]
        classId = int(detected_classes[0][i])
        score = detected_scores[0][i]
        if score > 0.5:
            xmin = left * initial_w
            ymin = bottom * initial_h
            xmax = right * initial_w
            ymax = top * initial_h
            if labels:
                print(labels[classId], 'score = ', score)
                if (labels[classId] == 'person') and (score >= 0.60):
                    capture_video(video_out_location)
            else:
                print('score = ', score)
            box = [xmin, ymin, xmax, ymax]
            rectangles.append(box)
            
def capture_video(video_out_location):
    camera.start_and_record_video(video_out_location, duration=5)
    camera.stop_preview()
    quit()
    
def main():
    global camera
    camera.start_preview(Preview.QTGL)
    config = camera.create_preview_configuration(main={"size": (640, 480)},
                                                 lores={"size": (320, 240), "format": "YUV420"})
    camera.configure(config)

    stride = camera.stream_configuration("lores")["stride"]
    camera.post_callback = DrawRectangles #called automatically by camera for each frame

    camera.start()

    while True:
        buffer = camera.capture_buffer("lores")
        #create a 2D numpy array representing a grayscale image
        grey = buffer[:stride * 240].reshape((240, stride))
        _ = InferenceTensorFlow(grey, "mobilenet_v2.tflite", "coco_labels.txt", "test.mp4")


if __name__ == '__main__':
    main()
