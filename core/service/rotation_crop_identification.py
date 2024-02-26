import base64
import os
from datetime import time

import cv2
import torch
import numpy as np
from PIL import Image
from flask import jsonify
from pyzbar import pyzbar
import time

from core.service.classification_legals import classification_legals_service

cls_dict = {
    0: "BOTTOM_LEFT",
    1: "BOTTOM_RIGHT",
    2: "DATE_OF_BIRTH",
    3: "DATE_OF_EXPIRY",
    4: "FACE",
    5: "FULL_NAME",
    6: "ID",
    7: "IDENTITY_CARD",
    8: "NATIONALITY",
    9: "PLACE_OF_ORIGIN",
    10: "PLACE_OF_RESIDENCE_1",
    11: "PLACE_OF_RESIDENCE_2",
    12: "QR_CODE",
    13: "SEX",
    14: "TOP_LEFT",
    15: "TOP_RIGHT"
}

model = None


def rotate_image(img_origin):
    """Rotate an image if its vertical size is longer than horizontalsize

    Args:
      img_origin: an original image (which is read from cv2.imread(sre))
        For example: cv2.imread("path_to_img")
    Return:
      Return an rotated image which has width longer than height

    Example usage:
      src = "/content/notccd.jpg"
      src_img = cv2.imread(src)
      rotated_img = rorate_image(src_img, src)
    """
    # Get image dimensions
    height, width = img_origin.shape[:2]

    # Check if width is less than height
    if width < height:
        # Rotate the image 90 degrees clockwise
        rotated_image = cv2.rotate(img_origin, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cv2.imwrite(src, rotated_image)
        return rotated_image

    else:
        return img_origin


# This function is used when 4 corner is detected at get_fourCorners
def sort_by2013(boxes, cls):
    boxes = torch.tensor(boxes)
    cls = torch.tensor(cls)
    # Get the indices that would sort cls ascent
    sorted_indices = torch.argsort(cls)
    # Sort cls tensor by indices ascent
    sorted_cls = cls[sorted_indices]
    # Sort xywh tensor based on sorted indices ascent
    tmp_boxes = boxes[sorted_indices]
    # Sort xywh based on 2-0-1-3
    sorted_boxes = tmp_boxes[torch.tensor([2, 0, 1, 3])]
    return sorted_boxes


# This function is used to check whether 4 corners are exist in bbox or not
def check_fourCornerExist(boxes) -> list:
    cls = boxes.cls
    list_check = [0., 1., 14., 15.]
    trigger = all(e in cls for e in list_check)
    return trigger


# This function is used to get 4 corners of card, which is used for get_transformedImage function
def get_fourCorners(img, model, conf_corner: float = 0.4):
    list_corner = [0., 1., 14., 15.]
    cls_founded = []
    box_founded = []
    # Starting detection to check 4 corners
    preds = model(img, save=False, imgsz=1024, conf=conf_corner)
    # Get boxes which are detected boxes
    for result in preds:
        boxes = result.boxes.to('cuda')  # Boxes object for bbox outputs and move it to cpu
    # Check whether yolo detect 4 corner or not
    if check_fourCornerExist(boxes):
        # print("4 Corners Detected!")
        boxes_cpu = boxes.cpu().numpy()
        cls = boxes_cpu.cls.tolist()
        xywh_boxes = boxes_cpu.xywh.tolist()
        for i in range(len(cls)):
            if cls[i] in list_corner:
                cls_founded.append(cls[i])
                box_founded.append(xywh_boxes[i])
        # print(cls_founded)
        # print(box_founded)
        # Sort 4 corner by 14-0-1-15
        sorted_xywh_boxes = sort_by2013(box_founded, cls_founded)
        # print(f"4 Corners after being sorted{sorted_xywh_boxes}")
        return sorted_xywh_boxes
    else:
        raise ValueError("Cannot detect 4 corners to transform!")


# This function is used to transform the image by provided 4 corners from get_fourCorners function
def get_transformedImage(img, model, conf_corner: float):  # -> np.ndarray
    """Starting to transform image by 4 detected corners of card

  Args:
    img: the image have been read by cv.imread and rotated by rotate_image function
  """
    coordinates = []
    # Get for sorted coners
    boxes = get_fourCorners(img, model, conf_corner)
    # Get cors x-y each box
    for i in range(len(boxes)):
        coordinates.append([abs(boxes[i][0]), abs(boxes[i][1]), abs(boxes[i][2]), abs(boxes[i][3])])

    # 14-0-1-15
    pt_A = [coordinates[0][0], coordinates[0][1]]
    pt_B = [coordinates[1][0], coordinates[1][1] + int(coordinates[1][3] / 2)]
    pt_C = [coordinates[2][0], coordinates[2][1] + int(coordinates[2][3] / 2)]
    pt_D = [coordinates[3][0], coordinates[3][1]]

    # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
    # Begin to perspectively transform image
    pts1 = np.float32([pt_A, pt_B,
                       pt_C, pt_D])

    aspect_ratio = maxHeight / maxWidth
    new_img_width = int(img.shape[1] * 0.6)
    new_img_height = int(new_img_width * aspect_ratio)
    pts2 = np.float32([[0, 0],
                       [0, new_img_height],
                       [new_img_width, new_img_height],
                       [new_img_width, 0]])
    # ======================================================

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (new_img_width, new_img_height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0, 0])

    return result


# This function is used to show the transformed image, which is returned from get_transformedImage
def display_transformedImage(img):
    """Display the rotated image if there are 4 corners to transform

    Args:
      img: the image have been read by cv.imread and rotated by rotate_image function and then being transformed by 4 corner of get_transformedImage function
    """
    # Generate a filename based on the current timestamp
    timestamp = int(time.time())
    results_path = '.results_detection/'
    # Save the image with a timestamped filename in the '/test' directory
    save_path = os.path.join(results_path, f'{timestamp}.jpg')
    cv2.imwrite(save_path, img)


# --------------------------------------------Starting detect information boxes---------------------------------------'

def get_boxesWithNoCorners(predicts) -> dict:
    """Remove 4 corners(which is only usable in get_transformedImage function) from boxes

  Args:
    predicts: the result of model(transformed_image)

  Returns:
    dict: type : cls_idx: box
    The dictionary with key is index of classes, value is the coordination of box
  """
    for result in predicts:
        objects = result.boxes  # Boxes object for bbox outputs
    objects = objects.to('cpu')
    # Get conf of each class:
    conf = objects.conf.tolist()
    # Remove 4 corners from boxes
    cls_idx = list(objects.cls.numpy())
    # Convert tensor to list
    boxes_detected = objects.xyxy.numpy().tolist()
    # Four corner needed to remove
    remove_idx = [0, 1, 14, 15]
    boxes_detected_no_corners = {}

    for i in range(len(cls_idx)):
        if cls_idx[i] in remove_idx:
            continue
        else:
            boxes_detected_no_corners[int(cls_idx[i])] = [boxes_detected[i], conf[i]]
    # return dict with type : cls_idx: box
    return boxes_detected_no_corners


def get_croppedImg(img: np.ndarray = None, box: list = None) -> np.ndarray:
    """Get the transformed_image and the box(contains xmin, ymin, xmax, ymax) to crop the image

    Args:
      img: The image have been perspective transformed(np.ndarray type)
      box: The box contains xmin, ymin, xmax, ymax from prediction of model

    Returns:
      The cropped image in np.ndarray type
    """
    if (img is None or box is None):
        raise ValueError("Image or box is None!")
    else:
        # Cropped image
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        cropped_image = img[ymin:ymax, xmin:xmax]

        return cropped_image


def get_allCroppedImg(predicts, img, vietOCR):
    """Crop all image by box and display it, then using vietOCR to read the content of each cropped image

    Args:
      predicts: the result of model(transformed_image)
      img: the image have been read by cv.imread and rotated by rotate_image function and then being transformed by 4 corner of get_transformedImage function
      vietOCR: the detector of vietOCR
        For example:
          config = Cfg.load_config_from_name('vgg_transformer')
          config['cnn']['pretrained']=False
          config['device'] = 'cpu'
          detector = Predictor(config)

    Returns:
      Nothing

    Example usage:
    get_allCroppedImg(preds, transformed_img, detector)
    """
    data = {}
    boxes_detected_noCorners = get_boxesWithNoCorners(predicts)
    for i in boxes_detected_noCorners:
        # print("=======================================================================================")
        # print(f"cls index:{i}")
        # print(f"Name of cls: {cls_dict[i]}")
        box = boxes_detected_noCorners[i][0]
        conf = boxes_detected_noCorners[i][1]
        # print(f"Box of object: {box}")
        cropped_img = get_croppedImg(img, box)
        # Resize image for better read
        h, w = cropped_img.shape[:2]
        if w < 100:
            cropped_img = cv2.resize(cropped_img, (int(w * 1.5), int(h * 1.5)))

        pil_img = Image.fromarray(cropped_img)

        if i == 12:
            qr_codes = pyzbar.decode(pil_img)
            if len(qr_codes) == 0:
                # print("Cannot read QR code!")
                qr_code = "Cannot read QR code!"
            else:
                qr_code = qr_codes[0]
                qr_data = qr_code.data.decode("utf-8")
                qr_code = f"{qr_data}"
            data[int(i)] = [qr_code, '', '']
        else:
            text, text_prob = vietOCR.predict(pil_img, return_prob=True)
            data[int(i)] = [text, text_prob, conf]
    return data


def check_image_quality(image_data: bytes) -> bool:
    # Decode image
    img_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Calculate image metrics (e.g., resolution, file size, format)
    image_size = len(image_data)
    resolution = img_np.shape[1], img_np.shape[0]  # width, height
    image_format = 'jpg' if image_data.startswith(b'\xff\xd8\xff') else None

    # Check conditions
    if resolution[0] < resolution[1]:
        if resolution[0] < 640 and resolution[1] < 640:
            print("Error: Minimum resolution not met (1280x720)")
            return False

    else:
        if resolution[0] < 640 and resolution[1] < 640:
            print("Error: Minimum resolution not met (1280x720)")
            return False

    if image_size > 20 * 1024 * 1024:
        print("Error: File size exceeds 20MB")
        return False

    if image_format != 'jpg':
        print("Error: Image format is not JPG")
        return False

    # Check blur using Laplacian method
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray_img, cv2.CV_64F).var()

    if blur_score < 100:
        print("Error: Image is too blurry")
        return False

    return True


def prediction(image_base64: str,
               model,
               vietOCR_detector,
               img_size: int = 1024,
               conf_corner: float = 0.4,
               conf_object: float = 0.4,
               ):
    start_time = time.time()  # Record the start time

    _, legal_type = classification_legals_service(image_base64)
    print(legal_type)

    if legal_type != 'CCCD_front':
        return jsonify(
            {'status': 'Error', 'message': {legal_type + ' is not support yet. Please try front of CCCD image'}})

    # Load the image
    image_data = base64.b64decode(image_base64)
    if not check_image_quality(image_data):
        return jsonify({'status': 'Error', 'message': 'Image quality check failed'})

    img_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    rotated_image = rotate_image(img_np)

    # Transforming the image for better detection
    transformed_img = get_transformedImage(rotated_image, model, conf_corner=conf_corner)

    # img = Image.fromarray(transformed_img)
    # img_base64 = base64.b64encode(img).decode("utf-8")
    #
    # image_bytes = np.array_str(transformed_img).encode('utf-8')
    #
    # # Encode the bytes as base64
    # base64_encoded = base64.b64encode(image_bytes)
    #
    # # Convert the base64 bytes to a string (if needed)
    # base64_string = base64_encoded.decode('utf-8')
    #
    # print("Base64 string:", base64_string)

    # Starting to predict by using transformed_img
    preds = model(transformed_img, save=True, imgsz=img_size, conf=conf_object)

    data = get_allCroppedImg(preds, transformed_img, vietOCR_detector)

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the time taken for execution

    print(f'Time: {execution_time}')

    return jsonify(format_data(data))


def format_data(data):
    # Text Prediction
    id = data[6][0]
    name = data[5][0]
    dob = data[2][0]
    sex = data[13][0]
    nationality = data[8][0]
    home = data[9][0]
    address = data[10][0] + ", " + data[11][0]
    doe = data[3][0]
    QR_code = data[12][0]
    # address_entities = {}

    # Text Prob-Confidence Prediction
    id_prob = data[6][1]
    name_prob = data[5][1]
    dob_prob = data[2][1]
    sex_prob = data[13][1]
    nationality_prob = data[8][1]
    home_prob = data[9][1]
    address_prob = (data[10][1] + data[11][1]) / 2.0
    doe_prob = data[3][1]

    # Label Prob-Confidence Prediction
    label_id_prob = data[6][2]
    label_name_prob = data[5][2]
    label_dob_prob = data[2][2]
    label_sex_prob = data[13][2]
    label_nationality_prob = data[8][2]
    label_home_prob = data[9][2]
    label_address_prob = (data[10][2] + data[11][2]) / 2.0
    label_doe_prob = data[3][2]

    text_overall_score = (id_prob
                          + name_prob
                          + dob_prob
                          + sex_prob
                          + nationality_prob
                          + home_prob
                          + address_prob
                          + doe_prob) / 8.0

    label_overall_score = (label_id_prob
                           + label_name_prob
                           + label_dob_prob
                           + label_sex_prob
                           + label_nationality_prob
                           + label_home_prob
                           + label_address_prob
                           + label_doe_prob) / 8.0

    format_data = {
        "errorCode": 0,
        "errorMessage": "",
        "data": [
            {
                "id": str(id),
                "id_text_prob": str(id_prob * 100)[0:5],
                "id_label_prob": str(label_id_prob * 100)[0:5],
                "name": str(name),
                "name_prob": str(name_prob * 100)[0:5],
                "label_name_prob": str(label_name_prob * 100)[0:5],
                "dob": str(dob),
                "dob_prob": str(dob_prob * 100)[0:5],
                "label_dob_prob": str(label_dob_prob * 100)[0:5],
                "sex": str(sex),
                "sex_prob": str(sex_prob * 100)[0:5],
                "label_sex_prob": str(label_sex_prob * 100)[0:5],
                "nationality": str(nationality),
                "nationality_prob": str(nationality_prob * 100)[0:5],
                "label_nationality_prob": str(label_nationality_prob * 100)[0:5],
                "home": str(home),
                "home_prob": str(home_prob * 100)[0:5],
                "label_home_prob": str(label_home_prob * 100)[0:5],
                "address": str(address),
                "address_prob": str(address_prob * 100)[0:5],
                "label_address_prob": str(label_address_prob * 100)[0:5],
                "doe": str(doe),
                "doe_prob": str(doe_prob * 100)[0:5],
                "label_doe_prob": str(label_doe_prob * 100)[0:5],
                "QR_code": str(QR_code),
                "text_overall_score": str(text_overall_score * 100)[0:5],
                "label_overall_score": str(label_overall_score * 100)[0:5],
                "address_entities": {
                    "province": "",
                    "district": "",
                    "ward": "",
                    "street": ""
                },
                "type": "cccd_front"
            }
        ]
    }
    return format_data
