import cv2
import logging

logging.basicConfig(level=logging.INFO)


def _make_picture(image_name: str, image_path: str = "."):
    """Make picture and store it in specified folder

    :param image_name: Name of image
    :param image_path: path where image shall be stored
    """
    logging.info('Start making picture by using camera')
    cap = cv2.VideoCapture(0)

    # Check whether user selected camera is opened successfully.
    if not (cap.isOpened()):
        logging.error("Could not open video device")
    else:
        # get frame
        ret, frame = cap.read()
        # resize image
        frame = cv2.resize(frame, [1024, 1024])
        # Display the resulting frame
        cv2.imwrite(f"{image_path}/{image_name}.jpg", frame)
        logging.info(f"Stored picture {image_name} in {image_path}")


def _augment_image(image, image_name, image_path="."):
    """Apply augmentation on image"""
    # horizontal flip
    cv2.imwrite(f"{image_path}/{image_name}.jpg", cv2.flip(image, 1))

    # shift color
    image = image + [20, 20, -20]
