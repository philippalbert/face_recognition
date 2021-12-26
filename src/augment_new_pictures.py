import cv2
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def _make_picture(image_name: str, image_path: str = "./data/", augment: bool = False) -> None:
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
        frame = cv2.resize(frame, [128, 128])
        # Display the resulting frame
        cv2.imwrite(f"{image_path}/{image_name}.jpg", frame)
        logging.info(f"Stored picture {image_name} in {image_path}")

    if augment:
        logging.info('Start augmenting images')
        _augment_image(frame, image_name + '_aug')


def _augment_image(image, image_name, image_path="./data/") -> None:
    """Apply augmentation on image"""
    # horizontal flip
    cv2.imwrite(f"{image_path}/{image_name}_hor_flip.jpg", cv2.flip(image, 1))

    # shift color by 20 for every channel
    for i, rgb in enumerate([[20, 20, -20], [20, -20, 20], [-20, 20, 20]]):
        # write image for every intensified channel
        cv2.imwrite(f"{image_path}/{image_name}_channel_{i}.jpg", image + rgb)

    # blur image with gaussian blur
    cv2.imwrite(f"{image_path}/{image_name}_gaus_blur.jpg", cv2.GaussianBlur(image, [19, 19], 0))

    # add gaussian noise to image
    noisy_image = _add_noise(image, method='gaussian')
    cv2.imwrite(f"{image_path}/{image_name}_gaus_noise.jpg", noisy_image)

    # add salt and pepper noise to image
    noisy_image = _add_noise(image, method='salt_pepper')
    cv2.imwrite(f"{image_path}/{image_name}_salt_pepper_noise.jpg", noisy_image)


def _add_noise(image, method='gaussian', sp_amount=0.01):
    """Add noise to a picture

    :Note:
        Two methods are implemented to add noise. One is gaussian method, the other is salt and pepper
    :param image: image to add noise to
    :param method: method to use to add noise
    :param sp_amount: amount of salted and peppered data points
    :return: noisy image
    """
    if method == 'gaussian':
        # define mean
        mean = 0
        # define standard deviation
        sd = 20

        # define 3 dimensional random matrix (aka tensor)
        rand_mat = np.random.normal(mean, sd, image.shape)

        # create image with noise
        n_i = image + rand_mat

    elif method == 'salt_pepper':

        # create empty matrix
        s_p_mat = np.empty(image.shape)

        # fill matrix with nan values
        s_p_mat.fill(np.nan)

        # add min to image
        min_pix = np.random.choice([1, 0], image.shape, p=[1 - sp_amount, sp_amount])
        # multiply random minimum pixels with original image
        n_i = image * min_pix

        # add max to image
        max_pix = np.random.choice([0, 255], image.shape, p=[1 - sp_amount, sp_amount])
        # add noise to image
        n_i = n_i + max_pix
        # make sure no element is greater than 255
        n_i = np.fmin(n_i, 255)

    else:
        raise NotImplementedError('Please make sure to use only implemented methods for adding noise')

    return n_i
