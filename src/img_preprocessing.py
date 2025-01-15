import numpy as np
import cv2

def dict_to_image(image_dict):
    """
    Convert a dictionary containing image bytes to a grayscale image.

    Parameters:
    image_dict (dict): A dictionary with a 'bytes' key containing the image byte string.

    Returns:
    img (numpy.ndarray): The decoded grayscale image.

    Raises:
    TypeError: If the input is not a dictionary with a 'bytes' key.
    """
    
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        byte_string = image_dict['bytes']
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")

