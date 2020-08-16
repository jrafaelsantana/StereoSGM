import os
import struct
import numpy as np
import cv2
import re
from struct import unpack
import pickle
import sys


def load_pfm(file):
    """
    Read a PFM file

    :param filename: Filename of .pfm file

    :return: A array with values
    """

    with open(file, "rb") as f:
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            print("ERROR: Not a valid PFM file", file=sys.stderr)
            sys.exit(1)

        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        line = f.readline().decode('latin-1')
        big_endian = True
        if "-" in line:
            big_endian = False

        samples = width * height * channels
        buffer = f.read(samples * 4)

        if big_endian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)

        return np.flipud(np.asarray(img).reshape(height, width))


def parseCalib(filename):
    """
    Read the calibration file

    :param filename: Calibration file name

    :return: Height, Width and Ndisp values
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    line = lines[4].strip()
    idx = line.find('=')
    width = int(line[idx+1:])

    line = lines[5].strip()
    idx = line.find('=')
    height = int(line[idx+1:])

    line = lines[6].strip()
    idx = line.find('=')
    ndisp = int(line[idx+1:])
    return height, width, ndisp


def normal(mean, std_dev):
    constant1 = 1. / (np.sqrt(2*np.pi) * std_dev)
    constant2 = -1. / (2 * std_dev * std_dev)
    return lambda x: constant1 * np.exp(constant2 * ((x - mean)**2))


def saveDisparity(disparity_map, filename):
    """
    Save the disparity map image

    :param disparity_map: Disparity map from algortihm
    :param filename: File name
    """

    assert len(disparity_map.shape) == 2
    cv2.imwrite(filename, disparity_map)


def write_pfm(file, image, scale = 1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)   


def saveTimeFile(times, path):
    o = open(path, "w")
    o.write("{}".format(times))
    o.close()


def testMk(dirName):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)


def recurMk(path):
    items = path.split("/")
    prefix = "/"
    for item in items:
        prefix = os.path.join(prefix, item)
        testMk(prefix)


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    file = 'obj/' + name + '.pkl'
    if os.path.exists(file):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return None
