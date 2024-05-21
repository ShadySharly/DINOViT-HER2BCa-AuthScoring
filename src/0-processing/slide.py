# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
import shutil
import glob
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import openslide
from openslide import OpenSlideError
import os
from PIL import Image
import re
import util
from util import Time
from functools import reduce
from metadata import *

def open_slide(filename):
  """
  Open a whole-slide image (*.svs, etc).

  Args:
    filename: Name of the slide file.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  try:
    slide = openslide.open_slide(filename)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide

def get_slide_num():
  """
  FunctionDescription

  Args:
  arg: Argument description

  returns:
  Returns description
  """
  slide_list = os.listdir(SLIDE_DIR)
  return len(slide_list)

def open_image(filename):
  """
  Open an image (*.jpg, *.png, etc).

  Args:
    filename: Name of the image file.

  returns:
    A PIL.Image.Image object representing an image.
  """
  print("File Name: " + filename)
  image = Image.open(filename)
  return image
  

def open_image_np(filename):
  """
  Open an image (*.jpg, *.png, etc) as an RGB NumPy array.

  Args:
    filename: Name of the image file.

  returns:
    A NumPy representing an RGB image.
  """
  pil_img = open_image(filename)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img

def get_image_num():
  """
  FunctionDescription

  Args:
  arg: Argument description

  returns:
  Returns description
  """
  image_list = os.listdir(GDC_TCGA_IMAGE_DIR)
  return len(image_list)

def get_image_index(image_name):
  """
  FunctionDescription

  Args:
    arg: Argument description

  returns:
    Returns description
  """
  image_index = image_name.split("-")[1]
  return int(image_index)

def get_last_image_number():
  """
  FunctionDescription

  Args:
    arg: Argument description

  returns:
    Returns description
  """
  image_list = os.listdir(GDC_TCGA_IMAGE_DIR)
  image_id_list = list(map(lambda image_name: int(image_name.split("-")[1]), image_list))
  last_image_index = reduce(lambda x, y: x if x >= y else y, image_id_list)
  print("Last Image Index: " + str(last_image_index))
  return last_image_index

def divide_multisample_images():
  """
  FunctionDescription

  Args:
    arg: Argument description

  returns:
    Returns description
  """
  for root, dirs, files in os.walk(GDC_TCGA_IMAGE_DIR):
    image_list = list(map(lambda file_name: os.path.join(root, file_name), files))
    print(image_list)
    list(map(lambda image_filepath: divide_image(image_filepath), image_list))

def divide_image(image_filepath, option=None):
  """
  FunctionDescription

  Args:
    arg: Argument description

  returns:
    Returns description
  """
  img = open_image(image_filepath)
  img_name = os.path.basename(image_filepath)
  slide_id_origin = SLIDE_PREFIX + img_name.split("-")[1]
  new_image_filepath = os.path.join(GDC_TCGA_MULTI_IMAGE_DIR, img_name)

  # Forced Division
  if (option == 0):
    print("Two Vertical Samples")
    save_samples(slide_id_origin, img, 2, 0)
    shutil.move(image_filepath, new_image_filepath)
    print("Forced Division")

  elif (option == 1):
    print("Two Horizontal Samples")
    save_samples(slide_id_origin, img, 2, 1)
    shutil.move(image_filepath, new_image_filepath)
    print("Forced Division")

  elif (option == 2):
    print("Three Vertical Samples")
    save_samples(slide_id_origin, img, 3, 0)
    shutil.move(image_filepath, new_image_filepath)
    print("Forced Division")

  elif (option == 3):
    print("Three Horizontal Samples")
    save_samples(slide_id_origin, img, 3, 1)
    shutil.move(image_filepath, new_image_filepath)
    print("Forced Division")

  # Check Automated Multi-Sample Image
  elif (is_multisample_image(img, 2, 0)):
    print("Two Vertical Samples")
    save_samples(slide_id_origin, img, 2, 0)
    shutil.move(image_filepath, new_image_filepath)

  elif (is_multisample_image(img, 2, 1)):
    print("Two Horizontal Samples")
    save_samples(slide_id_origin, img, 2, 1)
    shutil.move(image_filepath, new_image_filepath)

  elif (is_multisample_image(img, 3, 0)):
    print("Three Vertical Samples")
    save_samples(slide_id_origin, img, 3, 0)
    shutil.move(image_filepath, new_image_filepath)

  elif (is_multisample_image(img, 3, 1)):
    print("Three Horizontal Samples")
    save_samples(slide_id_origin, img, 3, 1)
    shutil.move(image_filepath, new_image_filepath)

  else:
    print("No Multi-Sample Image")

def get_sample_path(img_width, img_height, sample_img):
  """
  FunctionDescription

  Args:
    arg: Argument description

  returns:
    Returns description
  """
  sample_width, sample_height = sample_img.size

  image_index = str(get_last_image_number() + 1).zfill(4)
  sample_id = SLIDE_PREFIX + image_index
  magnification = str(SCALE_FACTOR) + "x"
  original_dim = str(img_width) + "x" + str(img_height)
  sample_dim = str(sample_width) + "x" + str(sample_height)

  sample_name = sample_id + "-" + magnification + "-" + original_dim + "-" + sample_dim + ".jpg"
  sample_path = os.path.join(GDC_TCGA_IMAGE_DIR, sample_name)
  print("Sample N° " + image_index + "Path: " + sample_path)
  return sample_id, sample_path

def save_samples(slide_id_origin, img, samples_num, orientation):
  """
  FunctionDescription

  Args:
    arg: Argument description

  returns:
    Returns description
  """
  width, height = img.size
  slide_barcode = util.get_slide_barcode(slide_id=slide_id_origin)
  slide_name = util.get_slide_name(slide_id=slide_id_origin)

  if (samples_num == 2 and orientation == 0):
    sample_width = width // 2
    print("Sample Width: " + str(sample_width))
    left_crop = img.crop((0, 0, sample_width, height))
    right_crop = img.crop((sample_width + 1, 0, width, height))

    left_sample_id, left_crop_path = get_sample_path(width, height, left_crop)
    left_crop.save(left_crop_path)

    right_sample_id, right_crop_path = get_sample_path(width, height, right_crop)
    right_crop.save(right_crop_path)

    util.add_image_data_row(left_sample_id, slide_barcode, slide_name, slide_id_origin)
    util.add_image_data_row(right_sample_id, slide_barcode, slide_name, slide_id_origin)

    print("Saved Two Vertical Samples")

  elif (samples_num == 2 and orientation == 1):
    sample_height = height // 2
    print("Sample Height: " + str(sample_height))
    upper_crop = img.crop((0, 0, width, sample_height))
    lower_crop = img.crop((0, sample_height + 1, width, height))
 
    upper_sample_id, upper_crop_path = get_sample_path(width, height, upper_crop)
    upper_crop.save(upper_crop_path)

    lower_sample_id, lower_crop_path = get_sample_path(width, height, lower_crop)
    lower_crop.save(lower_crop_path)

    util.add_image_data_row(upper_sample_id, slide_barcode, slide_name, slide_id_origin)
    util.add_image_data_row(lower_sample_id, slide_barcode, slide_name, slide_id_origin)

    print("Saved Two Horizontal Samples")

  elif (samples_num == 3 and orientation == 0):
    sample_width = width // 3
    print("Sample Width: " + str(sample_width))
    left_crop = img.crop((0, 0, sample_width, height))
    center_crop = img.crop((sample_width + 1, 0, 2 * sample_width + 1, height))
    right_crop = img.crop((2 * sample_width + 2, 0, width, height))

    left_sample_id, left_crop_path = get_sample_path(width, height, left_crop)
    left_crop.save(left_crop_path)

    center_sample_id, center_crop_path = get_sample_path(width, height, center_crop)
    center_crop.save(center_crop_path)

    right_sample_id, right_crop_path = get_sample_path(width, height, right_crop)
    right_crop.save(right_crop_path)

    util.add_image_data_row(left_sample_id, slide_barcode, slide_name, slide_id_origin)
    util.add_image_data_row(center_sample_id, slide_barcode, slide_name, slide_id_origin)
    util.add_image_data_row(right_sample_id, slide_barcode, slide_name, slide_id_origin)

    print("Saved Three Vertical Samples")

  elif (samples_num == 3 and orientation == 1):
    sample_height = height // 3
    print("Sample Height: " + str(sample_height))
    upper_crop = img.crop((0, 0, width, sample_height))
    center_crop = img.crop((0, sample_height + 1, width, 2 * sample_height + 1))
    lower_crop = img.crop((0, 2 * sample_height + 2, width, height))

    upper_sample_id, upper_crop_path = get_sample_path(width, height, upper_crop)
    upper_crop.save(upper_crop_path)

    center_sample_id, center_crop_path = get_sample_path(width, height, center_crop)
    center_crop.save(center_crop_path)

    lower_sample_id, lower_crop_path = get_sample_path(width, height, lower_crop)
    lower_crop.save(lower_crop_path)

    util.add_image_data_row(upper_sample_id, slide_barcode, slide_name, slide_id_origin)
    util.add_image_data_row(center_sample_id, slide_barcode, slide_name, slide_id_origin)
    util.add_image_data_row(lower_sample_id, slide_barcode, slide_name, slide_id_origin)

    print("Saved Three Horizontal Samples")

def restore_image_dataset():
  """
  FunctionDescription

  Args:
    arg: Argument description

  returns:
    Returns description
  """
  img_list = os.listdir(GDC_TCGA_IMAGE_DIR)
  img_multi_list = os.listdir(GDC_TCGA_MULTI_IMAGE_DIR)
  img_list.sort()
  img_sample_list = img_list[1041:]
  # Get Back Multi-Sample images into "images" directory
  list(map(lambda file_name: move_image_multi(file_name), img_multi_list))
  # Remove Sample Images from "image" directory
  list(map(lambda file_name: remove_sample(file_name), img_sample_list))
  

def remove_sample(file_name):
  file_path = os.path.join(GDC_TCGA_IMAGE_DIR, file_name)
  os.remove(file_path)

def move_image_multi(file_name):
  multi_image_path = os.path.join(GDC_TCGA_MULTI_IMAGE_DIR, file_name)
  image_path = os.path.join(GDC_TCGA_IMAGE_DIR, file_name)
  shutil.move(multi_image_path, image_path)

def is_multisample_image(img, samples_num, orientation):
  """
  Check if the given file corresponds to a slide including two or more samples of itself.

  Args:
    display_all_properties: If True, display all available slide properties.

  Returns:
    True if it's multisample slide or false if not (just one sample in slide)
  """
  delimeter_crop = get_sample_delimiter_crop(img, samples_num, orientation)
  pixel_crop_mean = np.mean(delimeter_crop)
  print("Delimeter Crop Pixel Mean: " + str(pixel_crop_mean))
  return pixel_crop_mean >= WHITENESS_TRESHOLD

def get_sample_delimiter_crop(img, samples_num, orientation):
  width, height = img.size
  center_pixel_x = width / samples_num
  center_pixel_y = height / samples_num

  if (orientation == 0):
    img_crop = get_vertical_crop(img, center_pixel_x, height)

  elif (orientation == 1):
    img_crop = get_horizontal_crop(img, center_pixel_y, width)
    
  np_crop = util.pil_to_np_rgb(img_crop)
  return np_crop

def get_vertical_crop(img, center_pixel, img_height):
  """
  FunctionDescription

  Args:
    arg: Argument description

  returns:
    Returns description
  """
  img_crop = img.crop((center_pixel - CROP_RATIO, 0, center_pixel + CROP_RATIO, img_height))
  return img_crop


def get_horizontal_crop(img, center_pixel, img_width):
  """
  FunctionDescription

  Args:
    arg: Argument description

  returns:
    Returns description
  """
  img_crop = img.crop((0, center_pixel - CROP_RATIO, img_width, center_pixel + CROP_RATIO))
  return img_crop

def get_training_slide_path(slide_number):
  """
  Convert slide number to a path to the corresponding WSI training slide file.

  Example:
    5 -> ../data/training_slides/TUPAC-TR-005.svs

  Args:
    slide_number: The slide number.

  Returns:
    Path to the WSI training slide file.
  """
  padded_sl_num = str(slide_number).zfill(4)
  slide_filepath = os.path.join(SLIDE_DIR, SLIDE_PREFIX + padded_sl_num + "." + SVS)

  if(os.path.isfile(slide_filepath)):
    return slide_filepath
  
  print("No existing slide N°: " + str(slide_number))
  return False

def get_tile_image_path(tile):
  """
  Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
  pixel width, and pixel height.

  Args:
    tile: Tile object.

  Returns:
    Path to image tile.
  """
  t = tile
  padded_sl_num = str(t.slide_num).zfill(4)
  tile_path = os.path.join(TILE_IMAGE_DIR, padded_sl_num,
                           SLIDE_PREFIX + padded_sl_num + "-" + TILE_SUFFIX + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
                             t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + "." + JPG)
  return tile_path


def get_tile_image_path_by_slide_row_col(slide_number, row, col):
  """
  Obtain tile image path using wildcard lookup with slide number, row, and column.

  Args:
    slide_number: The slide number.
    row: The row.
    col: The column.

  Returns:
    Path to image tile.
  """
  padded_sl_num = str(slide_number).zfill(4)
  wilcard_path = os.path.join(TILE_IMAGE_DIR, padded_sl_num,
                              SLIDE_PREFIX + padded_sl_num + "-" + TILE_SUFFIX + "-r%d-c%d-*." % (
                                row, col) + JPG)
  img_path = glob.glob(wilcard_path)[0]
  return img_path


def get_training_image_path(slide_number, large_w=None, large_h=None, small_w=None, small_h=None):
  """
  Convert slide number and optional dimensions to a training image path. If no dimensions are supplied,
  the corresponding file based on the slide number will be looked up in the file system using a wildcard.

  Example:
    5 -> ../data/training_png/TUPAC-TR-005-32x-49920x108288-1560x3384.png

  Args:
    slide_number: The slide number.
    large_w: Large image width.
    large_h: Large image height.
    small_w: Small image width.
    small_h: Small image height.

  Returns:
     Path to the image file.
  """

  padded_sl_num = str(slide_number).zfill(4)
  print("Slide number: " + str(padded_sl_num))
  if large_w is None and large_h is None and small_w is None and small_h is None:
    wildcard_path = os.path.join(GDC_TCGA_IMAGE_DIR, SLIDE_PREFIX + padded_sl_num + "*." + IMAGE_EXT)

    if(len(glob.glob(wildcard_path)) > 0):
      img_path = glob.glob(wildcard_path)[0]
      return img_path
  
    print("No existing image N°: " + str(slide_number))
    return False

  else:
    img_path = os.path.join(GDC_TCGA_IMAGE_DIR, SLIDE_PREFIX + padded_sl_num + "-" + str(
      SCALE_FACTOR) + "x-" + str(
      large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + IMAGE_EXT)
  return img_path


def get_training_thumbnail_path(slide_number, large_w=None, large_h=None, small_w=None, small_h=None):
  """
  Convert slide number and optional dimensions to a training thumbnail path. If no dimensions are
  supplied, the corresponding file based on the slide number will be looked up in the file system using a wildcard.

  Example:
    5 -> ../data/training_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384.jpg

  Args:
    slide_number: The slide number.
    large_w: Large image width.
    large_h: Large image height.
    small_w: Small image width.
    small_h: Small image height.

  Returns:
     Path to the thumbnail file.
  """
  padded_sl_num = str(slide_number).zfill(4)
  if large_w is None and large_h is None and small_w is None and small_h is None:
    wildcard_path = os.path.join(GDC_TCGA_THUMBNAIL_DIR, SLIDE_PREFIX + padded_sl_num + "*." + IMAGE_EXT)

    if(len(glob.glob(wildcard_path)) > 0):
      img_path = glob.glob(wildcard_path)[0]
      return img_path
  
    print("No existing thumbnail N°: " + str(slide_number))
    return False
  
  else:
    img_path = os.path.join(GDC_TCGA_THUMBNAIL_DIR, SLIDE_PREFIX + padded_sl_num + "-" + str(
      SCALE_FACTOR) + "x-" + str(
      large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + IMAGE_EXT)
  return img_path


def get_filter_image_path(slide_number, filter_number, filter_name_info):
  """
  Convert slide number, filter number, and text to a path to a filter image file.

  Example:
    5, 1, "rgb" -> ../data/filter_png/TUPAC-TR-005-001-rgb.png

  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.

  Returns:
    Path to the filter image file.
  """
  dir = FILTER_IMAGE_DIR
  if not os.path.exists(dir):
    os.makedirs(dir)
  img_path = os.path.join(dir, get_filter_image_filename(slide_number, filter_number, filter_name_info))
  return img_path


def get_filter_thumbnail_path(slide_number, filter_number, filter_name_info):
  """
  Convert slide number, filter number, and text to a path to a filter thumbnail file.

  Example:
    5, 1, "rgb" -> ../data/filter_thumbnail_jpg/TUPAC-TR-005-001-rgb.jpg

  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.

  Returns:
    Path to the filter thumbnail file.
  """
  dir = FILTER_THUMBNAIL_DIR
  if not os.path.exists(dir):
    os.makedirs(dir)
  img_path = os.path.join(dir, get_filter_image_filename(slide_number, filter_number, filter_name_info, thumbnail=True))
  return img_path


def get_filter_image_filename(slide_number, filter_number, filter_name_info, thumbnail=False):
  """
  Convert slide number, filter number, and text to a filter file name.

  Example:
    5, 1, "rgb", False -> TUPAC-TR-005-001-rgb.png
    5, 1, "rgb", True -> TUPAC-TR-005-001-rgb.jpg

  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.
    thumbnail: If True, produce thumbnail filename.

  Returns:
    The filter image or thumbnail file name.
  """
  if thumbnail:
    ext = JPG
  else:
    ext = JPG
  padded_sl_num = str(slide_number).zfill(4)
  padded_fi_num = str(filter_number).zfill(4)
  img_filename = SLIDE_PREFIX + padded_sl_num + "-" + padded_fi_num + "-" + FILTER_SUFFIX + filter_name_info + "." + ext
  return img_filename


def get_tile_summary_image_path(slide_number):
  """
  Convert slide number to a path to a tile summary image file.

  Example:
    5 -> ../data/tile_summary_png/TUPAC-TR-005-tile_summary.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the tile summary image file.
  """
  if not os.path.exists(TILE_SUMMARY_DIR):
    os.makedirs(TILE_SUMMARY_DIR)
  img_path = os.path.join(TILE_SUMMARY_DIR, get_tile_summary_image_filename(slide_number))
  return img_path


def get_tile_summary_thumbnail_path(slide_number):
  """
  Convert slide number to a path to a tile summary thumbnail file.

  Example:
    5 -> ../data/tile_summary_thumbnail_jpg/TUPAC-TR-005-tile_summary.jpg

  Args:
    slide_number: The slide number.

  Returns:
    Path to the tile summary thumbnail file.
  """
  if not os.path.exists(TILE_SUMMARY_THUMBNAIL_DIR):
    os.makedirs(TILE_SUMMARY_THUMBNAIL_DIR)
  img_path = os.path.join(TILE_SUMMARY_THUMBNAIL_DIR, get_tile_summary_image_filename(slide_number, thumbnail=True))
  return img_path


def get_tile_summary_on_original_image_path(slide_number):
  """
  Convert slide number to a path to a tile summary on original image file.

  Example:
    5 -> ../data/tile_summary_on_original_png/TUPAC-TR-005-tile_summary.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the tile summary on original image file.
  """
  if not os.path.exists(TILE_SUMMARY_ON_ORIGINAL_DIR):
    os.makedirs(TILE_SUMMARY_ON_ORIGINAL_DIR)
  img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_DIR, get_tile_summary_image_filename(slide_number))
  return img_path


def get_tile_summary_on_original_thumbnail_path(slide_number):
  """
  Convert slide number to a path to a tile summary on original thumbnail file.

  Example:
    5 -> ../data/tile_summary_on_original_thumbnail_jpg/TUPAC-TR-005-tile_summary.jpg

  Args:
    slide_number: The slide number.

  Returns:
    Path to the tile summary on original thumbnail file.
  """
  if not os.path.exists(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR):
    os.makedirs(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR)
  img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR,
                          get_tile_summary_image_filename(slide_number, thumbnail=True))
  return img_path


def get_top_tiles_on_original_image_path(slide_number):
  """
  Convert slide number to a path to a top tiles on original image file.

  Example:
    5 -> ../data/top_tiles_on_original_png/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the top tiles on original image file.
  """
  if not os.path.exists(TOP_TILES_ON_ORIGINAL_DIR):
    os.makedirs(TOP_TILES_ON_ORIGINAL_DIR)
  img_path = os.path.join(TOP_TILES_ON_ORIGINAL_DIR, get_top_tiles_image_filename(slide_number))
  return img_path


def get_top_tiles_on_original_thumbnail_path(slide_number):
  """
  Convert slide number to a path to a top tiles on original thumbnail file.

  Example:
    5 -> ../data/top_tiles_on_original_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.jpg

  Args:
    slide_number: The slide number.

  Returns:
    Path to the top tiles on original thumbnail file.
  """
  if not os.path.exists(TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR):
    os.makedirs(TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR)
  img_path = os.path.join(TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR,
                          get_top_tiles_image_filename(slide_number, thumbnail=True))
  return img_path


def get_tile_summary_image_filename(slide_number, thumbnail=False):
  """
  Convert slide number to a tile summary image file name.

  Example:
    5, False -> TUPAC-TR-005-tile_summary.png
    5, True -> TUPAC-TR-005-tile_summary.jpg

  Args:
    slide_number: The slide number.
    thumbnail: If True, produce thumbnail filename.

  Returns:
    The tile summary image file name.
  """
  if thumbnail:
    ext = JPG
  else:
    ext = JPG
  padded_sl_num = str(slide_number).zfill(4)

  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_filename = SLIDE_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_SUMMARY_SUFFIX + "." + ext

  return img_filename


def get_top_tiles_image_filename(slide_number, thumbnail=False):
  """
  Convert slide number to a top tiles image file name.

  Example:
    5, False -> TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png
    5, True -> TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.jpg

  Args:
    slide_number: The slide number.
    thumbnail: If True, produce thumbnail filename.

  Returns:
    The top tiles image file name.
  """
  if thumbnail:
    ext = JPG
  else:
    ext = JPG
  padded_sl_num = str(slide_number).zfill(4)

  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_filename = SLIDE_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TOP_TILES_SUFFIX + "." + ext

  return img_filename


def get_top_tiles_image_path(slide_number):
  """
  Convert slide number to a path to a top tiles image file.

  Example:
    5 -> ../data/top_tiles_png/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the top tiles image file.
  """
  if not os.path.exists(TOP_TILES_DIR):
    os.makedirs(TOP_TILES_DIR)
  img_path = os.path.join(TOP_TILES_DIR, get_top_tiles_image_filename(slide_number))
  return img_path


def get_top_tiles_thumbnail_path(slide_number):
  """
  Convert slide number to a path to a tile summary thumbnail file.

  Example:
    5 -> ../data/top_tiles_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.jpg
  Args:
    slide_number: The slide number.

  Returns:
    Path to the top tiles thumbnail file.
  """
  if not os.path.exists(TOP_TILES_THUMBNAIL_DIR):
    os.makedirs(TOP_TILES_THUMBNAIL_DIR)
  img_path = os.path.join(TOP_TILES_THUMBNAIL_DIR, get_top_tiles_image_filename(slide_number, thumbnail=True))
  return img_path


def get_tile_data_filename(slide_number):
  """
  Convert slide number to a tile data file name.

  Example:
    5 -> TUPAC-TR-005-32x-49920x108288-1560x3384-tile_data.csv

  Args:
    slide_number: The slide number.

  Returns:
    The tile data file name.
  """
  padded_sl_num = str(slide_number).zfill(4)

  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  data_filename = SLIDE_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_DATA_SUFFIX + ".csv"

  return data_filename


def get_tile_data_path(slide_number):
  """
  Convert slide number to a path to a tile data file.

  Example:
    5 -> ../data/tile_data/TUPAC-TR-005-32x-49920x108288-1560x3384-tile_data.csv

  Args:
    slide_number: The slide number.

  Returns:
    Path to the tile data file.
  """
  if not os.path.exists(TILE_DATA_DIR):
    os.makedirs(TILE_DATA_DIR)
  file_path = os.path.join(TILE_DATA_DIR, get_tile_data_filename(slide_number))
  return file_path


def get_filter_image_result(slide_number):
  """
  Convert slide number to the path to the file that is the final result of filtering.

  Example:
    5 -> ../data/filter_png/TUPAC-TR-005-32x-49920x108288-1560x3384-filtered.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the filter image file.
  """
  padded_sl_num = str(slide_number).zfill(4)
  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_path = os.path.join(FILTER_IMAGE_DIR, SLIDE_PREFIX + padded_sl_num + "-" + str(
    SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
    small_h) + "-" + FILTER_RESULT_TEXT + "." + JPG)
  return img_path


def get_filter_thumbnail_result(slide_number):
  """
  Convert slide number to the path to the file that is the final thumbnail result of filtering.

  Example:
    5 -> ../data/filter_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384-filtered.jpg

  Args:
    slide_number: The slide number.

  Returns:
    Path to the filter thumbnail file.
  """
  padded_sl_num = str(slide_number).zfill(4)
  training_img_path = get_training_image_path(slide_number)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_path = os.path.join(FILTER_THUMBNAIL_DIR, SLIDE_PREFIX + padded_sl_num + "-" + str(
    SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
    small_h) + "-" + FILTER_RESULT_TEXT + "." + JPG)
  return img_path


def parse_dimensions_from_image_filename(filename):
  """
  Parse an image filename to extract the original width and height and the converted width and height.

  Example:
    "TUPAC-TR-011-32x-97103x79079-3034x2471-tile_summary.png" -> (97103, 79079, 3034, 2471)

  Args:
    filename: The image filename.

  Returns:
    Tuple consisting of the original width, original height, the converted width, and the converted height.
  """
  m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
  large_w = int(m.group(1))
  large_h = int(m.group(2))
  small_w = int(m.group(3))
  small_h = int(m.group(4))
  return large_w, large_h, small_w, small_h


def small_to_large_mapping(small_pixel, large_dimensions):
  """
  Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.

  Args:
    small_pixel: The scaled-down width and height.
    large_dimensions: The width and height of the original whole-slide image.

  Returns:
    Tuple consisting of the scaled-up width and height.
  """
  small_x, small_y = small_pixel
  large_w, large_h = large_dimensions
  large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
  large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
  return large_x, large_y


def training_slide_to_image(slide_number):
  """
  Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.

  Args:
    slide_number: The slide number.
  """

  img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_number)

  img_path = get_training_image_path(slide_number, large_w, large_h, new_w, new_h)
  print("Saving image to: " + img_path)
  if not os.path.exists(GDC_TCGA_IMAGE_DIR):
    os.makedirs(GDC_TCGA_IMAGE_DIR)
  img.save(img_path)

  thumbnail_path = get_training_thumbnail_path(slide_number, large_w, large_h, new_w, new_h)
  save_thumbnail(img, THUMBNAIL_SIZE, thumbnail_path)  

def slide_to_scaled_pil_image(slide_number, scale_factor=None):
  """
  Convert a WSI training slide to a scaled-down PIL image.

  Args:
    slide_number: The slide number.

  Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
  """
  
  slide_filepath = get_training_slide_path(slide_number)
  print("Opening Slide #%s: %s" % (slide_number, slide_filepath))
  slide = open_slide(slide_filepath)
  scale_factor, level = get_scale_down_level(scale_factor)
  original_w, original_h = slide.dimensions
  new_w = slide.level_dimensions[level][0]
  new_h = slide.level_dimensions[level][1]
  whole_slide_image = slide.read_region((0, 0), level, (new_w, new_h))
  img = whole_slide_image.convert("RGB")
  '''
  whole_slide_image = whole_slide_image.convert("RGB")
  
  img = whole_slide_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
  '''
  return img, original_w, original_h, new_w, new_h

def get_scale_down_level(scale_factor):
  """
    Get the level correspondent to the scale factor input (0, 1, 2 or 3) supported.

  Args:
    scale_factor: Scale factor value (40, 20, 10, 8).

  Returns:
    Tuple consisting of the scale factor input and the correspondent level.
  """
  supported_scales = [8, 10, 20, 40]

  if(scale_factor not in supported_scales):
    scale_factor = SCALE_FACTOR
    level = map_scale_factor_to_level(scale_factor)

  return scale_factor, level

def map_scale_factor_to_level(scale_factor=None):
  """
    Get the level correspondent to the scale factor input (0, 1, 2 or 3) supported.

  Args:
    scale_factor: Scale factor value (40, 20, 10, 8). Scale factor 20 as default value and level 1 so on.

  Returns:
    Level correspondent to the scale factor input.
  """
  level = 1
  if(scale_factor == 40):
    level = 0
  elif(scale_factor == 20):
    level = 1
  elif(scale_factor == 10):
    level = 2
  elif(scale_factor == 8):
    level = 3
  return level

def slide_to_scaled_np_image(slide_number):
  """
  Convert a WSI training slide to a scaled-down NumPy image.

  Args:
    slide_number: The slide number.

  Returns:
    Tuple consisting of scaled-down NumPy image, original width, original height, new width, and new height.
  """
  pil_img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_number)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img, large_w, large_h, new_w, new_h


def show_slide(slide_number):
  """
  Display a WSI slide on the screen, where the slide has been scaled down and converted to a PIL image.

  Args:
    slide_number: The slide number.
  """
  pil_img = slide_to_scaled_pil_image(slide_number)[0]
  pil_img.show()


def save_thumbnail(pil_img, size, path, display_path=False):
  """
  Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.

  Args:
    pil_img: The PIL image to save as a thumbnail.
    size:  The maximum width or height of the thumbnail.
    path: The path to the thumbnail.
    display_path: If True, display thumbnail path in console.
  """
  max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
  img = pil_img.resize(max_size, Image.BILINEAR)
  if display_path:
    print("Saving thumbnail to: " + path)
  dir = os.path.dirname(path)
  if dir != '' and not os.path.exists(dir):
    os.makedirs(dir)
  img.save(path)


def get_num_training_slides():
  """
  Obtain the total number of WSI training slide images.

  Returns:
    The total number of WSI training slide images.
  """
  num_training_slides = len(glob.glob1(SLIDE_DIR, "*." + SVS))
  return num_training_slides

def get_num_slides():
  """
  Obtain the total number of WSI slides inside the ./slide dir.

  Returns:
    The total number of WSI slides.
  """
  num_training_slides = len(glob.glob1(SLIDE_DIR, "*." + SVS))
  return num_training_slides


def training_slide_range_to_images(start_ind, end_ind):
  """  
  Convert a range of WSI training slides to smaller images (in a format such as jpg or png).

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).

  Returns:
    The starting index and the ending index of the slides that were converted.
  """
  for slide_num in range(start_ind, end_ind + 1):
    slide_path = get_training_slide_path(slide_num)
    print("Trying to convert slide N°: " + str(slide_num))

    if(slide_path != False):
      training_slide_to_image(slide_num)

  return (start_ind, end_ind)

def singleprocess_training_slides_to_images():
  """
  Convert all WSI training slides to smaller images using a single process.
  """
  t = Time()

  num_train_images = get_slide_num()
  training_slide_range_to_images(1, num_train_images)

  t.elapsed_display()


def multiprocess_training_slides_to_images():
  """
  Convert all WSI training slides to smaller images using multiple processes (one process per core).
  Each process will process a range of slide numbers.
  """
  timer = Time()

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(num_processes)

  num_train_images = get_slide_num()

  if num_processes > num_train_images:
    num_processes = num_train_images
  images_per_process = num_train_images / num_processes

  print("Number of processes: " + str(num_processes))
  print("Number of training images: " + str(num_train_images))

  # each task specifies a range of slides
  tasks = []
  for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * images_per_process + 1
    end_index = num_process * images_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    tasks.append((start_index, end_index))
    if start_index == end_index:
      print("Task #" + str(num_process) + ": Process slide " + str(start_index))
    else:
      print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:

    results.append(pool.apply_async(training_slide_range_to_images, t))

  for result in results:
    (start_ind, end_ind) = result.get()
    if start_ind == end_ind:
      print("Done converting slide %d" % start_ind)
    else:
      print("Done converting slides %d through %d" % (start_ind, end_ind))

  timer.elapsed_display()


def slide_stats():
  """
  Display statistics/graphs about training slides.
  """
  t = Time()

  if not os.path.exists(STATS_DIR):
    os.makedirs(STATS_DIR)

  num_train_images = get_num_training_slides()
  slide_stats = []
  for slide_num in range(1, num_train_images + 1):
    slide_filepath = get_training_slide_path(slide_num)
    print("Opening Slide #%d: %s" % (slide_num, slide_filepath))
    slide = open_slide(slide_filepath)
    (width, height) = slide.dimensions
    print("  Dimensions: {:,d} x {:,d}".format(width, height))
    slide_stats.append((width, height))

  max_width = 0
  max_height = 0
  min_width = sys.maxsize
  min_height = sys.maxsize
  total_width = 0
  total_height = 0
  total_size = 0
  which_max_width = 0
  which_max_height = 0
  which_min_width = 0
  which_min_height = 0
  max_size = 0
  min_size = sys.maxsize
  which_max_size = 0
  which_min_size = 0
  for z in range(0, num_train_images):
    (width, height) = slide_stats[z]
    if width > max_width:
      max_width = width
      which_max_width = z + 1
    if width < min_width:
      min_width = width
      which_min_width = z + 1
    if height > max_height:
      max_height = height
      which_max_height = z + 1
    if height < min_height:
      min_height = height
      which_min_height = z + 1
    size = width * height
    if size > max_size:
      max_size = size
      which_max_size = z + 1
    if size < min_size:
      min_size = size
      which_min_size = z + 1
    total_width = total_width + width
    total_height = total_height + height
    total_size = total_size + size

  avg_width = total_width / num_train_images
  avg_height = total_height / num_train_images
  avg_size = total_size / num_train_images

  stats_string = ""
  stats_string += "%-11s {:14,d} pixels (slide #%d)".format(max_width) % ("Max width:", which_max_width)
  stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(max_height) % ("Max height:", which_max_height)
  stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(max_size) % ("Max size:", which_max_size)
  stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_width) % ("Min width:", which_min_width)
  stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_height) % ("Min height:", which_min_height)
  stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_size) % ("Min size:", which_min_size)
  stats_string += "\n%-11s {:14,d} pixels".format(round(avg_width)) % "Avg width:"
  stats_string += "\n%-11s {:14,d} pixels".format(round(avg_height)) % "Avg height:"
  stats_string += "\n%-11s {:14,d} pixels".format(round(avg_size)) % "Avg size:"
  stats_string += "\n"
  print(stats_string)

  stats_string += "\nslide number,width,height"
  for i in range(0, len(slide_stats)):
    (width, height) = slide_stats[i]
    stats_string += "\n%d,%d,%d" % (i + 1, width, height)
  stats_string += "\n"

  stats_file = open(os.path.join(STATS_DIR, "stats.txt"), "w")
  stats_file.write(stats_string)
  stats_file.close()

  t.elapsed_display()

  x, y = zip(*slide_stats)
  colors = np.random.rand(num_train_images)
  sizes = [10 for n in range(num_train_images)]
  plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
  plt.xlabel("width (pixels)")
  plt.ylabel("height (pixels)")
  plt.title("SVS Image Sizes")
  plt.set_cmap("prism")
  plt.tight_layout()
  plt.savefig(os.path.join(STATS_DIR, "svs-image-sizes.png"))
  plt.show()

  plt.clf()
  plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
  plt.xlabel("width (pixels)")
  plt.ylabel("height (pixels)")
  plt.title("SVS Image Sizes (Labeled with slide numbers)")
  plt.set_cmap("prism")
  for i in range(num_train_images):
    snum = i + 1
    plt.annotate(str(snum), (x[i], y[i]))
  plt.tight_layout()
  plt.savefig(os.path.join(STATS_DIR, "svs-image-sizes-slide-numbers.png"))
  plt.show()

  plt.clf()
  area = [w * h / 1000000 for (w, h) in slide_stats]
  plt.hist(area, bins=64)
  plt.xlabel("width x height (M of pixels)")
  plt.ylabel("# images")
  plt.title("Distribution of image sizes in millions of pixels")
  plt.tight_layout()
  plt.savefig(os.path.join(STATS_DIR, "distribution-of-svs-image-sizes.png"))
  plt.show()

  plt.clf()
  whratio = [w / h for (w, h) in slide_stats]
  plt.hist(whratio, bins=64)
  plt.xlabel("width to height ratio")
  plt.ylabel("# images")
  plt.title("Image shapes (width to height)")
  plt.tight_layout()
  plt.savefig(os.path.join(STATS_DIR, "w-to-h.png"))
  plt.show()

  plt.clf()
  hwratio = [h / w for (w, h) in slide_stats]
  plt.hist(hwratio, bins=64)
  plt.xlabel("height to width ratio")
  plt.ylabel("# images")
  plt.title("Image shapes (height to width)")
  plt.tight_layout()
  plt.savefig(os.path.join(STATS_DIR, "h-to-w.png"))
  plt.show()


def slide_info(display_all_properties=False):
  """
  Display information (such as properties) about training images.

  Args:
    display_all_properties: If True, display all available slide properties.
  """
  t = Time()

  num_train_images = get_num_training_slides()
  obj_pow_20_list = []
  obj_pow_40_list = []
  obj_pow_other_list = []
  for slide_num in range(1, num_train_images + 1):
    slide_filepath = get_training_slide_path(slide_num)
    print("\nOpening Slide #%d: %s" % (slide_num, slide_filepath))
    slide = open_slide(slide_filepath)
    print("Level count: %d" % slide.level_count)
    print("Level dimensions: " + str(slide.level_dimensions))
    print("Level downsamples: " + str(slide.level_downsamples))
    print("Dimensions: " + str(slide.dimensions))
    objective_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    print("Objective power: " + str(objective_power))
    if objective_power == 20:
      obj_pow_20_list.append(slide_num)
    elif objective_power == 40:
      obj_pow_40_list.append(slide_num)
    else:
      obj_pow_other_list.append(slide_num)
    print("Associated images:")
    for ai_key in slide.associated_images.keys():
      print("  " + str(ai_key) + ": " + str(slide.associated_images.get(ai_key)))
    print("Format: " + str(slide.detect_format(slide_filepath)))
    if display_all_properties:
      print("Properties:")
      for prop_key in slide.properties.keys():
        print("  Property: " + str(prop_key) + ", value: " + str(slide.properties.geGt(prop_key)))

  print("\n\nSlide Magnifications:")
  print("  20x Slides: " + str(obj_pow_20_list))
  print("  40x Slides: " + str(obj_pow_40_list))
  print("  ??x Slides: " + str(obj_pow_other_list) + "\n")

  t.elapsed_display()