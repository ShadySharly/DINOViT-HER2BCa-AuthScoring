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

import os
import glob
import shutil
from os import path
import csv
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False
DATA = "data"
SLIDE = "slide"
MANIFEST = "manifest"
IMAGE = "image"
TXT = "txt"
SVS = "svs"
DOT = "."
GDC_TCGA = "GDC_TCGA"

# GENERAL PATHS
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
BASE_DIR = os.path.join(ROOT_DIR, DATA)
SLIDE_DIR = os.path.join(BASE_DIR, GDC_TCGA, SLIDE)
MANIFEST_DIR = os.path.join(BASE_DIR, GDC_TCGA, MANIFEST)
IMAGE_DIR = os.path.join(BASE_DIR, GDC_TCGA, IMAGE)


# GDC_TCGA PATHS
GDC_TCGA_DATA_DIR = os.path.join(BASE_DIR, GDC_TCGA)
GDC_TCGA_EVAL_DIR = os.path.join(GDC_TCGA_DATA_DIR, "evaluation")
GDC_TCGA_TRAIN_DIR = os.path.join(GDC_TCGA_DATA_DIR, "training")
GDC_TCGA_MANIFEST_DIR = os.path.join(GDC_TCGA_DATA_DIR, "manifest")
GDC_TCGA_SLIDE_DIR = os.path.join(GDC_TCGA_DATA_DIR, "slide")

# UCH_CPDAI PATHS
UCH_CPDAI_DATA_DIR = os.path.join(BASE_DIR, "UCH_CPDAI")


def pil_to_np_rgb(pil_img):
  """
  Convert a PIL Image to a NumPy array.

  Note that RGB PIL (w, h) -> NumPy (h, w, 3).

  Args:
    pil_img: The PIL Image.

  Returns:
    The PIL image converted to a NumPy array.
  """
  t = Time()
  rgb = np.asarray(pil_img)
  np_info(rgb, "RGB", t.elapsed())
  return rgb


def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.

  Args:
    np_img: The image represented as a NumPy array.

  Returns:
     The NumPy array converted to a PIL Image.
  """
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return Image.fromarray(np_img)


def np_info(np_arr, name=None, elapsed=None):
  """
  Display information (shape, type, max, min, etc) about a NumPy array.

  Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
  """

  if name is None:
    name = "NumPy Array"
  if elapsed is None:
    elapsed = "---"

  if ADDITIONAL_NP_STATS is False:
    print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
  else:
    # np_arr = np.asarray(np_arr)
    max = np_arr.max()
    min = np_arr.min()
    mean = np_arr.mean()
    is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
    print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
      name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


def display_img(np_img, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
  """
  Convert a NumPy array to a PIL image, add text to the image, and display the image.

  Args:
    np_img: Image as a NumPy array.
    text: The text to add to the image.
    font_path: The path to the font to use.
    size: The font size
    color: The font color
    background: The background color
    border: The border color
    bg: If True, add rectangle background behind text
  """
  result = np_to_pil(np_img)
  # if gray, convert to RGB for display
  if result.mode == 'L':
    result = result.convert('RGB')
  draw = ImageDraw.Draw(result)
  if text is not None:
    font = ImageFont.truetype(font_path, size)
    if bg:
      (x, y) = draw.textsize(text, font)
      draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
    draw.text((2, 0), text, color, font=font)
  result.show()


def mask_rgb(rgb, mask):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  t = Time()
  result = rgb * np.dstack([mask, mask, mask])
  np_info(result, "Mask RGB", t.elapsed())
  return result


class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed

def txt_to_csv(file_path, dest_path):

  if(os.path.exists(file_path)):

    if(os.path.exists(dest_path)):
      file_name = Path(file_path).stem
      txt_file = pd.read_csv(file_path, delimiter="\t")
      return txt_file.to_csv(dest_path + "/" + file_name + ".csv", index=None)

    print("No dest_path exists")
    return False

  print("No file_path exists")
  return False


def filter_manifest():
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
  manifest_file = open(GDC_TCGA_MANIFEST_DIR + "/gdc_manifest_3111.txt", "r")
  filtered_manifest_file = open(GDC_TCGA_MANIFEST_DIR + "/gdc_manifest_general.txt", "w")
  patient_barcodes = brca_file['slide_name'].tolist()
  filtered_manifest_file.write(manifest_file.readline())
  manifest_list = manifest_file.readlines()

  manifest_list = list(filter(lambda manifest_line : 
    list(filter(lambda patient : patient in manifest_line, patient_barcodes)),
  manifest_list))

  list(map(lambda manifest_line : filtered_manifest_file.write(manifest_line), manifest_list))

  filtered_manifest_file.close()

def add_slide_barcode():
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
  slide_name_list = brca_file['slide_name'].to_list()
  barcode_name_list = list(map(lambda name: extract_patient_barcode(name), slide_name_list))
  brca_file['slide_barcode'] = barcode_name_list
  brca_file.to_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", index=False, sep=";")

def rename_slide_dataset():
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  wsi_name_list = os.listdir(GDC_TCGA_SLIDE_DIR)
  list(map(lambda wsi_name : rename_slide(wsi_name), wsi_name_list))

def rename_slide(slide_name):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """

  if(is_slide_name(slide_name)):

    brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    slide_row = brca_file.loc[brca_file['slide_name'] == os.path.splitext(slide_name)[0]]
    slide_id = slide_row.iloc[0]['slide_id']

    src_name = GDC_TCGA_SLIDE_DIR + "/" + slide_name
    dest_name = GDC_TCGA_SLIDE_DIR + "/" + slide_id + ".svs"

    os.rename(src_name, dest_name)
    print(src_name + " TO " + dest_name)

def extract_patient_barcode(file_name):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  name_split = file_name.split("-")
  slide_barcode = name_split[0] + "-" + name_split[1] + "-" + name_split[2]
  return slide_barcode

def move_manifests_to_slides():
  """
  Move the WSI present inside the Manifest downloads directory (ONLY SVS files) to the Slides directory.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  slide_file_list = get_svs_files_from_dir(MANIFEST_DIR)
  list(map(lambda src_path: move_single_slide(src_path), slide_file_list))
    
def move_single_slide(src_path):
  slide_name = os.path.basename(src_path)
  dest_path = os.path.join(GDC_TCGA_SLIDE_DIR, slide_name)

  if(os.path.isfile(dest_path) == False):
    shutil.move(src_path, dest_path)
    print("Moved slide: " + slide_name)

  else:
    print("Not moved existing slide: " + slide_name)

def is_svs_file(file_name):
  """
  Move the WSI present inside the Manifest downloads directory (ONLY SVS files) to the Slides directory.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  file_split = os.path.splitext(file_name)
  extension = file_split[1]
  
  if(extension == ".svs"):
    return True
  
  return False


def get_svs_files_from_dir(dir_name):
  """
  Get all SVS files from the specified directory and returns a list with them.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  svs_list = []
  for root, dirs, files in os.walk(dir_name):
    svs_files = list(map(lambda file_name: os.path.join(root, file_name), files))
    svs_filtered_list = list(filter(lambda file_path: is_svs_file(file_path), svs_files))
    svs_list = svs_list + svs_filtered_list
  return svs_list

def get_slide_name(slide_id=None, slide_barcode=None):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
  if(slide_id != None):
    slide_row = brca_file.loc[brca_file['slide_id'] == slide_id]

  elif(slide_barcode != None):
    slide_row = brca_file.loc[brca_file['slide_barcode'] == slide_barcode]

  slide_name = slide_row.iloc[0]['slide_name']
  return slide_name

def is_slide_in_manifest(manifest_id, slide_id=None, slide_barcode=None, slide_name=None):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  manifest_file_name = "gdc_manifest_100_" + str(manifest_id) + DOT + TXT
  manifest_path = os.path.join(GDC_TCGA_MANIFEST_DIR, manifest_file_name)
  manifest_file = open(manifest_path, "r")
  header = manifest_file.readline()
  manifest_row_list = manifest_file.readlines()

  if(slide_id != None):
    slide_name = get_slide_name(slide_id)

  elif(slide_barcode != None):
    slide_name = get_slide_name(slide_barcode)

  for row in manifest_row_list:
    if slide_name in row:
      return True
  return False

def is_slide_id(file_name):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
  slide_id_list = brca_file['slide_id'].tolist()
  
  return os.path.splitext(file_name)[0] in slide_id_list

def is_slide_name(file_name):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
  slide_name_list = brca_file['slide_name'].tolist()
  
  return os.path.splitext(file_name)[0] in slide_name_list

def is_slide_barcode(file_name):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
  slide_barcode_list = brca_file['slide_barcode'].tolist()
  
  return os.path.splitext(file_name)[0] in slide_barcode_list

def get_status_slides_downloads(manifest_id, file_dir):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
  slide_name_list = brca_file['slide_name'].to_list()
  success_download_list = os.listdir(file_dir)
  success_download_list = list(map(lambda file_name: get_slide_id_from_file_name(file_name), success_download_list))
  success_download_list = list(map(lambda slide_id: get_slide_name(slide_id), success_download_list))

  manifest_slides_list = list(filter(lambda name: is_slide_in_manifest(manifest_id, slide_name=name), slide_name_list))

  failed_donwload_list = list(filter(lambda slide_name: slide_name not in success_download_list, manifest_slides_list))
  success_download_list = list(filter(lambda slide_name: slide_name in success_download_list, manifest_slides_list))

  print("Failed Downloads Slide Names")
  list(map(lambda name: print(name), failed_donwload_list))
  print("N°: " + str(len(failed_donwload_list)) + "\n")

  print("Successful Downloads Slide Names")
  list(map(lambda name: print(name), success_download_list))
  print("N°: " + str(len(success_download_list)))

  return failed_donwload_list

def get_not_in_manifest_slides(manifest_id, file_dir):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
  slide_name_list = brca_file['slide_name'].to_list()
  file_name_list = os.listdir(file_dir)
  file_name_list = list(map(lambda file_name: get_slide_id_from_file_name(file_name), file_name_list))
  file_name_list = list(map(lambda slide_id: get_slide_name(slide_id), file_name_list))

  manifest_slides_list = list(filter(lambda name: is_slide_in_manifest(manifest_id, slide_name=name), slide_name_list))
  not_belonging_slides = list(filter(lambda name: name not in manifest_slides_list, file_name_list))

  print("Not Belonging Slides to Manifest N° " + str(manifest_id))
  list(map(lambda name: print(name), not_belonging_slides))
  print("N°: " + str(len(not_belonging_slides)) + "\n")

  return not_belonging_slides

def get_slide_id_from_file_name(file_name):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  file_split = os.path.splitext(file_name)

  if file_split[1] == ".svs":
    slide_id = file_split[0]
  
  elif file_split[1] == ".jpg":
    image_name = file_split[0]
    image_split = image_name.split("-")
    slide_id = image_split[0] + "-" + image_split[1]

  return slide_id

def check_slide_downloads(file_path):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
  status_list = brca_file['status'].tolist()
  print(status_list)
  successful_list = os.listdir(file_path)
  successful_list = list(map(lambda file_name: os.path.splitext(file_name)[0], successful_list))

def show_list(list):
  list(map(lambda name: print(name), list))
  

if __name__ == "__main__":

  #manifest_csv = txt_to_csv(GDC_TCGA_MANIFEST_DIR + "/gdc_manifest_3111.txt", GDC_TCGA_MANIFEST_DIR)
  #csv_to_txt(GDC_TCGA_MANIFEST_DIR + "/gdc_manifest_3111.csv", GDC_TCGA_MANIFEST_DIR)
  #filter_manifest()
  #rename_dataset()
  #rename_wsi()
  #rename_wsi_dataset()
  #rename_wsi_dataset()
  #set_slide_id_to_barcode()
  #filter_manifest()
  move_manifests_to_slides()
  rename_slide_dataset()
  #get_status_slides_downloads(10, IMAGE_DIR)
  #get_status_slides_downloads(9, IMAGE_DIR)
  #ver = is_slide_id("TCGA-0019")
  #print(str(ver))
  #svs_files = get_svs_files_from_dir(MANIFEST_DIR)
  #print(svs_files)


