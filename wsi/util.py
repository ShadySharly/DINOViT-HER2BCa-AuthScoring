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
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False

# GENERAL PATHS
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# GDC_TCGA PATHS
GDC_TCGA_DATA_DIR = os.path.join(DATA_DIR, "GDC_TCGA")
GDC_TCGA_EVAL_DIR = os.path.join(GDC_TCGA_DATA_DIR, "evaluation")
GDC_TCGA_TRAIN_DIR = os.path.join(GDC_TCGA_DATA_DIR, "training")
GDC_TCGA_MANIFEST_DIR = os.path.join(GDC_TCGA_DATA_DIR, "manifest")
GDC_TCGA_WSI_DIR = os.path.join(GDC_TCGA_DATA_DIR, "wsi")

# UCH_CPDAI PATHS
UCH_CPDAI_DATA_DIR = os.path.join(DATA_DIR, "UCH_CPDAI")


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
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Data.csv", delimiter=";")
  manifest_file = open(GDC_TCGA_MANIFEST_DIR + "/gdc_manifest_3111.txt", "r")
  filtered_manifest_file = open(GDC_TCGA_MANIFEST_DIR + "/gdc_manifest_general.txt", "w")
  patient_barcodes = brca_file['patient_barcode'].tolist()
  filtered_manifest_file.write(manifest_file.readline())
  manifest_list = manifest_file.readlines()

  manifest_list = list(filter(lambda manifest_line : 
    list(filter(lambda patient : patient + '-01Z-00-DX1' in manifest_line, patient_barcodes)),
  manifest_list))

  list(map(lambda manifest_line : filtered_manifest_file.write(manifest_line), manifest_list))

  filtered_manifest_file.close()

def rename_wsi_dataset():
  wsi_name_list = os.listdir(GDC_TCGA_WSI_DIR)
  list(map(lambda wsi_name : rename_wsi(wsi_name), wsi_name_list))

def rename_wsi(wsi_name):
  brca_file = pd.read_csv(GDC_TCGA_DATA_DIR + "/TCGA-BRCA_Data.csv", delimiter=";")
  patient_id = extract_patient_barcode(wsi_name)
  patient_row = brca_file.loc[brca_file['patient_barcode'] == patient_id]
  patient_id = patient_row.iloc[0]['patient_id']

  src_name = GDC_TCGA_WSI_DIR + "/" + wsi_name
  dest_name = GDC_TCGA_WSI_DIR + "/" + patient_id + ".svs"

  os.rename(src_name, dest_name)

def extract_patient_barcode(file_name):
  name_split = file_name.split("-")
  patient_barcode = name_split[0] + "-" + name_split[1] + "-" + name_split[2]
  return patient_barcode

def partition_manifest(max_slides):
  manifest_file = open(GDC_TCGA_MANIFEST_DIR + "/gdc_manifest_general.txt", "r")

  manifest_header = manifest_file.readline()
  manifest_patients = manifest_file.readlines()

  while
  



def create_manifest_partition(header, file, max_slides, part_id, start_index, end_index):
  partition_name = "gdc_manifest_" + max_slides + "_" + part_id + ".txt"
  partition_file = open(partition_name, "w")
  partition_file.write(header)



  
if __name__ == "__main__":

  #manifest_csv = txt_to_csv(GDC_TCGA_MANIFEST_DIR + "/gdc_manifest_3111.txt", GDC_TCGA_MANIFEST_DIR)
  #csv_to_txt(GDC_TCGA_MANIFEST_DIR + "/gdc_manifest_3111.csv", GDC_TCGA_MANIFEST_DIR)
  #filter_manifest()
  #rename_dataset()
  #rename_wsi()
  #rename_wsi_dataset()
  My_list = [*range(100, 1057, 100)] + [1057]
  print(My_list)
  partition_manifest(100)

