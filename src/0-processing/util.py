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
import csv
import glob
import random
import shutil
import pathlib
import datetime
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from metadata import *

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False


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
        print(
            "%-20s | Time: %-14s  Type: %-7s Shape: %s"
            % (name, str(elapsed), np_arr.dtype, np_arr.shape)
        )
    else:
        # np_arr = np.asarray(np_arr)
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print(
            "%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s"
            % (
                name,
                str(elapsed),
                min,
                max,
                mean,
                is_binary,
                np_arr.dtype,
                np_arr.shape,
            )
        )


def display_img(
    np_img,
    text=None,
    font_path="/Library/Fonts/Arial Bold.ttf",
    size=48,
    color=(255, 0, 0),
    background=(255, 255, 255),
    border=(0, 0, 0),
    bg=False,
):
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
    if result.mode == "L":
        result = result.convert("RGB")
    draw = ImageDraw.Draw(result)
    if text is not None:
        font = ImageFont.truetype(font_path, size)
        if bg:
            (x, y) = draw.textsize(text, font)  # type: ignore
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)  # type: ignore
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


def filter_manifest():
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

    Args:
      rgb: RGB image as a NumPy array.
      mask: An image mask to determine which pixels in the original image should be displayed.

    Returns:
      NumPy array representing an RGB image with mask applied.
    """
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    manifest_file = open(MANIFEST_DIR + "/gdc_manifest_3111.txt", "r")
    filtered_manifest_file = open(MANIFEST_DIR + "/gdc_manifest_general.txt", "w")
    patient_barcodes = brca_file["slide_name"].tolist()
    filtered_manifest_file.write(manifest_file.readline())
    manifest_list = manifest_file.readlines()

    manifest_list = list(
        filter(
            lambda manifest_line: list(
                filter(lambda patient: patient in manifest_line, patient_barcodes)
            ),
            manifest_list,
        )
    )

    list(
        map(
            lambda manifest_line: filtered_manifest_file.write(manifest_line),
            manifest_list,
        )
    )

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
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    slide_name_list = brca_file["slide_name"].to_list()
    barcode_name_list = list(
        map(lambda name: extract_patient_barcode(name), slide_name_list)
    )
    brca_file["slide_barcode"] = barcode_name_list
    brca_file.to_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", index=False, sep=";")


def rename_slide_dataset():
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

    Args:
      rgb: RGB image as a NumPy array.
      mask: An image mask to determine which pixels in the original image should be displayed.

    Returns:
      NumPy array representing an RGB image with mask applied.
    """
    wsi_name_list = os.listdir(SLIDE_DIR)
    list(map(lambda wsi_name: rename_slide(wsi_name), wsi_name_list))


def rename_slide(slide_name):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

    Args:
      rgb: RGB image as a NumPy array.
      mask: An image mask to determine which pixels in the original image should be displayed.

    Returns:
      NumPy array representing an RGB image with mask applied.
    """

    if is_slide_name(slide_name):
        brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
        slide_row = brca_file.loc[
            brca_file["slide_name"] == os.path.splitext(slide_name)[0]
        ]
        slide_id = slide_row.iloc[0]["slide_id"]

        src_name = SLIDE_DIR + "/" + slide_name
        dest_name = SLIDE_DIR + "/" + slide_id + ".svs"

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
    dest_path = os.path.join(SLIDE_DIR, slide_name)

    if os.path.isfile(dest_path) == False:
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

    if extension == ".svs":
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
        svs_filtered_list = list(
            filter(lambda file_path: is_svs_file(file_path), svs_files)
        )
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
    slide_row = ""
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    if slide_id != None:
        slide_row = brca_file.loc[brca_file["slide_id"] == slide_id]

    elif slide_barcode != None:
        slide_row = brca_file.loc[brca_file["slide_barcode"] == slide_barcode]

    slide_name = slide_row.iloc[0]["slide_name"]  # type: ignore
    return slide_name


def get_slide_barcode(slide_id=None, slide_name=None):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

    Args:
      rgb: RGB image as a NumPy array.
      mask: An image mask to determine which pixels in the original image should be displayed.

    Returns:
      NumPy array representing an RGB image with mask applied.
    """
    slide_row = ""
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    if slide_id != None:
        slide_row = brca_file.loc[brca_file["slide_id"] == slide_id]

    elif slide_name != None:
        slide_row = brca_file.loc[brca_file["slide_name"] == slide_name]

    slide_barcode = slide_row.iloc[0]["slide_barcode"]  # type: ignore
    return slide_barcode


def is_slide_in_manifest(
    manifest_id, slide_id=None, slide_barcode=None, slide_name=None
):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

    Args:
      rgb: RGB image as a NumPy array.
      mask: An image mask to determine which pixels in the original image should be displayed.

    Returns:
      NumPy array representing an RGB image with mask applied.
    """
    manifest_file_name = "gdc_manifest_100_" + str(manifest_id) + DOT + TXT
    manifest_path = os.path.join(MANIFEST_DIR, manifest_file_name)
    manifest_file = open(manifest_path, "r")
    header = manifest_file.readline()
    manifest_row_list = manifest_file.readlines()

    if slide_id != None:
        slide_name = get_slide_name(slide_id)

    elif slide_barcode != None:
        slide_name = get_slide_name(slide_barcode)

    for row in manifest_row_list:
        if slide_name != None and slide_name in row:
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
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    slide_id_list = brca_file["slide_id"].tolist()

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
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    slide_name_list = brca_file["slide_name"].tolist()

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
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    slide_barcode_list = brca_file["slide_barcode"].tolist()

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
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    slide_name_list = brca_file["slide_name"].to_list()
    success_download_list = os.listdir(file_dir)
    success_download_list = list(
        map(
            lambda file_name: get_slide_id_from_file_name(file_name),
            success_download_list,
        )
    )
    success_download_list = list(
        map(lambda slide_id: get_slide_name(slide_id), success_download_list)
    )

    manifest_slides_list = list(
        filter(
            lambda name: is_slide_in_manifest(manifest_id, slide_name=name),
            slide_name_list,
        )
    )

    failed_donwload_list = list(
        filter(
            lambda slide_name: slide_name not in success_download_list,
            manifest_slides_list,
        )
    )
    success_download_list = list(
        filter(
            lambda slide_name: slide_name in success_download_list, manifest_slides_list
        )
    )

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
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    slide_name_list = brca_file["slide_name"].to_list()
    file_name_list = os.listdir(file_dir)
    file_name_list = list(
        map(lambda file_name: get_slide_id_from_file_name(file_name), file_name_list)
    )
    file_name_list = list(
        map(lambda slide_id: get_slide_name(slide_id), file_name_list)
    )

    manifest_slides_list = list(
        filter(
            lambda name: is_slide_in_manifest(manifest_id, slide_name=name),
            slide_name_list,
        )
    )
    not_belonging_slides = list(
        filter(lambda name: name not in manifest_slides_list, file_name_list)
    )

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
    slide_id = ""

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
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    status_list = brca_file["status"].tolist()
    print(status_list)
    successful_list = os.listdir(file_path)
    successful_list = list(
        map(lambda file_name: os.path.splitext(file_name)[0], successful_list)
    )


def add_image_data_row(
    slide_id=None, slide_barcode=None, slide_name=None, slide_id_origin=None
):
    brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
    brca_file.loc[len(brca_file)] = [  # type: ignore
        slide_id,
        slide_barcode,
        slide_name,
        slide_id_origin,
    ]
    brca_file.to_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", index=False, sep=";")


def get_dir_size(dir):
    """
    FunctionDescription

    Args:
    arg: Argument description

    returns:
    Returns description
    """
    file_list = os.listdir(dir)
    return len(file_list)


def is_tile_dir(slide_number):
    padded_sl_num = str(slide_number).zfill(4)
    tile_dir_path = os.path.join(TILE_IMAGE_DIR, padded_sl_num)
    return os.path.isdir(tile_dir_path)


def is_filter_dir(slide_number):
    padded_sl_num = str(slide_number).zfill(4)
    tile_dir_path = os.path.join(FILTER_IMAGE_DIR, padded_sl_num)
    return os.path.isdir(tile_dir_path)


def merge_directories(src_dirs_path=None, dest_dir_path=None):
    if src_dirs_path != None and dest_dir_path != None:
        dirs_path = os.listdir(src_dirs_path)
        list(map(lambda src_path: move_single_dir(src_path, dest_dir_path), dirs_path))

    return False


def move_single_dir(src_dir, dest_dir):
    files = os.listdir(src_dir)


def list_unknown_tiles_dir():
    filtered_image_path_list = list(
        map(
            lambda src_file: get_slide_id_from_file_name(src_file).split("-")[1],
            os.listdir(FILTER_IMAGE_DIR),
        )
    )
    tiled_image_dir_list = list(
        map(
            lambda src_file: os.path.basename(src_file),
            os.listdir(TILE_IMAGE_DIR),
        )
    )
    return list(
        filter(
            lambda silde_id: silde_id not in tiled_image_dir_list,
            filtered_image_path_list,
        )
    )


def create_tiles_overall():
    """
    Creates a CSV file with a tile overall summary, including all patients and their correspondent tiles generated, and more detailed tile information.
    With X e |N : {1 -> {X}} == {Y}
        patient_id ( TCGA-XXXX )
        tile_id    ( TCGA-XXXX-{Y} )
        tile_name  ( TCGA-XXXX-tile-r{Y}-c{Y}-x{Y}-y{Y}-w{Y}-h{Y} )
        row        ( {Y} )
        col        ( {Y} )
        x_axis     ( {Y} )
        y_axis     ( {Y} )
        width      ( {Y} )
        height     ( {Y} )

    Args:
    arg: Argument description

    returns:
    Returns description
    """
    # Nombre del archivo CSV
    file_path = os.path.join(TILE_DIR, TILE_OVERALL_CSV)

    # Datos iniciales
    header = [
        [
            "patient_id",
            "tile_id",
            "tile_name",
            "row",
            "col",
            "x_axis",
            "y_axis",
            "width",
            "height",
        ],
    ]

    # Escribir en el archivo CSV
    with open(file_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(header)
        list(
            map(
                lambda tile_row: csv_writer.writerows(tile_row),
                create_tile_overall_rows(),
            )
        )

    print(f"File {file_path} has been created with initial data")


def create_tile_overall_rows():
    overall_rows_list = []
    tile_id_number = autoincremental_generator()
    for root, dirs, files in os.walk(TILE_IMAGE_DIR):
        for dir in dirs:
            patient_tile_paths = os.listdir(os.path.join(root, dir))
            patient_tile_rows = list(
                map(
                    lambda tile_path: create_tile_single_row(
                        tile_path, next(tile_id_number)
                    ),
                    patient_tile_paths,
                )
            )
            overall_rows_list.append(patient_tile_rows)
            tile_id_number.send(True)
    return overall_rows_list


def autoincremental_generator():
    contador = 1
    while True:
        reset = yield contador
        if reset:
            contador = 0
        else:
            contador += 1


def create_tile_single_row(tile_path, tile_id_number):
    tile_name, extension = os.path.splitext(os.path.basename(tile_path))
    tile_row = tile_name.split("-")
    patient_id = tile_row[0] + "-" + tile_row[1]
    tile_id = patient_id + "-" + str(tile_id_number)
    del tile_row[0:3]
    tile_row.insert(0, patient_id)
    tile_row.insert(1, tile_id)
    tile_row.insert(2, tile_name)
    tile_row[3] = tile_row[3][1:]
    tile_row[4] = tile_row[4][1:]
    tile_row[5] = tile_row[5][1:]
    tile_row[6] = tile_row[6][1:]
    tile_row[7] = tile_row[7][1:]
    tile_row[8] = tile_row[8][1:]

    return tile_row


def create_tiles_summary():
    patient_dictionary = {}
    file_path = os.path.join(TILE_DIR, TILE_SUMMARY_CSV)

    # Recorrer el directorio y sus subdirectorios
    for root, dirs, files in os.walk(TILE_IMAGE_DIR):
        # Calcular la cantidad de archivos en la carpeta actual
        patient_number = os.path.basename(root)
        if patient_number != "tiles_jpg":
            tile_count = len(files)
            # Almacenar la cantidad de archivos en el diccionario
            patient_dictionary[patient_number] = tile_count

    # Crear un archivo CSV para almacenar los resultados
    with open(file_path, "w", newline="") as csvfile:
        # Crear un objeto escritor CSV
        csv_writer = csv.writer(csvfile)

        # Escribir la cabecera del CSV
        csv_writer.writerow(["patient_id", "tile_count"])

        # Escribir los datos en el CSV
        for patient_number, tile_count in patient_dictionary.items():
            patient_id = SLIDE_PREFIX + patient_number
            csv_writer.writerow([patient_id, tile_count])

        # Calcular el total de archivos
        total_count = sum(patient_dictionary.values())

        # Escribir la fila de total
        csv_writer.writerow(["total", total_count])

    print(
        f'Se ha creado el archivo CSV "resumen_archivos.csv" con el resumen de archivos.'
    )


def rename_gdc_tile_dataset():
    # Ruta del archivo CSV
    csv_path = os.path.join(TILE_DIR, TILE_OVERALL_CSV)

    # Ruta del directorio donde se encuentran los archivos
    dir_path = "ruta/del/directorio"

    # Leer el CSV con pandas
    df = pd.read_csv(csv_path)

    # Iterar sobre las filas del DataFrame
    for index, row in df.iterrows():
        # Obtener el nombre actual y el nuevo nombre
        nombre_actual = os.path.join(dir_path, row["name"])
        nuevo_nombre = os.path.join(dir_path, str(row["id"]))

        # Renombrar el archivo si existe
        if os.path.exists(nombre_actual):
            os.rename(nombre_actual, nuevo_nombre)
            print(f"Archivo renombrado: {nombre_actual} -> {nuevo_nombre}")
        else:
            print(f"Archivo no encontrado: {nombre_actual}")

    if is_slide_name(slide_name):
        brca_file = pd.read_csv(GDC_TCGA_DIR + "/TCGA-BRCA_Paper.csv", delimiter=";")
        slide_row = brca_file.loc[
            brca_file["slide_name"] == os.path.splitext(slide_name)[0]
        ]
        slide_id = slide_row.iloc[0]["slide_id"]

        src_name = SLIDE_DIR + "/" + slide_name
        dest_name = SLIDE_DIR + "/" + slide_id + ".svs"

        os.rename(src_name, dest_name)
        print(src_name + " TO " + dest_name)