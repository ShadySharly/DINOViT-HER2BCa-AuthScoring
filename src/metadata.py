import os
from PIL import Image

## EXTENSIONS ____________________________________
SVS = "svs"
JPG = "jpg"
PNG = "png"
TXT = "txt"
CSV = "csv"
DOT = "."
UND = "_"
IMAGE_EXT = JPG

### ROOT VARIABLES AND PATHS #############################################################################################
DATA = "data"
FONT = "font"
MANIFEST = "manifest"
SRC = "src"

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, DATA)
FONT_DIR = os.path.join(ROOT_DIR, FONT)
MANIFEST_DIR = os.path.join(ROOT_DIR, MANIFEST)
SRC_DIR = os.path.join(ROOT_DIR, SRC)

### SUBDIR VARIABLES AND PATHS ############################################################################################
GDC_TCGA = "GDC_TCGA"
UCH_CPDAI = "UCH_CPDAI"
CKPT = "ckpt"
FILTER = "filter"
IMAGE = "image"
IMAGE_MULTI = "image_multi"
SLIDE = "slide"
THUMBNAIL = "thumbnail"
TILE = "tile"
PROCESSING = "processing"
TRAINING = "training"
CKPT_DIR = os.path.join(DATA_DIR, CKPT)
GDC_TCGA_DIR = os.path.join(DATA_DIR, GDC_TCGA)
UCH_CPDAI_DIR = os.path.join(DATA_DIR, UCH_CPDAI)

### PARTITIONING PARAMETERS ###############################################################################################
PARTITION = "partition"
GDC_TCGA_PARTITION_DIR = os.path.join(GDC_TCGA_DIR, PARTITION)
UCH_CPDAI_PARTITION_DIR = os.path.join(UCH_CPDAI_DIR, PARTITION)
PARTITION_CSV = os.path.join(PARTITION, DOT, CSV)
PARTITION_NUM = 10
TRAIN_RATIO = 0.7
EVAL_RATIO = 1 - TRAIN_RATIO

### TRAINING PARAMETERS ###################################################################################################


### EVALUATION PARAMETERS #################################################################################################


### PRE-PROCESSING PARAMETERS #############################################################################################
SLIDE_PREFIX = "TCGA-"
SCALE_FACTOR = 20
THUMBNAIL_SIZE = 300
CROP_RATIO = 10
WHITENESS_TRESHOLD = 240
IMAGE_BATCH = 400
CPU_COUNT = 20
Image.MAX_IMAGE_PIXELS = None

## TILES AND TOP TILES GENERATION ___________
TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10
SCORE_HIGH_THRESH = 0.725
SCORE_LOW_THRESH = 0.2
MIN_SCORE_THRESH = 0.725
MIN_TISSUE_THRESH = 80
NUM_TOP_TILES = 70
ROW_TILE_SIZE = 256
COL_TILE_SIZE = 256
FILTER_BY_SCORE = True
FILTER_BY_TISSUE_PERCENTAGE = True

DISPLAY_TILE_SUMMARY_LABELS = True
TILE_LABEL_TEXT_SIZE = 10
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = True
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = True

TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

HSV_PURPLE = 270
HSV_PINK = 330

## CKPTS AND IMAGE ________________________________
GDC_TCGA_IMAGE_DIR = os.path.join(GDC_TCGA_DIR, IMAGE)
GDC_TCGA_MULTI_IMAGE_DIR = os.path.join(GDC_TCGA_DIR, IMAGE_MULTI)
GDC_TCGA_THUMBNAIL_DIR = os.path.join(GDC_TCGA_DIR, THUMBNAIL)
UCH_CPDAI_IMAGE_DIR = os.path.join(UCH_CPDAI_DIR, IMAGE)

CKPT_GDCA_TCGA = os.path.join(GDC_TCGA_DIR, CKPT)
CKPT_UCH_CPDAI = os.path.join(UCH_CPDAI_DIR, CKPT)
STATS_DIR = os.path.join(GDC_TCGA_DIR, "svs_stats")

## SLIDE __________________________________________
SLIDE_DIR = os.path.join(GDC_TCGA_DIR, SLIDE)

## FILTER _________________________________________
FILTER_DIR = os.path.join(GDC_TCGA_DIR, FILTER)

FILTER_SUFFIX = "filter-"  # Example: "filter-"
FILTER_RESULT_TEXT = "filtered"
FILTER_PAGINATION_SIZE = 50
FILTER_PAGINATE = True

FILTER_IMAGE_DIR = os.path.join(FILTER_DIR, "filter_" + JPG)
FILTER_THUMBNAIL_DIR = os.path.join(FILTER_DIR, "filter_thumbnail_" + JPG)
FILTER_HTML_DIR = FILTER_DIR

## TILE ___________________________________________
TILE_DIR = os.path.join(GDC_TCGA_DIR, TILE)

TILE_SUFFIX = "tile"
TILE_DATA_SUFFIX = "tile_data"
TILE_SUMMARY_SUFFIX = "tile_summary"
TILE_OVERALL_SUFFIX = "tile_overall"
TILE_SUMMARY_CSV = TILE_SUMMARY_SUFFIX + DOT + CSV
TILE_OVERALL_CSV = TILE_OVERALL_SUFFIX + DOT + CSV
TOP_TILES_SUFFIX = "top_tile_summary"
TILE_SUMMARY_PAGINATION_SIZE = 50
TILE_SUMMARY_PAGINATE = True

TILE_DATA_DIR = os.path.join(TILE_DIR, "tile_data")
TILE_SUMMARY_DIR = os.path.join(TILE_DIR, "tile_summary_" + JPG)
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(TILE_DIR, "tile_summary_on_original_" + JPG)
TILE_SUMMARY_THUMBNAIL_DIR = os.path.join(TILE_DIR, "tile_summary_thumbnail_" + JPG)
TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(
    TILE_DIR, "tile_summary_on_original_thumbnail_" + JPG
)
TILE_IMAGE_DIR = os.path.join(TILE_DIR, "tiles_" + JPG)
TOP_TILES_DIR = os.path.join(TILE_DIR, TOP_TILES_SUFFIX + "_" + JPG)
TOP_TILES_THUMBNAIL_DIR = os.path.join(TILE_DIR, TOP_TILES_SUFFIX + "_thumbnail_" + JPG)
TOP_TILES_ON_ORIGINAL_DIR = os.path.join(
    TILE_DIR, TOP_TILES_SUFFIX + "_on_original_" + JPG
)
TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(
    TILE_DIR, TOP_TILES_SUFFIX + "_on_original_thumbnail_" + JPG
)
TILE_SUMMARY_HTML_DIR = TILE_DIR

## FONTS ________________________________
FONT_PATH = os.path.join(FONT_DIR, "Arial Bold.ttf")
SUMMARY_TITLE_FONT_PATH = os.path.join(FONT_DIR, "Courier New Bold.ttf")
