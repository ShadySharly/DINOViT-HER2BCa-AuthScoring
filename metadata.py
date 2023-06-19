import os

# ROOT_DIR NAMES
DATA = "data"
FONTS = "fonts"
MANIFEST = "manifest"
TRAINING = "training"
WSI = "wsi"

# SUB_DIR NAMES
GDC_TCGA = "GDC_TCGA"
UCH_CPDAI = "UCH_CPDAI"
IMAGE = "image"
FILTER = "filter"
TILE = "tile"
SLIDE = "slide"
THUMBNAIL = "THUMBNAIL"
IMAGE_MULTI = "image_multi"

# EXTENTIONS
SVS = "svs"
JPG = "jpg"
PNG = "png"
TXT = "txt"
CSV = "csv"

# PARAMETERS AND CONSTANTS
IMAGE_EXT = JPG
SLIDE_PREFIX = "TCGA-"
SCALE_FACTOR = 20
THUMBNAIL_SIZE = 300
NUM_SLIDES = 1041
CROP_RATIO = 10
WHITENESS_TRESHOLD = 240

# PATHS
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
BASE_DIR = os.path.join(ROOT_DIR, DATA)

SLIDE_DIR = os.path.join(BASE_DIR, GDC_TCGA, SLIDE)
IMAGE_GDC_TCGA_DIR = os.path.join(BASE_DIR, GDC_TCGA, IMAGE)
IMAGE_UCH_CPDAI_DIR = os.path.join(BASE_DIR, UCH_CPDAI, IMAGE)
IMAGE_MULTI_GDC_TCGA_DIR = os.path.join(BASE_DIR, GDC_TCGA, IMAGE_MULTI)
THUMBNAIL_GDC_TCGA_DIR = os.path.join(BASE_DIR, GDC_TCGA, THUMBNAIL)

FILTER_SUFFIX = "filter-"  # Example: "filter-"
FILTER_RESULT_TEXT = "filtered"
FILTER_PAGINATION_SIZE = 50
FILTER_PAGINATE = True
FILTER_HTML_DIR = BASE_DIR
FILTER_DIR = os.path.join(BASE_DIR, GDC_TCGA, "filter_" + JPG)
FILTER_THUMBNAIL_DIR = os.path.join(BASE_DIR, "filter_thumbnail_" + JPG)

TILE_SUMMARY_SUFFIX = "tile_summary"
TILE_SUMMARY_PAGINATION_SIZE = 50
TILE_SUMMARY_PAGINATE = True
TILE_SUMMARY_HTML_DIR = BASE_DIR
TILE_SUMMARY_DIR = os.path.join(BASE_DIR, GDC_TCGA, "tile_summary_" + JPG)
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, GDC_TCGA, "tile_summary_on_original_" + JPG)
TILE_SUMMARY_THUMBNAIL_DIR = os.path.join(BASE_DIR, GDC_TCGA, "tile_summary_thumbnail_" + JPG)
TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR, GDC_TCGA, "tile_summary_on_original_thumbnail_" + JPG)

TILE_SUFFIX = "tile"
TILE_DATA_SUFFIX = "tile_data"
TOP_TILES_SUFFIX = "top_tile_summary"
TILE_DIR = os.path.join(BASE_DIR, "tiles_" + JPG)
TILE_DATA_DIR = os.path.join(BASE_DIR, GDC_TCGA, TILE_DATA_SUFFIX)
TOP_TILES_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_" + JPG)
TOP_TILES_THUMBNAIL_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_thumbnail_" + JPG)
TOP_TILES_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_on_original_" + JPG)
TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR,
                                                   TOP_TILES_SUFFIX + "_on_original_thumbnail_" + JPG)
STATS_DIR = os.path.join(BASE_DIR, "svs_stats")
FONT_PATH = os.path.join(ROOT_DIR, FONTS, "Arial Bold.ttf")
SUMMARY_TITLE_FONT_PATH = os.path.join(ROOT_DIR, FONTS, "Courier New Bold.ttf")
