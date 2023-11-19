import sys, os
from metadata import *
sys.path.append(os.path.join(sys.path[0], PROCESSING))

from tiles import *
from filter import *
from slide import *
from util import *

def main():
    print(sys.path[0])
    #multiprocess_filtered_images_to_tiles()

if __name__ == "__main__":
    main()
