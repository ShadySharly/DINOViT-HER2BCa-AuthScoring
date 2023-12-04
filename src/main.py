import sys, os
from metadata import *
sys.path.append(os.path.join(sys.path[0], PROCESSING))
sys.path.append(os.path.join(sys.path[0], TRAINING))

from tiles import *
from filter import *
from slide import *
from util import *
from main_dino import *

def main():
    #multiprocess_filtered_images_to_tiles(start_ind=0)
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)

if __name__ == "__main__":
    main()