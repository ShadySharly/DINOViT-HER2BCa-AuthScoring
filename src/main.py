import sys, os
from metadata import *

sys.path.append(os.path.join(sys.path[0], PROCESSING))
sys.path.append(os.path.join(sys.path[0], TRAINING))

from tiles import *
from filter import *
from slide import *
from util import *
from part import *
from main_dino import *


def main():
    #create_tiles_overall()
    #create_tiles_summary()
    
    #parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    #args = parser.parse_args()
    #print(args)
    #Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #train_dino(args)
    multiprocess_filtered_images_to_tiles(image_num_list=[87], save_summary=True, summary_dir="0087_score09", save_top_tiles=False, score_treshold=0.9)
    

if __name__ == "__main__":
    main()
