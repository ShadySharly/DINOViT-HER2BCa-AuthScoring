import sys, os
from metadata import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

sys.path.append(os.path.join(sys.path[0], PRETRAIN))
sys.path.append(os.path.join(sys.path[0], PROCESSING))

from tiles import *
from filter import *
from slide import *
from util import *
from part import *
from main_dino import *

def main():
    '''
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()   
    train_dino(args)
    '''
    #pt_file = os.path.join(CKPT_DIR, '2024-05-18T01:34:46', EMBED, 'student_epoch_0.pt')
    pt_file = os.path.join(CKPT_DIR, '2024-05-18T01:34:46', 'checkpoint.pth')
    tensor = torch.load(pt_file)
    print(tensor)

    # Suponiendo que tensor es el tensor que proporcionaste
    # Obtener la forma del tensor (número de embeddings, dimensión de embedding)
    shape = tensor.shape
    print("Forma del tensor:", shape)

    # Obtener el tipo de datos del tensor
    dtype = tensor.dtype
    print("Tipo de datos del tensor:", dtype)

    # Verificar si el tensor está en la GPU
    device = tensor.device
    print("Dispositivo del tensor:", device)

    # Convertir el tensor a un array numpy para facilitar su manipulación y visualización
    tensor_array = tensor.cpu().detach().numpy()
    print("Tensor convertido a array numpy:")
    print(tensor_array)

    # Explorar más detalles sobre el tensor, como mínimo, máximo, media, etc.
    print("Mínimo valor del tensor:", torch.min(tensor))
    print("Máximo valor del tensor:", torch.max(tensor))
    print("Media de los valores del tensor:", torch.mean(tensor))
    #multiprocess_filtered_images_to_tiles(image_num_list=[87], save_summary=True, save_data=True, save_top_tiles=False)
    

if __name__ == "__main__":
    main()
