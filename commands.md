### Used commands for interaction with DIINF servers

### Conection to Open VPN Usach

sudo openvpn --config profile-5153.ovpn

### Connect to server

ssh cperezs@tumi.diinf.usach.cl

### RSYNC COMMANDS

# Check files to Sync

rsync -av --dry-run <origin> <remote>

# Sync files

rsync -av <origin> <remote>

# Check Excluding (Exclude more files or dirs adding more '--exclude' flags)

rsync -av --exclude '<name>' --dry-run <origin> <remote>
rsync -av --exclude 'data' --dry-run ./DINOViT-HER2BCa-AuthScoring cperezs@tumi.diinf.usach.cl:
rsync -av --exclude 'data' --dry-run cperezs@tumi.diinf.usach.cl:~/DINOViT-HER2BCa-AuthScoring .

# Sync Excluding

rsync -av --exclude '<name>' <origin> <remote>
rsync -av --exclude 'data' ./DINOViT-HER2BCa-AuthScoring cperezs@tumi.diinf.usach.cl:
rsync -av --exclude 'data' cperezs@tumi.diinf.usach.cl:~/DINOViT-HER2BCa-AuthScoring .

# Sync ORIGIN to SERVER

rsync -av --dry-run ./DINOViT-HER2BCa-AuthScoring cperezs@tumi.diinf.usach.cl:
rsync -av ./DINOViT-HER2BCa-AuthScoring cperezs@tumi.diinf.usach.cl:

# Sync SERVER to ORIGIN

rsync -av --dry-run cperezs@tumi.diinf.usach.cl:~/DINOViT-HER2BCa-AuthScoring .
rsync -av --dry-run cperezs@tumi.diinf.usach.cl: ./DINOViT-HER2BCa-AuthScoring

rsync -av cperezs@tumi.diinf.usach.cl:~/DINOViT-HER2BCa-AuthScoring .
rsync -av cperezs@tumi.diinf.usach.cl: ./DINOViT-HER2BCa-AuthScoring

# SERVER path

cperezs@tumi.diinf.usach.cl:

### DIR COMMANDS

# NUMBER OF FILES

ls | wc -l

# SIZE OF DIR

du -h

### SCREEN COMMANDS

Comandos Descripción
CTRL+a c Crea una nueva ventana
CTRL+a ” Lista de todas las ventanas creadas.
CTRL+a a Con este comando puedes eliminar un CTRL+a. Es útil si te equivocas.
CTRL+a d Deja la sesión en ejecución.

screen -S session1 Crear una nueva sesion
screen -ls Listar las sesiones
screen -r numero_proceso Retomar sesion
screen -X -S <process_number> quit Eliminar sesion y salir

### RUN DISTRIBUITED SCRIPT

python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /path/to/TCGA_PRETRAINING_DIR/patch_256_pretraining/ --output_dir /path/to/TCGA_PRETRAINING_DIR/ckpts/pretrain/ --epochs 100

python -m torch.distributed.launch --nproc_per_node=8 main_dino4k.py --arch vit_xs --data_path /path/to TCGA_PRETRAINING_DIR/region_4k_pretraining/ --output_dir /path/to/TCGA_PRETRAINING_DIR/ckpts/pretrain/ --epochs 100

export OMP_NUM_THREADS=4 # Try with 4 threads per process
python -m torch.distributed.run --nproc_per_node=3 main_dino.py --epoch 100

export OMP_NUM_THREADS=4 # Try with 4 threads per process
python -m torch.distributed.run --nproc_per_node=3 main_dino4k.py --epoch 100

### RUN WITH DIFFERENT VALUES

Start with a lower number of threads and gradually increase to find the optimal setting for your hardware and workload. Common values to try are 1, 2, 4, and 8:

- Since you have powerful GPUs, setting OMP_NUM_THREADS to a moderate number (like 4) should be a good starting point. This helps prevent CPU thread contention.

export OMP_NUM_THREADS=1 # Try with 1 thread per process
python -m torch.distributed.run --nproc_per_node=8 main_dino.py

export OMP_NUM_THREADS=2 # Try with 2 threads per process
python -m torch.distributed.run --nproc_per_node=8 main_dino.py

export OMP_NUM_THREADS=4 # Try with 4 threads per process
python -m torch.distributed.run --nproc_per_node=8 main_dino.py

export OMP_NUM_THREADS=8 # Try with 8 threads per process
python -m torch.distributed.run --nproc_per_node=8 main_dino.py

### MONITORING AND DEBUGGING

Monitor GPU Utilization:
Use nvidia-smi to monitor the GPU utilization and memory usage during training:

watch -n 1 nvidia-smi

### COMMENTS FORMAT

"""
FunctionDescription

Args:
arg: Argument description

returns:
Returns description
"""
