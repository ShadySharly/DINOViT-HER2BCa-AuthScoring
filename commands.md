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
rsync -av --exclude --dry-run '<name>' <origin> <remote>
rsync -av --exclude --dry-run 'data' ./DINOViT-HER2BCa-AuthScoring cperezs@tumi.diinf.usach.cl:

# Sync Excluding
rsync -av --exclude '<name>' <origin> <remote>
rsync -av --exclude 'data' ./DINOViT-HER2BCa-AuthScoring cperezs@tumi.diinf.usach.cl:

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
Comandos	                    Descripción
CTRL+a c	                    Crea una nueva ventana
CTRL+a ”	                    Lista de todas las ventanas creadas.
CTRL+a a	                    Con este comando puedes eliminar un CTRL+a. Es útil si te equivocas.
CTRL+a d                          Deja la sesión en ejecución. 

screen -S session1              Crear una nueva sesion
screen -ls                      Listar las sesiones
screen -r numero_proceso        Retomar sesion
screen -X -S <process_number> quit  Eliminar sesion y salir

### COMMENTS FORMAT
"""
FunctionDescription

Args:
arg: Argument description

returns:
Returns description
"""
