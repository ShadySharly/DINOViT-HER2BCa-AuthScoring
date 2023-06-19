Used commands for interaction with DIINF servers

Conection to Open VPN Usach
sudo openvpn --config profile-5153.ovpn
a,59.B,82.c,17

Connect to server
ssh cperezs@tumi.diinf.usach.cl

Check files to Sync
< rsync -av --dry-run . cperezs@tumi.diinf.usach.cl:~/home/DIINF/cperezs >
rsync -av --dry-run --exclude=manifest,slide,image ./DINOViT-HER2BCa-AuthScoring cperezs@tumi.diinf.usach.cl:

Sync ORIGIN to SERVER 
< rsync -r origin/ server/ >
rsync -avr ./DINOViT-HER2BCa-AuthScoring cperezs@tumi.diinf.usach.cl:

Sync SERVER to ORIGIN
< rsync -r server/ origin/ >
rsync -av --dry-run . ~/home/DIINF/cperezs/dino/data/GDC_TCGA/image

SERVER path
cperezs@tumi.diinf.usach.cl:

SCREEN
Comandos	                    Descripción
CTRL+a c	                    Crea una nueva ventana
CTRL+a ”	                    Lista de todas las ventanas creadas.
CTRL+a a	                    Con este comando puedes eliminar un CTRL+a. Es útil si te equivocas.
CTRL+a
CTRL+d                          Deja la sesión en ejecución. 

screen -S session1              Crear una nueva sesion
screen -ls                      Listar las sesiones
screen -r numero_proceso        Retomar sesion
screen -X -S <process_number> quit  Eliminar sesion y salir