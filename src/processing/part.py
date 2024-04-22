import os
import csv
import glob
import random
import pathlib
from metadata import *

def createNamesFile():
    namesFile = open("../ChileanDatasets/gsIDs.txt", "w+")
    workingDir = "../ChileanDatasets/Normal/"
    for name in glob.glob(workingDir + "*"):
        nameID = pathlib.Path(name)
        nameID = nameID.stem
        namesFile.write("%s\n" % nameID)
    namesFile.close()


def createWSIList(wsiFileName="../ChileanDatasets/wsiIDs.txt"):
    my_file = open(wsiFileName, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    return data_into_list


def createGSList(wsiFileName="../ChileanDatasets/gsIDs.txt"):
    my_file = open(wsiFileName, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    return data_into_list


def createPartitions():
    # Counts the number of aparitions of every wsi into the evaluation dataset (wsiEval)
    wsiEvalCount = [0] * 30
    partitions = []
    wsiList = createWSIList()
    random.shuffle(wsiList)

    while invalidPartitions(wsiEvalCount):
        wsiEvalCount = [0] * 30
        partitions = []
        wsiTrain = []
        wsiEval = []
        partition = []

        for p in range(10):
            random.seed()
            wsiListRand = wsiList.copy()
            random.shuffle(wsiListRand)
            wsiTrain = wsiListRand[0:21]
            wsiEval = wsiListRand[21:30]
            partition = [wsiTrain, wsiEval]
            partitions.append(partition)
            addWSICounter(wsiList, wsiEvalCount, wsiEval)

    return partitions


def invalidPartitions(wsiEvalCount):
    for wsiCount in wsiEvalCount:
        if wsiCount == 0:
            return True

    return False


def addWSICounter(wsiList, wsiEvalCount, wsiEval):
    wsiIndex = 0
    for wsi in wsiList:
        if wsi in wsiEval:
            wsiEvalCount[wsiIndex] += 1
        wsiIndex += 1


def getGSFromWSIId(wsiId, gsIDslist):
    gsBelongingToWSI = list(filter(lambda gs: wsiId in gs, gsIDslist))
    print("WSI ID: %s\n" % wsiId)
    print("\n".join(gsBelongingToWSI))  # type: ignore
    return gsBelongingToWSI


def createPartitionsCSV(partitions):
    gsIDsList = createGSList()
    header = ["partition", "set", "patient", "id"]

    with open("../ChileanDatasets/partitions.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        partID = 1

        for partition in partitions:
            setID = "training"
            for set in partition:
                for wsiID in set:
                    gs_of_wsi = getGSFromWSIId(wsiID, gsIDsList)
                    list(
                        map(
                            lambda gsID: writer.writerow([partID, setID, wsiID, gsID]),
                            gs_of_wsi,
                        )
                    )

                setID = "evaluation"
            partID += 1


def create_gdc_partitions():
    

    
    for partition_id in range(1, PARTITION_NUM + 1):
        patient_dirs = os.listdir(TILE_IMAGE_DIR)
        csv_data = []

        # Declare name for the CSV file for the current partitio
        file_name = PARTITION + UND + str(partition_id) + DOT + CSV
        file_path = os.path.join(TILE_DIR, PARTITION, file_name)

        # Barajar los pacientes aleatoriamente
        random.shuffle(patient_dirs)
        # Calcular la cantidad de pacientes para entrenamiento y evaluación (70 / 30)
        num_train = int(len(patient_dirs) * TRAIN_RATIO)
        num_eval = len(patient_dirs) - num_train



        with open(file_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            # Escribir encabezados
            csv_writer.writerow(["id", "set", "patient_id", "tile_id"])
            # Escribir datos
            csv_writer.writerows(csv_data)

        print(f"Archivo CSV creado en: {file_path}")

def add_tiles_to_partitions():
    return ""