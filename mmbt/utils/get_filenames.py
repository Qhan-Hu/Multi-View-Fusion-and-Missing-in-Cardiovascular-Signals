import os
import glob

def GetSubFiles(folder_path, extension):
    file_paths = glob.glob(os.path.join(folder_path, f'*.{extension}'))
    file_names = [os.path.basename(file_path) for file_path in file_paths]
    return file_names


def GetSubFolders(MainFolder):
    SubFolderList = []
    for fileName in os.listdir(MainFolder):
        SubFolderList.append(fileName)
    return SubFolderList

