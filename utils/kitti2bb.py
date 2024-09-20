import sys
import os
import numpy as np
import pickle

sys.path.append('.')

from utils.parseTrackletXML import parseXML
from utils.boundingbox import BoundingBox

def kitti2bb(annotation_file, export2file = False):
    parsedTracklet = parseXML(annotation_file)
    labels = {}
    for i in range(len(parsedTracklet)):
        frame = parsedTracklet[i].firstFrame
        if frame not in labels:
            labels[frame] = []
        labels[frame].append(BoundingBox(center = parsedTracklet[i].trans[0],
                                        dimensions=parsedTracklet[i].size,
                                        rotation=parsedTracklet[i].rots[0,2],
                                        label=parsedTracklet[i].objectType))
    if export2file:
        export_file = annotation_file.replace('.xml', '.pkl')
        with open(export_file, 'wb') as f:
            pickle.dump(labels, f)
    return labels

if __name__ == '__main__':
    annotation_folder = "/media/ssd-roger/Sajjad/HiDrive Dataset/EGO_annotation/WMG_700"
    for file in os.listdir(annotation_folder):
        if file.endswith('.xml'):
            kitti2bb(os.path.join(annotation_folder, file))