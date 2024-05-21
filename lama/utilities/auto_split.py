from lama import common
from pathlib import Path
from logzero import logger as logging
import nrrd
import SimpleITK as sitk
import os
from scipy import ndimage
import numpy as np

target_dir = Path("E:/240417_two_tumours/monailabel_Bq/monai_two_tumours/labels/original")

out_dir = Path("E:/split_tumours/")

volpaths = common.get_file_paths(target_dir)

for path in volpaths:
    logging.info(f"Doing {os.path.basename(path)}")

    loader = common.LoadImage(path)
    img = loader.img
    logging.info(f"img loaded:")

    binary_mask = img > 0

    mask = sitk.ConnectedComponent(binary_mask)
    logging.info(f"Component Analysis done")

    # Find connected components
    mask = sitk.GetArrayFromImage(mask)
    labels, num_labels = ndimage.label(mask)

    component_sizes = np.bincount(labels.flatten())

    # Get the labels of the two largest components (excluding background)

    largest_components = np.argsort(component_sizes)[-3:-1]

    # Create a mask to keep only the largest components (excluding background)
    mask_to_keep = np.isin(labels, largest_components)

    # Apply the mask to keep only the largest components in the labels array
    out_mask = np.where(mask_to_keep, labels, 0)

    out_mask[out_mask > 0] = 1

    leftmost_voxel = float('inf')
    leftmost_label = None

    # Iterate over the two largest components
    for component_label in largest_components:
        # Find the bounding box of the component
        component_indices = np.where(labels == component_label)
        bbox = tuple(slice(min(axis), max(axis) + 1) for axis in component_indices)

        print(bbox)

        # Find the leftmost voxel in the component along the z-axis
        max_z = np.max(component_indices[2])

        # Update leftmost voxel and its label if this component is more leftmost
        if max_z < leftmost_voxel:
            leftmost_voxel = max_z
            leftmost_label = component_label
            print(leftmost_voxel, leftmost_label)

    # Modify the leftmost component in the largest component
    #out_mask = sitk.GetArrayFromImage(img)
    print(np.amax(out_mask))
    print(np.shape(out_mask))

    component_indices = np.where(labels == leftmost_label)
    print(np.shape(component_indices))
    bbox = tuple(slice(min(axis), max(axis) + 1) for axis in component_indices)

    # Update the values in the out_mask array using the global indices
    out_mask[component_indices] = 2

    print(np.amax(out_mask))
    print(np.shape(out_mask))
    lab_to_write = sitk.GetImageFromArray(out_mask)
    lab_to_write.CopyInformation(img)

    sitk.WriteImage(lab_to_write, str(out_dir / os.path.basename(path)), True)


