"""
Load data for the respective datatypes.

The DataLoader.factory method returns the correct DataLoader subclass for the current data type (Intensity jacobians, organ volume...)
The DataLoader.line_iterator is returns LineData objects with all the data for a line needed to do the analysis.


Notes
-----
Currently: Converts output from registration to 8bit. As the registration pipeline only accepts 8bit images at the moment
this is ok. When we change to allow 16 bit images, we may have to change a few things in here <- put this in the class definition

JacobianDataGetter and IntensityDataGetter are currently the same (VoxelDataGetter subclasses) but they are separate classes as
we might add normalisation etc to IntensityDataGetter


- Refactor so lineIterator is same for each class to reduce code redundancy
- Large amounts of baseline daa is now being used. We need to modify code so for all lines, baseline data is
loaded only once
"""

from pathlib import Path
from typing import Union, List, Iterator, Tuple, Iterable, Callable
import math

import numpy as np
from addict import Dict
from logzero import logger as logging
import pandas as pd

from lama import common
from lama.img_processing.misc import blur
from lama.paths import specimen_iterator


GLCM_FILE_SUFFIX = '.npz'
DEFAULT_FWHM = 100  # um
DEFAULT_VOXEL_SIZE = 14.0


class LineData:
    """
    Holds the input data that will be analysed.
    Just a wrpper around a pandas DataFrame with methods to get various elements
    """
    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 info: pd.DataFrame,
                 line: str,
                 shape: Tuple,
                 paths: Tuple[List],
                 mask: np.ndarray = None,
                 outdirs = None,
                 cluster_data = None,
                 normalise: Callable = None):
        """
        Holds the input data to be used in the stats tests
        Parameters
        ----------
        data
            2D np.ndarray
                voxel_data
                    row: specimens
                    columns: data points
            pd.DataFrame
            organ volume data. Same as above but a pd.DataFrame with organ labels as column headers

        info
            columns:
                - specimen (index)
                - staging (the staging metric)
                - line
                - genotype
        shape
            Shape of an input volume
        paths:
            The input paths used to generate the data
            [0] Wildtype
            [1] mutants

        """
        self.data = data
        self.info = info
        self.shape = shape
        self.line = line
        self.paths = paths
        self.outdirs = None
        self.size = np.prod(shape)
        self.mask = mask

        if data.shape[0] != len(info):
            raise ValueError

    def mutant_ids(self):
        return self.info[self.info.genotype == 'mutant'].index

    def specimen_ids(self) -> List:
        return self.info.index.values

    def genotypes(self):
        return self.info.genotype

    def get_num_chunks(self, log = False):
        """
        Using the size of the dataset and the available memory, get the number of chunks needed to analyse the data without
        maxing out the memory.

        The overhead is quite large
            First the data is written to a binary file for R to read in
            Then there is the data (pvalues and t-statistics) output by R into another binary file
            Plus any overhead R encounters when fitting the linear models

        Currently The overhead factor is set to 10X the size of the data being analysed

        Parameters
        ----------
        chunk_size

        Returns
        -------

        """

        overhead_factor = 100

        bytes_free = common.available_memory()
        num_samples, num_voxels = self.data.shape

        try: # numpy array
            dtype_size = self.data.dtype.itemsize
        except AttributeError:
            dtype_size = self.data.values.dtype.itemsize

        data_size = dtype_size * num_samples * num_voxels

        # # Testing
        # bytes_free = 92.165 * (1024**3)
        # data_size = 14.529 * (1024 **3)

        num_chunks = math.ceil((data_size * overhead_factor) / bytes_free)

        if log:
            logging.info(f'\nAvailable memory: {round(bytes_free / (1024 **3), 3)} GB\nSize of raw data: {round(data_size / (1024**3), 3)} GB')

        if num_chunks > 1:
            return num_chunks
        else:
            return 1

    def chunks(self) -> Iterator[np.ndarray]:
        """
        Return chunks of the data.

        Yields
        -------
        Chunks split column-wise (axis=1)

        Notes
        -----
        np.array_split returns a view on the data not a copy

        # TODO: Allow chunk size to be set via config
        """
        num_chunks = self.get_num_chunks()
        chunks = np.array_split(self.data, num_chunks, axis=1)

        for data_chunk in chunks:
            if isinstance(data_chunk, pd.DataFrame):
                yield data_chunk.values
            else:
                yield data_chunk

    @property  # delete
    def mask_size(self) -> int:
        return self.mask[self.mask == 1].size


class DataLoader:
    """
    Base class for loading in data

    Notes
    -----
    TODO: add support for memory mapping data
    """
    def __init__(self,
                 wt_dir: Path,
                 mut_dir: Path,
                 mask: np.ndarray,
                 config: Dict,
                 label_info_file: Path,
                 lines_to_process: Union[List, None] = None,
                 baseline_file: Union[str, None] = None):
        """

        Parameters
        ----------
        wt_dir
        mut_dir
        mask
        config
        label_info_file
        lines_to_process
        baseline_file
            Path to csv containing baseline ids to use.
            If None, use all baselines
        """

        self.label_info: pd.DataFrame = None

        self.baseline_ids = self.load_baseline_ids(baseline_file)

        if label_info_file:
            self.label_info = pd.read_csv(label_info_file)

        self.wt_dir = wt_dir
        self.mut_dir = mut_dir
        self.config = config
        self.label_info_file = label_info_file
        self.lines_to_process = lines_to_process
        self.mask = mask  # 3D mask
        self.shape = None

        # This is set <- ?
        self.normaliser = None

        self.blur_fwhm = config.get('blur', DEFAULT_FWHM)
        self.voxel_size = config.get('voxel_size', DEFAULT_VOXEL_SIZE)


    @staticmethod
    def factory(type_: str):
        """
        Return an instance of the appropriate data loader class for the data type

        Parameters
        ----------
        type_
            the type of data to prepare

        Returns
        -------

        """

        if type_ == 'intensity':
            return IntensityDataLoader
        elif type_ == 'jacobians':
            return JacobianDataLoader
        elif type_ == 'organ_volumes':
            return OrganVolumeDataGetter
        else:
            raise ValueError(f'{type_} is not a valid stats analysis type')

    def load_baseline_ids(self, path):
        if path:
            ids = []
            with open(path, 'r') as fh:
                for line in fh:
                    ids.append(line.strip())
            return ids

    def _read(self, paths: List[Path]) -> np.ndarray:
        """
        Read in the data an a return a common 2D array independent on input data type

        Parameters
        ----------
        paths
            The paths to the data

        Returns
        -------
        2D array. Rows: specimens. columns: datapoints

        """

        raise NotImplementedError

    def cluster_data(self):
        raise NotImplementedError

    def filter_specimens(self, specimen_paths: List, staing: pd.DataFrame):
        to_drop = []
        filtered_paths = []
        for spec in specimen_paths:
            spec_id = spec.stem
            if spec_id in self.baseline_ids:
                filtered_paths.append(spec)
            else:
                to_drop.append(spec_id)
        filtered_staging = staing.drop(to_drop)

        if len(filtered_staging) != len(filtered_paths):
            raise ValueError('Check specimen ids in baseline_id file. They must match the file path ids')
        if len(filtered_paths) < 1:
            raise ValueError('0 baselines are included, Check specimen ids in baseline_id file. They must match the file path ids')

        return filtered_paths, filtered_staging

    def line_iterator(self) -> LineData:
        """
        The interface to this class. Calling this function yields and InpuData object
        per line that can be used to go into the statistics pipeline.

        The wildtype data is the same for each mutant line so we don't have to do multiple reads of the potentially
        large dataset

        Returns
        -------
        LineData
        """
        wt_metadata = self._get_metadata(self.wt_dir)
        wt_paths = list(wt_metadata['data_path'])

        wt_staging = get_staging_data(self.wt_dir)
        wt_staging['genotype'] = 'wildtype'

        if self.baseline_ids:
            wt_paths, wt_staging = self.filter_specimens(wt_paths, wt_staging)

        logging.info('loading baseline data')
        wt_vols = self._read(wt_paths)

        if self.normaliser:
            self.normaliser.add_reference(wt_vols)

            # ->temp bodge to get mask in there
            self.normaliser.mask = self.mask
            # <-bodge
            self.normaliser.normalise(wt_vols)

        # Make a 2D array of the WT data
        masked_wt_data = np.array([x.ravel() for x in wt_vols])

        mut_metadata = self._get_metadata(self.mut_dir, self.lines_to_process)

        # Iterate over the lines
        logging.info('loading mutant data')

        mut_gb = mut_metadata.groupby('line')

        for line, mut_df in mut_gb:

            mut_paths = list(mut_df['data_path'])
            mut_vols = self._read(mut_paths)

            if self.normaliser:
                self.normaliser.normalise(mut_vols)
            masked_mut_data = np.array([x.ravel() for x in mut_vols])

            # Make dataframe of specimen_id, genotype, staging
            mut_staging = get_staging_data(self.mut_dir, line=line)
            mut_staging['genotype'] = 'mutant'

            staging = pd.concat((wt_staging, mut_staging))
            # Id there is a value column, change to staging. TODO: make lama spitout staging header instead of value
            if 'value' in staging:
                staging.rename(columns={'value': 'staging'}, inplace=True)

            data = np.vstack((masked_wt_data, masked_mut_data))

            # cluster_data = self.cluster_data(data)  # The data to use for doing t-sne and clustering

            input_ = LineData(data, staging, line, self.shape, (wt_paths, mut_paths), self.mask)
            yield input_


class VoxelDataLoader(DataLoader):
    """
    Process the Spatial Jacobians generated during registration
    """
    def __init__(self, *args, **kwargs):
        super(VoxelDataLoader, self).__init__(*args, **kwargs)

    def cluster_data(self, data):
        pass
        #self.labe

    def _read(self, paths: Iterable) -> List[np.ndarray]:
        """
        - Read in the voxel-based data into 3D arrays
        - Apply guassian blur to the 3D image
        - mask
        - Unravel


        Parameters
        ----------
        paths
            Path to load

        Returns
        -------
        List of numpy arrays of blurred, masked, and raveled data
        """

        images = []

        for data_path in paths:
            logging.info(f'loading data: {data_path.name}')
            loader = common.LoadImage(data_path)

            if not self.shape:
                self.shape = loader.array.shape

            blurred_array = blur(loader.array, self.blur_fwhm, self.voxel_size)
            masked = blurred_array[self.mask != False]

            images.append(masked)

        return images

    def _get_data_file_path(self):
        """
        Return the path to tghe data for a specimen
        This is implemented in the subclasses as different datatypes may have different locations for the data files.
        For exmaple the registration data is in a seperate subfolder in the registration data dir.

        Returns
        -------

        """
        raise NotImplementedError

    def _get_metadata(self, root_dir: Path, lines_to_process: Union[List, None] = None) -> pd.DataFrame:
        """
        Get the data paths for the data type specified by 'datatype'

        Parameters
        ----------
        root_dir
            Registration output directory to search

        Returns
        -------
        DataFrame with columns:
            specimen', 'line', 'path'

        Raises
        ------
        FileNotFoundError if any data is missing
        """
        reg_out_dir = root_dir / 'output'
        specimen_info = []

        for line_dir in reg_out_dir.iterdir():

            if not line_dir.is_dir():
                continue

            if lines_to_process and line_dir.name not in lines_to_process:
                continue

            for spec_dir in line_dir.iterdir():

                if str(spec_dir).endswith('_'):  # legacy 'stats_' directory
                    continue

                spec_out_dir = spec_dir / 'output'

                if not spec_out_dir.is_dir():
                    raise FileNotFoundError(f"Cannot find 'output' directory for {spec_dir}\n"
                                            f"Please check data within {line_dir} folder")

                # data_dir contains the specimen data we are after
                data_dir = spec_out_dir / self.data_folder_name / self.data_sub_folder

                if not data_dir.is_dir():
                    raise FileNotFoundError(f'Cannot find data directory: {data_dir}')

                # Get the path to the data file for this specimen
                # Data file  will have same name as specimen with an image extension
                data_file = self._get_data_file_path(data_dir, spec_dir)

                if data_file and data_file.is_file():
                    # For each specimen we have: id, line and the data file path
                    specimen_info.append([spec_dir.name, line_dir.name, data_file, spec_out_dir])

                else:
                    raise FileNotFoundError(f'Data file missing: {data_file}')

        df = pd.DataFrame.from_records(specimen_info, columns=['specimen', 'line', 'data_path', 'output_dir'])
        return df


class JacobianDataLoader(VoxelDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datatype = 'jacobians'
        self.data_folder_name = 'jacobians'
        self.data_sub_folder = self.config['jac_folder']

    def _get_data_file_path(self, data_dir: Path, spec_dir: Path) -> Path:
        res = list(data_dir.glob(f'{spec_dir.name}*'))
        if res:
            return res[0]


class IntensityDataLoader(VoxelDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datatype = 'intensity'
        self.data_folder_name = 'registrations'
        self.data_sub_folder = self.config['reg_folder']

    def _get_data_file_path(self, data_dir: Path, spec_dir: Path) -> Path:
        # Intensity data is in a subfolder named the same as the specimen
        intensity_dir  = data_dir / spec_dir.name
        res = list(intensity_dir.glob(f'{spec_dir.name}*'))
        if res:
            return res[0]


class OrganVolumeDataGetter(DataLoader):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def line_iterator(self) -> LineData:
        wt_data = self._get_organ_volumes(self.wt_dir)
        mut_data = self._get_organ_volumes(self.mut_dir)

        # Iterate over the lines
        mut_gb = mut_data.groupby('line')
        for line, mut_df in mut_gb:

            if self.lines_to_process and line not in self.lines_to_process:
                continue

            mut_vols = mut_df.drop(columns=['line'])
            wt_vols = wt_data.drop(columns=['line'])

            # Make dataframe of specimen_id, genotype, staging
            wt_staging = get_staging_data(self.wt_dir)
            wt_staging['genotype'] = 'wildtype'
            mut_staging = get_staging_data(self.mut_dir, line=line)
            mut_staging['genotype'] = 'mutant'

            staging = pd.concat((wt_staging, mut_staging))
            # Id there is a value column, change to staging. TODO: make lama spitout staging header instead of value
            if 'value' in staging:
                staging.rename(columns={'value': 'staging'}, inplace=True)

            data = pd.concat((wt_vols, mut_vols))
            input_ = LineData(data, staging, line, self.shape, ([self.wt_dir], [self.mut_dir]))
            yield input_

    def get_metadata(self):
        """
        Override the parent class to get the organ volume paths rather than the volumes

        -------

        """
        pass

    def _get_organ_volumes(self, root_dir: Path) -> pd.DataFrame:
        """
        Given a root registration directory, collate all the organ volume CSVs into one file.
        Write out the combined organ volume CSV into the root registration directory.

        Parameters
        ----------
        root_dir
            The path to the root registration directory

        Returns
        -------
        The combined dataframe of all the organ volumes
        """
        output_dir = root_dir / 'output'

        dataframes = []

        for line_dir, specimen_dir in specimen_iterator(output_dir):

            organ_vol_file = specimen_dir / 'output' / common.ORGAN_VOLUME_CSV_FILE

            if not organ_vol_file.is_file():
                raise FileNotFoundError(f'Cannot find organ volume file {organ_vol_file}')

            df = pd.read_csv(organ_vol_file, index_col=0)
            self._drop_empty_columns(df)
            df['line'] = line_dir.name
            dataframes.append(df)

        # Write the concatenated organ vol file to single csv
        if not dataframes:
            raise ValueError(f'No data forund in output directory: {output_dir}')
        all_organs = pd.concat(dataframes)

        return all_organs

    def _drop_empty_columns(self, data: pd.DataFrame):
        """
        Rop data columns for the organ volumes that are not present in the label info file

        Returns
        -------

        """
        if self.label_info is not None:

            to_drop = []

            for organ_column in data:
                if not int(organ_column) in self.label_info.label.values:  # Maybe some gaps in the labelling
                    to_drop.append(organ_column)

            # Drop labels that are not present in the label info file
            data.drop(columns=to_drop, inplace=True)


def load_mask(parent_dir: Path, mask_path: Path) -> np.ndarray:
    """
    Mask is used in multiple datagetter so we load it independently of the classes.

    Parameters
    ----------
    parent_dir
        ?
    mask_path
        mmask_name

    Raises
    ------
    ValueError if mask contains anything other than ones and zeroes

    Returns
    -------
    mask 3D
    """
    mask = common.LoadImage(parent_dir / mask_path).array

    if set([0, 1]) != set(np.unique(mask)):
        logging.error("Mask image should contain only ones and zeros ")
        raise ValueError("Mask image should contain only ones and zeros ")

    return mask


def get_staging_data(root: Path, line=None) -> pd.DataFrame:
    """
    Collate all the staging data from a folder. Include specimens from all lines.
    Save a combined csv in the 'output' directory and return as a DataFrame too.

    Parameters
    ----------
    root
        The root directory to search
    line
        Only select staging data for this line

    """

    output_dir = root / 'output'

    dataframes = []

    for line_dir, specimen_dir in specimen_iterator(output_dir):

        if line and line_dir.name != line:
            continue

        staging_file = specimen_dir / 'output' / common.STAGING_INFO_FILENAME

        if not staging_file.is_file():
            raise FileNotFoundError(f'Cannot find organ volume file {staging_file}')

        df = pd.read_csv(staging_file, index_col=0)
        df['line'] = line_dir.name
        dataframes.append(df)

    # Write the concatenated organ vol file to single csv
    staging = pd.concat(dataframes)

    # Temp fix to deal with old data
    # If first column is 1 or 'value', change it to staging
    staging.rename(columns={'1': 'staging', 'value': 'staging'}, inplace=True)

    outpath = output_dir / common.STAGING_INFO_FILENAME
    staging.to_csv(outpath)

    return staging
