#! /usr/bin/env python3
"""
This script is supposed to run on python3. if you get a ModuleNotFound : rospkg error. run pip3 install bagpy and try again.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from sensor_msgs.msg import PointField, PointCloud2
from std_msgs.msg import Header
from collections import OrderedDict
from datetime import datetime
import numpy.lib.recfunctions as rfn
import struct, os, re, math
# import rosbag
import numpy as np
import pandas as pd
import warnings
from numpy.linalg import inv
from tqdm import tqdm
# from logger import LOGGER
# from exceptions import NotABagError, InvalidTopicError
from sklearn.cluster import DBSCAN
from typing import Union, Tuple, List
from copy import deepcopy

# logger = LOGGER(Path(__file__).name, "debug").logger


class COLORS:
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[1;32m"
    reset = "\x1b[0m"
    green_reg = "\x1b[0;32m"
    cyan = "\x1b[0;36m"


class PcdHelper:
    """
    custom utilities for working with pointcloud2 data
    """

    def __init__(self):
        pass

    ################################
    #### COMMON UTILITIES START ####
    ################################

    def pf_to_type_size(self):
        """
        return: dict containing the mapping of Pointfield datatype (1-8) to bytesize and type, e.g 1 --> ("I", 1)
        """
        pf_to_type_size = OrderedDict(
            (
                (PointField.INT8, ("I", 1)),
                (PointField.UINT8, ("U", 1)),
                (PointField.INT16, ("I", 2)),
                (PointField.UINT16, ("U", 2)),
                (PointField.INT32, ("I", 4)),
                (PointField.UINT32, ("U", 4)),
                (PointField.FLOAT32, ("F", 4)),
                (PointField.FLOAT64, ("F", 8)),
            )
        )

        return pf_to_type_size

    def pcd_metadata_template(self):
        """
        returns an ordered dict template for generating  pcd header
        """
        metadata = OrderedDict(
            (
                ("VERSION", 0.7),
                ("FIELDS", []),
                ("SIZE", []),
                ("TYPE", []),
                ("COUNT", []),
                ("WIDTH", 0),
                ("HEIGHT", 0),
                ("VIEWPOINT", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                ("POINTS", 0),
                ("DATA", "binary"),
            )
        )

        return metadata

    def transform_coordinates(
        self,
        cloud: np.recarray,
        transformation_mat: np.ndarray,
        location_fields=["x", "y", "z"],
    ) -> np.recarray:
        """
        transforms the input cloud using tranformation matrix

        Args:
            cloud: cloud to be transformed
            transformation_mat: coordinate transformation matrix [R|t]
            location_fields: optional, list of fields to transform using trandformation mat.

        retunrs:
            locs: tranformed location_fields
        """

        cloud = deepcopy(cloud)

        # x,y,z, (n,)
        locs = rfn.repack_fields(cloud[location_fields])
    

        # (n,) -> (n,4)
        locs_3d = np.column_stack((rfn.structured_to_unstructured(locs), np.ones((locs.shape[0],1))))
        # locs_3d = np.hstack(
        #     (np.array(locs.tolist()), np.ones((np.array(locs.tolist()).shape[0], 1)))
        # )

        locs_transformed = np.matmul(transformation_mat, locs_3d.T).T

        locs[location_fields[0]], locs[location_fields[1]], locs[location_fields[2]] = (
            locs_transformed[:, 0],
            locs_transformed[:, 1],
            locs_transformed[:, 2],
        )

        return locs

    def get_dtype_mappings(self):
        """
        return: [Ordered dict] dtype mappings form POINTFIELD to np and vice versa
            pf_to_np: POINTFIELD to np
            np_to_pf: np to POINTFIELD
        """

        # mappings of POINTFIELD dtype to numpy dtype
        pf_to_np = OrderedDict(
            (
                (PointField.INT8, np.dtype("int8")),
                (PointField.UINT8, np.dtype("uint8")),
                (PointField.INT16, np.dtype("int16")),
                (PointField.UINT16, np.dtype("uint16")),
                (PointField.INT32, np.dtype("int32")),
                (PointField.UINT32, np.dtype("uint32")),
                (PointField.FLOAT32, np.dtype("float32")),
                (PointField.FLOAT64, np.dtype("float64")),
            )
        )

        # mappings of numpy dtype to POINTFIELD dtype
        np_to_pf = OrderedDict((nptype, pftype) for pftype, nptype in pf_to_np.items())

        return pf_to_np, np_to_pf

    def project_uv_to_xyz(self, intrinsic, extrinsic, w, pts_2d):
        """
        backproject pixel point on 3d coord based on given w
        Args:
            intrinsic: camera intrinsic parameters
            extrinsic: extrinsic matrix to 3d coord frame
            w: w from uvw value to calculate xyz
            pts_2d: list of pixel points (u,v)
        returns:
            out: numpy recarray with x,y,z,u,v,w fields
        """

        proj = np.matmul(intrinsic, extrinsic)
        proj_hom = np.vstack((proj, [0, 0, 0, 1]))
        proj_inv = inv(proj_hom)

        dtype = np.dtype(
            {
                "names": ["x", "y", "z", "u", "v", "w"],
                "formats": [
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                    np.float32,
                ],
            }
        )

        out = np.recarray((len(pts_2d),), dtype)

        for idx, pt in enumerate(pts_2d):
            u, v = pt
            pt_hom = np.array([u * w, v * w, w, 1])
            x, y, z, _ = np.matmul(proj_inv, pt_hom.T)
            temp = np.array((x, y, z, u, v, w), dtype)
            out[idx] = temp

        return out

    def np_to_pf_list(self, np_array: np.recarray) -> List[PointField]:
        """
        convert np recarray to pointfields list

        Args:
            np_array: a numpy recarry with dtype info [names and formats]
        returns:
            list of pointfields infered from dtype info of array.
        """

        assert (
            np_array.dtype.names is not None
        ), "Please pass a np array with dtype name and format info"

        _, np_to_pf = self.get_dtype_mappings()
        pfs = []
        # do not use offset info from array dtype because it may contain pad bytes
        # e.g when a recarray is a subarray of another recarray, and the fields are not repacked -> leads to pad bytes
        # if you need the list of pointfileds in accordance with the offset info from array itself, modify the func.
        # currently it infers the offset from dtype.temsize for each field , starting from 0
        offset = 0
        for dt in np_array.dtype.descr:
            np_dtype = np.dtype(dt[1])
            pf = PointField(
                name=dt[0], datatype=np_to_pf[np_dtype], offset=offset, count=1
            )
            offset += np_dtype.itemsize
            pfs.append(pf)

        return pfs

    def dbscan(
        self,
        eps: float,
        min_points: int,
        cloud: np.recarray,
        fields: list = None,
        x_thresh: Union[int, float, tuple] = None,
        v_thresh: Union[int, float, tuple] = None,
        z_thresh: Union[int, float, tuple] = None,
        sorted: bool = False,
    ) -> np.recarray:
        """
        perform dbscan clustering on given cloud and append cluster id field to the cloud.
        cloud must have "index" field.
        if x_thresh and/or v_thresh is given, the dbscan is performed on preprocessed cloud.
        all background points including cluster outliers have lables -1.

        Args:
            eps: dbscan eps parameter
            min_points: dbscan min_samples parameter
            cloud : numpy structured array
            fields: fields to be used for clustering
                    if fields is None, entire cloud is used in clustering
            sorted: if True, returns sorted cloud according to cluster id [-1 to len(db.labels_)]

        returns:
            cloud with additional cluster_id field assigned to each record in input cloud

        Process
        -------
        1.  A new field with fieldname `cluster_id` is added to `cloud`. The default value is -1.
            i.e, all points are labelled as background initially. The values will be updated based on dbscan output.
        2.  Based on the values of `x_thresh` and `v_thresh`, a  new preprocessed cloud is obtained.
            If both the parameters are None/Not supplied, then preprocessed cloud = raw cloud.
        3.  The dbscan [from sklearn] with given args is performed on preprocessed cloud.
            This gives us cluster labels (i.e `cluster_id`) for each point in preprocessed cloud.
        4.  The index location of each point of preprocessed cloud in raw cloud is matched by `index` field.
            The values of `cluster_id` field in raw cloud is updated based on the labels obtained from step 3.

        TODO :  allow custom distance metric for dbscan.
        """

        if (cloud is None) or (cloud.shape[0] == 0):
            logger.warning(f"No Cloud provided, dbscan returns 'None'.")
            return

        assert cloud.dtype.names is not None, "Cloud must have fieldnames"

        if "cluster_id" in cloud.dtype.names:
            raise NotImplementedError("Cannot cluster already clustered cloud.")

        if fields is None:
            fields = list(cloud.dtype.names)

        # all the field passed for clustering must be present in the cloud.
        assert all(
            [f in cloud.dtype.names for f in fields]
        ), f"cloud is missing atleast one field from {tuple(fields)}."

        row_cloud = cloud.copy()

        thresholded_cloud = self._preprocess_cloud(cloud, x_thresh, v_thresh, z_thresh)

        # cluster id array with all values = -1
        # replace actual cluster points with their ids
        dtype = np.dtype({"names": ["cluster_id"], "formats": [np.int16]})
        cl_ids = np.recarray(row_cloud.shape, dtype)
        cl_ids.fill(-1)

        raw_clustered_cloud = self.merge_arrays_vertically(row_cloud, cl_ids)

        if thresholded_cloud.shape[0] < min_points:
            return raw_clustered_cloud

        db = DBSCAN(eps=eps, min_samples=min_points)
        # maybe can avoid using pandas
        db.fit(pd.DataFrame.from_records(thresholded_cloud[fields]))

        # sanity check
        if db.labels_.shape[0] != thresholded_cloud.shape[0]:
            raise ValueError("dbscan output labels shape does not match cloud shape")

        for idx, cl_idx in zip(thresholded_cloud["index"], db.labels_):
            arg = np.where(raw_clustered_cloud["index"] == idx)[0].item()
            raw_clustered_cloud[arg]["cluster_id"] = cl_idx

        if sorted:
            raw_clustered_cloud.sort("cluster_id")

        return raw_clustered_cloud

    ##############################
    #### COMMON UTILITIES END ####
    ##############################

    #####################################
    #### PCD WRITING UTILITIES START ####
    #####################################

    def generate_seq(self, num_points, field_name="index", dt=np.uint16, start_point=0):
        """
        generates numpy seq 0-num_points
        Args:
            num_points: number of points in a cloud
            field_name: what name to give sequence field, default is "seq"
            dt: datatype, default uint16, if num_point is more, uint32 is taken
        return:
            seq: generated seq of 0-num_points with provided datatype and field name
        """

        # datatype for seq col, uint16 if it can handle num_points, else uint32
        dt = [np.uint32 if num_points > np.iinfo(dt).max else dt]
        # numpy dtype object with field name and datatype
        d_type = np.dtype({"names": [field_name], "formats": dt})
        # generated seq
        seq = np.linspace(start_point, num_points, num_points, endpoint=False, dtype=d_type)

        return seq

    def _handle_extra_bytes(self, extra_bytes, dummy_prefix="PAD"):
        """
        handles extra bites in creating pcd header, currently only handled byte sizes corresponding to POINTFIELD datatypes
        Args:
            extra_bytes: number of extra bytes
            pc2_pcd_mappings: mappings of pc2 datatype number to pointfield size, type
            dummy_prefix: prefix for pad bytes.
        return:
            typ: list of types for datatypes of extra bytes
            size: list of sizes, sum of the size list must be equal to num of extra bytes
            names: list of dummy field names
            counts: list of counts
        """
        # cannot handle 0 bytes
        assert extra_bytes > 0, f"Bytes greater than 0 expected, got: {extra_bytes} "
        # name of padded fields
        d_name = dummy_prefix
        # mappings of pf dtype(1-7) to type and size, e.g 1 --> ("I", 1)
        pf_to_type_size = self.pf_to_type_size()
        # initialising the placeholder for metadata of extra bytes
        typ, size, names, counts = [], [], [], []

        # reduce the byte count to less than 4
        while extra_bytes >= 4:
            # do not care about the datatype as long a it holds 4 bytes
            # select first value from the result, any datatype containing 4 bytes will do
            d_type, d_size = [
                (dtp, size) for dtp, size in pf_to_type_size.values() if size == 4
            ][0]

            # for  each data field, create metadata entry
            names.append(d_name)
            # type of the data field
            typ.append(d_type)
            # size of the datafield
            size.append(d_size)
            # counts (num of points)
            counts.append(1)
            # update extra byte count
            extra_bytes -= 4

        # only continue if there are more extra bytes
        if extra_bytes > 0:

            # we have no datatype containing 3 byte size, desolve it into 1 byte + 2 bytes
            if extra_bytes == 3:

                d_type, d_size = [
                    (dtp, size) for dtp, size in pf_to_type_size.values() if size == 1
                ][0]
                names.append(d_name)
                typ.append(d_type)
                size.append(d_size)
                counts.append(1)

                # update the counter
                extra_bytes -= 1

            d_type, d_size = [
                (dtp, size)
                for dtp, size in pf_to_type_size.values()
                if size == extra_bytes
            ][0]
            names.append(d_name)
            typ.append(d_type)
            size.append(d_size)
            counts.append(1)

        # sum of the size list must be equal to num of extra bytes
        return typ, size, names, counts

    def pc2_to_pcd_metadata(self, pc2_msg, dummy_prefix="PAD"):
        """
        generate pcd header for pc2 msg

        Args:
            pc2_msg: ros point cloud2 msg
            dummy_prefix: a prefix for pad bytes present in pc2_msg, if any.
        return: metadata dict for pcd header
        """
        # dummy field corresponding to extra bytes in point
        d_field = dummy_prefix
        # total byte size of a single point
        point_size = pc2_msg.point_step
        # template for writing pcd header
        metadata = self.pcd_metadata_template()
        # some metadata
        metadata["WIDTH"] = pc2_msg.width
        metadata["HEIGHT"] = pc2_msg.height
        metadata["POINTS"] = pc2_msg.height * pc2_msg.width
        #         md["POINTS"] = int(len(msg.data)/point_size)
        # mapping of pc2 datatype to pcd datatype
        pf_to_type_size = self.pf_to_type_size()

        offset = 0
        fields = [f for f in pc2_msg.fields]
        total_fields = len(fields)
        # iterate over fields
        for i, field in enumerate(fields):

            # mapping of pointfield datatype to field_type byte size
            field_typ, field_size = pf_to_type_size[field.datatype]

            # if the offset is not the same as actual offset of the field - keep adding dummy fields
            while offset != field.offset:
                # size of extra bytes
                extra_byte = field.offset - offset
                # get the lists of type, size, dummy_names ("_"), and counts of extra bytes
                # take number of extra bytes are map it to pointfield datatype
                d_types, d_sizes, d_names, d_counts = self._handle_extra_bytes(
                    extra_byte, dummy_prefix=d_field
                )
                # update the offset to account for above extra bytes
                offset = field.offset
                #  add meta data of dummy bytes
                metadata["FIELDS"].extend(d_names)
                metadata["TYPE"].extend(d_types)
                metadata["SIZE"].extend(d_sizes)
                metadata["COUNT"].extend(d_counts)

            # current offset is same as field offset
            if offset == field.offset:
                # add metadata of the field
                metadata["FIELDS"].append(field.name)
                metadata["TYPE"].extend(field_typ)
                metadata["SIZE"].append(field_size)
                metadata["COUNT"].append(field.count)
                # update the offset - current field metadata is taken care of
                offset += field_size

            # last field
            if (i + 1 == total_fields) and (field.offset + field_size != point_size):
                # update the offset to get total byte size of the point
                # field offset plus the size of the data
                offset = field.offset + field_size
                # calculate number of extra bytes between end of one point and start of new point
                extra_byte = point_size - offset
                # generate metadata for extra bytes
                d_types, d_sizes, d_names, d_counts = self._handle_extra_bytes(
                    extra_byte, dummy_prefix=d_field
                )
                # append metadata of extra bytes to main lists
                metadata["FIELDS"].extend(d_names)
                metadata["TYPE"].extend(d_types)
                metadata["SIZE"].extend(d_sizes)
                metadata["COUNT"].extend(d_counts)

        return metadata

    def metadata_to_pcd_header(self, metadata, rename_pad=True):
        """
        Given metadata as dictionary, return a string header.
        Args:
            metadata: metadata dict for pcd header
        return: list of strings for pcd header

        """
        # do not indent it, indenting will lead to store more bytes in pcd file binary_data
        template = """\
# .PCD v.7 - Point Cloud Data file format
VERSION {VERSION}
FIELDS {FIELDS}
SIZE {SIZE}
TYPE {TYPE}
COUNT {COUNT}
WIDTH {WIDTH}
HEIGHT {HEIGHT}
VIEWPOINT {VIEWPOINT}
POINTS {POINTS}
DATA {DATA}
"""
        # create a copy of the metadata
        str_metadata = metadata.copy()
        # leave padded byte fields as is
        if not rename_pad:
            str_metadata["FIELDS"] = " ".join(metadata["FIELDS"])
        # change padded byte fields to PAD
        else:
            new_fields = []
            for f in metadata["FIELDS"]:
                if f == "_":
                    new_fields.append("PAD")
                else:
                    new_fields.append(f)
            # convert each entry in metadata to string and return as list of strings
            str_metadata["FIELDS"] = " ".join(new_fields)
        str_metadata["SIZE"] = " ".join(map(str, metadata["SIZE"]))
        str_metadata["TYPE"] = " ".join(metadata["TYPE"])
        str_metadata["COUNT"] = " ".join(map(str, metadata["COUNT"]))
        str_metadata["WIDTH"] = str(metadata["WIDTH"])
        str_metadata["HEIGHT"] = str(metadata["HEIGHT"])
        str_metadata["VIEWPOINT"] = " ".join(map(str, metadata["VIEWPOINT"]))
        str_metadata["POINTS"] = str(metadata["POINTS"])
        tmpl = template.format(**str_metadata)

        return tmpl

    def bag_to_pcd(
        self,
        bagfile: Path,
        save_dir: Path,
        topic: str,
        sensor: str,
        sensor_id: int = 1,
        overwrite: bool = False,
    ):
        """
        read pc2 msg from bagfile and write it to pcd
        naming of pcd : <sensor>_<sensor_id>__<%YYYY-%MM-%DD-%HH-%MM-%SS-%MSE>
        sensor id is `0` padded to 2 decimals (i.e 01 .... 10)

        Args:
            bagfile: path to bag file
            save_dir: path to save the pcds
            topic: pc2 topic
            sensor_id: used as prefix to pcd, see naming of pcd above
            overwrite: if user want to overwrite the eixsting data or not, if False, user will not be asked to over write
                       if true and the data is already extracted before, user will be asked if the data should be overwritten or not
        """

        # extra check
        bagfile = Path(bagfile)
        save_dir = Path(save_dir)

        if bagfile.suffix != ".bag":
            raise NotABagError(
                f"Invalid File : {COLORS.red}{bagfile.name!r}{COLORS.reset}"
            )

        # rosbag bag object
        bag = rosbag.Bag(bagfile)
        bag_name = bagfile.name

        # checking if the topic exist in rosbag
        topic_list = bag.get_type_and_topic_info()[1].keys()
        if topic not in topic_list:
            logger.warning(
                f"Skipping {COLORS.yellow}[{bag_name!r} : {topic!r}]{COLORS.reset}, : {COLORS.red}Topic not found{COLORS.reset}"
            )
            return
        # topic must publish pointcloud2 msg
        topic_msg_type = bag.get_type_and_topic_info()[1][topic].msg_type
        if topic_msg_type != "sensor_msgs/PointCloud2":
            raise InvalidTopicError(
                f"Invaid topic for {sensor!r}: {COLORS.green}{topic!r}{COLORS.reset}"
            )

        WRITE = True

        # total number of msgs on topic=topic within this rosbag
        tot_msgs = bag.get_type_and_topic_info()[1][topic].message_count
        saved_msgs = len(list(save_dir.glob("*.pcd")))

        # if the save dir is populated with same number of files as total_msgs, do not write the pcds
        if saved_msgs == tot_msgs:
            WRITE = False
            logger.warning(
                f"Skipping {COLORS.yellow}[{bag_name!r} : {topic!r}]{COLORS.reset}, : {COLORS.green}Data Exist{COLORS.reset}"
            )
            if overwrite:
                ow = input(
                    f"Want to overwrite the content of '{topic}' for '{bag_name}'\n"
                )
                if ow.lower() in ["yes", "y"]:
                    WRITE = True

        if WRITE:

            save_dir.mkdir(parents=True, exist_ok=True)
            # listen for msgs comming from topic=topic,
            # loop over the msgs, convert header time into datetime, create name string for saving the pcd
            for tpc, msg, t in tqdm(
                bag.read_messages(topics=topic),
                leave=True,
                total=tot_msgs,
                colour="green",
                dynamic_ncols=True,
                desc=f"{COLORS.cyan}| {bag_name} - {sensor}_{str(sensor_id).zfill(2)} - {topic} |{COLORS.reset}",
            ):

                header = msg.header
                # epoch time in seconds
                epoch_time_sec = header.stamp.to_sec()
                # get the ms
                ms = format(epoch_time_sec, "f").split(".")[-1][:3]
                # epoch_time/1e9 if epoch time is in nsec, epoch_time/1e6 if in ms
                # date time string = %YYYY-%MM-%DD-%HH-%MM-%SS-%MSE
                date_time = datetime.fromtimestamp(float(epoch_time_sec)).strftime(
                    "%Y-%m-%d-%H-%M-%S-{}".format(ms)
                )
                # pcd name string
                pcd_name = f"{sensor}_{str(sensor_id).zfill(2)}__{date_time}.pcd"

                # metadata created from pc2 msg
                metadata = self.pc2_to_pcd_metadata(msg)
                # pcd header to write to binary pcd file - this is standard header
                pcd_header = self.metadata_to_pcd_header(metadata)
                pcd_header_bin = pcd_header.encode("ASCII")

                # actual binary data comming from pc2 msg
                data_bin = msg.data

                # writing the pcd file
                with open(os.path.join(save_dir, pcd_name), "wb") as pcd:
                    #                 for ln in pcd_header:
                    #                     pcd.write(bytearray(ln, "ascii"))
                    pcd.write(pcd_header_bin)
                    pcd.write(data_bin)
            logger.info(
                f"Success : {COLORS.green}[{bag_name!r} : {topic!r}]{COLORS.reset}"
            )

    def np_to_pcd(self, np_array, save_path,file_name):
        """
        write numpy array as pcd file
        Args:
            np_array: numpy array with field and dtype info
            save_path: path to save the pcd [with .pcd extension]
        """
        # the input array must have field names, and dtypes
        assert (
            np_array.dtype.fields is not None
        ), "Expected numpy array with fields and dtype info"

        if not str(save_path).endswith(".pcd"):
            save_path = Path(save_path).joinpath(file_name)

        assert Path(
            save_path
        ).parent.is_dir(), (
            f"Directory does not exist : {str(Path(save_path).absolute().parent)!r} "
        )

        # get pcd metadata dict from numpy array
        metadata = self.np_to_metadata(np_array)
        # metadata dict -> pcd file header
        pcd_header = self.metadata_to_pcd_header(metadata)
        # ASCII for pcd
        pcd_header_bin = pcd_header.encode("ASCII")

        # Approach 1 -> using python struct library for binary encoding
        # generate type_str for encoding to binary data
        # type_str = self.get_type_str(metadata)
        # binary string containing all the data
        #         data_bin = b''
        #         st = struct.Struct(type_str).pack

        # Approach 2 -> using numpy
        # faster than struct
        data_bin = np_array.tobytes()

        # convert each point to binary representation and generate one binary string
        #         for point in range(metadata["POINTS"]):
        #             point_bin = np_array.to_bytes()
        # #             point_bin = st(*np_array[point])
        #             data_bin += point_bin

        # writing the pcd file
        with open(str(save_path), "wb") as pcd:
            pcd.write(pcd_header_bin)
            pcd.write(data_bin)

    def merge_arrays_vertically(self, *args):
        """
        Merge all the arrays left to right vertically (in input seqeunce)
        Args:
            numpy arrays with dtype and field names
        return:
            merged array with all the columns
        descr:
            it will merge the second input array to the right side of first array
        """
        # Convert args tuple to a list so we can modify it
        args = list(args)
        # num of rows of 0th array
        num_rows = args[0].shape[0]

        # check for the type of all input args
        assert all(
            isinstance(arr, np.ndarray) for arr in args
        ), "expected input : numpy arrays"
        # all the input arrays must have field names, and dtypes
        assert all(
            arr.dtype.fields is not None for arr in args
        ), "expected all arrays are with field names, and dtypes"
        # check for shape compatibility while merging
        assert all(
            arr.shape[0] == num_rows for arr in args
        ), "expected all arrays with same row count"

        # merge all the arrays left to right (in input seqeunce)
        merged_array = rfn.merge_arrays((args), flatten=True, asrecarray=True)

        return merged_array

    ###################################
    #### PCD WRITING UTILITIES END ####
    ###################################

    #####################################
    #### PCD READING UTILITIES START ####
    #####################################

    def get_type_str(
        self, metadata, dummy_name="PAD", remove_pad=True, byte_order="little_endian"
    ):
        """
        create a type string from metadata for unpacking the bytes
        Args:
            metadata: metadata dict generated from pcd header
            remove_pad: if True, padding bytes will not be decoded while constructing type string, instead "x" will be put
            dummy_field: name of the field corresponding to pad bytes
            byte_order: in which order the bytes were encoded

        return: type string for unpacking the bytes
        """
        meta = metadata.copy()
        #         h = meta["HEIGHT"]
        #         w = meta["WIDTH"]
        # field sizes will be identical to meta["size"] if all counts are 1
        # else ith element will be the multiplication of meta["size"][i]*meta["count"][i]
        field_sizes = [
            meta["SIZE"][i] * meta["COUNT"][i] for i in range(len(meta["SIZE"]))
        ]
        # expected sizes of fields
        expected_sizes = [0, 1, 2, 4, 8]

        # Lookup table according to python struct library
        unpacking_lut = {
            "F": {2: "e", 4: "f", 8: "d"},
            "I": {1: "b", 2: "h", 4: "i", 8: "q"},
            "U": {1: "B", 2: "H", 4: "I", 8: "Q"},
            "byte_order": {"little_endian": "<", "big_endian": ">", "native": "@"},
            "pad_byte": "x",
        }

        # define the byte order to unpack
        start_byte = unpacking_lut["byte_order"][byte_order]
        # intantiate type string starting with starting byte
        type_str = start_byte + ""

        if remove_pad:
            # do not decode padded bytes
            # for ith type and ith size, get PYTHON STRUCT decoding string format, and add it to type string
            for typ, size, field in zip(meta["TYPE"], field_sizes, meta["FIELDS"]):
                if field != dummy_name:
                    # processing actual field
                    char_fmt = unpacking_lut[typ][size]
                    type_str += char_fmt
                else:
                    # pad field encountered
                    char_fmt = str(size) + unpacking_lut["pad_byte"]
                    type_str += char_fmt

            # alternative way of getting type str, more compact but less readable
            # char_str = ''.join(unpacking_lut[typ][size] if field != dummy_field else str(size)+unpacking_lut["pad_byte"] for typ, size, field in zip(md["TYPE"], field_sizes, md["FIELDS"]))
            # type_str += char_str

        else:
            # decode padded bytes - user wants to keep padded bytes
            # field byte size must be in expected sizes, otherwise it will give unpacking lut key error
            for idx, sz in enumerate(field_sizes):
                assert (
                    sz in expected_sizes
                ), f"expected byte size in: {expected_sizes}, got '{sz}' at index '{idx}'"

            # for ith type and ith size, get PYTHON STRUCT decoding string format, and add it to type string
            for typ, size in zip(meta["TYPE"], field_sizes):
                if size == 0:
                    continue
                else:
                    # cannot handle size of 15? (non existant in lut)
                    char_fmt = unpacking_lut[typ][size]
                    type_str += char_fmt
            # alternative way of getting type str, more compact but less readable
            # char_str = ''.join(unpacking_lut[typ][size] for typ, size in zip(meta["TYPE"], field_sizes))
            # type_str += char_str

        # total byte size of one point calculated from generate type string
        str_size = struct.calcsize(type_str)
        # total byte size of one point calculated from sizes of all the fields from pcd metadata
        point_size = sum(field_sizes)
        # size calculated with type string must match  the actual size, otherwise decoding will be erroneous
        assert (
            str_size == point_size
        ), f"Calculated byte size of a point from type string : {str_size} DIFFERES from actual byte size {point_size}"

        return type_str

    def np_to_metadata(self, np_array):
        """
        convert numpy structured array with dtype info into metadata dict for writing pcd
        Args:
            np_array: A numpy structured array with field names, and dtype info for each columns
        return:
            metadata: metadata dict based on numpy array
        """
        # TODO height*width = num_points, height == 1 always???

        # the input array must have field names, and dtypes
        assert (
            np_array.dtype.fields is not None
        ), "Expected numpy array with fields and dtype info"

        # dtype mappings from np to pf and vice versa
        pf_to_np, np_to_pf = self.get_dtype_mappings()
        # pf to size and type mappings
        pf_to_type_size = self.pf_to_type_size()
        # get the metadata template
        metadata = self.pcd_metadata_template()

        # field names
        field_names = list(np_array.dtype.names)
        # number of fields
        num_fields = len(np_array.dtype)
        # number of points = number of rows
        num_points = np_array.shape[0]

        # list of np dtypes for each columns
        np_dtypes = [np_array.dtype[i] for i in range(num_fields)]

        # only datatypes that can be converted to pf datatype are expected
        for dt in np_dtypes:
            assert (
                dt in np_to_pf.keys()
            ), f"numpy dtype '{dt}' cannot be converted to pf type\nAvailable numpy dtypes are: {list(np_to_pf.keys())}"

        # list of pf dtypes for each columns
        pf_dtypes = [np_to_pf[i] for i in np_dtypes]
        # pf to type and byte size, list of tuples, e.g, 1 --> [("I", 1)..... ("I", 1)]
        type_size = [pf_to_type_size[i] for i in pf_dtypes]
        # 0th element in each tuple is type (i.e first column), 1st element is byte size
        # use tuple unpacking within list comprehension to get each column as a list
        # list of types and sizes corresp. to each field
        types, sizes = [list(col) for col in zip(*type_size)]

        # populate the metadata template
        metadata["FIELDS"] = field_names
        metadata["TYPE"] = types
        metadata["SIZE"] = sizes
        metadata["WIDTH"] = num_points
        metadata["HEIGHT"] = 1
        metadata["POINTS"] = num_points
        metadata["COUNT"] = [1] * len(field_names)

        return metadata

    def metadata_to_npdtype(self, metadata, dummy_prefix="PAD", remove_pad=True):
        """
        create a list of tuples containing field name, np_dtype
        Args:
            metadata: metadata of cloud
            dummy_name: name of the dummy field
        return:
            np_dtypes: list of tuples containing field name, np_dtype
        """
        dummy_count = 0
        # list containing numpy datatypes
        np_dtypes = []
        # metadata
        fields = []
        types = []
        sizes = []

        for f, t, s in zip(metadata["FIELDS"], metadata["TYPE"], metadata["SIZE"]):
            # only append the metadata if it is not dummy field
            if f != dummy_prefix:
                fields.append(f)
                types.append(t)
                sizes.append(s)

            else:
                # check if user wants to remove pad bytes or not, if not then append field name with _0 suffix
                if not remove_pad:
                    fields.append(dummy_prefix + f"_{dummy_count}")
                    dummy_count += 1
                    types.append(t)
                    sizes.append(s)

                # if user wants to remove pad bytes, discard the corespondig metadata
                else:
                    continue

        # pointfield to (type, size) mapping e.g 1 --> ("I", 1)
        pf_to_type_size = self.pf_to_type_size()
        # pointfield to numpy dtype mappings
        pf_to_np, _ = self.get_dtype_mappings()

        for f, t, s in zip(fields, types, sizes):

            # get the pf datatype (key) where type,size in ith iteration matches to pointfield to (type, size) mapping
            pf = [k for k, v in pf_to_type_size.items() if v == (t, s)][0]

            # append tuple conataining field name, and numpy stype
            np_dtypes.append((f, pf_to_np[pf]))

        return np_dtypes

    def read_pcd(
        self,
        pcd_file,
        dummy_name="PAD",
        byte_order="little_endian",
        remove_pad=True,
        calc_vxvy=False,
    ):
        """
        reads pcd_file and returns numpy array of cloud
        IMPORTANT : REMOVE PAD = FALSE WILL BREAK THE CODE IF THE COUNT*SIZE IS not in [0,1,2,4,8]
        Args:
            pcd_file: pcd file path to read
            remove_pad: wether to remove dummy bytes, if True - only decode bytes that have values of actual fields
            dummy_field: name of the paded field
            byte_order: byte encoding order
        return:
            cloud: numpy array containing dtypes and field names
            [OPTIONAL] metadata: metadta dict containing pcd header data
        """
        pcd_file = str(pcd_file)
        assert pcd_file.endswith(
            ".pcd"
        ), f"Expected a file with .pcd extension, got '{pcd_file.split('.')[-1]}'"
        # metadata template for pcd
        metadata = self.pcd_metadata_template()

        # type mappings POINTFIELD to np and vice versa
        pf_to_np, np_to_pf = self.get_dtype_mappings()

        with open(pcd_file, "rb") as pcd:
            for line in pcd:
                ln = line.strip().decode("utf-8")
                # first line, or any line that is not imortance
                if ln.startswith("#") or len(ln) < 2:
                    continue
                # RE matching with the data of the header
                match = re.match("(\w+)\s+([\w\s\.]+)", ln)
                # no match detected, meaning the header is faulty
                if not match:
                    warnings.warn(f"warning: can't understand line: {ln}")
                    continue
                # header key, and values - all are strings
                key, value = match.group(1), match.group(2)

                # let the version stay default 0.7
                if key == "VERSION":
                    pass
                # this should also be same, but if the pcd containes data about it , update it - dtype float
                # specifies an acquisition viewpoint for the points in the datase
                # viewpoint format -  translation (tx ty tz) + quaternion (qw qx qy qz)
                if key == "VIEWPOINT":
                    metadata[key] = list(map(float, value.split()))
                # these fields should be converted into int data type - only one entry
                if key in ["POINTS", "HEIGHT", "WIDTH"]:
                    metadata[key] = int(value)
                # these fields belong to int, but list of entries
                if key in ["SIZE", "COUNT"]:
                    metadata[key] = list(map(int, value.split()))
                # convert single string to list of strings
                if key in ["TYPE", "FIELDS", "DATA"]:
                    metadata[key] = value.split() if len(value.split()) > 1 else value

                # here begins the actual binary data
                if ln.startswith("DATA"):
                    break

            # actual binary data
            binary_data = pcd.read()

            # np_dtypes for creating structured numpy array
            np_dtypes = self.metadata_to_npdtype(
                metadata, dummy_prefix=dummy_name, remove_pad=remove_pad
            )
            # only datatypes that can be converted to pf datatype are expected
            for f_name, dt in np_dtypes:
                assert (
                    dt in np_to_pf.keys()
                ), f"numpy dtype '{dt}' cannot be converted to pf type\nAvailable numpy dtypes are: {list(np_to_pf.keys())}"
            # size of a point as a list of individual field sizes
            field_sizes = [
                metadata["SIZE"][i] * metadata["COUNT"][i]
                for i in range(len(metadata["SIZE"]))
            ]
            # another way of getting the size of a point - both  sizes must match
            point_size = int(len(binary_data) / metadata["POINTS"])

            # checking if the read binary data makes sense
            # size1 = meta["size"]*count gives size of a field, sum of sizes of all fields gives point size
            # toatal num points are heigh*width, stored as meta["points"]
            # size2 = total length of binary data / total num of points should give us point size
            # size1 must be equal to size2, else there is some error in reading the file

            # binary data should be at least as long as numpoints*size
            assert sum(field_sizes) * metadata["POINTS"] <= len(
                binary_data
            ), f"Point size does not  match with header data"

            # get type string to decode the bytes
            type_str = self.get_type_str(
                metadata,
                remove_pad=remove_pad,
                dummy_name=dummy_name,
                byte_order=byte_order,
            )

            # decoding
            start = 0
            cloud = []
            # iterate over total points, for each point - get the chunk of binary data size corresp. to point size
            # decode using python struct libray and type string
            for pnt in range(metadata["POINTS"]):
                end = start + sum(field_sizes)
                pt = struct.unpack(type_str, binary_data[start:end])
                #print(pt)
                cloud.append(pt)
                start = end

            # cast as numpy recarray from list of tuples
            cloud_np = np.rec.array(cloud, dtype=np_dtypes)

            if calc_vxvy:
                cloud_np = self._calc_vxvy(cloud_np=cloud_np)
            # KEPT FOR FUTURE REFERENCE
            # cloud_df = pd.DataFrame.from_records(cloud_np)

            # if calc_vxvy:
            #     cloud_df["v_x"] = cloud_df.apply(
            #         lambda row: (
            #             math.cos(row.azimuth_angle)
            #             * (row.range_rate * math.cos(row.elevation_angle))
            #         ),
            #         axis=1,
            #     )

            #     cloud_df["v_y"] = cloud_df.apply(
            #         lambda row: (
            #             math.sin(row.azimuth_angle)
            #             * (row.range_rate * math.cos(row.elevation_angle))
            #         ),
            #         axis=1,
            #     )

            # cloud_np = cloud_df.to_records(index=False)
            if "index" not in cloud_np.dtype.names:
                seq = self.generate_seq(cloud_np.shape[0])
                cloud_np = self.merge_arrays_vertically(seq, cloud_np)

            return cloud_np, self.np_to_metadata(cloud_np)

    def _calc_vxvy(self, cloud_np):

        vx_func = lambda row: math.cos(row.azimuth_angle) * (
            row.range_rate * math.cos(row.elevation_angle)
        )
        vy_func = lambda row: math.sin(row.azimuth_angle) * (
            row.range_rate * math.cos(row.elevation_angle)
        )

        vx = np.asarray(
            list(map(vx_func, cloud_np)),
            dtype={"names": ["v_x"], "formats": [np.float32]},
        )
        vy = np.asarray(
            list(map(vy_func, cloud_np)),
            dtype={"names": ["v_y"], "formats": [np.float32]},
        )
        cloud_np = rfn.merge_arrays((cloud_np, vx, vy), flatten=True, asrecarray=True)

        return cloud_np

    def np_records_to_df(self, np_records, index=None):
        """
        create pandas df from numpy records array
        Args:
            np_records: a numpy record array with dtype and field info
            index: which field to treat as index, if the no records contain "seq" or "index"
                   it will be treated as index else None
        return:
            df: pandas df with dtype info for each column
        """
        # posible field names infered from numpy array
        field_names = list(np_records.dtype.names)
        dtypes = [np_records.dtype.fields[i][0] for i in np_records.dtype.names]

        # if user has passed index, check if it exist as a column
        if index is not None:
            # user chosen index must be in field names
            assert (
                index in field_names
            ), f"Expected index column from :\n {field_names}, got : '{index}'"
        else:
            # if index is none, use "seq" or "index" as index
            if "seq" in field_names:
                index = ["seq"]
            elif "index" in field_names:
                index = ["index"]

        # generate df from np records
        df = pd.DataFrame.from_records(np_records, index=index)

        return df

    def df_to_np_records(self, df):
        """
        create pandas df from numpy records array
        Args:
            df: a pandas df with dtype and field info
        return:
            np_records: a numpy records array with dtype and field name info.
        """

        # posible field names infered from numpy array
        field_names = list(df.columns)
        possible_idx = ["seq", "index", "idx"]
        # check if any of the argument of possible idx list exist in field names
        if any(map(lambda v: v in possible_idx, field_names)):
            choice = input(
                f"found 1 index like column in existing columns : 'seq'.\nType 'YES' to save it as index"
            )

            if choice.lower() in ["y", "yes"]:
                index = False
        else:
            choice = input(
                f"Found No index like column in existing columns.\nSave current index as a col?"
            )

            if choice.lower() in ["y", "yes"]:
                index = True

        # generate df from np records
        np_records = pd.DataFrame.to_records(df, index=index)

        return np_records

    def _apply_threshold(
        self,
        cloud: np.recarray,
        fieldname: str,
        value: Union[float, int, Tuple[int]],
        absolute: bool = False,
    ) -> np.recarray:
        """
        apply threshold of `value` to the `fieldname` field of cloud

        Args:
            cloud: pointcloud, must have `fieldname` field in cloud.dtype.names
            fieldname: name of the field to threshold
            value: threshold value
            abs: wether to use absolute values when thresholding, default is False
        returns:
            cloud: with `fieldname` thresholded by `value`
        """

        cloud = cloud.copy()

        assert (
            fieldname in cloud.dtype.names
        ), f"Cloud does not contain {fieldname!r} field"

        assert isinstance(
            value, (int, float, tuple)
        ), f"invalid type for {fieldname!r} {type(value)}."

        # conditional string to evaluate
        eval_str = "cloud[fieldname]"

        if absolute:
            eval_str = f"abs({eval_str})"

        if isinstance(value, (float, int)):
            cloud = cloud[eval(eval_str) > value]
            return cloud

        cloud = cloud[(eval(eval_str) > value[0]) & (eval(eval_str) <= value[1])]

        return cloud

    def _preprocess_cloud(
        self,
        cloud: np.recarray,
        x_thresh: Union[int, Tuple[int]],
        v_thresh: Union[float, int, Tuple[int]],
        z_thresh: Union[int, Tuple[int]] = None,
    ) -> np.recarray:
        """
        apply x_thresh and v_thresh to cloud.
        both threshold must be either int(upper_threshold) of tuple(low, up).

        Args:
            cloud: a numpy recarray representing a point cloud.
                it must have `x`, and `range_rate` fields
            x_thresh: int or tuple for x threshold
            v_thresh: int or tuple for range_rate_threshold

        returns:
            cloud: thresholded numpy array
        """
        X_CLIP, V_CLIP, Z_CLIP = True, True, True

        if x_thresh is None:
            X_CLIP = False

        if v_thresh is None:
            V_CLIP = False

        if z_thresh is None:
            Z_CLIP = False

        # return original cloud if x_thresh and v_thresh both are None
        if (not X_CLIP) and (not V_CLIP) and (not Z_CLIP):
            return cloud

        if X_CLIP:
            cloud = self._apply_threshold(cloud, "x", x_thresh)

        if V_CLIP:
            cloud = self._apply_threshold(cloud, "range_rate", v_thresh, absolute=True)

        if Z_CLIP:
            cloud = self._apply_threshold(cloud, "z_ground", z_thresh)

        return cloud

    def pc2msg_to_recarray(self, pc2_msg, pad="_", add_index=True) -> np.recarray:
        """
        converts PointCloud2 to numpy recarray and adds index field

        Args:
            pc2_msg : ROS PointCloud2 msg from sensor
        returns:
            cloud : a numpy recarray
        """
        import time

        metadata_dict = self.pc2_to_pcd_metadata(pc2_msg, dummy_prefix=pad)
        np_dtypes = self.metadata_to_npdtype(
            metadata_dict, dummy_prefix=pad, remove_pad=False
        )
        cloud_from_buffer = np.frombuffer(pc2_msg.data, np_dtypes)
        # remove pad fields
        cloud = cloud_from_buffer[[f for f, _ in np_dtypes if not f.startswith(pad)]]
        cloud = cloud.view(np.recarray)
        cloud = np.lib.recfunctions.repack_fields(cloud)
        if (add_index) and ("index" not in cloud.dtype.names):
            seq = self.generate_seq(cloud.shape[0])
            cloud = self.merge_arrays_vertically(cloud, seq)
        # print(cloud.dtype)

        return cloud

    def recarray_to_pc2msg(self, recarray: np.recarray, header=None):
        """
        converts a numpy recarray into sensor PointCloud2 msg

        Args:
            recarray: a numpy structured array
            header: header for pc2 msg
        returns:
            sensor msg PointCloud2
        """

        if header is None:
            header = Header(frame_id="map")
        # by default, recarray has shape (n,)
        # convert it to (1,n)
        arr_2d = np.atleast_2d(recarray)
        # print(self.np_to_pf_list(recarray))

        pc2_msg = PointCloud2(
            header=header,
            height=arr_2d.shape[0],
            width=arr_2d.shape[1],
            fields=self.np_to_pf_list(recarray),
            point_step=recarray.dtype.itemsize,
            row_step=arr_2d.shape[1] * recarray.dtype.itemsize,
            data=arr_2d.tobytes(),
        )

        return pc2_msg

    def project_pcd_on_image(
        self,
        pcd_np: np.recarray,
        image_shape: tuple,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        sensor: str,
        x_thresh: Union[int, Tuple[int], None] = None,
        v_thresh: Union[float, int, Tuple[float], None] = None,
    ) -> np.recarray:
        """
        calculates pixel coord of each point from numpy array of pcd file and returns numpy array of points that fits in the image

        Args:
            pcd_np: a numpy record array containing pcd data
            image_shape: tuple
            intrinsic: camera intrinsic calibration matrix
            extrinsic: exctrinsic calibration matrix of sensor with camera
            sensor: lidar or radar
            x_thresh : lower and upper threshold for x (used only in radar)
            v_thresh : velocity threshold in m/s (used only in radar)
        return:
             projected_cloud:
                            recarray with points that falls within the image with u,v,w fields added.
                            shape(input) != shape(output)
        """

        assert sensor.lower() in [
            "lidar",
            "radar",
        ], f"Expected sensor from ['lidar', 'radar'], got: '{sensor}'."

        # field names of input array
        field_names = pcd_np.dtype.names
        must_have_fields = ["x", "y", "z", "index"]

        assert all(
            field in field_names for field in must_have_fields
        ), f"at least one of the following fields are not present in input array:\n{must_have_fields}"

        cloud = pcd_np.copy()
        # TODO : remove ground
        if sensor.lower() == "radar":
            cloud = self._preprocess_cloud(cloud, x_thresh, v_thresh)

        # need (n,4) dimensions, 4th col values = 1
        cloud_xyz = cloud[["x", "y", "z", "index"]].copy()  # (n,4)
        cloud_xyz["index"] = 1  # (n,4)-[x,y,z,index(dummy)]
        # index array to keep track of points
        index = cloud["index"]
        # projection matrix
        p = np.matmul(intrinsic, extrinsic)  # (3,3)*(3,4) -> (3,4)
        # multiply points in 3d sensor dim to projection matrix to get u,v,w
        projected_points = np.matmul(
            p, rfn.structured_to_unstructured(cloud_xyz).transpose()
        )  # (3,4)*(4,n) -> (3,n)
        w = projected_points[2, :]
        # camera has only 2d, but each point in projected_points is 3 dimensional(u,v,w)
        # normalise each point by third dim (w) to get the actual pixel coord of point
        projected_points = np.array(
            [
                projected_points[0, :] / projected_points[2, :],  # u = x/w
                projected_points[1, :] / projected_points[2, :],  # v = y/w
            ]
        )  # (2,n)
        projected_points = np.transpose(projected_points)  # (n,2) - (u,v)
        # convert all the converted points into int from float
        # projected_points = projected_points.astype(int)
        # stack index
        projected_points = np.column_stack((index, projected_points, w))

        dtype = np.dtype(
            {
                "names": ["index", "u", "v", "w"],
                "formats": [np.uint32, np.float32, np.float32, np.float32],
            }
        )
        # # recarray
        # projected_points = np.array(np.rec.fromrecords(projected_points), dtype=dtype)
        projected_points = rfn.unstructured_to_structured(projected_points, dtype=dtype)
        (rows, cols, channels) = image_shape

        # keep only those points which are within image bounds
        projected_points = projected_points[
            (projected_points["u"] > 0)
            & (projected_points["v"] > 0)
            & (projected_points["u"] < cols)
            & (projected_points["v"] < rows)
        ]

        if projected_points.size == 0:
            return None

        indices = [
            idx
            for idx, _ in enumerate(pcd_np["index"])
            if _ in projected_points["index"]
        ]
        raw_points_on_image = pcd_np[indices]

        points_with_uv = rfn.merge_arrays(
            (raw_points_on_image, projected_points[["u", "v", "w"]]),
            flatten=True,
            asrecarray=True,
        )

        return points_with_uv


###################################
#### PCD READING UTILITIES END ####
###################################

    def transform_coordinates_return_with_orginal_intensity(
        self,
        cloud: np.recarray,
        transformation_mat: np.ndarray,
        location_fields=["x", "y", "z"],
    ) -> np.recarray:
        """
        transforms the input cloud using tranformation matrix

        Args:
            cloud: cloud to be transformed
            transformation_mat: coordinate transformation matrix [R|t]
            location_fields: optional, list of fields to transform using trandformation mat.

        retunrs:
            cloud: tranformed location_fields in the cloud data
        """

        cloud = deepcopy(cloud)

        # x,y,z, (n,)
        locs = rfn.repack_fields(cloud[location_fields])
    

        # (n,) -> (n,4)
        locs_3d = np.column_stack((rfn.structured_to_unstructured(locs), np.ones((locs.shape[0],1))))
        # locs_3d = np.hstack(
        #     (np.array(locs.tolist()), np.ones((np.array(locs.tolist()).shape[0], 1)))
        # )

        locs_transformed = np.matmul(transformation_mat, locs_3d.T).T

        cloud[location_fields[0]], cloud[location_fields[1]], cloud[location_fields[2]] = (
            locs_transformed[:, 0],
            locs_transformed[:, 1],
            locs_transformed[:, 2],
        )

        return cloud



    def project_points_uv(
        self,
        pts_xyz: np.ndarray,                # shape (N, 3)
        image_shape: Tuple[int, int, int],  # (rows, cols, channels)
        intrinsic: np.ndarray,              # 3x3
        extrinsic: np.ndarray               # 3x4  (camera <- world/sensor)
    ) -> np.recarray:
        """
        Projects arbitrary 3D points to an image using 3x3 K and 3x4 [R|t].
        Returns recarray with fields ['u','v','w'] for points inside the image bounds.
        """
        assert pts_xyz.ndim == 2 and pts_xyz.shape[1] == 3, "pts_xyz must be (N,3)"
        rows, cols = (image_shape[0], image_shape[1])

        # build a recarray similar to your project_pcd_on_image input
        N = pts_xyz.shape[0]
        rec_dtype = np.dtype([
            ("x", np.float64), ("y", np.float64), ("z", np.float64),
            ("index", np.uint32)
        ])
        cloud = np.empty(N, dtype=rec_dtype)
        cloud["x"] = pts_xyz[:, 0]
        cloud["y"] = pts_xyz[:, 1]
        cloud["z"] = pts_xyz[:, 2]
        cloud["index"] = np.arange(N, dtype=np.uint32)

        # build (N,4) with last col = 1
        cloud_xyz = cloud[["x", "y", "z", "index"]].copy()
        cloud_xyz["index"] = 1

        # P = K [R|t]
        P = intrinsic @ extrinsic  # (3x3)*(3x4) -> (3x4)

        proj = P @ rfn.structured_to_unstructured(cloud_xyz).T  # (3x4)*(4xN) -> (3xN)
        w = proj[2, :]

        # normalize
        uv = np.vstack([proj[0, :] / w, proj[1, :] / w])  # (2xN)
        uv = uv.T  # (N,2): [u,v]

        # keep only points with w>0 and inside the image
        mask = (w > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cols) & (uv[:, 1] < rows)
        if not np.any(mask):
            return None

        uvw = np.column_stack([uv[mask, 0], uv[mask, 1], w[mask]])
        dtype = np.dtype({"names": ["u", "v", "w"], "formats": [np.float32, np.float32, np.float32]})
        uvw_rec = rfn.unstructured_to_structured(uvw, dtype=dtype)
        return uvw_rec