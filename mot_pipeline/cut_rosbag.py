# python3 mot_pipeline/cut_rosbag.py \
#   --input /path/to/input_bag \
#   --output /path/to/output_bag_90_150 \
#   --storage-id sqlite3 \
#   --start 90 \
#   --end 150

# python3 mot_pipeline/cut_rosbag.py \
#   --input /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-33-37-818/inani__2026-02-27_07-33-37-818__bag1 \
#   --output /home/vh17r/ros2_yolo/inani/new25/new25_bag1 \
#   --storage-id sqlite3 \
#   --start 1772174290.300 \
#   --end 1772174319.300 \
#   --absolute


#new   - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-34-02-937 from 1772192080.300 to 1772192110.300 (low dense)
#new1  - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-34-02-937 from 1772192065.300 to 1772192085.300 (low dense but decide if its good)

#new2  - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-52-38-646 from 1772193183.000 to 1772193202.000 (low dense)
#new3  - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-52-38-646 from 1772193205.000 to 1772193225.000 (low dense)
#new4  - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-17-21-694 from 1772191055.000 to 1772191085.000 (mid dense)
#new5  - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-17-21-694 from 1772192094.000 to 1772191124.000 (mid dense)
#new6  - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-17-21-694 from 1772192134.000 to 1772191164.000 (low dense)
#new7  - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-17-21-694 from 1772192178.000 to 1772191208.000 (low dense)
#new8  - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-17-21-694 from 1772192222.000 to 1772191260.000 (low dense)
#new9  - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-17-21-694 from 1772192210.000 to 1772191225.000 (low dense)(small bag only 15 sec)

#new10 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-12-20-443 from 1772190770.000 to 1772190800.200 (high dense)
#new11 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-12-20-443 from 1772190835.000 to 1772190850.000 (high dense)(short bag only 15 sec)
#new12 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-12-20-443 from 1772190870.000 to 1772190900.000 (high dense)
#new13 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-12-20-443 from 1772191000.300 to 1772191030.300 (mid dense)
#new14 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-12-20-443 from 1772190907.000 to 1772190937.000 (mid dense)
#new15 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-12-20-443 from 1772190955.000 to 1772190985.500 (mid dense)

#new16 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_12-37-44-642 from 1772192279.900 to 1772192296.900 (low dense)
#new17 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_13-01-05-418 from 1772193801.800 to 1772193829.800 (mid dense)

#new18
#new18_bag1 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag1 from 1772174340.700 to 1772174370.600 (high dense)
#new18_bag2 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag2 from 1772174340.700 to 1772174370.600 (high dense)
#new19
#new19_bag1 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag1 from 1772174430.800 to 1772174460.600 (high dense)
#new19_bag2 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag2 from 1772174430.700 to 1772174460.600 (high dense)
#new20
#new20_bag1 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag1 from 1772174465.700 to 1772174490.600 (high dense)
#new20_bag2 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag2 from 1772174465.700 to 1772174490.600 (high dense)
#new21
#new21_bag1 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag1 from 1772174505.700 to 1772174535.600 (high dense)
#new21_bag2 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag2 from 1772174505.700 to 1772174535.600 (high dense)
#new22
#new22_bag1 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag1 from 1772174600.700 to 1772174620.600 (mid dense)
#new22_bag2 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag2 from 1772174600.700 to 1772174620.600 (mid dense)

#new23
#new23_bag1 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-43-40-839/inani__2026-02-27_07-43-40-839__bag1 from 1772174650.000 to 1772174675.000 (mid dense)
#new23_bag2 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-43-40-839/inani__2026-02-27_07-43-40-839__bag2 from 1772174650.000 to 1772174675.000 (mid dense)
#new24
#new24_bag1 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-43-40-839/inani__2026-02-27_07-43-40-839__bag1 from 1772174830.000 to 1772174850.000 (low dense had bicycle)
#new24_bag2 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-43-40-839/inani__2026-02-27_07-43-40-839__bag2 from 1772174830.000 to 1772174850.000 (low dense had bicycle)

#new25
#new25_bag1 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-33-37-818/inani__2026-02-27_07-33-37-818__bag1 from 1772174290.300 to 1772174319.300 (low dense)
#new25_bag2 - /home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-33-37-818/inani__2026-02-27_07-33-37-818__bag2 from 1772174290.300 to 1772174319.300 (low dense)

import argparse
import os
import sys

import rosbag2_py


def make_reader(uri: str, storage_id: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions('', '')
    )
    return reader


def make_writer(uri: str, storage_id: str):
    if os.path.exists(uri):
        raise RuntimeError(
            f"Output path already exists: {uri}\n"
            f"Choose a new output directory name."
        )

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions('', '')
    )
    return writer


def copy_topics(reader, writer):
    topics = reader.get_all_topics_and_types()
    for topic in topics:
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic.name,
                type=topic.type,
                serialization_format=getattr(topic, "serialization_format", "cdr")
            )
        )
    return topics


def main():
    parser = argparse.ArgumentParser(
        description="Cut a ROS 2 bag between two timestamps."
    )
    parser.add_argument("--input", required=True, help="Input bag directory")
    parser.add_argument("--output", required=True, help="Output bag directory")
    parser.add_argument(
        "--storage-id",
        default="sqlite3",
        help="Bag storage plugin, e.g. sqlite3 or mcap"
    )
    parser.add_argument(
        "--start",
        type=float,
        required=True,
        help="Start time in seconds"
    )
    parser.add_argument(
        "--end",
        type=float,
        required=True,
        help="End time in seconds"
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Interpret --start/--end as absolute bag timestamps (Unix epoch seconds). "
             "Default: treat them as seconds from bag start."
    )

    args = parser.parse_args()

    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")

    reader = make_reader(args.input, args.storage_id)
    writer = make_writer(args.output, args.storage_id)
    copy_topics(reader, writer)

    first_bag_ts = None
    start_ns = None
    end_ns = None

    total_read = 0
    total_written = 0
    first_written_ts = None
    last_written_ts = None

    while reader.has_next():
        topic, data, t = reader.read_next()
        total_read += 1

        if first_bag_ts is None:
            first_bag_ts = t

            if args.absolute:
                start_ns = int(args.start * 1e9)
                end_ns = int(args.end * 1e9)
            else:
                start_ns = first_bag_ts + int(args.start * 1e9)
                end_ns = first_bag_ts + int(args.end * 1e9)

        if t < start_ns:
            continue

        if t > end_ns:
            break

        writer.write(topic, data, t)
        total_written += 1

        if first_written_ts is None:
            first_written_ts = t
        last_written_ts = t

    print(f"Read messages   : {total_read}")
    print(f"Written messages: {total_written}")

    if total_written == 0:
        print("No messages found in the requested time window.", file=sys.stderr)
    else:
        print(f"First written ts: {first_written_ts} ns")
        print(f"Last written ts : {last_written_ts} ns")


if __name__ == "__main__":
    main()