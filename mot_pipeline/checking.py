import rosbag2_py
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from rosidl_runtime_py import message_to_yaml  # built-in safe serializer

LOCAL_TZ = ZoneInfo("Europe/Berlin")
FMT = "%Y-%m-%d_%H-%M-%S-%f"

def ns_to_local_str(nanoseconds: int) -> str:
    #sec, nsec = divmod(nanoseconds, 1_000_000_000)
    dt = datetime.fromtimestamp((sec + nsec) / 1e9, tz=timezone.utc)
    return dt.astimezone(LOCAL_TZ).strftime(FMT)[:-3]

bag_path   = "/home/vh17r/ros2_yolo/inani/inani__2025-10-17_07-30-27-642__bag1"
topic_name = "/object_list/tracks"
output_path = "output.txt"

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
    rosbag2_py.ConverterOptions("", "")
)

type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
if topic_name not in type_map:
    raise RuntimeError(f"Topic {topic_name} not in bag. Found: {list(type_map)}")

MsgType = get_message(type_map[topic_name])

with open(output_path, "w", encoding="utf-8") as f:
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(data, MsgType)

        # Optional: clear heavy byte arrays
        if hasattr(msg, "objects"):
            for o in msg.objects:
                if hasattr(o, "cluster") and hasattr(o.cluster, "data"):
                    o.cluster.data = []  # keep file small

        yaml_str = message_to_yaml(msg)  # <-- fixed for Humble

        f.write(f"---\nTopic: {topic}\nTime: {ns_to_local_str(t_ns)}\nType: {type_map[topic_name]}\nData:\n{yaml_str}\n")

print(f"✅ Saved readable messages from {topic_name} to {output_path}")
