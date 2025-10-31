import itertools

import rclpy
from message_filters import TimeSynchronizer
from rclpy.clock import ROSClock
from rclpy.duration import Duration
from rclpy.logging import LoggingSeverity
from rclpy.time import Time


class ApproximateTimeSynchronizer(TimeSynchronizer):
    """
    Approximately synchronizes messages by their timestamps.

    Differences vs. stock:
    - slop: tolerance window for sync (seconds)
    - queue_offset: per-topic fixed time offsets (ns) to compensate sensor delays
    - allow_headerless: assign ROS time to headerless messages
    - primary_index: **NEW** index (0-based) of the topic that should be first
      in the emitted callback argument order, regardless of subscriber order.
    """

    def __init__(
        self,
        fs,
        queue_size,
        slop,
        queue_offset=False,
        allow_headerless=False,
        primary_index=None,
    ):
        """
        Args:
            fs: list of Subscribers
            queue_size: int
            slop: float (seconds)
            queue_offset: list[int] in ns (same length as fs) or False
            allow_headerless: bool
            primary_index: int or None
                If set, the synchronized callback will always receive the
                message from this index as the **first** argument, with the
                rest following in their original relative order.
        """
        TimeSynchronizer.__init__(self, fs, queue_size)
        self.slop = Duration(seconds=slop)
        self.allow_headerless = allow_headerless
        self.queue_offset = queue_offset
        if primary_index is not None:
            if not isinstance(primary_index, int):
                raise TypeError("primary_index must be an int or None")
            if not (0 <= primary_index < len(fs)):
                raise ValueError(
                    f"primary_index out of range (got {primary_index}, "
                    f"valid 0..{len(fs)-1})"
                )
        self.primary_index = primary_index

    def add(self, msg, my_queue, my_queue_index=None):
        if not hasattr(msg, "header") or not hasattr(msg.header, "stamp"):
            if not self.allow_headerless:
                msg_filters_logger = rclpy.logging.get_logger("message_filters_approx")
                msg_filters_logger.set_level(LoggingSeverity.INFO)
                msg_filters_logger.warn(
                    "can not use message filters messages "
                    "without timestamp infomation when "
                    '"allow_headerless" is disabled. '
                    "auto assign ROSTIME to headerless "
                    "messages once enabling constructor "
                    'option of "allow_headerless".'
                )
                return
            stamp = ROSClock().now()
        else:
            stamp = msg.header.stamp
            if not hasattr(stamp, "nanoseconds"):
                stamp = Time.from_msg(stamp)

        new_timestamp = stamp.nanoseconds
        if my_queue_index is not None and self.queue_offset:
            new_timestamp -= self.queue_offset[my_queue_index]

        self.lock.acquire()
        try:
            # Store message in per-topic dict keyed by timestamp
            my_queue[new_timestamp] = msg
            while len(my_queue) > self.queue_size:
                del my_queue[min(my_queue)]

            # Choose which queues to search (all, except possibly the one we just updated)
            search_queues = (
                self.queues
                if my_queue_index is None
                else self.queues[:my_queue_index] + self.queues[my_queue_index + 1 :]
            )

            # Collect candidate stamps within slop for each other topic
            stamps = []
            for queue in search_queues:
                topic_stamps = []
                for s in queue:
                    stamp_delta = Duration(nanoseconds=abs(s - new_timestamp))
                    if stamp_delta > self.slop:
                        continue
                    topic_stamps.append((Time(nanoseconds=s, clock_type=stamp.clock_type), stamp_delta))
                if not topic_stamps:
                    return
                topic_stamps = sorted(topic_stamps, key=lambda x: x[1])  # nearest first
                stamps.append(topic_stamps)

            # Try Cartesian products of best candidates
            for vv in itertools.product(*[list(zip(*s))[0] for s in stamps]):
                vv = list(vv)
                # Insert current topic's time back in the right place
                if my_queue_index is not None:
                    vv.insert(my_queue_index, stamp)
                qt = list(zip(self.queues, vv))

                # Check all present and within slop together
                if ((max(vv) - min(vv)) < self.slop) and (len([1 for q, t in qt if t.nanoseconds not in q]) == 0):
                    msgs = [q[t.nanoseconds] for q, t in qt]

                    # ---- NEW: Reorder so that primary_index (e.g., LiDAR) is first ----
                    if self.primary_index is not None:
                        # Bring primary message to front, keep others in original relative order
                        primary_msg = msgs[self.primary_index]
                        msgs = [primary_msg] + [m for i, m in enumerate(msgs) if i != self.primary_index]

                    # Emit synchronized messages
                    self.signalMessage(*msgs)

                    # Pop used stamps from all queues
                    for q, t in qt:
                        del q[t.nanoseconds]
                    break  # done after first valid sync
        finally:
            self.lock.release()

"""
import itertools

import rclpy
from message_filters import TimeSynchronizer
from rclpy.clock import ROSClock
from rclpy.duration import Duration
from rclpy.logging import LoggingSeverity
from rclpy.time import Time


class ApproximateTimeSynchronizer(TimeSynchronizer):
    
    Approximately synchronizes messages by their timestamps.

    :class:`ApproximateTimeSynchronizer` synchronizes incoming message filters
    by the timestamps contained in their messages' headers. The API is the same
    as TimeSynchronizer except for an extra `slop` parameter in the constructor
    that defines the delay (in seconds) with which messages can be synchronized.
    The ``queue_offset`` option allow to have temporal offset between subscribers
    , define as a list of offset int in nanoseconds.
    The ``allow_headerless`` option specifies whether to allow storing
    headerless messages with current ROS time instead of timestamp. You should
    avoid this as much as you can, since the delays are unpredictable.
    

    def __init__(self, fs, queue_size, slop, queue_offset=False, allow_headerless=False):
        TimeSynchronizer.__init__(self, fs, queue_size)
        self.slop = Duration(seconds=slop)
        self.allow_headerless = allow_headerless
        self.queue_offset = queue_offset

    def add(self, msg, my_queue, my_queue_index=None):
        if not hasattr(msg, "header") or not hasattr(msg.header, "stamp"):
            if not self.allow_headerless:
                msg_filters_logger = rclpy.logging.get_logger("message_filters_approx")
                msg_filters_logger.set_level(LoggingSeverity.INFO)
                msg_filters_logger.warn(
                    "can not use message filters messages "
                    "without timestamp infomation when "
                    '"allow_headerless" is disabled. '
                    "auto assign ROSTIME to headerless "
                    "messages once enabling constructor "
                    'option of "allow_headerless".'
                )
                return

            stamp = ROSClock().now()
        else:
            stamp = msg.header.stamp
            if not hasattr(stamp, "nanoseconds"):
                stamp = Time.from_msg(stamp)
            # print(stamp)
        new_timestamp = stamp.nanoseconds
        if my_queue_index is not None and self.queue_offset:
            new_timestamp -= self.queue_offset[my_queue_index]
        self.lock.acquire()
        my_queue[new_timestamp] = msg
        while len(my_queue) > self.queue_size:
            del my_queue[min(my_queue)]
        # self.queues = [topic_0 {stamp: msg}, topic_1 {stamp: msg}, ...]
        if my_queue_index is None:
            search_queues = self.queues
        else:
            search_queues = self.queues[:my_queue_index] + self.queues[my_queue_index + 1 :]
        # sort and leave only reasonable stamps for synchronization
        stamps = []
        for queue in search_queues:
            topic_stamps = []
            for s in queue:
                stamp_delta = Duration(nanoseconds=abs(s - new_timestamp))
                if stamp_delta > self.slop:
                    continue  # far over the slop
                topic_stamps.append(((Time(nanoseconds=s, clock_type=stamp.clock_type)), stamp_delta))
            if not topic_stamps:
                self.lock.release()
                return
            topic_stamps = sorted(topic_stamps, key=lambda x: x[1])
            stamps.append(topic_stamps)
        for vv in itertools.product(*[list(zip(*s))[0] for s in stamps]):
            vv = list(vv)
            # insert the new message
            if my_queue_index is not None:
                vv.insert(my_queue_index, stamp)
            qt = list(zip(self.queues, vv))
            if ((max(vv) - min(vv)) < self.slop) and (len([1 for q, t in qt if t.nanoseconds not in q]) == 0):
                msgs = [q[t.nanoseconds] for q, t in qt]
                self.signalMessage(*msgs)
                for q, t in qt:
                    del q[t.nanoseconds]
                break  # fast finish after the synchronization
        self.lock.release()
"""