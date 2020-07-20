import av
import os
import sys
import torchvision.transforms.functional as F

MAX_SCALE = (320, 480)  # h, w


def load_video(filename):
    container = av.open(filename)
    return container


def save_video(frames, filename):
    return


def get_frames(container, start_sec, end_sec, time_base):
    margin = 1024
    start_pts = int(start_sec / time_base)
    end_pts = int(end_sec / time_base)
    seek_offset = max(start_pts - margin, 0)
    # seek to nearest key frame, and decode from it to get subsequent frames in [start_pts, end_pts].
    container.seek(seek_offset, any_frame=False, backward=True, stream=container.streams.video[0])
    frames = []
    for frame in container.decode(video=0):
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            image = frame.to_image()
            scaled_image = F.resize(image, size=MAX_SCALE)
            frames.append(scaled_image)
        else:
            break
    return frames


def gen_video(filename):
    part = int(sys.argv[1])
    total = int(sys.argv[2])
    prefix = sys.argv[3]
    video_list = open(filename, "r").readlines()
    for i, item in enumerate(video_list):
        if i % total != part:
            continue
        tts = item.strip().split()
        video_name = tts[0]
        labels = tts[1:]
        num = len(labels) // 3
        video_path = os.path.join(prefix, video_name)
        container = load_video(video_path)
        time_base = float(container.streams.video[0].time_base)
        for n in range(num):
            s = n * 3
            e = (n + 1) * 3
            label, start_sec, end_sec = labels[s:e]
            start_sec = float(start_sec)
            end_sec = float(end_sec)
            frames = get_frames(container, start_sec, end_sec, time_base)


