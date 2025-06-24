import os
import json

import click
import numpy as np
import sys
from os.path import dirname, join
import torch
from tqdm import tqdm

sys.path.append(dirname(dirname(__file__)))
import torch as th
from timelens import attention_average_network
from timelens.common import (
    hybrid_storage,
    image_sequence,
    os_tools,
    pytorch_tools,
    transformers
)
from torchvision import transforms

def _interpolate(
        network,
        transform_list,
        interframe_events_iterator,
        boundary_frames_iterator,
        number_of_frames_to_interpolate,
        output_folder
):
    reverse = False
    
    output_frames, output_timestamps = [], []
    pytorch_tools.set_fastest_cuda_mode()
    combined_iterator = zip(boundary_frames_iterator, interframe_events_iterator) # 文件的迭代器
    if reverse:
        combined_iterator = reversed(list(combined_iterator))
    counter = 0 # 计算完成插值的帧
    
    for (left_frame, right_frame), event_sequence in tqdm(combined_iterator, desc = 'interpolate frames'):
        # print("Counter: %04d" % counter)
        output_timestamps += list( # 计算插值后的时间戳
            np.linspace(
                event_sequence.start_time(),
                event_sequence.end_time(),
                2 + number_of_frames_to_interpolate,
            )
        )[:-1]
        # print(event_sequence._features[:10])
        # if reverse:
            # event_sequence.reverse()
        iterator_over_splits = event_sequence.make_iterator_over_splits(
            number_of_frames_to_interpolate
        )
        output_frames.append(left_frame)# 输出帧包含插值的起始帧
        if reverse:
            output_frames[-1].save(join(output_folder, "{:06d}.png".format(1000 - counter)))
        else:
            output_frames[-1].save(join(output_folder, "{:06d}.png".format(counter)))
            
        counter += 1
        
        for split_index, (left_events, right_events) in enumerate(iterator_over_splits):
            # left_events = np.zeros(shape = left_events.shape)
            # right_events = np.zeros(shape = right_events.shape)
            print("Events left: ", len(left_events._features), "Events right: ", len(right_events._features))
            example = _pack_to_example(
                left_frame,
                right_frame,
                left_events,
                right_events,
                float(split_index + 1.0) / (number_of_frames_to_interpolate + 1.0),
            )# 只接受左右两侧的事件，不是端到端的
            example = transformers.apply_transforms(example, transform_list)
            example = transformers.collate([example])
            example = pytorch_tools.move_tensors_to_cuda(example)

            with torch.no_grad():
                frame, _ = network.run_fast(example)
    
            interpolated = th.clamp(
                frame.squeeze().cpu().detach(), 0, 1,
            )# 截断？
            output_frames.append(transforms.ToPILImage()(interpolated))
            if reverse:
                output_frames[-1].save(join(output_folder, "{:06d}.png".format(1000 - counter)))
            else:
                output_frames[-1].save(join(output_folder, "{:06d}.png".format(counter)))
            counter += 1

    output_frames.append(right_frame)
    if reverse:
        output_frames[-1].save(join(output_folder, "{:06d}.png".format(1000 - counter)))
    else:
        output_frames[-1].save(join(output_folder, "{:06d}.png".format(counter)))
    counter += 1

    return output_frames, output_timestamps


def _load_network(checkpoint_file):
    network = attention_average_network.AttentionAverage()
    network.from_legacy_checkpoint(checkpoint_file)
    network.cuda()
    network.eval()
    return network


def _pack_to_example(left_image, right_image, left_events, right_events, right_weight):
    return {
        "before": {"rgb_image": left_image, "events": left_events},
        "middle": {"weight": right_weight},
        "after": {"rgb_image": right_image, "events": right_events},
    }


def run_recursively(
        checkpoint_file,
        root_event_folder,
        root_image_folder,
        root_output_folder,
        number_of_frames_to_skip,
        number_of_frames_to_insert,
        bias,
):
    (root_image_folder, root_event_folder, root_output_folder) = [
        os.path.abspath(folder)
        for folder in [root_image_folder, root_event_folder, root_output_folder]
    ]

    # here we initialize the remapping function for events
    remapping_maps = None
    transform_list = transformers.initialize_transformers()
    network = _load_network(checkpoint_file)
    leaf_image_folders = os_tools.find_leaf_folders(root_image_folder)
    # print(len(leaf_image_folders))
    for leaf_image_folder in leaf_image_folders:
        relative_path = os.path.relpath(leaf_image_folder, root_image_folder) # 计算相对路径
        print("Processing {}".format(relative_path))
        leaf_event_folder = os.path.join(root_event_folder, relative_path)
        leaf_output_folder = os.path.join(root_output_folder, relative_path)
        storage = hybrid_storage.HybridStorage.from_folders(
            leaf_event_folder, leaf_image_folder, "*.npz", "*.png"
        )# 从文件中获取信息
        # print("make event iterator")
        interframe_events_iterator = storage.make_interframe_events_iterator(
            number_of_frames_to_skip, bias,
        )
        boundary_frames_iterator = storage.make_boundary_frames_iterator(
            number_of_frames_to_skip
        )
        
        print("Processing {}".format(leaf_output_folder))
        
        os.makedirs(leaf_output_folder, exist_ok=True)# 若没有输出文件夹则创建

        output_frames, output_timestamps = _interpolate(# 插值
            network,
            transform_list,
            interframe_events_iterator,
            boundary_frames_iterator,
            number_of_frames_to_insert,
            leaf_output_folder,
        )
        output_image_sequence = image_sequence.ImageSequence(
            output_frames, output_timestamps
        )

        input_image_sequence = storage._images.skip_and_repeat(number_of_frames_to_skip, number_of_frames_to_insert)
        output_image_sequence.to_folder(leaf_output_folder, file_template="frame_{:06d}.png") # 插值后图片
        output_image_sequence.to_video(os.path.join(leaf_output_folder, "interpolated.mp4")) # 插值后视频
        input_image_sequence.to_video(os.path.join(leaf_output_folder, "input.mp4")) # 输入视频


@click.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.argument("root_event_folder", type=click.Path(exists=True))
@click.argument("root_image_folder", type=click.Path(exists=True))
@click.argument("root_output_folder", type=click.Path(exists=False))
@click.argument("number_of_frames_to_skip", default=1)
@click.argument("number_of_frames_to_insert", default=1)
@click.argument("bias", default=0)
def main(
        checkpoint_file,
        root_event_folder,
        root_image_folder,
        root_output_folder,
        number_of_frames_to_skip,
        number_of_frames_to_insert,
        bias
):

    run_recursively(
        checkpoint_file,
        root_event_folder,
        root_image_folder,
        root_output_folder,
        number_of_frames_to_skip,
        number_of_frames_to_insert, # 每两帧之间需要插帧的数量
        bias,
    )


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
