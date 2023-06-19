import os
import numpy as np
import matplotlib.pyplot as plt


def rms(ref, tgt, normalize=True):
    # SEQ LINE PTS XYZ(YZ)
    dev = (ref - tgt)
    if normalize:
        ref_distance = ref ** 2
        ref_distance = np.sum(ref_distance, axis=-1) ** (1 / 2)
        tgt_distance = tgt ** 2
        tgt_distance = np.sum(tgt_distance, axis=-1) ** (1 / 2)
        ref_distance[ref_distance == 0] = tgt_distance[ref_distance == 0]
    else:
        ref_distance = 1

    dev = dev ** 2
    dev = np.sum(dev, axis=-1) ** (1 / 2) / ref_distance
    return dev


def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))


def rms_ratio(ref, tgt, do_sigmoid=True, distance='l1'):
    # SEQ PROB
    if do_sigmoid:
        dev = sigmoid(tgt) / sigmoid(ref)
    else:
        dev = tgt / ref

    if distance == 'l2':
        dev = dev ** 2
        dev = np.abs(dev - 1)
    else:
        dev = np.abs(dev)
        dev = np.abs(dev - 1)
    return dev


def draw_boxplot(dev: np.ndarray, fix_value: (float, None), checkpoint: (int, None), filename: str) -> None:
    box_list = []
    labels = []
    for i in range(dev.shape[-1]):
        box_list.append(dev[:, i])
        labels.append(i + 1)
        if checkpoint and i == checkpoint:
            if fix_value is not None:
                plt.ylim(0, fix_value)
            plt.boxplot(box_list, labels=labels)
            plt.savefig(f'{filename}_{checkpoint}.jpg', dpi=192)
            plt.cla()
            plt.close()
    plt.boxplot(box_list, labels=labels)
    plt.savefig(f'{filename}_{dev.shape[-1]}.jpg', dpi=192)
    plt.cla()
    plt.close()


def draw_xyplot(dev: np.ndarray, checkpoint: (int, None), filename: str) -> None:
    box_list = []
    labels = []
    for i in range(dev.shape[-1]):
        box_list.append(dev[:, i])
        labels.append(i + 1)
        if checkpoint and i == checkpoint:
            plt.boxplot(box_list, labels=labels)
            plt.savefig(f'{filename}_{checkpoint}.jpg', dpi=192)
            plt.cla()
            plt.close()
    plt.boxplot(box_list, labels=labels)
    plt.savefig(f'{filename}_{dev.shape[-1]}.jpg', dpi=192)
    plt.cla()
    plt.close()


def draw_tracking_chart(img_path, save_path, positions, lane_lines, road_edges):
    """
    positions(list(ref, tgt)) : seq x pts x dim(xyz)
    lane_lines(list(ref, tgt)) : 4 x seq x pts x dim(yz)
    road_edges(list(ref, tgt)) : 2 x seq x pts x dim(yz)
    """
    color_list = ['b', 'r']
    alpha_list = [0.5, 0.5]
    if not os.path.exists(f'{save_path}'):
        os.mkdir(f'{save_path}')
    for sub_task in ['plan', 'road_information']:  # ['plan', 'lane_line', 'road_edge']:
        if not os.path.exists(f'{save_path}/{sub_task}'):
            os.mkdir(f'{save_path}/{sub_task}')
    equal_sequences = positions[0].shape[0] == lane_lines[0].shape[1] == road_edges[0].shape[1]
    assert equal_sequences, "position, lane lines, roar edges share different shape"

    sequence_frames = positions[0].shape[0]
    for seq in range(sequence_frames):
        img = plt.imread(f"{img_path}/{seq:04}.jpg")
        fig, ax = plt.subplots()
        fig.autolayout = True
        ax.imshow(img, extent=[-30, 30, 0, 60])

        # 重疊
        for position, road_edge, color, alpha in zip(positions, road_edges, color_list, alpha_list):
            ax.plot(position[seq, :17, 1], position[seq, :17, 0], '-o', c=color, markersize=2, alpha=alpha)
        plt.savefig(f"{save_path}/plan/{seq:04}.jpg", dpi=256)
        plt.cla()
        plt.close('all')

        # 分別各畫一張

        # for position, road_edge, color, alpha in zip(positions, road_edges, color_list, alpha_list):
        #     fig, ax = plt.subplots()
        #     fig.autolayout = True
        #     ax.imshow(img, extent=[-30, 30, 0, 60])
        #     ax.plot(position[seq, :17, 1], position[seq, :17, 0], '-o', c=color, markersize=2, alpha=alpha)
        #     plt.savefig(f"{save_path}/plan/{seq:04}_{color}.jpg", dpi=256)
        #     plt.cla()
        #     plt.close('all')

        # 畫lane line + rode edge
        img = plt.imread(f"{img_path}/{seq:04}.jpg")
        fig, ax = plt.subplots()
        fig.autolayout = True
        ax.imshow(img, extent=[-10, 10, 0, 20])
        for position, lane_line, color, alpha, in zip(positions, lane_lines, color_list, alpha_list):
            for line in lane_line:
                r = np.clip((3 - (position[seq, :17, 0] / 20)) / 3, 0, 1)
                ax.plot(line[seq, :17, 0] * 4 * r, position[seq, :17, 0] / 10, '-o', c='r', markersize=2, alpha=alpha)
            break
        # plt.savefig(f"{save_path}/lane_line/{seq:04}.jpg", dpi=256)
        # plt.cla()
        # plt.close('all')
        #
        # img = plt.imread(f"{img_path}/{seq:04}.jpg")
        # fig, ax = plt.subplots()
        # fig.autolayout = True
        # ax.imshow(img, extent=[-50, 50, -5, 95])
        for position, road_edge, color, alpha, in zip(positions, road_edges, color_list, alpha_list):
            for edge in road_edge:
                r = np.clip((3 - (position[seq, :17, 0] / 20)) / 3, 0, 1)
                ax.plot(edge[seq, :17, 0] * 4 * r, position[seq, :17, 0] / 10, '-o', c='y', markersize=2, alpha=alpha)
            break
        plt.savefig(f"{save_path}/road_information/{seq:04}.jpg", dpi=256)
        plt.cla()
        plt.close('all')
        if (seq + 1) % 20 == 0 or seq + 1 == sequence_frames:
            print(f'[Info] Now drawing tracking images ... {seq+1}/{sequence_frames}')
