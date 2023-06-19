import os
import numpy as np
import matplotlib.pyplot as plt
from analyze_tool.tool_utils import rms, rms_ratio, draw_boxplot


class LinesBase:
    def __init__(self):
        self.name = None
        self.save_path = None
        self.pred_pts = None
        self.pt_dim = None
        self.lines = []
        self.normalize = None
        self.fix_box_max_value = None
        self.seq = None
        self.prob = None
        self.arr_dict = dict()
        self.color_list = ['b', 'r']
        self.alpha_list = [0.5, 0.5]

    def model_match_test(self, ref: np.ndarray, tgt: np.ndarray) -> dict:
        assert ref.shape[0] == tgt.shape[0]

        self.seq = ref.shape[0]
        # self.output_raw_data(ref, tgt)
        self.calculate_deviation(ref, tgt)
        if self.prob is not None:
            ref_prob = ref[:, self.prob[0]:self.prob[1]]
            tgt_prob = tgt[:, self.prob[0]:self.prob[1]]
            self.calculate_prob_deviation(ref_prob, tgt_prob)
        # visualize
        self.extract_lines(ref, tgt)
        # self.draw_xy_chart()
        return self.arr_dict

    def calculate_prob_deviation(self, ref, tgt):
        dev = rms_ratio(ref, tgt, do_sigmoid=True, distance='l1')
        box_list = []
        labels = []
        for i in range(dev.shape[0]):
            max_value = np.max(dev[i, [1, 3, 5, 7]])
            if max_value > 0.2:
                print(i, max_value)
            box_list.append(dev[i, [1, 3, 5, 7]])
            labels.append(i + 1)
        plt.figure(figsize=(19.2, 10.8), dpi=100)
        plt.boxplot(box_list, labels=labels)
        plt.savefig(f'{self.save_path}/prob_dev.jpg')
        plt.cla()
        plt.close()

    def output_raw_data(self, ref, tgt):
        s, l, p, d = self.seq, len(self.lines), self.pred_pts, self.pt_dim
        sz = self.pred_pts * self.pt_dim * len(self.lines)
        data = ['mean', 'std']
        for data_type, i in zip(data, range(len(data))):
            ref_slice = ref[:, i * sz:(i + 1) * sz].reshape(s, l, p, d)
            tgt_slice = tgt[:, i * sz:(i + 1) * sz].reshape(s, l, p, d)
            for line_no, line in enumerate(self.lines):
                ref_no_slice = ref_slice[:, line_no, :8, :]
                tgt_no_slice = tgt_slice[:, line_no, :8, :]
                with open(f"{line_no}.{self.name}_{line}_{data_type}.csv", 'w') as f:
                    title = 'seq_no'
                    for pt_no in range(ref_no_slice.shape[1]):
                        title += f',pt-{pt_no}-y,pt-{pt_no}-z'
                    title += '\n'
                    f.writelines(title)
                    for seq_no in range(ref_no_slice.shape[0]):
                        line = f"{seq_no}"
                        ref_single_data = ref_no_slice[seq_no, :, :]
                        ref_single_data = ref_single_data.reshape(-1)
                        for number in ref_single_data:
                            line += f",{number}"
                        line += '\n'
                        f.writelines(line)

    def calculate_deviation(self, ref, tgt):
        s, l, p, d = self.seq, len(self.lines), self.pred_pts, self.pt_dim
        sz = self.pred_pts * self.pt_dim * len(self.lines)
        data = ['mean', 'std']
        for data_type, i in zip(data, range(len(data))):
            ref_slice = ref[:, i * sz:(i + 1) * sz].reshape(s, l, p, d)
            tgt_slice = tgt[:, i * sz:(i + 1) * sz].reshape(s, l, p, d)
            dev_slice = rms(ref_slice, tgt_slice, normalize=self.normalize)
            # dev_slice = dev_slice.reshape(-1, self.pred_pts)
            for i in range(l):
                dev_line_slice = dev_slice[:, i, :]
                checkpoint = self.pred_pts // 4
                # visualize
                draw_boxplot(dev_line_slice, self.fix_box_max_value, checkpoint, f"{self.save_path}/{data_type}_{i}_diff")
            # draw_boxplot(dev_slice, self.fix_box_max_value, checkpoint, f"{self.save_path}/{data_type}_diff")

    def extract_lines(self, ref, tgt):
        sz = self.pred_pts * self.pt_dim
        for i, key in enumerate(self.arr_dict):
            self.arr_dict[key].append(ref[:, i * sz:(i + 1) * sz].reshape(self.seq, self.pred_pts, self.pt_dim))
            self.arr_dict[key].append(tgt[:, i * sz:(i + 1) * sz].reshape(self.seq, self.pred_pts, self.pt_dim))

    def draw_xy_chart(self):
        if not os.path.exists(f'{self.save_path}/track/'):
            os.mkdir(f'{self.save_path}/track/')
        for seq in range(self.seq):
            for line in self.arr_dict:
                for color, alpha, pts in zip(self.color_list, self.alpha_list, self.arr_dict[line]):
                    # plt.scatter(pts[seq, :, 0], pts[seq, :, 1], s=[10] * self.pred_pts, c=color, alpha=alpha)
                    plt.plot(pts[seq, :, 0], pts[seq, :, 1], '-o', c=color, alpha=alpha)
            plt.savefig(f'{self.save_path}/track/{seq:04}.jpg', dpi=192)
            plt.cla()
            plt.close('all')


class RoadEdges(LinesBase):
    def __init__(self, root_path, normalize=True):
        super().__init__()
        self.name = 'road_edges'
        self.save_path = f'{root_path}/{self.name}'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.pred_pts = 33
        self.pt_dim = 2
        self.lines = ['left', 'right']
        self.fix_box_max_value = 2
        self.normalize = normalize
        self.arr_dict = dict()
        for key in self.lines:
            self.arr_dict[key] = list()


class LaneLines(LinesBase):
    def __init__(self, root_path, normalize=True):
        super().__init__()
        self.name = 'lane_lines'
        self.save_path = f'{root_path}/{self.name}'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.pred_pts = 33
        self.pt_dim = 2
        self.lines = ['left_far', 'left_near', 'right_near', 'right_far']
        self.fix_box_max_value = 0.3
        self.normalize = normalize
        self.prob = [536 - 8, 536]
        self.arr_dict = dict()
        for key in self.lines:
            self.arr_dict[key] = list()
