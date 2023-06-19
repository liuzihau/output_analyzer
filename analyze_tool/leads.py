import os
import numpy as np
import matplotlib.pyplot as plt
from analyze_tool.tool_utils import rms, rms_ratio, draw_boxplot


class Leads:
    def __init__(self, root_path, normalize=False):
        self.name = "leads"
        self.save_path = f'{root_path}/{self.name}'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.plans = 2
        self.pred_ts = 6  # [0, 2.0, 4.0, 6.0, 8.0, 10.0]
        self.pt_dim = 2  # xy
        self.va_dim = 2  # [v]elocity, [a]cceleration
        self.prob = np.array([[48, 51], [99, 102], [102, 105]])  # prob in 0, 2, 4s
        self.ts = ['0', '2.0', '4.0']
        self.fix_box_max_value = None
        self.normalize = normalize
        self.seq = None
        self.arr_dict = dict()
        for key in self.ts:
            self.arr_dict[key] = list()
        self.color_list = ['b', 'r']
        self.alpha_list = [0.5, 0.5]

    def model_match_test(self, ref: np.ndarray, tgt: np.ndarray) -> None:
        assert ref.shape[0] == tgt.shape[0]

        self.seq = ref.shape[0]
        self.calculate_deviation(ref, tgt)

        if self.prob is not None:
            ref_prob_list = []
            tgt_prob_list = []
            for i in range(self.prob.shape[0]):
                ref_prob_list.append(ref[:, self.prob[i, 0]:self.prob[i, 1]])
                tgt_prob_list.append(tgt[:, self.prob[i, 0]:self.prob[i, 1]])
            ref_prob = np.concatenate(ref_prob_list, axis=1)
            tgt_prob = np.concatenate(tgt_prob_list, axis=1)
            self.calculate_prob_deviation(ref_prob, tgt_prob)
        # visualize
        # self.extract_lines(ref, tgt)
        # self.draw_xy_chart()

    def calculate_prob_deviation(self, ref, tgt):
        dev = rms_ratio(ref, tgt, do_sigmoid=True, distance='l1')
        box_list = []
        labels = []
        for i in range(dev.shape[0]):
            max_value = np.max(dev[i, :])
            if max_value > 0.2:
                print(i, max_value)
            box_list.append(dev[i, :])
            labels.append(i + 1)
        plt.figure(figsize=(19.2, 10.8), dpi=100)
        plt.boxplot(box_list, labels=labels)
        plt.savefig(f'{self.save_path}/prob_dev.jpg')
        plt.cla()
        plt.close()

    def calculate_deviation(self, ref: np.ndarray, tgt: np.ndarray):
        plan_sz = self.pred_ts * self.pt_dim * self.va_dim * 2 + 3  # ( *2 means(mean, std)  +3 means prob)
        mean_sz = self.pred_ts * self.pt_dim * self.va_dim  # 24
        plans = range(self.plans)
        data = ['mean', 'std']
        for plan_idx in plans:
            for data_type, data_idx in zip(data, range(len(data))):
                start_point = plan_idx * plan_sz + data_idx * mean_sz
                end_point = plan_idx * plan_sz + (data_idx + 1) * mean_sz
                ref_mean = ref[:, start_point:end_point]
                ref_mean = ref_mean.reshape(self.seq, self.pred_ts, self.pt_dim + self.va_dim)
                tgt_mean = tgt[:, start_point:end_point]
                tgt_mean = tgt_mean.reshape(self.seq, self.pred_ts, self.pt_dim + self.va_dim)

                ref_mean_xy = ref_mean[:, :, :2]
                tgt_mean_xy = tgt_mean[:, :, :2]
                dev_mean_xy = rms(ref_mean_xy, tgt_mean_xy)
                # visualize
                draw_boxplot(dev_mean_xy, None, None, f"{self.save_path}/{plan_idx}_{data_type}_xy")

                ref_mean_va = ref_mean[:, :, 2:]
                tgt_mean_va = tgt_mean[:, :, 2:]
                dev_mean_va = rms(ref_mean_va, tgt_mean_va)
                # visualize
                draw_boxplot(dev_mean_va, None, None, f"{self.save_path}/{plan_idx}_{data_type}_va")

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
