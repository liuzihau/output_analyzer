import os
import numpy as np
import matplotlib.pyplot as plt
from analyze_tool.tool_utils import rms, rms_ratio, draw_boxplot


class PlansBase:
    def __init__(self):
        self.name = None
        self.root_path = None
        self.save_path = None
        self.pred_pts = None
        self.pt_dim = None
        self.element_category = []
        self.plan_count = None
        self.normalize = None
        self.seq = None
        self.prob = None
        self.arr_dict = dict()
        self.color_list = ['b', 'r']
        self.alpha_list = [0.7, 0.4]

    def model_match_test(self, ref: np.ndarray, tgt: np.ndarray) -> dict:
        assert ref.shape[0] == tgt.shape[0]

        self.seq = ref.shape[0]

        if self.prob is not None:
            ref_prob = ref[:, self.prob]
            tgt_prob = tgt[:, self.prob]
            self.draw_prob_distribution(ref_prob, tgt_prob)
            self.calculate_prob_deviation(ref_prob, tgt_prob)  # ok

        best_ref, best_tgt = self.find_best_plan(ref, tgt)  # ok
        # self.output_raw_data(best_ref, best_tgt)
        self.calculate_deviation(best_ref, best_tgt)

        # visualize
        self.extract_positions(best_ref, best_tgt)
        # self.draw_xy_chart()
        return self.arr_dict

    def draw_prob_distribution(self, ref_prob, tgt_prob):
        labels = ['ref', 'tgt']
        box_list = [ref_prob.reshape(-1), tgt_prob.reshape(-1)]
        plt.boxplot(box_list, labels=labels)
        plt.savefig(f'{self.save_path}/prob_distribution.jpg', dpi=192)
        plt.cla()
        plt.close()

    def calculate_prob_deviation(self, ref, tgt):
        dev = rms_ratio(ref, tgt, do_sigmoid=True, distance='l1')
        box_list = []
        labels = []
        # for i in range(dev.shape[0]):
        box_list.append(dev.reshape(-1))
        labels.append('dev')
        plt.figure(figsize=(19.2, 10.8), dpi=100)
        plt.boxplot(box_list, labels=labels)
        plt.savefig(f'{self.save_path}/prob_dev.jpg')
        plt.cla()
        plt.close()

    def find_best_plan(self, ref, tgt):
        prob_position = [990 + i * 991 for i in range(self.plan_count)]
        ref_prob_best = np.argmax(ref[:, prob_position], axis=1)
        tgt_prob_best = np.argmax(tgt[:, prob_position], axis=1)
        best_ref = []
        best_tgt = []
        for i, (no_ref, no_tgt) in enumerate(zip(ref_prob_best, tgt_prob_best)):
            best_ref.append(ref[[i], no_ref * 991:(no_ref + 1) * 991 - 1])  # no need prob
            best_tgt.append(tgt[[i], no_tgt * 991:(no_tgt + 1) * 991 - 1])
        best_ref = np.concatenate(best_ref, axis=0)
        best_tgt = np.concatenate(best_tgt, axis=0)
        return best_ref, best_tgt

    def output_raw_data(self, best_ref, best_tgt):
        # [SEQ 990]
        value_type = ['mean', 'std']
        # s 2 33 5 3
        s, v, p, e, d = self.seq, len(value_type), self.pred_pts, len(self.element_category), self.pt_dim

        # handle mean and std
        sz = self.pred_pts * self.pt_dim * len(self.element_category)  # 495
        for v_t, i in zip(value_type, range(len(value_type))):
            ref_slice = best_ref[:, i * sz:(i + 1) * sz].reshape(s, p, e, d)
            tgt_slice = best_tgt[:, i * sz:(i + 1) * sz].reshape(s, p, e, d)
            for no, cat in enumerate(self.element_category):
                # cat = ['position', 'velocity', 'acceleration', 'rotation', 'rotation_rate']
                ref_no_slice = ref_slice[:, :8, no, :]
                tgt_no_slice = tgt_slice[:, :8, no, :]
                with open(f"{no}.{cat}_{v_t}.csv", 'w') as f:
                    title = 'seq_no'
                    for pt_no in range(ref_no_slice.shape[1]):
                        title += f',pt-{pt_no}-x,pt-{pt_no}-y,pt-{pt_no}-z'
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

    def calculate_deviation(self, best_ref, best_tgt):
        # [SEQ 990]
        value_type = ['mean', 'std']
        # s 2 33 5 3
        s, v, p, e, d = self.seq, len(value_type), self.pred_pts, len(self.element_category), self.pt_dim

        sz = self.pred_pts * self.pt_dim * len(self.element_category)  # 495
        for v_t, i in zip(value_type, range(len(value_type))):
            ref_slice = best_ref[:, i * sz:(i + 1) * sz].reshape(s, p, e, d)
            tgt_slice = best_tgt[:, i * sz:(i + 1) * sz].reshape(s, p, e, d)
            for no, cat in enumerate(self.element_category):
                # with open(f"{no}.csv", 'w') as f:
                dev_slice = rms(ref_slice[:, :, no, :], tgt_slice[:, :, no, :], normalize=self.normalize)
                # if cat == 'position' and i==0:
                #     # print(dev_slice[3,:])
                #     print(np.concatenate([ref_slice[3, :, no, :], tgt_slice[3, :, no, :]], axis=1))
                #     break
                #     print(ref_slice[3, :, no, :].shape, tgt_slice[3, :, no, :].shape)
                # indices = np.where(dev_slice > 0.2)
                # values = dev_slice[indices]
                # combined = np.vstack((indices[0], indices[1], values)).T
                # sorted_combined = combined[combined[:, 2].argsort()[::-1]]
                # with open(f"{self.save_path}/{cat}_{v_t}_diff.csv", 'w') as f:
                #     for row in range(combined.shape[0]):
                #         for i, item in enumerate(combined[row]):
                #             f.write(str(item))
                #             if i != 2:
                #                 f.write(',')
                #             else:
                #                 f.write('\n')
                dev_slice = dev_slice.reshape(-1, self.pred_pts)
                checkpoint = self.pred_pts // 4
                # visualize
                draw_boxplot(dev_slice, 4, checkpoint, f"{self.save_path}/{cat}_{v_t}_diff")

    def extract_positions(self, best_ref, best_tgt):
        best_ref = best_ref.reshape(self.seq, 2, self.pred_pts, len(self.element_category), self.pt_dim)
        best_tgt = best_tgt.reshape(self.seq, 2, self.pred_pts, len(self.element_category), self.pt_dim)
        for i, key in enumerate(self.arr_dict):
            self.arr_dict[key].append(best_ref[:, 0, :, i, :])
            self.arr_dict[key].append(best_tgt[:, 0, :, i, :])
        # for key in self.arr_dict:
        #     print(f"{key} : {len(self.arr_dict[key])}")
        #     print(self.arr_dict[key][0].shape)

    def draw_xy_chart(self):
        if not os.path.exists(f'{self.save_path}/track/'):
            os.mkdir(f'{self.save_path}/track/')
        for seq in range(self.seq):
            for category in self.arr_dict:
                if not os.path.exists(f'{self.save_path}/track/{category}'):
                    os.mkdir(f'{self.save_path}/track/{category}')

                img = plt.imread(f"{self.root_path}/input_imgs/{seq:04}.jpg")
                fig, ax = plt.subplots()
                fig.autolayout = True
                ax.imshow(img, extent=[-150, 150, -10, 290])

                for color, alpha, pts in zip(self.color_list, self.alpha_list, self.arr_dict[category]):
                    ax.plot(pts[seq, :, 1], pts[seq, :, 0], '-o', c=color, markersize=2, alpha=alpha)

                plt.savefig(f"{self.save_path}/track/{category}/{seq:04}.jpg", dpi=256)
                # plt.plot(pts[seq, :, 1], pts[seq, :, 0], '-o', c=color, alpha=alpha)
                # plt.savefig(f"{self.save_path}/track/{category}/{seq:04}.jpg", dpi=192)
                plt.cla()
                plt.close('all')


class Plans(PlansBase):
    def __init__(self, root_path, normalize=True):
        super().__init__()
        self.name = 'plans'
        self.root_path = root_path
        self.save_path = f'{root_path}/{self.name}'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.pred_pts = 33
        self.pt_dim = 3
        self.plan_count = 5
        self.element_category = ['position', 'velocity', 'acceleration', 'rotation', 'rotation_rate']
        self.normalize = normalize
        self.fix_box_max_value = 0.8
        self.prob = [990 + i * 991 for i in range(5)]
        self.arr_dict = dict()
        for key in self.element_category:
            self.arr_dict[key] = list()
