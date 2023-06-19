import numpy as np
from tmp_utils import get_config
from analyze_tool.lines import LaneLines, RoadEdges
from analyze_tool.leads import Leads
from analyze_tool.plans import Plans
from analyze_tool.tool_utils import draw_tracking_chart


CFG = get_config("./config/config.yaml")

if CFG['compare']['pc2pc']:
    out_name1, out_name2 = CFG['compare']['pc2pc_out_name']
    data = CFG['data_path1'].split('/')[-1]
    outputs1 = np.load(f"{CFG['out_root']}/{data}/{out_name1}.npy")
    outputs2 = np.load(f"{CFG['out_root']}/{data}/{out_name2}.npy")
elif CFG['compare']['device2pc']:
    out_name1, out_name2 = CFG['compare']['device2pc_out_name']
    data = CFG['data_path1'].split('/')[-1]
    outputs1 = np.load(f"{CFG['out_root']}/{data}/{out_name1}.npy")
    outputs2 = np.load(f"{CFG['out_root']}/{data}/{out_name2}.npy")
else:
    raise FileNotFoundError("no output file exists")
img_path = f"{CFG['out_root']}/{data}/big_input_imgs"
track_path = f"{CFG['out_root']}/{data}/track"


# 0~ 4955(0+4955)
plan = Plans(f"{CFG['out_root']}/{data}", normalize=False)
plan_track_dict = plan.model_match_test(outputs1[:, :4955], outputs2[:, :4955])

# 4955~5491(4955+536)
lane_lines = LaneLines(f"{CFG['out_root']}/{data}", normalize=False)
lane_lines_track_dict = lane_lines.model_match_test(outputs1[:, 4955:5491], outputs2[:, 4955:5491])

# 5491~5755(5491+264)
road_edges = RoadEdges(f"{CFG['out_root']}/{data}", normalize=False)
road_edges_track_dict = road_edges.model_match_test(outputs1[:, 5491:5755], outputs2[:, 5491:5755])

# 5755~5860(5755+105)
leads = Leads(f"{CFG['out_root']}/{data}", normalize=False)
leads.model_match_test(outputs1[:, 5755:5860], outputs2[:, 5755:5860])

# positions = plan_track_dict['position']
#
# lane_lines_ref_list = []
# lane_lines_tgt_list = []
# for key in lane_lines_track_dict:
#     ref, tgt = lane_lines_track_dict[key]
#     lane_lines_ref_list.append(ref)
#     lane_lines_tgt_list.append(tgt)
# lane_lines = [np.stack(lane_lines_ref_list, axis=0), np.stack(lane_lines_tgt_list, axis=0)]
#
# road_edges_ref_list = []
# road_edges_tgt_list = []
# for key in road_edges_track_dict:
#     ref, tgt = road_edges_track_dict[key]
#     road_edges_ref_list.append(ref)
#     road_edges_tgt_list.append(tgt)
# road_edges = [np.stack(road_edges_ref_list, axis=0), np.stack(road_edges_tgt_list, axis=0)]
# draw_tracking_chart(img_path, track_path, positions, lane_lines, road_edges)
#
