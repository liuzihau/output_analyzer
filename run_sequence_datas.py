import os
import numpy as np
from tmp_utils import (
    get_config,
    get_data_shape,
    get_file_length,
    set_out_folder,
    load_onnx_model,
    read_packaged_input_data,
    read_file_to_inputs,
    split_input_data,
    save_ref_image,
    concatenate_and_save
)


def get_output():
    onnx_outputs = []
    thneed_outputs = []
    for i, data in enumerate(datas):
        path = f"{CFG['data_path1']}/{data}"
        files = os.listdir(path)
        onnx_inputs, thneed_output = read_file_to_inputs(path, files, DATA_SHAPE, out_folder, i)
        if thneed_output is not None:
            onnx_output = onnx_model.run(model_output_names, onnx_inputs)[0]
            onnx_outputs.append(onnx_output.astype(np.float64))
            thneed_outputs.append(thneed_output.reshape(1, -1).astype(np.float64))
    return onnx_outputs, thneed_outputs


def get_test_output():
    """
    unpacked, image is modified by other tools
    """
    onnx_outputs1 = []
    onnx_outputs2 = []
    for i, (data1, data2) in enumerate(zip(datas, datas2)):
        path1 = f"{CFG['data_path1']}/{data1}"
        files1 = os.listdir(path1)
        onnx_inputs1, _ = read_file_to_inputs(path1, files1, DATA_SHAPE, out_folder, i)
        onnx_output1 = onnx_model.run(model_output_names, onnx_inputs1)[0]

        path2 = f"{CFG['data_path2']}/{data2}"
        files2 = os.listdir(path2)
        onnx_inputs2, _ = read_file_to_inputs(path2, files2, DATA_SHAPE, out_folder, i)
        for key in onnx_inputs1:
            if key not in onnx_inputs2:
                onnx_inputs2[key] = onnx_inputs1[key]
        onnx_output2 = onnx_model.run(model_output_names, onnx_inputs2)[0]

        onnx_outputs1.append(onnx_output1)
        onnx_outputs2.append(onnx_output2)
    return onnx_outputs1, onnx_outputs2


if __name__ == "__main__":
    CFG = get_config("./config/config.yaml")
    DATA_SHAPE = get_data_shape()
    datas, out_folder = set_out_folder(CFG['data_path1'], CFG['out_root'])
    onnx_model, model_output_names = load_onnx_model(CFG['model_path'])
    packs = get_file_length(DATA_SHAPE)

    if CFG['compare']['pc2pc']:
        datas2, _ = set_out_folder(CFG['data_path2'])
        onnx_outputs1, onnx_outputs2 = get_test_output()
        out_name1, out_name2 = CFG['compare']['pc2pc_out_name']
        concatenate_and_save(onnx_outputs1, onnx_outputs2, out_name1, out_name2, out_folder)
    elif CFG['compare']['device2pc']:
        onnx_outputs, thneed_outputs = get_output()
        out_name1, out_name2 = CFG['compare']['device2pc_out_name']
        concatenate_and_save(onnx_outputs, thneed_outputs, out_name1, out_name2, out_folder)
