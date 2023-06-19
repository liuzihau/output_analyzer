import os
import yaml
import numpy as np
import cv2
import onnx
import onnxruntime as rt


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    return opt


def get_data_shape():
    shape_dict = dict()
    shape_dict['features_buffer'] = (1, 99, 128)
    shape_dict['nav_features'] = (1, 64)
    shape_dict['traffic_convention'] = (1, 2)
    shape_dict['desire'] = (1, 100, 8)
    shape_dict['input_imgs'] = (1, 12, 128, 256)
    shape_dict['big_input_imgs'] = (1, 12, 128, 256)
    shape_dict['output'] = (6108,)
    return shape_dict


def get_file_length(shape_dict):
    fb = np.product(shape_dict['features_buffer'])
    nf = np.product(shape_dict['nav_features'])
    tc = np.product(shape_dict['traffic_convention'])
    de = np.product(shape_dict['desire'])
    ou = np.product(shape_dict['output'])
    return fb, nf, tc, de, ou


def set_out_folder(data_root: str, out_folder_name: str = './exp'):
    if not os.path.exists(f'./{out_folder_name}'):
        os.mkdir(f'{out_folder_name}')
    out_folder = f"{out_folder_name}/" + data_root.split('/')[-1]
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    datas = sorted(os.listdir(data_root), key=lambda x: int(x))
    return datas, out_folder


def load_onnx_model(model_path):
    onnx_weight = onnx.load(model_path)
    model_output_names = [node.name for node in onnx_weight.graph.output]
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    onnx_model = rt.InferenceSession(model_path, providers=providers)
    return onnx_model, model_output_names


def save_ref_image(stacked_img, out_name):
    img_list = []
    for img in [stacked_img[:6, :, :], stacked_img[6:, :, :]]:
        y = np.zeros((128 * 2, 256 * 2), np.uint8)
        y[::2, ::2] = img[0, :, :].astype(np.uint8)
        y[::2, 1::2] = img[1, :, :].astype(np.uint8)
        y[1::2, ::2] = img[2, :, :].astype(np.uint8)
        y[1::2, 1::2] = img[3, :, :].astype(np.uint8)
        v = img[4, :, :].astype(np.uint8)
        u = img[5, :, :].astype(np.uint8)

        # # Resize u and v color channels to be the same size as y
        u = cv2.resize(u, (y.shape[1], y.shape[0]))
        v = cv2.resize(v, (y.shape[1], y.shape[0]))

        yvu = cv2.merge((y, v, u))  # Stack planes to 3D matrix (use Y,V,U ordering)
        img_list.append(cv2.cvtColor(yvu, cv2.COLOR_YUV2BGR))
    final_img = np.concatenate(img_list, axis=0)
    print(out_name)
    cv2.imwrite(out_name, final_img)


def concatenate_and_save(output1, output2, name_of_output1, name_of_output2, out_folder):
    onnx_outputs = np.concatenate(output1, axis=0)
    thneed_outputs = np.concatenate(output2, axis=0)
    np.save(f"{out_folder}/{name_of_output1}.npy", onnx_outputs)
    np.save(f"{out_folder}/{name_of_output2}.npy", thneed_outputs)
    print(onnx_outputs.shape, thneed_outputs.shape)


def read_packaged_input_data(path, packs):
    fb, nf, tc, de, ou = packs
    # features_buffer, traffic_convention, desire, big_input_imgs, input_imgs, output
    img_package = np.fromfile(f'{path}/img_inputs.bin', np.float32)
    img_package = img_package.reshape(-1, 24, 128, 256)
    file_package = np.fromfile(f'{path}/files.bin', np.float32)
    file_package = file_package.reshape(-1, fb + nf + tc + de + ou)
    assert file_package.shape[0] == img_package.shape[0], "files and img sequence should be the same"
    sequences = file_package.shape[0]
    return img_package, file_package, sequences


def read_file_to_inputs(path, files, shape_dict, out_folder, idx):
    onnx_inputs = dict()
    origin_output = None
    for file in files:
        feature = np.fromfile(f'{path}/{file}', np.float32)
        expect_shape = shape_dict[file.split('.')[0]]
        feature = feature.reshape(expect_shape)
        feature = feature.astype(np.float16)
        if 'output' in file:
            origin_output = feature
        else:
            onnx_inputs[file.split('.')[0]] = feature

        if 'input_imgs' in file:
            if not os.path.exists(f'{out_folder}/{file.split(".")[0]}'):
                os.mkdir(f'{out_folder}/{file.split(".")[0]}')
            image_save_path = f'{out_folder}/{file.split(".")[0]}/{idx:04d}.jpg'
            tmp = feature.reshape(12, 128, 256)
            save_ref_image(tmp, image_save_path)
    return onnx_inputs, origin_output


def split_input_data(img_package, file_package, sequences, packs, shape_dict):
    onnx_inputs_sequences = []
    fb, nf, tc, de, ou = packs
    big_input_imgs_acc = img_package[:, :12, :, :]
    input_imgs_acc = img_package[:, 12:, :, :]
    features_buffer_acc = file_package[:, :fb]
    nav_features_acc = file_package[:, fb:fb + nf]
    traffic_convention_acc = file_package[:, fb + nf:fb + nf + tc]
    desire_acc = file_package[:, fb + nf + tc: fb + nf + tc + de]
    output_acc = file_package[:, fb + nf + tc + de:]
    for sequence in range(sequences):
        onnx_inputs = dict()
        onnx_inputs['big_input_imgs'] = big_input_imgs_acc[sequence, :, :, :].reshape(shape_dict['big_input_imgs'])
        onnx_inputs['input_imgs'] = input_imgs_acc[sequence, :, :, :].reshape(shape_dict['input_imgs'])
        onnx_inputs['features_buffer'] = features_buffer_acc[sequence, :].reshape(shape_dict['features_buffer'])
        onnx_inputs['nav_features'] = nav_features_acc[sequence, :].reshape(shape_dict['nav_features'])
        onnx_inputs['traffic_convention'] = traffic_convention_acc[sequence, :].reshape(
            shape_dict['traffic_convention'])
        onnx_inputs['desire'] = desire_acc[sequence, :].reshape(shape_dict['desire'])
        for key in onnx_inputs:
            onnx_inputs[key] = onnx_inputs[key].astype(np.float16)
        onnx_inputs_sequences.append(onnx_inputs)
    return onnx_inputs_sequences, output_acc
