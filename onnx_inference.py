import os.path

import cv2
import glob
import tqdm
import argparse

import numpy as np
import onnxruntime as rt


def load_onnx_model(onnx_model_path, providers=None):
    """
    Load the ONNX model using onnxruntime.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
        providers (list): List of execution providers. Defaults to None.

    Returns:
        onnxruntime.InferenceSession: ONNX runtime session.
    """
    ort_session = rt.InferenceSession(onnx_model_path, providers=providers)
    return ort_session


def preprocess_image(image, input_size=(224, 224), mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    """
    Preprocess the input image.

    Args:
        image (numpy.ndarray): Input image.
        input_size (tuple): Target input size. Defaults to (224, 224).
        mean (tuple): Mean values for normalization. Defaults to (123.675, 116.28, 103.53).
        std (tuple): Standard deviation values for normalization. Defaults to (58.395, 57.12, 57.375).

    Returns:
        numpy.ndarray: Preprocessed input data.
    """
    input_data = cv2.resize(image, input_size)
    input_data = (input_data - np.array(mean)) / np.array(std)
    input_data = input_data.astype(np.float32)
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def compare_features(embedding_1, embedding_2):
    """
    Compare two embeddings using cosine similarity.

    Args:
        embedding_1 (numpy.ndarray): First embedding.
        embedding_2 (numpy.ndarray): Second embedding.

    Returns:
        float: Cosine similarity score.
    """
    embedding_1 = embedding_1 / np.linalg.norm(embedding_1)
    embedding_2 = embedding_2 / np.linalg.norm(embedding_2)
    return (embedding_1 @ embedding_2.T)[0][0]


def main(args):
    """
    Main function to process images and compare features.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    ort_session = load_onnx_model(args.onnx_model, providers=["CPUExecutionProvider", "CPUExecutionProvider"])

    image_paths = glob.glob(args.image_folder + "*.jpeg")

    embedding_list = []
    for image_path in tqdm.tqdm(image_paths, total=len(image_paths), desc="Processing images..."):
        image = cv2.imread(image_path)
        input_data = preprocess_image(image)
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outputs = ort_session.run(None, ort_inputs)[0]
        embedding_list.append(ort_outputs)

    num_embeddings = len(embedding_list)
    for probe_id in range(num_embeddings):
        for gallery_id in range(probe_id + 1, num_embeddings):
            similarity = compare_features(embedding_list[probe_id], embedding_list[gallery_id])
            print(f"Similarity of {os.path.basename(image_paths[probe_id])} and "
                  f"{os.path.basename(image_paths[gallery_id])}: {similarity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image feature comparison using an ONNX model.")
    parser.add_argument("--onnx_model", type=str, default="./dinov2_vits14.onnx", help="Path to the ONNX model.")
    parser.add_argument("--image_folder", type=str, default="./inputs/", help="Path to the folder containing images.")
    args = parser.parse_args()

    main(args)
