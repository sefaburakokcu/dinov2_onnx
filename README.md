# DINOv2 ONNX Inference

This repository contains a Python script (`onnx_inference.py`) for extracting and comparing image features using the DINOv2 model in ONNX format. The script uses the [onnxruntime](https://onnxruntime.ai/) library to load and run the model.

## Dependencies

- [ONNXRuntime](https://onnxruntime.ai/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

## Models

Models can be obtained by exporting [PyTorch models](https://github.com/facebookresearch/dinov2) to ONNX models.
To export models to ONNX, you can utilize the [onnx_export_merge](https://github.com/sefaburakokcu/dinov2) branch.

In addition, you can download pre-exported models from:

<u>Google Drive</u>
* [dinov2_vits14](https://drive.google.com/file/d/1U1rB42VPEKJ_IBp2VllVoHs501tN4tbW/view?usp=drive_link)
* [dinov2_vitb14](https://drive.google.com/file/d/1vpMx_CIKNtiDE9vaAdXfSOlcIJuxPaCo/view?usp=drive_link)

<u>Hugging Face</u>
* [dinov2_vits14](https://huggingface.co/sefaburak/dinov2-small-onnx)

## Usage

1. Clone the repository:

   ```bash
   git clone git@github.com:sefaburakokcu/dinov2_onnx.git
   cd dinov2_onnx
   ```

2. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Make sure you have Python and pip installed.

3. Place your input images in the `inputs` directory.

4. Run the inference script:

   ```bash
   python onnx_inference.py
   ```

   Optional arguments:

   - `--onnx_model`: Path to the ONNX model file (default: `./dinov2_vits14.onnx`).
   - `--image_folder`: Path to the folder containing input images (default: `./inputs/`).

## License

This project is licensed under the [MIT License](LICENSE).