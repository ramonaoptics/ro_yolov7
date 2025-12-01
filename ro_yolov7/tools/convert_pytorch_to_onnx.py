import argparse
from pathlib import Path

import onnx
import torch

from ro_yolov7.tools.convert_from_yolov7 import convert_from_yolov7

# from onnxconverter_common import float16


def convert_pytorch_to_onnx(pytorch_model_path, height, width, channels=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 20251118 - John - Changed from float16 to float32 due to onnxconverter-common
    # 1.15+ compatibility issues. The float16 conversion with onnxconverter-common
    # 1.15+ creates type mismatches that ONNX Runtime cannot load. Float32 models
    # work correctly and the performance difference is acceptable.

    pytorch_model_path = Path(pytorch_model_path)
    ro_pytorch_model_path = pytorch_model_path.parent / f"ro_{pytorch_model_path.name}"
    # convert from yolov7 net format to ro_yolov7 format
    convert_from_yolov7(pytorch_model_path)

    checkpoint = torch.load(ro_pytorch_model_path, map_location=device, weights_only=False)
    # ensure we target the model here as we have loaded a checkpoint
    model = checkpoint["model"]

    model = model.float().eval().to(device)

    dummy_input = torch.randn((1, int(channels), int(height), int(width))).to(device)

    onnx_model_path = Path(pytorch_model_path).with_suffix(".onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,  # Use legacy exporter to avoid tensor subclass issues with opset 18
    )

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    print("ONNX model is valid.")

    # Note: float16 conversion is disabled due to onnxconverter-common 1.15+
    # compatibility issues. Models are now exported in float32 format for
    # maximum compatibility
    # if data_type == "float16":
    #     onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)

    return onnx_model


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLOv7 pytorch model to an exported onnx format'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input PyTorch model file (.pt)'
    )
    parser.add_argument(
        '--resize_height',
        type=str,
        help='Resize height of image data during training and inference'
    )
    parser.add_argument(
        '--resize_width',
        type=str,
        help='Resize width of image data during training and inference'
    )
    parser.add_argument(
        '--channels',
        default=1,
        type=str,
        help='The number of color channels in the images that this model processes'
    )
    args = parser.parse_args()
    convert_pytorch_to_onnx(
        args.input_file,
        args.resize_height,
        args.resize_width,
        args.channels,
    )


if __name__ == '__main__':
    main()
