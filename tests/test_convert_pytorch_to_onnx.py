from pathlib import Path
import subprocess
import onnx


def test_convert_pytorch_to_onnx():
    pytorch_model_path = Path(__file__).parent / "test_pytorch_model.pt"
    height, width = (512, 512)

    result = subprocess.run([
        "ro_yolov7_convert_pytorch_to_onnx",
        str(pytorch_model_path),
        "--resize_height", str(height),
        "--resize_width", str(width),
    ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, \
        f"Conversion failed with return code {result.returncode}\nStderr: {result.stderr}"

    onnx_model_path = pytorch_model_path.with_suffix(".onnx")
    assert onnx_model_path.exists()

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
