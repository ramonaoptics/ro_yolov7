from pathlib import Path
import shutil
import subprocess
import tempfile
import onnx

import ro_yolov7


def test_convert_pytorch_to_onnx():
    pytorch_model_path = Path(ro_yolov7.__file__).parent / "yolov7-tiny.pt"
    height, width = (512, 512)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input_path = Path(tmpdir) / pytorch_model_path.name
        shutil.copy(pytorch_model_path, tmp_input_path)

        onnx_model_path = tmp_input_path.with_suffix(".onnx")

        cmd = [
            "ro_yolov7_convert_pytorch_to_onnx",
            str(tmp_input_path),
            "--resize_height", str(height),
            "--resize_width", str(width),
            "--channels", "3"  # we use three channels here for the base yolov7 tiny checkpoint
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode == 0, \
            f"Conversion failed with return code {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"

        assert onnx_model_path.exists(), "ONNX output file was not created"

        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
