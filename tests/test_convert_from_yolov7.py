from pathlib import Path
import tempfile
import subprocess
import torch

import ro_yolov7


def test_convert_from_yolov7():
    input_file = Path(ro_yolov7.__file__).parent / "yolov7-tiny.pt"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "ro_yolov7-tiny.pt"

        cmd = [
            'ro_yolov7_convert_from_yolov7',
            str(input_file),
            '--output', str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode == 0, \
            f"Conversion failed with return code {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"

        assert output_path.exists(), "Output file was not created"

        model = torch.load(output_path, map_location='cpu', weights_only=False)
        assert model is not None, "Converted model could not be loaded"
