import argparse
from pathlib import Path

import ro_yolov7.tools.unpickler


def convert_from_yolov7(input_file, output=None):
    """Convert models pretrained from YOLOv7 to be loadable by torch.
    The problem is that the models assume that yolov7 is in the path.
    This isn't always going to be the case when the models are installed
    in a larger conda environment.
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output is None:
        output = input_file.parent / f"ro_{input_file.name}"
    else:
        output = Path(output)

    print(f"Loading model from {input_file}...")

    import torch
    with open(input_file, 'rb') as f:
        model = torch.load(
            f,
            map_location=torch.device('cpu'),
            weights_only=False,
            pickle_module=ro_yolov7.tools.unpickler,
        )

    print(f"Saving model to {output}...")
    torch.save(model, output)

    print(f"Conversion complete: {output}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLOv7 models to be loadable with ro_yolov7 package'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input PyTorch model file (.pt)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output file (default: input filename with _ro suffix)'
    )
    args = parser.parse_args()
    convert_from_yolov7(args.input_file, args.output)


if __name__ == '__main__':
    main()
