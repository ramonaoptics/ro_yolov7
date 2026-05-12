from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

requirements = [
    "matplotlib",
    "numpy>=2.0",
    "onnx",
    "onnxscript",
    "opencv-python",
    "Pillow",
    "PyYAML",
    "requests",
    "scipy",
    "tifffile",
    "torch",
    "torchvision",
    "tqdm",
]


setup(
    name="ro-yolov7",
    version="0.5.0",
    author="Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao, John Efromon, Mark Harfouche",
    description="YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WongKinYiu/yolov7",
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        "ro_yolov7": [
            "*.pt",
            "cfg/**/*.yaml",
            "data/**/*.yaml",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "ro_yolov7_train=ro_yolov7.train:main",
            "ro_yolov7_convert_from_yolov7=ro_yolov7.tools.convert_from_yolov7:main",
            "ro_yolov7_convert_pytorch_to_onnx=ro_yolov7.tools.convert_pytorch_to_onnx:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
