# This provides the tools to rename models as ro_yolov7.models
# so that they can be loaded by torch.load
# without specifying a particular picklemodule

# Load the entire pickle module to mock the base pickemodule
from pickle import *   # noqa
from pickle import Unpickler


class Unpickler(Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, mod_name, name):
        if mod_name.startswith("models.yolo"):
            mod_name = mod_name.replace(
                "models.yolo",
                "ro_yolov7.models.yolo",
            )
        elif mod_name.startswith("models.common"):
            mod_name = mod_name.replace(
                "models.common",
                "ro_yolov7.models.common",
            )
        return super().find_class(mod_name, name)
