#include <torch/extension.h>
#include "tv3d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tv3d_forward", &tv3d_forward, "3D total variation forward pass");
  m.def("tv3d_backward", &tv3d_backward, "3D total variation backward pass");
}
