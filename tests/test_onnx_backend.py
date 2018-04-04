# Copyright Vertex.AI.

import json
import os
import unittest

import onnx
import onnx.backend.test
import onnx_plaidml.backend as opb

import plaidml

# Register plugins
pytest_plugins = 'onnx.backend.test.report',


class BackendTest(onnx.backend.test.BackendTest):

    def __init__(self, backend, name):
        super(BackendTest, self).__init__(backend, name)

        # Has issues on osx tests; temporarily disabling.
        self.exclude('test_slice_start_out_of_bounds_opencl_cpu.0')

        # Unimplemented functionality
        self.exclude('test_ReflectionPad2d_')  # Requires Pad(reflect)
        self.exclude('test_ReplicationPad2d_')  # Requires Pad(edge)
        self.exclude('test_edge_pad_')  # Requires Pad(edge)
        self.exclude('test_reflect_pad_')  # Requires Pad(reflect)
        self.exclude('test_gather_1_')  # Requires Gather on non-outermost axis
        self.exclude('test_hardmax_one_hot_')  # Requires filtered Hardmax
        self.exclude('test_top_k_')  # Requires TopK
        self.exclude('test_Upsample_nearest_scale_2d_')  # Requires Upsample

        # Needs to be debugged
        self.exclude('test_GLU_opencl_cpu.0')
        self.exclude('test_GLU_dim_opencl_cpu.0')
        self.exclude('test_Softplus_opencl_cpu.0')
        self.exclude('test_Softmin_opencl_cpu.0')
        self.exclude('test_bvlc_alexnet_opencl_cpu.0')
        self.exclude('test_densenet121_opencl_cpu.0')
        self.exclude('test_shufflenet_opencl_cpu.0')
        self.exclude('test_squeezenet_opencl_cpu.0')
        self.exclude('test_operator_chunk_opencl_cpu.0')
        self.exclude('test_operator_transpose_')

        # These work, but they're slow, and they don't work if they're all together --
        # likely due to holding onto temporary allocations on the GPU.
        self.exclude('test_resnet50_')
        self.exclude('test_inception_v1_')
        self.exclude('test_inception_v2_')
        self.exclude('test_vgg16_')
        self.exclude('test_vgg19_')

    def _add_test(self, category, test_name, test_func, report_item, devices=None):
        if not devices:
            devices = tuple(opb.PlaidMLBackend.device_configs.keys())
        return super(BackendTest, self)._add_test(category, test_name, test_func, report_item,
                                                  devices)


# Import test cases for unittest
globals().update(BackendTest(opb, __name__).enable_report().test_cases)

if __name__ == '__main__':
    unittest.main()
