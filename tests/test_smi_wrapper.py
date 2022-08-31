# from unittest import TestCase

# import pint

# import src.smi_wrapper

# from src.smi_wrapper import SMIWrapper


# class TestGPU(TestCase):

#     def setUp(self) -> None:
#         self.ureg = src.smi_wrapper.ureg
#         self.smi_wrapper = SMIWrapper()
#         self.smi_wrapper.open()

#     def test_get_performance_state(self):
#         for gpu in self.smi_wrapper.gpus:
#             state = gpu.get_performance_state()
#             print(f"[GPU {gpu.index}] PerformanceState: {state}")
#             self.assertEqual(int, type(state), "Expected state to be of type int")

#     def tearDown(self) -> None:
#         self.smi_wrapper.close()

#     def test_get_power_usage(self):
#         for gpu in self.smi_wrapper.gpus:
#             power = gpu.get_power_usage()
#             print(f"[GPU {gpu.index}] PerformanceState: {power}")
#             self.assertEqual(int, type(power.magnitude), "Expected state to be of type int")
#             self.assertEqual(self.ureg.milliwatts, power.units, "Expected unit to be mW")
