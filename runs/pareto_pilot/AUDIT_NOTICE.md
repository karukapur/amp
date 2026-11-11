The original 75.2% at 6.94× storage compression is historical and backend-dependent.

The audit found that PyTorch 2.4.1 MPS rounds quantization half-ties differently from CPU. After making ties-to-even explicit, the original policy scores 74.4% on MPS and 74.2% on CPU. Repeating the complete search selects a different policy at the same storage size, scoring 73.4% on MPS versus 73.0% FP32. No statistically established accuracy improvement was found.

The 6.94× parameter-storage estimate was independently verified by actual packing and reloading. It is not a measured runtime memory or speed improvement; inference is simulated in FP32.

See [the audit report](../pareto_audit/REPORT.md) and [corrected frontier](../pareto_audit/corrected_mps_batch8/frontier.csv). Original pilot files are preserved.
