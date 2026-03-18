
import sys
import mlx.core as mx
import mlx_lm.lora

sys.argv = ['mlx_lm.lora', '--config', 'results/task1/config_r8.yaml']
try:
    mlx_lm.lora.main()
except Exception as e:
    print(f"\n__ERROR__={e}")

if hasattr(mx.metal, 'get_peak_memory'):
    peak_mb = mx.metal.get_peak_memory() / (1024 * 1024)
    print(f"\n__PEAK_MEM_MB__={peak_mb:.2f}")
