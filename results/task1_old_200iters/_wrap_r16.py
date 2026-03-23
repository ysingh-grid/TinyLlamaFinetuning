
import sys, mlx.core as mx, mlx_lm.lora as lm_lora
sys.argv = ['mlx_lm.lora', '--config', 'results/task1/config_r16.yaml']
try:
    lm_lora.main()
except SystemExit:
    pass
if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_peak_memory'):
    peak = mx.metal.get_peak_memory() / (1024*1024)
    print(f'__PEAK_MB__={peak:.2f}')
else:
    print('__PEAK_MB__=0')
