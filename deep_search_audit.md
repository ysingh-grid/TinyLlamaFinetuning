**Findings (severity-ordered)**

1. **Critical: your LoRA and QLoRA pipelines are effectively the same experiment.**  
   Evidence in code:
   - Both non-`full` techniques use the same model argument path by default (`./models/tinyllama-4bit-base`): [sweep_mlx_lora.py:69](/Users/ysingh/PyCharmMiscProject/sweep_mlx_lora.py:69), [sweep_mlx_lora.py:153](/Users/ysingh/PyCharmMiscProject/sweep_mlx_lora.py:153)  
   - Both non-`full` techniques set `fine_tune_type: "lora"` with same LoRA config block: [sweep_mlx_lora.py:317](/Users/ysingh/PyCharmMiscProject/sweep_mlx_lora.py:317)  
   Evidence in artifacts:
   - `lora` and `qlora` adapter configs are functionally identical (same base model, rank, lr, etc.): [adapter_config.json](/Users/ysingh/PyCharmMiscProject/mlx_best_models/lora_r32_lr1e-4_ga6/adapter_config.json:1), [adapter_config.json](/Users/ysingh/PyCharmMiscProject/mlx_best_models/qlora_r32_lr1e-4_ga6/adapter_config.json:1)  
   - I verified identical SHA256 for both adapter weight files and identical generated response text across 500/500 prompts in run `20260223_143739`.

2. **Critical: eval base-model wiring is inconsistent with how LoRA/QLoRA were trained.**  
   - Eval loads `lora_ft`/`qlora_ft` adapters onto hub model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`: [evaluation/models.json:8](/Users/ysingh/PyCharmMiscProject/evaluation/models.json:8)  
   - The stored adapter metadata says those adapters were trained against `./models/tinyllama-4bit-base`: [adapter_config.json:26](/Users/ysingh/PyCharmMiscProject/mlx_best_models/lora_r32_lr1e-4_ga6/adapter_config.json:26), [adapter_config.json:26](/Users/ysingh/PyCharmMiscProject/mlx_best_models/qlora_r32_lr1e-4_ga6/adapter_config.json:26)  
   This can degrade quality because deltas are applied to a different base regime than training.

3. **High: full retrain/smoke scripts have path contract mismatch (can gate the wrong artifact).**  
   - Retrain configs write to `./mlx_best_models/full`: [full_safe_sanity.yaml:12](/Users/ysingh/PyCharmMiscProject/experiments/full_safe_sanity.yaml:12), [full_safe_scale.yaml:12](/Users/ysingh/PyCharmMiscProject/experiments/full_safe_scale.yaml:12), [full_safe_epoch1.yaml:12](/Users/ysingh/PyCharmMiscProject/experiments/full_safe_epoch1.yaml:12)  
   - Retrain runner prints output as `full_retrain`: [run_full_retrain_safe.sh:30](/Users/ysingh/PyCharmMiscProject/run_full_retrain_safe.sh:30)  
   - Smoke eval expects `./mlx_best_models/full_retrain`: [run_full_smoke_eval.sh:18](/Users/ysingh/PyCharmMiscProject/run_full_smoke_eval.sh:18), [run_full_smoke_eval.sh:47](/Users/ysingh/PyCharmMiscProject/run_full_smoke_eval.sh:47)

4. **High: CI reporting is misleading vs displayed metrics.**  
   - Bootstrap CI is computed on a tie-adjusted score (`win=1, tie=0.5, loss=0`): [pipeline.py:600](/Users/ysingh/PyCharmMiscProject/evaluation/pipeline.py:600), [pipeline.py:636](/Users/ysingh/PyCharmMiscProject/evaluation/pipeline.py:636)  
   - Report table labels it as generic `95% CI` next to `Win Rate` and `Effective Win Rate`: [pipeline.py:782](/Users/ysingh/PyCharmMiscProject/evaluation/pipeline.py:782)  
   This is why CI bounds can appear incompatible with the raw win-rate column.

5. **High: your “best model” sweep selection is fragile.**  
   - Loss parser takes the last regex match in logs, not a strongly defined validation objective: [sweep_mlx_lora.py:159](/Users/ysingh/PyCharmMiscProject/sweep_mlx_lora.py:159)  
   - Sweep config evaluates/saves only once per epoch in many cases (`steps_per_eval = steps_per_epoch`, `save_every = steps_per_epoch`): [sweep_mlx_lora.py:305](/Users/ysingh/PyCharmMiscProject/sweep_mlx_lora.py:305), [sweep_mlx_lora.py:306](/Users/ysingh/PyCharmMiscProject/sweep_mlx_lora.py:306)  
   This weakens checkpoint/model selection.

6. **Medium: prompt masking is off in trained adapters (`mask_prompt: false`).**  
   - Present in both LoRA and QLoRA adapter configs: [adapter_config.json:24](/Users/ysingh/PyCharmMiscProject/mlx_best_models/lora_r32_lr1e-4_ga6/adapter_config.json:24), [adapter_config.json:24](/Users/ysingh/PyCharmMiscProject/mlx_best_models/qlora_r32_lr1e-4_ga6/adapter_config.json:24)  
   For instruction tuning, this often hurts response quality (model learns to predict prompt tokens too).

7. **Medium: eval/judge setup has bias/robustness risks.**  
   - LM Studio judge parser uses substring checks (`"left" in text` before `"right"`), which can misparse verbose outputs containing both words: [judge_with_lmstudio.py:36](/Users/ysingh/PyCharmMiscProject/evaluation/judge_with_lmstudio.py:36)  
   - `run_scoring` silently drops judgments with missing `item_id` matches (no coverage assert): [pipeline.py:667](/Users/ysingh/PyCharmMiscProject/evaluation/pipeline.py:667)

8. **Medium: generation/eval defaults can mislead comparisons.**  
   - Eval wrapper defaults to only 50 prompts: [run_eval_6models.sh:12](/Users/ysingh/PyCharmMiscProject/evaluation/run_eval_6models.sh:12)  
   - Generation reseeds RNG on every prompt (`mx.random.seed(seed)`), which is unusual and can reduce realistic sampling variation: [pipeline.py:209](/Users/ysingh/PyCharmMiscProject/evaluation/pipeline.py:209)

9. **Low but important hygiene: training entrypoints use system `python3`, not `.venv`.**  
   - [run_train.sh:13](/Users/ysingh/PyCharmMiscProject/run_train.sh:13), [run_qlora.sh:13](/Users/ysingh/PyCharmMiscProject/run_qlora.sh:13), [run_full.sh:14](/Users/ysingh/PyCharmMiscProject/run_full.sh:14)

10. **Data reality mismatch:** prep script says 10k examples, current dataset is 5k total (4000/500/500).  
   - Script expectation: [prepare_dataset.py:10](/Users/ysingh/PyCharmMiscProject/prepare_dataset.py:10)  
   - Observed on disk: `wc -l` showed `4000/500/500`.

---

**What most likely explains the reversed ranking**
1. LoRA vs QLoRA is not a true comparison in this repo state (same effective training path + same resulting weights).  
2. Adapters are likely evaluated on mismatched base setup vs training.  
3. Instruction-tuning objective/settings (`mask_prompt: false`, aggressive LR combos, weak checkpoint selection) likely degraded base chat behavior.  
4. Single-model judge plus response-length preference amplified the gap (inference from run stats: FT outputs were much shorter on average than baselines).

---

**What is already correct**
- Pairing side balance is clean: `250/250` left-right for each pair in [pairing_summary.json](/Users/ysingh/PyCharmMiscProject/evaluation/runs/20260223_143739/pairing/pairing_summary.json:1).  
So this is not a left/right leakage bug.

---

**Immediate fix order**
1. Split LoRA and QLoRA into truly distinct training definitions.  
2. Align eval base model with training base for each adapter (or retrain with intended eval base).  
3. Fix `full` vs `full_retrain` path contract across retrain/smoke scripts.  
4. Turn on `mask_prompt: true` for instruction SFT runs and rerun.  
5. Fix CI labeling/columns in scoring report.  
6. Add automated guard: fail if `lora` and `qlora` adapters/checksum or response-text overlap is ~100%.

If you want, I can implement these fixes directly in code and add a validation checklist script so this cannot regress again.
