# Sequential GPU experiment queue (local RTX 3060 6 GB). Every step checkpoints and resumes.
# Budgeted version (v2, 2026-11-22): workload sizes reduced to fit a 5-hour total budget; see docs/PROGRESS.md.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_gpu_queue.ps1 [-WaitPid <pid>]
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HOME = "E:\hf_cache"
$env:PYTHONUTF8 = "1"
$py = ".\.venv\Scripts\python.exe"
$log = "results\logs\gpu_queue.log"
New-Item -ItemType Directory -Force results\logs | Out-Null

function Step($name, [string[]]$cmdArgs) {
    # resume support: skip steps that already finished successfully in an earlier session
    if (Select-String -Path $log -Pattern "END $name exit=0" -SimpleMatch -Quiet -ErrorAction SilentlyContinue) {
        "$(Get-Date -Format s) SKIP $name (already completed)" | Tee-Object -FilePath $log -Append
        return
    }
    $t0 = Get-Date
    "$(Get-Date -Format s) START $name" | Tee-Object -FilePath $log -Append
    & $py @cmdArgs *>> "results\logs\$name.log"
    "$(Get-Date -Format s) END $name exit=$LASTEXITCODE min=$([math]::Round(((Get-Date)-$t0).TotalMinutes,1))" | Tee-Object -FilePath $log -Append
}

if ($WaitPid -gt 0) {
    "$(Get-Date -Format s) WAIT for pid $WaitPid (exp1_struct still running)" | Tee-Object -FilePath $log -Append
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 10 }
}
# exp1 resumes from its checkpoint if anything is missing
Step "exp1_struct"   @("experiments/run_inference.py","--exp","exp1","--name","struct_b16","--prompt","struct-v1","--policy","static","--batch","16")
Step "baselines"     @("experiments/baselines/run_baselines.py")
Step "exp1_caption"  @("experiments/run_inference.py","--exp","exp1","--name","caption_b16","--prompt","caption-v1","--subset","test","--policy","static","--batch","16")
Step "exp4_llm"      @("experiments/narrative/run_exp4.py","--phase","run")
Step "exp2_offline"  @("experiments/batching/run_exp2.py","--part","offline","--repeats","3","--n","32","--n-indep","16","--batches","1","4","8","16","32","--big-batch-check")
Step "exp3_views"    @("experiments/run_inference.py","--exp","exp3","--name","views_batched_b16","--mode","views","--subset","groups_all","--policy","static","--batch","16")
Step "exp3_context"  @("experiments/run_inference.py","--exp","exp3","--name","views_context","--mode","context","--subset","groups_all","--policy","static","--batch","2")
foreach ($c in @("blur:2","blur:5","downscale:0.25","downscale:0.1","lowlight:2.0","lowlight:3.5","occlusion:0.25","occlusion:0.5","jpeg:20","jpeg:5")) {
    $n = $c.Replace(":","_")
    Step "exp5_$n" @("experiments/run_inference.py","--exp","exp5","--name",$n,"--subset","robust","--degrade",$c,"--policy","static","--batch","16")
}
Step "exp2_online"   @("experiments/batching/run_exp2.py","--part","online","--repeats","2","--n","24","--rates","0.1","0.3")
"$(Get-Date -Format s) QUEUE DONE" | Tee-Object -FilePath $log -Append
