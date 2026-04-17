#!/usr/bin/env python3
"""Build YOLOv8n TensorRT engine - final version."""
import subprocess, sys, time
from pathlib import Path

onnx_path = str(Path(__file__).resolve().parent / "models" / "yolov8n.onnx")
engine_path = onnx_path.replace(".onnx", "_fp16.engine")
trtexec = "/usr/src/tensorrt/bin/trtexec"

print(f"Building YOLOv8n TensorRT engine (static model, no shape hints)...")
print(f"ONNX: {onnx_path}")
print(f"Engine: {engine_path}")

# Restore backup if exists
bak = engine_path + ".bak"
if Path(bak).exists():
    import shutil
    shutil.copy(bak, engine_path)
    print(f"Restored backup to {engine_path}")
    # Check if existing engine works
    size = Path(engine_path).stat().st_size / 1024 / 1024
    print(f"Existing engine: {size:.1f} MB")
    print("\nIf this engine causes CuTensor errors at runtime, the detector will auto-fallback to DNN.")
    print("You can run the system now:")
    print("  bash run_demo.sh --no-window")
    sys.exit(0)

# Only build if no backup exists
cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16"]
print(f"\nRunning: {' '.join(cmd)}")
print("(This may take several minutes for a static model with full optimization...)\n")

start = time.time()
try:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    last_lines = []
    for line in iter(proc.stdout.readline, ''):
        line = line.rstrip('\n')
        last_lines.append(line)
        # Print progress lines
        low = line.lower()
        if any(k in low for k in ["parsing", "building", "optimizing", "layer",
                                   "tactic", "timing", "memory", "error", "fail"]):
            elapsed = time.time() - start
            print(f"[{elapsed:.0f}s] {line}")
    
    rc = proc.wait(timeout=600)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.1f}s, exit code={rc}")
    
    if rc == 0 and Path(engine_path).exists():
        size_mb = Path(engine_path).stat().st_size / 1024 / 1024
        print(f"✅ Engine built! Size: {size_mb:.1f} MB")
    else:
        print("❌ Build failed. Last output:")
        for l in last_lines[-15:]:
            print(f"  {l}")

except subprocess.TimeoutExpired:
    proc.kill()
    print("❌ Timed out after 10 minutes")
except Exception as e:
    print(f"❌ Error: {e}")
