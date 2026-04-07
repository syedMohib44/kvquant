# Why not QWEN 3
For Qwen3.5, unquantized generation works perfectly; quantized generation falls back gracefully because Qwen3.5 uses a hybrid architecture (transformer + linear attention layers).