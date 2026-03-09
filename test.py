def apply_mindspeed_fla_patch():
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
        import fla.ops.gated_delta_rule.chunk
        
        # 1. 欺骗 transformers，让它认为当前环境支持 flash-linear-attention
        modeling_qwen3_5.is_flash_linear_attention_available = lambda: True
        
        # 2. 从昇腾优化的 mindspeed 中提取原生 Triton 算子
        from mindspeed.lite.ops.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu
        
        # 3. 将 fla 包底层的 GPU Triton 算子强制重定向为 MindSpeed 算子
        fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h
        fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_bwd_dhu = chunk_gated_delta_rule_bwd_dhu
        print("Success: Applied MindSpeed NPU Triton FLA patch for Qwen3.5")
    except Exception as e:
        print(f"Failed to apply patch: {e}")
