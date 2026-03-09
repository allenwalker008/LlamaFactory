def patch_qwen3_5_fla_for_mindspeed(config: "PretrainedConfig") -> None:
    model_type = getattr(config, "model_type", None)
    if model_type not in ["qwen3_next", "qwen3_5"]:
        return

    try:
        if model_type == "qwen3_next":
            from transformers.models.qwen3_next import modeling_qwen3_next as modeling_module
        else:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling_module
            
        import fla.ops.gated_delta_rule.chunk as fla_chunk
        
        # 1. 放行 transformers 里的 assert 环境拦截
        modeling_module.is_flash_linear_attention_available = lambda: True
        
        # 2. 从昇腾优化的 mindspeed 中提取原生 8 个 Triton 算子
        from mindspeed.lite.ops.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu
        from mindspeed.lite.ops.triton.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
        from mindspeed.lite.ops.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from mindspeed.lite.ops.triton.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
        
        # 3. 将 fla 包底层的 GPU Triton 算子强制重定向为 MindSpeed 算子，完成“移花接木”
        fla_chunk.chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h
        fla_chunk.chunk_gated_delta_rule_bwd_dhu = chunk_gated_delta_rule_bwd_dhu
        fla_chunk.chunk_bwd_dqkwg = chunk_bwd_dqkwg
        fla_chunk.chunk_bwd_dv_local = chunk_bwd_dv_local
        fla_chunk.chunk_fwd_o = chunk_fwd_o
        fla_chunk.chunk_scaled_dot_kkt_fwd = chunk_scaled_dot_kkt_fwd
        fla_chunk.prepare_wy_repr_bwd = prepare_wy_repr_bwd
        fla_chunk.recompute_w_u_fwd = recompute_w_u_fwd

        logger.info_rank0(f"Success: Applied MindSpeed NPU Triton FLA patch for {model_type}")
    except Exception as e:
        logger.warning_rank0(f"Failed to apply MindSpeed NPU Triton FLA patch for {model_type}: {e}")
