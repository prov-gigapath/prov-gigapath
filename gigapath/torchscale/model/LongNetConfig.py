LongNet_8_layers_256_dim_mlp2 = {
    'encoder_layers': 8,
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 512,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4]',
    'segment_length': '[512, 1024, 2048]',
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_12_layers_256_dim_mlp2 = {
    'encoder_layers': 12,
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 512,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4]',
    'segment_length': '[512, 1024, 2048]',
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_8_layers_256_dim = {
    'encoder_layers': 8,
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_12_layers_256_dim = {
    'encoder_layers': 12,
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_3_layers_384_dim = {
    'encoder_layers': 3,
    'encoder_embed_dim': 384,
    'encoder_ffn_embed_dim': 1536,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_6_layers_384_dim = {
    'encoder_layers': 6,
    'encoder_embed_dim': 384,
    'encoder_ffn_embed_dim': 1536,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_12_layers_384_dim = {
    'encoder_layers': 12,
    'encoder_embed_dim': 384,
    'encoder_ffn_embed_dim': 1536,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_12_layers_512_dim = {
    'encoder_layers': 12,
    'encoder_embed_dim': 512,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 8,
    'dilated_ratio': '[1, 2, 4]',
    'segment_length': '[512, 1024, 2048]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_3_layers_768_dim = {
    'encoder_layers': 3,
    'encoder_embed_dim': 768,
    'encoder_ffn_embed_dim': 3072,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_6_layers_768_dim = {
    'encoder_layers': 6,
    'encoder_embed_dim': 768,
    'encoder_ffn_embed_dim': 3072,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 4096, 8192, 16384, 65536]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_8_layers_768_dim = {
    'encoder_layers': 8,
    'encoder_embed_dim': 768,
    'encoder_ffn_embed_dim': 3072,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_12_layers_768_dim = {
    'encoder_layers': 12,
    'encoder_embed_dim': 768,
    'encoder_ffn_embed_dim': 3072,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_8_layers_1024_dim = {
    'encoder_layers': 8,
    'encoder_embed_dim': 1024,
    'encoder_ffn_embed_dim': 4096,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    #'segment_length': '[512, 1024, 2048, 4096, 8192]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}


LongNet_24_layers_1024_dim = {
    'encoder_layers': 24,
    'encoder_embed_dim': 1024,
    'encoder_ffn_embed_dim': 4096,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    #'segment_length': '[512, 1024, 2048, 4096, 8192]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_3_layers_1536_dim = {
    'encoder_layers': 3,
    'encoder_embed_dim': 1536,
    'encoder_ffn_embed_dim': 6144,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_6_layers_1536_dim = {
    'encoder_layers': 6,
    'encoder_embed_dim': 1536,
    'encoder_ffn_embed_dim': 6144,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_8_layers_1536_dim = {
    'encoder_layers': 8,
    'encoder_embed_dim': 1536,
    'encoder_ffn_embed_dim': 6144,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    #'segment_length': '[512, 1024, 2048, 4096, 8192]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_12_layers_1536_dim = {
    'encoder_layers': 12,
    'encoder_embed_dim': 1536,
    'encoder_ffn_embed_dim': 6144,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    #'segment_length': '[512, 1024, 2048, 4096, 8192]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_Vanilla_12_layers_256_dim = {
    'encoder_layers': 12,
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 512,
    'encoder_attention_heads': 8,
    'dilated_ratio': '[1]',
    'segment_length': '[10000000]',
    'block_shift': False,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_Vanilla_6_layers_768_dim = {
    'encoder_layers': 6,
    'encoder_embed_dim': 768,
    'encoder_ffn_embed_dim': 3072,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1]',
    'segment_length': '[10000000]',
    'block_shift': False,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_Vanilla_6_layers_1536_dim = {
    'encoder_layers': 6,
    'encoder_embed_dim': 1536,
    'encoder_ffn_embed_dim': 6144,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1]',
    'segment_length': '[10000000]',
    'block_shift': False,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_test = {
    'encoder_layers': 1,
    'encoder_embed_dim': 192,
    'encoder_ffn_embed_dim': 192,
    'encoder_attention_heads': 8,
    'dilated_ratio': '[1, 2, 4]',
    'segment_length': '[512, 1024, 2048]',
    'flash_attention': True,
    'block_shift': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}