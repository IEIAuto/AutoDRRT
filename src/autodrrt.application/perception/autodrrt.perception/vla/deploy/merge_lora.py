import torch

def merge_lora_linear(base_weight, lora_A, lora_B, alpha):
    r = lora_A.shape[0]
    scale = alpha / r
    return base_weight + scale * (lora_B @ lora_A)


def merge_all_lora_layers(state_dict, alpha=16, verbose=True):
    merged_state = {}

    for key in state_dict:
        if "base_layer.weight" in key:
            # 构造 lora 对应 key
            lora_prefix = key.replace("base_layer.weight", "")
            lora_A_key = lora_prefix + "lora_A.default.weight"
            lora_B_key = lora_prefix + "lora_B.default.weight"

            if lora_A_key in state_dict and lora_B_key in state_dict:
                W_base = state_dict[key]
                A = state_dict[lora_A_key]
                B = state_dict[lora_B_key]

                # 合并
                W_eff = merge_lora_linear(W_base, A, B, alpha)
                merged_key = key.replace("base_layer.", "")  # e.g. x.q_proj.weight

                if verbose:
                    print(f"✔ Merged {merged_key} from {key} + LoRA")

                merged_state[merged_key] = W_eff
            else:
                print(f"⚠ Warning: missing LoRA weights for {key}, skipped.")
        elif "lora_A" in key or "lora_B" in key:
            # 跳过 LoRA 权重
            continue
        else:
            # 保留其他非 LoRA 层
            merged_state[key] = state_dict[key]

    return merged_state


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pth", required=True, help="LoRA 微调后的 .pth 文件")
    parser.add_argument("--output_pth", required=True, help="融合后输出的 .pth 文件")
    parser.add_argument("--alpha", type=float, default=16, help="LoRA alpha 超参数")
    args = parser.parse_args()

    print(f"🔍 加载权重: {args.input_pth}")
    state_dict = torch.load(args.input_pth, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    print("🚀 开始融合 LoRA 权重...")
    merged_state = merge_all_lora_layers(state_dict, alpha=args.alpha)

    # 保存
    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)
    torch.save(merged_state, args.output_pth)
    print(f"✅ 融合完成，保存到: {args.output_pth}")
