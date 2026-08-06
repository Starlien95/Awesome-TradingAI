#!/usr/bin/env python3
"""
verify_prediction_consistency.py
验证 Qlib 模型和原生 PyTorch 模型的预测一致性
"""
import pickle

import numpy as np
import qlib
import torch


def verify_consistency():
    print("=" * 60)
    print("  验证 Qlib 模型 vs 原生 PyTorch 模型 预测一致性")
    print("=" * 60)

    # 初始化 Qlib
    qlib.init(provider_uri="./qlib_data/15m")

    # =========================================
    # 1. 加载 Qlib 封装的模型
    # =========================================
    print("\n1. 加载模型...")

    with open("./qlib_models/mlp/15m/qlib_mlp_15m_model.pkl", "rb") as f:
        qlib_model = pickle.load(f)

    # 从 Qlib 模型中提取 PyTorch 模型
    pytorch_model_from_qlib = qlib_model.dnn_model

    # 加载独立保存的 PyTorch 模型
    pytorch_model_standalone = torch.load(
        "./qlib_models/mlp/15m/mlp_15m_model.pth",
        map_location="cpu",
        weights_only=False
    )

    print(f"   Qlib 内部模型类型: {type(pytorch_model_from_qlib)}")
    print(f"   独立 PyTorch 模型类型: {type(pytorch_model_standalone)}")

    # ⭐ 关键修复：将两个模型都移到 CPU
    pytorch_model_from_qlib = pytorch_model_from_qlib.cpu()
    pytorch_model_standalone = pytorch_model_standalone.cpu()
    print("   ✅ 已将两个模型都移到 CPU")

    # =========================================
    # 2. 验证模型权重是否一致
    # =========================================
    print("\n2. 验证模型权重...")

    qlib_state = pytorch_model_from_qlib.state_dict()
    standalone_state = pytorch_model_standalone.state_dict()

    weights_match = True
    for key in qlib_state.keys():
        if key not in standalone_state:
            print(f"   ❌ 缺少权重: {key}")
            weights_match = False
        else:
            # ⭐ 确保两个张量都在 CPU 上再比较
            tensor1 = qlib_state[key].cpu()
            tensor2 = standalone_state[key].cpu()
            if not torch.allclose(tensor1, tensor2, atol=1e-6):
                print(f"   ❌ 权重不一致: {key}")
                weights_match = False

    if weights_match:
        print("   ✅ 所有权重完全一致")

    # =========================================
    # 3. 准备测试数据
    # =========================================
    print("\n3. 准备测试数据...")

    np.random.seed(42)
    test_features = np.random.randn(1, 52).astype(np.float32)

    print(f"   测试数据形状: {test_features.shape}")
    print(f"   特征样例 (前5个): {test_features[0, :5].round(4)}")

    # =========================================
    # 4. 分别用两个模型预测
    # =========================================
    print("\n4. 执行预测...")

    # 设置为评估模式
    pytorch_model_from_qlib.eval()
    pytorch_model_standalone.eval()

    # 数据也在 CPU 上
    test_tensor = torch.from_numpy(test_features).cpu()

    with torch.no_grad():
        pred_qlib = pytorch_model_from_qlib(test_tensor).cpu().numpy()
        pred_standalone = pytorch_model_standalone(test_tensor).cpu().numpy()

    print(f"   Qlib 模型预测:     {pred_qlib.flatten()[0]:.8f}")
    print(f"   独立 PyTorch 预测: {pred_standalone.flatten()[0]:.8f}")

    # =========================================
    # 5. 验证一致性
    # =========================================
    print("\n5. 验证预测一致性...")

    diff = np.abs(pred_qlib - pred_standalone)
    print(f"   预测差异: {diff.flatten()[0]:.2e}")

    if np.allclose(pred_qlib, pred_standalone, atol=1e-6):
        print("   ✅ 预测结果完全一致！")
    else:
        print("   ❌ 预测结果存在差异")

    # =========================================
    # 6. 批量验证（更多样本）
    # =========================================
    print("\n6. 批量验证 (1000 个样本)...")

    batch_features = np.random.randn(1000, 52).astype(np.float32)
    batch_tensor = torch.from_numpy(batch_features).cpu()

    with torch.no_grad():
        batch_pred_qlib = pytorch_model_from_qlib(batch_tensor).cpu().numpy()
        batch_pred_standalone = pytorch_model_standalone(batch_tensor).cpu().numpy()

    max_diff = np.abs(batch_pred_qlib - batch_pred_standalone).max()
    mean_diff = np.abs(batch_pred_qlib - batch_pred_standalone).mean()

    print(f"   最大差异: {max_diff:.2e}")
    print(f"   平均差异: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("   ✅ 批量验证通过！")
    else:
        print("   ⚠️ 存在微小数值差异（可能是浮点精度问题）")

    print("\n" + "=" * 60)
    print("  验证完成")
    print("=" * 60)

    return {
        "weights_match": weights_match,
        "max_prediction_diff": float(max_diff),
        "mean_prediction_diff": float(mean_diff)
    }


if __name__ == "__main__":
    verify_consistency()
