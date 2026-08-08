import torch
from thop import profile
from hazard_models import load_cnn_model, load_resnet18_model, DEVICE

CNN_CHECKPOINT = "hazard_cnn_results/custom_cnn_from_scratch_best_model.pth"
RESNET_CHECKPOINT = "hazard_resnet_results/resnet_18_pretrained_frozen_backbone_best_model.pth"

dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

cnn = load_cnn_model(CNN_CHECKPOINT)
resnet = load_resnet18_model(RESNET_CHECKPOINT)

cnn_macs, cnn_params = profile(cnn, inputs=(dummy_input,))
resnet_macs, resnet_params = profile(resnet, inputs=(dummy_input,))


# FLOPs ≈ 2 * MACs
cnn_gflops = (2 * cnn_macs) / 1e9
resnet_gflops = (2 * resnet_macs) / 1e9

print("CNN")
print(f"Params: {cnn_params:,}")
print(f"GFLOPs: {cnn_gflops:.4f}")

print("\nResNet-18")
print(f"Params: {resnet_params:,}")
print(f"GFLOPs: {resnet_gflops:.4f}")