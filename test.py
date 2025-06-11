import torch

# 加载模型参数字典
state_dict = torch.load('params\swin_tiny_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth', map_location='cpu')  # 加 map_location 保证跨设备兼容

mode_state_dict = []

for key in state_dict:
    if "class_embed" not in key:
        mode_state_dict.append()


torch.save(state_dict, 'modified.pth')
