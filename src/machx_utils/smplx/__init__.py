import os
import numpy as np
import torch

def get_mark_direction(a, b):
    return (a * a) / (a * a + b * b)

class SmplxPara:
    def __init__(self, smplxpara):
        if isinstance(smplxpara, str) and os.path.exists(smplxpara):
            smplxpara = np.load(smplxpara)
            self._paras = {}
            self._paras['betas'] = torch.from_numpy(smplxpara['smplx_shape']).float()      # [1, num_betas]
            self._paras['expression'] = torch.from_numpy(smplxpara['smplx_expr']).float()  # [1, num_expression_coeffs]
            
            self._paras['transl'] = torch.from_numpy(smplxpara['cam_trans']).float()
            self._paras['global_orient'] = torch.from_numpy(smplxpara['smplx_root_pose']).float()  # [1, 3]
            self._paras['body_pose'] = torch.from_numpy(smplxpara['smplx_body_pose']).unsqueeze(0).float() # [1, 21*3] 或 [1, 21, 3]
            self._paras['left_hand_pose']=torch.from_numpy(smplxpara['smplx_lhand_pose']).unsqueeze(0).float()
            self._paras['right_hand_pose']=torch.from_numpy(smplxpara['smplx_rhand_pose']).unsqueeze(0).float()
            self._paras['jaw_pose']=torch.from_numpy(smplxpara['smplx_jaw_pose']).float()
            self._paras['leye_pose']=torch.from_numpy(smplxpara['smplx_leye_pose']).float()
            self._paras['reye_pose']=torch.from_numpy(smplxpara['smplx_reye_pose']).float()
            
            self._paras['focal'] = smplxpara['focal'].tolist()
            self._paras['princpt'] = smplxpara['princpt'].tolist()

    def init_drn(self):
        root_pose = self._paras['global_orient'][0]
        from scipy.spatial.transform import Rotation
        r = Rotation.from_rotvec(root_pose)
        rotation_matrix = r.as_matrix()
        vector_direction = np.array([0, 0, 1])
        vector_direction = np.dot(rotation_matrix, vector_direction)
        
        if vector_direction[2] < 0 and np.abs(vector_direction[2]) > np.abs(vector_direction[0]):
            direction = "front"
        elif vector_direction[2] > 0 and np.abs(vector_direction[2]) > np.abs(vector_direction[0]):
            direction = "back"
        elif vector_direction[0] < 0 and np.abs(vector_direction[0]) > np.abs(vector_direction[2]):
            direction = "left"
        elif vector_direction[0] > 0 and np.abs(vector_direction[0]) > np.abs(vector_direction[2]):
            direction = "right"
        
        if direction in ['front', 'back']:
            mark_direction = str(get_mark_direction(vector_direction[2], vector_direction[0]))
        elif direction in ['left', 'right']:
            mark_direction = str(get_mark_direction(vector_direction[0], vector_direction[2]))
        else:
            mark_direction = str(0)

        vector_direction = [str(item) for item in vector_direction]
        return direction, vector_direction, mark_direction