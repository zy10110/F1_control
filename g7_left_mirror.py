"""FSNH wrist-relative LEFT input -> equivalent RIGHT model pose."""
import numpy as np


def canonical_right(frame):
    if frame['hand']!=1:
        raise ValueError('左手镜像入口只接受 hand=1 的官方帧')
    # Physical reflection plus vendor palm basis change gives S=diag(1,-1,1).
    # R' = S R S; quaternion vector parts are axial (det(S)*S).
    return dict(frame,hand=2,source_hand=1,pose_space='mirrored_right',
                positions_m=(np.asarray(frame['positions_m'])*[1,-1,1]).tolist(),
                quaternions_wxyz=(np.asarray(frame['quaternions_wxyz'])*[1,-1,1,-1]).tolist())
