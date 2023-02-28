import numpy as np
import re
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def get_re_result(regex_pattern, target_str, assertion_information="Bad regex!"):
    """进行正则表达式search匹配并返回需要匹配的目标的tuple"""
    match_result = re.search(regex_pattern, target_str)
    assert match_result is not None, assertion_information
    return match_result.groups()


index_cnt = 0
def load_bvh_model(lines, pos, father_index):
    """递归读取bvh中的模型信息，lines为readlines后的bvh，pos是当前读取到第几行，father_index是该节点的父节点"""
    joint_type = lines[pos].strip().split()[0]
    # print(joint_type)
    assert joint_type == "End" or joint_type == "JOINT" or joint_type == "ROOT", "Bad joint name"

    joint_name = []
    joint_parent = [father_index]
    tmp_offsets = get_re_result(r"OFFSET\s*([-]*[0-9]\.[0-9]*)\s*([-]*[0-9]\.[0-9]*)\s*([-]*[0-9]\.[0-9]*)", lines[pos+2], "No OFFSET lines")
    float_offsets = []
    for i in range(len(tmp_offsets)):
        float_offsets.append(float(tmp_offsets[i]))
    offsets = [float_offsets]
    if joint_type == "End":
        joint_name = None
        # offsets.append(get_re_result(r"\sOFFSET\s*([-]*[0-9]\.[0-9]*)\s*([-]*[0-9]\.[0-9]*)\s*([-]*[0-9]\.[0-9]*)", lines[pos+2], "No OFFSET lines"))
        return joint_name, joint_parent, offsets, pos+4
    
    global index_cnt
    cur_index = index_cnt
    index_cnt += 1
    joint_name.append(get_re_result(r"JOINT\s*(\w*)", lines[pos], "No JOINT")[0] if joint_type == "JOINT" else get_re_result(r"ROOT\s*(\w*)", lines[pos], "No JOINT")[0])
    # print(f"find {joint_name}")
    # channels = get_re_result(r"\sCHANNELS\s*([\w\s]*)", lines[pos+3], "No CHANNELS lines")
    pos = pos + 4
    while "}" not in lines[pos]:
        tmp_joint_name, tmp_joint_parent, tmp_offsets, pos = load_bvh_model(lines, pos, cur_index)
        if tmp_joint_name:
            joint_name += tmp_joint_name
            joint_parent += tmp_joint_parent
            offsets += tmp_offsets
        # print(lines[pos])
    return joint_name, joint_parent, offsets, pos+1



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    f = open(bvh_file_path, 'r')
    lines = f.readlines()
    f.close()

    get_re_result(r"HIERARCHY", lines[0], "Bad first line no HIERARCHY")
    # get_re_result(r"ROOT\s(\w*)", lines[1], "Bad second line no ROOT")
    # print(lines[3])
    # # OFFSET   0.000000   0.000000   0.000000
    # offsets = get_re_result(r"OFFSET\s*([-]*[0-9]\.[0-9]*)\s*([-]*[0-9]\.[0-9]*)\s*([-]*[0-9]\.[0-9]*)", lines[3], "No ROOT OFFSET lines")
    # channels = get_re_result(r"\sCHANNELS\s([\w\s]*)", lines[4], "No ROOT CHANNELS lines")
    joint_name, joint_parent, tmp_joint_offset, pos = load_bvh_model(lines, 1, -1)
    # for i in range(len(joint_name)):
    #     print(f"{i}, {joint_name[i]}, {joint_parent[i]}, {tmp_joint_offset[i]}")
    joint_offset = np.array(tmp_joint_offset)
    # print(joint_offset)
    return joint_name, joint_parent, joint_offset



def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    motion_data = motion_data[frame_id]
    root_pos = motion_data[0:3] # Front 3 is position X, Y, Z
    motion_data = motion_data[3:]
    rotations = []
    # print(len(joint_name))
    for i in range(len(joint_offset)):
        # print(i,motion_data[i*3 : i*3+3])
        rotations.append(R.from_euler('XYZ', motion_data[i*3 : i*3+3], degrees=True))
    joint_positions = []
    joint_orientations = []
    for i in range(len(joint_offset)):
        parent = joint_parent[i]
        if parent == -1:
            joint_orientations.append(rotations[0].as_quat())
            joint_positions.append(root_pos)
        else:
            # print(parent, i)
            joint_orientations.append((R.from_quat(joint_orientations[parent]) * rotations[i]).as_quat())
            joint_positions.append(joint_positions[parent] + R.from_quat(joint_orientations[parent]).as_matrix() @ joint_offset[i])
    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    T_joint_name, T_joint_parent, T_joint_offse = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offse = part1_calculate_T_pose(A_pose_bvh_path)
    T_motion_data = load_motion_data(T_pose_bvh_path)
    A_motion_data = load_motion_data(A_pose_bvh_path)
    motion_data = None
    return motion_data


if __name__ == "__main__":
    part1_calculate_T_pose("data/walk60.bvh")