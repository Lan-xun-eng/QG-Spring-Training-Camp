import json
import numpy as np

json_file = "E:\\pycharm项目\\QG\\data\\data(1).json"

class CoordinateSystem:
    def __init__(self, axes, vectors):
        self.axes = np.array(axes)
        self.vectors = np.array(vectors)

    @staticmethod
    def is_right_axes(axes):
        mat = np.array(axes)

        # 判断一：是否为方阵
        if mat.shape[0] != mat.shape[1]:
            return False
        # 判断二：行列式是否为0
        det = np.linalg.det(mat)
        return abs(det) > 1e-10

    def transfer(self, target_axes):
        if not self.is_right_axes(target_axes):
            raise ValueError("目标坐标系轴线性相关，无法转移坐标")
        B = np.array(target_axes, dtype=float)

        # 向量在绝对坐标轴的表示
        abs_vecs = self.vectors @ self.axes.T
        # 将目标坐标轴矩阵转化为逆矩阵
        B_inv = np.linalg.inv(B)
        # 在目标坐标轴向量的新坐标
        new_vectors = abs_vecs @ B_inv.T

        new_cs = CoordinateSystem(target_axes, new_vectors)
        return new_cs

    def project(self, target_axes):
        if not self.is_right_axes(target_axes):
            raise ValueError("目标坐标系轴线性相关，无法计算投影")
        B = np.array(target_axes, dtype=float)

        # 向量在绝对坐标轴的表示
        abs_vecs = self.vectors @ self.axes.T
        # 坐标轴单位化
        norms = np.linalg.norm(B, axis=0)
        unit_axes = B / norms
        # 计算投影
        pj = abs_vecs @ unit_axes

        return pj

    def angle(self, target_axes):
        if not self.is_right_axes(target_axes):
            raise ValueError("目标坐标系轴线性相关，无法计算夹角")
        B = np.array(target_axes, dtype=float)

        # 向量在绝对坐标轴的表示
        abs_vecs = self.vectors @ self.axes.T   # (n, dim)
        # 计算向量和坐标轴的长度
        vec_norms = np.linalg.norm(abs_vecs, axis=1)   # (n,)
        axes_norms = np.linalg.norm(B, axis=0)   # (dim,)
        # 计算向量与目标坐标轴的点积
        dot = abs_vecs @ B   # (n, dim)
        # 计算余弦值
        cos = dot / (vec_norms[:, np.newaxis] * axes_norms)
        cos = np.clip(cos, -1.0, 1.0)
        # 计算夹角
        angles = np.arccos(cos)

        return angles

    def area(self, target_axes):
        if not self.is_right_axes(target_axes):
            raise ValueError("目标坐标系轴线性相关，无法计算面积缩放倍数")
        B = np.array(target_axes, dtype=float)
        area = abs(np.linalg.det(B))

        return area

def group_process(group):
    print(f"\n======== 当前处理组：{group["group_name"]} ========")

    cs = CoordinateSystem(group["ori_axis"], group["vectors"])

    for index, task in enumerate(group["tasks"], 1):
        task_type = task["type"]

        if task_type == "change_axis":
            target_axes = task["obj_axis"]
            cs = cs.transfer(target_axes)
            print("\n转移后的向量坐标（在新坐标系下）:\n")
            print(cs.vectors)

        elif task_type == "axis_projection":
            target_axes = cs.axes
            pj = cs.project(target_axes)
            print(f"\n投影长度（每个向量在各轴上的投影）:\n{pj}")

        elif task_type == "axis_angle":
            target_axes = cs.axes
            angles = cs.angle(target_axes)
            print(f"\n夹角（弧度，每个向量与各轴的夹角）:\n{angles}")

        elif task_type == "area":
            target_axes = cs.axes
            area = cs.area(target_axes)
            print(f"\n面积缩放倍数:{area}")

        else:
            print(f"\n未知任务类型: {task_type}")

def main(json_file):
    with open(json_file, 'r', encoding = 'utf-8') as f:
        data = json.load(f)

    for group in data:
        group_process(group)

if __name__ == "__main__":
    main(json_file)
