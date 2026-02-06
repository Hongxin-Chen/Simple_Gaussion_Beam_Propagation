"""
简化高斯光束ABCD矩阵传播计算器 - 核心函数库
- 所有透镜简化为薄透镜
- 束腰位置固定在z=0处
- 支持凸透镜和凹透镜
- 通过ABCD矩阵计算任意位置的光斑大小和曲率半径
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SimpleLensSystem:
    """简化透镜系统类，只处理薄透镜的ABCD矩阵变换"""
    
    def __init__(self):
        self.elements = []  # 存储光学元件 (type, parameter)
    
    def add_propagation(self, distance):
        """
        添加自由空间传播
        
        ABCD矩阵: [[1, d], [0, 1]]
        """
        self.elements.append(('propagation', distance))
    
    def add_thin_lens(self, focal_length):
        """
        添加薄透镜
        
        ABCD矩阵: [[1, 0], [-1/f, 1]]
        
        参数:
        focal_length: 焦距 (m)
                     正值 = 凸透镜（会聚）
                     负值 = 凹透镜（发散）
        """
        self.elements.append(('lens', focal_length))
    
    def get_abcd_matrix(self, element_type, parameter):
        """
        获取单个光学元件的ABCD矩阵
        
        公式:
        1. 自由空间传播: M = [[1, d], [0, 1]]
        2. 薄透镜: M = [[1, 0], [-1/f, 1]]
        """
        if element_type == 'propagation':
            # 自由空间传播矩阵
            d = parameter
            return np.array([[1, d], [0, 1]])
        elif element_type == 'lens':
            # 薄透镜矩阵
            f = parameter
            if f == 0:
                return np.array([[1, 0], [0, 1]])  # 避免除以零
            return np.array([[1, 0], [-1/f, 1]])
        else:
            return np.array([[1, 0], [0, 1]])  # 单位矩阵
    
    def get_total_abcd_matrix(self):
        """计算整个系统的ABCD矩阵"""
        M = np.array([[1.0, 0.0], [0.0, 1.0]])  # 单位矩阵
        for element_type, parameter in self.elements:
            M_element = self.get_abcd_matrix(element_type, parameter)
            M = M_element @ M  # 矩阵右乘（光从右向左传播）
        return M
    
    def transform_q(self, q_in):
        """
        使用ABCD矩阵变换q参数
        
        公式: q_out = (A·q_in + B) / (C·q_in + D)
        """
        M = self.get_total_abcd_matrix()
        A, B = M[0, 0], M[0, 1]
        C, D = M[1, 0], M[1, 1]
        
        # ABCD变换公式
        q_out = (A * q_in + B) / (C * q_in + D)
        return q_out


class SimpleGaussianBeam:
    """高斯光束类，支持任意束腰位置和M²参数"""
    
    def __init__(self, wavelength, waist_radius, M2=1.0, waist_position=0.0):
        """
        初始化高斯光束
        
        参数:
        wavelength: 波长 (m)
        waist_radius: 束腰半径 (m)
        M2: 光束质量因子 (默认1.0为理想高斯光束)
        waist_position: 束腰位置 (m)，默认在z=0处
        """
        self.wavelength = wavelength
        self.w0 = waist_radius
        self.M2 = M2
        self.z_waist = waist_position
        self.z_R = self.calculate_rayleigh_length()
    
    def calculate_rayleigh_length(self):
        """
        计算瑞利长度（考虑M²参数）
        
        公式: z_R = π·w₀² / (λ·M²)
        
        瑞利长度是光束从束腰传播到光斑半径增大√2倍的距离
        """
        return (np.pi * self.w0**2) / (self.wavelength * self.M2)
    
    def beam_radius(self, z):
        """
        计算在传播距离z处的光斑半径 w(z)
        
        公式: w(z) = w₀·M²·√[1 + ((z-z_waist)/z_R)²]
        
        其中 z_R = π·w₀²/(λ·M²)
        
        参数:
        z: 绝对位置坐标 (m)
        
        返回:
        w(z): 光斑半径 (m)
        """
        return self.w0 * self.M2 * np.sqrt(1 + ((z - self.z_waist) / self.z_R)**2)
    
    def wavefront_curvature(self, z):
        """
        计算波前曲率半径 R(z)
        
        公式: R(z) = (z-z_waist)·[1 + (z_R/(z-z_waist))²]
        
        参数:
        z: 绝对位置坐标 (m)
        
        返回:
        R(z): 波前曲率半径 (m)，在束腰处为无穷大
        """
        z = np.asarray(z)
        z_rel = z - self.z_waist
        # 避免除以零
        z_safe = np.where(np.abs(z_rel) < 1e-10, 1e-10, z_rel)
        R = z_safe * (1 + (self.z_R / z_safe)**2)
        return R
    
    def q_parameter(self, z):
        """
        计算复数q参数
        
        公式: q(z) = (z - z_waist) + i·z_R
        
        其中 z_R = π·w₀²/(λ·M²)
        
        参数:
        z: 绝对位置坐标 (m)
        
        返回:
        q: 复数q参数
        """
        return (z - self.z_waist) + 1j * self.z_R
    
    def divergence_angle(self):
        """
        计算远场发散角 (半角)
        
        公式: θ = λ·M² / (π·w₀)
        
        返回:
        θ: 发散角 (rad)
        """
        return self.wavelength * self.M2 / (np.pi * self.w0)
    
    @staticmethod
    def extract_beam_params(q, wavelength):
        """
        从q参数提取光束参数
        
        公式: 1/q = 1/R - i·λ/(π·w²)
        
        返回:
        (w, R): 光斑半径和波前曲率半径
        """
        q_inv = 1 / q
        real_part = np.real(q_inv)
        
        # 处理束腰处的情况（实部接近零，曲率半径为无穷大）
        if np.abs(real_part) < 1e-10:
            R = np.inf
        else:
            R = 1 / real_part
        
        # w² = λ / (π·|Im(1/q)|)
        w_squared = wavelength / (np.pi * np.abs(np.imag(q_inv)))
        w = np.sqrt(w_squared)
        return w, R


def calculate_beam_at_position(beam, lens_list, z_target):
    """
    计算在目标位置z_target处的光束参数（光斑半径和曲率半径）
    支持任意束腰位置和双向传播
    
    参数:
    beam: SimpleGaussianBeam对象
    lens_list: 透镜列表，每个元素包含 {'position': z_pos, 'f': focal_length, 'type': 'converging'/'diverging'}
    z_target: 目标位置 (m)
    
    返回:
    (w, R): 光斑半径和波前曲率半径
    """
    # 初始q参数（在束腰处）
    q_at_waist = 1j * beam.z_R
    
    # 按位置排序透镜
    sorted_lenses = sorted(lens_list, key=lambda x: x['position'])
    
    # 确定计算路径：从束腰到目标位置
    z_waist = beam.z_waist
    
    # 先从束腰传播到目标位置，考虑路径上的所有透镜
    current_z = z_waist
    current_q = q_at_waist
    
    # 过滤出在束腰和目标之间的透镜
    if z_target >= z_waist:
        # 向前传播
        relevant_lenses = [lens for lens in sorted_lenses if z_waist < lens['position'] <= z_target]
        forward_propagation = True
    else:
        # 向后传播
        relevant_lenses = [lens for lens in sorted_lenses if z_target <= lens['position'] < z_waist]
        relevant_lenses = list(reversed(relevant_lenses))
        forward_propagation = False
    
    for lens in relevant_lenses:
        lens_pos = lens['position']
        
        # 传播到透镜
        dz_to_lens = lens_pos - current_z
        current_q = current_q + dz_to_lens
        
        # 通过透镜
        if forward_propagation:
            # 正向通过透镜：M = [[1, 0], [-1/f, 1]]
            M = np.array([[1, 0], [-1/lens['f'], 1]])
        else:
            # 反向通过透镜：使用逆矩阵 M^-1 = [[1, 0], [1/f, 1]]
            M = np.array([[1, 0], [1/lens['f'], 1]])
        
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        current_q = (A * current_q + B) / (C * current_q + D)
        
        current_z = lens_pos
    
    # 最后传播到目标位置
    dz_final = z_target - current_z
    current_q = current_q + dz_final
    
    # 提取光束参数
    w, R = beam.extract_beam_params(current_q, beam.wavelength)
    return w, R


def calculate_beam_regions(beam, lens_list):
    """
    计算每个区域的高斯光束参数
    N个透镜将空间分为N+1个区域，每个区域都是一个高斯光束
    区域编号：包含初始束腰的区域为区域0，向前为负，向后为正
    
    参数:
    beam: SimpleGaussianBeam对象
    lens_list: 透镜列表
    
    返回:
    list of dict: 每个区域的光束信息 [{'region': i, 'waist_pos': z, 'waist_radius': w0, 'z_R': zR}]
    """
    # 按位置排序透镜
    sorted_lenses = sorted(lens_list, key=lambda x: x['position'])
    
    # 创建所有区域的边界
    boundaries = [0] + [lens['position'] for lens in sorted_lenses] + [float('inf')]
    
    # 找到初始束腰所在的区域索引
    waist_region_idx = 0
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= beam.z_waist < boundaries[i + 1]:
            waist_region_idx = i
            break
    
    # 初始q参数（在束腰处）
    q_at_waist = 1j * beam.z_R
    
    all_regions = []
    
    # 1. 先计算包含束腰的区域（区域0）
    all_regions.append({
        'region': 0,
        'start_z': boundaries[waist_region_idx],
        'end_z': boundaries[waist_region_idx + 1],
        'waist_pos': beam.z_waist,
        'waist_radius': beam.w0,
        'z_R': beam.z_R
    })
    
    # 2. 向后传播（区域1, 2, 3...）
    current_z = beam.z_waist
    current_q = q_at_waist
    
    for i in range(waist_region_idx, len(sorted_lenses)):
        lens = sorted_lenses[i]
        lens_pos = lens['position']
        
        # 传播到透镜
        dz_to_lens = lens_pos - current_z
        current_q = current_q + dz_to_lens
        
        # 通过透镜
        M = np.array([[1, 0], [-1/lens['f'], 1]])
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        q_after_lens = (A * current_q + B) / (C * current_q + D)
        
        # 计算新束腰位置
        z_waist_relative = -np.real(q_after_lens)
        z_waist_absolute = lens_pos + z_waist_relative
        
        # 计算新束腰半径和瑞利长度
        z_R_new = np.imag(q_after_lens)
        w0_new = np.sqrt(beam.wavelength * z_R_new / (np.pi * beam.M2))
        
        # 添加区域信息
        region_num = i - waist_region_idx + 1
        all_regions.append({
            'region': region_num,
            'start_z': lens_pos,
            'end_z': boundaries[i + 2] if i + 2 < len(boundaries) else float('inf'),
            'waist_pos': z_waist_absolute,
            'waist_radius': w0_new,
            'z_R': z_R_new
        })
        
        current_q = q_after_lens
        current_z = lens_pos
    
    # 3. 向前反向传播（区域-1, -2, -3...）
    current_z = beam.z_waist
    current_q = q_at_waist
    
    for i in range(waist_region_idx - 1, -1, -1):
        lens = sorted_lenses[i]
        lens_pos = lens['position']
        
        # 反向传播到透镜
        dz_to_lens = lens_pos - current_z  # 这是负值
        current_q = current_q + dz_to_lens
        
        # 反向通过透镜（使用逆矩阵）
        M = np.array([[1, 0], [1/lens['f'], 1]])
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        q_before_lens = (A * current_q + B) / (C * current_q + D)
        
        # 计算这个区域的束腰位置
        z_waist_relative = -np.real(q_before_lens)
        z_waist_absolute = lens_pos + z_waist_relative
        
        # 计算束腰半径和瑞利长度
        z_R_new = np.imag(q_before_lens)
        w0_new = np.sqrt(beam.wavelength * z_R_new / (np.pi * beam.M2))
        
        # 添加区域信息
        region_num = -(waist_region_idx - i)
        all_regions.insert(0, {
            'region': region_num,
            'start_z': boundaries[i],
            'end_z': lens_pos,
            'waist_pos': z_waist_absolute,
            'waist_radius': w0_new,
            'z_R': z_R_new
        })
        
        current_q = q_before_lens
        current_z = lens_pos
    
    return all_regions


def calculate_waist_after_lens(beam, lens_list):
    """
    计算通过每个透镜后的束腰位置和束腰半径
    支持任意束腰位置
    
    参数:
    beam: SimpleGaussianBeam对象
    lens_list: 透镜列表
    
    返回:
    list of dict: 每个透镜后的束腰信息 [{'lens_pos': z, 'waist_pos': z_waist, 'waist_radius': w0}]
    """
    # 初始q参数（在束腰处）
    q_at_waist = 1j * beam.z_R
    
    # 按位置排序透镜
    sorted_lenses = sorted(lens_list, key=lambda x: x['position'])
    
    current_z = beam.z_waist
    current_q = q_at_waist
    waist_info = []
    
    for lens in sorted_lenses:
        lens_pos = lens['position']
        
        # 传播到透镜
        dz_to_lens = lens_pos - current_z
        current_q = current_q + dz_to_lens
        
        # 通过透镜
        M = np.array([[1, 0], [-1/lens['f'], 1]])
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        q_after_lens = (A * current_q + B) / (C * current_q + D)
        
        # 计算新束腰位置：在束腰处，q的实部为0
        # 从透镜后位置传播距离d使得 Re(q_after_lens + d) = 0
        # 所以 d = -Re(q_after_lens)
        z_waist_relative = -np.real(q_after_lens)
        z_waist_absolute = lens_pos + z_waist_relative
        
        # 计算新束腰半径：在束腰处，w = sqrt(lambda * z_R / pi / M^2)
        z_R_new = np.imag(q_after_lens)
        w0_new = np.sqrt(beam.wavelength * z_R_new / (np.pi * beam.M2))
        
        waist_info.append({
            'lens_pos': lens_pos,
            'waist_pos': z_waist_absolute,
            'waist_radius': w0_new
        })
        
        current_q = q_after_lens
        current_z = lens_pos
    
    return waist_info


def plot_beam_evolution_interactive(beam, lens_list, z_max):
    """绘制交互式光束半径和曲率半径图"""
    # 创建z轴数组
    z_points = np.linspace(0, z_max, 2000)
    
    # 计算每个位置的光束参数
    w_values = []
    R_values = []
    
    for z in z_points:
        w, R = calculate_beam_at_position(beam, lens_list, z)
        w_values.append(w * 1e3)  # 转换为mm
        R_values.append(R)
    
    w_values = np.array(w_values)
    R_values = np.array(R_values)
    
    # 处理无穷大的曲率半径
    R_plot = np.copy(R_values)
    R_plot[np.abs(R_plot) > z_max * 5] = np.nan
    R_plot[np.isinf(R_plot)] = np.nan
    
    # 创建双子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Beam Radius vs Distance', 'Wavefront Curvature vs Distance'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    # 子图1: 光束半径
    fig.add_trace(
        go.Scatter(
            x=z_points,
            y=w_values,
            mode='lines',
            name='w(z)',
            line=dict(color='blue', width=2),
            hovertemplate='z: %{x:.4f} m<br>w: %{y:.4f} mm<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 添加束腰参考线
    fig.add_trace(
        go.Scatter(
            x=[0, z_max],
            y=[beam.w0 * beam.M2 * 1e3, beam.w0 * beam.M2 * 1e3],
            mode='lines',
            name=f'w₀·M² = {beam.w0*beam.M2*1e3:.3f} mm',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='w₀·M²: %{y:.3f} mm<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 子图2: 波前曲率
    fig.add_trace(
        go.Scatter(
            x=z_points,
            y=R_plot,
            mode='lines',
            name='R(z)',
            line=dict(color='purple', width=2),
            hovertemplate='z: %{x:.4f} m<br>R: %{y:.4f} m<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 标记瑞利长度
    for row in [1, 2]:
        fig.add_vline(
            x=beam.z_R,
            line_dash="dash",
            line_color="green",
            annotation_text=f"z_R={beam.z_R*1e2:.1f}cm" if row == 1 else "",
            annotation_position="top",
            row=row, col=1
        )
    
    # 标记透镜位置
    for lens in lens_list:
        lens_type = 'Convex' if lens['type'] == 'converging' else 'Concave'
        for row in [1, 2]:
            fig.add_vline(
                x=lens['position'],
                line_dash="solid",
                line_color="orange",
                line_width=2,
                annotation_text=f"{lens_type}<br>f={lens['f']*1e3:.0f}mm" if row == 1 else "",
                annotation_position="top",
                row=row, col=1
            )
    
    # 更新布局
    fig.update_xaxes(title_text="Distance z (m)", row=2, col=1)
    fig.update_yaxes(title_text="Radius w(z) (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Curvature R(z) (m)", row=2, col=1)
    
    fig.update_layout(
        height=800,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def plot_beam_envelope_interactive(beam_x, beam_y, lens_list, z_max):
    """绘制交互式二维光束包络图，上半部分显示Y方向，下半部分显示X方向"""
    # 创建z轴数组
    z_points_m = np.linspace(0, z_max, 2000)
    z_points = z_points_m * 1e2  # 转换为cm
    
    # 计算X和Y方向每个位置的光束半径
    w_x_values = []
    w_y_values = []
    
    for z_m in z_points_m:
        w_x, _ = calculate_beam_at_position(beam_x, lens_list, z_m)
        w_y, _ = calculate_beam_at_position(beam_y, lens_list, z_m)
        w_x_values.append(w_x * 1e3)  # 转换为mm
        w_y_values.append(w_y * 1e3)  # 转换为mm
    
    w_x_values = np.array(w_x_values)
    w_y_values = np.array(w_y_values)
    
    # 创建图形
    fig = go.Figure()
    
    # 绘制Y方向填充区域（上半部分，正值）
    fig.add_trace(go.Scatter(
        x=np.concatenate([z_points, z_points[::-1]]),
        y=np.concatenate([w_y_values, np.zeros_like(w_y_values)]),
        fill='toself',
        fillcolor='rgba(255, 100, 100, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 绘制X方向填充区域（下半部分，负值）
    fig.add_trace(go.Scatter(
        x=np.concatenate([z_points, z_points[::-1]]),
        y=np.concatenate([np.zeros_like(w_x_values), -w_x_values[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Y方向上边界（红色）
    fig.add_trace(go.Scatter(
        x=z_points,
        y=w_y_values,
        mode='lines',
        name='Y Direction (Upper)',
        line=dict(color='red', width=2),
        hovertemplate='z: %{x:.2f} cm<br>w_y: %{y:.4f} mm<extra></extra>'
    ))
    
    # X方向下边界（蓝色）
    fig.add_trace(go.Scatter(
        x=z_points,
        y=-w_x_values,
        mode='lines',
        name='X Direction (Lower)',
        line=dict(color='blue', width=2),
        hovertemplate='z: %{x:.2f} cm<br>w_x: %{y:.4f} mm<extra></extra>'
    ))
    
    # 标记束腰位置
    # X方向束腰
    if 0 <= beam_x.z_waist <= z_max:
        fig.add_vline(
            x=beam_x.z_waist * 1e2,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"X Waist (z={beam_x.z_waist*1e2:.2f}cm)",
            annotation_position="bottom left"
        )
    
    # Y方向束腰
    if 0 <= beam_y.z_waist <= z_max:
        fig.add_vline(
            x=beam_y.z_waist * 1e2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Y Waist (z={beam_y.z_waist*1e2:.2f}cm)",
            annotation_position="top left"
        )
    
    # 标记透镜位置
    for lens in lens_list:
        if lens['type'] == 'converging':
            lens_type = 'Convex'
            lens_color = 'rgba(255, 140, 0, 0.9)'
        else:
            lens_type = 'Concave'
            lens_color = 'rgba(30, 144, 255, 0.9)'
        
        # 获取透镜位置的光束半径（取X和Y的最大值）
        w_x_at_lens, _ = calculate_beam_at_position(beam_x, lens_list, lens['position'])
        w_y_at_lens, _ = calculate_beam_at_position(beam_y, lens_list, lens['position'])
        lens_height = max(w_x_at_lens, w_y_at_lens) * 1e3 * 1.5
        
        # 绘制透镜竖线
        fig.add_trace(go.Scatter(
            x=[lens['position'] * 1e2, lens['position'] * 1e2],
            y=[-lens_height, lens_height],
            mode='lines',
            name=f"{lens_type} (f={lens['f']*1e3:.0f}mm)",
            line=dict(color=lens_color, width=2),
            showlegend=True,
            hovertemplate=f"{lens_type}<br>f={lens['f']*1e3:.0f}mm<br>z={lens['position']*1e2:.2f}cm<extra></extra>"
        ))
    
    # 添加区域标注
    # 先找到整个绘图范围内Y方向的最大半径，用于统一标注高度
    z_all_samples = np.linspace(0, z_max, 200)
    max_w_y_global = 0
    for z_sample in z_all_samples:
        w_y_sample, _ = calculate_beam_at_position(beam_y, lens_list, z_sample)
        if w_y_sample > max_w_y_global:
            max_w_y_global = w_y_sample
    
    # 统一的标注高度：在全局最高点上方20%
    unified_y_pos = max_w_y_global * 1e3 * 1.2
    
    regions_x = calculate_beam_regions(beam_x, lens_list)
    for region_info in regions_x:
        # 计算区域中心位置（横坐标在区域宽度中间）
        start_z = region_info['start_z']
        end_z = region_info['end_z']
        
        if np.isinf(end_z):
            # 如果结束位置是无穷，使用z_max作为结束位置
            end_z_for_calc = z_max
        else:
            end_z_for_calc = end_z
        
        center_z = (start_z + end_z_for_calc) / 2
        
        # 确保中心位置在绘图范围内
        if center_z <= z_max:
            # 添加区域标注文字（纵坐标统一在最上方），横坐标转换为cm
            fig.add_annotation(
                x=center_z * 1e2,
                y=unified_y_pos,
                text=f"区域{region_info['region']}",
                showarrow=False,
                font=dict(size=11, color='rgba(80, 80, 80, 0.5)'),
                bgcolor='rgba(255, 255, 255, 0.6)',
                borderpad=2,
                bordercolor='rgba(200, 200, 200, 0.3)',
                borderwidth=1
            )
    
    # 更新布局
    fig.update_layout(
        title='2D Beam Propagation (Y-axis: upper, X-axis: lower)',
        xaxis_title='Distance z (cm)',
        yaxis_title='Radius (mm)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_beam_radius_interactive(beam, lens_list, z_max):
    """绘制交互式光束半径图"""
    # 创建z轴数组
    z_points = np.linspace(0, z_max, 2000)
    
    # 计算每个位置的光束参数
    w_values = []
    
    for z in z_points:
        w, _ = calculate_beam_at_position(beam, lens_list, z)
        w_values.append(w * 1e3)  # 转换为mm
    
    w_values = np.array(w_values)
    
    # 创建Plotly图形
    fig = go.Figure()
    
    # 添加光束半径曲线
    fig.add_trace(go.Scatter(
        x=z_points,
        y=w_values,
        mode='lines',
        name='Beam Radius w(z)',
        line=dict(color='blue', width=2),
        hovertemplate='z: %{x:.4f} m<br>w: %{y:.4f} mm<extra></extra>'
    ))
    
    # 添加束腰参考线
    fig.add_trace(go.Scatter(
        x=[0, z_max],
        y=[beam.w0 * beam.M2 * 1e3, beam.w0 * beam.M2 * 1e3],
        mode='lines',
        name=f'w₀·M² = {beam.w0*beam.M2*1e3:.3f} mm',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='w₀·M²: %{y:.3f} mm<extra></extra>'
    ))
    
    # 标记瑞利长度
    fig.add_vline(
        x=beam.z_R,
        line_dash="dash",
        line_color="green",
        annotation_text=f"z_R = {beam.z_R*1e2:.2f} cm",
        annotation_position="top"
    )
    
    # 标记透镜位置
    for lens in lens_list:
        lens_type = 'Convex' if lens['type'] == 'converging' else 'Concave'
        fig.add_vline(
            x=lens['position'],
            line_dash="solid",
            line_color="orange",
            line_width=2,
            annotation_text=f"{lens_type}<br>f={lens['f']*1e3:.1f}mm",
            annotation_position="top"
        )
    
    # 更新布局
    fig.update_layout(
        title='Beam Radius vs Distance',
        xaxis_title='Distance z (m)',
        yaxis_title='Radius w(z) (mm)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig
    """绘制光束半径和曲率半径随传播距离的变化"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建z轴数组 - 使用更多的点以获得平滑曲线
    z_points = np.linspace(0, z_max, 2000)
    
    # 计算每个位置的光束参数
    w_values = []
    R_values = []
    
    for z in z_points:
        w, R = calculate_beam_at_position(beam, lens_list, z)
        w_values.append(w * 1e3)  # 转换为mm
        R_values.append(R)
    
    w_values = np.array(w_values)
    R_values = np.array(R_values)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: 光斑半径
    ax1.plot(z_points, w_values, 'b-', linewidth=2, label=f'w(z)')
    ax1.axhline(beam.w0 * 1e3, color='r', linestyle='--', alpha=0.5, label=f'w_0 = {beam.w0*1e3:.3f} mm')
    ax1.axvline(beam.z_R, color='g', linestyle='--', alpha=0.5, label=f'z_R = {beam.z_R*1e3:.1f} mm')
    
    # 标记透镜位置
    for i, lens in enumerate(lens_list):
        lens_type_str = 'Convex' if lens['type'] == 'converging' else 'Concave'
        ax1.axvline(lens['position'], color='orange', linestyle='-', alpha=0.7, linewidth=2)
        ax1.text(lens['position'], ax1.get_ylim()[1] * 0.9, 
                f"{lens_type_str}\nf={lens['f']*1e3:.1f}mm", 
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Propagation Distance z (m)', fontsize=12)
    ax1.set_ylabel('Beam Radius w(z) (mm)', fontsize=12)
    ax1.set_title('Beam Radius vs Propagation Distance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 波前曲率半径
    # 限制R的显示范围，避免无穷大
    R_plot = np.copy(R_values)
    R_plot[np.abs(R_plot) > z_max * 5] = np.nan  # 过大的值设为nan
    R_plot[np.isinf(R_plot)] = np.nan  # 无穷大设为nan
    
    ax2.plot(z_points, R_plot, 'purple', linewidth=2, label='R(z)')
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # 标记透镜位置
    for lens in lens_list:
        ax2.axvline(lens['position'], color='orange', linestyle='-', alpha=0.7, linewidth=2)
    
    ax2.set_xlabel('Propagation Distance z (m)', fontsize=12)
    ax2.set_ylabel('Wavefront Curvature R(z) (m)', fontsize=12)
    ax2.set_title('Wavefront Curvature vs Propagation Distance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_curvature_interactive(beam, lens_list, z_max, direction='X'):
    """绘制交互式波前曲率演化图"""
    # 创建z轴数组
    z_points_m = np.linspace(0, z_max, 2000)
    z_points = z_points_m * 1e2  # 转换为cm
    
    # 计算每个位置的曲率半径
    R_values = []
    
    for z_m in z_points_m:
        _, R = calculate_beam_at_position(beam, lens_list, z_m)
        R_values.append(R)
    
    R_values = np.array(R_values)
    
    # 处理无穷大的曲率半径
    R_plot = np.copy(R_values)
    R_plot[np.abs(R_plot) > z_max * 5] = np.nan
    R_plot[np.isinf(R_plot)] = np.nan
    
    # 根据方向选择颜色
    line_color = 'blue' if direction == 'X' else 'red'
    
    # 创建单图
    fig = go.Figure()
    
    # 添加曲率半径曲线
    fig.add_trace(
        go.Scatter(
            x=z_points,
            y=R_plot,
            mode='lines',
            name=f'R_{direction.lower()}(z)',
            line=dict(color=line_color, width=2),
            hovertemplate='z: %{x:.2f} cm<br>R: %{y:.4f} m<extra></extra>'
        )
    )
    
    # 标记透镜位置
    for lens in lens_list:
        if lens['type'] == 'converging':
            lens_type = 'Convex'
            lens_color = 'rgba(255, 140, 0, 0.7)'
        else:
            lens_type = 'Concave'
            lens_color = 'rgba(30, 144, 255, 0.7)'
        
        fig.add_vline(
            x=lens['position'] * 1e2,
            line_dash="solid",
            line_width=2,
            line_color=lens_color,
            annotation_text=f"{lens_type}<br>f={lens['f']*1e3:.0f}mm",
            annotation_position="top"
        )
    
    # 更新布局
    fig.update_xaxes(title_text='Distance (cm)', gridcolor='lightgray')
    fig.update_yaxes(title_text='Wavefront Curvature R (m)', gridcolor='lightgray')
    fig.update_layout(
        title=f'Wavefront Curvature ({direction} Direction)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        plot_bgcolor='white'
    )
    
    return fig


def plot_beam_envelope_2d(beam, lens_list, z_max):
    """绘制高斯光束传播的二维包络图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建z轴数组
    z_points = np.linspace(0, z_max, 2000)
    
    # 计算每个位置的光束半径
    w_values_upper = []
    w_values_lower = []
    
    for z in z_points:
        w, _ = calculate_beam_at_position(beam, lens_list, z)
        w_values_upper.append(w * 1e3)  # 转换为mm
        w_values_lower.append(-w * 1e3)  # 负值表示下边界
    
    w_values_upper = np.array(w_values_upper)
    w_values_lower = np.array(w_values_lower)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制光束包络
    ax.fill_between(z_points, w_values_lower, w_values_upper, 
                     alpha=0.3, color='blue', label='Beam Envelope')
    ax.plot(z_points, w_values_upper, 'b-', linewidth=2, label='Upper Boundary')
    ax.plot(z_points, w_values_lower, 'b-', linewidth=2, label='Lower Boundary')
    
    # 绘制束腰位置
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Waist (z=0)')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # 标记透镜位置
    for i, lens in enumerate(lens_list):
        lens_type_str = 'Convex' if lens['type'] == 'converging' else 'Concave'
        ax.axvline(lens['position'], color='orange', linestyle='-', alpha=0.7, linewidth=3)
        
        # 在透镜位置绘制透镜符号
        w_at_lens, _ = calculate_beam_at_position(beam, lens_list, lens['position'])
        lens_height = max(w_at_lens * 1e3 * 1.5, ax.get_ylim()[1] * 0.8)
        
        ax.plot([lens['position'], lens['position']], 
                [-lens_height, lens_height], 
                'o-', color='orange', linewidth=4, markersize=8)
        
        ax.text(lens['position'], lens_height * 1.1, 
                f"{lens_type_str}\nf={lens['f']*1e3:.1f}mm", 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.set_xlabel('Propagation Distance z (m)', fontsize=13)
    ax.set_ylabel('Beam Radius (mm)', fontsize=13)
    ax.set_title('2D Beam Propagation Envelope', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
