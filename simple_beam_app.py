"""
简化高斯光束ABCD矩阵计算器 - Streamlit界面
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from simple_gaussian_beam import (
    SimpleLensSystem,
    SimpleGaussianBeam,
    calculate_beam_at_position,
    calculate_beam_regions,
    plot_beam_envelope_interactive,
    plot_curvature_interactive
)


def main():
    """Streamlit交互式应用"""
    st.set_page_config(page_title="简化高斯光束计算器", page_icon="🔬", layout="wide")
    
    st.title('🔬 简化高斯光束计算器')
    st.markdown('---')
    
    # 显示说明
    with st.expander("📖 程序说明", expanded=False):
        # 提供PDF下载链接
        import os
        pdf_path = os.path.join(os.path.dirname(__file__), "DerivationofGaussionBeam.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄下载高斯光束推导PDF",
                    data=pdf_file,
                    file_name="DerivationofGaussionBeam.pdf",
                    mime="application/pdf"
                )
        st.markdown("""
        ## 计算原理
        
        本程序基于 **复曲率半径q参数** 和 **ABCD矩阵** 方法计算高斯光束在光学系统中的传播。
        
        ---
        
        ### 1️⃣ q参数定义
        
        高斯光束在任意位置 z 的状态可用复数 q 参数完整描述：
        
        $$q(z) = z - z_0 + i z_R$$
        
        其中：
        - $z_0$：束腰位置（本程序固定为 0）
        - $z_R = \\frac{\\pi w_0^2 M^2}{\\lambda}$：瑞利长度
        - $w_0$：束腰半径
        - $M^2$：光束质量因子
        
        **简化形式**（束腰在原点）：
        $$q(z) = z + i z_R$$
        
        **发散角**（远场半角）：
        $$\\theta = \\frac{\\lambda M^2}{\\pi w_0}$$
        
        ---
        
        ### 2️⃣ ABCD矩阵变换
        
        光学元件对q参数的作用通过ABCD矩阵表示：
        
        $$q_{out} = \\frac{A q_{in} + B}{C q_{in} + D}$$
        
        **自由空间传播距离 d：**
        
        $M_{\\text{propagation}} = \\begin{pmatrix}1 & d\\\\0 & 1\\end{pmatrix}$
        
        **薄透镜焦距 f：**
        
        $M_{\\text{lens}} = \\begin{pmatrix}1 & 0\\\\-1/f & 1\\end{pmatrix}$
        
        - **凸透镜**（会聚）：$f > 0$
        - **凹透镜**（发散）：$f < 0$
        
        ---
        
        ### 3️⃣ 光束参数解耦
        
        从q参数提取物理量（光斑半径w 和 波前曲率半径R）：
        
        $$\\frac{1}{q} = \\frac{1}{R(z)} - i\\frac{\\lambda}{\\pi w^2(z)}$$
        
        **求解步骤**：
        
        1. 计算 $\\frac{1}{q}$ 的实部和虚部：
           $$\\text{Re}\\left(\\frac{1}{q}\\right) = \\frac{1}{R(z)}, \\quad \\text{Im}\\left(\\frac{1}{q}\\right) = -\\frac{\\lambda}{\\pi w^2(z)}$$
        
        2. 提取光斑半径：
           $$w(z) = \\sqrt{-\\frac{\\lambda}{\\pi \\cdot \\text{Im}(1/q)}}$$
        
        3. 提取曲率半径：
           $$R(z) = \\frac{1}{\\text{Re}(1/q)}$$
           
           特殊情况：当 $\\text{Re}(1/q) \\approx 0$ 时，$R(z) \\to \\infty$（平面波）
        
        ---
        
        ### 4️⃣ 计算流程
        
        1. **初始化**：在束腰处 $q_0 = 0 + i z_R$
        2. **传播到第一个透镜**：应用传播矩阵 $q_1 = q_0 + d_1$
        3. **通过透镜**：应用透镜矩阵 $q_2 = \\frac{q_1}{1 - q_1/f}$
        4. **继续传播**：重复步骤2-3直到目标位置
        5. **解耦参数**：从最终的 $q$ 提取 $w$ 和 $R$
        
        ---
        
        ### 🔧 简化假设
        
        - 束腰固定在 **z = 0** 处
        - 仅考虑 **薄透镜**（厚度忽略不计）
        - 傍轴近似（小角度传播）
        """)
    
    # 侧边栏: 光束参数
    st.sidebar.header('光束参数')
    
    wavelength_nm = st.sidebar.number_input(
        '波长 λ (nm)',
        min_value=200.0,
        max_value=2000.0,
        value=532.0,
        step=1.0,
        help='常用激光波长：532nm(绿光), 1064nm(红外)'
    )
    wavelength = wavelength_nm * 1e-9
    
    # X方向参数
    st.sidebar.markdown('### X方向参数')
    waist_position_x_cm = st.sidebar.number_input(
        '束腰位置 z_waist_x (cm)',
        min_value=-1000.0,
        max_value=1000.0,
        value=0.0,
        step=0.1,
        help='X方向束腰的绝对位置'
    )
    waist_position_x = waist_position_x_cm * 1e-2
    
    waist_diameter_x_mm = st.sidebar.number_input(
        '束腰直径 D₀_x (mm)',
        min_value=0.00002,
        max_value=20.0,
        value=0.2,
        step=0.00001,
        format="%.5f",
        help='X方向在束腰处的光束直径（D₀ = 2w₀）'
    )
    w0_x = waist_diameter_x_mm / 2 * 1e-3
    
    M2_x = st.sidebar.number_input(
        '光束质量因子 M²_x',
        min_value=1.0,
        max_value=10.0,
        value=1.0,
        step=0.001,
        format="%.3f",
        help='M²=1为理想高斯光束，M²>1为非理想光束'
    )
    
    # Y方向参数
    st.sidebar.markdown('### Y方向参数')
    waist_position_y_cm = st.sidebar.number_input(
        '束腰位置 z_waist_y (cm)',
        min_value=-1000.0,
        max_value=1000.0,
        value=0.0,
        step=0.1,
        help='Y方向束腰的绝对位置'
    )
    waist_position_y = waist_position_y_cm * 1e-2
    
    waist_diameter_y_mm = st.sidebar.number_input(
        '束腰直径 D₀_y (mm)',
        min_value=0.00002,
        max_value=20.0,
        value=0.2,
        step=0.00001,
        format="%.5f",
        help='Y方向在束腰处的光束直径（D₀ = 2w₀）'
    )
    w0_y = waist_diameter_y_mm / 2 * 1e-3
    
    M2_y = st.sidebar.number_input(
        '光束质量因子 M²_y',
        min_value=1.0,
        max_value=10.0,
        value=1.0,
        step=0.001,
        format="%.3f",
        help='M²=1为理想高斯光束，M²>1为非理想光束'
    )
    
    # 创建X和Y方向的光束
    beam_x = SimpleGaussianBeam(wavelength, w0_x, M2_x, waist_position_x)
    beam_y = SimpleGaussianBeam(wavelength, w0_y, M2_y, waist_position_y)
    
    # 显示计算的瑞利长度和发散角
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric('X方向', '')
        st.metric('z_R_x (cm)', f'{beam_x.z_R * 1e2:.2f}')
        st.metric('θ_x (全角, mrad)', f'{beam_x.divergence_angle() * 2 * 1e3:.4f}')
    with col2:
        st.metric('Y方向', '')
        st.metric('z_R_y (cm)', f'{beam_y.z_R * 1e2:.2f}')
        st.metric('θ_y (全角, mrad)', f'{beam_y.divergence_angle() * 2 * 1e3:.4f}')
    
    # 侧边栏: 透镜系统配置
    st.sidebar.markdown('---')
    st.sidebar.header('透镜系统配置')
    
    num_lenses = st.sidebar.number_input(
        '透镜数量',
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )
    
    lens_list = []
    
    for i in range(num_lenses):
        st.sidebar.markdown(f'---')
        st.sidebar.markdown(f'**透镜 {i+1}**')
        
        lens_position_cm = st.sidebar.number_input(
            f'透镜位置 (cm)',
            min_value=0.1,
            max_value=1000.0,
            value=10.0 * (i + 1),
            step=0.1,
            key=f'lens_pos_{i}'
        )
        lens_position = lens_position_cm * 1e-2  # cm -> m
        
        focal_length_mm = st.sidebar.number_input(
            f'焦距 f (mm)',
            min_value=-10000.0,
            max_value=10000.0,
            value=100.0,
            step=1.0,
            key=f'focal_{i}',
            help='正值=凸透镜（会聚），负值=凹透镜（发散）'
        )
        
        # 根据焦距符号自动判断透镜类型
        focal_length = focal_length_mm * 1e-3  # mm -> m
        if focal_length > 0:
            lens_type_key = 'converging'
        elif focal_length < 0:
            lens_type_key = 'diverging'
        else:
            lens_type_key = 'converging'  # 默认凸透镜
        
        lens_list.append({
            'position': lens_position,
            'f': focal_length,
            'type': lens_type_key
        })
    
    # 按位置排序透镜
    lens_list.sort(key=lambda x: x['position'])
    
    # 侧边栏: 传播距离
    st.sidebar.markdown('---')
    st.sidebar.header('传播距离')
    
    z_max_cm = st.sidebar.number_input(
        '最大传播距离 (cm)',
        min_value=1.0,
        max_value=1000.0,
        value=50.0,
        step=1.0
    )
    z_max = z_max_cm * 1e-2
    
    # 绘制二维光束包络图
    st.markdown('---')
    
    with st.spinner('正在生成包络图...'):
        fig_envelope = plot_beam_envelope_interactive(beam_x, beam_y, lens_list, z_max)
        st.plotly_chart(fig_envelope, width='stretch')
    
    # 绘制曲率演化图
    st.markdown('---')
    
    # 添加切换按钮选择X或Y方向
    direction = st.radio(
        '选择显示方向',
        ['X方向', 'Y方向'],
        horizontal=True,
        help='切换显示X或Y方向的波前曲率演化'
    )
    
    with st.spinner('正在生成曲率图...'):
        if direction == 'X方向':
            fig_curvature = plot_curvature_interactive(beam_x, lens_list, z_max, direction='X')
        else:
            fig_curvature = plot_curvature_interactive(beam_y, lens_list, z_max, direction='Y')
        st.plotly_chart(fig_curvature, width='stretch')
    
    # 特定位置光斑参数查询
    st.markdown('---')
    st.header('📍 特定位置光斑参数查询')
    
    z_query_input = st.text_input(
        '输入查询位置 (cm)，多个位置用逗号分隔',
        value='10, 20, 30',
        help='例如: 10, 20, 30'
    )
    
    if z_query_input:
        try:
            z_positions_cm = [float(z.strip()) for z in z_query_input.split(',')]
            z_positions = [z * 1e-2 for z in z_positions_cm]  # cm -> m
            
            # 创建查询结果表格
            query_data = []
            for z_cm, z_m in zip(z_positions_cm, z_positions):
                w_x, R_x = calculate_beam_at_position(beam_x, lens_list, z_m)
                w_y, R_y = calculate_beam_at_position(beam_y, lens_list, z_m)
                
                # 处理无穷大的曲率半径
                R_x_str = '∞' if np.isinf(R_x) or np.abs(R_x) > z_max * 10 else f'{R_x:.4f}'
                R_y_str = '∞' if np.isinf(R_y) or np.abs(R_y) > z_max * 10 else f'{R_y:.4f}'
                
                query_data.append({
                    'z位置 (cm)': f'{z_cm:.2f}',
                    'w_x (mm)': f'{w_x * 1e3:.4f}',
                    'R_x (m)': R_x_str,
                    'w_y (mm)': f'{w_y * 1e3:.4f}',
                    'R_y (m)': R_y_str
                })
            
            import pandas as pd
            df_query = pd.DataFrame(query_data)
            st.dataframe(df_query, use_container_width=True)
            
        except ValueError:
            st.error('请输入有效的数字，多个位置用逗号分隔')
    
    # 各区域高斯光束参数
    st.markdown('---')
    st.header('🔍 各区域高斯光束参数')
    st.markdown(f'**{len(lens_list)}个透镜将空间分为{len(lens_list) + 1}个区域，每个区域都有独立的高斯光束参数**')
    
    # 计算X和Y方向的区域信息
    regions_x = calculate_beam_regions(beam_x, lens_list)
    regions_y = calculate_beam_regions(beam_y, lens_list)
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.subheader('X方向')
        region_data_x = []
        for info in regions_x:
            # 处理无穷大的结束位置
            if np.isinf(info['end_z']):
                range_str = f"{info['start_z'] * 1e2:.2f} - ∞"
            else:
                range_str = f"{info['start_z'] * 1e2:.2f} - {info['end_z'] * 1e2:.2f}"
            
            region_data_x.append({
                '区域': f'区域{info["region"]}',
                '范围 (cm)': range_str,
                '束腰位置 (cm)': f'{info["waist_pos"] * 1e2:.2f}',
                '束腰半径 (mm)': f'{info["waist_radius"] * 1e3:.4f}',
                'z_R (cm)': f'{info["z_R"] * 1e2:.2f}'
            })
        
        import pandas as pd
        df_region_x = pd.DataFrame(region_data_x)
        st.dataframe(df_region_x, use_container_width=True)
    
    with col_y:
        st.subheader('Y方向')
        region_data_y = []
        for info in regions_y:
            # 处理无穷大的结束位置
            if np.isinf(info['end_z']):
                range_str = f"{info['start_z'] * 1e2:.2f} - ∞"
            else:
                range_str = f"{info['start_z'] * 1e2:.2f} - {info['end_z'] * 1e2:.2f}"
            
            region_data_y.append({
                '区域': f'区域{info["region"]}',
                '范围 (cm)': range_str,
                '束腰位置 (cm)': f'{info["waist_pos"] * 1e2:.2f}',
                '束腰半径 (mm)': f'{info["waist_radius"] * 1e3:.4f}',
                'z_R (cm)': f'{info["z_R"] * 1e2:.2f}'
            })
        
        df_region_y = pd.DataFrame(region_data_y)
        st.dataframe(df_region_y, use_container_width=True)


if __name__ == '__main__':
    main()
