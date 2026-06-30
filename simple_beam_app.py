"""
简化高斯光束ABCD矩阵计算器 - Streamlit界面
"""

import html
from datetime import datetime
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from simple_gaussian_beam import (
    SimpleLensSystem,
    SimpleGaussianBeam,
    calculate_beam_at_position,
    calculate_beam_regions,
    filter_lenses_by_direction,
    plot_beam_envelope_interactive,
    plot_curvature_interactive,
    plot_spot_intensity_row_interactive
)


def inject_print_styles():
    """Keep printed/PDF output focused on the current results."""
    st.markdown(
        """
        <style>
        .print-only {
            display: none;
        }

        .print-page-break {
            display: none;
        }

        @media print {
            @page {
                margin: 12mm;
            }

            html, body, .stApp {
                background: #ffffff !important;
            }

            header,
            footer,
            [data-testid="stSidebar"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            [data-testid="stTextInput"],
            [data-testid="stRadio"],
            button,
            iframe[title="streamlit.components.v1.html"] {
                display: none !important;
            }

            [data-testid="stAppViewContainer"] .main .block-container {
                max-width: 100% !important;
                padding: 0 !important;
            }

            [data-testid="stPlotlyChart"],
            [data-testid="stDataFrame"] {
                break-inside: avoid;
                page-break-inside: avoid;
            }

            .print-only {
                display: block !important;
                margin: 0 0 16px 0;
            }

            .print-timestamp {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                font-size: 11px;
                color: #4b5563;
                text-align: right;
                margin: 0 0 8px 0;
            }

            .print-page-break {
                display: block !important;
                break-before: page;
                page-break-before: always;
                height: 0;
                margin: 0;
                padding: 0;
            }

            .print-summary {
                border: 1px solid #d8dee4;
                border-radius: 6px;
                padding: 12px 14px;
                margin-bottom: 18px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                color: #111827;
            }

            .print-summary h2 {
                font-size: 18px;
                margin: 0 0 10px 0;
            }

            .print-summary h3 {
                font-size: 13px;
                margin: 12px 0 6px 0;
            }

            .print-summary table {
                width: 100%;
                border-collapse: collapse;
                font-size: 11px;
            }

            .print-summary th,
            .print-summary td {
                border: 1px solid #d8dee4;
                padding: 5px 6px;
                text-align: left;
            }

            .print-summary th {
                background: #f6f8fa;
                font-weight: 600;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_print_button():
    """Render a small browser-print button for saving the current view as PDF."""
    components.html(
        """
        <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        button {
            width: 100%;
            min-height: 38px;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 6px;
            background: #ffffff;
            color: rgb(49, 51, 63);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
        }

        button:hover {
            border-color: rgb(255, 75, 75);
            color: rgb(255, 75, 75);
        }
        </style>
        <button
            type="button"
            onclick="
                const doc = window.parent.document;
                const originalTitle = doc.title;
                doc.title = '\u200b';
                window.parent.focus();
                setTimeout(() => {
                    window.parent.print();
                    setTimeout(() => { doc.title = originalTitle; }, 500);
                }, 50);
            "
        >
            打印PDF
        </button>
        """,
        height=42,
    )


def render_print_summary(
    wavelength_nm,
    waist_position_x_cm,
    waist_diameter_x_mm,
    M2_x,
    beam_x,
    waist_position_y_cm,
    waist_diameter_y_mm,
    M2_y,
    beam_y,
    lens_list,
    lens_list_x,
    lens_list_y,
    z_max_cm,
):
    """Render a print-only snapshot of the current sidebar configuration."""
    printed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def axis_label(axis):
        return {
            'both': '普通薄透镜（X/Y）',
            'x': '柱透镜（仅作用X方向）',
            'y': '柱透镜（仅作用Y方向）',
        }.get(axis, '普通薄透镜（X/Y）')

    def power_label(lens_type):
        return '会聚' if lens_type == 'converging' else '发散'

    lens_rows = []
    for idx, lens in enumerate(lens_list, start=1):
        lens_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{lens['position'] * 1e2:.2f}</td>"
            f"<td>{lens['f'] * 1e3:.2f}</td>"
            f"<td>{html.escape(power_label(lens['type']))}</td>"
            f"<td>{html.escape(axis_label(lens.get('axis', 'both')))}</td>"
            "</tr>"
        )

    if not lens_rows:
        lens_rows.append("<tr><td colspan='5'>无透镜</td></tr>")

    st.markdown(
        f"""
        <div class="print-only print-timestamp">打印时间：{printed_at}</div>
        <div class="print-only print-summary">
            <h2>当前计算参数</h2>
            <h3>光束参数</h3>
            <table>
                <tr>
                    <th>参数</th>
                    <th>X方向</th>
                    <th>Y方向</th>
                </tr>
                <tr>
                    <td>束腰位置 (cm)</td>
                    <td>{waist_position_x_cm:.2f}</td>
                    <td>{waist_position_y_cm:.2f}</td>
                </tr>
                <tr>
                    <td>束腰直径 (mm)</td>
                    <td>{waist_diameter_x_mm:.5f}</td>
                    <td>{waist_diameter_y_mm:.5f}</td>
                </tr>
                <tr>
                    <td>M²</td>
                    <td>{M2_x:.3f}</td>
                    <td>{M2_y:.3f}</td>
                </tr>
                <tr>
                    <td>瑞利长度 z_R (cm)</td>
                    <td>{beam_x.z_R * 1e2:.2f}</td>
                    <td>{beam_y.z_R * 1e2:.2f}</td>
                </tr>
                <tr>
                    <td>发散全角 (mrad)</td>
                    <td>{beam_x.divergence_angle() * 2 * 1e3:.4f}</td>
                    <td>{beam_y.divergence_angle() * 2 * 1e3:.4f}</td>
                </tr>
            </table>
            <h3>系统设置</h3>
            <table>
                <tr><th>波长 (nm)</th><th>最大传播距离 (cm)</th><th>X有效透镜数</th><th>Y有效透镜数</th></tr>
                <tr>
                    <td>{wavelength_nm:.1f}</td>
                    <td>{z_max_cm:.2f}</td>
                    <td>{len(lens_list_x)}</td>
                    <td>{len(lens_list_y)}</td>
                </tr>
            </table>
            <h3>透镜列表</h3>
            <table>
                <tr>
                    <th>#</th>
                    <th>位置 (cm)</th>
                    <th>焦距 (mm)</th>
                    <th>焦度</th>
                    <th>作用方向</th>
                </tr>
                {''.join(lens_rows)}
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("高斯光束的传播", width="large")
def show_program_notes():
    """Show the theory notes in a compact dialog."""

    # ---------- 1. 表达式 ----------
    st.markdown("### ● 高斯光束表达式")
    st.markdown(
        "傍轴近似下，基模高斯光束的复振幅为"
    )
    st.latex(
        r"E(r,z) = "
        r"E_0\frac{w_0}{w(z)}\;"
        r"\exp\!\left[-\frac{r^2}{w^2(z)}\right]\;"
        r"\exp\!\left[-i\left(kz + \frac{kr^2}{2R(z)} - \psi(z)\right)\right]"
    )
    st.markdown(
        "- **$E_0\\,w_0/w(z)$**：峰值振幅。光束展宽时下降，总功率不变。\n"
        "- **$\\exp[-r^2/w^2(z)]$**：横向场包络为高斯函数，$w(z)$ 为场振幅 $1/e$ 半径。\n"
        "- **$\\exp(-ikz)$**：沿 $z$ 轴的平面波传播相位。\n"
        "- **$\\exp[-ikr^2/(2R(z))]$**：波前弯曲的二次相位，$R(z)$ 为曲率半径。\n"
        "- **$\\exp[i\\psi(z)]$**：Gouy 相移，光束穿过束腰附近时额外积累的相位。"
    )

    # ---------- 2. 强度 ----------
    st.markdown("### ● 强度分布与像散光束")
    st.markdown(
        "实验中测量的是强度 $I = |E|^2$："
    )
    st.latex(
        r"I(r,z) = I_0\left(\frac{w_0}{w(z)}\right)^2 "
        r"\exp\!\left[-\frac{2r^2}{w^2(z)}\right]"
    )
    st.markdown(
        "对像散光束或柱透镜系统，$x$、$y$ 方向独立描述："
    )
    st.latex(
        r"\frac{I(x,y,z)}{I_0} = "
        r"\exp\!\left[-2\left(\frac{x^2}{w_x^2(z)} + \frac{y^2}{w_y^2(z)}\right)\right]"
    )

    # ---------- 3. 关键参数 ----------
    st.markdown("### ● 光束半径、曲率与发散角")

    st.markdown("光束半径沿传播方向的演化：")
    st.latex(r"w(z) = w_0\sqrt{1 + \left(\frac{z - z_0}{z_R}\right)^2}")

    st.markdown("瑞利长度——束腰到光斑半径扩大 $\\sqrt{2}$ 倍的距离：")
    st.latex(r"\boxed{\ z_R = \frac{\pi w_0^2}{\lambda M^2}\ }")

    st.markdown("波前曲率半径：")
    st.latex(
        r"R(z) = (z - z_0)\left[1 + \left(\frac{z_R}{z - z_0}\right)^2\right]"
    )
    st.markdown(
        "束腰处 $R \\to \\infty$（波前为平面）；"
        "远场 $|z - z_0| \\gg z_R$ 时 $R \\approx z - z_0$（趋近球面波）。"
    )

    st.markdown("远场发散半角：")
    st.latex(r"\boxed{\ \theta = \frac{\lambda M^2}{\pi w_0}\ }")

    st.caption(
        "**为什么束腰越小越发散？**  \n"
        "**从波动学观点看：** 光束被压得越窄，横向振幅分布越陡，"
        "光场相邻位置之间的差异越大。自由空间中的衍射正是这种横向不均匀性"
        "在传播中被展开的结果，因此束腰越小，展开得越快。  \n"
        "**从角谱观点看：** 一个很窄的光斑必须由很多不同倾角的平面波叠加出来。"
        "束腰 $w_0$ 越小，需要的倾角范围越宽；这些倾斜分量继续向前传播时，"
        "就会表现为更大的发散角。$z_R$ 正是衡量这一扩散尺度的特征长度。"
    )

    # ---------- 4. q 参数 ----------
    st.markdown("### ● q 参数：宽度与曲率的复化")

    st.markdown(
        "如果每次经过自由空间或透镜都分别追踪 $w(z)$ 和 $R(z)$，计算会很繁琐。"
        "q 参数把它们打包成一个复数："
    )
    st.latex(r"\boxed{\ q(z) = (z - z_0) + i\,z_R\ }")

    st.markdown("它与 $w$、$R$ 的关系为")
    st.latex(
        r"\frac{1}{q(z)} = \frac{1}{R(z)} - i\,\frac{M^2\lambda}{\pi w^2(z)}"
    )
    st.markdown("反过来，从 $q$ 直接读出：")
    st.latex(
        r"w(z) = \sqrt{\frac{M^2\lambda}{\pi\,|\operatorname{Im}(1/q)|}}, \qquad "
        r"R(z) = \frac{1}{\operatorname{Re}(1/q)}"
    )
    st.markdown(
        "束腰处 $\\operatorname{Re}(1/q) = 0$，$R \\to \\infty$，波前为平面。"
    )

    # ---------- 5. ABCD 矩阵 ----------
    st.markdown("### ● ABCD 矩阵与 q 变换")

    st.markdown(
        "傍轴光学系统中，每个元件可用一个 $2 \\times 2$ 矩阵表示。"
        "自由传播和薄透镜不需要重新求解波动方程，直接用矩阵对 $q$ 做线性分式变换即可。"
    )
    st.latex(r"M = \begin{pmatrix}A & B \\ C & D\end{pmatrix}")

    st.markdown("q 参数的变换律：")
    st.latex(r"\boxed{\ q_{\text{out}} = \frac{A\,q_{\text{in}} + B}{C\,q_{\text{in}} + D}\ }")

    st.markdown("本程序用到两个基本矩阵：")
    st.latex(
        r"M_{\text{自由传播}}(d) = \begin{pmatrix}1 & d \\ 0 & 1\end{pmatrix}, \qquad "
        r"M_{\text{薄透镜}}(f) = \begin{pmatrix}1 & 0 \\ -1/f & 1\end{pmatrix}"
    )
    st.markdown(
        "- **自由传播**：$q$ 的实部随距离推进，描述衍射导致的光束展宽和波前弯曲。\n"
        "- **薄透镜**：对波前施加二次相位（$\\propto r^2/f$），改变后续聚焦或发散行为。\n"
        "- **柱透镜**：只在一个横向方向施加焦度，因此仅影响 $q_x$ 或 $q_y$。"
    )


@st.dialog("V2.0 更新说明（2026.6.30）", width="large")
def show_update_notes():
    """Show release notes in a readable dialog."""
    st.markdown(
        "**1. 增加柱透镜模块**  \n"
        "透镜可选择同时作用于 X/Y，或仅作用于 X、Y 单一方向。"
    )
    st.markdown(
        "**2. 新增特定点光斑图**  \n"
        "查询任意传播位置时，可同步查看该位置的二维高斯强度分布。"
    )
    st.markdown(
        "**3. 支持一键打印结果**  \n"
        "在 sidebar 中点击 **打印PDF**，即可将当前参数、透镜信息和输出结果保存为 PDF。"
    )
    st.markdown(
        "**4. 重新整理计算公式说明**  \n"
        "优化了高斯光束表达式、ABCD 矩阵、q 参数与传播物理图像的说明。"
    )


def main():
    """Streamlit交互式应用"""
    st.set_page_config(page_title="高斯光束计算器V2.0", layout="wide")
    inject_print_styles()

    title_left, title_center, title_right = st.columns([1, 6, 1], vertical_alignment="center")
    with title_center:
        st.markdown(
            "<h1 style='text-align:center;margin:0;"
            "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;"
            "font-weight:650;letter-spacing:0'>高斯光束计算器</h1>",
            unsafe_allow_html=True,
        )
    with title_right:
        notes_col, help_col = st.columns(2)
        with notes_col:
            if st.button("💡", key="update_notes_btn"):
                show_update_notes()
        with help_col:
            if st.button("📖", key="program_notes_btn"):
                show_program_notes()
    st.markdown(
        "<hr style='margin:0.35rem 0 0.45rem;border:0;border-top:1px solid #d8dee4;'>",
        unsafe_allow_html=True,
    )
    
    # 侧边栏: 光束参数
    st.sidebar.header('光束参数')
    
    wavelength_nm = st.sidebar.number_input(
        '波长 λ (nm)',
        min_value=200.0,
        max_value=2000.0,
        value=532.0,
        step=1.0
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

        lens_axis_label = st.sidebar.selectbox(
            '透镜类型 / 作用方向',
            ['普通薄透镜（X/Y）', '柱透镜（仅作用X方向）', '柱透镜（仅作用Y方向）'],
            key=f'lens_axis_{i}',
            help='这里的X/Y表示焦度作用方向，不是柱透镜的几何轴向'
        )
        lens_axis = {
            '普通薄透镜（X/Y）': 'both',
            '柱透镜（仅作用X方向）': 'x',
            '柱透镜（仅作用Y方向）': 'y'
        }[lens_axis_label]
        
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
        if focal_length == 0:
            st.sidebar.warning(f'透镜 {i+1} 的焦距不能为 0，当前已跳过。')
            continue
        if focal_length > 0:
            lens_type_key = 'converging'
        elif focal_length < 0:
            lens_type_key = 'diverging'
        else:
            lens_type_key = 'converging'  # 默认凸透镜
        
        lens_list.append({
            'position': lens_position,
            'f': focal_length,
            'type': lens_type_key,
            'axis': lens_axis
        })
    
    # 按位置排序透镜
    lens_list.sort(key=lambda x: x['position'])
    lens_list_x = filter_lenses_by_direction(lens_list, 'x')
    lens_list_y = filter_lenses_by_direction(lens_list, 'y')
    
    # 侧边栏: 传播距离
    st.sidebar.markdown('---')
    st.sidebar.header('传播距离')
    
    z_max_cm = st.sidebar.number_input(
        '最大传播距离 (cm)',
        min_value=1.0,
        max_value=2000.0,
        value=50.0,
        step=1.0
    )
    z_max = z_max_cm * 1e-2

    st.sidebar.markdown('---')
    with st.sidebar:
        render_print_button()

    render_print_summary(
        wavelength_nm,
        waist_position_x_cm,
        waist_diameter_x_mm,
        M2_x,
        beam_x,
        waist_position_y_cm,
        waist_diameter_y_mm,
        M2_y,
        beam_y,
        lens_list,
        lens_list_x,
        lens_list_y,
        z_max_cm,
    )
    
    # 绘制二维光束包络图
    with st.spinner('正在生成包络图...'):
        fig_envelope = plot_beam_envelope_interactive(beam_x, beam_y, lens_list, z_max)
        st.plotly_chart(fig_envelope, use_container_width=True)
    
    # 各区域高斯光束参数
    st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
    st.markdown('---')
    st.header('🔍 各区域高斯光束参数')
    st.markdown(f'**X方向受{len(lens_list_x)}个透镜影响，Y方向受{len(lens_list_y)}个透镜影响；每个方向按其有效透镜独立分区。**')
    
    # 计算X和Y方向的区域信息
    regions_x = calculate_beam_regions(beam_x, lens_list_x)
    regions_y = calculate_beam_regions(beam_y, lens_list_y)
    
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
            spot_data = []
            for z_cm, z_m in zip(z_positions_cm, z_positions):
                w_x, R_x = calculate_beam_at_position(beam_x, lens_list_x, z_m)
                w_y, R_y = calculate_beam_at_position(beam_y, lens_list_y, z_m)

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
                spot_data.append({
                    'z_cm': z_cm,
                    'w_x': w_x,
                    'w_y': w_y
                })

            import pandas as pd
            df_query = pd.DataFrame(query_data)
            st.dataframe(df_query, use_container_width=True)

            st.subheader('2D光斑强度分布')
            spot_extent_mm = 3 * max(
                max(item['w_x'], item['w_y']) * 1e3 for item in spot_data
            )
            fig_spots = plot_spot_intensity_row_interactive(
                spot_data,
                extent_mm=spot_extent_mm
            )
            st.plotly_chart(fig_spots, use_container_width=True)

        except ValueError:
            st.error('请输入有效的数字，多个位置用逗号分隔')

    # 绘制曲率演化图
    st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
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
        st.plotly_chart(fig_curvature, use_container_width=True)


if __name__ == '__main__':
    main()
