"""
生成完整的 5 字符验证码图片。

每张图片包含 5 个随机字符：
    0-9、A-Z、a-z

输出目录：

    data_full/
    ├── 3fA9k_00000.png
    ├── 8Zq2M_00001.png
    ├── ...
    └── boxes.json

文件名中的下划线前部分就是验证码标签。

例如：

    3fA9k_00000.png

标签为：

    3fA9k
"""

import glob
import json
import math
import os
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ============================================================
# 配置
# ============================================================

OUTPUT_DIR = "./data_full"

# 总图片数量
TOTAL_IMAGES = 10

# 每张验证码包含的字符数量
CAPTCHA_LENGTH = 5

# 最终图片尺寸
IMG_HEIGHT = 40
IMG_WIDTH = 160

# 每个字符的槽位尺寸
CHAR_SLOT_SIZE = 32

# 字符绘制时的字体大小
FONT_SIZE = 28

# 每个字符最大旋转角度
MAX_ROTATION = 18

# 字符最长边占单个槽位的比例
# 提高比例让字符本身占更多像素，在32px小图里更容易辨认
CHAR_SIZE_RATIO = 0.90

# 是否启用单字符仿射扭曲
ENABLE_DISTORTION = True

# 单字符扭曲概率（原0.70太高，几乎每个字符都被斜切，调低）
DISTORTION_PROBABILITY = 0.30

# 是否启用整张图片的波浪扭曲
ENABLE_WAVE_DISTORTION = True

# 整张图片波浪扭曲概率
WAVE_PROBABILITY = 0.20

# 干扰线概率（原0.75太高，调低避免遮挡过多）
NOISE_LINE_PROBABILITY = 0.30

# 网格线概率（原0.75太高，调低避免遮挡过多）
GRID_LINE_PROBABILITY = 0.25

# 模糊概率
BLUR_PROBABILITY = 0.15

# 字符集合
CHARACTERS = list(
    string.digits
    + string.ascii_uppercase
    + string.ascii_lowercase
)

# ============================================================
# 查找字体
# ============================================================

def find_fonts():
    """查找系统中的 TrueType 字体。"""

    candidates = []

    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/mnt/c/Windows/Fonts",
    ]

    for directory in search_dirs:
        if os.path.exists(directory):
            candidates.extend(
                glob.glob(
                    os.path.join(
                        directory,
                        "**",
                        "*.ttf",
                    ),
                    recursive=True,
                )
            )

    return candidates

# ============================================================
# 加载字体
# ============================================================

def load_fonts():
    """加载普通字体和粗体字体。"""

    font_paths = find_fonts()

    if not font_paths:
        print("警告：没有找到 .ttf 字体文件。")
        print("将使用 PIL 内置字体。")

        default_font = ImageFont.load_default()

        return [default_font], [default_font]

    preferred_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/mnt/c/Windows/Fonts/arial.ttf",
        "/mnt/c/Windows/Fonts/arialbd.ttf",
        "/mnt/c/Windows/Fonts/verdana.ttf",
        "/mnt/c/Windows/Fonts/verdanab.ttf",
    ]

    font_paths = [
        path
        for path in preferred_fonts
        if path in font_paths
    ] + [
        path
        for path in font_paths
        if path not in preferred_fonts
    ]

    normal_fonts = []
    supported_font_paths = []

    def font_supports_all_chars(font):
        draw = ImageDraw.Draw(
            Image.new("L", (1, 1))
        )
        for ch in CHARACTERS:
            mask = font.getmask(ch)
            if mask.getbbox() is None:
                return False
            bbox = draw.textbbox((0, 0), ch, font=font)
            if bbox is None or bbox[2] == 0 or bbox[3] == 0:
                return False
        return True

    for font_path in font_paths:
        if len(normal_fonts) >= 10:
            break

        try:
            font = ImageFont.truetype(
                font_path,
                FONT_SIZE,
            )

            if not font_supports_all_chars(font):
                raise ValueError(
                    "字体不支持全部目标字符"
                )

            normal_fonts.append(font)
            supported_font_paths.append(font_path)

        except Exception as error:
            print(
                f"跳过无法加载的字体：{font_path}"
            )
            print(f"原因：{error}")

    if not normal_fonts:
        print("字体加载失败，将使用 PIL 内置字体。")

        default_font = ImageFont.load_default()

        return [default_font], [default_font]

    bold_paths = [
        path
        for path in supported_font_paths
        if "bold" in path.lower()
    ]

    if not bold_paths:
        bold_paths = supported_font_paths

    bold_fonts = []

    for font_path in bold_paths:
        try:
            bold_fonts.append(
                ImageFont.truetype(
                    font_path,
                    FONT_SIZE,
                )
            )
        except Exception:
            pass

    if not bold_fonts:
        bold_fonts = normal_fonts

    return normal_fonts, bold_fonts

fonts_normal, fonts_bold = load_fonts()

# ============================================================
# 裁剪图片四周黑色空白
# ============================================================

def crop_to_content(image, padding=0):
    """
    裁剪字符周围的黑色空白。

    padding：
        裁剪后额外保留的黑色边缘像素。
    """

    bbox = image.getbbox()

    if bbox is None:
        return image

    left, top, right, bottom = bbox

    left = max(0, left - padding)
    top = max(0, top - padding)

    right = min(
        image.width,
        right + padding,
    )

    bottom = min(
        image.height,
        bottom + padding,
    )

    return image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

# ============================================================
# 对单个字符进行仿射扭曲
# ============================================================

def distort_char_image(char_img):
    """
    对单个字符进行仿射扭曲。

    参数参考 generate.txt：

        shear_x: -0.25 到 0.25
        shear_y: -0.15 到 0.15

    变形前后都增加和保留黑色边缘，
    避免字符笔画被裁剪。
    """

    # 变形前增加边缘空间
    border = max(
        10,
        int(max(char_img.size) * 0.20),
    )

    padded = Image.new(
        "L",
        (
            char_img.width + border * 2,
            char_img.height + border * 2,
        ),
        color=0,
    )

    padded.paste(
        char_img,
        (border, border),
    )

    width, height = padded.size

    # 与 generate.txt 相同的仿射扭曲范围
    shear_x = random.uniform(
        -0.25,
        0.25,
    )

    shear_y = random.uniform(
        -0.15,
        0.15,
    )

    affine_data = (
        1.0,
        shear_x,
        -shear_x * height / 2,

        shear_y,
        1.0,
        -shear_y * width / 2,
    )

    distorted = padded.transform(
        (width, height),
        Image.Transform.AFFINE,
        affine_data,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )

    # 扭曲后重新裁剪，避免产生大量空白
    distorted = crop_to_content(
        distorted,
        padding=2,
    )

    return distorted

# ============================================================
# 生成单个字符图片
# ============================================================

def generate_single_char(char):
    """
    生成一个清晰、完整、经过随机变形的字符。

    处理流程：

        高分辨率绘制
            ↓
        自动裁剪空白
            ↓
        仿射扭曲
            ↓
        再次裁剪
            ↓
        放大到槽位比例
            ↓
        随机宽高变形
            ↓
        旋转
            ↓
        返回字符图片
    """

    # 与 generate.txt 一样，使用较大的临时画布
    temp_size = 256

    temp_img = Image.new(
        "L",
        (
            temp_size,
            temp_size,
        ),
        color=0,
    )

    temp_draw = ImageDraw.Draw(temp_img)

    # 40% 概率使用粗体字体
    if random.random() < 0.40:
        font = random.choice(fonts_bold)
    else:
        font = random.choice(fonts_normal)

    # 获取字符实际边界
    bbox = temp_draw.textbbox(
        (0, 0),
        char,
        font=font,
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 将字符绘制到临时画布中央
    draw_x = (
        temp_size - text_width
    ) // 2 - bbox[0]

    draw_y = (
        temp_size - text_height
    ) // 2 - bbox[1]

    temp_draw.text(
        (
            draw_x,
            draw_y,
        ),
        char,
        fill=255,
        font=font,
    )

    # 第一次裁剪
    char_img = crop_to_content(
        temp_img,
        padding=2,
    )

    if char_img.getbbox() is None:
        return Image.new(
            "L",
            (
                CHAR_SLOT_SIZE,
                CHAR_SLOT_SIZE,
            ),
            color=0,
        )

    # 随机仿射扭曲
    if (
        ENABLE_DISTORTION
        and random.random()
        < DISTORTION_PROBABILITY
    ):
        char_img = distort_char_image(
            char_img
        )

    # 扭曲后再次裁剪
    char_img = crop_to_content(
        char_img,
        padding=2,
    )

    # --------------------------------------------------------
    # 按比例缩放字符
    # --------------------------------------------------------

    target_size = int(
        CHAR_SLOT_SIZE * CHAR_SIZE_RATIO
    )

    char_width, char_height = char_img.size

    scale = target_size / max(
        char_width,
        char_height,
    )

    new_width = max(
        1,
        int(char_width * scale),
    )

    new_height = max(
        1,
        int(char_height * scale),
    )

    char_img = char_img.resize(
        (
            new_width,
            new_height,
        ),
        resample=Image.Resampling.LANCZOS,
    )

    # --------------------------------------------------------
    # 随机宽高变形
    # 参考 generate.txt：
    #
    # scale_x = 0.9 到 1.1
    # scale_y = 0.8 到 1.2
    # --------------------------------------------------------

    scale_x = random.uniform(
        0.90,
        1.10,
    )

    scale_y = random.uniform(
        0.80,
        1.20,
    )

    new_width = max(
        1,
        int(char_img.width * scale_x),
    )

    new_height = max(
        1,
        int(char_img.height * scale_y),
    )

    char_img = char_img.resize(
        (
            new_width,
            new_height,
        ),
        resample=Image.Resampling.BICUBIC,
    )

    # 防止宽高变形后超出字符槽位
    if (
        char_img.width > target_size
        or char_img.height > target_size
    ):
        scale_down = min(
            target_size / char_img.width,
            target_size / char_img.height,
        )

        char_img = char_img.resize(
            (
                max(
                    1,
                    int(
                        char_img.width
                        * scale_down
                    ),
                ),
                max(
                    1,
                    int(
                        char_img.height
                        * scale_down
                    ),
                ),
            ),
            resample=Image.Resampling.LANCZOS,
        )

    # --------------------------------------------------------
    # 随机旋转
    # --------------------------------------------------------

    rotation_canvas = Image.new(
        "L",
        (
            CHAR_SLOT_SIZE * 2,
            CHAR_SLOT_SIZE * 2,
        ),
        color=0,
    )

    paste_x = (
        rotation_canvas.width
        - char_img.width
    ) // 2

    paste_y = (
        rotation_canvas.height
        - char_img.height
    ) // 2

    rotation_canvas.paste(
        char_img,
        (
            paste_x,
            paste_y,
        ),
    )

    angle = random.uniform(
        -MAX_ROTATION,
        MAX_ROTATION,
    )

    rotation_canvas = rotation_canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=0,
    )

    # 旋转后从中心裁剪回单字符槽位
    left = (
        rotation_canvas.width
        - CHAR_SLOT_SIZE
    ) // 2

    top = (
        rotation_canvas.height
        - CHAR_SLOT_SIZE
    ) // 2

    char_img = rotation_canvas.crop(
        (
            left,
            top,
            left + CHAR_SLOT_SIZE,
            top + CHAR_SLOT_SIZE,
        )
    )

    return char_img

# ============================================================
# 将字符安全粘贴到完整验证码图片
# ============================================================

def paste_char_safely(
    background,
    char_img,
    x,
    y,
):
    """
    将字符粘贴到背景中。

    如果字符部分超出图片边界，则自动裁剪，
    避免出现负坐标或越界问题。

    使用最大值混合，防止字符互相覆盖擦除。
    """

    bg_width, bg_height = background.size
    char_width, char_height = char_img.size

    # 计算字符和背景的交集区域
    src_left = max(0, -x)
    src_top = max(0, -y)

    dst_left = max(0, x)
    dst_top = max(0, y)

    visible_width = min(
        char_width - src_left,
        bg_width - dst_left,
    )

    visible_height = min(
        char_height - src_top,
        bg_height - dst_top,
    )

    if (
        visible_width <= 0
        or visible_height <= 0
    ):
        return background

    src_crop = char_img.crop(
        (
            src_left,
            src_top,
            src_left + visible_width,
            src_top + visible_height,
        )
    )

    bg_crop = background.crop(
        (
            dst_left,
            dst_top,
            dst_left + visible_width,
            dst_top + visible_height,
        )
    )

    blended = Image.fromarray(
        np.maximum(
            np.array(bg_crop),
            np.array(src_crop),
        ).astype(np.uint8)
    )

    background.paste(
        blended,
        (
            dst_left,
            dst_top,
        ),
    )

    return background

# ============================================================
# 波浪扭曲
# ============================================================

def apply_wave_distortion(
    image,
    amplitude=None,
    frequency=None,
):
    """
    对整张验证码进行温和波浪变形。

    与原版本相比，边界不使用 np.roll，
    避免图片左右边缘出现循环残影。
    """

    if amplitude is None:
        amplitude = random.uniform(
            1.5,
            4.0,
        )

    if frequency is None:
        frequency = random.uniform(
            0.10,
            0.25,
        )

    arr = np.array(image)
    height, width = arr.shape

    output = np.zeros_like(arr)

    phase = random.uniform(
        0,
        2 * math.pi,
    )

    for y in range(height):

        shift = int(
            amplitude
            * math.sin(
                2 * math.pi
                * frequency
                * y
                / height
                + phase
            )
        )

        if shift > 0:
            output[y, shift:] = arr[y, :-shift]

        elif shift < 0:
            output[y, :shift] = arr[y, -shift:]

        else:
            output[y] = arr[y]

    return Image.fromarray(
        output
    )

# ============================================================
# 添加网格线
# ============================================================

def apply_grid_lines(
    image,
    spacing_x=None,
    spacing_y=None,
    line_intensity=None,
):
    """添加轻微网格干扰。"""

    if spacing_x is None:
        spacing_x  = random.randint(
            30,
            40,
        )

    if spacing_y is None:
        spacing_y = random.randint(
            30,
            40,
        )

    if line_intensity is None:
        line_intensity = random.randint(
            35,
            80,
        )

    arr = np.array(
        image
    ).copy()

    height, width = arr.shape

    for y in range(
        0,
        height,
        spacing_y,
    ):
        sign = (
            1
            if random.random() < 0.5
            else -1
        )

        arr[y, :] = np.clip(
            arr[y, :].astype(int)
            + sign * line_intensity,
            0,
            255,
        ).astype(np.uint8)

    for x in range(
        0,
        width,
        spacing_x,
    ):
        sign = (
            1
            if random.random() < 0.5
            else -1
        )

        arr[:, x] = np.clip(
            arr[:, x].astype(int)
            + sign * line_intensity,
            0,
            255,
        ).astype(np.uint8)

    return Image.fromarray(
        arr
    )

# ============================================================
# 添加干扰线
# ============================================================

def apply_noise_lines(
    image,
    num_lines=None,
):
    """添加少量随机干扰线。"""

    if num_lines is None:
        num_lines = random.randint(
            1,
            3,
        )

    result = image.copy()
    draw = ImageDraw.Draw(result)

    width, height = result.size

    for _ in range(num_lines):

        x1 = random.randint(
            0,
            width - 1,
        )

        y1 = random.randint(
            0,
            height - 1,
        )

        x2 = random.randint(
            0,
            width - 1,
        )

        y2 = random.randint(
            0,
            height - 1,
        )

        draw.line(
            [
                (x1, y1),
                (x2, y2),
            ],
            fill=random.randint(
                10,
                60,
            ),
            width=random.randint(
                1,
                2,
            ),
        )

    return result

# ============================================================
# 添加随机噪声点
# ============================================================

def apply_random_noise_dots(
    image,
    max_dots=20,
):
    """添加少量像素噪声。"""

    arr = np.array(
        image
    ).copy()

    height, width = arr.shape

    dot_count = random.randint(
        0,
        max_dots,
    )

    for _ in range(dot_count):

        x = random.randint(
            0,
            width - 1,
        )

        y = random.randint(
            0,
            height - 1,
        )

        arr[y, x] = random.randint(
            60,
            120,
        )

    return Image.fromarray(
        arr
    )

# ============================================================
# 生成一张完整验证码
# ============================================================

def generate_captcha_image(label):
    """
    生成完整验证码图片。

    返回：

        image:
            最终验证码图片。

        boxes:
            每个字符的坐标框。
    """

    image = Image.new(
        "L",
        (
            IMG_WIDTH,
            IMG_HEIGHT,
        ),
        color=0,
    )

    boxes = []

    slot_width = IMG_WIDTH / CAPTCHA_LENGTH

    for index, char in enumerate(label):

        # 生成单个字符
        char_img = generate_single_char(
            char
        )

        # 字符的基础位置
        base_x = int(
            index * slot_width
            + (
                slot_width
                - CHAR_SLOT_SIZE
            ) / 2
        )

        base_y = int(
            (IMG_HEIGHT - CHAR_SLOT_SIZE)
            / 2
        )

        # 参考 generate.txt 添加轻微随机偏移
        jitter_x = random.randint(
            -3,
            3,
        )

        jitter_y = random.randint(
            -4,
            4,
        )

        paste_x = base_x + jitter_x
        paste_y = base_y + jitter_y

        # 限制字符框位置，避免大量越界
        paste_x = max(
            -2,
            min(
                paste_x,
                IMG_WIDTH - CHAR_SLOT_SIZE + 2,
            ),
        )

        paste_y = max(
            -2,
            min(
                paste_y,
                IMG_HEIGHT - CHAR_SLOT_SIZE + 2,
            ),
        )

        image = paste_char_safely(
            image,
            char_img,
            paste_x,
            paste_y,
        )

        # 坐标框限制在最终图片范围内
        box_left = max(
            0,
            paste_x,
        )

        box_top = max(
            0,
            paste_y,
        )

        box_right = min(
            IMG_WIDTH,
            paste_x + CHAR_SLOT_SIZE,
        )

        box_bottom = min(
            IMG_HEIGHT,
            paste_y + CHAR_SLOT_SIZE,
        )

        boxes.append(
            [
                box_left,
                box_top,
                box_right,
                box_bottom,
            ]
        )

    # --------------------------------------------------------
    # 整张图片添加轻微波浪扭曲
    # --------------------------------------------------------

    if (
        ENABLE_WAVE_DISTORTION
        and random.random()
        < WAVE_PROBABILITY
    ):
        image = apply_wave_distortion(
            image
        )

    # --------------------------------------------------------
    # 添加轻微网格线
    # --------------------------------------------------------

    if random.random() < GRID_LINE_PROBABILITY:
        image = apply_grid_lines(
            image
        )

    # --------------------------------------------------------
    # 添加少量干扰线
    # --------------------------------------------------------

    if random.random() < NOISE_LINE_PROBABILITY:
        image = apply_noise_lines(
            image,
            num_lines=random.randint(
                1,
                3,
            ),
        )

    # --------------------------------------------------------
    # 添加轻微模糊
    # --------------------------------------------------------

    if random.random() < BLUR_PROBABILITY:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(
                    0.15,
                    0.35,
                )
            )
        )

    # --------------------------------------------------------
    # 添加少量噪声点
    # --------------------------------------------------------

    image = apply_random_noise_dots(
        image,
        max_dots=20,
    )

    return image, boxes

# ============================================================
# 生成数据集
# ============================================================

def main():

    # 确保输出目录存在
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    metadata = {}

    print(
        f"开始生成 {TOTAL_IMAGES} 张验证码图片"
    )

    print(
        f"验证码长度：{CAPTCHA_LENGTH}"
    )

    print(
        f"图片尺寸：{IMG_WIDTH}×{IMG_HEIGHT}"
    )

    print(
        f"字符尺寸：约 "
        f"{int(CHAR_SLOT_SIZE * CHAR_SIZE_RATIO)}×"
        f"{int(CHAR_SLOT_SIZE * CHAR_SIZE_RATIO)}"
    )

    for index in range(TOTAL_IMAGES):

        # 随机生成 5 字符标签
        label = "".join(
            random.choice(CHARACTERS)
            for _ in range(CAPTCHA_LENGTH)
        )

        image, boxes = generate_captcha_image(
            label
        )

        filename = (
            f"{label}_{index:05d}.png"
        )

        filepath = os.path.join(
            OUTPUT_DIR,
            filename,
        )

        image.save(
            filepath,
            format="PNG",
        )

        metadata[filename] = {
            "label": label,
            "boxes": boxes,
        }

        if (index + 1) % 1000 == 0:
            print(
                f"已生成 "
                f"{index + 1}/{TOTAL_IMAGES}"
            )

    # 保存标签和字符坐标
    metadata_path = os.path.join(
        OUTPUT_DIR,
        "boxes.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "生成完成：",
        os.path.abspath(OUTPUT_DIR),
    )

    print(
        "元数据文件：",
        os.path.abspath(metadata_path),
    )

if __name__ == "__main__":
    main()