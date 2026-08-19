"""
生成单字符图片数据集：0-9、A-Z、a-z。

目录结构：

    data/
    ├── digit_0/
    ├── upper_A/
    ├── lower_a/
    └── class_mapping.txt
"""

import glob
import os
import random
import string

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ============================================================
# 配置
# ============================================================

OUTPUT_DIR = "./data"

# 62 个类别，每类 10 张
TOTAL_IMAGES = 8000

# 最终输出图片尺寸
IMAGE_SIZE = 64

# 字体大小
FONT_SIZE = 40

# 最大旋转角度
MAX_ROTATION = 20

# 字符最长边占最终图片的比例
# 建议范围：0.78 到 0.88
CHAR_SIZE_RATIO = 0.82

# 是否启用扭曲
ENABLE_DISTORTION = True

# 扭曲概率
DISTORTION_PROBABILITY = 0.70

CHARACTERS = list(
    string.digits
    + string.ascii_uppercase
    + string.ascii_lowercase
)

# ============================================================
# 类别名称
# ============================================================

def get_class_name(char):
    """将字符转换成安全的类别目录名。"""

    if char.isdigit():
        return f"digit_{char}"

    if char.isupper():
        return f"upper_{char}"

    return f"lower_{char}"

# ============================================================
# 查找字体
# ============================================================

def find_fonts():
    """查找系统中的 TrueType 字体。"""

    font_paths = []

    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/mnt/c/Windows/Fonts",
    ]

    for directory in search_dirs:
        if os.path.exists(directory):
            font_paths.extend(
                glob.glob(
                    os.path.join(directory, "**", "*.ttf"),
                    recursive=True,
                )
            )

    return font_paths

# ============================================================
# 获取字符有效区域
# ============================================================

def crop_to_content(image, padding=0):
    """
    裁剪图片四周的黑色空白区域。

    padding 表示裁剪后额外保留的黑色边缘。
    """

    bbox = image.getbbox()

    if bbox is None:
        return image

    left, top, right, bottom = bbox

    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)

    return image.crop(
        (left, top, right, bottom)
    )

# ============================================================
# 安全的字符扭曲
# ============================================================

def distort_char_image(char_img):
    """
    对字符进行温和的仿射扭曲。

    与随机 QUAD 透视变形相比，这种方式不容易把字符
    的局部拉到图片外面。
    """

    # 变形前增加黑色边缘，防止字符贴近边界
    border = max(12, int(max(char_img.size) * 0.20))

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

    # 温和的水平、垂直倾斜
    shear_x = random.uniform(-0.25, 0.25)
    shear_y = random.uniform(-0.15, 0.15)

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

    # 重新裁剪字符内容，并保留少量边缘
    distorted = crop_to_content(
        distorted,
        padding=3,
    )

    return distorted

# ============================================================
# 生成单张字符图片
# ============================================================

def generate_char_image(char):
    """生成一张清晰、完整的单字符图片。"""

    # 使用高分辨率临时画布
    temp_size = 256

    temp_img = Image.new(
        "L",
        (temp_size, temp_size),
        color=0,
    )

    temp_draw = ImageDraw.Draw(temp_img)

    font = random.choice(fonts)

    # 获取字符边界框
    bbox = temp_draw.textbbox(
        (0, 0),
        char,
        font=font,
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 将字符绘制到画布中央
    draw_x = (temp_size - text_width) // 2 - bbox[0]
    draw_y = (temp_size - text_height) // 2 - bbox[1]

    temp_draw.text(
        (draw_x, draw_y),
        char,
        fill=255,
        font=font,
    )

    # 裁剪字符周围的空白
    char_img = crop_to_content(
        temp_img,
        padding=2,
    )

    if char_img.getbbox() is None:
        return Image.new(
            "L",
            (IMAGE_SIZE, IMAGE_SIZE),
            color=0,
        )

    # 随机扭曲
    if (
        ENABLE_DISTORTION
        and random.random() < DISTORTION_PROBABILITY
    ):
        char_img = distort_char_image(char_img)

    # 重新裁剪，避免扭曲后产生过多空白
    char_img = crop_to_content(
        char_img,
        padding=2,
    )

    # --------------------------------------------------------
    # 放大字符
    # --------------------------------------------------------

    target_size = int(
        IMAGE_SIZE * CHAR_SIZE_RATIO
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
        (new_width, new_height),
        resample=Image.Resampling.LANCZOS,
    )

    # --------------------------------------------------------
    # 随机宽高变形
    # --------------------------------------------------------

    scale_x = random.uniform(0.9, 1.1)
    scale_y = random.uniform(0.8, 1.2)

    new_width = max(
        1,
        int(char_img.width * scale_x),
    )

    new_height = max(
        1,
        int(char_img.height * scale_y),
    )

    char_img = char_img.resize(
        (new_width, new_height),
        resample=Image.Resampling.BICUBIC,
    )

    # 防止宽高变形后超出安全尺寸
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
                max(1, int(char_img.width * scale_down)),
                max(1, int(char_img.height * scale_down)),
            ),
            resample=Image.Resampling.LANCZOS,
        )

    # --------------------------------------------------------
    # 放到更大的画布中央
    # --------------------------------------------------------

    canvas_size = IMAGE_SIZE * 2

    canvas = Image.new(
        "L",
        (canvas_size, canvas_size),
        color=0,
    )

    paste_x = (
        canvas_size - char_img.width
    ) // 2

    paste_y = (
        canvas_size - char_img.height
    ) // 2

    # 不再使用 char_img 作为 mask，
    # 避免边缘灰度被重复衰减
    canvas.paste(
        char_img,
        (paste_x, paste_y),
    )

    # --------------------------------------------------------
    # 随机旋转
    # --------------------------------------------------------

    angle = random.uniform(
        -MAX_ROTATION,
        MAX_ROTATION,
    )

    canvas = canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=0,
    )

    # --------------------------------------------------------
    # 可选轻微模糊
    # --------------------------------------------------------

    if random.random() < 0.12:
        canvas = canvas.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.1, 0.3),
            )
        )

    # --------------------------------------------------------
    # 中心裁剪为 64×64
    # --------------------------------------------------------

    left = (
        canvas_size - IMAGE_SIZE
    ) // 2

    top = (
        canvas_size - IMAGE_SIZE
    ) // 2

    image = canvas.crop(
        (
            left,
            top,
            left + IMAGE_SIZE,
            top + IMAGE_SIZE,
        )
    )

    # --------------------------------------------------------
    # 添加极少量噪声
    # --------------------------------------------------------

    pixels = image.load()

    for _ in range(random.randint(0, 5)):
        x = random.randint(
            0,
            IMAGE_SIZE - 1,
        )

        y = random.randint(
            0,
            IMAGE_SIZE - 1,
        )

        # 噪声不要太强，避免影响字符清晰度
        pixels[x, y] = random.randint(80, 180)

    return image

# ============================================================
# 初始化字体
# ============================================================

REQUIRED_CHARACTERS = set(CHARACTERS)

font_paths = find_fonts()

preferred_fonts = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/mnt/c/Windows/Fonts/arial.ttf",
    "/mnt/c/Windows/Fonts/arialbd.ttf",
    "/mnt/c/Windows/Fonts/trebuc.ttf",
    "/mnt/c/Windows/Fonts/verdana.ttf",
]

# 将首选字体路径排在前面
font_paths = [
    path
    for path in preferred_fonts
    if path in font_paths
] + [
    path
    for path in font_paths
    if path not in preferred_fonts
]

if not font_paths:
    print("警告：没有找到 .ttf 字体文件。")
    print("将使用 PIL 内置字体。")

    fonts = [
        ImageFont.load_default()
    ]

else:
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

    print(
        f"找到 {len(font_paths)} 个字体文件，"
        "将从中筛选可用字体。"
    )

    fonts = []

    def supports_all_chars(font):
        draw = ImageDraw.Draw(
            Image.new("L", (1, 1))
        )
        for ch in REQUIRED_CHARACTERS:
            mask = font.getmask(ch)
            if mask.getbbox() is None:
                return False
            bbox = draw.textbbox((0, 0), ch, font=font)
            if bbox is None or bbox[2] == 0 or bbox[3] == 0:
                return False
        return True

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(
                font_path,
                FONT_SIZE,
            )

            if not supports_all_chars(font):
                raise ValueError("字体不支持全部目标字符")

            fonts.append(font)

            # 如果已经找到了足够多的好字体，就可以停止筛选
            if len(fonts) >= 10:
                break

        except Exception as error:
            print(
                f"跳过字体：{font_path}，原因：{error}"
            )

    if not fonts:
        print("没有可用字体，将使用 PIL 内置字体。")

        fonts = [
            ImageFont.load_default()
        ]

# ============================================================
# 计算图片数量
# ============================================================

images_per_class = (
    TOTAL_IMAGES // len(CHARACTERS)
)

actual_total = (
    images_per_class * len(CHARACTERS)
)

print()
print("字符数量：", len(CHARACTERS))
print("字符列表：", "".join(CHARACTERS))
print(f"每个类别生成：{images_per_class} 张")
print(f"预计生成总数：{actual_total} 张")
print(
    "输出目录：",
    os.path.abspath(OUTPUT_DIR),
)
print()

# ============================================================
# 生成数据集
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

mapping_path = os.path.join(
    OUTPUT_DIR,
    "class_mapping.txt",
)

with open(
    mapping_path,
    "w",
    encoding="utf-8",
) as mapping_file:

    mapping_file.write(
        "class_name\tcharacter\tclass_index\n"
    )

    for class_index, char in enumerate(CHARACTERS):

        class_name = get_class_name(char)

        class_dir = os.path.join(
            OUTPUT_DIR,
            class_name,
        )

        os.makedirs(
            class_dir,
            exist_ok=True,
        )

        mapping_file.write(
            f"{class_name}\t{char}\t{class_index}\n"
        )

        for i in range(images_per_class):
            image = generate_char_image(char)

            filename = os.path.join(
                class_dir,
                f"{class_name}_{i:05d}.png",
            )

            image.save(filename)

        print(
            f"Generated {images_per_class} images "
            f"for class {repr(char)} -> {class_name}"
        )

print()
print(
    "Done. Dataset saved to:",
    os.path.abspath(OUTPUT_DIR),
)
print(
    "类别映射文件：",
    os.path.abspath(mapping_path),
)
print(
    f"实际生成图片总数：{actual_total}"
)