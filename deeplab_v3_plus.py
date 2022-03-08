"""DeepLabV3+ 模型，用于做 cityscapes 数据集的 segmentation 任务。"""

import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Cityscapes 的图片高度宽度实际大小为 1024, 2048，比例为 1:2，所以这里输入给模型的图片
# 也保持这个 1: 2 的比例，以减少等比例缩放图片产生的空白区域。
# model_image_height 需要是 320, 400, 512 等数。
model_image_height = 320
MODEL_IMAGE_SIZE = model_image_height, 2 * model_image_height


def get_images_list(images_dir):
    """从指定文件夹中找出所有图片，并返回一个图片列表。

    Arguments:
        images_dir: 一个字符串，是所有图片存放的路径。

    Returns:
        images_list: 一个列表，包含了文件夹内所有图片的名字，并且每个图片的名字都是其完整
            的文件路径。
    """

    images_list = []
    # images_list 里面的每个图片文件，应该包含其文件路径，因为后续 tf.dataset 要
    # 用到文件路径。

    for dir_path, dirnames, images_name in os.walk(images_dir):
        # os.walk 会进入到每一个文件夹，返回3个元素：当前文件夹的路径 dir_path，当前文件
        # 夹下的子文件夹 dirnames，当前文件夹内所有图片的名称 images_name。
        images_with_path = []
        for image_name in images_name:
            image_with_path = os.path.join(dir_path, image_name)
            images_with_path.append(image_with_path)

        images_list += images_with_path

    return images_list


def get_labels_list(labels_dir):
    """从指定文件夹中找出所有的标签，并返回一个标签列表。

    Arguments:
        labels_dir: 一个字符串，是所有图片标签文件存放的路径。

    Returns:
        labels_list: 一个列表，包含了文件夹内所有图片标签的名字，并且每个标签的名字都是其
            完整的文件路径。
    """

    labels_list = []

    for dir_path, _, labels_name in os.walk(labels_dir):
        # 文件夹内有多种标签，只提取 semantic segmentation 标签。
        segmentation_label = []
        for label_name in labels_name:
            if label_name.endswith('labelIds.png'):
                label_with_path = os.path.join(dir_path, label_name)
                segmentation_label.append(label_with_path)

        labels_list += segmentation_label

    return labels_list


def create_masks_list(images_list, labels_list):
    """根据输入的图片名字列表，创建一一对应的标签名字列表。因为每个图片有 3 种雾化效果，（即
    3 种 beta 值），所以会有 3 个标签对应同一个图片。

    Arguments:
        images_list: 一个列表，包含了所有图片的名字，并且每个图片的名字都是其完整的文件
            路径。
        labels_list: 一个列表，包含了 images_list 内所有图片对应标签的名字，并且每个标
            签的名字都是其完整的文件路径。标签列表大小是图片列表大小的 ⅓ 。

    Returns:
        masks_list: 一个列表，包含了图片列表内，所有图片对应的标签。等于把 labels_list
            扩大 3 倍，然后和 images_list 进行一一对应的结果。
    """

    masks_list = []
    # ========================做一个进度条===================================
    progbar = keras.utils.Progbar(len(images_list), width=50, interval=1,
                                  unit_name="step")
    progbar_counter = 0
    print('Creating the masks_list ...')
    # ========================做一个进度条===================================
    for image in images_list:
        progbar.update(progbar_counter)
        progbar_counter += 1

        image_name = os.path.basename(image)
        # image_name_starters 是图片名字的前几位，例如 'aachen_000000_000019'。
        image_name_starters = image_name.split('_leftImg8bit')[0]
        for label in labels_list:
            label_name = os.path.basename(label)
            if label_name.startswith(image_name_starters):
                masks_list.append(label)

    return masks_list


def read_image(image_path, mask=False):
    """返回一个 float32 格式的 mask 张量，或是一个图片张量。

    Arguments:
        image_path: 一个字符串，是一张图片的完整文件路径。
        mask: 一个布尔值，如果为 True，表示将返回一个 mask 张量。

    Returns:
        image: 一个 float32 型张量，表示一张图片，或是一张图片的标签 mask。
    """

    image = tf.io.read_file(image_path)
    if mask:
        # 解码图片张量，默认转换为 tf.uint8 格式张量，channels=1 表示灰度图。
        image = tf.image.decode_png(image, channels=1)
        # 检查张量形状， TF 推荐使用 ensure_shape 而非 set_shape。
        tf.ensure_shape(image, [None, None, 1])
        # resize 改变图片大小，同时会把图片张量转换为 float32 格式。注意 resize 中
        # 不要设置  preserve_aspect_ratio=True，否则模型训练时形状无法匹配。
        # 必须设置 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR ！这样 mask 的值
        # 才不会发生变化！否则将使用默认的 bilinear 差值方式。
        image = tf.image.resize(images=image, size=MODEL_IMAGE_SIZE,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # mask的张量中，数值实际为整数，代表某个类别。
        image = tf.cast(image, dtype=tf.float32)

    else:
        # channels=3 表示 RGB 图。
        image = tf.image.decode_png(image, channels=3)
        tf.ensure_shape(image, [None, None, 3])
        # image 也配合 mask 使用 NEAREST_NEIGHBOR 插值方法，效果应该会更好。
        image = tf.image.resize(images=image, size=MODEL_IMAGE_SIZE,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 当使用 NEAREST_NEIGHBOR 插值法时，resize 返回的数据类型是 uint8。虽然下面
        # image 参与了计算 /127.5，但是它不会自动从 uint8 变为 float32 类型。所以要转
        # 换一下。
        image = tf.cast(image, dtype=tf.float32)
        image = image / 127.5 - 1  # 像素被限制在 [-1, 1] 之间

    return image


def load_data(image_path, mask_path):
    """输入的是 1 个图片及其对应 mask 的地址，将返回该图片张量及其 mask 张量。

    Arguments:
        image_path: 一个字符串，是一张图片的完整文件路径。
        mask_path: 一个字符串，是该图片对应 mask 的完整文件路径。

    Returns:
        image: 一个 float32 型张量，表示输入的图片。
        mask: 一个 float32 型张量，表示输入图片的标签 mask。
    """

    image = read_image(image_path)
    mask = read_image(mask_path, mask=True)

    return image, mask


def image_mask_dataset(images_list, masks_list, batch_size):
    """该函数将返回一个 tf.data.Dataset 对象，代表一个批次的图片张量和 mask 张量。

    Arguments:
        images_list: 一个列表，包含了所有图片的名字，并且每个图片的名字都是其完整的文件
            路径。
        masks_list: 一个列表，包含了图片列表内，和所有图片对应一一对应的标签 mask。
        batch_size: 一个整数，表示训练 Keras 模型时使用数据的批次大小。

    Returns:
        dataset: 一个 tf.data.Dataset 对象，包含图片和标签 2 个 elements，形状分别为
            (4, 512, 512, 3)，(4, 512, 512, 1)，数据类型为 tf.float32。该 dataset
            可以直接作为 Keras 的 fit 函数输入数据。
    """

    # from_tensor_slices 将根据输入的切片，生成 dataset。因为此时输入的是图片和 mask 的
    # 2 个列表，所以此时 dataset 的结果是一对标量，完整结果如下：
    # <TensorSliceDataset shapes: ((), ()), types: (tf.string, tf.string)>。
    dataset = tf.data.Dataset.from_tensor_slices((images_list, masks_list))

    # map 方法，对 dataset 使用 load_data 函数，返回结果为：
    # <ParallelMapDataset shapes: ((512, 512, 3),(512, 1024, 1)),
    # types: (tf.float32, tf.float32)>，（假定输入图片大小为 (512, 1024, 3)）。
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    # batch 用于产生批量数据。drop_remainder 为 True 表示如果最后一批的数量小于
    # batch_size，则丢弃最后一批。输出结果为：
    # <BatchDataset shapes: ((batch_size, 512, 1024, 3),
    # (batch_size, 512, 1024, 1)), types: (tf.float32, tf.float32)>
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # prefetch 用于提前取出若干 batch 数据，使得模型无须等待数据转换，加快模型训练速度。
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def convolution_block(block_input, num_filters=256, kernel_size=3,
                      dilation_rate=1, padding='same', use_bias=False,
                      separableconv=True):
    """这个卷积模块做 3 个操作：卷积，BN，Relu。当 dilation_rate 不等于1时，用放大
    卷积替换普通卷积。

    Arguments：
        inputs: 一个张量。数据类型为 float32。
        num_filters： 一个整数，是卷积模块的过滤器数量。
        kernel_size： 一个整数，是卷积核的大小。
        dilation_rate： 一个整数，设置放大卷积的放大倍率。默认为 1， 也就是不使用放大卷积。
        padding： 一个字符串，是卷积的 padding 方式。
        use_bias： 一个布尔值，只有在为 True 时卷积层才使用 bias 权重。
        separableconv： 一个布尔值，设置是否使用 Separableconv2D。

    Returns:
        x: 一个张量，经过卷积，Batch Normalization 和激活函数的处理，形状和输入张量相同。
    """

    if separableconv:
        x = keras.layers.SeparableConv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias,
            depthwise_initializer=keras.initializers.HeNormal(),
            pointwise_initializer=keras.initializers.HeNormal(),
        )(block_input)
    else:
        x = keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
        )(block_input)

    # 可以尝试使用 SpatialDropout2D，减轻过拟合。
    # x = keras.layers.SpatialDropout2D(rate=0.2)(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x


def dilated_spatial_pyramid_pooling(dspp_input):
    """放大卷积金字塔。

    放大卷积金字塔，是 DeepLabv3+ 的一个重要模块，也叫 Atrous Spatial Pyramid
    Pooling（简称 ASPP），位于编码器 encoder 的顶部。
    ASPP 借助 Atrous convolution（dilation convolution）使得模型有 5 种不同大小的感
    受域（3 个空洞卷积，1 个普通卷积，再加 1 个平均池化），将小的特征信息和大的特征信息进
    行混合，从而获得更好的场景信息 semantic information。
    """

    dims = dspp_input.shape
    x = keras.layers.AveragePooling2D(
        pool_size=(dims[-3], dims[-2]))(dspp_input)

    x = convolution_block(x, kernel_size=1, use_bias=True)

    out_pool = keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation='bilinear')(x)

    # 可以改动 dilation_rate，尝试不同的效果。
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = keras.layers.Concatenate(axis=-1)(
        [out_pool, out_1, out_6, out_12, out_18])

    output = convolution_block(x, kernel_size=1)

    return output


def inceptionresnetv2_deeplabv3plus(model_image_size, num_classes,
                                    rate_dropout=0.1):
    """使用 Inception-ResNet V2 作为 backbone，创建一个 DeepLab-V3+ 模型。

    Arguments：
        model_image_size: 一个整数型元祖，表示输入给模型的图片大小，格式为
            (height, width, depth)。注意对于 Inception-ResNet V2，要求输入图片的像
            素值必须转换到 [-1, 1] 之间。
            为了便于说明数组形状，这里假设模型输入图片的大小为 (512, 1024, 3)。
        num_classes： 一个整数，是模型中需要区分类别的数量。
        rate_dropout： 一个浮点数，范围是 [0, 1]，表示 SpatialDropout2D 层的比例。

    Returns:
        model: 一个以 Inception-ResNet V2 为脊柱的 DeepLab-V3+ 模型。
    """

    # 新建模型之前，先用 clear_session 把状态清零。
    keras.backend.clear_session()

    model_input = keras.Input(shape=(*model_image_size, 3))
    conv_base = keras.applications.InceptionResNetV2(
        include_top=False,
        input_tensor=model_input)

    # low_level_feature_backbone 形状为 (None, 124, 252, 192)。
    low_level_feature_backbone = conv_base.get_layer(
        'activation_4').output

    # 因为需要 low_level_feature_backbone的特征图为 128, 256, 所以要用 2次
    # Conv2DTranspose。
    for _ in range(2):
        low_level_feature_backbone = convolution_block(
            low_level_feature_backbone)
        low_level_feature_backbone = keras.layers.Conv2DTranspose(
            filters=256, kernel_size=3,
            kernel_initializer=keras.initializers.HeNormal())(
            low_level_feature_backbone)

    # low_level_feature_backbone 形状为 (None, 128, 256, 256)。
    low_level_feature_backbone = convolution_block(
        low_level_feature_backbone, num_filters=256, kernel_size=1)

    if rate_dropout != 0:
        low_level_feature_backbone = keras.layers.SpatialDropout2D(
            rate=rate_dropout)(low_level_feature_backbone)

    # encoder_backbone 形状为 (None, 30, 62, 1088)。
    encoder_backbone = conv_base.get_layer('block17_10_ac').output

    # 在特征图被放大之前，都算作 encoder 部分，因为这部分内容都是在图片进行理解，所以 DSPP
    # 也算作 encoder 部分。下面进行 DSPP 操作。
    # encoder_backbone 形状为 (None, 30, 62, 256)。
    encoder_backbone = dilated_spatial_pyramid_pooling(encoder_backbone)

    # 下面进入解码器 decoder 部分，开始放大特征图。
    # encoder_backbone 形状为 (None, 32, 64, 256)。
    decoder_backbone = keras.layers.Conv2DTranspose(
        256, kernel_size=3, kernel_initializer=keras.initializers.HeNormal())(
        encoder_backbone)
    # encoder_backbone 形状为 (None, 128, 256, 256)。
    decoder_backbone = keras.layers.UpSampling2D(
        size=(4, 4), interpolation='bilinear')(decoder_backbone)

    if rate_dropout != 0:
        decoder_backbone = keras.layers.SpatialDropout2D(rate=rate_dropout)(
            decoder_backbone)

    # x 形状为 (None, 128, 256, 512)。
    x = keras.layers.Concatenate()(
        [decoder_backbone, low_level_feature_backbone])

    # 下面进行2次卷积，将前面 2 个合并的分支信息进行处理，特征通道数量将变为 256。
    for _ in range(2):
        x = convolution_block(x)

        if rate_dropout != 0:
            x = keras.layers.SpatialDropout2D(rate=rate_dropout)(x)

    # x 形状为 (None, 512, 1024, 256)。
    x = keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    if rate_dropout != 0:
        x = keras.layers.SpatialDropout2D(rate=rate_dropout)(x)

    # 尝试增加一个分支 down_sampling_1，特征图大小和输入图片大小一样。目的是使得在预测结果
    # mask 中，物体的轮廓更准确。
    down_sampling_1 = model_input
    down_sampling_1 = convolution_block(down_sampling_1, num_filters=8,
                                        separableconv=False)
    for _ in range(2):
        down_sampling_1 = convolution_block(down_sampling_1, num_filters=8)

    if rate_dropout != 0:
        down_sampling_1 = keras.layers.SpatialDropout2D(rate=rate_dropout)(
            down_sampling_1)

    # x 形状为 (None, 512, 1024, 264)。
    x = keras.layers.Concatenate()([down_sampling_1, x])
    for _ in range(2):
        x = convolution_block(x, num_filters=64)

    # model_output 形状为 (None, 512, 1024, 34)。
    model_output = keras.layers.Conv2D(
        num_classes, kernel_size=(1, 1), padding='same')(x)

    model = keras.Model(inputs=model_input, outputs=model_output)

    return model


def infer(model, image_tensor):
    """使用模型进行推理。

    Arguments：
        model: 一个训练好的 DeepLab-V3+ 模型。
        image_tensor: 一个 float32 型 3D 图片张量，表示一张需要进行推理的图片，张量格
            式为(*MODEL_IMAGE_SIZE, 3)。
            为了便于说明数组形状，这里假设模型输入图片的大小为 (512, 1024, 3)。

    Returns:
        predictions: 一个 float32 型 2D 张量，表示模型对输入图片的推理结果，张量形状为
            (512, 1024)。
    """

    # predictions 的张量形状为 (1, 512, 1024, 34)。
    predictions = model.predict(np.expand_dims(image_tensor, axis=0))

    # predictions 的张量形状为 (512, 1024, 34)。
    predictions = np.squeeze(predictions)

    # argmax 将去掉最后一个维度，输出的张量形状为 (512, 1024)。
    predictions = np.argmax(predictions, axis=-1)

    return predictions


def decode_segmentation_masks(mask, colormap, num_classes):
    """将模型的推理结果 mask 转换为一个 RGB 格式的图片数组。

    Arguments：
        mask: 一个 float32 型 2D 张量，表示模型对输入图片的推理结果，张量形状为
            (*MODEL_IMAGE_SIZE)。张量内的每一个值代表一个类别，数值范围是 [0, 33]。
            为了便于说明数组形状，这里假设模型输入图片的大小为 (512, 1024, 3)，则 mask
            的形状将为 (512, 1024)。
        colormap: 一个 unit8 型 2D 数组，形状为(34, 3)。代表了每一个类别的颜色对应关系。
            数组中的第 0 维度大小为 34，表示有 34 个类别，第 1 维度大小为 3，表示每个类
            别的 RGB 3 个值。
        num_classes: 一个整数。代表分类任务需要区分的类别数量。对 Cityscapes 来说，这个
            num_classes 为 34 。

    Returns:
        rgb_mask: 一个 unit8 型 3D 数组，数组的形状为 (512, 1024, 3)。该数组是一个
            彩色图片，用各种颜色区分了图片中的各个类别。
    """

    # r，g，b 三个数组的形状均为 (512, 1024)。
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    for i in range(num_classes):

        # current_class 是一个布尔数组，形状为 (512, 1024)，属于当前类别的位置将为 True。
        current_class = (mask == i)

        # r_value，是 3 个整数，分别表示当前类别的 RGB 数值。
        r_value, g_value, b_value = colormap[i]

        # 对 mask 中属于当前类别的区域，涂上当前类别的颜色，也就是赋予 RGB 值。
        # r，g，b 三个数组的形状均为 (512, 1024)。
        r[current_class] = r_value
        g[current_class] = g_value
        b[current_class] = b_value

    # rgb_mask 形状为 (512, 1024, 3)。
    rgb_mask = np.stack([r, g, b], axis=2)

    return rgb_mask


def get_overlay(image_tensor, rgb_mask):
    """将两个输入的图片进行叠加，形成一个新的叠加图片。

    Arguments：
        image_tensor: 一个 float32 型 3D 图片张量，张量形状为(512, 1024, 3)。
            为了便于说明数组形状，这里假设模型输入图片的大小为 (512, 1024, 3)。
        rgb_mask: 一个 unit8 型 3D 数组，数组的形状为 (512, 1024, 3)。该数组是一个
            彩色图片，用各种颜色区分了图片中的各个类别。

    Returns:
        overlay: 一个 OpenCV 图片对象，大小为 (512, 1024, 3)。是输入的图片张量和
            rgb_mask 的叠加结果。
    """

    # 将 image_tensor 转换为一个 Pillow 图片对象。pil_image 数值范围是 [0, 255]。
    pil_image = keras.preprocessing.image.array_to_img(image_tensor)

    # 把图片从 float32 格式转换为 unit8 格式。image_array 形状为 (512, 1024, 3)。
    image_array = np.array(pil_image).astype(np.uint8)

    # addWeighted 用于将图片叠加。
    # 函数 signature：addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
    # 计算方式 dst = src1*alpha + src2*beta + gamma;
    overlay = cv2.addWeighted(image_array, 0.2, rgb_mask, 0.8, 0)

    return overlay


def plot_image_overlay_rgbmask(images_to_plot, figsize=(5, 3)):
    """将 images_to_plot 中的所有图片画在一个 figure 中。

    Arguments：
        images_to_plot: 一个列表，包含 3 个图片。
        figsize: 一个元祖，包含 2 个浮点数，表示这个 figure 的宽度和高度。
    """

    # subplots 可以生成 axes 对象，用于放入多个图。具体用法可参见下面链接。
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    _, ax = plt.subplots(nrows=1, ncols=len(images_to_plot), figsize=figsize)

    for i in range(len(images_to_plot)):
        each_image = images_to_plot[i]

        ax[i].matshow(keras.preprocessing.image.array_to_img(each_image))

        # ax[i].axis('off')  # 如果要关闭全部图像的坐标轴，用此行代码代替下面一行代码。
        plt.axis('off')

    plt.show()


def plot_predictions(images_list, colormap, model, num_classes):
    """遍历 images_list 中的每一张图片，对其进行推理得到推理 mask。然后把每一张输入图片和
     推理 mask 进行重叠得到重叠图片。最后，把这三者：原始图片，重叠图片，推理 mask，放在同
     一行画出来。

    Arguments：
        images_list: 一个列表，包含了所有图片的名字，并且每个图片的名字都是其完整的
            文件路径。
        colormap: 一个 unit8 型 2D 数组，形状为(34, 3)。代表了每一个类别的颜色对应关系。
            数组中的第 0 维度大小为 34，表示有 34 个类别，第 1 维度大小为 3，表示每个
            类别的 RGB 3 个值。
        model: 一个训练好的 DeepLabV3+ 模型。
        num_classes: 一个整数。代表分类任务需要区分的类别数量。对 Cityscapes 来说，这个
            num_classes 为 34 。
    """

    # 遍历 images_list， 读取每一张图片。
    for image_file in images_list:
        # image_tensor 形状为 (512, 1024, 3)。
        image_tensor = read_image(image_file)

        # 用模型对图片进行推理，得到推理结果 mask。inferred_mask 形状为 (512, 1024)。
        inferred_mask = infer(image_tensor=image_tensor, model=model)

        # 将推理结果 mask，转换为 RGB 格式的图片数组。rgb_mask 形状为 (512, 1024, 3)。
        rgb_mask = decode_segmentation_masks(inferred_mask,
                                             colormap, num_classes)

        # 将原始图片及其推理结果 rgb_mask，进行重叠得到一个重叠的图片 overlay。
        # overlay 形状为 (512, 1024, 3)。
        overlay = get_overlay(image_tensor, rgb_mask)

        # 将原始图片，重叠的图片 overlay，推理结果 rgb_mask，这三者放在同一行并画出来。
        plot_image_overlay_rgbmask([image_tensor, overlay, rgb_mask],
                                   figsize=(18, 14))
