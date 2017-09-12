# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_float
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Алгоритм KMeans реализован в классе sklearn.cluster.KMeans. Так как это один из примеров unsupervised-задачи,
# для обучения достаточно передать только матрицу объектов.

# В качестве метрики будем использовать PSNR — адаптация метрики MSE для задачи нахождениях сходства изображений.

# Для работы с изображениями мы рекомендуем воспользоваться пакетом scikit-image.
# Чтобы загрузить изображение, необходимо выполнить следующую команду:
# from skimage.io import imread
# image = imread('parrots_4.jpg')

# После этих действий переменная image будет содержать изображение в виде numpy-массива размера n * m * 3,
# где n и m соответствуют размерам изображения, а 3 соответствует формату представления RGB.

# Если вы хотите вывести изображение на экран, необходимо, чтобы у вас была установлена библиотека matplotlib.
# С помощью нее это делается следующим образом:
# import pylab
# pylab.imshow(image)

# Если вы работаете в ipython-notebook'е, то вам необходимо перед выполнением кода выше исполнить
# в любой ячейке инструкцию:
# %matplotlib inline

# Задание

# 1. Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все значения в интервал от 0 до 1.
# Для этого можно воспользоваться функцией img_as_float из модуля skimage.
# Обратите внимание на этот шаг, так как при работе с исходным изображением вы получите некорректный результат.

image = imread('clustering/parrots.jpg')
image = img_as_float(image)
length, height, colors = image.shape
# plt.imshow(image)
# plt.show()

# 2. Создайте матрицу объекты-признаки:
# характеризуйте каждый пиксель тремя координатами - значениями интенсивности в пространстве RGB.

pixels = pd.DataFrame(np.reshape(image, (length*height, colors)), columns=['R', 'G', 'B'])

# 3. Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241.
# После выделения кластеров все пиксели, отнесенные в один кластер,
# попробуйте заполнить двумя способами: медианным и средним цветом по кластеру.


def clusterization(pix_df, n_clusters=8):
    print 'Clustering: ' + str(n_clusters)

    pix_df = pix_df.copy()
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    pix_df['cluster'] = model.fit_predict(pix_df)
    means = pix_df.groupby('cluster').mean().values
    mean_pixels = [means[c] for c in pix_df['cluster'].values]
    mean_image = np.reshape(mean_pixels, (length, height, colors))
    imsave('clustering/mean/parrots_' + str(n_clusters) + '.jpg', mean_image)
    medians = pix_df.groupby('cluster').median().values
    median_pixels = [medians[c] for c in pix_df['cluster'].values]
    median_image = np.reshape(median_pixels, (length, height, colors))
    imsave('clustering/median/parrots_' + str(n_clusters) + '.jpg', median_image)
    return mean_image, median_image

# 4. Измерьте качество получившейся сегментации с помощью метрики PSNR.
# Эту метрику нужно реализовать самостоятельно (см. определение).
# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio


def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return 10 * math.log10(float(1) / mse)

# 5. Найдите минимальное количество кластеров, при котором значение PSNR выше 20
# (можно рассмотреть не более 20 кластеров, но не забудьте рассмотреть оба способа заполнения пикселей одного кластера).
# Это число и будет ответом в данной задаче.

for n in xrange(1, 21):
    mean_img, median_img = clusterization(pixels, n)
    psnr_mean, psnr_median = psnr(image, mean_img), psnr(image, median_img)
    print psnr_mean, psnr_median
    if psnr_mean > 20 or psnr_median > 20:
        print(n)
        break
