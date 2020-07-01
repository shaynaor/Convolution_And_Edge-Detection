from ex2_utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1


def check_sum_diff(image1: np.ndarray, kernel: np.ndarray, thresh: float) -> bool:
    return ((conv2D(image1, kernel) - cv2.filter2D(image1, -1, kernel, borderType=cv2.BORDER_REPLICATE)).sum()) < thresh


def test_conv1D():
    signal1 = np.array([1, 2, 3, -8])
    signal2 = np.array([0, 0, 0])
    signal3 = np.array([1.5, 3.7, -0.3])
    kernel1 = np.array([0, 1, 0.5, -5])
    kernel2 = np.array([0, 0, 0, 0, 0])
    kernel3 = np.array([0, -1, -3.7])

    np.testing.assert_array_equal(conv1D(signal1, kernel1), np.convolve(signal1, kernel1))
    np.testing.assert_array_equal(conv1D(signal1, kernel2), np.convolve(signal1, kernel2))
    np.testing.assert_array_equal(conv1D(signal1, kernel3), np.convolve(signal1, kernel3))
    np.testing.assert_array_equal(conv1D(signal2, kernel1), np.convolve(signal2, kernel1))
    np.testing.assert_array_equal(conv1D(signal2, kernel2), np.convolve(signal2, kernel2))
    np.testing.assert_array_equal(conv1D(signal2, kernel3), np.convolve(signal2, kernel3))
    np.testing.assert_array_equal(conv1D(signal3, kernel1), np.convolve(signal3, kernel1))
    np.testing.assert_array_equal(conv1D(signal3, kernel2), np.convolve(signal3, kernel2))
    np.testing.assert_array_equal(conv1D(signal3, kernel3), np.convolve(signal3, kernel3))

    print("FINISH test_conv1D() successfully")


def test_conv2D():
    # read images
    img_path_boxman = 'boxman.jpg'
    img_path_monkey = 'codeMonkey.jpeg'
    image_boxman = imReadAndConvert(img_path_boxman, LOAD_GRAY_SCALE)
    image_monkey = imReadAndConvert(img_path_monkey, LOAD_GRAY_SCALE)

    kernel1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    kernel2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel2 = kernel2 / kernel2.sum()
    kernel3 = np.array([[-1.5, 2.1], [4.4, 6.5]])
    kernel3 = kernel3 / kernel3.sum()
    kernel4 = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 10, 11], [10, 11, 12, 13, 14], [1, 2, 3, 4, 5]])
    kernel4 = kernel4 / kernel4.sum()

    assert check_sum_diff(image_boxman, kernel1, 0.1)
    assert check_sum_diff(image_boxman, kernel2, 0.1)
    assert check_sum_diff(image_boxman, kernel3, 0.1)
    assert check_sum_diff(image_boxman, kernel4, 0.1)

    assert check_sum_diff(image_monkey, kernel1, 0.1)
    assert check_sum_diff(image_monkey, kernel2, 0.1)
    assert check_sum_diff(image_monkey, kernel3, 0.1)
    assert check_sum_diff(image_monkey, kernel4, 0.1)

    print("FINISH test_conv2D() successfully")


def mse(img1: np.ndarray, img2: np.ndarray, thresh) -> bool:
    return np.square(np.subtract(img1, img2)).mean() < thresh


def test_convDerivative():
    # read images
    img_path_boxman = 'boxman.jpg'
    img_path_monkey = 'codeMonkey.jpeg'
    image_boxman = imReadAndConvert(img_path_boxman, LOAD_GRAY_SCALE)
    image_monkey = imReadAndConvert(img_path_monkey, LOAD_GRAY_SCALE)

    ker_x = np.array([[-1, 0, 1]])
    ker_y = ker_x.reshape((3, 1))

    x_der1, y_der1 = convDerivative(image_boxman)[2:]
    x_der2, y_der2 = convDerivative(image_monkey)[2:]

    assert mse(x_der1, cv2.filter2D(image_boxman, -1, ker_x, borderType=cv2.BORDER_REPLICATE), 0.1)
    assert mse(y_der1, cv2.filter2D(image_boxman, -1, ker_y, borderType=cv2.BORDER_REPLICATE), 0.1)

    assert mse(x_der2, cv2.filter2D(image_monkey, -1, ker_x, borderType=cv2.BORDER_REPLICATE), 0.1)
    assert mse(x_der2, cv2.filter2D(image_monkey, -1, ker_y, borderType=cv2.BORDER_REPLICATE), 0.1)

    print("FINISH test_convDerivative() successfully")


def test_edgeDetectionZeroCrossingLOG():
    # read images
    img_path_boxman = 'boxman.jpg'
    img_path_monkey = 'codeMonkey.jpeg'
    img_path_coins = 'coins.jpg'
    image_boxman = imReadAndConvert(img_path_boxman, LOAD_GRAY_SCALE)
    image_monkey = imReadAndConvert(img_path_monkey, LOAD_GRAY_SCALE)
    image_coins = imReadAndConvert(img_path_coins, LOAD_GRAY_SCALE)

    log_boxman = edgeDetectionZeroCrossingLOG(image_boxman)
    log_monkey = edgeDetectionZeroCrossingLOG(image_monkey)
    log_coins = edgeDetectionZeroCrossingLOG(image_coins)

    # show boxman
    plt.gray()
    fig = plt.figure(0)
    fig.canvas.set_window_title('log_boxman')
    plt.imshow(log_boxman)
    # show monkey
    fig = plt.figure(1)
    fig.canvas.set_window_title('log_monkey')
    plt.imshow(log_monkey)
    # show coins
    fig = plt.figure(2)
    fig.canvas.set_window_title('log_coins')
    plt.imshow(log_coins)

    plt.show()


def test_edgeDetectionSobel():
    # read images
    img_path_boxman = 'boxman.jpg'
    img_path_monkey = 'codeMonkey.jpeg'
    img_path_coins = 'coins.jpg'
    image_boxman = imReadAndConvert(img_path_boxman, LOAD_GRAY_SCALE)
    image_monkey = imReadAndConvert(img_path_monkey, LOAD_GRAY_SCALE)
    image_coins = imReadAndConvert(img_path_coins, LOAD_GRAY_SCALE)
    # apply sobol
    cv_res_codeMonkey, my_res_codeMonkey = edgeDetectionSobel(image_monkey, thresh=0.1)
    cv_res_boxman, my_res_boxman = edgeDetectionSobel(image_boxman, thresh=0.2)
    cv_res_coins, my_res_coins = edgeDetectionSobel(image_coins, thresh=0.4)

    # show monkey
    plt.gray()
    fig = plt.figure(0)
    fig.canvas.set_window_title('cv_sobel_res_codeMonkey')
    plt.imshow(cv_res_codeMonkey)
    fig = plt.figure(1)
    fig.canvas.set_window_title('my_sobel_res_codeMonkey')
    plt.imshow(my_res_codeMonkey)
    plt.show()

    # show boxman
    plt.gray()
    fig = plt.figure(0)
    fig.canvas.set_window_title('cv_sobel_res_boxman')
    plt.imshow(cv_res_boxman)
    fig = plt.figure(1)
    fig.canvas.set_window_title('my_sobel_res_boxman')
    plt.imshow(my_res_boxman)
    plt.show()

    # show coins
    plt.gray()
    fig = plt.figure(0)
    fig.canvas.set_window_title('cv_sobel_res_coins')
    plt.imshow(cv_res_coins)
    fig = plt.figure(1)
    fig.canvas.set_window_title('my_sobel_res_coins')
    plt.imshow(my_res_coins)
    plt.show()


def test_edgeDetectionCanny():
    # read images
    img_path_boxman = 'boxman.jpg'
    img_path_coins = 'coins.jpg'
    image_boxman = imReadAndConvert(img_path_boxman, LOAD_GRAY_SCALE)
    image_coins = imReadAndConvert(img_path_coins, LOAD_GRAY_SCALE)

    cv_canny_boxman, my_canny_boxman = edgeDetectionCanny(image_boxman, 75, 30)
    cv_canny_coins, my_canny_coins = edgeDetectionCanny(image_coins, 75, 30)

    # show boxman
    plt.gray()
    fig = plt.figure(0)
    fig.canvas.set_window_title('cv_canny_boxman')
    plt.imshow(cv_canny_boxman)
    fig = plt.figure(1)
    fig.canvas.set_window_title('my_canny_boxman')
    plt.imshow(cv_canny_boxman)
    plt.show()

    # show coins
    plt.gray()
    fig = plt.figure(0)
    fig.canvas.set_window_title('cv_canny_coins')
    plt.imshow(cv_canny_coins)
    fig = plt.figure(1)
    fig.canvas.set_window_title('my_canny_coins')
    plt.imshow(my_canny_coins)
    plt.show()


def test_blurImage():
    # read images
    img_path_boxman = 'boxman.jpg'
    img_path_monkey = 'codeMonkey.jpeg'
    img_path_coins = 'coins.jpg'
    image_boxman = imReadAndConvert(img_path_boxman, LOAD_GRAY_SCALE)
    image_monkey = imReadAndConvert(img_path_monkey, LOAD_GRAY_SCALE)
    image_coins = imReadAndConvert(img_path_coins, LOAD_GRAY_SCALE)

    kernel_size = 5

    my_boxman_blur, cv_boxman_blur = blurImage1(image_boxman, kernel_size), blurImage2(image_boxman, kernel_size)
    my_monkey_blur, cv_monkey_blur = blurImage1(image_monkey, kernel_size), blurImage2(image_monkey, kernel_size)
    my_coins_blur, cv_coins_blur = blurImage1(image_coins, kernel_size), blurImage2(image_coins, kernel_size)

    assert mse(my_boxman_blur, cv_boxman_blur, 0.1)
    assert mse(my_monkey_blur, cv_monkey_blur, 0.1)
    assert mse(my_coins_blur, cv_coins_blur, 0.1)

    print("FINISH test_blurImage() successfully")





def main():
    print("ID:", myID())
    test_conv1D()
    test_conv2D()
    test_convDerivative()
    test_edgeDetectionSobel()
    test_edgeDetectionZeroCrossingLOG()
    test_edgeDetectionCanny()
    test_blurImage()
    print("FINISH all tests")





if __name__ == '__main__':
    main()
