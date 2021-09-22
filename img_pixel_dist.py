from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2

def img_diff(first, second):
    imageA = cv2.imread(first)
    imageB = cv2.imread(second)
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM (Structural Similarity Index): {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show the output images
    filename = "diff_" + first + "_" + second + "_" + str(score) + ".png"
    cv2.imwrite(filename, diff)

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-1", "--first", required=True,
        help="first input image")
    ap.add_argument("-2", "--second", required=True,
        help="second")
    ap.add_argument("-3", "--third", required=True,
        help="third")
    ap.add_argument("-4", "--fourth", required=True,
        help="fourth")
    args = vars(ap.parse_args())

    img_diff(args["first"], args["second"])
    img_diff(args["first"], args["third"])
    img_diff(args["first"], args["fourth"])

    img_diff(args["second"], args["third"])
    img_diff(args["second"], args["fourth"])

    img_diff(args["third"], args["fourth"])