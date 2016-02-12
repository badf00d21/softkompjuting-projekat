import cv2
import dlib
import numpy

#%matplotlib inline
import matplotlib.pyplot as plt 
import collections

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab
#pylab.rcParams['figure.figsize'] = 16, 12 # 

import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))

##im1, landmarks1 = read_im_and_landmarks("C:\Users\Aleksandar\Desktop\cosby.jpg")
#im2, landmarks2 = read_im_and_landmarks("C:\Users\Aleksandar\Desktop\qwe.jpg")

#print landmarks1[0][0]
#print landmarks1[1][0]
#for x in range (0, landmarks1.__len__()-1):

##print landmarks1.__len__()

##image = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
##image2 = annotate_landmarks(image,landmarks1)
#cv2.circle(image,(230,157),3,(0,255,255),-1)
#cv2.circle(image,(392,140),3,(0,255,255),-1)
##plt.imshow(image2)

##result_dic = dict()

def calculate_face_width(landmarks, dic):
    x1= landmarks[0][0,0]
    y1= landmarks[0][0,1]
    x2= landmarks[16][0,0]
    y2= landmarks[16][0,1]
    face_width = numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2)))
    dic["face_width"] = face_width
 #   print "face width:",face_width
    x1= landmarks[27][0,0]
    y1= landmarks[27][0,1]
    x2= landmarks[33][0,0]
    y2= landmarks[33][0,1]
    nose_height = numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2)))
    dic["nose_height"] = nose_height
  #  print "nose height:",nose_height
    x1= landmarks[48][0,0]
    y1= landmarks[48][0,1]
    x2= landmarks[54][0,0]
    y2= landmarks[54][0,1]
    mouth_width = numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2)))
    dic["mouth_width"] = mouth_width
 #   print "mouth width:",mouth_width
    x1= landmarks[39][0,0]
    y1= landmarks[39][0,1]
    x2= landmarks[42][0,0]
    y2= landmarks[42][0,1]
    between_eyes = numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2)))
    dic["between_eyes"] = between_eyes
 #   print "between eyes distance:",between_eyes
    x1= landmarks[52][0,0]
    y1= landmarks[52][0,1]
    x2= landmarks[56][0,0]
    y2= landmarks[56][0,1]
    mouth_thick = numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2)))
    dic["mouth_thick"] = mouth_thick
  #  print "mouth thickness:",mouth_thick
    x1= landmarks[31][0,0]
    y1= landmarks[31][0,1]
    x2= landmarks[35][0,0]
    y2= landmarks[35][0,1]
    nose_width = numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2)))
    dic["nose_width"] = nose_width
 #   print "nose width:",nose_width
    x1= landmarks[36][0,0]
    y1= landmarks[36][0,1]
    x2= landmarks[39][0,0]
    y2= landmarks[39][0,1]
    eye_width = numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2)))
    dic["eye_width"] = eye_width
 #   print "eye width:",eye_width
    x1= landmarks[41][0,0]
    y1= landmarks[41][0,1]
    x2= landmarks[37][0,0]
    y2= landmarks[37][0,1]
    eye_height = numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2)))
    dic["eye_height"] = eye_height
   # print "eye heigth:",eye_height
# eye angles// different
   
    outer_up_line = distance_between2_points(landmarks[36][0,0], landmarks[37][0,0], landmarks[36][0,1],landmarks[37][0,1])
    outer_down_line = distance_between2_points(landmarks[36][0,0], landmarks[41][0,0], landmarks[36][0,1],landmarks[41][0,1])
    outer_right_line =distance_between2_points(landmarks[41][0,0], landmarks[37][0,0], landmarks[41][0,1] ,landmarks[37][0,1])
    print "gornja, donja, desna:{0} {1} {2} ", outer_up_line, outer_down_line, outer_right_line
    outer_eye_angle = angle_3_points(outer_up_line,outer_down_line,outer_right_line)
    dic["outer_eye_angle"] = outer_eye_angle

    inner_up_line = distance_between2_points(landmarks[38][0,0],landmarks[39][0,0], landmarks[38][0,1], landmarks[39][0,1])
    inner_down_line = distance_between2_points(landmarks[39][0,0],landmarks[40][0,0], landmarks[39][0,1], landmarks[40][0,1])
    inner_right_line =distance_between2_points(landmarks[38][0,0], landmarks[40][0,0],landmarks[38][0,1], landmarks[40][0,1])
    print "gornja, donja, desna:{0} {1} {2} ", inner_up_line, inner_down_line, inner_right_line
    inner_eye_angle = angle_3_points(inner_up_line,inner_down_line,inner_right_line)
    dic["inner_eye_angle"] = inner_eye_angle

    upper_lip_height = distance_between2_points(landmarks[61][0,0], landmarks[50][0,0],landmarks[61][0,1], landmarks[50][0,1])
    lower_lip_height = distance_between2_points(landmarks[67][0,0], landmarks[58][0,0], landmarks[67][0,1],landmarks[58][0,1])
    print "lip sizes: {0} {1}", upper_lip_height, lower_lip_height
    dic["lip_proportion"] = upper_lip_height / lower_lip_height
    
def distance_between2_points(x1,x2,y1,y2):
    return numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2)))

def angle_3_points(CA, CB, AB):
    angle = numpy.degrees(numpy.arccos( (numpy.power(CA,2) + numpy.power(CB,2) - numpy.power(AB,2)) / ( 2 * CA * CB ) ))
    print "the angle is:",angle
    return angle

def calculate_proportion(landmarks):
    x1= landmarks[0][0,0]
    y1= landmarks[0][0,1]
    x2= landmarks[1][0,0]
    y2= landmarks[1][0,1]
    proportion = numpy.sqrt(numpy.abs(numpy.power(x1-x2,2) + numpy.power(y1-y2,2))) * 10
   # print "proportion:",proportion
    return proportion

from math import ceil, floor
def float_round(num, places = 0, direction = floor):
    return direction(num * (10**places)) / float(10**places)

def calculate_percentage(dic, prop):

    r_dic = dict()
    for key in dic:
        r_dic[key] = dic[key] / prop
        r_dic[key] = float_round(r_dic[key], 3, round) #round naturally
    r_dic["outer_eye_angle"] = dic["outer_eye_angle"]
    r_dic["inner_eye_angle"] = dic["inner_eye_angle"]
    r_dic["lip_proportion"] = dic["lip_proportion"]
       # print key , " = ", dic[key], "%"
    return r_dic

def save_image_fun(path, image):
    print path
    return cv2.imwrite(path,image)

def make_description(dic):
    retVal = ''
    if dic['mouth_thick'] < 0.12:
        retVal += " Ispod proseka je debljina usta.</br>"
    else:
        retVal += "Debljina usta iznad proseka.</br>"

    if dic['eye_width'] < 0.13:
        retVal += " Ispod proseka je sirina ociju.</br>"
    else:
        retVal += "Iznad proseka je sirina ociju.</br>"

    if dic['between_eyes'] < 0.208:
        retVal += " Ispod proseka je izmedju ociju.</br>"
    else:
        retVal += "Iznad proseka je izmedju ociju.</br>"

    if dic['eye_height'] < 0.046:
        retVal += " Ispod proseka je visina ociju.</br>"
    else:
        retVal += "Iznad proseka je visina ociju.</br>"

    if dic['eye_width'] < 0.29:
        retVal += " Ispod proseka je visina nosa.</br>"
    else:
        retVal += "Iznad proseka je visina nosa.</br>"

    if dic['face_width'] < 0.75:
        retVal += " Ispod proseka je sirina lica.</br>"
    else:
        retVal += "Iznad proseka je sirina lica.</br>"

    if dic['mouth_width'] < 0.29:
        retVal += " Ispod proseka je sirina usta.</br>"
    else:
        retVal += "Iznad proseka je sirina usta.</br>"

    if dic['outer_eye_angle'] < 60:
        retVal += " Ispod proseka je spoljasnji ugao oka.</br>"
    else:
        retVal += "Iznad proseka je spoljasnji ugao oka.</br>"

    if dic['inner_eye_angle'] < 60:
        retVal += " Ispod proseka je unutrasnji ugao oka.</br>"
    else:
        retVal += "Iznad proseka je unutrasnji ugao oka.</br>"

    if dic['lip_proportion'] < 1:
        retVal += " Veca je donja usna.</br>"
    else:
        retVal += "Veca je gornja usna.</br>"

    return retVal
##calculate_face_width(landmarks1, result_dic )
##p = calculate_proportion(landmarks1)
##print "-------------------"
##calculate_percentage(result_dic, p)