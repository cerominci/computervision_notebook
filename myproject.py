# %% [markdown]
# # VISION ARTIFICIAL PROJECT
# #### by Ceren Karaoglan

# %% [markdown]
# ### RECTIF

# %%
import numpy as np
import cv2   as cv
import matplotlib.pyplot as plt
from umucv.util import putText

def readrgb(file):
    return cv.cvtColor( cv.imread('../images/'+file), cv.COLOR_BGR2RGB)

def readtxtfile(file):
    return '../images/'+file

# %% [markdown]
# Firstly we have to manually take the coordinates of our reference:

# %%
def get_reference_coordinates(file_path):
    image_points = []
    real_points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            img_x,img_y,reel_x,reel_y = map(float, line.strip().split(','))
            image_points.append([img_x,img_y])
            real_points.append([reel_x,reel_y])
    return np.array(image_points, dtype='float32'),np.array(real_points,dtype='float32')

# %% [markdown]
# We find homography matrix by real and image coordinate of the reference. We use this matrix to rectify the image, therefore measuring
# the distance in real life.

# %%
def measure_distance(image_path, reference_points_file, img1_coords, img2_coords):
    img = readrgb(image_path)
    plt.imshow(img)

    # Load the reference points from the text file
    image_points, real_world_points = get_reference_coordinates(reference_points_file)

    H, _ = cv.findHomography(image_points, real_world_points)

    img1_h = np.append(img1_coords, 1)
    img2_h = np.append(img2_coords, 1)

    img1_rectified = H @ img1_h
    img2_rectified = H @ img2_h

    img1_rectified /= img1_rectified[2]
    img2_rectified /= img2_rectified[2]

    distance = np.linalg.norm(img1_rectified[:2] - img2_rectified[:2])
    
    return distance

# %%
def result(image, img1,img2 ,distance):
    for point in [img1,img2]:
        cv.circle(image, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)
    point1, point2 =img1,img2
    cv.line(image, (int(img1[0]), int(img1[1])), (int(img2[0]), int(img2[1])), (255, 0, 0), 2)

    mid_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

    cv.putText(image, f"{distance:.2f} cm", (int(mid_point[0]), int(mid_point[1]) - 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Distance Measured")
    plt.show()

# %% [markdown]
# In this case of coins.png, we have to load of its text file of ref coordinates(i used gimp to measure the ruler's four coordinates):

# %%
file_path = readtxtfile('coordinates.txt')
img = readrgb('coins.png')

# %% [markdown]
# i again used gimp to measure the coins' coordinate

# %%
coin1 = np.array([275, 139], dtype='float32')  # Coordinates of the first coin
coin2 = np.array([441, 20], dtype='float32')  # Coordinates of the second co

# %%
distance = measure_distance('coins.png', file_path, coin1, coin2)

# %%
result(img,coin1,coin2,distance)

# %% [markdown]
# final result is:

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# Now we'll measure the distance at which the shot is made in gol-eder.png

# %%
file_path = readtxtfile('coordinates_goal.txt')
img = readrgb('gol-eder.png')

# %%
img = readrgb('gol-eder.png')
person = np.array([260, 163], dtype='float32')
goal = np.array([0, 136], dtype='float32') 

distance = measure_distance('gol-eder.png', file_path, person, goal)

result(img,person,goal,distance)

# %% [markdown]
# result is:![rectified.png](attachment:a46c66d1-e029-4ea8-b9b9-b69c97fe21f3.png)
# 

# %% [markdown]
# ### HANDS

# %%
#!/usr/bin/env python

#https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

import cv2 as cv
import numpy as np

from umucv.stream import autoStream
from umucv.util import putText

# %%
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# %% [markdown]
# We implement additional features such as calculating the number of fingers, distance from the camera and angle of orientation of the hand, also I observe that thumb finger wrongfully detected so extra case is added.

# %%
def count_fingers(points):
    tips = [4, 8, 12, 16, 20]
    
    #the base of each finger
    base = [2, 5, 9, 13, 17]
    

    extended = 0
    
    for i in range(1, 5):
        if points[tips[i]][1] < points[base[i]][1]:
            extended += 1
    
    # Special case for thumb
    thumb_tip = points[4]
    thumb_mcp = points[2] 
    thumb_cmc = points[1]

    # Calculate angle between thumb and index finger base to decide if thumb is extended
    thumb_angle = np.degrees(np.arctan2(thumb_tip[1] - thumb_cmc[1], thumb_tip[0] - thumb_cmc[0]) - np.arctan2(thumb_mcp[1] - thumb_cmc[1], thumb_mcp[0] - thumb_cmc[0]))
    thumb_angle = np.abs(thumb_angle)
    if thumb_angle < 20:
        extended += 1

    return extended



# %% [markdown]
# To calculate the distance, we use reference values to calculate the ratio of our distance to the camera. For angle of orientation, we use wrist as origin and middle finger as line in x-y coordinate.

# %%
def calculate_distance_and_orientation(points):
    wrist = points[0]
    middle_finger_tip = points[12]
    index_finger_base = points[5]
    pinky_base = points[17]
    
    # Distance: using the length between wrist and middle finger tip, and between index finger base and pinky base
    wrist_to_middle_finger_tip = np.linalg.norm(wrist - middle_finger_tip)
    index_to_pinky_base = np.linalg.norm(index_finger_base - pinky_base)
    average_hand_size = (wrist_to_middle_finger_tip + index_to_pinky_base) / 2
    
    # Reference values for a typical hand size and distance
    reference_hand_size = 200  # pixels
    reference_distance = 50  # cm
    
    distance = reference_distance * (reference_hand_size / average_hand_size)
    
    # Orientation: angle between the wrist and the middle finger tip
    orientation_vector = middle_finger_tip - wrist
    angle = np.arctan2(orientation_vector[1], orientation_vector[0])
    angle_degrees = np.degrees(angle)
    
    if angle_degrees < 0:
        angle_degrees += 360
    
    return distance, angle_degrees



# %%
for _, frame in autoStream():
    H, W, _ = frame.shape
    imagecv = cv.flip(frame, 1)
    image = cv.cvtColor(imagecv, cv.COLOR_BGR2RGB)
    results = hands.process(image)
    
    points = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for k in range(21):
                x = hand_landmarks.landmark[k].x
                y = hand_landmarks.landmark[k].y
                points.append([int(x * W), int(y * H)])
            break

        points = np.array(points)
        
        # Count the number of fingers extended
        num_fingers = count_fingers(points)
        putText(imagecv, f"Fingers: {num_fingers}", (10, 30))
        
        # Calculate distance and orientation
        distance, angle = calculate_distance_and_orientation(points)
        putText(imagecv, f"Distance: {distance:.2f}", (10, 60))
        putText(imagecv, f"Angle: {angle:.2f}", (10, 90))
        
        # Draw key points
        cv.line(imagecv, points[5], points[8], color=(0, 255, 255), thickness=3)
        center = np.mean(points[[5, 0, 17]], axis=0)
        radio = np.linalg.norm(center - points[5])
        cv.circle(imagecv, center.astype(int), int(radio), color=(0, 255, 255), thickness=3)

    cv.imshow("mirror", imagecv)

# %% [markdown]
# ![Screenshot 2024-06-28 at 17.22.06.png](attachment:faf5a118-f389-4491-918b-d199dda637eb.png)

# %% [markdown]
# ![Screenshot 2024-06-28 at 17.22.19.png](attachment:0b63a171-cbfb-49d7-907e-cabe70d46bfc.png)

# %% [markdown]
# ### FILTROS

# %% [markdown]
# Constructing the filters

# %%
import numpy as np
import cv2   as cv
from umucv.util import putText

def readfile(file):
    return '../images/'+file

def apply_filter(img, filter_type, param1, param2):
    if filter_type == 1:  # Box Blur
        return cv.blur(img, (param1, param1))
    elif filter_type == 2:  # Gaussian Blur
        return cv.GaussianBlur(img, (param1*2+1, param1*2+1), param2)
    elif filter_type == 3:  # Median Blur
        return cv.medianBlur(img, param1*2+1)
    elif filter_type == 4:  # Bilateral Filter
        return cv.bilateralFilter(img, param1*2+1, param2, param2)
    elif filter_type == 5:  # Min Filter
        return cv.erode(img, np.ones((param1, param1), np.uint8))
    elif filter_type == 6:  # Max Filter
        return cv.dilate(img, np.ones((param1, param1), np.uint8))
    else:  # Do nothing
        return img

# %% [markdown]
# Creating the trackbars

# %%
def nothing(x):
    pass

img = cv.imread(readfile('cards.png'))
h, w, _ = img.shape

# Create windows
cv.namedWindow('Filtered Image')
cv.namedWindow('Help')

# Create trackbars
cv.createTrackbar('Filter', 'Filtered Image', 0, 6, nothing)
cv.createTrackbar('Param1', 'Filtered Image', 1, 50, nothing)
cv.createTrackbar('Param2', 'Filtered Image', 1, 50, nothing)


# %% [markdown]
# Created a help section. Applied the filter to half of the selected image. Arranged c, r, h buttons accordingly.

# %%
help_img = np.zeros((400, 1000, 3), dtype=np.uint8)
putText(help_img, "BLUR FILTERS", (10, 30))
filters = ["0: do nothing", "1: box", "2: Gaussian", "3: median", "4: bilateral", "5: min", "6: max"]
for i, text in enumerate(filters):
    putText(help_img, text, (10, 70 + i*30))

putText(help_img, "c: color/monochrome", (10, 280))
putText(help_img, "r: only roi", (10, 310))
putText(help_img, "h: show/hide help", (10, 340))

show_help = True
color_mode = True
roi_only = False

while True:
    filter_index = cv.getTrackbarPos('Filter', 'Filtered Image')
    param1 = cv.getTrackbarPos('Param1', 'Filtered Image')
    param2 = cv.getTrackbarPos('Param2', 'Filtered Image')


    roi = img[:, :w//2]

    if roi_only:
        # Apply the filter to the ROI
        filtered_roi = apply_filter(roi, filter_index, param1, param2)
        combined_img = img.copy()
        combined_img[:, :w//2] = filtered_roi
    else:
        # Apply the filter to the entire image
        combined_img = apply_filter(img, filter_index, param1, param2)

    if not color_mode:
        combined_img = cv.cvtColor(combined_img, cv.COLOR_BGR2GRAY)
        combined_img = cv.cvtColor(combined_img, cv.COLOR_GRAY2BGR) 

    # Show the combined image
    cv.imshow('Filtered Image', combined_img)

    # Show the help window
    if show_help:
        cv.imshow('Help', help_img)
    else:
        cv.destroyWindow('Help')

    # Wait for key press
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        show_help = not show_help
    elif key == ord('c'):
        color_mode = not color_mode
    elif key == ord('r'):
        roi_only = not roi_only

# Release resources
cv.destroyAllWindows()

# %% [markdown]
# ![Screenshot 2024-07-01 at 11.52.03.png](attachment:645fe2a5-404c-4ffc-ba1a-f633cb9847bb.png)

# %% [markdown]
# ![Screenshot 2024-07-01 at 11.52.52.png](attachment:55848f27-bf63-49e0-8a26-774cf49356c2.png)

# %% [markdown]
# ### RA

# %% [markdown]
# We create ar effect by using pygame package. Here I used a render image for virtual object and adjust the size accordingly. At the end the events are handled by specific type of keys, such as closing the video capture or putting the object at desired point.

# %%
import cv2
import numpy as np
import pygame
from pygame.locals import *

def readfile(file):
    return '../images/'+file

pygame.init()

# Screen dimensions
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('AR Object Placement')

# Virtual object dimensions
object_width, object_height = 80, 80

# Load a virtual object (you can replace this with your own image)
virtual_object = pygame.image.load(readfile('virtual-assistant-object-medal-3d-illustration-png.png')).convert_alpha()
virtual_object = pygame.transform.scale(virtual_object, (object_width, object_height))


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from camera.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 


    frame = cv2.resize(frame, (screen_width, screen_height))

    # Convert frame to Pygame surface
    frame = np.rot90(frame)  # Rotate frame 90 degrees
    frame = pygame.surfarray.make_surface(frame)

    
    screen.blit(frame, (0, 0))

    # Get mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()

    # Draw virtual object at mouse position
    object_rect = virtual_object.get_rect(center=(mouse_x, mouse_y))
    screen.blit(virtual_object, object_rect)

   
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                cap.release()
                cv2.destroyAllWindows()
                exit()
        elif event.type == MOUSEBUTTONDOWN:
            print(f"Object placed at ({mouse_x}, {mouse_y})")


cap.release()
cv2.destroyAllWindows()


# %% [markdown]
# result:
# ![Screenshot 2024-06-28 at 20.09.55.png](attachment:19c3f042-c37f-4a5f-ac55-9a55f60c1100.png)

# %% [markdown]
# ### CLASIFICADOR

# %% [markdown]
# Imported the model file and necessary libraries. Created a directory containing several pizza images. Converted the images to Mediapipe image format and get the embedding descriptors. Find the cosine similarity of the images.

# %%
import os
import cv2 as cv
import numpy as np
import urllib.request
import mediapipe as mp


from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ImageEmbedder


MODEL_PATH = "/Users/cerenkaraoglan/umucv/notebooks/mobilenet_v3_small.tflite"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite", MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
ImageEmbedderOptions = vision.ImageEmbedderOptions
ImageEmbedder = vision.ImageEmbedder

options = ImageEmbedderOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    l2_normalize=True,
    quantize=True
)
embedder = ImageEmbedder.create_from_options(options)

reference_descriptor = None


def process_image(image_path, method):
    global reference_descriptor
    
    frame = cv.imread(image_path)
    
    mp_image = mp.Image.create_from_file(image_path)
    
    embedding_result = embedder.embed(mp_image)
    descriptor = np.array(embedding_result.embeddings[0].embedding)
    
    if method == "embedding":
        if reference_descriptor is None:
            reference_descriptor = descriptor
        else:
            similarity = np.dot(descriptor, reference_descriptor) / (np.linalg.norm(descriptor) * np.linalg.norm(reference_descriptor))
    
    return frame, descriptor

def process_directory(directory, method):
    global reference_descriptor
    reference_descriptor = None
    
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    descriptors = []
    
    for image_file in image_files:
        frame, descriptor = process_image(image_file, method)
        descriptors.append((image_file, descriptor))
    
    return descriptors


def compare_images(descriptors):
    similarities = []
    for i, (img1, desc1) in enumerate(descriptors):
        for j, (img2, desc2) in enumerate(descriptors):
            if i < j:
                similarity = np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))
                print(f"Comparing {img1} and {img2}: {similarity}")  # Debugging information
                similarities.append((img1, img2, similarity))
    return similarities

def main(directory, method):
    descriptors = process_directory(directory, method)
    compare_images(descriptors)
    

directory = '/Users/cerenkaraoglan/Downloads/pizza/'
main(directory, "embedding")


# %% [markdown]
# ### SIFT

# %% [markdown]
# Revised the clasificador code adding a SIFT method to create descriptor instead of embedding.

# %%
import os
import cv2 as cv
import numpy as np
import urllib.request
import mediapipe as mp


MODEL_PATH = "/Users/cerenkaraoglan/umucv/notebooks/mobilenet_v3_small.tflite"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite", MODEL_PATH)


from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ImageEmbedder

BaseOptions = mp.tasks.BaseOptions
ImageEmbedderOptions = vision.ImageEmbedderOptions
ImageEmbedder = vision.ImageEmbedder

options = ImageEmbedderOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    l2_normalize=True,
    quantize=True
)
embedder = ImageEmbedder.create_from_options(options)


sift = cv.SIFT_create()

reference_descriptor = None


def extract_sift_features(image_path):
    frame = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    

    keypoints, descriptors = sift.detectAndCompute(frame, None)
    
    return keypoints, descriptors

def process_image(image_path, method):
    global reference_descriptor
    

    frame = cv.imread(image_path)
    

    mp_image = mp.Image.create_from_file(image_path)
    

    embedding_result = embedder.embed(mp_image)
    descriptor = np.array(embedding_result.embeddings[0].embedding)
    
    if method == "embedding":
        if reference_descriptor is None:
            reference_descriptor = descriptor
        else:
            similarity = np.dot(descriptor, reference_descriptor) / (np.linalg.norm(descriptor) * np.linalg.norm(reference_descriptor))
    
    return frame, descriptor


def process_directory(directory, method):
    global reference_descriptor
    reference_descriptor = None
    
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    descriptors = []
    
    for image_file in image_files:
        frame, descriptor = process_image(image_file, method)
        descriptors.append((image_file, descriptor))
    
    return descriptors


def compare_images(descriptors, method="embedding", sift_threshold=10):
    similarities = []
    
    if method == "embedding":

        for i, (img1, desc1) in enumerate(descriptors):
            for j, (img2, desc2) in enumerate(descriptors):
                if i < j:
                    similarity = np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))
                    print(f"Comparing {img1} and {img2}: {similarity}")
                    similarities.append((img1, img2, similarity))
    elif method == "sift":

        for i, (img1, _) in enumerate(descriptors):
            for j, (img2, _) in enumerate(descriptors):
                if i < j:
                    keypoints1, descriptors1 = extract_sift_features(img1)
                    keypoints2, descriptors2 = extract_sift_features(img2)
                    
        
                    bf = cv.BFMatcher()
                    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
                    
            
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                    
          
                    similarity = len(good_matches)
                    print(f"Comparing {img1} and {img2}: {similarity}")
                    similarities.append((img1, img2, similarity))
                    
    return similarities


def main(directory, method):
    descriptors = process_directory(directory, method)
    compare_images(descriptors, method)

directory = '/Users/cerenkaraoglan/Downloads/pizza/'
main(directory, "sift")


# %% [markdown]
# ### MAPA

# %%
#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np
from umucv.util import putText

# %% [markdown]
# Created fov and size variables of my phone camera. Found focal length using them

# %%
points = deque(maxlen=2)

fov = 120  # fov of my iphone camera, adjust accordingly
frame_width = 1920 
frame_height = 1080 
map_scale = 1.0

fov_horizontal_rad = np.radians(fov)

f = frame_width / (2 * np.tan(fov_horizontal_rad / 2))

# %%
def calculate_pixel_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# %% [markdown]
# Helper function to find the 3d vector

# %%
def calculate_3d_vector(p, q, W, H, f):
    X = p - W / 2
    Y = q - H / 2
    Z = f
    return np.array([X, Y, Z])

# %%
def angle_from_center(p, frame_size):
    center = np.array(frame_size) / 2
    delta = np.array(p) - center
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    return angle


# %%
# triangulation by simple approach
def triangulate_position(distances, angles):
    # Convert angles to radians
    angles_radians = np.radians(angles)
    
    # Triangulation based on distances and angles
    x1 = distances[0] * np.cos(angles_radians[0])
    y1 = distances[0] * np.sin(angles_radians[0])
    x2 = distances[1] * np.cos(angles_radians[1])
    y2 = distances[1] * np.sin(angles_radians[1])
    
    # Calculate approximate position relative to camera
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    
    return (x, y)

# %% [markdown]
# Find the angle between two vectors using dot product equation:

# %%
def calculate_angle_between_vectors(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cos_alpha = dot_product / (norm_vec1 * norm_vec2)
    cos_alpha = np.clip(cos_alpha, -1, 1)
    angle = np.arccos(cos_alpha)
    angle_degree = np.degrees(angle)
    return angle_degree

# %% [markdown]
# Function to select points in live camera:

# %%
def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", fun)

# %% [markdown]
# Calculate the necessary characteristic between two points and print:

# %%
for key, frame in autoStream():
    for p in points:
        cv.circle(frame, p, 3, (0, 0, 255), -1)
    if len(points) == 2:
        cv.line(frame, points[0], points[1], (0, 0, 255))
        c = np.mean(points, axis=0).astype(int)
        
        distance = calculate_pixel_distance(points[0], points[1])

        angle1 = angle_from_center(points[0], frame.shape[:2])
        angle2 = angle_from_center(points[1], frame.shape[:2])

        vec1 = calculate_3d_vector(points[0][0], points[0][1], frame_width, frame_height, f)
        vec2 = calculate_3d_vector(points[1][0], points[1][1], frame_width, frame_height, f)
        
        angle = calculate_angle_between_vectors(vec1, vec2)
        

        putText(frame, f'{distance:.1f} pix', c)

        putText(frame, f'{angle:.1f} deg', (c[0], c[1] + 20))

        position = triangulate_position([distance, distance], [angle1, angle2])
        putText(frame, f'estimated position: {position}', (c[0], c[1] + 40))
    
    cv.imshow('webcam', frame)

cv.destroyAllWindows()

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### FILTROS II

# %% [markdown]
# #### a)
# The cascading property implies that applying multiple smaller Gaussian filters should yield a result similar to applying one larger Gaussian filter. Cascaded blurred image's sigma should be square root of the sum of square sigmas.

# %%
import numpy as np
import cv2 as cv

def readfile(file):
    return '../images/' + file

def check_cascading_property(image):
    blurred1 = cv.GaussianBlur(image, (5, 5), 1)


    blurred2 = cv.GaussianBlur(blurred1, (5, 5), 1)


    cascaded_sigma = np.sqrt(1**2 + 1**2)
    cascaded_blurred = cv.GaussianBlur(image, (11, 11), cascaded_sigma)

    cv.imshow('Double Blurred Image', blurred2)
    cv.imshow('Cascaded Blurred Image', cascaded_blurred)


    mse = np.mean((blurred2 - cascaded_blurred) ** 2)
    print(f"MSE between Double Blurred and Cascaded Blurred: {mse}")

   
    if mse < 1.0:
        print("Cascading property holds true.")
    else:
        print("Cascading property does not hold true.")

    cv.waitKey(0)
    cv.destroyAllWindows()

image = cv.imread(readfile('cards.png'))
check_cascading_property(image)


# %% [markdown]
# ![cascaded.png](attachment:36eabfe7-1e30-4905-a766-58ea983a7ebc.png)

# %% [markdown]
# #### b)

# %% [markdown]
# The Gaussian filter is separable, meaning that the 2D convolution of an image with a Gaussian kernel can be broken down into two 1D convolutions. Convolved the image with a 1D Gaussian kernel in the horizontal direction and with a 1D Gaussian kernel in the vertical direction.

# %%
import numpy as np
import cv2 as cv

def readfile(file):
    return '../images/' + file

def check_separability_property(image):
    sigma = 1
    kernel_size = 11
    kernel_2d = cv.getGaussianKernel(kernel_size, sigma)
    kernel_2d = np.outer(kernel_2d, kernel_2d.T)


    kernel_1d_x = cv.getGaussianKernel(kernel_size, sigma)
    kernel_1d_y = cv.getGaussianKernel(kernel_size, sigma)

    # Compute the product of the 1D kernels
    kernel_1d = np.outer(kernel_1d_x, kernel_1d_y)

  
    print("Separability property:")
    print(np.allclose(kernel_2d, kernel_1d))

 
    blurred_2d = cv.filter2D(image, -1, kernel_2d)
    blurred_1d = cv.filter2D(image, -1, kernel_1d)

    cv.imshow('2D Blurred Image', blurred_2d)
    cv.imshow('1D Blurred Image', blurred_1d)
    cv.waitKey(0)
    cv.destroyAllWindows()

image = cv.imread(readfile('cards.png'))
check_separability_property(image)


# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# #### c)

# %% [markdown]
# Compared the time between the manual-constructed and opencv's convolution method. Convolved the padded image with simple averaging kernel

# %%
import numpy as np
import cv2 as cv
import time

def convolve_manual(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    

    output = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded_image[i:i+kh, j:j+kw] * kernel)
    
    return output

    
kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    

start_time = time.time()
output_manual = convolve_manual(image, kernel)
time_manual = time.time() - start_time
    

start_time = time.time()
output_opencv = cv.filter2D(image, -1, kernel)
time_opencv = time.time() - start_time
    
    # Compare the results
print(f"Manual convolution time: {time_manual:.6f} seconds")
print(f"OpenCV convolution time: {time_opencv:.6f} seconds")




# %% [markdown]
# ![Screenshot 2024-06-29 at 17.38.37.png](attachment:98c34f70-0a5f-40eb-85ee-e26753086d82.png)

# %% [markdown]
# #### d)

# %% [markdown]
# Wrote convolution.c file. Converted it into convolution.dylib(for MAC).Dynamicaly linked the library. Define a function for conv function arranging the parameters according to python. Constructed a python function using the fun written in c. 
# Code runtime decreases significantly.

# %%
import numpy as np
import ctypes
import os
import cv2 as cv
from PIL import Image

cwd = os.getcwd()
lib_path = os.path.join(cwd, 'convolution.dylib')


conv_lib = ctypes.CDLL(lib_path)

# Define the function signature for convolution
convolution_func = conv_lib.convolution
convolution_func.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),  # image
    ctypes.c_int,  # height
    ctypes.c_int,  # width
    ctypes.c_int,  # channels
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # kernel
    ctypes.c_int,  # kernel_size
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS')  # output
]

def convolution(image, kernel):
    height, width, channels = image.shape
    kernel_size = len(kernel)  # Calculate kernel size based on the length of the flattened kernel array


    flattened_kernel = kernel.flatten().astype(np.float32)

    output = np.zeros_like(image, dtype=np.uint8)

    # Call the C function
    convolution_func(image, height, width, channels, flattened_kernel, kernel_size, output)

    return output




image = np.array(Image.open('../images/cards.png'))


kernel = np.ones((3, 3), dtype=np.float32) / 9.0


image = cv.cvtColor(image, cv.COLOR_BGR2RGB) 
result_c = convolution(image, kernel)


cv.imshow('Original Image', image)
cv.imshow('Result using C Convolution', result_c)
cv.waitKey(0)
cv.destroyAllWindows()


# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# #### d)

# %% [markdown]
# The integral image is computed iteratively using the formula: S(x,y)=I(x-1,y-1)+S(x−1,y)+S(x,y−1)−S(x−1,y−1).
# Computed all integral images at given coordinate.
# Found sum regions using the formula to obtain box filtered image at all coordinates.
# 

# %%
import numpy as np
import cv2 as cv

def readfile(file):
    return '../images/' + file

def box_filter_integral_image(img, ksize):
    h, w, channels = img.shape
    
    # Compute integral image (S)
    integral_img = np.zeros((h+1, w+1, channels), dtype=np.uint64)
    
    for y in range(1, h+1):
        for x in range(1, w+1):
            for c in range(channels):
                integral_img[y, x, c] = (int(img[y-1, x-1, c]) 
                                         + integral_img[y-1, x, c] 
                                         + integral_img[y, x-1, c] 
                                         - integral_img[y-1, x-1, c])
    
    # Compute box filtered image
    box_filtered_img = np.zeros_like(img, dtype=np.uint8)
    
    half_k = ksize // 2
    for y in range(h):
        for x in range(w):
            for c in range(channels):
                x1 = max(x - half_k, 0)
                y1 = max(y - half_k, 0)
                x2 = min(x + half_k, w - 1)
                y2 = min(y + half_k, h - 1)
                
                # Compute sum using integral image
                sum_region = (integral_img[y2+1, x2+1, c] 
                              - integral_img[y1, x2+1, c] 
                              - integral_img[y2+1, x1, c] 
                              + integral_img[y1, x1, c])
                
                # Compute average
                box_filtered_img[y, x, c] = sum_region // (ksize * ksize)
    
    return box_filtered_img

img = cv.imread(readfile('coins.png'))
ksize = 5  # Kernel size of the box filter

filtered_img = box_filter_integral_image(img, ksize)


cv.imshow('Original Image', img)
cv.imshow('Box Filtered Image', filtered_img)
cv.waitKey(0)
cv.destroyAllWindows()


# %% [markdown]
# ![Screenshot 2024-06-29 at 18.54.35.png](attachment:df1cbea2-8d39-4c32-bbf7-d42ac80d4699.png)

# %% [markdown]
# ### POLYGON

# %% [markdown]
# Applied canny edge detector and finde the contours. Extract the biggest contour as card is likely to be largest. Refined the approximation by adjusting vertex position by updating the minimum distance points.

# %%
import cv2 as cv
import numpy as np

def readfile(file):
    return '../images/' + file


img = cv.imread(readfile('card_student.jpeg'))


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 100, 200)


contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


largest_contour = max(contours, key=cv.contourArea)


epsilon = 0.01 * cv.arcLength(largest_contour, True)
approx = cv.approxPolyDP(largest_contour, epsilon, True)


for i in range(len(approx)):
    vertex = approx[i][0]
    closest_point = None
    min_distance = float('inf')
    for point in largest_contour:
        distance = np.linalg.norm(vertex - point[0])
        if distance < min_distance:
            min_distance = distance
            closest_point = point[0]
    approx[i][0] = closest_point


cv.drawContours(img, [largest_contour], 0, (0, 0, 255), 2)
cv.drawContours(img, [approx], 0, (0, 255, 0), 2)


cv.imshow('Result', img)
cv.waitKey(0)
cv.destroyAllWindows()


# %% [markdown]
# ![Screenshot 2024-06-29 at 21.53.51.png](attachment:c1ae5205-72ac-47c1-8759-69fbed91f43e.png)

# %% [markdown]
# ### VROT

# %% [markdown]
# Firstly, we create mask from the live image and detect the corners to append as tracks. Then we calculate the avg displacement to detect the movement is whether up,down,left or right. Calculated average angular rotation by finding the angle between the last and first tracks and multiplying with the frame per pixel.
# We find the forward/rotation movement by comparing which vector type has a direction similar to residual vectors: radial or tangential vectors. If radial is more similar, it means it has a forward movement.
# We have a yellow residual vector by deducting the average movement from the residual vector. This shows a more clear way to see the direction.

# %%
#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream, sourceArgs
from umucv.util import putText
import time

tracks = []
track_len = 3
detect_interval = 5

corners_params = dict(maxCorners=500, qualityLevel=0.1, minDistance=10, blockSize=7)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def draw_arrow(image, start, end, color, thickness=2):
    cv.arrowedLine(image, start, end, color, thickness, tipLength=0.5)

prev_time = time.time()
for n, (key, frame) in enumerate(autoStream()):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    t0 = time.time()
    if len(tracks):
        p0 = np.float32([t[-1] for t in tracks])
        p1, _, _ = cv.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)
        p0r, _, _ = cv.calcOpticalFlowPyrLK(gray, prevgray, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(axis=1)
        good = d < 1

        new_tracks = []
        for t, (x, y), ok in zip(tracks, p1.reshape(-1, 2), good):
            if not ok:
                continue
            t.append([x, y])
            if len(t) > track_len:
                del t[0]
            new_tracks.append(t)

        tracks = new_tracks

        # Draw the trajectories
        cv.polylines(frame, [np.int32(t) for t in tracks], isClosed=False, color=(0, 0, 255))
        for t in tracks:
            x, y = np.int32(t[-1])
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Calculate the average displacement in x and y directions
        if len(tracks) > 0:
            avg_dx = sum([track[-1][0] - track[0][0] for track in tracks]) / len(tracks)
            avg_dy = sum([track[-1][1] - track[0][1] for track in tracks]) / len(tracks)

            # Determine the camera movement direction
            if abs(avg_dx) > abs(avg_dy):
                if avg_dx > 0:
                    camera_direction = "LEFT"
                else:
                    camera_direction = "RIGHT"
            else:
                if avg_dy > 0:
                    camera_direction = "UP"
                else:
                    camera_direction = "DOWN"

            # Estimate the angular speed of rotation
            avg_angle = sum([np.arctan2(t[-1][1] - t[0][1], t[-1][0] - t[0][0]) for t in tracks]) / len(tracks)
            avg_angle_deg = avg_angle * 180 / np.pi

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            angular_speed = avg_angle_deg * fps

            # Detect forward/backward movement and rotation around the optical axis
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            radial_vectors = np.array([[x - cx, y - cy] for x, y in p1.reshape(-1, 2)])
            tangential_vectors = np.array([[-y + cy, x - cx] for x, y in p1.reshape(-1, 2)])

            residual_vectors = p1.reshape(-1, 2) - p0.reshape(-1, 2)
            radial_similarity = np.dot(residual_vectors, radial_vectors.T) / (np.linalg.norm(residual_vectors, axis=1) * np.linalg.norm(radial_vectors, axis=1))
            tangential_similarity = np.dot(residual_vectors, tangential_vectors.T) / (np.linalg.norm(residual_vectors, axis=1) * np.linalg.norm(tangential_vectors, axis=1))

            if np.mean(radial_similarity) > np.mean(tangential_similarity):
                movement_type = "FORWARD"
                # Draw green radial vectors
                for (x, y), r in zip(p1.reshape(-1, 2), radial_vectors):
                    end_point = (int(x + r[0] * 10), int(y + r[1] * 10))
                    draw_arrow(frame, (int(x), int(y)), end_point, (0, 255, 0), 1)
            else:
                movement_type = "ROTATION"
                # Draw blue tangential vectors
                for (x, y), t in zip(p1.reshape(-1, 2), tangential_vectors):
                    end_point = (int(x + t[0] * 10), int(y + t[1] * 10))
                    draw_arrow(frame, (int(x), int(y)), end_point, (255, 0, 0), 1)

            # Draw red residual vectors
            for (x1, y1), (x2, y2) in zip(p0.reshape(-1, 2), p1.reshape(-1, 2)):
                draw_arrow(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

            # Draw yellow residual vectors after removing average movement
            for (x1, y1), (x2, y2) in zip(p0.reshape(-1, 2), p1.reshape(-1, 2)):
                x2 -= avg_dx
                y2 -= avg_dy
                draw_arrow(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)

            putText(frame, f'{len(tracks)} corners, {(time.time() - t0)*1000:.0f}ms, {camera_direction}, {angular_speed:.2f} deg/s, {movement_type}')

    t1 = time.time()

    if n % detect_interval == 0:
        mask = np.zeros_like(gray)
        mask[:] = 255
        for x, y in [np.int32(t[-1]) for t in tracks]:
            cv.circle(mask, (x, y), 5, 0, -1)
        corners = cv.goodFeaturesToTrack(gray, mask=mask, **corners_params)
        if corners is not None:
            for [(x, y)] in np.float32(corners):
                tracks.append([[x, y]])

    cv.imshow('input', frame)
    prevgray = gray

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()





# %% [markdown]
# ![Screenshot 2024-06-30 at 11.17.49.png](attachment:04e78823-e849-420c-a581-29e4682fc6d2.png)

# %% [markdown]
# ![vrot.png](attachment:3612fdaf-03ff-4a1f-9088-96d3c154105a.png)

# %% [markdown]
# ### DL

# %% [markdown]
# Installed ultralytics. Created a dataset file by my own animal images. Put them in the correct directories such as train,val,test. For the images' text annotations, I used labelImg application. Using a YOLO model, I trained my model and evaluated on my validation set.



# %%
from ultralytics import YOLO

# %%
dataset_yaml = """
train: /Users/cerenkaraoglan/umucv/notebooks/my_dataset/images/train
val: /Users/cerenkaraoglan/umucv/notebooks/my_dataset/images/val
test: /Users/cerenkaraoglan/umucv/notebooks/my_dataset/images/test

nc: 24 # number of classes
names: ['dog' ,'person', 'cat', 'tv', 'car', 'meatballs', 'marinara sauce', 'tomato soup', 'chicken noodle soup', 'french onion soup', 'chicken breast', 'ribs',
 'pulled pork', 'hamburger cavity',
'bird', 'duck', 'racoon', 'gorilla', 'monkey', 'elephant', 'antelope', 'giraffe', 'hippo', 'swan'
]  # list of class names
"""

with open('dataset.yaml', 'w') as file:
    file.write(dataset_yaml)

# %%
# Initialize the YOLO model
model = YOLO('yolov8n.yaml')  # You can choose different configurations like 'yolov8s.yaml', 'yolov8m.yaml', etc.

# Train the model
model.train(data='/Users/cerenkaraoglan/umucv/notebooks/my_dataset/dataset.yaml', epochs=100, imgsz=640)


# %%
# Evaluate the model on the validation set
results = model.val()

# Print the evaluation results
print(results)


# %% [markdown]
# ![Screenshot 2024-06-30 at 22.34.29.png](attachment:7d314962-20ad-4fc4-a44b-d3ffed182cc0.png)

# %% [markdown]
# ### CARD

# %% [markdown]
# Could not properly do this. Approach is using ORB to detect descriptors both in the card image and the frame. After finding card's location, time to find the photo section.(used top-left coordinate of photo in the card image). Resized the placeholder image to paste to the proper location.

# %%
import cv2
import numpy as np

def readfile(file):
    return '../images/' + file

student_card_template = cv.imread(readfile('card_student.jpeg'))
placeholder_image = cv.imread(readfile('AP-Trump-Mug-840x840.jpg'))

placeholder_image_height, placeholder_image_width = placeholder_image.shape[:2]

cap = cv2.VideoCapture(0)


orb = cv2.ORB_create()

kp_template, des_template = orb.detectAndCompute(student_card_template, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

if des_template is None:
    raise ValueError("No descriptors found in the template image.")


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the frame
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    # Ensure descriptors are found
    if des_frame is not None:
        # Match descriptors between the template and the frame
        matches = bf.match(des_template, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:
            # Extract location of good matches
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)

            # Find homography matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Get the dimensions of the student card template
                h, w = student_card_template.shape[:2]

                # Define the corners of the template
                pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

                # Transform the corners to the frame
                dst = cv2.perspectiveTransform(pts, M)

                # Draw the detected card area on the frame
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                # Get the perspective transform matrix to warp the card area
                M_warp = cv2.getPerspectiveTransform(dst, pts)

                # Warp the detected card area to a top-down view
                warped_card = cv2.warpPerspective(frame, M_warp, (w, h))

                # Define the coordinates of the photo area in the warped card (adjust as needed)
                x1, y1 = 180, 1088
                x2, y2 = x1 + placeholder_image_width, y1 + placeholder_image_height


                resized_placeholder_image = cv2.resize(placeholder_image, (x2 - x1, y2 - y1))


                warped_card[y1:y2, x1:x2] = resized_placeholder_image


                M_warp_inv = cv2.getPerspectiveTransform(pts, dst)
                frame = cv2.warpPerspective(warped_card, M_warp_inv, (frame.shape[1], frame.shape[0]), frame, borderMode=cv2.BORDER_TRANSPARENT)

    cv2.imshow('Student Card Photo Replacement', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()



# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### MODEL3D

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### PANO

# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob

def readrgb(file):
    return cv.cvtColor( cv.imread('../images/'+file), cv.COLOR_BGR2RGB) 

def desp(d):
    dx,dy = d
    return np.array([
            [1,0,dx],
            [0,1,dy],
            [0,0,1]])

pi = np.pi
degree = pi/180


sift = cv.SIFT_create()
bf = cv.BFMatcher()

def match(query, model):
    x1 = query
    x2 = model
    (k1, d1) = sift.detectAndCompute(x1, None)
    (k2, d2) = sift.detectAndCompute(x2, None)

    matches = bf.knnMatch(d1,d2,k=2)
    # ratio test
    good = []
    for m in matches:
        if len(m) == 2:
            best, second = m
            if best.distance < 0.75*second.distance:
                good.append(best)

    if len(good) < 6: return 6, None

    src_pts = np.array([ k2[m.trainIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)
    dst_pts = np.array([ k1[m.queryIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3)

    return sum(mask.flatten()>0), H

pano = [readrgb(x) for x in sorted(glob.glob('../images/pano/pano*.jpg'))]
h,w,_ = pano[6].shape
mw,mh = 400,100
T = desp((mw,float(mh)))
sz = (w+2*mw,h+2*mh)
base = cv.warpPerspective(pano[6], T , sz)


_,H67 = match(pano[6],pano[7])
cv.warpPerspective(pano[7],T@H67,sz, base, 0, cv.BORDER_TRANSPARENT)
_,H65 = match(pano[6],pano[5])
cv.warpPerspective(pano[5],T@H65,sz, base, 0, cv.BORDER_TRANSPARENT)
plt.imshow(base);

# %% [markdown]
# This also did not fully accomplished. I tried to find the keypoints and descriptors of the images using SIFT and stitched the randomly selected compatible pairs.

# %%
import cv2
import numpy as np
import os
import random

def stitch_random_images(image_folder):
    images = [cv2.imread(os.path.join(image_folder, f)) for f in os.listdir(image_folder)]


    keypoints = []
    descriptors = []
    for image in images:
        kp, desc = cv2.SIFT_create().detectAndCompute(image, None)
        keypoints.append(kp)
        descriptors.append(desc)

   
    stitcher = cv2.Stitcher_create()

    # Select compatible pairs and stitch them
    while True:
        # Randomly select a pair of images
        i, j = random.sample(range(len(images)), 2)

       
        matcher = cv2.BFMatcher()
        matches = matcher.match(descriptors[i], descriptors[j])

        # Check if the pair is compatible
        if len(matches) > 10 and len(matches) / min(len(keypoints[i]), len(keypoints[j])) > 0.1:
            # Estimate the homography
            src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[j][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

            # Stitch the images
            status, result = stitcher.stitch([images[i], images[j]], H)
            if status == cv2.STITCHER_OK:
                break

    return result

# Example usage
image_folder = '/Users/cerenkaraoglan/umucv/images/pano/'
stitched_image = stitch_random_images(image_folder)


# Display the panorama
plt.figure(figsize=(12, 6))
plt.imshow(stitched_image)
plt.axis('off')
plt.show()

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### SUDOKU

# %% [markdown]
# This didn't work contrary to what I expect and I couldn't find why. I firstly preprocessed the image to find the contours better. Found the sudoku grid by setting a large area and 4 vertices which implies it is a sudoku grid. Then I warped the image. Installed the pytesseract library to use the OCR method, to changing digit photos into string. To extract digits, I iteratively go over the cells.( created the cells array by adjusting the cell size to split)

# %%
import cv2 as cv
import numpy as np
import pytesseract
from matplotlib import pyplot as plt



def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to find Sudoku grid
def find_sudoku_grid(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    largest_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 10000: 
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                largest_contour = approx
                break
    
    if largest_contour is None:
        return None
    
    return largest_contour.reshape(4, 2)


def warp_perspective(image, grid_corners):
    grid_corners = np.float32(grid_corners)
    dest_corners = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
    
    matrix = cv.getPerspectiveTransform(grid_corners, dest_corners)
    warped = cv.warpPerspective(image, matrix, (450, 450))
    
    return warped

# Function to extract digits from Sudoku cells using OCR
def extract_digits_from_cells(cells):
    cells_digits = []
    for row in range(9):
        row_digits = []
        for col in range(9):
            cell = cells[row * 9 + col]
            gray = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
            text = pytesseract.image_to_string(thresh, config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
            if text.isdigit():
                row_digits.append(int(text))
            else:
                row_digits.append(0)
        cells_digits.append(row_digits)
    
    return cells_digits

def process_camera_feed():
    cap = cv.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
     
        processed_frame = preprocess_image(frame)
        
        # Find the Sudoku grid
        grid_corners = find_sudoku_grid(processed_frame)
        
        if grid_corners is not None:
            # Warp perspective to extract Sudoku cells
            warped_image = warp_perspective(frame, grid_corners)

            # Divide the Sudoku grid into cells
            cells = []
            cell_size = warped_image.shape[0] // 9
            for row in range(9):
                for col in range(9):
                    cell = warped_image[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size]
                    cells.append(cell)

            # Extract digits from cells using OCR
            sudoku_digits = extract_digits_from_cells(cells)

            for row in sudoku_digits:
                print(row)

            # Display the Sudoku grid and extracted digits
            cv.imshow('Warped Sudoku Image', warped_image)
        
        # Display the original frame
        cv.imshow('Live Camera Feed', frame)
        
        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all windows
    cap.release()
    cv.destroyAllWindows()

# Call the function to start processing the live camera feed
process_camera_feed()



