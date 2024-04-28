
import numpy as np # type: ignore
import cv2 as cv # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from reportlab.pdfgen import canvas # type: ignore
from reportlab.lib.pagesizes import letter # type: ignore
from reportlab.lib import colors # type: ignore
from reportlab.lib.units import inch, cm # type: ignore
from PIL import Image  # type: ignore

def page_flip_capture(video_path):
    cap = cv.VideoCapture(video_path)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 200,
                        qualityLevel = 0.1,
                        minDistance = 2,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (200, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    flip_detected = True
    need_to_cap = True 

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    saved_frames = []
    frame_diff_list = []
    plot_diff_list = []
    frame_idx_list = []
    frame_idx = 0
    idx = 0
    green_signal_list = []
    red_signal_list = []

    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No more frames grabbed!')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if flip_detected:
            mask1 = np.zeros_like(frame_gray)
            mask1[:] = 255   
            # print('good features to track')
            p0 = cv.goodFeaturesToTrack(old_gray, mask = mask1, **feature_params)
            flip_detected = False
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

            # print(good_new)
            # print(good_old)

            if(len(good_new) == len(good_old)):
                frame_diff = 0
                for i in range(len(good_new)):
                    old_pos_x = good_old[i][0]
                    old_pos_y = good_old[i][1]
                    new_pos_x = good_new[i][0]
                    new_pos_y = good_new[i][1]

                    frame_diff += (old_pos_x - new_pos_x)**2 + (old_pos_y - new_pos_y)**2

                frame_diff /= len(good_old)
                frame_idx += 1
                frame_idx_list.append(frame_idx)
                frame_diff_list.append(frame_diff)
                # print(frame_diff)

                avg = 0
                if len(frame_diff_list) >= 20:
                    for i in range(-1, -21, -1):
                        avg += frame_diff_list[i]
                
                avg /= 20

                plot_diff_list.append(avg)
                # print(avg, 'avgggggggg\n')
                if avg >= 500:
                    # print(" flip detected !!!!!11")
                    # print(" flip detected !!!!!11")
                    need_to_cap = True 
                    green_signal_list.append(frame_idx)

                if need_to_cap and avg < 1.0 and avg > 0.0:
                    # print("capture called!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    # print("capture called!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    name = 'saved_frame_1_%d.jpg'%idx
                    cv.imwrite(name, frame)

                    # cframe = frame.copy()

                    saved_frames.append(frame)
                    mask = np.zeros_like(old_frame)
                    old_gray = frame_gray.copy()
                    need_to_cap = False 
                    flip_detected = True 
                    red_signal_list.append(frame_idx)
                    idx += 1

                    continue


                # print('frame diff')


        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 1)
            frame = cv.circle(frame, (int(a), int(b)), 3, color[i].tolist(), -1)
        
        cframe = frame.copy()
        cmask = mask.copy()
        cframe = cv.resize(cframe, (900, 900)) 
        cmask = cv.resize(cmask, (900, 900)) 
        img = cv.add(cframe, cmask)

        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv.destroyAllWindows()
    x = frame_idx_list 
    y = plot_diff_list

    plt.plot(x, y) 

    plt.xlabel('frame number') 
    plt.ylabel('motion strength') 

    plt.vlines(green_signal_list, ymin = 0, ymax = max(plot_diff_list),
           colors = 'green',
           label = 'page_flipping detected', alpha = 0.2)
    
    plt.vlines(red_signal_list, ymin = 0, ymax = max(plot_diff_list),
           colors = 'red',
           label = 'frame_to_capture_detected')
    
    plt.show() 

    return saved_frames;

def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)

def scan(img):
    # Resize image to workable size
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv.resize(img, None, fx=resize_scale, fy=resize_scale)
    # Create a copy of resized original image for later use
    orig_img = img.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv.Canny(gray, 0, 200)
    canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
 
    # Finding contours for the detected edges.
    contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv.contourArea, reverse=True)[:5]
 
    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return orig_img
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv.arcLength(c, True)
        corners = cv.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    # For 4 corner points being detected.
    corners = order_points(corners)
 
    destination_corners = find_dest(corners)
 
    h, w = orig_img.shape[:2]
    # Getting the homography.
    M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv.INTER_LINEAR)
    # cv.imshow('final',final)
    return final


def create_pdf(images_list):
    res = canvas.Canvas("content_pdf_1.pdf", pagesize=letter)
    res.setFillColor(colors.grey)

    idx = 0
    for saved_image in images_list:
        # res.drawImage(saved_image, 1080, 0)
    
        # image_path = "saved_frame_7_0.jpg"
        # img = utils.ImageReader(image_path)
        # img = Image.open(saved_image)
        img = saved_image
        img_height, img_width = img.shape[:2]
        aspect = img_height / float(img_width)

        display_width = 500
        display_height = (display_width * aspect)

        img_name = "page%i.jpg"%idx
        cv.imwrite(img_name, img)
        res.drawImage(img_name, 15, 0, width=display_width, height=display_height)
        res.showPage()
        idx += 1
    res.save()

# create_pdf(images_list=images_list)

if __name__ == "__main__":
    video_path = "1.mp4"
    saved_frames = page_flip_capture(video_path)
    print("frame capturing done!")
    scanned_images_list = []
    for image1 in saved_frames:
        scanned_image = scan(image1)
        scanned_images_list.append(scanned_image) 
    print("scanning done!")
    create_pdf(images_list = scanned_images_list)
    print("pdf creation done!")

