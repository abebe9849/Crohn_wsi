import numpy as np 
import pandas as pd 
import glob
import openslide
import matplotlib.pyplot as plt
import cv2,math,sys,time,os
from tqdm import tqdm
import itertools
import warnings
from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d
#predcit = pd.read_csv("/home/u094724e/ダウンロード/byori/cam/outputs/2021-04-22/09-31-24/predict.csv")
date = "/home/u094724e/ダウンロード/byori/cam/outputs/2021-04-10/06-33-22"
predcit = pd.read_csv(f"{date}/predict.csv")
predcit = predcit.sort_values("pred_1")

def water_shed(gray_img,color_img,p=0.1):
        ### ~~~~~~~~~~
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sure_bg = cv2.dilate(gray_img, kernel, iterations=2)## はいぱら
    dist = cv2.distanceTransform(gray_img, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, p * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    #fig = plt.figure(figsize=(20,20))
    #plt.imshow(dist,cmap="gray")
    #plt.show()
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_labels, markers = cv2.connectedComponents(sure_fg)
    """
    fig = plt.figure(figsize=(20,20))
    plt.imshow(markers)
    plt.show()
    """
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(color_img, markers)
    return markers

def large_tile(df_tmp,idx):
    img =cv2.imread(df_tmp["file_png"].values[idx])
    img_l_path = df_tmp["file_ndpi"].values[idx]
    img_y = df_tmp["np_y"].values[idx]
    img_x = df_tmp["np_x"].values[idx]
    slide = openslide.OpenSlide(img_l_path)
    pil_img = slide.read_region((img_x*8,img_y*8),0,(128*8,128*8))
    np_img = np.asarray(pil_img)[:,:,:3]
    return np_img

def img_and(mask,canny):
    """
    共にuint8のndarray0~255
    img1:mask
    img2:canny
    
    """
    canny = (255-canny)/255
    mask = mask/255
    
    new_img = np.logical_and(canny,mask)
    return new_img
    
def sharpen_image(img,k):
    #kernel = np.array([[-k / 9, -k / 9, -k / 9],[-k / 9, 1 + 8 * k / 9, k / 9],[-k / 9, -k / 9, -k / 9]], np.float32)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], np.float32) 
    new_img = cv2.filter2D(img, -1, kernel).astype("uint8")
    #new_img2 = cv2.filter2D(new_img, -1, kernel).astype("uint8")
    return new_img


def each_cell(markers,thresh=0.03,plot=True):
    u, counts = np.unique(markers.ravel(), return_counts=True)
    areas = []
    emp = np.zeros_like(markers)
    margin_num = 0
    for cls,s in zip(u,counts):
        if cls<0:
            """
            tmp_img = (markers == cls)
            plt.imshow(tmp_img)
            plt.show()
            """
            continue#枠(-1)
        if s<500:continue
        if s >1024*1024*thresh:
            if plot:
                tmp_img = (markers == cls)
                plt.title(f"class{cls} area{s}")
                plt.imshow(tmp_img)
                plt.show()
            margin_num+=1
            continue#大きい余白を除く
        tmp_img = (markers == cls)
        emp[np.where(markers == cls)]=cls
        if cls%200==0 and plot:
            plt.title(f"class{cls} area{s}")
            plt.imshow(tmp_img)
            plt.show()
        areas.append(s)
        #if cls>20:break
    back_area = counts[2]
    #print(back_area)
    return emp,areas,back_area,margin_num

def margin_to_black(markers,color_img,gray_img,distant=None,p=0.1,thresh = 0.03,thresh_margin=0.05,plot=False):
    u, counts = np.unique(markers.ravel(), return_counts=True)

    """
    for cls,s in zip(u,counts):
        if s >1024*1024*thresh_margin:
            gray_img[np.where(markers == cls)]=0"""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sure_bg = cv2.dilate(gray_img, kernel, iterations=2)## はいぱら
    if distant is None:
        
        dist = cv2.distanceTransform(gray_img, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, p * dist.max(), 255, cv2.THRESH_BINARY)
    else:
        """
        for cls,s in zip(u,counts):
            if s >1024*1024*thresh_margin:
                distant[np.where(markers == cls)]=0"""
        dist = cv2.distanceTransform(distant, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, p * dist.max()*0.05, 255, cv2.THRESH_BINARY)

    
    sure_fg = sure_fg.astype(np.uint8)
    #fig = plt.figure(figsize=(20,20))
    #print("distant!!!!")
    #plt.imshow(dist,cmap="gray")
    #plt.show()
    #plt.imshow(gray_img,cmap="gray")
    #plt.show()
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_labels, markers = cv2.connectedComponents(sure_fg)
    """
    fig = plt.figure(figsize=(20,20))
    plt.imshow(markers)
    plt.show()
    """
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(color_img, markers)
    
    u, counts = np.unique(markers.ravel(), return_counts=True)
    areas = []
    emp = np.zeros_like(markers)
    margin_num = 0
    for cls,s in zip(u,counts):
        if cls<0:
            """
            tmp_img = (markers == cls)
            plt.imshow(tmp_img)
            plt.show()
            """
            continue#枠(-1)
        if s<500:continue
        if s >1024*1024*thresh:
            margin_num+=1
            if plot:
                tmp_img = (markers == cls)
                plt.title(f"class{cls} area{s}")
                plt.imshow(tmp_img)
                plt.show()
            
            continue#大きい余白を除く
        tmp_img = (markers == cls)
        emp[np.where(markers == cls)]=cls
        if cls%200==0 and plot:
            plt.title(f"class{cls} area{s}")
            plt.imshow(tmp_img)
            plt.show()
        areas.append(s)
        #if cls>20:break
    back_area = counts[2]
    #print(back_area)
    return emp,areas,back_area,margin_num

def analyze_data(areas,backs,figname,distants=None,ellipses_rates=None):
    nums = [len(area) for area in areas]
    plt.figure()
    plt.hist(np.array(nums),bins=100)
    if figname:
        plt.savefig(f"{figname}_num.png")
    else:
        plt.show()
    areas_f = list(itertools.chain.from_iterable(areas))
    plt.figure()
    plt.title(f"mean{np.mean(np.array(areas_f))},medain{np.median(np.array(areas_f))}")
    plt.hist(np.array(areas_f),bins=100)
    if figname: 
        plt.savefig(f"{figname}_areas.png")
    else:
        plt.show()
    plt.figure()
    plt.title(f"mean{np.mean(np.array(backs))},medain{np.median(np.array(backs))}")
    plt.hist(np.array(backs),bins=100)
    if figname:
        plt.savefig(f"{figname}_backs.png")
    else:
        plt.show()

    if distants:
        distants_f = list(itertools.chain.from_iterable(distants))
        plt.figure()
        plt.hist(np.array(distants_f),bins=100)
        plt.savefig(f"{figname}_distants.png")
        with open(f"{figname}_distants.pickle", 'wb') as f:
            pickle.dump(distants, f)

    if ellipses_rates:
        ellipses_rates_f = list(itertools.chain.from_iterable(ellipses_rates))
        plt.figure()
        plt.hist(np.array(ellipses_rates_f),bins=100)
        plt.savefig(f"{figname}_ellipses_rates.png")
        with open(f"{figname}_ellipses_rates.pickle", 'wb') as f:
            pickle.dump(ellipses_rates, f)



    with open(f"{figname}_backs.pickle", 'wb') as f:
        pickle.dump(backs, f)
    with open(f"{figname}_areas.pickle", 'wb') as f:
        pickle.dump(areas, f)
    with open(f"{figname}_nums.pickle", 'wb') as f:
        pickle.dump(nums, f)

def compare_data(areas1,backs1,areas2,backs2,distant=None,ellipses_rate=None,figname=None):
    nums1 = [len(area) for area in areas1]
    nums2 = [len(area) for area in areas2]
    plt.figure()
    plt.hist(np.array(nums1),bins=100)
    plt.hist(np.array(nums2),bins=100)

    if figname:
        plt.savefig(f"{figname}_num.png")
    else:
        plt.show()
    areas_f1 = list(itertools.chain.from_iterable(areas1))
    areas_f2 = list(itertools.chain.from_iterable(areas2))
    plt.figure()
    plt.title(f"mean{np.mean(np.array(areas_f1))},{np.mean(np.array(areas_f2))},medain{np.median(np.array(areas_f1))},{np.median(np.array(areas_f2))}")
    plt.hist(np.array(areas_f1),bins=100)
    plt.hist(np.array(areas_f2),bins=100)
    if figname: 
        plt.savefig(f"{figname}_areas.png")
    else:
        plt.show()
    plt.figure()
    plt.title(f"mean{np.mean(np.array(backs1))},{np.mean(np.array(backs2))},medain{np.median(np.array(backs1))},{np.median(np.array(backs2))}")
    plt.hist(np.array(backs1),bins=100)
    plt.hist(np.array(backs2),bins=100)
    if figname:
        plt.savefig(f"{figname}_backs.png")
    else:
        plt.show()

    if distant:
        distant_f1 = list(itertools.chain.from_iterable(distant[0]))
        distant_f2 = list(itertools.chain.from_iterable(distant[1]))
        plt.figure()
        plt.title(f"mean{np.mean(np.array(distant_f1))},{np.mean(np.array(distant_f2))},medain{np.median(np.array(distant_f1))},{np.median(np.array(distant_f2))}")
        plt.hist(np.array(distant_f1),bins=100)
        plt.hist(np.array(distant_f2),bins=100)
        plt.savefig(f"{figname}_distant.png")
    if ellipses_rate:
        ellipses_rate_f1 = list(itertools.chain.from_iterable(ellipses_rate[0]))
        ellipses_rate_f2 = list(itertools.chain.from_iterable(ellipses_rate[1]))
        plt.figure()
        plt.title(f"mean{np.mean(np.array(ellipses_rate_f1))},{np.mean(np.array(ellipses_rate_f2))},medain{np.median(np.array(ellipses_rate_f1))},{np.median(np.array(ellipses_rate_f2))}")
        plt.hist(np.array(ellipses_rate_f1),bins=100)
        plt.hist(np.array(ellipses_rate_f2),bins=100)
        plt.savefig(f"{figname}_ellipses_rate.png")



    
import pickle
def calc_analyze(df_tmp,label,num):
    areas_ = []
    backgrounds_ = []
    for idx in tqdm(range(num)):
        if label==0:
            img = large_tile(predcit,idx)
        else:
            img = large_tile(predcit,-(idx+1))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        retval, mask = cv2.threshold(gray, thresh=gray.mean(0).mean(0), maxval=255, type=cv2.THRESH_BINARY)
        markers = water_shed(mask,img)
        new_markers,areas,back_area,margin_num = each_cell(markers,plot=False)#こちらのほうがきれいにとれる
        if len(areas)<20:##len(areas)<20のものはなにか問題ある(余白が半分以上、他の組織、ぼやけてる)
            gray_3 = cv2.Canny(gray,0,30)
            kernel_size = (5,5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            dilation = cv2.dilate(gray_3,kernel,iterations = 1)
            dilation = 255-dilation
            gray_3 = img_and(mask,gray_3)
            gray_3 = (gray_3*255).astype("uint8")
            gray_color = cv2.cvtColor(gray_3,cv2.COLOR_GRAY2RGB)
            new_markers_2,areas_new,back_area_new,margin_num_new=margin_to_black(markers,img,mask,distant=dilation,p=0.1,thresh = 0.03,thresh_margin=0.05,plot=False)
            
        """
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(1, 4, 1)
        plt.imshow(img)
        ax2 = fig.add_subplot(1, 4, 2)
        plt.imshow(markers)
        ax3 = fig.add_subplot(1, 4, 3)
        plt.imshow(new_markers)
        """
        backgrounds_.append(back_area)
        if len(areas)<20 :
            #ax4 = fig.add_subplot(1, 4, 4)
            #plt.title(f"old {len(areas)}:{margin_num}^new{len(areas_new)}:{margin_num_new}")
            #plt.imshow(new_markers_2)
            areas_.append(areas_new)
        else:
            areas_.append(areas)
        #plt.show()
        
    return areas_,backgrounds_

def each_cell_vori(markers,thresh=0.03,plot=True):
    u, counts = np.unique(markers.ravel(), return_counts=True)
    areas = []
    emp = np.zeros_like(markers)
    margin_num = 0
    points = []##ボロノイ分割よう
    ellipses  = []
    ellipses_rate = []
    emp_2 = np.zeros_like(markers)
    
    for cls,s in zip(u,counts):
        if cls<0:
            """
            tmp_img = (markers == cls)
            plt.imshow(tmp_img)
            plt.show()
            """
            continue#枠(-1)
        if s<500:continue
        if s >1024*1024*thresh:
            if plot:
                tmp_img = (markers == cls)
                plt.title(f"class{cls} area{s}")
                plt.imshow(tmp_img)
                plt.show()
            margin_num+=1
            continue#大きい余白を除く
        tmp_img = (markers == cls)
        emp[np.where(markers == cls)]=cls
        ### 距離計測
        emp_2[np.where(markers == cls)]=cls
        mu = cv2.moments(emp_2.astype("uint8"), False)
        try:
            x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
            points.append([x,y])
        except Exception as e:
            print("center not...")
            x,y = (0,0)
            points.append([x,y])

        
        contours, hierarchy = cv2.findContours(emp_2.astype("uint8"),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#RETR_LIST
        #print("~~~~",len(contours))
        

        mx_area = 0
        for cont in contours:
            
            x_,y_,w,h = cv2.boundingRect(cont)
            area = w*h
        
            if area > mx_area:
                mx = x_,y_,w,h
                mx_area = area
                if area<10 or len(cont)<5:continue
                try:
                    ellipse = cv2.fitEllipse(cont)

                except Exception as e:
                    print(len(cont))
           # ellipse_h = ellipse[1][0]
           # ellipse_w = ellipse[1][1]

        try:
            ellipses.append([ellipse[1][0],ellipse[1][1]])
            ellipses_rate.append(ellipse[1][0]/ellipse[1][1])##h/w
        except Exception as e:
            print(e)
            #sys.exit()
        
        
        emp_2 = cv2.circle(emp_2, (x,y), 4, 100, 2, 4)
        #plt.imshow(emp_2)
        #plt.show()
        
        emp_2[np.where(markers == cls)]=0
        
        
        if cls%200==0 and plot:
            plt.title(f"class{cls} area{s}")
            plt.imshow(tmp_img)
            plt.show()
        areas.append(s)
        #if cls>20:break
    back_area = counts[2]
    #print(back_area)
    #plt.imshow(emp_2)
    #plt.show()
    

    for xy in  points:
        x = xy[0]
        y =xy[1]        
        emp = cv2.circle(emp, (x,y), 4, 100, 2, 4)
    #plt.imshow(emp)
    #plt.show()
    norms = []
    try:
        tri = Delaunay(points)
        tri_points = tri.points
        #fig = delaunay_plot_2d(tri)
        #plt.show()
        
        for i in range(len(tri_points)):
            for j in range(i,len(tri_points)):
                if i==j:continue
                x_i = tri_points[i][0]
                y_i = tri_points[i][1]
                x_j = tri_points[j][0]
                y_j = tri_points[j][1]
                nolm = ((x_i-x_j)**2+(y_i-y_j)**2)**0.5
                norms.append(nolm)
    except Exception as e:
        plt.imshow(emp)
        plt.savefig("error.png")
        plt.imshow(markers)
        plt.savefig("error1.png")


        
    return emp,areas,back_area,margin_num,norms,ellipses_rate


def margin_to_black_plus(markers,color_img,gray_img,distant=None,p=0.1,thresh = 0.03,thresh_margin=0.05,plot=False):
    ##細胞間距離、円度合いをついか
    u, counts = np.unique(markers.ravel(), return_counts=True)
    points = []##ボロノイ分割よう
    ellipses  = []
    ellipses_rate = []
    emp_2 = np.zeros_like(markers)

    """
    for cls,s in zip(u,counts):
        if s >1024*1024*thresh_margin:
            gray_img[np.where(markers == cls)]=0"""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sure_bg = cv2.dilate(gray_img, kernel, iterations=2)## はいぱら
    if distant is None:
        
        dist = cv2.distanceTransform(gray_img, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, p * dist.max(), 255, cv2.THRESH_BINARY)
    else:
        """
        for cls,s in zip(u,counts):
            if s >1024*1024*thresh_margin:
                distant[np.where(markers == cls)]=0"""
        dist = cv2.distanceTransform(distant, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, p * dist.max()*0.05, 255, cv2.THRESH_BINARY)

    
    sure_fg = sure_fg.astype(np.uint8)
    #fig = plt.figure(figsize=(20,20))
    #print("distant!!!!")
    #plt.imshow(dist,cmap="gray")
    #plt.show()
    #plt.imshow(gray_img,cmap="gray")
    #plt.show()
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_labels, markers = cv2.connectedComponents(sure_fg)
    """
    fig = plt.figure(figsize=(20,20))
    plt.imshow(markers)
    plt.show()
    """
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(color_img, markers)
    
    u, counts = np.unique(markers.ravel(), return_counts=True)
    areas = []
    emp = np.zeros_like(markers)
    margin_num = 0
    for cls,s in zip(u,counts):
        if cls<0:
            """
            tmp_img = (markers == cls)
            plt.imshow(tmp_img)
            plt.show()
            """
            continue#枠(-1)
        if s<500:continue
        if s >1024*1024*thresh:
            margin_num+=1
            if plot:
                tmp_img = (markers == cls)
                plt.title(f"class{cls} area{s}")
                plt.imshow(tmp_img)
                plt.show()
            
            continue#大きい余白を除く
        tmp_img = (markers == cls)
        emp[np.where(markers == cls)]=cls
        
        ### 距離計測
        emp_2[np.where(markers == cls)]=cls
        mu = cv2.moments(emp_2.astype("uint8"), False)

        try:
            x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
            points.append([x,y])
        except Exception as e:
            print("center not...")
            x,y = (0,0)
            points.append([x,y])
        
        contours, hierarchy = cv2.findContours(emp_2.astype("uint8"),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#RETR_LIST
        #print("~~~~",len(contours))
        
        mx_area = 0
        for cont in contours:
            
            x_,y_,w,h = cv2.boundingRect(cont)
            area = w*h
        
            if area > mx_area:
                mx = x_,y_,w,h
                mx_area = area
                if area<10 or len(cont)<5:continue
                ellipse = cv2.fitEllipse(cont)
           # ellipse_h = ellipse[1][0]
           # ellipse_w = ellipse[1][1]
        try:
            ellipses.append([ellipse[1][0],ellipse[1][1]])
        except Exception as e:
            plt.imshow(color_img)
            plt.savefig("error2.png")
            plt.imshow(emp)
            plt.savefig("error.png")
            plt.imshow(gray_img)
            plt.savefig("error3.png")
            sys.exit()
        
        ellipses_rate.append(ellipse[1][0]/ellipse[1][1])##h/w
        
        emp_2 = cv2.circle(emp_2, (x,y), 4, 100, 2, 4)
        #plt.imshow(emp_2)
        #plt.show()
        emp_2[np.where(markers == cls)]=0
        
        if cls%200==0 and plot:
            plt.title(f"class{cls} area{s}")
            plt.imshow(tmp_img)
            plt.show()
        areas.append(s)
        #if cls>20:break
    back_area = counts[2]
    
    for xy in  points:
        x = xy[0]
        y =xy[1]        
        emp = cv2.circle(emp, (x,y), 4, 100, 2, 4)
    #plt.imshow(emp)
    #plt.show()
    norms = []

    try:
        tri = Delaunay(points)
        tri_points = tri.points
        #fig = delaunay_plot_2d(tri)
        #plt.show()
        
        for i in range(len(tri_points)):
            for j in range(i,len(tri_points)):
                if i==j:continue
                x_i = tri_points[i][0]
                y_i = tri_points[i][1]
                x_j = tri_points[j][0]
                y_j = tri_points[j][1]
                nolm = ((x_i-x_j)**2+(y_i-y_j)**2)**0.5
                norms.append(nolm)
    except Exception as e:
        plt.imshow(emp)
        plt.savefig("error.png")
        plt.imshow(markers)
        plt.savefig("error1.png")
        plt.imshow(color_img)
        plt.savefig("error2.png")
        print(len(areas))
    return emp,areas,back_area,margin_num,norms,ellipses_rate


def calc_analyze_plus(df_tmp,label,num):
    areas_ = []
    backgrounds = []
    distants = []
    ellipses_rates = []
    for idx in tqdm(range(num)):
        if label==0:
            img = large_tile(predcit,idx)
        else:
            img = large_tile(predcit,-(idx+1))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        retval, mask = cv2.threshold(gray, thresh=gray.mean(0).mean(0), maxval=255, type=cv2.THRESH_BINARY)
        markers = water_shed(mask,img)
        #new_markers,areas,back_area,margin_num = each_cell(markers,plot=False)#こちらのほうがきれいにとれる
        new_markers,areas,back_area,margin_num,norms,ellipses_rate = each_cell_vori(markers,plot=False)
        if len(areas)<20:##len(areas)<20のものはなにか問題ある(余白が半分以上、他の組織、ぼやけてる)
            gray_3 = cv2.Canny(gray,0,30)
            kernel_size = (5,5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            dilation = cv2.dilate(gray_3,kernel,iterations = 1)
            dilation = 255-dilation
            gray_3 = img_and(mask,gray_3)
            gray_3 = (gray_3*255).astype("uint8")
            gray_color = cv2.cvtColor(gray_3,cv2.COLOR_GRAY2RGB)
            new_markers_2,areas_new,back_area_new,margin_num_new,norms_new,ellipses_rate_new  = margin_to_black_plus(markers,img,mask,distant=dilation,p=0.1,thresh = 0.03,thresh_margin=0.05,plot=False)
            #new_markers_2,areas_new,back_area_new,margin_num_new=margin_to_black(markers,img,mask,distant=dilation,p=0.1,thresh = 0.03,thresh_margin=0.05,plot=False)
            
        """
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(1, 4, 1)
        plt.imshow(img)
        ax2 = fig.add_subplot(1, 4, 2)
        plt.imshow(markers)
        ax3 = fig.add_subplot(1, 4, 3)
        plt.imshow(new_markers)
        """
        
        if len(areas)<20 and len(areas)>0:
            #ax4 = fig.add_subplot(1, 4, 4)
            #plt.title(f"old {len(areas)}:{margin_num}^new{len(areas_new)}:{margin_num_new}")
            #plt.imshow(new_markers_2)
            areas_.append(areas_new)
            distants.append(norms_new)
            ellipses_rates.append(ellipses_rate_new)
            backgrounds.append(back_area)
        elif len(areas)>=20:
            areas_.append(areas)
            distants.append(norms)
            ellipses_rates.append(ellipses_rate)
            backgrounds.append(back_area)
        #plt.show()
        
    return areas_,backgrounds,distants,ellipses_rates


import os

date = "/home/u094724e/ダウンロード/byori/cam/analyze_fat/remove_re"
new_dir_path = f"{date}/analyze"

try:
    os.mkdir(new_dir_path)
except Exception as e:
    print(e)

predcit = pd.read_csv("/home/u094724e/ダウンロード/byori/cam/analyze_fat/predict_0410_p_re.csv")
areas1,backgrounds1,distants1,ellipses_rates1 = calc_analyze_plus(predcit,0,len(predcit))
analyze_data(areas1,backgrounds1,figname=f"{date}/analyze/pred_p_re",distants=distants1,ellipses_rates=ellipses_rates1)

predcit = pd.read_csv("/home/u094724e/ダウンロード/byori/cam/analyze_fat/predict_0410_n_re.csv")
areas1,backgrounds1,distants1,ellipses_rates1 = calc_analyze_plus(predcit,0,len(predcit))
analyze_data(areas1,backgrounds1,figname=f"{date}/analyze/pred_n_re",distants=distants1,ellipses_rates=ellipses_rates1)
sys.exit()

areas1,backgrounds1,distants1,ellipses_rates1 = calc_analyze_plus(predcit,1,4000)
analyze_data(areas1,backgrounds1,figname=f"{date}/analyze/pred_1",dstants=distants1,ellipses_rates=ellipses_rates1)

areas0,backgrounds0,distants0,ellipses_rates0 = calc_analyze_plus(predcit,0,4000)
analyze_data(areas0,backgrounds0,figname=f"{date}/analyze/pred_0",distants=distants0,ellipses_rates=ellipses_rates0)

compare_data(areas1,backgrounds1,areas0,backgrounds0,distant=[distants1,distants0],ellipses_rate=[ellipses_rates1,ellipses_rates0],figname=f"{date}/analyze//compare")
sys.exit()


areas1,backgrounds1 = calc_analyze(predcit,1,2000)
analyze_data(areas1,backgrounds1,figname=f"{date}/analyze/pred_1")
areas0,backgrounds0 = calc_analyze(predcit,0,2000)
analyze_data(areas0,backgrounds0,figname=f"{date}/analyze/pred_0")

compare_data(areas1,backgrounds1,areas0,backgrounds0,distant=None,ellipses_rate=None,figname=f"{date}/analyze//compare")
