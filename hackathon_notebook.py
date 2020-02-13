# coding: utf-8

import cv2
import shutil
import os
import glob
import traceback
import pandas as pd
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
# get_ipython().magic('matplotlib inline')


def get_sub_images(X, kp_pdf, show_dense='N'):  
    # find number of clusters
    X_db= StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.25, min_samples=10).fit(X_db)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
#     if show_dense == 'Y':
#         colors = [plt.cm.Spectral(each)
#                   for each in np.linspace(0, 1, len(unique_labels))]
#         for k, col in zip(unique_labels, colors):
#             if k == -1:
#                 # Black used for noise.
#                 col = [0, 0, 0, 1]

#             class_member_mask = (labels == k)

#             xy = X[class_member_mask & core_samples_mask]
#             plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                      markeredgecolor='k', markersize=10)

#             xy = X[class_member_mask & ~core_samples_mask]
#             plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                      markeredgecolor='k', markersize=4)
#         plt.show()
    # get final clusters
    clustering = AgglomerativeClustering(linkage='average', n_clusters=n_clusters_)
    clustering.fit(X)
    colors = [plt.cm.Spectral(each)for each in np.linspace(0, 1, n_clusters_)]
    
    # create sub images 
    sub_images = {}

    for j in set(clustering.labels_):
        indices = [i for i, x in enumerate(clustering.labels_) if x == j]
        vert_1 = (min(X[indices][:,0]),min(X[indices][:,1]))
        vert_2 = (max(X[indices][:,0]),min(X[indices][:,1]))
        vert_3 = (max(X[indices][:,0]),max(X[indices][:,1]))
        vert_4 = (min(X[indices][:,0]),max(X[indices][:,1]))
        sub_images[str(j)] = np.array([vert_1,vert_2,vert_3,vert_4]) 
    
    # plot final
    f = plt.figure()
    for j in set(clustering.labels_):
        indices = [i for i, x in enumerate(clustering.labels_) if x == j]
        plt.plot(X[indices][:,0],X[indices][:,1],'o', markerfacecolor=tuple(colors[j]),
                 markeredgecolor='k', markersize=5, hold=True)
        plt.plot(sub_images[str(j)][:,0],sub_images[str(j)][:,1],'k')
        plt.plot([sub_images[str(j)][0,0],sub_images[str(j)][3,0]],[sub_images[str(j)][0,1],sub_images[str(j)][3,1]],'k')
    # plt.show()
    f.savefig(kp_pdf)
    return sub_images


# In[15]:

def crop_image(img, cluster_val, cluster_name):
    min_points = cluster_val.min(axis=0)
    max_points = cluster_val.max(axis=0)
    # point_dim = max_points.diff(min_points)
    point_dim = np.subtract(max_points, min_points)
    x = int(min_points[0] - point_dim[0]/20)
    y = int(min_points[1] - point_dim[1]/20)
    w = int(point_dim[0] + point_dim[0]/10)
    h = int(point_dim[1] + point_dim[1]/10)
    
#     print cluster_name, cluster_val
#     print min_points, max_points
#     print x, y, w, h
    crop_img = img[y: y + h , x: x + w]
    cv2.imwrite(cluster_name, crop_img)


def generate_kp(file_list):
    if os.path.exists("updated_image/"):
        shutil.rmtree("updated_image/")
    extended = True
    hessian_threshold = 300
    mask = None
    # d_lim = 1000
    surf = cv2.xfeatures2d.SURF_create(
        hessian_threshold,
        upright=True,
        extended=extended
    )
    #orb_detector = cv2.ORB_create(nfeatures=1000)
    for image in file_list:
        print image
        img_name = image.rsplit("/", 1)[1]
        image_path = "updated_image/" + img_name + "/"
        org_name = image_path + img_name + "_org.jpg"
        grey_name = image_path + img_name + "_grey.jpg"
        kp_name = image_path + img_name + "_kp.jpg"
        kp_pdf = image_path + img_name + "_kp.pdf"
        # kp_csv = image_path + img_name + "_kp.csv"
        try:
            img = cv2.imread(image)
            if img is not None:
                if os.path.exists(image_path):
                    shutil.rmtree(image_path)
                os.makedirs(image_path)
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(grey_name, gray_image)
                gray_image = cv2.imread(grey_name)
                print gray_image.shape
                #key, desc = orb_detector.detectAndCompute(img, mask)
                key, desc = surf.detectAndCompute(gray_image, mask)
                print len(key)#, len(desc)
                # print key[0].angle, key[0].size, key[0].octave, key[0].pt
                img_arr = np.array([pnt.pt for pnt in key])
                #print img_arr
                x, y = [int(x) for x in key[0].pt]
                print x, y
                # h = w = int(key[0].size)
                kp_img = cv2.drawKeypoints(img, key, None, (0,0,255))
                cv2.imwrite(org_name, img)
                cv2.imwrite(kp_name, kp_img)
                # pd.DataFrame(img_arr).to_csv(
                #     kp_csv,
                #     float_format = '%.15f',
                #     index = False,
                #     encoding='utf-8'
                # )
                # return img_arr
                for cluster_key, cluster_val in get_sub_images(img_arr, kp_pdf).items():
                    cluster_name = image_path + img_name + "_" + cluster_key + ".jpg"
                    crop_image(img, cluster_val, cluster_name)
                    # cv2.imwrite("crop_ing.jpg", crop_img)
            else:
                print "unsupported type or blank image"
#             break
        except Exception as e:
            print e
            print traceback.print_exc()


file_list = glob.glob("images/*")[:30]
generate_kp(file_list)

