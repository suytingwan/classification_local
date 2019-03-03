import sys
import os

from sklearn.cluster import MiniBatchKMeans, KMeans
import time
import numpy as np
import re
from multiprocessing import Process

def cluster(tag_id, cluster_num, contsign_list, feature_list, id_list, filesave, class_):
    mb_kmeans = MiniBatchKMeans(init='k-means++', n_clusters=cluster_num,
                                batch_size=50000, n_init=10, compute_labels=True)
    dis_matrix = mb_kmeans.fit_transform(feature_list)
    labels = mb_kmeans.labels_
    inertias = mb_kmeans.inertia_
    
    dis_dic = {}
    cnt_dic = {}

    for index, dislst in enumerate(dis_matrix):
        y = labels[index]
        distance = dislst[y]
        if y not in dis_dic:
            dis_dic[y] = 0
            cnt_dic[y] = 0
        dis_dic[y] += distance
        cnt_dic[y] += 1

    for key in dis_dic:
        dis_dic[key] /= 1.0 * cnt_dic[key]

    fwrite = open(filesave, 'w')
    for x,y,z in zip(contsign_list, labels, id_list):
        if z == class_:
            fwrite.write("%s|%s|%s_%.5f\n" % (z, x, y, dis_dic[y]))
    fwrite.close()
'''
def lod(tag_id, neighbors_num, contsign_list, feature_list):
    clf = LocalOutlierFactor(n_neighbors=neighbors_num)
    y_pred = clf.fit_predict(feature_list)
    lof = -1 * clf.negative_outlier_factor_
    sys.stdout.write("%s\t" % (tag_id))
    for x, y in zip(contsign_list, lof):
        sys.stdout.write("%s|%.5f\t" %(x, y))
    sys.stdout.write("\n")
'''
def fea2id(filereads, filewrite, class_, sep='\t'):
    tag_id = None
    current_tag_id = None
    id_list = []
    contsign_list = []
    feature_list = []
    neighbors_num = 20
    cluster_num = 2
    min_image_num = 20

    for fileread in filereads:
        fread = open(fileread)
        for line in fread:
            arr = line.strip().split(sep)
            if len(arr) < 2:
                sys.stderr.write("line error: %s", " ".join(arr))
                continue
            current_tag_id = arr[0]
            if 'val' in arr[1]:
                pass
            else:
                id_list.append(arr[0])
                contsign_list.append(arr[1])
                feature_list.append(arr[2:])
                tag_id = current_tag_id
        fread.close()
    
    if len(contsign_list) > min_image_num:
        cluster(tag_id,cluster_num, contsign_list, feature_list, id_list, filewrite, class_)

    else:
        sys.stdout.write("%s\t" %(tag_id))
        for i, item in enumerate(contsign_list):
            sys.stdout.write("%s|%s\t" % (item, 0))
        sys.stdout.write("\n")

def fea2id_proc(names, start_id, end_id, dicts):
    path_in = 'split_rm_noise_feature'
    path_out = 'split_rm_noise_cluster'
    for i in range(start_id, end_id):
        filename = names[i]
        #correlate top 5
        class_ = filename.split('_')[2]
        class_ = class_.split('.')[0]
        
        #top 200
        extracts = [class_]
        filereads = []
        for extract in extracts:
            fileread = 'extract_class_{}.txt'.format(extract)
            filereads.append(os.path.join(path_in, fileread))

        filewrite = re.sub('extract','cluster',filename)
        fea2id(filereads, os.path.join(path_out, filewrite),class_)
        print('clustering class {} succeed'.format(filename))
 

def extract_fromtxt(filename):
    fread = open(filename)
    names = []
    for line in fread:
        names.append(line.strip().split(' ')[0])
    fread.close()
    return names

def extract_correlate(fileread):
    fread = open(fileread)
    dicts = {}
    for line in fread:
        info = line.strip().split(':')
        #print(info)
        dicts[info[0]] = info[1].split(' ')
    return dicts

if __name__ == '__main__':
    #path_in = 'extract_data/temp_100'
    path_in = 'split_all'

    #filenames = extract_fromtxt('tongji_result/yilou.txt')
    filenames = os.listdir(path_in)
    #path_out = 'extract_data/temp_100_cluster'
    path_out = 'split_cluster_all_result3'

    #filenames = os.listdir(path_in)
    #filternames = os.listdir(path_out)
    dicts = extract_correlate('label_correlated_matrix.txt')
    length = len(filenames)

    proc_num = int(sys.argv[1])
    proc_arr = []
    l = len(filenames)
    for i in range(proc_num):
        start_id = int(i * l / proc_num)
        end_id = int((i+1) * l / proc_num)
        p = Process(target=fea2id_proc, args=(filenames, start_id,end_id,dicts))
        p.start()
        proc_arr.append(p)

    for p in proc_arr:
        p.join()
   
