#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:25:31 2022
@author: dlr\hu_xk
"""
import haversine as hs
import re
import json
import csv
from sklearn.cluster import DBSCAN
import numpy as np
import argparse
import sympy
import geopy.distance
import matplotlib.pyplot as plt
from geopy.distance import geodesic as GD

from numpy import trapz

bool_debug = 0
bool_plot = 0
cannot_in_one_cluster = [] 
Total_systems = 20
pi = 3.14159265359

def calculate_median(l):
    l = sorted(l)
    l_len = len(l)
    if l_len < 1:
        return None
    if l_len % 2 == 0 :
        return ( l[int((l_len-1)/2)] + l[int((l_len+1)/2)] ) / 2.0
    else:
        return l[int((l_len-1)/2)]



# l = [1]
# print( calculate_median(l) )

# l = [3,1,2]
# print( calculate_median(l) )

# l = [1,2,3,4]
# print( calculate_median(l) )
# remove_list =  ['china','russia','canada', 'india','algeria','europe',  'asia','us', 'canadians', 'american','usa','america', \
#                 'africa', 'u.s.','u.s','west africa', 'united states','australia','chinese','russian', 'russians', 'middle east',\
#                     'american', 'nigeria', 'yemen',  'mongolia', 'cambodia', 'korea', 'romania', 'indian ocean', 'brazil', 'japan', 'japanese', 'hawaii', 'pacific', 'sudan', 'chile', 'canadians','canadian','americans', 'asia', 'european', 'united states of america','europe', 'north africa', 'north america', 'south america','western europe']
remove_list =  ['china','russia','canada', 'india','algeria','europe',  'asia','us', 'canadians', 'american','usa','america', \
                'africa', 'u.s.','u.s','west africa', 'united states','australia','chinese','russian', 'russians', 'middle east',\
                    'american',  'canadians','canadian','americans', 'asia', 'european', 'united states of america','europe', 'north africa', 'north america', 'south america','western europe']
max_error = 20039
def evaluate(detected, ground_truth, compared={},show_index=0):

    max_error_num = 0
    true_zero = 0
    if detected:
        zero_count = 0
        dis_errors = []
        T_P = 0
        F_N = 0
        F_P = 0
        for key in ground_truth.keys():
            # print('truth',ground_truth[key])
            if not key in detected:
                F_N += len(ground_truth[key])
                continue
            visited = []
            visited = 0
            cur_FN = len(ground_truth[key])
            # print('truth',ground_truth[key])
            # print('predict',detected[key])
            t_t_p = 0
            t_f_p = 0
            if key not in compared:
                continue
            # print(len(detected[key]))
            for i, place in enumerate(detected[key]):
                bool_matched = False
                bool_in_compare = False
                for c_place in compared[key]:
                    if is_equal(c_place, place) : #and not (c_place['lat'] == 0 and c_place['lon'] == 0):
                        bool_in_compare = True
                        break
                if not bool_in_compare:
                    continue
                for j in range(visited, len(ground_truth[key])):
                    true_place = ground_truth[key][j]         
                # for j, true_place in enumerate(ground_truth[key]):
                    # visited = j+1
                    if 1: #j not in visited:
                        # if place['end'] < true_place['start']:
                        #     break
                        if  is_equal(place, true_place):
                                
                                T_P += 1
                                t_t_p += 1
                                cur_FN -= 1
                                bool_matched = True
                                
                               # visited.append(j)
                                visited = j+1
                                if (place['lat']==0 and place['lon'] == 0):  #  ::
                                    # if show_index in  [3,5]:
                                    #     print(show_index, key,(place['LOC'], place['lat'],place['lon']), (true_place['lat'],true_place['lon']))

                                    zero_count += 1
                                    # continue
                                    dis = max_error
                                else:
                                   # dis = geopy.distance.distance((place['lat'],place['lon']), (true_place['lat'],true_place['lon'])).km
                                    #dis = GD((place['lat'],place['lon']), (true_place['lat'],true_place['lon'])).km
                                    dis = hs.haversine((place['lat'],place['lon']), \
                                                    (true_place['lat'],true_place['lon']))
                                    if dis > 161 and place['LOC'].lower() not in remove_list: #  :
                                       if not(true_place['lat']==0 and true_place['lon'] == 0):
                                            max_error_num += 1
                                            if bool_debug:

                                                if show_index in  [11,18]:
                                                    print(show_index, key,(place['LOC'], place['lat'],place['lon']), (true_place['lat'],true_place['lon']))
                                if not(true_place['lat']==0 and true_place['lon'] == 0) and not true_place['LOC'].lower() in remove_list:
                                    dis_errors.append(dis)
                                else:
                                    true_zero += 1
                                break
                if not bool_matched:
                    F_P += 1
                    t_f_p += 1
            F_N += cur_FN
            # if cur_FN:
            #     print(cur_FN)
            #     import pdb
            #     pdb.set_trace()

            # print(t_t_p,t_f_p,cur_FN)
        # import pdb
        # pdb.set_trace()
        try:
            ave_error = sum(dis_errors)/len(dis_errors)
            acu161count = 0
            for dis in dis_errors:
                if dis < 161:
                    acu161count += 1
            ACU161 = acu161count/len(dis_errors)
            median_error = calculate_median(dis_errors)
            P = T_P/(T_P+F_P) 
            R = T_P/(T_P+F_N) 
            F = (2*P*R) / (P+R)
            dis_errors.sort()
            dim_error = [(np.log(x+1)/np.log(max_error)) for x in dis_errors]
            y = np.array(dim_error)
            
            # Compute the area using the composite trapezoidal rule.
            area = trapz(y)/(len(dis_errors))

            return P,R,F, ave_error, median_error, ACU161, acu161count, len(dis_errors), zero_count, true_zero,area,dis_errors, max_error_num
        except:
            return 0,0,0, 20000, 1000, 0, 0, 0, 0.8, 0, 0.8,[], 0
    else: 
        return 0,0,0, 20000, 1000, 0, 0, 0, 0.8, 0, 0.8,[], 0
def is_overlapping(x1,x2,y1,y2):
    return max(x1,y1) <= min(x2,y2)

    
def is_equal(cur_place,target_place):
    if cur_place['start'] == target_place['start'] and cur_place['end'] == target_place['end'] \
    or (is_overlapping(cur_place['start'], cur_place['end'],\
                        target_place['start'],target_place['end']) \
          and re.sub('\W+','', cur_place['LOC']).lower() == re.sub('\W+','', target_place['LOC']).lower()):
        return True
    else:
        return False

# def is_equal_tuple(cur_place,target_place):
#     if cur_place[1] == target_place[1] and cur_place[2] == target_place[2] \
#     or (is_overlapping(cur_place[1], cur_place[2],\
#                         target_place[1],target_place[2]) \
#           and re.sub('\W+','', cur_place[0]).lower() == re.sub('\W+','', target_place[0]).lower()):
#         return True
#     else:
#         return False

    
    
# def is_equal_loos(cur_place,target_place):
#     if re.sub('\W+','', cur_place['LOC']).lower() == re.sub('\W+','', target_place['LOC']).lower():
#         return True
#     else:
#         return False
    
def max_cluster(valid_group,eps_km, min_samples):
    coordinates = []
    for place, di in valid_group:
       #  if di != 1 or 3 > len(valid_group):
        coordinates.append([place['lat'],place['lon']])
    # if len(valid_group)-1 > min_samples:
    #     min_samples = len(valid_group)-1
    clustering = DBSCAN(eps=eps_km/6371., min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coordinates))
    clusters = {}
    for i, item in enumerate(clustering.labels_):
        if item != -1:
            if item not in clusters:
                clusters[item]=[i]
            else:
                temp = clusters[item]
                temp.append(i)
                check =  all(index in temp for index in cannot_in_one_cluster)
                if not check:
                    clusters[item].append(i)
    max_item = min_samples
    
    
    max_i = -1
    if clusters:
        for key in clusters.keys():
            if len(clusters[key]) >= max_item:
                max_item = len(clusters[key])
                max_i = key
    return clusters, max_i

def dbscan_latlong(valid_group,eps_km=200, min_samples=2, eps_2=10, weights = [1,1,1,1,1,1], bool_weight=0):
    clusters, max_i = max_cluster(valid_group,eps_km, min_samples)
    # coordinates = []
    # for place, di in valid_group:
    #    #  if di != 1 or 3 > len(valid_group):
    #     coordinates.append([place['lat'],place['lon']])
    # # if len(valid_group)-1 > min_samples:
    # #     min_samples = len(valid_group)-1
    # clustering = DBSCAN(eps=eps_km/6371., min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coordinates))
    # clusters = {}
    # for i, item in enumerate(clustering.labels_):
    #     if item != -1:
    #         if item not in clusters:
    #             clusters[item]=[i]
    #         else:
    #             temp = clusters[item]
    #             temp.append(i)
    #             check =  all(index in temp for index in cannot_in_one_cluster)
    #             if not check:
    #                 clusters[item].append(i)
    # max_item = min_samples
    
    
    # max_i = -1
    # if clusters:
    #     for key in clusters.keys():
    #         if len(clusters[key]) >= max_item:
    #             max_item = len(clusters[key])
    #             max_i = key
    
    # max_i = []
    # if clusters:
    #     for key in clusters.keys():
    #         if len(clusters[key]) > max_item:
    #             max_item = len(clusters[key])
    #             max_i = [key]
    #         elif len(clusters[key]) == max_item:
    #             max_i.append(key)
                 
    # if len(max_i) > 1:
    #     # import pdb
    #     # pdb.set_trace()
    #     print('make choice')
    #     min_index = 0
    #     min_distance = 100000000000000
    #     for index in max_i:
    #         total = 0
    #         for i, place_i in enumerate(clusters[index]):
    #             placei_lat = valid_group[place_i][0]['lat']
    #             placei_lon = valid_group[place_i][0]['lon']
    #             for j in range(i+1,len(clusters[index])):
    #                 place_j = clusters[index][j]
    #                 placej_lat = valid_group[place_j][0]['lat']
    #                 placej_lon = valid_group[place_j][0]['lon']
    #                 total +=  hs.haversine((placei_lat,placei_lon), \
    #                                                (placej_lat,placej_lon))
    #         if min_distance > total:
    #             min_distance = total
    #             min_index = index
                    
    #     max_i = min_index
    # elif max_i:
    #     max_i = max_i[0]
    # else:
    #     max_i = -1
        
        
    if max_i != -1:
        max_j = -1
        new_places = []
        for place_i  in clusters[max_i]:
            new_places.append(valid_group[place_i])
        new_clusters, max_j = max_cluster(new_places,eps_2, min_samples)
        if max_j != -1:
            clusters = new_clusters
            max_i = max_j
            valid_group = new_places
        sum_lat = 0
        # import pdb
        # pdb.set_trace()
        sum_lon = 0
        weight_sum = 0
        if bool_weight == 0:
            weights = [1]*Total_systems
        for place_i  in clusters[max_i]:
           #  if valid_group[place_i][1] == 5:
          #       continue
            weight_sum += weights[valid_group[place_i][1]]

        for place_i in clusters[max_i]:
            # if valid_group[place_i][1] == 5:
          #       continue
            place = valid_group[place_i][0]
            sum_lat += (weights[valid_group[place_i][1]]/weight_sum)*place['lat']
            sum_lon += (weights[valid_group[place_i][1]]/weight_sum)*place['lon']
        # ave_lat = sum_lat /len(clusters[max_i])
        # ave_lon = sum_lon /len(clusters[max_i])
        

        new_place = {}
        new_place['lat'] = sum_lat
        new_place['lon'] = sum_lon
        return new_place
    only_place = {'lat':0, 'lon':0}
    # return  only_place  #valid_group[-1][0]

    for place in valid_group:
        if not (place[0]['lat'] == 0 and place[0]['lon'] == 0):
              only_place = place[0]
        if place[1] == 0 and not (place[0]['lat'] == 0 and place[0]['lon'] == 0):
              # print('clavin', place)
              return place[0]
        
    return  only_place  #valid_group[-1][0]
    # if valid_group[place_i][1] == 4:
    # return valid_group[-1][0] {'lat':0, 'lon':0}

def weighted(valid_group, weights):
    # import pdb
    # pdb.set_trace()
    sum_lat = 0
    sum_lon = 0
    weight_sum = 0
    for place, di  in valid_group:
        weight_sum += weights[di]
        
    for place, di in valid_group:
        sum_lat += (weights[di]/weight_sum) * place['lat']
        sum_lon += (weights[di]/weight_sum) * place['lon']
    # ave_lat = sum_lat /len(clusters[max_i])
    # ave_lon = sum_lon /len(clusters[max_i])
    new_place = {}
    new_place['lat'] = sum_lat
    new_place['lon'] = sum_lon
    return new_place
    
    
def voting(group,eps_km=800, min_samples=4, eps_2=10, weights = [1,1,1,1,1,1], bool_weight=0):
    valid_group = []
    for place in group:
        if not(place[0]['lat'] == 0 and place[0]['lon'] == 0):
            valid_group.append(place)
    if valid_group:
        if bool_weight==1:
            new_place = weighted(valid_group, weights)
        else:
            new_place = dbscan_latlong(valid_group,eps_km, min_samples, eps_2, weights, bool_weight)
        new_place['start'] = valid_group[0][0]['start']
        new_place['end'] = valid_group[0][0]['end']
        new_place['LOC'] = valid_group[0][0]['LOC']

        # sum_lat = 0
        # sum_lon = 0
        # for place in valid_group:
        #     sum_lat += place['lat']
        #     sum_lon += place['lon']
        # ave_lat = sum_lat /len(valid_group)
        # ave_lon = sum_lon /len(valid_group)
        # new_place = valid_group[-1]
        # new_place['lat'] = ave_lat
        # new_place['lon'] = ave_lon
        return new_place
    else:
        return group[-1][0]

def maxDisjointIntervals(list_):
    new_places = []
    # Lambda function to sort the list 
    # elements by second element of pairs
    list_.sort(key = lambda x: x['end'])
     
    # First interval will always be
    # included in set
    # print("[", list_[0][0], ", ", list_[0][1], "]")
    new_places.append(list_[0])
    # End point of first interval
    r1 = list_[0]['end']
     
    for i in range(1, len(list_)):
        l1 = list_[i]['start']
        r2 = list_[i]['end']
         
        # Check if given interval overlap with
        # previously included interval, if not
        # then include this interval and update
        # the end point of last added interval
        if l1 > r1:
            new_places.append(list_[i])
            # print("[", l1, ", ", r2, "]")
            r1 = r2
    return  new_places
 
def merge(datasets, keys, eps_km=800, min_samples=4, eps_2=10, weights = [1,1,1,1,1,1], \
          bool_weight=0, must_contain_da = -1,true_dictionary={},base={}):
    new_dataset = {}
    count = 0
    for key in keys:
        added_places_index = []
        count+=1
        groups = []
        if bool_debug:
            print('*'*50)
            print(count,key)
            if key in true_dictionary:
                print('true_data=', true_dictionary[key])
        # import pdb
        # pdb.set_trace()
        # if str(key) == '536248966007230464':
        #     import pdb
        #     pdb.set_trace()
        
        for i, dataset in enumerate(datasets):
            if bool_debug:

                if key in dataset:
                    newlist = sorted(dataset[key], key=lambda d: d['start']) 
                    print('data'+str(i)+'=',newlist)
                else:
                    print('data'+str(i)+'=','[{}]')
            if key not in dataset:
                cur_detection = []
            else:
                cur_detection = dataset[key]
            visisted = {}
            for p, cur_place in enumerate(cur_detection):
                group = []
                if (i,p) not in added_places_index:
                    group.append((cur_place,i))
                    #datasets[i][key].pop(p)
                    #print(i, len(datasets[i][key]))
                    for j in range(i+1,len(datasets)):
                        if key in datasets[j]:
                            target = datasets[j][key]
                            if j not in visisted:
                                start_search = 0
                            else:
                                start_search = visisted[j]+1
                            for k in range(start_search, len(target)):
                                target_place = target[k]
                            #for k, target_place in enumerate(target):
                                if cur_place['start'] > target_place['end']:
                                    continue
                                if cur_place['end'] < target_place['start']:
                                    break
                                if is_equal(cur_place,target_place):
                                    visisted[j]=k
                                    group.append((target_place,j))
                                    #datasets[j][key].pop(k)
                                    #print(j, len(datasets[j][key]))
                                    added_places_index.append((j,k))
                                    break
                    # some approaches have more than one votes            
                    for pp in group:
                        if pp[1] in [0,1]:
                            new_place = pp
                            for n in range(0):
                                group.append(new_place)
                            break
                    groups.append(group)
        # print('group found')
        new_places = []
        single_places = []
        # if str(key) == '11854_Poole1860':
        #     import pdb
        #     pdb.set_trace()

        for group in groups:
            bool_check = 0
            for temp_place in group:
                if must_contain_da >= 0:
                    if temp_place[1] == must_contain_da:
                        bool_check = 1
                        break
                else:
                    bool_check = 1
                    break
            if bool_check:
                if len(group) >= 2:
                    place = voting(group,eps_km, min_samples, eps_2, weights, bool_weight)
                  #  if  not(place['lat'] == 0 and place['lon'] == 0):
                    new_places.append(place)
                else:
                    if len(group) != 0:
                        # for place in group:
                            # if place[1]==0:
                        place = group[0][0]
                if not(place['lat'] == 0 and place['lon'] == 0):
                    
                    new_places.append(place)
                else:
                    bool_add = 0
                    if key in base:
                        for base_place in base[key]:
                            if is_equal(place,base_place):
                                 new_places.append(base_place)
                                 bool_add = 1
                                 break
                    if not bool_add:
                        new_places.append(place)
                                # break
                    # single_places.append(group[-1][0])
        # if single_places:
        #     new_places.extend(maxDisjointIntervals(single_places))
        new_dataset[key] = new_places
        newlist = sorted(new_dataset[key], key=lambda d: d['start']) 
        if bool_debug:
            print('new_dataset', newlist)
        
    return new_dataset
 #        for detection in dicts:
 #            for first_place in first_dict:
 #                for second_place in detection:
 #                    if first_place['LOC'] == second_place['LOC']:
 #                        elif fullstring.find(substring) != -1:
 # first_place['LOC'] == second_place['LOC']:
 #        initial_detection = dicts[0]
 #        for i in range(1,len(dicts)):
            
def split_true(data, max_count, ID_expansion):
    new_data = {}
    split_dicts = {}
    for key in data:
        if len (data[key]) > max_count:
            splits = []
            last_pos = 0
            new_places = []
            for i, place in enumerate(data[key]):
                if i < max_count + last_pos*max_count:
                    new_places.append(place)
                else:
                    new_data[key+str(len(splits))*ID_expansion] = new_places
                    splits.append(new_places[-1]['end'])
                    new_places = [place]
                    last_pos +=1
            if new_places:
                new_data[key+str(len(splits))*ID_expansion] = new_places
                new_places = []
            split_dicts[key]=splits
        else:
            new_data[key] = data[key]
            split_dicts[key]=[]
        # if len (data[key])  > 1000:
        #     import pdb
        #     pdb.set_trace()
    
    return new_data, split_dicts

def split_estimation(data, split_dicts, ID_expansion):
    new_data = {}
    # split_dicts = {}
    for key in data:
        if key in split_dicts:
            if len (split_dicts[key]):
                splits = split_dicts[key]
                last_pos = 0
                new_places = []
                for i, place in enumerate(data[key]):
                    if len(splits) == last_pos:
                        new_places.append(place)
                    else:
                        if place['end'] <= splits[last_pos]:
                            new_places.append(place)
                        else:
                            new_data[key+str(last_pos)*ID_expansion] = new_places
                            new_places = [place]
                            last_pos +=1
                if new_places:
                    new_data[key+str(last_pos)*ID_expansion] = new_places
                    new_places = []
            else:
                new_data[key] = data[key]
    return new_data
    
def ensamble_systems(systems, votes):
    new_systems = []
    for i, sys in enumerate(systems):
        new_systems.extend([sys]*votes[i])
    return new_systems

def load_estimation(file,split_dicts, ID_expansion):
    try:
        io = open(file,"r")
        cam_esti_dictionary = json.load(io)
        sorted_set = {}
        for key in cam_esti_dictionary:
            sorted_set[key] = sorted(cam_esti_dictionary[key], key=lambda d: d['start']) 
        cam_esti_dictionary = sorted_set
        # print(len(cam_esti_dictionary))
        cam_esti_dictionary = split_estimation(cam_esti_dictionary, split_dicts, ID_expansion)
        # print('split',len(cam_esti_dictionary))
    except:
        cam_esti_dictionary = {}
    return cam_esti_dictionary


def importance(datasets, eps_km, min_samples, eps_2=10, bool_weight =0, must_contain_da = -1 ):
    return_result = []
    MES = []
    ACU161s = []
    AUCs = []
    ID_expansion = 5
    max_place = 30
    total_161 = []
    total_me = []
    total_auc = []
    for data in datasets:
        io = open('../data/'+data+'.json',"r")
        true_dictionary = json.load(io)
        sorted_set = {}
        for key in true_dictionary:
            sorted_set[key] = sorted(true_dictionary[key], key=lambda d: d['start']) 
        true_dictionary = sorted_set
        print(len(true_dictionary))
        true_dictionary, split_dicts = split_true(true_dictionary, max_place, ID_expansion)
        print('split',len(true_dictionary))
        cam_esti_dictionary = load_estimation('../data/camcode_'+data+'.json',split_dicts, ID_expansion) 
        clavin_esti_dictionary = load_estimation('../data/clavin_'+data+'.json',split_dicts, ID_expansion)
        stan_esti_dictionary = load_estimation('../data/true_normi_'+data+'.json',split_dicts, ID_expansion)
        ein_esti_dictionary = load_estimation('../data/edinburgh_'+data+'.json',split_dicts, ID_expansion)
        dbp_esti_dictionary = load_estimation('../data/dbpedia_'+data+'.json',split_dicts, ID_expansion)
        geopop_esti_dictionary =  load_estimation('../data/geonamespop_'+data+'.json',split_dicts, ID_expansion)
        mordecai_esti_dictionary =  load_estimation('../data/mordecai_'+data+'.json',split_dicts, ID_expansion)
        topocluster_esti_dictionary =  load_estimation('../data/topocluster_'+data+'.json',split_dicts, ID_expansion)
        CBH_esti_dictionary = load_estimation('../data/CBH_'+data+'.json',split_dicts, ID_expansion)
        SHS_esti_dictionary = load_estimation('../data/SHS_'+data+'.json',split_dicts, ID_expansion)
        CHF_esti_dictionary = load_estimation('../data/CHF_'+data+'.json',split_dicts, ID_expansion)
        adapter_esti_dictionary = load_estimation('../data/adapter_'+data+'.json',split_dicts, ID_expansion)
        fishing_esti_dictionary = load_estimation('../data/fishing_'+data+'.json',split_dicts, ID_expansion)
        blink_esti_dictionary = load_estimation('../data/blink_'+data+'.json',split_dicts, ID_expansion)
        extend_esti_dictionary = load_estimation('../data/extend_'+data+'.json',split_dicts, ID_expansion)
        rel_esti_dictionary =  load_estimation('../data/rel_'+data+'.json',split_dicts, ID_expansion)
        genre_esti_dictionary =  load_estimation('../data/genre_'+data+'.json',split_dicts, ID_expansion)
        dca_esti_dictionary =  load_estimation('../data/dca_'+data+'.json',split_dicts, ID_expansion)
        luke_esti_dictionary =  load_estimation('../data/luke_'+data+'.json',split_dicts, ID_expansion)
        bootleg_esti_dictionary =  load_estimation('../data/bootleg_'+data+'.json',split_dicts, ID_expansion)
        all_systems = [fishing_esti_dictionary, dca_esti_dictionary,  rel_esti_dictionary, blink_esti_dictionary, bootleg_esti_dictionary,  \
                        genre_esti_dictionary, extend_esti_dictionary, luke_esti_dictionary, stan_esti_dictionary, adapter_esti_dictionary, \
                            geopop_esti_dictionary,clavin_esti_dictionary,topocluster_esti_dictionary, mordecai_esti_dictionary, \
                          CBH_esti_dictionary, SHS_esti_dictionary,CHF_esti_dictionary, cam_esti_dictionary,dbp_esti_dictionary,ein_esti_dictionary]
        Total_systems = len(all_systems)
        weights = [1]*Total_systems
            
        new_places = merge(all_systems, true_dictionary.keys(), eps_km, min_samples, eps_2, weights, \
                            bool_weight,must_contain_da,true_dictionary)
        P,R,F, ave_error, median_error, ACU161, acu161count, dis_errors, zero_count, \
            true_zero, area , dis_list, max_error_num=\
                evaluate(new_places, true_dictionary, true_dictionary) #
        q161results = []
        meresults = []
        aucresults = []

        for i, system in enumerate(all_systems):
            new_systems=[all_systems[j] for j in range(len(all_systems)) if j != i ]
            new_places1 = merge(new_systems, true_dictionary.keys(), eps_km, min_samples, eps_2, weights, \
                            bool_weight,must_contain_da,true_dictionary)
            P,R,F, ave_error1, median_error, ACU1611, acu161count, dis_errors, zero_count, \
                true_zero, area1 , dis_list, max_error_num=\
                    evaluate(new_places1, true_dictionary, true_dictionary) #
            q161results.append(ACU161-ACU1611)
            meresults.append(ave_error-ave_error1)
            aucresults.append(area-area1)
        total_161.append(q161results)
        total_me.append(meresults)
        total_auc.append(aucresults)
    return np.mean(total_161, axis=0),np.mean(total_me, axis=0),np.mean(total_auc, axis=0)

def senstive_analysis(datasets, eps_kms, min_samples, eps_2s, bool_weight =0, must_contain_da = -1 ):
    return_result = []
    MES = []
    ACU161s = []
    AUCs = []
    ID_expansion = 5
    max_place = 30
    total_161 = []
    total_me = []
    total_auc = []
    for eps in eps_kms:
        for min_sample in min_samples:
            for eps_2 in eps_2s:
                q161results = []
                meresults = []
                aucresults = []
            
                for data in datasets:
                    io = open('../data/'+data+'.json',"r")
                    true_dictionary = json.load(io)
                    sorted_set = {}
                    for key in true_dictionary:
                        sorted_set[key] = sorted(true_dictionary[key], key=lambda d: d['start']) 
                    true_dictionary = sorted_set
                    print(len(true_dictionary))
                    true_dictionary, split_dicts = split_true(true_dictionary, max_place, ID_expansion)
                    print('split',len(true_dictionary))
                    cam_esti_dictionary = load_estimation('../data/camcode_'+data+'.json',split_dicts, ID_expansion) 
                    clavin_esti_dictionary = load_estimation('../data/clavin_'+data+'.json',split_dicts, ID_expansion)
                    stan_esti_dictionary = load_estimation('../data/true_normi_'+data+'.json',split_dicts, ID_expansion)
                    ein_esti_dictionary = load_estimation('../data/edinburgh_'+data+'.json',split_dicts, ID_expansion)
                    dbp_esti_dictionary = load_estimation('../data/dbpedia_'+data+'.json',split_dicts, ID_expansion)
                    geopop_esti_dictionary =  load_estimation('../data/geonamespop_'+data+'.json',split_dicts, ID_expansion)
                    mordecai_esti_dictionary =  load_estimation('../data/mordecai_'+data+'.json',split_dicts, ID_expansion)
                    topocluster_esti_dictionary =  load_estimation('../data/topocluster_'+data+'.json',split_dicts, ID_expansion)
                    CBH_esti_dictionary = load_estimation('../data/CBH_'+data+'.json',split_dicts, ID_expansion)
                    SHS_esti_dictionary = load_estimation('../data/SHS_'+data+'.json',split_dicts, ID_expansion)
                    CHF_esti_dictionary = load_estimation('../data/CHF_'+data+'.json',split_dicts, ID_expansion)
                    adapter_esti_dictionary = load_estimation('../data/adapter_'+data+'.json',split_dicts, ID_expansion)
                    fishing_esti_dictionary = load_estimation('../data/fishing_'+data+'.json',split_dicts, ID_expansion)
                    blink_esti_dictionary = load_estimation('../data/blink_'+data+'.json',split_dicts, ID_expansion)
                    extend_esti_dictionary = load_estimation('../data/extend_'+data+'.json',split_dicts, ID_expansion)
                    rel_esti_dictionary =  load_estimation('../data/rel_'+data+'.json',split_dicts, ID_expansion)
                    genre_esti_dictionary =  load_estimation('../data/genre_'+data+'.json',split_dicts, ID_expansion)
                    dca_esti_dictionary =  load_estimation('../data/dca_'+data+'.json',split_dicts, ID_expansion)
                    luke_esti_dictionary =  load_estimation('../data/luke_'+data+'.json',split_dicts, ID_expansion)
                    bootleg_esti_dictionary =  load_estimation('../data/bootleg_'+data+'.json',split_dicts, ID_expansion)
                    all_systems = [fishing_esti_dictionary, dca_esti_dictionary,  rel_esti_dictionary, blink_esti_dictionary, bootleg_esti_dictionary,  \
                                    genre_esti_dictionary, extend_esti_dictionary, luke_esti_dictionary, stan_esti_dictionary, adapter_esti_dictionary, \
                                        geopop_esti_dictionary,clavin_esti_dictionary,topocluster_esti_dictionary, mordecai_esti_dictionary, \
                                      CBH_esti_dictionary, SHS_esti_dictionary,CHF_esti_dictionary, cam_esti_dictionary,dbp_esti_dictionary,ein_esti_dictionary]
                    Total_systems = len(all_systems)
                    weights = [1]*Total_systems
                    systems = [genre_esti_dictionary, genre_esti_dictionary, genre_esti_dictionary,blink_esti_dictionary,blink_esti_dictionary,  luke_esti_dictionary, luke_esti_dictionary,  \
                                  \
                                    \
                                    cam_esti_dictionary,  ein_esti_dictionary, CBH_esti_dictionary, SHS_esti_dictionary]
                    new_places = merge(systems, true_dictionary.keys(), eps, min_sample, eps_2, weights, \
                                        bool_weight,must_contain_da,true_dictionary)
                    P,R,F, ave_error, median_error, ACU161, acu161count, dis_errors, zero_count, \
                        true_zero, area , dis_list, max_error_num=\
                            evaluate(new_places, true_dictionary, true_dictionary) #
                    q161results.append(ACU161)
                    meresults.append(ave_error)
                    aucresults.append(area)    
                total_161.append(sum(q161results)/len(q161results))
                total_me.append(sum(meresults)/len(meresults))
                total_auc.append(sum(aucresults)/len(aucresults))
                print('cur_res', total_161[-1], total_auc[-1], total_me[-1])
    return total_161, total_me, total_auc


def evaluate1(eps_km=800, min_samples=4, eps_2=10, data='geocorpora',weights = [1,1,1,1,1,1], bool_weight=0,must_contain_da=-1, write=0):
    return_result = []
    MES = []
    ACU161s = []
    AUCs = []
    stan_esti_dictionary = {}
    ein_esti_dictionary = {}
    ID_expansion = 5
    max_place = 30
    io = open('../data/'+data+'.json',"r")
    true_dictionary = json.load(io)
    sorted_set = {}
    for key in true_dictionary:
        sorted_set[key] = sorted(true_dictionary[key], key=lambda d: d['start']) 
    true_dictionary = sorted_set
    # print(len(true_dictionary))
    true_dictionary, split_dicts = split_true(true_dictionary, max_place, ID_expansion)
    # print('split',len(true_dictionary))
    cam_esti_dictionary = load_estimation('../data/camcode_'+data+'.json',split_dicts, ID_expansion) 
    clavin_esti_dictionary = load_estimation('../data/clavin_'+data+'.json',split_dicts, ID_expansion)
    stan_esti_dictionary = load_estimation('../data/true_normi_'+data+'.json',split_dicts, ID_expansion)
    ein_esti_dictionary = load_estimation('../data/edinburgh_'+data+'.json',split_dicts, ID_expansion)
    dbp_esti_dictionary = load_estimation('../data/dbpedia_'+data+'.json',split_dicts, ID_expansion)
    geopop_esti_dictionary =  load_estimation('../data/geonamespop_'+data+'.json',split_dicts, ID_expansion)
    mordecai_esti_dictionary =  load_estimation('../data/mordecai_'+data+'.json',split_dicts, ID_expansion)
    topocluster_esti_dictionary =  load_estimation('../data/topocluster_'+data+'.json',split_dicts, ID_expansion)
    CBH_esti_dictionary = load_estimation('../data/CBH_'+data+'.json',split_dicts, ID_expansion)
    SHS_esti_dictionary = load_estimation('../data/SHS_'+data+'.json',split_dicts, ID_expansion)
    CHF_esti_dictionary = load_estimation('../data/CHF_'+data+'.json',split_dicts, ID_expansion)
    adapter_esti_dictionary = load_estimation('../data/adapter_'+data+'.json',split_dicts, ID_expansion)
    fishing_esti_dictionary = load_estimation('../data/fishing_'+data+'.json',split_dicts, ID_expansion)
    blink_esti_dictionary = load_estimation('../data/blink_'+data+'.json',split_dicts, ID_expansion)
    extend_esti_dictionary = load_estimation('../data/extend_'+data+'.json',split_dicts, ID_expansion)
    rel_esti_dictionary =  load_estimation('../data/rel_'+data+'.json',split_dicts, ID_expansion)
    genre_esti_dictionary =  load_estimation('../data/genre_'+data+'.json',split_dicts, ID_expansion)
    dca_esti_dictionary =  load_estimation('../data/dca_'+data+'.json',split_dicts, ID_expansion)
    luke_esti_dictionary =  load_estimation('../data/luke_'+data+'.json',split_dicts, ID_expansion)
    bootleg_esti_dictionary =  load_estimation('../data/bootleg_'+data+'.json',split_dicts, ID_expansion)
    # import pdb
    # pdb.set_trace()
    ACU161s_subsets = []
    MES_subsets = []
    AUC_subsets = []

    # /home/hu_xk/Workplace/DLR-disasters/Edinburgh/edinburgh_geocorpora.json , ein_esti_dictionary , ein_esti_dictionary ,ein_esti_dictionary,  dbp_esti_dictionary ,  stan_esti_dictionary CBH_esti_dictionary dbp_esti_dictionary, topocluster_esti_dictionary, ein_esti_dictionary, mordecai_esti_dictionary, stan_esti_dictionary, babelfy_esti_dictionary, 
    voting_count = 2
    voting_aves = []
    voting_161s = []
    voting_areas = []
    total_dis_list = []
    systems = []
    systems1 =  []
    systems2 = []
    systems3 = []
    systems4 = []
    systems5 = []
    systems6 = []
    systems7 = []
    systems8 = []

    systems =   ensamble_systems( [genre_esti_dictionary, blink_esti_dictionary,  luke_esti_dictionary, \
                  \
                    \
                    cam_esti_dictionary, ein_esti_dictionary,CBH_esti_dictionary, SHS_esti_dictionary], [3,2,2,1,1,1,1,1,1])       
        
    Total_systems =20
    if len(weights) < Total_systems:
        weights = [1]*Total_systems
    # P,R,F, ave_error, median_error, ACU161, acu161count,dis_errors, zero_count, true_zero, area, dis_list = evaluate(new_places, true_dictionary,true_dictionary) #
    # MES.append(int(ave_error))
    # ACU161s.append(round(ACU161, 2))
    # voting1_ave = ave_error
    # voting1_161 = ACU161
    # print('voting1', P,R,F, ave_error, median_error, ACU161, acu161count,area) wat_esti_dictionary, 
    # total_dis_list.append(dis_list) , SHS_esti_dictionary, stan_esti_dictionary,  CBH_esti_dictionary dbp_esti_dictionary,  CHF_esti_dictionary wat_esti_dictionary,tagme_esti_dictionary, 
    # systems1 = [genre_esti_dictionary, genre_esti_dictionary, genre_esti_dictionary,blink_esti_dictionary,blink_esti_dictionary,  luke_esti_dictionary, \
    #               \
    #                 \
    #                cam_esti_dictionary, ein_esti_dictionary,CBH_esti_dictionary, SHS_esti_dictionary,adapter_esti_dictionary]
        
    # systems2 = [genre_esti_dictionary, genre_esti_dictionary, genre_esti_dictionary,blink_esti_dictionary, extend_esti_dictionary, luke_esti_dictionary, \
    #               \
    #                 \
    #                cam_esti_dictionary, ein_esti_dictionary, CBH_esti_dictionary, SHS_esti_dictionary,adapter_esti_dictionary,clavin_esti_dictionary]
     
    # systems3 = [genre_esti_dictionary, genre_esti_dictionary, genre_esti_dictionary,blink_esti_dictionary,blink_esti_dictionary, luke_esti_dictionary, luke_esti_dictionary, \
    #               \
    #                 \
    #                 bootleg_esti_dictionary, cam_esti_dictionary, ein_esti_dictionary, SHS_esti_dictionary, CBH_esti_dictionary]
     
    # systems4 = [genre_esti_dictionary, genre_esti_dictionary, genre_esti_dictionary,blink_esti_dictionary,blink_esti_dictionary,  luke_esti_dictionary,  luke_esti_dictionary,\
    #               \
    #                 \
    #                 cam_esti_dictionary, ein_esti_dictionary, SHS_esti_dictionary, CBH_esti_dictionary]
     
    # systems4 = [genre_esti_dictionary, blink_esti_dictionary,extend_esti_dictionary, luke_esti_dictionary, \
    #               \
    #                 \
    #                 cam_esti_dictionary, ein_esti_dictionary,CBH_esti_dictionary, SHS_esti_dictionary]
    # vots = 
    # new_systems = []
    # for i, sys in enumerate(systems4):
    #     new_systems.extend([sys]*vots[i])
    # # systems4 = new_systems
    # systems2 = ensamble_systems( [genre_esti_dictionary, blink_esti_dictionary,  luke_esti_dictionary, \
    #               \
    #                 \
    #                 cam_esti_dictionary, ein_esti_dictionary,  SHS_esti_dictionary,adapter_esti_dictionary ], [3,2,2,1,1,1,1])       

    # systems3 = ensamble_systems( [genre_esti_dictionary, blink_esti_dictionary,  luke_esti_dictionary, bootleg_esti_dictionary,\
    #               \
    #                 \
    #                 cam_esti_dictionary, ein_esti_dictionary,  SHS_esti_dictionary], [3,2,2,1,1,1,1]) 
    
    # systems4 = ensamble_systems( [genre_esti_dictionary, blink_esti_dictionary,  luke_esti_dictionary, bootleg_esti_dictionary,\
    #               \
    #                 \
    #                 cam_esti_dictionary, ein_esti_dictionary,  CBH_esti_dictionary], [3,2,2,1,1,1,1])       
        
    #systems5 = ensamble_systems( [genre_esti_dictionary, blink_esti_dictionary,  luke_esti_dictionary, \
    #              \
    #                \
    #                cam_esti_dictionary, adapter_esti_dictionary, ein_esti_dictionary, CBH_esti_dictionary,  SHS_esti_dictionary], [3,2,2,1,1,1,1,1,1])   
        
    #systems6 = [genre_esti_dictionary, genre_esti_dictionary, genre_esti_dictionary,blink_esti_dictionary,blink_esti_dictionary,  luke_esti_dictionary, luke_esti_dictionary, bootleg_esti_dictionary,  \
    #              \
    #                \
    #                cam_esti_dictionary, CBH_esti_dictionary, adapter_esti_dictionary, ein_esti_dictionary,  SHS_esti_dictionary]    
    
    # systems7 = ensamble_systems( [genre_esti_dictionary, blink_esti_dictionary,  luke_esti_dictionary, \
    #               \
    #                 \
    #                 cam_esti_dictionary, ein_esti_dictionary,CBH_esti_dictionary, SHS_esti_dictionary], [3,2,2,1,1,1,1])       
        
    # systems8 = ensamble_systems( [genre_esti_dictionary, blink_esti_dictionary,  luke_esti_dictionary, \
    #               \
    #                 \
    #                 cam_esti_dictionary, ein_esti_dictionary,CBH_esti_dictionary, SHS_esti_dictionary], [3,2,2,2,1,1,1])       

        
    # systems5 = [genre_esti_dictionary, blink_esti_dictionary, extend_esti_dictionary, luke_esti_dictionary, \
    #               \
    #                 \
    #                 cam_esti_dictionary, ein_esti_dictionary,CBH_esti_dictionary, SHS_esti_dictionary]
    # vots = [5,3,2,3,2,2,1,1]
    # new_systems = []
    # for i, sys in enumerate(systems5):
    #     new_systems.extend([sys]*vots[i])
    # systems5 = new_systems
        
    # systems6 = [genre_esti_dictionary, genre_esti_dictionary, genre_esti_dictionary,blink_esti_dictionary,blink_esti_dictionary,  luke_esti_dictionary, luke_esti_dictionary, \
    #               \
    #                 \
    #                 cam_esti_dictionary, ein_esti_dictionary, SHS_esti_dictionary, CBH_esti_dictionary]
       
    new_places = {}
    new_places = merge(systems, true_dictionary.keys(), eps_km, min_samples,eps_2, weights, \
  bool_weight,must_contain_da,true_dictionary)
    #     
        
    # Total_systems = len(systems1)
        # systems6 = [genre_esti_dictionary, genre_esti_dictionary, genre_esti_dictionary,blink_esti_dictionary,blink_esti_dictionary, extend_esti_dictionary, luke_esti_dictionary, luke_esti_dictionary, \
        #           \
        #             \
        #            cam_esti_dictionary, ein_esti_dictionary, SHS_esti_dictionary, CBH_esti_dictionary]

    # systems6 = [genre_esti_dictionary, genre_esti_dictionary, genre_esti_dictionary,blink_esti_dictionary,blink_esti_dictionary, extend_esti_dictionary, luke_esti_dictionary, luke_esti_dictionary, \
    #               \
    #                 \
    #                cam_esti_dictionary, ein_esti_dictionary, SHS_esti_dictionary, CBH_esti_dictionary, adapter_esti_dictionary]
    

    Total_systems = len(systems1)
    if len(weights) < Total_systems:
        weights = [1]*Total_systems
    # new_places1 = {}
     
    new_places1 = merge(systems1, true_dictionary.keys(), eps_km, min_samples, eps_2,weights, bool_weight,must_contain_da,true_dictionary)
    new_places2 = merge(systems2, true_dictionary.keys(), eps_km, min_samples, eps_2,weights, bool_weight,must_contain_da,true_dictionary)
    new_places3 = merge(systems3, true_dictionary.keys(), eps_km, min_samples, eps_2,weights, bool_weight,must_contain_da,true_dictionary)
    new_places4 = merge(systems4, true_dictionary.keys(), eps_km, min_samples, eps_2,weights, bool_weight,must_contain_da,true_dictionary)
    new_places5= merge(systems5, true_dictionary.keys(), eps_km, min_samples, eps_2,weights, bool_weight,must_contain_da,true_dictionary)
    new_places6= merge(systems6, true_dictionary.keys(), eps_km, min_samples, eps_2,weights, bool_weight,must_contain_da,true_dictionary)
    new_places7= merge(systems7, true_dictionary.keys(), eps_km, min_samples, eps_2,weights, bool_weight,must_contain_da,true_dictionary)
    new_places8= merge(systems8, true_dictionary.keys(), eps_km, min_samples, eps_2,weights, bool_weight,must_contain_da,true_dictionary)

    # P,R,F, ave_error, median_error, ACU161, acu161count,dis_errors, zero_count, true_zero, area, dis_list = evaluate(new_places1, true_dictionary,true_dictionary)#,true_dictionary
    # MES.append(int(ave_error))
    # ACU161s.append(round(ACU161, 2))
    # print('voting2', P,R,F, ave_error, median_error, ACU161, acu161count,area)
    # voting2_ave = ave_error
    # voting2_161 = ACU161
    # total_dis_list.append(dis_list) new_places1,new_places2,new_places3, new_places4,
    # all_systems = []
    all_systems = [fishing_esti_dictionary, dca_esti_dictionary,  rel_esti_dictionary, blink_esti_dictionary, bootleg_esti_dictionary,  \
                    genre_esti_dictionary, extend_esti_dictionary, luke_esti_dictionary, stan_esti_dictionary, adapter_esti_dictionary, \
                        geopop_esti_dictionary,clavin_esti_dictionary,topocluster_esti_dictionary, mordecai_esti_dictionary, \
                      CBH_esti_dictionary, SHS_esti_dictionary,CHF_esti_dictionary, cam_esti_dictionary, new_places]
    partial_systems = [dbp_esti_dictionary, ein_esti_dictionary] # 
    # all_systems.extend(partial_systems) #babelfy_esti_dictionary, tagme_esti_dictionary, mordecai_esti_dictionary,
    #all_systems = [topocluster_esti_dictionary,mordecai_esti_dictionary]
    # # all_systems = [new_places,new_places2,new_places3, new_places4, new_places5,new_places6,bootleg_esti_dictionary]
    #partial_systems = []
    single_ave = []
    single_161 = []
    single_areas = []
    colors = ['blue','red']
    alphas = [0.8,0.5]
    color_c = 0
    width = 12
    height = 5
    fig, axs = plt.subplots(figsize=(width, height), nrows=1, ncols=2)
    for i, system in enumerate(all_systems):
        P,R,F, ave_error, median_error, ACU161, acu161count, dis_errors, zero_count, \
            true_zero, area , dis_list, max_error_num=\
                evaluate(system, true_dictionary, true_dictionary, show_index=i) #
        if bool_plot:
            if i in [11,18]:
                new_list = [(np.log(x+1)/np.log(max_error)) for x in dis_list]
                axs[0].plot(range(len(dis_list)), dis_list, color=colors[color_c],linewidth=1)
                axs[0].fill_between(range(len(dis_list)), dis_list, alpha=alphas[color_c], color=colors[color_c])
                axs[1].plot(range(len(dis_list)), new_list, color=colors[color_c],linewidth=1)
                axs[1].fill_between(range(len(dis_list)), new_list, alpha=alphas[color_c],  color=colors[color_c])
                color_c += 1

        print(P,R,F, ave_error, median_error, ACU161,dis_errors, acu161count,area,zero_count,max_error_num)
        MES.append(int(ave_error))
        ACU161s.append(round(ACU161, 2))
        AUCs.append(round(area, 2))
        total_dis_list.append(dis_list)
    ACU161s_subsets.append(ACU161s)
    MES_subsets.append(MES)
    AUC_subsets.append(AUCs)

    if bool_plot:
        axs[0].legend(labels=['CLAVIN','Voting'],fontsize=12)
        axs[1].legend(labels=['CLAVIN','Voting'],fontsize=12)
        axs[0].set_ylabel('distance error (km)',fontsize=14)
        axs[0].set_xlabel('index of toponyms',fontsize=14)
        axs[1].set_ylabel('log distance error (km)',fontsize=14)
        axs[1].set_xlabel('index of toponyms',fontsize=14)
        fig1 = plt.gcf()
        fig1.set_size_inches(width, height)
        fig1.savefig('auc.png', dpi=300,bbox_inches='tight')
        plt.show()
    
    recall = []
    if write:
        for i, system in enumerate(partial_systems):
            cur_ACU161s = []
            cur_MES = []
            cur_AUC = []
            P,R,F, ave_error, median_error, ACU161, acu161count,dis_errors, \
            zero_count, true_zero, area , dis_list, max_error_num = evaluate(system, true_dictionary,true_dictionary)
            recall.append(R)
            cur_MES.append(int(ave_error))
            cur_ACU161s.append(round(ACU161, 2))
            cur_AUC.append(round(area, 2))
            print('origin ', P,R,F, ave_error, median_error, ACU161, acu161count)

            total_dis_list.append(dis_list)
            for another_system in all_systems:
            #new_places1 = merge(systems, true_dictionary.keys(), eps_km, min_samples, weights, bool_weight,4,true_dictionary)
                P,R,F, ave_error, median_error, ACU161, acu161count,dis_errors,\
                    zero_count, true_zero, area , dis_list, max_error_num = evaluate(another_system, true_dictionary,system)
                cur_MES.append(int(ave_error))
                cur_ACU161s.append(round(ACU161, 2))
                cur_AUC.append(round(area, 2))
                print('align ', P,R,F, ave_error, median_error, ACU161, acu161count)
                total_dis_list.append(dis_list)
            ACU161s_subsets.append(cur_ACU161s)
            MES_subsets.append(cur_MES)
            AUC_subsets.append(cur_AUC)

        
    return ACU161s_subsets, MES_subsets, AUC_subsets, total_dis_list, recall

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--eps', type=int, default=300)
    parser.add_argument('--min_samples', type=int, default=2)
    parser.add_argument('--eps_2', type=float, default=10)

    parser.add_argument('--data', type=str, default='geocorpora')
    parser.add_argument('--bool_weight', type=int, default=0)
    parser.add_argument('--must_contain_da', type=int, default=-1)
    parser.add_argument('--write', type=int, default=0)
    parser.add_argument('--bool_importance', type=int, default=0)
    parser.add_argument('--bool_senstive', type=int, default=0)

    weights = [1]*Total_systems
    args = parser.parse_args()
    print ('eps: '+str(args.eps))
    print ('min_samples: '+str(args.min_samples))
    print ('data: '+str(args.data))
    total_acu161 = []
    total_me = []
    total_auc = []
    base_dir = '../../experiments/'
    datasets =['lgl','neel','trnews','gwn','geocorpora','geovirus','wiktor','wotr', 'LDC', 'TUD','semeval', '19th']
    if args.bool_importance:
        r1, r2, r3 = importance(datasets, args.eps, args.min_samples, args.eps2, bool_weight =0, must_contain_da = -1 )
        print(r1)
        print(r2)
        print(r3)
    elif args.bool_senstive:
        # datasets = ['neel','trnews']
        # r1, r2, r3 = senstive_analysis(datasets, list(range(100,5000,300)), [args.min_samples], [args.eps_2], bool_weight =0, must_contain_da = -1 )
        # print('result1:', r1, r2, r3)

        # r7, r8, r9 = senstive_analysis(datasets, [args.eps], [args.min_samples], list(range(100,300,30)), bool_weight =0, must_contain_da = -1 )
        # print('result2:', r7, r8, r9)

        r4, r5, r6 = senstive_analysis(datasets, [args.eps], list(range(1,12,1)), [args.eps_2], bool_weight =0, must_contain_da = -1 )
        # print(r1, r2, r3)
        print(r4, r5, r6)
        # print(r7, r8, r9)
        
    else:
        if args.write:
            results = []
            recalls1 = []
            recalls2 = []
            for data in datasets: #'geovirus',
                ACU161s_subsets, MES_subsets, AUC_subsets, total_dis_list, recall = evaluate1(args.eps,args.min_samples, args.eps_2, data, weights, args.bool_weight, args.must_contain_da, 1)
                recalls1.append(recall[0])
                recalls2.append(recall[1])
                # results.extend(return_result)
                total_acu161.append(ACU161s_subsets)
                total_me.append(MES_subsets)
                total_auc.append(AUC_subsets)
            for toponym in range(len(total_acu161[0])):
                results = []
                for data in total_acu161:
                    results.append(data[toponym])
                
                output = base_dir+'acu161'+str(toponym)+'2.csv'
                # print(results)
                with open(output, 'w') as f:
                    wtr = csv.writer(f, delimiter= ',')
                    wtr.writerows(results)
    
            for toponym in range(len(total_me[0])):
                results = []
                for data in total_me:
                    results.append(data[toponym])
                
                output = base_dir+'me'+str(toponym)+'2.csv'
                # print(results)
                with open(output, 'w') as f:
                    wtr = csv.writer(f, delimiter= ',')
                    wtr.writerows(results)   
    
            for toponym in range(len(total_auc[0])):
                results = []
                for data in total_auc:
                    results.append(data[toponym])
                
                output = base_dir+'auc'+str(toponym)+'2.csv'
                # print(results)
                with open(output, 'w') as f:
                    wtr = csv.writer(f, delimiter= ',')
                    wtr.writerows(results)
            print(recalls1,recalls2)
            print(np.mean(recalls1),np.mean(recalls2))
        else:
            evaluate1(args.eps,args.min_samples,args.eps_2,args.data, weights, args.bool_weight, args.must_contain_da)


if __name__ == '__main__':
    main()
    
