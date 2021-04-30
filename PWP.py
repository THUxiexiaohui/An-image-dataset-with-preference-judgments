#Code for the Preference-Winning-Penalty(PWP) metric
#Reference: Preference-based Evaluation Metrics for Web Image Search. SIGIR 2020

import json
import scipy.stats as stats
import math
import numpy as np

def rbp(r, p, method = 0):
    ll = 0
    now_p = 1
    for i in range(len(r)):
        ll += r[i] * now_p
        now_p *= p
    ll *= (1-p)
    return ll

def dcg_at_k(r, k, method=0):

    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def score_map(score_list):
    #major voting
    vote = [[0, 0], [1, 0], [2,0]]
    for i in range(len(score_list)):
        score_list[i] = int(float(score_list[i]))
        if score_list[i] < 0:
            score_list[i] = 0
            vote[0][1] += 1
            continue
        elif score_list[i] > 0:
            score_list[i] = 2
            vote[2][1] += 1
            continue
        else:
            score_list[i] = 1
            vote[1][1] += 1
            continue

    vote = sorted(vote, key = lambda x:x[1], reverse=True)
    score = vote[0][0]
    return score

def read_data():
    #Read the position info of images
    with open("Image_position.json", "r") as load_f:
        query_img_position_dic = json.load(load_f)

    #Read SERP level preferecne
    SERP_level_pref_dic = {}
    with open("SERP_level_preference", "r") as f_in:
        line = f_in.readline() #format
        while True:
            line = f_in.readline()
            if not line:
                break
            query, pref_j = line.strip().split("\t")
            SERP_level_pref_dic[query] = int(pref_j)

    #Read preference judgments of image pairs
    query_image_pref_dict = {}
    with open("image_pairs_annotation", "r") as f_in:
        line = f_in.readline()
        while True:
            line = f_in.readline()
            if not line:
                break
            arr = line.strip().split("\t")
            query = arr[0]
            if query not in query_image_pref_dict:
                query_image_pref_dict[query] = [{}, {}, {}] #sogou, baidu, sogou-baidu
            image_pairs = arr[1].split(",")
            score = score_map(arr[2:])
            index_pairs = []
            for image_pair in image_pairs:
                engine, image_pair = image_pair.split("/")[0], image_pair.split("_")[1]
                image_pair = image_pair[0:len(image_pair) - 4]
                index_pairs.append([engine, int(image_pair)])
            pair = [index_pairs[0][1], index_pairs[1][1]]
            if index_pairs[0][0] == 'sogou' and index_pairs[1][0] == 'sogou':
                record_indx = 0
            if index_pairs[0][0] == 'baidu' and index_pairs[1][0] == 'baidu':
                record_indx = 1
            if index_pairs[0][0] == 'sogou' and index_pairs[1][0] == 'baidu':
                record_indx = 2
            if index_pairs[0][0] == 'baidu' and index_pairs[1][0] == 'sogou':
                score = 2 - score
                pair = [index_pairs[1][1], index_pairs[0][1]]
                record_indx = 2
            if pair[0] not in query_image_pref_dict[query][record_indx]:
                query_image_pref_dict[query][record_indx][pair[0]] = {}
            query_image_pref_dict[query][record_indx][pair[0]][pair[1]] = score
            if record_indx == 2:
                continue
            if pair[1] not in query_image_pref_dict[query][record_indx]:
                query_image_pref_dict[query][record_indx][pair[1]] = {}
            query_image_pref_dict[query][record_indx][pair[1]][pair[0]] = 2 - score
    return [query_image_pref_dict, SERP_level_pref_dic, query_img_position_dic]


def PWP(score_arr, position_arr):
    score_cnt = [[0, 0], [0, 0], [0, 0, 0], {}, {}]
    strategy = 4
    for system_i in range(0, 2):
        s_arr = score_arr[system_i]
        p_arr = position_arr[str(system_i)]
        for host in s_arr:
            for enemy in s_arr[host]:
                if enemy < host:
                    continue
                if strategy == 4: # Nearby principle
                    col_info_left, col_info_right = p_arr[str(host)][1], \
                                                    p_arr[str(enemy)][1]
                    if abs(col_info_left - col_info_right) > 2:
                        continue
                    if (s_arr[host][enemy] == 0) or (s_arr[host][enemy] == 1):
                        score_cnt[system_i][0] += 1
                    score_cnt[system_i][1] += 1

    system_i = 2
    s_arr = score_arr[system_i]
    for host in s_arr:
        for enemy in s_arr[host]:
            if host not in score_cnt[3]:
                score_cnt[3][host] = 0
            if enemy not in score_cnt[4]:
                score_cnt[4][enemy] = 0
            if s_arr[host][enemy] == 0:
                score_cnt[2][0] += 1
                score_cnt[4][enemy] += 1
            if s_arr[host][enemy] == 1:
                score_cnt[2][1] += 1
            if s_arr[host][enemy] == 2:
                score_cnt[2][2] += 1
                score_cnt[3][host] += 1

    a = 0.7
    b = 1-a
    c = 0.1
    PMR_sogou = score_cnt[0][0] * 1.0 / score_cnt[0][1]
    PMR_baidu = score_cnt[1][0] * 1.0 / score_cnt[1][1]
    WR_sogou = score_cnt[2][0] * 1.0 / sum(score_cnt[2])
    WR_baidu = score_cnt[2][2] * 1.0 / sum(score_cnt[2])
    PW_sogou = a * PMR_sogou + b * WR_sogou
    PW_baidu = a * PMR_baidu + b * WR_baidu
    sogou_loose_map = score_cnt[3]
    baidu_loose_map = score_cnt[4]
    PB_sogou = 1
    PB_baidu = 1
    for sk in sogou_loose_map:
        if sogou_loose_map[sk] == len(baidu_loose_map):
            PB_sogou = PB_sogou * c
    for bk in baidu_loose_map:
        if baidu_loose_map[bk] == len(sogou_loose_map):
            PB_baidu = PB_baidu * c
    return [PW_sogou * PB_sogou, PW_baidu * PB_baidu]


def read_rel_data():
    query_image_rel_dic = {}
    with open("relevance_data", "r") as f_in:
        line = f_in.readline()
        while True:
            line = f_in.readline()
            if not line:
                break
            arr = line.strip().split("\t")
            query = arr[0]
            engine = arr[1].split("/")[0]
            img_id = int(arr[1].split("_")[1].split(".")[0])
            rel_score = float(arr[2])
            engine_id = 0 if engine == "sogou" else 1
            if query not in query_image_rel_dic:
                query_image_rel_dic[query] = {}
            if engine_id not in query_image_rel_dic[query]:
                query_image_rel_dic[query][engine_id] = []
            query_image_rel_dic[query][engine_id].append([img_id, rel_score])
    return query_image_rel_dic


def relevance_metric(metric_name, score_arr):
    sorted_score_arr = [sorted(score_arr[0], key=lambda x:x[0]), sorted(score_arr[1], key=lambda x:x[0])]
    new_arr = [[], []]
    for i in range(2):
        for j in range(len(sorted_score_arr[i])):
            new_arr[i].append(sorted_score_arr[i][j][1])

    if metric_name == "RBP":
        sogou_score = rbp(new_arr[0], 0.99)
        baidu_score = rbp(new_arr[1], 0.99)
    if metric_name == "NDCG":
        sogou_score = ndcg_at_k(new_arr[0], 10)
        baidu_score = ndcg_at_k(new_arr[1], 10)
    return [sogou_score, baidu_score]
    

if __name__ == "__main__":
    query_image_pref_dict, SERP_level_pref_dic, query_img_position_dic = read_data()
    query_image_rel_dic = read_rel_data()
    metric_scores = []
    SERP_scores = []
    for query in SERP_level_pref_dic:
        golden_standard = SERP_level_pref_dic[query]
        sogou_score, baidu_score = PWP(query_image_pref_dict[query], query_img_position_dic[query])
        # sogou_score, baidu_score = relevance_metric("NDCG", query_image_rel_dic[query])
        SERP_scores.append(golden_standard)
        metric_scores.append(1.0 / (1 + math.exp(sogou_score - baidu_score)))
    print(stats.pearsonr(metric_scores, SERP_scores))


