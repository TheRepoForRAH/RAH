import os
import sys
import json
import pandas as pd
from tqdm import tqdm

def read_json_lines(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_reviews(data_dir, domains):
    reviews = [] 
    for domain in domains:
        review_path = os.path.join(data_dir, f"{domain}.csv")
        review = pd.read_csv(review_path)
        reviews.append(review)
    return reviews

def read_metas(data_dir, domains):
    metas = []
    for domain in domains:
        meta_path = os.path.join(data_dir, f"meta_{domain}.csv")
        meta = pd.read_csv(meta_path)
        metas.append(meta)
    return metas

def get_one_user(reviewerID, review_list = [], meta_list = [], domain_list=[]):

    result_list = []
    if len(domain_list)<len(review_list):
        domain_list = domain_list + ["default"] * (len(review_list) - len(domain_list))
    for review, meta, domain in zip(review_list, meta_list, domain_list):
        selected_review = review.loc[review["reviewerID"]==reviewerID]
        selected_meta = meta.loc[meta['asin'].isin(review["asin"])]
        # overall,verified,reviewTime,reviewerID,asin,style,reviewerName,reviewText,summary,unixReviewTime,vote,image
        useful_columns = ["reviewerID", "reviewerName", "reviewTime", "asin", "title", "overall", "reviewText", "description"]
        result = pd.merge(selected_review, selected_meta, left_on="asin", right_on="asin").loc[:, useful_columns]
        result = result.drop_duplicates(subset=["reviewerID", "title"])
        result["reviewTime"] = pd.to_datetime(result["reviewTime"])
        result = result.sort_values("reviewTime")
        result["domain"] = domain

        result_list.append(result)
    
    return result_list

def find_strict_users(users, reviews, metas, strict_level=3):

    # strict user
    strict_users = []
    for user_id in tqdm(users):
        strict = True
        history = get_one_user(user_id, reviews, metas)
        for h in history:
            score_count = dict(h["overall"].value_counts())
            if h.loc[h["overall"]<=2].shape[0] < strict_level:
                strict = False
                break
        
        if strict:
            strict_users.append(user_id)
    return strict_users

def find_tradeoff_users(users, reviews, metas):

    # tradeoff user
    tradeoff_users = []
    for user_id in tqdm(users):
        history = get_one_user(user_id, reviews, metas)
        dislike_cnt = 0
        like_cnt = 0
        for h in history:
            score_count = dict(h["overall"].value_counts())
            dislike_cnt += score_count.get(1, 0) + score_count.get(2, 0)
            like_cnt += score_count.get(4, 0) + score_count.get(5, 0)

        if dislike_cnt > 0 and like_cnt > 0:
            tradeoff_users.append((user_id, min(dislike_cnt,like_cnt)/max(like_cnt,dislike_cnt)))
        else:
            tradeoff_users.append((user_id, 0))
            
    # the closer to 1, the more tradeoff
    tradeoff_users = sorted(tradeoff_users, key=lambda x:x[1], reverse=True)
    return tradeoff_users


# operation for assistant data

def analyze_records(records):


    learned_items = set()
    seen_list = []
    unseen_list = []
    for record in records:
        if "learn" in record.get("type"):
            learned_items.add(record["item"]["name"])
        if record.get("type") == "act":
            item = record["item"]["name"]
            user_action = record["user_action"]
            accurate = record["result"]["accurate"]
            info = (item, user_action, accurate)
            if item in learned_items:
                seen_list.append(info)
            else:
                unseen_list.append(info)
    return learned_items, seen_list, unseen_list


def make_movie_name(item):
    return item.replace("VHS", "") + " (Movie)"

def make_book_name(item):
    return item + " (Book)"

def make_game_name(item):
    return item + " (Video Game)"

def make_name(item, domain):
    if "movie" in domain.lower():
        return make_movie_name(item)
    if "book" in domain.lower():
        return make_book_name(item)
    if "game" in domain.lower():
        return make_game_name(item)
    
    return item

SCORE_MAP = {5:"like", 4:"like", 3:"neutral", 2:"dislike", 1:"dislike"}
