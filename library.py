from collections import defaultdict
import os
import json

def deep_defaultdict():
    return defaultdict(deep_defaultdict)

def read_jsons_from_dir(directory, filename_to_feature=None):
    json_files = [ pos_json for pos_json in os.listdir(directory) if pos_json.endswith('.json') ]
    data = []

    for index, js in enumerate(json_files):
        try:
            with open(os.path.join(directory, js)) as json_file:
                d = json.load(json_file)
                if filename_to_feature is not None:
                    d[filename_to_feature] = js.split(".")[0]
        except:
            with open(os.path.join(directory, js)) as json_file:
                d = []
                for line in json_file:
                    try:
                        d.append(json.loads(line))
                    except:
                        print(js.split(".")[0])
                    if filename_to_feature is not None:
                        d[-1][filename_to_feature] = js.split(".")[0]
                

        data.append(d)
    
    return data

def load_item_library(item_library_path):
    item_library = {}
    data_list = read_jsons_from_dir(item_library_path)
    for data in data_list:
        try:
            for d in data:
                item_name = d[0]
                if type(d[1])==str:
                    item_library[item_name] = {"item":d[0], "item_information":d[1]}
                elif type(d[1])==dict:
                    item_library[item_name] = d[1]
        except:
            d = data
            item_name = d[0]
            if type(d[1])==str:
                item_library[item_name] = {"item":d[0], "item_information":d[1]}
            elif type(d[1])==dict:
                item_library[item_name] = d[1]
    return item_library

def load_personality_library(personality_library_path):
    personality_library = deep_defaultdict()

    mode_list = os.listdir(personality_library_path)
    for mode in mode_list:
        mode_path = os.path.join(personality_library_path, mode)
        domain_list = os.listdir(mode_path)

        for domain in domain_list:
            domain_path = os.path.join(mode_path, domain)
            data_list = read_jsons_from_dir(domain_path, filename_to_feature = "user_id")

            for data in data_list:
                data=[data] if type(data)!=list else data

                for d in data:
                    user_id = d["user_id"]
                    personality_library[mode][domain][user_id] = d  # Key:Value

    return personality_library

def load_record_library(record_library_path, on_types=["learn-act-critic"]):
    record_library = deep_defaultdict()
    data_list = read_jsons_from_dir(record_library_path, filename_to_feature = "user_id")
    for data in data_list:
        for d in data:
            if d["type"] not in on_types:
                continue
            # KEY: type/user_id/item_id/action/
            record_library[d["type"]][d["user_id"]][d["item_id"]][d["user_action"]] = d  # indicate domain

    return record_library

def read_cross1k_processed(dataset_dir):
    data_map = {}
    user_id_list = list(filter(lambda x: x[0]=="A", os.listdir(dataset_dir)))

    for user_id in user_id_list:
        data_map[user_id] = {}
        user_path = os.path.join(dataset_dir, user_id)
        group_list = os.listdir(user_path)

        for group in group_list:
            data_map[user_id][group] = {}
            group_path = os.path.join(user_path, group)
            domain_filename_list = os.listdir(group_path)

            for domain_filename in domain_filename_list:
                domain_file_path = os.path.join(group_path, domain_filename)
                with open(domain_file_path, "r") as f:
                    domain_data = json.load(f)
                domain = domain_filename.split(".")[0]
                data_map[user_id][group][domain] = domain_data


    return data_map


class Library:
    def __init__(self, item_path, record_path, personality_path, history_path):

        self.item_path = item_path
        self.record_path = record_path
        self.personality_path = personality_path
        self.history_path = history_path

        self.item_dict = load_item_library(item_path)                        # item_name
        self.record_dict = load_record_library(record_path)                  # type/user_id/item_id/action/
        self.personality_dict = load_personality_library(personality_path)   # mode/domain/user_id
        self.history_dict = read_cross1k_processed(history_path)             # user_id/group/domain
        
    def save_item(self, item_id, item_name, item_response, update=False):
        if item_name not in self.item_dict or update:
            # state
            self.item_dict[item_name] = {"item":item_name, "item_information":item_response}
            # file
            save_dir_path = self.item_path
            save_file_path = os.path.join(save_dir_path, "items.json")
            os.makedirs(save_dir_path, exist_ok=True)
            with open(save_file_path, "a") as f:
                json.dump([item_name, self.item_dict[item_name]], f)
                f.write("\n")
        else:
            print("Exists and not update")
    
    def save_personality(self, mode, domain, user_id, personality, update=False):
        if (not self.personality_dict[mode][domain][user_id]) or update:
            # state
            self.personality_dict[mode][domain][user_id] = personality
            # file
            save_dir_path = os.path.join(self.personality_path, mode, domain)
            save_file_path = os.path.join(save_dir_path, f"{user_id}.json")
            os.makedirs(save_dir_path, exist_ok=True)
            with open(save_file_path, "a") as f:
                json.dump(personality, f)
                f.write("\n")
        else:
            print("Exists and not update")
   
    def save_record(self, r_type, user_id, item_id, action, record, update=False):
        if (not self.record_dict[r_type][user_id][item_id][action]) or update:
            # state
            self.record_dict[r_type][user_id][item_id][action] = record

            # file
            save_dir_path = self.record_path
            save_file_path = os.path.join(save_dir_path, f"{user_id}.json")
            os.makedirs(save_dir_path, exist_ok=True)

            with open(save_file_path, "a") as f:
                json.dump(record, f) 
                f.write("\n")
        else:
            print("Exists and not update")

        
