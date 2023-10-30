import os
import re
import sys
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_api import chatapi

class Assistant:
    def __init__(
        self, perceive_agent, learn_agent, action_agent, reflect_agent, critic_agent, user_id, library=None):
        self.set_agents(perceive_agent, learn_agent, action_agent, reflect_agent, critic_agent)
        self.prefer = []
        self.disprefer = []
        self.like_items=[]
        self.dislike_items=[]
        self.record = []
        self.user_id = user_id
        self.library = library

    # [low-level]
    def describe_item(self, item_id=None, item_name="", perceive_agent=None, log=True):
        perceive_agent = perceive_agent or self.perceive_agent
        # process
        if item_name in self.library.item_dict:                          # reuse
            item_response = self.library.item_dict[item_name]
        else:
            item_response = perceive_agent.respond(item = item_name)
            self.library.save_item(item_id, item_name, item_response)    

        if log:
            print("---Item----------------------------------------------------")
            print(item_response)

        return item_response

    def reflect(self, new_prefer, new_disprefer, log=True):
        reflect_agent = self.reflect_agent
        one_record = self.get_record_tempelate(record_type = "reflect")

        # process
        process = []
        user_prefer_candidate = self.prefer + new_prefer
        user_disprefer_candidate =self.disprefer + new_disprefer
        reflect_response = reflect_agent.respond(user_prefer_candidate, user_disprefer_candidate)
        reflected_prefer, reflected_disprefer = reflect_response["user_prefer"], reflect_response["user_disprefer"]
        reflect_log = reflect_response["response"]
        process.append(
            {
                "exist_personality":{"prefer":self.prefer, "disprefer":self.disprefer},
                "new_personality":{"prefer":new_prefer, "disprefer":new_disprefer},
                "log":{"reflect":reflect_log}
            }
        )
        if log:
            print("---Reflect----------------------------------------------------")
            print(reflect_log)

        # update
        one_record["process"] = process
        if len(reflected_prefer)>=len(self.prefer)//5:
            self.prefer = reflected_prefer
            one_record["result"]["personality"]["prefer"] = reflected_prefer
        else: 
            # avoid incorrect reflection having extreme change
            one_record["result"]["personality"]["prefer"] = self.prefer
            print("prfer too short")
        
        if len(reflected_disprefer)>=len(self.disprefer)//5:
            self.disprefer = reflected_disprefer
            one_record["result"]["personality"]["disprefer"] = reflected_disprefer
        else:
            # avoid incorrect reflection having extreme change
            one_record["result"]["personality"]["disprefer"] = self.disprefer
            print("disprefer too short")

        
        self.record.append(one_record)

        return one_record

    def act(self, item_id, item_name, user_action=None, prompt_path=None, log=True, save=True, **kwargs):

        perceive_agent = self.perceive_agent  
        action_agent = self.action_agent
        one_record = self.get_record_tempelate(record_type="act", item_id=item_id, item_name=item_name, item_info="", user_action="", user_comment="")
        item_response = self.describe_item(item_id, item_name, perceive_agent, log)
  
        # process
        action_response = action_agent.respond(
                                    **item_response, 
                                    user_prefer = self.prefer,
                                    user_disprefer = self.disprefer)

        action_log = action_response.pop("response")
        # record
        assistant_action = action_response["action"][:10]
        if user_action is not None:
            one_record["user_action"] = user_action
            if ("dislike" in assistant_action.lower() and "dislike" in user_action.lower()) or \
                ("dislike" not in assistant_action.lower() and "dislike" not in user_action.lower()):
                accurate = "True"
            else:
                accurate = "False"
        else:
            accurate = None

        if log:
            print(f"---Action ----------------------------------------------------")
            print(assistant_action, accurate)

        one_record["process"].append(
            {
                "index":0,
                "assistant_action":assistant_action,
                "log":{"action":action_log},
                "accurate":accurate
            }
        )

        # update state
        one_record["item"]["information"] = item_response["item_information"]
        one_record["result"]["assistant_action"] = assistant_action
        one_record["result"]["accurate"] = accurate
        self.record.append(one_record)
        if save:
            self.library.save_record("act", self.user_id, item_id, user_action, one_record)
        return one_record


    def step_learn_act_critic(self, item_id, item_name, user_action, user_comment=None, max_try_times=2, log=True, save=True, **kwargs):

        is_new = False
        # used agents
        perceive_agent, learn_agent, action_agent, critic_agent = self.perceive_agent, self.learn_agent, self.action_agent, self.critic_agent

        # check learn record exist
        # 1. direct resuse
        if self.library.record_dict["learn-act-critic"][self.user_id][item_id][user_action]:                
            one_record = self.library.record_dict["learn-act-critic"][self.user_id][item_id][user_action]
            new_user_prefer = one_record["process"][-1]["new_personality"]["prefer"]
            new_user_disprefer = one_record["process"][-1]["new_personality"]["disprefer"]
        # 2. init record
        else:
            is_new = True
            one_record = self.get_record_tempelate(record_type="learn-act-critic", item_id=item_id, item_name=item_name, 
                                                        user_action=user_action, user_comment=user_comment)
            # item
            item_response = self.describe_item(item_id, item_name, perceive_agent)

            # learn-act-critic process
            (index, previous_learn, process) = (-1, None, [])
            while True:
                index += 1
                # (1) learn
                learn_response = learn_agent.respond(**item_response, user_action=user_action, user_comment=user_comment, 
                                                        previous_learn=previous_learn, **kwargs)
                new_user_prefer = learn_response["user_prefer"]
                new_user_disprefer = learn_response["user_disprefer"]
                learn_log = learn_response.pop("response")
                if log:
                    print(f"---Learn {index}----------------------------------------------------")
                    print(learn_log)

                # (2) act
                action_response = action_agent.respond(**item_response, **learn_response, 
                                                        user_history_like=None, user_history_dislike=None, **kwargs)   # without history in this loop
                action_log = action_response.pop("response")
                if log:
                    print(f"---Action {index}----------------------------------------------------")
                    print(action_log)

                # (3) critic
                critic_response = critic_agent.respond(prediction_action=action_response["action"],
                                                        groundtruth_action=user_action,
                                                        **item_response, **learn_response)
                critic_log = critic_response.pop("response")
                if log:
                    print(f"---Critic {index}----------------------------------------------------")
                    print(critic_log)
                
                accurate = critic_response["accurate"]
                reasons = critic_response["reasons"]
                suggestions = critic_response["suggestions"]
                previous_learn = self.get_previous_learn(new_user_prefer, new_user_disprefer, action_response["action"], reasons, suggestions)
                
                # (4) record process
                process.append(
                    {
                        "index":index,
                        "new_personality":{"prefer":new_user_prefer, "disprefer":new_user_disprefer},
                        "assistant_action":action_response["action"],
                        "accurate":accurate,
                        "suggestion":suggestions,
                        "reasons":reasons,
                        "log":{ "learn":learn_log, "action":action_log, "critic":critic_log }
                    }
                )
                if "True" in accurate:
                    stop_reason = "success"
                    print("Critic Success!")
                    break
                if index >= max_try_times:
                    stop_reason = "max_try"
                    print("Reach max try!")
                    break

            # update state
            one_record["item"]["information"] = item_response["item_information"]
            one_record["process"] = process
            one_record["stop_reason"] = stop_reason

        prefer_candidate = self.prefer + [new_user_prefer]
        disprefer_candidate = self.disprefer + [new_user_disprefer]
        one_record["result"]["new_personality"] = {"prefer":new_user_prefer, "disprefer":new_user_disprefer}
        one_record["result"]["personality"] = {"prefer":prefer_candidate, "disprefer":disprefer_candidate}
        if is_new and save:
            self.library.save_record("learn-act-critic", self.user_id, item_id, user_action, one_record)

        return one_record

    def step_learn(self, item_id, item_name, user_action, user_comment=None, log=True, save=True, **kwargs):

        is_new = False
        # used agents
        perceive_agent = self.perceive_agent
        learn_agent = self.learn_agent

        # check learn record exist
        # 1. direct reuse
        if self.library.record_dict["learn"][self.user_id][item_id][user_action]:                   
            one_record = self.library.record_dict["learn"][self.user_id][item_id][user_action] 
            new_user_prefer = one_record["process"][0]["new_personality"]["prefer"]
            new_user_disprefer = one_record["process"][0]["new_personality"]["disprefer"]
        # 2. indirect or init
        else:
            is_new = True
            # 2.1 indirect reuse
            if self.library.record_dict["learn-act-critic"][self.user_id][item_id][user_action]:
                one_record = self.library.record_dict["learn-act-critic"][self.user_id][item_id][user_action]  
                process = one_record["process"][:1]
                new_user_prefer = process[0]["new_personality"]["prefer"]
                new_user_disprefer = process[0]["new_personality"]["disprefer"]

            # 2.2 init record
            else:          
                one_record = self.get_record_tempelate(record_type="learn", item_id=item_id, item_name=item_name, 
                                                        user_action=user_action, user_comment=user_comment)
                # item
                item_response = self.describe_item(item_id, item_name, perceive_agent)
                one_record["item"]["information"] = item_response["item_information"]
                # learn-act-critic process
                (index, previous_learn, process) = (-1, None, [])
                while True:
                    index += 1
                    # (1) learn
                    learn_response = learn_agent.respond(**item_response, user_action=user_action, user_comment=user_comment, 
                                                            previous_learn=previous_learn, **kwargs)
                    new_user_prefer = learn_response["user_prefer"]
                    new_user_disprefer = learn_response["user_disprefer"]
                    learn_log = learn_response.pop("response")
                    if log:
                        print(f"---Learn {index}----------------------------------------------------")
                        print(learn_log)
                    # (2) record process
                    process.append(
                        {
                            "index":index,
                            "new_personality":{"prefer":new_user_prefer, "disprefer":new_user_disprefer},
                            "assistant_action":action_response["action"],
                            "accurate":accurate,
                            "suggestion":suggestions,
                            "reasons":reasons,
                            "log":{"learn":learn_log}
                        }
                    )

                    # (3) stop judge
                    stop_reason = "max_try"
                    print("Pure learn, just one time!")
                    break
                one_record["stop_reason"] = stop_reason
            
            # update state
            one_record["process"] = process

        prefer_candidate = self.prefer + [new_user_prefer]
        disprefer_candidate = self.disprefer + [new_user_disprefer]
        one_record["result"]["new_personality"] = {"prefer":new_user_prefer, "disprefer":new_user_disprefer}
        one_record["result"]["personality"] = {"prefer":prefer_candidate, "disprefer":disprefer_candidate}
        if is_new and save:
            self.library.save_record("learn", self.user_id, item_id, user_action, one_record)
        
        return one_record            


    # 4. helper
    def get_previous_learn(self, previous_prefer, previous_disprefer, previous_action, reasons, suggestions):
        previous_learn = f"Previous Prefer: {previous_prefer}\n"
        previous_learn+= f"Previous Disprefer: {previous_disprefer}\n" 
        previous_learn+= f"Previous Action is '{previous_action}' but it is wrong.\n"
        previous_learn+= f"Reaons: {reasons}\n"
        previous_learn+= f"Suggestions: {suggestions}"
        return previous_learn

    def get_record_tempelate(self, record_type="learn", item_id="", item_name="", item_info="", user_action="", user_comment="" ):
        if "learn" in record_type:
            one_record = {
                "type":record_type,      
                "item_id":item_id,    # for compatable
                "item":{
                    "id":item_id,
                    "name":item_name,
                    "information":item_info
                },
                "user_action":user_action,
                "user_comment":user_comment,
                "process":[],
                "result":{
                    "new_personality":{},
                    "personality":{},
                    "stop_reason":""            
                }
            }
        elif "reflect" in record_type:
            one_record = {
                "type":record_type,      
                "process":[],
                "result":{
                    "personality":{},       
                }
            }  
        elif "act" in record_type:
            one_record = {
                "type":record_type,      # enumerate
                "item_id":item_id,    # for compatable
                "item":{
                    "id":item_id,
                    "name":item_name,
                    "information":item_info
                },
                "user_action":user_action,
                "user_comment":user_comment,
                "process":[],
                "result":{
                    "assistant_action":{},
                    "accurate":None
                }}
        else:
            one_record = {}

        return one_record


    # 5. set features
    def set_agents(self, perceive_agent=None, learn_agent=None, action_agent=None, reflect_agent=None, critic_agent=None):
        if perceive_agent:
            self.perceive_agent = perceive_agent
        if learn_agent:
            self.learn_agent = learn_agent
        if action_agent:
            self.action_agent = action_agent
        if reflect_agent:
            self.reflect_agent = reflect_agent
        if critic_agent:
            self.critic_agent = critic_agent

    def set_personality(self, prefer=None, disprefer=None):
        if prefer:
            self.prefer = prefer
        if disprefer:
            self.disprefer = disprefer

    def set_like_dislike_items(self, like_items, dislike_items, with_describe=False):
        """Set like and dislike items and describe them for action agent"""
        self.like_items=[]
        self.dislike_items=[]
        self.add_like_dislike_items(like_items, dislike_items, with_describe)
                
    def add_like_dislike_items(self, like_items, dislike_items, with_describe=False):
        """Add like and dislike items and describe them for action agent"""
        for item in like_items:
            if with_describe:
                item_description=self.describe_item(item)
                self.like_items.append((item, item_description['item_information']))
            else:
                self.like_items.append((item, ""))

        for item in dislike_items:
            if with_describe:
                item_description=self.describe_item(item)
                self.dislike_items.append((item, item_description['item_information']))
            else:
                self.dislike_items.append((item, ""))


    # 6. use library

    def load_item_from_library(self):
        return self.library.item_dict      

    def load_personality_from_library(self, mode, domain, user_id=None):
        user_id = user_id or self.user_id
        if self.library.personality_dict[mode][domain][user_id]:
            personality = self.library.personality_dict[mode][domain][user_id]

        # extract personality
        if "prefer" in personality:
            self.prefer = personality["prefer"]
        if "disprefer" in personality:
            self.disprefer = personality["disprefer"]
        
        return None
    
    def load_history_from_library(self, user_id, group, domain):
        return self.library.history[user_id][group][domain]

    def save_personality_to_library(self, mode, domain):
        personality = {"prefer":self.prefer, "disprefer":self.disprefer}
        self.library.save_personality(mode, domain, self.user_id, personality, False)