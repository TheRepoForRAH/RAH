import os
import re
import sys
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_api import chatapi

# helper

def load_prompt(prompt_path):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return prompt

def format_prefer_disprefer(user_prefer, user_disprefer):
    if user_prefer is None:
        user_prefer = []
    if user_disprefer is None:
        user_disprefer = []

    if isinstance(user_prefer, list):
        user_prefer = "\n".join([f"$$ {prefer}" for prefer in user_prefer])
    if isinstance(user_disprefer, list):
        user_disprefer = "\n".join([f"$$ {disprefer}" for disprefer in user_disprefer])
    
    if len(user_prefer)==0:
        user_prefer = "unknown yet"
    if len(user_disprefer)==0:
        user_disprefer = "unknown yet"
    
    return user_prefer, user_disprefer

def format_history_like_dislike(like_items, dislike_items):
    if like_items is None:
        like_items = []
    if dislike_items is None:
        dislike_items = []

    if isinstance(like_items, list) and len(like_items)>0:
        like_items = str(like_items)
    if isinstance(dislike_items, list) and len(dislike_items)>0:
        dislike_items = str(dislike_items)
    
    if len(like_items)==0:
        like_items = "unknown yet"
    if len(dislike_items)==0:
        dislike_items = "unknown yet"
    
    return like_items, dislike_items

class GeneralAgent():
    def __init__(self, prompt_path, llm_args): # can be abstract
        self.llm_args = llm_args
        self.sys_prompt_temp = load_prompt(prompt_path)
    
    def render_sys_prompt(self):    
        return self.sys_prompt_temp

    def render_human_prompt(self):
        raise NotImplementedError

    def respond(self):
        raise NotImplementedError
    
    def postprocess(self):
        raise NotImplementedError

# component agent
class PerceiveAgent(GeneralAgent):
    def __init__(self, prompt_path, llm_args):
        super().__init__( prompt_path, llm_args)

    def render_human_prompt(self, item):
        return f"Item: {item}"

    def respond(self, item): 
        sys_prompt = self.render_sys_prompt()
        human_prompt = self.render_human_prompt(item)
        response = chatapi(sys_prompt, human_prompt, **self.llm_args)
        return self.postprocess(item, response)

    def postprocess(self, item, response):
        return {
                "item":item,
                "item_information":response
            }

class LearnAgent(GeneralAgent):
    def __init__(self, prompt_path, llm_args):
        super().__init__(prompt_path, llm_args)
    
    def render_human_prompt(self, item, item_information, user_action, user_comment=None, previous_learn=None):
        human_prompt = f"Item: {item}\nItem information: {item_information}\nUser Action:{user_action}\n"
        if user_comment and len(user_comment)>0:
            human_prompt+= f"User Comment on Item:{user_comment}\n"
        if previous_learn:
            human_prompt += f"Previous learn:{previous_learn}\n"
        return human_prompt

    def respond(self, item, item_information, user_action, user_comment=None, previous_learn=None, **kwargs):
        sys_prompt = self.render_sys_prompt()
        human_prompt = self.render_human_prompt(item, item_information, user_action, user_comment, previous_learn)
        response = chatapi(sys_prompt, human_prompt, **self.llm_args)
        return self.postprocess(response, user_action)

    def postprocess(self, response, user_action):
        prefer_pattern = r'\$\$ prefer:(.*?)(?:\n|$)'
        disprefer_pattern = r'\$\$ disprefer:(.*?)(?:\n|$)'
        prefer_matches = re.findall(prefer_pattern, response, re.DOTALL)
        disprefer_matches = re.findall(disprefer_pattern, response, re.DOTALL)
        user_prefer = list(match.strip() for match in prefer_matches)
        user_disprefer = list(match.strip() for match in disprefer_matches)

        if "dislike" in user_action.lower():
            user_prefer = []
        else:
            user_disprefer = []

        return {"response":response, "user_prefer":user_prefer, "user_disprefer":user_disprefer}


class ActionAgent(GeneralAgent):
    def __init__(self, prompt_path, llm_args, with_personality=True):
        super().__init__(prompt_path, llm_args)

    def render_human_prompt(self, item, item_information=None, 
                            user_prefer=None, user_disprefer=None):
        human_prompt = f"Item: {item}\n"
        if item_information:
            human_prompt += f"Item information: {item_information}\n"

        user_prefer, user_disprefer = format_prefer_disprefer(user_prefer, user_disprefer)
        human_prompt += f"User Prefer: \n{user_prefer}\nUser Disprefer: \n{user_disprefer}\n"

        return human_prompt

    
    def respond(self, item, item_information=None, 
                            user_prefer=None, user_disprefer=None, **kwargs):
        sys_prompt = self.render_sys_prompt()
        human_prompt = self.render_human_prompt(item, item_information, user_prefer, user_disprefer)
        response = chatapi(sys_prompt, human_prompt, **self.llm_args)
        return self.postprocess(response)


    def postprocess(self, response):
        comment_pattern = r'User Comment:(.*?)(?:\n|$)'   
        action_pattern = r'User Action:(.*?)(?:\n|$)'     

        comment_matches = re.findall(comment_pattern, response, re.DOTALL)
        action_matches = re.findall(action_pattern, response, re.DOTALL)

        comment = ", ".join(match.strip() for match in comment_matches)
        action = ", ".join(match.strip() for match in action_matches)

        return {"response":response, "comment":comment, "action":action}

class CriticAgent(GeneralAgent):
    def __init__(self, prompt_path, llm_args):
        super().__init__(prompt_path, llm_args)

    def render_human_prompt(self, user_prefer, user_disprefer,    
                            item, item_information, 
                            prediction_action, groundtruth_action):
        user_prefer, user_disprefer = format_prefer_disprefer(user_prefer, user_disprefer)
        data = {
            "user_prefer":user_prefer, "user_disprefer":user_disprefer,
            "item":item, "item_information":item_information,
            "prediction_action":prediction_action, "groundtruth_action":groundtruth_action
        }
        human_prompt = "User Prefer: \n{user_prefer}\nUser Disprefer: \n{user_disprefer}\nItem: {item}\nItem Information: \n{item_information}\nUser's action (to the item) prediction: {prediction_action}\nUser's action (to the item) groundtruth: {groundtruth_action}\n".format(**data)
        return human_prompt

    def respond(self, user_prefer, user_disprefer, 
                item, item_information, 
                prediction_action, groundtruth_action, **kwargs):
        sys_prompt = self.render_sys_prompt()
        human_prompt = self.render_human_prompt(
                    user_prefer, user_disprefer, 
                    item, item_information, 
                    prediction_action, groundtruth_action)
        response = chatapi(sys_prompt, human_prompt, **self.llm_args)
        return self.postprocess(response)

    def postprocess(self, response):

        accurate_pattern = r'Accurate:(.*?)(?:\n|$)'
        reason_pattern = r'\$\$ reason:(.*?)(?:\n|$)'
        suggestion_pattern = r'\$\$ suggestion:(.*?)(?:\n|$)'

        accurate_matches = re.findall(accurate_pattern, response, re.DOTALL)
        reason_matches = re.findall(reason_pattern, response, re.DOTALL)
        suggestion_matches = re.findall(suggestion_pattern, response, re.DOTALL)
        
        accurate = " ".join(match.strip() for match in accurate_matches)
        reasons = "; ".join(match.strip() for match in reason_matches)
        suggestions = "; ".join(match.strip() for match in suggestion_matches)
        
        return {"response":response, "accurate":accurate, "suggestions":suggestions, "reasons":reasons}

class ReflectAgent(GeneralAgent):
    def __init__(self, prompt_path, llm_args):
        super().__init__(prompt_path, llm_args)

    def render_human_prompt(self,user_prefer, user_disprefer):
        user_prefer, user_disprefer = format_prefer_disprefer(user_prefer, user_disprefer)
        human_prompt = f"User Prefer: \n{user_prefer}\nUser Disprefer: \n{user_disprefer}\n"
        return human_prompt

    def respond(self, user_prefer, user_disprefer, **kwargs):
        sys_prompt = self.render_sys_prompt()
        human_prompt = self.render_human_prompt(user_prefer, user_disprefer)
        response = chatapi(sys_prompt, human_prompt, **self.llm_args)
        return self.postprocess(response)

    def postprocess(self, response):
        prefer_pattern = r'\$\$ prefer:(.*?)(?:\n|$)'
        disprefer_pattern = r'\$\$ disprefer:(.*?)(?:\n|$)'
        prefer_matches = re.findall(prefer_pattern, response, re.DOTALL)
        disprefer_matches = re.findall(disprefer_pattern, response, re.DOTALL)
        user_prefer = list(match.strip() for match in prefer_matches)
        user_disprefer = list(match.strip() for match in disprefer_matches)

        if len(user_prefer) !=0 or len(user_disprefer) !=0:
            return {"response":response, "user_prefer":user_prefer, "user_disprefer":user_disprefer}
        
        prefer_section = re.search(r'Optimized User Prefer:(.*?)Optimized User Disprefer:', response, re.DOTALL | re.IGNORECASE)
        disprefer_section = re.search(r'Optimized User Disprefer:(.*)', response, re.DOTALL | re.IGNORECASE)

        if prefer_section and disprefer_section:
            user_prefer = prefer_section.group(1).replace("\n","").strip().split('$$')[1:]
            user_disprefer = disprefer_section.group(1).replace("\n","").strip().split('$$')[1:]

        return {"response":response, "user_prefer":user_prefer, "user_disprefer":user_disprefer}

def get_agents(prompt_dir):

    prompt_path = os.path.join(prompt_dir, "perceive.txt")
    llm_args = {"model_name":"gpt-4-0613"}
    perceive_agent = PerceiveAgent(prompt_path, llm_args)

    prompt_path = os.path.join(prompt_dir, "learn.txt")
    llm_args = {"model_name":"gpt-4-0613"}
    learn_agent = LearnAgent(prompt_path, llm_args)

    prompt_path = os.path.join(prompt_dir, "action.txt")
    llm_args = {"model_name":"gpt-4-0613"}
    action_agent = ActionAgent(prompt_path, llm_args)

    prompt_path = os.path.join(prompt_dir, "reflect.txt")
    llm_args = {"model_name":"gpt-4-0613"}
    reflect_agent = ReflectAgent(prompt_path, llm_args)

    prompt_path = os.path.join(prompt_dir, "critic.txt")
    llm_args = {"model_name":"gpt-4-0613"}
    critic_agent = CriticAgent(prompt_path, llm_args)

    return perceive_agent, learn_agent, action_agent, reflect_agent, critic_agent