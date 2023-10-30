import os
import openai
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# openai.api_key = "your key here"

def chatapi(system_prompt, user_prompt, model_name="gpt-4", temperature=0.7):
    while True:
        try:
            time.sleep(random.randint(1, 8))  # for concurrent
            s = openai.ChatCompletion.create(
                model=model_name,
                temperature = temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return s['choices'][0]['message']['content']
        except openai.error.Timeout as e:
            #Handle timeout error, e.g. retry or log
            print(f"OpenAI API request timed out: {e}")
            pass
        except openai.error.APIError as e:
            #Handle API error, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error, e.g. check network or log
            print(f"OpenAI API request failed to connect: {e}")
            pass
        except openai.error.InvalidRequestError as e:
            #Handle invalid request error, e.g. validate parameters or log
            print(f"OpenAI API request was invalid: {e}")
            pass
        except openai.error.AuthenticationError as e:
            #Handle authentication error, e.g. check credentials or log
            print(f"OpenAI API request was not authorized: {e}")
            pass
        except openai.error.PermissionError as e:
            #Handle permission error, e.g. check scope or log
            print(f"OpenAI API request was not permitted: {e}")
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error, e.g. wait or log
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API ServiceUnavailableError: {e}")
            pass
        except KeyError as e:
            print("KeyError")
            pass

    time.sleep(1)
    return "None"
