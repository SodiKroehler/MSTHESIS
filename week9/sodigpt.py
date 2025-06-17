import numpy as np
import pandas as pd
from openai import OpenAI
import time
import json
import re
from rapidfuzz import fuzz, process
from tqdm import tqdm

# class sodiBatchDetails:
#     def __init__(self, batch_size, max_tokens, temperature, prompt, pattern, source_column_name, target_column_llmresp, output_json_parser, json_delimiter):
#         self.batch_size = batch_size
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.prompt = prompt
#         self.pattern = pattern
#         self.source_column_name = source_column_name
#         self.target_column_llmresp = target_column_llmresp
#         self.target_columns = ["out"]
#         self.target_columns_in_json = ["out"]
#         self.json_delimiter = json_delimiter
        



class sodiGPT:
    def __init__(self, system_prompt = "You are a helpful assistant."):
        OPENAI_API_KEY = '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'
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.system_prompt = system_prompt
        self.temperature = 0
        self.batch_size=5
        self.max_tokens=1000
        self.temperature=0
        self.prompt="You are a helpful assistant."
        self.pattern=""
        self.source_column_name="text1"
        self.target_column_llmresp="llm_output"
        self.target_columns = ["out"]
        self.target_columns_in_json = ["out"]
        self.output_json_parser=json.loads
        self.json_delimiter="```"


    def call_getResponse(self, prompt, pattern="", text=""):
        if pattern:
            prompt = prompt.replace(pattern, text)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=self.temperature
            )

            # output = response['choices'][0]['message']['content'].strip()
            output = response.choices[0].message.content.strip()
            return response

        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def call(self, prompt, pattern="", text=""):
        resp = self.call_getResponse(prompt, pattern, text)
        if resp is None:
            return None
        else:
            try:
                output = resp.choices[0].message.content.strip()
                return output
            except Exception as e:
                print(f"Error: {e}")
                return None
            
    def get_json_string_from_llmresp(self, llm_response, delimiter=None):
        if not delimiter:
            delimiter = self.json_delimiter
            
        start, end = 0, len(llm_response)

        delim_match = list(re.finditer(delimiter, llm_response))
        if len(delim_match) > 2:
            start = delim_match[0].start()
            end = delim_match[-1].end()
        else:
            print(f"Not enough occurrences of delimiter '{delimiter}' in the response.")

        return llm_response[start:end].strip()

        
    def get_int_from_llmresp(self, llm_response, default_value=-1.0):
        try:
            lint = re.search(r'\d+\.\d+', llm_response)
            return float(lint.group())
        except Exception as e:
            print(f"Error in parsing float from response: {e}")
            return default_value
        
    def get_int_from_llmresp(self, llm_response, default_value=-1):
        try:
            lint = re.search(r'\d+', llm_response)
            return int(lint.group())
        except Exception as e:
            print(f"Error in parsing integer from response: {e}")
            return default_value
        
    def parse_json(self, llm_response):
        try:
            lint = json.loads(llm_response)
            return lint
        except Exception as e:
            print(f"Error in parsing JSON from response: {e}")
            return None

    
    def fuzz_get(obj, search_key, desired_type, default_value=None):
        score_cuttoff = 80
        if default_value is None:
            default_value = desired_type()

        if not obj:
            return default_value
        
        if not search_key and isinstance(obj, desired_type):
            return obj
        elif not search_key:
            return default_value
        
        if isinstance(obj, dict):
            if search_key in obj and isinstance(obj[search_key], desired_type):
                return obj[search_key]
            else:
                possMatches = []
                for k, v in obj.items():
                    score = fuzz.ratio(search_key, k)
                    if score > score_cuttoff and isinstance(v, desired_type):
                        return v
                    elif score > score_cuttoff:
                        possMatches.append([k, v])
                if len(possMatches) < 1:
                    return default_value
                else:
                    possMatches = sorted(possMatches, key=lambda x: x[1], reverse=True)
                    return possMatches[0][1]
        elif isinstance(obj, desired_type):
            return obj
        else:
            return default_value



        
    def call_batch(self, batch):
        if batch.shape[0] < 1:
            print("empty batch")
            return None

        # Ensure index is 0-based and add an explicit index column
        batch = batch.reset_index(drop=True)
        batch["row_index"] = batch.index


        joined_texts = "\n".join([
            f"{i}. {text}" for i, text in zip(batch["row_index"], batch[self.source_column_name])
        ])
        prompt = self.prompt.replace(self.pattern, joined_texts)

        try:

            response = self.call_getResponse(prompt)
            output = self.parse_json(self.get_json_string_from_llmresp(response))

            if output is None:
                print("parsing failed")
                batch.loc[0, self.target_column_llmresp] = response
                batch.drop(columns=["row_index"], inplace=True)
                return batch
            
            output_by_index = {}
            for item in output:
                theID = self.fuzz_get(item, "id", str, None)
                if theID is not None:
                    for jtarget in self.target_columns_in_json:
                        item[theID] = self.fuzz_get(item, jtarget, str, None)

            for target_col, json_key in zip(self.target_columns, self.target_columns_in_json):
                batch[target_col] = batch["row_index"].apply(
                    lambda idx: output_by_index.get(idx, {}).get(json_key)
                )
            # Remove the explicit index column
            batch.drop(columns=["row_index"], inplace=True)

            return batch

        except Exception as e:
            print(f"Error in batch processing: {e}")
            return None


        
    def process_df_chunked(self, df):
        if df.shape[0] < 1:
            print("empty batch")
            return None
        
        start = time.time()
        num_batchs = int(np.ceil(df.shape[0] / self.batch_size))


        for i in tqdm(range(num_batchs), desc="Processing Batches", unit="batch"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, df.shape[0])
            batch = df.iloc[start_idx:end_idx].copy()

            batch = self.call_batch(batch)
            if batch is None:
                print(f"Batch {i} processing failed")
                continue

            df.iloc[start_idx:end_idx] = batch

        end = time.time()
        duration = (end - start) * 1000

        return df, duration