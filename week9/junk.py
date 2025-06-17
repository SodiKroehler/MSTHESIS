from sodigpt import sodiBatchDetails, sodiGPT
import json
import pandas as pd


y_labels = {
    1: "Political Factors and Implications",
    2: "Public Sentiment",
    3: "Cultural Identity",
    4: "Morality and Ethics",
    5: "Fairness and Equality",
    6: "Legality, Constitutionality, Jurisdiction",
    7: "Crime and Punishment",
    8: "Security and Defense",
    9: "Health and Safety",
    10: "Quality of Life",
    11: "Economics",
    12: "Capacity and Resources",
    13: "Policy Description, Prescription, Evaluation",
    14: "External Regulation and Reputation",
    15: "Other"
}

class sodiGPT:
    def __init__(self, system_prompt = "You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.temperature = 0

    def call_getResponse(self, prompt):
        # Placeholder implementation for demonstration
        return f"Response to: {prompt}"

y_labels__string = json.dumps(y_labels, indent=4)

harvard_df = pd.read_csv('final_pca_and_gpt_framing.csv')
grouped_by_frame = harvard_df.groupby('frame').apply(lambda x: x.sample(n=2, random_state=42)).reset_index(drop=True)
grouped_by_frame_json = grouped_by_frame.to_json(orient='records', indent=4)

prompt = "what color is the sky"
sodiGPT2 = sodiGPT()
resp = sodiGPT2.call_getResponse(prompt)


    batch[self.batch_details.target_columns] = batch.apply(
        lambda row: self.batch_details.apply_func(row, row.name, *self.batch_details.apply_func_args), axis=1
    )




            
    def call_batch(self, batch):
        joined_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(batch[self.batch_details.source_column_name])])
        prompt = self.batch_details["prompt"].replace(self.batch_details["pattern"], joined_texts)
        if batch.shape[0] < 1:
            print("empty batch")
            return None
        try:
            response = self.call_getResponse(prompt)
            output = self.parse_json(self.get_json_string_from_llmresp(response))
            if output is None:
                print("parsing failed")
                batch.loc[0, self.batch_details.target_column_llmresp] = response
                return batch
            else:
                batch[self.batch_details.target_columns] = batch.apply(self.batch_details.apply_func, axis=1)

            return batch
        
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return None
        




        grouped_by_frame.shape
el_string = grouped_by_frame.to_json(orient='records', indent=4)

# system_prompt = f"""
# You are a PhD-level expert in political communication and media framing. 
# Your task is to classify the dominant frame expressed in a piece of media text 
# using the following 15-frame schema derived from Card et al. (2022) and Boydstun et al. (2014). 
# These frames represent common thematic lenses used in U.S. political discourse. 
# You must assign a single dominant frame_code (1–15) for each text, and you may optionally 
# propose a new_frame_code and brief justification if the standard set does not fully capture the meaning.

# ## Standard Media Frames and Codes
# {y_labels__string}

# ## Framing Guidelines
# Choose the dominant frame that best captures the main lens through which the issue is presented.

# Ignore superficial keywords; instead, identify the core logic, emphasis, or metaphor used to frame the topic.

# If no existing frame captures the framing well, assign 15 and propose a new_frame_code label with rationale.


# ## Few-Shot Examples

# Here are some texts which have been coded by human coders. Use them as reference material.

# {el_string}



# Enclose the JSON in three backticks, like this:
# ```
# [\{"id": 40, "frame": POLITICAL\}, ...]
# ```
# """




system_prompt = (
    "You are a PhD-level expert in political communication and media framing.\n"
    "Your task is to classify the **dominant frame** expressed in each piece of media text, using the 15-frame schema derived from Card et al. (2022) and Boydstun et al. (2014).\n\n"
    "These frames represent common thematic lenses used in U.S. political discourse.\n\n"
    "---\n\n"
    "## Standard Media Frames and Codes\n\n"
    "Each frame has a numeric frame_code (1–15) and a label. Choose only one dominant frame per text.\n\n"
    f"{y_labels__string}\n\n"
    "---\n\n"
    "## Framing Guidelines\n\n"
    "- Assign the **dominant frame** that best captures the main **logic**, **emphasis**, or **metaphor** of the piece.\n"
    "- Do **not** rely on surface keywords. Focus on the **underlying argument** or lens.\n"
    "- If no frame adequately captures the text, assign frame_code `15` (\"Other\") and propose:\n"
    "  - `new_frame_code`: a descriptive label\n"
    "  - `rationale`: 1–2 sentence explanation\n\n"
    "---\n\n"
    "## Few-Shot Examples\n\n"
    "Use these human-coded examples to guide your classification:\n\n"
    f"{el_string}\n\n"
    "---\n\n"
    "## Output Format\n\n"
    "Return results in **valid JSON**, as a list of objects—one per text. Each object must include:\n"
    "- `id`: the numeric identifier of the text\n"
    "- `frame`: the most appropriate standard frame label from the list\n\n"
    "Return your result as a JSON array enclosed in three backticks, one object per text, like this:\n"
    "``` \n"
    "[\\{\"id\": 1, \"frame\": ECONOMIC\\}, ...]\n"
    "```\n\n"
    "### Text to analyze:\n"
)




# system_prompt = f"""
# You are a PhD-level expert in political communication and media framing. 
# Your task is to classify the dominant frame expressed in a piece of media text 
# using the following 15-frame schema derived from Card et al. (2022) and Boydstun et al. (2014). 
# These frames represent common thematic lenses used in U.S. political discourse. 
# You must assign a single dominant frame_code (1–15) for each text, and you may optionally 
# propose a new_frame_code and brief justification if the standard set does not fully capture the meaning.

# ## Standard Media Frames and Codes
# {y_labels__string}

# ## Framing Guidelines
# Choose the dominant frame that best captures the main lens through which the issue is presented.

# Ignore superficial keywords; instead, identify the core logic, emphasis, or metaphor used to frame the topic.

# If no existing frame captures the framing well, assign 15 and propose a new_frame_code label with rationale.


# ## Few-Shot Examples

# Here are some texts which have been coded by human coders. Use them as reference material.

# {el_string}



# Enclose the JSON in three backticks, like this:
# ```
# [\{"id": 40, "frame": POLITICAL\}, ...]
# ```
# """



# system_prompt = f"""
# You are a PhD-level expert in political communication and media framing.
# Your task is to classify the **dominant frame** expressed in each piece of media text, using the 15-frame schema derived from Card et al. (2022) and Boydstun et al. (2014).

# These frames represent common thematic lenses used in U.S. political discourse.

# ---

# ## Standard Media Frames and Codes

# Each frame has a numeric frame_code (1–15) and a label. Choose only one dominant frame per text.

# {y_labels__string}

# ---

# ## Framing Guidelines

# - Assign the **dominant frame** that best captures the main **logic**, **emphasis**, or **metaphor** of the piece.
# - Do **not** rely on surface keywords. Focus on the **underlying argument** or lens.
# - If no frame adequately captures the text, assign frame_code `15` ("Other") and propose:
#   - `new_frame_code`: a descriptive label
#   - `rationale`: 1–2 sentence explanation

# ---

# ## Few-Shot Examples

# Use these human-coded examples to guide your classification:

# {el_string}

# ---

# ## Output Format

# Return results in **valid JSON**, as a list of objects—one per text. Each object must include:
# - `id`: the numeric identifier of the text
# - `frame`: the most appropriate standard frame label from the list


# Return your result as a JSON array enclosed in three backticks, one object per text, like this:
# ``` 
# [\{"id": 1, "frame": ECONOMIC\}, ...]
# ```


# ### Text to analyze:
# """


req_token_count = prompt_tokens = self.count_tokens(f"\n\n{self.system_prompt} \n\n {self.prompt}")
    if req_token_count > self.tpm_limit:
        throw ValueError(f"Prompt token count exceeds limit: {req_token_count} > {self.tpm_limit}")



            def wait_if_needed(self, new_tokens):
        now = time.time()
        elapsed = now - self.window_start

        if elapsed > 60:
            self.resets_since_calling += 1
            self.reset_window()

        if self.used_tokens + new_tokens > self.tpm_limit:
            sleep_time = 60 - elapsed
            print(f"⏳ Throttling... sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
            self.reset_window()
        
        self.used_tokens += new_tokens


    import time

    def get_json_string_from_llmresp(self, llm_response, delimiter=None):
        if not delimiter:
            delimiter = self.json_delimiter

        if not llm_response:
            raise ValueError("No llm response in json parser")  
            
        start, end = 0, len(llm_response)

        delim_match = list(re.finditer(delimiter, llm_response))
        if len(delim_match) >= 2:
            start = delim_match[0].start()
            end = delim_match[-1].end()

            # Check for the word "json" right after the first delimiter and remove it if it exists
            after_first_delim = llm_response[start + len(delimiter):].strip()
            if after_first_delim.lower().startswith("json"):
                start += len(delimiter) + len("json")
        else:
            print(f"Not enough occurrences of delimiter '{delimiter}' in the response.")

        return llm_response[start:end].strip()



  def get_json_string_from_llmresp(self, llm_response, delimiter=None):
        if not delimiter:
            delimiter = self.json_delimiter

        if not llm_response:
            raise ValueError("No llm response in json parser")  
            
        start, end = 0, len(llm_response)

        delim_match = list(re.finditer(delimiter, llm_response))
        if len(delim_match) >= 2:
            start = delim_match[0].start()
            end = delim_match[-1].end()
        else:
            print(f"Not enough occurrences of delimiter '{delimiter}' in the response.")

        
        return llm_response[start:end].strip()


        def parse_json(self, llm_response):
            try:
                lint = json.loads(llm_response)
                return lint
            except json.JSONDecodeError as e:
                print(f"JSON parsing error at line {e.lineno}, column {e.colno}: {e.msg}")
                print(f"Failed JSON snippet: {llm_response[e.pos-50:e.pos+50]}")
                return None
            except Exception as e:
                print(f"Unexpected error in parsing JSON: {e}")
                return None
            


              def process_df_chunked(self, df):
        if df.shape[0] < 1:
            print("empty batch")
            return None
        
        start = time.time()
        num_batchs = int(np.ceil(df.shape[0] / self.batch_size))

        self.reset_window()
        for i in tqdm(range(num_batchs), desc="Processing Batches", unit="batch"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, df.shape[0])
            batch = df.iloc[start_idx:end_idx].copy()

            print(f"starting batch between idx {start_idx} and {end_idx}")

            batch = self.call_batch(batch)
            if batch is None:
                print(f"Batch {i} processing failed")
                continue

            # Update the original DataFrame with the modified batch
            for col in batch.columns:
                df.loc[start_idx:end_idx-1, col] = batch[col].values

        end = time.time()
        duration = (end - start) * 1000

        return df, duration


        gdf['frame_code'] = gdf.apply(lambda row: get_frame_code_from_frame(row['gpt_frame']), axis=1)

        gdf['gpt_frame'] = gdf['gpt_frame'].str.upper()


for col in batch.columns:
    if col not in df.columns:
        default_value = batch[col].dtype.type() if pd.api.types.is_numeric_dtype(batch[col]) else ''
        df[col] = default_value

    df.iloc[start_idx:end_idx, df.columns.get_loc(col)] = batch[col].values