
# def gpt_call(pregnant_prompt):
    
#     client = OpenAI(api_key=OPENAI_API_KEY)
#     content, error = None, None
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a journalist who analyzes frames in media articles. You should answer in json only, using the following labels: " + json.dumps(y_labels_semantic_order)},
#                 {"role": "user", "content": pregnant_prompt}],
#             max_tokens=1000,
#             temperature=0
#         )

#         content = response.choices[0].message.content.strip()
#     except Exception as e:
#         error = f"Error in calling gpt: {e}"
#     return content, error




# for x, item in idf.iterrows():
#     x = item['leaning']
#     y = item['frame']
#     label = y_labels[int(y)]
#     ax.scatter(x, y, color='blue')
#     # ax.text(x + 0.1, y, label, verticalalignment='center', fontsize=9)
#     ax.text(x + 0.1, y, f"{x:.2f}", va='center', fontsize=9)
# texts = {
#     1: "The president thinks that democrats are evil and stupid",
#     2: "The president thinks that republicans are evil and stupid",
#     3: "The president thinks that independents are evil and stupid",
#     4:"The president thinks that the media is evil and stupid",
#     5: "The president thinks that the government is evil and stupid",
# }
# raw_resp = gpt_call(
#     "For each of the following texts, output json of the id of the text and the framing
#     "of the text, like this: \\`[{text1: 2.0},{text1: 2.0} ]\\`" + json.dumps(texts))
# print(raw_resp[0])
# cleaned = raw_resp[0].strip('`').split('\n', 1)[-1].rsplit('\n', 1)[0]
# json = json.loads(cleaned[0])
# print(json)




# frames_dict = {}

# with open('./media_frames_corpus/annotations/codes.json') as f:
#     frames_dict = json.load(f)
# frames = {v.upper(): float(k) for k, v in frames_dict.items()}