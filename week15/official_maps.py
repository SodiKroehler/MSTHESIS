
poli_3_map = {
    "FAR_LEFT": -1,
    "LEFT": -1,
    "LEAN_LEFT": -1,
    "CENTER": 0,
    "LEAN_RIGHT": 1,
    "RIGHT": 1,
    "FAR_RIGHT": 1,
    "UNDEFINED": None
}

poli_leaning_map = {
    "FAR_LEFT": -3,
    "LEFT": -2,
    "LEAN_LEFT": -1,
    "SLIGHT_LEFT": -1,
    "CENTER": 0,
    "LEAN_RIGHT": 1,
    "SLIGHT_RIGHT": 1,
    "RIGHT": 2,
    "FAR_RIGHT": 3,
    "NONE": None,
    "UNDEFINED": None,
    "left": -2,
    "right": 2,
    "center": 0,
    "none": None,
    "global": None,
    '': None
}

f_poli_leaning_map = {
    -3: "FAR_LEFT",
    -2: "LEFT",
    -1: "LEAN_LEFT",
    0: "CENTER",
    1: "LEAN_RIGHT",
    2: "RIGHT",
    3: "FAR_RIGHT",
    None: "UNDEFINED"
}

source_leaning_map = {
    "apnews.com": "LEFT",
    "english.news.cn": "LEFT",
    "msnbc.com": "LEFT",
    "theguardian.com": "LEFT",
    "aljazeera.com": "LEAN_LEFT",
    "npr.org": "LEAN_LEFT",
    "nytimes.com": "LEAN_LEFT",
    "bbc.com": "CENTER",
    "bbc.co.uk": "CENTER",
    "reuters.com": "CENTER",
    "washingtonexaminer.com": "LEAN_RIGHT",
    "dailycaller.com": "RIGHT",
    "foxnews.com": "RIGHT",
    "foxbusiness.com": "RIGHT",  # added for manual pull
}
hue_colors = {
    'LEFT': '#1f3b99',        # dark blue
    'LEAN_LEFT': '#6b8dd6',   # lighter blue
    'CENTER': '#bfbfbf',      # neutral gray
    'LEAN_RIGHT': '#d97b7b',  # lighter red
    'RIGHT': '#8b0000'        # dark red
}

want_to_see_map = {
    "Yes, this seems like the content I usually read": 2,
    "Yes, but this seems to be outside my usual content": 1,
    "No, this is outside my usual content, and I don't like it": -2,
    "No, this seems like the content I usually read, but I don't like this piece": -1,
    "This question is irrelevant here": 0,
    "THE_SAME": 2,
    "SIMILARLY": 1,
    "DIFFERENTLY": -1,
    "WOULD_NOT_HAVE_WRITTEN_ABOUT_IT": 0,
    "VERY_DIFFERENTLY": -2
}

# DO NOT USE - KEEP ALWAYS IN NUM EXCEPT FOR HUMAN CODING
f_want_to_see_map = {
    2: "Yes, this seems like the content I usually read",
    1: "Yes, but this seems to be outside my usual content",
    -1: "No, this seems like the content I usually read, but I don't like this piece",
    -2: "No, this is outside my usual content, and I don't like it",
    0: "This question is irrelevant here"
}

ai_leaning_map = {
    "UNDEFINED": None,
    "NEUTRAL": 0,
    "OPTIMISTIC": 1,
    "PESSIMISTIC": -1,
    "PRO_AI": 1,
    "ANTI_AI": -1
}

f_ai_leaning_map = {
    0: "NEUTRAL",
    1: "OPTIMISTIC",
    -1: "PESSIMISTIC",
    None: "UNDEFINED",
}

aireg_leaning_map = {
    "UNDEFINED": None,
    "NEUTRAL": 0,
    "PRO_REGULATION": 1,
    "ANTI_REGULATION": -1
}

f_aireg_leaning_map = {
    0: "NEUTRAL",
    1: "PRO_REGULATION",
    -1: "ANTI_REGULATION",
    None: "UNDEFINED"
}

imm_leaning_map = {
    "UNDEFINED": None,
    "NEUTRAL": 0,
    "PERMISSIVE": 1,
    "STRICT": -1,
    "PRO_IMMIGRATION": 1,
    "ANTI_IMMIGRATION": -1
}

f_imm_leaning_map = {
    0: "NEUTRAL",
    1: "PERMISSIVE",
    -1: "STRICT",
    None: "UNDEFINED"
}

bias_amount_map = {
    "LOW": 1,
    "NONE": 0,
    "MEDIUM": 2,
    "HIGH": 3,
    "UNDEFINED": None
}

f_bias_amount_map = {
    0: "NONE",
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
    None: "UNDEFINED"
}

rater_map = {
    "sodi.kroehler@gmail.com": "sodi",
    "emmetmathieu@gmail.com": "emmet",
    "sodikroehler@gmail.com": "sodi",
    "emmet.mathieu@gmail.com": "emmet"
}

f_rater_map = {
    "sodi": "sodikroehler@gmail.com",
    "emmet": "emmet.mathieu@gmail.com"
}

pull_article_subject_map = {
    "ai_march": "AI",
    "ai_pass_1": "AI",
    "july_all25_imm": "IMMIGRATION",
    "july_all25_ai": "AI",
    "df6_ai": "AI",
    "df7_imm": "IMMIGRATION",
    "unfiltered_march": "AI",
    "df4": "AI",
    "df5": "AI",
    "df7_ai": "AI",
    "df6_imm": "IMMIGRATION"
}
