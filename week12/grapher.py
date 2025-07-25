import json
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Load 3D matrix
with open("procon_matrix_v1.json", "r") as f:
    matrix = json.load(f)

topic_labels_df = pd.read_csv("topic_labels.csv", dtype={"topic": int, "label": str})
topic_lookup = dict(zip(topic_labels_df["topic"], topic_labels_df["label"]))

df = pd.read_csv("procon_coh_for_mapping.csv")
df.set_index("idx", inplace=True)

entries = []
leaning_offset = -3  # matrix[0] corresponds to leaning = -3

for xi, x_layer in enumerate(matrix):  # x = leaning
    for yi, y_layer in enumerate(x_layer):  # y = topic
        for zi, ids in enumerate(y_layer):  # z = IR
            if ids:
                leaning = xi + leaning_offset
                topic = yi
                ir = zi
                entries.append((leaning, topic, ir, ids))

# App setup
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Ambivalence Mapping of ProCon Data"),
    
    # Top control section: left = controls, right = IR reference
    html.Div([
        # LEFT: Plane and axis controls
        html.Div([
            html.Label("Select Plane:"),
            dcc.Dropdown(
                id="plane",
                options=[
                    {"label": "Leaning–Topic", "value": "leaning-topic"},
                    {"label": "Leaning–IR", "value": "leaning-ir"},
                    {"label": "Topic–IR", "value": "topic-ir"},
                ],
                value="leaning-topic"
            ),
            html.Br(),
            html.Label("Select Fixed Axis Value:"),
            dcc.Dropdown(id="fixed", value=0)
        ], style={"width": "80%", "display": "inline-block", "verticalAlign": "top", "padding": "1rem"}),

        # RIGHT: IR reference
        html.Div([
            html.Table([
                html.Tr([html.Th("IR"), html.Th("Interpretive Repertoire")]),
                html.Tr([html.Td("0"), html.Td("Inspired")]),
                html.Tr([html.Td("1"), html.Td("Popular")]),
                html.Tr([html.Td("2"), html.Td("Moral")]),
                html.Tr([html.Td("3"), html.Td("Civic")]),
                html.Tr([html.Td("4"), html.Td("Economic")]),
                html.Tr([html.Td("5"), html.Td("Functional")]),
                html.Tr([html.Td("6"), html.Td("Ecological")]),
            ])
        ], style={"width": "15%", "display": "inline-block", "verticalAlign": "top", "padding": "1rem", "borderLeft": "1px solid #ccc"})
    ]),
    
    # Main graph
    dcc.Graph(id="scatter"),
    
    # Text output section
    html.Div(id="text-output", style={"whiteSpace": "pre-wrap", "marginTop": "20px", "padding": "1rem"})
])


@app.callback(
    Output("scatter", "figure"),
    Input("plane", "value"),
    Input("fixed", "value")
)
def update_graph(plane, fixed_val):
    points = []
    for leaning, topic, ir, ids in entries:
        if plane == "leaning-topic" and ir == fixed_val:
            points.append((leaning, topic, ids))
        elif plane == "leaning-ir" and topic == fixed_val:
            points.append((leaning, ir, ids))
        elif plane == "topic-ir" and leaning == fixed_val:
            points.append((topic, ir, ids))

    if not points:
        return px.scatter(x=[], y=[], title="No data")

    i_vals, j_vals, ids_list = zip(*points)
    sizes = [len(ids) for ids in ids_list]
    hover_texts = [", ".join(map(str, ids)) for ids in ids_list]

    axis_labels = {
        "leaning-topic": {"x": "Leaning", "y": "Topic"},
        "leaning-ir": {"x": "Leaning", "y": "IR"},
        "topic-ir": {"x": "Topic", "y": "IR"}
    }

    fig = px.scatter(
        x=i_vals,
        y=j_vals,
        size=sizes,
        hover_name=hover_texts,
        labels=axis_labels[plane],
        title=f"{plane.replace('-', ' vs. ').title()} at {fixed_val}"
    )
    fig.update_traces(customdata=ids_list)
    return fig

@app.callback(
    Output("fixed", "options"),
    Input("plane", "value")
)
def update_fixed_options(plane):
    if plane == "leaning-topic":
        return [{"label": f"IR {i}", "value": i} for i in range(7)]
    
    elif plane == "leaning-ir":
        return [
            {
                "label": f"Topic {i} – {topic_lookup.get(i, 'Unknown')}",
                "value": i
            }
            for i in range(97)
        ]

    elif plane == "topic-ir":
        return [{"label": f"Leaning {i}", "value": i} for i in range(-4, 5)]



@app.callback(
    Output("text-output", "children"),
    Input("scatter", "clickData")
)
def show_text(clickData):
    if not clickData:
        return "Click a point to view the text snippets."

    ids = clickData["points"][0]["customdata"]
    if isinstance(ids, int):
        ids = [ids]

    output_chunks = []
    for i in ids:
        if i in df.index:
            output_chunks.append(f"[ID {i}] — {df.loc[i, 'clean']}")
            # output_chunks.append(df.loc[i, "clean"])
        else:
            output_chunks.append(f"[ID {i}] — no matching text found")

    return "\n\n---\n\n".join(output_chunks)


app.run(debug=True)
