from flask import Flask, render_template
import json
import plotly
from pca import fig

app = Flask(__name__)

g = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route("/")
def index():
    return render_template("index.html", title = "Home", graphJSON = g)


if __name__ == "__main__":
    app.run(debug=True)
