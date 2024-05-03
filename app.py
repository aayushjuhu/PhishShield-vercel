# importing required libraries

from feature import FeatureExtraction
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings('ignore')

file = open("pickle/psodt.pkl", "rb")
dtc = pickle.load(file)
file.close()


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)
        feature_print = obj.getPrint()
        y_pred = dtc.predict(x)[0]

        # 1 is safe
        # -1 is unsafe
        y_pro_phishing = dtc.predict_proba(x)[0, 0]
        y_pro_non_phishing = dtc.predict_proba(x)[0, 1]
        # if(y_pred ==1 ):
        # pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index1.html', xx=round(y_pro_non_phishing, 2), url=url, feature_print=feature_print)
    return render_template("index1.html", xx=-1)


if __name__ == "__main__":
    app.run(debug=True)
