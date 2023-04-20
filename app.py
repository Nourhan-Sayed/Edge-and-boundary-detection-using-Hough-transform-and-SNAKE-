from flask import Flask, render_template,request, jsonify, redirect, session
import canny
import hough
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import helpers
from Utilities import *
from snakes_show import *
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'super secret key'

chainCode=[]
additionalData=""
@app.route("/",methods=['GET','POST'])
def index():
    # if session.get("mode") is None:
    #     session['mode'] = "canny"
    #     session['shape'] = "lines"
    print("MODE: ", session["mode"])
    print("SHAPE: ", session["shape"])
    if session.get("path") is not None:
        uploaded_path = session['path']
    else:
        uploaded_path = "static/images/Chess_Board.png"
    uploaded_path = uploaded_path.replace('../','')
    if session["mode"]=="canny":
        img = cv2.imread(uploaded_path,0)
        if session["shape"] =="lines":
            img = helpers.resize(img,1)
            hough.hough_lines(img)
            output_img="../static/images/output.jpg"
        if session["shape"] =="circles":
            img = cv2.imread(uploaded_path)
            img = helpers.resize(img,0)
            img_path = "static/images/resized.jpg"
            output_path = hough.hough_circles(img_path, 1, 200, 10, 100, 0.4)
        # canny.canny_edge_detector(img,20,40)
    else:
        img = cv2.imread(uploaded_path)
        print("starting contour")
        points,area,perimeter=snakes(img,int(session["contourData"]["alpha"]),int(session["contourData"]["beta"]),int(session["contourData"]["gamma"]) )
        print("done contour")
        global chainCode
        chainCode=chain_code(points)
        global additionalData
        additionalData="Area of contour : "+str(area)+" perimeter : "+str(perimeter)
    print("UPLOADED PATH: ", uploaded_path)
    print(additionalData)
    return render_template('index.html', area = additionalData ,uploaded_path = uploaded_path)

@app.route("/contourData",methods=['GET','POST'])
def contourData():
    print("enter data")
    print("from contour Data",additionalData)
    print("chain Code : ",chainCode)
    return jsonify({"area":additionalData,"chainCode":chainCode})

@app.route("/sendData",methods=['GET','POST'])
def send_data():
    if request.method =='POST':
        if 'file1' in request.files:
            file1 = request.files['file1']
            path = os.path.join("../static/images/", file1.filename)
            session['path'] = path
        try:
            mode = request.get_json("mode")
            session["mode"]= mode["mode"]
            shape = request.get_json("shape")
            session["shape"]= shape["shape"]
            if session["mode"]!="canny":
                session["contourData"]=request.get_json("alpha")
                print(session["contourData"])
        except:
            session["mode"]="canny"
            session["shape"]="lines"

    return redirect("/")

if __name__ == '__main__':

    app.run(debug=True)