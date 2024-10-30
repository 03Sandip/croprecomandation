
from flask import Flask,request,jsonify
import pickle
import numpy as np


model=pickle.load(open('cropmodel.pkl','rb'))
app =Flask(__name__)

@app.route('/')
def home():
    return"Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    N = request.form.get('N')
    P = request.form.get('P')
    K = request.form.get('K')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')

    '''recommendation= {'P':P,'N':N,'K':K,'temperature':temperature,'humidity':humidity,'ph':ph,'rainfall':rainfall}'''
    input_query = np.array([[N,P,K,temperature,humidity,ph,rainfall]])

    recommendation = model.predict(input_query)[0]

    return jsonify(recommendation)


if __name__=='__main__':
    app.run(debug=True)




