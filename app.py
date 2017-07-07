from flask import Flask, render_template,request
import numpy as np
import keras.models, sys, os, json

sys.path.append(os.path.abspath("./model"))

from load import * 

app = Flask(__name__)

global model, graph

model, graph = init()

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	
	rawStats = request.get_data()

	json_arr = json.loads(rawStats)

	stats = np.asarray(json_arr)

	flat = np.empty([1,7])

	for x in xrange(0,6):
		flat.itemset(x,stats.item(x))

	with graph.as_default():
		out = model.predict(flat)
		print(out)
		print(np.argmax(out,axis=1))
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 9990))
	app.run(host='0.0.0.0', port=port)
	