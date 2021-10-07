from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import model as model

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    score = model.arbol.score(model.X_test,model.y_test)
    print(score)
    print(model.arbol.predict([[False,False,False,True,True,True,True,True,True,True,True,True,True,False,True,False,True,True,True,True,]]))
    return render_template("poll.html")

@app.route("/predict", methods=['POST'])
def predict():
    aux = {}
    for v,e in request.form.items():
        if(e=='False'):
            aux[v]=False
        else:
            aux[v]=True
            
        
    result = model.arbol.predict([[bool(aux['Breathing_Problem']),bool(aux['Fever']),bool(aux['Dry_Cough']),bool(aux['Sore_throat']),bool(aux['Running_Nose']),bool(aux['Asthma']),bool(aux['Chronic_Lung_Disease']),bool(aux['Headache']),bool(aux['Heart_Disease']),bool(aux['Diabetes']),bool(aux['Hyper_Tension']),bool(aux['Fatigue']),bool(aux['Gastrointestinal']),bool(aux['Abroad_travel']),bool(aux['Contact_with_COVID_Patient']),bool(aux['Attended_Large_Gathering']),bool(aux['Visited_Public_Exposed_Places']),bool(aux['Family_working_in_Public_Exposed_Places']),bool(aux['Wearing_Masks']),bool(aux['Sanitization_from_Market'])]])
    return render_template("result.html", result=result[0])

if __name__ == "__main__":
    app.run(debug=True)