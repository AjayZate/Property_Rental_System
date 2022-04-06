import pickle

from flask import Flask, render_template, request

app = Flask(__name__)
DT = pickle.load(open('DT_Rental_System_Model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Location = request.form['Location']
        if Location == 'Whitefield':
            Location = 1
        else:
            Location = 2

        BHK = int(request.form['BHK'])
        Furnishing = int(request.form['Furnishing'])
        Sq_ft = float(request.form['Sq. ft'])
        Old = int(request.form['Old'])
        Floor = int(request.form['Floor'])

        prediction = DT.predict([[Location, BHK, Furnishing, Sq_ft, Old, Floor]])
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'You can rent your Property at Rs.{output} per month')
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
