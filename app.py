from flask import Flask, render_template, request, redirect, session, flash, jsonify
import json, datetime, asyncio, os, shutil, threading
from twitter_analysis_function import run_twitter_analysis, run_analysis_check
import pyrebase


# database config 
firebase_config = {
    "apiKey": "AIzaSyAFZSLIhF0wZq3ZDTwM2ZyDosCmHbx_W_Y",
    "authDomain": "twitter-analysis-f5677.firebaseapp.com",
    "databaseURL": "https://twitter-analysis-f5677-default-rtdb.firebaseio.com",
    "projectId": "twitter-analysis-f5677",
    "storageBucket": "twitter-analysis-f5677.appspot.com",
    "messagingSenderId": "551243643778",
    "appId": "1:551243643778:web:7537e1177d9dc54fd42d9b"
}
firebase = pyrebase.initialize_app(firebase_config)
fire_db = firebase.database()


app = Flask(__name__)
app.secret_key = 'twitter'

data_folder = os.path.abspath('static/assets/images')
most_eng_html = os.path.abspath("templates/most_engagement.html")
tasks = []

@app.route("/")
def homepage():
    email = session.get('email')
    if email != None:
        return redirect("/search_data")
    else:
        return render_template("registration.html")

# add database so that it'll be secure to maintain users
@app.route("/register", methods=["POST"])
def register_user():
    if request.method == 'POST':
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_pass = request.form.get("confirm_password")

        if password != confirm_pass:
            return redirect("/")
        else:
            user = {
                "name": name,
                "email": email,
                "password": password
            }

            fire_db.child("users").push(user)
            
            session['email'] = email
            return redirect("/search_data")

@app.route("/login", methods=["POST"])
def login():
    
    if request.method == 'POST':
        email =  request.form.get("login_email")
        password = request.form.get("login_password")

        db_user = fire_db.child("users").order_by_child("email").equal_to(email).get()
        if len(db_user.val()) == 0:
            return redirect('/')
        else:
            for key, val in db_user.val().items():
                if val['password'] == password :
                    session['email'] = email
                    return redirect('/search_data')
                else:
                    return redirect("/")

        return redirect("/") 


@app.route("/logout")
def logout():
    session.pop('email')
    return redirect("/")

@app.route("/search_data", methods=["GET", "POST"])
def make_search():

    if request.method == 'GET':
        email = session.get('email')
        if email != None:

            for filename in os.listdir(data_folder):

                if not (filename == 'close.png' or filename == 'company-brand-white.svg' or filename == 'company_brand.png') :
                    file_path = os.path.join(data_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))

            with open(f"{most_eng_html}", 'w') as f:
                f.write("")

            return render_template("search_input.html")
        else:
            return redirect('/')

    elif request.method == 'POST':
        if session.get('email') != None:
            keywords = request.form.get("keyword")
            location = request.form.get("location")
            location = location.lower()
            date = request.form.get("date")

            if date != '':

                current_date = datetime.datetime.now()
                selected_date = datetime.datetime.strptime(date, "%Y-%m-%d")

                duration = current_date - selected_date

                if duration.days <= 7:

                    thread_task = threading.Thread(target = run_twitter_analysis, args=[keywords, location, selected_date])
                    thread_task.start()
                    tasks.append(thread_task)

                    return redirect("/running")
                      
                else:
                    flash("Oops, please adjust the date as mentioned below and retry")
            else:
                flash("Date cannot be Empty..!")

            return redirect("/search_data")


@app.route("/checking")
def checking():

    obj = {
        "status": "processing"
    }
    print(obj)

    files = os.listdir(data_folder)

    if len(files) == 18:
        obj['status'] = "completed"
        return jsonify(obj)

    elif len(files) < 18:
        for i in files:
            if os.path.basename(i).split(".")[0].strip() == 'error' :
                obj['status'] = 'error'
                flash("Error occured")
        
        return jsonify(obj)

    else:
        return jsonify(obj)


@app.route("/running")
def running():
    return render_template("running.html")

@app.route("/results")
def results():
    email = session.get('email')
    if email != None:
        return render_template("most_engagement.html")
    else:
        return redirect("/")

if __name__ == '__main__':
    app.run(debug=True)