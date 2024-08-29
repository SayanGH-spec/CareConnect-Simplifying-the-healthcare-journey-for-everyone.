from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = "Sayan_02"

class HealthcareManagementSystem:
    def __init__(self):
        self.hospitals = None
        self.patients = None
        self.model = None
        self.appointments = None
        self.cached_recommendations = {}

    def load_data(self):
        self.load_hospitals()
        self.load_patients()
        self.load_appointments()

    def load_hospitals(self):
        self.hospitals = pd.read_csv('data/hospital_data.csv')

    def load_patients(self):
        self.patients = pd.read_csv('data/patient_data.csv')

    def load_appointments(self):
        self.appointments = pd.read_csv('data/appointments.csv')

    def train_model(self):
        X = self.hospitals[['Availability_Beds_Cardiology', 'Availability_Beds_Neurology', 'Availability_Beds_Orthopedics', 'Location']]
        X = pd.get_dummies(X, columns=['Location'], drop_first=True)  # One-hot encoding for location
        y = self.hospitals['Specialty']
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

    def predict_specialty(self, patient_features, selected_specialty):
        # Unpack patient_features
        age, gender, location = patient_features
        
        # Encode gender
        gender_encoded = 1 if gender == 'Male' else 0

        # One-hot encode location
        location_encoded = 1 if location == 'Kolkata' else 0
        
        # Create a list with 46 features, setting the first three features as age, gender, and location, and the rest as dummy values
        patient_features_encoded = [age, gender_encoded, location_encoded] + [0] * 42
        
        # Predict the specialty using the trained model
        predicted_specialty = self.model.predict([patient_features_encoded])[0]

        # Check if the predicted specialty matches the selected specialty, if not, use the selected specialty
        if predicted_specialty != selected_specialty:
            predicted_specialty = selected_specialty

        return predicted_specialty

    def book_ambulance(self, location):
        print(f"Ambulance booked to {location}")

    def recommend_hospitals(self, location, specialty):
        cache_key = (location, specialty)
        if cache_key in self.cached_recommendations:
            return self.cached_recommendations[cache_key]
        else:
            filtered_hospitals = self.hospitals[(self.hospitals['Location'] == location) & (self.hospitals['Specialty'] == specialty)]

            if not filtered_hospitals.empty:
                recommendations = filtered_hospitals.sample(n=min(5, len(filtered_hospitals)))
            else:
                specialty_hospitals = self.hospitals[self.hospitals['Specialty'] == specialty]
                if not specialty_hospitals.empty:
                    recommendations = specialty_hospitals.sample(n=min(5, len(specialty_hospitals)))
                else:
                    recommendations = self.hospitals.sample(n=min(5, len(self.hospitals)))

            self.cached_recommendations[cache_key] = recommendations
            return recommendations

    def check_bed_availability(self, hospital_name, specialty):
        hospital = self.hospitals[(self.hospitals['Hospital_Name'] == hospital_name) & (self.hospitals['Specialty'] == specialty)]
        if not hospital.empty:
            return True if hospital[f'Availability_Beds_{specialty}'].iloc[0] > 0 else False
        else:
            return False

    def register_patient(self, patient_data):
        with open('data/patient_data.csv', 'a') as f:
            f.write('\n' + ','.join(map(str, patient_data.values())))

    def schedule_appointment(self, appointment_data):
        with open('data/appointments.csv', 'a') as f:
            f.write('\n' + ','.join(map(str, appointment_data.values())))

    def verify_credentials(self, username, password):
        patient_name = self.patients['Patient_Name']
        if username in patient_name.tolist():
            # Get the index of the patient
            idx = patient_name[patient_name == username].index[0]
            # Get the corresponding password which is the patient name
            if self.patients.iloc[idx]['Patient_Name'] == password:
                return True
        return False

    def get_hospitals(self):
        return self.hospitals.to_dict(orient='records')

    def get_specialties(self):
        return self.hospitals['Specialty'].unique().tolist()

hms = HealthcareManagementSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        patient_data = {
            'Patient_ID': hms.patients['Patient_ID'].max() + 1,
            'Patient_Name': request.form['patient_name'],
            'Gender': request.form['gender'],
            'Age': int(request.form['age']),
            'Location': request.form['location'],
        }
        hms.register_patient(patient_data)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.pop('user', None)
        username = request.form['username']
        password = request.form['password']
        if hms.verify_credentials(username, password):
            session['user'] = username
            return redirect(url_for('appointments'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/appointments', methods=['GET', 'POST'])
def appointments():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        patient_location = hms.patients.loc[hms.patients['Patient_Name'] == session['user'], 'Location'].iloc[0]
        selected_specialty = request.form['specialty']
        appointment_data = {
            'Patient_Name': session['user'],
            'Gender': hms.patients.loc[hms.patients['Patient_Name'] == session['user'], 'Gender'].iloc[0],
            'Age': hms.patients.loc[hms.patients['Patient_Name'] == session['user'], 'Age'].iloc[0],
            'Location': patient_location,
            'Appointment_Date': datetime.strptime(request.form['appointment_date'], '%Y-%m-%d').date(),
            'Hospital_Name': request.form['hospital_name'],
            'Specialty': selected_specialty
        }
        if not hms.check_bed_availability(request.form['hospital_name'], selected_specialty):
            return render_template('error.html', message="No specalist doctor available in this hospital as your requirments")
        
        # Get patient features for prediction
        patient_features = [
            int(hms.patients.loc[hms.patients['Patient_Name'] == session['user'], 'Age'].iloc[0]),
            request.form['gender'],
            patient_location
        ]
        
        # Predict the specialty
        predicted_specialty = hms.predict_specialty(patient_features, selected_specialty)
        
        # Get recommended hospitals based on predicted specialty
        recommended_hospitals = hms.recommend_hospitals(patient_location, predicted_specialty)
        
        # Extract hospital names from recommended_hospitals DataFrame
        hospital_names = recommended_hospitals['Hospital_Name'].tolist()
        
        # Schedule the appointment
        hms.schedule_appointment(appointment_data)
        
        # Book the ambulance
        hms.book_ambulance(patient_location)
        
        # Retrieve the number of available beds
        available_beds = hms.hospitals.loc[(hms.hospitals['Hospital_Name'] == request.form['hospital_name']) & (hms.hospitals['Specialty'] == selected_specialty), f'Availability_Beds_{selected_specialty}'].iloc[0]
        
        return render_template('success.html', predicted_specialty=predicted_specialty, recommended_hospitals=hospital_names, available_beds=available_beds, hospital_name=request.form['hospital_name'])

    # Fetching hospital data and specialties to render the page
    hospital_data = hms.get_hospitals()
    specialties = hms.get_specialties()
    
    return render_template('appointments.html', hospital_data=hospital_data, specialties=specialties)

@app.route('/emergency', methods=['GET', 'POST'])
def emergency():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        patient_location = hms.patients.loc[hms.patients['Patient_Name'] == session['user'], 'Location'].iloc[0]
        selected_specialty = request.form['specialty']
        
        # Find hospitals with available beds for the selected specialty in the patient's location
        available_hospitals = hms.hospitals[(hms.hospitals['Location'] == patient_location) & (hms.hospitals['Specialty'] == selected_specialty) & (hms.hospitals[f'Availability_Beds_{selected_specialty}'] > 0)]
        
        if available_hospitals.empty:
            return render_template('error.html', message="No hospitals available for emergency in your area")
        
        # Book the ambulance
        hms.book_ambulance(patient_location)
        
        # Redirect to success page
        return render_template('emergency_success.html', available_hospitals=available_hospitals)

    # Fetching specialties to render the page
    specialties = hms.get_specialties()
    
    return render_template('emergency.html', specialties=specialties)

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        location = hms.patients.loc[hms.patients['Patient_Name'] == session['user'], 'Location'].iloc[0]
        specialty = request.form['specialty']
        recommended_hospitals = hms.recommend_hospitals(location, specialty)
        return render_template('recommendations.html', recommended_hospitals=recommended_hospitals)
    return render_template('recommendations.html', recommended_hospitals=None)

@app.route('/api/hospitals')
def api_get_hospitals():
    hospitals = hms.get_hospitals()
    return jsonify(hospitals)

@app.route('/api/specialties')
def api_get_specialties():
    specialties = hms.get_specialties()
    return jsonify(specialties)

if __name__ == '__main__':
    hms.load_data()
    hms.train_model()
    app.run(debug=True)
