import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pickled model
model = pickle.load(open("diabetic.pkl", "rb"))

# Function to make predictions
def predict_diabetes(data):
    # Standardize the input data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Make predictions
    prediction = model.predict(data_scaled)
    return prediction

# Streamlit web app
def main():
    st.title("Diabetes Prediction Web App")

    # Create input fields for user to input data
    pregnancies = st.slider("Number of Pregnancies", 0, 17, 1)
    glucose = st.slider("Plasma Glucose Concentration", 0, 199, 117)
    blood_pressure = st.slider("Blood Pressure", 0, 122, 72)
    skin_thickness = st.slider("Skin Thickness", 0, 99, 23)
    insulin = st.slider("Insulin Level", 0, 846, 30)
    bmi = st.slider("BMI", 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
    age = st.slider("Age", 21, 81, 29)

    # Collect user input into a DataFrame
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    # Display the user input
    st.subheader("User Input:")
    st.write(user_data)

    # Make predictions
    if st.button("Predict"):
        prediction = predict_diabetes(user_data)
        if prediction[0] == 0:
            st.success("No Diabetes (Outcome 0)")
        else:
            st.error("Diabetes Detected (Outcome 1)")

if __name__ == "__main__":
    main()
