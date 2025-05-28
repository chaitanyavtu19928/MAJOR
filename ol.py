import json
from PIL import Image
import io
import os
import numpy as np
import streamlit as st
from streamlit import session_state
from tensorflow.keras.models import load_model
from keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input
import base64
import tensorflow as tf
from tensorflow import keras

# Initialize session state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0

selected_model = "EfficientNetB0"
model = keras.models.load_model("models/EfficientNetB0_model.h5")

def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Account created successfully! You can now login.")
            else:
                st.error("Passwords do not match. Please try again.")

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                render_dashboard(user)
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None

def predict(image_path, model):
    img = img_preprocessing.load_img(image_path, target_size=(224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    
    # Define classes and their corresponding severity levels
    classes = {
        'Age Degeneration': 'Moderate',
        'Cataract': 'Moderate',
        'Diabetes': 'Severe',
        'Glaucoma': 'Severe',
        'Hypertension': 'Moderate',
        'Myopia': 'Mild',
        'Normal': 'None',
        'Others': 'Variable'
    }
    
    predicted_class = list(classes.keys())[np.argmax(predictions)]  # Get the condition name
    severity_level = classes[predicted_class]  # Get severity level based on predicted class

    return predicted_class, severity_level

def generate_medical_report(predicted_label):
    # Define class labels and corresponding medical information
    medical_info = {
        "Age Degeneration": {
            "report": "The patient appears to have age-related degeneration. Further evaluation and management are recommended to prevent vision loss.",
            "preventative_measures": [
                "Regular eye exams are crucial for early detection and intervention",
                "Maintain a healthy lifestyle with a balanced diet and regular exercise",
                "Protect eyes from UV rays with sunglasses when outdoors",
            ],
            "precautionary_measures": [
                "Schedule regular follow-ups with an eye specialist",
                "Consider supplements recommended by your doctor to support eye health",
            ],
        },
        "Cataract": {
            "report": "It seems like the patient has cataracts. While common and treatable, it's important to address symptoms and consider treatment options.",
            "preventative_measures": [
                "Protect eyes from UV exposure with sunglasses",
                "Quit smoking if applicable, as it can increase cataract risk",
                "Maintain overall health with a balanced diet and regular exercise",
            ],
            "precautionary_measures": [
                "Consult with an eye specialist for personalized treatment options",
                "Discuss surgical options if cataracts significantly affect daily activities",
            ],
        },
        "Diabetes": {
            "report": "The patient appears to have diabetes. It's crucial to manage blood sugar levels effectively to prevent complications, including diabetic retinopathy.",
            "preventative_measures": [
                "Monitor blood sugar levels regularly as advised by your doctor",
                "Follow a diabetic-friendly diet rich in fruits, vegetables, and whole grains",
                "Engage in regular physical activity to improve insulin sensitivity",
            ],
            "precautionary_measures": [
                "Attend regular check-ups with healthcare providers to monitor diabetes management",
                "Consult with an ophthalmologist to assess eye health and discuss preventive measures",
            ],
        },
        "Glaucoma": {
            "report": "The patient shows signs of glaucoma. Early detection and treatment are essential to prevent vision loss.",
            "preventative_measures": [
                "Attend regular eye exams, especially if at risk for glaucoma",
                "Follow treatment plans prescribed by your eye specialist",
                "Manage intraocular pressure through medication or other interventions",
            ],
            "precautionary_measures": [
                "Be vigilant for changes in vision and report them promptly to your doctor",
                "Discuss surgical options if medication alone isn't controlling glaucoma effectively",
            ],
        },
        "Hypertension": {
            "report": "It appears the patient has hypertension. Proper management is crucial to prevent potential eye complications.",
            "preventative_measures": [
                "Monitor blood pressure regularly and follow treatment plans prescribed by your doctor",
                "Adopt a heart-healthy diet low in sodium and high in fruits and vegetables",
                "Engage in regular physical activity to help lower blood pressure",
            ],
            "precautionary_measures": [
                "Attend regular check-ups with healthcare providers to monitor blood pressure control",
                "Inform your eye specialist about hypertension diagnosis for comprehensive care",
            ],
        },
        "Myopia": {
            "report": "The patient appears to have myopia. While common, it's important to monitor vision changes and consider corrective measures if needed.",
            "preventative_measures": [
                "Attend regular eye exams to monitor vision changes",
                "Take breaks during prolonged near work to reduce eye strain",
                "Consider corrective lenses or refractive surgery if vision significantly affects daily activities",
            ],
            "precautionary_measures": [
                "Discuss with an eye specialist for personalized recommendations based on severity",
                "Monitor for any progression of myopia and adjust treatment as necessary",
            ],
        },
        "Normal": {
            "report": "Great news! It seems like the patient's eyes are healthy. Regular check-ups are recommended to maintain eye health.",
            "preventative_measures": [
                "Continue with regular eye exams for ongoing monitoring",
                "Maintain overall health with a balanced diet and regular exercise",
                "Protect eyes from UV exposure with sunglasses when outdoors",
            ],
            "precautionary_measures": [
                "Stay informed about any changes in vision and report them promptly",
                "Schedule annual comprehensive eye check-ups to ensure continued eye health",
            ],
        },
        "Others": {
            "report": "The patient's condition falls into a category not specifically listed. Further evaluation and consultation with a healthcare provider are recommended.",
            "preventative_measures": [
                "Attend follow-up appointments as advised by your healthcare provider",
                "Discuss any concerns or symptoms with your doctor for appropriate management",
                "Follow recommended lifestyle measures for overall health and well-being",
            ],
            "precautionary_measures": [
                "Seek clarification from your healthcare provider regarding your specific condition",
                "Follow treatment plans or recommendations provided by specialists involved in your care",
            ],
        },
    }

    # Retrieve medical information based on predicted label
    medical_report = medical_info[predicted_label]["report"]
    preventative_measures = medical_info[predicted_label]["preventative_measures"]
    precautionary_measures = medical_info[predicted_label]["precautionary_measures"]

    # Generate conversational medical report with each section in a paragraphic fashion
    report = (
        "Medical Report:\n\n"
        + medical_report
        + "\n\nPreventative Measures:\n\n- "
        + ",\n- ".join(preventative_measures)
        + "\n\nPrecautionary Measures:\n\n- "
        + ",\n- ".join(precautionary_measures)
    )

    return report, precautionary_measures

def initialize_database(json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")

def save_image(image_file, json_file_path="data.json"):
    try:
        if image_file is None:
            st.warning("No file uploaded.")
            return

        if not session_state["logged_in"] or not session_state["user_info"]:
            st.warning("Please log in before uploading images.")
            return

        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user_info in data["users"]:
            if user_info["email"] == session_state["user_info"]["email"]:
                image = Image.open(image_file)

                if image.mode == "RGBA":
                    image = image.convert("RGB")

                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

                user_info["Pupil"] = image_base64

                with open(json_file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)

                session_state["user_info"]["Pupil"] = image_base64
                return

        st.error("User  not found.")
    except Exception as e:
        st.error(f"Error saving Pupil image to JSON: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "report": None,
            "precautions": None,
            "Pupil": None
        }
        data["users"].append(user_info)

        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None

def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    if st.button("Login"):
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None

def render_dashboard(user_info, json_file_path="data.json"):
    st.title(f"Welcome to the Dashboard, {user_info['name']}!")
    st.subheader("User  Information:")
    st.write(f"Name: {user_info['name']}")
    st.write(f"Sex: {user_info['sex']}")
    st.write(f"Age: {user_info['age']}")

    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
        for user in data["users"]:
            if user["email"] == user_info["email"]:
                if "Pupil" in user and user["Pupil"] is not None:
                    image_data = base64.b64decode(user["Pupil"])
                    st.image(Image.open(io.BytesIO(image_data)), caption="Uploaded Pupil Image", use_column_width=True)

                if isinstance(user_info["precautions"], list):
                    st.subheader("Precautions:")
                    for precaution in user_info["precautions"]:
                        st.write(precaution)                    
                else:
                    st.warning("Reminder: Please upload Pupil images and generate a report.")

def fetch_precautions(user_info):
    return (
        user_info["precautions"]
        if user_info["precautions"] is not None
        else "Please upload Pupil images and generate a report."
    )

def main(json_file_path="data.json"):
    st.markdown(
        """
        <style>
            body {
                background-color: #0b1e34;
                color: white;
            }
            .st-bw {
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Pupillometry Analysis System")
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "Dashboard", "Upload Eye Image", "View Reports"),
        key="Pupillometry Analysis System",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Upload Eye Image":
        if session_state.get("logged_in"):
            st.title("Upload Pupil Image")
            uploaded_image = st.file_uploader(
                "Choose a Pupil image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
            )
            if uploaded_image is not None:
                st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
                if st.button("Predict Condition"):
                    condition, severity_level = predict(uploaded_image, model)
                    st.write("Predicted Condition: ", condition)
                    st.write("Severity Level: ", severity_level)  # Display severity level
                    report, precautions = generate_medical_report(condition)
                    st.write(report)
                    st.write("\nAdditional Precautionary Measures:\n- " + ",\n- ".join(precautions))

                    with open(json_file_path, "r+") as json_file:
                        data = json.load(json_file)
                        user_index = next((i for i, user in enumerate(data["users"]) if user["email"] == session_state["user_info"]["email"]), None)
                        if user_index is not None:
                            user_info = data["users"][user_index]
                            user_info["report"] = report
                            user_info["precautions"] = precautions
                            session_state["user_info"] = user_info
                            json_file.seek(0)
                            json.dump(data, json_file, indent=4)
                            json_file.truncate()
                        else:
                            st.error("User  not found.")
        else:
            st.warning("Please login/signup to upload a pupil image.")

    elif page == "View Reports":
        if session_state.get("logged_in"):
            st.title("View Reports")
            user_info = get_user_info(session_state["user_info"]["email"], json_file_path)
            if user_info is not None:
                if user_info["report"] is not None:
                    st.subheader("Pupil Report:")
                    st.write(user_info["report"])
                else:
                    st.warning("No reports available.")
            else:
                st.warning("User  information not found.")
        else:
            st.warning("Please login/signup to view reports.")

if __name__ == "__main__":
    initialize_database()
    main()
