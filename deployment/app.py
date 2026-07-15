
import gradio as gr
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


MODEL_REPO = "kavita89/tourism-model"

# Download model
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="MLOps_model.joblib"
)

model = joblib.load(model_path)

# Try downloading preprocessor from HF, otherwise load local copy
try:
    preprocessor_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="preprocessor.pkl"
    )
except Exception:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocessor_path = os.path.join(
        BASE_DIR,
        "model_building",
        "preprocessor.pkl"
    )

preprocessor = joblib.load(preprocessor_path)


joblib.dump(model, "MLOps_model.joblib")
joblib.dump(preprocessor, "preprocessor.pkl")

from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_file(
    path_or_fileobj="MLOps_model.joblib",
    path_in_repo="MLOps_model.joblib",
    repo_id="kavita89/tourism-model",
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="preprocessor.pkl",
    path_in_repo="preprocessor.pkl",
    repo_id="kavita89/tourism-model",
    repo_type="model"
)


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict(
    age,
    type_of_contact,
    city_tier,
    duration_of_pitch,
    occupation,
    gender,
    number_of_person_visiting,
    number_of_followups,
    product_pitched,
    preferred_property_star,
    marital_status,
    number_of_trips,
    passport,
    pitch_satisfaction_score,
    own_car,
    number_of_children_visiting,
    designation,
    monthly_income,
):

    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    numerical_cols = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "MonthlyIncome"
    ]

    categorical_cols = [
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation"
    ]

    processed = preprocessor.transform(
        input_df[numerical_cols + categorical_cols]
    )

    prediction = model.predict(processed)[0]

    if prediction == 1:
        return "✅ Customer is likely to purchase the Wellness Tourism Package."
    else:
        return "❌ Customer is unlikely to purchase the Wellness Tourism Package."

## --------------------------------------------------
# Gradio Interface
# --------------------------------------------------

with gr.Blocks(title="Wellness Tourism Package Purchase Prediction") as demo:

    gr.Markdown("# 🌍 Wellness Tourism Package Purchase Prediction")
    gr.Markdown(
        "Enter customer information below to predict whether they are likely to purchase the Wellness Tourism Package."
    )

    with gr.Row():
        age = gr.Slider(18, 90, value=30, label="Age")
        type_of_contact = gr.Dropdown(
            ["Self Enquiry", "Company Invited"],
            label="Type of Contact"
        )
        city_tier = gr.Dropdown([1, 2, 3], label="City Tier")

    with gr.Row():
        duration_of_pitch = gr.Slider(0, 100, value=15, label="Duration of Pitch")
        occupation = gr.Dropdown(
            ["Salaried", "Freelancer", "Small Business", "Large Business"],
            label="Occupation"
        )
        gender = gr.Dropdown(["Male", "Female"], label="Gender")

    with gr.Row():
        number_of_person_visiting = gr.Slider(
            1, 5, value=2, label="Number of Persons Visiting"
        )
        number_of_followups = gr.Slider(
            0, 10, value=3, label="Number of Followups"
        )
        product_pitched = gr.Dropdown(
            ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
            label="Product Pitched"
        )

    with gr.Row():
        preferred_property_star = gr.Dropdown(
            [3, 4, 5],
            label="Preferred Property Star"
        )
        marital_status = gr.Dropdown(
            ["Single", "Married", "Divorced"],
            label="Marital Status"
        )
        number_of_trips = gr.Slider(
            0, 20, value=2, label="Number of Trips"
        )

    with gr.Row():
        passport = gr.Radio(
            [0, 1],
            value=0,
            label="Passport (0=No, 1=Yes)"
        )
        pitch_satisfaction_score = gr.Slider(
            1, 5,
            value=3,
            label="Pitch Satisfaction Score"
        )
        own_car = gr.Radio(
            [0, 1],
            value=0,
            label="Own Car (0=No, 1=Yes)"
        )

    with gr.Row():
        number_of_children_visiting = gr.Slider(
            0,
            3,
            value=0,
            label="Number of Children Visiting"
        )
        designation = gr.Dropdown(
            ["Manager", "Executive", "Senior Manager", "AVP", "VP", "Director"],
            label="Designation"
        )
        monthly_income = gr.Number(
            value=25000,
            label="Monthly Income"
        )

    predict_btn = gr.Button("Predict")

    prediction = gr.Textbox(
        label="Prediction",
        lines=2
    )

    predict_btn.click(
        fn=predict,
        inputs=[
            age,
            type_of_contact,
            city_tier,
            duration_of_pitch,
            occupation,
            gender,
            number_of_person_visiting,
            number_of_followups,
            product_pitched,
            preferred_property_star,
            marital_status,
            number_of_trips,
            passport,
            pitch_satisfaction_score,
            own_car,
            number_of_children_visiting,
            designation,
            monthly_income,
        ],
        outputs=prediction,
    )

if __name__ == "__main__":
    demo.launch()
