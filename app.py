import streamlit as st
import pandas as pd
import joblib
from groq import Groq

st.set_page_config(page_title="CareCompass", page_icon="🧭", layout="wide")

# ---- Load data & model ----
@st.cache_data
def load_data():
    patients_df = pd.read_csv('patients.csv')
    doctors_df = pd.read_csv('doctors.csv')
    appointments_df = pd.read_csv('appointments.csv')
    schemes_df = pd.read_csv('schemes.csv')
    return patients_df, doctors_df, appointments_df, schemes_df

@st.cache_resource
def load_model():
    return joblib.load('consultation_model.pkl')

patients_df, doctors_df, appointments_df, schemes_df = load_data()
consultation_model = load_model()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

st.title("🧭 CareCompass")
st.caption("An integrated patient helpdesk assistant: chatbot, welfare scheme eligibility, consultation time prediction, and AI-generated discharge reports.")

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chatbot", "📋 Scheme Eligibility", "⏱️ Consultation Time", "📄 Discharge Report"
])

# ---------------- TAB 1: CHATBOT ----------------
with tab1:
    st.subheader("Ask the Hospital Assistant")
    st.caption("Ask about hospital hours, departments, doctor availability, or visiting hours.")

    def hospital_chatbot(query, doctors_df):
        query = query.lower()
        if 'hours' in query or 'timing' in query or 'open' in query:
            return "The hospital is open Monday to Saturday, 9 AM to 6 PM. Emergency services are available 24/7."
        if 'department' in query:
            depts = doctors_df['department'].unique()
            return f"We have the following departments: {', '.join(depts)}."
        for _, doc in doctors_df.iterrows():
            if doc['name'].lower() in query:
                return (f"Dr. {doc['name']} ({doc['department']}) is available on "
                         f"{doc['available_days']}, {doc['available_hours']}.")
        for dept in doctors_df['department'].unique():
            if dept.lower() in query:
                dept_doctors = doctors_df[doctors_df['department'] == dept]
                lines = [f"Dr. {row['name']} — {row['available_days']}, {row['available_hours']}"
                         for _, row in dept_doctors.iterrows()]
                return f"{dept} doctors and their availability:\n" + "\n".join(lines)
        if 'visitor' in query or 'visiting' in query:
            return "Visiting hours are 4 PM to 6 PM daily. Maximum 2 visitors per patient at a time."
        return "I'm not sure I understood that. You can ask me about hospital hours, departments, doctor availability, or visiting hours."

    user_query = st.text_input("Your question", placeholder="e.g. When is Neurology available?")
    if st.button("Ask", key="chatbot_btn"):
        if user_query.strip():
            answer = hospital_chatbot(user_query, doctors_df)
            st.info(answer)
        else:
            st.warning("Type a question first.")

# ---------------- TAB 2: SCHEME ELIGIBILITY ----------------
with tab2:
    st.subheader("Government Welfare Scheme Eligibility Checker")
    st.caption("Real scheme data sourced from MyScheme.gov.in, checked against patient profiles. "
                "Note: this checks a curated 'Health & Wellness' category subset (212 schemes) — "
                "some criteria could not be auto-extracted from free-text and are flagged for manual review.")

    selected_patient_id = st.selectbox("Select a patient", patients_df['patient_id'] + " - " + patients_df['name'])
    patient_id = selected_patient_id.split(" - ")[0]
    patient = patients_df[patients_df['patient_id'] == patient_id].iloc[0]

    st.write(f"**Age:** {patient['age']} | **Category:** {patient['category']} | "
             f"**Annual Income:** ₹{patient['annual_income']:,} | **State:** {patient['state']}")

    if st.button("Check Eligibility"):
        verified, needs_review = [], []
        for _, scheme in schemes_df.iterrows():
            has_criteria = (pd.notna(scheme['income_limit']) or pd.notna(scheme['age_limit']) or
                             pd.notna(scheme['category_req']) or pd.notna(scheme['state_req']))
            if has_criteria:
                eligible, reasons = True, []
                if pd.notna(scheme['income_limit']):
                    if patient['annual_income'] <= scheme['income_limit']:
                        reasons.append(f"Income within limit (₹{scheme['income_limit']:,.0f})")
                    else:
                        eligible = False
                if pd.notna(scheme['age_limit']):
                    if patient['age'] >= scheme['age_limit']:
                        reasons.append(f"Meets age requirement ({scheme['age_limit']}+)")
                    else:
                        eligible = False
                if pd.notna(scheme['category_req']):
                    if patient['category'] in str(scheme['category_req']):
                        reasons.append(f"Category matches ({patient['category']})")
                    else:
                        eligible = False
                if pd.notna(scheme['state_req']):
                    if patient['state'] == scheme['state_req']:
                        reasons.append(f"State matches ({patient['state']})")
                    else:
                        eligible = False
                if eligible:
                    verified.append({'Scheme': scheme['scheme_name'], 'Reason': '; '.join(reasons)})
            else:
                needs_review.append({'Scheme': scheme['scheme_name'], 'Eligibility Text': str(scheme['eligibility'])[:150] + '...'})

        st.success(f"✅ Verified Eligible: {len(verified)} schemes")
        st.dataframe(pd.DataFrame(verified), use_container_width=True, hide_index=True)

        with st.expander(f"📋 Needs Manual Review ({len(needs_review)} schemes — criteria not auto-extractable)"):
            st.dataframe(pd.DataFrame(needs_review), use_container_width=True, hide_index=True)

# ---------------- TAB 3: CONSULTATION TIME PREDICTION ----------------
with tab3:
    st.subheader("Predict Consultation Time")
    st.caption("Random Forest / Linear Regression model trained on scheduling patterns (booking ratio, time of day).")

    col1, col2 = st.columns(2)
    with col1:
        pred_doctor = st.selectbox("Doctor", doctors_df['doctor_id'] + " - " + doctors_df['name'])
        total_slots = st.slider("Total slots that day", 10, 30, 20)
        booked_slots = st.slider("Booked slots so far", 0, total_slots, int(total_slots * 0.6))
    with col2:
        appt_hour = st.slider("Appointment hour (24h)", 9, 17, 11)
        appt_minute = st.selectbox("Minute", [0, 15, 30, 45])

    if st.button("Predict Duration"):
        booking_ratio = booked_slots / total_slots
        start_time_minutes = appt_hour * 60 + appt_minute

        input_row = pd.DataFrame([{
            'total_slots_that_day': total_slots,
            'booked_slots_that_day': booked_slots,
            'booking_ratio': booking_ratio,
            'start_time_minutes': start_time_minutes
        }])

        predicted_duration = consultation_model.predict(input_row)[0]
        st.success(f"⏱️ Predicted consultation duration: **{predicted_duration:.1f} minutes**")
        st.caption(f"Booking ratio: {booking_ratio:.0%} — busier days tend to run slightly longer.")

# ---------------- TAB 4: DISCHARGE REPORT ----------------
with tab4:
    st.subheader("AI-Generated Discharge Report")
    st.caption("Pulls full visit details via Visit ID across Patient, Doctor, and Appointment tables, then generates a report using Groq (Llama 3.3 70B).")

    selected_visit = st.selectbox("Select a Visit ID", appointments_df['visit_id'])

    if st.button("Generate Report"):
        visit = appointments_df[appointments_df['visit_id'] == selected_visit].iloc[0]
        patient = patients_df[patients_df['patient_id'] == visit['patient_id']].iloc[0]
        doctor = doctors_df[doctors_df['doctor_id'] == visit['doctor_id']].iloc[0]

        with st.expander("🔍 Raw visit data used"):
            st.json({
                'patient_name': patient['name'], 'age': int(patient['age']), 'gender': patient['gender'],
                'medical_history': patient['medical_history'], 'doctor': doctor['name'],
                'department': doctor['department'], 'visit_date': visit['visit_date'],
                'reason': visit['reason_for_visit'], 'symptoms': visit['symptoms'],
                'diagnosis': visit['diagnosis'], 'medication': visit['prescribed_medication'],
                'duration_min': int(visit['consultation_duration_min'])
            })

        prompt = f"""You are a hospital assistant generating a discharge report. Write a clear, professional, easy-to-understand discharge summary based on the following visit details:

Patient: {patient['name']}, {patient['age']} years old, {patient['gender']}
Medical History: {patient['medical_history']}
Attending Doctor: Dr. {doctor['name']} ({doctor['department']})
Visit Date: {visit['visit_date']}
Reason for Visit: {visit['reason_for_visit']}
Symptoms Reported: {visit['symptoms']}
Diagnosis: {visit['diagnosis']}
Prescribed Medication: {visit['prescribed_medication']}
Consultation Duration: {visit['consultation_duration_min']} minutes

Write a discharge report with these sections: Summary, Diagnosis Notes, Medication Instructions, and Follow-up Recommendations. Base the Diagnosis Notes strictly on the diagnosis provided above — do not invent a different diagnosis. Keep it concise, warm, and easy for a patient to understand — avoid overly technical jargon."""

        with st.spinner("Generating report..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
        st.markdown(response.choices[0].message.content)
