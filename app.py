import streamlit as st
import pandas as pd
import pickle

# Load the model and label encoders
with open('random_forest_model_course_recommendation.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Assuming the label encoder for the target (Preferred Course) is also stored
course_encoder = label_encoders['target']  # Adjust this if the key is different

# Streamlit user interface
def main():
    
    st.title('Course Recommendation System')


    
    st.markdown("#### PROJECT WORK BY: Rashsan")
    
    
    st.write('Answer these questions to get recommended courses.')
    # User inputs for each feature
    input_data = {}
    input_data['Class'] = st.selectbox('Secondary School Class/Department', options=['technical', 'science', 'Arts', 'commercial'])
    input_data['Learning style'] = st.selectbox('Preferred Learning Style', options=['calculation', 'calculations', 'kinesthetics', 'logical', 'writing', 'Auditory', 'Reading'])
    input_data['Work experience'] = st.selectbox('Do you have any Work Experience', options=['No', 'Yes'])
    input_data['Academic subjects'] = st.selectbox('A subject you like', options=['maths', 'computer science', 'physics', 'biology', 'further maths', 'chemistry', 'English', 'civic', 'literature', 'civic education', 'government'])
    input_data['Subject dislikes'] = st.selectbox('A subject you dislikes', options=['economics', 'english', 'Agric', 'geography', 'biology', 'civic', 'maths', 'physics', 'chemistry'])
    input_data['Strengths'] = st.selectbox('What is your strength', options=['critical thinking', 'technical proficiency', 'team collaboration', 'communication', 'creativity', 'team work'])
    input_data['Weakness'] = st.selectbox('What is your weakness', options=['writing', 'Time management', 'public speaking', 'concentration', 'Technical skills', 'communication'])
    input_data['Career aspiration'] = st.selectbox('Career Aspiration', options=['cloud engineer', 'data scientist', 'software engineer', 'programmer', 'machine engineering', 'Business Analyst', 'Information scientist', 'Data Analyst', 'Business Administrator', 'network engineer', 'cyber security analyst', 'product marketing manager', 'news broadcaster', 'Writer', 'auditor', 'librarian', 'database manager', 'information broker'])
    input_data['Extra-curricular'] = st.selectbox('Extra-curricular Activities', options=['drawing', 'math club', 'jet club', 'press club', 'reading', 'writing'])
    Jamb_score = st.number_input('Jamb Score') 
    # Button to make prediction
    if st.button('Recommend Courses'):
        # Preprocess inputs: encode categorical data using the loaded label encoders
        try:
            for key, value in input_data.items():
                if key in label_encoders and value is not None:
                    encoder = label_encoders[key]
                    input_data[key] = encoder.transform([value])[0]
        except Exception as e:
            st.error(f"Error in processing input: {str(e)}")
    
        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])
    
        # # Predict the probabilities
        # probabilities = model.predict_proba(input_df)[0]
        # classes = model.classes_
    
        # # Decode class labels to actual course names
        # decoded_classes = course_encoder.inverse_transform(classes)
    
        # # Get the top 5 predictions
        # top_indices = probabilities.argsort()[-5:][::-1]
        # top_courses = decoded_classes[top_indices]
        # top_probabilities = probabilities[top_indices] * 100  # Convert probability to percentage
    
        department_cutoffs = {
        'Computer Science': 220,
        'Information and Communication Science': 190,
        'Telecommunication Science': 220,
        'Mass Communication': 240,
        'Library and Information Science': 190
        }
    
            # Predict the probabilities
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_
    
        # Decode class labels to actual course names
        decoded_classes = course_encoder.inverse_transform(classes)
    
        # Get the top 5 predictions
        top_indices = probabilities.argsort()[-5:][::-1]
        top_courses = decoded_classes[top_indices]
        top_probabilities = probabilities[top_indices] * 100  # Convert probability to percentage
    
        # Check if user meets department requirements based on Jamb score
         # Replace with actual Jamb score input
    
        meets_requirement = []
        for course in top_courses:
            if course in department_cutoffs and Jamb_score >= department_cutoffs[course]:
                meets_requirement.append('Yes')
            else:
                meets_requirement.append('No')
    
        # Prepare the DataFrame for display
        results_df = pd.DataFrame({
            'Recommended Course': top_courses,
            'Confidence (%)': top_probabilities,
            'Meets Requirement': meets_requirement
        })
        st.table(results_df)


                
if __name__ == "__main__":
    main()
