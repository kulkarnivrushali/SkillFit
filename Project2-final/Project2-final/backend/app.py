from flask import Flask, render_template, request, session, redirect, url_for
import os
from random import shuffle
import random 

app = Flask(__name__)

# Secret Key for session management
app.config['SECRET_KEY'] = 'your_strong_secret_key'

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Predefined skill ontology
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

# Utility function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate MCQs
def generate_mcqs(skills):
    # Array of question templates
    question_templates = [
        "What is {skill} primarily used for?",
        "Which of the following best describes {skill}?",
        "Identify a key feature of {skill}.",
        "What is a common challenge in using {skill}?",
        "Choose an application of {skill} from the options below."
    ]
    
    # Array of option templates
    option_templates = [
        ["{skill} Basics", "{skill} Advanced Use", "Common Challenges of {skill}", "Applications of {skill}"],
        ["Core Features of {skill}", "Advanced Concepts of {skill}", "Common Issues with {skill}", "Practical Uses of {skill}"],
        ["Introduction to {skill}", "{skill} Best Practices", "Limitations of {skill}", "Examples of {skill} Usage"]
    ]
    
    # Define question distribution
    question_distribution = {
        "technical": 12,  # 60% of 20 questions
        "soft": 4,        # 20% of 20 questions
        "managerial": 4   # 20% of 20 questions
    }
    
    mcqs = {"technical": [], "soft": [], "managerial": []}
    
    for category, count in question_distribution.items():
        # Get the skill set for this category
        skill_set = skills.get(category, SKILL_ONTOLOGY.get(category, []))
        
        for i, skill in enumerate(skill_set[:count]):
            # Randomly choose a question template and generate a question
            question_template = random.choice(question_templates)
            question = question_template.format(skill=skill)
            
            # Randomly choose an option template and generate options
            options_template = random.choice(option_templates)
            options = [opt.format(skill=skill) for opt in options_template]
            random.shuffle(options)  # Shuffle the options
            
            # Add the MCQ to the list
            mcqs[category].append({
                "id": f"{category}_{i + 1}",
                "question": question,
                "options": options,
                "correct_answer": options[0]  # The first option is always correct for simplicity
            })
    
    return mcqs

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Mock skill extraction (replace with actual implementation)
            extracted_skills = {
                "technical": ["python", "sql", "java"],
                "soft": ["communication", "teamwork"],
                "managerial": ["leadership", "management"]
            }
            session['extracted_skills'] = extracted_skills
            return render_template('skills.html', skills=extracted_skills)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Start test route
@app.route('/start_test')
def start_test():
    extracted_skills = {
        "technical": ["Python", "SQL","Database","Java"],
        "soft": ["Communication","Teamwork"],
        "managerial": ["Leadership"]
    }

    mcqs = generate_mcqs(extracted_skills)
    session['mcqs'] = mcqs  # Save MCQs in session for scoring
    return render_template('test.html', mcqs=mcqs)

# Submit test route
@app.route('/submit_test', methods=['POST'])
def submit_test():
    answers = request.form
    mcqs = session.get('mcqs', {})
    score = 0
    total = 0

    # Calculate the score
    for category, questions in mcqs.items():
        for question in questions:
            total += 1
            qid = question['id']
            if answers.get(qid) == question['correct_answer']:
                score += 1

    # Calculate percentage
    percentage = (score / total) * 100 if total > 0 else 0

    # Prepare result context
    result = {
        "score": score,
        "total": total,
        "percentage": round(percentage, 2),
    }

    return render_template('score.html', result=result)


# Instructions route
@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

# Eligibility route
@app.route('/eligibility')
def eligibility_check():
    extracted_skills = session.get('extracted_skills', {})
    technical_skills = SKILL_ONTOLOGY['technical']
    matched_skills = set(extracted_skills.get('technical', [])).intersection(technical_skills)
    eligible = len(matched_skills) >= 2
    return render_template('eligibility.html', eligible=eligible)

if __name__ == '__main__':
    app.run(debug=True)




'''
from flask import Flask, render_template, request, session, redirect, url_for
import os
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from transformers import pipeline
from random import shuffle

app = Flask(__name__)

# Secret Key for session management
app.config['SECRET_KEY'] = 'your_strong_secret_key'

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Predefined skill ontology
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

# Utility function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate MCQs using Hugging Face Transformers
def generate_mcqs(extracted_skills):
    mcqs = {"technical": [], "soft": [], "managerial": []}
    question_generator = pipeline("text-generation", model="gpt2", pad_token_id=50256)
    question_distribution = {
        "technical": 12,  # 60% of 20 questions
        "soft": 4,       # 20% of 20 questions
        "managerial": 4   # 20% of 20 questions
    }

    # Fill missing categories with ontology skills
    for category in question_distribution:
        if not extracted_skills.get(category):
            extracted_skills[category] = SKILL_ONTOLOGY[category][:question_distribution[category]]

    for category, count in question_distribution.items():
        skills = extracted_skills.get(category, [])
        for skill in skills[:count]:
            prompt = f"Generate a multiple-choice question about the skill '{skill}', with four options and a correct answer."
            try:
                result = question_generator(prompt, max_length=50, num_return_sequences=1, truncation=True)
                question_text = result[0]["generated_text"].split('\n')[0].strip()

                # Generate options
                options = [f"Option A: {skill} use case"]
                incorrect_options = [f"Option B: {skill} basics", f"Option C: {skill} challenges", f"Option D: {skill} applications"]
                shuffle(incorrect_options)
                options.extend(incorrect_options[:3])
                shuffle(options)

                # Add generated question to MCQs
                mcqs[category].append({
                    "id": f"{category}_{len(mcqs[category])}",
                    "question": question_text,
                    "options": options,
                    "correct_answer": options[0]  # Correct answer is always the first option
                })
            except Exception as e:
                print(f"Error generating question for skill {skill}: {e}")

    # Shuffle questions in each category
    for questions in mcqs.values():
        shuffle(questions)

    session['mcqs'] = mcqs
    return mcqs

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            session['extracted_skills'] = skills
            return render_template('skills.html', skills=skills)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Start test route
@app.route('/start_test', methods=['GET'])
def start_test():
    extracted_skills = session.get('extracted_skills', {})
    if not extracted_skills:
        return render_template('error.html', error_message='No skills found. Please upload your resume.')
    mcqs = generate_mcqs(extracted_skills)
    return render_template('test.html', mcqs=mcqs)

# Submit test route
@app.route('/submit_test', methods=['POST'])
def submit_test():
    submitted_answers = request.form
    mcqs = session.get('mcqs', {})
    score = 0
    total_questions = sum(len(questions) for questions in mcqs.values())
    correct_answers = {q['id']: q['correct_answer'] for questions in mcqs.values() for q in questions}

    for qid, answer in submitted_answers.items():
        if correct_answers.get(qid) == answer:
            score += 1

    percentage = (score / total_questions) * 100 if total_questions > 0 else 0
    return render_template('score.html', score={"score": score, "total": total_questions, "percentage": round(percentage, 2)})

# Instructions route
@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

@app.route('/eligibility')
def eligibility_check():
    extracted_skills = session.get('extracted_skills', {})
    if not extracted_skills:
        return render_template('error.html', error_message="No skills found. Please upload your resume first.")
    
    # Define eligibility logic (example: at least 2 technical skills match the ontology)
    technical_skills = SKILL_ONTOLOGY.get('technical', [])
    matched_skills = set(extracted_skills.get('technical', [])).intersection(technical_skills)
    eligible = len(matched_skills) >= 2
    
    return render_template('eligibility.html', eligible=eligible)

if __name__ == '__main__':
    app.run(debug=True)
'''


'''  
from flask import Flask, render_template, request, jsonify, session
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from transformers import pipeline
from random import shuffle
import random

app = Flask(__name__)

# Secret Key for session management (replace with a strong random key)
app.config['SECRET_KEY'] = 'your_strong_secret_key'

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Predefined skill ontology
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Generate MCQs using Hugging Face Transformers
def generate_mcqs(extracted_skills):
    mcqs = {"technical": [], "soft": [], "managerial": []}
    question_generator = pipeline("text-generation", model="gpt2", pad_token_id=50256)  # Updated model usage
    question_distribution = {
        "technical": 12,  # 60% of 20 questions
        "soft": 4,       # 20% of 20 questions
        "managerial": 4   # 20% of 20 questions
    }

    # Fill missing categories with ontology skills (ensure at least one question per category)
    for category in question_distribution:
        if not extracted_skills.get(category):
            extracted_skills[category] = SKILL_ONTOLOGY[category][:question_distribution[category]]

    for category, count in question_distribution.items():
        skills = extracted_skills.get(category, [])
        for skill in skills[:count]:
            prompt = f"Generate a multiple-choice question about the skill '{skill}', with four options and a correct answer."
            try:
                result = question_generator(prompt, max_length=50, num_return_sequences=1, truncation=True)
                question_text = result[0]["generated_text"].split('\n')[0].strip()

                # Generate 4 unique options by shuffling and selecting 3 incorrect options
                options = [f"Option A: {skill} use case"]
                incorrect_options = [f"Option B: {skill} basics", f"Option C: {skill} challenges", f"Option D: {skill} applications"]
                shuffle(incorrect_options)
                options.extend(incorrect_options[:3])
                shuffle(options)

                # Add generated question to MCQs with the correct answer
                mcqs[category].append({
                    "id": f"{category}_{len(mcqs[category])}",
                    "question": question_text,
                    "options": options,
                    "correct_answer": options[0]  # Correct answer is always the first option
                })
            except Exception as e:
                print(f"Error generating question for skill {skill}: {e}")

    # Ensure consistent ordering of questions
    for questions in mcqs.values():
        shuffle(questions)

    # Store generated MCQs in session
    session['mcqs'] = mcqs 

    return mcqs

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            session['extracted_skills'] = skills  # Store extracted skills in session
            return render_template('skills.html', skills=skills)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Start test route
@app.route('/start_test', methods=['GET'])
def start_test():
    # Get extracted skills from the session
    extracted_skills = session.get('extracted_skills', {})  # Use session.get to handle potential absence of 'extracted_skills'
    if not extracted_skills:
        return render_template('error.html', error_message='No skills found. Please upload your resume.')
    mcqs = generate_mcqs(extracted_skills)
    return render_template('test.html', mcqs=mcqs)

# Submit test route
@app.route('/submit_test', methods=['POST'])
def submit_test():
    submitted_answers = request.form
    score = 0

    # Get MCQs from the session
    mcqs = session.get('mcqs', {})  # Use session.get to handle potential absence of 'mcqs'

    if not mcqs:
        return render_template('error.html', error_message='Test not started. Please begin the test.')

    for qid, answer in submitted_answers.items():
        if qid in mcqs:
            if mcqs[qid]['correct_answer'] == answer:
                score += 1

    total_questions = len(mcqs)
    percentage = (score / total_questions) * 100

    return render_template('score.html', score={"score": score, "total": total_questions, "percentage": round(percentage, 2)})

# Eligibility route
@app.route('/eligibility')
def eligibility_check():
    extracted_skills = {
        'technical': ['python', 'java', 'html'],
        'soft': ['communication', 'teamwork'],
        'managerial': ['leadership', 'management']
    }
    eligible = len(set(extracted_skills['technical']).intersection(SKILL_ONTOLOGY['technical'])) >= 2
    return render_template('eligibility.html', eligible=eligible)

# Instructions route
@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

if __name__ == '__main__':
    app.run(debug=True)'''



'''
from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score
from random import shuffle

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Predefined skill ontology (for eligibility check)
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

QUESTION_TEMPLATES = {
    "technical": [
        {"template": "Which is a common use case for {skill}?", "options": ["Backend development", "Cloud computing", "Security analysis", "Front-end design"]},
        {"template": "What is {skill} primarily used for?", "options": ["Web development", "Data analysis", "Game development", "System automation"]},
        {"template": "Which of the following best describes {skill}?", "options": ["A programming language", "A database", "A framework", "A debugging tool"]},
    ],
    "soft": [
        {"template": "How does {skill} improve teamwork?", "options": ["Better collaboration", "Avoiding communication", "Strict rule-following", "Conflict generation"]},
        {"template": "What is an effective way to improve {skill}?", "options": ["Practice active listening", "Avoid difficult tasks", "Rely solely on intuition", "Use formal language"]},
    ],
    "managerial": [
        {"template": "How does {skill} help in project success?", "options": ["Clear goal-setting", "Avoiding delegation", "Improvisation", "Reducing workload"]},
        {"template": "What does {skill} involve?", "options": ["Effective decision-making", "Avoiding communication", "Leading without strategy", "Strict adherence to rules"]},
    ]
}

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to calculate accuracy for extracted skills
def calculate_accuracy(extracted_skills):
    predefined_skills = [skill.lower() for skill in SKILL_ONTOLOGY["technical"] + SKILL_ONTOLOGY["soft"] + SKILL_ONTOLOGY["managerial"]]
    all_extracted_skills = [skill.lower() for skills in extracted_skills.values() for skill in skills]

    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    return round((correct_skills / total_skills) * 100, 2) if total_skills > 0 else 0

# Updated eligibility logic to allow at least 2 skills to match
def check_eligibility(extracted_skills):
    total_matching_skills = 0

    # Check technical skills category for at least 2 matching skills
    technical_skills = SKILL_ONTOLOGY.get('technical', [])
    matching_technical_skills = set(extracted_skills.get('technical', []))
    total_matching_skills += len(matching_technical_skills)

    return total_matching_skills >= 2

# Generate MCQs using extracted skills and Trivia API
# Generate MCQs using extracted skills and Trivia API
def generate_mcqs():
    mcqs = []
    correct_answers = {}
    question_id = 1
    for category, templates in QUESTION_TEMPLATES.items():
        skills = SKILL_ONTOLOGY[category][:5]  # Limit to 5 skills per category for variety
        for skill in skills:
            for template in templates:
                question_text = template["template"].format(skill=skill.capitalize())
                options = template["options"]
                correct_answer = options[0]  # The first option is the correct answer
                shuffle(options)

                mcqs.append({
                    "id": f"q{question_id}",
                    "question": f"{question_id}. {question_text}",
                    "options": options
                })
                correct_answers[f"q{question_id}"] = correct_answer
                question_id += 1

    # Save correct answers to a JSON file
    with open("correct_answers.json", "w") as f:
        json.dump(correct_answers, f)

    return mcqs


# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, extract skills, and render results."""
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)
            eligible = check_eligibility(skills)

            return render_template('skills.html', skills=skills, accuracy=accuracy, eligible=eligible)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Route to start the test and display MCQs
@app.route('/start_test', methods=['GET'])
def start_test():
    """Generate MCQs and display the test."""
    mcqs = generate_mcqs()
    return render_template('test.html', mcqs=mcqs)

# Submit test and calculate score
@app.route('/submit_test', methods=['POST'])
def submit_test():
    """Calculate and display the score."""
    submitted_answers = request.form.to_dict()

    # Load correct answers from JSON file
    with open("correct_answers.json") as f:
        correct_answers = json.load(f)

    # Calculate score
    score = sum(1 for qid, answer in submitted_answers.items() if correct_answers.get(qid) == answer)
    total_questions = len(correct_answers)
    percentage = (score / total_questions) * 100

    return render_template(
        'score.html',
        score={"score": score, "total": total_questions, "percentage": round(percentage, 2)}
    )

# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills.get(category, []))
            predicted_skills = set(extracted_skills.get(category, []))

            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    accuracy = round((correct_predictions / total_skills) * 100, 2) if total_skills > 0 else 0
    precision = precision_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                                 [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    recall = recall_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                           [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    f1 = f1_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                  [1] * correct_predictions + [0] * (total_skills - correct_predictions))

    return accuracy, precision, recall, f1

@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/instructions')
def instructions():
    """Render instructions for the test."""
    return render_template('instructions.html')


@app.route('/eligibility')
def eligibility_check():
    """Render the eligibility check page."""
    eligible = False
    # Example: Assume extracted_skills are passed through context or session
    extracted_skills = {
        'technical': ['python', 'java', 'html'],
        'soft': ['communication', 'teamwork'],
        'managerial': ['leadership', 'management']
    }
    eligible = check_eligibility(extracted_skills)
    return render_template('eligibility.html', eligible=eligible)

if __name__ == '__main__':
    app.run(debug=True)
'''

'''
from flask import Flask, render_template, request, jsonify
import os
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from random import shuffle

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Skill ontology for MCQ generation
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate MCQs
def generate_mcqs(extracted_skills):
    """Generate MCQs based on extracted skills."""
    mcqs = {"technical": [], "soft": [], "managerial": []}
    question_templates = {
        "technical": [
            {"template": "What is {skill} primarily used for?", "options": ["Web development", "Data analysis", "System automation", "Game development"]},
            {"template": "Which of the following is a key feature of {skill}?", "options": ["Ease of use", "Scalability", "Interoperability", "High performance"]},
        ],
        "soft": [
            {"template": "Which is the most effective way to improve {skill}?", "options": ["Practice active listening", "Avoid difficult situations", "Rely solely on intuition", "Use formal language"]},
            {"template": "What does {skill} primarily help achieve in a team setting?", "options": ["Better collaboration", "Individual success", "Conflict avoidance", "Faster task completion"]},
        ],
        "managerial": [
            {"template": "How does {skill} contribute to a project's success?", "options": ["Clear goal-setting", "Reducing workload", "Improvisation", "Avoiding delegation"]},
            {"template": "Which of the following describes {skill}?", "options": ["Effective decision-making", "Strict adherence to rules", "Avoiding feedback", "Leading without communication"]},
        ],
    }

    for category, skills in extracted_skills.items():
        if category not in question_templates:
            continue
        for skill in skills:
            templates = question_templates[category]
            for template_data in templates:
                question_text = template_data["template"].format(skill=skill.capitalize())
                options = template_data["options"]
                correct_answer = options[0]
                shuffle(options)

                mcqs[category].append({
                    "question": question_text,
                    "options": options,
                    "answer": correct_answer
                })

    for questions in mcqs.values():
        shuffle(questions)

    return mcqs

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and skill extraction."""
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part provided."}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            mcqs = generate_mcqs(skills)
            return jsonify({"skills": skills, "mcqs": mcqs})
        except Exception as e:
            return jsonify({"error": f"Error extracting skills: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type."}), 400

@app.route('/start_test', methods=['GET'])
def start_test():
    """Render the test page with generated MCQs."""
    extracted_skills = {
        'technical': ['python', 'java'],
        'soft': ['communication'],
        'managerial': ['leadership']
    }
    mcqs = generate_mcqs(extracted_skills)
    return render_template('test.html', mcqs=mcqs)

@app.route('/submit_test', methods=['POST'])
def submit_test():
    """Handle test submission and calculate scores."""
    submitted_answers = request.form.to_dict()
    correct_answers = {
        "technical_0": "Web development",
        "soft_0": "Practice active listening",
        "managerial_0": "Clear goal-setting",
    }

    # Calculate score
    score = sum(1 for qid, answer in submitted_answers.items() if correct_answers.get(qid) == answer)
    total_questions = len(correct_answers)
    percentage = (score / total_questions) * 100

    # Render score page
    return render_template(
        'score.html',
        score={
            "score": score,
            "total": total_questions,
            "percentage": round(percentage, 2)
        }
    )

if __name__ == '__main__':
    app.run(debug=True)
'''


'''   better than others 
from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from random import shuffle, choice

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate realistic MCQs
def generate_mcqs(extracted_skills):
    mcqs = {"technical": [], "soft": [], "managerial": []}
    question_templates = {
        "technical": [
            {"template": "What is {skill} primarily used for?", "options": ["Web development", "Data analysis", "System automation", "Game development"]},
            {"template": "Which of the following is a key feature of {skill}?", "options": ["Ease of use", "Scalability", "Interoperability", "High performance"]},
        ],
        "soft": [
            {"template": "Which is the most effective way to improve {skill}?", "options": ["Practice active listening", "Avoid difficult situations", "Rely solely on intuition", "Use formal language"]},
            {"template": "What does {skill} primarily help achieve in a team setting?", "options": ["Better collaboration", "Individual success", "Conflict avoidance", "Faster task completion"]},
        ],
        "managerial": [
            {"template": "How does {skill} contribute to a project's success?", "options": ["Clear goal-setting", "Reducing workload", "Improvisation", "Avoiding delegation"]},
            {"template": "Which of the following describes {skill}?", "options": ["Effective decision-making", "Strict adherence to rules", "Avoiding feedback", "Leading without communication"]},
        ],
    }

    for category, skills in extracted_skills.items():
        if category not in question_templates:
            continue
        for skill in skills:
            templates = question_templates[category]
            for template_data in templates:
                question_text = template_data["template"].format(skill=skill.capitalize())
                options = template_data["options"]
                correct_answer = options[0]
                shuffle(options)

                mcqs[category].append({
                    "question": question_text,
                    "options": options,
                    "answer": correct_answer
                })

    for questions in mcqs.values():
        shuffle(questions)

    return mcqs

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part provided."}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            mcqs = generate_mcqs(skills)
            return jsonify({"skills": skills, "mcqs": mcqs})
        except Exception as e:
            return jsonify({"error": f"Error extracting skills: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type."}), 400

@app.route('/start_test', methods=['GET'])
def start_test():
    extracted_skills = {
        'technical': ['python', 'java'],
        'soft': ['communication'],
        'managerial': ['leadership']
    }
    mcqs = generate_mcqs(extracted_skills)
    return render_template('test.html', mcqs=mcqs, duration=30)

@app.route('/submit_test', methods=['POST'])
def submit_test():
    submitted_answers = request.form
    correct_answers = {
        "technical_0": "Web development",
        "soft_0": "Practice active listening",
        "managerial_0": "Clear goal-setting",
    }

    score = sum(1 for qid, answer in submitted_answers.items() if correct_answers.get(qid) == answer)
    total_questions = len(correct_answers)
    percentage = (score / total_questions) * 100

    category_scores = {"technical": 1, "soft": 1, "managerial": 1}  # Mock category scores
    return render_template('score.html', score={"score": score, "total": total_questions, "percentage": round(percentage, 2), "category_scores": category_scores})

if __name__ == '__main__':
    app.run(debug=True)
'''





'''from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import pipeline
from random import shuffle

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Predefined skill ontology (for eligibility check)
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate MCQs using Hugging Face Transformers
def generate_mcqs(extracted_skills):
    mcqs = {"technical": [], "soft": [], "managerial": []}
    question_generator = pipeline("text-generation", model="gpt2")  # Replace with a more specific model if needed
    question_distribution = {
        "technical": 12,  # 60% of 20 questions
        "soft": 4,        # 20% of 20 questions
        "managerial": 4   # 20% of 20 questions
    }

    for category, count in question_distribution.items():
        skills = extracted_skills.get(category, [])
        for skill in skills[:count]:
            prompt = f"Create a multiple-choice question for the skill: {skill}."
            result = question_generator(prompt, max_length=50, num_return_sequences=1)
            question_text = result[0]["generated_text"]

            mcqs[category].append({
                "id": f"{category}_{len(mcqs[category])}",
                "question": question_text,
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Option A"  # Mock answer for simplicity
            })

    # Shuffle questions in each category
    for questions in mcqs.values():
        shuffle(questions)

    return mcqs

# Home route
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

# Route to upload resume and extract skills
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, extract skills, and render results."""
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)
            eligible = check_eligibility(skills)

            return render_template('skills.html', skills=skills, accuracy=accuracy, eligible=eligible)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Route to start the test and display MCQs
@app.route('/start_test', methods=['GET'])
def start_test():
    """Generate MCQs and display the test page."""
    extracted_skills = {
        'technical': ['python', 'java', 'sql'],
        'soft': ['communication', 'teamwork'],
        'managerial': ['leadership', 'strategic planning']
    }
    mcqs = generate_mcqs(extracted_skills)
    return render_template('test.html', mcqs=mcqs, duration=30)

# Route to submit test and calculate the score
@app.route('/submit_test', methods=['POST'])
def submit_test():
    """Calculate and display the score."""
    submitted_answers = request.form
    correct_answers = {
        "technical_python": "Option A",  # Mock correct answers
        "technical_java": "Option B",
        # Add more correct answers as needed
    }

    score = 0
    for qid, answer in submitted_answers.items():
        if correct_answers.get(qid) == answer:
            score += 1

    total_questions = len(correct_answers)
    percentage = (score / total_questions) * 100

    return render_template('score.html', score={"score": score, "total": total_questions, "percentage": round(percentage, 2)})

# Function to calculate accuracy for extracted skills
def calculate_accuracy(extracted_skills):
    predefined_skills = [skill.lower() for skill in SKILL_ONTOLOGY["technical"] + SKILL_ONTOLOGY["soft"] + SKILL_ONTOLOGY["managerial"]]
    all_extracted_skills = [skill.lower() for skills in extracted_skills.values() for skill in skills]

    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    return round((correct_skills / total_skills) * 100, 2) if total_skills > 0 else 0

# Function to check eligibility
def check_eligibility(extracted_skills):
    technical_skills = SKILL_ONTOLOGY.get('technical', [])
    matching_skills = set(extracted_skills.get('technical', [])).intersection(technical_skills)
    return len(matching_skills) >= 2

# Eligibility route to show eligibility status
@app.route('/eligibility')
def eligibility_check():
    """Render the eligibility check page."""
    eligible = False
    extracted_skills = {
        'technical': ['python', 'java', 'html'],
        'soft': ['communication', 'teamwork'],
        'managerial': ['leadership', 'management']
    }
    eligible = check_eligibility(extracted_skills)
    return render_template('eligibility.html', eligible=eligible)

# Instructions route for test
@app.route('/instructions')
def instructions():
    """Render the instructions page."""
    return render_template('instructions.html')

if __name__ == '__main__':
    app.run(debug=True) '''







"""from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score
import requests

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Predefined skill ontology (for eligibility check)
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

# Trivia API endpoint
TRIVIA_API_URL = "https://opentdb.com/api.php"

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    
    return render_template('index.html')

# Function to calculate accuracy for extracted skills
def calculate_accuracy(extracted_skills):
    predefined_skills = [skill.lower() for skill in SKILL_ONTOLOGY["technical"] + SKILL_ONTOLOGY["soft"] + SKILL_ONTOLOGY["managerial"]]
    all_extracted_skills = [skill.lower() for skills in extracted_skills.values() for skill in skills]

    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    return round((correct_skills / total_skills) * 100, 2) if total_skills > 0 else 0

# Updated eligibility logic to allow at least 2 skills to match
def check_eligibility(extracted_skills):
    total_matching_skills = 0

    # Check technical skills category for at least 2 matching skills
    technical_skills = SKILL_ONTOLOGY.get('technical', [])
    matching_technical_skills = set(extracted_skills.get('technical', []))
    total_matching_skills += len(matching_technical_skills)

    # If at least 2 skills match in the technical category, they are eligible
    if total_matching_skills >= 2:
        return True
    else:
        return False

# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)
            eligible = check_eligibility(skills)

            return render_template('skills.html', skills=skills, accuracy=accuracy, eligible=eligible)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Generate MCQs using Open Trivia Database API
def generate_mcqs(extracted_skills):
    mcqs = {"technical": [], "soft": [], "managerial": []}
    api_url = "https://opentdb.com/api.php"

    total_questions = 20
    question_distribution = {
        "technical": int(total_questions * 0.6),
        "soft": int(total_questions * 0.2),
        "managerial": int(total_questions * 0.2)
    }

    for category, count in question_distribution.items():
        response = requests.get(api_url, params={
            "amount": count,
            "type": "multiple",
            "difficulty": "medium"
        })
        if response.status_code == 200:
            questions = response.json().get('results', [])
            for question in questions:
                mcqs[category].append({
                    "id": f"{category}_{len(mcqs[category])}",
                    "question": question['question'],
                    "options": question['incorrect_answers'] + [question['correct_answer']],
                    "answer": question['correct_answer']
                })

    # Shuffle the options for each question
    for category_questions in mcqs.values():
        for question in category_questions:
            question['options'] = sorted(question['options'])

    return mcqs

# Route to start the test and display MCQs
@app.route('/start_test', methods=['GET'])
def start_test():
    
    extracted_skills = {
        'technical': ['python', 'java'],
        'soft': ['communication'],
        'managerial': ['leadership']
    }
    mcqs = generate_mcqs(extracted_skills)
    return render_template('test.html', mcqs=mcqs, duration=30)

# Submit test and calculate score
@app.route('/submit_test', methods=['POST'])
def submit_test():
    
    submitted_answers = request.json.get('answers', {})
    correct_answers = request.json.get('correct_answers', {})

    score = 0
    for category, questions in correct_answers.items():
        for qid, correct_answer in questions.items():
            if submitted_answers.get(qid) == correct_answer:
                score += 1

    return jsonify({"score": score, "total": len(correct_answers)})

# Test the model with sample test data
def test_model_with_test_data():
    
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills.get(category, []))
            predicted_skills = set(extracted_skills.get(category, []))

            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    accuracy = round((correct_predictions / total_skills) * 100, 2) if total_skills > 0 else 0
    precision = precision_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                                 [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    recall = recall_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                           [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    f1 = f1_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                  [1] * correct_predictions + [0] * (total_skills - correct_predictions))

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
   
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

# Eligibility route to show eligibility status
@app.route('/eligibility')
def eligibility_check():
   
    eligible = False
    # Example: Assume extracted_skills are passed through context or session
    extracted_skills = {
        'technical': ['python', 'java', 'html'],
        'soft': ['communication', 'teamwork'],
        'managerial': ['leadership', 'management']
    }
    eligible = check_eligibility(extracted_skills)
    return render_template('eligibility.html', eligible=eligible)

@app.route('/mcq')
def mcq():
    
    return render_template('mcq.html')

@app.route('/instructions')
def instructions():
    
    return render_template('instructions.html')

if __name__ == '__main__':
    app.run(debug=True)"""










'''from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Predefined skill ontology (for eligibility check)
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

# Function to calculate accuracy for extracted skills
def calculate_accuracy(extracted_skills):
    predefined_skills = [skill.lower() for skill in SKILL_ONTOLOGY["technical"] + SKILL_ONTOLOGY["soft"] + SKILL_ONTOLOGY["managerial"]]
    all_extracted_skills = [skill.lower() for skills in extracted_skills.values() for skill in skills]

    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    return round((correct_skills / total_skills) * 100, 2) if total_skills > 0 else 0

# Updated eligibility logic to allow at least 2 skills to match
def check_eligibility(extracted_skills):
    total_matching_skills = 0

    # Check technical skills category for at least 2 matching skills
    technical_skills = SKILL_ONTOLOGY.get('technical', [])
    matching_technical_skills = set(extracted_skills.get('technical', []))
    total_matching_skills += len(matching_technical_skills)

    # If at least 2 skills match in the technical category, they are eligible
    if total_matching_skills >= 2:
        return True
    else:
        return False

# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, extract skills, and render results."""
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)
            eligible = check_eligibility(skills)

            return render_template('skills.html', skills=skills, accuracy=accuracy, eligible=eligible)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills.get(category, []))
            predicted_skills = set(extracted_skills.get(category, []))

            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    accuracy = round((correct_predictions / total_skills) * 100, 2) if total_skills > 0 else 0
    precision = precision_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                                 [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    recall = recall_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                           [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    f1 = f1_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                  [1] * correct_predictions + [0] * (total_skills - correct_predictions))

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

# Eligibility route to show eligibility status
@app.route('/eligibility')
def eligibility_check():
    """Render the eligibility check page."""
    eligible = False
    # Example: Assume extracted_skills are passed through context or session
    extracted_skills = {
        'technical': ['python', 'java', 'html'],
        'soft': ['communication', 'teamwork'],
        'managerial': ['leadership', 'management']
    }
    eligible = check_eligibility(extracted_skills)
    return render_template('eligibility.html', eligible=eligible)

@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

@app.route('/instructions')
def instructions():
    """Display instructions for the test."""
    return render_template('instructions.html')

if __name__ == '__main__':
    app.run(debug=True)'''








'''from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Predefined skill ontology (for eligibility check)
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

# Function to calculate accuracy for extracted skills
def calculate_accuracy(extracted_skills):
    predefined_skills = ["java", "sql", "django", "python", "mysql", "javascript", "git", "html", "css", "communication", "problem-solving", "teamwork", "management"]
    all_extracted_skills = [skill for skills in extracted_skills.values() for skill in skills]
    
    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    if total_skills == 0:  # Avoid division by zero
        return 0
    accuracy = (correct_skills / total_skills) * 100
    return round(accuracy, 2)

# Function to check eligibility based on 30% match in each skill category
# Function to check eligibility based on 30% match in technical skills
def check_eligibility(extracted_skills):
    eligible = True

    # Check technical skills category for 30% match
    technical_skills = SKILL_ONTOLOGY.get('technical', [])
    matching_technical_skills = set(extracted_skills.get('technical', []))
    technical_match_percentage = (len(matching_technical_skills) / len(technical_skills)) * 100

    # If technical skills match is less than 30%, set eligible to False
    if technical_match_percentage < 30:
        eligible = False
    
    # Check other categories (soft and managerial) without the 30% match restriction
    if eligible:  # If still eligible, check soft and managerial skills
        for category in ['soft', 'managerial']:
            skills = SKILL_ONTOLOGY.get(category, [])
            matching_skills = set(extracted_skills.get(category, []))
            match_percentage = (len(matching_skills) / len(skills)) * 100

            # If any category has a match below 30%, eligibility fails
            if match_percentage < 30:
                eligible = False
                break
    
    return eligible
'''


'''def check_eligibility(extracted_skills):
    eligible = True
    for category, skills in SKILL_ONTOLOGY.items():
        # Normalize skills to lowercase for consistent matching
        extracted_category_skills = {skill.lower() for skill in extracted_skills.get(category, [])}
        predefined_category_skills = {skill.lower() for skill in skills}

        # Calculate the number of matching skills
        matching_skills = extracted_category_skills.intersection(predefined_category_skills)

        # Calculate the match percentage
        match_percentage = (len(matching_skills) / len(predefined_category_skills)) * 100
        print(f"Category: {category}, Matching Skills: {matching_skills}, Match Percentage: {match_percentage}%")
        
        if match_percentage < 30:
            eligible = False
            break

    return eligible'''
'''
# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, extract skills, and render results."""
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)

            # Check eligibility
            eligible = check_eligibility(skills)

            return render_template('skills.html', skills=skills, accuracy=accuracy, eligible=eligible)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills.get(category, []))
            predicted_skills = set(extracted_skills.get(category, []))

            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    accuracy = round((correct_predictions / total_skills) * 100, 2) if total_skills > 0 else 0
    precision = precision_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                                 [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    recall = recall_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                           [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    f1 = f1_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                  [1] * correct_predictions + [0] * (total_skills - correct_predictions))

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

@app.route('/eligibility')
def eligibility_check():
    """Render the eligibility check page."""
    # Just return a placeholder page or logic based on your requirements
    return render_template('eligibility.html')

if __name__ == '__main__':
    app.run(debug=True)'''






'''from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

# Function to calculate accuracy for extracted skills
def calculate_accuracy(extracted_skills):
    predefined_skills = ["java", "sql", "django", "python", "mysql", "javascript", "git", "html", "css", "communication", "problem-solving", "teamwork", "management"]
    all_extracted_skills = [skill for skills in extracted_skills.values() for skill in skills]
    
    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    if total_skills == 0:  # Avoid division by zero
        return 0
    accuracy = (correct_skills / total_skills) * 100
    return round(accuracy, 2)

# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, extract skills, and render results."""
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)
            return render_template('skills.html', skills=skills, accuracy=accuracy)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')



# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills.get(category, []))
            predicted_skills = set(extracted_skills.get(category, []))

            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    accuracy = round((correct_predictions / total_skills) * 100,2) if total_skills > 0 else 0
    precision = precision_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                                 [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    recall = recall_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                           [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    f1 = f1_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                  [1] * correct_predictions + [0] * (total_skills - correct_predictions))

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

if __name__ == '__main__':
    app.run(debug=True)'''






'''from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

# Function to calculate accuracy for extracted skills
def calculate_accuracy(extracted_skills):
    """Calculate the accuracy of extracted skills compared to a predefined set."""
    predefined_skills = ["java", "sql", "django", "python","mysql","javascript","git","html","css", "communication", "problem-solving", "teamwork", "management"]
    all_extracted_skills = [skill for skills in extracted_skills.values() for skill in skills]
    
    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    if total_skills == 0:  # Avoid division by zero
        return 0
    accuracy = (correct_skills / total_skills) * 100
    return round(accuracy, 2)

# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, extract skills, and render results."""
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract skills from the file
        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)  # Calculate accuracy of skill extraction
            
            return render_template('skills.html', skills=skills, accuracy=accuracy)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Function to calculate precision, recall, and F1-score
def calculate_metrics(extracted_skills, ground_truth):
    """Calculate precision, recall, and F1 score for extracted skills."""
    all_true_skills = []
    all_predicted_skills = []

    for resume, skills in ground_truth.items():
        predicted_skills = extracted_skills.get(resume, {})
        for category in ["technical", "soft", "managerial"]:
            true_skills = skills.get(category, [])
            predicted = predicted_skills.get(category, [])

            for skill in true_skills:
                all_true_skills.append(1)
                all_predicted_skills.append(1 if skill in predicted else 0)
            for skill in predicted:
                if skill not in true_skills:
                    all_true_skills.append(0)
                    all_predicted_skills.append(1)

    accuracy = sum(1 for true, pred in zip(all_true_skills, all_predicted_skills) if true == pred) / len(all_true_skills)
    precision = precision_score(all_true_skills, all_predicted_skills)
    recall = recall_score(all_true_skills, all_predicted_skills)
    f1 = f1_score(all_true_skills, all_predicted_skills)

    return accuracy, precision, recall, f1


# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        # If ground_truth_skills is a list, iterate over all categories
        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills)  # Directly treat it as a list of skills
            predicted_skills = set(extracted_skills.get(category, []))  # Default to empty list if category not found

            # Count correct matches
            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    # Ensure we do not divide by zero
    if total_skills == 0:
        accuracy = 0
    else:
        accuracy = (correct_predictions / total_skills) * 100

    precision = 0.9  # Placeholder for precision calculation
    recall = 0.85  # Placeholder for recall calculation
    f1 = 0.87  # Placeholder for f1-score calculation

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

# MCQ page
@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

if __name__ == '__main__':
    app.run(debug=True)'''

'''from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy, calculate_accuracy
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, extract skills, and render results."""
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract skills from the file
        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills, test_data)  # Calculate accuracy of skill extraction
            
            return render_template('skills.html', skills=skills, accuracy=accuracy)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_resumes = len(test_data)
    total_ground_truth_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        # Assuming the resume_file is a PDF file in TEST_DATA_FOLDER
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        # Compare extracted skills with ground truth for each category
        for category in ['technical', 'soft', 'managerial']:
            true_skills = ground_truth_skills.get(category, [])
            predicted_skills = extracted_skills.get(category, [])
            
            correct_predictions += len(set(true_skills).intersection(set(predicted_skills)))
            total_ground_truth_skills += len(true_skills)

    # Avoid division by zero
    if total_ground_truth_skills == 0:
        accuracy = 0
    else:
        accuracy = (correct_predictions / total_ground_truth_skills) * 100

    precision = 0.9  # Placeholder for actual precision calculation
    recall = 0.85  # Placeholder for actual recall calculation
    f1 = 0.87  # Placeholder for actual f1-score calculation

    return accuracy, precision, recall, f1'''

'''def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_resumes = len(test_data)

    for resume_file, ground_truth_skills in test_data.items():
        # Assuming the resume_file is a PDF file in TEST_DATA_FOLDER
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        # Comparing extracted skills with the ground truth
        correct_predictions += len(set(extracted_skills['technical']).intersection(set(ground_truth_skills)))

    accuracy = (correct_predictions / total_resumes) * 100
    precision = 0.9  # Placeholder for actual precision calculation
    recall = 0.85  # Placeholder for actual recall calculation
    f1 = 0.87  # Placeholder for actual f1-score calculation

    return accuracy, precision, recall, f1'''

'''from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

# Function to calculate accuracy for extracted skills
def calculate_metrics(extracted_skills, ground_truth):
    """Calculate precision, recall, and F1 score for extracted skills."""
    all_true_skills = []
    all_predicted_skills = []

    for resume, skills in ground_truth.items():
        predicted_skills = extracted_skills.get(resume, {})
        for category in ["technical", "soft", "managerial"]:
            true_skills = skills.get(category, [])
            predicted = predicted_skills.get(category, [])

            for skill in true_skills:
                all_true_skills.append(1)
                all_predicted_skills.append(1 if skill in predicted else 0)
            for skill in predicted:
                if skill not in true_skills:
                    all_true_skills.append(0)
                    all_predicted_skills.append(1)

    # Calculate metrics using sklearn
    precision = precision_score(all_true_skills, all_predicted_skills)
    recall = recall_score(all_true_skills, all_predicted_skills)
    f1 = f1_score(all_true_skills, all_predicted_skills)

    accuracy = sum(1 for true, pred in zip(all_true_skills, all_predicted_skills) if true == pred) / len(all_true_skills)
    return accuracy, precision, recall, f1


# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, extract skills, and render results."""
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract skills from the file
        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)  # Calculate accuracy of skill extraction
            
            return render_template('skills.html', skills=skills, accuracy=accuracy)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Function to calculate precision, recall, and F1-score
def calculate_metrics(extracted_skills, ground_truth):
    """Calculate precision, recall, and F1 score for extracted skills."""
    all_true_skills = []
    all_predicted_skills = []

    for resume, skills in ground_truth.items():
        predicted_skills = extracted_skills.get(resume, {})
        for category in ["technical", "soft", "managerial"]:
            true_skills = skills.get(category, [])
            predicted = predicted_skills.get(category, [])

            for skill in true_skills:
                all_true_skills.append(1)
                all_predicted_skills.append(1 if skill in predicted else 0)
            for skill in predicted:
                if skill not in true_skills:
                    all_true_skills.append(0)
                    all_predicted_skills.append(1)

    # Calculate metrics using sklearn
    precision = precision_score(all_true_skills, all_predicted_skills)
    recall = recall_score(all_true_skills, all_predicted_skills)
    f1 = f1_score(all_true_skills, all_predicted_skills)

    accuracy = sum(1 for true, pred in zip(all_true_skills, all_predicted_skills) if true == pred) / len(all_true_skills)
    return accuracy, precision, recall, f1


# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_resumes = len(test_data)

    for resume_file in os.listdir(TEST_DATA_FOLDER):
        if resume_file.endswith('.pdf'):
            resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
            extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

            expected_skills = test_data.get(resume_file, [])
            if not expected_skills:
                continue

            correct_predictions += len(set(extracted_skills['technical']).intersection(set(expected_skills)))

    accuracy = correct_predictions / total_resumes
    precision = 0.9  # Placeholder
    recall = 0.85  # Placeholder
    f1 = 0.87  # Placeholder

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

# MCQ page
@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

if __name__ == '__main__':
    app.run(debug=True)'''







'''from flask import Flask, render_template, request, jsonify
import os
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
import json
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Path to your test data folder
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')

# Load the ground truth data (you can adjust the path based on your project structure)
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

def calculate_accuracy(extracted_skills):
    # Example predefined skill set
    predefined_skills = ["python", "java", "sql", "communication", "teamwork", "leadership"]
    
    # Flatten the list of extracted skills (if it's a dictionary, which it is in this case)
    all_extracted_skills = [skill for skills in extracted_skills.values() for skill in skills]
    
    # Count how many skills are correct (match between predefined and extracted skills)
    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)
    
    # If there are no predefined skills to compare, return 0 accuracy
    if total_skills == 0:
        return 0  # Avoid division by zero
    
    # Calculate accuracy percentage
    accuracy = (correct_skills / total_skills) * 100
    return round(accuracy, 2)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract skills from the resume
        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            
            # Calculate accuracy of the skill extraction
            accuracy = calculate_accuracy(skills)  # Pass extracted skills to calculate_accuracy
            
            # Pass skills and accuracy to the template
            return render_template('skills.html', skills=skills, accuracy=accuracy)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')
    
def calculate_metrics(extracted_skills, ground_truth):
    """Calculate accuracy, precision, recall, and F1 score based on predefined skills."""
    all_true_skills = []
    all_predicted_skills = []
    
    # Compare extracted skills with ground truth
    for resume, skills in ground_truth.items():
        predicted_skills = extracted_skills.get(resume, {})
        
        for category in ["technical", "soft", "managerial"]:
            true_skills = skills.get(category, [])
            predicted = predicted_skills.get(category, [])
            
            # Append true and predicted skills for metrics calculation
            for skill in true_skills:
                all_true_skills.append(1)
                all_predicted_skills.append(1 if skill in predicted else 0)
            for skill in predicted:
                if skill not in true_skills:
                    all_true_skills.append(0)
                    all_predicted_skills.append(1)
    
    # Calculate the metrics
    accuracy = sum([1 for true, pred in zip(all_true_skills, all_predicted_skills) if true == pred]) / len(all_true_skills)
    precision = precision_score(all_true_skills, all_predicted_skills)
    recall = recall_score(all_true_skills, all_predicted_skills)
    f1 = f1_score(all_true_skills, all_predicted_skills)

    return accuracy, precision, recall, f1

def test_model_with_test_data():
    correct_predictions = 0
    total_resumes = len(test_data)  # Assuming 'test_data' is loaded properly

    for resume_file in os.listdir(TEST_DATA_FOLDER):
        if resume_file.endswith('.pdf'):
            # Extract skills from the resume
            resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
            extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))
            
            # Compare with expected skills
            expected_skills = test_data.get(resume_file, [])
            if not expected_skills:
                continue  # Skip if no expected skills are found for this file

            # Calculate accuracy by comparing the extracted and expected skills
            correct_predictions += len(set(extracted_skills['technical']).intersection(set(expected_skills)))

    accuracy = correct_predictions / total_resumes  # Accuracy as a float
    precision = 0.9  # Replace with actual precision calculation
    recall = 0.85  # Replace with actual recall calculation
    f1 = 0.87  # Replace with actual F1 score calculation

    return accuracy, precision, recall, f1  # Return as a tuple


@app.route('/test', methods=['GET'])
def test():
    accuracy, precision, recall, f1 = test_model_with_test_data()
    
    # Pass all metrics to the template
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/mcq')
def mcq():
    return render_template('mcq.html')

if __name__ == '__main__':
    app.run(debug=True)'''

'''@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract skills from the resume
        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            return render_template('skills.html', skills=skills)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')'''
'''
@app.route('/test_accuracy')
def test_accuracy():
    test_folder = os.path.join(app.root_path, 'test_data')
    expected_skills_file = os.path.join(test_folder, 'test_data.json')

    # Calculate accuracy
    accuracy = calculate_accuracy(test_folder, expected_skills_file)
    return render_template('skills.html', accuracy=accuracy, skills={})

if __name__ == '__main__':
    app.run(debug=True)
'''
'''from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from skill_extraction import extract_skills

app = Flask(__name__)

# Upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Extract skills from the resume
        try:
            skills, experience, projects = extract_skills(filename)
            return render_template('skills.html', skills=skills, experience=experience, projects=projects)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')


@app.route('/mcq')
def mcq():
    return render_template('mcq.html')


if __name__ == '__main__':
    app.run(debug=True)
'''