import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
import json

nltk.download('punkt')
nltk.download('wordnet')


class JobPortal:
    def __init__(self, name, url):
        self.name = name
        self.url = url


class JobSearchAI:
    def __init__(self):
        self.job_portals = [
            JobPortal("Indeed", "https://www.indeed.com"),
            JobPortal("Glassdoor", "https://www.glassdoor.com"),
            JobPortal("LinkedIn", "https://www.linkedin.com")
        ]

        self.user_profile = {
            "name": "",
            "career_history": [],
            "skills": [],
            "interests": []
        }

        self.job_listings = []
        self.filtered_jobs = []
        self.recommended_jobs = []
        self.applications = []

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.nlp = spacy.load('en_core_web_sm')

        self.matcher = Matcher(self.nlp.vocab)
        self.patterns = [
            [{'LOWER': 'location'}],
            [{'LOWER': 'salary'}],
            [{'LOWER': 'company'}, {'LOWER': 'reputation'}],
            [{'LOWER': 'skills'}],
            [{'LOWER': 'career'}, {'LOWER': 'growth'}, {'LOWER': 'potential'}]
        ]

        self.interview_questions = {}
        self.preparation_materials = {}

        self.career_insights = {}

    def find_jobs(self, keywords):
        for portal in self.job_portals:
            response = requests.get(f"{portal.url}/jobs?q={keywords}")
            soup = BeautifulSoup(response.content, "html.parser")
            jobs = soup.find_all("div", class_="job")
            for job in jobs:
                title = job.find("h2").text.strip()
                company = job.find("span", class_="company").text.strip()
                location = job.find("span", class_="location").text.strip()
                description = job.find("div", class_="description").text.strip()
                salary = job.find("span", class_="salary")
                salary = salary.text.strip() if salary else ""

                self.job_listings.append({
                    "title": title,
                    "company": company,
                    "location": location,
                    "description": description,
                    "salary": salary
                })

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in self.stop_words]
        return " ".join(tokens)

    def analyze_job_descriptions(self):
        job_texts = [listing["description"] for listing in self.job_listings]
        preprocessed_texts = [self.preprocess_text(text) for text in job_texts]
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        for i, listing in enumerate(self.job_listings):
            listing["text"] = preprocessed_texts[i]
            listing["top_similar"] = similarity_matrix[i].argsort()[::-1][1:4].tolist()

    def filter_jobs(self):
        self.matcher.add("FilterLabel", None, *self.patterns)

        for listing in self.job_listings:
            doc = self.nlp(listing["text"])
            matches = self.matcher(doc)
            filtered_labels = [self.nlp.vocab.strings[label] for _, label, _ in matches]
            listing["filtered_labels"] = filtered_labels

        for listing in self.job_listings:
            labels = listing["filtered_labels"]
            filtered_labels = [label for label in labels if label in ["location", "salary", "company reputation", "skills", "career growth potential"]]
            listing["filtered_labels"] = filtered_labels

        preferred_labels = ["location", "salary"]
        self.filtered_jobs = [listing for listing in self.job_listings if any(label in preferred_labels for label in listing["filtered_labels"])]

    def recommend_jobs(self):
        user_skills = set(self.user_profile["skills"])
        user_interests = set(self.user_profile["interests"])

        for listing in self.filtered_jobs:
            job_skills = set(self.preprocess_text(listing["description"]).split())
            if len(job_skills.intersection(user_skills)) >= 2 or len(job_skills.intersection(user_interests)) >= 2:
                self.recommended_jobs.append(listing)

    def generate_resume(self):
        career_history = "\n- ".join(self.user_profile["career_history"])
        qualifications = "\n- ".join(self.user_profile["skills"])

        resume = f"Career History:\n- {career_history}\n\nQualifications:\n- {qualifications}"
        return resume

    def generate_interview_materials(self):
        with open("interview_questions.json", "r") as interview_questions_file:
            self.interview_questions = json.load(interview_questions_file)

        for job in self.recommended_jobs:
            company = job["company"]
            if company not in self.preparation_materials:
                self.preparation_materials[company] = []

            if company in self.interview_questions:
                self.preparation_materials[company].extend(self.interview_questions[company])

    def prepare_career_insights(self):
        news_articles = requests.get("https://api.google.com/news/articles")
        industry_reports = requests.get("https://api.google.com/industry/reports")
        social_media_trends = requests.get("https://api.google.com/social_media/trends")

        self.career_insights['news_articles'] = news_articles.json()
        self.career_insights['industry_reports'] = industry_reports.json()
        self.career_insights['social_media_trends'] = social_media_trends.json()

    def execute(self):
        keywords = input("Enter job keywords: ")
        self.find_jobs(keywords)

        self.user_profile["name"] = input("Enter your name: ")
        career_history = input("Enter your career history (comma separated): ")
        self.user_profile["career_history"] = [history.strip() for history in career_history.split(",")]

        skills = input("Enter your skills (comma separated): ")
        self.user_profile["skills"] = [skill.strip() for skill in skills.split(",")]

        interests = input("Enter your interests (comma separated): ")
        self.user_profile["interests"] = [interest.strip() for interest in interests.split(",")]

        self.analyze_job_descriptions()
        self.filter_jobs()

        if self.filtered_jobs:
            self.recommend_jobs()
            if self.recommended_jobs:
                resume = self.generate_resume()
                self.generate_interview_materials()
                self.prepare_career_insights()

                print("Recommended Jobs:")
                for i, job in enumerate(self.recommended_jobs):
                    print(f"{i+1}. {job['title']} at {job['company']}, {job['location']}")

                apply_job = input("Do you want to apply for a job? (y/n) ")
                if apply_job.lower() == "y":
                    job_number = int(input("Enter job number: "))
                    job = self.recommended_jobs[job_number-1]
                    application = {
                        "name": self.user_profile["name"],
                        "job_title": job["title"],
                        "company": job["company"],
                        "location": job["location"],
                        "resume": resume
                    }
                    self.applications.append(application)
                    print("Application submitted successfully!")

                company = input("Enter a company name to get interview materials: ")
                if company in self.preparation_materials:
                    interview_materials = self.preparation_materials[company]
                    print("Interview Materials:")
                    for i, material in enumerate(interview_materials):
                        print(f"{i+1}. {material}")
                else:
                    print("No interview materials available for the company.")

                print("Career Insights and Trends:")
                print(json.dumps(self.career_insights, indent=2))
            else:
                print("No recommended jobs available.")
        else:
            print("No jobs found for the given keywords.")


# Execute the program
if __name__ == "__main__":
    ai = JobSearchAI()
    ai.execute()