# pca-news-virality-dashboard

This interactive analytics dashboard visualizes **top viral news on Twitter**, enables rich exploration through filters, interactive charts, PCA-based dimensionality reduction, and clean UI components built using **Vue 3**, **D3.js**, **Flask**, **Dash**, **TailwindCSS**.

### Features
#### Frontend (Vue 3 + D3.js)
- Interactive data table for exploring viral news items
- Dataset size selector (Top N records)
- Column visibility toggles
- Dynamic D3 word cloud visualization
- Vue 3 Composition API for modular and reactive UI
- TailwindCSS and Flowbite for responsive design
- Clean component-based architecture

#### Backend (Flask + Dash + PCA)
- Flask backend serving the Vue application
- API endpoint returning dataset in JSON format
- Integrated Dash application 
- PCA visualization of news metadata such as source, domain, type, and tweet count
- Machine learning preprocessing using:
  - StandardScaler  
  - LabelEncoder  
  - OneHotEncoder  
  - PCA

### Installation and Setup
0. Requirements
python system-level installation

1. Navigate to Project Directory
cd <path-to-project>;

2. Setup the Python Virtual Environment
	2.1 Create Virtual Environment
	python -m venv venv
	
	2.2 Activate Virtual Environment			
	[Windows] venv/Scripts/activate
	[MacOS/Linux] source venv/bin/activate

	2.3 Update pip
	python -m pip install --upgrade pip

	2.4 Install Dependencies
	pip install -r requirements.txt

3. Start Flask's Local Development Server [Port: 8080]
python app.py 

4. Integration
Add 'localhost:8080' as href to the link
Note : In app.py file, Change Port Number - 8080 if the port is already busy with some other service

