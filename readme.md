GridSense AI – AI-Powered Smart Grid Control Room

An end-to-end AI decision support system for power transmission networks that predicts technical losses, identifies high-risk assets, evaluates investment decisions, and simulates autonomous grid optimization.
The system combines Machine Learning, Graph Neural Networks, Forecasting, and Reinforcement Learning into a unified control room dashboard built using Streamlit.

Problem Statement
Power transmission networks face:
  High technical losses
  Overloaded corridors
  Lack of real-time risk visibility
  Reactive maintenance decisions
  Poor investment prioritization
  India’s national target is to reduce transmission losses below 2% by 2030.
  Achieving this requires predictive, intelligent grid management.
  
Solution Overview
  National Grid Optimizer provides a unified AI control room that enables:
  Line-level loss prediction
  Risk-based prioritization
  ROI-driven infrastructure planning
  Weather-based dynamic capacity estimation
  Network-level risk analysis
  Autonomous battery dispatch simulation
  
Key Features
1. National View – Grid Health Monitoring
Average network loss
Critical corridors (>2% loss)
Live transmission network visualization
Overall grid status indicator
2. AI Risk Engine (ML)
Predicts expected technical loss for each transmission line.
Outputs:
Highest risk lines
Predicted loss percentage
High-risk corridor identification
Model: Regression-based loss prediction trained on synthetic grid features.
3. ROI Planner – Investment Optimizer
Evaluates whether infrastructure upgrades are financially viable.
Simulation includes:
Conductor upgrade / intervention cost
Annual savings
Payback period
10-year financial impact
Recommended action
Helps prioritize cost-effective grid upgrades.
4. Load & DLR Forecast – Dynamic Line Rating
Estimates transmission capacity based on weather conditions.
Inputs:
Ambient temperature
Wind speed
Outputs:
Capacity gain (%)
Estimated loss reduction
Weather-adjusted capacity visualization
5. Network Intelligence – Triple-Threat GNN
Graph Neural Network evaluates risk at each bus across three dimensions:
Technical Loss Risk
Commercial Loss Risk
Stability Risk
Provides:
Per-bus risk scores
Combined risk ranking
High-risk node identification
6. Grid Autopilot – Reinforcement Learning
Simulates autonomous grid optimization.
The RL agent:
Dispatches battery storage
Minimizes technical losses
Improves grid stability
Outputs:
Reward convergence
Battery dispatch over time
Estimated performance improvement


Tech Stack

Frontend
  Streamlit
Data & Processing
  Python
  Pandas
  NumPy
Machine Learning
  Scikit-learn (Risk Model)
  PyTorch / PyTorch Geometric (GNN)
Forecasting
  Darts (for time-series models)
Optimization
  Custom simulation logic
Visualization
  Plotly
  Network graphs

  
Project Structure
project/
│
├── app.py
├── models/
│   ├── risk_model.pkl
│   ├── gnn_model.pt
│   ├── rl_agent.pt
│   └── forecaster.pkl
│
├── data/
├── train/
│   ├── train_risk_model.py
│   ├── train_gnn.py
│   ├── train_rl.py
│   └── train_forecaster.py
│
└── utils/


Installation

Clone the repository
git clone https://github.com/bhuvikamehta/Grid-Sense-AI-CreaTech.git
cd Grid-Sense-AI-CreaTech

Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

Install dependencies
pip install -r requirements.txt

Run the Application
streamlit run app.py

The dashboard will open at:
http://localhost:8501

Model Training (Optional)
If models are not available:
python train/train_risk_model.py
python train/train_gnn.py
python train/train_rl.py
python train/train_forecaster.py


Demo Workflow
  View overall grid health (National View)
  Identify risky lines (AI Risk Engine)
  Simulate financial intervention (ROI Planner)
  Adjust weather for dynamic capacity (DLR)
  Analyze network risks (GNN)
  Run autonomous optimization (RL)
