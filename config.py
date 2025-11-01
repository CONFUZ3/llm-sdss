"""Configuration constants and styling for the application."""

# Color scheme for the app
COLOR_PRIMARY = "#4A90E2"   # Primary blue
COLOR_SECONDARY = "#29B6F6" # Light blue
COLOR_SUCCESS = "#66BB6A"   # Green
COLOR_WARNING = "#FFA726"   # Orange
COLOR_DANGER = "#EF5350"    # Red
COLOR_BACKGROUND = "#F5F7F9" # Light grayish blue
COLOR_TEXT = "#2C3E50"      # Dark blue-gray


def load_custom_css():
    """Return custom CSS for Streamlit styling."""
    return '''
    <style>
    .main {
        background-color: #F5F7F9;
        color: #2C3E50;
    }
    .stButton > button {
        color: white;
        background-color: #4A90E2;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2979B5;
    }
    .stProgress > div > div {
        background-color: #4A90E2;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .stAlert > div {
        border-radius: 4px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    .stSidebar .sidebar-content {
        background-color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        border-right: 1px solid #E0E0E0;
        border-left: 1px solid #E0E0E0;
        border-top: 1px solid #E0E0E0;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A90E2;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .dataframe {
        font-family: 'Source Sans Pro', sans-serif;
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th {
        background-color: #4A90E2;
        color: white;
        text-align: left;
        padding: 12px;
    }
    .dataframe td {
        padding: 12px;
        border-bottom: 1px solid #E0E0E0;
    }
    .dataframe tr:nth-child(even) {
        background-color: #F8F9FA;
    }
    </style>
    '''

