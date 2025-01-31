import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Function to perform clustering
def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['Age', 'Income', 'Spending Score']])
    return data, kmeans

# UI for user input
def user_input():
    st.title("Customer Segmentation")
    st.write("Please input customer data.")

    # Create input fields for customer data
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income (in thousands)", min_value=10, max_value=200, value=50)
    spending_score = st.number_input("Spending Score (1 to 100)", min_value=1, max_value=100, value=50)

    # Create a submit button to process data
    submit_button = st.button("Submit")

    return age, income, spending_score, submit_button

def main():
    # Load your existing CSV file containing customer data
    # Change the path to the location where your CSV is saved
    customer_data = pd.read_csv("C:/Users/aravi/OneDrive/Desktop/codeClause/project 3/customer_data.csv")

    # Ensure that the CSV file has the necessary columns
    if not all(col in customer_data.columns for col in ['Customer ID', 'Age', 'Income', 'Spending Score']):
        st.error("CSV file must contain 'Customer ID', 'Age', 'Income', and 'Spending Score' columns.")
        return

    # Collect user input
    age, income, spending_score, submit_button = user_input()

    if submit_button:
        # Create a DataFrame with the input data
        new_data = pd.DataFrame([[age, income, spending_score]], columns=['Age', 'Income', 'Spending Score'])
        new_data['Customer ID'] = customer_data['Customer ID'].max() + 1  # Assign a new customer ID
        
        # Combine the user input data with the loaded dataset
        full_data = pd.concat([customer_data, new_data], ignore_index=True)
        
        # Apply K-Means clustering
        n_clusters = 3  # You can experiment with this number
        clustered_data, kmeans_model = perform_clustering(full_data, n_clusters)

        # Display the clustered data
        st.write("Clustered Data:")
        st.write(clustered_data)

        # Visualize the clustering result
        st.subheader("Customer Segmentation Visualization")
        sns.scatterplot(x='Income', y='Spending Score', hue='Cluster', data=clustered_data, palette='Set1')
        plt.title('Customer Segments')
        st.pyplot(plt)

# Run the Streamlit app
if __name__ == "__main__":
    main()


# use this code to run the streamlit app
# python -m streamlit run "c:/Users/aravi/OneDrive/Desktop/codeClause/project 3/customer_segmentation.py"
