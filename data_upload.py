import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime
from database import (
    init_db, get_session, DataUpload, 
    import_csv_to_db, export_template,
    update_models
)

def data_upload_section(store_id):
    """
    Create a data upload section for store owners
    
    Parameters:
    -----------
    store_id : str
        The ID of the store uploading data
    """
    st.markdown("## 📤 Data Upload")
    st.write("Upload your store data to get personalized predictions and insights.")
    
    # Initialize database
    engine = init_db()
    
    # Create tabs for upload and history
    tab1, tab2, tab3 = st.tabs(["Upload Data", "Upload History", "Download Template"])
    
    # Upload Data Tab
    with tab1:
        st.write("Please upload your store data in CSV format.")
        st.write("Make sure your file has the following columns: date, category, daily_sales, daily_revenue, customer_footfall, margin, stock_level, promotion_active")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="store_data_upload")
        
        if uploaded_file is not None:
            # Read and display preview
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                # Check for required columns
                required_columns = ['date', 'category', 'daily_sales', 'daily_revenue', 
                                   'customer_footfall', 'margin', 'stock_level', 'promotion_active']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    # Validate data types
                    try:
                        # Check date format
                        pd.to_datetime(df['date'])
                        
                        # Process upload
                        if st.button("Process Upload"):
                            # Create a temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                                # Reset file pointer
                                uploaded_file.seek(0)
                                # Write to temp file
                                temp_file.write(uploaded_file.read())
                                temp_file_path = temp_file.name
                            
                            try:
                                # Create upload record
                                session = get_session(engine)
                                new_upload = DataUpload(
                                    store_id=store_id,
                                    filename=uploaded_file.name,
                                    upload_date=datetime.now()
                                )
                                session.add(new_upload)
                                session.commit()
                                upload_id = new_upload.id
                                session.close()
                                
                                # Import data
                                with st.spinner("Processing data..."):
                                    success, message = import_csv_to_db(
                                        temp_file_path, 
                                        store_id, 
                                        upload_id,
                                        engine
                                    )
                                
                                # Remove temp file
                                os.unlink(temp_file_path)
                                
                                if success:
                                    st.success(message)
                                    
                                    # Update prediction models
                                    with st.spinner("Updating prediction models..."):
                                        update_models(engine)
                                    
                                    st.success("Models updated successfully! Your dashboard will now show predictions based on your data.")
                                else:
                                    st.error(message)
                            
                            except Exception as e:
                                st.error(f"Error processing upload: {str(e)}")
                                # Clean up
                                if os.path.exists(temp_file_path):
                                    os.unlink(temp_file_path)
                    
                    except Exception as e:
                        st.error(f"Error validating data: {str(e)}")
                        st.write("Please make sure your date column is in YYYY-MM-DD format.")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Upload History Tab
    with tab2:
        st.write("View your previous data uploads")
        
        # Get upload history
        session = get_session(engine)
        uploads = session.query(DataUpload).filter_by(store_id=store_id).order_by(DataUpload.upload_date.desc()).all()
        
        if uploads:
            # Create a DataFrame for display
            uploads_data = []
            for upload in uploads:
                uploads_data.append({
                    "ID": upload.id,
                    "Filename": upload.filename,
                    "Upload Date": upload.upload_date.strftime("%Y-%m-%d %H:%M"),
                    "Processed": "✅" if upload.processed else "❌",
                    "Records": upload.record_count
                })
            
            uploads_df = pd.DataFrame(uploads_data)
            st.dataframe(uploads_df)
        else:
            st.info("No upload history found. Upload your first data file to get started!")
        
        session.close()
    
    # Download Template Tab
    with tab3:
        st.write("Download a template CSV file to help you format your data correctly.")
        
        if st.button("Download Template"):
            template_path = export_template()
            
            # Read the template file
            with open(template_path, "r") as file:
                template_content = file.read()
            
            # Provide download link
            st.download_button(
                label="Download Template CSV",
                data=template_content,
                file_name="data_upload_template.csv",
                mime="text/csv"
            )
            
            # Clean up
            os.remove(template_path)
        
        st.write("### Template Format Instructions")
        st.write("""
        Your data file should include the following columns:
        
        - **date**: Date in YYYY-MM-DD format
        - **category**: Product category name
        - **daily_sales**: Number of units sold
        - **daily_revenue**: Total revenue for the day in currency units
        - **customer_footfall**: Number of customers who visited the store
        - **margin**: Profit margin percentage
        - **stock_level**: Current inventory level in units
        - **promotion_active**: Whether a promotion was active (True/False)
        
        Each row should represent data for a specific category on a specific date.
        """)

def get_store_data(store_id, start_date=None, end_date=None):
    """
    Get data for a specific store from the database
    
    Parameters:
    -----------
    store_id : str
        The ID of the store
    start_date : datetime.date, optional
        Start date for filtering data
    end_date : datetime.date, optional
        End date for filtering data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing store data
    """
    engine = init_db()
    session = get_session(engine)
    
    # Build query
    query = session.query(DataUpload).filter_by(store_id=store_id, processed=True)
    
    # Check if any uploads exist
    uploads_exist = session.query(query.exists()).scalar()
    
    if not uploads_exist:
        session.close()
        return None
    
    # Import the StoreData model here to avoid circular imports
    from database import StoreData
    
    # Build query for store data
    query = session.query(StoreData).filter_by(store_id=store_id)
    
    if start_date:
        query = query.filter(StoreData.date >= start_date)
    if end_date:
        query = query.filter(StoreData.date <= end_date)
    
    # Execute query
    results = query.all()
    
    # Convert to DataFrame
    data = []
    for result in results:
        data.append({
            'store_id': result.store_id,
            'date': result.date,
            'category': result.category,
            'daily_sales': result.daily_sales,
            'daily_revenue': result.daily_revenue,
            'customer_footfall': result.customer_footfall,
            'margin': result.margin,
            'stock_level': result.stock_level,
            'promotion_active': result.promotion_active
        })
    
    session.close()
    
    if data:
        return pd.DataFrame(data)
    else:
        return None
