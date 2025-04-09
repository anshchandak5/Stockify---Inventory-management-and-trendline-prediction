import sqlite3
import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Create the base class for SQLAlchemy models
Base = declarative_base()

# Define the Store model
class Store(Base):
    __tablename__ = 'stores'
    
    id = Column(Integer, primary_key=True)
    store_id = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    area = Column(String(100), nullable=False)
    city = Column(String(100), nullable=False)
    zone = Column(String(100), nullable=False)
    
    # Relationships
    data_uploads = relationship("DataUpload", back_populates="store")
    store_data = relationship("StoreData", back_populates="store")

# Define the DataUpload model to track file uploads
class DataUpload(Base):
    __tablename__ = 'data_uploads'
    
    id = Column(Integer, primary_key=True)
    store_id = Column(String(20), ForeignKey('stores.store_id'), nullable=False)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.now)
    processed = Column(Boolean, default=False)
    record_count = Column(Integer, default=0)
    
    # Relationships
    store = relationship("Store", back_populates="data_uploads")

# Define the StoreData model for actual store data
class StoreData(Base):
    __tablename__ = 'store_data'
    
    id = Column(Integer, primary_key=True)
    store_id = Column(String(20), ForeignKey('stores.store_id'), nullable=False)
    date = Column(Date, nullable=False)
    category = Column(String(100), nullable=False)
    daily_sales = Column(Integer, nullable=False)
    daily_revenue = Column(Float, nullable=False)
    customer_footfall = Column(Integer, nullable=False)
    margin = Column(Float, nullable=False)
    stock_level = Column(Integer, nullable=False)
    promotion_active = Column(Boolean, default=False)
    upload_id = Column(Integer, ForeignKey('data_uploads.id'), nullable=False)
    
    # Relationships
    store = relationship("Store", back_populates="store_data")

# Database initialization function
def init_db(db_path='stockify.db'):
    """Initialize the database and create tables if they don't exist"""
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    return engine

# Session factory function
def get_session(engine):
    """Create a new session"""
    Session = sessionmaker(bind=engine)
    return Session()

# Function to initialize the database with existing stores from the app
def initialize_stores(engine, stores_data):
    """Initialize the stores table with existing store data"""
    session = get_session(engine)
    
    for store_id, data in stores_data.items():
        if 'store_id' in data:
            # Check if store already exists
            existing_store = session.query(Store).filter_by(store_id=data['store_id']).first()
            if not existing_store:
                # Extract city and zone from area if not provided
                area = data.get('area', '')
                city = data.get('city', 'Nagpur')  # Default city
                zone = area.split()[-1] if area else ''  # Extract zone from area name
                
                # Create new store
                new_store = Store(
                    store_id=data['store_id'],
                    name=f"Store {store_id}",
                    area=area,
                    city=city,
                    zone=zone
                )
                session.add(new_store)
    
    session.commit()
    session.close()

# Function to import data from CSV to database
def import_csv_to_db(file_path, store_id, upload_id, engine):
    """Import data from a CSV file to the database"""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['date', 'category', 'daily_sales', 'daily_revenue', 
                            'customer_footfall', 'margin', 'stock_level', 'promotion_active']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Convert data types
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['promotion_active'] = df['promotion_active'].astype(bool)
        
        # Add store_id and upload_id
        df['store_id'] = store_id
        df['upload_id'] = upload_id
        
        # Insert data into database
        session = get_session(engine)
        
        # First, delete any existing data for this store with the same dates
        dates = df['date'].unique()
        session.query(StoreData).filter(
            StoreData.store_id == store_id,
            StoreData.date.in_(dates)
        ).delete(synchronize_session=False)
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Insert new records
        for record in records:
            new_data = StoreData(**record)
            session.add(new_data)
        
        # Update the upload record
        upload = session.query(DataUpload).filter_by(id=upload_id).first()
        if upload:
            upload.processed = True
            upload.record_count = len(records)
        
        session.commit()
        session.close()
        
        return True, f"Successfully imported {len(records)} records"
    
    except Exception as e:
        return False, f"Error importing data: {str(e)}"

# Function to export template CSV
def export_template():
    """Create and export a template CSV file for data upload"""
    template_data = {
        'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'category': ['Electronics', 'Clothing', 'Electronics', 'Clothing'],
        'daily_sales': [10, 25, 15, 30],
        'daily_revenue': [5000, 2500, 7500, 3000],
        'customer_footfall': [50, 50, 60, 60],
        'margin': [20, 40, 20, 40],
        'stock_level': [100, 200, 90, 170],
        'promotion_active': [True, False, True, False]
    }
    
    df = pd.DataFrame(template_data)
    template_path = 'data_upload_template.csv'
    df.to_csv(template_path, index=False)
    return template_path

# Function to get aggregated data for city/zone analysis
def get_aggregated_data(engine, area=None, zone=None, city=None, start_date=None, end_date=None):
    """Get aggregated data for city or zone-wide analysis"""
    session = get_session(engine)
    
    # Build the query
    query = session.query(StoreData)
    
    # Join with Store to filter by area/zone/city
    if area or zone or city:
        query = query.join(Store)
        
        if area:
            query = query.filter(Store.area == area)
        if zone:
            query = query.filter(Store.zone == zone)
        if city:
            query = query.filter(Store.city == city)
    
    # Filter by date range
    if start_date:
        query = query.filter(StoreData.date >= start_date)
    if end_date:
        query = query.filter(StoreData.date <= end_date)
    
    # Execute query and convert to DataFrame
    results = query.all()
    
    # Convert SQLAlchemy objects to dictionaries
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
    
    # Convert to DataFrame
    if data:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

# Function to update models with new data
def update_models(engine):
    """Update prediction models with the latest data"""
    # This function would be implemented to retrain models
    # based on the latest data in the database
    pass
