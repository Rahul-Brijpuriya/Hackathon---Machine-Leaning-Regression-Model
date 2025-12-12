
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import os

def generate_addresses():
    input_file = "data/raw/amazon_delivery.csv"
    output_file = "data/processed/order_addresses.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(input_file)
    
    # For demonstration purposes and to avoid hitting API rate limits immediately, 
    # we will process a sample. The user can remove .head() to process all.
    # Nominatim has a strict usage policy (1 request/sec). 
    # Processing 43k rows would take 12 hours.
    # Ideally, use a paid API or an offline geocoder for this volume.
    
    print("Processing first 10 rows for demonstration...")
    df_subset = df.head(10).copy() 
    
    geolocator = Nominatim(user_agent="amazon_delivery_hackathon_demo")
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    # Function to get address safely
    def get_address(lat, lon):
        try:
            # Check for invalid coordinates
            if lat == 0 or lon == 0:
                return "Invalid Coordinates"
            
            location = geocode((lat, lon), language='en')
            return location.address if location else "Address Not Found"
        except Exception as e:
            return f"Error: {str(e)}"

    print("Geocoding Store Addresses...")
    df_subset['Store_Address'] = df_subset.apply(
        lambda row: get_address(row['Store_Latitude'], row['Store_Longitude']), axis=1
    )
    
    print("Geocoding Drop Addresses...")
    df_subset['Drop_Address'] = df_subset.apply(
        lambda row: get_address(row['Drop_Latitude'], row['Drop_Longitude']), axis=1
    )
    
    # Select columns
    result_df = df_subset[['Order_ID', 'Store_Address', 'Drop_Address']]
    
    # Save
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    result_df.to_csv(output_file, index=False)
    print(f"Addresses saved to {output_file}")

if __name__ == "__main__":
    generate_addresses()
