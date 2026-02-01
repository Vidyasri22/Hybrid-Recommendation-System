"""
Method Description:
This recommendation system leverages XGBoost to predict Yelp ratings, significantly improving Assignment 3’s hybrid model (collaborative filtering with XGBoost). I expanded to 27 features from only user and business files, adding 10 business features (stars, review count, latitude, longitude, price, credit cards, appointment-only, reservations, table service, wheelchair accessibility) and 17 user features (review count, friends, useful, funny, cool, fans, elite years, average stars, 9 compliment types). These capture business quality, service, and user engagement, expertise, social influence, unlike Assignment 3’s limited check-in/photo counts. Dropping collaborative filtering simplified computation, while Spark RDDs process yelp_train.csv, user.json, and business.json efficiently, using defaults/random imputation for missing data. XGBoost (1000 trees, depth 5, eta 0.05) was tuned manually: broad parameter tests, then fine-tuning. Validation on yelp_val.csv shows a sharper RMSE, proving the feature-rich model’s edge.

Error Distribution:
>=0 and <1: 102277
>=1 and <2: 32735
>=2 and <3: 6198
>=3 and <4: 833
>=4: 1

RMSE: 0.979064428050631

Execution Time: 367.04 seconds
"""

import os
import sys
import time
import json
import numpy as np
import xgboost as xgb
import random
import csv
from pyspark import SparkContext

# Constants for default feature values
# Business: stars, reviews, lat, lon, price, cards, appt, reserv, table, wheel
busi_default = (0.0, 0.0, 0.0, 0.0, 2, 0, 0, 0, 0, 0)
# User: reviews, friends, useful, funny, cool, fans, elite, avg_stars, compliments
user_default = (0.0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  

# Utility functions for feature extraction for business and user details

# fucntion for RestaurantsPriceRange2" field out of a business records
def extract_price(attributes):
    
    # if the attribute is not there, it returns the random integer 1-4 
    if not attributes or 'RestaurantsPriceRange2' not in attributes:
        return random.randint(1, 4)
    try:
        # If present and non-empty, change to int.
        price = attributes['RestaurantsPriceRange2']
        return int(price) if price else random.randint(1, 4)
    except (ValueError, TypeError):
        return random.randint(1, 4)
    
# function for normalizing a variety of boolean representations.
def extract_boolean(attributes, key):
    
    # if the attribute is missing then it returns 0 or 1
    if not attributes or key not in attributes:
        return random.randint(0, 1)
    
    value = attributes[key]
    
    # if it is true return 1
    if isinstance(value, bool):
        return int(value)
    # if it is false return 0
    if isinstance(value, str):
        return 1 if value.lower() == 'true' else 0
    
    return random.randint(0, 1)


# Function for Count years a user was elite
def count_elite_years(elite):
   
    # if elite field is not a comma-delimited list of years
    if not elite or str(elite).lower() in ('none', ''):
        return 0
    # Otherwise split on , and count.
    return len(str(elite).split(','))

# Function for counting the no of friends for a user
def count_friends(friends):
    # if friends are none it will return 0
    if not friends or str(friends).lower() in ('none', ''):
        return 0
    # Otherwise split on , and count.
    return len(str(friends).split(','))

# Spark setup and initialization
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
spark_context = SparkContext('local[*]', 'yelp_predictor')
spark_context.setLogLevel("ERROR")

# function for reading the JSON file using RDD
def read_json_rdd(file_path):
    return spark_context.textFile(file_path).map(json.loads)

# fucntion for accumulating business features
def extract_business_features(business_rdd):
    
    """Processing business data into feature tuples."""
    
    # skips the records without business_id, and get all the numeric fields
    # and if any fields are missing it will return to default features
    def parse_business(record):
        biz_id = record.get('business_id')
        if not biz_id:
            return None
        attrs = record.get('attributes', {})
        
        # Safely extract numeric fields, using defaults if None or missing
        stars = float(record.get('stars', busi_default[0]) or busi_default[0])
        review_count = float(record.get('review_count', busi_default[1]) or busi_default[1])
        latitude = float(record.get('latitude', busi_default[2]) or busi_default[2])
        longitude = float(record.get('longitude', busi_default[3]) or busi_default[3])

        return (biz_id, (
            stars,
            review_count,
            latitude,
            longitude,
            extract_price(attrs),
            extract_boolean(attrs, 'BusinessAcceptsCreditCards'),
            extract_boolean(attrs, 'ByAppointmentOnly'),
            extract_boolean(attrs, 'RestaurantsReservations'),
            extract_boolean(attrs, 'RestaurantsTableService'),
            extract_boolean(attrs, 'WheelchairAccessible')
        ))
    
    # Transform each JSON record into (business_id, features).
    return business_rdd.map(parse_business).filter(lambda x: x is not None)

# function for accualating user features
def extract_user_features(user_rdd):
    
    """Process user data into feature tuples."""
    
    # skips the records without user_id, 
    # and if any fields are missing it will return to default features
    def parse_user(record):
        usr_id = record.get('user_id')
        if not usr_id:
            return None
        
        # Safely extract numeric fields, using defaults if None or missing
        review_count = float(record.get('review_count', user_default[0]) or user_default[0])
        avg_stars = float(record.get('average_stars', user_default[7]) or user_default[7])

        return (usr_id, (
            review_count,
            count_friends(record.get('friends')),
            int(record.get('useful', user_default[2])),
            int(record.get('funny', user_default[3])),
            int(record.get('cool', user_default[4])),
            int(record.get('fans', user_default[5])),
            count_elite_years(record.get('elite')),
            avg_stars,
            # compliments—hot, profile, list, note, plain, cool, funny, writer, photos
            int(record.get('compliment_hot', user_default[8])),
            int(record.get('compliment_profile', user_default[9])),
            int(record.get('compliment_list', user_default[10])),
            int(record.get('compliment_note', user_default[11])),
            int(record.get('compliment_plain', user_default[12])),
            int(record.get('compliment_cool', user_default[13])),
            int(record.get('compliment_funny', user_default[14])),
            int(record.get('compliment_writer', user_default[15])),
            int(record.get('compliment_photos', user_default[16]))
        ))
    
    # Transform each JSON record into (user_id, features).
    return user_rdd.map(parse_user).filter(lambda x: x is not None)

# function for building the combined feature RDD
def build_feature_rdd(data_rdd, biz_map, usr_map):
    
    """Combine user and business features with input data."""
    
    return data_rdd.map(lambda x: (
        x[0], # user_id
        x[1], # businees_id 
        float(x[2]) if len(x) > 2 and x[2] else None,
        *biz_map.get(x[1], busi_default), # extracted_business_features
        *usr_map.get(x[0], user_default)  # extracted_user_features
    ))

# fucntion for calculating RMSE and error distributions
def compute_rmse(test_file, prediction_file):
    
    """Compute RMSE and error distribution between true and predicted ratings."""
    
    true_values = {}
    predicted_values = {}
    
    # reading the values from the actual output csv file and storing them
    try:
        with open(test_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    # dictionary for true_values
                    true_values[(row[0], row[1])] = float(row[2])
    except Exception as e:
        print(f"Error reading test file: {e}")
        return None, None
    
    # reading the values from predicted output file and storing them
    try:
        with open(prediction_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                # dictionary for predicted_values
                predicted_values[(row[0], row[1])] = float(row[2])
    except Exception as e:
        print(f"Error reading prediction file: {e}")
        return None, None
    
    # if not any true_values found return None
    if not true_values:
        print("No true ratings found.")
        return None, None
    
    #  # Match pairs & compute absolute errors
    errors = [abs(predicted_values[key] - true_values[key]) for key in true_values if key in predicted_values]
    if not errors:
        print("No matching pairs for RMSE.")
        return None, None

    # Compute Root Mean square Error
    rmse = np.sqrt(np.mean([e**2 for e in errors]))

    # Computing error distribution
    bins = [0, 0, 0, 0, 0]  # >=0 and <1, >=1 and <2, >=2 and <3, >=3 and <4, >=4
    for e in errors:
        if 0 <= e < 1:
            bins[0] += 1
        elif 1 <= e < 2:
            bins[1] += 1
        elif 2 <= e < 3:
            bins[2] += 1
        elif 3 <= e < 4:
            bins[3] += 1
        elif e >= 4:
            bins[4] += 1

    return rmse, bins

def main():
    
    if len(sys.argv) != 4:
        print("Enter the arguments in the proper order of filename, data folder, validation file and test file")
        sys.exit(1)
    
    # execution of the program starts
    start = time.time()
    
    # getting the arguments
    data_dir, test_path, output_path = sys.argv[1:4]

    # Loading and processing business data
    biz_rdd = read_json_rdd(os.path.join(data_dir, 'business.json'))
    biz_features = extract_business_features(biz_rdd).collectAsMap()
    
    # Loading and processing the user data
    usr_rdd = read_json_rdd(os.path.join(data_dir, 'user.json'))
    usr_features = extract_user_features(usr_rdd).collectAsMap()

    # Loading and processing training data as RDD of lists [user, business, rating]
    train_file = os.path.join(data_dir, 'yelp_train.csv')
    train_rdd = spark_context.textFile(train_file)
    header = train_rdd.first()
    train_data = train_rdd.filter(lambda x: x != header)\
                         .map(lambda x: list(csv.reader([x], skipinitialspace=True))[0])\
                         .filter(lambda x: len(x) == 3)
    # getting all the train features 
    train_features = build_feature_rdd(train_data, biz_features, usr_features)\
                     .filter(lambda x: x[2] is not None)\
                     .collect()

    # Build full feature vectors, drop any with missing rating, collect to driver, then split into X_train and y_train.
    X_train = np.array([row[3:] for row in train_features])
    y_train = np.array([row[2] for row in train_features])

    # Training XGBoost model
    model = xgb.XGBRegressor(
        eta=0.05,
        max_depth=5,
        n_estimators=1000,
        seed=42,
        n_jobs=-1,
        subsample=0.9,
        colsample_bytree=0.7,
        alpha=0.3,
        reg_lambda=1.0
    )
    model.fit(X_train, y_train)

    # Loading and processing the test data
    test_rdd = spark_context.textFile(test_path)
    test_header = test_rdd.first()
    test_data = test_rdd.filter(lambda x: x != test_header)\
                        .map(lambda x: list(csv.reader([x], skipinitialspace=True))[0])\
                        .map(lambda x: (x[0], x[1], None)) # we ignore the “true” rating field here, because it’s None for prediction.
            
    # Build & collect the test feature matrix; store corresponding user/business IDs for writing output.
    test_features = build_feature_rdd(test_data, biz_features, usr_features).collect()
    X_test = np.array([row[3:] for row in test_features])
    user_ids = [row[0] for row in test_features]
    biz_ids = [row[1] for row in test_features]

    # Predict and clip the results
    predictions = np.clip(model.predict(X_test), 0.5, 5.0)

    # Write predictions to output file
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'business_id', 'prediction'])
        for uid, bid, pred in zip(user_ids, biz_ids, predictions):
            writer.writerow([uid, bid, f'{pred:.8f}'])

    # Compute and display RMSE and error distribution
    #rmse, error_bins = compute_rmse(test_path, output_path)
    #if rmse is not None:
        #print(f"RMSE: {rmse:.4f}")
        #print("Error Distribution:")
        #print(f">=0 and <1: {error_bins[0]}")
        #print(f">=1 and <2: {error_bins[1]}")
        #print(f">=2 and <3: {error_bins[2]}")
        #print(f">=3 and <4: {error_bins[3]}")
        #print(f">=4: {error_bins[4]}")
    #else:
        #print("Unable to compute RMSE and error distribution.")
    
    # stopping the execution time
    execution_time = time.time() - start
    print(f"Execution time: {execution_time:.2f} seconds")
    spark_context.stop()

if __name__ == '__main__':
    main()