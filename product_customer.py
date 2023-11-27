#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings 



warnings.filterwarnings('ignore')#ignore warnings in program


# Load datasets
customer = pd.read_csv('olist_customers_dataset.csv')
location = pd.read_csv('olist_geolocation_dataset.csv')
order_item = pd.read_csv('olist_order_items_dataset.csv')
order = pd.read_csv('olist_orders_dataset.csv')
payment = pd.read_csv('olist_order_payments_dataset.csv')
review = pd.read_csv('olist_order_reviews_dataset.csv')
product = pd.read_csv('olist_products_dataset.csv')
seller = pd.read_csv('olist_sellers_dataset.csv')
translation = pd.read_csv('product_category_name_translation.csv')

# Remove duplicates from location
location_no_dup = location.drop_duplicates()
new_location = location[(location['geolocation_lat'] > -50) &
                        (location['geolocation_lat'] < 10) &
                        (location['geolocation_lng'] > -80) &
                        (location['geolocation_lng'] < -35)]

# Preprocess data for association rules
merged_df = product.merge(translation, left_on='product_category_name', right_on='product_category_name', how='left')
merged_df.drop(columns=['product_category_name'], inplace=True)
merged_product_name = merged_df.merge(order_item, left_on='product_id', right_on='product_id', how='left')
merged_product_name.drop(columns=['product_id'], inplace=True)
merged_product_name.drop(columns=['product_name_lenght', 'product_description_lenght', 'product_photos_qty',
                                  'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'],
                         inplace=True)
data_column = merged_product_name['product_category_name_english']



# Group the data by 'Category_Name' and sum the 'Quantity' for each category
category_quantities = merged_product_name.groupby('product_category_name_english')['order_item_id'].sum().reset_index()
category_quantities = category_quantities.sort_values(by='order_item_id', ascending=False)

# Plot quantity by category using Seaborn
st.subheader('Quantity by Category')
plt.figure(figsize=(15, 6))
sns.barplot(x='product_category_name_english', y='order_item_id', data=category_quantities)
plt.xlabel('Category')
plt.ylabel('Total Quantity Ordered')
plt.title('Bar Chart of Total Quantity Ordered by Category')
plt.xticks(rotation=45, ha='right')
st.pyplot()



# Plot customer distribution using Seaborn
st.subheader('Customer Distribution by State')

city_counts = customer['customer_city'].value_counts()
threshold = 0.01
small_cities = city_counts[city_counts / city_counts.sum() < threshold]
big_city = city_counts[city_counts / city_counts.sum() >= threshold]

plt.figure(figsize=(8, 8))
plt.pie(big_city, labels=big_city.index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Customer Distribution by State (1% Minimum)')
st.pyplot()


#merge payment and order_item datasframe to make get the payment value for each order
sales = pd.merge(payment, order_item, on='order_id', how='inner')
# Merge the result with the 'order' DataFrame to get the date of purchase of each order
sales = pd.merge(sales, order, on='order_id', how='inner')
sales = pd.merge(sales, product, on= 'product_id', how = 'inner')

# Merge the two dataframes on the common columns 'product_category name' 
sales = pd.merge(sales, translation, on='product_category_name', how= 'inner')

# # Drop the redundant 'Zip_Prefix' column
sales.drop(columns=['product_category_name'], inplace=True)


from mlxtend.frequent_patterns import association_rules, apriori


# Convert product_category_name to a list of lists (transactions)
transactions = sales.pivot_table(index='order_id',columns ='product_category_name_english',values = 'order_item_id',aggfunc='sum').fillna(0)

def encode(x):
    if x <=0:
        return 0
    else:
        return 1
df_pivot = transactions.applymap(encode)

support = 0.00001
frequent_items = apriori(df_pivot, min_support=support, use_colnames=True)
frequent_items.sort_values('support', ascending=False)


metric = 'lift'
min_treshold = 0.00001

rules = association_rules(frequent_items, metric=metric, min_threshold=min_treshold)[['antecedents','consequents','support','confidence','lift']]
rules.reset_index(drop=True).sort_values('confidence',ascending=False, inplace = True)
# Scatter plot rules by support, confidence, and color by lift
plt.figure(figsize=(12, 8))
scatter = plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.1)

# Add colorbar
plt.colorbar(scatter, label='Lift')

# Set labels and title
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules: Support vs Confidence (Color by Lift)')
st.pyplot()


customer_sales = pd.merge(payment, order, on='order_id')
columns_to_drop = ['order_id','payment_sequential','payment_installments','payment_type','order_status','order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']
customer_sales.drop(columns= columns_to_drop , inplace=True)




# Cluster customers using KMeans
st.subheader('Customer Clustering')
customer_sales = pd.merge(payment, order, on='order_id')
columns_to_drop = ['order_id', 'payment_sequential', 'payment_installments', 'payment_type', 'order_status',
                   'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                   'order_delivered_customer_date', 'order_estimated_delivery_date']
customer_sales.drop(columns=columns_to_drop, inplace=True)

# Group by 'customer_id' and sum the prices
customer_sales = customer_sales.groupby('customer_id')['payment_value'].sum().reset_index()

sample = ['payment_value']

# Extracting features for clustering
X = customer_sales[sample]

scaler = StandardScaler(with_mean=False)

# standardize the data
X_scaled = scaler.fit_transform(X)

# Apply KMeans with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=3)
pred_kmean_optimal = kmeans_optimal.fit_predict(X_scaled)

# Plotting the KMeans clusters with the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], [0] * len(X_scaled), c=pred_kmean_optimal, cmap='viridis', edgecolors='k')
plt.scatter(kmeans_optimal.cluster_centers_[:, 0], [0] * 3, s=300, c='red', marker='X', label='Centroids')
plt.title(f'KMeans Clustering with 3 Clusters')
plt.xlabel('Feature 1')
plt.legend()
st.pyplot()



# Plot sales growth over time using Matplotlib
def classify_cat(x):
    if x in ['office_furniture', 'furniture_decor', 'furniture_living_room', 'kitchen_dining_laundry_garden_furniture', 'bed_bath_table', 'home_comfort', 'home_comfort_2', 'home_construction', 'garden_tools', 'furniture_bedroom', 'furniture_mattress_and_upholstery']:
        return 'Furniture'
    elif x in ['auto', 'computers_accessories', 'musical_instruments', 'consoles_games', 'watches_gifts', 'air_conditioning', 'telephony', 'electronics', 'fixed_telephony', 'tablets_printing_image', 'computers', 'small_appliances_home_oven_and_coffee', 'small_appliances', 'audio', 'signaling_and_security', 'security_and_services']:
        return 'Electronics'
    elif x in ['fashio_female_clothing', 'fashion_male_clothing', 'fashion_bags_accessories', 'fashion_shoes', 'fashion_sport', 'fashion_underwear_beach', 'fashion_childrens_clothes', 'baby', 'cool_stuff']:
        return 'Fashion'
    elif x in ['housewares', 'home_confort', 'home_appliances', 'home_appliances_2', 'flowers', 'costruction_tools_garden', 'garden_tools', 'construction_tools_lights', 'costruction_tools_tools', 'luggage_accessories', 'la_cuisine', 'pet_shop', 'market_place']:
        return 'Home & Garden'
    elif x in ['sports_leisure', 'toys', 'cds_dvds_musicals', 'music', 'dvds_blu_ray', 'cine_photo', 'party_supplies', 'christmas_supplies', 'arts_and_craftmanship', 'art']:
        return 'Entertainment'
    elif x in ['health_beauty', 'perfumery', 'diapers_and_hygiene']:
        return 'Beauty & Health'
    elif x in ['food_drink', 'drinks', 'food']:
        return 'Food & Drinks'
    elif x in ['books_general_interest', 'books_technical', 'books_imported', 'stationery']:
        return 'Books & Stationery'
    elif x in ['construction_tools_construction', 'construction_tools_safety', 'industry_commerce_and_business', 'agro_industry_and_commerce']:
        return 'Industry & Construction'
    
sales['product_category_name_english'] = sales.product_category_name_english.apply(classify_cat)


sales['order_purchase_timestamp'] = pd.to_datetime(sales['order_purchase_timestamp'], format='%Y-%m-%d %H:%M:%S')
sales['year_month'] = sales['order_purchase_timestamp'].dt.to_period('M')
monthly_sales = sales.groupby('year_month')['payment_value'].sum().reset_index()
monthly_sales['growth_rate'] = monthly_sales['payment_value'].pct_change() * 100
monthly_sales['year_month'] = monthly_sales['year_month'].dt.strftime('%Y-%m')

# Plot location scatter using Geopandas and Matplotlib
# limit the range locations to remove noise and increase accuracy of the locations
new_location = location[(location['geolocation_lat'] > -50) &
                        (location['geolocation_lat'] < 10) &
                        (location['geolocation_lng'] > -80) &
                        (location['geolocation_lng'] < -35)]
st.subheader('Location Scatterplot in Brazil')
min_lat, max_lat = new_location['geolocation_lat'].min(), new_location['geolocation_lat'].max()
min_lon, max_lon = new_location['geolocation_lng'].min(), new_location['geolocation_lng'].max()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.cx[min_lon:max_lon, min_lat:max_lat]

ax = world[world.continent == "South America"].plot(color="white", edgecolor="black", figsize=(10, 10))

scatter = plt.scatter(
    location['geolocation_lng'],
    location['geolocation_lat'],
    s=20,
    c=location['geolocation_zip_code_prefix'],
    cmap='viridis',
    edgecolor='none',
    alpha=0.3
)

plt.xlim(min_lon, max_lon)
plt.ylim(min_lat, max_lat)

cbar = plt.colorbar(scatter, label='Zip Code')

plt.title('Location Scatterplot in Brazil')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

st.pyplot()



# Plot sales by category using Seaborn
st.subheader('Category-wise Total Sales by Year')
concat_sales = pd.merge(payment, customer, on='customer_id', how ='inner')
concat_sales = concat_sales.drop(columns=['order_purchase_timestamp', 'order_approved_at',
                                          'order_delivered_carrier_date', 'order_delivered_customer_date',
                                          'order_estimated_delivery_date', 'product_name_lenght',
                                          'product_description_lenght', 'product_photos_qty', 'product_weight_g',
                                          'product_length_cm', 'product_height_cm', 'product_width_cm'])
concat_sales = pd.merge(concat_sales, location, left_on='customer_zip_code_prefix',
                        right_on='geolocation_zip_code_prefix', how='inner')


count_sales = concat_sales.groupby(['year_month', 'product_category_name_english'])['price'].sum().reset_index()

custom_palette = sns.color_palette("husl", 24)
plt.figure(figsize=(16, 8))
sns.barplot(x='product_category_name_english', y='price', hue='year_month', data=concat_sales, palette=custom_palette)
plt.title('Category-wise Total Sales by Year')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Year')
st.pyplot()

