#!/usr/bin/env python
# coding: utf-8

# In[2]:

import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
import statistics
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime
from datetime import date
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.seasonal import seasonal_decompose
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Model Evaluation Metrics')

# Sidebar for selecting analysis type
selected_analysis = st.sidebar.selectbox('Choose Analysis', ['Churn Analysis','RFM Analysis','Sales Performance Evaluation'])

if selected_analysis  == 'Churn Analysis':

    customer = pd.read_csv('olist_customers_dataset.csv')
    order = pd.read_csv('olist_orders_dataset.csv')
    review = pd.read_csv('olist_order_reviews_dataset.csv')
    payment = pd.read_csv('olist_order_payments_dataset.csv')

    pd.set_option('display.max_columns', None)

    customer[customer['customer_unique_id'].duplicated()]

    order['order_approved_at'].fillna(0, inplace=True)
    order['order_delivered_carrier_date'].fillna(0, inplace=True)
    order['order_delivered_customer_date'].fillna(0, inplace=True)


    time = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']

    for i in time:
      order[i] = order[i] = pd.to_datetime(order[i], format='%Y-%m-%d %H:%M:%S', errors='coerce')


    order.info()

    order['order_approved_at'].fillna(0, inplace=True)
    order['order_delivered_carrier_date'].fillna(0, inplace=True)
    order['order_delivered_customer_date'].fillna(0, inplace=True)

    order.info()

    time = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']

    for i in time:
      order[i] = order[i] = pd.to_datetime(order[i], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    order.info()

    order[order['customer_id'].duplicated()]

    cm1 = customer.merge(order, how='inner', on='customer_id')

    cm1.info()

    cm1[cm1['customer_unique_id'].duplicated()]

    cm1 = cm1.sort_values('order_purchase_timestamp').groupby('customer_unique_id').tail(1)

    time = ['review_creation_date','review_answer_timestamp']

    for i in time:
      review[i] = pd.to_datetime(review[i])

    review[review['order_id'].duplicated()]


    review = review.sort_values('review_answer_timestamp').groupby('order_id').tail(1)

    review['review_comment_title'].fillna('Null', inplace=True)
    review['review_comment_message'].fillna('Null', inplace=True)

    review.info()

    cm2 = cm1.merge(review, how='left', on='order_id')

    cm2['review_score'].fillna(0, inplace=True)
    cm2['review_comment_title'].fillna('None', inplace=True)
    cm2['review_comment_message'].fillna('No Comment', inplace=True)

    cm2.info()

    payment[payment['order_id'].duplicated()]

    payment_type =payment.groupby('order_id')['payment_type'].agg(lambda x: pd.Series.mode(x)[0])

    payment_type

    payment_sequential = payment.groupby('order_id').agg(payment_sequential=('payment_sequential','median'))
    payment_sequential

    payment_installments = payment.groupby('order_id').agg(payment_installments=('payment_installments','median'))
    payment_installments

    cm3 = cm2.merge(payment_type, how='inner', on='order_id')

    cm3 = cm3.merge(payment_sequential, how='inner', on='order_id')

    cm3 = cm3.merge(payment_installments, how='inner', on='order_id')

    cm3.isnull().sum()

    cm3[cm3['customer_unique_id'].duplicated()]

    cm3['expected_duration'] = (cm3['order_estimated_delivery_date'] - cm3['order_purchase_timestamp']).dt.total_seconds() / 3600

    cm3['expected_duration'] = cm3['expected_duration'].astype('timedelta64[h]')

    cm3['ship_duration'] = (cm3['order_delivered_customer_date'] - cm3['order_purchase_timestamp']).dt.total_seconds() /3600
    cm3['ship_duration'] = cm3['ship_duration'].astype('timedelta64[h]')

    cm3 = cm3.rename(columns = {"order_purchase_timestamp": "last_order"})

    now = '2018-12-31 23:59:59'
    def churned(x):
        try:
            chn_now = pd.to_datetime(now)
            chn_last_order = pd.to_datetime(x)
            timedelta = chn_now - chn_last_order
            chnday = timedelta.days
            if chnday > 360:
                return True
            else:
                return False
        except:
            return True

    cm3['churned'] = cm3.last_order.apply(lambda x: churned(x))

    cm3['churned'].unique()

    cm3_count = cm3.groupby('churned').agg(count_of_churn=('churned','count'))

    def table(col_name):
        x = cm3.groupby(['churned', col_name])['customer_id'].nunique().reset_index(name='Customer')
        x1 = cm3.groupby([col_name])['customer_id'].nunique().reset_index(name='AllCustomer')
        x = x.merge(x1, on=col_name)
        x['Percentage'] = round(x['Customer']*100/x['AllCustomer'],2)
        return x

    col_name = 'customer_state'

    table(col_name)

    col_name = 'payment_type'

    table(col_name)

    col_name = 'order_status'

    table(col_name)

    col_name = 'review_score'

    table(col_name)

    template = go.layout.Template()
    template.layout = go.Layout(
        margin=dict(t=0, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Graph 1
    st.subheader('Customer State vs Churned')
    fig1 = px.histogram(cm3, x='customer_state', color='churned', barmode='group', width=800, height=500)
    fig1.update_layout(template=template)
    st.plotly_chart(fig1)

    # Graph 2
    st.subheader('Order Status vs Churned')
    fig2 = px.histogram(cm3, x='order_status', color='churned', barmode='group', width=800, height=500)
    fig2.update_layout(template=template)
    st.plotly_chart(fig2)

    # Graph 3
    st.subheader('Payment Type vs Churned')
    fig3 = px.histogram(cm3, x='payment_type', color='churned', barmode='group', width=800, height=500)
    fig3.update_layout(template=template)
    st.plotly_chart(fig3)

    # Graph 4
    st.subheader('Review Score vs Churned')
    fig4 = px.histogram(cm3, x='review_score', color='churned', barmode='group', width=800, height=500)
    fig4.update_layout(template=template)
    st.plotly_chart(fig4)


    cm3.info()

    cmml = cm3.drop(['customer_id','customer_unique_id','customer_zip_code_prefix','customer_city','order_id','last_order','order_approved_at',
                        'order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','review_id','review_creation_date',
                        'review_answer_timestamp','review_comment_message','review_comment_title'], axis=1)

    cmml.info()
    cmml['expected_duration'] = cmml['expected_duration'].dt.total_seconds().astype(float)
    cmml['ship_duration'] = cmml['ship_duration'].dt.total_seconds().astype(float)
    cmml.info()

    numerical_cols = [cname for cname in cmml.columns
                     if cmml[cname].dtype in ["int64", "float64"]]
    cmml[numerical_cols].head(10)

    cmml['churned'].replace(to_replace='True', value=1, inplace=True)
    cmml['churned'].replace(to_replace='False', value=0, inplace=True)

    cmml['churned']=cmml['churned'].astype('int')
    cmml['ship_duration'].fillna('0', inplace=True)

    cmml['ship_duration'] = cmml['ship_duration'].astype(float)
    cmml

    categorical_cols = [column for column, is_type in (cmml.dtypes=="object").items() if is_type]
    categorical_cols

    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    cmml['customer_state'] = labelencoder.fit_transform(cmml['customer_state'])

    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    cmml['order_status'] = labelencoder.fit_transform(cmml['order_status'])

    cmml.info()


    cm_dummies = pd.get_dummies(cmml)

    cm_dummies

    X = cm_dummies.loc[:, cm_dummies.columns != 'churned']
    y = cm_dummies['churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn import metrics
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
    from sklearn.metrics import make_scorer,accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,log_loss
    from sklearn.metrics import confusion_matrix

    all_model = [LogisticRegression,
                 KNeighborsClassifier,
                 RandomForestClassifier,
                 GradientBoostingClassifier,
                 GaussianNB,
                 XGBClassifier]

    model_name = ['LogisticRegression',
                 'KNeighborsClassifier',
                 'RandomForestClassifier',
                 'GradientBoostingClassifier',
                 'GaussianNB',
                 'XGBClassifier']

    ## loop for all model
    datatr = []
    datasc = []
    Recall =[]
    Precision =[]
    auc =[]

    for idx, model_type in enumerate(all_model):
        AccTrain = []
        AccTest = []
        RecallTemp = []
        PrecisionTemp = []
        AucTemp = []

        model = model_type()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        AccTrain.append(model.score(X_train , y_train))
        AccTest.append(model.score(X_test , y_test))
        RecallTemp.append(recall_score(y_test,y_pred))
        PrecisionTemp.append(precision_score(y_test,y_pred))
        AucTemp.append(roc_auc_score(y_test, y_pred))

        datatr.append(np.mean(AccTrain))
        datasc.append(np.mean(AccTest))
        Recall.append(np.mean(RecallTemp))
        Precision.append(np.mean(PrecisionTemp))
        auc.append(np.mean(AucTemp))

    data_result = pd.DataFrame()
    data_result['model'] = model_name
    data_result['Accuracy training'] = datatr
    data_result['Accuracy test'] = datasc
    data_result['Precision'] = Precision
    data_result['Recall']= Recall
    data_result['AUC']=auc
    data_result['gap'] = abs(data_result['Accuracy training'] - data_result['Accuracy test'])
    data_result.sort_values(by='Accuracy test',ascending=False)



    st.title('Machine Learning Model Evaluation Metrics')
    st.dataframe(data_result.style.highlight_max(axis=0), height=300)

    # Correlation matrix
    correlation_matrix = data_result[['Accuracy training', 'Accuracy test', 'Precision', 'Recall','AUC', 'gap']].corr()

    # Heatmap
    st.write('## Correlation Matrix of Evaluation Metrics')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Evaluation Metrics')
    st.pyplot()




    plot_data = data_result.melt(id_vars='model', var_name='metric', value_name='value')

    # Streamlit app
    st.title('Model Evaluation Metrics')

    # Set the default metric
    default_metric = 'Accuracy test'

    # Filter data based on the default metric
    filtered_data = plot_data[plot_data['metric'] == default_metric]



    # Combined bar graph
    st.write('## Combined Model Evaluation Metrics')

    # Create a function to display the value as a tooltip
    def show_values_on_bars(ax):
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)

    # Create bar plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='model', y='value', hue='metric', data=plot_data, palette='viridis')

    # Show values on the bars
    show_values_on_bars(ax)

    # Customize the plot
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('Combined Model Evaluation Metrics')
    plt.legend(title='Metric', loc='upper right')

    # Display the plot using st.pyplot()
    st.pyplot()


    import matplotlib.pyplot as plt
    import streamlit as st
    from sklearn.metrics import roc_curve, auc

    st.title('ROC Curves for Different Models')

    # Create a Plotly figure
    fig = go.Figure()

    for idx, model_type in enumerate(all_model):
        model = model_type()
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc_value = auc(fpr, tpr)

        # Add ROC curve trace to the figure
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name[idx]} (AUC = {roc_auc_value:.2f})'))

    # Add the diagonal line for random guessing
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))

    # Update layout
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title='Receiver Operating Characteristic (ROC) Curves',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    )

    # Display the plot using st.plotly_chart()
    st.plotly_chart(fig)







    st.title('Confusion Matrices for Different Models')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, model_type in enumerate(all_model):
        model = model_type()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'Confusion Matrix - {model_name[idx]}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    st.pyplot()

    # Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    st.write('## Precision-Recall Curve')

    # Create a Plotly figure
    fig = go.Figure()

    for idx, model_type in enumerate(all_model):
        model = model_type()
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        average_precision = average_precision_score(y_test, y_probs)

        # Add Precision-Recall curve trace to the figure
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'{model_name[idx]} (AP={average_precision:.2f})'))

    # Update layout
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        title='Precision-Recall Curve',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    )

    # Display the plot using st.plotly_chart()
    st.plotly_chart(fig)
        # Learning Curve
        

        
    st.write('## Precision-Recall Threshold Curve')

    # Create a Plotly figure
    fig = go.Figure()

    for idx, model_type in enumerate(all_model):
        model = model_type()
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

        # Add Precision-Recall Threshold curve traces to the figure
        fig.add_trace(go.Scatter(x=thresholds, y=precisions[:-1], mode='lines', name=f'{model_name[idx]} Precision'))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls[:-1], mode='lines', name=f'{model_name[idx]} Recall'))

    # Update layout
    fig.update_layout(
        xaxis_title='Threshold',
        yaxis_title='Score',
        title='Precision-Recall Threshold Curve',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    )

    # Display the plot using st.plotly_chart()
    st.plotly_chart(fig)

    # Calibration Curve
   

    st.write('## Calibration Curve')

# Create a Plotly figure
    fig = go.Figure()

    for idx, model_type in enumerate(all_model):
        model = model_type()
        model.fit(X_train, y_train)
        prob_pos = model.predict_proba(X_test)[:, 1]

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        # Add Calibration curve traces to the figure
        fig.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives, mode='lines', name=f'{model_name[idx]}'))

    # Update layout
    fig.update_layout(
        xaxis_title='Mean Predicted Value',
        yaxis_title='Fraction of Positives',
        title='Calibration Curve',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    )

    # Display the plot using st.plotly_chart()
    st.plotly_chart(fig)


elif selected_analysis == 'RFM Analysis':


    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    rfm = pd.read_csv("rfm.csv")

    ll_r = rfm.Recency.quantile(0.25)
    mid_r = rfm.Recency.quantile(0.50)
    ul_r = rfm.Recency.quantile(0.75)

    def recency_label(recent):
        if recent <= ll_r:
            return 1
        elif (recent > ll_r) and (recent <= mid_r):
            return 2
        elif (recent > mid_r) and (recent <= ul_r):
            return 3
        elif recent > ul_r:
            return 4

    ll_m = rfm.Monetary.quantile(0.25)
    mid_m = rfm.Monetary.quantile(0.50)
    ul_m = rfm.Monetary.quantile(0.75)

    def monetary_label(money):
        if money <= ll_m:
            return 4
        elif (money > ll_m) and (money <= mid_m):
            return 3
        elif (money > mid_m) and (money <= ul_m):
            return 2
        elif money > ul_m:
            return 1
        
    

    # Apply labels to create 'rank_rm'
    rfm['recency_label'] = rfm.Recency.apply(recency_label)
    rfm['monetary_label'] = rfm.Monetary.apply(monetary_label)
    rfm['rank_rm'] = list(zip(rfm.recency_label, rfm.monetary_label))


    # Create a Streamlit app
    st.title('Recency Distribution Dashboard')

    # Plot the distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.distplot(rfm.Recency, ax=ax)
    ax.axvline(rfm.Recency.mean(), color='red', linestyle='--', label='Mean')
    ax.axvline(rfm.Recency.median(), color='black', linestyle='--', label='Median')
    plt.title('Distribution of Recency', fontweight='bold', fontsize=20)
    plt.xlabel('Recency', fontsize=15, color='black')
    plt.ylabel('Value', fontsize=15, color='black')
    plt.legend()  # Add legend for mean and median
    st.pyplot(fig)  # Display the plot in Streamlit

    # Display mean, median, and skewness
    st.write('Mean of recency:', rfm.Recency.mean())
    st.write('Median of recency:', rfm.Recency.median())
    st.write('Skewness of recency:', rfm.Recency.skew())
    
    Q1 = np.quantile(rfm.Frequency, 0.25)
    Q3 = np.quantile(rfm.Frequency, 0.75)
    IQR = Q3 - Q1
    frequencyDistribution = rfm[~((rfm.Frequency < Q1 - 1.5 * IQR) | (rfm.Frequency > Q3 + 1.5 * IQR))]
    frequencyDistribution.head()


    st.title('Distribution of Frequency without Outliers')

    # Plot the distribution
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.distplot(frequencyDistribution.Frequency, ax=ax)
    ax.axvline(frequencyDistribution.Frequency.mean(), c='red', label='Mean')
    ax.axvline(frequencyDistribution.Frequency.median(), c='black', label='Median')

    # Customize plot
    ax.set_xlabel('Frequency', color='black', fontsize=15)
    ax.set_title('Distribution of frequency without outliers', color='black', fontsize=20, fontweight='bold')
    ax.legend()

    # Display the plot using Streamlit
    st.pyplot(fig)

    Q1 = np.quantile(rfm.Monetary, 0.25)
    Q3 = np.quantile(rfm.Monetary, 0.75)
    IQR = Q3 - Q1
    monetaryDistribution = rfm[~((rfm.Monetary < Q1 - 1.5 * IQR) | (rfm.Monetary > Q3 + 1.5 * IQR))]
    monetaryDistribution.head()
    
    
    # Display mean, median, and skewness for Frequency
    st.write('Mean of Frequency:', rfm.Frequency.mean())
    st.write('Median of Frequency:', rfm.Frequency.median())
    st.write('Skewness of Frequency:', rfm.Frequency.skew())

    st.title('Distribution of Monetary without Outliers')

    # Plot the distribution
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.distplot(monetaryDistribution.Monetary, ax=ax)
    ax.axvline(monetaryDistribution.Monetary.mean(), c='red', label='Mean')
    ax.axvline(monetaryDistribution.Monetary.median(), c='black', label='Median')

    # Customize plot
    ax.set_xlabel('Monetary', color='black', fontsize=15)
    ax.set_title('Distribution of monetary without outliers', color='black', fontsize=20, fontweight='bold')
    ax.legend()

    # Display the plot using Streamlit
    st.pyplot(fig)

    # Display mean, median, and skewness for Monetary
    st.write('Mean of Monetary:', rfm.Monetary.mean())
    st.write('Median of Monetary:', rfm.Monetary.median())
    st.write('Skewness of Monetary:', rfm.Monetary.skew())

   
    st.header('Boxplots')
    fig_boxplots, ax_boxplots = plt.subplots(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(y='Recency', data=rfm)
    plt.title('Recency Distribution')

    plt.subplot(1, 3, 2)
    sns.boxplot(y='Frequency', data=rfm)
    plt.title('Frequency Distribution')

    plt.subplot(1, 3, 3)
    sns.boxplot(y='Monetary', data=rfm)
    plt.title('Monetary Distribution')

    st.pyplot(fig_boxplots)

    st.header('Distribution of RFM Ranks')
    rank_df = rfm['rank_rm'].value_counts().reset_index()
    rank_df.columns = ['Rank', 'Count']

    fig_rank, ax_rank = plt.subplots(figsize=(10, 6))
    ax_rank.bar(rank_df['Rank'].astype(str), rank_df['Count'], color='skyblue')
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title('Distribution of RFM Ranks')
    st.pyplot(fig_rank)  # Display the rank distribution plot in Streamlit
    
    important_ranks = {
        "(Recency - 1, Monetary - 1)": "They are very recent and have spent a lot of money",
        "(Recency - 1, Monetary - 2)": "They are very recent and have spent a good amount of money",
        "(Recency - 2, Monetary - 1)": "They are recent and have spent a lot of money",
        "(Recency - 2, Monetary - 2)": "They are recent and have spent a good amount of money",
        "(Recency - 1, Monetary - 3)": "They are very recent and have spent a decent amount of money"
    }

    least_important_ranks = {
        "(Recency - 4, Monetary - 4)": "They are not at all recent and spend a negligible amount of money",
        "(Recency - 4, Monetary - 3)": "They are not at all recent and spend a decent amount of money",
        "(Recency - 4, Monetary - 2)": "They are not at all recent and spend a good amount of money",
        "(Recency - 3, Monetary - 4)": "They are not very recent and spend a negligible amount of money",
        "(Recency - 3, Monetary - 3)": "They are not very recent and spend a decent amount of money"
    }

    # Display the ranks using st.write
    st.write("Most Important Ranks:")
    for rank, description in important_ranks.items():
        st.write(f"{rank} - {description}")

    st.write("Least Important Ranks:")
    for rank, description in least_important_ranks.items():
        st.write(f"{rank} - {description}")

    # Create the 'Churn' column
    rfm['Churn'] = rfm.Recency.apply(lambda x: 1 if x > rfm.Recency.mean() else 0)
    # Calculate churn counts
    churn_counts = rfm['Churn'].value_counts()

    # Plot the pie chart
    fig_churn, ax_churn = plt.subplots(figsize=(8, 8))
    ax_churn.pie(churn_counts, labels=['Not Churned', 'Churned'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
    plt.title('Churn Distribution')
    st.pyplot(fig_churn)  # Display the churn distribution pie chart in Streamlit

elif selected_analysis == 'Sales Performance Evaluation':
    

    # Load datasets
    customers = pd.read_csv('olist_customers_dataset.csv')
    order_items = pd.read_csv('olist_order_items_dataset.csv')
    products = pd.read_csv('olist_products_dataset.csv')
    orders = pd.read_csv('olist_orders_dataset.csv')
    translations = pd.read_csv('product_category_name_translation.csv')

    # Merge relevant datasets
    merged_data = pd.merge(order_items, products, on='product_id')
    merged_data = pd.merge(merged_data, orders, on='order_id')
    merged_data = pd.merge(merged_data, customers, on='customer_id')
    merged_data = pd.merge(merged_data, translations, on='product_category_name', how='left')

    # Calculate total sales
    merged_data['total_sales'] = merged_data['price'] + merged_data['freight_value']

    # Define the provided mapping
    category_mapping = {
        'health_beauty': 'Personal Care',
        'computers_accessories': 'Electronics',
        'auto': 'Automotive',
        'bed_bath_table': 'Home and Kitchen',
        'furniture_decor': 'Home and Kitchen',
        'sports_leisure': 'Sports and Outdoors',
        'perfumery': 'Personal Care',
        'housewares': 'Home and Kitchen',
        'telephony': 'Electronics',
        'watches_gifts': 'Fashion Accessories',
        'food_drink': 'Food and Drink',
        'baby': 'Baby',
        'stationery': 'Office Supplies',
        'tablets_printing_image': 'Electronics',
        'toys': 'Toys and Games',
        'garden_tools': 'Home and Garden',
        'fashion_bags_accessories': 'Fashion Accessories',
        'small_appliances': 'Appliances',
        'consoles_games': 'Electronics',
        'audio': 'Electronics',
        'fashion_shoes': 'Fashion',
        'cool_stuff': 'Miscellaneous',
        'air_conditioning': 'Appliances',
        'construction_tools_construction': 'Tools',
        'kitchen_dining_laundry_garden_furniture': 'Home and Garden',
        'fashion_male_clothing': 'Fashion',
        'pet_shop': 'Pets',
        'office_furniture': 'Furniture',
        'market_place': 'Marketplace',
        'electronics': 'Electronics',
        'party_supplies': 'Party Supplies',
        'home_confort': 'Home and Kitchen',
        'agro_industry_and_commerce': 'Industry and Commerce',
        'furniture_mattress_and_upholstery': 'Furniture',
        'books_technical': 'Books',
        'musical_instruments': 'Musical Instruments',
        'furniture_living_room': 'Furniture',
        'industry_commerce_and_business': 'Industry and Commerce',
        'fashion_underwear_beach': 'Fashion',
        'signaling_and_security': 'Security',
        'christmas_supplies': 'Holiday Supplies',
        'cds_dvds_musicals': 'Entertainment',
        'dvds_blu_ray': 'Entertainment',
        'flowers': 'Home and Garden',
        'arts_and_craftsmanship': 'Arts and Crafts',
        'diapers_and_hygiene': 'Baby',
        'security_and_services': 'Security',
    }

    # Map the product categories to more general categories
    merged_data['general_category'] = merged_data['product_category_name_english'].map(category_mapping)

    # Convert 'order_purchase_timestamp' to datetime
    merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])

    # Resample total sales over time for monthly trends
    time_sales_monthly = merged_data.resample('M', on='order_purchase_timestamp')['total_sales'].sum().reset_index()

    # Resample total sales over time for weekly trends
    time_sales_weekly = merged_data.resample('W', on='order_purchase_timestamp')['total_sales'].sum().reset_index()

    # Linear Regression for Monthly Trends
    X_monthly = np.arange(len(time_sales_monthly)).reshape(-1, 1)
    y_monthly = time_sales_monthly['total_sales'].values

    X_train_monthly, X_test_monthly, y_train_monthly, y_test_monthly = train_test_split(
        X_monthly, y_monthly, test_size=0.2, random_state=42)

    regressor_monthly = LinearRegression()
    regressor_monthly.fit(X_train_monthly, y_train_monthly)
    y_pred_monthly = regressor_monthly.predict(X_test_monthly)

    # Linear Regression for Weekly Trends
    X_weekly = np.arange(len(time_sales_weekly)).reshape(-1, 1)
    y_weekly = time_sales_weekly['total_sales'].values

    X_train_weekly, X_test_weekly, y_train_weekly, y_test_weekly = train_test_split(
        X_weekly, y_weekly, test_size=0.2, random_state=42)

    regressor_weekly = LinearRegression()
    regressor_weekly.fit(X_train_weekly, y_train_weekly)
    y_pred_weekly = regressor_weekly.predict(X_test_weekly)

    # Streamlit app
    st.title('Sales Performance Evaluation')

    # Display sales by general product category
    st.header('Sales by General Product Category')
    category_sales_table = merged_data.groupby('general_category')['total_sales'].sum().sort_values(ascending=False).reset_index()
    st.write(category_sales_table)

    # Display top 10 most sale categories (bar chart)
    st.header('Top 10 Most Sale Categories')
    top_10_categories = merged_data.groupby('general_category')['total_sales'].sum().nlargest(10)
    st.bar_chart(top_10_categories)

    # Display total sales over time for monthly trends
    st.header('Total Sales Over Time (Monthly)')
    st.line_chart(time_sales_monthly.set_index('order_purchase_timestamp'))

    # Display total sales over time for weekly trends
    st.header('Total Sales Over Time (Weekly)')
    st.line_chart(time_sales_weekly.set_index('order_purchase_timestamp'))

    # Perform seasonal decomposition for monthly trends
    result_monthly = seasonal_decompose(time_sales_monthly.set_index('order_purchase_timestamp'), model='additive', period=12)

    # Perform seasonal decomposition for weekly trends
    result_weekly = seasonal_decompose(time_sales_weekly.set_index('order_purchase_timestamp'), model='additive', period=7)

    # Display the decompositions and residuals for monthly trends
    st.header('Seasonal Decomposition (Monthly)')
    st.line_chart(result_monthly.observed)
    st.write("**Observed (Monthly)**")

    st.line_chart(result_monthly.trend)
    st.write("**Trend (Monthly)**")

    st.line_chart(result_monthly.seasonal)
    st.write("**Seasonal (Monthly)**")

    # Display the residuals for monthly trends using Matplotlib
    st.header('Residual Graph (Monthly)')
    fig_monthly, ax_monthly = plt.subplots()
    ax_monthly.plot(result_monthly.resid.index, result_monthly.resid)
    ax_monthly.set_xlabel('Date')
    ax_monthly.set_ylabel('Residuals')
    ax_monthly.set_title('Residuals Over Time (Monthly)')
    st.pyplot(fig_monthly)
    st.write("**Residuals Over Time (Monthly)**")

    # Regression Analysis for Monthly Trends
    st.header('Regression Analysis (Monthly)')
    st.write(f'Coefficient (Monthly): {regressor_monthly.coef_[0]:.2f}')
    st.write(f'Intercept (Monthly): {regressor_monthly.intercept_:.2f}')
    st.write(f'Mean Squared Error (Monthly): {mean_squared_error(y_test_monthly, y_pred_monthly):.2f}')
    st.write(f'R2 Score (Monthly): {r2_score(y_test_monthly, y_pred_monthly):.2f}')

    # Plot the original time series and its regression line for Monthly Trends
    st.header('Total Sales Over Time with Regression Line (Monthly)')
    fig_monthly_regression = plt.figure()
    plt.plot(time_sales_monthly['order_purchase_timestamp'], time_sales_monthly['total_sales'], label='Actual Sales')
    plt.plot(time_sales_monthly['order_purchase_timestamp'].iloc[X_test_monthly.flatten()],
             y_pred_monthly, label='Predicted Sales', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.title('Total Sales Over Time with Regression Line (Monthly)')
    plt.legend()
    st.pyplot(fig_monthly_regression)


    # Display the decompositions and residuals for weekly trends
    st.header('Seasonal Decomposition (Weekly)')
    st.line_chart(result_weekly.observed)
    st.write("**Observed (Weekly)**")

    st.line_chart(result_weekly.trend)
    st.write("**Trend (Weekly)**")

    st.line_chart(result_weekly.seasonal)
    st.write("**Seasonal (Weekly)**")

    # Display the residuals for weekly trends using Matplotlib
    st.header('Residual Graph (Weekly)')
    fig_weekly, ax_weekly = plt.subplots()
    ax_weekly.plot(result_weekly.resid.index, result_weekly.resid)
    ax_weekly.set_xlabel('Date')
    ax_weekly.set_ylabel('Residuals')
    ax_weekly.set_title('Residuals Over Time (Weekly)')
    st.pyplot(fig_weekly)
    st.write("**Residuals Over Time (Weekly)**")

    # Regression Analysis for Weekly Trends
    st.header('Regression Analysis (Weekly)')
    st.write(f'Coefficient (Weekly): {regressor_weekly.coef_[0]:.2f}')
    st.write(f'Intercept (Weekly): {regressor_weekly.intercept_:.2f}')
    st.write(f'Mean Squared Error (Weekly): {mean_squared_error(y_test_weekly, y_pred_weekly):.2f}')
    st.write(f'R2 Score (Weekly): {r2_score(y_test_weekly, y_pred_weekly):.2f}')

    # Plot the original time series and its regression line for Weekly Trends
    st.header('Total Sales Over Time with Regression Line (Weekly)')
    fig_weekly_regression = plt.figure()
    plt.plot(time_sales_weekly['order_purchase_timestamp'], time_sales_weekly['total_sales'], label='Actual Sales')
    plt.plot(time_sales_weekly['order_purchase_timestamp'].iloc[X_test_weekly.flatten()],
             y_pred_weekly, label='Predicted Sales', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.title('Total Sales Over Time with Regression Line (Weekly)')
    plt.legend()
    st.pyplot(fig_weekly_regression)