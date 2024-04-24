import streamlit as st
import numpy as np
import re
import pandas as pd
import os
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go


def plot_best_fit_line(x_column, y_column, data):
    # Linear regression to find the best fit line with intercept forced to zero
    x_values = data[x_column].values.reshape(-1, 1)
    y_values = data[y_column].values.reshape(-1, 1)

    # Fit the linear regression model with intercept set to False
    model = LinearRegression(fit_intercept=False)
    model.fit(x_values, y_values)

    # Calculate the y values for the best fit line with intercept equal to zero
    line_of_best_fit = model.predict(x_values)

    # Get the slope from the model
    slope = model.coef_[0][0]

    # Create a plot
    fig = go.Figure(labels={"x": "Spectrophotometer", "y": f"{y_column} Absorbance"})

    # Add the scatter plot
    fig.add_trace(go.Scatter(x=x_values.flatten(), y=y_values.flatten(), mode="markers", name=f"{y_column} Data"))

    # Add the best fit line with intercept equal to zero
    fig.add_trace(go.Scatter(x=x_values.flatten(), y=line_of_best_fit.flatten(), mode="lines", name=f"Best Fit Line for {y_column} (Intercept=0)"))

    # Update layout
    fig.update_layout(
        title_text=f"Scatter Plot with Best Fit Line for {y_column}",
        showlegend=True,
        width=800,
        height=800
    )

    # Show the plot using Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display the equation of the best fit line with intercept equal to zero
    st.write(f"The best fit line for {y_column} with intercept equal to zero is y = {slope:.4f}x")

    return slope


def limit_filter_bar_plot(device_list, avg_cv_values, lower_limit, upper_limit):
    df = pd.DataFrame({"Device": device_list, "Avg_CV%": avg_cv_values})
    df["Within Limits"] = (df["Avg_CV%"] >= lower_limit) & (df["Avg_CV%"] <= upper_limit)
    st.subheader("Devices within limits")
    st.dataframe(df, hide_index=True)


def L1(path, target_file):
    with st.expander("L1 Test results"):
        L1_528, L1_620, L1_367, L1_405 = st.tabs(["528", "620", "367", "405"])

        with L1_528:
            all_L1_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L1') != -1 and x.find('528') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L1_files]))
            if len(all_L1_files) == 0:
                st.write("No 528 L1 files found")

            else:
                st.subheader("Level 1", help="CONTINUOUS BLANK TEST WITHOUT INCUBATOR.")
                L1_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L1_files if x.find(device) != -1]

                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))

                                L1_df.at["Version", file_name.split('.')[0]] = version
                                L1_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L1_df.at["DAC", file_name.split('.')[0]] = dac
                                L1_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L1_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        # Convert the value to a numeric type (float or int)
                                        try:
                                            value = float(value)  # You can use int() if the values are integers
                                        except ValueError:
                                            # Handle the case where the conversion fails (e.g., non-numeric values)
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)

                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L1_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L1_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L1_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L1_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L1_df.at[k, file_name.split('.')[0]] = data_point

                # summary dataframe
                L1_summary_df = pd.DataFrame()
                L1_summary_df['Device'] = device_list
                L1_summary_df['Number of Files'] = [len([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L1_df.columns if x.find(device) != -1]
                    avg_cv = L1_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L1_summary_df["Avg_CV%"] = device_avg_cv
                L1_summary_df['file'] = ['  ||  '.join([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L1_df)
                st.subheader("Summary")
                st.dataframe(L1_summary_df,hide_index=True)

                limit_filter_bar_plot(device_list, L1_summary_df["Avg_CV%"].values, 0/1000, 45/1000)

                with st.form("Graph analysis for L1 528"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L1_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L1_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L1_620:
            all_L1_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L1') != -1 and x.find('620') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L1_files]))
            if len(all_L1_files) == 0:
                st.write("No 620 L1 files found")
                
            else:
                st.subheader("Level 1", help="CONTINUOUS BLANK TEST WITHOUT INCUBATOR.")
                L1_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L1_files if x.find(device) != -1]
                    files.sort()

                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))

                                L1_df.at["Version", file_name.split('.')[0]] = version
                                L1_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L1_df.at["DAC", file_name.split('.')[0]] = dac
                                L1_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L1_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        # Convert the value to a numeric type (float or int)
                                        try:
                                            value = float(value)  # You can use int() if the values are integers
                                        except ValueError:
                                            # Handle the case where the conversion fails (e.g., non-numeric values)
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L1_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L1_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L1_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L1_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L1_df.at[k, file_name.split('.')[0]] = data_point

                # summary dataframe
                L1_summary_df = pd.DataFrame()
                L1_summary_df['Device'] = device_list
                L1_summary_df['Number of Files'] = [len([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L1_df.columns if x.find(device) != -1]
                    avg_cv = L1_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L1_summary_df["Avg_CV%"] = device_avg_cv
                L1_summary_df['file'] = ['  ||  '.join([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L1_df)
                st.subheader("Summary")
                st.dataframe(L1_summary_df,hide_index=True)

                limit_filter_bar_plot(device_list, L1_summary_df["Avg_CV%"].values, 0/1000, 60/1000)
                
                with st.form("Graph analysis for L1 620"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L1_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L1_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L1_367:
            all_L1_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L1') != -1 and x.find('367') != -1 and x.find('367') == -1]
            device_list = list(set([x.split('_')[-4] for x in all_L1_files]))
            if len(all_L1_files) == 0:
                st.write("No 367 L1 files found")
                
            else:
                st.subheader("Level 1", help="CONTINUOUS BLANK TEST WITHOUT INCUBATOR.")
                L1_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L1_files if x.find(device) != -1]
                    files.sort()

                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))

                                L1_df.at["Version", file_name.split('.')[0]] = version
                                L1_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L1_df.at["DAC", file_name.split('.')[0]] = dac
                                L1_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L1_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        # Convert the value to a numeric type (float or int)
                                        try:
                                            value = float(value)  # You can use int() if the values are integers
                                        except ValueError:
                                            # Handle the case where the conversion fails (e.g., non-numeric values)
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L1_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L1_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L1_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L1_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L1_df.at[k, file_name.split('.')[0]] = data_point

                # summary dataframe
                L1_summary_df = pd.DataFrame()
                L1_summary_df['Device'] = device_list
                L1_summary_df['Number of Files'] = [len([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L1_df.columns if x.find(device) != -1]
                    avg_cv = L1_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L1_summary_df["Avg_CV%"] = device_avg_cv
                L1_summary_df['file'] = ['  ||  '.join([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L1_df)
                st.subheader("Summary")
                st.dataframe(L1_summary_df,hide_index=True)

                limit_filter_bar_plot(device_list, L1_summary_df["Avg_CV%"].values, 0/1000, 60/1000)

                with st.form("Graph analysis for L1 367"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L1_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L1_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L1_405:
            all_L1_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L1') != -1 and x.find('405') != -1 and x.find('405') == -1]
            device_list = list(set([x.split('_')[-4] for x in all_L1_files]))
            if len(all_L1_files) == 0:
                st.write("No 405 L1 files found")
                
            else:
                st.subheader("Level 1", help="CONTINUOUS BLANK TEST WITHOUT INCUBATOR.")
                L1_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L1_files if x.find(device) != -1]
                    files.sort()

                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))

                                L1_df.at["Version", file_name.split('.')[0]] = version
                                L1_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L1_df.at["DAC", file_name.split('.')[0]] = dac
                                L1_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L1_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        # Convert the value to a numeric type (float or int)
                                        try:
                                            value = float(value)  # You can use int() if the values are integers
                                        except ValueError:
                                            # Handle the case where the conversion fails (e.g., non-numeric values)
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L1_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L1_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L1_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L1_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L1_df.at[k, file_name.split('.')[0]] = data_point

                L1_summary_df = pd.DataFrame()
                L1_summary_df['Device'] = device_list
                L1_summary_df['Number of Files'] = [len([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L1_df.columns if x.find(device) != -1]
                    avg_cv = L1_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L1_summary_df["Avg_CV%"] = device_avg_cv
                L1_summary_df['file'] = ['  ||  '.join([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L1_df)
                st.subheader("Summary")
                st.dataframe(L1_summary_df,hide_index=True)


                limit_filter_bar_plot(device_list, L1_summary_df["Avg_CV%"].values, 0/1000, 60/1000)
    
                with st.form("Graph analysis for L1 405"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L1_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L1_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)


def L3(path, target_file):
    with st.expander("L3 Test results"):
        L3_528, L3_620, L3_367, L3_405 = st.tabs(["528", "620", "367", "405"])

        with L3_528:
            all_L3_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L3') != -1 and x.find('528') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L3_files]))
            if len(all_L3_files) == 0:
                st.write("No 528 L3 files found")
            
            else:
                st.subheader("Level 3", help = "REINSERTION TEST WITH WATER")
                L3_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])
                
                for device in device_list:
                    files = [x for x in all_L3_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))

                                L3_df.at["Version", file_name.split('.')[0]] = version
                                L3_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L3_df.at["DAC", file_name.split('.')[0]] = dac
                                L3_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L3_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        try:
                                            value = float(value)  
                                        except ValueError:
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)


                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L3_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L3_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L3_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L3_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L3_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L3_summary_df = pd.DataFrame()
                L3_summary_df = pd.DataFrame()
                L3_summary_df['Device'] = device_list
                L3_summary_df['Number of Files'] = [len([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L3_df.columns if x.find(device) != -1]
                    avg_cv = L3_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L3_summary_df["Avg_CV%"] = device_avg_cv
                L3_summary_df['file'] = ['  ||  '.join([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L3_df)
                st.subheader("Summary")
                st.dataframe(L3_summary_df,hide_index=True)
            
                limit_filter_bar_plot(device_list, L3_summary_df["Avg_CV%"].values, 0/1000, 140/1000)
                
                with st.form("Graph analysis for L3 528"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L3_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L3_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L3_620:
            all_L3_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L3') != -1 and x.find('620') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L3_files]))
            
            if len(all_L3_files) == 0:
                st.write("No 620 L3 files found")
            
            else:
                st.subheader("Level 3", help = "REINSERTION TEST WITH WATER")
                L3_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])
                
                for device in device_list:
                    files = [x for x in all_L3_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))

                                L3_df.at["Version", file_name.split('.')[0]] = version
                                L3_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L3_df.at["DAC", file_name.split('.')[0]] = dac
                                L3_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L3_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        try:
                                            value = float(value)  
                                        except ValueError:
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)


                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L3_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L3_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L3_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L3_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L3_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L3_summary_df = pd.DataFrame()
                L3_summary_df = pd.DataFrame()
                L3_summary_df['Device'] = device_list
                L3_summary_df['Number of Files'] = [len([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L3_df.columns if x.find(device) != -1]
                    avg_cv = L3_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L3_summary_df["Avg_CV%"] = device_avg_cv
                L3_summary_df['file'] = ['  ||  '.join([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L3_df)
                st.subheader("Summary")
                st.dataframe(L3_summary_df,hide_index=True)
            
                limit_filter_bar_plot(device_list, L3_summary_df["Avg_CV%"].values, 0/1000, 200/1000)
                
                with st.form("Graph analysis for L3 620"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L3_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L3_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L3_367:
            all_L3_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L3') != -1 and x.find('367') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L3_files]))
            if len(all_L3_files) == 0:
                st.write("No 367 L3 files found")
            
            else:
                st.subheader("Level 3", help = "REINSERTION TEST WITH WATER")
                L3_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])
                
                for device in device_list:
                    files = [x for x in all_L3_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))

                                L3_df.at["Version", file_name.split('.')[0]] = version
                                L3_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L3_df.at["DAC", file_name.split('.')[0]] = dac
                                L3_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L3_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        try:
                                            value = float(value)  
                                        except ValueError:
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)


                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L3_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L3_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L3_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L3_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L3_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L3_summary_df = pd.DataFrame()
                L3_summary_df = pd.DataFrame()
                L3_summary_df['Device'] = device_list
                L3_summary_df['Number of Files'] = [len([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L3_df.columns if x.find(device) != -1]
                    avg_cv = L3_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L3_summary_df["Avg_CV%"] = device_avg_cv
                L3_summary_df['file'] = ['  ||  '.join([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L3_df)
                st.subheader("Summary")
                st.dataframe(L3_summary_df,hide_index=True)
            
                limit_filter_bar_plot(device_list, L3_summary_df["Avg_CV%"].values, 0/1000, 380/1000)
                
                with st.form("Graph analysis for L3 367"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L3_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L3_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L3_405:
            all_L3_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L3') != -1 and x.find('405') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L3_files]))
            if len(all_L3_files) == 0:
                st.write("No 405 L3 files found")
            
            else:
                st.subheader("Level 3", help = "REINSERTION TEST WITH WATER")
                L3_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])
                
                for device in device_list:
                    files = [x for x in all_L3_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))

                                L3_df.at["Version", file_name.split('.')[0]] = version
                                L3_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L3_df.at["DAC", file_name.split('.')[0]] = dac
                                L3_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L3_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        try:
                                            value = float(value)  
                                        except ValueError:
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)


                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L3_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L3_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L3_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L3_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L3_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L3_summary_df = pd.DataFrame()
                L3_summary_df = pd.DataFrame()
                L3_summary_df['Device'] = device_list
                L3_summary_df['Number of Files'] = [len([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L3_df.columns if x.find(device) != -1]
                    avg_cv = L3_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L3_summary_df["Avg_CV%"] = device_avg_cv
                L3_summary_df['file'] = ['  ||  '.join([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L3_df)
                st.subheader("Summary")
                st.dataframe(L3_summary_df,hide_index=True)
            
                limit_filter_bar_plot(device_list, L3_summary_df["Avg_CV%"].values, 0/1000, 200/1000)
                
                with st.form("Graph analysis for L3 405"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L3_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L3_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)


def L5(path, target_file):
    with st.expander("L5 Test results"):
        L5_528, L5_620, L5_367, L5_405 = st.tabs(["528", "620", "367", "405"])

        with L5_528:
            all_L5_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L5') != -1 and x.find('528') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L5_files]))
            if len(all_L5_files) == 0:
                st.write("No 528 L5 files found")
            
            else:
                st.subheader("Level 5", help = "CONTINUOUS BLANK TEST WITH INCUBATOR")
                L5_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L5_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))
                                battery = re.search(r'Battery : (\d+)', file_content).group(1)
                                core_temp_match = re.search(r'Core Temperature : (\d+)', file_content)

                                if core_temp_match:
                                    core_Temp = core_temp_match.group(1)
                                else:
                                    # Handle the case when the pattern is not found, e.g., set core_Temp to a default value
                                    core_Temp = "N/A"


                                L5_df.at["Version", file_name.split('.')[0]] = version
                                L5_df.at["Battery", file_name.split('.')[0]] = battery
                                L5_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L5_df.at["DAC", file_name.split('.')[0]] = dac
                                L5_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L5_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity
                                L5_df.at["Core Temp.", file_name.split('.')[0]] = core_Temp

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        # Convert the value to a numeric type (float or int)
                                        try:
                                            value = float(value)  # You can use int() if the values are integers
                                        except ValueError:
                                            # Handle the case where the conversion fails (e.g., non-numeric values)
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)
                                
                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L5_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L5_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L5_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L5_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L5_df.at[k, file_name.split('.')[0]] = data_point
                
                # summery dataframe
                L5_summary_df = pd.DataFrame()
                L5_summary_df = pd.DataFrame()
                L5_summary_df['Device'] = device_list
                L5_summary_df['Number of Files'] = [len([x for x in all_L5_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L5_df.columns if x.find(device) != -1]
                    avg_cv = L5_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L5_summary_df["Avg_CV%"] = device_avg_cv
                L5_summary_df['file'] = ['  ||  '.join([x for x in all_L5_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L5_df)
                st.subheader("Summary")
                st.dataframe(L5_summary_df,hide_index=True)
                
                limit_filter_bar_plot(device_list, L5_summary_df["Avg_CV%"].values, 0/1000, 50/1000)
                
                with st.form("Graph analysis for L5 528"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L5_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L5_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L5_620:
            all_L5_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L5') != -1 and x.find('620') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L5_files]))
            if len(all_L5_files) == 0:
                st.write("No 528 L5 files found")
            
            else:
                st.subheader("Level 5", help = "CONTINUOUS BLANK TEST WITH INCUBATOR")
                L5_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L5_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))
                                battery = re.search(r'Battery : (\d+)', file_content).group(1)
                                core_temp_match = re.search(r'Core Temperature : (\d+)', file_content)

                                if core_temp_match:
                                    core_Temp = core_temp_match.group(1)
                                else:
                                    # Handle the case when the pattern is not found, e.g., set core_Temp to a default value
                                    core_Temp = "N/A"


                                L5_df.at["Version", file_name.split('.')[0]] = version
                                L5_df.at["Battery", file_name.split('.')[0]] = battery
                                L5_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L5_df.at["DAC", file_name.split('.')[0]] = dac
                                L5_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L5_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity
                                L5_df.at["Core Temp.", file_name.split('.')[0]] = core_Temp

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        # Convert the value to a numeric type (float or int)
                                        try:
                                            value = float(value)  # You can use int() if the values are integers
                                        except ValueError:
                                            # Handle the case where the conversion fails (e.g., non-numeric values)
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L5_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L5_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L5_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L5_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L5_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L5_summary_df = pd.DataFrame()
                L5_summary_df = pd.DataFrame()
                L5_summary_df['Device'] = device_list
                L5_summary_df['Number of Files'] = [len([x for x in all_L5_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L5_df.columns if x.find(device) != -1]
                    avg_cv = L5_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L5_summary_df["Avg_CV%"] = device_avg_cv
                L5_summary_df['file'] = ['  ||  '.join([x for x in all_L5_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L5_df)
                st.subheader("Summary")
                st.dataframe(L5_summary_df,hide_index=True)
                
                limit_filter_bar_plot(device_list, L5_summary_df["Avg_CV%"].values, 0/1000, 60/1000)
                
                with st.form("Graph analysis for L5 620"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L5_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L5_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L5_367:
            all_L5_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L5') != -1 and x.find('367') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L5_files]))
            
            if len(all_L5_files) == 0:
                st.write("No 528 L5 files found")
            
            else:
                st.subheader("Level 5", help = "CONTINUOUS BLANK TEST WITH INCUBATOR")
                L5_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L5_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))
                                battery = re.search(r'Battery : (\d+)', file_content).group(1)
                                core_temp_match = re.search(r'Core Temperature : (\d+)', file_content)

                                if core_temp_match:
                                    core_Temp = core_temp_match.group(1)
                                else:
                                    # Handle the case when the pattern is not found, e.g., set core_Temp to a default value
                                    core_Temp = "N/A"


                                L5_df.at["Version", file_name.split('.')[0]] = version
                                L5_df.at["Battery", file_name.split('.')[0]] = battery
                                L5_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L5_df.at["DAC", file_name.split('.')[0]] = dac
                                L5_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L5_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity
                                L5_df.at["Core Temp.", file_name.split('.')[0]] = core_Temp

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        # Convert the value to a numeric type (float or int)
                                        try:
                                            value = float(value)  # You can use int() if the values are integers
                                        except ValueError:
                                            # Handle the case where the conversion fails (e.g., non-numeric values)
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L5_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L5_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L5_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L5_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L5_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L5_summary_df = pd.DataFrame()
                L5_summary_df = pd.DataFrame()
                L5_summary_df['Device'] = device_list
                L5_summary_df['Number of Files'] = [len([x for x in all_L5_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L5_df.columns if x.find(device) != -1]
                    avg_cv = L5_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L5_summary_df["Avg_CV%"] = device_avg_cv
                L5_summary_df['file'] = ['  ||  '.join([x for x in all_L5_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L5_df)
                st.subheader("Summary")
                st.dataframe(L5_summary_df,hide_index=True)
                
                limit_filter_bar_plot(device_list, L5_summary_df["Avg_CV%"].values, 0/1000, 110/1000)
                
                with st.form("Graph analysis for L5 367"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L5_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L5_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L5_405:
            all_L6_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L5') != -1 and x.find('405') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L6_files]))
            if len(all_L5_files) == 0:
                st.write("No 528 L5 files found")
            
            else:
                st.subheader("Level 5", help = "CONTINUOUS BLANK TEST WITH INCUBATOR")
                L5_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L6_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()

                            # Use regular expressions to extract relevant information
                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))
                                battery = re.search(r'Battery : (\d+)', file_content).group(1)
                                core_temp_match = re.search(r'Core Temperature : (\d+)', file_content)

                                if core_temp_match:
                                    core_Temp = core_temp_match.group(1)
                                else:
                                    # Handle the case when the pattern is not found, e.g., set core_Temp to a default value
                                    core_Temp = "N/A"


                                L5_df.at["Version", file_name.split('.')[0]] = version
                                L5_df.at["Battery", file_name.split('.')[0]] = battery
                                L5_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L5_df.at["DAC", file_name.split('.')[0]] = dac
                                L5_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L5_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity
                                L5_df.at["Core Temp.", file_name.split('.')[0]] = core_Temp

                                data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                if data_point_match:
                                    lines = data_point_match.group(1).split('\n')

                                    data_points = []
                                    for line in lines[1:-1]:
                                        parts = line.split(':')[-1].split(',')
                                        value = parts[0].strip()

                                        # Convert the value to a numeric type (float or int)
                                        try:
                                            value = float(value)  # You can use int() if the values are integers
                                        except ValueError:
                                            # Handle the case where the conversion fails (e.g., non-numeric values)
                                            print(f"Skipping non-numeric value: {value}")
                                            continue
                                        data_points.append(value)

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L5_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L5_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L5_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L5_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L5_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L5_summary_df = pd.DataFrame()
                L5_summary_df = pd.DataFrame()
                L5_summary_df['Device'] = device_list
                L5_summary_df['Number of Files'] = [len([x for x in all_L5_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L5_df.columns if x.find(device) != -1]
                    avg_cv = L5_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L5_summary_df["Avg_CV%"] = device_avg_cv
                L5_summary_df['file'] = ['  ||  '.join([x for x in all_L5_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L5_df)
                st.subheader("Summary")
                st.dataframe(L5_summary_df,hide_index=True)
                
                limit_filter_bar_plot(device_list, L5_summary_df["Avg_CV%"].values, 0/1000, 60/1000)
                
                with st.form("Graph analysis for L5 405"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L5_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L5_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)


def L6(path, target_file):
    with st.expander("L6 Test results"):
        L6_528, L6_620, L6_367, L6_405 = st.tabs(["528", "620", "367", "405"])

        with L6_528:
            all_L6_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L6') != -1 and x.find('528') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L6_files]))
            if len(all_L6_files) == 0:
                st.write("No 528 L6 files found")
            
            else:
                st.subheader("Level 6", help = "CONTINUOUS WATER TEST WITH INCUBATOR")
                L6_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L6_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()


                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))
                                battery = re.search(r'Battery : (\d+)', file_content).group(1)
                                if battery:
                                    battery = battery
                                else:
                                    battery = "N/A"

                                core_temp_match = re.search(r'Core Temperature : (\d+)', file_content)

                                if core_temp_match:
                                    core_Temp = core_temp_match.group(1)
                                else:
                                    core_Temp = "N/A"


                                L6_df.at["Version", file_name.split('.')[0]] = version
                                L6_df.at["Battery", file_name.split('.')[0]] = battery
                                L6_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L6_df.at["DAC", file_name.split('.')[0]] = dac
                                L6_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L6_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity
                                L6_df.at["Core Temp.", file_name.split('.')[0]] = core_Temp

                                data_points_index = re.findall("Press Enter to start([\s\S]+)Done.", file_content)
                                for j, data_point in enumerate(data_points_index):
                                    # Extract numerical values from the data
                                    numbers = re.findall(r'\b\d+\b', data_point)

                                    # Convert the numbers to integers
                                    data_points = [int(num) for i, num in enumerate(numbers) if i % 5 == 0]

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L6_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L6_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L6_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L6_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L6_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L6_summary_df = pd.DataFrame()
                L6_summary_df = pd.DataFrame()
                L6_summary_df['Device'] = device_list
                L6_summary_df['Number of Files'] = [len([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L6_df.columns if x.find(device) != -1]
                    avg_cv = L6_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L6_summary_df["Avg_CV%"] = device_avg_cv
                L6_summary_df['file'] = ['  ||  '.join([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L6_df)
                st.subheader("Summary")
                st.dataframe(L6_summary_df,hide_index=True)
                
                limit_filter_bar_plot(device_list, L6_summary_df["Avg_CV%"].values, 0/1000, 85/1000)
                
                with st.form("Graph analysis for L6 528"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L6_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L6_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L6_620:
            all_L6_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L6') != -1 and x.find('620') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L6_files]))
            if len(all_L6_files) == 0:
                st.write("No 620 L6 files found")
            
            else:
                st.subheader("Level 6", help = "CONTINUOUS WATER TEST WITH INCUBATOR")
                L6_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L6_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()


                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))
                                battery = re.search(r'Battery : (\d+)', file_content).group(1)
                                if battery:
                                    battery = battery
                                else:
                                    battery = "N/A"

                                core_temp_match = re.search(r'Core Temperature : (\d+)', file_content)

                                if core_temp_match:
                                    core_Temp = core_temp_match.group(1)
                                else:
                                    core_Temp = "N/A"


                                L6_df.at["Version", file_name.split('.')[0]] = version
                                L6_df.at["Battery", file_name.split('.')[0]] = battery
                                L6_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L6_df.at["DAC", file_name.split('.')[0]] = dac
                                L6_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L6_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity
                                L6_df.at["Core Temp.", file_name.split('.')[0]] = core_Temp

                                data_points_index = re.findall("Press Enter to start([\s\S]+)Done.", file_content)
                                for j, data_point in enumerate(data_points_index):
                                    # Extract numerical values from the data
                                    numbers = re.findall(r'\b\d+\b', data_point)

                                    # Convert the numbers to integers
                                    data_points = [int(num) for i, num in enumerate(numbers) if i % 5 == 0]

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L6_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L6_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L6_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L6_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L6_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L6_summary_df = pd.DataFrame()
                L6_summary_df = pd.DataFrame()
                L6_summary_df['Device'] = device_list
                L6_summary_df['Number of Files'] = [len([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L6_df.columns if x.find(device) != -1]
                    avg_cv = L6_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L6_summary_df["Avg_CV%"] = device_avg_cv
                L6_summary_df['file'] = ['  ||  '.join([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L6_df)
                st.subheader("Summary")
                st.dataframe(L6_summary_df,hide_index=True)
                
                limit_filter_bar_plot(device_list, L6_summary_df["Avg_CV%"].values, 0/1000, 85/1000)
                
                with st.form("Graph analysis for L6 620"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L6_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L6_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L6_367:
            all_L6_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L6') != -1 and x.find('367') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L6_files]))
            if len(all_L6_files) == 0:
                st.write("No 367 L6 files found")
            
            else:
                st.subheader("Level 6", help = "CONTINUOUS WATER TEST WITH INCUBATOR")
                L6_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L6_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()


                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))
                                battery = re.search(r'Battery : (\d+)', file_content).group(1)
                                if battery:
                                    battery = battery
                                else:
                                    battery = "N/A"

                                core_temp_match = re.search(r'Core Temperature : (\d+)', file_content)

                                if core_temp_match:
                                    core_Temp = core_temp_match.group(1)
                                else:
                                    core_Temp = "N/A"


                                L6_df.at["Version", file_name.split('.')[0]] = version
                                L6_df.at["Battery", file_name.split('.')[0]] = battery
                                L6_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L6_df.at["DAC", file_name.split('.')[0]] = dac
                                L6_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L6_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity
                                L6_df.at["Core Temp.", file_name.split('.')[0]] = core_Temp

                                data_points_index = re.findall("Press Enter to start([\s\S]+)Done.", file_content)
                                for j, data_point in enumerate(data_points_index):
                                    # Extract numerical values from the data
                                    numbers = re.findall(r'\b\d+\b', data_point)

                                    # Convert the numbers to integers
                                    data_points = [int(num) for i, num in enumerate(numbers) if i % 5 == 0]

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L6_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L6_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L6_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L6_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L6_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L6_summary_df = pd.DataFrame()
                L6_summary_df = pd.DataFrame()
                L6_summary_df['Device'] = device_list
                L6_summary_df['Number of Files'] = [len([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L6_df.columns if x.find(device) != -1]
                    avg_cv = L6_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L6_summary_df["Avg_CV%"] = device_avg_cv
                L6_summary_df['file'] = ['  ||  '.join([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L6_df)
                st.subheader("Summary")
                st.dataframe(L6_summary_df,hide_index=True)
                
                limit_filter_bar_plot(device_list, L6_summary_df["Avg_CV%"].values, 0/1000, 200/1000)

                with st.form("Graph analysis for L6 367"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L6_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L6_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)

        with L6_405:
            all_L6_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L6') != -1 and x.find('405') != -1]
            device_list = list(set([x.split('_')[-4] for x in all_L6_files]))
            if len(all_L6_files) == 0:
                st.write("No 405 L6 files found")
            
            else:
                st.subheader("Level 6", help = "CONTINUOUS WATER TEST WITH INCUBATOR")
                L6_df = pd.DataFrame()
                df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                for device in device_list:
                    files = [x for x in all_L6_files if x.find(device) != -1]
                    files.sort()
                    for i, file_name in enumerate(files):
                        with open(os.path.join(path, file_name), 'r') as file:
                            file_content = file.read()


                            version_match = re.search(r'Version : (.+)', file_content)
                            wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                            dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                            current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                            adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                            if version_match and wavelength_match and dac_match and current_match and adj_intensity_match:
                                version = version_match.group(1)
                                wavelength = int(wavelength_match.group(1))
                                dac = int(dac_match.group(1))
                                current = float(current_match.group(1))
                                adj_intensity = int(adj_intensity_match.group(1))
                                battery = re.search(r'Battery : (\d+)', file_content).group(1)
                                if battery:
                                    battery = battery
                                else:
                                    battery = "N/A"

                                core_temp_match = re.search(r'Core Temperature : (\d+)', file_content)

                                if core_temp_match:
                                    core_Temp = core_temp_match.group(1)
                                else:
                                    core_Temp = "N/A"


                                L6_df.at["Version", file_name.split('.')[0]] = version
                                L6_df.at["Battery", file_name.split('.')[0]] = battery
                                L6_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                L6_df.at["DAC", file_name.split('.')[0]] = dac
                                L6_df.at["Current(mA)", file_name.split('.')[0]] = current
                                L6_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity
                                L6_df.at["Core Temp.", file_name.split('.')[0]] = core_Temp

                                data_points_index = re.findall("Press Enter to start([\s\S]+)Done.", file_content)
                                for j, data_point in enumerate(data_points_index):
                                    # Extract numerical values from the data
                                    numbers = re.findall(r'\b\d+\b', data_point)

                                    # Convert the numbers to integers
                                    data_points = [int(num) for i, num in enumerate(numbers) if i % 5 == 0]

                                    # Calculate statistics
                                    mean = np.mean(data_points)
                                    sd = np.std(data_points)
                                    cv = (sd / mean) * 100
                                    data_range = max(data_points) - min(data_points)

                                    # Update the DataFrame
                                    L6_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                    L6_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                    L6_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                    L6_df.at["Range", file_name.split('.')[0]] = data_range
                                    for k, data_point in enumerate(data_points):
                                        L6_df.at[k, file_name.split('.')[0]] = data_point

                # summery dataframe
                L6_summary_df = pd.DataFrame()
                L6_summary_df = pd.DataFrame()
                L6_summary_df['Device'] = device_list
                L6_summary_df['Number of Files'] = [len([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]

                # Calculate average CV% for each device
                device_avg_cv = []
                for device in device_list:
                    same_device_files = [x for x in L6_df.columns if x.find(device) != -1]
                    avg_cv = L6_df.loc["CV%", same_device_files].mean()
                    device_avg_cv.append(avg_cv)

                L6_summary_df["Avg_CV%"] = device_avg_cv
                L6_summary_df['file'] = ['  ||  '.join([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]
                st.dataframe(L6_df)
                st.subheader("Summary")
                st.dataframe(L6_summary_df,hide_index=True)
                
                limit_filter_bar_plot(device_list, L6_summary_df["Avg_CV%"].values, 0/1000, 85/1000)
                
                with st.form("Graph analysis for L6 405"):
                    st.subheader("Graph analysis")
                    select_device = st.selectbox("Select device", device_list)
                    submit = st.form_submit_button("Submit")

                    if submit:
                        columns_with_device = [col for col in L6_df.columns if select_device in col]
                        columns_with_device.sort()
                        for col in columns_with_device:
                            y_data = L6_df.loc[0:, col]
                            st.write(f"Graph for {col}")
                            fig = px.line(x=range(len(y_data)), y=y_data, title=f"Graph for {col}", labels={"x": "Index", "y": "Value"})
                            st.plotly_chart(fig, use_container_width=True)


def L7(path, target_file):
    with st.expander("L7 Test results"):
        L7_528, L7_620, L7_367, L7_405 = st.tabs(["528", "620", "367", "405"])

        with L7_528:
            with st.form("L7_528_Form"):
                reference_file_L7_528 = st.file_uploader("L7 files detected, please Upload Reference file", type=["txt"])
                st.form_submit_button("Analyse")
                if reference_file_L7_528:
                    values = [float(line.strip().decode('utf-8')) for line in reference_file_L7_528 if line.strip().replace(b'.', b'', 1).isdigit()]
                    repeated_values = np.tile(values, 3)
                    repeated_values.sort()
                slopes = []
                all_L7_files = [x for x in os.listdir(path) if x.endswith(".txt") and 'L7' in x and '528' in x]
                device_list = list(set([x.split('_')[-3] for x in all_L7_files]))
                if len(all_L7_files) == 0:
                    st.write("No 528 L7 files found")
                else:
                    st.subheader("Level 7", help="DEVICE TRACEABILITY WITH UV-SPECTROPHOTOMETER")
                    L7_df = pd.DataFrame()
                    df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                    for device in device_list:
                        files = [x for x in all_L7_files if device in x]
                        files.sort()

                        for i, file_name in enumerate(files):
                            with open(os.path.join(path, file_name), 'r') as file:
                                file_content = file.read()

                                version_match = re.search(r'Version : (.+)', file_content)
                                wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                                dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                                current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                                adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                                if all([version_match, wavelength_match, dac_match, current_match, adj_intensity_match]):
                                    version = version_match.group(1)
                                    wavelength = int(wavelength_match.group(1))
                                    dac = int(dac_match.group(1))
                                    current = float(current_match.group(1))
                                    adj_intensity = int(adj_intensity_match.group(1))

                                    L7_df.at["Version", file_name.split('.')[0]] = version
                                    L7_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                    L7_df.at["DAC", file_name.split('.')[0]] = dac
                                    L7_df.at["Current(mA)", file_name.split('.')[0]] = current
                                    L7_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                    data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                    if data_point_match:
                                        lines = data_point_match.group(1).split('\n')

                                        data_points = []
                                        for line in lines[1:-1]:
                                            parts = line.split(':')[-1].split(',')
                                            value = parts[0].strip()

                                            try:
                                                value = float(value)  
                                            except ValueError:
                                                print(f"Skipping non-numeric value: {value}")
                                                continue
                                            data_points.append(value)

                                        mean = np.mean(data_points)
                                        sd = np.std(data_points)
                                        cv = (sd / mean) * 100
                                        data_range = max(data_points) - min(data_points)

                                        L7_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                        L7_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                        L7_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                        L7_df.at["Range", file_name.split('.')[0]] = data_range

                                        for k, data_point in enumerate(data_points):
                                            L7_df.at[k, file_name.split('.')[0]] = data_point

                    st.dataframe(L7_df)

                    if reference_file_L7_528:
                        analysis_data = pd.DataFrame({"Reference": repeated_values})
                        analysis_data.index = range(1, 31)

                        device_intensity = L7_df.loc[1:, :].astype(float)
                        blank_intensity = L7_df.loc[0, :].astype(float)

                        absorption_values = pd.DataFrame(np.log10(blank_intensity / device_intensity), columns=L7_df.columns)
                        analysis_data = pd.concat([analysis_data, absorption_values], axis=1)
                        st.dataframe(analysis_data)

                        # plot the graph here
                        st.subheader("Best Fit Line Plot")
                        
                        for i in range(1, len(analysis_data.columns)):
                            st.write(f"Device: {analysis_data.columns[i]}")
                            slope = plot_best_fit_line("Reference", analysis_data.columns[i], analysis_data)
                            slopes.append(slope)
                            st.markdown("---")

                        # summary dataframe
                        L7_summary_df = pd.DataFrame()
                        L7_summary_df['Device'] = device_list
                        L7_summary_df['Number of Files'] = [len([x for x in all_L7_files if device in x]) for device in device_list]

                        L7_summary_df["Best fit slope"] = slopes
                        L7_summary_df['file'] = ['  ||  '.join([x for x in all_L7_files if device in x]) for device in device_list]
                        st.dataframe(L7_summary_df)
                        
                        limit_filter_bar_plot(device_list, slopes, 7905/10000, 8745/10000)

        with L7_620:
            with st.form("L7_620_Form"):
                reference_file_L7_620 = st.file_uploader("L7 files detected, please Upload Reference file", type=["txt"])
                st.form_submit_button("Analyse")
                if reference_file_L7_620:
                    values = [float(line.strip().decode('utf-8')) for line in reference_file_L7_620 if line.strip().replace(b'.', b'', 1).isdigit()]
                    repeated_values = np.tile(values, 3)
                    repeated_values.sort()
                slopes = []
                all_L7_files = [x for x in os.listdir(path) if x.endswith(".txt") and 'L7' in x and '620' in x]
                device_list = list(set([x.split('_')[-3] for x in all_L7_files]))
                if len(all_L7_files) == 0:
                    st.write("No 620 L7 files found")
            
                else:
                    st.subheader("Level 7", help="DEVICE TRACEABILITY WITH UV-SPECTROPHOTOMETER")
                    L7_df = pd.DataFrame()
                    df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                    for device in device_list:
                        files = [x for x in all_L7_files if device in x]
                        files.sort()

                        for i, file_name in enumerate(files):
                            with open(os.path.join(path, file_name), 'r') as file:
                                file_content = file.read()

                                version_match = re.search(r'Version : (.+)', file_content)
                                wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                                dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                                current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                                adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                                if all([version_match, wavelength_match, dac_match, current_match, adj_intensity_match]):
                                    version = version_match.group(1)
                                    wavelength = int(wavelength_match.group(1))
                                    dac = int(dac_match.group(1))
                                    current = float(current_match.group(1))
                                    adj_intensity = int(adj_intensity_match.group(1))

                                    L7_df.at["Version", file_name.split('.')[0]] = version
                                    L7_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                    L7_df.at["DAC", file_name.split('.')[0]] = dac
                                    L7_df.at["Current(mA)", file_name.split('.')[0]] = current
                                    L7_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                    data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                    if data_point_match:
                                        lines = data_point_match.group(1).split('\n')

                                        data_points = []
                                        for line in lines[1:-1]:
                                            parts = line.split(':')[-1].split(',')
                                            value = parts[0].strip()

                                            try:
                                                value = float(value)  
                                            except ValueError:

                                                print(f"Skipping non-numeric value: {value}")
                                                continue
                                            data_points.append(value)
                                        
                                        mean = np.mean(data_points)
                                        sd = np.std(data_points)
                                        cv = (sd / mean) * 100
                                        data_range = max(data_points) - min(data_points)

                                        L7_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                        L7_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                        L7_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                        L7_df.at["Range", file_name.split('.')[0]] = data_range

                                        for k, data_point in enumerate(data_points):
                                            L7_df.at[k, file_name.split('.')[0]] = data_point

                    st.dataframe(L7_df)

                    if reference_file_L7_620:
                        analysis_data = pd.DataFrame({"Reference": repeated_values})
                        analysis_data.index = range(1, 31)

                        device_intensity = L7_df.loc[1:, :].astype(float)
                        blank_intensity = L7_df.loc[0, :].astype(float)

                        absorption_values = pd.DataFrame(np.log10(blank_intensity / device_intensity), columns=L7_df.columns)
                        analysis_data = pd.concat([analysis_data, absorption_values], axis=1)
                        st.dataframe(analysis_data)

                        st.subheader("Best Fit Line Plot")

                        for i in range(1, len(analysis_data.columns)):
                            st.write(f"Device: {analysis_data.columns[i]}")
                            slope = plot_best_fit_line("Reference", analysis_data.columns[i], analysis_data)
                            slopes.append(slope)

                        # summary dataframe
                        L7_summary_df = pd.DataFrame()
                        L7_summary_df['Device'] = device_list
                        L7_summary_df['Number of Files'] = [len([x for x in all_L7_files if device in x]) for device in device_list]

                        L7_summary_df["Best fit slope"] = slopes
                        L7_summary_df['file'] = ['  ||  '.join([x for x in all_L7_files if device in x]) for device in device_list]
                        st.dataframe(L7_summary_df)
                        
                        limit_filter_bar_plot(device_list, slopes, 9900/10000, 10500/10000)

        with L7_367:
            with st.form("L7_367_Form"):
                reference_file_L7_367 = st.file_uploader("L7 files detected, please Upload Reference file", type=["txt"])
                st.form_submit_button("Analyse")
                if reference_file_L7_367:
                    values = [float(line.strip().decode('utf-8')) for line in reference_file_L7_367 if line.strip().replace(b'.', b'', 1).isdigit()]
                    repeated_values = np.tile(values, 3)
                    repeated_values.sort()
                slopes = []
                all_L7_files = [x for x in os.listdir(path) if x.endswith(".txt") and 'L7' in x and '367' in x]
                device_list = list(set([x.split('_')[-3] for x in all_L7_files]))
                if len(all_L7_files) == 0:
                    st.write("No 367 L7 files found")
                
                else:
                    st.subheader("Level 7", help="DEVICE TRACEABILITY WITH UV-SPECTROPHOTOMETER")
                    L7_df = pd.DataFrame()
                    df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                    for device in device_list:
                        files = [x for x in all_L7_files if device in x]
                        files.sort()

                        for i, file_name in enumerate(files):
                            with open(os.path.join(path, file_name), 'r') as file:
                                file_content = file.read()

                                version_match = re.search(r'Version : (.+)', file_content)
                                wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                                dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                                current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                                adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                                if all([version_match, wavelength_match, dac_match, current_match, adj_intensity_match]):
                                    version = version_match.group(1)
                                    wavelength = int(wavelength_match.group(1))
                                    dac = int(dac_match.group(1))
                                    current = float(current_match.group(1))
                                    adj_intensity = int(adj_intensity_match.group(1))

                                    L7_df.at["Version", file_name.split('.')[0]] = version
                                    L7_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                    L7_df.at["DAC", file_name.split('.')[0]] = dac
                                    L7_df.at["Current(mA)", file_name.split('.')[0]] = current
                                    L7_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                    data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                    if data_point_match:
                                        lines = data_point_match.group(1).split('\n')

                                        data_points = []
                                        for line in lines[1:-1]:
                                            parts = line.split(':')[-1].split(',')
                                            value = parts[0].strip()

                                            # Convert the value to a numeric type (float or int)
                                            try:
                                                value = float(value)  # You can use int() if the values are integers
                                            except ValueError:
                                                # Handle the case where the conversion fails (e.g., non-numeric values)
                                                print(f"Skipping non-numeric value: {value}")
                                                continue
                                            data_points.append(value)
                                        
                                        mean = np.mean(data_points)
                                        sd = np.std(data_points)
                                        cv = (sd / mean) * 100
                                        data_range = max(data_points) - min(data_points)

                                        # Update the DataFrame
                                        L7_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                        L7_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                        L7_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                        L7_df.at["Range", file_name.split('.')[0]] = data_range

                                        for k, data_point in enumerate(data_points):
                                            L7_df.at[k, file_name.split('.')[0]] = data_point

                    st.dataframe(L7_df)

                    if reference_file_L7_367:
                        analysis_data = pd.DataFrame({"Reference": repeated_values})
                        analysis_data.index = range(1, 31)

                        device_intensity = L7_df.loc[1:, :].astype(float)
                        blank_intensity = L7_df.loc[0, :].astype(float)

                        absorption_values = pd.DataFrame(np.log10(blank_intensity / device_intensity), columns=L7_df.columns)
                        analysis_data = pd.concat([analysis_data, absorption_values], axis=1)
                        st.dataframe(analysis_data)

                        # plot the graph here
                        st.subheader("Best Fit Line Plot")

                        for i in range(1, len(analysis_data.columns)):
                            st.write(f"Device: {analysis_data.columns[i]}")
                            slope = plot_best_fit_line("Reference", analysis_data.columns[i], analysis_data)
                            slopes.append(slope)
                            st.markdown("---")

                        # summary dataframe
                        L7_summary_df = pd.DataFrame()
                        L7_summary_df['Device'] = device_list
                        L7_summary_df['Number of Files'] = [len([x for x in all_L7_files if device in x]) for device in device_list]

                        L7_summary_df["Best fit slope"] = slopes
                        L7_summary_df['file'] = ['  ||  '.join([x for x in all_L7_files if device in x]) for device in device_list]
                        st.dataframe(L7_summary_df)
                    
                        limit_filter_bar_plot(device_list, slopes, 7286/10000, 9346/10000)

        with L7_405:
            with st.form("L7_405_Form"):
                reference_file_L7_405 = st.file_uploader("L7 files detected, please Upload Reference file", type=["txt"])
                st.form_submit_button("Analyse")
                if reference_file_L7_405:
                    values = [float(line.strip().decode('utf-8')) for line in reference_file_L7_405 if line.strip().replace(b'.', b'', 1).isdigit()]
                    repeated_values = np.tile(values, 3)
                    repeated_values.sort()
                slopes = []
                all_L7_files = [x for x in os.listdir(path) if x.endswith(".txt") and 'L7' in x and '405' in x]
                device_list = list(set([x.split('_')[-3] for x in all_L7_files]))
                if len(all_L7_files) == 0:
                    st.write("No 405 L7 files found")
                
                else:
                    st.subheader("Level 7", help="DEVICE TRACEABILITY WITH UV-SPECTROPHOTOMETER")
                    L7_df = pd.DataFrame()
                    df_avg_cv = pd.DataFrame(index=["Avg_CV%"])

                    for device in device_list:
                        files = [x for x in all_L7_files if device in x]
                        files.sort()

                        for i, file_name in enumerate(files):
                            with open(os.path.join(path, file_name), 'r') as file:
                                file_content = file.read()

                                version_match = re.search(r'Version : (.+)', file_content)
                                wavelength_match = re.search(r'Enter Wavelength : (\d+)', file_content)
                                dac_match = re.search(r'Adjusted DAC Value : (\d+)', file_content)
                                current_match = re.search(r'Current drawn : ([\d.]+) mA', file_content)
                                adj_intensity_match = re.search(r'Adjusted Intensity : (\d+)', file_content)

                                if all([version_match, wavelength_match, dac_match, current_match, adj_intensity_match]):
                                    version = version_match.group(1)
                                    wavelength = int(wavelength_match.group(1))
                                    dac = int(dac_match.group(1))
                                    current = float(current_match.group(1))
                                    adj_intensity = int(adj_intensity_match.group(1))

                                    L7_df.at["Version", file_name.split('.')[0]] = version
                                    L7_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                                    L7_df.at["DAC", file_name.split('.')[0]] = dac
                                    L7_df.at["Current(mA)", file_name.split('.')[0]] = current
                                    L7_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity

                                    data_point_match = re.search("Press Enter to start([\s\S]+)Done.", file_content)
                                    if data_point_match:
                                        lines = data_point_match.group(1).split('\n')

                                        data_points = []
                                        for line in lines[1:-1]:
                                            parts = line.split(':')[-1].split(',')
                                            value = parts[0].strip()

                                            # Convert the value to a numeric type (float or int)
                                            try:
                                                value = float(value)  # You can use int() if the values are integers
                                            except ValueError:
                                                # Handle the case where the conversion fails (e.g., non-numeric values)
                                                print(f"Skipping non-numeric value: {value}")
                                                continue
                                            data_points.append(value)
                                        
                                        mean = np.mean(data_points)
                                        sd = np.std(data_points)
                                        cv = (sd / mean) * 100
                                        data_range = max(data_points) - min(data_points)

                                        # Update the DataFrame
                                        L7_df.at["Mean", file_name.split('.')[0]] = mean.round(2)
                                        L7_df.at["SD", file_name.split('.')[0]] = sd.round(4)
                                        L7_df.at["CV%", file_name.split('.')[0]] = cv.round(4)
                                        L7_df.at["Range", file_name.split('.')[0]] = data_range

                                        for k, data_point in enumerate(data_points):
                                            L7_df.at[k, file_name.split('.')[0]] = data_point

                    st.dataframe(L7_df)

                    if reference_file_L7_405:
                        values = [float(line.strip().decode('utf-8')) for line in reference_file if line.strip().replace(b'.', b'', 1).isdigit()]
                        repeated_values = np.tile(values, 3)
                        repeated_values.sort()
                        analysis_data = pd.DataFrame({"Reference": repeated_values})
                        analysis_data.index = range(1, 31)

                        device_intensity = L7_df.loc[1:, :].astype(float)
                        blank_intensity = L7_df.loc[0, :].astype(float)

                        absorption_values = pd.DataFrame(np.log10(blank_intensity / device_intensity), columns=L7_df.columns)
                        analysis_data = pd.concat([analysis_data, absorption_values], axis=1)
                        st.dataframe(analysis_data)

                        # plot the graph here
                        st.subheader("Best Fit Line Plot")

                        for i in range(1, len(analysis_data.columns)):
                            st.write(f"Device: {analysis_data.columns[i]}")
                            slope = plot_best_fit_line("Reference", analysis_data.columns[i], analysis_data)
                            slopes.append(slope)
                            st.markdown("---")

                        # summary dataframe
                        L7_summary_df = pd.DataFrame()
                        L7_summary_df['Device'] = device_list
                        L7_summary_df['Number of Files'] = [len([x for x in all_L7_files if device in x]) for device in device_list]

                        L7_summary_df["Best fit slope"] = slopes
                        L7_summary_df['file'] = ['  ||  '.join([x for x in all_L7_files if device in x]) for device in device_list]
                        st.dataframe(L7_summary_df)
                        limit_filter_bar_plot(device_list, slopes, 9900/10000, 10500/10000)


def process(path,name):
    all_files = [x for x in os.listdir(path) if x.endswith(".txt")]

    L1_files = [x for x in all_files if 'L1' in x]
    L3_files = [x for x in all_files if 'L3' in x]
    L5_files = [x for x in all_files if 'L5' in x]
    L6_files = [x for x in all_files if 'L6' in x]
    L7_files = [x for x in all_files if 'L7' in x]

        
    if L1_files:
        l1_name = L1_files[0]
        L1(path,l1_name)

    if L3_files:
        l3_name = L3_files[0]
        L3(path,l3_name)

    if L5_files:
        l5_name = L5_files[0]
        L5(path,l5_name)

    if L6_files:
        l6_name = L6_files[0]
        L6(path,l6_name)
        
    if L7_files:    
    
        l7_name = L7_files[0]
        L7(path,l7_name)


def main():
    with st.sidebar.form("file_upload_form"):
        st.markdown("## Upload Files", help="Please upload the files to be processed, you can alse select multiple files form different levels at once")
        uploaded_files = st.file_uploader("You can choose multiple files", type=["txt"], accept_multiple_files=True)

        submit_button = st.form_submit_button("Process Files")

    if uploaded_files:
        data_directory = "uploaded"
        os.makedirs(data_directory, exist_ok=True)

        for i, file in enumerate(uploaded_files):
            file_path = os.path.join(data_directory, file.name)

            with open(file_path, "wb") as f:
                f.write(file.read())
        st.success(f"Files uploaded successfully")

        name = uploaded_files[0].name
        
        df = process(data_directory,name)
    
        os.system(f"rm -rf {data_directory}")


if __name__ == "__main__":
    st.set_page_config(layout="wide",initial_sidebar_state="auto")
    st.title("Device Testing")
    main()
