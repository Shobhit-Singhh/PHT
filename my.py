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
    fig = go.Figure()

    # Add the scatter plot
    fig.add_trace(go.Scatter(x=x_values.flatten(), y=y_values.flatten(), mode="markers", name=f"{y_column} Data"))

    # Add the best fit line with intercept equal to zero
    fig.add_trace(go.Scatter(x=x_values.flatten(), y=line_of_best_fit.flatten(), mode="lines", name=f"Best Fit Line for {y_column} (Intercept=0)"))

    # Update layout
    fig.update_layout(
        title_text=f"Scatter Plot with Best Fit Line (Intercept=0) for {y_column}",
        showlegend=True,
        width=800,
        height=800
    )

    # Show the plot using Streamlit
    st.plotly_chart(fig)

    # Display the equation of the best fit line with intercept equal to zero
    st.write(f"The best fit line for {y_column} with intercept equal to zero is y = {slope:.4f}x")

    return slope


def limit_filter_bar_plot(device_list, avg_cv_values, lower_limit, upper_limit):
    df = pd.DataFrame({"Device": device_list, "Avg_CV%": avg_cv_values})
    df["Within Limits"] = (df["Avg_CV%"] >= lower_limit) & (df["Avg_CV%"] <= upper_limit)
    
    st.write(df)


def L1(path, target_file):
    with st.expander("L1 Test results"):
        all_L1_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L1') != -1]
        device_list = list(set([x.split('_')[0] for x in all_L1_files]))

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

                        data_points_index = re.findall("Press Enter to start...([\s\S]+)Done.", file_content)
                        for j, data_point in enumerate(data_points_index):
                            # Extract numerical values from the data
                            numbers = re.findall(r'\b\d+\b', data_point)

                            # Convert the numbers to integers
                            data_points = [int(num) for i, num in enumerate(numbers) if i % 3 == 0]

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
        L1_summary_df["Avg_CV%"] = [L1_df.loc["CV%"].mean() for device in device_list]
        L1_summary_df['file'] = ['  ||  '.join([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]

        st.dataframe(L1_df)
        st.dataframe(L1_summary_df)

        with st.form("L1 Filter"):
            L1_Avg_CV_limit = st.slider("Select the Avg_CV% limit", 1, 1000, (0,45), 1)
            submit_button = st.form_submit_button("Process Files")

        if submit_button:
            limit_filter_bar_plot(device_list, L1_summary_df["Avg_CV%"].values, L1_Avg_CV_limit[0]/1000, L1_Avg_CV_limit[1]/1000)
            st.markdown("---")


def L3(path, target_file):
    with st.expander("L3 Test results"):
        all_L3_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L3') != -1]
        device_list = list(set([x.split('_')[0] for x in all_L3_files]))
        # st.write(all_L3_files)
        # st.write(device_list)
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

                        data_points_index = re.findall("Press Enter to start...([\s\S]+)Done.", file_content)
                        for j, data_point in enumerate(data_points_index):
                            # Extract numerical values from the data
                            numbers = re.findall(r'\b\d+\b', data_point)

                            # Convert the numbers to integers
                            data_points = [int(num) for i, num in enumerate(numbers) if i % 3 == 0]


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
        L3_summary_df['Device'] = device_list
        L3_summary_df['Number of Files'] = [len([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]
        L3_summary_df["Avg_CV%"] = [L3_df.loc["CV%"].mean() for device in device_list]
        L3_summary_df['file'] = ['  ||  '.join([x for x in all_L3_files if x.find(device) != -1]) for device in device_list]

        st.dataframe(L3_df)
        st.dataframe(L3_summary_df)
        
        with st.form("L3 Filter"):
            L3_Avg_CV_limit = st.slider("Select the Avg_CV% limit", 1, 1000, (0, 140), 1)
            submit_button = st.form_submit_button("Process Files")
            
        if submit_button:
            limit_filter_bar_plot(device_list, L3_summary_df["Avg_CV%"].values, L3_Avg_CV_limit[0]/1000, L3_Avg_CV_limit[1]/1000)
            st.markdown("---")


def L5(path, target_file):
    with st.expander("L5 Test results"):
        all_L6_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L6') != -1]
        device_list = list(set([x.split('_')[0] for x in all_L6_files]))
        # st.write(all_L6_files)
        # st.write(device_list)
        st.subheader("Level 5", help = "CONTINUOUS BLANK TEST WITH INCUBATOR")
        L6_df = pd.DataFrame()
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


                        L6_df.at["Version", file_name.split('.')[0]] = version
                        L6_df.at["Battery", file_name.split('.')[0]] = battery
                        L6_df.at["Wavelength", file_name.split('.')[0]] = wavelength
                        L6_df.at["DAC", file_name.split('.')[0]] = dac
                        L6_df.at["Current(mA)", file_name.split('.')[0]] = current
                        L6_df.at["Adj. Int", file_name.split('.')[0]] = adj_intensity
                        L6_df.at["Core Temp.", file_name.split('.')[0]] = core_Temp

                        data_points_index = re.findall("Press Enter to start...([\s\S]+)Done.", file_content)
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
        L6_summary_df['Device'] = device_list
        L6_summary_df['Number of Files'] = [len([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]
        L6_summary_df["Avg_CV%"] = [L6_df.loc["CV%"].mean() for device in device_list]
        L6_summary_df['file'] = ['  ||  '.join([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]

        st.dataframe(L6_df)
        st.dataframe(L6_summary_df)
        
        with st.form("L5 Filter"):
            L6_Avg_CV_limit = st.slider("Select the Avg_CV% limit", 1, 1000, (0, 50), 1)
            submit_button = st.form_submit_button("Process Files")
        
        if submit_button:
            limit_filter_bar_plot(device_list, L6_summary_df["Avg_CV%"].values, L6_Avg_CV_limit[0]/1000, L6_Avg_CV_limit[1]/1000)
            st.markdown("---")


def L6(path, target_file):
    with st.expander("L6 Test results"):
        all_L6_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L6') != -1]
        device_list = list(set([x.split('_')[0] for x in all_L6_files]))
        # st.write(all_L6_files)
        # st.write(device_list)
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

                        data_points_index = re.findall("Press Enter to start...([\s\S]+)Done.", file_content)
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
        L6_summary_df['Device'] = device_list
        L6_summary_df['Number of Files'] = [len([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]
        L6_summary_df["Avg_CV%"] = [L6_df.loc["CV%"].mean() for device in device_list]
        L6_summary_df['file'] = ['  ||  '.join([x for x in all_L6_files if x.find(device) != -1]) for device in device_list]

        st.dataframe(L6_df)
        st.dataframe(L6_summary_df)
        
        with st.form("L6 Filter"):
            L6_Avg_CV_limit = st.slider("Select the Avg_CV% limit", 1, 1000, (0, 85), 1)
            submit_button = st.form_submit_button("Process Files")
            
        if submit_button:
            limit_filter_bar_plot(device_list, L6_summary_df["Avg_CV%"].values, L6_Avg_CV_limit[0]/1000, L6_Avg_CV_limit[1]/1000)
            st.markdown("---")


def L7(path, target_file):
    with st.expander("L7 Test results"):
        with st.form("form"):
            reference_file = st.file_uploader("L7 files detected, please Upload Reference file", type=["txt"])
            st.form_submit_button("Analyse")

            all_L7_files = [x for x in os.listdir(path) if x.endswith(".txt") and 'L7' in x]
            device_list = list(set([x.split('_')[0] for x in all_L7_files]))

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

                            data_points_index = re.findall("Press Enter to start...([\s\S]+)Done.", file_content)

                            for j, data_point in enumerate(data_points_index):
                                # Extract numerical values from the data
                                numbers = re.findall(r'\b\d+\b', data_point)
                                # Convert the numbers to integers
                                data_points = [int(num) for i, num in enumerate(numbers) if i % 3 == 0]

                                # Calculate statistics
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

            if reference_file:
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
                slopes = []

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
        
        with st.form("L7 Filter"):
            L7_Best_fit_slope_limit = st.slider("Select the Best fit slope limit", 0, 10000, (7905, 8745), 1)
            submit_button = st.form_submit_button("Process Files")
        
        if submit_button:
            limit_filter_bar_plot(device_list, slopes, L7_Best_fit_slope_limit[0]/10000, L7_Best_fit_slope_limit[1]/10000)
            st.markdown("---")


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
        st.markdown("## Upload Files", help="Instructions: Please upload the files to be processed. You are allowed to select multiple files of different levels as well.")
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
