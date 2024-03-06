import streamlit as st
import numpy as np
import re
import pandas as pd
import os

def L1(path, target_file):
    with st.expander("L1 Test results"):
        all_L1_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L1') != -1]
        device_list = list(set([x.split('_')[0] for x in all_L1_files]))
        # st.write(all_L1_files)
        # st.write(device_list)
        st.subheader("Level 1", help = "_")
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
                        
        # summery dataframe
        L1_summary_df = pd.DataFrame()
        L1_summary_df['Device'] = device_list
        L1_summary_df['Number of Files'] = [len([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]
        L1_summary_df["Avg_CV%"] = [L1_df.loc["CV%"].mean() for device in device_list]
        L1_summary_df['file'] = ['  ||  '.join([x for x in all_L1_files if x.find(device) != -1]) for device in device_list]
        
        
        st.dataframe(L1_df)
        st.dataframe(L1_summary_df)


def L3(path, target_file):
    with st.expander("L3 Test results"):
        all_L3_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L3') != -1]
        device_list = list(set([x.split('_')[0] for x in all_L3_files]))
        # st.write(all_L3_files)
        # st.write(device_list)
        st.subheader("Level 3", help = "_")
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


def L5(path, target_file):
    with st.expander("L5 Test results"):
        all_L6_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L6') != -1]
        device_list = list(set([x.split('_')[0] for x in all_L6_files]))
        # st.write(all_L6_files)
        # st.write(device_list)
        st.subheader("Level 5", help = "_")
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
                        core_temp_match = re.search(r'Core Temp. : (\d+)', file_content)

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
    

def L6(path, target_file):
    with st.expander("L6 Test results"):
        all_L6_files = [x for x in os.listdir(path) if x.endswith(".txt") and x.find('L6') != -1]
        device_list = list(set([x.split('_')[0] for x in all_L6_files]))
        # st.write(all_L6_files)
        # st.write(device_list)
        st.subheader("Level 6", help = "_")
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
                        core_temp_match = re.search(r'Core Temp. : (\d+)', file_content)

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
    
    
def process(path,name):
    all_files = [x for x in os.listdir(path) if x.endswith(".txt")]
    
    L1_files = [x for x in all_files if x.find('L1') != -1]
    L3_files = [x for x in all_files if x.find('L3') != -1]
    L5_files = [x for x in all_files if x.find('L5') != -1]
    L6_files = [x for x in all_files if x.find('L6') != -1]

    L7_files = [x for x in all_files if x.find('L7') != -1]

    if L1_files:
        l1_name = L1_files[0]
        df = L1(path,l1_name)

    if L3_files:
        l3_name = L3_files[0]
        df = L3(path,l3_name)

    if L5_files:
        l5_name = L5_files[0]
        df = L5(path,l5_name)

    if L6_files:
        l6_name = L6_files[0]
        df = L6(path,l6_name)
    
    if L7_files:
        l7_name = L7_files[0]
        df = L7(path,l7_name)

def main():
    with st.sidebar.form("file_upload_form"):
        st.write("## Upload Files")
        uploaded_files = st.file_uploader("You can choose multiple files", type=["txt"], accept_multiple_files=True)
        submit_button = st.form_submit_button("Process Files")

    if submit_button:
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

