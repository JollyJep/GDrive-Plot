import io, os
import queue
import time
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from Google import Create_Service
import pandas as pd
import threading as th
import matplotlib.pyplot as plt
import numpy as np
import scipy
from datetime import datetime
import colorsys
import sigfig
import sympy
from sympy import oo
from scipy.signal import lfilter
from numpy import ma
from openpyxl import load_workbook

CLIENT_SECRET_FILE = 'credentials.json'  # file storing API secret and OAuth2 data
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']
service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)  # Launch connection to Google Drive
folder_id = '1tDl1kuyby0I4bIWasvJW4ve0e_VIBZhF'  # Id of import folder
folder_id2 = '1psz16HbUXYlQ12T9476trbyL33-u6h8l'
keep_going = True  # Variable used to detect keyboard interupt


def key_capture_process():  # Thread used in parallel to specifically check for keyboard interupt and safely abort program, not used due to compatability
    global keep_going
    input()
    keep_going = False


def main():  # Main thread used to launch other parallel tasks
    plot_queue = queue.Queue()  # Method to share data safely from Google Drive thread to Plotting Thread
    lock = th.Semaphore()  # Semaphore to prevent thread clashing
    # th.Thread(target=key_capture_process, args=(), name='key_capture_process', daemon=True).start()   #Thread used in parallel to specifically check for keyboard interupt and safely abort program, caused compatability issues
    th.Thread(target=g_drive_daemon, args=(plot_queue, service, folder_id, lock),
              name='google_drive_daemon').start()  # Thread used in parallel to check for new files in Google Drive
    th.Thread(target=plot_hub, args=(plot_queue, service, folder_id2, lock),
              name='plot_thread').start()  # Thread used in parallel to plot data, convert file types and upload data


def g_drive_daemon(plot_queue, service, folder_id,
                   lock):  # Thread used in parallel to check for new files in Google Drive
    query = f"parents = '{folder_id}' and trashed = false"  # Query to send to Google Drive API
    try:
        with open(os.path.join('./tmp/Archive.txt'),
                  "r") as f:  # Crash mitigation file (saved after every drive search, shows already plotted data files)
            lines = f.read().splitlines()
            f.close()
        store_df = pd.DataFrame(np.zeros(shape=(len(lines), 4)), columns=["kind", "id", "name", "mimeType"])
        store_df['id'] = np.resize(lines, len(store_df))
    except:
        store_df = pd.DataFrame(
            columns=["kind", "id", "name", "mimeType"])  # For when Archive.txt is empty or not yet created
    request_time = 0.5
    while keep_going:  # Will repeat until keyboard interupt, will gracefully exit
        try:
            time.sleep(request_time)  # Time between API requests as to not overload the API
            lock.acquire()  # Semaphore used to only allow one thread to send API requests at once
            response = service.files().list(q=query).execute()  # Send query to API (What files in import folder?)
            files = response.get('files')  # Data from request
            nextPageToken = response.get('nextPageToken')
            while nextPageToken:  # Used if more than a page of files
                response = service.files().list(q=query, pageToken=nextPageToken).execute()
                files.extend(response.get('files'))
                nextPageToken = response.get('nextPageToken')
            lock.release()
            pd.set_option('display.max_columns', 100)
            df = pd.DataFrame(
                files)  # Stores file metadata in a Pandas Dataframe for analysis (Basically Python Spreadsheet)
            if not df.equals(store_df):  # Detects if there are new files
                for index, row in df.iterrows():
                    if row.values[1] not in store_df.values:
                        plot_queue.put((str(row.values[1]), str(row.values[
                                                                    2])))  # 2nd level check to check individual files, if files are different, then the file ID and name are sent for processing with the Plot Thread

                with open(os.path.join('./tmp/Archive.txt'),
                          "w") as f:  # Save already requested plots to prevent duplicate plots at a crash
                    for index, row in df.iterrows():
                        f.write(row.values[1] + '\n')  # Only ID is needed to detect new files
                    f.close()
            store_df = df
            request_time = 0.5  # Predict that the API now works again and can take requests at the usual rate
        except:  # System to increase request time if the Google API starts to fail
            if request_time < 5:
                request_time += 0.1


def plot_hub(plot_queue, service, folder_id2,
             lock):  # Thread used in parallel to plot data, convert file types and upload data
    plt.tick_params(bottom=True, top=True, left=True, right=True, direction='in')  # Matplotlib graph styling
    plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = '16'
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    while keep_going:  # Will repeat until keyboard interrupt, will gracefully exit (Now redundant)
        output_intergration = []
        output_efficiency = []
        output_radioiso = []
        output_peak = []
        output_fwhm = []
        id, name = plot_queue.get()  # Retrieve data from Google Drive Thread
        if name[-4:] == ".TKA":  # Prevents wrong files being plotted
            lock.acquire()  # Only one thread can do API requests
            request = service.files().get_media(fileId=id)  # Prepping Google Drive API to download files
            fh = io.BytesIO()  # Prepping local computer to take downloads
            downloader = MediaIoBaseDownload(fh, request)  # Creates download environment
            name = name[:-4]
            done = False
            while not done:  # Download the data
                status, done = downloader.next_chunk()
            fh.seek(0)
            lock.release()  # No longer using Google API, allow other threads to use it
            with open(os.path.join('./tmp/local_data.csv'),
                      "wb") as f:  # Save the .TKA file in an easier to work with format (locally)
                f.write(fh.read())
                f.close()
            master_data = pd.read_csv('./tmp/local_data.csv')  # Create internal spreadsheet of data
            y = master_data.dropna().reset_index(drop=True)
            y = y.values.ravel()
            times = y[:2]
            n = 15  # the larger n is, the smoother curve will be
            b = [1.0 / n] * n
            a = 1
            y = y[2:]
            x = []
            for value, _ in enumerate(y):
                x.append(value + 1)
            x = np.array(x)
            x_store = x
            x = mca_to_kev(x)
            ax.plot(x, y, '--', marker='+')
            peaks,_ = scipy.signal.find_peaks(y, height=5)
            colour = (1, 0, 0)
            col = np.array([0, 65536, 65536], dtype=np.uint16)
            change = np.array([65536 / len(peaks), 0, 0], dtype=np.uint8)
            for peak in peaks:
                repeat = True
                fail = False
                index = 0
                p0 = (1, x_store[peak], 8)
                old_cv = False
                old_params = False
                if peak == 0 or peak == len(y):
                    repeat = False
                    fail = True
                else:
                    if y[peak] >= y[peak-1] and y[peak] >= y[peak +1] and y[peak] > y[peak-2] and y[peak] > y[peak + 2]:
                        higher = 3
                        lower = 2
                    else:
                        repeat = False
                        fail = True
                while repeat:
                    if len(y[peak - index - 3: peak-index + 1]) < 3 or len(y[peak + index: peak + index + 4]) < 3 or len(y[peak - index - 6: peak-index -2]) < 3 or len(y[peak + index + 3: peak + index + 7]) < 3:
                        repeat = False
                        break
                    mean_l = np.mean(y[peak - index - 3: peak-index + 1])
                    mean_r = np.mean(y[peak + index: peak + index + 4])
                    if mean_l > np.mean(y[peak - index - 6: peak-index -2]):
                        higher +=3
                    if mean_r > np.mean(y[peak + index + 3: peak + index + 7]):
                        lower += 3
                    if mean_l <= np.mean(y[peak - index - 6: peak-index -2]) or mean_r <= np.mean(y[peak + index + 3: peak + index + 7]):
                        repeat = False
                    x_gauss = x_store[peak - lower:peak + higher]
                    y_gauss = y[peak - lower:peak + higher]
                    if len(y_gauss) > 6:
                        try:
                            params, cv = scipy.optimize.curve_fit(gauss, x_gauss, y_gauss, p0, maxfev=100000)
                            a, mean, sigma = params
                            squaredDiffs = np.square(y_gauss - gauss(x_gauss, a, mean, sigma))
                            squaredDiffsFromMean = np.square(y_gauss - np.mean(y_gauss))
                            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
                            if rSquared > 0.9:
                                old_x = x_gauss
                                old_y = y_gauss
                                old_params = params
                                old_cv = cv
                            else:
                                repeat = False
                                break
                        except:
                            repeat = False
                            print(old_params, old_cv)
                            fail = True
                            break
                if not fail and not isinstance(old_cv, bool) and not isinstance(old_params, bool):
                    x_con = mca_to_kev(old_params[1])
                    x_conu = x_con * np.sqrt(np.diag(old_cv))[1] / old_params[1]
                    x_coni = mca_to_kev(old_x)
                    x_consig = mca_to_kev(old_params[2])
                    x_consigu = x_consig * np.sqrt(np.diag(old_cv))[2] / old_params[2]
                    params_con = [old_params[0], x_con, x_consig]
                    errors_con = [np.sqrt(np.diag(old_cv))[0], x_conu, x_consigu]
                    ax.vlines(x_con, ymin=min(y) * 0.95, ymax=max(y) * 1.05, color=colour, label="(" + str(sigfig.round(x_con,  x_conu)) + ") KeV")
                    N_peak = gauss_integrate(x_coni, params_con, np.mean([old_y[0], old_y[-1]]), errors_con)
                    output_intergration.append(N_peak)
                    efficiency = efficiency_curve(x_con)
                    output_peak.append([x_con, x_conu])
                    output_efficiency.append(efficiency)
                    output_fwhm.append([x_consig * 2 * np.sqrt(2 * np.log(2)), x_consigu * 2 * np.sqrt(2 * np.log(2))])
                    radioiso = isotope_detector(x_con, x_conu)
                    output_radioiso.append(radioiso)
                    #plt.text(old_params[1], max(y) * 0.9,
                             #"" + str(sigfig.round(old_params[1], np.sqrt(np.diag(cv))[1])) + "", color=colour,
                             #rotation=90)
                col += change
                colour = colorsys.hsv_to_rgb(float(col[0]) / 65536, 1, 1)


            output = [name]
            for value, item in enumerate(output_peak):
                output.append("Peak Energy:")
                output.append("(" + str(sigfig.round(output_peak[value][0],  output_peak[value][1])) + ") KeV")
                output.append("FWHM:")
                output.append("(" + str(sigfig.round(output_fwhm[value][0],  output_fwhm[value][1])) + ") KeV")
                output.append("Area of photopeak:")
                output.append(str(sigfig.round(output_intergration[value][0],  output_intergration[value][1])))
                output.append("Efficiency")
                output.append(str(output_efficiency[value]))
                output.append("Potential radioisotopes:")
                for isotope in output_radioiso[value]:
                    output.append(isotope)
                output.append("-----------------------------------------------------------------------------------------\n")
            with open(r'./tmp/' + name + "_plotting_variables.txt" , 'w') as fp:
                for item in output:
                    # write each item on a new line
                    fp.write("%s\n" % item)

            plt.ylabel(r"Counts", fontsize=20)
            plt.xlabel("Energy/KeV", fontsize=20)
            plotname = './tmp/' + name + '.png'  # Creating output plot file name
            plt.legend()
            ax.set_yscale("log")
            #plt.show()
            #exit()
            plt.savefig(plotname, dpi=300)  # Save plot
            ax.clear()
            excelname = './tmp/' + name + '.xlsx'  # Creating output excel data file name
            master_data.to_excel(excelname)  # Save excel
            filenames = [name + ".xlsx", name + ".png", name + "_plotting_variables.txt"]
            mime_types = ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                          'image/png', 'text/plain']  # Define filetypes for Google Drive API
            #lock.acquire()  # Protect Google API against simultaneous usage
            #for file, mime in zip(filenames, mime_types):  # Upload data to Google Drive in output folder
            #    file_metadata = {
            #        'name': file,
            #        'parents': [folder_id2]
            #    }
            #    media = MediaFileUpload('./tmp/{0}'.format(file), mimetype=mime)
            #    service.files().create(
            #        body=file_metadata,
            #        media_body=media,
            #        fields='id'
            #    ).execute()
            now = datetime.now()
            finish_time = now.strftime("%A, %d, %m, %Y\n%H:%M:%S")
            print(name + " complete" + "\n" + finish_time + "\n")
            #lock.release()  # Allow other threads to access the API


def gauss(x, a, mean, sigma):
    return a * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def mca_to_kev(x):
    return 0.276 * x + 0.5 # Replace with found values


def efficiency_curve(x):
    return (2 * 8.66 / 100) / ((x / 103) ** 1.1 + (103 / x) ** 3.6)


def gauss_integrate(x, params, height, errors):
    local_x = sympy.symbols('x')
    a, mean, sigma = params
    local_gauss = a * sympy.exp(-(local_x - mean) ** 2 / (2 * sigma ** 2))
    area = float(sympy.integrate(local_gauss, (local_x, min(x), max(x)))) #- height * (max(x) - min(x))
    return (area, area * np.sqrt((errors[0] / a) ** 2 + (errors[1] / mean) ** 2 + (errors[2] / sigma) ** 2))


#def radioactivity(N_peak, times, energy, uncertainty):
#    efficiency = efficiency_curve(energy)
#    #isotope_detector(energy, uncertainty)
#    return


def isotope_detector(energy, uncertainty):
    master_data = pd.ExcelFile('./Data/Nuclide_data_store.xlsx')
    global sheets
    old_energy_levels = []
    old_intensities = []
    output = []
    nearest = True
    for sheet in sheets:
        df = pd.read_excel(master_data, sheet)
        try:
            max(df['Energy (keV)']) < energy
            min(df['Energy (keV)']) > energy
        except:
            continue
        if len(df['Energy (keV)']) >= 2 and max(df['Energy (keV)']) > energy > min(df['Energy (keV)']):
            energy_levels = find_neighbours(energy, df, colname='Energy (keV)')
            intensities = [df['Intensity (%)'][energy_levels[0]], df['Intensity (%)'][energy_levels[1]]]
        elif len(df['Energy (keV)']) == 1:
            energy_levels = [0, 0]
            intensities = [df['Intensity (%)'][energy_levels[0]], 0]
        elif max(df['Energy (keV)']) < energy and len(df['Energy (keV)']) > 0:
            energy_levels = [df['Energy (keV)'].idxmax(), df['Energy (keV)'].idxmax()]
            intensities = [df['Intensity (%)'][energy_levels[0]], 0]
        elif min(df['Energy (keV)']) > energy and len(df['Energy (keV)']) > 0:
            energy_levels = [df['Energy (keV)'].idxmin(), df['Energy (keV)'].idxmin()]
            intensities = [0, df['Intensity (%)'][energy_levels[1]]]
        else:
            continue
        if nearest == True and len(old_energy_levels) > 1:
            if abs(energy - df['Energy (keV)'][energy_levels[0]]) < abs(energy - dfo['Energy (keV)'][old_energy_levels[0]]) and abs(energy - df['Energy (keV)'][energy_levels[0]]) < abs(energy - dfo['Energy (keV)'][old_energy_levels[1]]) or abs(energy - df['Energy (keV)'][energy_levels[0]]) < abs(energy - dfo['Energy (keV)'][old_energy_levels[0]]) and abs(energy - df['Energy (keV)'][energy_levels[0]]) < abs(energy - dfo['Energy (keV)'][old_energy_levels[1]]):
                old_energy_levels = energy_levels + [sheet]
                old_intensities = intensities
                dfo = df
        elif nearest == True:
            old_energy_levels = energy_levels + [sheet]
            old_intensities = intensities
            dfo = df
        if np.isclose(df['Energy (keV)'][energy_levels[0]], energy, rtol=0.01) or np.isclose(df['Energy (keV)'][energy_levels[1]], energy, rtol=0.01):
            nearest = False
            if len(old_energy_levels) == 0:
                old_energy_levels = energy_levels + [sheet]
                output.append(sheet.strip('.l'))
                old_intensities = intensities
                dfo = df
            else:
                if max(intensities) > max(old_intensities):
                    old_energy_levels = energy_levels + [sheet]
                    output.append(sheet.strip('.l'))
                    old_intensities = intensities
                    dfo = df
    return output










def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind]


def get_sheetnames_xlsx():
    wb = load_workbook('./Data/Nuclide_data_store.xlsx', read_only=True, keep_links=False)
    return wb.sheetnames

sheets = get_sheetnames_xlsx()
main()