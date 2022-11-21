import io, os
import queue
import time

import scipy.signal
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from Google import Create_Service
import pandas as pd
import threading as th
import matplotlib.pyplot as plt
import numpy as np


CLIENT_SECRET_FILE = 'credentials.json' #file storing API secret and OAuth2 data
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']
service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES) #Launch connection to Google Drive
folder_id = '1tDl1kuyby0I4bIWasvJW4ve0e_VIBZhF' #Id of import folder
folder_id2 = '1psz16HbUXYlQ12T9476trbyL33-u6h8l'
keep_going = True   #Variable used to detect keyboard interupt


def key_capture_thread():   #Thread used in parallel to specifically check for keyboard interupt and safely abort program
    global keep_going
    input()
    keep_going = False

def main(): #Main thread used to launch other parallel tasks
    plot_queue = queue.Queue()  #Method to share data safely from Google Drive thread to Plotting Thread
    lock = th.Semaphore()   #Semaphore to prevent thread clashing
    th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()   #Thread used in parallel to specifically check for keyboard interupt and safely abort program
    th.Thread(target=g_drive_daemon, args=(plot_queue, service, folder_id, lock), name='google_drive_daemon').start()   #Thread used in parallel to check for new files in Google Drive
    th.Thread(target=plot_hub, args=(plot_queue,service, folder_id2, lock), name='plot_thread').start() #Thread used in parallel to plot data, convert file types and upload data


def g_drive_daemon(plot_queue, service, folder_id, lock):   #Thread used in parallel to check for new files in Google Drive
    query = f"parents = '{folder_id}' and trashed = false"  #Query to send to Google Drive API
    try:
        with open(os.path.join('./tmp/Archive.txt'), "r") as f: #Crash mitigation file (saved after every drive search, shows already plotted data files)
            lines = f.readlines()
            f.close()
        store_df = pd.DataFrame(columns=["kind", "id", "name", "mimeType"])
        store_df.assign(id=lines)
    except:
        store_df = pd.DataFrame(columns=["kind", "id", "name", "mimeType"]) #For when Archive.txt is empty or not yet created
    while keep_going:   #Will repeat until keyboard interupt, will gracefully exit
        time.sleep(2)   #Time between API requests as to not overload the API
        lock.acquire()  #Semaphore used to only allow one thread to send API requests at once
        response = service.files().list(q=query).execute()  #Send query to API (What files in import folder?)
        files = response.get('files')   #Data from request
        nextPageToken = response.get('nextPageToken')
        while nextPageToken:    #Used if more than a page of files
            response = service.files().list(q=query, pageToken=nextPageToken).execute()
            files.extend(response.get('files'))
            nextPageToken = response.get('nextPageToken')
        lock.release()
        pd.set_option('display.max_columns', 100)
        df = pd.DataFrame(files)    #Stores file metadata in a Pandas Dataframe for analysis (Basically Python Spreadsheet)
        if not df.equals(store_df): #Detects if there are new files
            try:
                for index, row in df.iterrows():
                    if row["id"] not in store_df["id"]:
                        plot_queue.put((str(row["id"]), str(row["name"]))) #2nd level check to check individual files, if files are different, then the file ID and name are sent for processing with the Plot Thread
            except:
                print()

            with open(os.path.join('./tmp/Archive.txt'), "w") as f: #Save already requested plots to prevent duplicate plots at a crash
                for index, row in df.iterrows():
                    f.write(row["id"] + '\n')   #Only ID is needed to detect new files
                f.close()
                time.sleep(10)  #Wait until next set of requests (allows time for Plot Thread)
            store_df = df


def plot_hub(plot_queue, service, folder_id2, lock):    #Thread used in parallel to plot data, convert file types and upload data
    plt.tick_params(bottom=True, top=True, left=True, right=True, direction='in')   #Matplotlib graph styling
    plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = '16'
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    while keep_going:   #Will repeat until keyboard interupt, will gracefully exit
        id, name = plot_queue.get() #Retrieve data from Google Drive Thread
        if name[-4:] == ".tpl": #Prevents wrong files being plotted
            lock.acquire()  #Only one thread can do API requests
            request = service.files().get_media(fileId=id)  #Prepping Google Drive API to download files
            fh = io.BytesIO()   #Prepping local computer to take downloads
            downloader = MediaIoBaseDownload(fh, request)   #Creates download environment
            name = name[:-4]
            done = False
            while not done: #Download the data
                status, done = downloader.next_chunk()
            fh.seek(0)
            lock.release()  #No longer using Google API, allow other threads to use it
            with open(os.path.join('./tmp/local_data.csv'), "wb") as f: #Save the .tpl file in an easier to work with format (locally)
                f.write(fh.read())
                f.close()
            master_data = pd.read_csv('./tmp/local_data.csv')   #Create internal spreadsheet of data
            x = master_data['Energy(keV)'].dropna().reset_index(drop=True)  #Define plotting variables
            #print(len(x))
            x = x[2:]
            y = master_data['Counts'].dropna().reset_index(drop=True)
            times = y[:2]
            y = y[2:]
            ax.plot(x, y, '--', marker='+')
            peaks = scipy.signal.find_peaks_cwt(y, np.arange(300, 700))
            ax.vlines(x[peaks], ymin=min(y)*0.95, ymax=max(y)*1.05)
            #print(x[peaks])
            plt.ylabel(r"Counts", fontsize=20)
            plt.xlabel("Energy/keV", fontsize=20)
            plotname = './tmp/' + name + '.png' #Creating output plot file name
            plt.savefig(plotname, dpi=300)  #Save plot
            ax.clear()
            excelname = './tmp/' + name +'.xlsx'    #Creating output excel data file name
            master_data.to_excel(excelname)     #Save excel
            filenames = [name + ".xlsx", name + ".png"]
            mime_types = ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'image/png'] #Define filetypes for Google Drive API
            lock.acquire()  #Protect Google API against simultaneous usage
            for file, mime in zip(filenames, mime_types):   #Upload data to Google Drive in output folder
                file_metadata = {
                    'name': file,
                    'parents': [folder_id2]
                }
                media = MediaFileUpload('./tmp/{0}'.format(file), mimetype=mime)
                service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
            lock.release()  #Allow other threads to access the API



def gauss(x, a, b, mean, sigma):
    return a + b * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))



main()