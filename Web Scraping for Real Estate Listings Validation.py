#import math
import pandas as pd
import requests
from bs4 import BeautifulSoup
#import numpy as np
from selenium import webdriver
from difflib import SequenceMatcher


#Flow 
#input excel: adds, opps -> Google_parsper() -> GG_results ->
#results -> Property_parser() -> reportMaker(results, GG_results)
#-> output excels: results and report

#Alert for when script finishes
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz


api = pd.read_csv('https://guiker.com/api/reports/inventories?secret=GvB82MPhFMaBgmt5&page=1&perPage=6000')
api.fillna(0)

adds = pd.read_excel(r'C:\Users\Ricardo\Google Drive\McGill\Guiker Internship\adds_2021-02-15.xlsx') 
opps = pd.read_excel(r'C:\Users\Ricardo\Google Drive\McGill\Guiker Internship\opps_2021-02-15.xlsx') 
opps = opps.fillna(0)
opps[["Correct_Bathrooms"]] = opps[["Correct_Bathrooms"]].astype(int)
opps[["Correct_Bedrooms"]] = opps[["Correct_Bedrooms"]].astype(int)



adds['Urls'] = ""

opps = pd.merge(opps, api, how='left', on ='listing_id')

opps = opps.rename(columns={"description": "Correct_Description"})


adds = adds[adds['opportunity_id'].notna()]

#drop all listings in adds that don't contain Montreal or Toronto
adds = adds[adds["Address"].str.contains('Montreal') |  adds["Address"].str.contains('Toronto')]



for i in range(adds.shape[0]):
    adds['Urls'].iloc[i] = (("https://www.google.com/search?q=" + adds['Address'].iloc[i]).replace(" ", "%20")) + "%20Zumper"


# Parses google results and returns zumper urls
def Google_parser(adds_df):

    
    results = {}
    
    #for i in range(10):
    for i in range(adds_df.shape[0]): 
        
        opp_id = (adds_df['opportunity_id'].iloc[i])
        
        Address = (adds_df['Address'].iloc[i])
        listing_id = adds_df['listing_id'].iloc[i]
        print("Google parser:",opp_id)
        
        link = (adds_df['Urls'].iloc[i])
        
        r = requests.get(link)
        html = r.text
        soup = BeautifulSoup(html, "html.parser")

        results[opp_id] = [listing_id, Address, []]
        
        #count = 0
        for a in soup.find_all('a', href=True):
            if (a['href']).startswith('/url?q=https://www.zumper'): 
                url = (a['href'])
                if "montreal" in url:
                    url = (url.split("montreal-qc")[0] + "montreal-qc")[7:] 
                elif "toronto" in url:
                    url = (url.split("toronto-on")[0] + "toronto-on")[7:]                

                else:
                    continue
                
                
                results[opp_id][2].append(url)
                
                #count += 1
        #print(count)
    
    return results



#Iterate through Zumper URLS and parse Zumper 
#to returns characteristics of listings
def Property_parser(results_df, opps):

            
    merged = results_df.merge(opps[['opportunity_id','Correct_Price', 'Correct_Bedrooms', 'Correct_Bathrooms', 'Correct_Description']], how='left', on ='opportunity_id')
    merged[["Zumper_Address"]] = merged[["Zumper_Address"]].astype(str)
    merged[["Correct_Description"]] = merged[["Correct_Description"]].astype(str)
    merged[["Agent"]] = merged[["Agent"]].astype(str)
    merged[["Availability"]] = merged[["Availability"]].astype(str)
    merged[["Photos"]] = merged[["Photos"]].astype(str)

    for i in range(merged.shape[0]):
        #testing purposes
        #link = "https://www.zumper.com/address/4650-ave-bonavista-montreal-qc-h3w-2h5-can"
        
        link = merged['Zumper_URL'][i]
        print("Property_parser :", merged['opportunity_id'][i], link)
        r = requests.get(link)
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        #bebugging purposes
        #print(soup)
        
        
        #When page is a list of properties        
        if ((soup.find(text="Available Listings Nearby")) and (not(soup.find(text="Fully occupied")))): 
                      
            
            try:
            #print(soup)
                driver = webdriver.Firefox(executable_path=r'C:\Users\Ricardo\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\selenium\webdriver\common\geckodriver.exe')
                driver.get(link)
                
                html = driver.page_source
            
            # catch WebDriverException and others
            except:
                print("Error in Property_parser :", merged['opportunity_id'][i], link)
                continue
                
            
            
            soup = BeautifulSoup(html, "html.parser")
            
            rslt = soup.findAll('td', class_='ListingsTableZUM_tD__1aaq2')
            
            
            
            for j in range (0, int(len(rslt)/7)):
            
                
                agent = rslt[4 + (7*j)].text
                
                if agent == "Guiker":
                    x = soup.findAll('a', class_= 'ListingsTableZUM_btn__oq8B1')

                    link = ("https://www.zumper.com" + (x[j]['href']))
                    
                    
                    recdf = pd.DataFrame (columns = ['opportunity_id', 'listing_id', 'Address','Zumper_URL', 'Zumper_Address', 'Agent', 'Availability','Photos', 'Price', 'Bedrooms','Bathrooms', 'description'])
                    #print("1:",recdf)
                    recdf = recdf.append({'opportunity_id':results_df['opportunity_id'][i], 'Address':results_df['Address'][i], 'Zumper_URL':link}, ignore_index=True)
                    #print("2:",recdf)
                    recdf = Property_parser(recdf, opps)
                    #print("Recdf:",recdf)
                    #print("Merged:",merged)
                    merged = merged.append(recdf, ignore_index=True)
                    
                    driver.quit();

        
        #Simple listing page ei not a list of properties  
        elif soup.find(text="Apartment Contact"): 
            
            
            agent = soup.find('span', class_= 'AgentInfo_name__345Q8')
            
            if agent == None: 
                agent = soup.find('a', class_= 'AgentInfo_name__345Q8')
            
            Zadd = soup.find('div', class_='Header_subHeader__3xHsf Header_subHeader__3xHsf')
            
            if Zadd == None:
                try:
                    Zadd =soup.find('a', class_='Breadcrumbs_breadcrumbLink__22PLC MobileWeb_mWebLink__1T5gU').next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling
                    merged.at[i, 'Zumper_Address']= Zadd.text
                except AttributeError:
                        if Zadd == None:
                            try:
                                Zadd =soup.find('a', class_='Breadcrumbs_breadcrumbLink__22PLC MobileWeb_mWebLink__1T5gU').next_sibling.next_sibling.next_sibling.next_sibling
                                merged.at[i, 'Zumper_Address']= Zadd.text
                            except:
                                Zadd="No Zumper address information"
                                merged.at[i, 'Zumper_Address']= Zadd
            else:
                merged.at[i, 'Zumper_Address']= Zadd.text
            
            
            
            available = soup.find('i', class_="PropertyBadge_badge__1WxIy PropertyBadge_darkGray__15wI5")
            
            photos = soup.find('div', class_="NoMedia_text__3rF_e")
            
            price = soup.find('div', class_='Header_price__2pAfA Header_price__2pAfA')
            
            bb= soup.findAll('div', class_='SummaryIcon_summaryText__2Su6m')
            
            
            desc = soup.find('div', class_='Description_description__1gLHl')
            
            if desc == None:
                desc = ""
                
            else:
                merged[["description"]] = merged[["description"]].astype(str)
                merged.at[i, 'description']= desc.text
            
            if len(bb) > 0:
                bedrooms = (bb[0]).text
                
            else:
                bedrooms = None
                
            if len(bb) > 1:
                bathrooms = (bb[1]).text
            else:
                bathrooms = None
                
            if (bedrooms == "Studio") or (bedrooms == None):
                bedrooms = 0
                
            else:
                if (bedrooms[0][0] == 'S') or (bedrooms[0][0] == 'R') :
                    bedrooms = 0;
                else:
                    bedrooms = int(bedrooms[0][0])

            if bathrooms == None:
                bathrooms = 0
            else:                    
                bathrooms = int(bathrooms[0][0]) 

            merged.at[i, 'Bathrooms']= bathrooms
            
            merged.at[i, 'Bedrooms']= bedrooms
                
            
            if price == None:
                merged[["Price"]] = merged[["Price"]].astype(float)
                merged.at[i, 'Price']= 0.0
            
            else: 
                price = price.text
                price = float(price.replace("$", "").replace(",", "")[:4])
                merged[["Price"]].astype(float)
                merged.at[i, 'Price']= price
            
            
            if photos == None:
                
                merged.at[i, 'Photos']= "Has photos"
                
            else:
                merged.at[i, 'Photos']= photos.text
               
            
            merged.at[i, 'Agent']= agent.text
            
            if available != None:
                merged.at[i, 'Availability']= available.text
            
            else:
                merged.at[i, 'Availability']= "Available"
                
       
        else:
            continue

    return merged




def reportMaker(results_df, GG_results):
    
    report = pd.DataFrame (columns = ['opportunity_id', 'listing_id', 'Address','Status', 'Availability', 'Photos', 'Zumper_URL', 'Price', 'Bedrooms', 'Bathrooms','Description_Accuracy', 'Accuracy', 'Correct_Bedrooms', 'Correct_Bathrooms', 'Correct_Price', 'Correct_Description']) #, 'Posted_Correctly'])
    
    
    for key, value in GG_results.items():
        print("reportMaker:", key)
        status = "Not posted";
        address = GG_results[key][1]
        listing_id = GG_results[key][0]
        availability = "Not Guiker"
        photos = "Not Guiker"
        zURL = "Not Guiker"
        price = 0
        bedrooms = None
        bathrooms = None
        accuracy = 0 
        
        desc_ratio = 0.0
             
        
        report = report.append({'opportunity_id': key,'listing_id': listing_id,'Address': address, \
                                'Status':status, 'Availability': availability, \
                                    'Photos':photos, 'Zumper_URL':zURL, \
                                        'Price': price, 'Bedrooms': bedrooms,\
                                            'Bathrooms': bathrooms, \
                                                'Description_Accuracy':desc_ratio,\
                                                'Accuracy': accuracy}, ignore_index=True) # 'Posted_Correctly': "No"}, ignore_index=True)
        
    
        for i in range(results_df.shape[0]):
        

            if key == results_df['opportunity_id'][i]:
                
                report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Correct_Price'] = results_df['Correct_Price'][i]
                report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Correct_Bedrooms'] = results_df['Correct_Bedrooms'][i]
                report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Correct_Bathrooms'] = results_df['Correct_Bathrooms'][i]
                report['Correct_Description'] = report['Correct_Description'].astype(str)
                report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Correct_Description'] = results_df['Correct_Description'][i]
                
                if (results_df['Agent'][i] == 'Guiker'): 

                    #execute str accuracy here                    
                    if (type(results_df['description'][i]) == str) and (type(results_df['Correct_Description'][i]) == str):
                        desc_ratio= SequenceMatcher(None, results_df['description'][i], results_df['Correct_Description'][i]).ratio()
                    
                    
                    if (results_df['Price'][i] == results_df['Correct_Price'][i]) \
                        and (results_df['Bedrooms'][i] == results_df['Correct_Bedrooms'][i]) \
                            and (results_df['Bathrooms'][i] == results_df['Correct_Bathrooms'][i])\
                                and (desc_ratio >= 0.70) and (results_df['Photos'][i] == "Has photos"): 
                        accuracy  = 6

                        zURL = results_df['Zumper_URL'][i]
                        availability = results_df['Availability'][i]
                        photos = results_df['Photos'][i]
                        status = "Posted";
                        price = results_df['Price'][i]
                        bedrooms = results_df['Bedrooms'][i]
                        bathrooms = results_df['Bathrooms'][i] 
                        
                        report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Accuracy'] = accuracy
                        report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Status'] = status
                        report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Availability'] = availability
                        report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Photos'] = photos
                        report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Zumper_URL'] = zURL
                        report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Price'] = price
                        report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bedrooms'] = bedrooms
                        report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bathrooms'] = bathrooms        
                        report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Description_Accuracy'] = desc_ratio   
                        
                    
                
                    elif (results_df['Price'][i] == results_df['Correct_Price'][i]) \
                        and (results_df['Bedrooms'][i] == results_df['Correct_Bedrooms'][i]) \
                            and (results_df['Bathrooms'][i] == results_df['Correct_Bathrooms'][i])\
                                and (results_df['Photos'][i] == "Has photos"): 
                        accuracy  = 5
                        if accuracy > report['Accuracy'][report.index[report['opportunity_id'] == key].tolist()[0]]:

                            zURL = results_df['Zumper_URL'][i]
                            availability = results_df['Availability'][i]
                            photos = results_df['Photos'][i]
                            status = "Posted";
                            price = results_df['Price'][i]
                            bedrooms = results_df['Bedrooms'][i]
                            bathrooms = results_df['Bathrooms'][i] 
                            
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Accuracy'] = accuracy
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Status'] = status
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Availability'] = availability
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Photos'] = photos
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Zumper_URL'] = zURL
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Price'] = price
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bedrooms'] = bedrooms
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bathrooms'] = bathrooms        
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Description_Accuracy'] = desc_ratio
                            
                        
                    elif (results_df['Bedrooms'][i] == results_df['Correct_Bedrooms'][i]) \
                        and  (results_df['Bathrooms'][i] == results_df['Correct_Bathrooms'][i])\
                            and (results_df['Photos'][i] == "Has photos"): 
                        accuracy  = 4
                        
                        if accuracy > report['Accuracy'][report.index[report['opportunity_id'] == key].tolist()[0]]:
                            zURL = results_df['Zumper_URL'][i]
                            availability = results_df['Availability'][i]
                            photos = results_df['Photos'][i]
                            status = "Posted";
                            price = results_df['Price'][i]
                            

                            bedrooms = results_df['Bedrooms'][i]
                            bathrooms = results_df['Bathrooms'][i]         
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Accuracy'] = accuracy
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Status'] = status
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Availability'] = availability
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Photos'] = photos
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Zumper_URL'] = zURL
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Price'] = price
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bedrooms'] = bedrooms
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bathrooms'] = bathrooms     
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Description_Accuracy'] = desc_ratio
                        
                    elif (results_df['Bedrooms'][i] == results_df['Correct_Bedrooms'][i]) and (results_df['Photos'][i] == "Has photos"): 
                        accuracy  = 3
                        
                        if accuracy > report['Accuracy'][report.index[report['opportunity_id'] == key].tolist()[0]]:            
                            zURL = results_df['Zumper_URL'][i]
                            availability = results_df['Availability'][i]
                            photos = results_df['Photos'][i]
                            status = "Posted";
                            price = results_df['Price'][i]
                            bedrooms = results_df['Bedrooms'][i]
                            bathrooms = results_df['Bathrooms'][i]          
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Accuracy'] = accuracy
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Status'] = status
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Availability'] = availability
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Photos'] = photos
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Zumper_URL'] = zURL
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Price'] = price
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bedrooms'] = bedrooms
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bathrooms'] = bathrooms     
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Description_Accuracy'] = desc_ratio
                    
                    elif (results_df['Photos'][i] == "Has photos"):
                        accuracy  = 2
                        if accuracy > report['Accuracy'][report.index[report['opportunity_id'] == key].tolist()[0]]:            
                            zURL = results_df['Zumper_URL'][i]
                            availability = results_df['Availability'][i]
                            photos = results_df['Photos'][i]
                            status = "Posted";
                            price = results_df['Price'][i]
                            bedrooms = results_df['Bedrooms'][i]
                            bathrooms = results_df['Bathrooms'][i]          
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Accuracy'] = accuracy
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Status'] = status
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Availability'] = availability
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Photos'] = photos
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Zumper_URL'] = zURL
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Price'] = price
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bedrooms'] = bedrooms
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bathrooms'] = bathrooms     
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Description_Accuracy'] = desc_ratio
                            
                    else:
                        accuracy  = 1
                        if accuracy > report['Accuracy'][report.index[report['opportunity_id'] == key].tolist()[0]]:            
                            zURL = results_df['Zumper_URL'][i]
                            availability = results_df['Availability'][i]
                            photos = results_df['Photos'][i]
                            status = "Posted";
                            price = results_df['Price'][i]
                            bedrooms = results_df['Bedrooms'][i]
                            bathrooms = results_df['Bathrooms'][i]          
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Accuracy'] = accuracy
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Status'] = status
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Availability'] = availability
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Photos'] = photos
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Zumper_URL'] = zURL
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Price'] = price
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bedrooms'] = bedrooms
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Bathrooms'] = bathrooms     
                            report.at[(report.index[report['opportunity_id'] == key].tolist()[0]), 'Description_Accuracy'] = desc_ratio
                                


    return report



# =============================================================================
# Testing purposes
# opps_list = pd.DataFrame (columns = ['opportunity_id', 'weblink'])
# 
# opps_list = opps_list.append({'opportunity_id': "0066g00001G5l8e", 'weblink':"https://guiker.com/listings/28058"}, ignore_index=True)
# 
# =============================================================================


GG_results = Google_parser(adds)

#Testing purposes
#GG_results = {'0066g00002NbNSV': ['1214-170 rue Rioux, H3C 2A5, Montreal', ['https://www.zumper.com/apartments-for-rent/717590p/1-bedroom-quartier-ville-marie-montreal-qc']]}

results = pd.DataFrame (columns = ['opportunity_id', 'listing_id', 'Address','Zumper_URL', 'Zumper_Address', 'Agent', 'Availability','Photos', 'Price', 'Bedrooms','Bathrooms', 'description'])


for opp_id, value in GG_results.items():
  for j in range(len(value[2])):
        results = results.append({'opportunity_id':opp_id, 'listing_id': GG_results[opp_id][0],'Address':GG_results[opp_id][1], 'Zumper_URL':GG_results[opp_id][2][j]}, ignore_index=True)


#results = results.append({'opportunity_id':'0066g00002NbNSV', 'Address':'1214-170 rue Rioux, H3C 2A5, Montreal ', \
#                         'Zumper_URL':'https://www.zumper.com/apartments-for-rent/717590p/1-bedroom-quartier-ville-marie-montreal-qc'}, ignore_index=True)


results = Property_parser(results, opps)

#Print complete dataframe
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(results)

report = reportMaker(results, GG_results)



results.to_excel(r'C:\Users\Ricardo\Google Drive\McGill\Guiker Internship\results_2021-02-15.xlsx')

report.to_excel(r'C:\Users\Ricardo\Google Drive\McGill\Guiker Internship\report_2021-02-15.xlsx')


    
print("Done")
#Sound for when script is finished running
#winsound.Beep(freq, duration)





