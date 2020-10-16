import pdb
import time
import pandas
import urllib, os
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as BeautifulSoup


# Given CIK of a company, it returns the links to all the 10-K reports
def get_list(cik):

    base_url = "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K&count=1000&output=xml&CIK=" + str(cik)
    
    href = []

    sec_page = urlopen(Request(base_url, headers={'User-Agent': 'Mozilla'}))
    # sec_page = urllib.request.urlopen(base_url)
    sec_soup = BeautifulSoup(sec_page)
    
    filings = sec_soup.findAll('filing')
    
    for filing in filings:
        report_year = int(filing.datefiled.get_text()[0:4])
        if (filing.type.get_text() == "10-K"):
            # print(filing.filinghref.get_text())
            href.append(filing.filinghref.get_text())        
    
    return href



# Given links of 10-K reports it downloads txt file from each of the links 
# and creates the folder for a company(cik) with all its reports
def download_report(url_list, cik):
    years = []
    
    target_base_url = 'http://www.sec.gov'
    
    for report_url in url_list:
        report_page = urlopen(Request(report_url, headers={'User-Agent': 'Mozilla'}))
        # report_page = urllib.request.urlopen(report_url)
        report_soup = BeautifulSoup(report_page)
        
        xbrl_file = report_soup.findAll('tr')
        
        for item in xbrl_file:
            try:
                if 'text file' in item.findAll('td')[1].get_text():
                    # Get year in which it was filed
                    year = item.findAll('td')[2].get_text().split('-')[1]
                    if int(year) >= 0 and int(year) <= 50:
                        year = '20' + year
                    else:
                        year = '19' + year

                    # Get the txt file
                    txt_link = target_base_url + item.findAll('td')[2].find('a')['href']

                    txt_report = urlopen(Request(txt_link, headers={'User-Agent': 'Mozilla'}))
                    # txt_report = urllib.request.urlopen(txt_link)

                    # Make folder for the company if it dosent exist
                    if not os.path.exists('dataset/' + cik):
                        os.makedirs('dataset/' + cik)

                    # Create txt file for the report in its folder
                    output = open('dataset/' + cik + '/' + year + '.txt','wb')
                    output.write(txt_report.read())
                    output.close()

                    years.append(year)

                    # Wait for a while 
                    time.sleep(3)
            except:
                pass


    return years


fwrite = open('index.txt', 'a')
# fwrite.write('CIK|Ticker|Name|Exchange|SIC|Business|Incorporated|IRS|Years_of_files\n')

for line in open("cik_ticker.csv", 'r'):
    row = line.split('|')
    cik = row[0]

    print('Trying for: ' + str(cik))

    url_list= get_list(cik)
    print('Got list for for: ' + str(cik))
    years = download_report(url_list, cik)

    # pdb.set_trace()
    if years:
        years = ','.join(years)
        write_line = line.strip() + '|' + years + '\n'
        fwrite.write(write_line)
        print(cik)