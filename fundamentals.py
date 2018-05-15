import pandas as pd
import numpy as np
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# initiate selenium - chromedriver
option = webdriver.ChromeOptions()
option.add_argument("--disable-infobars")
option.add_argument('--incognito')

# open a browser window
browser = webdriver.Chrome(executable_path='/usr/bin/chromedriver', chrome_options=option)

# navigate to the website page
browser.get('http://www.financialwebsite.com/stocklistpage')

# extract links to all stocks in the website
stocklinks = browser.find_elements_by_xpath('//table[@class="pcq_tbl MT10"]//*//a')
stocklist = [(l.text, l.get_attribute('href')) for l in stocklinks]

# extract fundamental ratios
ratios = []

count = 0
for l in stocklist:
    count += 1
    print(count, l[0])
    
    browser.get(l[1])
    f = browser.find_element_by_link_text('FINANCIALS')
    # clicking buttons when an ad is blocking it or page keeps loading indefinitely
    f.send_keys(u'\ue007')
    f = browser.find_element_by_link_text('Ratios')
    while f != '':
        # clicking buttons when an ad is blocking it or page keeps loading indefinitely
        browser.execute_script('arguments[0].click();', f)
        # locate the table containing the ratios by xpath
        r = browser.find_elements_by_xpath('//table[@class="table4"]')
        tbl = [ele for ele in r if len(ele.text) > 200]
        r_data = []
        if len(tbl) > 0:
            # extract tabular data
            for row in tbl[0].find_elements_by_xpath('.//tr'):
                r_data.append([col.text for col in row.find_elements_by_xpath('.//td')])
            ncol = max([len(d) for d in r_data])
            r_data = [d for d in r_data if len(d)==ncol]
            r_df = pd.DataFrame(r_data)
            r_df.columns = r_df.iloc[0,:].tolist()
            r_df.index = r_df.iloc[:,0].tolist()
            r_df = r_df.iloc[2:,1:]
            r_df = r_df.replace('-', np.nan)
            r_df = r_df.replace('', np.nan)
            r_df.dropna(axis=1, how='all', inplace=True)
            r_df = r_df.apply(lambda x: x.str.replace(',','').astype(float))
            r_df = r_df.T
            r_df.index.name = 'date'
            r_df.reset_index(inplace=True)
            try:
                r_df['symbol'] = re.search('NSE: .+?\|', browser.find_element_by_css_selector('div.PB10').text).group()[5:-1]
            except:
                break
            ratios.append(r_df)
        try:
            f = browser.find_element_by_link_text('Previous Years »')
        except NoSuchElementException:
            f = ''


# put all the extracted data together
ratios_all = pd.concat(ratios, ignore_index=True)

# basic cleaning of the data
ratios_all.dropna(axis=1, how='all', inplace=True)

ratios_all['Book Value [Excl. Reval Reserve]/Share (Rs.)'] = ratios_all[['Book Value [Excl. Reval Reserve]/Share (Rs.)', 'Book Value [ExclRevalReserve]/Share (Rs.)']].min(axis=1)
ratios_all.drop('Book Value [ExclRevalReserve]/Share (Rs.)', axis=1, inplace=True)

ratios_all['Book Value [Incl. Reval Reserve]/Share (Rs.)'] = ratios_all[['Book Value [Incl. Reval Reserve]/Share (Rs.)', 'Book Value [InclRevalReserve]/Share (Rs.)']].min(axis=1)
ratios_all.drop('Book Value [InclRevalReserve]/Share (Rs.)', axis=1, inplace=True)

ratios_all['Diluted EPS (Rs.)'] = ratios_all[['Diluted EPS (Rs.)', 'Diluted Eps (Rs.)']].min(axis=1)
ratios_all.drop('Diluted Eps (Rs.)', axis=1, inplace=True)

ratios_all['Dividend/Share (Rs.)'] = ratios_all[['Dividend / Share(Rs.)', 'Dividend/Share (Rs.)']].min(axis=1)
ratios_all.drop('Dividend / Share(Rs.)', axis=1, inplace=True)

ratios_all['Earnings Yield'] = ratios_all[['Earnings Yield', 'Earnings Yield (X)']].min(axis=1)
ratios_all.drop('Earnings Yield (X)', axis=1, inplace=True)

ratios_all['Enterprise Value (Rs. Cr)'] = ratios_all[['Enterprise Value (Cr.)', 'Enterprise Value (Rs. Cr)']].min(axis=1)
ratios_all.drop('Enterprise Value (Cr.)', axis=1, inplace=True)

ratios_all['Price To Book Value (X)'] = ratios_all[['Price To Book Value (X)', 'Price/BV (X)']].min(axis=1)
ratios_all.drop('Price/BV (X)', axis=1, inplace=True)

ratios_all['Price/Net Operating Revenue'] = ratios_all[['Price To Sales (X)', 'Price/Net Operating Revenue']].min(axis=1)
ratios_all.drop('Price To Sales (X)', axis=1, inplace=True)

ratios_all['Return on Equity / Networth (%)'] = ratios_all[['Return on Equity / Networth (%)', 'Return on Networth / Equity (%)']].min(axis=1)
ratios_all.drop('Return on Networth / Equity (%)', axis=1, inplace=True)


ratios_all = ratios_all[['symbol', 'date'] + [c for c in ratios_all.columns if c not in ['symbol', 'date']]]
ratios_all['date'] = ratios_all.date.apply(lambda x: datetime.strptime(x, '%b %y') + relativedelta(months=1, days=-1))
ratios_all.to_csv('ratios.csv', index=False)
