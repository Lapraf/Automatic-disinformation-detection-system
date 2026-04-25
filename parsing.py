import re
import time
import pandas as pd
from selenium import webdriver
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC

def setup_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("--headless=new")

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    options.add_argument(f'user-agent={user_agent}')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(60)
        return driver
    except Exception as e:
        print(f"Помилка ініціалізації WebDriver: {str(e)}")
        return None

def parse_pravda_by_date(driver, days):
    print("\nПарсинг Української Правди")
    news_items = []
    base_url = "https://www.pravda.com.ua/news/date_{}/"
    
    current_date = datetime.now()
    for i in range(days):
        date_str = (current_date - timedelta(days=i)).strftime("%d%m%Y")
        url = base_url.format(date_str)
        
        try:
            driver.get(url)
            time.sleep(5)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "a")))

            articles = driver.find_elements(By.CSS_SELECTOR, "a[href*='/news/']")
            
            for article in articles:
                title = article.text.strip()
                if title and len(title) > 10:
                    news_items.append({
                        "Text": title,
                        "Link": "pravda.com.ua",
                        "labels": int(1),
                    })
            print(f"Дата {date_str}: знайдено {len(articles)} новин")
            
        except Exception as e:
            print(f"Помилка на сторінці {url}: {str(e)}")
            
    return news_items

def parse_stopfake_factcheck(driver, pages):
    print("\nПарсинг StopFake FactCheck")
    news_items = []
    base_url = "https://www.stopfake.org/uk/category/factcheck_facebook_ua/"
    
    for i in range(1, pages + 1):
        url = base_url if i == 1 else f"{base_url}page/{i}/"
        
        try:
            driver.get(url)
            time.sleep(3)
            

            articles = driver.find_elements(By.CSS_SELECTOR, "h3.td-module-title a")
            
            if not articles:
                articles = driver.find_elements(By.CSS_SELECTOR, "article h3.entry-title a")
                
            for article in articles:
                title = article.text.strip()
                title = re.sub(r"^(Фейк:|Маніпуляція:|Діпфейк:|Відеофейк:)\s*[:—–-]*\s*","", title,flags=re.IGNORECASE)
                if title and len(title) > 10:
                    news_items.append({
                        "Text": title,
                        "Link": "stopfake.org",
                        "labels": int(0)
                    })
            print(f"Сторінка {i}: знайдено {len(articles)} новин")
            
        except Exception as e:
            print(f"Помилка на сторінці {url}: {str(e)}")
            
    return news_items

def parse():
    driver = setup_driver()
    if not driver:
        print("Не вдалося запустити WebDriver")
        return 
    try:
        news = []
        pravda_news = parse_pravda_by_date(driver, days=50)
        news.extend(pravda_news)

        stopfake_news = parse_stopfake_factcheck(driver, pages=387)
        news.extend(stopfake_news)
        
        if news:
            df = pd.DataFrame(news)
            df.drop_duplicates(subset="Text", inplace=True)
            df_telegram = pd.read_csv("model_training/raw_news_train/data_set_4.csv", usecols = ["Text", "Label", "Link"])
            df_telegram['Label'] = df_telegram['Label'].astype(int)
            df_telegram = df_telegram.rename(columns = {"Label" : 'labels'})
            df_final = pd.concat([df, df_telegram])
            df_final.to_csv("model_training/raw_news_train/raw_news.csv", index=False, encoding="utf-8-sig", sep = "|")
            print(f"\nУспішно зібрано {len(df)} новин")
        else:
            print("\nНе вдалося зібрати жодної новини")
            
    finally:
        driver.quit()

if __name__ == "__main__":
    parse()

