
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


# -----------------------------------------
# WEBSITE
# -----------------------------------------

BASE_URL = "https://books.toscrape.com/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# -----------------------------------------
# 3 CATEGORIES
# -----------------------------------------

CATEGORIES = {
    "Travel": "catalogue/category/books/travel_2/index.html",
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "http://books.toscrape.com/catalogue/category/books/fiction_10/index.html"
}


BOOKS_PER_CATEGORY = 20


# -----------------------------------------
# FUNCTION TO GET SOUP
# -----------------------------------------

def get_soup(url):

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=10
    )

    response.raise_for_status()

    return BeautifulSoup(
        response.text,
        "html.parser"
    )


# -----------------------------------------
# SCRAPE CATEGORY
# -----------------------------------------

def scrape_category(category_name, category_path):

    category_url = urljoin(
        BASE_URL,
        category_path
    )

    books = []
    current_url = category_url

    while current_url and len(books) < BOOKS_PER_CATEGORY:

        print(f"Scraping {category_name}...")

        soup = get_soup(current_url)

        products = soup.select(
            "article.product_pod"
        )

        for product in products:

            if len(books) >= BOOKS_PER_CATEGORY:
                break

            # ---------------------------------
            # TITLE
            # ---------------------------------

            title_tag = product.select_one(
                "h3 a"
            )

            if title_tag:
                title = title_tag.get(
                    "title",
                    ""
                ).strip()
            else:
                title = ""


            # ---------------------------------
            # PRICE
            # ---------------------------------

            price_tag = product.select_one(
                ".price_color"
            )

            if price_tag:
                price = price_tag.get_text(
                    strip=True
                )
            else:
                price = ""


            # ---------------------------------
            # STAR RATING
            # ---------------------------------

            rating_tag = product.select_one(
                ".star-rating"
            )

            if rating_tag:

                classes = rating_tag.get(
                    "class",
                    []
                )

                star_rating = next(
                    (
                        item
                        for item in classes
                        if item != "star-rating"
                    ),
                    ""
                )

            else:
                star_rating = ""


            # ---------------------------------
            # AVAILABILITY
            # ---------------------------------

            availability_tag = product.select_one(
                ".availability"
            )

            if availability_tag:

                availability = availability_tag.get_text(
                    " ",
                    strip=True
                )

            else:
                availability = ""


            # ---------------------------------
            # ADD BOOK
            # ---------------------------------

            books.append({
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category_name
            })


        # ---------------------------------
        # NEXT PAGE
        # ---------------------------------

        next_button = soup.select_one(
            "li.next a"
        )

        if next_button:

            next_page = next_button.get(
                "href"
            )

            current_url = urljoin(
                current_url,
                next_page
            )

        else:

            current_url = None


    return books


# -----------------------------------------
# SCRAPE ALL CATEGORIES
# -----------------------------------------

all_books = []

for category, path in CATEGORIES.items():

    category_books = scrape_category(
        category,
        path
    )

    all_books.extend(
        category_books
    )

    print(
        f"{category}: {len(category_books)} books"
    )


# -----------------------------------------
# CREATE DATAFRAME
# -----------------------------------------

df = pd.DataFrame(all_books)


# =========================================================
# DATA CLEANING
# =========================================================

print("\nCleaning data...")


# -----------------------------------------
# 1. CLEAN PRICE
# -----------------------------------------

# Example:
# "£45.17" -> 45.17

df["price_gbp"] = (
    df["price"]
    .astype(str)
    .str.replace("£", "", regex=False)
    .str.strip()
)

# Invalid values become NaN
df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)


# -----------------------------------------
# 2. CLEAN STAR RATING
# -----------------------------------------

rating_mapping = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(
    rating_mapping
)


# -----------------------------------------
# 3. MEDIAN IMPUTATION
# -----------------------------------------

# Invalid price values are replaced
# with the median price.

price_median = df["price_gbp"].median()

df["price_gbp"] = df["price_gbp"].fillna(
    price_median
)


# Invalid ratings are replaced
# with the median rating.

rating_median = df["rating"].median()

df["rating"] = df["rating"].fillna(
    rating_median
)


# Convert rating to integer
df["rating"] = (
    df["rating"]
    .round()
    .astype(int)
)


# -----------------------------------------
# 4. PARSE AVAILABILITY
# -----------------------------------------

def parse_availability(value):

    value = str(value).strip().lower()

    if "in stock" in value:
        return True

    elif "out of stock" in value:
        return False

    else:
        return None


df["in_stock"] = df["availability"].apply(
    parse_availability
)


# -----------------------------------------
# 5. DROP INVALID AVAILABILITY ROWS
# -----------------------------------------

before = len(df)

df = df.dropna(
    subset=["in_stock"]
)

after = len(df)

print(
    f"Dropped {before - after} rows "
    "with invalid availability."
)


# Convert to boolean
df["in_stock"] = df["in_stock"].astype(bool)


# -----------------------------------------
# RESET INDEX
# -----------------------------------------

df = df.reset_index(drop=True)


# =========================================================
# FINAL DATASET
# =========================================================

print("\nFINAL DATASET")
print("=" * 80)

print(df.head(10))


# -----------------------------------------
# DATA TYPES
# -----------------------------------------

print("\nDATA TYPES")
print("=" * 80)

print(df.dtypes)


# -----------------------------------------
# DATASET SHAPE
# -----------------------------------------

print("\nDATASET SHAPE")
print("=" * 80)

print(df.shape)


# -----------------------------------------
# BOOK COUNT BY CATEGORY
# -----------------------------------------

print("\nBOOKS PER CATEGORY")
print("=" * 80)

print(
    df["category"].value_counts()
)


# -----------------------------------------
# SAVE CLEAN CSV
# -----------------------------------------

df.to_csv(
    "books_cleaned.csv",
    index=False,
    encoding="utf-8"
)


print("\nCSV file created successfully:")
print("books_cleaned.csv")

df.head(5)


# 3. Convert price_gbp to a price_inr

# CONVERT GBP TO INR

# Project-defined fixed conversion rate
# 1 GBP = 105.50 INR

GBP_TO_INR = 105.50

df["price_inr"] = (
    df["price_gbp"] * GBP_TO_INR
)

# Round to 2 decimal places
df["price_inr"] = df["price_inr"].round(2)


# DISPLAY PRICE CONVERSION

print("\nPRICE CONVERSION")
print("=" * 50)

print(
    df[
        ["price_gbp", "price_inr"]
    ].head(10)
)



# CHECK COLUMNS

print("\nALL COLUMNS")
print("=" * 60)
print(df.columns.tolist())

# Columns to keep for final cleaned dataset
keep_columns = [
    "title",
    "category",
    "price_gbp",
    "price_inr",
    "rating",
    "in_stock"
]

# Columns that will be dropped
drop_columns = [
    column for column in df.columns
    if column not in keep_columns
]

print("\nCOLUMNS TO DROP:")
print(drop_columns)

# Drop unnecessary/raw columns
df = df.drop(columns=drop_columns)

print("\nFINAL COLUMNS:")
print(df.columns.tolist())

print("\nFINAL DATASET:")
print(df.head())



# CHECK NUMERICAL COLUMNS

numerical_columns = [
    "price_gbp",
    "price_inr",
    "rating"
]

print("\nNUMERICAL COLUMNS")
print("=" * 60)

# Check data types
print("\nData types:")
print(df[numerical_columns].dtypes)

# Check missing values
print("\nMissing values:")
print(df[numerical_columns].isnull().sum())

# Statistical summary
print("\nNumerical summary:")
print(df[numerical_columns].describe())

# Check whether columns are actually numeric
print("\nIs numeric:")
for column in numerical_columns:
    print(
        f"{column}: "
        f"{pd.api.types.is_numeric_dtype(df[column])}"
    )

    
# CHECK COLUMNS

print("\nALL COLUMNS")
print("=" * 60)
print(df.columns.tolist())

# Columns to keep for final cleaned dataset
keep_columns = [
    "title",
    "category",
    "price_gbp",
    "price_inr",
    "rating",
    "in_stock"
]

# Columns that will be dropped
drop_columns = [
    column for column in df.columns
    if column not in keep_columns
]

print("\nCOLUMNS TO DROP:")
print(drop_columns)

# Drop unnecessary/raw columns
df = df.drop(columns=drop_columns)

print("\nFINAL COLUMNS:")
print(df.columns.tolist())

print("\nFINAL DATASET:")
print(df.head())


# 4: sqlite 3
import sqlite3
import pandas as pd

# create database connection
conn = sqlite3.connect('books_database.db')
cursor = conn.cursor()

# enable forgiven key support
cursor.execute("PRAGMA foreign_keys = ON;")

# 1. create tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
  );
  ''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS books(
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
''')
conn.commit()
print("Tables created successfully")



# =========================================================
# INSERT CATEGORIES
# =========================================================

for category in df["category"].dropna().unique():

    cursor.execute(
        """
        INSERT OR IGNORE INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )


# =========================================================
# INSERT BOOKS
# =========================================================

for _, row in df.iterrows():

    cursor.execute(
        """
        SELECT category_id
        FROM categories
        WHERE category_name = ?
        """,
        (row["category"],)
    )

    category_id = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            int(row["in_stock"]),
            category_id
        )
    )

conn.commit()

print("Cleaned data inserted successfully!")


# =========================================================
# SQL QUERY 1
# SELECT + WHERE
# =========================================================

query1 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
"""

result1 = pd.read_sql_query(query1, conn)

print("\nQUERY 1: SELECT + WHERE")
print(query1)
print(result1)


# =========================================================
# SQL QUERY 2
# ORDER BY + LIMIT
# =========================================================

query2 = """
SELECT title, price_gbp, rating
FROM books
ORDER BY rating DESC, price_gbp DESC
LIMIT 10
"""

result2 = pd.read_sql_query(query2, conn)

print("\nQUERY 2: ORDER BY + LIMIT")
print(query2)
print(result2)


# =========================================================
# SQL QUERY 3
# DISTINCT
# =========================================================

query3 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name
"""

result3 = pd.read_sql_query(query3, conn)

print("\nQUERY 3: DISTINCT")
print(query3)
print(result3)


# =========================================================
# SQL QUERY 4
# IN
# =========================================================

query4 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC
"""

result4 = pd.read_sql_query(query4, conn)

print("\nQUERY 4: IN")
print(query4)
print(result4)


# =========================================================
# SQL QUERY 5
# BETWEEN
# =========================================================

query5 = """
SELECT title, price_gbp, price_inr
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp
"""

result5 = pd.read_sql_query(query5, conn)

print("\nQUERY 5: BETWEEN")
print(query5)
print(result5)


# =========================================================
# SQL QUERY 6
# JOIN
# =========================================================

query6 = """
SELECT
    c.category_name,
    b.title,
    b.rating,
    b.price_gbp,
    b.price_inr
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
ORDER BY c.category_name, b.rating DESC
LIMIT 10
"""

result6 = pd.read_sql_query(query6, conn)

print("\nQUERY 6: JOIN")
print(query6)
print(result6)


# =========================================================
# SAVE QUERY OUTPUTS
# =========================================================

result1.to_csv("query1_output.csv", index=False)
result2.to_csv("query2_output.csv", index=False)
result3.to_csv("query3_output.csv", index=False)
result4.to_csv("query4_output.csv", index=False)
result5.to_csv("query5_output.csv", index=False)
result6.to_csv("query6_output.csv", index=False)

print("\nAll query outputs saved successfully.")


# =========================================================
# CLOSE DATABASE
# =========================================================

conn.close()



# =========================================================
# CONNECT TO DATABASE
# =========================================================

conn = sqlite3.connect("books_database.db")


# =========================================================
# 1. READ TWO SQL QUERY RESULTS INTO PANDAS
#    USING pd.read_sql()
# =========================================================

query1 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
"""

query2 = """
SELECT title, price_gbp, rating
FROM books
ORDER BY rating DESC, price_gbp DESC
LIMIT 10
"""

df_query1 = pd.read_sql(query1, conn)
df_query2 = pd.read_sql(query2, conn)

print("\nQUERY 1 RESULT")
print(df_query1)

print("\nQUERY 2 RESULT")
print(df_query2)


# =========================================================
# 2. READ THE TWO DATABASE TABLES INTO PANDAS
# =========================================================

books_df = pd.read_sql(
    """
    SELECT
        book_id,
        title,
        price_gbp,
        price_inr,
        rating,
        in_stock,
        category_id
    FROM books
    """,
    conn
)

categories_df = pd.read_sql(
    """
    SELECT
        category_id,
        category_name
    FROM categories
    """,
    conn
)


# =========================================================
# 3. REPRODUCE JOIN USING pd.merge()
#    NO SQL JOIN USED HERE
# =========================================================

pandas_join = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

# Select the same columns as the SQL JOIN
pandas_join = pandas_join[
    [
        "category_name",
        "title",
        "rating",
        "price_gbp",
        "price_inr"
    ]
]

# Same ordering as SQL JOIN
pandas_join = pandas_join.sort_values(
    by=["category_name", "rating"],
    ascending=[True, False]
)

# Same LIMIT 10 as the SQL query
pandas_join = pandas_join.head(10).reset_index(drop=True)


print("\nPANDAS JOIN RESULT")
print(pandas_join)


# =========================================================
# 4. RUN THE ORIGINAL SQL JOIN AGAIN
# =========================================================

join_query = """
SELECT
    c.category_name,
    b.title,
    b.rating,
    b.price_gbp,
    b.price_inr
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
ORDER BY c.category_name, b.rating DESC
LIMIT 10
"""

sql_join = pd.read_sql(join_query, conn)

sql_join = sql_join.reset_index(drop=True)


# =========================================================
# 5. COMPARE SQL JOIN AND PANDAS JOIN
# =========================================================

# Make sure column order is identical
pandas_join = pandas_join[sql_join.columns]

equivalent = sql_join.equals(pandas_join)

print("\nJOIN COMPARISON")
print("=" * 60)

print("SQL JOIN:")
print(sql_join)

print("\nPANDAS pd.merge() JOIN:")
print(pandas_join)

print("\nAre both results equivalent?")
print(equivalent)


# =========================================================
# 6. OPTIONAL: CHECK DIFFERENCES
# =========================================================

if equivalent:
    print("\nSUCCESS: SQL JOIN and pandas.merge() produce equivalent output.")
else:
    print("\nResults are different. Checking differences...")

    print("\nSQL JOIN shape:")
    print(sql_join.shape)

    print("\nPandas JOIN shape:")
    print(pandas_join.shape)


# =========================================================
# CLOSE DATABASE
# =========================================================

conn.close()
