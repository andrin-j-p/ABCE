from Sugarsim.read_data import read_dataframe

def main():
  df_location = read_dataframe("CleanGeography_PUBLIC.dta")
  print(df_location.head())

if __name__ == "__main__":
    main()
