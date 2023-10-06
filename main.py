from Sugarsim.read_data import read_dataframe

def main():
  df_location = read_dataframe("CleanGeography_PUBLIC.dta")
  df_hh1 = read_dataframe("GE_HH-Census-BL_PUBLIC.dta")
  df_hh2 = read_dataframe("GE_HH-Survey-BL_PUBLIC.dta")
  df_hh3 = read_dataframe("GE_HH-SampleMaster_PUBLIC.dta")
  print("terminated")

if __name__ == "__main__":
    main()
