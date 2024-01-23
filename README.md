# Stock_market_prediction
Project to predict the stock market.

# Challanges
final_df = past_100_days.append(data_testing, ignore_index = True)

In the new version of Pandas, the append method is changed to _append. You can simply use '_append' instead of append, i.e., df._append(df2).

change complete:
final_df = past_100_days._append(data_testing, ignore_index = True)

