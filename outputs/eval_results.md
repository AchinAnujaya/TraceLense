# TraceLens Evaluation Report

**Overall Pass Rate:** 14/15 (93.3%)

| ID   | Tier       | Question                                                                                      | Expected                                      |   Steps |   Latency (s) | Status   |
|:-----|:-----------|:----------------------------------------------------------------------------------------------|:----------------------------------------------|--------:|--------------:|:---------|
| Q01  | Simple     | How many total rows are in the dataset?                                                       | 2,500 rows                                    |       2 |          2.94 | PASS     |
| Q02  | Simple     | What is the total revenue across all orders?                                                  | $17,789,870.65                                |       3 |          3.72 | PASS     |
| Q03  | Simple     | How many orders have a deal_status of Refunded?                                               | 505 orders                                    |       3 |          5.32 | PASS     |
| Q04  | Simple     | What is the average unit_price across all rows?                                               | $667.73                                       |       3 |          3.86 | PASS     |
| Q05  | Simple     | Which sales_channel has the highest number of orders?                                         | Self-Serve                                    |       3 |          5.23 | PASS     |
| Q06  | Multi-Step | What is the total Closed Won revenue from the North region?                                   | $4,165,453.75 (including whitespace variants) |       3 |          4.19 | PASS     |
| Q07  | Multi-Step | Which product_category generated the highest Closed Won revenue?                              | Security Suite                                |       3 |          4.4  | PASS     |
| Q08  | Multi-Step | What is the average discount_rate for AI Copilot orders in the Self-Serve channel?            | 0.10 (or 10.1%)                               |       3 |          4.43 | FAIL     |
| Q09  | Multi-Step | How many rows have missing (null) values in the revenue column?                               | 113 rows                                      |       2 |          2.4  | PASS     |
| Q10  | Multi-Step | What is the highest single-order revenue in the Direct sales channel?                         | $35,976.00                                    |       3 |          5.46 | PASS     |
| Q11  | Tricky     | What is the average customer satisfaction score?                                              | Graceful refusal (column missing)             |       2 |          3.14 | PASS     |
| Q12  | Tricky     | What is the total revenue from the Antarctica region?                                         | Graceful handling (no matching rows / 0)      |       3 |          4.44 | PASS     |
| Q13  | Tricky     | What is the net profit margin after employee salaries?                                        | Graceful refusal (cost/salary data missing)   |       2 |          3.24 | PASS     |
| Q14  | Tricky     | Plot a bar chart of total revenue by product_category.                                        | Valid chart PNG generated                     |       5 |          7.11 | PASS     |
| Q15  | Tricky     | Compare average units_sold across sales channels for Closed Won deals and create a bar chart. | Chart generated + channel names cited         |       5 |          8.26 | PASS     |
