
# coding: utf-8

# In[1]:


import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import Window


# In[102]:


original_df = pd.read_csv("originalDataset.csv")
original_df1 = original_df.where((pd.notnull(original_df)), None)


# In[236]:


original_data = sqlContext.createDataFrame(original_df1).withColumn(
    "win_flag",
    F.when(
        F.col("Winner") == "India",
        F.lit("win")
    ).when(
        (F.col("Winner") == "no result") | (F.col("Winner") == "tied"),
        F.lit("tie")
    ).otherwise(F.lit("lose"))
).withColumn(
    "ground_new",
    F.split("Ground", " \(")[0]
)


# In[104]:


continious_df = pd.read_csv("ContinousDataset.csv")
continious_df1 = continious_df.where((pd.notnull(continious_df)), None)


# In[105]:


continous_data = sqlContext.createDataFrame(continious_df1).withColumn(
    "win_flag",
    F.when(
        F.col("Winner") == "India",
        F.lit("win")
    ).when(
        F.col("Winner") == "no result",
        F.lit("tie")
    ).otherwise(F.lit("lose"))
)


# In[106]:


original_data.persist()
original_data.show()


# In[107]:


continous_data.persist()
continous_data.show()


# # 1. What is India’s total Win/Loss/Tie percentage?

# In[237]:


india_filtered_data = original_data.filter(
    (F.col("Team 1") == "India") |
    (F.col("Team 2") == "India")
)


# In[238]:


match_india_played = india_filtered_data.distinct().count()


# In[239]:


win_loss_percentage = india_filtered_data.groupby(
    "win_flag"
).agg(
    F.count("Team 1").alias("match_count")
).withColumn(
    "percentage",
    (F.col("match_count")*100.0)/F.lit(match_india_played)
)


# In[240]:


win_loss_percentage.show()


# # 2. What is India’s Win/Loss/Tie percentage in away and home matches?

# In[148]:


home_grounds = continous_data.withColumn(
    "home_flag",
    F.when(
        F.col("Host_Country") == "India",
        "home"
    ).otherwise("away")
).select(
    "Ground",
    "home_flag"
).distinct()


# In[331]:


grouped_data = india_filtered_data.join(
    home_grounds,
    on = india_filtered_data.ground_new == home_grounds.Ground,
    how = "left"
).withColumn(
    "home_flag",
    F.when(
        F.col("home_flag").isNull(),
        F.lit("away")
    ).otherwise(F.col("home_flag"))
)


# In[242]:


away_home_match_count = grouped_data.groupby(
    "home_flag"
).agg(
    F.count("Team 1").alias("away_home_count")
)


# In[243]:


home_away_percent = grouped_data.groupby(
    "win_flag",
    "home_flag"
).agg(
    F.count("Team 1").alias("match_count")
).join(
    away_home_match_count,
    on = "home_flag",
    how = "inner"
).withColumn(
    "percentage",
    (F.col("match_count")*100.0)/F.col("away_home_count")
)


# In[244]:


home_away_percent.orderBy("home_flag").show()


# # 3. How many matches has India played against different ICC teams?

# In[268]:


match_count1 = india_filtered_data.filter(
    F.col("Team 1") == "India"
).groupby(
    F.col("Team 2").alias("team")
).agg(F.count(F.col("Scorecard")).alias("match_count"))

match_count2 = india_filtered_data.filter(
    F.col("Team 2") == "India"
).groupby(
    F.col("Team 1").alias("team")
).agg(F.count(F.col("Scorecard")).alias("match_count"))

match_against_icc = match_count1.union(match_count2).groupby(
"team"
).agg(
    F.sum(F.col("match_count")).alias("match_count")
)


# In[269]:


match_against_icc.orderBy("team").show()


# # How many matches India has won or lost against different teams?

# In[325]:


match_won_lose_count1 = india_filtered_data.filter(
    F.col("Team 1") == "India"
).groupby(
    F.col("Team 2").alias("team"),
    "win_flag"
).agg(F.count(F.col("Scorecard")).alias("match_count"))


# In[326]:


match_won_lose_count2 = india_filtered_data.filter(
    F.col("Team 2") == "India"
).groupby(
    F.col("Team 1").alias("team"),
    "win_flag"
).agg(F.count(F.col("Scorecard")).alias("match_count"))


# In[327]:


match_won_lose_count = match_won_lose_count1.union(match_won_lose_count2).groupby(
    "team"
).pivot(
    "win_flag"
).agg(
    F.sum(F.col("match_count")).alias("match_count")
)


# In[328]:


match_won_lose_count.orderBy("team").show(40)


# # 5. Which are the home and away grounds where India has played most number of matches?
# 

# In[ ]:


window = Window.partitionBy("home_flag").orderBy(F.col("match_count").desc())


# In[293]:


home_away_count = grouped_data.groupby(
    F.col("ground_new").alias("ground"),
    "home_flag"
).agg(
    F.count(F.col("Scorecard")).alias("match_count")
).withColumn(
    "rank",
    F.dense_rank().over(window)
).filter(F.col("rank") <=5 )


# In[294]:


home_away_count.show()


# # 6. What has been the average Indian win or loss by Runs per year?

# In[323]:


win_loss_by_run = india_filtered_data.withColumn(
    "run_wicket_count",
    F.split(F.col("Margin"), " ")[0]
).withColumn(
    "run_or_wicket",
    F.split(F.col("Margin"), " ")[1]
).filter(
    F.col("run_or_wicket") == "runs"
).withColumn(
    "year",
    F.trim(F.split(F.col("Match Date"), ",")[1])
).groupby(
    "year"
).pivot("win_flag").agg(
    F.round(F.avg(F.col("run_wicket_count")), 2).alias("avg_run")
).fillna(0)


# In[324]:


win_loss_by_run.orderBy("year").show(50)


# In[322]:


win_loss_by_run.count()


# In[2]:


df1 = pd.read_csv("address_sample.csv")


# In[3]:


df = sqlContext.createDataFrame(df1)


# # practice

# In[ ]:


"""
    SELECT 
        Shippers.ShipperName, COUNT(Orders.ShipperID) AS NumberOfOrders
    FROM 
        Orders 
    LEFT JOIN 
        Shippers 
    ON 
        Orders.ShipperID = Shippers.ShipperID  
    AND 
        Shippers. OrderDate >= Orders.OrderDate 
    AND 
        datediff(from_unixtime(cast(UNIX_TIMESTAMP(Shippers.OrderDate,'yyyy-MM-dd HH:mm:ss')as bigint)),
        from_unixtime(cast(UNIX_TIMESTAMP(Orders.OrderDate,'yyyy-MM-dd HH:mm:ss')as bigint))) BETWEEN 1 and 10 
    GROUP BY 
        ShipperName
"""

aviral
kaushik kuma


# In[ ]:


import pyspark.sql.functions as F


# In[ ]:


Orders.alias('o').join(
    Shippers.alias('s'),
    on = ["ShipperId"],
    how = "left"
).filter(
    (
        F.col("s.OrderDate") >= F.col("o.OrderDate")
    ) &
    (
        F.datediff(
            F.from_unixtime(
                F.unix_timestamp(
                    F.col('s.OrderDate'),
                    'yyyy-MM-dd HH:mm:ss'
                )
            ).cast('date'),
            F.from_unixtime(
                F.unix_timestamp(
                    F.col('o.OrderDate'),
                    'yyyy-MM-dd HH:mm:ss'
                )
            ).cast('date')
        ).between(1,10)
    )

).groupby(
    "ShipperName"
).agg(
    F.count(F.col("o.ShipperID")).alias("NumberOfOrders")
)


# In[ ]:


df.withColumn(
    "new_date",
    F.from_unixtime(
                F.unix_timestamp(
                    F.col('s.OrderDate'),
                    'yyyy-MM-dd HH:mm:ss'
                )
            ).cast('float')
)


# In[2]:


a = 1
list(a)


# In[ ]:


from pyspark.sql import Window
window = Window.partitionBy("id")


# In[ ]:


a=df.groupby(df['id']).agg({"date": "max"}
df = df.join(
    a,
    on = "id",
    how = "inner"
)
df.show()

from pyspark.sql import Window
import pyspark.sql.functions as F
window = Window.partitionBy("id")
a = df.withColumn(
    (F.max(F.col("date")).over(window)).alias("max")
)
a.show()


# In[ ]:


import pyspark.sql.w as 


# In[10]:


df = sqlContext.createDataFrame([('2018-07-01 17:11:35','2018-07-03 17:11:35')], ['a', 'b'] )


# In[13]:


df = df.withColumn(
    "date1",
    F.from_unixtime(
                F.unix_timestamp(
                    F.col('a'),
                    'yyyy-MM-dd HH:mm:ss'
                )
            ).cast('date')
    
).withColumn(
    "date2",
    F.from_unixtime(
                F.unix_timestamp(
                    F.col('b'),
                    'yyyy-MM-dd HH:mm:ss'
                )
            ).cast('date')
    
)


# In[15]:


df.show()


# In[17]:


df.withColumn(
    "date_diff",
    F.datediff(
        F.col("date1"),
        F.col("date2")
    ).between(1,10)
).show()


# In[22]:


def NapierConstant(runs):
    return 2 + 1/contfrac(1, 2, runs)

def contfrac(v1, v2, limit):
    if v1 == limit:
        temp1 = (v1*1.0/v2)
        print v1, v2
        print limit, temp1
        return temp1
    else:
        temp2 = v1+(v1/contfrac(v1+1, v2+1, limit))
        print limit, temp2
        return temp2


# In[24]:


print(NapierConstant(8))


# In[25]:


def get_e(lim):
    return 2 + 1/r(1, lim)

def r(v1, lim):
    if v1 == lim:
        return v1 + v1/(v1+1)
    else:
        return v1 + v1/(r(v1+1, lim))


# In[27]:


get_e(1)


# In[1]:


['a','b','c','d']


# In[2]:


list(zip(['a','b','c','d']))


# In[6]:


a = {'a':1}


# In[ ]:


a.


# In[2]:


some_df = sc.parallelize([
 ("A", "no"),
 ("B", "yes"),
 ("B", "yes"),
 ("B", "no")]
 ).toDF(["user_id", "phone_number"])


# In[3]:


some_df.show()


# In[4]:


pandas_df = some_df.toPandas()


# In[5]:


pandas_df

