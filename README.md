Geo-Spatial Market Intelligence & Customer Feedback Analytics 
An end-to-end Machine Learning project designed to extract actionable business insights from geo-spatial data and unstructured customer reviews, tailored for retail expansion and quality management. 

Project Overview & Business Value 
When expanding a business or optimizing existing operations, companies often struggle with two critical questions: Where is the ideal location to expand? and What do customers in specific micro-markets actually care about? 
This project solves this problem by combining Unsupervised Machine Learning and Natural Language Processing (NLP) to turn raw Yelp data into a data-driven decision tool.  
Target Audience: Franchise owners, retail strategists, and business analysts looking for data-backed location scouting and market intelligence. 

 Technical Approach & Pipeline 
     1. Spatial Hotspot Detection (DBSCAN) 
Instead of relying on arbitrary administrative boundaries (ZIP codes), we apply DBSCAN clustering using the Haversine metric directly on latitude and longitude coordinates. This identifies organic, high density business hotspots while filtering out regional noise. 
     2. Category Stacking and Micro-Market Profiling 
For each identified hotspot, the pipeline performs categorical analysis. By stripping away generic terms (e.g., "Food", "Restaurants"), we reveal the true commercial identity and competitive density of each zone. 
     3. Customer Sentiment & Topic Modeling (LDA) 
To understand consumer pain points and local trends within each specific hotspot, we route text reviews through a Latent Dirichlet Allocation (LDA) model. This extracts the top abstract topics conversed by customers in each cluster. 

Key Features & Visualizations 
Hotspot Visual Mapping: Geographic scatter plots mapping distinct business zones. 
Competitor Density Analysis: Stacked bar charts showing business category distributions per cluster. 
Quality Benchmarking: Boxplots evaluating average customer satisfaction (business_stars) across different zones. 
 
Limitations & Future Work 
Data Sparsity: Topic modeling performance drops in clusters with low review counts (<10 reviews). Future iterations could implement synthetic text augmentation. 
Static Snapshot: The current model evaluates historical data. Integrating a time-series element would allow for predictive trend analysis.

Project Dashboard
![Dashboard](<img width="1536" height="752" alt="Figure_1" src="https://github.com/user-attachments/assets/8fc51e3e-db4f-4fc6-a016-f1613f6bed4d" />
)
