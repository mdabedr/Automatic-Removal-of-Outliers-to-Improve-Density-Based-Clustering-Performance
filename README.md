# Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance
This is a course Project for the University of Alberta course CMPUT 697, Fall 2019. This project aims to improve the clustering performance of HDBSCAN, a well-known hierarchical density-based clustering algorithm by  automatically  removing  outliers.  
We  propose  6  different  methods  that  leverage  well-known algorithms to remove outliers from data automatically. 
Experiments on simulated data demonstrate that one of these variants, consistently performs well in the automatic removal 
of noise, thus improving the performance of HDBSCAN.

**Details of the dataset**
For this task, 6 datasets were generated with ground truth values. Each dataset is 2D with various numbers of clusters, different densities, and distribution of noise. The following figures show a visual representation of the datasets and statistics about the data.

<p align="center">
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/f00510fb-45df-4447-897d-2b7a2f5516c4.png" width=50% height=50%>
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/a84d20dd-e12d-44f4-bec8-cfdeea833d4a.png" width=50% height=50%>
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/41d15437-066c-4415-8d23-8302cf0f3db1.png" width=50% height=50%>
</p>


**Results**
In the results section we look at each dataset individually by looking at the number of clusters discovered, the number of ground truth clusters, the number of mis-clustered points, the number of pruned inliers, etc. We also report two performance evaluation metrics, DBCV and ARI. 

**Dataset 1:**
<p align="center">
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/ced777c5-0269-426c-b930-511a817f5731.png" width=50% height=50%>
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/a8bf8f10-7284-4253-9e84-295e3c57d420.png" width=50% height=50%>
</p>

**Dataset 2:**
<p align="center">
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/91e3cefd-99c7-4747-8493-4fe796249580.png" width=50% height=50%>
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/bb2e0bf1-6903-4cc4-8af8-160fb5c613f0.png" width=50% height=50%>
</p>

**Dataset 3:**
<p align="center">
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/cf964196-fe2d-4583-9bab-038a7cb24845.png" width=50% height=50%>
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/976834c6-3d33-4a3b-a0ab-64ab4e137155.png" width=50% height=50%>
</p>

**Dataset 4:**
<p align="center">
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/4dadbbe7-45d2-4ee5-a90a-e84e4bd19c4e.png" width=50% height=50%>
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/b5b8b9d5-5478-4c79-978d-394c35f37a3a.png" width=50% height=50%>
</p>

**Dataset 5:**
<p align="center">
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/216ac3da-78b5-4ca9-a477-ed05d745b6a6.png" width=50% height=50%>
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/83912909-618f-4557-926b-c0436412f934.png" width=50% height=50%>
</p>

**Dataset 6:**
<p align="center">
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/6c89af12-3e06-4483-98ef-64e1e7170222.png" width=50% height=50%>
<img src="https://github.com/mdabedr/Automatic-Removal-of-Outliers-to-Improve-Density-Based-Clustering-Performance/assets/35268893/1a15677c-ea20-4f8f-8e90-3bc5bef8bde8.png" width=50% height=50%>
</p>




