![image](https://github.com/user-attachments/assets/9abf8056-7adf-4b71-8949-e6e9bba25c7a)# Minimising the Plot Size of Carbon Capture Plant:  
## Layout Optimisation Considering Cost and Safety Factors

![1739546286079-d87eb73b-d0c6-4a07-bda9-d48c4b21e2d0_1](https://github.com/user-attachments/assets/75223b89-4017-4654-9a47-11a38b19fe71)


## Overview  
This repository contains research, computational models, and analysis for optimizing the land footprint of **carbon capture plants (CCPs)**. The study employs **Mixed-Integer Linear Programming (MILP)** to determine the optimal placement of process units while considering **cost and safety constraints**.

The objective is to **reduce land footprint** while ensuring compliance with safety regulations, focusing on **piping/pumping costs, land purchase, and safety distances**.

## 📌 Objectives  
- 🏭 **Optimize plant layout** to minimize space usage.  
- 🔥 **Ensure safety compliance** using **Fire & Explosion Index (F&EI)** and **Chemical Exposure Index (CEI)**.  
- 💰 **Develop a cost estimation model** based on engineering principles.  
- 📊 **Analyze land footprint trends** for plant capacities ranging from **300MWe to 2200MWe**.  
- 📏 **Compare square vs. rectangular plots** to determine efficiency gains.  

---

## 🛠 Methodology  
### **1️⃣ MILP-Based Layout Optimization**  
- Applied a **2D MILP model** to optimize equipment placement.  
- Considered **land purchase, piping costs, and pumping costs** as constraints.  

### **2️⃣ Safety and Risk Assessment**  
- **Dow Fire and Explosion Index (F&EI)** and **Chemical Exposure Index (CEI)** used.  
- Incorporated **Industrial Risk Insurers' safety distances**.  

### **3️⃣ Cost Estimation Approach**  
- Based on **Towler & Sinnott**'s **cost correlations**.  
- Compared against **IEA's cost estimates**.  

### **4️⃣ Computational Strategy**  
- **Problem decomposition** was used to manage large-scale flowsheets (up to **161 units**).  
- Layouts were tested for **square vs. rectangular plots**.  

---

## 📊 Key Findings  
✔ **Rectangular plots reduce land footprint by 40-47%** compared to square plots.  
✔ **Land footprint is mainly dictated by safety distances**, not process unit sizes.  
✔ **Reduction in footprint is limited** by **safety regulations**.  
✔ **Land requirement per MWe decreases sharply** from **300MWe to 1300MWe**, then stabilizes.  

---
