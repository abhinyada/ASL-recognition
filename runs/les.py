import pandas as pd

'''data = {"Employee": ["Snigdha", "Addhyan", "Manan"],
        "Department": ["HR", "IT", "Finance"],
        "Salary": [50190, 71300, 64390]
        }
df = pd.DataFrame(data)
df["Experience"] = [5,7,3]
print(df)'''
data=pd.DataFrame({
    "Employee": ["Raj", "Neha", "Arjun","Sanchi","Aryan"],
    "Working Hours": [None, 8, 7,9, 6],
    "Project Name": ["AI Model",None,"Data Analysis","Web Dev","Cloud"],
    "Attendance":["Present","Present",None,"Present","Absent"]
})
print(data)