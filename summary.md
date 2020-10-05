Feature összefüggőség

# LIME eredmények
*('fxx', y) 
fxx: adott feature, y: 1000 prediction-ből hányszor volt benne a top 10 featureben a LIME szerint.*
| Nem | Rassz | 
|---|---|
| (’f83’, 531), | (’f83’, 982), | 
| (’f10’, 521), | (’f58’, 835), |
| (’f41’, 519), | (’f81’, 795), |
| (’f84’, 517), | (’f48’, 686), | 
| (’f14’, 511), | (’f55’, 632), |
| (’f53’, 504), | (’f37’, 599), |
| (’f43’, 498)  | (’f89’, 580), | 
| (’f42’, 490), | (’f86’, 491), |
| (’f76’, 476), | (’f49’, 486), |
| (’f102’, 467) | (’f50’, 484)  |

# **Módszertan**
Ezen feature-öket úgy módosítjuk, hogy az érzékeny információkra való következtetés ne legyen lehetséges. Ezt megtehetjük a feature törlésével, vagy a feature értékének a módosításával.

*Feature törlése:* Ebben az esetben a meghatározott feature-öket kitöröljük az arclenyomat vektorokból. Mivel a vektor dimenziója megváltozott, az egyes classifier-öket újra kell tanítani. 

*Feature módosítása:* Ebben az esetben nem szükséges újra tanítani a modelleket. 
Lehetőségek: 
1. feature értékét 0-ra állítjuk,
2. random számmal helyettesítjük
3. multiplikatív zaj hozzáadása
4. additív zaj hozzáadása

# **Kapott eredmények**

- Top 10 feature kivétele: A modell pontossága nem változik. 94% accuracy, recall stb. 

- Random feature-ök kivétele, a kivett mennyiség folyamatos növelésével 
![metrics_random_RF_Race_face.png](../_resources/c1f46d7e27af4e4ebfff25976e9310d8.png)


![metrics_random_RF_ID_face.png](../_resources/5ceffc66a67a456e980ea3438da4fb2d.png)



![metrics_random_RF_sex_celebA.png](../_resources/28ff072307cc4597a97cd65603fbda6f.png)

- top feature-ök 0-val helyettesítve:

|   |  accuracy | recall  |  f1  |
|---|---|---|---|
| face recognition |0.9044025157232705|0.9044025157232705 |0.9019326921097836|
| face recognition modified df |0.8641509433962264|0.8641509433962264|0.8731651798163765|
| race prediction |0.8716981132075472 |0.8716981132075472|0.8654302124955726|
| race prediction modified df| 0.839622641509434|0.839622641509434|0.8454371824252951|

- //TODO
random számmal helyettesítve
multiplikatív zaj hozzáadása
additív zaj hozzáadása


# Feature összefüggőség vizsgálata
Ezek az eredmények nem az elvártak szerint alakultak, ezért az egyes feature-ök összefüggőségét kezdtük vizsgálni. 
//TODO -korreláció (Tamás)

Az egyes feature-ök összefüggősége a Random Forest vagy a Decision Tree osztályozó modell döntési fái alapján is igazolhatók. 
![Screenshot 2020-10-05 at 21.00.23.png](../_resources/50688d13e7dc4d91ada297ed049f3109.png)







