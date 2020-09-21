#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Part 1: Tokenization of ingredients and Document-Term Matrix
import pandas as pd
import numpy as np

#load dataset
df = pd.read_csv('cosmetics.csv')


# In[17]:


#initialize list of ingredients and index for each ingredient
ingredient_idx = {}
ingredient_list = []
idx = 0

#filtering overlapping ingredients and labeling with index
for i in range(len(df)): 
    ingredients = df['Ingredients'][i]
    ingredients_lower = ingredients.lower()
    tokens = ingredients_lower.split(', ')
    ingredient_list.append(tokens)
    
    for j in tokens:
        if j not in ingredient_idx:
            ingredient_idx[j] = idx
            idx += 1


# In[18]:


#create document-term matrix or "cosmetic-ingredient" matrix
M = len(df)
N = len(ingredient_idx)

#fill matrix with 0
A = np.zeros((M, N))


# In[19]:


#function to fill A with either 1 or 0 
def encoder(ingredients):
    x = np.zeros(N)
    for ingredient in ingredients:
        # Get the index for each ingredient
        idx = ingredient_idx[ingredient]
        # Put 1 at the corresponding indices
        x[idx] = 1
    return x


# In[20]:


#apply one-hot encoding to document-term matrix 
count = 0

for ingredients in ingredient_list:
    A[count, :] = encoder(ingredients)
    count += 1


# In[21]:


from sklearn.manifold import TSNE

#we currently have 6833 features; want to downsize to 2d
#use T-SNE to reduce dimension while keeping similarities between points
#this takes some time to run
model = TSNE(n_components = 2, learning_rate = 200, random_state = 42)
tsne_features = model.fit_transform(A)

#assign x and y values
df['X'] = tsne_features[:,0]
df['Y'] = tsne_features[:,1]


# In[22]:


#Part 2: Visualization
df_visualize = pd.read_csv('cosmetic_TSNE.csv')


# In[23]:


# cosmetic filtering options 
product_cat = ['Moisturizer', 'Cleanser', 'Treatment', 'Face Mask', 'Eye cream', 'Sun protect']
skin_type = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']


# In[26]:


from bokeh.io import show, curdoc, output_notebook, push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnSource, HoverTool, Select, Paragraph, TextInput
from bokeh.layouts import widgetbox, column, row
from ipywidgets import interact

#make scatter plot  
source = ColumnDataSource(df_visualize)
plot = figure(x_axis_label = 'T-SNE 1', y_axis_label = 'T-SNE 2', 
              width = 500, height = 400)
plot.circle(x = 'X', y = 'Y', source = source, 
            size = 10, color = '#FF7373', alpha = .8)

plot.background_fill_color = "beige"
plot.background_fill_alpha = 0.2


# In[27]:


#add hover tool
hover = HoverTool(tooltips = [('Item', '@Name'),
                              ('Brand', '@Brand'),
                              ('Price', '$@Price'),
                              ('Rank', '@Rank')])
plot.add_tools(hover)


# In[28]:


#define labels to display
def update(op1 = product_cat[0], op2 = skin_type[0]):
    a_b = op1 + '_' + op2
    new_data = {
        'X' : df_visualize[df_visualize['Label'] == a_b]['X'],
        'Y' : df_visualize[df_visualize['Label'] == a_b]['Y'],
        'Name' : df_visualize[df_visualize['Label'] == a_b]['Name'],
        'Brand' : df_visualize[df_visualize['Label'] == a_b]['Brand'],
        'Price' : df_visualize[df_visualize['Label'] == a_b]['Price'],
        'Rank' : df_visualize[df_visualize['Label'] == a_b]['Rank'],
    }
    source.data = new_data
    push_notebook()


# In[31]:


#display the plot
#this plot shows the proximity of products by ingredient compositions
output_notebook()

interact(update, op1 = product_cat, op2 = skin_type)
show(plot, notebook_handle = True)


# In[32]:


#Content-based Filtering
import ipywidgets as widgets
from IPython.display import display

value = input("What's your skin type? Choose from Dry, Combination, Oily, Sensitive, and Normal ")
value_2 = input("Product name? ")

btn = widgets.Button(description='Content-based')
display(btn)


def content_based(obj):
    df_test = df_visualize[df_visualize.Label.str.contains(value)]
    df_2 = df_test.reset_index().drop('index', axis = 1)
    df_2['dist'] = 0.0
    myItem = df_2[df_2.Name.str.contains(value_2)]

    P1 = np.array([myItem['X'].values, myItem['Y'].values])

    #cosine similarities with other items
    for i in range(len(df_2)):
        P2 = np.array([df_2['X'][i], df_2['Y'][i]]).reshape(-1,1)
        df_2.dist[i] = ((P1 * P2).sum())/((np.sqrt(np.sum(P1**2)))*(np.sqrt(np.sum(P2**2))))
    df_2.sort_values('dist', ascending=False, inplace=True)

    
    Moisturizer = df_2[df_2.Label.str.contains ('Moisturizer')]
    Cleanser = df_2[df_2.Label.str.contains ('Cleanser')]
    Treatment = df_2[df_2.Label.str.contains ('Treatment')]
    Face_mask = df_2[df_2.Label.str.contains ('Face Mask')]
    Eye_cream = df_2[df_2.Label.str.contains ('Eye cream')]
    Sun_protect = df_2[df_2.Label.str.contains ('Sun protect')]

    #return rank for moisturizer, cleanser, treatment, face mask, eye cream, sun protect
    Moisturizer_rank = Moisturizer[['Name', 'dist']][1:].head(5)
    Cleanser_rank = Cleanser[['Name', 'dist']][1:].head(5)
    Treatment_rank = Treatment[['Name', 'dist']][1:].head(5)
    Face_mask_rank = Face_mask[['Name', 'dist']][1:].head(5)
    Eye_cream_rank = Eye_cream[['Name', 'dist']][1:].head(5)
    Sun_protect_rank = Sun_protect[['Name', 'dist']][1:].head(5)

    recommendations = [Moisturizer_rank, Cleanser_rank, Treatment_rank, Face_mask_rank, Eye_cream_rank, Sun_protect_rank]

    print("Recommendations across 6 product categories: \n")
    #return recommendations across six categories
    product_count = 0
    for rec in recommendations:
        rec.reset_index(drop=True, inplace=True)
        print(product_cat[product_count], '\n', rec, '\n')
        product_count += 1


btn.on_click(content_based)


# In[33]:


#TF-IDF Filtering
#return product containing ingredients with high IF-IPF val
#these recommendations are not ranked; all three are equally recommended

value_2 = input("What's your skin type? \nChoose from Dry, Combination, Oily, Sensitive, and Normal ") #absolutely need for both types of filtering
df_test_2 = df_visualize[df_visualize.Label.str.contains(value_2)]
df_2 = df_test_2.reset_index().drop('index', axis = 1)
effect = input("What beauty effect do you need? \nChoose from anti aging, moisturizing, oil control, acne treatment, redness control, and reduced pores: ")
if effect == 'anti aging':
    effect = df_test_2[df_test_2.Ingredients.str.contains('Algae' or 'Beta-Glucan' or 'Collagen' or 'Hyaluronic Acid' or 'Vitamin A')]
elif effect == 'moisturizing':
    effect = df_test_2[df_test_2.Ingredients.str.contains('Algae' or 'Aloe Vera' or 'Amino Acid' or 'Beta-Glucan' or 'Cetyl Alcohol' or 'Collagen' or 'Glycerin' or 'Hyaluronic' or 'Olive')]
elif effect == 'oil control':
    effect = df_test_2[df_test_2.Ingredients.str.contains('Witch Hazel' or 'Willow Bark' or 'Vitamin A' or 'Sulfur' or 'Salicylic Acid')]
elif effect == 'acne treatment':
    effect = df_test_2[df_test_2.Ingredients.str.contains('Zinc Oxide' or 'Sulfur')]
elif effect == 'redness control':
    effect = df_test_2[df_test_2.Ingredients.str.contains('Vitamin K' or 'Willow Bark' or 'Niacinamide')]
elif effect == 'reduced pores':
    effect = df_test_2[df_test_2.Ingredients.str.contains('Witch Hazel' or 'Willow Bark' or 'Vitamin A' or 'Niacinamide')]
else:
    print("Please choose from given effects!")
    
effect = effect.reset_index().drop('index', axis = 1)
btn_2 = widgets.Button(description='IF-IPF filtering results')
display(btn_2)
    

def if_ipf(obj):
    x = {}
    N = len(df_2)
    m = len(effect)
    ingredient_list = []

    for i in range(m):
        ingredients = effect['Ingredients'][i]
        ingredients_split = ingredients.split(', ')
        for m in ingredients_split:
            if m not in ingredient_list:
                ingredient_list.append(m)
        n_p = len(ingredients_split)
        rank = pd.Index(ingredients_split)
        for j in ingredients_split:
            alpha = rank.get_loc(j)
            if type(alpha) is np.ndarray:
                alpha = alpha[0] #if it's in multiple locations, just take the first one 
            
            ingredient_frequency = (n_p - alpha)/n_p

            if j not in x:
                x[j] = ingredient_frequency     
            else:
                x[j] += float(ingredient_frequency)


    pf = {}
    IPF = {}
    num_product = len(df_2)

    for l in ingredient_list:
        pf[l] = 0
        IPF[l] = 0
        for k in range(num_product):
            if l in df_2['Ingredients'][k]:
                pf[l] += 1
                IPF[l] = np.log(num_product/pf[l])

    IF_IPF = {m : n * IPF[m] for m, n in x.items() if m in IPF}
    IF_IPF = sorted(IF_IPF, key=IF_IPF.get, reverse=True)
    
    limit = 3

    Moisturizer = df_2[df_2.Label.str.contains ('Moisturizer')]
    Cleanser = df_2[df_2.Label.str.contains ('Cleanser')]
    Treatment = df_2[df_2.Label.str.contains ('Treatment')]
    Face_mask = df_2[df_2.Label.str.contains ('Face Mask')]
    Eye_cream = df_2[df_2.Label.str.contains ('Eye cream')]
    Sun_protect = df_2[df_2.Label.str.contains ('Sun protect')]
    
    products = [Moisturizer, Cleanser, Treatment, Face_mask, Eye_cream, Sun_protect]

    print("Recommendations for Moisturizer, Cleanser, Treatment, Face Mask, Eye Cream, and Sun Protect in order: \n")
    for i in products:
        i.reset_index(drop=True, inplace=True)
        index = 0
        for j in range(len(i)):
            if IF_IPF[0] in i['Ingredients'][j]:
                recommendations = i['Name'][j]
                print(recommendations)
                index += 1
                if index == limit:
                    break
            elif IF_IPF[1] in i['Ingredients'][j]:
                recommendations = i['Name'][j]
                print(recommendations)
                index += 1
                if index == limit:
                    break
            
        print("\n")
    

btn_2.on_click(if_ipf)


# In[ ]:


reviews = pd.read_csv('reviews.csv')


# In[ ]:


#Validation_1 (Content-based)
#Calculate how much the product is recommended by the people who have the same skin type as the participant
#The reviews are already filtered by the participant's skin type
#Recommended percentage = number of recommends tag/number of reviews

#Skin type: Normal

#1. Moisturizer
Moisturizer_val_1 = reviews[reviews.Name.str.contains ('Vitamin C Glow Moisturizer')]
print(Moisturizer_val_1['Recommends'].value_counts()[0]/len(Moisturizer_val_1)*100, "% of people who have Normal skin recommends this Moisturizer")    

#2. Cleanser
Cleanser_val_1 = reviews[reviews.Name.str.contains ('Clean Bee Ultra Gentle Facial Cleanser')]
print(Cleanser_val_1['Recommends'].value_counts()[0]/len(Cleanser_val_1)*100, "% of people who have Normal skin recommends this Cleanser")    

#3. Treatment
Treatment_val_1 = reviews[reviews.Name.str.contains ('Potent-C™ Vitamin C Power Serum')]
print(Treatment_val_1['Recommends'].value_counts()[0]/len(Treatment_val_1)*100, "% of people who have Normal skin recommends this Treatment")    

#4. Face Mask
Face_Mask_val_1 = reviews[reviews.Name.str.contains ('Face Mask - Aloe Vera - Replenishing')]
print(Face_Mask_val_1['Recommends'].value_counts()[0]/len(Face_Mask_val_1)*100, "% of people who have Normal skin recommends this Face Mask")    

#5. Eye Cream
Eye_Cream_val_1 = reviews[reviews.Name.str.contains ('Wide Awake Brightening Eye Cream with Avocado')]
print(Eye_Cream_val_1['Recommends'].value_counts()[0]/len(Eye_Cream_val_1)*100, "% of people who have Normal skin recommends this Eye Cream")    

#6. Sun Protect
Sun_Protect_val_1 = reviews[reviews.Name.str.contains ('Tarteguard Mineral Powder Sunscreen Broad Spec')]
print(Sun_Protect_val_1['Recommends'].value_counts()[0]/len(Sun_Protect_val_1)*100, "% of people who have Normal skin recommends this Sun Protect")    


# In[ ]:


#Validation_2 (IF-IPF)
#Calculate how much the product is recommended by the people who have the same skin type as the participant
#The reviews are already filtered by the participant's skin type
#Recommended percentage = number of recommends tag/number of reviews

#Skin type: Oily

#1. Moisturizer
Moisturizer_val_2 = reviews[reviews.Name.str.contains ('Protini™ Polypeptide Moisturizer')]
print(Moisturizer_val_2['Recommends'].value_counts()[0]/len(Moisturizer_val_2)*100, "% of people who have Oily skin recommends this Moisturizer")    

#2. Cleanser
Cleanser_val_2 = reviews[reviews.Name.str.contains ('T.L.C. Sukari Babyfacial™')]
print(Cleanser_val_2['Recommends'].value_counts()[0]/len(Cleanser_val_2)*100, "% of people who have Oily skin recommends this Cleanser")    

#3. Treatment
Treatment_val_2 = reviews[reviews.Name.str.contains ('C-Firma™ Day Serum')]
print(Treatment_val_2['Recommends'].value_counts()[0]/len(Treatment_val_2)*100, "% of people who have Oily skin recommends this Treatment")    

#4. Face Mask
Face_Mask_val_2 = reviews[reviews.Name.str.contains ('Hydra-Therapy Skin Vitality Treatment Masks')]
print(Face_Mask_val_2['Recommends'].value_counts()[0]/len(Face_Mask_val_2)*100, "% of people who have Oily skin recommends this Face Mask")    

#5. Eye Cream
Eye_Cream_val_2 = reviews[reviews.Name.str.contains ('C-Tango™ Multivitamin Eye Cream')]
print(Eye_Cream_val_2['Recommends'].value_counts()[0]/len(Eye_Cream_val_2)*100, "% of people who have Oily skin recommends this Eye Cream")    

#6. Sun Protect
Sun_Protect_val_2 = reviews[reviews.Name.str.contains ('Umbra™ Sheer Physical Daily Defense Broad Spectrum Sunscreen SPF 30')]
print(Sun_Protect_val_2['Recommends'].value_counts()[0]/len(Sun_Protect_val_2)*100, "% of people who have Oily skin recommends this Sun Protect")    


# In[ ]:


#Validation_3 (IF-IPF)
#This one takes longer to run
#Calculate how much the product is recommended by the people who have the same skin type as the participant
#The reviews are already filtered by the participant's skin type
#Recommended percentage = number of recommends tag/number of reviews

#Skin type: Normal

#1. Moisturizer
Moisturizer_val_3 = reviews[reviews.Name.str.contains ('Crème de la Mer')]
print(Moisturizer_val_3['Recommends'].value_counts()[0]/len(Moisturizer_val_3)*100, "% of people who have Normal skin recommends this Moisturizer")    

#2. Cleanser
Cleanser_val_3 = reviews[reviews.Name.str.contains ('Treatment Cleansing Foam Hydrating Cleanser')]
print(Cleanser_val_3['Recommends'].value_counts()[0]/len(Cleanser_val_3)*100, "% of people who have Normal skin recommends this Cleanser")    

#3. Treatment
Treatment_val_3 = reviews[reviews.Name.str.contains ('The Treatment Lotion')]
print(Treatment_val_3['Recommends'].value_counts()[0]/len(Treatment_val_3)*100, "% of people who have Normal skin recommends this Treatment")    

#4. Face Mask
Face_Mask_val_3 = reviews[reviews.Name.str.contains ('Dermask Water Jet Vital Hydra Solution™')]
print(Face_Mask_val_3['Recommends'].value_counts()[0]/len(Face_Mask_val_3)*100, "% of people who have Normal skin recommends this Face Mask")    

#5. Eye Cream
Eye_Cream_val_3 = reviews[reviews.Name.str.contains ('Advanced Night Repair Eye Supercharged Complex')]
print(Eye_Cream_val_3['Recommends'].value_counts()[0]/len(Eye_Cream_val_3)*100, "% of people who have Normal skin recommends this Eye Cream")    

#6. Sun Protect
Sun_Protect_val_3 = reviews[reviews.Name.str.contains ('Silken Pore Perfecting Sunscreen Broad Spectrum')]
print(Sun_Protect_val_3['Recommends'].value_counts()[0]/len(Sun_Protect_val_3)*100, "% of people who have Normal skin recommends this Sun Protect")    


# In[ ]:


#Validation_4 (Content-based)
#Calculate how much the product is recommended by the people who have the same skin type as the participant
#The reviews are already filtered by the participant's skin type
#Recommended percentage = number of recommends tag/number of reviews

#Skin type: Oily

#1. Moisturizer
Moisturizer_val_4 = reviews[reviews.Name.str.contains ('Ferulic')]
print(Moisturizer_val_4['Recommends'].value_counts()[0]/len(Moisturizer_val_4)*100, "% of people who have Oily skin recommends this Moisturizer")    

#2. Cleanser
Cleanser_val_4 = reviews[reviews.Name.str.contains ('Blackhead Solutions Self-heating Blackhead Ext')]
print(Cleanser_val_4['Recommends'].value_counts()[0]/len(Cleanser_val_4)*100, "% of people who have Oily skin recommends this Cleanser")    

#3. Treatment
Treatment_val_4 = reviews[reviews.Name.str.contains ('Advanced Génifique Youth Activating Serum')]
print(Treatment_val_4['Recommends'].value_counts()[0]/len(Treatment_val_4)*100, "% of people who have Oily skin recommends this Treatment")    

#4. Face Mask
Face_Mask_val_4 = reviews[reviews.Name.str.contains ('Purity Made Simple Pore Extractor Mask')]
print(Face_Mask_val_4['Recommends'].value_counts()[0]/len(Face_Mask_val_4)*100, "% of people who have Oily skin recommends this Face Mask")    

#5. Eye Cream
Eye_Cream_val_4 = reviews[reviews.Name.str.contains ('Pep-Start Eye Cream')]
print(Eye_Cream_val_4['Recommends'].value_counts()[0]/len(Eye_Cream_val_4)*100, "% of people who have Oily skin recommends this Eye Cream")    

#6. Sun Protect
Sun_Protect_val_4 = reviews[reviews.Name.str.contains ('Ultimate Sun Protection Lotion WetForce SPF 50+')]
print(Sun_Protect_val_4['Recommends'].value_counts()[0]/len(Sun_Protect_val_4)*100, "% of people who have Oily skin recommends this Sun Protect")    


# In[ ]:


#Oily CBF average
oily_cbf = ((85.71+52.94+91.10+89.33+76.19+70)/6)

#Oily IF-IPF average
oily_if_ipf = ((80.47+85.65+70.08+100+93.13+52.78)/6)

print('Products recommended with IF-IPF are this much more efficient:', (oily_if_ipf/oily_cbf)-1, '%')


# In[ ]:


#Normal CBF average
normal_cbf = ((86.96+77.14+81.82+91.30+75.53+82.76)/6)

#Normal IF-IPF average
normal_if_ipf = ((72+85.11+100+83.33+81.25+73.91)/6)

print('Products recommended with IF-IPF are this much more efficient', (normal_if_ipf/normal_cbf)-1, '%')


# In[ ]:





# In[ ]:




