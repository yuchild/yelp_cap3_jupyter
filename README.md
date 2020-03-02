# Yelp Recommender Systems for Scottsdale Arizona

### Project Status: [Active]

## Project Intro/Objective
The purpose of this project is to create recommenders to help users find establishments they like in Scottsdale Arizona. There are two types of recommenders employed to service this goal: collaborative filtering and content based recommenders. The collaborative filters separates pivots users and business on average user ratings. The content based recommender uses review text and categories of businesses on their vectorized cosine similarities.

### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling
* Model Validation

### Technologies
* Python and Libraries (pandas, numpy, sklearn, tensorflow, keras, surprise, wordcloud, matplotlib)
* Docker (for Tensorflow GPU)
* flask (api, not operational yet)
* Postman (api, not operational yet)

## Project Description
The project is based on the open yelp dataset found [here](yelp.com/dataset). The recommenders will mainly come from three sources: user.json, review.json, and business.json, which will be merged to create a super table with an inner join based on the feature business_id.

### EDA



(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modeling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)


## To Run This Project for Yourself
1. Clone this [repo](https://github.com/yuchild/yelp_cap3_jupyter.git).
2. Create a folder within the repo named *data* and download dataset from yelp [here](https://www.yelp.com/dataset/download).
    *This is because Github does not accept files larger than 2.3GB*
    *Yelp will ask for your information prior to download*
3. Unzip the yelp_dataset.tar inside the data folder you created
4. Delete the yelp_dataset.tar file, as this is 3.9GB
5. Make sure your workstation has 64GB of ram, or use AWS services with enough 64GB+ ram capacity.


## Contact
* You can find me on, [you can do that here](gstudents.slack.com).
