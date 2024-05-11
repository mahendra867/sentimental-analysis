from flask import Flask, render_template, request
from NLP_PROJECT.pipeline.prediction import PredictionPipeline,CustomData
from NLP_PROJECT import logger

app = Flask(__name__)

@app.route('/',methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) 
def predict_datapoint():
    if request.method=="GET":
        return render_template("index.html")  
    else:
        data = CustomData(
                           
                            review=request.form.get("review")
                            
        )

        dataframe=data.get_data_as_dataframe()
        logger.info('initiated prediction')
        predict_pipeline=PredictionPipeline()
        data = dataframe['review'].tolist()
        prediction=predict_pipeline.predict(data)
        logger.info(f'made prediction the value is : {prediction}')
        final_result = str(round(float(prediction[0]), 1))
        logger.info(f'converted the prediction value to str is : {final_result}')
        logger.info('made prediction and returning to results.html')
        return render_template("results.html", final_result=final_result)

logger.info('done with prediction')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9095)
