import paddlex as pdx
predictor = pdx.deploy.Predictor('/home/aistudio/output/inference_model/inference_model')
img_path = '/home/aistudio/work/97ce96584315ba13a4927c8a0.jpg'
result = predictor.predict(img_file=img_path)
print(result)