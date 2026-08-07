from ultralytics import YOLO

if __name__ == '__main__':
    # Load model
    model = YOLO("yolo26n.pt")

    # Train
    results = model.train(
        data="F:/ASL dataset/American Sign Language Letters.v1-v1.yolo26/data.yaml", #source
        epochs=50, #times the model will go through the entire dataset
        imgsz=640, #resolution
        batch=8,   #images the model process at once
        device=0,  #gpu / cpu gpu=0 for first gpu, cpu=-1 it basically means ki what is being used for training
        workers=4,     #how many CPU processes are used to load         
        patience=20,  #if doesn’t improve for 20 epochs, training stops automatically
        project="runs/train", #Main folder where all training results will be saved
        name="asl_custom", 
        exist_ok=True
    )
    print("finished training")