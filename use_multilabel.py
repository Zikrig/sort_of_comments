from services.multilabel_model_trainer import MultiLabelTrainer

trainer = MultiLabelTrainer("data/multilabel.csv", output_dir="models")
save_path = trainer.train(text_column="text", label_columns=["id_1","id_2","id_3","id_4","id_5","id_6","id_7","id_8","id_9"])
