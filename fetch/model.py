import torch
import torch.nn as nn

__all__= [
    "TorchVisionModel",
    "PulsarModel",
]

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TorchvisionModel(nn.Module):
    # TODO: Add other torchvision model parameter sizes as needed
    PARAMS = {"DenseNet121": 1024,
             "DenseNet169": 1664,
             "DenseNet201": 1920,
             "VGG16": 512,
             "VGG19": 512,
    }   
    def __init__(self, model_name: str, out_features: int, unfreeze_layers: int = 0) -> None:
        r""" Creates a model based on a pre-trained Torchvision
        model like DenseNet121
        
        Args:
            model_name: The name of the pre-trained model to use
            out_features: Number of output features for classifier 
                          This is the k training hyperparamter
                          referred to in the original FETCH paper
            unfreeze_layers: Number of layers to unfreeze.  
                             Default is ``0``

                             Note: What counts as a layer varies depending on model
        """
        super().__init__()
        
        print(f"Initializing torchvision model {model_name}", flush=True)

        self.model_name = model_name
        weights = f"{model_name}_Weights.DEFAULT"
        self.features = self.PARAMS[model_name]
        self.out_features = out_features

        # Make input data compatible with pre-trained network
        self.block1= nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2, stride=(1, 1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU(),
        )

        # Get the pre-trained model from PyTorch
        self.model = torch.hub.load("pytorch/vision", model_name.lower(), weights=weights)

        # Freeze all layers to start
        for param in self.model.parameters():
            param.requires_grad = False

        if self.model_name.startswith("DenseNet"):
            self._unfreeze_densenet(unfreeze_layers)
        elif self.model_name.startswith("VGG"):
            # Need to replace avgpool layer before classifier
            self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self._unfreeze_vgg(unfreeze_layers)
        
        # Replace/set the classifier layer with single dense layer
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=self.out_features),
            nn.Dropout(p=0.3),
        )

    def _unfreeze_densenet(self, unfreeze_layers: int) -> None:
        r""" Go through each dense layer in each dense block and enable
        gradients until we hit the layer count or run out of layers

        Args:
            unfreeze_layers: Number of layers to unfreeze
        """

        if unfreeze_layers == 0:
            return

        count = 0

        for layer in reversed(list(self.model.features.denseblock4.children())):
            count += 1
            for param in layer.parameters():
                param.requires_grad = True

            if count == unfreeze_layers:
                return

        for layer in reversed(list(self.model.features.denseblock3.children())):
            count += 1
            for param in layer.parameters():
                param.requires_grad = True

            if count == unfreeze_layers:
                return

        for layer in reversed(list(self.model.features.denseblock2.children())):
            count += 1
            for param in layer.parameters():
                param.requires_grad = True

            if count == unfreeze_layers:
                return

        for layer in reversed(list(self.model.features.denseblock1.children())):
            count += 1
            for param in layer.parameters():
                param.requires_grad = True

            if count == unfreeze_layers:
                return
            
    def _unfreeze_vgg(self, unfreeze_layers: int) -> None:
        r""" The layers we need to unfreeze reside in features model component
        Here we unfreeze Conv2d and ReLU in pairs.  Since we are going backward
        through model, ReLU will be encountered first and then Conv2d

        Args:
            unfreeze_layers: Number of layers to unfreeze
        """
        if unfreeze_layers == 0:
            return

        count = 0

        for name, module in reversed(list(self.model.named_modules())):

            if (name.startswith("model.features")):
                if (isinstance(module, nn.Conv2d)):
                    count += 1
                    for param in module.parameters():
                        param.requires_grad = True
                    
                if (isinstance(module, nn.ReLU)):
                    for param in module.parameters():
                        param.requires_grad = True
                if count == unfreeze_layers:
                    return

    # TODO: Add unfreeze functions for other torchvision models
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.block1(data)
        output = self.model(output)

        # Need to manually apply sigmoid if we are not transfer
        # training individual torchvision model
        if self.out_features == 1 and not(self.model.training):
            output = nn.functional.sigmoid(output)

        return output.squeeze()

class PulsarModel(nn.Module):
    def __init__(self, freq_module: nn.Module = None, dm_module: nn.Module = None, k: int = 64) -> None:
        r""" Builds a combined pulsar prediction model using pre-trained freq and dm modules

        Args: 
            freq_module: A pre-trained Torchvision model trained on frequency data
            dm_module: A pre-trained Torchvision model trained on dm data
            k: Number of hyperparameters to include in model
               Default value is 64
            
            Note: This is the k training hyperparamter referred to in the original FETCH paper
        """
        super().__init__()
    
        if freq_module is None:
            self.freq_model = TorchvisionModel("DenseNet201", k)
            self.dm_model = TorchvisionModel("DenseNet201", k)
        else:
            self.freq_model = freq_module
            self.dm_model = dm_module

        # Final process of combined freq and DM data
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=k, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Linear(in_features=k, out_features=1),
        )

    def forward(self, freq_input: torch.Tensor, dm_input: torch.Tensor) -> torch.Tensor:
        # Process freq and dm data separately
        freq_output = self.freq_model(freq_input)
        dm_output = self.dm_model(dm_input)

        # Combine the outputs and produce final classification
        output = torch.mul(freq_output, dm_output)
        output = self.classifier(output)

        # Need to manually apply sigmoid if we are not training
        if not(self.training):
            output = nn.functional.sigmoid(output)

        return output.squeeze()