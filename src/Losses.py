import torch

#Computes the Gram Matrix for style representation
def gram_matrix(tensor):   

    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


#Content loss between target and content image
def compute_content_loss(target_features, content_features):  
    
    return torch.mean((target_features - content_features) ** 2)


#Computes style loss across multiple layers
def compute_style_loss(target_features, style_grams, style_weights): 
    
    style_loss = 0

    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]

        layer_loss = torch.mean((target_gram - style_gram) ** 2)

        _, d, h, w = target_feature.shape
        style_loss += style_weights[layer] * layer_loss / (d * h * w)

    return style_loss


#Combines content and style losses
def compute_total_loss(content_loss, style_loss, content_weight, style_weight): 
    
    return content_weight * content_loss + style_weight * style_loss
