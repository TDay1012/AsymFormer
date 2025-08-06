import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class InteractionMatrixGenerator:
    def __init__(self, model_path, input_size, hidden_size, output_size):
        self.model = SimpleMLP(input_size, hidden_size, output_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_interactions(self, input_data):
        """
        Predicts the interaction matrix for a batch of sequences.

        Parameters:
        - input_data: Tensor of shape (B, T, N, J, D) where B is the batch size,
          T is the number of time steps, N is the number of individuals, J is the number of joints,
          and D is the dimensionality of each joint's features.

        Returns:
        - interaction_matrix: Tensor of shape (B, N, N) representing the interaction between individuals.
        """
        # input_data = input_data[:,1::2]
        B, T, N, J, D = input_data.shape

        interaction_matrix = torch.eye(N).repeat(B, 1, 1)


        # Flatten the input to match the expected input shape of the InteractionDetector

        # Iterate over all pairs of individuals
        for i in range(N):
            for j in range(i+1, N):
                # Extract the features for the pair of individuals
                if i != j:
                    pair_features = torch.cat((input_data[:, :, i], input_data[:, :, j]), dim=2)
                    pair_features = pair_features.reshape(B, -1)
                    # Predict the interaction for this pair
                    interaction_score = self.model(pair_features)
                    interaction_score = (interaction_score > 0.5).float()     # 改为权重
                    # Store the interaction score in the matrix
                    interaction_matrix[:, i, j] = interaction_score.squeeze(-1)
                    interaction_matrix[:, j, i] = interaction_score.squeeze(-1)  # Make the matrix symmetric

        return interaction_matrix

    def to(self, device):
        # 将内部模型移动到指定设备，并返回自身（方便链式调用）
        self.model = self.model.to(device)
        return self

