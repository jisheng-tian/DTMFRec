import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class TemporalWeightingModule(nn.Module):
    """
    Temporal Weighting Module: Assigns increasing weights to each position
    in the sequence to enhance the influence of recent items.
    """

    def __init__(self, max_seq_length, weight_type='exponential'):
        super().__init__()
        self.weight_type = weight_type
        if weight_type == 'linear':
            # Linearly increasing weights
            weights = torch.linspace(0.0, 2.0, max_seq_length)
        elif weight_type == 'exponential':
            # Exponentially increasing weights
            weights = torch.exp(torch.linspace(0, 3, max_seq_length))
        else:
            raise ValueError("Weight type must be either 'linear' or 'exponential'")
        # Register weights as a non-trainable buffer
        self.register_buffer('weights', weights.view(1, -1, 1))

    def forward(self, x):
        seq_length = x.shape[1]
        # Slice the weights to match the input sequence length
        weights_for_sequence = self.weights[:, :seq_length, :]
        return x * weights_for_sequence


class GatedRecurrentCell(nn.Module):  # rglru
    """
    Gated Recurrent Cell: A custom recursive processing unit with a
    dynamic gating mechanism.
    """

    def __init__(self, dimension, expansion_factor=4):
        super().__init__()
        self.dimension = dimension
        self.internal_dim = dimension * expansion_factor
        self.gate_decay_constant = 3  # Hyperparameter for adaptive alpha decay

        self.activation_projection = nn.Linear(dimension, self.internal_dim)
        self.input_projection = nn.Linear(dimension, self.internal_dim)
        self.gate_parameters = nn.Parameter(torch.Tensor(self.internal_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.activation_projection.weight, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_normal_(self.input_projection.weight, mode="fan_in", nonlinearity="linear")
        # Initialize gate parameters to favor high values (close to 1)
        self.gate_parameters.data.uniform_(
            torch.logit(torch.tensor(0.9)),
            torch.logit(torch.tensor(0.999)),
        )

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        hidden_state = torch.zeros(batch_size, self.internal_dim, device=x.device)
        outputs = []

        for t in range(sequence_length):
            current_input = x[:, t, :]

            projected_activation = self.activation_projection(current_input)
            projected_input = self.input_projection(current_input)

            reset_gate = torch.sigmoid(projected_activation)
            input_gate = torch.sigmoid(projected_input)

            # Base update rate
            alpha = torch.sigmoid(self.gate_parameters)
            # Adapt the update rate based on the reset gate
            adaptive_alpha = alpha / (self.gate_decay_constant ** reset_gate)

            # Update hidden state
            hidden_state = adaptive_alpha * hidden_state + ((1 - adaptive_alpha ** 2) ** 0.5) * (
                    input_gate * projected_input)

            outputs.append(hidden_state.unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization Layer.
    """

    def __init__(self, dimension):
        super().__init__()
        self.scale = dimension ** 0.5
        self.gain = nn.Parameter(torch.ones(dimension))

    def forward(self, x):
        assert x.size(-1) == self.gain.size(0), \
            f"RMSNorm dimension mismatch: Input dim {x.size(-1)} != Parameter dim {self.gain.size(0)}"
        return F.normalize(x, dim=-1) * self.scale * self.gain


class ResidualGatedBlock(nn.Module):
    """
    Residual Gated Block: A residual block that combines the GatedRecurrentCell
    and a feed-forward network.
    """

    def __init__(self, dimension, mlp_expansion_factor=4, dropout=0.1):
        super().__init__()
        self.dimension = dimension
        self.norm1 = RMSNorm(dimension)
        self.norm2 = RMSNorm(dimension)
        self.residual_scale = nn.Parameter(torch.ones(1))

        self.gated_recurrent_unit = GatedRecurrentCell(dimension, expansion_factor=2)

        # Projection from the recurrent unit's internal dimension back to the main dimension
        self.recurrent_output_projection = nn.Sequential(
            nn.Linear(dimension * 2, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, dimension)
        )

        # Standard feed-forward network part of the block
        self.feed_forward_network = nn.Sequential(
            nn.Linear(dimension, dimension * mlp_expansion_factor * 2),
            nn.GLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * mlp_expansion_factor, dimension),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # First residual connection
        residual_stream_1 = x
        x_norm1 = self.norm1(x)
        recurrent_out = self.gated_recurrent_unit(x_norm1)
        projected_out = self.recurrent_output_projection(recurrent_out)
        x = projected_out + residual_stream_1

        # Second residual connection
        residual_stream_2 = x
        x_norm2 = self.norm2(x)
        ffn_out = self.feed_forward_network(x_norm2)
        x = ffn_out + residual_stream_2

        return x


class DualPathRecurrentProcessor(nn.Module):
    """
    Dual-Path Recurrent Processor: Processes the original sequence and a
    temporally-weighted version of the sequence in parallel.
    """

    def __init__(self, dimension, max_seq_length):
        super().__init__()
        self.unweighted_path = GatedRecurrentCell(dimension, expansion_factor=2)
        self.weighted_path = GatedRecurrentCell(dimension, expansion_factor=2)
        self.recency_weighting_module = TemporalWeightingModule(max_seq_length, weight_type='exponential')
        self.path_fusion_projection = nn.Linear(dimension * 4, dimension)

    def forward(self, x):
        # Process the original, unweighted sequence
        unweighted_output = self.unweighted_path(x)

        # Create and process the temporally-weighted sequence
        weighted_input = self.recency_weighting_module(x)

        weighted_output = self.weighted_path(weighted_input)

        # Fuse the outputs of the two paths
        combined_features = torch.cat([unweighted_output, weighted_output], dim=-1)
        fused_output = self.path_fusion_projection(combined_features)
        return fused_output


class AdaptiveFusionGate(nn.Module):
    """
    Adaptive Fusion Gate: Dynamically computes fusion weights for multiple
    feature streams based on sequence characteristics.
    """

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

        # Encodes sequence length information
        self.length_encoder = nn.Sequential(
            nn.Linear(1, dimension),
            nn.ReLU()
        )

        # Network to compute the gating weights
        self.gating_network = nn.Sequential(
            nn.Linear(dimension * 4, dimension),
            nn.GELU(),
            nn.Linear(dimension, 3)  # Produces 3 weights for 3 input streams
        )

    def forward(self, feature_stream1, feature_stream2, feature_stream3, sequence_lengths):
        """
        Args:
            feature_stream1: The first feature stream (e.g., standard Mamba output).
            feature_stream2: The second feature stream (e.g., time-weighted Mamba output).
            feature_stream3: The third feature stream (e.g., recurrent network output).
            sequence_lengths: Tensor containing the length of each sequence in the batch.
        """
        batch_size, seq_len, _ = feature_stream1.size()

        # Encode sequence length, normalizing for stability
        length_embedding = self.length_encoder(sequence_lengths.unsqueeze(1).float() / 100.0)
        length_embedding = length_embedding.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate all features to serve as input for the gating network
        gate_input = torch.cat([feature_stream1, feature_stream2, feature_stream3, length_embedding], dim=-1)
        gating_weights = self.gating_network(gate_input)

        # Apply softmax to get normalized weights
        normalized_weights = F.softmax(gating_weights, dim=-1)

        # Unpack weights and apply them to the feature streams
        w1 = normalized_weights[..., 0].unsqueeze(-1)
        w2 = normalized_weights[..., 1].unsqueeze(-1)
        w3 = normalized_weights[..., 2].unsqueeze(-1)

        fused_output = w1 * feature_stream1 + w2 * feature_stream2 + w3 * feature_stream3
        return fused_output


class HybridSequenceBlock(nn.Module):
    """
    Hybrid Sequence Block: The core processing unit that fuses Mamba
    and custom recurrent network streams.
    """

    def __init__(self, d_model, d_state, d_conv, expand, max_seq_length):
        super(HybridSequenceBlock, self).__init__()

        # Mamba state space model layer
        self.mamba_processor = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        # Module to emphasize recent items
        self.recency_enhancer = TemporalWeightingModule(max_seq_length)

        # Custom recurrent network modules
        self.dual_path_recurrent_processor = DualPathRecurrentProcessor(d_model, max_seq_length)
        self.residual_gated_block = ResidualGatedBlock(d_model)

        # Gating mechanism to fuse the parallel streams
        self.adaptive_fusion_gate = AdaptiveFusionGate(d_model)

        # Final projection layer
        self.final_projection = nn.Linear(d_model, d_model)

    def forward(self, input_sequence, sequence_lengths=None):
        # --- Parallel Processing Streams ---

        # Stream 1: Standard Mamba processing
        standard_mamba_output = self.mamba_processor(input_sequence)

        # Stream 2: Mamba processing on a temporally-weighted sequence
        weighted_input = self.recency_enhancer(input_sequence)
        temporal_mamba_output = self.mamba_processor(weighted_input)

        # Stream 3: Combined output from custom recurrent networks
        dual_path_output = self.dual_path_recurrent_processor(input_sequence)
        residual_gated_output = self.residual_gated_block(input_sequence)
        combined_recurrent_output = dual_path_output + residual_gated_output

        # Dynamically fuse the three streams
        fused_features = self.adaptive_fusion_gate(
            standard_mamba_output,
            temporal_mamba_output,
            combined_recurrent_output,
            sequence_lengths
        )

        # Apply final projection
        final_output = self.final_projection(fused_features)
        return final_output


class DTMFRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(DTMFRec, self).__init__(config, dataset)

        # Model hyperparameters
        self.hidden_dim = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_prob"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]

        # Mamba-specific hyperparameters
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        # For CE loss
        self.temperature = config["temperature"]

        # Core model components
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_dim, padding_idx=0
        )
        self.input_layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-12)
        self.input_dropout = nn.Dropout(self.dropout_rate)

        # Stack of hybrid encoder layers
        self.hybrid_encoder_layers = nn.ModuleList([
            DTMFRecLayer(
                d_model=self.hidden_dim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_rate,
                is_first_layer=(i == 0),
                max_seq_length=self.max_seq_length
            ) for i in range(self.num_layers)
        ])

        # Loss function definition
        if self.loss_type == "BPR":
            self.loss_function = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Ensure 'loss_type' is one of ['BPR', 'CE']!")

        # Initialize model weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_sequence, item_sequence_length):
        item_embeddings = self.item_embedding(item_sequence)
        item_embeddings = self.input_dropout(item_embeddings)
        item_embeddings = self.input_layer_norm(item_embeddings)

        sequence_hidden_states = item_embeddings
        for layer in self.hybrid_encoder_layers:
            sequence_hidden_states = layer(sequence_hidden_states, item_sequence_length)

        # Get the representation of the last item in each sequence
        final_representation = self.gather_indexes(sequence_hidden_states, item_sequence_length - 1)
        return final_representation

    def calculate_loss(self, interaction):
        item_sequence = interaction[self.ITEM_SEQ]
        item_sequence_length = interaction[self.ITEM_SEQ_LEN]

        sequence_representation = self.forward(item_sequence, item_sequence_length)
        positive_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            negative_items = interaction[self.NEG_ITEM_ID]
            positive_item_embeddings = self.item_embedding(positive_items)
            negative_item_embeddings = self.item_embedding(negative_items)

            positive_scores = torch.sum(sequence_representation * positive_item_embeddings, dim=-1)
            negative_scores = torch.sum(sequence_representation * negative_item_embeddings, dim=-1)

            loss = self.loss_function(positive_scores, negative_scores)
            return loss
        else:  # Cross-Entropy Loss
            all_item_embeddings = self.item_embedding.weight
            logits = torch.matmul(sequence_representation, all_item_embeddings.transpose(0, 1))
            logits /= self.temperature
            loss = self.loss_function(logits, positive_items)
            return loss

    def predict(self, interaction):
        item_sequence = interaction[self.ITEM_SEQ]
        item_sequence_length = interaction[self.ITEM_SEQ_LEN]
        target_item = interaction[self.ITEM_ID]

        sequence_representation = self.forward(item_sequence, item_sequence_length)
        target_item_embedding = self.item_embedding(target_item)

        scores = torch.mul(sequence_representation, target_item_embedding).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_sequence = interaction[self.ITEM_SEQ]
        item_sequence_length = interaction[self.ITEM_SEQ_LEN]

        sequence_representation = self.forward(item_sequence, item_sequence_length)
        all_item_embeddings = self.item_embedding.weight

        scores = torch.matmul(sequence_representation, all_item_embeddings.transpose(0, 1))
        return scores


class DTMFRecLayer(nn.Module):

    def __init__(self, d_model, d_state, d_conv, expand, dropout, is_first_layer, max_seq_length):
        super().__init__()
        self.is_first_layer = is_first_layer
        self.hybrid_sequence_block = HybridSequenceBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            max_seq_length=max_seq_length
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.positionwise_ffn = PositionwiseFeedForward(
            dimension=d_model,
            inner_dimension=d_model * 4,
            dropout=dropout
        )

    def forward(self, input_tensor, sequence_lengths=None):
        # Main processing block
        block_output = self.hybrid_sequence_block(input_tensor, sequence_lengths)

        # Apply residual connection, but skip for the first layer as a design choice
        if self.is_first_layer:
            hidden_states = self.layer_norm(self.dropout(block_output))
        else:
            hidden_states = self.layer_norm(self.dropout(block_output) + input_tensor)

        # Position-wise FFN with its own residual connection inside
        final_states = self.positionwise_ffn(hidden_states)
        return final_states


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network: A standard FFN with GELU activation
    and a residual connection.
    """

    def __init__(self, dimension, inner_dimension, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dimension, inner_dimension)
        self.fc2 = nn.Linear(inner_dimension, dimension)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dimension, eps=1e-12)

    def forward(self, input_tensor):
        residual = input_tensor
        hidden_states = self.fc1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Add & Norm
        output = self.layer_norm(hidden_states + residual)
        return output
