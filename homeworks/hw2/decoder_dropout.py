"""Drop whole preceding-token embeddings, leaving supplied labels untouched."""
import math
import torch


def decoder_dropout_inputs(model, labels, probability, *, generator):
    """Return training-only decoder embeddings; zero keeps the original path.

    Use a separate generator so masking does not shift ordinary model-dropout
    random draws. The start token, end token and padding remain unchanged.
    No token IDs, reference labels or source strings are rewritten.
    """
    if not math.isfinite(probability) or not 0 <= probability < 1:
        raise ValueError('decoder embedding dropout must be in [0, 1)')
    if probability == 0 or not model.training:
        return {}
    ids = model.prepare_decoder_input_ids_from_labels(labels)
    eligible = ids.ne(model.config.pad_token_id) & ids.ne(model.config.eos_token_id)
    eligible[:, 0] = False
    embeddings = model.get_input_embeddings()(ids)
    dropped = torch.rand(ids.shape, device=ids.device, generator=generator).lt(probability) & eligible
    # Inverted dropout preserves the expected embedding of eligible tokens.
    scale = torch.where(eligible, (~dropped).to(embeddings.dtype)/(1-probability), 1.)
    return {'decoder_inputs_embeds': embeddings * scale.unsqueeze(-1)}
