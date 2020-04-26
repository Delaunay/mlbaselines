from dataclasses import dataclass, asdict

import torchvision.transforms as transforms


class Adversary:
    def get_noise(self, adversarial, original):
        noises = []
        to_pil = transforms.ToPILImage()
        n = (adversarial - original)

        for i in range(n.shape[0]):
            # we use the original image min/max to keep the intensity simlar so people can gauge
            # how small the difference is
            img = n[i] - original[i].min() / original[i].max()
            noises.append(to_pil(img))

        return noises

    def report(self):
        return self.stats.report()


@dataclass
class AdversaryStat:
    """
    Parameters
    ----------

    initial_prediction: int
        Initial prediction of the network before tampering

    initial_confidence: float
        Confidence of the initial prediction

    initial_target_confidence: float
        Initial confidence of our target class

    final_prediction: int
        Network's prediction after tampering, if successful it should be equal to target_class

    final_target_confidence: float
        How confident the network is about its prediction

    final_truth_confidence: float
        Confidence of the initial predicted class after tampering

    """
    initial_prediction: int = 0
    initial_confidence: float = 0
    initial_target_confidence: float = 0
    final_prediction: int = 0
    final_target_confidence: float = 0
    final_truth_confidence: float = 0

    def report(self):
        return asdict(self)

    def update(self, class_predicted, prediction_confidence, target_confidence, prediction):
        if self.initial_prediction == 0:
            self.initial_confidence = prediction_confidence.item()
            self.initial_prediction = class_predicted.item()
            self.initial_target_confidence = target_confidence.item()
        else:
            self.final_target_confidence = target_confidence.item()
            self.final_prediction = class_predicted.item()

        if self.initial_prediction != 0:
            self.final_truth_confidence = prediction[0, self.initial_prediction].item()
