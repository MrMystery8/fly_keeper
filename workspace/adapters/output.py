"""Future output decoders belong here; no behavior/controller is supplied yet."""

class OutputDecoder:
    def decode(self, activity):
        raise NotImplementedError("Select and document a readout population for the future environment.")
