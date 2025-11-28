class TaskParser:
    """
    Turns language commands into navigation goals.
    """

    def parse(self, text):
        text = text.lower()

        if "left" in text:
            return {"task": "cover_area", "region": "left"}

        if "right" in text:
            return {"task": "cover_area", "region": "right"}

        if "uncertainty" in text:
            return {"task": "reduce_uncertainty"}

        if "replan" in text:
            return {"task": "trigger_replan"}

        return {"task": "unknown"}
