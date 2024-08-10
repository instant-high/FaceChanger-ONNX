import argparse
import os
import json
from types import SimpleNamespace
from warnings import warn
from facechanger.constants import INDICES
import copy

class UserInputHandler:
    NO_CLICK=0b00
    LEFT_CLICK=0b01
    RIGHT_CLICK=0b10

    def __init__(self, filter):
        self.filter = filter
        self.features = None
        self.selected = None
        self.start = None
        self.wip = None

    def click(self, event, x, y, flags, params):
        if self.features is not None:
            if flags == self.NO_CLICK:
                # Reset if no buttons is clicked
                if self.selected is not None:
                    self.selected = None
                    self.filter = self.wip
                    self.wip = None
                    self.start = None
            else:
                # Otherwise check what feature was clicked
                if self.selected is None and flags in [self.LEFT_CLICK, self.RIGHT_CLICK]:
                    for k, v in INDICES.items():
                        points = self.features[v]
                        if (points.min(axis=0)<(x,y)).all() and (points.max(axis=0)>(x,y)).all():
                            self.selected = k
                            self.start = (x,y)
                            self.wip = copy.deepcopy(self.filter)

                # Apply change in filter if an object was selected
                if self.selected is not None:
                    if flags == self.LEFT_CLICK:
                        type = "trans"
                        scale = 1
                    elif flags == self.RIGHT_CLICK:
                        type  = "zoom"
                        scale = 0.01
                    else:
                        type  = "zoom"
                        scale = 0
                    self.wip[self.selected][type][0]=self.filter[self.selected][type][0]+(x-self.start[0])*scale
                    self.wip[self.selected][type][1]=self.filter[self.selected][type][1]+(y-self.start[1])*scale

    def get_filter(self):
        if self.wip is not None:
            return self.wip
        else:
            return self.filter

def parse_args():
    parser = argparse.ArgumentParser(description="Face Changer ONNX, no realtime")
    parser.add_argument("--filter", type=str,required=False,help="The path to the filter that should be used. Size 256/512 filters are not compatible.")
    parser.add_argument("--save",type=str,required=False,help="The path (JSON) where the resulting filter of the interactive session should be saved.")
    parser.add_argument("--input",type=str,required=False,help="The path input file for the face change. Uses webcam if none is provided.")
    parser.add_argument("--output",type=str,required=False,help="The path where the result should be saved.")
    
    parser.add_argument("--size", type=int, default=256, help="Internal processing size, 256(faster) or 512(higher quality)")
    
    parser.add_argument("--crop",type=float,default=0.95, required=False,help="Face detection bbox scale")
    
    parser.add_argument("--scale", default=10, help="Rescale factor video, 10 = 1")
    
    parser.add_argument("--audio", default=False, action="store_true", help="Keep audio")
    parser.add_argument("--startpos", type=int, default=0, help="Frame to start from")
    parser.add_argument("--endpos", type=int, default=0, help="Frame to end inference")
             
    args = parser.parse_args()

    processed = SimpleNamespace()
    
    # load filter
    if args.filter is not None:
        if not os.path.exists(args.filter):
            parser.error("The provided filter path does not exist.")
        else:
            with open(args.filter) as f:
                try:
                    processed.filter = json.load(f)
                except json.decoder.JSONDecodeError:
                    parser.error("The provided filter is not a valid JSON file.")            
    else:
        with open(os.path.join("filters","default.json")) as f:
            processed.filter = json.load(f)
    
    # save filter
    if args.save is not None:
        if os.path.exists(args.save):
            warn(f"Filter save path {args.save} does already exist. Overwrite? [y/n]")
            answer = input()
            if answer.lower() not in ["y", "yes"]:
                warn(f"Filter will not be saved.")
                args.save = None
        elif not args.save.lower().endswith(".json"):
            warn("The save path for the filter does not end in .json, the resulting file will be a JSON file nevertheless.")
    processed.save = args.save
    
    # input image/video
    if args.input is not None:
        if not os.path.exists(args.input):
            parser.error("The provided input path does not exist.")
        processed.input = args.input

    # output image/video
    if args.output is not None:
        processed.output = args.output
    else:
        args.output = None
        processed.output = args.output
    
    processed.size = args.size    
    processed.crop = args.crop
    processed.scale = args.scale
    processed.audio = args.audio
    processed.startpos = args.startpos
    processed.endpos = args.endpos
    
    return processed

