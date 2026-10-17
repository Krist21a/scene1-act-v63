"""Check every decoded frame emitted by ffmpeg framemd5."""
import sys,json,time
from pathlib import Path
count=0; unique=set(); repeated=0; previous=None; first=None; last=None; pts_errors=[]
for line in sys.stdin:
    if line.startswith('#') or not line.strip():continue
    stream,dts,pts,duration,size,hashval=[p.strip() for p in line.split(',')]
    pts=int(pts);duration=int(duration)
    if first is None:first=pts
    if last is not None and pts!=last+1:pts_errors.append([count,last,pts])
    if duration!=1:pts_errors.append(['duration',count,duration])
    last=pts;count+=1;unique.add(hashval);repeated+=hashval==previous;previous=hashval
    if count%15000==0: print('decoded',count,'frames',file=sys.stderr,flush=True)
report={'decoded_frames':count,'expected_frames':84555,'frame_count_matches':count==84555,'time_base':'1/30','first_pts':first,'last_pts':last,'pts_gaps_or_duration_errors':pts_errors,'distinct_decoded_frame_hashes':len(unique),'consecutive_identical_frames':repeated,'all_frames_decoded':count==84555 and not pts_errors}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));assert report['all_frames_decoded']
