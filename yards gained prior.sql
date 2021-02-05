drop table yardsgainedprior



select 
	dt0.posteam, 
	dt0.gameid, 
	dt0.TimeSecs,
	case when dt1.[Yards Gained] is null then -999 else dt1.[Yards Gained] end as YardsGainedOnPriorPlay
	into yardsgainedprior
from 
(
	select 
		posteam, 
		gameid, 
		drive, 
		down, 
		timesecs, 
		row_number() over (partition by posteam, gameid order by timesecs desc) as PlayID
	from excessive_celebration.dbo.NFLPLays_casted
) dt0 
left join 
(
	select 
		posteam, 
		gameid, 
		drive, 
		down, 
		timesecs, 
		row_number() over (partition by posteam, gameid order by timesecs desc) as PlayID,
		[Yards Gained]
	from excessive_celebration.dbo.NFLPLays_casted
) dt1 on dt0.posteam = dt1.posteam and dt0.GameID = dt1.GameID and dt0.PlayID = dt1.PlayID+1 


where dt0.posteam!='' 
