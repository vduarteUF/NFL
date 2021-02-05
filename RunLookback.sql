select distinct v2.GameID, 
	   cast(v1.Drive as int) 'Current Drive', 
	   cast(v2.Drive as int) 'Last Run Drive', 
	   v1.qtr 'Current qtr',
	   v2.qtr 'Last Run qtr',  
	   v1.down 'Current down',
	   v2.down 'Last Run down',  
	   v1.TimeSecs, 
	   v1.PlayType
from NFLPlays_allstring v1
	inner join NFLPlays_allstring v2
		on v2.GameID = v1.GameID
		and v2.posteam = v1.posteam
where v1.GameID = '2009091309'
and v1.down != 'NA'
and v2.down != 'NA'
and v1.PlayType != 'No Play'
and cast(v2.qtr as int) <= cast(v1.qtr as int)
and (cast(v2.Drive as int) < cast(v1.Drive as int) or (cast(v2.Drive as int) = cast(v1.Drive as int) and cast(v2.down as int) < cast(v1.down as int)))
and v2.PlayType = 'Run'
order by 1,2