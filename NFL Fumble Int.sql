
select 
	p1.PosTeam as posteam, 
	p1.GameID as GameID,
	p1.TimeSecs as TimeSecs,
	case when max(TotalFumbles) is null then 0 else max(TotalFumbles) end as PriorFumblesInGame,
	case when max(TotalInterception) is null then 0 else max(TotalInterception) end as PriorInterceptInGame
	into NFL_FumbleInt
from excessive_celebration.dbo.NFLPLays_casted p1
left join 
	(
		select 
			posteam, 
			gameid, 
			timesecs,
			sum(cast(Fumble as float)) as TotalFumbles, 
			sum(cast(InterceptionThrown as float)) as TotalInterception
		from excessive_celebration.dbo.NFLPLays_casted 
		group by posteam, gameid, TimeSecs
	) p2 on p1.posteam = p2.posteam and p1.GameID = p2.GameID and p1.TimeSecs<p2.TimeSecs
group by p1.posteam, p1.gameid, p1.TimeSecs
