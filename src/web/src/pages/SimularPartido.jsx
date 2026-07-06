import { useState, useEffect, useRef } from 'react'
import { getEquipos, simularPartido } from '../api.js'
import {
  Swords, BarChart3, Play, Loader2, ArrowRight,
  CircleDot, Users, TrendingUp, Trophy
} from 'lucide-react'

export default function SimularPartido() {
  const [equipos, setEquipos] = useState([])
  const [local, setLocal] = useState('')
  const [visitante, setVisitante] = useState('')
  const [resultado, setResultado] = useState(null)
  const [monteCarlo, setMonteCarlo] = useState(null)
  const [loading, setLoading] = useState(false)
  const [modo, setModo] = useState('individual')
  const [animatingSet, setAnimatingSet] = useState(-1)
  const resultRef = useRef(null)

  useEffect(() => {
    getEquipos()
      .then(data => {
        setEquipos(data.equipos || [])
        if (data.equipos?.length >= 2) {
          setLocal(data.equipos[0].nombre)
          setVisitante(data.equipos[1].nombre)
        }
      })
      .catch(console.error)
  }, [])

  const simular = async () => {
    if (!local || !visitante || local === visitante) return
    setLoading(true)
    setResultado(null)
    setMonteCarlo(null)
    setAnimatingSet(-1)

    try {
      if (modo === 'montecarlo') {
        const data = await simularPartido({
          local, visitante,
          n_simulaciones_mc: 2000,
          generar_puntos: false,
        })
        setMonteCarlo(data)
      } else {
        const data = await simularPartido({
          local, visitante,
          generar_puntos: true,
          generar_stats_jugadores: true,
        })
        setResultado(data)
        for (let i = 0; i < data.sets.length; i++) {
          await new Promise(r => setTimeout(r, 500))
          setAnimatingSet(i)
        }
      }
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth' }), 300)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const getTeamColors = (nombre) => {
    const eq = equipos.find(e => e.nombre === nombre)
    return eq?.colores || { primary: '#607D8B', secondary: '#fff' }
  }

  return (
    <div className="page-container">
      <h1 className="section-title fade-in">
        <Swords size={24} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem', color: 'var(--accent-green)' }} />
        Simular Partido
      </h1>
      <p className="section-subtitle fade-in">
        Elige dos equipos y simula un partido completo con resultado punto a punto
      </p>

      {/* Team Selection */}
      <div className="card fade-in" style={{ maxWidth: 700, margin: '0 auto' }}>
        <div className="grid-2">
          <div>
            <label htmlFor="select-local">Equipo Local</label>
            <select id="select-local" value={local} onChange={e => setLocal(e.target.value)}>
              {equipos.map(eq => (
                <option key={eq.nombre} value={eq.nombre}>{eq.nombre}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="select-visitante">Equipo Visitante</label>
            <select id="select-visitante" value={visitante} onChange={e => setVisitante(e.target.value)}>
              {equipos.map(eq => (
                <option key={eq.nombre} value={eq.nombre}>{eq.nombre}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Mode Selection */}
        <div className="mt-2" style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className={`btn ${modo === 'individual' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setModo('individual')}
          >
            <CircleDot size={16} />
            Partido Individual
          </button>
          <button
            className={`btn ${modo === 'montecarlo' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setModo('montecarlo')}
          >
            <BarChart3 size={16} />
            Monte Carlo (2000)
          </button>
        </div>

        <div className="mt-2">
          <button
            className="btn btn-primary btn-lg"
            style={{ width: '100%' }}
            onClick={simular}
            disabled={loading || !local || !visitante || local === visitante}
          >
            {loading ? (
              <><Loader2 size={18} className="spinner" style={{ border: 'none', animation: 'spin 0.7s linear infinite' }} /> Simulando...</>
            ) : (
              <><Play size={18} /> Simular</>
            )}
          </button>
          {local === visitante && local && (
            <p style={{ color: 'var(--accent-red)', fontSize: '0.8rem', marginTop: '0.4rem' }}>
              Los equipos deben ser diferentes
            </p>
          )}
        </div>
      </div>

      {/* Results */}
      <div ref={resultRef}>
        {/* Individual Match Result */}
        {resultado && (
          <div className="mt-4 fade-in">
            {/* Scoreboard */}
            <div className="scoreboard">
              <div className="scoreboard-team">
                <div
                  className="team-card-icon"
                  style={{ background: getTeamColors(resultado.local).primary, width: 52, height: 52, fontSize: '1.2rem' }}
                >
                  {resultado.local.slice(0, 2).toUpperCase()}
                </div>
                <div className="scoreboard-team-name">{resultado.local}</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Local</div>
              </div>

              <div style={{ textAlign: 'center' }}>
                <div className="scoreboard-sets">
                  {resultado.sets_local}
                  <span className="scoreboard-separator"> — </span>
                  {resultado.sets_visitante}
                </div>
                <div className="set-scores">
                  {resultado.sets.map((s, i) => (
                    <div key={i}
                      className={`set-score-pill ${animatingSet >= i ? (s.ganador === resultado.local ? 'won' : 'lost') : ''}`}
                      style={{ opacity: animatingSet >= i ? 1 : 0.2, transition: 'opacity 0.3s ease' }}>
                      {s.puntos_local}-{s.puntos_visitante}
                    </div>
                  ))}
                </div>
                <div className="mt-1" style={{
                  color: 'var(--accent-gold)',
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  textShadow: '0 0 10px rgba(245, 158, 11, 0.2)',
                }}>
                  Ganador: {resultado.ganador}
                </div>
              </div>

              <div className="scoreboard-team">
                <div
                  className="team-card-icon"
                  style={{ background: getTeamColors(resultado.visitante).primary, width: 52, height: 52, fontSize: '1.2rem' }}
                >
                  {resultado.visitante.slice(0, 2).toUpperCase()}
                </div>
                <div className="scoreboard-team-name">{resultado.visitante}</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Visitante</div>
              </div>
            </div>

            {/* Set Details */}
            <div className="mt-3">
              <h3 className="section-title" style={{ fontSize: '1.2rem' }}>Detalle por Set</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }} className="mt-2">
                {resultado.sets.map((set, i) => (
                  <div key={i} className={`card ${animatingSet >= i ? 'slide-in' : ''}`}
                    style={{ opacity: animatingSet >= i ? 1 : 0.12, animationDelay: `${i * 0.08}s` }}>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-1">
                        <span style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                          Set {set.numero}
                        </span>
                        <span className="font-mono" style={{
                          fontWeight: 700,
                          fontSize: '1.15rem',
                          color: 'var(--accent-green)',
                        }}>
                          {set.puntos_local} — {set.puntos_visitante}
                        </span>
                      </div>
                      <span style={{
                        fontWeight: 600,
                        fontSize: '0.85rem',
                        color: set.ganador === resultado.local ? 'var(--accent-green)' : 'var(--accent-purple)',
                      }}>
                        {set.ganador}
                      </span>
                    </div>

                    {/* Player Stats Table */}
                    {(set.stats_local?.length > 0 || set.stats_visitante?.length > 0) && (
                      <div className="mt-2" style={{ fontSize: '0.75rem' }}>
                        <div className="grid-2" style={{ gap: '1rem' }}>
                          {[
                            { stats: set.stats_local, team: resultado.local, color: 'var(--accent-green)' },
                            { stats: set.stats_visitante, team: resultado.visitante, color: 'var(--accent-purple)' },
                          ].map(({ stats, team, color }) => (
                            <div key={team}>
                              <div style={{
                                fontWeight: 600, marginBottom: '0.4rem',
                                color, fontSize: '0.7rem',
                                textTransform: 'uppercase', letterSpacing: '0.04em',
                                display: 'flex', alignItems: 'center', gap: '0.3rem',
                              }}>
                                <Users size={11} /> {team}
                              </div>
                              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                <thead>
                                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                    <th style={{ textAlign: 'left', padding: '0.2rem 0.2rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.65rem' }}>Jugador</th>
                                    <th style={{ textAlign: 'center', padding: '0.2rem 0.15rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.65rem' }}>Pts</th>
                                    <th style={{ textAlign: 'center', padding: '0.2rem 0.15rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.65rem' }}>Ace</th>
                                    <th style={{ textAlign: 'center', padding: '0.2rem 0.15rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.65rem' }}>Atq</th>
                                    <th style={{ textAlign: 'center', padding: '0.2rem 0.15rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.65rem' }}>Blq</th>
                                    <th style={{ textAlign: 'center', padding: '0.2rem 0', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.65rem' }}>Rec</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {stats?.slice(0, 7).map((p, j) => (
                                    <tr key={j} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                      <td style={{ padding: '0.2rem 0', color: 'var(--text-secondary)', maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {p.jugador?.split(' ').slice(-1)[0]}
                                      </td>
                                      <td className="font-mono" style={{ textAlign: 'center', padding: '0.2rem 0.15rem', fontWeight: 600 }}>{p.puntos || 0}</td>
                                      <td className="font-mono" style={{ textAlign: 'center', padding: '0.2rem 0.15rem', color: 'var(--text-muted)' }}>{p.aces || 0}</td>
                                      <td className="font-mono" style={{ textAlign: 'center', padding: '0.2rem 0.15rem', color: 'var(--text-muted)' }}>{p.ataques_ganados || 0}</td>
                                      <td className="font-mono" style={{ textAlign: 'center', padding: '0.2rem 0.15rem', color: 'var(--text-muted)' }}>{p.bloqueos || 0}</td>
                                      <td className="font-mono" style={{ textAlign: 'center', padding: '0.2rem 0', color: 'var(--text-muted)' }}>{p.recepciones_exc || 0}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Match Summary — Accumulated Player Stats */}
            {resultado.sets?.some(s => s.stats_local?.length > 0) && (() => {
              // Accumulate stats across all sets
              const accum = (team) => {
                const playerMap = {}
                resultado.sets.forEach(s => {
                  const stats = team === 'local' ? s.stats_local : s.stats_visitante
                  stats?.forEach(p => {
                    const name = p.jugador || 'Desconocido'
                    if (!playerMap[name]) playerMap[name] = { jugador: name, puntos: 0, aces: 0, ataques_ganados: 0, bloqueos: 0, recepciones_exc: 0 }
                    playerMap[name].puntos += (p.puntos || 0)
                    playerMap[name].aces += (p.aces || 0)
                    playerMap[name].ataques_ganados += (p.ataques_ganados || 0)
                    playerMap[name].bloqueos += (p.bloqueos || 0)
                    playerMap[name].recepciones_exc += (p.recepciones_exc || 0)
                  })
                })
                return Object.values(playerMap).sort((a, b) => b.puntos - a.puntos)
              }

              const localAccum = accum('local')
              const visitAccum = accum('visitante')
              const allPlayers = [...localAccum.map(p => ({ ...p, team: resultado.local })),
              ...visitAccum.map(p => ({ ...p, team: resultado.visitante }))]
              const mvp = allPlayers.sort((a, b) => b.puntos - a.puntos)[0]

              return (
                <div className="mt-3">
                  <h3 className="section-title" style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Trophy size={18} style={{ color: 'var(--accent-gold)' }} />
                    Resumen del Partido
                  </h3>

                  {/* MVP */}
                  {mvp && (
                    <div className="card mt-2" style={{
                      background: 'linear-gradient(135deg, rgba(245,158,11,0.06), rgba(245,158,11,0.02))',
                      border: '1px solid rgba(245,158,11,0.2)',
                      textAlign: 'center', padding: '1rem',
                    }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--accent-gold)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600, marginBottom: '0.3rem' }}>
                        ⭐ MVP del Partido
                      </div>
                      <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>{mvp.jugador}</div>
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{mvp.team}</div>
                      <div className="font-mono" style={{ marginTop: '0.3rem', fontSize: '0.85rem', color: 'var(--accent-green)' }}>
                        {mvp.puntos} pts · {mvp.aces} aces · {mvp.ataques_ganados} atq · {mvp.bloqueos} blq
                      </div>
                    </div>
                  )}

                  {/* Accumulated Tables */}
                  <div className="grid-2 mt-2" style={{ gap: '1rem', fontSize: '0.78rem' }}>
                    {[
                      { players: localAccum, team: resultado.local, color: 'var(--accent-green)' },
                      { players: visitAccum, team: resultado.visitante, color: 'var(--accent-purple)' },
                    ].map(({ players, team, color }) => (
                      <div key={team} className="card">
                        <div style={{ fontWeight: 600, marginBottom: '0.5rem', color, fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                          {team} — Acumulado
                        </div>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                          <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                              <th style={{ textAlign: 'left', padding: '0.25rem 0.2rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.7rem' }}>Jugador</th>
                              <th style={{ textAlign: 'center', padding: '0.25rem 0.2rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.7rem' }}>Pts</th>
                              <th style={{ textAlign: 'center', padding: '0.25rem 0.2rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.7rem' }}>Ace</th>
                              <th style={{ textAlign: 'center', padding: '0.25rem 0.2rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.7rem' }}>Atq</th>
                              <th style={{ textAlign: 'center', padding: '0.25rem 0.2rem', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.7rem' }}>Blq</th>
                              <th style={{ textAlign: 'center', padding: '0.25rem 0', fontWeight: 500, color: 'var(--text-muted)', fontSize: '0.7rem' }}>Rec</th>
                            </tr>
                          </thead>
                          <tbody>
                            {players.map((p, j) => (
                              <tr key={j} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                <td style={{ padding: '0.25rem 0', color: 'var(--text-secondary)', maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                  {p.jugador?.split(' ').slice(-1)[0]}
                                </td>
                                <td className="font-mono" style={{ textAlign: 'center', padding: '0.25rem 0.2rem', fontWeight: 700 }}>{p.puntos}</td>
                                <td className="font-mono" style={{ textAlign: 'center', padding: '0.25rem 0.2rem', color: 'var(--text-muted)' }}>{p.aces}</td>
                                <td className="font-mono" style={{ textAlign: 'center', padding: '0.25rem 0.2rem', color: 'var(--text-muted)' }}>{p.ataques_ganados}</td>
                                <td className="font-mono" style={{ textAlign: 'center', padding: '0.25rem 0.2rem', color: 'var(--text-muted)' }}>{p.bloqueos}</td>
                                <td className="font-mono" style={{ textAlign: 'center', padding: '0.25rem 0', color: 'var(--text-muted)' }}>{p.recepciones_exc}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ))}
                  </div>
                </div>
              )
            })()}
          </div>
        )}

        {/* Monte Carlo Results */}
        {monteCarlo && (
          <div className="mt-4 fade-in">
            <h3 className="section-title" style={{ textAlign: 'center', fontSize: '1.2rem' }}>
              <BarChart3 size={20} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.4rem', color: 'var(--accent-green)' }} />
              Monte Carlo — {monteCarlo.n_simulaciones} simulaciones
            </h3>

            {/* Win probabilities */}
            <div className="scoreboard mt-2">
              <div className="scoreboard-team">
                <div className="scoreboard-team-name">{monteCarlo.local}</div>
                <div className="scoreboard-sets" style={{
                  color: monteCarlo.prob_local > 0.5 ? 'var(--accent-green)' : 'var(--text-muted)',
                  textShadow: monteCarlo.prob_local > 0.5 ? '0 0 15px rgba(34, 197, 94, 0.3)' : 'none',
                }}>
                  {(monteCarlo.prob_local * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>VS</div>
              <div className="scoreboard-team">
                <div className="scoreboard-team-name">{monteCarlo.visitante}</div>
                <div className="scoreboard-sets" style={{
                  color: monteCarlo.prob_visitante > 0.5 ? 'var(--accent-green)' : 'var(--text-muted)',
                  textShadow: monteCarlo.prob_visitante > 0.5 ? '0 0 15px rgba(34, 197, 94, 0.3)' : 'none',
                }}>
                  {(monteCarlo.prob_visitante * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Score Distribution */}
            <div className="card mt-3" style={{ maxWidth: 600, margin: '1rem auto' }}>
              <h4 style={{ marginBottom: '1rem', fontWeight: 600, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <TrendingUp size={16} style={{ color: 'var(--accent-green)' }} />
                Distribución de Resultados
              </h4>
              {Object.entries(monteCarlo.distribucion || {})
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([score, pct]) => {
                  const isLocalWin = parseInt(score[0]) > parseInt(score[2])
                  return (
                    <div key={score} style={{
                      display: 'flex', alignItems: 'center', gap: '0.75rem',
                      marginBottom: '0.4rem',
                    }}>
                      <span className="font-mono" style={{
                        fontWeight: 700, width: 36, textAlign: 'center', fontSize: '0.85rem',
                        color: isLocalWin ? 'var(--accent-green)' : 'var(--accent-purple)',
                      }}>
                        {score}
                      </span>
                      <div style={{
                        flex: 1, height: 20, background: 'var(--bg-primary)',
                        borderRadius: 10, overflow: 'hidden',
                      }}>
                        <div style={{
                          width: `${Math.max(pct * 100, 1)}%`,
                          height: '100%',
                          background: isLocalWin
                            ? 'linear-gradient(90deg, #22C55E, #4ade80)'
                            : 'linear-gradient(90deg, #8b5cf6, #a78bfa)',
                          borderRadius: 10,
                          transition: 'width 0.8s cubic-bezier(0.22, 1, 0.36, 1)',
                          boxShadow: isLocalWin
                            ? '0 0 8px rgba(34,197,94,0.3)'
                            : '0 0 8px rgba(139,92,246,0.3)',
                        }} />
                      </div>
                      <span className="font-mono" style={{
                        fontWeight: 600, width: 50, textAlign: 'right',
                        fontSize: '0.82rem',
                      }}>
                        {(pct * 100).toFixed(1)}%
                      </span>
                    </div>
                  )
                })}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
