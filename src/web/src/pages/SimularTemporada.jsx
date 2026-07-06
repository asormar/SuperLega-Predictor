import { useState, useEffect, useRef, useCallback } from 'react'
import { getEquipos, iniciarTemporada, simularJornada } from '../api.js'
import {
  Trophy, Play, Pause, SkipForward, Loader2, CheckSquare, Square,
  Medal, History, AlertCircle, Users, RotateCcw, Calendar, ChevronLeft, ChevronRight
} from 'lucide-react'

const MAX_EQUIPOS = 12
const JORNADA_DELAY_MS = 5000

export default function SimularTemporada() {
  const [equipos, setEquipos] = useState([])
  const [seleccionados, setSeleccionados] = useState([])
  const [dobleVuelta, setDobleVuelta] = useState(true)
  const [semilla, setSemilla] = useState(42)

  // Estado de la simulacion dinamica
  const [phase, setPhase] = useState('config') // 'config' | 'running' | 'complete'
  const [schedule, setSchedule] = useState([])
  const [totalJornadas, setTotalJornadas] = useState(0)
  const [currentJornadaIndex, setCurrentJornadaIndex] = useState(-1)
  const [currentStandings, setCurrentStandings] = useState([])
  const [currentPlayerStats, setCurrentPlayerStats] = useState([])
  const [jornadaHistory, setJornadaHistory] = useState([])
  const [selectedJornadaIndex, setSelectedJornadaIndex] = useState(null)
  // Evita que fetchJornada auto-seleccione la última jornada cuando
  // el usuario ya eligió manualmente una distinta via selectJornada().
  const userSelectedJornadaRef = useRef(false)
  const [isPaused, setIsPaused] = useState(false)
  const [isLoadingJornada, setIsLoadingJornada] = useState(false)
  const [error, setError] = useState(null)

  const timerRef = useRef(null)
  const resultRef = useRef(null)

  useEffect(() => {
    getEquipos()
      .then(data => {
        const eqs = data.equipos || []
        setEquipos(eqs)
        setSeleccionados(eqs.filter(e => e.categoria === 'actual').map(e => e.nombre))
      })
      .catch(console.error)
  }, [])

  // Limpieza del timer al desmontar
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const equiposActuales = equipos.filter(e => e.categoria === 'actual')
  const equiposHistoricos = equipos.filter(e => e.categoria === 'historico')

  const toggleEquipo = (nombre) => {
    setSeleccionados(prev => {
      if (prev.includes(nombre)) return prev.filter(n => n !== nombre)
      if (prev.length >= MAX_EQUIPOS) return prev
      return [...prev, nombre]
    })
  }

  const getTeamColor = (nombre) => {
    const eq = equipos.find(e => e.nombre === nombre)
    return eq?.colores?.primary || '#607D8B'
  }

  const totalPartidos = seleccionados.length * (seleccionados.length - 1) * (dobleVuelta ? 1 : 0.5)

  // ─────────────────────────────────────────────────────────
  // Logica de simulacion jornada a jornada
  // ─────────────────────────────────────────────────────────

  const iniciarSimulacion = async () => {
    if (seleccionados.length < 2) return
    setError(null)
    setIsLoadingJornada(true)
    try {
      const init = await iniciarTemporada({
        equipos: seleccionados,
        doble_vuelta: dobleVuelta,
        semilla: semilla,
      })
      setSchedule(init.schedule)
      setTotalJornadas(init.total_jornadas)
      setCurrentStandings(init.initial_standings)
      setCurrentPlayerStats(init.initial_player_stats)
      setJornadaHistory([])
      setSelectedJornadaIndex(null)
      userSelectedJornadaRef.current = false
      setCurrentJornadaIndex(-1)
      setPhase('running')
      setIsPaused(false)
    } catch (err) {
      setError(err.message)
      console.error(err)
    } finally {
      setIsLoadingJornada(false)
    }
  }

  const fetchJornada = useCallback(async (jornadaIdx) => {
    if (jornadaIdx < 0 || jornadaIdx >= totalJornadas) return
    setIsLoadingJornada(true)
    setError(null)
    try {
      const resp = await simularJornada({
        equipos: seleccionados,
        doble_vuelta: dobleVuelta,
        schedule: schedule,
        jornada_index: jornadaIdx,
        current_standings: currentStandings,
        current_player_stats: currentPlayerStats,
        semilla: semilla,
      })
      setCurrentStandings(resp.updated_standings)
      setCurrentPlayerStats(resp.updated_player_stats)
      setCurrentJornadaIndex(resp.jornada_index)

      const snapshot = {
        jornada_index: resp.jornada_index,
        jornada_num: resp.jornada_num ?? resp.jornada_index + 1,
        matches: resp.matches,
        standings_snapshot: resp.updated_standings,
        player_stats_snapshot: resp.updated_player_stats,
      }
      setJornadaHistory(prev => {
        const filtered = prev.filter(j => j.jornada_index !== snapshot.jornada_index)
        return [...filtered, snapshot].sort((a, b) => a.jornada_index - b.jornada_index)
      })
      if (!userSelectedJornadaRef.current) {
        setSelectedJornadaIndex(resp.jornada_index)
      }
      if (resp.is_complete) {
        setPhase('complete')
        setIsPaused(false)
      }
    } catch (err) {
      setError(err.message)
      console.error(err)
      setIsPaused(true)
    } finally {
      setIsLoadingJornada(false)
    }
  }, [seleccionados, dobleVuelta, schedule, totalJornadas, currentStandings, currentPlayerStats, semilla])

  // Programar la siguiente jornada
  useEffect(() => {
    if (phase !== 'running') return
    if (isPaused) return
    if (isLoadingJornada) return
    if (currentJornadaIndex >= totalJornadas - 1) return

    timerRef.current = setTimeout(() => {
      fetchJornada(currentJornadaIndex + 1)
    }, JORNADA_DELAY_MS)

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [phase, isPaused, isLoadingJornada, currentJornadaIndex, totalJornadas, fetchJornada])

  // Scroll automatico al resultado
  useEffect(() => {
    if (currentJornadaIndex >= 0 && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [currentJornadaIndex])

  const togglePausa = () => {
    if (phase === 'complete') return
    setIsPaused(prev => !prev)
  }

  const siguienteJornada = () => {
    if (timerRef.current) clearTimeout(timerRef.current)
    if (currentJornadaIndex < totalJornadas - 1) {
      fetchJornada(currentJornadaIndex + 1)
    }
  }

  const nuevaSimulacion = () => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setPhase('config')
    setSchedule([])
    setTotalJornadas(0)
    setCurrentJornadaIndex(-1)
    setCurrentStandings([])
    setCurrentPlayerStats([])
    setJornadaHistory([])
    setSelectedJornadaIndex(null)
    userSelectedJornadaRef.current = false
    setIsPaused(false)
    setIsLoadingJornada(false)
    setError(null)
  }

  const jornadaLabel = () => {
    const displayIdx = selectedJornadaIndex ?? currentJornadaIndex
    if (displayIdx < 0) return 'Preparando…'
    const isNotLatest = selectedJornadaIndex !== null && selectedJornadaIndex !== currentJornadaIndex
    let label
    if (dobleVuelta) {
      const half = totalJornadas / 2
      const phaseLabel = displayIdx < half ? 'Ida' : 'Vuelta'
      label = `Jornada ${displayIdx + 1} de ${totalJornadas} · ${phaseLabel}`
    } else {
      label = `Jornada ${displayIdx + 1} de ${totalJornadas}`
    }
    if (isNotLatest) return `${label} (vista)`
    return label
  }

  const progressPct = totalJornadas > 0
    ? Math.round(((currentJornadaIndex + 1) / totalJornadas) * 100)
    : 0

  // ─────────────────────────────────────────────────────────
  // Render: configuracion
  // ─────────────────────────────────────────────────────────

  if (phase === 'config') {
    return (
      <div className="page-container">
        <h1 className="section-title fade-in">
          <Trophy size={24} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem', color: 'var(--accent-gold)' }} />
          Simular Temporada
        </h1>
        <p className="section-subtitle fade-in">
          Simula una temporada completa jornada a jornada, con pausa y avance manual
        </p>

        <div className="card fade-in">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ fontWeight: 700, fontSize: '0.95rem' }}>
              Equipos
              <span className="font-mono" style={{
                marginLeft: '0.5rem',
                color: seleccionados.length >= MAX_EQUIPOS ? 'var(--accent-gold)' : 'var(--accent-green)',
                fontSize: '0.85rem',
              }}>
                {seleccionados.length}/{MAX_EQUIPOS}
              </span>
            </h3>
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              <button className="btn btn-secondary" style={{ fontSize: '0.75rem', padding: '0.35rem 0.65rem' }}
                onClick={() => setSeleccionados(equiposActuales.map(e => e.nombre))}>
                <CheckSquare size={14} /> Actuales
              </button>
              <button className="btn btn-secondary" style={{ fontSize: '0.75rem', padding: '0.35rem 0.65rem' }}
                onClick={() => setSeleccionados([])}>
                <Square size={14} /> Ninguno
              </button>
            </div>
          </div>

          {seleccionados.length >= MAX_EQUIPOS && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: '0.4rem',
              padding: '0.5rem 0.75rem', marginBottom: '0.75rem',
              background: 'rgba(245, 158, 11, 0.08)',
              border: '1px solid rgba(245, 158, 11, 0.2)',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.8rem', color: 'var(--accent-gold)',
            }}>
              <AlertCircle size={14} />
              Límite de {MAX_EQUIPOS} equipos alcanzado. Deselecciona uno para añadir otro.
            </div>
          )}

          <div style={{ marginBottom: '0.5rem', fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent-green)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Temporada Actual
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(155px, 1fr))',
            gap: '0.4rem',
            marginBottom: '1rem',
          }}>
            {equiposActuales.map(eq => {
              const selected = seleccionados.includes(eq.nombre)
              const disabled = !selected && seleccionados.length >= MAX_EQUIPOS
              return (
                <button
                  key={eq.nombre}
                  onClick={() => !disabled && toggleEquipo(eq.nombre)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '0.45rem',
                    padding: '0.5rem 0.7rem',
                    background: selected ? 'rgba(34, 197, 94, 0.08)' : 'var(--bg-primary)',
                    border: `1px solid ${selected ? 'rgba(34, 197, 94, 0.3)' : 'var(--border-subtle)'}`,
                    borderRadius: 'var(--radius-sm)',
                    color: disabled ? 'var(--text-muted)' : selected ? 'var(--text-primary)' : 'var(--text-secondary)',
                    cursor: disabled ? 'not-allowed' : 'pointer',
                    fontFamily: 'var(--font-body)',
                    fontSize: '0.82rem',
                    fontWeight: selected ? 600 : 400,
                    opacity: disabled ? 0.4 : 1,
                    transition: 'all 150ms ease',
                  }}
                >
                  <span className="team-dot" style={{ background: eq.colores?.primary || '#607D8B' }} />
                  {eq.nombre}
                </button>
              )
            })}
          </div>

          <div style={{ marginBottom: '0.5rem', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <History size={12} /> Históricos
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(155px, 1fr))',
            gap: '0.4rem',
            marginBottom: '1rem',
          }}>
            {equiposHistoricos.map(eq => {
              const selected = seleccionados.includes(eq.nombre)
              const disabled = !selected && seleccionados.length >= MAX_EQUIPOS
              return (
                <button
                  key={eq.nombre}
                  onClick={() => !disabled && toggleEquipo(eq.nombre)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '0.45rem',
                    padding: '0.5rem 0.7rem',
                    background: selected ? 'rgba(34, 197, 94, 0.08)' : 'var(--bg-primary)',
                    border: `1px solid ${selected ? 'rgba(34, 197, 94, 0.3)' : 'var(--border-subtle)'}`,
                    borderRadius: 'var(--radius-sm)',
                    color: disabled ? 'var(--text-muted)' : selected ? 'var(--text-text-primary)' : 'var(--text-secondary)',
                    cursor: disabled ? 'not-allowed' : 'pointer',
                    fontFamily: 'var(--font-body)',
                    fontSize: '0.82rem',
                    fontWeight: selected ? 600 : 400,
                    opacity: disabled ? 0.4 : 1,
                    transition: 'all 150ms ease',
                  }}
                >
                  <span className="team-dot" style={{ background: eq.colores?.primary || '#607D8B' }} />
                  {eq.nombre}
                </button>
              )
            })}
          </div>

          <div className="mt-2" style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', margin: 0, textTransform: 'none', letterSpacing: 'normal', fontSize: '0.85rem' }}>
              <input
                type="checkbox"
                checked={dobleVuelta}
                onChange={e => setDobleVuelta(e.target.checked)}
                style={{ accentColor: 'var(--accent-green)', width: 16, height: 16 }}
              />
              Doble vuelta (ida y vuelta)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
              Semilla:
              <input
                type="number"
                value={semilla}
                onChange={e => setSemilla(parseInt(e.target.value) || 0)}
                style={{
                  width: 80, padding: '0.3rem 0.5rem',
                  background: 'var(--bg-primary)', color: 'var(--text-primary)',
                  border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)',
                  fontFamily: 'var(--font-mono)', fontSize: '0.85rem',
                }}
              />
              <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                (controla el orden del calendario y los resultados)
              </span>
            </label>
          </div>

          {error && (
            <div style={{
              marginTop: '1rem', padding: '0.6rem 0.8rem',
              background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: 'var(--radius-sm)', color: 'var(--accent-red)', fontSize: '0.85rem',
            }}>
              {error}
            </div>
          )}

          <div className="mt-3">
            <button
              className="btn btn-primary btn-lg"
              style={{ width: '100%' }}
              onClick={iniciarSimulacion}
              disabled={isLoadingJornada || seleccionados.length < 2}
            >
              {isLoadingJornada ? (
                <><Loader2 size={18} style={{ animation: 'spin 0.7s linear infinite' }} /> Inicializando…</>
              ) : (
                <><Play size={18} /> Iniciar simulación ({seleccionados.length} equipos, ~{Math.round(totalPartidos)} partidos)</>
              )}
            </button>
          </div>
        </div>
      </div>
    )
  }

  // ─────────────────────────────────────────────────────────
  // Render: simulacion en curso / completada
  // ─────────────────────────────────────────────────────────

  const isComplete = phase === 'complete'
  const isRunning = phase === 'running'
  const standings = currentStandings
  const topPlayers = [...currentPlayerStats]
    .sort((a, b) => (b.puntos || 0) - (a.puntos || 0))
    .slice(0, 10)

  const selectedJornada = (() => {
    if (selectedJornadaIndex !== null) {
      return jornadaHistory.find(j => j.jornada_index === selectedJornadaIndex) || null
    }
    if (jornadaHistory.length > 0) {
      return jornadaHistory[jornadaHistory.length - 1]
    }
    return null
  })()
  const displayMatches = selectedJornada?.matches ?? []
  const isViewingPastJornada = selectedJornadaIndex !== null
    && selectedJornadaIndex !== currentJornadaIndex

  const selectJornada = (idx) => {
    userSelectedJornadaRef.current = true
    setSelectedJornadaIndex(idx)
  }

  const jumpToLatest = () => {
    userSelectedJornadaRef.current = false
    setSelectedJornadaIndex(currentJornadaIndex)
  }

  return (
    <div className="page-container">
      <h1 className="section-title fade-in">
        <Trophy size={24} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem', color: 'var(--accent-gold)' }} />
        Simular Temporada
      </h1>

      {/* Header de control */}
      <div className="card fade-in" ref={resultRef} style={{
        borderLeft: `4px solid ${isComplete ? 'var(--accent-green)' : isPaused ? 'var(--accent-gold)' : 'var(--accent-green)'}`,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '0.75rem' }}>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
              {isComplete ? 'Temporada finalizada' : isPaused ? '⏸ Pausado' : '▶ Simulando'}
            </div>
            <div style={{ fontSize: '1.05rem', fontWeight: 700, marginTop: '0.15rem' }}>
              {jornadaLabel()}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            {!isComplete && (
              <>
                <button
                  className="btn"
                  onClick={togglePausa}
                  disabled={isLoadingJornada && currentJornadaIndex === -1}
                  style={{
                    background: isPaused ? 'var(--accent-gold)' : 'var(--bg-elevated)',
                    color: isPaused ? '#000' : 'var(--text-primary)',
                    border: `1px solid ${isPaused ? 'var(--accent-gold)' : 'var(--border-subtle)'}`,
                  }}
                >
                  {isPaused ? <><Play size={16} /> Reanudar</> : <><Pause size={16} /> Pausar</>}
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={siguienteJornada}
                  disabled={isLoadingJornada || currentJornadaIndex >= totalJornadas - 1}
                >
                  <SkipForward size={16} /> Siguiente
                </button>
              </>
            )}
            <button className="btn btn-secondary" onClick={nuevaSimulacion}>
              <RotateCcw size={16} /> Nueva
            </button>
          </div>
        </div>

        {/* Barra de progreso */}
        <div style={{ width: '100%', height: 8, background: 'var(--bg-primary)', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            width: `${progressPct}%`,
            height: '100%',
            background: isComplete ? 'var(--accent-green)' : isPaused ? 'var(--accent-gold)' : 'var(--accent-green)',
            transition: 'width 400ms ease',
          }} />
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.35rem', fontFamily: 'var(--font-mono)' }}>
          {progressPct}% completado
        </div>

        {error && (
          <div style={{
            marginTop: '0.75rem', padding: '0.5rem 0.75rem',
            background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 'var(--radius-sm)', color: 'var(--accent-red)', fontSize: '0.85rem',
          }}>
            {error}
          </div>
        )}
      </div>

      {/* Selector y resultados de jornada */}
      {jornadaHistory.length > 0 && (
        <div className="card mt-3 fade-in" key={`jornada-${selectedJornadaIndex}`}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '0.75rem' }}>
            <h3 style={{ fontWeight: 700, fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.4rem', margin: 0 }}>
              <Calendar size={16} style={{ color: 'var(--accent-green)' }} />
              {selectedJornada
                ? `Resultados — Jornada ${selectedJornada.jornada_num}`
                : 'Resultados de la jornada'}
              {isLoadingJornada && (
                <Loader2 size={14} style={{ marginLeft: '0.4rem', animation: 'spin 0.7s linear infinite' }} />
              )}
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <button
                className="btn"
                onClick={() => {
                  if (selectedJornadaIndex > 0) selectJornada(selectedJornadaIndex - 1)
                }}
                disabled={selectedJornadaIndex === null || selectedJornadaIndex <= 0}
                style={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-subtle)',
                  padding: '0.3rem 0.5rem',
                }}
                title="Jornada anterior"
              >
                <ChevronLeft size={14} />
              </button>
              <select
                value={selectedJornadaIndex ?? ''}
                onChange={e => selectJornada(Number(e.target.value))}
                style={{
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.3rem 0.6rem',
                  fontSize: '0.85rem',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {jornadaHistory.map(j => {
                  const isLatest = j.jornada_index === currentJornadaIndex
                  return (
                    <option key={j.jornada_index} value={j.jornada_index}>
                      {`Jornada ${j.jornada_num}${isLatest ? ' (actual)' : ''}`}
                    </option>
                  )
                })}
              </select>
              <button
                className="btn"
                onClick={() => {
                  if (selectedJornadaIndex < currentJornadaIndex) selectJornada(selectedJornadaIndex + 1)
                }}
                disabled={selectedJornadaIndex === null || selectedJornadaIndex >= currentJornadaIndex}
                style={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-subtle)',
                  padding: '0.3rem 0.5rem',
                }}
                title="Jornada siguiente"
              >
                <ChevronRight size={14} />
              </button>
              {isViewingPastJornada && (
                <button
                  className="btn"
                  onClick={jumpToLatest}
                  style={{
                    background: 'var(--accent-gold)',
                    color: '#000',
                    border: '1px solid var(--accent-gold)',
                    padding: '0.3rem 0.6rem',
                    fontSize: '0.8rem',
                  }}
                  title="Volver a la jornada actual"
                >
                  <SkipForward size={14} /> Actual
                </button>
              )}
            </div>
          </div>

          <div style={{
            display: 'flex',
            gap: '0.3rem',
            overflowX: 'auto',
            paddingBottom: '0.4rem',
            marginBottom: '0.75rem',
            borderBottom: '1px solid var(--border-subtle)',
          }}>
            {jornadaHistory.map(j => {
              const isActive = j.jornada_index === selectedJornadaIndex
              const isLatest = j.jornada_index === currentJornadaIndex
              return (
                <button
                  key={j.jornada_index}
                  onClick={() => selectJornada(j.jornada_index)}
                  className="slide-in"
                  style={{
                    flex: '0 0 auto',
                    padding: '0.3rem 0.7rem',
                    fontSize: '0.78rem',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: isActive ? 700 : 500,
                    background: isActive
                      ? 'var(--accent-green)'
                      : isLatest
                        ? 'var(--bg-elevated)'
                        : 'var(--bg-primary)',
                    color: isActive ? '#000' : isLatest ? 'var(--accent-gold)' : 'var(--text-secondary)',
                    border: `1px solid ${isActive ? 'var(--accent-green)' : isLatest ? 'var(--accent-gold)' : 'var(--border-subtle)'}`,
                    borderRadius: 'var(--radius-sm)',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25rem',
                  }}
                >
                  J{j.jornada_num}
                  {isLatest && !isActive && <span style={{ fontSize: '0.65rem' }}>●</span>}
                </button>
              )
            })}
          </div>

          {selectedJornada && (
            <div style={{ display: 'grid', gap: '0.5rem' }}>
              {displayMatches.map((m, i) => {
                const localGano = m.ganador === m.local
                return (
                  <div key={i} className="slide-in" style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr auto 1fr',
                    alignItems: 'center',
                    padding: '0.65rem 0.85rem',
                    background: 'var(--bg-primary)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-sm)',
                    borderLeft: `4px solid ${getTeamColor(m.local)}`,
                    animationDelay: `${i * 0.05}s`,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', justifyContent: 'flex-end' }}>
                      <span style={{ fontWeight: localGano ? 700 : 500, color: localGano ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                        {m.local}
                      </span>
                      <span className="team-dot" style={{ background: getTeamColor(m.local) }} />
                    </div>
                    <div className="font-mono" style={{
                      padding: '0.2rem 0.6rem',
                      background: 'var(--bg-elevated)',
                      borderRadius: 'var(--radius-sm)',
                      fontWeight: 700,
                      fontSize: '0.9rem',
                      border: '1px solid var(--border-subtle)',
                    }}>
                      {m.resultado}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="team-dot" style={{ background: getTeamColor(m.visitante) }} />
                      <span style={{ fontWeight: !localGano ? 700 : 500, color: !localGano ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                        {m.visitante}
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Clasificacion acumulada */}
      {standings.length > 0 && (
        <div className="mt-4 fade-in">
          <div className="flex justify-between items-center mb-2">
            <h2 className="section-title" style={{ marginBottom: '0.15rem' }}>
              {isComplete ? '🏆 Clasificación Final' : '📋 Clasificación'}
            </h2>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {isComplete ? `${standings.reduce((acc, s) => acc + s.pj, 0) / 2} partidos` : 'Actualizada jornada a jornada'}
            </span>
          </div>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th style={{ width: 36 }}>#</th>
                  <th>Equipo</th>
                  <th style={{ textAlign: 'right' }}>Pts</th>
                  <th style={{ textAlign: 'right' }}>PJ</th>
                  <th style={{ textAlign: 'right' }}>PG</th>
                  <th style={{ textAlign: 'right' }}>PP</th>
                  <th style={{ textAlign: 'right' }}>SG</th>
                  <th style={{ textAlign: 'right' }}>SP</th>
                  <th style={{ textAlign: 'right' }}>SR</th>
                </tr>
              </thead>
              <tbody>
                {standings.map((row, i) => {
                  const total = standings.length
                  const posColor = i === 0 ? 'var(--accent-gold)' :
                                  i <= 2 ? 'var(--accent-green)' :
                                  i >= total - 2 ? 'var(--accent-red)' :
                                  'var(--text-muted)'
                  return (
                    <tr key={row.equipo} className="slide-in" style={{ animationDelay: `${i * 0.03}s` }}>
                      <td>
                        <span style={{ fontWeight: 700, color: posColor, display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                          {i === 0 && <Medal size={14} />}
                          {i + 1}
                        </span>
                      </td>
                      <td>
                        <div className="team-badge">
                          <span className="team-dot" style={{ background: getTeamColor(row.equipo) }} />
                          <span style={{ fontWeight: 600 }}>{row.equipo}</span>
                        </div>
                      </td>
                      <td className="font-mono" style={{ textAlign: 'right', fontWeight: 700, color: 'var(--accent-green)' }}>{row.puntos}</td>
                      <td className="font-mono" style={{ textAlign: 'right' }}>{row.pj}</td>
                      <td className="font-mono" style={{ textAlign: 'right', color: 'var(--accent-green)' }}>{row.pg}</td>
                      <td className="font-mono" style={{ textAlign: 'right', color: row.pp > 0 ? 'var(--accent-red)' : 'var(--text-muted)' }}>{row.pp}</td>
                      <td className="font-mono" style={{ textAlign: 'right' }}>{row.sg}</td>
                      <td className="font-mono" style={{ textAlign: 'right' }}>{row.sp}</td>
                      <td className="font-mono" style={{ textAlign: 'right', fontWeight: 600 }}>{row.sr}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Top jugadores */}
      {topPlayers.length > 0 && (
        <div className="card mt-3 fade-in">
          <h3 style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Users size={16} style={{ color: 'var(--accent-green)' }} />
            Top Jugadores {isComplete ? '— Temporada Completa' : '— Hasta el Momento'}
          </h3>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th style={{ width: 30 }}>#</th>
                  <th>Jugador</th>
                  <th>Equipo</th>
                  <th style={{ textAlign: 'right' }}>Pts</th>
                  <th style={{ textAlign: 'right' }}>Aces</th>
                  <th style={{ textAlign: 'right' }}>Atq</th>
                  <th style={{ textAlign: 'right' }}>Blq</th>
                  <th style={{ textAlign: 'right' }}>Rec</th>
                  <th style={{ textAlign: 'right' }}>PJ</th>
                </tr>
              </thead>
              <tbody>
                {topPlayers.map((p, i) => (
                  <tr key={`${p.equipo}-${p.jugador}`} className="slide-in" style={{ animationDelay: `${i * 0.02}s` }}>
                    <td>
                      <span style={{
                        fontWeight: 700,
                        color: i === 0 ? 'var(--accent-gold)' : i < 3 ? 'var(--accent-green)' : 'var(--text-muted)',
                      }}>
                        {i === 0 && '⭐ '}{i + 1}
                      </span>
                    </td>
                    <td style={{ fontWeight: 600 }}>{p.jugador}</td>
                    <td>
                      <div className="team-badge">
                        <span className="team-dot" style={{ background: getTeamColor(p.equipo) }} />
                        {p.equipo}
                      </div>
                    </td>
                    <td className="font-mono" style={{ textAlign: 'right', fontWeight: 700, color: 'var(--accent-green)' }}>{p.puntos}</td>
                    <td className="font-mono" style={{ textAlign: 'right' }}>{p.aces || 0}</td>
                    <td className="font-mono" style={{ textAlign: 'right' }}>{p.ataques_ganados || 0}</td>
                    <td className="font-mono" style={{ textAlign: 'right' }}>{p.bloqueos || 0}</td>
                    <td className="font-mono" style={{ textAlign: 'right' }}>{p.recepciones_exc || 0}</td>
                    <td className="font-mono" style={{ textAlign: 'right', color: 'var(--text-muted)' }}>{p.partidos || 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Indicador de carga inicial */}
      {currentJornadaIndex < 0 && isLoadingJornada && (
        <div className="card mt-3" style={{ textAlign: 'center', padding: '2rem' }}>
          <Loader2 size={32} style={{ animation: 'spin 0.7s linear infinite', color: 'var(--accent-green)' }} />
          <div style={{ marginTop: '0.5rem', color: 'var(--text-muted)' }}>Cargando primera jornada…</div>
        </div>
      )}

      {/* Temporizador / estado de espera */}
      {isRunning && !isPaused && !isLoadingJornada && currentJornadaIndex >= 0 && currentJornadaIndex < totalJornadas - 1 && (
        <div style={{
          textAlign: 'center', marginTop: '1.5rem', fontSize: '0.8rem', color: 'var(--text-muted)',
        }}>
          Siguiente jornada en {JORNADA_DELAY_MS / 1000}s…
        </div>
      )}
    </div>
  )
}
