import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getEquipos } from '../api.js'
import {
  Swords, Trophy, Users, Database, Crosshair, TrendingUp,
  ChevronRight, Activity, History
} from 'lucide-react'

export default function Dashboard() {
  const [equipos, setEquipos] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getEquipos()
      .then(data => setEquipos(data.equipos || []))
      .catch(err => console.error('Error cargando equipos:', err))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="page-container">
      {/* Hero Section */}
      <section className="hero-glow fade-in" style={{ textAlign: 'center', padding: '3.5rem 0 2.5rem', position: 'relative', zIndex: 1 }}>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 900,
          fontSize: 'clamp(2rem, 5vw, 3.2rem)',
          lineHeight: 1.1,
          marginBottom: '0.75rem',
          letterSpacing: '-0.03em',
          color: 'var(--text-heading)',
        }}>
          Simulador{' '}
          <span style={{
            color: 'var(--accent-green)',
            textShadow: '0 0 20px rgba(34, 197, 94, 0.25)',
          }}>SuperLega</span>
        </h1>
        <p style={{
          color: 'var(--text-secondary)',
          fontSize: '1.05rem',
          maxWidth: 560,
          margin: '0 auto 2rem',
          lineHeight: 1.6,
        }}>
          Simula partidos y temporadas completas de la liga italiana de volleyball
          con modelos de Machine Learning y simulación Monte Carlo
        </p>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link to="/simular-partido" className="btn btn-primary btn-lg">
            <Swords size={18} />
            Simular Partido
          </Link>
          <Link to="/simular-temporada" className="btn btn-gold btn-lg">
            <Trophy size={18} />
            Simular Temporada
          </Link>
        </div>
      </section>

      {/* Stats Overview */}
      <section className="mt-4 fade-in" style={{ animationDelay: '0.1s' }}>
        <div className="stats-grid">
          <div className="stat-card">
            <Users size={20} style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }} />
            <div className="stat-value">{equipos.length}</div>
            <div className="stat-label">Equipos</div>
          </div>
          <div className="stat-card">
            <Database size={20} style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }} />
            <div className="stat-value">674</div>
            <div className="stat-label">Jugadores</div>
          </div>
          <div className="stat-card">
            <Activity size={20} style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }} />
            <div className="stat-value">10</div>
            <div className="stat-label">Temporadas</div>
          </div>
          <div className="stat-card">
            <Crosshair size={20} style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }} />
            <div className="stat-value">62.2<span style={{ fontSize: '0.9rem', opacity: 0.7 }}>%</span></div>
            <div className="stat-label">Precisión ML</div>
          </div>
        </div>
      </section>

      {/* Teams Grid */}
      <section className="mt-4">
        <div className="flex justify-between items-center" style={{ marginBottom: '1.25rem' }}>
          <div>
            <h2 className="section-title" style={{ marginBottom: '0.15rem' }}>Equipos Disponibles</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{equipos.filter(e => e.categoria === 'actual').length} actuales + {equipos.filter(e => e.categoria === 'historico').length} históricos</p>
          </div>
        </div>

        {loading ? (
          <div className="loading-spinner">
            <div className="spinner" />
            Cargando equipos...
          </div>
        ) : (
          <div className="teams-grid">
            {equipos.map((equipo, i) => (
              <Link
                key={equipo.nombre}
                to={`/equipo/${encodeURIComponent(equipo.nombre)}`}
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <div
                  className="card card-interactive fade-in"
                  style={{ animationDelay: `${i * 0.04}s` }}
                >
                  <div className="team-card-header">
                    <div
                      className="team-card-icon"
                      style={{ background: equipo.colores?.primary || '#607D8B' }}
                    >
                      {equipo.nombre.slice(0, 2).toUpperCase()}
                    </div>
                    <div className="team-card-info" style={{ flex: 1 }}>
                      <h3>{equipo.nombre}</h3>
                      <p style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        {equipo.num_jugadores} jugadores
                        {equipo.categoria === 'historico' && (
                          <span style={{
                            display: 'inline-flex', alignItems: 'center', gap: '0.15rem',
                            fontSize: '0.65rem', color: 'var(--text-muted)',
                            background: 'rgba(255,255,255,0.05)',
                            padding: '0.1rem 0.35rem',
                            borderRadius: '4px',
                            border: '1px solid var(--border-subtle)',
                          }}>
                            <History size={9} /> hist.
                          </span>
                        )}
                      </p>
                    </div>
                    <ChevronRight size={16} style={{ color: 'var(--text-muted)', opacity: 0.5 }} />
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      Fuerza
                    </span>
                    <span className="font-mono" style={{
                      fontWeight: 600,
                      fontSize: '0.85rem',
                      color: equipo.fuerza >= 0.6 ? 'var(--accent-green)' :
                        equipo.fuerza >= 0.45 ? 'var(--accent-gold)' : 'var(--accent-red)',
                    }}>
                      {(equipo.fuerza * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="strength-bar">
                    <div
                      className="strength-fill"
                      style={{
                        width: `${equipo.fuerza * 100}%`,
                        background: equipo.fuerza >= 0.6 ? 'var(--accent-green)' :
                          equipo.fuerza >= 0.45 ? 'var(--accent-gold)' : 'var(--accent-red)',
                        boxShadow: equipo.fuerza >= 0.6 ? '0 0 8px rgba(34,197,94,0.3)' : 'none',
                      }}
                    />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
