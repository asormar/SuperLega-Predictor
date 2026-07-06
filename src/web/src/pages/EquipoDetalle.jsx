import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getEquipo } from '../api.js'
import {
  Users, Swords, ArrowLeft, Crosshair,
  TrendingUp, Zap, Shield, Award
} from 'lucide-react'

export default function EquipoDetalle() {
  const { nombre } = useParams()
  const [equipo, setEquipo] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    getEquipo(nombre)
      .then(data => setEquipo(data))
      .catch(err => console.error('Error:', err))
      .finally(() => setLoading(false))
  }, [nombre])

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-spinner">
          <div className="spinner" />
          Cargando equipo...
        </div>
      </div>
    )
  }

  if (!equipo) {
    return (
      <div className="page-container">
        <h1 className="section-title">Equipo no encontrado</h1>
        <Link to="/" className="btn btn-secondary mt-2">
          <ArrowLeft size={16} /> Volver al inicio
        </Link>
      </div>
    )
  }

  const primaryColor = equipo.colores?.primary || '#607D8B'
  const strengthColor = equipo.fuerza >= 0.6 ? 'var(--accent-green)' :
                         equipo.fuerza >= 0.45 ? 'var(--accent-gold)' : 'var(--accent-red)'

  return (
    <div className="page-container">
      {/* Header */}
      <div className="fade-in" style={{
        display: 'flex', alignItems: 'center', gap: '1.25rem',
        marginBottom: '1.5rem',
      }}>
        <div style={{
          width: 64, height: 64, borderRadius: 'var(--radius-lg)',
          background: primaryColor,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: '1.6rem',
          color: 'white',
          boxShadow: `0 0 20px ${primaryColor}33`,
        }}>
          {equipo.nombre.slice(0, 2).toUpperCase()}
        </div>
        <div>
          <h1 className="section-title" style={{ marginBottom: '0.2rem' }}>{equipo.nombre}</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            SuperLega {equipo.temporada}
            <span style={{ color: 'var(--border-glass)' }}>·</span>
            Fuerza:
            <span className="font-mono" style={{ fontWeight: 700, color: strengthColor }}>
              {(equipo.fuerza * 100).toFixed(0)}%
            </span>
          </p>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="flex gap-1 mb-4">
        <Link to="/simular-partido" className="btn btn-primary">
          <Swords size={16} /> Simular Partido
        </Link>
        <Link to="/" className="btn btn-secondary">
          <ArrowLeft size={16} /> Volver
        </Link>
      </div>

      {/* Stats Summary */}
      <div className="stats-grid mb-4 fade-in" style={{ animationDelay: '0.1s' }}>
        <div className="stat-card">
          <Users size={18} style={{ color: 'var(--text-muted)', marginBottom: '0.4rem' }} />
          <div className="stat-value">{equipo.jugadores?.length || 0}</div>
          <div className="stat-label">Jugadores</div>
        </div>
        <div className="stat-card">
          <Crosshair size={18} style={{ color: 'var(--text-muted)', marginBottom: '0.4rem' }} />
          <div className="stat-value">{(equipo.fuerza * 100).toFixed(0)}<span style={{ fontSize: '0.9rem', opacity: 0.6 }}>%</span></div>
          <div className="stat-label">Fuerza</div>
        </div>
        <div className="stat-card">
          <Award size={18} style={{ color: 'var(--text-muted)', marginBottom: '0.4rem' }} />
          <div className="stat-value" style={{ fontSize: '1.1rem' }}>
            {equipo.jugadores?.[0]?.nombre?.split(' ').slice(-1)[0] || '—'}
          </div>
          <div className="stat-label">Mejor Anotador</div>
        </div>
        <div className="stat-card">
          <Zap size={18} style={{ color: 'var(--text-muted)', marginBottom: '0.4rem' }} />
          <div className="stat-value">
            {equipo.jugadores?.[0]?.puntos_por_set?.toFixed(1) || '—'}
          </div>
          <div className="stat-label">Pts/Set (Top)</div>
        </div>
      </div>

      {/* Roster Table */}
      <h2 className="section-title" style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Shield size={20} style={{ color: 'var(--accent-green)' }} />
        Plantilla
      </h2>
      <p className="section-subtitle">Estadísticas por set de la temporada actual</p>

      {equipo.jugadores?.length > 0 ? (
        <div className="table-wrapper fade-in" style={{ animationDelay: '0.15s' }}>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Jugador</th>
                <th style={{ textAlign: 'right' }}>Pts/Set</th>
                <th style={{ textAlign: 'right' }}>Aces/Set</th>
                <th style={{ textAlign: 'right' }}>Atq/Set</th>
                <th style={{ textAlign: 'right' }}>Bloq/Set</th>
                <th style={{ textAlign: 'right' }}>Pts Total</th>
              </tr>
            </thead>
            <tbody>
              {equipo.jugadores.map((jug, i) => (
                <tr key={i} className="slide-in" style={{ animationDelay: `${i * 0.025}s` }}>
                  <td className="font-mono" style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.8rem' }}>{i + 1}</td>
                  <td style={{ fontWeight: 600 }}>{jug.nombre}</td>
                  <td className="font-mono" style={{
                    textAlign: 'right', fontWeight: 700,
                    color: jug.puntos_por_set >= 4 ? 'var(--accent-gold)' :
                           jug.puntos_por_set >= 2 ? 'var(--accent-green)' : 'var(--text-secondary)',
                  }}>
                    {jug.puntos_por_set?.toFixed(2) || '—'}
                  </td>
                  <td className="text-right font-mono">{jug.aces_por_set?.toFixed(2) || '—'}</td>
                  <td className="text-right font-mono">{jug.ataques_ganados_por_set?.toFixed(2) || '—'}</td>
                  <td className="text-right font-mono">{jug.bloqueos_por_set?.toFixed(2) || '—'}</td>
                  <td className="text-right font-mono" style={{ fontWeight: 600, color: 'var(--accent-green)' }}>
                    {jug.puntos_total || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="card" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
          No hay datos de jugadores disponibles para este equipo
        </div>
      )}
    </div>
  )
}
