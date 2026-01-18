import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import sys
import time

def load_frame(filename):
    """加载单帧数据"""
    with open(filename, 'r') as f:
        num_particles = int(f.readline())
        data = np.loadtxt(f)
    return data

def create_animation(output_dir='output', output_file='animation.mp4'):
    """创建3D动画"""
    frame_files = sorted(glob.glob(os.path.join(output_dir, 'frame_*.txt')))
    
    if not frame_files:
        print(f"❌ No frame files found in {output_dir}")
        return
    
    print(f"✓ Found {len(frame_files)} frames")
    print(f"✓ Preparing visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 20)
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Height (Z)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Collision Detection', fontsize=14, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    xx, yy = np.meshgrid(np.linspace(-10, 10, 11), np.linspace(-10, 10, 11))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.15, color='gray', edgecolor='white', linewidth=0.5)
    
    data = load_frame(frame_files[0])
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                        s=data[:, 3] * 1200,
                        c=data[:, 2],
                        cmap='rainbow', alpha=0.85, 
                        edgecolors='black', linewidth=0.5,
                        vmin=0, vmax=20)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Height', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    def update(frame_idx):
        ax.cla()
        
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 20)
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_zlabel('Height (Z)', fontsize=12, fontweight='bold')
        ax.set_title(f'Frame {frame_idx+1}/{len(frame_files)} - GPU Collision Detection', 
                    fontsize=14, fontweight='bold')
        ax.view_init(elev=25, azim=45 + frame_idx * 0.2)
        
        ax.plot_surface(xx, yy, zz, alpha=0.15, color='gray', 
                       edgecolor='white', linewidth=0.5)
        
        data = load_frame(frame_files[frame_idx])
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                           s=data[:, 3] * 1200, c=data[:, 2],
                           cmap='rainbow', alpha=0.85, 
                           edgecolors='black', linewidth=0.5,
                           vmin=0, vmax=20)
        
        info = f'Particles: {len(data)}\n'
        info += f'Radius: {data[:, 3].min():.3f} - {data[:, 3].max():.3f}'
        ax.text2D(0.02, 0.98, info, transform=ax.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        
        if frame_idx % 30 == 0:
            progress = (frame_idx + 1) / len(frame_files) * 100
            print(f"  Rendering: {progress:.1f}% ({frame_idx+1}/{len(frame_files)})")
        
        return scatter,
    
    print(f"✓ Rendering animation...")
    start_time = time.time()
    
    anim = FuncAnimation(fig, update, frames=len(frame_files),
                        interval=33, blit=False)
    
    print(f"\n✓ Encoding video...")
    writer = FFMpegWriter(fps=30, bitrate=12000,
                         metadata={'artist': 'GPU Collision Simulator'})
    
    anim.save(output_file, writer=writer, dpi=120)
    plt.close(fig)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ Video Complete!")
    print(f"  Output: {output_file}")
    print(f"  Duration: {len(frame_files)/30:.1f}s")
    print(f"  Size: {os.path.getsize(output_file)/1e6:.1f} MB")
    print(f"  Render time: {total_time:.1f}s")
    print(f"{'='*60}")

if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'animation.mp4'
    
    print("=" * 60)
    print("  GPU Collision Detection - Video Generator")
    print("=" * 60)
    create_animation(output_dir, output_file)