import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class Plotter:   
    
    def __init__(self, data=None, model=None):
        
        self.data = data
        self.model = model
        
        
    def learning_curves(self):
        
        log = self.model.log
        epochs = np.arange(0, log['N_epochs'], log['freq_log'])

        fig, axes = plt.subplots(1, 2, figsize=(8, 2.5))
        
        loss_list = ['loss_test', 'loss_ICBC', 'loss_F']
        
        for ax in axes:        
            for i, loss in enumerate(loss_list):
                ax.plot(epochs, log[loss], c=f'C{i}', label=loss)
                ax.axhline(log[loss][-1], c=f'C{i}', lw=1, ls='--')

     
            ax.set_yscale('log')
            ax.grid(ls='--')
           
        axes[0].set_ylim(1e-11, 1e-01)        
        axes[0].legend(loc=1)

        plt.tight_layout()
        seed = self.model.seed
        fig_file = self.model.model_path.joinpath(f'learning_curves_{seed}.png')
        plt.savefig(fig_file)
        plt.show()


    def performance(self):

        # Continuous rho plot
        if isinstance(self.data.rho, list):
            self._performance_continuous()
        # Discrete rho plots
        else:
            self._performance_discrete()
        
    
    def _performance_continuous(self):
        
        fig, axes = plt.subplots(6, 5, sharey='row', sharex='row', figsize=(14, 15))

        rho_lst = np.linspace(self.data.r_min, self.data.r_max, 5)
        for i, rho in enumerate(rho_lst):
            rho = int(rho)
            axes[0, i].set_title(fr'$\rho={rho}$')
            # get mesh data and prediction
            X_mesh, u_mesh, t_ticks, x_ticks = self.data.mesh(rho)
            u_pred = self.model(X_mesh).numpy()
            u_t, u_xx, source_term, F_res, F_res_weighted = self.model._F_residuals(tf.convert_to_tensor(X_mesh))
            # reshaping
            u_mesh = u_mesh.reshape(len(x_ticks), len(t_ticks))
            u_pred = u_pred.reshape(len(x_ticks), len(t_ticks))
            u_res = np.abs(u_pred-u_mesh).reshape(len(x_ticks), len(t_ticks))
            F_res = np.abs(F_res).reshape(len(x_ticks), len(t_ticks))
            F_res_weighted = np.abs(F_res_weighted).reshape(len(x_ticks), len(t_ticks)) 

            axes[0, i].contourf(t_ticks, x_ticks, u_mesh, levels=100, cmap='coolwarm')
            axes[1, i].contourf(t_ticks, x_ticks, u_pred, levels=100, cmap='coolwarm')
            axes[2, i].contourf(t_ticks, x_ticks, u_res, levels=100, cmap='jet')
            axes[3, i].contourf(t_ticks, x_ticks, F_res, levels=100, cmap='jet')
            axes[4, i].contourf(t_ticks, x_ticks, F_res_weighted, levels=100, cmap='jet')

            #for t in [0.001, 0.0015, 0.0020, 0.0025, 0.0030]:
            for t in [0, 0.0040]:
                # time lines
                axes[0, i].axvline(t, ls='--', lw=1, c='white')
                axes[1, i].axvline(t, ls='--', lw=1, c='white')
                axes[2, i].axvline(t, ls='--', lw=1, c='white')
                # Cut data and prediction
                X_cut, u_cut = self.data.cut(t, rho)
                u_pred = self.model(X_cut)
                axes[5, i].plot(X_cut[:,1], u_cut, ls='--', lw=1, c='black')
                axes[5, i].plot(X_cut[:,1], u_pred, lw=1, c='red')
                axes[5, i].grid(ls='--')
                
              

        # dummy legend()
        axes[5, 0].plot([], [], ls='--', lw=1, c='black', label='Ref.')
        axes[5, 0].plot([], [], lw=1, c='red', label='Pred.')
        axes[5, 0].legend(frameon=False)
        axes[5, 0].set_xlim(self.data.x_min, self.data.x_max)
        axes[5, 0].set_ylim(0, 1)
        
        # axis label
        axes[0, 0].set_ylabel('Reference')
        axes[1, 0].set_ylabel('Prediction')
        axes[2, 0].set_ylabel('Data Residuals')
        axes[3, 0].set_ylabel('Physics Residuals')
        axes[4, 0].set_ylabel('Physics Residuals (weighted)')
        axes[5, 0].set_ylabel(r'$x$-cuts')

        seed = self.model.seed
        plt.tight_layout()
        fig_file = self.model.model_path.joinpath(f'performance_{seed}.png')
        plt.savefig(fig_file)
        plt.show()
  
    
    def _performance_discrete(self):
                
        fig, axes = plt.subplots(1, 6, figsize=(18, 2.5))

        # get mesh data and prediction
        X_mesh, u_mesh, t_ticks, x_ticks = self.data.mesh()
        u_pred = self.model(X_mesh).numpy()
        
        u_t, u_xx, source_term, F_res, F_res_weighted = self.model._F_residuals(tf.convert_to_tensor(X_mesh))
        # reshaping
        u_mesh = u_mesh.reshape(len(x_ticks), len(t_ticks))
        u_pred = u_pred.reshape(len(x_ticks), len(t_ticks))
        u_res = np.abs(u_pred-u_mesh).reshape(len(x_ticks), len(t_ticks))
        F_res = np.abs(F_res).reshape(len(x_ticks), len(t_ticks))
        F_res_weighted = np.abs(F_res_weighted).reshape(len(x_ticks), len(t_ticks)) 

        cont = axes[0].contourf(t_ticks, x_ticks, u_mesh, levels=100, cmap='coolwarm', vmin=0, vmax=1)
        fig.colorbar(cont, ax=axes[0])
        
        cont = axes[1].contourf(t_ticks, x_ticks, u_pred, levels=100, cmap='coolwarm', vmin=0, vmax=1)
        fig.colorbar(cont, ax=axes[1])
        
        
        tri = axes[2].contourf(t_ticks, x_ticks, u_res, levels=100, cmap='jet')
        cbar = fig.colorbar(tri, ax=axes[2])
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_offset_position('left')
    
        tri = axes[3].contourf(t_ticks, x_ticks, F_res, levels=100, cmap='jet')
        cbar = fig.colorbar(tri, ax=axes[3])
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_offset_position('left')
        
        tri = axes[4].contourf(t_ticks, x_ticks, F_res_weighted, levels=100, cmap='jet')
        cbar = fig.colorbar(tri, ax=axes[4])
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_offset_position('left')

        for t in [0, 0.0040]:
            # time lines
            axes[0].axvline(t, ls='--', lw=1, c='white')
            axes[1].axvline(t, ls='--', lw=1, c='white')
            axes[2].axvline(t, ls='--', lw=1, c='white')
            # Cut data and prediction
            X_cut, u_cut = self.data.cut(t)
            u_pred = self.model(X_cut)
            axes[5].plot(X_cut[:,1], u_cut, ls='--', lw=1, c='black')
            axes[5].plot(X_cut[:,1], u_pred, lw=1, c='red')
            axes[5].grid(ls='--')
                
              

        # dummy legend()
        axes[5].plot([], [], ls='--', lw=1, c='black', label='Ref.')
        axes[5].plot([], [], lw=1, c='red', label='Pred.')
        axes[5].legend(frameon=False)
        axes[5].set_xlim(self.data.x_min, self.data.x_max)
        axes[5].set_ylim(0, 1)
        
        # axis label
        axes[0].set_title('Reference')
        axes[1].set_title('Prediction')
        axes[2].set_title('Data Residuals')
        axes[3].set_title('Physics Residuals')
        axes[4].set_title('Physics Residuals (weighted)')
        axes[5].set_title(r'$x$-cuts')

        seed = self.model.seed
        plt.tight_layout()
        fig_file = self.model.model_path.joinpath(f'performance_{seed}.png')
        plt.savefig(fig_file)
        plt.show()