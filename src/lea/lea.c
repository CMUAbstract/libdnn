#include "lea.h"

#include <msp430.h>
#include <libio/console.h>
#include <libmspdriver/driverlib.h>
#include <libdsp/DSPLib.h>
#include <libalpaca/alpaca.h>

#include "blas.h"
#include "mem.h"
#include "state.h"
#include "misc.h"

void task_calibrate();
TASK(TASK_UID_BLAS_OFFSET + 12, task_calibrate);

static __fram bool DMA_initialized = false;
static __fram uint16_t tile_size = 0;

uint16_t check_calibrate(void){
	if(!DMA_initialized) {
		PRINTF("\r\n Initializing DMA");
		DMA_disableTransferDuringReadModifyWrite();
		for(uint16_t i = 0; i < 3; i++) {
			dmaConfig[i].channelSelect = i << 4;
			dmaConfig[i].transferModeSelect = DMA_TRANSFER_BLOCK;
			dmaConfig[i].transferUnitSelect = DMA_SIZE_SRCWORD_DSTWORD;
			DMA_init(&dmaConfig[i]);
			DMA_enableInterrupt(dmaConfig[i].channelSelect);
		}
		DMA_initialized = 1;
	}
	if(tile_size != 0) return tile_size;
	TASK_REF(task_calibrate)->info.return_task = CUR_TASK;
	TRANSITION_TO(task_calibrate);
	return 0;
}

uint16_t greatest_tile_size(uint16_t dim, uint16_t max) {
	if(dim < max) return dim;

	uint16_t i = 2;
	uint16_t max_divisor = i;
	while(i <= max && i <= dim) {
        if(dim % i == 0) max_divisor = i;
    	i += 2;
    }
    return max_divisor;
}

// Shut off CPU and do a DMA transfer
void DMA_startSleepTransfer(uint16_t channel) {
	uint16_t interruptState = __get_interrupt_state();
    __disable_interrupt();
   	DMA_startTransfer(channel);
	__bis_SR_register(GIE + LPM0_bits); 
	__set_interrupt_state(interruptState);	
}

void task_calibrate() {
	if(CUR_SCRATCH[0] == 0) {
		CUR_SCRATCH[1] = CONFIG_TILE_SIZE;
		CUR_SCRATCH[0] = 1;
#ifdef CONFIG_INTERMITTENT
		P1OUT = 0x01;
		P1DIR = 0x01;
		while(1) {}
#else
		transition_to(CUR_TASK);
#endif
	} 
	if(CUR_SCRATCH[0] == 3) {
		CUR_SCRATCH[0] = 1;
#ifdef CONFIG_INTERMITTENT
		P1OUT = 0x01;
		P1DIR = 0x01;
		while(1) {}
#else
		transition_to(CUR_TASK);
#endif
	} else if(CUR_SCRATCH[0] == 1 && tile_size == 0) {
		CUR_SCRATCH[0] = 2;
		msp_mac_q15_params params = {.length = CUR_SCRATCH[1]};
		msp_status status;
		status = msp_mac_q15(&params, tsrc1, tsrc2, tdest);
		PRINTF("\r\n Done init: status: %u tile_size %u", status, CUR_SCRATCH[1]);
		msp_checkStatus(status);
		write_to_gbuf((uint8_t *)(CUR_SCRATCH + 1),
			(uint8_t *)(&tile_size), sizeof(uint16_t));
		transition_to(CUR_TASK);
	} else if(CUR_SCRATCH[0] == 2 && tile_size == 0) {
		scratch_bak[0] = 3;
		scratch_bak[1] = CUR_SCRATCH[1] >> 1;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1),
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);	
	}
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup);
}

void __attribute__((interrupt(DMA_VECTOR)))dma_isr_handler(void) {
    switch (__even_in_range(DMAIV, DMAIV_DMA2IFG)) {
    	case DMAIV_DMA0IFG:
        case DMAIV_DMA1IFG:
        case DMAIV_DMA2IFG:
            break;
        default: 
          break;
   }
   __bic_SR_register_on_exit(LPM0_bits);
}

#ifndef MSP_DISABLE_LEA
void __attribute__ ((interrupt(LEA_VECTOR)))msp_lea_isr(void) {
    uint16_t flags = LEAIFG;
    LEAIFG |= flags;
    msp_lea_ifg = flags;
    __bic_SR_register_on_exit(LPM0_bits);
}
#endif
