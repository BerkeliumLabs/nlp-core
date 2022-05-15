import chalk from 'chalk';
import { Spinner } from 'cli-spinner'
import cliSpinners from 'cli-spinners';


export const bkSpinner = {
    spinnerAnimation: () => {
        const spinner = new Spinner(chalk.magentaBright(` Data Encoding... %s`));
        let spinnerString = '';
        cliSpinners.dots.frames.map((spinText)=>{
            spinnerString += spinText;
        });
        spinner.setSpinnerString(spinnerString);
        spinner.setSpinnerDelay(cliSpinners.dots.interval);
        spinner.start();
        //console.log(spinnerString);
        return spinner;
    },
    stopSpinner: (spinner) => {
        spinner.stop();
    }
    /* loadingAnimation: (() => {
        const h = cliSpinners.dots8Bit.frames;
        let i = 0;

        return setInterval(() => {
            i = (i > cliSpinners.dots8Bit.frames.length) ? 0 : i;
            console.clear();
            console.log(h[i], 'Encording Data...');
            i++;
        }, cliSpinners.dots8Bit.interval);
    })() */
}