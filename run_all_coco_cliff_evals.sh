#python run_coco.py --model_path /home/svdon/data/cliff/trained_models/swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_swapent_syslowcon/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_swapent_syslowcon/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_swapent_syslowcon/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_swapent_syslowcon/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_swapent_syslowcon/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_swapent_syslowcon/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_swapent_syslowcon/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_swapent_syslowcon/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_cnndm_swapent_syslowcon/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_cnndm_swapent_syslowcon/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_cnndm_swapent_syslowcon/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_cnndm_swapent_syslowcon/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_cnndm_swapent_syslowcon/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_cnndm_swapent_syslowcon/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_cnndm_swapent_syslowcon/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_cnndm_swapent_syslowcon/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/mask_ent_maskrel_regenrel_regenent_swapent_syslowcon_2 --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_all/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_all/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/mask_ent_maskrel_regenrel_regenent_swapent_syslowcon_2 --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_all/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_all/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/mask_ent_maskrel_regenrel_regenent_swapent_syslowcon_2 --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_all/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_all/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/mask_ent_maskrel_regenrel_regenent_swapent_syslowcon_2 --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_all/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_all/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/mask_ent_maskrel_regenrel_regenent_swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_cnndm_all/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_cnndm_all/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/mask_ent_maskrel_regenrel_regenent_swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_cnndm_all/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_cnndm_all/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/mask_ent_maskrel_regenrel_regenent_swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_cnndm_all/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_cnndm_all/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/mask_ent_maskrel_regenrel_regenent_swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_cnndm_all/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_cnndm_all/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/maskrel_regenrel_regenent_swapent_syslowcon_2 --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_all_but_maskent/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_all_but_maskent/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/maskrel_regenrel_regenent_swapent_syslowcon_2 --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/cliff_xsum_all_but_maskent/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/cliff_xsum_all_but_maskent/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/maskrel_regenrel_regenent_swapent_syslowcon_2 --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_all_but_maskent/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_all_but_maskent/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/maskrel_regenrel_regenent_swapent_syslowcon_2 --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/cliff_xsum_all_but_maskent/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/cliff_xsum_all_but_maskent/span_coco_score.txt --mask span \
#python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/fc_plus_swapent_syslowcon_xsum/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/fc_plus_swapent_syslowcon_xsum/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/fc_plus_swapent_syslowcon_xsum/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/fc_plus_swapent_syslowcon_xsum/span_coco_score.txt --mask span \
#python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum/span_coco_score.txt --mask span \
python run_coco.py --model_path /external1/svdon/coursepaper/checkpoints/fc_plus_swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /external1/svdon/coursepaper/coursepaper_dataset/cliff/data/cnndm_raw/test.source --summaries_path /external1/svdon/coursepaper/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum/formatted-test.txt --bin_dir /external1/svdon/coursepaper/coursepaper_dataset/cliff/data/cnndm_binarized --output_file /external1/svdon/coursepaper/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum/token_coco_score.txt --mask token \
 && python run_coco.py --model_path /external1/svdon/coursepaper/checkpoints/fc_plus_swapent_syslowcon_xsum --model_name checkpoint_best.pt --source_path /external1/svdon/coursepaper/coursepaper_dataset/cliff/data/cnndm_raw/test.source --summaries_path /external1/svdon/coursepaper/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum/formatted-test.txt --bin_dir /external1/svdon/coursepaper/coursepaper_dataset/cliff/data/cnndm_binarized --output_file /external1/svdon/coursepaper/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum_cnndm/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum_cnndm/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_swapent_syslowcon_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum_cnndm/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/fc_plus_swapent_syslowcon_xsum_cnndm/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_all_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/fc_plus_all_xsum/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/fc_plus_all_xsum/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_all_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/fc_plus_all_xsum/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/fc_plus_all_xsum/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_all_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/fc_plus_all_xsum/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/fc_plus_all_xsum/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_all_xsum --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/fc_plus_all_xsum/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/fc_plus_all_xsum/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_all_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/fc_plus_all_xsum_cnndm/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/fc_plus_all_xsum_cnndm/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_all_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/xsum_raw/test.source --summaries_path /home/svdon/data/bart_predicts/xsum/fc_plus_all_xsum_cnndm/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/xsum_binarized --output_file /home/svdon/data/bart_predicts/xsum/fc_plus_all_xsum_cnndm/span_coco_score.txt --mask span \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_all_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/fc_plus_all_xsum_cnndm/formatted-test.txt --bin_dcd ir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/fc_plus_all_xsum_cnndm/token_coco_score.txt --mask token \
# && python run_coco.py --model_path /home/svdon/data/cliff/trained_models/fc_plus_all_xsum_cnndm --model_name checkpoint_best.pt --source_path /home/svdon/data/cliff/data/cnndm_raw/test.source --summaries_path /home/svdon/data/bart_predicts/cnndm/fc_plus_all_xsum_cnndm/formatted-test.txt --bin_dir /home/svdon/data/cliff/data/cnndm_binarized --output_file /home/svdon/data/bart_predicts/cnndm/fc_plus_all_xsum_cnndm/span_coco_score.txt --mask span