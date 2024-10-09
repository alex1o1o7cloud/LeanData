import Mathlib

namespace peach_bun_weight_l2223_222342

theorem peach_bun_weight (O triangle : ℕ) 
  (h1 : O = 2 * triangle + 40) 
  (h2 : O + 80 = triangle + 200) : 
  O + triangle = 280 := 
by 
  sorry

end peach_bun_weight_l2223_222342


namespace correct_addition_by_changing_digit_l2223_222338

theorem correct_addition_by_changing_digit :
  ∃ (d : ℕ), (d < 10) ∧ (d = 4) ∧
  (374 + (500 + d) + 286 = 1229 - 50) :=
by
  sorry

end correct_addition_by_changing_digit_l2223_222338


namespace square_difference_l2223_222327

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 6) : (x - y)^2 = 57 :=
by
  sorry

end square_difference_l2223_222327


namespace petya_oranges_l2223_222390

theorem petya_oranges (m o : ℕ) (h1 : m + 6 * m + o = 20) (h2 : 6 * m > o) : o = 6 :=
by 
  sorry

end petya_oranges_l2223_222390


namespace mango_distribution_l2223_222364

theorem mango_distribution (friends : ℕ) (initial_mangos : ℕ) 
    (share_left : ℕ) (share_right : ℕ) 
    (eat_mango : ℕ) (pass_mango_right : ℕ)
    (H1 : friends = 100) 
    (H2 : initial_mangos = 2019)
    (H3 : share_left = 2) 
    (H4 : share_right = 1) 
    (H5 : eat_mango = 1) 
    (H6 : pass_mango_right = 1) :
    ∃ final_count, final_count = 8 :=
by
  -- Proof is omitted.
  sorry

end mango_distribution_l2223_222364


namespace proof_fraction_l2223_222384

def find_fraction (x : ℝ) : Prop :=
  (2 / 9) * x = 10 → (2 / 5) * x = 18

-- Optional, you can define x based on the condition:
noncomputable def certain_number : ℝ := 10 * (9 / 2)

theorem proof_fraction :
  find_fraction certain_number :=
by
  intro h
  sorry

end proof_fraction_l2223_222384


namespace sum_of_ages_is_correct_l2223_222346

-- Define the present ages of A, B, and C
def present_age_A : ℕ := 11

-- Define the ratio conditions from 3 years ago
def three_years_ago_ratio (A B C : ℕ) : Prop :=
  B - 3 = 2 * (A - 3) ∧ C - 3 = 3 * (A - 3)

-- The statement we want to prove
theorem sum_of_ages_is_correct {A B C : ℕ} (hA : A = 11)
  (h_ratio : three_years_ago_ratio A B C) :
  A + B + C = 57 :=
by
  -- The proof part will be handled here
  sorry

end sum_of_ages_is_correct_l2223_222346


namespace time_jogging_l2223_222379

def distance := 25     -- Distance jogged (in kilometers)
def speed := 5        -- Speed (in kilometers per hour)

theorem time_jogging :
  (distance / speed) = 5 := 
by
  sorry

end time_jogging_l2223_222379


namespace lcm_condition_l2223_222373

theorem lcm_condition (m : ℕ) (h_m_pos : m > 0) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 36 :=
by
  sorry

end lcm_condition_l2223_222373


namespace original_team_size_l2223_222374

theorem original_team_size (n : ℕ) (W : ℕ) :
  (W = n * 94) →
  ((W + 110 + 60) / (n + 2) = 92) →
  n = 7 :=
by
  intro hW_avg hnew_avg
  -- The proof steps would go here
  sorry

end original_team_size_l2223_222374


namespace calculate_angle_l2223_222360

def degrees_to_seconds (d m s : ℕ) : ℕ :=
  d * 3600 + m * 60 + s

def seconds_to_degrees (s : ℕ) : (ℕ × ℕ × ℕ) :=
  (s / 3600, (s % 3600) / 60, s % 60)

theorem calculate_angle : 
  (let d1 := 50
   let m1 := 24
   let angle1_sec := degrees_to_seconds d1 m1 0
   let angle1_sec_tripled := 3 * angle1_sec
   let (d1', m1', s1') := seconds_to_degrees angle1_sec_tripled

   let d2 := 98
   let m2 := 12
   let s2 := 25
   let angle2_sec := degrees_to_seconds d2 m2 s2
   let angle2_sec_divided := angle2_sec / 5
   let (d2', m2', s2') := seconds_to_degrees angle2_sec_divided

   let total_sec := degrees_to_seconds d1' m1' s1' + degrees_to_seconds d2' m2' s2'
   let (final_d, final_m, final_s) := seconds_to_degrees total_sec
   (final_d, final_m, final_s)) = (170, 50, 29) := by sorry

end calculate_angle_l2223_222360


namespace total_number_of_outfits_l2223_222340

noncomputable def number_of_outfits (shirts pants ties jackets : ℕ) :=
  shirts * pants * ties * jackets

theorem total_number_of_outfits :
  number_of_outfits 8 5 5 3 = 600 :=
by
  sorry

end total_number_of_outfits_l2223_222340


namespace pirate_treasure_chest_coins_l2223_222392

theorem pirate_treasure_chest_coins:
  ∀ (gold_coins silver_coins bronze_coins: ℕ) (chests: ℕ),
    gold_coins = 3500 →
    silver_coins = 500 →
    bronze_coins = 2 * silver_coins →
    chests = 5 →
    (gold_coins / chests + silver_coins / chests + bronze_coins / chests = 1000) :=
by
  intros gold_coins silver_coins bronze_coins chests gold_eq silv_eq bron_eq chest_eq
  sorry

end pirate_treasure_chest_coins_l2223_222392


namespace system_solution_l2223_222353

theorem system_solution (m n : ℝ) (h1 : -2 * m * 5 + 5 * 2 = 15) (h2 : 5 + 7 * n * 2 = 14) :
  ∃ (a b : ℝ), (-2 * m * (a + b) + 5 * (a - 2 * b) = 15) ∧ ((a + b) + 7 * n * (a - 2 * b) = 14) ∧ (a = 4) ∧ (b = 1) :=
by
  -- The proof is intentionally omitted
  sorry

end system_solution_l2223_222353


namespace man_has_2_nickels_l2223_222355

theorem man_has_2_nickels
  (d n : ℕ)
  (h1 : 10 * d + 5 * n = 70)
  (h2 : d + n = 8) :
  n = 2 := 
by
  -- omit the proof
  sorry

end man_has_2_nickels_l2223_222355


namespace circle_y_coords_sum_l2223_222302

theorem circle_y_coords_sum (x y : ℝ) (hc : (x + 3)^2 + (y - 5)^2 = 64) (hx : x = 0) : y = 5 + Real.sqrt 55 ∨ y = 5 - Real.sqrt 55 → (5 + Real.sqrt 55) + (5 - Real.sqrt 55) = 10 := 
by
  intros
  sorry

end circle_y_coords_sum_l2223_222302


namespace min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l2223_222339

noncomputable def min_value_expression (a b : ℝ) (hab : 2 * a + b = 1) : ℝ :=
  4 * a^2 + b^2 + 1 / (a * b)

theorem min_value_expression_geq_17_div_2 {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (hab: 2 * a + b = 1) :
  min_value_expression a b hab ≥ 17 / 2 :=
sorry

theorem min_value_expression_eq_17_div_2_for_specific_a_b :
  min_value_expression (1/3) (1/3) (by norm_num) = 17 / 2 :=
sorry

end min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l2223_222339


namespace short_sleeve_shirts_l2223_222352

theorem short_sleeve_shirts (total_shirts long_sleeve_shirts short_sleeve_shirts : ℕ) 
  (h1 : total_shirts = 9) 
  (h2 : long_sleeve_shirts = 5)
  (h3 : short_sleeve_shirts = total_shirts - long_sleeve_shirts) : 
  short_sleeve_shirts = 4 :=
by 
  sorry

end short_sleeve_shirts_l2223_222352


namespace ant_prob_bottom_vertex_l2223_222304

theorem ant_prob_bottom_vertex :
  let top := 1
  let first_layer := 4
  let second_layer := 4
  let bottom := 1
  let prob_first_layer := 1 / first_layer
  let prob_second_layer := 1 / second_layer
  let prob_bottom := 1 / (second_layer + bottom)
  prob_first_layer * prob_second_layer * prob_bottom = 1 / 80 :=
by
  sorry

end ant_prob_bottom_vertex_l2223_222304


namespace score_stability_l2223_222301

theorem score_stability (mean_A mean_B : ℝ) (h_mean_eq : mean_A = mean_B)
  (variance_A variance_B : ℝ) (h_variance_A : variance_A = 0.06) (h_variance_B : variance_B = 0.35) :
  variance_A < variance_B :=
by
  -- Theorem statement and conditions sufficient to build successfully
  sorry

end score_stability_l2223_222301


namespace neg_exists_le_eq_forall_gt_l2223_222377

open Classical

variable {n : ℕ}

theorem neg_exists_le_eq_forall_gt :
  (¬ ∃ (n : ℕ), n > 0 ∧ 2^n ≤ 2 * n + 1) ↔
  (∀ (n : ℕ), n > 0 → 2^n > 2 * n + 1) :=
by 
  sorry

end neg_exists_le_eq_forall_gt_l2223_222377


namespace center_cell_value_l2223_222378

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l2223_222378


namespace convex_polygon_sides_eq_49_l2223_222359

theorem convex_polygon_sides_eq_49 
  (n : ℕ)
  (hn : n > 0) 
  (h : (n * (n - 3)) / 2 = 23 * n) : n = 49 :=
sorry

end convex_polygon_sides_eq_49_l2223_222359


namespace find_ordered_triples_l2223_222351

-- Define the problem conditions using Lean structures.
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_ordered_triples (a b c : ℕ) :
  (is_perfect_square (a^2 + 2 * b + c) ∧
   is_perfect_square (b^2 + 2 * c + a) ∧
   is_perfect_square (c^2 + 2 * a + b))
  ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 43 ∧ b = 127 ∧ c = 106) :=
by sorry

end find_ordered_triples_l2223_222351


namespace john_tv_show_duration_l2223_222376

def john_tv_show (seasons_before : ℕ) (episodes_per_season : ℕ) (additional_episodes : ℕ) (episode_duration : ℝ) : ℝ :=
  let total_episodes_before := seasons_before * episodes_per_season
  let last_season_episodes := episodes_per_season + additional_episodes
  let total_episodes := total_episodes_before + last_season_episodes
  total_episodes * episode_duration

theorem john_tv_show_duration :
  john_tv_show 9 22 4 0.5 = 112 := 
by
  sorry

end john_tv_show_duration_l2223_222376


namespace cartons_per_box_l2223_222350

open Nat

theorem cartons_per_box (cartons packs sticks brown_boxes total_sticks : ℕ) 
  (h1 : cartons * (packs * sticks) * brown_boxes = total_sticks) 
  (h2 : packs = 5) 
  (h3 : sticks = 3) 
  (h4 : brown_boxes = 8) 
  (h5 : total_sticks = 480) :
  cartons = 4 := 
by 
  sorry

end cartons_per_box_l2223_222350


namespace geometric_sequence_problem_l2223_222308

variable (a : ℕ → ℝ)
variable (r : ℝ) (hpos : ∀ n, 0 < a n)

theorem geometric_sequence_problem
  (hgeom : ∀ n, a (n+1) = a n * r)
  (h_eq : a 1 * a 3 + 2 * a 3 * a 5 + a 5 * a 7 = 4) :
  a 2 + a 6 = 2 :=
sorry

end geometric_sequence_problem_l2223_222308


namespace arithmetic_series_sum_l2223_222357

theorem arithmetic_series_sum : 
  let a := -41
  let d := 2
  let n := 22
  let l := 1
  let Sn := n * (a + l) / 2
  a = -41 ∧ d = 2 ∧ l = 1 ∧ n = 22 → Sn = -440 :=
by 
  intros a d n l Sn h
  sorry

end arithmetic_series_sum_l2223_222357


namespace total_weight_AlF3_10_moles_l2223_222347

noncomputable def molecular_weight_AlF3 (atomic_weight_Al: ℝ) (atomic_weight_F: ℝ) : ℝ :=
  atomic_weight_Al + 3 * atomic_weight_F

theorem total_weight_AlF3_10_moles :
  let atomic_weight_Al := 26.98
  let atomic_weight_F := 19.00
  let num_moles := 10
  molecular_weight_AlF3 atomic_weight_Al atomic_weight_F * num_moles = 839.8 :=
by
  sorry

end total_weight_AlF3_10_moles_l2223_222347


namespace chess_tournament_total_players_l2223_222336

theorem chess_tournament_total_players :
  ∃ (n: ℕ), 
    (∀ (players: ℕ) (points: ℕ -> ℕ), 
      (players = n + 15) ∧
      (∀ p, points p = points p / 2 + points p / 2) ∧
      (∀ i < 15, ∀ j < 15, points i = points j / 2) → 
      players = 36) :=
by
  sorry

end chess_tournament_total_players_l2223_222336


namespace find_a_if_lines_parallel_l2223_222349

theorem find_a_if_lines_parallel (a : ℝ) (h1 : ∃ y : ℝ, y = - (a / 4) * (1 : ℝ) + (1 / 4)) (h2 : ∃ y : ℝ, y = - (1 / a) * (1 : ℝ) + (1 / (2 * a))) : a = -2 :=
sorry

end find_a_if_lines_parallel_l2223_222349


namespace calculate_fraction_l2223_222343

theorem calculate_fraction :
  (2019 + 1981)^2 / 121 = 132231 := 
  sorry

end calculate_fraction_l2223_222343


namespace cells_at_end_of_12th_day_l2223_222337

def initial_organisms : ℕ := 8
def initial_cells_per_organism : ℕ := 4
def total_initial_cells : ℕ := initial_organisms * initial_cells_per_organism
def division_period_days : ℕ := 3
def total_duration_days : ℕ := 12
def complete_periods : ℕ := total_duration_days / division_period_days
def common_ratio : ℕ := 3

theorem cells_at_end_of_12th_day :
  total_initial_cells * common_ratio^(complete_periods - 1) = 864 := by
  sorry

end cells_at_end_of_12th_day_l2223_222337


namespace find_square_l2223_222397

theorem find_square (q x : ℝ) 
  (h1 : x + q = 74) 
  (h2 : x + 2 * q^2 = 180) : 
  x = 66 :=
by {
  sorry
}

end find_square_l2223_222397


namespace solution_x_l2223_222325

noncomputable def find_x (x : ℝ) : Prop :=
  (Real.log (x^4))^2 = (Real.log x)^6

theorem solution_x (x : ℝ) : find_x x ↔ (x = 1 ∨ x = Real.exp 2 ∨ x = Real.exp (-2)) :=
sorry

end solution_x_l2223_222325


namespace div64_by_expression_l2223_222307

theorem div64_by_expression {n : ℕ} (h : n > 0) : ∃ k : ℤ, (3^(2 * n + 2) - 8 * ↑n - 9) = 64 * k :=
by
  sorry

end div64_by_expression_l2223_222307


namespace combined_selling_price_correct_l2223_222330

def ArticleA_Cost : ℝ := 500
def ArticleA_Profit_Percent : ℝ := 0.45
def ArticleB_Cost : ℝ := 300
def ArticleB_Profit_Percent : ℝ := 0.30
def ArticleC_Cost : ℝ := 1000
def ArticleC_Profit_Percent : ℝ := 0.20
def Sales_Tax_Percent : ℝ := 0.12

def CombinedSellingPrice (A_cost A_profit_percent B_cost B_profit_percent C_cost C_profit_percent tax_percent : ℝ) : ℝ :=
  let A_selling_price := A_cost * (1 + A_profit_percent)
  let A_final_price := A_selling_price * (1 + tax_percent)
  let B_selling_price := B_cost * (1 + B_profit_percent)
  let B_final_price := B_selling_price * (1 + tax_percent)
  let C_selling_price := C_cost * (1 + C_profit_percent)
  let C_final_price := C_selling_price * (1 + tax_percent)
  A_final_price + B_final_price + C_final_price

theorem combined_selling_price_correct :
  CombinedSellingPrice ArticleA_Cost ArticleA_Profit_Percent ArticleB_Cost ArticleB_Profit_Percent ArticleC_Cost ArticleC_Profit_Percent Sales_Tax_Percent = 2592.8 := by
  sorry

end combined_selling_price_correct_l2223_222330


namespace subcommittees_with_at_least_one_teacher_l2223_222391

-- Define the total number of members and the count of teachers
def total_members : ℕ := 12
def teacher_count : ℕ := 5
def subcommittee_size : ℕ := 5

-- Define binomial coefficient calculation
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Problem statement: number of five-person subcommittees with at least one teacher
theorem subcommittees_with_at_least_one_teacher :
  binom total_members subcommittee_size - binom (total_members - teacher_count) subcommittee_size = 771 := by
  sorry

end subcommittees_with_at_least_one_teacher_l2223_222391


namespace mixture_weight_l2223_222316

def almonds := 116.67
def walnuts := almonds / 5
def total_weight := almonds + walnuts

theorem mixture_weight : total_weight = 140.004 := by
  sorry

end mixture_weight_l2223_222316


namespace jose_age_is_26_l2223_222322

def Maria_age : ℕ := 14
def Jose_age (m : ℕ) : ℕ := m + 12

theorem jose_age_is_26 (m j : ℕ) (h1 : j = m + 12) (h2 : m + j = 40) : j = 26 :=
by
  sorry

end jose_age_is_26_l2223_222322


namespace inequality_may_not_hold_l2223_222383

theorem inequality_may_not_hold (m n : ℝ) (h : m > n) : ¬ (m^2 > n^2) :=
by
  -- Leaving the proof out according to the instructions.
  sorry

end inequality_may_not_hold_l2223_222383


namespace neg_product_B_l2223_222395

def expr_A := (-1 / 3) * (1 / 4) * (-6)
def expr_B := (-9) * (1 / 8) * (-4 / 7) * 7 * (-1 / 3)
def expr_C := (-3) * (-1 / 2) * 7 * 0
def expr_D := (-1 / 5) * 6 * (-2 / 3) * (-5) * (-1 / 2)

theorem neg_product_B :
  expr_B < 0 :=
by
  sorry

end neg_product_B_l2223_222395


namespace correct_calculation_result_l2223_222369

theorem correct_calculation_result (n : ℤ) (h1 : n - 59 = 43) : n - 46 = 56 :=
by {
  sorry -- Proof is omitted
}

end correct_calculation_result_l2223_222369


namespace no_common_points_l2223_222363

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x
noncomputable def g (a b x : ℝ) : ℝ := b + a * Real.log (x - 1)
noncomputable def h (a x : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)
noncomputable def G (a : ℝ) : ℝ := -a^2 / 4 + 1 - a * Real.log (a / 2)

theorem no_common_points (a b : ℝ) (h1 : 1 ≤ a) :
  (∀ x > 1, f a x ≠ g a b x) ↔ b < 3 / 4 + Real.log 2 :=
by
  sorry

end no_common_points_l2223_222363


namespace present_age_of_B_l2223_222332

theorem present_age_of_B
  (A B : ℕ)
  (h1 : A = B + 5)
  (h2 : A + 30 = 2 * (B - 30)) :
  B = 95 :=
by { sorry }

end present_age_of_B_l2223_222332


namespace bike_trike_race_l2223_222386

theorem bike_trike_race (P : ℕ) (B T : ℕ) (h1 : B = (3 * P) / 5) (h2 : T = (2 * P) / 5) (h3 : 2 * B + 3 * T = 96) :
  P = 40 :=
by
  sorry

end bike_trike_race_l2223_222386


namespace neg_p_equiv_l2223_222300

-- The proposition p
def p : Prop := ∀ x : ℝ, x^2 - 1 < 0

-- Equivalent Lean theorem statement
theorem neg_p_equiv : ¬ p ↔ ∃ x₀ : ℝ, x₀^2 - 1 ≥ 0 :=
by
  sorry

end neg_p_equiv_l2223_222300


namespace change_in_expression_l2223_222333

theorem change_in_expression (x a : ℝ) (ha : 0 < a) :
  (x^3 - 3*x + 1) + (3*a*x^2 + 3*a^2*x + a^3 - 3*a) = (x + a)^3 - 3*(x + a) + 1 ∧
  (x^3 - 3*x + 1) + (-3*a*x^2 + 3*a^2*x - a^3 + 3*a) = (x - a)^3 - 3*(x - a) + 1 :=
by sorry

end change_in_expression_l2223_222333


namespace intersection_point_l2223_222385

-- Mathematical problem translated to Lean 4 statement

theorem intersection_point : 
  ∃ x y : ℝ, y = -3 * x + 1 ∧ y + 1 = 15 * x ∧ x = 1 / 9 ∧ y = 2 / 3 := 
by
  sorry

end intersection_point_l2223_222385


namespace statement_A_statement_C_statement_D_l2223_222344

variable (a : ℕ → ℝ) (A B : ℝ)

-- Condition: The sequence satisfies the recurrence relation
def recurrence_relation (n : ℕ) : Prop :=
  a (n + 2) = A * a (n + 1) + B * a n

-- Statement A: A=1 and B=-1 imply periodic with period 6
theorem statement_A (h : ∀ n, recurrence_relation a 1 (-1) n) :
  ∀ n, a (n + 6) = a n := 
sorry

-- Statement C: A=3 and B=-2 imply the derived sequence is geometric
theorem statement_C (h : ∀ n, recurrence_relation a 3 (-2) n) :
  ∃ r : ℝ, ∀ n, a (n + 1) - a n = r * (a n - a (n - 1)) :=
sorry

-- Statement D: A+1=B, a1=0, a2=B imply {a_{2n}} is increasing
theorem statement_D (hA : ∀ n, recurrence_relation a A (A + 1) n)
  (h1 : a 1 = 0) (h2 : a 2 = A + 1) :
  ∀ n, a (2 * (n + 1)) > a (2 * n) :=
sorry

end statement_A_statement_C_statement_D_l2223_222344


namespace circle_symmetry_line_l2223_222335

theorem circle_symmetry_line :
  ∃ l: ℝ → ℝ → Prop, 
    (∀ x y, l x y → x - y + 2 = 0) ∧ 
    (∀ x y, l x y ↔ (x + 2)^2 + (y - 2)^2 = 4) :=
sorry

end circle_symmetry_line_l2223_222335


namespace income_second_day_l2223_222382

theorem income_second_day (x : ℕ) 
  (h_condition : (200 + x + 750 + 400 + 500) / 5 = 400) : x = 150 :=
by 
  -- Proof omitted.
  sorry

end income_second_day_l2223_222382


namespace three_pow_sub_cube_eq_two_l2223_222393

theorem three_pow_sub_cube_eq_two (k : ℕ) (h : 30^k ∣ 929260) : 3^k - k^3 = 2 := 
sorry

end three_pow_sub_cube_eq_two_l2223_222393


namespace ants_meeting_points_l2223_222329

/-- Definition for the problem setup: two ants running at constant speeds around a circle. -/
structure AntsRunningCircle where
  laps_ant1 : ℕ
  laps_ant2 : ℕ

/-- Theorem stating that given the laps completed by two ants in opposite directions on a circle, 
    they will meet at a specific number of distinct points. -/
theorem ants_meeting_points 
  (ants : AntsRunningCircle)
  (h1 : ants.laps_ant1 = 9)
  (h2 : ants.laps_ant2 = 6) : 
    ∃ n : ℕ, n = 5 := 
by
  -- Proof goes here
  sorry

end ants_meeting_points_l2223_222329


namespace gcd_polynomials_l2223_222310

theorem gcd_polynomials (b : ℕ) (hb : 2160 ∣ b) : 
  Nat.gcd (b ^ 2 + 9 * b + 30) (b + 6) = 12 := 
  sorry

end gcd_polynomials_l2223_222310


namespace turtle_finishes_in_10_minutes_l2223_222323

def skunk_time : ℕ := 6
def rabbit_speed_ratio : ℕ := 3
def turtle_speed_ratio : ℕ := 5
def rabbit_time := skunk_time / rabbit_speed_ratio
def turtle_time := turtle_speed_ratio * rabbit_time

theorem turtle_finishes_in_10_minutes : turtle_time = 10 := by
  sorry

end turtle_finishes_in_10_minutes_l2223_222323


namespace midpoint_coordinates_l2223_222331

theorem midpoint_coordinates :
  let x1 := 2
  let y1 := -3
  let z1 := 5
  let x2 := 8
  let y2 := 3
  let z2 := -1
  ( (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2 ) = (5, 0, 2) :=
by
  sorry

end midpoint_coordinates_l2223_222331


namespace count_integers_between_cubes_l2223_222361

noncomputable def a := (10.1)^3
noncomputable def b := (10.4)^3

theorem count_integers_between_cubes : 
  ∃ (count : ℕ), count = 94 ∧ (1030.031 < a) ∧ (a < b) ∧ (b < 1124.864) := 
  sorry

end count_integers_between_cubes_l2223_222361


namespace trigonometric_identity_l2223_222372

-- Define the conditions and the target statement
theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l2223_222372


namespace shaded_area_of_four_circles_l2223_222368

open Real

noncomputable def area_shaded_region (r : ℝ) (num_circles : ℕ) : ℝ :=
  let area_quarter_circle := (π * r^2) / 4
  let area_triangle := (r * r) / 2
  let area_one_checkered_region := area_quarter_circle - area_triangle
  let num_checkered_regions := num_circles * 2
  num_checkered_regions * area_one_checkered_region

theorem shaded_area_of_four_circles : area_shaded_region 5 4 = 50 * (π - 2) :=
by
  sorry

end shaded_area_of_four_circles_l2223_222368


namespace fraction_of_fifth_set_l2223_222370

theorem fraction_of_fifth_set :
  let total_match_duration := 11 * 60 + 5
  let fifth_set_duration := 8 * 60 + 11
  (fifth_set_duration : ℚ) / total_match_duration = 3 / 4 := 
sorry

end fraction_of_fifth_set_l2223_222370


namespace construction_company_sand_weight_l2223_222314

theorem construction_company_sand_weight :
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  sand_weight = 8.11 :=
by
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  -- Observing that 14.02 - 5.91 = 8.11
  have h : sand_weight = 8.11 := by sorry
  exact h

end construction_company_sand_weight_l2223_222314


namespace total_races_needed_to_determine_champion_l2223_222317

-- Defining the initial conditions
def num_sprinters : ℕ := 256
def lanes : ℕ := 8
def sprinters_per_race := lanes
def eliminated_per_race := sprinters_per_race - 1

-- The statement to be proved: The number of races required to determine the champion
theorem total_races_needed_to_determine_champion :
  ∃ (races : ℕ), races = 37 ∧
  ∀ s : ℕ, s = num_sprinters → 
  ∀ l : ℕ, l = lanes → 
  ∃ e : ℕ, e = eliminated_per_race →
  s - (races * e) = 1 :=
by sorry

end total_races_needed_to_determine_champion_l2223_222317


namespace monthly_rent_is_3600_rs_l2223_222326

def shop_length_feet : ℕ := 20
def shop_width_feet : ℕ := 15
def annual_rent_per_square_foot_rs : ℕ := 144

theorem monthly_rent_is_3600_rs :
  (shop_length_feet * shop_width_feet) * annual_rent_per_square_foot_rs / 12 = 3600 :=
by sorry

end monthly_rent_is_3600_rs_l2223_222326


namespace cos6_plus_sin6_equal_19_div_64_l2223_222321

noncomputable def cos6_plus_sin6 (θ : ℝ) : ℝ :=
  (Real.cos θ) ^ 6 + (Real.sin θ) ^ 6

theorem cos6_plus_sin6_equal_19_div_64 (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) :
  cos6_plus_sin6 θ = 19 / 64 := by
  sorry

end cos6_plus_sin6_equal_19_div_64_l2223_222321


namespace functional_solutions_l2223_222371

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x * f y + y * f x = (x + y) * (f x) * (f y)

theorem functional_solutions (f : ℝ → ℝ) (h : functional_equation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∃ (a : ℝ), ∀ x : ℝ, (x ≠ 0 → f x = 1) ∧ (x = 0 → f x = a)) :=
  sorry

end functional_solutions_l2223_222371


namespace blood_pressure_systolic_diastolic_l2223_222319

noncomputable def blood_pressure (t : ℝ) : ℝ :=
110 + 25 * Real.sin (160 * t)

theorem blood_pressure_systolic_diastolic :
  (∀ t : ℝ, blood_pressure t ≤ 135) ∧ (∀ t : ℝ, blood_pressure t ≥ 85) :=
by
  sorry

end blood_pressure_systolic_diastolic_l2223_222319


namespace smallest_positive_value_floor_l2223_222354

noncomputable def g (x : ℝ) : ℝ := Real.cos x - Real.sin x + 4 * Real.tan x

theorem smallest_positive_value_floor :
  ∃ s > 0, g s = 0 ∧ ⌊s⌋ = 3 :=
sorry

end smallest_positive_value_floor_l2223_222354


namespace stable_set_even_subset_count_l2223_222318

open Finset

-- Definitions
def is_stable (S : Finset (ℕ × ℕ)) : Prop :=
  ∀ ⦃x y⦄, (x, y) ∈ S → ∀ x' y', x' ≤ x → y' ≤ y → (x', y') ∈ S

-- Main statement
theorem stable_set_even_subset_count (S : Finset (ℕ × ℕ)) (hS : is_stable S):
  (∃ E O : ℕ, E ≥ O ∧ E + O = 2 ^ (S.card)) :=
  sorry

end stable_set_even_subset_count_l2223_222318


namespace candy_ratio_l2223_222315

theorem candy_ratio
  (red_candies : ℕ)
  (yellow_candies : ℕ)
  (blue_candies : ℕ)
  (total_candies : ℕ)
  (remaining_candies : ℕ)
  (h1 : red_candies = 40)
  (h2 : yellow_candies = 3 * red_candies - 20)
  (h3 : remaining_candies = 90)
  (h4 : total_candies = remaining_candies + yellow_candies)
  (h5 : blue_candies = total_candies - red_candies - yellow_candies) :
  blue_candies / yellow_candies = 1 / 2 :=
sorry

end candy_ratio_l2223_222315


namespace field_ratio_l2223_222358

theorem field_ratio (l w : ℕ) (h_l : l = 20) (pond_side : ℕ) (h_pond_side : pond_side = 5)
  (h_area_pond : pond_side * pond_side = (1 / 8 : ℚ) * l * w) : l / w = 2 :=
by 
  sorry

end field_ratio_l2223_222358


namespace arrange_numbers_l2223_222341

variable {a : ℝ}

theorem arrange_numbers (h1 : -1 < a) (h2 : a < 0) : (1 / a < a) ∧ (a < a ^ 2) ∧ (a ^ 2 < |a|) :=
by 
  sorry

end arrange_numbers_l2223_222341


namespace ratio_volumes_equal_ratio_areas_l2223_222309

-- Defining necessary variables and functions
variables (R : ℝ) (S_sphere S_cone V_sphere V_cone : ℝ)

-- Conditions
def surface_area_sphere : Prop := S_sphere = 4 * Real.pi * R^2
def volume_sphere : Prop := V_sphere = (4 / 3) * Real.pi * R^3
def volume_polyhedron : Prop := V_cone = (S_cone * R) / 3

-- Theorem statement
theorem ratio_volumes_equal_ratio_areas
  (h1 : surface_area_sphere R S_sphere)
  (h2 : volume_sphere R V_sphere)
  (h3 : volume_polyhedron R S_cone V_cone)
  : (V_sphere / V_cone) = (S_sphere / S_cone) :=
sorry

end ratio_volumes_equal_ratio_areas_l2223_222309


namespace inequality_proof_l2223_222328

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a ^ 3 / (a ^ 2 + a * b + b ^ 2)) + (b ^ 3 / (b ^ 2 + b * c + c ^ 2)) + (c ^ 3 / (c ^ 2 + c * a + a ^ 2)) ≥ (a + b + c) / 3 :=
by
  sorry

end inequality_proof_l2223_222328


namespace total_albums_l2223_222313

-- Definitions based on given conditions
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- The final statement to be proved
theorem total_albums : adele_albums + bridget_albums + katrina_albums + miriam_albums = 585 :=
by
  sorry

end total_albums_l2223_222313


namespace negation_of_proposition_p_l2223_222356

def f : ℝ → ℝ := sorry

theorem negation_of_proposition_p :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔ (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := 
by
  sorry

end negation_of_proposition_p_l2223_222356


namespace stephanie_speed_l2223_222305

noncomputable def distance : ℝ := 15
noncomputable def time : ℝ := 3

theorem stephanie_speed :
  distance / time = 5 := 
sorry

end stephanie_speed_l2223_222305


namespace compare_series_l2223_222388

theorem compare_series (x y : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) : 
  (1 / (1 - x^2) + 1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by
  sorry

end compare_series_l2223_222388


namespace intersection_complement_l2223_222399

open Set

variable (U : Set ℕ) (P Q : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (hP : P = {1, 2, 3, 4, 5})
variable (hQ : Q = {3, 4, 5, 6, 7})

theorem intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_l2223_222399


namespace y_n_sq_eq_3_x_n_sq_add_1_l2223_222366

def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 1) => 4 * x n - x (n - 1)

def y : ℕ → ℤ
| 0       => 1
| 1       => 2
| (n + 1) => 4 * y n - y (n - 1)

theorem y_n_sq_eq_3_x_n_sq_add_1 (n : ℕ) : y n ^ 2 = 3 * (x n) ^ 2 + 1 :=
sorry

end y_n_sq_eq_3_x_n_sq_add_1_l2223_222366


namespace altitude_in_scientific_notation_l2223_222303

theorem altitude_in_scientific_notation : 
  (389000 : ℝ) = 3.89 * (10 : ℝ) ^ 5 :=
by
  sorry

end altitude_in_scientific_notation_l2223_222303


namespace range_of_c_div_a_l2223_222380

-- Define the conditions and variables
variables (a b c : ℝ)

-- Define the given conditions
def conditions : Prop :=
  (a ≥ b ∧ b ≥ c) ∧ (a + b + c = 0)

-- Define the range of values for c / a
def range_for_c_div_a : Prop :=
  -2 ≤ c / a ∧ c / a ≤ -1/2

-- The theorem statement to prove
theorem range_of_c_div_a (h : conditions a b c) : range_for_c_div_a a c := 
  sorry

end range_of_c_div_a_l2223_222380


namespace inverse_h_l2223_222311

def f (x : ℝ) : ℝ := 5 * x - 7
def g (x : ℝ) : ℝ := 3 * x + 2
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_h : (∀ x : ℝ, h (15 * x + 3) = x) :=
by
  -- Proof would go here
  sorry

end inverse_h_l2223_222311


namespace enclosed_area_l2223_222396

theorem enclosed_area {x y : ℝ} (h : x^2 + y^2 = 2 * |x| + 2 * |y|) : ∃ (A : ℝ), A = 8 :=
sorry

end enclosed_area_l2223_222396


namespace range_of_y_l2223_222348

theorem range_of_y (y : ℝ) (h1: 1 / y < 3) (h2: 1 / y > -4) : y > 1 / 3 :=
by
  sorry

end range_of_y_l2223_222348


namespace priya_trip_time_l2223_222320

noncomputable def time_to_drive_from_X_to_Z_at_50_mph : ℝ := 5

theorem priya_trip_time :
  (∀ (distance_YZ distance_XZ : ℝ), 
    distance_YZ = 60 * 2.0833333333333335 ∧
    distance_XZ = distance_YZ * 2 →
    time_to_drive_from_X_to_Z_at_50_mph = distance_XZ / 50 ) :=
sorry

end priya_trip_time_l2223_222320


namespace sum_series_l2223_222324

theorem sum_series :
  3 * (List.sum (List.map (λ n => n - 1) (List.range' 2 14))) = 273 :=
by
  sorry

end sum_series_l2223_222324


namespace min_value_geometric_seq_l2223_222345

theorem min_value_geometric_seq (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * r)
  (h3 : a 5 * a 4 * a 2 * a 1 = 16) :
  a 1 + a 5 = 4 :=
sorry

end min_value_geometric_seq_l2223_222345


namespace alice_marble_groups_l2223_222387

-- Define the number of each colored marble Alice has
def pink_marble := 1
def blue_marble := 1
def white_marble := 1
def black_marbles := 4

-- The function to count the number of different groups of two marbles Alice can choose
noncomputable def count_groups : Nat :=
  let total_colors := 4  -- Pink, Blue, White, and one representative black
  1 + (total_colors.choose 2)

-- The theorem statement 
theorem alice_marble_groups : count_groups = 7 := by 
  sorry

end alice_marble_groups_l2223_222387


namespace div_by_3_pow_101_l2223_222306

theorem div_by_3_pow_101 : ∀ (n : ℕ), (∀ k : ℕ, (3^(k+1)) ∣ (2^(3^k) + 1)) → 3^101 ∣ 2^(3^100) + 1 :=
by
  sorry

end div_by_3_pow_101_l2223_222306


namespace sum_of_integers_is_18_l2223_222362

theorem sum_of_integers_is_18 (a b c d : ℕ) 
  (h1 : a * b + c * d = 38)
  (h2 : a * c + b * d = 34)
  (h3 : a * d + b * c = 43) : 
  a + b + c + d = 18 := 
  sorry

end sum_of_integers_is_18_l2223_222362


namespace y_greater_than_one_l2223_222375

variable (x y : ℝ)

theorem y_greater_than_one (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
sorry

end y_greater_than_one_l2223_222375


namespace greatest_brownies_produced_l2223_222398

theorem greatest_brownies_produced (p side_length a b brownies : ℕ) :
  (4 * side_length = p) →
  (p = 40) →
  (brownies = side_length * side_length) →
  ((side_length - a - 2) * (side_length - b - 2) = 2 * (2 * (side_length - a) + 2 * (side_length - b) - 4)) →
  (a = 4) →
  (b = 4) →
  brownies = 100 :=
by
  intros h_perimeter h_perimeter_value h_brownies h_eq h_a h_b
  sorry

end greatest_brownies_produced_l2223_222398


namespace distinct_solution_count_l2223_222365

theorem distinct_solution_count : ∀ (x : ℝ), (|x - 10| = |x + 4|) → x = 3 :=
by
  sorry

end distinct_solution_count_l2223_222365


namespace problem_1_problem_2_l2223_222312

noncomputable def f (x m : ℝ) := |x - 4 / m| + |x + m|

theorem problem_1 (m : ℝ) (hm : 0 < m) (x : ℝ) : f x m ≥ 4 := sorry

theorem problem_2 (m : ℝ) (hm : f 2 m > 5) : 
  m ∈ Set.Ioi ((1 + Real.sqrt 17) / 2) ∪ Set.Ioo 0 1 := sorry

end problem_1_problem_2_l2223_222312


namespace cone_lateral_area_l2223_222394

theorem cone_lateral_area (r l S: ℝ) (h1: r = 1 / 2) (h2: l = 1) (h3: S = π * r * l) : 
  S = π / 2 :=
by
  sorry

end cone_lateral_area_l2223_222394


namespace ant_trip_ratio_l2223_222367

theorem ant_trip_ratio (A B : ℕ) (x c : ℕ) (h1 : A * x = c) (h2 : B * (3 / 2 * x) = 3 * c) :
  B = 2 * A :=
by
  sorry

end ant_trip_ratio_l2223_222367


namespace line_passes_through_quadrants_l2223_222381

variables (a b c p : ℝ)

-- Given conditions
def conditions :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  (a + b) / c = p ∧ 
  (b + c) / a = p ∧ 
  (c + a) / b = p

-- Goal statement
theorem line_passes_through_quadrants : conditions a b c p → 
  (∃ x : ℝ, x > 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p < 0) :=
sorry

end line_passes_through_quadrants_l2223_222381


namespace tangent_line_equation_l2223_222389

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + a * x^2 + (a - 3) * x

noncomputable def f' (a x : ℝ) : ℝ :=
  3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_equation (a : ℝ) (h : ∀ x : ℝ, f a (-x) = f a x) :
    9 * (2 : ℝ) - f a 2 - 16 = 0 :=
by
  sorry

end tangent_line_equation_l2223_222389


namespace percentage_increase_l2223_222334

theorem percentage_increase (W E : ℝ) (P : ℝ) :
  W = 200 →
  E = 204 →
  (∃ P, E = W * (1 + P / 100) * 0.85) →
  P = 20 :=
by
  intros hW hE hP
  -- Proof could be added here.
  sorry

end percentage_increase_l2223_222334
