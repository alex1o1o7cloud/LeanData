import Mathlib

namespace y_is_less_than_x_by_9444_percent_l507_50789

theorem y_is_less_than_x_by_9444_percent (x y : ℝ) (h : x = 18 * y) : (x - y) / x * 100 = 94.44 :=
by
  sorry

end y_is_less_than_x_by_9444_percent_l507_50789


namespace correct_value_l507_50762

theorem correct_value (x : ℝ) (h : x / 3.6 = 2.5) : (x * 3.6) / 2 = 16.2 :=
by {
  -- Proof would go here
  sorry
}

end correct_value_l507_50762


namespace length_of_AD_in_parallelogram_l507_50738

theorem length_of_AD_in_parallelogram
  (x : ℝ)
  (AB BC CD : ℝ)
  (AB_eq : AB = x + 3)
  (BC_eq : BC = x - 4)
  (CD_eq : CD = 16)
  (parallelogram_ABCD : AB = CD ∧ AD = BC) :
  AD = 9 := by
sorry

end length_of_AD_in_parallelogram_l507_50738


namespace remainder_mod_7_l507_50706

theorem remainder_mod_7 : (4 * 6^24 + 3^48) % 7 = 5 := by
  sorry

end remainder_mod_7_l507_50706


namespace k_range_correct_l507_50718

noncomputable def k_range (k : ℝ) : Prop :=
  (∀ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
  (∀ x : ℝ, k * x ^ 2 + k * x + 1 > 0) ∧
  ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∨
   (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0)) ∧
  ¬ ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
    (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0))

theorem k_range_correct (k : ℝ) : k_range k ↔ (-3 < k ∧ k < 0) ∨ (3 ≤ k ∧ k < 4) :=
sorry

end k_range_correct_l507_50718


namespace product_roots_cos_pi_by_9_cos_2pi_by_9_l507_50776

theorem product_roots_cos_pi_by_9_cos_2pi_by_9 :
  ∀ (d e : ℝ), (∀ x, x^2 + d * x + e = (x - Real.cos (π / 9)) * (x - Real.cos (2 * π / 9))) → 
    d * e = -5 / 64 :=
by
  sorry

end product_roots_cos_pi_by_9_cos_2pi_by_9_l507_50776


namespace factor_expression_l507_50773

theorem factor_expression (x y : ℝ) : 5 * x * (x + 1) + 7 * (x + 1) - 2 * y * (x + 1) = (x + 1) * (5 * x + 7 - 2 * y) :=
by
  sorry

end factor_expression_l507_50773


namespace tiger_catch_distance_correct_l507_50779

noncomputable def tiger_catch_distance (tiger_leaps_behind : ℕ) (tiger_leaps_per_minute : ℕ) (deer_leaps_per_minute : ℕ) (tiger_m_per_leap : ℕ) (deer_m_per_leap : ℕ) : ℕ :=
  let initial_distance := tiger_leaps_behind * tiger_m_per_leap
  let tiger_per_minute := tiger_leaps_per_minute * tiger_m_per_leap
  let deer_per_minute := deer_leaps_per_minute * deer_m_per_leap
  let gain_per_minute := tiger_per_minute - deer_per_minute
  let time_to_catch := initial_distance / gain_per_minute
  time_to_catch * tiger_per_minute

theorem tiger_catch_distance_correct :
  tiger_catch_distance 50 5 4 8 5 = 800 :=
by
  -- This is the placeholder for the proof.
  sorry

end tiger_catch_distance_correct_l507_50779


namespace lemonade_percentage_l507_50766

theorem lemonade_percentage (L : ℝ) : 
  (0.4 * (1 - L / 100) + 0.6 * 0.55 = 0.65) → L = 20 :=
by
  sorry

end lemonade_percentage_l507_50766


namespace students_catching_up_on_homework_l507_50736

theorem students_catching_up_on_homework
  (total_students : ℕ)
  (half_doing_silent_reading : ℕ)
  (third_playing_board_games : ℕ)
  (remain_catching_up_homework : ℕ) :
  total_students = 24 →
  half_doing_silent_reading = total_students / 2 →
  third_playing_board_games = total_students / 3 →
  remain_catching_up_homework = total_students - (half_doing_silent_reading + third_playing_board_games) →
  remain_catching_up_homework = 4 :=
by
  intros h_total h_half h_third h_remain
  sorry

end students_catching_up_on_homework_l507_50736


namespace consecutive_product_even_product_divisible_by_6_l507_50761

theorem consecutive_product_even (n : ℕ) : ∃ k, n * (n + 1) = 2 * k := 
sorry

theorem product_divisible_by_6 (n : ℕ) : 6 ∣ (n * (n + 1) * (2 * n + 1)) :=
sorry

end consecutive_product_even_product_divisible_by_6_l507_50761


namespace factorize_difference_of_squares_l507_50745

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2 * x) * (3 + 2 * x) :=
sorry

end factorize_difference_of_squares_l507_50745


namespace tickets_distribution_correct_l507_50744

def tickets_distribution (tickets programs : nat) (A_tickets_min : nat) : nat :=
sorry

theorem tickets_distribution_correct :
  tickets_distribution 6 4 3 = 17 :=
by
  sorry

end tickets_distribution_correct_l507_50744


namespace swimming_both_days_l507_50788

theorem swimming_both_days
  (total_students swimming_today soccer_today : ℕ)
  (students_swimming_yesterday students_soccer_yesterday : ℕ)
  (soccer_today_swimming_yesterday soccer_today_soccer_yesterday : ℕ)
  (swimming_today_swimming_yesterday swimming_today_soccer_yesterday : ℕ) :
  total_students = 33 ∧
  swimming_today = 22 ∧
  soccer_today = 22 ∧
  soccer_today_swimming_yesterday = 15 ∧
  soccer_today_soccer_yesterday = 15 ∧
  swimming_today_swimming_yesterday = 15 ∧
  swimming_today_soccer_yesterday = 15 →
  ∃ (swimming_both_days : ℕ), swimming_both_days = 4 :=
by
  sorry

end swimming_both_days_l507_50788


namespace general_term_formula_l507_50786

def sequence_sums (n : ℕ) : ℕ := 2 * n^2 + n

theorem general_term_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : S = sequence_sums) :
  (∀ n, a n = S n - S (n-1)) → ∀ n, a n = 4 * n - 1 :=
by
  sorry

end general_term_formula_l507_50786


namespace find_AD_find_a_rhombus_l507_50728

variable (a : ℝ) (AB AD : ℝ)

-- Problem 1: Given AB = 2, find AD
theorem find_AD (h1 : AB = 2)
    (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = AB ∨ x = AD) : AD = 5 := sorry

-- Problem 2: Find the value of a such that ABCD is a rhombus
theorem find_a_rhombus (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = 2 → AB = AD → x = a ∨ AB = AD → x = 10) :
    a = 10 := sorry

end find_AD_find_a_rhombus_l507_50728


namespace solution_set_of_x_squared_gt_x_l507_50717

theorem solution_set_of_x_squared_gt_x :
  { x : ℝ | x^2 > x } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end solution_set_of_x_squared_gt_x_l507_50717


namespace marks_deducted_per_wrong_answer_l507_50777

theorem marks_deducted_per_wrong_answer
  (correct_awarded : ℕ)
  (total_marks : ℕ)
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (final_marks : ℕ) :
  correct_awarded = 3 →
  total_marks = 38 →
  total_questions = 70 →
  correct_answers = 27 →
  incorrect_answers = total_questions - correct_answers →
  final_marks = total_marks →
  final_marks = correct_answers * correct_awarded - incorrect_answers * 1 →
  1 = 1
  := by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end marks_deducted_per_wrong_answer_l507_50777


namespace quad_function_one_zero_l507_50715

theorem quad_function_one_zero (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 6 * x + 1 = 0 ∧ (∀ x1 x2 : ℝ, m * x1^2 - 6 * x1 + 1 = 0 ∧ m * x2^2 - 6 * x2 + 1 = 0 → x1 = x2)) ↔ (m = 0 ∨ m = 9) :=
by
  sorry

end quad_function_one_zero_l507_50715


namespace find_speed_l507_50707

noncomputable def circumference := 15 / 5280 -- miles
noncomputable def increased_speed (r : ℝ) := r + 5 -- miles per hour
noncomputable def reduced_time (t : ℝ) := t - 1 / 10800 -- hours
noncomputable def original_distance (r t : ℝ) := r * t
noncomputable def new_distance (r t : ℝ) := increased_speed r * reduced_time t

theorem find_speed (r t : ℝ) (h1 : original_distance r t = circumference) 
(h2 : new_distance r t = circumference) : r = 13.5 := by
  sorry

end find_speed_l507_50707


namespace figure_can_be_rearranged_to_square_l507_50753

def can_form_square (n : ℕ) : Prop :=
  let s := Nat.sqrt n
  s * s = n

theorem figure_can_be_rearranged_to_square (n : ℕ) :
  (∃ a b c : ℕ, a + b + c = n) → (can_form_square n) → (n % 1 = 0) :=
by
  intros _ _
  sorry

end figure_can_be_rearranged_to_square_l507_50753


namespace not_distributive_add_mul_l507_50796

-- Definition of the addition operation on pairs of real numbers
def pair_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.fst + b.fst, a.snd + b.snd)

-- Definition of the multiplication operation on pairs of real numbers
def pair_mul (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.fst * b.fst - a.snd * b.snd, a.fst * b.snd + a.snd * b.fst)

-- The problem statement: distributive law of addition over multiplication does not hold
theorem not_distributive_add_mul (a b c : ℝ × ℝ) :
  pair_add a (pair_mul b c) ≠ pair_mul (pair_add a b) (pair_add a c) :=
sorry

end not_distributive_add_mul_l507_50796


namespace helen_hand_washing_time_l507_50725

theorem helen_hand_washing_time :
  (52 / 4) * 30 / 60 = 6.5 := by
  sorry

end helen_hand_washing_time_l507_50725


namespace find_abc_value_l507_50702

open Real

/- Defining the conditions -/
variables (a b c : ℝ)
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a * (b + c) = 156) (h5 : b * (c + a) = 168) (h6 : c * (a + b) = 176)

/- Prove the value of abc -/
theorem find_abc_value :
  a * b * c = 754 :=
sorry

end find_abc_value_l507_50702


namespace value_of_a_minus_n_plus_k_l507_50772

theorem value_of_a_minus_n_plus_k :
  ∃ (a k n : ℤ), 
    (∀ x : ℤ, (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n) ∧ 
    (a - n + k = 3) :=
sorry

end value_of_a_minus_n_plus_k_l507_50772


namespace broken_pieces_correct_l507_50703

variable (pieces_transported : ℕ)
variable (shipping_cost_per_piece : ℝ)
variable (compensation_per_broken_piece : ℝ)
variable (total_profit : ℝ)
variable (broken_pieces : ℕ)

def logistics_profit (pieces_transported : ℕ) (shipping_cost_per_piece : ℝ) 
                     (compensation_per_broken_piece : ℝ) (broken_pieces : ℕ) : ℝ :=
  shipping_cost_per_piece * (pieces_transported - broken_pieces) - compensation_per_broken_piece * broken_pieces

theorem broken_pieces_correct :
  pieces_transported = 2000 →
  shipping_cost_per_piece = 0.2 →
  compensation_per_broken_piece = 2.3 →
  total_profit = 390 →
  logistics_profit pieces_transported shipping_cost_per_piece compensation_per_broken_piece broken_pieces = total_profit →
  broken_pieces = 4 :=
by
  intros
  sorry

end broken_pieces_correct_l507_50703


namespace incorrect_conclusion_l507_50771

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (2 * x)

theorem incorrect_conclusion :
  ¬ (∀ x : ℝ, f ( (3 * Real.pi) / 4 - x ) + f x = 0) :=
by
  sorry

end incorrect_conclusion_l507_50771


namespace total_area_of_paths_l507_50774

theorem total_area_of_paths:
  let bed_width := 4
  let bed_height := 3
  let num_beds_width := 3
  let num_beds_height := 5
  let path_width := 2

  let total_bed_width := num_beds_width * bed_width
  let total_path_width := (num_beds_width + 1) * path_width
  let total_width := total_bed_width + total_path_width

  let total_bed_height := num_beds_height * bed_height
  let total_path_height := (num_beds_height + 1) * path_width
  let total_height := total_bed_height + total_path_height

  let total_area_greenhouse := total_width * total_height
  let total_area_beds := num_beds_width * num_beds_height * bed_width * bed_height

  let total_area_paths := total_area_greenhouse - total_area_beds

  total_area_paths = 360 :=
by sorry

end total_area_of_paths_l507_50774


namespace geom_inequality_l507_50704

noncomputable def geom_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geom_inequality (a1 q : ℝ) (h_q : q ≠ 0) :
  (a1 * (a1 * q^2)) > 0 :=
by
  sorry

end geom_inequality_l507_50704


namespace max_marks_paper_one_l507_50787

theorem max_marks_paper_one (M : ℝ) : 
  (0.42 * M = 64) → (M = 152) :=
by
  sorry

end max_marks_paper_one_l507_50787


namespace parity_of_E2021_E2022_E2023_l507_50767

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 0
  else seq (n - 2) + seq (n - 3)

theorem parity_of_E2021_E2022_E2023 :
  is_odd (seq 2021) ∧ is_even (seq 2022) ∧ is_odd (seq 2023) :=
by
  sorry

end parity_of_E2021_E2022_E2023_l507_50767


namespace john_protest_days_l507_50769

theorem john_protest_days (days1: ℕ) (days2: ℕ) (days3: ℕ): 
  days1 = 4 → 
  days2 = (days1 + (days1 / 4)) → 
  days3 = (days2 + (days2 / 2)) → 
  (days1 + days2 + days3) = 17 :=
by
  intros h1 h2 h3
  sorry

end john_protest_days_l507_50769


namespace find_x_minus_y_l507_50741

theorem find_x_minus_y {x y z : ℤ} (h1 : x - (y + z) = 5) (h2 : x - y + z = -1) : x - y = 2 :=
by
  sorry

end find_x_minus_y_l507_50741


namespace number_of_initial_cans_l507_50700

theorem number_of_initial_cans (n : ℕ) (T : ℝ)
  (h1 : T = n * 36.5)
  (h2 : T - (2 * 49.5) = (n - 2) * 30) :
  n = 6 :=
sorry

end number_of_initial_cans_l507_50700


namespace tan_sum_pi_over_4_sin_cos_fraction_l507_50746

open Real

variable (α : ℝ)

axiom tan_α_eq_2 : tan α = 2

theorem tan_sum_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
sorry

theorem sin_cos_fraction (α : ℝ) (h : tan α = 2) : (sin α + cos α) / (sin α - cos α) = 3 :=
sorry

end tan_sum_pi_over_4_sin_cos_fraction_l507_50746


namespace intersection_S_T_eq_T_l507_50733

noncomputable def S : Set ℝ := { y | ∃ x : ℝ, y = 3^x }
noncomputable def T : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l507_50733


namespace value_of_J_l507_50797

-- Given conditions
variables (Y J : ℤ)

-- Condition definitions
axiom condition1 : 150 < Y ∧ Y < 300
axiom condition2 : Y = J^2 * J^3
axiom condition3 : ∃ n : ℤ, Y = n^3

-- Goal: Value of J
theorem value_of_J : J = 3 :=
by { sorry }  -- Proof omitted

end value_of_J_l507_50797


namespace solution_set_for_inequality_l507_50722

theorem solution_set_for_inequality : 
  { x : ℝ | x * (x - 1) < 2 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_for_inequality_l507_50722


namespace painting_cost_conversion_l507_50758

def paintingCostInCNY (paintingCostNAD : ℕ) (usd_to_nad : ℕ) (usd_to_cny : ℕ) : ℕ :=
  paintingCostNAD * (1 / usd_to_nad) * usd_to_cny

theorem painting_cost_conversion :
  (paintingCostInCNY 105 7 6 = 90) :=
by
  sorry

end painting_cost_conversion_l507_50758


namespace primes_pos_int_solutions_l507_50747

theorem primes_pos_int_solutions 
  (p : ℕ) [hp : Fact (Nat.Prime p)] (a b : ℕ) (h1 : ∃ k : ℤ, (4 * a + p : ℤ) + k * (4 * b + p : ℤ) = b * k * a)
  (h2 : ∃ m : ℤ, (a^2 : ℤ) + m * (b^2 : ℤ) = b * m * a) : a = b ∨ a = b * p :=
  sorry

end primes_pos_int_solutions_l507_50747


namespace man_distance_from_start_l507_50716

noncomputable def distance_from_start (west_distance north_distance : ℝ) : ℝ :=
  Real.sqrt (west_distance^2 + north_distance^2)

theorem man_distance_from_start :
  distance_from_start 10 10 = Real.sqrt 200 :=
by
  sorry

end man_distance_from_start_l507_50716


namespace result_of_4_times_3_l507_50721

def operation (a b : ℕ) : ℕ :=
  a^2 + a * Nat.factorial b - b^2

theorem result_of_4_times_3 : operation 4 3 = 31 := by
  sorry

end result_of_4_times_3_l507_50721


namespace hannah_games_l507_50783

theorem hannah_games (total_points : ℕ) (avg_points_per_game : ℕ) (h1 : total_points = 312) (h2 : avg_points_per_game = 13) :
  total_points / avg_points_per_game = 24 :=
sorry

end hannah_games_l507_50783


namespace odd_function_strictly_decreasing_l507_50775

noncomputable def f (x : ℝ) : ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom negative_condition (x : ℝ) (hx : x > 0) : f x < 0

theorem odd_function : ∀ x : ℝ, f (-x) = -f x :=
by sorry

theorem strictly_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
by sorry

end odd_function_strictly_decreasing_l507_50775


namespace length_of_each_piece_l507_50754

theorem length_of_each_piece (rod_length : ℝ) (num_pieces : ℕ) (h₁ : rod_length = 42.5) (h₂ : num_pieces = 50) : (rod_length / num_pieces * 100) = 85 := 
by 
  sorry

end length_of_each_piece_l507_50754


namespace billion_to_scientific_l507_50793
noncomputable def scientific_notation_of_billion (n : ℝ) : ℝ := n * 10^9
theorem billion_to_scientific (a : ℝ) : scientific_notation_of_billion a = 1.48056 * 10^11 :=
by sorry

end billion_to_scientific_l507_50793


namespace extreme_values_of_f_max_min_values_on_interval_l507_50765

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.exp x)

theorem extreme_values_of_f : 
  (∃ x_max : ℝ, f x_max = 2 / Real.exp 1 ∧ ∀ x : ℝ, f x ≤ 2 / Real.exp 1) :=
sorry

theorem max_min_values_on_interval : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 
    (f 1 = 2 / Real.exp 1 ∧ ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → f x ≤ 2 / Real.exp 1)
     ∧ (f 2 = 4 / (Real.exp 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 2, 4 / (Real.exp 2) ≤ f x)) :=
sorry

end extreme_values_of_f_max_min_values_on_interval_l507_50765


namespace find_larger_number_l507_50701

variable (L S : ℕ)

theorem find_larger_number 
  (h1 : L - S = 1355) 
  (h2 : L = 6 * S + 15) : 
  L = 1623 := 
sorry

end find_larger_number_l507_50701


namespace sum_of_cubes_roots_poly_l507_50760

theorem sum_of_cubes_roots_poly :
  (∀ (a b c : ℂ), (a^3 - 2*a^2 + 2*a - 3 = 0) ∧ (b^3 - 2*b^2 + 2*b - 3 = 0) ∧ (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5) :=
by
  sorry

end sum_of_cubes_roots_poly_l507_50760


namespace max_weak_quartets_120_l507_50740

noncomputable def max_weak_quartets (n : ℕ) : ℕ :=
  -- Placeholder definition to represent the maximum weak quartets
  sorry  -- To be replaced with the actual mathematical definition

theorem max_weak_quartets_120 : max_weak_quartets 120 = 4769280 := by
  sorry

end max_weak_quartets_120_l507_50740


namespace calc_pairs_count_l507_50709

theorem calc_pairs_count :
  ∃! (ab : ℤ × ℤ), (ab.1 + ab.2 = ab.1 * ab.2) :=
by
  sorry

end calc_pairs_count_l507_50709


namespace find_actual_price_of_good_l507_50781

theorem find_actual_price_of_good (P : ℝ) (price_after_discounts : P * 0.93 * 0.90 * 0.85 * 0.75 = 6600) :
  P = 11118.75 :=
by
  sorry

end find_actual_price_of_good_l507_50781


namespace harry_total_expenditure_l507_50791

theorem harry_total_expenditure :
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_packets := 3
  let tomato_packets := 4
  let chili_pepper_packets := 5
  (pumpkin_packets * pumpkin_price) + (tomato_packets * tomato_price) + (chili_pepper_packets * chili_pepper_price) = 18.00 :=
by
  sorry

end harry_total_expenditure_l507_50791


namespace maximize_angle_distance_l507_50705

noncomputable def f (x : ℝ) : ℝ :=
  40 * x / (x * x + 500)

theorem maximize_angle_distance :
  ∃ x : ℝ, x = 10 * Real.sqrt 5 ∧ ∀ y : ℝ, y ≠ x → f y < f x :=
sorry

end maximize_angle_distance_l507_50705


namespace distance_symmetric_parabola_l507_50755

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def parabola (x : ℝ) : ℝ := 3 - x^2

theorem distance_symmetric_parabola (A B : ℝ × ℝ) 
  (hA : A.2 = parabola A.1) 
  (hB : B.2 = parabola B.1)
  (hSym : A.1 + A.2 = 0 ∧ B.1 + B.2 = 0) 
  (hDistinct : A ≠ B) :
  distance A B = 3 * sqrt 2 :=
by
  sorry

end distance_symmetric_parabola_l507_50755


namespace range_of_x_l507_50724

theorem range_of_x (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
sorry

end range_of_x_l507_50724


namespace handshake_problem_l507_50794

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem handshake_problem : combinations 40 2 = 780 := 
by
  sorry

end handshake_problem_l507_50794


namespace mildred_oranges_l507_50737

theorem mildred_oranges (original after given : ℕ) (h1 : original = 77) (h2 : after = 79) (h3 : given = after - original) : given = 2 :=
by
  sorry

end mildred_oranges_l507_50737


namespace total_cost_of_selling_watermelons_l507_50792

-- Definitions of the conditions:
def watermelon_weight : ℝ := 23.0
def daily_prices : List ℝ := [2.10, 1.90, 1.80, 2.30, 2.00, 1.95, 2.20]
def discount_threshold : ℕ := 15
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def number_of_watermelons : ℕ := 18

-- The theorem statement:
theorem total_cost_of_selling_watermelons :
  let average_price := (daily_prices.sum / daily_prices.length)
  let total_weight := number_of_watermelons * watermelon_weight
  let initial_cost := total_weight * average_price
  let discounted_cost := if number_of_watermelons > discount_threshold then initial_cost * (1 - discount_rate) else initial_cost
  let final_cost := discounted_cost * (1 + sales_tax_rate)
  final_cost = 796.43 := by
    sorry

end total_cost_of_selling_watermelons_l507_50792


namespace rhombus_area_l507_50759

-- Define the lengths of the diagonals
def d1 : ℝ := 25
def d2 : ℝ := 30

-- Statement to prove that the area of the rhombus is 375 square centimeters
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 25) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 375 := by
  -- Proof to be provided
  sorry

end rhombus_area_l507_50759


namespace part_a_possible_final_number_l507_50708

theorem part_a_possible_final_number :
  ∃ (n : ℕ), n = 97 ∧ 
  (∃ f : {x // x ≠ 0} → ℕ → ℕ, 
    f ⟨1, by decide⟩ 0 = 1 ∧ 
    f ⟨2, by decide⟩ 1 = 2 ∧ 
    f ⟨4, by decide⟩ 2 = 4 ∧ 
    f ⟨8, by decide⟩ 3 = 8 ∧ 
    f ⟨16, by decide⟩ 4 = 16 ∧ 
    f ⟨32, by decide⟩ 5 = 32 ∧ 
    f ⟨64, by decide⟩ 6 = 64 ∧ 
    f ⟨128, by decide⟩ 7 = 128 ∧ 
    ∀ i j : {x // x ≠ 0}, f i j = (f i j - f i j)) := sorry

end part_a_possible_final_number_l507_50708


namespace complex_fraction_sum_l507_50751

theorem complex_fraction_sum :
  let a := (1 : ℂ)
  let b := (0 : ℂ)
  (a + b) = 1 :=
by
  sorry

end complex_fraction_sum_l507_50751


namespace find_a_l507_50711

noncomputable def f (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, g a x = 2 * x) ∧ (deriv f 1 = 2) ∧ f 1 = 2 → a = 4 :=
by
  -- Math proof goes here
  sorry

end find_a_l507_50711


namespace term_217_is_61st_l507_50756

variables {a_n : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) (a_15 a_45 : ℝ) : Prop :=
  ∃ (a₁ d : ℝ), (∀ n, a_n n = a₁ + (n - 1) * d) ∧ a_n 15 = a_15 ∧ a_n 45 = a_45

theorem term_217_is_61st (h : arithmetic_sequence a_n 33 153) : a_n 61 = 217 := sorry

end term_217_is_61st_l507_50756


namespace find_AX_l507_50795

theorem find_AX (AC BC BX : ℝ) (h1 : AC = 27) (h2 : BC = 40) (h3 : BX = 36)
    (h4 : ∀ (AX : ℝ), AX = AC * BX / BC) : 
    ∃ AX, AX = 243 / 10 :=
by
  sorry

end find_AX_l507_50795


namespace collinear_iff_real_simple_ratio_l507_50757

theorem collinear_iff_real_simple_ratio (a b c : ℂ) : (∃ k : ℝ, a = k * b + (1 - k) * c) ↔ ∃ r : ℝ, (a - b) / (a - c) = r :=
sorry

end collinear_iff_real_simple_ratio_l507_50757


namespace inequality_holds_for_all_x_l507_50720

theorem inequality_holds_for_all_x (m : ℝ) : (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m - 1)*x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by {
  sorry
}

end inequality_holds_for_all_x_l507_50720


namespace ratio_speed_car_speed_bike_l507_50743

def speed_of_tractor := 575 / 23
def speed_of_bike := 2 * speed_of_tractor
def speed_of_car := 540 / 6
def ratio := speed_of_car / speed_of_bike

theorem ratio_speed_car_speed_bike : ratio = 9 / 5 := by
  sorry

end ratio_speed_car_speed_bike_l507_50743


namespace P_neither_l507_50739

-- Definition of probabilities according to given conditions
def P_A : ℝ := 0.63      -- Probability of answering the first question correctly
def P_B : ℝ := 0.50      -- Probability of answering the second question correctly
def P_A_and_B : ℝ := 0.33  -- Probability of answering both questions correctly

-- Theorem to prove the probability of answering neither of the questions correctly
theorem P_neither : (1 - (P_A + P_B - P_A_and_B)) = 0.20 := by
  sorry

end P_neither_l507_50739


namespace max_black_cells_1000_by_1000_l507_50778

def maxBlackCells (m n : ℕ) : ℕ :=
  if m = 1 then n else if n = 1 then m else m + n - 2

theorem max_black_cells_1000_by_1000 : maxBlackCells 1000 1000 = 1998 :=
  by sorry

end max_black_cells_1000_by_1000_l507_50778


namespace total_points_scored_l507_50727

theorem total_points_scored (m2 m3 m1 o2 o3 o1 : ℕ) 
  (H1 : m2 = 25) 
  (H2 : m3 = 8) 
  (H3 : m1 = 10) 
  (H4 : o2 = 2 * m2) 
  (H5 : o3 = m3 / 2) 
  (H6 : o1 = m1 / 2) : 
  (2 * m2 + 3 * m3 + m1) + (2 * o2 + 3 * o3 + o1) = 201 := 
by
  sorry

end total_points_scored_l507_50727


namespace a_squared_divisible_by_b_l507_50742

theorem a_squared_divisible_by_b (a b : ℕ) (h1 : a < 1000) (h2 : b > 0) 
    (h3 : ∃ k, a ^ 21 = b ^ 10 * k) : ∃ m, a ^ 2 = b * m := 
by
  sorry

end a_squared_divisible_by_b_l507_50742


namespace pancakes_eaten_by_older_is_12_l507_50713

/-- Pancake problem conditions -/
def initial_pancakes : ℕ := 19
def final_pancakes : ℕ := 11
def younger_eats_per_cycle : ℕ := 1
def older_eats_per_cycle : ℕ := 3
def grandma_bakes_per_cycle : ℕ := 2
def net_reduction_per_cycle := younger_eats_per_cycle + older_eats_per_cycle - grandma_bakes_per_cycle
def total_pancakes_eaten_by_older (cycles : ℕ) := older_eats_per_cycle * cycles

/-- Calculate the cycles based on net reduction -/
def cycles : ℕ := (initial_pancakes - final_pancakes) / net_reduction_per_cycle

/-- Prove the number of pancakes the older grandchild eats is 12 based on given conditions --/
theorem pancakes_eaten_by_older_is_12 : total_pancakes_eaten_by_older cycles = 12 := by
  sorry

end pancakes_eaten_by_older_is_12_l507_50713


namespace woman_stop_time_l507_50750

-- Conditions
def man_speed := 5 -- in miles per hour
def woman_speed := 15 -- in miles per hour
def wait_time := 4 -- in minutes
def man_speed_mpm : ℚ := man_speed * (1 / 60) -- convert to miles per minute
def distance_covered := man_speed_mpm * wait_time

-- Definition of the relative speed between the woman and the man
def relative_speed := woman_speed - man_speed
def relative_speed_mpm : ℚ := relative_speed * (1 / 60) -- convert to miles per minute

-- The Proof statement
theorem woman_stop_time :
  (distance_covered / relative_speed_mpm) = 2 :=
by
  sorry

end woman_stop_time_l507_50750


namespace lucas_raspberry_candies_l507_50749

-- Define the problem conditions and the question
theorem lucas_raspberry_candies :
  ∃ (r l : ℕ), (r = 3 * l) ∧ ((r - 5) = 4 * (l - 5)) ∧ (r = 45) :=
by
  sorry

end lucas_raspberry_candies_l507_50749


namespace discount_percentage_is_30_l507_50726

theorem discount_percentage_is_30 
  (price_per_pant : ℝ) (num_of_pants : ℕ)
  (price_per_sock : ℝ) (num_of_socks : ℕ)
  (total_spend_after_discount : ℝ)
  (original_pants_price := num_of_pants * price_per_pant)
  (original_socks_price := num_of_socks * price_per_sock)
  (original_total_price := original_pants_price + original_socks_price)
  (discount_amount := original_total_price - total_spend_after_discount)
  (discount_percentage := (discount_amount / original_total_price) * 100) :
  (price_per_pant = 110) ∧ 
  (num_of_pants = 4) ∧ 
  (price_per_sock = 60) ∧ 
  (num_of_socks = 2) ∧ 
  (total_spend_after_discount = 392) →
  discount_percentage = 30 := by
  sorry

end discount_percentage_is_30_l507_50726


namespace algebraic_expression_value_l507_50748

-- Define the problem conditions and the final proof statement.
theorem algebraic_expression_value : 
  (∀ m n : ℚ, (2 * m - 1 = 0) → (1 / 2 * n - 2 * m = 0) → m ^ 2023 * n ^ 2022 = 1 / 2) :=
by
  sorry

end algebraic_expression_value_l507_50748


namespace Fr_zero_for_all_r_l507_50790

noncomputable def F (r : ℕ) (x y z A B C : ℝ) : ℝ :=
  x^r * Real.sin (r * A) + y^r * Real.sin (r * B) + z^r * Real.sin (r * C)

theorem Fr_zero_for_all_r
  (x y z A B C : ℝ)
  (h_sum : ∃ k : ℤ, A + B + C = k * Real.pi)
  (hF1 : F 1 x y z A B C = 0)
  (hF2 : F 2 x y z A B C = 0)
  : ∀ r : ℕ, F r x y z A B C = 0 :=
sorry

end Fr_zero_for_all_r_l507_50790


namespace complex_number_in_first_quadrant_l507_50730

def is_in_first_quadrant (z : ℂ) : Prop :=
  0 < z.re ∧ 0 < z.im

theorem complex_number_in_first_quadrant (z : ℂ) (h : 0 < z.re ∧ 0 < z.im) : is_in_first_quadrant z :=
by sorry

end complex_number_in_first_quadrant_l507_50730


namespace Thabo_harcdover_nonfiction_books_l507_50731

theorem Thabo_harcdover_nonfiction_books 
  (H P F : ℕ)
  (h1 : P = H + 20)
  (h2 : F = 2 * P)
  (h3 : H + P + F = 180) : 
  H = 30 :=
by
  sorry

end Thabo_harcdover_nonfiction_books_l507_50731


namespace emma_age_proof_l507_50710

theorem emma_age_proof (Inez Zack Jose Emma : ℕ)
  (hJose : Jose = 20)
  (hZack : Zack = Jose + 4)
  (hInez : Inez = Zack - 12)
  (hEmma : Emma = Jose + 5) :
  Emma = 25 :=
by
  sorry

end emma_age_proof_l507_50710


namespace sin_inequality_l507_50782

theorem sin_inequality (d n : ℤ) (hd : d ≥ 1) (hnsq : ∀ k : ℤ, k * k ≠ d) (hn : n ≥ 1) :
  (n * Real.sqrt d + 1) * |Real.sin (n * Real.pi * Real.sqrt d)| ≥ 1 := by
  sorry

end sin_inequality_l507_50782


namespace bus_stops_per_hour_l507_50785

-- Define the constants and conditions given in the problem
noncomputable def speed_without_stoppages : ℝ := 54 -- km/hr
noncomputable def speed_with_stoppages : ℝ := 45 -- km/hr

-- Theorem statement to prove the number of minutes the bus stops per hour
theorem bus_stops_per_hour : (speed_without_stoppages - speed_with_stoppages) / (speed_without_stoppages / 60) = 10 :=
by
  sorry

end bus_stops_per_hour_l507_50785


namespace calculate_correct_subtraction_l507_50712

theorem calculate_correct_subtraction (x : ℤ) (h : x - 63 = 24) : x - 36 = 51 :=
by
  sorry

end calculate_correct_subtraction_l507_50712


namespace find_ab_value_l507_50719

-- Definitions from the conditions
def ellipse_eq (a b : ℝ) : Prop := b^2 - a^2 = 25
def hyperbola_eq (a b : ℝ) : Prop := a^2 + b^2 = 49

-- Main theorem statement
theorem find_ab_value {a b : ℝ} (h_ellipse : ellipse_eq a b) (h_hyperbola : hyperbola_eq a b) : 
  |a * b| = 2 * Real.sqrt 111 :=
by
  -- Proof goes here
  sorry

end find_ab_value_l507_50719


namespace expected_value_of_winnings_is_5_l507_50799

namespace DiceGame

def sides : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def winnings (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then 2 * roll else 0

noncomputable def expectedValue : ℚ :=
  (winnings 2 + winnings 4 + winnings 6 + winnings 8) / 8

theorem expected_value_of_winnings_is_5 :
  expectedValue = 5 := by
  sorry

end DiceGame

end expected_value_of_winnings_is_5_l507_50799


namespace positive_difference_eq_30_l507_50768

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l507_50768


namespace bess_throw_distance_l507_50798

-- Definitions based on the conditions
def bess_throws (x : ℝ) : ℝ := 4 * 2 * x
def holly_throws : ℝ := 5 * 8
def total_throws (x : ℝ) : ℝ := bess_throws x + holly_throws

-- Lean statement for the proof
theorem bess_throw_distance (x : ℝ) (h : total_throws x = 200) : x = 20 :=
by 
  sorry

end bess_throw_distance_l507_50798


namespace xy_product_of_sample_l507_50780

/-- Given a sample {9, 10, 11, x, y} such that the average is 10 and the standard deviation is sqrt(2), 
    prove that the product of x and y is 96. -/
theorem xy_product_of_sample (x y : ℝ) 
  (h_avg : (9 + 10 + 11 + x + y) / 5 = 10)
  (h_stddev : ( (9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2 ) / 5 = 2) :
  x * y = 96 :=
by
  -- Proof goes here
  sorry

end xy_product_of_sample_l507_50780


namespace negation_correct_l507_50714

-- Definitions needed from the conditions:
def is_positive (m : ℝ) : Prop := m > 0
def square (m : ℝ) : ℝ := m * m

-- The original proposition
def original_proposition (m : ℝ) : Prop := is_positive m → square m > 0

-- The negation of the proposition
def negated_proposition (m : ℝ) : Prop := ¬is_positive m → ¬(square m > 0)

-- The theorem to prove that the negated proposition is the negation of the original proposition
theorem negation_correct (m : ℝ) : (original_proposition m) ↔ (negated_proposition m) :=
by
  sorry

end negation_correct_l507_50714


namespace sufficient_not_necessary_condition_l507_50752

theorem sufficient_not_necessary_condition (x : ℝ) : (x^2 - 2 * x < 0) → (|x - 1| < 2) ∧ ¬( (|x - 1| < 2) → (x^2 - 2 * x < 0)) :=
by sorry

end sufficient_not_necessary_condition_l507_50752


namespace additional_savings_in_cents_l507_50770

/-
The book has a cover price of $30.
There are two discount methods to compare:
1. First $5 off, then 25% off.
2. First 25% off, then $5 off.
Prove that the difference in final costs (in cents) between these two discount methods is 125 cents.
-/
def book_price : ℝ := 30
def discount_cash : ℝ := 5
def discount_percentage : ℝ := 0.25

def final_price_apply_cash_first (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (price - cash_discount) * (1 - percentage_discount)

def final_price_apply_percentage_first (price : ℝ) (percentage_discount : ℝ) (cash_discount : ℝ) : ℝ :=
  (price * (1 - percentage_discount)) - cash_discount

def savings_comparison (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (final_price_apply_cash_first price cash_discount percentage_discount) - 
  (final_price_apply_percentage_first price percentage_discount cash_discount)

theorem additional_savings_in_cents : 
  savings_comparison book_price discount_cash discount_percentage * 100 = 125 :=
  by sorry

end additional_savings_in_cents_l507_50770


namespace total_cost_paper_plates_and_cups_l507_50732

theorem total_cost_paper_plates_and_cups :
  ∀ (P C : ℝ), (20 * P + 40 * C = 1.20) → (100 * P + 200 * C = 6.00) := by
  intros P C h
  sorry

end total_cost_paper_plates_and_cups_l507_50732


namespace find_rainy_days_l507_50735

theorem find_rainy_days 
  (n d T H P R : ℤ) 
  (h1 : R + (d - R) = d)
  (h2 : 3 * (d - R) = T)
  (h3 : n * R = H)
  (h4 : T = H + P)
  (hd : 1 ≤ d ∧ d ≤ 31)
  (hR_range : 0 ≤ R ∧ R ≤ d) :
  R = (3 * d - P) / (n + 3) :=
sorry

end find_rainy_days_l507_50735


namespace dave_total_time_l507_50729

variable (W J : ℕ)

-- Given conditions
def time_walked := W = 9
def ratio := J / W = 4 / 3

-- Statement to prove
theorem dave_total_time (time_walked : time_walked W) (ratio : ratio J W) : W + J = 21 := 
by
  sorry

end dave_total_time_l507_50729


namespace triangle_is_right_l507_50763

variable {a b c : ℝ}

theorem triangle_is_right
  (h : a^3 + (Real.sqrt 2 / 4) * b^3 + (Real.sqrt 3 / 9) * c^3 - (Real.sqrt 6 / 2) * a * b * c = 0) :
  (a * a + b * b = c * c) :=
sorry

end triangle_is_right_l507_50763


namespace coconut_grove_yield_l507_50784

theorem coconut_grove_yield (x : ℕ)
  (h1 : ∀ y, y = x + 3 → 60 * y = 60 * (x + 3))
  (h2 : ∀ z, z = x → 120 * z = 120 * x)
  (h3 : ∀ w, w = x - 3 → 180 * w = 180 * (x - 3))
  (avg_yield : 100 = 100)
  (total_trees : 3 * x = (x + 3) + x + (x - 3)) :
  60 * (x + 3) + 120 * x + 180 * (x - 3) = 300 * x →
  x = 6 :=
by
  sorry

end coconut_grove_yield_l507_50784


namespace platform_length_is_correct_l507_50734

noncomputable def length_of_platform (train1_speed_kmph : ℕ) (train2_speed_kmph : ℕ) (cross_time_s : ℕ) (platform_time_s : ℕ) : ℕ :=
  let train1_speed_mps := train1_speed_kmph * 5 / 18
  let train2_speed_mps := train2_speed_kmph * 5 / 18
  let relative_speed := train1_speed_mps + train2_speed_mps
  let total_distance := relative_speed * cross_time_s
  let train1_length := 2 * total_distance / 3
  let platform_length := train1_speed_mps * platform_time_s
  platform_length

theorem platform_length_is_correct : length_of_platform 48 42 12 45 = 600 :=
by
  sorry

end platform_length_is_correct_l507_50734


namespace mayoral_election_l507_50764

theorem mayoral_election :
  ∀ (X Y Z : ℕ), (X = Y + (Y / 2)) → (Y = Z - (2 * Z / 5)) → (Z = 25000) → X = 22500 :=
by
  intros X Y Z h1 h2 h3
  -- Proof here, not necessary for the task
  sorry

end mayoral_election_l507_50764


namespace find_principal_6400_l507_50723

theorem find_principal_6400 (CI SI P : ℝ) (R T : ℝ) 
  (hR : R = 5) (hT : T = 2) 
  (hSI : SI = P * R * T / 100) 
  (hCI : CI = P * (1 + R / 100) ^ T - P) 
  (hDiff : CI - SI = 16) : 
  P = 6400 := 
by 
  sorry

end find_principal_6400_l507_50723
