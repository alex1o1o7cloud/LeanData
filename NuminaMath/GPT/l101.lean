import Mathlib

namespace NUMINAMATH_GPT_transformed_sequence_has_large_element_l101_10168

noncomputable def transformed_value (a : Fin 25 → ℤ) (i : Fin 25) : ℤ :=
  a i + a ((i + 1) % 25)

noncomputable def perform_transformation (a : Fin 25 → ℤ) (n : ℕ) : Fin 25 → ℤ :=
  if n = 0 then a
  else perform_transformation (fun i => transformed_value a i) (n - 1)

theorem transformed_sequence_has_large_element :
  ∀ a : Fin 25 → ℤ,
    (∀ i : Fin 13, a i = 1) →
    (∀ i : Fin 12, a (i + 13) = -1) →
    ∃ i : Fin 25, perform_transformation a 100 i > 10^20 :=
by
  sorry

end NUMINAMATH_GPT_transformed_sequence_has_large_element_l101_10168


namespace NUMINAMATH_GPT_inequality_proof_l101_10124

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a + 1) * (b + 1) * (a + c) * (b + c) > 16 * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l101_10124


namespace NUMINAMATH_GPT_fraction_positive_implies_x_greater_than_seven_l101_10171

variable (x : ℝ)

theorem fraction_positive_implies_x_greater_than_seven (h : -6 / (7 - x) > 0) : x > 7 := by
  sorry

end NUMINAMATH_GPT_fraction_positive_implies_x_greater_than_seven_l101_10171


namespace NUMINAMATH_GPT_allison_upload_rate_l101_10120

theorem allison_upload_rate (x : ℕ) (h1 : 15 * x + 30 * x = 450) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_allison_upload_rate_l101_10120


namespace NUMINAMATH_GPT_sasha_studies_more_avg_4_l101_10108

-- Define the differences recorded over the five days
def differences : List ℤ := [20, 0, 30, -20, -10]

-- Calculate the average difference
def average_difference (diffs : List ℤ) : ℚ :=
  (List.sum diffs : ℚ) / (List.length diffs : ℚ)

-- The statement to prove
theorem sasha_studies_more_avg_4 :
  average_difference differences = 4 := by
  sorry

end NUMINAMATH_GPT_sasha_studies_more_avg_4_l101_10108


namespace NUMINAMATH_GPT_geom_seq_fraction_l101_10183

theorem geom_seq_fraction (a_1 a_2 a_3 a_4 a_5 q : ℝ)
  (h1 : q > 0)
  (h2 : a_2 = q * a_1)
  (h3 : a_3 = q^2 * a_1)
  (h4 : a_4 = q^3 * a_1)
  (h5 : a_5 = q^4 * a_1)
  (h_arith : a_2 - (1/2) * a_3 = (1/2) * a_3 - a_1) :
  (a_3 + a_4) / (a_4 + a_5) = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_fraction_l101_10183


namespace NUMINAMATH_GPT_problem_statement_l101_10112

def f (x : ℝ) : ℝ := x^3 + x^2 + 2

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem problem_statement : odd_function f → f (-2) = -14 := by
  intro h
  sorry

end NUMINAMATH_GPT_problem_statement_l101_10112


namespace NUMINAMATH_GPT_negation_of_inverse_true_l101_10114

variables (P : Prop)

theorem negation_of_inverse_true (h : ¬P → false) : ¬P := by
  sorry

end NUMINAMATH_GPT_negation_of_inverse_true_l101_10114


namespace NUMINAMATH_GPT_no_integer_roots_of_quadratic_l101_10148

theorem no_integer_roots_of_quadratic
  (a b c : ℤ) (f : ℤ → ℤ)
  (h_def : ∀ x, f x = a * x * x + b * x + c)
  (h_a_nonzero : a ≠ 0)
  (h_f0_odd : Odd (f 0))
  (h_f1_odd : Odd (f 1)) :
  ∀ x : ℤ, f x ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_of_quadratic_l101_10148


namespace NUMINAMATH_GPT_find_k_l101_10161

variable (k : ℕ) (hk : k > 0)

theorem find_k (h : (24 - k) / (8 + k) = 1) : k = 8 :=
by sorry

end NUMINAMATH_GPT_find_k_l101_10161


namespace NUMINAMATH_GPT_expression_eval_l101_10199

theorem expression_eval :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
by
  sorry

end NUMINAMATH_GPT_expression_eval_l101_10199


namespace NUMINAMATH_GPT_original_number_input_0_2_l101_10176

theorem original_number_input_0_2 (x : ℝ) (hx : x ≠ 0) (h : (1 / (1 / x - 1) - 1 = -0.75)) : x = 0.2 := 
sorry

end NUMINAMATH_GPT_original_number_input_0_2_l101_10176


namespace NUMINAMATH_GPT_ab_value_is_3360_l101_10173

noncomputable def find_ab (a b : ℤ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧
  (∃ r s : ℤ, 
    (x : ℤ) → 
      (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s)) ∧ 
      (2 * r + s = -a) ∧ 
      (r^2 + 2 * r * s = b) ∧ 
      (r^2 * s = -16 * a))

theorem ab_value_is_3360 (a b : ℤ) (h : find_ab a b) : |a * b| = 3360 :=
sorry

end NUMINAMATH_GPT_ab_value_is_3360_l101_10173


namespace NUMINAMATH_GPT_calculate_molecular_weight_l101_10191

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def num_atoms_C := 3
def num_atoms_H := 6
def num_atoms_O := 1

def molecular_weight (nC nH nO : ℕ) (wC wH wO : ℝ) : ℝ :=
  (nC * wC) + (nH * wH) + (nO * wO)

theorem calculate_molecular_weight :
  molecular_weight num_atoms_C num_atoms_H num_atoms_O atomic_weight_C atomic_weight_H atomic_weight_O = 58.078 :=
by
  sorry

end NUMINAMATH_GPT_calculate_molecular_weight_l101_10191


namespace NUMINAMATH_GPT_weekly_caloric_deficit_l101_10150

-- Define the conditions
def daily_calories (day : String) : Nat :=
  if day = "Saturday" then 3500 else 2500

def daily_burn : Nat := 3000

-- Define the total calories consumed in a week
def total_weekly_consumed : Nat :=
  (2500 * 6) + 3500

-- Define the total calories burned in a week
def total_weekly_burned : Nat :=
  daily_burn * 7

-- Define the weekly deficit
def weekly_deficit : Nat :=
  total_weekly_burned - total_weekly_consumed

-- The proof goal
theorem weekly_caloric_deficit : weekly_deficit = 2500 :=
by
  -- Proof steps would go here; however, per instructions, we use sorry
  sorry

end NUMINAMATH_GPT_weekly_caloric_deficit_l101_10150


namespace NUMINAMATH_GPT_total_birds_in_pet_store_l101_10105

theorem total_birds_in_pet_store
  (number_of_cages : ℕ)
  (parrots_per_cage : ℕ)
  (parakeets_per_cage : ℕ)
  (total_birds_in_cage : ℕ)
  (total_birds : ℕ) :
  number_of_cages = 8 →
  parrots_per_cage = 2 →
  parakeets_per_cage = 7 →
  total_birds_in_cage = parrots_per_cage + parakeets_per_cage →
  total_birds = number_of_cages * total_birds_in_cage →
  total_birds = 72 := by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_birds_in_pet_store_l101_10105


namespace NUMINAMATH_GPT_circle_area_from_tangency_conditions_l101_10140

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - 20 * y^2 = 24

-- Tangency to the x-axis implies the circle's lowest point touches the x-axis
def tangent_to_x_axis (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ r y₀, circle 0 y₀ ∧ y₀ = r

-- The circle is given as having tangency conditions to derive from
theorem circle_area_from_tangency_conditions (circle : ℝ → ℝ → Prop) :
  (∀ x y, circle x y → (x = 0 ∨ hyperbola x y)) →
  tangent_to_x_axis circle →
  ∃ area, area = 504 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_area_from_tangency_conditions_l101_10140


namespace NUMINAMATH_GPT_intersection_is_negative_real_l101_10130

def setA : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x + 1}
def setB : Set ℝ := {y : ℝ | ∃ x : ℝ, y = - x ^ 2}

theorem intersection_is_negative_real :
  setA ∩ setB = {y : ℝ | y ≤ 0} := 
sorry

end NUMINAMATH_GPT_intersection_is_negative_real_l101_10130


namespace NUMINAMATH_GPT_rope_segments_after_folds_l101_10156

theorem rope_segments_after_folds (n : ℕ) : 
  (if n = 1 then 3 else 
   if n = 2 then 5 else 
   if n = 3 then 9 else 2^n + 1) = 2^n + 1 :=
by sorry

end NUMINAMATH_GPT_rope_segments_after_folds_l101_10156


namespace NUMINAMATH_GPT_compute_value_3_std_devs_less_than_mean_l101_10132

noncomputable def mean : ℝ := 15
noncomputable def std_dev : ℝ := 1.5
noncomputable def skewness : ℝ := 0.5
noncomputable def kurtosis : ℝ := 0.6

theorem compute_value_3_std_devs_less_than_mean : 
  ¬∃ (value : ℝ), value = mean - 3 * std_dev :=
sorry

end NUMINAMATH_GPT_compute_value_3_std_devs_less_than_mean_l101_10132


namespace NUMINAMATH_GPT_find_a_l101_10178

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : deriv (f a) (-1) = 4) : 
  a = 10 / 3 :=
sorry

end NUMINAMATH_GPT_find_a_l101_10178


namespace NUMINAMATH_GPT_find_number_l101_10136

theorem find_number (x : ℚ) (h : x / 11 + 156 = 178) : x = 242 :=
sorry

end NUMINAMATH_GPT_find_number_l101_10136


namespace NUMINAMATH_GPT_tim_movie_marathon_l101_10110

variables (first_movie second_movie third_movie fourth_movie fifth_movie sixth_movie seventh_movie : ℝ)

/-- Tim's movie marathon --/
theorem tim_movie_marathon
  (first_movie_duration : first_movie = 2)
  (second_movie_duration : second_movie = 1.5 * first_movie)
  (third_movie_duration : third_movie = 0.8 * (first_movie + second_movie))
  (fourth_movie_duration : fourth_movie = 2 * second_movie)
  (fifth_movie_duration : fifth_movie = third_movie - 0.5)
  (sixth_movie_duration : sixth_movie = (second_movie + fourth_movie) / 2)
  (seventh_movie_duration : seventh_movie = 45 / fifth_movie) :
  first_movie + second_movie + third_movie + fourth_movie + fifth_movie + sixth_movie + seventh_movie = 35.8571 :=
sorry

end NUMINAMATH_GPT_tim_movie_marathon_l101_10110


namespace NUMINAMATH_GPT_solve_diamond_eq_l101_10158

noncomputable def diamond_op (a b : ℝ) := a / b

theorem solve_diamond_eq (x : ℝ) (h : x ≠ 0) : diamond_op 2023 (diamond_op 7 x) = 150 ↔ x = 1050 / 2023 := by
  sorry

end NUMINAMATH_GPT_solve_diamond_eq_l101_10158


namespace NUMINAMATH_GPT_steves_initial_emails_l101_10119

theorem steves_initial_emails (E : ℝ) (ht : E / 2 = (0.6 * E) + 120) : E = 400 :=
  by sorry

end NUMINAMATH_GPT_steves_initial_emails_l101_10119


namespace NUMINAMATH_GPT_general_term_formula_l101_10172

variable {a_n : ℕ → ℕ} -- Sequence {a_n}
variable {S_n : ℕ → ℕ} -- Sum of the first n terms

-- Condition given in the problem
def S_n_condition (n : ℕ) : ℕ :=
  2 * n^2 + n

theorem general_term_formula (n : ℕ) (h₀ : ∀ (n : ℕ), S_n n = 2 * n^2 + n) :
  a_n n = 4 * n - 1 :=
sorry

end NUMINAMATH_GPT_general_term_formula_l101_10172


namespace NUMINAMATH_GPT_total_candies_l101_10125

theorem total_candies (Linda_candies Chloe_candies : ℕ) (h1 : Linda_candies = 34) (h2 : Chloe_candies = 28) :
  Linda_candies + Chloe_candies = 62 := by
  sorry

end NUMINAMATH_GPT_total_candies_l101_10125


namespace NUMINAMATH_GPT_oshea_large_planters_l101_10143

theorem oshea_large_planters {total_seeds small_planter_capacity num_small_planters large_planter_capacity : ℕ} 
  (h1 : total_seeds = 200)
  (h2 : small_planter_capacity = 4)
  (h3 : num_small_planters = 30)
  (h4 : large_planter_capacity = 20) :
  (total_seeds - num_small_planters * small_planter_capacity) / large_planter_capacity = 4 :=
by
  sorry

end NUMINAMATH_GPT_oshea_large_planters_l101_10143


namespace NUMINAMATH_GPT_iter_f_eq_l101_10186

namespace IteratedFunction

def f (n : ℕ) (x : ℕ) : ℕ :=
  if 2 * x <= n then
    2 * x
  else
    2 * n - 2 * x + 1

def iter_f (n m : ℕ) (x : ℕ) : ℕ :=
  (Nat.iterate (f n) m) x

variables (n m : ℕ) (S : Fin n.succ → Fin n.succ)

theorem iter_f_eq (h : iter_f n m 1 = 1) (k : Fin n.succ) :
  iter_f n m k = k := by
  sorry

end IteratedFunction

end NUMINAMATH_GPT_iter_f_eq_l101_10186


namespace NUMINAMATH_GPT_certain_number_correct_l101_10160

theorem certain_number_correct : 
  (h1 : 29.94 / 1.45 = 17.9) -> (2994 / 14.5 = 1790) :=
by 
  sorry

end NUMINAMATH_GPT_certain_number_correct_l101_10160


namespace NUMINAMATH_GPT_team_size_per_team_l101_10117

theorem team_size_per_team (managers employees teams people_per_team : ℕ) 
  (h1 : managers = 23) 
  (h2 : employees = 7) 
  (h3 : teams = 6) 
  (h4 : people_per_team = (managers + employees) / teams) : 
  people_per_team = 5 :=
by 
  sorry

end NUMINAMATH_GPT_team_size_per_team_l101_10117


namespace NUMINAMATH_GPT_complex_number_in_third_quadrant_l101_10147

open Complex

noncomputable def complex_number : ℂ := (1 - 3 * I) / (1 + 2 * I)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem complex_number_in_third_quadrant : in_third_quadrant complex_number :=
sorry

end NUMINAMATH_GPT_complex_number_in_third_quadrant_l101_10147


namespace NUMINAMATH_GPT_find_some_expression_l101_10104

noncomputable def problem_statement : Prop :=
  ∃ (some_expression : ℝ), 
    (5 + 7 / 12 = 6 - some_expression) ∧ 
    (some_expression = 0.4167)

theorem find_some_expression : problem_statement := 
  sorry

end NUMINAMATH_GPT_find_some_expression_l101_10104


namespace NUMINAMATH_GPT_total_length_of_pencil_l101_10146

def purple := 3
def black := 2
def blue := 1
def total_length := purple + black + blue

theorem total_length_of_pencil : total_length = 6 := 
by 
  sorry -- proof not needed

end NUMINAMATH_GPT_total_length_of_pencil_l101_10146


namespace NUMINAMATH_GPT_prod_sum_rel_prime_l101_10133

theorem prod_sum_rel_prime (a b : ℕ) 
  (h1 : a * b + a + b = 119)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 25)
  (h4 : b < 25) : 
  a + b = 27 := 
sorry

end NUMINAMATH_GPT_prod_sum_rel_prime_l101_10133


namespace NUMINAMATH_GPT_farmer_shipped_67_dozens_l101_10144

def pomelos_in_box (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 20 else if box_type = "large" then 30 else 0

def total_pomelos_last_week : ℕ := 360

def boxes_this_week (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 8 else if box_type = "large" then 7 else 0

def damage_boxes (box_type : String) : ℕ :=
  if box_type = "small" then 3 else if box_type = "medium" then 2 else if box_type = "large" then 2 else 0

def loss_percentage (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 15 else if box_type = "large" then 20 else 0

def total_pomelos_shipped_this_week : ℕ :=
  (boxes_this_week "small") * (pomelos_in_box "small") +
  (boxes_this_week "medium") * (pomelos_in_box "medium") +
  (boxes_this_week "large") * (pomelos_in_box "large")

def total_pomelos_lost_this_week : ℕ :=
  (damage_boxes "small") * (pomelos_in_box "small") * (loss_percentage "small") / 100 +
  (damage_boxes "medium") * (pomelos_in_box "medium") * (loss_percentage "medium") / 100 +
  (damage_boxes "large") * (pomelos_in_box "large") * (loss_percentage "large") / 100

def total_pomelos_shipped_successfully_this_week : ℕ :=
  total_pomelos_shipped_this_week - total_pomelos_lost_this_week

def total_pomelos_for_both_weeks : ℕ :=
  total_pomelos_last_week + total_pomelos_shipped_successfully_this_week

def total_dozens_shipped : ℕ :=
  total_pomelos_for_both_weeks / 12

theorem farmer_shipped_67_dozens :
  total_dozens_shipped = 67 := 
by sorry

end NUMINAMATH_GPT_farmer_shipped_67_dozens_l101_10144


namespace NUMINAMATH_GPT_calculate_expected_value_of_S_l101_10151

-- Define the problem context
variables (boys girls : ℕ)
variable (boy_girl_pair_at_start : Bool)

-- Define the expected value function
def expected_S (boys girls : ℕ) (boy_girl_pair_at_start : Bool) : ℕ :=
  if boy_girl_pair_at_start then 10 else sorry  -- we only consider the given scenario

-- The theorem to prove
theorem calculate_expected_value_of_S :
  expected_S 5 15 true = 10 :=
by
  -- proof needs to be filled in
  sorry

end NUMINAMATH_GPT_calculate_expected_value_of_S_l101_10151


namespace NUMINAMATH_GPT_solution_set_absolute_value_l101_10196

theorem solution_set_absolute_value (x : ℝ) : 
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solution_set_absolute_value_l101_10196


namespace NUMINAMATH_GPT_value_at_one_positive_l101_10159

-- Define the conditions
variable {f : ℝ → ℝ} 

-- f is a monotonically increasing function
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement: proving that f(1) > 0
theorem value_at_one_positive (h1 : monotone_increasing f) (h2 : odd_function f) : f 1 > 0 :=
sorry

end NUMINAMATH_GPT_value_at_one_positive_l101_10159


namespace NUMINAMATH_GPT_k_h_5_eq_148_l101_10118

def h (x : ℤ) : ℤ := 4 * x + 6
def k (x : ℤ) : ℤ := 6 * x - 8

theorem k_h_5_eq_148 : k (h 5) = 148 := by
  sorry

end NUMINAMATH_GPT_k_h_5_eq_148_l101_10118


namespace NUMINAMATH_GPT_fifteenth_number_in_base_8_l101_10190

theorem fifteenth_number_in_base_8 : (15 : ℕ) = 1 * 8 + 7 := 
sorry

end NUMINAMATH_GPT_fifteenth_number_in_base_8_l101_10190


namespace NUMINAMATH_GPT_no_solution_iff_n_eq_minus_half_l101_10139

theorem no_solution_iff_n_eq_minus_half (n x y z : ℝ) :
  (¬∃ x y z : ℝ, 2 * n * x + y = 2 ∧ n * y + z = 2 ∧ x + 2 * n * z = 2) ↔ n = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_iff_n_eq_minus_half_l101_10139


namespace NUMINAMATH_GPT_initial_incorrect_average_l101_10192

theorem initial_incorrect_average :
  let avg_correct := 24
  let incorrect_insertion := 26
  let correct_insertion := 76
  let n := 10  
  let correct_sum := avg_correct * n
  let incorrect_sum := correct_sum - correct_insertion + incorrect_insertion   
  avg_correct * n - correct_insertion + incorrect_insertion = incorrect_sum →
  incorrect_sum / n = 19 :=
by 
  sorry

end NUMINAMATH_GPT_initial_incorrect_average_l101_10192


namespace NUMINAMATH_GPT_number_of_rocks_chosen_l101_10113

open Classical

theorem number_of_rocks_chosen
  (total_rocks : ℕ)
  (slate_rocks : ℕ)
  (pumice_rocks : ℕ)
  (granite_rocks : ℕ)
  (probability_both_slate : ℚ) :
  total_rocks = 44 →
  slate_rocks = 14 →
  pumice_rocks = 20 →
  granite_rocks = 10 →
  probability_both_slate = (14 / 44) * (13 / 43) →
  2 = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_number_of_rocks_chosen_l101_10113


namespace NUMINAMATH_GPT_proof_expr_is_neg_four_ninths_l101_10102

noncomputable def example_expr : ℚ := (-3 / 2) ^ 2021 * (2 / 3) ^ 2023

theorem proof_expr_is_neg_four_ninths : example_expr = (-4 / 9) := 
by 
  -- Here the proof would be placed
  sorry

end NUMINAMATH_GPT_proof_expr_is_neg_four_ninths_l101_10102


namespace NUMINAMATH_GPT_expand_product_l101_10155

-- Definitions of the polynomial functions
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 + x + 1

-- Statement of the theorem
theorem expand_product : ∀ x : ℝ, (f x) * (g x) = x^3 + 4*x^2 + 4*x + 3 :=
by
  -- Proof goes here, but is omitted for the statement only
  sorry

end NUMINAMATH_GPT_expand_product_l101_10155


namespace NUMINAMATH_GPT_roots_diff_l101_10167

theorem roots_diff (m : ℝ) : 
  (∃ α β : ℝ, 2 * α * α - m * α - 8 = 0 ∧ 
              2 * β * β - m * β - 8 = 0 ∧ 
              α ≠ β ∧ 
              α - β = m - 1) ↔ (m = 6 ∨ m = -10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_roots_diff_l101_10167


namespace NUMINAMATH_GPT_frank_spends_more_l101_10101

def cost_computer_table : ℕ := 140
def cost_computer_chair : ℕ := 100
def cost_joystick : ℕ := 20
def frank_share_joystick : ℕ := cost_joystick / 4
def eman_share_joystick : ℕ := cost_joystick * 3 / 4

def total_spent_frank : ℕ := cost_computer_table + frank_share_joystick
def total_spent_eman : ℕ := cost_computer_chair + eman_share_joystick

theorem frank_spends_more : total_spent_frank - total_spent_eman = 30 :=
by
  sorry

end NUMINAMATH_GPT_frank_spends_more_l101_10101


namespace NUMINAMATH_GPT_solve_quadratic_expr_l101_10195

theorem solve_quadratic_expr (x : ℝ) (h : 2 * x^2 - 5 = 11) : 
  4 * x^2 + 4 * x + 1 = 33 + 8 * Real.sqrt 2 ∨ 4 * x^2 + 4 * x + 1 = 33 - 8 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_solve_quadratic_expr_l101_10195


namespace NUMINAMATH_GPT_find_abc_l101_10134

theorem find_abc :
  ∃ a b c : ℝ, 
    -- Conditions
    (a + b + c = 12) ∧ 
    (2 * b = a + c) ∧ 
    ((a + 2) * (c + 5) = (b + 2) * (b + 2)) ∧ 
    -- Correct answers
    ((a = 1 ∧ b = 4 ∧ c = 7) ∨ 
     (a = 10 ∧ b = 4 ∧ c = -2)) := 
  by 
    sorry

end NUMINAMATH_GPT_find_abc_l101_10134


namespace NUMINAMATH_GPT_smallest_possible_value_of_a_largest_possible_value_of_a_l101_10135

-- Define that a is a positive integer and there are exactly 10 perfect squares greater than a and less than 2a

variable (a : ℕ) (h1 : a > 0)
variable (h2 : ∃ (s : ℕ) (t : ℕ), s + 10 = t ∧ (s^2 > a) ∧ (s + 9)^2 < 2 * a ∧ (t^2 - 10) + 9 < 2 * a)

-- Prove the smallest value of a
theorem smallest_possible_value_of_a : a = 481 :=
by sorry

-- Prove the largest value of a
theorem largest_possible_value_of_a : a = 684 :=
by sorry

end NUMINAMATH_GPT_smallest_possible_value_of_a_largest_possible_value_of_a_l101_10135


namespace NUMINAMATH_GPT_find_y_l101_10182

theorem find_y (x : ℝ) (y : ℝ) (h : (3 + y)^5 = (1 + 3 * y)^4) (hx : x = 1.5) : y = 1.5 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_find_y_l101_10182


namespace NUMINAMATH_GPT_proof_problem_l101_10165

-- Define sets
def N_plus : Set ℕ := {x | x > 0}  -- Positive integers
def Z : Set ℤ := {x | true}        -- Integers
def Q : Set ℚ := {x | true}        -- Rational numbers

-- Lean problem statement
theorem proof_problem : 
  (0 ∉ N_plus) ∧ 
  (((-1)^3 : ℤ) ∈ Z) ∧ 
  (π ∉ Q) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l101_10165


namespace NUMINAMATH_GPT_real_root_exists_for_all_K_l101_10137

theorem real_root_exists_for_all_K (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
sorry

end NUMINAMATH_GPT_real_root_exists_for_all_K_l101_10137


namespace NUMINAMATH_GPT_minimum_value_of_f_l101_10189

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l101_10189


namespace NUMINAMATH_GPT_average_of_xyz_l101_10184

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 4 :=
sorry

end NUMINAMATH_GPT_average_of_xyz_l101_10184


namespace NUMINAMATH_GPT_part_a_l101_10170

theorem part_a (x y : ℝ) (hx : 1 > x ∧ x ≥ 0) (hy : 1 > y ∧ y ≥ 0) : 
  ⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋ := sorry

end NUMINAMATH_GPT_part_a_l101_10170


namespace NUMINAMATH_GPT_ratio_of_first_term_to_common_difference_l101_10152

theorem ratio_of_first_term_to_common_difference 
  (a d : ℤ) 
  (h : 15 * a + 105 * d = 3 * (10 * a + 45 * d)) :
  a = -2 * d :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_first_term_to_common_difference_l101_10152


namespace NUMINAMATH_GPT_system_of_equations_correct_l101_10141

variable (x y : ℝ)

def correct_system_of_equations : Prop :=
  (3 / 60) * x + (5 / 60) * y = 1.2 ∧ x + y = 16

theorem system_of_equations_correct :
  correct_system_of_equations x y :=
sorry

end NUMINAMATH_GPT_system_of_equations_correct_l101_10141


namespace NUMINAMATH_GPT_jenny_sold_boxes_l101_10197

-- Given conditions as definitions
def cases : ℕ := 3
def boxes_per_case : ℕ := 8

-- Mathematically equivalent proof problem
theorem jenny_sold_boxes : cases * boxes_per_case = 24 := by
  sorry

end NUMINAMATH_GPT_jenny_sold_boxes_l101_10197


namespace NUMINAMATH_GPT_find_x_plus_y_l101_10127

theorem find_x_plus_y (x y : ℝ) (h1 : |x| + x + y = 12) (h2 : x + |y| - y = 16) :
  x + y = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l101_10127


namespace NUMINAMATH_GPT_plane_equation_proof_l101_10187

-- Define the parametric representation of the plane
def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - t, 1 + 2 * s, 4 - s + 3 * t)

-- Define the plane equation form
def plane_equation (x y z : ℝ) (A B C D : ℤ) : Prop :=
  (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + (D : ℝ) = 0

-- Define the normal vector derived from the cross product
def normal_vector : ℝ × ℝ × ℝ := (6, -5, 2)

-- Define the initial point used to calculate D
def initial_point : ℝ × ℝ × ℝ := (2, 1, 4)

-- Proposition to prove the equation of the plane
theorem plane_equation_proof :
  ∃ (A B C D : ℤ), A = 6 ∧ B = -5 ∧ C = 2 ∧ D = -15 ∧
    ∀ x y z : ℝ, plane_equation x y z A B C D ↔
      ∃ s t : ℝ, plane_parametric s t = (x, y, z) :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_proof_l101_10187


namespace NUMINAMATH_GPT_Gerald_initial_notebooks_l101_10162

variable (J G : ℕ)

theorem Gerald_initial_notebooks (h1 : J = G + 13)
    (h2 : J - 5 - 6 = 10) :
    G = 8 :=
sorry

end NUMINAMATH_GPT_Gerald_initial_notebooks_l101_10162


namespace NUMINAMATH_GPT_noah_sales_value_l101_10129

def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def price_large : ℕ := 60
def price_small : ℕ := 30

def this_month_large_sales : ℕ := 2 * last_month_large_sales
def this_month_small_sales : ℕ := 2 * last_month_small_sales

def this_month_large_sales_value : ℕ := this_month_large_sales * price_large
def this_month_small_sales_value : ℕ := this_month_small_sales * price_small

def this_month_total_sales : ℕ := this_month_large_sales_value + this_month_small_sales_value

theorem noah_sales_value :
  this_month_total_sales = 1200 :=
by
  sorry

end NUMINAMATH_GPT_noah_sales_value_l101_10129


namespace NUMINAMATH_GPT_integer_bases_not_divisible_by_5_l101_10109

theorem integer_bases_not_divisible_by_5 :
  ∀ b ∈ ({3, 5, 7, 10, 12} : Set ℕ), (b - 1) ^ 2 % 5 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_integer_bases_not_divisible_by_5_l101_10109


namespace NUMINAMATH_GPT_dishonest_dealer_profit_percent_l101_10142

theorem dishonest_dealer_profit_percent
  (C : ℝ) -- assumed cost price for 1 kg of goods
  (SP_600 : ℝ := C) -- selling price for 600 grams is equal to the cost price for 1 kg
  (CP_600 : ℝ := 0.6 * C) -- cost price for 600 grams
  : (SP_600 - CP_600) / CP_600 * 100 = 66.67 := by
  sorry

end NUMINAMATH_GPT_dishonest_dealer_profit_percent_l101_10142


namespace NUMINAMATH_GPT_blueberries_in_blue_box_l101_10149

theorem blueberries_in_blue_box (S B : ℕ) (h1 : S - B = 15) (h2 : S + B = 87) : B = 36 :=
by sorry

end NUMINAMATH_GPT_blueberries_in_blue_box_l101_10149


namespace NUMINAMATH_GPT_profit_percentage_l101_10153

theorem profit_percentage (cost_price selling_price : ℝ) (h₁ : cost_price = 32) (h₂ : selling_price = 56) : 
  ((selling_price - cost_price) / cost_price) * 100 = 75 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l101_10153


namespace NUMINAMATH_GPT_polynomial_divisible_by_24_l101_10193

theorem polynomial_divisible_by_24 (n : ℤ) : 24 ∣ (n^4 + 6 * n^3 + 11 * n^2 + 6 * n) :=
sorry

end NUMINAMATH_GPT_polynomial_divisible_by_24_l101_10193


namespace NUMINAMATH_GPT_time_difference_l101_10188

/-
Malcolm's speed: 5 minutes per mile
Joshua's speed: 7 minutes per mile
Race length: 12 miles
Question: Prove that the time difference between Joshua crossing the finish line after Malcolm is 24 minutes
-/
noncomputable def time_taken (speed: ℕ) (distance: ℕ) : ℕ :=
  speed * distance

theorem time_difference :
  let malcolm_speed := 5
  let joshua_speed := 7
  let race_length := 12
  let malcolm_time := time_taken malcolm_speed race_length
  let joshua_time := time_taken joshua_speed race_length
  malcolm_time < joshua_time →
  joshua_time - malcolm_time = 24 :=
by
  intros malcolm_speed joshua_speed race_length malcolm_time joshua_time malcolm_time_lt_joshua_time
  sorry

end NUMINAMATH_GPT_time_difference_l101_10188


namespace NUMINAMATH_GPT_translation_correct_l101_10154

theorem translation_correct : 
  ∀ (x y : ℝ), (y = -(x-1)^2 + 3) → (x, y) = (0, 0) ↔ (x - 1, y - 3) = (0, 0) :=
by 
  sorry

end NUMINAMATH_GPT_translation_correct_l101_10154


namespace NUMINAMATH_GPT_polynomial_factors_l101_10174

theorem polynomial_factors (h k : ℤ)
  (h1 : 3 * (-2)^4 - 2 * h * (-2)^2 + h * (-2) + k = 0)
  (h2 : 3 * 1^4 - 2 * h * 1^2 + h * 1 + k = 0)
  (h3 : 3 * (-3)^4 - 2 * h * (-3)^2 + h * (-3) + k = 0) :
  |3 * h - 2 * k| = 11 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factors_l101_10174


namespace NUMINAMATH_GPT_problem_statement_l101_10145

theorem problem_statement (x : ℝ) (h : x + x⁻¹ = 3) : x^7 - 6 * x^5 + 5 * x^3 - x = 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l101_10145


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l101_10169

def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l101_10169


namespace NUMINAMATH_GPT_cans_restocked_after_second_day_l101_10138

theorem cans_restocked_after_second_day :
  let initial_cans := 2000
  let first_day_taken := 500 
  let first_day_restock := 1500
  let second_day_taken := 1000 * 2
  let total_given_away := 2500
  let remaining_after_second_day_before_restock := initial_cans - first_day_taken + first_day_restock - second_day_taken
  (total_given_away - remaining_after_second_day_before_restock) = 1500 := 
by {
  sorry
}

end NUMINAMATH_GPT_cans_restocked_after_second_day_l101_10138


namespace NUMINAMATH_GPT_cost_price_of_watch_l101_10116

-- Let C be the cost price of the watch
variable (C : ℝ)

-- Conditions: The selling price at a loss of 8% and the selling price with a gain of 4% if sold for Rs. 140 more
axiom loss_condition : 0.92 * C + 140 = 1.04 * C

-- Objective: Prove that C = 1166.67
theorem cost_price_of_watch : C = 1166.67 :=
by
  have h := loss_condition
  sorry

end NUMINAMATH_GPT_cost_price_of_watch_l101_10116


namespace NUMINAMATH_GPT_value_of_f_at_2_l101_10123

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem value_of_f_at_2 : f 2 = 3 := sorry

end NUMINAMATH_GPT_value_of_f_at_2_l101_10123


namespace NUMINAMATH_GPT_set_operations_l101_10177

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6})
variable (hA : A = {2, 4, 5})
variable (hB : B = {1, 2, 5})

theorem set_operations :
  (A ∩ B = {2, 5}) ∧ (A ∪ (U \ B) = {2, 3, 4, 5, 6}) :=
by
  sorry

end NUMINAMATH_GPT_set_operations_l101_10177


namespace NUMINAMATH_GPT_sum_of_factorization_constants_l101_10157

theorem sum_of_factorization_constants (p q r s t : ℤ) (y : ℤ) :
  (512 * y ^ 3 + 27 = (p * y + q) * (r * y ^ 2 + s * y + t)) →
  p + q + r + s + t = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_factorization_constants_l101_10157


namespace NUMINAMATH_GPT_ratio_t_q_l101_10180

theorem ratio_t_q (q r s t : ℚ) (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) : 
  t / q = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_t_q_l101_10180


namespace NUMINAMATH_GPT_problem_statement_l101_10106

noncomputable def f1 (x : ℝ) : ℝ := x ^ 2

noncomputable def f2 (x : ℝ) : ℝ := 8 / x

noncomputable def f (x : ℝ) : ℝ := f1 x + f2 x

theorem problem_statement (a : ℝ) (h : a > 3) : 
  ∃ x1 x2 x3 : ℝ, 
  (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) ∧ 
  (f x1 = f a ∧ f x2 = f a ∧ f x3 = f a) ∧ 
  (x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0) := 
sorry

end NUMINAMATH_GPT_problem_statement_l101_10106


namespace NUMINAMATH_GPT_average_age_of_large_family_is_correct_l101_10175

def average_age_of_family 
  (num_grandparents : ℕ) (avg_age_grandparents : ℕ) 
  (num_parents : ℕ) (avg_age_parents : ℕ) 
  (num_children : ℕ) (avg_age_children : ℕ) 
  (num_siblings : ℕ) (avg_age_siblings : ℕ)
  (num_cousins : ℕ) (avg_age_cousins : ℕ)
  (num_aunts : ℕ) (avg_age_aunts : ℕ) : ℕ := 
  let total_age := num_grandparents * avg_age_grandparents + 
                   num_parents * avg_age_parents + 
                   num_children * avg_age_children + 
                   num_siblings * avg_age_siblings + 
                   num_cousins * avg_age_cousins + 
                   num_aunts * avg_age_aunts
  let total_family_members := num_grandparents + num_parents + num_children + num_siblings + num_cousins + num_aunts
  (total_age : ℕ) / total_family_members

theorem average_age_of_large_family_is_correct :
  average_age_of_family 4 67 3 41 5 8 2 35 3 22 2 45 = 35 := 
by 
  sorry

end NUMINAMATH_GPT_average_age_of_large_family_is_correct_l101_10175


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_a7_div2_eq_10_l101_10163

theorem arithmetic_sequence_a4_a7_div2_eq_10 (a : ℕ → ℝ) (h : a 4 + a 6 = 20) : (a 3 + a 6) / 2 = 10 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_a7_div2_eq_10_l101_10163


namespace NUMINAMATH_GPT_range_of_a_l101_10181

theorem range_of_a (a : ℝ) (h_decreasing : ∀ x y : ℝ, x < y → (a-1)^x > (a-1)^y) : 1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l101_10181


namespace NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l101_10131

variable (M W : ℕ) -- M represents the amount of milk, W represents the amount of water

theorem initial_ratio_of_milk_to_water (h1 : M + W = 45) (h2 : 8 * M = 9 * (W + 23)) :
  M / W = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l101_10131


namespace NUMINAMATH_GPT_dog_probability_l101_10111

def prob_machine_A_transforms_cat_to_dog : ℚ := 1 / 3
def prob_machine_B_transforms_cat_to_dog : ℚ := 2 / 5
def prob_machine_C_transforms_cat_to_dog : ℚ := 1 / 4

def prob_cat_remains_after_A : ℚ := 1 - prob_machine_A_transforms_cat_to_dog
def prob_cat_remains_after_B : ℚ := 1 - prob_machine_B_transforms_cat_to_dog
def prob_cat_remains_after_C : ℚ := 1 - prob_machine_C_transforms_cat_to_dog

def prob_cat_remains : ℚ := prob_cat_remains_after_A * prob_cat_remains_after_B * prob_cat_remains_after_C

def prob_dog_out_of_C : ℚ := 1 - prob_cat_remains

theorem dog_probability : prob_dog_out_of_C = 7 / 10 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_dog_probability_l101_10111


namespace NUMINAMATH_GPT_parking_spots_full_iff_num_sequences_l101_10164

noncomputable def num_parking_sequences (n : ℕ) : ℕ :=
  (n + 1) ^ (n - 1)

-- Statement of the theorem
theorem parking_spots_full_iff_num_sequences (n : ℕ) :
  ∀ (a : ℕ → ℕ), (∀ (i : ℕ), i < n → a i ≤ n) → 
  (∀ (j : ℕ), j ≤ n → (∃ i, i < n ∧ a i = j)) ↔ 
  num_parking_sequences n = (n + 1) ^ (n - 1) :=
sorry

end NUMINAMATH_GPT_parking_spots_full_iff_num_sequences_l101_10164


namespace NUMINAMATH_GPT_one_third_of_four_l101_10103

theorem one_third_of_four : (1/3) * 4 = 2 :=
by
  sorry

end NUMINAMATH_GPT_one_third_of_four_l101_10103


namespace NUMINAMATH_GPT_minimum_value_expression_l101_10179

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l101_10179


namespace NUMINAMATH_GPT_symmetric_line_equation_l101_10107

-- Define the original line as an equation in ℝ².
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define the line of symmetry.
def line_of_symmetry (x : ℝ) : Prop := x = 1

-- The theorem stating the equation of the symmetric line.
theorem symmetric_line_equation (x y : ℝ) :
  original_line x y → line_of_symmetry x → (x + 2 * y - 3 = 0) :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_l101_10107


namespace NUMINAMATH_GPT_thomas_task_completion_l101_10128

theorem thomas_task_completion :
  (∃ T E : ℝ, (1 / T + 1 / E = 1 / 8) ∧ (13 / T + 6 / E = 1)) →
  ∃ T : ℝ, T = 14 :=
by
  sorry

end NUMINAMATH_GPT_thomas_task_completion_l101_10128


namespace NUMINAMATH_GPT_AmpersandDoubleCalculation_l101_10166

def ampersand (x : Int) : Int := 7 - x
def doubleAmpersand (x : Int) : Int := (x - 7)

theorem AmpersandDoubleCalculation : doubleAmpersand (ampersand 12) = -12 :=
by
  -- This is where the proof would go, which shows the steps described in the solution.
  sorry

end NUMINAMATH_GPT_AmpersandDoubleCalculation_l101_10166


namespace NUMINAMATH_GPT_parallelogram_midpoints_XY_square_l101_10185

theorem parallelogram_midpoints_XY_square (A B C D X Y : ℝ)
  (AB CD : ℝ) (BC DA : ℝ) (angle_D : ℝ)
  (mid_X : X = (B + C) / 2) (mid_Y : Y = (D + A) / 2)
  (h1: AB = 10) (h2: BC = 17) (h3: CD = 10) (h4 : angle_D = 60) :
  (XY ^ 2 = 219 / 4) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_midpoints_XY_square_l101_10185


namespace NUMINAMATH_GPT_root_value_l101_10122

theorem root_value (m : ℝ) (h : 2 * m^2 - 7 * m + 1 = 0) : m * (2 * m - 7) + 5 = 4 := by
  sorry

end NUMINAMATH_GPT_root_value_l101_10122


namespace NUMINAMATH_GPT_abs_eq_neg_of_le_zero_l101_10115

theorem abs_eq_neg_of_le_zero (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end NUMINAMATH_GPT_abs_eq_neg_of_le_zero_l101_10115


namespace NUMINAMATH_GPT_compute_expression_l101_10198

theorem compute_expression :
  6 * (2 / 3)^4 - 1 / 6 = 55 / 54 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l101_10198


namespace NUMINAMATH_GPT_must_divisor_of_a_l101_10126

-- The statement
theorem must_divisor_of_a (a b c d : ℕ) (h1 : Nat.gcd a b = 18)
    (h2 : Nat.gcd b c = 45) (h3 : Nat.gcd c d = 60) (h4 : 90 < Nat.gcd d a ∧ Nat.gcd d a < 120) : 
    5 ∣ a := 
sorry

end NUMINAMATH_GPT_must_divisor_of_a_l101_10126


namespace NUMINAMATH_GPT_horses_added_l101_10121

-- Define the problem parameters and conditions.
def horses_initial := 3
def water_per_horse_drinking_per_day := 5
def water_per_horse_bathing_per_day := 2
def days := 28
def total_water := 1568

-- Define the assumption based on the given problem.
def total_water_per_horse_per_day := water_per_horse_drinking_per_day + water_per_horse_bathing_per_day
def total_water_initial_horses := horses_initial * total_water_per_horse_per_day * days
def water_for_new_horses := total_water - total_water_initial_horses
def daily_water_consumption_new_horses := water_for_new_horses / days
def number_of_new_horses := daily_water_consumption_new_horses / total_water_per_horse_per_day

-- The theorem to prove number of horses added.
theorem horses_added : number_of_new_horses = 5 := 
  by {
    -- This is where you would put the proof steps.
    sorry -- skipping the proof for now
  }

end NUMINAMATH_GPT_horses_added_l101_10121


namespace NUMINAMATH_GPT_third_sec_second_chap_more_than_first_sec_third_chap_l101_10194

-- Define the page lengths for each section in each chapter
def first_chapter : List ℕ := [20, 10, 30]
def second_chapter : List ℕ := [5, 12, 8, 22]
def third_chapter : List ℕ := [7, 11]

-- Define the specific sections of interest
def third_section_second_chapter := second_chapter[2]  -- 8
def first_section_third_chapter := third_chapter[0]   -- 7

-- The theorem we want to prove
theorem third_sec_second_chap_more_than_first_sec_third_chap :
  third_section_second_chapter - first_section_third_chapter = 1 :=
by
  -- Sorry is used here to skip the proof.
  sorry

end NUMINAMATH_GPT_third_sec_second_chap_more_than_first_sec_third_chap_l101_10194


namespace NUMINAMATH_GPT_mixed_oil_rate_l101_10100

theorem mixed_oil_rate :
  let oil1 := (10, 50)
  let oil2 := (5, 68)
  let oil3 := (8, 42)
  let oil4 := (7, 62)
  let oil5 := (12, 55)
  let oil6 := (6, 75)
  let total_cost := oil1.1 * oil1.2 + oil2.1 * oil2.2 + oil3.1 * oil3.2 + oil4.1 * oil4.2 + oil5.1 * oil5.2 + oil6.1 * oil6.2
  let total_volume := oil1.1 + oil2.1 + oil3.1 + oil4.1 + oil5.1 + oil6.1
  (total_cost / total_volume : ℝ) = 56.67 :=
by
  sorry

end NUMINAMATH_GPT_mixed_oil_rate_l101_10100
