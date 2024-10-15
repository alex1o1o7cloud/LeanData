import Mathlib

namespace NUMINAMATH_GPT_negation_of_universal_l663_66392
-- Import the Mathlib library to provide the necessary mathematical background

-- State the theorem that we want to prove. This will state that the negation of the universal proposition is an existential proposition
theorem negation_of_universal :
  (¬ (∀ x : ℝ, x > 0)) ↔ (∃ x : ℝ, x ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_universal_l663_66392


namespace NUMINAMATH_GPT_geometric_sequence_a_l663_66300

theorem geometric_sequence_a (a : ℝ) (h1 : a > 0) (h2 : ∃ r : ℝ, 280 * r = a ∧ a * r = 180 / 49) :
  a = 32.07 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_a_l663_66300


namespace NUMINAMATH_GPT_age_ratio_in_two_years_l663_66312

theorem age_ratio_in_two_years :
  ∀ (B M : ℕ), B = 10 → M = B + 12 → (M + 2) / (B + 2) = 2 := by
  intros B M hB hM
  sorry

end NUMINAMATH_GPT_age_ratio_in_two_years_l663_66312


namespace NUMINAMATH_GPT_incorrect_equation_is_wrong_l663_66341

-- Specifications and conditions
def speed_person_a : ℝ := 7
def speed_person_b : ℝ := 6.5
def head_start : ℝ := 5

-- Define the time variable
variable (x : ℝ)

-- The correct equation based on the problem statement
def correct_equation : Prop := speed_person_a * x - head_start = speed_person_b * x

-- The incorrect equation to prove incorrect
def incorrect_equation : Prop := speed_person_b * x = speed_person_a * x - head_start

-- The Lean statement to prove that the incorrect equation is indeed incorrect
theorem incorrect_equation_is_wrong (h : correct_equation x) : ¬ incorrect_equation x := by
  sorry

end NUMINAMATH_GPT_incorrect_equation_is_wrong_l663_66341


namespace NUMINAMATH_GPT_probability_all_same_color_l663_66394

open scoped Classical

noncomputable def num_black : ℕ := 5
noncomputable def num_red : ℕ := 4
noncomputable def num_green : ℕ := 6
noncomputable def num_blue : ℕ := 3
noncomputable def num_yellow : ℕ := 2

noncomputable def total_marbles : ℕ :=
  num_black + num_red + num_green + num_blue + num_yellow

noncomputable def prob_all_same_color : ℚ :=
  let p_black := if num_black >= 4 then 
      (num_black / total_marbles) * ((num_black - 1) / (total_marbles - 1)) *
      ((num_black - 2) / (total_marbles - 2)) * ((num_black - 3) / (total_marbles - 3)) else 0
  let p_green := if num_green >= 4 then 
      (num_green / total_marbles) * ((num_green - 1) / (total_marbles - 1)) *
      ((num_green - 2) / (total_marbles - 2)) * ((num_green - 3) / (total_marbles - 3)) else 0
  p_black + p_green

theorem probability_all_same_color :
  prob_all_same_color = 0.004128 :=
sorry

end NUMINAMATH_GPT_probability_all_same_color_l663_66394


namespace NUMINAMATH_GPT_count_two_digit_numbers_with_at_least_one_5_l663_66314

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def has_digit_5 (n : ℕ) : Prop := ∃ (a b : ℕ), is_two_digit (10 * a + b) ∧ (a = 5 ∨ b = 5)

theorem count_two_digit_numbers_with_at_least_one_5 : 
  ∃ count : ℕ, (∀ n, is_two_digit n → has_digit_5 n → n ∈ Finset.range (100)) ∧ count = 18 := 
sorry

end NUMINAMATH_GPT_count_two_digit_numbers_with_at_least_one_5_l663_66314


namespace NUMINAMATH_GPT_calculate_savings_l663_66363

/-- Given the income is 19000 and the income to expenditure ratio is 5:4, prove the savings of 3800. -/
theorem calculate_savings (i : ℕ) (exp : ℕ) (rat : ℕ → ℕ → Prop)
  (h_income : i = 19000)
  (h_ratio : rat 5 4)
  (h_exp_eq : ∃ x, i = 5 * x ∧ exp = 4 * x) :
  i - exp = 3800 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_savings_l663_66363


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l663_66345

theorem necessary_but_not_sufficient_condition
  (a : ℝ)
  (h : ∃ x : ℝ, a * x^2 - 2 * x + 1 < 0) :
  (a < 2 ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l663_66345


namespace NUMINAMATH_GPT_M_inter_N_eq_l663_66358

open Set

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {3, 4, 5, 6}

theorem M_inter_N_eq : M ∩ N = {3, 4} := 
by 
  sorry

end NUMINAMATH_GPT_M_inter_N_eq_l663_66358


namespace NUMINAMATH_GPT_xy_div_eq_one_third_l663_66357

theorem xy_div_eq_one_third (x y z : ℝ) 
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y / z = 6) : 
  x / y = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_xy_div_eq_one_third_l663_66357


namespace NUMINAMATH_GPT_recipe_total_cups_l663_66378

noncomputable def total_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℕ) : ℕ :=
  let part := sugar_cups / sugar_ratio
  let butter_cups := butter_ratio * part
  let flour_cups := flour_ratio * part
  butter_cups + flour_cups + sugar_cups

theorem recipe_total_cups : 
  total_cups 2 7 5 10 = 28 :=
by
  sorry

end NUMINAMATH_GPT_recipe_total_cups_l663_66378


namespace NUMINAMATH_GPT_dave_deleted_apps_l663_66377

theorem dave_deleted_apps :
  ∃ d : ℕ, d = 150 - 65 :=
sorry

end NUMINAMATH_GPT_dave_deleted_apps_l663_66377


namespace NUMINAMATH_GPT_distinct_symbols_count_l663_66372

/-- A modified Morse code symbol is represented by a sequence of dots, dashes, and spaces, where spaces can only appear between dots and dashes but not at the beginning or end of the sequence. -/
def valid_sequence_length_1 := 2
def valid_sequence_length_2 := 2^2
def valid_sequence_length_3 := 2^3 + 3
def valid_sequence_length_4 := 2^4 + 3 * 2^4 + 3 * 2^4 
def valid_sequence_length_5 := 2^5 + 4 * 2^5 + 6 * 2^5 + 4 * 2^5

theorem distinct_symbols_count : 
  valid_sequence_length_1 + valid_sequence_length_2 + valid_sequence_length_3 + valid_sequence_length_4 + valid_sequence_length_5 = 609 := by
  sorry

end NUMINAMATH_GPT_distinct_symbols_count_l663_66372


namespace NUMINAMATH_GPT_problem_one_problem_two_l663_66336

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x + 4

-- Problem (I)
theorem problem_one (m : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → f x m < 0) ↔ m ≤ -5 :=
sorry

-- Problem (II)
theorem problem_two (m : ℝ) :
  (∀ x, (x = 1 ∨ x = 2) → abs ((f x m - x^2) / m) < 1) ↔ (-4 < m ∧ m ≤ -2) :=
sorry

end NUMINAMATH_GPT_problem_one_problem_two_l663_66336


namespace NUMINAMATH_GPT_point_P_in_third_quadrant_l663_66317

def point_in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

theorem point_P_in_third_quadrant :
  point_in_third_quadrant (-3) (-2) :=
by
  sorry -- Proof of the statement, as per the steps given.

end NUMINAMATH_GPT_point_P_in_third_quadrant_l663_66317


namespace NUMINAMATH_GPT_sum_and_product_of_roots_l663_66383

theorem sum_and_product_of_roots (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b = 0 → x = -2 ∨ x = 3) → a + b = -7 :=
by
  sorry

end NUMINAMATH_GPT_sum_and_product_of_roots_l663_66383


namespace NUMINAMATH_GPT_min_a_b_l663_66309

theorem min_a_b (a b : ℕ) (h1 : 43 * a + 17 * b = 731) (h2 : a ≤ 17) (h3 : b ≤ 43) : a + b = 17 :=
by
  sorry

end NUMINAMATH_GPT_min_a_b_l663_66309


namespace NUMINAMATH_GPT_rectangular_prism_cut_corners_edges_l663_66370

def original_edges : Nat := 12
def corners : Nat := 8
def new_edges_per_corner : Nat := 3
def total_new_edges : Nat := corners * new_edges_per_corner

theorem rectangular_prism_cut_corners_edges :
  original_edges + total_new_edges = 36 := sorry

end NUMINAMATH_GPT_rectangular_prism_cut_corners_edges_l663_66370


namespace NUMINAMATH_GPT_arithmetic_mean_of_reciprocals_of_first_four_primes_l663_66308

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_reciprocals_of_first_four_primes_l663_66308


namespace NUMINAMATH_GPT_student_estimated_score_l663_66315

theorem student_estimated_score :
  (6 * 5 + 3 * 5 * (3 / 4) + 2 * 5 * (1 / 3) + 1 * 5 * (1 / 4)) = 41.25 :=
by
 sorry

end NUMINAMATH_GPT_student_estimated_score_l663_66315


namespace NUMINAMATH_GPT_jessica_carrots_l663_66335

theorem jessica_carrots
  (joan_carrots : ℕ)
  (total_carrots : ℕ)
  (jessica_carrots : ℕ) :
  joan_carrots = 29 →
  total_carrots = 40 →
  jessica_carrots = total_carrots - joan_carrots →
  jessica_carrots = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_jessica_carrots_l663_66335


namespace NUMINAMATH_GPT_evaluate_expression_l663_66324

theorem evaluate_expression : 1 + 1 / (2 + 1 / (3 + 2)) = 16 / 11 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l663_66324


namespace NUMINAMATH_GPT_min_value_4x_plus_inv_l663_66364

noncomputable def min_value_function (x : ℝ) := 4 * x + 1 / (4 * x - 5)

theorem min_value_4x_plus_inv (x : ℝ) (h : x > 5 / 4) : min_value_function x = 7 :=
sorry

end NUMINAMATH_GPT_min_value_4x_plus_inv_l663_66364


namespace NUMINAMATH_GPT_inscribed_square_properties_l663_66384

theorem inscribed_square_properties (r : ℝ) (s : ℝ) (d : ℝ) (A_circle : ℝ) (A_square : ℝ) (total_diagonals : ℝ) (hA_circle : A_circle = 324 * Real.pi) (hr : r = Real.sqrt 324) (hd : d = 2 * r) (hs : s = d / Real.sqrt 2) (hA_square : A_square = s ^ 2) (htotal_diagonals : total_diagonals = 2 * d) :
  A_square = 648 ∧ total_diagonals = 72 :=
by sorry

end NUMINAMATH_GPT_inscribed_square_properties_l663_66384


namespace NUMINAMATH_GPT_cos_identity_l663_66344

theorem cos_identity (α : ℝ) : 
  3.4028 * (Real.cos α)^4 + 4 * (Real.cos α)^3 - 8 * (Real.cos α)^2 - 3 * (Real.cos α) + 1 = 
  2 * (Real.cos (7 * α / 2)) * (Real.cos (α / 2)) := 
by sorry

end NUMINAMATH_GPT_cos_identity_l663_66344


namespace NUMINAMATH_GPT_exists_integers_a_b_c_d_and_n_l663_66369

theorem exists_integers_a_b_c_d_and_n (n a b c d : ℕ)
  (h1 : a = 10) 
  (h2 : b = 15) 
  (h3 : c = 8) 
  (h4 : d = 3) 
  (h5 : n = 16) :
  a^4 + b^4 + c^4 + 2 * d^4 = n^4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_exists_integers_a_b_c_d_and_n_l663_66369


namespace NUMINAMATH_GPT_smallest_possible_value_of_other_integer_l663_66361

theorem smallest_possible_value_of_other_integer 
  (n : ℕ) (hn_pos : 0 < n) (h_eq : (Nat.lcm 75 n) / (Nat.gcd 75 n) = 45) : n = 135 :=
by sorry

end NUMINAMATH_GPT_smallest_possible_value_of_other_integer_l663_66361


namespace NUMINAMATH_GPT_combined_weight_of_Alexa_and_Katerina_l663_66356

variable (total_weight: ℝ) (alexas_weight: ℝ) (michaels_weight: ℝ)

theorem combined_weight_of_Alexa_and_Katerina
  (h1: total_weight = 154)
  (h2: alexas_weight = 46)
  (h3: michaels_weight = 62) :
  total_weight - michaels_weight = 92 :=
by 
  sorry

end NUMINAMATH_GPT_combined_weight_of_Alexa_and_Katerina_l663_66356


namespace NUMINAMATH_GPT_adult_ticket_cost_l663_66310

theorem adult_ticket_cost (A Tc : ℝ) (T C : ℕ) (M : ℝ) 
  (hTc : Tc = 3.50) 
  (hT : T = 21) 
  (hC : C = 16) 
  (hM : M = 83.50) 
  (h_eq : 16 * Tc + (↑(T - C)) * A = M) : 
  A = 5.50 :=
by sorry

end NUMINAMATH_GPT_adult_ticket_cost_l663_66310


namespace NUMINAMATH_GPT_cost_of_each_soda_l663_66367

theorem cost_of_each_soda (total_paid : ℕ) (number_of_sodas : ℕ) (change_received : ℕ) 
  (h1 : total_paid = 20) 
  (h2 : number_of_sodas = 3) 
  (h3 : change_received = 14) : 
  (total_paid - change_received) / number_of_sodas = 2 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_soda_l663_66367


namespace NUMINAMATH_GPT_triple_layers_area_l663_66348

-- Defining the conditions
def hall : Type := {x // x = 10 * 10}
def carpet1 : hall := ⟨60, sorry⟩ -- First carpet size: 6 * 8
def carpet2 : hall := ⟨36, sorry⟩ -- Second carpet size: 6 * 6
def carpet3 : hall := ⟨35, sorry⟩ -- Third carpet size: 5 * 7

-- The final theorem statement
theorem triple_layers_area : ∃ area : ℕ, area = 6 :=
by
  have intersection_area : ℕ := 2 * 3
  use intersection_area
  sorry

end NUMINAMATH_GPT_triple_layers_area_l663_66348


namespace NUMINAMATH_GPT_simplify_fraction_part1_simplify_fraction_part2_l663_66397

-- Part 1
theorem simplify_fraction_part1 (x : ℝ) (h1 : x ≠ -2) :
  (x^2 / (x + 2)) + ((4 * x + 4) / (x + 2)) = x + 2 :=
sorry

-- Part 2
theorem simplify_fraction_part2 (x : ℝ) (h1 : x ≠ 1) :
  (x^2 / ((x - 1)^2)) / ((1 - 2 * x) / (x - 1) - (x - 1)) = -1 / (x - 1) :=
sorry

end NUMINAMATH_GPT_simplify_fraction_part1_simplify_fraction_part2_l663_66397


namespace NUMINAMATH_GPT_abc_one_eq_sum_l663_66375

theorem abc_one_eq_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b * c = 1) :
  (a^2 * b^2) / ((a^2 + b * c) * (b^2 + a * c))
  + (a^2 * c^2) / ((a^2 + b * c) * (c^2 + a * b))
  + (b^2 * c^2) / ((b^2 + a * c) * (c^2 + a * b))
  = 1 / (a^2 + 1 / a) + 1 / (b^2 + 1 / b) + 1 / (c^2 + 1 / c) := by
  sorry

end NUMINAMATH_GPT_abc_one_eq_sum_l663_66375


namespace NUMINAMATH_GPT_truck_speed_kmph_l663_66395

theorem truck_speed_kmph (d : ℕ) (t : ℕ) (km_m : ℕ) (hr_s : ℕ) 
  (h1 : d = 600) (h2 : t = 20) (h3 : km_m = 1000) (h4 : hr_s = 3600) : 
  (d / t) * (hr_s / km_m) = 108 := by
  sorry

end NUMINAMATH_GPT_truck_speed_kmph_l663_66395


namespace NUMINAMATH_GPT_oleg_bought_bar_for_60_rubles_l663_66362

theorem oleg_bought_bar_for_60_rubles (n : ℕ) (h₁ : 96 = n * (1 + n / 100)) : n = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_oleg_bought_bar_for_60_rubles_l663_66362


namespace NUMINAMATH_GPT_distance_between_B_and_C_l663_66321

theorem distance_between_B_and_C
  (A B C : Type)
  (AB : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (h_AB : AB = 10)
  (h_angle_A : angle_A = 60)
  (h_angle_B : angle_B = 75) :
  ∃ BC : ℝ, BC = 5 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_B_and_C_l663_66321


namespace NUMINAMATH_GPT_nala_seashells_l663_66329

theorem nala_seashells (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 2 * (a + b)) : a + b + c = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_nala_seashells_l663_66329


namespace NUMINAMATH_GPT_find_x_l663_66304

theorem find_x (x : ℝ) : 9999 * x = 724787425 ↔ x = 72487.5 := 
sorry

end NUMINAMATH_GPT_find_x_l663_66304


namespace NUMINAMATH_GPT_lowest_score_l663_66385

-- Define the conditions
def test_scores (s1 s2 s3 : ℕ) := s1 = 86 ∧ s2 = 112 ∧ s3 = 91
def max_score := 120
def target_average := 95
def num_tests := 5
def total_points_needed := target_average * num_tests

-- Define the proof statement
theorem lowest_score 
  (s1 s2 s3 : ℕ)
  (condition1 : test_scores s1 s2 s3)
  (max_pts : ℕ := max_score) 
  (target_avg : ℕ := target_average) 
  (num_tests : ℕ := num_tests)
  (total_needed : ℕ := total_points_needed) :
  ∃ s4 s5 : ℕ, s4 ≤ max_pts ∧ s5 ≤ max_pts ∧ s4 + s5 + s1 + s2 + s3 = total_needed ∧ (s4 = 66 ∨ s5 = 66) :=
by
  sorry

end NUMINAMATH_GPT_lowest_score_l663_66385


namespace NUMINAMATH_GPT_intersection_point_in_polar_coordinates_l663_66323

theorem intersection_point_in_polar_coordinates (theta : ℝ) (rho : ℝ) (h₁ : theta = π / 3) (h₂ : rho = 2 * Real.cos theta) (h₃ : rho > 0) : rho = 1 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_intersection_point_in_polar_coordinates_l663_66323


namespace NUMINAMATH_GPT_set_P_equality_l663_66371

open Set

variable {U : Set ℝ} (P : Set ℝ)
variable (h_univ : U = univ) (h_def : P = {x | abs (x - 2) ≥ 1})

theorem set_P_equality : P = {x | x ≥ 3 ∨ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_set_P_equality_l663_66371


namespace NUMINAMATH_GPT_non_chocolate_candy_count_l663_66333

theorem non_chocolate_candy_count (total_candy : ℕ) (total_bags : ℕ) 
  (chocolate_hearts_bags : ℕ) (chocolate_kisses_bags : ℕ) (each_bag_pieces : ℕ) 
  (non_chocolate_bags : ℕ) : 
  total_candy = 63 ∧ 
  total_bags = 9 ∧ 
  chocolate_hearts_bags = 2 ∧ 
  chocolate_kisses_bags = 3 ∧ 
  total_candy / total_bags = each_bag_pieces ∧ 
  total_bags - (chocolate_hearts_bags + chocolate_kisses_bags) = non_chocolate_bags ∧ 
  non_chocolate_bags * each_bag_pieces = 28 :=
by
  -- use "sorry" to skip the proof
  sorry

end NUMINAMATH_GPT_non_chocolate_candy_count_l663_66333


namespace NUMINAMATH_GPT_sum_ages_l663_66380

variable (Bob_age Carol_age : ℕ)

theorem sum_ages (h1 : Bob_age = 16) (h2 : Carol_age = 50) (h3 : Carol_age = 3 * Bob_age + 2) :
  Bob_age + Carol_age = 66 :=
by
  sorry

end NUMINAMATH_GPT_sum_ages_l663_66380


namespace NUMINAMATH_GPT_percentage_of_boys_currently_l663_66368

variables (B G : ℕ)

theorem percentage_of_boys_currently
  (h1 : B + G = 50)
  (h2 : B + 50 = 95) :
  (B * 100) / 50 = 90 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_boys_currently_l663_66368


namespace NUMINAMATH_GPT_probability_odd_sum_of_6_balls_drawn_l663_66374

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_odd_sum_of_6_balls_drawn :
  let n := 11
  let k := 6
  let total_ways := binom n k
  let odd_count := 6
  let even_count := 5
  let cases := 
    (binom odd_count 1 * binom even_count (k - 1)) +
    (binom odd_count 3 * binom even_count (k - 3)) +
    (binom odd_count 5 * binom even_count (k - 5))
  let favorable_outcomes := cases
  let probability := favorable_outcomes / total_ways
  probability = 118 / 231 := 
by {
  sorry
}

end NUMINAMATH_GPT_probability_odd_sum_of_6_balls_drawn_l663_66374


namespace NUMINAMATH_GPT_solve_m_l663_66340

theorem solve_m (x y m : ℝ) (h1 : 4 * x + 2 * y = 3 * m) (h2 : 3 * x + y = m + 2) (h3 : y = -x) : m = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_m_l663_66340


namespace NUMINAMATH_GPT_minimum_phi_l663_66330

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the condition for g overlapping with f after shifting by φ
noncomputable def shifted_g (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * φ)

theorem minimum_phi (φ : ℝ) (h : φ > 0) :
  (∃ (x : ℝ), shifted_g x φ = f x) ↔ (∃ k : ℕ, φ = Real.pi / 6 + k * Real.pi) :=
sorry

end NUMINAMATH_GPT_minimum_phi_l663_66330


namespace NUMINAMATH_GPT_escalator_time_l663_66350

theorem escalator_time (escalator_speed person_speed length : ℕ) 
    (h1 : escalator_speed = 12) 
    (h2 : person_speed = 2) 
    (h3 : length = 196) : 
    (length / (escalator_speed + person_speed) = 14) :=
by
  sorry

end NUMINAMATH_GPT_escalator_time_l663_66350


namespace NUMINAMATH_GPT_at_least_one_less_than_two_l663_66365

theorem at_least_one_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 2) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
sorry

end NUMINAMATH_GPT_at_least_one_less_than_two_l663_66365


namespace NUMINAMATH_GPT_original_amount_l663_66388

variable (M : ℕ)

def initialAmountAfterFirstLoss := M - M / 3
def amountAfterFirstWin := initialAmountAfterFirstLoss M + 10
def amountAfterSecondLoss := amountAfterFirstWin M - (amountAfterFirstWin M) / 3
def amountAfterSecondWin := amountAfterSecondLoss M + 20
def finalAmount := amountAfterSecondWin M - (amountAfterSecondWin M) / 4

theorem original_amount : finalAmount M = M → M = 30 :=
by
  sorry

end NUMINAMATH_GPT_original_amount_l663_66388


namespace NUMINAMATH_GPT_shifted_sine_monotonically_increasing_l663_66319

noncomputable def shifted_sine_function (x : ℝ) : ℝ :=
  3 * Real.sin (2 * x - (2 * Real.pi / 3))

theorem shifted_sine_monotonically_increasing :
  ∀ x y : ℝ, (x ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) → (y ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) → x < y → shifted_sine_function x < shifted_sine_function y :=
by
  sorry

end NUMINAMATH_GPT_shifted_sine_monotonically_increasing_l663_66319


namespace NUMINAMATH_GPT_find_constants_l663_66332

variables {A B C x : ℝ}

theorem find_constants (h : (A = 6) ∧ (B = -5) ∧ (C = 5)) :
  (x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1) :=
by sorry

end NUMINAMATH_GPT_find_constants_l663_66332


namespace NUMINAMATH_GPT_cube_sum_of_edges_corners_faces_eq_26_l663_66376

theorem cube_sum_of_edges_corners_faces_eq_26 :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 :=
by
  let edges := 12
  let corners := 8
  let faces := 6
  sorry

end NUMINAMATH_GPT_cube_sum_of_edges_corners_faces_eq_26_l663_66376


namespace NUMINAMATH_GPT_hyperbola_standard_equations_l663_66339

-- Definitions derived from conditions
def focal_distance (c : ℝ) : Prop := c = 8
def eccentricity (e : ℝ) : Prop := e = 4 / 3
def equilateral_focus (c : ℝ) : Prop := c^2 = 36

-- Theorem stating the standard equations given the conditions
noncomputable def hyperbola_equation1 (y2 : ℝ) (x2 : ℝ) : Prop :=
y2 / 36 - x2 / 28 = 1

noncomputable def hyperbola_equation2 (x2 : ℝ) (y2 : ℝ) : Prop :=
x2 / 18 - y2 / 18 = 1

theorem hyperbola_standard_equations
  (c y2 x2 : ℝ)
  (c_focus : focal_distance c)
  (e_value : eccentricity (4 / 3))
  (equi_focus : equilateral_focus c) :
  hyperbola_equation1 y2 x2 ∧ hyperbola_equation2 x2 y2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_standard_equations_l663_66339


namespace NUMINAMATH_GPT_length_more_than_breadth_l663_66399

theorem length_more_than_breadth (length cost_per_metre total_cost : ℝ) (breadth : ℝ) :
  length = 60 → cost_per_metre = 26.50 → total_cost = 5300 → 
  (total_cost = (2 * length + 2 * breadth) * cost_per_metre) → length - breadth = 20 :=
by
  intros hlength hcost_per_metre htotal_cost hperimeter_cost
  rw [hlength, hcost_per_metre] at hperimeter_cost
  sorry

end NUMINAMATH_GPT_length_more_than_breadth_l663_66399


namespace NUMINAMATH_GPT_only_correct_option_is_C_l663_66302

-- Definitions of the conditions as per the given problem
def option_A (a : ℝ) : Prop := a^2 * a^3 = a^6
def option_B (a : ℝ) : Prop := (a^2)^3 = a^5
def option_C (a b : ℝ) : Prop := (a * b)^3 = a^3 * b^3
def option_D (a : ℝ) : Prop := a^8 / a^2 = a^4

-- The theorem stating that only option C is correct
theorem only_correct_option_is_C (a b : ℝ) : 
  ¬(option_A a) ∧ ¬(option_B a) ∧ option_C a b ∧ ¬(option_D a) :=
by sorry

end NUMINAMATH_GPT_only_correct_option_is_C_l663_66302


namespace NUMINAMATH_GPT_Gerald_toy_cars_l663_66316

theorem Gerald_toy_cars :
  let initial_toy_cars := 20
  let fraction_donated := 1 / 4
  let donated_toy_cars := initial_toy_cars * fraction_donated
  let remaining_toy_cars := initial_toy_cars - donated_toy_cars
  remaining_toy_cars = 15 := 
by
  sorry

end NUMINAMATH_GPT_Gerald_toy_cars_l663_66316


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l663_66337

def speed_of_boat (V_b : ℝ) : Prop :=
  let stream_speed := 4  -- speed of the stream in km/hr
  let downstream_distance := 168  -- distance traveled downstream in km
  let time := 6  -- time taken to travel downstream in hours
  (downstream_distance = (V_b + stream_speed) * time)

theorem boat_speed_in_still_water : ∃ V_b, speed_of_boat V_b ∧ V_b = 24 := 
by
  exists 24
  unfold speed_of_boat
  simp
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l663_66337


namespace NUMINAMATH_GPT_brad_age_proof_l663_66379

theorem brad_age_proof :
  ∀ (Shara_age Jaymee_age Average_age Brad_age : ℕ),
  Jaymee_age = 2 * Shara_age + 2 →
  Average_age = (Shara_age + Jaymee_age) / 2 →
  Brad_age = Average_age - 3 →
  Shara_age = 10 →
  Brad_age = 13 :=
by
  intros Shara_age Jaymee_age Average_age Brad_age
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_brad_age_proof_l663_66379


namespace NUMINAMATH_GPT_tangent_inclination_point_l663_66346

theorem tangent_inclination_point :
  ∃ a : ℝ, (2 * a = 1) ∧ ((a, a^2) = (1 / 2, 1 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_inclination_point_l663_66346


namespace NUMINAMATH_GPT_inequality_proof_l663_66389

theorem inequality_proof (a b m n p : ℝ) (h1 : a > b) (h2 : m > n) (h3 : p > 0) : n - a * p < m - b * p :=
sorry

end NUMINAMATH_GPT_inequality_proof_l663_66389


namespace NUMINAMATH_GPT_percentage_increase_efficiency_l663_66355

-- Defining the times taken by Sakshi and Tanya
def sakshi_time : ℕ := 12
def tanya_time : ℕ := 10

-- Defining the efficiency in terms of work per day for Sakshi and Tanya
def sakshi_efficiency : ℚ := 1 / sakshi_time
def tanya_efficiency : ℚ := 1 / tanya_time

-- The statement of the proof: percentage increase
theorem percentage_increase_efficiency : 
  100 * ((tanya_efficiency - sakshi_efficiency) / sakshi_efficiency) = 20 := 
by
  -- The actual proof will go here
  sorry

end NUMINAMATH_GPT_percentage_increase_efficiency_l663_66355


namespace NUMINAMATH_GPT_vernal_equinox_shadow_length_l663_66373

-- Lean 4 statement
theorem vernal_equinox_shadow_length :
  ∀ (a : ℕ → ℝ), (a 4 = 10.5) → (a 10 = 4.5) → 
  (∀ (n m : ℕ), a (n + 1) = a n + (a 2 - a 1)) → 
  a 7 = 7.5 :=
by
  intros a h_4 h_10 h_progression
  sorry

end NUMINAMATH_GPT_vernal_equinox_shadow_length_l663_66373


namespace NUMINAMATH_GPT_dad_steps_l663_66391

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end NUMINAMATH_GPT_dad_steps_l663_66391


namespace NUMINAMATH_GPT_windows_ways_l663_66381

theorem windows_ways (n : ℕ) (h : n = 8) : (n * (n - 1)) = 56 :=
by
  sorry

end NUMINAMATH_GPT_windows_ways_l663_66381


namespace NUMINAMATH_GPT_fraction_inequality_l663_66322

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  b / (a - c) < a / (b - d) :=
sorry

end NUMINAMATH_GPT_fraction_inequality_l663_66322


namespace NUMINAMATH_GPT_experts_expected_points_probability_fifth_envelope_l663_66354

theorem experts_expected_points (n : ℕ) (h1 : n = 100) (h2 : n = 13) :
  ∃ e : ℚ, e = 465 :=
sorry

theorem probability_fifth_envelope (m : ℕ) (h1 : m = 13) :
  ∃ p : ℚ, p = 0.715 :=
sorry

end NUMINAMATH_GPT_experts_expected_points_probability_fifth_envelope_l663_66354


namespace NUMINAMATH_GPT_optimal_floor_optimal_floor_achieved_at_three_l663_66386

theorem optimal_floor : ∀ (n : ℕ), n > 0 → (n + 9 / n : ℝ) ≥ 6 := sorry

theorem optimal_floor_achieved_at_three : ∃ n : ℕ, (n > 0 ∧ (n + 9 / n : ℝ) = 6) := sorry

end NUMINAMATH_GPT_optimal_floor_optimal_floor_achieved_at_three_l663_66386


namespace NUMINAMATH_GPT_average_stickers_per_pack_l663_66331

-- Define the conditions given in the problem
def pack1 := 5
def pack2 := 7
def pack3 := 7
def pack4 := 10
def pack5 := 11
def num_packs := 5
def total_stickers := pack1 + pack2 + pack3 + pack4 + pack5

-- Statement to prove the average number of stickers per pack
theorem average_stickers_per_pack :
  (total_stickers / num_packs) = 8 := by
  sorry

end NUMINAMATH_GPT_average_stickers_per_pack_l663_66331


namespace NUMINAMATH_GPT_geometric_sequence_iff_arithmetic_sequence_l663_66382

/-
  Suppose that {a_n} is an infinite geometric sequence with common ratio q, where q^2 ≠ 1.
  Also suppose that {b_n} is a sequence of positive natural numbers (ℕ).
  Prove that {a_{b_n}} forms a geometric sequence if and only if {b_n} forms an arithmetic sequence.
-/

theorem geometric_sequence_iff_arithmetic_sequence
  (a : ℕ → ℕ) (b : ℕ → ℕ) (q : ℝ)
  (h_geom_a : ∃ a1, ∀ n, a n = a1 * q ^ (n - 1))
  (h_q_squared_ne_one : q^2 ≠ 1)
  (h_bn_positive : ∀ n, 0 < b n) :
  (∃ a1, ∃ q', ∀ n, a (b n) = a1 * q' ^ n) ↔ (∃ d, ∀ n, b (n + 1) - b n = d) := 
sorry

end NUMINAMATH_GPT_geometric_sequence_iff_arithmetic_sequence_l663_66382


namespace NUMINAMATH_GPT_intersection_A_B_l663_66338

-- Define set A and its condition
def A : Set ℝ := { y | ∃ (x : ℝ), y = x^2 }

-- Define set B and its condition
def B : Set ℝ := { x | ∃ (y : ℝ), y = Real.sqrt (1 - x^2) }

-- Define the set intersection A ∩ B
def A_intersect_B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- The theorem statement
theorem intersection_A_B :
  A ∩ B = { x : ℝ | 0 ≤ x ∧ x ≤ 1 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l663_66338


namespace NUMINAMATH_GPT_number_of_geese_l663_66353

theorem number_of_geese (A x n k : ℝ) 
  (h1 : A = k * x * n)
  (h2 : A = (k + 20) * x * (n - 75))
  (h3 : A = (k - 15) * x * (n + 100)) 
  : n = 300 :=
sorry

end NUMINAMATH_GPT_number_of_geese_l663_66353


namespace NUMINAMATH_GPT_solution_set_of_inequality_l663_66359

theorem solution_set_of_inequality (x : ℝ) : (|x + 1| - |x - 3| ≥ 0) ↔ (1 ≤ x) := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l663_66359


namespace NUMINAMATH_GPT_find_m_l663_66327

theorem find_m {m : ℝ} :
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l663_66327


namespace NUMINAMATH_GPT_class_avg_GPA_l663_66306

theorem class_avg_GPA (n : ℕ) (h1 : n > 0) : 
  ((1 / 4 : ℝ) * 92 + (3 / 4 : ℝ) * 76 = 80) :=
sorry

end NUMINAMATH_GPT_class_avg_GPA_l663_66306


namespace NUMINAMATH_GPT_min_possible_value_box_l663_66343

theorem min_possible_value_box (a b : ℤ) (h_ab : a * b = 35) : a^2 + b^2 ≥ 74 := sorry

end NUMINAMATH_GPT_min_possible_value_box_l663_66343


namespace NUMINAMATH_GPT_exists_four_functions_l663_66352

theorem exists_four_functions 
  (f : ℝ → ℝ)
  (h_periodic : ∀ x, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
sorry

end NUMINAMATH_GPT_exists_four_functions_l663_66352


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l663_66320

theorem arithmetic_expression_evaluation : 
  (5 * 7 - (3 * 2 + 5 * 4) / 2) = 22 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l663_66320


namespace NUMINAMATH_GPT_third_chapter_is_24_pages_l663_66396

-- Define the total number of pages in the book
def total_pages : ℕ := 125

-- Define the number of pages in the first chapter
def first_chapter_pages : ℕ := 66

-- Define the number of pages in the second chapter
def second_chapter_pages : ℕ := 35

-- Define the number of pages in the third chapter
def third_chapter_pages : ℕ := total_pages - (first_chapter_pages + second_chapter_pages)

-- Prove that the number of pages in the third chapter is 24
theorem third_chapter_is_24_pages : third_chapter_pages = 24 := by
  sorry

end NUMINAMATH_GPT_third_chapter_is_24_pages_l663_66396


namespace NUMINAMATH_GPT_new_trailer_homes_count_l663_66360

theorem new_trailer_homes_count :
  let old_trailers : ℕ := 30
  let old_avg_age : ℕ := 15
  let years_since : ℕ := 3
  let new_avg_age : ℕ := 10
  let total_age := (old_trailers * (old_avg_age + years_since)) + (3 * new_trailers)
  let total_trailers := old_trailers + new_trailers
  let total_avg_age := total_age / total_trailers
  total_avg_age = new_avg_age → new_trailers = 34 :=
by
  sorry

end NUMINAMATH_GPT_new_trailer_homes_count_l663_66360


namespace NUMINAMATH_GPT_remainder_of_3_to_40_plus_5_mod_5_l663_66342

theorem remainder_of_3_to_40_plus_5_mod_5 : (3^40 + 5) % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3_to_40_plus_5_mod_5_l663_66342


namespace NUMINAMATH_GPT_quadratic_roots_algebraic_expression_value_l663_66393

-- Part 1: Proof statement for the roots of the quadratic equation
theorem quadratic_roots : (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 7 ∧ x₂ = 2 - Real.sqrt 7 ∧ (∀ x : ℝ, x^2 - 4 * x - 3 = 0 → x = x₁ ∨ x = x₂)) :=
by
  sorry

-- Part 2: Proof statement for the algebraic expression value
theorem algebraic_expression_value (a : ℝ) (h : a^2 = 3 * a + 10) :
  (a + 4) * (a - 4) - 3 * (a - 1) = -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_algebraic_expression_value_l663_66393


namespace NUMINAMATH_GPT_largest_n_exists_l663_66328

theorem largest_n_exists :
  ∃ (n : ℕ), (∃ (x y z : ℕ), n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) ∧
    ∀ (m : ℕ), (∃ (x y z : ℕ), m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) →
    n ≥ m :=
  sorry

end NUMINAMATH_GPT_largest_n_exists_l663_66328


namespace NUMINAMATH_GPT_doubled_cylinder_volume_l663_66307

theorem doubled_cylinder_volume (r h : ℝ) (V : ℝ) (original_volume : V = π * r^2 * h) (V' : ℝ) : (2 * 2 * π * r^2 * h = 40) := 
by 
  have original_volume := 5
  sorry

end NUMINAMATH_GPT_doubled_cylinder_volume_l663_66307


namespace NUMINAMATH_GPT_inscribed_rectangle_area_l663_66303

theorem inscribed_rectangle_area (A S x : ℝ) (hA : A = 18) (hS : S = (x * x) * 2) (hx : x = 2):
  S = 8 :=
by
  -- The proofs steps will go here
  sorry

end NUMINAMATH_GPT_inscribed_rectangle_area_l663_66303


namespace NUMINAMATH_GPT_garbage_accumulation_correct_l663_66305

-- Given conditions
def garbage_days_per_week : ℕ := 3
def garbage_per_collection : ℕ := 200
def duration_weeks : ℕ := 2

-- Week 1: Full garbage accumulation
def week1_garbage_accumulation : ℕ := garbage_days_per_week * garbage_per_collection

-- Week 2: Half garbage accumulation due to the policy
def week2_garbage_accumulation : ℕ := week1_garbage_accumulation / 2

-- Total garbage accumulation over the 2 weeks
def total_garbage_accumulation (week1 week2 : ℕ) : ℕ := week1 + week2

-- Proof statement
theorem garbage_accumulation_correct :
  total_garbage_accumulation week1_garbage_accumulation week2_garbage_accumulation = 900 := by
  sorry

end NUMINAMATH_GPT_garbage_accumulation_correct_l663_66305


namespace NUMINAMATH_GPT_pascal_fifth_element_15th_row_l663_66334

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end NUMINAMATH_GPT_pascal_fifth_element_15th_row_l663_66334


namespace NUMINAMATH_GPT_minimum_n_required_l663_66398

def A_0 : (ℝ × ℝ) := (0, 0)

def is_on_x_axis (A : ℝ × ℝ) : Prop := A.snd = 0
def is_on_y_equals_x_squared (B : ℝ × ℝ) : Prop := B.snd = B.fst ^ 2
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop := sorry

def A_n (n : ℕ) : ℝ × ℝ := sorry
def B_n (n : ℕ) : ℝ × ℝ := sorry

def euclidean_distance (P Q : ℝ × ℝ) : ℝ :=
  ((Q.fst - P.fst) ^ 2 + (Q.snd - P.snd) ^ 2) ^ (1/2)

theorem minimum_n_required (n : ℕ) (h1 : ∀ n, is_on_x_axis (A_n n))
    (h2 : ∀ n, is_on_y_equals_x_squared (B_n n))
    (h3 : ∀ n, is_equilateral_triangle (A_n (n-1)) (B_n n) (A_n n)) :
    (euclidean_distance A_0 (A_n n) ≥ 50) → n ≥ 17 :=
by sorry

end NUMINAMATH_GPT_minimum_n_required_l663_66398


namespace NUMINAMATH_GPT_tom_needs_noodle_packages_l663_66390

def beef_pounds : ℕ := 10
def noodle_multiplier : ℕ := 2
def initial_noodles : ℕ := 4
def package_weight : ℕ := 2

theorem tom_needs_noodle_packages :
  (noodle_multiplier * beef_pounds - initial_noodles) / package_weight = 8 := 
by 
  -- Faithfully skipping the solution steps
  sorry

end NUMINAMATH_GPT_tom_needs_noodle_packages_l663_66390


namespace NUMINAMATH_GPT_apple_pies_l663_66351

theorem apple_pies (total_apples not_ripe_apples apples_per_pie : ℕ) 
    (h1 : total_apples = 34) 
    (h2 : not_ripe_apples = 6) 
    (h3 : apples_per_pie = 4) : 
    (total_apples - not_ripe_apples) / apples_per_pie = 7 :=
by 
    sorry

end NUMINAMATH_GPT_apple_pies_l663_66351


namespace NUMINAMATH_GPT_one_point_one_billion_in_scientific_notation_l663_66326

noncomputable def one_point_one_billion : ℝ := 1.1 * 10^9

theorem one_point_one_billion_in_scientific_notation :
  1.1 * 10^9 = 1100000000 :=
by
  sorry

end NUMINAMATH_GPT_one_point_one_billion_in_scientific_notation_l663_66326


namespace NUMINAMATH_GPT_vector_equation_solution_l663_66387

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) (h : 3 • a + 4 • (b - x) = 0) : 
  x = (3 / 4) • a + b := 
sorry

end NUMINAMATH_GPT_vector_equation_solution_l663_66387


namespace NUMINAMATH_GPT_find_a_if_circle_l663_66366

noncomputable def curve_eq (a x y : ℝ) : ℝ :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a

def is_circle_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, curve_eq a x y = 0 → (∃ k : ℝ, curve_eq a x y = k * (x^2 + y^2))

theorem find_a_if_circle :
  (∀ a : ℝ, is_circle_condition a → a = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_circle_l663_66366


namespace NUMINAMATH_GPT_sin_pi_minus_alpha_l663_66325

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin α = 5/13) : Real.sin (π - α) = 5/13 :=
by
  sorry

end NUMINAMATH_GPT_sin_pi_minus_alpha_l663_66325


namespace NUMINAMATH_GPT_range_of_a_l663_66347

theorem range_of_a (x a : ℝ) (hp : x^2 + 2 * x - 3 > 0) (hq : x > a)
  (h_suff : x^2 + 2 * x - 3 > 0 → ¬ (x > a)):
  a ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l663_66347


namespace NUMINAMATH_GPT_initial_amount_celine_had_l663_66318

-- Define the costs and quantities
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def change_received : ℕ := 200

-- Calculate costs and total amount
def cost_laptops : ℕ := num_laptops * laptop_cost
def cost_smartphones : ℕ := num_smartphones * smartphone_cost
def total_cost : ℕ := cost_laptops + cost_smartphones
def initial_amount : ℕ := total_cost + change_received

-- The statement to prove
theorem initial_amount_celine_had : initial_amount = 3000 := by
  sorry

end NUMINAMATH_GPT_initial_amount_celine_had_l663_66318


namespace NUMINAMATH_GPT_base2_to_base4_conversion_l663_66349

theorem base2_to_base4_conversion :
  (2 ^ 8 + 2 ^ 6 + 2 ^ 4 + 2 ^ 3 + 2 ^ 2 + 1) = (1 * 4^3 + 1 * 4^2 + 3 * 4^1 + 1 * 4^0) :=
by 
  sorry

end NUMINAMATH_GPT_base2_to_base4_conversion_l663_66349


namespace NUMINAMATH_GPT_non_degenerate_ellipse_condition_l663_66301

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 3 * x^2 + 6 * y^2 - 12 * x + 18 * y = k) ↔ k > -51 / 2 :=
sorry

end NUMINAMATH_GPT_non_degenerate_ellipse_condition_l663_66301


namespace NUMINAMATH_GPT_subset_condition_intersection_condition_l663_66311

-- Definitions of the sets A and B
def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, 3 * a}

-- Theorem statements
theorem subset_condition (a : ℝ) : A ⊆ B a → (4 / 3) ≤ a ∧ a ≤ 2 := 
by 
  sorry

theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → (2 / 3) < a ∧ a < 4 := 
by 
  sorry

end NUMINAMATH_GPT_subset_condition_intersection_condition_l663_66311


namespace NUMINAMATH_GPT_midpoint_sum_coordinates_l663_66313

theorem midpoint_sum_coordinates (x y : ℝ) 
  (midpoint_cond_x : (x + 10) / 2 = 4) 
  (midpoint_cond_y : (y + 4) / 2 = -8) : 
  x + y = -22 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_sum_coordinates_l663_66313
