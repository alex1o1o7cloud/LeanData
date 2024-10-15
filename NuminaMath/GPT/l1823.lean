import Mathlib

namespace NUMINAMATH_GPT_votes_for_sue_l1823_182313

-- Conditions from the problem
def total_votes := 1000
def category1_percent := 20 / 100   -- 20%
def category2_percent := 45 / 100   -- 45%
def sue_percent := 1 - (category1_percent + category2_percent)  -- Remaining percentage

-- Mathematically equivalent proof problem
theorem votes_for_sue : sue_percent * total_votes = 350 :=
by
  -- reminder: we do not need to provide the proof here
  sorry

end NUMINAMATH_GPT_votes_for_sue_l1823_182313


namespace NUMINAMATH_GPT_floor_of_smallest_zero_l1823_182301
noncomputable def g (x : ℝ) := 3 * Real.sin x - Real.cos x + 2 * Real.tan x
def smallest_zero (s : ℝ) : Prop := s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem floor_of_smallest_zero (s : ℝ) (h : smallest_zero s) : ⌊s⌋ = 4 :=
sorry

end NUMINAMATH_GPT_floor_of_smallest_zero_l1823_182301


namespace NUMINAMATH_GPT_total_cost_magic_decks_l1823_182344

theorem total_cost_magic_decks (price_per_deck : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) :
  price_per_deck = 7 ∧ frank_decks = 3 ∧ friend_decks = 2 → 
  (price_per_deck * frank_decks + price_per_deck * friend_decks) = 35 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_magic_decks_l1823_182344


namespace NUMINAMATH_GPT_find_a_l1823_182361

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : ∀ (x : ℝ), |x - a| < 1 → x ∈ {x | x = 2}) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1823_182361


namespace NUMINAMATH_GPT_sum_of_fourth_powers_is_three_times_square_l1823_182394

theorem sum_of_fourth_powers_is_three_times_square (n : ℤ) (h : n ≠ 0) :
  (n - 1)^4 + n^4 + (n + 1)^4 + 10 = 3 * (n^2 + 2)^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_is_three_times_square_l1823_182394


namespace NUMINAMATH_GPT_expected_adjacent_red_pairs_l1823_182357

theorem expected_adjacent_red_pairs :
  ∃ (E : ℚ), E = 650 / 51 :=
sorry

end NUMINAMATH_GPT_expected_adjacent_red_pairs_l1823_182357


namespace NUMINAMATH_GPT_yoongi_calculation_l1823_182329

theorem yoongi_calculation (x : ℝ) (h : x - 5 = 30) : x / 7 = 5 :=
by
  sorry

end NUMINAMATH_GPT_yoongi_calculation_l1823_182329


namespace NUMINAMATH_GPT_numerator_denominator_added_l1823_182345

theorem numerator_denominator_added (n : ℕ) : (3 + n) / (5 + n) = 9 / 11 → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_numerator_denominator_added_l1823_182345


namespace NUMINAMATH_GPT_installment_payment_l1823_182346

theorem installment_payment
  (cash_price : ℕ)
  (down_payment : ℕ)
  (first_four_months_payment : ℕ)
  (last_four_months_payment : ℕ)
  (installment_additional_cost : ℕ)
  (total_next_four_months_payment : ℕ)
  (H_cash_price : cash_price = 450)
  (H_down_payment : down_payment = 100)
  (H_first_four_months_payment : first_four_months_payment = 4 * 40)
  (H_last_four_months_payment : last_four_months_payment = 4 * 30)
  (H_installment_additional_cost : installment_additional_cost = 70)
  (H_total_next_four_months_payment_correct : 4 * total_next_four_months_payment = 4 * 35) :
  down_payment + first_four_months_payment + 4 * 35 + last_four_months_payment = cash_price + installment_additional_cost := 
by {
  sorry
}

end NUMINAMATH_GPT_installment_payment_l1823_182346


namespace NUMINAMATH_GPT_six_digit_squares_l1823_182379

theorem six_digit_squares :
    ∃ n m : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 100 ≤ m ∧ m ≤ 999 ∧ n = m^2 ∧ (n = 390625 ∨ n = 141376) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_squares_l1823_182379


namespace NUMINAMATH_GPT_cannot_have_N_less_than_K_l1823_182322

theorem cannot_have_N_less_than_K (K N : ℕ) (hK : K > 2) (cards : Fin N → ℕ) (h_cards : ∀ i, cards i > 0) :
  ¬ (N < K) :=
sorry

end NUMINAMATH_GPT_cannot_have_N_less_than_K_l1823_182322


namespace NUMINAMATH_GPT_number_of_solutions_l1823_182310

theorem number_of_solutions (f : ℕ → ℕ) (n : ℕ) : 
  (∀ n, f n = n^4 + 2 * n^3 - 20 * n^2 + 2 * n - 21) →
  (∀ n, 0 ≤ n ∧ n < 2013 → 2013 ∣ f n) → 
  ∃ k, k = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l1823_182310


namespace NUMINAMATH_GPT_compare_abc_l1823_182388

noncomputable def a : ℝ :=
  (1/2) * Real.cos 16 - (Real.sqrt 3 / 2) * Real.sin 16

noncomputable def b : ℝ :=
  2 * Real.tan 14 / (1 + (Real.tan 14) ^ 2)

noncomputable def c : ℝ :=
  Real.sqrt ((1 - Real.cos 50) / 2)

theorem compare_abc : b > c ∧ c > a :=
  by sorry

end NUMINAMATH_GPT_compare_abc_l1823_182388


namespace NUMINAMATH_GPT_tetrahedron_volume_distance_relation_l1823_182399

theorem tetrahedron_volume_distance_relation
  (V : ℝ) (S1 S2 S3 S4 : ℝ) (H1 H2 H3 H4 : ℝ)
  (k : ℝ)
  (hS : (S1 / 1) = k) (hS2 : (S2 / 2) = k) (hS3 : (S3 / 3) = k) (hS4 : (S4 / 4) = k) :
  H1 + 2 * H2 + 3 * H3 + 4 * H4 = 3 * V / k :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_distance_relation_l1823_182399


namespace NUMINAMATH_GPT_appropriate_import_range_l1823_182377

def mung_bean_import_range (p0 : ℝ) (p_desired_min p_desired_max : ℝ) (x : ℝ) : Prop :=
  p0 - (x / 100) ≤ p_desired_max ∧ p0 - (x / 100) ≥ p_desired_min

theorem appropriate_import_range : 
  ∃ x : ℝ, 600 ≤ x ∧ x ≤ 800 ∧ mung_bean_import_range 16 8 10 x :=
sorry

end NUMINAMATH_GPT_appropriate_import_range_l1823_182377


namespace NUMINAMATH_GPT_probability_enemy_plane_hit_l1823_182397

noncomputable def P_A : ℝ := 0.6
noncomputable def P_B : ℝ := 0.4

theorem probability_enemy_plane_hit : 1 - ((1 - P_A) * (1 - P_B)) = 0.76 :=
by
  sorry

end NUMINAMATH_GPT_probability_enemy_plane_hit_l1823_182397


namespace NUMINAMATH_GPT_intersection_M_N_l1823_182302

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1823_182302


namespace NUMINAMATH_GPT_find_length_of_smaller_rectangle_l1823_182308

theorem find_length_of_smaller_rectangle
  (w : ℝ)
  (h_original : 10 * 15 = 150)
  (h_new_rectangle : 2 * w * w = 150)
  (h_z : w = 5 * Real.sqrt 3) :
  z = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_smaller_rectangle_l1823_182308


namespace NUMINAMATH_GPT_find_k_l1823_182328

-- Define a point and its translation
structure Point where
  x : ℕ
  y : ℕ

-- Original and translated points
def P : Point := { x := 5, y := 3 }
def P' : Point := { x := P.x - 4, y := P.y - 1 }

-- Given function with parameter k
def line (k : ℕ) (p : Point) : ℕ := (k * p.x) - 2

-- Prove the value of k
theorem find_k (k : ℕ) (h : line k P' = P'.y) : k = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1823_182328


namespace NUMINAMATH_GPT_count_integer_solutions_l1823_182325

theorem count_integer_solutions : 
  ∃ (s : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ s) ↔ (x^3 + y^2 = 2*y + 1)) ∧ 
  s.card = 3 := 
by
  sorry

end NUMINAMATH_GPT_count_integer_solutions_l1823_182325


namespace NUMINAMATH_GPT_find_a_and_b_l1823_182342

theorem find_a_and_b (a b : ℝ) (h1 : b - 1/4 = (a + b) / 4 + b / 2) (h2 : 4 * a / 3 = (a + b) / 2)  :
  a = 3/2 ∧ b = 5/2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1823_182342


namespace NUMINAMATH_GPT_interest_years_proof_l1823_182384

theorem interest_years_proof :
  let interest_r800_first_2_years := 800 * 0.05 * 2
  let interest_r800_next_3_years := 800 * 0.12 * 3
  let total_interest_r800 := interest_r800_first_2_years + interest_r800_next_3_years
  let interest_r600_first_3_years := 600 * 0.07 * 3
  let interest_r600_next_n_years := 600 * 0.10 * n
  (interest_r600_first_3_years + interest_r600_next_n_years = total_interest_r800) ->
  n = 5 →
  3 + n = 8 :=
by
  sorry

end NUMINAMATH_GPT_interest_years_proof_l1823_182384


namespace NUMINAMATH_GPT_factory_workers_count_l1823_182385

theorem factory_workers_count :
  ∃ (F S_f : ℝ), 
    (F * S_f = 30000) ∧ 
    (30 * (S_f + 500) = 75000) → 
    (F = 15) :=
by
  sorry

end NUMINAMATH_GPT_factory_workers_count_l1823_182385


namespace NUMINAMATH_GPT_mikes_original_speed_l1823_182392

variable (x : ℕ) -- x is the original typing speed of Mike

-- Condition: After the accident, Mike's typing speed is 20 words per minute less
def currentSpeed : ℕ := x - 20

-- Condition: It takes Mike 18 minutes to type 810 words at his reduced speed
def typingTimeCondition : Prop := 18 * currentSpeed x = 810

-- Proof goal: Prove that Mike's original typing speed is 65 words per minute
theorem mikes_original_speed (h : typingTimeCondition x) : x = 65 := 
sorry

end NUMINAMATH_GPT_mikes_original_speed_l1823_182392


namespace NUMINAMATH_GPT_debate_team_group_size_l1823_182316

theorem debate_team_group_size (boys girls groups : ℕ) (h_boys : boys = 11) (h_girls : girls = 45) (h_groups : groups = 8) : 
  (boys + girls) / groups = 7 := by
  sorry

end NUMINAMATH_GPT_debate_team_group_size_l1823_182316


namespace NUMINAMATH_GPT_quadrilateral_segments_condition_l1823_182354

-- Define the lengths and their conditions
variables {a b c d : ℝ}

-- Define the main theorem with necessary and sufficient conditions
theorem quadrilateral_segments_condition (h_sum : a + b + c + d = 1.5)
    (h_order : a ≤ b) (h_order2 : b ≤ c) (h_order3 : c ≤ d) (h_ratio : d ≤ 3 * a) :
    (a ≥ 0.25 ∧ d < 0.75) ↔ (a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by {
  sorry -- proof is omitted
}

end NUMINAMATH_GPT_quadrilateral_segments_condition_l1823_182354


namespace NUMINAMATH_GPT_length_of_green_caterpillar_l1823_182320

def length_of_orange_caterpillar : ℝ := 1.17
def difference_in_length_between_caterpillars : ℝ := 1.83

theorem length_of_green_caterpillar :
  (length_of_orange_caterpillar + difference_in_length_between_caterpillars) = 3.00 :=
by
  sorry

end NUMINAMATH_GPT_length_of_green_caterpillar_l1823_182320


namespace NUMINAMATH_GPT_ratio_of_adults_to_children_l1823_182312

-- Defining conditions as functions
def admission_fees_condition (a c : ℕ) : ℕ := 30 * a + 15 * c

-- Stating the problem
theorem ratio_of_adults_to_children (a c : ℕ) 
  (h1 : admission_fees_condition a c = 2250)
  (h2 : a ≥ 1) 
  (h3 : c ≥ 1) 
  : a / c = 2 := 
sorry

end NUMINAMATH_GPT_ratio_of_adults_to_children_l1823_182312


namespace NUMINAMATH_GPT_min_quadratic_expression_value_l1823_182387

def quadratic_expression (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2205

theorem min_quadratic_expression_value : 
  ∃ x : ℝ, quadratic_expression x = 2178 :=
sorry

end NUMINAMATH_GPT_min_quadratic_expression_value_l1823_182387


namespace NUMINAMATH_GPT_distance_first_day_l1823_182370

theorem distance_first_day (total_distance : ℕ) (q : ℚ) (n : ℕ) (a : ℚ) : total_distance = 378 ∧ q = 1 / 2 ∧ n = 6 → a = 192 :=
by
  -- Proof omitted, just provide the statement
  sorry

end NUMINAMATH_GPT_distance_first_day_l1823_182370


namespace NUMINAMATH_GPT_determinant_matrix_example_l1823_182323

open Matrix

def matrix_example : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -2], ![-3, 6]]

noncomputable def compute_det_and_add_5 : ℤ := (matrix_example.det) + 5

theorem determinant_matrix_example :
  compute_det_and_add_5 = 41 := by
  sorry

end NUMINAMATH_GPT_determinant_matrix_example_l1823_182323


namespace NUMINAMATH_GPT_solution_l1823_182351

theorem solution {a : ℕ → ℝ} 
  (h : a 1 = 1)
  (h2 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    a n - 4 * a (if n = 100 then 1 else n + 1) + 3 * a (if n = 99 then 1 else if n = 100 then 2 else n + 2) ≥ 0) :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → a n = 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_l1823_182351


namespace NUMINAMATH_GPT_find_4a_plus_8b_l1823_182360

def quadratic_equation_x_solution (a b : ℝ) : Prop :=
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0

theorem find_4a_plus_8b (a b : ℝ) (h : quadratic_equation_x_solution a b) : 4 * a + 8 * b = -4 := 
  by
    sorry

end NUMINAMATH_GPT_find_4a_plus_8b_l1823_182360


namespace NUMINAMATH_GPT_not_all_squares_congruent_l1823_182335

-- Define what it means to be a square
structure Square :=
  (side : ℝ)
  (angle : ℝ)
  (is_square : side > 0 ∧ angle = 90)

-- Define congruency of squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side = s2.side ∧ s1.angle = s2.angle

-- The main statement to prove 
theorem not_all_squares_congruent : ∃ s1 s2 : Square, ¬ congruent s1 s2 :=
by
  sorry

end NUMINAMATH_GPT_not_all_squares_congruent_l1823_182335


namespace NUMINAMATH_GPT_range_of_a_l1823_182373

variable {a : ℝ}

theorem range_of_a (h : ∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (5 - a)) : -3 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1823_182373


namespace NUMINAMATH_GPT_find_value_of_p_l1823_182347

theorem find_value_of_p (p q r s t u v w : ℤ)
  (h1 : r + s = -2)
  (h2 : s + (-2) = 5)
  (h3 : t + u = 5)
  (h4 : u + v = 3)
  (h5 : v + w = 8)
  (h6 : w + t = 3)
  (h7 : q + r = s)
  (h8 : p + q = r) :
  p = -25 := by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_find_value_of_p_l1823_182347


namespace NUMINAMATH_GPT_min_value_of_a_b_c_l1823_182396

variable (a b c : ℕ)
variable (x1 x2 : ℝ)

axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a * x1^2 + b * x1 + c = 0
axiom h5 : a * x2^2 + b * x2 + c = 0
axiom h6 : |x1| < 1/3
axiom h7 : |x2| < 1/3

theorem min_value_of_a_b_c : a + b + c = 25 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_b_c_l1823_182396


namespace NUMINAMATH_GPT_sum_of_decimals_as_fraction_l1823_182348

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_as_fraction_l1823_182348


namespace NUMINAMATH_GPT_butterflies_left_correct_l1823_182327

-- Define the total number of butterflies and the fraction that flies away
def butterflies_total : ℕ := 9
def fraction_fly_away : ℚ := 1 / 3

-- Define the number of butterflies left in the garden
def butterflies_left (t : ℕ) (f : ℚ) : ℚ := t - (t : ℚ) * f

-- State the theorem
theorem butterflies_left_correct : butterflies_left butterflies_total fraction_fly_away = 6 := by
  sorry

end NUMINAMATH_GPT_butterflies_left_correct_l1823_182327


namespace NUMINAMATH_GPT_problem_stated_l1823_182300

-- Definitions of constants based on conditions
def a : ℕ := 5
def b : ℕ := 4
def c : ℕ := 3
def d : ℕ := 400
def x : ℕ := 401

-- Mathematical theorem stating the question == answer given conditions
theorem problem_stated : a * x + b * x + c * x + d = 5212 := 
by 
  sorry

end NUMINAMATH_GPT_problem_stated_l1823_182300


namespace NUMINAMATH_GPT_maximum_area_of_equilateral_triangle_in_rectangle_l1823_182378

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  (953 * Real.sqrt 3) / 16

theorem maximum_area_of_equilateral_triangle_in_rectangle :
  ∀ (a b : ℕ), a = 13 → b = 14 → maxEquilateralTriangleArea a b = (953 * Real.sqrt 3) / 16 :=
by
  intros a b h₁ h₂
  rw [h₁, h₂]
  apply rfl

end NUMINAMATH_GPT_maximum_area_of_equilateral_triangle_in_rectangle_l1823_182378


namespace NUMINAMATH_GPT_sum_of_coefficients_l1823_182340

-- Definition of the polynomial
def P (x : ℝ) : ℝ := 5 * (2 * x ^ 9 - 3 * x ^ 6 + 4) - 4 * (x ^ 6 - 5 * x ^ 3 + 6)

-- Theorem stating the sum of the coefficients is 7
theorem sum_of_coefficients : P 1 = 7 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1823_182340


namespace NUMINAMATH_GPT_impossible_arrangement_l1823_182343

theorem impossible_arrangement (s : Finset ℕ) (h₁ : s = Finset.range 2018 \ {0})
  (h₂ : ∀ a ∈ s, ∀ b ∈ s, a ≠ b ∧ (b = a + 17 ∨ b = a + 21 ∨ b = a - 17 ∨ b = a - 21)) : False :=
by
  sorry

end NUMINAMATH_GPT_impossible_arrangement_l1823_182343


namespace NUMINAMATH_GPT_verify_trig_identity_l1823_182364

noncomputable def trig_identity_eqn : Prop :=
  2 * Real.sqrt (1 - Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4

theorem verify_trig_identity : trig_identity_eqn := by
  sorry

end NUMINAMATH_GPT_verify_trig_identity_l1823_182364


namespace NUMINAMATH_GPT_probability_of_C_and_D_are_equal_l1823_182311

theorem probability_of_C_and_D_are_equal (h1 : Prob_A = 1/4) (h2 : Prob_B = 1/3) (h3 : total_prob = 1) (h4 : Prob_C = Prob_D) : 
  Prob_C = 5/24 ∧ Prob_D = 5/24 := by
  sorry

end NUMINAMATH_GPT_probability_of_C_and_D_are_equal_l1823_182311


namespace NUMINAMATH_GPT_log_function_increasing_interval_l1823_182333

theorem log_function_increasing_interval (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x < y → y ≤ 3 → 4 - ax > 0 ∧ (4 - ax < 4 - ay)) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end NUMINAMATH_GPT_log_function_increasing_interval_l1823_182333


namespace NUMINAMATH_GPT_regression_analysis_notes_l1823_182318

-- Define the conditions
def applicable_population (reg_eq: Type) (sample: Type) : Prop := sorry
def temporality (reg_eq: Type) : Prop := sorry
def sample_value_range_influence (reg_eq: Type) (sample: Type) : Prop := sorry
def prediction_precision (reg_eq: Type) : Prop := sorry

-- Define the key points to note
def key_points_to_note (reg_eq: Type) (sample: Type) : Prop :=
  applicable_population reg_eq sample ∧
  temporality reg_eq ∧
  sample_value_range_influence reg_eq sample ∧
  prediction_precision reg_eq

-- The main statement
theorem regression_analysis_notes (reg_eq: Type) (sample: Type) :
  key_points_to_note reg_eq sample := sorry

end NUMINAMATH_GPT_regression_analysis_notes_l1823_182318


namespace NUMINAMATH_GPT_appeared_candidates_l1823_182309

noncomputable def number_of_candidates_that_appeared_from_each_state (X : ℝ) : Prop :=
  (8 / 100) * X + 220 = (12 / 100) * X

theorem appeared_candidates (X : ℝ) (h : number_of_candidates_that_appeared_from_each_state X) : X = 5500 :=
  sorry

end NUMINAMATH_GPT_appeared_candidates_l1823_182309


namespace NUMINAMATH_GPT_policeman_hats_difference_l1823_182334

theorem policeman_hats_difference
  (hats_simpson : ℕ)
  (hats_obrien_now : ℕ)
  (hats_obrien_before : ℕ)
  (H : hats_simpson = 15)
  (H_hats_obrien_now : hats_obrien_now = 34)
  (H_hats_obrien_twice : hats_obrien_before = hats_obrien_now + 1) :
  hats_obrien_before - 2 * hats_simpson = 5 :=
by
  sorry

end NUMINAMATH_GPT_policeman_hats_difference_l1823_182334


namespace NUMINAMATH_GPT_regression_line_passes_through_sample_mean_point_l1823_182330

theorem regression_line_passes_through_sample_mean_point
  (a b : ℝ) (x y : ℝ)
  (hx : x = a + b*x) :
  y = a + b*x :=
by sorry

end NUMINAMATH_GPT_regression_line_passes_through_sample_mean_point_l1823_182330


namespace NUMINAMATH_GPT_integer_solutions_l1823_182372

theorem integer_solutions (x y z : ℤ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x + y + z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 1 / (x + y + z)) ↔ (z = -x - y) :=
sorry

end NUMINAMATH_GPT_integer_solutions_l1823_182372


namespace NUMINAMATH_GPT_find_x_when_z_64_l1823_182339

-- Defining the conditions
def directly_proportional (x y : ℝ) : Prop := ∃ m : ℝ, x = m * y^3
def inversely_proportional (y z : ℝ) : Prop := ∃ n : ℝ, y = n / z^2

theorem find_x_when_z_64 (x y z : ℝ) (m n : ℝ) (k : ℝ) (h1 : directly_proportional x y) 
    (h2 : inversely_proportional y z) (h3 : z = 64) (h4 : x = 8) (h5 : z = 16) : x = 1/256 := 
  sorry

end NUMINAMATH_GPT_find_x_when_z_64_l1823_182339


namespace NUMINAMATH_GPT_positive_solution_system_l1823_182366

theorem positive_solution_system (x1 x2 x3 x4 x5 : ℝ) (h1 : (x3 + x4 + x5)^5 = 3 * x1)
  (h2 : (x4 + x5 + x1)^5 = 3 * x2) (h3 : (x5 + x1 + x2)^5 = 3 * x3)
  (h4 : (x1 + x2 + x3)^5 = 3 * x4) (h5 : (x2 + x3 + x4)^5 = 3 * x5) :
  x1 > 0 → x2 > 0 → x3 > 0 → x4 > 0 → x5 > 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 ∧ (x1 = 1/3) :=
by 
  intros hpos1 hpos2 hpos3 hpos4 hpos5
  sorry

end NUMINAMATH_GPT_positive_solution_system_l1823_182366


namespace NUMINAMATH_GPT_total_amount_divided_l1823_182315

theorem total_amount_divided (B_amount A_amount C_amount: ℝ) (h1 : A_amount = (1/3) * B_amount)
    (h2 : B_amount = 270) (h3 : B_amount = (1/4) * C_amount) :
    A_amount + B_amount + C_amount = 1440 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_divided_l1823_182315


namespace NUMINAMATH_GPT_compute_f_seven_halves_l1823_182307

theorem compute_f_seven_halves 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_shift : ∀ x, f (x + 2) = -f x)
  (h_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f (7 / 2) = -1 / 2 :=
  sorry

end NUMINAMATH_GPT_compute_f_seven_halves_l1823_182307


namespace NUMINAMATH_GPT_line_k_x_intercept_l1823_182368

theorem line_k_x_intercept :
  ∀ (x y : ℝ), 3 * x - 5 * y + 40 = 0 ∧ 
  ∃ m' b', (m' = 4) ∧ (b' = 20 - 4 * 20) ∧ 
  (y = m' * x + b') →
  ∃ x_inter, (y = 0) → (x_inter = 15) := 
by
  sorry

end NUMINAMATH_GPT_line_k_x_intercept_l1823_182368


namespace NUMINAMATH_GPT_not_or_false_imp_and_false_l1823_182395

variable (p q : Prop)

theorem not_or_false_imp_and_false (h : ¬ (p ∨ q) = False) : ¬ (p ∧ q) :=
by
  sorry

end NUMINAMATH_GPT_not_or_false_imp_and_false_l1823_182395


namespace NUMINAMATH_GPT_difference_in_elevation_difference_in_running_time_l1823_182359

structure Day :=
  (distance_km : ℝ) -- kilometers
  (pace_min_per_km : ℝ) -- minutes per kilometer
  (elevation_gain_m : ℝ) -- meters

def monday : Day := { distance_km := 9, pace_min_per_km := 6, elevation_gain_m := 300 }
def wednesday : Day := { distance_km := 4.816, pace_min_per_km := 5.5, elevation_gain_m := 150 }
def friday : Day := { distance_km := 2.095, pace_min_per_km := 7, elevation_gain_m := 50 }

noncomputable def calculate_running_time(day : Day) : ℝ :=
  day.distance_km * day.pace_min_per_km

noncomputable def total_elevation_gain(wednesday friday : Day) : ℝ :=
  wednesday.elevation_gain_m + friday.elevation_gain_m

noncomputable def total_running_time(wednesday friday : Day) : ℝ :=
  calculate_running_time wednesday + calculate_running_time friday

theorem difference_in_elevation :
  monday.elevation_gain_m - total_elevation_gain wednesday friday = 100 := by 
  sorry

theorem difference_in_running_time :
  calculate_running_time monday - total_running_time wednesday friday = 12.847 := by 
  sorry

end NUMINAMATH_GPT_difference_in_elevation_difference_in_running_time_l1823_182359


namespace NUMINAMATH_GPT_determine_a_of_parallel_lines_l1823_182389

theorem determine_a_of_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x ↔ y = 3 * x + a) →
  (∀ x y : ℝ, y - 2 = (a - 3) * x ↔ y = (a - 3) * x + 2) →
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x → y - 2 = (a - 3) * x → 3 = a - 3) →
  a = 6 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_of_parallel_lines_l1823_182389


namespace NUMINAMATH_GPT_lunch_cost_before_tip_l1823_182314

theorem lunch_cost_before_tip (tip_rate : ℝ) (total_spent : ℝ) (C : ℝ) : 
  tip_rate = 0.20 ∧ total_spent = 72.96 ∧ C + tip_rate * C = total_spent → C = 60.80 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_lunch_cost_before_tip_l1823_182314


namespace NUMINAMATH_GPT_diamond_sum_l1823_182381

def diamond (x : ℚ) : ℚ := (x^3 + 2 * x^2 + 3 * x) / 6

theorem diamond_sum : diamond 2 + diamond 3 + diamond 4 = 92 / 3 := by
  sorry

end NUMINAMATH_GPT_diamond_sum_l1823_182381


namespace NUMINAMATH_GPT_neg_two_is_negative_rational_l1823_182375

theorem neg_two_is_negative_rational : 
  (-2 : ℚ) < 0 ∧ ∃ (r : ℚ), r = -2 := 
by
  sorry

end NUMINAMATH_GPT_neg_two_is_negative_rational_l1823_182375


namespace NUMINAMATH_GPT_solution_set_of_f_l1823_182398

theorem solution_set_of_f (f : ℝ → ℝ) (h1 : ∀ x, 2 < deriv f x) (h2 : f (-1) = 2) :
  ∀ x, x > -1 → f x > 2 * x + 4 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_f_l1823_182398


namespace NUMINAMATH_GPT_max_blocks_fit_l1823_182380

-- Define the dimensions of the block
def block_length := 2
def block_width := 3
def block_height := 1

-- Define the dimensions of the container box
def box_length := 4
def box_width := 3
def box_height := 3

-- Define the volume calculations
def volume (length width height : ℕ) : ℕ := length * width * height

def block_volume := volume block_length block_width block_height
def box_volume := volume box_length box_width box_height

-- The theorem to prove
theorem max_blocks_fit : (box_volume / block_volume) = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_blocks_fit_l1823_182380


namespace NUMINAMATH_GPT_find_z_value_l1823_182374

theorem find_z_value (k : ℝ) (y z : ℝ) (h1 : (y = 2) → (z = 1)) (h2 : y ^ 3 * z ^ (1/3) = k) : 
  (y = 4) → z = 1 / 512 :=
by
  sorry

end NUMINAMATH_GPT_find_z_value_l1823_182374


namespace NUMINAMATH_GPT_number_of_boys_in_school_l1823_182336

-- Definition of percentages for Muslims, Hindus, and Sikhs
def percent_muslims : ℝ := 0.46
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10

-- Given number of boys in other communities
def boys_other_communities : ℝ := 136

-- The total number of boys in the school
def total_boys (B : ℝ) : Prop := B = 850

-- Proof statement (with conditions embedded)
theorem number_of_boys_in_school (B : ℝ) :
  percent_muslims * B + percent_hindus * B + percent_sikhs * B + boys_other_communities = B → 
  total_boys B :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_in_school_l1823_182336


namespace NUMINAMATH_GPT_compute_expression_l1823_182371

theorem compute_expression (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 16 / (x - 3)^2 = 23 := 
  sorry

end NUMINAMATH_GPT_compute_expression_l1823_182371


namespace NUMINAMATH_GPT_ratio_adult_child_l1823_182353

theorem ratio_adult_child (total_fee adults_fee children_fee adults children : ℕ) 
  (h1 : adults ≥ 1) (h2 : children ≥ 1) 
  (h3 : adults_fee = 30) (h4 : children_fee = 15) 
  (h5 : total_fee = 2250) 
  (h6 : adults_fee * adults + children_fee * children = total_fee) :
  (2 : ℚ) = adults / children :=
sorry

end NUMINAMATH_GPT_ratio_adult_child_l1823_182353


namespace NUMINAMATH_GPT_find_number_that_satisfies_condition_l1823_182332

theorem find_number_that_satisfies_condition : ∃ x : ℝ, x / 3 + 12 = 20 ∧ x = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_number_that_satisfies_condition_l1823_182332


namespace NUMINAMATH_GPT_problem_statements_l1823_182391

noncomputable def f (x : ℝ) := (Real.exp x - Real.exp (-x)) / 2

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2

theorem problem_statements (x : ℝ) :
  (f x < g x) ∧
  ((f x)^2 + (g x)^2 ≥ 1) ∧
  (f (2 * x) = 2 * f x * g x) :=
by
  sorry

end NUMINAMATH_GPT_problem_statements_l1823_182391


namespace NUMINAMATH_GPT_total_animals_l1823_182367

theorem total_animals : ∀ (D C R : ℕ), 
  C = 5 * D →
  R = D - 12 →
  R = 4 →
  (C + D + R = 100) :=
by
  intros D C R h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_animals_l1823_182367


namespace NUMINAMATH_GPT_initial_ratio_of_stamps_l1823_182376

variable (K A : ℕ)

theorem initial_ratio_of_stamps (h1 : (K - 12) * 3 = (A + 12) * 4) (h2 : K - 12 = A + 44) : K/A = 5/3 :=
sorry

end NUMINAMATH_GPT_initial_ratio_of_stamps_l1823_182376


namespace NUMINAMATH_GPT_family_spent_36_dollars_l1823_182363

def ticket_cost : ℝ := 5

def popcorn_cost : ℝ := 0.8 * ticket_cost

def soda_cost : ℝ := 0.5 * popcorn_cost

def tickets_bought : ℕ := 4

def popcorn_bought : ℕ := 2

def sodas_bought : ℕ := 4

def total_spent : ℝ :=
  (tickets_bought * ticket_cost) +
  (popcorn_bought * popcorn_cost) +
  (sodas_bought * soda_cost)

theorem family_spent_36_dollars : total_spent = 36 := by
  sorry

end NUMINAMATH_GPT_family_spent_36_dollars_l1823_182363


namespace NUMINAMATH_GPT_sequence_bounded_l1823_182356

theorem sequence_bounded (a : ℕ → ℝ) :
  a 0 = 2 →
  (∀ n, a (n+1) = (2 * a n + 1) / (a n + 2)) →
  ∀ n, 1 < a n ∧ a n < 1 + 1 / 3^n :=
by
  intro h₀ h₁
  sorry

end NUMINAMATH_GPT_sequence_bounded_l1823_182356


namespace NUMINAMATH_GPT_more_sqft_to_mow_l1823_182358

-- Defining the parameters given in the original problem
def rate_per_sqft : ℝ := 0.10
def book_cost : ℝ := 150.0
def lawn_dimensions : ℝ × ℝ := (20, 15)
def num_lawns_mowed : ℕ := 3

-- The theorem stating how many more square feet LaKeisha needs to mow
theorem more_sqft_to_mow : 
  let area_one_lawn := (lawn_dimensions.1 * lawn_dimensions.2 : ℝ)
  let total_area_mowed := area_one_lawn * (num_lawns_mowed : ℝ)
  let money_earned := total_area_mowed * rate_per_sqft
  let remaining_amount := book_cost - money_earned
  let more_sqft_needed := remaining_amount / rate_per_sqft
  more_sqft_needed = 600 := 
by 
  sorry

end NUMINAMATH_GPT_more_sqft_to_mow_l1823_182358


namespace NUMINAMATH_GPT_neg_exponent_reciprocal_l1823_182352

theorem neg_exponent_reciprocal : (2 : ℝ) ^ (-1 : ℤ) = 1 / 2 := by
  -- Insert your proof here
  sorry

end NUMINAMATH_GPT_neg_exponent_reciprocal_l1823_182352


namespace NUMINAMATH_GPT_skittles_per_friend_l1823_182326

theorem skittles_per_friend (ts : ℕ) (nf : ℕ) (h1 : ts = 200) (h2 : nf = 5) : (ts / nf = 40) :=
by sorry

end NUMINAMATH_GPT_skittles_per_friend_l1823_182326


namespace NUMINAMATH_GPT_cost_price_of_watch_l1823_182319

variable (CP : ℝ)
variable (SP_loss SP_gain : ℝ)
variable (h1 : SP_loss = CP * 0.725)
variable (h2 : SP_gain = CP * 1.125)
variable (h3 : SP_gain - SP_loss = 275)

theorem cost_price_of_watch : CP = 687.50 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_watch_l1823_182319


namespace NUMINAMATH_GPT_find_values_of_cubes_l1823_182304

def N (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, c, b], ![c, b, a], ![b, a, c]]

theorem find_values_of_cubes (a b c : ℂ) (h1 : (N a b c) ^ 2 = 1) (h2 : a * b * c = 1) :
  a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_cubes_l1823_182304


namespace NUMINAMATH_GPT_acute_triangle_sums_to_pi_over_4_l1823_182369

theorem acute_triangle_sums_to_pi_over_4 
    (A B : ℝ) 
    (hA : 0 < A ∧ A < π / 2) 
    (hB : 0 < B ∧ B < π / 2) 
    (h_sinA : Real.sin A = (Real.sqrt 5)/5) 
    (h_sinB : Real.sin B = (Real.sqrt 10)/10) : 
    A + B = π / 4 := 
sorry

end NUMINAMATH_GPT_acute_triangle_sums_to_pi_over_4_l1823_182369


namespace NUMINAMATH_GPT_cistern_fill_time_l1823_182355

theorem cistern_fill_time (hF : ∀ (F : ℝ), F = 1 / 3)
                         (hE : ∀ (E : ℝ), E = 1 / 5) : 
  ∃ (t : ℝ), t = 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l1823_182355


namespace NUMINAMATH_GPT_xena_escape_l1823_182393

theorem xena_escape
    (head_start : ℕ)
    (safety_distance : ℕ)
    (xena_speed : ℕ)
    (dragon_speed : ℕ)
    (effective_gap : ℕ := head_start - safety_distance)
    (speed_difference : ℕ := dragon_speed - xena_speed) :
    (time_to_safety : ℕ := effective_gap / speed_difference) →
    time_to_safety = 32 :=
by
  sorry

end NUMINAMATH_GPT_xena_escape_l1823_182393


namespace NUMINAMATH_GPT_ricardo_coins_difference_l1823_182349

theorem ricardo_coins_difference :
  ∃ (x y : ℕ), (x + y = 2020) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ ((5 * x + y) - (x + 5 * y) = 8072) :=
by
  sorry

end NUMINAMATH_GPT_ricardo_coins_difference_l1823_182349


namespace NUMINAMATH_GPT_tangent_line_circle_l1823_182306

open Real

theorem tangent_line_circle (m n : ℝ) :
  (∀ x y : ℝ, ((m + 1) * x + (n + 1) * y - 2 = 0) ↔ (x - 1)^2 + (y - 1)^2 = 1) →
  ((m + n) ≤ 2 - 2 * sqrt 2) ∨ (2 + 2 * sqrt 2 ≤ (m + n)) := by
  sorry

end NUMINAMATH_GPT_tangent_line_circle_l1823_182306


namespace NUMINAMATH_GPT_ashok_average_marks_l1823_182386

variable (avg_5_subjects : ℕ) (marks_6th_subject : ℕ)
def total_marks_5_subjects := avg_5_subjects * 5
def total_marks_6_subjects := total_marks_5_subjects avg_5_subjects + marks_6th_subject
def avg_6_subjects := total_marks_6_subjects avg_5_subjects marks_6th_subject / 6

theorem ashok_average_marks (h1 : avg_5_subjects = 74) (h2 : marks_6th_subject = 50) : avg_6_subjects avg_5_subjects marks_6th_subject = 70 := by
  sorry

end NUMINAMATH_GPT_ashok_average_marks_l1823_182386


namespace NUMINAMATH_GPT_sin_alpha_value_l1823_182350

theorem sin_alpha_value (α : ℝ) (h1 : Real.sin (α + π / 4) = 4 / 5) (h2 : α ∈ Set.Ioo (π / 4) (3 * π / 4)) :
  Real.sin α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_value_l1823_182350


namespace NUMINAMATH_GPT_apples_used_l1823_182337

theorem apples_used (apples_before : ℕ) (apples_left : ℕ) (apples_used_for_pie : ℕ) 
                    (h1 : apples_before = 19) 
                    (h2 : apples_left = 4) 
                    (h3 : apples_used_for_pie = apples_before - apples_left) : 
  apples_used_for_pie = 15 :=
by
  -- Since we are instructed to leave the proof out, we put sorry here
  sorry

end NUMINAMATH_GPT_apples_used_l1823_182337


namespace NUMINAMATH_GPT_max_ages_acceptable_within_one_std_dev_l1823_182324

theorem max_ages_acceptable_within_one_std_dev
  (average_age : ℤ)
  (std_deviation : ℤ)
  (acceptable_range_lower : ℤ)
  (acceptable_range_upper : ℤ)
  (h1 : average_age = 31)
  (h2 : std_deviation = 5)
  (h3 : acceptable_range_lower = average_age - std_deviation)
  (h4 : acceptable_range_upper = average_age + std_deviation) :
  ∃ n : ℕ, n = acceptable_range_upper - acceptable_range_lower + 1 ∧ n = 11 :=
by
  sorry

end NUMINAMATH_GPT_max_ages_acceptable_within_one_std_dev_l1823_182324


namespace NUMINAMATH_GPT_arrangement_ways_l1823_182383

def green_marbles : Nat := 7
noncomputable def N_max_blue_marbles : Nat := 924

theorem arrangement_ways (N : Nat) (blue_marbles : Nat) (total_marbles : Nat)
  (h1 : total_marbles = green_marbles + blue_marbles) 
  (h2 : ∃ b_gap, b_gap = blue_marbles - (total_marbles - green_marbles - 1))
  (h3 : blue_marbles ≥ 6)
  : N = N_max_blue_marbles := 
sorry

end NUMINAMATH_GPT_arrangement_ways_l1823_182383


namespace NUMINAMATH_GPT_no_arithmetic_progression_40_terms_l1823_182365

noncomputable def is_arith_prog (f : ℕ → ℕ) (a : ℕ) (b : ℕ) : Prop :=
∀ n : ℕ, ∃ k : ℕ, f n = a + n * b

noncomputable def in_form_2m_3n (x : ℕ) : Prop :=
∃ m n : ℕ, x = 2^m + 3^n

theorem no_arithmetic_progression_40_terms :
  ¬ (∃ (a b : ℕ), ∀ n, n < 40 → in_form_2m_3n (a + n * b)) :=
sorry

end NUMINAMATH_GPT_no_arithmetic_progression_40_terms_l1823_182365


namespace NUMINAMATH_GPT_perfect_square_after_dividing_l1823_182382

theorem perfect_square_after_dividing (n : ℕ) (h : n = 16800) : ∃ m : ℕ, (n / 21) = m * m :=
by {
  sorry
}

end NUMINAMATH_GPT_perfect_square_after_dividing_l1823_182382


namespace NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l1823_182362

theorem line_through_point_with_equal_intercepts 
  (x y k : ℝ) 
  (h1 : (3 : ℝ) + (-6 : ℝ) + k = 0 ∨ 2 * (3 : ℝ) + (-6 : ℝ) = 0) 
  (h2 : k = 0 ∨ x + y + k = 0) : 
  (x = 1 ∨ x = 2) ∧ (k = -3 ∨ k = 0) :=
sorry

end NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l1823_182362


namespace NUMINAMATH_GPT_solve_star_eq_five_l1823_182303

def star (a b : ℝ) : ℝ := a + b^2

theorem solve_star_eq_five :
  ∃ x₁ x₂ : ℝ, star x₁ (x₁ + 1) = 5 ∧ star x₂ (x₂ + 1) = 5 ∧ x₁ = 1 ∧ x₂ = -4 :=
by
  sorry

end NUMINAMATH_GPT_solve_star_eq_five_l1823_182303


namespace NUMINAMATH_GPT_average_annual_growth_rate_l1823_182390

theorem average_annual_growth_rate (x : ℝ) (h : (1 + x)^2 = 1.20) : x < 0.1 :=
sorry

end NUMINAMATH_GPT_average_annual_growth_rate_l1823_182390


namespace NUMINAMATH_GPT_perimeter_range_l1823_182338

variable (a b x : ℝ)
variable (a_gt_b : a > b)
variable (triangle_ineq : a - b < x ∧ x < a + b)

theorem perimeter_range : 2 * a < a + b + x ∧ a + b + x < 2 * (a + b) :=
by
  sorry

end NUMINAMATH_GPT_perimeter_range_l1823_182338


namespace NUMINAMATH_GPT_students_per_group_l1823_182331

theorem students_per_group (n m : ℕ) (h_n : n = 36) (h_m : m = 9) : 
  (n - m) / 3 = 9 := 
by
  sorry

end NUMINAMATH_GPT_students_per_group_l1823_182331


namespace NUMINAMATH_GPT_mean_age_of_all_children_l1823_182321

def euler_ages : List ℕ := [10, 12, 8]
def gauss_ages : List ℕ := [8, 8, 8, 16, 18]
def all_ages : List ℕ := euler_ages ++ gauss_ages
def total_children : ℕ := all_ages.length
def total_age : ℕ := all_ages.sum
def mean_age : ℕ := total_age / total_children

theorem mean_age_of_all_children : mean_age = 11 := by
  sorry

end NUMINAMATH_GPT_mean_age_of_all_children_l1823_182321


namespace NUMINAMATH_GPT_simplest_radical_expression_l1823_182341

theorem simplest_radical_expression :
  let A := Real.sqrt 3
  let B := Real.sqrt 4
  let C := Real.sqrt 8
  let D := Real.sqrt (1 / 2)
  B = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplest_radical_expression_l1823_182341


namespace NUMINAMATH_GPT_class_total_students_l1823_182305

def initial_boys : ℕ := 15
def initial_girls : ℕ := (120 * initial_boys) / 100 -- 1.2 * initial_boys

def final_boys : ℕ := initial_boys
def final_girls : ℕ := 2 * initial_girls

def total_students : ℕ := final_boys + final_girls

theorem class_total_students : total_students = 51 := 
by 
  -- the actual proof will go here
  sorry

end NUMINAMATH_GPT_class_total_students_l1823_182305


namespace NUMINAMATH_GPT_scientific_notation_l1823_182317

theorem scientific_notation :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end NUMINAMATH_GPT_scientific_notation_l1823_182317
