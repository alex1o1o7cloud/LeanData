import Mathlib

namespace NUMINAMATH_GPT_cost_per_pie_eq_l960_96032

-- We define the conditions
def price_per_piece : ℝ := 4
def pieces_per_pie : ℕ := 3
def pies_per_hour : ℕ := 12
def actual_revenue : ℝ := 138

-- Lean theorem statement
theorem cost_per_pie_eq : (price_per_piece * pieces_per_pie * pies_per_hour - actual_revenue) / pies_per_hour = 0.50 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_cost_per_pie_eq_l960_96032


namespace NUMINAMATH_GPT_dot_product_computation_l960_96042

open Real

variables (a b : ℝ) (θ : ℝ)

noncomputable def dot_product (u v : ℝ) : ℝ :=
  u * v * cos θ

noncomputable def magnitude (v : ℝ) : ℝ :=
  abs v

theorem dot_product_computation (a b : ℝ) (h1 : θ = 120) (h2 : magnitude a = 4) (h3 : magnitude b = 4) :
  dot_product b (3 * a + b) = -8 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_computation_l960_96042


namespace NUMINAMATH_GPT_jump_rope_cost_l960_96039

def cost_board_game : ℕ := 12
def cost_playground_ball : ℕ := 4
def saved_money : ℕ := 6
def uncle_money : ℕ := 13
def additional_needed : ℕ := 4

theorem jump_rope_cost :
  let total_money := saved_money + uncle_money
  let total_needed := total_money + additional_needed
  let combined_cost := cost_board_game + cost_playground_ball
  let cost_jump_rope := total_needed - combined_cost
  cost_jump_rope = 7 := by
  sorry

end NUMINAMATH_GPT_jump_rope_cost_l960_96039


namespace NUMINAMATH_GPT_carl_teaches_periods_l960_96007

theorem carl_teaches_periods (cards_per_student : ℕ) (students_per_class : ℕ) (pack_cost : ℕ) (amount_spent : ℕ) (cards_per_pack : ℕ) :
  cards_per_student = 10 →
  students_per_class = 30 →
  pack_cost = 3 →
  amount_spent = 108 →
  cards_per_pack = 50 →
  (amount_spent / pack_cost) * cards_per_pack / (cards_per_student * students_per_class) = 6 :=
by
  intros hc hs hp ha hpkg
  /- proof steps would go here -/
  sorry

end NUMINAMATH_GPT_carl_teaches_periods_l960_96007


namespace NUMINAMATH_GPT_friends_bought_color_box_l960_96028

variable (total_pencils : ℕ) (pencils_per_box : ℕ) (chloe_pencils : ℕ)

theorem friends_bought_color_box : 
  (total_pencils = 42) → 
  (pencils_per_box = 7) → 
  (chloe_pencils = pencils_per_box) → 
  (total_pencils - chloe_pencils) / pencils_per_box = 5 := 
by 
  intros ht hb hc
  sorry

end NUMINAMATH_GPT_friends_bought_color_box_l960_96028


namespace NUMINAMATH_GPT_simplified_expression_l960_96037

theorem simplified_expression :
  (0.2 * 0.4 - 0.3 / 0.5) + (0.6 * 0.8 + 0.1 / 0.2) - 0.9 * (0.3 - 0.2 * 0.4) = 0.262 :=
by
  sorry

end NUMINAMATH_GPT_simplified_expression_l960_96037


namespace NUMINAMATH_GPT_sum_distances_saham_and_mother_l960_96089

theorem sum_distances_saham_and_mother :
  let saham_distance := 2.6
  let mother_distance := 5.98
  saham_distance + mother_distance = 8.58 :=
by
  sorry

end NUMINAMATH_GPT_sum_distances_saham_and_mother_l960_96089


namespace NUMINAMATH_GPT_invertible_my_matrix_l960_96057

def my_matrix : Matrix (Fin 2) (Fin 2) ℚ := ![![4, 5], ![-2, 9]]

noncomputable def inverse_of_my_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  Matrix.det my_matrix • Matrix.adjugate my_matrix

theorem invertible_my_matrix :
  inverse_of_my_matrix = (1 / 46 : ℚ) • ![![9, -5], ![2, 4]] :=
by
  sorry

end NUMINAMATH_GPT_invertible_my_matrix_l960_96057


namespace NUMINAMATH_GPT_set_representation_l960_96084

def is_Natural (n : ℕ) : Prop :=
  n ≠ 0

def condition (x : ℕ) : Prop :=
  x^2 - 3*x < 0

theorem set_representation :
  {x : ℕ | condition x ∧ is_Natural x} = {1, 2} := 
sorry

end NUMINAMATH_GPT_set_representation_l960_96084


namespace NUMINAMATH_GPT_valentines_given_l960_96014

theorem valentines_given (original current given : ℕ) (h1 : original = 58) (h2 : current = 16) (h3 : given = original - current) : given = 42 := by
  sorry

end NUMINAMATH_GPT_valentines_given_l960_96014


namespace NUMINAMATH_GPT_fraction_to_terminating_decimal_l960_96055

theorem fraction_to_terminating_decimal :
  (45 / (2^2 * 5^3) : ℚ) = 0.09 :=
by sorry

end NUMINAMATH_GPT_fraction_to_terminating_decimal_l960_96055


namespace NUMINAMATH_GPT_integer_roots_of_quadratic_l960_96052

theorem integer_roots_of_quadratic (b : ℤ) :
  (∃ x : ℤ, x^2 + 4 * x + b = 0) ↔ b = -12 ∨ b = -5 ∨ b = 3 ∨ b = 4 :=
sorry

end NUMINAMATH_GPT_integer_roots_of_quadratic_l960_96052


namespace NUMINAMATH_GPT_always_true_inequality_l960_96030

theorem always_true_inequality (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by
  sorry

end NUMINAMATH_GPT_always_true_inequality_l960_96030


namespace NUMINAMATH_GPT_correct_option_d_l960_96067

-- Define the conditions as separate lemmas
lemma option_a_incorrect : ¬ (Real.sqrt 18 + Real.sqrt 2 = 2 * Real.sqrt 5) :=
sorry 

lemma option_b_incorrect : ¬ (Real.sqrt 18 - Real.sqrt 2 = 4) :=
sorry

lemma option_c_incorrect : ¬ (Real.sqrt 18 * Real.sqrt 2 = 36) :=
sorry

-- Define the statement to prove
theorem correct_option_d : Real.sqrt 18 / Real.sqrt 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_d_l960_96067


namespace NUMINAMATH_GPT_dog_has_fewer_lives_than_cat_l960_96015

noncomputable def cat_lives : ℕ := 9
noncomputable def mouse_lives : ℕ := 13
noncomputable def dog_lives : ℕ := mouse_lives - 7
noncomputable def dog_less_lives : ℕ := cat_lives - dog_lives

theorem dog_has_fewer_lives_than_cat : dog_less_lives = 3 := by
  sorry

end NUMINAMATH_GPT_dog_has_fewer_lives_than_cat_l960_96015


namespace NUMINAMATH_GPT_max_non_real_roots_l960_96061

theorem max_non_real_roots (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (A B : ℕ → ℕ) (h_turns : ∀ i < 3 * n, A i + B i = 1),
    (∀ i, (A i + B (i + 1)) % 3 = 0) →
    ∃ k, ∀ m, ∃ j < n, j % 2 = 1 → j + m * 2 ≤ 2 * k + j - m)
  → (∃ k, k = (n + 1) / 2) :=
sorry

end NUMINAMATH_GPT_max_non_real_roots_l960_96061


namespace NUMINAMATH_GPT_sum_of_a_b_c_l960_96043

theorem sum_of_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc1 : a * b + c = 47) (habc2 : b * c + a = 47) (habc3 : a * c + b = 47) : a + b + c = 48 := 
sorry

end NUMINAMATH_GPT_sum_of_a_b_c_l960_96043


namespace NUMINAMATH_GPT_first_group_size_l960_96006

theorem first_group_size
  (x : ℕ)
  (h1 : 2 * x + 22 + 16 + 14 = 68) : 
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_first_group_size_l960_96006


namespace NUMINAMATH_GPT_business_fraction_l960_96038

theorem business_fraction (x : ℚ) (H1 : 3 / 4 * x * 60000 = 30000) : x = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_business_fraction_l960_96038


namespace NUMINAMATH_GPT_split_tips_evenly_l960_96022

theorem split_tips_evenly :
  let julie_cost := 10
  let letitia_cost := 20
  let anton_cost := 30
  let total_cost := julie_cost + letitia_cost + anton_cost
  let tip_rate := 0.2
  let total_tip := total_cost * tip_rate
  let tip_per_person := total_tip / 3
  tip_per_person = 4 := by
  sorry

end NUMINAMATH_GPT_split_tips_evenly_l960_96022


namespace NUMINAMATH_GPT_probability_blue_is_4_over_13_l960_96056

def num_red : ℕ := 5
def num_green : ℕ := 6
def num_yellow : ℕ := 7
def num_blue : ℕ := 8
def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue

def probability_blue : ℚ := num_blue / total_jelly_beans

theorem probability_blue_is_4_over_13
  (h_num_red : num_red = 5)
  (h_num_green : num_green = 6)
  (h_num_yellow : num_yellow = 7)
  (h_num_blue : num_blue = 8) :
  probability_blue = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_blue_is_4_over_13_l960_96056


namespace NUMINAMATH_GPT_average_time_correct_l960_96013

-- Define the times for each runner
def y_time : ℕ := 58
def z_time : ℕ := 26
def w_time : ℕ := 2 * z_time

-- Define the number of runners
def num_runners : ℕ := 3

-- Calculate the summed time of all runners
def total_time : ℕ := y_time + z_time + w_time

-- Calculate the average time
def average_time : ℚ := total_time / num_runners

-- Statement of the proof problem
theorem average_time_correct : average_time = 45.33 := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_average_time_correct_l960_96013


namespace NUMINAMATH_GPT_jenny_proposal_time_l960_96036

theorem jenny_proposal_time (total_time research_time report_time proposal_time : ℕ) 
  (h1 : total_time = 20) 
  (h2 : research_time = 10) 
  (h3 : report_time = 8) 
  (h4 : proposal_time = total_time - research_time - report_time) : 
  proposal_time = 2 := 
by
  sorry

end NUMINAMATH_GPT_jenny_proposal_time_l960_96036


namespace NUMINAMATH_GPT_solve_for_y_l960_96096

theorem solve_for_y (y : ℝ) : (10 - y) ^ 2 = 4 * y ^ 2 → y = 10 / 3 ∨ y = -10 :=
by
  intro h
  -- The proof steps would go here, but we include sorry to allow for compilation.
  sorry

end NUMINAMATH_GPT_solve_for_y_l960_96096


namespace NUMINAMATH_GPT_geometric_sequence_sum_l960_96094

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q a2005 a2006 : ℝ), 
    (∀ n, a (n + 1) = a n * q) ∧
    q > 1 ∧
    a2005 + a2006 = 2 ∧ 
    a2005 * a2006 = 3 / 4 ∧ 
    a (2005) = a2005 ∧ 
    a (2006) = a2006 → 
    a (2007) + a (2008) = 18 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l960_96094


namespace NUMINAMATH_GPT_determine_OP_l960_96012

variable (a b c d : ℝ)
variable (O A B C D P : ℝ)
variable (p : ℝ)

def OnLine (O A B C D P : ℝ) : Prop := O < A ∧ A < B ∧ B < C ∧ C < D ∧ B < P ∧ P < C

theorem determine_OP (h : OnLine O A B C D P) 
(hAP : P - A = p - a) 
(hPD : D - P = d - p) 
(hBP : P - B = p - b) 
(hPC : C - P = c - p) 
(hAP_PD_BP_PC : (p - a) / (d - p) = (p - b) / (c - p)) :
  p = (a * c - b * d) / (a - b + c - d) :=
sorry

end NUMINAMATH_GPT_determine_OP_l960_96012


namespace NUMINAMATH_GPT_smallest_n_divisible_l960_96065

theorem smallest_n_divisible (n : ℕ) : 
  (450 ∣ n^3) ∧ (2560 ∣ n^4) ↔ n = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_n_divisible_l960_96065


namespace NUMINAMATH_GPT_polynomial_value_at_neg3_l960_96048

def polynomial (a b c x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 7

theorem polynomial_value_at_neg3 (a b c : ℝ) (h : polynomial a b c 3 = 65) :
  polynomial a b c (-3) = -79 := 
sorry

end NUMINAMATH_GPT_polynomial_value_at_neg3_l960_96048


namespace NUMINAMATH_GPT_quadratic_equation_m_value_l960_96073

theorem quadratic_equation_m_value (m : ℝ) (h : m ≠ 2) : m = -2 :=
by
  -- details of the proof go here
  sorry

end NUMINAMATH_GPT_quadratic_equation_m_value_l960_96073


namespace NUMINAMATH_GPT_fraction_evaluation_l960_96064

theorem fraction_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/5) = 5/8 := by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l960_96064


namespace NUMINAMATH_GPT_integer_equality_condition_l960_96024

theorem integer_equality_condition
  (x y z : ℤ)
  (h : x * (x - y) + y * (y - z) + z * (z - x) = 0) :
  x = y ∧ y = z :=
sorry

end NUMINAMATH_GPT_integer_equality_condition_l960_96024


namespace NUMINAMATH_GPT_original_number_is_40_l960_96095

theorem original_number_is_40 (x : ℝ) (h : 1.25 * x - 0.70 * x = 22) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_40_l960_96095


namespace NUMINAMATH_GPT_B_visits_A_l960_96031

/-- Students A, B, and C were surveyed on whether they have visited cities A, B, and C -/
def student_visits_city (student : Type) (city : Type) : Prop := sorry -- assume there's a definition

variables (A_student B_student C_student : Type) (city_A city_B city_C : Type)

variables 
  -- A's statements
  (A_visits_more_than_B : student_visits_city A_student city_A → ¬ student_visits_city A_student city_B → ∃ city, student_visits_city B_student city ∧ ¬ student_visits_city A_student city)
  (A_not_visit_B : ¬ student_visits_city A_student city_B)
  -- B's statement
  (B_not_visit_C : ¬ student_visits_city B_student city_C)
  -- C's statement
  (all_three_same_city : student_visits_city A_student city_A → student_visits_city B_student city_A → student_visits_city C_student city_A)

theorem B_visits_A : student_visits_city B_student city_A :=
by
  sorry

end NUMINAMATH_GPT_B_visits_A_l960_96031


namespace NUMINAMATH_GPT_operation_result_l960_96085

-- Define x and the operations
def x : ℕ := 40

-- Define the operation sequence
def operation (y : ℕ) : ℕ :=
  let step1 := y / 4
  let step2 := step1 * 5
  let step3 := step2 + 10
  let step4 := step3 - 12
  step4

-- The statement we need to prove
theorem operation_result : operation x = 48 := by
  sorry

end NUMINAMATH_GPT_operation_result_l960_96085


namespace NUMINAMATH_GPT_monotonicity_of_f_sum_of_squares_of_roots_l960_96003

noncomputable def f (x a : Real) : Real := Real.log x - a * x^2

theorem monotonicity_of_f (a : Real) :
  (a ≤ 0 → ∀ x y : Real, 0 < x → x < y → f x a < f y a) ∧
  (a > 0 → ∀ x y : Real, 0 < x → x < Real.sqrt (1/(2 * a)) → Real.sqrt (1/(2 * a)) < y → f x a < f (Real.sqrt (1/(2 * a))) a ∧ f (Real.sqrt (1/(2 * a))) a > f y a) :=
by sorry

theorem sum_of_squares_of_roots (a x1 x2 : Real) (h1 : f x1 a = 0) (h2 : f x2 a = 0) (h3 : x1 ≠ x2) :
  x1^2 + x2^2 > 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_GPT_monotonicity_of_f_sum_of_squares_of_roots_l960_96003


namespace NUMINAMATH_GPT_carnival_game_ratio_l960_96088

theorem carnival_game_ratio (L W : ℕ) (h_ratio : 4 * L = W) (h_lost : L = 7) : W = 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_carnival_game_ratio_l960_96088


namespace NUMINAMATH_GPT_find_first_number_l960_96049

theorem find_first_number
  (avg1 : (20 + 40 + 60) / 3 = 40)
  (avg2 : 40 - 4 = (x + 70 + 28) / 3)
  (sum_eq : x + 70 + 28 = 108) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l960_96049


namespace NUMINAMATH_GPT_percent_game_of_thrones_altered_l960_96087

def votes_game_of_thrones : ℕ := 10
def votes_twilight : ℕ := 12
def votes_art_of_deal : ℕ := 20

def altered_votes_art_of_deal : ℕ := votes_art_of_deal - (votes_art_of_deal * 80 / 100)
def altered_votes_twilight : ℕ := votes_twilight / 2
def total_altered_votes : ℕ := altered_votes_art_of_deal + altered_votes_twilight + votes_game_of_thrones

theorem percent_game_of_thrones_altered :
  ((votes_game_of_thrones * 100) / total_altered_votes) = 50 := by
  sorry

end NUMINAMATH_GPT_percent_game_of_thrones_altered_l960_96087


namespace NUMINAMATH_GPT_time_to_reach_ship_l960_96068

/-- The scuba diver's descent problem -/

def rate_of_descent : ℕ := 35  -- in feet per minute
def depth_of_ship : ℕ := 3500  -- in feet

theorem time_to_reach_ship : depth_of_ship / rate_of_descent = 100 := by
  sorry

end NUMINAMATH_GPT_time_to_reach_ship_l960_96068


namespace NUMINAMATH_GPT_num_women_in_luxury_suite_l960_96050

theorem num_women_in_luxury_suite (total_passengers : ℕ) (pct_women : ℕ) (pct_women_luxury : ℕ)
  (h_total_passengers : total_passengers = 300)
  (h_pct_women : pct_women = 50)
  (h_pct_women_luxury : pct_women_luxury = 15) :
  (total_passengers * pct_women / 100) * pct_women_luxury / 100 = 23 := 
by
  sorry

end NUMINAMATH_GPT_num_women_in_luxury_suite_l960_96050


namespace NUMINAMATH_GPT_denominator_exceeds_numerator_by_263_l960_96062

def G : ℚ := 736 / 999

theorem denominator_exceeds_numerator_by_263 : 999 - 736 = 263 := by
  -- Since 736 / 999 is the simplest form already, we simply state the obvious difference
  rfl

end NUMINAMATH_GPT_denominator_exceeds_numerator_by_263_l960_96062


namespace NUMINAMATH_GPT_price_second_oil_per_litre_is_correct_l960_96066

-- Definitions based on conditions
def price_first_oil_per_litre := 54
def volume_first_oil := 10
def volume_second_oil := 5
def mixture_rate_per_litre := 58
def total_volume := volume_first_oil + volume_second_oil
def total_cost_mixture := total_volume * mixture_rate_per_litre
def total_cost_first_oil := volume_first_oil * price_first_oil_per_litre

-- The statement to prove
theorem price_second_oil_per_litre_is_correct (x : ℕ) (h : total_cost_first_oil + (volume_second_oil * x) = total_cost_mixture) : x = 66 :=
by
  sorry

end NUMINAMATH_GPT_price_second_oil_per_litre_is_correct_l960_96066


namespace NUMINAMATH_GPT_intersection_A_B_l960_96025

def A : Set ℝ := {y | ∃ x : ℝ, y = x ^ (1 / 3)}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_B :
  A ∩ B = {x | x > 1} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l960_96025


namespace NUMINAMATH_GPT_question1_question2_l960_96080

variable (a : ℤ)
def point_P : (ℤ × ℤ) := (2*a - 2, a + 5)

-- Part 1: If point P lies on the x-axis, its coordinates are (-12, 0).
theorem question1 (h1 : a + 5 = 0) : point_P a = (-12, 0) :=
sorry

-- Part 2: If point P lies in the second quadrant and the distance from point P to the x-axis is equal to the distance from point P to the y-axis,
-- the value of a^2023 + 2023 is 2022.
theorem question2 (h2 : 2*a - 2 < 0) (h3 : -(2*a - 2) = a + 5) : a ^ 2023 + 2023 = 2022 :=
sorry

end NUMINAMATH_GPT_question1_question2_l960_96080


namespace NUMINAMATH_GPT_number_of_real_solutions_l960_96092

theorem number_of_real_solutions (floor : ℝ → ℤ) 
  (h_floor : ∀ x, floor x = ⌊x⌋)
  (h_eq : ∀ x, 9 * x^2 - 45 * floor (x^2 - 1) + 94 = 0) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_real_solutions_l960_96092


namespace NUMINAMATH_GPT_Jake_width_proof_l960_96069

-- Define the dimensions of Sara's birdhouse in feet
def Sara_width_feet := 1
def Sara_height_feet := 2
def Sara_depth_feet := 2

-- Convert the dimensions to inches
def Sara_width_inch := Sara_width_feet * 12
def Sara_height_inch := Sara_height_feet * 12
def Sara_depth_inch := Sara_depth_feet * 12

-- Calculate Sara's birdhouse volume
def Sara_volume := Sara_width_inch * Sara_height_inch * Sara_depth_inch

-- Define the dimensions of Jake's birdhouse in inches
def Jake_height_inch := 20
def Jake_depth_inch := 18
def Jake_volume (Jake_width_inch : ℝ) := Jake_width_inch * Jake_height_inch * Jake_depth_inch

-- Difference in volume
def volume_difference := 1152

-- Prove the width of Jake's birdhouse
theorem Jake_width_proof : ∃ (W : ℝ), Jake_volume W - Sara_volume = volume_difference ∧ W = 22.4 := by
  sorry

end NUMINAMATH_GPT_Jake_width_proof_l960_96069


namespace NUMINAMATH_GPT_number_of_days_same_l960_96021

-- Defining volumes as given in the conditions.
def volume_project1 : ℕ := 100 * 25 * 30
def volume_project2 : ℕ := 75 * 20 * 50

-- The mathematical statement we want to prove.
theorem number_of_days_same : volume_project1 = volume_project2 → ∀ d : ℕ, d > 0 → d = d :=
by
  sorry

end NUMINAMATH_GPT_number_of_days_same_l960_96021


namespace NUMINAMATH_GPT_builder_windows_installed_l960_96098

theorem builder_windows_installed (total_windows : ℕ) (hours_per_window : ℕ) (total_hours_left : ℕ) :
  total_windows = 14 → hours_per_window = 4 → total_hours_left = 36 → (total_windows - total_hours_left / hours_per_window) = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_builder_windows_installed_l960_96098


namespace NUMINAMATH_GPT_isosceles_triangles_possible_l960_96091

theorem isosceles_triangles_possible :
  ∃ (sticks : List ℕ), (sticks = [1, 1, 2, 2, 3, 3] ∧ 
    ∀ (a b c : ℕ), a ∈ sticks → b ∈ sticks → c ∈ sticks → 
    ((a + b > c ∧ b + c > a ∧ c + a > b) → a = b ∨ b = c ∨ c = a)) :=
sorry

end NUMINAMATH_GPT_isosceles_triangles_possible_l960_96091


namespace NUMINAMATH_GPT_interest_rate_A_l960_96010

-- Given conditions
variables (Principal : ℝ := 4000)
variables (interestRate_C : ℝ := 11.5 / 100)
variables (gain_B : ℝ := 180)
variables (time : ℝ := 3)
variables (interest_from_C : ℝ := Principal * interestRate_C * time)
variables (interest_to_A : ℝ := interest_from_C - gain_B)

-- The proof goal
theorem interest_rate_A (R : ℝ) : 
  1200 = Principal * (R / 100) * time → 
  R = 10 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_A_l960_96010


namespace NUMINAMATH_GPT_probability_A_given_B_l960_96018

namespace ProbabilityProof

def total_parts : ℕ := 100
def A_parts_produced : ℕ := 0
def A_parts_qualified : ℕ := 35
def B_parts_produced : ℕ := 60
def B_parts_qualified : ℕ := 50

def event_A (x : ℕ) : Prop := x ≤ B_parts_qualified + A_parts_qualified
def event_B (x : ℕ) : Prop := x ≤ A_parts_produced

-- Formalizing the probability condition P(A | B) = 7/8, logically this should be revised with practical events.
theorem probability_A_given_B : (event_B x → event_A x) := sorry

end ProbabilityProof

end NUMINAMATH_GPT_probability_A_given_B_l960_96018


namespace NUMINAMATH_GPT_similar_right_triangles_l960_96070

theorem similar_right_triangles (x c : ℕ) 
  (h1 : 12 * 6 = 9 * x) 
  (h2 : c^2 = x^2 + 6^2) :
  x = 8 ∧ c = 10 :=
by
  sorry

end NUMINAMATH_GPT_similar_right_triangles_l960_96070


namespace NUMINAMATH_GPT_octagon_area_in_square_l960_96045

/--
An octagon is inscribed in a square such that each vertex of the octagon cuts off a corner
triangle from the square. Each triangle has legs equal to one-fourth of the square's side.
If the perimeter of the square is 160 centimeters, what is the area of the octagon?
-/
theorem octagon_area_in_square
  (side_of_square : ℝ)
  (h1 : 4 * (side_of_square / 4) = side_of_square)
  (h2 : 8 * (side_of_square / 4) = side_of_square)
  (perimeter_of_square : ℝ)
  (h3 : perimeter_of_square = 160)
  (area_of_square : ℝ)
  (h4 : area_of_square = side_of_square^2)
  : ∃ (area_of_octagon : ℝ), area_of_octagon = 1400 := by
  sorry

end NUMINAMATH_GPT_octagon_area_in_square_l960_96045


namespace NUMINAMATH_GPT_double_x_value_l960_96023

theorem double_x_value (x : ℝ) (h : x / 2 = 32) : 2 * x = 128 := by
  sorry

end NUMINAMATH_GPT_double_x_value_l960_96023


namespace NUMINAMATH_GPT_solution_f_derivative_l960_96077

noncomputable def f (x : ℝ) := Real.sqrt x

theorem solution_f_derivative :
  (deriv f 1) = 1 / 2 :=
by
  -- This is where the proof would go, but for now, we just state sorry.
  sorry

end NUMINAMATH_GPT_solution_f_derivative_l960_96077


namespace NUMINAMATH_GPT_min_omega_value_l960_96076

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega_value (ω : ℝ) (φ : ℝ) (h_ω_pos : ω > 0)
  (h_even : ∀ x : ℝ, f ω φ x = f ω φ (-x))
  (h_symmetry : f ω φ 1 = 0 ∧ ∀ x : ℝ, f ω φ (1 + x) = - f ω φ (1 - x)) :
  ω = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_omega_value_l960_96076


namespace NUMINAMATH_GPT_percentage_loss_l960_96081

variable (CP SP : ℝ) (Loss : ℝ := CP - SP) (Percentage_of_Loss : ℝ := (Loss / CP) * 100)

theorem percentage_loss (h1: CP = 1600) (h2: SP = 1440) : Percentage_of_Loss = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_loss_l960_96081


namespace NUMINAMATH_GPT_comics_in_box_l960_96072

def comics_per_comic := 25
def total_pages := 150
def existing_comics := 5

def torn_comics := total_pages / comics_per_comic
def total_comics := torn_comics + existing_comics

theorem comics_in_box : total_comics = 11 := by
  sorry

end NUMINAMATH_GPT_comics_in_box_l960_96072


namespace NUMINAMATH_GPT_peanuts_in_box_after_addition_l960_96075

theorem peanuts_in_box_after_addition : 4 + 12 = 16 := by
  sorry

end NUMINAMATH_GPT_peanuts_in_box_after_addition_l960_96075


namespace NUMINAMATH_GPT_number_of_apples_l960_96005

theorem number_of_apples (C : ℝ) (A : ℕ) (total_cost : ℝ) (price_diff : ℝ) (num_oranges : ℕ)
  (h_price : C = 0.26)
  (h_price_diff : price_diff = 0.28)
  (h_num_oranges : num_oranges = 7)
  (h_total_cost : total_cost = 4.56) :
  A * C + num_oranges * (C + price_diff) = total_cost → A = 3 := 
by
  sorry

end NUMINAMATH_GPT_number_of_apples_l960_96005


namespace NUMINAMATH_GPT_angle_A_value_l960_96011

/-- 
In triangle ABC, the sides opposite to angles A, B, C are a, b, and c respectively.
Given:
  - C = π / 3,
  - b = √6,
  - c = 3,
Prove that A = 5π / 12.
-/
theorem angle_A_value (a b c : ℝ) (A B C : ℝ) (hC : C = Real.pi / 3) (hb : b = Real.sqrt 6) (hc : c = 3) :
  A = 5 * Real.pi / 12 :=
sorry

end NUMINAMATH_GPT_angle_A_value_l960_96011


namespace NUMINAMATH_GPT_range_of_x_l960_96097

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) :
    Real.pi / 4 ≤ x ∧ x ≤ 5 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l960_96097


namespace NUMINAMATH_GPT_bianca_marathon_total_miles_l960_96034

theorem bianca_marathon_total_miles : 8 + 4 = 12 :=
by
  sorry

end NUMINAMATH_GPT_bianca_marathon_total_miles_l960_96034


namespace NUMINAMATH_GPT_linear_function_above_x_axis_l960_96041

theorem linear_function_above_x_axis (a : ℝ) :
  (-1 < a ∧ a < 2 ∧ a ≠ 0) ↔
  (∀ x, -2 ≤ x ∧ x ≤ 1 → ax + a + 2 > 0) :=
sorry

end NUMINAMATH_GPT_linear_function_above_x_axis_l960_96041


namespace NUMINAMATH_GPT_abhay_speed_l960_96093

-- Definitions of the problem's conditions
def condition1 (A S : ℝ) : Prop := 42 / A = 42 / S + 2
def condition2 (A S : ℝ) : Prop := 42 / (2 * A) = 42 / S - 1

-- Define Abhay and Sameer's speeds and declare the main theorem
theorem abhay_speed (A S : ℝ) (h1 : condition1 A S) (h2 : condition2 A S) : A = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_abhay_speed_l960_96093


namespace NUMINAMATH_GPT_combined_salaries_l960_96086

variable {A B C E : ℝ}
variable (D : ℝ := 7000)
variable (average_salary : ℝ := 8400)
variable (n : ℕ := 5)

theorem combined_salaries (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ E) 
  (h4 : B ≠ C) (h5 : B ≠ E) (h6 : C ≠ E)
  (h7 : average_salary = (A + B + C + D + E) / n) :
  A + B + C + E = 35000 :=
by
  sorry

end NUMINAMATH_GPT_combined_salaries_l960_96086


namespace NUMINAMATH_GPT_determine_b_l960_96017

noncomputable def f (x b : ℝ) : ℝ := 1 / (3 * x + b)

noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, f (f_inv x) b = x) -> b = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_b_l960_96017


namespace NUMINAMATH_GPT_largest_pentagon_angle_is_179_l960_96082

-- Define the interior angles of the pentagon
def angle1 (x : ℝ) := x + 2
def angle2 (x : ℝ) := 2 * x + 3
def angle3 (x : ℝ) := 3 * x - 5
def angle4 (x : ℝ) := 4 * x + 1
def angle5 (x : ℝ) := 5 * x - 1

-- Define the sum of the interior angles of a pentagon
def pentagon_angle_sum := angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36

-- Define the largest angle function
def largest_angle (x : ℝ) := 5 * x - 1

-- The main theorem stating the largest angle measure
theorem largest_pentagon_angle_is_179 (h : angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36 = 540) :
  largest_angle 36 = 179 :=
sorry

end NUMINAMATH_GPT_largest_pentagon_angle_is_179_l960_96082


namespace NUMINAMATH_GPT_total_trucks_l960_96008

theorem total_trucks {t : ℕ} (h1 : 2 * t + t = 300) : t = 100 := 
by sorry

end NUMINAMATH_GPT_total_trucks_l960_96008


namespace NUMINAMATH_GPT_area_bounded_by_parabola_and_x_axis_l960_96033

/-- Define the parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x - x^2

/-- The function for the x-axis -/
def x_axis : ℝ := 0

/-- Prove that the area bounded by the parabola and x-axis between x = 0 and x = 2 is 4/3 -/
theorem area_bounded_by_parabola_and_x_axis : 
  (∫ x in (0 : ℝ)..(2 : ℝ), parabola x) = 4 / 3 := by
    sorry

end NUMINAMATH_GPT_area_bounded_by_parabola_and_x_axis_l960_96033


namespace NUMINAMATH_GPT_sum_of_squares_not_square_l960_96029

theorem sum_of_squares_not_square (a : ℕ) : 
  ¬ ∃ b : ℕ, (a - 1)^2 + a^2 + (a + 1)^2 = b^2 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_squares_not_square_l960_96029


namespace NUMINAMATH_GPT_arithmetic_sequence_difference_l960_96047

noncomputable def arithmetic_difference (d: ℚ) (b₁: ℚ) : Prop :=
  (50 * b₁ + ((50 * 49) / 2) * d = 150) ∧
  (50 * (b₁ + 50 * d) + ((50 * 149) / 2) * d = 250)

theorem arithmetic_sequence_difference {d b₁ : ℚ} (h : arithmetic_difference d b₁) :
  (b₁ + d) - b₁ = (200 / 1295) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_difference_l960_96047


namespace NUMINAMATH_GPT_knights_rearrangement_impossible_l960_96054

theorem knights_rearrangement_impossible :
  ∀ (b : ℕ → ℕ → Prop), (b 0 0 = true) ∧ (b 0 2 = true) ∧ (b 2 0 = true) ∧ (b 2 2 = true) ∧
  (b 0 0 = b 0 2) ∧ (b 2 0 ≠ b 2 2) → ¬(∃ (b' : ℕ → ℕ → Prop), 
  (b' 0 0 ≠ b 0 0) ∧ (b' 0 2 ≠ b 0 2) ∧ (b' 2 0 ≠ b 2 0) ∧ (b' 2 2 ≠ b 2 2) ∧ 
  (b' 0 0 ≠ b' 0 2) ∧ (b' 2 0 ≠ b' 2 2)) :=
by { sorry }

end NUMINAMATH_GPT_knights_rearrangement_impossible_l960_96054


namespace NUMINAMATH_GPT_sticks_needed_for_4x4_square_largest_square_with_100_sticks_l960_96053

-- Problem a)
def sticks_needed_for_square (n: ℕ) : ℕ := 2 * n * (n + 1)

theorem sticks_needed_for_4x4_square : sticks_needed_for_square 4 = 40 :=
by
  sorry

-- Problem b)
def max_square_side_length (total_sticks : ℕ) : ℕ × ℕ :=
  let n := Nat.sqrt (total_sticks / 2)
  if 2*n*(n+1) <= total_sticks then (n, total_sticks - 2*n*(n+1)) else (n-1, total_sticks - 2*(n-1)*n)

theorem largest_square_with_100_sticks : max_square_side_length 100 = (6, 16) :=
by
  sorry

end NUMINAMATH_GPT_sticks_needed_for_4x4_square_largest_square_with_100_sticks_l960_96053


namespace NUMINAMATH_GPT_inequality_not_true_l960_96078

theorem inequality_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬ (a > 0) :=
sorry

end NUMINAMATH_GPT_inequality_not_true_l960_96078


namespace NUMINAMATH_GPT_factorial_div_eq_l960_96027

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end NUMINAMATH_GPT_factorial_div_eq_l960_96027


namespace NUMINAMATH_GPT_find_ab_l960_96020

theorem find_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_area_9 : (1/2) * (12 / a) * (12 / b) = 9) : 
  a * b = 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_ab_l960_96020


namespace NUMINAMATH_GPT_cube_volume_correct_l960_96009

-- Define the height and base dimensions of the pyramid
def pyramid_height := 15
def pyramid_base_length := 12
def pyramid_base_width := 8

-- Define the side length of the cube-shaped box
def cube_side_length := max pyramid_height pyramid_base_length

-- Define the volume of the cube-shaped box
def cube_volume := cube_side_length ^ 3

-- Theorem statement: the volume of the smallest cube-shaped box that can fit the pyramid is 3375 cubic inches
theorem cube_volume_correct : cube_volume = 3375 := by
  sorry

end NUMINAMATH_GPT_cube_volume_correct_l960_96009


namespace NUMINAMATH_GPT_sam_gave_plums_l960_96059

variable (initial_plums : ℝ) (total_plums : ℝ) (plums_given : ℝ)

theorem sam_gave_plums (h1 : initial_plums = 7.0) (h2 : total_plums = 10.0) (h3 : total_plums = initial_plums + plums_given) :
  plums_given = 3 := 
by
  sorry

end NUMINAMATH_GPT_sam_gave_plums_l960_96059


namespace NUMINAMATH_GPT_expected_winnings_is_minus_half_l960_96026

-- Define the given condition in Lean
noncomputable def prob_win_side_1 : ℚ := 1 / 4
noncomputable def prob_win_side_2 : ℚ := 1 / 4
noncomputable def prob_lose_side_3 : ℚ := 1 / 3
noncomputable def prob_no_change_side_4 : ℚ := 1 / 6

noncomputable def win_amount_side_1 : ℚ := 2
noncomputable def win_amount_side_2 : ℚ := 4
noncomputable def lose_amount_side_3 : ℚ := -6
noncomputable def no_change_amount_side_4 : ℚ := 0

-- Define the expected value function
noncomputable def expected_winnings : ℚ :=
  (prob_win_side_1 * win_amount_side_1) +
  (prob_win_side_2 * win_amount_side_2) +
  (prob_lose_side_3 * lose_amount_side_3) +
  (prob_no_change_side_4 * no_change_amount_side_4)

-- Statement to prove
theorem expected_winnings_is_minus_half : expected_winnings = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_expected_winnings_is_minus_half_l960_96026


namespace NUMINAMATH_GPT_gcd_of_2535_5929_11629_l960_96002

theorem gcd_of_2535_5929_11629 : Nat.gcd (Nat.gcd 2535 5929) 11629 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_of_2535_5929_11629_l960_96002


namespace NUMINAMATH_GPT_option_b_is_incorrect_l960_96001

theorem option_b_is_incorrect : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_GPT_option_b_is_incorrect_l960_96001


namespace NUMINAMATH_GPT_product_of_y_coordinates_on_line_l960_96016

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem product_of_y_coordinates_on_line (y1 y2 : ℝ) (h1 : distance (4, -1) (-2, y1) = 8) (h2 : distance (4, -1) (-2, y2) = 8) :
  y1 * y2 = -27 :=
sorry

end NUMINAMATH_GPT_product_of_y_coordinates_on_line_l960_96016


namespace NUMINAMATH_GPT_wheat_bread_served_l960_96040

noncomputable def total_bread_served : ℝ := 0.6
noncomputable def white_bread_served : ℝ := 0.4

theorem wheat_bread_served : total_bread_served - white_bread_served = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_wheat_bread_served_l960_96040


namespace NUMINAMATH_GPT_sides_equal_max_diagonal_at_most_two_l960_96063

variable {n : ℕ}
variable (P : Polygon n)
variable (is_convex : P.IsConvex)
variable (max_diagonal : ℝ)
variable (sides_equal_max_diagonal : list ℝ)
variable (length_sides_equal_max_diagonal : sides_equal_max_diagonal.length)

-- Here we assume the basic conditions given in the problem:
-- 1. The polygon P is convex.
-- 2. The number of sides equal to the longest diagonal are stored in sides_equal_max_diagonal.

theorem sides_equal_max_diagonal_at_most_two :
  is_convex → length_sides_equal_max_diagonal ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_sides_equal_max_diagonal_at_most_two_l960_96063


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l960_96044

theorem geometric_sequence_third_term (q : ℝ) (b1 : ℝ) (h1 : abs q < 1)
    (h2 : b1 / (1 - q) = 8 / 5) (h3 : b1 * q = -1 / 2) :
    b1 * q^2 / 2 = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l960_96044


namespace NUMINAMATH_GPT_problem_statement_l960_96079

theorem problem_statement (x y : ℕ) (h1 : x = 3) (h2 :y = 5) :
  (x^5 + 2*y^2 - 15) / 7 = 39 + 5 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l960_96079


namespace NUMINAMATH_GPT_sum_series_l960_96099

theorem sum_series : ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2)) = 1 := 
sorry

end NUMINAMATH_GPT_sum_series_l960_96099


namespace NUMINAMATH_GPT_integer_solutions_l960_96000

theorem integer_solutions (a b c : ℤ) (h₁ : 1 < a) 
    (h₂ : a < b) (h₃ : b < c) 
    (h₄ : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) 
    ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by sorry

end NUMINAMATH_GPT_integer_solutions_l960_96000


namespace NUMINAMATH_GPT_left_handed_and_like_scifi_count_l960_96058

-- Definitions based on the problem conditions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def like_scifi_members : ℕ := 18
def right_handed_not_like_scifi : ℕ := 4

-- Main proof statement
theorem left_handed_and_like_scifi_count :
  ∃ x : ℕ, (left_handed_members - x) + (like_scifi_members - x) + x + right_handed_not_like_scifi = total_members ∧ x = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_left_handed_and_like_scifi_count_l960_96058


namespace NUMINAMATH_GPT_gcd_360_150_l960_96074

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end NUMINAMATH_GPT_gcd_360_150_l960_96074


namespace NUMINAMATH_GPT_mean_of_eight_numbers_l960_96083

theorem mean_of_eight_numbers (sum_of_numbers : ℚ) (h : sum_of_numbers = 3/4) : 
  sum_of_numbers / 8 = 3/32 := by
  sorry

end NUMINAMATH_GPT_mean_of_eight_numbers_l960_96083


namespace NUMINAMATH_GPT_sum_of_reciprocals_l960_96019

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 16) (h₂ : x * y = 55) :
  (1 / x + 1 / y) = 16 / 55 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l960_96019


namespace NUMINAMATH_GPT_arriving_late_l960_96060

-- Definitions from conditions
def usual_time : ℕ := 24
def slower_factor : ℚ := 3 / 4

-- Derived from conditions
def slower_time : ℚ := usual_time * (4 / 3)

-- To be proven
theorem arriving_late : slower_time - usual_time = 8 := by
  sorry

end NUMINAMATH_GPT_arriving_late_l960_96060


namespace NUMINAMATH_GPT_monotonicity_and_extrema_l960_96004

noncomputable def f (x : ℝ) := (2 * x) / (x + 1)

theorem monotonicity_and_extrema :
  (∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 5 → f x1 < f x2) ∧
  (f 3 = 5 / 4) ∧
  (f 5 = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_monotonicity_and_extrema_l960_96004


namespace NUMINAMATH_GPT_find_income_l960_96035

variable (x : ℝ)

def income : ℝ := 5 * x
def expenditure : ℝ := 4 * x
def savings : ℝ := income x - expenditure x

theorem find_income (h : savings x = 4000) : income x = 20000 :=
by
  rw [savings, income, expenditure] at h
  sorry

end NUMINAMATH_GPT_find_income_l960_96035


namespace NUMINAMATH_GPT_relationship_roots_geometric_progression_l960_96051

theorem relationship_roots_geometric_progression 
  (x y z p q r : ℝ)
  (h1 : x^2 ≠ y^2 ∧ y^2 ≠ z^2 ∧ x^2 ≠ z^2) -- Distinct non-zero numbers
  (h2 : y^2 = x^2 * r)
  (h3 : z^2 = y^2 * r)
  (h4 : x + y + z = p)
  (h5 : x * y + y * z + z * x = q)
  (h6 : x * y * z = r) : r^2 = 1 := sorry

end NUMINAMATH_GPT_relationship_roots_geometric_progression_l960_96051


namespace NUMINAMATH_GPT_composite_for_large_n_l960_96071

theorem composite_for_large_n (m : ℕ) (hm : m > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → Nat.Prime (2^m * 2^(2^n) + 1) = false :=
sorry

end NUMINAMATH_GPT_composite_for_large_n_l960_96071


namespace NUMINAMATH_GPT_hyperbola_sufficient_but_not_necessary_asymptote_l960_96046

-- Define the equation of the hyperbola and the related asymptotes
def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptote_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

-- Stating the theorem that expresses the sufficiency but not necessity
theorem hyperbola_sufficient_but_not_necessary_asymptote (a b : ℝ) :
  (∃ x y, hyperbola_eq a b x y) → (∀ x y, asymptote_eq a b x y) ∧ ¬ (∀ x y, (asymptote_eq a b x y) → (hyperbola_eq a b x y)) := 
sorry

end NUMINAMATH_GPT_hyperbola_sufficient_but_not_necessary_asymptote_l960_96046


namespace NUMINAMATH_GPT_factorial_trailing_zeros_500_l960_96090

theorem factorial_trailing_zeros_500 :
  let count_factors_of_five (n : ℕ) : ℕ := n / 5 + n / 25 + n / 125
  count_factors_of_five 500 = 124 :=
by
  sorry  -- The proof is not required as per the instructions.

end NUMINAMATH_GPT_factorial_trailing_zeros_500_l960_96090
