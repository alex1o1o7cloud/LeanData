import Mathlib

namespace NUMINAMATH_GPT_merchant_spent_for_belle_l220_22022

def dress_cost (S : ℤ) (H : ℤ) : ℤ := 6 * S + 3 * H
def hat_cost (S : ℤ) (H : ℤ) : ℤ := 3 * S + 5 * H
def belle_cost (S : ℤ) (H : ℤ) : ℤ := S + 2 * H

theorem merchant_spent_for_belle :
  ∃ (S H : ℤ), dress_cost S H = 105 ∧ hat_cost S H = 70 ∧ belle_cost S H = 25 :=
by
  sorry

end NUMINAMATH_GPT_merchant_spent_for_belle_l220_22022


namespace NUMINAMATH_GPT_mr_callen_total_loss_l220_22007

noncomputable def total_loss : ℤ :=
  let bought_paintings_price := 15 * 60
  let bought_wooden_toys_price := 12 * 25
  let bought_handmade_hats_price := 20 * 15
  let total_bought_price := bought_paintings_price + bought_wooden_toys_price + bought_handmade_hats_price
  let sold_paintings_price := 15 * (60 - (60 * 18 / 100))
  let sold_wooden_toys_price := 12 * (25 - (25 * 25 / 100))
  let sold_handmade_hats_price := 20 * (15 - (15 * 10 / 100))
  let total_sold_price := sold_paintings_price + sold_wooden_toys_price + sold_handmade_hats_price
  total_bought_price - total_sold_price

theorem mr_callen_total_loss : total_loss = 267 := by
  sorry

end NUMINAMATH_GPT_mr_callen_total_loss_l220_22007


namespace NUMINAMATH_GPT_inequality_x_alpha_y_beta_l220_22085

theorem inequality_x_alpha_y_beta (x y α β : ℝ) (hx : 0 < x) (hy : 0 < y) 
(hα : 0 < α) (hβ : 0 < β) (hαβ : α + β = 1) : x^α * y^β ≤ α * x + β * y := 
sorry

end NUMINAMATH_GPT_inequality_x_alpha_y_beta_l220_22085


namespace NUMINAMATH_GPT_exists_sequences_satisfying_conditions_l220_22093

noncomputable def satisfies_conditions (n : ℕ) (hn : Odd n) 
  (a : Fin n → ℕ) (b : Fin n → ℕ) : Prop :=
  ∀ (k : Fin n), 0 < k.val → k.val < n →
    ∀ (i : Fin n),
      let in3n := 3 * n;
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n ≠
      (a i + b i) % in3n ∧
      (a i + b i) % in3n ≠
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ∧
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ≠
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n

theorem exists_sequences_satisfying_conditions :
  ∀ n : ℕ, Odd n → ∃ (a : Fin n → ℕ) (b : Fin n → ℕ),
    satisfies_conditions n sorry a b :=
sorry

end NUMINAMATH_GPT_exists_sequences_satisfying_conditions_l220_22093


namespace NUMINAMATH_GPT_interest_cannot_be_determined_without_investment_amount_l220_22079

theorem interest_cannot_be_determined_without_investment_amount :
  ∀ (interest_rate : ℚ) (price : ℚ) (invested_amount : Option ℚ),
  interest_rate = 0.16 → price = 128 → invested_amount = none → False :=
by
  sorry

end NUMINAMATH_GPT_interest_cannot_be_determined_without_investment_amount_l220_22079


namespace NUMINAMATH_GPT_problem_omega_pow_l220_22028

noncomputable def omega : ℂ := Complex.I -- Define a non-real root for x² = 1; an example choice could be i, the imaginary unit.

theorem problem_omega_pow :
  omega^2 = 1 → 
  (1 - omega + omega^2)^6 + (1 + omega - omega^2)^6 = 730 := 
by
  intro h1
  -- proof steps omitted
  sorry

end NUMINAMATH_GPT_problem_omega_pow_l220_22028


namespace NUMINAMATH_GPT_determine_common_difference_l220_22008

variables {a : ℕ → ℤ} {d : ℤ}

-- Definition of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 + n * d

-- The given condition in the problem
def given_condition (a : ℕ → ℤ) (d : ℤ) : Prop :=
  3 * a 6 = a 3 + a 4 + a 5 + 6

-- The theorem to prove
theorem determine_common_difference
  (h_seq : arithmetic_seq a d)
  (h_cond : given_condition a d) :
  d = 1 :=
sorry

end NUMINAMATH_GPT_determine_common_difference_l220_22008


namespace NUMINAMATH_GPT_correct_option_is_C_l220_22037

theorem correct_option_is_C 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (D : Prop)
  (hA : ¬ A)
  (hB : ¬ B)
  (hD : ¬ D)
  (hC : C) :
  C := by
  exact hC

end NUMINAMATH_GPT_correct_option_is_C_l220_22037


namespace NUMINAMATH_GPT_compare_logs_l220_22044

open Real

noncomputable def a := log 6 / log 3
noncomputable def b := 1 / log 5
noncomputable def c := log 14 / log 7

theorem compare_logs : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_compare_logs_l220_22044


namespace NUMINAMATH_GPT_tan_subtraction_example_l220_22032

noncomputable def tan_subtraction_identity (alpha beta : ℝ) : ℝ :=
  (Real.tan alpha - Real.tan beta) / (1 + Real.tan alpha * Real.tan beta)

theorem tan_subtraction_example (theta : ℝ) (h : Real.tan theta = 1 / 2) :
  Real.tan (π / 4 - theta) = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_subtraction_example_l220_22032


namespace NUMINAMATH_GPT_gcd_of_35_and_number_between_70_and_90_is_7_l220_22010

def number_between_70_and_90 (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 90

def gcd_is_7 (a b : ℕ) : Prop :=
  Nat.gcd a b = 7

theorem gcd_of_35_and_number_between_70_and_90_is_7 : 
  ∃ (n : ℕ), number_between_70_and_90 n ∧ gcd_is_7 35 n ∧ (n = 77 ∨ n = 84) :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_35_and_number_between_70_and_90_is_7_l220_22010


namespace NUMINAMATH_GPT_find_percentage_l220_22068

theorem find_percentage (x : ℝ) (h1 : x = 780) (h2 : ∀ P : ℝ, P / 100 * x = 225 - 30) : P = 25 :=
by
  -- Definitions and conditions here
  -- Recall: x = 780 and P / 100 * x = 195
  sorry

end NUMINAMATH_GPT_find_percentage_l220_22068


namespace NUMINAMATH_GPT_calculate_adult_chaperones_l220_22057

theorem calculate_adult_chaperones (students : ℕ) (student_fee : ℕ) (adult_fee : ℕ) (total_fee : ℕ) 
  (h_students : students = 35) 
  (h_student_fee : student_fee = 5) 
  (h_adult_fee : adult_fee = 6) 
  (h_total_fee : total_fee = 199) : 
  ∃ (A : ℕ), 35 * student_fee + A * adult_fee = 199 ∧ A = 4 := 
by
  sorry

end NUMINAMATH_GPT_calculate_adult_chaperones_l220_22057


namespace NUMINAMATH_GPT_garden_watering_system_pumps_l220_22059

-- Define conditions
def rate := 500 -- gallons per hour
def time := 30 / 60 -- hours, i.e., converting 30 minutes to hours

-- Theorem statement
theorem garden_watering_system_pumps :
  rate * time = 250 := by
  sorry

end NUMINAMATH_GPT_garden_watering_system_pumps_l220_22059


namespace NUMINAMATH_GPT_line_y2_not_pass_second_quadrant_l220_22000

theorem line_y2_not_pass_second_quadrant {a b : ℝ} (h1 : a < 0) (h2 : b > 0) :
  ¬∃ x : ℝ, x < 0 ∧ bx + a > 0 :=
by
  sorry

end NUMINAMATH_GPT_line_y2_not_pass_second_quadrant_l220_22000


namespace NUMINAMATH_GPT_calculate_total_cost_l220_22099

theorem calculate_total_cost : 
  let piano_cost := 500
  let lesson_cost_per_lesson := 40
  let number_of_lessons := 20
  let discount_rate := 0.25
  let missed_lessons := 3
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_lesson_cost := number_of_lessons * lesson_cost_per_lesson
  let discount := total_lesson_cost * discount_rate
  let discounted_lesson_cost := total_lesson_cost - discount
  let cost_of_missed_lessons := missed_lessons * lesson_cost_per_lesson
  let effective_lesson_cost := discounted_lesson_cost + cost_of_missed_lessons
  let total_cost := piano_cost + effective_lesson_cost + sheet_music_cost + maintenance_fees
  total_cost = 1395 :=
by
  sorry

end NUMINAMATH_GPT_calculate_total_cost_l220_22099


namespace NUMINAMATH_GPT_exists_real_polynomial_l220_22094

noncomputable def has_negative_coeff (p : Polynomial ℝ) : Prop :=
  ∃ i, (p.coeff i) < 0

noncomputable def all_positive_coeff (n : ℕ) (p : Polynomial ℝ) : Prop :=
  ∀ i, (Polynomial.derivative^[n] p).coeff i > 0

theorem exists_real_polynomial :
  ∃ p : Polynomial ℝ, has_negative_coeff p ∧ (∀ n > 1, all_positive_coeff n p) :=
sorry

end NUMINAMATH_GPT_exists_real_polynomial_l220_22094


namespace NUMINAMATH_GPT_total_snow_volume_l220_22029

theorem total_snow_volume (length width initial_depth additional_depth: ℝ) 
  (h_length : length = 30) 
  (h_width : width = 3) 
  (h_initial_depth : initial_depth = 3 / 4) 
  (h_additional_depth : additional_depth = 1 / 4) : 
  (length * width * initial_depth) + (length * width * additional_depth) = 90 := 
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_total_snow_volume_l220_22029


namespace NUMINAMATH_GPT_coordinates_of_B_l220_22074

theorem coordinates_of_B (m : ℝ) (h : m + 2 = 0) : 
  (m + 5, m - 1) = (3, -3) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l220_22074


namespace NUMINAMATH_GPT_inverse_47_mod_48_l220_22001

theorem inverse_47_mod_48 : ∃ x, x < 48 ∧ x > 0 ∧ 47 * x % 48 = 1 :=
sorry

end NUMINAMATH_GPT_inverse_47_mod_48_l220_22001


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l220_22035

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : a 5 + a 4 = 18) (hS_def : ∀ n, S n = n * (a 1 + a n) / 2) : S 8 = 72 := 
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l220_22035


namespace NUMINAMATH_GPT_unique_solution_to_equation_l220_22087

theorem unique_solution_to_equation (a : ℝ) (h : ∀ x : ℝ, a * x^2 + Real.sin x ^ 2 = a^2 - a) : a = 1 :=
sorry

end NUMINAMATH_GPT_unique_solution_to_equation_l220_22087


namespace NUMINAMATH_GPT_initial_blocks_l220_22090

-- Definitions of the given conditions
def blocks_eaten : ℕ := 29
def blocks_remaining : ℕ := 26

-- The statement we need to prove
theorem initial_blocks : blocks_eaten + blocks_remaining = 55 :=
by
  -- Proof is not required as per instructions
  sorry

end NUMINAMATH_GPT_initial_blocks_l220_22090


namespace NUMINAMATH_GPT_factor_tree_value_l220_22077

theorem factor_tree_value :
  let F := 7 * (2 * 2)
  let H := 11 * 2
  let G := 11 * H
  let X := F * G
  X = 6776 :=
by
  sorry

end NUMINAMATH_GPT_factor_tree_value_l220_22077


namespace NUMINAMATH_GPT_recurring_decimal_to_rational_l220_22027

theorem recurring_decimal_to_rational : 
  (0.125125125 : ℝ) = 125 / 999 :=
sorry

end NUMINAMATH_GPT_recurring_decimal_to_rational_l220_22027


namespace NUMINAMATH_GPT_neg_sqrt_17_bounds_l220_22021

theorem neg_sqrt_17_bounds :
  (16 < 17) ∧ (17 < 25) ∧ (16 = 4^2) ∧ (25 = 5^2) ∧ (4 < Real.sqrt 17) ∧ (Real.sqrt 17 < 5) →
  (-5 < -Real.sqrt 17) ∧ (-Real.sqrt 17 < -4) :=
by
  sorry

end NUMINAMATH_GPT_neg_sqrt_17_bounds_l220_22021


namespace NUMINAMATH_GPT_evaluate_expression_l220_22046

noncomputable def g (A B C D x : ℝ) : ℝ := A * x^3 + B * x^2 - C * x + D

theorem evaluate_expression (A B C D : ℝ) (h1 : g A B C D 2 = 5) (h2 : g A B C D (-1) = -8) (h3 : g A B C D 0 = 2) :
  -12 * A + 6 * B - 3 * C + D = 27.5 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l220_22046


namespace NUMINAMATH_GPT_sector_area_l220_22045

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) (area : ℝ) 
  (h1 : arc_length = 6) 
  (h2 : central_angle = 2) 
  (h3 : radius = arc_length / central_angle): 
  area = (1 / 2) * arc_length * radius := 
  sorry

end NUMINAMATH_GPT_sector_area_l220_22045


namespace NUMINAMATH_GPT_subtraction_of_fractions_l220_22053

theorem subtraction_of_fractions :
  let S_1 := 3 + 5 + 7
  let S_2 := 2 + 4 + 6
  let S_3 := 2 + 4 + 6
  let S_4 := 3 + 5 + 7
  (S_1 / S_2 - S_3 / S_4) = 9 / 20 :=
by
  let S_1 := 3 + 5 + 7
  let S_2 := 2 + 4 + 6
  let S_3 := 2 + 4 + 6
  let S_4 := 3 + 5 + 7
  sorry

end NUMINAMATH_GPT_subtraction_of_fractions_l220_22053


namespace NUMINAMATH_GPT_triangle_II_area_l220_22076

noncomputable def triangle_area (base : ℝ) (height : ℝ) : ℝ :=
  1 / 2 * base * height

theorem triangle_II_area (a b : ℝ) :
  let I_area := triangle_area (a + b) (a + b)
  let II_area := 2 * I_area
  II_area = (a + b) ^ 2 :=
by
  let I_area := triangle_area (a + b) (a + b)
  let II_area := 2 * I_area
  sorry

end NUMINAMATH_GPT_triangle_II_area_l220_22076


namespace NUMINAMATH_GPT_Julia_played_with_kids_on_Monday_l220_22062

theorem Julia_played_with_kids_on_Monday (kids_tuesday : ℕ) (more_kids_monday : ℕ) :
  kids_tuesday = 14 → more_kids_monday = 8 → (kids_tuesday + more_kids_monday = 22) :=
by
  sorry

end NUMINAMATH_GPT_Julia_played_with_kids_on_Monday_l220_22062


namespace NUMINAMATH_GPT_point_not_in_fourth_quadrant_l220_22084

theorem point_not_in_fourth_quadrant (m : ℝ) : ¬(m-2 > 0 ∧ m+1 < 0) := 
by
  -- Since (m+1) - (m-2) = 3, which is positive,
  -- m+1 > m-2, thus the statement ¬(m-2 > 0 ∧ m+1 < 0) holds.
  sorry

end NUMINAMATH_GPT_point_not_in_fourth_quadrant_l220_22084


namespace NUMINAMATH_GPT_Ali_possible_scores_l220_22036

-- Defining the conditions
def categories := 5
def questions_per_category := 3
def correct_answers_points := 12
def total_questions := categories * questions_per_category
def incorrect_answers := total_questions - correct_answers_points

-- Defining the bonuses based on cases

-- All 3 incorrect answers in 1 category
def case_1_bonus := 4
def case_1_total := correct_answers_points + case_1_bonus

-- 3 incorrect answers split into 2 categories
def case_2_bonus := 3
def case_2_total := correct_answers_points + case_2_bonus

-- 3 incorrect answers split into 3 categories
def case_3_bonus := 2
def case_3_total := correct_answers_points + case_3_bonus

theorem Ali_possible_scores : 
  case_1_total = 16 ∧ case_2_total = 15 ∧ case_3_total = 14 :=
by
  -- Skipping the proof here
  sorry

end NUMINAMATH_GPT_Ali_possible_scores_l220_22036


namespace NUMINAMATH_GPT_quadratic_rewrite_l220_22078

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 6
noncomputable def c : ℕ := 284
noncomputable def quadratic_coeffs_sum : ℕ := a + b + c

theorem quadratic_rewrite :
  (∃ a b c : ℕ, 6 * (x : ℕ) ^ 2 + 72 * x + 500 = a * (x + b) ^ 2 + c) →
  quadratic_coeffs_sum = 296 := by sorry

end NUMINAMATH_GPT_quadratic_rewrite_l220_22078


namespace NUMINAMATH_GPT_sides_of_original_polygon_l220_22003

-- Define the sum of interior angles formula for a polygon with n sides
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the total sum of angles for the resulting polygon
def sum_of_new_polygon_angles : ℝ := 1980

-- The lean theorem statement to prove
theorem sides_of_original_polygon (n : ℕ) :
    sum_interior_angles n = sum_of_new_polygon_angles →
    n = 13 →
    12 ≤ n+1 ∧ n+1 ≤ 14 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_sides_of_original_polygon_l220_22003


namespace NUMINAMATH_GPT_second_sweet_red_probability_l220_22023

theorem second_sweet_red_probability (x y : ℕ) : 
  (y / (x + y : ℝ)) = y / (x + y + 10) * x / (x + y) + (y + 10) / (x + y + 10) * y / (x + y) :=
by
  sorry

end NUMINAMATH_GPT_second_sweet_red_probability_l220_22023


namespace NUMINAMATH_GPT_sum_first_49_odd_numbers_l220_22098

theorem sum_first_49_odd_numbers : (49^2 = 2401) :=
by
  sorry

end NUMINAMATH_GPT_sum_first_49_odd_numbers_l220_22098


namespace NUMINAMATH_GPT_find_fraction_l220_22004

theorem find_fraction (a b : ℝ) (h₁ : a ≠ b) (h₂ : a / b + (a + 6 * b) / (b + 6 * a) = 2) :
  a / b = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_fraction_l220_22004


namespace NUMINAMATH_GPT_trig_identity_l220_22069

variable (α : Real)
variable (h : Real.tan α = 2)

theorem trig_identity :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_trig_identity_l220_22069


namespace NUMINAMATH_GPT_period_of_sin_sub_cos_l220_22083

open Real

theorem period_of_sin_sub_cos :
  ∃ T > 0, ∀ x, sin x - cos x = sin (x + T) - cos (x + T) ∧ T = 2 * π := sorry

end NUMINAMATH_GPT_period_of_sin_sub_cos_l220_22083


namespace NUMINAMATH_GPT_area_of_rhombus_l220_22024

noncomputable def diagonal_length_1 : ℕ := 30
noncomputable def diagonal_length_2 : ℕ := 14

theorem area_of_rhombus (d1 d2 : ℕ) (h1 : d1 = diagonal_length_1) (h2 : d2 = diagonal_length_2) : 
  (d1 * d2) / 2 = 210 :=
by 
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l220_22024


namespace NUMINAMATH_GPT_janice_remaining_time_l220_22031

theorem janice_remaining_time
  (homework_time : ℕ := 30)
  (clean_room_time : ℕ := homework_time / 2)
  (walk_dog_time : ℕ := homework_time + 5)
  (take_out_trash_time : ℕ := homework_time / 6)
  (total_time_before_movie : ℕ := 120) :
  (total_time_before_movie - (homework_time + clean_room_time + walk_dog_time + take_out_trash_time)) = 35 :=
by
  sorry

end NUMINAMATH_GPT_janice_remaining_time_l220_22031


namespace NUMINAMATH_GPT_scientific_notation_per_capita_GDP_l220_22058

theorem scientific_notation_per_capita_GDP (GDP : ℝ) (h : GDP = 104000): 
  GDP = 1.04 * 10^5 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_per_capita_GDP_l220_22058


namespace NUMINAMATH_GPT_race_result_l220_22030

theorem race_result
    (distance_race : ℕ)
    (distance_diff : ℕ)
    (distance_second_start_diff : ℕ)
    (speed_xm speed_xl : ℕ)
    (h1 : distance_race = 100)
    (h2 : distance_diff = 20)
    (h3 : distance_second_start_diff = 20)
    (xm_wins_first_race : speed_xm * distance_race >= speed_xl * (distance_race - distance_diff)) :
    speed_xm * (distance_race + distance_second_start_diff) >= speed_xl * (distance_race + distance_diff) :=
by
  sorry

end NUMINAMATH_GPT_race_result_l220_22030


namespace NUMINAMATH_GPT_greatest_prime_factor_187_l220_22018

theorem greatest_prime_factor_187 : ∃ p : ℕ, Prime p ∧ p ∣ 187 ∧ ∀ q : ℕ, Prime q ∧ q ∣ 187 → p ≥ q := by
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_187_l220_22018


namespace NUMINAMATH_GPT_sqrt_floor_eq_l220_22063

theorem sqrt_floor_eq (n : ℕ) (hn : 0 < n) :
    ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 3)⌋ = ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ :=
by
  sorry

end NUMINAMATH_GPT_sqrt_floor_eq_l220_22063


namespace NUMINAMATH_GPT_total_test_subjects_l220_22067

-- Defining the conditions as mathematical entities
def number_of_colors : ℕ := 5
def unique_two_color_codes : ℕ := number_of_colors * number_of_colors
def excess_subjects : ℕ := 6

-- Theorem stating the question and correct answer
theorem total_test_subjects :
  unique_two_color_codes + excess_subjects = 31 :=
by
  -- Leaving the proof as sorry, since the task only requires statement creation
  sorry

end NUMINAMATH_GPT_total_test_subjects_l220_22067


namespace NUMINAMATH_GPT_shopkeeper_loss_percent_l220_22071

theorem shopkeeper_loss_percent 
  (C : ℝ) (P : ℝ) (L : ℝ) 
  (hC : C = 100) 
  (hP : P = 10) 
  (hL : L = 50) : 
  ((C - (((C * (1 - L / 100)) * (1 + P / 100))) / C) * 100) = 45 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_loss_percent_l220_22071


namespace NUMINAMATH_GPT_arithmetic_seq_a10_l220_22009

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_seq_a10 (h_arith : arithmetic_sequence a) (h2 : a 3 = 5) (h5 : a 6 = 11) : a 10 = 19 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a10_l220_22009


namespace NUMINAMATH_GPT_rain_probability_weekend_l220_22014

theorem rain_probability_weekend :
  let p_rain_F := 0.60
  let p_rain_S := 0.70
  let p_rain_U := 0.40
  let p_no_rain_F := 1 - p_rain_F
  let p_no_rain_S := 1 - p_rain_S
  let p_no_rain_U := 1 - p_rain_U
  let p_no_rain_all_days := p_no_rain_F * p_no_rain_S * p_no_rain_U
  let p_rain_at_least_one_day := 1 - p_no_rain_all_days
  (p_rain_at_least_one_day * 100 = 92.8) := sorry

end NUMINAMATH_GPT_rain_probability_weekend_l220_22014


namespace NUMINAMATH_GPT_maintenance_cost_relation_maximize_average_profit_l220_22075

def maintenance_cost (n : ℕ) : ℕ :=
  if n = 1 then 0 else 1400 * n - 1000

theorem maintenance_cost_relation :
  maintenance_cost 2 = 1800 ∧ maintenance_cost 5 = 6000 ∧
  (∀ n, n ≥ 2 → maintenance_cost n = 1400 * n - 1000) :=
by
  sorry

noncomputable def average_profit (n : ℕ) : ℝ :=
  if n < 2 then 0 else 60000 - (1 / n) * (137600 + 1400 * ((n - 1) * (n + 2) / 2) - 1000 * (n - 1))

theorem maximize_average_profit (n : ℕ) :
  n = 14 ↔ (average_profit n = 40700) :=
by
  sorry

end NUMINAMATH_GPT_maintenance_cost_relation_maximize_average_profit_l220_22075


namespace NUMINAMATH_GPT_find_d_l220_22042

variable {x1 x2 k d : ℝ}

axiom h₁ : x1 ≠ x2
axiom h₂ : 4 * x1^2 - k * x1 = d
axiom h₃ : 4 * x2^2 - k * x2 = d
axiom h₄ : x1 + x2 = 2

theorem find_d : d = -12 := by
  sorry

end NUMINAMATH_GPT_find_d_l220_22042


namespace NUMINAMATH_GPT_totalGames_l220_22072

-- Define Jerry's original number of video games
def originalGames : ℕ := 7

-- Define the number of video games Jerry received for his birthday
def birthdayGames : ℕ := 2

-- Statement: Prove that the total number of games Jerry has now is 9
theorem totalGames : originalGames + birthdayGames = 9 := by
  sorry

end NUMINAMATH_GPT_totalGames_l220_22072


namespace NUMINAMATH_GPT_smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l220_22061

theorem smallest_N_such_that_N_and_N_squared_end_in_same_three_digits :
  ∃ N : ℕ, (N > 0) ∧ (N % 1000 = (N^2 % 1000)) ∧ (1 ≤ N / 100 % 10) ∧ (N = 376) :=
by
  sorry

end NUMINAMATH_GPT_smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l220_22061


namespace NUMINAMATH_GPT_percentage_of_75_eq_percent_of_450_l220_22006

theorem percentage_of_75_eq_percent_of_450 (x : ℝ) (h : (x / 100) * 75 = 0.025 * 450) : x = 15 := 
sorry

end NUMINAMATH_GPT_percentage_of_75_eq_percent_of_450_l220_22006


namespace NUMINAMATH_GPT_solve_prime_equation_l220_22064

def is_prime (n : ℕ) : Prop := ∀ k, k < n ∧ k > 1 → n % k ≠ 0

theorem solve_prime_equation (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r)
  (h : 5 * p = q^3 - r^3) : p = 67 ∧ q = 7 ∧ r = 2 :=
sorry

end NUMINAMATH_GPT_solve_prime_equation_l220_22064


namespace NUMINAMATH_GPT_perpendicular_lines_slope_l220_22016

theorem perpendicular_lines_slope {m : ℝ} : 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0 → (m * (-1/2)) = -1) → 
  m = 2 :=
by 
  intros h_perpendicular h_slope
  sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_l220_22016


namespace NUMINAMATH_GPT_possible_values_2n_plus_m_l220_22048

theorem possible_values_2n_plus_m :
  ∀ (n m : ℤ), 3 * n - m < 5 → n + m > 26 → 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
by sorry

end NUMINAMATH_GPT_possible_values_2n_plus_m_l220_22048


namespace NUMINAMATH_GPT_incorrect_description_l220_22089

-- Conditions
def population_size : ℕ := 2000
def sample_size : ℕ := 150

-- Main Statement
theorem incorrect_description : ¬ (sample_size = 150) := 
by sorry

end NUMINAMATH_GPT_incorrect_description_l220_22089


namespace NUMINAMATH_GPT_evaluate_expression_l220_22012

theorem evaluate_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ( (1 / a^2 + 1 / b^2)⁻¹ = a^2 * b^2 / (a^2 + b^2) ) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l220_22012


namespace NUMINAMATH_GPT_polynomial_decomposition_l220_22082

noncomputable def s (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 1
noncomputable def t (x : ℝ) : ℝ := x + 18

def g (x : ℝ) : ℝ := 3 * x^4 + 9 * x^3 - 7 * x^2 + 2 * x + 6
def e (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem polynomial_decomposition : s 1 + t (-1) = 27 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_decomposition_l220_22082


namespace NUMINAMATH_GPT_train_length_is_correct_l220_22088

noncomputable def lengthOfTrain (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  speed_m_s * time_s

theorem train_length_is_correct : lengthOfTrain 60 15 = 250.05 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l220_22088


namespace NUMINAMATH_GPT_prime_iff_factorial_mod_l220_22049

theorem prime_iff_factorial_mod (p : ℕ) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end NUMINAMATH_GPT_prime_iff_factorial_mod_l220_22049


namespace NUMINAMATH_GPT_diagonals_bisect_in_rhombus_l220_22047

axiom Rhombus : Type
axiom Parallelogram : Type

axiom isParallelogram : Rhombus → Parallelogram
axiom diagonalsBisectEachOther : Parallelogram → Prop

theorem diagonals_bisect_in_rhombus (R : Rhombus) :
  ∀ (P : Parallelogram), isParallelogram R = P → diagonalsBisectEachOther P → diagonalsBisectEachOther (isParallelogram R) :=
by
  sorry

end NUMINAMATH_GPT_diagonals_bisect_in_rhombus_l220_22047


namespace NUMINAMATH_GPT_sum_of_reciprocals_eq_three_l220_22011

theorem sum_of_reciprocals_eq_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1/x + 1/y) = 3 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_eq_three_l220_22011


namespace NUMINAMATH_GPT_cost_of_tax_free_item_D_l220_22050

theorem cost_of_tax_free_item_D 
  (P_A P_B P_C : ℝ)
  (H1 : 0.945 * P_A + 1.064 * P_B + 1.18 * P_C = 225)
  (H2 : 0.045 * P_A + 0.12 * P_B + 0.18 * P_C = 30) :
  250 - (0.945 * P_A + 1.064 * P_B + 1.18 * P_C) = 25 := 
by
  -- The proof steps would go here.
  sorry

end NUMINAMATH_GPT_cost_of_tax_free_item_D_l220_22050


namespace NUMINAMATH_GPT_find_least_d_l220_22070

theorem find_least_d :
  ∃ d : ℕ, (d % 7 = 1) ∧ (d % 5 = 2) ∧ (d % 3 = 2) ∧ d = 92 :=
by 
  sorry

end NUMINAMATH_GPT_find_least_d_l220_22070


namespace NUMINAMATH_GPT_first_player_has_winning_strategy_l220_22039

-- Define the initial heap sizes and rules of the game.
def initial_heaps : List Nat := [38, 45, 61, 70]

-- Define a function that checks using the rules whether the first player has a winning strategy given the initial heap sizes.
def first_player_wins : Bool :=
  -- placeholder for the actual winning strategy check logic
  sorry

-- Theorem statement referring to the equivalency proof problem where player one is established to have the winning strategy.
theorem first_player_has_winning_strategy : first_player_wins = true :=
  sorry

end NUMINAMATH_GPT_first_player_has_winning_strategy_l220_22039


namespace NUMINAMATH_GPT_initial_unread_messages_correct_l220_22015

-- Definitions based on conditions
def messages_read_per_day := 20
def messages_new_per_day := 6
def duration_in_days := 7
def effective_reading_rate := messages_read_per_day - messages_new_per_day

-- The initial number of unread messages
def initial_unread_messages := duration_in_days * effective_reading_rate

-- The theorem we want to prove
theorem initial_unread_messages_correct :
  initial_unread_messages = 98 :=
sorry

end NUMINAMATH_GPT_initial_unread_messages_correct_l220_22015


namespace NUMINAMATH_GPT_pentagon_area_inequality_l220_22081

-- Definitions for the problem
structure Point :=
(x y : ℝ)

structure Triangle :=
(A B C : Point)

noncomputable def area (T : Triangle) : ℝ :=
  1 / 2 * abs ((T.B.x - T.A.x) * (T.C.y - T.A.y) - (T.C.x - T.A.x) * (T.B.y - T.A.y))

structure Pentagon :=
(A B C D E : Point)

noncomputable def pentagon_area (P : Pentagon) : ℝ :=
  area ⟨P.A, P.B, P.C⟩ + area ⟨P.A, P.C, P.D⟩ + area ⟨P.A, P.D, P.E⟩ -
  area ⟨P.E, P.B, P.C⟩

-- Given conditions
variables (A B C D E F : Point)
variables (P : Pentagon) 
-- P is a convex pentagon with points A, B, C, D, E in order 

-- Intersection point of AD and EC is F 
axiom intersect_diagonals (AD EC : Triangle) : AD.C = F ∧ EC.B = F

-- Theorem statement
theorem pentagon_area_inequality :
  let AED := Triangle.mk A E D
  let EDC := Triangle.mk E D C
  let EAB := Triangle.mk E A B
  let DCB := Triangle.mk D C B
  area AED + area EDC + area EAB + area DCB > pentagon_area P :=
  sorry

end NUMINAMATH_GPT_pentagon_area_inequality_l220_22081


namespace NUMINAMATH_GPT_largest_neg_integer_solution_l220_22065

theorem largest_neg_integer_solution 
  (x : ℤ) 
  (h : 34 * x + 6 ≡ 2 [ZMOD 20]) : 
  x = -6 := 
sorry

end NUMINAMATH_GPT_largest_neg_integer_solution_l220_22065


namespace NUMINAMATH_GPT_lattice_points_distance_5_l220_22080

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem lattice_points_distance_5 : 
  ∃ S : Finset (ℤ × ℤ × ℤ), 
    (∀ p ∈ S, is_lattice_point p.1 p.2.1 p.2.2) ∧
    S.card = 78 :=
by
  sorry

end NUMINAMATH_GPT_lattice_points_distance_5_l220_22080


namespace NUMINAMATH_GPT_tangent_circle_equation_l220_22060

theorem tangent_circle_equation :
  (∃ m : Real, ∃ n : Real,
    (∀ x y : Real, (x - m)^2 + (y - n)^2 = 36) ∧ 
    ((m - 0)^2 + (n - 3)^2 = 25) ∧
    n = 6 ∧ (m = 4 ∨ m = -4)) :=
sorry

end NUMINAMATH_GPT_tangent_circle_equation_l220_22060


namespace NUMINAMATH_GPT_smallest_n_satisfying_conditions_l220_22043

theorem smallest_n_satisfying_conditions : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ x : ℕ, 3 * n = x^4) ∧ (∃ y : ℕ, 2 * n = y^5) ∧ n = 432 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_satisfying_conditions_l220_22043


namespace NUMINAMATH_GPT_problem_concentric_circles_chord_probability_l220_22051

open ProbabilityTheory

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h : r1 < r2) : ℝ :=
1/6

theorem problem_concentric_circles_chord_probability :
  probability_chord_intersects_inner_circle 1.5 3 
  (by norm_num) = 1/6 :=
sorry

end NUMINAMATH_GPT_problem_concentric_circles_chord_probability_l220_22051


namespace NUMINAMATH_GPT_max_largest_integer_l220_22013

theorem max_largest_integer (A B C D E : ℕ) (h₀ : A ≤ B) (h₁ : B ≤ C) (h₂ : C ≤ D) (h₃ : D ≤ E) 
(h₄ : (A + B + C + D + E) = 225) (h₅ : E - A = 10) : E = 215 :=
sorry

end NUMINAMATH_GPT_max_largest_integer_l220_22013


namespace NUMINAMATH_GPT_paco_salty_cookies_left_l220_22091

-- Define the initial number of salty cookies Paco had
def initial_salty_cookies : ℕ := 26

-- Define the number of salty cookies Paco ate
def eaten_salty_cookies : ℕ := 9

-- The theorem statement that Paco had 17 salty cookies left
theorem paco_salty_cookies_left : initial_salty_cookies - eaten_salty_cookies = 17 := 
 by
  -- Here we skip the proof by adding sorry
  sorry

end NUMINAMATH_GPT_paco_salty_cookies_left_l220_22091


namespace NUMINAMATH_GPT_angle_terminal_side_equiv_l220_22025

-- Define the function to check angle equivalence
def angle_equiv (θ₁ θ₂ : ℝ) : Prop := ∃ k : ℤ, θ₁ = θ₂ + k * 360

-- Theorem statement
theorem angle_terminal_side_equiv : angle_equiv 330 (-30) :=
  sorry

end NUMINAMATH_GPT_angle_terminal_side_equiv_l220_22025


namespace NUMINAMATH_GPT_basketball_committee_l220_22033

theorem basketball_committee (total_players guards : ℕ) (choose_committee choose_guard : ℕ) :
  total_players = 12 → guards = 4 → choose_committee = 3 → choose_guard = 1 →
  (guards * ((total_players - guards).choose (choose_committee - choose_guard)) = 112) :=
by
  intros h_tp h_g h_cc h_cg
  rw [h_tp, h_g, h_cc, h_cg]
  simp
  norm_num
  sorry

end NUMINAMATH_GPT_basketball_committee_l220_22033


namespace NUMINAMATH_GPT_possible_rectangle_configurations_l220_22040

-- Define the conditions as variables
variables (m n : ℕ)
-- Define the number of segments
def segments (m n : ℕ) : ℕ := 2 * m * n + m + n

theorem possible_rectangle_configurations : 
  (segments m n = 1997) → (m = 2 ∧ n = 399) ∨ (m = 8 ∧ n = 117) ∨ (m = 23 ∧ n = 42) :=
by
  sorry

end NUMINAMATH_GPT_possible_rectangle_configurations_l220_22040


namespace NUMINAMATH_GPT_num_arrangements_with_ab_together_l220_22026

theorem num_arrangements_with_ab_together (products : Fin 5 → Type) :
  (∃ A B : Fin 5 → Type, A ≠ B) →
  ∃ (n : ℕ), n = 48 :=
by
  sorry

end NUMINAMATH_GPT_num_arrangements_with_ab_together_l220_22026


namespace NUMINAMATH_GPT_hyuksu_total_meat_l220_22092

/-- 
Given that Hyuksu ate 2.6 kilograms (kg) of meat yesterday and 5.98 kilograms (kg) of meat today,
prove that the total kilograms (kg) of meat he ate in two days is 8.58 kg.
-/
theorem hyuksu_total_meat (yesterday today : ℝ) (hy1 : yesterday = 2.6) (hy2 : today = 5.98) :
  yesterday + today = 8.58 := 
by
  rw [hy1, hy2]
  norm_num

end NUMINAMATH_GPT_hyuksu_total_meat_l220_22092


namespace NUMINAMATH_GPT_transfer_deck_l220_22052

-- Define the conditions
variables {k n : ℕ}

-- Assume conditions explicitly
axiom k_gt_1 : k > 1
axiom cards_deck : 2*n = 2*n -- Implicitly states that we have 2n cards

-- Define the problem statement
theorem transfer_deck (k_gt_1 : k > 1) (cards_deck : 2*n = 2*n) : n = k - 1 :=
sorry

end NUMINAMATH_GPT_transfer_deck_l220_22052


namespace NUMINAMATH_GPT_tan_add_pi_div_three_l220_22054

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end NUMINAMATH_GPT_tan_add_pi_div_three_l220_22054


namespace NUMINAMATH_GPT_sum_of_35_consecutive_squares_div_by_35_l220_22002

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_35_consecutive_squares_div_by_35 (n : ℕ) :
  (sum_of_squares (n + 35) - sum_of_squares n) % 35 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_35_consecutive_squares_div_by_35_l220_22002


namespace NUMINAMATH_GPT_find_linear_combination_l220_22096

variable (a b c : ℝ)

theorem find_linear_combination (h1 : a + 2 * b - 3 * c = 4)
                               (h2 : 5 * a - 6 * b + 7 * c = 8) :
  9 * a + 2 * b - 5 * c = 24 :=
sorry

end NUMINAMATH_GPT_find_linear_combination_l220_22096


namespace NUMINAMATH_GPT_baseball_cards_per_friend_l220_22020

theorem baseball_cards_per_friend (total_cards : ℕ) (total_friends : ℕ) (h1 : total_cards = 24) (h2 : total_friends = 4) : (total_cards / total_friends) = 6 := 
by
  sorry

end NUMINAMATH_GPT_baseball_cards_per_friend_l220_22020


namespace NUMINAMATH_GPT_exists_m_divisible_l220_22005

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x + 2

-- Define the 100th iterate of f
def f_iter (n : ℕ) : ℕ := 3^n

-- Define the condition that needs to be proven
theorem exists_m_divisible : ∃ m : ℕ, (3^100 * m + (3^100 - 1)) % 1988 = 0 :=
sorry

end NUMINAMATH_GPT_exists_m_divisible_l220_22005


namespace NUMINAMATH_GPT_paula_paint_cans_needed_l220_22055

-- Let's define the initial conditions and required computations in Lean.
def initial_rooms : ℕ := 48
def cans_lost : ℕ := 4
def remaining_rooms : ℕ := 36
def large_rooms_to_paint : ℕ := 8
def normal_rooms_to_paint : ℕ := 20
def paint_per_large_room : ℕ := 2 -- as each large room requires twice as much paint

-- Define a function to compute the number of cans required.
def cans_needed (initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room : ℕ) : ℕ :=
  let rooms_lost := initial_rooms - remaining_rooms
  let cans_per_room := rooms_lost / cans_lost
  let total_room_equivalents := large_rooms_to_paint * paint_per_large_room + normal_rooms_to_paint
  total_room_equivalents / cans_per_room

theorem paula_paint_cans_needed : cans_needed initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room = 12 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_paula_paint_cans_needed_l220_22055


namespace NUMINAMATH_GPT_curve_transformation_l220_22086

theorem curve_transformation (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →
  (x' = 4 * x) →
  (y' = 2 * y) →
  (x'^2 / 16 + y'^2 / 4 = 1) :=
by
  sorry

end NUMINAMATH_GPT_curve_transformation_l220_22086


namespace NUMINAMATH_GPT_exists_k_divides_poly_l220_22066

theorem exists_k_divides_poly (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : a 2 = 1) 
  (h₂ : ∀ k : ℕ, a (k + 2) = a (k + 1) + a k) :
  ∀ (m : ℕ), m > 0 → ∃ k : ℕ, m ∣ (a k ^ 4 - a k - 2) :=
by
  sorry

end NUMINAMATH_GPT_exists_k_divides_poly_l220_22066


namespace NUMINAMATH_GPT_arrange_BANANA_l220_22034

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end NUMINAMATH_GPT_arrange_BANANA_l220_22034


namespace NUMINAMATH_GPT_sum_of_min_value_and_input_l220_22019

def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem sum_of_min_value_and_input : 
  let a := -1
  let b := 3 * a - a ^ 3
  a + b = -3 := 
by
  let a := -1
  let b := 3 * a - a ^ 3
  sorry

end NUMINAMATH_GPT_sum_of_min_value_and_input_l220_22019


namespace NUMINAMATH_GPT_city_G_has_highest_percentage_increase_l220_22041

-- Define the population data as constants.
def population_1990_F : ℕ := 50
def population_2000_F : ℕ := 60
def population_1990_G : ℕ := 60
def population_2000_G : ℕ := 80
def population_1990_H : ℕ := 90
def population_2000_H : ℕ := 110
def population_1990_I : ℕ := 120
def population_2000_I : ℕ := 150
def population_1990_J : ℕ := 150
def population_2000_J : ℕ := 190

-- Define the function that calculates the percentage increase.
def percentage_increase (pop_1990 pop_2000 : ℕ) : ℚ :=
  (pop_2000 : ℚ) / (pop_1990 : ℚ)

-- Calculate the percentage increases for each city.
def percentage_increase_F := percentage_increase population_1990_F population_2000_F
def percentage_increase_G := percentage_increase population_1990_G population_2000_G
def percentage_increase_H := percentage_increase population_1990_H population_2000_H
def percentage_increase_I := percentage_increase population_1990_I population_2000_I
def percentage_increase_J := percentage_increase population_1990_J population_2000_J

-- Prove that City G has the greatest percentage increase.
theorem city_G_has_highest_percentage_increase :
  percentage_increase_G > percentage_increase_F ∧ 
  percentage_increase_G > percentage_increase_H ∧
  percentage_increase_G > percentage_increase_I ∧
  percentage_increase_G > percentage_increase_J :=
by sorry

end NUMINAMATH_GPT_city_G_has_highest_percentage_increase_l220_22041


namespace NUMINAMATH_GPT_probability_two_red_marbles_drawn_l220_22038

/-- A jar contains two red marbles, three green marbles, and ten white marbles and no other marbles.
Two marbles are randomly drawn from this jar without replacement. -/
theorem probability_two_red_marbles_drawn (total_marbles red_marbles green_marbles white_marbles : ℕ)
    (draw_without_replacement : Bool) :
    total_marbles = 15 ∧ red_marbles = 2 ∧ green_marbles = 3 ∧ white_marbles = 10 ∧ draw_without_replacement = true →
    (2 / 15) * (1 / 14) = 1 / 105 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_probability_two_red_marbles_drawn_l220_22038


namespace NUMINAMATH_GPT_sum_reciprocals_factors_12_l220_22073

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocals_factors_12_l220_22073


namespace NUMINAMATH_GPT_three_pow_2010_mod_eight_l220_22017

theorem three_pow_2010_mod_eight : (3^2010) % 8 = 1 :=
  sorry

end NUMINAMATH_GPT_three_pow_2010_mod_eight_l220_22017


namespace NUMINAMATH_GPT_albert_needs_more_money_l220_22097

-- Definitions derived from the problem conditions
def cost_paintbrush : ℝ := 1.50
def cost_paints : ℝ := 4.35
def cost_easel : ℝ := 12.65
def money_albert_has : ℝ := 6.50

-- Statement asserting the amount of money Albert needs
theorem albert_needs_more_money : (cost_paintbrush + cost_paints + cost_easel) - money_albert_has = 12 :=
by
  sorry

end NUMINAMATH_GPT_albert_needs_more_money_l220_22097


namespace NUMINAMATH_GPT_adult_ticket_cost_l220_22095

theorem adult_ticket_cost 
  (child_ticket_cost : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (adults_attended : ℕ)
  (children_tickets : ℕ)
  (adults_ticket_cost : ℕ)
  (h1 : child_ticket_cost = 6)
  (h2 : total_tickets = 225)
  (h3 : total_cost = 1875)
  (h4 : adults_attended = 175)
  (h5 : children_tickets = total_tickets - adults_attended)
  (h6 : total_cost = adults_attended * adults_ticket_cost + children_tickets * child_ticket_cost) :
  adults_ticket_cost = 9 :=
sorry

end NUMINAMATH_GPT_adult_ticket_cost_l220_22095


namespace NUMINAMATH_GPT_norma_initial_cards_l220_22056

theorem norma_initial_cards (x : ℝ) 
  (H1 : x + 70 = 158) : 
  x = 88 :=
by
  sorry

end NUMINAMATH_GPT_norma_initial_cards_l220_22056
