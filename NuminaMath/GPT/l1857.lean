import Mathlib

namespace NUMINAMATH_GPT_max_value_expression_l1857_185712

noncomputable def f : Real → Real := λ x => 3 * Real.sin x + 4 * Real.cos x

theorem max_value_expression (θ : Real) (h_max : ∀ x, f x ≤ 5) :
  (3 * Real.sin θ + 4 * Real.cos θ = 5) →
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 65 / 7 := by
  sorry

end NUMINAMATH_GPT_max_value_expression_l1857_185712


namespace NUMINAMATH_GPT_real_part_of_z_given_condition_l1857_185756

open Complex

noncomputable def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_of_z_given_condition :
  ∀ (z : ℂ), (i * (z + 1) = -3 + 2 * i) → real_part_of_z z = 1 :=
by
  intro z h
  sorry

end NUMINAMATH_GPT_real_part_of_z_given_condition_l1857_185756


namespace NUMINAMATH_GPT_green_light_probability_l1857_185748

-- Define the durations of the red, green, and yellow lights
def red_light_duration : ℕ := 30
def green_light_duration : ℕ := 25
def yellow_light_duration : ℕ := 5

-- Define the total cycle time
def total_cycle_time : ℕ := red_light_duration + green_light_duration + yellow_light_duration

-- Define the expected probability
def expected_probability : ℚ := 5 / 12

-- Prove the probability of seeing a green light equals the expected_probability
theorem green_light_probability :
  (green_light_duration : ℚ) / (total_cycle_time : ℚ) = expected_probability :=
by
  sorry

end NUMINAMATH_GPT_green_light_probability_l1857_185748


namespace NUMINAMATH_GPT_seq_contains_exactly_16_twos_l1857_185777

-- Define a helper function to count occurrences of a digit in a number
def count_digit (d : Nat) (n : Nat) : Nat :=
  (n.digits 10).count d

-- Define a function to sum occurrences of the digit '2' in a list of numbers
def total_twos_in_sequence (seq : List Nat) : Nat :=
  seq.foldl (λ acc n => acc + count_digit 2 n) 0

-- Define the sequence we are interested in
def seq : List Nat := [2215, 2216, 2217, 2218, 2219, 2220, 2221]

-- State the theorem we need to prove
theorem seq_contains_exactly_16_twos : total_twos_in_sequence seq = 16 := 
by
  -- We do not provide the proof here according to the given instructions
  sorry

end NUMINAMATH_GPT_seq_contains_exactly_16_twos_l1857_185777


namespace NUMINAMATH_GPT_value_of_f_f_2_l1857_185761

def f (x : ℤ) : ℤ := 4 * x^2 + 2 * x - 1

theorem value_of_f_f_2 : f (f 2) = 1481 := by
  sorry

end NUMINAMATH_GPT_value_of_f_f_2_l1857_185761


namespace NUMINAMATH_GPT_find_abs_ab_l1857_185759

def ellipse_foci_distance := 5
def hyperbola_foci_distance := 7

def ellipse_condition (a b : ℝ) := b^2 - a^2 = ellipse_foci_distance^2
def hyperbola_condition (a b : ℝ) := a^2 + b^2 = hyperbola_foci_distance^2

theorem find_abs_ab (a b : ℝ) (h_ellipse : ellipse_condition a b) (h_hyperbola : hyperbola_condition a b) :
  |a * b| = 2 * Real.sqrt 111 :=
by
  sorry

end NUMINAMATH_GPT_find_abs_ab_l1857_185759


namespace NUMINAMATH_GPT_log2_a_div_b_squared_l1857_185765

variable (a b : ℝ)
variable (ha_ne_1 : a ≠ 1) (hb_ne_1 : b ≠ 1)
variable (ha_pos : 0 < a) (hb_pos : 0 < b)
variable (h1 : 2 ^ (Real.log 32 / Real.log b) = a)
variable (h2 : a * b = 128)

theorem log2_a_div_b_squared :
  (Real.log ((a / b) : ℝ) / Real.log 2) ^ 2 = 29 + (49 / 4) :=
sorry

end NUMINAMATH_GPT_log2_a_div_b_squared_l1857_185765


namespace NUMINAMATH_GPT_fraction_product_eq_l1857_185710

theorem fraction_product_eq :
  (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_eq_l1857_185710


namespace NUMINAMATH_GPT_complement_setP_in_U_l1857_185728

def setU : Set ℝ := {x | -1 < x ∧ x < 3}
def setP : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem complement_setP_in_U : (setU \ setP) = {x | 2 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_setP_in_U_l1857_185728


namespace NUMINAMATH_GPT_roller_coaster_cost_l1857_185799

variable (ferris_wheel_rides : Nat) (log_ride_rides : Nat) (rc_rides : Nat)
variable (ferris_wheel_cost : Nat) (log_ride_cost : Nat)
variable (initial_tickets : Nat) (additional_tickets : Nat)
variable (total_needed_tickets : Nat)

theorem roller_coaster_cost :
  ferris_wheel_rides = 2 →
  log_ride_rides = 7 →
  rc_rides = 3 →
  ferris_wheel_cost = 2 →
  log_ride_cost = 1 →
  initial_tickets = 20 →
  additional_tickets = 6 →
  total_needed_tickets = initial_tickets + additional_tickets →
  let total_ride_costs := ferris_wheel_rides * ferris_wheel_cost + log_ride_rides * log_ride_cost
  let rc_cost := (total_needed_tickets - total_ride_costs) / rc_rides
  rc_cost = 5 := by
  sorry

end NUMINAMATH_GPT_roller_coaster_cost_l1857_185799


namespace NUMINAMATH_GPT_smallest_solution_l1857_185760

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_smallest_solution_l1857_185760


namespace NUMINAMATH_GPT_cos_double_angle_l1857_185770

theorem cos_double_angle (α : ℝ) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) : Real.cos α = 1 / 3 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l1857_185770


namespace NUMINAMATH_GPT_max_xy_min_function_l1857_185741

-- Problem 1: Prove that the maximum value of xy is 8 given the conditions
theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 8) : xy ≤ 8 :=
sorry

-- Problem 2: Prove that the minimum value of the function is 9 given the conditions
theorem min_function (x : ℝ) (hx : -1 < x) : (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

end NUMINAMATH_GPT_max_xy_min_function_l1857_185741


namespace NUMINAMATH_GPT_find_slope_and_intercept_l1857_185758

noncomputable def line_equation_to_slope_intercept_form 
  (x y : ℝ) : Prop :=
  (3 * (x - 2) - 4 * (y + 3) = 0) ↔ (y = (3 / 4) * x - 4.5)

theorem find_slope_and_intercept : 
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), (line_equation_to_slope_intercept_form x y) → m = 3/4 ∧ b = -4.5) :=
sorry

end NUMINAMATH_GPT_find_slope_and_intercept_l1857_185758


namespace NUMINAMATH_GPT_polynomial_solution_l1857_185774

noncomputable def P : ℝ → ℝ := sorry

theorem polynomial_solution (x : ℝ) :
  (∃ P : ℝ → ℝ, (∀ x, P x = (P 0) + (P 1) * x + (P 2) * x^2) ∧ 
  (P (-2) = 4)) →
  (P x = (4 * x^2 - 6 * x) / 7) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l1857_185774


namespace NUMINAMATH_GPT_RU_eq_825_l1857_185780

variables (P Q R S T U : Type)
variables (PQ QR RP QS SR : ℝ)
variables (RU : ℝ)
variables (hPQ : PQ = 13)
variables (hQR : QR = 30)
variables (hRP : RP = 26)
variables (hQS : QS = 10)
variables (hSR : SR = 20)

theorem RU_eq_825 :
  RU = 8.25 :=
sorry

end NUMINAMATH_GPT_RU_eq_825_l1857_185780


namespace NUMINAMATH_GPT_average_age_is_correct_l1857_185704

-- Define the conditions
def num_men : ℕ := 6
def num_women : ℕ := 9
def average_age_men : ℕ := 57
def average_age_women : ℕ := 52
def total_age_men : ℕ := num_men * average_age_men
def total_age_women : ℕ := num_women * average_age_women
def total_age : ℕ := total_age_men + total_age_women
def total_people : ℕ := num_men + num_women
def average_age_group : ℕ := total_age / total_people

-- The proof will require showing average_age_group is 54, left as sorry.
theorem average_age_is_correct : average_age_group = 54 := sorry

end NUMINAMATH_GPT_average_age_is_correct_l1857_185704


namespace NUMINAMATH_GPT_remainder_zero_when_divided_by_condition_l1857_185742

noncomputable def remainder_problem (x : ℂ) : ℂ :=
  (2 * x^5 - x^4 + x^2 - 1) * (x^3 - 1)

theorem remainder_zero_when_divided_by_condition (x : ℂ) (h : x^2 - x + 1 = 0) :
  remainder_problem x % (x^2 - x + 1) = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_zero_when_divided_by_condition_l1857_185742


namespace NUMINAMATH_GPT_loss_percentage_l1857_185729

/-
Books Problem:
Determine the loss percentage on the first book given:
1. The cost of the first book (C1) is Rs. 280.
2. The total cost of two books is Rs. 480.
3. The second book is sold at a gain of 19%.
4. Both books are sold at the same price.
-/

theorem loss_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 = 280)
  (h2 : C1 + C2 = 480) (h3 : SP2 = C2 * 1.19) (h4 : SP1 = SP2) : 
  (C1 - SP1) / C1 * 100 = 15 := 
by
  sorry

end NUMINAMATH_GPT_loss_percentage_l1857_185729


namespace NUMINAMATH_GPT_symmetry_with_respect_to_line_x_eq_1_l1857_185775

theorem symmetry_with_respect_to_line_x_eq_1 (f : ℝ → ℝ) :
  ∀ x, f (x - 1) = f (1 - x) ↔ x - 1 = 1 - x :=
by
  sorry

end NUMINAMATH_GPT_symmetry_with_respect_to_line_x_eq_1_l1857_185775


namespace NUMINAMATH_GPT_sum_of_squares_of_coeffs_l1857_185706

theorem sum_of_squares_of_coeffs (c1 c2 c3 c4 : ℝ) (h1 : c1 = 3) (h2 : c2 = 6) (h3 : c3 = 15) (h4 : c4 = 6) :
  c1^2 + c2^2 + c3^2 + c4^2 = 306 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_coeffs_l1857_185706


namespace NUMINAMATH_GPT_union_P_complement_Q_l1857_185773

open Set

def P : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }
def R : Set ℝ := { x | -2 < x ∧ x < 2 }
def PQ_union : Set ℝ := P ∪ R

theorem union_P_complement_Q : PQ_union = { x | -2 < x ∧ x ≤ 3 } :=
by sorry

end NUMINAMATH_GPT_union_P_complement_Q_l1857_185773


namespace NUMINAMATH_GPT_last_digit_of_2_pow_2010_l1857_185746

theorem last_digit_of_2_pow_2010 : (2 ^ 2010) % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_of_2_pow_2010_l1857_185746


namespace NUMINAMATH_GPT_given_system_solution_l1857_185701

noncomputable def solve_system : Prop :=
  ∃ x y z : ℝ, 
  x + y + z = 1 ∧ 
  x^2 + y^2 + z^2 = 1 ∧ 
  x^3 + y^3 + z^3 = 89 / 125 ∧ 
  (x = 2 / 5 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = 2 / 5 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = 2 / 5 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = 2 / 5)

theorem given_system_solution : solve_system :=
sorry

end NUMINAMATH_GPT_given_system_solution_l1857_185701


namespace NUMINAMATH_GPT_speed_ratio_l1857_185719

theorem speed_ratio (L v_a v_b : ℝ) (h1 : v_a = c * v_b) (h2 : (L / v_a) = (0.8 * L / v_b)) :
  v_a / v_b = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_speed_ratio_l1857_185719


namespace NUMINAMATH_GPT_unique_integer_solution_l1857_185751

theorem unique_integer_solution (n : ℤ) :
  (⌊n^2 / 4 + n⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 10) :=
by sorry

end NUMINAMATH_GPT_unique_integer_solution_l1857_185751


namespace NUMINAMATH_GPT_Kristyna_number_l1857_185736

theorem Kristyna_number (k n : ℕ) (h1 : k = 6 * n + 3) (h2 : 3 * n + 1 + 2 * n = 1681) : k = 2019 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Kristyna_number_l1857_185736


namespace NUMINAMATH_GPT_remainder_of_division_l1857_185718

theorem remainder_of_division :
  ∀ (x : ℝ), (3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8) % (x ^ 2 - 3 * x + 2) = 74 * x - 76 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l1857_185718


namespace NUMINAMATH_GPT_PetyaWinsAgainstSasha_l1857_185752

def MatchesPlayed (name : String) : Nat :=
if name = "Petya" then 12 else if name = "Sasha" then 7 else if name = "Misha" then 11 else 0

def TotalGames : Nat := 15

def GamesMissed (name : String) : Nat :=
if name = "Petya" then TotalGames - MatchesPlayed name else 
if name = "Sasha" then TotalGames - MatchesPlayed name else
if name = "Misha" then TotalGames - MatchesPlayed name else 0

def CanNotMissConsecutiveGames : Prop := True

theorem PetyaWinsAgainstSasha : (GamesMissed "Misha" = 4) ∧ CanNotMissConsecutiveGames → 
  ∃ (winsByPetya : Nat), winsByPetya = 4 :=
by
  sorry

end NUMINAMATH_GPT_PetyaWinsAgainstSasha_l1857_185752


namespace NUMINAMATH_GPT_owners_riding_to_total_ratio_l1857_185772

theorem owners_riding_to_total_ratio (R W : ℕ) (h1 : 4 * R + 6 * W = 90) (h2 : R + W = 18) : R / (R + W) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_owners_riding_to_total_ratio_l1857_185772


namespace NUMINAMATH_GPT_Albert_more_than_Joshua_l1857_185722

def Joshua_rocks : ℕ := 80

def Jose_rocks : ℕ := Joshua_rocks - 14

def Albert_rocks : ℕ := Jose_rocks + 20

theorem Albert_more_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end NUMINAMATH_GPT_Albert_more_than_Joshua_l1857_185722


namespace NUMINAMATH_GPT_find_a_l1857_185732

theorem find_a 
  (a : ℝ) 
  (h : 1 - 2 * a = a - 2) 
  (h1 : 1 - 2 * a = a - 2) 
  : a = 1 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_a_l1857_185732


namespace NUMINAMATH_GPT_cubic_yard_to_cubic_meter_l1857_185776

/-- Define the conversion from yards to meters. -/
def yard_to_meter : ℝ := 0.9144

/-- Theorem stating how many cubic meters are in one cubic yard. -/
theorem cubic_yard_to_cubic_meter :
  (yard_to_meter ^ 3 : ℝ) = 0.7636 :=
by
  sorry

end NUMINAMATH_GPT_cubic_yard_to_cubic_meter_l1857_185776


namespace NUMINAMATH_GPT_x_eq_1_iff_quadratic_eq_zero_l1857_185707

theorem x_eq_1_iff_quadratic_eq_zero :
  ∀ x : ℝ, (x = 1) ↔ (x^2 - 2 * x + 1 = 0) := by
  sorry

end NUMINAMATH_GPT_x_eq_1_iff_quadratic_eq_zero_l1857_185707


namespace NUMINAMATH_GPT_cherry_sodas_l1857_185743

theorem cherry_sodas (C O : ℕ) (h1 : O = 2 * C) (h2 : C + O = 24) : C = 8 :=
by sorry

end NUMINAMATH_GPT_cherry_sodas_l1857_185743


namespace NUMINAMATH_GPT_factorize_ab_factorize_x_l1857_185787

-- Problem 1: Factorization of a^3 b - 2 a^2 b^2 + a b^3
theorem factorize_ab (a b : ℤ) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = a * b * (a - b)^2 := 
by sorry

-- Problem 2: Factorization of (x^2 + 4)^2 - 16 x^2
theorem factorize_x (x : ℤ) : (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 :=
by sorry

end NUMINAMATH_GPT_factorize_ab_factorize_x_l1857_185787


namespace NUMINAMATH_GPT_peanuts_added_l1857_185715

theorem peanuts_added (initial_peanuts final_peanuts added_peanuts : ℕ) 
(h1 : initial_peanuts = 10) 
(h2 : final_peanuts = 18) 
(h3 : final_peanuts = initial_peanuts + added_peanuts) : 
added_peanuts = 8 := 
by {
  sorry
}

end NUMINAMATH_GPT_peanuts_added_l1857_185715


namespace NUMINAMATH_GPT_minimum_value_l1857_185781

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 2) : 
  (1 / m + 2 / n) ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1857_185781


namespace NUMINAMATH_GPT_number_b_is_three_times_number_a_l1857_185720

theorem number_b_is_three_times_number_a (A B : ℕ) (h1 : A = 612) (h2 : B = 3 * A) : B = 1836 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_number_b_is_three_times_number_a_l1857_185720


namespace NUMINAMATH_GPT_exists_integer_multiple_of_3_2008_l1857_185796

theorem exists_integer_multiple_of_3_2008 :
  ∃ k : ℤ, 3 ^ 2008 ∣ (k ^ 3 - 36 * k ^ 2 + 51 * k - 97) :=
sorry

end NUMINAMATH_GPT_exists_integer_multiple_of_3_2008_l1857_185796


namespace NUMINAMATH_GPT_danny_bottle_caps_l1857_185730

variable (caps_found : Nat) (caps_existing : Nat)
variable (wrappers_found : Nat) (wrappers_existing : Nat)

theorem danny_bottle_caps:
  caps_found = 58 → caps_existing = 12 →
  wrappers_found = 25 → wrappers_existing = 11 →
  (caps_found + caps_existing) - (wrappers_found + wrappers_existing) = 34 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_danny_bottle_caps_l1857_185730


namespace NUMINAMATH_GPT_simplify_and_rationalize_denominator_l1857_185791

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_rationalize_denominator_l1857_185791


namespace NUMINAMATH_GPT_tank_filling_l1857_185771

theorem tank_filling (A_rate B_rate : ℚ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) :
  (1 / (A_rate - B_rate)) = 18 :=
by
  sorry

end NUMINAMATH_GPT_tank_filling_l1857_185771


namespace NUMINAMATH_GPT_fault_line_movement_l1857_185750

theorem fault_line_movement
  (moved_past_year : ℝ)
  (moved_year_before : ℝ)
  (h1 : moved_past_year = 1.25)
  (h2 : moved_year_before = 5.25) :
  moved_past_year + moved_year_before = 6.50 :=
by
  sorry

end NUMINAMATH_GPT_fault_line_movement_l1857_185750


namespace NUMINAMATH_GPT_operation_example_result_l1857_185726

def myOperation (A B : ℕ) : ℕ := (A^2 + B^2) / 3

theorem operation_example_result : myOperation (myOperation 6 3) 9 = 102 := by
  sorry

end NUMINAMATH_GPT_operation_example_result_l1857_185726


namespace NUMINAMATH_GPT_find_value_of_x_l1857_185705

theorem find_value_of_x :
  ∃ x : ℝ, (0.65 * x = 0.20 * 747.50) ∧ x = 230 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_x_l1857_185705


namespace NUMINAMATH_GPT_percentage_salt_solution_l1857_185700

theorem percentage_salt_solution (P : ℝ) (V_initial V_added V_final : ℝ) (C_initial C_final : ℝ) :
  V_initial = 30 ∧ C_initial = 0.20 ∧ V_final = 60 ∧ C_final = 0.40 → 
  V_added = 30 → 
  (C_initial * V_initial + (P / 100) * V_added) / V_final = C_final →
  P = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percentage_salt_solution_l1857_185700


namespace NUMINAMATH_GPT_triangle_sides_ratio_l1857_185745

theorem triangle_sides_ratio (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 2 * a)
  (ha_pos : a > 0) : b / a = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_triangle_sides_ratio_l1857_185745


namespace NUMINAMATH_GPT_problem_statement_l1857_185753

def Omega (n : ℕ) : ℕ := 
  -- Number of prime factors of n, counting multiplicity
  sorry

def f1 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 1 (mod 4)
  sorry

def f3 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 3 (mod 4)
  sorry

theorem problem_statement : 
  f3 (6 ^ 2020) - f1 (6 ^ 2020) = (1 / 10 : ℚ) * (6 ^ 2021 - 3 ^ 2021 - 2 ^ 2021 - 1) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1857_185753


namespace NUMINAMATH_GPT_speed_of_second_cyclist_l1857_185725

theorem speed_of_second_cyclist (v : ℝ) 
  (circumference : ℝ) 
  (time : ℝ) 
  (speed_first_cyclist : ℝ)
  (meet_time : ℝ)
  (circ_full: circumference = 300) 
  (time_full: time = 20)
  (speed_first: speed_first_cyclist = 7)
  (meet_full: meet_time = time):

  v = 8 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_second_cyclist_l1857_185725


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l1857_185788

theorem trajectory_of_midpoint (M : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ) :
  (P.1^2 + P.2^2 = 1) ∧
  (P.1 = M.1 ∧ P.2 = 2 * M.2) ∧ 
  (N.1 = P.1 ∧ N.2 = 0) ∧ 
  (M.1 = (P.1 + N.1) / 2 ∧ M.2 = (P.2 + N.2) / 2)
  → M.1^2 + 4 * M.2^2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l1857_185788


namespace NUMINAMATH_GPT_smallest_n_for_inequality_l1857_185754

theorem smallest_n_for_inequality :
  ∃ n : ℤ, (∀ w x y z : ℝ, 
    (w^2 + x^2 + y^2 + z^2)^3 ≤ n * (w^6 + x^6 + y^6 + z^6)) ∧ 
    (∀ m : ℤ, (∀ w x y z : ℝ, 
    (w^2 + x^2 + y^2 + z^2)^3 ≤ m * (w^6 + x^6 + y^6 + z^6)) → m ≥ 64) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_inequality_l1857_185754


namespace NUMINAMATH_GPT_smallest_pythagorean_sum_square_l1857_185798

theorem smallest_pythagorean_sum_square (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p^2 + q^2 = r^2) :
  ∃ (k : ℤ), k = 4 ∧ (p + q + r)^2 ≥ k :=
by
  sorry

end NUMINAMATH_GPT_smallest_pythagorean_sum_square_l1857_185798


namespace NUMINAMATH_GPT_taqeesha_grade_l1857_185797

theorem taqeesha_grade (s : ℕ → ℕ) (h1 : (s 16) = 77) (h2 : (s 17) = 78) : s 17 - s 16 = 94 :=
by
  -- Add definitions and sorry to skip the proof
  sorry

end NUMINAMATH_GPT_taqeesha_grade_l1857_185797


namespace NUMINAMATH_GPT_cubic_roots_inequalities_l1857_185779

theorem cubic_roots_inequalities 
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ z : ℂ, (a * z^3 + b * z^2 + c * z + d = 0) → z.re < 0) :
  a * b > 0 ∧ b * c - a * d > 0 ∧ a * d > 0 :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_inequalities_l1857_185779


namespace NUMINAMATH_GPT_problem_one_problem_two_l1857_185793

variable {α : ℝ}

theorem problem_one (h : Real.tan (π + α) = -1 / 2) :
  (2 * Real.cos (π - α) - 3 * Real.sin (π + α)) / (4 * Real.cos (α - 2 * π) + Real.sin (4 * π - α)) = -7 / 9 :=
sorry

theorem problem_two (h : Real.tan (π + α) = -1 / 2) :
  Real.sin (α - 7 * π) * Real.cos (α + 5 * π) = -2 / 5 :=
sorry

end NUMINAMATH_GPT_problem_one_problem_two_l1857_185793


namespace NUMINAMATH_GPT_derivative_of_f_l1857_185714

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos x

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = Real.exp (2 * x) * (2 * Real.cos x - Real.sin x) :=
by
  intro x
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_derivative_of_f_l1857_185714


namespace NUMINAMATH_GPT_product_of_solutions_abs_eq_four_l1857_185749

theorem product_of_solutions_abs_eq_four :
  (∀ x : ℝ, (|x - 5| - 4 = 0) → (x = 9 ∨ x = 1)) →
  (9 * 1 = 9) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_product_of_solutions_abs_eq_four_l1857_185749


namespace NUMINAMATH_GPT_point_in_second_quadrant_l1857_185795

def isInSecondQuadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : isInSecondQuadrant (-1) 1 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l1857_185795


namespace NUMINAMATH_GPT_probability_of_three_even_numbers_l1857_185784

theorem probability_of_three_even_numbers (n : ℕ) (k : ℕ) (p_even : ℚ) (p_odd : ℚ) (comb : ℕ → ℕ → ℕ) 
    (h_n : n = 5) (h_k : k = 3) (h_p_even : p_even = 1/2) (h_p_odd : p_odd = 1/2) 
    (h_comb : comb 5 3 = 10) :
    comb n k * (p_even ^ k) * (p_odd ^ (n - k)) = 5 / 16 :=
by sorry

end NUMINAMATH_GPT_probability_of_three_even_numbers_l1857_185784


namespace NUMINAMATH_GPT_range_of_a_l1857_185711

noncomputable def discriminant (a : ℝ) : ℝ :=
  (2 * a)^2 - 4 * 1 * 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1857_185711


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l1857_185721

theorem geometric_sequence_first_term (a r : ℚ) (third_term fourth_term : ℚ) 
  (h1 : third_term = a * r^2)
  (h2 : fourth_term = a * r^3)
  (h3 : third_term = 27)
  (h4 : fourth_term = 36) : 
  a = 243 / 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l1857_185721


namespace NUMINAMATH_GPT_number_of_adults_in_sleeper_class_l1857_185739

-- Number of passengers in the train
def total_passengers : ℕ := 320

-- Percentage of passengers who are adults
def percentage_adults : ℚ := 75 / 100

-- Percentage of adults who are in the sleeper class
def percentage_adults_sleeper_class : ℚ := 15 / 100

-- Mathematical statement to prove
theorem number_of_adults_in_sleeper_class :
  (total_passengers * percentage_adults * percentage_adults_sleeper_class) = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_adults_in_sleeper_class_l1857_185739


namespace NUMINAMATH_GPT_rate_per_sq_meter_l1857_185767

theorem rate_per_sq_meter
  (L : ℝ) (W : ℝ) (total_cost : ℝ)
  (hL : L = 6) (hW : W = 4.75) (h_total_cost : total_cost = 25650) :
  total_cost / (L * W) = 900 :=
by
  sorry

end NUMINAMATH_GPT_rate_per_sq_meter_l1857_185767


namespace NUMINAMATH_GPT_julia_average_speed_l1857_185785

theorem julia_average_speed :
  let distance1 := 45
  let speed1 := 15
  let distance2 := 15
  let speed2 := 45
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 18 := by
sorry

end NUMINAMATH_GPT_julia_average_speed_l1857_185785


namespace NUMINAMATH_GPT_total_amount_l1857_185768

theorem total_amount (P Q R : ℝ) (h1 : R = 2 / 3 * (P + Q)) (h2 : R = 3200) : P + Q + R = 8000 := 
by
  sorry

end NUMINAMATH_GPT_total_amount_l1857_185768


namespace NUMINAMATH_GPT_remainder_when_divided_by_seven_l1857_185766

theorem remainder_when_divided_by_seven (n : ℕ) (h₁ : n^3 ≡ 3 [MOD 7]) (h₂ : n^4 ≡ 2 [MOD 7]) : 
  n ≡ 6 [MOD 7] :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_seven_l1857_185766


namespace NUMINAMATH_GPT_averageSpeed_is_45_l1857_185763

/-- Define the upstream and downstream speeds of the fish --/
def fishA_upstream_speed := 40
def fishA_downstream_speed := 60
def fishB_upstream_speed := 30
def fishB_downstream_speed := 50
def fishC_upstream_speed := 45
def fishC_downstream_speed := 65
def fishD_upstream_speed := 35
def fishD_downstream_speed := 55
def fishE_upstream_speed := 25
def fishE_downstream_speed := 45

/-- Define a function to calculate the speed in still water --/
def stillWaterSpeed (upstream_speed : ℕ) (downstream_speed : ℕ) : ℕ :=
  (upstream_speed + downstream_speed) / 2

/-- Calculate the still water speed for each fish --/
def fishA_speed := stillWaterSpeed fishA_upstream_speed fishA_downstream_speed
def fishB_speed := stillWaterSpeed fishB_upstream_speed fishB_downstream_speed
def fishC_speed := stillWaterSpeed fishC_upstream_speed fishC_downstream_speed
def fishD_speed := stillWaterSpeed fishD_upstream_speed fishD_downstream_speed
def fishE_speed := stillWaterSpeed fishE_upstream_speed fishE_downstream_speed

/-- Calculate the average speed of all fish in still water --/
def averageSpeedInStillWater :=
  (fishA_speed + fishB_speed + fishC_speed + fishD_speed + fishE_speed) / 5

/-- The statement to prove --/
theorem averageSpeed_is_45 : averageSpeedInStillWater = 45 :=
  sorry

end NUMINAMATH_GPT_averageSpeed_is_45_l1857_185763


namespace NUMINAMATH_GPT_map_at_three_l1857_185702

variable (A B : Type)
variable (a : ℝ)
variable (f : ℝ → ℝ)
variable (h_map : ∀ x : ℝ, f x = a * x - 1)
variable (h_cond : f 2 = 3)

theorem map_at_three : f 3 = 5 := by
  sorry

end NUMINAMATH_GPT_map_at_three_l1857_185702


namespace NUMINAMATH_GPT_infinite_solutions_of_system_l1857_185764

theorem infinite_solutions_of_system :
  ∃x y : ℝ, (3 * x - 4 * y = 10 ∧ 6 * x - 8 * y = 20) :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_of_system_l1857_185764


namespace NUMINAMATH_GPT_subsets_union_l1857_185783

theorem subsets_union (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 2^(n-1) + 1) 
  (A : Fin m → Finset (Fin n)) (hA : ∀ i j, i ≠ j → A i ≠ A j) 
  (hB : ∀ i, A i ≠ ∅) : 
  ∃ i j k, i ≠ j ∧ A i ∪ A j = A k := 
sorry

end NUMINAMATH_GPT_subsets_union_l1857_185783


namespace NUMINAMATH_GPT_steven_falls_correct_l1857_185789

/-
  We will model the problem where we are given the conditions about the falls of Steven, Stephanie,
  and Sonya, and then prove that the number of times Steven fell is 3.
-/

variables (S : ℕ) -- Steven's falls

-- Conditions
def stephanie_falls := S + 13
def sonya_falls := 6 
def sonya_condition := 6 = (stephanie_falls / 2) - 2

-- Theorem statement
theorem steven_falls_correct : S = 3 :=
by {
  -- Note: the actual proof steps would go here, but are omitted per instructions
  sorry
}

end NUMINAMATH_GPT_steven_falls_correct_l1857_185789


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1857_185717

theorem simplify_expr1 : 
  (1:ℝ) * (-3:ℝ) ^ 0 + (- (1/2:ℝ)) ^ (-2:ℝ) - (-3:ℝ) ^ (-1:ℝ) = 16 / 3 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 
  ((-2 * x^3) ^ 2 * (-x^2)) / ((-x)^2) ^ 3 = -4 * x^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1857_185717


namespace NUMINAMATH_GPT_range_of_a_for_solution_set_l1857_185744

theorem range_of_a_for_solution_set (a : ℝ) :
  ((∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1)) :=
sorry

end NUMINAMATH_GPT_range_of_a_for_solution_set_l1857_185744


namespace NUMINAMATH_GPT_average_pages_per_day_l1857_185738

variable (total_pages : ℕ := 160)
variable (pages_read : ℕ := 60)
variable (days_left : ℕ := 5)

theorem average_pages_per_day : (total_pages - pages_read) / days_left = 20 := by
  sorry

end NUMINAMATH_GPT_average_pages_per_day_l1857_185738


namespace NUMINAMATH_GPT_minimum_value_proof_l1857_185762

noncomputable def minimum_value : ℝ :=
  3 + 2 * Real.sqrt 2

theorem minimum_value_proof (a b : ℝ) (h_line_eq : ∀ x y : ℝ, a * x + b * y = 1)
  (h_ab_pos : a * b > 0)
  (h_center_bisect : ∃ x y : ℝ, (x - 1)^2 + (y - 2)^2 <= x^2 + y^2) :
  (1 / a + 1 / b) ≥ minimum_value :=
by
  -- Sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_minimum_value_proof_l1857_185762


namespace NUMINAMATH_GPT_sin_order_l1857_185794

theorem sin_order :
  ∀ (sin₁ sin₂ sin₃ sin₄ sin₆ : ℝ),
  sin₁ = Real.sin 1 ∧ 
  sin₂ = Real.sin 2 ∧ 
  sin₃ = Real.sin 3 ∧ 
  sin₄ = Real.sin 4 ∧ 
  sin₆ = Real.sin 6 →
  sin₂ > sin₁ ∧ sin₁ > sin₃ ∧ sin₃ > sin₆ ∧ sin₆ > sin₄ :=
by
  sorry

end NUMINAMATH_GPT_sin_order_l1857_185794


namespace NUMINAMATH_GPT_inequality_proof_l1857_185792

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1857_185792


namespace NUMINAMATH_GPT_harry_travel_time_l1857_185703

def t_bus1 : ℕ := 15
def t_bus2 : ℕ := 25
def t_bus_journey : ℕ := t_bus1 + t_bus2
def t_walk : ℕ := t_bus_journey / 2
def t_total : ℕ := t_bus_journey + t_walk

theorem harry_travel_time : t_total = 60 := by
  -- Will be proved afterwards
  sorry

end NUMINAMATH_GPT_harry_travel_time_l1857_185703


namespace NUMINAMATH_GPT_one_girl_made_a_mistake_l1857_185716

variables (c_M c_K c_L c_O : ℤ)

theorem one_girl_made_a_mistake (h₁ : c_M + c_K = c_L + c_O + 12) (h₂ : c_K + c_L = c_M + c_O - 7) :
  false := by
  -- Proof intentionally missing
  sorry

end NUMINAMATH_GPT_one_girl_made_a_mistake_l1857_185716


namespace NUMINAMATH_GPT_percentage_of_oysters_with_pearls_l1857_185757

def jamie_collects_oysters (oysters_per_dive dives total_pearls : ℕ) : ℕ :=
  oysters_per_dive * dives

def percentage_with_pearls (total_pearls total_oysters : ℕ) : ℕ :=
  (total_pearls * 100) / total_oysters

theorem percentage_of_oysters_with_pearls :
  ∀ (oysters_per_dive dives total_pearls : ℕ),
  oysters_per_dive = 16 →
  dives = 14 →
  total_pearls = 56 →
  percentage_with_pearls total_pearls (jamie_collects_oysters oysters_per_dive dives total_pearls) = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_of_oysters_with_pearls_l1857_185757


namespace NUMINAMATH_GPT_perimeter_correct_l1857_185769

open EuclideanGeometry

noncomputable def perimeter_of_figure : ℝ := 
  let AB : ℝ := 6
  let BC : ℝ := AB
  let AD : ℝ := AB / 2
  let DC : ℝ := AD
  let DE : ℝ := AD
  let EA : ℝ := DE
  let EF : ℝ := EA / 2
  let FG : ℝ := EF
  let GH : ℝ := FG / 2
  let HJ : ℝ := GH
  let JA : ℝ := HJ
  AB + BC + DC + DE + EF + FG + GH + HJ + JA

theorem perimeter_correct : perimeter_of_figure = 23.25 :=
by
  -- proof steps would go here, but are not required for this problem transformation
  sorry

end NUMINAMATH_GPT_perimeter_correct_l1857_185769


namespace NUMINAMATH_GPT_investment_ratio_l1857_185724

theorem investment_ratio (total_investment Jim_investment : ℕ) (h₁ : total_investment = 80000) (h₂ : Jim_investment = 36000) :
  (total_investment - Jim_investment) / Nat.gcd (total_investment - Jim_investment) Jim_investment = 11 ∧ Jim_investment / Nat.gcd (total_investment - Jim_investment) Jim_investment = 9 :=
by
  sorry

end NUMINAMATH_GPT_investment_ratio_l1857_185724


namespace NUMINAMATH_GPT_tshirt_costs_more_than_jersey_l1857_185733

open Nat

def cost_tshirt : ℕ := 192
def cost_jersey : ℕ := 34

theorem tshirt_costs_more_than_jersey :
  cost_tshirt - cost_jersey = 158 :=
by sorry

end NUMINAMATH_GPT_tshirt_costs_more_than_jersey_l1857_185733


namespace NUMINAMATH_GPT_tulip_area_of_flower_bed_l1857_185713

theorem tulip_area_of_flower_bed 
  (CD CF : ℝ) (DE : ℝ := 4) (EF : ℝ := 3) 
  (triangle : ∀ (A B C : ℝ), A = B + C) : 
  CD * CF = 12 :=
by sorry

end NUMINAMATH_GPT_tulip_area_of_flower_bed_l1857_185713


namespace NUMINAMATH_GPT_tangency_point_l1857_185747

theorem tangency_point (x y : ℝ) : 
  y = x ^ 2 + 20 * x + 70 ∧ x = y ^ 2 + 70 * y + 1225 →
  (x, y) = (-19 / 2, -69 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_tangency_point_l1857_185747


namespace NUMINAMATH_GPT_tangent_line_equation_at_x_zero_l1857_185723

noncomputable def curve (x : ℝ) : ℝ := x + Real.exp (2 * x)

theorem tangent_line_equation_at_x_zero :
  ∃ (k b : ℝ), (∀ x : ℝ, curve x = k * x + b) :=
by
  let df := fun (x : ℝ) => (deriv curve x)
  have k : ℝ := df 0
  have b : ℝ := curve 0 - k * 0
  use k, b
  sorry

end NUMINAMATH_GPT_tangent_line_equation_at_x_zero_l1857_185723


namespace NUMINAMATH_GPT_proof_4_minus_a_l1857_185727

theorem proof_4_minus_a :
  ∀ (a b : ℚ),
    (5 + a = 7 - b) →
    (3 + b = 8 + a) →
    4 - a = 11 / 2 :=
by
  intros a b h1 h2
  sorry

end NUMINAMATH_GPT_proof_4_minus_a_l1857_185727


namespace NUMINAMATH_GPT_plot_length_l1857_185735

def breadth : ℝ := 40 -- Derived from conditions and cost equation solution
def length : ℝ := breadth + 20
def cost_per_meter : ℝ := 26.50
def total_cost : ℝ := 5300

theorem plot_length :
  (2 * (breadth + (breadth + 20))) * cost_per_meter = total_cost → length = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_plot_length_l1857_185735


namespace NUMINAMATH_GPT_intersecting_lines_c_plus_d_l1857_185740

theorem intersecting_lines_c_plus_d (c d : ℝ) 
  (h1 : ∀ y, ∃ x, x = (1/3) * y + c) 
  (h2 : ∀ x, ∃ y, y = (1/3) * x + d)
  (P : (3:ℝ) = (1 / 3) * (3:ℝ) + c) 
  (Q : (3:ℝ) = (1 / 3) * (3:ℝ) + d) : 
  c + d = 4 := 
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_c_plus_d_l1857_185740


namespace NUMINAMATH_GPT_smallest_arithmetic_mean_divisible_by_1111_l1857_185790

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end NUMINAMATH_GPT_smallest_arithmetic_mean_divisible_by_1111_l1857_185790


namespace NUMINAMATH_GPT_andrew_cookies_per_day_l1857_185708

/-- Number of days in May --/
def days_in_may : ℤ := 31

/-- Cost per cookie in dollars --/
def cost_per_cookie : ℤ := 15

/-- Total amount spent by Andrew on cookies in dollars --/
def total_amount_spent : ℤ := 1395

/-- Total number of cookies purchased by Andrew --/
def total_cookies : ℤ := total_amount_spent / cost_per_cookie

/-- Number of cookies purchased per day --/
def cookies_per_day : ℤ := total_cookies / days_in_may

theorem andrew_cookies_per_day : cookies_per_day = 3 := by
  sorry

end NUMINAMATH_GPT_andrew_cookies_per_day_l1857_185708


namespace NUMINAMATH_GPT_solve_inequality_l1857_185734

-- Declare the necessary conditions as variables in Lean
variables (a c : ℝ)

-- State the Lean theorem
theorem solve_inequality :
  (∀ x : ℝ, (ax^2 + 5 * x + c > 0) ↔ (1/3 < x ∧ x < 1/2)) →
  a < 0 →
  a = -6 ∧ c = -1 :=
  sorry

end NUMINAMATH_GPT_solve_inequality_l1857_185734


namespace NUMINAMATH_GPT_pedestrians_speed_ratio_l1857_185755

-- Definitions based on conditions
variable (v v1 v2 : ℝ)

-- Conditions
def first_meeting (v1 v : ℝ) := (1 / 3) * v1 = (1 / 4) * v
def second_meeting (v2 v : ℝ) := (5 / 12) * v2 = (1 / 6) * v

-- Theorem Statement
theorem pedestrians_speed_ratio (h1 : first_meeting v1 v) (h2 : second_meeting v2 v) : v1 / v2 = 15 / 8 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_pedestrians_speed_ratio_l1857_185755


namespace NUMINAMATH_GPT_no_d1_d2_multiple_of_7_l1857_185786
open Function

theorem no_d1_d2_multiple_of_7 (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 100) :
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  ¬(d1 * d2 % 7 = 0) :=
by
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  sorry

end NUMINAMATH_GPT_no_d1_d2_multiple_of_7_l1857_185786


namespace NUMINAMATH_GPT_profit_percentage_l1857_185782

-- Given conditions
def CP : ℚ := 25 / 15
def SP : ℚ := 32 / 12

-- To prove profit percentage is 60%
theorem profit_percentage (CP SP : ℚ) (hCP : CP = 25 / 15) (hSP : SP = 32 / 12) :
  (SP - CP) / CP * 100 = 60 := 
by 
  sorry

end NUMINAMATH_GPT_profit_percentage_l1857_185782


namespace NUMINAMATH_GPT_find_hanyoung_weight_l1857_185731

variable (H J : ℝ)

def hanyoung_is_lighter (H J : ℝ) : Prop := H = J - 4
def sum_of_weights (H J : ℝ) : Prop := H + J = 88

theorem find_hanyoung_weight (H J : ℝ) (h1 : hanyoung_is_lighter H J) (h2 : sum_of_weights H J) : H = 42 :=
by
  sorry

end NUMINAMATH_GPT_find_hanyoung_weight_l1857_185731


namespace NUMINAMATH_GPT_length_of_platform_l1857_185709

theorem length_of_platform
  (length_of_train time_crossing_platform time_crossing_pole : ℝ) 
  (length_of_train_eq : length_of_train = 400)
  (time_crossing_platform_eq : time_crossing_platform = 45)
  (time_crossing_pole_eq : time_crossing_pole = 30) :
  ∃ (L : ℝ), (400 + L) / time_crossing_platform = length_of_train / time_crossing_pole :=
by {
  use 200,
  sorry
}

end NUMINAMATH_GPT_length_of_platform_l1857_185709


namespace NUMINAMATH_GPT_iron_weighs_more_l1857_185737

-- Define the weights of the metal pieces
def weight_iron : ℝ := 11.17
def weight_aluminum : ℝ := 0.83

-- State the theorem to prove that the difference in weights is 10.34 pounds
theorem iron_weighs_more : weight_iron - weight_aluminum = 10.34 :=
by sorry

end NUMINAMATH_GPT_iron_weighs_more_l1857_185737


namespace NUMINAMATH_GPT_participating_girls_l1857_185778

theorem participating_girls (total_students boys_participation girls_participation participating_students : ℕ)
  (h1 : total_students = 800)
  (h2 : boys_participation = 2)
  (h3 : girls_participation = 3)
  (h4 : participating_students = 550) :
  (4 / total_students) * (boys_participation / 3) * total_students + (4 * girls_participation / 4) * total_students = 4 * 150 :=
by
  sorry

end NUMINAMATH_GPT_participating_girls_l1857_185778
