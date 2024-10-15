import Mathlib

namespace NUMINAMATH_GPT_fraction_to_decimal_l1157_115746

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 := 
sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1157_115746


namespace NUMINAMATH_GPT_apple_difference_l1157_115712

def carla_apples : ℕ := 7
def tim_apples : ℕ := 1

theorem apple_difference : carla_apples - tim_apples = 6 := by
  sorry

end NUMINAMATH_GPT_apple_difference_l1157_115712


namespace NUMINAMATH_GPT_total_cost_correct_l1157_115766

-- Define the parameters
variables (a : ℕ) -- the number of books
-- Define the constants and the conditions
def unit_price : ℝ := 8
def shipping_fee_percentage : ℝ := 0.10

-- Define the total cost including the shipping fee
def total_cost (a : ℕ) : ℝ := unit_price * (1 + shipping_fee_percentage) * a

-- Prove that the total cost is equal to the expected amount
theorem total_cost_correct : total_cost a = 8 * (1 + 0.10) * a := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1157_115766


namespace NUMINAMATH_GPT_temperature_problem_l1157_115774

theorem temperature_problem (N : ℤ) (M L : ℤ) :
  M = L + N →
  (M - 10) - (L + 6) = 4 ∨ (M - 10) - (L + 6) = -4 →
  (N - 16 = 4 ∨ 16 - N = 4) →
  ((N = 20 ∨ N = 12) → 20 * 12 = 240) :=
by
   sorry

end NUMINAMATH_GPT_temperature_problem_l1157_115774


namespace NUMINAMATH_GPT_tan_alpha_eq_2_l1157_115797

theorem tan_alpha_eq_2 (α : Real) (h : Real.tan α = 2) : 
  1 / (Real.sin (2 * α) + Real.cos (α) ^ 2) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_2_l1157_115797


namespace NUMINAMATH_GPT_find_value_of_x_l1157_115737

theorem find_value_of_x (x : ℕ) (h : (50 + x / 90) * 90 = 4520) : x = 4470 :=
sorry

end NUMINAMATH_GPT_find_value_of_x_l1157_115737


namespace NUMINAMATH_GPT_series_sum_eq_l1157_115734

noncomputable def sum_series : ℝ :=
∑' n : ℕ, if h : n > 0 then (4 * n + 3) / ((4 * n)^2 * (4 * n + 4)^2) else 0

theorem series_sum_eq :
  sum_series = 1 / 256 := by
  sorry

end NUMINAMATH_GPT_series_sum_eq_l1157_115734


namespace NUMINAMATH_GPT_probability_rain_once_l1157_115793

theorem probability_rain_once (p : ℚ) 
  (h₁ : p = 1 / 2) 
  (h₂ : 1 - p = 1 / 2) 
  (h₃ : (1 - p) ^ 4 = 1 / 16) 
  : 1 - (1 - p) ^ 4 = 15 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_rain_once_l1157_115793


namespace NUMINAMATH_GPT_boys_amount_per_person_l1157_115740

theorem boys_amount_per_person (total_money : ℕ) (total_children : ℕ) (per_girl : ℕ) (number_of_boys : ℕ) (amount_per_boy : ℕ) : 
  total_money = 460 ∧
  total_children = 41 ∧
  per_girl = 8 ∧
  number_of_boys = 33 → 
  amount_per_boy = 12 :=
by sorry

end NUMINAMATH_GPT_boys_amount_per_person_l1157_115740


namespace NUMINAMATH_GPT_clock_displays_unique_digits_minutes_l1157_115756

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end NUMINAMATH_GPT_clock_displays_unique_digits_minutes_l1157_115756


namespace NUMINAMATH_GPT_farm_field_proof_l1157_115739

section FarmField

variables 
  (planned_rate daily_rate : ℕ) -- planned_rate is 260 hectares/day, daily_rate is 85 hectares/day 
  (extra_days remaining_hectares : ℕ) -- extra_days is 2, remaining_hectares is 40
  (max_hours_per_day : ℕ) -- max_hours_per_day is 12

-- Definitions for soils
variables
  (A_percent B_percent C_percent : ℚ) (A_hours B_hours C_hours : ℕ)
  -- A_percent is 0.4, B_percent is 0.3, C_percent is 0.3
  -- A_hours is 4, B_hours is 6, C_hours is 3

-- Given conditions
axiom planned_rate_eq : planned_rate = 260
axiom daily_rate_eq : daily_rate = 85
axiom extra_days_eq : extra_days = 2
axiom remaining_hectares_eq : remaining_hectares = 40
axiom max_hours_per_day_eq : max_hours_per_day = 12

axiom A_percent_eq : A_percent = 0.4
axiom B_percent_eq : B_percent = 0.3
axiom C_percent_eq : C_percent = 0.3

axiom A_hours_eq : A_hours = 4
axiom B_hours_eq : B_hours = 6
axiom C_hours_eq : C_hours = 3

-- Theorem stating the problem
theorem farm_field_proof :
  ∃ (total_area initial_days : ℕ),
    total_area = 340 ∧ initial_days = 2 :=
by
  sorry

end FarmField

end NUMINAMATH_GPT_farm_field_proof_l1157_115739


namespace NUMINAMATH_GPT_Jack_emails_evening_l1157_115792

theorem Jack_emails_evening : 
  ∀ (morning_emails evening_emails : ℕ), 
  (morning_emails = 9) ∧ 
  (evening_emails = morning_emails - 2) → 
  evening_emails = 7 := 
by
  intros morning_emails evening_emails
  sorry

end NUMINAMATH_GPT_Jack_emails_evening_l1157_115792


namespace NUMINAMATH_GPT_interval_intersection_l1157_115767

theorem interval_intersection (x : ℝ) :
  (4 * x > 2 ∧ 4 * x < 3) ∧ (5 * x > 2 ∧ 5 * x < 3) ↔ (x > 1/2 ∧ x < 3/5) :=
by
  sorry

end NUMINAMATH_GPT_interval_intersection_l1157_115767


namespace NUMINAMATH_GPT_three_digit_numbers_m_l1157_115750

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem three_digit_numbers_m (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 ∧ sum_of_digits n = 12 ∧ 100 ≤ 2 * n ∧ 2 * n ≤ 999 ∧ sum_of_digits (2 * n) = 6 → ∃! (m : ℕ), n = m :=
sorry

end NUMINAMATH_GPT_three_digit_numbers_m_l1157_115750


namespace NUMINAMATH_GPT_convert_speed_to_mps_l1157_115763

-- Define given speeds and conversion factors
def speed_kmph : ℝ := 63
def kilometers_to_meters : ℝ := 1000
def hours_to_seconds : ℝ := 3600

-- Assert the conversion
theorem convert_speed_to_mps : speed_kmph * (kilometers_to_meters / hours_to_seconds) = 17.5 := by
  sorry

end NUMINAMATH_GPT_convert_speed_to_mps_l1157_115763


namespace NUMINAMATH_GPT_find_m_squared_plus_n_squared_l1157_115776

theorem find_m_squared_plus_n_squared (m n : ℝ) (h1 : (m - n) ^ 2 = 8) (h2 : (m + n) ^ 2 = 2) : m ^ 2 + n ^ 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_squared_plus_n_squared_l1157_115776


namespace NUMINAMATH_GPT_correct_answer_l1157_115717

def M : Set ℤ := {x | |x| < 5}

theorem correct_answer : {0} ⊆ M := by
  sorry

end NUMINAMATH_GPT_correct_answer_l1157_115717


namespace NUMINAMATH_GPT_stacy_history_paper_length_l1157_115775

theorem stacy_history_paper_length
  (days : ℕ)
  (pages_per_day : ℕ)
  (h_days : days = 6)
  (h_pages_per_day : pages_per_day = 11) :
  (days * pages_per_day) = 66 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_stacy_history_paper_length_l1157_115775


namespace NUMINAMATH_GPT_farmer_revenue_correct_l1157_115787

-- Define the conditions
def average_bacon : ℕ := 20
def price_per_pound : ℕ := 6
def size_factor : ℕ := 1 / 2

-- Calculate the bacon from the runt pig
def bacon_from_runt := average_bacon * size_factor

-- Calculate the revenue from selling the bacon
def revenue := bacon_from_runt * price_per_pound

-- Lean 4 Statement to prove
theorem farmer_revenue_correct :
  revenue = 60 :=
sorry

end NUMINAMATH_GPT_farmer_revenue_correct_l1157_115787


namespace NUMINAMATH_GPT_boxes_needed_for_loose_crayons_l1157_115730

-- Definitions based on conditions
def boxes_francine : ℕ := 5
def loose_crayons_francine : ℕ := 5
def loose_crayons_friend : ℕ := 27
def total_crayons_francine : ℕ := 85
def total_boxes_needed : ℕ := 2

-- The theorem to prove
theorem boxes_needed_for_loose_crayons 
  (hf : total_crayons_francine = boxes_francine * 16 + loose_crayons_francine)
  (htotal_loose : loose_crayons_francine + loose_crayons_friend = 32)
  (hboxes : boxes_francine = 5) : 
  total_boxes_needed = 2 :=
sorry

end NUMINAMATH_GPT_boxes_needed_for_loose_crayons_l1157_115730


namespace NUMINAMATH_GPT_calculate_sum_l1157_115762

open Real

theorem calculate_sum :
  (-1: ℝ) ^ 2023 + (1/2) ^ (-2: ℝ) + 3 * tan (pi / 6) - (3 - pi) ^ 0 + |sqrt 3 - 2| = 4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_sum_l1157_115762


namespace NUMINAMATH_GPT_solve_equation_floor_l1157_115724

theorem solve_equation_floor (x : ℚ) :
  (⌊(5 + 6 * x) / 8⌋ : ℚ) = (15 * x - 7) / 5 ↔ x = 7 / 15 ∨ x = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_floor_l1157_115724


namespace NUMINAMATH_GPT_tan_sum_identity_l1157_115720

theorem tan_sum_identity (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : Real.tan x = 1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_sum_identity_l1157_115720


namespace NUMINAMATH_GPT_find_values_l1157_115786

variable (circle triangle : ℕ)

axiom condition1 : triangle = circle + circle + circle
axiom condition2 : triangle + circle = 40

theorem find_values : circle = 10 ∧ triangle = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_values_l1157_115786


namespace NUMINAMATH_GPT_sin_thirteen_pi_over_six_l1157_115722

-- Define a lean statement for the proof problem
theorem sin_thirteen_pi_over_six : Real.sin (13 * Real.pi / 6) = 1 / 2 := 
by 
  -- Add the proof later (or keep sorry if the proof is not needed)
  sorry

end NUMINAMATH_GPT_sin_thirteen_pi_over_six_l1157_115722


namespace NUMINAMATH_GPT_maximize_annual_profit_l1157_115742

theorem maximize_annual_profit : 
  ∃ n : ℕ, n ≠ 0 ∧ (∀ m : ℕ, m ≠ 0 → (110 * n - (n * n + n) - 90) / n ≥ (110 * m - (m * m + m) - 90) / m) ↔ n = 5 := 
by 
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_maximize_annual_profit_l1157_115742


namespace NUMINAMATH_GPT_sum_of_pairwise_relatively_prime_numbers_l1157_115796

theorem sum_of_pairwise_relatively_prime_numbers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
    (h4 : a * b * c = 302400) (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
    a + b + c = 320 :=
sorry

end NUMINAMATH_GPT_sum_of_pairwise_relatively_prime_numbers_l1157_115796


namespace NUMINAMATH_GPT_part_I_part_II_l1157_115747

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem part_I (x : ℝ) : (f x > 5) ↔ (x < -3 ∨ x > 2) :=
  sorry

theorem part_II (a : ℝ) : (∀ x, f x < a ↔ false) ↔ (a ≤ 3) :=
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1157_115747


namespace NUMINAMATH_GPT_total_earnings_correct_l1157_115784

noncomputable def total_earnings (a_days b_days c_days b_share : ℝ) : ℝ :=
  let a_work_per_day := 1 / a_days
  let b_work_per_day := 1 / b_days
  let c_work_per_day := 1 / c_days
  let combined_work_per_day := a_work_per_day + b_work_per_day + c_work_per_day
  let b_fraction_of_total_work := b_work_per_day / combined_work_per_day
  let total_earnings := b_share / b_fraction_of_total_work
  total_earnings

theorem total_earnings_correct :
  total_earnings 6 8 12 780.0000000000001 = 2340 :=
by
  sorry

end NUMINAMATH_GPT_total_earnings_correct_l1157_115784


namespace NUMINAMATH_GPT_julie_aaron_age_l1157_115781

variables {J A m : ℕ}

theorem julie_aaron_age : (J = 4 * A) → (J + 10 = m * (A + 10)) → (m = 4) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_julie_aaron_age_l1157_115781


namespace NUMINAMATH_GPT_product_of_A_and_B_l1157_115755

theorem product_of_A_and_B (A B : ℕ) (h1 : 3 / 9 = 6 / A) (h2 : B / 63 = 6 / A) : A * B = 378 :=
  sorry

end NUMINAMATH_GPT_product_of_A_and_B_l1157_115755


namespace NUMINAMATH_GPT_pete_should_leave_by_0730_l1157_115713

def walking_time : ℕ := 10
def train_time : ℕ := 80
def latest_arrival_time : String := "0900"
def departure_time : String := "0730"

theorem pete_should_leave_by_0730 :
  (latest_arrival_time = "0900" → walking_time = 10 ∧ train_time = 80 → departure_time = "0730") := by
  sorry

end NUMINAMATH_GPT_pete_should_leave_by_0730_l1157_115713


namespace NUMINAMATH_GPT_abs_diff_m_n_l1157_115757

variable (m n : ℝ)

theorem abs_diff_m_n (h1 : m * n = 6) (h2 : m + n = 7) (h3 : m^2 - n^2 = 13) : |m - n| = 13 / 7 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_m_n_l1157_115757


namespace NUMINAMATH_GPT_sequence_property_l1157_115710

def sequence_conditions (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  ∀ n ≥ 3, S n + S (n - 2) = 2 * S (n - 1) + n

theorem sequence_property (a : ℕ → ℕ) (S : ℕ → ℕ) (h : sequence_conditions a S) : 
  ∀ n ≥ 3, a n = a (n - 1) + n :=
  sorry

end NUMINAMATH_GPT_sequence_property_l1157_115710


namespace NUMINAMATH_GPT_remainder_is_15x_minus_14_l1157_115795

noncomputable def remainder_polynomial_division : Polynomial ℝ :=
  (Polynomial.X ^ 4) % (Polynomial.X ^ 2 - 3 * Polynomial.X + 2)

theorem remainder_is_15x_minus_14 :
  remainder_polynomial_division = 15 * Polynomial.X - 14 :=
by
  sorry

end NUMINAMATH_GPT_remainder_is_15x_minus_14_l1157_115795


namespace NUMINAMATH_GPT_smaller_rectangle_perimeter_l1157_115780

def perimeter_original_rectangle (a b : ℝ) : Prop := 2 * (a + b) = 100
def number_of_cuts (vertical_cuts horizontal_cuts : ℕ) : Prop := vertical_cuts = 7 ∧ horizontal_cuts = 10
def total_length_of_cuts (a b : ℝ) : Prop := 7 * b + 10 * a = 434

theorem smaller_rectangle_perimeter (a b : ℝ) (vertical_cuts horizontal_cuts : ℕ) (m n : ℕ) :
  perimeter_original_rectangle a b →
  number_of_cuts vertical_cuts horizontal_cuts →
  total_length_of_cuts a b →
  (m = 8) →
  (n = 11) →
  (a / 8 + b / 11) * 2 = 11 :=
by
  sorry

end NUMINAMATH_GPT_smaller_rectangle_perimeter_l1157_115780


namespace NUMINAMATH_GPT_intersection_M_N_l1157_115785

def M : Set ℝ := {x : ℝ | |x| < 1}
def N : Set ℝ := {x : ℝ | x^2 - x < 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1157_115785


namespace NUMINAMATH_GPT_power_mul_eq_l1157_115748

variable (a : ℝ)

theorem power_mul_eq :
  (-a)^2 * a^4 = a^6 :=
by sorry

end NUMINAMATH_GPT_power_mul_eq_l1157_115748


namespace NUMINAMATH_GPT_perpendicular_bisector_eq_l1157_115728

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- Prove that the perpendicular bisector of line segment AB has the equation 3x - y - 9 = 0
theorem perpendicular_bisector_eq :
  (∀ x y : ℝ, C1 x y → C2 x y → 3 * x - y - 9 = 0) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_bisector_eq_l1157_115728


namespace NUMINAMATH_GPT_find_fraction_l1157_115772

theorem find_fraction (x : ℝ) (h1 : 7 = (1 / 10) / 100 * 7000) (h2 : x * 7000 - 7 = 700) : x = 707 / 7000 :=
by sorry

end NUMINAMATH_GPT_find_fraction_l1157_115772


namespace NUMINAMATH_GPT_arrange_moon_l1157_115754

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ways_to_arrange_moon : ℕ :=
  let total_letters := 4
  let repeated_O_count := 2
  factorial total_letters / factorial repeated_O_count

theorem arrange_moon : ways_to_arrange_moon = 12 := 
by {
  sorry -- Proof is omitted as instructed
}

end NUMINAMATH_GPT_arrange_moon_l1157_115754


namespace NUMINAMATH_GPT_min_value_eq_144_l1157_115701

noncomputable def min_value (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) : ℝ :=
  if x <= 0 ∨ y <= 0 ∨ z <= 0 ∨ w <= 0 then 0 else (x + y + z) / (x * y * z * w)

theorem min_value_eq_144 (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) :
  min_value x y z w h_pos_x h_pos_y h_pos_z h_pos_w h_sum = 144 :=
sorry

end NUMINAMATH_GPT_min_value_eq_144_l1157_115701


namespace NUMINAMATH_GPT_square_of_binomial_l1157_115758

theorem square_of_binomial (a b : ℝ) : 
  (a - 5 * b)^2 = a^2 - 10 * a * b + 25 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_l1157_115758


namespace NUMINAMATH_GPT_calc_result_l1157_115741

theorem calc_result (initial_number : ℕ) (square : ℕ → ℕ) (subtract_five : ℕ → ℕ) : 
  initial_number = 7 ∧ (square 7 = 49) ∧ (subtract_five 49 = 44) → 
  subtract_five (square initial_number) = 44 := 
by
  sorry

end NUMINAMATH_GPT_calc_result_l1157_115741


namespace NUMINAMATH_GPT_solve_system_1_solve_system_2_solve_system_3_solve_system_4_l1157_115788

-- System 1
theorem solve_system_1 (x y : ℝ) (h1 : x = y + 1) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
  sorry

-- System 2
theorem solve_system_2 (x y : ℝ) (h1 : 3 * x + y = 8) (h2 : x - y = 4) : x = 3 ∧ y = -1 :=
by
  sorry

-- System 3
theorem solve_system_3 (x y : ℝ) (h1 : 5 * x + 3 * y = 2) (h2 : 3 * x + 2 * y = 1) : x = 1 ∧ y = -1 :=
by
  sorry

-- System 4
theorem solve_system_4 (x y z : ℝ) (h1 : x + y = 3) (h2 : y + z = -2) (h3 : z + x = 9) : x = 7 ∧ y = -4 ∧ z = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_1_solve_system_2_solve_system_3_solve_system_4_l1157_115788


namespace NUMINAMATH_GPT_prop_A_prop_B_prop_C_prop_D_l1157_115771

-- Proposition A: For all x ∈ ℝ, x² - x + 1 > 0
theorem prop_A (x : ℝ) : x^2 - x + 1 > 0 :=
sorry

-- Proposition B: a² + a = 0 is not a sufficient and necessary condition for a = 0
theorem prop_B : ¬(∀ a : ℝ, (a^2 + a = 0 ↔ a = 0)) :=
sorry

-- Proposition C: a > 1 and b > 1 is a sufficient and necessary condition for a + b > 2 and ab > 1
theorem prop_C (a b : ℝ) : (a > 1 ∧ b > 1) ↔ (a + b > 2 ∧ a * b > 1) :=
sorry

-- Proposition D: a > 4 is a necessary and sufficient condition for the roots of the equation x² - ax + a = 0 to be all positive
theorem prop_D (a : ℝ) : (a > 4) ↔ (∀ x : ℝ, x ≠ 0 → (x^2 - a*x + a = 0 → x > 0)) :=
sorry

end NUMINAMATH_GPT_prop_A_prop_B_prop_C_prop_D_l1157_115771


namespace NUMINAMATH_GPT_smallest_n_for_factors_l1157_115778

theorem smallest_n_for_factors (k : ℕ) (hk : (∃ p : ℕ, k = 2^p) ) :
  ∃ (n : ℕ), ( 5^2 ∣ n * k * 36 * 343 ) ∧ ( 3^3 ∣ n * k * 36 * 343 ) ∧ n = 75 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_factors_l1157_115778


namespace NUMINAMATH_GPT_max_value_fraction_l1157_115764

theorem max_value_fraction : ∀ (x y : ℝ), (-5 ≤ x ∧ x ≤ -1) → (1 ≤ y ∧ y ≤ 3) → (1 + y / x ≤ -2) :=
  by
    intros x y hx hy
    sorry

end NUMINAMATH_GPT_max_value_fraction_l1157_115764


namespace NUMINAMATH_GPT_album_count_l1157_115769

def albums_total (A B K M C : ℕ) : Prop :=
  A = 30 ∧ B = A - 15 ∧ K = 6 * B ∧ M = 5 * K ∧ C = 3 * M ∧ (A + B + K + M + C) = 1935

theorem album_count (A B K M C : ℕ) : albums_total A B K M C :=
by
  sorry

end NUMINAMATH_GPT_album_count_l1157_115769


namespace NUMINAMATH_GPT_tropical_fish_count_l1157_115773

theorem tropical_fish_count (total_fish : ℕ) (koi_count : ℕ) (total_fish_eq : total_fish = 52) (koi_count_eq : koi_count = 37) : 
    (total_fish - koi_count) = 15 := by
    sorry

end NUMINAMATH_GPT_tropical_fish_count_l1157_115773


namespace NUMINAMATH_GPT_sequence_initial_term_l1157_115721

theorem sequence_initial_term (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n + 1) = a n + n)
  (h2 : a 61 = 2010) : a 1 = 180 :=
by
  sorry

end NUMINAMATH_GPT_sequence_initial_term_l1157_115721


namespace NUMINAMATH_GPT_bicycle_distance_l1157_115770

def distance : ℝ := 15

theorem bicycle_distance :
  ∀ (x y : ℝ),
  (x + 6) * (y - 5 / 60) = x * y →
  (x - 5) * (y + 6 / 60) = x * y →
  x * y = distance :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_bicycle_distance_l1157_115770


namespace NUMINAMATH_GPT_max_constant_inequality_l1157_115799

theorem max_constant_inequality (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha1 : a ≤ 1)
    (hb : 0 ≤ b) (hb1 : b ≤ 1)
    (hc : 0 ≤ c) (hc1 : c ≤ 1)
    (hd : 0 ≤ d) (hd1 : d ≤ 1) 
    : a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^3 + b^3 + c^3 + d^3) :=
sorry

end NUMINAMATH_GPT_max_constant_inequality_l1157_115799


namespace NUMINAMATH_GPT_sum_of_solutions_of_quadratic_l1157_115768

theorem sum_of_solutions_of_quadratic :
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  roots_sum = 3 / 2 :=
by
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  have h1 : roots_sum = 3 / 2 := by sorry
  exact h1

end NUMINAMATH_GPT_sum_of_solutions_of_quadratic_l1157_115768


namespace NUMINAMATH_GPT_x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l1157_115743

theorem x_plus_inv_x_eq_two_implies_x_pow_six_eq_one
  (x : ℝ) (h : x + 1/x = 2) : x^6 = 1 :=
sorry

end NUMINAMATH_GPT_x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l1157_115743


namespace NUMINAMATH_GPT_fraction_simplification_l1157_115751

theorem fraction_simplification (x : ℝ) (h: x ≠ 1) : (5 * x / (x - 1) - 5 / (x - 1)) = 5 := 
sorry

end NUMINAMATH_GPT_fraction_simplification_l1157_115751


namespace NUMINAMATH_GPT_charles_whistles_l1157_115738

theorem charles_whistles (S : ℕ) (C : ℕ) (h1 : S = 223) (h2 : S = C + 95) : C = 128 :=
by
  sorry

end NUMINAMATH_GPT_charles_whistles_l1157_115738


namespace NUMINAMATH_GPT_probability_X_eq_Y_correct_l1157_115705

noncomputable def probability_X_eq_Y : ℝ :=
  let lower_bound := -20 * Real.pi
  let upper_bound := 20 * Real.pi
  let total_pairs := (upper_bound - lower_bound) * (upper_bound - lower_bound)
  let matching_pairs := 81
  matching_pairs / total_pairs

theorem probability_X_eq_Y_correct :
  probability_X_eq_Y = 81 / 1681 :=
by
  unfold probability_X_eq_Y
  sorry

end NUMINAMATH_GPT_probability_X_eq_Y_correct_l1157_115705


namespace NUMINAMATH_GPT_probability_of_2_reds_before_3_greens_l1157_115704

theorem probability_of_2_reds_before_3_greens :
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  (favorable_arrangements / total_arrangements : ℚ) = (2 / 7 : ℚ) :=
by
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  have fraction_computation :
    (favorable_arrangements : ℚ) / (total_arrangements : ℚ) = (2 / 7 : ℚ)
  {
    sorry
  }
  exact fraction_computation

end NUMINAMATH_GPT_probability_of_2_reds_before_3_greens_l1157_115704


namespace NUMINAMATH_GPT_most_stable_performance_l1157_115708

theorem most_stable_performance 
    (s_A s_B s_C s_D : ℝ)
    (hA : s_A = 1.5)
    (hB : s_B = 2.6)
    (hC : s_C = 1.7)
    (hD : s_D = 2.8)
    (mean_score : ∀ (x : ℝ), x = 88.5) :
    s_A < s_C ∧ s_C < s_B ∧ s_B < s_D := by
  sorry

end NUMINAMATH_GPT_most_stable_performance_l1157_115708


namespace NUMINAMATH_GPT_total_guests_l1157_115798

-- Define the conditions.
def number_of_tables := 252.0
def guests_per_table := 4.0

-- Define the statement to prove.
theorem total_guests : number_of_tables * guests_per_table = 1008.0 := by
  sorry

end NUMINAMATH_GPT_total_guests_l1157_115798


namespace NUMINAMATH_GPT_initial_bottle_caps_l1157_115744

theorem initial_bottle_caps 
    (x : ℝ) 
    (Nancy_bottle_caps : ℝ) 
    (Marilyn_current_bottle_caps : ℝ) 
    (h1 : Nancy_bottle_caps = 36.0)
    (h2 : Marilyn_current_bottle_caps = 87)
    (h3 : x + Nancy_bottle_caps = Marilyn_current_bottle_caps) : 
    x = 51 := 
by 
  sorry

end NUMINAMATH_GPT_initial_bottle_caps_l1157_115744


namespace NUMINAMATH_GPT_length_of_each_piece_after_subdividing_l1157_115719

theorem length_of_each_piece_after_subdividing (total_length : ℝ) (num_initial_cuts : ℝ) (num_pieces_given : ℝ) (num_subdivisions : ℝ) (final_length : ℝ) : 
  total_length = 200 → 
  num_initial_cuts = 4 → 
  num_pieces_given = 2 → 
  num_subdivisions = 2 → 
  final_length = (total_length / num_initial_cuts / num_subdivisions) → 
  final_length = 25 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end NUMINAMATH_GPT_length_of_each_piece_after_subdividing_l1157_115719


namespace NUMINAMATH_GPT_greatest_two_digit_multiple_of_17_l1157_115752

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end NUMINAMATH_GPT_greatest_two_digit_multiple_of_17_l1157_115752


namespace NUMINAMATH_GPT_number_of_rows_l1157_115782

theorem number_of_rows (total_chairs : ℕ) (chairs_per_row : ℕ) (r : ℕ) 
  (h1 : total_chairs = 432) (h2 : chairs_per_row = 16) (h3 : total_chairs = chairs_per_row * r) : r = 27 :=
sorry

end NUMINAMATH_GPT_number_of_rows_l1157_115782


namespace NUMINAMATH_GPT_total_amount_is_70000_l1157_115783

-- Definitions based on the given conditions
def total_amount_divided (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  amount_10 + amount_20

def interest_earned (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  (amount_10 * 10 / 100) + (amount_20 * 20 / 100)

-- Statement to be proved
theorem total_amount_is_70000 (amount_10: ℕ) (amount_20: ℕ) (total_interest: ℕ) :
  amount_10 = 60000 →
  total_interest = 8000 →
  interest_earned amount_10 amount_20 = total_interest →
  total_amount_divided amount_10 amount_20 = 70000 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_amount_is_70000_l1157_115783


namespace NUMINAMATH_GPT_average_is_3_l1157_115765

theorem average_is_3 (A B C : ℝ) (h1 : 1501 * C - 3003 * A = 6006)
                              (h2 : 1501 * B + 4504 * A = 7507)
                              (h3 : A + B = 1) :
  (A + B + C) / 3 = 3 :=
by sorry

end NUMINAMATH_GPT_average_is_3_l1157_115765


namespace NUMINAMATH_GPT_area_difference_l1157_115706

theorem area_difference (r1 r2 : ℝ) (h1 : r1 = 30) (h2 : r2 = 15 / 2) :
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_area_difference_l1157_115706


namespace NUMINAMATH_GPT_girls_left_to_play_kho_kho_l1157_115753

theorem girls_left_to_play_kho_kho (B G x : ℕ) 
  (h_eq : B = G)
  (h_twice : B = 2 * (G - x))
  (h_total : B + G = 32) :
  x = 8 :=
by sorry

end NUMINAMATH_GPT_girls_left_to_play_kho_kho_l1157_115753


namespace NUMINAMATH_GPT_solution_exists_l1157_115731

theorem solution_exists (x : ℝ) :
  (|2 * x - 3| ≤ 3 ∧ (1 / x) < 1 ∧ x ≠ 0) ↔ (1 < x ∧ x ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_exists_l1157_115731


namespace NUMINAMATH_GPT_value_of_8b_l1157_115789

theorem value_of_8b (a b : ℝ) (h1 : 6 * a + 3 * b = 3) (h2 : b = 2 * a - 3) : 8 * b = -8 := by
  sorry

end NUMINAMATH_GPT_value_of_8b_l1157_115789


namespace NUMINAMATH_GPT_part1_part2_part3_l1157_115716

-- Define the necessary constants and functions as per conditions
variable (a : ℝ) (f : ℝ → ℝ)
variable (hpos : a > 0) (hfa : f a = 1)

-- Conditions based on the problem statement
variable (hodd : ∀ x, f (-x) = -f x)
variable (hfe : ∀ x1 x2, f (x1 - x2) = (f x1 * f x2 + 1) / (f x2 - f x1))

-- 1. Prove that f(2a) = 0
theorem part1  : f (2 * a) = 0 := sorry

-- 2. Prove that there exists a constant T > 0 such that f(x + T) = f(x)
theorem part2 : ∃ T > 0, ∀ x, f (x + 4 * a) = f x := sorry

-- 3. Prove f(x) is decreasing on (0, 4a) given x ∈ (0, 2a) implies f(x) > 0
theorem part3 (hx_correct : ∀ x, 0 < x ∧ x < 2 * a → 0 < f x) :
  ∀ x1 x2, 0 < x2 ∧ x2  < x1 ∧ x1 < 4 * a → f x2 > f x1 := sorry

end NUMINAMATH_GPT_part1_part2_part3_l1157_115716


namespace NUMINAMATH_GPT_parabola_focus_l1157_115725

theorem parabola_focus (y x : ℝ) (h : y^2 = 4 * x) : x = 1 → y = 0 → (1, 0) = (1, 0) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_focus_l1157_115725


namespace NUMINAMATH_GPT_set_intersection_example_l1157_115700

theorem set_intersection_example (A : Set ℕ) (B : Set ℕ) (hA : A = {1, 3, 5}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_example_l1157_115700


namespace NUMINAMATH_GPT_sum_of_possible_radii_l1157_115714

theorem sum_of_possible_radii :
  ∃ r1 r2 : ℝ, 
    (∀ r, (r - 5)^2 + r^2 = (r + 2)^2 → r = r1 ∨ r = r2) ∧ 
    r1 + r2 = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_radii_l1157_115714


namespace NUMINAMATH_GPT_sum_base7_l1157_115733

def base7_to_base10 (n : ℕ) : ℕ := 
  -- Function to convert base 7 to base 10 (implementation not shown)
  sorry

def base10_to_base7 (n : ℕ) : ℕ :=
  -- Function to convert base 10 to base 7 (implementation not shown)
  sorry

theorem sum_base7 (a b : ℕ) (ha : a = base7_to_base10 12) (hb : b = base7_to_base10 245) :
  base10_to_base7 (a + b) = 260 :=
sorry

end NUMINAMATH_GPT_sum_base7_l1157_115733


namespace NUMINAMATH_GPT_least_n_for_reducible_fraction_l1157_115777

theorem least_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, n - 13 = 71 * k) ∧ n = 84 := by
  sorry

end NUMINAMATH_GPT_least_n_for_reducible_fraction_l1157_115777


namespace NUMINAMATH_GPT_solve_for_p_l1157_115759

-- Conditions
def C1 (n : ℕ) : Prop := (3 : ℚ) / 4 = n / 48
def C2 (m n : ℕ) : Prop := (3 : ℚ) / 4 = (m + n) / 96
def C3 (p m : ℕ) : Prop := (3 : ℚ) / 4 = (p - m) / 160

-- Theorem to prove
theorem solve_for_p (n m p : ℕ) (h1 : C1 n) (h2 : C2 m n) (h3 : C3 p m) : p = 156 := 
by 
    sorry

end NUMINAMATH_GPT_solve_for_p_l1157_115759


namespace NUMINAMATH_GPT_gcd_of_polynomials_l1157_115703

/-- Given that a is an odd multiple of 7877, the greatest common divisor of
       7a^2 + 54a + 117 and 3a + 10 is 1. -/
theorem gcd_of_polynomials (a : ℤ) (h1 : a % 2 = 1) (h2 : 7877 ∣ a) :
  Int.gcd (7 * a ^ 2 + 54 * a + 117) (3 * a + 10) = 1 :=
sorry

end NUMINAMATH_GPT_gcd_of_polynomials_l1157_115703


namespace NUMINAMATH_GPT_fewer_people_third_bus_l1157_115707

noncomputable def people_first_bus : Nat := 12
noncomputable def people_second_bus : Nat := 2 * people_first_bus
noncomputable def people_fourth_bus : Nat := people_first_bus + 9
noncomputable def total_people : Nat := 75
noncomputable def people_other_buses : Nat := people_first_bus + people_second_bus + people_fourth_bus
noncomputable def people_third_bus : Nat := total_people - people_other_buses

theorem fewer_people_third_bus :
  people_second_bus - people_third_bus = 6 :=
by
  sorry

end NUMINAMATH_GPT_fewer_people_third_bus_l1157_115707


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_a5_product_l1157_115715

theorem arithmetic_sequence_a1_a5_product 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = 3) 
  (h_cond : (1 / a 1) + (1 / a 5) = 6 / 5) : 
  a 1 * a 5 = 5 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1_a5_product_l1157_115715


namespace NUMINAMATH_GPT_integral_evaluation_l1157_115723

noncomputable def integral_value : Real :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - (x - 1)^2) - x)

theorem integral_evaluation :
  integral_value = (Real.pi / 4) - 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_integral_evaluation_l1157_115723


namespace NUMINAMATH_GPT_normal_price_of_article_l1157_115735

theorem normal_price_of_article (P : ℝ) (sale_price : ℝ) (discount1 discount2 : ℝ) 
  (h1 : discount1 = 0.10) 
  (h2 : discount2 = 0.20) 
  (h3 : sale_price = 72) 
  (h4 : sale_price = (P * (1 - discount1)) * (1 - discount2)) : 
  P = 100 :=
by 
  sorry

end NUMINAMATH_GPT_normal_price_of_article_l1157_115735


namespace NUMINAMATH_GPT_Jed_cards_after_4_weeks_l1157_115729

theorem Jed_cards_after_4_weeks :
  (∀ n: ℕ, (if n % 2 = 0 then 20 + 4*n - 2*n else 20 + 4*n - 2*(n-1)) = 40) :=
by {
  sorry
}

end NUMINAMATH_GPT_Jed_cards_after_4_weeks_l1157_115729


namespace NUMINAMATH_GPT_num_units_from_batch_B_l1157_115726

theorem num_units_from_batch_B
  (A B C : ℝ) -- quantities of products from batches A, B, and C
  (h_arith_seq : B - A = C - B) -- batches A, B, and C form an arithmetic sequence
  (h_total : A + B + C = 240)    -- total units from three batches
  (h_sample_size : A + B + C = 60)  -- sample size drawn equals 60
  : B = 20 := 
by {
  sorry
}

end NUMINAMATH_GPT_num_units_from_batch_B_l1157_115726


namespace NUMINAMATH_GPT_n_cubed_plus_20n_div_48_l1157_115749

theorem n_cubed_plus_20n_div_48 (n : ℕ) (h_even : n % 2 = 0) : (n^3 + 20 * n) % 48 = 0 :=
sorry

end NUMINAMATH_GPT_n_cubed_plus_20n_div_48_l1157_115749


namespace NUMINAMATH_GPT_reciprocals_not_arithmetic_sequence_l1157_115794

theorem reciprocals_not_arithmetic_sequence 
  (a b c : ℝ) (h : 2 * b = a + c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_neq : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ¬ (1 / a + 1 / c = 2 / b) :=
by
  sorry

end NUMINAMATH_GPT_reciprocals_not_arithmetic_sequence_l1157_115794


namespace NUMINAMATH_GPT_Niklaus_walked_distance_l1157_115709

noncomputable def MilesToFeet (miles : ℕ) : ℕ := miles * 5280
noncomputable def YardsToFeet (yards : ℕ) : ℕ := yards * 3

theorem Niklaus_walked_distance (n_feet : ℕ) :
  MilesToFeet 4 + YardsToFeet 975 + n_feet = 25332 → n_feet = 1287 := by
  sorry

end NUMINAMATH_GPT_Niklaus_walked_distance_l1157_115709


namespace NUMINAMATH_GPT_no_real_solutions_quadratic_solve_quadratic_eq_l1157_115718

-- For Equation (1)

theorem no_real_solutions_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = -4 ∧ c = 5 ∧ (b^2 - 4 * a * c < 0)) :
  ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 := 
by
  sorry

-- For Equation (2)

theorem solve_quadratic_eq {x : ℝ} (h_eq : (x + 1) * (x + 2) = 2 * x + 4) :
  x = -2 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_quadratic_solve_quadratic_eq_l1157_115718


namespace NUMINAMATH_GPT_find_B_values_l1157_115791

theorem find_B_values (A B : ℤ) (h1 : 800 < A) (h2 : A < 1300) (h3 : B > 1) (h4 : A = B ^ 4) : B = 5 ∨ B = 6 := 
sorry

end NUMINAMATH_GPT_find_B_values_l1157_115791


namespace NUMINAMATH_GPT_sum_nat_numbers_from_1_to_5_l1157_115702

theorem sum_nat_numbers_from_1_to_5 : (1 + 2 + 3 + 4 + 5 = 15) :=
by
  sorry

end NUMINAMATH_GPT_sum_nat_numbers_from_1_to_5_l1157_115702


namespace NUMINAMATH_GPT_remainder_sum_products_l1157_115790

theorem remainder_sum_products (a b c d : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) 
  (hd : d % 7 = 6) : 
  ((a * b + c * d) % 7) = 1 :=
by sorry

end NUMINAMATH_GPT_remainder_sum_products_l1157_115790


namespace NUMINAMATH_GPT_smallest_positive_integer_l1157_115745

theorem smallest_positive_integer (
  a : ℕ
) : 
  (a ≡ 5 [MOD 6]) ∧ (a ≡ 7 [MOD 8]) → a = 23 :=
by sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1157_115745


namespace NUMINAMATH_GPT_find_nm_l1157_115736

theorem find_nm (h : 62^2 + 122^2 = 18728) : 
  ∃ (n m : ℕ), (n = 92 ∧ m = 30) ∨ (n = 30 ∧ m = 92) ∧ n^2 + m^2 = 9364 := 
by 
  sorry

end NUMINAMATH_GPT_find_nm_l1157_115736


namespace NUMINAMATH_GPT_probability_at_tree_correct_expected_distance_correct_l1157_115760

-- Define the initial conditions
def initial_tree (n : ℕ) : ℕ := n + 1
def total_trees (n : ℕ) : ℕ := 2 * n + 1

-- Define the probability that the drunkard is at each tree T_i (1 <= i <= 2n+1) at the end of the nth minute
def probability_at_tree (n i : ℕ) : ℚ :=
  if 1 ≤ i ∧ i ≤ total_trees n then
    (Nat.choose (2*n) (i-1)) / (2^(2*n))
  else
    0

-- Define the expected distance between the final position and the initial tree T_{n+1}
def expected_distance (n : ℕ) : ℚ :=
  n * (Nat.choose (2*n) n) / (2^(2*n))

-- Statements to prove
theorem probability_at_tree_correct (n i : ℕ) (hi : 1 ≤ i ∧ i ≤ total_trees n)  :
  probability_at_tree n i = (Nat.choose (2*n) (i-1)) / (2^(2*n)) :=
by
  sorry

theorem expected_distance_correct (n : ℕ) :
  expected_distance n = n * (Nat.choose (2*n) n) / (2^(2*n)) :=
by
  sorry

end NUMINAMATH_GPT_probability_at_tree_correct_expected_distance_correct_l1157_115760


namespace NUMINAMATH_GPT_michael_ratio_l1157_115711

-- Definitions
def Michael_initial := 42
def Brother_initial := 17

-- Conditions
def Brother_after_candy_purchase := 35
def Candy_cost := 3
def Brother_before_candy := Brother_after_candy_purchase + Candy_cost
def x := Brother_before_candy - Brother_initial

-- Prove the ratio of the money Michael gave to his brother to his initial amount is 1:2
theorem michael_ratio :
  x * 2 = Michael_initial := by
  sorry

end NUMINAMATH_GPT_michael_ratio_l1157_115711


namespace NUMINAMATH_GPT_inequality_solution_set_l1157_115732

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 4)^2

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 1) / (x + 4)^2 ≥ 0} = {x : ℝ | x ≠ -4} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1157_115732


namespace NUMINAMATH_GPT_smallest_x_l1157_115727

theorem smallest_x (x : ℕ) (h900 : 900 = 2^2 * 3^2 * 5^2) (h1152 : 1152 = 2^7 * 3^2) : 
  (900 * x) % 1152 = 0 ↔ x = 32 := 
by
  sorry

end NUMINAMATH_GPT_smallest_x_l1157_115727


namespace NUMINAMATH_GPT_part1_part2_l1157_115761

variables {a_n b_n : ℕ → ℤ} {k m : ℕ}

-- Part 1: Arithmetic Sequence
axiom a2_eq_3 : a_n 2 = 3
axiom S5_eq_25 : (5 * (2 * (a_n 1 + 2 * (a_n 1 + 1)) / 2)) = 25

-- Part 2: Geometric Sequence
axiom b1_eq_1 : b_n 1 = 1
axiom q_eq_3 : ∀ n, b_n n = 3^(n-1)

noncomputable def arithmetic_seq (n : ℕ) : ℤ :=
  2 * n - 1

theorem part1 : (a_n 2 + a_n 4) / 2 = 5 :=
  sorry

theorem part2 (k : ℕ) (hk : 0 < k) : ∃ m, b_n k = arithmetic_seq m ∧ m = (3^(k-1) + 1) / 2 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1157_115761


namespace NUMINAMATH_GPT_add_neg_two_l1157_115779

theorem add_neg_two : 1 + (-2 : ℚ) = -1 := by
  sorry

end NUMINAMATH_GPT_add_neg_two_l1157_115779
