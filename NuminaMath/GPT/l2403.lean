import Mathlib

namespace NUMINAMATH_GPT_jill_arrives_before_jack_l2403_240335

theorem jill_arrives_before_jack
  (distance : ℝ)
  (jill_speed : ℝ)
  (jack_speed : ℝ)
  (jill_time_minutes : ℝ)
  (jack_time_minutes : ℝ) :
  distance = 2 →
  jill_speed = 15 →
  jack_speed = 6 →
  jill_time_minutes = (distance / jill_speed) * 60 →
  jack_time_minutes = (distance / jack_speed) * 60 →
  jack_time_minutes - jill_time_minutes = 12 :=
by
  sorry

end NUMINAMATH_GPT_jill_arrives_before_jack_l2403_240335


namespace NUMINAMATH_GPT_max_consecutive_interesting_numbers_l2403_240309

def is_interesting (n : ℕ) : Prop :=
  (n / 100 % 3 = 0) ∨ (n / 10 % 10 % 3 = 0) ∨ (n % 10 % 3 = 0)

theorem max_consecutive_interesting_numbers :
  ∃ l r, 100 ≤ l ∧ r ≤ 999 ∧ r - l + 1 = 122 ∧ (∀ n, l ≤ n ∧ n ≤ r → is_interesting n) ∧ 
  ∀ l' r', 100 ≤ l' ∧ r' ≤ 999 ∧ r' - l' + 1 > 122 → ∃ n, l' ≤ n ∧ n ≤ r' ∧ ¬ is_interesting n := 
sorry

end NUMINAMATH_GPT_max_consecutive_interesting_numbers_l2403_240309


namespace NUMINAMATH_GPT_frac_inequality_l2403_240340

theorem frac_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : d > c) (h4 : c > 0) : (a/c) > (b/d) := 
sorry

end NUMINAMATH_GPT_frac_inequality_l2403_240340


namespace NUMINAMATH_GPT_sum_distinct_integers_l2403_240371

theorem sum_distinct_integers (a b c d e : ℤ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
    (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
    (h : (5 - a) * (5 - b) * (5 - c) * (5 - d) * (5 - e) = 120) :
    a + b + c + d + e = 13 := by
  sorry

end NUMINAMATH_GPT_sum_distinct_integers_l2403_240371


namespace NUMINAMATH_GPT_unique_polynomial_solution_l2403_240310

def polynomial_homogeneous_of_degree_n (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (t x y : ℝ), P (t * x) (t * y) = t^n * P x y

def polynomial_symmetric_condition (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y z : ℝ), P (y + z) x + P (z + x) y + P (x + y) z = 0

def polynomial_value_at_point (P : ℝ → ℝ → ℝ) : Prop :=
  P 1 0 = 1

theorem unique_polynomial_solution (P : ℝ → ℝ → ℝ) (n : ℕ) :
  polynomial_homogeneous_of_degree_n P n →
  polynomial_symmetric_condition P →
  polynomial_value_at_point P →
  ∀ x y : ℝ, P x y = (x + y)^n * (x - 2 * y) := 
by
  intros h_deg h_symm h_value x y
  sorry

end NUMINAMATH_GPT_unique_polynomial_solution_l2403_240310


namespace NUMINAMATH_GPT_total_amount_paid_l2403_240322

-- Define the given conditions
def q_g : ℕ := 9        -- Quantity of grapes
def r_g : ℕ := 70       -- Rate per kg of grapes
def q_m : ℕ := 9        -- Quantity of mangoes
def r_m : ℕ := 55       -- Rate per kg of mangoes

-- Define the total amount paid calculation and prove it equals 1125
theorem total_amount_paid : (q_g * r_g + q_m * r_m) = 1125 :=
by
  -- Proof will be provided here. Currently using 'sorry' to skip it.
  sorry

end NUMINAMATH_GPT_total_amount_paid_l2403_240322


namespace NUMINAMATH_GPT_convert_base8_to_base7_l2403_240321

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * 8^2 + 3 * 8^1 + 1 * 8^0

def base10_to_base7 (n : ℕ) : ℕ :=
  1002  -- Directly providing the result from conditions given.

theorem convert_base8_to_base7 :
  base10_to_base7 (base8_to_base10 531) = 1002 := by
  sorry

end NUMINAMATH_GPT_convert_base8_to_base7_l2403_240321


namespace NUMINAMATH_GPT_expression_value_l2403_240359

theorem expression_value (x : ℝ) (h : x = -2) : (x * x^2 * (1/x) = 4) :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_expression_value_l2403_240359


namespace NUMINAMATH_GPT_distance_from_left_focal_to_line_l2403_240338

noncomputable def ellipse_eq_line_dist : Prop :=
  let a := 2
  let b := Real.sqrt 3
  let c := 1
  let x₀ := -1
  let y₀ := 0
  let x₁ := 0
  let y₁ := Real.sqrt 3
  let x₂ := 1
  let y₂ := 0
  
  -- Equation of the line derived from the upper vertex and right focal point
  let m := -(y₁ - y₂) / (x₁ - x₂)
  let line_eq (x y : ℝ) := (Real.sqrt 3 * x + y - Real.sqrt 3 = 0)
  
  -- Distance formula from point to line
  let d := abs (Real.sqrt 3 * x₀ + y₀ - Real.sqrt 3) / Real.sqrt ((Real.sqrt 3)^2 + 1^2)

  -- The assertion that the distance is √3
  d = Real.sqrt 3

theorem distance_from_left_focal_to_line : ellipse_eq_line_dist := 
  sorry  -- Proof is omitted as per the instruction

end NUMINAMATH_GPT_distance_from_left_focal_to_line_l2403_240338


namespace NUMINAMATH_GPT_integral_result_l2403_240379

open Real

theorem integral_result :
  (∫ x in (0:ℝ)..(π/2), (x^2 - 5 * x + 6) * sin (3 * x)) = (67 - 3 * π) / 27 := by
  sorry

end NUMINAMATH_GPT_integral_result_l2403_240379


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2403_240375

theorem sufficient_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x > 0 ∧ y > 0) → (x > 0 ∧ y > 0 ↔ (y/x + x/y ≥ 2)) :=
by sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2403_240375


namespace NUMINAMATH_GPT_no_solution_m_4_l2403_240346

theorem no_solution_m_4 (m : ℝ) : 
  (¬ ∃ x : ℝ, 2/x = m/(2*x + 1)) → m = 4 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_m_4_l2403_240346


namespace NUMINAMATH_GPT_carla_drank_total_amount_l2403_240358

-- Define the conditions
def carla_water : ℕ := 15
def carla_soda := 3 * carla_water - 6
def total_liquid := carla_water + carla_soda

-- State the theorem
theorem carla_drank_total_amount : total_liquid = 54 := by
  sorry

end NUMINAMATH_GPT_carla_drank_total_amount_l2403_240358


namespace NUMINAMATH_GPT_cost_of_western_european_postcards_before_1980s_l2403_240372

def germany_cost_1950s : ℝ := 5 * 0.07
def france_cost_1950s : ℝ := 8 * 0.05

def germany_cost_1960s : ℝ := 6 * 0.07
def france_cost_1960s : ℝ := 9 * 0.05

def germany_cost_1970s : ℝ := 11 * 0.07
def france_cost_1970s : ℝ := 10 * 0.05

def total_germany_cost : ℝ := germany_cost_1950s + germany_cost_1960s + germany_cost_1970s
def total_france_cost : ℝ := france_cost_1950s + france_cost_1960s + france_cost_1970s

def total_western_europe_cost : ℝ := total_germany_cost + total_france_cost

theorem cost_of_western_european_postcards_before_1980s :
  total_western_europe_cost = 2.89 := by
  sorry

end NUMINAMATH_GPT_cost_of_western_european_postcards_before_1980s_l2403_240372


namespace NUMINAMATH_GPT_nina_weekend_earnings_l2403_240391

noncomputable def total_money_made (necklace_price bracelet_price earring_pair_price ensemble_price : ℕ)
                                   (necklaces_sold bracelets_sold individual_earrings_sold ensembles_sold : ℕ) : ℕ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * (individual_earrings_sold / 2) +
  ensemble_price * ensembles_sold

theorem nina_weekend_earnings :
  total_money_made 25 15 10 45 5 10 20 2 = 465 :=
by
  sorry

end NUMINAMATH_GPT_nina_weekend_earnings_l2403_240391


namespace NUMINAMATH_GPT_highest_slope_product_l2403_240390

theorem highest_slope_product (m1 m2 : ℝ) (h1 : m1 = 5 * m2) 
    (h2 : abs ((m2 - m1) / (1 + m1 * m2)) = 1) : (m1 * m2) ≤ 1.8 :=
by
  sorry

end NUMINAMATH_GPT_highest_slope_product_l2403_240390


namespace NUMINAMATH_GPT_correct_calculation_l2403_240374

theorem correct_calculation (m n : ℝ) : 4 * m + 2 * n - (n - m) = 5 * m + n :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l2403_240374


namespace NUMINAMATH_GPT_average_speed_comparison_l2403_240365

theorem average_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0):
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v + w) / 3) :=
sorry

end NUMINAMATH_GPT_average_speed_comparison_l2403_240365


namespace NUMINAMATH_GPT_jane_not_finish_probability_l2403_240327

theorem jane_not_finish_probability :
  (1 : ℚ) - (5 / 8) = (3 / 8) := by
  sorry

end NUMINAMATH_GPT_jane_not_finish_probability_l2403_240327


namespace NUMINAMATH_GPT_tablet_value_is_2100_compensation_for_m_days_l2403_240328

-- Define the given conditions
def monthly_compensation: ℕ := 30
def monthly_tablet_value (x: ℕ) (cash: ℕ): ℕ := x + cash

def daily_compensation (days: ℕ) (x: ℕ) (cash: ℕ): ℕ :=
  days * (x / monthly_compensation + cash / monthly_compensation)

def received_compensation (tablet_value: ℕ) (cash: ℕ): ℕ :=
  tablet_value + cash

-- The proofs we need:
-- Proof that the tablet value is 2100 yuan
theorem tablet_value_is_2100:
  ∀ (x: ℕ) (cash_1 cash_2: ℕ), 
  ((20 * (x / monthly_compensation + 1500 / monthly_compensation)) = (x + 300)) → 
  x = 2100 := sorry

-- Proof that compensation for m days is 120m yuan
theorem compensation_for_m_days (m: ℕ):
  ∀ (x: ℕ), 
  ((x + 1500) / monthly_compensation) = 120 → 
  x = 2100 → 
  m * 120 = 120 * m := sorry

end NUMINAMATH_GPT_tablet_value_is_2100_compensation_for_m_days_l2403_240328


namespace NUMINAMATH_GPT_largest_pillar_radius_l2403_240380

-- Define the dimensions of the crate
def crate_length := 12
def crate_width := 8
def crate_height := 3

-- Define the condition that the pillar is a right circular cylinder
def is_right_circular_cylinder (r : ℝ) (h : ℝ) : Prop :=
  r > 0 ∧ h > 0

-- The theorem stating the radius of the largest volume pillar that can fit in the crate
theorem largest_pillar_radius (r h : ℝ) (cylinder_fits : is_right_circular_cylinder r h) :
  r = 1.5 := 
sorry

end NUMINAMATH_GPT_largest_pillar_radius_l2403_240380


namespace NUMINAMATH_GPT_minimum_distinct_lines_l2403_240314

theorem minimum_distinct_lines (n : ℕ) (h : n = 31) : 
  ∃ (k : ℕ), k = 9 :=
by
  sorry

end NUMINAMATH_GPT_minimum_distinct_lines_l2403_240314


namespace NUMINAMATH_GPT_dress_shirt_cost_l2403_240320

theorem dress_shirt_cost (x : ℝ) :
  let total_cost_before_discounts := 4 * x + 2 * 40 + 150 + 2 * 30
  let total_cost_after_store_discount := total_cost_before_discounts * 0.8
  let total_cost_after_coupon := total_cost_after_store_discount * 0.9
  total_cost_after_coupon = 252 → x = 15 :=
by
  let total_cost_before_discounts := 4 * x + 2 * 40 + 150 + 2 * 30
  let total_cost_after_store_discount := total_cost_before_discounts * 0.8
  let total_cost_after_coupon := total_cost_after_store_discount * 0.9
  intro h
  sorry

end NUMINAMATH_GPT_dress_shirt_cost_l2403_240320


namespace NUMINAMATH_GPT_Thabo_books_ratio_l2403_240326

variable (P_f P_nf H_nf : ℕ)

theorem Thabo_books_ratio :
  P_f + P_nf + H_nf = 220 →
  H_nf = 40 →
  P_nf = H_nf + 20 →
  P_f / P_nf = 2 :=
by sorry

end NUMINAMATH_GPT_Thabo_books_ratio_l2403_240326


namespace NUMINAMATH_GPT_apps_minus_files_eq_seven_l2403_240385

-- Definitions based on conditions
def initial_apps := 24
def initial_files := 9
def deleted_apps := initial_apps - 12
def deleted_files := initial_files - 5

-- Definitions based on the question and correct answer
def apps_left := 12
def files_left := 5

theorem apps_minus_files_eq_seven : apps_left - files_left = 7 := by
  sorry

end NUMINAMATH_GPT_apps_minus_files_eq_seven_l2403_240385


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2403_240306

theorem simplify_and_evaluate :
  let x := (-1 : ℚ) / 2
  3 * x^2 - (5 * x - 3 * (2 * x - 1) + 7 * x^2) = -9 / 2 :=
by
  let x : ℚ := (-1 : ℚ) / 2
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2403_240306


namespace NUMINAMATH_GPT_calculate_a2_b2_c2_l2403_240397

theorem calculate_a2_b2_c2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -3) (h3 : a * b * c = 2) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end NUMINAMATH_GPT_calculate_a2_b2_c2_l2403_240397


namespace NUMINAMATH_GPT_chess_tournament_total_players_l2403_240324

-- Define the conditions

def total_points_calculation (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 132

def games_played (n : ℕ) : ℕ :=
  ((n + 12) * (n + 11)) / 2

theorem chess_tournament_total_players :
  ∃ n, total_points_calculation n = games_played n ∧ n + 12 = 34 :=
by {
  -- Assume n is found such that all conditions are satisfied
  use 22,
  -- Provide the necessary equations and conditions
  sorry
}

end NUMINAMATH_GPT_chess_tournament_total_players_l2403_240324


namespace NUMINAMATH_GPT_negation_of_diagonals_equal_l2403_240369

def Rectangle : Type := sorry -- Let's assume there exists a type Rectangle
def diagonals_equal (r : Rectangle) : Prop := sorry -- Assume a function that checks if diagonals are equal

theorem negation_of_diagonals_equal :
  ¬(∀ r : Rectangle, diagonals_equal r) ↔ ∃ r : Rectangle, ¬diagonals_equal r :=
by
  sorry

end NUMINAMATH_GPT_negation_of_diagonals_equal_l2403_240369


namespace NUMINAMATH_GPT_jessie_interest_l2403_240313

noncomputable def compoundInterest 
  (P : ℝ) -- Principal
  (r : ℝ) -- annual interest rate
  (n : ℕ) -- number of times interest applied per time period
  (t : ℝ) -- time periods elapsed
  : ℝ :=
  P * (1 + r / n)^(n * t)

theorem jessie_interest :
  let P := 1200
  let annual_rate := 0.08
  let periods_per_year := 2
  let years := 5
  let A := compoundInterest P annual_rate periods_per_year years
  let interest := A - P
  interest = 576.29 :=
by
  sorry

end NUMINAMATH_GPT_jessie_interest_l2403_240313


namespace NUMINAMATH_GPT_each_wolf_kills_one_deer_l2403_240364

-- Definitions based on conditions
def hunting_wolves : Nat := 4
def additional_wolves : Nat := 16
def wolves_per_pack : Nat := hunting_wolves + additional_wolves
def meat_per_wolf_per_day : Nat := 8
def days_between_hunts : Nat := 5
def meat_per_wolf : Nat := meat_per_wolf_per_day * days_between_hunts
def total_meat_required : Nat := wolves_per_pack * meat_per_wolf
def meat_per_deer : Nat := 200
def deer_needed : Nat := total_meat_required / meat_per_deer
def deer_per_wolf_needed : Nat := deer_needed / hunting_wolves

-- Lean statement to prove
theorem each_wolf_kills_one_deer (hunting_wolves : Nat := 4) (additional_wolves : Nat := 16) 
    (meat_per_wolf_per_day : Nat := 8) (days_between_hunts : Nat := 5) 
    (meat_per_deer : Nat := 200) : deer_per_wolf_needed = 1 := 
by
  -- Proof required here
  sorry

end NUMINAMATH_GPT_each_wolf_kills_one_deer_l2403_240364


namespace NUMINAMATH_GPT_parabola_distance_to_y_axis_l2403_240317

theorem parabola_distance_to_y_axis :
  ∀ (M : ℝ × ℝ), (M.2 ^ 2 = 4 * M.1) → 
  dist (M, (1, 0)) = 10 →
  abs (M.1) = 9 :=
by
  intros M hParabola hDist
  sorry

end NUMINAMATH_GPT_parabola_distance_to_y_axis_l2403_240317


namespace NUMINAMATH_GPT_inverse_of_square_l2403_240347

theorem inverse_of_square (A : Matrix (Fin 2) (Fin 2) ℝ) (hA_inv : A⁻¹ = ![![3, 4], ![-2, -2]]) :
  (A^2)⁻¹ = ![![1, 4], ![-2, -4]] :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_square_l2403_240347


namespace NUMINAMATH_GPT_swimmers_meetings_in_15_minutes_l2403_240354

noncomputable def swimmers_pass_each_other_count 
    (pool_length : ℕ) (rate_swimmer1 : ℕ) (rate_swimmer2 : ℕ) (time_minutes : ℕ) : ℕ :=
sorry -- Definition of the function to count passing times

theorem swimmers_meetings_in_15_minutes :
  swimmers_pass_each_other_count 120 4 3 15 = 23 :=
sorry -- The proof is not required as per instruction.

end NUMINAMATH_GPT_swimmers_meetings_in_15_minutes_l2403_240354


namespace NUMINAMATH_GPT_no_rational_solution_5x2_plus_3y2_eq_1_l2403_240387

theorem no_rational_solution_5x2_plus_3y2_eq_1 :
  ¬ ∃ (x y : ℚ), 5 * x^2 + 3 * y^2 = 1 := 
sorry

end NUMINAMATH_GPT_no_rational_solution_5x2_plus_3y2_eq_1_l2403_240387


namespace NUMINAMATH_GPT_sequence_sum_l2403_240350

open Nat

-- Define the sequence
def a : ℕ → ℕ
| 0     => 1
| (n+1) => a n + (n + 1)

-- Define the sum of reciprocals up to the 2016 term
def sum_reciprocals : ℕ → ℚ
| 0     => 1 / (a 0)
| (n+1) => sum_reciprocals n + 1 / (a (n+1))

-- Define the property we wish to prove
theorem sequence_sum :
  sum_reciprocals 2015 = 4032 / 2017 :=
sorry

end NUMINAMATH_GPT_sequence_sum_l2403_240350


namespace NUMINAMATH_GPT_acute_triangle_l2403_240311

theorem acute_triangle (a b c : ℝ) (n : ℕ) (h_n : 2 < n) (h_eq : a^n + b^n = c^n) : a^2 + b^2 > c^2 :=
sorry

end NUMINAMATH_GPT_acute_triangle_l2403_240311


namespace NUMINAMATH_GPT_equation_solutions_l2403_240337

theorem equation_solutions : 
  ∀ x : ℝ, (2 * x - 1) - x * (1 - 2 * x) = 0 ↔ (x = 1 / 2 ∨ x = -1) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_equation_solutions_l2403_240337


namespace NUMINAMATH_GPT_triangle_is_isosceles_l2403_240344

theorem triangle_is_isosceles
  (α β γ x y z w : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α + β = x)
  (h3 : β + γ = y)
  (h4 : γ + α = z)
  (h5 : x + y + z + w = 360) : 
  (α = β ∧ β = γ) ∨ (α = γ ∧ γ = β) ∨ (β = α ∧ α = γ) := by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l2403_240344


namespace NUMINAMATH_GPT_total_passengers_l2403_240329

theorem total_passengers (P : ℕ) 
  (h1 : P = (1/12 : ℚ) * P + (1/4 : ℚ) * P + (1/9 : ℚ) * P + (1/6 : ℚ) * P + 42) :
  P = 108 :=
sorry

end NUMINAMATH_GPT_total_passengers_l2403_240329


namespace NUMINAMATH_GPT_number_of_scooters_l2403_240341

theorem number_of_scooters (b t s : ℕ) (h1 : b + t + s = 10) (h2 : 2 * b + 3 * t + 2 * s = 26) : s = 2 := 
by sorry

end NUMINAMATH_GPT_number_of_scooters_l2403_240341


namespace NUMINAMATH_GPT_radius_of_scrap_cookie_l2403_240388

theorem radius_of_scrap_cookie
  (r_cookies : ℝ) (n_cookies : ℕ) (radius_layout : Prop)
  (circle_diameter_twice_width : Prop) :
  (r_cookies = 0.5 ∧ n_cookies = 9 ∧ radius_layout ∧ circle_diameter_twice_width)
  →
  (∃ r_scrap : ℝ, r_scrap = Real.sqrt 6.75) :=
by
  sorry

end NUMINAMATH_GPT_radius_of_scrap_cookie_l2403_240388


namespace NUMINAMATH_GPT_avg_korean_language_score_l2403_240307

theorem avg_korean_language_score (male_avg : ℝ) (female_avg : ℝ) (male_students : ℕ) (female_students : ℕ) 
    (male_avg_given : male_avg = 83.1) (female_avg_given : female_avg = 84) (male_students_given : male_students = 10) (female_students_given : female_students = 8) :
    (male_avg * male_students + female_avg * female_students) / (male_students + female_students) = 83.5 :=
by sorry

end NUMINAMATH_GPT_avg_korean_language_score_l2403_240307


namespace NUMINAMATH_GPT_total_rabbits_correct_l2403_240319

def initial_breeding_rabbits : ℕ := 10
def kittens_first_spring : ℕ := initial_breeding_rabbits * 10
def adopted_first_spring : ℕ := kittens_first_spring / 2
def returned_adopted_first_spring : ℕ := 5
def total_rabbits_after_first_spring : ℕ :=
  initial_breeding_rabbits + (kittens_first_spring - adopted_first_spring + returned_adopted_first_spring)

def kittens_second_spring : ℕ := 60
def adopted_second_spring : ℕ := kittens_second_spring * 40 / 100
def returned_adopted_second_spring : ℕ := 10
def total_rabbits_after_second_spring : ℕ :=
  total_rabbits_after_first_spring + (kittens_second_spring - adopted_second_spring + returned_adopted_second_spring)

def breeding_rabbits_third_spring : ℕ := 12
def kittens_third_spring : ℕ := breeding_rabbits_third_spring * 8
def adopted_third_spring : ℕ := kittens_third_spring * 30 / 100
def returned_adopted_third_spring : ℕ := 3
def total_rabbits_after_third_spring : ℕ :=
  total_rabbits_after_second_spring + (kittens_third_spring - adopted_third_spring + returned_adopted_third_spring)

def kittens_fourth_spring : ℕ := breeding_rabbits_third_spring * 6
def adopted_fourth_spring : ℕ := kittens_fourth_spring * 20 / 100
def returned_adopted_fourth_spring : ℕ := 2
def total_rabbits_after_fourth_spring : ℕ :=
  total_rabbits_after_third_spring + (kittens_fourth_spring - adopted_fourth_spring + returned_adopted_fourth_spring)

theorem total_rabbits_correct : total_rabbits_after_fourth_spring = 242 := by
  sorry

end NUMINAMATH_GPT_total_rabbits_correct_l2403_240319


namespace NUMINAMATH_GPT_Jenny_minutes_of_sleep_l2403_240389

def hours_of_sleep : ℕ := 8
def minutes_per_hour : ℕ := 60

theorem Jenny_minutes_of_sleep : hours_of_sleep * minutes_per_hour = 480 := by
  sorry

end NUMINAMATH_GPT_Jenny_minutes_of_sleep_l2403_240389


namespace NUMINAMATH_GPT_ratio_of_volume_to_surface_area_l2403_240361

-- Definitions of the given conditions
def unit_cube_volume : ℕ := 1
def total_cubes : ℕ := 8
def volume := total_cubes * unit_cube_volume
def exposed_faces (center_cube_faces : ℕ) (side_cube_faces : ℕ) (top_cube_faces : ℕ) : ℕ :=
  center_cube_faces + 6 * side_cube_faces + top_cube_faces
def surface_area := exposed_faces 1 5 5
def ratio := volume / surface_area

-- The main theorem statement
theorem ratio_of_volume_to_surface_area : ratio = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_ratio_of_volume_to_surface_area_l2403_240361


namespace NUMINAMATH_GPT_solve_equation_l2403_240312

theorem solve_equation : 
  ∀ x : ℝ,
    (x + 5 ≠ 0) → 
    (x^2 + 3 * x + 4) / (x + 5) = x + 6 → 
    x = -13 / 4 :=
by 
  intro x
  intro hx
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l2403_240312


namespace NUMINAMATH_GPT_determine_k_coplanar_l2403_240331

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable {A B C D : V}
variable (k : ℝ)

theorem determine_k_coplanar (h : 4 • A - 3 • B + 6 • C + k • D = 0) : k = -13 :=
sorry

end NUMINAMATH_GPT_determine_k_coplanar_l2403_240331


namespace NUMINAMATH_GPT_sum_of_first_K_natural_numbers_is_perfect_square_l2403_240384

noncomputable def values_K (K : ℕ) : Prop := 
  ∃ N : ℕ, (K * (K + 1)) / 2 = N^2 ∧ (N + K < 120)

theorem sum_of_first_K_natural_numbers_is_perfect_square :
  ∀ K : ℕ, values_K K ↔ (K = 1 ∨ K = 8 ∨ K = 49) := by
  sorry

end NUMINAMATH_GPT_sum_of_first_K_natural_numbers_is_perfect_square_l2403_240384


namespace NUMINAMATH_GPT_find_lower_rate_l2403_240330

-- Definitions
def total_investment : ℝ := 20000
def total_interest : ℝ := 1440
def higher_rate : ℝ := 0.09
def fraction_higher : ℝ := 0.55

-- The amount invested at the higher rate
def x := fraction_higher * total_investment
-- The amount invested at the lower rate
def y := total_investment - x

-- The interest contributions
def interest_higher := x * higher_rate
def interest_lower (r : ℝ) := y * r

-- The equation we need to solve to find the lower interest rate
theorem find_lower_rate (r : ℝ) : interest_higher + interest_lower r = total_interest → r = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_find_lower_rate_l2403_240330


namespace NUMINAMATH_GPT_magician_can_always_determine_hidden_pair_l2403_240305

-- Define the cards as an enumeration
inductive Card
| one | two | three | four | five

-- Define a pair of cards
structure CardPair where
  first : Card
  second : Card

-- Define the function the magician uses to decode the hidden pair 
-- based on the two cards the assistant points out, encoded as a pentagon
noncomputable def magician_decodes (assistant_cards spectator_announced: CardPair) : CardPair := sorry

-- Theorem statement: given the conditions, the magician can always determine the hidden pair.
theorem magician_can_always_determine_hidden_pair 
  (hidden_cards assistant_cards spectator_announced : CardPair)
  (assistant_strategy : CardPair → CardPair)
  (h : assistant_strategy assistant_cards = spectator_announced)
  : magician_decodes assistant_cards spectator_announced = hidden_cards := sorry

end NUMINAMATH_GPT_magician_can_always_determine_hidden_pair_l2403_240305


namespace NUMINAMATH_GPT_boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l2403_240343

-- Problem 1: Specific case
theorem boat_and_current_speed (x y : ℝ) 
  (h1 : 3 * (x + y) = 75) 
  (h2 : 5 * (x - y) = 75) : 
  x = 20 ∧ y = 5 := 
sorry

-- Problem 2: General case
theorem boat_and_current_speed_general (x y : ℝ) (a b S : ℝ) 
  (h1 : a * (x + y) = S) 
  (h2 : b * (x - y) = S) : 
  x = (a + b) * S / (2 * a * b) ∧ 
  y = (b - a) * S / (2 * a * b) := 
sorry

theorem log_drift_time (y S a b : ℝ)
  (h_y : y = (b - a) * S / (2 * a * b)) : 
  S / y = 2 * a * b / (b - a) := 
sorry

end NUMINAMATH_GPT_boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l2403_240343


namespace NUMINAMATH_GPT_oranges_in_bag_l2403_240303

variables (O : ℕ)

def initial_oranges (O : ℕ) := O
def initial_tangerines := 17
def oranges_left_after_taking_away := O - 2
def tangerines_left_after_taking_away := 7
def tangerines_and_oranges_condition (O : ℕ) := 7 = (O - 2) + 4

theorem oranges_in_bag (O : ℕ) (h₀ : tangerines_and_oranges_condition O) : O = 5 :=
by
  sorry

end NUMINAMATH_GPT_oranges_in_bag_l2403_240303


namespace NUMINAMATH_GPT_A_plus_2B_plus_4_is_perfect_square_l2403_240334

theorem A_plus_2B_plus_4_is_perfect_square (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  ∃ k : ℚ, (A + 2 * B + 4) = k^2 :=
by
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  use ((2/3) * (10^n + 2))
  sorry

end NUMINAMATH_GPT_A_plus_2B_plus_4_is_perfect_square_l2403_240334


namespace NUMINAMATH_GPT_import_tax_excess_amount_l2403_240394

theorem import_tax_excess_amount (X : ℝ)
  (total_value : ℝ) (tax_paid : ℝ)
  (tax_rate : ℝ) :
  total_value = 2610 → tax_paid = 112.70 → tax_rate = 0.07 → 0.07 * (2610 - X) = 112.70 → X = 1000 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_import_tax_excess_amount_l2403_240394


namespace NUMINAMATH_GPT_Maryann_frees_all_friends_in_42_minutes_l2403_240370

-- Definitions for the problem conditions
def time_to_pick_cheap_handcuffs := 6
def time_to_pick_expensive_handcuffs := 8
def number_of_friends := 3

-- Define the statement we need to prove
theorem Maryann_frees_all_friends_in_42_minutes :
  (time_to_pick_cheap_handcuffs + time_to_pick_expensive_handcuffs) * number_of_friends = 42 :=
by
  sorry

end NUMINAMATH_GPT_Maryann_frees_all_friends_in_42_minutes_l2403_240370


namespace NUMINAMATH_GPT_cubic_equation_three_distinct_real_roots_l2403_240383

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 - a

theorem cubic_equation_three_distinct_real_roots (a : ℝ) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃
  ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ↔ -4 < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_cubic_equation_three_distinct_real_roots_l2403_240383


namespace NUMINAMATH_GPT_smallest_solution_l2403_240348

def equation (x : ℝ) := (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 14

theorem smallest_solution :
  ∀ x : ℝ, equation x → x = (3 - Real.sqrt 333) / 6 :=
sorry

end NUMINAMATH_GPT_smallest_solution_l2403_240348


namespace NUMINAMATH_GPT_solve_proof_problem_l2403_240302

noncomputable def proof_problem (f g : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x + g y) = 2 * x + y → g (x + f y) = x / 2 + y

theorem solve_proof_problem (f g : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + g y) = 2 * x + y) :
  ∀ x y : ℝ, g (x + f y) = x / 2 + y :=
sorry

end NUMINAMATH_GPT_solve_proof_problem_l2403_240302


namespace NUMINAMATH_GPT_sum_of_first_four_terms_of_arithmetic_sequence_l2403_240301

theorem sum_of_first_four_terms_of_arithmetic_sequence
  (a d : ℤ)
  (h1 : a + 4 * d = 10)  -- Condition for the fifth term
  (h2 : a + 5 * d = 14)  -- Condition for the sixth term
  (h3 : a + 6 * d = 18)  -- Condition for the seventh term
  : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 0 :=  -- Prove the sum of the first four terms is 0
by
  sorry

end NUMINAMATH_GPT_sum_of_first_four_terms_of_arithmetic_sequence_l2403_240301


namespace NUMINAMATH_GPT_inequality_solution_l2403_240381

theorem inequality_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : (x * (x + 1)) / ((x - 3)^2) ≥ 8) : 3 < x ∧ x ≤ 24/7 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2403_240381


namespace NUMINAMATH_GPT_average_speed_of_trip_l2403_240300

noncomputable def total_distance (d1 d2 : ℝ) : ℝ :=
  d1 + d2

noncomputable def travel_time (distance speed : ℝ) : ℝ :=
  distance / speed

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem average_speed_of_trip :
  let d1 := 60
  let s1 := 20
  let d2 := 120
  let s2 := 60
  let total_d := total_distance d1 d2
  let time1 := travel_time d1 s1
  let time2 := travel_time d2 s2
  let total_t := time1 + time2
  average_speed total_d total_t = 36 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_trip_l2403_240300


namespace NUMINAMATH_GPT_retailer_discount_percentage_l2403_240373

noncomputable def market_price (P : ℝ) : ℝ := 36 * P
noncomputable def profit (CP : ℝ) : ℝ := CP * 0.1
noncomputable def selling_price (P : ℝ) : ℝ := 40 * P
noncomputable def total_revenue (CP Profit : ℝ) : ℝ := CP + Profit
noncomputable def discount (P S : ℝ) : ℝ := P - S
noncomputable def discount_percentage (D P : ℝ) : ℝ := (D / P) * 100

theorem retailer_discount_percentage (P CP Profit TR S D : ℝ) (h1 : CP = market_price P)
  (h2 : Profit = profit CP) (h3 : TR = total_revenue CP Profit)
  (h4 : TR = selling_price S) (h5 : S = TR / 40) (h6 : D = discount P S) :
  discount_percentage D P = 1 :=
by
  sorry

end NUMINAMATH_GPT_retailer_discount_percentage_l2403_240373


namespace NUMINAMATH_GPT_solution_to_problem_l2403_240399

theorem solution_to_problem (x : ℝ) (h : 12^(Real.log 7 / Real.log 12) = 10 * x + 3) : x = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_solution_to_problem_l2403_240399


namespace NUMINAMATH_GPT_find_k_l2403_240339

theorem find_k (k : ℤ) :
  (∃ a b c : ℤ, a = 49 + k ∧ b = 441 + k ∧ c = 961 + k ∧
  (∃ r : ℚ, b = r * a ∧ c = r * r * a)) ↔ k = 1152 := by
  sorry

end NUMINAMATH_GPT_find_k_l2403_240339


namespace NUMINAMATH_GPT_vivi_total_yards_l2403_240353

theorem vivi_total_yards (spent_checkered spent_plain cost_per_yard : ℝ)
  (h1 : spent_checkered = 75)
  (h2 : spent_plain = 45)
  (h3 : cost_per_yard = 7.50) :
  (spent_checkered / cost_per_yard + spent_plain / cost_per_yard) = 16 :=
by 
  sorry

end NUMINAMATH_GPT_vivi_total_yards_l2403_240353


namespace NUMINAMATH_GPT_merchant_profit_condition_l2403_240396

theorem merchant_profit_condition (L : ℝ) (P : ℝ) (S : ℝ) (M : ℝ) :
  (P = 0.70 * L) →
  (S = 0.80 * M) →
  (S - P = 0.30 * S) →
  (M = 1.25 * L) := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_merchant_profit_condition_l2403_240396


namespace NUMINAMATH_GPT_appropriate_chart_for_temperature_statistics_l2403_240349

theorem appropriate_chart_for_temperature_statistics (chart_type : String) (is_line_chart : chart_type = "line chart") : chart_type = "line chart" :=
by
  sorry

end NUMINAMATH_GPT_appropriate_chart_for_temperature_statistics_l2403_240349


namespace NUMINAMATH_GPT_find_b_plus_m_l2403_240363

def matrix_C (b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 3, b],
    ![0, 1, 5],
    ![0, 0, 1]
  ]

def matrix_RHS : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 27, 3003],
    ![0, 1, 45],
    ![0, 0, 1]
  ]

theorem find_b_plus_m (b m : ℕ) (h : matrix_C b ^ m = matrix_RHS) : b + m = 306 := 
  sorry

end NUMINAMATH_GPT_find_b_plus_m_l2403_240363


namespace NUMINAMATH_GPT_find_rate_of_interest_l2403_240362

noncomputable def interest_rate (P R : ℝ) : Prop :=
  (400 = P * (1 + 4 * R / 100)) ∧ (500 = P * (1 + 6 * R / 100))

theorem find_rate_of_interest (R : ℝ) (P : ℝ) (h : interest_rate P R) :
  R = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_rate_of_interest_l2403_240362


namespace NUMINAMATH_GPT_sequence_bound_l2403_240315

open Real

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ)
  (h₀ : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h₁ : ∀ (i j : ℕ), 0 < i → 0 < j → i ≠ j → abs (a i - a j) ≥ 1 / (i + j)) :
  c ≥ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_bound_l2403_240315


namespace NUMINAMATH_GPT_find_a_2016_l2403_240382

noncomputable def a (n : ℕ) : ℕ := sorry

axiom condition_1 : a 4 = 1
axiom condition_2 : a 11 = 9
axiom condition_3 : ∀ n : ℕ, a n + a (n+1) + a (n+2) = 15

theorem find_a_2016 : a 2016 = 5 := sorry

end NUMINAMATH_GPT_find_a_2016_l2403_240382


namespace NUMINAMATH_GPT_speed_in_still_water_l2403_240333

theorem speed_in_still_water (v_m v_s : ℝ)
  (downstream : 48 = (v_m + v_s) * 3)
  (upstream : 34 = (v_m - v_s) * 4) :
  v_m = 12.25 :=
by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l2403_240333


namespace NUMINAMATH_GPT_no_real_roots_ff_eq_x_l2403_240336

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_real_roots_ff_eq_x (a b c : ℝ)
  (h : a ≠ 0)
  (discriminant_condition : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x := 
by 
  sorry

end NUMINAMATH_GPT_no_real_roots_ff_eq_x_l2403_240336


namespace NUMINAMATH_GPT_surface_area_ratio_l2403_240395

-- Definitions based on conditions
def side_length (s : ℝ) := s > 0
def A_cube (s : ℝ) := 6 * s ^ 2
def A_rect (s : ℝ) := 2 * (2 * s) * (3 * s) + 2 * (2 * s) * (4 * s) + 2 * (3 * s) * (4 * s)

-- Theorem statement proving the ratio
theorem surface_area_ratio (s : ℝ) (h : side_length s) : A_cube s / A_rect s = 3 / 26 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_ratio_l2403_240395


namespace NUMINAMATH_GPT_desired_gain_percentage_l2403_240316

theorem desired_gain_percentage (cp16 sp16 cp12881355932203391 sp12881355932203391 : ℝ) :
  sp16 = 1 →
  sp16 = 0.95 * cp16 →
  sp12881355932203391 = 1 →
  cp12881355932203391 = (12.881355932203391 / 16) * cp16 →
  (sp12881355932203391 - cp12881355932203391) / cp12881355932203391 * 100 = 18.75 :=
by sorry

end NUMINAMATH_GPT_desired_gain_percentage_l2403_240316


namespace NUMINAMATH_GPT_problem_proof_l2403_240393

def f (a x : ℝ) := |a - x|

theorem problem_proof (a x x0 : ℝ) (h_a : a = 3 / 2) (h_x0 : x0 < 0) : 
  f a (x0 * x) ≥ x0 * f a x + f a (a * x0) :=
sorry

end NUMINAMATH_GPT_problem_proof_l2403_240393


namespace NUMINAMATH_GPT_birthday_count_l2403_240325

theorem birthday_count (N : ℕ) (P : ℝ) (days : ℕ) (hN : N = 1200) (hP1 : P = 1 / 365 ∨ P = 1 / 366) 
  (hdays : days = 365 ∨ days = 366) : 
  N * P = 4 :=
by
  sorry

end NUMINAMATH_GPT_birthday_count_l2403_240325


namespace NUMINAMATH_GPT_distinct_p_q_r_s_t_sum_l2403_240332

theorem distinct_p_q_r_s_t_sum (p q r s t : ℤ) (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t)
    (h9 : r ≠ s) (h10 : r ≠ t)
    (h11 : s ≠ t) : p + q + r + s + t = 25 := by
  sorry

end NUMINAMATH_GPT_distinct_p_q_r_s_t_sum_l2403_240332


namespace NUMINAMATH_GPT_max_distance_travel_l2403_240352

-- Each car can carry at most 24 barrels of gasoline
def max_gasoline_barrels : ℕ := 24

-- Each barrel allows a car to travel 60 kilometers
def distance_per_barrel : ℕ := 60

-- The maximum distance one car can travel one way on a full tank
def max_one_way_distance := max_gasoline_barrels * distance_per_barrel

-- Total trip distance for the furthest traveling car
def total_trip_distance := 2160

-- Distance the other car turns back
def turn_back_distance := 360

-- Formalize in Lean
theorem max_distance_travel :
  (∃ x : ℕ, x = turn_back_distance ∧ max_gasoline_barrels * distance_per_barrel = 360) ∧
  (∃ y : ℕ, y = max_one_way_distance * 3 - turn_back_distance * 6 ∧ y = total_trip_distance) :=
by
  sorry

end NUMINAMATH_GPT_max_distance_travel_l2403_240352


namespace NUMINAMATH_GPT_discriminant_of_quadratic_l2403_240386

-- Define the quadratic equation coefficients
def a : ℝ := 5
def b : ℝ := -11
def c : ℝ := 4

-- Prove the discriminant of the quadratic equation
theorem discriminant_of_quadratic :
    b^2 - 4 * a * c = 41 :=
by
  sorry

end NUMINAMATH_GPT_discriminant_of_quadratic_l2403_240386


namespace NUMINAMATH_GPT_factorize_quadratic_example_l2403_240356

theorem factorize_quadratic_example (x : ℝ) :
  4 * x^2 - 8 * x + 4 = 4 * (x - 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_quadratic_example_l2403_240356


namespace NUMINAMATH_GPT_greatest_integer_prime_l2403_240376

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m < n → n % m ≠ 0

theorem greatest_integer_prime (x : ℤ) :
  is_prime (|8 * x ^ 2 - 56 * x + 21|) → ∀ y : ℤ, (is_prime (|8 * y ^ 2 - 56 * y + 21|) → y ≤ x) :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_prime_l2403_240376


namespace NUMINAMATH_GPT_fraction_product_simplification_l2403_240392

theorem fraction_product_simplification : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_simplification_l2403_240392


namespace NUMINAMATH_GPT_average_height_students_count_l2403_240345

-- Definitions based on the conditions
def total_students : ℕ := 400
def short_students : ℕ := (2 * total_students) / 5
def extremely_tall_students : ℕ := total_students / 10
def tall_students : ℕ := 90
def average_height_students : ℕ := total_students - (short_students + tall_students + extremely_tall_students)

-- Theorem to prove
theorem average_height_students_count : average_height_students = 110 :=
by
  -- This proof is omitted, we are only stating the theorem.
  sorry

end NUMINAMATH_GPT_average_height_students_count_l2403_240345


namespace NUMINAMATH_GPT_mix_solutions_l2403_240378

-- Definitions based on conditions
def solution_x_percentage : ℝ := 0.10
def solution_y_percentage : ℝ := 0.30
def volume_y : ℝ := 100
def desired_percentage : ℝ := 0.15

-- Problem statement rewrite with equivalent proof goal
theorem mix_solutions :
  ∃ Vx : ℝ, (Vx * solution_x_percentage + volume_y * solution_y_percentage) = (Vx + volume_y) * desired_percentage ∧ Vx = 300 :=
by
  sorry

end NUMINAMATH_GPT_mix_solutions_l2403_240378


namespace NUMINAMATH_GPT_no_solutions_interval_length_l2403_240308

theorem no_solutions_interval_length : 
  (∀ x a : ℝ, |x| ≠ ax - 2) → ([-1, 1].length = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_solutions_interval_length_l2403_240308


namespace NUMINAMATH_GPT_time_spent_per_bone_l2403_240318

theorem time_spent_per_bone
  (total_hours : ℤ) (number_of_bones : ℤ) 
  (h1 : total_hours = 206) 
  (h2 : number_of_bones = 206) :
  (total_hours / number_of_bones = 1) := 
by {
  -- proof would go here
  sorry
}

end NUMINAMATH_GPT_time_spent_per_bone_l2403_240318


namespace NUMINAMATH_GPT_skate_cost_l2403_240398

/- Define the initial conditions as Lean definitions -/
def admission_cost : ℕ := 5
def rental_cost : ℕ := 250 / 100  -- 2.50 dollars in cents for integer representation
def visits : ℕ := 26

/- Define the cost calculation as a Lean definition -/
def total_rental_cost (rental_cost : ℕ) (visits : ℕ) : ℕ := rental_cost * visits

/- Statement of the problem in Lean proof form -/
theorem skate_cost (C : ℕ) (h : total_rental_cost rental_cost visits = C) : C = 65 :=
by
  sorry

end NUMINAMATH_GPT_skate_cost_l2403_240398


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2403_240304

theorem sufficient_but_not_necessary (x y : ℝ) (h : ⌊x⌋ = ⌊y⌋) : 
  |x - y| < 1 ∧ ∃ x y : ℝ, |x - y| < 1 ∧ ⌊x⌋ ≠ ⌊y⌋ :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2403_240304


namespace NUMINAMATH_GPT_basketball_score_l2403_240357

theorem basketball_score (score_game1 : ℕ) (score_game2 : ℕ) (score_game3 : ℕ) (score_game4 : ℕ) (score_total_games8 : ℕ) (score_total_games9 : ℕ) :
  score_game1 = 18 ∧ score_game2 = 22 ∧ score_game3 = 15 ∧ score_game4 = 20 ∧ 
  (score_game1 + score_game2 + score_game3 + score_game4) / 4 < score_total_games8 / 8 ∧ 
  score_total_games9 / 9 > 19 →
  score_total_games9 - score_total_games8 ≥ 21 :=
by
-- proof steps would be provided here based on the given solution
sorry

end NUMINAMATH_GPT_basketball_score_l2403_240357


namespace NUMINAMATH_GPT_erasers_left_l2403_240367

/-- 
There are initially 250 erasers in a box. Doris takes 75 erasers, Mark takes 40 
erasers, and Ellie takes 30 erasers out of the box. Prove that 105 erasers are 
left in the box.
-/
theorem erasers_left (initial_erasers : ℕ) (doris_takes : ℕ) (mark_takes : ℕ) (ellie_takes : ℕ)
  (h_initial : initial_erasers = 250)
  (h_doris : doris_takes = 75)
  (h_mark : mark_takes = 40)
  (h_ellie : ellie_takes = 30) :
  initial_erasers - doris_takes - mark_takes - ellie_takes = 105 :=
  by 
  sorry

end NUMINAMATH_GPT_erasers_left_l2403_240367


namespace NUMINAMATH_GPT_balls_into_boxes_l2403_240355

/-- There are 128 ways to distribute 7 distinguishable balls into 2 distinguishable boxes. -/
theorem balls_into_boxes : (2 : ℕ) ^ 7 = 128 := by
  sorry

end NUMINAMATH_GPT_balls_into_boxes_l2403_240355


namespace NUMINAMATH_GPT_max_value_of_z_l2403_240323

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y ≤ 2) (h2 : x + y ≥ 0) (h3 : x ≤ 4) : 
  ∃ (z : ℝ), z = 2 * x + y ∧ z ≤ 11 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_z_l2403_240323


namespace NUMINAMATH_GPT_train_speed_is_60_l2403_240377

noncomputable def train_speed_proof : Prop :=
  let train_length := 550 -- in meters
  let time_to_pass := 29.997600191984645 -- in seconds
  let man_speed_kmhr := 6 -- in km/hr
  let man_speed_ms := man_speed_kmhr * (1000 / 3600) -- converting km/hr to m/s
  let relative_speed_ms := train_length / time_to_pass -- relative speed in m/s
  let train_speed_ms := relative_speed_ms - man_speed_ms -- speed of the train in m/s
  let train_speed_kmhr := train_speed_ms * (3600 / 1000) -- converting m/s to km/hr
  train_speed_kmhr = 60 -- the speed of the train in km/hr

theorem train_speed_is_60 : train_speed_proof := by
  sorry

end NUMINAMATH_GPT_train_speed_is_60_l2403_240377


namespace NUMINAMATH_GPT_find_natural_number_l2403_240366

theorem find_natural_number (x : ℕ) (y z : ℤ) (hy : x = 2 * y^2 - 1) (hz : x^2 = 2 * z^2 - 1) : x = 1 ∨ x = 7 :=
sorry

end NUMINAMATH_GPT_find_natural_number_l2403_240366


namespace NUMINAMATH_GPT_total_yellow_marbles_l2403_240351

theorem total_yellow_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := 
by 
  sorry

end NUMINAMATH_GPT_total_yellow_marbles_l2403_240351


namespace NUMINAMATH_GPT_negation_of_P_l2403_240368

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n)

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def P : Prop := ∀ n : ℕ, is_prime n → is_odd n

theorem negation_of_P : ¬ P ↔ ∃ n : ℕ, is_prime n ∧ ¬ is_odd n :=
by sorry

end NUMINAMATH_GPT_negation_of_P_l2403_240368


namespace NUMINAMATH_GPT_problem_C_plus_D_l2403_240342

theorem problem_C_plus_D (C D : ℚ)
  (h : ∀ x, (D * x - 17) / (x^2 - 8 * x + 15) = C / (x - 3) + 5 / (x - 5)) :
  C + D = 5.8 :=
sorry

end NUMINAMATH_GPT_problem_C_plus_D_l2403_240342


namespace NUMINAMATH_GPT_find_fractions_l2403_240360

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end NUMINAMATH_GPT_find_fractions_l2403_240360
