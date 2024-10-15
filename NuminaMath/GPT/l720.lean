import Mathlib

namespace NUMINAMATH_GPT_mira_jogs_hours_each_morning_l720_72047

theorem mira_jogs_hours_each_morning 
  (h : ℝ) -- number of hours Mira jogs each morning
  (speed : ℝ) -- Mira's jogging speed in miles per hour
  (days : ℝ) -- number of days Mira jogs
  (total_distance : ℝ) -- total distance Mira jogs

  (H1 : speed = 5) 
  (H2 : days = 5) 
  (H3 : total_distance = 50) 
  (H4 : total_distance = speed * h * days) :

  h = 2 :=
by
  sorry

end NUMINAMATH_GPT_mira_jogs_hours_each_morning_l720_72047


namespace NUMINAMATH_GPT_total_snowfall_l720_72012

theorem total_snowfall (morning afternoon : ℝ) (h1 : morning = 0.125) (h2 : afternoon = 0.5) :
  morning + afternoon = 0.625 := by
  sorry

end NUMINAMATH_GPT_total_snowfall_l720_72012


namespace NUMINAMATH_GPT_bryan_samples_l720_72027

noncomputable def initial_samples_per_shelf : ℕ := 128
noncomputable def shelves : ℕ := 13
noncomputable def samples_removed_per_shelf : ℕ := 2
noncomputable def remaining_samples_per_shelf := initial_samples_per_shelf - samples_removed_per_shelf
noncomputable def total_remaining_samples := remaining_samples_per_shelf * shelves

theorem bryan_samples : total_remaining_samples = 1638 := 
by 
  sorry

end NUMINAMATH_GPT_bryan_samples_l720_72027


namespace NUMINAMATH_GPT_triangle_circle_square_value_l720_72031

theorem triangle_circle_square_value (Δ : ℝ) (bigcirc : ℝ) (square : ℝ) 
  (h1 : 2 * Δ + 3 * bigcirc + square = 45)
  (h2 : Δ + 5 * bigcirc + 2 * square = 58)
  (h3 : 3 * Δ + bigcirc + 3 * square = 62) :
  Δ + 2 * bigcirc + square = 35 :=
sorry

end NUMINAMATH_GPT_triangle_circle_square_value_l720_72031


namespace NUMINAMATH_GPT_circumference_of_jack_head_l720_72004

theorem circumference_of_jack_head (J C : ℝ) (h1 : (2 / 3) * C = 10) (h2 : (1 / 2) * J + 9 = 15) :
  J = 12 :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_jack_head_l720_72004


namespace NUMINAMATH_GPT_sharona_bought_more_pencils_l720_72081

-- Define constants for the amounts paid
def amount_paid_jamar : ℚ := 1.43
def amount_paid_sharona : ℚ := 1.87

-- Define the function that computes the number of pencils given the price per pencil and total amount paid
def num_pencils (amount_paid : ℚ) (price_per_pencil : ℚ) : ℚ := amount_paid / price_per_pencil

-- Define the theorem stating that Sharona bought 4 more pencils than Jamar
theorem sharona_bought_more_pencils {price_per_pencil : ℚ} (h_price : price_per_pencil > 0) :
  num_pencils amount_paid_sharona price_per_pencil = num_pencils amount_paid_jamar price_per_pencil + 4 :=
sorry

end NUMINAMATH_GPT_sharona_bought_more_pencils_l720_72081


namespace NUMINAMATH_GPT_sign_of_c_l720_72084

/-
Define the context and conditions as Lean axioms.
-/

variables (a b c : ℝ)

-- Axiom: The sum of coefficients is less than zero
axiom h1 : a + b + c < 0

-- Axiom: The quadratic equation has no real roots, thus the discriminant is less than zero
axiom h2 : (b^2 - 4*a*c) < 0

/-
Formal statement of the proof problem:
-/

theorem sign_of_c : c < 0 :=
by
  -- We state that the proof of c < 0 follows from the given axioms
  sorry

end NUMINAMATH_GPT_sign_of_c_l720_72084


namespace NUMINAMATH_GPT_initial_fee_l720_72013

theorem initial_fee (initial_fee : ℝ) : 
  (∀ (distance_charge_per_segment travel_total_charge : ℝ), 
    distance_charge_per_segment = 0.35 → 
    3.6 / 0.4 * distance_charge_per_segment + initial_fee = travel_total_charge → 
    travel_total_charge = 5.20)
    → initial_fee = 2.05 :=
by
  intro h
  specialize h 0.35 5.20
  sorry

end NUMINAMATH_GPT_initial_fee_l720_72013


namespace NUMINAMATH_GPT_female_managers_count_l720_72032

-- Definitions based on conditions
def total_employees : Nat := 250
def female_employees : Nat := 90
def total_managers : Nat := 40
def male_associates : Nat := 160

-- Statement to prove
theorem female_managers_count : (total_managers = 40) :=
by
  sorry

end NUMINAMATH_GPT_female_managers_count_l720_72032


namespace NUMINAMATH_GPT_cost_per_item_l720_72050

theorem cost_per_item (total_profit : ℝ) (total_customers : ℕ) (purchase_percentage : ℝ) (pays_advertising : ℝ)
    (H1: total_profit = 1000)
    (H2: total_customers = 100)
    (H3: purchase_percentage = 0.80)
    (H4: pays_advertising = 1000)
    : (total_profit / (total_customers * purchase_percentage)) = 12.50 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_item_l720_72050


namespace NUMINAMATH_GPT_inscribed_circle_radius_l720_72089

theorem inscribed_circle_radius (a b c : ℝ) (R : ℝ) (r : ℝ) :
  a = 20 → b = 20 → d = 25 → r = 6 := 
by
  -- conditions of the problem
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l720_72089


namespace NUMINAMATH_GPT_g_of_negative_8_l720_72000

def f (x : ℝ) : ℝ := 4 * x - 9
def g (y : ℝ) : ℝ := y^2 + 6 * y - 7

theorem g_of_negative_8 : g (-8) = -87 / 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_g_of_negative_8_l720_72000


namespace NUMINAMATH_GPT_inequality_solution_exists_l720_72019

theorem inequality_solution_exists (x m : ℝ) (h1: 1 < x) (h2: x ≤ 2) (h3: x > m) : m < 2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_exists_l720_72019


namespace NUMINAMATH_GPT_initial_scooter_value_l720_72042

theorem initial_scooter_value (V : ℝ) 
    (h : (9 / 16) * V = 22500) : V = 40000 :=
sorry

end NUMINAMATH_GPT_initial_scooter_value_l720_72042


namespace NUMINAMATH_GPT_triangle_area_l720_72073

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 180 := 
by 
  -- proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_triangle_area_l720_72073


namespace NUMINAMATH_GPT_find_z_l720_72029

open Complex

theorem find_z (z : ℂ) : (1 + 2*I) * z = 3 - I → z = (1/5) - (7/5)*I :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_z_l720_72029


namespace NUMINAMATH_GPT_least_subtract_divisible_l720_72007

theorem least_subtract_divisible:
  ∃ n : ℕ, n = 31 ∧ (13603 - n) % 87 = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_subtract_divisible_l720_72007


namespace NUMINAMATH_GPT_days_for_B_l720_72058

theorem days_for_B
  (x : ℝ)
  (hA : 15 ≠ 0)
  (h_nonzero_fraction : 0.5833333333333334 ≠ 0)
  (hfraction : 0 <  0.5833333333333334 ∧ 0.5833333333333334 < 1)
  (h_fraction_work_left : 5 * (1 / 15 + 1 / x) = 0.5833333333333334) :
  x = 20 := by
  sorry

end NUMINAMATH_GPT_days_for_B_l720_72058


namespace NUMINAMATH_GPT_laura_annual_income_l720_72072

variable (p : ℝ) -- percentage p
variable (A T : ℝ) -- annual income A and total income tax T

def tax1 : ℝ := 0.01 * p * 35000
def tax2 : ℝ := 0.01 * (p + 3) * (A - 35000)
def tax3 : ℝ := 0.01 * (p + 5) * (A - 55000)

theorem laura_annual_income (h_cond1 : A > 55000)
  (h_tax : T = 350 * p + 600 + 0.01 * (p + 5) * (A - 55000))
  (h_paid_tax : T = (0.01 * (p + 0.45)) * A):
  A = 75000 := by
  sorry

end NUMINAMATH_GPT_laura_annual_income_l720_72072


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l720_72037

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 1 / x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / 2 → (∀ y : ℝ, y < x → f y < f x) :=
by
  intro x h
  intro y hy
  sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l720_72037


namespace NUMINAMATH_GPT_perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l720_72008

-- Mathematical definitions and theorems required for the problem
theorem perpendicular_lines_condition (m : ℝ) :
  3 * m + m * (2 * m - 1) = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

-- Translate the specific problem into Lean
theorem perpendicular_lines_sufficient_not_necessary (m : ℝ) (h : 3 * m + m * (2 * m - 1) = 0) :
  m = -1 ∨ (m ≠ -1 ∧ 3 * m + m * (2 * m - 1) = 0) :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l720_72008


namespace NUMINAMATH_GPT_solution_set_inequality_l720_72095

theorem solution_set_inequality (a c : ℝ)
  (h : ∀ x : ℝ, (ax^2 + 2*x + c < 0) ↔ (x < -1/3 ∨ x > 1/2)) :
  (∀ x : ℝ, (cx^2 - 2*x + a ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 3)) :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l720_72095


namespace NUMINAMATH_GPT_power_of_five_trailing_zeros_l720_72040

theorem power_of_five_trailing_zeros (n : ℕ) (h : n = 1968) : 
  ∃ k : ℕ, 5^n = 10^k ∧ k ≥ 1968 := 
by 
  sorry

end NUMINAMATH_GPT_power_of_five_trailing_zeros_l720_72040


namespace NUMINAMATH_GPT_average_speed_whole_journey_l720_72082

theorem average_speed_whole_journey (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 54
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  let V_avg := total_distance / total_time
  V_avg = 64.8 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_whole_journey_l720_72082


namespace NUMINAMATH_GPT_factorize_a_cubed_minus_a_l720_72014

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_a_cubed_minus_a_l720_72014


namespace NUMINAMATH_GPT_solution_l720_72039

theorem solution (x y : ℝ) (h₁ : x + 3 * y = -1) (h₂ : x - 3 * y = 5) : x^2 - 9 * y^2 = -5 := 
by
  sorry

end NUMINAMATH_GPT_solution_l720_72039


namespace NUMINAMATH_GPT_certain_number_l720_72078

-- Define the conditions as variables
variables {x : ℝ}

-- Define the proof problem
theorem certain_number (h : 0.15 * x = 0.025 * 450) : x = 75 :=
sorry

end NUMINAMATH_GPT_certain_number_l720_72078


namespace NUMINAMATH_GPT_books_written_by_Zig_l720_72092

theorem books_written_by_Zig (F Z : ℕ) (h1 : Z = 4 * F) (h2 : F + Z = 75) : Z = 60 := by
  sorry

end NUMINAMATH_GPT_books_written_by_Zig_l720_72092


namespace NUMINAMATH_GPT_sum_slope_y_intercept_l720_72057

theorem sum_slope_y_intercept (A B C F : ℝ × ℝ) (midpoint_A_C : F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) 
  (coords_A : A = (0, 6)) (coords_B : B = (0, 0)) (coords_C : C = (8, 0)) :
  let slope : ℝ := (F.2 - B.2) / (F.1 - B.1)
  let y_intercept : ℝ := B.2
  slope + y_intercept = 3 / 4 := by
{
  -- proof steps
  sorry
}

end NUMINAMATH_GPT_sum_slope_y_intercept_l720_72057


namespace NUMINAMATH_GPT_satisfy_eqn_l720_72090

/-- 
  Prove that the integer pairs (0, 1), (0, -1), (1, 0), (-1, 0), (2, 2), (-2, -2)
  are the only pairs that satisfy x^5 + y^5 = (x + y)^3
-/
theorem satisfy_eqn (x y : ℤ) : 
  (x, y) = (0, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (1, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (2, 2) ∨ (x, y) = (-2, -2) ↔ 
  x^5 + y^5 = (x + y)^3 := 
by 
  sorry

end NUMINAMATH_GPT_satisfy_eqn_l720_72090


namespace NUMINAMATH_GPT_geometric_sequence_sum_l720_72022

open Nat

noncomputable def geometric_sum (a q n : ℕ) : ℕ :=
  a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (S : ℕ → ℕ) (q a₁ : ℕ)
  (h_q: q = 2)
  (h_S5: S 5 = 1)
  (h_S: ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) :
  S 10 = 33 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l720_72022


namespace NUMINAMATH_GPT_base9_subtraction_l720_72025

theorem base9_subtraction (a b : Nat) (h1 : a = 256) (h2 : b = 143) : 
  (a - b) = 113 := 
sorry

end NUMINAMATH_GPT_base9_subtraction_l720_72025


namespace NUMINAMATH_GPT_volleyball_team_selection_l720_72026

/-- A set representing players on the volleyball team -/
def players : Finset String := {
  "Missy", "Lauren", "Liz", -- triplets
  "Anna", "Mia",           -- twins
  "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10" -- other players
}

/-- The triplets -/
def triplets : Finset String := {"Missy", "Lauren", "Liz"}

/-- The twins -/
def twins : Finset String := {"Anna", "Mia"}

/-- The number of ways to choose 7 starters given the restrictions -/
theorem volleyball_team_selection : 
  let total_ways := (players.card.choose 7)
  let select_3_triplets := (players \ triplets).card.choose 4
  let select_2_twins := (players \ twins).card.choose 5
  let select_all_restriction := (players \ (triplets ∪ twins)).card.choose 2
  total_ways - select_3_triplets - select_2_twins + select_all_restriction = 9778 := by
  sorry

end NUMINAMATH_GPT_volleyball_team_selection_l720_72026


namespace NUMINAMATH_GPT_nate_age_is_14_l720_72002

def nate_current_age (N : ℕ) : Prop :=
  ∃ E : ℕ, E = N / 2 ∧ N - E = 7

theorem nate_age_is_14 : nate_current_age 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_nate_age_is_14_l720_72002


namespace NUMINAMATH_GPT_necessary_condition_for_inequality_l720_72091

-- Definitions based on the conditions in a)
variables (A B C D : ℝ)

-- Main statement translating c) into Lean
theorem necessary_condition_for_inequality (h : C < D) : A > B :=
by sorry

end NUMINAMATH_GPT_necessary_condition_for_inequality_l720_72091


namespace NUMINAMATH_GPT_xy_difference_l720_72053

noncomputable def x : ℝ := Real.sqrt 3 + 1
noncomputable def y : ℝ := Real.sqrt 3 - 1

theorem xy_difference : x^2 * y - x * y^2 = 4 := by
  sorry

end NUMINAMATH_GPT_xy_difference_l720_72053


namespace NUMINAMATH_GPT_cross_section_perimeter_l720_72093

-- Define the lengths of the diagonals AC and BD.
def length_AC : ℝ := 8
def length_BD : ℝ := 12

-- Define the perimeter calculation for the cross-section quadrilateral
-- that passes through the midpoint E of AB and is parallel to BD and AC.
theorem cross_section_perimeter :
  let side1 := length_AC / 2
  let side2 := length_BD / 2
  let perimeter := 2 * (side1 + side2)
  perimeter = 20 :=
by
  sorry

end NUMINAMATH_GPT_cross_section_perimeter_l720_72093


namespace NUMINAMATH_GPT_length_of_segment_XY_l720_72046

noncomputable def rectangle_length (A B C D : ℝ) (BX DY : ℝ) : ℝ :=
  2 * BX + DY

theorem length_of_segment_XY (A B C D : ℝ) (BX DY : ℝ) (h1 : C = 2 * B) (h2 : BX = 4) (h3 : DY = 10) :
  rectangle_length A B C D BX DY = 13 :=
by
  rw [rectangle_length, h2, h3]
  sorry

end NUMINAMATH_GPT_length_of_segment_XY_l720_72046


namespace NUMINAMATH_GPT_PersonYs_speed_in_still_water_l720_72094

def speed_in_still_water (speed_X : ℕ) (t_1 t_2 : ℕ) (x : ℕ) : Prop :=
  ∀ y : ℤ, 4 * (6 - y + x + y) = 4 * 6 + 4 * x ∧ 16 * (x + y) = 16 * (6 + y) + 4 * (x - 6) →
  x = 10

theorem PersonYs_speed_in_still_water :
  speed_in_still_water 6 4 16 10 :=
by
  sorry

end NUMINAMATH_GPT_PersonYs_speed_in_still_water_l720_72094


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l720_72001

theorem quadratic_inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, (2 < x ∧ x < 3) → (ax^2 + 5*x + b > 0)) →
  ∃ x : ℝ, (-1/2 < x ∧ x < -1/3) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l720_72001


namespace NUMINAMATH_GPT_solution_set_empty_l720_72080

variable (m x : ℝ)
axiom no_solution (h : (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) : (1 + m = 0)

theorem solution_set_empty :
  (∀ x, (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end NUMINAMATH_GPT_solution_set_empty_l720_72080


namespace NUMINAMATH_GPT_no_coprime_odd_numbers_for_6_8_10_l720_72069

theorem no_coprime_odd_numbers_for_6_8_10 :
  ∀ (m n : ℤ), m > n ∧ n > 0 ∧ (m.gcd n = 1) ∧ (m % 2 = 1) ∧ (n % 2 = 1) →
    (1 / 2 : ℚ) * (m^2 - n^2) ≠ 6 ∨ (m * n) ≠ 8 ∨ (1 / 2 : ℚ) * (m^2 + n^2) ≠ 10 :=
by
  sorry

end NUMINAMATH_GPT_no_coprime_odd_numbers_for_6_8_10_l720_72069


namespace NUMINAMATH_GPT_small_panda_bears_count_l720_72062

theorem small_panda_bears_count :
  ∃ (S : ℕ), ∃ (B : ℕ),
    B = 5 ∧ 7 * (25 * S + 40 * B) = 2100 ∧ S = 4 :=
by
  exists 4
  exists 5
  repeat { sorry }

end NUMINAMATH_GPT_small_panda_bears_count_l720_72062


namespace NUMINAMATH_GPT_marble_selection_probability_l720_72083

theorem marble_selection_probability :
  let total_marbles := 9
  let selected_marbles := 4
  let total_ways := Nat.choose total_marbles selected_marbles
  let red_marbles := 3
  let blue_marbles := 3
  let green_marbles := 3
  let ways_one_red := Nat.choose red_marbles 1
  let ways_two_blue := Nat.choose blue_marbles 2
  let ways_one_green := Nat.choose green_marbles 1
  let favorable_outcomes := ways_one_red * ways_two_blue * ways_one_green
  (favorable_outcomes : ℚ) / total_ways = 3 / 14 :=
by
  sorry

end NUMINAMATH_GPT_marble_selection_probability_l720_72083


namespace NUMINAMATH_GPT_max_price_of_most_expensive_product_l720_72067

noncomputable def greatest_possible_price
  (num_products : ℕ)
  (avg_price : ℕ)
  (min_price : ℕ)
  (mid_price : ℕ)
  (higher_price_count : ℕ)
  (total_retail_price : ℕ)
  (least_expensive_total_price : ℕ)
  (remaining_price : ℕ)
  (less_expensive_total_price : ℕ) : ℕ :=
  total_retail_price - least_expensive_total_price - less_expensive_total_price

theorem max_price_of_most_expensive_product :
  greatest_possible_price 20 1200 400 1000 10 (20 * 1200) (10 * 400) (20 * 1200 - 10 * 400) (9 * 1000) = 11000 :=
by
  sorry

end NUMINAMATH_GPT_max_price_of_most_expensive_product_l720_72067


namespace NUMINAMATH_GPT_circle_radius_l720_72005

theorem circle_radius {s r : ℝ} (h : 4 * s = π * r^2) (h1 : 2 * r = s * Real.sqrt 2) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l720_72005


namespace NUMINAMATH_GPT_polar_to_rectangular_l720_72038

theorem polar_to_rectangular :
  let x := 16
  let y := 12
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  let new_r := 2 * r
  let new_θ := θ / 2
  let cos_half_θ := Real.sqrt ((1 + (x / r)) / 2)
  let sin_half_θ := Real.sqrt ((1 - (x / r)) / 2)
  let new_x := new_r * cos_half_θ
  let new_y := new_r * sin_half_θ
  new_x = 40 * Real.sqrt 0.9 ∧ new_y = 40 * Real.sqrt 0.1 := by
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l720_72038


namespace NUMINAMATH_GPT_find_x_l720_72075

theorem find_x (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (x, 1))
  (hb : b = (2, x))
  (hc : c = (1, -2))
  (h_perpendicular : (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2)) = 0) :
  x = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_x_l720_72075


namespace NUMINAMATH_GPT_inequality_proof_l720_72066

open Real

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (hSum : x + y + z = 1) :
  x * y / sqrt (x * y + y * z) + y * z / sqrt (y * z + z * x) + z * x / sqrt (z * x + x * y) ≤ sqrt 2 / 2 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l720_72066


namespace NUMINAMATH_GPT_smallest_n_condition_l720_72059

-- Define the conditions
def condition1 (x : ℤ) : Prop := 2 * x - 3 ≡ 0 [ZMOD 13]
def condition2 (y : ℤ) : Prop := 3 * y + 4 ≡ 0 [ZMOD 13]

-- Problem statement: finding n such that the expression is a multiple of 13
theorem smallest_n_condition (x y : ℤ) (n : ℤ) :
  condition1 x → condition2 y → x^2 - x * y + y^2 + n ≡ 0 [ZMOD 13] → n = 1 := 
by
  sorry

end NUMINAMATH_GPT_smallest_n_condition_l720_72059


namespace NUMINAMATH_GPT_subtraction_correct_l720_72021

theorem subtraction_correct : 900000009000 - 123456789123 = 776543220777 :=
by
  -- Placeholder proof to ensure it compiles
  sorry

end NUMINAMATH_GPT_subtraction_correct_l720_72021


namespace NUMINAMATH_GPT_diameter_is_10sqrt6_l720_72006

noncomputable def radius (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  Real.sqrt (A / Real.pi)

noncomputable def diameter (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  2 * radius A hA

theorem diameter_is_10sqrt6 (A : ℝ) (hA : A = 150 * Real.pi) :
  diameter A hA = 10 * Real.sqrt 6 :=
  sorry

end NUMINAMATH_GPT_diameter_is_10sqrt6_l720_72006


namespace NUMINAMATH_GPT_find_smaller_number_l720_72087

-- Define the two numbers such that one is 3 times the other
def numbers (x : ℝ) := (x, 3 * x)

-- Define the condition that the sum of the two numbers is 14
def sum_condition (x y : ℝ) : Prop := x + y = 14

-- The theorem we want to prove
theorem find_smaller_number (x : ℝ) (hx : sum_condition x (3 * x)) : x = 3.5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_smaller_number_l720_72087


namespace NUMINAMATH_GPT_unique_solution_of_fraction_eq_l720_72076

theorem unique_solution_of_fraction_eq (x : ℝ) : (1 / (x - 1) = 2 / (x - 2)) ↔ (x = 0) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_of_fraction_eq_l720_72076


namespace NUMINAMATH_GPT_endpoint_of_parallel_segment_l720_72064

theorem endpoint_of_parallel_segment (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hA : A = (2, 1)) (h_parallel : B.snd = A.snd) (h_length : abs (B.fst - A.fst) = 5) :
  B = (7, 1) ∨ B = (-3, 1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_endpoint_of_parallel_segment_l720_72064


namespace NUMINAMATH_GPT_stuffed_animal_tickets_correct_l720_72016

-- Define the total tickets spent
def total_tickets : ℕ := 14

-- Define the tickets spent on the hat
def hat_tickets : ℕ := 2

-- Define the tickets spent on the yoyo
def yoyo_tickets : ℕ := 2

-- Define the tickets spent on the stuffed animal
def stuffed_animal_tickets : ℕ := total_tickets - (hat_tickets + yoyo_tickets)

-- The theorem we want to prove.
theorem stuffed_animal_tickets_correct :
  stuffed_animal_tickets = 10 :=
by
  sorry

end NUMINAMATH_GPT_stuffed_animal_tickets_correct_l720_72016


namespace NUMINAMATH_GPT_goods_train_speed_l720_72049

-- Define the given constants
def train_length : ℕ := 370 -- in meters
def platform_length : ℕ := 150 -- in meters
def crossing_time : ℕ := 26 -- in seconds
def conversion_factor : ℕ := 36 / 10 -- conversion from m/s to km/hr

-- Define the total distance covered
def total_distance : ℕ := train_length + platform_length -- in meters

-- Define the speed of the train in m/s
def speed_m_per_s : ℕ := total_distance / crossing_time

-- Define the speed of the train in km/hr
def speed_km_per_hr : ℕ := speed_m_per_s * conversion_factor

-- The proof problem statement
theorem goods_train_speed : speed_km_per_hr = 72 := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_goods_train_speed_l720_72049


namespace NUMINAMATH_GPT_y_is_80_percent_less_than_x_l720_72023

theorem y_is_80_percent_less_than_x (x y : ℝ) (h : x = 5 * y) : ((x - y) / x) * 100 = 80 :=
by sorry

end NUMINAMATH_GPT_y_is_80_percent_less_than_x_l720_72023


namespace NUMINAMATH_GPT_time_to_eat_cereal_l720_72071

noncomputable def MrFatRate : ℝ := 1 / 40
noncomputable def MrThinRate : ℝ := 1 / 15
noncomputable def CombinedRate : ℝ := MrFatRate + MrThinRate
noncomputable def CerealAmount : ℝ := 4
noncomputable def TimeToFinish : ℝ := CerealAmount / CombinedRate
noncomputable def expected_time : ℝ := 96

theorem time_to_eat_cereal :
  TimeToFinish = expected_time :=
by
  sorry

end NUMINAMATH_GPT_time_to_eat_cereal_l720_72071


namespace NUMINAMATH_GPT_Sequential_structure_not_conditional_l720_72097

-- Definitions based on provided conditions
def is_conditional (s : String) : Prop :=
  s = "Loop structure" ∨ s = "If structure" ∨ s = "Until structure"

-- Theorem stating that Sequential structure is the one that doesn't contain a conditional judgment box
theorem Sequential_structure_not_conditional :
  ¬ is_conditional "Sequential structure" :=
by
  intro h
  cases h <;> contradiction

end NUMINAMATH_GPT_Sequential_structure_not_conditional_l720_72097


namespace NUMINAMATH_GPT_cost_of_article_l720_72036

variable {C G : ℝ}

theorem cost_of_article (h : 350 = C * (1 + (G + 5) / 100)) (h' : 340 = C * (1 + G / 100)) : C = 200 := by
  sorry

end NUMINAMATH_GPT_cost_of_article_l720_72036


namespace NUMINAMATH_GPT_probability_odd_number_die_l720_72056

theorem probability_odd_number_die :
  let total_outcomes := 6
  let favorable_outcomes := 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_odd_number_die_l720_72056


namespace NUMINAMATH_GPT_inequality_proof_l720_72003

theorem inequality_proof
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  x^2 + y^2 + z^2 + x * y + y * z + z * x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l720_72003


namespace NUMINAMATH_GPT_find_n_l720_72010

theorem find_n (n m : ℕ) (h : m = 4) (eq1 : (1/5)^m * (1/4)^n = 1/(10^4)) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l720_72010


namespace NUMINAMATH_GPT_inequality_proof_l720_72065

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l720_72065


namespace NUMINAMATH_GPT_coconut_grove_nut_yield_l720_72052

/--
In a coconut grove, the trees produce nuts based on some given conditions. Prove that the number of nuts produced by (x + 4) trees per year is 720 when x is 8. The conditions are:

1. (x + 4) trees yield a certain number of nuts per year.
2. x trees yield 120 nuts per year.
3. (x - 4) trees yield 180 nuts per year.
4. The average yield per year per tree is 100.
5. x is 8.
-/

theorem coconut_grove_nut_yield (x : ℕ) (y z w: ℕ) (h₁ : x = 8) (h₂ : y = 120) (h₃ : z = 180) (h₄ : w = 100) :
  ((x + 4) * w) - (x * y + (x - 4) * z) = 720 := 
by
  sorry

end NUMINAMATH_GPT_coconut_grove_nut_yield_l720_72052


namespace NUMINAMATH_GPT_student_marks_l720_72034

theorem student_marks (M P C X : ℕ) 
  (h1 : M + P = 60)
  (h2 : C = P + X)
  (h3 : M + C = 80) : X = 20 :=
by sorry

end NUMINAMATH_GPT_student_marks_l720_72034


namespace NUMINAMATH_GPT_cotangent_identity_l720_72045

noncomputable def cotangent (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem cotangent_identity (x : ℝ) (i : ℂ) (n : ℕ) (k : ℕ) (h : (0 < k) ∧ (k < n)) :
  ((x + i) / (x - i))^n = 1 → x = cotangent (k * Real.pi / n) := 
sorry

end NUMINAMATH_GPT_cotangent_identity_l720_72045


namespace NUMINAMATH_GPT_distance_is_correct_l720_72086

noncomputable def distance_from_home_to_forest_park : ℝ := 11  -- distance in kilometers

structure ProblemData where
  v : ℝ                  -- Xiao Wu's bicycling speed (in meters per minute)
  t_catch_up : ℝ          -- time it takes for father to catch up (in minutes)
  d_forest : ℝ            -- distance from catch-up point to forest park (in kilometers)
  t_remaining : ℝ        -- time remaining for Wu to reach park after wallet delivered (in minutes)
  bike_speed_factor : ℝ   -- speed factor of father's car compared to Wu's bike
  
open ProblemData

def problem_conditions : ProblemData :=
  { v := 350,
    t_catch_up := 7.5,
    d_forest := 3.5,
    t_remaining := 10,
    bike_speed_factor := 5 }

theorem distance_is_correct (data : ProblemData) :
  data.v = 350 →
  data.t_catch_up = 7.5 →
  data.d_forest = 3.5 →
  data.t_remaining = 10 →
  data.bike_speed_factor = 5 →
  distance_from_home_to_forest_park = 11 := 
by
  intros
  sorry

end NUMINAMATH_GPT_distance_is_correct_l720_72086


namespace NUMINAMATH_GPT_max_two_digit_number_divisible_by_23_l720_72098

theorem max_two_digit_number_divisible_by_23 :
  ∃ n : ℕ, 
    (n < 100) ∧ 
    (1000 ≤ n * 109) ∧ 
    (n * 109 < 10000) ∧ 
    (n % 23 = 0) ∧ 
    (n / 23 < 10) ∧ 
    (n = 69) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_two_digit_number_divisible_by_23_l720_72098


namespace NUMINAMATH_GPT_jack_total_yen_l720_72060

def pounds := 42
def euros := 11
def yen := 3000
def pounds_per_euro := 2
def yen_per_pound := 100

theorem jack_total_yen : (euros * pounds_per_euro + pounds) * yen_per_pound + yen = 9400 := by
  sorry

end NUMINAMATH_GPT_jack_total_yen_l720_72060


namespace NUMINAMATH_GPT_B_cycling_speed_l720_72070

theorem B_cycling_speed (v : ℝ) : 
  (∀ (t : ℝ), 10 * t + 30 = B_start_distance) ∧ 
  (B_start_distance = 60) ∧ 
  (t = 3) →
  v = 20 :=
sorry

end NUMINAMATH_GPT_B_cycling_speed_l720_72070


namespace NUMINAMATH_GPT_volume_of_cube_l720_72043

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cube_l720_72043


namespace NUMINAMATH_GPT_Alice_min_speed_l720_72074

theorem Alice_min_speed (d : ℝ) (v_bob : ℝ) (delta_t : ℝ) (v_alice : ℝ) :
  d = 180 ∧ v_bob = 40 ∧ delta_t = 0.5 ∧ 0 < v_alice ∧ v_alice * (d / v_bob - delta_t) ≥ d →
  v_alice > 45 :=
by
  sorry

end NUMINAMATH_GPT_Alice_min_speed_l720_72074


namespace NUMINAMATH_GPT_candy_store_truffle_price_l720_72085

def total_revenue : ℝ := 212
def fudge_revenue : ℝ := 20 * 2.5
def pretzels_revenue : ℝ := 3 * 12 * 2.0
def truffles_quantity : ℕ := 5 * 12

theorem candy_store_truffle_price (total_revenue fudge_revenue pretzels_revenue truffles_quantity : ℝ) : 
  (total_revenue - (fudge_revenue + pretzels_revenue)) / truffles_quantity = 1.50 := 
by 
  sorry

end NUMINAMATH_GPT_candy_store_truffle_price_l720_72085


namespace NUMINAMATH_GPT_smallest_resolvable_debt_l720_72061

theorem smallest_resolvable_debt (p g : ℤ) : 
  ∃ p g : ℤ, (500 * p + 350 * g = 50) ∧ ∀ D > 0, (∃ p g : ℤ, 500 * p + 350 * g = D) → 50 ≤ D :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_resolvable_debt_l720_72061


namespace NUMINAMATH_GPT_determine_m_l720_72077

theorem determine_m (x m : ℝ) (h₁ : 2 * x + m = 6) (h₂ : x = 2) : m = 2 := by
  sorry

end NUMINAMATH_GPT_determine_m_l720_72077


namespace NUMINAMATH_GPT_lucas_min_deliveries_l720_72020

theorem lucas_min_deliveries (cost_of_scooter earnings_per_delivery fuel_cost_per_delivery parking_fee_per_delivery : ℕ)
  (cost_eq : cost_of_scooter = 3000)
  (earnings_eq : earnings_per_delivery = 12)
  (fuel_cost_eq : fuel_cost_per_delivery = 4)
  (parking_fee_eq : parking_fee_per_delivery = 1) :
  ∃ d : ℕ, 7 * d ≥ cost_of_scooter ∧ d = 429 := by
  sorry

end NUMINAMATH_GPT_lucas_min_deliveries_l720_72020


namespace NUMINAMATH_GPT_common_difference_and_first_three_terms_l720_72055

-- Given condition that for any n, the sum of the first n terms of an arithmetic progression is equal to 5n^2.
def arithmetic_sum_property (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 5 * n ^ 2

-- Define the nth term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n-1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n_terms (a1 d n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d)/2

-- Conditions and prove that common difference d is 10 and the first three terms are 5, 15, and 25
theorem common_difference_and_first_three_terms :
  (∃ (a1 d : ℕ), arithmetic_sum_property (sum_first_n_terms a1 d) ∧ d = 10 ∧ nth_term a1 d 1 = 5 ∧ nth_term a1 d 2 = 15 ∧ nth_term a1 d 3  = 25) :=
sorry

end NUMINAMATH_GPT_common_difference_and_first_three_terms_l720_72055


namespace NUMINAMATH_GPT_solution_set_inequality_range_of_t_l720_72044

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem solution_set_inequality :
  {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 2} :=
sorry

theorem range_of_t (t : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f (x - t) ≤ x - 2) ↔ 3 ≤ t ∧ t ≤ 3 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_range_of_t_l720_72044


namespace NUMINAMATH_GPT_arithmetic_expression_proof_l720_72088

theorem arithmetic_expression_proof : 4 * 6 * 8 + 18 / 3 ^ 2 = 194 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_proof_l720_72088


namespace NUMINAMATH_GPT_other_root_eq_l720_72041

theorem other_root_eq (b : ℝ) : (∀ x, x^2 + b * x - 2 = 0 → (x = 1 ∨ x = -2)) :=
by
  intro x hx
  have : x = 1 ∨ x = -2 := sorry
  exact this

end NUMINAMATH_GPT_other_root_eq_l720_72041


namespace NUMINAMATH_GPT_squirrel_divides_acorns_l720_72054

theorem squirrel_divides_acorns (total_acorns parts_per_month remaining_acorns month_acorns winter_months spring_acorns : ℕ)
  (h1 : total_acorns = 210)
  (h2 : parts_per_month = 3)
  (h3 : winter_months = 3)
  (h4 : remaining_acorns = 60)
  (h5 : month_acorns = total_acorns / winter_months)
  (h6 : spring_acorns = 30)
  (h7 : month_acorns - remaining_acorns = spring_acorns / parts_per_month) :
  parts_per_month = 3 :=
by
  sorry

end NUMINAMATH_GPT_squirrel_divides_acorns_l720_72054


namespace NUMINAMATH_GPT_smallest_t_eq_3_over_4_l720_72051

theorem smallest_t_eq_3_over_4 (t : ℝ) :
  (∀ t : ℝ,
    (16 * t^3 - 49 * t^2 + 35 * t - 6) / (4 * t - 3) + 7 * t = 8 * t - 2 → t >= (3 / 4)) ∧
  (∃ t₀ : ℝ, (16 * t₀^3 - 49 * t₀^2 + 35 * t₀ - 6) / (4 * t₀ - 3) + 7 * t₀ = 8 * t₀ - 2 ∧ t₀ = (3 / 4)) :=
sorry

end NUMINAMATH_GPT_smallest_t_eq_3_over_4_l720_72051


namespace NUMINAMATH_GPT_set_union_example_l720_72030

theorem set_union_example (x : ℕ) (M N : Set ℕ) (h1 : M = {0, x}) (h2 : N = {1, 2}) (h3 : M ∩ N = {2}) :
  M ∪ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_set_union_example_l720_72030


namespace NUMINAMATH_GPT_infinite_coprime_pairs_divisibility_l720_72063

theorem infinite_coprime_pairs_divisibility :
  ∃ (S : ℕ → ℕ × ℕ), (∀ n, Nat.gcd (S n).1 (S n).2 = 1 ∧ (S n).1 ∣ (S n).2^2 - 5 ∧ (S n).2 ∣ (S n).1^2 - 5) ∧
  Function.Injective S :=
sorry

end NUMINAMATH_GPT_infinite_coprime_pairs_divisibility_l720_72063


namespace NUMINAMATH_GPT_minimum_flower_cost_l720_72099

def vertical_strip_width : ℝ := 3
def horizontal_strip_height : ℝ := 2
def bed_width : ℝ := 11
def bed_height : ℝ := 6

def easter_lily_cost : ℝ := 3
def dahlia_cost : ℝ := 2.5
def canna_cost : ℝ := 2

def vertical_strip_area : ℝ := vertical_strip_width * bed_height
def horizontal_strip_area : ℝ := horizontal_strip_height * bed_width
def overlap_area : ℝ := vertical_strip_width * horizontal_strip_height
def remaining_area : ℝ := (bed_width * bed_height) - vertical_strip_area - (horizontal_strip_area - overlap_area)

def easter_lily_area : ℝ := horizontal_strip_area - overlap_area
def dahlia_area : ℝ := vertical_strip_area
def canna_area : ℝ := remaining_area

def easter_lily_total_cost : ℝ := easter_lily_area * easter_lily_cost
def dahlia_total_cost : ℝ := dahlia_area * dahlia_cost
def canna_total_cost : ℝ := canna_area * canna_cost

def total_cost : ℝ := easter_lily_total_cost + dahlia_total_cost + canna_total_cost

theorem minimum_flower_cost : total_cost = 157 := by
  sorry

end NUMINAMATH_GPT_minimum_flower_cost_l720_72099


namespace NUMINAMATH_GPT_sum_first_five_terms_l720_72035

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 d : ℝ, ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Define the specific condition a_5 + a_8 - a_10 = 2
def specific_condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 8 - a 10 = 2

-- Define the sum of the first five terms S₅
def S5 (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 

-- The statement to be proved
theorem sum_first_five_terms (a : ℕ → ℝ) (h₁ : arithmetic_sequence a) (h₂ : specific_condition a) : 
  S5 a = 10 :=
sorry

end NUMINAMATH_GPT_sum_first_five_terms_l720_72035


namespace NUMINAMATH_GPT_hyperbola_equation_l720_72068

-- Define the conditions of the problem
def center_at_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0
def focus_on_y_axis (x : ℝ) : Prop := x = 0
def focal_distance (d : ℝ) : Prop := d = 4
def point_on_hyperbola (x y : ℝ) : Prop := x = 1 ∧ y = -Real.sqrt 3

-- Final statement to prove
theorem hyperbola_equation :
  (center_at_origin 0 0) ∧
  (focus_on_y_axis 0) ∧
  (focal_distance 4) ∧
  (point_on_hyperbola 1 (-Real.sqrt 3))
  → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a = Real.sqrt 3 ∧ b = 1) ∧ (∀ x y : ℝ, x^2 - (y^2 / 3) = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l720_72068


namespace NUMINAMATH_GPT_each_cow_gives_5_liters_per_day_l720_72033

-- Define conditions
def cows : ℕ := 52
def weekly_milk : ℕ := 1820
def days_in_week : ℕ := 7

-- Define daily_milk as the daily milk production
def daily_milk := weekly_milk / days_in_week

-- Define milk_per_cow as the amount of milk each cow produces per day
def milk_per_cow := daily_milk / cows

-- Statement to prove
theorem each_cow_gives_5_liters_per_day : milk_per_cow = 5 :=
by
  -- This is where you would normally fill in the proof steps
  sorry

end NUMINAMATH_GPT_each_cow_gives_5_liters_per_day_l720_72033


namespace NUMINAMATH_GPT_twenty_two_percent_of_three_hundred_l720_72028

theorem twenty_two_percent_of_three_hundred : 
  (22 / 100) * 300 = 66 :=
by
  sorry

end NUMINAMATH_GPT_twenty_two_percent_of_three_hundred_l720_72028


namespace NUMINAMATH_GPT_B_starts_after_A_l720_72017

theorem B_starts_after_A :
  ∀ (A_walk_speed B_cycle_speed dist_from_start t : ℝ), 
    A_walk_speed = 10 →
    B_cycle_speed = 20 →
    dist_from_start = 80 →
    B_cycle_speed * (dist_from_start - A_walk_speed * t) / A_walk_speed = t →
    t = 4 :=
by 
  intros A_walk_speed B_cycle_speed dist_from_start t hA_speed hB_speed hdist heq;
  sorry

end NUMINAMATH_GPT_B_starts_after_A_l720_72017


namespace NUMINAMATH_GPT_remainder_expression_mod_l720_72096

/-- 
Let the positive integers s, t, u, and v leave remainders of 6, 9, 13, and 17, respectively, 
when divided by 23. Also, let s > t > u > v.
We want to prove that the remainder when 2 * (s - t) - 3 * (t - u) + 4 * (u - v) is divided by 23 is 12.
-/
theorem remainder_expression_mod (s t u v : ℕ) (hs : s % 23 = 6) (ht : t % 23 = 9) (hu : u % 23 = 13) (hv : v % 23 = 17)
  (h_gt : s > t ∧ t > u ∧ u > v) : (2 * (s - t) - 3 * (t - u) + 4 * (u - v)) % 23 = 12 :=
by
  sorry

end NUMINAMATH_GPT_remainder_expression_mod_l720_72096


namespace NUMINAMATH_GPT_michael_height_l720_72079

theorem michael_height (flagpole_height flagpole_shadow michael_shadow : ℝ) 
                        (h1 : flagpole_height = 50) 
                        (h2 : flagpole_shadow = 25) 
                        (h3 : michael_shadow = 5) : 
                        (michael_shadow * (flagpole_height / flagpole_shadow) = 10) :=
by
  sorry

end NUMINAMATH_GPT_michael_height_l720_72079


namespace NUMINAMATH_GPT_binomial_coeff_equal_l720_72024

theorem binomial_coeff_equal (n : ℕ) (h₁ : 6 ≤ n) (h₂ : (n.choose 5) * 3^5 = (n.choose 6) * 3^6) :
  n = 7 := sorry

end NUMINAMATH_GPT_binomial_coeff_equal_l720_72024


namespace NUMINAMATH_GPT_point_P_in_fourth_quadrant_l720_72009

def point_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_P_in_fourth_quadrant (m : ℝ) : point_in_fourth_quadrant (1 + m^2) (-1) :=
by
  sorry

end NUMINAMATH_GPT_point_P_in_fourth_quadrant_l720_72009


namespace NUMINAMATH_GPT_mean_of_remaining_four_numbers_l720_72048

theorem mean_of_remaining_four_numbers 
  (a b c d max_num : ℝ) 
  (h1 : max_num = 105) 
  (h2 : (a + b + c + d + max_num) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.75 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_remaining_four_numbers_l720_72048


namespace NUMINAMATH_GPT_smallest_sum_of_sequence_l720_72018

theorem smallest_sum_of_sequence {
  A B C D k : ℕ
} (h1 : 2 * B = A + C)
  (h2 : D - C = (C - B) ^ 2)
  (h3 : 4 * B = 3 * C)
  (h4 : B = 3 * k)
  (h5 : C = 4 * k)
  (h6 : A = 2 * k)
  (h7 : D = 4 * k + k ^ 2) :
  A + B + C + D = 14 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_sequence_l720_72018


namespace NUMINAMATH_GPT_group_left_to_clean_is_third_group_l720_72011

-- Definition of group sizes
def group1 := 7
def group2 := 10
def group3 := 16
def group4 := 18

-- Definitions and conditions
def total_students := group1 + group2 + group3 + group4
def lecture_factor := 4
def english_students := 7  -- From solution: must be 7 students attending the English lecture
def math_students := lecture_factor * english_students

-- Hypothesis of the students allocating to lectures
def students_attending_lectures := english_students + math_students
def students_left_to_clean := total_students - students_attending_lectures

-- The statement to be proved in Lean
theorem group_left_to_clean_is_third_group
  (h : students_left_to_clean = group3) :
  students_left_to_clean = 16 :=
sorry

end NUMINAMATH_GPT_group_left_to_clean_is_third_group_l720_72011


namespace NUMINAMATH_GPT_negation_of_p_is_false_l720_72015

def prop_p : Prop :=
  ∀ x : ℝ, 1 < x → (Real.log (x + 2) / Real.log 3 - 2 / 2^x) > 0

theorem negation_of_p_is_false : ¬(∃ x : ℝ, 1 < x ∧ (Real.log (x + 2) / Real.log 3 - 2 / 2^x) ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_p_is_false_l720_72015
