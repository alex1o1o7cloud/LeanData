import Mathlib

namespace binom_20_10_eq_184756_l51_51325

theorem binom_20_10_eq_184756 
  (h1 : Nat.choose 19 9 = 92378)
  (h2 : Nat.choose 19 10 = Nat.choose 19 9) : 
  Nat.choose 20 10 = 184756 := 
by
  sorry

end binom_20_10_eq_184756_l51_51325


namespace difference_in_biking_distance_l51_51374

def biking_rate_alberto : ℕ := 18  -- miles per hour
def biking_rate_bjorn : ℕ := 20    -- miles per hour

def start_time_alberto : ℕ := 9    -- a.m.
def start_time_bjorn : ℕ := 10     -- a.m.

def end_time : ℕ := 15            -- 3 p.m. in 24-hour format

def biking_duration_alberto : ℕ := end_time - start_time_alberto
def biking_duration_bjorn : ℕ := end_time - start_time_bjorn

def distance_alberto : ℕ := biking_rate_alberto * biking_duration_alberto
def distance_bjorn : ℕ := biking_rate_bjorn * biking_duration_bjorn

theorem difference_in_biking_distance : 
  (distance_alberto - distance_bjorn) = 8 := by
  sorry

end difference_in_biking_distance_l51_51374


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l51_51854

noncomputable def arithmetic_sequence (a n d : ℝ) : ℝ := 
  a + (n - 1) * d

noncomputable def geometric_sequence_sum (b1 r n : ℝ) : ℝ := 
  b1 * (1 - r^n) / (1 - r)

theorem arithmetic_sequence_general_formula (a1 d : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) : 
  ∀ n, arithmetic_sequence a1 n d = (n + 1) / 2 :=
by 
  sorry

theorem geometric_sequence_sum_first_n_terms (a1 d b1 b4 : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) 
  (h3 : b1 = a1) (h4 : b4 = arithmetic_sequence a1 15 d) (h5 : b4 = 8) :
  ∀ n, geometric_sequence_sum b1 2 n = 2^n - 1 :=
by 
  sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l51_51854


namespace possible_values_x_l51_51267

-- Define the conditions
def gold_coin_worth (x y : ℕ) (g s : ℝ) : Prop :=
  g = (1 + x / 100.0) * s ∧ s = (1 - y / 100.0) * g

-- Define the main theorem statement
theorem possible_values_x : ∀ (x y : ℕ) (g s : ℝ), gold_coin_worth x y g s → 
  (∃ (n : ℕ), n = 12) :=
by
  -- Definitions based on given conditions
  intro x y g s h
  obtain ⟨hx, hy⟩ := h

  -- Placeholder for proof; skip with sorry
  sorry

end possible_values_x_l51_51267


namespace probability_bc_seated_next_l51_51434

theorem probability_bc_seated_next {P : ℝ} : 
  P = 2 / 3 :=
sorry

end probability_bc_seated_next_l51_51434


namespace find_a_l51_51691

-- Defining the curve y and its derivative y'
def y (x : ℝ) (a : ℝ) : ℝ := x^4 + a * x^2 + 1
def y' (x : ℝ) (a : ℝ) : ℝ := 4 * x^3 + 2 * a * x

theorem find_a (a : ℝ) : 
  y' (-1) a = 8 -> a = -6 := 
by
  -- proof here
  sorry

end find_a_l51_51691


namespace find_m_l51_51352

theorem find_m (x y m : ℤ) (h1 : 3 * x + 4 * y = 7) (h2 : 5 * x - 4 * y = m) (h3 : x + y = 0) : m = -63 := by
  sorry

end find_m_l51_51352


namespace S8_value_l51_51624

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

def condition_a3_a6 (a : ℕ → ℝ) : Prop :=
  a 3 = 9 - a 6

theorem S8_value (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_formula : sum_of_first_n_terms S a)
  (h_condition : condition_a3_a6 a) :
  S 8 = 72 :=
by
  sorry

end S8_value_l51_51624


namespace proposition_does_not_hold_at_2_l51_51734

variable (P : ℕ+ → Prop)
open Nat

theorem proposition_does_not_hold_at_2
  (h₁ : ¬ P 3)
  (h₂ : ∀ k : ℕ+, P k → P (k + 1)) :
  ¬ P 2 :=
by
  sorry

end proposition_does_not_hold_at_2_l51_51734


namespace circle_radius_l51_51873

theorem circle_radius (A : ℝ) (r : ℝ) (h : A = 36 * Real.pi) (h2 : A = Real.pi * r ^ 2) : r = 6 :=
sorry

end circle_radius_l51_51873


namespace line_parallel_to_parallel_set_l51_51470

variables {Point Line Plane : Type} 
variables (a : Line) (α : Plane)
variables (parallel : Line → Plane → Prop) (parallel_set : Line → Plane → Prop)

-- Definition for line parallel to plane
axiom line_parallel_to_plane : parallel a α

-- Goal: line a is parallel to a set of parallel lines within plane α
theorem line_parallel_to_parallel_set (h : parallel a α) : parallel_set a α := 
sorry

end line_parallel_to_parallel_set_l51_51470


namespace product_of_common_ratios_l51_51043

theorem product_of_common_ratios (x p r a2 a3 b2 b3 : ℝ)
  (h1 : a2 = x * p) (h2 : a3 = x * p^2)
  (h3 : b2 = x * r) (h4 : b3 = x * r^2)
  (h5 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2))
  (h_nonconstant : x ≠ 0) (h_diff_ratios : p ≠ r) :
  p * r = 9 :=
by
  sorry

end product_of_common_ratios_l51_51043


namespace megatek_manufacturing_percentage_l51_51508

-- Define the given conditions
def sector_deg : ℝ := 18
def full_circle_deg : ℝ := 360

-- Define the problem as a theorem statement in Lean
theorem megatek_manufacturing_percentage : 
  (sector_deg / full_circle_deg) * 100 = 5 := 
sorry

end megatek_manufacturing_percentage_l51_51508


namespace min_deg_g_correct_l51_51457

open Polynomial

noncomputable def min_deg_g {R : Type*} [CommRing R]
  (f g h : R[X])
  (hf : f.natDegree = 10)
  (hh : h.natDegree = 11)
  (h_eq : 5 * f + 6 * g = h) :
  Nat :=
11

theorem min_deg_g_correct {R : Type*} [CommRing R]
  (f g h : R[X])
  (hf : f.natDegree = 10)
  (hh : h.natDegree = 11)
  (h_eq : 5 * f + 6 * g = h) :
  (min_deg_g f g h hf hh h_eq = 11) :=
sorry

end min_deg_g_correct_l51_51457


namespace knife_value_l51_51411

def sheep_sold (n : ℕ) : ℕ := n * n

def valid_units_digits (m : ℕ) : Bool :=
  (m ^ 2 = 16) ∨ (m ^ 2 = 36)

theorem knife_value (n : ℕ) (k : ℕ) (m : ℕ) (H1 : sheep_sold n = n * n) (H2 : n = 10 * k + m) (H3 : valid_units_digits m = true) :
  2 = 2 :=
by
  sorry

end knife_value_l51_51411


namespace remainder_of_sum_is_12_l51_51134

theorem remainder_of_sum_is_12 (D k1 k2 : ℤ) (h1 : 242 = k1 * D + 4) (h2 : 698 = k2 * D + 8) : (242 + 698) % D = 12 :=
by
  sorry

end remainder_of_sum_is_12_l51_51134


namespace geom_sequence_sum_first_ten_terms_l51_51196

noncomputable def geom_sequence_sum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

theorem geom_sequence_sum_first_ten_terms (a : ℕ) (q : ℕ) (h1 : a * (1 + q) = 6) (h2 : a * q^3 * (1 + q) = 48) :
  geom_sequence_sum a q 10 = 2046 :=
sorry

end geom_sequence_sum_first_ten_terms_l51_51196


namespace ratio_of_tangent_to_circumference_l51_51039

theorem ratio_of_tangent_to_circumference
  {r x : ℝ}  -- radius of the circle and length of the tangent
  (hT : x = 2 * π * r)  -- given the length of tangent PQ
  (hA : (1 / 2) * x * r = π * r^2)  -- given the area equivalence

  : (x / (2 * π * r)) = 1 :=  -- desired ratio
by
  -- proof omitted, just using sorry to indicate proof
  sorry

end ratio_of_tangent_to_circumference_l51_51039


namespace no_primes_divisible_by_45_l51_51788

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_primes_divisible_by_45 : 
  ∀ p, is_prime p → ¬ (45 ∣ p) := 
by
  sorry

end no_primes_divisible_by_45_l51_51788


namespace time_per_window_l51_51740

-- Definitions of the given conditions
def total_windows : ℕ := 10
def installed_windows : ℕ := 6
def remaining_windows := total_windows - installed_windows
def total_hours : ℕ := 20
def hours_per_window := total_hours / remaining_windows

-- The theorem we need to prove
theorem time_per_window : hours_per_window = 5 := by
  -- This is where the proof would go
  sorry

end time_per_window_l51_51740


namespace maximize_profit_l51_51648

/-- 
The total number of rooms in the hotel 
-/
def totalRooms := 80

/-- 
The initial rent when the hotel is fully booked 
-/
def initialRent := 160

/-- 
The loss in guests for each increase in rent by 20 yuan 
-/
def guestLossPerIncrease := 3

/-- 
The increase in rent 
-/
def increasePer20Yuan := 20

/-- 
The daily service and maintenance cost per occupied room
-/
def costPerOccupiedRoom := 40

/-- 
Maximize profit given the conditions
-/
theorem maximize_profit : 
  ∃ x : ℕ, x = 360 ∧ 
            ∀ y : ℕ,
              (initialRent - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (x - initialRent) / increasePer20Yuan)
              ≥ (y - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (y - initialRent) / increasePer20Yuan) := 
sorry

end maximize_profit_l51_51648


namespace parity_of_expression_l51_51014

theorem parity_of_expression (o1 o2 n : ℕ) (h1 : o1 % 2 = 1) (h2 : o2 % 2 = 1) : 
  ((o1 * o1 + n * (o1 * o2)) % 2 = 1 ↔ n % 2 = 0) :=
by sorry

end parity_of_expression_l51_51014


namespace solve_for_y_l51_51359

theorem solve_for_y (x y : ℝ) (hx : x > 1) (hy : y > 1) (h1 : 1 / x + 1 / y = 1) (h2 : x * y = 9) :
  y = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end solve_for_y_l51_51359


namespace clear_time_is_approximately_7_point_1_seconds_l51_51403

-- Constants for the lengths of the trains in meters
def length_train1 : ℕ := 121
def length_train2 : ℕ := 165

-- Constants for the speeds of the trains in km/h
def speed_train1 : ℕ := 80
def speed_train2 : ℕ := 65

-- Kilometer to meter conversion
def km_to_meter (km : ℕ) : ℕ := km * 1000

-- Hour to second conversion
def hour_to_second (h : ℕ) : ℕ := h * 3600

-- Relative speed of the trains in meters per second
noncomputable def relative_speed_m_per_s : ℕ := 
  (km_to_meter (speed_train1 + speed_train2)) / hour_to_second 1

-- Total distance to be covered in meters
def total_distance : ℕ := length_train1 + length_train2

-- Time to be completely clear of each other in seconds
noncomputable def clear_time : ℝ := total_distance / (relative_speed_m_per_s : ℝ)

theorem clear_time_is_approximately_7_point_1_seconds :
  abs (clear_time - 7.1) < 0.01 :=
by
  sorry

end clear_time_is_approximately_7_point_1_seconds_l51_51403


namespace trucks_more_than_buses_l51_51777

theorem trucks_more_than_buses (b t : ℕ) (h₁ : b = 9) (h₂ : t = 17) : t - b = 8 :=
by
  sorry

end trucks_more_than_buses_l51_51777


namespace discount_correct_l51_51474

variable {a : ℝ} (discount_percent : ℝ) (profit_percent : ℝ → ℝ)

noncomputable def calc_discount : ℝ :=
  discount_percent

theorem discount_correct :
  (discount_percent / 100) = (33 + 1 / 3) / 100 →
  profit_percent (discount_percent / 100) = (3 / 2) * (discount_percent / 100) →
  a * (1 - discount_percent / 100) * (1 + profit_percent (discount_percent / 100)) = a →
  discount_percent = 33 + 1 / 3 :=
by sorry

end discount_correct_l51_51474


namespace blue_paint_needed_l51_51363

theorem blue_paint_needed (total_cans : ℕ) (blue_ratio : ℕ) (yellow_ratio : ℕ)
  (h_ratio: blue_ratio = 5) (h_yellow_ratio: yellow_ratio = 3) (h_total: total_cans = 45) : 
  ⌊total_cans * (blue_ratio : ℝ) / (blue_ratio + yellow_ratio)⌋ = 28 :=
by
  sorry

end blue_paint_needed_l51_51363


namespace range_of_a_l51_51464

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * Real.log x - a * x

theorem range_of_a (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, 0 < x → 0 ≤ 2 * x^2 - a * x + a) ↔ 0 < a ∧ a ≤ 8 :=
by
  sorry

end range_of_a_l51_51464


namespace coordinates_of_F_double_prime_l51_51236

-- Definitions of transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Definition of initial point F
def F : ℝ × ℝ := (1, 1)

-- Definition of the transformations applied to point F
def F_prime : ℝ × ℝ := reflect_x F
def F_double_prime : ℝ × ℝ := reflect_y_eq_x F_prime

-- Theorem statement
theorem coordinates_of_F_double_prime : F_double_prime = (-1, 1) :=
by
  sorry

end coordinates_of_F_double_prime_l51_51236


namespace total_distance_of_relay_race_l51_51895

theorem total_distance_of_relay_race 
    (fraction_siwon : ℝ := 3/10) 
    (fraction_dawon : ℝ := 4/10) 
    (distance_together : ℝ := 140) :
    (fraction_siwon + fraction_dawon) * 200 = distance_together :=
by
    sorry

end total_distance_of_relay_race_l51_51895


namespace vacation_cost_split_l51_51576

theorem vacation_cost_split (t d : ℕ) 
  (h_total : 105 + 125 + 175 = 405)
  (h_split : 405 / 3 = 135)
  (h_t : t = 135 - 105)
  (h_d : d = 135 - 125) : 
  t - d = 20 := by
  sorry

end vacation_cost_split_l51_51576


namespace middle_part_of_sum_is_120_l51_51926

theorem middle_part_of_sum_is_120 (x : ℚ) (h : 2 * x + x + (1 / 2) * x = 120) : 
  x = 240 / 7 := sorry

end middle_part_of_sum_is_120_l51_51926


namespace coeff_x3y2z5_in_expansion_l51_51972

def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3y2z5_in_expansion :
  let x := 1
  let y := 1
  let z := 1
  let x_term := 2 * x
  let y_term := y
  let z_term := z
  let target_term := x_term ^ 3 * y_term ^ 2 * z_term ^ 5
  let coeff := 2^3 * binomialCoeff 10 3 * binomialCoeff 7 2 * binomialCoeff 5 5
  coeff = 20160 :=
by
  sorry

end coeff_x3y2z5_in_expansion_l51_51972


namespace brown_ball_weight_l51_51219

def total_weight : ℝ := 9.12
def weight_blue : ℝ := 6
def weight_brown : ℝ := 3.12

theorem brown_ball_weight : total_weight - weight_blue = weight_brown :=
by 
  sorry

end brown_ball_weight_l51_51219


namespace independent_and_dependent_variables_l51_51197

variable (R V : ℝ)

theorem independent_and_dependent_variables (h : V = (4 / 3) * Real.pi * R^3) :
  (∃ R : ℝ, ∀ V : ℝ, V = (4 / 3) * Real.pi * R^3) ∧ (∃ V : ℝ, ∃ R' : ℝ, V = (4 / 3) * Real.pi * R'^3) :=
by
  sorry

end independent_and_dependent_variables_l51_51197


namespace ants_rice_transport_l51_51848

/-- 
Given:
  1) 12 ants can move 24 grains of rice in 6 trips.

Prove:
  How many grains of rice can 9 ants move in 9 trips?
-/
theorem ants_rice_transport :
  (9 * 9 * (24 / (12 * 6))) = 27 := 
sorry

end ants_rice_transport_l51_51848


namespace linear_function_solution_l51_51940

open Function

theorem linear_function_solution (f : ℝ → ℝ)
  (h_lin : ∃ k b, k ≠ 0 ∧ ∀ x, f x = k * x + b)
  (h_comp : ∀ x, f (f x) = 4 * x - 1) :
  (∀ x, f x = 2 * x - 1 / 3) ∨ (∀ x, f x = -2 * x + 1) :=
by
  sorry

end linear_function_solution_l51_51940


namespace range_of_a_l51_51574

variable (a : ℝ)
def A := Set.Ico (-2 : ℝ) 4
def B := {x : ℝ | x^2 - a * x - 4 ≤ 0 }

theorem range_of_a (h : B a ⊆ A) : 0 ≤ a ∧ a < 3 :=
by
  sorry

end range_of_a_l51_51574


namespace part1_part2_l51_51044

noncomputable def h (x : ℝ) : ℝ := x^2

noncomputable def phi (x : ℝ) : ℝ := 2 * Real.exp 1 * Real.log x

noncomputable def F (x : ℝ) : ℝ := h x - phi x

theorem part1 :
  ∃ (x : ℝ), x > 0 ∧ Real.log x = 1 ∧ F x = 0 :=
sorry

theorem part2 :
  ∃ (k b : ℝ), 
  (∀ x > 0, h x ≥ k * x + b) ∧
  (∀ x > 0, phi x ≤ k * x + b) ∧
  (k = 2 * Real.exp 1 ∧ b = -Real.exp 1) :=
sorry

end part1_part2_l51_51044


namespace arctan_sum_l51_51653

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l51_51653


namespace f_positive_l51_51013

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

variables (x0 x1 : ℝ)

theorem f_positive (hx0 : f x0 = 0) (hx1 : 0 < x1) (hx0_gt_x1 : x1 < x0) : 0 < f x1 :=
sorry

end f_positive_l51_51013


namespace line_slope_intercept_l51_51218

theorem line_slope_intercept (x y : ℝ) (k b : ℝ) (h : 3 * x + 4 * y + 5 = 0) :
  k = -3 / 4 ∧ b = -5 / 4 :=
by sorry

end line_slope_intercept_l51_51218


namespace triangle_angle_sum_l51_51026

theorem triangle_angle_sum (A B C : Type) (angle_ABC angle_BAC angle_ACB : ℝ)
  (h₁ : angle_ABC = 110)
  (h₂ : angle_BAC = 45)
  (triangle_sum : angle_ABC + angle_BAC + angle_ACB = 180) :
  angle_ACB = 25 :=
by
  sorry

end triangle_angle_sum_l51_51026


namespace annie_passes_bonnie_first_l51_51944

def bonnie_speed (v : ℝ) := v
def annie_speed (v : ℝ) := 1.3 * v
def track_length := 500

theorem annie_passes_bonnie_first (v t : ℝ) (ht : 0.3 * v * t = track_length) : 
  (annie_speed v * t) / track_length = 4 + 1 / 3 :=
by 
  sorry

end annie_passes_bonnie_first_l51_51944


namespace find_required_school_year_hours_l51_51891

-- Define constants for the problem
def summer_hours_per_week : ℕ := 40
def summer_weeks : ℕ := 12
def summer_earnings : ℕ := 6000
def school_year_weeks : ℕ := 36
def school_year_earnings : ℕ := 9000

-- Calculate total summer hours, hourly rate, total school year hours, and required school year weekly hours
def total_summer_hours := summer_hours_per_week * summer_weeks
def hourly_rate := summer_earnings / total_summer_hours
def total_school_year_hours := school_year_earnings / hourly_rate
def required_school_year_hours_per_week := total_school_year_hours / school_year_weeks

-- Prove the required hours per week is 20
theorem find_required_school_year_hours : required_school_year_hours_per_week = 20 := by
  sorry

end find_required_school_year_hours_l51_51891


namespace tray_contains_40_brownies_l51_51701

-- Definitions based on conditions
def tray_length : ℝ := 24
def tray_width : ℝ := 15
def brownie_length : ℝ := 3
def brownie_width : ℝ := 3

-- The mathematical statement to prove
theorem tray_contains_40_brownies :
  (tray_length * tray_width) / (brownie_length * brownie_width) = 40 :=
by
  sorry

end tray_contains_40_brownies_l51_51701


namespace intersection_complement_l51_51277

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 3, 4})
variable (hB : B = {4, 5})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  rw [hU, hA, hB]
  ext
  simp
  sorry

end intersection_complement_l51_51277


namespace sufficient_not_necessary_l51_51048

variable (p q : Prop)

theorem sufficient_not_necessary (h1 : p ∧ q) (h2 : ¬¬p) : ¬¬p :=
by
  sorry

end sufficient_not_necessary_l51_51048


namespace brad_has_9_green_balloons_l51_51563

theorem brad_has_9_green_balloons (total_balloons red_balloons : ℕ) (h_total : total_balloons = 17) (h_red : red_balloons = 8) : total_balloons - red_balloons = 9 :=
by {
  sorry
}

end brad_has_9_green_balloons_l51_51563


namespace johnny_practice_l51_51184

variable (P : ℕ) -- Current amount of practice in days
variable (h : P = 40) -- Given condition translating Johnny's practice amount
variable (d : ℕ) -- Additional days needed

theorem johnny_practice : d = 80 :=
by
  have goal : 3 * P = P + d := sorry
  have initial_condition : P = 40 := sorry
  have required : d = 3 * 40 - 40 := sorry
  sorry

end johnny_practice_l51_51184


namespace find_ab_l51_51566

theorem find_ab (a b : ℕ) (h1 : 1 <= a) (h2 : a < 10) (h3 : 0 <= b) (h4 : b < 10) (h5 : 66 * ((1 : ℝ) + ((10 * a + b : ℕ) / 100) - (↑(10 * a + b) / 99)) = 0.5) : 10 * a + b = 75 :=
by
  sorry

end find_ab_l51_51566


namespace x_intercept_of_l1_is_2_l51_51669

theorem x_intercept_of_l1_is_2 (a : ℝ) (l1_perpendicular_l2 : ∀ (x y : ℝ), 
  ((a+3)*x + y - 4 = 0) -> (x + (a-1)*y + 4 = 0) -> False) : 
  ∃ b : ℝ, (2*b + 0 - 4 = 0) ∧ b = 2 := 
by
  sorry

end x_intercept_of_l1_is_2_l51_51669


namespace prime_divisor_congruent_one_mod_p_l51_51942

theorem prime_divisor_congruent_one_mod_p (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ q ∣ p^p - 1 ∧ q % p = 1 :=
sorry

end prime_divisor_congruent_one_mod_p_l51_51942


namespace complement_of_intersection_l51_51023

def S : Set ℝ := {-2, -1, 0, 1, 2}
def T : Set ℝ := {x | x + 1 ≤ 2}
def complement (A B : Set ℝ) : Set ℝ := {x ∈ B | x ∉ A}

theorem complement_of_intersection :
  complement (S ∩ T) S = {2} :=
by
  sorry

end complement_of_intersection_l51_51023


namespace base3_composite_numbers_l51_51785

theorem base3_composite_numbers:
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 12002110 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2210121012 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 121212 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 102102 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 1001 * AB = a * b) :=
by {
  sorry
}

end base3_composite_numbers_l51_51785


namespace Deepak_and_Wife_meet_time_l51_51861

theorem Deepak_and_Wife_meet_time 
    (circumference : ℕ) 
    (Deepak_speed : ℕ)
    (wife_speed : ℕ) 
    (conversion_factor_km_hr_to_m_hr : ℕ) 
    (minutes_per_hour : ℕ) :
    circumference = 726 →
    Deepak_speed = 4500 →  -- speed in meters per hour
    wife_speed = 3750 →  -- speed in meters per hour
    conversion_factor_km_hr_to_m_hr = 1000 →
    minutes_per_hour = 60 →
    (726 / ((4500 + 3750) / 1000) * 60 = 5.28) :=
by 
    sorry

end Deepak_and_Wife_meet_time_l51_51861


namespace overtakes_in_16_minutes_l51_51045

def number_of_overtakes (track_length : ℕ) (speed_a : ℕ) (speed_b : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let relative_speed := speed_a - speed_b
  let time_per_overtake := track_length / relative_speed
  time_seconds / time_per_overtake

theorem overtakes_in_16_minutes :
  number_of_overtakes 200 6 4 16 = 9 :=
by
  -- We will insert calculations or detailed proof steps if needed
  sorry

end overtakes_in_16_minutes_l51_51045


namespace numberOfZeros_l51_51224

noncomputable def g (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem numberOfZeros :
  ∃ x ∈ Set.Ioo 1 (Real.exp Real.pi), g x = 0 ∧ ∀ y ∈ Set.Ioo 1 (Real.exp Real.pi), g y = 0 → y = x := 
sorry

end numberOfZeros_l51_51224


namespace number_of_teachers_l51_51326

theorem number_of_teachers (total_population sample_size teachers_within_sample students_within_sample : ℕ) 
    (h_total_population : total_population = 3000) 
    (h_sample_size : sample_size = 150) 
    (h_students_within_sample : students_within_sample = 140) 
    (h_teachers_within_sample : teachers_within_sample = sample_size - students_within_sample) 
    (h_ratio : (total_population - students_within_sample) * sample_size = total_population * teachers_within_sample) : 
    total_population - students_within_sample = 200 :=
by {
  sorry
}

end number_of_teachers_l51_51326


namespace solve_for_x_l51_51418

noncomputable def solution_x : ℝ := -1011.5

theorem solve_for_x (x : ℝ) (h : (2023 + x)^2 = x^2) : x = solution_x :=
by sorry

end solve_for_x_l51_51418


namespace ratio_length_breadth_l51_51139

noncomputable def b : ℝ := 18
noncomputable def l : ℝ := 972 / b

theorem ratio_length_breadth
  (A : ℝ)
  (h1 : b = 18)
  (h2 : l * b = 972) :
  (l / b) = 3 :=
by
  sorry

end ratio_length_breadth_l51_51139


namespace lisa_speed_l51_51859

-- Define conditions
def distance : ℕ := 256
def time : ℕ := 8

-- Define the speed calculation theorem
theorem lisa_speed : (distance / time) = 32 := 
by {
  sorry
}

end lisa_speed_l51_51859


namespace total_sleep_correct_l51_51887

namespace SleepProblem

def recommended_sleep_per_day : ℝ := 8
def sleep_days_part1 : ℕ := 2
def sleep_hours_part1 : ℝ := 3
def days_in_week : ℕ := 7
def remaining_days := days_in_week - sleep_days_part1
def percentage_sleep : ℝ := 0.6
def sleep_per_remaining_day := recommended_sleep_per_day * percentage_sleep

theorem total_sleep_correct (h1 : 2 * sleep_hours_part1 = 6)
                            (h2 : remaining_days = 5)
                            (h3 : sleep_per_remaining_day = 4.8)
                            (h4 : remaining_days * sleep_per_remaining_day = 24) :
  2 * sleep_hours_part1 + remaining_days * sleep_per_remaining_day = 30 := by
  sorry

end SleepProblem

end total_sleep_correct_l51_51887


namespace cost_price_computer_table_l51_51733

noncomputable def approx_eq (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

theorem cost_price_computer_table (SP : ℝ) (CP : ℝ) (h : SP = 7967) (h2 : SP = 1.24 * CP) : 
  approx_eq CP 6424 0.01 :=
by
  sorry

end cost_price_computer_table_l51_51733


namespace discount_is_25_l51_51771

def original_price : ℕ := 76
def discounted_price : ℕ := 51
def discount_amount : ℕ := original_price - discounted_price

theorem discount_is_25 : discount_amount = 25 := by
  sorry

end discount_is_25_l51_51771


namespace purple_shoes_count_l51_51070

-- Define the conditions
def total_shoes : ℕ := 1250
def blue_shoes : ℕ := 540
def remaining_shoes : ℕ := total_shoes - blue_shoes
def green_shoes := remaining_shoes / 2
def purple_shoes := green_shoes

-- State the theorem to be proven
theorem purple_shoes_count : purple_shoes = 355 := 
by
-- Proof can be filled in here (not needed for the task)
sorry

end purple_shoes_count_l51_51070


namespace no_int_solutions_a_b_l51_51036

theorem no_int_solutions_a_b :
  ¬ ∃ (a b : ℤ), a^2 + 1998 = b^2 :=
by
  sorry

end no_int_solutions_a_b_l51_51036


namespace solve_for_a_l51_51689

theorem solve_for_a (a b : ℝ) (h₁ : b = 4 * a) (h₂ : b = 20 - 7 * a) : a = 20 / 11 :=
by
  sorry

end solve_for_a_l51_51689


namespace solve_for_x_l51_51551

theorem solve_for_x (x : ℝ) (h : (x+10) / (x-4) = (x-3) / (x+6)) : x = -48 / 23 :=
by
  sorry

end solve_for_x_l51_51551


namespace audio_space_per_hour_l51_51179

/-
The digital music library holds 15 days of music.
The library occupies 20,000 megabytes of disk space.
The library contains both audio and video files.
Video files take up twice as much space per hour as audio files.
There is an equal number of hours for audio and video.
-/

theorem audio_space_per_hour (total_days : ℕ) (total_space : ℕ) (equal_hours : Prop) (video_space : ℕ → ℕ) 
  (H1 : total_days = 15)
  (H2 : total_space = 20000)
  (H3 : equal_hours)
  (H4 : ∀ x, video_space x = 2 * x) :
  ∃ x, x = 37 :=
by
  sorry

end audio_space_per_hour_l51_51179


namespace smaller_cube_edge_length_l51_51581

theorem smaller_cube_edge_length (x : ℝ) 
    (original_edge_length : ℝ := 7)
    (increase_percentage : ℝ := 600) 
    (original_surface_area_formula : ℝ := 6 * original_edge_length^2)
    (new_surface_area_formula : ℝ := (1 + increase_percentage / 100) * original_surface_area_formula) :
  ∃ x : ℝ, 6 * x^2 * (original_edge_length ^ 3 / x ^ 3) = new_surface_area_formula → x = 1 := by
  sorry

end smaller_cube_edge_length_l51_51581


namespace ice_cube_count_l51_51476

theorem ice_cube_count (cubes_per_tray : ℕ) (tray_count : ℕ) (H1: cubes_per_tray = 9) (H2: tray_count = 8) :
  cubes_per_tray * tray_count = 72 :=
by
  sorry

end ice_cube_count_l51_51476


namespace sphere_surface_area_l51_51868

theorem sphere_surface_area (a : ℝ) (d : ℝ) (S : ℝ) : 
  a = 3 → d = Real.sqrt 7 → S = 40 * Real.pi := by
  sorry

end sphere_surface_area_l51_51868


namespace proof1_proof2_proof3_proof4_l51_51537

noncomputable def calc1 : ℝ := 3.21 - 1.05 - 1.95
noncomputable def calc2 : ℝ := 15 - (2.95 + 8.37)
noncomputable def calc3 : ℝ := 14.6 * 2 - 0.6 * 2
noncomputable def calc4 : ℝ := 0.25 * 1.25 * 32

theorem proof1 : calc1 = 0.21 := by
  sorry

theorem proof2 : calc2 = 3.68 := by
  sorry

theorem proof3 : calc3 = 28 := by
  sorry

theorem proof4 : calc4 = 10 := by
  sorry

end proof1_proof2_proof3_proof4_l51_51537


namespace math_problem_l51_51923

theorem math_problem :
  ((-1)^2023 - (27^(1/3)) - (16^(1/2)) + (|1 - Real.sqrt 3|)) = -9 + Real.sqrt 3 :=
by
  sorry

end math_problem_l51_51923


namespace pre_bought_ticket_price_l51_51993

variable (P : ℕ)

theorem pre_bought_ticket_price :
  (20 * P = 6000 - 2900) → P = 155 :=
by
  intro h
  sorry

end pre_bought_ticket_price_l51_51993


namespace sum_of_first_six_terms_of_geom_seq_l51_51408

theorem sum_of_first_six_terms_of_geom_seq :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let S6 := a * (1 - r^6) / (1 - r)
  S6 = 4095 / 12288 := by
sorry

end sum_of_first_six_terms_of_geom_seq_l51_51408


namespace combine_like_terms_l51_51025

variable (a : ℝ)

theorem combine_like_terms : 3 * a^2 + 5 * a^2 - a^2 = 7 * a^2 := 
by sorry

end combine_like_terms_l51_51025


namespace root_condition_l51_51008

noncomputable def f (a : ℝ) (x : ℝ) := a * x^3 - 3 * x^2 + 1

theorem root_condition (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 0 ∧ ∀ x ≠ x₀, f a x ≠ 0 ∧ x₀ < 0) → a > 2 :=
sorry

end root_condition_l51_51008


namespace find_b_l51_51384

theorem find_b (b c : ℝ) : 
  (-11 : ℝ) = (-1)^2 + (-1) * b + c ∧ 
  17 = 3^2 + 3 * b + c ∧ 
  6 = 2^2 + 2 * b + c → 
  b = 14 / 3 :=
by
  sorry

end find_b_l51_51384


namespace koschei_coin_count_l51_51128

theorem koschei_coin_count (a : ℕ) :
  (a % 10 = 7) ∧
  (a % 12 = 9) ∧
  (300 ≤ a ∧ a ≤ 400) →
  a = 357 :=
sorry

end koschei_coin_count_l51_51128


namespace new_average_score_l51_51636

theorem new_average_score (n : ℕ) (initial_avg : ℕ) (grace_marks : ℕ) (h1 : n = 35) (h2 : initial_avg = 37) (h3 : grace_marks = 3) : initial_avg + grace_marks = 40 := by
  sorry

end new_average_score_l51_51636


namespace total_difference_proof_l51_51523

-- Definitions for the initial quantities
def initial_tomatoes : ℕ := 17
def initial_carrots : ℕ := 13
def initial_cucumbers : ℕ := 8

-- Definitions for the picked quantities
def picked_tomatoes : ℕ := 5
def picked_carrots : ℕ := 6

-- Definitions for the given away quantities
def given_away_tomatoes : ℕ := 3
def given_away_carrots : ℕ := 2

-- Definitions for the remaining quantities 
def remaining_tomatoes : ℕ := initial_tomatoes - (picked_tomatoes - given_away_tomatoes)
def remaining_carrots : ℕ := initial_carrots - (picked_carrots - given_away_carrots)

-- Definitions for the difference quantities
def difference_tomatoes : ℕ := initial_tomatoes - remaining_tomatoes
def difference_carrots : ℕ := initial_carrots - remaining_carrots

-- Definition for the total difference
def total_difference : ℕ := difference_tomatoes + difference_carrots

-- Lean Theorem Statement
theorem total_difference_proof : total_difference = 6 := by
  -- Proof is omitted
  sorry

end total_difference_proof_l51_51523


namespace initial_number_of_girls_l51_51141

theorem initial_number_of_girls (n A : ℕ) (new_girl_weight : ℕ := 80) (original_girl_weight : ℕ := 40)
  (avg_increase : ℕ := 2)
  (condition : n * (A + avg_increase) - n * A = 40) :
  n = 20 :=
by
  sorry

end initial_number_of_girls_l51_51141


namespace burger_share_per_person_l51_51821

-- Definitions based on conditions
def foot_to_inches : ℕ := 12
def burger_length_foot : ℕ := 1
def burger_length_inches : ℕ := burger_length_foot * foot_to_inches

theorem burger_share_per_person : (burger_length_inches / 2) = 6 := by
  sorry

end burger_share_per_person_l51_51821


namespace no_solution_for_x_l51_51685

theorem no_solution_for_x (a : ℝ) (h : a ≤ 8) : ¬ ∃ x : ℝ, |x - 5| + |x + 3| < a :=
by
  sorry

end no_solution_for_x_l51_51685


namespace cindy_added_pens_l51_51307

-- Definitions based on conditions:
def initial_pens : ℕ := 20
def mike_pens : ℕ := 22
def sharon_pens : ℕ := 19
def final_pens : ℕ := 65

-- Intermediate calculations:
def pens_after_mike : ℕ := initial_pens + mike_pens
def pens_after_sharon : ℕ := pens_after_mike - sharon_pens

-- Proof statement:
theorem cindy_added_pens : pens_after_sharon + 42 = final_pens :=
by
  sorry

end cindy_added_pens_l51_51307


namespace main_problem_l51_51210

-- Define the set A
def A (a : ℝ) : Set ℝ :=
  {0, 1, a^2 - 2 * a}

-- Define the main problem as a theorem
theorem main_problem (a : ℝ) (h : a ∈ A a) : a = 1 ∨ a = 3 :=
  sorry

end main_problem_l51_51210


namespace trailing_zeros_a6_l51_51881

theorem trailing_zeros_a6:
  (∃ a : ℕ+ → ℚ, 
    a 1 = 3 / 2 ∧ 
    (∀ n : ℕ+, a (n + 1) = (1 / 2) * (a n + (1 / a n))) ∧
    (∃ k, 10^k ≤ a 6 ∧ a 6 < 10^(k + 1))) →
  (∃ m, m = 22) :=
sorry

end trailing_zeros_a6_l51_51881


namespace find_integer_l51_51608

theorem find_integer (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.cos (n * Real.pi / 180) = Real.sin (312 * Real.pi / 180)) :
  n = 42 :=
by
  sorry

end find_integer_l51_51608


namespace smallest_m_for_divisibility_l51_51697

theorem smallest_m_for_divisibility : 
  ∃ (m : ℕ), 2^1990 ∣ 1989^m - 1 ∧ m = 2^1988 := 
sorry

end smallest_m_for_divisibility_l51_51697


namespace alice_twice_bob_in_some_years_l51_51982

def alice_age (B : ℕ) : ℕ := B + 10
def future_age_condition (A : ℕ) : Prop := A + 5 = 19
def twice_as_old_condition (A B x : ℕ) : Prop := A + x = 2 * (B + x)

theorem alice_twice_bob_in_some_years :
  ∃ x, ∀ A B,
  alice_age B = A →
  future_age_condition A →
  twice_as_old_condition A B x := by
  sorry

end alice_twice_bob_in_some_years_l51_51982


namespace bruce_total_payment_l51_51718

def grapes_quantity : ℕ := 8
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55

def cost_grapes : ℕ := grapes_quantity * grapes_rate
def cost_mangoes : ℕ := mangoes_quantity * mangoes_rate
def total_cost : ℕ := cost_grapes + cost_mangoes

theorem bruce_total_payment : total_cost = 1055 := by
  sorry

end bruce_total_payment_l51_51718


namespace no_month_5_mondays_and_5_thursdays_l51_51951

theorem no_month_5_mondays_and_5_thursdays (n : ℕ) (h : n = 28 ∨ n = 29 ∨ n = 30 ∨ n = 31) :
  ¬ (∃ (m : ℕ) (t : ℕ), m = 5 ∧ t = 5 ∧ 5 * (m + t) ≤ n) := by sorry

end no_month_5_mondays_and_5_thursdays_l51_51951


namespace distribution_schemes_l51_51710

theorem distribution_schemes 
    (total_professors : ℕ)
    (high_schools : Finset ℕ) 
    (A : ℕ) 
    (B : ℕ) 
    (C : ℕ)
    (D : ℕ)
    (cond1 : total_professors = 6) 
    (cond2 : A = 1)
    (cond3 : B ≥ 1)
    (cond4 : C ≥ 1)
    (D' := (total_professors - A - B - C)) 
    (cond5 : D' ≥ 1) : 
    ∃ N : ℕ, N = 900 := by
  sorry

end distribution_schemes_l51_51710


namespace greater_number_l51_51879

theorem greater_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : x = 35 := 
by sorry

end greater_number_l51_51879


namespace solution_inequality_1_range_of_a_l51_51057

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 2)

theorem solution_inequality_1 :
  {x : ℝ | f x < 3} = {x : ℝ | - (1/2) < x ∧ x < (5/2)} :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x < a) → a > 2 :=
by
  sorry

end solution_inequality_1_range_of_a_l51_51057


namespace product_of_decimals_l51_51694

def x : ℝ := 0.8
def y : ℝ := 0.12

theorem product_of_decimals : x * y = 0.096 :=
by
  sorry

end product_of_decimals_l51_51694


namespace inequality_interval_l51_51726

theorem inequality_interval : ∀ x : ℝ, (x^2 - 3 * x - 4 < 0) ↔ (-1 < x ∧ x < 4) :=
by
  intro x
  sorry

end inequality_interval_l51_51726


namespace no_real_solutions_for_identical_lines_l51_51414

theorem no_real_solutions_for_identical_lines :
  ¬∃ (a d : ℝ), (∀ x y : ℝ, 5 * x + a * y + d = 0 ↔ 2 * d * x - 3 * y + 8 = 0) :=
by
  sorry

end no_real_solutions_for_identical_lines_l51_51414


namespace sin_593_l51_51094

theorem sin_593 (h : Real.sin (37 * Real.pi / 180) = 3/5) : 
  Real.sin (593 * Real.pi / 180) = -3/5 :=
by
sorry

end sin_593_l51_51094


namespace remainder_of_poly_div_l51_51865

theorem remainder_of_poly_div (x : ℤ) : 
  (x + 1)^2009 % (x^2 + x + 1) = x + 1 :=
by
  sorry

end remainder_of_poly_div_l51_51865


namespace statement_C_l51_51811

variables (a b c d : ℝ)

theorem statement_C (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end statement_C_l51_51811


namespace find_positive_n_l51_51130

def consecutive_product (k : ℕ) : ℕ := k * (k + 1) * (k + 2)

theorem find_positive_n (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  n^6 + 5*n^3 + 4*n + 116 = consecutive_product k ↔ n = 3 := 
by 
  sorry

end find_positive_n_l51_51130


namespace purchasing_plans_count_l51_51803

theorem purchasing_plans_count :
  ∃ n : ℕ, n = 2 ∧ (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = 35) :=
sorry

end purchasing_plans_count_l51_51803


namespace arithmetic_sequence_26th_term_l51_51259

theorem arithmetic_sequence_26th_term (a d : ℤ) (h1 : a = 3) (h2 : a + d = 13) (h3 : a + 2 * d = 23) : 
  a + 25 * d = 253 :=
by
  -- specifications for variables a, d, and hypotheses h1, h2, h3
  sorry

end arithmetic_sequence_26th_term_l51_51259


namespace smallest_k_remainder_1_l51_51321

theorem smallest_k_remainder_1
  (k : ℤ) : 
  (k > 1) ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 4 = 1)
  ↔ k = 105 :=
by
  sorry

end smallest_k_remainder_1_l51_51321


namespace weight_left_after_two_deliveries_l51_51953

-- Definitions and conditions
def initial_load : ℝ := 50000
def first_store_percentage : ℝ := 0.1
def second_store_percentage : ℝ := 0.2

-- The statement to be proven
theorem weight_left_after_two_deliveries :
  let weight_after_first_store := initial_load * (1 - first_store_percentage)
  let weight_after_second_store := weight_after_first_store * (1 - second_store_percentage)
  weight_after_second_store = 36000 :=
by sorry  -- Proof omitted

end weight_left_after_two_deliveries_l51_51953


namespace sqrt_mixed_number_l51_51959

theorem sqrt_mixed_number :
  (Real.sqrt (8 + 9/16)) = (Real.sqrt 137) / 4 :=
by
  sorry

end sqrt_mixed_number_l51_51959


namespace trajectory_of_M_l51_51998

theorem trajectory_of_M {x y x₀ y₀ : ℝ} (P_on_parabola : x₀^2 = 2 * y₀)
(line_PQ_perpendicular : ∀ Q : ℝ, true)
(vector_PM_PQ_relation : x₀ = x ∧ y₀ = 2 * y) :
  x^2 = 4 * y := by
  sorry

end trajectory_of_M_l51_51998


namespace first_instance_height_35_l51_51461
noncomputable def projectile_height (t : ℝ) : ℝ := -5 * t^2 + 30 * t

theorem first_instance_height_35 {t : ℝ} (h : projectile_height t = 35) :
  t = 3 - Real.sqrt 2 :=
sorry

end first_instance_height_35_l51_51461


namespace find_x_l51_51619

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x^2 + 18 * x * y = x^3 + 3 * x^2 * y + 6 * x) : 
  x = 3 :=
sorry

end find_x_l51_51619


namespace sum_digits_of_three_digit_numbers_l51_51721

theorem sum_digits_of_three_digit_numbers (a c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hc : 1 ≤ c ∧ c < 10) 
  (h1 : (300 + 10 * a + 7) + 414 = 700 + 10 * c + 1)
  (h2 : ∃ k : ℤ, 700 + 10 * c + 1 = 11 * k) :
  a + c = 14 :=
by
  sorry

end sum_digits_of_three_digit_numbers_l51_51721


namespace total_people_in_group_l51_51794

theorem total_people_in_group (men women children : ℕ)
  (h1 : men = 2 * women)
  (h2 : women = 3 * children)
  (h3 : children = 30) :
  men + women + children = 300 :=
by
  sorry

end total_people_in_group_l51_51794


namespace graph_of_4x2_minus_9y2_is_pair_of_straight_lines_l51_51274

theorem graph_of_4x2_minus_9y2_is_pair_of_straight_lines :
  (∀ x y : ℝ, (4 * x^2 - 9 * y^2 = 0) → (x / y = 3 / 2 ∨ x / y = -3 / 2)) :=
by
  sorry

end graph_of_4x2_minus_9y2_is_pair_of_straight_lines_l51_51274


namespace radius_of_sphere_find_x_for_equation_l51_51853

-- Problem I2.1
theorem radius_of_sphere (r : ℝ) (V : ℝ) (h : V = 36 * π) : r = 3 :=
sorry

-- Problem I2.2
theorem find_x_for_equation (x : ℝ) (r : ℝ) (h_r : r = 3) (h : r^x + r^(1-x) = 4) (h_x_pos : x > 0) : x = 1 :=
sorry

end radius_of_sphere_find_x_for_equation_l51_51853


namespace kayla_total_items_l51_51001

theorem kayla_total_items (Tc : ℕ) (Ts : ℕ) (Kc : ℕ) (Ks : ℕ) 
  (h1 : Tc = 2 * Kc) (h2 : Ts = 2 * Ks) (h3 : Tc = 12) (h4 : Ts = 18) : Kc + Ks = 15 :=
by
  sorry

end kayla_total_items_l51_51001


namespace inscribed_rectangle_area_l51_51910

variable (a b h x : ℝ)
variable (h_pos : 0 < h) (a_b_pos : a > b) (b_pos : b > 0) (a_pos : a > 0) (x_pos : 0 < x) (hx : x < h)

theorem inscribed_rectangle_area (hb : b > 0) (ha : a > 0) (hx : 0 < x) (hxa : x < h) : 
  x * (a - b) * (h - x) / h = x * (a - b) * (h - x) / h := by
  sorry

end inscribed_rectangle_area_l51_51910


namespace multiple_optimal_solutions_for_z_l51_51430

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 0 2
def B := Point.mk (-2) (-2)
def C := Point.mk 2 0

def z (a : ℝ) (P : Point) : ℝ := P.y - a * P.x

def maxz_mult_opt_solutions (a : ℝ) : Prop :=
  z a A = z a B ∨ z a A = z a C ∨ z a B = z a C

theorem multiple_optimal_solutions_for_z :
  (maxz_mult_opt_solutions (-1)) ∧ (maxz_mult_opt_solutions 2) :=
by
  sorry

end multiple_optimal_solutions_for_z_l51_51430


namespace count_unbroken_matches_l51_51746

theorem count_unbroken_matches :
  let n_1 := 5 * 12  -- number of boxes in the first set
  let matches_1 := n_1 * 20  -- total matches in first set of boxes
  let broken_1 := n_1 * 3  -- total broken matches in first set of boxes
  let unbroken_1 := matches_1 - broken_1  -- unbroken matches in first set of boxes

  let n_2 := 4  -- number of extra boxes
  let matches_2 := n_2 * 25  -- total matches in extra boxes
  let broken_2 := (matches_2 / 5)  -- total broken matches in extra boxes (20%)
  let unbroken_2 := matches_2 - broken_2  -- unbroken matches in extra boxes

  let total_unbroken := unbroken_1 + unbroken_2  -- total unbroken matches

  total_unbroken = 1100 := 
by
  sorry

end count_unbroken_matches_l51_51746


namespace find_third_divisor_l51_51633

theorem find_third_divisor (n : ℕ) (d : ℕ) 
  (h1 : (n - 4) % 12 = 0)
  (h2 : (n - 4) % 16 = 0)
  (h3 : (n - 4) % d = 0)
  (h4 : (n - 4) % 21 = 0)
  (h5 : (n - 4) % 28 = 0)
  (h6 : n = 1012) :
  d = 3 :=
by
  sorry

end find_third_divisor_l51_51633


namespace distance_city_A_to_C_l51_51329

variable (V_E V_F : ℝ) -- Define the average speeds of Eddy and Freddy
variable (time : ℝ) -- Define the time variable

-- Given conditions
def eddy_time : time = 3 := sorry
def freddy_time : time = 3 := sorry
def eddy_distance : ℝ := 600
def speed_ratio : V_E = 2 * V_F := sorry

-- Derived condition for Eddy's speed
def eddy_speed : V_E = eddy_distance / time := sorry

-- Derived conclusion for Freddy's distance
theorem distance_city_A_to_C (time : ℝ) (V_F : ℝ) : V_F * time = 300 := 
by 
  sorry

end distance_city_A_to_C_l51_51329


namespace rainfall_on_Monday_l51_51495

theorem rainfall_on_Monday (rain_on_Tuesday : ℝ) (difference : ℝ) (rain_on_Tuesday_eq : rain_on_Tuesday = 0.2) (difference_eq : difference = 0.7) :
  ∃ rain_on_Monday : ℝ, rain_on_Monday = rain_on_Tuesday + difference := 
sorry

end rainfall_on_Monday_l51_51495


namespace gain_percentage_is_twenty_l51_51113

theorem gain_percentage_is_twenty (SP CP Gain : ℝ) (h0 : SP = 90) (h1 : Gain = 15) (h2 : SP = CP + Gain) : (Gain / CP) * 100 = 20 :=
by
  sorry

end gain_percentage_is_twenty_l51_51113


namespace minute_hand_length_l51_51438

theorem minute_hand_length (r : ℝ) (h : 20 * (2 * Real.pi / 60) * r = Real.pi / 3) : r = 1 / 2 :=
by
  sorry

end minute_hand_length_l51_51438


namespace number_of_minibuses_l51_51370

theorem number_of_minibuses (total_students : ℕ) (capacity : ℕ) (h : total_students = 48) (h_capacity : capacity = 8) : 
  ∃ minibuses, minibuses = (total_students + capacity - 1) / capacity ∧ minibuses = 7 :=
by
  have h1 : (48 + 8 - 1) = 55 := by simp [h, h_capacity]
  have h2 : 55 / 8 = 6 := by simp [h, h_capacity]
  use 7
  sorry

end number_of_minibuses_l51_51370


namespace B_pow_5_eq_rB_plus_sI_l51_51580

def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 4, 5]

def I : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 0, 1]

theorem B_pow_5_eq_rB_plus_sI : 
  ∃ (r s : ℤ), r = 1169 ∧ s = -204 ∧ B^5 = r • B + s • I := 
by
  use 1169
  use -204
  sorry

end B_pow_5_eq_rB_plus_sI_l51_51580


namespace quadratic_distinct_real_roots_l51_51046

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ^ 2 - 2 * x₁ + m = 0 ∧ x₂ ^ 2 - 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end quadratic_distinct_real_roots_l51_51046


namespace notebook_and_pen_prices_l51_51770

theorem notebook_and_pen_prices (x y : ℕ) (h1 : 2 * x + y = 30) (h2 : x = 2 * y) :
  x = 12 ∧ y = 6 :=
by
  sorry

end notebook_and_pen_prices_l51_51770


namespace initial_candies_l51_51924

-- Define the conditions
def candies_given_older_sister : ℕ := 7
def candies_given_younger_sister : ℕ := 6
def candies_left : ℕ := 15

-- Conclude the initial number of candies
theorem initial_candies : (candies_given_older_sister + candies_given_younger_sister + candies_left) = 28 := by
  sorry

end initial_candies_l51_51924


namespace Ed_cats_l51_51467

variable (C F : ℕ)

theorem Ed_cats 
  (h1 : F = 2 * (C + 2))
  (h2 : 2 + C + F = 15) : 
  C = 3 := by 
  sorry

end Ed_cats_l51_51467


namespace thirty_percent_less_eq_one_fourth_more_l51_51992

theorem thirty_percent_less_eq_one_fourth_more (x : ℝ) (hx1 : 0.7 * 90 = 63) (hx2 : (5 / 4) * x = 63) : x = 50 :=
sorry

end thirty_percent_less_eq_one_fourth_more_l51_51992


namespace total_snails_and_frogs_l51_51187

-- Define the number of snails and frogs in the conditions.
def snails : Nat := 5
def frogs : Nat := 2

-- State the problem: proving that the total number of snails and frogs equals 7.
theorem total_snails_and_frogs : snails + frogs = 7 := by
  -- Proof is omitted as the user requested only the statement.
  sorry

end total_snails_and_frogs_l51_51187


namespace pears_picking_total_l51_51354

theorem pears_picking_total :
  let Jason_day1 := 46
  let Keith_day1 := 47
  let Mike_day1 := 12
  let Alicia_day1 := 28
  let Tina_day1 := 33
  let Nicola_day1 := 52

  let Jason_day2 := Jason_day1 / 2
  let Keith_day2 := Keith_day1 / 2
  let Mike_day2 := Mike_day1 / 2
  let Alicia_day2 := 2 * Alicia_day1
  let Tina_day2 := 2 * Tina_day1
  let Nicola_day2 := 2 * Nicola_day1

  let Jason_day3 := (Jason_day1 + Jason_day2) / 2
  let Keith_day3 := (Keith_day1 + Keith_day2) / 2
  let Mike_day3 := (Mike_day1 + Mike_day2) / 2
  let Alicia_day3 := (Alicia_day1 + Alicia_day2) / 2
  let Tina_day3 := (Tina_day1 + Tina_day2) / 2
  let Nicola_day3 := (Nicola_day1 + Nicola_day2) / 2

  let Jason_total := Jason_day1 + Jason_day2 + Jason_day3
  let Keith_total := Keith_day1 + Keith_day2 + Keith_day3
  let Mike_total := Mike_day1 + Mike_day2 + Mike_day3
  let Alicia_total := Alicia_day1 + Alicia_day2 + Alicia_day3
  let Tina_total := Tina_day1 + Tina_day2 + Tina_day3
  let Nicola_total := Nicola_day1 + Nicola_day2 + Nicola_day3

  let overall_total := Jason_total + Keith_total + Mike_total + Alicia_total + Tina_total + Nicola_total

  overall_total = 747 := by
  intro Jason_day1 Jason_day2 Jason_day3 Jason_total
  intro Keith_day1 Keith_day2 Keith_day3 Keith_total
  intro Mike_day1 Mike_day2 Mike_day3 Mike_total
  intro Alicia_day1 Alicia_day2 Alicia_day3 Alicia_total
  intro Tina_day1 Tina_day2 Tina_day3 Tina_total
  intro Nicola_day1 Nicola_day2 Nicola_day3 Nicola_total

  sorry

end pears_picking_total_l51_51354


namespace abs_g_eq_abs_gx_l51_51602

noncomputable def g (x : ℝ) : ℝ :=
if -3 <= x ∧ x <= 0 then x^2 - 2 else -x + 2

noncomputable def abs_g (x : ℝ) : ℝ :=
if -3 <= x ∧ x <= -Real.sqrt 2 then x^2 - 2
else if -Real.sqrt 2 < x ∧ x <= Real.sqrt 2 then 2 - x^2
else if Real.sqrt 2 < x ∧ x <= 2 then 2 - x
else x - 2

theorem abs_g_eq_abs_gx (x : ℝ) (hx1 : -3 <= x ∧ x <= -Real.sqrt 2) 
  (hx2 : -Real.sqrt 2 < x ∧ x <= Real.sqrt 2)
  (hx3 : Real.sqrt 2 < x ∧ x <= 2)
  (hx4 : 2 < x ∧ x <= 3) :
  abs_g x = |g x| :=
by
  sorry

end abs_g_eq_abs_gx_l51_51602


namespace hydrogen_atoms_in_compound_l51_51783

theorem hydrogen_atoms_in_compound :
  ∀ (H_atoms Br_atoms O_atoms total_molecular_weight weight_H weight_Br weight_O : ℝ),
  Br_atoms = 1 ∧ O_atoms = 3 ∧ total_molecular_weight = 129 ∧ 
  weight_H = 1 ∧ weight_Br = 79.9 ∧ weight_O = 16 →
  H_atoms = 1 :=
by
  sorry

end hydrogen_atoms_in_compound_l51_51783


namespace original_price_sarees_l51_51379

theorem original_price_sarees (P : ℝ) (h : 0.85 * 0.80 * P = 272) : P = 400 :=
by
  sorry

end original_price_sarees_l51_51379


namespace speed_of_man_l51_51614

theorem speed_of_man 
  (L : ℝ) 
  (V_t : ℝ) 
  (T : ℝ) 
  (conversion_factor : ℝ) 
  (kmph_to_mps : ℝ → ℝ)
  (final_conversion : ℝ → ℝ) 
  (relative_speed : ℝ) 
  (Vm : ℝ) : Prop := 
L = 220 ∧ V_t = 59 ∧ T = 12 ∧ 
conversion_factor = 1000 / 3600 ∧ 
kmph_to_mps V_t = V_t * conversion_factor ∧ 
relative_speed = L / T ∧ 
Vm = relative_speed - (kmph_to_mps V_t) ∧ 
final_conversion Vm = Vm * 3.6 ∧ 
final_conversion Vm = 6.984

end speed_of_man_l51_51614


namespace all_positive_rationals_are_red_l51_51250

-- Define the property of being red for rational numbers
def is_red (x : ℚ) : Prop :=
  ∃ n : ℕ, ∃ (f : ℕ → ℚ), f 0 = 1 ∧ (∀ m : ℕ, f (m + 1) = f m + 1 ∨ f (m + 1) = f m / (f m + 1)) ∧ f n = x

-- Proposition stating that all positive rational numbers are red
theorem all_positive_rationals_are_red :
  ∀ x : ℚ, 0 < x → is_red x :=
  by sorry

end all_positive_rationals_are_red_l51_51250


namespace scientific_notation_of_million_l51_51792

theorem scientific_notation_of_million (x : ℝ) (h : x = 2600000) : x = 2.6 * 10^6 := by
  sorry

end scientific_notation_of_million_l51_51792


namespace greatest_value_of_x_l51_51994

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l51_51994


namespace jill_has_6_more_dolls_than_jane_l51_51332

theorem jill_has_6_more_dolls_than_jane
  (total_dolls : ℕ) 
  (jane_dolls : ℕ) 
  (more_dolls_than : ℕ → ℕ → Prop)
  (h1 : total_dolls = 32) 
  (h2 : jane_dolls = 13) 
  (jill_dolls : ℕ)
  (h3 : more_dolls_than jill_dolls jane_dolls) :
  (jill_dolls - jane_dolls) = 6 :=
by
  -- the proof goes here
  sorry

end jill_has_6_more_dolls_than_jane_l51_51332


namespace volume_of_sphere_l51_51355

theorem volume_of_sphere
  (a b c : ℝ)
  (h1 : a * b * c = 4 * Real.sqrt 6)
  (h2 : a * b = 2 * Real.sqrt 3)
  (h3 : b * c = 4 * Real.sqrt 3)
  (O_radius : ℝ := Real.sqrt (a^2 + b^2 + c^2) / 2) :
  4 / 3 * Real.pi * O_radius^3 = 32 * Real.pi / 3 := by
  sorry

end volume_of_sphere_l51_51355


namespace soccer_team_games_count_l51_51813

variable (total_games won_games : ℕ)
variable (h1 : won_games = 70)
variable (h2 : won_games = total_games / 2)

theorem soccer_team_games_count : total_games = 140 :=
by
  -- Proof goes here
  sorry

end soccer_team_games_count_l51_51813


namespace trajectory_of_moving_circle_l51_51715

def circle1 (x y : ℝ) := (x + 4) ^ 2 + y ^ 2 = 2
def circle2 (x y : ℝ) := (x - 4) ^ 2 + y ^ 2 = 2

theorem trajectory_of_moving_circle (x y : ℝ) : 
  (x = 0) ∨ (x ^ 2 / 2 - y ^ 2 / 14 = 1) := 
  sorry

end trajectory_of_moving_circle_l51_51715


namespace rectangle_area_l51_51931

variable (w l : ℕ)
variable (A : ℕ)
variable (H1 : l = 5 * w)
variable (H2 : 2 * l + 2 * w = 180)

theorem rectangle_area : A = 1125 :=
by
  sorry

end rectangle_area_l51_51931


namespace person_B_days_l51_51371

theorem person_B_days (A_days : ℕ) (combined_work : ℚ) (x : ℕ) : 
  A_days = 30 → combined_work = (1 / 6) → 3 * (1 / 30 + 1 / x) = combined_work → x = 45 :=
by
  intros hA hCombined hEquation
  sorry

end person_B_days_l51_51371


namespace geometric_sequence_sum_l51_51400

variable (a : ℕ → ℝ)

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sum (h1 : geometric_sequence a)
  (h2 : a 1 > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = 6 :=
sorry

end geometric_sequence_sum_l51_51400


namespace quiz_total_points_l51_51857

theorem quiz_total_points (points : ℕ → ℕ) 
  (h1 : ∀ n, points (n+1) = points n + 4)
  (h2 : points 2 = 39) : 
  (points 0 + points 1 + points 2 + points 3 + points 4 + points 5 + points 6 + points 7) = 360 :=
sorry

end quiz_total_points_l51_51857


namespace find_days_l51_51554

theorem find_days
  (wages1 : ℕ) (workers1 : ℕ) (days1 : ℕ)
  (wages2 : ℕ) (workers2 : ℕ) (days2 : ℕ)
  (h1 : wages1 = 9450) (h2 : workers1 = 15) (h3 : wages2 = 9975)
  (h4 : workers2 = 19) (h5 : days2 = 5) :
  days1 = 6 := 
by
  -- Insert proof here
  sorry

end find_days_l51_51554


namespace _l51_51114

open Nat

/-- Function to check the triangle inequality theorem -/
def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : canFormTriangle 6 4 5 := by
  /- Proof omitted -/
  sorry

end _l51_51114


namespace opposite_of_neg5_l51_51894

-- Define the concept of the opposite of a number
def opposite (x : Int) : Int :=
  -x

-- The proof problem: Prove that the opposite of -5 is 5
theorem opposite_of_neg5 : opposite (-5) = 5 :=
by
  sorry

end opposite_of_neg5_l51_51894


namespace tourist_total_value_l51_51897

theorem tourist_total_value
    (tax_rate : ℝ)
    (V : ℝ)
    (tax_paid : ℝ)
    (exempt_amount : ℝ) :
    exempt_amount = 600 ∧
    tax_rate = 0.07 ∧
    tax_paid = 78.4 →
    (tax_rate * (V - exempt_amount) = tax_paid) →
    V = 1720 :=
by
  intros h1 h2
  have h_exempt : exempt_amount = 600 := h1.left
  have h_tax_rate : tax_rate = 0.07 := h1.right.left
  have h_tax_paid : tax_paid = 78.4 := h1.right.right
  sorry

end tourist_total_value_l51_51897


namespace geometric_sequence_a6_a8_sum_l51_51835

theorem geometric_sequence_a6_a8_sum 
  (a : ℕ → ℕ) (q : ℕ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 1 + a 3 = 5)
  (h2 : a 2 + a 4 = 10) : 
  a 6 + a 8 = 160 := 
sorry

end geometric_sequence_a6_a8_sum_l51_51835


namespace find_k_l51_51748

theorem find_k (m n : ℝ) 
  (h₁ : m = k * n + 5) 
  (h₂ : m + 2 = k * (n + 0.5) + 5) : 
  k = 4 :=
by
  sorry

end find_k_l51_51748


namespace find_values_of_symbols_l51_51996

theorem find_values_of_symbols (a b : ℕ) (h1 : a + b + b = 55) (h2 : a + b = 40) : b = 15 ∧ a = 25 :=
  by
    sorry

end find_values_of_symbols_l51_51996


namespace min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l51_51958

-- Define the first problem
theorem min_cuts_for_eleven_day_stay : 
  (∀ (chain_len num_days : ℕ), chain_len = 11 ∧ num_days = 11 
  → (∃ (cuts : ℕ), cuts = 2)) := 
sorry

-- Define the second problem
theorem max_days_with_n_cuts : 
  (∀ (n chain_len days : ℕ), chain_len = (n + 1) * 2 ^ n - 1 
  → days = (n + 1) * 2 ^ n - 1) := 
sorry

end min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l51_51958


namespace expression_evaluation_l51_51222

noncomputable def evaluate_expression (a b c : ℚ) : ℚ :=
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7)

theorem expression_evaluation : 
  ∀ (a b c : ℚ), c = b - 11 → b = a + 3 → a = 5 → 
  (a + 2) ≠ 0 → (b - 3) ≠ 0 → (c + 7) ≠ 0 → 
  evaluate_expression a b c = 72 / 35 :=
by
  intros a b c hc hb ha h1 h2 h3
  rw [ha, hb, hc, evaluate_expression]
  -- The proof is not required.
  sorry

end expression_evaluation_l51_51222


namespace sum_of_arithmetic_series_51_to_100_l51_51183

theorem sum_of_arithmetic_series_51_to_100 :
  let first_term := 51
  let last_term := 100
  let n := (last_term - first_term) + 1
  2 * (n / 2) * (first_term + last_term) / 2 = 3775 :=
by
  sorry

end sum_of_arithmetic_series_51_to_100_l51_51183


namespace perpendicular_k_parallel_k_l51_51505

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the scalar multiple operations and vector operations
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 + v₂.1, v₂.2 + v₂.2)
def sub (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 - v₂.1, v₂.2 - v₂.2)
def dot (v₁ v₂ : ℝ × ℝ) : ℝ := (v₁.1 * v₂.1 + v₁.2 * v₂.2)

-- Problem 1: If k*a + b is perpendicular to a - 3*b, then k = 19
theorem perpendicular_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  dot vak amb = 0 → k = 19 := sorry

-- Problem 2: If k*a + b is parallel to a - 3*b, then k = -1/3 and they are in opposite directions
theorem parallel_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  ∃ m : ℝ, vak = smul m amb ∧ m < 0 → k = -1/3 := sorry

end perpendicular_k_parallel_k_l51_51505


namespace unique_root_of_linear_equation_l51_51059

theorem unique_root_of_linear_equation (a b : ℝ) (h : a ≠ 0) : ∃! x : ℝ, a * x = b :=
by
  sorry

end unique_root_of_linear_equation_l51_51059


namespace calc_expression1_calc_expression2_l51_51818

theorem calc_expression1 : (1 / 3)^0 + Real.sqrt 27 - abs (-3) + Real.tan (Real.pi / 4) = 1 + 3 * Real.sqrt 3 - 2 :=
by
  sorry

theorem calc_expression2 (x : ℝ) : (x + 2)^2 - 2 * (x - 1) = x^2 + 2 * x + 6 :=
by
  sorry

end calc_expression1_calc_expression2_l51_51818


namespace cosine_double_angle_identity_l51_51847

theorem cosine_double_angle_identity (α : ℝ) (h : Real.sin (α + 7 * Real.pi / 6) = 1) :
  Real.cos (2 * α - 2 * Real.pi / 3) = 1 := by
  sorry

end cosine_double_angle_identity_l51_51847


namespace opposite_of_neg2023_l51_51181

def opposite (x : Int) := -x

theorem opposite_of_neg2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg2023_l51_51181


namespace arithmetic_expression_evaluation_l51_51020

theorem arithmetic_expression_evaluation :
  (-12 * 6) - (-4 * -8) + (-15 * -3) - (36 / (-2)) = -77 :=
by
  sorry

end arithmetic_expression_evaluation_l51_51020


namespace negation_of_proposition_l51_51622

noncomputable def P (x : ℝ) : Prop := x^2 + 1 ≥ 0

theorem negation_of_proposition :
  (¬ ∀ x, x > 1 → P x) ↔ (∃ x, x > 1 ∧ ¬ P x) :=
sorry

end negation_of_proposition_l51_51622


namespace min_colors_needed_l51_51607

theorem min_colors_needed (n : ℕ) : 
  (n + (n * (n - 1)) / 2 ≥ 12) → (n = 5) :=
by
  sorry

end min_colors_needed_l51_51607


namespace janelle_total_marbles_l51_51731

def initial_green_marbles := 26
def bags_of_blue_marbles := 12
def marbles_per_bag := 15
def gift_red_marbles := 7
def gift_green_marbles := 9
def gift_blue_marbles := 12
def gift_red_marbles_given := 3
def returned_blue_marbles := 8

theorem janelle_total_marbles :
  let total_green := initial_green_marbles - gift_green_marbles
  let total_blue := (bags_of_blue_marbles * marbles_per_bag) - gift_blue_marbles + returned_blue_marbles
  let total_red := gift_red_marbles - gift_red_marbles_given
  total_green + total_blue + total_red = 197 :=
by
  sorry

end janelle_total_marbles_l51_51731


namespace exists_infinitely_many_N_l51_51412

open Set

-- Conditions: Definition of the initial set S_0 and recursive sets S_n
variable {S_0 : Set ℕ} (h0 : Set.Finite S_0) -- S_0 is a finite set of positive integers
variable (S : ℕ → Set ℕ) 
(has_S : ∀ n, ∀ a, a ∈ S (n+1) ↔ (a-1 ∈ S n ∧ a ∉ S n ∨ a-1 ∉ S n ∧ a ∈ S n))

-- Main theorem: Proving the existence of infinitely many integers N such that 
-- S_N = S_0 ∪ {N + a : a ∈ S_0}
theorem exists_infinitely_many_N : 
  ∃ᶠ N in at_top, S N = S_0 ∪ {n | ∃ a ∈ S_0, n = N + a} := 
sorry

end exists_infinitely_many_N_l51_51412


namespace petya_wins_with_optimal_play_l51_51521

theorem petya_wins_with_optimal_play :
  ∃ (n m : ℕ), n = 2000 ∧ m = (n * (n - 1)) / 2 ∧
  (∀ (v_cut : ℕ), ∀ (p_cut : ℕ), v_cut = 1 ∧ (p_cut = 2 ∨ p_cut = 3) ∧
  ((∃ k, m - v_cut = 4 * k) → ∃ k, m - v_cut - p_cut = 4 * k + 1) → 
  ∃ k, m - p_cut = 4 * k + 3) :=
sorry

end petya_wins_with_optimal_play_l51_51521


namespace billy_raspberry_juice_billy_raspberry_juice_quarts_l51_51185

theorem billy_raspberry_juice (V : ℚ) (h : V / 12 + 1 = 3) : V = 24 :=
by sorry

theorem billy_raspberry_juice_quarts (V : ℚ) (h : V / 12 + 1 = 3) : V / 4 = 6 :=
by sorry

end billy_raspberry_juice_billy_raspberry_juice_quarts_l51_51185


namespace solve_quadratic_completing_square_l51_51512

theorem solve_quadratic_completing_square (x : ℝ) :
  (2 * x^2 - 4 * x - 1 = 0) ↔ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
by
  sorry

end solve_quadratic_completing_square_l51_51512


namespace intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l51_51449

-- Definitions based on the conditions
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x - a * y + 2 = 0
def perpendicular (a : ℝ) : Prop := a = 0
def parallel (a : ℝ) : Prop := a = 1 ∨ a = -1

-- Theorem 1: Intersection point when a = 0 is (-2, 2)
theorem intersection_point_zero_a_0 :
  ∀ x y : ℝ, l₁ 0 x y → l₂ 0 x y → (x, y) = (-2, 2) := 
by
  sorry

-- Theorem 2: Line l₁ always passes through (0, 2)
theorem l₁_passes_through_0_2 :
  ∀ a : ℝ, l₁ a 0 2 := 
by
  sorry

-- Theorem 3: l₁ is perpendicular to l₂ implies a = 0
theorem l₁_perpendicular_l₂ :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ∀ m n, (a * m + (n / a) = 0)) → (a = 0) :=
by
  sorry

-- Theorem 4: l₁ is parallel to l₂ implies a = 1 or a = -1
theorem l₁_parallel_l₂ :
  ∀ a : ℝ, parallel a → (a = 1 ∨ a = -1) :=
by
  sorry

end intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l51_51449


namespace transaction_gain_per_year_l51_51221

theorem transaction_gain_per_year
  (principal : ℝ) (borrow_rate : ℝ) (lend_rate : ℝ) (time : ℕ)
  (principal_eq : principal = 5000)
  (borrow_rate_eq : borrow_rate = 0.04)
  (lend_rate_eq : lend_rate = 0.06)
  (time_eq : time = 2) :
  (principal * lend_rate * time - principal * borrow_rate * time) / time = 100 := by
  sorry

end transaction_gain_per_year_l51_51221


namespace green_competition_l51_51643

theorem green_competition {x : ℕ} (h : 0 ≤ x ∧ x ≤ 25) : 
  5 * x - (25 - x) ≥ 85 :=
by
  sorry

end green_competition_l51_51643


namespace epsilon_max_success_ratio_l51_51625

theorem epsilon_max_success_ratio :
  ∃ (x y z w u v: ℕ), 
  (y ≠ 350) ∧
  0 < x ∧ 0 < z ∧ 0 < u ∧ 
  x < y ∧ z < w ∧ u < v ∧
  x + z + u < y + w + v ∧
  y + w + v = 800 ∧
  (x / y : ℚ) < (210 / 350 : ℚ) ∧ 
  (z / w : ℚ) < (delta_day_2_ratio) ∧ 
  (u / v : ℚ) < (delta_day_3_ratio) ∧ 
  (x + z + u) / 800 = (789 / 800 : ℚ) := 
by
  sorry

end epsilon_max_success_ratio_l51_51625


namespace qualified_products_correct_l51_51885

def defect_rate : ℝ := 0.005
def total_produced : ℝ := 18000

theorem qualified_products_correct :
  total_produced * (1 - defect_rate) = 17910 := by
  sorry

end qualified_products_correct_l51_51885


namespace simplify_expression_l51_51426

variable (a : ℝ) (ha : a ≠ 0)

theorem simplify_expression : (21 * a^3 - 7 * a) / (7 * a) = 3 * a^2 - 1 := by
  sorry

end simplify_expression_l51_51426


namespace sam_total_yellow_marbles_l51_51596

def sam_original_yellow_marbles : Float := 86.0
def sam_yellow_marbles_given_by_joan : Float := 25.0

theorem sam_total_yellow_marbles : sam_original_yellow_marbles + sam_yellow_marbles_given_by_joan = 111.0 := by
  sorry

end sam_total_yellow_marbles_l51_51596


namespace cricket_avg_score_l51_51798

theorem cricket_avg_score
  (avg_first_two : ℕ)
  (num_first_two : ℕ)
  (avg_all_five : ℕ)
  (num_all_five : ℕ)
  (avg_first_two_eq : avg_first_two = 40)
  (num_first_two_eq : num_first_two = 2)
  (avg_all_five_eq : avg_all_five = 22)
  (num_all_five_eq : num_all_five = 5) :
  ((num_all_five * avg_all_five - num_first_two * avg_first_two) / (num_all_five - num_first_two) = 10) :=
by
  sorry

end cricket_avg_score_l51_51798


namespace sequence_third_term_l51_51985

theorem sequence_third_term (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 5) : a 3 = 4 := by
  sorry

end sequence_third_term_l51_51985


namespace paula_candies_l51_51074

def candies_per_friend (total_candies : ℕ) (number_of_friends : ℕ) : ℕ :=
  total_candies / number_of_friends

theorem paula_candies :
  let initial_candies := 20
  let additional_candies := 4
  let total_candies := initial_candies + additional_candies
  let number_of_friends := 6
  candies_per_friend total_candies number_of_friends = 4 :=
by
  sorry

end paula_candies_l51_51074


namespace eval_g_six_times_at_2_l51_51819

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem eval_g_six_times_at_2 : g (g (g (g (g (g 2))))) = 4 := sorry

end eval_g_six_times_at_2_l51_51819


namespace graph_is_empty_l51_51357

theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + y^2 - 9 * x - 4 * y + 17 ≠ 0 :=
by
  intros x y
  sorry

end graph_is_empty_l51_51357


namespace compute_expression_l51_51628

def sum_of_squares := 7^2 + 5^2
def square_of_sum := (7 + 5)^2
def sum_of_both := sum_of_squares + square_of_sum
def final_result := 2 * sum_of_both

theorem compute_expression : final_result = 436 := by
  sorry

end compute_expression_l51_51628


namespace complement_U_A_l51_51698

-- Definitions of U and A based on problem conditions
def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 2}

-- Definition of the complement in Lean
def complement (A B : Set ℤ) : Set ℤ := {x | x ∈ A ∧ x ∉ B}

-- The main statement to be proved
theorem complement_U_A :
  complement U A = {1, 3} :=
sorry

end complement_U_A_l51_51698


namespace john_saved_120_dollars_l51_51058

-- Defining the conditions
def num_machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := num_machines * ball_bearings_per_machine
def regular_price_per_bearing : ℝ := 1
def sale_price_per_bearing : ℝ := 0.75
def bulk_discount : ℝ := 0.20
def discounted_price_per_bearing : ℝ := sale_price_per_bearing - (bulk_discount * sale_price_per_bearing)

-- Calculate total costs
def total_cost_without_sale : ℝ := total_ball_bearings * regular_price_per_bearing
def total_cost_with_sale : ℝ := total_ball_bearings * discounted_price_per_bearing

-- Calculate the savings
def savings : ℝ := total_cost_without_sale - total_cost_with_sale

-- The theorem we want to prove
theorem john_saved_120_dollars : savings = 120 := by
  sorry

end john_saved_120_dollars_l51_51058


namespace integral_sin_pi_half_to_three_pi_half_l51_51979

theorem integral_sin_pi_half_to_three_pi_half :
  ∫ x in (Set.Icc (Real.pi / 2) (3 * Real.pi / 2)), Real.sin x = 0 :=
by
  sorry

end integral_sin_pi_half_to_three_pi_half_l51_51979


namespace number_of_correct_calculations_is_one_l51_51444

/- Given conditions -/
def cond1 (a : ℝ) : Prop := a^2 * a^2 = 2 * a^2
def cond2 (a b : ℝ) : Prop := (a - b)^2 = a^2 - b^2
def cond3 (a : ℝ) : Prop := a^2 + a^3 = a^5
def cond4 (a b : ℝ) : Prop := (-2 * a^2 * b^3)^3 = -6 * a^6 * b^3
def cond5 (a : ℝ) : Prop := (-a^3)^2 / a = a^5

/- Statement to prove the number of correct calculations is 1 -/
theorem number_of_correct_calculations_is_one :
  (¬ (cond1 a)) ∧ (¬ (cond2 a b)) ∧ (¬ (cond3 a)) ∧ (¬ (cond4 a b)) ∧ (cond5 a) → 1 = 1 :=
by
  sorry

end number_of_correct_calculations_is_one_l51_51444


namespace dandelions_initial_l51_51054

theorem dandelions_initial (y w : ℕ) (h1 : y + w = 35) (h2 : y - 2 = 2 * (w - 6)) : y = 20 ∧ w = 15 :=
by
  sorry

end dandelions_initial_l51_51054


namespace two_thirds_greater_l51_51583

theorem two_thirds_greater :
  let epsilon : ℚ := (2 : ℚ) / (3 * 10^8)
  let decimal_part : ℚ := 66666666 / 10^8
  (2 / 3) - decimal_part = epsilon := by
  sorry

end two_thirds_greater_l51_51583


namespace simplify_expression_l51_51920

variable (x y : ℝ)

theorem simplify_expression : (-(3 * x * y - 2 * x ^ 2) - 2 * (3 * x ^ 2 - x * y)) = (-4 * x ^ 2 - x * y) :=
by
  sorry

end simplify_expression_l51_51920


namespace magnitude_of_b_is_5_l51_51200

variable (a b : ℝ × ℝ)
variable (h_a : a = (3, -2))
variable (h_ab : a + b = (0, 2))

theorem magnitude_of_b_is_5 : ‖b‖ = 5 :=
by
  sorry

end magnitude_of_b_is_5_l51_51200


namespace find_natural_number_l51_51339

-- Definitions reflecting the conditions and result
def is_sum_of_two_squares (n : ℕ) := ∃ a b : ℕ, a * a + b * b = n

def has_exactly_one_not_sum_of_two_squares (n : ℕ) :=
  ∃! x : ℤ, ¬is_sum_of_two_squares (x.natAbs % n)

theorem find_natural_number (n : ℕ) (h : n ≥ 2) : 
  has_exactly_one_not_sum_of_two_squares n ↔ n = 4 :=
sorry

end find_natural_number_l51_51339


namespace calculate_length_X_l51_51365

theorem calculate_length_X 
  (X : ℝ)
  (h1 : 3 + X + 4 = 5 + 7 + X)
  : X = 5 :=
sorry

end calculate_length_X_l51_51365


namespace roots_condition_l51_51564

theorem roots_condition (r1 r2 p : ℝ) (h_eq : ∀ x : ℝ, x^2 + p * x + 12 = 0 → (x = r1 ∨ x = r2))
(h_distinct : r1 ≠ r2) (h_vieta1 : r1 + r2 = -p) (h_vieta2 : r1 * r2 = 12) : 
|r1| > 3 ∨ |r2| > 3 :=
by
  sorry

end roots_condition_l51_51564


namespace second_coloring_book_pictures_l51_51504

theorem second_coloring_book_pictures (P1 P2 P_colored P_left : ℕ) (h1 : P1 = 23) (h2 : P_colored = 44) (h3 : P_left = 11) (h4 : P1 + P2 = P_colored + P_left) :
  P2 = 32 :=
by
  rw [h1, h2, h3] at h4
  linarith

end second_coloring_book_pictures_l51_51504


namespace polynomial_coefficients_l51_51709

theorem polynomial_coefficients (a : ℕ → ℤ) :
  (∀ x : ℤ, (2 * x - 1) * ((x + 1) ^ 7) = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + 
  (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 + (a 7) * x^7 + (a 8) * x^8) →
  (a 0 = -1) ∧
  (a 0 + a 2 + a 4 + a 6 + a 8 = 64) ∧
  (a 1 + 2 * (a 2) + 3 * (a 3) + 4 * (a 4) + 5 * (a 5) + 6 * (a 6) + 7 * (a 7) + 8 * (a 8) = 704) := by
  sorry

end polynomial_coefficients_l51_51709


namespace range_of_a_l51_51490

open Real

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, otimes x (x + a) < 1) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l51_51490


namespace square_side_length_l51_51083

variable (s : ℝ)
variable (k : ℝ := 6)

theorem square_side_length :
  s^2 = k * 4 * s → s = 24 :=
by
  intro h
  sorry

end square_side_length_l51_51083


namespace part1_part2_l51_51808

/-
Part 1: Given the conditions of parabola and line intersection, prove the range of slope k of the line.
-/
theorem part1 (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  k > -2 + 2 * Real.sqrt 2 ∨ k < -2 - 2 * Real.sqrt 2 :=
  sorry

/-
Part 2: Given the conditions of locus of point Q on the line segment P1P2, prove the equation of the locus.
-/
theorem part2 (x y : ℝ) (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  2 * x - y + 1 = 0 ∧ (-Real.sqrt 2 - 1 < x ∧ x < Real.sqrt 2 - 1 ∧ x ≠ -1) :=
  sorry

end part1_part2_l51_51808


namespace empire_state_building_height_l51_51943

theorem empire_state_building_height (h_top_floor : ℕ) (h_antenna_spire : ℕ) (total_height : ℕ) :
  h_top_floor = 1250 ∧ h_antenna_spire = 204 ∧ total_height = h_top_floor + h_antenna_spire → total_height = 1454 :=
by
  sorry

end empire_state_building_height_l51_51943


namespace license_plate_combinations_l51_51251

-- Definition for the conditions of the problem
def num_license_plate_combinations : ℕ :=
  let num_letters := 26
  let num_digits := 10
  let choose_two_distinct_letters := (num_letters * (num_letters - 1)) / 2
  let arrange_pairs := 2
  let choose_positions := 6
  let digit_permutations := num_digits ^ 2
  choose_two_distinct_letters * arrange_pairs * choose_positions * digit_permutations

-- The theorem we are proving
theorem license_plate_combinations :
  num_license_plate_combinations = 390000 :=
by
  -- The proof would be provided here.
  sorry

end license_plate_combinations_l51_51251


namespace exists_integers_for_S_geq_100_l51_51231

theorem exists_integers_for_S_geq_100 (S : ℤ) (hS : S ≥ 100) :
  ∃ (T C B : ℤ) (P : ℤ),
    T > 0 ∧ C > 0 ∧ B > 0 ∧
    T > C ∧ C > B ∧
    T + C + B = S ∧
    T * C * B = P ∧
    (∀ (T₁ C₁ B₁ T₂ C₂ B₂ : ℤ), 
      T₁ > 0 ∧ C₁ > 0 ∧ B₁ > 0 ∧ 
      T₂ > 0 ∧ C₂ > 0 ∧ B₂ > 0 ∧ 
      T₁ > C₁ ∧ C₁ > B₁ ∧ 
      T₂ > C₂ ∧ C₂ > B₂ ∧ 
      T₁ + C₁ + B₁ = S ∧ 
      T₂ + C₂ + B₂ = S ∧ 
      T₁ * C₁ * B₁ = T₂ * C₂ * B₂ → 
      (T₁ = T₂) ∧ (C₁ = C₂) ∧ (B₁ = B₂) → false) :=
sorry

end exists_integers_for_S_geq_100_l51_51231


namespace simplify_fraction_l51_51565

theorem simplify_fraction (a : ℕ) (h : a = 3) : (10 * a ^ 3) / (55 * a ^ 2) = 6 / 11 :=
by sorry

end simplify_fraction_l51_51565


namespace functional_eqn_even_function_l51_51916

variable {R : Type*} [AddGroup R] (f : R → ℝ)

theorem functional_eqn_even_function
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_func_eq : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b) :
  ∀ x, f (-x) = f x :=
by
  sorry

end functional_eqn_even_function_l51_51916


namespace find_s_l51_51660

def f (x s : ℝ) := 3 * x^5 + 2 * x^4 - x^3 + 4 * x^2 - 5 * x + s

theorem find_s (s : ℝ) (h : f 3 s = 0) : s = -885 :=
  by sorry

end find_s_l51_51660


namespace anne_already_made_8_drawings_l51_51309

-- Define the conditions as Lean definitions
def num_markers : ℕ := 12
def drawings_per_marker : ℚ := 3 / 2 -- Equivalent to 1.5
def remaining_drawings : ℕ := 10

-- Calculate the total number of drawings Anne can make with her markers
def total_drawings : ℚ := num_markers * drawings_per_marker

-- Calculate the already made drawings
def already_made_drawings : ℚ := total_drawings - remaining_drawings

-- The theorem to prove
theorem anne_already_made_8_drawings : already_made_drawings = 8 := 
by 
  have h1 : total_drawings = 18 := by sorry -- Calculating total drawings as 18
  have h2 : already_made_drawings = 8 := by sorry -- Calculating already made drawings as total drawings minus remaining drawings
  exact h2

end anne_already_made_8_drawings_l51_51309


namespace quadratic_complete_square_l51_51327

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 2 * x + 3 = (x - 1)^2 + 2) := 
by
  intro x
  sorry

end quadratic_complete_square_l51_51327


namespace rectangle_y_value_l51_51413

theorem rectangle_y_value (y : ℝ) (h1 : -2 < 6) (h2 : y > 2) 
    (h3 : 8 * (y - 2) = 64) : y = 10 :=
by
  sorry

end rectangle_y_value_l51_51413


namespace fourth_term_of_geometric_sequence_l51_51680

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) :=
  a * r ^ (n - 1)

theorem fourth_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) (ar5_eq : a * r ^ 5 = 32) 
  (a_eq : a = 81) :
  geometric_sequence a r 4 = 24 := 
by 
  sorry

end fourth_term_of_geometric_sequence_l51_51680


namespace sum_of_coefficients_l51_51440

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 : ℤ)
  (h : (1 - 2 * X)^5 = a + a1 * X + a2 * X^2 + a3 * X^3 + a4 * X^4 + a5 * X^5) :
  a1 + a2 + a3 + a4 + a5 = -2 :=
by {
  -- the proof steps would go here
  sorry
}

end sum_of_coefficients_l51_51440


namespace current_population_is_15336_l51_51362

noncomputable def current_population : ℝ :=
  let growth_rate := 1.28
  let future_population : ℝ := 25460.736
  let years := 2
  future_population / (growth_rate ^ years)

theorem current_population_is_15336 :
  current_population = 15536 := sorry

end current_population_is_15336_l51_51362


namespace circle_equation_l51_51681

variable (x y : ℝ)

def center : ℝ × ℝ := (4, -6)
def radius : ℝ := 3

theorem circle_equation : (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ (x - 4)^2 + (y + 6)^2 = 9 :=
by
  sorry

end circle_equation_l51_51681


namespace students_taking_art_l51_51860

theorem students_taking_art :
  ∀ (total_students music_students both_music_art neither_music_art : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_music_art = 10 →
  neither_music_art = 470 →
  (total_students - neither_music_art) - (music_students - both_music_art) - both_music_art = 10 :=
by
  intros total_students music_students both_music_art neither_music_art h_total h_music h_both h_neither
  sorry

end students_taking_art_l51_51860


namespace selling_price_same_loss_as_profit_l51_51772

theorem selling_price_same_loss_as_profit (cost_price selling_price_with_profit selling_price_with_loss profit loss : ℝ)
  (h1 : selling_price_with_profit - cost_price = profit)
  (h2 : cost_price - selling_price_with_loss = loss)
  (h3 : profit = loss) :
  selling_price_with_loss = 52 :=
by
  have h4 : selling_price_with_profit = 66 := by sorry
  have h5 : cost_price = 59 := by sorry
  have h6 : profit = 66 - 59 := by sorry
  have h7 : profit = 7 := by sorry
  have h8 : loss = 59 - selling_price_with_loss := by sorry
  have h9 : loss = 7 := by sorry
  have h10 : selling_price_with_loss = 59 - loss := by sorry
  have h11 : selling_price_with_loss = 59 - 7 := by sorry
  have h12 : selling_price_with_loss = 52 := by sorry
  exact h12

end selling_price_same_loss_as_profit_l51_51772


namespace area_triangle_3_6_l51_51647

/-
Problem: Prove that the area of a triangle with base 3 meters and height 6 meters is 9 square meters.
Definitions: 
- base: The base of the triangle is 3 meters.
- height: The height of the triangle is 6 meters.
Conditions: 
- The area of a triangle formula.
Correct Answer: 9 square meters.
-/

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

theorem area_triangle_3_6 : area_of_triangle 3 6 = 9 := by
  sorry

end area_triangle_3_6_l51_51647


namespace find_f3_value_l51_51629

noncomputable def f (x : ℚ) : ℚ := (x^2 + 2*x + 1) / (4*x - 5)

theorem find_f3_value : f 3 = 16 / 7 :=
by sorry

end find_f3_value_l51_51629


namespace arctan_sum_eq_half_pi_l51_51544

theorem arctan_sum_eq_half_pi (y : ℚ) :
  2 * Real.arctan (1 / 3) + Real.arctan (1 / 10) + Real.arctan (1 / 30) + Real.arctan (1 / y) = Real.pi / 2 →
  y = 547 / 620 := by
  sorry

end arctan_sum_eq_half_pi_l51_51544


namespace sixteen_pow_five_eq_four_pow_p_l51_51095

theorem sixteen_pow_five_eq_four_pow_p (p : ℕ) (h : 16^5 = 4^p) : p = 10 := 
  sorry

end sixteen_pow_five_eq_four_pow_p_l51_51095


namespace rectangle_diagonal_length_l51_51427

theorem rectangle_diagonal_length
  (a b : ℝ)
  (h1 : a = 40 * Real.sqrt 2)
  (h2 : b = 2 * a) :
  Real.sqrt (a^2 + b^2) = 160 := by
  sorry

end rectangle_diagonal_length_l51_51427


namespace num_teachers_l51_51515

-- This statement involves defining the given conditions and stating the theorem to be proved.
theorem num_teachers (parents students total_people : ℕ) (h_parents : parents = 73) (h_students : students = 724) (h_total : total_people = 1541) :
  total_people - (parents + students) = 744 :=
by
  -- Including sorry to skip the proof, as required.
  sorry

end num_teachers_l51_51515


namespace normal_price_of_article_l51_51828

theorem normal_price_of_article 
  (P : ℝ) 
  (h : (P * 0.88 * 0.78 * 0.85) * 1.06 = 144) : 
  P = 144 / (0.88 * 0.78 * 0.85 * 1.06) :=
sorry

end normal_price_of_article_l51_51828


namespace annulus_divide_l51_51099

theorem annulus_divide (r : ℝ) (h₁ : 2 < 14) (h₂ : 2 > 0) (h₃ : 14 > 0)
    (h₄ : π * 196 - π * r^2 = π * r^2 - π * 4) : r = 10 := 
sorry

end annulus_divide_l51_51099


namespace exists_integers_m_n_for_inequalities_l51_51815

theorem exists_integers_m_n_for_inequalities (a b : ℝ) (h : a ≠ b) : ∃ (m n : ℤ), 
  (a * (m : ℝ) + b * (n : ℝ) < 0) ∧ (b * (m : ℝ) + a * (n : ℝ) > 0) :=
sorry

end exists_integers_m_n_for_inequalities_l51_51815


namespace average_marks_l51_51040

theorem average_marks (A : ℝ) :
  let marks_first_class := 25 * A
  let marks_second_class := 30 * 60
  let total_marks := 55 * 50.90909090909091
  marks_first_class + marks_second_class = total_marks → A = 40 :=
by
  sorry

end average_marks_l51_51040


namespace percent_decrease_of_y_l51_51779

theorem percent_decrease_of_y (k x y q : ℝ) (h_inv_prop : x * y = k) (h_pos : 0 < x ∧ 0 < y) (h_q : 0 < q) :
  let x' := x * (1 + q / 100)
  let y' := y * 100 / (100 + q)
  (y - y') / y * 100 = (100 * q) / (100 + q) :=
by
  sorry

end percent_decrease_of_y_l51_51779


namespace problem_l51_51661

theorem problem (a k : ℕ) (h_a_pos : 0 < a) (h_a_k_pos : 0 < k) (h_div : (a^2 + k) ∣ ((a - 1) * a * (a + 1))) : k ≥ a :=
sorry

end problem_l51_51661


namespace consecutive_odd_integer_sum_l51_51018

theorem consecutive_odd_integer_sum {n : ℤ} (h1 : n = 17 ∨ n + 2 = 17) (h2 : n + n + 2 ≥ 36) : (n = 17 → n + 2 = 19) ∧ (n + 2 = 17 → n = 15) :=
by
  sorry

end consecutive_odd_integer_sum_l51_51018


namespace jim_travels_20_percent_of_jill_l51_51807

def john_distance : ℕ := 15
def jill_travels_less : ℕ := 5
def jim_distance : ℕ := 2
def jill_distance : ℕ := john_distance - jill_travels_less

theorem jim_travels_20_percent_of_jill :
  (jim_distance * 100) / jill_distance = 20 := by
  sorry

end jim_travels_20_percent_of_jill_l51_51807


namespace max_S_is_9_l51_51955

-- Definitions based on the conditions
def a (n : ℕ) : ℤ := 28 - 3 * n
def S (n : ℕ) : ℤ := n * (25 + a n) / 2

-- The theorem to be proved
theorem max_S_is_9 : ∃ n : ℕ, n = 9 ∧ S n = 117 :=
by
  sorry

end max_S_is_9_l51_51955


namespace initial_skittles_geq_16_l51_51572

variable (S : ℕ) -- S represents the total number of Skittles Lillian had initially
variable (L : ℕ) -- L represents the number of Skittles Lillian kept as leftovers

theorem initial_skittles_geq_16 (h1 : S = 8 * 2 + L) : S ≥ 16 :=
by
  sorry

end initial_skittles_geq_16_l51_51572


namespace even_perfect_square_factors_l51_51281

theorem even_perfect_square_factors :
  let factors := 2^6 * 5^4 * 7^3
  ∃ (count : ℕ), count = (3 * 3 * 2) ∧
  ∀ (a b c : ℕ), (0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ c ∧ c ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 
  a % 2 = 0 ∧ 2 ≤ a ∧ c % 2 = 0 ∧ b % 2 = 0) → 
  a * b * c < count :=
by
  sorry

end even_perfect_square_factors_l51_51281


namespace inverse_variation_solution_l51_51907

noncomputable def const_k (x y : ℝ) := (x^2) * (y^4)

theorem inverse_variation_solution (x y : ℝ) (k : ℝ) (h1 : x = 8) (h2 : y = 2) (h3 : k = const_k x y) :
  ∀ y' : ℝ, y' = 4 → const_k x y' = 1024 → x^2 = 4 := by
  intros
  sorry

end inverse_variation_solution_l51_51907


namespace water_usage_in_May_l51_51105

theorem water_usage_in_May (x : ℝ) (h_cost : 45 = if x ≤ 12 then 2 * x 
                                                else if x ≤ 18 then 24 + 2.5 * (x - 12) 
                                                else 39 + 3 * (x - 18)) : x = 20 :=
sorry

end water_usage_in_May_l51_51105


namespace second_dog_miles_per_day_l51_51518

-- Definitions describing conditions
section DogWalk
variable (total_miles_week : ℕ)
variable (first_dog_miles_day : ℕ)
variable (days_in_week : ℕ)

-- Assert conditions given in the problem
def condition1 := total_miles_week = 70
def condition2 := first_dog_miles_day = 2
def condition3 := days_in_week = 7

-- The theorem to prove
theorem second_dog_miles_per_day
  (h1 : condition1 total_miles_week)
  (h2 : condition2 first_dog_miles_day)
  (h3 : condition3 days_in_week) :
  (total_miles_week - days_in_week * first_dog_miles_day) / days_in_week = 8 :=
sorry
end DogWalk

end second_dog_miles_per_day_l51_51518


namespace average_speed_before_increase_l51_51410

-- Definitions for the conditions
def t_before := 12   -- Travel time before the speed increase in hours
def t_after := 10    -- Travel time after the speed increase in hours
def speed_diff := 20 -- Speed difference between before and after in km/h

-- Variable for the speed before increase
variable (s_before : ℕ) -- Average speed before the speed increase in km/h

-- Definitions for the speeds
def s_after := s_before + speed_diff -- Average speed after the speed increase in km/h

-- Equations derived from the problem conditions
def dist_eqn_before := s_before * t_before
def dist_eqn_after := s_after * t_after

-- The proof problem stated in Lean
theorem average_speed_before_increase : dist_eqn_before = dist_eqn_after → s_before = 100 := by
  sorry

end average_speed_before_increase_l51_51410


namespace volume_of_water_in_prism_l51_51934

-- Define the given dimensions and conditions
def length_x := 20 -- cm
def length_y := 30 -- cm
def length_z := 40 -- cm
def angle := 30 -- degrees
def total_volume := 24 -- liters

-- The wet fraction of the upper surface
def wet_fraction := 1 / 4

-- Correct answer to be proven
def volume_water := 18.8 -- liters

theorem volume_of_water_in_prism :
  -- Given the conditions
  (length_x = 20) ∧ (length_y = 30) ∧ (length_z = 40) ∧ (angle = 30) ∧ (wet_fraction = 1 / 4) ∧ (total_volume = 24) →
  -- Prove that the volume of water is as calculated
  volume_water = 18.8 :=
sorry

end volume_of_water_in_prism_l51_51934


namespace distribute_money_equation_l51_51056

theorem distribute_money_equation (x : ℕ) (hx : x > 0) : 
  (10 : ℚ) / x = (40 : ℚ) / (x + 6) := 
sorry

end distribute_money_equation_l51_51056


namespace expression_value_l51_51876

theorem expression_value : (28 * 2 + (48 / 6) ^ 2 - 5) * (69 / 3) + 24 * (3 ^ 2 - 2) = 2813 := by
  sorry

end expression_value_l51_51876


namespace right_triangle_equation_l51_51588

-- Let a, b, and c be the sides of a right triangle with a^2 + b^2 = c^2
variables (a b c : ℕ)
-- Define the semiperimeter
def semiperimeter (a b c : ℕ) : ℕ := (a + b + c) / 2
-- Define the radius of the inscribed circle
def inscribed_radius (a b c : ℕ) : ℚ := (a * b) / (2 * semiperimeter a b c)
-- State the theorem to prove
theorem right_triangle_equation : 
    ∀ a b c : ℕ, a^2 + b^2 = c^2 → semiperimeter a b c + inscribed_radius a b c = a + b := by
  sorry

end right_triangle_equation_l51_51588


namespace largest_common_term_l51_51904

-- Definitions for the first arithmetic sequence
def arithmetic_seq1 (n : ℕ) : ℕ := 2 + 5 * n

-- Definitions for the second arithmetic sequence
def arithmetic_seq2 (m : ℕ) : ℕ := 5 + 8 * m

-- Main statement of the problem
theorem largest_common_term (n m k : ℕ) (a : ℕ) :
  (a = arithmetic_seq1 n) ∧ (a = arithmetic_seq2 m) ∧ (1 ≤ a) ∧ (a ≤ 150) →
  a = 117 :=
by {
  sorry
}

end largest_common_term_l51_51904


namespace sum_three_digit_even_integers_l51_51759

theorem sum_three_digit_even_integers :
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 247050 :=
by
  let a := 100
  let d := 2
  let l := 998
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  sorry

end sum_three_digit_even_integers_l51_51759


namespace alice_bob_sum_proof_l51_51665

noncomputable def alice_bob_sum_is_22 : Prop :=
  ∃ A B : ℕ, (1 ≤ A ∧ A ≤ 50) ∧ (1 ≤ B ∧ B ≤ 50) ∧ (B % 3 = 0) ∧ (∃ k : ℕ, 2 * B + A = k^2) ∧ (A + B = 22)

theorem alice_bob_sum_proof : alice_bob_sum_is_22 :=
sorry

end alice_bob_sum_proof_l51_51665


namespace birds_in_marsh_end_of_day_l51_51269

def geese_initial : Nat := 58
def ducks : Nat := 37
def geese_flew_away : Nat := 15
def swans : Nat := 22
def herons : Nat := 2

theorem birds_in_marsh_end_of_day : 
  58 - 15 + 37 + 22 + 2 = 104 := by
  sorry

end birds_in_marsh_end_of_day_l51_51269


namespace find_a_equiv_l51_51071

theorem find_a_equiv (a x : ℝ) (h : ∀ x, (a * x^2 + 20 * x + 25) = (2 * x + 5) * (2 * x + 5)) : a = 4 :=
by
  sorry

end find_a_equiv_l51_51071


namespace value_of_a_27_l51_51650

def a_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem value_of_a_27 (a : ℕ → ℕ) (h : a_sequence a) : a 27 = 702 :=
sorry

end value_of_a_27_l51_51650


namespace odd_function_alpha_l51_51390
open Real

noncomputable def f (x : ℝ) : ℝ :=
  cos x * (sin x + sqrt 3 * cos x) - sqrt 3 / 2

noncomputable def g (x : ℝ) (α : ℝ) : ℝ :=
  f (x + α)

theorem odd_function_alpha (α : ℝ) (a : α > 0) :
  (∀ x : ℝ, g x α = - g (-x) α) ↔ 
  ∃ k : ℕ, α = (2 * k - 1) * π / 6 := sorry

end odd_function_alpha_l51_51390


namespace ratio_of_cream_l51_51638

theorem ratio_of_cream
  (joes_initial_coffee : ℕ := 20)
  (joe_cream_added : ℕ := 3)
  (joe_amount_drank : ℕ := 4)
  (joanns_initial_coffee : ℕ := 20)
  (joann_amount_drank : ℕ := 4)
  (joann_cream_added : ℕ := 3) :
  let joe_final_cream := (joe_cream_added - joe_amount_drank * (joe_cream_added / (joe_cream_added + joes_initial_coffee)))
  let joann_final_cream := joann_cream_added
  (joe_final_cream / joanns_initial_coffee + joann_cream_added = 15 / 23) :=
sorry

end ratio_of_cream_l51_51638


namespace intersection_count_l51_51952

theorem intersection_count :
  ∀ {x y : ℝ}, (2 * x - 2 * y + 4 = 0 ∨ 6 * x + 2 * y - 8 = 0) ∧ (y = -x^2 + 2 ∨ 4 * x - 10 * y + 14 = 0) → 
  (x ≠ 0 ∨ y ≠ 2) ∧ (x ≠ -1 ∨ y ≠ 1) ∧ (x ≠ 1 ∨ y ≠ -1) ∧ (x ≠ 2 ∨ y ≠ 2) → 
  ∃! (p : ℝ × ℝ), (p = (0, 2) ∨ p = (-1, 1) ∨ p = (1, -1) ∨ p = (2, 2)) := sorry

end intersection_count_l51_51952


namespace trig_identity_A_trig_identity_D_l51_51331

theorem trig_identity_A : 
  (Real.tan (25 * Real.pi / 180) + Real.tan (20 * Real.pi / 180) + Real.tan (25 * Real.pi / 180) * Real.tan (20 * Real.pi / 180) = 1) :=
by sorry

theorem trig_identity_D : 
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180) = 4) :=
by sorry

end trig_identity_A_trig_identity_D_l51_51331


namespace sum_q_evals_l51_51668

noncomputable def q : ℕ → ℤ := sorry -- definition of q will be derived from conditions

theorem sum_q_evals :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) + (q 7) + (q 8) + (q 9) +
  (q 10) + (q 11) + (q 12) + (q 13) + (q 14) + (q 15) + (q 16) + (q 17) + (q 18) = 456 :=
by
  -- Given conditions
  have h1 : q 1 = 3 := sorry
  have h6 : q 6 = 23 := sorry
  have h12 : q 12 = 17 := sorry
  have h17 : q 17 = 31 := sorry
  -- Proof outline (solved steps omitted for clarity)
  sorry

end sum_q_evals_l51_51668


namespace sector_area_max_angle_l51_51322

theorem sector_area_max_angle (r : ℝ) (θ : ℝ) (h : 0 < r ∧ r < 10) 
  (H : 2 * r + r * θ = 20) : θ = 2 :=
by
  sorry

end sector_area_max_angle_l51_51322


namespace max_profit_price_l51_51061

-- Define the initial conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 90
def initial_sales_volume : ℝ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  let selling_price := initial_selling_price + x
  let sales_volume := initial_sales_volume - x * sales_volume_decrease
  let profit_per_item := selling_price - purchase_price
  profit_per_item * sales_volume

-- The statement that needs to be proved
theorem max_profit_price : ∃ x : ℝ, x = 10 ∧ (initial_selling_price + x = 100) := by
  sorry

end max_profit_price_l51_51061


namespace equilateral_triangle_area_l51_51215

theorem equilateral_triangle_area (h : ∀ (a : ℝ), a = 2 * Real.sqrt 3) : 
  ∃ (a : ℝ), a = 4 * Real.sqrt 3 := 
sorry

end equilateral_triangle_area_l51_51215


namespace value_of_expression_is_correct_l51_51291

-- Defining the sub-expressions as Lean terms
def three_squared : ℕ := 3^2
def intermediate_result : ℕ := three_squared - 3
def final_result : ℕ := intermediate_result^2

-- The statement we need to prove
theorem value_of_expression_is_correct : final_result = 36 := by
  sorry

end value_of_expression_is_correct_l51_51291


namespace cannot_reach_eighth_vertex_l51_51245

def Point := ℕ × ℕ × ℕ

def symmetry (p1 p2 : Point) : Point :=
  let (a, b, c) := p1
  let (a', b', c') := p2
  (2 * a' - a, 2 * b' - b, 2 * c' - c)

def vertices : List Point :=
  [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

theorem cannot_reach_eighth_vertex : ∀ (p : Point), p ∈ vertices → ∀ (q : Point), q ∈ vertices → 
  ¬(symmetry p q = (1, 1, 1)) :=
by
  sorry

end cannot_reach_eighth_vertex_l51_51245


namespace triangle_sides_ratios_l51_51266

theorem triangle_sides_ratios (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) :
  a / (b + c) = b / (a + c) + c / (a + b) :=
sorry

end triangle_sides_ratios_l51_51266


namespace minimum_value_expression_l51_51168

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, 
    (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
    x = (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c)) ∧
    x = -17 + 12 * Real.sqrt 2 := 
sorry

end minimum_value_expression_l51_51168


namespace Petya_cannot_achieve_goal_l51_51836

theorem Petya_cannot_achieve_goal (n : ℕ) (h : n ≥ 2) :
  ¬ (∃ (G : ℕ → Prop), (∀ i : ℕ, (G i ↔ (G ((i + 2) % (2 * n))))) ∨ (G (i + 1) ≠ G (i + 2))) :=
sorry

end Petya_cannot_achieve_goal_l51_51836


namespace line_through_points_l51_51450

theorem line_through_points (m b : ℝ)
  (h_slope : m = (-1 - 3) / (-3 - 1))
  (h_point : 3 = m * 1 + b) :
  m + b = 3 :=
sorry

end line_through_points_l51_51450


namespace dragos_wins_l51_51422

variable (S : Set ℕ) [Infinite S]
variable (x : ℕ → ℕ)
variable (M N : ℕ)
variable (p : ℕ)

theorem dragos_wins (h_prime_p : Nat.Prime p) (h_subset_S : p ∈ S) 
  (h_xn_distinct : ∀ i j, i ≠ j → x i ≠ x j) 
  (h_pM_div_xn : ∀ n, n ≥ N → p^M ∣ x n): 
  ∃ N, ∀ n, n ≥ N → p^M ∣ x n :=
sorry

end dragos_wins_l51_51422


namespace consistent_values_l51_51870

theorem consistent_values (a x: ℝ) :
    (12 * x^2 + 48 * x - a + 36 = 0) ∧ ((a + 60) * x - 3 * (a - 20) = 0) ↔
    ((a = -12 ∧ x = -2) ∨ (a = 0 ∧ x = -1) ∨ (a = 180 ∧ x = 2)) := 
by
  -- proof steps should be filled here
  sorry

end consistent_values_l51_51870


namespace sequence_of_8_numbers_l51_51262

theorem sequence_of_8_numbers :
  ∃ (a b c d e f g h : ℤ), 
    a + b + c = 100 ∧ b + c + d = 100 ∧ c + d + e = 100 ∧ 
    d + e + f = 100 ∧ e + f + g = 100 ∧ f + g + h = 100 ∧ 
    a = 20 ∧ h = 16 ∧ 
    (a, b, c, d, e, f, g, h) = (20, 16, 64, 20, 16, 64, 20, 16) :=
by
  sorry

end sequence_of_8_numbers_l51_51262


namespace Mira_trips_to_fill_tank_l51_51173

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cube (a : ℝ) : ℝ :=
  a^3

noncomputable def number_of_trips (cube_side : ℝ) (sphere_diameter : ℝ) : ℕ :=
  let r := sphere_diameter / 2
  let sphere_volume := volume_of_sphere r
  let cube_volume := volume_of_cube cube_side
  Nat.ceil (cube_volume / sphere_volume)

theorem Mira_trips_to_fill_tank : number_of_trips 8 6 = 5 :=
by
  sorry

end Mira_trips_to_fill_tank_l51_51173


namespace eval_expr_l51_51068

theorem eval_expr : (3 : ℚ) / (2 - (5 / 4)) = 4 := by
  sorry

end eval_expr_l51_51068


namespace area_of_triangle_aef_l51_51024

noncomputable def length_ab : ℝ := 10
noncomputable def width_ad : ℝ := 6
noncomputable def diagonal_ac : ℝ := Real.sqrt (length_ab^2 + width_ad^2)
noncomputable def segment_length_ac : ℝ := diagonal_ac / 4
noncomputable def area_aef : ℝ := (1/2) * segment_length_ac * ((60 * diagonal_ac) / diagonal_ac^2)

theorem area_of_triangle_aef : area_aef = 7.5 := by
  sorry

end area_of_triangle_aef_l51_51024


namespace rowing_speed_in_still_water_l51_51610

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.4) (t : ℝ)
  (h2 : (v + c) * t = (v - c) * (2 * t)) : 
  v = 4.2 :=
by
  sorry

end rowing_speed_in_still_water_l51_51610


namespace point_inside_circle_range_l51_51454

theorem point_inside_circle_range (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) :=
  by
  sorry

end point_inside_circle_range_l51_51454


namespace find_angle_APB_l51_51382

-- Definitions based on conditions
def r1 := 2 -- Radius of semicircle SAR
def r2 := 3 -- Radius of semicircle RBT

def angle_AO1S := 70
def angle_BO2T := 40

def angle_AO1R := 180 - angle_AO1S
def angle_BO2R := 180 - angle_BO2T

def angle_PA := 90
def angle_PB := 90

-- Statement of the theorem
theorem find_angle_APB : angle_PA + angle_AO1R + angle_BO2R + angle_PB + 110 = 540 :=
by
  -- Unused in proof: added only to state theorem 
  have _ := angle_PA
  have _ := angle_AO1R
  have _ := angle_BO2R
  have _ := angle_PB
  have _ := 110
  sorry

end find_angle_APB_l51_51382


namespace max_count_larger_than_20_l51_51395

noncomputable def max_larger_than_20 (int_list : List Int) : Nat :=
  (int_list.filter (λ n => n > 20)).length

theorem max_count_larger_than_20 (a1 a2 a3 a4 a5 a6 a7 a8 : Int)
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 10) :
  ∃ (k : Nat), k = 7 ∧ max_larger_than_20 [a1, a2, a3, a4, a5, a6, a7, a8] = k :=
sorry

end max_count_larger_than_20_l51_51395


namespace compare_exp_sin_ln_l51_51501

theorem compare_exp_sin_ln :
  let a := Real.exp 0.1 - 1
  let b := Real.sin 0.1
  let c := Real.log 1.1
  c < b ∧ b < a :=
by
  sorry

end compare_exp_sin_ln_l51_51501


namespace polynomial_equality_l51_51264

theorem polynomial_equality (x y : ℝ) (h₁ : 3 * x + 2 * y = 6) (h₂ : 2 * x + 3 * y = 7) : 
  14 * x^2 + 25 * x * y + 14 * y^2 = 85 := 
by
  sorry

end polynomial_equality_l51_51264


namespace average_weight_increase_l51_51546

theorem average_weight_increase (A : ℝ) :
  let initial_total_weight := 10 * A
  let new_total_weight := initial_total_weight - 65 + 97
  let new_average := new_total_weight / 10
  let increase := new_average - A
  increase = 3.2 :=
by
  sorry

end average_weight_increase_l51_51546


namespace each_friend_gets_four_pieces_l51_51804

noncomputable def pieces_per_friend : ℕ :=
  let oranges := 80
  let pieces_per_orange := 10
  let friends := 200
  (oranges * pieces_per_orange) / friends

theorem each_friend_gets_four_pieces :
  pieces_per_friend = 4 :=
by
  sorry

end each_friend_gets_four_pieces_l51_51804


namespace right_triangle_area_l51_51226

theorem right_triangle_area :
  ∃ (a b c : ℕ), (c^2 = a^2 + b^2) ∧ (2 * b^2 - 23 * b + 11 = 0) ∧ (a * b / 2 = 330) :=
sorry

end right_triangle_area_l51_51226


namespace probability_one_defective_l51_51369

def total_bulbs : ℕ := 20
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs
def probability_non_defective_both : ℚ := (16 / 20) * (15 / 19)
def probability_at_least_one_defective : ℚ := 1 - probability_non_defective_both

theorem probability_one_defective :
  probability_at_least_one_defective = 7 / 19 :=
by
  sorry

end probability_one_defective_l51_51369


namespace intersection_equal_l51_51075

noncomputable def M := { y : ℝ | ∃ x : ℝ, y = Real.log (x + 1) / Real.log (1 / 2) ∧ x ≥ 3 }
noncomputable def N := { x : ℝ | x^2 + 2 * x - 3 ≤ 0 }

theorem intersection_equal : M ∩ N = {a : ℝ | -3 ≤ a ∧ a ≤ -2} :=
by
  sorry

end intersection_equal_l51_51075


namespace total_age_l51_51840

variable (A B : ℝ)

-- Conditions
def condition1 : Prop := A / B = 3 / 4
def condition2 : Prop := A - 10 = (1 / 2) * (B - 10)

-- Statement
theorem total_age : condition1 A B → condition2 A B → A + B = 35 := by
  sorry

end total_age_l51_51840


namespace quadrilateral_property_indeterminate_l51_51060

variable {α : Type*}
variable (Q A : α → Prop)

theorem quadrilateral_property_indeterminate :
  (¬ ∀ x, Q x → A x) → ¬ ((∃ x, Q x ∧ A x) ↔ False) :=
by
  intro h
  sorry

end quadrilateral_property_indeterminate_l51_51060


namespace line_intersects_circle_chord_min_length_l51_51366

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L based on parameter m
def L (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that for any real number m, line L intersects circle C at two points.
theorem line_intersects_circle (m : ℝ) : 
  ∃ x y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ C x y₁ ∧ C x y₂ ∧ L m x y₁ ∧ L m x y₂ :=
sorry

-- Prove the equation of line L in slope-intercept form when the chord cut by circle C has minimum length.
theorem chord_min_length : ∃ (m : ℝ), ∀ x y : ℝ, 
  L m x y ↔ y = 2 * x - 5 :=
sorry

end line_intersects_circle_chord_min_length_l51_51366


namespace find_x_set_l51_51051

theorem find_x_set (a : ℝ) (h : 0 < a ∧ a < 1) : 
  {x : ℝ | a ^ (x + 3) > a ^ (2 * x)} = {x : ℝ | x > 3} :=
sorry

end find_x_set_l51_51051


namespace standard_concession_l51_51069

theorem standard_concession (x : ℝ) : 
  (∀ (x : ℝ), (2000 - (x / 100) * 2000) - 0.2 * (2000 - (x / 100) * 2000) = 1120) → x = 30 := 
by 
  sorry

end standard_concession_l51_51069


namespace club_truncator_more_wins_than_losses_l51_51334

noncomputable def clubTruncatorWinsProbability : ℚ :=
  let total_matches := 8
  let prob := 1/3
  -- The combinatorial calculations for the balanced outcomes
  let balanced_outcomes := 70 + 560 + 420 + 28 + 1
  let total_outcomes := 3^total_matches
  let prob_balanced := balanced_outcomes / total_outcomes
  let prob_more_wins_or_more_losses := 1 - prob_balanced
  (prob_more_wins_or_more_losses / 2)

theorem club_truncator_more_wins_than_losses : 
  clubTruncatorWinsProbability = 2741 / 6561 := 
by 
  sorry

#check club_truncator_more_wins_than_losses

end club_truncator_more_wins_than_losses_l51_51334


namespace buddy_thursday_cards_l51_51932

-- Definitions from the given conditions
def monday_cards : ℕ := 30
def tuesday_cards : ℕ := monday_cards / 2
def wednesday_cards : ℕ := tuesday_cards + 12
def thursday_extra_cards : ℕ := tuesday_cards / 3
def thursday_cards : ℕ := wednesday_cards + thursday_extra_cards

-- Theorem to prove the total number of baseball cards on Thursday
theorem buddy_thursday_cards : thursday_cards = 32 :=
by
  -- Proof steps would go here, but we just provide the result for now
  sorry

end buddy_thursday_cards_l51_51932


namespace tangent_parallel_x_axis_tangent_45_degrees_x_axis_l51_51781

-- Condition: Define the curve
def curve (x : ℝ) : ℝ := x^2 - 1

-- Condition: Calculate derivative
def derivative_curve (x : ℝ) : ℝ := 2 * x

-- Part (a): Point where tangent is parallel to the x-axis
theorem tangent_parallel_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 0 ∧ x = 0 ∧ y = -1) :=
  sorry

-- Part (b): Point where tangent forms a 45 degree angle with the x-axis
theorem tangent_45_degrees_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 1 ∧ x = 1/2 ∧ y = -3/4) :=
  sorry

end tangent_parallel_x_axis_tangent_45_degrees_x_axis_l51_51781


namespace sad_girls_count_l51_51695

-- Statement of the problem in Lean 4
theorem sad_girls_count :
  ∀ (total_children happy_children sad_children neither_happy_nor_sad children boys girls happy_boys boys_neither_happy_nor_sad : ℕ),
    total_children = 60 →
    happy_children = 30 →
    sad_children = 10 →
    neither_happy_nor_sad = 20 →
    children = total_children →
    boys = 19 →
    girls = total_children - boys →
    happy_boys = 6 →
    boys_neither_happy_nor_sad = 7 →
    girls = 41 →
    sad_children = 10 →
    (sad_children = 6 + (total_children - boys - girls - neither_happy_nor_sad - happy_children)) → 
    ∃ sad_girls, sad_girls = 4 := by
  sorry

end sad_girls_count_l51_51695


namespace ages_of_Linda_and_Jane_l51_51159

theorem ages_of_Linda_and_Jane : 
  ∃ (J L : ℕ), 
    (L = 2 * J + 3) ∧ 
    (∃ (p : ℕ), Nat.Prime p ∧ p = L - J) ∧ 
    (L + J = 4 * J - 5) ∧ 
    (L = 19 ∧ J = 8) :=
by
  sorry

end ages_of_Linda_and_Jane_l51_51159


namespace find_remainder_mod_105_l51_51948

-- Define the conditions as a set of hypotheses
variables {n a b c : ℕ}
variables (hn : n > 0)
variables (ha : a < 3) (hb : b < 5) (hc : c < 7)
variables (h3 : n % 3 = a) (h5 : n % 5 = b) (h7 : n % 7 = c)
variables (heq : 4 * a + 3 * b + 2 * c = 30)

-- State the theorem
theorem find_remainder_mod_105 : n % 105 = 29 :=
by
  -- Hypotheses block for documentation
  have ha_le : 0 ≤ a := sorry
  have hb_le : 0 ≤ b := sorry
  have hc_le : 0 ≤ c := sorry
  sorry

end find_remainder_mod_105_l51_51948


namespace x_finishes_remaining_work_in_14_days_l51_51658

-- Define the work rates of X and Y
def work_rate_X : ℚ := 1 / 21
def work_rate_Y : ℚ := 1 / 15

-- Define the amount of work Y completed in 5 days
def work_done_by_Y_in_5_days : ℚ := 5 * work_rate_Y

-- Define the remaining work after Y left
def remaining_work : ℚ := 1 - work_done_by_Y_in_5_days

-- Define the number of days needed for X to finish the remaining work
def x_days_remaining : ℚ := remaining_work / work_rate_X

-- Statement to prove
theorem x_finishes_remaining_work_in_14_days : x_days_remaining = 14 := by
  sorry

end x_finishes_remaining_work_in_14_days_l51_51658


namespace simplify_exponential_expression_l51_51007

theorem simplify_exponential_expression :
  (3 * (-5)^2)^(3/4) = (75)^(3/4) := 
  sorry

end simplify_exponential_expression_l51_51007


namespace grape_juice_amount_l51_51911

theorem grape_juice_amount 
  (T : ℝ) -- total amount of the drink 
  (orange_juice_percentage watermelon_juice_percentage : ℝ) -- percentages 
  (combined_amount_of_oj_wj : ℝ) -- combined amount of orange and watermelon juice 
  (h1 : orange_juice_percentage = 0.15)
  (h2 : watermelon_juice_percentage = 0.60)
  (h3 : combined_amount_of_oj_wj = 120)
  (h4 : combined_amount_of_oj_wj = (orange_juice_percentage + watermelon_juice_percentage) * T) : 
  (T * (1 - (orange_juice_percentage + watermelon_juice_percentage)) = 40) := 
sorry

end grape_juice_amount_l51_51911


namespace find_q_l51_51654

noncomputable def has_two_distinct_negative_roots (q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
  (x₁ ^ 4 + q * x₁ ^ 3 + 2 * x₁ ^ 2 + q * x₁ + 4 = 0) ∧ 
  (x₂ ^ 4 + q * x₂ ^ 3 + 2 * x₂ ^ 2 + q * x₂ + 4 = 0)

theorem find_q (q : ℝ) : 
  has_two_distinct_negative_roots q ↔ q ≤ 3 / Real.sqrt 2 := sorry

end find_q_l51_51654


namespace solution_set_l51_51432

-- Define the conditions
variables {f : ℝ → ℝ}

-- Condition: f(x) is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x

-- Condition: xf'(x) + f(x) < 0 for x in (-∞, 0)
axiom condition1 : ∀ x : ℝ, x < 0 → x * (deriv f x) + f x < 0

-- Condition: f(-2) = 0
axiom f_neg2_zero : f (-2) = 0

-- Goal: Prove the solution set of the inequality xf(x) < 0 is {x | -2 < x < 0 ∨ 0 < x < 2}
theorem solution_set : ∀ x : ℝ, (x * f x < 0) ↔ (-2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2) := by
  sorry

end solution_set_l51_51432


namespace total_copies_in_half_hour_l51_51526

-- Definitions of the machine rates and their time segments.
def machine1_rate := 35 -- copies per minute
def machine2_rate := 65 -- copies per minute
def machine3_rate1 := 50 -- copies per minute for the first 15 minutes
def machine3_rate2 := 80 -- copies per minute for the next 15 minutes
def machine4_rate1 := 90 -- copies per minute for the first 10 minutes
def machine4_rate2 := 60 -- copies per minute for the next 20 minutes

-- Time intervals for different machines
def machine3_time1 := 15 -- minutes
def machine3_time2 := 15 -- minutes
def machine4_time1 := 10 -- minutes
def machine4_time2 := 20 -- minutes

-- Proof statement
theorem total_copies_in_half_hour : 
  (machine1_rate * 30) + 
  (machine2_rate * 30) + 
  ((machine3_rate1 * machine3_time1) + (machine3_rate2 * machine3_time2)) + 
  ((machine4_rate1 * machine4_time1) + (machine4_rate2 * machine4_time2)) = 
  7050 :=
by 
  sorry

end total_copies_in_half_hour_l51_51526


namespace current_population_correct_l51_51712

def initial_population : ℕ := 4079
def percentage_died : ℕ := 5
def percentage_left : ℕ := 15

def calculate_current_population (initial_population : ℕ) (percentage_died : ℕ) (percentage_left : ℕ) : ℕ :=
  let died := (initial_population * percentage_died) / 100
  let remaining_after_bombardment := initial_population - died
  let left := (remaining_after_bombardment * percentage_left) / 100
  remaining_after_bombardment - left

theorem current_population_correct : calculate_current_population initial_population percentage_died percentage_left = 3295 :=
  by
  unfold calculate_current_population
  sorry

end current_population_correct_l51_51712


namespace average_age_of_4_students_l51_51498

theorem average_age_of_4_students (avg_age_15 : ℕ) (num_students_15 : ℕ)
    (avg_age_10 : ℕ) (num_students_10 : ℕ) (age_15th_student : ℕ) :
    avg_age_15 = 15 ∧ num_students_15 = 15 ∧ avg_age_10 = 16 ∧ num_students_10 = 10 ∧ age_15th_student = 9 → 
    (56 / 4 = 14) := by
  sorry

end average_age_of_4_students_l51_51498


namespace average_marks_all_students_l51_51481

theorem average_marks_all_students
  (n1 n2 : ℕ)
  (avg1 avg2 : ℕ)
  (h1 : avg1 = 40)
  (h2 : avg2 = 80)
  (h3 : n1 = 30)
  (h4 : n2 = 50) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 65 :=
by
  sorry

end average_marks_all_students_l51_51481


namespace f_2023_value_l51_51752

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : 2^n = a + b) : f a + f b = n^2 + 1

theorem f_2023_value : f 2023 = 107 :=
by 
  sorry

end f_2023_value_l51_51752


namespace apples_kilos_first_scenario_l51_51396

noncomputable def cost_per_kilo_oranges : ℝ := 29
noncomputable def cost_per_kilo_apples : ℝ := 29
noncomputable def cost_first_scenario : ℝ := 419
noncomputable def cost_second_scenario : ℝ := 488
noncomputable def kilos_oranges_first_scenario : ℝ := 6
noncomputable def kilos_oranges_second_scenario : ℝ := 5
noncomputable def kilos_apples_second_scenario : ℝ := 7

theorem apples_kilos_first_scenario
  (O A : ℝ) 
  (cost1 cost2 : ℝ) 
  (k_oranges1 k_oranges2 k_apples2 : ℝ) 
  (hO : O = 29) (hA : A = 29) 
  (hCost1 : k_oranges1 * O + x * A = cost1) 
  (hCost2 : k_oranges2 * O + k_apples2 * A = cost2) 
  : x = 8 :=
by
  have hO : O = 29 := sorry
  have hA : A = 29 := sorry
  have h1 : k_oranges1 * O + x * A = cost1 := sorry
  have h2 : k_oranges2 * O + k_apples2 * A = cost2 := sorry
  sorry

end apples_kilos_first_scenario_l51_51396


namespace range_of_f_1_over_f_2_l51_51905

theorem range_of_f_1_over_f_2 {f : ℝ → ℝ} (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  1 / 8 < f 1 / f 2 ∧ f 1 / f 2 < 1 / 4 :=
by sorry

end range_of_f_1_over_f_2_l51_51905


namespace bee_honeycomb_path_l51_51229

theorem bee_honeycomb_path (x1 x2 x3 : ℕ) (honeycomb_grid : Prop)
  (shortest_path : ℕ) (honeycomb_property : shortest_path = 100)
  (path_decomposition : x1 + x2 + x3 = 100) : x1 = 50 ∧ x2 + x3 = 50 := 
sorry

end bee_honeycomb_path_l51_51229


namespace arithmetic_progression_num_terms_l51_51637

theorem arithmetic_progression_num_terms (a d n : ℕ) (h_even : n % 2 = 0) 
    (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 30)
    (h_sum_even : (n / 2) * (2 * a + 2 * d + (n - 2) * d) = 36)
    (h_diff_last_first : (n - 1) * d = 12) :
    n = 8 := 
sorry

end arithmetic_progression_num_terms_l51_51637


namespace number_of_valid_n_l51_51550

theorem number_of_valid_n : 
  ∃ (c : Nat), (∀ n : Nat, (n + 9) * (n - 4) * (n - 13) < 0 → n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 ∨ n = 12) ∧ c = 11 :=
by
  sorry

end number_of_valid_n_l51_51550


namespace blocks_needed_for_enclosure_l51_51031

noncomputable def volume_of_rectangular_prism (length: ℝ) (width: ℝ) (height: ℝ) : ℝ :=
  length * width * height

theorem blocks_needed_for_enclosure 
  (length width height thickness : ℝ)
  (H_length : length = 15)
  (H_width : width = 12)
  (H_height : height = 6)
  (H_thickness : thickness = 1.5) :
  volume_of_rectangular_prism length width height - 
  volume_of_rectangular_prism (length - 2 * thickness) (width - 2 * thickness) (height - thickness) = 594 :=
by
  sorry

end blocks_needed_for_enclosure_l51_51031


namespace price_of_skateboard_l51_51015

-- Given condition (0.20 * p = 300)
variable (p : ℝ)
axiom upfront_payment : 0.20 * p = 300

-- Theorem statement to prove the price of the skateboard
theorem price_of_skateboard : p = 1500 := by
  sorry

end price_of_skateboard_l51_51015


namespace sam_read_pages_l51_51845

-- Define conditions
def assigned_pages : ℕ := 25
def harrison_pages : ℕ := assigned_pages + 10
def pam_pages : ℕ := harrison_pages + 15
def sam_pages : ℕ := 2 * pam_pages

-- Prove the target theorem
theorem sam_read_pages : sam_pages = 100 := by
  sorry

end sam_read_pages_l51_51845


namespace find_number_l51_51831

theorem find_number (x : ℝ) (hx : (50 + 20 / x) * x = 4520) : x = 90 :=
sorry

end find_number_l51_51831


namespace area_of_circle_l51_51817

theorem area_of_circle :
  (∃ (x y : ℝ), x^2 + y^2 - 6 * x + 8 * y = -9) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  -- We need to prove the area is 16π
  sorry

end area_of_circle_l51_51817


namespace max_value_3absx_2absy_l51_51789

theorem max_value_3absx_2absy (x y : ℝ) (h : x^2 + y^2 = 9) : 
  3 * abs x + 2 * abs y ≤ 9 :=
sorry

end max_value_3absx_2absy_l51_51789


namespace evaluate_expression_l51_51739

theorem evaluate_expression :
  2 ^ (0 ^ (1 ^ 9)) + ((2 ^ 0) ^ 1) ^ 9 = 2 := 
sorry

end evaluate_expression_l51_51739


namespace projectile_height_35_l51_51743

noncomputable def projectile_height (t : ℝ) : ℝ := -4.9 * t^2 + 30 * t

theorem projectile_height_35 (t : ℝ) :
  projectile_height t = 35 ↔ t = 10/7 :=
by {
  sorry
}

end projectile_height_35_l51_51743


namespace total_trees_cut_down_l51_51590

-- Definitions based on conditions in the problem
def trees_per_day_james : ℕ := 20
def days_with_just_james : ℕ := 2
def total_trees_by_james := trees_per_day_james * days_with_just_james

def brothers : ℕ := 2
def days_with_brothers : ℕ := 3
def trees_per_day_brothers := (20 * (100 - 20)) / 100 -- 20% fewer than James
def trees_per_day_total := brothers * trees_per_day_brothers + trees_per_day_james

def total_trees_with_brothers := trees_per_day_total * days_with_brothers

-- The statement to be proved
theorem total_trees_cut_down : total_trees_by_james + total_trees_with_brothers = 136 := by
  sorry

end total_trees_cut_down_l51_51590


namespace product_of_bc_l51_51673

theorem product_of_bc
  (b c : Int)
  (h1 : ∀ r, r^2 - r - 1 = 0 → r^5 - b * r - c = 0) :
  b * c = 15 :=
by
  -- We start the proof assuming the conditions
  sorry

end product_of_bc_l51_51673


namespace not_age_of_child_l51_51753

theorem not_age_of_child (ages : Set ℕ) (h_ages : ∀ x ∈ ages, 4 ≤ x ∧ x ≤ 10) : 
  5 ∉ ages := by
  let number := 1122
  have h_number : number % 5 ≠ 0 := by decide
  have h_divisible : ∀ x ∈ ages, number % x = 0 := sorry
  exact sorry

end not_age_of_child_l51_51753


namespace complement_of_angle_A_l51_51954

theorem complement_of_angle_A (A : ℝ) (h : A = 76) : 90 - A = 14 := by
  sorry

end complement_of_angle_A_l51_51954


namespace range_of_a_l51_51468

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a ∈ Set.Icc (-1 : ℝ) 3 :=
by
  sorry

end range_of_a_l51_51468


namespace work_together_l51_51603

theorem work_together (A_rate B_rate : ℝ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) : (1 / (A_rate + B_rate) = 6) :=
by
  -- we only need to write the statement, proof is not required.
  sorry

end work_together_l51_51603


namespace polynomial_solution_l51_51890

variable (P : ℚ) -- Assuming P is a constant polynomial

theorem polynomial_solution (P : ℚ) 
  (condition : P + (2 : ℚ) * X^2 + (5 : ℚ) * X - (2 : ℚ) = (2 : ℚ) * X^2 + (5 : ℚ) * X + (4 : ℚ)): 
  P = 6 := 
  sorry

end polynomial_solution_l51_51890


namespace perpendicular_lines_condition_l51_51453

theorem perpendicular_lines_condition (m : ℝ) :
  (m = -1) ↔ ((m * 2 + 1 * m * (m - 1)) = 0) :=
sorry

end perpendicular_lines_condition_l51_51453


namespace cheese_cookie_packs_l51_51568

def packs_per_box (P : ℕ) : Prop :=
  let cartons := 12
  let boxes_per_carton := 12
  let total_boxes := cartons * boxes_per_carton
  let total_cost := 1440
  let box_cost := total_cost / total_boxes
  let pack_cost := 1
  P = box_cost / pack_cost

theorem cheese_cookie_packs : packs_per_box 10 := by
  sorry

end cheese_cookie_packs_l51_51568


namespace prime_iff_satisfies_condition_l51_51511

def satisfies_condition (n : ℕ) : Prop :=
  if n = 2 then True
  else if 2 < n then ∀ k : ℕ, 2 ≤ k ∧ k < n → ¬ (k ∣ n)
  else False

theorem prime_iff_satisfies_condition (n : ℕ) : Prime n ↔ satisfies_condition n := by
  sorry

end prime_iff_satisfies_condition_l51_51511


namespace arithmetic_sequence_general_term_l51_51510

theorem arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  (∃ d : ℚ, d = -1/2 ∧ ∀ n ≥ 2, (1 / S n) - (1 / S (n - 1)) = d) :=
sorry

theorem general_term (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  ∀ n, a n = if n = 1 then 3 else 18 / ((8 - 3 * n) * (5 - 3 * n)) :=
sorry

end arithmetic_sequence_general_term_l51_51510


namespace hiker_total_distance_l51_51417

theorem hiker_total_distance :
  let day1_distance := 18
  let day1_speed := 3
  let day2_speed := day1_speed + 1
  let day1_time := day1_distance / day1_speed
  let day2_time := day1_time - 1
  let day2_distance := day2_speed * day2_time
  let day3_speed := 5
  let day3_time := 3
  let day3_distance := day3_speed * day3_time
  let total_distance := day1_distance + day2_distance + day3_distance
  total_distance = 53 :=
by
  sorry

end hiker_total_distance_l51_51417


namespace prime_product_correct_l51_51295

theorem prime_product_correct 
    (p1 : Nat := 1021031) (pr1 : Prime p1)
    (p2 : Nat := 237019) (pr2 : Prime p2) :
    p1 * p2 = 241940557349 :=
by
  sorry

end prime_product_correct_l51_51295


namespace most_likely_wins_l51_51552

theorem most_likely_wins {N : ℕ} (h : N > 0) :
  let p := 1 / 2
  let n := 2 * N
  let E := n * p
  E = N := 
by
  sorry

end most_likely_wins_l51_51552


namespace larger_number_is_50_l51_51086

theorem larger_number_is_50 (x y : ℤ) (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 :=
sorry

end larger_number_is_50_l51_51086


namespace A_and_B_mutually_exclusive_l51_51522

-- Definitions of events based on conditions
def A (a : ℕ) : Prop := a = 3
def B (a : ℕ) : Prop := a = 4

-- Define mutually exclusive
def mutually_exclusive (P Q : ℕ → Prop) : Prop :=
  ∀ a, P a → Q a → false

-- Problem statement: Prove A and B are mutually exclusive.
theorem A_and_B_mutually_exclusive :
  mutually_exclusive A B :=
sorry

end A_and_B_mutually_exclusive_l51_51522


namespace james_used_5_containers_l51_51320

-- Conditions
def initial_balls : ℕ := 100
def balls_given_away : ℕ := initial_balls / 2
def remaining_balls : ℕ := initial_balls - balls_given_away
def balls_per_container : ℕ := 10

-- Question (statement of the theorem to prove)
theorem james_used_5_containers : (remaining_balls / balls_per_container) = 5 := by
  sorry

end james_used_5_containers_l51_51320


namespace ae_length_l51_51532

theorem ae_length (AB CD AC AE : ℝ) (h: 2 * AE + 3 * AE = 34): 
  AE = 34 / 5 := by
  -- Proof steps will go here
  sorry

end ae_length_l51_51532


namespace tent_ratio_l51_51150

-- Define variables for tents in different parts of the camp
variables (N E C S T : ℕ)

-- Given conditions
def northernmost (N : ℕ) := N = 100
def center (C N : ℕ) := C = 4 * N
def southern (S : ℕ) := S = 200
def total (T N C E S : ℕ) := T = N + C + E + S

-- Main theorem statement for the proof
theorem tent_ratio (N E C S T : ℕ) 
  (hn : northernmost N)
  (hc : center C N) 
  (hs : southern S)
  (ht : total T N C E S) :
  E / N = 2 :=
by sorry

end tent_ratio_l51_51150


namespace pizzeria_provolone_shred_l51_51154

theorem pizzeria_provolone_shred 
    (cost_blend : ℝ) 
    (cost_mozzarella : ℝ) 
    (cost_romano : ℝ) 
    (cost_provolone : ℝ) 
    (prop_mozzarella : ℝ) 
    (prop_romano : ℝ) 
    (prop_provolone : ℝ) 
    (shredded_mozzarella : ℕ) 
    (shredded_romano : ℕ) 
    (shredded_provolone_needed : ℕ) :
  cost_blend = 696.05 ∧ 
  cost_mozzarella = 504.35 ∧ 
  cost_romano = 887.75 ∧ 
  cost_provolone = 735.25 ∧ 
  prop_mozzarella = 2 ∧ 
  prop_romano = 1 ∧ 
  prop_provolone = 2 ∧ 
  shredded_mozzarella = 20 ∧ 
  shredded_romano = 10 → 
  shredded_provolone_needed = 20 :=
by {
  sorry -- proof to be provided
}

end pizzeria_provolone_shred_l51_51154


namespace sphere_segment_volume_l51_51784

theorem sphere_segment_volume (r : ℝ) (ratio_surface_to_base : ℝ) : r = 10 → ratio_surface_to_base = 10 / 7 → ∃ V : ℝ, V = 288 * π :=
by
  intros
  sorry

end sphere_segment_volume_l51_51784


namespace congruence_problem_l51_51506

theorem congruence_problem (x : ℤ) (h : 5 * x + 9 ≡ 4 [ZMOD 18]) : 3 * x + 15 ≡ 12 [ZMOD 18] :=
sorry

end congruence_problem_l51_51506


namespace find_k_and_slope_l51_51234

theorem find_k_and_slope : 
  ∃ k : ℝ, (∃ y : ℝ, (3 + y = 8) ∧ (k = -3 * 3 + y)) ∧ (k = -4) ∧ 
  (∀ x y : ℝ, (x + y = 8) → (∃ m b : ℝ, y = m * x + b ∧ m = -1)) :=
by {
  sorry
}

end find_k_and_slope_l51_51234


namespace find_x_l51_51824

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 6) : x = 14 :=
by
  sorry

end find_x_l51_51824


namespace mutually_exclusive_necessary_for_complementary_l51_51732

variables {Ω : Type} -- Define the sample space type
variables (A1 A2 : Ω → Prop) -- Define the events as predicates over the sample space

-- Define mutually exclusive events
def mutually_exclusive (A1 A2 : Ω → Prop) : Prop :=
∀ ω, A1 ω → ¬ A2 ω

-- Define complementary events
def complementary (A1 A2 : Ω → Prop) : Prop :=
∀ ω, (A1 ω ↔ ¬ A2 ω)

-- The proof problem: Statement 1 is a necessary but not sufficient condition for Statement 2
theorem mutually_exclusive_necessary_for_complementary (A1 A2 : Ω → Prop) :
  (mutually_exclusive A1 A2) → (complementary A1 A2) → (mutually_exclusive A1 A2) ∧ ¬ (complementary A1 A2 → mutually_exclusive A1 A2) :=
sorry

end mutually_exclusive_necessary_for_complementary_l51_51732


namespace range_of_a_l51_51589

noncomputable def A (x : ℝ) : Prop := (3 * x) / (x + 1) ≤ 2
noncomputable def B (x a : ℝ) : Prop := a - 2 < x ∧ x < 2 * a + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, A x ↔ B x a) ↔ (1 / 2 < a ∧ a ≤ 1) := by
sorry

end range_of_a_l51_51589


namespace positive_number_y_l51_51571

theorem positive_number_y (y : ℕ) (h1 : y > 0) (h2 : y^2 / 100 = 9) : y = 30 :=
by
  sorry

end positive_number_y_l51_51571


namespace average_of_roots_l51_51192

theorem average_of_roots (p q : ℝ) (h : ∃ x1 x2 : ℝ, 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0 ∧ x1 ≠ x2):
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0) → 
  (x1 + x2) / 2 = 1 :=
by
  sorry

end average_of_roots_l51_51192


namespace range_of_m_l51_51878

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end range_of_m_l51_51878


namespace de_morgan_birth_year_jenkins_birth_year_l51_51463

open Nat

theorem de_morgan_birth_year
  (x : ℕ) (hx : x = 43) (hx_square : x * x = 1849) :
  1849 - 43 = 1806 :=
by
  sorry

theorem jenkins_birth_year
  (a b : ℕ) (ha : a = 5) (hb : b = 6) (m : ℕ) (hm : m = 31) (n : ℕ) (hn : n = 5)
  (ha_sq : a * a = 25) (hb_sq : b * b = 36) (ha4 : a * a * a * a = 625)
  (hb4 : b * b * b * b = 1296) (hm2 : m * m = 961) (hn4 : n * n * n * n = 625) :
  1921 - 61 = 1860 ∧
  1922 - 62 = 1860 ∧
  1875 - 15 = 1860 :=
by
  sorry

end de_morgan_birth_year_jenkins_birth_year_l51_51463


namespace cats_new_total_weight_l51_51612

noncomputable def total_weight (weights : List ℚ) : ℚ :=
  weights.sum

noncomputable def remove_min_max_weight (weights : List ℚ) : ℚ :=
  let min_weight := weights.minimum?.getD 0
  let max_weight := weights.maximum?.getD 0
  weights.sum - min_weight - max_weight

theorem cats_new_total_weight :
  let weights := [3.5, 7.2, 4.8, 6, 5.5, 9, 4]
  remove_min_max_weight weights = 27.5 := by
  sorry

end cats_new_total_weight_l51_51612


namespace roger_earned_54_dollars_l51_51649

-- Definitions based on problem conditions
def lawns_had : ℕ := 14
def lawns_forgot : ℕ := 8
def earn_per_lawn : ℕ := 9

-- The number of lawns actually mowed
def lawns_mowed : ℕ := lawns_had - lawns_forgot

-- The amount of money earned
def money_earned : ℕ := lawns_mowed * earn_per_lawn

-- Proof statement: Roger actually earned 54 dollars
theorem roger_earned_54_dollars : money_earned = 54 := sorry

end roger_earned_54_dollars_l51_51649


namespace original_book_pages_l51_51420

theorem original_book_pages (n k : ℕ) (h1 : (n * (n + 1)) / 2 - (2 * k + 1) = 4979)
: n = 100 :=
by
  sorry

end original_book_pages_l51_51420


namespace horizontal_asymptote_l51_51531

def numerator (x : ℝ) : ℝ :=
  15 * x^4 + 3 * x^3 + 7 * x^2 + 6 * x + 2

def denominator (x : ℝ) : ℝ :=
  5 * x^4 + x^3 + 4 * x^2 + 2 * x + 1

noncomputable def rational_function (x : ℝ) : ℝ :=
  numerator x / denominator x

theorem horizontal_asymptote :
  ∃ y : ℝ, (∀ x : ℝ, x ≠ 0 → rational_function x = y) ↔ y = 3 :=
by
  sorry

end horizontal_asymptote_l51_51531


namespace prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l51_51223

noncomputable def market_supply_function (P : ℝ) : ℝ := 6 * P - 312

noncomputable def market_demand_function (a b P : ℝ) : ℝ := a - b * P

noncomputable def price_elasticity_supply (P_e Q_e : ℝ) : ℝ := 6 * (P_e / Q_e)

noncomputable def price_elasticity_demand (b P_e Q_e : ℝ) : ℝ := -b * (P_e / Q_e)

noncomputable def tax_rate := 30

noncomputable def consumer_price_after_tax := 118

theorem prove_market_demand (a P_e Q_e : ℝ) :
  1.5 * |price_elasticity_demand 4 P_e Q_e| = price_elasticity_supply P_e Q_e →
  market_demand_function a 4 P_e = a - 4 * P_e := sorry

theorem prove_tax_revenue (Q_d : ℝ) :
  Q_d = 216 →
  Q_d * tax_rate = 6480 := sorry

theorem prove_per_unit_tax_rate (t : ℝ) :
  t = 60 → 4 * t = 240 := sorry

theorem prove_tax_revenue_specified (t : ℝ) :
  t = 60 →
  (288 * t - 2.4 * t^2) = 8640 := sorry

end prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l51_51223


namespace angle_A_is_pi_div_3_length_b_l51_51921

open Real

theorem angle_A_is_pi_div_3
  (A B C : ℝ) (a b c : ℝ)
  (hABC : A + B + C = π)
  (m : ℝ × ℝ) (n : ℝ × ℝ)
  (hm : m = (sqrt 3, cos (π - A) - 1))
  (hn : n = (cos (π / 2 - A), 1))
  (horthogonal : m.1 * n.1 + m.2 * n.2 = 0) :
  A = π / 3 := 
sorry

theorem length_b 
  (A B : ℝ) (a b : ℝ)
  (hA : A = π / 3)
  (ha : a = 2)
  (hcosB : cos B = sqrt 3 / 3) :
  b = 4 * sqrt 2 / 3 :=
sorry

end angle_A_is_pi_div_3_length_b_l51_51921


namespace find_min_value_expression_l51_51632

noncomputable def minValueExpression (θ : ℝ) : ℝ :=
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ

theorem find_min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 ∧ minValueExpression θ = 3 * Real.sqrt 2 :=
sorry

end find_min_value_expression_l51_51632


namespace sum_six_consecutive_integers_l51_51064

-- Statement of the problem
theorem sum_six_consecutive_integers (n : ℤ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l51_51064


namespace quadratic_function_order_l51_51178

theorem quadratic_function_order (a b c : ℝ) (h_neg_a : a < 0) 
  (h_sym : ∀ x, (a * (x + 2)^2 + b * (x + 2) + c) = (a * (2 - x)^2 + b * (2 - x) + c)) :
  (a * (-1992)^2 + b * (-1992) + c) < (a * (1992)^2 + b * (1992) + c) ∧
  (a * (1992)^2 + b * (1992) + c) < (a * (0)^2 + b * (0) + c) :=
by
  sorry

end quadratic_function_order_l51_51178


namespace even_numbers_set_l51_51898

-- Define the set of all even numbers in set-builder notation
def even_set : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- Theorem stating that this set is the set of all even numbers
theorem even_numbers_set :
  ∀ x : ℤ, (x ∈ even_set ↔ ∃ n : ℤ, x = 2 * n) := by
  sorry

end even_numbers_set_l51_51898


namespace solve_x_l51_51917

theorem solve_x 
  (x : ℝ) 
  (h : (2 / x) + (3 / x) / (6 / x) = 1.25) : 
  x = 8 / 3 := 
sorry

end solve_x_l51_51917


namespace partial_fraction_sum_l51_51148

theorem partial_fraction_sum :
  (∃ A B C D E : ℝ, 
    (∀ x : ℝ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -5 → 
    (1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5))) ∧
    (A + B + C + D + E = 1 / 30)) :=
sorry

end partial_fraction_sum_l51_51148


namespace common_ratio_of_geometric_seq_l51_51609

variable {a : ℕ → ℚ} -- The sequence
variable {d : ℚ} -- Common difference

-- Assuming the arithmetic and geometric sequence properties
def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_seq (a1 a4 a5 : ℚ) (q : ℚ) : Prop :=
  a4 = a1 * q ∧ a5 = a4 * q

theorem common_ratio_of_geometric_seq (h_arith: is_arithmetic_seq a d) (h_nonzero_d : d ≠ 0)
  (h_geometric: is_geometric_seq (a 1) (a 4) (a 5) (1 / 3)) : (a 4 / a 1) = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_seq_l51_51609


namespace x_coordinate_of_first_point_l51_51981

theorem x_coordinate_of_first_point (m n : ℝ) :
  (m = 2 * n + 3) ↔ (∃ (p1 p2 : ℝ × ℝ), p1 = (m, n) ∧ p2 = (m + 2, n + 1) ∧ 
    (p1.1 = 2 * p1.2 + 3) ∧ (p2.1 = 2 * p2.2 + 3)) :=
by
  sorry

end x_coordinate_of_first_point_l51_51981


namespace product_of_numbers_l51_51875

variable {x y : ℝ}

theorem product_of_numbers (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 40 * k) : 
  x * y = 6400 / 63 := by
  sorry

end product_of_numbers_l51_51875


namespace x_minus_y_options_l51_51502

theorem x_minus_y_options (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) :
  (x - y ≠ 2013) ∧ (x - y ≠ 2014) ∧ (x - y ≠ 2015) ∧ (x - y ≠ 2016) := 
sorry

end x_minus_y_options_l51_51502


namespace correct_statements_truth_of_statements_l51_51601

-- Define basic properties related to factor and divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Given conditions as definitions
def condition_A : Prop := is_factor 4 100
def condition_B1 : Prop := is_divisor 19 133
def condition_B2 : Prop := ¬ is_divisor 19 51
def condition_C1 : Prop := is_divisor 30 90
def condition_C2 : Prop := ¬ is_divisor 30 53
def condition_D1 : Prop := is_divisor 7 21
def condition_D2 : Prop := ¬ is_divisor 7 49
def condition_E : Prop := is_factor 10 200

-- Statement that needs to be proved
theorem correct_statements : 
  (condition_A ∧ 
  (condition_B1 ∧ condition_B2) ∧ 
  condition_E) :=
by sorry -- proof to be inserted

-- Equivalent Lean 4 statement with all conditions encapsulated
theorem truth_of_statements :
  (is_factor 4 100) ∧ 
  (is_divisor 19 133 ∧ ¬ is_divisor 19 51) ∧ 
  is_factor 10 200 :=
by sorry -- proof to be inserted

end correct_statements_truth_of_statements_l51_51601


namespace find_vector_c_l51_51442

-- Definitions of the given vectors
def vector_a : ℝ × ℝ := (3, -1)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (2 * vector_a.1 + vector_b.1, 2 * vector_a.2 + vector_b.2)

-- The goal is to prove that vector_c = (5, 0)
theorem find_vector_c : vector_c = (5, 0) :=
by
  -- proof steps would go here
  sorry

end find_vector_c_l51_51442


namespace minimum_floor_sum_l51_51639

theorem minimum_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ⌊a^2 + b^2 / c⌋ + ⌊b^2 + c^2 / a⌋ + ⌊c^2 + a^2 / b⌋ = 34 :=
sorry

end minimum_floor_sum_l51_51639


namespace sum_of_digits_joeys_age_l51_51705

-- Given conditions
variables (C : ℕ) (J : ℕ := C + 2) (Z : ℕ := 1)

-- Define the condition that the sum of Joey's and Chloe's ages will be an integral multiple of Zoe's age.
def sum_is_multiple_of_zoe (n : ℕ) : Prop :=
  ∃ k : ℕ, (J + C) = k * Z

-- Define the problem of finding the sum of digits the first time Joey's age alone is a multiple of Zoe's age.
def sum_of_digits_first_multiple (J Z : ℕ) : ℕ :=
  (J / 10) + (J % 10)

-- The theorem we need to prove
theorem sum_of_digits_joeys_age : (sum_of_digits_first_multiple J Z = 1) :=
sorry

end sum_of_digits_joeys_age_l51_51705


namespace diana_age_is_8_l51_51062

noncomputable def age_of_grace_last_year : ℕ := 3
noncomputable def age_of_grace_today : ℕ := age_of_grace_last_year + 1
noncomputable def age_of_diana_today : ℕ := 2 * age_of_grace_today

theorem diana_age_is_8 : age_of_diana_today = 8 :=
by
  -- The proof would go here
  sorry

end diana_age_is_8_l51_51062


namespace union_inter_distrib_inter_union_distrib_l51_51293

section
variables {α : Type*} (A B C : Set α)

-- Problem (a)
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) :=
sorry

-- Problem (b)
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) :=
sorry
end

end union_inter_distrib_inter_union_distrib_l51_51293


namespace isabel_spending_ratio_l51_51866

theorem isabel_spending_ratio :
  ∀ (initial_amount toy_cost remaining_amount : ℝ),
    initial_amount = 204 ∧
    toy_cost = initial_amount / 2 ∧
    remaining_amount = 51 →
    ((initial_amount - toy_cost - remaining_amount) / remaining_amount) = 1 / 2 :=
by
  intros
  sorry

end isabel_spending_ratio_l51_51866


namespace base_12_addition_l51_51298

theorem base_12_addition (A B: ℕ) (hA: A = 10) (hB: B = 11) : 
  8 * 12^2 + A * 12 + 2 + (3 * 12^2 + B * 12 + 7) = 1 * 12^3 + 0 * 12^2 + 9 * 12 + 9 := 
by
  sorry

end base_12_addition_l51_51298


namespace coeff_x2_in_PQ_is_correct_l51_51488

variable (c : ℝ)

def P (x : ℝ) : ℝ := 2 * x^3 + 4 * x^2 - 3 * x + 1
def Q (x : ℝ) : ℝ := 3 * x^3 + c * x^2 - 8 * x - 5

def coeff_x2 (x : ℝ) : ℝ := -20 - 2 * c

theorem coeff_x2_in_PQ_is_correct :
  (4 : ℝ) * (-5) + (-3) * c + c = -20 - 2 * c := by
  sorry

end coeff_x2_in_PQ_is_correct_l51_51488


namespace binary_sum_correct_l51_51825

-- Definitions of the binary numbers
def bin1 : ℕ := 0b1011
def bin2 : ℕ := 0b101
def bin3 : ℕ := 0b11001
def bin4 : ℕ := 0b1110
def bin5 : ℕ := 0b100101

-- The statement to prove
theorem binary_sum_correct : bin1 + bin2 + bin3 + bin4 + bin5 = 0b1111010 := by
  sorry

end binary_sum_correct_l51_51825


namespace mark_total_cost_is_correct_l51_51773

variable (hours : ℕ) (hourly_rate part_cost : ℕ)

def total_cost (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) :=
  hours * hourly_rate + part_cost

theorem mark_total_cost_is_correct : 
  hours = 2 → hourly_rate = 75 → part_cost = 150 → total_cost hours hourly_rate part_cost = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end mark_total_cost_is_correct_l51_51773


namespace extra_days_per_grade_below_b_l51_51707

theorem extra_days_per_grade_below_b :
  ∀ (total_days lying_days grades_below_B : ℕ), 
  total_days = 26 → lying_days = 14 → grades_below_B = 4 → 
  (total_days - lying_days) / grades_below_B = 3 :=
by
  -- conditions and steps of the proof will be here
  sorry

end extra_days_per_grade_below_b_l51_51707


namespace geometric_locus_points_l51_51368

theorem geometric_locus_points :
  (∀ x y : ℝ, (y^2 = x^2) ↔ (y = x ∨ y = -x)) ∧
  (∀ x : ℝ, (x^2 - 2 * x + 1 = 0) ↔ (x = 1)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = 4 * (y - 1)) ↔ (x = 0 ∧ y = 2)) ∧
  (∀ x y : ℝ, (x^2 - 2 * x * y + y^2 = -1) ↔ false) :=
by
  sorry

end geometric_locus_points_l51_51368


namespace triangle_min_value_l51_51634

open Real

theorem triangle_min_value
  (A B C : ℝ)
  (h_triangle: A + B + C = π)
  (h_sin: sin (2 * A + B) = 2 * sin B) :
  tan A + tan C + 2 / tan B ≥ 2 :=
sorry

end triangle_min_value_l51_51634


namespace number_of_candidates_is_9_l51_51961

-- Defining the problem
def num_ways_to_select_president_and_vp (n : ℕ) : ℕ :=
  n * (n - 1)

-- Main theorem statement
theorem number_of_candidates_is_9 (n : ℕ) (h : num_ways_to_select_president_and_vp n = 72) : n = 9 :=
by
  sorry

end number_of_candidates_is_9_l51_51961


namespace min_shirts_to_save_money_l51_51862

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 60 + 11 * x < 20 + 15 * x ∧ (∀ y : ℕ, 60 + 11 * y < 20 + 15 * y → y ≥ x) ∧ x = 11 :=
by
  sorry

end min_shirts_to_save_money_l51_51862


namespace balloons_in_package_initially_l51_51049

-- Definition of conditions
def friends : ℕ := 5
def balloons_given_back : ℕ := 11
def balloons_after_giving_back : ℕ := 39

-- Calculation for original balloons each friend had
def original_balloons_each_friend := balloons_after_giving_back + balloons_given_back

-- Theorem: Number of balloons in the package initially
theorem balloons_in_package_initially : 
  (original_balloons_each_friend * friends) = 250 :=
by
  sorry

end balloons_in_package_initially_l51_51049


namespace unique_two_digit_integer_solution_l51_51063

variable {s : ℕ}

-- Conditions
def is_two_digit_positive_integer (s : ℕ) : Prop :=
  10 ≤ s ∧ s < 100

def last_two_digits_of_13s_are_52 (s : ℕ) : Prop :=
  13 * s % 100 = 52

-- Theorem statement
theorem unique_two_digit_integer_solution (h1 : is_two_digit_positive_integer s)
                                          (h2 : last_two_digits_of_13s_are_52 s) :
  s = 4 :=
sorry

end unique_two_digit_integer_solution_l51_51063


namespace count_final_numbers_l51_51766

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l51_51766


namespace smaller_octagon_half_area_l51_51889

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end smaller_octagon_half_area_l51_51889


namespace Ann_end_blocks_l51_51483

-- Define blocks Ann initially has and finds
def initialBlocksAnn : ℕ := 9
def foundBlocksAnn : ℕ := 44

-- Define blocks Ann ends with
def finalBlocksAnn : ℕ := initialBlocksAnn + foundBlocksAnn

-- The proof goal
theorem Ann_end_blocks : finalBlocksAnn = 53 := by
  -- Use sorry to skip the proof
  sorry

end Ann_end_blocks_l51_51483


namespace range_of_f_lt_0_l51_51416

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

variable (f : ℝ → ℝ)
variable (h_odd : is_odd f)
variable (h_decreasing : decreasing_on f (Set.Iic 0))
variable (h_at_2 : f 2 = 0)

theorem range_of_f_lt_0 : ∀ x, x ∈ (Set.Ioo (-2) 0 ∪ Set.Ioi 2) → f x < 0 := by
  sorry

end range_of_f_lt_0_l51_51416


namespace smallest_angle_equilateral_triangle_l51_51833

-- Definitions corresponding to the conditions
structure EquilateralTriangle :=
(vertices : Fin 3 → ℝ × ℝ)
(equilateral : ∀ i j, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))

def point_on_line_segment (p1 p2 : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
((1 - t) * p1.1 + t * p2.1, (1 - t) * p1.2 + t * p2.2)

-- Given an equilateral triangle ABC with vertices A, B, C,
-- and points D on AB, E on AC, D1 on BC, and E1 on BC,
-- such that AB = DB + BD_1 and AC = CE + CE_1,
-- prove the smallest angle between DE_1 and ED_1 is 60 degrees.

theorem smallest_angle_equilateral_triangle
  (ABC : EquilateralTriangle)
  (A B C D E D₁ E₁ : ℝ × ℝ)
  (on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = point_on_line_segment A B t)
  (on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = point_on_line_segment A C t)
  (on_BC : ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ D₁ = point_on_line_segment B C t₁ ∧
                         0 ≤ t₂ ∧ t₂ ≤ 1 ∧ E₁ = point_on_line_segment B C t₂)
  (AB_property : dist A B = dist D B + dist B D₁)
  (AC_property : dist A C = dist E C + dist C E₁) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 60 ∧ θ = 60 :=
sorry

end smallest_angle_equilateral_triangle_l51_51833


namespace unattainable_value_of_y_l51_51930

theorem unattainable_value_of_y (x : ℚ) (h : x ≠ -5/4) :
  ¬ ∃ y : ℚ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3/4 :=
by
  sorry

end unattainable_value_of_y_l51_51930


namespace tony_schooling_years_l51_51243

theorem tony_schooling_years:
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  first_degree + additional_degrees + graduate_degree = 14 :=
by {
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  show first_degree + additional_degrees + graduate_degree = 14
  sorry
}

end tony_schooling_years_l51_51243


namespace part_I_part_II_l51_51676

def f (x : ℝ) : ℝ := abs (2 * x - 7) + 1

def g (x : ℝ) : ℝ := abs (2 * x - 7) - 2 * abs (x - 1) + 1

theorem part_I :
  {x : ℝ | f x ≤ x} = {x : ℝ | (8 / 3) ≤ x ∧ x ≤ 6} := sorry

theorem part_II (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 := sorry

end part_I_part_II_l51_51676


namespace term_containing_x3_l51_51330

-- Define the problem statement in Lean 4
theorem term_containing_x3 (a : ℝ) (x : ℝ) (hx : x ≠ 0) 
(h_sum_coeff : (2 + a) ^ 5 = 0) :
  (2 * x + a / x) ^ 5 = -160 * x ^ 3 :=
sorry

end term_containing_x3_l51_51330


namespace irrational_sqrt3_l51_51837

def is_irrational (x : ℝ) : Prop := ∀ (a b : ℤ), b ≠ 0 → x ≠ a / b

theorem irrational_sqrt3 :
  let A := 22 / 7
  let B := 0
  let C := Real.sqrt 3
  let D := 3.14
  is_irrational C :=
by
  sorry

end irrational_sqrt3_l51_51837


namespace D_300_l51_51137

def D (n : ℕ) : ℕ :=
sorry

theorem D_300 : D 300 = 73 := 
by 
sorry

end D_300_l51_51137


namespace tangent_line_parallel_to_x_axis_l51_51406

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def f_derivative (x : ℝ) : ℝ := (1 - Real.log x) / (x^2)

theorem tangent_line_parallel_to_x_axis :
  ∀ x₀ : ℝ, 
  f_derivative x₀ = 0 → 
  f x₀ = 1 / Real.exp 1 :=
by
  intro x₀ h_deriv_zero
  sorry

end tangent_line_parallel_to_x_axis_l51_51406


namespace store_profit_l51_51991

variable (C : ℝ)  -- Cost price of a turtleneck sweater

noncomputable def initial_marked_price : ℝ := 1.20 * C
noncomputable def new_year_marked_price : ℝ := 1.25 * initial_marked_price C
noncomputable def discount_amount : ℝ := 0.08 * new_year_marked_price C
noncomputable def final_selling_price : ℝ := new_year_marked_price C - discount_amount C
noncomputable def profit : ℝ := final_selling_price C - C

theorem store_profit (C : ℝ) : profit C = 0.38 * C :=
by
  -- The detailed steps are omitted, as required by the instructions.
  sorry

end store_profit_l51_51991


namespace total_amount_shared_l51_51423

theorem total_amount_shared (A B C : ℕ) (h1 : A = 24) (h2 : 2 * A = 3 * B) (h3 : 8 * A = 4 * C) :
  A + B + C = 156 :=
sorry

end total_amount_shared_l51_51423


namespace parallel_lines_m_l51_51937

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, x + 2 * m * y - 1 = 0 → (3 * m - 1) * x - m * y - 1 = 0)
  → m = 0 ∨ m = 1 / 6 := 
sorry

end parallel_lines_m_l51_51937


namespace total_marks_l51_51304

variable (A M SS Mu : ℝ)

-- Conditions
def cond1 : Prop := M = A - 20
def cond2 : Prop := SS = Mu + 10
def cond3 : Prop := Mu = 70
def cond4 : Prop := M = (9 / 10) * A

-- Theorem statement
theorem total_marks (A M SS Mu : ℝ) (h1 : cond1 A M)
                                      (h2 : cond2 SS Mu)
                                      (h3 : cond3 Mu)
                                      (h4 : cond4 A M) :
    A + M + SS + Mu = 530 :=
by 
  sorry

end total_marks_l51_51304


namespace Jen_distance_from_start_l51_51340

-- Define the rate of Jen's walking (in miles per hour)
def walking_rate : ℝ := 4

-- Define the time Jen walks forward (in hours)
def forward_time : ℝ := 2

-- Define the time Jen walks back (in hours)
def back_time : ℝ := 1

-- Define the distance walked forward
def distance_forward : ℝ := walking_rate * forward_time

-- Define the distance walked back
def distance_back : ℝ := walking_rate * back_time

-- Define the net distance from the starting point
def net_distance : ℝ := distance_forward - distance_back

-- Theorem stating the net distance from the starting point is 4.0 miles
theorem Jen_distance_from_start : net_distance = 4.0 := by
  sorry

end Jen_distance_from_start_l51_51340


namespace nat_le_two_pow_million_l51_51908

theorem nat_le_two_pow_million (n : ℕ) (h : n ≤ 2^1000000) : 
  ∃ (x : ℕ → ℕ) (k : ℕ), k ≤ 1100000 ∧ x 0 = 1 ∧ x k = n ∧ 
  ∀ (i : ℕ), 1 ≤ i → i ≤ k → ∃ (r s : ℕ), 0 ≤ r ∧ r ≤ s ∧ s < i ∧ x i = x r + x s :=
sorry

end nat_le_two_pow_million_l51_51908


namespace beads_cost_is_three_l51_51651

-- Define the given conditions
def cost_of_string_per_bracelet : Nat := 1
def selling_price_per_bracelet : Nat := 6
def number_of_bracelets_sold : Nat := 25
def total_profit : Nat := 50

-- The amount spent on beads per bracelet
def amount_spent_on_beads_per_bracelet (B : Nat) : Prop :=
  B = (total_profit + number_of_bracelets_sold * (cost_of_string_per_bracelet + B) - number_of_bracelets_sold * selling_price_per_bracelet) / number_of_bracelets_sold 

-- The main goal is to prove that the amount spent on beads is 3
theorem beads_cost_is_three : amount_spent_on_beads_per_bracelet 3 :=
by sorry

end beads_cost_is_three_l51_51651


namespace production_days_l51_51946

theorem production_days (n : ℕ) 
    (h1 : 70 * n + 90 = 75 * (n + 1)) : n = 3 := 
sorry

end production_days_l51_51946


namespace porch_width_l51_51155

theorem porch_width (L_house W_house total_area porch_length W : ℝ)
  (h1 : L_house = 20.5) (h2 : W_house = 10) (h3 : total_area = 232) (h4 : porch_length = 6) (h5 : total_area = (L_house * W_house) + (porch_length * W)) :
  W = 4.5 :=
by 
  sorry

end porch_width_l51_51155


namespace distance_between_cityA_and_cityB_l51_51995

noncomputable def distanceBetweenCities (time_to_cityB time_from_cityB saved_time round_trip_speed: ℝ) : ℝ :=
  let total_distance := 90 * (time_to_cityB + saved_time + time_from_cityB + saved_time) / 2
  total_distance / 2

theorem distance_between_cityA_and_cityB 
  (time_to_cityB : ℝ)
  (time_from_cityB : ℝ)
  (saved_time : ℝ)
  (round_trip_speed : ℝ)
  (distance : ℝ)
  (h1 : time_to_cityB = 6)
  (h2 : time_from_cityB = 4.5)
  (h3 : saved_time = 0.5)
  (h4 : round_trip_speed = 90)
  (h5 : distanceBetweenCities time_to_cityB time_from_cityB saved_time round_trip_speed = distance)
: distance = 427.5 := by
  sorry

end distance_between_cityA_and_cityB_l51_51995


namespace slower_time_l51_51974

-- Definitions for the problem conditions
def num_stories : ℕ := 50
def lola_time_per_story : ℕ := 12
def tara_time_per_story : ℕ := 10
def tara_stop_time : ℕ := 4
def tara_num_stops : ℕ := num_stories - 2 -- Stops on each floor except the first and last

-- Calculations based on the conditions
def lola_total_time : ℕ := num_stories * lola_time_per_story
def tara_total_time : ℕ := num_stories * tara_time_per_story + tara_num_stops * tara_stop_time

-- Target statement to be proven
theorem slower_time : tara_total_time = 692 := by
  sorry  -- Proof goes here (excluded as per instructions)

end slower_time_l51_51974


namespace number_of_sides_of_polygon_24_deg_exterior_angle_l51_51749

theorem number_of_sides_of_polygon_24_deg_exterior_angle :
  (∀ (n : ℕ), (∀ (k : ℕ), k = 360 / 24 → n = k)) :=
by
  sorry

end number_of_sides_of_polygon_24_deg_exterior_angle_l51_51749


namespace least_alpha_prime_l51_51306

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_distinct_prime (α β : ℕ) : Prop :=
  α ≠ β ∧ is_prime α ∧ is_prime β

theorem least_alpha_prime (α : ℕ) :
  is_distinct_prime α (180 - 2 * α) → α ≥ 41 :=
sorry

end least_alpha_prime_l51_51306


namespace proof_of_expression_value_l51_51127

theorem proof_of_expression_value (m n : ℝ) 
  (h1 : m^2 - 2019 * m = 1) 
  (h2 : n^2 - 2019 * n = 1) : 
  (m^2 - 2019 * m + 3) * (n^2 - 2019 * n + 4) = 20 := 
by 
  sorry

end proof_of_expression_value_l51_51127


namespace angle_ADE_l51_51302

-- Definitions and conditions
variable (x : ℝ)

def angle_ABC := 60
def angle_CAD := x
def angle_BAD := x
def angle_BCA := 120 - 2 * x
def angle_DCE := 180 - (120 - 2 * x)

-- Theorem statement
theorem angle_ADE (x : ℝ) : angle_CAD x = x → angle_BAD x = x → angle_ABC = 60 → 
                            angle_DCE x = 180 - angle_BCA x → 
                            120 - 3 * x = 120 - 3 * x := 
by
  intro h1 h2 h3 h4
  sorry

end angle_ADE_l51_51302


namespace yellow_balls_in_bag_l51_51182

theorem yellow_balls_in_bag (y : ℕ) (r : ℕ) (P_red : ℚ) (h_r : r = 8) (h_P_red : P_red = 1 / 3) 
  (h_prob : P_red = r / (r + y)) : y = 16 :=
by
  sorry

end yellow_balls_in_bag_l51_51182


namespace percentage_born_in_july_l51_51180

def total_scientists : ℕ := 150
def scientists_born_in_july : ℕ := 15

theorem percentage_born_in_july : (scientists_born_in_july * 100 / total_scientists) = 10 := by
  sorry

end percentage_born_in_july_l51_51180


namespace triangle_final_position_after_rotation_l51_51275

-- Definitions for the initial conditions
def square_rolls_clockwise_around_octagon : Prop := 
  true -- placeholder definition, assume this defines the motion correctly

def triangle_initial_position : ℕ := 0 -- representing bottom as 0

-- Defining the proof problem
theorem triangle_final_position_after_rotation :
  square_rolls_clockwise_around_octagon →
  triangle_initial_position = 0 →
  triangle_initial_position = 0 :=
by
  intros
  sorry

end triangle_final_position_after_rotation_l51_51275


namespace intersection_A_B_union_A_B_subset_C_B_l51_51983

open Set

noncomputable def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 9}
noncomputable def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 6} :=
by
  sorry

theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 9} :=
by
  sorry

theorem subset_C_B (a : ℝ) : C a ⊆ B → 2 ≤ a ∧ a ≤ 8 :=
by
  sorry

end intersection_A_B_union_A_B_subset_C_B_l51_51983


namespace students_in_both_clubs_l51_51929

variables (Total Students RoboticClub ScienceClub EitherClub BothClubs : ℕ)

theorem students_in_both_clubs
  (h1 : Total = 300)
  (h2 : RoboticClub = 80)
  (h3 : ScienceClub = 130)
  (h4 : EitherClub = 190)
  (h5 : EitherClub = RoboticClub + ScienceClub - BothClubs) :
  BothClubs = 20 :=
by
  sorry

end students_in_both_clubs_l51_51929


namespace least_xy_value_l51_51228

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
sorry

end least_xy_value_l51_51228


namespace eq_inf_solutions_l51_51728

theorem eq_inf_solutions (a b : ℝ) : 
    (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + b)) ↔ b = -(4 / 3) * a := by
  sorry

end eq_inf_solutions_l51_51728


namespace probability_of_exactly_one_instrument_l51_51582

-- Definitions
def total_people : ℕ := 800
def fraction_play_at_least_one_instrument : ℚ := 2 / 5
def num_play_two_or_more_instruments : ℕ := 96

-- Calculation
def num_play_at_least_one_instrument := fraction_play_at_least_one_instrument * total_people
def num_play_exactly_one_instrument := num_play_at_least_one_instrument - num_play_two_or_more_instruments

-- Probability calculation
def probability_play_exactly_one_instrument := num_play_exactly_one_instrument / total_people

-- Proof statement
theorem probability_of_exactly_one_instrument :
  probability_play_exactly_one_instrument = 0.28 := by
  sorry

end probability_of_exactly_one_instrument_l51_51582


namespace xyz_value_l51_51863

theorem xyz_value
  (x y z : ℝ)
  (h1 : (x + y + z) * (xy + xz + yz) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3)
  : xyz = 5 :=
by
  sorry

end xyz_value_l51_51863


namespace rhombus_area_l51_51212

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  0.5 * d1 * d2

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 3) (h2 : d2 = 4) : area_of_rhombus d1 d2 = 6 :=
by
  sorry

end rhombus_area_l51_51212


namespace gross_pay_is_450_l51_51918

def net_pay : ℤ := 315
def taxes : ℤ := 135
def gross_pay : ℤ := net_pay + taxes

theorem gross_pay_is_450 : gross_pay = 450 := by
  sorry

end gross_pay_is_450_l51_51918


namespace common_difference_is_3_l51_51486

variable {a : ℕ → ℤ} {d : ℤ}

-- Definitions of conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition_1 (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 11 = 24

def condition_2 (a : ℕ → ℤ) : Prop :=
  a 4 = 3

-- Theorem statement to prove
theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ)
  (ha : is_arithmetic_sequence a d)
  (hc1 : condition_1 a d)
  (hc2 : condition_2 a) :
  d = 3 := by
  sorry

end common_difference_is_3_l51_51486


namespace mole_fractions_C4H8O2_l51_51714

/-- 
Given:
- The molecular formula of C4H8O2,
- 4 moles of carbon (C) atoms,
- 8 moles of hydrogen (H) atoms,
- 2 moles of oxygen (O) atoms.

Prove that:
The mole fractions of each element in C4H8O2 are:
- Carbon (C): 2/7
- Hydrogen (H): 4/7
- Oxygen (O): 1/7
--/
theorem mole_fractions_C4H8O2 :
  let m_C := 4
  let m_H := 8
  let m_O := 2
  let total_moles := m_C + m_H + m_O
  let mole_fraction_C := m_C / total_moles
  let mole_fraction_H := m_H / total_moles
  let mole_fraction_O := m_O / total_moles
  mole_fraction_C = 2 / 7 ∧ mole_fraction_H = 4 / 7 ∧ mole_fraction_O = 1 / 7 := by
  sorry

end mole_fractions_C4H8O2_l51_51714


namespace n_n_plus_one_div_by_2_l51_51085

theorem n_n_plus_one_div_by_2 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 99) : 2 ∣ n * (n + 1) :=
by
  sorry

end n_n_plus_one_div_by_2_l51_51085


namespace area_difference_l51_51283

-- Define the areas of individual components
def area_of_square : ℕ := 1
def area_of_small_triangle : ℚ := (1 / 2) * area_of_square
def area_of_large_triangle : ℚ := (1 / 2) * (1 * 2 * area_of_square)

-- Define the total area of the first figure
def first_figure_area : ℚ := 
    8 * area_of_square +
    6 * area_of_small_triangle +
    2 * area_of_large_triangle

-- Define the total area of the second figure
def second_figure_area : ℚ := 
    4 * area_of_square +
    6 * area_of_small_triangle +
    8 * area_of_large_triangle

-- Define the statement to prove the difference in areas
theorem area_difference : second_figure_area - first_figure_area = 2 := by
    -- sorry is used to indicate that the proof is omitted
    sorry

end area_difference_l51_51283


namespace rings_sold_l51_51556

theorem rings_sold (R : ℕ) : 
  ∀ (num_necklaces total_sales necklace_price ring_price : ℕ),
  num_necklaces = 4 →
  total_sales = 80 →
  necklace_price = 12 →
  ring_price = 4 →
  num_necklaces * necklace_price + R * ring_price = total_sales →
  R = 8 := 
by 
  intros num_necklaces total_sales necklace_price ring_price h1 h2 h3 h4 h5
  sorry

end rings_sold_l51_51556


namespace collinear_vectors_l51_51570

open Vector

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def not_collinear (a b : V) : Prop :=
¬(∃ k : ℝ, k ≠ 0 ∧ a = k • b)

theorem collinear_vectors
  {a b m n : V}
  (h1 : m = a + b)
  (h2 : n = 2 • a + 2 • b)
  (h3 : not_collinear a b) :
  ∃ k : ℝ, k ≠ 0 ∧ n = k • m :=
by
  sorry

end collinear_vectors_l51_51570


namespace perpendicular_condition_l51_51631

def vector_a : ℝ × ℝ := (4, 3)
def vector_b : ℝ × ℝ := (-1, 2)

def add_vector_scaled (a b : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  (a.1 + k * b.1, a.2 + k * b.2)

def sub_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perpendicular_condition (k : ℝ) :
  dot_product (add_vector_scaled vector_a vector_b k) (sub_vector vector_a vector_b) = 0 ↔ k = 23 / 3 :=
by
  sorry

end perpendicular_condition_l51_51631


namespace evaluate_root_power_l51_51174

theorem evaluate_root_power : (Real.sqrt (Real.sqrt 9))^12 = 729 := 
by sorry

end evaluate_root_power_l51_51174


namespace find_x_l51_51711

variable (x : ℕ)

def f (x : ℕ) : ℕ := 2 * x + 5
def g (y : ℕ) : ℕ := 3 * y

theorem find_x (h : g (f x) = 123) : x = 18 :=
by {
  sorry
}

end find_x_l51_51711


namespace percentage_of_profit_without_discount_l51_51987

-- Definitions for the conditions
def cost_price : ℝ := 100
def discount_rate : ℝ := 0.04
def profit_rate : ℝ := 0.32

-- The statement to prove
theorem percentage_of_profit_without_discount :
  let selling_price := cost_price + (profit_rate * cost_price)
  (selling_price - cost_price) / cost_price * 100 = 32 := by
  let selling_price := cost_price + (profit_rate * cost_price)
  sorry

end percentage_of_profit_without_discount_l51_51987


namespace TruckCapacities_RentalPlanExists_MinimumRentalCost_l51_51076

-- Problem 1
theorem TruckCapacities (x y : ℕ) (h1: 2 * x + y = 10) (h2: x + 2 * y = 11) :
  x = 3 ∧ y = 4 :=
by
  sorry

-- Problem 2
theorem RentalPlanExists (a b : ℕ) (h: 3 * a + 4 * b = 31) :
  (a = 9 ∧ b = 1) ∨ (a = 5 ∧ b = 4) ∨ (a = 1 ∧ b = 7) :=
by
  sorry

-- Problem 3
theorem MinimumRentalCost (a b : ℕ) (h1: 3 * a + 4 * b = 31) 
  (h2: 100 * a + 120 * b = 940) :
  ∃ a b, a = 1 ∧ b = 7 :=
by
  sorry

end TruckCapacities_RentalPlanExists_MinimumRentalCost_l51_51076


namespace find_r_squared_l51_51729

noncomputable def parabola_intersect_circle_radius_squared : Prop :=
  ∀ (x y : ℝ), y = (x - 1)^2 ∧ x - 3 = (y + 2)^2 → (x - 3/2)^2 + (y + 3/2)^2 = 1/2

theorem find_r_squared : parabola_intersect_circle_radius_squared :=
sorry

end find_r_squared_l51_51729


namespace seven_by_seven_grid_partition_l51_51933

theorem seven_by_seven_grid_partition : 
  ∀ (x y : ℕ), 4 * x + 3 * y = 49 ∧ x + y ≥ 16 → x = 1 :=
by sorry

end seven_by_seven_grid_partition_l51_51933


namespace total_distance_of_journey_l51_51708

variables (x v : ℝ)
variable (d : ℝ := 600)  -- d is the total distance given by the solution to be 600 miles

-- Define the conditions stated in the problem
def condition_1 := (x = 10 * v)  -- x = 10 * v (from first part of the solution)
def condition_2 := (3 * v * d - 90 * v = -28.5 * 3 * v)  -- 2nd condition translated from second part

theorem total_distance_of_journey : 
  ∀ (x v : ℝ), condition_1 x v ∧ condition_2 x v -> x = d :=
sorry

end total_distance_of_journey_l51_51708


namespace smoke_diagram_total_height_l51_51494

theorem smoke_diagram_total_height : 
  ∀ (h1 h2 h3 h4 h5 : ℕ),
    h1 < h2 ∧ h2 < h3 ∧ h3 < h4 ∧ h4 < h5 ∧ 
    (h2 - h1 = 2) ∧ (h3 - h2 = 2) ∧ (h4 - h3 = 2) ∧ (h5 - h4 = 2) ∧ 
    (h5 = h1 + h2) → 
    h1 + h2 + h3 + h4 + h5 = 50 := 
by 
  sorry

end smoke_diagram_total_height_l51_51494


namespace correct_grid_l51_51165

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l51_51165


namespace exists_fraction_expression_l51_51976

theorem exists_fraction_expression (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) :
  ∃ (m : ℕ) (h₀ : 3 ≤ m) (h₁ : m ≤ p - 2) (x y : ℕ), (m : ℚ) / (p^2 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ) :=
sorry

end exists_fraction_expression_l51_51976


namespace find_acid_percentage_l51_51600

theorem find_acid_percentage (P : ℕ) (x : ℕ) (h1 : 4 + x = 20) 
  (h2 : x = 20 - 4) 
  (h3 : (P : ℝ)/100 * 4 + 0.75 * 16 = 0.72 * 20) : P = 60 :=
by
  sorry

end find_acid_percentage_l51_51600


namespace factor_expression_l51_51548

theorem factor_expression (x : ℝ) : 60 * x + 45 = 15 * (4 * x + 3) :=
by
  sorry

end factor_expression_l51_51548


namespace ratio_of_length_to_width_of_field_is_two_to_one_l51_51975

-- Definitions based on conditions
def lengthOfField : ℕ := 80
def widthOfField (field_area pond_area : ℕ) : ℕ := field_area / lengthOfField
def pondSideLength : ℕ := 8
def pondArea : ℕ := pondSideLength * pondSideLength
def fieldArea : ℕ := pondArea * 50
def lengthMultipleOfWidth (length width : ℕ) := ∃ k : ℕ, length = k * width

-- Main statement to prove the ratio of length to width is 2:1
theorem ratio_of_length_to_width_of_field_is_two_to_one :
  lengthMultipleOfWidth lengthOfField (widthOfField fieldArea pondArea) →
  lengthOfField = 2 * (widthOfField fieldArea pondArea) :=
by
  -- Conditions
  have h1 : pondSideLength = 8 := rfl
  have h2 : pondArea = pondSideLength * pondSideLength := rfl
  have h3 : fieldArea = pondArea * 50 := rfl
  have h4 : lengthOfField = 80 := rfl
  sorry

end ratio_of_length_to_width_of_field_is_two_to_one_l51_51975


namespace radius_of_cone_base_l51_51765

theorem radius_of_cone_base {R : ℝ} {theta : ℝ} (hR : R = 6) (htheta : theta = 120) :
  ∃ r : ℝ, r = 2 :=
by
  sorry

end radius_of_cone_base_l51_51765


namespace average_interest_rate_l51_51480

theorem average_interest_rate (total_investment : ℝ) (rate1 rate2 : ℝ) (annual_return1 annual_return2 : ℝ) 
  (h1 : total_investment = 6000) 
  (h2 : rate1 = 0.035) 
  (h3 : rate2 = 0.055) 
  (h4 : annual_return1 = annual_return2) :
  (annual_return1 + annual_return2) / total_investment * 100 = 4.3 :=
by
  sorry

end average_interest_rate_l51_51480


namespace math_problem_l51_51657

theorem math_problem
  (z : ℝ)
  (hz : z = 80)
  (y : ℝ)
  (hy : y = (1/4) * z)
  (x : ℝ)
  (hx : x = (1/3) * y)
  (w : ℝ)
  (hw : w = x + y + z) :
  x = 20 / 3 ∧ w = 320 / 3 :=
by
  sorry

end math_problem_l51_51657


namespace nell_baseball_cards_l51_51535

theorem nell_baseball_cards 
  (ace_cards_now : ℕ) 
  (extra_baseball_cards : ℕ) 
  (B : ℕ) : 
  ace_cards_now = 55 →
  extra_baseball_cards = 123 →
  B = ace_cards_now + extra_baseball_cards →
  B = 178 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end nell_baseball_cards_l51_51535


namespace work_rate_ab_l51_51769

variables (A B C : ℝ)

-- Defining the work rates as per the conditions
def work_rate_bc := 1 / 6 -- (b and c together in 6 days)
def work_rate_ca := 1 / 3 -- (c and a together in 3 days)
def work_rate_c := 1 / 8 -- (c alone in 8 days)

-- The main theorem that proves a and b together can complete the work in 4 days,
-- based on the above conditions.
theorem work_rate_ab : 
  (B + C = work_rate_bc) ∧ (C + A = work_rate_ca) ∧ (C = work_rate_c) 
  → (A + B = 1 / 4) :=
by sorry

end work_rate_ab_l51_51769


namespace larger_number_value_l51_51960

theorem larger_number_value (L S : ℕ) (h1 : L - S = 20775) (h2 : L = 23 * S + 143) : L = 21713 :=
sorry

end larger_number_value_l51_51960


namespace solve_problem_l51_51206

noncomputable def problem_statement : ℤ :=
  (-3)^6 / 3^4 - 4^3 * 2^2 + 9^2

theorem solve_problem : problem_statement = -166 :=
by 
  -- Proof omitted
  sorry

end solve_problem_l51_51206


namespace jeffery_fish_count_l51_51700

variable (J R Y : ℕ)

theorem jeffery_fish_count :
  (R = 3 * J) → (Y = 2 * R) → (J + R + Y = 100) → (Y = 60) :=
by
  intros hR hY hTotal
  have h1 : R = 3 * J := hR
  have h2 : Y = 2 * R := hY
  rw [h1, h2] at hTotal
  sorry

end jeffery_fish_count_l51_51700


namespace problem_find_f_l51_51460

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_find_f {k : ℝ} :
  (∀ x : ℝ, x * (f (x + 1) - f x) = f x) →
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|) →
  (∀ x : ℝ, 0 < x → f x = k * x) :=
by
  intro h1 h2
  apply sorry

end problem_find_f_l51_51460


namespace rain_on_first_day_l51_51903

theorem rain_on_first_day (x : ℝ) (h1 : x >= 0)
  (h2 : (2 * x) + 50 / 100 * (2 * x) = 3 * x) 
  (h3 : 6 * 12 = 72)
  (h4 : 3 * 3 = 9)
  (h5 : x + 2 * x + 3 * x = 6 * x)
  (h6 : 6 * x + 21 - 9 = 72) : x = 10 :=
by 
  -- Proof would go here, but we skip it according to instructions
  sorry

end rain_on_first_day_l51_51903


namespace curves_tangent_at_m_eq_two_l51_51386

-- Definitions of the ellipsoid and hyperbola equations.
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 = 2
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

-- The proposition to be proved.
theorem curves_tangent_at_m_eq_two :
  ∃ m : ℝ, (∀ x y : ℝ, ellipse x y ∧ hyperbola x y m → m = 2) :=
sorry

end curves_tangent_at_m_eq_two_l51_51386


namespace gcd_greatest_possible_value_l51_51524

noncomputable def Sn (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem gcd_greatest_possible_value (n : ℕ) (hn : 0 < n) : 
  Nat.gcd (3 * Sn n) (n + 1) = 1 :=
sorry

end gcd_greatest_possible_value_l51_51524


namespace B_subset_A_implies_m_values_l51_51252

noncomputable def A : Set ℝ := { x | x^2 + x - 6 = 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }
def possible_m_values : Set ℝ := {1/3, -1/2}

theorem B_subset_A_implies_m_values (m : ℝ) : B m ⊆ A → m ∈ possible_m_values := by
  sorry

end B_subset_A_implies_m_values_l51_51252


namespace rhombus_diagonal_length_l51_51969

theorem rhombus_diagonal_length (d1 : ℝ) : 
  (d1 * 12) / 2 = 60 → d1 = 10 := 
by 
  sorry

end rhombus_diagonal_length_l51_51969


namespace point_B_coordinates_sum_l51_51874

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end point_B_coordinates_sum_l51_51874


namespace sum_of_two_digit_numbers_with_gcd_lcm_l51_51529

theorem sum_of_two_digit_numbers_with_gcd_lcm (x y : ℕ) (h1 : Nat.gcd x y = 8) (h2 : Nat.lcm x y = 96)
  (h3 : 10 ≤ x ∧ x < 100) (h4 : 10 ≤ y ∧ y < 100) : x + y = 56 :=
sorry

end sum_of_two_digit_numbers_with_gcd_lcm_l51_51529


namespace find_a_10_l51_51373

-- Definitions and conditions from the problem
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

variable (a : ℕ → ℕ)

-- Conditions given
axiom a_3 : a 3 = 3
axiom S_3 : S a 3 = 6
axiom arithmetic_seq : is_arithmetic_sequence a

-- Proof problem statement
theorem find_a_10 : a 10 = 10 := 
sorry

end find_a_10_l51_51373


namespace number_of_shampoos_l51_51383

-- Define necessary variables in conditions
def h := 10 -- time spent hosing in minutes
def t := 55 -- total time spent cleaning in minutes
def p := 15 -- time per shampoo in minutes

-- State the theorem
theorem number_of_shampoos (h t p : Nat) (h_val : h = 10) (t_val : t = 55) (p_val : p = 15) :
    (t - h) / p = 3 := by
  -- Proof to be filled in
  sorry

end number_of_shampoos_l51_51383


namespace arithmetic_sequence_sum_l51_51119

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℤ) (S_n : ℕ → ℤ),
  (∀ n : ℕ, S_n n = (n * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) / 2) →
  S_n 17 = 170 →
  a_n 7 + a_n 8 + a_n 12 = 30 := 
by
  sorry

end arithmetic_sequence_sum_l51_51119


namespace graph_symmetry_l51_51446

/-- Theorem:
The functions y = 2^x and y = 2^{-x} are symmetric about the y-axis.
-/
theorem graph_symmetry :
  ∀ (x : ℝ), (∃ (y : ℝ), y = 2^x) →
  (∃ (y' : ℝ), y' = 2^(-x)) →
  (∀ (y : ℝ), ∃ (x : ℝ), (y = 2^x ↔ y = 2^(-x)) → y = 2^x → y = 2^(-x)) :=
by
  intro x
  intro h1
  intro h2
  intro y
  exists x
  intro h3
  intro hy
  sorry

end graph_symmetry_l51_51446


namespace fixed_point_coordinates_l51_51055

theorem fixed_point_coordinates (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : (2, 2) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^(x-2) + 1)} := 
by
  -- Proof goes here
  sorry

end fixed_point_coordinates_l51_51055


namespace correct_propositions_l51_51005

-- Define propositions
def proposition1 : Prop :=
  ∀ x, 2 * (Real.cos (1/3 * x + Real.pi / 4))^2 - 1 = -Real.sin (2 * x / 3)

def proposition2 : Prop :=
  ∃ α : ℝ, Real.sin α + Real.cos α = 3 / 2

def proposition3 : Prop :=
  ∀ α β : ℝ, (0 < α ∧ α < Real.pi / 2) → (0 < β ∧ β < Real.pi / 2) → α < β → Real.tan α < Real.tan β

def proposition4 : Prop :=
  ∀ x, x = Real.pi / 8 → Real.sin (2 * x + 5 * Real.pi / 4) = -1

def proposition5 : Prop :=
  Real.sin ( 2 * (Real.pi / 12) + Real.pi / 3 ) = 0

-- Define the main theorem combining correct propositions
theorem correct_propositions : 
  proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ proposition4 ∧ ¬proposition5 :=
  by
  -- Since we only need to state the theorem, we use sorry.
  sorry

end correct_propositions_l51_51005


namespace james_total_earnings_l51_51834

-- Define the earnings for January
def januaryEarnings : ℕ := 4000

-- Define the earnings for February based on January
def februaryEarnings : ℕ := 2 * januaryEarnings

-- Define the earnings for March based on February
def marchEarnings : ℕ := februaryEarnings - 2000

-- Define the total earnings including January, February, and March
def totalEarnings : ℕ := januaryEarnings + februaryEarnings + marchEarnings

-- State the theorem: total earnings should be 18000
theorem james_total_earnings : totalEarnings = 18000 := by
  sorry

end james_total_earnings_l51_51834


namespace solve_for_x_l51_51299

theorem solve_for_x (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) : 
  (x + 5) / (x - 3) = (x - 4) / (x + 2) → x = 1 / 7 :=
by
  sorry

end solve_for_x_l51_51299


namespace chebyshev_substitution_even_chebyshev_substitution_odd_l51_51965

def T (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the first kind
def U (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the second kind

theorem chebyshev_substitution_even (k : ℕ) (α : ℝ) :
  T (2 * k) (Real.sin α) = (-1)^k * Real.cos ((2 * k) * α) ∧
  U ((2 * k) - 1) (Real.sin α) = (-1)^(k + 1) * (Real.sin ((2 * k) * α) / Real.cos α) :=
by
  sorry

theorem chebyshev_substitution_odd (k : ℕ) (α : ℝ) :
  T (2 * k + 1) (Real.sin α) = (-1)^k * Real.sin ((2 * k + 1) * α) ∧
  U (2 * k) (Real.sin α) = (-1)^k * (Real.cos ((2 * k + 1) * α) / Real.cos α) :=
by
  sorry

end chebyshev_substitution_even_chebyshev_substitution_odd_l51_51965


namespace num_terms_arithmetic_sequence_is_41_l51_51035

-- Definitions and conditions
def first_term : ℤ := 200
def common_difference : ℤ := -5
def last_term : ℤ := 0

-- Definition of the n-th term of arithmetic sequence
def nth_term (a : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a + (n - 1) * d

-- Statement to prove
theorem num_terms_arithmetic_sequence_is_41 : 
  ∃ n : ℕ, nth_term first_term common_difference n = 0 ∧ n = 41 :=
by 
  sorry

end num_terms_arithmetic_sequence_is_41_l51_51035


namespace alloy_ratio_proof_l51_51514

def ratio_lead_to_tin_in_alloy_a (x y : ℝ) (ha : 0 < x) (hb : 0 < y) : Prop :=
  let weight_tin_in_a := (y / (x + y)) * 170
  let weight_tin_in_b := (3 / 8) * 250
  let total_tin := weight_tin_in_a + weight_tin_in_b
  total_tin = 221.25

theorem alloy_ratio_proof (x y : ℝ) (ha : 0 < x) (hb : 0 < y) (hc : ratio_lead_to_tin_in_alloy_a x y ha hb) : y / x = 3 :=
by
  -- Proof is omitted
  sorry

end alloy_ratio_proof_l51_51514


namespace monica_tiles_l51_51672

theorem monica_tiles (room_length : ℕ) (room_width : ℕ) (border_tile_size : ℕ) (inner_tile_size : ℕ) 
  (border_tiles : ℕ) (inner_tiles : ℕ) (total_tiles : ℕ) :
  room_length = 24 ∧ room_width = 18 ∧ border_tile_size = 2 ∧ inner_tile_size = 3 ∧ 
  border_tiles = 38 ∧ inner_tiles = 32 → total_tiles = 70 :=
by {
  sorry
}

end monica_tiles_l51_51672


namespace starting_even_number_l51_51767

def is_even (n : ℤ) : Prop := n % 2 = 0

def span_covered_by_evens (count : ℤ) : ℤ := count * 2 - 2

theorem starting_even_number
  (count : ℤ)
  (end_num : ℤ)
  (H1 : is_even end_num)
  (H2 : count = 20)
  (H3 : end_num = 55) :
  ∃ start_num, is_even start_num ∧ start_num = end_num - span_covered_by_evens count + 1 := 
sorry

end starting_even_number_l51_51767


namespace sixth_graders_more_than_seventh_l51_51663

theorem sixth_graders_more_than_seventh
  (bookstore_sells_pencils_in_whole_cents : True)
  (seventh_graders : ℕ)
  (sixth_graders : ℕ)
  (seventh_packs_payment : ℕ)
  (sixth_packs_payment : ℕ)
  (each_pack_contains_two_pencils : True)
  (seventh_graders_condition : seventh_graders = 25)
  (seventh_packs_payment_condition : seventh_packs_payment * seventh_graders = 275)
  (sixth_graders_condition : sixth_graders = 36 / 2)
  (sixth_packs_payment_condition : sixth_packs_payment * sixth_graders = 216) : 
  sixth_graders - seventh_graders = 7 := sorry

end sixth_graders_more_than_seventh_l51_51663


namespace problem_statement_l51_51296

def f (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem problem_statement : f (f (-1)) = 10 := by
  sorry

end problem_statement_l51_51296


namespace smallest_five_digit_neg_int_congruent_to_one_mod_17_l51_51980

theorem smallest_five_digit_neg_int_congruent_to_one_mod_17 :
  ∃ (x : ℤ), x < -9999 ∧ x % 17 = 1 ∧ x = -10011 := by
  -- The proof would go here
  sorry

end smallest_five_digit_neg_int_congruent_to_one_mod_17_l51_51980


namespace expanded_form_correct_l51_51248

theorem expanded_form_correct :
  (∃ a b c : ℤ, (∀ x : ℚ, 2 * (x - 3)^2 - 12 = a * x^2 + b * x + c) ∧ (10 * a - b - 4 * c = 8)) :=
by
  sorry

end expanded_form_correct_l51_51248


namespace percentage_not_pens_pencils_erasers_l51_51265

-- Define the given percentages
def percentPens : ℝ := 42
def percentPencils : ℝ := 25
def percentErasers : ℝ := 12
def totalPercent : ℝ := 100

-- The goal is to prove that the percentage of sales that were not pens, pencils, or erasers is 21%
theorem percentage_not_pens_pencils_erasers :
  totalPercent - (percentPens + percentPencils + percentErasers) = 21 := by
  sorry

end percentage_not_pens_pencils_erasers_l51_51265


namespace new_average_income_l51_51939

/-!
# Average Monthly Income Problem

## Problem Statement
Given:
1. The average monthly income of a family of 4 earning members was Rs. 735.
2. One of the earning members died, and the average income changed.
3. The income of the deceased member was Rs. 1170.

Prove that the new average monthly income of the family is Rs. 590.
-/

theorem new_average_income (avg_income : ℝ) (num_members : ℕ) (income_deceased : ℝ) (new_num_members : ℕ) 
  (h1 : avg_income = 735) 
  (h2 : num_members = 4) 
  (h3 : income_deceased = 1170) 
  (h4 : new_num_members = 3) : 
  (num_members * avg_income - income_deceased) / new_num_members = 590 := 
by 
  sorry

end new_average_income_l51_51939


namespace magic_square_l51_51220

variable (a b c d e s: ℕ)

axiom h1 : 30 + e + 18 = s
axiom h2 : 15 + c + d = s
axiom h3 : a + 27 + b = s
axiom h4 : 30 + 15 + a = s
axiom h5 : e + c + 27 = s
axiom h6 : 18 + d + b = s
axiom h7 : 30 + c + b = s
axiom h8 : a + c + 18 = s

theorem magic_square : d + e = 47 :=
by
  sorry

end magic_square_l51_51220


namespace measure_of_angle_C_l51_51856

theorem measure_of_angle_C (A B C : ℕ) (h1 : A + B = 150) (h2 : A + B + C = 180) : C = 30 := 
by
  sorry

end measure_of_angle_C_l51_51856


namespace gcd_of_factors_l51_51513

theorem gcd_of_factors (a b : ℕ) (h : a * b = 360) : 
    ∃ n : ℕ, n = 19 :=
by
  sorry

end gcd_of_factors_l51_51513


namespace Alyssa_total_spent_l51_51162

-- Declare the costs of grapes and cherries.
def costOfGrapes : ℝ := 12.08
def costOfCherries : ℝ := 9.85

-- Total amount spent by Alyssa.
def totalSpent : ℝ := 21.93

-- Statement to prove that the sum of the costs is equal to the total spent.
theorem Alyssa_total_spent (g : ℝ) (c : ℝ) (t : ℝ) 
  (hg : g = costOfGrapes) 
  (hc : c = costOfCherries) 
  (ht : t = totalSpent) :
  g + c = t := by
  sorry

end Alyssa_total_spent_l51_51162


namespace pq_sum_l51_51037

theorem pq_sum (p q : ℝ) 
  (h1 : p / 3 = 9) 
  (h2 : q / 3 = 15) : 
  p + q = 72 :=
sorry

end pq_sum_l51_51037


namespace total_cars_in_group_l51_51388

theorem total_cars_in_group (C : ℕ)
  (h1 : 37 ≤ C)
  (h2 : ∃ n ≥ 51, n ≤ C)
  (h3 : ∃ n ≤ 49, n + 51 = C - 37) :
  C = 137 :=
by
  sorry

end total_cars_in_group_l51_51388


namespace sum_of_reciprocals_of_squares_l51_51158

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 3) : (1 : ℚ)/a^2 + (1 : ℚ)/b^2 = 10/9 := 
sorry

end sum_of_reciprocals_of_squares_l51_51158


namespace necessarily_positive_b_plus_3c_l51_51439

theorem necessarily_positive_b_plus_3c 
  (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * c > 0 := 
sorry

end necessarily_positive_b_plus_3c_l51_51439


namespace cannonball_maximum_height_l51_51852

def height_function (t : ℝ) := -20 * t^2 + 100 * t + 36

theorem cannonball_maximum_height :
  ∃ t₀ : ℝ, ∀ t : ℝ, height_function t ≤ height_function t₀ ∧ height_function t₀ = 161 :=
by
  sorry

end cannonball_maximum_height_l51_51852


namespace algebraic_expression_evaluation_l51_51584

open Real

noncomputable def x : ℝ := 2 - sqrt 3

theorem algebraic_expression_evaluation :
  (7 + 4 * sqrt 3) * x^2 - (2 + sqrt 3) * x + sqrt 3 = 2 + sqrt 3 :=
by
  sorry

end algebraic_expression_evaluation_l51_51584


namespace smallest_value_arithmetic_geometric_seq_l51_51010

theorem smallest_value_arithmetic_geometric_seq :
  ∃ (E F G H : ℕ), (E < F) ∧ (F < G) ∧ (F * 4 = G * 7) ∧ (E + G = 2 * F) ∧ (F * F * 49 = G * G * 16) ∧ (E + F + G + H = 97) := 
sorry

end smallest_value_arithmetic_geometric_seq_l51_51010


namespace necessary_but_not_sufficient_l51_51656

theorem necessary_but_not_sufficient (x : ℝ) : ( (x + 1) * (x + 2) > 0 → (x + 1) * (x^2 + 2) > 0 ) :=
by
  intro h
  -- insert steps urther here, if proof was required
  sorry

end necessary_but_not_sufficient_l51_51656


namespace minimize_triangle_area_minimize_product_PA_PB_l51_51935

-- Define the initial conditions and geometry setup
def point (x y : ℝ) := (x, y)
def line_eq (a b : ℝ) := ∀ x y : ℝ, x / a + y / b = 1

-- Point P
def P := point 2 1

-- Condition: the line passes through point P and intersects the axes
def line_through_P (a b : ℝ) := line_eq a b ∧ (2 / a + 1 / b = 1) ∧ a > 2 ∧ b > 1

-- Prove that the line minimizing the area of triangle AOB is x + 2y - 4 = 0
theorem minimize_triangle_area (a b : ℝ) (h : line_through_P a b) :
  a = 4 ∧ b = 2 → line_eq 4 2 := 
sorry

-- Prove that the line minimizing the product |PA||PB| is x + y - 3 = 0
theorem minimize_product_PA_PB (a b : ℝ) (h : line_through_P a b) :
  a = 3 ∧ b = 3 → line_eq 3 3 := 
sorry

end minimize_triangle_area_minimize_product_PA_PB_l51_51935


namespace simplify_composite_product_fraction_l51_51971

def first_four_composite_product : ℤ := 4 * 6 * 8 * 9
def next_four_composite_product : ℤ := 10 * 12 * 14 * 15
def expected_fraction_num : ℤ := 12
def expected_fraction_den : ℤ := 175

theorem simplify_composite_product_fraction :
  (first_four_composite_product / next_four_composite_product : ℚ) = (expected_fraction_num / expected_fraction_den) :=
by
  rw [first_four_composite_product, next_four_composite_product]
  norm_num
  sorry

end simplify_composite_product_fraction_l51_51971


namespace distance_from_point_to_circle_center_l51_51791

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def circle_center : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_from_point_to_circle_center :
  distance (polar_to_rect 2 (Real.pi / 3)) circle_center = Real.sqrt 3 := sorry

end distance_from_point_to_circle_center_l51_51791


namespace carmen_candles_needed_l51_51209

-- Definitions based on the conditions

def candle_lifespan_1_hour : Nat := 8  -- a candle lasts 8 nights when burned 1 hour each night
def nights_total : Nat := 24  -- total nights

-- Question: How many candles are needed if burned 2 hours a night?

theorem carmen_candles_needed (h : candle_lifespan_1_hour / 2 = 4) :
  nights_total / 4 = 6 := 
  sorry

end carmen_candles_needed_l51_51209


namespace sum_of_digits_of_N_l51_51282

theorem sum_of_digits_of_N :
  ∃ N : ℕ, 
    10 ≤ N ∧ N < 100 ∧
    5655 % N = 11 ∧ 
    5879 % N = 14 ∧ 
    ((N / 10) + (N % 10)) = 8 := 
sorry

end sum_of_digits_of_N_l51_51282


namespace plane_centroid_l51_51118

theorem plane_centroid (a b : ℝ) (h : 1 / a ^ 2 + 1 / b ^ 2 + 1 / 25 = 1 / 4) :
  let p := a / 3
  let q := b / 3
  let r := 5 / 3
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 369 / 400 :=
by
  sorry

end plane_centroid_l51_51118


namespace carol_packs_l51_51239

theorem carol_packs (n_invites n_per_pack : ℕ) (h1 : n_invites = 12) (h2 : n_per_pack = 4) : n_invites / n_per_pack = 3 :=
by
  sorry

end carol_packs_l51_51239


namespace download_speeds_l51_51135

theorem download_speeds (x : ℕ) (s4 : ℕ := 4) (s5 : ℕ := 60) :
  (600 / x - 600 / (15 * x) = 140) → (x = s4 ∧ 15 * x = s5) := by
  sorry

end download_speeds_l51_51135


namespace distance_between_foci_of_ellipse_l51_51696

theorem distance_between_foci_of_ellipse :
  ∃ (a b c : ℝ),
  -- Condition: axes are parallel to the coordinate axes (implicitly given by tangency points).
  a = 3 ∧
  b = 2 ∧
  c = Real.sqrt (a^2 - b^2) ∧
  2 * c = 2 * Real.sqrt 5 :=
sorry

end distance_between_foci_of_ellipse_l51_51696


namespace geometric_product_is_geometric_l51_51487

theorem geometric_product_is_geometric (q : ℝ) (a : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  ∀ n, (a n) * (a (n + 1)) = (q^2) * (a (n - 1) * a n) := by
  sorry

end geometric_product_is_geometric_l51_51487


namespace arcsin_one_eq_pi_div_two_l51_51016

noncomputable def arcsin : ℝ → ℝ := sorry -- Define arcsin function

theorem arcsin_one_eq_pi_div_two : arcsin 1 = Real.pi / 2 := sorry

end arcsin_one_eq_pi_div_two_l51_51016


namespace grid_diagonal_segments_l51_51768

theorem grid_diagonal_segments (m n : ℕ) (hm : m = 100) (hn : n = 101) :
    let d := m + n - gcd m n
    d = 200 := by
  sorry

end grid_diagonal_segments_l51_51768


namespace unit_cost_calculation_l51_51319

theorem unit_cost_calculation : 
  ∀ (total_cost : ℕ) (ounces : ℕ), total_cost = 84 → ounces = 12 → (total_cost / ounces = 7) :=
by
  intros total_cost ounces h1 h2
  sorry

end unit_cost_calculation_l51_51319


namespace square_area_correct_l51_51429

-- Define the length of the side of the square
def side_length : ℕ := 15

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- Define the area calculation for a triangle using the square area division
def triangle_area (square_area : ℕ) : ℕ := square_area / 2

-- Theorem stating that the area of a square with given side length is 225 square units
theorem square_area_correct : square_area side_length = 225 := by
  sorry

end square_area_correct_l51_51429


namespace non_zero_digits_fraction_l51_51077

def count_non_zero_digits (n : ℚ) : ℕ :=
  -- A placeholder for the actual implementation.
  sorry

theorem non_zero_digits_fraction : count_non_zero_digits (120 / (2^4 * 5^9 : ℚ)) = 3 :=
  sorry

end non_zero_digits_fraction_l51_51077


namespace intersection_reciprocal_sum_l51_51241

open Real

theorem intersection_reciprocal_sum :
    ∀ (a b : ℝ),
    (∃ x : ℝ, x - 1 = a ∧ 3 / x = b) ∧
    (a * b = 3) →
    ∃ s : ℝ, (s = (a + b) / 3 ∨ s = -(a + b) / 3) ∧ (1 / a + 1 / b = s) := by
  sorry

end intersection_reciprocal_sum_l51_51241


namespace sum_of_perimeters_l51_51190

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) :
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * (Real.sqrt x^2 + Real.sqrt y^2) :=
by
  sorry

end sum_of_perimeters_l51_51190


namespace slices_per_pizza_l51_51160

theorem slices_per_pizza (total_slices number_of_pizzas slices_per_pizza : ℕ) 
  (h_total_slices : total_slices = 168) 
  (h_number_of_pizzas : number_of_pizzas = 21) 
  (h_division : total_slices / number_of_pizzas = slices_per_pizza) : 
  slices_per_pizza = 8 :=
sorry

end slices_per_pizza_l51_51160


namespace bus_tour_total_sales_l51_51415

noncomputable def total_sales (total_tickets sold_senior_tickets : Nat) (cost_senior_ticket cost_regular_ticket : Nat) : Nat :=
  let sold_regular_tickets := total_tickets - sold_senior_tickets
  let sales_senior := sold_senior_tickets * cost_senior_ticket
  let sales_regular := sold_regular_tickets * cost_regular_ticket
  sales_senior + sales_regular

theorem bus_tour_total_sales :
  total_sales 65 24 10 15 = 855 := by
    sorry

end bus_tour_total_sales_l51_51415


namespace seashells_total_l51_51543

theorem seashells_total :
    let Sam := 35
    let Joan := 18
    let Alex := 27
    Sam + Joan + Alex = 80 :=
by
    sorry

end seashells_total_l51_51543


namespace fraction_speed_bus_train_l51_51530

theorem fraction_speed_bus_train :
  let speed_train := 16 * 5
  let speed_bus := 480 / 8
  let speed_train_prop := speed_train = 80
  let speed_bus_prop := speed_bus = 60
  speed_bus / speed_train = 3 / 4 :=
by
  sorry

end fraction_speed_bus_train_l51_51530


namespace find_a_l51_51646

-- Given function
def quadratic_func (a x : ℝ) := a * (x - 1)^2 - a

-- Conditions
def condition1 (a : ℝ) := a ≠ 0
def condition2 (x : ℝ) := -1 ≤ x ∧ x ≤ 4
def min_value (y : ℝ) := y = -4

theorem find_a (a : ℝ) (ha : condition1 a) :
  ∃ a, (∀ x, condition2 x → quadratic_func a x = -4) → (a = 4 ∨ a = -1 / 2) :=
sorry

end find_a_l51_51646


namespace g_value_l51_51957

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) - 1

theorem g_value (ω φ : ℝ) (h : ∀ x : ℝ, f ω φ (π / 4 - x) = f ω φ (π / 4 + x)) :
  g ω φ (π / 4) = -1 :=
sorry

end g_value_l51_51957


namespace inequality_solution_l51_51525

theorem inequality_solution (x : ℝ) : 
  (2*x - 1) / (x - 3) ≥ 1 ↔ (x > 3 ∨ x ≤ -2) :=
by 
  sorry

end inequality_solution_l51_51525


namespace find_set_l51_51047

/-- Definition of set A -/
def setA : Set ℝ := { x : ℝ | abs x < 4 }

/-- Definition of set B -/
def setB : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 > 0 }

/-- Definition of the intersection A ∩ B -/
def intersectionAB : Set ℝ := { x : ℝ | abs x < 4 ∧ (x > 3 ∨ x < 1) }

/-- Definition of the set we want to find -/
def setDesired : Set ℝ := { x : ℝ | abs x < 4 ∧ ¬(abs x < 4 ∧ (x > 3 ∨ x < 1)) }

/-- The statement to prove -/
theorem find_set :
  setDesired = { x : ℝ | 1 ≤ x ∧ x ≤ 3 } :=
sorry

end find_set_l51_51047


namespace area_of_concentric_ring_l51_51202

theorem area_of_concentric_ring (r_large : ℝ) (r_small : ℝ) 
  (h1 : r_large = 10) 
  (h2 : r_small = 6) : 
  (π * r_large^2 - π * r_small^2) = 64 * π :=
by {
  sorry
}

end area_of_concentric_ring_l51_51202


namespace probability_of_PAIR_letters_in_PROBABILITY_l51_51655

theorem probability_of_PAIR_letters_in_PROBABILITY : 
  let total_letters := 11
  let favorable_letters := 4
  favorable_letters / total_letters = 4 / 11 :=
by
  let total_letters := 11
  let favorable_letters := 4
  show favorable_letters / total_letters = 4 / 11
  sorry

end probability_of_PAIR_letters_in_PROBABILITY_l51_51655


namespace hyperbola_condition_l51_51149

theorem hyperbola_condition (m : ℝ) : ((m - 2) * (m + 3) < 0) ↔ (-3 < m ∧ m < 0) := by
  sorry

end hyperbola_condition_l51_51149


namespace problem_solution_l51_51558

theorem problem_solution (a b c : ℤ)
  (h1 : a + 5 = b)
  (h2 : 5 + b = c)
  (h3 : b + c = a) : b = -10 :=
by
  sorry

end problem_solution_l51_51558


namespace minimum_ab_ge_four_l51_51424

variable (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
variable (h : 1 / a + 4 / b = Real.sqrt (a * b))

theorem minimum_ab_ge_four : a * b ≥ 4 := by
  sorry

end minimum_ab_ge_four_l51_51424


namespace Alice_favorite_number_l51_51666

theorem Alice_favorite_number :
  ∃ n : ℕ, (30 ≤ n ∧ n ≤ 70) ∧ (7 ∣ n) ∧ ¬(3 ∣ n) ∧ (4 ∣ (n / 10 + n % 10)) ∧ n = 35 :=
by
  sorry

end Alice_favorite_number_l51_51666


namespace length_of_chord_l51_51754

theorem length_of_chord {x1 x2 : ℝ} (h1 : ∃ (y : ℝ), y^2 = 8 * x1)
                                   (h2 : ∃ (y : ℝ), y^2 = 8 * x2)
                                   (h_midpoint : (x1 + x2) / 2 = 3) :
  x1 + x2 + 4 = 10 :=
sorry

end length_of_chord_l51_51754


namespace fourth_term_is_fifteen_l51_51122

-- Define the problem parameters
variables (a d : ℕ)

-- Define the conditions
def sum_first_third_term : Prop := (a + (a + 2 * d) = 10)
def fourth_term_def : ℕ := a + 3 * d

-- Declare the theorem to be proved
theorem fourth_term_is_fifteen (h1 : sum_first_third_term a d) : fourth_term_def a d = 15 :=
sorry

end fourth_term_is_fifteen_l51_51122


namespace math_problem_l51_51988

theorem math_problem 
  (num := 1 * 2 * 3 * 4 * 5 * 6 * 7)
  (den := 1 + 2 + 3 + 4 + 5 + 6 + 7) :
  (num / den) = 180 :=
by
  sorry

end math_problem_l51_51988


namespace quadratic_equations_with_common_root_l51_51758

theorem quadratic_equations_with_common_root :
  ∃ (p1 q1 p2 q2 : ℝ),
    p1 ≠ p2 ∧ q1 ≠ q2 ∧
    ∀ x : ℝ,
      (x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) →
      (x = 2 ∨ (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ ((x = r1 ∧ x == 2) ∨ (x = r2 ∧ x == 2)))) :=
sorry

end quadratic_equations_with_common_root_l51_51758


namespace distance_between_foci_of_ellipse_l51_51436

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 21 :=
by
  sorry

end distance_between_foci_of_ellipse_l51_51436


namespace minimum_shirts_for_savings_l51_51033

theorem minimum_shirts_for_savings (x : ℕ) : 75 + 8 * x < 16 * x ↔ 10 ≤ x :=
by
  sorry

end minimum_shirts_for_savings_l51_51033


namespace therapy_hours_l51_51968

variable (F A n : ℕ)
variable (h1 : F = A + 20)
variable (h2 : F + 2 * A = 188)
variable (h3 : F + A * (n - 1) = 300)

theorem therapy_hours : n = 5 := by
  sorry

end therapy_hours_l51_51968


namespace time_calculation_correct_l51_51682

theorem time_calculation_correct :
  let start_hour := 3
  let start_minute := 0
  let start_second := 0
  let hours_to_add := 158
  let minutes_to_add := 55
  let seconds_to_add := 32
  let total_seconds := seconds_to_add + minutes_to_add * 60 + hours_to_add * 3600
  let new_hour := (start_hour + (total_seconds / 3600) % 12) % 12
  let new_minute := (start_minute + (total_seconds / 60) % 60) % 60
  let new_second := (start_second + total_seconds % 60) % 60
  let A := new_hour
  let B := new_minute
  let C := new_second
  A + B + C = 92 :=
by
  sorry

end time_calculation_correct_l51_51682


namespace diagonal_less_than_half_perimeter_l51_51843

theorem diagonal_less_than_half_perimeter (a b c d x : ℝ) 
  (h1 : x < a + b) (h2 : x < c + d) : x < (a + b + c + d) / 2 := 
by
  sorry

end diagonal_less_than_half_perimeter_l51_51843


namespace range_of_a_l51_51120

def satisfies_p (x : ℝ) : Prop := (2 * x - 1) / (x - 1) ≤ 0

def satisfies_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) < 0

def sufficient_but_not_necessary (p q : ℝ → Prop) : Prop :=
  ∀ x, p x → q x ∧ ∃ x, q x ∧ ¬(p x)

theorem range_of_a :
  (∀ (x a : ℝ), satisfies_p x → satisfies_q x a → 0 ≤ a ∧ a < 1 / 2) ↔ (∀ a, 0 ≤ a ∧ a < 1 / 2) := by sorry

end range_of_a_l51_51120


namespace sum_first_19_terms_l51_51719

variable {α : Type} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + n * d

def sum_of_arithmetic_sequence (a d : α) (n : ℕ) : α := (n : α) / 2 * (2 * a + (n - 1) * d)

theorem sum_first_19_terms (a d : α) 
  (h1 : ∀ n, arithmetic_sequence a d (2 + n) + arithmetic_sequence a d (16 + n) = 10)
  (S19 : α) :
  sum_of_arithmetic_sequence a d 19 = 95 := by
  sorry

end sum_first_19_terms_l51_51719


namespace ratio_youngest_sister_to_yvonne_l51_51343

def laps_yvonne := 10
def laps_joel := 15
def joel_ratio := 3

theorem ratio_youngest_sister_to_yvonne
  (laps_yvonne : ℕ)
  (laps_joel : ℕ)
  (joel_ratio : ℕ)
  (H_joel : laps_joel = 3 * (laps_yvonne / joel_ratio))
  : (laps_joel / joel_ratio) = laps_yvonne / 2 :=
by
  sorry

end ratio_youngest_sister_to_yvonne_l51_51343


namespace jar_size_is_half_gallon_l51_51489

theorem jar_size_is_half_gallon : 
  ∃ (x : ℝ), (48 = 3 * 16) ∧ (16 + 16 * x + 16 * 0.25 = 28) ∧ x = 0.5 :=
by
  -- Implementation goes here
  sorry

end jar_size_is_half_gallon_l51_51489


namespace correct_fraction_l51_51350

theorem correct_fraction (x y : ℤ) (h : (5 / 6 : ℚ) * 384 = (x / y : ℚ) * 384 + 200) : x / y = 5 / 16 :=
by
  sorry

end correct_fraction_l51_51350


namespace smallest_two_ks_l51_51912

theorem smallest_two_ks (k : ℕ) (h : ℕ → Prop) : 
  (∀ k, (k^2 + 36) % 180 = 0 → k = 12 ∨ k = 18) :=
by {
 sorry
}

end smallest_two_ks_l51_51912


namespace max_regions_by_five_lines_l51_51261

theorem max_regions_by_five_lines : 
  ∀ (R : ℕ → ℕ), R 1 = 2 → R 2 = 4 → (∀ n, R (n + 1) = R n + (n + 1)) → R 5 = 16 :=
by
  intros R hR1 hR2 hRec
  sorry

end max_regions_by_five_lines_l51_51261


namespace probability_major_A_less_than_25_l51_51909

def total_students : ℕ := 100 -- assuming a total of 100 students for simplicity

def male_percent : ℝ := 0.40
def major_A_percent : ℝ := 0.50
def major_B_percent : ℝ := 0.30
def major_C_percent : ℝ := 0.20
def major_A_25_or_older_percent : ℝ := 0.60
def major_A_less_than_25_percent : ℝ := 1 - major_A_25_or_older_percent

theorem probability_major_A_less_than_25 :
  (major_A_percent * major_A_less_than_25_percent) = 0.20 :=
by
  sorry

end probability_major_A_less_than_25_l51_51909


namespace difference_of_numbers_l51_51458

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 34800) (h2 : b % 25 = 0) (h3 : b / 100 = a) : b - a = 32112 := by
  sorry

end difference_of_numbers_l51_51458


namespace combinations_problem_l51_51011

open Nat

-- Definitions for combinations
def C (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

-- Condition: Number of ways to choose 2 sergeants out of 6
def C_6_2 : Nat := C 6 2

-- Condition: Number of ways to choose 20 soldiers out of 60
def C_60_20 : Nat := C 60 20

-- Theorem statement for the problem
theorem combinations_problem :
  3 * C_6_2 * C_60_20 = 3 * 15 * C 60 20 := by
  simp [C_6_2, C_60_20, C]
  sorry

end combinations_problem_l51_51011


namespace jake_weight_l51_51227

theorem jake_weight:
  ∃ (J S : ℝ), (J - 8 = 2 * S) ∧ (J + S = 290) ∧ (J = 196) :=
by
  sorry

end jake_weight_l51_51227


namespace average_score_10_students_l51_51300

theorem average_score_10_students (x : ℝ)
  (h1 : 15 * 70 = 1050)
  (h2 : 25 * 78 = 1950)
  (h3 : 15 * 70 + 10 * x = 25 * 78) :
  x = 90 :=
sorry

end average_score_10_students_l51_51300


namespace determine_n_l51_51760

-- Define the condition
def eq1 := (1 : ℚ) / (2 ^ 10) + (1 : ℚ) / (2 ^ 9) + (1 : ℚ) / (2 ^ 8)
def eq2 (n : ℚ) := n / (2 ^ 10)

-- The lean statement for the proof problem
theorem determine_n : ∃ (n : ℤ), eq1 = eq2 n ∧ n > 0 ∧ n = 7 := by
  sorry

end determine_n_l51_51760


namespace purely_imaginary_sufficient_but_not_necessary_l51_51764

theorem purely_imaginary_sufficient_but_not_necessary (a b : ℝ) (h : ¬(b = 0)) : 
  (a = 0 → p ∧ q) → (q ∧ ¬p) :=
by
  sorry

end purely_imaginary_sufficient_but_not_necessary_l51_51764


namespace sqrt_x_minus_5_meaningful_iff_x_ge_5_l51_51500

theorem sqrt_x_minus_5_meaningful_iff_x_ge_5 (x : ℝ) : (∃ y : ℝ, y^2 = x - 5) ↔ (x ≥ 5) :=
sorry

end sqrt_x_minus_5_meaningful_iff_x_ge_5_l51_51500


namespace total_people_on_bus_l51_51029

def initial_people := 4
def added_people := 13

theorem total_people_on_bus : initial_people + added_people = 17 := by
  sorry

end total_people_on_bus_l51_51029


namespace maximum_alpha_l51_51997

noncomputable def is_in_F (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (3 * x) ≥ f (f (2 * x)) + x

theorem maximum_alpha :
  (∀ f : ℝ → ℝ, is_in_F f → ∀ x > 0, f x ≥ (1 / 2) * x) := 
by
  sorry

end maximum_alpha_l51_51997


namespace gcf_120_180_240_is_60_l51_51263

theorem gcf_120_180_240_is_60 : Nat.gcd (Nat.gcd 120 180) 240 = 60 := by
  sorry

end gcf_120_180_240_is_60_l51_51263


namespace stickers_per_friend_l51_51869

variable (d: ℕ) (h_d : d > 0)

theorem stickers_per_friend (h : 72 % d = 0) : 72 / d = 72 / d := by
  sorry

end stickers_per_friend_l51_51869


namespace triangle_sides_consecutive_and_angle_relationship_l51_51089

theorem triangle_sides_consecutive_and_angle_relationship (a b c : ℕ) 
  (h1 : a < b) (h2 : b < c) (h3 : b = a + 1) (h4 : c = b + 1) 
  (angle_A angle_B angle_C : ℝ) 
  (h_angle_sum : angle_A + angle_B + angle_C = π) 
  (h_angle_relation : angle_B = 2 * angle_A) : 
  (a, b, c) = (4, 5, 6) :=
sorry

end triangle_sides_consecutive_and_angle_relationship_l51_51089


namespace hyperbola_properties_l51_51877

theorem hyperbola_properties (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (c := Real.sqrt (a^2 + b^2))
  (F2 := (c, 0)) (P : ℝ × ℝ)
  (h_perpendicular : ∃ (x y : ℝ), P = (x, y) ∧ y = -a/b * (x - c))
  (h_distance : Real.sqrt ((P.1 - c)^2 + P.2^2) = 2)
  (h_slope : P.2 / (P.1 - c) = -1/2) :
  
  b = 2 ∧
  (∀ x y, x^2 - y^2 / 4 = 1 ↔ x^2 - y^2 / b^2 = 1) ∧
  P = (Real.sqrt (5) / 5, 2 * Real.sqrt (5) / 5) :=
sorry

end hyperbola_properties_l51_51877


namespace earnings_per_widget_l51_51542

-- Defining the conditions as constants
def hours_per_week : ℝ := 40
def hourly_wage : ℝ := 12.50
def total_weekly_earnings : ℝ := 700
def widgets_produced : ℝ := 1250

-- We need to prove earnings per widget
theorem earnings_per_widget :
  (total_weekly_earnings - (hours_per_week * hourly_wage)) / widgets_produced = 0.16 := by
  sorry

end earnings_per_widget_l51_51542


namespace gas_price_l51_51485

theorem gas_price (x : ℝ) (h1 : 10 * (x + 0.30) = 12 * x) : x + 0.30 = 1.80 := by
  sorry

end gas_price_l51_51485


namespace train_speed_kmph_l51_51177

/-- Given that the length of the train is 200 meters and it crosses a pole in 9 seconds,
the speed of the train in km/hr is 80. -/
theorem train_speed_kmph (length : ℝ) (time : ℝ) (length_eq : length = 200) (time_eq : time = 9) : 
  (length / time) * (3600 / 1000) = 80 :=
by
  sorry

end train_speed_kmph_l51_51177


namespace sum_div_mult_sub_result_l51_51146

-- Define the problem with conditions and expected answer
theorem sum_div_mult_sub_result :
  3521 + 480 / 60 * 3 - 521 = 3024 :=
by 
  sorry

end sum_div_mult_sub_result_l51_51146


namespace sphere_volume_ratio_l51_51640

theorem sphere_volume_ratio
  (r R : ℝ)
  (h : (4:ℝ) * π * r^2 / (4 * π * R^2) = (4:ℝ) / 9) : 
  (r^3 / R^3 = (8:ℝ) / 27) := by
  sorry

end sphere_volume_ratio_l51_51640


namespace indeterminate_equation_solution_l51_51392

theorem indeterminate_equation_solution (x y : ℝ) (n : ℕ) :
  (x^2 + (x + 1)^2 = y^2) ↔ 
  (x = 1/4 * ((1 + Real.sqrt 2)^(2*n + 1) + (1 - Real.sqrt 2)^(2*n + 1) - 2) ∧ 
   y = 1/(2 * Real.sqrt 2) * ((1 + Real.sqrt 2)^(2*n + 1) - (1 - Real.sqrt 2)^(2*n + 1))) := 
sorry

end indeterminate_equation_solution_l51_51392


namespace intersection_of_A_and_B_l51_51445

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by
  sorry

end intersection_of_A_and_B_l51_51445


namespace range_of_a_l51_51441

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2^x - a ≥ 0) ↔ (a ≤ 0) :=
by
  sorry

end range_of_a_l51_51441


namespace winning_percentage_l51_51124

theorem winning_percentage (total_votes winner_votes : ℕ) 
  (h1 : winner_votes = 1344) 
  (h2 : winner_votes - 288 = total_votes - winner_votes) : 
  (winner_votes * 100 / total_votes = 56) :=
sorry

end winning_percentage_l51_51124


namespace actual_distance_traveled_l51_51103

theorem actual_distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 15) : D = 40 := 
sorry

end actual_distance_traveled_l51_51103


namespace incorrect_locus_proof_l51_51846

-- Conditions given in the problem
def condition_A (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus ↔ conditions p)

def condition_B (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (conditions p ↔ p ∈ locus)

def condition_C (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus → conditions p) ∧ (∃ q, conditions q ∧ q ∈ locus)

def condition_D (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (p ∈ locus ↔ conditions p)

def condition_E (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (conditions p ↔ p ∈ locus) ∧ (¬ conditions p ↔ p ∉ locus)

-- Statement to be proved
theorem incorrect_locus_proof (locus : Set Point) (conditions : Point → Prop) :
  ¬ condition_C locus conditions :=
sorry

end incorrect_locus_proof_l51_51846


namespace no_positive_rational_solution_l51_51349

theorem no_positive_rational_solution :
  ¬ ∃ q : ℚ, 0 < q ∧ q^3 - 10 * q^2 + q - 2021 = 0 :=
by sorry

end no_positive_rational_solution_l51_51349


namespace intersection_x_value_l51_51816

theorem intersection_x_value :
  (∃ x y : ℝ, y = 5 * x - 20 ∧ y = 110 - 3 * x ∧ x = 16.25) := sorry

end intersection_x_value_l51_51816


namespace trajectory_of_P_l51_51211
-- Import entire library for necessary definitions and theorems.

-- Define the properties of the conic sections.
def ellipse (x y : ℝ) (n : ℝ) : Prop :=
  x^2 / 4 + y^2 / n = 1

def hyperbola (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 8 - y^2 / m = 1

-- Define the condition where the conic sections share the same foci.
def shared_foci (n m : ℝ) : Prop :=
  4 - n = 8 + m

-- The main theorem stating the relationship between m and n forming a straight line.
theorem trajectory_of_P : ∀ (n m : ℝ), shared_foci n m → (m + n + 4 = 0) :=
by
  intros n m h
  sorry

end trajectory_of_P_l51_51211


namespace determine_c_l51_51053

-- Definitions of the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 1

-- Hypothesis for the sequence to be geometric
def geometric_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∃ c, ∀ n, a (n + 1) + c = r * (a n + c)

-- The goal to prove
theorem determine_c (a : ℕ → ℕ) (c : ℕ) (r := 2) :
  seq a →
  geometric_seq a c →
  c = 1 :=
by
  intros h_seq h_geo
  sorry

end determine_c_l51_51053


namespace election_votes_l51_51290

theorem election_votes (V : ℕ) (h1 : ∃ Vb, Vb = 2509 ∧ (0.8 * V : ℝ) = (Vb + 0.15 * (V : ℝ)) + Vb) : V = 7720 :=
sorry

end election_votes_l51_51290


namespace monotonic_decreasing_interval_l51_51795

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval :
  {x : ℝ | x > 0} ∩ {x : ℝ | deriv f x < 0} = {x : ℝ | x > Real.exp 1} :=
by sorry

end monotonic_decreasing_interval_l51_51795


namespace total_books_is_10_l51_51913

def total_books (B : ℕ) : Prop :=
  (2 / 5 : ℚ) * B + (3 / 10 : ℚ) * B + ((3 / 10 : ℚ) * B - 1) + 1 = B

theorem total_books_is_10 : total_books 10 := by
  sorry

end total_books_is_10_l51_51913


namespace rate_of_current_l51_51271

/-- The speed of a boat in still water is 20 km/hr, and the rate of current is c km/hr.
    The distance travelled downstream in 24 minutes is 9.2 km. What is the rate of the current? -/
theorem rate_of_current (c : ℝ) (h : 24/60 = 0.4 ∧ 9.2 = (20 + c) * 0.4) : c = 3 :=
by
  sorry  -- Proof is not required, only the statement is necessary.

end rate_of_current_l51_51271


namespace upper_side_length_trapezoid_l51_51704

theorem upper_side_length_trapezoid
  (L U : ℝ) 
  (h : ℝ := 8) 
  (A : ℝ := 72) 
  (cond1 : U = L - 6)
  (cond2 : 1/2 * (L + U) * h = A) :
  U = 6 := 
by 
  sorry

end upper_side_length_trapezoid_l51_51704


namespace train_speed_proof_l51_51175

theorem train_speed_proof : 
  ∀ (V_A V_B : ℝ) (T_A T_B : ℝ), 
  T_A = 9 ∧ 
  T_B = 4 ∧ 
  V_B = 90 ∧ 
  (V_A / V_B = T_B / T_A) → 
  V_A = 40 := 
by
  intros V_A V_B T_A T_B h
  obtain ⟨hT_A, hT_B, hV_B, hprop⟩ := h
  sorry

end train_speed_proof_l51_51175


namespace find_y_l51_51028

theorem find_y (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 24) : y = 120 :=
by
  sorry

end find_y_l51_51028


namespace solve_inequality_l51_51906

def within_interval (x : ℝ) : Prop :=
  x < 2 ∧ x > -5

theorem solve_inequality (x : ℝ) : (x^2 + 3 * x < 10) ↔ within_interval x :=
sorry

end solve_inequality_l51_51906


namespace monotone_increasing_interval_l51_51247

noncomputable def f (x : ℝ) : ℝ := (x / (x^2 + 1)) + 1

theorem monotone_increasing_interval :
  ∀ x : ℝ, (-1 < x ∧ x < 1) ↔ ∀ ε > 0, ∃ δ > 0, ∀ x₁ x₂, (-1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧ |x₁ - x₂| < δ) → f x₁ ≤ f x₂ + ε := 
sorry

end monotone_increasing_interval_l51_51247


namespace ellipse_foci_l51_51303

noncomputable def focal_coordinates (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

-- Given the equation of the ellipse: x^2 / a^2 + y^2 / b^2 = 1
def ellipse_equation (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

-- Proposition stating that if the ellipse equation holds for a=√5 and b=2, then the foci are at (± c, 0)
theorem ellipse_foci (x y : ℝ) (h : ellipse_equation x y (Real.sqrt 5) 2) :
  y = 0 ∧ (x = 1 ∨ x = -1) :=
sorry

end ellipse_foci_l51_51303


namespace hyperbola_eccentricity_asymptotes_l51_51032

theorem hyperbola_eccentricity_asymptotes :
  (∃ e: ℝ, ∃ m: ℝ, 
    (∀ x y, (x^2 / 8 - y^2 / 4 = 1) → e = Real.sqrt 6 / 2 ∧ y = m * x) ∧ 
    (m = Real.sqrt 2 / 2 ∨ m = -Real.sqrt 2 / 2)) :=
sorry

end hyperbola_eccentricity_asymptotes_l51_51032


namespace problem_l51_51002

theorem problem (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
  sorry

end problem_l51_51002


namespace people_sharing_bill_l51_51142

theorem people_sharing_bill (total_bill : ℝ) (tip_percent : ℝ) (share_per_person : ℝ) (n : ℝ) :
  total_bill = 211.00 →
  tip_percent = 0.15 →
  share_per_person = 26.96 →
  abs (n - 9) < 1 :=
by
  intros h1 h2 h3
  sorry

end people_sharing_bill_l51_51142


namespace man_rowing_upstream_speed_l51_51112

theorem man_rowing_upstream_speed (V_down V_m V_up V_s : ℕ) 
  (h1 : V_down = 41)
  (h2 : V_m = 33)
  (h3 : V_down = V_m + V_s)
  (h4 : V_up = V_m - V_s) 
  : V_up = 25 := 
by
  sorry

end man_rowing_upstream_speed_l51_51112


namespace weight_of_new_person_l51_51618

-- Definitions for the conditions given.

-- Average weight increase
def avg_weight_increase : ℝ := 2.5

-- Number of persons
def num_persons : ℕ := 8

-- Weight of the person being replaced
def weight_replaced : ℝ := 65

-- Total weight increase
def total_weight_increase : ℝ := num_persons * avg_weight_increase

-- Statement to prove the weight of the new person
theorem weight_of_new_person : 
  ∃ (W_new : ℝ), W_new = weight_replaced + total_weight_increase :=
sorry

end weight_of_new_person_l51_51618


namespace xyz_sum_l51_51132

theorem xyz_sum (x y z : ℕ) (h1 : xyz = 240) (h2 : xy + z = 46) (h3 : x + yz = 64) : x + y + z = 20 :=
sorry

end xyz_sum_l51_51132


namespace sum_of_first_15_terms_of_geometric_sequence_l51_51978

theorem sum_of_first_15_terms_of_geometric_sequence (a r : ℝ) 
  (h₁ : (a * (1 - r^5)) / (1 - r) = 10) 
  (h₂ : (a * (1 - r^10)) / (1 - r) = 50) : 
  (a * (1 - r^15)) / (1 - r) = 210 := 
by 
  sorry

end sum_of_first_15_terms_of_geometric_sequence_l51_51978


namespace divide_one_meter_into_100_parts_l51_51279

theorem divide_one_meter_into_100_parts :
  (1 / 100 : ℝ) = 1 / 100 := 
by
  sorry

end divide_one_meter_into_100_parts_l51_51279


namespace sets_relation_l51_51559

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def M : Set ℚ := {x | ∃ (m : ℤ), x = m + 1/6}
def S : Set ℚ := {x | ∃ (s : ℤ), x = s/2 - 1/3}
def P : Set ℚ := {x | ∃ (p : ℤ), x = p/2 + 1/6}

theorem sets_relation : M ⊆ S ∧ S = P := by
  sorry

end sets_relation_l51_51559


namespace number_of_true_statements_l51_51079

theorem number_of_true_statements 
  (a b c : ℝ) 
  (Hc : c ≠ 0) : 
  ((a > b → a * c^2 > b * c^2) ∧ (a * c^2 ≤ b * c^2 → a ≤ b)) ∧ 
  ¬((a * c^2 > b * c^2 → a > b) ∨ (a ≤ b → a * c^2 ≤ b * c^2)) :=
by
  sorry

end number_of_true_statements_l51_51079


namespace least_number_to_add_l51_51443

theorem least_number_to_add (a : ℕ) (b : ℕ) (n : ℕ) (h : a = 1056) (h1: b = 26) (h2 : n = 10) : 
  (a + n) % b = 0 := 
sorry

end least_number_to_add_l51_51443


namespace geometric_progression_identity_l51_51102

theorem geometric_progression_identity (a b c : ℝ) (h : b^2 = a * c) : 
  (a + b + c) * (a - b + c) = a^2 + b^2 + c^2 := 
by
  sorry

end geometric_progression_identity_l51_51102


namespace magnet_cost_times_sticker_l51_51977

theorem magnet_cost_times_sticker
  (M S A : ℝ)
  (hM : M = 3)
  (hA : A = 6)
  (hMagnetCost : M = (1/4) * 2 * A) :
  M = 4 * S :=
by
  -- Placeholder, the actual proof would go here
  sorry

end magnet_cost_times_sticker_l51_51977


namespace train_speed_in_m_per_s_l51_51131

theorem train_speed_in_m_per_s (speed_kmph : ℕ) (h : speed_kmph = 162) :
  (speed_kmph * 1000) / 3600 = 45 :=
by {
  sorry
}

end train_speed_in_m_per_s_l51_51131


namespace combined_length_of_trains_is_correct_l51_51164

noncomputable def combined_length_of_trains : ℕ :=
  let speed_A := 120 * 1000 / 3600 -- speed of train A in m/s
  let speed_B := 100 * 1000 / 3600 -- speed of train B in m/s
  let speed_motorbike := 64 * 1000 / 3600 -- speed of motorbike in m/s
  let relative_speed_A := (120 - 64) * 1000 / 3600 -- relative speed of train A with respect to motorbike in m/s
  let relative_speed_B := (100 - 64) * 1000 / 3600 -- relative speed of train B with respect to motorbike in m/s
  let length_A := relative_speed_A * 75 -- length of train A in meters
  let length_B := relative_speed_B * 90 -- length of train B in meters
  length_A + length_B

theorem combined_length_of_trains_is_correct :
  combined_length_of_trains = 2067 :=
  by
  sorry

end combined_length_of_trains_is_correct_l51_51164


namespace correct_statements_l51_51782

theorem correct_statements (a b c x : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -2 ∨ x ≥ 6)
  (hb : b = -4 * a)
  (hc : c = -12 * a) : 
  (a < 0) ∧ 
  (∀ x, cx^2 - bx + a < 0 ↔ -1/6 < x ∧ x < 1/2) ∧ 
  (a + b + c > 0) :=
by
  sorry

end correct_statements_l51_51782


namespace sum_of_squares_of_roots_eq_226_l51_51579

theorem sum_of_squares_of_roots_eq_226 (s_1 s_2 : ℝ) (h_eq : ∀ x, x^2 - 16 * x + 15 = 0 → (x = s_1 ∨ x = s_2)) :
  s_1^2 + s_2^2 = 226 := by
  sorry

end sum_of_squares_of_roots_eq_226_l51_51579


namespace max_area_ABC_l51_51730

noncomputable def q (p : ℝ) : ℝ := p^2 - 7*p + 10

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem max_area_ABC : ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ 
  triangle_area (2, 0) (5, 4) (p, q p) = 0.536625 := sorry

end max_area_ABC_l51_51730


namespace ratio_of_segments_l51_51484

theorem ratio_of_segments (a b : ℕ) (ha : a = 200) (hb : b = 40) : a / b = 5 :=
by sorry

end ratio_of_segments_l51_51484


namespace measure_of_C_l51_51289

-- Define angles and their magnitudes
variables (A B C X : Type) [LinearOrder C]
def angle_measure (angle : Type) : ℕ := sorry
def parallel (l1 l2 : Type) : Prop := sorry
def transversal (l1 l2 l3 : Type) : Prop := sorry
def alternate_interior (angle1 angle2 : Type) : Prop := sorry
def adjacent (angle1 angle2 : Type) : Prop := sorry
def complementary (angle1 angle2 : Type) : Prop := sorry

-- The given conditions
axiom h1 : parallel A X
axiom h2 : transversal A B X
axiom h3 : angle_measure A = 85
axiom h4 : angle_measure B = 35
axiom h5 : alternate_interior C A
axiom h6 : complementary B X
axiom h7 : adjacent C X

-- Define the proof problem
theorem measure_of_C : angle_measure C = 85 :=
by {
  -- The proof goes here, skipping with sorry
  sorry
}

end measure_of_C_l51_51289


namespace problem_value_l51_51238

theorem problem_value :
  1 - (-2) - 3 - (-4) - 5 - (-6) = 5 :=
by sorry

end problem_value_l51_51238


namespace exists_group_of_three_friends_l51_51679

-- Defining the context of the problem
def people := Fin 10 -- a finite set of 10 people
def quarrel (x y : people) : Prop := -- a predicate indicating a quarrel between two people
sorry

-- Given conditions
axiom quarreled_pairs : ∃ S : Finset (people × people), S.card = 14 ∧ 
  ∀ {x y : people}, (x, y) ∈ S → x ≠ y ∧ quarrel x y

-- Question: Prove there exists a set of 3 friends among these 10 people
theorem exists_group_of_three_friends (p : Finset people):
  ∃ (group : Finset people), group.card = 3 ∧ ∀ {x y : people}, 
  x ∈ group → y ∈ group → x ≠ y → ¬ quarrel x y :=
sorry

end exists_group_of_three_friends_l51_51679


namespace arithmetic_mean_two_digit_multiples_of_8_l51_51858

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l51_51858


namespace min_abs_difference_on_hyperbola_l51_51256

theorem min_abs_difference_on_hyperbola : 
  ∀ (x y : ℝ), (x^2 / 8 - y^2 / 4 = 1) → abs (x - y) ≥ 2 := 
by
  intros x y hxy
  sorry

end min_abs_difference_on_hyperbola_l51_51256


namespace find_interest_rate_l51_51161

theorem find_interest_rate
  (P A : ℝ) (n t : ℕ) (r : ℝ)
  (hP : P = 100)
  (hA : A = 121.00000000000001)
  (hn : n = 2)
  (ht : t = 1)
  (compound_interest : A = P * (1 + r / n) ^ (n * t)) :
  r = 0.2 :=
by
  sorry

end find_interest_rate_l51_51161


namespace sum_of_three_numbers_l51_51883

theorem sum_of_three_numbers (x y z : ℕ) (h1 : x ≤ y) (h2 : y ≤ z) (h3 : y = 7) 
    (h4 : (x + y + z) / 3 = x + 12) (h5 : (x + y + z) / 3 = z - 18) : 
    x + y + z = 39 :=
by
  sorry

end sum_of_three_numbers_l51_51883


namespace bryce_raisins_l51_51793

theorem bryce_raisins:
  ∃ x : ℕ, (x - 8 = x / 3) ∧ x = 12 :=
by 
  sorry

end bryce_raisins_l51_51793


namespace equal_areas_triangle_height_l51_51188

theorem equal_areas_triangle_height (l b h : ℝ) (hlb : l > b) 
  (H1 : l * b = (1/2) * l * h) : h = 2 * b :=
by 
  -- skipping proof
  sorry

end equal_areas_triangle_height_l51_51188


namespace not_lucky_1994_l51_51820

def is_valid_month (m : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12

def is_valid_day (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 31

def is_lucky_year (y : ℕ) : Prop :=
  ∃ (m d : ℕ), is_valid_month m ∧ is_valid_day d ∧ m * d = y

theorem not_lucky_1994 : ¬ is_lucky_year 94 := 
by
  sorry

end not_lucky_1994_l51_51820


namespace equation_is_point_l51_51900

-- Definition of the condition in the problem
def equation (x y : ℝ) := x^2 + 36*y^2 - 12*x - 72*y + 36 = 0

-- The theorem stating the equivalence to the point (6, 1)
theorem equation_is_point :
  ∀ (x y : ℝ), equation x y → (x = 6 ∧ y = 1) :=
by
  intros x y h
  -- The proof steps would go here
  sorry

end equation_is_point_l51_51900


namespace sam_total_spent_l51_51337

-- Define the values of a penny and a dime in dollars
def penny_value : ℝ := 0.01
def dime_value : ℝ := 0.10

-- Define what Sam spent
def friday_spent : ℝ := 2 * penny_value
def saturday_spent : ℝ := 12 * dime_value

-- Define total spent
def total_spent : ℝ := friday_spent + saturday_spent

theorem sam_total_spent : total_spent = 1.22 := 
by
  -- The following is a placeholder for the actual proof
  sorry

end sam_total_spent_l51_51337


namespace set_complement_union_eq_l51_51832

open Set

variable (U : Set ℕ) (P : Set ℕ) (Q : Set ℕ)

theorem set_complement_union_eq :
  U = {1, 2, 3, 4, 5, 6} →
  P = {1, 3, 5} →
  Q = {1, 2, 4} →
  (U \ P) ∪ Q = {1, 2, 4, 6} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end set_complement_union_eq_l51_51832


namespace sqrt_eq_sum_seven_l51_51966

open Real

theorem sqrt_eq_sum_seven (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
    sqrt (64 - x^2) + sqrt (36 - x^2) = 7 :=
by
  sorry

end sqrt_eq_sum_seven_l51_51966


namespace range_of_a_l51_51294

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ↔ 0 ≤ a ∧ a < 4 := sorry

end range_of_a_l51_51294


namespace domain_of_function_l51_51421

/-- Prove the domain of the function f(x) = log10(2 * cos x - 1) + sqrt(49 - x^2) -/
theorem domain_of_function :
  { x : ℝ | -7 ≤ x ∧ x < - (5 * Real.pi) / 3 ∨ - Real.pi / 3 < x ∧ x < Real.pi / 3 ∨ (5 * Real.pi) / 3 < x ∧ x ≤ 7 }
  = { x : ℝ | 2 * Real.cos x - 1 > 0 ∧ 49 - x^2 ≥ 0 } :=
by {
  sorry
}

end domain_of_function_l51_51421


namespace volleyball_shotput_cost_l51_51152

theorem volleyball_shotput_cost (x y : ℝ) :
  (2*x + 3*y = 95) ∧ (5*x + 7*y = 230) :=
  sorry

end volleyball_shotput_cost_l51_51152


namespace company_employee_count_l51_51255

theorem company_employee_count (E : ℝ) (H1 : E > 0) (H2 : 0.60 * E = 0.55 * (E + 30)) : E + 30 = 360 :=
by
  -- The proof steps would go here, but that is not required.
  sorry

end company_employee_count_l51_51255


namespace scientific_notation_of_1500_l51_51735

theorem scientific_notation_of_1500 :
  (1500 : ℝ) = 1.5 * 10^3 :=
sorry

end scientific_notation_of_1500_l51_51735


namespace floor_e_equals_two_l51_51317

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l51_51317


namespace greatest_sum_consecutive_lt_400_l51_51902

noncomputable def greatest_sum_of_consecutive_integers (n : ℤ) : ℤ :=
if n * (n + 1) < 400 then n + (n + 1) else 0

theorem greatest_sum_consecutive_lt_400 : ∃ n : ℤ, n * (n + 1) < 400 ∧ greatest_sum_of_consecutive_integers n = 39 :=
by
  sorry

end greatest_sum_consecutive_lt_400_l51_51902


namespace number_of_red_balls_l51_51156

theorem number_of_red_balls (total_balls : ℕ) (probability : ℚ) (num_red_balls : ℕ) 
  (h1 : total_balls = 12) 
  (h2 : probability = 1 / 22) 
  (h3 : (num_red_balls * (num_red_balls - 1) : ℚ) / (total_balls * (total_balls - 1)) = probability) :
  num_red_balls = 3 := 
by
  sorry

end number_of_red_balls_l51_51156


namespace proof_problem_l51_51126

open Classical

variable (x y z : ℝ)

theorem proof_problem
  (cond1 : 0 < x ∧ x < 1)
  (cond2 : 0 < y ∧ y < 1)
  (cond3 : 0 < z ∧ z < 1)
  (cond4 : x * y * z = (1 - x) * (1 - y) * (1 - z)) :
  ((1 - x) * y ≥ 1/4) ∨ ((1 - y) * z ≥ 1/4) ∨ ((1 - z) * x ≥ 1/4) := by
  sorry

end proof_problem_l51_51126


namespace evaluate_expression_l51_51962

theorem evaluate_expression (x : ℝ) : 
  (36 + 12 * x) ^ 2 - (12^2 * x^2 + 36^2) = 864 * x :=
by
  sorry

end evaluate_expression_l51_51962


namespace find_y_intercept_of_second_parabola_l51_51278

theorem find_y_intercept_of_second_parabola :
  ∃ D : ℝ × ℝ, D = (0, 9) ∧ 
    (∃ A : ℝ × ℝ, A = (10, 4) ∧ 
     ∃ B : ℝ × ℝ, B = (6, 0) ∧ 
     (∀ x y : ℝ, y = (-1/4) * x ^ 2 + 5 * x - 21 → A = (10, 4)) ∧ 
     (∀ x y : ℝ, y = (1/4) * (x - B.1) ^ 2 + B.2 ∧ y = 4 ∧ B = (6, 0) → A = (10, 4))) :=
  sorry

end find_y_intercept_of_second_parabola_l51_51278


namespace negation_correct_l51_51348

namespace NegationProof

-- Define the original proposition 
def orig_prop : Prop := ∃ x : ℝ, x ≤ 0

-- Define the negation of the original proposition
def neg_prop : Prop := ∀ x : ℝ, x > 0

-- The theorem we need to prove
theorem negation_correct : ¬ orig_prop = neg_prop := by
  sorry

end NegationProof

end negation_correct_l51_51348


namespace solve_system_l51_51838

-- Definitions for the system of equations.
def system_valid (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

-- Main theorem to prove.
theorem solve_system (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : 
  system_valid y x₁ x₂ x₃ x₄ x₅ →
  ((y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨ 
  (y = 2 → ∃ (t : ℝ), x₁ = t ∧ x₂ = t ∧ x₃ = t ∧ x₄ = t ∧ x₅ = t) ∨ 
  (y^2 + y - 1 = 0 → ∃ (u v : ℝ), 
    x₁ = u ∧ 
    x₅ = v ∧ 
    x₂ = y * u - v ∧ 
    x₃ = -y * (u + v) ∧ 
    x₄ = y * v - u ∧ 
    (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2))) :=
by
  intro h
  sorry

end solve_system_l51_51838


namespace fraction_of_time_riding_at_15mph_l51_51938

variable (t_5 t_15 : ℝ)

-- Conditions
def no_stops : Prop := (t_5 ≠ 0 ∧ t_15 ≠ 0)
def average_speed (t_5 t_15 : ℝ) : Prop := (5 * t_5 + 15 * t_15) / (t_5 + t_15) = 10

-- Question to be proved
theorem fraction_of_time_riding_at_15mph (h1 : no_stops t_5 t_15) (h2 : average_speed t_5 t_15) :
  t_15 / (t_5 + t_15) = 1 / 2 :=
sorry

end fraction_of_time_riding_at_15mph_l51_51938


namespace triangle_area_formula_l51_51230

theorem triangle_area_formula (a b c R : ℝ) (α β γ : ℝ) 
    (h1 : a / (Real.sin α) = 2 * R) 
    (h2 : b / (Real.sin β) = 2 * R) 
    (h3 : c / (Real.sin γ) = 2 * R) :
    let S := (1 / 2) * a * b * (Real.sin γ)
    S = a * b * c / (4 * R) := 
by 
  sorry

end triangle_area_formula_l51_51230


namespace range_of_a_l51_51101

theorem range_of_a (a : ℝ) : 
  4 * a^2 - 12 * (a + 6) > 0 ↔ a < -3 ∨ a > 6 := 
by sorry

end range_of_a_l51_51101


namespace find_f_3_l51_51575

def f (x : ℝ) : ℝ := x^2 + 4 * x + 8

theorem find_f_3 : f 3 = 29 := by
  sorry

end find_f_3_l51_51575


namespace subtracted_value_l51_51780

-- Given conditions
def chosen_number : ℕ := 110
def result_number : ℕ := 110

-- Statement to prove
theorem subtracted_value : ∃ y : ℕ, 3 * chosen_number - y = result_number ∧ y = 220 :=
by
  sorry

end subtracted_value_l51_51780


namespace kim_monthly_expenses_l51_51776

-- Define the conditions

def initial_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def payback_period : ℕ := 10

-- Define the proof statement
theorem kim_monthly_expenses :
  ∃ (E : ℝ), 
    (payback_period * (monthly_revenue - E) = initial_cost) → (E = 1500) :=
by
  sorry

end kim_monthly_expenses_l51_51776


namespace equal_triples_l51_51674

theorem equal_triples (a b c x : ℝ) (h_abc : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : (xb + (1 - x) * c) / a = (x * c + (1 - x) * a) / b ∧ 
          (x * c + (1 - x) * a) / b = (x * a + (1 - x) * b) / c) : a = b ∧ b = c := by
  sorry

end equal_triples_l51_51674


namespace james_fish_weight_l51_51097

theorem james_fish_weight :
  let trout := 200
  let salmon := trout + (trout * 0.5)
  let tuna := 2 * salmon
  trout + salmon + tuna = 1100 := 
by
  sorry

end james_fish_weight_l51_51097


namespace primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l51_51136

-- Define Part (a)
theorem primitive_root_coprime_distinct_residues (m k : ℕ) (h: Nat.gcd m k = 1) :
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∀ i j s t, (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k) :=
sorry

-- Define Part (b)
theorem noncoprime_non_distinct_residues (m k : ℕ) (h: Nat.gcd m k > 1) :
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∃ i j x t, (i ≠ x ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a x * b t) % (m * k) :=
sorry

end primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l51_51136


namespace sum_of_cubes_l51_51237

theorem sum_of_cubes (k : ℤ) : 
  24 * k = (k + 2)^3 + (-k)^3 + (-k)^3 + (k - 2)^3 :=
by
  sorry

end sum_of_cubes_l51_51237


namespace find_X_l51_51727

theorem find_X : ∃ X : ℝ, 1.5 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1200.0000000000002 ∧ X = 0.3 :=
by
  sorry

end find_X_l51_51727


namespace magician_earned_4_dollars_l51_51949

-- Define conditions
def price_per_deck := 2
def initial_decks := 5
def decks_left := 3

-- Define the number of decks sold
def decks_sold := initial_decks - decks_left

-- Define the total money earned
def money_earned := decks_sold * price_per_deck

-- Theorem to prove the money earned is 4 dollars
theorem magician_earned_4_dollars : money_earned = 4 := by
  sorry

end magician_earned_4_dollars_l51_51949


namespace number_machine_output_l51_51067

def number_machine (n : ℕ) : ℕ :=
  let step1 := n * 3
  let step2 := step1 + 20
  let step3 := step2 / 2
  let step4 := step3 ^ 2
  let step5 := step4 - 45
  step5

theorem number_machine_output : number_machine 90 = 20980 := by
  sorry

end number_machine_output_l51_51067


namespace total_vehicles_l51_51143

theorem total_vehicles (morn_minivans afternoon_minivans evening_minivans night_minivans : Nat)
                       (morn_sedans afternoon_sedans evening_sedans night_sedans : Nat)
                       (morn_SUVs afternoon_SUVs evening_SUVs night_SUVs : Nat)
                       (morn_trucks afternoon_trucks evening_trucks night_trucks : Nat)
                       (morn_motorcycles afternoon_motorcycles evening_motorcycles night_motorcycles : Nat) :
                       morn_minivans = 20 → afternoon_minivans = 22 → evening_minivans = 15 → night_minivans = 10 →
                       morn_sedans = 17 → afternoon_sedans = 13 → evening_sedans = 19 → night_sedans = 12 →
                       morn_SUVs = 12 → afternoon_SUVs = 15 → evening_SUVs = 18 → night_SUVs = 20 →
                       morn_trucks = 8 → afternoon_trucks = 10 → evening_trucks = 14 → night_trucks = 20 →
                       morn_motorcycles = 5 → afternoon_motorcycles = 7 → evening_motorcycles = 10 → night_motorcycles = 15 →
                       morn_minivans + afternoon_minivans + evening_minivans + night_minivans +
                       morn_sedans + afternoon_sedans + evening_sedans + night_sedans +
                       morn_SUVs + afternoon_SUVs + evening_SUVs + night_SUVs +
                       morn_trucks + afternoon_trucks + evening_trucks + night_trucks +
                       morn_motorcycles + afternoon_motorcycles + evening_motorcycles + night_motorcycles = 282 :=
by
  intros
  sorry

end total_vehicles_l51_51143


namespace correct_statements_l51_51225

-- Definitions based on the conditions and question
def S (n : ℕ) : ℤ := -n^2 + 7 * n + 1

-- Definition of the sequence an
def a (n : ℕ) : ℤ := 
  if n = 1 then 7 
  else S n - S (n - 1)

-- Theorem statements based on the correct answers derived from solution
theorem correct_statements :
  (∀ n : ℕ, n > 4 → a n < 0) ∧ (S 3 = S 4 ∧ (∀ m : ℕ, S m ≤ S 3)) :=
by {
  sorry
}

end correct_statements_l51_51225


namespace solution_to_water_l51_51186

theorem solution_to_water (A W S T: ℝ) (h1: A = 0.04) (h2: W = 0.02) (h3: S = 0.06) (h4: T = 0.48) :
  (T * (W / S) = 0.16) :=
by
  sorry

end solution_to_water_l51_51186


namespace cardinals_home_runs_second_l51_51407

-- Define the conditions
def cubs_home_runs_third : ℕ := 2
def cubs_home_runs_fifth : ℕ := 1
def cubs_home_runs_eighth : ℕ := 2
def cubs_total_home_runs := cubs_home_runs_third + cubs_home_runs_fifth + cubs_home_runs_eighth
def cubs_more_than_cardinals : ℕ := 3
def cardinals_home_runs_fifth : ℕ := 1

-- Define the proof problem
theorem cardinals_home_runs_second :
  (cubs_total_home_runs = cardinals_total_home_runs + cubs_more_than_cardinals) →
  (cardinals_total_home_runs - cardinals_home_runs_fifth = 1) :=
sorry

end cardinals_home_runs_second_l51_51407


namespace no_real_roots_abs_eq_l51_51620

theorem no_real_roots_abs_eq (x : ℝ) : 
  |2*x - 5| + |3*x - 7| + |5*x - 11| = 2015/2016 → false :=
by sorry

end no_real_roots_abs_eq_l51_51620


namespace best_fit_model_l51_51750

theorem best_fit_model 
  (R2_model1 R2_model2 R2_model3 R2_model4 : ℝ)
  (h1 : R2_model1 = 0.976)
  (h2 : R2_model2 = 0.776)
  (h3 : R2_model3 = 0.076)
  (h4 : R2_model4 = 0.351) : 
  (R2_model1 > R2_model2) ∧ (R2_model1 > R2_model3) ∧ (R2_model1 > R2_model4) :=
by
  sorry

end best_fit_model_l51_51750


namespace evaluate_expression_l51_51213

theorem evaluate_expression :
  ((3.5 / 0.7) * (5 / 3) + (7.2 / 0.36) - ((5 / 3) * (0.75 / 0.25))) = 23.3335 :=
by
  sorry

end evaluate_expression_l51_51213


namespace find_a_b_l51_51936

theorem find_a_b (a b : ℤ) : (∀ (s : ℂ), s^2 + s - 1 = 0 → a * s^18 + b * s^17 + 1 = 0) → (a = 987 ∧ b = -1597) :=
by
  sorry

end find_a_b_l51_51936


namespace f_10_equals_1_l51_51006

noncomputable def f : ℝ → ℝ 
| x => sorry 

axiom odd_f_x_minus_1 : ∀ x : ℝ, f (-x-1) = -f (x-1)
axiom even_f_x_plus_1 : ∀ x : ℝ, f (-x+1) = f (x+1)
axiom f_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = 2^x

theorem f_10_equals_1 : f 10 = 1 :=
by
  sorry -- The actual proof goes here.

end f_10_equals_1_l51_51006


namespace average_apples_per_hour_l51_51675

theorem average_apples_per_hour (total_apples : ℝ) (total_hours : ℝ) (h1 : total_apples = 5.0) (h2 : total_hours = 3.0) : total_apples / total_hours = 1.67 :=
  sorry

end average_apples_per_hour_l51_51675


namespace man_speed_42_minutes_7_km_l51_51012

theorem man_speed_42_minutes_7_km 
  (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ)
  (h1 : distance = 7) 
  (h2 : time_minutes = 42) 
  (h3 : time_hours = time_minutes / 60) :
  distance / time_hours = 10 := by
  sorry

end man_speed_42_minutes_7_km_l51_51012


namespace area_of_inner_square_l51_51970

theorem area_of_inner_square (s₁ s₂ : ℝ) (side_length_WXYZ : ℝ) (WI : ℝ) (area_IJKL : ℝ) 
  (h1 : s₁ = 10) 
  (h2 : s₂ = 10 - 2 * Real.sqrt 2)
  (h3 : side_length_WXYZ = 10)
  (h4 : WI = 2)
  (h5 : area_IJKL = (s₂)^2): 
  area_IJKL = 102 - 20 * Real.sqrt 2 :=
by
  sorry

end area_of_inner_square_l51_51970


namespace system_of_equations_solution_l51_51397

theorem system_of_equations_solution :
  ∃ x y z : ℝ, x + y = 1 ∧ y + z = 2 ∧ z + x = 3 ∧ x = 1 ∧ y = 0 ∧ z = 2 :=
by
  sorry

end system_of_equations_solution_l51_51397


namespace proof_combination_l51_51919

open Classical

theorem proof_combination :
  (∃ x : ℝ, x^3 < 1) ∧ (¬ ∃ x : ℚ, x^2 = 2) ∧ (¬ ∀ x : ℕ, x^3 > x^2) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by
  have h1 : ∃ x : ℝ, x^3 < 1 := sorry
  have h2 : ¬ ∃ x : ℚ, x^2 = 2 := sorry
  have h3 : ¬ ∀ x : ℕ, x^3 > x^2 := sorry
  have h4 : ∀ x : ℝ, x^2 + 1 > 0 := sorry
  exact ⟨h1, h2, h3, h4⟩

end proof_combination_l51_51919


namespace molecular_weight_cao_is_correct_l51_51041

-- Define the atomic weights of calcium and oxygen
def atomic_weight_ca : ℝ := 40.08
def atomic_weight_o : ℝ := 16.00

-- Define the molecular weight of CaO
def molecular_weight_cao : ℝ := atomic_weight_ca + atomic_weight_o

-- State the theorem to prove
theorem molecular_weight_cao_is_correct : molecular_weight_cao = 56.08 :=
by
  sorry

end molecular_weight_cao_is_correct_l51_51041


namespace soldiers_in_groups_l51_51098

theorem soldiers_in_groups (x : ℕ) (h1 : x % 2 = 1) (h2 : x % 3 = 2) (h3 : x % 5 = 3) : x % 30 = 23 :=
by
  sorry

end soldiers_in_groups_l51_51098


namespace min_double_rooms_needed_min_triple_rooms_needed_with_discount_l51_51623

-- Define the conditions 
def double_room_price : ℕ := 200
def triple_room_price : ℕ := 250
def total_students : ℕ := 50
def male_students : ℕ := 27
def female_students : ℕ := 23
def discount : ℚ := 0.2
def max_double_rooms : ℕ := 15

-- Define the property for part (1)
theorem min_double_rooms_needed (d : ℕ) (t : ℕ) : 
  2 * d + 3 * t = total_students ∧
  2 * (d - 1) + 3 * t ≠ total_students :=
sorry

-- Define the property for part (2)
theorem min_triple_rooms_needed_with_discount (d : ℕ) (t : ℕ) : 
  d + t = total_students ∧
  d ≤ max_double_rooms ∧
  2 * d + 3 * t = total_students ∧
  (1* (d - 1) + 3 * t ≠ total_students) :=
sorry

end min_double_rooms_needed_min_triple_rooms_needed_with_discount_l51_51623


namespace CarlaDailyItems_l51_51736

theorem CarlaDailyItems (leaves bugs days : ℕ) 
  (h_leaves : leaves = 30) 
  (h_bugs : bugs = 20) 
  (h_days : days = 10) : 
  (leaves + bugs) / days = 5 := 
by 
  sorry

end CarlaDailyItems_l51_51736


namespace hamburger_cost_l51_51761

variable (H : ℝ)

theorem hamburger_cost :
  (H + 2 + 3 = 20 - 11) → (H = 4) :=
by
  sorry

end hamburger_cost_l51_51761


namespace butterfly_probability_l51_51121

-- Define the vertices of the cube
inductive Vertex
| A | B | C | D | E | F | G | H

open Vertex

-- Define the edges of the cube
def edges : Vertex → List Vertex
| A => [B, D, E]
| B => [A, C, F]
| C => [B, D, G]
| D => [A, C, H]
| E => [A, F, H]
| F => [B, E, G]
| G => [C, F, H]
| H => [D, E, G]

-- Define a function to simulate the butterfly's movement
noncomputable def move : Vertex → ℕ → List (Vertex × ℕ)
| v, 0 => [(v, 0)]
| v, n + 1 =>
  let nextMoves := edges v
  nextMoves.bind (λ v' => move v' n)

-- Define the probability calculation part
noncomputable def probability_of_visiting_all_vertices (n_moves : ℕ) : ℚ :=
  let total_paths := (3 ^ n_moves : ℕ)
  let valid_paths := 27 -- Based on given final solution step
  valid_paths / total_paths

-- Statement of the problem in Lean 4
theorem butterfly_probability :
  probability_of_visiting_all_vertices 11 = 27 / 177147 :=
by
  sorry

end butterfly_probability_l51_51121


namespace sequence_expression_l51_51147

noncomputable def seq (n : ℕ) : ℝ := 
  match n with
  | 0 => 1  -- note: indexing from 1 means a_1 corresponds to seq 0 in Lean
  | m+1 => seq m / (3 * seq m + 1)

theorem sequence_expression (n : ℕ) : 
  ∀ n, seq (n + 1) = 1 / (3 * (n + 1) - 2) := 
sorry

end sequence_expression_l51_51147


namespace income_is_12000_l51_51088

theorem income_is_12000 (P : ℝ) : (P * 1.02 = 12240) → (P = 12000) :=
by
  intro h
  sorry

end income_is_12000_l51_51088


namespace tan_angle_addition_l51_51021

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 :=
sorry

end tan_angle_addition_l51_51021


namespace baseball_card_value_decrease_l51_51081

theorem baseball_card_value_decrease (V0 : ℝ) (V1 V2 : ℝ) :
  V1 = V0 * 0.5 → V2 = V1 * 0.9 → (V0 - V2) / V0 * 100 = 55 :=
by 
  intros hV1 hV2
  sorry

end baseball_card_value_decrease_l51_51081


namespace radius_of_smaller_circle_l51_51540

theorem radius_of_smaller_circle (R : ℝ) (n : ℕ) (r : ℝ) 
  (hR : R = 10) 
  (hn : n = 7) 
  (condition : 2 * R = 2 * r * n) :
  r = 10 / 7 :=
by
  sorry

end radius_of_smaller_circle_l51_51540


namespace maplewood_total_population_l51_51419

-- Define the number of cities
def num_cities : ℕ := 25

-- Define the bounds for the average population
def lower_bound : ℕ := 5200
def upper_bound : ℕ := 5700

-- Define the average population, calculated as the midpoint of the bounds
def average_population : ℕ := (lower_bound + upper_bound) / 2

-- Define the total population as the product of the number of cities and the average population
def total_population : ℕ := num_cities * average_population

-- Theorem statement to prove the total population is 136,250
theorem maplewood_total_population : total_population = 136250 := by
  -- Insert formal proof here
  sorry

end maplewood_total_population_l51_51419


namespace cubic_root_conditions_l51_51722

-- Define the cubic polynomial
def cubic (a b : ℝ) (x : ℝ) : ℝ := x^3 + a * x + b

-- Define a predicate for the cubic equation having exactly one real root
def has_one_real_root (a b : ℝ) : Prop :=
  ∀ y : ℝ, cubic a b y = 0 → ∃! x : ℝ, cubic a b x = 0

-- Theorem statement
theorem cubic_root_conditions (a b : ℝ) :
  (a = -3 ∧ b = -3) ∨ (a = -3 ∧ b > 2) ∨ (a = 0 ∧ b = 2) → has_one_real_root a b :=
sorry

end cubic_root_conditions_l51_51722


namespace percent_savings_correct_l51_51022

theorem percent_savings_correct :
  let cost_of_package := 9
  let num_of_rolls_in_package := 12
  let cost_per_roll_individually := 1
  let cost_per_roll_in_package := cost_of_package / num_of_rolls_in_package
  let savings_per_roll := cost_per_roll_individually - cost_per_roll_in_package
  let percent_savings := (savings_per_roll / cost_per_roll_individually) * 100
  percent_savings = 25 :=
by
  sorry

end percent_savings_correct_l51_51022


namespace fractional_inspection_l51_51272

theorem fractional_inspection:
  ∃ (J E A : ℝ),
  J + E + A = 1 ∧
  0.005 * J + 0.007 * E + 0.012 * A = 0.01 :=
by
  sorry

end fractional_inspection_l51_51272


namespace polynomial_sum_l51_51964

noncomputable def p (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
noncomputable def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
noncomputable def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = 12 * x - 12 := by
  sorry

end polynomial_sum_l51_51964


namespace initial_gummy_worms_l51_51839

variable (G : ℕ)

theorem initial_gummy_worms (h : (G : ℚ) / 16 = 4) : G = 64 :=
by
  sorry

end initial_gummy_worms_l51_51839


namespace combined_area_percentage_l51_51503

theorem combined_area_percentage (D_S : ℝ) (D_R : ℝ) (D_T : ℝ) (A_S A_R A_T : ℝ)
  (h1 : D_R = 0.20 * D_S)
  (h2 : D_T = 0.40 * D_R)
  (h3 : A_R = Real.pi * (D_R / 2) ^ 2)
  (h4 : A_T = Real.pi * (D_T / 2) ^ 2)
  (h5 : A_S = Real.pi * (D_S / 2) ^ 2) :
  ((A_R + A_T) / A_S) * 100 = 4.64 := by
  sorry

end combined_area_percentage_l51_51503


namespace find_cd_l51_51268

def g (c d x : ℝ) := c * x^3 - 7 * x^2 + d * x - 4

theorem find_cd : ∃ c d : ℝ, (g c d 2 = -4) ∧ (g c d (-1) = -22) ∧ (c = 19/3) ∧ (d = -8/3) := 
by
  sorry

end find_cd_l51_51268


namespace sum_of_extreme_values_of_g_l51_51273

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - 2 * abs (x - 3)

theorem sum_of_extreme_values_of_g :
  ∃ (min_val max_val : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≥ min_val) ∧ 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≤ max_val) ∧ 
    (min_val = -8) ∧ 
    (max_val = 0) ∧ 
    (min_val + max_val = -8) := 
by
  sorry

end sum_of_extreme_values_of_g_l51_51273


namespace evaluate_expression_l51_51286

theorem evaluate_expression : (6^6) * (12^6) * (6^12) * (12^12) = 72^18 := 
by sorry

end evaluate_expression_l51_51286


namespace probability_odd_sum_l51_51194

-- Definitions based on the conditions
def cards : List ℕ := [1, 2, 3, 4, 5]

def is_odd_sum (a b : ℕ) : Prop := (a + b) % 2 = 1

def combinations (n k : ℕ) : ℕ := (Nat.choose n k)

-- Main statement
theorem probability_odd_sum :
  (combinations 5 2) = 10 → -- Total combinations of 2 cards from 5
  (∃ N, N = 6 ∧ (N:ℚ)/(combinations 5 2) = 3/5) :=
by 
  sorry

end probability_odd_sum_l51_51194


namespace remainder_8357_to_8361_div_9_l51_51087

theorem remainder_8357_to_8361_div_9 :
  (8357 + 8358 + 8359 + 8360 + 8361) % 9 = 3 := 
by
  sorry

end remainder_8357_to_8361_div_9_l51_51087


namespace travel_time_K_l51_51459

theorem travel_time_K (d x : ℝ) (h_pos_d : d > 0) (h_x_pos : x > 0) (h_time_diff : (d / (x - 1/2)) - (d / x) = 1/2) : d / x = 40 / x :=
by
  sorry

end travel_time_K_l51_51459


namespace num_arithmetic_sequences_l51_51699

theorem num_arithmetic_sequences (a d : ℕ) (n : ℕ) (h1 : n >= 3) (h2 : n * (2 * a + (n - 1) * d) = 2 * 97^2) :
  ∃ seqs : ℕ, seqs = 4 :=
by sorry

end num_arithmetic_sequences_l51_51699


namespace express_in_scientific_notation_l51_51802

theorem express_in_scientific_notation (x : ℝ) (h : x = 720000) : x = 7.2 * 10^5 :=
by sorry

end express_in_scientific_notation_l51_51802


namespace _l51_51553

noncomputable def probability_event_b_given_a : ℕ → ℕ → ℕ → ℕ × ℕ → ℚ
| zeros, ones, twos, (1, drawn_label) =>
  if drawn_label = 1 then
    (ones * (ones - 1)) / (zeros + ones + twos).choose 2
  else 0
| _, _, _, _ => 0

lemma probability_theorem :
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  (1 - 1) * (ones - 1)/(total.choose 2) = 1/7 :=
by
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  let draw_label := 1
  let event_b_given_a := probability_event_b_given_a zeros ones twos (1, draw_label)
  have pos_cases : (ones * (ones - 1))/(total.choose 2) = 1 / 7 := by sorry
  exact pos_cases

end _l51_51553


namespace kelsey_more_than_ekon_l51_51855

theorem kelsey_more_than_ekon :
  ∃ (K E U : ℕ), (K = 160) ∧ (E = U - 17) ∧ (K + E + U = 411) ∧ (K - E = 43) :=
by
  sorry

end kelsey_more_than_ekon_l51_51855


namespace ratio_of_small_rectangle_length_to_width_l51_51280

-- Define the problem conditions
variables (s : ℝ)

-- Define the length and width of the small rectangle
def length_of_small_rectangle := 3 * s
def width_of_small_rectangle := s

-- Prove that the ratio of the length to the width of the small rectangle is 3
theorem ratio_of_small_rectangle_length_to_width : 
  length_of_small_rectangle s / width_of_small_rectangle s = 3 :=
by
  sorry

end ratio_of_small_rectangle_length_to_width_l51_51280


namespace distinct_fib_sum_2017_l51_51686

-- Define the Fibonacci sequence as given.
def fib : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => (fib (n+1)) + (fib n)

-- Define the predicate for representing a number as a sum of distinct Fibonacci numbers.
def can_be_written_as_sum_of_distinct_fibs (n : ℕ) : Prop :=
  ∃ s : Finset ℕ, (s.sum fib = n) ∧ (∀ (i j : ℕ), i ≠ j → i ∉ s → j ∉ s)

theorem distinct_fib_sum_2017 : ∃! s : Finset ℕ, s.sum fib = 2017 ∧ (∀ (i j : ℕ), i ≠ j → i ≠ j → i ∉ s → j ∉ s) :=
sorry

end distinct_fib_sum_2017_l51_51686


namespace clock_strike_time_l51_51967

theorem clock_strike_time (t : ℕ) (n m : ℕ) (I : ℕ) : 
  t = 12 ∧ n = 3 ∧ m = 6 ∧ 2 * I = t → (m - 1) * I = 30 := by 
  sorry

end clock_strike_time_l51_51967


namespace inf_solutions_l51_51270

theorem inf_solutions (x y z : ℤ) : 
  ∃ (infinitely many relatively prime solutions : ℕ), x^2 + y^2 = z^5 + z :=
sorry

end inf_solutions_l51_51270


namespace cost_for_paving_is_486_l51_51311

-- Definitions and conditions
def ratio_longer_side : ℝ := 4
def ratio_shorter_side : ℝ := 3
def diagonal : ℝ := 45
def cost_per_sqm : ℝ := 0.5 -- converting pence to pounds

-- Mathematical formulation
def longer_side (x : ℝ) : ℝ := ratio_longer_side * x
def shorter_side (x : ℝ) : ℝ := ratio_shorter_side * x
def area_of_rectangle (l w : ℝ) : ℝ := l * w
def cost_paving (area : ℝ) (cost_per_sqm : ℝ) : ℝ := area * cost_per_sqm

-- Main problem: given the conditions, prove that the cost is £486.
theorem cost_for_paving_is_486 (x : ℝ) 
  (h1 : (ratio_longer_side^2 + ratio_shorter_side^2) * x^2 = diagonal^2) :
  cost_paving (area_of_rectangle (longer_side x) (shorter_side x)) cost_per_sqm = 486 :=
by
  sorry

end cost_for_paving_is_486_l51_51311


namespace fraction_of_automobile_installment_credit_extended_by_finance_companies_l51_51201

theorem fraction_of_automobile_installment_credit_extended_by_finance_companies
  (total_consumer_credit : ℝ)
  (percentage_auto_credit : ℝ)
  (credit_extended_by_finance_companies : ℝ)
  (total_auto_credit_fraction : percentage_auto_credit = 0.36)
  (total_consumer_credit_value : total_consumer_credit = 475)
  (credit_extended_by_finance_companies_value : credit_extended_by_finance_companies = 57) :
  credit_extended_by_finance_companies / (percentage_auto_credit * total_consumer_credit) = 1 / 3 :=
by
  -- The proof part will go here.
  sorry

end fraction_of_automobile_installment_credit_extended_by_finance_companies_l51_51201


namespace sin_of_right_angle_l51_51775

theorem sin_of_right_angle (A B C : Type)
  (angle_A : Real) (AB BC : Real)
  (h_angleA : angle_A = 90)
  (h_AB : AB = 16)
  (h_BC : BC = 24) :
  Real.sin (angle_A) = 1 :=
by
  sorry

end sin_of_right_angle_l51_51775


namespace kids_all_three_activities_l51_51541

-- Definitions based on conditions
def total_kids : ℕ := 40
def kids_tubing : ℕ := total_kids / 4
def kids_tubing_rafting : ℕ := kids_tubing / 2
def kids_tubing_rafting_kayaking : ℕ := kids_tubing_rafting / 3

-- Theorem statement: proof of the final answer
theorem kids_all_three_activities : kids_tubing_rafting_kayaking = 1 := by
  sorry

end kids_all_three_activities_l51_51541


namespace leak_empties_tank_in_10_hours_l51_51193

theorem leak_empties_tank_in_10_hours :
  (∀ (A L : ℝ), (A = 1/5) → (A - L = 1/10) → (1 / L = 10)) 
  := by
  intros A L hA hAL
  sorry

end leak_empties_tank_in_10_hours_l51_51193


namespace find_c_l51_51027

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 3) (h3 : abs (6 + 4 * c) = 14) : c = 2 :=
by {
  sorry
}

end find_c_l51_51027


namespace find_b_l51_51555

theorem find_b (a : ℝ) (A : ℝ) (B : ℝ) (b : ℝ)
  (ha : a = 5) 
  (hA : A = Real.pi / 6) 
  (htanB : Real.tan B = 3 / 4)
  (hsinB : Real.sin B = 3 / 5):
  b = 6 := 
by 
  sorry

end find_b_l51_51555


namespace two_person_subcommittees_l51_51956

def committee_size := 8
def sub_committee_size := 2
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem two_person_subcommittees : combination committee_size sub_committee_size = 28 := by
  sorry

end two_person_subcommittees_l51_51956


namespace problem_inequality_l51_51591

theorem problem_inequality (a b c : ℝ) : a^2 + b^2 + c^2 + 4 ≥ ab + 3*b + 2*c := 
by 
  sorry

end problem_inequality_l51_51591


namespace a3_value_l51_51465

theorem a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (x : ℝ) :
  ( (1 + x) * (a - x) ^ 6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 ) →
  ( a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 ) →
  a = 1 →
  a₃ = -5 :=
by
  sorry

end a3_value_l51_51465


namespace alison_money_l51_51593

theorem alison_money (k b br bt al : ℝ) 
  (h1 : al = 1/2 * bt) 
  (h2 : bt = 4 * br) 
  (h3 : br = 2 * k) 
  (h4 : k = 1000) : 
  al = 4000 := 
by 
  sorry

end alison_money_l51_51593


namespace coffee_cost_l51_51630

theorem coffee_cost :
  ∃ y : ℕ, 
  (∃ x : ℕ, 3 * x + 2 * y = 630 ∧ 2 * x + 3 * y = 690) → y = 162 :=
by
  sorry

end coffee_cost_l51_51630


namespace angle_complement_supplement_l51_51145

theorem angle_complement_supplement (θ : ℝ) (h1 : 90 - θ = (1/3) * (180 - θ)) : θ = 45 :=
by
  sorry

end angle_complement_supplement_l51_51145


namespace general_formula_sum_first_n_terms_l51_51038

open BigOperators

def geometric_sequence (a_3 : ℚ) (q : ℚ) : ℕ → ℚ
| 0       => 1 -- this is a placeholder since sequence usually start from 1
| (n + 1) => 1 * q ^ n

def sum_geometric_sequence (a_1 q : ℚ) (n : ℕ) : ℚ :=
  a_1 * (1 - q ^ n) / (1 - q)

theorem general_formula (a_3 : ℚ) (q : ℚ) (n : ℕ) (h_a3 : a_3 = 1 / 4) (h_q : q = -1 / 2) :
  geometric_sequence a_3 q (n + 1) = (-1 / 2) ^ n :=
by
  sorry

theorem sum_first_n_terms (a_1 q : ℚ) (n : ℕ) (h_a1 : a_1 = 1) (h_q : q = -1 / 2) :
  sum_geometric_sequence a_1 q n = 2 / 3 * (1 - (-1 / 2) ^ n) :=
by
  sorry

end general_formula_sum_first_n_terms_l51_51038


namespace find_x_l51_51312

-- Define the conditions as variables and the target equation
variable (x : ℝ)

theorem find_x : 67 * x - 59 * x = 4828 → x = 603.5 := by
  intro h
  sorry

end find_x_l51_51312


namespace uki_total_earnings_l51_51203

def cupcake_price : ℝ := 1.50
def cookie_price : ℝ := 2.00
def biscuit_price : ℝ := 1.00
def daily_cupcakes : ℕ := 20
def daily_cookies : ℕ := 10
def daily_biscuits : ℕ := 20
def days : ℕ := 5

theorem uki_total_earnings :
  5 * ((daily_cupcakes * cupcake_price) + (daily_cookies * cookie_price) + (daily_biscuits * biscuit_price)) = 350 :=
by
  -- This is a placeholder for the proof
  sorry

end uki_total_earnings_l51_51203


namespace necessary_but_not_sufficient_l51_51922

-- Definitions extracted from the problem conditions
def isEllipse (k : ℝ) : Prop := (9 - k > 0) ∧ (k - 7 > 0) ∧ (9 - k ≠ k - 7)

-- The necessary but not sufficient condition for the ellipse equation
theorem necessary_but_not_sufficient : 
  (7 < k ∧ k < 9) → isEllipse k → (isEllipse k ↔ (7 < k ∧ k < 9)) := 
by 
  sorry

end necessary_but_not_sufficient_l51_51922


namespace eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l51_51034

theorem eight_digit_numbers_with_012 :
  let total_sequences := 3^8 
  let invalid_sequences := 3^7 
  total_sequences - invalid_sequences = 4374 :=
by sorry

theorem eight_digit_numbers_with_00012222 :
  let total_sequences := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 4)
  let invalid_sequences := Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 4)
  total_sequences - invalid_sequences = 175 :=
by sorry

theorem eight_digit_numbers_starting_with_1_0002222 :
  let number_starting_with_1 := Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 4)
  number_starting_with_1 = 35 :=
by sorry

end eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l51_51034


namespace distinct_sequences_count_l51_51479

-- Define the set of available letters excluding 'M' for start and 'S' for end
def available_letters : List Char := ['A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C']

-- Define the cardinality function for the sequences under given specific conditions.
-- This will check specific prompt format; you may want to specify permutations, combinations based on calculations but in the spirit, we are sticking to detail.
def count_sequences (letters : List Char) (n : Nat) : Nat :=
  if letters = available_letters ∧ n = 5 then 
    -- based on detailed calculation in the solution
    480
  else
    0

-- Theorem statement in Lean 4 to verify the number of sequences
theorem distinct_sequences_count : count_sequences available_letters 5 = 480 := 
sorry

end distinct_sequences_count_l51_51479


namespace hyperbola_asymptotes_l51_51545

noncomputable def eccentricity_asymptotes (a b : ℝ) (h₁ : a > 0) (h₂ : b = Real.sqrt 15 * a) : Prop :=
  ∀ (x y : ℝ), (y = (Real.sqrt 15) * x) ∨ (y = -(Real.sqrt 15) * x)

theorem hyperbola_asymptotes (a : ℝ) (h₁ : a > 0) :
  eccentricity_asymptotes a (Real.sqrt 15 * a) h₁ (by simp) :=
sorry

end hyperbola_asymptotes_l51_51545


namespace inscribed_square_ratio_l51_51635

theorem inscribed_square_ratio
  (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 13) (h₁ : a^2 + b^2 = c^2)
  (x y : ℝ) (hx : x = 60 / 17) (hy : y = 144 / 17) :
  (x / y) = 5 / 12 := sorry

end inscribed_square_ratio_l51_51635


namespace thirtieth_triangular_number_is_465_l51_51882

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem thirtieth_triangular_number_is_465 : triangular_number 30 = 465 :=
by
  sorry

end thirtieth_triangular_number_is_465_l51_51882


namespace tangent_line_slope_l51_51594

theorem tangent_line_slope (k : ℝ) :
  (∃ m : ℝ, (m^3 - m^2 + m = k * m) ∧ (k = 3 * m^2 - 2 * m + 1)) →
  (k = 1 ∨ k = 3 / 4) :=
by
  -- Proof goes here
  sorry

end tangent_line_slope_l51_51594


namespace beaker_volume_l51_51385

theorem beaker_volume {a b c d e f g h i j : ℝ} (h₁ : a = 7) (h₂ : b = 4) (h₃ : c = 5)
                      (h₄ : d = 4) (h₅ : e = 6) (h₆ : f = 8) (h₇ : g = 7)
                      (h₈ : h = 3) (h₉ : i = 9) (h₁₀ : j = 6) :
  (a + b + c + d + e + f + g + h + i + j) / 5 = 11.8 :=
by
  sorry

end beaker_volume_l51_51385


namespace total_shopping_cost_l51_51335

theorem total_shopping_cost 
  (sandwiches : ℕ := 3)
  (sandwich_cost : ℕ := 3)
  (water_bottle : ℕ := 1)
  (water_cost : ℕ := 2)
  : sandwiches * sandwich_cost + water_bottle * water_cost = 11 :=
by
  sorry

end total_shopping_cost_l51_51335


namespace prod_eq_diff_squares_l51_51377

variable (a b : ℝ)

theorem prod_eq_diff_squares :
  ( (1 / 4 * a + b) * (b - 1 / 4 * a) = b^2 - (1 / 16 * a^2) ) :=
by
  sorry

end prod_eq_diff_squares_l51_51377


namespace sequence_geometric_progression_iff_b1_eq_b2_l51_51207

theorem sequence_geometric_progression_iff_b1_eq_b2 
  (b : ℕ → ℝ) 
  (h0 : ∀ n, b n > 0)
  (h1 : ∀ n, b (n + 2) = 3 * b n * b (n + 1)) :
  (∃ r : ℝ, ∀ n, b (n + 1) = r * b n) ↔ b 1 = b 0 :=
sorry

end sequence_geometric_progression_iff_b1_eq_b2_l51_51207


namespace triangle_XYZ_ratio_l51_51310

theorem triangle_XYZ_ratio (XZ YZ : ℝ)
  (hXZ : XZ = 9) (hYZ : YZ = 40)
  (XY : ℝ) (hXY : XY = Real.sqrt (XZ ^ 2 + YZ ^ 2))
  (ZD : ℝ) (hZD : ZD = Real.sqrt (XZ * YZ))
  (XJ YJ : ℝ) (hXJ : XJ = Real.sqrt (XZ * (XZ + 2 * ZD)))
  (hYJ : YJ = Real.sqrt (YZ * (YZ + 2 * ZD)))
  (ratio : ℝ) (h_ratio : ratio = (XJ + YJ + XY) / XY) :
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ ratio = p / q ∧ p + q = 203 := sorry

end triangle_XYZ_ratio_l51_51310


namespace C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l51_51140

noncomputable def parametric_C1 (α : ℝ) : ℝ × ℝ := (2 + Real.cos α, 4 + Real.sin α)

noncomputable def polar_C2 (ρ θ m : ℝ) : ℝ := ρ * (Real.cos θ - m * Real.sin θ) + 1

theorem C1_Cartesian_equation :
  ∀ (x y : ℝ), (∃ α : ℝ, parametric_C1 α = (x, y)) ↔ (x - 2)^2 + (y - 4)^2 = 1 := sorry

theorem C2_Cartesian_equation :
  ∀ (x y m : ℝ), (∃ ρ θ : ℝ, polar_C2 ρ θ m = 0 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)
  ↔ x - m * y + 1 = 0 := sorry

def closest_point_on_C1_to_x_axis : ℝ × ℝ := (2, 3)

theorem m_value_when_C2_passes_through_P :
  ∃ (m : ℝ), x - m * y + 1 = 0 ∧ x = 2 ∧ y = 3 ∧ m = 1 := sorry

end C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l51_51140


namespace average_speed_l51_51360

def dist1 : ℝ := 60
def dist2 : ℝ := 30
def time : ℝ := 2

theorem average_speed : (dist1 + dist2) / time = 45 := by
  sorry

end average_speed_l51_51360


namespace michael_total_earnings_l51_51538

-- Define the cost of large paintings and small paintings
def large_painting_cost : ℕ := 100
def small_painting_cost : ℕ := 80

-- Define the number of large and small paintings sold
def large_paintings_sold : ℕ := 5
def small_paintings_sold : ℕ := 8

-- Calculate Michael's total earnings
def total_earnings : ℕ := (large_painting_cost * large_paintings_sold) + (small_painting_cost * small_paintings_sold)

-- Prove: Michael's total earnings are 1140 dollars
theorem michael_total_earnings : total_earnings = 1140 := by
  sorry

end michael_total_earnings_l51_51538


namespace gcd_lcm_product_180_l51_51389

theorem gcd_lcm_product_180 (a b : ℕ) (g l : ℕ) (ha : a > 0) (hb : b > 0) (hg : g > 0) (hl : l > 0) 
  (h₁ : g = gcd a b) (h₂ : l = lcm a b) (h₃ : g * l = 180):
  ∃(n : ℕ), n = 8 :=
by
  sorry

end gcd_lcm_product_180_l51_51389


namespace car_average_speed_l51_51019

theorem car_average_speed :
  let distance_uphill := 100
  let distance_downhill := 50
  let speed_uphill := 30
  let speed_downhill := 80
  let total_distance := distance_uphill + distance_downhill
  let time_uphill := distance_uphill / speed_uphill
  let time_downhill := distance_downhill / speed_downhill
  let total_time := time_uphill + time_downhill
  let average_speed := total_distance / total_time
  average_speed = 37.92 := by
  sorry

end car_average_speed_l51_51019


namespace equation_equiv_product_zero_l51_51678

theorem equation_equiv_product_zero (a b x y : ℝ) :
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1) →
  ∃ (m n p : ℤ), (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5 ∧ m * n * p = 0 :=
by
  intros h
  sorry

end equation_equiv_product_zero_l51_51678


namespace max_discount_rate_l51_51246

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l51_51246


namespace quadratic_inequality_l51_51425

theorem quadratic_inequality (a x : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) := 
sorry

end quadratic_inequality_l51_51425


namespace circle_equation_l51_51249

theorem circle_equation :
  ∃ x y : ℝ, x = 2 ∧ y = 0 ∧ ∀ (p q : ℝ), ((p - x)^2 + q^2 = 4) ↔ (p^2 + q^2 - 4 * p = 0) :=
sorry

end circle_equation_l51_51249


namespace triangle_area_proof_l51_51093

noncomputable def segment_squared (a b : ℝ) : ℝ := a ^ 2 - b ^ 2

noncomputable def triangle_conditions (a b c : ℝ): Prop :=
  segment_squared b a = a ^ 2 - c ^ 2

noncomputable def area_triangle_OLK (r a b c : ℝ) (cond : triangle_conditions a b c): ℝ :=
  (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3))

theorem triangle_area_proof (r a b c : ℝ) (cond : triangle_conditions a b c) :
  area_triangle_OLK r a b c cond = (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3)) :=
sorry

end triangle_area_proof_l51_51093


namespace find_number_l51_51318

theorem find_number (x : ℝ) : 0.5 * 56 = 0.3 * x + 13 ↔ x = 50 :=
by
  -- Proof would go here
  sorry

end find_number_l51_51318


namespace simon_gift_bags_l51_51285

theorem simon_gift_bags (rate_per_day : ℕ) (days : ℕ) (total_bags : ℕ) :
  rate_per_day = 42 → days = 13 → total_bags = rate_per_day * days → total_bags = 546 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end simon_gift_bags_l51_51285


namespace ratio_Raphael_to_Manny_l51_51204

-- Define the pieces of lasagna each person will eat
def Manny_pieces : ℕ := 1
def Kai_pieces : ℕ := 2
def Lisa_pieces : ℕ := 2
def Aaron_pieces : ℕ := 0
def Total_pieces : ℕ := 6

-- Calculate the remaining pieces for Raphael
def Raphael_pieces : ℕ := Total_pieces - (Manny_pieces + Kai_pieces + Lisa_pieces + Aaron_pieces)

-- Prove that the ratio of Raphael's pieces to Manny's pieces is 1:1
theorem ratio_Raphael_to_Manny : Raphael_pieces = Manny_pieces :=
by
  -- Provide the actual proof logic, but currently leaving it as a placeholder
  sorry

end ratio_Raphael_to_Manny_l51_51204


namespace adam_coins_value_l51_51652

theorem adam_coins_value (num_coins : ℕ) (subset_value: ℕ) (subset_num: ℕ) (total_value: ℕ)
  (h1 : num_coins = 20)
  (h2 : subset_value = 16)
  (h3 : subset_num = 4)
  (h4 : total_value = num_coins * (subset_value / subset_num)) :
  total_value = 80 := 
by
  sorry

end adam_coins_value_l51_51652


namespace transformed_equation_sum_l51_51292

theorem transformed_equation_sum (a b : ℝ) (h_eqn : ∀ x : ℝ, x^2 - 6 * x - 5 = 0 ↔ (x + a)^2 = b) :
  a + b = 11 :=
sorry

end transformed_equation_sum_l51_51292


namespace quadrant_of_angle_l51_51684

variable (α : ℝ)

theorem quadrant_of_angle (h₁ : Real.sin α < 0) (h₂ : Real.tan α > 0) : 
  3 * (π / 2) < α ∧ α < 2 * π ∨ π < α ∧ α < 3 * (π / 2) :=
by
  sorry

end quadrant_of_angle_l51_51684


namespace total_laps_jogged_l51_51082

-- Defining the conditions
def jogged_PE_class : ℝ := 1.12
def jogged_track_practice : ℝ := 2.12

-- Statement to prove
theorem total_laps_jogged : jogged_PE_class + jogged_track_practice = 3.24 := by
  -- Proof would go here
  sorry

end total_laps_jogged_l51_51082


namespace geometric_series_properties_l51_51364

theorem geometric_series_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  a 3 = 3 ∧ a 10 = 384 → 
  q = 2 ∧ 
  (∀ n, a n = (3 / 4) * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (3 / 4) * (2 ^ n - 1)) :=
by
  intro h
  -- Proofs will go here, if necessary.
  sorry

end geometric_series_properties_l51_51364


namespace cos_double_angle_l51_51724

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1 / 5) : Real.cos (2 * α) = 23 / 25 :=
sorry

end cos_double_angle_l51_51724


namespace proof_problem_l51_51737

variable (a b c x y z : ℝ)

theorem proof_problem
  (h1 : x + y - z = a - b)
  (h2 : x - y + z = b - c)
  (h3 : - x + y + z = c - a) : 
  x + y + z = 0 := by
  sorry

end proof_problem_l51_51737


namespace find_f_one_l51_51756

noncomputable def f_inv (x : ℝ) : ℝ := 2^(x + 1)

theorem find_f_one : ∃ f : ℝ → ℝ, (∀ y, f (f_inv y) = y) ∧ f 1 = -1 :=
by
  sorry

end find_f_one_l51_51756


namespace sum_of_ratios_is_3_or_neg3_l51_51716

theorem sum_of_ratios_is_3_or_neg3 
  (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a / b + b / c + c / a : ℚ).den = 1 ) 
  (h5 : (b / a + c / b + a / c : ℚ).den = 1) :
  (a / b + b / c + c / a = 3 ∨ a / b + b / c + c / a = -3) ∧ 
  (b / a + c / b + a / c = 3 ∨ b / a + c / b + a / c = -3) := 
sorry

end sum_of_ratios_is_3_or_neg3_l51_51716


namespace correct_transformation_l51_51616

variable (a b : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : b ≠ 0)
variable (h₂ : a / 2 = b / 3)

theorem correct_transformation : 3 / b = 2 / a :=
by
  sorry

end correct_transformation_l51_51616


namespace correct_option_is_B_l51_51626

theorem correct_option_is_B (a : ℝ) : 
  (¬ (-2 * a^2 * b)^3 = -6 * a^6 * b^3) ∧
  (a^7 / a = a^6) ∧
  (¬ (a + 1)^2 = a^2 + 1) ∧
  (¬ 2 * a + 3 * b = 5 * a * b) :=
by
  sorry

end correct_option_is_B_l51_51626


namespace common_ratio_of_gp_l51_51447

theorem common_ratio_of_gp (a r : ℝ) (h1 : r ≠ 1) 
  (h2 : (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 343) : r = 6 := 
by
  sorry

end common_ratio_of_gp_l51_51447


namespace least_number_of_faces_l51_51080

def faces_triangular_prism : ℕ := 5
def faces_quadrangular_prism : ℕ := 6
def faces_triangular_pyramid : ℕ := 4
def faces_quadrangular_pyramid : ℕ := 5
def faces_truncated_quadrangular_pyramid : ℕ := 6

theorem least_number_of_faces : faces_triangular_pyramid < faces_triangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_pyramid ∧
                                faces_triangular_pyramid < faces_truncated_quadrangular_pyramid 
                                :=
by {
  sorry
}

end least_number_of_faces_l51_51080


namespace prime_difference_fourth_powers_is_not_prime_l51_51613

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_difference_fourth_powers_is_not_prime (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) : 
  ¬ is_prime (p^4 - q^4) :=
sorry

end prime_difference_fourth_powers_is_not_prime_l51_51613


namespace actual_diameter_of_tissue_is_0_03_mm_l51_51191

-- Defining necessary conditions
def magnified_diameter_meters : ℝ := 0.15
def magnification_factor : ℝ := 5000
def meters_to_millimeters : ℝ := 1000

-- Prove that the actual diameter of the tissue is 0.03 millimeters
theorem actual_diameter_of_tissue_is_0_03_mm :
  (magnified_diameter_meters * meters_to_millimeters) / magnification_factor = 0.03 := 
  sorry

end actual_diameter_of_tissue_is_0_03_mm_l51_51191


namespace half_angle_in_quadrant_l51_51645

theorem half_angle_in_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 / 2) * Real.pi) :
  (π / 2 < α / 2 ∧ α / 2 < π) ∨ (3 * π / 2 < α / 2 ∧ α / 2 < 2 * π) :=
sorry

end half_angle_in_quadrant_l51_51645


namespace cost_of_each_shirt_l51_51801

theorem cost_of_each_shirt (initial_money : ℕ) (cost_pants : ℕ) (money_left : ℕ) (shirt_cost : ℕ)
  (h1 : initial_money = 109)
  (h2 : cost_pants = 13)
  (h3 : money_left = 74)
  (h4 : initial_money - (2 * shirt_cost + cost_pants) = money_left) :
  shirt_cost = 11 :=
by
  sorry

end cost_of_each_shirt_l51_51801


namespace not_prime_5n_plus_3_l51_51157

theorem not_prime_5n_plus_3 (n a b : ℕ) (hn_pos : n > 0) (ha_pos : a > 0) (hb_pos : b > 0)
  (ha : 2 * n + 1 = a^2) (hb : 3 * n + 1 = b^2) : ¬Prime (5 * n + 3) :=
by
  sorry

end not_prime_5n_plus_3_l51_51157


namespace entree_cost_l51_51260

theorem entree_cost (E : ℝ) :
  let appetizer := 9
  let dessert := 11
  let tip_rate := 0.30
  let total_cost_with_tip := 78
  let total_cost_before_tip := appetizer + 2 * E + dessert
  total_cost_with_tip = total_cost_before_tip + (total_cost_before_tip * tip_rate) →
  E = 20 :=
by
  intros appetizer dessert tip_rate total_cost_with_tip total_cost_before_tip h
  sorry

end entree_cost_l51_51260


namespace man_l51_51945

theorem man's_speed_upstream (v : ℝ) (downstream_speed : ℝ) (stream_speed : ℝ) :
  downstream_speed = v + stream_speed → stream_speed = 1 → downstream_speed = 10 → v - stream_speed = 8 :=
by
  intros h1 h2 h3
  sorry

end man_l51_51945


namespace maximum_ab_minimum_frac_minimum_exp_l51_51615

variable {a b : ℝ}

theorem maximum_ab (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  ab <= 1/8 :=
sorry

theorem minimum_frac (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2/a + 1/b >= 8 :=
sorry

theorem minimum_exp (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2^a + 4^b >= 2 * Real.sqrt 2 :=
sorry

end maximum_ab_minimum_frac_minimum_exp_l51_51615


namespace circle_radius_l51_51328

theorem circle_radius (x y : ℝ) : x^2 + y^2 - 2*y = 0 → ∃ r : ℝ, r = 1 :=
by
  sorry

end circle_radius_l51_51328


namespace girls_friends_count_l51_51478

variable (days_in_week : ℕ)
variable (total_friends : ℕ)
variable (boys : ℕ)

axiom H1 : days_in_week = 7
axiom H2 : total_friends = 2 * days_in_week
axiom H3 : boys = 11

theorem girls_friends_count : total_friends - boys = 3 :=
by sorry

end girls_friends_count_l51_51478


namespace airplane_seats_theorem_l51_51884

def airplane_seats_proof : Prop :=
  ∀ (s : ℝ),
  (∃ (first_class business_class economy premium_economy : ℝ),
    first_class = 30 ∧
    business_class = 0.4 * s ∧
    economy = 0.6 * s ∧
    premium_economy = s - (first_class + business_class + economy)) →
  s = 150

theorem airplane_seats_theorem : airplane_seats_proof :=
sorry

end airplane_seats_theorem_l51_51884


namespace nth_odd_and_sum_first_n_odds_l51_51533

noncomputable def nth_odd (n : ℕ) : ℕ := 2 * n - 1

noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n ^ 2

theorem nth_odd_and_sum_first_n_odds :
  nth_odd 100 = 199 ∧ sum_first_n_odds 100 = 10000 :=
by
  sorry

end nth_odd_and_sum_first_n_odds_l51_51533


namespace donny_spending_l51_51353

theorem donny_spending :
  (15 + 28 + 13) / 2 = 28 :=
by
  sorry

end donny_spending_l51_51353


namespace arrange_in_ascending_order_l51_51472

open Real

noncomputable def a := log 3 / log (1/2)
noncomputable def b := log 5 / log (1/2)
noncomputable def c := log (1/2) / log (1/3)

theorem arrange_in_ascending_order : b < a ∧ a < c :=
by
  sorry

end arrange_in_ascending_order_l51_51472


namespace swimming_class_attendance_l51_51586

def total_students : ℕ := 1000
def chess_ratio : ℝ := 0.25
def swimming_ratio : ℝ := 0.50

def chess_students := chess_ratio * total_students
def swimming_students := swimming_ratio * chess_students

theorem swimming_class_attendance :
  swimming_students = 125 :=
by
  sorry

end swimming_class_attendance_l51_51586


namespace g_at_5_l51_51100

variable (g : ℝ → ℝ)

-- Define the condition on g
def functional_condition : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1

-- The statement proven should be g(5) = 8 given functional_condition
theorem g_at_5 (h : functional_condition g) : g 5 = 8 := by
  sorry

end g_at_5_l51_51100


namespace abc_plus_2p_zero_l51_51941

variable (a b c p : ℝ)

-- Define the conditions
def cond1 : Prop := a + 2 / b = p
def cond2 : Prop := b + 2 / c = p
def cond3 : Prop := c + 2 / a = p
def nonzero_and_distinct : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main statement we want to prove
theorem abc_plus_2p_zero (h1 : cond1 a b p) (h2 : cond2 b c p) (h3 : cond3 c a p) (h4 : nonzero_and_distinct a b c) : 
  a * b * c + 2 * p = 0 := 
by 
  sorry

end abc_plus_2p_zero_l51_51941


namespace solution_to_inequality_l51_51198

theorem solution_to_inequality :
  { x : ℝ | ((x^2 - 1) / (x - 4)^2) ≥ 0 } = { x : ℝ | x ≤ -1 ∨ (1 ≤ x ∧ x < 4) ∨ x > 4 } := 
sorry

end solution_to_inequality_l51_51198


namespace fraction_addition_l51_51687

theorem fraction_addition :
  (2 / 5 : ℚ) + (3 / 8) = 31 / 40 :=
sorry

end fraction_addition_l51_51687


namespace largest_real_solution_sum_l51_51742

theorem largest_real_solution_sum (d e f : ℕ) (x : ℝ) (h : d = 13 ∧ e = 61 ∧ f = 0) : 
  (∃ d e f : ℕ, d + e + f = 74) ↔ 
  (n : ℝ) * n = (x - d)^2 ∧ 
  (∀ x : ℝ, 
    (4 / (x - 4)) + (6 / (x - 6)) + (18 / (x - 18)) + (20 / (x - 20)) = x^2 - 13 * x - 6 → 
    n = x) :=
sorry

end largest_real_solution_sum_l51_51742


namespace tom_won_whack_a_mole_l51_51451

variable (W : ℕ)  -- let W be the number of tickets Tom won playing 'whack a mole'
variable (won_skee_ball : ℕ := 25)  -- Tom won 25 tickets playing 'skee ball'
variable (spent_on_hat : ℕ := 7)  -- Tom spent 7 tickets on a hat
variable (tickets_left : ℕ := 50)  -- Tom has 50 tickets left

theorem tom_won_whack_a_mole :
  W + 25 + 50 = 57 →
  W = 7 :=
by
  sorry  -- proof goes here

end tom_won_whack_a_mole_l51_51451


namespace number_to_add_l51_51394

theorem number_to_add (a m : ℕ) (h₁ : a = 7844213) (h₂ : m = 549) :
  ∃ n, (a + n) % m = 0 ∧ n = m - (a % m) :=
by
  sorry

end number_to_add_l51_51394


namespace man_finishes_work_in_100_days_l51_51520

variable (M W : ℝ)
variable (H1 : 10 * M * 6 + 15 * W * 6 = 1)
variable (H2 : W * 225 = 1)

theorem man_finishes_work_in_100_days (M W : ℝ) (H1 : 10 * M * 6 + 15 * W * 6 = 1) (H2 : W * 225 = 1) : M = 1 / 100 :=
by
  sorry

end man_finishes_work_in_100_days_l51_51520


namespace imaginary_part_of_z_l51_51688

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 :=
sorry

end imaginary_part_of_z_l51_51688


namespace combined_area_of_walls_l51_51091

theorem combined_area_of_walls (A : ℕ) 
  (h1: ∃ (A : ℕ), A ≥ 0)
  (h2 : (A - 2 * 40 - 40 = 180)) :
  A = 300 := 
sorry

end combined_area_of_walls_l51_51091


namespace target_runs_l51_51757

theorem target_runs (r1 r2 : ℝ) (o1 o2 : ℕ) (target : ℝ) :
  r1 = 3.6 ∧ o1 = 10 ∧ r2 = 6.15 ∧ o2 = 40 → target = (r1 * o1) + (r2 * o2) := by
  sorry

end target_runs_l51_51757


namespace solution_unique_2014_l51_51208

theorem solution_unique_2014 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (2 * x - 2 * y + 1 / z = 1 / 2014) ∧
  (2 * y - 2 * z + 1 / x = 1 / 2014) ∧
  (2 * z - 2 * x + 1 / y = 1 / 2014) →
  x = 2014 ∧ y = 2014 ∧ z = 2014 :=
by
  sorry

end solution_unique_2014_l51_51208


namespace convert_110110001_to_base4_l51_51723

def binary_to_base4_conversion (b : ℕ) : ℕ :=
  -- assuming b is the binary representation of the number to be converted
  1 * 4^4 + 3 * 4^3 + 2 * 4^2 + 0 * 4^1 + 1 * 4^0

theorem convert_110110001_to_base4 : binary_to_base4_conversion 110110001 = 13201 :=
  sorry

end convert_110110001_to_base4_l51_51723


namespace kelly_total_snacks_l51_51030

theorem kelly_total_snacks (peanuts raisins : ℝ) (h₁ : peanuts = 0.1) (h₂ : raisins = 0.4) :
  peanuts + raisins = 0.5 :=
by
  simp [h₁, h₂]
  sorry

end kelly_total_snacks_l51_51030


namespace participated_in_both_l51_51375

-- Define the conditions
def total_students := 40
def math_competition := 31
def physics_competition := 20
def not_participating := 8

-- Define number of students participated in both competitions
def both_competitions := 59 - total_students

-- Theorem statement
theorem participated_in_both : both_competitions = 19 := 
sorry

end participated_in_both_l51_51375


namespace ratio_of_a_to_b_l51_51346

theorem ratio_of_a_to_b 
  (b c a : ℝ)
  (h1 : b / c = 1 / 5) 
  (h2 : a / c = 1 / 7.5) : 
  a / b = 2 / 3 :=
by
  sorry

end ratio_of_a_to_b_l51_51346


namespace find_gamma_l51_51078

variable (γ δ : ℝ)

def directly_proportional (γ δ : ℝ) : Prop := ∃ c : ℝ, γ = c * δ

theorem find_gamma (h1 : directly_proportional γ δ) (h2 : γ = 5) (h3 : δ = -10) : δ = 25 → γ = -25 / 2 := by
  sorry

end find_gamma_l51_51078


namespace total_cups_of_liquid_drunk_l51_51984

-- Definitions for the problem conditions
def elijah_pints : ℝ := 8.5
def emilio_pints : ℝ := 9.5
def cups_per_pint : ℝ := 2
def elijah_cups : ℝ := elijah_pints * cups_per_pint
def emilio_cups : ℝ := emilio_pints * cups_per_pint
def total_cups : ℝ := elijah_cups + emilio_cups

-- Theorem to prove the required equality
theorem total_cups_of_liquid_drunk : total_cups = 36 :=
by
  sorry

end total_cups_of_liquid_drunk_l51_51984


namespace temp_on_Monday_l51_51258

variable (M T W Th F : ℤ)

-- Given conditions
axiom sum_MTWT : M + T + W + Th = 192
axiom sum_TWTF : T + W + Th + F = 184
axiom temp_F : F = 34
axiom exists_day_temp_42 : ∃ (day : String), 
  (day = "Monday" ∨ day = "Tuesday" ∨ day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday") ∧
  (if day = "Monday" then M else if day = "Tuesday" then T else if day = "Wednesday" then W else if day = "Thursday" then Th else F) = 42

-- Prove temperature of Monday is 42
theorem temp_on_Monday : M = 42 := 
by
  sorry

end temp_on_Monday_l51_51258


namespace value_of_a_plus_b_l51_51482

-- Define the given nested fraction expression
def nested_expr := 1 + 1 / (1 + 1 / (1 + 1))

-- Define the simplified form of the expression
def simplified_form : ℚ := 13 / 8

-- The greatest common divisor condition
def gcd_condition : ℕ := Nat.gcd 13 8

-- The ultimate theorem to prove
theorem value_of_a_plus_b : 
  nested_expr = simplified_form ∧ gcd_condition = 1 → 13 + 8 = 21 := 
by 
  sorry

end value_of_a_plus_b_l51_51482


namespace simple_interest_for_2_years_l51_51003

noncomputable def calculate_simple_interest (P r t : ℝ) : ℝ :=
  (P * r * t) / 100

theorem simple_interest_for_2_years (CI P r t : ℝ) (hCI : CI = P * (1 + r / 100)^t - P)
  (hCI_value : CI = 615) (r_value : r = 5) (t_value : t = 2) : 
  calculate_simple_interest P r t = 600 :=
by
  sorry

end simple_interest_for_2_years_l51_51003


namespace difference_in_lengths_l51_51491

def speed_of_first_train := 60 -- in km/hr
def time_to_cross_pole_first_train := 3 -- in seconds
def speed_of_second_train := 90 -- in km/hr
def time_to_cross_pole_second_train := 2 -- in seconds

noncomputable def length_of_first_train : ℝ := (speed_of_first_train * (5 / 18)) * time_to_cross_pole_first_train
noncomputable def length_of_second_train : ℝ := (speed_of_second_train * (5 / 18)) * time_to_cross_pole_second_train

theorem difference_in_lengths : abs (length_of_second_train - length_of_first_train) = 0.01 :=
by
  -- The full proof would be placed here.
  sorry

end difference_in_lengths_l51_51491


namespace candy_boxes_system_l51_51107

-- Given conditions and definitions
def sheets_total (x y : ℕ) : Prop := x + y = 35
def sheet_usage (x y : ℕ) : Prop := 20 * x = 30 * y / 2

-- Statement
theorem candy_boxes_system (x y : ℕ) (h1 : sheets_total x y) (h2 : sheet_usage x y) : 
  (x + y = 35) ∧ (20 * x = 30 * y / 2) := 
by
sorry

end candy_boxes_system_l51_51107


namespace magician_assistant_trick_l51_51585

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end magician_assistant_trick_l51_51585


namespace fractional_part_of_students_who_walk_home_l51_51084

def fraction_bus := 1 / 3
def fraction_automobile := 1 / 5
def fraction_bicycle := 1 / 8
def fraction_scooter := 1 / 10

theorem fractional_part_of_students_who_walk_home :
  (1 : ℚ) - (fraction_bus + fraction_automobile + fraction_bicycle + fraction_scooter) = 29 / 120 :=
by
  sorry

end fractional_part_of_students_who_walk_home_l51_51084


namespace area_of_trapezoid_MBCN_l51_51358

variables {AB BC MN : ℝ}
variables {Area_ABCD Area_MBCN : ℝ}
variables {Height : ℝ}

-- Given conditions
def cond1 : Area_ABCD = 40 := sorry
def cond2 : AB = 8 := sorry
def cond3 : BC = 5 := sorry
def cond4 : MN = 2 := sorry
def cond5 : Height = 5 := sorry

-- Define the theorem to be proven
theorem area_of_trapezoid_MBCN : 
  Area_ABCD = AB * BC → MN + BC = 6 → Height = 5 →
  Area_MBCN = (1/2) * (MN + BC) * Height → 
  Area_MBCN = 15 :=
by
  intros h1 h2 h3 h4
  sorry

end area_of_trapezoid_MBCN_l51_51358


namespace maximum_cows_l51_51659

theorem maximum_cows (s c : ℕ) (h1 : 30 * s + 33 * c = 1300) (h2 : c > 2 * s) : c ≤ 30 :=
by
  -- Proof would go here
  sorry

end maximum_cows_l51_51659


namespace compare_abc_l51_51560

noncomputable def a : ℝ := 2^(1/2)
noncomputable def b : ℝ := 3^(1/3)
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > a ∧ a > c :=
by
  sorry

end compare_abc_l51_51560


namespace arithmetic_sequence_30th_term_l51_51244

theorem arithmetic_sequence_30th_term (a1 a2 a3 d a30 : ℤ) 
 (h1 : a1 = 3) (h2 : a2 = 12) (h3 : a3 = 21) 
 (h4 : d = a2 - a1) (h5 : a3 = a1 + 2 * d) 
 (h6 : a30 = a1 + 29 * d) : 
 a30 = 264 :=
by
  sorry

end arithmetic_sequence_30th_term_l51_51244


namespace max_real_root_lt_100_l51_51787

theorem max_real_root_lt_100 (k a b c : ℕ) (r : ℝ)
  (ha : ∃ m : ℕ, a = k^m)
  (hb : ∃ n : ℕ, b = k^n)
  (hc : ∃ l : ℕ, c = k^l)
  (one_real_solution : b^2 = 4 * a * c)
  (r_is_root : ∃ r : ℝ, a * r^2 - b * r + c = 0)
  (r_lt_100 : r < 100) :
  r ≤ 64 := sorry

end max_real_root_lt_100_l51_51787


namespace complex_powers_i_l51_51805

theorem complex_powers_i (i : ℂ) (h : i^2 = -1) :
  (i^123 - i^321 + i^432 = -2 * i + 1) :=
by
  -- sorry to skip the proof
  sorry

end complex_powers_i_l51_51805


namespace find_constants_l51_51641

theorem find_constants (A B C : ℤ) (h1 : 1 = A + B) (h2 : -2 = C) (h3 : 5 = -A) :
  A = -5 ∧ B = 6 ∧ C = -2 :=
by {
  sorry
}

end find_constants_l51_51641


namespace equal_intercepts_l51_51806

theorem equal_intercepts (a : ℝ) (h : ∃ (x y : ℝ), (x = (2 + a) / a ∧ y = 2 + a ∧ x = y)) :
  a = -2 ∨ a = 1 :=
by sorry

end equal_intercepts_l51_51806


namespace selection_schemes_l51_51466

theorem selection_schemes (people : Finset ℕ) (A B C : ℕ) (h_people : people.card = 5) 
(h_A_B_individuals : A ∈ people ∧ B ∈ people) (h_A_B_C_exclusion : A ≠ C ∧ B ≠ C) :
  ∃ (number_of_schemes : ℕ), number_of_schemes = 36 :=
by
  sorry

end selection_schemes_l51_51466


namespace eval_frac_equal_two_l51_51973

noncomputable def eval_frac (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : ℂ :=
  (a^8 + b^8) / (a^2 + b^2)^4

theorem eval_frac_equal_two (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : eval_frac a b h1 h2 h3 = 2 :=
by {
  sorry
}

end eval_frac_equal_two_l51_51973


namespace skillful_hands_wire_cut_l51_51547

theorem skillful_hands_wire_cut :
  ∃ x : ℕ, (1000 = 15 * x) ∧ (1040 = 15 * x) ∧ x = 66 :=
by
  sorry

end skillful_hands_wire_cut_l51_51547


namespace fin_solutions_l51_51774

theorem fin_solutions (u : ℕ) (hu : u > 0) :
  ∃ N : ℕ, ∀ n a b : ℕ, n > N → ¬ (n! = u^a - u^b) :=
sorry

end fin_solutions_l51_51774


namespace number_of_pear_trees_l51_51914

theorem number_of_pear_trees (A P : ℕ) (h1 : A + P = 46)
  (h2 : ∀ (s : Finset (Fin 46)), s.card = 28 → ∃ (i : Fin 46), i ∈ s ∧ i < A)
  (h3 : ∀ (s : Finset (Fin 46)), s.card = 20 → ∃ (i : Fin 46), i ∈ s ∧ A ≤ i) :
  P = 27 :=
by
  sorry

end number_of_pear_trees_l51_51914


namespace joanna_reading_rate_l51_51163

variable (P : ℝ)

theorem joanna_reading_rate (h : 3 * P + 6.5 * P + 6 * P = 248) : P = 16 := by
  sorry

end joanna_reading_rate_l51_51163


namespace max_value_negative_one_l51_51345

theorem max_value_negative_one (f : ℝ → ℝ) (hx : ∀ x, x < 1 → f x ≤ -1) :
  ∀ x, x < 1 → ∃ M, (∀ y, y < 1 → f y ≤ M) ∧ f x = M :=
sorry

end max_value_negative_one_l51_51345


namespace stella_doll_price_l51_51117

theorem stella_doll_price 
  (dolls_count clocks_count glasses_count : ℕ)
  (price_per_clock price_per_glass cost profit : ℕ)
  (D : ℕ)
  (h1 : dolls_count = 3)
  (h2 : clocks_count = 2)
  (h3 : glasses_count = 5)
  (h4 : price_per_clock = 15)
  (h5 : price_per_glass = 4)
  (h6 : cost = 40)
  (h7 : profit = 25)
  (h8 : 3 * D + 2 * price_per_clock + 5 * price_per_glass = cost + profit) :
  D = 5 :=
by
  sorry

end stella_doll_price_l51_51117


namespace polygon_with_15_diagonals_has_7_sides_l51_51324

-- Define the number of diagonals formula for a regular polygon
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement
theorem polygon_with_15_diagonals_has_7_sides :
  ∃ n : ℕ, number_of_diagonals n = 15 ∧ n = 7 :=
by
  sorry

end polygon_with_15_diagonals_has_7_sides_l51_51324


namespace r_squared_sum_l51_51950

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l51_51950


namespace tiffany_mile_fraction_l51_51725

/-- Tiffany's daily running fraction (x) for Wednesday, Thursday, and Friday must be 1/3
    such that both Billy and Tiffany run the same total miles over a week. --/
theorem tiffany_mile_fraction :
  ∃ x : ℚ, (3 * 1 + 1) = 1 + (3 * 2 + 3 * x) → x = 1 / 3 :=
by
  sorry

end tiffany_mile_fraction_l51_51725


namespace distribute_5_cousins_in_4_rooms_l51_51670

theorem distribute_5_cousins_in_4_rooms : 
  let rooms := 4
  let cousins := 5
  ∃ ways : ℕ, ways = 67 ∧ rooms = 4 ∧ cousins = 5 := sorry

end distribute_5_cousins_in_4_rooms_l51_51670


namespace solution_set_of_inequality_l51_51642

theorem solution_set_of_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end solution_set_of_inequality_l51_51642


namespace prob_three_friends_same_group_l51_51072

theorem prob_three_friends_same_group :
  let students := 800
  let groups := 4
  let group_size := students / groups
  let p_same_group := 1 / groups
  p_same_group * p_same_group = 1 / 16 := 
by
  sorry

end prob_three_friends_same_group_l51_51072


namespace find_n_solution_l51_51844

def product_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.prod

theorem find_n_solution : ∃ n : ℕ, n > 0 ∧ n^2 - 17 * n + 56 = product_of_digits n ∧ n = 4 := 
by
  sorry

end find_n_solution_l51_51844


namespace total_beakers_count_l51_51763

variable (total_beakers_with_ions : ℕ) 
variable (drops_per_test : ℕ)
variable (total_drops_used : ℕ) 
variable (beakers_without_ions : ℕ)

theorem total_beakers_count
  (h1 : total_beakers_with_ions = 8)
  (h2 : drops_per_test = 3)
  (h3 : total_drops_used = 45)
  (h4 : beakers_without_ions = 7) : 
  (total_drops_used / drops_per_test) = (total_beakers_with_ions + beakers_without_ions) :=
by
  -- Proof to be filled in
  sorry

end total_beakers_count_l51_51763


namespace smallest_expression_l51_51693

theorem smallest_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  (y / x = 1 / 2) ∧ (y / x < x + y) ∧ (y / x < x * y) ∧ (y / x < x - y) ∧ (y / x < x / y) :=
by
  -- The proof is to be filled by the user
  sorry

end smallest_expression_l51_51693


namespace range_of_a_l51_51562

theorem range_of_a (a : ℝ) : 
  (∀ x, (x ≤ 1 ∨ x ≥ 3) ↔ ((a ≤ x ∧ x ≤ a + 1) → (x ≤ 1 ∨ x ≥ 3))) → 
  (a ≤ 0 ∨ a ≥ 3) :=
by
  sorry

end range_of_a_l51_51562


namespace min_value_fraction_l51_51398

theorem min_value_fraction (a b : ℝ) (h₀ : a > b) (h₁ : a * b = 1) :
  ∃ c, c = (2 * Real.sqrt 2) ∧ (a^2 + b^2) / (a - b) ≥ c :=
by sorry

end min_value_fraction_l51_51398


namespace number_of_teachers_under_40_in_sample_l51_51308

def proportion_teachers_under_40 (total_teachers teachers_under_40 : ℕ) : ℚ :=
  teachers_under_40 / total_teachers

def sample_teachers_under_40 (sample_size : ℕ) (proportion : ℚ) : ℚ :=
  sample_size * proportion

theorem number_of_teachers_under_40_in_sample
(total_teachers teachers_under_40 teachers_40_and_above sample_size : ℕ)
(h_total : total_teachers = 400)
(h_under_40 : teachers_under_40 = 250)
(h_40_and_above : teachers_40_and_above = 150)
(h_sample_size : sample_size = 80)
: sample_teachers_under_40 sample_size 
  (proportion_teachers_under_40 total_teachers teachers_under_40) = 50 := by
sorry

end number_of_teachers_under_40_in_sample_l51_51308


namespace emily_quiz_score_l51_51830

theorem emily_quiz_score :
  ∃ x : ℕ, 94 + 88 + 92 + 85 + 97 + x = 6 * 90 :=
by
  sorry

end emily_quiz_score_l51_51830


namespace find_y_l51_51747

theorem find_y (x y : ℕ) (hx_positive : 0 < x) (hy_positive : 0 < y) (hmod : x % y = 9) (hdiv : (x : ℝ) / (y : ℝ) = 96.25) : y = 36 :=
sorry

end find_y_l51_51747


namespace six_a_seven_eight_b_div_by_45_l51_51050

/-- If the number 6a78b is divisible by 45, then a + b = 6. -/
theorem six_a_seven_eight_b_div_by_45 (a b : ℕ) (h1: 0 ≤ a ∧ a < 10) (h2: 0 ≤ b ∧ b < 10)
  (h3 : (6 * 10^4 + a * 10^3 + 7 * 10^2 + 8 * 10 + b) % 45 = 0) : a + b = 6 := 
by
  sorry

end six_a_seven_eight_b_div_by_45_l51_51050


namespace balls_problem_l51_51009

noncomputable def red_balls_initial := 420
noncomputable def total_balls_initial := 600
noncomputable def percent_red_required := 60 / 100

theorem balls_problem :
  ∃ (x : ℕ), 420 - x = (3 / 5) * (600 - x) :=
by
  sorry

end balls_problem_l51_51009


namespace x_y_square_sum_l51_51557

theorem x_y_square_sum (x y : ℝ) (h1 : x - y = -1) (h2 : x * y = 1 / 2) : x^2 + y^2 = 2 := 
by 
  sorry

end x_y_square_sum_l51_51557


namespace a_values_unique_solution_l51_51850

theorem a_values_unique_solution :
  (∀ a : ℝ, ∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) →
  (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end a_values_unique_solution_l51_51850


namespace even_function_f_D_l51_51915

noncomputable def f_A (x : ℝ) : ℝ := 2 * |x| - 1
def D_f_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

def f_B (x : ℕ) : ℕ := x^2 + x

def f_C (x : ℝ) : ℝ := x ^ 3

noncomputable def f_D (x : ℝ) : ℝ := x^2
def D_f_D := {x : ℝ | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)}

theorem even_function_f_D : 
  ∀ x ∈ D_f_D, f_D (-x) = f_D (x) :=
sorry

end even_function_f_D_l51_51915


namespace books_bought_at_bookstore_l51_51896

-- Define the initial count of books
def initial_books : ℕ := 72

-- Define the number of books received each month from the book club
def books_from_club (months : ℕ) : ℕ := months

-- Number of books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Number of books bought
def books_from_yard_sales : ℕ := 2

-- Number of books donated and sold
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final total count of books
def final_books : ℕ := 81

-- Calculate the number of books acquired and then removed, and prove 
-- the number of books bought at the bookstore halfway through the year
theorem books_bought_at_bookstore (months : ℕ) (b : ℕ) :
  initial_books + books_from_club months + books_from_daughter + books_from_mother + books_from_yard_sales + b - books_donated - books_sold = final_books → b = 5 :=
by sorry

end books_bought_at_bookstore_l51_51896


namespace three_point_one_two_six_as_fraction_l51_51497

theorem three_point_one_two_six_as_fraction : (3126 / 1000 : ℚ) = 1563 / 500 := 
by 
  sorry

end three_point_one_two_six_as_fraction_l51_51497


namespace math_problem_l51_51376

theorem math_problem (a b c d m : ℝ) (h1 : a = -b) (h2 : a ≠ 0) (h3 : c * d = 1)
  (h4 : m = -1 ∨ m = 3) : (a + b) * (c / d) + m * c * d + (b / a) = 2 ∨ (a + b) * (c / d) + m * c * d + (b / a) = -2 :=
by
  sorry

end math_problem_l51_51376


namespace angle_A_is_30_degrees_l51_51073

theorem angle_A_is_30_degrees {A : ℝ} (hA_acute : 0 < A ∧ A < π / 2) (hA_sin : Real.sin A = 1 / 2) : A = π / 6 :=
sorry

end angle_A_is_30_degrees_l51_51073


namespace squirrels_in_tree_l51_51356

-- Definitions based on the conditions
def nuts : Nat := 2
def squirrels : Nat := nuts + 2

-- Theorem stating the main proof problem
theorem squirrels_in_tree : squirrels = 4 := by
  -- Proof steps would go here, but we're adding sorry to skip them
  sorry

end squirrels_in_tree_l51_51356


namespace initial_students_count_l51_51017

-- Definitions based on conditions
def initial_average_age (T : ℕ) (n : ℕ) : Prop := T = 14 * n
def new_average_age_after_adding (T : ℕ) (n : ℕ) : Prop := (T + 5 * 17) / (n + 5) = 15

-- Main proposition stating the problem
theorem initial_students_count (n : ℕ) (T : ℕ) 
  (h1 : initial_average_age T n)
  (h2 : new_average_age_after_adding T n) :
  n = 10 :=
by
  sorry

end initial_students_count_l51_51017


namespace target_heart_rate_of_30_year_old_l51_51664

variable (age : ℕ) (T M : ℕ)

def maximum_heart_rate (age : ℕ) : ℕ :=
  210 - age

def target_heart_rate (M : ℕ) : ℕ :=
  (75 * M) / 100

theorem target_heart_rate_of_30_year_old :
  maximum_heart_rate 30 = 180 →
  target_heart_rate (maximum_heart_rate 30) = 135 :=
by
  intros h1
  sorry

end target_heart_rate_of_30_year_old_l51_51664


namespace circle_area_l51_51166

open Real

noncomputable def radius_square (x : ℝ) (DE : ℝ) (EF : ℝ) : ℝ :=
  let DE_square := DE^2
  let r_square_1 := x^2 + DE_square
  let product_DE_EF := DE * EF
  let r_square_2 := product_DE_EF + x^2
  r_square_2

theorem circle_area (x : ℝ) (h1 : OE = x) (h2 : DE = 8) (h3 : EF = 4) :
  π * radius_square x 8 4 = 96 * π :=
by
  sorry

end circle_area_l51_51166


namespace max_f_and_sin_alpha_l51_51110

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x

theorem max_f_and_sin_alpha :
  (∀ x : ℝ, f x ≤ Real.sqrt 5) ∧ (∃ α : ℝ, (α + Real.arccos (1 / Real.sqrt 5) = π / 2 + 2 * π * some_integer) ∧ (f α = Real.sqrt 5) ∧ (Real.sin α = 1 / Real.sqrt 5)) :=
by
  sorry

end max_f_and_sin_alpha_l51_51110


namespace sufficient_but_not_necessary_condition_l51_51717

theorem sufficient_but_not_necessary_condition (a b : ℝ) : (b ≥ 0 → a^2 + b ≥ 0) ∧ ¬(∀ a b, a^2 + b ≥ 0 → b ≥ 0) := by
  sorry

end sufficient_but_not_necessary_condition_l51_51717


namespace arithmetic_sequence_problem_l51_51405

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)  -- the arithmetic sequence
  (S : ℕ → ℤ)  -- the sum of the first n terms
  (m : ℕ)      -- the m in question
  (h1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h2 : S (2 * m - 1) = 18) :
  m = 5 := 
sorry

end arithmetic_sequence_problem_l51_51405


namespace discount_percentage_l51_51125

theorem discount_percentage (wm_cost dryer_cost after_discount before_discount discount_amount : ℝ)
    (h0 : wm_cost = 100) 
    (h1 : dryer_cost = wm_cost - 30) 
    (h2 : after_discount = 153) 
    (h3 : before_discount = wm_cost + dryer_cost) 
    (h4 : discount_amount = before_discount - after_discount) 
    (h5 : (discount_amount / before_discount) * 100 = 10) : 
    True := sorry

end discount_percentage_l51_51125


namespace find_integer_solutions_l51_51899

theorem find_integer_solutions :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    (a * b - 2 * c * d = 3) ∧ (a * c + b * d = 1) } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end find_integer_solutions_l51_51899


namespace jaxon_toys_l51_51786

-- Definitions as per the conditions
def toys_jaxon : ℕ := sorry
def toys_gabriel : ℕ := 2 * toys_jaxon
def toys_jerry : ℕ := 2 * toys_jaxon + 8
def total_toys : ℕ := toys_jaxon + toys_gabriel + toys_jerry

-- Theorem to prove
theorem jaxon_toys : total_toys = 83 → toys_jaxon = 15 := sorry

end jaxon_toys_l51_51786


namespace num_individuals_eliminated_l51_51170

theorem num_individuals_eliminated (pop_size : ℕ) (sample_size : ℕ) :
  (pop_size % sample_size) = 2 :=
by
  -- Given conditions
  let pop_size := 1252
  let sample_size := 50
  -- Proof skipped
  sorry

end num_individuals_eliminated_l51_51170


namespace total_number_of_bills_received_l51_51851

open Nat

-- Definitions based on the conditions:
def total_withdrawal_amount : ℕ := 600
def bill_denomination : ℕ := 20

-- Mathematically equivalent proof problem
theorem total_number_of_bills_received : (total_withdrawal_amount / bill_denomination) = 30 := 
by
  sorry

end total_number_of_bills_received_l51_51851


namespace definite_integral_ln_squared_l51_51814

noncomputable def integralFun : ℝ → ℝ := λ x => x * (Real.log x) ^ 2

theorem definite_integral_ln_squared (f : ℝ → ℝ) (a b : ℝ):
  (f = integralFun) → 
  (a = 1) → 
  (b = 2) → 
  ∫ x in a..b, f x = 2 * (Real.log 2) ^ 2 - 2 * Real.log 2 + 3 / 4 :=
by
  intros hfa hao hbo
  rw [hfa, hao, hbo]
  sorry

end definite_integral_ln_squared_l51_51814


namespace opposite_of_83_is_84_l51_51822

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l51_51822


namespace count_rhombuses_in_large_triangle_l51_51493

-- Definitions based on conditions
def large_triangle_side_length : ℕ := 10
def small_triangle_side_length : ℕ := 1
def small_triangle_count : ℕ := 100
def rhombuses_of_8_triangles := 84

-- Problem statement in Lean 4
theorem count_rhombuses_in_large_triangle :
  ∀ (large_side small_side small_count : ℕ),
  large_side = large_triangle_side_length →
  small_side = small_triangle_side_length →
  small_count = small_triangle_count →
  (∃ (rhombus_count : ℕ), rhombus_count = rhombuses_of_8_triangles) :=
by
  intros large_side small_side small_count h_large h_small h_count
  use 84
  sorry

end count_rhombuses_in_large_triangle_l51_51493


namespace meal_cost_before_tax_and_tip_l51_51925

theorem meal_cost_before_tax_and_tip (total_expenditure : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (base_meal_cost : ℝ):
  total_expenditure = 35.20 →
  tax_rate = 0.08 →
  tip_rate = 0.18 →
  base_meal_cost * (1 + tax_rate + tip_rate) = total_expenditure →
  base_meal_cost = 28 :=
by
  intros h_total h_tax h_tip h_eq
  sorry

end meal_cost_before_tax_and_tip_l51_51925


namespace seeds_per_bed_l51_51561

theorem seeds_per_bed (total_seeds : ℕ) (flower_beds : ℕ) (h1 : total_seeds = 60) (h2 : flower_beds = 6) : total_seeds / flower_beds = 10 := by
  sorry

end seeds_per_bed_l51_51561


namespace black_lambs_correct_l51_51401

-- Define the total number of lambs
def total_lambs : ℕ := 6048

-- Define the number of white lambs
def white_lambs : ℕ := 193

-- Define the number of black lambs
def black_lambs : ℕ := total_lambs - white_lambs

-- The goal is to prove that the number of black lambs is 5855
theorem black_lambs_correct : black_lambs = 5855 := by
  sorry

end black_lambs_correct_l51_51401


namespace relay_team_order_count_l51_51578

def num_ways_to_order_relay (total_members : Nat) (jordan_lap : Nat) : Nat :=
  if jordan_lap = total_members then (total_members - 1).factorial else 0

theorem relay_team_order_count : num_ways_to_order_relay 5 5 = 24 :=
by
  -- the proof would go here
  sorry

end relay_team_order_count_l51_51578


namespace carla_bought_marbles_l51_51342

def starting_marbles : ℕ := 2289
def total_marbles : ℝ := 2778.0

theorem carla_bought_marbles : (total_marbles - starting_marbles) = 489 := 
by
  sorry

end carla_bought_marbles_l51_51342


namespace unique_triple_solution_l51_51367

theorem unique_triple_solution {x y z : ℤ} (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (H1 : x ∣ y * z - 1) (H2 : y ∣ z * x - 1) (H3 : z ∣ x * y - 1) :
  (x, y, z) = (5, 3, 2) :=
sorry

end unique_triple_solution_l51_51367


namespace complement_of_M_l51_51409

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}

theorem complement_of_M :
  (U \ M) = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end complement_of_M_l51_51409


namespace inverse_proportion_quadrants_l51_51361

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  ∀ (x y : ℝ), y = k^2 / x → (x > 0 → y > 0) ∧ (x < 0 → y < 0) :=
by
  sorry

end inverse_proportion_quadrants_l51_51361


namespace ordered_pair_for_quadratic_with_same_roots_l51_51253

theorem ordered_pair_for_quadratic_with_same_roots (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ (x = 7 ∨ x = 1)) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = 7 ∨ x = 1)) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end ordered_pair_for_quadratic_with_same_roots_l51_51253


namespace ram_krish_task_completion_l51_51702

theorem ram_krish_task_completion
  (ram_days : ℝ)
  (krish_efficiency_factor : ℝ)
  (task_time : ℝ) 
  (H1 : krish_efficiency_factor = 2)
  (H2 : ram_days = 27) 
  (H3 : task_time = 9) :
  (1 / task_time) = (1 / ram_days + 1 / (ram_days / krish_efficiency_factor)) := 
sorry

end ram_krish_task_completion_l51_51702


namespace tangent_position_is_six_l51_51456

def clock_radius : ℝ := 30
def disk_radius : ℝ := 15
def initial_tangent_position := 12
def final_tangent_position := 6

theorem tangent_position_is_six :
  (∃ (clock_radius disk_radius : ℝ), clock_radius = 30 ∧ disk_radius = 15) →
  (initial_tangent_position = 12) →
  (final_tangent_position = 6) :=
by
  intros h1 h2
  sorry

end tangent_position_is_six_l51_51456


namespace part1_part2_l51_51351

theorem part1 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 2) : a^2 + b^2 = 21 :=
  sorry

theorem part2 (a b : ℝ) (h1 : a + b = 10) (h2 : a^2 + b^2 = 50^2) : a * b = -1200 :=
  sorry

end part1_part2_l51_51351


namespace parabola_directrix_l51_51106

theorem parabola_directrix (p : ℝ) (h_focus : ∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x + 3*y - 4 = 0) : 
  ∀ x y : ℝ, y^2 = 2*p*x → x = -p/2 := 
sorry

end parabola_directrix_l51_51106


namespace selection_count_l51_51677

theorem selection_count :
  (Nat.choose 6 3) * (Nat.choose 5 2) = 200 := 
sorry

end selection_count_l51_51677


namespace superdomino_probability_l51_51901

-- Definitions based on conditions
def is_superdomino (a b : ℕ) : Prop := 0 ≤ a ∧ a ≤ 12 ∧ 0 ≤ b ∧ b ≤ 12
def is_superdouble (a b : ℕ) : Prop := a = b
def total_superdomino_count : ℕ := 13 * 13
def superdouble_count : ℕ := 13

-- Proof statement
theorem superdomino_probability : (superdouble_count : ℚ) / total_superdomino_count = 13 / 169 :=
by
  sorry

end superdomino_probability_l51_51901


namespace trapezoid_smallest_angle_l51_51799

theorem trapezoid_smallest_angle (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : 2 * a + 3 * d = 180) : 
  a = 20 :=
by
  sorry

end trapezoid_smallest_angle_l51_51799


namespace ratio_of_roots_ratio_l51_51947

noncomputable def sum_roots_first_eq (a b c : ℝ) := b / a
noncomputable def product_roots_first_eq (a b c : ℝ) := c / a
noncomputable def sum_roots_second_eq (a b c : ℝ) := a / c
noncomputable def product_roots_second_eq (a b c : ℝ) := b / c

theorem ratio_of_roots_ratio (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (h3 : (b ^ 2 - 4 * a * c) > 0)
  (h4 : (a ^ 2 - 4 * c * b) > 0)
  (h5 : sum_roots_first_eq a b c ≥ 0)
  (h6 : product_roots_first_eq a b c = 9 * sum_roots_second_eq a b c) :
  sum_roots_first_eq a b c / product_roots_second_eq a b c = -3 :=
sorry

end ratio_of_roots_ratio_l51_51947


namespace find_x_proportionally_l51_51323

theorem find_x_proportionally (k m x z : ℝ) (h1 : ∀ y, x = k * y^2) (h2 : ∀ z, y = m / (Real.sqrt z)) (h3 : x = 7 ∧ z = 16) :
  ∃ x, x = 7 / 9 := by
  sorry

end find_x_proportionally_l51_51323


namespace sum_A_B_equals_1_l51_51999

-- Definitions for the digits and the properties defined in conditions
variables (A B C D : ℕ)
variable (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (h_digit_bounds : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
noncomputable def ABCD := 1000 * A + 100 * B + 10 * C + D
axiom h_mult : ABCD * 2 = ABCD * 10

theorem sum_A_B_equals_1 : A + B = 1 :=
by
  sorry

end sum_A_B_equals_1_l51_51999


namespace bags_needed_l51_51176

theorem bags_needed (expected_people extra_people extravagant_bags average_bags : ℕ) 
    (h1 : expected_people = 50) 
    (h2 : extra_people = 40) 
    (h3 : extravagant_bags = 10) 
    (h4 : average_bags = 20) : 
    (expected_people + extra_people - (extravagant_bags + average_bags) = 60) :=
by {
  sorry
}

end bags_needed_l51_51176


namespace number_of_slices_l51_51706

theorem number_of_slices 
  (pepperoni ham sausage total_meat pieces_per_slice : ℕ)
  (h1 : pepperoni = 30)
  (h2 : ham = 2 * pepperoni)
  (h3 : sausage = pepperoni + 12)
  (h4 : total_meat = pepperoni + ham + sausage)
  (h5 : pieces_per_slice = 22) :
  total_meat / pieces_per_slice = 6 :=
by
  sorry

end number_of_slices_l51_51706


namespace refrigerator_volume_unit_l51_51199

theorem refrigerator_volume_unit (V : ℝ) (u : String) : 
  V = 150 → (u = "Liters" ∨ u = "Milliliters" ∨ u = "Cubic meters") → 
  u = "Liters" :=
by
  intro hV hu
  sorry

end refrigerator_volume_unit_l51_51199


namespace vendelin_pastels_l51_51867

theorem vendelin_pastels (M V W : ℕ) (h1 : M = 5) (h2 : V < 5) (h3 : W = M + V) (h4 : M + V + W = 7 * V) : W = 7 := 
sorry

end vendelin_pastels_l51_51867


namespace problem1_problem2_l51_51254

-- Problem 1 Lean Statement
theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 6) (h2 : 9 ^ n = 2) : 3 ^ (m - 2 * n) = 3 :=
by
  sorry

-- Problem 2 Lean Statement
theorem problem2 (x : ℝ) (n : ℕ) (h : x ^ (2 * n) = 3) : (x ^ (3 * n)) ^ 2 - (x ^ 2) ^ (2 * n) = 18 :=
by
  sorry

end problem1_problem2_l51_51254


namespace units_digit_sum_even_20_to_80_l51_51189

theorem units_digit_sum_even_20_to_80 :
  let a := 20
  let d := 2
  let l := 80
  let n := ((l - a) / d) + 1 -- Given by the formula l = a + (n-1)d => n = (l - a) / d + 1
  let sum := (n * (a + l)) / 2
  (sum % 10) = 0 := sorry

end units_digit_sum_even_20_to_80_l51_51189


namespace sequence_general_term_l51_51315

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n / (a n + 3)) :
  ∀ n : ℕ, n > 0 → a n = 3 / (n + 2) := 
by 
  sorry

end sequence_general_term_l51_51315


namespace find_integer_pairs_l51_51692

theorem find_integer_pairs :
  {p : ℤ × ℤ | p.1 * (p.1 + 1) * (p.1 + 7) * (p.1 + 8) = p.2^2} =
  {(1, 12), (1, -12), (-9, 12), (-9, -12), (0, 0), (-8, 0), (-4, -12), (-4, 12), (-1, 0), (-7, 0)} :=
sorry

end find_integer_pairs_l51_51692


namespace cosine_135_eq_neg_sqrt_2_div_2_l51_51167

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l51_51167


namespace multiply_1546_by_100_l51_51667

theorem multiply_1546_by_100 : 15.46 * 100 = 1546 :=
by
  sorry

end multiply_1546_by_100_l51_51667


namespace find_kn_l51_51372

section
variables (k n : ℝ)

def system_infinite_solutions (k n : ℝ) :=
  ∃ (y : ℝ → ℝ) (x : ℝ → ℝ),
  (∀ y, k * y + x y + n = 0) ∧
  (∀ y, |y - 2| + |y + 1| + |1 - y| + |y + 2| + x y = 0)

theorem find_kn :
  { (k, n) | system_infinite_solutions k n } = {(4, 0), (-4, 0), (2, 4), (-2, 4), (0, 6)} :=
sorry
end

end find_kn_l51_51372


namespace range_arcsin_x_squared_minus_x_l51_51205

noncomputable def range_of_arcsin : Set ℝ :=
  {x | -Real.arcsin (1 / 4) ≤ x ∧ x ≤ Real.pi / 2}

theorem range_arcsin_x_squared_minus_x :
  ∀ x : ℝ, ∃ y ∈ range_of_arcsin, y = Real.arcsin (x^2 - x) :=
by
  sorry

end range_arcsin_x_squared_minus_x_l51_51205


namespace sally_balloon_count_l51_51738

theorem sally_balloon_count 
  (joan_balloons : Nat)
  (jessica_balloons : Nat)
  (total_balloons : Nat)
  (sally_balloons : Nat)
  (h_joan : joan_balloons = 9)
  (h_jessica : jessica_balloons = 2)
  (h_total : total_balloons = 16)
  (h_eq : total_balloons = joan_balloons + jessica_balloons + sally_balloons) : 
  sally_balloons = 5 :=
by
  sorry

end sally_balloon_count_l51_51738


namespace find_a_b_l51_51195

theorem find_a_b (a b : ℝ)
  (h1 : a < 0)
  (h2 : (-b / a) = -((1 / 2) - (1 / 3)))
  (h3 : (2 / a) = -((1 / 2) * (1 / 3))) : 
  a + b = -14 :=
sorry

end find_a_b_l51_51195


namespace find_a_l51_51592

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l51_51592


namespace negation_of_proposition_l51_51872

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
sorry

end negation_of_proposition_l51_51872


namespace regression_slope_interpretation_l51_51242

-- Define the variables and their meanings
variable {x y : ℝ}

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.8 * x + 4.6

-- Define the proof statement
theorem regression_slope_interpretation (hx : ∀ x, y = regression_line x) :
  ∀ delta_x : ℝ, delta_x = 1 → (regression_line (x + delta_x) - regression_line x) = 0.8 :=
by
  intros delta_x h_delta_x
  rw [h_delta_x, regression_line, regression_line]
  simp
  sorry

end regression_slope_interpretation_l51_51242


namespace number_in_tens_place_is_7_l51_51617

theorem number_in_tens_place_is_7
  (digits : Finset ℕ)
  (a b c : ℕ)
  (h1 : digits = {7, 5, 2})
  (h2 : 100 * a + 10 * b + c > 530)
  (h3 : 100 * a + 10 * b + c < 710)
  (h4 : a ∈ digits)
  (h5 : b ∈ digits)
  (h6 : c ∈ digits)
  (h7 : ∀ x ∈ digits, x ≠ a → x ≠ b → x ≠ c) :
  b = 7 := sorry

end number_in_tens_place_is_7_l51_51617


namespace range_of_x_l51_51989

noncomputable def function_y (x : ℝ) : ℝ := 2 / (Real.sqrt (x + 4))

theorem range_of_x : ∀ x : ℝ, (∃ y : ℝ, y = function_y x) → x > -4 :=
by
  intro x h
  sorry

end range_of_x_l51_51989


namespace rationalize_denominator_l51_51893

theorem rationalize_denominator (a b c : Real) (h : b*c*c = a) :
  2 / (b + c) = (c*c) / (3 * 2) :=
by
  sorry

end rationalize_denominator_l51_51893


namespace tom_ratio_is_three_fourths_l51_51599

-- Define the years for the different programs
def bs_years : ℕ := 3
def phd_years : ℕ := 5
def tom_years : ℕ := 6
def normal_years : ℕ := bs_years + phd_years

-- Define the ratio of Tom's time to the normal time
def ratio : ℚ := tom_years / normal_years

theorem tom_ratio_is_three_fourths :
  ratio = 3 / 4 :=
by
  unfold ratio normal_years bs_years phd_years tom_years
  -- continued proof steps would go here
  sorry

end tom_ratio_is_three_fourths_l51_51599


namespace hydrogen_burns_oxygen_certain_l51_51428

-- define what it means for a chemical reaction to be well-documented and known to occur
def chemical_reaction (reactants : String) (products : String) : Prop :=
  (reactants = "2H₂ + O₂") ∧ (products = "2H₂O")

-- Event description and classification
def event_is_certain (event : String) : Prop :=
  event = "Hydrogen burns in oxygen to form water"

-- Main statement
theorem hydrogen_burns_oxygen_certain :
  ∀ (reactants products : String), (chemical_reaction reactants products) → event_is_certain "Hydrogen burns in oxygen to form water" :=
by
  intros reactants products h
  have h1 : reactants = "2H₂ + O₂" := h.1
  have h2 : products = "2H₂O" := h.2
  -- proof omitted
  exact sorry

end hydrogen_burns_oxygen_certain_l51_51428


namespace shaded_area_between_circles_l51_51233

theorem shaded_area_between_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5)
  (tangent : True) -- This represents that the circles are externally tangent
  (circumscribed : True) -- This represents the third circle circumscribing the two circles
  : ∃ r3 : ℝ, r3 = 9 ∧ π * r3^2 - (π * r1^2 + π * r2^2) = 40 * π :=
  sorry

end shaded_area_between_circles_l51_51233


namespace identity_element_is_neg4_l51_51690

def op (a b : ℝ) := a + b + 4

def is_identity (e : ℝ) := ∀ a : ℝ, op e a = a

theorem identity_element_is_neg4 : ∃ e : ℝ, is_identity e ∧ e = -4 :=
by
  use -4
  sorry

end identity_element_is_neg4_l51_51690


namespace negation_of_p_l51_51404

theorem negation_of_p : 
  (¬(∀ x : ℝ, |x| < 0)) ↔ (∃ x : ℝ, |x| ≥ 0) :=
by {
  sorry
}

end negation_of_p_l51_51404


namespace find_k_l51_51381

theorem find_k (x k : ℤ) (h : 2 * k - x = 2) (hx : x = -4) : k = -1 :=
by
  rw [hx] at h
  -- Substituting x = -4 into the equation
  sorry  -- Skipping further proof steps

end find_k_l51_51381


namespace largest_n_satisfying_ineq_l51_51287
  
theorem largest_n_satisfying_ineq : ∃ n : ℕ, (n < 10) ∧ ∀ m : ℕ, (m < 10) → m ≤ n ∧ (n < 10) ∧ (m < 10) → n = 9 :=
by
  sorry

end largest_n_satisfying_ineq_l51_51287


namespace solution_set_of_inequality_l51_51605

theorem solution_set_of_inequality (a : ℝ) (h1 : 2 * a - 3 < 0) (h2 : 1 - a < 0) : 1 < a ∧ a < 3 / 2 :=
by
  sorry

end solution_set_of_inequality_l51_51605


namespace find_q_l51_51587

-- Given polynomial Q(x) with coefficients p, q, d
variables {p q d : ℝ}

-- Define the polynomial Q(x)
def Q (x : ℝ) := x^3 + p * x^2 + q * x + d

-- Assume the conditions of the problem
theorem find_q (h1 : d = 5)                   -- y-intercept is 5
    (h2 : (-p / 3) = -d)                    -- mean of zeros = product of zeros
    (h3 : (-p / 3) = 1 + p + q + d)          -- mean of zeros = sum of coefficients
    : q = -26 := 
    sorry

end find_q_l51_51587


namespace smallest_t_satisfies_equation_l51_51812

def satisfies_equation (t x y : ℤ) : Prop :=
  (x^2 + y^2)^2 + 2 * t * x * (x^2 + y^2) = t^2 * y^2

theorem smallest_t_satisfies_equation : ∃ t x y : ℤ, t > 0 ∧ x > 0 ∧ y > 0 ∧ satisfies_equation t x y ∧
  ∀ t' x' y' : ℤ, t' > 0 ∧ x' > 0 ∧ y' > 0 ∧ satisfies_equation t' x' y' → t' ≥ t :=
sorry

end smallest_t_satisfies_equation_l51_51812


namespace jim_needs_more_miles_l51_51517

-- Define the conditions
def totalMiles : ℕ := 1200
def drivenMiles : ℕ := 923

-- Define the question and the correct answer
def remainingMiles : ℕ := totalMiles - drivenMiles

-- The theorem statement
theorem jim_needs_more_miles : remainingMiles = 277 :=
by
  -- This will contain the proof which is to be done later
  sorry

end jim_needs_more_miles_l51_51517


namespace largest_divisor_of_expression_l51_51703

theorem largest_divisor_of_expression (n : ℤ) : ∃ k : ℤ, k = 6 ∧ (n^3 - n + 15) % k = 0 := 
by
  use 6
  sorry

end largest_divisor_of_expression_l51_51703


namespace sector_area_l51_51963

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) :
  (1 / 2) * α * r ^ 2 = Real.pi := by
  sorry

end sector_area_l51_51963


namespace find_point_W_coordinates_l51_51880

theorem find_point_W_coordinates 
(O U S V : ℝ × ℝ)
(hO : O = (0, 0))
(hU : U = (3, 3))
(hS : S = (3, 0))
(hV : V = (0, 3))
(hSquare : (O.1 - U.1)^2 + (O.2 - U.2)^2 = 18)
(hArea_Square : 3 * 3 = 9) :
  ∃ W : ℝ × ℝ, W = (3, 9) ∧ 1 / 2 * (abs (S.1 - V.1) * abs (W.2 - S.2)) = 9 :=
by
  sorry

end find_point_W_coordinates_l51_51880


namespace minimum_a_condition_l51_51341

-- Define the quadratic function
def f (a x : ℝ) := x^2 + a * x + 1

-- Define the condition that the function remains non-negative in the open interval (0, 1/2)
def f_non_negative_in_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < 1 / 2 → f a x ≥ 0

-- State the theorem that the minimum value for a with the given condition is -5/2
theorem minimum_a_condition : ∀ (a : ℝ), f_non_negative_in_interval a → a ≥ -5 / 2 :=
by sorry

end minimum_a_condition_l51_51341


namespace range_of_a_l51_51108

theorem range_of_a (a : ℝ) :
  (∀ x, (x < -1 ∨ x > 5) ∨ (a < x ∧ x < a + 8)) ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end range_of_a_l51_51108


namespace toms_weekly_revenue_l51_51115

def crabs_per_bucket : Nat := 12
def number_of_buckets : Nat := 8
def price_per_crab : Nat := 5
def days_per_week : Nat := 7

theorem toms_weekly_revenue :
  (crabs_per_bucket * number_of_buckets * price_per_crab * days_per_week) = 3360 :=
by
  sorry

end toms_weekly_revenue_l51_51115


namespace minimize_quadratic_l51_51301

def f (x : ℝ) := 3 * x^2 - 18 * x + 7

theorem minimize_quadratic : ∃ x : ℝ, f x = -20 ∧ ∀ y : ℝ, f y ≥ -20 := by
  sorry

end minimize_quadratic_l51_51301


namespace daily_avg_for_entire_month_is_correct_l51_51042

-- conditions
def avg_first_25_days := 63
def days_first_25 := 25
def avg_last_5_days := 33
def days_last_5 := 5
def total_days := days_first_25 + days_last_5

-- question: What is the daily average for the entire month?
theorem daily_avg_for_entire_month_is_correct : 
  (avg_first_25_days * days_first_25 + avg_last_5_days * days_last_5) / total_days = 58 := by
  sorry

end daily_avg_for_entire_month_is_correct_l51_51042


namespace alcohol_concentration_l51_51597

theorem alcohol_concentration 
  (x : ℝ) -- concentration of alcohol in the first vessel (as a percentage)
  (h1 : 0 ≤ x ∧ x ≤ 100) -- percentage is between 0 and 100
  (h2 : (x / 100) * 2 + (55 / 100) * 6 = (37 / 100) * 10) -- given condition for concentration balance
  : x = 20 :=
sorry

end alcohol_concentration_l51_51597


namespace recruits_count_l51_51333

def x := 50
def y := 100
def z := 170

theorem recruits_count :
  ∃ n : ℕ, n = 211 ∧ (∀ a b c : ℕ, (b = 4 * a ∨ a = 4 * c ∨ c = 4 * b) → (b + 100 = a + 150) ∨ (a + 50 = c + 150) ∨ (c + 170 = b + 100)) :=
sorry

end recruits_count_l51_51333


namespace toby_total_time_l51_51507

theorem toby_total_time (d1 d2 d3 d4 : ℕ)
  (speed_loaded speed_unloaded : ℕ)
  (time1 time2 time3 time4 total_time : ℕ)
  (h1 : d1 = 180)
  (h2 : d2 = 120)
  (h3 : d3 = 80)
  (h4 : d4 = 140)
  (h5 : speed_loaded = 10)
  (h6 : speed_unloaded = 20)
  (h7 : time1 = d1 / speed_loaded)
  (h8 : time2 = d2 / speed_unloaded)
  (h9 : time3 = d3 / speed_loaded)
  (h10 : time4 = d4 / speed_unloaded)
  (h11 : total_time = time1 + time2 + time3 + time4) :
  total_time = 39 := by
  sorry

end toby_total_time_l51_51507


namespace lower_limit_of_range_l51_51469

theorem lower_limit_of_range (A : Set ℕ) (range_A : ℕ) (h1 : ∀ n ∈ A, Prime n∧ n ≤ 36) (h2 : range_A = 14)
  (h3 : ∃ x, x ∈ A ∧ ¬(∃ y, y ∈ A ∧ y > x)) (h4 : ∃ x, x ∈ A ∧ x = 31): 
  ∃ m, m ∈ A ∧ m = 17 := 
sorry

end lower_limit_of_range_l51_51469


namespace gcd_9155_4892_l51_51762

theorem gcd_9155_4892 : Nat.gcd 9155 4892 = 1 := 
by 
  sorry

end gcd_9155_4892_l51_51762


namespace company_p_percentage_increase_l51_51462

theorem company_p_percentage_increase :
  (460 - 400.00000000000006) / 400.00000000000006 * 100 = 15 := 
by
  sorry

end company_p_percentage_increase_l51_51462


namespace cookies_left_after_week_l51_51841

theorem cookies_left_after_week (cookies_in_jar : ℕ) (total_taken_out_in_4_days : ℕ) (same_amount_each_day : Prop)
  (h1 : cookies_in_jar = 70) (h2 : total_taken_out_in_4_days = 24) :
  ∃ (cookies_left : ℕ), cookies_left = 28 :=
by
  sorry

end cookies_left_after_week_l51_51841


namespace distance_between_points_l51_51153

open Real

theorem distance_between_points :
  ∀ (x1 y1 x2 y2 : ℝ),
  (x1, y1) = (-3, 1) →
  (x2, y2) = (5, -5) →
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 10 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end distance_between_points_l51_51153


namespace brad_must_make_5_trips_l51_51316

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r ^ 2 * h

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r ^ 2 * h

theorem brad_must_make_5_trips (r_barrel h_barrel r_bucket h_bucket : ℝ)
    (h1 : r_barrel = 10) (h2 : h_barrel = 15) (h3 : r_bucket = 10) (h4 : h_bucket = 10) :
    let trips := volume_of_cylinder r_barrel h_barrel / volume_of_cone r_bucket h_bucket
    let trips_needed := Int.ceil trips
    trips_needed = 5 := 
by
  sorry

end brad_must_make_5_trips_l51_51316


namespace ribbons_count_l51_51138

theorem ribbons_count (ribbons : ℕ) 
  (yellow_frac purple_frac orange_frac : ℚ)
  (black_ribbons : ℕ)
  (h1 : yellow_frac = 1/4)
  (h2 : purple_frac = 1/3)
  (h3 : orange_frac = 1/6)
  (h4 : ribbons - (yellow_frac * ribbons + purple_frac * ribbons + orange_frac * ribbons) = black_ribbons) :
  ribbons * orange_frac = 160 / 6 :=
by {
  sorry
}

end ribbons_count_l51_51138


namespace fourth_number_in_pascals_triangle_row_15_l51_51336

theorem fourth_number_in_pascals_triangle_row_15 : (Nat.choose 15 3) = 455 :=
by sorry

end fourth_number_in_pascals_triangle_row_15_l51_51336


namespace factorization_correct_l51_51492

theorem factorization_correct (a b : ℝ) : 6 * a * b - a^2 - 9 * b^2 = -(a - 3 * b)^2 :=
by
  sorry

end factorization_correct_l51_51492


namespace handshake_count_l51_51595

theorem handshake_count (n : ℕ) (m : ℕ) (couples : ℕ) (people : ℕ) 
  (h1 : couples = 15) 
  (h2 : people = 2 * couples)
  (h3 : people = 30)
  (h4 : n = couples) 
  (h5 : m = people / 2)
  (h6 : ∀ i : ℕ, i < m → ∀ j : ℕ, j < m → i ≠ j → i * j + i ≠ n 
    ∧ j * i + j ≠ n) 
  : n * (n - 1) / 2 + (2 * n - 2) * n = 315 :=
by
  sorry

end handshake_count_l51_51595


namespace mean_of_remaining_four_numbers_l51_51133

theorem mean_of_remaining_four_numbers (a b c d : ℝ) (h: (a + b + c + d + 105) / 5 = 90) :
  (a + b + c + d) / 4 = 86.25 :=
by
  sorry

end mean_of_remaining_four_numbers_l51_51133


namespace marshmallow_total_l51_51171

-- Define the number of marshmallows each kid can hold
def Haley := 8
def Michael := 3 * Haley
def Brandon := Michael / 2

-- Prove the total number of marshmallows held by all three is 44
theorem marshmallow_total : Haley + Michael + Brandon = 44 := by
  sorry

end marshmallow_total_l51_51171


namespace cassandra_makes_four_pies_l51_51313

-- Define the number of dozens and respective apples per dozen
def dozens : ℕ := 4
def apples_per_dozen : ℕ := 12

-- Define the total number of apples
def total_apples : ℕ := dozens * apples_per_dozen

-- Define apples per slice and slices per pie
def apples_per_slice : ℕ := 2
def slices_per_pie : ℕ := 6

-- Calculate the number of slices and number of pies based on conditions
def total_slices : ℕ := total_apples / apples_per_slice
def total_pies : ℕ := total_slices / slices_per_pie

-- Prove that the number of pies is 4
theorem cassandra_makes_four_pies : total_pies = 4 := by
  sorry

end cassandra_makes_four_pies_l51_51313


namespace intersection_complement_eq_l51_51216

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Theorem
theorem intersection_complement_eq : (A ∩ (U \ B)) = {2, 3} :=
by 
  sorry

end intersection_complement_eq_l51_51216


namespace degrees_to_radians_neg_210_l51_51986

theorem degrees_to_radians_neg_210 :
  -210 * (Real.pi / 180) = - (7 / 6) * Real.pi :=
by
  sorry

end degrees_to_radians_neg_210_l51_51986


namespace closure_of_A_range_of_a_l51_51499

-- Definitions for sets A and B
def A (x : ℝ) : Prop := x < -1 ∨ x > -0.5
def B (x a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

-- 1. Closure of A
theorem closure_of_A :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ -0.5) ↔ (∀ x : ℝ, A x) :=
sorry

-- 2. Range of a when A ∪ B = ℝ
theorem range_of_a (B_condition : ∀ x : ℝ, B x a) :
  (∀ a : ℝ, -1 ≤ x ∨ x ≥ -0.5) ↔ (-1.5 ≤ a ∧ a ≤ 0) :=
sorry

end closure_of_A_range_of_a_l51_51499


namespace time_ratio_A_to_B_l51_51338

theorem time_ratio_A_to_B (T_A T_B : ℝ) (hB : T_B = 36) (hTogether : 1 / T_A + 1 / T_B = 1 / 6) : T_A / T_B = 1 / 5 :=
by
  sorry

end time_ratio_A_to_B_l51_51338


namespace total_flowers_bouquets_l51_51886

-- Define the number of tulips Lana picked
def tulips : ℕ := 36

-- Define the number of roses Lana picked
def roses : ℕ := 37

-- Define the number of extra flowers Lana picked
def extra_flowers : ℕ := 3

-- Prove that the total number of flowers used by Lana for the bouquets is 76
theorem total_flowers_bouquets : (tulips + roses + extra_flowers) = 76 :=
by
  sorry

end total_flowers_bouquets_l51_51886


namespace probability_reach_correct_l51_51065

noncomputable def probability_reach (n : ℕ) : ℚ :=
  (2/3) + (1/12) * (1 - (-1/3)^(n-1))

theorem probability_reach_correct (n : ℕ) (P_n : ℚ) :
  P_n = probability_reach n :=
by
  sorry

end probability_reach_correct_l51_51065


namespace minimize_surface_area_l51_51471

theorem minimize_surface_area (V r h : ℝ) (hV : V = π * r^2 * h) (hA : 2 * π * r^2 + 2 * π * r * h = 2 * π * r^2 + 2 * π * r * h) : 
  (h / r) = 2 := 
by
  sorry

end minimize_surface_area_l51_51471


namespace distance_upstream_l51_51800

variable (v : ℝ) -- speed of the stream in km/h
variable (t : ℝ := 6) -- time of each trip in hours
variable (d_down : ℝ := 24) -- distance for downstream trip in km
variable (u : ℝ := 3) -- speed of man in still water in km/h

/- The distance the man swam upstream -/
theorem distance_upstream : 
  24 = (u + v) * t → 
  ∃ (d_up : ℝ), 
    d_up = (u - v) * t ∧
    d_up = 12 :=
by
  sorry

end distance_upstream_l51_51800


namespace average_is_correct_l51_51790

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

def sum_of_numbers : ℕ := numbers.foldr (· + ·) 0

def number_of_values : ℕ := numbers.length

def average : ℚ := sum_of_numbers / number_of_values

theorem average_is_correct : average = 114391.82 := by
  sorry

end average_is_correct_l51_51790


namespace complex_pow_imaginary_unit_l51_51473

theorem complex_pow_imaginary_unit (i : ℂ) (h : i^2 = -1) : i^2015 = -i :=
sorry

end complex_pow_imaginary_unit_l51_51473


namespace minimum_value_l51_51129

open Real

-- Statement of the conditions
def conditions (a b c : ℝ) : Prop :=
  -0.5 < a ∧ a < 0.5 ∧ -0.5 < b ∧ b < 0.5 ∧ -0.5 < c ∧ c < 0.5

-- Expression to be minimized
noncomputable def expression (a b c : ℝ) : ℝ :=
  1 / ((1 - a) * (1 - b) * (1 - c)) + 1 / ((1 + a) * (1 + b) * (1 + c))

-- Minimum value to prove
theorem minimum_value (a b c : ℝ) (h : conditions a b c) : expression a b c ≥ 4.74 :=
sorry

end minimum_value_l51_51129


namespace polynomial_factors_integers_l51_51052

theorem polynomial_factors_integers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 500)
  (h₃ : ∃ a : ℤ, n = a * (a + 1)) :
  n ≤ 21 :=
sorry

end polynomial_factors_integers_l51_51052


namespace union_M_N_eq_l51_51387

def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N : Set ℝ := {0, 4}

theorem union_M_N_eq : M ∪ N = Set.Icc 0 4 := 
  by
    sorry

end union_M_N_eq_l51_51387


namespace find_y_when_z_is_three_l51_51577

theorem find_y_when_z_is_three
  (k : ℝ) (y z : ℝ)
  (h1 : y = 3)
  (h2 : z = 1)
  (h3 : y ^ 4 * z ^ 2 = k)
  (hc : z = 3) :
  y ^ 4 = 9 :=
sorry

end find_y_when_z_is_three_l51_51577


namespace smallest_mu_exists_l51_51477

theorem smallest_mu_exists (a b c d : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :
  ∃ μ : ℝ, μ = (3 / 2) - (3 / (4 * Real.sqrt 2)) ∧ 
    (a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + μ * b^2 * c + c^2 * d) :=
by
  sorry

end smallest_mu_exists_l51_51477


namespace percentage_problem_l51_51536

theorem percentage_problem (x : ℝ) (h : 0.3 * 0.4 * x = 45) : 0.4 * 0.3 * x = 45 :=
by
  sorry

end percentage_problem_l51_51536


namespace proof_op_nabla_l51_51611

def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem proof_op_nabla :
  op_nabla (op_nabla (1/2) (1/3)) (1/4) = 9 / 11 := by
  sorry

end proof_op_nabla_l51_51611


namespace rods_in_one_mile_l51_51431

theorem rods_in_one_mile (chains_in_mile : ℕ) (rods_in_chain : ℕ) (mile_to_chain : 1 = 10 * chains_in_mile) (chain_to_rod : 1 = 22 * rods_in_chain) :
  1 * 220 = 10 * 22 :=
by sorry

end rods_in_one_mile_l51_51431


namespace betty_needs_more_flies_l51_51297

def flies_per_day := 2
def days_per_week := 7
def flies_needed_per_week := flies_per_day * days_per_week

def flies_caught_morning := 5
def flies_caught_afternoon := 6
def fly_escaped := 1

def flies_caught_total := flies_caught_morning + flies_caught_afternoon - fly_escaped

theorem betty_needs_more_flies : 
  flies_needed_per_week - flies_caught_total = 4 := by
  sorry

end betty_needs_more_flies_l51_51297


namespace similar_triangle_perimeter_l51_51144

noncomputable def is_similar_triangles (a b c a' b' c' : ℝ) := 
  ∃ (k : ℝ), k > 0 ∧ (a = k * a') ∧ (b = k * b') ∧ (c = k * c')

noncomputable def is_isosceles (a b c : ℝ) := (a = b) ∨ (a = c) ∨ (b = c)

theorem similar_triangle_perimeter :
  ∀ (a b c a' b' c' : ℝ),
    is_isosceles a b c → 
    is_similar_triangles a b c a' b' c' →
    c' = 42 →
    (a = 12) → 
    (b = 12) → 
    (c = 14) →
    (b' = 36) →
    (a' = 36) →
    a' + b' + c' = 114 :=
by
  intros
  sorry

end similar_triangle_perimeter_l51_51144


namespace area_of_square_BDEF_l51_51720

noncomputable def right_triangle (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
∃ (AB BC AC : ℝ), AB = 15 ∧ BC = 20 ∧ AC = Real.sqrt (AB^2 + BC^2)

noncomputable def is_square (B D E F : Type*) [MetricSpace B] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
∃ (BD DE EF FB : ℝ), BD = DE ∧ DE = EF ∧ EF = FB

noncomputable def height_of_triangle (E H M : Type*) [MetricSpace E] [MetricSpace H] [MetricSpace M] : Prop :=
∃ (EH : ℝ), EH = 2

theorem area_of_square_BDEF (A B C D E F H M N : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace F]
  [MetricSpace H] [MetricSpace M] [MetricSpace N]
  (H1 : right_triangle A B C)
  (H2 : is_square B D E F)
  (H3 : height_of_triangle E H M) :
  ∃ (area : ℝ), area = 100 :=
by
  sorry

end area_of_square_BDEF_l51_51720


namespace add_fractions_l51_51378

theorem add_fractions (x : ℝ) (h : x ≠ 1) : (1 / (x - 1) + 3 / (x - 1)) = (4 / (x - 1)) :=
by
  sorry

end add_fractions_l51_51378


namespace solve_equation_l51_51380

theorem solve_equation (x : ℝ) : 
  (4 * (1 - x)^2 = 25) ↔ (x = -3 / 2 ∨ x = 7 / 2) := 
by 
  sorry

end solve_equation_l51_51380


namespace set_equality_l51_51751

theorem set_equality (M P : Set (ℝ × ℝ))
  (hM : M = {p : ℝ × ℝ | p.1 + p.2 < 0 ∧ p.1 * p.2 > 0})
  (hP : P = {p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0}) : M = P :=
by
  sorry

end set_equality_l51_51751


namespace find_unknown_number_l51_51519

theorem find_unknown_number :
  (0.86 ^ 3 - 0.1 ^ 3) / (0.86 ^ 2) + x + 0.1 ^ 2 = 0.76 → 
  x = 0.115296 :=
sorry

end find_unknown_number_l51_51519


namespace total_muffins_correct_l51_51475

-- Define the conditions
def boys_count := 3
def muffins_per_boy := 12
def girls_count := 2
def muffins_per_girl := 20

-- Define the question and answer
def total_muffins_for_sale : Nat :=
  boys_count * muffins_per_boy + girls_count * muffins_per_girl

theorem total_muffins_correct :
  total_muffins_for_sale = 76 := by
  sorry

end total_muffins_correct_l51_51475


namespace at_least_fifty_same_leading_coefficient_l51_51827

-- Define what it means for two quadratic polynomials to intersect exactly once
def intersect_once (P Q : Polynomial ℝ) : Prop :=
∃ x, P.eval x = Q.eval x ∧ ∀ y ≠ x, P.eval y ≠ Q.eval y

-- Define the main theorem and its conditions
theorem at_least_fifty_same_leading_coefficient 
  (polynomials : Fin 100 → Polynomial ℝ)
  (h1 : ∀ i j, i ≠ j → intersect_once (polynomials i) (polynomials j))
  (h2 : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
        ¬∃ x, (polynomials i).eval x = (polynomials j).eval x ∧ (polynomials j).eval x = (polynomials k).eval x) : 
  ∃ (S : Finset (Fin 100)), S.card ≥ 50 ∧ ∃ a, ∀ i ∈ S, (polynomials i).leadingCoeff = a :=
sorry

end at_least_fifty_same_leading_coefficient_l51_51827


namespace marcia_project_hours_l51_51745

theorem marcia_project_hours (minutes_spent : ℕ) (minutes_per_hour : ℕ) 
  (h1 : minutes_spent = 300) 
  (h2 : minutes_per_hour = 60) : 
  (minutes_spent / minutes_per_hour) = 5 :=
by
  sorry

end marcia_project_hours_l51_51745


namespace distance_between_centers_eq_l51_51433

theorem distance_between_centers_eq (r1 r2 : ℝ) : ∃ d : ℝ, (d = r1 * Real.sqrt 2) := by
  sorry

end distance_between_centers_eq_l51_51433


namespace probability_top_three_same_color_l51_51644

/-- 
  A theorem stating the probability that the top three cards from a shuffled 
  standard deck of 52 cards are all of the same color is \(\frac{12}{51}\).
-/
theorem probability_top_three_same_color : 
  let deck := 52
  let colors := 2
  let cards_per_color := 26
  let favorable_outcomes := 2 * 26 * 25 * 24
  let total_outcomes := 52 * 51 * 50
  favorable_outcomes / total_outcomes = 12 / 51 :=
by
  sorry

end probability_top_three_same_color_l51_51644


namespace omega_value_l51_51528

noncomputable def f (ω : ℝ) (k : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x - Real.pi / 6) + k

theorem omega_value (ω k : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, f ω k x ≤ f ω k (Real.pi / 3)) → ω = 8 :=
by sorry

end omega_value_l51_51528


namespace sin_double_angle_l51_51849

theorem sin_double_angle (alpha : ℝ) (h1 : Real.cos (alpha + π / 4) = 3 / 5)
  (h2 : π / 2 ≤ alpha ∧ alpha ≤ 3 * π / 2) : Real.sin (2 * alpha) = 7 / 25 := 
sorry

end sin_double_angle_l51_51849


namespace simplify_fraction_l51_51509

theorem simplify_fraction :
  (2023^3 - 3 * 2023^2 * 2024 + 4 * 2023 * 2024^2 - 2024^3 + 2) / (2023 * 2024) = 2023 := by
sorry

end simplify_fraction_l51_51509


namespace expr_C_always_positive_l51_51826

-- Define the expressions as Lean definitions
def expr_A (x : ℝ) : ℝ := x^2
def expr_B (x : ℝ) : ℝ := abs (-x + 1)
def expr_C (x : ℝ) : ℝ := (-x)^2 + 2
def expr_D (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem expr_C_always_positive : ∀ (x : ℝ), expr_C x > 0 :=
by
  sorry

end expr_C_always_positive_l51_51826


namespace bags_needed_l51_51549

-- Definitions for the condition
def total_sugar : ℝ := 35.5
def bag_capacity : ℝ := 0.5

-- Theorem statement to solve the problem
theorem bags_needed : total_sugar / bag_capacity = 71 := 
by 
  sorry

end bags_needed_l51_51549


namespace clock_correction_calculation_l51_51004

noncomputable def clock_correction : ℝ :=
  let daily_gain := 5/4
  let hourly_gain := daily_gain / 24
  let total_hours := (9 * 24) + 9
  let total_gain := total_hours * hourly_gain
  total_gain

theorem clock_correction_calculation : clock_correction = 11.72 := by
  sorry

end clock_correction_calculation_l51_51004


namespace track_circumference_is_720_l51_51217

variable (P Q : Type) -- Define the types of P and Q, e.g., as points or runners.

noncomputable def circumference_of_the_track (C : ℝ) : Prop :=
  ∃ y : ℝ, 
  (∃ first_meeting_condition : Prop, first_meeting_condition = (150 = y - 150) ∧
  ∃ second_meeting_condition : Prop, second_meeting_condition = (2*y - 90 = y + 90) ∧
  C = 2 * y)

theorem track_circumference_is_720 :
  circumference_of_the_track 720 :=
by
  sorry

end track_circumference_is_720_l51_51217


namespace ninth_group_number_l51_51092

-- Conditions
def num_workers : ℕ := 100
def sample_size : ℕ := 20
def group_size : ℕ := num_workers / sample_size
def fifth_group_number : ℕ := 23

-- Theorem stating the result for the 9th group number.
theorem ninth_group_number : ∃ n : ℕ, n = 43 :=
by
  -- We calculate the numbers step by step.
  have interval : ℕ := group_size
  have difference : ℕ := 9 - 5
  have increment : ℕ := difference * interval
  have ninth_group_num : ℕ := fifth_group_number + increment
  use ninth_group_num
  sorry

end ninth_group_number_l51_51092


namespace pond_sustain_capacity_l51_51567

-- Defining the initial number of frogs
def initial_frogs : ℕ := 5

-- Defining the number of tadpoles
def number_of_tadpoles (frogs: ℕ) : ℕ := 3 * frogs

-- Defining the number of matured tadpoles (those that survive to become frogs)
def matured_tadpoles (tadpoles: ℕ) : ℕ := (2 * tadpoles) / 3

-- Defining the total number of frogs after tadpoles mature
def total_frogs_after_mature (initial_frogs: ℕ) (matured_tadpoles: ℕ) : ℕ :=
  initial_frogs + matured_tadpoles

-- Defining the number of frogs that need to find a new pond
def frogs_to_leave : ℕ := 7

-- Defining the number of frogs the pond can sustain
def frogs_pond_can_sustain (total_frogs: ℕ) (frogs_to_leave: ℕ) : ℕ :=
  total_frogs - frogs_to_leave

-- The main theorem stating the number of frogs the pond can sustain given the conditions
theorem pond_sustain_capacity : frogs_pond_can_sustain
  (total_frogs_after_mature initial_frogs (matured_tadpoles (number_of_tadpoles initial_frogs)))
  frogs_to_leave = 8 := by
  -- proof goes here
  sorry

end pond_sustain_capacity_l51_51567


namespace glucose_solution_l51_51810

theorem glucose_solution (x : ℝ) (h : (15 / 100 : ℝ) = (6.75 / x)) : x = 45 :=
sorry

end glucose_solution_l51_51810


namespace average_hamburgers_sold_per_day_l51_51604

theorem average_hamburgers_sold_per_day 
  (total_hamburgers : ℕ) (days_in_week : ℕ)
  (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 :=
by
  sorry

end average_hamburgers_sold_per_day_l51_51604


namespace find_white_daisies_l51_51104

theorem find_white_daisies (W P R : ℕ) 
  (h1 : P = 9 * W) 
  (h2 : R = 4 * P - 3) 
  (h3 : W + P + R = 273) : 
  W = 6 :=
by
  sorry

end find_white_daisies_l51_51104


namespace jeremy_sticker_distribution_l51_51257

def number_of_ways_to_distribute_stickers (total_stickers sheets : ℕ) : ℕ :=
  (Nat.choose (total_stickers - 1) (sheets - 1))

theorem jeremy_sticker_distribution : number_of_ways_to_distribute_stickers 10 3 = 36 :=
by
  sorry

end jeremy_sticker_distribution_l51_51257


namespace ratio_a_to_c_l51_51516

theorem ratio_a_to_c {a b c : ℚ} (h1 : a / b = 4 / 3) (h2 : b / c = 1 / 5) :
  a / c = 4 / 5 := 
sorry

end ratio_a_to_c_l51_51516


namespace tan_45_degree_is_one_l51_51990

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l51_51990


namespace problem1_problem2_l51_51123

variable {m x : ℝ}

-- Definition of the function f
def f (x m : ℝ) : ℝ := |x - m| + |x|

-- Statement for Problem (1)
theorem problem1 (h : f 1 m = 1) : 
  ∀ x, f x 1 < 2 ↔ (-1 / 2) < x ∧ x < (3 / 2) := 
sorry

-- Statement for Problem (2)
theorem problem2 (h : ∀ x, f x m ≥ m^2) : 
  -1 ≤ m ∧ m ≤ 1 := 
sorry

end problem1_problem2_l51_51123


namespace scientific_notation_of_trade_volume_l51_51214

-- Define the total trade volume
def total_trade_volume : ℕ := 175000000000

-- Define the expected scientific notation result
def expected_result : ℝ := 1.75 * 10^11

-- Theorem stating the problem
theorem scientific_notation_of_trade_volume :
  (total_trade_volume : ℝ) = expected_result := by
  sorry

end scientific_notation_of_trade_volume_l51_51214


namespace maximum_x1_x2_x3_l51_51598

theorem maximum_x1_x2_x3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
  x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
  x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
  x1 + x2 + x3 ≤ 61 := 
by sorry

end maximum_x1_x2_x3_l51_51598


namespace find_other_endpoint_l51_51809

set_option pp.funBinderTypes true

def circle_center : (ℝ × ℝ) := (5, -2)
def diameter_endpoint1 : (ℝ × ℝ) := (1, 2)
def diameter_endpoint2 : (ℝ × ℝ) := (9, -6)

theorem find_other_endpoint (c : ℝ × ℝ) (e1 : ℝ × ℝ) (e2 : ℝ × ℝ) : 
  c = circle_center ∧ e1 = diameter_endpoint1 → e2 = diameter_endpoint2 := by
  sorry

end find_other_endpoint_l51_51809


namespace arithmetic_mean_geom_mean_ratio_l51_51534

theorem arithmetic_mean_geom_mean_ratio {a b : ℝ} (h1 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h2 : a > b) (h3 : b > 0) : 
  (∃ k : ℤ, k = 34 ∧ abs ((a / b) - 34) ≤ 0.5) :=
sorry

end arithmetic_mean_geom_mean_ratio_l51_51534


namespace maximal_n_for_sequence_l51_51393

theorem maximal_n_for_sequence
  (a : ℕ → ℤ)
  (n : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 2 → a i + a (i + 1) + a (i + 2) > 0)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 4 → a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) < 0)
  : n ≤ 9 :=
sorry

end maximal_n_for_sequence_l51_51393


namespace find_x_l51_51864

-- Define the conditions according to the problem statement
variables {C x : ℝ} -- C is the cost per liter of pure spirit, x is the volume of water in the first solution

-- Condition 1: The cost for the first solution
def cost_first_solution (C : ℝ) (x : ℝ) : Prop := 0.50 = C * (1 / (1 + x))

-- Condition 2: The cost for the second solution (approximating 0.4999999999999999 as 0.50)
def cost_second_solution (C : ℝ) : Prop := 0.50 = C * (1 / 3)

-- The theorem to prove: x = 2 given the two conditions
theorem find_x (C : ℝ) (x : ℝ) (h1 : cost_first_solution C x) (h2 : cost_second_solution C) : x = 2 := 
sorry

end find_x_l51_51864


namespace coffee_machine_price_l51_51713

noncomputable def original_machine_price : ℝ :=
  let coffees_prior_cost_per_day := 2 * 4
  let new_coffees_cost_per_day := 3
  let daily_savings := coffees_prior_cost_per_day - new_coffees_cost_per_day
  let total_savings := 36 * daily_savings
  let discounted_price := total_savings
  let discount := 20
  discounted_price + discount

theorem coffee_machine_price
  (coffees_prior_cost_per_day : ℝ := 2 * 4)
  (new_coffees_cost_per_day : ℝ := 3)
  (daily_savings : ℝ := coffees_prior_cost_per_day - new_coffees_cost_per_day)
  (total_savings : ℝ := 36 * daily_savings)
  (discounted_price : ℝ := total_savings)
  (discount : ℝ := 20) :
  original_machine_price = 200 :=
by
  sorry

end coffee_machine_price_l51_51713


namespace abs_neg_two_equals_two_l51_51172

theorem abs_neg_two_equals_two : |(-2 : ℤ)| = 2 := 
by 
  sorry

end abs_neg_two_equals_two_l51_51172


namespace sequence_relation_l51_51169

theorem sequence_relation (b : ℕ → ℚ) : 
  b 1 = 2 ∧ b 2 = 5 / 11 ∧ (∀ n ≥ 3, b n = b (n-2) * b (n-1) / (3 * b (n-2) - b (n-1)))
  ↔ b 2023 = 5 / 12137 :=
by sorry

end sequence_relation_l51_51169


namespace molecular_weight_of_NH4Cl_l51_51116

theorem molecular_weight_of_NH4Cl (weight_8_moles : ℕ) (weight_per_mole : ℕ) :
  weight_8_moles = 424 →
  weight_per_mole = 53 →
  weight_8_moles / 8 = weight_per_mole :=
by
  intro h1 h2
  sorry

end molecular_weight_of_NH4Cl_l51_51116


namespace boxes_of_apples_l51_51671

theorem boxes_of_apples (n_crates apples_per_crate rotten_apples apples_per_box : ℕ) 
  (h1 : n_crates = 12) 
  (h2 : apples_per_crate = 42)
  (h3: rotten_apples = 4) 
  (h4 : apples_per_box = 10) : 
  (n_crates * apples_per_crate - rotten_apples) / apples_per_box = 50 :=
by
  sorry

end boxes_of_apples_l51_51671


namespace necessary_but_not_sufficient_l51_51539

variables (P Q : Prop)
variables (p : P) (q : Q)

-- Propositions
def quadrilateral_has_parallel_and_equal_sides : Prop := P
def is_rectangle : Prop := Q

-- Necessary but not sufficient condition
theorem necessary_but_not_sufficient (h : P → Q) : ¬(Q → P) :=
by sorry

end necessary_but_not_sufficient_l51_51539


namespace range_of_a_l51_51871

def quadratic_inequality (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 ≤ 0

theorem range_of_a :
  ¬ quadratic_inequality a ↔ -1 < a ∧ a < 3 :=
  by
  sorry

end range_of_a_l51_51871


namespace solve_sin_cos_eqn_l51_51627

theorem solve_sin_cos_eqn (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = 1) :
  x = 0 ∨ x = Real.pi / 2 :=
sorry

end solve_sin_cos_eqn_l51_51627


namespace clock_first_ring_at_midnight_l51_51621

theorem clock_first_ring_at_midnight (rings_every_n_hours : ℕ) (rings_per_day : ℕ) (hours_in_day : ℕ) :
  rings_every_n_hours = 3 ∧ rings_per_day = 8 ∧ hours_in_day = 24 →
  ∃ first_ring_time : Nat, first_ring_time = 0 :=
by
  sorry

end clock_first_ring_at_midnight_l51_51621


namespace total_calories_box_l51_51240

-- Definitions from the conditions
def bags := 6
def cookies_per_bag := 25
def calories_per_cookie := 18

-- Given the conditions, prove the total calories equals 2700
theorem total_calories_box : bags * cookies_per_bag * calories_per_cookie = 2700 := by
  sorry

end total_calories_box_l51_51240


namespace temperature_on_Saturday_l51_51662

theorem temperature_on_Saturday 
  (avg_temp : ℕ)
  (sun_temp : ℕ) 
  (mon_temp : ℕ) 
  (tue_temp : ℕ) 
  (wed_temp : ℕ) 
  (thu_temp : ℕ) 
  (fri_temp : ℕ)
  (saturday_temp : ℕ)
  (h_avg : avg_temp = 53)
  (h_sun : sun_temp = 40)
  (h_mon : mon_temp = 50) 
  (h_tue : tue_temp = 65) 
  (h_wed : wed_temp = 36) 
  (h_thu : thu_temp = 82) 
  (h_fri : fri_temp = 72) 
  (h_week : 7 * avg_temp = sun_temp + mon_temp + tue_temp + wed_temp + thu_temp + fri_temp + saturday_temp) :
  saturday_temp = 26 := 
by
  sorry

end temperature_on_Saturday_l51_51662


namespace operations_equivalent_l51_51000

theorem operations_equivalent (x : ℚ) : 
  ((x * (5 / 6)) / (2 / 3) - 2) = (x * (5 / 4) - 2) :=
sorry

end operations_equivalent_l51_51000


namespace scale_division_l51_51066

theorem scale_division (total_feet : ℕ) (inches_extra : ℕ) (part_length : ℕ) (total_parts : ℕ) :
  total_feet = 6 → inches_extra = 8 → part_length = 20 → 
  total_parts = (6 * 12 + 8) / 20 → total_parts = 4 :=
by
  intros
  sorry

end scale_division_l51_51066


namespace distance_between_closest_points_of_circles_l51_51606

theorem distance_between_closest_points_of_circles :
  let circle1_center : ℝ × ℝ := (3, 3)
  let circle2_center : ℝ × ℝ := (20, 15)
  let circle1_radius : ℝ := 3
  let circle2_radius : ℝ := 15
  let distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (15 - 3)^2)
  distance_between_centers - (circle1_radius + circle2_radius) = 2.81 :=
by {
  sorry
}

end distance_between_closest_points_of_circles_l51_51606


namespace probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l51_51276

namespace Sprinters

def P_A : ℚ := 2 / 5
def P_B : ℚ := 3 / 4
def P_C : ℚ := 1 / 3

def P_all_qualified := P_A * P_B * P_C
def P_two_qualified := P_A * P_B * (1 - P_C) + P_A * (1 - P_B) * P_C + (1 - P_A) * P_B * P_C
def P_at_least_one_qualified := 1 - (1 - P_A) * (1 - P_B) * (1 - P_C)

theorem probability_all_qualified : P_all_qualified = 1 / 10 :=
by 
  -- proof here
  sorry

theorem probability_two_qualified : P_two_qualified = 23 / 60 :=
by 
  -- proof here
  sorry

theorem probability_at_least_one_qualified : P_at_least_one_qualified = 9 / 10 :=
by 
  -- proof here
  sorry

end Sprinters

end probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l51_51276


namespace largest_number_using_digits_l51_51573

theorem largest_number_using_digits (d1 d2 d3 : ℕ) (h1 : d1 = 7) (h2 : d2 = 1) (h3 : d3 = 0) : 
  ∃ n : ℕ, (n = 710) ∧ (∀ m : ℕ, (m = d1 * 100 + d2 * 10 + d3) ∨ (m = d1 * 100 + d3 * 10 + d2) ∨ (m = d2 * 100 + d1 * 10 + d3) ∨ 
  (m = d2 * 100 + d3 * 10 + d1) ∨ (m = d3 * 100 + d1 * 10 + d2) ∨ (m = d3 * 100 + d2 * 10 + d1) → n ≥ m) := 
by
  sorry

end largest_number_using_digits_l51_51573


namespace solution_set_empty_range_l51_51527

theorem solution_set_empty_range (a : ℝ) : 
  (∀ x : ℝ, ax^2 + ax + 3 < 0 → false) ↔ (0 ≤ a ∧ a ≤ 12) := 
sorry

end solution_set_empty_range_l51_51527


namespace horizontal_distance_parabola_l51_51796

theorem horizontal_distance_parabola :
  ∀ x_p x_q : ℝ, 
  (x_p^2 + 3*x_p - 4 = 8) → 
  (x_q^2 + 3*x_q - 4 = 0) → 
  x_p ≠ x_q → 
  abs (x_p - x_q) = 2 :=
sorry

end horizontal_distance_parabola_l51_51796


namespace arithmetic_mean_of_two_digit_multiples_of_5_l51_51829

theorem arithmetic_mean_of_two_digit_multiples_of_5:
  let smallest := 10
  let largest := 95
  let num_terms := 18
  let sum := 945
  let mean := (sum : ℝ) / (num_terms : ℝ)
  mean = 52.5 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_5_l51_51829


namespace find_2u_plus_3v_l51_51778

theorem find_2u_plus_3v (u v : ℚ) (h1 : 5 * u - 6 * v = 28) (h2 : 3 * u + 5 * v = -13) :
  2 * u + 3 * v = -7767 / 645 := 
sorry

end find_2u_plus_3v_l51_51778


namespace ratio_of_distances_l51_51888

theorem ratio_of_distances (d_5 d_4 : ℝ) (h1 : d_5 + d_4 ≤ 26.67) (h2 : d_5 / 5 + d_4 / 4 = 6) : 
  d_5 / (d_5 + d_4) = 1 / 2 :=
sorry

end ratio_of_distances_l51_51888


namespace different_product_l51_51683

theorem different_product :
  let P1 := 190 * 80
  let P2 := 19 * 800
  let P3 := 19 * 8 * 10
  let P4 := 19 * 8 * 100
  P3 ≠ P1 ∧ P3 ≠ P2 ∧ P3 ≠ P4 :=
by
  sorry

end different_product_l51_51683


namespace cadence_worked_longer_by_5_months_l51_51452

-- Definitions
def months_old_company : ℕ := 36

def salary_old_company : ℕ := 5000

def salary_new_company : ℕ := 6000

def total_earnings : ℕ := 426000

-- Prove that Cadence worked 5 months longer at her new company
theorem cadence_worked_longer_by_5_months :
  ∃ x : ℕ, 
  total_earnings = salary_old_company * months_old_company + 
                  salary_new_company * (months_old_company + x)
  ∧ x = 5 :=
by {
  sorry
}

end cadence_worked_longer_by_5_months_l51_51452


namespace decrease_of_negative_distance_l51_51109

theorem decrease_of_negative_distance (x : Int) (increase : Int → Int) (decrease : Int → Int) :
  (increase 30 = 30) → (decrease 5 = -5) → (decrease 5 = -5) :=
by
  intros
  sorry

end decrease_of_negative_distance_l51_51109


namespace allan_has_4_more_balloons_than_jake_l51_51928

namespace BalloonProblem

def initial_balloons_allan : Nat := 6
def initial_balloons_jake : Nat := 2
def additional_balloons_jake : Nat := 3
def additional_balloons_allan : Nat := 4
def given_balloons_jake : Nat := 2
def given_balloons_allan : Nat := 3

def final_balloons_allan : Nat := (initial_balloons_allan + additional_balloons_allan) - given_balloons_allan
def final_balloons_jake : Nat := (initial_balloons_jake + additional_balloons_jake) - given_balloons_jake

theorem allan_has_4_more_balloons_than_jake :
  final_balloons_allan = final_balloons_jake + 4 :=
by
  -- proof is skipped with sorry
  sorry

end BalloonProblem

end allan_has_4_more_balloons_than_jake_l51_51928


namespace compute_expression_l51_51391

theorem compute_expression : 1 + 6 * 2 - 3 + 5 * 4 / 2 = 20 :=
by sorry

end compute_expression_l51_51391


namespace sqrt_expression_identity_l51_51741

theorem sqrt_expression_identity :
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := 
by
  sorry

end sqrt_expression_identity_l51_51741


namespace evaluate_expression_l51_51797

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l51_51797


namespace pet_center_final_count_l51_51232

/-!
# Problem: Count the total number of pets in a pet center after a series of adoption and collection events.
-/

def initialDogs : Nat := 36
def initialCats : Nat := 29
def initialRabbits : Nat := 15
def initialBirds : Nat := 10

def dogsAdopted1 : Nat := 20
def rabbitsAdopted1 : Nat := 5

def catsCollected : Nat := 12
def rabbitsCollected : Nat := 8
def birdsCollected : Nat := 5

def catsAdopted2 : Nat := 10
def birdsAdopted2 : Nat := 4

def finalDogs : Nat :=
  initialDogs - dogsAdopted1

def finalCats : Nat :=
  initialCats + catsCollected - catsAdopted2

def finalRabbits : Nat :=
  initialRabbits - rabbitsAdopted1 + rabbitsCollected

def finalBirds : Nat :=
  initialBirds + birdsCollected - birdsAdopted2

def totalPets (d c r b : Nat) : Nat :=
  d + c + r + b

theorem pet_center_final_count : 
  totalPets finalDogs finalCats finalRabbits finalBirds = 76 := by
  -- This is where we would provide the proof, but it's skipped as per the instructions.
  sorry

end pet_center_final_count_l51_51232


namespace triangle_angle_tangent_condition_l51_51842

theorem triangle_angle_tangent_condition
  (A B C : ℝ)
  (h1 : A + C = 2 * B)
  (h2 : Real.tan A * Real.tan C = 2 + Real.sqrt 3) :
  (A = Real.pi / 4 ∧ B = Real.pi / 3 ∧ C = 5 * Real.pi / 12) ∨
  (A = 5 * Real.pi / 12 ∧ B = Real.pi / 3 ∧ C = Real.pi / 4) :=
  sorry

end triangle_angle_tangent_condition_l51_51842


namespace janice_total_cost_is_correct_l51_51569

def cost_of_items (cost_juices : ℕ) (juices : ℕ) (cost_sandwiches : ℕ) (sandwiches : ℕ) (cost_pastries : ℕ) (pastries : ℕ) (cost_salad : ℕ) (discount_salad : ℕ) : ℕ :=
  let one_sandwich := cost_sandwiches / sandwiches
  let one_juice := cost_juices / juices
  let total_pastries := pastries * cost_pastries
  let discounted_salad := cost_salad - (cost_salad * discount_salad / 100)
  one_sandwich + one_juice + total_pastries + discounted_salad

-- Conditions
def cost_juices := 10
def juices := 5
def cost_sandwiches := 6
def sandwiches := 2
def cost_pastries := 4
def pastries := 2
def cost_salad := 8
def discount_salad := 20

-- Expected Total Cost
def expected_total_cost := 1940 -- in cents to avoid float numbers

theorem janice_total_cost_is_correct : 
  cost_of_items cost_juices juices cost_sandwiches sandwiches cost_pastries pastries cost_salad discount_salad = expected_total_cost :=
by
  simp [cost_of_items, cost_juices, juices, cost_sandwiches, sandwiches, cost_pastries, pastries, cost_salad, discount_salad]
  norm_num
  sorry

end janice_total_cost_is_correct_l51_51569


namespace smallest_GCD_value_l51_51096

theorem smallest_GCD_value (a b c d N : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : N > 5)
    (hc1 : Nat.gcd a b = 1 ∨ Nat.gcd a c = 1 ∨ Nat.gcd a d = 1 ∨ Nat.gcd b c = 1 ∨ Nat.gcd b d = 1 ∨ Nat.gcd c d = 1)
    (hc2 : Nat.gcd a b = 2 ∨ Nat.gcd a c = 2 ∨ Nat.gcd a d = 2 ∨ Nat.gcd b c = 2 ∨ Nat.gcd b d = 2 ∨ Nat.gcd c d = 2)
    (hc3 : Nat.gcd a b = 3 ∨ Nat.gcd a c = 3 ∨ Nat.gcd a d = 3 ∨ Nat.gcd b c = 3 ∨ Nat.gcd b d = 3 ∨ Nat.gcd c d = 3)
    (hc4 : Nat.gcd a b = 4 ∨ Nat.gcd a c = 4 ∨ Nat.gcd a d = 4 ∨ Nat.gcd b c = 4 ∨ Nat.gcd b d = 4 ∨ Nat.gcd c d = 4)
    (hc5 : Nat.gcd a b = 5 ∨ Nat.gcd a c = 5 ∨ Nat.gcd a d = 5 ∨ Nat.gcd b c = 5 ∨ Nat.gcd b d = 5 ∨ Nat.gcd c d = 5)
    (hcN : Nat.gcd a b = N ∨ Nat.gcd a c = N ∨ Nat.gcd a d = N ∨ Nat.gcd b c = N ∨ Nat.gcd b d = N ∨ Nat.gcd c d = N):
    N = 14 :=
sorry

end smallest_GCD_value_l51_51096


namespace wall_building_time_l51_51235

theorem wall_building_time (n t : ℕ) (h1 : n * t = 48) (h2 : n = 4) : t = 12 :=
by
  -- appropriate proof steps would go here
  sorry

end wall_building_time_l51_51235


namespace kylie_coins_count_l51_51347

theorem kylie_coins_count 
  (P : ℕ) 
  (from_brother : ℕ) 
  (from_father : ℕ) 
  (given_to_Laura : ℕ) 
  (coins_left : ℕ) 
  (h1 : from_brother = 13) 
  (h2 : from_father = 8) 
  (h3 : given_to_Laura = 21) 
  (h4 : coins_left = 15) : (P + from_brother + from_father) - given_to_Laura = coins_left → P = 15 :=
by
  sorry

end kylie_coins_count_l51_51347


namespace missing_number_l51_51435

theorem missing_number (x : ℝ) (h : 0.72 * 0.43 + x * 0.34 = 0.3504) : x = 0.12 :=
by sorry

end missing_number_l51_51435


namespace coaches_needed_l51_51892

theorem coaches_needed (x : ℕ) : 44 * x + 64 = 328 := by
  sorry

end coaches_needed_l51_51892


namespace explicit_formula_inequality_solution_l51_51314

noncomputable def f (x : ℝ) : ℝ := (x : ℝ) / (x^2 + 1)

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x → y < b → x < y → f x < f y
def f_half_eq_two_fifths : Prop := f (1/2) = 2/5

-- Questions rewritten as goals
theorem explicit_formula :
  odd_function f ∧ increasing_on_interval f (-1) 1 ∧ f_half_eq_two_fifths →
  ∀ x, f x = x / (x^2 + 1) := by 
sorry

theorem inequality_solution :
  odd_function f ∧ increasing_on_interval f (-1) 1 →
  ∀ t, (f (t - 1) + f t < 0) ↔ (0 < t ∧ t < 1/2) := by 
sorry

end explicit_formula_inequality_solution_l51_51314


namespace muffs_bought_before_december_correct_l51_51496

/-- Total ear muffs bought by customers in December. -/
def muffs_bought_in_december := 6444

/-- Total ear muffs bought by customers in all. -/
def total_muffs_bought := 7790

/-- Ear muffs bought before December. -/
def muffs_bought_before_december : Nat :=
  total_muffs_bought - muffs_bought_in_december

/-- Theorem stating the number of ear muffs bought before December. -/
theorem muffs_bought_before_december_correct :
  muffs_bought_before_december = 1346 :=
by
  unfold muffs_bought_before_december
  unfold total_muffs_bought
  unfold muffs_bought_in_december
  sorry

end muffs_bought_before_december_correct_l51_51496


namespace form_a_set_l51_51455

def is_definitive (description: String) : Prop :=
  match description with
  | "comparatively small numbers" => False
  | "non-negative even numbers not greater than 10" => True
  | "all triangles" => True
  | "points in the Cartesian coordinate plane with an x-coordinate of zero" => True
  | "tall male students" => False
  | "students under 17 years old in a certain class" => True
  | _ => False

theorem form_a_set :
  is_definitive "comparatively small numbers" = False ∧
  is_definitive "non-negative even numbers not greater than 10" = True ∧
  is_definitive "all triangles" = True ∧
  is_definitive "points in the Cartesian coordinate plane with an x-coordinate of zero" = True ∧
  is_definitive "tall male students" = False ∧
  is_definitive "students under 17 years old in a certain class" = True :=
by
  repeat { split };
  exact sorry

end form_a_set_l51_51455


namespace sum_of_integers_equals_75_l51_51288

theorem sum_of_integers_equals_75 
  (n m : ℤ) 
  (h1 : n * (n + 1) * (n + 2) = 924) 
  (h2 : m * (m + 1) * (m + 2) * (m + 3) = 924) 
  (sum_seven_integers : ℤ := n + (n + 1) + (n + 2) + m + (m + 1) + (m + 2) + (m + 3)) :
  sum_seven_integers = 75 := 
  sorry

end sum_of_integers_equals_75_l51_51288


namespace quadratic_min_value_l51_51402

theorem quadratic_min_value : ∀ x : ℝ, x^2 - 6 * x + 13 ≥ 4 := 
by 
  sorry

end quadratic_min_value_l51_51402


namespace polynomial_factors_sum_l51_51111

theorem polynomial_factors_sum (a b : ℝ) 
  (h : ∃ c : ℝ, (∀ x: ℝ, x^3 + a * x^2 + b * x + 8 = (x + 1) * (x + 2) * (x + c))) : 
  a + b = 21 :=
sorry

end polynomial_factors_sum_l51_51111


namespace intersection_M_N_l51_51823

noncomputable def M : Set ℝ := { x | x^2 ≤ x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = { x | 0 < x ∧ x ≤ 1 } :=
  sorry

end intersection_M_N_l51_51823


namespace geometric_sequence_collinear_vectors_l51_51399

theorem geometric_sequence_collinear_vectors (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (a2 a3 : ℝ)
  (h_a2 : a 2 = a2)
  (h_a3 : a 3 = a3)
  (h_parallel : 3 * a2 = 2 * a3) :
  (a2 + a 4) / (a3 + a 5) = 2 / 3 := 
by
  sorry

end geometric_sequence_collinear_vectors_l51_51399


namespace candidate_percentage_l51_51151

theorem candidate_percentage (P : ℝ) (l : ℝ) (V : ℝ) : 
  l = 5000.000000000007 ∧ 
  V = 25000.000000000007 ∧ 
  V - 2 * (P / 100) * V = l →
  P = 40 :=
by
  sorry

end candidate_percentage_l51_51151


namespace rhombus_perimeter_l51_51305

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
    let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
    (4 * s) = 52 :=
by
  sorry

end rhombus_perimeter_l51_51305


namespace no_such_function_exists_l51_51448

noncomputable def f : ℕ → ℕ := sorry

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), (∀ n > 1, f n = f (f (n-1)) + f (f (n+1))) ∧ (∀ n, f n > 0) :=
sorry

end no_such_function_exists_l51_51448


namespace total_arrangements_l51_51927

def total_members : ℕ := 6
def days : ℕ := 3
def people_per_day : ℕ := 2

def A_cannot_on_14 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 14 = 1

def B_cannot_on_16 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 16 = 2

theorem total_arrangements (arrangement : ℕ → ℕ) :
  (∀ arrangement, A_cannot_on_14 arrangement ∧ B_cannot_on_16 arrangement) →
  (total_members.choose 2 * (total_members - 2).choose 2 - 
  2 * (total_members - 1).choose 1 * (total_members - 2).choose 2 +
  (total_members - 2).choose 1 * (total_members - 3).choose 1)
  = 42 := 
by
  sorry

end total_arrangements_l51_51927


namespace number_of_real_values_p_l51_51344

theorem number_of_real_values_p (p : ℝ) :
  (∀ p: ℝ, x^2 - (p + 1) * x + (p + 1)^2 = 0 -> (p + 1) ^ 2 = 0) ↔ p = -1 := by
  sorry

end number_of_real_values_p_l51_51344


namespace simplify_sqrt_product_l51_51437

theorem simplify_sqrt_product (y : ℝ) (hy : y > 0) : 
  (Real.sqrt (45 * y) * Real.sqrt (20 * y) * Real.sqrt (30 * y) = 30 * y * Real.sqrt (30 * y)) :=
by
  sorry

end simplify_sqrt_product_l51_51437


namespace greatest_number_of_bouquets_l51_51284

def cherry_lollipops := 4
def orange_lollipops := 6
def raspberry_lollipops := 8
def lemon_lollipops := 10
def candy_canes := 12
def chocolate_coins := 14

theorem greatest_number_of_bouquets : 
  Nat.gcd cherry_lollipops (Nat.gcd orange_lollipops (Nat.gcd raspberry_lollipops (Nat.gcd lemon_lollipops (Nat.gcd candy_canes chocolate_coins)))) = 2 := 
by 
  sorry

end greatest_number_of_bouquets_l51_51284


namespace LCM_14_21_l51_51744

theorem LCM_14_21 : Nat.lcm 14 21 = 42 := 
by
  sorry

end LCM_14_21_l51_51744


namespace value_of_f_x_plus_5_l51_51090

open Function

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem value_of_f_x_plus_5 (x : ℝ) : f (x + 5) = 3 * x + 16 :=
by
  sorry

end value_of_f_x_plus_5_l51_51090


namespace inequality_solutions_l51_51755

theorem inequality_solutions :
  (∀ x : ℝ, 2 * x / (x + 1) < 1 ↔ -1 < x ∧ x < 1) ∧
  (∀ a x : ℝ,
    (x^2 + (2 - a) * x - 2 * a ≥ 0 ↔
      (a = -2 → True) ∧
      (a > -2 → (x ≤ -2 ∨ x ≥ a)) ∧
      (a < -2 → (x ≤ a ∨ x ≥ -2)))) :=
by
  sorry

end inequality_solutions_l51_51755
