import Mathlib

namespace NUMINAMATH_GPT_solve_for_z_l744_74455

variable (x y z : ℝ)

theorem solve_for_z (h : 1 / x - 1 / y = 1 / z) : z = x * y / (y - x) := 
sorry

end NUMINAMATH_GPT_solve_for_z_l744_74455


namespace NUMINAMATH_GPT_smallest_positive_period_pi_interval_extrema_l744_74404

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.cos (x + Real.pi / 3) + Real.sqrt 3

theorem smallest_positive_period_pi : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

theorem interval_extrema :
  ∃ x_max x_min : ℝ, 
  -Real.pi / 4 ≤ x_max ∧ x_max ≤ Real.pi / 6 ∧ f x_max = 2 ∧
  -Real.pi / 4 ≤ x_min ∧ x_min ≤ Real.pi / 6 ∧ f x_min = -1 ∧ 
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 2 ∧ f x ≥ -1) :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_pi_interval_extrema_l744_74404


namespace NUMINAMATH_GPT_range_of_m_n_l744_74401

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  m * Real.exp x + x^2 + n * x

theorem range_of_m_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0) ∧ (∀ x : ℝ, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_n_l744_74401


namespace NUMINAMATH_GPT_log_graph_passes_fixed_point_l744_74413

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem log_graph_passes_fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  log_a a (-1 + 2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_log_graph_passes_fixed_point_l744_74413


namespace NUMINAMATH_GPT_total_bottles_needed_l744_74436

-- Definitions from conditions
def large_bottle_capacity : ℕ := 450
def small_bottle_capacity : ℕ := 45
def extra_large_bottle_capacity : ℕ := 900

-- Theorem statement
theorem total_bottles_needed :
  ∃ (num_large_bottles num_small_bottles : ℕ), 
    num_large_bottles * large_bottle_capacity + num_small_bottles * small_bottle_capacity = extra_large_bottle_capacity ∧ 
    num_large_bottles + num_small_bottles = 2 :=
by
  sorry

end NUMINAMATH_GPT_total_bottles_needed_l744_74436


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l744_74473

variable (S : ℕ → ℕ) -- Define a function S that gives the sum of the first n terms.
variable (n : ℕ)     -- Define a natural number n.

-- Conditions based on the problem statement
axiom h1 : S n = 3
axiom h2 : S (2 * n) = 10

-- The theorem we need to prove
theorem arithmetic_sequence_sum : S (3 * n) = 21 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l744_74473


namespace NUMINAMATH_GPT_find_base_side_length_l744_74408

-- Regular triangular pyramid properties and derived values
variables
  (a l h : ℝ) -- side length of the base, slant height, and height of the pyramid
  (V : ℝ) -- volume of the pyramid

-- Given conditions
def inclined_to_base_plane_at_angle (angle : ℝ) := angle = 45
def volume_of_pyramid (V : ℝ) := V = 18

-- Prove the side length of the base
theorem find_base_side_length
  (h_eq : h = a * Real.sqrt 3 / 3)
  (volume_eq : V = 1 / 3 * (a * a * Real.sqrt 3 / 4) * h)
  (volume_given : V = 18) :
  a = 6 := by
  sorry

end NUMINAMATH_GPT_find_base_side_length_l744_74408


namespace NUMINAMATH_GPT_derivative_at_two_l744_74402

theorem derivative_at_two {f : ℝ → ℝ} (f_deriv : ∀x, deriv f x = 2 * x - 4) : deriv f 2 = 0 := 
by sorry

end NUMINAMATH_GPT_derivative_at_two_l744_74402


namespace NUMINAMATH_GPT_boat_speed_24_l744_74477

def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  let speed_downstream := x + 3
  let time := 1 / 4 -- 15 minutes in hours
  let distance := 6.75
  let equation := distance = speed_downstream * time
  equation ∧ x = 24

theorem boat_speed_24 (x : ℝ) (rate_of_current : ℝ) (time_minutes : ℝ) (distance_traveled : ℝ) 
  (h1 : rate_of_current = 3) (h2 : time_minutes = 15) (h3 : distance_traveled = 6.75) : speed_of_boat_in_still_water 24 := 
by
  -- Convert time in minutes to hours
  have time_in_hours : ℝ := time_minutes / 60
  -- Effective downstream speed
  have effective_speed := 24 + rate_of_current
  -- The equation to be satisfied
  have equation := distance_traveled = effective_speed * time_in_hours
  -- Simplify and solve
  sorry

end NUMINAMATH_GPT_boat_speed_24_l744_74477


namespace NUMINAMATH_GPT_tan_diff_eq_sqrt_three_l744_74474

open Real

theorem tan_diff_eq_sqrt_three (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : cos α * cos β = 1 / 6) (h5 : sin α * sin β = 1 / 3) : 
  tan (β - α) = sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_diff_eq_sqrt_three_l744_74474


namespace NUMINAMATH_GPT_expression_value_l744_74417

noncomputable def compute_expression (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80

theorem expression_value (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1)
    : compute_expression ω h h2 = -ω^2 :=
sorry

end NUMINAMATH_GPT_expression_value_l744_74417


namespace NUMINAMATH_GPT_opposite_of_x_is_positive_l744_74493

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end NUMINAMATH_GPT_opposite_of_x_is_positive_l744_74493


namespace NUMINAMATH_GPT_distinct_points_4_l744_74450

theorem distinct_points_4 (x y : ℝ) :
  (x + y = 7 ∨ 3 * x - 2 * y = -6) ∧ (x - y = -2 ∨ 4 * x + y = 10) →
  (x, y) =
    (5 / 2, 9 / 2) ∨ 
    (x, y) = (1, 6) ∨
    (x, y) = (-2, 0) ∨ 
    (x, y) = (14 / 11, 74 / 11) :=
sorry

end NUMINAMATH_GPT_distinct_points_4_l744_74450


namespace NUMINAMATH_GPT_calculate_value_l744_74443

theorem calculate_value : (535^2 - 465^2) / 70 = 1000 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l744_74443


namespace NUMINAMATH_GPT_jaymee_older_than_twice_shara_l744_74484

-- Given conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 22

-- Theorem to prove how many years older Jaymee is than twice Shara's age
theorem jaymee_older_than_twice_shara : jaymee_age - 2 * shara_age = 2 := by
  sorry

end NUMINAMATH_GPT_jaymee_older_than_twice_shara_l744_74484


namespace NUMINAMATH_GPT_total_pokemon_cards_l744_74491

def initial_cards : Nat := 27
def received_cards : Nat := 41
def lost_cards : Nat := 20

theorem total_pokemon_cards : initial_cards + received_cards - lost_cards = 48 := by
  sorry

end NUMINAMATH_GPT_total_pokemon_cards_l744_74491


namespace NUMINAMATH_GPT_total_oil_leak_l744_74447

-- Definitions for the given conditions
def before_repair_leak : ℕ := 6522
def during_repair_leak : ℕ := 5165
def total_leak : ℕ := 11687

-- The proof statement (without proof, only the statement)
theorem total_oil_leak :
  before_repair_leak + during_repair_leak = total_leak :=
sorry

end NUMINAMATH_GPT_total_oil_leak_l744_74447


namespace NUMINAMATH_GPT_quadratic_root_range_quadratic_product_of_roots_l744_74496

-- Problem (1): Prove the range of m.
theorem quadratic_root_range (m : ℝ) :
  (∀ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 → x1 ≠ x2) ↔ m < 1 := 
sorry

-- Problem (2): Prove the existence of m such that x1 * x2 = 0.
theorem quadratic_product_of_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 ∧ x1 * x2 = 0) ↔ m = -1 := 
sorry

end NUMINAMATH_GPT_quadratic_root_range_quadratic_product_of_roots_l744_74496


namespace NUMINAMATH_GPT_min_value_f_max_value_bac_l744_74467

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| - |x - 1|

theorem min_value_f : ∃ k : ℝ, (∀ x : ℝ, f x ≥ k) ∧ k = -2 := 
by
  sorry

theorem max_value_bac (a b c : ℝ) 
  (h1 : a^2 + c^2 + b^2 / 2 = 2) : 
  ∃ m : ℝ, (∀ a b c : ℝ, a^2 + c^2 + b^2 / 2 = 2 → b * (a + c) ≤ m) ∧ m = 2 := 
by
  sorry

end NUMINAMATH_GPT_min_value_f_max_value_bac_l744_74467


namespace NUMINAMATH_GPT_speed_of_second_train_equivalent_l744_74438

noncomputable def relative_speed_in_m_per_s (time_seconds : ℝ) (total_distance_m : ℝ) : ℝ :=
total_distance_m / time_seconds

noncomputable def relative_speed_in_km_per_h (relative_speed_m_per_s : ℝ) : ℝ :=
relative_speed_m_per_s * 3.6

noncomputable def speed_of_second_train (relative_speed_km_per_h : ℝ) (speed_of_first_train_km_per_h : ℝ) : ℝ :=
relative_speed_km_per_h - speed_of_first_train_km_per_h

theorem speed_of_second_train_equivalent
  (length_of_first_train length_of_second_train : ℝ)
  (speed_of_first_train_km_per_h : ℝ)
  (time_of_crossing_seconds : ℝ) :
  speed_of_second_train
    (relative_speed_in_km_per_h (relative_speed_in_m_per_s time_of_crossing_seconds (length_of_first_train + length_of_second_train)))
    speed_of_first_train_km_per_h = 36 := by
  sorry

end NUMINAMATH_GPT_speed_of_second_train_equivalent_l744_74438


namespace NUMINAMATH_GPT_triangle_longest_side_l744_74489

theorem triangle_longest_side (y : ℝ) (h₁ : 8 + (y + 5) + (3 * y + 2) = 45) : 
  ∃ s1 s2 s3, s1 = 8 ∧ s2 = y + 5 ∧ s3 = 3 * y + 2 ∧ (s1 + s2 + s3 = 45) ∧ (s3 = 24.5) := 
by
  sorry

end NUMINAMATH_GPT_triangle_longest_side_l744_74489


namespace NUMINAMATH_GPT_option_d_correct_l744_74414

theorem option_d_correct (a b c : ℝ) (h : a > b ∧ b > c ∧ c > 0) : a / b < a / c :=
by
  sorry

end NUMINAMATH_GPT_option_d_correct_l744_74414


namespace NUMINAMATH_GPT_phantom_needs_more_money_l744_74468

variable (given_money black_ink_price red_ink_price yellow_ink_price total_black_inks total_red_inks total_yellow_inks : ℕ)

def total_cost (total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price : ℕ) : ℕ :=
  total_black_inks * black_ink_price + total_red_inks * red_ink_price + total_yellow_inks * yellow_ink_price

theorem phantom_needs_more_money
  (h_given : given_money = 50)
  (h_black : black_ink_price = 11)
  (h_red : red_ink_price = 15)
  (h_yellow : yellow_ink_price = 13)
  (h_total_black : total_black_inks = 2)
  (h_total_red : total_red_inks = 3)
  (h_total_yellow : total_yellow_inks = 2) :
  given_money < total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price →
  total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price - given_money = 43 := by
  sorry

end NUMINAMATH_GPT_phantom_needs_more_money_l744_74468


namespace NUMINAMATH_GPT_total_earnings_l744_74403

theorem total_earnings (x y : ℝ) (h1 : 20 * x * y - 18 * x * y = 120) : 
  18 * x * y + 20 * x * y + 20 * x * y = 3480 := 
by
  sorry

end NUMINAMATH_GPT_total_earnings_l744_74403


namespace NUMINAMATH_GPT_cost_of_five_dozen_apples_l744_74499

theorem cost_of_five_dozen_apples 
  (cost_four_dozen : ℝ) 
  (cost_one_dozen : ℝ) 
  (cost_five_dozen : ℝ) 
  (h1 : cost_four_dozen = 31.20) 
  (h2 : cost_one_dozen = cost_four_dozen / 4) 
  (h3 : cost_five_dozen = 5 * cost_one_dozen)
  : cost_five_dozen = 39.00 :=
sorry

end NUMINAMATH_GPT_cost_of_five_dozen_apples_l744_74499


namespace NUMINAMATH_GPT_anand_present_age_l744_74479

theorem anand_present_age (A B : ℕ) 
  (h1 : B = A + 10)
  (h2 : A - 10 = (B - 10) / 3) :
  A = 15 :=
sorry

end NUMINAMATH_GPT_anand_present_age_l744_74479


namespace NUMINAMATH_GPT_triangle_area_divided_l744_74405

theorem triangle_area_divided {baseA heightA baseB heightB : ℝ} 
  (h1 : baseA = 1) 
  (h2 : heightA = 1)
  (h3 : baseB = 2)
  (h4 : heightB = 1)
  : (1 / 2 * baseA * heightA + 1 / 2 * baseB * heightB = 1.5) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_divided_l744_74405


namespace NUMINAMATH_GPT_pluto_orbit_scientific_notation_l744_74465

theorem pluto_orbit_scientific_notation : 5900000000 = 5.9 * 10^9 := by
  sorry

end NUMINAMATH_GPT_pluto_orbit_scientific_notation_l744_74465


namespace NUMINAMATH_GPT_hyperbola_m_value_l744_74454

noncomputable def m_value : ℝ := 2 * (Real.sqrt 2 - 1)

theorem hyperbola_m_value (a : ℝ) (m : ℝ) (AF_2 AF_1 BF_2 BF_1 : ℝ)
  (h1 : a = 1)
  (h2 : AF_2 = m)
  (h3 : AF_1 = 2 + AF_2)
  (h4 : AF_1 = m + BF_2)
  (h5 : BF_2 = 2)
  (h6 : BF_1 = 4)
  (h7 : BF_1 = Real.sqrt 2 * AF_1) :
  m = m_value :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_m_value_l744_74454


namespace NUMINAMATH_GPT_pool_filled_in_48_minutes_with_both_valves_open_l744_74460

def rate_first_valve_fills_pool_in_2_hours (V1 : ℚ) : Prop :=
  V1 * 120 = 12000

def rate_second_valve_50_more_than_first (V1 V2 : ℚ) : Prop :=
  V2 = V1 + 50

def pool_capacity : ℚ := 12000

def combined_rate (V1 V2 combinedRate : ℚ) : Prop :=
  combinedRate = V1 + V2

def time_to_fill_pool_with_both_valves_open (combinedRate time : ℚ) : Prop :=
  time = pool_capacity / combinedRate

theorem pool_filled_in_48_minutes_with_both_valves_open
  (V1 V2 combinedRate time : ℚ) :
  rate_first_valve_fills_pool_in_2_hours V1 →
  rate_second_valve_50_more_than_first V1 V2 →
  combined_rate V1 V2 combinedRate →
  time_to_fill_pool_with_both_valves_open combinedRate time →
  time = 48 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pool_filled_in_48_minutes_with_both_valves_open_l744_74460


namespace NUMINAMATH_GPT_jelly_beans_in_jar_X_l744_74475

theorem jelly_beans_in_jar_X : 
  ∀ (X Y : ℕ), (X + Y = 1200) → (X = 3 * Y - 400) → X = 800 :=
by
  sorry

end NUMINAMATH_GPT_jelly_beans_in_jar_X_l744_74475


namespace NUMINAMATH_GPT_Isabel_earning_l744_74478

-- Define the number of bead necklaces sold
def bead_necklaces : ℕ := 3

-- Define the number of gem stone necklaces sold
def gemstone_necklaces : ℕ := 3

-- Define the cost of each necklace
def cost_per_necklace : ℕ := 6

-- Calculate the total number of necklaces sold
def total_necklaces : ℕ := bead_necklaces + gemstone_necklaces

-- Calculate the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings is 36 dollars
theorem Isabel_earning : total_earnings = 36 := by
  sorry

end NUMINAMATH_GPT_Isabel_earning_l744_74478


namespace NUMINAMATH_GPT_exists_positive_integer_m_l744_74441

theorem exists_positive_integer_m (m : ℕ) (h_positive : m > 0) : 
  ∃ (m : ℕ), m > 0 ∧ ∃ k : ℕ, 8 * m = k^2 := 
sorry

end NUMINAMATH_GPT_exists_positive_integer_m_l744_74441


namespace NUMINAMATH_GPT_smallest_value_proof_l744_74421

noncomputable def smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) : Prop :=
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/x^2

theorem smallest_value_proof (x : ℝ) (h : 0 < x ∧ x < 1) : smallest_value x h :=
  sorry

end NUMINAMATH_GPT_smallest_value_proof_l744_74421


namespace NUMINAMATH_GPT_compute_a2004_l744_74424

def recurrence_sequence (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 0
  else sorry -- We'll define recurrence operations in the proofs

theorem compute_a2004 : recurrence_sequence 2004 = -2^1002 := 
sorry -- Proof omitted

end NUMINAMATH_GPT_compute_a2004_l744_74424


namespace NUMINAMATH_GPT_perpendicular_bisector_midpoint_l744_74419

theorem perpendicular_bisector_midpoint :
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  3 * R.1 - 2 * R.2 = -15 :=
by
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  sorry

end NUMINAMATH_GPT_perpendicular_bisector_midpoint_l744_74419


namespace NUMINAMATH_GPT_least_tiles_required_l744_74462

def floor_length : ℕ := 5000
def floor_breadth : ℕ := 1125
def gcd_floor : ℕ := Nat.gcd floor_length floor_breadth
def tile_area : ℕ := gcd_floor ^ 2
def floor_area : ℕ := floor_length * floor_breadth
def tiles_count : ℕ := floor_area / tile_area

theorem least_tiles_required : tiles_count = 360 :=
by
  sorry

end NUMINAMATH_GPT_least_tiles_required_l744_74462


namespace NUMINAMATH_GPT_extra_interest_is_correct_l744_74488

def principal : ℝ := 5000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def simple_interest (P R T : ℝ) : ℝ := P * R * T

def interest1 : ℝ := simple_interest principal rate1 time
def interest2 : ℝ := simple_interest principal rate2 time

def extra_interest : ℝ := interest1 - interest2

theorem extra_interest_is_correct : extra_interest = 600 := by
  sorry

end NUMINAMATH_GPT_extra_interest_is_correct_l744_74488


namespace NUMINAMATH_GPT_compute_sum_of_squares_l744_74463

noncomputable def polynomial_roots (p q r : ℂ) : Prop := 
  (p^3 - 15 * p^2 + 22 * p - 8 = 0) ∧ 
  (q^3 - 15 * q^2 + 22 * q - 8 = 0) ∧ 
  (r^3 - 15 * r^2 + 22 * r - 8 = 0) 

theorem compute_sum_of_squares (p q r : ℂ) (h : polynomial_roots p q r) :
  (p + q) ^ 2 + (q + r) ^ 2 + (r + p) ^ 2 = 406 := 
sorry

end NUMINAMATH_GPT_compute_sum_of_squares_l744_74463


namespace NUMINAMATH_GPT_product_of_primes_l744_74451

theorem product_of_primes :
  let p1 := 11
  let p2 := 13
  let p3 := 997
  p1 * p2 * p3 = 142571 :=
by
  sorry

end NUMINAMATH_GPT_product_of_primes_l744_74451


namespace NUMINAMATH_GPT_find_ratio_l744_74469

noncomputable def ratio_CN_AN (BM MC BK AB CN AN : ℝ) (h1 : BM / MC = 4 / 5) (h2 : BK / AB = 1 / 5) : Prop :=
  CN / AN = 5 / 24

theorem find_ratio (BM MC BK AB CN AN : ℝ) (h1 : BM / MC = 4 / 5) (h2 : BK / AB = 1 / 5) (h3 : BM + MC = BC) (h4 : BK = BK) (h5 : BK + AB = 6 * BK) : 
  ratio_CN_AN BM MC BK AB CN AN h1 h2 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_l744_74469


namespace NUMINAMATH_GPT_expand_expression_l744_74442

theorem expand_expression : ∀ (x : ℝ), 2 * (x + 3) * (x^2 - 2*x + 7) = 2*x^3 + 2*x^2 + 2*x + 42 := 
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_expression_l744_74442


namespace NUMINAMATH_GPT_largest_k_for_positive_root_l744_74495

theorem largest_k_for_positive_root : ∃ k : ℤ, k = 1 ∧ ∀ k' : ℤ, (k' > 1) → ¬ (∃ x > 0, 3 * x * (2 * k' * x - 5) - 2 * x^2 + 8 = 0) :=
by
  sorry

end NUMINAMATH_GPT_largest_k_for_positive_root_l744_74495


namespace NUMINAMATH_GPT_least_positive_multiple_of_17_gt_500_l744_74480

theorem least_positive_multiple_of_17_gt_500 : ∃ n: ℕ, n > 500 ∧ n % 17 = 0 ∧ n = 510 := by
  sorry

end NUMINAMATH_GPT_least_positive_multiple_of_17_gt_500_l744_74480


namespace NUMINAMATH_GPT_cellphone_loading_time_approximately_l744_74416

noncomputable def cellphone_loading_time_minutes : ℝ :=
  let T := 533.78 -- Solution for T from solving the given equation
  T / 60

theorem cellphone_loading_time_approximately :
  abs (cellphone_loading_time_minutes - 8.90) < 0.01 :=
by 
  -- The proof goes here, but we are just required to state it
  sorry

end NUMINAMATH_GPT_cellphone_loading_time_approximately_l744_74416


namespace NUMINAMATH_GPT_tan_product_identity_l744_74429

theorem tan_product_identity (A B : ℝ) (hA : A = 20) (hB : B = 25) (hSum : A + B = 45) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 := 
  by
  sorry

end NUMINAMATH_GPT_tan_product_identity_l744_74429


namespace NUMINAMATH_GPT_problem1_problem2_l744_74427

theorem problem1 : 101 * 99 = 9999 := 
by sorry

theorem problem2 : 32 * 2^2 + 14 * 2^3 + 10 * 2^4 = 400 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l744_74427


namespace NUMINAMATH_GPT_ring_toss_total_earnings_l744_74486

noncomputable def daily_earnings : ℕ := 144
noncomputable def number_of_days : ℕ := 22
noncomputable def total_earnings : ℕ := daily_earnings * number_of_days

theorem ring_toss_total_earnings :
  total_earnings = 3168 := by
  sorry

end NUMINAMATH_GPT_ring_toss_total_earnings_l744_74486


namespace NUMINAMATH_GPT_inverse_proportion_incorrect_D_l744_74433

theorem inverse_proportion_incorrect_D :
  ∀ (x y x1 y1 x2 y2 : ℝ), (y = -3 / x) ∧ (y1 = -3 / x1) ∧ (y2 = -3 / x2) ∧ (x1 < x2) → ¬(y1 < y2) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_incorrect_D_l744_74433


namespace NUMINAMATH_GPT_largest_n_for_negative_sum_l744_74432

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ} -- common difference of the arithmetic sequence

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

theorem largest_n_for_negative_sum
  (h_arith_seq : is_arithmetic_sequence a d)
  (h_first_term : a 0 < 0)
  (h_sum_2015_2016 : a 2014 + a 2015 > 0)
  (h_product_2015_2016 : a 2014 * a 2015 < 0) :
  (∀ n, sum_of_first_n_terms a n < 0 → n ≤ 4029) ∧ (sum_of_first_n_terms a 4029 < 0) :=
sorry

end NUMINAMATH_GPT_largest_n_for_negative_sum_l744_74432


namespace NUMINAMATH_GPT_ellipse_eccentricity_equilateral_triangle_l744_74470

theorem ellipse_eccentricity_equilateral_triangle
  (c a : ℝ) (h : c / a = 1 / 2) : eccentricity = 1 / 2 :=
by
  -- Proof goes here, we add sorry to skip proof content
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_equilateral_triangle_l744_74470


namespace NUMINAMATH_GPT_solve_for_a_l744_74497

theorem solve_for_a (x a : ℝ) (hx_pos : 0 < x) (hx_sqrt1 : x = (a+1)^2) (hx_sqrt2 : x = (a-3)^2) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l744_74497


namespace NUMINAMATH_GPT_quadratic_solution_l744_74445

theorem quadratic_solution (x : ℝ) (h_eq : x^2 - 3 * x - 6 = 0) (h_neq : x ≠ 0) :
    x = (3 + Real.sqrt 33) / 2 ∨ x = (3 - Real.sqrt 33) / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l744_74445


namespace NUMINAMATH_GPT_evaluate_expression_l744_74464

theorem evaluate_expression :
  let a := 2020
  let b := 2016
  (2^a + 2^b) / (2^a - 2^b) = 17 / 15 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l744_74464


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l744_74485

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) :=
sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l744_74485


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l744_74458

theorem distance_between_foci_of_ellipse :
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  2 * c = 4 * Real.sqrt 15 :=
by
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  show 2 * c = 4 * Real.sqrt 15
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l744_74458


namespace NUMINAMATH_GPT_shaggy_seeds_l744_74434

theorem shaggy_seeds {N : ℕ} (h1 : 50 < N) (h2 : N < 65) (h3 : N = 60) : 
  ∃ L : ℕ, L = 54 := by
  let L := 54
  sorry

end NUMINAMATH_GPT_shaggy_seeds_l744_74434


namespace NUMINAMATH_GPT_cost_of_scissor_l744_74456

noncomputable def scissor_cost (initial_money: ℕ) (scissors: ℕ) (eraser_count: ℕ) (eraser_cost: ℕ) (remaining_money: ℕ) :=
  (initial_money - remaining_money - (eraser_count * eraser_cost)) / scissors

theorem cost_of_scissor : scissor_cost 100 8 10 4 20 = 5 := 
by 
  sorry 

end NUMINAMATH_GPT_cost_of_scissor_l744_74456


namespace NUMINAMATH_GPT_evaluate_expression_l744_74439

variable (x y : ℝ)
variable (h₀ : x ≠ 0)
variable (h₁ : y ≠ 0)
variable (h₂ : 5 * x ≠ 3 * y)

theorem evaluate_expression : 
  (5 * x - 3 * y)⁻¹ * ((5 * x)⁻¹ - (3 * y)⁻¹) = -1 / (15 * x * y) :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l744_74439


namespace NUMINAMATH_GPT_expression_divisible_by_3_l744_74430

theorem expression_divisible_by_3 (k : ℤ) : ∃ m : ℤ, (2 * k + 3)^2 - 4 * k^2 = 3 * m :=
by
  sorry

end NUMINAMATH_GPT_expression_divisible_by_3_l744_74430


namespace NUMINAMATH_GPT_solve_fractional_equation_l744_74412

theorem solve_fractional_equation (x : ℝ) (h : (x + 1) / (4 * x^2 - 1) = (3 / (2 * x + 1)) - (4 / (4 * x - 2))) : x = 6 := 
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l744_74412


namespace NUMINAMATH_GPT_solve_for_y_l744_74444

theorem solve_for_y (y : ℝ) (h : (↑(30 * y) + (↑(30 * y) + 17) ^ (1 / 3)) ^ (1 / 3) = 17) :
  y = 816 / 5 := 
sorry

end NUMINAMATH_GPT_solve_for_y_l744_74444


namespace NUMINAMATH_GPT_certain_number_approximation_l744_74452

theorem certain_number_approximation (h1 : 2994 / 14.5 = 177) (h2 : 29.94 / x = 17.7) : x = 2.57455 := by
  sorry

end NUMINAMATH_GPT_certain_number_approximation_l744_74452


namespace NUMINAMATH_GPT_seymour_flats_of_roses_l744_74420

-- Definitions used in conditions
def flats_of_petunias := 4
def petunias_per_flat := 8
def venus_flytraps := 2
def fertilizer_per_petunia := 8
def fertilizer_per_rose := 3
def fertilizer_per_venus_flytrap := 2
def total_fertilizer := 314

-- Compute the total fertilizer for petunias and Venus flytraps
def total_fertilizer_petunias := flats_of_petunias * petunias_per_flat * fertilizer_per_petunia
def total_fertilizer_venus_flytraps := venus_flytraps * fertilizer_per_venus_flytrap

-- Remaining fertilizer for roses
def remaining_fertilizer_for_roses := total_fertilizer - total_fertilizer_petunias - total_fertilizer_venus_flytraps

-- Define roses per flat and the fertilizer used per flat of roses
def roses_per_flat := 6
def fertilizer_per_flat_of_roses := roses_per_flat * fertilizer_per_rose

-- The number of flats of roses
def flats_of_roses := remaining_fertilizer_for_roses / fertilizer_per_flat_of_roses

-- The proof problem statement
theorem seymour_flats_of_roses : flats_of_roses = 3 := by
  sorry

end NUMINAMATH_GPT_seymour_flats_of_roses_l744_74420


namespace NUMINAMATH_GPT_bullet_train_speed_l744_74481

theorem bullet_train_speed 
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (speed_train2 : ℝ)
  (time_cross : ℝ)
  (combined_length : ℝ)
  (time_cross_hours : ℝ)
  (relative_speed : ℝ)
  (speed_train1 : ℝ) :
  length_train1 = 270 → 
  length_train2 = 230.04 →
  speed_train2 = 80 →
  time_cross = 9 →
  combined_length = (length_train1 + length_train2) / 1000 →
  time_cross_hours = time_cross / 3600 →
  relative_speed = combined_length / time_cross_hours →
  relative_speed = speed_train1 + speed_train2 →
  speed_train1 = 120.016 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_bullet_train_speed_l744_74481


namespace NUMINAMATH_GPT_gerald_bars_l744_74423

theorem gerald_bars (G : ℕ) 
  (H1 : ∀ G, ∀ teacher_bars : ℕ, teacher_bars = 2 * G → total_bars = G + teacher_bars) 
  (H2 : ∀ total_bars : ℕ, total_squares = total_bars * 8 → total_squares_needed = 24 * 7) 
  (H3 : ∀ total_squares : ℕ, total_squares_needed = 24 * 7) 
  : G = 7 :=
by
  sorry

end NUMINAMATH_GPT_gerald_bars_l744_74423


namespace NUMINAMATH_GPT_second_set_parallel_lines_l744_74492

theorem second_set_parallel_lines (n : ℕ) (h1 : 5 * (n - 1) = 420) : n = 85 :=
by sorry

end NUMINAMATH_GPT_second_set_parallel_lines_l744_74492


namespace NUMINAMATH_GPT_circumference_of_jogging_track_l744_74425

-- Definitions for the given conditions
def speed_deepak : ℝ := 4.5
def speed_wife : ℝ := 3.75
def meet_time : ℝ := 4.32

-- The theorem stating the problem
theorem circumference_of_jogging_track : 
  (speed_deepak + speed_wife) * meet_time = 35.64 :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_jogging_track_l744_74425


namespace NUMINAMATH_GPT_B_investment_l744_74490

theorem B_investment (A : ℝ) (t_B : ℝ) (profit_ratio : ℝ) (B_investment_result : ℝ) : 
  A = 27000 → t_B = 4.5 → profit_ratio = 2 → B_investment_result = 36000 :=
by
  intro hA htB hpR
  sorry

end NUMINAMATH_GPT_B_investment_l744_74490


namespace NUMINAMATH_GPT_fixed_point_of_function_l744_74483

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ x : ℝ, (x = 1 → a^(x-1) = 1) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_function_l744_74483


namespace NUMINAMATH_GPT_coin_arrangements_l744_74453

/-- We define the conditions for Robert's coin arrangement problem. -/
def gold_coins := 5
def silver_coins := 5
def total_coins := gold_coins + silver_coins

/-- We define the number of ways to arrange 5 gold coins and 5 silver coins in 10 positions,
using the binomial coefficient. -/
def arrangements_colors : ℕ := Nat.choose total_coins gold_coins

/-- We define the number of possible configurations for the orientation of the coins
such that no two adjacent coins are face to face. -/
def arrangements_orientation : ℕ := 11

/-- The total number of distinguishable arrangements of the coins. -/
def total_arrangements : ℕ := arrangements_colors * arrangements_orientation

theorem coin_arrangements : total_arrangements = 2772 := by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_coin_arrangements_l744_74453


namespace NUMINAMATH_GPT_find_xy_plus_yz_plus_xz_l744_74437

theorem find_xy_plus_yz_plus_xz
  (x y z : ℝ)
  (h₁ : x > 0)
  (h₂ : y > 0)
  (h₃ : z > 0)
  (eq1 : x^2 + x * y + y^2 = 75)
  (eq2 : y^2 + y * z + z^2 = 64)
  (eq3 : z^2 + z * x + x^2 = 139) :
  x * y + y * z + z * x = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_xy_plus_yz_plus_xz_l744_74437


namespace NUMINAMATH_GPT_expression_evaluates_to_4_l744_74410

theorem expression_evaluates_to_4 :
  2 * Real.cos (Real.pi / 6) + (- 1 / 2 : ℝ)⁻¹ + |Real.sqrt 3 - 2| + (2 * Real.sqrt (9 / 4))^0 + Real.sqrt 9 = 4 := 
by
  sorry

end NUMINAMATH_GPT_expression_evaluates_to_4_l744_74410


namespace NUMINAMATH_GPT_certain_number_is_17_l744_74449

theorem certain_number_is_17 (x : ℤ) (h : 2994 / x = 177) : x = 17 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_17_l744_74449


namespace NUMINAMATH_GPT_tangent_line_to_ellipse_l744_74448

theorem tangent_line_to_ellipse (k : ℝ) :
  (∀ x : ℝ, (x / 2 + 2 * (k * x + 2) ^ 2) = 2) →
  k^2 = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_ellipse_l744_74448


namespace NUMINAMATH_GPT_sqrt_of_square_neg_three_l744_74428

theorem sqrt_of_square_neg_three : Real.sqrt ((-3 : ℝ)^2) = 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_of_square_neg_three_l744_74428


namespace NUMINAMATH_GPT_current_value_l744_74487

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end NUMINAMATH_GPT_current_value_l744_74487


namespace NUMINAMATH_GPT_exists_infinitely_many_solutions_l744_74409

theorem exists_infinitely_many_solutions :
  ∃ m : ℕ, m > 0 ∧ (∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) →
    (1/a + 1/b + 1/c + 1/(a*b*c) = m / (a + b + c))) :=
sorry

end NUMINAMATH_GPT_exists_infinitely_many_solutions_l744_74409


namespace NUMINAMATH_GPT_solution_part1_solution_part2_l744_74482

variable (f : ℝ → ℝ) (a x m : ℝ)

def problem_statement :=
  (∀ x : ℝ, f x = abs (x - a)) ∧
  (∀ x : ℝ, f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5)

theorem solution_part1 (x : ℝ) (h : problem_statement f a) : a = 2 :=
by
  sorry

theorem solution_part2 (x : ℝ) (h : problem_statement f a) :
  (∀ x : ℝ, f x + f (x + 5) ≥ m) → m ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_solution_part1_solution_part2_l744_74482


namespace NUMINAMATH_GPT_find_N_l744_74418

variables (k N : ℤ)

theorem find_N (h : ((k * N + N) / N - N) = k - 2021) : N = 2022 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l744_74418


namespace NUMINAMATH_GPT_trigonometric_expression_in_third_quadrant_l744_74407

theorem trigonometric_expression_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin α < 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.tan α > 0) : 
  ¬ (Real.tan α - Real.sin α < 0) :=
sorry

end NUMINAMATH_GPT_trigonometric_expression_in_third_quadrant_l744_74407


namespace NUMINAMATH_GPT_find_f_six_l744_74422

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the function definition

axiom f_property : ∀ x y : ℝ, f (x - y) = f x * f y
axiom f_nonzero : ∀ x : ℝ, f x ≠ 0
axiom f_two : f 2 = 5

theorem find_f_six : f 6 = 1 / 5 :=
sorry

end NUMINAMATH_GPT_find_f_six_l744_74422


namespace NUMINAMATH_GPT_race_total_distance_l744_74406

theorem race_total_distance (D : ℝ) 
  (A_time : D / 20 = D / 25 + 1) 
  (beat_distance : D / 20 * 25 = D + 20) : 
  D = 80 :=
sorry

end NUMINAMATH_GPT_race_total_distance_l744_74406


namespace NUMINAMATH_GPT_Zain_coins_total_l744_74446

theorem Zain_coins_total :
  ∀ (quarters dimes nickels : ℕ),
  quarters = 6 →
  dimes = 7 →
  nickels = 5 →
  Zain_coins = quarters + 10 + (dimes + 10) + (nickels + 10) →
  Zain_coins = 48 :=
by intros quarters dimes nickels hq hd hn Zain_coins
   sorry

end NUMINAMATH_GPT_Zain_coins_total_l744_74446


namespace NUMINAMATH_GPT_a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l744_74494

noncomputable def T_a : ℝ := 7.5
noncomputable def T_b : ℝ := 10
noncomputable def rounds_a (n : ℕ) : ℝ := n * T_a
noncomputable def rounds_b (n : ℕ) : ℝ := n * T_b

theorem a_beats_b_by_one_round_in_4_round_race :
  rounds_a 4 = rounds_b 3 := by
  sorry

theorem a_beats_b_by_T_a_minus_T_b :
  T_b - T_a = 2.5 := by
  sorry

end NUMINAMATH_GPT_a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l744_74494


namespace NUMINAMATH_GPT_gillian_spent_multiple_of_sandi_l744_74426

theorem gillian_spent_multiple_of_sandi
  (sandi_had : ℕ := 600)
  (gillian_spent : ℕ := 1050)
  (sandi_spent : ℕ := sandi_had / 2)
  (diff : ℕ := gillian_spent - sandi_spent)
  (extra : ℕ := 150)
  (multiple_of_sandi : ℕ := (diff - extra) / sandi_spent) : 
  multiple_of_sandi = 1 := 
  by sorry

end NUMINAMATH_GPT_gillian_spent_multiple_of_sandi_l744_74426


namespace NUMINAMATH_GPT_functional_equation_solution_l744_74472

theorem functional_equation_solution (f : ℚ → ℕ) :
  (∀ (x y : ℚ) (hx : 0 < x) (hy : 0 < y),
    f (x * y) * Nat.gcd (f x * f y) (f (x⁻¹) * f (y⁻¹)) = (x * y) * f (x⁻¹) * f (y⁻¹))
  → (∀ (x : ℚ) (hx : 0 < x), f x = x.num) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l744_74472


namespace NUMINAMATH_GPT_max_k_value_l744_74440

noncomputable def circle_equation (x y : ℝ) : Prop :=
x^2 + y^2 - 8 * x + 15 = 0

noncomputable def point_on_line (k x y : ℝ) : Prop :=
y = k * x - 2

theorem max_k_value (k : ℝ) :
  (∃ x y, circle_equation x y ∧ point_on_line k x y ∧ (x - 4)^2 + y^2 = 1) →
  k ≤ 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_k_value_l744_74440


namespace NUMINAMATH_GPT_inequality_solutions_l744_74400

theorem inequality_solutions (p p' q q' : ℕ) (hp : p ≠ p') (hq : q ≠ q') (hp_pos : 0 < p) (hp'_pos : 0 < p') (hq_pos : 0 < q) (hq'_pos : 0 < q') :
  (-(q : ℚ) / p > -(q' : ℚ) / p') ↔ (q * p' < p * q') :=
by
  sorry

end NUMINAMATH_GPT_inequality_solutions_l744_74400


namespace NUMINAMATH_GPT_combined_alloy_tin_amount_l744_74435

theorem combined_alloy_tin_amount
  (weight_A weight_B weight_C : ℝ)
  (ratio_lead_tin_A : ℝ)
  (ratio_tin_copper_B : ℝ)
  (ratio_copper_tin_C : ℝ)
  (amount_tin : ℝ) :
  weight_A = 150 → weight_B = 200 → weight_C = 250 →
  ratio_lead_tin_A = 5/3 → ratio_tin_copper_B = 2/3 → ratio_copper_tin_C = 4 →
  amount_tin = ((3/8) * weight_A) + ((2/5) * weight_B) + ((1/5) * weight_C) →
  amount_tin = 186.25 :=
by sorry

end NUMINAMATH_GPT_combined_alloy_tin_amount_l744_74435


namespace NUMINAMATH_GPT_find_X_l744_74471

def spadesuit (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 7

theorem find_X (X : ℝ) (h : spadesuit X 5 = 23) : X = 7.75 :=
by sorry

end NUMINAMATH_GPT_find_X_l744_74471


namespace NUMINAMATH_GPT_find_costs_of_A_and_B_find_price_reduction_l744_74466

-- Definitions for part 1
def cost_of_type_A_and_B (x y : ℕ) : Prop :=
  (5 * x + 3 * y = 450) ∧ (10 * x + 8 * y = 1000)

-- Part 1: Prove that x and y satisfy the cost conditions
theorem find_costs_of_A_and_B (x y : ℕ) (hx : 5 * x + 3 * y = 450) (hy : 10 * x + 8 * y = 1000) : 
  x = 60 ∧ y = 50 :=
sorry

-- Definitions for part 2
def daily_profit_condition (m : ℕ) : Prop :=
  (100 + 20 * m > 200) ∧ ((80 - m) * (100 + 20 * m) + 7000 = 10000)

-- Part 2: Prove that the price reduction m meets the profit condition
theorem find_price_reduction (m : ℕ) (hm : 100 + 20 * m > 200) (hp : (80 - m) * (100 + 20 * m) + 7000 = 10000) : 
  m = 10 :=
sorry

end NUMINAMATH_GPT_find_costs_of_A_and_B_find_price_reduction_l744_74466


namespace NUMINAMATH_GPT_prove_range_of_xyz_l744_74431

variable (x y z a : ℝ)

theorem prove_range_of_xyz 
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2 / 2)
  (ha : 0 < a) :
  (0 ≤ x ∧ x ≤ 2 * a / 3) ∧ (0 ≤ y ∧ y ≤ 2 * a / 3) ∧ (0 ≤ z ∧ z ≤ 2 * a / 3) :=
sorry

end NUMINAMATH_GPT_prove_range_of_xyz_l744_74431


namespace NUMINAMATH_GPT_max_value_frac_l744_74457

theorem max_value_frac (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  ∃ z, z = (x + y) / x ∧ z ≤ 2 / 3 := by
  sorry

end NUMINAMATH_GPT_max_value_frac_l744_74457


namespace NUMINAMATH_GPT_train_speed_is_60_kmph_l744_74415

noncomputable def train_length : ℝ := 110
noncomputable def time_to_pass_man : ℝ := 5.999520038396929
noncomputable def man_speed_kmph : ℝ := 6

theorem train_speed_is_60_kmph :
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / time_to_pass_man
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph = 60 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_60_kmph_l744_74415


namespace NUMINAMATH_GPT_correct_calculation_l744_74476

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := 
by sorry

end NUMINAMATH_GPT_correct_calculation_l744_74476


namespace NUMINAMATH_GPT_truth_probability_l744_74461

variables (P_A P_B P_AB : ℝ)

theorem truth_probability (h1 : P_B = 0.60) (h2 : P_AB = 0.48) : P_A = 0.80 :=
by
  have h3 : P_AB = P_A * P_B := sorry  -- Placeholder for the rule: P(A and B) = P(A) * P(B)
  rw [h2, h1] at h3
  sorry

end NUMINAMATH_GPT_truth_probability_l744_74461


namespace NUMINAMATH_GPT_impossible_to_equalize_numbers_l744_74459

theorem impossible_to_equalize_numbers (nums : Fin 6 → ℤ) :
  ¬ (∃ n : ℤ, ∀ i : Fin 6, nums i = n) :=
sorry

end NUMINAMATH_GPT_impossible_to_equalize_numbers_l744_74459


namespace NUMINAMATH_GPT_find_first_m_gt_1959_l744_74411

theorem find_first_m_gt_1959 :
  ∃ m n : ℕ, 8 * m - 7 = n^2 ∧ m > 1959 ∧ m = 2017 :=
by
  sorry

end NUMINAMATH_GPT_find_first_m_gt_1959_l744_74411


namespace NUMINAMATH_GPT_minutes_between_bathroom_visits_l744_74498

-- Definition of the conditions
def movie_duration_hours : ℝ := 2.5
def bathroom_uses : ℕ := 3
def minutes_per_hour : ℝ := 60

-- Theorem statement for the proof
theorem minutes_between_bathroom_visits :
  let total_movie_minutes := movie_duration_hours * minutes_per_hour
  let intervals := bathroom_uses + 1
  total_movie_minutes / intervals = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_minutes_between_bathroom_visits_l744_74498
