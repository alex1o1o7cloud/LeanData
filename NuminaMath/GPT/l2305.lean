import Mathlib

namespace NUMINAMATH_GPT_boat_license_combinations_l2305_230564

theorem boat_license_combinations :
  let letters := ['A', 'M', 'S']
  let non_zero_digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let any_digit := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  3 * 9 * 10^4 = 270000 := 
by 
  sorry

end NUMINAMATH_GPT_boat_license_combinations_l2305_230564


namespace NUMINAMATH_GPT_multiplication_sequence_result_l2305_230555

theorem multiplication_sequence_result : (1 * 3 * 5 * 7 * 9 * 11 = 10395) :=
by
  sorry

end NUMINAMATH_GPT_multiplication_sequence_result_l2305_230555


namespace NUMINAMATH_GPT_num_valid_permutations_l2305_230550

theorem num_valid_permutations : 
  let digits := [2, 0, 2, 3]
  let num_2 := 2
  let total_permutations := Nat.factorial 4 / (Nat.factorial num_2 * Nat.factorial 1 * Nat.factorial 1)
  let valid_start_2 := Nat.factorial 3
  let valid_start_3 := Nat.factorial 3 / Nat.factorial 2
  total_permutations = 12 ∧ valid_start_2 = 6 ∧ valid_start_3 = 3 ∧ (valid_start_2 + valid_start_3 = 9) := 
by
  sorry

end NUMINAMATH_GPT_num_valid_permutations_l2305_230550


namespace NUMINAMATH_GPT_octal_to_decimal_equiv_l2305_230500

-- Definitions for the octal number 724
def d0 := 4
def d1 := 2
def d2 := 7

-- Definition for the base
def base := 8

-- Calculation of the decimal equivalent
def calc_decimal : ℕ :=
  d0 * base^0 + d1 * base^1 + d2 * base^2

-- The proof statement
theorem octal_to_decimal_equiv : calc_decimal = 468 := by
  sorry

end NUMINAMATH_GPT_octal_to_decimal_equiv_l2305_230500


namespace NUMINAMATH_GPT_shelby_rain_drive_time_eq_3_l2305_230521

-- Definitions as per the conditions
def distance (v : ℝ) (t : ℝ) : ℝ := v * t
def total_distance := 24 -- in miles
def total_time := 50 / 60 -- in hours (converted to minutes)
def non_rainy_speed := 30 / 60 -- in miles per minute
def rainy_speed := 20 / 60 -- in miles per minute

-- Lean statement of the proof problem
theorem shelby_rain_drive_time_eq_3 :
  ∃ x : ℝ,
  (distance non_rainy_speed (total_time - x / 60) + distance rainy_speed (x / 60) = total_distance)
  ∧ (0 ≤ x) ∧ (x ≤ total_time * 60) →
  x = 3 := 
sorry

end NUMINAMATH_GPT_shelby_rain_drive_time_eq_3_l2305_230521


namespace NUMINAMATH_GPT_subset_A_if_inter_eq_l2305_230578

variable {B : Set ℝ}

def A : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem subset_A_if_inter_eq:
  A ∩ B = B ↔ B = ∅ ∨ B = {1} ∨ B = { x | 0 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_subset_A_if_inter_eq_l2305_230578


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l2305_230540

theorem distance_between_parallel_lines
  (l1 : ∀ (x y : ℝ), 2*x + y + 1 = 0)
  (l2 : ∀ (x y : ℝ), 4*x + 2*y - 1 = 0) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 5 / 10 := by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l2305_230540


namespace NUMINAMATH_GPT_tomatoes_for_5_liters_l2305_230587

theorem tomatoes_for_5_liters (kg_per_3_liters : ℝ) (liters_needed : ℝ) :
  (kg_per_3_liters = 69 / 3) → (liters_needed = 5) → (kg_per_3_liters * liters_needed = 115) := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_tomatoes_for_5_liters_l2305_230587


namespace NUMINAMATH_GPT_distance_from_mo_l2305_230591

-- Definitions based on conditions
-- 1. Grid squares have side length 1 cm.
-- 2. Shape shaded gray on the grid.
-- 3. The total shaded area needs to be divided into two equal parts.
-- 4. The line to be drawn is parallel to line MO.

noncomputable def grid_side_length : ℝ := 1.0
noncomputable def shaded_area : ℝ := 10.0
noncomputable def line_mo_distance (d : ℝ) : Prop := 
  ∃ parallel_line_distance, parallel_line_distance = d ∧ 
    ∃ equal_area, 2 * equal_area = shaded_area ∧ equal_area = 5.0

-- Theorem: The parallel line should be drawn at 2.6 cm 
theorem distance_from_mo (d : ℝ) : 
  d = 2.6 ↔ line_mo_distance d := 
by
  sorry

end NUMINAMATH_GPT_distance_from_mo_l2305_230591


namespace NUMINAMATH_GPT_part1_part2_l2305_230537

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Given conditions
axiom a_3a_5 : a 3 * a 5 = 63
axiom a_2a_6 : a 2 + a 6 = 16

-- Part (1) Proving the general formula
theorem part1 : 
  (∀ n : ℕ, a n = 12 - n) :=
sorry

-- Part (2) Proving the maximum value of S_n
theorem part2 :
  (∃ n : ℕ, (S n = (n * (12 - (n - 1) / 2)) → (n = 11 ∨ n = 12) ∧ (S n = 66))) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2305_230537


namespace NUMINAMATH_GPT_fraction_addition_l2305_230514

/--
The value of 2/5 + 1/3 is 11/15.
-/
theorem fraction_addition :
  (2 / 5 : ℚ) + (1 / 3) = 11 / 15 := 
sorry

end NUMINAMATH_GPT_fraction_addition_l2305_230514


namespace NUMINAMATH_GPT_find_number_l2305_230520

theorem find_number (x : ℕ) (n : ℕ) (h1 : x = 4) (h2 : x + n = 5) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2305_230520


namespace NUMINAMATH_GPT_geom_seq_inv_sum_eq_l2305_230519

noncomputable def geom_seq (a_1 r : ℚ) (n : ℕ) : ℚ := a_1 * r^n

theorem geom_seq_inv_sum_eq
    (a_1 r : ℚ)
    (h_sum : geom_seq a_1 r 0 + geom_seq a_1 r 1 + geom_seq a_1 r 2 + geom_seq a_1 r 3 = 15/8)
    (h_prod : geom_seq a_1 r 1 * geom_seq a_1 r 2 = -9/8) :
  1 / geom_seq a_1 r 0 + 1 / geom_seq a_1 r 1 + 1 / geom_seq a_1 r 2 + 1 / geom_seq a_1 r 3 = -5/3 :=
sorry

end NUMINAMATH_GPT_geom_seq_inv_sum_eq_l2305_230519


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2305_230561

theorem necessary_but_not_sufficient :
    (∀ (x y : ℝ), x > 2 ∧ y > 3 → x + y > 5 ∧ x * y > 6) ∧ 
    ¬(∀ (x y : ℝ), x + y > 5 ∧ x * y > 6 → x > 2 ∧ y > 3) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2305_230561


namespace NUMINAMATH_GPT_largest_possible_value_of_p_l2305_230536

theorem largest_possible_value_of_p (m n p : ℕ) (h1 : m ≤ n) (h2 : n ≤ p)
  (h3 : 2 * m * n * p = (m + 2) * (n + 2) * (p + 2)) : p ≤ 130 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_value_of_p_l2305_230536


namespace NUMINAMATH_GPT_inequality_holds_l2305_230556

theorem inequality_holds (x a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (a < c ∧ c < b) ∨ (b < c ∧ c < a)) 
  (h5 : (x - a) * (x - b) * (x - c) > 0) :
  (1 / (x - a)) + (1 / (x - b)) > 1 / (x - c) := 
by sorry

end NUMINAMATH_GPT_inequality_holds_l2305_230556


namespace NUMINAMATH_GPT_range_of_a_l2305_230530

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f a x ≤ f a (x + ε))) → a ≤ -1/4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2305_230530


namespace NUMINAMATH_GPT_no_divisor_30_to_40_of_2_pow_28_minus_1_l2305_230535

theorem no_divisor_30_to_40_of_2_pow_28_minus_1 :
  ¬ ∃ n : ℕ, (30 ≤ n ∧ n ≤ 40 ∧ n ∣ (2^28 - 1)) :=
by
  sorry

end NUMINAMATH_GPT_no_divisor_30_to_40_of_2_pow_28_minus_1_l2305_230535


namespace NUMINAMATH_GPT_positive_value_of_n_l2305_230571

theorem positive_value_of_n (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 25 = 0 ∧ ∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) →
  n = 20 :=
by
  sorry

end NUMINAMATH_GPT_positive_value_of_n_l2305_230571


namespace NUMINAMATH_GPT_transform_point_c_l2305_230538

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_diag (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem transform_point_c :
  let C := (3, 2)
  let C' := reflect_x C
  let C'' := reflect_y C'
  let C''' := reflect_diag C''
  C''' = (-2, -3) :=
by
  sorry

end NUMINAMATH_GPT_transform_point_c_l2305_230538


namespace NUMINAMATH_GPT_readers_both_l2305_230549

-- Definitions of the number of readers
def total_readers : ℕ := 150
def readers_science_fiction : ℕ := 120
def readers_literary_works : ℕ := 90

-- Statement of the proof problem
theorem readers_both :
  (readers_science_fiction + readers_literary_works - total_readers) = 60 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_readers_both_l2305_230549


namespace NUMINAMATH_GPT_roots_of_polynomial_l2305_230533

theorem roots_of_polynomial :
  (∀ x : ℝ, (x^2 - 5 * x + 6) * x * (x - 5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l2305_230533


namespace NUMINAMATH_GPT_max_hours_is_70_l2305_230501

-- Define the conditions
def regular_hourly_rate : ℕ := 8
def first_20_hours : ℕ := 20
def max_weekly_earnings : ℕ := 660
def overtime_rate_multiplier : ℕ := 25

-- Define the overtime hourly rate
def overtime_hourly_rate : ℕ := regular_hourly_rate + (regular_hourly_rate * overtime_rate_multiplier / 100)

-- Define the earnings for the first 20 hours
def earnings_first_20_hours : ℕ := regular_hourly_rate * first_20_hours

-- Define the maximum overtime earnings
def max_overtime_earnings : ℕ := max_weekly_earnings - earnings_first_20_hours

-- Define the maximum overtime hours
def max_overtime_hours : ℕ := max_overtime_earnings / overtime_hourly_rate

-- Define the maximum total hours
def max_total_hours : ℕ := first_20_hours + max_overtime_hours

-- Theorem to prove that the maximum number of hours is 70
theorem max_hours_is_70 : max_total_hours = 70 :=
by
  sorry

end NUMINAMATH_GPT_max_hours_is_70_l2305_230501


namespace NUMINAMATH_GPT_intersection_empty_l2305_230585

-- Define the set M
def M : Set ℝ := { x | ∃ y, y = Real.log (1 - x)}

-- Define the set N
def N : Set (ℝ × ℝ) := { p | ∃ x, ∃ y, (p = (x, y)) ∧ (y = Real.exp x) ∧ (x ∈ Set.univ)}

-- Prove that M ∩ N = ∅
theorem intersection_empty : M ∩ (Prod.fst '' N) = ∅ :=
by
  sorry

end NUMINAMATH_GPT_intersection_empty_l2305_230585


namespace NUMINAMATH_GPT_min_shirts_to_save_money_l2305_230594

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 75 + 8 * x < 12 * x ∧ x = 19 :=
sorry

end NUMINAMATH_GPT_min_shirts_to_save_money_l2305_230594


namespace NUMINAMATH_GPT_jerome_classmates_count_l2305_230529

theorem jerome_classmates_count (C F : ℕ) (h1 : F = C / 2) (h2 : 33 = C + F + 3) : C = 20 :=
by
  sorry

end NUMINAMATH_GPT_jerome_classmates_count_l2305_230529


namespace NUMINAMATH_GPT_fox_initial_coins_l2305_230567

theorem fox_initial_coins :
  ∃ (x : ℕ), ∀ (c1 c2 c3 : ℕ),
    c1 = 3 * x - 50 ∧
    c2 = 3 * c1 - 50 ∧
    c3 = 3 * c2 - 50 ∧
    3 * c3 - 50 = 20 →
    x = 25 :=
by
  sorry

end NUMINAMATH_GPT_fox_initial_coins_l2305_230567


namespace NUMINAMATH_GPT_find_m_value_l2305_230560

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

noncomputable def find_m (m : ℝ) : Prop :=
  f (f m) = f 2002 - 7 / 2

theorem find_m_value : find_m (-3 / 8) :=
by
  unfold find_m
  sorry

end NUMINAMATH_GPT_find_m_value_l2305_230560


namespace NUMINAMATH_GPT_range_of_m_for_distinct_real_roots_of_quadratic_l2305_230526

theorem range_of_m_for_distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 4*x1 - m = 0 ∧ x2^2 + 4*x2 - m = 0) ↔ m > -4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_distinct_real_roots_of_quadratic_l2305_230526


namespace NUMINAMATH_GPT_max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l2305_230581

-- Problem (1)
theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x*y ≤ 4 :=
sorry

-- Additional statement to show when the maximum is achieved
theorem max_xy_is_4 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x = 4 ∧ y = 1 ↔ x*y = 4 :=
sorry

-- Problem (2)
theorem min_x_plus_y (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x + y ≥ 9 :=
sorry

-- Additional statement to show when the minimum is achieved
theorem min_x_plus_y_is_9 (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x = 6 ∧ y = 3 ↔ x + y = 9 :=
sorry

end NUMINAMATH_GPT_max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l2305_230581


namespace NUMINAMATH_GPT_share_of_C_l2305_230505

theorem share_of_C (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 578) : 
  C = 408 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_share_of_C_l2305_230505


namespace NUMINAMATH_GPT_area_of_isosceles_trapezoid_l2305_230532

variable (a b c d : ℝ) -- Variables for sides and bases of the trapezoid

-- Define isosceles trapezoid with given sides and bases
def is_isosceles_trapezoid (a b c d : ℝ) (h : ℝ) :=
  a = b ∧ c = 10 ∧ d = 16 ∧ (∃ (h : ℝ), a^2 = h^2 + ((d - c) / 2)^2 ∧ a = 5)

-- Lean theorem for the area of the isosceles trapezoid
theorem area_of_isosceles_trapezoid :
  ∀ (a b c d : ℝ) (h : ℝ), is_isosceles_trapezoid a b c d h
  → (1 / 2) * (c + d) * h = 52 :=
by
  sorry

end NUMINAMATH_GPT_area_of_isosceles_trapezoid_l2305_230532


namespace NUMINAMATH_GPT_intersection_y_condition_l2305_230589

theorem intersection_y_condition (a : ℝ) :
  (∃ x y : ℝ, 2 * x - a * y + 2 = 0 ∧ x + y = 0 ∧ y < 0) → a < -2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_y_condition_l2305_230589


namespace NUMINAMATH_GPT_variance_le_second_moment_l2305_230598

noncomputable def variance (X : ℝ → ℝ) (MX : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - MX)^2]

noncomputable def second_moment (X : ℝ → ℝ) (C : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - C)^2]

theorem variance_le_second_moment (X : ℝ → ℝ) :
  ∀ C : ℝ, C ≠ MX → variance X MX ≤ second_moment X C := 
by
  sorry

end NUMINAMATH_GPT_variance_le_second_moment_l2305_230598


namespace NUMINAMATH_GPT_fourth_term_expansion_l2305_230523

def binomial_term (n r : ℕ) (a b : ℚ) : ℚ :=
  (Nat.descFactorial n r) / (Nat.factorial r) * a^(n - r) * b^r

theorem fourth_term_expansion (x : ℚ) (hx : x ≠ 0) : 
  binomial_term 6 3 2 (-(1 / (x^(1/3)))) = (-160 / x) :=
by
  sorry

end NUMINAMATH_GPT_fourth_term_expansion_l2305_230523


namespace NUMINAMATH_GPT_triangle_area_is_96_l2305_230592

-- Definitions of radii and sides being congruent
def tangent_circles (radius1 radius2 : ℝ) : Prop :=
  ∃ (O O' : ℝ × ℝ), dist O O' = radius1 + radius2

-- Given conditions
def radius_small : ℝ := 2
def radius_large : ℝ := 4
def sides_congruent (AB AC : ℝ) : Prop :=
  AB = AC

-- Theorem stating the goal
theorem triangle_area_is_96 
  (O O' : ℝ × ℝ)
  (AB AC : ℝ)
  (circ_tangent : tangent_circles radius_small radius_large)
  (sides_tangent : sides_congruent AB AC) :
  ∃ (BC : ℝ), ∃ (AF : ℝ), (1/2) * BC * AF = 96 := 
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_96_l2305_230592


namespace NUMINAMATH_GPT_backpack_original_price_l2305_230583

-- Define original price of a ring-binder
def original_ring_binder_price : ℕ := 20

-- Define the number of ring-binders bought
def number_of_ring_binders : ℕ := 3

-- Define the new price increase for the backpack
def backpack_price_increase : ℕ := 5

-- Define the new price decrease for the ring-binder
def ring_binder_price_decrease : ℕ := 2

-- Define the total amount spent
def total_amount_spent : ℕ := 109

-- Define the original price of the backpack variable
variable (B : ℕ)

-- Theorem statement: under these conditions, the original price of the backpack must be 50
theorem backpack_original_price :
  (B + backpack_price_increase) + ((original_ring_binder_price - ring_binder_price_decrease) * number_of_ring_binders) = total_amount_spent ↔ B = 50 :=
by 
  sorry

end NUMINAMATH_GPT_backpack_original_price_l2305_230583


namespace NUMINAMATH_GPT_relationship_among_abc_l2305_230547

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1/2)

theorem relationship_among_abc : a > b ∧ b > c :=
by {
  sorry
}

end NUMINAMATH_GPT_relationship_among_abc_l2305_230547


namespace NUMINAMATH_GPT_charlie_collected_15_seashells_l2305_230577

variables (c e : ℝ)

-- Charlie collected 10 more seashells than Emily
def charlie_more_seashells := c = e + 10

-- Emily collected one-third the number of seashells Charlie collected
def emily_seashells := e = c / 3

theorem charlie_collected_15_seashells (hc: charlie_more_seashells c e) (he: emily_seashells c e) : c = 15 := 
by sorry

end NUMINAMATH_GPT_charlie_collected_15_seashells_l2305_230577


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l2305_230576

theorem system1_solution (p q : ℝ) 
  (h1 : p + q = 4)
  (h2 : 2 * p - q = 5) : 
  p = 3 ∧ q = 1 := 
sorry

theorem system2_solution (v t : ℝ)
  (h3 : 2 * v + t = 3)
  (h4 : 3 * v - 2 * t = 3) :
  v = 9 / 7 ∧ t = 3 / 7 :=
sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l2305_230576


namespace NUMINAMATH_GPT_train_cross_time_approx_l2305_230563
noncomputable def time_to_cross_bridge
  (train_length : ℝ) (bridge_length : ℝ) (speed_kmh : ℝ) : ℝ :=
  ((train_length + bridge_length) / (speed_kmh * 1000 / 3600))

theorem train_cross_time_approx (train_length bridge_length speed_kmh : ℝ)
  (h_train_length : train_length = 250)
  (h_bridge_length : bridge_length = 300)
  (h_speed_kmh : speed_kmh = 44) :
  abs (time_to_cross_bridge train_length bridge_length speed_kmh - 45) < 1 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_time_approx_l2305_230563


namespace NUMINAMATH_GPT_students_selected_from_grade_10_l2305_230593

theorem students_selected_from_grade_10 (students_grade10 students_grade11 students_grade12 total_selected : ℕ)
  (h_grade10 : students_grade10 = 1200)
  (h_grade11 : students_grade11 = 1000)
  (h_grade12 : students_grade12 = 800)
  (h_total_selected : total_selected = 100) :
  students_grade10 * total_selected = 40 * (students_grade10 + students_grade11 + students_grade12) :=
by
  sorry

end NUMINAMATH_GPT_students_selected_from_grade_10_l2305_230593


namespace NUMINAMATH_GPT_sum_of_marked_angles_l2305_230543

theorem sum_of_marked_angles (sum_of_angles_around_vertex : ℕ := 360) 
    (vertices : ℕ := 7) (triangles : ℕ := 3) 
    (sum_of_interior_angles_triangle : ℕ := 180) :
    (vertices * sum_of_angles_around_vertex - triangles * sum_of_interior_angles_triangle) = 1980 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_marked_angles_l2305_230543


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l2305_230557

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

theorem parallel_vectors_x_value (x : ℝ) :
  ∀ k : ℝ,
  k ≠ 0 ∧ k * 1 = -2 ∧ k * -2 = x →
  x = 4 :=
by
  intros k hk
  have hk1 : k * 1 = -2 := hk.2.1
  have hk2 : k * -2 = x := hk.2.2
  -- Proceed from here to the calculations according to the steps in b):
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l2305_230557


namespace NUMINAMATH_GPT_product_of_two_numbers_l2305_230527

-- Definitions and conditions
def HCF (a b : ℕ) : ℕ := 9
def LCM (a b : ℕ) : ℕ := 200

-- Theorem statement
theorem product_of_two_numbers (a b : ℕ) (H₁ : HCF a b = 9) (H₂ : LCM a b = 200) : a * b = 1800 :=
by
  -- Injecting HCF and LCM conditions into the problem
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2305_230527


namespace NUMINAMATH_GPT_boat_travel_time_difference_l2305_230524

noncomputable def travel_time_difference (v : ℝ) : ℝ :=
  let d := 90
  let t_downstream := 2.5191640969412834
  let t_upstream := d / (v - 3)
  t_upstream - t_downstream

theorem boat_travel_time_difference :
  ∃ v : ℝ, travel_time_difference v = 0.5088359030587166 := 
by
  sorry

end NUMINAMATH_GPT_boat_travel_time_difference_l2305_230524


namespace NUMINAMATH_GPT_tetrahedron_edge_length_l2305_230597

-- Definitions corresponding to the conditions of the problem.
def radius : ℝ := 2

def diameter : ℝ := 2 * radius

/-- Centers of four mutually tangent balls -/
def center_distance : ℝ := diameter

/-- The side length of the square formed by the centers of four balls on the floor. -/
def side_length_of_square : ℝ := center_distance

/-- The edge length of the tetrahedron circumscribed around the four balls. -/
def edge_length_tetrahedron : ℝ := side_length_of_square

-- The statement to be proved.
theorem tetrahedron_edge_length :
  edge_length_tetrahedron = 4 :=
by
  sorry  -- Proof to be constructed

end NUMINAMATH_GPT_tetrahedron_edge_length_l2305_230597


namespace NUMINAMATH_GPT_initial_men_count_l2305_230502

-- Definitions based on problem conditions
def initial_days : ℝ := 18
def extra_men : ℝ := 400
def final_days : ℝ := 12.86

-- Proposition to show the initial number of men based on conditions
theorem initial_men_count (M : ℝ) (h : M * initial_days = (M + extra_men) * final_days) : M = 1000 := by
  sorry

end NUMINAMATH_GPT_initial_men_count_l2305_230502


namespace NUMINAMATH_GPT_factor_polynomial_l2305_230584

theorem factor_polynomial (x y z : ℤ) :
  x * (y - z) ^ 3 + y * (z - x) ^ 3 + z * (x - y) ^ 3 = (x - y) * (y - z) * (z - x) * (x + y + z) := 
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l2305_230584


namespace NUMINAMATH_GPT_shared_vertex_angle_of_triangle_and_square_l2305_230528

theorem shared_vertex_angle_of_triangle_and_square (α β γ δ ε ζ η θ : ℝ) :
  (α = 60 ∧ β = 60 ∧ γ = 60 ∧ δ = 90 ∧ ε = 90 ∧ ζ = 90 ∧ η = 90 ∧ θ = 90) →
  θ = 90 :=
by
  sorry

end NUMINAMATH_GPT_shared_vertex_angle_of_triangle_and_square_l2305_230528


namespace NUMINAMATH_GPT_fraction_of_original_price_l2305_230588

theorem fraction_of_original_price
  (CP SP : ℝ)
  (h1 : SP = 1.275 * CP)
  (f: ℝ)
  (h2 : f * SP = 0.85 * CP)
  : f = 17 / 25 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_original_price_l2305_230588


namespace NUMINAMATH_GPT_smallest_b_for_N_fourth_power_l2305_230541

theorem smallest_b_for_N_fourth_power : 
  ∃ (b : ℤ), (∀ n : ℤ, 7 * b^2 + 7 * b + 7 = n^4) ∧ b = 18 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_for_N_fourth_power_l2305_230541


namespace NUMINAMATH_GPT_limit_proof_l2305_230558

open Real

-- Define the conditions
axiom sin_6x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |sin (6 * x) / (6 * x) - 1| < ε
axiom arctg_2x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |arctan (2 * x) / (2 * x) - 1| < ε

-- State the limit proof problem
theorem limit_proof :
  (∃ ε > 0, ∀ x : ℝ, |x| < ε → x ≠ 0 →
  |(x * sin (6 * x)) / (arctan (2 * x)) ^ 2 - (3 / 2)| < ε) :=
sorry

end NUMINAMATH_GPT_limit_proof_l2305_230558


namespace NUMINAMATH_GPT_two_people_lying_l2305_230511

def is_lying (A B C D : Prop) : Prop :=
  (A ↔ ¬B) ∧ (B ↔ ¬C) ∧ (C ↔ ¬B) ∧ (D ↔ ¬A)

theorem two_people_lying (A B C D : Prop) (LA LB LC LD : Prop) :
  is_lying A B C D → (LA → ¬A) → (LB → ¬B) → (LC → ¬C) → (LD → ¬D) → (LA ∧ LC ∧ ¬LB ∧ ¬LD) :=
by
  sorry

end NUMINAMATH_GPT_two_people_lying_l2305_230511


namespace NUMINAMATH_GPT_solve_for_k_l2305_230531

-- Define the hypotheses as Lean statements
theorem solve_for_k (x k : ℝ) (h₁ : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_k_l2305_230531


namespace NUMINAMATH_GPT_polynomial_divisibility_l2305_230504

theorem polynomial_divisibility : 
  ∃ k : ℤ, (k = 8) ∧ (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x - 2) = 0) ∧ 
           (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x^2 + 1) = 0) :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l2305_230504


namespace NUMINAMATH_GPT_work_completion_l2305_230544

theorem work_completion (a b : ℕ) (h1 : a + b = 5) (h2 : a = 10) : b = 10 := by
  sorry

end NUMINAMATH_GPT_work_completion_l2305_230544


namespace NUMINAMATH_GPT_functional_equation_solution_l2305_230552

noncomputable def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * x + f x * f y) = x * f (x + y)

theorem functional_equation_solution (f : ℝ → ℝ) :
  func_equation f →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l2305_230552


namespace NUMINAMATH_GPT_drop_volume_l2305_230599

theorem drop_volume :
  let leak_rate := 3 -- drops per minute
  let pot_volume := 3 * 1000 -- volume in milliliters
  let time := 50 -- minutes
  let total_drops := leak_rate * time -- total number of drops
  (pot_volume / total_drops) = 20 := 
by
  let leak_rate : ℕ := 3
  let pot_volume : ℕ := 3 * 1000
  let time : ℕ := 50
  let total_drops := leak_rate * time
  have h : (pot_volume / total_drops) = 20 := by sorry
  exact h

end NUMINAMATH_GPT_drop_volume_l2305_230599


namespace NUMINAMATH_GPT_adjacent_product_negative_l2305_230534

noncomputable def a_seq : ℕ → ℚ
| 0 => 15
| (n+1) => (a_seq n) - (2 / 3)

theorem adjacent_product_negative :
  ∃ n : ℕ, a_seq 22 * a_seq 23 < 0 :=
by
  -- From the conditions, it is known that a_seq satisfies the recursive definition
  --
  -- We seek to prove that a_seq 22 * a_seq 23 < 0
  sorry

end NUMINAMATH_GPT_adjacent_product_negative_l2305_230534


namespace NUMINAMATH_GPT_wheel_rpm_l2305_230509

noncomputable def radius : ℝ := 175
noncomputable def speed_kmh : ℝ := 66
noncomputable def speed_cmm := speed_kmh * 100000 / 60 -- convert from km/h to cm/min
noncomputable def circumference := 2 * Real.pi * radius -- circumference of the wheel
noncomputable def rpm := speed_cmm / circumference -- revolutions per minute

theorem wheel_rpm : rpm = 1000 := by
  sorry

end NUMINAMATH_GPT_wheel_rpm_l2305_230509


namespace NUMINAMATH_GPT_shifted_graph_sum_l2305_230566

theorem shifted_graph_sum :
  let f (x : ℝ) := 3*x^2 - 2*x + 8
  let g (x : ℝ) := f (x - 6)
  let a := 3
  let b := -38
  let c := 128
  a + b + c = 93 :=
by
  sorry

end NUMINAMATH_GPT_shifted_graph_sum_l2305_230566


namespace NUMINAMATH_GPT_maximum_n_l2305_230506

/-- Definition of condition (a): For any three people, there exist at least two who know each other. -/
def condition_a (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 3 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), G.Adj a b

/-- Definition of condition (b): For any four people, there exist at least two who do not know each other. -/
def condition_b (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 4 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), ¬ G.Adj a b

theorem maximum_n (G : SimpleGraph V) [Fintype V] (h1 : condition_a G) (h2 : condition_b G) : 
  Fintype.card V ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_maximum_n_l2305_230506


namespace NUMINAMATH_GPT_min_value_of_c_l2305_230507

theorem min_value_of_c (a b c : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_ineq1 : a < b) 
  (h_ineq2 : b < 2 * b) 
  (h_ineq3 : 2 * b < c)
  (h_unique_sol : ∃ x : ℝ, 3 * x + (|x - a| + |x - b| + |x - (2 * b)| + |x - c|) = 3000) :
  c = 502 := sorry

end NUMINAMATH_GPT_min_value_of_c_l2305_230507


namespace NUMINAMATH_GPT_sum_geometric_arithmetic_progression_l2305_230595

theorem sum_geometric_arithmetic_progression :
  ∃ (a b r d : ℝ), a = 1 * r ∧ b = 1 * r^2 ∧ b = a + d ∧ 16 = b + d ∧ (a + b = 12.64) :=
by
  sorry

end NUMINAMATH_GPT_sum_geometric_arithmetic_progression_l2305_230595


namespace NUMINAMATH_GPT_hash_7_2_eq_24_l2305_230574

def hash_op (a b : ℕ) : ℕ := 4 * a - 2 * b

theorem hash_7_2_eq_24 : hash_op 7 2 = 24 := by
  sorry

end NUMINAMATH_GPT_hash_7_2_eq_24_l2305_230574


namespace NUMINAMATH_GPT_geometric_sum_proof_l2305_230568

theorem geometric_sum_proof (S : ℕ → ℝ) (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
    (hS3 : S 3 = 8) (hS6 : S 6 = 7)
    (Sn_def : ∀ n, S n = a 0 * (1 - r ^ n) / (1 - r)) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = -7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_proof_l2305_230568


namespace NUMINAMATH_GPT_find_other_number_l2305_230590

theorem find_other_number (HCF LCM one_number other_number : ℤ)
  (hHCF : HCF = 12)
  (hLCM : LCM = 396)
  (hone_number : one_number = 48)
  (hrelation : HCF * LCM = one_number * other_number) :
  other_number = 99 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l2305_230590


namespace NUMINAMATH_GPT_prove_jimmy_is_2_determine_rachel_age_l2305_230548

-- Define the conditions of the problem
variables (a b c r1 r2 : ℤ)

-- Condition 1: Rachel's age and Jimmy's age are roots of the quadratic equation
def is_root (p : ℤ → ℤ) (x : ℤ) : Prop := p x = 0

def quadratic_eq (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Condition 2: Sum of the coefficients is a prime number
def sum_of_coefficients_is_prime : Prop :=
  Nat.Prime (a + b + c).natAbs

-- Condition 3: Substituting Rachel’s age into the quadratic equation gives -55
def substitute_rachel_is_minus_55 (r : ℤ) : Prop :=
  quadratic_eq a b c r = -55

-- Question 1: Prove Jimmy is 2 years old
theorem prove_jimmy_is_2 (h1 : is_root (quadratic_eq a b c) r1)
                           (h2 : is_root (quadratic_eq a b c) r2)
                           (h3 : sum_of_coefficients_is_prime a b c)
                           (h4 : substitute_rachel_is_minus_55 a b c r1) :
  r2 = 2 :=
sorry

-- Question 2: Determine Rachel's age
theorem determine_rachel_age (h1 : is_root (quadratic_eq a b c) r1)
                             (h2 : is_root (quadratic_eq a b c) r2)
                             (h3 : sum_of_coefficients_is_prime a b c)
                             (h4 : substitute_rachel_is_minus_55 a b c r1)
                             (h5 : r2 = 2) :
  r1 = 7 :=
sorry

end NUMINAMATH_GPT_prove_jimmy_is_2_determine_rachel_age_l2305_230548


namespace NUMINAMATH_GPT_parallelogram_area_ratio_l2305_230551

theorem parallelogram_area_ratio (
  AB CD BC AD AP CQ BP DQ: ℝ)
  (h1 : AB = 13)
  (h2 : CD = 13)
  (h3 : BC = 15)
  (h4 : AD = 15)
  (h5 : AP = 10 / 3)
  (h6 : CQ = 10 / 3)
  (h7 : BP = 29 / 3)
  (h8 : DQ = 29 / 3)
  : ((area_APDQ / area_BPCQ) = 19) :=
sorry

end NUMINAMATH_GPT_parallelogram_area_ratio_l2305_230551


namespace NUMINAMATH_GPT_profit_percent_is_20_l2305_230570

variable (C S : ℝ)

-- Definition from condition: The cost price of 60 articles is equal to the selling price of 50 articles
def condition : Prop := 60 * C = 50 * S

-- Definition of profit percent to be proven as 20%
def profit_percent_correct : Prop := ((S - C) / C) * 100 = 20

theorem profit_percent_is_20 (h : condition C S) : profit_percent_correct C S :=
sorry

end NUMINAMATH_GPT_profit_percent_is_20_l2305_230570


namespace NUMINAMATH_GPT_farm_needs_12880_ounces_of_horse_food_per_day_l2305_230508

-- Define the given conditions
def ratio_sheep_to_horses : ℕ × ℕ := (1, 7)
def food_per_horse_per_day : ℕ := 230
def number_of_sheep : ℕ := 8

-- Define the proof goal
theorem farm_needs_12880_ounces_of_horse_food_per_day :
  let number_of_horses := number_of_sheep * ratio_sheep_to_horses.2
  number_of_horses * food_per_horse_per_day = 12880 :=
by
  sorry

end NUMINAMATH_GPT_farm_needs_12880_ounces_of_horse_food_per_day_l2305_230508


namespace NUMINAMATH_GPT_determine_a_l2305_230579

open Complex

noncomputable def complex_eq_real_im_part (a : ℝ) : Prop :=
  let z := (a - I) * (1 + I) / I
  (z.re, z.im) = ((a - 1 : ℝ), -(a + 1 : ℝ))

theorem determine_a (a : ℝ) (h : complex_eq_real_im_part a) : a = -1 :=
sorry

end NUMINAMATH_GPT_determine_a_l2305_230579


namespace NUMINAMATH_GPT_balance_blue_balls_l2305_230516

variables (G B Y W : ℝ)

-- Definitions based on conditions
def condition1 : Prop := 3 * G = 6 * B
def condition2 : Prop := 2 * Y = 5 * B
def condition3 : Prop := 6 * B = 4 * W

-- Statement of the problem
theorem balance_blue_balls (h1 : condition1 G B) (h2 : condition2 Y B) (h3 : condition3 B W) :
  4 * G + 2 * Y + 2 * W = 16 * B :=
sorry

end NUMINAMATH_GPT_balance_blue_balls_l2305_230516


namespace NUMINAMATH_GPT_customer_payment_eq_3000_l2305_230554

theorem customer_payment_eq_3000 (cost_price : ℕ) (markup_percentage : ℕ) (payment : ℕ)
  (h1 : cost_price = 2500)
  (h2 : markup_percentage = 20)
  (h3 : payment = cost_price + (markup_percentage * cost_price / 100)) :
  payment = 3000 :=
by
  sorry

end NUMINAMATH_GPT_customer_payment_eq_3000_l2305_230554


namespace NUMINAMATH_GPT_freezer_temp_calculation_l2305_230582

def refrigerator_temp : ℝ := 4
def freezer_temp (rt : ℝ) (d : ℝ) : ℝ := rt - d

theorem freezer_temp_calculation :
  (freezer_temp refrigerator_temp 22) = -18 :=
by
  sorry

end NUMINAMATH_GPT_freezer_temp_calculation_l2305_230582


namespace NUMINAMATH_GPT_frog_return_prob_A_after_2022_l2305_230586

def initial_prob_A : ℚ := 1
def transition_prob_A_to_adj : ℚ := 1/3
def transition_prob_adj_to_A : ℚ := 1/3
def transition_prob_adj_to_adj : ℚ := 2/3

noncomputable def prob_A_return (n : ℕ) : ℚ :=
if (n % 2 = 0) then
  (2/9) * (1/2^(n/2)) + (1/9)
else
  0

theorem frog_return_prob_A_after_2022 : prob_A_return 2022 = (2/9) * (1/2^1010) + (1/9) :=
by
  sorry

end NUMINAMATH_GPT_frog_return_prob_A_after_2022_l2305_230586


namespace NUMINAMATH_GPT_large_cube_side_length_l2305_230515

theorem large_cube_side_length (s1 s2 s3 : ℝ) (h1 : s1 = 1) (h2 : s2 = 6) (h3 : s3 = 8) : 
  ∃ s_large : ℝ, s_large^3 = s1^3 + s2^3 + s3^3 ∧ s_large = 9 := 
by 
  use 9
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_large_cube_side_length_l2305_230515


namespace NUMINAMATH_GPT_possible_values_a1_l2305_230517

def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem possible_values_a1 {a : ℕ → ℤ} (h1 : ∀ n : ℕ, a n + a (n + 1) = 2 * n - 1)
  (h2 : ∃ k : ℕ, sequence_sum a k = 190 ∧ sequence_sum a (k + 1) = 190) :
  (a 0 = -20 ∨ a 0 = 19) :=
sorry

end NUMINAMATH_GPT_possible_values_a1_l2305_230517


namespace NUMINAMATH_GPT_mabel_total_tomatoes_l2305_230510

def tomatoes_first_plant : ℕ := 12

def tomatoes_second_plant : ℕ := (2 * tomatoes_first_plant) - 6

def tomatoes_combined_first_two : ℕ := tomatoes_first_plant + tomatoes_second_plant

def tomatoes_third_plant : ℕ := tomatoes_combined_first_two / 2

def tomatoes_each_fourth_fifth_plant : ℕ := 3 * tomatoes_combined_first_two

def tomatoes_combined_fourth_fifth : ℕ := 2 * tomatoes_each_fourth_fifth_plant

def tomatoes_each_sixth_seventh_plant : ℕ := (3 * tomatoes_combined_first_two) / 2

def tomatoes_combined_sixth_seventh : ℕ := 2 * tomatoes_each_sixth_seventh_plant

def total_tomatoes : ℕ := tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant + tomatoes_combined_fourth_fifth + tomatoes_combined_sixth_seventh

theorem mabel_total_tomatoes : total_tomatoes = 315 :=
by
  sorry

end NUMINAMATH_GPT_mabel_total_tomatoes_l2305_230510


namespace NUMINAMATH_GPT_equation_of_line_l2305_230573

-- Define the points P and Q
def P : (ℝ × ℝ) := (3, 2)
def Q : (ℝ × ℝ) := (4, 7)

-- Prove that the equation of the line passing through points P and Q is 5x - y - 13 = 0
theorem equation_of_line : ∃ (A B C : ℝ), A = 5 ∧ B = -1 ∧ C = -13 ∧
  ∀ x y : ℝ, (y - 2) / (7 - 2) = (x - 3) / (4 - 3) → 5 * x - y - 13 = 0 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_l2305_230573


namespace NUMINAMATH_GPT_max_subset_no_ap_l2305_230525

theorem max_subset_no_ap (n : ℕ) (H : n ≥ 4) :
  ∃ (s : Finset ℝ), (s.card ≥ ⌊Real.sqrt (2 * n / 3)⌋₊ + 1) ∧
  ∀ (a b c : ℝ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → a ≠ c → b ≠ c → (a, b, c) ≠ (a + b - c, b, c) :=
sorry

end NUMINAMATH_GPT_max_subset_no_ap_l2305_230525


namespace NUMINAMATH_GPT_distinct_arrangements_l2305_230545

-- Defining the conditions as constants
def num_women : ℕ := 9
def num_men : ℕ := 3
def total_slots : ℕ := num_women + num_men

-- Using the combination formula directly as part of the statement
theorem distinct_arrangements : Nat.choose total_slots num_men = 220 := by
  sorry

end NUMINAMATH_GPT_distinct_arrangements_l2305_230545


namespace NUMINAMATH_GPT_yearly_profit_l2305_230522

variable (num_subletters : ℕ) (rent_per_subletter_per_month rent_per_month : ℕ)

theorem yearly_profit (h1 : num_subletters = 3)
                     (h2 : rent_per_subletter_per_month = 400)
                     (h3 : rent_per_month = 900) :
  12 * (num_subletters * rent_per_subletter_per_month - rent_per_month) = 3600 :=
by
  sorry

end NUMINAMATH_GPT_yearly_profit_l2305_230522


namespace NUMINAMATH_GPT_det_A_eq_6_l2305_230562

open Matrix

variables {R : Type*} [Field R]

def A (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![a, 2], ![-3, d]]

def B (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![2 * a, 1], ![-1, d]]

noncomputable def B_inv (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  let detB := (2 * a * d + 1)
  ![![d / detB, -1 / detB], ![1 / detB, (2 * a) / detB]]

theorem det_A_eq_6 (a d : R) (hB_inv : (A a d) + (B_inv a d) = 0) : det (A a d) = 6 :=
  sorry

end NUMINAMATH_GPT_det_A_eq_6_l2305_230562


namespace NUMINAMATH_GPT_range_of_a_l2305_230512

-- Let us define the problem conditions and statement in Lean
theorem range_of_a
  (a : ℝ)
  (h : ∀ x y : ℝ, x < y → (3 - a)^x > (3 - a)^y) :
  2 < a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2305_230512


namespace NUMINAMATH_GPT_students_per_table_correct_l2305_230553

-- Define the number of tables and students
def num_tables := 34
def num_students := 204

-- Define x as the number of students per table
def students_per_table := 6

-- State the theorem
theorem students_per_table_correct : num_students / num_tables = students_per_table :=
by
  sorry

end NUMINAMATH_GPT_students_per_table_correct_l2305_230553


namespace NUMINAMATH_GPT_system_solution_l2305_230513

theorem system_solution (x y z : ℝ) :
    x + y + z = 2 ∧ 
    x^2 + y^2 + z^2 = 26 ∧
    x^3 + y^3 + z^3 = 38 →
    (x = 1 ∧ y = 4 ∧ z = -3) ∨
    (x = 1 ∧ y = -3 ∧ z = 4) ∨
    (x = 4 ∧ y = 1 ∧ z = -3) ∨
    (x = 4 ∧ y = -3 ∧ z = 1) ∨
    (x = -3 ∧ y = 1 ∧ z = 4) ∨
    (x = -3 ∧ y = 4 ∧ z = 1) := by
  sorry

end NUMINAMATH_GPT_system_solution_l2305_230513


namespace NUMINAMATH_GPT_yolanda_walking_rate_correct_l2305_230542

-- Definitions and conditions
def distance_XY : ℕ := 65
def bobs_walking_rate : ℕ := 7
def bobs_distance_when_met : ℕ := 35
def yolanda_start_time (t: ℕ) : ℕ := t + 1 -- Yolanda starts walking 1 hour earlier

-- Yolanda's walking rate calculation
def yolandas_walking_rate : ℕ := 5

theorem yolanda_walking_rate_correct { time_bob_walked : ℕ } 
  (h1 : distance_XY = 65)
  (h2 : bobs_walking_rate = 7)
  (h3 : bobs_distance_when_met = 35) 
  (h4 : time_bob_walked = bobs_distance_when_met / bobs_walking_rate)
  (h5 : yolanda_start_time time_bob_walked = 6) -- since bob walked 5 hours, yolanda walked 6 hours
  (h6 : distance_XY - bobs_distance_when_met = 30) :
  yolandas_walking_rate = ((distance_XY - bobs_distance_when_met) / yolanda_start_time time_bob_walked) := 
sorry

end NUMINAMATH_GPT_yolanda_walking_rate_correct_l2305_230542


namespace NUMINAMATH_GPT_boards_nailing_l2305_230580

variables {x y a b : ℕ} 

theorem boards_nailing :
  (2 * x + 3 * y = 87) ∧
  (3 * a + 5 * b = 94) →
  (x + y = 30) ∧ (a + b = 30) :=
by
  sorry

end NUMINAMATH_GPT_boards_nailing_l2305_230580


namespace NUMINAMATH_GPT_diff_12_358_7_2943_l2305_230575

theorem diff_12_358_7_2943 : 12.358 - 7.2943 = 5.0637 :=
by
  -- Proof is not required, so we put sorry
  sorry

end NUMINAMATH_GPT_diff_12_358_7_2943_l2305_230575


namespace NUMINAMATH_GPT_derivative_log_base2_l2305_230565

noncomputable def log_base2 (x : ℝ) := Real.log x / Real.log 2

theorem derivative_log_base2 (x : ℝ) (h : x > 0) : 
  deriv (fun x => log_base2 x) x = 1 / (x * Real.log 2) :=
by
  sorry

end NUMINAMATH_GPT_derivative_log_base2_l2305_230565


namespace NUMINAMATH_GPT_minimum_value_of_f_ge_7_l2305_230572

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem minimum_value_of_f_ge_7 {x : ℝ} (hx : x > 0) : f x ≥ 7 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_ge_7_l2305_230572


namespace NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l2305_230518

theorem tenth_term_arithmetic_sequence (a d : ℕ) 
  (h1 : a + 2 * d = 10) 
  (h2 : a + 5 * d = 16) : 
  a + 9 * d = 24 := 
by 
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l2305_230518


namespace NUMINAMATH_GPT_students_at_start_of_year_l2305_230569

variable (S : ℕ)

def initial_students := S
def students_left := 6
def students_new := 42
def end_year_students := 47

theorem students_at_start_of_year :
  initial_students + (students_new - students_left) = end_year_students → initial_students = 11 :=
by
  sorry

end NUMINAMATH_GPT_students_at_start_of_year_l2305_230569


namespace NUMINAMATH_GPT_sum_of_first_33_terms_arith_seq_l2305_230539

noncomputable def sum_arith_prog (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_33_terms_arith_seq :
  ∃ (a_1 d : ℝ), (4 * a_1 + 64 * d = 28) → (sum_arith_prog a_1 d 33 = 231) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_33_terms_arith_seq_l2305_230539


namespace NUMINAMATH_GPT_arrange_numbers_l2305_230596

theorem arrange_numbers (x y z : ℝ) (h1 : x = 20.8) (h2 : y = 0.82) (h3 : z = Real.log 20.8) : z < y ∧ y < x :=
by
  sorry

end NUMINAMATH_GPT_arrange_numbers_l2305_230596


namespace NUMINAMATH_GPT_find_y_from_equation_l2305_230546

theorem find_y_from_equation (y : ℕ) 
  (h : (12 ^ 2) * (6 ^ 3) / y = 72) : 
  y = 432 :=
  sorry

end NUMINAMATH_GPT_find_y_from_equation_l2305_230546


namespace NUMINAMATH_GPT_composite_quotient_l2305_230559

def first_eight_composites := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) := l.foldl (· * ·) 1

theorem composite_quotient :
  let numerator := product first_eight_composites
  let denominator := product next_eight_composites
  numerator / denominator = (1 : ℚ)/(1430 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_composite_quotient_l2305_230559


namespace NUMINAMATH_GPT_melanie_attended_games_l2305_230503

/-- Melanie attended 5 football games if there were 12 total games and she missed 7. -/
theorem melanie_attended_games (totalGames : ℕ) (missedGames : ℕ) (h₁ : totalGames = 12) (h₂ : missedGames = 7) :
  totalGames - missedGames = 5 := 
sorry

end NUMINAMATH_GPT_melanie_attended_games_l2305_230503
