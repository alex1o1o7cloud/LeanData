import Mathlib

namespace NUMINAMATH_GPT_intersection_x_value_l930_93079

theorem intersection_x_value :
  ∀ x y: ℝ,
    (y = 3 * x - 15) ∧ (3 * x + y = 120) → x = 22.5 := by
  sorry

end NUMINAMATH_GPT_intersection_x_value_l930_93079


namespace NUMINAMATH_GPT_set_range_of_three_numbers_l930_93078

theorem set_range_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : (a + b + c) / 3 = 6) 
(h4 : b = 6) (h5 : c = 10) : c - a = 8 := by
  sorry

end NUMINAMATH_GPT_set_range_of_three_numbers_l930_93078


namespace NUMINAMATH_GPT_sally_nickels_count_l930_93022

theorem sally_nickels_count (original_nickels dad_nickels mom_nickels : ℕ) 
    (h1: original_nickels = 7) 
    (h2: dad_nickels = 9) 
    (h3: mom_nickels = 2) 
    : original_nickels + dad_nickels + mom_nickels = 18 :=
by
  sorry

end NUMINAMATH_GPT_sally_nickels_count_l930_93022


namespace NUMINAMATH_GPT_extremum_f_range_a_for_no_zeros_l930_93073

noncomputable def f (a b x : ℝ) : ℝ :=
  (a * (x - 1) + b * Real.exp x) / Real.exp x

theorem extremum_f (a b : ℝ) (h_a_ne_zero : a ≠ 0) :
  (∃ (x : ℝ), a = -1 ∧ b = 0 ∧ f a b x = -1 / Real.exp 2) := sorry

theorem range_a_for_no_zeros (a : ℝ) :
  (∀ x : ℝ, a * x - a + Real.exp x ≠ 0) ↔ (-Real.exp 2 < a ∧ a < 0) := sorry

end NUMINAMATH_GPT_extremum_f_range_a_for_no_zeros_l930_93073


namespace NUMINAMATH_GPT_rectangular_prism_diagonals_l930_93089

structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)
  (length : ℝ)
  (height : ℝ)
  (width : ℝ)
  (length_ne_height : length ≠ height)
  (height_ne_width : height ≠ width)
  (width_ne_length : width ≠ length)

def diagonals (rp : RectangularPrism) : ℕ :=
  let face_diagonals := 12
  let space_diagonals := 4
  face_diagonals + space_diagonals

theorem rectangular_prism_diagonals (rp : RectangularPrism) :
  rp.faces = 6 →
  rp.edges = 12 →
  rp.vertices = 8 →
  diagonals rp = 16 ∧ 4 = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rectangular_prism_diagonals_l930_93089


namespace NUMINAMATH_GPT_tangent_line_is_tangent_l930_93076

noncomputable def func1 (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def func2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem tangent_line_is_tangent
  (a : ℝ) (h_tangent : ∃ x₀ : ℝ, func2 a x₀ = 2 * x₀ ∧ (deriv (func2 a) x₀ = 2))
  (deriv_eq : deriv func1 1 = 2)
  : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_is_tangent_l930_93076


namespace NUMINAMATH_GPT_max_prime_factors_of_c_l930_93007

-- Definitions of conditions
variables (c d : ℕ)
variable (prime_factor_count : ℕ → ℕ)
variable (gcd : ℕ → ℕ → ℕ)
variable (lcm : ℕ → ℕ → ℕ)

-- Conditions
axiom gcd_condition : prime_factor_count (gcd c d) = 11
axiom lcm_condition : prime_factor_count (lcm c d) = 44
axiom fewer_prime_factors : prime_factor_count c < prime_factor_count d

-- Proof statement
theorem max_prime_factors_of_c : prime_factor_count c ≤ 27 := 
sorry

end NUMINAMATH_GPT_max_prime_factors_of_c_l930_93007


namespace NUMINAMATH_GPT_smallest_n_exists_l930_93024

theorem smallest_n_exists :
  ∃ (a1 a2 a3 a4 a5 : ℤ), a1 + a2 + a3 + a4 + a5 = 1990 ∧ a1 * a2 * a3 * a4 * a5 = 1990 :=
sorry

end NUMINAMATH_GPT_smallest_n_exists_l930_93024


namespace NUMINAMATH_GPT_range_of_m_l930_93056

-- Define the conditions for p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
  (x₁^2 + 2 * m * x₁ + 1 = 0) ∧ (x₂^2 + 2 * m * x₂ + 1 = 0)

def q (m : ℝ) : Prop := ¬ ∃ x : ℝ, x^2 + 2 * (m-2) * x - 3 * m + 10 = 0

-- The main theorem
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ 
  (m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)) := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l930_93056


namespace NUMINAMATH_GPT_solution_of_abs_square_inequality_l930_93087

def solution_set := {x : ℝ | (1 ≤ x ∧ x ≤ 3) ∨ x = -2}

theorem solution_of_abs_square_inequality (x : ℝ) :
  (abs (x^2 - 4) ≤ x + 2) ↔ (x ∈ solution_set) :=
by
  sorry

end NUMINAMATH_GPT_solution_of_abs_square_inequality_l930_93087


namespace NUMINAMATH_GPT_ratio_s_to_t_l930_93052

theorem ratio_s_to_t (b : ℝ) (s t : ℝ)
  (h1 : s = -b / 10)
  (h2 : t = -b / 6) :
  s / t = 3 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_s_to_t_l930_93052


namespace NUMINAMATH_GPT_symmetric_line_l930_93067

theorem symmetric_line (x y : ℝ) : 
  (∀ (x y  : ℝ), 2 * x + y - 1 = 0) ∧ (∀ (x  : ℝ), x = 1) → (2 * x - y - 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_l930_93067


namespace NUMINAMATH_GPT_balls_in_each_package_l930_93096

theorem balls_in_each_package (x : ℕ) (h : 21 * x = 399) : x = 19 :=
by
  sorry

end NUMINAMATH_GPT_balls_in_each_package_l930_93096


namespace NUMINAMATH_GPT_parabola_no_real_intersection_l930_93021

theorem parabola_no_real_intersection (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -4) (h₃ : c = 5) :
  ∀ (x : ℝ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_parabola_no_real_intersection_l930_93021


namespace NUMINAMATH_GPT_quotient_of_a_by_b_l930_93044

-- Definitions based on given conditions
def a : ℝ := 0.0204
def b : ℝ := 17

-- Statement to be proven
theorem quotient_of_a_by_b : a / b = 0.0012 := 
by
  sorry

end NUMINAMATH_GPT_quotient_of_a_by_b_l930_93044


namespace NUMINAMATH_GPT_jersey_sum_adjacent_gt_17_l930_93088

theorem jersey_sum_adjacent_gt_17 (a : ℕ → ℕ) (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ n, 0 < a n ∧ a n ≤ 10) (h_circle : ∀ n, a n = a (n % 10)) :
  ∃ n, a n + a (n+1) + a (n+2) > 17 :=
by
  sorry

end NUMINAMATH_GPT_jersey_sum_adjacent_gt_17_l930_93088


namespace NUMINAMATH_GPT_smallest_divisor_of_2880_that_gives_perfect_square_is_5_l930_93000

theorem smallest_divisor_of_2880_that_gives_perfect_square_is_5 :
  (∃ x : ℕ, x ≠ 0 ∧ 2880 % x = 0 ∧ (∃ y : ℕ, 2880 / x = y * y) ∧ x = 5) := by
  sorry

end NUMINAMATH_GPT_smallest_divisor_of_2880_that_gives_perfect_square_is_5_l930_93000


namespace NUMINAMATH_GPT_delaney_bus_miss_theorem_l930_93053

def delaneyMissesBus : Prop :=
  let busDeparture := 8 * 60               -- bus departure time in minutes (8:00 a.m.)
  let travelTime := 30                     -- travel time in minutes
  let departureTime := 7 * 60 + 50         -- departure time from home in minutes (7:50 a.m.)
  let arrivalTime := departureTime + travelTime -- arrival time at the pick-up point
  arrivalTime - busDeparture = 20 -- he misses the bus by 20 minutes

theorem delaney_bus_miss_theorem : delaneyMissesBus := sorry

end NUMINAMATH_GPT_delaney_bus_miss_theorem_l930_93053


namespace NUMINAMATH_GPT_total_weight_is_correct_l930_93060

def siblings_suitcases : Nat := 1 + 2 + 3 + 4 + 5 + 6
def weight_per_sibling_suitcase : Nat := 10
def total_weight_siblings : Nat := siblings_suitcases * weight_per_sibling_suitcase

def parents : Nat := 2
def suitcases_per_parent : Nat := 3
def weight_per_parent_suitcase : Nat := 12
def total_weight_parents : Nat := parents * suitcases_per_parent * weight_per_parent_suitcase

def grandparents : Nat := 2
def suitcases_per_grandparent : Nat := 2
def weight_per_grandparent_suitcase : Nat := 8
def total_weight_grandparents : Nat := grandparents * suitcases_per_grandparent * weight_per_grandparent_suitcase

def other_relatives_suitcases : Nat := 8
def weight_per_other_relatives_suitcase : Nat := 15
def total_weight_other_relatives : Nat := other_relatives_suitcases * weight_per_other_relatives_suitcase

def total_weight_all_suitcases : Nat := total_weight_siblings + total_weight_parents + total_weight_grandparents + total_weight_other_relatives

theorem total_weight_is_correct : total_weight_all_suitcases = 434 := by {
  sorry
}

end NUMINAMATH_GPT_total_weight_is_correct_l930_93060


namespace NUMINAMATH_GPT_farmer_plough_rate_l930_93080

theorem farmer_plough_rate (x : ℝ) (h1 : 85 * ((1400 / x) + 2) + 40 = 1400) : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_farmer_plough_rate_l930_93080


namespace NUMINAMATH_GPT_solve_for_2a_plus_b_l930_93061

variable (a b : ℝ)

theorem solve_for_2a_plus_b (h1 : 4 * a ^ 2 - b ^ 2 = 12) (h2 : 2 * a - b = 4) : 2 * a + b = 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_2a_plus_b_l930_93061


namespace NUMINAMATH_GPT_number_13_on_top_after_folds_l930_93020

/-
A 5x5 grid of numbers from 1 to 25 with the following sequence of folds:
1. Fold along the diagonal from bottom-left to top-right
2. Fold the left half over the right half
3. Fold the top half over the bottom half
4. Fold the bottom half over the top half
Prove that the number 13 ends up on top after all folds.
-/

def grid := (⟨ 5, 5 ⟩ : Nat × Nat)

def initial_grid : ℕ → ℕ := λ n => if 1 ≤ n ∧ n ≤ 25 then n else 0

def fold_diagonal (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 1 fold

def fold_left_over_right (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 2 fold

def fold_top_over_bottom (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 3 fold

def fold_bottom_over_top (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 4 fold

theorem number_13_on_top_after_folds : (fold_bottom_over_top (fold_top_over_bottom (fold_left_over_right (fold_diagonal initial_grid)))) 13 = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_13_on_top_after_folds_l930_93020


namespace NUMINAMATH_GPT_simplify_fraction_l930_93091

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l930_93091


namespace NUMINAMATH_GPT_pool_filling_time_l930_93027

theorem pool_filling_time :
  let pool_capacity := 12000 -- in cubic meters
  let first_valve_time := 120 -- in minutes
  let first_valve_rate := pool_capacity / first_valve_time -- in cubic meters per minute
  let second_valve_rate := first_valve_rate + 50 -- in cubic meters per minute
  let combined_rate := first_valve_rate + second_valve_rate -- in cubic meters per minute
  let time_to_fill := pool_capacity / combined_rate -- in minutes
  time_to_fill = 48 :=
by
  sorry

end NUMINAMATH_GPT_pool_filling_time_l930_93027


namespace NUMINAMATH_GPT_not_sufficient_not_necessary_l930_93002

theorem not_sufficient_not_necessary (a : ℝ) :
  ¬ ((a^2 > 1) → (1/a > 0)) ∧ ¬ ((1/a > 0) → (a^2 > 1)) := sorry

end NUMINAMATH_GPT_not_sufficient_not_necessary_l930_93002


namespace NUMINAMATH_GPT_total_amount_spent_on_cookies_l930_93028

def days_in_april : ℕ := 30
def cookies_per_day : ℕ := 3
def cost_per_cookie : ℕ := 18

theorem total_amount_spent_on_cookies : days_in_april * cookies_per_day * cost_per_cookie = 1620 := by
  sorry

end NUMINAMATH_GPT_total_amount_spent_on_cookies_l930_93028


namespace NUMINAMATH_GPT_distance_at_1_5_l930_93058

def total_distance : ℝ := 174
def speed : ℝ := 60
def travel_time (x : ℝ) : ℝ := total_distance - speed * x

theorem distance_at_1_5 :
  travel_time 1.5 = 84 := by
  sorry

end NUMINAMATH_GPT_distance_at_1_5_l930_93058


namespace NUMINAMATH_GPT_pages_left_in_pad_l930_93031

-- Definitions from conditions
def total_pages : ℕ := 120
def science_project_pages (total : ℕ) : ℕ := total * 25 / 100
def math_homework_pages : ℕ := 10

-- Proving the final number of pages left
theorem pages_left_in_pad :
  let remaining_pages_after_usage := total_pages - science_project_pages total_pages - math_homework_pages
  let pages_left_after_art_project := remaining_pages_after_usage / 2
  pages_left_after_art_project = 40 :=
by
  sorry

end NUMINAMATH_GPT_pages_left_in_pad_l930_93031


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l930_93009

/-- Given a boat's speed along the stream and against the stream, prove its speed in still water. -/
theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11)
  (h2 : b - s = 5) : b = 8 :=
sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l930_93009


namespace NUMINAMATH_GPT_sum_series_l930_93034

theorem sum_series :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_sum_series_l930_93034


namespace NUMINAMATH_GPT_inequality_transitive_l930_93082

theorem inequality_transitive (a b c : ℝ) (h : a < b) (h' : b < c) : a - c < b - c :=
by
  sorry

end NUMINAMATH_GPT_inequality_transitive_l930_93082


namespace NUMINAMATH_GPT_number_of_teams_l930_93006

theorem number_of_teams (n : ℕ) (h1 : ∀ k, k = 10) (h2 : n * 10 * (n - 1) / 2 = 1900) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_teams_l930_93006


namespace NUMINAMATH_GPT_dad_strawberries_weight_proof_l930_93043

/-
Conditions:
1. total_weight (the combined weight of Marco's and his dad's strawberries) is 23 pounds.
2. marco_weight (the weight of Marco's strawberries) is 14 pounds.
We need to prove that dad_weight (the weight of dad's strawberries) is 9 pounds.
-/

def total_weight : ℕ := 23
def marco_weight : ℕ := 14

def dad_weight : ℕ := total_weight - marco_weight

theorem dad_strawberries_weight_proof : dad_weight = 9 := by
  sorry

end NUMINAMATH_GPT_dad_strawberries_weight_proof_l930_93043


namespace NUMINAMATH_GPT_sum_of_solutions_l930_93005

theorem sum_of_solutions (x : ℝ) :
  (∀ x : ℝ, x^3 + x^2 - 6*x - 20 = 4*x + 24) →
  let polynomial := (x^3 + x^2 - 10*x - 44);
  (polynomial = 0) →
  let a := 1;
  let b := 1;
  -b/a = -1 :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_l930_93005


namespace NUMINAMATH_GPT_golf_balls_dozen_count_l930_93054

theorem golf_balls_dozen_count (n d : Nat) (h1 : n = 108) (h2 : d = 12) : n / d = 9 :=
by
  sorry

end NUMINAMATH_GPT_golf_balls_dozen_count_l930_93054


namespace NUMINAMATH_GPT_remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l930_93008

theorem remainder_of_9_6_plus_8_7_plus_7_8_mod_7 : (9^6 + 8^7 + 7^8) % 7 = 2 := 
by sorry

end NUMINAMATH_GPT_remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l930_93008


namespace NUMINAMATH_GPT_Lee_charge_per_lawn_l930_93068

theorem Lee_charge_per_lawn
  (x : ℝ)
  (mowed_lawns : ℕ)
  (total_earned : ℝ)
  (tips : ℝ)
  (tip_amount : ℝ)
  (num_customers_tipped : ℕ)
  (earnings_from_mowing : ℝ)
  (total_earning_with_tips : ℝ) :
  mowed_lawns = 16 →
  total_earned = 558 →
  num_customers_tipped = 3 →
  tip_amount = 10 →
  tips = num_customers_tipped * tip_amount →
  earnings_from_mowing = mowed_lawns * x →
  total_earning_with_tips = earnings_from_mowing + tips →
  total_earning_with_tips = total_earned →
  x = 33 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_Lee_charge_per_lawn_l930_93068


namespace NUMINAMATH_GPT_inequality_abc_l930_93026

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
    a^2 + b^2 + c^2 + 3 ≥ (1 / a) + (1 / b) + (1 / c) + a + b + c :=
sorry

end NUMINAMATH_GPT_inequality_abc_l930_93026


namespace NUMINAMATH_GPT_initial_liquid_A_quantity_l930_93051

theorem initial_liquid_A_quantity
  (x : ℝ)
  (init_A init_B init_C : ℝ)
  (removed_A removed_B removed_C : ℝ)
  (added_B added_C : ℝ)
  (new_A new_B new_C : ℝ)
  (h1 : init_A / init_B = 7 / 5)
  (h2 : init_A / init_C = 7 / 3)
  (h3 : init_A + init_B + init_C = 15 * x)
  (h4 : removed_A = 7 / 15 * 9)
  (h5 : removed_B = 5 / 15 * 9)
  (h6 : removed_C = 3 / 15 * 9)
  (h7 : new_A = init_A - removed_A)
  (h8 : new_B = init_B - removed_B + added_B)
  (h9 : new_C = init_C - removed_C + added_C)
  (h10 : new_A / (new_B + new_C) = 7 / 10)
  (h11 : added_B = 6)
  (h12 : added_C = 3) : 
  init_A = 35.7 :=
sorry

end NUMINAMATH_GPT_initial_liquid_A_quantity_l930_93051


namespace NUMINAMATH_GPT_evaluate_f_at_2_l930_93023

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := 3 * x^6 - 2 * x^5 + x^3 + 1

theorem evaluate_f_at_2 : f 2 = 34 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_evaluate_f_at_2_l930_93023


namespace NUMINAMATH_GPT_lighter_dog_weight_l930_93011

theorem lighter_dog_weight
  (x y z : ℕ)
  (h1 : x + y + z = 36)
  (h2 : y + z = 3 * x)
  (h3 : x + z = 2 * y) :
  x = 9 :=
by
  sorry

end NUMINAMATH_GPT_lighter_dog_weight_l930_93011


namespace NUMINAMATH_GPT_line_sum_slope_intercept_l930_93069

theorem line_sum_slope_intercept (m b : ℝ) (x y : ℝ)
  (hm : m = 3)
  (hpoint : (x, y) = (-2, 4))
  (heq : y = m * x + b) :
  m + b = 13 :=
by
  sorry

end NUMINAMATH_GPT_line_sum_slope_intercept_l930_93069


namespace NUMINAMATH_GPT_geometric_sequence_11th_term_l930_93025

theorem geometric_sequence_11th_term (a r : ℝ) (h₁ : a * r ^ 4 = 8) (h₂ : a * r ^ 7 = 64) : 
  a * r ^ 10 = 512 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_11th_term_l930_93025


namespace NUMINAMATH_GPT_quadratic_discriminant_l930_93018

theorem quadratic_discriminant : 
  let a := 4
  let b := -6
  let c := 9
  (b^2 - 4 * a * c = -108) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_l930_93018


namespace NUMINAMATH_GPT_ellipse_area_quadrants_eq_zero_l930_93010

theorem ellipse_area_quadrants_eq_zero 
(E : Type)
(x y : E → ℝ) 
(h_ellipse : ∀ (x y : ℝ), (x - 19)^2 / (19 * 1998) + (y - 98)^2 / (98 * 1998) = 1998) 
(R1 R2 R3 R4 : ℝ)
(H1 : ∀ (R1 R2 R3 R4 : ℝ), R1 = R_ellipse / 4 ∧ R2 = R_ellipse / 4 ∧ R3 = R_ellipse / 4 ∧ R4 = R_ellipse / 4)
: R1 - R2 + R3 - R4 = 0 := 
by 
sorry

end NUMINAMATH_GPT_ellipse_area_quadrants_eq_zero_l930_93010


namespace NUMINAMATH_GPT_initial_pennies_indeterminate_l930_93030

-- Conditions
def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def mom_nickels : ℕ := 2
def total_nickels_now : ℕ := 18

-- Proof problem statement
theorem initial_pennies_indeterminate :
  ∀ (initial_nickels dad_nickels mom_nickels total_nickels_now : ℕ), 
  initial_nickels = 7 → dad_nickels = 9 → mom_nickels = 2 → total_nickels_now = 18 → 
  (∃ (initial_pennies : ℕ), true) → false :=
by
  sorry

end NUMINAMATH_GPT_initial_pennies_indeterminate_l930_93030


namespace NUMINAMATH_GPT_find_larger_number_l930_93090

variable (x y : ℕ)

theorem find_larger_number (h1 : x = 7) (h2 : x + y = 15) : y = 8 := by
  sorry

end NUMINAMATH_GPT_find_larger_number_l930_93090


namespace NUMINAMATH_GPT_largest_n_l930_93039

noncomputable def a (n : ℕ) (x : ℤ) : ℤ := 2 + (n - 1) * x
noncomputable def b (n : ℕ) (y : ℤ) : ℤ := 3 + (n - 1) * y

theorem largest_n {n : ℕ} (x y : ℤ) :
  a 1 x = 2 ∧ b 1 y = 3 ∧ 3 * a 2 x < 2 * b 2 y ∧ a n x * b n y = 4032 →
  n = 367 :=
sorry

end NUMINAMATH_GPT_largest_n_l930_93039


namespace NUMINAMATH_GPT_relay_team_order_count_l930_93038

theorem relay_team_order_count :
  ∃ (orders : ℕ), orders = 6 :=
by
  let team_members := 4
  let remaining_members := team_members - 1  -- Excluding Lisa
  let first_lap_choices := remaining_members.choose 3  -- Choices for the first lap
  let third_lap_choices := (remaining_members - 1).choose 2  -- Choices for the third lap
  let fourth_lap_choices := (remaining_members - 2).choose 1  -- The last remaining member choices
  have orders := first_lap_choices * third_lap_choices * fourth_lap_choices
  use orders
  sorry

end NUMINAMATH_GPT_relay_team_order_count_l930_93038


namespace NUMINAMATH_GPT_amy_l930_93035

theorem amy's_speed (a b : ℝ) (s : ℝ) 
  (h1 : ∀ (major minor : ℝ), major = 2 * minor) 
  (h2 : ∀ (w : ℝ), w = 4) 
  (h3 : ∀ (t_diff : ℝ), t_diff = 48) 
  (h4 : 2 * a + 2 * Real.pi * Real.sqrt ((4 * b^2 + b^2) / 2) - (2 * a + 2 * Real.pi * Real.sqrt (((2 * b + 8)^2 + (b + 4)^2) / 2)) = 48 * s) :
  s = Real.pi / 2 := sorry

end NUMINAMATH_GPT_amy_l930_93035


namespace NUMINAMATH_GPT_bird_families_difference_l930_93040

-- Define the conditions
def bird_families_to_africa : ℕ := 47
def bird_families_to_asia : ℕ := 94

-- The proof statement
theorem bird_families_difference : (bird_families_to_asia - bird_families_to_africa = 47) :=
by
  sorry

end NUMINAMATH_GPT_bird_families_difference_l930_93040


namespace NUMINAMATH_GPT_min_sugar_l930_93098

theorem min_sugar (f s : ℝ) (h₁ : f ≥ 8 + (3/4) * s) (h₂ : f ≤ 2 * s) : s ≥ 32 / 5 :=
sorry

end NUMINAMATH_GPT_min_sugar_l930_93098


namespace NUMINAMATH_GPT_bob_is_47_5_l930_93072

def bob_age (a b : ℝ) := b = 3 * a - 20
def sum_of_ages (a b : ℝ) := b + a = 70

theorem bob_is_47_5 (a b : ℝ) (h1 : bob_age a b) (h2 : sum_of_ages a b) : b = 47.5 :=
by
  sorry

end NUMINAMATH_GPT_bob_is_47_5_l930_93072


namespace NUMINAMATH_GPT_largest_integer_value_of_x_l930_93032

theorem largest_integer_value_of_x (x : ℤ) (h : 8 - 5 * x > 22) : x ≤ -3 :=
sorry

end NUMINAMATH_GPT_largest_integer_value_of_x_l930_93032


namespace NUMINAMATH_GPT_certain_event_abs_nonneg_l930_93063

theorem certain_event_abs_nonneg (x : ℝ) : |x| ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_certain_event_abs_nonneg_l930_93063


namespace NUMINAMATH_GPT_minEmployees_correct_l930_93045

noncomputable def minEmployees (seaTurtles birdMigration bothTurtlesBirds turtlesPlants allThree : ℕ) : ℕ :=
  let onlySeaTurtles := seaTurtles - (bothTurtlesBirds + turtlesPlants - allThree)
  let onlyBirdMigration := birdMigration - (bothTurtlesBirds + allThree - turtlesPlants)
  onlySeaTurtles + onlyBirdMigration + bothTurtlesBirds + turtlesPlants + allThree

theorem minEmployees_correct :
  minEmployees 120 90 30 50 15 = 245 := by
  sorry

end NUMINAMATH_GPT_minEmployees_correct_l930_93045


namespace NUMINAMATH_GPT_impossible_fractions_l930_93066

theorem impossible_fractions (a b c r s t : ℕ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t)
  (h1 : a * b + 1 = r ^ 2) (h2 : a * c + 1 = s ^ 2) (h3 : b * c + 1 = t ^ 2) :
  ¬ (∃ (k1 k2 k3 : ℕ), rt / s = k1 ∧ rs / t = k2 ∧ st / r = k3) :=
by
  sorry

end NUMINAMATH_GPT_impossible_fractions_l930_93066


namespace NUMINAMATH_GPT_negation_of_forall_ge_zero_l930_93097

theorem negation_of_forall_ge_zero :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
sorry

end NUMINAMATH_GPT_negation_of_forall_ge_zero_l930_93097


namespace NUMINAMATH_GPT_compute_fraction_l930_93001

theorem compute_fraction :
  ( (11^4 + 400) * (25^4 + 400) * (37^4 + 400) * (49^4 + 400) * (61^4 + 400) ) /
  ( (5^4 + 400) * (17^4 + 400) * (29^4 + 400) * (41^4 + 400) * (53^4 + 400) ) = 799 := 
by
  sorry

end NUMINAMATH_GPT_compute_fraction_l930_93001


namespace NUMINAMATH_GPT_ratio_product_even_odd_composite_l930_93033

theorem ratio_product_even_odd_composite :
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = (2^10) / (3^6 * 5^2 * 7) :=
by
  sorry

end NUMINAMATH_GPT_ratio_product_even_odd_composite_l930_93033


namespace NUMINAMATH_GPT_alice_bob_not_next_to_each_other_l930_93042

open Nat

theorem alice_bob_not_next_to_each_other (A B C D E : Type) :
  let arrangements := 5!
  let together := 4! * 2
  arrangements - together = 72 :=
by
  let arrangements := 5!
  let together := 4! * 2
  sorry

end NUMINAMATH_GPT_alice_bob_not_next_to_each_other_l930_93042


namespace NUMINAMATH_GPT_value_of_a_l930_93093

theorem value_of_a (a : ℝ) :
  (∃ (l1 l2 : (ℝ × ℝ × ℝ)),
   l1 = (1, -a, a) ∧ l2 = (3, 1, 2) ∧
   (∃ (m1 m2 : ℝ), 
    (m1 = (1 : ℝ) / a ∧ m2 = -3) ∧ 
    (m1 * m2 = -1))) → a = 3 :=
by sorry

end NUMINAMATH_GPT_value_of_a_l930_93093


namespace NUMINAMATH_GPT_evaluate_fraction_sum_l930_93046

-- Define the problem conditions and target equation
theorem evaluate_fraction_sum
    (p q r : ℝ)
    (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 8) :
    6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 11 / 5 := by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_sum_l930_93046


namespace NUMINAMATH_GPT_is_divisible_by_N2_l930_93048

def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def eulers_totient (n : ℕ) : ℕ :=
  Nat.totient n

theorem is_divisible_by_N2 (N1 N2 : ℕ) (h_coprime : are_coprime N1 N2) 
  (k := eulers_totient N2) : 
  (N1 ^ k - 1) % N2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_is_divisible_by_N2_l930_93048


namespace NUMINAMATH_GPT_find_n_l930_93017

def x := 3
def y := 1
def n := x - 3 * y^(x - y) + 1

theorem find_n : n = 1 :=
by
  unfold n x y
  sorry

end NUMINAMATH_GPT_find_n_l930_93017


namespace NUMINAMATH_GPT_solve_y_l930_93049

theorem solve_y :
  ∀ y : ℚ, 6 * (4 * y - 1) - 3 = 3 * (2 - 5 * y) ↔ y = 5 / 13 :=
by
  sorry

end NUMINAMATH_GPT_solve_y_l930_93049


namespace NUMINAMATH_GPT_range_of_a_l930_93037

theorem range_of_a :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 2 * x * (3 * x + a) < 1) → a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l930_93037


namespace NUMINAMATH_GPT_football_game_cost_l930_93050

theorem football_game_cost :
  ∀ (total_spent strategy_game_cost batman_game_cost football_game_cost : ℝ),
  total_spent = 35.52 →
  strategy_game_cost = 9.46 →
  batman_game_cost = 12.04 →
  total_spent - strategy_game_cost - batman_game_cost = football_game_cost →
  football_game_cost = 13.02 :=
by
  intros total_spent strategy_game_cost batman_game_cost football_game_cost h1 h2 h3 h4
  have : football_game_cost = 13.02 := sorry
  exact this

end NUMINAMATH_GPT_football_game_cost_l930_93050


namespace NUMINAMATH_GPT_no_two_right_angles_in_triangle_l930_93095

theorem no_two_right_angles_in_triangle 
  (α β γ : ℝ)
  (h1 : α + β + γ = 180) :
  ¬ (α = 90 ∧ β = 90) :=
by
  sorry

end NUMINAMATH_GPT_no_two_right_angles_in_triangle_l930_93095


namespace NUMINAMATH_GPT_other_acute_angle_in_right_triangle_l930_93070

theorem other_acute_angle_in_right_triangle (α : ℝ) (β : ℝ) (γ : ℝ) 
  (h1 : α + β + γ = 180) (h2 : γ = 90) (h3 : α = 30) : β = 60 := 
sorry

end NUMINAMATH_GPT_other_acute_angle_in_right_triangle_l930_93070


namespace NUMINAMATH_GPT_cost_price_250_l930_93059

theorem cost_price_250 (C : ℝ) (h1 : 0.90 * C = C - 0.10 * C) (h2 : 1.10 * C = C + 0.10 * C) (h3 : 1.10 * C - 0.90 * C = 50) : C = 250 := 
by
  sorry

end NUMINAMATH_GPT_cost_price_250_l930_93059


namespace NUMINAMATH_GPT_difference_sixth_seventh_l930_93074

theorem difference_sixth_seventh
  (A1 A2 A3 A4 A5 A6 A7 A8 : ℕ)
  (h_avg_8 : (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8) / 8 = 25)
  (h_avg_2 : (A1 + A2) / 2 = 20)
  (h_avg_3 : (A3 + A4 + A5) / 3 = 26)
  (h_A8 : A8 = 30)
  (h_A6_A8 : A6 = A8 - 6) :
  A7 - A6 = 4 :=
by
  sorry

end NUMINAMATH_GPT_difference_sixth_seventh_l930_93074


namespace NUMINAMATH_GPT_intersection_of_sets_l930_93065

noncomputable def setM : Set ℝ := { x | x + 1 > 0 }
noncomputable def setN : Set ℝ := { x | 2 * x - 1 < 0 }

theorem intersection_of_sets : setM ∩ setN = { x : ℝ | -1 < x ∧ x < 1 / 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l930_93065


namespace NUMINAMATH_GPT_find_varphi_l930_93014

theorem find_varphi (ϕ : ℝ) (h0 : 0 < ϕ ∧ ϕ < π / 2) :
  (∀ x₁ x₂, |(2 * Real.cos (2 * x₁)) - (2 * Real.cos (2 * x₂ - 2 * ϕ))| = 4 → 
    ∃ (x₁ x₂ : ℝ), |x₁ - x₂| = π / 6 
  ) → ϕ = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_varphi_l930_93014


namespace NUMINAMATH_GPT_mixed_number_fraction_division_and_subtraction_l930_93036

theorem mixed_number_fraction_division_and_subtraction :
  ( (11 / 6) / (11 / 4) ) - (1 / 2) = 1 / 6 := 
sorry

end NUMINAMATH_GPT_mixed_number_fraction_division_and_subtraction_l930_93036


namespace NUMINAMATH_GPT_rest_duration_per_kilometer_l930_93055

theorem rest_duration_per_kilometer
  (speed : ℕ)
  (total_distance : ℕ)
  (total_time : ℕ)
  (walking_time : ℕ := total_distance / speed * 60)  -- walking_time in minutes
  (rest_time : ℕ := total_time - walking_time)  -- total resting time in minutes
  (number_of_rests : ℕ := total_distance - 1)  -- number of rests after each kilometer
  (duration_per_rest : ℕ := rest_time / number_of_rests)
  (h1 : speed = 10)
  (h2 : total_distance = 5)
  (h3 : total_time = 50) : 
  (duration_per_rest = 5) := 
sorry

end NUMINAMATH_GPT_rest_duration_per_kilometer_l930_93055


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l930_93084

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 7 / 3) :
  S 5 / S 3 = 5 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l930_93084


namespace NUMINAMATH_GPT_large_cross_area_is_60_cm_squared_l930_93041

noncomputable def small_square_area (s : ℝ) := s * s
noncomputable def large_square_area (s : ℝ) := 4 * small_square_area s
noncomputable def small_cross_area (s : ℝ) := 5 * small_square_area s
noncomputable def large_cross_area (s : ℝ) := 5 * large_square_area s
noncomputable def remaining_area (s : ℝ) := large_cross_area s - small_cross_area s

theorem large_cross_area_is_60_cm_squared :
  ∃ (s : ℝ), remaining_area s = 45 → large_cross_area s = 60 :=
by
  sorry

end NUMINAMATH_GPT_large_cross_area_is_60_cm_squared_l930_93041


namespace NUMINAMATH_GPT_pow_mult_same_base_l930_93029

theorem pow_mult_same_base (a b : ℕ) : 10^a * 10^b = 10^(a + b) := by 
  sorry

example : 10^655 * 10^652 = 10^1307 :=
  pow_mult_same_base 655 652

end NUMINAMATH_GPT_pow_mult_same_base_l930_93029


namespace NUMINAMATH_GPT_units_digit_of_expression_l930_93083

theorem units_digit_of_expression :
  (6 * 16 * 1986 - 6 ^ 4) % 10 = 0 := 
sorry

end NUMINAMATH_GPT_units_digit_of_expression_l930_93083


namespace NUMINAMATH_GPT_largest_number_l930_93075

theorem largest_number (P Q R S T : ℕ) 
  (hP_digits_prime : ∃ p1 p2, P = 10 * p1 + p2 ∧ Prime P ∧ Prime (p1 + p2))
  (hQ_multiple_of_5 : Q % 5 = 0)
  (hR_odd_non_prime : Odd R ∧ ¬ Prime R)
  (hS_prime_square : ∃ p, Prime p ∧ S = p * p)
  (hT_mean_prime : T = (P + Q) / 2 ∧ Prime T)
  (hP_range : 10 ≤ P ∧ P ≤ 99)
  (hQ_range : 2 ≤ Q ∧ Q ≤ 19)
  (hR_range : 2 ≤ R ∧ R ≤ 19)
  (hS_range : 2 ≤ S ∧ S ≤ 19)
  (hT_range : 2 ≤ T ∧ T ≤ 19) :
  max P (max Q (max R (max S T))) = Q := 
by 
  sorry

end NUMINAMATH_GPT_largest_number_l930_93075


namespace NUMINAMATH_GPT_solve_m_l930_93086

theorem solve_m (m : ℝ) :
  (∃ x > 0, (2 * m - 4) ^ 2 = x ∧ (3 * m - 1) ^ 2 = x) →
  (m = -3 ∨ m = 1) :=
by 
  sorry

end NUMINAMATH_GPT_solve_m_l930_93086


namespace NUMINAMATH_GPT_radius_of_inscribed_sphere_l930_93015

theorem radius_of_inscribed_sphere (a b c s : ℝ)
  (h1: 2 * (a * b + a * c + b * c) = 616)
  (h2: a + b + c = 40)
  : s = Real.sqrt 246 ↔ (2 * s) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_sphere_l930_93015


namespace NUMINAMATH_GPT_explicit_formula_for_f_l930_93004

def f (k : ℕ) : ℚ :=
  if k = 1 then 4 / 3
  else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3

theorem explicit_formula_for_f (k : ℕ) (hk : k ≥ 1) : 
  (f k = if k = 1 then 4 / 3 else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3) ∧ 
  ∀ k ≥ 2, 2 * f k = f (k - 1) - k * 5^k + 2^k :=
by {
  sorry
}

end NUMINAMATH_GPT_explicit_formula_for_f_l930_93004


namespace NUMINAMATH_GPT_championship_winner_is_902_l930_93012

namespace BasketballMatch

inductive Class : Type
| c901
| c902
| c903
| c904

open Class

def A_said (champ third : Class) : Prop :=
  champ = c902 ∧ third = c904

def B_said (fourth runner_up : Class) : Prop :=
  fourth = c901 ∧ runner_up = c903

def C_said (third champ : Class) : Prop :=
  third = c903 ∧ champ = c904

def half_correct (P Q : Prop) : Prop := 
  (P ∧ ¬Q) ∨ (¬P ∧ Q)

theorem championship_winner_is_902 (A_third B_fourth B_runner_up C_third : Class) 
  (H_A : half_correct (A_said c902 A_third) (A_said A_third c902))
  (H_B : half_correct (B_said B_fourth B_runner_up) (B_said B_runner_up B_fourth))
  (H_C : half_correct (C_said C_third c904) (C_said c904 C_third)) :
  ∃ winner, winner = c902 :=
sorry

end BasketballMatch

end NUMINAMATH_GPT_championship_winner_is_902_l930_93012


namespace NUMINAMATH_GPT_find_n_l930_93062

theorem find_n :
  ∃ n : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = n ^ 5 ∧ 
  (∀ m : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = m ^ 5 → m = 144) :=
by
  sorry

end NUMINAMATH_GPT_find_n_l930_93062


namespace NUMINAMATH_GPT_fourth_person_height_l930_93077

variable (H : ℝ)
variable (height1 height2 height3 height4 : ℝ)

theorem fourth_person_height
  (h1 : height1 = H)
  (h2 : height2 = H + 2)
  (h3 : height3 = H + 4)
  (h4 : height4 = H + 10)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 78) :
  height4 = 84 :=
by
  sorry

end NUMINAMATH_GPT_fourth_person_height_l930_93077


namespace NUMINAMATH_GPT_eiffel_tower_model_height_l930_93016

theorem eiffel_tower_model_height 
  (H1 : ℝ) (W1 : ℝ) (W2 : ℝ) (H2 : ℝ)
  (h1 : H1 = 324)
  (w1 : W1 = 8000000)  -- converted 8000 tons to 8000000 kg
  (w2 : W2 = 1)
  (h_eq : (H2 / H1)^3 = W2 / W1) : 
  H2 = 1.62 :=
by
  rw [h1, w1, w2] at h_eq
  sorry

end NUMINAMATH_GPT_eiffel_tower_model_height_l930_93016


namespace NUMINAMATH_GPT_power_sum_divisible_by_five_l930_93071

theorem power_sum_divisible_by_five : 
  (3^444 + 4^333) % 5 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_power_sum_divisible_by_five_l930_93071


namespace NUMINAMATH_GPT_Hector_gumballs_l930_93003

theorem Hector_gumballs :
  ∃ (total_gumballs : ℕ)
  (gumballs_Todd : ℕ) (gumballs_Alisha : ℕ) (gumballs_Bobby : ℕ) (gumballs_remaining : ℕ),
  gumballs_Todd = 4 ∧
  gumballs_Alisha = 2 * gumballs_Todd ∧
  gumballs_Bobby = 4 * gumballs_Alisha - 5 ∧
  gumballs_remaining = 6 ∧
  total_gumballs = gumballs_Todd + gumballs_Alisha + gumballs_Bobby + gumballs_remaining ∧
  total_gumballs = 45 :=
by
  sorry

end NUMINAMATH_GPT_Hector_gumballs_l930_93003


namespace NUMINAMATH_GPT_num_new_books_not_signed_l930_93013

theorem num_new_books_not_signed (adventure_books mystery_books science_fiction_books non_fiction_books used_books signed_books : ℕ)
    (h1 : adventure_books = 13)
    (h2 : mystery_books = 17)
    (h3 : science_fiction_books = 25)
    (h4 : non_fiction_books = 10)
    (h5 : used_books = 42)
    (h6 : signed_books = 10) : 
    (adventure_books + mystery_books + science_fiction_books + non_fiction_books) - used_books - signed_books = 13 := 
by
  sorry

end NUMINAMATH_GPT_num_new_books_not_signed_l930_93013


namespace NUMINAMATH_GPT_greatest_common_ratio_l930_93092

theorem greatest_common_ratio {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : (b = (a + c) / 2 → b^2 = a * c) ∨ (c = (a + b) / 2 ∧ b = -a / 2)) :
  ∃ r : ℝ, r = -2 :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_ratio_l930_93092


namespace NUMINAMATH_GPT_shadow_length_of_flagpole_is_correct_l930_93019

noncomputable def length_of_shadow_flagpole : ℕ :=
  let h_flagpole : ℕ := 18
  let shadow_building : ℕ := 60
  let h_building : ℕ := 24
  let similar_conditions : Prop := true
  45

theorem shadow_length_of_flagpole_is_correct :
  length_of_shadow_flagpole = 45 := by
  sorry

end NUMINAMATH_GPT_shadow_length_of_flagpole_is_correct_l930_93019


namespace NUMINAMATH_GPT_quadratic_rewrite_sum_l930_93094

theorem quadratic_rewrite_sum (a b c : ℝ) (x : ℝ) :
  -3 * x^2 + 15 * x + 75 = a * (x + b)^2 + c → (a + b + c) = 88.25 :=
sorry

end NUMINAMATH_GPT_quadratic_rewrite_sum_l930_93094


namespace NUMINAMATH_GPT_distance_between_trees_l930_93081

-- Variables representing the total length of the yard and the number of trees.
variable (length_of_yard : ℕ) (number_of_trees : ℕ)

-- The given conditions
def yard_conditions (length_of_yard number_of_trees : ℕ) :=
  length_of_yard = 700 ∧ number_of_trees = 26

-- The proof statement: If the yard is 700 meters long and there are 26 trees, 
-- then the distance between two consecutive trees is 28 meters.
theorem distance_between_trees (length_of_yard : ℕ) (number_of_trees : ℕ)
  (h : yard_conditions length_of_yard number_of_trees) : 
  (length_of_yard / (number_of_trees - 1)) = 28 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l930_93081


namespace NUMINAMATH_GPT_largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l930_93099

-- Definitions based on conditions
def isProperFraction (n d : ℕ) : Prop := n < d
def isImproperFraction (n d : ℕ) : Prop := n ≥ d
def isMixedNumber (w n d : ℕ) : Prop := w > 0 ∧ isProperFraction n d

-- Fractional part is 1/9, meaning all fractions considered have part = 1/9
def fractionalPart := 1 / 9

-- Lean 4 statements to verify the correct answers
theorem largest_proper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isProperFraction n d ∧ (n, d) = (8, 9) := sorry

theorem smallest_improper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isImproperFraction n d ∧ (n, d) = (9, 9) := sorry

theorem smallest_mixed_number : ∃ w n d : ℕ, fractionalPart = n / d ∧ isMixedNumber w n d ∧ ((w, n, d) = (1, 1, 9) ∨ (w, n, d) = (10, 9)) := sorry

end NUMINAMATH_GPT_largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l930_93099


namespace NUMINAMATH_GPT_no_common_root_of_polynomials_l930_93064

theorem no_common_root_of_polynomials (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) : 
  ∀ x : ℝ, ¬ (x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_common_root_of_polynomials_l930_93064


namespace NUMINAMATH_GPT_largest_proper_divisor_condition_l930_93085

def is_proper_divisor (n k : ℕ) : Prop :=
  k > 1 ∧ k < n ∧ n % k = 0

theorem largest_proper_divisor_condition (n p : ℕ) (hp : is_proper_divisor n p) (hl : ∀ k, is_proper_divisor n k → k ≤ n / p):
  n = 12 ∨ n = 33 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_largest_proper_divisor_condition_l930_93085


namespace NUMINAMATH_GPT_ratio_first_term_l930_93047

theorem ratio_first_term (x : ℕ) (r : ℕ × ℕ) (h₀ : r = (6 - x, 7 - x)) 
        (h₁ : x ≥ 3) (h₂ : r.1 < r.2) : r.1 < 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_first_term_l930_93047


namespace NUMINAMATH_GPT_longest_collection_has_more_pages_l930_93057

noncomputable def miles_pages_per_inch := 5
noncomputable def daphne_pages_per_inch := 50
noncomputable def miles_height_inches := 240
noncomputable def daphne_height_inches := 25

noncomputable def miles_total_pages := miles_height_inches * miles_pages_per_inch
noncomputable def daphne_total_pages := daphne_height_inches * daphne_pages_per_inch

theorem longest_collection_has_more_pages :
  max miles_total_pages daphne_total_pages = 1250 := by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_longest_collection_has_more_pages_l930_93057
