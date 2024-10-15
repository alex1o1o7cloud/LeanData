import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_first_term_l710_71039

theorem geometric_sequence_first_term (a : ℕ) (r : ℕ)
    (h1 : a * r^2 = 27) 
    (h2 : a * r^3 = 81) : 
    a = 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l710_71039


namespace NUMINAMATH_GPT_total_distance_traveled_l710_71036

-- Definitions of conditions
def bess_throw_distance : ℕ := 20
def bess_throws : ℕ := 4
def holly_throw_distance : ℕ := 8
def holly_throws : ℕ := 5
def bess_effective_throw_distance : ℕ := 2 * bess_throw_distance

-- Theorem statement
theorem total_distance_traveled :
  (bess_throws * bess_effective_throw_distance + holly_throws * holly_throw_distance) = 200 := 
  by sorry

end NUMINAMATH_GPT_total_distance_traveled_l710_71036


namespace NUMINAMATH_GPT_part1_part2_i_part2_ii_l710_71015

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x * Real.log x - 1

theorem part1 (a : ℝ) (x : ℝ) : f x a + x^2 * f (1 / x) a = 0 :=
by sorry

theorem part2_i (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) : 2 < a :=
by sorry

theorem part2_ii (a : ℝ) (x1 x2 x3 : ℝ) (h : x1 < x2 ∧ x2 < x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) : x1 + x3 > 2 * a - 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_i_part2_ii_l710_71015


namespace NUMINAMATH_GPT_initial_mixture_volume_l710_71016

/--
Given:
1. A mixture initially contains 20% water.
2. When 13.333333333333334 liters of water is added, water becomes 25% of the new mixture.

Prove that the initial volume of the mixture is 200 liters.
-/
theorem initial_mixture_volume (V : ℝ) (h1 : V > 0) (h2 : 0.20 * V + 13.333333333333334 = 0.25 * (V + 13.333333333333334)) : V = 200 :=
sorry

end NUMINAMATH_GPT_initial_mixture_volume_l710_71016


namespace NUMINAMATH_GPT_geom_seq_a3_a5_product_l710_71055

-- Defining the conditions: a sequence and its sum formula
def geom_seq (a : ℕ → ℕ) := ∃ r : ℕ, ∀ n, a (n+1) = a n * r

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n-1) + a 1

-- The theorem statement
theorem geom_seq_a3_a5_product (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : geom_seq a) (h2 : sum_first_n_terms a S) : a 3 * a 5 = 16 := 
sorry

end NUMINAMATH_GPT_geom_seq_a3_a5_product_l710_71055


namespace NUMINAMATH_GPT_unique_intersection_value_k_l710_71037

theorem unique_intersection_value_k (k : ℝ) : (∀ x y: ℝ, (y = x^2) ∧ (y = 3*x + k) ↔ k = -9/4) :=
by
  sorry

end NUMINAMATH_GPT_unique_intersection_value_k_l710_71037


namespace NUMINAMATH_GPT_bus_dispatch_interval_l710_71029

-- Variables representing the speeds of Xiao Nan and the bus
variable (V_1 V_2 : ℝ)
-- The interval between the dispatch of two buses
variable (interval : ℝ)

-- Stating the conditions in Lean

-- Xiao Nan notices a bus catches up with him every 10 minutes
def cond1 : Prop := ∃ s, s = 10 * (V_1 - V_2)

-- Xiao Yu notices he encounters a bus every 5 minutes
def cond2 : Prop := ∃ s, s = 5 * (V_1 + 3 * V_2)

-- Proof statement
theorem bus_dispatch_interval (h1 : cond1 V_1 V_2) (h2 : cond2 V_1 V_2) : interval = 8 := by
  -- Proof would be provided here
  sorry

end NUMINAMATH_GPT_bus_dispatch_interval_l710_71029


namespace NUMINAMATH_GPT_sum_of_terms_in_fractional_array_l710_71033

theorem sum_of_terms_in_fractional_array :
  (∑' (r : ℕ) (c : ℕ), (1 : ℝ) / ((3 * 4) ^ r) * (1 / (4 ^ c))) = (1 / 33) := sorry

end NUMINAMATH_GPT_sum_of_terms_in_fractional_array_l710_71033


namespace NUMINAMATH_GPT_total_distance_l710_71056

variable {D : ℝ}

theorem total_distance (h1 : D / 3 > 0)
                       (h2 : (2 / 3 * D) - (1 / 6 * D) > 0)
                       (h3 : (1 / 2 * D) - (1 / 10 * D) = 180) :
    D = 450 := 
sorry

end NUMINAMATH_GPT_total_distance_l710_71056


namespace NUMINAMATH_GPT_total_balloons_l710_71028

theorem total_balloons
  (g b y r : ℕ)  -- Number of green, blue, yellow, and red balloons respectively
  (equal_groups : g = b ∧ b = y ∧ y = r)
  (anya_took : y / 2 = 84) :
  g + b + y + r = 672 := by
sorry

end NUMINAMATH_GPT_total_balloons_l710_71028


namespace NUMINAMATH_GPT_find_x_for_f_of_one_fourth_l710_71085

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
if h : x < 1 then 2^(-x) else Real.log x / Real.log 4 

-- Define the proof problem
theorem find_x_for_f_of_one_fourth : 
  ∃ x : ℝ, (f x = 1 / 4) ∧ (x = Real.sqrt 2)  :=
sorry

end NUMINAMATH_GPT_find_x_for_f_of_one_fourth_l710_71085


namespace NUMINAMATH_GPT_tangent_lines_diff_expected_l710_71006

noncomputable def tangent_lines_diff (a : ℝ) (k1 k2 : ℝ) : Prop :=
  let curve (x : ℝ) := a * x + 2 * Real.log (|x|)
  let deriv (x : ℝ) := a + 2 / x
  -- Tangent conditions at some x1 > 0 for k1
  (∃ x1 : ℝ, 0 < x1 ∧ k1 = deriv x1 ∧ curve x1 = k1 * x1)
  -- Tangent conditions at some x2 < 0 for k2
  ∧ (∃ x2 : ℝ, x2 < 0 ∧ k2 = deriv x2 ∧ curve x2 = k2 * x2)
  -- The lines' slopes relations
  ∧ k1 > k2

theorem tangent_lines_diff_expected (a k1 k2 : ℝ) (h : tangent_lines_diff a k1 k2) :
  k1 - k2 = 4 / Real.exp 1 :=
sorry

end NUMINAMATH_GPT_tangent_lines_diff_expected_l710_71006


namespace NUMINAMATH_GPT_heights_proportional_l710_71038

-- Define the problem conditions
def sides_ratio (a b c : ℕ) : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5

-- Define the heights
def heights_ratio (h1 h2 h3 : ℕ) : Prop := h1 / h2 = 20 / 15 ∧ h2 / h3 = 15 / 12

-- Problem statement: Given the sides ratio, prove the heights ratio
theorem heights_proportional {a b c h1 h2 h3 : ℕ} (h : sides_ratio a b c) :
  heights_ratio h1 h2 h3 :=
sorry

end NUMINAMATH_GPT_heights_proportional_l710_71038


namespace NUMINAMATH_GPT_ellipse_minimum_distance_point_l710_71071

theorem ellipse_minimum_distance_point :
  ∃ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1) ∧ (∀ p, x - 2 * y - 12 = 0 → dist (x, y) p ≥ dist (2, -3) p) :=
sorry

end NUMINAMATH_GPT_ellipse_minimum_distance_point_l710_71071


namespace NUMINAMATH_GPT_largest_integral_x_l710_71092

theorem largest_integral_x (x : ℤ) : (2 / 7 : ℝ) < (x / 6) ∧ (x / 6) < (7 / 9) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_integral_x_l710_71092


namespace NUMINAMATH_GPT_smallest_positive_x_l710_71082

theorem smallest_positive_x
  (x : ℕ)
  (h1 : x % 3 = 2)
  (h2 : x % 7 = 6)
  (h3 : x % 8 = 7) : x = 167 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_x_l710_71082


namespace NUMINAMATH_GPT_dave_tray_problem_l710_71031

theorem dave_tray_problem (n_trays_per_trip : ℕ) (n_trips : ℕ) (n_second_table : ℕ) : 
  (n_trays_per_trip = 9) → (n_trips = 8) → (n_second_table = 55) → 
  (n_trays_per_trip * n_trips - n_second_table = 17) :=
by
  sorry

end NUMINAMATH_GPT_dave_tray_problem_l710_71031


namespace NUMINAMATH_GPT_quadruplets_satisfy_l710_71064

-- Define the condition in the problem
def equation (x y z w : ℝ) : Prop :=
  1 + (1 / x) + (2 * (x + 1) / (x * y)) + (3 * (x + 1) * (y + 2) / (x * y * z)) + (4 * (x + 1) * (y + 2) * (z + 3) / (x * y * z * w)) = 0

-- State the theorem
theorem quadruplets_satisfy (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  equation x y z w ↔ (x = -1 ∨ y = -2 ∨ z = -3 ∨ w = -4) :=
by
  sorry

end NUMINAMATH_GPT_quadruplets_satisfy_l710_71064


namespace NUMINAMATH_GPT_inequality_solution_l710_71026

theorem inequality_solution (m : ℝ) (x : ℝ) (hm : 0 ≤ m ∧ m ≤ 1) (ineq : m * x^2 - 2 * x - m ≥ 2) : x ≤ -1 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l710_71026


namespace NUMINAMATH_GPT_solve_exponential_diophantine_equation_l710_71069

theorem solve_exponential_diophantine_equation :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 → (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_exponential_diophantine_equation_l710_71069


namespace NUMINAMATH_GPT_undefined_values_l710_71018

theorem undefined_values (a : ℝ) : a = -3 ∨ a = 3 ↔ (a^2 - 9 = 0) := sorry

end NUMINAMATH_GPT_undefined_values_l710_71018


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l710_71000

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : x - y = 20) (h2 : x * y = 9) :
  x^2 + y^2 = 418 :=
sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l710_71000


namespace NUMINAMATH_GPT_line_parallelism_theorem_l710_71022

-- Definitions of the relevant geometric conditions
variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions as hypotheses
axiom line_parallel_plane (m : Line) (α : Plane) : Prop
axiom line_in_plane (n : Line) (α : Plane) : Prop
axiom plane_intersection_line (α β : Plane) : Line
axiom line_parallel (m n : Line) : Prop

-- The problem statement in Lean 4
theorem line_parallelism_theorem 
  (h1 : line_parallel_plane m α) 
  (h2 : line_in_plane n β) 
  (h3 : plane_intersection_line α β = n) 
  (h4 : line_parallel_plane m β) : line_parallel m n :=
sorry

end NUMINAMATH_GPT_line_parallelism_theorem_l710_71022


namespace NUMINAMATH_GPT_poly_square_of_binomial_l710_71086

theorem poly_square_of_binomial (x y : ℝ) : (x + y) * (x - y) = x^2 - y^2 := 
by 
  sorry

end NUMINAMATH_GPT_poly_square_of_binomial_l710_71086


namespace NUMINAMATH_GPT_percentage_alcohol_final_l710_71076

-- Let's define the given conditions
variable (A B totalVolume : ℝ)
variable (percentAlcoholA percentAlcoholB : ℝ)
variable (approxA : ℝ)

-- Assume the conditions
axiom condition1 : percentAlcoholA = 0.20
axiom condition2 : percentAlcoholB = 0.50
axiom condition3 : totalVolume = 15
axiom condition4 : approxA = 10
axiom condition5 : A = approxA
axiom condition6 : B = totalVolume - A

-- The proof statement
theorem percentage_alcohol_final : 
  (0.20 * A + 0.50 * B) / 15 * 100 = 30 :=
by 
  -- Introduce enough structure for Lean to handle the problem.
  sorry

end NUMINAMATH_GPT_percentage_alcohol_final_l710_71076


namespace NUMINAMATH_GPT_downstream_speed_l710_71030

variable (Vu : ℝ) (Vs : ℝ)

theorem downstream_speed (h1 : Vu = 25) (h2 : Vs = 35) : (2 * Vs - Vu = 45) :=
by
  sorry

end NUMINAMATH_GPT_downstream_speed_l710_71030


namespace NUMINAMATH_GPT_positive_real_solution_l710_71008

def polynomial (x : ℝ) : ℝ := x^4 + 10*x^3 - 2*x^2 + 12*x - 9

theorem positive_real_solution (h : polynomial 1 = 0) : polynomial 1 > 0 := sorry

end NUMINAMATH_GPT_positive_real_solution_l710_71008


namespace NUMINAMATH_GPT_loss_percentage_on_first_book_l710_71093

theorem loss_percentage_on_first_book 
    (C1 C2 SP : ℝ) 
    (H1 : C1 = 210) 
    (H2 : C1 + C2 = 360) 
    (H3 : SP = 1.19 * C2) 
    (H4 : SP = 178.5) :
    ((C1 - SP) / C1) * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_on_first_book_l710_71093


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l710_71088

variable (V_m V_s : ℝ)

/-- The speed of a man in still water -/
theorem speed_of_man_in_still_water (h_downstream : 18 = (V_m + V_s) * 3)
                                     (h_upstream : 12 = (V_m - V_s) * 3) :
    V_m = 5 := 
sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l710_71088


namespace NUMINAMATH_GPT_calculate_large_exponent_l710_71005

theorem calculate_large_exponent : (1307 * 1307)^3 = 4984209203082045649 :=
by {
   sorry
}

end NUMINAMATH_GPT_calculate_large_exponent_l710_71005


namespace NUMINAMATH_GPT_find_denominators_l710_71011

theorem find_denominators (f1 f2 f3 f4 f5 f6 f7 f8 f9 : ℚ)
  (h1 : f1 = 1/3) (h2 : f2 = 1/7) (h3 : f3 = 1/9) (h4 : f4 = 1/11) (h5 : f5 = 1/33)
  (h6 : ∃ (d₁ d₂ d₃ d₄ : ℕ), f6 = 1/d₁ ∧ f7 = 1/d₂ ∧ f8 = 1/d₃ ∧ f9 = 1/d₄ ∧
    (∀ d, d ∈ [d₁, d₂, d₃, d₄] → d % 10 = 5))
  (h7 : f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 = 1) :
  ∃ (d₁ d₂ d₃ d₄ : ℕ), (d₁ = 5) ∧ (d₂ = 15) ∧ (d₃ = 45) ∧ (d₄ = 385) :=
by
  sorry

end NUMINAMATH_GPT_find_denominators_l710_71011


namespace NUMINAMATH_GPT_can_capacity_l710_71049

-- Definition for the capacity of the can
theorem can_capacity 
  (milk_ratio water_ratio : ℕ) 
  (add_milk : ℕ) 
  (final_milk_ratio final_water_ratio : ℕ) 
  (capacity : ℕ) 
  (initial_milk initial_water : ℕ) 
  (h_initial_ratio : milk_ratio = 4 ∧ water_ratio = 3) 
  (h_additional_milk : add_milk = 8) 
  (h_final_ratio : final_milk_ratio = 2 ∧ final_water_ratio = 1) 
  (h_initial_amounts : initial_milk = 4 * (capacity - add_milk) / 7 ∧ initial_water = 3 * (capacity - add_milk) / 7) 
  (h_full_capacity : (initial_milk + add_milk) / initial_water = 2) 
  : capacity = 36 :=
sorry

end NUMINAMATH_GPT_can_capacity_l710_71049


namespace NUMINAMATH_GPT_find_xy_l710_71052

theorem find_xy (x y : ℝ) :
  0.75 * x - 0.40 * y = 0.20 * 422.50 →
  0.30 * x + 0.50 * y = 0.35 * 530 →
  x = 52.816 ∧ y = -112.222 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_xy_l710_71052


namespace NUMINAMATH_GPT_sum_products_of_chords_l710_71017

variable {r x y u v : ℝ}

theorem sum_products_of_chords (h1 : x * y = u * v) (h2 : 4 * r^2 = (x + y)^2 + (u + v)^2) :
  x * (x + y) + u * (u + v) = 4 * r^2 := by
sorry

end NUMINAMATH_GPT_sum_products_of_chords_l710_71017


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l710_71068

theorem necessary_but_not_sufficient (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≠ b) : ab > 0 :=
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l710_71068


namespace NUMINAMATH_GPT_multiply_negatives_l710_71012

theorem multiply_negatives : (-2) * (-3) = 6 :=
  by 
  sorry

end NUMINAMATH_GPT_multiply_negatives_l710_71012


namespace NUMINAMATH_GPT_find_c_l710_71073

variable (c : ℝ)

theorem find_c (h : c * (1 + 1/2 + 1/3 + 1/4) = 1) : c = 12 / 25 :=
by 
  sorry

end NUMINAMATH_GPT_find_c_l710_71073


namespace NUMINAMATH_GPT_bacteria_growth_rate_l710_71074

theorem bacteria_growth_rate (P : ℝ) (r : ℝ) : 
  (P * r ^ 25 = 2 * (P * r ^ 24) ) → r = 2 :=
by sorry

end NUMINAMATH_GPT_bacteria_growth_rate_l710_71074


namespace NUMINAMATH_GPT_canoes_vs_kayaks_l710_71046

theorem canoes_vs_kayaks (C K : ℕ) (h1 : 9 * C + 12 * K = 432) (h2 : C = 4 * K / 3) : C - K = 6 :=
sorry

end NUMINAMATH_GPT_canoes_vs_kayaks_l710_71046


namespace NUMINAMATH_GPT_all_Xanths_are_Yelps_and_Wicks_l710_71040

-- Definitions for Zorbs, Yelps, Xanths, and Wicks
variable {U : Type} (Zorb Yelp Xanth Wick : U → Prop)

-- Conditions from the problem
axiom all_Zorbs_are_Yelps : ∀ u, Zorb u → Yelp u
axiom all_Xanths_are_Zorbs : ∀ u, Xanth u → Zorb u
axiom all_Xanths_are_Wicks : ∀ u, Xanth u → Wick u

-- The goal is to prove that all Xanths are Yelps and are Wicks
theorem all_Xanths_are_Yelps_and_Wicks : ∀ u, Xanth u → Yelp u ∧ Wick u := sorry

end NUMINAMATH_GPT_all_Xanths_are_Yelps_and_Wicks_l710_71040


namespace NUMINAMATH_GPT_average_sleep_is_8_l710_71083

-- Define the hours of sleep for each day
def mondaySleep : ℕ := 8
def tuesdaySleep : ℕ := 7
def wednesdaySleep : ℕ := 8
def thursdaySleep : ℕ := 10
def fridaySleep : ℕ := 7

-- Calculate the total hours of sleep over the week
def totalSleep : ℕ := mondaySleep + tuesdaySleep + wednesdaySleep + thursdaySleep + fridaySleep
-- Define the total number of days
def totalDays : ℕ := 5

-- Calculate the average sleep per night
def averageSleepPerNight : ℕ := totalSleep / totalDays

-- Prove the statement
theorem average_sleep_is_8 : averageSleepPerNight = 8 := 
by
  -- All conditions are automatically taken into account as definitions
  -- Add a placeholder to skip the actual proof
  sorry

end NUMINAMATH_GPT_average_sleep_is_8_l710_71083


namespace NUMINAMATH_GPT_set_intersection_complement_l710_71066

open Set

variable (U P Q: Set ℕ)

theorem set_intersection_complement (hU: U = {1, 2, 3, 4}) (hP: P = {1, 2}) (hQ: Q = {2, 3}) :
  P ∩ (U \ Q) = {1} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l710_71066


namespace NUMINAMATH_GPT_solve_inequalities_l710_71002

theorem solve_inequalities {x : ℝ} :
  (3 * x + 1) / 2 > x ∧ (4 * (x - 2) ≤ x - 5) ↔ (-1 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_GPT_solve_inequalities_l710_71002


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l710_71054

variable (a b : ℝ)

theorem quadratic_inequality_solution_set :
  (∀ x : ℝ, (a + b) * x + 2 * a - 3 * b < 0 ↔ x > -(3 / 4)) →
  (∀ x : ℝ, (a - 2 * b) * x ^ 2 + 2 * (a - b - 1) * x + (a - 2) > 0 ↔ -3 + 2 / b < x ∧ x < -1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l710_71054


namespace NUMINAMATH_GPT_number_and_sum_of_g3_l710_71003

-- Define the function g with its conditions
variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (x * g y - x) = 2 * x * y + g x)

-- Define the problem parameters
def n : ℕ := sorry -- Number of possible values of g(3)
def s : ℝ := sorry -- Sum of all possible values of g(3)

-- The main statement to be proved
theorem number_and_sum_of_g3 : n * s = 0 := sorry

end NUMINAMATH_GPT_number_and_sum_of_g3_l710_71003


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l710_71025

theorem sum_of_squares_of_roots :
  ∀ (r₁ r₂ : ℝ), (r₁ + r₂ = 15) → (r₁ * r₂ = 6) → (r₁^2 + r₂^2 = 213) :=
by
  intros r₁ r₂ h_sum h_prod
  -- Proof goes here, but skipping it for now
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l710_71025


namespace NUMINAMATH_GPT_sin_square_general_proposition_l710_71061

-- Definitions for the given conditions
def sin_square_sum_30_90_150 : Prop :=
  (Real.sin (30 * Real.pi / 180))^2 + (Real.sin (90 * Real.pi / 180))^2 + (Real.sin (150 * Real.pi / 180))^2 = 3/2

def sin_square_sum_5_65_125 : Prop :=
  (Real.sin (5 * Real.pi / 180))^2 + (Real.sin (65 * Real.pi / 180))^2 + (Real.sin (125 * Real.pi / 180))^2 = 3/2

-- The general proposition we want to prove
theorem sin_square_general_proposition (α : ℝ) : 
  sin_square_sum_30_90_150 ∧ sin_square_sum_5_65_125 →
  (Real.sin (α * Real.pi / 180 - 60 * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180 + 60 * Real.pi / 180))^2 = 3/2 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sin_square_general_proposition_l710_71061


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l710_71067

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x > 2 → |x| ≥ 1) ∧ (∃ x : ℝ, |x| ≥ 1 ∧ ¬ (x > 2)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l710_71067


namespace NUMINAMATH_GPT_halfway_miles_proof_l710_71050

def groceries_miles : ℕ := 10
def haircut_miles : ℕ := 15
def doctor_miles : ℕ := 5

def total_miles : ℕ := groceries_miles + haircut_miles + doctor_miles

theorem halfway_miles_proof : total_miles / 2 = 15 := by
  -- calculation to follow
  sorry

end NUMINAMATH_GPT_halfway_miles_proof_l710_71050


namespace NUMINAMATH_GPT_lives_per_player_l710_71048

theorem lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8) (h2 : additional_players = 2) (h3 : total_lives = 60) : 
  total_lives / (initial_players + additional_players) = 6 :=
by 
  sorry

end NUMINAMATH_GPT_lives_per_player_l710_71048


namespace NUMINAMATH_GPT_smallest_x_for_equation_l710_71058

theorem smallest_x_for_equation :
  ∃ x : ℝ, x = -15 ∧ (∀ y : ℝ, 3*y^2 + 39*y - 75 = y*(y + 16) → x ≤ y) ∧ 
  3*(-15)^2 + 39*(-15) - 75 = -15*(-15 + 16) :=
sorry

end NUMINAMATH_GPT_smallest_x_for_equation_l710_71058


namespace NUMINAMATH_GPT_niko_percentage_profit_l710_71010

theorem niko_percentage_profit
    (pairs_sold : ℕ)
    (cost_per_pair : ℕ)
    (profit_5_pairs : ℕ)
    (total_profit : ℕ)
    (num_pairs_remaining : ℕ)
    (cost_remaining_pairs : ℕ)
    (profit_remaining_pairs : ℕ)
    (percentage_profit : ℕ)
    (cost_5_pairs : ℕ):
    pairs_sold = 9 →
    cost_per_pair = 2 →
    profit_5_pairs = 1 →
    total_profit = 3 →
    num_pairs_remaining = 4 →
    cost_remaining_pairs = 8 →
    profit_remaining_pairs = 2 →
    percentage_profit = 25 →
    cost_5_pairs = 10 →
    (profit_remaining_pairs * 100 / cost_remaining_pairs) = percentage_profit :=
by
    intros
    sorry

end NUMINAMATH_GPT_niko_percentage_profit_l710_71010


namespace NUMINAMATH_GPT_paint_coverage_l710_71090

theorem paint_coverage 
  (width height cost_per_quart money_spent area : ℕ)
  (cover : ℕ → ℕ → ℕ)
  (num_sides quarts_purchased : ℕ)
  (total_area num_quarts : ℕ)
  (sqfeet_per_quart : ℕ) :
  width = 5 
  → height = 4 
  → cost_per_quart = 2 
  → money_spent = 20 
  → num_sides = 2
  → cover width height = area
  → area * num_sides = total_area
  → money_spent / cost_per_quart = quarts_purchased
  → total_area / quarts_purchased = sqfeet_per_quart
  → total_area = 40 
  → quarts_purchased = 10 
  → sqfeet_per_quart = 4 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_paint_coverage_l710_71090


namespace NUMINAMATH_GPT_bert_earns_more_l710_71098

def bert_toy_phones : ℕ := 8
def bert_price_per_phone : ℕ := 18
def tory_toy_guns : ℕ := 7
def tory_price_per_gun : ℕ := 20

theorem bert_earns_more : (bert_toy_phones * bert_price_per_phone) - (tory_toy_guns * tory_price_per_gun) = 4 := by
  sorry

end NUMINAMATH_GPT_bert_earns_more_l710_71098


namespace NUMINAMATH_GPT_joan_seashells_total_l710_71081

-- Definitions
def original_seashells : ℕ := 70
def additional_seashells : ℕ := 27
def total_seashells : ℕ := original_seashells + additional_seashells

-- Proof Statement
theorem joan_seashells_total : total_seashells = 97 := by
  sorry

end NUMINAMATH_GPT_joan_seashells_total_l710_71081


namespace NUMINAMATH_GPT_system_of_linear_equations_m_l710_71065

theorem system_of_linear_equations_m (x y m : ℝ) :
  (2 * x + y = 1 + 2 * m) →
  (x + 2 * y = 2 - m) →
  (x + y > 0) →
  ((2 * m + 1) * x - 2 * m < 1) →
  (x > 1) →
  (-3 < m ∧ m < -1/2) ∧ (m = -2 ∨ m = -1) :=
by
  intros h1 h2 h3 h4 h5
  -- Placeholder for proof steps
  sorry

end NUMINAMATH_GPT_system_of_linear_equations_m_l710_71065


namespace NUMINAMATH_GPT_grasshopper_twenty_five_jumps_l710_71023

noncomputable def sum_natural (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem grasshopper_twenty_five_jumps :
  let total_distance := sum_natural 25
  total_distance % 2 = 1 -> 0 % 2 = 0 -> total_distance ≠ 0 :=
by
  intros total_distance_odd zero_even
  sorry

end NUMINAMATH_GPT_grasshopper_twenty_five_jumps_l710_71023


namespace NUMINAMATH_GPT_centroid_triangle_PQR_l710_71021

theorem centroid_triangle_PQR (P Q R S : ℝ × ℝ) 
  (P_coord : P = (2, 5)) 
  (Q_coord : Q = (9, 3)) 
  (R_coord : R = (4, -4))
  (S_is_centroid : S = (
    (P.1 + Q.1 + R.1) / 3,
    (P.2 + Q.2 + R.2) / 3)) :
  9 * S.1 + 4 * S.2 = 151 / 3 :=
by
  sorry

end NUMINAMATH_GPT_centroid_triangle_PQR_l710_71021


namespace NUMINAMATH_GPT_find_alpha_l710_71079

theorem find_alpha (α : Real) (hα : 0 < α ∧ α < π) :
  (∃ x : Real, (|2 * x - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * x| = Real.sin α) ∧ 
  ∀ y : Real, (|2 * y - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * y| = Real.sin α) → y = x) →
  α = π / 12 ∨ α = 11 * π / 12 :=
by
  sorry

end NUMINAMATH_GPT_find_alpha_l710_71079


namespace NUMINAMATH_GPT_bugs_eat_total_flowers_l710_71078

def num_bugs : ℝ := 2.0
def flowers_per_bug : ℝ := 1.5
def total_flowers_eaten : ℝ := 3.0

theorem bugs_eat_total_flowers : 
  (num_bugs * flowers_per_bug) = total_flowers_eaten := 
  by 
    sorry

end NUMINAMATH_GPT_bugs_eat_total_flowers_l710_71078


namespace NUMINAMATH_GPT_axis_of_symmetry_of_f_l710_71070

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x + 1)

theorem axis_of_symmetry_of_f : (axis_of_symmetry : ℝ) = -1 :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_of_f_l710_71070


namespace NUMINAMATH_GPT_tank_salt_solution_l710_71096

theorem tank_salt_solution (x : ℝ) (hx1 : 0.20 * x / (3 / 4 * x + 30) = 1 / 3) : x = 200 :=
by sorry

end NUMINAMATH_GPT_tank_salt_solution_l710_71096


namespace NUMINAMATH_GPT_find_sports_package_channels_l710_71089

-- Defining the conditions
def initial_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def reduce_package_by : ℕ := 10
def supreme_sports_package : ℕ := 7
def final_channels : ℕ := 147

-- Defining the situation before the final step
def channels_after_reduction := initial_channels - channels_taken_away + channels_replaced - reduce_package_by
def channels_after_supreme := channels_after_reduction + supreme_sports_package

-- Prove the original sports package added 8 channels
theorem find_sports_package_channels : ∀ sports_package_added : ℕ,
  sports_package_added + channels_after_supreme = final_channels → sports_package_added = 8 :=
by
  intro sports_package_added
  intro h
  sorry

end NUMINAMATH_GPT_find_sports_package_channels_l710_71089


namespace NUMINAMATH_GPT_trig_identity_sum_l710_71097

-- Define the trigonometric functions and their properties
def sin_210_eq : Real.sin (210 * Real.pi / 180) = - Real.sin (30 * Real.pi / 180) := by
  sorry

def cos_60_eq : Real.cos (60 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) := by
  sorry

-- The goal is to prove that the sum of these specific trigonometric values is 0
theorem trig_identity_sum : Real.sin (210 * Real.pi / 180) + Real.cos (60 * Real.pi / 180) = 0 := by
  rw [sin_210_eq, cos_60_eq]
  sorry

end NUMINAMATH_GPT_trig_identity_sum_l710_71097


namespace NUMINAMATH_GPT_cot_half_angle_product_geq_3sqrt3_l710_71009

noncomputable def cot (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)

theorem cot_half_angle_product_geq_3sqrt3 {A B C : ℝ} (h : A + B + C = π) :
    cot (A / 2) * cot (B / 2) * cot (C / 2) ≥ 3 * Real.sqrt 3 := 
  sorry

end NUMINAMATH_GPT_cot_half_angle_product_geq_3sqrt3_l710_71009


namespace NUMINAMATH_GPT_combination_value_l710_71072

theorem combination_value (m : ℕ) (h : (1 / (Nat.choose 5 m) - 1 / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m))) : 
    Nat.choose 8 m = 28 := 
sorry

end NUMINAMATH_GPT_combination_value_l710_71072


namespace NUMINAMATH_GPT_cost_of_one_dozen_pens_l710_71027

theorem cost_of_one_dozen_pens (pen pencil : ℝ) (h_ratios : pen = 5 * pencil) (h_total : 3 * pen + 5 * pencil = 240) :
  12 * pen = 720 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_dozen_pens_l710_71027


namespace NUMINAMATH_GPT_sum_of_numbers_in_ratio_with_lcm_l710_71004

theorem sum_of_numbers_in_ratio_with_lcm (a b : ℕ) (h_lcm : Nat.lcm a b = 36) (h_ratio : a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3) : a + b = 30 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_in_ratio_with_lcm_l710_71004


namespace NUMINAMATH_GPT_lawsuit_win_probability_l710_71034

theorem lawsuit_win_probability (P_L1 P_L2 P_W1 P_W2 : ℝ) (h1 : P_L2 = 0.5) 
  (h2 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20 * P_W1 * P_W2)
  (h3 : P_W1 + P_L1 = 1)
  (h4 : P_W2 + P_L2 = 1) : 
  P_W1 = 1 / 2.20 :=
by
  sorry

end NUMINAMATH_GPT_lawsuit_win_probability_l710_71034


namespace NUMINAMATH_GPT_part1_l710_71062

def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

def setB (x : ℝ) : Prop := x ≠ 0 ∧ x ≤ 5 ∧ 0 < x

def setC (a x : ℝ) : Prop := 3 * a ≤ x ∧ x ≤ 2 * a + 1

def setInter (x : ℝ) : Prop := setA x ∧ setB x

theorem part1 (a : ℝ) : (∀ x, setC a x → setInter x) ↔ (0 < a ∧ a ≤ 1 / 2 ∨ 1 < a) :=
sorry

end NUMINAMATH_GPT_part1_l710_71062


namespace NUMINAMATH_GPT_gcd_stamps_pages_l710_71077

def num_stamps_book1 : ℕ := 924
def num_stamps_book2 : ℕ := 1200

theorem gcd_stamps_pages : Nat.gcd num_stamps_book1 num_stamps_book2 = 12 := by
  sorry

end NUMINAMATH_GPT_gcd_stamps_pages_l710_71077


namespace NUMINAMATH_GPT_math_proof_l710_71013

theorem math_proof :
  ∀ (x y z : ℚ), (2 * x - 3 * y - 2 * z = 0) →
                  (x + 3 * y - 28 * z = 0) →
                  (z ≠ 0) →
                  (x^2 + 3 * x * y * z) / (y^2 + z^2) = 280 / 37 :=
by
  intros x y z h1 h2 h3
  sorry

end NUMINAMATH_GPT_math_proof_l710_71013


namespace NUMINAMATH_GPT_real_polynomial_has_exactly_one_real_solution_l710_71084

theorem real_polynomial_has_exactly_one_real_solution:
  ∀ a : ℝ, ∃! x : ℝ, x^3 - a * x^2 - 3 * a * x + a^2 - 1 = 0 := 
by
  sorry

end NUMINAMATH_GPT_real_polynomial_has_exactly_one_real_solution_l710_71084


namespace NUMINAMATH_GPT_age_sum_proof_l710_71053

theorem age_sum_proof (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 20) : a + b + c = 52 :=
by
  sorry

end NUMINAMATH_GPT_age_sum_proof_l710_71053


namespace NUMINAMATH_GPT_HCF_a_b_LCM_a_b_l710_71099

-- Given the HCF condition
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Given numbers
def a : ℕ := 210
def b : ℕ := 286

-- Given HCF condition
theorem HCF_a_b : HCF a b = 26 := by
  sorry

-- LCM definition based on the product and HCF
def LCM (a b : ℕ) : ℕ := (a * b) / HCF a b

-- Theorem to prove
theorem LCM_a_b : LCM a b = 2310 := by
  sorry

end NUMINAMATH_GPT_HCF_a_b_LCM_a_b_l710_71099


namespace NUMINAMATH_GPT_daniel_stickers_l710_71044

def stickers_data 
    (total_stickers : Nat)
    (fred_extra : Nat)
    (andrew_kept : Nat) : Prop :=
  total_stickers = 750 ∧ fred_extra = 120 ∧ andrew_kept = 130

theorem daniel_stickers (D : Nat) :
  stickers_data 750 120 130 → D + (D + 120) = 750 - 130 → D = 250 :=
by
  intros h_data h_eq
  sorry

end NUMINAMATH_GPT_daniel_stickers_l710_71044


namespace NUMINAMATH_GPT_find_b_minus_d_squared_l710_71063

theorem find_b_minus_d_squared (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3) :
  (b - d) ^ 2 = 25 :=
sorry

end NUMINAMATH_GPT_find_b_minus_d_squared_l710_71063


namespace NUMINAMATH_GPT_remainder_of_polynomial_division_l710_71043

theorem remainder_of_polynomial_division :
  ∀ (x : ℂ), ((x + 2) ^ 2023) % (x^2 + x + 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_division_l710_71043


namespace NUMINAMATH_GPT_ann_age_is_26_l710_71020

theorem ann_age_is_26
  (a b : ℕ)
  (h1 : a + b = 50)
  (h2 : b = 2 * a / 3 + 2 * (a - b)) :
  a = 26 :=
by
  sorry

end NUMINAMATH_GPT_ann_age_is_26_l710_71020


namespace NUMINAMATH_GPT_min_sum_of_factors_l710_71042

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 1806) (h2 : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) : a + b + c ≥ 112 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l710_71042


namespace NUMINAMATH_GPT_probability_all_correct_l710_71032

noncomputable def probability_mcq : ℚ := 1 / 3
noncomputable def probability_true_false : ℚ := 1 / 2

theorem probability_all_correct :
  (probability_mcq * probability_true_false * probability_true_false) = (1 / 12) :=
by
  sorry

end NUMINAMATH_GPT_probability_all_correct_l710_71032


namespace NUMINAMATH_GPT_correct_sample_in_survey_l710_71041

-- Definitions based on conditions:
def total_population := 1500
def surveyed_population := 150
def sample_description := "the national security knowledge of the selected 150 teachers and students"

-- Hypotheses: conditions
variables (pop : ℕ) (surveyed : ℕ) (description : String)
  (h1 : pop = total_population)
  (h2 : surveyed = surveyed_population)
  (h3 : description = sample_description)

-- Theorem we want to prove
theorem correct_sample_in_survey : description = sample_description :=
  by sorry

end NUMINAMATH_GPT_correct_sample_in_survey_l710_71041


namespace NUMINAMATH_GPT_find_y_common_solution_l710_71051

theorem find_y_common_solution (y : ℝ) :
  (∃ x : ℝ, x^2 + y^2 = 11 ∧ x^2 = 4*y - 7) ↔ (7/4 ≤ y ∧ y ≤ Real.sqrt 11) :=
by
  sorry

end NUMINAMATH_GPT_find_y_common_solution_l710_71051


namespace NUMINAMATH_GPT_solve_for_diamond_l710_71087

theorem solve_for_diamond (d : ℤ) (h : d * 9 + 5 = d * 10 + 2) : d = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_diamond_l710_71087


namespace NUMINAMATH_GPT_solve_equation_l710_71014

theorem solve_equation (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 → 
  x ≠ 3 → 
  x ≠ -2 → 
  x = -7 / 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l710_71014


namespace NUMINAMATH_GPT_largest_nonrepresentable_by_17_11_l710_71045

/--
In the USA, standard letter-size paper is 8.5 inches wide and 11 inches long. The largest integer that cannot be written as a sum of a whole number (possibly zero) of 17's and a whole number (possibly zero) of 11's is 159.
-/
theorem largest_nonrepresentable_by_17_11 : 
  ∀ (a b : ℕ), (∀ (n : ℕ), n = 17 * a + 11 * b -> n ≠ 159) ∧ 
               ¬ (∃ (a b : ℕ), 17 * a + 11 * b = 159) :=
by
  sorry

end NUMINAMATH_GPT_largest_nonrepresentable_by_17_11_l710_71045


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l710_71047

theorem arithmetic_sequence_common_difference
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (a₁ d : ℤ)
  (h1 : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h2 : ∀ n, a n = a₁ + (n - 1) * d)
  (h3 : S 5 = 5 * (a 4) - 10) :
  d = 2 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l710_71047


namespace NUMINAMATH_GPT_range_of_t_l710_71095

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (a t : ℝ) := 2 * a * t - t^2

theorem range_of_t (t : ℝ) (a : ℝ) (x : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x)
                   (h₂ : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 1 → f x₁ ≤ f x₂)
                   (h₃ : f (-1) = -1) (h₄ : -1 ≤ x ∧ x ≤ 1 → f x ≤ t^2 - 2 * a * t + 1)
                   (h₅ : -1 ≤ a ∧ a ≤ 1) :
  t ≥ 2 ∨ t = 0 ∨ t ≤ -2 := sorry

end NUMINAMATH_GPT_range_of_t_l710_71095


namespace NUMINAMATH_GPT_max_abs_f_lower_bound_l710_71007

theorem max_abs_f_lower_bound (a b M : ℝ) (hM : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → abs (x^2 + a*x + b) ≤ M) : 
  M ≥ 1/2 :=
sorry

end NUMINAMATH_GPT_max_abs_f_lower_bound_l710_71007


namespace NUMINAMATH_GPT_solve_equation_l710_71035

theorem solve_equation (x : ℝ) (h₁ : x ≠ -11) (h₂ : x ≠ -5) (h₃ : x ≠ -12) (h₄ : x ≠ -4) :
  (1 / (x + 11) + 1 / (x + 5) = 1 / (x + 12) + 1 / (x + 4)) ↔ x = -8 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l710_71035


namespace NUMINAMATH_GPT_alec_string_ways_l710_71057

theorem alec_string_ways :
  let letters := ['A', 'C', 'G', 'N']
  let num_ways := 24 * 2 * 2
  num_ways = 96 := 
by
  sorry

end NUMINAMATH_GPT_alec_string_ways_l710_71057


namespace NUMINAMATH_GPT_jeffrey_fills_crossword_l710_71019

noncomputable def prob_fill_crossword : ℚ :=
  let total_clues := 10
  let prob_knowing_all_clues := (1 / 2) ^ total_clues
  let prob_case_1 := (2 ^ 5) / (2 ^ total_clues)
  let prob_case_2 := (2 ^ 5) / (2 ^ total_clues)
  let prob_case_3 := 25 / (2 ^ total_clues)
  let overcounted_case := prob_knowing_all_clues
  (prob_case_1 + prob_case_2 + prob_case_3 - overcounted_case)

theorem jeffrey_fills_crossword : prob_fill_crossword = 11 / 128 := by
  sorry

end NUMINAMATH_GPT_jeffrey_fills_crossword_l710_71019


namespace NUMINAMATH_GPT_frustum_small_cone_height_is_correct_l710_71075

noncomputable def frustum_small_cone_height (altitude : ℝ) 
                                             (lower_base_area : ℝ) 
                                             (upper_base_area : ℝ) : ℝ :=
  let r1 := Real.sqrt (lower_base_area / Real.pi)
  let r2 := Real.sqrt (upper_base_area / Real.pi)
  let H := 2 * altitude
  altitude

theorem frustum_small_cone_height_is_correct 
  (altitude : ℝ)
  (lower_base_area : ℝ)
  (upper_base_area : ℝ)
  (h1 : altitude = 16)
  (h2 : lower_base_area = 196 * Real.pi)
  (h3 : upper_base_area = 49 * Real.pi ) : 
  frustum_small_cone_height altitude lower_base_area upper_base_area = 16 := by
  sorry

end NUMINAMATH_GPT_frustum_small_cone_height_is_correct_l710_71075


namespace NUMINAMATH_GPT_total_drink_ounces_l710_71091

def total_ounces_entire_drink (coke_parts sprite_parts md_parts coke_ounces : ℕ) : ℕ :=
  let total_parts := coke_parts + sprite_parts + md_parts
  let ounces_per_part := coke_ounces / coke_parts
  total_parts * ounces_per_part

theorem total_drink_ounces (coke_parts sprite_parts md_parts coke_ounces : ℕ) (coke_cond : coke_ounces = 8) (parts_cond : coke_parts = 4 ∧ sprite_parts = 2 ∧ md_parts = 5) : 
  total_ounces_entire_drink coke_parts sprite_parts md_parts coke_ounces = 22 :=
by
  sorry

end NUMINAMATH_GPT_total_drink_ounces_l710_71091


namespace NUMINAMATH_GPT_more_blue_marbles_l710_71094

theorem more_blue_marbles (r_boxes b_boxes marbles_per_box : ℕ) 
    (red_total_eq : r_boxes * marbles_per_box = 70) 
    (blue_total_eq : b_boxes * marbles_per_box = 126) 
    (r_boxes_eq : r_boxes = 5) 
    (b_boxes_eq : b_boxes = 9) 
    (marbles_per_box_eq : marbles_per_box = 14) : 
    126 - 70 = 56 := 
by 
  sorry

end NUMINAMATH_GPT_more_blue_marbles_l710_71094


namespace NUMINAMATH_GPT_range_of_k_for_distinct_real_roots_l710_71080

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ k*x1^2 - 2*x1 - 1 = 0 ∧ k*x2^2 - 2*x2 - 1 = 0) → k > -1 ∧ k ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_for_distinct_real_roots_l710_71080


namespace NUMINAMATH_GPT_factor_correct_l710_71001

noncomputable def factor_expr (x : ℝ) : ℝ :=
  75 * x^3 - 225 * x^10
  
noncomputable def factored_form (x : ℝ) : ℝ :=
  75 * x^3 * (1 - 3 * x^7)

theorem factor_correct (x : ℝ): 
  factor_expr x = factored_form x :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_factor_correct_l710_71001


namespace NUMINAMATH_GPT_base_conversion_proof_l710_71059

-- Definitions of the base-converted numbers
def b1463_7 := 3 * 7^0 + 6 * 7^1 + 4 * 7^2 + 1 * 7^3  -- 1463 in base 7
def b121_5 := 1 * 5^0 + 2 * 5^1 + 1 * 5^2  -- 121 in base 5
def b1754_6 := 4 * 6^0 + 5 * 6^1 + 7 * 6^2 + 1 * 6^3  -- 1754 in base 6
def b3456_7 := 6 * 7^0 + 5 * 7^1 + 4 * 7^2 + 3 * 7^3  -- 3456 in base 7

-- Formalizing the proof goal
theorem base_conversion_proof : (b1463_7 / b121_5 : ℤ) - b1754_6 * 2 + b3456_7 = 278 := by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_base_conversion_proof_l710_71059


namespace NUMINAMATH_GPT_min_values_of_exprs_l710_71024

theorem min_values_of_exprs (r s : ℝ) (hr : 0 < r) (hs : 0 < s) (h : (r + s - r * s) * (r + s + r * s) = r * s) :
  (r + s - r * s) = -3 + 2 * Real.sqrt 3 ∧ (r + s + r * s) = 3 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_min_values_of_exprs_l710_71024


namespace NUMINAMATH_GPT_four_gt_sqrt_fourteen_l710_71060

theorem four_gt_sqrt_fourteen : 4 > Real.sqrt 14 := 
  sorry

end NUMINAMATH_GPT_four_gt_sqrt_fourteen_l710_71060
