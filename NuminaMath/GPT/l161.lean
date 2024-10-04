import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Logic.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbSpace
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Algebra.Module
import Mathlib.Topology.Basic
import Real

namespace lighthouse_coverage_l161_161546

theorem lighthouse_coverage (A B C D : Point) :
  ∃ φA φB φC φD : ℝ,
    (lamp A 90 φA).covers_plane ∧ 
    (lamp B 90 φB).covers_plane ∧
    (lamp C 90 φC).covers_plane ∧
    (lamp D 90 φD).covers_plane :=
by
  sorry

end lighthouse_coverage_l161_161546


namespace fraction_of_clever_integers_divisible_by_18_l161_161880

-- Define a clever integer
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_digits_eq (n : ℕ) (s : ℕ) : Prop :=
  n.digits.sum = s
def clever_integer (n : ℕ) : Prop :=
  n > 10 ∧ n < 130 ∧ is_even n ∧ sum_of_digits_eq n 12

-- Define property of being divisible by 18
def divisible_by_18 (n : ℕ) : Prop := n % 18 = 0

/-- Statement of the problem -/
theorem fraction_of_clever_integers_divisible_by_18 :
  (∃ (N : ℕ), ∃ (L : fin N → ℕ), (∀ i, clever_integer (L i)) ∧ (∀ i, divisible_by_18 (L i)) ∧ (0 < N)) →
  (∃ (M : ℕ), ∃ (P : fin M → ℕ), (∀ j, clever_integer (P j)) ∧ (∀ j, ¬ divisible_by_18 (P j)) ∧ (0 < M)) →
  1 = 1 :=
by
    intros h1 h2
    sorry

end fraction_of_clever_integers_divisible_by_18_l161_161880


namespace solution_set_of_inequality_l161_161152

/-- Given an even function f that is monotonically increasing on [0, ∞) with f(3) = 0,
    show that the solution set for xf(2x - 1) < 0 is (-∞, -1) ∪ (0, 2). -/
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_value : f 3 = 0) :
  {x : ℝ | x * f (2*x - 1) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l161_161152


namespace find_g_25_l161_161129

noncomputable def g (x : ℝ) : ℝ := sorry

axiom h₁ : ∀ (x y : ℝ), x > 0 → y > 0 → g (x / y) = (y / x) * g x
axiom h₂ : g 50 = 4

theorem find_g_25 : g 25 = 4 / 25 :=
by {
  sorry
}

end find_g_25_l161_161129


namespace find_ratio_l161_161561

variable {d : ℕ}
variable {a : ℕ → ℝ}

-- Conditions: arithmetic sequence with non-zero common difference, and geometric sequence terms
axiom arithmetic_sequence (n : ℕ) : a n = a 1 + (n - 1) * d
axiom non_zero_d : d ≠ 0
axiom geometric_sequence : (a 1 + 2*d)^2 = a 1 * (a 1 + 8*d)

-- Theorem to prove the desired ratio
theorem find_ratio : (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
sorry

end find_ratio_l161_161561


namespace flour_per_special_crust_l161_161235

-- Definitions of daily pie crusts and flour usage for standard crusts
def daily_pie_crusts := 50
def flour_per_standard_crust := 1 / 10
def total_daily_flour := daily_pie_crusts * flour_per_standard_crust

-- Definitions for special pie crusts today
def special_pie_crusts := 25
def total_special_flour := total_daily_flour / special_pie_crusts

-- Problem statement in Lean
theorem flour_per_special_crust :
  total_special_flour = 1 / 5 := by
  sorry

end flour_per_special_crust_l161_161235


namespace audrey_lost_pieces_l161_161448

theorem audrey_lost_pieces {total_pieces_on_board : ℕ} {thomas_lost : ℕ} {initial_pieces_each : ℕ} (h1 : total_pieces_on_board = 21) (h2 : thomas_lost = 5) (h3 : initial_pieces_each = 16) :
  (initial_pieces_each - (total_pieces_on_board - (initial_pieces_each - thomas_lost))) = 6 :=
by
  sorry

end audrey_lost_pieces_l161_161448


namespace total_time_correct_l161_161183

def greta_time : ℝ := 6.5
def george_time : ℝ := greta_time - 1.5
def gloria_time : ℝ := 2 * george_time
def gary_time : ℝ := (george_time + gloria_time) + 1.75
def gwen_time : ℝ := (greta_time + george_time) - 0.40 * (greta_time + george_time)
def total_time : ℝ := greta_time + george_time + gloria_time + gary_time + gwen_time

theorem total_time_correct : total_time = 45.15 := by
  sorry

end total_time_correct_l161_161183


namespace perp_AE_EC_perp_AF_FB_perp_AH_BC_l161_161801

-- Geometry setup and conditions
variables (A B C I D M E F H : Point)
variables (triangle : Triangle A B C)
variables (incenter : Incenter I triangle)
variables (AI_intersect_BC : Line A I ∩ Line B C = D)
variables (midpoint_AD : Midpoint M A D)
variables (MB_intersect_circumcircle : Line M B ∩ (Circumcircle B I C) = E)
variables (MC_intersect_circumcircle : Line M C ∩ (Circumcircle B I C) = F)
variable (BF_intersect_CE : Line B F ∩ Line C E = H)

-- Proof statement for AE ⊥ EC
theorem perp_AE_EC : Perp (Line A E) (Line E C) :=
sorry

-- Proof statement for AF ⊥ FB
theorem perp_AF_FB : Perp (Line A F) (Line F B) :=
sorry

-- Proof statement for AH ⊥ BC
theorem perp_AH_BC : Perp (Line A H) (Line B C) :=
sorry

end perp_AE_EC_perp_AF_FB_perp_AH_BC_l161_161801


namespace find_d_l161_161741

theorem find_d (d q : ℝ) :
  (∃ q : ℝ, ∃ p : polynomial ℝ,
    p = 3 * X ^ 3 - C d * X + 18 ∧
    (3 * X ^ 3 - C d * X + 18) = (X ^ 2 + C q * X + 2) * p) →
  d = -6 :=
by
  -- The proof is omitted
  sorry

end find_d_l161_161741


namespace cricket_run_rate_and_boundaries_l161_161998

/-
  Given the first 10 overs of a cricket game have a run rate of 2.1 runs per over,
  the target is 282 runs, the team needs at least 15 boundaries and 5 sixes,
  and the team should lose no more than 3 wickets,
  prove that the required run rate for the remaining 30 overs is 8.7 runs per over
  and the minimum runs from boundaries and sixes should be 90 runs.
-/

noncomputable def calculate_required_run_rate (runs_scored_over_10_overs : ℕ) (target : ℕ) (remaining_overs : ℕ) : ℝ :=
  (target - runs_scored_over_10_overs) / remaining_overs

noncomputable def calculate_minimum_runs (boundaries : ℕ) (sixes : ℕ) : ℕ :=
  (boundaries * 4) + (sixes * 6)

theorem cricket_run_rate_and_boundaries (
  (run_rate_first_10_overs : ℝ) (target : ℕ) (overs_first_part : ℕ) (overs_remaining : ℕ)
  (min_boundaries : ℕ) (min_sixes : ℕ) (max_wickets : ℕ) :
  run_rate_first_10_overs = 2.1 ∧ target = 282 ∧ overs_first_part = 10 ∧ overs_remaining = 30 ∧
  min_boundaries = 15 ∧ min_sixes = 5 ∧ max_wickets = 3)
  : calculate_required_run_rate(21, 282, 30) = 8.7 ∧ calculate_minimum_runs(15, 5) = 90 := by
  sorry

end cricket_run_rate_and_boundaries_l161_161998


namespace ice_cream_sundaes_l161_161444

theorem ice_cream_sundaes (flavors : Finset String) (vanilla : String) (h1 : vanilla ∈ flavors) (h2 : flavors.card = 8) :
  let remaining_flavors := flavors.erase vanilla
  remaining_flavors.card = 7 :=
by
  sorry

end ice_cream_sundaes_l161_161444


namespace find_a_l161_161828

-- Conditions as definitions:
variable (a : ℝ) (b : ℝ)
variable (A : ℝ × ℝ := (0, 0)) (B : ℝ × ℝ := (a, 0)) (C : ℝ × ℝ := (0, b))
noncomputable def area (a b : ℝ) : ℝ := (1 / 2) * a * b

-- Given conditions:
axiom h1 : b = 4
axiom h2 : area a b = 28
axiom h3 : a > 0

-- The proof goal:
theorem find_a : a = 14 := by
  -- proof omitted
  sorry

end find_a_l161_161828


namespace fraction_simplification_l161_161778

theorem fraction_simplification :
  10 * (1/2 + 1/5 + 1/10)⁻¹ = 25 / 2 :=
by
  sorry

end fraction_simplification_l161_161778


namespace general_formula_sum_first_100_terms_l161_161136

-- Definition of the arithmetic sequence
def a_sequence (n : ℕ) : ℕ := n

-- Definitions from the conditions
def a_5 : ℕ := a_sequence 5
def S_5 : ℕ := (∑ i in finset.range 5, a_sequence (i + 1))

-- Defining b_sequence using the given formula
def b_sequence (n : ℕ) : ℝ := 1 / (a_sequence n * a_sequence (n + 1))

-- Main theorem statements
theorem general_formula :
  a_sequence 5 = 5 →
  S_5 = 15 →
  ∀ n : ℕ, a_sequence n = n :=
by
  sorry

theorem sum_first_100_terms (s := ∑ n in finset.range 100, b_sequence (n + 1)) :
  a_sequence 5 = 5 →
  S_5 = 15 →
  s = 100 / 101 :=
by
  sorry

end general_formula_sum_first_100_terms_l161_161136


namespace manager_salary_l161_161318

theorem manager_salary
    (average_salary_employees : ℝ)
    (num_employees : ℕ)
    (increase_in_average_due_to_manager : ℝ)
    (total_salary_20_employees : ℝ)
    (new_average_salary : ℝ)
    (total_salary_with_manager : ℝ) :
  average_salary_employees = 1300 →
  num_employees = 20 →
  increase_in_average_due_to_manager = 100 →
  total_salary_20_employees = average_salary_employees * num_employees →
  new_average_salary = average_salary_employees + increase_in_average_due_to_manager →
  total_salary_with_manager = new_average_salary * (num_employees + 1) →
  total_salary_with_manager - total_salary_20_employees = 3400 :=
by 
  sorry

end manager_salary_l161_161318


namespace number_of_squares_in_100th_ring_l161_161077

def a : ℕ → ℕ
| 1     := 4
| (n+1) := a n + 8

theorem number_of_squares_in_100th_ring : a 100 = 796 :=
by
    sorry

end number_of_squares_in_100th_ring_l161_161077


namespace number_of_southbound_vehicles_l161_161757

variable (speed_northbound : ℝ) (speed_southbound : ℝ) (vehicles_passed : ℕ) (time_interval_minutes : ℕ) (section_length_miles : ℕ)

def traffic_conditions : Prop :=
  speed_northbound = 60 ∧ 
  speed_southbound = 50 ∧ 
  vehicles_passed = 30 ∧ 
  time_interval_minutes = 6 ∧ 
  section_length_miles = 150

theorem number_of_southbound_vehicles (h : traffic_conditions speed_northbound speed_southbound vehicles_passed time_interval_minutes section_length_miles) : 
    abs ((30 / (60 + 50) * (150 / (6 / 60))) - 450) ≤ min (abs ((30 / (60 + 50) * (150 / (6 / 60))) - 300)) (min (abs ((30 / (60 + 50) * (150 / (6 / 60))) - 375)) (abs ((30 / (60 + 50) * (150 / (6 / 60))) - 500))) :=
by {
  sorry
}

end number_of_southbound_vehicles_l161_161757


namespace cos_difference_simplification_l161_161295

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  x - y = -1 / (2 * Real.sqrt 5) :=
sorry

end cos_difference_simplification_l161_161295


namespace grocery_store_price_l161_161413

-- Definitions based on the conditions
def bulk_price_per_case : ℝ := 12.00
def bulk_cans_per_case : ℝ := 48.0
def grocery_cans_per_pack : ℝ := 12.0
def additional_cost_per_can : ℝ := 0.25

-- The proof statement
theorem grocery_store_price : 
  (bulk_price_per_case / bulk_cans_per_case + additional_cost_per_can) * grocery_cans_per_pack = 6.00 :=
by
  sorry

end grocery_store_price_l161_161413


namespace sum_of_squares_is_77_l161_161743

-- Definitions based on the conditions
def consecutive_integers (a : ℕ) : set ℕ := {a - 1, a, a + 1}
def product_of_consecutive_integers (a : ℕ) : ℕ := (a - 1) * a * (a + 1)
def sum_of_consecutive_integers (a : ℕ) : ℕ := (a - 1) + a + (a + 1)
def sum_of_squares_of_consecutive_integers (a : ℕ) : ℕ := (a - 1)^2 + a^2 + (a + 1)^2

-- Condition that the product of these integers is 8 times their sum
axiom product_condition (a : ℕ) (h : a > 0) : product_of_consecutive_integers a = 8 * sum_of_consecutive_integers a

-- Statement to prove
theorem sum_of_squares_is_77 (a : ℕ) (h : a > 0) (hc : product_of_consecutive_integers a = 8 * sum_of_consecutive_integers a) : sum_of_squares_of_consecutive_integers a = 77 :=
by
  sorry

end sum_of_squares_is_77_l161_161743


namespace least_squares_minimizes_sum_of_squared_errors_l161_161338

-- Define the data and notation
variables {n : ℕ} 
variables {y_i : Fin n → ℝ} -- observed values
variables {y_hat : Fin n → ℝ} -- predicted values

-- Definition of the least squares objective
def sum_of_squared_errors (y_i y_hat : Fin n → ℝ) : ℝ :=
  ∑ i, (y_i i - y_hat i)^2

-- The theorem stating the objective of least squares
theorem least_squares_minimizes_sum_of_squared_errors :
  (∃ (f : Fin n → ℝ), sum_of_squared_errors y_i f = ⨉) :=
sorry

end least_squares_minimizes_sum_of_squared_errors_l161_161338


namespace magnitude_a_minus_2b_l161_161181

variables {V : Type} [inner_product_space ℝ V] [normed_space ℝ V]

def vector_a : V := sorry
def vector_b : V := sorry

def magnitude_of_vector (v : V) : ℝ := real.sqrt (inner_product_space.norm_sq v)

theorem magnitude_a_minus_2b
  (h_norm_a : ∥vector_a∥ = 2)
  (h_norm_b : ∥vector_b∥ = 1)
  (h_angle : inner_product_space.real_angle vector_a vector_b = real.pi / 3) :
  ∥vector_a - 2 • vector_b∥ = 2 :=
sorry

end magnitude_a_minus_2b_l161_161181


namespace exists_valid_star_arrangement_no_valid_arrangement_less_than_7_l161_161409

def star_arrangement_valid (arr : matrix (fin 4) (fin 4) bool) : Prop :=
  ∀ (r1 r2 : fin 4) (c1 c2 : fin 4),
    r1 ≠ r2 → c1 ≠ c2 → !(arr r1 c1 ∧ arr r1 c2 ∧ arr r2 c1 ∧ arr r2 c2)

theorem exists_valid_star_arrangement :
  ∃ arr : matrix (fin 4) (fin 4) bool, (∑ i j, if arr i j then 1 else 0) = 7 ∧ star_arrangement_valid arr :=
sorry

theorem no_valid_arrangement_less_than_7 :
  ∀ (arr : matrix (fin 4) (fin 4) bool), (∑ i j, if arr i j then 1 else 0) < 7 → ¬ star_arrangement_valid arr :=
sorry

end exists_valid_star_arrangement_no_valid_arrangement_less_than_7_l161_161409


namespace right_angled_triangles_count_l161_161184

theorem right_angled_triangles_count : 
  ∃ n : ℕ, n = 12 ∧ ∀ (a b c : ℕ), (a = 2016^(1/2)) → (a^2 + b^2 = c^2) →
  (∃ (n k : ℕ), (c - b) = n ∧ (c + b) = k ∧ 2 ∣ n ∧ 2 ∣ k ∧ (n * k = 2016)) :=
by {
  sorry
}

end right_angled_triangles_count_l161_161184


namespace minimum_y_squared_l161_161661

-- Representing the given trapezoid and its properties
structure IsoscelesTrapezoid where
  E F G H : Point
  EF GH EG FH : ℝ
  mid_EF : Point
  circle_center : Point
  isosceles : EFGH_isosceles
  EF_eq : EF = 100
  GH_eq : GH = 25
  EG_eq : EG = y
  FH_eq : FH = y
  circle_tangent_to_EG : is_tangent circle_center EG
  circle_tangent_to_FH : is_tangent circle_center FH

-- Proving the main statement about smallest y^2
theorem minimum_y_squared (trapezoid : IsoscelesTrapezoid) (y : ℝ)
  (hy : y > 0) : y^2 = 2031.25 :=
by
  sorry

end minimum_y_squared_l161_161661


namespace An_is_integer_l161_161925

theorem An_is_integer 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : a > b)
  (θ : ℝ) (h_theta : θ > 0 ∧ θ < Real.pi / 2)
  (h_sin : Real.sin θ = 2 * (a * b) / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, ((a^2 + b^2)^n * Real.sin (n * θ) : ℝ) = k :=
by sorry

end An_is_integer_l161_161925


namespace placemat_length_l161_161429

theorem placemat_length :
  let R := 5
  let width := 1
  let θ := 22.5
  let sinθ := Real.sin (θ * Real.pi / 180)
  let y := Real.sqrt 24.75 - 2.5 * Real.sqrt (2 - Real.sqrt 2)
  (∀ n : ℕ, n = 8 →
    ∀ mat_width : ℕ, mat_width = width →
    ∀ (R : ℝ), R = 5 →
    ∀ θ : ℝ, θ = 22.5 →
    ∀ sinθ : ℝ, sinθ = Real.sin (θ * Real.pi / 180),
      y = Real.sqrt 24.75 - 2.5 * Real.sqrt (2 - Real.sqrt 2)) :=
by
  sorry

end placemat_length_l161_161429


namespace computeProduct_correct_l161_161071

noncomputable def computeProduct : ℕ :=
    (floor (real.sqrt (real.sqrt 1))) * (floor (real.sqrt (real.sqrt 3))) * (floor (real.sqrt (real.sqrt 5))) *
    (floor (real.sqrt (real.sqrt 7))) * (floor (real.sqrt (real.sqrt 9))) * (floor (real.sqrt (real.sqrt 11))) *
    (floor (real.sqrt (real.sqrt 13))) * (floor (real.sqrt (real.sqrt 15))) *
    (floor (real.sqrt (real.sqrt 17))) * (floor (real.sqrt (real.sqrt 19))) *
    (floor (real.sqrt (real.sqrt 21))) * (floor (real.sqrt (real.sqrt 23))) *
    (floor (real.sqrt (real.sqrt 25))) * (floor (real.sqrt (real.sqrt 27))) *
    (floor (real.sqrt (real.sqrt 29))) * (floor (real.sqrt (real.sqrt 31))) *
    (floor (real.sqrt (real.sqrt 33))) * (floor (real.sqrt (real.sqrt 35))) *
    (floor (real.sqrt (real.sqrt 37))) * (floor (real.sqrt (real.sqrt 39))) *
    (floor (real.sqrt (real.sqrt 41))) * (floor (real.sqrt (real.sqrt 43))) *
    (floor (real.sqrt (real.sqrt 45))) * (floor (real.sqrt (real.sqrt 47))) *
    (floor (real.sqrt (real.sqrt 49))) * (floor (real.sqrt (real.sqrt 51))) *
    (floor (real.sqrt (real.sqrt 53))) * (floor (real.sqrt (real.sqrt 55))) *
    (floor (real.sqrt (real.sqrt 57))) * (floor (real.sqrt (real.sqrt 59))) *
    (floor (real.sqrt (real.sqrt 61))) * (floor (real.sqrt (real.sqrt 63))) *
    (floor (real.sqrt (real.sqrt 65))) * (floor (real.sqrt (real.sqrt 67))) *
    (floor (real.sqrt (real.sqrt 69))) * (floor (real.sqrt (real.sqrt 71))) *
    (floor (real.sqrt (real.sqrt 73))) * (floor (real.sqrt (real.sqrt 75))) *
    (floor (real.sqrt (real.sqrt 77))) * (floor (real.sqrt (real.sqrt 79))) *
    (floor (real.sqrt (real.sqrt 81))) * (floor (real.sqrt (real.sqrt 83))) *
    (floor (real.sqrt (real.sqrt 85))) * (floor (real.sqrt (real.sqrt 87))) *
    (floor (real.sqrt (real.sqrt 89))) * (floor (real.sqrt (real.sqrt 91))) *
    (floor (real.sqrt (real.sqrt 93))) * (floor (real.sqrt (real.sqrt 95))) *
    (floor (real.sqrt (real.sqrt 97))) * (floor (real.sqrt (real.sqrt 99))) *
    (floor (real.sqrt (real.sqrt 101))) * (floor (real.sqrt (real.sqrt 103))) *
    (floor (real.sqrt (real.sqrt 105))) * (floor (real.sqrt (real.sqrt 107))) *
    (floor (real.sqrt (real.sqrt 109))) * (floor (real.sqrt (real.sqrt 111))) *
    (floor (real.sqrt (real.sqrt 113))) * (floor (real.sqrt (real.sqrt 115))) *
    (floor (real.sqrt (real.sqrt 117))) * (floor (real.sqrt (real.sqrt 119))) *
    (floor (real.sqrt (real.sqrt 121))) * (floor (real.sqrt (real.sqrt 123))) *
    (floor (real.sqrt (real.sqrt 125))) *
    ((floor (real.sqrt (real.sqrt 15))) * (floor (real.sqrt (real.sqrt 255))) *
    (floor (real.sqrt (real.sqrt 1295)))) ^ 2

theorem computeProduct_correct : 
    computeProduct = 225 * 2^21 := sorry

end computeProduct_correct_l161_161071


namespace closest_integer_to_sum_is_102_l161_161107

noncomputable def sum_term (n : ℕ) : ℝ := 1 / (n ^ 2 - 9)

noncomputable def compounded_sum (a b : ℕ) : ℝ := ∑ n in Finset.range (b - a + 1) \u4 { a + i | i ∈ Finset.range (b - a + 1) }, sum_term (a + n)

noncomputable def scaled_sum (a b : ℕ) : ℝ := 500 * compounded_sum a b

theorem closest_integer_to_sum_is_102 :
  Int.floor (scaled_sum 4 15000 + 0.5) = 102 :=
begin
  sorry
end

end closest_integer_to_sum_is_102_l161_161107


namespace probability_two_same_number_l161_161868

theorem probability_two_same_number :
  let rolls := 5
  let sides := 8
  let total_outcomes := sides ^ rolls
  let favorable_outcomes := 8 * 7 * 6 * 5 * 4
  let probability_all_different := (favorable_outcomes : ℚ) / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same = (3256 : ℚ) / 4096 :=
by 
  sorry

end probability_two_same_number_l161_161868


namespace upper_limit_of_Arun_weight_l161_161208

axiom Arun_weight_conditions (w : ℝ) (X : ℝ) : 
  (66 < w ∧ w < 72) ∧ 
  (60 < w ∧ w < X) ∧ 
  (w ≤ 69) ∧ 
  (68 = (66 + 69) / 2)

theorem upper_limit_of_Arun_weight (w X : ℝ) (h : Arun_weight_conditions w X) : 
  X = 69 :=
sorry

end upper_limit_of_Arun_weight_l161_161208


namespace modular_inverse_l161_161112

theorem modular_inverse (b : ℤ) (h1 : 35 * b ≡ 1 [MOD 36]) : b ≡ 35 [MOD 36] :=
sorry

end modular_inverse_l161_161112


namespace sum_of_digits_of_N_l161_161259

-- Define N
def N := 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 + 10^9

-- Define function to calculate sum of digits
def sum_of_digits(n: Nat) : Nat :=
  n.digits 10 |>.sum

-- Theorem statement
theorem sum_of_digits_of_N : sum_of_digits N = 7 :=
  sorry

end sum_of_digits_of_N_l161_161259


namespace acquaintance_bound_l161_161985

theorem acquaintance_bound
  {n d k : ℕ} (h₁ : d ≤ n)
  (h₂ : k ≤ d)
  (h₃ : ∃ (group : Finset ℕ), group.card = k ∧ ∀ x y ∈ group, x ≠ y → ¬ acquainted x y) :
  ∀ (m : ℕ), m ≤ (n^2 / 4) :=
begin
  sorry
end

end acquaintance_bound_l161_161985


namespace eval_nested_radical_l161_161093

-- The statement of the problem in Lean 4
theorem eval_nested_radical :
  let x := sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...))))
  in x = Real.sqrt 13 / 2 - 1/2 :=
sorry  -- No proof required

end eval_nested_radical_l161_161093


namespace painters_workdays_l161_161654

theorem painters_workdays (five_painters_days : ℝ) (four_painters_days : ℝ) : 
  (5 * five_painters_days = 9) → (4 * four_painters_days = 9) → (four_painters_days = 2.25) :=
by
  intros h1 h2
  sorry

end painters_workdays_l161_161654


namespace sum_m_is_neg_2_l161_161579

noncomputable def sum_of_integers_m_satisfying_conditions : ℤ :=
  let fractional_equation (x m : ℤ) := (x + m) / (x + 2) - m / (x - 2) = 1 in
  let inequalities_system (m y : ℤ) := (m - 6 * y > 2 ∧ y - 4 ≤ 3 * y + 4) in
  let num_integer_solutions (m : ℤ) := ∑ y in (Set.Icc Int.min_int Int.max_int), if inequalities_system m y then 1 else 0 in
  let valid_m_values := {m : ℤ | fractional_equation (2 - 2 * m) m ∧ (num_integer_solutions m = 4)} in
  ∑ m in valid_m_values, m

theorem sum_m_is_neg_2 : sum_of_integers_m_satisfying_conditions = -2 :=
  sorry

end sum_m_is_neg_2_l161_161579


namespace sphere_in_trihedral_angle_problem_l161_161432

noncomputable def angle_ks_o : Real :=
  Real.arcsin (1 / 5)

noncomputable def area_cross_section : Real :=
  144 / 25

theorem sphere_in_trihedral_angle_problem 
  (O S K L M: Point) 
  (plane1 plane2 plane3 : Plane) 
  (cross_section_area1 cross_section_area2 : Real) 
  (h1 : inscribed_sphere O S K L M) 
  (h2 : tangent_plane_area O S plane1 = 4) 
  (h3 : tangent_plane_area O S plane2 = 9) 
  (h4 : ∀ {A B C : Angle}, A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  : angle_ks_o = Real.arcsin (1 / 5) 
      ∧ area_cross_section = 144 / 25 := 
sorry

end sphere_in_trihedral_angle_problem_l161_161432


namespace solve_equation_l161_161304

theorem solve_equation (x : ℚ) : 3 * (x - 2) = 2 - 5 * (x - 2) ↔ x = 9 / 4 := by
  sorry

end solve_equation_l161_161304


namespace parts_outside_3sigma_l161_161634

noncomputable def normal_distribution_outside_3sigma (μ σ : ℝ) : ℕ := 3

theorem parts_outside_3sigma (μ σ : ℝ) :
  ∀ (num_parts : ℕ), num_parts = 1000 → 
  normal_distribution_outside_3sigma μ σ = 3 :=
by
  intros num_parts h
  rw h
  sorry

end parts_outside_3sigma_l161_161634


namespace floor_neg_seven_over_four_l161_161502

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l161_161502


namespace area_of_closed_figure_l161_161084

noncomputable def area_between_y_eq_x_and_y_eq_x_cube : ℝ :=
  ∫ x in -1..1, (x - x^3)

theorem area_of_closed_figure : area_between_y_eq_x_and_y_eq_x_cube = 1 / 2 :=
  sorry

end area_of_closed_figure_l161_161084


namespace eccentricity_range_l161_161162

-- Definitions for the problem
def isEllipse (x y : ℝ) (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2) + (y^2 / b^2) = 1

def isFocus (c a b : ℝ) : Prop :=
  c = Real.sqrt (a^2 - b^2)

def lineThroughFocus (l : ℝ → ℝ) (c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ y : ℝ, l y = k * y + c

-- Proving the main statement
theorem eccentricity_range (a b : ℝ) (F : ℝ × ℝ) (e : ℝ) :
  a > b > 0 ∧ F = (Real.sqrt (a^2 - b^2), 0) ∧
  (∀ l : ℝ → ℝ, lineThroughFocus l (Real.sqrt (a^2 - b^2)) → 
                ∃ A B : ℝ × ℝ, (isEllipse A.1 A.2 a b) ∧ (isEllipse B.1 B.2 a b) ∧
                (let OA := A in let OB := B in 
                  OA.1 * OB.1 + OA.2 * OB.2 = 0)) →
  e = Real.sqrt (1 - (b^2 / a^2)) →
  e ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 := 
by
  sorry

end eccentricity_range_l161_161162


namespace cost_of_headphones_l161_161063

-- Define the constants for the problem
def bus_ticket_cost : ℕ := 11
def drinks_and_snacks_cost : ℕ := 3
def wifi_cost_per_hour : ℕ := 2
def trip_hours : ℕ := 3
def earnings_per_hour : ℕ := 12
def total_earnings := earnings_per_hour * trip_hours
def total_expenses_without_headphones := bus_ticket_cost + drinks_and_snacks_cost + (wifi_cost_per_hour * trip_hours)

-- Prove the cost of headphones, H, is $16 
theorem cost_of_headphones : total_earnings = total_expenses_without_headphones + 16 := by
  -- setup the goal
  sorry

end cost_of_headphones_l161_161063


namespace return_to_original_position_l161_161041

structure Transformation :=
  (apply : ℝ × ℝ → ℝ × ℝ)
  (inverse : Transformation)

structure Square :=
  (W X Y Z : ℝ × ℝ)
  (transformation_sequence : list Transformation)

def identity_transformation := 
  { apply := id,
    inverse := ⟨id, id⟩ }

def rotation_90_clockwise := 
  { apply := λ (p : ℝ × ℝ), (2 + (p.snd - 2), 2 - (p.fst - 2)),
    inverse := sorry }

def rotation_90_counterclockwise := 
  { apply := λ (p : ℝ × ℝ), (2 - (p.snd - 2), 2 + (p.fst - 2)),
    inverse := sorry }

def reflection_y_eq_2 := 
  { apply := λ (p : ℝ × ℝ), (p.fst, 4 - p.snd),
    inverse := sorry }

def reflection_x_eq_2 := 
  { apply := λ (p : ℝ × ℝ), (4 - p.fst, p.snd),
    inverse := sorry }

def transform_square (sq : Square) (seq : list Transformation) : Square :=
{ W := seq.foldl (λ acc f, f.apply acc) sq.W,
  X := seq.foldl (λ acc f, f.apply acc) sq.X,
  Y := seq.foldl (λ acc f, f.apply acc) sq.Y,
  Z := seq.foldl (λ acc f, f.apply acc) sq.Z,
  transformation_sequence := seq }

def sq := 
{ W := (2, 3),
  X := (3, 2),
  Y := (2, 1),
  Z := (1, 2),
  transformation_sequence := [] }

theorem return_to_original_position : 
  ∃ seq : list Transformation, 
    transform_square sq seq = sq ∧ seq.length = 12 ∧
    seq.all (λ t, t = rotation_90_clockwise ∨ t = rotation_90_counterclockwise ∨ 
                      t = reflection_y_eq_2 ∨ t = reflection_x_eq_2) ∧
    seq.nat_pow 2 = 2 ^ 22 :=
sorry

end return_to_original_position_l161_161041


namespace area_not_touched_by_ball_l161_161040

theorem area_not_touched_by_ball :
  (∃ R a ε, R = 1 ∧ a = 4 * sqrt 6 ∧ δ = a / 2 ∧
   all_faces : area_of_tetra_inside_not_touched_by_ball =
   72 * sqrt 3) :=
by
  sorry

end area_not_touched_by_ball_l161_161040


namespace cos_pi_minus_2alpha_l161_161144

theorem cos_pi_minus_2alpha (α : ℝ) (h : cos (π / 2 - α) = 1 / 3) : cos (π - 2 * α) = -7 / 9 :=
sorry

end cos_pi_minus_2alpha_l161_161144


namespace sin_product_identity_l161_161087

noncomputable def deg_to_rad (x : ℝ) : ℝ := x * real.pi / 180

theorem sin_product_identity : 
  Real.sin (deg_to_rad 10) * Real.sin (deg_to_rad 50) * Real.sin (deg_to_rad 70) = 1 / 8 :=
by
  sorry

end sin_product_identity_l161_161087


namespace infinite_nested_radical_l161_161099

theorem infinite_nested_radical : 
  (x : Real) (h : x = Real.sqrt (3 - x)) : 
  x = (-1 + Real.sqrt 13) / 2 :=
by
  sorry

end infinite_nested_radical_l161_161099


namespace complex_div_conjugate_l161_161939

-- Definition of the complex number z
def z : ℂ := 4 - I

-- Definition of the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- The theorem stating the required proof
theorem complex_div_conjugate (z : ℂ) (hz : z = 4 - I) : (z_conjugate / z) = (15 / 17) + (8 / 17) * I :=
by
  sorry

end complex_div_conjugate_l161_161939


namespace largest_integer_x_l161_161896

theorem largest_integer_x (x : ℤ) : 
  (0.2 : ℝ) < (x : ℝ) / 7 ∧ (x : ℝ) / 7 < (7 : ℝ) / 12 → x = 4 :=
sorry

end largest_integer_x_l161_161896


namespace isosceles_similar_conditions_l161_161791

def is_isosceles (T : Triangle) : Prop := 
  T.AB = T.AC ∨ T.BC = T.BA ∨ T.CA = T.CB

def is_right_triangle (T : Triangle) : Prop := 
  T.angle_A = 90 ∨ T.angle_B = 90 ∨ T.angle_C = 90

def is_similar (T1 T2 : Triangle) : Prop := 
  ∃ (f : T1 → T2), f.is_similitude

theorem isosceles_similar_conditions (T1 T2 : Triangle) :
  (is_isosceles T1 ∧ is_isosceles T2) →
  ((is_right_triangle T1 ∧ is_right_triangle T2) ∨ (T1.angle_A = T2.angle_A ∧ is_isosceles T1 ∧ is_isosceles T2)) →
  is_similar T1 T2 :=
sorry

end isosceles_similar_conditions_l161_161791


namespace min_value_proof_l161_161966

noncomputable def min_value (a b : ℝ) (h : log 3 (2 * a + b) = 1 + log (sqrt 3) (sqrt (a * b))) : ℝ :=
  a + 2 * b

theorem min_value_proof : 
  ∀ a b : ℝ, (log 3 (2 * a + b) = 1 + log (sqrt 3) (sqrt (a * b))) → (∃ a b : ℝ, a + 2 * b = 3) :=
begin
  intros a b h,
  sorry
end

end min_value_proof_l161_161966


namespace f_eq_g_at_3_l161_161734

variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, 2 < x ∧ x < 4 → 2 < f x ∧ f x < 4)
variable (h2 : ∀ x, 2 < x ∧ x < 4 → 2 < g x ∧ g x < 4)
variable (h3 : ∀ x, 2 < x ∧ x < 4 → f (g x) = x ∧ g (f x) = x)
variable (h4 : ∀ x, 2 < x ∧ x < 4 → f x * g x = x^2)

theorem f_eq_g_at_3 (h1 h2 h3 h4) : f 3 = g 3 :=
  sorry

end f_eq_g_at_3_l161_161734


namespace program1_values_program2_values_l161_161378

theorem program1_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧
  a = -5 ∧ b = 8 ∧ c = 8 :=
by sorry

theorem program2_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧ c = a ∧
  a = -5 ∧ b = 8 ∧ c = -5 :=
by sorry

end program1_values_program2_values_l161_161378


namespace PQRS_is_parallelogram_and_area_correct_l161_161950

noncomputable def point := (ℝ × ℝ × ℝ)

def p : point := (2, -2, 1)
def q : point := (4, -6, 3)
def r : point := (3, -3, 0)
def s : point := (5, -7, 2)

def vector_sub (a b : point) : point :=
  (a.1 - b.1, a.2 - b.2, a.3 - b.3)

def cross_product (u v : point) : point :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def vector_magnitude (v : point) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def is_parallelogram : Prop :=
  vector_sub q p = vector_sub s r

def parallelogram_area : ℝ :=
  vector_magnitude (cross_product (vector_sub q p) (vector_sub r p))

theorem PQRS_is_parallelogram_and_area_correct :
  is_parallelogram ∧ parallelogram_area = 2 * real.sqrt 6 :=
  by
    unfold is_parallelogram,
    unfold parallelogram_area,
    unfold vector_sub,
    unfold cross_product,
    unfold vector_magnitude,
    sorry

end PQRS_is_parallelogram_and_area_correct_l161_161950


namespace circle_area_l161_161636

theorem circle_area (x y : ℝ) :
  (3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
  (π * ((1 / 2) * (1 / 2)) = (π / 4)) := 
by
  intro h
  sorry

end circle_area_l161_161636


namespace proof_problem_l161_161251

def floor (x : ℝ) : ℤ := ⌊x⌋
def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem proof_problem
  (n : ℕ)
  (p : ℕ → ℕ)
  (p_odd_prime : ∀ i, i < 2*n → nat.prime (p i) ∧ p i % 2 = 1)
  (sum_eq_zero_mod_4 : ∑ i in finset.range (2*n), p i % 4 = 0)
  (gcd_two_n_one : nat.gcd 2 n = 1)
  (non_quadratic_residue :
    ∀ i, i < 2*n →
    ¬ ∃ x, (4 * (finset.sum (finset.range 1006) (λ k : ℕ, (-1)^⌊2016 * k / 1000⌋ * frac (2016 * k / 1007)) + 2 / 1007) ≡ x^2 [MOD (p i)]) ) :
  4 ∣ nat.sigma (finset.prod (finset.range (2 * n)) (λ i, p i)) :=
sorry

end proof_problem_l161_161251


namespace monotonic_intervals_and_extreme_values_tangent_line_at_origin_l161_161946

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem monotonic_intervals_and_extreme_values :
  (∀ x ∈ Ioo (-1 : ℝ) 1, deriv f x < 0) ∧
  (∀ x ∈ Ioo (-∞ : ℝ) (-1), deriv f x > 0) ∧
  (∀ x ∈ Ioo (1 : ℝ) (+∞), deriv f x > 0) ∧
  (f (-1) = 3) ∧
  (f (1) = -1) :=
sorry

theorem tangent_line_at_origin :
  ∃ m b : ℝ, tangent_line_at f 0 = (λ x : ℝ, -3 * x + 1) ∧
  tangent_line_at f 0 = (λ x : ℝ, m * x + b) :=
sorry

end monotonic_intervals_and_extreme_values_tangent_line_at_origin_l161_161946


namespace parallelepiped_analogy_l161_161389

-- Define the possible plane figures
inductive PlaneFigure
| Triangle
| Trapezoid
| Parallelogram
| Rectangle

-- Define the concept of a parallelepiped
structure Parallelepiped : Type

-- The theorem asserting the parallelogram is the correct analogy
theorem parallelepiped_analogy : 
  ∀ (fig : PlaneFigure), 
    (fig = PlaneFigure.Parallelogram) ↔ 
    (fig = PlaneFigure.Parallelogram) :=
by sorry

end parallelepiped_analogy_l161_161389


namespace compare_f_values_l161_161320

noncomputable def f : ℝ → ℝ := sorry

axiom differentiable_f : differentiable ℝ f
axiom periodicity_f : ∀ x : ℝ, f(x + 2) - f(x) = 2 * f(1)
axiom symmetry_f : ∀ x : ℝ, f(x + 1) = f(-x - 1)
axiom value_f : ∀ x : ℝ, (2 ≤ x ∧ x ≤ 4) → f(x) = x^2 + 2 * x * (deriv f 2)

theorem compare_f_values : f (-1 / 2) < f (16 / 3) := by
  sorry

end compare_f_values_l161_161320


namespace f_ff_neg4_l161_161556

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then f (x + 1) else x^2 - 3 * x - 4

theorem f_ff_neg4 : f (f (-4)) = -6 :=
by {
  -- This is where the proof would go
  sorry
}

end f_ff_neg4_l161_161556


namespace f_f_neg_4_eq_neg_6_l161_161553

def f (x : ℝ) : ℝ :=
if x ≤ 0 then f (x + 1) else x^2 - 3 * x - 4

-- Our goal is to prove the following statement
theorem f_f_neg_4_eq_neg_6 : f (f (-4)) = -6 := 
by
  sorry

end f_f_neg_4_eq_neg_6_l161_161553


namespace triangles_with_positive_area_l161_161609

theorem triangles_with_positive_area (x y : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 5) (h₂ : 1 ≤ y ∧ y ≤ 3) : 
    ∃ (n : ℕ), n = 420 := 
sorry

end triangles_with_positive_area_l161_161609


namespace game_final_configurations_l161_161286

theorem game_final_configurations : 
  let initial_white_checkers : ℕ := 2
  let initial_black_checker : ℕ := 1
  let total_squares : ℕ := 2011
  let required_moves := total_squares - (initial_white_checkers + initial_black_checker)
  required_moves + 1 = 2009 :=
by {
  let initial_grid : list (char × ℕ) := [('W', 1), ('W', 1), ('B', 1)] ++ list.repeat ('_', (total_squares - 3))
  let run : nat → nat := λ n, n + 1
  let fight : nat → nat × nat := λ n, (n + 1, n + 1)
  let commutative_property : ∀ (runs fights : nat), run (run (required_moves - fights)) + fights = required_moves :=
    λ runs fights, by { sorry }
  let final_configurations : nat := required_moves + 1
  exact concl == final_configurations
}

end game_final_configurations_l161_161286


namespace limit_does_not_exist_l161_161103

noncomputable def f (x : ℝ) : ℝ := x^2 - 3*x + 2
noncomputable def g (x : ℝ) : ℝ := |x^2 - 6*x + 8|

theorem limit_does_not_exist : ¬ (∃ L : ℝ, tendsto (λ x : ℝ, f x / g x) (𝓝 2) (𝓝 L)) :=
by
  sorry

end limit_does_not_exist_l161_161103


namespace simplify_expression_l161_161783

theorem simplify_expression : (3^3 * 3^(-4)) / (3^2 * 3^(-5)) = 1 / 6561 := by
  sorry

end simplify_expression_l161_161783


namespace probability_fewer_tails_than_heads_l161_161788

theorem probability_fewer_tails_than_heads : 
  (∃ (prob_eq : ℚ), prob_eq = (70 / 256 : ℚ) ∧ 
  (∃ (prob_fewer : ℚ), prob_fewer = (93 / 256 : ℚ) ∧
  2 * prob_fewer + prob_eq = 1)) :=
by
  -- Definition based on the conditions and the binomial theorem
  let total_outcomes : ℚ := 256
  let prob_equal_heads_tails : ℚ := 70 / total_outcomes
  let prob_fewer_tails_heads : ℚ := 93 / total_outcomes
  existsi prob_equal_heads_tails
  existsi prob_fewer_tails_heads
  split
  · show prob_equal_heads_tails = 70 / total_outcomes; rfl
  split
  · show prob_fewer_tails_heads = 93 / total_outcomes; rfl
  -- Ensure the final condition holds
  show 2 * prob_fewer_tails_heads + prob_equal_heads_tails = 1
  calc
    2 * prob_fewer_tails_heads + prob_equal_heads_tails 
        = 2 * (93 / total_outcomes) + prob_equal_heads_tails : by rw rfl
    ... = 2 * (93 / 256) + 70 / 256 : by rw rfl
    ... = 186 / 256 + 70 / 256 : by norm_num
    ... = (186 + 70) / 256 : by ring
    ... = 256 / 256 : by norm_num
    ... = 1 : by norm_num

end probability_fewer_tails_than_heads_l161_161788


namespace tan_x_eq_cot_sum_sec2_x_eq_csc2_sum_l161_161624

theorem tan_x_eq_cot_sum (A B C x : ℝ) (h1 : cos (x + A) * cos (x + B) * cos (x + C) + cos x ^ 3 = 0) (h2 : A + B + C = π) :
  tan x = cot A + cot B + cot C :=
sorry

theorem sec2_x_eq_csc2_sum (A B C x : ℝ) (h1 : cos (x + A) * cos (x + B) * cos (x + C) + cos x ^ 3 = 0) (h2 : A + B + C = π) :
  sec x ^ 2 = csc A ^ 2 + csc B ^ 2 + csc C ^ 2 :=
sorry

end tan_x_eq_cot_sum_sec2_x_eq_csc2_sum_l161_161624


namespace eval_nested_radical_l161_161094

-- The statement of the problem in Lean 4
theorem eval_nested_radical :
  let x := sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...))))
  in x = Real.sqrt 13 / 2 - 1/2 :=
sorry  -- No proof required

end eval_nested_radical_l161_161094


namespace simplify_sin_l161_161301

theorem simplify_sin (α : ℝ) : sin (π / 2 - α) = cos α :=
  sorry

end simplify_sin_l161_161301


namespace eval_floor_neg_seven_fourths_l161_161509

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l161_161509


namespace train_crosses_pole_in_3_seconds_l161_161841

def train_speed_kmph : ℝ := 60
def train_length_m : ℝ := 50

def speed_conversion (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

def crossing_time (distance_m : ℝ) (speed_mps : ℝ) : ℝ := distance_m / speed_mps

theorem train_crosses_pole_in_3_seconds :
  crossing_time train_length_m (speed_conversion train_speed_kmph) = 3 :=
by
  sorry

end train_crosses_pole_in_3_seconds_l161_161841


namespace cos_difference_simplification_l161_161297

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  x - y = -1 / (2 * Real.sqrt 5) :=
sorry

end cos_difference_simplification_l161_161297


namespace count_irrational_numbers_l161_161052

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), ↑q = x

-- Given real numbers
def numbers : list ℝ := [-real.sqrt 3, 0.21, real.pi / 2, 22 / 7, real.cbrt 9, 0.20202]

-- Define the set of irrational numbers based on the problem conditions
def irrational_numbers := {x ∈ numbers | is_irrational x}

theorem count_irrational_numbers : irrational_numbers.card = 3 := by sorry

end count_irrational_numbers_l161_161052


namespace train_crosses_pole_in_3_seconds_l161_161837

def train_problem (speed_kmh : ℕ) (length_m : ℕ) : ℕ :=
  let speed_ms := (speed_kmh * 1000) / 3600 in
  length_m / speed_ms

theorem train_crosses_pole_in_3_seconds :
  train_problem 60 50 = 3 :=
by
  -- We add a 'sorry' to skip the proof
  sorry

end train_crosses_pole_in_3_seconds_l161_161837


namespace sum_of_possible_b_values_l161_161760

theorem sum_of_possible_b_values :
  (∑ b in {b : ℤ | ∃ p q : ℤ, p + q = b ∧ p * q = 3 * b}, b) = 84 :=
by
  sorry

end sum_of_possible_b_values_l161_161760


namespace determine_angle_B_l161_161567

open_locale real
open_locale vector_space

variables {A B C G : Type*}

-- Given that G is the centroid of triangle ABC
def is_centroid (G A B C : Type*) : Prop :=
  ∀ (GA GB GC : vector_space ℝ),
  GA + GB + GC = 0

-- Given the condition
def given_condition (sinA sinB sinC : ℝ) (GA GB GC : vector_space ℝ) : Prop :=
  sinA • GA + sinB • GB + sinC • GC = 0

theorem determine_angle_B (G A B C : Type*)
  (sinA sinB sinC : ℝ)
  (GA GB GC : vector_space ℝ)
  (centroid_cond : is_centroid G A B C)
  (cond : given_condition sinA sinB sinC GA GB GC) :
  B = π / 3 :=
  sorry

end determine_angle_B_l161_161567


namespace Vanya_433_sum_l161_161377

theorem Vanya_433_sum : 
  ∃ (A B : ℕ), 
  A + B = 91 
  ∧ (3 * A + 7 * B = 433) 
  ∧ (∃ (subsetA subsetB : Finset ℕ),
      (∀ x ∈ subsetA, x ∈ Finset.range (13 + 1))
      ∧ (∀ x ∈ subsetB, x ∈ Finset.range (13 + 1))
      ∧ subsetA ∩ subsetB = ∅
      ∧ subsetA ∪ subsetB = Finset.range (13 + 1)
      ∧ subsetA.card = 5
      ∧ subsetA.sum id = A
      ∧ subsetB.sum id = B) :=
by
  sorry

end Vanya_433_sum_l161_161377


namespace football_kick_distance_l161_161463

theorem football_kick_distance (a : ℕ) (avg : ℕ) (x : ℕ)
  (h1 : a = 43)
  (h2 : avg = 37)
  (h3 : 3 * avg = a + 2 * x) :
  x = 34 :=
by
  sorry

end football_kick_distance_l161_161463


namespace polynomial_degree_is_five_l161_161875

noncomputable def expr1 := λ (x : ℚ), x^3
noncomputable def expr2 := λ (x : ℚ), x^2 - 1 / x^2
noncomputable def expr3 := λ (x : ℚ), 1 - 1 / x + 1 / x^3

noncomputable def product := λ (x : ℚ), (expr1(x) * expr2(x)) * expr3(x)
noncomputable def degree := polynomial.degree (polynomial.COEFFS product)

theorem polynomial_degree_is_five : ∀ x : ℚ, polynomial.degree (product x) = 5 := 
by
  sorry

end polynomial_degree_is_five_l161_161875


namespace floor_of_neg_seven_fourths_l161_161516

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l161_161516


namespace choose_100_disjoint_chords_same_sum_l161_161363

theorem choose_100_disjoint_chords_same_sum (n : ℕ) (h : n = 2^500) :
  ∃ (chords : Finset (Fin n × Fin n)), 
    chords.card = 100 ∧ 
    (∀ (a b : Fin n × Fin n) (ha : a ∈ chords) (hb : b ∈ chords), 
      a ≠ b → disjoint (Finset.singleton a.1 ∪ Finset.singleton a.2) (Finset.singleton b.1 ∪ Finset.singleton b.2)) ∧
    (∃ k : ℕ, ∀ (a : Fin n × Fin n) (ha : a ∈ chords), a.1.val + a.2.val = k) :=
sorry

end choose_100_disjoint_chords_same_sum_l161_161363


namespace cube_4_is_sum_of_consecutive_odds_k_cubed_sum_35_is_k_6_middle_numbers_for_k_10_sum_of_cubes_to_11_l161_161220

-- Prove that 4^3 is the sum of the consecutive odd numbers 13, 15, 17, and 19.
theorem cube_4_is_sum_of_consecutive_odds :
  4^3 = 13 + 15 + 17 + 19 :=
sorry

-- Prove that if k^3 is expressed as the sum of k consecutive odd numbers and one of them is 35, then k = 6.
theorem k_cubed_sum_35_is_k_6 (k : ℕ) (h : ∃ (seq : ℕ → ℕ), (∀ n, seq n = 2*n + (k^2 - k + 1)) ∧
  (k : ℕ) ∧ seq 4 = 35) : k = 6 :=
sorry

-- Prove that for k = 10, the 5th and 6th numbers in the sequence summing up to 10^3 are 99 and 101.
noncomputable def sequence (n k : ℕ) := 2*n + (k^2 - k + 1)

theorem middle_numbers_for_k_10 :
  let seq := sequence in seq 4 10 = 99 ∧ seq 5 10 = 101 :=
sorry

-- Prove that the sum of cubes from 1^3 to 11^3 is 4356.
theorem sum_of_cubes_to_11 :
  ∑ n in Finset.range (11 + 1), n^3 = 4356 :=
sorry

end cube_4_is_sum_of_consecutive_odds_k_cubed_sum_35_is_k_6_middle_numbers_for_k_10_sum_of_cubes_to_11_l161_161220


namespace least_number_to_divisible_l161_161386

theorem least_number_to_divisible (k : ℕ) (h : k = 29989) : ∃ n : ℕ, (k + n) % 73 = 0 ∧ n = 21 :=
by
  use 21
  split
  · have prem : k % 73 = 52 := sorry
    rw [← add_mod_left, ← prem]
    norm_num
  · rfl

end least_number_to_divisible_l161_161386


namespace B_initial_investment_l161_161013

theorem B_initial_investment (x : ℝ) (hA_initial : 6000) (hA_withdraw : 1000) (hB_advance : 1000) 
  (h_total_profit : 630) (hA_share : 357) (hB_share : h_total_profit - hA_share) 
  (hA_investment : (6000 * 8 + (6000 - 1000) * 4)) 
  (hB_investment : (x * 8 + (x + 1000) * 4)) :
  (hA_share / hA_investment = hB_share / hB_investment) → x = 4000 := 
by
  sorry

end B_initial_investment_l161_161013


namespace num_satisfying_elements_l161_161690

-- Definitions based on given conditions
inductive Elements : Type
| A0
| A1
| A2
| A3

open Elements

def op (x y : Elements) : Elements :=
  match x, y with
  | A0, A0 => A0
  | A0, A1 => A1
  | A0, A2 => A2
  | A0, A3 => A3
  | A1, A0 => A1
  | A1, A1 => A2
  | A1, A2 => A3
  | A1, A3 => A0
  | A2, A0 => A2
  | A2, A1 => A3
  | A2, A2 => A0
  | A2, A3 => A1
  | A3, A0 => A3
  | A3, A1 => A0
  | A3, A2 => A1
  | A3, A3 => A2

-- The proof statement
theorem num_satisfying_elements : 
  (List.length (List.filter (λ x, op (op x x) A2 = A0) [A0, A1, A2, A3])) = 2
  := 
  sorry

end num_satisfying_elements_l161_161690


namespace bill_is_19_l161_161862

variable (C : ℕ) -- Caroline's age
variable (BillAge DanielAge AlexAge GrandmotherAge : ℕ)
variable (totalAge : ℕ)

-- Conditions
def bill_age_condition := BillAge = 2 * C - 1
def daniel_age_condition := DanielAge = C / 2
def alex_age_condition := AlexAge = BillAge
def grandmother_age_condition := GrandmotherAge = 4 * C
def total_age_condition := C + BillAge + DanielAge + AlexAge + GrandmotherAge = 108

-- Prove that Bill is 19 years old given the conditions
theorem bill_is_19 
  (bill_age : bill_age_condition)
  (daniel_age : daniel_age_condition)
  (alex_age : alex_age_condition)
  (grandmother_age : grandmother_age_condition)
  (total_age : total_age_condition) 
: BillAge = 19 := sorry

end bill_is_19_l161_161862


namespace beetles_eaten_per_day_l161_161465
-- Import the Mathlib library

-- Declare the conditions as constants
def bird_eats_beetles_per_day : Nat := 12
def snake_eats_birds_per_day : Nat := 3
def jaguar_eats_snakes_per_day : Nat := 5
def number_of_jaguars : Nat := 6

-- Define the theorem and provide the expected proof
theorem beetles_eaten_per_day :
  12 * (3 * (5 * 6)) = 1080 := by
  sorry

end beetles_eaten_per_day_l161_161465


namespace regular_hexagon_area_l161_161427

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem regular_hexagon_area 
  (A C : ℝ × ℝ)
  (hA : A = (0, 0))
  (hC : C = (8, 2))
  (h_eq_side_length : ∀ x y : ℝ × ℝ, dist A.1 A.2 C.1 C.2 = dist x.1 x.2 y.1 y.2) :
  hexagon_area = 34 * Real.sqrt 3 :=
by
  -- sorry indicates the proof is omitted
  sorry

end regular_hexagon_area_l161_161427


namespace geometric_properties_of_triangles_l161_161056

theorem geometric_properties_of_triangles
    (A B C D E F G : Type *)
    (h1 : acute_triangle A B C)
    (h2: isosceles_right_triangle B C D)
    (h3: isosceles_right_triangle A B E)
    (h4: isosceles_right_triangle C A F)
    (h5: ∠BDC = 90)
    (h6: ∠BAE = 90)
    (h7: ∠CFA = 90)
    (h8: isosceles_right_triangle E F G)
    (h9: quadrilateral B C F E)
    (h10: ∠EFG = 90) :
    CA = √2 \ AD ∧ ∠GAD = 135 :=
by
  sorry

end geometric_properties_of_triangles_l161_161056


namespace yard_length_calculation_l161_161628

theorem yard_length_calculation (n_trees : ℕ) (distance : ℕ) (h1 : n_trees = 26) (h2 : distance = 32) : (n_trees - 1) * distance = 800 :=
by
  -- This is where the proof would go.
  sorry

end yard_length_calculation_l161_161628


namespace first_prize_probability_any_prize_probability_l161_161090

open ProbabilityTheory Classical

-- Assume we have 6 balls: 3 red labeled A, B, C and 3 white labeled by any white identity f'(x0) = 0.

def balls : Finset (String × Bool) := 
  { ("A", true), ("B", true), ("C", true), ("f'(x_0)=0", false), ("f'(x_0)=0", false), ("f'(x_0)=0", false) }

def draw (s : Finset (String × Bool)) : Finset (Finset (String × Bool)) :=
  s.powerset.filter (λ x, x.card = 2)

-- Define the probability definitions for the first and any prize case
def prob_first_prize (s : Finset (String × Bool)) : ℚ :=
  ((draw s).filter (λ x, x.filter (λ y, y.2 = true).card = 2).card : ℚ) / (draw s).card

def prob_any_prize (s : Finset (String × Bool)) : ℚ :=
  1 - (((draw s).filter (λ x, x.filter (λ y, y.2 = false).card = 2).card : ℚ) / (draw s).card)

-- Theorems
theorem first_prize_probability : prob_first_prize balls = 1 / 5 := 
by
  sorry

theorem any_prize_probability : prob_any_prize balls = 4 / 5 := 
by
  sorry

end first_prize_probability_any_prize_probability_l161_161090


namespace parallelepiped_intersection_l161_161131

/-- Given a parallelepiped A B C D A₁ B₁ C₁ D₁.
    Point X is chosen on edge A₁ D₁, and point Y is chosen on edge B C.
    It is known that A₁ X = 5, B Y = 3, and B₁ C₁ = 14.
    The plane C₁ X Y intersects ray D A at point Z.
    Prove that D Z = 20. -/
theorem parallelepiped_intersection
  (A B C D A₁ B₁ C₁ D₁ X Y Z : ℝ)
  (h₁: A₁ - X = 5)
  (h₂: B - Y = 3)
  (h₃: B₁ - C₁ = 14) :
  D - Z = 20 :=
sorry

end parallelepiped_intersection_l161_161131


namespace length_AB_is_sqrt_15_l161_161171

-- Conditions: 
def parabola (p : ℝ) := {A | ∃ (x y : ℝ), y^2 = 2 * p * x ∧ p > 0}
def focus := (1 : ℝ, 0 : ℝ)
def point_P := (1 : ℝ, 1 : ℝ)
def line_through_P (k : ℝ) := {A | ∃ (x y : ℝ), y - 1 = k * (x - 1)}

-- Let points A and B be the intersection points of the line and the parabola
def intersection_AB (p k : ℝ) := 
  {A | ∃ (x y : ℝ), (y^2 = 2 * p * x ∧ y - 1 = k * (x - 1))}

-- Condition: P is midpoint of AB
def P_is_midpoint (A B : ℝ × ℝ) := 
  point_P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Length of segment AB
def length_AB (A B : ℝ × ℝ) := 
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Main statement to prove
theorem length_AB_is_sqrt_15 (p k : ℝ) (A B : ℝ × ℝ) 
  (h_parabola : parabola p) 
  (h_focus : focus = (1, 0)) 
  (h_point_P : point_P = (1, 1)) 
  (h_line : line_through_P k) 
  (h_intersection : intersection_AB p k) 
  (h_midpoint : P_is_midpoint A B) :
  length_AB A B = real.sqrt 15 := 
sorry

end length_AB_is_sqrt_15_l161_161171


namespace speed_conversion_l161_161832

theorem speed_conversion (speed_mps : ℝ) (h : speed_mps = 200.016) : 
  let conversion_factor : ℝ := 3.6 in
  speed_mps * conversion_factor = 720.0576 :=
by
  sorry

end speed_conversion_l161_161832


namespace greatest_value_x_plus_y_l161_161785

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end greatest_value_x_plus_y_l161_161785


namespace enclosed_area_onepoint57_l161_161668

noncomputable def g (x : ℝ) : ℝ :=
  1 + Real.sqrt (1 - x^2)

theorem enclosed_area_onepoint57 :
  let area := π / 2
  in Real.floor (100 * area) / 100 = 1.57 :=
by
  -- Let area first
  let area := π / 2
  -- The proof of the computation goes here.
  sorry

end enclosed_area_onepoint57_l161_161668


namespace exists_S_proper_not_Zplus_proper_l161_161660

-- Define S-proper condition
def S_proper (A S : Set ℕ) : Prop :=
  ∃ N : ℕ, ∀ (a ∈ A) (b : ℕ), 0 ≤ b ∧ b < a → 
    ∃ (n : ℕ) (s : Fin n → ℕ), (∀ i, s i ∈ S) ∧ (1 ≤ n ∧ n ≤ N) ∧ (b ≡ ∑ i in Finset.range n, s ⟨i, Finset.mem_range_succ_of_le (Nat.le_of_lt (Finset.mem_range.mp i.val_lt))⟩[natAddGroup a])

-- Define the statement to prove
theorem exists_S_proper_not_Zplus_proper :
  ∃ S : Set ℕ, S ⊆ Set.univ ∧ S_proper (SetOf Nat.Prime) S ∧ ¬ S_proper Set.univ S :=
begin
  sorry
end

end exists_S_proper_not_Zplus_proper_l161_161660


namespace infinite_nested_sqrt_l161_161097

theorem infinite_nested_sqrt :
  ∃ x : ℝ, x = sqrt (3 - x) ∧ x = ( -1 + sqrt 13) / 2 :=
begin
  sorry
end

end infinite_nested_sqrt_l161_161097


namespace distance_midpoint_O_l161_161446

def cos : ℝ → ℝ := real.cos
def sin : ℝ → ℝ := real.sin

noncomputable def A : ℝ × ℝ := (cos (110 * real.pi / 180), sin (110 * real.pi / 180))
noncomputable def B : ℝ × ℝ := (cos (50 * real.pi / 180), sin (50 * real.pi / 180))

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def distance_from_origin (P : ℝ × ℝ) : ℝ := real.sqrt (P.1^2 + P.2^2)

theorem distance_midpoint_O :
  distance_from_origin (midpoint A B) = sqrt 3 / 2 := 
sorry

end distance_midpoint_O_l161_161446


namespace find_a_l161_161204

theorem find_a (a : ℝ) (h : (∃ k : ℝ, ax + 2 * y - 1 = 0 ∧ 2 * x + y - 1 = 0) ∧ (ax + 2 * y - 1 = 0) ⊥ (2 * x + y - 1 = 0)) : a = -1 := 
by
  sorry

end find_a_l161_161204


namespace problem_solution_l161_161261

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 else -x^2

noncomputable def f_inv (y : ℝ) : ℝ :=
if y >= 0 then real.sqrt y else -real.sqrt (-y)

theorem problem_solution :
  (f_inv 9)^2 + (f_inv (-49))^2 = 58 := by
  sorry

end problem_solution_l161_161261


namespace probability_of_distance_less_than_8_l161_161857
noncomputable theory

def probability_AM_lessThan_8 (d_AB d_BC θ: ℝ) (h_AB : d_AB = 10) (h_BC : d_BC = 6) (h_θ : 0 < θ ∧ θ < π) : ℝ :=
  let distance_from_circle_center := sqrt ((d_AB * sin θ)^2 + (d_AB * cos θ - d_BC)^2)
  (if distance_from_circle_center < 8 then θ else 0) / π

theorem probability_of_distance_less_than_8 :
  ∀ (d_AB d_BC: ℝ) (h_AB : d_AB = 10) (h_BC : d_BC = 6),
  (∫ θ in 0..π, if distance_from_circle_center d_AB d_BC θ h_AB h_BC < 8 then 1 else 0) / π = 1 / 6 :=
by
  sorry

end probability_of_distance_less_than_8_l161_161857


namespace total_selling_price_l161_161423

theorem total_selling_price
  (CP : ℕ) (Gain : ℕ) (TCP : ℕ)
  (h1 : CP = 1200)
  (h2 : Gain = 3 * CP)
  (h3 : TCP = 18 * CP) :
  ∃ TSP : ℕ, TSP = 25200 := 
by
  sorry

end total_selling_price_l161_161423


namespace distinct_ways_to_distribute_balls_l161_161964

theorem distinct_ways_to_distribute_balls (h₁ : ∀ b₁ b₂ : ℕ, b₁ = b₂) (h₂ : ∀ b₁ b₂ : ℕ, b₁ = b₂) (h₃ : ∀ b : ℕ, b ≥ 1) : finset.card {s : finset (multiset ℕ) | finset.card s = 4 ∧ multiset.card (finset.sum s) = 6 ∧ ∀ b ∈ s, multiset.card b ≥ 1} = 2 := by
  sorry

end distinct_ways_to_distribute_balls_l161_161964


namespace max_g_l161_161933

theorem max_g : 
  (∀ f : ℝ → ℝ, 
     (∀ x : ℝ, f x = x ^ (-2)) ∧ f 3 = 1 / 9 → 
     ∀ x ∈ set.Icc 1 3, is_max_on (λ x, (x - 1) * f x) (set.Icc 1 3) x →
     ∃ c ∈ set.Icc 1 3, (x - 1) * f x = 1 / 4  ) sorry

end max_g_l161_161933


namespace three_f_x_expression_l161_161147

variable (f : ℝ → ℝ)
variable (h : ∀ x > 0, f (3 * x) = 3 / (3 + 2 * x))

theorem three_f_x_expression (x : ℝ) (hx : x > 0) : 3 * f x = 27 / (9 + 2 * x) :=
by sorry

end three_f_x_expression_l161_161147


namespace lighthouses_visible_from_anywhere_l161_161544

-- A theorem that proves four arbitrary placed lighthouses with each lamp illuminating 90 degrees of angle
-- can be rotated such that at least one lamp is visible from every point in the plane.
theorem lighthouses_visible_from_anywhere (lighthouse : Fin 4 → Point) (angle : Fin 4 → ℝ) : 
  (∀ i : Fin 4, angle i = 90) →
  ∃ (orientation : Fin 4 → ℝ), 
    ∀ (p : Point), (∃ i : Fin 4, lamp_visible_from_point (orientation i) (angle i) (lighthouse i) p) :=
by
  sorry

end lighthouses_visible_from_anywhere_l161_161544


namespace f_f_neg_4_eq_neg_6_l161_161554

def f (x : ℝ) : ℝ :=
if x ≤ 0 then f (x + 1) else x^2 - 3 * x - 4

-- Our goal is to prove the following statement
theorem f_f_neg_4_eq_neg_6 : f (f (-4)) = -6 := 
by
  sorry

end f_f_neg_4_eq_neg_6_l161_161554


namespace min_distance_zero_l161_161004

variable (U g τ : ℝ)

def y₁ (t : ℝ) : ℝ := U * t - (g * t^2) / 2
def y₂ (t : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2
def s (t : ℝ) : ℝ := |U * τ - g * t * τ + (g * τ^2) / 2|

theorem min_distance_zero
  (U g τ : ℝ)
  (h : 2 * U ≥ g * τ)
  : ∃ t : ℝ, t = τ / 2 + U / g ∧ s t = 0 := sorry

end min_distance_zero_l161_161004


namespace minimum_cards_to_ensure_60_of_same_color_l161_161763

-- Define the conditions as Lean definitions
def total_cards : ℕ := 700
def ratio_red_orange_yellow : ℕ × ℕ × ℕ := (1, 3, 4)
def ratio_green_blue_white : ℕ × ℕ × ℕ := (3, 1, 6)
def yellow_more_than_blue : ℕ := 50

-- Define the proof goal
theorem minimum_cards_to_ensure_60_of_same_color :
  ∀ (x y : ℕ),
  (total_cards = (1 * x + 3 * x + 4 * x + 3 * y + y + 6 * y)) ∧
  (4 * x = y + yellow_more_than_blue) →
  min_cards :=
  -- Sorry here to indicate that proof is not provided
  sorry

end minimum_cards_to_ensure_60_of_same_color_l161_161763


namespace question_1_question_2_l161_161064

theorem question_1 : (567 + 345 * 566) / (567 * 345 + 222) = 1 := 
by 
  sorry

theorem question_2 : (∑ n in Finset.range 100, (n+1) * 2 * (n+1) * 3 * (n+1)) /
                        (∑ n in Finset.range 100, (n+1)*2 * 3 * (n+1)*4) =
                      1/4 :=
by 
  sorry

end question_1_question_2_l161_161064


namespace solve_for_x_l161_161193

theorem solve_for_x (x : ℝ) (y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 :=
by
  -- The proof will go here
  sorry

end solve_for_x_l161_161193


namespace ellipse_PA1_PF2_dot_product_min_value_ellipse_PA1_PF2_sum_value_min_dot_final_answer_l161_161161

noncomputable def ellipse : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

def A1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (1, 0)

theorem ellipse_PA1_PF2_dot_product_min_value :
  ∃ P ∈ ellipse, 
    let PA1 := (A1.1 - P.1, A1.2 - P.2),
        PF2 := (F2.1 - P.1, F2.2 - P.2),
        PA1_dot_PF2 := PA1.1 * PF2.1 + PA1.2 * PF2.2 in
    PA1_dot_PF2 = (P_x : ℝ) → (left_f : ℝ × ℝ) → (right_f : ℝ × ℝ) → PA1.1 * PF2.1 + PA1.2 * PF2.2
    ∧ ∀P ∈ ellipse, (left_f PA1 right_f PF2 = min PA1_dot_PF2) :
  ∃ PA1 PF2 ∈ ellipse, PA1 * PF2 >= 0
   sorry

theorem ellipse_PA1_PF2_sum_value_min_dot:
  ∀ P ∈ ellipse, 
    let PA1 := (A1.1 - P.1, A1.2 - P.2),
        PF2 := (F2.1 - P.1, F2.2 - P.2),
    ∃ left_f mid_P right_f ∈ PA1 PF2,
    sorry

theorem final_answer :
    ∀ (P ∈ ellipse), 
        let PA1 := (A1.1 - P.1, A1.2 - P.2),
            PF2 := (F2.1 - P.1, F2.2 - P.2),
    ∃ f1 f2,
      PA1_dot_PF2 * PF2 = PA1 ∧
      (P = A1) ∧ 
      0 <= PA1_dot_PF2 ≤ PA1
  
#align PA1 sum value of vectors PF2
ending

end ellipse_PA1_PF2_dot_product_min_value_ellipse_PA1_PF2_sum_value_min_dot_final_answer_l161_161161


namespace total_beetles_eaten_each_day_l161_161467

-- Definitions from the conditions
def birds_eat_per_day : ℕ := 12
def snakes_eat_per_day : ℕ := 3
def jaguars_eat_per_day : ℕ := 5
def number_of_jaguars : ℕ := 6

-- Theorem statement
theorem total_beetles_eaten_each_day :
  (number_of_jaguars * jaguars_eat_per_day) * snakes_eat_per_day * birds_eat_per_day = 1080 :=
by sorry

end total_beetles_eaten_each_day_l161_161467


namespace line_parallel_to_plane_intersection_l161_161354

theorem line_parallel_to_plane_intersection
  (l : Line) (P Q : Plane) (m : Line) 
  (h1 : l ∥ P) 
  (h2 : Q.contains l)
  (h3 : ∃ m, P ∩ Q = m) :
  l ∥ m := 
sorry

end line_parallel_to_plane_intersection_l161_161354


namespace xyz_inequality_l161_161685

theorem xyz_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ (3/4) :=
sorry

end xyz_inequality_l161_161685


namespace internal_diagonal_passes_through_cubes_l161_161412

theorem internal_diagonal_passes_through_cubes :
  let a := 180
  let b := 360
  let c := 450
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c) = 720 :=
by
  sorry

end internal_diagonal_passes_through_cubes_l161_161412


namespace coeff_x3_in_product_l161_161883

-- Definitions of the given polynomials
def P1 : Polynomial ℤ := 3 * X^4 + 2 * X^3 - 4 * X + 5
def P2 : Polynomial ℤ := 2 * X^2 - 3 * X + 4

-- The statement of the problem
theorem coeff_x3_in_product :
  (P1 * P2).coeff 3 = 8 :=
by
  sorry

end coeff_x3_in_product_l161_161883


namespace term_2005_is_1004th_l161_161759

-- Define the first term and the common difference
def a1 : Int := -1
def d : Int := 2

-- Define the general term formula of the arithmetic sequence
def a_n (n : Nat) : Int :=
  a1 + (n - 1) * d

-- State the theorem that the year 2005 is the 1004th term in the sequence
theorem term_2005_is_1004th : ∃ n : Nat, a_n n = 2005 ∧ n = 1004 := by
  sorry

end term_2005_is_1004th_l161_161759


namespace trig_lemma_l161_161400

theorem trig_lemma (x : ℝ) : 
  ((sin x)^6 + (cos x)^6 - 1)^3 + 27 * (sin x)^6 * (cos x)^6 = 0 := 
by 
  -- This is a placeholder for the actual proof.
  sorry

end trig_lemma_l161_161400


namespace sequence_form_l161_161646

/-- Define the sequence a_n such that each positive integer k appears k times -/
def a (n : ℕ) : ℕ :=
  let m := Nat.floor (0.5 + sqrt (2.0 * ↑n))
  if n - m * (m - 1) / 2 ≤ m then m else m + 1

/-- Hypotheses stating the form of a_n -/
theorem sequence_form (b c d : ℤ) :
  (∀ n : ℕ, n > 0 → (a n : ℤ) = b * ((Nat.floor (sqrt (n + c))) ^ 2) + d) → b + c + d = 1 :=
sorry

end sequence_form_l161_161646


namespace polynomial_has_real_root_l161_161892

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x^2 - 4 * x + b = 0 := 
sorry

end polynomial_has_real_root_l161_161892


namespace rotated_translated_line_eq_l161_161737

theorem rotated_translated_line_eq :
  ∀ (x y : ℝ), y = 3 * x → y = - (1 / 3) * x + (1 / 3) :=
by
  sorry

end rotated_translated_line_eq_l161_161737


namespace jump_rope_cost_l161_161877

def cost_board_game : ℕ := 12
def cost_playground_ball : ℕ := 4
def saved_money : ℕ := 6
def uncle_money : ℕ := 13
def additional_needed : ℕ := 4

theorem jump_rope_cost :
  let total_money := saved_money + uncle_money
  let total_needed := total_money + additional_needed
  let combined_cost := cost_board_game + cost_playground_ball
  let cost_jump_rope := total_needed - combined_cost
  cost_jump_rope = 7 := by
  sorry

end jump_rope_cost_l161_161877


namespace simplify_eq_l161_161874

theorem simplify_eq {x y z : ℕ} (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  9 * (x : ℝ) - ((10 / (2 * y) / 3 + 7 * z) * Real.pi) =
  9 * (x : ℝ) - (5 * Real.pi / (3 * y) + 7 * z * Real.pi) := by
  sorry

end simplify_eq_l161_161874


namespace exists_angle_greater_than_75_l161_161675

noncomputable def angle_condition (A B C : Point) (P : Point) (O1 O2 : Point) : Prop :=
  (segment_between A C P ∧ 
   segment_between B C P ∧ 
   circumcenter A B P O1 ∧
   circumcenter A C P O2 ∧
   distance B C = distance O1 O2) →
  (∃ θ : ℝ, (angle A B C θ ∧  θ > 75))

noncomputable def angle_of_triangle_greater_than_75_degrees (A B C P O1 O2: Point) : Prop :=
  angle_condition A B C P O1 O2

theorem exists_angle_greater_than_75 (A B C P O1 O2 : Point) :
  angle_of_triangle_greater_than_75_degrees A B C P O1 O2 := by
  sorry

end exists_angle_greater_than_75_l161_161675


namespace compare_exponents_l161_161455

noncomputable def exp_of_log (a : ℝ) (b : ℝ) : ℝ :=
  Real.exp ((1 / b) * Real.log a)

theorem compare_exponents :
  let a := exp_of_log 4 4
  let b := exp_of_log 5 5
  let c := exp_of_log 16 16
  let d := exp_of_log 25 25
  a = max a (max b (max c d)) ∧
  b = max (min a (max b (max c d))) (max (min b (max c d)) (max (min c d) (min d (min a b))))
  :=
  by
    sorry

end compare_exponents_l161_161455


namespace cone_radius_l161_161974

theorem cone_radius
    (l : ℝ) (n : ℝ) (r : ℝ)
    (h1 : l = 2 * Real.pi)
    (h2 : n = 120)
    (h3 : l = (n * Real.pi * r) / 180 ) :
    r = 3 :=
sorry

end cone_radius_l161_161974


namespace dependence_of_Q_l161_161256

theorem dependence_of_Q (a d k : ℕ) :
    ∃ (Q : ℕ), Q = (2 * k * (2 * a + 4 * k * d - d)) 
                - (k * (2 * a + (2 * k - 1) * d)) 
                - (k / 2 * (2 * a + (k - 1) * d)) 
                → Q = k * a + 13 * k^2 * d := 
sorry

end dependence_of_Q_l161_161256


namespace max_element_in_A_l161_161343

noncomputable def A : Set ℝ := {x : ℝ | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2020 ∧ x = Real.sqrt n n}

theorem max_element_in_A :
  ∃ x ∈ A, ∀ y ∈ A, y ≤ x ∧ x = Real.cbrt 3 :=
begin
  sorry
end

end max_element_in_A_l161_161343


namespace length_of_AP_l161_161640

/-- 
Given: 
- ABCD is a square with side length 8
- WXYZ is a rectangle with ZY = 12 and XY = 8
- AD and WX are perpendicular
- The shaded area is one-third of the area of WXYZ
Prove:
- The length of AP is 4
-/
theorem length_of_AP (ABCD_square : ∀ (A B C D : Type), side_length ABCD = 8) 
(WXYZ_rectangle : ∀ (W X Y Z : Type), ZY = 12 ∧ XY = 8)
(perpendicular_AD_WX : AD ⊥ WX)
(shaded_area_one_third : shaded_area = 1 / 3 * area_WXYZ) : 
AP = 4 :=
sorry

end length_of_AP_l161_161640


namespace domain_of_log_function_l161_161727

theorem domain_of_log_function (x : ℝ) : 1 - x > 0 ↔ x < 1 := by
  sorry

end domain_of_log_function_l161_161727


namespace inverse_of_A_is_zero_matrix_l161_161111

-- Matrix definition
def A : Matrix (Fin 2) (Fin 2) ℝ := λ i j,
  if i = 0 then
    if j = 0 then 5
    else 15
  else
    if j = 0 then 2
    else 6

-- Zero matrix definition
def zero_matrix : Matrix (Fin 2) (Fin 2) ℝ := λ _ _, 0

-- Proof statement
theorem inverse_of_A_is_zero_matrix :
  det A = 0 → inverse A = zero_matrix :=
by
  sorry

end inverse_of_A_is_zero_matrix_l161_161111


namespace semi_circle_radius_l161_161035

theorem semi_circle_radius (length width : ℝ) (π r : ℝ) (rectangle_area semi_circle_area : ℝ) :
  length = 8 ∧ width = real.pi ∧
  rectangle_area = length * width ∧
  semi_circle_area = (1 / 2) * real.pi * r^2 ∧
  semi_circle_area = rectangle_area → r = 4 := 
by
  sorry

end semi_circle_radius_l161_161035


namespace consecutive_integers_sum_of_squares_l161_161746

theorem consecutive_integers_sum_of_squares :
  ∃ a : ℕ, 0 < a ∧ ((a - 1) * a * (a + 1) = 8 * (a - 1 + a + a + 1)) → 
  ((a - 1) ^ 2 + a ^ 2 + (a + 1) ^ 2 = 77) :=
begin
  sorry
end

end consecutive_integers_sum_of_squares_l161_161746


namespace eigenvalues_of_matrix_l161_161104

theorem eigenvalues_of_matrix :
  ∃ (v : Fin 2 → ℝ) (k : ℝ), v ≠ ![0, 0] ∧ (λ (v : Fin 2 → ℝ), ![2 * v 0 + 9 * v 1, 3 * v 0 + 2 * v 1] = λ v, ![k * v 0, k * v 1]) :=
sorry

end eigenvalues_of_matrix_l161_161104


namespace average_score_is_correct_l161_161870

def scores : List ℝ := [94.5, 87.5, 99.75, 95.5, 91, 97.25]
def numChildren : ℝ := 6
def total := List.sum scores
def average := total / numChildren

theorem average_score_is_correct : average = 94.25 :=
by
  sorry

end average_score_is_correct_l161_161870


namespace charlie_delta_four_products_l161_161414

noncomputable def charlie_delta_purchase_ways : ℕ := 1363

theorem charlie_delta_four_products :
  let cakes := 6
  let cookies := 4
  let total := cakes + cookies
  ∃ ways : ℕ, ways = charlie_delta_purchase_ways :=
by
  sorry

end charlie_delta_four_products_l161_161414


namespace reflection_of_A_l161_161280

open EuclideanGeometry

noncomputable def symmetric_point_on_circumcircle (A B C A' M N : Point) (h1 : A' ∈ lineSegment B C)
  (h2 : M ∈ perpendicularBisector A' B ∩ lineSegment A B)
  (h3 : N ∈ perpendicularBisector A' C ∩ lineSegment A C) : Prop :=
  let A'_sym := reflection A' line (M, N) in Circumcircle A B C A'_sym

-- The statement to prove
theorem reflection_of_A'_on_circumcircle (A B C A' M N : Point) (h1 : A' ∈ lineSegment B C)
  (h2 : M ∈ perpendicularBisector A' B ∩ lineSegment A B)
  (h3 : N ∈ perpendicularBisector A' C ∩ lineSegment A C) :
  symmetric_point_on_circumcircle A B C A' M N h1 h2 h3 :=
sorry

end reflection_of_A_l161_161280


namespace probability_of_exactly_one_common_venue_l161_161653

noncomputable def probability_one_common_venue : ℚ :=
  let total_ways : ℕ := (Nat.choose 4 2) * (Nat.choose 4 2)
  let common_ways : ℕ := 4 * Nat.factorial 3 / Nat.factorial (3 - 2)
  (common_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_exactly_one_common_venue :
  probability_one_common_venue = 2 / 3 := by
  sorry

end probability_of_exactly_one_common_venue_l161_161653


namespace predict_sales_amount_l161_161159

theorem predict_sales_amount :
  let x_data := [2, 4, 5, 6, 8]
  let y_data := [30, 40, 50, 60, 70]
  let b := 7
  let x := 10 -- corresponding to 10,000 yuan investment
  let a := 15 -- \hat{a} calculated from the regression equation and data points
  let regression (x : ℝ) := b * x + a
  regression x = 85 :=
by
  -- Proof skipped
  sorry

end predict_sales_amount_l161_161159


namespace arithmetic_sequence_properties_l161_161137

variable {N : Type*} [linearOrderedSemiring N]

def Sn (p : ℝ) (n : ℕ) : ℝ := p * n ^ 2 + 2 * n

def an (n : ℕ) : ℝ := 2 * n + 1

def bn (n : ℕ) : ℝ := 3 ^ (n - 1)

def cn (n : ℕ) : ℕ → ℝ
| (n) => if n % 2 = 0 then 3 ^ (n / 2) else 2 * n + 1

def Tn (n : ℕ) : ℝ :=
if n % 2 = 0 
then (n * (n + 1)) / 2 + (3 * ((3 ^ n) - 1)) / 8
else ((n + 1) * (n + 2)) / 2 + (3 ^ n - 3) / 8

theorem arithmetic_sequence_properties (p : ℝ) (n : ℕ) (k : ℕ) (h : p = 1) :
  (∀ n : ℕ, an n + 1 = 2n + 1) ∧
  (∀ bn (3 ^ (n - 1) < 0)) ∧ 
  (∀ n ∈ ∅, Tn n = (if n % 2 = 0 then (n * (n + 1)) / 2 + (3 * ((3 ^ n) - 1)) / 8 else ((n + 1) * (n + 2)) / 2 + (3 ^ n - 3) / 8)) :=
by
  sorry

end arithmetic_sequence_properties_l161_161137


namespace proof_problem_l161_161461

-- Definitions
def is_factor (a b : ℕ) : Prop := ∃ k, b = a * k
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Conditions
def condition_A : Prop := is_factor 4 24
def condition_B : Prop := is_divisor 19 152 ∧ ¬ is_divisor 19 96
def condition_E : Prop := is_factor 6 180

-- Proof problem statement
theorem proof_problem : condition_A ∧ condition_B ∧ condition_E :=
by sorry

end proof_problem_l161_161461


namespace sufficient_but_not_necessary_l161_161141

theorem sufficient_but_not_necessary (a b c : ℝ) :
  (b^2 = a * c → (c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) ∨ (b = 0)) ∧ 
  ¬ ((c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) → b^2 = a * c) :=
by
  sorry

end sufficient_but_not_necessary_l161_161141


namespace solve_for_m_l161_161421

theorem solve_for_m (m : ℝ) (h : (4 * m + 6) * (2 * m - 5) = 159) : m = 5.3925 :=
sorry

end solve_for_m_l161_161421


namespace total_pizza_weight_l161_161091

theorem total_pizza_weight :
  let rachel_base := 400
  let rachel_mushrooms := 100
  let rachel_olives := 50
  let rachel_mozzarella := 60
  let bella_base := 350
  let bella_cheese := 75
  let bella_onions := 55
  let bella_peppers := 35
  let rachel_total := rachel_base + rachel_mushrooms + rachel_olives + rachel_mozzarella
  let bella_total := bella_base + bella_cheese + bella_onions + bella_peppers 
  rachel_total + bella_total = 1125 :=
by
  -- Step to enter the proof mode
  let rachel_base := 400
  let rachel_mushrooms := 100
  let rachel_olives := 50
  let rachel_mozzarella := 60
  let bella_base := 350
  let bella_cheese := 75
  let bella_onions := 55
  let bella_peppers := 35
  let rachel_total := rachel_base + rachel_mushrooms + rachel_olives + rachel_mozzarella
  let bella_total := bella_base + bella_cheese + bella_onions + bella_peppers
  have h1 : rachel_total = 610, by sorry
  have h2 : bella_total = 515, by sorry
  show 610 + 515 = 1125, by sorry

end total_pizza_weight_l161_161091


namespace range_of_a_l161_161954

theorem range_of_a (a : ℝ) :
  (∀ x, (x < -1 ∨ x > 5) ∨ (a < x ∧ x < a + 8)) ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end range_of_a_l161_161954


namespace floor_of_neg_seven_fourths_l161_161490

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l161_161490


namespace sum_inequality_equals_l161_161073

noncomputable def sum_inequality : ℚ :=
  ∑' (a : ℕ) in finset.Ico 1 ∞, ∑' (b : ℕ) in finset.Ico (a+1) ∞, ∑' (c : ℕ) in finset.Ico (b+1) ∞, (1 : ℚ) / (2^a * 4^b * 6^c)

theorem sum_inequality_equals :
  sum_inequality = 1 / 45225 := by
  sorry

end sum_inequality_equals_l161_161073


namespace ratio_EF_DE_CG_FG_l161_161758

variable (A B C D E F G : Type)
variable [EquilateralTriangle ABC]
variable [Point A1 : A ∈ Angle "equals to 15 degrees"]
variable [Point A2 : A ∈ Angle "equals to 30 degrees"]
variable [Point A3 : A ∈ Angle "equals to 45 degrees"]
variable [Line AA1 intersects DC at E]
variable [Line AA2 intersects DC at F]
variable [Line AA3 intersects DC at G]

theorem ratio_EF_DE_CG_FG :
  DE / EF = CG / (2 * FG) :=
sorry

end ratio_EF_DE_CG_FG_l161_161758


namespace floor_of_neg_seven_fourths_l161_161485

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l161_161485


namespace necessary_but_not_sufficient_l161_161124

-- Definitions used in the conditions
variable (a b : ℝ)

-- The Lean 4 theorem statement for the proof problem
theorem necessary_but_not_sufficient : (a > b - 1) ∧ ¬ (a > b) ↔ a > b := 
sorry

end necessary_but_not_sufficient_l161_161124


namespace clock_gain_correction_l161_161814

theorem clock_gain_correction :
  ∀ (gain_daily : ℝ) (days_total hours_elapsed : ℝ)
  (gain_hourly : ℝ) (total_gain_minutes : ℝ) (correction_needed : ℝ),
  gain_daily = 3.25 →
  days_total = 9 →
  hours_elapsed = 220 →
  gain_hourly = gain_daily / 24 →
  total_gain_minutes = hours_elapsed * gain_hourly →
  correction_needed = total_gain_minutes →
  correction_needed ≈ 29.8 := 
begin
  intros,
  sorry
end

end clock_gain_correction_l161_161814


namespace surface_area_geometric_mean_volume_geometric_mean_l161_161271

noncomputable def surfaces_areas_proof (r : ℝ) (π : ℝ) : Prop :=
  let F_1 := 6 * π * r^2
  let F_2 := 4 * π * r^2
  let F_3 := 9 * π * r^2
  F_1^2 = F_2 * F_3

noncomputable def volumes_proof (r : ℝ) (π : ℝ) : Prop :=
  let V_1 := 2 * π * r^3
  let V_2 := (4 / 3) * π * r^3
  let V_3 := π * r^3
  V_1^2 = V_2 * V_3

theorem surface_area_geometric_mean (r : ℝ) (π : ℝ) : surfaces_areas_proof r π := 
  sorry

theorem volume_geometric_mean (r : ℝ) (π : ℝ) : volumes_proof r π :=
  sorry

end surface_area_geometric_mean_volume_geometric_mean_l161_161271


namespace people_left_on_beach_l161_161360

theorem people_left_on_beach : 
  ∀ (initial_first_row initial_second_row initial_third_row left_first_row left_second_row left_third_row : ℕ),
  initial_first_row = 24 →
  initial_second_row = 20 →
  initial_third_row = 18 →
  left_first_row = 3 →
  left_second_row = 5 →
  initial_first_row - left_first_row + (initial_second_row - left_second_row) + initial_third_row = 54 :=
by
  intros initial_first_row initial_second_row initial_third_row left_first_row left_second_row left_third_row
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end people_left_on_beach_l161_161360


namespace engine_efficiency_l161_161854

noncomputable def efficiency_of_engine : ℝ :=
let P1 := 101325, -- Pa
    T1 := 293.15, -- K
    V1 := 1e-3,   -- m³
    R := 8.314,   -- J/(mol K)
    QH := 2000,   -- J
    γ := 7/5,     -- Ratio of specific heats for diatomic gas (nitrogen)
    n := (P1 * V1) / (R * T1), -- Number of moles using ideal gas law
    ΔU := (5/2) * n * R * 2314, -- Change in internal energy during isochoric heating
    T2 := T1 + 2314, -- Final temperature after isochoric heating
    P2 := (n * R * T2) / V1, -- Pressure after isochoric heating
    V3 := V1 * (P2 / P1)^(1/γ),
    W_adiabatic := (P2 * V1 - P1 * V3) / (γ - 1),
    W_isobaric := P1 * (V1 - V3),
    W_net := W_adiabatic + W_isobaric
in W_net / QH

theorem engine_efficiency : efficiency_of_engine = 0.33 :=
by sorry

end engine_efficiency_l161_161854


namespace parametric_to_cartesian_l161_161410

variable (θ : ℝ)
def x := 3 + 4 * Real.cos θ
def y := -2 + 4 * Real.sin θ

theorem parametric_to_cartesian :
  (x - 3)^2 + (y + 2)^2 = 16 := 
sorry

end parametric_to_cartesian_l161_161410


namespace prism_volume_l161_161829

noncomputable def volume_of_prism (a b c : ℝ) : ℝ :=
  a * b * c

theorem prism_volume (a b c : ℝ)
  (h1 : a * b = 10)
  (h2 : b * c = 15)
  (h3 : c * a = 18) :
  volume_of_prism a b c = 30 * real.sqrt 3 :=
by
  sorry

end prism_volume_l161_161829


namespace sqrt_three_irrational_l161_161853

-- Define what it means for a number to be rational
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- State that sqrt(3) is irrational
theorem sqrt_three_irrational : is_irrational (Real.sqrt 3) :=
sorry

end sqrt_three_irrational_l161_161853


namespace malachi_selfies_total_l161_161990

theorem malachi_selfies_total (x y : ℕ) 
  (h_ratio : 10 * y = 17 * x)
  (h_diff : y = x + 630) : 
  x + y = 2430 :=
sorry

end malachi_selfies_total_l161_161990


namespace min_shoeing_time_l161_161411

theorem min_shoeing_time
  (num_blacksmiths : ℕ) (num_horses : ℕ) (hooves_per_horse : ℕ) (minutes_per_hoof : ℕ)
  (h_blacksmiths : num_blacksmiths = 48)
  (h_horses : num_horses = 60)
  (h_hooves_per_horse : hooves_per_horse = 4)
  (h_minutes_per_hoof : minutes_per_hoof = 5) :
  (num_horses * hooves_per_horse * minutes_per_hoof) / num_blacksmiths = 25 := 
by
  sorry

end min_shoeing_time_l161_161411


namespace monotonic_decreasing_m_l161_161335

def power_function (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - 2*m - 2) * x^(m - 2)

theorem monotonic_decreasing_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → (power_function m x) ≤ (power_function m (x + 1))) → m = -1 :=
sorry

end monotonic_decreasing_m_l161_161335


namespace floor_neg_seven_quarter_l161_161478

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l161_161478


namespace painted_faces_l161_161891

theorem painted_faces (n_cuboids : ℕ) (faces_per_cuboid : ℕ) (h1 : n_cuboids = 5) (h2 : faces_per_cuboid = 6) : n_cuboids * faces_per_cuboid = 30 :=
by
  rw [h1, h2]
  norm_num

end painted_faces_l161_161891


namespace hash_of_hash_of_hash_of_70_l161_161879

def hash (N : ℝ) : ℝ := 0.4 * N + 2

theorem hash_of_hash_of_hash_of_70 : hash (hash (hash 70)) = 8 := by
  sorry

end hash_of_hash_of_hash_of_70_l161_161879


namespace rational_points_division_l161_161462

structure RationalPoint where
  x_num : Int
  x_den : Nat
  y_num : Int
  y_den : Nat
  x_den_ne_zero : x_den ≠ 0
  y_den_ne_zero : y_den ≠ 0
  x_coprime : Nat.gcd x_num.natAbs x_den = 1
  y_coprime : Nat.gcd y_num.natAbs y_den = 1

def is_odd (n : Int) : Prop := (n % 2 = 1)
def is_even (n : Int) : Prop := not (is_odd n)

def A (p : RationalPoint) : Prop := is_odd p.x_den ∧ is_odd p.y_den
def B (p : RationalPoint) : Prop := 
  (is_odd p.x_den ∧ is_even p.y_den) ∨ 
  (is_even p.x_den ∧ is_odd p.y_den)
def C (p : RationalPoint) : Prop := is_even p.x_den ∧ is_even p.y_den

axiom line_contains_at_most_two_sets (l : Set RationalPoint) : 
  (∃ p1 p2 p3, p1 ∈ l ∧ p2 ∈ l ∧ p3 ∈ l ∧ 
   (A p1 ∨ B p1 ∨ C p1) ∧ 
   (A p2 ∨ B p2 ∨ C p2) ∧ 
   (A p3 ∨ B p3 ∨ C p3) → false)

axiom circle_contains_all_three_sets (center : RationalPoint) (radius : Float) (c : Set RationalPoint) :
  radius > 0 → 
  (∃ p1 p2 p3, p1 ∈ c ∧ p2 ∈ c ∧ p3 ∈ c ∧ A p1 ∧ B p2 ∧ C p3) 

theorem rational_points_division (p : RationalPoint) :
  (¬ (A p ∨ B p ∨ C p) → false) :=
by
  sorry

end rational_points_division_l161_161462


namespace floor_neg_seven_over_four_l161_161491

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l161_161491


namespace perimeter_of_parallelogram_AGHI_l161_161982

-- Define the variables and conditions
variable {A B C G H I : Type}
variable [metric_space B]

def side_lengths (AB AC BC : ℝ) : Prop :=
AB = 20 ∧ AC = 20 ∧ BC = 24

-- Define points G, H, I and their conditions
def points_on_sides (G H I : Type) (AB AC BC : Type) : Prop :=
(G ∈ AB) ∧ (H ∈ BC) ∧ (I ∈ AC)

-- Define parallels
def parallels (GH HI AC AB : Type) : Prop :=
(GH ∥ AC) ∧ (HI ∥ AB)

-- Problem statement
theorem perimeter_of_parallelogram_AGHI 
(AB AC BC : ℝ) (G H I : Type) (GH HI AC AB : Type) :
side_lengths AB AC BC →
points_on_sides G H I AB AC BC →
parallels GH HI AC AB → 
perimeter (GH HI) = 40 :=
sorry

end perimeter_of_parallelogram_AGHI_l161_161982


namespace cos_alpha_is_negative_four_fifths_l161_161909

variable (α : ℝ)
variable (H1 : Real.sin α = 3 / 5)
variable (H2 : π / 2 < α ∧ α < π)

theorem cos_alpha_is_negative_four_fifths (H1 : Real.sin α = 3 / 5) (H2 : π / 2 < α ∧ α < π) :
  Real.cos α = -4 / 5 :=
sorry

end cos_alpha_is_negative_four_fifths_l161_161909


namespace perimeter_of_parallelogram_AGHI_l161_161981

-- Define the variables and conditions
variable {A B C G H I : Type}
variable [metric_space B]

def side_lengths (AB AC BC : ℝ) : Prop :=
AB = 20 ∧ AC = 20 ∧ BC = 24

-- Define points G, H, I and their conditions
def points_on_sides (G H I : Type) (AB AC BC : Type) : Prop :=
(G ∈ AB) ∧ (H ∈ BC) ∧ (I ∈ AC)

-- Define parallels
def parallels (GH HI AC AB : Type) : Prop :=
(GH ∥ AC) ∧ (HI ∥ AB)

-- Problem statement
theorem perimeter_of_parallelogram_AGHI 
(AB AC BC : ℝ) (G H I : Type) (GH HI AC AB : Type) :
side_lengths AB AC BC →
points_on_sides G H I AB AC BC →
parallels GH HI AC AB → 
perimeter (GH HI) = 40 :=
sorry

end perimeter_of_parallelogram_AGHI_l161_161981


namespace closest_point_to_origin_l161_161168

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - π / 6)

theorem closest_point_to_origin :
  (closest_point : ℝ × ℝ) = (π / 24, 0) :=
sorry

end closest_point_to_origin_l161_161168


namespace extreme_points_values_l161_161165

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + 4 * Real.log x

theorem extreme_points_values (a b : ℝ) :
  (∀ x : ℝ, f a b x = a * x ^ 2 + b * x + 4 * Real.log x) → 
  (∀ x : ℝ, Deriv (f a b) x = 2 * a * x + b + 4 / x) →
  (f a b 1 = 1 * 1 ^ 2 + b * 1 + 4 * Real.log 1) → 
  (f a b 2 = a * 2 ^ 2 + b * 2 + 4 * Real.log 2) →
  a = 1 ∧ b = -6 ∧
  f 1 (-6) 1 = -5 ∧ 
  f 1 (-6) 2 = -8 + 4 * Real.log 2 :=
by
  intros
  sorry

end extreme_points_values_l161_161165


namespace sum_evaluation_l161_161075
noncomputable theory
open_locale big_operators

-- Define the main statement
theorem sum_evaluation : (∑' (a b c : ℕ) in {p | 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 < p.3}, (1 / (2^a * 4^b * 6^c))) = 1 / 1771 := 
sorry

end sum_evaluation_l161_161075


namespace segment_equality_l161_161422

-- Define points A, B, C, and the intersections B1, A1, and M0
variables (A B C B1 A1 M3 M0 : Point)
-- Define the line segments
variables (CA CB CM3 : Line)

-- Assume given conditions as geometric facts
-- B1 is on line segment CA
def on_CA : isOn CA B1 := sorry
-- A1 is on line segment CB
def on_CB : isOn CB A1 := sorry
-- M0 is on line median CM3
def on_CM3 : isOn CM3 M0 := sorry

-- Median CM3 of triangle ABC intersects the line
def median_C : isMedian (A, B, C) CM3 := sorry

-- Prove the desired equation
theorem segment_equality :
  1/2 * (AB1 / B1C + BA1 / A1C) = M3M0 / M0C :=
sorry

end segment_equality_l161_161422


namespace sufficient_but_not_necessary_condition_l161_161747

theorem sufficient_but_not_necessary_condition :
  (∀ x ∈ set.Icc (1 : ℝ) 2, x^2 - 5 ≤ 0) ∧ ¬(∀ x ∈ set.Icc (1 : ℝ) 2, x^2 - 4 ≤ 0 → x^2 - 5 = 0) :=
by sorry

end sufficient_but_not_necessary_condition_l161_161747


namespace term_five_eq_nine_l161_161936

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- The sum of the first n terms of the sequence equals n^2.
axiom sum_formula : ∀ n, S n = n^2

-- Definition of the nth term in terms of the sequence sum.
def a_n (n : ℕ) : ℕ := S n - S (n - 1)

-- Goal: Prove that the 5th term, a(5), equals 9.
theorem term_five_eq_nine : a_n S 5 = 9 :=
by
  sorry

end term_five_eq_nine_l161_161936


namespace num_valid_plantings_l161_161818

universe u

inductive Crop
| Corn
| Wheat
| Soybeans
| Potatoes
| Rice

open Crop

-- Define the 4 sections of the 2x2 grid
structure Grid :=
  (top_left : Crop)
  (top_right : Crop)
  (bottom_left : Crop)
  (bottom_right : Crop)

-- Define adjacency rules according to the conditions given in the problem
def valid_adj (c1 c2 : Crop) : Prop :=
  match (c1, c2) with
  | (Corn, Wheat) | (Wheat, Corn) => False
  | (Soybeans, Potatoes) | (Potatoes, Soybeans) => False
  | (Rice, Soybeans) | (Soybeans, Rice) => False
  | (Rice, Potatoes) | (Potatoes, Rice) => False
  | _ => True
  end

def valid_grid (g : Grid) : Prop :=
  valid_adj g.top_left g.top_right ∧
  valid_adj g.top_left g.bottom_left ∧
  valid_adj g.top_right g.bottom_right ∧
  valid_adj g.bottom_left g.bottom_right ∧
  valid_adj g.top_left g.bottom_right ∧
  valid_adj g.top_right g.bottom_left

-- Main theorem statement
theorem num_valid_plantings : 
  ∃ (n : ℕ), n = 70 ∧ (∃ (configs : List Grid), configs.length = n ∧ ∀ g ∈ configs, valid_grid g) :=
  sorry

end num_valid_plantings_l161_161818


namespace inequality_proof_l161_161666

theorem inequality_proof (n : ℕ) (a : Finₓ n → ℝ) 
  (h1 : ∀ i, 0 ≤ a i) 
  (h2 : ∑ i, a i = 4) 
  (h3 : n ≥ 3) : 
  ∑ i, (a i)^3 * a ((i + 1) % n) ≤ 27 := 
by 
  sorry

end inequality_proof_l161_161666


namespace four_pow_sum_is_perfect_square_l161_161527

theorem four_pow_sum_is_perfect_square (x y z : ℤ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (∃ m : ℤ, 4^x + 4^y + 4^z = m^2) ↔ (z = 2 * y - x - 1) :=
sorry

end four_pow_sum_is_perfect_square_l161_161527


namespace max_value_of_function_l161_161143

theorem max_value_of_function (α : ℝ) : 
  ∃ x : ℝ, 1 - sin (x + α) ^ 2 + cos (x + α) * sin (x + α) ≤ (sqrt 2 + 1) / 2 := sorry

end max_value_of_function_l161_161143


namespace eval_floor_neg_seven_fourths_l161_161507

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l161_161507


namespace time_to_cover_escalator_l161_161443

noncomputable def average_speed (initial_speed final_speed : ℝ) : ℝ :=
  (initial_speed + final_speed) / 2

noncomputable def combined_speed (escalator_speed person_average_speed : ℝ) : ℝ :=
  escalator_speed + person_average_speed

noncomputable def coverage_time (length combined_speed : ℝ) : ℝ :=
  length / combined_speed

theorem time_to_cover_escalator
  (escalator_speed : ℝ := 20)
  (length : ℝ := 300)
  (initial_person_speed : ℝ := 3)
  (final_person_speed : ℝ := 5) :
  coverage_time length (combined_speed escalator_speed (average_speed initial_person_speed final_person_speed)) = 12.5 :=
by
  sorry

end time_to_cover_escalator_l161_161443


namespace fruit_punch_total_l161_161311

section fruit_punch
variable (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) (total_punch : ℝ)

axiom h1 : orange_punch = 4.5
axiom h2 : cherry_punch = 2 * orange_punch
axiom h3 : apple_juice = cherry_punch - 1.5
axiom h4 : total_punch = orange_punch + cherry_punch + apple_juice

theorem fruit_punch_total : total_punch = 21 := sorry

end fruit_punch

end fruit_punch_total_l161_161311


namespace cube_root_equality_l161_161189

theorem cube_root_equality (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  (∛(9 * a + 2 * b / c) = 3 * a * ∛(2 * b / c)) ↔ (c = 2 * b * (9 * a^3 - 1) / (3 * a)) :=
sorry

end cube_root_equality_l161_161189


namespace marble_prob_l161_161366

theorem marble_prob (a c x y p q : ℕ) (h1 : 2 * a + c = 36) 
    (h2 : (x / a) * (x / a) * (y / c) = 1 / 3) 
    (h3 : (a - x) / a * (a - x) / a * (c - y) / c = p / q) 
    (hpq_rel_prime : Nat.gcd p q = 1) : p + q = 65 := by
  sorry

end marble_prob_l161_161366


namespace find_CD_l161_161216

theorem find_CD (A B C D : Type) 
  [is_triangle A B C] [is_right_triangle A B C] 
  (h_angle_C : ∠C = 90)
  (h_BD_eq_BC : segment_on_extension BD AB)
  (h_BC : BC = 7)
  (h_AC : AC = 24) :
  CD = 8 * sqrt 7 :=
sorry

end find_CD_l161_161216


namespace quadratic_y_real_l161_161191

noncomputable def roots (x : ℝ) : set ℝ :=
  let D := 81 * (x^2 - 2/9 * x - 16/3)
  in
  if D ≥ 0 then
    let r1 := (-(2/9) - real.sqrt((4/81) + (64/3))) / 2
    let r2 := (-(2/9) + real.sqrt((4/81) + (64/3))) / 2
    {r : ℝ | r ≤ r1 ∨ r ≥ r2}
  else
    ∅

theorem quadratic_y_real (x : ℝ) (y : ℝ) :
  9 * y^2 + 9 * x * y + x + 8 = 0 →
  y ∈ ℝ →
  x ∈ roots x :=
  by
    sorry

end quadratic_y_real_l161_161191


namespace sum_alternating_series_l161_161065

theorem sum_alternating_series : 
  (∑ k in finset.range 1008, (2 * k + 1) + (-(2 * (k + 1)))) = -1008 := 
by
  sorry

end sum_alternating_series_l161_161065


namespace length_of_AP_l161_161641

/-- 
Given: 
- ABCD is a square with side length 8
- WXYZ is a rectangle with ZY = 12 and XY = 8
- AD and WX are perpendicular
- The shaded area is one-third of the area of WXYZ
Prove:
- The length of AP is 4
-/
theorem length_of_AP (ABCD_square : ∀ (A B C D : Type), side_length ABCD = 8) 
(WXYZ_rectangle : ∀ (W X Y Z : Type), ZY = 12 ∧ XY = 8)
(perpendicular_AD_WX : AD ⊥ WX)
(shaded_area_one_third : shaded_area = 1 / 3 * area_WXYZ) : 
AP = 4 :=
sorry

end length_of_AP_l161_161641


namespace trader_marked_price_percentage_above_cost_price_l161_161434

theorem trader_marked_price_percentage_above_cost_price 
  (CP MP SP : ℝ) 
  (discount loss : ℝ)
  (h_discount : discount = 0.07857142857142857)
  (h_loss : loss = 0.01)
  (h_SP_discount : SP = MP * (1 - discount))
  (h_SP_loss : SP = CP * (1 - loss)) :
  (MP / CP - 1) * 100 = 7.4285714285714 := 
sorry

end trader_marked_price_percentage_above_cost_price_l161_161434


namespace f_3_equals_12_l161_161457

theorem f_3_equals_12 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y) + 2 * x * y) (h2 : f(1) = 2) : f(3) = 12 :=
sorry

end f_3_equals_12_l161_161457


namespace no_p_n_eq_5_l161_161262

def largest_prime_divisor (n : ℕ) : ℕ :=
  -- implementation to find largest prime divisor, left as a stub
  sorry

noncomputable def p (n : ℕ) : ℕ :=
  if n = 1 then 2 else largest_prime_divisor (nat.factorial (n-1) + 1)

theorem no_p_n_eq_5 (n : ℕ) : p (nat.succ n) ≠ 5 :=
  sorry

end no_p_n_eq_5_l161_161262


namespace max_g_l161_161932

theorem max_g : 
  (∀ f : ℝ → ℝ, 
     (∀ x : ℝ, f x = x ^ (-2)) ∧ f 3 = 1 / 9 → 
     ∀ x ∈ set.Icc 1 3, is_max_on (λ x, (x - 1) * f x) (set.Icc 1 3) x →
     ∃ c ∈ set.Icc 1 3, (x - 1) * f x = 1 / 4  ) sorry

end max_g_l161_161932


namespace rectangle_shaded_area_l161_161995

/-- Given rectangle PQRS with dimensions PS = 2 and PQ = 4,
    and points T, U, V, W such that RT = RU = PW = PV = a.
    If VU and WT pass through the center of the rectangle,
    and the shaded region is 1/8 the area of PQRS,
    then a = 1/3. -/
theorem rectangle_shaded_area (PS PQ a : ℝ) (h1 : PS = 2) (h2 : PQ = 4)
  (h3 : VU_WT_center : ∃ O, O = (PQ / 2, PS / 2) ∧ (VU_O_center O) ∧ (WT_O_center O))
  (h4 : shaded_fraction : shaded_area PQ PS RT RU PW PV = (1 / 8) * area PQ PS) :
  a = (1 / 3) :=
sorry

end rectangle_shaded_area_l161_161995


namespace pet_store_puppies_sold_l161_161033

theorem pet_store_puppies_sold :
  ∃ P : ℕ, (2 * 6 + P * 5 = 17) ∧ (P = 1) :=
by {
  let P := 1,
  use P,
  split,
  { norm_num },
  { refl }
}

end pet_store_puppies_sold_l161_161033


namespace home_electronics_percentage_l161_161019

-- Define the conditions given in the problem
def microphotonics_pct : ℝ := 0.14
def food_additives_pct : ℝ := 0.15
def gmo_pct : ℝ := 0.19
def industrial_lubricants_pct : ℝ := 0.08
def basic_astrophysics_pct : ℝ := (72 / 360) * 100 / 100

-- Define the question: What is the percentage for home electronics?
def home_electronics_pct : ℝ := 1 - (microphotonics_pct + food_additives_pct + gmo_pct + industrial_lubricants_pct + basic_astrophysics_pct)

-- The theorem to prove
theorem home_electronics_percentage :
  home_electronics_pct = 0.24 := 
by
  -- We will prove the theorem here
  sorry

end home_electronics_percentage_l161_161019


namespace travel_time_NY_to_Miami_l161_161034

def average_speed_NY_to_Chicago : ℝ := 500
def distance_NY_to_Chicago : ℝ := 800
def headwind_NY_to_Chicago : ℝ := 50

def stopover_time_Chicago : ℝ := 1

def distance_Chicago_to_Miami : ℝ := 1200
def tailwind_Chicago_to_Miami : ℝ := 25
def average_speed_Chicago_to_Miami : ℝ := 550

def total_travel_time : ℝ :=
  let effective_speed_NY_to_Chicago := average_speed_NY_to_Chicago - headwind_NY_to_Chicago
  let time_NY_to_Chicago := distance_NY_to_Chicago / effective_speed_NY_to_Chicago
  let effective_speed_Chicago_to_Miami := average_speed_Chicago_to_Miami + tailwind_Chicago_to_Miami
  let time_Chicago_to_Miami := distance_Chicago_to_Miami / effective_speed_Chicago_to_Miami
  time_NY_to_Chicago + stopover_time_Chicago + time_Chicago_to_Miami

theorem travel_time_NY_to_Miami : total_travel_time ≈ 4.87 :=
  by
  sorry

end travel_time_NY_to_Miami_l161_161034


namespace sum_of_all_possible_values_of_intersection_points_l161_161905

theorem sum_of_all_possible_values_of_intersection_points :
  let N_vals := {0, 1, 3, 4, 6, 7, 8, 9, 10} in
  ∑ N in N_vals, N = 48 :=
by {
  sorry  -- Proof is omitted as per instructions
}

end sum_of_all_possible_values_of_intersection_points_l161_161905


namespace complex_conjugate_l161_161970

-- Definition of conditions
def has_imaginary_part_gt_zero (z : ℂ) : Prop :=
  z.im > 0

def satisfies_equation (z : ℂ) : Prop :=
  z^2 + 4 = 0

-- The main theorem statement
theorem complex_conjugate :
  ∀ (z : ℂ), satisfies_equation(z) → has_imaginary_part_gt_zero(z) → 
  conj( z / (1 + z) ) = (4/5) - (2/5) * I :=
by
  sorry

end complex_conjugate_l161_161970


namespace train_crosses_pole_in_3_seconds_l161_161842

def train_speed_kmph : ℝ := 60
def train_length_m : ℝ := 50

def speed_conversion (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

def crossing_time (distance_m : ℝ) (speed_mps : ℝ) : ℝ := distance_m / speed_mps

theorem train_crosses_pole_in_3_seconds :
  crossing_time train_length_m (speed_conversion train_speed_kmph) = 3 :=
by
  sorry

end train_crosses_pole_in_3_seconds_l161_161842


namespace simplify_fraction_l161_161781

theorem simplify_fraction : (3^3 * 3^(-4) / (3^2 * 3^(-5)) = 1 / 3^8) := by
  sorry

end simplify_fraction_l161_161781


namespace sum_evaluation_l161_161074
noncomputable theory
open_locale big_operators

-- Define the main statement
theorem sum_evaluation : (∑' (a b c : ℕ) in {p | 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 < p.3}, (1 / (2^a * 4^b * 6^c))) = 1 / 1771 := 
sorry

end sum_evaluation_l161_161074


namespace xiaolin_distance_l161_161392

theorem xiaolin_distance (speed : ℕ) (time : ℕ) (distance : ℕ)
    (h1 : speed = 80) (h2 : time = 28) : distance = 2240 :=
by
  have h3 : distance = time * speed := by sorry
  rw [h1, h2] at h3
  exact h3

end xiaolin_distance_l161_161392


namespace closest_integer_to_sum_is_102_l161_161108

noncomputable def sum_term (n : ℕ) : ℝ := 1 / (n ^ 2 - 9)

noncomputable def compounded_sum (a b : ℕ) : ℝ := ∑ n in Finset.range (b - a + 1) \u4 { a + i | i ∈ Finset.range (b - a + 1) }, sum_term (a + n)

noncomputable def scaled_sum (a b : ℕ) : ℝ := 500 * compounded_sum a b

theorem closest_integer_to_sum_is_102 :
  Int.floor (scaled_sum 4 15000 + 0.5) = 102 :=
begin
  sorry
end

end closest_integer_to_sum_is_102_l161_161108


namespace parallelogram_AGHI_perimeter_l161_161979

theorem parallelogram_AGHI_perimeter 
  (A B C G H I : Type*)
  [H2 : Equiv A B C 20] [H3 : Equiv A C B 20] [H1 : Equiv B C 24]
  (G_on_AB : G ∈ segment B 20) (H_on_BC : H ∈ segment C 24) (I_on_AC : I ∈ segment A 20)
  (GH_parallel_AC : line GH ∥ line AC) (HI_parallel_AB : line HI ∥ line AB)
  (triangles_similarity : (Triangle A B G H) ≃ (Triangle H I C))
: perimeter (Parallelogram A G H I) = 40 := 
sorry

end parallelogram_AGHI_perimeter_l161_161979


namespace angle_between_AB_and_AC_is_pi_div_3_l161_161177

noncomputable def vector (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

noncomputable def vec_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2).sqrt

noncomputable def cos_angle_between (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

noncomputable def angle_between (u v : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (cos_angle_between u v)

def A : ℝ × ℝ × ℝ := vector 0 2 3
def B : ℝ × ℝ × ℝ := vector (-2) 1 6
def C : ℝ × ℝ × ℝ := vector 1 (-1) 5

def AB : ℝ × ℝ × ℝ := vec_sub B A
def AC : ℝ × ℝ × ℝ := vec_sub C A

theorem angle_between_AB_and_AC_is_pi_div_3 : angle_between AB AC = real.pi / 3 := by
  sorry

end angle_between_AB_and_AC_is_pi_div_3_l161_161177


namespace factorization_l161_161524

theorem factorization (x y : ℝ) : 
  (x + y) ^ 2 + 4 * (x - y) ^ 2 - 4 * (x ^ 2 - y ^ 2) = (x - 3 * y) ^ 2 :=
by
  sorry

end factorization_l161_161524


namespace amount_paid_is_51_l161_161611

def original_price : ℕ := 204
def discount_fraction : ℚ := 0.75
def paid_fraction : ℚ := 1 - discount_fraction

theorem amount_paid_is_51 : paid_fraction * original_price = 51 := by
  sorry

end amount_paid_is_51_l161_161611


namespace remainder_of_M_mod_55_l161_161081

def M : ℕ := -- define M as the concatenation of integers from 1 to 55
  let concatenated_str := (List.range' 1 55).foldl (λ acc n, acc ++ toString n) ""
  concatenated_str.toNat

theorem remainder_of_M_mod_55 : M % 55 = 45 := 
  sorry

end remainder_of_M_mod_55_l161_161081


namespace exists_valid_star_arrangement_no_valid_arrangement_less_than_7_l161_161408

def star_arrangement_valid (arr : matrix (fin 4) (fin 4) bool) : Prop :=
  ∀ (r1 r2 : fin 4) (c1 c2 : fin 4),
    r1 ≠ r2 → c1 ≠ c2 → !(arr r1 c1 ∧ arr r1 c2 ∧ arr r2 c1 ∧ arr r2 c2)

theorem exists_valid_star_arrangement :
  ∃ arr : matrix (fin 4) (fin 4) bool, (∑ i j, if arr i j then 1 else 0) = 7 ∧ star_arrangement_valid arr :=
sorry

theorem no_valid_arrangement_less_than_7 :
  ∀ (arr : matrix (fin 4) (fin 4) bool), (∑ i j, if arr i j then 1 else 0) < 7 → ¬ star_arrangement_valid arr :=
sorry

end exists_valid_star_arrangement_no_valid_arrangement_less_than_7_l161_161408


namespace noah_small_paintings_sold_last_month_l161_161698

theorem noah_small_paintings_sold_last_month
  (large_painting_price small_painting_price : ℕ)
  (large_paintings_sold_last_month : ℕ)
  (total_sales_this_month : ℕ)
  (sale_multiplier : ℕ)
  (x : ℕ)
  (h1 : large_painting_price = 60)
  (h2 : small_painting_price = 30)
  (h3 : large_paintings_sold_last_month = 8)
  (h4 : total_sales_this_month = 1200)
  (h5 : sale_multiplier = 2) :
  (2 * ((large_paintings_sold_last_month * large_painting_price) + (x * small_painting_price)) = total_sales_this_month) → x = 4 :=
by
  sorry

end noah_small_paintings_sold_last_month_l161_161698


namespace geometry_problem_l161_161245

-- Define the problem setting
structure Point :=
(x : ℝ) (y : ℝ)

def right_triangle (A B C : Point) : Prop :=
C.x = 0 ∧ C.y = 0 ∧ (A.y = 0 ∨ B.y = 0) ∧ A.x ≠ 0 ∧ B.x ≠ 0 ∧ A ≠ B

def centroid (A B C G : Point) : Prop :=
G.x = (A.x + B.x + C.x) / 3 ∧ G.y = (A.y + B.y + C.y) / 3

def circumcircle (A B C : Point) : Prop :=
sorry -- Definition of circumcircle (ignored for brevity)

def perpendicular (P L : Point → Prop) : Prop :=
sorry -- Definition of perpendicularity (ignored for brevity)

-- Translate conditions
theorem geometry_problem {A B C P Q X Y G : Point}
  (hABC : right_triangle A B C)
  (hG : centroid A B C G)
  (hk1 : circumcircle A G C)
  (hk2 : circumcircle B G C)
  (hPQ_on_AB : ∃ P Q, P ∈ (line A B) ∧ Q ∈ (line A B))
  (hPX_perp_to_AC : perpendicular P (line A C))
  (hQY_perp_to_BC : perpendicular Q (line B C))
  (hX_on_k1 : X ∈ circumcircle A G C)
  (hY_on_k2: Y ∈ circumcircle B G C) :
  (C.dist X * C.dist Y) / (A.dist B) ^ 2 = 4 / 9 :=
sorry

end geometry_problem_l161_161245


namespace count_multiples_between_l161_161285

theorem count_multiples_between (low high n : ℕ) (h_lcm : Nat.lcm 12 18 = n) :
  (100 ≤ low ∧ low ≤ high ∧ high ≤ 500) →
  low = 108 ∧ high = 468 →
  ∃ k : ℕ, k = (high - low) / n + 1 ∧ k = 11 :=
begin
  intros h_range h_bounds,
  sorry
end


end count_multiples_between_l161_161285


namespace projection_of_a_on_b_is_neg4_l161_161967

def a : ℝ × ℝ := (-8, 1)
def b : ℝ × ℝ := (3, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def scalar_projection (v1 v2 : ℝ × ℝ) : ℝ :=
  (dot_product v1 v2) / (magnitude v2)

theorem projection_of_a_on_b_is_neg4 :
  scalar_projection a b = -4 := 
sorry

end projection_of_a_on_b_is_neg4_l161_161967


namespace count_functions_l161_161258

def A : Set ℕ := { n : ℕ | 1 ≤ n ∧ n ≤ 2011 }

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  ∀ n ∈ A, f n ≤ n ∧ ∃ s : Set ℕ, (s.card = 2010 ∧ ∀ k ∈ A, f k ∈ s)

theorem count_functions :
  (∃ S : Finset (ℕ → ℕ), (∀ f ∈ S, satisfies_conditions f) ∧ S.card = 2^2011 - 2012) :=
sorry

end count_functions_l161_161258


namespace minimum_k_condition_l161_161135

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then 2/3 else 1/(3^n)

def sum_S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, sequence_a (i + 1))

theorem minimum_k_condition (k : ℚ) (h : k < 5/6) : ∀ n, sum_S n < k :=
  sorry

end minimum_k_condition_l161_161135


namespace base_rep_of_625_l161_161543

theorem base_rep_of_625 (b : ℕ) : 
  (b^3 ≤ 625 ∧ 625 < b^4 ∧ 
  let repr := (625 / b^3, (625 % b^3) / b^2, ((625 % b^3) % b^2) / b, (625 % b)) in
  let final_digits := ((625 % b^3) % b^2) / b + (625 % b) in
  (repr.2%2 = 1 ∧ final_digits%2 = 1)) → 
  b = 6 :=
by
  sorry

end base_rep_of_625_l161_161543


namespace angle_with_same_terminal_side_l161_161531

noncomputable def co_terminal_angles : ℤ → ℝ :=
  λ k, -7 * Real.pi / 8 + 2 * k * Real.pi

theorem angle_with_same_terminal_side (k : ℤ) : ∃ θ, θ = co_terminal_angles k :=
  exists.intro (co_terminal_angles k) rfl

end angle_with_same_terminal_side_l161_161531


namespace distinct_prime_factors_l161_161082

theorem distinct_prime_factors (M : ℝ) (log11M : ℝ) (log7log11M : ℝ) (log5log7log11M : ℝ) (log3log5log7log11M : ℝ) :
  log3log5log7log11M = 5 →
  log5log7log11M = 3^5 →
  log7log11M = 5^(3^5) →
  log11M = 7^(5^(3^5)) →
  2 * M = 11^(7^(5^(3^5))) →
  2 =
  (if is_prime 2 then 1 else 0) +
  (if is_prime 11 then 1 else 0) :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end distinct_prime_factors_l161_161082


namespace tan_four_pi_over_three_l161_161903

theorem tan_four_pi_over_three : tan (4 * Real.pi / 3) = Real.sqrt 3 :=
by
  -- Import necessary conditions
  sorry

end tan_four_pi_over_three_l161_161903


namespace min_chocolates_exists_l161_161353

theorem min_chocolates_exists :
  ∃ C : ℕ, C % 6 = 4 ∧ C % 8 = 6 ∧ C % 10 = 8 ∧ ∀ D : ℕ, (D % 6 = 4 ∧ D % 8 = 6 ∧ D % 10 = 8) → C ≤ D :=
by
  use 118
  split
  . exact nat.mod_eq_of_lt rfl
  . exact nat.mod_eq_of_lt rfl
  . exact nat.mod_eq_of_lt rfl
  . intro D hD
    sorry

end min_chocolates_exists_l161_161353


namespace smallest_x_for_multiple_l161_161384

def factors_500 : ℕ := 2^1 * 5^3
def factors_864 : ℕ := 2^5 * 3^3

theorem smallest_x_for_multiple:
  ∃ x: ℕ, x > 0 ∧ 500 * x % 864 = 0 ∧ ∀ y: ℕ, y > 0 ∧ 500 * y % 864 = 0 → x ≤ y :=
begin
  have factor_500: 500 = factors_500,
  { sorry }, -- placeholder for the proof, not required here
  have factor_864: 864 = factors_864,
  { sorry }, -- placeholder for the proof, not required here
  use 432,
  split,
  { -- prove 432 > 0
    sorry
  },
  split,
  { -- prove 500 * 432 is a multiple of 864
    sorry
  },
  { -- prove 432 is the smallest positive integer with this property
    intros y hy1 hy2,
    exact nat.le_of_dvd hy1 (dvd_trans (nat.dvd_of_mod_eq_zero hy2) (show 864 ∣ 500 * 432, by sorry)) -- this tactic ensures the minimality property
  }
end

end smallest_x_for_multiple_l161_161384


namespace function_satisfies_conditions_l161_161106

theorem function_satisfies_conditions :
  (∃ f : ℤ × ℤ → ℝ,
    (∀ x y z : ℤ, f (x, y) * f (y, z) * f (z, x) = 1) ∧
    (∀ x : ℤ, f (x + 1, x) = 2) ∧
    (∀ x y : ℤ, f (x, y) = 2 ^ (x - y))) :=
by
  sorry

end function_satisfies_conditions_l161_161106


namespace exists_point_on_transformed_graph_l161_161575

theorem exists_point_on_transformed_graph (f : ℝ → ℝ) :
  f 12 = 10 → ∃ (y : ℝ), 3 * y = f (3 * 4) / 3 + 3 ∧ 4 + y = 55 / 9 := 
by
  intro h
  use 19 / 9
  split
  { calc
      3 * (19 / 9)
          = 19 / 3 : by norm_num
      ... = f 12 / 3 + 3 : by { rw h, norm_num } }
  { calc 
      4 + 19 / 9 
          = 55 / 9 : by norm_num }

end exists_point_on_transformed_graph_l161_161575


namespace sum_inequality_equals_l161_161072

noncomputable def sum_inequality : ℚ :=
  ∑' (a : ℕ) in finset.Ico 1 ∞, ∑' (b : ℕ) in finset.Ico (a+1) ∞, ∑' (c : ℕ) in finset.Ico (b+1) ∞, (1 : ℚ) / (2^a * 4^b * 6^c)

theorem sum_inequality_equals :
  sum_inequality = 1 / 45225 := by
  sorry

end sum_inequality_equals_l161_161072


namespace range_of_m_l161_161670

variable (x m : ℝ)

def p : Prop := ∃ x_0 : ℝ, x_0^2 + m ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h_pq : ¬ p ∨ q = False) : m ≤ -2 := by
  sorry

end range_of_m_l161_161670


namespace original_weight_l161_161043

namespace MarbleProblem

def remainingWeightAfterCuts (w : ℝ) : ℝ :=
  w * 0.70 * 0.70 * 0.85

theorem original_weight (w : ℝ) : remainingWeightAfterCuts w = 124.95 → w = 299.94 :=
by
  intros h
  sorry

end MarbleProblem

end original_weight_l161_161043


namespace total_packing_peanuts_used_l161_161775

def large_order_weight : ℕ := 200
def small_order_weight : ℕ := 50
def large_orders_sent : ℕ := 3
def small_orders_sent : ℕ := 4

theorem total_packing_peanuts_used :
  (large_orders_sent * large_order_weight) + (small_orders_sent * small_order_weight) = 800 := 
by
  sorry

end total_packing_peanuts_used_l161_161775


namespace john_avg_speed_last_30_minutes_l161_161658

open Real

/-- John drove 160 miles in 120 minutes. His average speed during the first
30 minutes was 55 mph, during the second 30 minutes was 75 mph, and during
the third 30 minutes was 60 mph. Prove that his average speed during the
last 30 minutes was 130 mph. -/
theorem john_avg_speed_last_30_minutes (total_distance : ℝ) (total_time_minutes : ℝ)
  (speed_1 : ℝ) (speed_2 : ℝ) (speed_3 : ℝ) (speed_4 : ℝ) :
  total_distance = 160 →
  total_time_minutes = 120 →
  speed_1 = 55 →
  speed_2 = 75 →
  speed_3 = 60 →
  (speed_1 + speed_2 + speed_3 + speed_4) / 4 = total_distance / (total_time_minutes / 60) →
  speed_4 = 130 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end john_avg_speed_last_30_minutes_l161_161658


namespace fixed_point_line_l161_161160

noncomputable def ellipse_equation (a : ℝ) : String :=
  "The equation of the ellipse E is: " ++
  if a^2 = 5/8 then "8x^2 / 5 + 8y^2 / 3 = 1" else "undefined"

theorem fixed_point_line (a : ℝ) (x0 y0 : ℝ)
  (H1 : a^2 = 5 / 8)
  (H2 : (x0, y0) ∈ ({p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (1 - a^2) = 1} : set (ℝ × ℝ)))
  (H3 : x0, y0 > 0)
  (H4 : let c := sqrt (2 * a^2 - 1) in
        let F1 := (-c, 0) in
        let F2 := (c, 0) in
        let P := (x0, y0) in
        let Q := (0, c * y0 / (c - x0)) in
        let kF1P := y0 / (x0 + c) in
        let kF2P := y0 / (x0 - c) in
        let kF1Q := c * y0 / ((c - x0) * (x0 + c)) in
        kF1Q * kF1P = -1)
  : x0 + y0 = 1 :=
sorry

end fixed_point_line_l161_161160


namespace collinear_probability_l161_161647

theorem collinear_probability :
  let N := 16
  let M := 4
  let total_dots := finset.Icc 1 N
  let horizontal_lines := 4
  let vertical_lines := 4
  let diagonal_lines := 2
  let total_collinear_sets := horizontal_lines + vertical_lines + diagonal_lines
  let total_combinations := nat.choose N M
  let collinear_probability := total_collinear_sets / total_combinations
  in collinear_probability = 1 / 182 := by
  sorry

end collinear_probability_l161_161647


namespace sum_base9_to_base9_eq_l161_161846

-- Definition of base 9 numbers
def base9_to_base10 (n : ℕ) : ℕ :=
  let digit1 := n % 10
  let digit2 := (n / 10) % 10
  let digit3 := (n / 100) % 10
  digit1 + 9 * digit2 + 81 * digit3

-- Definition of base 10 to base 9 conversion
def base10_to_base9 (n : ℕ) : ℕ :=
  let digit1 := n % 9
  let digit2 := (n / 9) % 9
  let digit3 := (n / 81) % 9
  digit1 + 10 * digit2 + 100 * digit3

-- The theorem to prove
theorem sum_base9_to_base9_eq :
  let x := base9_to_base10 236
  let y := base9_to_base10 327
  let z := base9_to_base10 284
  base10_to_base9 (x + y + z) = 858 :=
by {
  sorry
}

end sum_base9_to_base9_eq_l161_161846


namespace analytical_expression_range_of_t_l161_161132

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem analytical_expression (x : ℝ) :
  (f (x + 1) - f x = 2 * x - 2) ∧ (f 1 = -2) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x > 0 ∧ f (x + t) < 0 → x = 1) ↔ (-2 <= t ∧ t < -1) :=
by
  sorry

end analytical_expression_range_of_t_l161_161132


namespace a_minus_b_equals_40_l161_161971

def sum_of_arithmetic_series (start : ℕ) (end : ℕ) (diff : ℕ) : ℕ :=
  let n := (end - start) / diff + 1
  in (n / 2) * (start + end)

def a : ℕ := sum_of_arithmetic_series 2 80 2
def b : ℕ := sum_of_arithmetic_series 1 79 2

theorem a_minus_b_equals_40 : a - b = 40 := by
  sorry

end a_minus_b_equals_40_l161_161971


namespace lighthouses_visible_from_anywhere_l161_161545

-- A theorem that proves four arbitrary placed lighthouses with each lamp illuminating 90 degrees of angle
-- can be rotated such that at least one lamp is visible from every point in the plane.
theorem lighthouses_visible_from_anywhere (lighthouse : Fin 4 → Point) (angle : Fin 4 → ℝ) : 
  (∀ i : Fin 4, angle i = 90) →
  ∃ (orientation : Fin 4 → ℝ), 
    ∀ (p : Point), (∃ i : Fin 4, lamp_visible_from_point (orientation i) (angle i) (lighthouse i) p) :=
by
  sorry

end lighthouses_visible_from_anywhere_l161_161545


namespace closest_integer_to_series_sum_l161_161110

theorem closest_integer_to_series_sum :
  round (500 * (∑ n in Finset.range 14997 \ Finset.range 3, 1 / (n + 4)^2 - 9)) = 153 :=
by
  sorry

end closest_integer_to_series_sum_l161_161110


namespace find_coefficients_l161_161742

-- Define the polynomial
def poly (a b : ℤ) (x : ℚ) : ℚ := a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8

-- Define the factor
def factor (x : ℚ) : ℚ := 3 * x^2 - 2 * x + 2

-- States that for a given polynomial and factor, the resulting (a, b) pair is (-51, 25)
theorem find_coefficients :
  ∃ a b c d : ℤ, 
  (∀ x, poly a b x = (factor x) * (c * x^2 + d * x + 4)) ∧ 
  a = -51 ∧ 
  b = 25 :=
by sorry

end find_coefficients_l161_161742


namespace right_angled_triangles_l161_161460

theorem right_angled_triangles (x y z : ℕ) : (x - 6) * (y - 6) = 18 ∧ (x^2 + y^2 = z^2)
  → (3 * (x + y + z) = x * y) :=
sorry

end right_angled_triangles_l161_161460


namespace modular_expression_divisible_by_twelve_l161_161765

theorem modular_expression_divisible_by_twelve
  (a b c d : ℕ)
  (ha : a < 12) (hb : b < 12) (hc : c < 12) (hd : d < 12)
  (h_abcd_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_a_invertible : Nat.gcd a 12 = 1) (h_b_invertible : Nat.gcd b 12 = 1)
  (h_c_invertible : Nat.gcd c 12 = 1) (h_d_invertible : Nat.gcd d 12 = 1)
  : (a * b * c + a * b * d + a * c * d + b * c * d) * Nat.mod_inv (a * b * c * d) 12 % 12 = 0 :=
by
  sorry

end modular_expression_divisible_by_twelve_l161_161765


namespace john_bar_weight_l161_161237

noncomputable def john_weight_bench_support : ℕ := 1000
noncomputable def safety_margin : ℝ := 0.20
noncomputable def john_weight : ℕ := 250

theorem john_bar_weight : 
  let support_weight := john_weight_bench_support
  let safety_weight_margin := support_weight * safety_margin
  let safe_total_weight := support_weight - safety_weight_margin
  let actual_weight_on_bar := safe_total_weight - john_weight
  in actual_weight_on_bar = 550 := by
  -- Computations here to verify the proof
  sorry

end john_bar_weight_l161_161237


namespace circle_center_and_radius_l161_161719

theorem circle_center_and_radius :
  ∀ (x y : ℝ), (x - 2) ^ 2 + (y + 3) ^ 2 = 2 → 
  (∀ h k r : ℝ, h = 2 ∧ k = -3 ∧ r = sqrt 2) :=
by
  intros x y h k r
  sorry

end circle_center_and_radius_l161_161719


namespace sum_of_coefficients_l161_161886

theorem sum_of_coefficients : 
  (let p := (x^2 - 3 * x * y + y^2)^6 in 
  p.subst (λ x, 1).subst (λ y, 1)).coeffs.sum = 64 := sorry

end sum_of_coefficients_l161_161886


namespace two_painters_days_l161_161283

-- Define the conditions and the proof problem
def five_painters_days : ℕ := 5
def days_per_five_painters : ℕ := 2
def total_painter_days : ℕ := five_painters_days * days_per_five_painters -- Total painter-days for the original scenario
def two_painters : ℕ := 2
def last_day_painter_half_day : ℕ := 1 -- Indicating that one painter works half a day on the last day
def last_day_work : ℕ := two_painters - last_day_painter_half_day / 2 -- Total work on the last day is equivalent to 1.5 painter-days

theorem two_painters_days : total_painter_days = 5 :=
by
  sorry -- Mathematical proof goes here

end two_painters_days_l161_161283


namespace sum_of_interchangeable_primes_l161_161385

def is_prime (n : ℕ) : Prop := sorry -- Assume the definition or use an existing one

def digits_interchanged (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) * 10 + (n / 10)

theorem sum_of_interchangeable_primes : 
  let primes := { n : ℕ | 20 < n ∧ n < 80 ∧ is_prime n ∧ is_prime (digits_interchanged n) } 
  in (∑ n in primes, n) = 291 := 
sorry

end sum_of_interchangeable_primes_l161_161385


namespace relationship_among_sets_l161_161590

-- Definitions of the integer sets E, F, and G
def E := {e : ℝ | ∃ m : ℤ, e = m + 1 / 6}
def F := {f : ℝ | ∃ n : ℤ, f = n / 2 - 1 / 3}
def G := {g : ℝ | ∃ p : ℤ, g = p / 2 + 1 / 6}

-- The theorem statement capturing the relationship among E, F, and G
theorem relationship_among_sets : E ⊆ F ∧ F = G := by
  sorry

end relationship_among_sets_l161_161590


namespace area_of_triangle_PTU_l161_161428

-- Definitions for the regular octagon and its properties
def regular_octagon_side_length := 3
def diagonal_length := 3 * (Real.sqrt (2 + Real.sqrt 2))
def included_angle_radian := (3 * Real.pi) / 4 -- 135 degrees in radians

-- Calculate the area of triangle PTU
theorem area_of_triangle_PTU : 
  let s := regular_octagon_side_length in
  let d := diagonal_length in
  let θ := included_angle_radian in
  (1/2) * d * d * Real.sin θ = (9 * Real.sqrt 2 + 9) / 2 :=
by
  sorry

end area_of_triangle_PTU_l161_161428


namespace binomial_expansion_third_term_coefficient_l161_161997

theorem binomial_expansion_third_term_coefficient :
  let C := Nat.choose in
  let T (r : ℕ) := C 7 r * (2 * x) ^ r in
  T 2 = 24 * x^2 :=
by
  sorry

end binomial_expansion_third_term_coefficient_l161_161997


namespace geometric_sequence_8th_term_l161_161459

theorem geometric_sequence_8th_term (a : ℚ) (r : ℚ) (n : ℕ) (h_a : a = 27) (h_r : r = 2/3) (h_n : n = 8) :
  a * r^(n-1) = 128 / 81 :=
by
  rw [h_a, h_r, h_n]
  sorry

end geometric_sequence_8th_term_l161_161459


namespace solve_for_x_l161_161195

theorem solve_for_x (x : ℝ) (y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 :=
by
  -- The proof will go here
  sorry

end solve_for_x_l161_161195


namespace angles_of_triangle_ABC_correct_l161_161612

def α := 30
def β := 120
def γ := 30

def is_triangle_sum (α β γ : Nat) : Prop := α + β + γ = 180

def angle_bisector_theorem (AB AC AK KC : ℕ) : Prop := 
  (AK / (AK + KC)) = (AB / AC)

def are_angles_of_triangle_ABC_correct (α β γ : ℕ) (AK KC BK : ℕ) (is_bisector : Prop) : Prop :=
  is_bisector ∧ AK / KC = 1 / 2 ∧ is_triangle_sum α β γ ∧ (sin α = 1 / 2) ∧ (sin β = sqrt 3 / 2) ∧ (sin γ = 1 / 2)

theorem angles_of_triangle_ABC_correct :
  are_angles_of_triangle_ABC_correct 30 120 30 1 2 2 (angle_bisector_theorem 1 1 1 2) :=
by sorry

end angles_of_triangle_ABC_correct_l161_161612


namespace land_profit_each_son_l161_161029

theorem land_profit_each_son :
  let hectares : ℝ := 3
  let m2_per_hectare : ℝ := 10000
  let total_sons : ℕ := 8
  let area_per_son := (hectares * m2_per_hectare) / total_sons
  let m2_per_portion : ℝ := 750
  let profit_per_portion : ℝ := 500
  let periods_per_year : ℕ := 12 / 3

  (area_per_son / m2_per_portion * profit_per_portion * periods_per_year = 10000) :=
by
  sorry

end land_profit_each_son_l161_161029


namespace sum_of_three_digit_numbers_divisible_by_7_5_3_l161_161350

theorem sum_of_three_digit_numbers_divisible_by_7_5_3 {N : ℕ} (h1 : N % 7 = 5) (h2 : N % 5 = 2) (h3 : N % 3 = 1) (h4 : 100 ≤ N ∧ N ≤ 999) :
    ∑ (N : ℕ) in finset.filter (λ N, N % 7 = 5 ∧ N % 5 = 2 ∧ N % 3 = 1) (finset.Icc 100 999), N = 4436 :=
begin
  sorry
end

end sum_of_three_digit_numbers_divisible_by_7_5_3_l161_161350


namespace problem_proof_l161_161123

noncomputable def hyperbola_foci (a : ℝ) (point : ℝ × ℝ) (Γ : ℝ → ℝ → Prop) : Prop :=
  let ⟨x, y⟩ := point in
  (Γ x y ∧ x = 2 ∧ y = 1 ∧ (-real.sqrt 3, 0) ∈ Γ ∧ (real.sqrt 3 , 0) ∈ Γ)

noncomputable def value_of_k (k : ℝ) (x_midpoint : ℝ) : Prop :=
  let y := k * 1 + 1 in
  (x_midpoint = 1) ∧ (k = (-1 + real.sqrt 5) / 2)

theorem problem_proof :
  ∀ a : ℝ,
  ((∀ x y : ℝ, x^2 / a^2 - y^2 = 1) → ∃ foci_1 foci_2 : ℝ × ℝ, 
  hyperbola_foci a (2, 1) (λ x y, x^2 / a^2 - y^2 = 1) ∧
  (foci_1 = (-real.sqrt 3, 0)) ∧ (foci_2 = (real.sqrt 3, 0))) ∧
  (∀ k : ℝ, (x_midpoint = 1) → value_of_k k x_midpoint) :=
sorry

end problem_proof_l161_161123


namespace aarti_three_times_work_l161_161045

variable (D : ℕ) (multi : ℕ → ℕ)

-- Define the initial conditions
def work_done_by_Aarti_in_9_days := (D = 9)

-- Define the relationship for multiple amounts of work
def three_times_work := (multi 3 = 3 * D)

-- The statement to prove
theorem aarti_three_times_work (hD : work_done_by_Aarti_in_9_days) (hMulti : three_times_work):
  multi 3 = 27 := by
  sorry

end aarti_three_times_work_l161_161045


namespace angle_between_A1B_and_plane_ABD_distance_from_A1_to_plane_AED_l161_161859

-- Definitions of points and conditions
variable (A B C A₁ B₁ C₁ D E : Point)
variable (triangle_base : IsoscelesRightTriangle A C B)
variable (prism_height : A₁ = A ↑(2)) -- A₁ is 2 units above A
variable (midpoint_D : Midpoint D C₁ C)
variable (midpoint_E : Midpoint E A₁ B)
variable (projection_E : IsCentroid G A B D)

-- Problem (1)
theorem angle_between_A1B_and_plane_ABD :
  ∠BetweenLinePlane (Line A₁ B) (Plane A B D) = arcsin (√2 / 3) := by sorry

-- Problem (2)
theorem distance_from_A1_to_plane_AED :
  DistanceFromPointToPlane A₁ (Plane A E D) = (2 * √6) / 3 := by sorry

end angle_between_A1B_and_plane_ABD_distance_from_A1_to_plane_AED_l161_161859


namespace vector_addition_correct_dot_product_correct_l161_161592

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem vector_addition_correct :
  let a := (1, 2)
  let b := (3, 1)
  vector_add a b = (4, 3) := by
  sorry

theorem dot_product_correct :
  let a := (1, 2)
  let b := (3, 1)
  dot_product a b = 5 := by
  sorry

end vector_addition_correct_dot_product_correct_l161_161592


namespace shaded_region_area_l161_161417

open Real

noncomputable def area_of_shaded_region (r : ℝ) (s : ℝ) (d : ℝ) : ℝ := 
  (1/4) * π * r^2 + (1/2) * (d - s)^2

theorem shaded_region_area :
  let r := 3
  let s := 2
  let d := sqrt 5
  area_of_shaded_region r s d = 9 * π / 4 + (9 - 4 * sqrt 5) / 2 :=
by
  sorry

end shaded_region_area_l161_161417


namespace product_of_consecutive_even_numbers_divisible_by_8_l161_161336

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 8 ∣ (2 * n * (2 * n + 2)) :=
by sorry

end product_of_consecutive_even_numbers_divisible_by_8_l161_161336


namespace find_lambda_l161_161594

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : vector := (2, -7)
def b : vector := (-2, -4)

theorem find_lambda (λ : ℝ) : dot_product (a.1 + λ * b.1, a.2 + λ * b.2) b = 0 → λ = 6 / 5 :=
by
  sorry

end find_lambda_l161_161594


namespace simplify_expression_l161_161390

theorem simplify_expression :
  (-(2/3) + -(7/6) - -(3/4) - +(1/4)) = (-(2/3) - (7/6) + (3/4) - (1/4)) :=
by
  sorry

end simplify_expression_l161_161390


namespace remainder_modulo_12_l161_161766

theorem remainder_modulo_12 
  (a b c d : ℕ) 
  (ha : a < 12) (hb : b < 12) (hc : c < 12) (hd : d < 12)
  (ha_ne : a ≠ b) (hb_ne : b ≠ c) (hc_ne : c ≠ d) (hd_ne : a ≠ d)
  (ha_gcd : Nat.gcd a 12 = 1) (hb_gcd : Nat.gcd b 12 = 1)
  (hc_gcd : Nat.gcd c 12 = 1) (hd_gcd : Nat.gcd d 12 = 1) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (a * b * c * d)⁻¹) % 12 = 0 :=
by
  sorry

end remainder_modulo_12_l161_161766


namespace ratio_of_side_lengths_l161_161856

theorem ratio_of_side_lengths (t s : ℕ) (ht : 2 * t + (20 - 2 * t) = 20) (hs : 4 * s = 20) :
  t / s = 4 / 3 :=
by
  sorry

end ratio_of_side_lengths_l161_161856


namespace maximize_operation_l161_161439

-- Definitions from the conditions
def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- The proof statement
theorem maximize_operation : ∃ n, is_three_digit_integer n ∧ (∀ m, is_three_digit_integer m → 3 * (300 - m) ≤ 600) :=
by {
  -- Placeholder for the actual proof
  sorry
}

end maximize_operation_l161_161439


namespace area_of_trapezium_l161_161396

theorem area_of_trapezium 
  (a b h : ℝ)
  (ha : a = 20)
  (hb : b = 18)
  (hh : h = 13)
  : (1/2 * (a + b) * h = 247) :=
by
  rw [ha, hb, hh]
  have h₁ : 1/2 * (20 + 18) * 13 = 1/2 * 38 * 13 := by ring
  have h₂ : 1/2 * 38 * 13 = 19 * 13 := by ring
  have h₃ : 19 * 13 = 247 := by norm_num
  rw [h₁, h₂, h₃]
  exact rfl
-- sorry

end area_of_trapezium_l161_161396


namespace distance_from_P_to_focus_l161_161825

-- Given Conditions
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def y_axis_distance (x : ℝ) : Prop := abs x = 4

-- Coordinates of the focus
def focus : ℝ × ℝ := (2, 0)

-- Distance function
def distance (a b: ℝ × ℝ) : ℝ := 
  real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Proof statement
theorem distance_from_P_to_focus : 
  ∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧ y_axis_distance P.1 ∧ distance P focus = 6 :=
by
  sorry

end distance_from_P_to_focus_l161_161825


namespace smallest_n_has_three_digits_l161_161255
open Nat

/--
Let n be the smallest positive integer such that:
  1. n is divisible by 30,
  2. n^2 is a perfect fourth power,
  3. n^4 is a perfect square.
Prove that the number of digits of n is 3.
-/
theorem smallest_n_has_three_digits :
  ∃ n : ℕ, (∀ m : ℕ, (m > 0 ∧ (m % 30 = 0) ∧ (∃ k : ℕ, m^2 = k^4) ∧ (∃ l : ℕ, m^4 = l^2)) → n ≤ m)
    ∧ n % 30 = 0
    ∧ (∃ k : ℕ, n^2 = k^4)
    ∧ (∃ l : ℕ, n^4 = l^2)
    ∧ (digits 10 n).length = 3 :=
begin
  sorry
end

end smallest_n_has_three_digits_l161_161255


namespace max_value_of_trigonometric_sum_l161_161536

/-- 
  For all real numbers θ₁, θ₂, θ₃, θ₄ and θ₅, 
  the expression cos(θ₁) * sin(θ₂) + 2 * cos(θ₂) * sin(θ₃) + 3 * cos(θ₃) * sin(θ₄) +
  4 * cos(θ₄) * sin(θ₅) + 5 * cos(θ₅) * sin(θ₁) is maximized with value 15/2.
-/
theorem max_value_of_trigonometric_sum (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) :
  cos θ₁ * sin θ₂ + 2 * cos θ₂ * sin θ₃ + 3 * cos θ₃ * sin θ₄ + 
  4 * cos θ₄ * sin θ₅ + 5 * cos θ₅ * sin θ₁ ≤ 15 / 2 := by
  sorry

end max_value_of_trigonometric_sum_l161_161536


namespace percentage_salt_solution_l161_161599

theorem percentage_salt_solution (P : ℝ) (V_initial V_added V_final : ℝ) (C_initial C_final : ℝ) :
  V_initial = 30 ∧ C_initial = 0.20 ∧ V_final = 60 ∧ C_final = 0.40 → 
  V_added = 30 → 
  (C_initial * V_initial + (P / 100) * V_added) / V_final = C_final →
  P = 60 :=
by
  intro h
  sorry

end percentage_salt_solution_l161_161599


namespace hyperbola_equation_l161_161940

-- Definitions of the given conditions
def ellipse_D (x y : ℝ) : Prop := (x^2 / 50 + y^2 / 25 = 1)
def circle_M (x y : ℝ) : Prop := (x^2 + (y-5)^2 = 9)
def hyperbola_G (x y : ℝ) (a b : ℝ) : Prop := (a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1)

-- The focus points shared by ellipse D and hyperbola G
def foci (F1 F2 : ℝ × ℝ) : Prop := (F1 = (-5, 0) ∧ F2 = (5, 0))

-- Asymptotes tangency condition for hyperbola G with circle M
def asymptote_tangent_condition (a b : ℝ) : Prop := (a^2 + b^2 = 25) ∧ ((|5*a| / sqrt (a^2 + b^2) = 3))

-- The final proof problem
theorem hyperbola_equation :
  ∃ a b : ℝ, foci (-5, 0) (5, 0) ∧ asymptote_tangent_condition a b ∧ hyperbola_G x y a b ↔ hyperbola_G x y 3 4 :=
by
  sorry

end hyperbola_equation_l161_161940


namespace angle_ADB_is_50_l161_161222

theorem angle_ADB_is_50
  (ABCD_is_convex : True)
  (angle_BCD : ℝ)
  (angle_ACB : ℝ)
  (angle_ABD : ℝ)
  (h1 : angle_BCD = 80)
  (h2 : angle_ACB = 50)
  (h3 : angle_ABD = 30)
  : ∠ ADB = 50 :=
by
  sorry

end angle_ADB_is_50_l161_161222


namespace P_ratio_one_l161_161333

-- Given conditions
variables (f P : Polynomial ℂ)
variable (r : ℕ → ℂ)

-- Conditions derived from the problem
axiom f_eq : f = X^2009 + 19 * X^2008 + 1
axiom f_roots : ∀ j : ℕ, 1 ≤ j ∧ j ≤ 2009 → f.eval (r j) = 0
axiom P_property : ∀ j : ℕ, 1 ≤ j ∧ j ≤ 2009 → P.eval (r j + 1 / r j) = 0

-- Prove that P(1) / P(-1) = 1
theorem P_ratio_one : P.eval 1 / P.eval (-1) = 1 :=
sorry

end P_ratio_one_l161_161333


namespace correct_shoe_size_production_priority_l161_161430

theorem correct_shoe_size_production_priority
  (total_people : ℕ)
  (shoe_sizes : list ℕ)
  (number_of_people : list ℕ)
  (median_shoe_size : ℕ)
  (mode_shoe_size : ℕ)
  (average_shoe_size : ℤ)
  (correct_statement : string) :
  total_people = 120 →
  shoe_sizes = [24, 24.5, 25, 25.5, 26, 26.5, 27] →
  number_of_people = [8, 15, 20, 25, 30, 20, 2] →
  median_shoe_size = 25.5 →
  mode_shoe_size = 26 →
  average_shoe_size = 25.5 →
  correct_statement = "D" :=
by
  intros h_total_people h_shoe_sizes h_number_of_people h_median_shoe_size h_mode_shoe_size h_average_shoe_size
  sorry

end correct_shoe_size_production_priority_l161_161430


namespace each_son_can_make_l161_161027

noncomputable def land_profit
    (total_land : ℕ) -- measured in hectares
    (num_sons : ℕ)
    (profit_per_section : ℕ) -- profit in dollars per 750 m^2 per 3 months
    (hectare_to_m2 : ℕ) -- conversion factor from hectares to square meters
    (section_area : ℕ) -- 750 m^2
    (periods_per_year : ℕ) : ℕ :=
  let each_son's_share := total_land * hectare_to_m2 / num_sons in
  let num_sections := each_son's_share / section_area in
  num_sections * profit_per_section * periods_per_year

theorem each_son_can_make
    (total_land : ℕ)
    (num_sons : ℕ)
    (profit_per_section : ℕ)
    (hectare_to_m2 : ℕ)
    (section_area : ℕ)
    (periods_per_year : ℕ) :
  total_land = 3 ∧
  num_sons = 8 ∧
  profit_per_section = 500 ∧
  hectare_to_m2 = 10000 ∧
  section_area = 750 ∧
  periods_per_year = 4 →
  land_profit total_land num_sons profit_per_section hectare_to_m2 section_area periods_per_year = 10000 :=
by
  intros h
  cases h
  sorry

end each_son_can_make_l161_161027


namespace sum_denominators_l161_161768

theorem sum_denominators (a b: ℕ) (h_coprime : Nat.gcd a b = 1) :
  (3:ℚ) / (5 * b) + (2:ℚ) / (9 * b) + (4:ℚ) / (15 * b) = 28 / 45 →
  5 * b + 9 * b + 15 * b = 203 :=
by
  sorry

end sum_denominators_l161_161768


namespace eccentricity_ellipse_eq_half_l161_161562

variables {a b c m n : ℝ}

-- Conditions for the ellipse and the hyperbola
def is_ellipse (a b : ℝ) := 0 < b ∧ b < a
def is_hyperbola (m n : ℝ) := 0 < m ∧ 0 < n

-- Conditions for the shared foci
def shared_foci (a b c m n : ℝ) := 
  c^2 = a^2 - b^2 ∧ c^2 = m^2 + n^2

-- Geometric mean condition
def geometric_mean (a c m : ℝ) := c^2 = a * m

-- Arithmetic mean condition
def arithmetic_mean (n m c : ℝ) := 2 * n^2 = 2 * m^2 + c^2

-- Eccentricity of the ellipse
def eccentricity (a c : ℝ) := c / a

theorem eccentricity_ellipse_eq_half
  (h_ellipse : is_ellipse a b)
  (h_hyperbola : is_hyperbola m n)
  (h_shared_foci : shared_foci a b c m n)
  (h_geometric_mean: geometric_mean a c m)
  (h_arithmetic_mean : arithmetic_mean n m c) :
  eccentricity a c = 1 / 2 := 
sorry

end eccentricity_ellipse_eq_half_l161_161562


namespace joan_total_apples_l161_161656

theorem joan_total_apples (initial_apples given_apples : ℕ) (h1 : initial_apples = 43) (h2 : given_apples = 27) : initial_apples + given_apples = 70 := 
by 
  rw [h1, h2]
  norm_num

end joan_total_apples_l161_161656


namespace least_area_regular_ngon_least_perimeter_regular_ngon_l161_161395

-- Define essential properties of polygons and circles
variables {n : ℕ} {S : Circle}

-- A definition of a circumscribed polygon
structure circumscribed_polygon (S : Circle) :=
  (vertices : list Point)
  (is_circumscribed : ∀ v ∈ vertices, dist v S.center = S.radius)

-- A definition of a regular polygon circumscribed around a circle
def is_regular (P : circumscribed_polygon S) : Prop :=
  (all_equal P.vertices.dist)

-- Area of a polygon (assume some area function already defined)
def area (P : circumscribed_polygon S) : ℝ := sorry

-- Perimeter of a polygon (assume some perimeter function already defined)
def perimeter (P : circumscribed_polygon S) : ℝ := sorry

-- Part (a) Theorem: Among all n-gons circumscribed about circle S, the one with the smallest area is the regular n-gon
theorem least_area_regular_ngon (P : circumscribed_polygon S) : is_regular P → ∀ Q : circumscribed_polygon S, area P ≤ area Q :=
sorry

-- Part (b) Theorem: Among all n-gons circumscribed about circle S, the one with the smallest perimeter is the regular n-gon
theorem least_perimeter_regular_ngon (P : circumscribed_polygon S) : is_regular P → ∀ Q : circumscribed_polygon S, perimeter P ≤ perimeter Q :=
sorry

end least_area_regular_ngon_least_perimeter_regular_ngon_l161_161395


namespace area_triangle_ABC_l161_161331

-- Definition of points and reflection functions
structure Point where
  x : ℤ
  y : ℤ

def reflectOverYAxis (p : Point) : Point :=
  {x := -p.x, y := p.y}

def reflectOverLineYEqNegX (p : Point) : Point :=
  {x := -p.y, y := -p.x}

-- Given conditions
def A := { x := 3, y := 4 } : Point
def B := reflectOverYAxis A
def C := reflectOverLineYEqNegX B

-- Lean statement to prove the area of triangle ABC
theorem area_triangle_ABC : (1/2 : ℚ) * 6 * 1 = 3 := by
  sorry

end area_triangle_ABC_l161_161331


namespace range_my_function_l161_161872

noncomputable def my_function (x : ℝ) : ℝ :=
  abs (x + 5) - abs (x - 3)

theorem range_my_function : set.range my_function = set.Icc (-8 : ℝ) (8 : ℝ) :=
  by
    sorry

end range_my_function_l161_161872


namespace sum_of_squares_is_77_l161_161744

-- Definitions based on the conditions
def consecutive_integers (a : ℕ) : set ℕ := {a - 1, a, a + 1}
def product_of_consecutive_integers (a : ℕ) : ℕ := (a - 1) * a * (a + 1)
def sum_of_consecutive_integers (a : ℕ) : ℕ := (a - 1) + a + (a + 1)
def sum_of_squares_of_consecutive_integers (a : ℕ) : ℕ := (a - 1)^2 + a^2 + (a + 1)^2

-- Condition that the product of these integers is 8 times their sum
axiom product_condition (a : ℕ) (h : a > 0) : product_of_consecutive_integers a = 8 * sum_of_consecutive_integers a

-- Statement to prove
theorem sum_of_squares_is_77 (a : ℕ) (h : a > 0) (hc : product_of_consecutive_integers a = 8 * sum_of_consecutive_integers a) : sum_of_squares_of_consecutive_integers a = 77 :=
by
  sorry

end sum_of_squares_is_77_l161_161744


namespace equal_squares_and_lshapes_possible_more_squares_than_lshapes_impossible_l161_161426

-- Definitions of conditions
def isValidPartition (x y : Nat) : Prop := 4 * x + 3 * y = 98

-- Part (a)
theorem equal_squares_and_lshapes_possible :
  ∃ (x y : Nat), x = y ∧ isValidPartition x y :=
begin
  use 14,
  use 14,
  split,
  { refl },
  { unfold isValidPartition, simp },
end

-- Part (b)
theorem more_squares_than_lshapes_impossible :
  ¬ ∃ (x y : Nat), x > y ∧ isValidPartition x y :=
by sorry

end equal_squares_and_lshapes_possible_more_squares_than_lshapes_impossible_l161_161426


namespace circumcenter_of_stp_l161_161560

theorem circumcenter_of_stp
  (K L M S T P O : Point)
  (triangle KLM : Triangle)
  (circle_K : Circle)
  (circle_M : Circle)
  (circumcircle_ω : Circle)
  (L_on_circle_K : CirclePassesThrough K L)
  (L_on_circle_M : CirclePassesThrough M L)
  (intersect_at_P : CircleIntersects circle_K circle_M P)
  (intersect_ω_at_S_T : CircleIntersects circumcircle_ω S ∧ CircleIntersects circumcircle_ω T)
  (LP_intersects_ω_at_O : LineIntersectsAt LP circumcircle_ω O)
  (acute_triangle : IsAcuteAngled KLM) :
  IsCircumcenter O S T P := 
sorry

end circumcenter_of_stp_l161_161560


namespace infinite_nested_radical_l161_161101

theorem infinite_nested_radical : 
  (x : Real) (h : x = Real.sqrt (3 - x)) : 
  x = (-1 + Real.sqrt 13) / 2 :=
by
  sorry

end infinite_nested_radical_l161_161101


namespace john_books_purchase_l161_161236

theorem john_books_purchase : 
  let john_money := 4575
  let book_price := 325
  john_money / book_price = 14 :=
by
  sorry

end john_books_purchase_l161_161236


namespace floor_neg_seven_fourths_l161_161475

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l161_161475


namespace area_of_polygon_intersection_l161_161817

-- Define the coordinates of points on the cube
def Point : Type := ℝ × ℝ × ℝ

def pointA : Point := (0, 0, 0)
def pointB : Point := (20, 0, 0)
def pointC : Point := (20, 0, 20)
def pointD : Point := (20, 20, 20)

def pointP : Point := (3, 0, 0)
def pointQ : Point := (20, 0, 8)
def pointR : Point := (20, 12, 20)

-- Define the cube and the conditions
def Cube : Type := { a : Point // a ∈ {pointA, pointB, pointC, pointD} }

-- Function to calculate the area, given vertices
def area_polygon (vertices : List Point) : ℝ :=
  -- Implementation of the area calculation (not provided, as we skip the proof)
  sorry

-- Statement of the theorem
theorem area_of_polygon_intersection : 
  area_polygon [pointA, pointB, pointC, pointD] = 400 :=
by
  sorry

end area_of_polygon_intersection_l161_161817


namespace floor_neg_seven_over_four_l161_161504

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l161_161504


namespace problem_1_problem_2_problem_3_l161_161321

def f (x : ℝ) := x / (1 + x^2)

theorem problem_1 : ∀ (a b : ℝ), 
  (∀ x, f (-x) = -f x) ∧ f (1 / 2) = 2 / 5 → 
  f = (λ x, x / (1 + x^2)) :=
sorry

theorem problem_2 :
  ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → 
  f x2 - f x1 > 0 :=
sorry

theorem problem_3 :
  ∀ t : ℝ, 0 < t ∧ t < 1 / 2 → 
  f (t - 1) + f t < 0 :=
sorry

end problem_1_problem_2_problem_3_l161_161321


namespace arithmetic_geometric_consecutive_l161_161068

theorem arithmetic_geometric_consecutive (a b c : ℝ) (r : ℝ) 
  (h1 : b = a)
  (h2 : c = a + r)
  (h3 : a² ≠ (a - r) * (a + r)) :
  a = b ∧ b = c := 
sorry

end arithmetic_geometric_consecutive_l161_161068


namespace radius_of_circular_film_l161_161272

theorem radius_of_circular_film (r_canister h_canister t_film R: ℝ) 
  (V: ℝ) (h1: r_canister = 5) (h2: h_canister = 10) 
  (h3: t_film = 0.2) (h4: V = 250 * Real.pi): R = 25 * Real.sqrt 2 :=
by
  sorry

end radius_of_circular_film_l161_161272


namespace integer_values_satisfying_square_root_condition_l161_161347

theorem integer_values_satisfying_square_root_condition :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 := sorry

end integer_values_satisfying_square_root_condition_l161_161347


namespace cube_coloring_l161_161701

theorem cube_coloring {colors : ℕ} (c : ℕ) (n1 n2 n3 : ℕ) (h_total_colors : colors = 5) (h_faces_painted : n1 = 1 ∧ n2 = 2 ∧ n3 = 3) (h_different_colors : ∀ i j ∈ {1,2,3,4,5,6}, i ≠ j → i.color ≠ j.color) :
  c = 13 := 
sorry

end cube_coloring_l161_161701


namespace area_of_triangle_l161_161917

theorem area_of_triangle (a b c : ℝ) (h₁ : a + b = 14) (h₂ : c = 10) (h₃ : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 24 :=
  sorry

end area_of_triangle_l161_161917


namespace hyperbola_hkab_result_l161_161218

noncomputable def hyperbola_hkab : ℝ :=
let h := 1 in
let k := -1 in
let a := 3 in
let c := 6 in
let b_squared := c * c - a * a in
let b := real.sqrt b_squared in
h + k + a + b

theorem hyperbola_hkab_result :
  hyperbola_hkab = 3 * real.sqrt 3 + 3 := by
sory

end hyperbola_hkab_result_l161_161218


namespace find_k_l161_161586

noncomputable def condition (k : ℝ) : Prop :=
  ∃ (x1 x2 b : ℝ), 
    (y1 : ℝ) = (6 - 3 * k) / x1 ∧
    (y2 : ℝ) = (6 - 3 * k) / x2 ∧
    y1 = -7 * x1 + b ∧
    y2 = -7 * x2 + b ∧
    k > 1 ∧
    k ≠ 2 ∧
    x1 * x2 > 0

theorem find_k : ∃ k, 1 < k ∧ k < 2 ∧ condition k :=
by 
  use 1.5
  sorry

end find_k_l161_161586


namespace total_get_well_cards_l161_161693

def dozens_to_cards (d : ℕ) : ℕ := d * 12
def hundreds_to_cards (h : ℕ) : ℕ := h * 100

theorem total_get_well_cards 
  (d_hospital : ℕ) (h_hospital : ℕ)
  (d_home : ℕ) (h_home : ℕ) :
  d_hospital = 25 ∧ h_hospital = 7 ∧ d_home = 39 ∧ h_home = 3 →
  (dozens_to_cards d_hospital + hundreds_to_cards h_hospital +
   dozens_to_cards d_home + hundreds_to_cards h_home) = 1768 :=
by
  intros
  sorry

end total_get_well_cards_l161_161693


namespace math_problem_l161_161962

def letters := "MATHEMATICS".toList

def vowels := "AAEII".toList
def consonants := "MTHMTCS".toList
def fixed_t := 'T'

def factorial (n : Nat) : Nat := 
  if n = 0 then 1 
  else n * factorial (n - 1)

def arrangements (n : Nat) (reps : List Nat) : Nat := 
  factorial n / reps.foldr (fun r acc => factorial r * acc) 1

noncomputable def vowel_arrangements := arrangements 5 [2, 2]
noncomputable def consonant_arrangements := arrangements 6 [2]

noncomputable def total_arrangements := vowel_arrangements * consonant_arrangements

theorem math_problem : total_arrangements = 10800 := by
  sorry

end math_problem_l161_161962


namespace original_number_l161_161822

theorem original_number (x : ℝ) (h : 1.2 * x = 1080) : x = 900 := by
  sorry

end original_number_l161_161822


namespace floor_neg_seven_quarter_l161_161477

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l161_161477


namespace quadrilateral_area_is_correct_l161_161425

noncomputable def area_of_quadrilateral : ℝ :=
  let A := (1:ℝ, 1:ℝ)
  let B := (4:ℝ, 1:ℝ)
  let C := (1:ℝ, 3:ℝ)
  let D := (20:ℝ, 22:ℝ)
  let triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
    0.5 * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))
  triangle_area A B D + triangle_area A C D

theorem quadrilateral_area_is_correct :
  area_of_quadrilateral = 50.5 := 
by
    sorry

end quadrilateral_area_is_correct_l161_161425


namespace transformed_variance_l161_161176

theorem transformed_variance (data : Fin 10 → ℝ) (h : variance data = 3) :
  variance (λ i, 2 * data i + 3) = 12 :=
sorry

end transformed_variance_l161_161176


namespace percentage_fruits_in_good_condition_l161_161831

-- Definitions
def total_fruits : ℕ := 2450
def oranges : ℕ := 600
def bananas : ℕ := 500
def apples : ℕ := 450
def pears : ℕ := 400
def strawberries : ℕ := 300
def kiwis : ℕ := 200

def rotten_percentage_oranges : ℝ := 0.14
def rotten_percentage_bananas : ℝ := 0.08
def rotten_percentage_apples : ℝ := 0.10
def rotten_percentage_pears : ℝ := 0.11
def rotten_percentage_strawberries : ℝ := 0.16
def rotten_percentage_kiwis : ℝ := 0.05

-- Calculate the number of rotten fruits for each type
def rotten_oranges : ℕ := (rotten_percentage_oranges * oranges).toNat
def rotten_bananas : ℕ := (rotten_percentage_bananas * bananas).toNat
def rotten_apples : ℕ := (rotten_percentage_apples * apples).toNat
def rotten_pears : ℕ := (rotten_percentage_pears * pears).toNat
def rotten_strawberries : ℕ := (rotten_percentage_strawberries * strawberries).toNat
def rotten_kiwis : ℕ := (rotten_percentage_kiwis * kiwis).toNat

def total_rotten_fruits : ℕ := rotten_oranges + rotten_bananas + rotten_apples + rotten_pears + rotten_strawberries + rotten_kiwis

def total_good_fruits : ℕ := total_fruits - total_rotten_fruits

-- Theorem statement
theorem percentage_fruits_in_good_condition : 
  ((total_good_fruits.toFloat / total_fruits.toFloat) * 100) = 88.94 := 
by
  sorry

end percentage_fruits_in_good_condition_l161_161831


namespace regular_pyramid_sum_of_distances_constant_l161_161133

theorem regular_pyramid_sum_of_distances_constant
  {P : Type} {pyramid : Type} [RegularPyramid pyramid] (base_plane : Plane (base pyramid))
  (any_point_P : Point base_plane) :
  let perpendicular := perpendicular_from_point_to_plane any_point_P base_plane
  in let intersections := intersection_points_with_faces perpendicular (faces pyramid)
  in sum_of_distances any_point_P intersections = constant :=
by sorry

end regular_pyramid_sum_of_distances_constant_l161_161133


namespace sharon_harvest_l161_161596

variable (G : ℝ) (S : ℝ)

-- Conditions
def GregHarvested (G : ℝ) : Prop := G = 0.4
def GregHarvestedMore (G S : ℝ) : Prop := G = S + 0.3

-- Theorem to prove
theorem sharon_harvest : GregHarvested G → GregHarvestedMore G S → S = 0.1 := by
  intro hg
  intro hgm
  have h : S = 0.4 - 0.3 := by
    rw [hgm, hg]
    exact rfl
  rw [sub_eq_add_neg, add_neg_self, add_zero] at h
  exact h
  sorry

end sharon_harvest_l161_161596


namespace number_of_integer_solutions_l161_161682

theorem number_of_integer_solutions (x : ℤ) :
  (∃ n : ℤ, n^2 = x^4 + 8*x^3 + 18*x^2 + 8*x + 36) ↔ x = -1 :=
sorry

end number_of_integer_solutions_l161_161682


namespace infinite_primes_x_squared_plus_x_plus_one_eq_py_l161_161707

theorem infinite_primes_x_squared_plus_x_plus_one_eq_py :
  ∃^∞ p : ℕ, Prime p ∧ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 + x + 1 = p * y :=
by
  sorry

end infinite_primes_x_squared_plus_x_plus_one_eq_py_l161_161707


namespace S_3_eq_11_S_n_general_l161_161257

-- Define x_n as {1, 2, ..., n}
def x_n (n : ℕ) : Finset ℕ := Finset.range (n + 1) \ {0}

-- Define f(A) as the smallest element in A
def f (A : Finset ℕ) : ℕ := A.min' ⟨1, by simp [x_n]⟩

-- Define S_n as the sum of f(A) across all non-empty subsets of x_n
def S_n (n : ℕ) : ℕ := 
  (Finset.powerset (x_n n)).filter (λ A, ¬A = ∅).sum f

-- Theorems to prove the given results
theorem S_3_eq_11 : S_n 3 = 11 :=
by sorry

theorem S_n_general (n : ℕ) : S_n n = 2^(n+1) - n - 2 :=
by sorry

end S_3_eq_11_S_n_general_l161_161257


namespace sequence_identity_l161_161754

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 - n + 1

noncomputable def a : ℕ → ℕ
| 1     := 2
| (n+1) := S (n+1) - S n

theorem sequence_identity (n : ℕ) :
  a n = if n = 1 then 2 else 4 * n - 3 :=
by
  sorry

end sequence_identity_l161_161754


namespace modulus_of_z_is_sqrt_10_l161_161929

open Complex

-- Define the complex number z using the given condition
def z : ℂ := 3 + 1 * I

-- The hypothesis given in the problem
axiom h_eq : z / (1 + I) = 1 - 2 * I

-- Define the expected modulus of the complex number z
def modulus_z : ℝ := Complex.abs z

-- The target statement is to prove that the modulus of z is √10
theorem modulus_of_z_is_sqrt_10 : modulus_z = Real.sqrt 10 := by
  sorry

end modulus_of_z_is_sqrt_10_l161_161929


namespace P_gt_neg1_l161_161270

noncomputable def X : MeasureTheory.Measure ℝ := sorry

axiom normal_dist (X : MeasureTheory.Measure ℝ) : True := sorry

variable {p : ℝ}

axiom P_gt_1 (hX : X) : prob {ω | ω > 1} = p := sorry

theorem P_gt_neg1 (hX : X) : prob {ω | ω > -1} = 1 - p := sorry

end P_gt_neg1_l161_161270


namespace simple_interest_rate_l161_161060

/-- 
  Given conditions:
  1. Time period T is 10 years.
  2. Simple interest SI is 7/5 of the principal amount P.
  Prove that the rate percent per annum R for which the simple interest is 7/5 of the principal amount in 10 years is 14%.
-/
theorem simple_interest_rate (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ) (hT : T = 10) (hSI : SI = (7 / 5) * P) : 
  (SI = (P * R * T) / 100) → R = 14 := 
by 
  sorry

end simple_interest_rate_l161_161060


namespace sqrt_equation_solution_l161_161187

theorem sqrt_equation_solution (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 :=
sorry

end sqrt_equation_solution_l161_161187


namespace inequality_holds_l161_161679

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end inequality_holds_l161_161679


namespace area_of_shaded_region_l161_161773

open BigOperators
open Real

/-- Define the squares and their placement -/
structure Square where
  side_length : ℝ
  pos_x : ℝ
  pos_y : ℝ

/-- Define the points and intersection -/
structure Point where
  x : ℝ
  y : ℝ

-- Definitions based on provided conditions
def squareA : Square := ⟨8, 0, 0⟩
def squareB : Square := ⟨6, 8, 0⟩

def point_D (s : Square) : Point := ⟨s.pos_x, s.pos_y + s.side_length⟩
def point_E (s : Square) : Point := ⟨s.pos_x + s.side_length, s.pos_y⟩
def point_P : Point := ⟨8, 6⟩
def point_A : Point := ⟨0, 0⟩

-- Area calculation for quadrilateral APEG
def area_APEG (A P E G : Point) : ℝ := 
  let AE := (E.x - A.x) + (P.x - E.x)
  let height := point_D squareA.y
  1 / 2 * (AE) * height

theorem area_of_shaded_region :
  let A := point_A
  let P := point_P
  let E := point_E squareB
  let G := point_D squareB
  area_APEG A P E G = 18 := by
  sorry

end area_of_shaded_region_l161_161773


namespace max_number_of_liars_l161_161210

def Room : Type := ℤ × ℤ   -- Represent each room by a coordinate pair (i, j) in a 4x4 grid

def isAdjacent (r1 r2 : Room) : Prop := 
  (r1.1 = r2.1 ∧ (r1.2 = r2.2 + 1 ∨ r1.2 + 1 = r2.2)) ∨ 
  (r1.2 = r2.2 ∧ (r1.1 = r2.1 + 1 ∨ r1.1 + 1 = r2.1))

def isLiar (p : Room → Bool) (r : Room) : Prop := 
  (∃ adj : Room, isAdjacent r adj ∧ p adj = true)

def maxLiars (p : Room → Bool) : ℕ := 
  (∑ r in list.finRange 16, if p (r / 4, r % 4) then 1 else 0)

theorem max_number_of_liars : ∀ (p : Room → Bool), (∀ r, isLiar p r) → ∀ i j, (0 ≤ i ∧ i < 4) → (0 ≤ j ∧ j < 4) → maxLiars p ≤ 8 :=
by
  sorry

end max_number_of_liars_l161_161210


namespace derivative_at_1_l161_161380

noncomputable def f : ℝ → ℝ := λ x, x^2

theorem derivative_at_1 : deriv f 1 = 2 :=
by
  sorry

end derivative_at_1_l161_161380


namespace part_I_part_II_l161_161948

noncomputable def f (x a : ℝ) := abs (x - 2 * a) - abs (x - a)

theorem part_I (a : ℝ) : (f 1 a > 1) ↔ (a ∈ set.Iio (-1) ∨ a ∈ set.Ioi 1) :=
sorry

theorem part_II (a : ℝ) (ha : a < 0) :
  (∀ x y ∈ set.Iic a, f x a ≤ abs (y + 2020) + abs (y - a)) ↔ (a ∈ set.Ico (-1010) 0) :=
sorry

end part_I_part_II_l161_161948


namespace hyperbola_integer_points_count_l161_161608

theorem hyperbola_integer_points_count :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), 
    (p ∈ S ↔ (∃ (x y : ℤ), p = (x, y) ∧ y = 2013 / x)) 
    ∧ S.card = 16 := 
by 
  sorry

end hyperbola_integer_points_count_l161_161608


namespace total_earnings_l161_161438

-- Definitions from the conditions.
def LaurynEarnings : ℝ := 2000
def AureliaEarnings : ℝ := 0.7 * LaurynEarnings

-- The theorem to prove.
theorem total_earnings (hL : LaurynEarnings = 2000) (hA : AureliaEarnings = 0.7 * 2000) :
    LaurynEarnings + AureliaEarnings = 3400 :=
by
    sorry  -- Placeholder for the proof.

end total_earnings_l161_161438


namespace phase_shift_of_sine_function_l161_161086

-- Definitions based on given conditions
def sine_function (A B C x : ℝ) : ℝ := A * Real.sin (B * x + C)

-- Problem statement: Prove the phase shift is -π/8
theorem phase_shift_of_sine_function :
  ∀ x : ℝ, sine_function 5 2 (π / 4) x = 5 * Real.sin (2 * x + π / 4) ->
  - (π / 4 / 2) = - (π / 8) :=
by sorry

end phase_shift_of_sine_function_l161_161086


namespace swimmer_speed_proof_l161_161433

/-- Define the swimmer's speed in still water, denoted as v -/
def swimmer_speed_in_still_water : ℝ := 4

/-- Given conditions for the problem:
    1. Speed of the current is 1 km/h
    2. Time taken to swim against the current for 6 km is 2 hours -/
structure Conditions :=
  (current_speed : ℝ := 1)
  (time_against_current : ℝ := 2)
  (distance_against_current : ℝ := 6)

/-- The theorem that needs to be proven:
    The swimmer's speed in still water given the conditions is 4 km/h -/
theorem swimmer_speed_proof (c : Conditions) : swimmer_speed_in_still_water = 4 :=
by
  sorry

end swimmer_speed_proof_l161_161433


namespace number_and_sum_of_f2_l161_161667

noncomputable def f : ℝ → ℝ := sorry

theorem number_and_sum_of_f2 :
  f 1 = 1 →
  (∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) →
  ∃ n s : ℕ, n × s = 2 ∧ n = 1 ∧ s = 2 :=
by
  intros h1 h2
  use (1, 2)
  split
  case left =>
    ring
  case right =>
    split
    exact rfl
    exact rfl
  sorry

end number_and_sum_of_f2_l161_161667


namespace area_of_triangle_l161_161972

theorem area_of_triangle
  (CD : ℝ) (hCD : CD = Real.sqrt 2)
  (is_45_45_90 : is_45_45_90_triangle CD)
  : area_of_triangle ABC = 2 := 
by
  sorry

-- Definitions to clarify the conditions (not part of the final theorem, just placeholders for clarity)

def is_45_45_90_triangle (CD : ℝ) : Prop :=
  ∃ AD AC : ℝ, AD = CD ∧ AC = CD * Real.sqrt 2

def area_of_triangle (ABC : Type) := 2

end area_of_triangle_l161_161972


namespace integer_points_on_hyperbola_l161_161603

theorem integer_points_on_hyperbola : 
  let points := {(x, y) : Int × Int | y * x = 2013} in points.size = 16 :=
by
  sorry

end integer_points_on_hyperbola_l161_161603


namespace walking_rate_ratio_l161_161810

theorem walking_rate_ratio (R R' : ℝ) (usual_time early_time : ℝ) (H1 : usual_time = 42) (H2 : early_time = 36) 
(H3 : R * usual_time = R' * early_time) : (R' / R = 7 / 6) :=
by
  -- proof to be completed
  sorry

end walking_rate_ratio_l161_161810


namespace total_fence_length_l161_161878

variable (Darren Doug : ℝ)

-- Definitions based on given conditions
def Darren_paints_more := Darren = 1.20 * Doug
def Darren_paints_360 := Darren = 360

-- The statement to prove
theorem total_fence_length (h1 : Darren_paints_more Darren Doug) (h2 : Darren_paints_360 Darren) : (Darren + Doug) = 660 := 
by
  sorry

end total_fence_length_l161_161878


namespace camel_cost_6000_l161_161806

variables (cost : Type) [linear_ordered_field cost]

def cost_of_camel (C H O E : cost) (H_nonzero : H ≠ 0) (O_nonzero : O ≠ 0) (E_nonzero : E ≠ 0) : Prop :=
  -- Conditions as per the problem statement
  (10 * C = 24 * H) ∧
  (16 * H = 4 * O) ∧
  (6 * O = 4 * E) ∧
  (10 * E = 150000) ∧
  (C = 6000)

theorem camel_cost_6000 (C H O E : cost) (H_nonzero : H ≠ 0) (O_nonzero : O ≠ 0) (E_nonzero : E ≠ 0) :
  cost_of_camel C H O E H_nonzero O_nonzero E_nonzero → C = 6000 :=
by
  intro h
  sorry

end camel_cost_6000_l161_161806


namespace sum_of_integers_l161_161726

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 15 :=
by {
    sorry
}

end sum_of_integers_l161_161726


namespace floor_of_neg_seven_fourths_l161_161484

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l161_161484


namespace calculation_correct_l161_161450

theorem calculation_correct : 
  ((2 * (15^2 + 35^2 + 21^2) - (3^4 + 5^4 + 7^4)) / (3 + 5 + 7)) = 45 := by
  sorry

end calculation_correct_l161_161450


namespace area_triang_ABF_l161_161648

theorem area_triang_ABF {A B C D E F G : Type} 
  (hD: D ∈ segment A B)
  (hE: E ∈ segment A C)
  (hF: F ∈ segment B C)
  (hG: G ∈ inside_triangle A B C)
  (hG': G ∈ {point_of_concurrence A F B E C D})
  (area_ABC : real := 15)
  (area_ABE : real := 5)
  (area_ACD : real := 10):
  ∃ area_ABF : real, area_ABF = 3 := by
  sorry

end area_triang_ABF_l161_161648


namespace hexagon_area_l161_161915

theorem hexagon_area 
  (area_triangle_QEP : ℝ)
  (H : area_triangle_QEP = 72) 
  : ∃ (s : ℝ), ((3 * real.sqrt 3 / 2) * (s ^ 2) = 864) :=
by
  sorry

end hexagon_area_l161_161915


namespace find_angle_P_l161_161055

-- Definitions for conditions
variables {α β P : ℝ}
variables {A D ACP PCD ABP PBD : ℝ}

def angle_A : ℝ := 39
def angle_D : ℝ := 27
def angle_ACP : ℝ := 2 * PCD
def angle_ABP : ℝ := 2 * PBD

-- The main theorem to prove
theorem find_angle_P (h_ACP : ∠ ACP = 2 * ∠ PCD) 
  (h_ABP : ∠ ABP = 2 * ∠ PBD) 
  (h_A : ∠ A = 39) 
  (h_D : ∠ D = 27 ) : 
  ∠ P = 31 :=
sorry

end find_angle_P_l161_161055


namespace people_left_on_beach_l161_161361

theorem people_left_on_beach : 
  ∀ (initial_first_row initial_second_row initial_third_row left_first_row left_second_row left_third_row : ℕ),
  initial_first_row = 24 →
  initial_second_row = 20 →
  initial_third_row = 18 →
  left_first_row = 3 →
  left_second_row = 5 →
  initial_first_row - left_first_row + (initial_second_row - left_second_row) + initial_third_row = 54 :=
by
  intros initial_first_row initial_second_row initial_third_row left_first_row left_second_row left_third_row
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end people_left_on_beach_l161_161361


namespace new_concentration_is_3194_percent_l161_161844

-- Definitions for the initial conditions.
def vessel1_capacity : ℝ := 2
def vessel1_concentration : ℝ := 0.25

def vessel2_capacity : ℝ := 6
def vessel2_concentration : ℝ := 0.40

def vessel3_capacity : ℝ := 3
def vessel3_concentration : ℝ := 0.55

def vessel4_capacity : ℝ := 4
def vessel4_concentration : ℝ := 0.30

def final_vessel_capacity : ℝ := 18

-- Define the total amount of alcohol in the final mixture.
def total_alcohol : ℝ :=
  vessel1_capacity * vessel1_concentration +
  vessel2_capacity * vessel2_concentration +
  vessel3_capacity * vessel3_concentration +
  vessel4_capacity * vessel4_concentration

-- Define the concentration calculation.
def new_concentration : ℝ := total_alcohol / final_vessel_capacity

-- The proof problem statement.
theorem new_concentration_is_3194_percent :
  (new_concentration * 100).round = 31.94 :=
by
  sorry

end new_concentration_is_3194_percent_l161_161844


namespace obtuse_triangle_ABC_l161_161776

variables {A B C : Point} {ℓ : Line}
variables (rA rB rC : ℝ)

-- Define the tangent conditions and pairwise externally tangent conditions
def circles_tangent_to_line (A B C : Point) (ℓ : Line) (rA rB rC : ℝ) : Prop :=
  tangent_to_line A ℓ rA ∧ tangent_to_line B ℓ rB ∧ tangent_to_line C ℓ rC ∧
  externally_tangent A B rA rB ∧ externally_tangent B C rB rC ∧ externally_tangent A C rA rC

-- Define the obtuse angle condition
def has_obtuse_angle (A B C : Point) : Prop :=
  ∃ γ : ℝ, right_angle < γ ∧ γ ≤ 106.26 ∧ obtuse_angle_of_triangle A B C γ

-- Main statement
theorem obtuse_triangle_ABC (h : circles_tangent_to_line A B C ℓ rA rB rC) : has_obtuse_angle A B C :=
sorry

end obtuse_triangle_ABC_l161_161776


namespace wolves_heads_count_l161_161824

/-- 
A person goes hunting in the jungle and discovers a pack of wolves.
It is known that this person has one head and two legs, 
an ordinary wolf has one head and four legs, and a mutant wolf has two heads and three legs.
The total number of heads of all the people and wolves combined is 21,
and the total number of legs is 57.
-/
theorem wolves_heads_count :
  ∃ (x y : ℕ), (x + 2 * y = 20) ∧ (4 * x + 3 * y = 55) ∧ (x + y > 0) ∧ (x + 2 * y + 1 = 21) ∧ (4 * x + 3 * y + 2 = 57) := 
by {
  sorry
}

end wolves_heads_count_l161_161824


namespace coeff_x4_expansion_l161_161319

theorem coeff_x4_expansion (x : ℝ) :
  (polynomial.coeff ((1 + 2 * polynomial.C x) * (1 - polynomial.C x) ^ 10) 4) = -30 :=
sorry

end coeff_x4_expansion_l161_161319


namespace train_crossing_time_l161_161840

/-- Prove the time it takes for a train of length 50 meters running at 60 km/hr to cross a pole is 3 seconds. -/
theorem train_crossing_time
  (speed_kmh : ℝ)
  (length_m : ℝ)
  (conversion_factor : ℝ)
  (time_seconds : ℝ) :
  speed_kmh = 60 →
  length_m = 50 →
  conversion_factor = 1000 / 3600 →
  time_seconds = 3 →
  time_seconds = length_m / (speed_kmh * conversion_factor) := 
by
  intros
  sorry

end train_crossing_time_l161_161840


namespace pet_store_cages_l161_161032

theorem pet_store_cages 
  (snakes parrots rabbits snake_cage_capacity parrot_cage_capacity rabbit_cage_capacity : ℕ)
  (h_snakes : snakes = 4) 
  (h_parrots : parrots = 6) 
  (h_rabbits : rabbits = 8) 
  (h_snake_cage_capacity : snake_cage_capacity = 2) 
  (h_parrot_cage_capacity : parrot_cage_capacity = 3) 
  (h_rabbit_cage_capacity : rabbit_cage_capacity = 4) 
  : (snakes / snake_cage_capacity) + (parrots / parrot_cage_capacity) + (rabbits / rabbit_cage_capacity) = 6 := 
by 
  sorry

end pet_store_cages_l161_161032


namespace length_AP_l161_161643

-- Definitions of the problem setup
structure Square :=
(side_length : ℕ)
(vertices : list (ℕ × ℕ)) -- simplified representation

structure Rectangle :=
(length : ℕ)
(width : ℕ)
(vertices : list (ℕ × ℕ)) -- simplified representation

-- Problem Definition
def ABCD : Square := {side_length := 8, vertices := [(0,0), (8,0), (8,8), (0,8)]}
def WXYZ : Rectangle := {length := 12, width := 8, vertices := [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]} -- coordinates are unspecified

-- Given conditions
def area_WXYZ := WXYZ.length * WXYZ.width
def shaded_area := area_WXYZ / 3

-- Hypothesis: assuming AD and WX are perpendicular, shaded region's width along AD is the same as the side length of the square
def width_AD := ABCD.side_length
def PD := shaded_area / width_AD

-- Theorem Statement
theorem length_AP : 8 - PD = 4 := by
  sorry

end length_AP_l161_161643


namespace problem_statement_l161_161263

theorem problem_statement 
  (x y z : ℝ) 
  (hx1 : x ≠ 1) 
  (hy1 : y ≠ 1) 
  (hz1 : z ≠ 1) 
  (hxyz : x * y * z = 1) : 
  x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 ≥ 1 :=
sorry

end problem_statement_l161_161263


namespace sequence_bounds_l161_161134

theorem sequence_bounds (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_a1 : a 1 = 1)
  (h_S : ∀ n : ℕ, n > 0 → S n = 2 * a (n + 1))
  (h_S1 : S 1 = 1)
  (h_S_gen : ∀ n : ℕ, n > 0 → S n = (3 / 2)^(n - 1))
  (h_b : ∀ n : ℕ, b n = (-1)^n / S n)
  (h_T : ∀ n : ℕ, T n = ∑ i in finset.range n, b (i + 1))
  (n : ℕ) (h_n : n ≥ 2) :
  1 / 3 ≤ |T n| ∧ |T n| ≤ 7 / 9 := sorry

end sequence_bounds_l161_161134


namespace airline_routes_coloring_l161_161627

variable (N : ℕ) (cities : Finset ℕ)
variable (routes : Finset (ℕ × ℕ))
variable (k : ℕ) (h1 : 2 ≤ k ∧ k ≤ N)
variable (subset_k : Finset (ℕ × ℕ) → Finset ℕ → Finset ℕ)
variable (h2 : ∀ (s : Finset ℕ), 2 ≤ s.card → s.card ≤ N → (subset_k routes s).card ≤ 2 * s.card - 2)

theorem airline_routes_coloring :
  ∃ (color : (ℕ × ℕ) → ℕ), ∀ (cycle : list (ℕ × ℕ)),
    (∀ e ∈ cycle, color e = 0 ∨ color e = 1) →
    ¬ (∀ e ∈ cycle, color e = 0) ∧ ¬ (∀ e ∈ cycle, color e = 1)
:= sorry

end airline_routes_coloring_l161_161627


namespace problem_1_problem_2_l161_161066

theorem problem_1 :
  (-4)^2 * ((-3 / 4) + (-5 / 8)) = -22 := 
by
  have h1 : (-4)^2 = 16 := by norm_num
  have h2 : (-3 / 4) + (-5 / 8) = -11 / 8 := by norm_num
  rw [h1, h2]
  norm_num

theorem problem_2 :
  -2 ^ 2 - (1 - 0.5) * (1 / 3) * (2 - (-4)^2) = -5 / 3 :=
by
  have h3 : -2 ^ 2 = -4 := by norm_num
  have h4 : (1 - 0.5) = 0.5 := by norm_num
  have h5 : (2 - (-4)^2) = -14 := by norm_num
  rw [h3, h4, h5]
  norm_num

end problem_1_problem_2_l161_161066


namespace max_intersections_intersections_ge_n_special_case_l161_161355

variable {n m : ℕ}

-- Conditions: n points on a circumference, m and n are positive integers, relatively prime, 6 ≤ 2m < n
def valid_conditions (n m : ℕ) : Prop := Nat.gcd m n = 1 ∧ 6 ≤ 2 * m ∧ 2 * m < n

-- Maximum intersections I = (m-1)n
theorem max_intersections (h : valid_conditions n m) : ∃ I, I = (m - 1) * n :=
by
  sorry

-- Prove I ≥ n
theorem intersections_ge_n (h : valid_conditions n m) : ∃ I, I ≥ n :=
by
  sorry

-- Special case: m = 3 and n is even
theorem special_case (h : valid_conditions n 3) (hn : Even n) : ∃ I, I = n :=
by
  sorry

end max_intersections_intersections_ge_n_special_case_l161_161355


namespace corn_seed_germination_probability_l161_161405

noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem corn_seed_germination_probability :
  binomial_prob 5 2 0.1 ≈ 0.07 :=
by sorry

end corn_seed_germination_probability_l161_161405


namespace guppy_to_goldfish_food_ratio_l161_161242

-- Definitions from conditions
def numGoldfish : ℕ := 2
def goldfishFoodPerFish : ℕ := 1
def numSwordtails : ℕ := 3
def swordtailFoodPerFish : ℕ := 2
def numGuppies : ℕ := 8
def totalFood : ℕ := 12

-- Theorem to prove the required ratio
theorem guppy_to_goldfish_food_ratio :
  let goldfishFood := numGoldfish * goldfishFoodPerFish in
  let swordtailFood := numSwordtails * swordtailFoodPerFish in
  let guppyFood := totalFood - (goldfishFood + swordtailFood) in
  let guppyFoodPerFish := guppyFood / numGuppies in
  guppyFoodPerFish / goldfishFoodPerFish = 1 / 2 :=
by
  sorry

end guppy_to_goldfish_food_ratio_l161_161242


namespace expand_product_l161_161522

theorem expand_product (x : ℝ) : (x + 3)^2 * (x - 5) = x^3 + x^2 - 21x - 45 :=
by
  sorry

end expand_product_l161_161522


namespace problem_irrational_number_l161_161851

theorem problem_irrational_number :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (√3 : ℝ) = a / b) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (0 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-2 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (1 / 2 : ℝ) = a / b)
:=
by
  sorry

end problem_irrational_number_l161_161851


namespace nth_equation_sum_l161_161277

theorem nth_equation_sum (n : ℕ) : 
  (\sum k in Finset.range (2 * n - 1), (n + k)) = (2 * n - 1) ^ 2 :=
by
  -- The proof would be provided here. Simply stating sorry as placeholder.
  sorry

end nth_equation_sum_l161_161277


namespace yolanda_walking_rate_l161_161793

-- Definitions for the conditions given in the problem
variables (X Y : ℝ) -- Points X and Y
def distance_X_to_Y := 52 -- Distance between X and Y in miles
def Bob_rate := 4 -- Bob's walking rate in miles per hour
def Bob_distance_walked := 28 -- The distance Bob walked in miles
def start_time_diff := 1 -- The time difference (in hours) between Yolanda and Bob starting

-- The statement to prove
theorem yolanda_walking_rate : 
  ∃ (y : ℝ), (distance_X_to_Y = y * (Bob_distance_walked / Bob_rate + start_time_diff) + Bob_distance_walked) ∧ y = 3 := by 
  sorry

end yolanda_walking_rate_l161_161793


namespace arithmetic_prog_sum_l161_161989

theorem arithmetic_prog_sum (a d : ℕ) (h1 : 15 * a + 105 * d = 60) : 2 * a + 14 * d = 8 :=
by
  sorry

end arithmetic_prog_sum_l161_161989


namespace percentage_markup_is_correct_l161_161330

def selling_price : ℝ := 5750
def cost_price : ℝ := 5000
def markup_percentage : ℝ := (selling_price - cost_price) / cost_price * 100

theorem percentage_markup_is_correct : markup_percentage = 15 := by
  sorry

end percentage_markup_is_correct_l161_161330


namespace solve_for_x_l161_161194

theorem solve_for_x (x : ℝ) (y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 :=
by
  -- The proof will go here
  sorry

end solve_for_x_l161_161194


namespace exp_graph_fixed_point_l161_161730

theorem exp_graph_fixed_point (a : ℝ) :
  ∃ (x y : ℝ), x = 3 ∧ y = 4 ∧ y = a^(x - 3) + 3 :=
by
  use 3
  use 4
  split
  · rfl
  split
  · rfl
  · sorry

end exp_graph_fixed_point_l161_161730


namespace consecutive_integers_sum_of_squares_l161_161745

theorem consecutive_integers_sum_of_squares :
  ∃ a : ℕ, 0 < a ∧ ((a - 1) * a * (a + 1) = 8 * (a - 1 + a + a + 1)) → 
  ((a - 1) ^ 2 + a ^ 2 + (a + 1) ^ 2 = 77) :=
begin
  sorry
end

end consecutive_integers_sum_of_squares_l161_161745


namespace hyperbola_integer_points_count_l161_161601

-- Definition of the hyperbolic equation
def hyperbola (x y : ℤ) : Prop :=
  y * x = 2013

-- Condition: We are looking for integer coordinate points (x, y)
def integer_coordinate_points : Set (ℤ × ℤ) :=
  {p | hyperbola p.fst p.snd}

-- Main proof statement
theorem hyperbola_integer_points_count : (integer_coordinate_points.to_finset.card = 16) :=
sorry

end hyperbola_integer_points_count_l161_161601


namespace circle_represents_real_l161_161620

theorem circle_represents_real
  (a : ℝ)
  (h : ∀ x y : ℝ, x^2 + y^2 + 2*y + 2*a - 1 = 0 → ∃ r : ℝ, r > 0) : 
  a < 1 := 
sorry

end circle_represents_real_l161_161620


namespace carrie_remaining_time_l161_161867

open Real

def carrie_speed : Real := 85
def total_trip_distance : Real := 510
def halfway_distance : Real := total_trip_distance / 2

theorem carrie_remaining_time :
  halfway_distance / carrie_speed = 3 := sorry

end carrie_remaining_time_l161_161867


namespace num_positive_integers_21n_perfect_square_l161_161113

noncomputable def num_positive_integers_le_500 (n : ℕ) : ℕ :=
if h : 0 < n ∧ n ≤ 500 ∧ ∃ k, 21 * n = k^2 then 1 else 0

theorem num_positive_integers_21n_perfect_square :
  ∑ n in finset.range 501, num_positive_integers_le_500 n = 4 := 
sorry

end num_positive_integers_21n_perfect_square_l161_161113


namespace log_sum_eq_neg_one_l161_161166

noncomputable def f (x : ℝ) (n : ℕ) : ℝ := x^(n + 1)

theorem log_sum_eq_neg_one :
  (∑ n in finset.range 2013, Real.log 2014 ((n : ℝ + 1) / (n + 2))) = -1 :=
by
  sorry

end log_sum_eq_neg_one_l161_161166


namespace closest_multiple_of_15_to_2028_l161_161387

theorem closest_multiple_of_15_to_2028 : ∃ n, n % 15 = 0 ∧ abs (2028 - n) = 3 :=
by
  use 2025
  split
  { norm_num }
  { norm_num [abs] }

end closest_multiple_of_15_to_2028_l161_161387


namespace floor_neg_seven_over_four_l161_161501

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l161_161501


namespace cookies_ratio_l161_161858

theorem cookies_ratio (T : ℝ) (h1 : 0 ≤ T) (h_total : 5 + T + 1.4 * T = 29) : T / 5 = 2 :=
by sorry

end cookies_ratio_l161_161858


namespace cone_volume_is_correct_l161_161023

-- Definitions
def sector : Type := {angle : ℝ // angle = (5 / 8)}

def radius : ℝ := 5

noncomputable def circumference (r : ℝ) (s : sector) : ℝ := (2 * r * Real.pi) * s.val

noncomputable def base_radius (s : ℝ) : ℝ := s / (2 * Real.pi)

noncomputable def height (r_base : ℝ) (slant_height : ℝ) : ℝ := Real.sqrt (slant_height ^ 2 - r_base ^ 2)

noncomputable def volume (r_base : ℝ) (h : ℝ) : ℝ := (1 / 3) * Real.pi * (r_base ^ 2) * h

-- Theorem statement
theorem cone_volume_is_correct :
  let s := (5 / 8)
  let r := radius
  let s_circumference := circumference r ⟨s, rfl⟩
  let r_base := base_radius s_circumference
  let h := height r_base r
  volume r_base h = 12.66 * Real.pi := by
  sorry

end cone_volume_is_correct_l161_161023


namespace remainder_of_power_is_41_l161_161901

theorem remainder_of_power_is_41 : 
  ∀ (n k : ℕ), n = 2019 → k = 2018 → (n^k) % 100 = 41 :=
  by 
    intros n k hn hk 
    rw [hn, hk] 
    exact sorry

end remainder_of_power_is_41_l161_161901


namespace count_mappings_A_to_B_l161_161179

noncomputable def number_of_mappings : ℕ := Nat.choose 99 49

theorem count_mappings_A_to_B
  (A : Fin 100) (B : Fin 50)
  (f : A → B)
  (h1 : ∀ a1 a2 : A, a1 ≤ a2 → f a1 ≤ f a2)
  (h2 : ∀ b : B, ∃ a : A, f a = b) :
  number_of_mappings = Nat.choose 99 49 :=
by
  sorry

end count_mappings_A_to_B_l161_161179


namespace candy_cases_total_l161_161315

theorem candy_cases_total
  (choco_cases lolli_cases : ℕ)
  (h1 : choco_cases = 25)
  (h2 : lolli_cases = 55) : 
  (choco_cases + lolli_cases) = 80 := by
-- The proof is omitted as requested.
sorry

end candy_cases_total_l161_161315


namespace problem_irrational_number_l161_161850

theorem problem_irrational_number :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (√3 : ℝ) = a / b) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (0 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-2 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (1 / 2 : ℝ) = a / b)
:=
by
  sorry

end problem_irrational_number_l161_161850


namespace melissa_driving_time_l161_161695

theorem melissa_driving_time
  (trips_per_month: ℕ)
  (months_per_year: ℕ)
  (total_hours_per_year: ℕ)
  (total_trips: ℕ)
  (hours_per_trip: ℕ) :
  trips_per_month = 2 ∧
  months_per_year = 12 ∧
  total_hours_per_year = 72 ∧
  total_trips = (trips_per_month * months_per_year) ∧
  hours_per_trip = (total_hours_per_year / total_trips) →
  hours_per_trip = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end melissa_driving_time_l161_161695


namespace find_f_value_l161_161681

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_value :
  f(0) = 0 →
  (∀ x, f(x) + f(1 - x) = 1) →
  (∀ x, f(x / 3) = (1 / 2) * f(x)) →
  (∀ x1 x2 : ℝ, 0 ≤ x1 → x1 < x2 → x2 ≤ 1 → f(x1) ≤ f(x2)) →
  f(1 / 2011) = 1 / 128 :=
by
  intros h0 h_sym h_func_eq h_monotonic
  sorry

end find_f_value_l161_161681


namespace total_fruit_punch_eq_21_l161_161308

def orange_punch : ℝ := 4.5
def cherry_punch := 2 * orange_punch
def apple_juice := cherry_punch - 1.5

theorem total_fruit_punch_eq_21 : orange_punch + cherry_punch + apple_juice = 21 := by 
  -- This is where the proof would go
  sorry

end total_fruit_punch_eq_21_l161_161308


namespace sum_of_all_roots_l161_161902

-- Define the first cubic equation and its sum of roots according to Vieta's formulas
def cubic_eq1 := 3 * x^3 + 2 * x^2 - 9 * x + 15 = 0
def sum_roots_cubic_eq1 := - (2 / 3)

-- Define the second cubic equation and its sum of roots according to Vieta's formulas
def cubic_eq2 := 4 * x^3 - 16 * x^2 + 10 = 0
def sum_roots_cubic_eq2 := - (-16 / 4)

-- The sum of all roots for the given composite equation
theorem sum_of_all_roots :
  (sum_roots_cubic_eq1 + sum_roots_cubic_eq2) = 10 / 3 :=
by
  unfold sum_roots_cubic_eq1 sum_roots_cubic_eq2
  norm_num
  sorry

end sum_of_all_roots_l161_161902


namespace total_time_iggy_runs_correct_l161_161207

noncomputable def total_time_iggy_runs : ℝ :=
  let monday_time := 3 * (10 + 1 + 0.5);
  let tuesday_time := 5 * (9 + 1 + 1);
  let wednesday_time := 7 * (12 - 2 + 2);
  let thursday_time := 10 * (8 + 2 + 4);
  let friday_time := 4 * (10 + 0.25);
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem total_time_iggy_runs_correct : total_time_iggy_runs = 354.5 := by
  sorry

end total_time_iggy_runs_correct_l161_161207


namespace gcd_lcm_mul_l161_161290

theorem gcd_lcm_mul (a b : ℤ) : (Int.gcd a b) * (Int.lcm a b) = a * b := by
  sorry

end gcd_lcm_mul_l161_161290


namespace range_m_l161_161860

def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - y^2 = 1

def focus_1 : ℝ × ℝ := (-Real.sqrt 5, 0)
def focus_2 : ℝ × ℝ := (Real.sqrt 5, 0)

def right_branch (x y : ℝ) : Prop := hyperbola x y ∧ y ≥ 1

theorem range_m (x y m : ℝ) (P : ℝ × ℝ) (hP : right_branch x y)
  (bisector : ∀ (M : ℝ × ℝ), M = (m, 0) → M lies on the angular_bisector focus_1 P focus_2) :
  -1/2 < m ∧ m < +∞ 
:=
sorry

end range_m_l161_161860


namespace cos_diff_l161_161293

theorem cos_diff (x y : ℝ) (hx : x = Real.cos (20 * Real.pi / 180)) (hy : y = Real.cos (40 * Real.pi / 180)) :
  x - y = 1 / 2 :=
by sorry

end cos_diff_l161_161293


namespace number_of_integer_solutions_l161_161761

theorem number_of_integer_solutions :
  {x : ℤ | |x - 2000| + |x| ≤ 9999}.finite.card = 9999 := 
sorry

end number_of_integer_solutions_l161_161761


namespace proof_problem_l161_161587

-- Define the propositions p and q
def p : Prop := ∀ x : ℝ, x ∈ Set.Ioo (-(Real.pi) / 2) 0 → Real.tan x < 0
def q : Prop := ∃ x0 : ℝ, x0 > 0 ∧ 2^x0 = 1 / 2

-- State the proof problem
theorem proof_problem : p ∧ ¬q := by
  sorry

end proof_problem_l161_161587


namespace exists_subset_with_triangular_numbers_l161_161329

theorem exists_subset_with_triangular_numbers :
  ∃ (S : Finset (Finset ℕ)), S.card = 50 ∧ (∀ s ∈ S, s ⊆ Finset.range 201 ∧ s.card ≥ 3) ∧
    ∃ (s ∈ S) (a b c ∈ s), a ≤ b ∧ b ≤ c ∧ a + b > c :=
by
  sorry

end exists_subset_with_triangular_numbers_l161_161329


namespace jim_taxi_total_charge_l161_161796

noncomputable def total_charge (initial_fee : ℝ) (per_mile_fee : ℝ) (mile_chunk : ℝ) (distance : ℝ) : ℝ :=
  initial_fee + (distance / mile_chunk) * per_mile_fee

theorem jim_taxi_total_charge :
  total_charge 2.35 0.35 (2/5) 3.6 = 5.50 :=
by
  sorry

end jim_taxi_total_charge_l161_161796


namespace find_second_number_l161_161010

theorem find_second_number (A B : ℝ) (h1 : A = 3200) (h2 : 0.10 * A = 0.20 * B + 190) : B = 650 :=
by
  sorry

end find_second_number_l161_161010


namespace midpoint_chords_eq_distance_l161_161978

/-- Given that O₁ is the midpoint of chord GH in circle O, and through O₁, two chords AB and CD are drawn.
Chords AC and BD intersect GH at points E and F, respectively. Prove that EO₁ = FO₁. -/
theorem midpoint_chords_eq_distance 
  (O₁ G H A B C D E F : Point) (O : Circle)
  (hO₁_midpoint : midpoint O₁ G H) 
  (hAB_through_O₁ : ∃ k₁, line_through_chord O₁ A B k₁) 
  (hCD_through_O₁ : ∃ k₂, line_through_chord O₁ C D k₂) 
  (hE_intersect : intersect_chord AC GH E) 
  (hF_intersect : intersect_chord BD GH F) :
  distance E O₁ = distance F O₁ := 
sorry

end midpoint_chords_eq_distance_l161_161978


namespace hyperbola_integer_points_count_l161_161606

theorem hyperbola_integer_points_count :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), 
    (p ∈ S ↔ (∃ (x y : ℤ), p = (x, y) ∧ y = 2013 / x)) 
    ∧ S.card = 16 := 
by 
  sorry

end hyperbola_integer_points_count_l161_161606


namespace speed_difference_ava_lily_l161_161058

theorem speed_difference_ava_lily
  (d : ℕ) (lily_time_min : ℕ) (ava_time_min : ℕ)
  (hd : d = 8) (hlily : lily_time_min = 40) (hava : ava_time_min = 15) :
  (8 / (15 / 60 : ℚ) - 8 / (40 / 60 : ℚ) = 20) :=
by
  rw [hd, hlily, hava]
  norm_num
  sorry

end speed_difference_ava_lily_l161_161058


namespace find_number_of_students_l161_161718

theorem find_number_of_students (N : ℕ) (h1 : T = 80 * N) (h2 : (T - 350) / (N - 5) = 90) 
: N = 10 := 
by 
  -- Proof steps would go here. Omitted as per the instruction.
  sorry

end find_number_of_students_l161_161718


namespace part_I_part_II_part_III_l161_161167

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp x / (a * x^2 + b * x + 1)

-- (I)
theorem part_I : ∀ x : ℝ, (f 1 1 x).deriv > 0 ↔ x < 0 ∨ x > 1  ∧ 
                           (f 1 1 x).deriv < 0 ↔ 0 < x ∧ x < 1 := 
sorry

-- (II)
theorem part_II : (∀ x ≥ 0, f 0 b x ≥ 1) → 0 ≤ b ∧ b ≤ 1 := 
sorry

-- (III)
theorem part_III {a : ℝ} (ha : a > 0) (x₁ x₂ : ℝ) : 
  (f a 0 x₁).deriv = 0 → 
  (f a 0 x₂).deriv = 0 → 
  f a 0 x₁ + f a 0 x₂ < Real.exp 1 := 
sorry

end part_I_part_II_part_III_l161_161167


namespace floor_neg_seven_over_four_l161_161500

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l161_161500


namespace floor_of_neg_seven_fourths_l161_161513

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l161_161513


namespace train_crossing_time_l161_161839

/-- Prove the time it takes for a train of length 50 meters running at 60 km/hr to cross a pole is 3 seconds. -/
theorem train_crossing_time
  (speed_kmh : ℝ)
  (length_m : ℝ)
  (conversion_factor : ℝ)
  (time_seconds : ℝ) :
  speed_kmh = 60 →
  length_m = 50 →
  conversion_factor = 1000 / 3600 →
  time_seconds = 3 →
  time_seconds = length_m / (speed_kmh * conversion_factor) := 
by
  intros
  sorry

end train_crossing_time_l161_161839


namespace complete_subgraph_n_plus_one_l161_161629

open BigOperators

variable {V : Type}

/-- If a graph G has 2n-1 vertices and removing any vertex produces a subgraph
    containing a complete subgraph with n vertices, then G has a complete subgraph
    with n+1 vertices. -/
theorem complete_subgraph_n_plus_one (G : SimpleGraph V) (n : ℕ) (h_size : Fintype.card V = 2 * n - 1)
  (h_property : ∀ v ∈ G.verts, ∃ H : SimpleGraph V, H = G.delete v ∧ ∃ K : SimpleGraph V, K.is_clique n) :
  ∃ K : SimpleGraph V, K.is_clique (n + 1) :=
sorry

end complete_subgraph_n_plus_one_l161_161629


namespace integer_values_satisfying_square_root_condition_l161_161348

theorem integer_values_satisfying_square_root_condition :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 := sorry

end integer_values_satisfying_square_root_condition_l161_161348


namespace number_of_correct_conclusions_l161_161050

theorem number_of_correct_conclusions : 
  (1 : if ∀ x:ℝ, x > 0 → x > real.sin x) ∧ 
  (2 : (∀ x : ℝ, (x - real.sin x = 0) → x = 0) → (∀ x : ℝ, x ≠ 0 → x - real.sin x ≠ 0)) ∧ 
  (3 : ∀ p q : Prop, (p ∧ q) → (p ∨ q)) ∧ 
  (¬ ∀ x : ℝ, x > 0 → x - real.log x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - real.log x₀ ≤ 0) = 3 :=
sorry

end number_of_correct_conclusions_l161_161050


namespace number_of_negatives_is_3_l161_161977

-- Define the given list of numbers
def given_list : List ℚ := [-7, 0, -3, 4/3, 9100, -0.27]

-- Define the predicate that counts the number of negative numbers in the list
def count_negatives (lst : List ℚ) : ℕ :=
  lst.count (λ x, x < 0)

-- Statement of the problem: Prove that the number of negative numbers in the given list is 3
theorem number_of_negatives_is_3 : count_negatives given_list = 3 := by
  sorry

end number_of_negatives_is_3_l161_161977


namespace fixed_point_of_shifted_exponential_l161_161733

theorem fixed_point_of_shifted_exponential (a : ℝ) (H : a^0 = 1) : a^(3-3) + 3 = 4 :=
by
  sorry

end fixed_point_of_shifted_exponential_l161_161733


namespace part1_part2_l161_161121

open Real

-- Condition: tan(alpha) = 3
variable {α : ℝ} (h : tan α = 3)

-- Proof of first part
theorem part1 : (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 :=
by
  sorry

-- Proof of second part
theorem part2 : 1 - 4 * sin α * cos α + 2 * cos α ^ 2 = 0 :=
by
  sorry

end part1_part2_l161_161121


namespace min_books_borrowed_by_rest_l161_161212

theorem min_books_borrowed_by_rest (total_students : ℕ) (no_books_students : ℕ) (one_book_students : ℕ) (two_books_students : ℕ) (average_books_per_student : ℝ) :
  total_students = 38 →
  no_books_students = 2 →
  one_book_students = 12 →
  two_books_students = 10 →
  average_books_per_student = 2 →
  let students_borrowing_books := total_students - no_books_students in
  let accounted_students := one_book_students + two_books_students in
  let rest_students := students_borrowing_books - accounted_students in
  let total_books_borrowed := total_students * average_books_per_student in
  let accounted_books := (one_book_students * 1) + (two_books_students * 2) in
  let rest_books := total_books_borrowed - accounted_books in
  rest_students > 0 →
  rest_books / rest_students = 4 :=
begin
  -- Proof omitted
  sorry
end

end min_books_borrowed_by_rest_l161_161212


namespace find_dihedral_angle_find_distance_B_to_plane_CMN_l161_161228

-- Definitions for the geometrical conditions
structure EquilateralTriangle (A B C : Type) :=
(side_length : ℝ)
(equilateral : True)

structure PerpendicularPlanes (P Q : Type) :=
(perpendicular : True)

structure Midpoint (A B M : Type) :=
(midpoint : True)

-- Given conditions
def triangleABC : EquilateralTriangle := {
  side_length := 4,
  equilateral := sorry
}

def perpendicularPlanesSAC_ABC : PerpendicularPlanes := {
  perpendicular := sorry
}

def lengthsSA_SC : Prop := (2 * Real.sqrt 3 = 2 * Real.sqrt 3)

def midpointM : Midpoint := {
  midpoint := sorry
}

def midpointN : Midpoint := {
  midpoint := sorry
}

-- Prove the required measures
theorem find_dihedral_angle (t : EquilateralTriangle) (p : PerpendicularPlanes) (l : Prop) (m1 m2 : Midpoint) :
  ∠(N - C M - B) = Real.arctan (2 * Real.sqrt 2) :=
sorry

theorem find_distance_B_to_plane_CMN (t : EquilateralTriangle) (p : PerpendicularPlanes) (l : Prop) (m1 m2 : Midpoint) :
  distance B (plane CMN) = (4 * Real.sqrt 2) / 3 :=
sorry

end find_dihedral_angle_find_distance_B_to_plane_CMN_l161_161228


namespace number_of_sleeping_students_l161_161021

theorem number_of_sleeping_students 
  (hexagon : Type) [regular_hexagon hexagon] 
  (side_length : ℝ) (h : side_length = 3)
  (snore_meter_reading : hexagon → ℕ)
  (sum_readings : ∑ corner, snore_meter_reading corner = 7) 
  : ∃ n, n = 3 :=
by
  sorry

end number_of_sleeping_students_l161_161021


namespace sqrt_nonsimplest_l161_161790

def is_simplest_form (r: ℝ) : Prop :=
  ∀ (a b : ℝ), a * b = r → r = √a → (a = r ∨ b = 1)

theorem sqrt_nonsimplest (h1: is_simplest_form (√5))
                          (h2: is_simplest_form (√3))
                          (h3: is_simplest_form (√13))
                          : ¬ is_simplest_form (√0.3) :=
sorry

end sqrt_nonsimplest_l161_161790


namespace ratio_of_sides_l161_161036

theorem ratio_of_sides (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 * y = x) :
  x / y = real.sqrt 3 :=
sorry

end ratio_of_sides_l161_161036


namespace product_OB_OD_independent_of_angle_BAD_l161_161700

noncomputable def point_O_distance_to_AC (a b : ℝ) (h : b > a) (A C O : Type) [metricSpace O] [dist A O = b] [dist C O = b] := true

theorem product_OB_OD_independent_of_angle_BAD (a b : ℝ) (h : b > a) 
  (rhombus : Type) [metricSpace rhombus]
  (A B C D O : rhombus) [dist A O = b] [dist C O = b] [dist A B = a] [dist A D = a] :
  dist O B * dist O D = b^2 - a^2 :=
by
  sorry

end product_OB_OD_independent_of_angle_BAD_l161_161700


namespace solve_for_x_l161_161303

theorem solve_for_x (x : ℂ) (h : 5 - 3 * (complex.I) * x = 7 - (complex.I) * x) : x = -complex.I :=
by
  sorry

end solve_for_x_l161_161303


namespace unique_function_solution_l161_161526

theorem unique_function_solution (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, x ≥ 1 → f x ≥ 1)
  (h₂ : ∀ x : ℝ, x ≥ 1 → f x ≤ 2 * (x + 1))
  (h₃ : ∀ x : ℝ, x ≥ 1 → f (x + 1) = (f x)^2/x - 1/x) :
  ∀ x : ℝ, x ≥ 1 → f x = x + 1 :=
by
  intro x hx
  sorry

end unique_function_solution_l161_161526


namespace two_pow_m_plus_2n_is_12_l161_161614

theorem two_pow_m_plus_2n_is_12 (m n : ℤ) (h1 : 2^m = 3) (h2 : 2^n = 2) : 2^(m + 2 * n) = 12 :=
  sorry

end two_pow_m_plus_2n_is_12_l161_161614


namespace heels_cost_correct_l161_161697

variable (initial_amount : ℕ) (remaining_amount : ℕ) (jumper_cost : ℕ) (tshirt_cost : ℕ) (heels_cost : ℕ)

noncomputable def total_spent : ℕ := initial_amount - remaining_amount
noncomputable def known_spent : ℕ := jumper_cost + tshirt_cost
noncomputable def heels_cost_calculated : ℕ := total_spent - known_spent

theorem heels_cost_correct :
  initial_amount = 26 →
  remaining_amount = 8 →
  jumper_cost = 9 →
  tshirt_cost = 4 →
  heels_cost_calculated = 5 := sorry

end heels_cost_correct_l161_161697


namespace number_of_paths_from_01_to_20_is_4_l161_161264

-- Definitions of lattice points and segments
def Point := (ℕ × ℕ)

def points_within_rectangle : set Point :=
  {(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)}

def segments : set (Point × Point) :=
  {((0,1), (1,1)),
   ((1,1), (2,1)),
   ((2,1), (2,0)),
   ((2,0), (1,0)),
   ((1,0), (0,0)),
   ((0,0), (0,1)),
   ((1,1), (1,0))}

-- The statement to prove
theorem number_of_paths_from_01_to_20_is_4 :
  ∃ (P : list (list (Point × Point))), 
    (∀ p ∈ P, p.head? = some ((0, 1), (1, 1)) ∨ p.head? = some ((0, 1), (0, 0)))
    ∧ (∀ p ∈ P, p.last? = some ((1, 0), (2, 0)) ∨ p.last? = some ((2, 1), (2, 0)))
    ∧ (∀ p ∈ P, ∀ s ∈ p, s ∈ segments)
    ∧ (|P| = 4) :=
sorry

end number_of_paths_from_01_to_20_is_4_l161_161264


namespace decreased_by_2_and_divided_by_13_l161_161621

noncomputable def number := 54

theorem decreased_by_2_and_divided_by_13 (x : ℕ) (h : (x - 5) / 7 = 7) : (x - 2) / 13 = 4 := 
by
  have hx : x = 54 := by sorry
  rw [hx]
  calc
  (54 - 2) / 13 = 4 : by norm_num
  
#eval @decreased_by_2_and_divided_by_13 number sorry

end decreased_by_2_and_divided_by_13_l161_161621


namespace difference_in_average_speed_l161_161756

-- Define variables and conditions
def car_distance : ℕ := 600
def car_R_speed : ℕ := 50
def car_R_time := car_distance / car_R_speed
def car_P_time := car_R_time - 2
def car_P_speed := car_distance / car_P_time

-- Define and prove the theorem
theorem difference_in_average_speed : car_P_speed - car_R_speed = 10 :=
by
  -- Statement of the theorem based on conditions
  have h1: car_R_time = 12 := by sorry
  have h2: car_P_time = 10 := by sorry
  have h3: car_P_speed = 60 := by sorry
  calc
    car_P_speed - car_R_speed = 60 - 50 : by 
      -- Simplify using previous results
      rw [←h3, ←h2, ←h1]
      -- Calculation details omitted
      sorry
    ... = 10 : by sorry

end difference_in_average_speed_l161_161756


namespace total_highlighters_l161_161983

def num_pink_highlighters := 9
def num_yellow_highlighters := 8
def num_blue_highlighters := 5

theorem total_highlighters : 
  num_pink_highlighters + num_yellow_highlighters + num_blue_highlighters = 22 :=
by
  sorry

end total_highlighters_l161_161983


namespace quadratic_solutions_l161_161752

theorem quadratic_solutions : ∀ x : ℝ, x^2 - 25 = 0 → (x = 5 ∨ x = -5) :=
by
  sorry

end quadratic_solutions_l161_161752


namespace lottery_numbers_bound_l161_161306

theorem lottery_numbers_bound (s : ℕ) (k : ℕ) (num_tickets : ℕ) (num_numbers : ℕ) (nums_per_ticket : ℕ)
  (h_tickets : num_tickets = 100) (h_numbers : num_numbers = 90) (h_nums_per_ticket : nums_per_ticket = 5)
  (h_s : s = num_tickets) (h_k : k = 49) :
  ∃ n : ℕ, n ≤ 10 :=
by
  sorry

end lottery_numbers_bound_l161_161306


namespace floor_neg_seven_fourths_l161_161474

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l161_161474


namespace floor_neg_seven_over_four_l161_161503

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l161_161503


namespace count_satisfying_integers_l161_161083

theorem count_satisfying_integers :
  let s := { x : ℤ | (-4 : ℤ) * x ≥ 2 * x + 16 ∧ (-3 : ℤ) * x ≤ 15 ∧ (5 : ℤ) * x ≤ x - 10 } in
  fintype.card s = 3 :=
by
  let s := { x : ℤ | (-4 : ℤ) * x ≥ 2 * x + 16 ∧ (-3 : ℤ) * x ≤ 15 ∧ (5 : ℤ) * x ≤ x - 10 }
  have : ∀ x ∈ s, x ∈ ({-5, -4, -3} : set ℤ), from sorry
  exact sorry

end count_satisfying_integers_l161_161083


namespace infinitenat_not_sum_square_prime_l161_161284

theorem infinitenat_not_sum_square_prime : ∀ k : ℕ, ¬ ∃ (n : ℕ) (p : ℕ), Prime p ∧ (3 * k + 2) ^ 2 = n ^ 2 + p :=
by
  intro k
  sorry

end infinitenat_not_sum_square_prime_l161_161284


namespace g_50_is_0_l161_161664

def phi (n : ℕ) : ℕ :=
  if n = 0 then 0 else (Finset.range n).filter (Nat.coprime n).card

def g (n : ℕ) : ℕ :=
  if n = 0 then 0 else sorry -- The exact definition will be inferred from the sums

theorem g_50_is_0 : g 50 = 0 :=
by
  have h : (Finset.sum (Finset.divisors 50) g) = phi 50
  { sorry },
  have h1 : (Finset.sum (Finset.divisors 50) g) = g 1 + g 2 + g 5 + g 10 + g 25 + g 50
  { sorry },
  have h2 : (g 1 = 1) ∧ (g 2 = 0) ∧ (g 5 = 3) ∧ (g 10 = 0) ∧ (g 25 = 16) ∧ (phi 50 = 20)
  { sorry },
  linarith

end g_50_is_0_l161_161664


namespace purchase_price_of_first_and_second_batches_total_profit_l161_161415

-- Definitions for the conditions:
def first_purchase_cost : ℝ := 40000
def second_purchase_cost : ℝ := 88000
def second_batch_multiplier : ℝ := 2
def price_increase : ℝ := 4
def selling_price : ℝ := 56
def discount_percentage : ℝ := 0.20
def remaining_units : ℝ := 150

-- Definition of the purchase prices and profit:
def purchase_prices (x : ℝ) (y : ℝ) : Prop :=
  first_purchase_cost / x * second_batch_multiplier = second_purchase_cost / y ∧
  y = x + price_increase

def profit (x y : ℝ) : ℝ :=
  (selling_price - x) * (first_purchase_cost / x) +
  (selling_price - y) * (second_purchase_cost / y - remaining_units) +
  remaining_units * (selling_price * (1 - discount_percentage) - y)

-- Theorem statements:
theorem purchase_price_of_first_and_second_batches :
  ∃ x y, purchase_prices x y ∧ x = 40 ∧ y = 44 :=
by
  sorry

theorem total_profit :
  ∃ x y, purchase_prices x y ∧ profit x y = 38320 :=
by
  sorry

end purchase_price_of_first_and_second_batches_total_profit_l161_161415


namespace divisibility_of_n_l161_161006

theorem divisibility_of_n (P : Polynomial ℤ) (k n : ℕ)
  (hk : k % 2 = 0)
  (h_odd_coeffs : ∀ i, i ≤ k → i % 2 = 1)
  (h_div : ∃ Q : Polynomial ℤ, (X + 1)^n - 1 = (P * Q)) :
  n % (k + 1) = 0 :=
sorry

end divisibility_of_n_l161_161006


namespace exists_color_with_non_isosceles_triangles_l161_161453

theorem exists_color_with_non_isosceles_triangles :
  ∀ (points : Fin 50 → Fin 4 → Prop),
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i → ¬ collinear (points i) (points j) (points k)) →
  ∃ c : Fin 4, ∃ S : Finset (Fin 50), S.card ≥ 13 ∧
  ∃ t : Finset ({s : Finset (Fin 50) // s.card = 3}),
    t.card ≥ 130 ∧ ∀ s ∈ t, ¬ is_isosceles_triangle ((λ x, points x) '' s) :=
sorry

end exists_color_with_non_isosceles_triangles_l161_161453


namespace arithmetic_sequence_l161_161169

theorem arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n + 1) : 
  ∀ n, a (n + 1) - a n = 3 := by
  sorry

end arithmetic_sequence_l161_161169


namespace eve_discovers_secret_l161_161046

theorem eve_discovers_secret (x : ℕ) : ∃ (n : ℕ), ∃ (is_prime : ℕ → Prop), (∀ m : ℕ, (is_prime (x + n * m)) ∨ (¬is_prime (x + n * m))) :=
  sorry

end eve_discovers_secret_l161_161046


namespace monotonicity_and_range_of_m_l161_161945

noncomputable def f (m x : ℝ) : ℝ := exp (m * x) + x^2 - m * x

theorem monotonicity_and_range_of_m {m : ℝ} :
  (∀ x1 x2 : ℝ, x1 ∈ Icc (-1 : ℝ) (1 : ℝ) → x2 ∈ Icc (-1 : ℝ) (1 : ℝ) →
    f m x1 - f m x2 ≤ exp 1 - 1) ↔ (m ∈ Icc (-1 : ℝ) 1) :=
  sorry

end monotonicity_and_range_of_m_l161_161945


namespace integer_points_on_hyperbola_l161_161605

theorem integer_points_on_hyperbola : 
  let points := {(x, y) : Int × Int | y * x = 2013} in points.size = 16 :=
by
  sorry

end integer_points_on_hyperbola_l161_161605


namespace function_symmetric_and_monotonic_l161_161582

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^4 - 2 * Real.sin x * Real.cos x - (Real.sin x)^4

theorem function_symmetric_and_monotonic :
  (∀ x, f (x + (3/8) * π) = f (x - (3/8) * π)) ∧
  (∀ x y, x ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → y ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → x < y → f x > f y) :=
by
  sorry

end function_symmetric_and_monotonic_l161_161582


namespace sum_of_valid_m_integers_l161_161580

theorem sum_of_valid_m_integers :
  ∀ (m x y : ℝ),
  (x = 2 - 2 * m ∧ x ≤ 6 ∧ x ≠ 2 ∧ x ≠ -2) ∧ 
  (m - 6 * y > 2 ∧ y - 4 ≤ 3 * y + 4) →
  ( ( ∃ y1 y2 y3 y4 : ℝ,
    -4 ≤ y1 ∧ y1 < (m - 2) / 6 ∧
    -4 ≤ y2 ∧ y2 < (m - 2) / 6 ∧
    -4 ≤ y3 ∧ y3 < (m - 2) / 6 ∧
    -4 ≤ y4 ∧ y4 < (m - 2) / 6 ∧
    y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4 ∧ 
    y1.floor = y1 ∧ y2.floor = y2 ∧ y3.floor = y3 ∧ y4.floor = y4 )) →
  sum (filter (λ m, m ∉ {0,2}) (nat.filter (λ m, -2 ≤ m ∧ m < 2))) = -2 :=
sorry

end sum_of_valid_m_integers_l161_161580


namespace simplify_cubed_root_l161_161289

theorem simplify_cubed_root : (∛(2^9 * 3^3 * 5^3 * 11^3) = 1320) := by
  sorry

end simplify_cubed_root_l161_161289


namespace total_spent_at_music_store_l161_161080

-- Defining the costs
def clarinet_cost : ℝ := 130.30
def song_book_cost : ℝ := 11.24

-- The main theorem to prove
theorem total_spent_at_music_store : clarinet_cost + song_book_cost = 141.54 :=
by
  sorry

end total_spent_at_music_store_l161_161080


namespace compare_abc_l161_161149

noncomputable section

def a : ℝ := (1 / 2) ^ (1 / 2)
def b : ℝ := Real.log 2 (1 / 3)
def c : ℝ := Real.log 2 3

theorem compare_abc :
  c > a ∧ a > b :=
by
  sorry

end compare_abc_l161_161149


namespace common_tangent_theorem_l161_161597

-- Define the first circle with given equation (x+2)^2 + (y-2)^2 = 1
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 1

-- Define the second circle with given equation (x-2)^2 + (y-5)^2 = 16
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

-- Define a predicate that expresses the concept of common tangents between two circles
def common_tangents_count (circle1 circle2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The statement to prove that the number of common tangents is 3
theorem common_tangent_theorem : common_tangents_count circle1 circle2 = 3 :=
by
  -- We would proceed with the proof if required, but we end with sorry as requested.
  sorry

end common_tangent_theorem_l161_161597


namespace eval_floor_neg_seven_fourths_l161_161510

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l161_161510


namespace constants_unique_l161_161893

theorem constants_unique (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → (5 * x) / ((x - 4) * (x - 2) ^ 2) = A / (x - 4) + B / (x - 2) + C / (x - 2) ^ 2) ↔
  A = 5 ∧ B = -5 ∧ C = -5 :=
by
  sorry

end constants_unique_l161_161893


namespace bread_cost_l161_161312

theorem bread_cost {packs_meat packs_cheese sandwiches : ℕ} 
  (cost_meat cost_cheese cost_sandwich coupon_meat coupon_cheese total_cost : ℝ) 
  (h_meat_cost : cost_meat = 5.00) 
  (h_cheese_cost : cost_cheese = 4.00)
  (h_coupon_meat : coupon_meat = 1.00)
  (h_coupon_cheese : coupon_cheese = 1.00)
  (h_cost_sandwich : cost_sandwich = 2.00)
  (h_packs_meat : packs_meat = 2)
  (h_packs_cheese : packs_cheese = 2)
  (h_sandwiches : sandwiches = 10)
  (h_total_revenue : total_cost = sandwiches * cost_sandwich) :
  ∃ (bread_cost : ℝ), bread_cost = total_cost - ((packs_meat * cost_meat - coupon_meat) + (packs_cheese * cost_cheese - coupon_cheese)) :=
sorry

end bread_cost_l161_161312


namespace hyperbola_integer_points_count_l161_161602

-- Definition of the hyperbolic equation
def hyperbola (x y : ℤ) : Prop :=
  y * x = 2013

-- Condition: We are looking for integer coordinate points (x, y)
def integer_coordinate_points : Set (ℤ × ℤ) :=
  {p | hyperbola p.fst p.snd}

-- Main proof statement
theorem hyperbola_integer_points_count : (integer_coordinate_points.to_finset.card = 16) :=
sorry

end hyperbola_integer_points_count_l161_161602


namespace center_on_line_AM_l161_161231

noncomputable def center_of_circumcircle (B C M : Point) : Point := sorry

-- Define the conditions
variables (M A B C O : Point)
variable (r : ℝ)
variable [order : is_ordered_ring ℝ]

-- The angle of incidence equals the angle of reflection
axiom incidence_reflection :
  ∀ (B C M A : Point), rfl -- Formal version to be written appropriately

-- The proof statement
theorem center_on_line_AM 
  (h_circumcircle : O = center_of_circumcircle B C M)
  (h_incidence_reflection : incidence_reflection B C M A) :
  lies_on_line O (line_through A M) :=
sorry

end center_on_line_AM_l161_161231


namespace inequality_pgcd_l161_161676

theorem inequality_pgcd (a b : ℕ) (h1 : a > b) (h2 : (a - b) ∣ (a^2 + b)) : 
  (a + 1) / (b + 1) ≤ Nat.gcd a b + 1 := 
sorry

end inequality_pgcd_l161_161676


namespace area_enclosed_by_circle_below_line_eq_l161_161379

theorem area_enclosed_by_circle_below_line_eq 
  (x y : ℝ) 
  (circle_eq : (x - 5)^2 + (y - 4)^2 = 16) 
  (line_eq : y = x - 1) 
  : area_below_line (x y : ℝ) 
    ((x - 5)^2 + (y - 4)^2 = 16) 
    (y = x - 1) 
    = (32 * real.pi) / 3 := 
by sorry

end area_enclosed_by_circle_below_line_eq_l161_161379


namespace floor_neg_seven_fourths_l161_161473

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l161_161473


namespace circle_equation_exists_shortest_chord_line_l161_161551

-- Condition 1: Points A and B
def point_A : (ℝ × ℝ) := (1, -2)
def point_B : (ℝ × ℝ) := (-1, 0)

-- Condition 2: Circle passes through A and B and sum of intercepts is 2
def passes_through (x y : ℝ) (D E F : ℝ) : Prop := 
  (x^2 + y^2 + D * x + E * y + F = 0)

def satisfies_intercepts (D E : ℝ) : Prop := (-D - E = 2)

-- Prove
theorem circle_equation_exists : 
  ∃ D E F, passes_through 1 (-2) D E F ∧ passes_through (-1) 0 D E F ∧ satisfies_intercepts D E :=
sorry

-- Given that P(2, 0.5) is inside the circle from above theorem
def point_P : (ℝ × ℝ) := (2, 0.5)

-- Prove the equation of the shortest chord line l
theorem shortest_chord_line :
  ∃ m b, m = -2 ∧ point_P.2 = m * (point_P.1 - 2) + b ∧ (∀ (x y : ℝ), 4 * x + 2 * y - 9 = 0) :=
sorry

end circle_equation_exists_shortest_chord_line_l161_161551


namespace BE_CE_lt_AD_l161_161279

theorem BE_CE_lt_AD 
    (A B C D E : Point) 
    (angles_equal : ∀ P Q R, ∡ P Q R = ∡ Q R P) 
    (acute_angle_B : ∀ Q R, ∡ B Q R < π / 2)
    (E_on_AD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (E = t • A + (1 - t) • D))
    (alpha : ℝ) 
    (h1 : ∡ CAD = alpha) 
    (h2 : ∡ ADC = alpha)
    (h3 : ∡ ABE = alpha)
    (h4 : ∡ DBE = alpha)
    : dist B E + dist C E < dist A D :=
sorry

end BE_CE_lt_AD_l161_161279


namespace false_statement_l161_161689

-- Definitions of sequence and sum
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

-- Given conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable h_arith : is_arithmetic_sequence a
variable h_sum_def : ∀ (n : ℕ), S n = sum_of_sequence a n
variable h_S2023 : S 2023 = 2023

-- Statements to check
def A : Prop := a 1012 = 1
def B : Prop := a 1013 ≥ 1
def C : Prop := S 2022 > 2022
def D : Prop := S 2024 ≥ 2024

-- Proof statement
theorem false_statement : ¬C :=
by
  -- the proof would go here
  sorry

end false_statement_l161_161689


namespace sin_alpha_beta_gamma_values_l161_161941

open Real

theorem sin_alpha_beta_gamma_values (α β γ : ℝ)
  (h1 : sin α = sin (α + β + γ) + 1)
  (h2 : sin β = 3 * sin (α + β + γ) + 2)
  (h3 : sin γ = 5 * sin (α + β + γ) + 3) :
  sin α * sin β * sin γ = (3/64) ∨ sin α * sin β * sin γ = (1/8) :=
sorry

end sin_alpha_beta_gamma_values_l161_161941


namespace minimum_distance_at_meeting_time_distance_glafira_to_meeting_l161_161001

variables (U g τ V : ℝ)
-- assumption: 2 * U ≥ g * τ
axiom h : 2 * U ≥ g * τ

noncomputable def motion_eq1 (t : ℝ) : ℝ := U * t - (g * t^2) / 2
noncomputable def motion_eq2 (t : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2

noncomputable def distance (t : ℝ) : ℝ := 
|motion_eq1 U g t - motion_eq2 U g τ t|

noncomputable def meeting_time : ℝ := (2 * U / g) + (τ / 2)

theorem minimum_distance_at_meeting_time : distance U g τ meeting_time = 0 := sorry

noncomputable def distance_from_glafira_to_meeting : ℝ := 
V * meeting_time

theorem distance_glafira_to_meeting : 
distance_from_glafira_to_meeting U g τ V = V * ((τ / 2) + (U / g)) := sorry

end minimum_distance_at_meeting_time_distance_glafira_to_meeting_l161_161001


namespace euler_inverse_relation_l161_161190

noncomputable def euler_relation (γ δ : ℝ) : Prop :=
  complex.exp (complex.I * γ) + complex.exp (complex.I * δ) = -5 / 8 + 9 / 10 * complex.I

theorem euler_inverse_relation (γ δ : ℝ) :
  euler_relation γ δ →
  complex.exp (-complex.I * γ) + complex.exp (-complex.I * δ) = -5 / 8 - 9 / 10 * complex.I :=
by
  intro h,
  sorry

end euler_inverse_relation_l161_161190


namespace angle_ratio_l161_161644

theorem angle_ratio (x : ℝ) (h1 : 3 * x = ∠ABC) (h2 : BM bisects ABP)
  (h3 : BP trisects ABC) (h4 : BQ trisects ABC) : 
  ∠MBQ / ∠ABQ = 3 / 4 :=
by
  sorry

end angle_ratio_l161_161644


namespace paths_from_A_to_D_l161_161116

theorem paths_from_A_to_D : 
    let paths_A_B := 2 in
    let paths_B_C := 2 in
    let paths_C_D := 2 in
    let direct_path_A_D := 1 in
    paths_A_B * paths_B_C * paths_C_D + direct_path_A_D = 9 :=
by
  sorry

end paths_from_A_to_D_l161_161116


namespace cost_price_is_correct_l161_161424

-- Define the conditions
def purchasing_clocks : ℕ := 150
def gain_60_clocks : ℝ := 0.12
def gain_90_clocks : ℝ := 0.18
def uniform_profit : ℝ := 0.16
def difference_in_profit : ℝ := 75

-- Define the cost price of each clock
noncomputable def C : ℝ := 125

-- Define and state the theorem
theorem cost_price_is_correct (C : ℝ) :
  (60 * C * (1 + gain_60_clocks) + 90 * C * (1 + gain_90_clocks)) - (150 * C * (1 + uniform_profit)) = difference_in_profit :=
sorry

end cost_price_is_correct_l161_161424


namespace find_a_b_odd_function_f_increasing_solve_inequality_l161_161944

-- Definitions from the conditions
def f (x : ℝ) (a b : ℝ) : ℝ := (a * 2^x + b + 1) / (2^x + 1)
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)
def increasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≤ f y

-- Main Lean 4 statements derived from the questions and conditions
theorem find_a_b_odd_function (a b : ℝ) (h_odd : is_odd (f a b)) (h_f2 : f 2 a b = 6 / 5) : a = 2 ∧ b = -3 :=
sorry

theorem f_increasing (a b : ℝ) (h_odd : is_odd (f a b)) (h_f2 : f 2 a b = 6 / 5): increasing (f a b) :=
sorry

theorem solve_inequality (a b : ℝ) (h_odd : is_odd (f a b)) (h_f2 : f 2 a b = 6 / 5) : 
  ∀ x : ℝ, 1 < x → x ≤ 6 / 5 → f (log (2 * x - 2) / log (1 / 2)) a b + f (log (1 - x / 2) / log 2) a b ≥ 0 :=
sorry

end find_a_b_odd_function_f_increasing_solve_inequality_l161_161944


namespace beetles_eaten_per_day_l161_161464
-- Import the Mathlib library

-- Declare the conditions as constants
def bird_eats_beetles_per_day : Nat := 12
def snake_eats_birds_per_day : Nat := 3
def jaguar_eats_snakes_per_day : Nat := 5
def number_of_jaguars : Nat := 6

-- Define the theorem and provide the expected proof
theorem beetles_eaten_per_day :
  12 * (3 * (5 * 6)) = 1080 := by
  sorry

end beetles_eaten_per_day_l161_161464


namespace prove_a_le_1_l161_161588

noncomputable def problem_statement (a : ℝ) : Prop :=
  ¬(∃ x ∈ set.Icc (1 : ℝ) 2, x ^ 2 - a < 0) → a ≤ 1

-- Statement for the theorem to be proven
theorem prove_a_le_1 (a : ℝ) : problem_statement a :=
sorry

end prove_a_le_1_l161_161588


namespace num_good_colorings_l161_161816

theorem num_good_colorings (n : ℕ) (h : n ≥ 4) :
  let colorings := { f : Fin n → Bool // -- Each vertex is colored either Black(0) or White(1)
    (∃ i1 i2, i1 ≠ i2 ∧ f i1 ≠ f i2) -- Precondition: Not all vertices are colored the same
    ∧ ∃ diagonals : List (Fin n × Fin n), 
      (∀ d, d ∈ diagonals → (f d.1 ≠ f d.2)) -- Each diagonal is multicolored
      ∧ -- Diagonals divide the n-gon into triangles without sharing points except vertices } 
  coloring /- Number of such valid n-gon colorings -/ := { f : Fin n → Bool | -- Each vertex is colored either Black(0) or White(1)
    (∃ i1 i2, i1 ≠ i2 ∧ f i1 ≠ f i2) -- Precondition: Not all vertices are colored the same
    ∧ ∃ diagonals : List (Fin n × Fin n), 
      (∀ d, d ∈ diagonals → (f d.1 ≠ f d.2)) -- Each diagonal is multicolored
      ∧ -- Diagonals divide the n-gon into triangles without sharing points except vertices }.card = n * (n - 1) :=
begin
  sorry,
end

end num_good_colorings_l161_161816


namespace compare_a_b_c_l161_161550

def a := Real.exp (-0.02)
def b := 0.01
def c := Real.log 1.01

theorem compare_a_b_c : a > b ∧ b > c := by
  sorry

end compare_a_b_c_l161_161550


namespace train_crosses_pole_in_3_seconds_l161_161835

def train_problem (speed_kmh : ℕ) (length_m : ℕ) : ℕ :=
  let speed_ms := (speed_kmh * 1000) / 3600 in
  length_m / speed_ms

theorem train_crosses_pole_in_3_seconds :
  train_problem 60 50 = 3 :=
by
  -- We add a 'sorry' to skip the proof
  sorry

end train_crosses_pole_in_3_seconds_l161_161835


namespace complex_value_of_z_l161_161613

theorem complex_value_of_z (z : ℂ) : (z - 1)^2 = -1 ↔ (z = 1 + complex.i ∨ z = 1 - complex.i) :=
by
  sorry

end complex_value_of_z_l161_161613


namespace constant_term_binomial_expansion_l161_161720

/--
 If the constant term in the expansion of (a * x^3 + 1 / sqrt x)^7 is 14, then a = 2.
-/
theorem constant_term_binomial_expansion (a : ℝ) 
  (h : (∃ (T₇ : ℝ), T₇ = (a^1 * (Nat.choose 7 6)) ∧ T₇ = 14)) : 
  a = 2 :=
by
  sorry

end constant_term_binomial_expansion_l161_161720


namespace simplify_cos_difference_l161_161299

noncomputable def cos (x : ℝ) : ℝ := real.cos x

def c := cos (20 * real.pi / 180)  -- cos(20°)
def d := cos (40 * real.pi / 180)  -- cos(40°)

theorem simplify_cos_difference :
  c - d =
  -- The expression below is placeholder; real expression involves radicals and squares
  sorry :=
by
  let c := cos (20 * real.pi / 180)
  let d := cos (40 * real.pi / 180)
  have h1 : d = 2 * c^2 - 1 := sorry
  let sqrt3 : ℝ := real.sqrt 3
  have h2 : c = (1 / 2) * d + (sqrt3 / 2) * real.sqrt (1 - d^2) := sorry
  sorry

end simplify_cos_difference_l161_161299


namespace max_value_of_a_l161_161025

theorem max_value_of_a :
  ∀ (a : ℚ),
  (∀ (m : ℚ), 1/3 < m ∧ m < a →
   (∀ (x : ℤ), 0 < x ∧ x ≤ 200 →
    ¬ (∃ (y : ℤ), y = m * x + 3 ∨ y = m * x + 1))) →
  a = 68/201 :=
by
  sorry

end max_value_of_a_l161_161025


namespace solve_real_equation_l161_161529

theorem solve_real_equation (x : ℝ) :
  x^2 * (x + 1)^2 + x^2 = 3 * (x + 1)^2 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 :=
by sorry

end solve_real_equation_l161_161529


namespace ratio_height_to_base_square_to_rectangle_l161_161999

theorem ratio_height_to_base_square_to_rectangle 
  (side : ℝ)
  (h_side : side = 4)
  (E F : ℝ)
  (h_midpoints : E = side / 2 ∧ F = side / 2)
  (AG BF : ℝ)
  (h_perpendicular : (AG ^ 2 + BF ^ 2) = side ^ 2 + (side / 2) ^ 2)
  (area_square area_rectangle : ℝ)
  (h_area : area_square = side ^ 2 ∧ area_rectangle = area_square)
  (height base : ℝ)
  (h_base : base = side / (real.sqrt 5))
  (h_height : height = area_rectangle / base) :
  (height / base) = 5 := 
sorry

end ratio_height_to_base_square_to_rectangle_l161_161999


namespace volume_of_cuboid_l161_161332

-- Defining the conditions
def point (A B C : ℝ) : Prop := 
  -- Placeholder for actual definition of the geometric setup
  sorry

def lengths (a b c : ℝ) : Prop := 
  a = 4 ∧ b = 5 ∧ c = 6

-- Defining the sought volume
def volume : ℝ := 90 * real.sqrt 6

-- The theorem to prove
theorem volume_of_cuboid :
  ∀ (A B C a b c : ℝ), point A B C → lengths a b c → 
  Mathlib.volume = 90 * real.sqrt 6 :=
by
  intros
  sorry

end volume_of_cuboid_l161_161332


namespace chi_squared_confidence_level_l161_161200

theorem chi_squared_confidence_level 
  (chi_squared_value : ℝ)
  (p_value_3841 : ℝ)
  (p_value_5024 : ℝ)
  (h1 : chi_squared_value = 4.073)
  (h2 : p_value_3841 = 0.05)
  (h3 : p_value_5024 = 0.025)
  (h4 : 3.841 ≤ chi_squared_value ∧ chi_squared_value < 5.024) :
  ∃ confidence_level : ℝ, confidence_level = 0.95 :=
by 
  sorry

end chi_squared_confidence_level_l161_161200


namespace puzzle_percentage_increase_l161_161657

theorem puzzle_percentage_increase:
  ∃ P : ℝ, (
    let x := 1000 + (P / 100) * 1000 in
    1000 + 2 * x = 4000
  ) ∧ P = 50 :=
begin
  -- This is where the proof would go
  sorry,
end

end puzzle_percentage_increase_l161_161657


namespace values_of_x_minus_y_l161_161126

theorem values_of_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : y > x) : x - y = -2 ∨ x - y = -8 :=
  sorry

end values_of_x_minus_y_l161_161126


namespace sum_of_squares_of_projections_constant_l161_161755

-- Define the sum of the squares of projections function
noncomputable def sum_of_squares_of_projections (a : ℝ) (α : ℝ) : ℝ :=
  let p1 := a * Real.cos α
  let p2 := a * Real.cos (Real.pi / 3 - α)
  let p3 := a * Real.cos (Real.pi / 3 + α)
  p1^2 + p2^2 + p3^2

-- Statement of the theorem
theorem sum_of_squares_of_projections_constant (a α : ℝ) : 
  sum_of_squares_of_projections a α = 3 / 2 * a^2 :=
sorry

end sum_of_squares_of_projections_constant_l161_161755


namespace coefficient_of_x_is_63_over_16_l161_161924

noncomputable def a := ∫ x in 0..(Real.pi / 2), (1 / 2 - (Real.sin (x / 2)) ^ 2)

theorem coefficient_of_x_is_63_over_16 :
  let expansion_coefficient := (1/2) ^ 5 * Nat.choose 9 4 * (1 : ℝ) 
  expansion_coefficient = 63 / 16 :=
by
  let expansion_coefficient := (1/2) ^ 5 * Nat.choose 9 4 * (1 : ℝ) 
  sorry

end coefficient_of_x_is_63_over_16_l161_161924


namespace sum_of_die_rolls_is_odd_probability_l161_161368

/-- Three fair coins are tossed once. For every head that appears, one fair die is rolled.
What is the probability that the sum of the die rolls is odd? (Note that if no die is rolled,
the sum is 0.) -/
theorem sum_of_die_rolls_is_odd_probability :
  let fair_coin_outcomes := {0, 1} -- 0 represents tail, 1 represents head
      fair_die_outcomes := {1, 2, 3, 4, 5, 6}
      coins_tossed := {coins ∈ list (list nat) | length coins = 3 ∧ ∀ c ∈ coins, c ∈ fair_coin_outcomes}
      roll_die n := {rolls ∈ list (list nat) | ∀ roll ∈ rolls, roll ∈ fair_die_outcomes ∧ length rolls = n}
      sum_is_odd sums := {s ∈ sums | s % 2 = 1}
      probability event outcomes := (fintype.card event).to_real / (fintype.card outcomes).to_real
  in probability (sum_is_odd (coins_tossed.bind (λ coins, match coins with
    | [0, 0, 0] := [0] -- no dice rolled, sum = 0
    | [1, 1, 1] := roll_die 3.sum
    | [1, 1, 0] := roll_die 2.sum
    | [1, 0, 0] := roll_die 1.sum
    | _ := [] -- no other cases due to conditional on coin outcomes
  end))) (coins_tossed.bind (λ coins, match coins with
    | [0, 0, 0] := [0]
    | _ := roll_die (coins.sum)
  end))) = 7 / 16 := sorry

end sum_of_die_rolls_is_odd_probability_l161_161368


namespace fraction_simplification_l161_161779

theorem fraction_simplification :
  10 * (1/2 + 1/5 + 1/10)⁻¹ = 25 / 2 :=
by
  sorry

end fraction_simplification_l161_161779


namespace simplify_cos_difference_l161_161300

noncomputable def cos (x : ℝ) : ℝ := real.cos x

def c := cos (20 * real.pi / 180)  -- cos(20°)
def d := cos (40 * real.pi / 180)  -- cos(40°)

theorem simplify_cos_difference :
  c - d =
  -- The expression below is placeholder; real expression involves radicals and squares
  sorry :=
by
  let c := cos (20 * real.pi / 180)
  let d := cos (40 * real.pi / 180)
  have h1 : d = 2 * c^2 - 1 := sorry
  let sqrt3 : ℝ := real.sqrt 3
  have h2 : c = (1 / 2) * d + (sqrt3 / 2) * real.sqrt (1 - d^2) := sorry
  sorry

end simplify_cos_difference_l161_161300


namespace set_complement_intersection_l161_161955

open Set

variable (U M N : Set ℕ)

theorem set_complement_intersection :
  U = {1, 2, 3, 4, 5, 6, 7} →
  M = {3, 4, 5} →
  N = {1, 3, 6} →
  {2, 7} = (U \ M) ∩ (U \ N) :=
by
  intros hU hM hN
  rw [hU, hM, hN]
  sorry

end set_complement_intersection_l161_161955


namespace cannot_empty_both_piles_l161_161699

theorem cannot_empty_both_piles
  (A B : ℕ)
  (initial_A : A = 1)
  (initial_B : B = 0)
  (step : ∀ (n : ℕ), n = (A - 1 + (B + 3)) ∨ n = ((A + 3) - 1 + B) ∨ n = (A - 4 + B) ∨ n = (A + B - 4)) :
  ¬ (A = 0 ∧ B = 0) :=
begin
  sorry
end

end cannot_empty_both_piles_l161_161699


namespace radius_difference_approx_l161_161748

theorem radius_difference_approx {r R : ℝ} (h : (π * R^2) / (π * r^2) = 5 / 2) : R - r ≈ 0.58 * r :=
by
  sorry

end radius_difference_approx_l161_161748


namespace combined_seq_20th_term_l161_161342

def arithmetic_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d
def geometric_seq (g : ℕ) (r : ℕ) (n : ℕ) : ℕ := g * r^(n - 1)

theorem combined_seq_20th_term :
  let a := 3
  let d := 4
  let g := 2
  let r := 2
  let n := 20
  arithmetic_seq a d n + geometric_seq g r n = 1048655 :=
by 
  sorry

end combined_seq_20th_term_l161_161342


namespace maximum_value_of_y_l161_161164

def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

def y (x : ℝ) : ℝ := (f x) ^ 2 + f (x ^ 2)

theorem maximum_value_of_y : ∃ x ∈ set.Icc 1 9, y x = 13 :=
by
  sorry

end maximum_value_of_y_l161_161164


namespace nine_a_eq_frac_minus_eighty_one_over_eleven_l161_161163

theorem nine_a_eq_frac_minus_eighty_one_over_eleven (a b : ℚ) 
  (h1 : 8 * a + 3 * b = 0) 
  (h2 : a = b - 3) : 
  9 * a = -81 / 11 := 
sorry

end nine_a_eq_frac_minus_eighty_one_over_eleven_l161_161163


namespace evaluate_integral_l161_161520

noncomputable def integral_problem : Real :=
  ∫ x in (-2 : Real)..(2 : Real), (Real.sqrt (4 - x^2) - x^2017)

theorem evaluate_integral :
  integral_problem = 2 * Real.pi :=
sorry

end evaluate_integral_l161_161520


namespace chef_cooked_potatoes_l161_161416

theorem chef_cooked_potatoes
  (total_potatoes : ℕ)
  (cooking_time_per_potato : ℕ)
  (remaining_cooking_time : ℕ)
  (left_potatoes : ℕ)
  (cooked_potatoes : ℕ) :
  total_potatoes = 16 →
  cooking_time_per_potato = 5 →
  remaining_cooking_time = 45 →
  remaining_cooking_time / cooking_time_per_potato = left_potatoes →
  total_potatoes - left_potatoes = cooked_potatoes →
  cooked_potatoes = 7 :=
by
  intros h_total h_cooking_time h_remaining_time h_left_potatoes h_cooked_potatoes
  sorry

end chef_cooked_potatoes_l161_161416


namespace beach_relaxing_people_l161_161359

def row1_original := 24
def row1_got_up := 3

def row2_original := 20
def row2_got_up := 5

def row3_original := 18

def total_left_relaxing (r1o r1u r2o r2u r3o : Nat) : Nat :=
  r1o + r2o + r3o - (r1u + r2u)

theorem beach_relaxing_people : total_left_relaxing row1_original row1_got_up row2_original row2_got_up row3_original = 54 :=
by
  sorry

end beach_relaxing_people_l161_161359


namespace land_profit_each_son_l161_161028

theorem land_profit_each_son :
  let hectares : ℝ := 3
  let m2_per_hectare : ℝ := 10000
  let total_sons : ℕ := 8
  let area_per_son := (hectares * m2_per_hectare) / total_sons
  let m2_per_portion : ℝ := 750
  let profit_per_portion : ℝ := 500
  let periods_per_year : ℕ := 12 / 3

  (area_per_son / m2_per_portion * profit_per_portion * periods_per_year = 10000) :=
by
  sorry

end land_profit_each_son_l161_161028


namespace ratio_q_t_l161_161794

/-- Define the area of one triangular region -/
def t : ℝ := 0.5 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12)

/-- Define the area of one quadrilateral region -/
def q : ℝ := 2 * t

/-- Prove that the ratio of the area of one quadrilateral to the area of one triangle is 2 -/
theorem ratio_q_t : q / t = 2 := by
  -- Proof is omitted
  sorry

end ratio_q_t_l161_161794


namespace floor_of_neg_seven_fourths_l161_161515

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l161_161515


namespace cos_difference_simplification_l161_161296

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  x - y = -1 / (2 * Real.sqrt 5) :=
sorry

end cos_difference_simplification_l161_161296


namespace inequality_holds_l161_161677

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
    a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
by
  sorry

end inequality_holds_l161_161677


namespace prob_not_same_city_l161_161374

def prob_A_city_A : ℝ := 0.6
def prob_B_city_A : ℝ := 0.3

theorem prob_not_same_city :
  (prob_A_city_A * (1 - prob_B_city_A) + (1 - prob_A_city_A) * prob_B_city_A) = 0.54 :=
by 
  -- This is just a placeholder to indicate that the proof is skipped
  sorry

end prob_not_same_city_l161_161374


namespace simplify_fraction_l161_161780

theorem simplify_fraction : (3^3 * 3^(-4) / (3^2 * 3^(-5)) = 1 / 3^8) := by
  sorry

end simplify_fraction_l161_161780


namespace probability_prime_multiple_of_3_l161_161383

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

def count_primes (n : ℕ) : ℕ :=
  (List.range n).filter is_prime |>.length

def count_primes_multiples_of_3 (n : ℕ) : ℕ :=
  (List.range n).filter (λ x => is_prime x ∧ is_multiple_of_3 x) |>.length

theorem probability_prime_multiple_of_3 :
  count_primes_multiples_of_3 31 = 1 ∧ probability_prime_multiple_of_3 1 30 = 1/30 := 
  sorry

end probability_prime_multiple_of_3_l161_161383


namespace find_BD_l161_161651

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables {area : A → B} {area_relation : B → C} {length : D → B}

theorem find_BD (a c : ℝ) 
  (h1 : ∃ Δ ΔABD ΔBCD : ℝ, ΔABD = (1 / 3) * Δ ∧ ΔBCD = (1 / 4) * Δ)
  (h2 : A = (D ∨ D ∈ interior (triangle A B C)))
  (h3 : segment_length_ad : a = distance A D)
  (h4 : segment_length_dc : c = distance D C)
  : ∃ BD : ℝ, BD = sqrt ((8 * c^2 + 3 * a^2) / 35) := 
sorry

end find_BD_l161_161651


namespace similar_triangles_perimeter_ratio_l161_161339

theorem similar_triangles_perimeter_ratio
  (a₁ a₂ s₁ s₂ : ℝ)
  (h₁ : a₁ / a₂ = 1 / 4)
  (h₂ : s₁ / s₂ = 1 / 2) :
  (s₁ / s₂ = 1 / 2) :=
by {
  sorry
}

end similar_triangles_perimeter_ratio_l161_161339


namespace elaine_earnings_increase_l161_161241

variable (E : ℝ) -- Elaine's earnings last year
variable (P : ℝ) -- Percentage increase in earnings

-- Conditions
variable (rent_last_year : ℝ := 0.20 * E)
variable (earnings_this_year : ℝ := E * (1 + P / 100))
variable (rent_this_year : ℝ := 0.30 * earnings_this_year)
variable (multiplied_rent_last_year : ℝ := 1.875 * rent_last_year)

-- Theorem to be proven
theorem elaine_earnings_increase (h : rent_this_year = multiplied_rent_last_year) : P = 25 :=
by
  sorry

end elaine_earnings_increase_l161_161241


namespace B_alone_completion_days_l161_161394

variables (A B C : ℕ) -- work rates of A, B, and C
variables (days : ℕ) -- number of days

-- Given conditions
def condition1 := A = 2 * B
def condition2 := C = 3 * A
def condition3 := (A + B + C) * days = 1
def condition4 := B_worked_days = days - 2

-- Proposition: B alone takes 81 days to complete the work
theorem B_alone_completion_days
    (h1: condition1)
    (h2: condition2)
    (h3: condition3)
    (h4: condition4) :
    B_worked_days = 81 := 
sorry

end B_alone_completion_days_l161_161394


namespace sum_of_valid_m_integers_l161_161581

theorem sum_of_valid_m_integers :
  ∀ (m x y : ℝ),
  (x = 2 - 2 * m ∧ x ≤ 6 ∧ x ≠ 2 ∧ x ≠ -2) ∧ 
  (m - 6 * y > 2 ∧ y - 4 ≤ 3 * y + 4) →
  ( ( ∃ y1 y2 y3 y4 : ℝ,
    -4 ≤ y1 ∧ y1 < (m - 2) / 6 ∧
    -4 ≤ y2 ∧ y2 < (m - 2) / 6 ∧
    -4 ≤ y3 ∧ y3 < (m - 2) / 6 ∧
    -4 ≤ y4 ∧ y4 < (m - 2) / 6 ∧
    y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4 ∧ 
    y1.floor = y1 ∧ y2.floor = y2 ∧ y3.floor = y3 ∧ y4.floor = y4 )) →
  sum (filter (λ m, m ∉ {0,2}) (nat.filter (λ m, -2 ≤ m ∧ m < 2))) = -2 :=
sorry

end sum_of_valid_m_integers_l161_161581


namespace stars_proof_l161_161406

noncomputable def stars_arrangement_possible : Prop :=
  ∃ (grid : Fin 4 → Fin 4 → Bool), 
    (Finset.card (Finset.univ.filter (λ i, Finset.univ.filter (λ j, grid i j).card) = 7) ∧ 
    (∀ i₁ i₂, i₁ ≠ i₂ → (Finset.univ.filter (λ j, grid i₁ j).card) =  (Finset.univ.filter (λ j, grid i₂ j).card)) ∧
    (∀ j₁ j₂, j₁ ≠ j₂ → (Finset.univ.filter (λ i, grid i j₁).card) = (Finset.univ.filter (λ i, grid i j₂).card))

noncomputable def fewer_than_7_stars_impossible : Prop :=
  ¬∃ (grid : Fin 4 → Fin 4 → Bool), 
    (Finset.card (Finset.univ.filter (λ i, Finset.univ.filter (λ j, grid i j).card) < 7) ∧ 
    (∀ i₁ i₂, i₁ ≠ i₂ → (Finset.univ.filter (λ j, grid i₁ j).card) ≤ (Finset.univ.filter (λ j, grid i₂ j).card)) ∧
    (∀ j₁ j₂, j₁ ≠ j₂ → (Finset.univ.filter (λ i, grid i j₁).card) ≤ (Finset.univ.filter (λ i, grid i j₂).card)))

theorem stars_proof :
  stars_arrangement_possible ∧ fewer_than_7_stars_impossible :=
  by
    sorry -- proof steps go here

end stars_proof_l161_161406


namespace min_time_needed_for_noodles_l161_161391

-- Defining the time durations for each step
def wash_pot_time := 2
def wash_vegetables_time := 6
def prepare_noodles_time := 2
def boil_water_time := 10
def cook_noodles_time := 3

-- The total time needed to prepare and cook the noodles
def total_time := wash_pot_time + boil_water_time + cook_noodles_time

-- Proof statement of the minimum time calculation
theorem min_time_needed_for_noodles :
  total_time = 15 :=
by
  -- Wash the pot, boil water concurrently with preparing other things
  have boil_and_wash : wash_pot_time + boil_water_time = 2 + 10 := by rfl
  have prepare_during_boil : wash_vegetables_time + prepare_noodles_time <= boil_water_time := by decide
  have concurrent_time : boil_and_wash + cook_noodles_time = 15 := by decide
  sorry

end min_time_needed_for_noodles_l161_161391


namespace problem_statement_l161_161952

noncomputable def a_n (n : ℕ) := 9 / 2 - n

noncomputable def S_n (n k : ℕ) := -1 / 2 * n ^ 2 + k * n

noncomputable def T_n (n : ℕ) :=
  (Finset.range n).sum (λ i, (i + 1) / 2^i)

theorem problem_statement (k : ℕ) (h1 : k ∈ Set.Icc 1 (Int.natAbs k)) (h2 : S_n k k = 8) :
  k = 4 ∧ (∀ n : ℕ, a_n n = 9 / 2 - n) ∧ (∀ n : ℕ, T_n n = 4 - (n + 2) / 2^(n - 1)) :=
by
  split
  sorry
  split
  sorry
  sorry

end problem_statement_l161_161952


namespace mitchell_pencils_l161_161276

/-- Mitchell and Antonio have a combined total of 54 pencils.
Mitchell has 6 more pencils than Antonio. -/
theorem mitchell_pencils (A M : ℕ) 
  (h1 : M = A + 6)
  (h2 : M + A = 54) : M = 30 :=
by
  sorry

end mitchell_pencils_l161_161276


namespace find_number_l161_161541

theorem find_number (x : ℝ) 
  (h : (28 + x / 69) * 69 = 1980) :
  x = 1952 :=
sorry

end find_number_l161_161541


namespace matrix_N_cross_product_l161_161085

theorem matrix_N_cross_product (v : ℝ^3) :
  let N := λ (v : ℝ^3), ![![0, -4, -1], ![4, 0, -3], ![1, 3, 0]]
  in N.mul_vec v = vector_cross_product ![3, -1, 4] v :=
by
  sorry

end matrix_N_cross_product_l161_161085


namespace rachel_reading_homework_l161_161287

theorem rachel_reading_homework (math_hw : ℕ) (additional_reading_hw : ℕ) (total_reading_hw : ℕ) 
  (h1 : math_hw = 8) (h2 : additional_reading_hw = 6) (h3 : total_reading_hw = math_hw + additional_reading_hw) :
  total_reading_hw = 14 :=
sorry

end rachel_reading_homework_l161_161287


namespace problem_equivalent_statement_l161_161797

-- Define the triangle and the conditions
structure RightIsoscelesTriangle :=
  (b c : ℝ)
  (is_right_isosceles : b^2 + b^2 = c^2)

-- Define the squares
structure Square :=
  (side : ℝ)
  (area : ℝ := side^2)

-- Define the conditions for the problem
noncomputable def square_area_condition (s : Square) (area_val : ℝ) := s.area = area_val

noncomputable def problem_statement : Prop :=
  ∃ (b c : ℝ) (ADEF GHIJ : Square),
    RightIsoscelesTriangle b c ∧
    square_area_condition ADEF 2250 ∧
    GHIJ.area = 2000

theorem problem_equivalent_statement :
  problem_statement :=
sorry

end problem_equivalent_statement_l161_161797


namespace smallest_positive_omega_l161_161585

theorem smallest_positive_omega 
    (omega : ℝ) 
    (h : ∀ x : ℝ, 
        (sin (omega * (x - π / (3 * omega)) + π / 3) = -sin (omega * x + π / 3))
    ) 
    : omega = 3 :=
sorry

end smallest_positive_omega_l161_161585


namespace find_distance_d_l161_161092

-- Define the side length of the equilateral triangle.
def side_length (ABC: Triangle) : ℝ := 800

-- Define the points outside the plane of the triangle.
structure PointOutside (A B C : Point) :=
  (P Q : Point)
  (outside_plane : ¬ coplanar {P, Q, A, B, C})
  (opposite_sides : P.z > 0 ∧ Q.z < 0)
  (equal_distances_p : dist P A = dist P B ∧ dist P B = dist P C)
  (equal_distances_q : dist Q A = dist Q B ∧ dist Q B = dist Q C)
  (dihedral_angle : angle (plane P A B) (plane Q A B) = π / 2)

-- There is a point O with equal distances to A, B, C, P, and Q.
def point_O (A B C P Q : Point) : Point :=
  { O : Point // dist O A = dist O B ∧ dist O B = dist O C ∧ 
                    dist O C = dist O P ∧ dist O P = dist O Q }

-- The distance d from O to each of the points is given as 377.96.
theorem find_distance_d (ABC: Triangle)
  (ABC_equilateral : is_equilateral ABC)
  (side_len_ABC : side_length ABC = 800)
  (A B C : Point)
  (P Q : Point)
  (points_outside : PointOutside A B C)
  (O : Point)
  (O_equal_distances : ∀ p ∈ {A, B, C, P, Q}, dist O p = dist O A) :
  ∃ d : ℝ, d = 377.96 :=
by
  sorry

end find_distance_d_l161_161092


namespace min_value_abs_function_l161_161327

theorem min_value_abs_function : ∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → (|x - 4| + |x - 6| = 2) :=
by
  sorry


end min_value_abs_function_l161_161327


namespace arithmetic_sequence_S30_l161_161351

variable {α : Type*} [OrderedAddCommGroup α]

-- Definitions from the conditions
def arithmetic_sum (n : ℕ) : α :=
  sorry -- Placeholder for the sequence sum definition

axiom S10 : arithmetic_sum 10 = 20
axiom S20 : arithmetic_sum 20 = 15

-- The theorem to prove
theorem arithmetic_sequence_S30 : arithmetic_sum 30 = -15 :=
  sorry -- Proof will be completed here

end arithmetic_sequence_S30_l161_161351


namespace find_other_number_l161_161325

theorem find_other_number (lcm_ab : Nat) (gcd_ab : Nat) (a b : Nat) 
  (hlcm : Nat.lcm a b = lcm_ab) 
  (hgcd : Nat.gcd a b = gcd_ab) 
  (ha : a = 210) 
  (hlcm_ab : lcm_ab = 2310) 
  (hgcd_ab : gcd_ab = 55) 
  : b = 605 := 
by 
  sorry

end find_other_number_l161_161325


namespace seating_arrangements_l161_161987

-- Define the basic setup for teams and the number of players in each team
def cubs : ℕ := 3
def red_sox : ℕ := 3
def yankees : ℕ := 2
def dodgers : ℕ := 2
def total_players : ℕ := cubs + red_sox + yankees + dodgers

-- Define the factorial function
@[simp] def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- The final proof statement
theorem seating_arrangements :
  total_players = 10 →
  cubs = 3 →
  red_sox = 3 →
  yankees = 2 →
  dodgers = 2 →
  fact 4 * fact cubs * fact red_sox * fact yankees * fact dodgers = 3456 :=
by
  intros h_total h_cubs h_red_sox h_yankees h_dodgers
  rw [h_cubs, h_red_sox, h_yankees, h_dodgers]
  simp
  -- we skip the explicit factorial calculations for brevity
  admit

end seating_arrangements_l161_161987


namespace initial_deposit_value_l161_161054

noncomputable def initial_deposit (A r : ℝ) (n t : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem initial_deposit_value :
  initial_deposit 914.6152747265625 0.05 12 7 ≈ 645.953292 :=
sorry

end initial_deposit_value_l161_161054


namespace problem_a_problem_b_l161_161672

-- Given
def dodecagon (P : Point) := regular_dodecagon P
def lines_intersect (G N M I : Point) : Prop := ∃ P, line G N ∩ line M I = {P}

-- Propositions to prove
theorem problem_a (G I P : Point) (GERMANYISHOT : dodecagon P) (H P : Point) (Gn Mi : Prop) :
  lines_intersect G N M I ->
  perimeter_triangle G I P = perimeter_dodecagon GERMANYISHOT :=
sorry

theorem problem_b (G I P A : Point) (GERMANYISHOT : dodecagon P) (H P : Point) (Gn Mi : Prop) :
  lines_intersect G N M I ->
  distance P A = side_length_dodecagon GERMANYISHOT :=
sorry

end problem_a_problem_b_l161_161672


namespace ordered_pair_for_quadratic_with_same_roots_l161_161770

theorem ordered_pair_for_quadratic_with_same_roots (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ (x = 7 ∨ x = 1)) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = 7 ∨ x = 1)) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end ordered_pair_for_quadratic_with_same_roots_l161_161770


namespace polar_equation_circle_l161_161789

theorem polar_equation_circle (ρ θ : ℝ) :
  (ρ = 1 → ∃ x y, x^2 + y^2 = 1 ∧ x = ρ * cos θ ∧ y = ρ * sin θ) :=
by 
  sorry

end polar_equation_circle_l161_161789


namespace max_weak_quartets_l161_161849

def Person := ℕ -- Defining a person as an individual in the natural numbers

def group := fin 120 -- Group of 120 people represented as a finite type of size 120

def friendship (p1 p2 : group) : Prop := sorry -- Placeholder to define a friendship relation

def is_weak_quartet (s : finset group) : Prop :=
  s.card = 4 ∧ ∃ (p1 p2 ∈ s), friendship p1 p2 ∧ ∀ p3 p4 ∈ s, p3 ≠ p1 ∨ p4 ≠ p2 ∨ ¬friendship p3 p4

theorem max_weak_quartets : 
  ∃ m : ℕ, m = 4769280 ∧ ∀ wq_set, m ≤ (finset.card (finset.filter is_weak_quartet wq_set)) :=
sorry

end max_weak_quartets_l161_161849


namespace find_a_b_l161_161576

-- Conditions defining the solution sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | -3 < x ∧ x < 2 }

-- The solution set of the inequality x^2 + ax + b < 0 is the intersection A∩B
def C : Set ℝ := A ∩ B

-- Proving that there exist values of a and b such that the solution set C corresponds to the inequality x^2 + ax + b < 0
theorem find_a_b : ∃ a b : ℝ, (∀ x : ℝ, C x ↔ x^2 + a*x + b < 0) ∧ a + b = -3 := 
by 
  sorry

end find_a_b_l161_161576


namespace fruit_punch_total_l161_161310

section fruit_punch
variable (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) (total_punch : ℝ)

axiom h1 : orange_punch = 4.5
axiom h2 : cherry_punch = 2 * orange_punch
axiom h3 : apple_juice = cherry_punch - 1.5
axiom h4 : total_punch = orange_punch + cherry_punch + apple_juice

theorem fruit_punch_total : total_punch = 21 := sorry

end fruit_punch

end fruit_punch_total_l161_161310


namespace cos_diff_l161_161292

theorem cos_diff (x y : ℝ) (hx : x = Real.cos (20 * Real.pi / 180)) (hy : y = Real.cos (40 * Real.pi / 180)) :
  x - y = 1 / 2 :=
by sorry

end cos_diff_l161_161292


namespace trapezoid_area_l161_161223

-- Definitions based on conditions
variable (ABC ADF : Type) -- representing triangles ABC and ADF
variable (area_small_triangles : ℕ) -- number of smallest triangles in ADF

-- Conditions
def similar_isosceles (T : Type) : Prop := ∃ AB AC, AB = AC
def area (T : Type) : ℕ := if T = ABC then 96 else if T = ADF then 16 else 0
def smallest_triangle_area : ℕ := 2
def num_smallest_triangles (T : Type) : ℕ := if T = ADF then 8 else 12

-- Main statement
theorem trapezoid_area (ABC ADF DBCE : Type)
  (h_similar : similar_isosceles ABC)
  (h_area_ABC : area ABC = 96)
  (h_smallest_triangle_area : smallest_triangle_area = 2)
  (h_num_smallest_triangles_ADF : num_smallest_triangles ADF = 8)
  : area DBCE = 80 := 
sorry

end trapezoid_area_l161_161223


namespace positive_integers_p_divisibility_l161_161528

theorem positive_integers_p_divisibility (p : ℕ) (hp : 0 < p) :
  (∃ n : ℕ, 0 < n ∧ p^n + 3^n ∣ p^(n+1) + 3^(n+1)) ↔ p = 3 ∨ p = 6 ∨ p = 15 :=
by sorry

end positive_integers_p_divisibility_l161_161528


namespace modulus_of_conjugate_l161_161328

theorem modulus_of_conjugate (z : ℂ) (h : z = (i / (1 - i))) : |conj z| = (Real.sqrt 2) / 2 :=
by
  sorry

end modulus_of_conjugate_l161_161328


namespace coef_xk_expansion_l161_161709

open BigOperators

theorem coef_xk_expansion (n k : ℕ) :
  coefficient (x^k) ((1 + x + x^2 + x^3)^n) = ∑ j in Finset.range (k / 2 + 1), choose n (k - 2 * j) * choose n j :=
sorry

end coef_xk_expansion_l161_161709


namespace inequality_solution_l161_161710

theorem inequality_solution (x : ℝ) : 
  (x ≠ -3) → (x ≠ 4) → 
  (x-3)/(x+3) > (2*x-1)/(x-4) ↔ 
  (x > -6 - 3 * real.sqrt 17 ∧ x < -6 + 3 * real.sqrt 17) ∨ (x > -3 ∧ x < 4) :=
by { sorry }

end inequality_solution_l161_161710


namespace range_of_a_l161_161117

theorem range_of_a (a : ℝ) : (∀ x > 0, log x ≥ (a / x - exp(1) * x + 2)) → a ≤ -2/exp(1) := 
by 
  sorry

end range_of_a_l161_161117


namespace each_son_can_make_l161_161026

noncomputable def land_profit
    (total_land : ℕ) -- measured in hectares
    (num_sons : ℕ)
    (profit_per_section : ℕ) -- profit in dollars per 750 m^2 per 3 months
    (hectare_to_m2 : ℕ) -- conversion factor from hectares to square meters
    (section_area : ℕ) -- 750 m^2
    (periods_per_year : ℕ) : ℕ :=
  let each_son's_share := total_land * hectare_to_m2 / num_sons in
  let num_sections := each_son's_share / section_area in
  num_sections * profit_per_section * periods_per_year

theorem each_son_can_make
    (total_land : ℕ)
    (num_sons : ℕ)
    (profit_per_section : ℕ)
    (hectare_to_m2 : ℕ)
    (section_area : ℕ)
    (periods_per_year : ℕ) :
  total_land = 3 ∧
  num_sons = 8 ∧
  profit_per_section = 500 ∧
  hectare_to_m2 = 10000 ∧
  section_area = 750 ∧
  periods_per_year = 4 →
  land_profit total_land num_sons profit_per_section hectare_to_m2 section_area periods_per_year = 10000 :=
by
  intros h
  cases h
  sorry

end each_son_can_make_l161_161026


namespace clock_angle_at_8_20_is_130_degrees_l161_161382

/--
A clock has 12 hours, and each hour represents 30 degrees.
The minute hand moves 6 degrees per minute.
The hour hand moves 0.5 degrees per minute from its current hour position.
Prove that the smaller angle between the hour and minute hands at 8:20 p.m. is 130 degrees.
-/
theorem clock_angle_at_8_20_is_130_degrees
    (hours_per_clock : ℝ := 12)
    (degrees_per_hour : ℝ := 360 / hours_per_clock)
    (minutes_per_hour : ℝ := 60)
    (degrees_per_minute : ℝ := 360 / minutes_per_hour)
    (hour_slider_per_minute : ℝ := degrees_per_hour / minutes_per_hour)
    (minute_hand_at_20 : ℝ := 20 * degrees_per_minute)
    (hour_hand_at_8: ℝ := 8 * degrees_per_hour)
    (hour_hand_move_in_20_minutes : ℝ := 20 * hour_slider_per_minute)
    (hour_hand_at_8_20 : ℝ := hour_hand_at_8 + hour_hand_move_in_20_minutes) :
  |hour_hand_at_8_20 - minute_hand_at_20| = 130 :=
by
  sorry

end clock_angle_at_8_20_is_130_degrees_l161_161382


namespace expected_profit_is_correct_l161_161819

noncomputable def expected_profit : ℝ :=
let profits : List ℝ := [50, 30, -20]
let probabilities : List ℝ := [0.6, 0.3, 0.1]
profits.zip probabilities |>.sum (λ (xp : ℝ × ℝ), xp.1 * xp.2)

theorem expected_profit_is_correct : expected_profit = 37 := by
  sorry

end expected_profit_is_correct_l161_161819


namespace minimum_distance_to_line_l161_161120

variables (a b : ℝ) (m n : ℝ)

def vector_a := (1 : ℝ, 0 : ℝ)
def vector_b := (0 : ℝ, 1 : ℝ)
def vector_c := (m, n)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def distance_from_point_to_line (m n : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * m + b * n + c) / sqrt (a^2 + b^2)

theorem minimum_distance_to_line :
  (dot_product (1 - m, 0 - n) (0 - m, 1 - n) = 0) →
  ∃ (m n : ℝ), distance_from_point_to_line m n 1 1 1 = sqrt 2 / 2 :=
sorry

end minimum_distance_to_line_l161_161120


namespace relationship_abc_l161_161254

def f (x : ℝ) : ℝ := 2^|x|

def a : ℝ := f (Real.log 10 / Real.log 3)
def b : ℝ := f (Real.log (1 / 99))
def c : ℝ := f 0

theorem relationship_abc : a > b ∧ b > c :=
by
  -- Definitions and conditions given in the problem
  have h1 : f (Real.log 10 / Real.log 3) > f (Real.log (1/99)),
    from sorry,  -- Placeholder for step showing a > b
  have h2 : f (Real.log (1/99)) > f 0,
    from sorry,  -- Placeholder for step showing b > c
  exact ⟨h1, h2⟩

end relationship_abc_l161_161254


namespace sequence_a_2015_l161_161645

theorem sequence_a_2015 : 
  ∃ (a : ℕ → ℤ), 
    a 1 = 2 ∧ a 2 = 10 ∧ 
    (∀ n : ℕ, n > 0 → a (n + 2) = a (n + 1) - a n) ∧ 
    a 2015 = -10 :=
by 
  use (λ n, if n % 6 = 1 then 2 else if n % 6 = 2 then 10 else if n % 6 = 3 then 8 else if n % 6 = 4 then -2 else if n % 6 = 5 then -10 else -8)
  -- Proof for the required sequence conditions will go here
  sorry

end sequence_a_2015_l161_161645


namespace value_of_a7_l161_161937

-- Let \( \{a_n\} \) be a sequence such that \( S_n \) denotes the sum of the first \( n \) terms.
-- Given \( S_{n+1}, S_{n+2}, S_{n+3} \) form an arithmetic sequence and \( a_2 = -2 \),
-- prove that \( a_7 = 64 \).

theorem value_of_a7 (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 2) + S (n + 1) = 2 * S n) →
  a 2 = -2 →
  (∀ n : ℕ, a (n + 2) = -2 * a (n + 1)) →
  a 7 = 64 :=
by
  -- skip the proof
  sorry

end value_of_a7_l161_161937


namespace resulting_polygon_has_18_sides_l161_161456

def regular_polygon_sides : ℕ → ℕ
| 0 => 5  -- Pentagon
| 1 => 3  -- Equilateral Triangle
| 2 => 8  -- Octagon
| 3 => 6  -- Hexagon
| 4 => 4  -- Square
| _ => 0  -- Not used

def adjacent_sides : ℕ → ℕ
| 0 => 1  -- Pentagon adjacent sides
| 1 => 2  -- Triangle adjacent sides
| 2 => 2  -- Octagon adjacent sides
| 3 => 2  -- Hexagon adjacent sides
| 4 => 1  -- Square adjacent sides
| _ => 0  -- Not used

theorem resulting_polygon_has_18_sides :
  let total_sides := (List.range 5).sum (λ i => regular_polygon_sides i)
  let shared_sides := (List.range 5).sum (λ i => adjacent_sides i)
  (total_sides - shared_sides) = 18 :=
by
  let total_sides := (List.range 5).sum (λ i => regular_polygon_sides i)
  let shared_sides := (List.range 5).sum (λ i => adjacent_sides i)
  have h : total_sides - shared_sides = 18 := sorry
  exact h

end resulting_polygon_has_18_sides_l161_161456


namespace floor_neg_seven_quarter_l161_161479

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l161_161479


namespace equal_product_of_distances_l161_161447

theorem equal_product_of_distances
  (A B C I D E F M N P : Point)
  (O : Circle)
  (rays_A : collinear A I D)
  (rays_B : collinear B I E)
  (rays_C : collinear C I F)
  (circ_A : on_circle A O)
  (circ_B : on_circle B O)
  (circ_C : on_circle C O)
  (circ_M : on_circle M O)
  (circ_N : on_circle N O)
  (circ_P : on_circle P O)
  (center_I : incenter I A B C)
  (incenter_M : incenter I M A)
  (incenter_N : incenter I N B)
  (incenter_P : incenter I P C) :
  dist A M * dist I D = dist B N * dist I E
  ∧ dist B N * dist I E = dist C P * dist I F :=
by
  sorry

end equal_product_of_distances_l161_161447


namespace min_abs_z_l161_161201

noncomputable def z_min_value (z : ℂ) : ℝ :=
  complex.abs z

theorem min_abs_z (z : ℂ) (h : complex.abs (z - 1) + complex.abs (z - 3 - 2 * complex.I) = 2 * Real.sqrt 2) :
  ∃ z_min : ℝ, ∀ w : ℂ, (complex.abs (w - 1) + complex.abs (w - 3 - 2 * complex.I) = 2 * Real.sqrt 2) → z_min_value w ≥ z_min :=
  ∃ z_min, z_min = 1 := sorry

end min_abs_z_l161_161201


namespace floor_neg_seven_over_four_l161_161495

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l161_161495


namespace total_beetles_eaten_each_day_l161_161466

-- Definitions from the conditions
def birds_eat_per_day : ℕ := 12
def snakes_eat_per_day : ℕ := 3
def jaguars_eat_per_day : ℕ := 5
def number_of_jaguars : ℕ := 6

-- Theorem statement
theorem total_beetles_eaten_each_day :
  (number_of_jaguars * jaguars_eat_per_day) * snakes_eat_per_day * birds_eat_per_day = 1080 :=
by sorry

end total_beetles_eaten_each_day_l161_161466


namespace a_squared_plus_b_squared_composite_l161_161749

theorem a_squared_plus_b_squared_composite (a b x1 x2 : ℕ) 
  (h_roots : ∀ x : ℕ, x^2 + a * x + (b + 1) = 0 → x ∈ {x1, x2}) :
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n := by
sorry

end a_squared_plus_b_squared_composite_l161_161749


namespace matthew_initial_crackers_l161_161694

theorem matthew_initial_crackers :
  ∃ C : ℕ,
  (∀ (crackers_per_friend cakes_per_friend : ℕ), cakes_per_friend * 4 = 98 → crackers_per_friend = cakes_per_friend → crackers_per_friend * 4 + 8 * 4 = C) ∧ C = 128 :=
sorry

end matthew_initial_crackers_l161_161694


namespace workers_count_l161_161016

theorem workers_count :
  ∃ (W : ℕ), 
  (∀ (A : ℕ), 65 * W = 55 * (W + 10)) ∧
  W = 55 :=
begin
  use 55,
  split,
  { intros A,
    have h : 65 * 55 = 55 * (55 + 10),
    { calc 65 * 55 = 65 * (50 + 5) : by rw (55_eq_50_plus_5)
          ... = (65 * 50) + (65 * 5) : by rw (mul_add 65 50 5)
          ... = (3250) + (325) : by norm_num,
    calc 55 * (55 + 10) = 55 * 65 : by ring
                    ... = 3250 + 325 : by norm_num,
    calc 3250 + 325 = 3250 + 325 : by refl
    },
    exact h,
  },
  { refl, }
}

end workers_count_l161_161016


namespace student_count_l161_161317

theorem student_count 
  (initial_avg_height : ℚ)
  (incorrect_height : ℚ)
  (actual_height : ℚ)
  (actual_avg_height : ℚ)
  (n : ℕ)
  (h1 : initial_avg_height = 175)
  (h2 : incorrect_height = 151)
  (h3 : actual_height = 136)
  (h4 : actual_avg_height = 174.5)
  (h5 : n > 0) : n = 30 :=
by
  sorry

end student_count_l161_161317


namespace parallelogram_opposite_sides_equal_trapezoid_opposite_sides_not_equal_necessarily_l161_161549

/--
A definition of a quadrilateral.
-/
structure Quadrilateral :=
  (A B C D : Point)

/--
A definition of a parallelogram.
-/
structure Parallelogram extends Quadrilateral :=
  (AB_parallel_CD : Parallel (Line A B) (Line C D))
  (AD_parallel_BC : Parallel (Line A D) (Line B C))

/--
A definition of a trapezoid.
-/
structure Trapezoid extends Quadrilateral :=
  (EF_parallel_GH : Parallel (Line E F) (Line G H))
  (FG_not_parallel : ¬Parallel (Line F G) (Line H E))

/--
Theorem: Prove that in a parallelogram, opposite sides are parallel and equal in length,
while in a trapezoid, opposite sides being parallel does not imply they are equal in length.
-/
theorem parallelogram_opposite_sides_equal (p : Parallelogram) :
  (length (Line p.A p.B) = length (Line p.C p.D)) ∧ (length (Line p.A p.D) = length (Line p.B p.C)) :=
sorry

theorem trapezoid_opposite_sides_not_equal_necessarily (t : Trapezoid) :
  ¬(length (Line t.E t.F) = length (Line t.G t.H)) :=
sorry

end parallelogram_opposite_sides_equal_trapezoid_opposite_sides_not_equal_necessarily_l161_161549


namespace least_positive_a_l161_161244

theorem least_positive_a (p : ℕ) [Fact (Nat.Prime p)] (hp : 2 < p) :
  ∃ (a : ℕ), 0 < a ∧ (∃ f g : Polynomial ℤ, a = (Polynomial.X - 1) * f + (∑ i in Finset.range p, Polynomial.X ^ i) * g) ∧ a = p := by
sory

end least_positive_a_l161_161244


namespace leftmost_digit_of_12_pow_37_l161_161926

theorem leftmost_digit_of_12_pow_37 
    (h1 : 0.3010 < real.log10 2 ∧ real.log10 2 < 0.3011)
    (h2 : 0.4771 < real.log10 3 ∧ real.log10 3 < 0.4772) :
    12^37 % 10^36 / 10^35 = 8 :=
by
  sorry

end leftmost_digit_of_12_pow_37_l161_161926


namespace modulus_complex_l161_161119

theorem modulus_complex (a b : ℝ) (h1: (1 + a * complex.I) * complex.I = 2 - b * complex.I) : 
  complex.abs (a + b * complex.I) = real.sqrt 5 :=
by
  sorry

end modulus_complex_l161_161119


namespace infinite_nested_radical_l161_161100

theorem infinite_nested_radical : 
  (x : Real) (h : x = Real.sqrt (3 - x)) : 
  x = (-1 + Real.sqrt 13) / 2 :=
by
  sorry

end infinite_nested_radical_l161_161100


namespace equilateral_triangle_cd_l161_161740

theorem equilateral_triangle_cd 
  (c d : ℝ)
  (h1 : (0 : ℂ) = (0 : ℂ))
  (h2 : (c + 7*complex.I : ℂ) = (c + 7*complex.I : ℂ))
  (h3 : (d + 19*complex.I : ℂ) = (d + 19*complex.I : ℂ))
  (this_eq : (d + 19 * complex.I) = (c + 7 * complex.I) * (complex.of_real (-1 / 2) + complex.I*(real.sqrt 3 / 2))) :
  c * d = -806 / 9 :=
by
  sorry

end equilateral_triangle_cd_l161_161740


namespace highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l161_161018

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def sales_volume_decrease_per_yuan_increase : ℝ := 2000

-- Question 1
theorem highest_price_without_lowering_revenue :
  ∀ (x : ℝ), 
  25 ≤ x ∧ (8 - (x - original_price) * 0.2) * x ≥ 25 * 8 → 
  x ≤ 40 :=
sorry

-- Question 2
noncomputable def tech_reform_fee (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600)
noncomputable def fixed_promotion_fee : ℝ := 50
noncomputable def variable_promotion_fee (x : ℝ) : ℝ := (1 / 5) * x

theorem minimum_annual_sales_volume_and_price (x : ℝ) (a : ℝ) :
  x > 25 →
  (a * x ≥ 25 * 8 + fixed_promotion_fee + tech_reform_fee x + variable_promotion_fee x) →
  (a ≥ 10.2 ∧ x = 30) :=
sorry

end highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l161_161018


namespace original_price_of_shirts_l161_161273

theorem original_price_of_shirts 
  (sale_price : ℝ) 
  (fraction_of_original : ℝ) 
  (original_price : ℝ) 
  (h1 : sale_price = 6) 
  (h2 : fraction_of_original = 0.25) 
  (h3 : sale_price = fraction_of_original * original_price) 
  : original_price = 24 := 
by 
  sorry

end original_price_of_shirts_l161_161273


namespace count_four_digit_even_numbers_excluding_5_and_6_l161_161598

theorem count_four_digit_even_numbers_excluding_5_and_6 : 
  ∃ n : ℕ, n = 1792 ∧ 
    (∀ d1 d2 d3 d4: ℕ, 
      d1 ∈ {1, 2, 3, 4, 7, 8, 9} →
      d2 ∈ {0, 1, 2, 3, 4, 7, 8, 9} →
      d3 ∈ {0, 1, 2, 3, 4, 7, 8, 9} →
      d4 ∈ {0, 2, 4, 8} →
      d1 > 0 ∧ d4 % 2 = 0) 
      ∧ n = 7 * 8 * 8 * 4 := 
by
  existsi 1792
  split
  focus
    reflexivity
  sorry

end count_four_digit_even_numbers_excluding_5_and_6_l161_161598


namespace area_of_larger_region_l161_161830

-- Definitions based on conditions
def unit_circle_radius : ℝ := 1
def segment_length : ℝ := 1

-- The statement of the proof problem
theorem area_of_larger_region :
  ∀ (circle_radius segment_length : ℝ),
    circle_radius = 1 ∧ segment_length = 1 →
    let area_of_larger_region := π - (π / 6 - (Real.sqrt 3 / 4)) in
    area_of_larger_region = 5 * π / 6 + Real.sqrt 3 / 4 :=
by
  intros circle_radius segment_length hconds
  sorry

end area_of_larger_region_l161_161830


namespace not_possible_l161_161232

theorem not_possible (a : ℕ → ℝ) (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 13 → a i + a (i+1) + a (i+2) > 0) (h2 : ∑ i in Icc 1 15, a i < 0) : False :=
by
  sorry

end not_possible_l161_161232


namespace median_of_scores_is_90_l161_161115

theorem median_of_scores_is_90 : 
  let scores := [86, 95, 97, 90, 88] in 
  let sorted_scores := List.sort (· ≤ ·) scores in 
  (sorted_scores.length = 5) →
  sorted_scores.nth 2 = some 90 :=
by
  let scores := [86, 95, 97, 90, 88]
  let sorted_scores := List.sort (· ≤ ·) scores
  have h1 : sorted_scores.length = 5 := by sorry
  exact h1
  have h2 : sorted_scores.nth 2 = some 90 := by sorry
  exact h2

end median_of_scores_is_90_l161_161115


namespace cos_diff_l161_161294

theorem cos_diff (x y : ℝ) (hx : x = Real.cos (20 * Real.pi / 180)) (hy : y = Real.cos (40 * Real.pi / 180)) :
  x - y = 1 / 2 :=
by sorry

end cos_diff_l161_161294


namespace expression_equals_five_l161_161186

theorem expression_equals_five (a : ℝ) (h : 2 * a^2 - 3 * a + 4 = 5) : 7 + 6 * a - 4 * a^2 = 5 :=
by
  sorry

end expression_equals_five_l161_161186


namespace hyperbola_equation_l161_161934

theorem hyperbola_equation (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
(hyperbola_def : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
(ellipse_def : ∀ x y, x^2 / c^2 + y^2 / d^2 = 1)
(shared_foci : a^2 + b^2 = c^2 - d^2)
(hyperbola_ecc : ∀ e_ell, (c / a) = 2 * e_ell → (c / a) = 2 * ((√(c^2 - d^2)) / c)) :
  ∀ x y, (x^2 / a^2 - y^2 / (3 / 2 * a^2) = 1) :=
by
  sorry

end hyperbola_equation_l161_161934


namespace find_v_plus_z_l161_161762

variable (x u v w z : ℂ)
variable (y : ℂ)
variable (condition1 : y = 2)
variable (condition2 : w = -x - u)
variable (condition3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I)

theorem find_v_plus_z : v + z = -4 :=
by
  have h1 : y = 2 := condition1
  have h2 : w = -x - u := condition2
  have h3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I := condition3
  sorry

end find_v_plus_z_l161_161762


namespace floor_neg_seven_fourths_l161_161471

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l161_161471


namespace solve_for_y_l161_161795

theorem solve_for_y (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := 
by 
  sorry

end solve_for_y_l161_161795


namespace coefficient_of_term_free_of_x_l161_161938

theorem coefficient_of_term_free_of_x 
  (n : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → n = 10) 
  (h2 : (n.choose 4 / n.choose 2) = 14 / 3) : 
  ∃ (c : ℚ), c = 5 :=
by
  sorry

end coefficient_of_term_free_of_x_l161_161938


namespace find_smaller_interior_angle_l161_161324

-- Define the 8 congruent isosceles trapezoids and their arrangement
structure KeystoneArch where
  n : ℕ -- number of trapezoids
  is_congruent : ∀ i j, i ≠ j → IsCongruent (trapezoid i) (trapezoid j)
  is_isosceles : ∀ i, IsIsosceles (trapezoid i)
  fits_together : ∀ i, FitsTogether (trapezoid i) (trapezoid ((i + 1) % n))
  horizontal_ends : HorizontalEnds (trapezoid 0) (trapezoid (n-1))

-- State the theorem we want to prove
theorem find_smaller_interior_angle (arch : KeystoneArch) (h : arch.n = 8) : 
 ∃ y : ℝ, y = 78.75 :=
sorry

end find_smaller_interior_angle_l161_161324


namespace divisor_of_sum_of_four_consecutive_integers_l161_161364

theorem divisor_of_sum_of_four_consecutive_integers (n : ℤ) :
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by sorry

end divisor_of_sum_of_four_consecutive_integers_l161_161364


namespace find_f_of_3_l161_161687

def f (x : ℝ) : ℝ :=
if x < -2 then 2 * x + 9 else 5 - 2 * x

theorem find_f_of_3 : f 3 = -1 :=
by 
  -- we will prove this by definition of piecewise function
  -- sorry is a placeholder to indicate skipping the proof
  sorry

end find_f_of_3_l161_161687


namespace exists_polynomial_for_cosine_l161_161542

noncomputable def ChebyshevPolynomial : ℕ → polynomial ℚ
| 0      := 1
| 1      := X
| (n+2)  := 2 * X * ChebyshevPolynomial (n+1) - ChebyshevPolynomial n

theorem exists_polynomial_for_cosine (n: ℕ) (h: n > 0) :
  ∃ p : polynomial ℚ, ∀ x, p (2 * cos x) = 2 * cos (n * x) :=
sorry

end exists_polynomial_for_cosine_l161_161542


namespace total_journey_time_l161_161986

theorem total_journey_time (river_speed : ℝ) (distance_upstream : ℝ) (boat_speed_still_water : ℝ) :
  river_speed = 2 ∧ distance_upstream = 56 ∧ boat_speed_still_water = 6 →
  (distance_upstream / (boat_speed_still_water - river_speed) + distance_upstream / (boat_speed_still_water + river_speed)) = 21 :=
by {
  intros h,
  rcases h with ⟨h_river_speed, h_distance_upstream, h_boat_speed_still_water⟩,
  rw [h_river_speed, h_distance_upstream, h_boat_speed_still_water],
  norm_num,
  sorry
}

end total_journey_time_l161_161986


namespace max_value_f_l161_161326

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem max_value_f : ∃ x ∈ set.Icc (-3 : ℝ) (3 : ℝ), f x = 16 := 
by
  sorry

end max_value_f_l161_161326


namespace convex_polygon_triangulation_black_white_difference_l161_161873

theorem convex_polygon_triangulation_black_white_difference
  (n : ℕ) (hn : n ≥ 4)
  (triangulation : list (list ℕ))
  (is_convex : convex_polygon n)
  (is_triangulation : valid_triangulation n triangulation)
  (triangle_classification : list (list ℕ) → ℕ)
  (classification : triangle_classification := λ t, if t has 2 sides shared with polygon then 1 else if t has 1 side shared then 0 else -1) :
  ∑ t in triangulation, classification t = 2 :=
by sorry

end convex_polygon_triangulation_black_white_difference_l161_161873


namespace max_g_on_1_3_l161_161930

noncomputable theory

def f (x : Real) : Real := x ^ (-2)

def g (x : Real) : Real := (x - 1) * f x

theorem max_g_on_1_3 : 
  ∃ x : Real, x ∈ set.Icc 1 3 ∧ g x = (1 / 4) := 
sorry

end max_g_on_1_3_l161_161930


namespace percentage_saved_l161_161815

-- Define the actual and saved amount.
def actual_investment : ℕ := 150000
def saved_amount : ℕ := 50000

-- Define the planned investment based on the conditions.
def planned_investment : ℕ := actual_investment + saved_amount

-- Proof goal: The percentage saved is 25%.
theorem percentage_saved : (saved_amount * 100) / planned_investment = 25 := 
by 
  sorry

end percentage_saved_l161_161815


namespace find_general_term_l161_161919

def sequence (n : ℕ) : ℝ := sorry
def partial_sum (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), sequence i

def constant_sequence (c : ℝ) : Prop :=
  ∀ n : ℕ, partial_sum n - n^2 * sequence n = c

noncomputable def general_term (n : ℕ) : ℝ :=
  if n = 0 then 1 else 2 / (n * (n + 1))

theorem find_general_term (c : ℝ)
  (h1 : sequence 1 = 1)
  (h2 : constant_sequence c) :
  ∀ n, sequence n = general_term n :=
sorry

end find_general_term_l161_161919


namespace point_inside_after_25_reflections_point_outside_after_24_reflections_l161_161739

-- Assume the definitions based on the conditions
def center_circle := (0, 0) -- Center of the circle
def radius_circle := 1 -- Radius of the circle
def start_distance := 50 -- Initial distance from point A to the center of the circle

-- Prove that point A can be moved inside the circle after 25 reflections
theorem point_inside_after_25_reflections :
  ∃ (n : ℕ), n = 25 ∧ (1 + 2 * n) ≥ start_distance :=
by
  existsi 25
  simp
  sorry

-- Prove that point A cannot be moved inside the circle with only 24 reflections
theorem point_outside_after_24_reflections :
  ∀ (n : ℕ), n = 24 → (1 + 2 * n) < start_distance :=
by
  intros n hn
  rw hn
  simp
  sorry

end point_inside_after_25_reflections_point_outside_after_24_reflections_l161_161739


namespace min_distance_zero_l161_161002

variable (U g τ : ℝ)

def y₁ (t : ℝ) : ℝ := U * t - (g * t^2) / 2
def y₂ (t : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2
def s (t : ℝ) : ℝ := |U * τ - g * t * τ + (g * τ^2) / 2|

theorem min_distance_zero
  (U g τ : ℝ)
  (h : 2 * U ≥ g * τ)
  : ∃ t : ℝ, t = τ / 2 + U / g ∧ s t = 0 := sorry

end min_distance_zero_l161_161002


namespace sin_order_l161_161787

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

end sin_order_l161_161787


namespace cylinder_volume_triple_l161_161420

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem cylinder_volume_triple (r : ℝ) (h h' : ℝ) (V : ℝ) 
  (hr : r = 8) (hh : h = 7) (hh' : h' = 21) (VV : V = volume_cylinder r h) :
  volume_cylinder r h' = 3 * V :=
by
  sorry

end cylinder_volume_triple_l161_161420


namespace unique_zero_function_l161_161885

noncomputable def f : ℤ → ℚ := sorry

axiom condition1 (x : ℤ) : f(x) ∈ ℚ

axiom condition2 {x y : ℤ} {c : ℚ} (hx : f(x) < c) (hy : c < f(y)) : ∃ z : ℤ, f(z) = c

axiom condition3 {x y z : ℤ} (h : x + y + z = 0) : f(x) + f(y) + f(z) = f(x) * f(y) * f(z)

theorem unique_zero_function : ∀ x : ℤ, f(x) = 0 :=
sorry

end unique_zero_function_l161_161885


namespace area_ABCD_l161_161914

structure Point where
  x : ℝ
  y : ℝ

structure Rectangle where
  A B C D : Point

def area (r : Rectangle) : ℝ :=
  let width := r.B.x - r.A.x
  let height := r.D.y - r.A.y
  width * height

def A := Point.mk 0 0
def B := Point.mk 2 0
def C := Point.mk 2 (2 * Real.sqrt 2)
def D := Point.mk 0 (2 * Real.sqrt 2)
def ABCD := Rectangle.mk A B C D

theorem area_ABCD : area ABCD = 4 * Real.sqrt 2 := by
  sorry

end area_ABCD_l161_161914


namespace walt_total_invested_l161_161282

-- Given Conditions
def invested_at_seven : ℝ := 5500
def total_interest : ℝ := 970
def interest_rate_seven : ℝ := 0.07
def interest_rate_nine : ℝ := 0.09

-- Define the total amount invested
noncomputable def total_invested : ℝ := 12000

-- Prove the total amount invested
theorem walt_total_invested :
  interest_rate_seven * invested_at_seven + interest_rate_nine * (total_invested - invested_at_seven) = total_interest :=
by
  -- The proof goes here
  sorry

end walt_total_invested_l161_161282


namespace smallest_positive_integer_ends_6996_l161_161669

theorem smallest_positive_integer_ends_6996 :
  ∃ m : ℕ, (m % 4 = 0 ∧ m % 9 = 0 ∧ ∀ d ∈ m.digits 10, d = 6 ∨ d = 9 ∧ m.digits 10 ∩ {6, 9} ≠ ∅ ∧ m % 10000 = 6996) :=
sorry

end smallest_positive_integer_ends_6996_l161_161669


namespace inequality_positive_real_xyz_l161_161683

theorem inequality_positive_real_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) := 
by
  -- Proof is to be constructed here
  sorry

end inequality_positive_real_xyz_l161_161683


namespace trajectory_eq_chord_length_AB_l161_161913

noncomputable def point_Q := (2 : ℝ, 0 : ℝ)

def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

def ratio_tangent_MQ (M : ℝ × ℝ) : Prop :=
  let (x, y) := M in
  ∃ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧
  Real.sqrt (x^2 + y^2 - 1) = Real.sqrt 2 * Real.sqrt ((x - 2)^2 + y^2)

theorem trajectory_eq :
  ∀ (M : ℝ × ℝ), ratio_tangent_MQ M → (M.1 - 4)^2 + M.2^2 = 7 :=
sorry

theorem chord_length_AB :
  ∀ (A B : ℝ × ℝ), 
    ratio_tangent_MQ A → 
    ratio_tangent_MQ B → 
    (∃ k : ℝ, A = (k + 2, k) ∧ B = (k + 2, k)) → 
    Real.sqrt 2 * (Real.sqrt ((A.1 - 4)^2 + A.2^2 + (B.1 - 4)^2 + B.2^2) / 2) = 2 * Real.sqrt 5 :=
sorry

end trajectory_eq_chord_length_AB_l161_161913


namespace pereskochizaborov_half_leaves_l161_161861

variable (V : ℝ)

-- Conditions provided
def bystrov_half_leaves := (1 / 10) * V
def shustrov_bystrov_half_leaves := (1 / 8) * V
def vostrov_shustrov_bystrov_half_leaves := (1 / 3) * V

-- Question to prove
theorem pereskochizaborov_half_leaves : 
  bystrov_half_leaves * 2 + (shustrov_bystrov_half_leaves - bystrov_half_leaves) * 2 + 
  ((vostrov_shustrov_bystrov_half_leaves - shustrov_bystrov_half_leaves) - 
  (shustrov_bystrov_half_leaves - bystrov_half_leaves)) * 2 =  (1 / 3) * V → 
  (V - (bystrov_half_leaves * 2 + 
  (shustrov_bystrov_half_leaves - bystrov_half_leaves) * 2 + 
  ((vostrov_shustrov_bystrov_half_leaves - shustrov_bystrov_half_leaves) - 
  (shustrov_bystrov_half_leaves - bystrov_half_leaves)) * 2)) / 2 =  (1 / 6) * V :=
by 
  intros h_cons1 h_cons2 h_cons3
  -- skip proof steps
  sorry

end pereskochizaborov_half_leaves_l161_161861


namespace find_imaginary_part_l161_161268

theorem find_imaginary_part (z : ℂ) (h : complex.I * (z - 4) = 3 + 2 * complex.I) : z.im = 3 := 
sorry

end find_imaginary_part_l161_161268


namespace remainder_sum_mod_5_l161_161899

theorem remainder_sum_mod_5 :
  ((1^3 + 1) + (2^3 + 1) + (3^3 + 1) + ... + (50^3 + 1)) % 5 = 0 :=
sorry

end remainder_sum_mod_5_l161_161899


namespace percentage_of_a_is_4b_l161_161713

variable (a b : ℝ)

theorem percentage_of_a_is_4b (h : a = 1.2 * b) : 4 * b = (10 / 3) * a := 
by 
    sorry

end percentage_of_a_is_4b_l161_161713


namespace hyperbola_eccentricity_is_sqrt5_div_2_l161_161572

noncomputable def hyperbola_eccentricity (a : ℝ) (h_pos : a > 0) (h_asymptote_perpendicular : 1 / a = 1 / 2) : ℝ :=
  let b := 1 in
  let c := Real.sqrt (a^2 + b^2) in
  c / a

theorem hyperbola_eccentricity_is_sqrt5_div_2 (a : ℝ) (h_pos : a > 0) (h_asymptote_perpendicular : 1 / a = 1 / 2) :
  hyperbola_eccentricity a h_pos h_asymptote_perpendicular = Real.sqrt 5 / 2 := sorry

end hyperbola_eccentricity_is_sqrt5_div_2_l161_161572


namespace floor_of_neg_seven_fourths_l161_161486

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l161_161486


namespace compute_z_pow_7_l161_161243

namespace ComplexProof

noncomputable def z : ℂ := (Real.sqrt 3 + Complex.I) / 2

theorem compute_z_pow_7 : z ^ 7 = - (Real.sqrt 3 / 2) - (1 / 2) * Complex.I :=
by
  sorry

end ComplexProof

end compute_z_pow_7_l161_161243


namespace celeste_song_probability_l161_161069

theorem celeste_song_probability :
  let favorite_song_duration := 375
  let total_songs := 12
  let total_duration := 420
  let total_permutations := Nat.factorial total_songs
  let favorable_permutations := 3 * Nat.factorial (total_songs - 1)
  let probability := 1 - (favorable_permutations / total_permutations : ℚ)
  in probability = 3/4 := by
  sorry

end celeste_song_probability_l161_161069


namespace Clinton_belts_l161_161869

variable {Shoes Belts Hats : ℕ}

theorem Clinton_belts :
  (Shoes = 14) → (Shoes = 2 * Belts) → Belts = 7 :=
by
  sorry

end Clinton_belts_l161_161869


namespace minimum_distance_at_meeting_time_distance_glafira_to_meeting_l161_161000

variables (U g τ V : ℝ)
-- assumption: 2 * U ≥ g * τ
axiom h : 2 * U ≥ g * τ

noncomputable def motion_eq1 (t : ℝ) : ℝ := U * t - (g * t^2) / 2
noncomputable def motion_eq2 (t : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2

noncomputable def distance (t : ℝ) : ℝ := 
|motion_eq1 U g t - motion_eq2 U g τ t|

noncomputable def meeting_time : ℝ := (2 * U / g) + (τ / 2)

theorem minimum_distance_at_meeting_time : distance U g τ meeting_time = 0 := sorry

noncomputable def distance_from_glafira_to_meeting : ℝ := 
V * meeting_time

theorem distance_glafira_to_meeting : 
distance_from_glafira_to_meeting U g τ V = V * ((τ / 2) + (U / g)) := sorry

end minimum_distance_at_meeting_time_distance_glafira_to_meeting_l161_161000


namespace number_of_distinct_m_l161_161996

theorem number_of_distinct_m (a b : ℤ) (m : ℤ) :
  (a * b = -16) → 
  (a + b = m) → 
  ∃! n, { m | ∃ a b, (a * b = -16) ∧ (a + b = m) }.card = n ∧ n = 5 :=
by
  intros h_ab h_sum
  use 5
  sorry

end number_of_distinct_m_l161_161996


namespace train_crosses_pole_in_3_seconds_l161_161843

def train_speed_kmph : ℝ := 60
def train_length_m : ℝ := 50

def speed_conversion (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

def crossing_time (distance_m : ℝ) (speed_mps : ℝ) : ℝ := distance_m / speed_mps

theorem train_crosses_pole_in_3_seconds :
  crossing_time train_length_m (speed_conversion train_speed_kmph) = 3 :=
by
  sorry

end train_crosses_pole_in_3_seconds_l161_161843


namespace area_ratio_depend_l161_161662

variable {A B C M P D : Type} [LinearOrderedField A]

-- Define the midpoint condition
def midpoint (M A B : A) : Prop := 2 * M = A + B

-- Define the point P condition
def point_on_segment (P M B : A) : Prop := M < P ∧ P < B

-- Define the parallel condition
def parallel (MD PC BC : A) : Prop := true  -- Assume MD is parallel to PC which is parallel to BC

-- Ratio of areas
def area_ratio (BPD ABC : A) (r x : A) : Prop :=
  r = (x * x - (1 / 4) * (A * B) * (A * B)) / (A * B * C)

-- Main theorem statement
theorem area_ratio_depend (A B C M P D : A) (x : A) :
  midpoint M A B → point_on_segment P M B → parallel D P C →
  ∃ r, area_ratio (B * P * D) (A * B * C) r x ↔ r = x :=
sorry -- Proof not required

end area_ratio_depend_l161_161662


namespace odd_function_expr_for_positive_x_l161_161458

noncomputable def f : ℝ → ℝ
| x => if x < 0 then x^2 - x else -x^2 - x

theorem odd_function_expr_for_positive_x (x : ℝ) (h1 : f x = x^2 - x ∧ x < 0) :
  ∀ x, x > 0 → f x = -x^2 - x :=
  by
    -- Assuming x > 0
    assume x hx,
    -- -x < 0
    have h2 : -x < 0 := by linarith,
    -- Using f(-x)
    have h3 : f (-x) = (-x)^2 - (-x) := by assumption,
    -- Simplify: f(-x) = x^2 + x
    rw [neg_sq x, neg_neg x] at h3,
    -- Since f is odd, f(x) = -f(-x)
    have h4 : f x = - (f (-x)) := by sorry,
    -- Therefore f(x) = - (x^2 + x)
    rw [h3] at h4,
    -- Hence f(x) = - x^2 - x
    exact h4

-- auxiliary simplification lemma
lemma neg_sq (x : ℝ) : (-x)^2 = x^2 :=
  by
    ring

lemma neg_neg (x : ℝ) : -(-x) = x :=
  by
    ring

end odd_function_expr_for_positive_x_l161_161458


namespace find_complex_number_l161_161577

open Complex

theorem find_complex_number (a b : ℝ) (z : ℂ) 
  (h₁ : (∀ b: ℝ, (b^2 + 4 * b + 4 = 0) ∧ (b + a = 0))) :
  z = 2 - 2 * Complex.I :=
  sorry

end find_complex_number_l161_161577


namespace sqrt_10_integer_decimal_partition_l161_161155

theorem sqrt_10_integer_decimal_partition:
  let a := Int.floor (Real.sqrt 10)
  let b := Real.sqrt 10 - a
  (Real.sqrt 10 + a) * b = 1 :=
by
  sorry

end sqrt_10_integer_decimal_partition_l161_161155


namespace horizontal_flip_of_f_l161_161947

-- Define the function f for the given intervals
def f (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then -3 - x
  else if -1 ≤ x ∧ x ≤ 1 then -√(4 - x^2) - 3
  else if 1 ≤ x ∧ x ≤ 4 then 2 * x - 6
  else 0  -- default case outside the given ranges

-- The function g as f(-x)
def g (x : ℝ) : ℝ :=
  f (-x)

-- The expected transformed function f(-x) 
def f_neg_x (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 4 then -3 + x
  else if -1 ≤ x ∧ x ≤ 1 then -√(4 - x^2) - 3
  else if -4 ≤ x ∧ x ≤ -1 then -2 * x - 6
  else 0  -- default case outside the given ranges

-- The goal is to prove that g(x) == f_neg_x(x) for all x in the domain of f
theorem horizontal_flip_of_f (x : ℝ) :
  g x = f_neg_x x :=
by
  sorry

end horizontal_flip_of_f_l161_161947


namespace probability_two_boys_l161_161864

-- Definitions for the conditions
def total_students : ℕ := 4
def boys : ℕ := 3
def girls : ℕ := 1
def select_students : ℕ := 2

-- Combination function definition
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_boys :
  (combination boys select_students) / (combination total_students select_students) = 1 / 2 := by
  sorry

end probability_two_boys_l161_161864


namespace simplify_cos_difference_l161_161298

noncomputable def cos (x : ℝ) : ℝ := real.cos x

def c := cos (20 * real.pi / 180)  -- cos(20°)
def d := cos (40 * real.pi / 180)  -- cos(40°)

theorem simplify_cos_difference :
  c - d =
  -- The expression below is placeholder; real expression involves radicals and squares
  sorry :=
by
  let c := cos (20 * real.pi / 180)
  let d := cos (40 * real.pi / 180)
  have h1 : d = 2 * c^2 - 1 := sorry
  let sqrt3 : ℝ := real.sqrt 3
  have h2 : c = (1 / 2) * d + (sqrt3 / 2) * real.sqrt (1 - d^2) := sorry
  sorry

end simplify_cos_difference_l161_161298


namespace vector_magnitude_not_in_specific_intervals_l161_161564

variables {ℝ : Type*} [normed_field ℝ] {λ : ℝ} 
variables {a b c : ℝ^2}
variables [fact (abs a = 1)] [fact (abs b = 2)] [fact (abs c = 3)]

noncomputable def magnitude (x : ℝ^2) : ℝ := abs x

theorem vector_magnitude_not_in_specific_intervals
  (h1 : magnitude a = 1)
  (h2 : magnitude b = 2)
  (h3 : magnitude c = 3)
  (h4 : 0 < λ ∧ λ < 1)
  (h5 : (b • c) = 0) :
  ¬ (∃ r, r ∈ Ιoo (-∞) ((6 : ℝ)/sqrt (13 : ℝ)-1) ∪ Ιoo (4) (∞) ∧ r = magnitude (a - λ • b - (1-λ) • c)) :=
sorry

end vector_magnitude_not_in_specific_intervals_l161_161564


namespace find_t_l161_161182

open_locale big_operators

def vec2 := (ℝ × ℝ)

def dot_product (u v : vec2) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def scalar_mult (t : ℝ) (u : vec2) : vec2 :=
  (t * u.1, t * u.2)

def vec_add (u v : vec2) : vec2 :=
  (u.1 + v.1, u.2 + v.2)

def a : vec2 := (1, -1)
def b : vec2 := (6, -4)

theorem find_t (t : ℝ) (h : dot_product a (vec_add (scalar_mult t a) b) = 0) : 
  t = -5 :=
by sorry

end find_t_l161_161182


namespace max_possible_value_of_k_l161_161969

noncomputable def max_k (x y : ℝ) (h : x > 0 ∧ y > 0) := ( -1 + Real.sqrt 56 ) / 2

theorem max_possible_value_of_k (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^2 * ((x / y)^2 + (y / x)^2) + 2 * k * (x / y + y / x)) :
  k ≤ max_k x y h_pos :=
sorry

end max_possible_value_of_k_l161_161969


namespace PQ_equals_2HP_l161_161988

noncomputable theory

-- Definitions of points and properties
variables (A B C D H P Q : Point)
variables [Tetrahedron A B C D]
variables (alt_conc : altitudes_concurrent A B C D H)
variables (DH_intersects_P : ∃ P, lies_on P (plane_of A B C) ∧ (line DH).intersects_at P)
variables (DH_intersects_Q : ∃ Q, (circumsphere A B C D).intersects_at Q ∧ Q ≠ D)

-- The statement we want to prove
theorem PQ_equals_2HP : dist Q P = 2 * dist H P :=
sorry

end PQ_equals_2HP_l161_161988


namespace alcohol_concentration_bound_l161_161957

noncomputable def operation (x y z : ℝ) (a_n : ℕ → ℝ) : ℕ → ℝ
| 0     := 0  -- assume a_0 = 0 for initial condition
| (n+1) :=
  let b_n := min (x + y - a_n n) z in  -- volume in B after pour from A
  let a := min (x + b_n - z) x in  -- volume in A after pour back from B
  a

theorem alcohol_concentration_bound (x y z : ℝ) (h1 : x < z) (h2 : y < z) :
  ∀ n : ℕ, operation x y z (λ _, 0) n ≤ (x * y) / (x + y) :=
begin
  sorry
end

end alcohol_concentration_bound_l161_161957


namespace log_sum_of_a5_a7_a9_l161_161951

-- Define the sequence {a_n}
variable {a : ℕ → ℝ}

-- Conditions of the problem
def condition1 (n : ℕ) : Prop := ∀ n : ℕ, 0 < n → log 3 (a n) + 1 = log 3 (a (n+1))
def condition2 : Prop := a 2 + a 4 + a 6 = 9

-- The Lean statement that we need to prove
theorem log_sum_of_a5_a7_a9 (h1 : condition1) (h2 : condition2) :
  log 3 (a 5 + a 7 + a 9) = 5 :=
by
  sorry

end log_sum_of_a5_a7_a9_l161_161951


namespace remainder_modulo_12_l161_161767

theorem remainder_modulo_12 
  (a b c d : ℕ) 
  (ha : a < 12) (hb : b < 12) (hc : c < 12) (hd : d < 12)
  (ha_ne : a ≠ b) (hb_ne : b ≠ c) (hc_ne : c ≠ d) (hd_ne : a ≠ d)
  (ha_gcd : Nat.gcd a 12 = 1) (hb_gcd : Nat.gcd b 12 = 1)
  (hc_gcd : Nat.gcd c 12 = 1) (hd_gcd : Nat.gcd d 12 = 1) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (a * b * c * d)⁻¹) % 12 = 0 :=
by
  sorry

end remainder_modulo_12_l161_161767


namespace unique_triplet_l161_161530

theorem unique_triplet (a b p : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) :
  (1 / (p : ℚ) = 1 / (a^2 : ℚ) + 1 / (b^2 : ℚ)) → (a = 2 ∧ b = 2 ∧ p = 2) :=
by
  sorry

end unique_triplet_l161_161530


namespace parabola_equation_and_directrix_l161_161557

theorem parabola_equation_and_directrix (x y : ℝ) :
  vertex_at_origin : (0, 0) ∧
  axis_of_symmetry_coord_axis : (x = 0 ∨ y = 0) ∧
  passes_through : (-3, 2) → 
  (y^2 = - (4 / 3) * x ∨ x^2 = (9 / 2) * y) ∧ 
  (x = (1 / 3) ∨ y = - (9 / 8)) :=
by sorry

end parabola_equation_and_directrix_l161_161557


namespace max_g_on_1_3_l161_161931

noncomputable theory

def f (x : Real) : Real := x ^ (-2)

def g (x : Real) : Real := (x - 1) * f x

theorem max_g_on_1_3 : 
  ∃ x : Real, x ∈ set.Icc 1 3 ∧ g x = (1 / 4) := 
sorry

end max_g_on_1_3_l161_161931


namespace distance_between_parallel_lines_l161_161367

theorem distance_between_parallel_lines (O : Point) (A B C D P Q : Point)
  (circle_intersects_lines : ∀ (X Y : Point), X ≠ Y → Circle O (distance O X) intersects (Line X Y))
  (length_AB : dist A B = 40)
  (length_BC : dist B C = 36)
  (length_CD : dist C D = 40)
  (midpoints : P = midpoint A B ∧ Q = midpoint C D)
  (chord_distances : ∀ (X Y : Point), X ≠ Y → ∃ d : ℝ , d = distance_between_parallel_lines X Y):
  d = 2 := 
sorry

end distance_between_parallel_lines_l161_161367


namespace floor_of_neg_seven_fourths_l161_161488

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l161_161488


namespace max_unique_three_digit_numbers_l161_161548

theorem max_unique_three_digit_numbers : 
  ∃ (s : Finset ℕ), (∀ n ∈ s, ∃ a b c : ℕ, a ∈ {6, 7, 8, 9} ∧ b ∈ {6, 7, 8, 9} ∧ c ∈ {6, 7, 8, 9} ∧ n = a * 100 + b * 10 + c) ∧ 
    (∀ n m ∈ s, n ≠ m → (n / 10) % 10 ≠ m % 10 ∧ n % 100 ≠ (m / 10) % 10) ∧ 
    s.card = 40 :=
sorry

end max_unique_three_digit_numbers_l161_161548


namespace height_relationship_l161_161372

-- Define the variables and conditions
variables {r1 r2 h1 h2 : ℝ}

-- Theorem statement
theorem height_relationship
  (h_volume : π * r1^2 * h1 = π * r2^2 * h2)
  (h_radius : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
begin
  sorry
end

end height_relationship_l161_161372


namespace floor_neg_seven_quarter_l161_161482

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l161_161482


namespace part1_part2_l161_161158

variable (S : ℕ → ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)

-- Given conditions
axiom sum_first_n : ∀ n : ℕ, S n = (finset.range n).sum a
axiom a1 : a 1 = 2
axiom a3 : a 3 = 8
axiom sum_relation : ∀ n : ℕ, n > 0 → S (n + 2) + S n = 2 * S (n + 1) + 3

-- First part: proving that the sequence is arithmetic and finding its general term
theorem part1 : (∀ n : ℕ, a n = 3 * n - 1) := sorry

-- Second part: finding the sum of the first n terms of the sequence b_n
def b : ℕ → ℕ := λ n, a n * 2^(n)
def T : ℕ → ℕ := λ n, (finset.range n).sum b

theorem part2 : ∀ n : ℕ, T n = (3 * n - 4) * 2^(n+1) + 8 := sorry

end part1_part2_l161_161158


namespace root_abs_lt_one_l161_161376

theorem root_abs_lt_one (a b : ℝ) (h1 : abs a + abs b < 1) (h2 : a^2 - 4 * b ≥ 0) :
  ∀ x1 x2 : ℝ, (x1^2 + a * x1 + b = 0 ∧ x2^2 + a * x2 + b = 0) →
  abs x1 < 1 ∧ abs x2 < 1 :=
by
  assume (hx : ∃ x1 x2 : ℝ, (x1^2 + a * x1 + b = 0 ∧ x2^2 + a * x2 + b = 0))
  have h : ¬ (abs x1 ≥ 1 ∨ abs x2 ≥ 1) → (abs x1 < 1 ∧ abs x2 < 1) 
  from sorry
  exact sorry

end root_abs_lt_one_l161_161376


namespace eval_floor_neg_seven_fourths_l161_161506

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l161_161506


namespace prove_CM_CN_constant_value_and_locus_of_C_minimum_area_of_isosceles_triangle_PQR_l161_161267

noncomputable def center_of_circle : ℝ × ℝ :=
(1, 0)

noncomputable def radius_of_circle : ℝ :=
4

def locus_of_C (x y : ℝ) : Prop :=
(x^2 / 4) + (y^2 / 3) = 1 ∧ y ≠ 0

theorem prove_CM_CN_constant_value_and_locus_of_C :
  ∀ C : ℝ × ℝ,
  let M := center_of_circle,
      N := (-1, 0 : ℝ × ℝ) in
  (locus_of_C C.1 C.2) →
  (dist (C.1, C.2) M + dist (C.1, C.2) N = 4) :=
begin
  sorry
end

noncomputable def intersection_points (k : ℝ) : set (ℝ × ℝ) :=
{x | ∃ x₁ y₁, x = (x₁, y₁) ∧ y₁ = k * x₁ ∧ locus_of_C x₁ y₁}

theorem minimum_area_of_isosceles_triangle_PQR :
  ∀ P Q R : ℝ × ℝ,
  ∀ k : ℝ,
  P ∈ intersection_points k →
  Q ∈ intersection_points k →
  (∃ (R : ℝ × ℝ), locus_of_C R.1 R.2 ∧
  (dist R P = dist R Q ∧ dist P Q ≠ 0)) →
  ∃ A : ℝ, A = (24 / 7) :=
begin
  sorry
end

end prove_CM_CN_constant_value_and_locus_of_C_minimum_area_of_isosceles_triangle_PQR_l161_161267


namespace domain_of_f_l161_161884

def f (x : ℝ) : ℝ := real.sqrt (4 - real.sqrt (6 - real.sqrt (7 - real.sqrt x)))

theorem domain_of_f :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 49 ↔ ∃ y : ℝ, y = f x :=
begin
  sorry
end

end domain_of_f_l161_161884


namespace f_ff_neg4_l161_161555

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then f (x + 1) else x^2 - 3 * x - 4

theorem f_ff_neg4 : f (f (-4)) = -6 :=
by {
  -- This is where the proof would go
  sorry
}

end f_ff_neg4_l161_161555


namespace am_gm_inequality_three_vars_l161_161956

theorem am_gm_inequality_three_vars (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) :
  (a + b + c) / 3 ≥ Real.cbrt (a * b * c) := 
sorry

end am_gm_inequality_three_vars_l161_161956


namespace comparison_five_and_two_sqrt_five_l161_161871

theorem comparison_five_and_two_sqrt_five : 5 > 2 * real.sqrt 5 := by
  sorry

end comparison_five_and_two_sqrt_five_l161_161871


namespace proof_p_and_q_l161_161922

variables {x m : ℝ} {A B : ℝ}

-- Definition for proposition p
def prop_p := ∀ x, x^2 + x + m > 0 → m > 1/4

-- Definition for proposition q
def prop_q (A B : ℝ) [triangle_ABC : (A > B ↔ real.sin A > real.sin B)] := true

-- Theorem to prove proposition p and q are true
theorem proof_p_and_q (h1 : ∀ x, x^2 + x + m > 0)
                      (h2 : ∀ {A B : ℝ}, A > B ↔ real.sin A > real.sin B) : prop_p x m ∧ prop_q A B := 
  sorry

end proof_p_and_q_l161_161922


namespace frog_jump_plan_l161_161673

-- Define the vertices of the hexagon
inductive Vertex
| A | B | C | D | E | F

open Vertex

-- Define adjacency in the regular hexagon
def adjacent (v1 v2 : Vertex) : Prop :=
  match v1, v2 with
  | A, B | A, F | B, C | B, A | C, D | C, B | D, E | D, C | E, F | E, D | F, A | F, E => true
  | _, _ => false

-- Define the problem
def frog_jump_sequences_count : ℕ :=
  26

theorem frog_jump_plan : frog_jump_sequences_count = 26 := 
  sorry

end frog_jump_plan_l161_161673


namespace deductive_reasoning_l161_161313

def statement_A (names_correct : Prop) (language_accords_truth: Prop)
                (affairs_success: Prop) (rituals_music_flourish: Prop)
                (punishments_proper: Prop) (people_proper: Prop) : Prop :=
  (¬ names_correct → ¬ language_accords_truth) ∧
  (¬ language_accords_truth → ¬ affairs_success) ∧
  (¬ affairs_success → ¬ rituals_music_flourish) ∧
  (¬ rituals_music_flourish → ¬ punishments_proper) ∧
  (¬ punishments_proper → ¬ people_proper) →
  (¬ names_correct → ¬ people_proper)

theorem deductive_reasoning (h: statement_A names_correct language_accords_truth affairs_success rituals_music_flourish punishments_proper people_proper) :
  (¬ names_correct → ¬ people_proper) :=
begin
  sorry
end

end deductive_reasoning_l161_161313


namespace symmetric_point_reflection_y_axis_l161_161722

theorem symmetric_point_reflection_y_axis (x y : ℝ) (h : (x, y) = (-2, 3)) :
  (-x, y) = (2, 3) :=
sorry

end symmetric_point_reflection_y_axis_l161_161722


namespace parallelogram_area_l161_161532

open Vector
open Real

-- Definitions
def u : ℝ³ := ⟨4, -1, 3⟩
def v : ℝ³ := ⟨-2, 2, 5⟩

-- Theorem statement
theorem parallelogram_area :
  let cross_product := cross u v in
  let magnitude := norm cross_product in
  magnitude = sqrt 833 :=
sorry

end parallelogram_area_l161_161532


namespace triangle_colors_l161_161803

noncomputable def color_tiles (n : ℕ) (color : ℕ × ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), i < n ∧ j < n →
    (color (i, j) = 0 → ∃ (count : ℕ), count % 2 = 0 ∧
    ∀ k l, (color (k, l) = 1 → ((k = i ± 1 ∧ l = j) ∨ (k = i ∧ l = j ± 1)))) ∧
  (color (i, j) = 1 → ∃ (count : ℕ), count % 2 = 1 ∧
    ∀ k l, (color (k, l) = 1 → ((k = i ± 1 ∧ l = j) ∨ (k = i ∧ l = j ± 1))))

theorem triangle_colors (n : ℕ) :
  ∃ (color : ℕ × ℕ → Prop), color_tiles n color ∧
  (color (0, 0) = color (n-1, 0)) ∧
  (color (0, 0) = color (n-1, n-1)) :=
sorry

end triangle_colors_l161_161803


namespace alternating_series_10000_sum_l161_161888

def alternating_sum_change_at_perfect_squares (n : ℕ) : ℤ :=
  let sum_term (k : ℕ) : ℕ := k * ((-1) ^ (nat.floor (real.sqrt k)).to_nat)
  ∑ i in finset.range n, sum_term i

theorem alternating_series_10000_sum : alternating_sum_change_at_perfect_squares 10000 = 1000000 := 
by sorry

end alternating_series_10000_sum_l161_161888


namespace floor_of_neg_seven_fourths_l161_161487

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l161_161487


namespace sum_of_first_n_terms_l161_161198

noncomputable def sequence_sum (x : ℝ) (n : ℕ) : ℝ :=
  if x = 0 then 0 else
  if x = 1 then n else
  x * (1 - x^n) / (1 - x)

theorem sum_of_first_n_terms (x : ℝ) (n : ℕ) :
  (∑ i in Finset.range n, x^i) = sequence_sum x n :=
by sorry

end sum_of_first_n_terms_l161_161198


namespace floor_neg_seven_fourths_l161_161472

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l161_161472


namespace hyperbola_integer_points_count_l161_161600

-- Definition of the hyperbolic equation
def hyperbola (x y : ℤ) : Prop :=
  y * x = 2013

-- Condition: We are looking for integer coordinate points (x, y)
def integer_coordinate_points : Set (ℤ × ℤ) :=
  {p | hyperbola p.fst p.snd}

-- Main proof statement
theorem hyperbola_integer_points_count : (integer_coordinate_points.to_finset.card = 16) :=
sorry

end hyperbola_integer_points_count_l161_161600


namespace hotel_charges_l161_161688

variables {R G P S T : ℝ}

-- Conditions
def cond1 : P = 0.75 * R := sorry
def cond2 : P = 0.90 * G := sorry
def cond3 : S = 1.15 * R := sorry
def cond4 : T = 0.80 * G := sorry

-- Proof problem
theorem hotel_charges (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) :
  S = 1.5333 * P ∧
  T = 0.8888 * P ∧
  (R - G) / G * 100 = 18 :=
by sorry

end hotel_charges_l161_161688


namespace find_function_form_l161_161007

noncomputable def target_function_form : Prop := 
  ∀ (x_0 y_0 : ℝ), 
  (log (x_0 - x_0^2 + 3) (y_0 - 6) = 
      log (x_0 - x_0^2 + 3) 
        ((|2*x_0 + 6| - |2*x_0 + 3|) / (3*x_0 + 7.5) * sqrt(x_0^2 + 5*x_0 + 6.25))) 
  → (∀ x, y = -0.05 * (x + 2)^2 + 2)

theorem find_function_form : target_function_form := sorry

end find_function_form_l161_161007


namespace alan_carla_weight_l161_161847

variable (a b c d : ℝ)

theorem alan_carla_weight (h1 : a + b = 280) (h2 : b + c = 230) (h3 : c + d = 250) (h4 : a + d = 300) :
  a + c = 250 := by
sorry

end alan_carla_weight_l161_161847


namespace floor_neg_seven_over_four_l161_161496

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l161_161496


namespace prime_square_subtract_divisible_l161_161206

theorem prime_square_subtract_divisible {p : ℕ} (hp_prime : p.prime) (hp_gt_3 : p > 3) : 
  4 * p^2 - 100 ≡ 0 [MOD 96] :=
sorry

end prime_square_subtract_divisible_l161_161206


namespace proof_area_rectangle_l161_161288

noncomputable def area_of_rectangle {A B C D E F G : Type} 
  (AD EG : ℝ)
  (altitude EF: ℝ) 
  (AB_half_AD : ℝ → Prop) : 
  Prop :=
AB_half_AD (AD / 2) →
(EG = 12) →
(altitude EF = 8) →
(area ABCD = 288 / 25)

axiom given_conditions 
  {A B C D : Type} {E F G : Type}
  (altitude_E_to_G : ℝ)
  (EG : ℝ)
  (AD_half_AB : Prop)
  : AD_half_AB (λ AD, design(float AD / 2)) →
    (EG = 12) →
    (altitude_E_to_G = 8)

theorem proof_area_rectangle 
  : ∀ {A B C D E F G : Type} 
   {E_to_G : ℝ} 
   {EG_side : ℝ} 
   (half_relation : ℝ → Prop),
   given_conditions E_to_G EG_side half_relation →
   area_of_rectangle A B C D E F G (AD) EG_side E_to_G half_relation :=
begin
  sorry
end

end proof_area_rectangle_l161_161288


namespace count_triangles_including_center_l161_161916

theorem count_triangles_including_center (n : ℕ) :
  let k := 2 * n + 1 in
  (k * n * (n + 1)) / 6 = n * (n + 1) * (2 * n + 1) / 6 :=
by
  let k := 2 * n + 1
  have : (k * n * (n + 1)) = n * (n + 1) * k, by sorry
  rw this
  rfl

end count_triangles_including_center_l161_161916


namespace sum_of_abs_arithmetic_sequence_l161_161352

theorem sum_of_abs_arithmetic_sequence {a_n : ℕ → ℤ} {S_n : ℕ → ℤ} 
  (hS3 : S_n 3 = 21) (hS9 : S_n 9 = 9) :
  ∃ (T_n : ℕ → ℤ), 
    (∀ (n : ℕ), n ≤ 5 → T_n n = -n^2 + 10 * n) ∧
    (∀ (n : ℕ), n ≥ 6 → T_n n = n^2 - 10 * n + 50) :=
sorry

end sum_of_abs_arithmetic_sequence_l161_161352


namespace length_AP_l161_161642

-- Definitions of the problem setup
structure Square :=
(side_length : ℕ)
(vertices : list (ℕ × ℕ)) -- simplified representation

structure Rectangle :=
(length : ℕ)
(width : ℕ)
(vertices : list (ℕ × ℕ)) -- simplified representation

-- Problem Definition
def ABCD : Square := {side_length := 8, vertices := [(0,0), (8,0), (8,8), (0,8)]}
def WXYZ : Rectangle := {length := 12, width := 8, vertices := [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]} -- coordinates are unspecified

-- Given conditions
def area_WXYZ := WXYZ.length * WXYZ.width
def shaded_area := area_WXYZ / 3

-- Hypothesis: assuming AD and WX are perpendicular, shaded region's width along AD is the same as the side length of the square
def width_AD := ABCD.side_length
def PD := shaded_area / width_AD

-- Theorem Statement
theorem length_AP : 8 - PD = 4 := by
  sorry

end length_AP_l161_161642


namespace shadow_boundary_l161_161833

theorem shadow_boundary (x : ℝ) : 
  let r := 2
  let center := (0 : ℝ, 0 : ℝ, 2 : ℝ)
  let light_source := (0 : ℝ, -2 : ℝ, 3 : ℝ)
  let g (x : ℝ) := -2 - real.sqrt (4 - x^2)
  g x = -2 - sqrt (4 - x^2) :=
sorry

end shadow_boundary_l161_161833


namespace floor_neg_seven_quarter_l161_161480

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l161_161480


namespace AT_bisects_MK_l161_161246

theorem AT_bisects_MK
  {A B C M R S U T D N K : Point}
  (hTriangle : Triangle A B C)
  (hM_midpoint_BC : M = midpoint B C)
  (hR_on_circumcircle : ∃ ℓ : Line, on_circumcircle A B C ℓ ∧ AM ∈ ℓ ∧ M ∈ ℓ)
  (hRS_parallel_BC : parallel (line_through R S) (line_through B C))
  (hU_foot_R_BC : U = foot R B C)
  (hT_reflection_U_R : T = reflection U R)
  (hAD_altitude : D = altitude A B C)
  (hN_midpoint_AD : N = midpoint A D)
  (hASN_inter_MN_K : meets_at (line_through A S) (line_through M N) K) :
  bisects_line AT MK :=
sorry

end AT_bisects_MK_l161_161246


namespace acute_triangle_pyramid_exists_l161_161706

theorem acute_triangle_pyramid_exists :
  ∀ (A B C : EuclideanGeometry.Point 3), 
  EuclideanGeometry.angled_triangle A B C → 
  (∀ (SA SB SC : EuclideanGeometry.Line 3), EuclideanGeometry.perpendicular SA SB ∧ 
  EuclideanGeometry.perpendicular SB SC ∧ EuclideanGeometry.perpendicular SC SA) → 
  ∃ (S : EuclideanGeometry.Point 3), 
  EuclideanGeometry.triangular_pyramid S A B C :=
by
  sorry

end acute_triangle_pyramid_exists_l161_161706


namespace integer_points_on_hyperbola_l161_161604

theorem integer_points_on_hyperbola : 
  let points := {(x, y) : Int × Int | y * x = 2013} in points.size = 16 :=
by
  sorry

end integer_points_on_hyperbola_l161_161604


namespace train_crosses_pole_in_3_seconds_l161_161836

def train_problem (speed_kmh : ℕ) (length_m : ℕ) : ℕ :=
  let speed_ms := (speed_kmh * 1000) / 3600 in
  length_m / speed_ms

theorem train_crosses_pole_in_3_seconds :
  train_problem 60 50 = 3 :=
by
  -- We add a 'sorry' to skip the proof
  sorry

end train_crosses_pole_in_3_seconds_l161_161836


namespace sufficient_not_necessary_l161_161142

def M : Set Int := {0, 1, 2}
def N : Set Int := {-1, 0, 1, 2}

theorem sufficient_not_necessary (a : Int) : a ∈ M → a ∈ N ∧ ¬(a ∈ N → a ∈ M) := by
  sorry

end sufficient_not_necessary_l161_161142


namespace function_solution_l161_161252

theorem function_solution (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (f = (λ x, 0) ∨ ∃ c : ℤ, f = λ x, 2 * x + c) :=
by
  sorry

end function_solution_l161_161252


namespace particle_speed_is_sqrt_13_l161_161031

def position (t : ℝ) : ℝ × ℝ := (3 * t + 1, -2 * t + 5)

theorem particle_speed_is_sqrt_13 : 
  ∃ v : ℝ, (∀ t : ℝ, v = Real.sqrt (9 + 4)) :=
begin
  use Real.sqrt 13,
  intros t,
  sorry
end

end particle_speed_is_sqrt_13_l161_161031


namespace standard_equation_of_ellipse_line_pq_fixed_point_l161_161920

-- Definitions from the conditions
def vertex (E : Ellipse) : Point := (0, 1)
def focal_length (E : Ellipse) : ℝ := 2 * Real.sqrt 3
def vertex_A (E : Ellipse) : Point := (-2, 0)
def vertex_B (E : Ellipse) : Point := (2, 0)

-- Definition of Ellipse with the given standard form
structure Ellipse where
  a b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  a_gt_b : b < a
  eqn : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → True

-- Definition of a point and line segment
structure Point where
  x : ℝ
  y : ℝ

-- Theorem to prove the standard equation of the ellipse
theorem standard_equation_of_ellipse (E : Ellipse) (a b : ℝ) 
  (h1 : vertex E = (0, 1)) 
  (h2 : focal_length E = 2 * Real.sqrt 3) : 
  E.eqn = λ x y, x^2 / 4 + y^2 = 1 := 
  sorry

-- Theorem to prove the line PQ passes through fixed point (1, 0)
theorem line_pq_fixed_point (E : Ellipse) (P Q : Point) (T : Point) 
  (h1 : vertex_A E = (-2, 0)) 
  (h2 : vertex_B E = (2, 0))
  (h3 : ∀ P ≠ vertex_A E ∧ P ≠ vertex_B E, ∃ T, T.x = 4) 
  (h4 : ∃ Q, Q ∈ E ∧ Q ≠ P) : 
  line_pq_passes_through_fixed_point E P Q (1, 0) :=
  sorry

end standard_equation_of_ellipse_line_pq_fixed_point_l161_161920


namespace positive_solutions_eq_one_l161_161538

theorem positive_solutions_eq_one : 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ sin (arccos( cot (arccos x))) = x) → 
  (∃ unique x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ sin (arccos( cot (arccos x))) = x) := 
sorry

end positive_solutions_eq_one_l161_161538


namespace imaginary_part_of_fraction_l161_161148

noncomputable def complex_imaginary_part : ℂ → ℝ
| ⟨x, y⟩ := y

theorem imaginary_part_of_fraction :
  complex_imaginary_part ((1 + 2 * complex.i) / complex.i) = -1 :=
sorry

end imaginary_part_of_fraction_l161_161148


namespace cyclic_quadrilateral_properties_l161_161151

variables {α : Type*} [linear_ordered_field α]

/-- Given that quadrilateral ABCD is inscribed in a circle (cyclic quadrilateral),
proves the properties of angles in the quadrilateral. --/
theorem cyclic_quadrilateral_properties 
  {A B C D : α} (inscribed : cyclic_quadrilateral A B C D) :
  (∀ (A C : α), A = C → A = 90) ∧ (∀ (A_ext C_ext : α), A_ext + C_ext = 180) :=
sorry

end cyclic_quadrilateral_properties_l161_161151


namespace bill_difference_l161_161070

-- Define the parameters for Christine and Alex
variable (c a : ℝ)
variable (hChristine : 0.15 * c = 3)
variable (hAlex : 0.10 * a = 4)

theorem bill_difference : a - c = 20 :=
by 
  -- Calculate Christine's and Alex's bills from the given conditions
  let hc := (show 0.15 * c = 3, from hChristine)
  have hc_solved : c = 20 := by sorry

  let ha := (show 0.10 * a = 4, from hAlex)
  have ha_solved : a = 40 := by sorry

  -- Prove the difference is 20
  sorry

end bill_difference_l161_161070


namespace robot_returns_to_starting_point_after_6_minutes_l161_161418

-- Definitions
def constant_speed : Prop := true
def turn_90_degrees_every_15_seconds : Prop := true
def moves_straight_between_turns : Prop := true

-- Theorem
theorem robot_returns_to_starting_point_after_6_minutes
    (h1 : constant_speed)
    (h2 : turn_90_degrees_every_15_seconds)
    (h3 : moves_straight_between_turns) : 
    ∃ t : ℕ, t = 6 * 60 ∧ 
    true := 
begin
  -- We still need to formally prove that the robot indeed returns to its
  -- starting point in 6 minutes, but for now, we'll mark it as 'sorry'.
  sorry
end

end robot_returns_to_starting_point_after_6_minutes_l161_161418


namespace kelsey_travel_time_l161_161239

-- Define the constants used in the conditions
def total_distance : ℝ := 400
def speed_first_half : ℝ := 25
def speed_second_half : ℝ := 40

-- Define the times taken for each half of the journey
def time_first_half : ℝ := (total_distance / 2) / speed_first_half
def time_second_half : ℝ := (total_distance / 2) / speed_second_half

-- Prove that the total travel time is 13 hours
theorem kelsey_travel_time : time_first_half + time_second_half = 13 := by
  -- The proof steps will be placed here
  sorry

end kelsey_travel_time_l161_161239


namespace problem1_problem2_problem3_problem4_l161_161711

-- Problem 1
theorem problem1 (x : ℝ) : 0.75 * x = (1 / 2) * 12 → x = 8 := 
by 
  intro h
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (0.7 / x) = (14 / 5) → x = 0.25 := 
by 
  intro h
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (1 / 6) * x = (2 / 15) * (2 / 3) → x = (8 / 15) := 
by 
  intro h
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : 4.5 * x = 4 * 27 → x = 24 := 
by 
  intro h
  sorry

end problem1_problem2_problem3_problem4_l161_161711


namespace solution_l161_161923

variables (A B C M N : Type)
variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (a b : V)
variables [Vec : VectorSpace ℝ V]

variables {AB AC BM CN BC CA MN : V}

def triangle_condition (AB AC BM CN BC CA : V) : Prop :=
  BM = (1/3) • BC ∧ CN = (1/3) • CA ∧ AB = a ∧ AC = b ∧
  BC = b - a ∧ CA = b

def vector_decomposition (MN : V) (r s : ℝ) : Prop :=
  MN = r • a + s • b

theorem solution (h1 : triangle_condition a b BM CN BC CA)
                (h2 : vector_decomposition MN (-2/3) (1/3)) :
  (let r := -2/3, s := 1/3 in r - s = -1) :=
by
  sorry

end solution_l161_161923


namespace inequality_sum_pos_l161_161140

theorem inequality_sum_pos (n : ℕ) (x : Fin n → ℝ) 
  (h_pos : ∀ i : Fin n, 0 < x i) (h_n : 2 ≤ n) :
  (Finset.univ.sum (λ i : Fin n, (1 + (x i) ^ 2) / (1 + (x i) * (x (i + 1) % n)))) ≥ n :=
by
  sorry

end inequality_sum_pos_l161_161140


namespace intersection_A_B_l161_161175

-- Define the sets A and B
def A : set (ℝ × ℝ) := { p | p.2 = p.1 + 3 }
def B : set (ℝ × ℝ) := { p | p.2 = 3 * p.1 - 1 }

-- Prove the intersection of A and B is {(2, 5)}
theorem intersection_A_B : A ∩ B = { (2, 5) } :=
by
  sorry

end intersection_A_B_l161_161175


namespace trigonometric_identity_l161_161146

theorem trigonometric_identity (α : ℝ) (h : tan (α + π / 4) = 2) :
  (sin α + 2 * cos α) / (sin α - 2 * cos α) = -7 / 5 :=
sorry

end trigonometric_identity_l161_161146


namespace largest_n_where_Sn_positive_is_4024_l161_161993

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

def sum_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 1 + a n) / 2

theorem largest_n_where_Sn_positive_is_4024
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1_positive : a 1 > 0)
  (h_a2012_a2013_sum_positive : a 2012 + a 2013 > 0)
  (h_a2012_a2013_product_negative : a 2012 * a 2013 < 0) :
  ∃ n : ℕ, n = 4024 ∧ sum_sequence a n > 0 ∧ ∀ m : ℕ, m > n → sum_sequence a m ≤ 0 :=
sorry

end largest_n_where_Sn_positive_is_4024_l161_161993


namespace angle_between_vec_a_b_l161_161569

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  Real.acos (dot_product / (magnitude_a * magnitude_b))

def vec_a : ℝ × ℝ := (Real.cos (20 * Real.pi / 180), Real.sin (20 * Real.pi / 180))
def vec_b : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (190 * Real.pi / 180))

theorem angle_between_vec_a_b : angle_between_vectors vec_a vec_b = Real.pi / 6 :=
by
  sorry

end angle_between_vec_a_b_l161_161569


namespace floor_neg_seven_over_four_l161_161497

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l161_161497


namespace distance_range_l161_161638

noncomputable def polarCurve (θ : ℝ) := if (0 ≤ θ ∧ θ ≤ π / 2) then 4 * Real.cos θ else 0
noncomputable def parametricLine (t : ℝ) := (-3 + t * Real.cos (π / 6), t * Real.sin (π / 6))

theorem distance_range :
  let C := λ α : ℝ, (2 + 2 * Real.cos α, 2 * Real.sin α) in
  let l := λ t : ℝ, parametricLine t in
  let d (α : ℝ) := 
    abs (2 + 2 * Real.cos α - 2 * Real.sqrt 3 * Real.sin α + 3) / Real.sqrt 4 in
  ∀ α, 0 ≤ α ∧ α ≤ π → (1 / 2 : ℝ) ≤ d α ∧ d α ≤ (7 / 2 : ℝ) :=
begin
  sorry
end

end distance_range_l161_161638


namespace area_scaled_l161_161340

variable (g : ℝ → ℝ)

def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem area_scaled (
  h: ∫ (x : ℝ) in -∞..∞, g x = 8) :
  ∫ (x : ℝ) in -∞..∞, 4 * g (x + 3) = 32 :=
by
  sorry

end area_scaled_l161_161340


namespace moles_of_H2O_formed_l161_161898

theorem moles_of_H2O_formed (moles_NH4NO3 moles_NaOH : ℕ) (percent_NaOH_reacts : ℝ)
  (h_decomposition : moles_NH4NO3 = 2) (h_NaOH : moles_NaOH = 2) 
  (h_percent : percent_NaOH_reacts = 0.85) : 
  (moles_NaOH * percent_NaOH_reacts = 1.7) :=
by
  sorry

end moles_of_H2O_formed_l161_161898


namespace profit_calculation_l161_161811

variable (a : ℝ)

def cost_price := a
def marked_price := cost_price * 1.5
def selling_price := marked_price * 0.7
def profit := selling_price - cost_price

theorem profit_calculation (a : ℝ) : profit = 0.05 * a := by
  sorry

end profit_calculation_l161_161811


namespace tan_half_angle_l161_161908

theorem tan_half_angle (α : ℝ) (h1 : π < α) (h2 : α < 3 * π / 2) (h3 : sin (3 * π / 2 + α) = 4 / 5) : tan (α / 2) = -3 :=
by
  sorry

end tan_half_angle_l161_161908


namespace student_ticket_price_l161_161307

-- Define the conditions
variables (S T : ℝ)
def condition1 := 4 * S + 3 * T = 79
def condition2 := 12 * S + 10 * T = 246

-- Prove that the price of a student ticket is 9 dollars, given the equations above
theorem student_ticket_price (h1 : condition1 S T) (h2 : condition2 S T) : T = 9 :=
sorry

end student_ticket_price_l161_161307


namespace paintings_per_room_l161_161692

theorem paintings_per_room (total_paintings : ℕ) (total_rooms : ℕ) 
    (h_paintings : total_paintings = 32) 
    (h_rooms : total_rooms = 4) : 
    total_paintings / total_rooms = 8 := by
  rw [h_paintings, h_rooms]
  norm_num

end paintings_per_room_l161_161692


namespace B_value_l161_161769

theorem B_value (A B : Nat) (hA : A < 10) (hB : B < 10) (h_div99 : (100000 * A + 10000 + 1000 * 5 + 100 * B + 90 + 4) % 99 = 0) :
  B = 3 :=
by
  -- skipping the proof
  sorry

end B_value_l161_161769


namespace percent_increase_correct_l161_161468

def last_year_price : ℝ := 85
def this_year_price : ℝ := 102
def discount_rate : ℝ := 0.15

def last_year_discount : ℝ := discount_rate * last_year_price
def this_year_discount : ℝ := discount_rate * this_year_price

def last_year_discounted_price : ℝ := last_year_price - last_year_discount
def this_year_discounted_price : ℝ := this_year_price - this_year_discount

def increase_in_cost : ℝ := this_year_discounted_price - last_year_discounted_price

def percent_increase : ℝ := (increase_in_cost / last_year_discounted_price) * 100

theorem percent_increase_correct : percent_increase = 20 := 
by
  -- This is where you would normally put the proof, but we'll skip it.
  sorry

end percent_increase_correct_l161_161468


namespace infinite_nested_sqrt_l161_161098

theorem infinite_nested_sqrt :
  ∃ x : ℝ, x = sqrt (3 - x) ∧ x = ( -1 + sqrt 13) / 2 :=
begin
  sorry
end

end infinite_nested_sqrt_l161_161098


namespace Valerie_stamps_problem_l161_161774

theorem Valerie_stamps_problem :
  ∃ x : ℕ, 
    let T := 3 in 
    let B := 2 in 
    let R := B + x in 
    let J := 2 * R in 
    T + B + R + J + 1 = 21 ∧ R = B + x ∧ x = 3 :=
by
  sorry

end Valerie_stamps_problem_l161_161774


namespace prism_volume_l161_161798

-- Conditions
variables {AB AC height : ℝ}
def is_right_triangle (ABC : Prop) : Prop := ABC = (AB = AC = real.sqrt 2)
def prism_height : ℝ := 3
def base_area (leg : ℝ) : ℝ := (leg * leg) / 2

-- Volume of the prism given the conditions
theorem prism_volume (ABC : Prop) 
  (h1 : is_right_triangle ABC) 
  (h2 : AB = real.sqrt 2) 
  (h3 : AC = real.sqrt 2) 
  (h4 : height = 3) :
  base_area AB * height = 3 := 
sorry

end prism_volume_l161_161798


namespace pencils_calculation_l161_161965

def num_pencil_boxes : ℝ := 4.0
def pencils_per_box : ℝ := 648.0
def total_pencils : ℝ := 2592.0

theorem pencils_calculation : (num_pencil_boxes * pencils_per_box) = total_pencils := 
by
  sorry

end pencils_calculation_l161_161965


namespace floor_neg_seven_over_four_l161_161499

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l161_161499


namespace integer_solutions_count_l161_161346

theorem integer_solutions_count :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 :=
by
  sorry

end integer_solutions_count_l161_161346


namespace same_type_quadratic_radicals_l161_161442

theorem same_type_quadratic_radicals :
  (sqrt 24 = 2 * sqrt 6) ∧ (sqrt 54 = 3 * sqrt 6) :=
by
  sorry

end same_type_quadratic_radicals_l161_161442


namespace min_x_for_sqrt_condition_l161_161202

theorem min_x_for_sqrt_condition :
  ∃ x : ℝ, (2 * x - 1 ≥ 0) ∧ ∀ y : ℝ, (2 * y - 1 ≥ 0) → (x ≤ y) :=
begin
  use (1/2),
  split,
  { linarith },
  { intros y hy,
    linarith }
end

end min_x_for_sqrt_condition_l161_161202


namespace factorization_l161_161525

theorem factorization (x y : ℝ) : 
  (x + y) ^ 2 + 4 * (x - y) ^ 2 - 4 * (x ^ 2 - y ^ 2) = (x - 3 * y) ^ 2 :=
by
  sorry

end factorization_l161_161525


namespace positive_integer_solution_of_inequality_l161_161334

theorem positive_integer_solution_of_inequality (x : ℕ) (h : 0 < x) : (3 * x - 1) / 2 + 1 ≥ 2 * x → x = 1 :=
by
  intros
  sorry

end positive_integer_solution_of_inequality_l161_161334


namespace increasing_interval_of_function_l161_161322

noncomputable def function_increasing_interval (y : ℝ → ℝ) : set ℝ :=
{x : ℝ | ∃ (c : ℝ), y c < y x ∧ c < x}

theorem increasing_interval_of_function 
  (x y : ℝ) 
  (f : ℝ → ℝ) 
  (h : f = λ x, 3 * x - x^3) 
  : function_increasing_interval f ⊆ set.Ioo (-1 : ℝ) (1 : ℝ) := 
by 
  sorry

end increasing_interval_of_function_l161_161322


namespace nine_square_sum_l161_161738

theorem nine_square_sum (A B : ℕ)
  (grid : array 3 (array 3 (option ℕ)))
  (H1 : grid[0] = #[none, none, some 3])
  (H2 : grid[1] = #[none, some 2, none])
  (H3 : grid[2] = #[some A, none, some B])
  (H4 : ∀ i, ((∃ j, grid[i] = #[some 1, some 2, some 3]) ∧
               (∃ j, grid[j] = #[some 1, some 2, some 3]))) :
  A + B = 4 :=
sorry

end nine_square_sum_l161_161738


namespace tan_theta_eq_neg2sqrt2_l161_161615

noncomputable def theta : ℝ := sorry

theorem tan_theta_eq_neg2sqrt2 (h1 : sin theta + cos theta = (2 * real.sqrt 2 - 1) / 3) (h2 : 0 < theta ∧ theta < real.pi) :
  real.tan theta = -2 * real.sqrt 2 :=
sorry

end tan_theta_eq_neg2sqrt2_l161_161615


namespace floor_neg_seven_fourths_l161_161476

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l161_161476


namespace diagonal_cubes_140_320_360_l161_161012

-- Define the problem parameters 
def length_x : ℕ := 140
def length_y : ℕ := 320
def length_z : ℕ := 360

-- Define the function to calculate the number of unit cubes the internal diagonal passes through.
def num_cubes_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - Nat.gcd x y - Nat.gcd y z - Nat.gcd z x + Nat.gcd (Nat.gcd x y) z

-- The target theorem to be proven
theorem diagonal_cubes_140_320_360 :
  num_cubes_diagonal length_x length_y length_z = 760 :=
by
  sorry

end diagonal_cubes_140_320_360_l161_161012


namespace milk_ratio_l161_161015

theorem milk_ratio (total_cartons regular_cartons chocolate_cartons : ℕ)
                   (h_total : total_cartons = 60)
                   (h_regular : regular_cartons = 12)
                   (h_chocolate : chocolate_cartons = 24) :
  (regular_cartons / Nat.gcd regular_cartons (Nat.gcd chocolate_cartons (total_cartons - regular_cartons - chocolate_cartons)))
  : (chocolate_cartons / Nat.gcd regular_cartons (Nat.gcd chocolate_cartons (total_cartons - regular_cartons - chocolate_cartons)))
  : ((total_cartons - regular_cartons - chocolate_cartons) / Nat.gcd regular_cartons (Nat.gcd chocolate_cartons (total_cartons - regular_cartons - chocolate_cartons))) = 1:2:2 :=
by
  sorry

end milk_ratio_l161_161015


namespace salary_reduction_l161_161341

theorem salary_reduction (S : ℝ) (x : ℝ) 
  (H1 : S > 0) 
  (H2 : 1.25 * S * (1 - 0.01 * x) = 1.0625 * S) : 
  x = 15 := 
  sorry

end salary_reduction_l161_161341


namespace elder_person_age_l161_161398

open Nat

variable (y e : ℕ)

-- Conditions
def age_difference := e = y + 16
def age_relation := e - 6 = 3 * (y - 6)

theorem elder_person_age
  (h1 : age_difference y e)
  (h2 : age_relation y e) :
  e = 30 :=
sorry

end elder_person_age_l161_161398


namespace hyperbola_equation_l161_161130

-- Conditions
def is_on_hyperbola (x y a b : ℝ) := (x^2) / (a^2) - (y^2) / (b^2) = 1

-- Problem statement
theorem hyperbola_equation :
  (∃ (a b : ℝ), (a^2 = 4) ∧ (b^2 = 3) ∧ (is_on_hyperbola (-4) 3 a b)
    ∧ (is_on_hyperbola (-3) (sqrt 15 / 2) a b)) → 
    ∃ (a b : ℝ), (a^2 = 4 ∧ b^2 = 3 ∧ 
    ∀ (x y : ℝ), is_on_hyperbola x y a b ↔ (x^2) / 4 - (y^2) / 3 = 1) :=
begin
  sorry
end

end hyperbola_equation_l161_161130


namespace floor_neg_seven_over_four_l161_161494

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l161_161494


namespace line_intersects_circle_l161_161552

-- Define the circle C with the equation x^2 + y^2 + 2x - 3 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 3 = 0

-- Define the line l with the equation x + a * y + 2 - a = 0, where a ∈ ℝ
def line_l (x y a : ℝ) : Prop := x + a * y + 2 - a = 0

-- A fixed point (-2, 1)
def fixed_point : ℝ × ℝ := (-2, 1)

-- A lemma stating that line l passes through the fixed point (-2, 1)
lemma line_passes_through_fixed_point (a : ℝ) : line_l (-2) 1 a :=
by simp [line_l]

-- A lemma stating that the fixed point (-2, 1) is inside the circle C
lemma point_inside_circle : ¬ circle_C (-2) 1 :=
by simp [circle_C]

-- The main theorem stating that the line l intersects the circle C
theorem line_intersects_circle (a : ℝ) : ∃ x y : ℝ, circle_C x y ∧ line_l x y a :=
begin
  use [-2, 1],
  split,
  { exact point_inside_circle },
  { exact line_passes_through_fixed_point a }
end

end line_intersects_circle_l161_161552


namespace ninth_observation_l161_161399

theorem ninth_observation (avg1 : ℝ) (avg2 : ℝ) (n1 n2 : ℝ) 
  (sum1 : n1 * avg1 = 120) 
  (sum2 : n2 * avg2 = 117) 
  (avg_decrease : avg1 - avg2 = 2) 
  (obs_count_change : n1 + 1 = n2) 
  : n2 * avg2 - n1 * avg1 = -3 :=
by
  sorry

end ninth_observation_l161_161399


namespace modular_expression_divisible_by_twelve_l161_161764

theorem modular_expression_divisible_by_twelve
  (a b c d : ℕ)
  (ha : a < 12) (hb : b < 12) (hc : c < 12) (hd : d < 12)
  (h_abcd_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_a_invertible : Nat.gcd a 12 = 1) (h_b_invertible : Nat.gcd b 12 = 1)
  (h_c_invertible : Nat.gcd c 12 = 1) (h_d_invertible : Nat.gcd d 12 = 1)
  : (a * b * c + a * b * d + a * c * d + b * c * d) * Nat.mod_inv (a * b * c * d) 12 % 12 = 0 :=
by
  sorry

end modular_expression_divisible_by_twelve_l161_161764


namespace fencing_cost_correct_l161_161037

noncomputable def total_cost_of_fencing : ℝ :=
  let area := 10092 in
  let ratio_l := 3 in
  let ratio_w := 4 in
  let π := Real.pi in
  let diameter := 25 in
  let cost_per_meter := 0.25 in
  let x := Real.sqrt (area / (ratio_l * ratio_w)) in
  let length := ratio_l * x in
  let width := ratio_w * x in
  let perimeter := 2 * (length + width) in
  let circumference := π * diameter in
  let total_fencing_length := perimeter + circumference in
  total_fencing_length * cost_per_meter

theorem fencing_cost_correct : total_cost_of_fencing ≈ 121.135 := 
by 
  sorry

end fencing_cost_correct_l161_161037


namespace inverse_f_138_l161_161943

noncomputable def f (x : ℝ) := 5 * x ^ 3 + 3

theorem inverse_f_138 : ∀ y, f y = 138 → y = 3 :=
by
  intro y hyp
  have h := congr_arg f (eq.symm hyp)
  dsimp at h
  sorry

end inverse_f_138_l161_161943


namespace sqrt_three_irrational_l161_161852

-- Define what it means for a number to be rational
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- State that sqrt(3) is irrational
theorem sqrt_three_irrational : is_irrational (Real.sqrt 3) :=
sorry

end sqrt_three_irrational_l161_161852


namespace find_blue_balls_l161_161808

theorem find_blue_balls 
  (B : ℕ)
  (red_balls : ℕ := 7)
  (green_balls : ℕ := 4)
  (prob_red_red : ℚ := 7 / 40) -- 0.175 represented as a rational number
  (h : (21 / ((11 + B) * (10 + B) / 2 : ℚ)) = prob_red_red) :
  B = 5 :=
sorry

end find_blue_balls_l161_161808


namespace PQ_sum_l161_161248

-- Define the problem conditions
variable (P Q x : ℝ)
variable (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3)))

-- Define the proof goal
theorem PQ_sum (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3))) : P + Q = 52 :=
sorry

end PQ_sum_l161_161248


namespace eval_floor_neg_seven_fourths_l161_161511

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l161_161511


namespace gcd_5039_3427_l161_161784

def a : ℕ := 5039
def b : ℕ := 3427

theorem gcd_5039_3427 : Nat.gcd a b = 7 := by
  sorry

end gcd_5039_3427_l161_161784


namespace Q_over_P_l161_161323

theorem Q_over_P :
  (∀ (x : ℝ), x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 6 → 
    (P / (x + 6) + Q / (x^2 - 6*x) = (x^2 - 3*x + 12) / (x^3 + x^2 - 24*x))) →
  Q / P = 5 / 3 :=
by
  sorry

end Q_over_P_l161_161323


namespace range_of_func_l161_161584

noncomputable def func (x : ℝ) : ℝ := (Real.log x / Real.log 2)^2 - 3 * (Real.log x / Real.log 2) + 6

theorem range_of_func : set.range (λ x, func x) (set.Icc 2 4) = set.Icc (15/4) 4 := by
  sorry

end range_of_func_l161_161584


namespace infinite_nested_sqrt_l161_161096

theorem infinite_nested_sqrt :
  ∃ x : ℝ, x = sqrt (3 - x) ∧ x = ( -1 + sqrt 13) / 2 :=
begin
  sorry
end

end infinite_nested_sqrt_l161_161096


namespace sum_of_squares_l161_161337

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
sorry

end sum_of_squares_l161_161337


namespace numeral_eq_7000_l161_161725

theorem numeral_eq_7000 
  (local_value face_value numeral : ℕ)
  (h1 : face_value = 7)
  (h2 : local_value - face_value = 6993) : 
  numeral = 7000 :=
by
  sorry

end numeral_eq_7000_l161_161725


namespace acute_triangle_pyramid_exists_l161_161705

theorem acute_triangle_pyramid_exists :
  ∀ (A B C : EuclideanGeometry.Point 3), 
  EuclideanGeometry.angled_triangle A B C → 
  (∀ (SA SB SC : EuclideanGeometry.Line 3), EuclideanGeometry.perpendicular SA SB ∧ 
  EuclideanGeometry.perpendicular SB SC ∧ EuclideanGeometry.perpendicular SC SA) → 
  ∃ (S : EuclideanGeometry.Point 3), 
  EuclideanGeometry.triangular_pyramid S A B C :=
by
  sorry

end acute_triangle_pyramid_exists_l161_161705


namespace max_f_l161_161949

noncomputable def h (m n x : ℝ) : ℝ := Real.log x - (2 * m + 3) * x - n

theorem max_f (m n : ℝ) :
  (∀ x : ℝ, 0 < x → h m n x ≤ 0) →
  (∃ (t : ℝ), t = 2 * m + 3 ∧ t > 0 ∧ (2 * m + 3) * n = (2 * m + 3) * (-Real.log (2 * m + 3) - 1)) →
  (∃ c : ℝ, f m n = c ∧ c = 1 / Real.exp 2) :=
sorry

end max_f_l161_161949


namespace trajectory_is_ellipse_l161_161574

-- Definition based on the given condition
def trajectory_condition (x y : ℝ) : Prop :=
  10 * Real.sqrt(x^2 + y^2) = abs(3 * x + 4 * y - 12)

-- The theorem that asserts the given condition results in an Ellipse
theorem trajectory_is_ellipse : 
  ∀ (x y : ℝ), trajectory_condition x y → ∃ a b c: ℝ, a * x^2 + b * y^2 = c ∧ c > 0 :=
by
  sorry

end trajectory_is_ellipse_l161_161574


namespace graduate_distribution_l161_161904

theorem graduate_distribution (graduates classes : ℕ) (Hgraduates : graduates = 5) (Hclasses : classes = 3) 
  (Hnonempty : ∀ (dist : (fin 3) → fin 5 → Prop), (∀ c : fin 3, ∃ g : fin 5, dist c g)) :
  (∃ f : fin 3 → fin 5 → bool, function.injective f ∧ 
    finset.card {G | f (fin.of_nat 0) G ∨ f (fin.of_nat 1) G ∨ f (fin.of_nat 2) G} = 150) := 
sorry

end graduate_distribution_l161_161904


namespace inequality_holds_l161_161680

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end inequality_holds_l161_161680


namespace sara_payment_equivalence_l161_161792

variable (cost_book1 cost_book2 change final_amount : ℝ)

theorem sara_payment_equivalence
  (h1 : cost_book1 = 5.5)
  (h2 : cost_book2 = 6.5)
  (h3 : change = 8)
  (h4 : final_amount = cost_book1 + cost_book2 + change) :
  final_amount = 20 := by
  sorry

end sara_payment_equivalence_l161_161792


namespace chebyshev_inequality_l161_161178

theorem chebyshev_inequality {n : ℕ} {a b : Fin n → ℝ} 
  (h₁ : ∀ i j, i ≤ j → a i ≤ a j) 
  (h₂ : ∀ i j, i ≤ j → b i ≤ b j) 
  : 
  (1 / (n : ℝ)) * ((Finset.range n).sum (λ i, a i * b (n - 1 - i))) 
  ≤ ((Finset.range n).sum a / n) * ((Finset.range n).sum b / n) 
  ∧ ((Finset.range n).sum a / n) * ((Finset.range n).sum b / n) 
  ≤ (1 / (n : ℝ)) * ((Finset.range n).sum (λ i, a i * b i)) :=
sorry

end chebyshev_inequality_l161_161178


namespace part_one_part_two_part_three_l161_161583

-- Given the function definition
def f (a : ℝ) (x : ℝ) := a^(x - a) + 1

-- Condition: f passes through the point (1/2, 2)
axiom (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (fixed_point : f a (1/2) = 2)

-- Goal (Ⅰ): Determine the value of a
theorem part_one : a = 1/2 := sorry

-- Using part_one result
noncomputable def g (x : ℝ) := f (1/2) (x + 1/2) - 1

-- Goal (Ⅱ): Find the explicit formula for g(x)
theorem part_two : g x = (1/2)^x := sorry

-- Using part_two result
def F (m : ℝ) (x : ℝ) := g (2 * x) - m * g (x - 1)

-- Minimum value h(m) for F(x) on [-1, 0]
def h (m : ℝ) : ℝ :=
if m ≤ 1 then 1 - 2 * m
else if 1 < m ∧ m < 2 then -m^2
else 4 - 4 * m

-- Goal (Ⅲ): Find the minimum value h(m)
theorem part_three : ∀ (m : ℝ), ∀ (x : ℝ) (hx_mem : x ∈ set.Icc (-1:ℝ) (0:ℝ)), 
  F m x ≥ h m := 
sorry

end part_one_part_two_part_three_l161_161583


namespace iodine_dilution_l161_161820

theorem iodine_dilution (x : ℕ) : 
    ∃ x, (350 + x) * 2 / 100 = 350 * 15 / 100 ∧ x = 2275 := 
by
  use 2275
  split
  · sorry
  · rfl

end iodine_dilution_l161_161820


namespace not_right_angled_triangle_group_D_l161_161441

theorem not_right_angled_triangle_group_D :
  ¬( (sqrt 3) ^ 2 + 2 ^ 2 = (sqrt 5) ^ 2 ) ∧
  (1 ^ 2 + (sqrt 2) ^ 2 = (sqrt 3) ^ 2) ∧
  (6 ^ 2 + 8 ^ 2 = 10 ^ 2) ∧
  (5 ^ 2 + 12 ^ 2 = 13 ^ 2) :=
by
  sorry

end not_right_angled_triangle_group_D_l161_161441


namespace least_positive_integer_divisible_by_1_to_9_div_2_l161_161973

theorem least_positive_integer_divisible_by_1_to_9_div_2 :
  let l := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9
  in l / 2 = 1260 :=
by
  let l := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9
  show l / 2 = 1260
  sorry

end least_positive_integer_divisible_by_1_to_9_div_2_l161_161973


namespace coffee_tea_overlap_l161_161889

theorem coffee_tea_overlap (c t : ℕ) (h_c : c = 80) (h_t : t = 70) : 
  ∃ (b : ℕ), b = 50 := 
by 
  sorry

end coffee_tea_overlap_l161_161889


namespace day_of_120th_in_N_minus_1_l161_161650

theorem day_of_120th_in_N_minus_1 
  (N : ℕ)
  (day_250th_N : Nat.mod 250 7 = Nat.mod_of_eq 3) -- 250th day of year N is a Wednesday
  (day_150th_N_plus_1 : Nat.mod 150 7 = Nat.mod_of_eq 3) -- 150th day of year N+1 is a Wednesday
  (day_365th_N_minus_2 : Nat.mod 365 7 = 0) -- 365th day of year N-2 is a Sunday
  : Nat.mod 120 7 = 1 := -- day_of_week 120th day of year N-1 is a Monday
sorry

end day_of_120th_in_N_minus_1_l161_161650


namespace number_of_green_pens_l161_161984

theorem number_of_green_pens
  (black_pens : ℕ := 6)
  (red_pens : ℕ := 7)
  (green_pens : ℕ)
  (probability_black : (black_pens : ℚ) / (black_pens + red_pens + green_pens : ℚ) = 1 / 3) :
  green_pens = 5 := 
sorry

end number_of_green_pens_l161_161984


namespace third_quadrant_probability_is_three_eighths_l161_161197

noncomputable def probability_graph_passes_through_third_quadrant : ℚ :=
  let a_values := {(1 / 3 : ℚ), (1 / 4), 3, 4}
  let b_values := {(-1 : ℚ), 1, -2, 2}
  let valid_combinations := [{a := 3, b := -1}, {a := 3, b := -2}, {a := 4, b := -1}, {a := 4, b := -2}, {a := (1 / 3), b := -2}, {a := (1 / 4), b := -2}]
  (valid_combinations.length : ℚ) / (a_values.size * b_values.size)

theorem third_quadrant_probability_is_three_eighths :
  probability_graph_passes_through_third_quadrant = 3 / 8 := sorry

end third_quadrant_probability_is_three_eighths_l161_161197


namespace find_fraction_l161_161196

theorem find_fraction
  (F : ℚ) (m : ℕ) 
  (h1 : F^m * (1 / 4)^2 = 1 / 10^4)
  (h2 : m = 4) : 
  F = 1 / 5 :=
by
  sorry

end find_fraction_l161_161196


namespace largest_common_term_arith_progressions_l161_161717

theorem largest_common_term_arith_progressions (a : ℕ) : 
  (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 3 + 9 * m ∧ a < 1000) → a = 984 := by
  -- Proof is not required, so we add sorry.
  sorry

end largest_common_term_arith_progressions_l161_161717


namespace part1_a_4_intersection_union_part2_range_of_a_l161_161691

section math_equivalent_problems

variable (a : ℝ)

def A : set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B : set ℝ := {x | x^2 + a < 0}

theorem part1_a_4_intersection_union :
  (a = -4) →
  (A ∩ B = {x | (1 / 2 ≤ x ∧ x < 2)}) ∧ (A ∪ B = {x | -2 < x ∧ x ≤ 3}) :=
by
  intros
  sorry

theorem part2_range_of_a :
  ( (compl A) ∩ B = B ) → (a ≥ - (1 / 4)) :=
by
  sorry

end math_equivalent_problems

end part1_a_4_intersection_union_part2_range_of_a_l161_161691


namespace quadratic_function_increasing_l161_161173

theorem quadratic_function_increasing (x : ℝ) : ((x - 1)^2 + 2 < (x + 1 - 1)^2 + 2) ↔ (x > 1) := by
  sorry

end quadratic_function_increasing_l161_161173


namespace sector_properties_l161_161217

/-- Given the radius and arc length, calculate the central angle and area of the sector -/
theorem sector_properties (r l : ℝ) (hr : r = 8) (hl : l = 12) : 
  let α := l / r in
  let S := 0.5 * l * r in
  (α = 3 / 2) ∧ (S = 48) :=
by
  sorry

end sector_properties_l161_161217


namespace original_numerical_equality_exists_l161_161214

theorem original_numerical_equality_exists :
  ∃ (Я ДЕД Ты НЕТ : ℤ), 
    Я = 3 ∧ 
    ДЕД = 202 ∧ 
    Ты = 96 ∧ 
    НЕТ = 109 ∧ 
    Я + ДЕД = Ты + НЕТ :=
by {
  -- Specification of the numerical values and equality would be as described
  use [3, 202, 96, 109],
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  linarith,
}

end original_numerical_equality_exists_l161_161214


namespace percentage_of_invalid_votes_l161_161633

-- Candidate A got 60% of the total valid votes.
-- The total number of votes is 560000.
-- The number of valid votes polled in favor of candidate A is 285600.
variable (total_votes valid_votes_A : ℝ)
variable (percent_A : ℝ := 0.60)
variable (valid_votes total_invalid_votes percent_invalid_votes : ℝ)

axiom h1 : total_votes = 560000
axiom h2 : valid_votes_A = 285600
axiom h3 : valid_votes_A = percent_A * valid_votes
axiom h4 : total_invalid_votes = total_votes - valid_votes
axiom h5 : percent_invalid_votes = (total_invalid_votes / total_votes) * 100

theorem percentage_of_invalid_votes : percent_invalid_votes = 15 := by
  sorry

end percentage_of_invalid_votes_l161_161633


namespace cubic_polynomial_evaluation_l161_161249

theorem cubic_polynomial_evaluation (Q : ℚ → ℚ) (m : ℚ)
  (hQ0 : Q 0 = 2 * m) 
  (hQ1 : Q 1 = 5 * m) 
  (hQm1 : Q (-1) = 0) : 
  Q 2 + Q (-2) = 8 * m := 
by
  sorry

end cubic_polynomial_evaluation_l161_161249


namespace shawn_divided_into_groups_l161_161708

theorem shawn_divided_into_groups :
  ∀ (total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups : ℕ),
  total_pebbles = 40 →
  red_pebbles = 9 →
  blue_pebbles = 13 →
  remaining_pebbles = total_pebbles - red_pebbles - blue_pebbles →
  remaining_pebbles % 3 = 0 →
  yellow_pebbles = blue_pebbles - 7 →
  remaining_pebbles = groups * yellow_pebbles →
  groups = 3 :=
by
  intros total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups
  intros h_total h_red h_blue h_remaining h_divisible h_yellow h_group
  sorry

end shawn_divided_into_groups_l161_161708


namespace stars_proof_l161_161407

noncomputable def stars_arrangement_possible : Prop :=
  ∃ (grid : Fin 4 → Fin 4 → Bool), 
    (Finset.card (Finset.univ.filter (λ i, Finset.univ.filter (λ j, grid i j).card) = 7) ∧ 
    (∀ i₁ i₂, i₁ ≠ i₂ → (Finset.univ.filter (λ j, grid i₁ j).card) =  (Finset.univ.filter (λ j, grid i₂ j).card)) ∧
    (∀ j₁ j₂, j₁ ≠ j₂ → (Finset.univ.filter (λ i, grid i j₁).card) = (Finset.univ.filter (λ i, grid i j₂).card))

noncomputable def fewer_than_7_stars_impossible : Prop :=
  ¬∃ (grid : Fin 4 → Fin 4 → Bool), 
    (Finset.card (Finset.univ.filter (λ i, Finset.univ.filter (λ j, grid i j).card) < 7) ∧ 
    (∀ i₁ i₂, i₁ ≠ i₂ → (Finset.univ.filter (λ j, grid i₁ j).card) ≤ (Finset.univ.filter (λ j, grid i₂ j).card)) ∧
    (∀ j₁ j₂, j₁ ≠ j₂ → (Finset.univ.filter (λ i, grid i j₁).card) ≤ (Finset.univ.filter (λ i, grid i j₂).card)))

theorem stars_proof :
  stars_arrangement_possible ∧ fewer_than_7_stars_impossible :=
  by
    sorry -- proof steps go here

end stars_proof_l161_161407


namespace football_game_attendance_l161_161047

theorem football_game_attendance :
  ∃ y : ℕ, (∃ x : ℕ, x + y = 280 ∧ 60 * x + 25 * y = 14000) ∧ y = 80 :=
by
  sorry

end football_game_attendance_l161_161047


namespace base_salary_at_least_l161_161369

-- Definitions for the conditions.
def previous_salary : ℕ := 75000
def commission_rate : ℚ := 0.15
def sale_value : ℕ := 750
def min_sales_required : ℚ := 266.67

-- Calculate the commission per sale
def commission_per_sale : ℚ := commission_rate * sale_value

-- Calculate the total commission for the minimum sales required
def total_commission : ℚ := min_sales_required * commission_per_sale

-- The base salary S required to not lose money
theorem base_salary_at_least (S : ℚ) : S + total_commission ≥ previous_salary ↔ S ≥ 45000 := 
by
  -- Use sorry to skip the proof
  sorry

end base_salary_at_least_l161_161369


namespace max_volume_range_of_a_x1_x2_inequality_l161_161224

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def g (a x : ℝ) : ℝ := (Real.exp (a * x^2) - Real.exp 1 * x + a * x^2 - 1) / x

theorem max_volume (x : ℝ) (hx : 1 < x) :
  ∃ V : ℝ, V = (Real.pi / 3) * ((Real.log x)^2 / x) ∧ V = (4 * Real.pi / (3 * (Real.exp 2)^2)) :=
sorry

theorem range_of_a (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  0 < a ∧ a < (1/2) * (Real.exp 1) :=
sorry

theorem x1_x2_inequality (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  x1^2 + x2^2 > 2 / Real.exp 1 :=
sorry

end max_volume_range_of_a_x1_x2_inequality_l161_161224


namespace largest_base7_five_digits_l161_161786

theorem largest_base7_five_digits :
  let largest_base7 := 6 * 7^4 + 6 * 7^3 + 6 * 7^2 + 6 * 7^1 + 6 * 7^0 in
  largest_base7 = 16806 := 
by
  let largest_base7 := 6 * 7^4 + 6 * 7^3 + 6 * 7^2 + 6 * 7^1 + 6 * 7^0
  have h : largest_base7 = 16806 := sorry
  exact h

end largest_base7_five_digits_l161_161786


namespace adam_spent_on_ferris_wheel_l161_161059

theorem adam_spent_on_ferris_wheel (t_initial t_left t_price : ℕ) (h1 : t_initial = 13)
  (h2 : t_left = 4) (h3 : t_price = 9) : t_initial - t_left = 9 ∧ (t_initial - t_left) * t_price = 81 := 
by
  sorry

end adam_spent_on_ferris_wheel_l161_161059


namespace probability_one_painted_face_l161_161419

def cube : ℕ := 5
def total_unit_cubes : ℕ := 125
def painted_faces_share_edge : Prop := true
def unit_cubes_with_one_painted_face : ℕ := 41

theorem probability_one_painted_face :
  ∃ (cube : ℕ) (total_unit_cubes : ℕ) (painted_faces_share_edge : Prop) (unit_cubes_with_one_painted_face : ℕ),
  cube = 5 ∧ total_unit_cubes = 125 ∧ painted_faces_share_edge ∧ unit_cubes_with_one_painted_face = 41 →
  (unit_cubes_with_one_painted_face : ℚ) / (total_unit_cubes : ℚ) = 41 / 125 :=
by 
  sorry

end probability_one_painted_face_l161_161419


namespace remainder_when_s_10_plus_1_div_s_minus_2_l161_161900

theorem remainder_when_s_10_plus_1_div_s_minus_2 :
  let f (s : ℤ) := s^10 + 1 in
  let remainder := f 2 in
  remainder = 1025 :=
by
  sorry

end remainder_when_s_10_plus_1_div_s_minus_2_l161_161900


namespace band_row_lengths_l161_161014

theorem band_row_lengths (x y : ℕ) :
  (x * y = 90) → (5 ≤ x ∧ x ≤ 20) → (Even y) → False :=
by sorry

end band_row_lengths_l161_161014


namespace scientific_notation_50000000000_l161_161735

theorem scientific_notation_50000000000 :
  ∃ (a : ℝ) (n : ℤ), 50000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ (a = 5.0 ∨ a = 5) ∧ n = 10 :=
by
  sorry

end scientific_notation_50000000000_l161_161735


namespace problem1_problem2_l161_161595

variables (a b : ℝ) (lambda : ℝ)
constants (dot_product : ℝ → ℝ → ℝ) (norm : ℝ → ℝ) (angle : ℝ → ℝ → ℝ)

-- Conditions
axiom norm_a : norm a = 1
axiom norm_b : norm b = 4
axiom angle_ab : angle a b = 60

-- Problem 1: Prove (2a - b) • (a + b) = -12
theorem problem1 : dot_product (2 * a - b) (a + b) = -12 := 
sorry

-- Problem 2: Prove λ = 12 when (a + b) ⊥ (λa - 2b)
axiom orthogonal_condition : dot_product (a + b) (lambda * a - 2 * b) = 0

theorem problem2 : lambda = 12 :=
sorry

end problem1_problem2_l161_161595


namespace ben_is_10_l161_161062

-- Define the ages of the cousins
def ages : List ℕ := [6, 8, 10, 12, 14]

-- Define the conditions
def wentToPark (x y : ℕ) : Prop := x + y = 18
def wentToLibrary (x y : ℕ) : Prop := x + y < 20
def stayedHome (ben young : ℕ) : Prop := young = 6 ∧ ben ∈ ages ∧ ben ≠ 6 ∧ ben ≠ 12

-- The main theorem stating Ben's age
theorem ben_is_10 : ∃ ben, stayedHome ben 6 ∧ 
  (∃ x y, wentToPark x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) ∧
  (∃ x y, wentToLibrary x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) :=
by
  use 10
  -- Proof steps would go here
  sorry

end ben_is_10_l161_161062


namespace cannot_determine_type_of_triangle_l161_161623

variable {A B C : ℝ}

def in_triangle_ABC (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π

def sin_cos_condition (A C : ℝ) : Prop :=
  sin A * sin C > cos A * cos C

theorem cannot_determine_type_of_triangle (A B C : ℝ) 
  (h1 : in_triangle_ABC A B C)
  (h2 : sin_cos_condition A C) : 
  ¬( (A < π / 2 ∧ B < π / 2 ∧ C < π / 2) ∨ (A = π / 2 ∨ B = π / 2 ∨ C = π / 2) ∨ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2) ) :=
sorry

end cannot_determine_type_of_triangle_l161_161623


namespace area_of_path_l161_161039

-- Define the given conditions
def length_grass_field : ℝ := 75
def width_grass_field : ℝ := 55
def path_width : ℝ := 2.5
def total_cost : ℝ := 6750
def cost_per_sq_m : ℝ := 10

-- Definition of the area of the path to be proved
theorem area_of_path :
  let length_including_path := length_grass_field + 2 * path_width in
  let width_including_path := width_grass_field + 2 * path_width in
  let area_including_path := length_including_path * width_including_path in
  let area_grass_field := length_grass_field * width_grass_field in
  let area_path := area_including_path - area_grass_field in
  area_path = 675 := 
by {
  let length_including_path := length_grass_field + 2 * path_width
  let width_including_path := width_grass_field + 2 * path_width
  let area_including_path := length_including_path * width_including_path
  let area_grass_field := length_grass_field * width_grass_field
  let area_path := area_including_path - area_grass_field
  have h_path_area : area_path = 675 := by sorry
  exact h_path_area
}

end area_of_path_l161_161039


namespace worker_ants_ratio_l161_161712

noncomputable def total_ants : ℝ := 110
noncomputable def female_worker_ants : ℝ := 44

def ratio_worker_ants_to_total_ants (W : ℝ) (total_ants : ℝ) : ℝ :=
  W / total_ants

theorem worker_ants_ratio (total_ants female_worker_ants : ℝ) :
  total_ants = 110 → female_worker_ants = 44 →
  let W := female_worker_ants / 0.80 in
  ratio_worker_ants_to_total_ants W total_ants = 1 / 2 :=
by
  intros h1 h2
  let W := female_worker_ants / 0.80
  unfold ratio_worker_ants_to_total_ants
  sorry

end worker_ants_ratio_l161_161712


namespace union_domain_range_l161_161887

noncomputable def f (x : ℝ) : ℝ := -x^2 - 2 * x + 8

theorem union_domain_range : 
  let A := set.Icc (-4 : ℝ) 2
  let B := set.Icc (0 : ℝ) 9
  A ∪ B = set.Icc (-4 : ℝ) 9 :=
by
  sorry

end union_domain_range_l161_161887


namespace sequence_general_formula_maximum_value_T_n_l161_161910

-- Define the general conditions for the geometric sequence and arithmetic mean property
variables {a_1 a_2 a_3 : ℝ} {n : ℕ}
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a(n + 1) = a(n) / 3

def arithmetic_mean (a_1 a_2 a_3 : ℝ) : Prop :=
  2 * (a_2 + 6) = a_1 + a_3

-- General formula for the n-th term
def a_n (n : ℕ) : ℝ := 27 * (1 / 3)^(n - 1)

-- Product of the first n terms
def T_n (n : ℕ) : ℝ := list.prod (list.map (λ i, a_n i) (list.range n))

-- Prove the general formula for the sequence
theorem sequence_general_formula (h1 : geometric_sequence a_n) (h2 : arithmetic_mean (a_n 1) (a_n 2) (a_n 3)) :
  ∀ n, a_n n = 27 * (1 / 3)^(n - 1) := sorry

-- Prove the maximum value of the product T_n
theorem maximum_value_T_n (h1 : geometric_sequence a_n) (h2 : arithmetic_mean (a_n 1) (a_n 2) (a_n 3)) :
  ∀ n, T_n 3 = 729 ∧ T_n 4 = 729 := sorry

end sequence_general_formula_maximum_value_T_n_l161_161910


namespace exists_shape_in_circle_with_area_l161_161020

-- Given conditions
def inscribed_circle (triangle_side: ℝ) := 
  ∃ r: ℝ, triangle_side = 6 ∧ r = 3 / real.sqrt (3)

def inscribed_shape (area: ℝ) :=
  ∃ shape: Type, ∃ r: ℝ, inscribed_circle 6 ∧ area = 6

-- Question: Does there exist a shape of area 6 square cm inscribed in the circle?
theorem exists_shape_in_circle_with_area :
  inscribed_shape 6 :=
by
  -- placeholder for the actual proof which demonstrates the existence of such shape
  sorry

end exists_shape_in_circle_with_area_l161_161020


namespace original_price_of_shirts_l161_161274

theorem original_price_of_shirts 
  (sale_price : ℝ) 
  (fraction_of_original : ℝ) 
  (original_price : ℝ) 
  (h1 : sale_price = 6) 
  (h2 : fraction_of_original = 0.25) 
  (h3 : sale_price = fraction_of_original * original_price) 
  : original_price = 24 := 
by 
  sorry

end original_price_of_shirts_l161_161274


namespace train_cross_bridge_time_l161_161959

/-
  Define the given conditions:
  - Length of the train (lt): 200 m
  - Speed of the train (st_kmh): 72 km/hr
  - Length of the bridge (lb): 132 m
-/

namespace TrainProblem

def length_of_train : ℕ := 200
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 132

/-
  Convert speed from km/hr to m/s
-/
def speed_of_train_ms : ℕ := speed_of_train_kmh * 1000 / 3600

/-
  Calculate total distance to be traveled (train length + bridge length).
-/
def total_distance : ℕ := length_of_train + length_of_bridge

/-
  Use the formula Time = Distance / Speed
-/
def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_cross_bridge_time : 
  (length_of_train = 200) →
  (speed_of_train_kmh = 72) →
  (length_of_bridge = 132) →
  time_to_cross_bridge = 16.6 :=
by
  intros lt st lb
  sorry

end TrainProblem

end train_cross_bridge_time_l161_161959


namespace fixed_point_of_shifted_exponential_l161_161732

theorem fixed_point_of_shifted_exponential (a : ℝ) (H : a^0 = 1) : a^(3-3) + 3 = 4 :=
by
  sorry

end fixed_point_of_shifted_exponential_l161_161732


namespace floor_of_neg_seven_fourths_l161_161517

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l161_161517


namespace minimum_weighings_l161_161771

theorem minimum_weighings (n : ℕ) (hl : ℕ) (hr : ℕ) 
  (hlight: ∀ i : ℕ, i < n → (hl i = 9 ∧ hr i = 9) ∨ (hl i = 10 ∧ hr i = 10)) 
  (hadj: ∃ i : ℕ, i < n ∧ hl i = 9 ∧ hl (i+1) = 9) : 
  3 ≤ minimum_weighings (λ w : finset ℕ, (∑ i in w, hl i)) :=
sorry

end minimum_weighings_l161_161771


namespace floor_of_neg_seven_fourths_l161_161518

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l161_161518


namespace trigonometric_identity_l161_161865

theorem trigonometric_identity :
  (sin 92 - sin 32 * cos 60) / cos 32 = sqrt 3 / 2 :=
by sorry

end trigonometric_identity_l161_161865


namespace min_distance_zero_l161_161005

variable (U g τ : ℝ)

def y₁ (t : ℝ) : ℝ := U * t - (g * t^2) / 2
def y₂ (t : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2
def s (t : ℝ) : ℝ := |U * τ - g * t * τ + (g * τ^2) / 2|

theorem min_distance_zero
  (U g τ : ℝ)
  (h : 2 * U ≥ g * τ)
  : ∃ t : ℝ, t = τ / 2 + U / g ∧ s t = 0 := sorry

end min_distance_zero_l161_161005


namespace shaded_region_area_l161_161011

-- Definitions of the conditions
def square_side_length : ℝ := 10
def A := (6 : ℝ, 12 : ℝ)
def B := (6 : ℝ, 0 : ℝ)
def square_area := square_side_length ^ 2

theorem shaded_region_area :
  let total_square_area := square_area,
      triangle_area := total_square_area / 2,
      half_triangle_area := triangle_area / 2,
      shaded_area := half_triangle_area * 2
  in shaded_area = 50 :=
begin
  sorry
end

end shaded_region_area_l161_161011


namespace train_crossing_time_l161_161963

noncomputable def length_of_train : ℕ := 250
noncomputable def length_of_bridge : ℕ := 350
noncomputable def speed_of_train_kmph : ℕ := 72

noncomputable def speed_of_train_mps : ℕ := (speed_of_train_kmph * 1000) / 3600

noncomputable def total_distance : ℕ := length_of_train + length_of_bridge

theorem train_crossing_time : total_distance / speed_of_train_mps = 30 := by
  sorry

end train_crossing_time_l161_161963


namespace find_101st_heaviest_coin_max_8_weighings_l161_161356

-- Definitions based on mathematics context
def Coin := Nat
def silver : List Coin := List.range 100
def gold : List Coin := List.range 100 ++ [100] -- Added 101st gold coin represented as 100, as in gold(0) to gold(100)

-- Balance scale comparison (heavier coin determination)
def is_heavier (a b : Coin) : Prop := a > b

-- Hypothetical proof proposition placeholder
theorem find_101st_heaviest_coin_max_8_weighings (s : List Coin) (g : List Coin) (h_unique_weights: ∀ (c1 c2 : Coin), c1 ≠ c2 → c1 ∈ s ∨ c1 ∈ g → c2 ∈ s ∨ c2 ∈ g → ¬ (c1 = c2)) :
  s.length = 100 → g.length = 101 → -- Given lengths of silver and gold coins
  (∃ (steps: Nat), steps <= 8 ∧ (exists_coin: Coin), (exists_index: Nat), exists_coin = (s ++ g).get! exists_index ∧ exists_index = 100) :=
sorry

end find_101st_heaviest_coin_max_8_weighings_l161_161356


namespace inradius_of_right_triangle_l161_161225

theorem inradius_of_right_triangle (PQ QR : ℝ) (angle_R_right : ∠R = π / 2) (PQ_eq : PQ = 15) (QR_eq : QR = 8) : 
  let PR := Real.sqrt (PQ^2 + QR^2)
  let area := 1/2 * PQ * QR
  let s := (PQ + QR + PR) / 2
  let r := area / s
  r = 3 :=
by
  -- Begin condition assumptions
  have hPQ: PQ = 15 := PQ_eq,
  have hQR: QR = 8 := QR_eq,
  have hPR: PR = Real.sqrt (PQ^2 + QR^2) := rfl,
  have hPR_value: PR = 17 := by sorry, -- This can be proved using the Pythagorean theorem.
  have hArea: area = 1/2 * PQ * QR := rfl,
  have hArea_value: area = 60 := by sorry, -- This can be proved using the given PQ and QR values.
  have hS: s = (PQ + QR + PR) / 2 := rfl,
  have hS_value: s = 20 := by sorry, -- This can be proved through addition and division.
  have hR: r = area / s := rfl,
  have hR_value: r = 3 := by sorry, -- Finally, this follows from the area and semiperimeter values.
  exact hR_value

end inradius_of_right_triangle_l161_161225


namespace distance_between_intersections_l161_161534

-- Definitions representing the given equations
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y = 0

-- Asserting the distance between the points of intersection
theorem distance_between_intersections :
  ∃ (p1 p2 : ℝ × ℝ),
    parabola p1.1 p1.2 ∧ circle p1.1 p1.2 ∧
    parabola p2.1 p2.2 ∧ circle p2.1 p2.2 ∧
    p1 ≠ p2 ∧
    ∥p1 - p2∥ = 2 * real.sqrt 3 :=
sorry

end distance_between_intersections_l161_161534


namespace proof_ngon_ratio_l161_161049

noncomputable def regular_ngon_ratio (n : ℕ) (vertices : Fin n → ℝ) (angles_eq : ∀ i j : Fin n, i ≠ j → vertices i = vertices j)
  (lengths_le : ∀ i j : Fin n, i ≤ j → (vertices (i + 1) % n) ≤ (vertices (j + 1) % n))
  : ℝ := 
  vertices n / vertices 1

theorem proof_ngon_ratio (n : ℕ) (vertices : Fin n → ℝ) (angles_eq : ∀ i j : Fin n, i ≠ j → vertices i = vertices j)
  (lengths_le : ∀ i j : Fin n, i ≤ j → (vertices (i + 1) % n) ≤ (vertices (j + 1) % n))
  : regular_ngon_ratio n vertices angles_eq lengths_le = 1 := 
sorry

end proof_ngon_ratio_l161_161049


namespace function_decreasing_interval_l161_161975

theorem function_decreasing_interval :
  (∃ f' : ℝ → ℝ, (∀ x, f' x = x^2 - 4 * x + 3) → 
  (∀ x, (x ∈ Ioo 0 2) ↔ f' (x + 1) < 0)) :=
begin
  sorry
end

end function_decreasing_interval_l161_161975


namespace eval_floor_neg_seven_fourths_l161_161505

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l161_161505


namespace train_stops_for_10_minutes_per_hour_l161_161521

-- Define the conditions
def speed_excluding_stoppages : ℕ := 48 -- in kmph
def speed_including_stoppages : ℕ := 40 -- in kmph

-- Define the question as proving the train stops for 10 minutes per hour
theorem train_stops_for_10_minutes_per_hour :
  (speed_excluding_stoppages - speed_including_stoppages) * 60 / speed_excluding_stoppages = 10 :=
by
  sorry

end train_stops_for_10_minutes_per_hour_l161_161521


namespace at_least_one_not_beyond_20m_l161_161630

variables (p q : Prop)

theorem at_least_one_not_beyond_20m : (¬ p ∨ ¬ q) ↔ ¬ (p ∧ q) :=
by sorry

end at_least_one_not_beyond_20m_l161_161630


namespace correct_propositions_l161_161729

-- Definitions as per the conditions
def is_Rectangle (q : Type) [Parallelogram q] : Prop :=
  ∀ (d1 d2 : Diagonal q), bisects d1 d2 ∧ (length d1 = length d2)

def quad_with_equal_diagonals_is_Rectangle (q : Type) [Quadrilateral q] : Prop :=
  ∀ (d1 d2 : Diagonal q), (length d1 = length d2) → is_Rectangle q

def is_Rhombus (q : Type) [Parallelogram q] : Prop :=
  ∀ (d : Diagonal q) (a1 a2 : Angle q), bisects d (a1, a2)

def parallelogram_with_diagonal_bisecting_angles_is_Rhombus (q : Type) [Parallelogram q] : Prop :=
  ∀ (d : Diagonal q) (a1 a2 : Angle q), bisects d (a1, a2) → is_Rhombus q

-- Problem statement in Lean
theorem correct_propositions :
  ∀ (q : Type), [Parallelogram q] → [Quadrilateral q] → 
    ((is_Rectangle q) ∧ ¬(quad_with_equal_diagonals_is_Rectangle q) ∧ (is_Rhombus q) ∧ (parallelogram_with_diagonal_bisecting_angles_is_Rhombus q)) :=
by
  -- Proof is skipped with 'sorry'
  sorry

end correct_propositions_l161_161729


namespace inequality_holds_l161_161678

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
    a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
by
  sorry

end inequality_holds_l161_161678


namespace initial_percentage_l161_161017

variable (P : ℝ)

theorem initial_percentage (P : ℝ) 
  (h1 : 0 ≤ P ∧ P ≤ 100)
  (h2 : (7600 * (1 - P / 100) * 0.75) = 5130) :
  P = 10 :=
by
  sorry

end initial_percentage_l161_161017


namespace exp_graph_fixed_point_l161_161731

theorem exp_graph_fixed_point (a : ℝ) :
  ∃ (x y : ℝ), x = 3 ∧ y = 4 ∧ y = a^(x - 3) + 3 :=
by
  use 3
  use 4
  split
  · rfl
  split
  · rfl
  · sorry

end exp_graph_fixed_point_l161_161731


namespace pow_divisible_by_13_l161_161704

theorem pow_divisible_by_13 (n : ℕ) (h : 0 < n) : (4^(2*n+1) + 3^(n+2)) % 13 = 0 :=
sorry

end pow_divisible_by_13_l161_161704


namespace projection_lengths_difference_l161_161715

noncomputable def parabola (p q : ℝ) := λ x : ℝ, x^2 + p * x + q

theorem projection_lengths_difference 
  (p q : ℝ)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h₁ : parabola p q x₁ = x₁)
  (h₂ : parabola p q x₂ = x₂)
  (h₃ : parabola p q x₃ = 2 * x₃)
  (h₄ : parabola p q x₄ = 2 * x₄)
  (h₅ : x₁ + x₂ = 1 - p)
  (h₆ : x₃ + x₄ = 2 - p) :
  (x₄ - x₂) - (x₁ - x₃) = 1 :=
sorry

end projection_lengths_difference_l161_161715


namespace S_25_equals_50_l161_161454

-- Defining the arithmetic sequence
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop :=
∀ n : ℕ, a_n n = a1 + (n - 1) * d

-- Given conditions
def given_conditions (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop :=
a_n 3 + a_n 14 + a_n 16 + a_n 19 = 8

-- Sum of first n terms of an arithmetic sequence
def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
n / 2 * (a_n 1 + a_n n)

theorem S_25_equals_50 (a_n : ℕ → ℝ) (a1 d : ℝ) :
arithmetic_sequence a_n a1 d →
given_conditions a_n a1 d →
sum_of_first_n_terms a_n 25 = 50 :=
by sorry

end S_25_equals_50_l161_161454


namespace floor_neg_seven_quarter_l161_161483

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l161_161483


namespace dot_product_value_l161_161180

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem dot_product_value
  (h1 : ∥a - 2 • b∥ = 1)
  (h2 : ∥2 • a + 3 • b∥ = 1 / 3) :
  (5 • a - 3 • b) ⬝ (a - 9 • b) = 80 / 9 :=
sorry

end dot_product_value_l161_161180


namespace hyperbola_integer_points_count_l161_161607

theorem hyperbola_integer_points_count :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), 
    (p ∈ S ↔ (∃ (x y : ℤ), p = (x, y) ∧ y = 2013 / x)) 
    ∧ S.card = 16 := 
by 
  sorry

end hyperbola_integer_points_count_l161_161607


namespace sum_of_eight_numbers_l161_161619

theorem sum_of_eight_numbers (average : ℝ) (h : average = 5) :
  (8 * average) = 40 :=
by
  sorry

end sum_of_eight_numbers_l161_161619


namespace expression_value_l161_161302

theorem expression_value (a : ℝ) (h_nonzero : a ≠ 0) (h_ne_two : a ≠ 2) (h_ne_neg_two : a ≠ -2) (h_ne_neg_one : a ≠ -1) (h_eq_one : a = 1) :
  1 - (((a-2)/a) / ((a^2-4)/(a^2+a))) = 1 / 3 :=
by
  sorry

end expression_value_l161_161302


namespace equal_reciprocal_radii_l161_161813

-- Define the quadrilateral and its properties
variables (A B C D E : Type) 

-- Define the properties given in the conditions
variables [inscribed_circle_quadrilateral A B C D E] -- A quadrilateral with an inscribed circle
variables (r_1 r_2 r_3 r_4 : ℝ) -- Radii of the inscribed circles

-- Given that AB is parallel to CD and BC = AD, AC and BD intersect at E 
variable (h_parallel : AB ∥ CD)
variable (h_equal_length : BC = AD)
variable (h_intersect : ∃ E, intersect AC BD)

-- Define that the radii r_i (inscribed in ABE, BCE, CDE, DAE) exist
axiom inscribed_radius_1 : r_1 = inscribed_radius ABE
axiom inscribed_radius_2 : r_2 = inscribed_radius BCE
axiom inscribed_radius_3 : r_3 = inscribed_radius CDE
axiom inscribed_radius_4 : r_4 = inscribed_radius DAE

-- Statement of the proof problem
theorem equal_reciprocal_radii : 
  (1 / r_1 + 1 / r_3) = (1 / r_2 + 1 / r_4) :=
sorry -- Proof skipped


end equal_reciprocal_radii_l161_161813


namespace tangent_lines_through_A_area_of_triangle_AOC_l161_161127

noncomputable def circle_eq : String := "x^2 + y^2 - 4x - 6y + 12 = 0"
noncomputable def center_C : (ℝ × ℝ) := (2, 3)
noncomputable def point_A : (ℝ × ℝ) := (3, 5)
noncomputable def origin_O : (ℝ × ℝ) := (0, 0)

theorem tangent_lines_through_A : 
  x = 3 ∨ y = (3 / 4) * x + (11 / 4) :=
  sorry

theorem area_of_triangle_AOC :
  let AO := sqrt ((point_A.1 - origin_O.1)^2 + (point_A.2 - origin_O.2)^2) in
  let line_AO := 5 * x - 3 * y = 0 in
  let d := abs (5 * 2 - 3 * 3) / sqrt (5^2 + (-3)^2) in
  1 / 2 * AO * d = 1 / 2 :=
  sorry

end tangent_lines_through_A_area_of_triangle_AOC_l161_161127


namespace train_crossing_time_l161_161838

/-- Prove the time it takes for a train of length 50 meters running at 60 km/hr to cross a pole is 3 seconds. -/
theorem train_crossing_time
  (speed_kmh : ℝ)
  (length_m : ℝ)
  (conversion_factor : ℝ)
  (time_seconds : ℝ) :
  speed_kmh = 60 →
  length_m = 50 →
  conversion_factor = 1000 / 3600 →
  time_seconds = 3 →
  time_seconds = length_m / (speed_kmh * conversion_factor) := 
by
  intros
  sorry

end train_crossing_time_l161_161838


namespace quadrilateral_ratio_l161_161674

variables {A B C D I : Type*}
variables [MetricSpace I]

-- Define the distances from the center I to the vertices.
axioms (IA : ℝ) (IB : ℝ) (IC : ℝ) (ID : ℝ)

-- Set the given distances
noncomputable def given_IA : IA = 5 := sorry
noncomputable def given_IB : IB = 7 := sorry
noncomputable def given_IC : IC = 4 := sorry
noncomputable def given_ID : ID = 9 := sorry

-- Define the ratio to be proven
noncomputable def ratio : ℝ := 
  (IA * IB) / (IC * ID)
noncomputable def answer : ratio = 35 / 36 := sorry

-- The Main theorem statement
theorem quadrilateral_ratio (h1 : given_IA) (h2 : given_IB) (h3 : given_IC) (h4 : given_ID) :
  ratio = 35 / 36 := 
  answer

end quadrilateral_ratio_l161_161674


namespace trapezoid_area_ABCD_l161_161736

noncomputable def trapezoid_area (AB CD BC : ℝ) (M : ℝ) (DM_angle_bisector_passes_M : Prop) : ℝ :=
  let AD := 8  -- inferred in the solution step
  let PC := 8  -- inferred height from Pythagorean theorem
  (1 / 2) * (AD + BC) * PC

theorem trapezoid_area_ABCD :
  let AB := 8
  let CD := 10
  let BC := 2
  let M := 4  -- midpoint of AB
  let DM_angle_bisector_passes_M := true
  trapezoid_area AB CD BC M DM_angle_bisector_passes_M = 40 :=
by
  sorry

end trapezoid_area_ABCD_l161_161736


namespace first_three_decimal_digits_l161_161777

theorem first_three_decimal_digits 
  (a b : ℝ) 
  (h_a : a = 10^1200) 
  (h_b : b = (a + 1)^(5/3)) :
  (floor (b * 10^3) % 1000 = 333) :=
by sorry

end first_three_decimal_digits_l161_161777


namespace sequence_properties_l161_161404

def arithmetic_sequence (n : ℕ) (d : ℤ) (a : ℤ) : ℤ :=
  a + (n - 1) * d

theorem sequence_properties (d : ℤ) (a1 a2 a3 a4 a5 a6 : ℤ) :
  a1 + a3 + a5 = 105 →
  a2 + a4 + a6 = 99 →
  a3 = a1 + 2 * d →
  a5 = a1 + 4 * d →
  a2 = a1 + d →
  a4 = a1 + 3 * d →
  a6 = a1 + 5 * d →
  d = -2 ∧ ∀ (n : ℕ), arithmetic_sequence n d a3 = 41 - 2 * n ∧ (∀ n, n ≤ 20 → 0 < arithmetic_sequence n d a3) ∧ (∀ n, 21 ≤ n → arithmetic_sequence n d a3 < 0) ∧  S_n_max n = 20 :=
by
  sorry

end sequence_properties_l161_161404


namespace xyz_inequality_l161_161686

theorem xyz_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ (3/4) :=
sorry

end xyz_inequality_l161_161686


namespace sufficient_but_not_necessary_l161_161616

-- Specify the conditions
variable (a b : ℝ)

-- Define the statement to be proved
theorem sufficient_but_not_necessary (h : a^2 = b^2) : (a = b ∨ a = -b) ∧ ¬ (h → a = b) := by
  sorry

end sufficient_but_not_necessary_l161_161616


namespace evaluate_expression_l161_161192

variable (x y z : ℤ)

theorem evaluate_expression :
  x = 3 → y = 2 → z = 4 → 3 * x - 4 * y + 5 * z = 21 :=
by
  intros hx hy hz
  rw [hx, hy, hz]
  sorry

end evaluate_expression_l161_161192


namespace part_I_part_II_l161_161992

/-- Problem statement from part (I) --/
theorem part_I (α : ℝ) (A : ℝ × ℝ) (k : ℝ) (C : set (ℝ × ℝ))
  (hA : A = (-1, 0))
  (hC : ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + y^2 - 6 * x + 5 = 0)
  (hk : k = tan α)
  (h_inter : ∃ (x y : ℝ), (x, y) ∈ C ∧ y = k * (x + 1))
  : α ∈ set.Icc 0 (π / 6) ∪ set.Icc (5 * π / 6) π :=
sorry

/-- Problem statement from part (II) --/
theorem part_II (B : ℝ × ℝ) (C : set (ℝ × ℝ))
  (hC : ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + y^2 - 6 * x + 5 = 0)
  (hB : ∀ (x y : ℝ), B = (x, y) → (x, y) ∈ C)
  : 3 * sqrt 3 - 4 ≤ sqrt 3 * (B.fst) + (B.snd) ∧ sqrt 3 * (B.fst) + (B.snd) ≤ 3 * sqrt 3 + 4 :=
sorry

end part_I_part_II_l161_161992


namespace potato_bag_weight_l161_161233

theorem potato_bag_weight:
  (∀ (persons: ℕ) (weight_per_person: ℝ) (total_cost: ℝ) (bag_cost: ℝ) (bags: ℝ),
    persons = 40 →
    weight_per_person = 1.5 →
    total_cost = 15 →
    bag_cost = 5 →
    bags = total_cost / bag_cost →
    persons * weight_per_person = 60 →
    bags = 3 →
    (persons * weight_per_person) / bags = 20) :=
begin
  intros,
  sorry,
end

end potato_bag_weight_l161_161233


namespace octagon_area_in_square_l161_161827

def main : IO Unit :=
  IO.println s!"Hello, Lean!"

theorem octagon_area_in_square :
  ∀ (s : ℝ), ∀ (area_square : ℝ), ∀ (area_octagon : ℝ),
  (s * 4 = 160) →
  (s = 40) →
  (area_square = s * s) →
  (area_square = 1600) →
  (∃ (area_triangle : ℝ), area_triangle = 50 ∧ 8 * area_triangle = 400) →
  (area_octagon = area_square - 400) →
  (area_octagon = 1200) :=
by
  intros s area_square area_octagon h1 h2 h3 h4 h5 h6
  sorry

end octagon_area_in_square_l161_161827


namespace seq_eventually_constant_iff_perfect_square_l161_161953

-- Define the digit sum function
def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence according to the problem's conditions
def seq (A : ℕ) : ℕ → ℕ 
| 0     := A
| (k+1) := seq k + digitSum (seq k)

theorem seq_eventually_constant_iff_perfect_square (A : ℕ) :
  ∃ N, ∀ n ≥ N, seq A n = seq A N ↔ ∃ n : ℕ, A = n^2 :=
sorry

end seq_eventually_constant_iff_perfect_square_l161_161953


namespace binary_to_decimal_example_l161_161804

theorem binary_to_decimal_example : 
  let binary_number := [1, 0, 1, 1, 1, 1, 0, 1, 1]
  (binary_number.reverse.zipWithIndex.map (λ pair, pair.1 * 2^pair.2)).sum = 379 :=
by 
  let binary_number := [1, 0, 1, 1, 1, 1, 0, 1, 1]
  unfold binary_number
  have h := binary_number.reverse.zipWithIndex.map (λ pair, pair.1 * 2^pair.2)
  unfold h
  sorry

end binary_to_decimal_example_l161_161804


namespace part1_part2_l161_161649

variable (A B C a b c S : ℝ)

-- Part (1)
theorem part1 (h1 : cos A = 4 / 5) : sin (B + C) / 2 ^ 2 + cos (2 * A) = 59 / 50 :=
  sorry

-- Part (2)
theorem part2 (h1 : b = 2)  
               (h2 : S = 3) 
               (h3 : cos A = 4 / 5) : 
               (let sinA := Real.sqrt (1 - (4/5)^2),
                     c := 2 * S / (1/2 * b * sinA),
                     a := Real.sqrt (b^2 + c^2 - 2 * b * c * (4/5))) in
                     (a / (2 * sinA) = 5 * Real.sqrt 13 / 6) :=
  sorry

end part1_part2_l161_161649


namespace ball_returns_to_Ben_after_three_throws_l161_161805

def circle_throw (n : ℕ) (skip : ℕ) (start : ℕ) : ℕ :=
  (start + skip) % n

theorem ball_returns_to_Ben_after_three_throws :
  ∀ (n : ℕ) (skip : ℕ) (start : ℕ),
  n = 15 → skip = 5 → start = 1 →
  (circle_throw n skip (circle_throw n skip (circle_throw n skip start))) = start :=
by
  intros n skip start hn hskip hstart
  sorry

end ball_returns_to_Ben_after_three_throws_l161_161805


namespace rooks_non_attacking_l161_161610

-- Number of ways to place 3 rooks on an 8x8 chessboard
-- assuming minimal interference from omitted cells
theorem rooks_non_attacking : 
  ∃ n : ℕ, n = (choose 8 3) * (choose 8 3) * fact 3 ∧ n = 16 :=
by
  sorry

end rooks_non_attacking_l161_161610


namespace trainB_speed_l161_161370

variable (v : ℕ)

def trainA_speed : ℕ := 30
def time_gap : ℕ := 2
def distance_overtake : ℕ := 360

theorem trainB_speed (h :  v > trainA_speed) : v = 42 :=
by
  sorry

end trainB_speed_l161_161370


namespace evaluate_expression_l161_161519

theorem evaluate_expression :
  ((
    ((3 + 2)⁻¹ * 2)⁻¹ + 2
  )⁻¹ + 2 = 20 / 9 : ℝ) :=
by
  sorry

end evaluate_expression_l161_161519


namespace integer_solutions_count_l161_161345

theorem integer_solutions_count :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 :=
by
  sorry

end integer_solutions_count_l161_161345


namespace cloud9_total_revenue_after_discounts_and_refunds_l161_161452

theorem cloud9_total_revenue_after_discounts_and_refunds :
  let individual_total := 12000
  let individual_early_total := 3000
  let group_a_total := 6000
  let group_a_participants := 8
  let group_b_total := 9000
  let group_b_participants := 15
  let group_c_total := 15000
  let group_c_participants := 22
  let individual_refund1 := 500
  let individual_refund1_count := 3
  let individual_refund2 := 300
  let individual_refund2_count := 2
  let group_refund := 800
  let group_refund_count := 2

  -- Discounts
  let early_booking_discount := 0.03
  let discount_between_5_and_10 := 0.05
  let discount_between_11_and_20 := 0.1
  let discount_21_and_more := 0.15

  -- Calculating individual bookings
  let individual_early_discount_total := individual_early_total * early_booking_discount
  let individual_total_after_discount := individual_total - individual_early_discount_total

  -- Calculating group bookings
  let group_a_discount := group_a_total * discount_between_5_and_10
  let group_a_early_discount := (group_a_total - group_a_discount) * early_booking_discount
  let group_a_total_after_discount := group_a_total - group_a_discount - group_a_early_discount

  let group_b_discount := group_b_total * discount_between_11_and_20
  let group_b_total_after_discount := group_b_total - group_b_discount

  let group_c_discount := group_c_total * discount_21_and_more
  let group_c_early_discount := (group_c_total - group_c_discount) * early_booking_discount
  let group_c_total_after_discount := group_c_total - group_c_discount - group_c_early_discount

  let total_group_after_discount := group_a_total_after_discount + group_b_total_after_discount + group_c_total_after_discount

  -- Calculating refunds
  let total_individual_refunds := (individual_refund1 * individual_refund1_count) + (individual_refund2 * individual_refund2_count)
  let total_group_refunds := group_refund

  let total_refunds := total_individual_refunds + total_group_refunds

  -- Final total calculation after all discounts and refunds
  let final_total := individual_total_after_discount + total_group_after_discount - total_refunds
  final_total = 35006.50 := by
  -- The rest of the proof would go here, but we use sorry to bypass the proof.
  sorry

end cloud9_total_revenue_after_discounts_and_refunds_l161_161452


namespace simple_random_sampling_independent_l161_161219

-- Definitions related to the simple random sampling
variable (N : ℕ) -- Number of possible individuals in the sampling

-- Mathematical condition expressing the independence and equality of selection probability
theorem simple_random_sampling_independent (individual : ℕ) (n : ℕ) :
  (1 ≤ individual ∧ individual ≤ N) →
  (∀ n : ℕ, 0 < n → n ≤ N → (P(\text{select} individual in n) = 1 / N)) :=
by sorry

end simple_random_sampling_independent_l161_161219


namespace sum_m_is_neg_2_l161_161578

noncomputable def sum_of_integers_m_satisfying_conditions : ℤ :=
  let fractional_equation (x m : ℤ) := (x + m) / (x + 2) - m / (x - 2) = 1 in
  let inequalities_system (m y : ℤ) := (m - 6 * y > 2 ∧ y - 4 ≤ 3 * y + 4) in
  let num_integer_solutions (m : ℤ) := ∑ y in (Set.Icc Int.min_int Int.max_int), if inequalities_system m y then 1 else 0 in
  let valid_m_values := {m : ℤ | fractional_equation (2 - 2 * m) m ∧ (num_integer_solutions m = 4)} in
  ∑ m in valid_m_values, m

theorem sum_m_is_neg_2 : sum_of_integers_m_satisfying_conditions = -2 :=
  sorry

end sum_m_is_neg_2_l161_161578


namespace circle_equation_passes_through_O_and_F_and_is_tangent_to_l_line_A_l161_161170

noncomputable theory

-- Definitions for the problem:
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def directrix (x : ℝ) : Prop := x = -1

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def origin (O : ℝ × ℝ) : Prop := O = (0, 0)

def symmetric_point (A A' : ℝ × ℝ) : Prop := A'.fst = A.fst ∧ A'.snd = -A.snd

def intersects_parabola (A B : ℝ × ℝ) : Prop := 
  ∃ (F : ℝ × ℝ), focus F ∧ (parabola A.fst A.snd ∧ parabola B.fst B.snd)

-- Statement to prove:
theorem circle_equation_passes_through_O_and_F_and_is_tangent_to_l :
  ∃ (a b : ℝ), (a = 1/2 ∧ (b = sqrt 2 ∨ b = -sqrt 2)) ∧
    (∀ x y : ℝ, ((x - a)^2 + (y - b)^2 = 9/4) ↔ ((x, y) = (0, 0) ∨ (x, y) = (1, 0))) := 
sorry

theorem line_A'_B_passes_through_fixed_point (A A' B : ℝ × ℝ) :
  intersects_parabola A B ∧ symmetric_point A A' → 
  ∃ (M : ℝ × ℝ), M = (-1, 0) ∧ ∀ x : ℝ, (line_through (A', B) x).snd = M.snd := 
sorry

end circle_equation_passes_through_O_and_F_and_is_tangent_to_l_line_A_l161_161170


namespace functional_eq_l161_161535

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem functional_eq {f : ℝ → ℝ} (h1 : ∀ x, x * (f (x + 1) - f x) = f x) (h2 : ∀ x y, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, ∀ x > 0, f x = k * x :=
sorry

end functional_eq_l161_161535


namespace even_function_evaluation_l161_161153

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

theorem even_function_evaluation (f : ℝ → ℝ) (h_even : is_even_function f)
  (h_pos : ∀ x, 0 < x → f(x) = -x^2 + x) :
  ∀ x, x < 0 → f(x)= -x^2 - x := 
by
  intros x hx
  have hx_neg : -x > 0 := by linarith
  rw [← h_even x, h_pos (-x) hx_neg]
  sorry

end even_function_evaluation_l161_161153


namespace photocopy_problem_l161_161723

variable (cost_per_copy : ℝ) (discount : ℝ) (copies_Steve : ℕ) (copies_David : ℕ) (if_submit_together : Bool)

-- Define the conditions and assertions
def total_copies_ordered (cost_per_copy : ℝ) (discount : ℝ) (copies_Steve : ℕ) (copies_David : ℕ) (if_submit_together : Bool) : ℕ :=
  if if_submit_together then copies_Steve + copies_David else copies_Steve + copies_David

-- Calculate the savings for each person
def each_saves (cost_per_copy : ℝ) (discount : ℝ) (copies_Steve : ℕ) (copies_David : ℕ) (if_submit_together : Bool) : ℝ :=
  let total_copies := total_copies_ordered cost_per_copy discount copies_Steve copies_David if_submit_together
  let total_cost_before_discount := total_copies * cost_per_copy
  let discount_amount := if total_copies > 100 then discount * total_cost_before_discount else 0
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let cost_per_person_with_discount := total_cost_after_discount / 2
  let original_cost_per_person := copies_Steve * cost_per_copy
  original_cost_per_person - cost_per_person_with_discount

-- Assertion statements
theorem photocopy_problem :
  cost_per_copy = 0.02 ∧ discount = 0.25 ∧ copies_Steve = 80 ∧ copies_David = 80 ∧ if_submit_together = true →
  total_copies_ordered cost_per_copy discount copies_Steve copies_David if_submit_together = 160 ∧
  each_saves cost_per_copy discount copies_Steve copies_David if_submit_together = 0.40 :=
by
  sorry

end photocopy_problem_l161_161723


namespace ratio_is_1_over_32_l161_161215

noncomputable def ratio_of_areas (a : ℝ) : ℝ :=
  -- define the area of triangle APJ and the face area of the cube,
  let area_APJ := (a^2 * Real.sqrt 2) / 8,
      face_area := a^2 in
  (area_APJ / face_area)^2

theorem ratio_is_1_over_32 (a : ℝ) : ratio_of_areas a = 1 / 32 :=
  sorry

end ratio_is_1_over_32_l161_161215


namespace julie_hours_per_week_school_year_l161_161238

-- Defining the assumptions
variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℝ)
variable (school_year_weeks : ℕ) (school_year_earnings : ℝ)

-- Assuming the given values
def assumptions : Prop :=
  summer_hours_per_week = 36 ∧ 
  summer_weeks = 10 ∧ 
  summer_earnings = 4500 ∧ 
  school_year_weeks = 45 ∧ 
  school_year_earnings = 4500

-- Proving that Julie must work 8 hours per week during the school year to make another $4500
theorem julie_hours_per_week_school_year : 
  assumptions summer_hours_per_week summer_weeks summer_earnings school_year_weeks school_year_earnings →
  (school_year_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_year_weeks = 8) :=
by
  sorry

end julie_hours_per_week_school_year_l161_161238


namespace steel_plate_minimization_l161_161375

theorem steel_plate_minimization : ∃ m n : ℕ, 2 * m + n = 15 ∧ m + 2 * n = 18 ∧ m + 3 * n = 27 ∧ m + n = 12 :=
by {
  use 4,
  use 8,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num },
}

end steel_plate_minimization_l161_161375


namespace percentage_decrease_wages_l161_161240

theorem percentage_decrease_wages (W : ℝ) (P : ℝ) : 
  (0.20 * W * (1 - P / 100)) = 0.70 * (0.20 * W) → 
  P = 30 :=
by
  sorry

end percentage_decrease_wages_l161_161240


namespace AR_eq_RA1_l161_161009

theorem AR_eq_RA1 (A B C A1 B1 P R : Point) (h_triangle : scalene_triangle A B C)
  (h_AA1_bisector : is_angle_bisector A A1)
  (h_BB1_bisector : is_angle_bisector B B1)
  (h_perpendicular_bisector : is_perpendicular_bisector (segment B B1) P (line AA1))
  (h_parallel : P B1 ∥ R A1) :
  A R = R A1 := 
sorry

end AR_eq_RA1_l161_161009


namespace triangle_area_ratio_l161_161632

theorem triangle_area_ratio
  (a b c : ℕ) (S_triangle : ℕ) -- assuming S_triangle represents the area of the original triangle
  (S_bisected_triangle : ℕ) -- assuming S_bisected_triangle represents the area of the bisected triangle
  (is_angle_bisector : ∀ x y z : ℕ, ∃ k, k = (2 * a * b * c * x) / ((a + b) * (a + c) * (b + c))) :
  S_bisected_triangle = (2 * a * b * c) / ((a + b) * (a + c) * (b + c)) * S_triangle :=
sorry

end triangle_area_ratio_l161_161632


namespace angle_bisector_of_CGE_l161_161213

open Function
open Geometry

variables {A B C D E F G : Point}

-- Define symmetric point
def symmetric_to_midpoint (P M Q : Point) : Prop :=
  midpoint P Q = M 

-- Define the problem statement
theorem angle_bisector_of_CGE (h1 : complete_quadrilateral A B C D E F)
                             (h2 : dist B C = dist E F)
                             (h3 : symmetric_to_midpoint A (midpoint C E) G) :
  angle_bisector (line D G) (angle C G E) :=
sorry

end angle_bisector_of_CGE_l161_161213


namespace oxygen_atoms_in_compound_l161_161022

noncomputable def number_of_oxygen_atoms
  (MW_compound H_weight Br_weight O_weight : ℚ)
  (MW_compound_val : MW_compound = 129) 
  (H_weight_val : H_weight = 1) 
  (Br_weight_val : Br_weight = 79.9) 
  (O_weight_val : O_weight = 16) : ℕ :=
let n := (MW_compound - (H_weight + Br_weight)) / O_weight in
nat.ceil n

theorem oxygen_atoms_in_compound
  (MW_compound : ℚ) (H_weight : ℚ) (Br_weight : ℚ) (O_weight : ℚ)
  (MW_compound_val : MW_compound = 129)
  (H_weight_val : H_weight = 1)
  (Br_weight_val : Br_weight = 79.9)
  (O_weight_val : O_weight = 16) :
  number_of_oxygen_atoms MW_compound H_weight Br_weight O_weight MW_compound_val H_weight_val Br_weight_val O_weight_val = 3 := 
by {
  unfold number_of_oxygen_atoms,
  have : (129 - (1 + 79.9)) / 16 = 3.00625, sorry,
  rw this,
  exact nat.ceil_eq 3 3.00625 (by norm_num),
  sorry -- additional steps to formalize the proof
}

end oxygen_atoms_in_compound_l161_161022


namespace rearrangement_impossible_l161_161918

-- Definition of sequence and problem conditions
def sequence := list.range' 1 1986 >>= λ n, [n, n]

-- Theorem stating the main problem
theorem rearrangement_impossible :
  ¬ (∃ f : list ℕ, function.bijective f ∧ (∀ k ∈ list.range' 1 1986, count_elements_between f k = k)) :=
  sorry

-- Helper functions to define the condition 'k numbers between the two occurrences of k'
noncomputable def count_elements_between (f : list ℕ) (k : ℕ) : ℕ :=
  let k_indices := list.indexes f k in
  if h : list.length k_indices = 2 then
    let [i1, i2] := k_indices, h.page_split in
    i2 - i1 - 1
  else
    0

-- Assertions that can be used for proving
example : sequence.length = 3972 := by norm_num
example (k : ℕ) (h : k ∈ list.range' 1 1986) : list.count k sequence = 2 := by
  simp [sequence, list.range'_eq_range'_dec, list.range]


end rearrangement_impossible_l161_161918


namespace second_term_of_geometric_series_l161_161855

theorem second_term_of_geometric_series (a r S term2 : ℝ) 
  (h1 : r = 1 / 4)
  (h2 : S = 40)
  (h3 : S = a / (1 - r))
  (h4 : term2 = a * r) : 
  term2 = 7.5 := 
  by
  sorry

end second_term_of_geometric_series_l161_161855


namespace parallelogram_AGHI_perimeter_l161_161980

theorem parallelogram_AGHI_perimeter 
  (A B C G H I : Type*)
  [H2 : Equiv A B C 20] [H3 : Equiv A C B 20] [H1 : Equiv B C 24]
  (G_on_AB : G ∈ segment B 20) (H_on_BC : H ∈ segment C 24) (I_on_AC : I ∈ segment A 20)
  (GH_parallel_AC : line GH ∥ line AC) (HI_parallel_AB : line HI ∥ line AB)
  (triangles_similarity : (Triangle A B G H) ≃ (Triangle H I C))
: perimeter (Parallelogram A G H I) = 40 := 
sorry

end parallelogram_AGHI_perimeter_l161_161980


namespace new_lamp_height_is_correct_l161_161652

-- Define the height of the old lamp
def old_lamp_height : ℝ := 1

-- Define the additional height of the new lamp
def additional_height : ℝ := 1.33

-- Proof statement
theorem new_lamp_height_is_correct :
  old_lamp_height + additional_height = 2.33 :=
sorry

end new_lamp_height_is_correct_l161_161652


namespace integers_even_condition_l161_161199

-- Definitions based on conditions
def is_even (n : ℤ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℤ) : Prop :=
(is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ ¬ is_even b ∧ is_even c)

-- Proof statement
theorem integers_even_condition (a b c : ℤ) (h : ¬ exactly_one_even a b c) :
  (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c) :=
sorry

end integers_even_condition_l161_161199


namespace simplify_expression_l161_161782

theorem simplify_expression : (3^3 * 3^(-4)) / (3^2 * 3^(-5)) = 1 / 6561 := by
  sorry

end simplify_expression_l161_161782


namespace largest_of_five_consecutive_sum_l161_161753

theorem largest_of_five_consecutive_sum (n : ℕ) 
  (h : n + (n+1) + (n+2) + (n+3) + (n+4) = 90) : 
  n + 4 = 20 :=
sorry

end largest_of_five_consecutive_sum_l161_161753


namespace sum_of_complex_series_l161_161890

noncomputable def complex_series_sum : ℂ :=
  ∑ k in finset.range (2002 + 1), (2 * k : ℂ) * complex.I ^ k

theorem sum_of_complex_series :
  complex_series_sum = -2000 + 1999 * complex.I :=
by
  sorry

end sum_of_complex_series_l161_161890


namespace solve_system_l161_161591

theorem solve_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y = 4 * z) (h2 : x / y = 81) (h3 : x * z = 36) :
  x = 36 ∧ y = 4 / 9 ∧ z = 1 :=
by
  sorry

end solve_system_l161_161591


namespace lighthouse_coverage_l161_161547

theorem lighthouse_coverage (A B C D : Point) :
  ∃ φA φB φC φD : ℝ,
    (lamp A 90 φA).covers_plane ∧ 
    (lamp B 90 φB).covers_plane ∧
    (lamp C 90 φC).covers_plane ∧
    (lamp D 90 φD).covers_plane :=
by
  sorry

end lighthouse_coverage_l161_161547


namespace train_speed_l161_161024

-- Definitions for the problem conditions
def distance_julie_traveled : ℝ := 12 -- miles
def distance_jim_walked : ℝ := 3.5 -- miles
def distance_jim_traveled_on_train : ℝ := Real.sqrt (12^2 + 3.5^2) -- miles

-- Time taken by Jim on the train from 1:00 PM to 1:12 PM, which is 1/5 hours.
def time_jim_on_train : ℝ := 1 / 5 -- hours

-- Calculating the train's speed
def speed_of_train : ℝ :=
  distance_jim_traveled_on_train / time_jim_on_train -- miles per hour

-- Statement to prove the speed of the train equals 62.5 mph
theorem train_speed : speed_of_train = 62.5 := by
  sorry

end train_speed_l161_161024


namespace smallest_k_divides_ab_l161_161260

theorem smallest_k_divides_ab (S : Finset ℕ) (hS : S = Finset.range (50 + 1).erase 0) :
  ∃ k : ℕ, (∀ T : Finset ℕ, T ⊆ S → T.card = k → ∃ a b ∈ T, a ≠ b ∧ (a + b) ∣ (a * b)) ∧ k = 39 := 
by
  let S := (Finset.range (50 + 1)).erase 0
  have hS : S = (Finset.range 51).erase 0 := rfl
  existsi 39
  split
  · intro T hT hTcard
    sorry   -- proof will go here
  · rfl

end smallest_k_divides_ab_l161_161260


namespace magnitude_projection_a_onto_b_l161_161565

def a : ℝ × ℝ × ℝ := (1, 0, 1)
def b : ℝ × ℝ × ℝ := (2, 1, 2)

theorem magnitude_projection_a_onto_b : 
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2 + b.3 * b.3)
  ∃ proj_magnitude : ℝ, proj_magnitude = abs (dot_product / magnitude_b) ∧ proj_magnitude = 4 / 3 :=
sorry

end magnitude_projection_a_onto_b_l161_161565


namespace birds_more_than_storks_l161_161807

theorem birds_more_than_storks :
  let birds := 6
  let initial_storks := 3
  let additional_storks := 2
  let total_storks := initial_storks + additional_storks
  birds - total_storks = 1 := by
  sorry

end birds_more_than_storks_l161_161807


namespace savings_increase_percentage_l161_161030

variables (I : ℝ) (savings_rate : ℝ) (income_increase : ℝ)
variables (initial_savings second_year_savings : ℝ)
variables (E1 E2 : ℝ)

-- Conditions
def initial_income := I
def savings := savings_rate * I
def expenditure1 := I - savings
def income2 := (1 + income_increase) * I
def expenditure2 := expenditure1
def total_expenditure := expenditure1 + expenditure2
def savings2 := income2 - expenditure2
def doubled_expenditure := 2 * expenditure1

-- Given Conditions
axiom savings_rate_20 : savings_rate = 0.2
axiom income_increase_20 : income_increase = 0.2
axiom total_expenditure_double : total_expenditure = doubled_expenditure
axiom expenditure_verified : E1 = expenditure1 ∧ E2 = expenditure2

-- Correct answer
theorem savings_increase_percentage : 
  (savings_rate_20 ∧ income_increase_20 ∧ expenditure_verified) → 
  savings2 - savings = 0.2 * I → 
  (savings2 - savings) / savings * 100 = 100 :=
by
  sorry

end savings_increase_percentage_l161_161030


namespace calculate_price_per_pound_of_meat_l161_161655

noncomputable def price_per_pound_of_meat : ℝ :=
  let total_hours := 50
  let w := 8
  let m_pounds := 20
  let fv_pounds := 15
  let fv_pp := 4
  let b_pounds := 60
  let b_pp := 1.5
  let j_wage := 10
  let j_hours := 10
  let j_rate := 1.5

  -- known costs
  let fv_cost := fv_pounds * fv_pp
  let b_cost := b_pounds * b_pp
  let j_cost := j_hours * j_wage * j_rate

  -- total costs
  let total_cost := total_hours * w
  let known_costs := fv_cost + b_cost + j_cost

  (total_cost - known_costs) / m_pounds

theorem calculate_price_per_pound_of_meat : price_per_pound_of_meat = 5 := by
  sorry

end calculate_price_per_pound_of_meat_l161_161655


namespace mass_of_quarter_ellipse_l161_161449

-- Defining the quarter ellipse
def quarter_ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1) ∧ (x ≥ 0) ∧ (y ≥ 0)

-- Defining the density function
def density (x y : ℝ) : ℝ := x * y

-- Statement of the main theorem
theorem mass_of_quarter_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ m, (∀ x y, quarter_ellipse a b x y → density x y) → m = (a * b * π) / 4 :=
begin
  sorry,
end

end mass_of_quarter_ellipse_l161_161449


namespace inequality_positive_real_xyz_l161_161684

theorem inequality_positive_real_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) := 
by
  -- Proof is to be constructed here
  sorry

end inequality_positive_real_xyz_l161_161684


namespace price_of_sour_apple_l161_161451

theorem price_of_sour_apple (sweet_price : ℝ) (sour_fraction : ℝ) (total_apples : ℕ) (total_earnings : ℝ) (num_sweet_apples : ℕ) (num_sour_apples : ℕ) (x : ℝ) :
  sweet_price = 0.5 →
  sour_fraction = 0.25 →
  total_apples = 100 →
  total_earnings = 40 →
  num_sweet_apples = 75 →
  num_sour_apples = 25 →
  37.5 + 25 * x = 40 →
  x = 0.1 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7
  rw h1 at h7
  rw h2 at h7
  rw h3 at h7
  rw h4 at h7
  rw h5 at h7
  rw h6 at h7
  -- Solve the equation 37.5 + 25 * x = 40
  have h8 : 25 * x = 2.5 := by linarith
  have h9 : x = 0.1 := by linarith
  exact h9

end price_of_sour_apple_l161_161451


namespace magnitude_projection_a_onto_b_l161_161566

def a : ℝ × ℝ × ℝ := (1, 0, 1)
def b : ℝ × ℝ × ℝ := (2, 1, 2)

theorem magnitude_projection_a_onto_b : 
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2 + b.3 * b.3)
  ∃ proj_magnitude : ℝ, proj_magnitude = abs (dot_product / magnitude_b) ∧ proj_magnitude = 4 / 3 :=
sorry

end magnitude_projection_a_onto_b_l161_161566


namespace Alice_and_Dave_weight_l161_161848

variable (a b c d : ℕ)

-- Conditions
variable (h1 : a + b = 230)
variable (h2 : b + c = 220)
variable (h3 : c + d = 250)

-- Proof statement
theorem Alice_and_Dave_weight :
  a + d = 260 :=
sorry

end Alice_and_Dave_weight_l161_161848


namespace tagged_fish_ratio_l161_161211

theorem tagged_fish_ratio (tagged_first_catch : ℕ) (total_second_catch : ℕ) (tagged_second_catch : ℕ) 
  (approx_total_fish : ℕ) (h1 : tagged_first_catch = 60) 
  (h2 : total_second_catch = 50) 
  (h3 : tagged_second_catch = 2) 
  (h4 : approx_total_fish = 1500) :
  tagged_second_catch / total_second_catch = 1 / 25 := by
  sorry

end tagged_fish_ratio_l161_161211


namespace amit_worked_days_l161_161440

theorem amit_worked_days (W : ℝ) (x : ℝ)
  (amit_rate : W / 10) 
  (ananthu_rate : W / 20)
  (total_days : 18)
  (total_work : x * amit_rate + (total_days - x) * ananthu_rate = W) : 
  x = 2 :=
sorry

end amit_worked_days_l161_161440


namespace circle_center_sum_l161_161533

theorem circle_center_sum (x y : ℝ) :
  x^2 + y^2 = 6 * x - 8 * y + 9 → (∃ h k r, (x - h)^2 + (y - k)^2 = r^2 ∧ h + k = -1) :=
begin
  -- proof goes here
  sorry
end

end circle_center_sum_l161_161533


namespace arrangement_count_l161_161362

def numArrangements : Nat := 15000

theorem arrangement_count (students events : ℕ) (nA nB : ℕ) 
  (A_ne_B : nA ≠ nB) 
  (all_students : students = 7) 
  (all_events : events = 5) 
  (one_event_per_student : ∀ (e : ℕ), e < events → ∃ s, s < students ∧ (∀ (s' : ℕ), s' < students → s' ≠ s → e ≠ s')) :
  numArrangements = 15000 := 
sorry

end arrangement_count_l161_161362


namespace approximate_probability_hit_shot_l161_161812

-- Define the data from the table
def shots : List ℕ := [10, 50, 100, 150, 200, 500, 1000, 2000]
def hits : List ℕ := [9, 40, 70, 108, 143, 361, 721, 1440]
def hit_rates : List ℚ := [0.9, 0.8, 0.7, 0.72, 0.715, 0.722, 0.721, 0.72]

-- State the theorem that the stabilized hit rate is approximately 0.72
theorem approximate_probability_hit_shot : 
  ∃ (p : ℚ), p = 0.72 ∧ 
  ∀ (n : ℕ), n ∈ [150, 200, 500, 1000, 2000] → 
     ∃ (r : ℚ), r = 0.72 ∧ 
     r = (hits.get ⟨shots.indexOf n, sorry⟩ : ℚ) / n := sorry

end approximate_probability_hit_shot_l161_161812


namespace length_of_BD_l161_161994

theorem length_of_BD
  (A B C D : Type)
  [EuclideanGeometry A]
  [RightAngleTriangle A B C]
  [right_angle : RightAngle A B C]
  (hAB : length B A = 40)
  (hAC : length A C = 90)
  (hAD_perp : Perpendicular A D B C) :
  length B D = 20 * sqrt 22009 / 97 :=
by
  sorry

end length_of_BD_l161_161994


namespace floor_neg_seven_over_four_l161_161492

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l161_161492


namespace find_radius_and_area_of_trapezoid_l161_161139

noncomputable def R := 2 * Real.sqrt 7
noncomputable def S_ABCD := 56 * Real.sqrt 7

structure IsoscelesTrapezoid := 
  (A B C D : Point)
  (AD_parallel_BC : Parallel AD BC)
  (AD_greater_BC : AD > BC)
  (circle_omega : Circle)
  (omega_tangent_BC_C : Tangent circle_omega BC C)
  (omega_intersects_CD_E : Intersects circle_omega CD E)
  (CE_eq_7 : Distance C E = 7)
  (ED_eq_9 : Distance E D = 9)

theorem find_radius_and_area_of_trapezoid (tr : IsoscelesTrapezoid) : 
  CircleRadius tr.circle_omega = R ∧ 
  TrapezoidArea tr.A tr.B tr.C tr.D = S_ABCD := 
by
  sorry

end find_radius_and_area_of_trapezoid_l161_161139


namespace problem_statement_l161_161291

theorem problem_statement (n m N k : ℕ)
  (h : (n^2 + 1)^(2^k) * (44 * n^3 + 11 * n^2 + 10 * n + 2) = N^m) :
  m = 1 :=
sorry

end problem_statement_l161_161291


namespace mass_percentage_O_in_acetone_l161_161897

def atomic_mass_C := 12.01 -- g/mol
def atomic_mass_H := 1.008 -- g/mol
def atomic_mass_O := 16.00 -- g/mol

def n_C := 3
def n_H := 6
def n_O := 1

def molar_mass_acetone := (n_C * atomic_mass_C) + (n_H * atomic_mass_H) + (n_O * atomic_mass_O)

def mass_percentage_O := (atomic_mass_O / molar_mass_acetone) * 100

theorem mass_percentage_O_in_acetone :
  mass_percentage_O ≈ 27.55 :=
sorry

end mass_percentage_O_in_acetone_l161_161897


namespace part1_part2_l161_161269

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

-- (1) Given a = -1, prove that the inequality f(x, -1) ≤ 0 implies x ≤ -1/3
theorem part1 (x : ℝ) : (f x (-1) ≤ 0) ↔ (x ≤ -1/3) :=
by
  sorry

-- (2) Given f(x) ≥ 0 for all x ≥ -1, prove that the range for a is a ≤ -3 or a ≥ 1
theorem part2 (a : ℝ) : (∀ x, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2_l161_161269


namespace proof_problem_l161_161637

-- Conditions given in the problem
def param_eq_line (t : ℝ) : ℝ × ℝ := (⟨ (sqrt 2) / 2 * t, 3 + (sqrt 2) / 2 * t ⟩ : ℝ × ℝ)
def polar_eq_curve (theta : ℝ) : ℝ := 4 * sin theta - 2 * cos theta

-- Cartesian equation of line l
def line_eq (x y : ℝ) : Prop := x - y + 3 = 0

-- Cartesian equation of curve C
def curve_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y = 0

-- Prove that the given conditions lead to these results
theorem proof_problem (x y : ℝ)
  (t θ : ℝ)
  (P : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (H1 : param_eq_line t = (x, y))
  (H2 : polar_eq_curve θ = sqrt (x^2 + y^2))
  (H3 : line_eq x y)
  (H4 : curve_eq x y)
  (H5 : P = (0, y)) -- intersection with y-axis
  (H6 : A ≠ B)
  (H7 : A = param_eq_line t)
  (H8 : B = param_eq_line t) :
  (line_eq x y) ∧ (curve_eq x y) ∧ 
  (∃ t1 t2 : ℝ, ((t1 + t2 = -2 * sqrt 2) ∧ (t1 * t2 = -3)) ∧ 
  ((1 / abs (sqrt (2 * (t1) ^ 2))) + (1 / abs (sqrt (2 * (t2) ^ 2))) = 2 * sqrt 5 / 3)) :=
sorry

end proof_problem_l161_161637


namespace total_fruit_punch_eq_21_l161_161309

def orange_punch : ℝ := 4.5
def cherry_punch := 2 * orange_punch
def apple_juice := cherry_punch - 1.5

theorem total_fruit_punch_eq_21 : orange_punch + cherry_punch + apple_juice = 21 := by 
  -- This is where the proof would go
  sorry

end total_fruit_punch_eq_21_l161_161309


namespace algebraic_expression_value_l161_161157

theorem algebraic_expression_value (m x n : ℝ)
  (h1 : (m + 3) * x ^ (|m| - 2) + 6 * m = 0)
  (h2 : n * x - 5 = x * (3 - n))
  (h3 : |m| = 2)
  (h4 : (m + 3) ≠ 0) :
  (m + x) ^ 2000 * (-m ^ 2 * n + x * n ^ 2) + 1 = 1 := by
  sorry

end algebraic_expression_value_l161_161157


namespace num_valid_even_numbers_l161_161185

def is_valid_digit (d : ℕ) : Prop :=
  d ∈ {1, 3, 4, 5, 6, 8}

def valid_even_number (n : ℕ) : Prop :=
  300 ≤ n ∧ n ≤ 800 ∧
  (∃ d1 d2 d3,
    n = d1 * 100 + d2 * 10 + d3 ∧
    d3 % 2 = 0 ∧  -- even last digit
    is_valid_digit d1 ∧
    is_valid_digit d2 ∧
    is_valid_digit d3 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)

theorem num_valid_even_numbers : (finset.range (800 + 1)).filter valid_even_number).card = 40 :=
sorry

end num_valid_even_numbers_l161_161185


namespace fewest_cookies_l161_161402

theorem fewest_cookies
  (area_art_cookies : ℝ)
  (area_roger_cookies : ℝ)
  (area_paul_cookies : ℝ)
  (area_trisha_cookies : ℝ)
  (h_art : area_art_cookies = 12)
  (h_roger : area_roger_cookies = 8)
  (h_paul : area_paul_cookies = 6)
  (h_trisha : area_trisha_cookies = 6)
  (dough : ℝ) :
  (dough / area_art_cookies) < (dough / area_roger_cookies) ∧
  (dough / area_art_cookies) < (dough / area_paul_cookies) ∧
  (dough / area_art_cookies) < (dough / area_trisha_cookies) := by
  sorry

end fewest_cookies_l161_161402


namespace net_change_wealth_l161_161696

-- Define the initial conditions
def initial_cash_A := 15000
def initial_house_value := 15000
def initial_cash_B := 20000
def first_transaction_price := 18000
def second_transaction_price := 12000

-- Prove the final net change in wealth for Mr. A and Mr. B
theorem net_change_wealth :
  let final_cash_A := initial_cash_A + first_transaction_price - second_transaction_price,
    final_cash_B := initial_cash_B - first_transaction_price + second_transaction_price,
    net_change_A := final_cash_A - initial_cash_A,
    net_change_B := final_cash_B - initial_cash_B
  in net_change_A = 6000 ∧ net_change_B = -6000 :=
by
  sorry

end net_change_wealth_l161_161696


namespace floor_neg_seven_over_four_l161_161498

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l161_161498


namespace calculate_expression_l161_161866

theorem calculate_expression :
  (1 / 2 : ℝ)⁻¹ + |real.sqrt 3 - 2| + real.sqrt 12 = 4 + real.sqrt 3 :=
by
  sorry

end calculate_expression_l161_161866


namespace average_percentage_decrease_l161_161048

theorem average_percentage_decrease (x : ℝ) (h : 0 < x ∧ x < 1) :
  (800 * (1 - x)^2 = 578) → x = 0.15 :=
by
  sorry

end average_percentage_decrease_l161_161048


namespace general_term_a_n_general_term_b_n_T_n_expression_l161_161570

noncomputable def a_n (n : ℕ) : ℕ := 3^(n-1)

noncomputable def b_n (n : ℕ) : ℕ := 2 * n + 1

noncomputable def T_n (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a_n i * b_n i

theorem general_term_a_n 
  (a1 : ℕ) (a4 : ℕ) (h1 : a1 = 1) (h2 : a4 = 27) : 
  ∀ n, a_n n = 3^(n-1) := 
by
  intro n
  sorry

theorem general_term_b_n 
  (b1 : ℕ) (S5 : ℕ) (h3 : b1 = 3) (h4 : S5 = 35) : 
  ∀ n, b_n n = 2 * n + 1 := 
by
  intro n
  sorry

theorem T_n_expression 
  (a1 : ℕ) (a4 : ℕ) (b1 : ℕ) (S5 : ℕ) (h1 : a1 = 1) (h2 : a4 = 27) (h3 : b1 = 3) (h4 : S5 = 35) :
  ∀ n, T_n n = n * 3^n := 
by
  intro n
  sorry

end general_term_a_n_general_term_b_n_T_n_expression_l161_161570


namespace exists_distinct_numbers_divisible_by_3_l161_161088

-- Define the problem in Lean with the given conditions and goal.
theorem exists_distinct_numbers_divisible_by_3 : 
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 ∧ d % 3 = 0 ∧
  (a + b + c) % d = 0 ∧ (a + b + d) % c = 0 ∧ (a + c + d) % b = 0 ∧ (b + c + d) % a = 0 :=
by
  sorry

end exists_distinct_numbers_divisible_by_3_l161_161088


namespace shadow_taller_pot_length_l161_161772

-- Definitions based on the conditions a)
def height_shorter_pot : ℕ := 20
def shadow_shorter_pot : ℕ := 10
def height_taller_pot : ℕ := 40

-- The proof problem
theorem shadow_taller_pot_length : 
  ∃ (S2 : ℕ), (height_shorter_pot / shadow_shorter_pot = height_taller_pot / S2) ∧ S2 = 20 :=
sorry

end shadow_taller_pot_length_l161_161772


namespace peter_completes_fourth_task_at_1_27_PM_l161_161702

noncomputable def start_time : ℕ := 9 * 60 -- 9:00 AM in minutes
noncomputable def third_task_end_time : ℕ := 12 * 60 + 20 -- 12:20 PM in minutes
noncomputable def task_duration : ℕ := (third_task_end_time - start_time) / 3 -- Duration of one task in minutes

theorem peter_completes_fourth_task_at_1_27_PM :
  let fourth_task_end_time := third_task_end_time + task_duration in
  fourth_task_end_time = 13 * 60 + 27 := -- 1:27 PM in minutes
by
  sorry

end peter_completes_fourth_task_at_1_27_PM_l161_161702


namespace book_width_l161_161221

noncomputable def golden_ratio : Real := (1 + Real.sqrt 5) / 2

theorem book_width (length : Real) (width : Real) 
(h1 : length = 20) 
(h2 : width / length = golden_ratio) : 
width = 12.36 := 
by 
  sorry

end book_width_l161_161221


namespace floor_of_neg_seven_fourths_l161_161514

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l161_161514


namespace find_a_l161_161928

theorem find_a (a : ℝ) : 
  (let general_term := λ (r : ℕ), (Nat.choose 6 r) * (-a)^r * x^(3 - (3/2)*r),
       coefficient := general_term 1 in
    coefficient = 30) → a = -5 :=
begin
  intros h,
  sorry
end

end find_a_l161_161928


namespace num_triangles_in_grid_l161_161800

def is_triangle (A B C : ℕ × ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  (B.1 - A.1) * (C.2 - A.2) ≠ (C.1 - A.1) * (B.2 - A.2)

def num_triangles (S : set (ℕ × ℕ)) : ℕ :=
  finset.card ((S.to_finset.powerset_len 3).filter (λ t, t.to_list.nth_le 0 sorry => ∧ t.to_list.nth_le 1 sorry => ∧ t.to_list.nth_le 2 sorry => is_triangle 
    {t.to_list.nth_le 0 sorry, t.to_list.nth_le 1 sorry, t.to_list.nth_le 2 sorry}))

theorem num_triangles_in_grid : 
  let S := {p : ℕ × ℕ | p.1 ∈ {0,1,2,3,4,5} ∧ p.2 ∈ {0,1,2,3,4,5}} in
  num_triangles S = 6768 :=
by
  sorry

end num_triangles_in_grid_l161_161800


namespace sum_of_distances_l161_161976

variables {α : Type*} [linear_ordered_field α]

-- Given conditions
variables (a b c : EuclideanSpace α) (f : ℝ) (hb : ∥b∥ = 2 * ∥c∥ = 2 * √3) (hbc : is_perpendicular b c)

-- Theorem statement
theorem sum_of_distances (ha : is_vector a) (hb : is_vector b) (hc : is_vector c) :
  ∥a - b∥ + ∥a - c∥ + ∥a + c∥ = 4 * √3 :=
sorry

end sum_of_distances_l161_161976


namespace find_a19_l161_161138

variable {α : Type*} [LinearOrderedField α]

-- Defining the conditions of the problem
def arithmetic_sequence (a : ℕ → α) (d : α) := ∀ n, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) := (n * (a 1 + a n) / 2)

variables (a : ℕ → α) (d : α)

-- Conditions
axiom S7_eq_21 : sum_of_first_n_terms a 7 = 21
axiom a2_mul_a6_eq_5 : a 2 * a 6 = 5
axiom d_neg : d < 0

-- The proof goal
theorem find_a19 : (S7_eq_21 → a2_mul_a6_eq_5 → d_neg → arithmetic_sequence a d) → a 19 = -12 :=
sorry

end find_a19_l161_161138


namespace floor_of_neg_seven_fourths_l161_161512

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l161_161512


namespace sequence_sum_l161_161226

theorem sequence_sum :
  ∃ (a : ℕ → ℕ), 
    a 1 = 1 ∧ 
    a 2 = 2 ∧ 
    (∀ n : ℕ, n > 0 → a (n + 2) - a n = 1 + (-1)^n) ∧ 
    (finset.range 51).sum a = 676 :=
begin
  sorry
end

end sequence_sum_l161_161226


namespace floor_neg_seven_over_four_l161_161493

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l161_161493


namespace no_perfect_square_with_digits_six_and_zero_l161_161089

theorem no_perfect_square_with_digits_six_and_zero :
  ¬ ∃ n : ℕ, (∃ k : ℕ, n = k * k) ∧ (∀ d, d ∈ nat.digits 10 n → d = 0 ∨ d = 6): 
by
  sorry

end no_perfect_square_with_digits_six_and_zero_l161_161089


namespace quadrilateral_is_rhombus_l161_161445

variables (A B C D P Q R S: Point)
variables (AB BC CD DA : LineSegment)
variables (APB BQC CRD DSA : Triangle)
variables (PQ RS: LineSegment)
variables (PQRS: Quadrilateral)

def is_similar_isosceles (Δ1 Δ2 : Triangle) : Prop :=
sorry -- definition of similar isosceles triangles

def is_convex (quad : Quadrilateral) : Prop :=
sorry -- definition of convex quadrilateral

def are_bases_of_similar_isosceles (quad : Quadrilateral) (t1 t2 t3 t4 : Triangle) : Prop :=
sorry -- definition that quadrilateral sides are bases of similar isosceles triangles

def is_rectangle (quad : Quadrilateral) : Prop :=
sorry -- definition of a rectangle

theorem quadrilateral_is_rhombus
  (hConvex : is_convex (Quadrilateral.mk A B C D))
  (hSimilarBases : are_bases_of_similar_isosceles (Quadrilateral.mk A B C D) APB BQC CRD DSA)
  (hRectangle : is_rectangle (Quadrilateral.mk P Q R S))
  (h_not_eq : PQ.length ≠ RS.length) :
  is_rhombus (Quadrilateral.mk A B C D) :=
sorry

end quadrilateral_is_rhombus_l161_161445


namespace floor_neg_seven_quarter_l161_161481

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l161_161481


namespace Jason_cards_l161_161234

theorem Jason_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 3) (h2 : cards_bought = 2) : remaining_cards = 1 :=
by
  sorry

end Jason_cards_l161_161234


namespace fraction_square_eq_decimal_l161_161381

theorem fraction_square_eq_decimal :
  ∃ (x : ℚ), x^2 = 0.04000000000000001 ∧ x = 1 / 5 :=
by
  sorry

end fraction_square_eq_decimal_l161_161381


namespace fraction_numerator_l161_161724

theorem fraction_numerator (x : ℚ) 
  (h1 : ∃ (n : ℚ), n = 4 * x - 9) 
  (h2 : x / (4 * x - 9) = 3 / 4) 
  : x = 27 / 8 := sorry

end fraction_numerator_l161_161724


namespace area_triangle_AJB_l161_161635

structure Rectangle :=
  (A B C D : Point)
  (AB : ℝ) (BC : ℝ)
  (AneqB : A ≠ B)
  (AneqD : A ≠ D)
  (BneqC : B ≠ C)
  (AeqB : dist A B = 8)
  (BeqC : dist B C = 4)

structure PointsOnLine :=
  (H I D C : Point)
  (DH : ℝ) (IC : ℝ)
  (HeqD : dist D H = 2)
  (IeqC : dist I C = 1)

structure Intersect :=
  (A B H I J : Point)
  (lineAH : Line A H)
  (lineBI : Line B I)
  (intersectJ : J ∈ lineAH ∧ J ∈ lineBI)

theorem area_triangle_AJB (r : Rectangle) (p : PointsOnLine) (i : Intersect) : 
  area (triangle i.A i.J i.B) = 128 / 5 := 
by sorry

end area_triangle_AJB_l161_161635


namespace total_earnings_l161_161437

-- Definitions from the conditions.
def LaurynEarnings : ℝ := 2000
def AureliaEarnings : ℝ := 0.7 * LaurynEarnings

-- The theorem to prove.
theorem total_earnings (hL : LaurynEarnings = 2000) (hA : AureliaEarnings = 0.7 * 2000) :
    LaurynEarnings + AureliaEarnings = 3400 :=
by
    sorry  -- Placeholder for the proof.

end total_earnings_l161_161437


namespace range_of_a_l161_161265

theorem range_of_a (a : Real) : 
  (∃ (A B : Set ℝ), A = {0, 1} ∧ B = {x | x > a} ∧ A ∩ B = ∅) → a ≥ 1 :=
by
  intro h
  cases h with A hA
  cases hA with B hAB
  cases hAB with hAdef hBdef
  cases hBdef with hAintersect hEmpty
  sorry

end range_of_a_l161_161265


namespace min_distance_sum_l161_161921

def point : Type := ℝ × ℝ

def A : point := (1, 1)
def B : point := (3, 3)
def P (x : ℝ) : point := (x, 0)

def distance (p1 p2 : point) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_sum : 
  ∃ x : ℝ, (∀ y : ℝ, distance (P y) A + distance (P y) B ≥ distance (P 2) A + distance (P 2) B) ∧ distance (P 2) A + distance (P 2) B = 2 * sqrt 5 :=
by
  sorry

end min_distance_sum_l161_161921


namespace value_of_c_l161_161750

theorem value_of_c :
  ∃ c, (∀ x : ℝ, (∃ d₁ : ℝ, x > d₁ ∧ d₁ = 1) →
        (∃ d₂ : ℝ, x > d₂ ∧ d₂ = 2001) →
        (∃ d₃ : ℝ, x > d₃ ∧ d₃ = 2001 ^ 2002) →
        (∃ d₄ : ℝ, x > d₄ ∧ d₄ = 2001 ^ 2002) → x > c) ∧ c = 2001 ^ 2002 :=
begin
  sorry
end

end value_of_c_l161_161750


namespace DE_parallel_or_coincident_FG_l161_161558

open EuclideanGeometry

noncomputable def circumcircle (A B C : Point) : Circle := sorry

def is_acute (A B C : Point) : Prop := sorry

def perpendicular_bisector (P Q : Point) : Line := sorry

def intersects_minor_arc (L : Line) (arc : Arc) : Point := sorry

theorem DE_parallel_or_coincident_FG
  (A B C D E F G : Point)
  (Gamma : Circle)
  (h_triangle : triangle A B C)
  (h_acute : is_acute A B C)
  (h_circumcircle : circumcircle A B C = Gamma)
  (h_D_on_AB : D ∈ segment A B)
  (h_E_on_AC : E ∈ segment A C)
  (h_AD_eq_AE : AD = AE)
  (h_perp_bisector_BD : let P := perpendicular_bisector B D in P intersects_minor_arc overarc (A B) Gamma = F)
  (h_perp_bisector_CE : let Q := perpendicular_bisector C E in Q intersects_minor_arc overarc (A C) Gamma = G) :
  (DE ∥ FG) ∨ (DE = FG) := sorry

end DE_parallel_or_coincident_FG_l161_161558


namespace calculate_beings_l161_161802

-- Define the type for beings: Plant or Zombie
inductive Being
| Plant
| Zombie
  deriving DecidableEq, Repr

-- Define the total number of beings
def total_num_beings : Nat := 11

-- Define the number of zombies
def num_zombies : Nat := 2

-- Define the number of plants
def num_plants : Nat := total_num_beings - num_zombies

-- Define the height arrangement by indexing beings
def height_arrangement : Fin total_num_beings → Being
| 0 => Being.Zombie -- shortest
| 1 => Being.Plant
| 2 => Being.Zombie
| _ => Being.Plant -- the rest are plants

-- Define the condition that "I am shorter than you" was heard 20 times
def shorter_than_you_count : Nat := 20

-- Calculate the claps during goodbye
def claps (n z : Nat) : Nat := 2 * (n - z)

-- Define the number of claps
def num_claps : Nat := 18

-- The main theorem
theorem calculate_beings : 
  total_num_beings = 11 ∧
  height_arrangement (Fin.ofNat 0) = Being.Zombie ∧
  height_arrangement (Fin.ofNat 1) = Being.Plant ∧
  height_arrangement (Fin.ofNat 2) = Being.Zombie ∧
  num_zombies = 2 ∧ 
  num_claps = 18 :=
by
  sorry

end calculate_beings_l161_161802


namespace problem1_problem2_l161_161559

-- Definitions for Part (1)
variables {ABC : Type*} [triangle ABC] 
variables {O : circumcircle ABC}
variables {M N : point O}
variables {C : point ABC}
variables {MN : line}
variables {P T : point O}
variables {I : incenter ABC}

-- assumption that M and N are midpoints
axiom M_midpoint_BC : is_midpoint M (arc O BC)
axiom N_midpoint_AC : is_midpoint N (arc O AC)
axiom line_C_parallel_MN : parallel (line_through C (line_through M N)) MN
-- assumption about intersections
axiom P_on_circumcircle : lies_on P O
axiom T_on_circumcircle : lies_on T O
-- relationship of P and T with circumcircle and line through I
axiom PI_intersects_circumcircle_at_T : intersects_circumcircle_again PI T

-- Problem Part (1)
theorem problem1 : MP * MT = NP * NT := sorry

-- Definitions for Part (2)
variables {Q : point (arc O (AB))}
variables {X Y : incenter Q}

axiom Q_not_in_ABT : (Q ≠ A) ∧ (Q ≠ B) ∧ (Q ≠ T)
axiom X_incenter_AQC : X = incenter (triangle AQC)
axiom Y_incenter_BQC : Y = incenter (triangle BQC)

-- Problem Part (2)
theorem problem2 : concyclic {Q, T, Y, X} := sorry

end problem1_problem2_l161_161559


namespace min_value_of_vec_diff_norm_l161_161907

-- Definitions for conditions
def vector_a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 1 - t, t)
def vector_b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)
def vector_sub (t : ℝ) : ℝ × ℝ × ℝ := let a := vector_a t; let b := vector_b t in (a.1 - b.1, a.2 - b.2, a.3 - b.3)
def vec_norm_sq (v : ℝ × ℝ × ℝ) : ℝ := v.1^2 + v.2^2 + v.3^2
def vec_norm (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (vec_norm_sq v)

-- The theorem statement
theorem min_value_of_vec_diff_norm : ∃ t : ℝ, vec_norm (vector_sub t) = 3 * Real.sqrt 5 / 5 := sorry

end min_value_of_vec_diff_norm_l161_161907


namespace relationship_l161_161882

def g (x : ℝ) : ℝ := x
def r (x : ℝ) : ℝ := Real.log (x + 1)
def φ (x : ℝ) : ℝ := x^3 - 1

def new_stationary_point (f : ℝ → ℝ) : ℝ :=
  @Classical.choose ℝ (λ x, f x = derivative f x) sorry

def α : ℝ := new_stationary_point g
def β : ℝ := new_stationary_point r
def γ : ℝ := new_stationary_point φ

theorem relationship : γ > α ∧ α > β :=
sorry

end relationship_l161_161882


namespace eval_floor_neg_seven_fourths_l161_161508

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l161_161508


namespace john_overall_profit_l161_161397

-- Definitions based on conditions
def cost_grinder : ℕ := 15000
def cost_mobile : ℕ := 8000
def loss_percentage_grinder : ℚ := 4 / 100
def profit_percentage_mobile : ℚ := 15 / 100

-- Calculations based on the conditions
def loss_amount_grinder := cost_grinder * loss_percentage_grinder
def selling_price_grinder := cost_grinder - loss_amount_grinder
def profit_amount_mobile := cost_mobile * profit_percentage_mobile
def selling_price_mobile := cost_mobile + profit_amount_mobile
def total_cost_price := cost_grinder + cost_mobile
def total_selling_price := selling_price_grinder + selling_price_mobile

-- Overall profit calculation
def overall_profit := total_selling_price - total_cost_price

-- Proof statement to prove the overall profit
theorem john_overall_profit : overall_profit = 600 := by
  sorry

end john_overall_profit_l161_161397


namespace maximum_dot_product_l161_161571

def vector_dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem maximum_dot_product (a : ℝ) (t : ℝ) (O A B P : ℝ × ℝ)
  (h1 : a > 0)
  (h2 : A = (a, 0))
  (h3 : B = (0, a))
  (h4 : P.1 = -a * t + a)
  (h5 : P.2 = a * t)
  (h6 : 0 ≤ t ∧ t ≤ 1) :
  ∃ t_max : ℝ, t_max = 0 ∧ vector_dot_product A P = a^2 :=
begin
  sorry
end

end maximum_dot_product_l161_161571


namespace problem1_problem2_problem3_l161_161156

-- Problem 1: Range of the function when a = 1/2
theorem problem1 (x : ℝ) : 
  f (a : ℝ → ℝ) := 1/2 * 4^x - 2^x + 1 ∀ x ∈ ℝ, f x x ∈ (1/2, +∞) :=
sorry

-- Problem 2: Range of a given unique zero point in (0, 1)
theorem problem2 (a : ℝ) : 
  f (a : ℝ → ℝ) := a * 4^x - 2^x + 1 ∀ (x ∈ ℝ) , 0 < x ∈ (0,1) < 1/4 :=
sorry

-- Problem 3: Decreasing function implies a ≤ 0
theorem problem3 (a : ℝ) : 
  f (a : ℝ → ℝ) a ∀ x ∈ ℝ, decreasing_function(a, f) a <= 0 :=
sorry

end problem1_problem2_problem3_l161_161156


namespace rectangle_count_in_3x6_grid_l161_161625

theorem rectangle_count_in_3x6_grid : 
  let grid_height := 3
  let grid_width := 6
  let h_lines := grid_height + 1
  let v_lines := grid_width + 1
  let binom := λ n k, Nat.choose n k
  let horizontal_vertical_rectangles := binom h_lines 2 * binom v_lines 2
  let diagonal_rectangles := 10 + 8 + 8
  in horizontal_vertical_rectangles + diagonal_rectangles = 152 := 
by
  -- Mathematically equivalent constructs and conditions are defined above.
  -- Proof would be given here as per Lean syntax.
  sorry

end rectangle_count_in_3x6_grid_l161_161625


namespace total_boxes_sold_is_189_l161_161714

-- Define the conditions
def boxes_sold_friday : ℕ := 40
def boxes_sold_saturday := 2 * boxes_sold_friday - 10
def boxes_sold_sunday := boxes_sold_saturday / 2
def boxes_sold_monday := boxes_sold_sunday + (boxes_sold_sunday / 4)

-- Define the total boxes sold over the four days
def total_boxes_sold := boxes_sold_friday + boxes_sold_saturday + boxes_sold_sunday + boxes_sold_monday

-- Theorem to prove the total number of boxes sold is 189
theorem total_boxes_sold_is_189 : total_boxes_sold = 189 := by
  sorry

end total_boxes_sold_is_189_l161_161714


namespace union_intersection_sets_l161_161266

variable (a : ℝ)

def A := {x | x^2 - 2 * x - 3 = 0}
def B := {x | (x - 1) * (x - a) = 0}

-- Define the sets explicitly
def A_set := {-1, 3}
def B_set := if a = 1 then {1} else if a = -1 then {-1, 1} else if a = 3 then {1, 3} else {1, a}

-- Proving A ∪ B and A ∩ B in different scenarios of a
theorem union_intersection_sets :
  (A_set ∪ B_set = if a = 1 then {-1, 1, 3}
                   else if a = -1 then {-1, 1, 3}
                   else if a = 3 then {-1, 1, 3}
                   else {-1, 1, 3, a}) ∧
  (A_set ∩ B_set = if a = 1 then ∅
                   else if a = -1 then {-1}
                   else if a = 3 then {3}
                   else ∅) := by
  sorry

end union_intersection_sets_l161_161266


namespace projection_of_2a_minus_b_l161_161927

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (angle_ab : real.angle) (norm_a norm_b : ℝ)

-- Given conditions
axiom angle_between_a_b : angle_ab = real.angle.pi_div_two
axiom norm_a_val : ∥a∥ = 2
axiom norm_b_val : ∥b∥ = 5

-- Proof statement
theorem projection_of_2a_minus_b (h_angle : angle_ab = real.angle.pi_div_three) 
  (ha : ∥a∥ = 2) (hb : ∥b∥ = 5) : 
  (inner_product_space.proj (2 • a - b) a).norm = 3 / 2 :=
sorry

end projection_of_2a_minus_b_l161_161927


namespace arithmetic_geometric_sequence_sum_l161_161968

theorem arithmetic_geometric_sequence_sum 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y z : ℝ, (x = a ∧ y = -4 ∧ z = b ∨ x = b ∧ y = -4 ∧ z = a) 
                   ∧ (x + z = 2 * y) ∧ (x * z = y^2)) : 
  a + b = 10 :=
by sorry

end arithmetic_geometric_sequence_sum_l161_161968


namespace sqrt_five_squared_minus_four_squared_eq_three_l161_161728

theorem sqrt_five_squared_minus_four_squared_eq_three : Real.sqrt (5 ^ 2 - 4 ^ 2) = 3 := by
  sorry

end sqrt_five_squared_minus_four_squared_eq_three_l161_161728


namespace square_angle_l161_161227

theorem square_angle (PQ QR : ℝ) (x : ℝ) (PQR_is_square : true)
  (angle_sum_of_triangle : ∀ a b c : ℝ, a + b + c = 180)
  (right_angle : ∀ a, a = 90) :
  x = 45 :=
by
  -- We start with the properties of the square (implicitly given by the conditions)
  -- Now use the conditions and provided values to conclude the proof
  sorry

end square_angle_l161_161227


namespace directional_derivative_sin_r_equals_cos_r_l161_161894

noncomputable def r (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)

noncomputable def directional_derivative (f : ℝ → ℝ) (x y z : ℝ) : ℝ :=
  (f ∘ r) (x, y, z)

theorem directional_derivative_sin_r_equals_cos_r (x y z : ℝ) :
    directional_derivative sin x y z = Real.cos (r x y z) :=
  sorry

end directional_derivative_sin_r_equals_cos_r_l161_161894


namespace cos_2alpha_value_l161_161145

theorem cos_2alpha_value (α β : ℝ) (h1 : (π / 2) < β ∧ β < α ∧ α < (3 * π / 4))
    (h2 : cos (α - β) = 12 / 13) (h3 : sin (α + β) = -3 / 5) : 
    cos (2 * α) = -33 / 65 := 
by
  sorry

end cos_2alpha_value_l161_161145


namespace floor_of_neg_seven_fourths_l161_161489

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l161_161489


namespace number_of_sheep_l161_161057

variable (S H C : ℕ)

def ratio_constraint : Prop := 4 * H = 7 * S ∧ 5 * S = 4 * C

def horse_food_per_day (H : ℕ) : ℕ := 230 * H
def sheep_food_per_day (S : ℕ) : ℕ := 150 * S
def cow_food_per_day (C : ℕ) : ℕ := 300 * C

def total_horse_food : Prop := horse_food_per_day H = 12880
def total_sheep_food : Prop := sheep_food_per_day S = 9750
def total_cow_food : Prop := cow_food_per_day C = 15000

theorem number_of_sheep (h1 : ratio_constraint S H C)
                        (h2 : total_horse_food H)
                        (h3 : total_sheep_food S)
                        (h4 : total_cow_food C) :
  S = 98 :=
sorry

end number_of_sheep_l161_161057


namespace triangle_count_with_perimeter_seven_l161_161961

theorem triangle_count_with_perimeter_seven :
  let T := { t : (ℕ × ℕ × ℕ) | 
    let (a, b, c) := t in 
    a + b + c = 7 ∧ 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    a + b > c ∧ 
    a + c > b ∧ 
    b + c > a 
  } in 
  T.to_finset.card = 2 :=
by sorry

end triangle_count_with_perimeter_seven_l161_161961


namespace train_length_is_900_l161_161044

-- Definitions of the given conditions
def speed_of_train := 63 -- in km/hr
def time_taken_to_cross := 53.99568034557235 -- in seconds
def speed_of_man := 3 -- in km/hr

-- Convert speeds to consistent units
def relative_speed_in_km_hr := speed_of_train - speed_of_man
def relative_speed_in_m_s := (relative_speed_in_km_hr * 1000) / 3600

-- Define the length of the train based on the given conditions
def length_of_train := relative_speed_in_m_s * time_taken_to_cross

-- The theorem to prove
theorem train_length_is_900 :
  length_of_train ≈ 900 :=
by
  sorry

end train_length_is_900_l161_161044


namespace smallest_natural_number_with_digit_sum_47_l161_161539

-- Define the function that calculates sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- State the theorem for the problem
theorem smallest_natural_number_with_digit_sum_47 : ∃ n : ℕ, sum_of_digits n = 47 ∧ ∀ k : ℕ, sum_of_digits k = 47 → k ≥ n :=
  ∃ n, sum_of_digits n = 47 ∧ ∀ k, sum_of_digits k = 47 → k ≥ n ∧ n = 299999

end smallest_natural_number_with_digit_sum_47_l161_161539


namespace problem_statement_l161_161991

open Real

noncomputable def curve1 (a : ℝ) (φ : ℝ) : ℝ × ℝ :=
  (a + a * cos φ, a * sin φ)

noncomputable def curve2 (b : ℝ) (φ : ℝ) : ℝ × ℝ :=
  (b * cos φ, b + b * sin φ)

noncomputable def ray (α : ℝ) (ρ : ℝ) : ℝ × ℝ :=
  (ρ * cos α, ρ * sin α)

def condition1 (a : ℝ) : Prop :=
  ∀ O A φ, curve1 a φ = (O, A)

def condition2 (b : ℝ) : Prop :=
  ∀ O B φ, curve2 b φ = (O, B)

def condition3 (α : ℝ) (ρ : ℝ) : Prop :=
  ∃ A B, ray α ρ = (A, B)

def condition4 (a : ℝ) : Prop :=
  ∃ α, α = 0 ∧ (ray α 1).fst = 1

def condition5 (b : ℝ) : Prop :=
  ∃ α, α = π / 2 ∧ (ray α 2).snd = 2

theorem problem_statement :
  ∃ a b, condition1 a ∧ condition2 b ∧ condition3 0 1 ∧ 
          condition4 a ∧ condition3 (π / 2) 2 ∧ condition5 b ∧
          a = (1 / 2) ∧ b = 1 ∧ 
          (∀ θ, 2 * (cos θ)^2 + 2 * (sin θ) * (cos θ) ≤ (sqrt 2) + 1) :=
begin
  sorry -- Proof is omitted
end

end problem_statement_l161_161991


namespace sum_of_squares_of_roots_eq_3232_l161_161540

theorem sum_of_squares_of_roots_eq_3232 :
  (∀ x : ℝ, (x^2 + 6*x)^2 - 1580*(x^2 + 6*x) + 1581 = 0) →
  let roots := { x : ℝ | (x + 3)^2 = 1587 ∨ (x + 3)^2 = 11 } in
  let sum_of_squares := ∑ x in roots, x^2 in
  sum_of_squares = 3232 :=
begin
  sorry
end

end sum_of_squares_of_roots_eq_3232_l161_161540


namespace complex_power_sum_l161_161076

theorem complex_power_sum : 
  ∀ (i : ℂ), i^2 = -1 → (∑ n in Finset.range 604, i^n) + 3 = 3 :=
by
  assume i h
  sorry

end complex_power_sum_l161_161076


namespace trigonometric_identity_l161_161122

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (3 * Real.cos x - Real.sin x) = 3 := 
by
  sorry

end trigonometric_identity_l161_161122


namespace friends_total_earnings_l161_161436

def Lauryn_earnings : ℝ := 2000
def Aurelia_fraction : ℝ := 0.7

def Aurelia_earnings : ℝ := Aurelia_fraction * Lauryn_earnings

def total_earnings : ℝ := Lauryn_earnings + Aurelia_earnings

theorem friends_total_earnings : total_earnings = 3400 := by
  -- We defined everything necessary here; the exact proof steps are omitted as per instructions.
  sorry

end friends_total_earnings_l161_161436


namespace gift_total_amount_l161_161618

theorem gift_total_amount (n : ℕ) (contributions : ℕ → ℝ) (hn : n = 10) 
(h_each_min : ∀ i, 0 ≤ i < n → contributions i ≥ 1)
(h_max : ∃ i, 0 ≤ i < n ∧ contributions i = 11) :
  (∑ i in finset.range n, contributions i) = 20 := 
sorry

end gift_total_amount_l161_161618


namespace speed_of_A_l161_161042

-- Define the speeds and times
def VB : ℝ := 5.555555555555555
def timeToOvertake : ℝ := 1.8
def headStartTime : ℝ := 0.5
def totalTimeA : ℝ := timeToOvertake + headStartTime

theorem speed_of_A :
    ∃ (VA : ℝ),
      VA * totalTimeA = VB * timeToOvertake :=
begin
    -- Proof goes here
    sorry
end

end speed_of_A_l161_161042


namespace number_of_trees_in_yard_l161_161209

-- Define the conditions as variables/constants
def yard_length : ℕ := 180
def distance_between_trees : ℕ := 18

-- Define the math proof problem.
theorem number_of_trees_in_yard : 
  ∃ n : ℕ, 
  (yard_length / distance_between_trees + 2 = n) ∧ n = 12 := 
by
  let spaces := yard_length / distance_between_trees
  let total_trees := spaces + 2
  use total_trees
  split
  · rfl
  · exact total_trees = 12

end number_of_trees_in_yard_l161_161209


namespace no_lines_parallel_in_plane_l161_161912

variable (a : Line) (α : Plane)

-- Conditions
axiom not_parallel_a_α : ¬ parallel a α
axiom not_contained_a_α : ¬ contained_in a α

-- Proof problem
theorem no_lines_parallel_in_plane : ∀ (l : Line), l ∈ α → ¬ parallel l a :=
  by
  sorry

end no_lines_parallel_in_plane_l161_161912


namespace min_value_f_l161_161537

noncomputable def f (x : ℝ) : ℝ := 25^x - 5^x + 2

theorem min_value_f : ∃ x ∈ ℝ, ∀ y ∈ ℝ, f y ≥ f x ∧ f x = 5 / 4 :=
by
  sorry

end min_value_f_l161_161537


namespace condition_for_M_eq_N_l161_161665

theorem condition_for_M_eq_N (a1 b1 c1 a2 b2 c2 : ℝ) 
    (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0) 
    (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) → 
    (M = {x : ℝ | a1 * x ^ 2 + b1 * x + c1 > 0} ∧
     N = {x : ℝ | a2 * x ^ 2 + b2 * x + c2 > 0} →
    (¬ (M = N))) ∨ (¬ (N = {} ↔ (M = N))) :=
sorry

end condition_for_M_eq_N_l161_161665


namespace friends_total_earnings_l161_161435

def Lauryn_earnings : ℝ := 2000
def Aurelia_fraction : ℝ := 0.7

def Aurelia_earnings : ℝ := Aurelia_fraction * Lauryn_earnings

def total_earnings : ℝ := Lauryn_earnings + Aurelia_earnings

theorem friends_total_earnings : total_earnings = 3400 := by
  -- We defined everything necessary here; the exact proof steps are omitted as per instructions.
  sorry

end friends_total_earnings_l161_161435


namespace sum_bn_l161_161174

def a_n (n : ℕ) : ℚ :=
  (finset.range n).sum (λ k, (k + 1 : ℚ) / (n + 1))

def b_n (n : ℕ) : ℚ :=
  (1 : ℚ) / (a_n n * a_n (n + 1))

theorem sum_bn (n : ℕ) :
  (finset.range n).sum b_n = 4 * n / (n + 1) :=
by
  sorry

end sum_bn_l161_161174


namespace find_angle_between_vectors_l161_161563

noncomputable def vector_length {n : Type*} [NormedGroup n] (v : n) : ℝ := ∥v∥

noncomputable def dot_product {n : Type*} [InnerProductSpace ℝ n] (v w : n) : ℝ := ⟪v, w⟫

variable {V : Type*} [InnerProductSpace ℝ V] 
variable (a b : V) 
variable (theta : ℝ)
variable (ha : vector_length a ≠ 0) 
variable (hb : vector_length b ≠ 0)
variable (h1 : vector_length a = (2 * real.sqrt 2 / 3) * vector_length b)
variable (h2 : dot_product (a - b) (3 * a + 2 * b) = 0)

theorem find_angle_between_vectors
  (ha : vector_length a ≠ 0)
  (hb : vector_length b ≠ 0)
  (h1 : vector_length a = (2 * real.sqrt 2 / 3) * vector_length b)
  (h2 : dot_product (a - b) (3 * a + 2 * b) = 0) :
  theta = real.arccos (2 / 3) :=
sorry

end find_angle_between_vectors_l161_161563


namespace charge_per_copy_Y_correct_l161_161906

-- Definitions based on conditions
def charge_per_copy_X := 1.20
def total_charge_70_copies_X := 70 * charge_per_copy_X
def additional_charge := 35
def total_charge_70_copies_Y := total_charge_70_copies_X + additional_charge

-- The proof problem statement
theorem charge_per_copy_Y_correct :
  (total_charge_70_copies_Y / 70) = 1.70 :=
by
  -- This is where the proof steps would go, for now we use sorry to skip it.
  sorry

end charge_per_copy_Y_correct_l161_161906


namespace sequence_a_n_sequence_T_n_l161_161250

open Nat

theorem sequence_a_n (n : ℕ) : 
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (S_n : ∀ n : ℕ, S n = ∑ i in range n, a i)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 1) = 2 * S n + 1) :
  a n = 3 ^ (n - 1) :=
sorry

theorem sequence_T_n (n : ℕ) : 
  (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)
  (S : ℕ → ℕ)
  (S_n : ∀ n : ℕ, S n = ∑ i in range n, a i)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 1) = 2 * S n + 1)
  (b_n : ∀ n : ℕ, b n = (3 * n - 1) * a n)
  (T_n : ∀ n : ℕ, T n = ∑ i in range n, b i) :
  T n = (3 * n / 2 - 5 / 4) * 3 ^ n + 5 / 4 :=
sorry

end sequence_a_n_sequence_T_n_l161_161250


namespace sqrt_1105_has_32_integer_points_l161_161799

theorem sqrt_1105_has_32_integer_points
    (x y : ℤ) (h : x^2 + y^2 = 1105) :
    ∃ (n : ℕ), n = 32 ∧ -- There are exactly 32 such integer pairs (x, y)
    (n = (∑ (x y : ℤ), if x^2 + y^2 = 1105 then 1 else 0)) := 
sorry

end sqrt_1105_has_32_integer_points_l161_161799


namespace floor_neg_seven_fourths_l161_161470

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l161_161470


namespace sum_of_exponents_2023_l161_161523

theorem sum_of_exponents_2023 : ∃ (exponents : List ℕ), 
  (∀ (e₁ e₂ : ℕ), e₁ ∈ exponents → e₂ ∈ exponents → e₁ ≠ e₂) ∧
  List.sum (exponents.map (λ e, 2^e)) = 2023 ∧
  List.sum exponents = 48 :=
by
  sorry

end sum_of_exponents_2023_l161_161523


namespace find_pure_imaginary_solutions_l161_161373

noncomputable def poly_eq_zero (x : ℂ) : Prop :=
  x^4 - 6 * x^3 + 13 * x^2 - 42 * x - 72 = 0

noncomputable def is_imaginary (x : ℂ) : Prop :=
  x.im ≠ 0 ∧ x.re = 0

theorem find_pure_imaginary_solutions :
  ∀ x : ℂ, poly_eq_zero x ∧ is_imaginary x ↔ (x = Complex.I * Real.sqrt 7 ∨ x = -Complex.I * Real.sqrt 7) :=
by sorry

end find_pure_imaginary_solutions_l161_161373


namespace angle_between_intersecting_chords_l161_161716

theorem angle_between_intersecting_chords (α β : ℝ) : 
  ∃ θ : ℝ, θ = (α + β) / 2 :=
by
  use (α + β) / 2
  sorry

end angle_between_intersecting_chords_l161_161716


namespace no_polynomials_for_harmonic_series_l161_161008

theorem no_polynomials_for_harmonic_series (p q : Polynomial ℝ) :
    ∀ n : ℕ, 1 + (Finset.range (n+1)).sum (λ i, 1 / (i+1 : ℝ)) ≠ (p.eval n) / (q.eval n) := sorry

end no_polynomials_for_harmonic_series_l161_161008


namespace triangle_perimeter_range_l161_161230

noncomputable def perimeter_range (a b c : ℝ) (B : ℝ) : Prop :=
  sin((3 / 2) * B + π / 4) = sqrt 2 / 2 ∧
  a + c = 2 ∧
  0 < B ∧ B < π →
  3 ≤ a + b + c ∧ a + b + c < 4

theorem triangle_perimeter_range (a b c B : ℝ) :
  sin((3 / 2) * B + π / 4) = sqrt 2 / 2 →
  a + c = 2 →
  0 < B → B < π →
  3 ≤ a + b + c ∧ a + b + c < 4 :=
by
  sorry

end triangle_perimeter_range_l161_161230


namespace pendulum_period_l161_161431

variable (m : ℝ) (L : ℝ) (g : ℝ) (θ₀ : ℝ) (T₀ : ℝ) (T : ℝ)

-- Given conditions
def is_pendulum (m : ℝ) (L : ℝ) (g : ℝ) (θ₀ : ℝ) (T₀ : ℝ) : Prop :=
  θ₀ < Real.pi / 2 ∧
  T₀ = 2 * Real.pi * Real.sqrt (L / g) ∧
  ∀ θ₀, T₀ = 2 * Real.pi * Real.sqrt (L / g) ∧ -- Period formula for length L

axiom air_resistance_friction : True  -- Ignore air resistance and friction

-- New pendulum conditions
def repeated_experiment (T : ℝ) (L : ℝ) (T₀ : ℝ) : Prop :=
  T = 2 * T₀

theorem pendulum_period (h: is_pendulum m L g θ₀ T₀) :
  repeated_experiment T (4 * L) T₀ :=
begin
  unfold is_pendulum at h,
  unfold repeated_experiment,
  rcases h with ⟨htheta₀, hT₀, hθ⟩,
  have hT : T = 2 * T₀,
  { rw hT₀,
    have : 4 * L / g = 4 * (L / g) := by 
    sorry,
    rw [Real.sqrt_mul, Real.sqrt_four],
    rw this,  
    sorry },
  exact hT,
end

end pendulum_period_l161_161431


namespace parabolic_bridge_width_proof_l161_161388

def parabolic_arch_bridge_width (a : ℝ) (vertex_height : ℝ) (initial_width : ℝ) (rise_water_level : ℝ) : ℝ :=
  let x := 4
  let y := -vertex_height
  let a := (x^2) / y
  let new_y := y + rise_water_level
  let new_x_squared := a * new_y
  2 * real.sqrt new_x_squared

theorem parabolic_bridge_width_proof :
  parabolic_arch_bridge_width (-8) 2 8 (1 / 2) = 4 * real.sqrt 3 :=
by
  sorry

end parabolic_bridge_width_proof_l161_161388


namespace intersection_of_lines_l161_161895

theorem intersection_of_lines : ∃ (x y : ℝ), 9 * x - 4 * y = 6 ∧ 7 * x + y = 17 ∧ (x, y) = (2, 3) := 
by
  sorry

end intersection_of_lines_l161_161895


namespace average_student_headcount_l161_161863

theorem average_student_headcount :
  let count_0304 := 10500
  let count_0405 := 10700
  let count_0506 := 11300
  let total_count := count_0304 + count_0405 + count_0506
  let number_of_terms := 3
  let average := total_count / number_of_terms
  average = 10833 :=
by
  sorry

end average_student_headcount_l161_161863


namespace vector_parallel_magnitude_l161_161622

theorem vector_parallel_magnitude (b a : Vector ℝ) (k : ℝ)
  (h1 : a = (2, 1))
  (h2 : b = k • a)
  (h3 : ∥b∥ = 2 * Real.sqrt 5) :
  b = (4, 2) ∨ b = (-4, -2) := 
by
  sorry

end vector_parallel_magnitude_l161_161622


namespace min_distance_zero_l161_161003

variable (U g τ : ℝ)

def y₁ (t : ℝ) : ℝ := U * t - (g * t^2) / 2
def y₂ (t : ℝ) : ℝ := U * (t - τ) - (g * (t - τ)^2) / 2
def s (t : ℝ) : ℝ := |U * τ - g * t * τ + (g * τ^2) / 2|

theorem min_distance_zero
  (U g τ : ℝ)
  (h : 2 * U ≥ g * τ)
  : ∃ t : ℝ, t = τ / 2 + U / g ∧ s t = 0 := sorry

end min_distance_zero_l161_161003


namespace dot_product_a_with_b_plus_c_l161_161568

def vector := ℝ × ℝ × ℝ

def a : vector := (2, -3, 1)
def b : vector := (2, 0, 3)
def c : vector := (0, 2, 2)

def vector_add (u v : vector) : vector :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem dot_product_a_with_b_plus_c : dot_product a (vector_add b c) = 3 :=
  sorry

end dot_product_a_with_b_plus_c_l161_161568


namespace calculate_shaded_area_of_octagon_l161_161826

def area_of_shaded_region (area_octagon : ℝ) (area_sectors : ℝ) : ℝ :=
  area_octagon - area_sectors

theorem calculate_shaded_area_of_octagon :
  let s := 8
  let r := 4
  let num_sides := 8
  let sector_angle := 2*Real.pi / num_sides
  let area_of_octagon := 8 * 32 * Real.sqrt((1 - Real.sqrt(2) / 2) / 2)
  let area_of_one_sector := r^2 * sector_angle / 2
  let area_of_sectors := 8 * area_of_one_sector
  area_of_shaded_region area_of_octagon area_of_sectors = 
    256 * Real.sqrt((1 - Real.sqrt(2) / 2) / 2) - 16 * Real.pi :=
by
  sorry

end calculate_shaded_area_of_octagon_l161_161826


namespace smallest_number_satisfying_conditions_l161_161821

open Nat

def satisfies_condition (n : ℕ) : Prop :=
  (n % 2 = 1) ∧
  (n % 3 = 2) ∧
  (n % 4 = 3) ∧
  (n % 5 = 4) ∧
  (n % 6 = 5) ∧
  (n % 7 = 6) ∧
  (n % 8 = 7) ∧
  (n % 9 = 8) ∧
  (n % 10 = 9)

theorem smallest_number_satisfying_conditions :
  ∃ n : ℕ, satisfies_condition n ∧ ∀ m : ℕ, satisfies_condition m → m ≥ n :=
  ∃ n : ℕ, n = 2519 ∧ satisfies_condition n ∧ ∀ m : ℕ, satisfies_condition m → m ≥ n := by
  exists 2519
  split
  · refl
  · split
    · -- Proof of satisfies_condition 2519
      sorry
    · -- Proof that 2519 is the smallest
      sorry

end smallest_number_satisfying_conditions_l161_161821


namespace machining_defect_probability_l161_161935

theorem machining_defect_probability :
  let defect_rate_process1 := 0.03
  let defect_rate_process2 := 0.05
  let non_defective_rate_process1 := 1 - defect_rate_process1
  let non_defective_rate_process2 := 1 - defect_rate_process2
  let non_defective_rate := non_defective_rate_process1 * non_defective_rate_process2
  let defective_rate := 1 - non_defective_rate
  defective_rate = 0.0785 :=
by
  sorry

end machining_defect_probability_l161_161935


namespace find_f2_l161_161125

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x * f' 2

theorem find_f2 : f 2 = -8 :=
by
  sorry

end find_f2_l161_161125


namespace area_of_trapezoid_TQRS_is_62_l161_161639

/-- Define the given conditions -/
variables (P Q R T S : Type)
variables [IsoscelesTriangle P Q R]
variables (triangles : Finset (Triangle P Q R))
variable (smallest_triangle_area : ℝ)
variable (triangle_PQR_area : ℝ)
variable (area_trapezoid : ℝ)

-- Given conditions as hypotheses
hypothesis (H1 : triangles.card = 8)
hypothesis (H2 : ∀ t ∈ triangles, area t = 2)
hypothesis (H3 : area P Q R = 72)
hypothesis (H4 : area (Triangle P T Q) = 5 * 2)

/-- Proof statement: prove the area of the trapezoid TQRS is 62 -/
theorem area_of_trapezoid_TQRS_is_62 :
  area_trapezoid = triangle_PQR_area - 5 * smallest_triangle_area :=
begin
  rw H3,
  rw H4,
  sorry,
end

end area_of_trapezoid_TQRS_is_62_l161_161639


namespace find_diameter_of_wheel_l161_161205

noncomputable def diameter_of_wheel (N : ℝ) (D : ℝ) : ℝ :=
  D / (N * Real.pi)

theorem find_diameter_of_wheel :
  diameter_of_wheel 15.013648771610555 1320 ≈ 28.01 :=
by {
  sorry
}

end find_diameter_of_wheel_l161_161205


namespace tangent_parallel_l161_161150

theorem tangent_parallel (a b : ℝ) 
  (h1 : b = (1 / 3) * a^3 - (1 / 2) * a^2 + 1) 
  (h2 : (a^2 - a) = 2) : 
  a = 2 ∨ a = -1 :=
by {
  -- proof skipped
  sorry
}

end tangent_parallel_l161_161150


namespace average_speed_ratio_l161_161809

-- Problem Definitions
def speed_still_water : ℝ := 20
def speed_current : ℝ := 4
def distance_each_leg : ℝ := 3

-- Problem Statement
theorem average_speed_ratio :
  let downstream_speed := speed_still_water + speed_current,
      upstream_speed := speed_still_water - speed_current,
      time_downstream := distance_each_leg / downstream_speed,
      time_upstream := distance_each_leg / upstream_speed,
      total_time := time_downstream + time_upstream,
      total_distance := 2 * distance_each_leg,
      average_speed := total_distance / total_time
  in 
  (average_speed / speed_still_water) = 24 / 25 :=
by
  sorry

end average_speed_ratio_l161_161809


namespace find_vector_magnitude_l161_161573

noncomputable def vector_a := (1 / 2 : ℝ, Real.sqrt 3 / 2)
noncomputable def vector_b (θ : ℝ) (magnitude_b : ℝ) : ℝ × ℝ :=
  let angle_in_radians := θ
  let b_magnitude := magnitude_b
  let a_magnitude := 1
  let a_dot_b := a_magnitude * b_magnitude * Real.cos angle_in_radians
  let b_x := magnitude_b * Real.cos (angle_in_radians)
  let b_y := magnitude_b * Real.sin (angle_in_radians)
  (b_x, b_y)

theorem find_vector_magnitude
  (θ : ℝ) (hθ : θ = 2 * Real.pi / 3) 
  (vector_a : ℝ × ℝ) (ha : vector_a = (1 / 2, Real.sqrt 3 / 2))
  (magnitude_b : ℝ) (hmagnitude_b : magnitude_b = 2) :
  ∥(2 * (vector_a.1, vector_a.2) + 3 * (vector_b θ magnitude_b).1, 
  2 * (vector_a.1, vector_a.2) + 3 * (vector_b θ magnitude_b).2)∥ = 2 * Real.sqrt 7 := by
  sorry

end find_vector_magnitude_l161_161573


namespace average_of_25_results_l161_161357

theorem average_of_25_results (first12_avg : ℕ -> ℕ -> ℕ)
                             (last12_avg : ℕ -> ℕ -> ℕ) 
                             (res13 : ℕ)
                             (avg_of_25 : ℕ) :
                             first12_avg 12 10 = 120
                             ∧ last12_avg 12 20 = 240
                             ∧ res13 = 90
                             ∧ avg_of_25 = (first12_avg 12 10 + last12_avg 12 20 + res13) / 25
                             → avg_of_25 = 18 := by
  sorry

end average_of_25_results_l161_161357


namespace volume_pyramid_correct_l161_161114

noncomputable def volume_of_regular_triangular_pyramid 
  (R : ℝ) (β : ℝ) (a : ℝ) : ℝ :=
  (a^3 * (Real.tan β)) / 24

theorem volume_pyramid_correct 
  (R : ℝ) (β : ℝ) (a : ℝ) : 
  volume_of_regular_triangular_pyramid R β a = (a^3 * (Real.tan β)) / 24 :=
sorry

end volume_pyramid_correct_l161_161114


namespace frobenius_divisibility_l161_161659

variables {K : Type*} [field K] [fintype K] 
variables {p : ℕ} [fact (nat.prime p)] [char_p K p]

-- Definition of the polynomial transformation
noncomputable def poly_transform (f : polynomial K) : polynomial K :=
polynomial.sum f (λ i a, polynomial.C a * polynomial.X^(p^i))

theorem frobenius_divisibility {f g : polynomial K} : 
  (poly_transform f) ∣ (poly_transform g) ↔ f ∣ g :=
sorry

end frobenius_divisibility_l161_161659


namespace find_n_l161_161316

-- Definitions based on conditions
variables (n : ℕ)
def avg_age_of_group : Prop := (T : ℤ) (T = n * 14)
def new_avg_age : Prop := (T : ℤ) (T = n * 14) → (T + 32 = (n + 1) * 15)

-- The desired proof statement
theorem find_n (h1 : avg_age_of_group n) (h2 : new_avg_age n) : n = 17 :=
sorry

end find_n_l161_161316


namespace inequality_square_l161_161188

theorem inequality_square (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by
  sorry

end inequality_square_l161_161188


namespace balls_per_color_l161_161275

theorem balls_per_color (total_balls : ℕ) (total_colors : ℕ)
  (h1 : total_balls = 350) (h2 : total_colors = 10) : 
  total_balls / total_colors = 35 :=
by
  sorry

end balls_per_color_l161_161275


namespace max_marked_points_l161_161876

-- Define the set of marked points
variable {Point : Type}
variable (marked_points : Set Point)

-- Define supporting line predicate
def supporting_line_exists (A B : Point) (line : Point → Prop) : Prop :=
  ∃ l : Point → Prop, (∀ p ∈ marked_points, l p) ∧ l A ∧ ¬l B

-- Prove that the maximum number of marked points where supporting line condition holds is 180
theorem max_marked_points (N : ℕ) (hN : ∀ A B : Point, A ∈ marked_points ∧ B ∈ marked_points → 
  supporting_line_exists marked_points A B) (hN_le_180 : N ≤ 180) : N = 180 :=
sorry

end max_marked_points_l161_161876


namespace find_m_l161_161911

-- Defining the hyperbola related parameters
def a : ℝ := 2
def b (m : ℝ) : ℝ := Real.sqrt m
def c (m : ℝ) : ℝ := Real.sqrt (a ^ 2 + m)

-- Defining the eccentricity of the hyperbola
def e (m : ℝ) : ℝ := (c m) / a

-- Conditions of the problem
axiom hyperbola_eq (x y m : ℝ) : (x ^ 2) / 4 - (y ^ 2) / m = 1
axiom eccentricity_eq (m : ℝ) : e m = 1

-- Proof: Find the value of m satisfying the conditions
theorem find_m (m : ℝ) (h1 : e m = 1) : m = 4 / 3 := by
  sorry

end find_m_l161_161911


namespace birds_on_fence_l161_161305

theorem birds_on_fence (B : ℕ) : ∃ B, (∃ S, S = 6 ∧ S = (B + 3) + 1) → B = 2 :=
by
  sorry

end birds_on_fence_l161_161305


namespace pure_imaginary_condition_l161_161671

def z1 : ℂ := 3 - 2 * Complex.I
def z2 (m : ℝ) : ℂ := 1 + m * Complex.I

theorem pure_imaginary_condition (m : ℝ) : z1 * z2 m ∈ {z : ℂ | z.re = 0} ↔ m = -3 / 2 := by
  sorry

end pure_imaginary_condition_l161_161671


namespace inequality_solution_l161_161128

-- Given function is odd and defined as f(x) = (1 - 2^x) / (2^x + 1)
def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 1)

-- The goal is to prove the specified inequality solution
theorem inequality_solution (t : ℝ) :
  f(t^2 - 2*t) + f(2*t^2 - 1) < 0 ↔ t > 1 ∨ t < -1/3 :=
by sorry

end inequality_solution_l161_161128


namespace max_length_BP_squared_l161_161663

variables {A B C T P : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited T] [Inhabited P]

noncomputable def circle_radius : ℝ :=
  12

def AB : ℝ :=
  24

def BP_max2 {A B C T P : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited T] [Inhabited P] : Prop :=
  let radius := circle_radius in
  let AB_len := AB in
  ∃ (BP : ℝ), BP^2 = 1296

theorem max_length_BP_squared :
  BP_max2 := 
sorry

end max_length_BP_squared_l161_161663


namespace otimes_evaluation_l161_161881

def otimes (a b : ℝ) : ℝ := a * b + a - b

theorem otimes_evaluation (a b : ℝ) : 
  otimes a b + otimes (b - a) b = b^2 - b := 
  by
  sorry

end otimes_evaluation_l161_161881


namespace parabola_hyperbola_focus_l161_161203

theorem parabola_hyperbola_focus (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, (y ^ 2 = 2 * p * x) ∧ (x ^ 2 / 4 - y ^ 2 / 5 = 1) → p = 6) :=
by
  sorry

end parabola_hyperbola_focus_l161_161203


namespace beach_relaxing_people_l161_161358

def row1_original := 24
def row1_got_up := 3

def row2_original := 20
def row2_got_up := 5

def row3_original := 18

def total_left_relaxing (r1o r1u r2o r2u r3o : Nat) : Nat :=
  r1o + r2o + r3o - (r1u + r2u)

theorem beach_relaxing_people : total_left_relaxing row1_original row1_got_up row2_original row2_got_up row3_original = 54 :=
by
  sorry

end beach_relaxing_people_l161_161358


namespace no_possible_values_for_n_l161_161349

theorem no_possible_values_for_n (n a : ℤ) (h : n > 1) (d : ℤ := 3) (Sn : ℤ := 180) :
  ∃ n > 1, ∃ k : ℤ, a = k^2 ∧ Sn = n / 2 * (2 * a + (n - 1) * d) :=
sorry

end no_possible_values_for_n_l161_161349


namespace Rose_final_tax_percentage_l161_161278

variable (total_amount : ℝ)
variable (clothing_percent : ℝ)
variable (food_percent : ℝ)
variable (electronics_percent : ℝ)
variable (other_items_percent : ℝ)
variable (clothing_tax : ℝ)
variable (food_tax : ℝ)
variable (electronics_tax : ℝ)
variable (other_items_tax : ℝ)
variable (loyalty_discount : ℝ)

-- Assume the following values from the conditions
def total_amount := 100
def clothing_percent := 0.40
def food_percent := 0.25
def electronics_percent := 0.20
def other_items_percent := 0.15

def clothing_tax := 0.05
def food_tax := 0.02
def electronics_tax := 0.10
def other_items_tax := 0.08

def loyalty_discount := 0.03

-- Calculate the amount spent on each category
def amount_clothing := total_amount * clothing_percent
def amount_food := total_amount * food_percent
def amount_electronics := total_amount * electronics_percent
def amount_other_items := total_amount * other_items_percent

-- Calculate the tax for each category
def tax_clothing := amount_clothing * clothing_tax
def tax_food := amount_food * food_tax
def tax_electronics := amount_electronics * electronics_tax
def tax_other_items := amount_other_items * other_items_tax

-- Calculate the total tax before discount
def total_tax := tax_clothing + tax_food + tax_electronics + tax_other_items

-- Apply the loyalty discount
def discount_on_tax := total_tax * loyalty_discount
def final_tax := total_tax - discount_on_tax

-- Calculate the percentage of the final tax relative to the total amount
def percentage_final_tax := (final_tax / total_amount) * 100

theorem Rose_final_tax_percentage : percentage_final_tax = 5.529 := by
  sorry

end Rose_final_tax_percentage_l161_161278


namespace quadrant_of_point_l161_161703

theorem quadrant_of_point (deg : ℝ) (h1 : deg = 2014) (sin_lt_zero : Real.sin deg < 0) (tan_gt_zero : Real.tan deg > 0) :
  -- Show that the point P(ϰ,ψ) lies in the second quadrant.
  (P : ℝ × ℝ) → P = (Real.sin deg, Real.tan deg) → P.1 < 0 ∧ P.2 > 0 :=
by
  intro P hP
  simp [hP, sin_lt_zero, tan_gt_zero]
  exact ⟨sin_lt_zero, tan_gt_zero⟩

end quadrant_of_point_l161_161703


namespace find_x_l161_161079

noncomputable def arithmetic_sequence (x : ℝ) : Prop := 
  (x + 1) - (1/3) = 4 * x - (x + 1)

theorem find_x :
  ∃ x : ℝ, arithmetic_sequence x ∧ x = 5 / 6 :=
by
  use 5 / 6
  unfold arithmetic_sequence
  sorry

end find_x_l161_161079


namespace negation_of_universal_quantifier_l161_161172

theorem negation_of_universal_quantifier :
  (∀ x : ℝ, sin x ≤ 1) ↔ ¬(∃ x : ℝ, sin x > 1) :=
by
  sorry

end negation_of_universal_quantifier_l161_161172


namespace closest_integer_to_series_sum_l161_161109

theorem closest_integer_to_series_sum :
  round (500 * (∑ n in Finset.range 14997 \ Finset.range 3, 1 / (n + 4)^2 - 9)) = 153 :=
by
  sorry

end closest_integer_to_series_sum_l161_161109


namespace probability_of_event_A_l161_161365

noncomputable def probability_both_pieces_no_less_than_three_meters (L : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  if h : L = a + b 
  then (if a ≥ 3 ∧ b ≥ 3 then (L - 2 * 3) / L else 0)
  else 0

theorem probability_of_event_A : 
  probability_both_pieces_no_less_than_three_meters 11 6 5 = 5 / 11 :=
by
  -- Additional context to ensure proper definition of the problem
  sorry

end probability_of_event_A_l161_161365


namespace cost_per_meter_l161_161344

theorem cost_per_meter (area : ℝ) (total_cost : ℝ) (ratio_length : ℕ) (ratio_width : ℕ)
  (h1 : ratio_length = 3) (h2 : ratio_width = 2) (h3 : area = 3750) (h4 : total_cost = 225) :
  let x := real.sqrt(area / (ratio_length * ratio_width : ℝ))
  let length := (ratio_length : ℝ) * x
  let width := (ratio_width : ℝ) * x
  let perimeter := 2 * (length + width)
  let cost_per_meter := total_cost / perimeter
  cost_per_meter * 100 = 90 :=
begin
  sorry
end

end cost_per_meter_l161_161344


namespace solution_exists_unique_l161_161751

theorem solution_exists_unique (x y : ℝ) : (x + y = 2 ∧ x - y = 0) ↔ (x = 1 ∧ y = 1) := 
by
  sorry

end solution_exists_unique_l161_161751


namespace line_segment_length_l161_161721

noncomputable def center_and_radius : ℝ × ℝ × ℝ :=
  let h₁ : ℝ := 1 in
  let k₁ : ℝ := 1 in
  let r₁ : ℝ := Real.sqrt 2 in
  (h₁, k₁, r₁)

theorem line_segment_length : ∀ (x y : ℝ), 
  (x^2 + y^2 - 2*x - 2*y + 1 = 0) ∧ (x - y = 0) →
  Real.sqrt ((3 / 2 - (-1 / 2)) ^ 2 + (3 / 2 - (-1 / 2)) ^ 2) = 2 * Real.sqrt 2 :=
by
  intros x y h
  have hx : x = 3 / 2 ∨ x = -1 / 2 := sorry
  have hy : y = 3 / 2 ∨ y = -1 / 2 := sorry
  exact Eq.symm sorry

end line_segment_length_l161_161721


namespace eval_nested_radical_l161_161095

-- The statement of the problem in Lean 4
theorem eval_nested_radical :
  let x := sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...))))
  in x = Real.sqrt 13 / 2 - 1/2 :=
sorry  -- No proof required

end eval_nested_radical_l161_161095


namespace square_of_111111111_palindromic_l161_161401

theorem square_of_111111111_palindromic :
  let x : ℕ := 111111111 in
  x^2 = 12345678987654321 :=
by
  let x : ℕ := 111111111
  sorry

end square_of_111111111_palindromic_l161_161401


namespace solve_sparrows_l161_161281

noncomputable def initial_sparrows (x y : ℕ) : Prop :=
  x + y = 25 ∧ x - 5 = 2 * (y - 2)

theorem solve_sparrows : ∃ (x y : ℕ), initial_sparrows x y ∧ x = 17 ∧ y = 8 :=
by
  use 17, 8
  unfold initial_sparrows
  split
  . constructor
    . rfl
    . rfl
  . constructor
    . rfl
    . rfl

end solve_sparrows_l161_161281


namespace pie_eating_contest_l161_161631

theorem pie_eating_contest :
  let first_student_round1 := (5 : ℚ) / 6
  let first_student_round2 := (1 : ℚ) / 6
  let second_student_total := (2 : ℚ) / 3
  let first_student_total := first_student_round1 + first_student_round2
  first_student_total - second_student_total = 1 / 3 :=
by
  sorry

end pie_eating_contest_l161_161631


namespace square_area_l161_161834

theorem square_area (x1 x2 : ℝ) (hx1 : x1^2 + 4 * x1 + 3 = 8) (hx2 : x2^2 + 4 * x2 + 3 = 8) (h_eq : y = 8) : 
  (|x1 - x2|) ^ 2 = 36 :=
sorry

end square_area_l161_161834


namespace razorback_tshirt_money_l161_161314

noncomputable def money_made_from_texas_tech_game (tshirt_price : ℕ) (total_sold : ℕ) (arkansas_sold : ℕ) : ℕ :=
  tshirt_price * (total_sold - arkansas_sold)

theorem razorback_tshirt_money :
  money_made_from_texas_tech_game 78 186 172 = 1092 := by
  sorry

end razorback_tshirt_money_l161_161314


namespace repeating_decimal_product_l161_161102

noncomputable def x : ℚ := 1 / 33
noncomputable def y : ℚ := 1 / 3

theorem repeating_decimal_product :
  (x * y) = 1 / 99 :=
by
  -- Definitions of x and y
  sorry

end repeating_decimal_product_l161_161102


namespace probability_not_all_same_color_l161_161626

def num_colors := 3
def draws := 3
def total_outcomes := num_colors ^ draws

noncomputable def prob_same_color : ℚ := (3 / total_outcomes)
noncomputable def prob_not_same_color : ℚ := 1 - prob_same_color

theorem probability_not_all_same_color :
  prob_not_same_color = 8 / 9 :=
by
  sorry

end probability_not_all_same_color_l161_161626


namespace length_of_ab_correct_l161_161229

-- Defining constants and hypotheses
def radius : ℝ := 3
def total_volume : ℝ := 216 * Real.pi

-- Volume formula for two hemispheres and the cylinder
def volume_hemisphere : ℝ := (2 / 3) * Real.pi * radius^3
def volume_two_hemispheres : ℝ := 2 * volume_hemisphere
def volume_cylinder (height : ℝ) : ℝ := Real.pi * radius^2 * height

-- Define the total volume
def geometric_body_volume (height : ℝ) : ℝ := volume_two_hemispheres + volume_cylinder(height)

-- Prove that given the total volume, the height (length of AB) is 20
theorem length_of_ab_correct :
  (∃ (height : ℝ), geometric_body_volume(height) = total_volume) → (∃ (height : ℝ), height = 20) :=
by
  sorry

end length_of_ab_correct_l161_161229


namespace grid_lines_count_l161_161960

   -- Definition of a 4x4 grid
   def grid_points : Finset (ℕ × ℕ) := 
   Finset.product (Finset.range 4) (Finset.range 4)

   -- Definition of what constitutes a line in the grid
   def is_line (p1 p2 : ℕ × ℕ) : Prop := 
   p1 ≠ p2 ∧ (p1.1 = p2.1 ∨ p1.2 = p2.2 ∨ 
              (p1.1 - p2.1 : ℤ) = (p1.2 - p2.2 : ℤ) ∨ 
              (p1.1 - p2.1 : ℤ) = (p2.2 - p1.2 : ℤ))

   -- The theorem substantiating the number of lines
   theorem grid_lines_count : 
     (Finset.card (Finset.filter (λ p : (ℕ × ℕ) × (ℕ × ℕ), is_line p.fst p.snd) 
     (Finset.product grid_points grid_points)) / 2 = 96 :=
   sorry
   
end grid_lines_count_l161_161960


namespace geometric_increasing_condition_l161_161253

structure GeometricSequence (a₁ q : ℝ) (a : ℕ → ℝ) :=
  (rec_rel : ∀ n : ℕ, a (n + 1) = a n * q)

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_increasing_condition (a₁ q : ℝ) (a : ℕ → ℝ) (h : GeometricSequence a₁ q a) :
  ¬ (q > 1 ↔ is_increasing a) := sorry

end geometric_increasing_condition_l161_161253


namespace locus_of_orthocenter_l161_161845

open EuclideanGeometry

noncomputable def isosceles_triangle_with_median (A B C O M N H : Point) : Prop :=
is_isosceles_triangle A B C ∧ 
is_on_line O B C ∧ 
has_circle_centered_at_with_radius O A OA ∧
meets_line_at_two_points_on_circle_with_radius O A B M ∧
meets_line_at_two_points_on_circle_with_radius O A C N ∧
is_orthocenter_of_triangle H A M N

theorem locus_of_orthocenter (A B C O M N H : Point) :
  isosceles_triangle_with_median A B C O M N H →
  ∃ l : Line, is_parallel l (line B C) ∧ ∀ P : Point, P ∈ orthocenter_locus H A M N → P ∈ l :=
begin
  sorry
end

end locus_of_orthocenter_l161_161845


namespace value_of_f2008_plus_f2009_l161_161154

variable {f : ℤ → ℤ}

-- Conditions
axiom h1 : ∀ x : ℤ, f (-(x) + 2) = -f (x + 2)
axiom h2 : ∀ x : ℤ, f (6 - x) = f x
axiom h3 : f 3 = 2

-- The theorem to prove
theorem value_of_f2008_plus_f2009 : f 2008 + f 2009 = -2 :=
  sorry

end value_of_f2008_plus_f2009_l161_161154


namespace sum_of_c_d_l161_161617

theorem sum_of_c_d (c d : ℝ) (g : ℝ → ℝ) 
(hg : ∀ x, g x = (x + 5) / (x^2 + c * x + d)) 
(hasymp : ∀ x, (x = 2 ∨ x = -3) → x^2 + c * x + d = 0) : 
c + d = -5 := 
by 
  sorry

end sum_of_c_d_l161_161617


namespace garden_width_l161_161038

theorem garden_width :
  ∃ w l : ℝ, (2 * l + 2 * w = 60) ∧ (l * w = 200) ∧ (l = 2 * w) ∧ (w = 10) :=
by
  sorry

end garden_width_l161_161038


namespace rectangle_quadrilateral_inequality_l161_161078

theorem rectangle_quadrilateral_inequality 
    (a b c d : ℝ)
    (ha : 0 ≤ a ∧ a ≤ 3 / 2) 
    (hb : 0 ≤ b ∧ b ≤ 2) 
    (hc : 0 ≤ c ∧ c ≤ 3 / 2) 
    (hd : 0 ≤ d ∧ d ≤ 2) :
    25 ≤ (9 / 2 + 2 * a^2) + (9 / 2 + 2 * c^2) + (8 + 2 * b^2) + (8 + 2 * d^2) ∧ 
    (9 / 2 + 2 * a^2) + (9 / 2 + 2 * c^2) + (8 + 2 * b^2) + (8 + 2 * d^2) ≤ 50 :=
begin
  sorry
end

end rectangle_quadrilateral_inequality_l161_161078


namespace expected_value_of_8_sided_die_winning_l161_161053

theorem expected_value_of_8_sided_die_winning :
  let p : ℕ → ℚ := λ n, 1/8
  let winnings : ℕ → ℚ := λ n, (n^3 : ℚ)
  (∑ n in Finset.range 8, p (n + 1) * winnings (n + 1)) = 162 :=
by
  sorry

end expected_value_of_8_sided_die_winning_l161_161053


namespace complement_intersection_l161_161403

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the intersection of A and B
def A_inter_B : Set ℕ := {x ∈ A | x ∈ B}

-- Define the complement of A_inter_B in U
def complement_U_A_inter_B : Set ℕ := {x ∈ U | x ∉ A_inter_B}

-- Prove that the complement of the intersection of A and B in U is {1, 4, 5}
theorem complement_intersection :
  complement_U_A_inter_B = {1, 4, 5} :=
by
  sorry

end complement_intersection_l161_161403


namespace negation_of_cond6_l161_161942

section CookingSkills
variables {Person : Type} {P Q C D : Person → Prop}

axiom cond1 : ∀ x, C x → P x
axiom cond2 : ∃ x, C x ∧ P x
axiom cond3 : ∀ x, D x → ¬ P x
axiom cond4 : ∀ x, D x → Q x
axiom cond5 : ∃ x, D x ∧ Q x
axiom cond6 : ∀ x, D x → P x

theorem negation_of_cond6 : cond5 ↔ ¬ cond6 :=
by
  sorry
end CookingSkills

end negation_of_cond6_l161_161942


namespace equation_solution_l161_161393

theorem equation_solution (x : ℝ) (h₁ : x^3 + 2 * x + 1 > 0) :
    (16 * 5^(2 * x - 1) - 2 * 5^(x - 1) - 0.048) * log (x^3 + 2 * x + 1) = 0 →
    x = 0 :=
by
  sorry

end equation_solution_l161_161393


namespace modulo12_impossible_modulo14_possible_l161_161469

def has_distinct_product_remainders_mod (lst : List ℕ) (m : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → j < lst.length → (lst.nthLe i (by linarith) * lst.nthLe j (by linarith)) % m ≠
    (lst.nthLe i.succ (by linarith) * lst.nthLe j.succ (by linarith)) % m

theorem modulo12_impossible : ¬ ∃ lst, lst.length = 5 ∧ ∀ (i j : ℕ), i < j → j < lst.length → 
(lst = List.range' 1 11) → has_distinct_product_remainders_mod lst 12 :=
sorry

theorem modulo14_possible : ∃ lst : List ℕ, lst.length = 5 ∧ (lst = [6, 1, 3, 5, 13] → has_distinct_product_remainders_mod lst 14) :=
sorry

end modulo12_impossible_modulo14_possible_l161_161469


namespace inscribed_sphere_centroid_analogy_l161_161061

def inscribed_circle_touches_midpoints (T : Type) [equilateral_triangle T]
  (circ : circle) (touches_midpoints : ∀ (s : side T), circle_touches_segment_midpoint circ s) : Prop :=
true

def inscribed_sphere_touches_centroid (T : Type) [regular_tetrahedron T]
  (sphere : sphere) (touches_centroid : ∀ (f : face T), sphere_touches_face_centroid sphere f) : Prop :=
true

theorem inscribed_sphere_centroid_analogy
  (T_tri : Type) [equilateral_triangle T_tri]
  (circ : circle)
  (touches_mid_tri : ∀ (s : side T_tri), circle_touches_segment_midpoint circ s)
  (T_tetra : Type) [regular_tetrahedron T_tetra]
  (sphere : sphere)
  (touches_mid_tetra : inscribed_circle_touches_midpoints T_tri circ touches_mid_tri) :
  inscribed_sphere_touches_centroid T_tetra sphere (λ f, sphere_touches_face_centroid sphere f) :=
sorry

end inscribed_sphere_centroid_analogy_l161_161061


namespace sum_of_numbers_with_remainder_one_l161_161067

-- Define the sequence and perform the summation
theorem sum_of_numbers_with_remainder_one (n : ℕ) (h1 : n > 0) (h2 : n ≤ 100) (h3 : (n % 3 = 1)) :
  ∑ k in (Finset.filter (λ k, k % 3 = 1) (Finset.range 101)), k = 1717 :=
by
  sorry

end sum_of_numbers_with_remainder_one_l161_161067


namespace solve_exp_eq_l161_161105

theorem solve_exp_eq (x : ℝ) (h : Real.sqrt ((1 + Real.sqrt 2)^x) + Real.sqrt ((1 - Real.sqrt 2)^x) = 2) : 
  x = 0 := 
sorry

end solve_exp_eq_l161_161105


namespace minimum_area_of_right_triangle_l161_161593

def parallel_lines_and_right_triangle (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : Prop :=
  ∃ (l1 l2 : ℝ → ℝ) (A : ℝ × ℝ),
    (∀ x, l1 x = l1(0) ∧ l2 x = l2(0)) ∧
    (0 < A.2 ∧ A.2 < a + b) ∧
    (let S_ABC (φ : ℝ) : ℝ := (a * b) / sin (2 * φ) in
      ∃ φ, 0 < φ ∧ φ < π/2 ∧ S_ABC φ = a * b)

theorem minimum_area_of_right_triangle (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  parallel_lines_and_right_triangle a b h_a_pos h_b_pos :=
by 
  sorry

end minimum_area_of_right_triangle_l161_161593


namespace sum_valid_B_values_l161_161823

-- Add a predicate for a digit sum being divisible by 9
def divisible_by_9 (n : ℕ) :=
  n % 9 = 0

-- Define the given condition for the known digits sum
def digit_sum := 44

-- Define the function which verifies the possible values of B
def check_B (B : ℕ) :=
  digit_sum + B

-- Define the set of possible values B should satisfy
def valid_B_values : List ℕ :=
  List.filter (λ B => divisible_by_9 (check_B B)) (List.range 10)

-- Prove that the sum of valid B values is 8
theorem sum_valid_B_values : valid_B_values.sum = 8 := by
  -- Proof placeholder
  sorry

end sum_valid_B_values_l161_161823


namespace candies_last_days_l161_161118

-- Definitions for the given conditions
def neighbors_candy : ℕ := 66
def sister_candy : ℕ := 15
def friends_candy : ℕ := 20
def traded_candy : ℕ := 10
def given_away_candy : ℕ := 5
def daily_eaten_candy : ℕ := 9

-- Given these conditions, prove that the remaining candy will last Sarah 9 full days
theorem candies_last_days 
  (n_candies : neighbors_candy) 
  (s_candies : sister_candy) 
  (f_candies : friends_candy) 
  (t_candies : traded_candy) 
  (g_candies : given_away_candy) 
  (d_candies : daily_eaten_candy) 
  : (66 + 15 + 20 - (10 + 5)) / 9 = 9 := 
by 
  sorry

end candies_last_days_l161_161118


namespace problem1_problem2_problem3_l161_161589

-- Definitions for the sets A and B
def A := {x : ℤ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x : ℝ | x^2 - 3 * m * x + 2 * m^2 - m - 1 < 0}

-- Problem 1: The number of non-empty proper subsets of A is 254.
theorem problem1 : Nat.card (finset.powerset A) - 2 = 254 := sorry

-- Problem 2: The range of m if B = ∅ is m = -2.
theorem problem2 : ∀ m : ℝ, B m = ∅ ↔ m = -2 := sorry

-- Problem 3: The range of m if A ⊇ B.
theorem problem3 : ∀ m : ℝ, (∀ x, x ∈ B m → x ∈ A) ↔ m ∈ set.Icc (-1 : ℝ) 2 ∨ m = -2 := sorry

end problem1_problem2_problem3_l161_161589


namespace tangent_to_circumcircle_l161_161247

variables {P : Type} [MetricSpace P] [NormedAddTorsor ℝ P]

-- Define the circles Γ1 and Γ2 with centers O₁ and O₂ respectively
variables (Γ₁ Γ₂ : set P) (O₁ O₂ : P)
-- Assume that there is a point A on the line segment O₁O₂
variable (A : P)
-- Define the intersection points C and D of Γ₁ and Γ₂
variables (C D : P)
-- Assume the line AD intersects Γ₁ a second time at S
variable (S : P)
-- The line CS intersects O₁O₂ at F
variable (F : P)
-- Define Γ₃ as the circumcircle of triangle AD
variable (Γ₃ : set P)
-- E is the second intersection point of Γ₁ and Γ₃
variable (E : P)

-- Main theorem statement
theorem tangent_to_circumcircle
  (h1 : A ∈ segment ℝ O₁ O₂)
  (h2 : C ≠ D)
  (h3 : C ∈ Γ₁ ∩ Γ₂)
  (h4 : D ∈ Γ₁ ∩ Γ₂)
  (h5 : S ∈ Γ₁)
  (h6 : ∃ p : P, p = F ∧ cs_affine_segment ℝ C S O₁ O₂ p)
  (h7 : E ∈ Γ₁ ∩ Γ₃) 
  (h8 : O₁ ≠ O₂) :
  is_tangent (line_through O₁ E) Γ₃ O₁ :=
sorry

end tangent_to_circumcircle_l161_161247


namespace magnitude_a_minus_2b_l161_161958

open Real InnerProductSpace

variable {V : Type*} [InnerProductSpace ℝ V] (a b : V)

-- Given conditions
def magnitude_b_eq_2_times_magnitude_a (h1 : ‖b‖ = 2 * ‖a‖) : ‖a‖ = 1 :=
by sorry

def angle_between_a_and_b_eq_120 (h2 : real.angle.cos (inner_product a b) := -1/2) : 
(inner_product a b) = -1 :=
by sorry

-- Objective
theorem magnitude_a_minus_2b (h1 : ‖b‖ = 2 * ‖a‖) (h2 : real.angle.cos (inner_product a b) := -1/2) :
  ‖a - 2 • b‖ = sqrt 21 :=
by sorry

end magnitude_a_minus_2b_l161_161958


namespace logarithmic_function_property_l161_161051

theorem logarithmic_function_property (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y) → (f = λ x, log a x) :=
begin
  -- This is where the proof would normally go.
  sorry
end

end logarithmic_function_property_l161_161051


namespace chord_ratio_l161_161371

open Real

variable {x : Real}
variable {EQ GQ HQ FQ : Real}

theorem chord_ratio (hxpos : 0 < x)
  (hEQ : EQ = x + 1)
  (hGQ : GQ = 2x)
  (hHQ : HQ = 3x)
  (H : EQ * FQ = GQ * HQ) :
  (FQ / HQ) = (2 * x) / (x + 1) :=
sorry

end chord_ratio_l161_161371
