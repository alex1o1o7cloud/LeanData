import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.SpecialFunctions.Exp.Basic
import Mathlib.Analysis.SpecialFunctions.Hyperbolic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Probability.Normal
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Triangle.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.EuclideanSpace.Basic

namespace find_x_plus_inv_x_l168_168545

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + x⁻³ = 110) : x + x⁻¹ = 5 :=
sorry

end find_x_plus_inv_x_l168_168545


namespace gas_consumption_increase_l168_168906

theorem gas_consumption_increase 
    (regression_eq : ∀ (x : ℝ), x = 0.23 → (170 / 23) * x - (31 / 23) = 0.35) :
  ∃ y : ℝ, y = 0.35 :=
begin
  use 0.35,
  have h : (170 / 23) * 0.23 - (31 / 23) = 0.35,
  { exact regression_eq 0.23 (by norm_num) },
  exact h,
end

end gas_consumption_increase_l168_168906


namespace derivative_of_given_function_l168_168498

noncomputable def given_function (x : ℝ) : ℝ := sin (cos 2) ^ 3 - (cos (30 * x) ^ 2) / (60 * sin (60 * x))

theorem derivative_of_given_function (x : ℝ) :
  deriv given_function x = 1 / (4 * sin (30 * x) ^ 2) :=
by
  sorry

end derivative_of_given_function_l168_168498


namespace sum_of_squares_of_medians_l168_168364

noncomputable def mAD (a b c : ℝ) := real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2
noncomputable def mBE (a b c : ℝ) := real.sqrt (2 * a^2 + 2 * c^2 - b^2) / 2
noncomputable def mCF (a b c : ℝ) := real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2

theorem sum_of_squares_of_medians (a b c : ℝ) (h₁ : a = 13) (h₂ : b = 13) (h₃ : c = 10) :
  (mAD a b c)^2 + (mBE a b c)^2 + (mCF a b c)^2 = 244 :=
by sorry

end sum_of_squares_of_medians_l168_168364


namespace parabola_problem_l168_168418

noncomputable def p_value_satisfy_all_conditions (p : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A B : ℝ × ℝ),
    F = (p / 2, 0) ∧
    (A.2 = A.1 - p / 2 ∧ (A.2)^2 = 2 * p * A.1) ∧
    (B.2 = B.1 - p / 2 ∧ (B.2)^2 = 2 * p * B.1) ∧
    (A.1 + B.1) / 2 = 3 * p / 2 ∧
    (A.2 + B.2) / 2 = p ∧
    (p - 2 = -3 * p / 2)

theorem parabola_problem : ∃ (p : ℝ), p_value_satisfy_all_conditions p ∧ p = 4 / 5 :=
by
  sorry

end parabola_problem_l168_168418


namespace find_a_l168_168276

noncomputable def A := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem find_a (a : ℝ) : A ∩ B a = {3} → a = 1 :=
begin
  intro h,
  -- The proof is omitted
  sorry
end

end find_a_l168_168276


namespace equation_of_tangent_line_at_1_maximum_value_of_h_g_x_less_than_1_plus_e_inv_2_l168_168193

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / Real.exp x
noncomputable def h (x : ℝ) : ℝ := 1 - x - x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x * deriv f x

theorem equation_of_tangent_line_at_1 :
  ∀ (x : ℝ), f x = (Real.log x + 1) / Real.exp x →
  ∀ y, y = f 1 → tangent_line f 1 (1, f 1) = y :=
by
  intro x _
  intro y _
  sorry

theorem maximum_value_of_h :
  ∀ (x : ℝ), h x = 1 - x - x * Real.log x →
  ∀ (a : ℝ), a = exp (-2) → ∀ y, y = 1 + exp (-2) → h a = y :=
by
  intro x _ 
  intro a _
  intro y _
  sorry

theorem g_x_less_than_1_plus_e_inv_2 :
  ∀ (x : ℝ), x > 0 → g x = x * deriv f x →
  ∀ y, y = 1 + exp (-2) → g x < y :=
by
  intro x posx _
  intro y _
  sorry

end equation_of_tangent_line_at_1_maximum_value_of_h_g_x_less_than_1_plus_e_inv_2_l168_168193


namespace pack_objects_in_boxes_l168_168257

theorem pack_objects_in_boxes (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  ∃ (f : fin (nk) → fin (k)), ∀ (b : fin (k)), (∃ (c₁ c₂ : fin (k)), c₁ ≠ c₂ ∧ (∀ x, f x = b → x / n = c₁ ∨ x / n = c₂)) :=
sorry

end pack_objects_in_boxes_l168_168257


namespace expected_subset_iff_property_P_l168_168575

section

variable {n : ℕ} (S_n : Finset ℕ) (A : Finset ℕ)
variable {a b c : ℕ}

/-- Definition of Sn -/
def S_n_def (n : ℕ) : Finset ℕ := Finset.range (2 * n).succ

/-- Definition of "expected subset" -/
def is_expected_subset (A : Finset ℕ) : Prop :=
    ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
                a ∈ S_n ∧ b ∈ S_n ∧ c ∈ S_n ∧ 
                (a + b) ∈ A ∧ (b + c) ∈ A ∧ (c + a) ∈ A

/-- Definition of property P -/
def has_property_P (A : Finset ℕ) : Prop := 
    ∃ x y z : ℕ, x < y ∧ y < z ∧ x + y > z ∧ (x + y + z) % 2 = 0

/-- proving equivalence of definitions -/
theorem expected_subset_iff_property_P (S_n : Finset ℕ) (A : Finset ℕ) :
    (S_n = S_n_def n) ∧ (n ≥ 4) →
    (is_expected_subset S_n A ↔ has_property_P A) := 
sorry

end

end expected_subset_iff_property_P_l168_168575


namespace fraction_of_hour_l168_168035

theorem fraction_of_hour (flashes : ℕ) (time_per_flash : ℕ) (total_time_in_seconds : ℕ) : 
  flashes = 240 
  ∧ time_per_flash = 15 
  ∧ total_time_in_seconds = flashes * time_per_flash 
  → total_time_in_seconds = 3600 → 3600 / 3600 = 1 :=
by
  intro h
  cases h with h1 hmore
  cases hmore with h2 h3
  rw ← h3
  have h4 : total_time_in_seconds = 3600 := h2
  intro h_
  rw h4
  simp
  sorry

end fraction_of_hour_l168_168035


namespace point_distance_from_y_axis_l168_168313

noncomputable def distanceFromYAxis (P : ℝ × ℝ) : ℝ := P.1.abs

noncomputable def distanceFromXAxis (P : ℝ × ℝ) : ℝ := P.2.abs

theorem point_distance_from_y_axis (x : ℝ) :
  let P := (x, -4)
  (distanceFromXAxis P) = 4 →
  (distanceFromXAxis P) = (1/2) * (distanceFromYAxis P) →
  distanceFromYAxis P = 8 :=
by
  intro h1 h2
  sorry

end point_distance_from_y_axis_l168_168313


namespace range_quadratic_function_l168_168324

theorem range_quadratic_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = x^2 - 2 * x + 5 ↔ y ∈ Set.Ici 4 :=
by 
  sorry

end range_quadratic_function_l168_168324


namespace swimming_pool_length_l168_168425

theorem swimming_pool_length : 
  ∀ (L : ℝ), 
  (12 + 2 * 4 = 20) ∧ -- width of pool plus deck widths
  (L + 2 * 4 = L + 8) ∧ -- length of pool plus deck lengths
  ((L + 8) * 20 = 360) -- total area of the pool and deck
  → L = 10 :=
by
  intro L
  intro h
  cases h with hw hd
  cases hd with hl ha
  sorry

end swimming_pool_length_l168_168425


namespace maple_trees_planted_plant_maple_trees_today_l168_168739

-- Define the initial number of maple trees
def initial_maple_trees : ℕ := 2

-- Define the number of maple trees the park will have after planting
def final_maple_trees : ℕ := 11

-- Define the number of popular trees, though it is irrelevant for the proof
def initial_popular_trees : ℕ := 5

-- The main statement to prove: number of maple trees planted today
theorem maple_trees_planted : ℕ :=
  final_maple_trees - initial_maple_trees

-- Prove that the number of maple trees planted today is 9
theorem plant_maple_trees_today :
  maple_trees_planted = 9 :=
by
  sorry

end maple_trees_planted_plant_maple_trees_today_l168_168739


namespace three_digit_integer_condition_l168_168122

theorem three_digit_integer_condition (n a b c : ℕ) (hn : 100 ≤ n ∧ n < 1000)
  (hdigits : n = 100 * a + 10 * b + c)
  (hdadigits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (fact_condition : 2 * n / 3 = a.factorial * b.factorial * c.factorial) :
  n = 432 := sorry

end three_digit_integer_condition_l168_168122


namespace inequality_abc_l168_168166

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) (h3 : 0 ≤ b) (h4 : b ≤ 1) (h5 : 0 ≤ c) (h6 : c ≤ 1) :
  (a / (b * c + 1)) + (b / (a * c + 1)) + (c / (a * b + 1)) ≤ 2 := by
  sorry

end inequality_abc_l168_168166


namespace probability_xi_gt_2_l168_168521

open ProbabilityTheory MeasureTheory

noncomputable theory

def ξ : Measure ℝ := MeasureTheory.Measure.Normal.measure 0 (6^2)

theorem probability_xi_gt_2 :
  (MeasureTheory.Measure.Normal.cumulative_distribution ξ) (2) = 0.1 :=
sorry

end probability_xi_gt_2_l168_168521


namespace polynomial_not_factorable_l168_168039

theorem polynomial_not_factorable (p : ℕ) (p_digits : list ℕ) (hn_pos : p_digits.head > 1) :
  nat.prime p →
  let P : polynomial ℤ := polynomial.of_digits 10 p_digits in
  ¬ ∃ A B : polynomial ℤ, A.degree > 0 ∧ B.degree > 0 ∧ P = A * B :=
by sorry

end polynomial_not_factorable_l168_168039


namespace append_digits_divisible_by_all_less_than_10_l168_168801

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168801


namespace number_of_eight_digit_numbers_with_product_9261_l168_168972

-- Defining the conditions
def is_eight_digit_number (n : Nat) : Prop := 
  10000000 ≤ n ∧ n < 100000000

def product_of_digits (n : Nat) : Nat :=
  (n.digits.List.map (λ (d : Nat) => d)).prod

-- The theorem to be proved
theorem number_of_eight_digit_numbers_with_product_9261 :
  {k // is_eight_digit_number k ∧ product_of_digits k = 9261} = 1680 :=
sorry

end number_of_eight_digit_numbers_with_product_9261_l168_168972


namespace min_2a_plus_3b_l168_168177

theorem min_2a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_parallel : (a * (b - 3) - 2 * b = 0)) :
  (2 * a + 3 * b) = 25 :=
by
  -- proof goes here
  sorry

end min_2a_plus_3b_l168_168177


namespace count_odd_sum_subsets_l168_168471

/-- Given a set of integers, we want to count the number of subsets 
    with exactly three different numbers such that their sum is odd. -/
theorem count_odd_sum_subsets:
  let S := {45, 52, 87, 90, 112, 143, 154} in
  ∃ n : ℕ, n = 19 ∧ (∃ (A ⊆ S), (A.card = 3 ∧ (A.sum % 2 = 1) ∧ (count ((A.card = 3) ∧ (A.sum % 2 = 1)) = n))) :=
sorry

end count_odd_sum_subsets_l168_168471


namespace dot_product_eq_neg6_l168_168577

def a : ℝ × ℝ := (-4, 7)
def b : ℝ × ℝ := (5, 2)

theorem dot_product_eq_neg6 : (a.1 * b.1 + a.2 * b.2) = -6 :=
by
  -- Definitions used directly from conditions in the problem
  let a1 := a.1
  let a2 := a.2
  let b1 := b.1
  let b2 := b.2
  -- Apply the specific dot product calculation
  have h1 : a1 * b1 = -20 := by rfl
  have h2 : a2 * b2 = 14 := by rfl
  calc
    a1 * b1 + a2 * b2 = -20 + 14 : by rw [h1, h2]
    ... = -6 : by norm_num

end dot_product_eq_neg6_l168_168577


namespace chord_bisected_by_point_of_ellipse_l168_168182

theorem chord_bisected_by_point_of_ellipse 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1)
  (bisecting_point : ∃ x y : ℝ, x = 4 ∧ y = 2) :
  ∃ a b c : ℝ, a = 1 ∧ b = 2 ∧ c = -8 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 :=
by
   sorry

end chord_bisected_by_point_of_ellipse_l168_168182


namespace simplify_fraction_1_simplify_fraction_2_simplify_series_l168_168684

noncomputable theory

def rationalize_fraction_1 : Prop :=
  ∃ (a b : ℝ), (3 = a) ∧ (sqrt 6 + sqrt 3 = b) ∧ (a / b = sqrt 6 - sqrt 3)

def rationalize_fraction_2 : Prop :=
  ∃ (x y z : ℝ), (x = 3 / (sqrt 4 + sqrt 1)) ∧
                 (y = 3 / (sqrt 7 + sqrt 4)) ∧
                 (z = 3 / (sqrt 10 + sqrt 7)) ∧
                 (x + y + z = sqrt 10 - 1)

def rationalize_series (n : ℕ) : Prop :=
  ∑ k in Finset.range n, (3 / (sqrt (3 * k + 1) + sqrt (3 * k - 2))) = sqrt (3 * n + 1) - 1

-- Statements that require a proof:
theorem simplify_fraction_1 : rationalize_fraction_1 := sorry

theorem simplify_fraction_2 : rationalize_fraction_2 := sorry

theorem simplify_series (n : ℕ) : rationalize_series n := sorry

end simplify_fraction_1_simplify_fraction_2_simplify_series_l168_168684


namespace smallest_digits_to_append_l168_168824

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168824


namespace part_one_part_two_case_one_part_two_case_two_l168_168572

namespace LogarithmicProblem

section part_one
variables (a : ℝ) (f : ℝ → ℝ)
hypothesis ha : a > 0 ∧ a ≠ 1
hypothesis hfa : ∀ x, f(x) = log a x
hypothesis h : f(8) = 3

-- Prove that a = 2
theorem part_one : a = 2 :=
by sorry
end part_one

section part_two
variables (a x : ℝ)
hypothesis ha : 0 < a ∧ a ≠ 1

-- Prove the cases for the inequality log_a x <= log_a (2 - 3x)

-- Case 1: a > 1
theorem part_two_case_one (ha : a > 1) :
  (0 < x ∧ x ≤ 1/2) ↔ log a x ≤ log a (2 - 3x) :=
by sorry

-- Case 2: 0 < a < 1
theorem part_two_case_two (ha : 0 < a ∧ a < 1) :
  (1/2 ≤ x ∧ x < 2/3) ↔ log a x ≤ log a (2 - 3x) :=
by sorry
end part_two

end LogarithmicProblem

end part_one_part_two_case_one_part_two_case_two_l168_168572


namespace largest_of_three_roots_l168_168747

theorem largest_of_three_roots (p q r : ℝ) (hpqr_sum : p + q + r = 3) 
    (hpqr_prod_sum : p * q + p * r + q * r = -8) (hpqr_prod : p * q * r = -15) :
    max p (max q r) = 3 := 
sorry

end largest_of_three_roots_l168_168747


namespace corn_harvest_l168_168454

theorem corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) (total_bushels : ℕ) :
  rows = 5 →
  stalks_per_row = 80 →
  stalks_per_bushel = 8 →
  total_bushels = (rows * stalks_per_row) / stalks_per_bushel →
  total_bushels = 50 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, mul_comm 5 80] at h4
  norm_num at h4
  exact h4

end corn_harvest_l168_168454


namespace round_robin_teams_l168_168750

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 :=
sorry

end round_robin_teams_l168_168750


namespace daily_rental_cost_l168_168403

theorem daily_rental_cost
  (x : ℝ)
  (daily_budget : ℝ)
  (max_miles_cost : ℝ)
  (h1 : daily_budget = 88)
  (h2 : max_miles_cost = 190 * 0.2) :
  x + max_miles_cost ≤ daily_budget → x ≤ 50 :=
by
  intros h
  rw [h1, h2] at h
  simp at h
  exact h

end daily_rental_cost_l168_168403


namespace fewer_motorcycles_sold_l168_168917

variable (P N : ℝ)

-- Conditions
-- Original revenue
def original_revenue : ℝ := P * N
-- New revenue
def new_revenue : ℝ := (P + 1000) * 63
-- Increase in revenue
def revenue_increase : ℝ := 26000
-- Revenue at new price
def revenue_at_new_price : ℝ := 594000
-- Revenue at original price
def revenue_at_original_price : ℝ := revenue_at_new_price - revenue_increase
-- Number of motorcycles sold initially
def original_motorcycles_sold : ℝ := revenue_at_original_price / P

-- The proof problem: We need to prove the difference in the number of motorcycles sold is 4
theorem fewer_motorcycles_sold :
  original_motorcycles_sold - 63 = 4 :=
by
  sorry

end fewer_motorcycles_sold_l168_168917


namespace comparison_l168_168988

noncomputable def a : ℝ := Real.logb 0.6 0.5
noncomputable def b : ℝ := Real.log 0.5
noncomputable def c : ℝ := 0.6 ^ 0.5

theorem comparison (ha : a = Real.logb 0.6 0.5) (hb : b = Real.log 0.5) (hc : c = 0.6 ^ 0.5) : 
  b < c ∧ c < a := 
by
  sorry

end comparison_l168_168988


namespace Raine_steps_to_school_l168_168683

-- Define Raine's conditions
variable (steps_total : ℕ) (days : ℕ) (round_trip_steps : ℕ)

-- Given conditions
def Raine_conditions := steps_total = 1500 ∧ days = 5 ∧ round_trip_steps = steps_total / days

-- Prove that the steps to school is 150 given Raine's conditions
theorem Raine_steps_to_school (h : Raine_conditions 1500 5 300) : (300 / 2) = 150 :=
by
  sorry

end Raine_steps_to_school_l168_168683


namespace migration_rate_ratio_l168_168431

variable (G M : ℕ)
variable (Gm : ℕ := 0.5 * G)
variable (Mm : ℕ := 0.2 * M)
variable (Gf : ℕ := G - Gm)
variable (Mf : ℕ := M - Mm)

theorem migration_rate_ratio (h1 : Gm = 0.5 * G) (h2 : Mm = 0.2 * M) :
  (Mm / Gm : ℚ) / (Mf / Gf) = 1 / 4 := by
  sorry

end migration_rate_ratio_l168_168431


namespace range_S6_l168_168665

theorem range_S6 (a1 d : ℕ) (S6 : ℤ)
  (h1 : 1 ≤ a1 + 4 * d)
  (h2 : a1 + 4 * d ≤ 4)
  (h3 : 2 ≤ a1 + 5 * d)
  (h4 : a1 + 5 * d ≤ 3) :
  -12 ≤ 6 * a1 + 15 * d ∧ 6 * a1 + 15 * d ≤ 42 := 
by
  sorry

end range_S6_l168_168665


namespace big_boxes_count_l168_168738

theorem big_boxes_count
  (soaps_per_package : ℕ)
  (packages_per_box : ℕ)
  (total_soaps : ℕ)
  (soaps_per_box : ℕ)
  (H1 : soaps_per_package = 192)
  (H2 : packages_per_box = 6)
  (H3 : total_soaps = 2304)
  (H4 : soaps_per_box = soaps_per_package * packages_per_box) :
  total_soaps / soaps_per_box = 2 :=
by
  sorry

end big_boxes_count_l168_168738


namespace triangle_problem_l168_168630

noncomputable theory

def triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  ∃ x y : ℝ, x = dist A B ∧ y = dist A C

theorem triangle_problem (A B C : ℝ) (hA : angle A = 90) (hBC : dist B C = 10) (h_tanC_cosB : tan (angle C) = 3 * cos (angle B)) :
  dist A B = 20 * Real.sqrt 2 / 3 :=
sorry

end triangle_problem_l168_168630


namespace smallest_number_of_digits_to_append_l168_168808

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168808


namespace original_price_increased_by_total_percent_l168_168248

noncomputable def percent_increase_sequence (P : ℝ) : ℝ :=
  let step1 := P * 1.15
  let step2 := step1 * 1.40
  let step3 := step2 * 1.20
  let step4 := step3 * 0.90
  let step5 := step4 * 1.25
  (step5 - P) / P * 100

theorem original_price_increased_by_total_percent (P : ℝ) : percent_increase_sequence P = 117.35 :=
by
  -- Sorry is used here for simplicity, but the automated proof will involve calculating the exact percentage increase step-by-step.
  sorry

end original_price_increased_by_total_percent_l168_168248


namespace decimal_89_to_binary_l168_168089

def decimal_to_binary (n : ℕ) : ℕ := sorry

theorem decimal_89_to_binary :
  decimal_to_binary 89 = 1011001 :=
sorry

end decimal_89_to_binary_l168_168089


namespace number_of_cakes_sold_l168_168941

-- Definitions based on the conditions provided
def cakes_made : ℕ := 173
def cakes_bought : ℕ := 103
def cakes_left : ℕ := 190

-- Calculate the initial total number of cakes
def initial_cakes : ℕ := cakes_made + cakes_bought

-- Calculate the number of cakes sold
def cakes_sold : ℕ := initial_cakes - cakes_left

-- The proof statement
theorem number_of_cakes_sold : cakes_sold = 86 :=
by
  unfold cakes_sold initial_cakes cakes_left cakes_bought cakes_made
  rfl

end number_of_cakes_sold_l168_168941


namespace solve_an_prove_Tn_gt_Sn_l168_168541

variables (a : ℕ → ℤ) (b : ℕ → ℤ) (S T : ℕ → ℤ)
variables (S4_eq : S 4 = 32) (T3_eq : T 3 = 16)
variables (a_arithmetic : ∃ d, ∀ n, a (n + 1) = a n + d)

def an_formula_correct : Prop :=
  ∃ c d, ∀ n, a n = d * n + c

def bn_definition : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def Sn_definition : Prop :=
  ∀ n, S n = ∑ i in Finset.range n, a (i + 1)

def Tn_definition : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b (i + 1)

theorem solve_an (hyp_seq : an_formula_correct a) : ∃ c, ∃ d, d = 2 ∧ c = 3 :=
  by sorry

theorem prove_Tn_gt_Sn (hyp_bn : bn_definition a b)
  (hyp_sn : Sn_definition a S)
  (hyp_tn : Tn_definition a b T) :
  ∀ n, n > 5 → T n > S n :=
  by sorry

end solve_an_prove_Tn_gt_Sn_l168_168541


namespace average_stamps_collected_l168_168687

theorem average_stamps_collected (a : ℕ → ℕ) 
  (h₁ : a 1 = 10) 
  (h₂ : ∀ n, a (n + 1) = a n + 10) 
  (h₃ : ∑ i in (Finset.range 7), a (i + 1) = 280) :
  (∑ i in (Finset.range 7), a (i + 1)) / 7 = 40 := 
by sorry

end average_stamps_collected_l168_168687


namespace arithmetic_sequence_sum_l168_168558

theorem arithmetic_sequence_sum :
  (∃ {a : ℕ → ℚ}, (S (3) = 0 ∧ S (5) = -5) ∧ 
  -- Define the sequence function
  (∀ n : ℕ, a n = 2 - n)
  ) →
  -- Define the term for the new sequence {1 / (a_(2n-1) * a_(2n+1))}
  (let new_seq := λ n : ℕ, 1 / (a (2 * n - 1) * a (2 * n + 1)) in
  -- Sum the first 8 terms of the new sequence
  (∑ k in finset.range 8, new_seq k) = -8 / 15) :=
sorry

end arithmetic_sequence_sum_l168_168558


namespace smallest_digits_to_append_l168_168847

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168847


namespace smallest_number_append_l168_168816

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168816


namespace maximum_distance_l168_168559

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3) ^ 2 + (y - 4) ^ 2 = 1

-- Define points A and B
def point_A := (0, -1)
def point_B := (0, 1)

-- Define the function for the sum of the squares of distances from P to A and B
def sum_squares_distances (P : ℝ × ℝ) : ℝ := 
  let dA := (fst P - fst point_A) ^ 2 + (snd P - snd point_A) ^ 2
  let dB := (fst P - fst point_B) ^ 2 + (snd P - snd point_B) ^ 2
  dA + dB

-- The maximum value and corresponding point should be specified
def P : ℝ × ℝ := (18/5, 24/5)

theorem maximum_distance :
  circle_C (fst P) (snd P) → sum_squares_distances P = 74 :=
by sorry

end maximum_distance_l168_168559


namespace count_sets_l168_168517

noncomputable def sumOfSubsetsSatisfying (n : ℕ) (x : Fin n → ℝ) (lambda : ℝ) : Prop :=
  n ≥ 2 ∧
  (∑ i, x i = 0) ∧
  (∑ i, (x i)^2 = 1) →
  let subsetsSatisfying := { A : Finset (Fin n) | ∑ i in A, x i ≥ lambda } in
  subsetsSatisfying.card ≤ (2 : ℝ)^(n - 3).toNat / lambda^2

theorem count_sets (n : ℕ) (x : Fin n → ℝ) (lambda : ℝ) :
  sumOfSubsetsSatisfying n x lambda :=
sorry

end count_sets_l168_168517


namespace smallest_digits_to_append_l168_168860

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168860


namespace seventh_grade_a_seventh_grade_b_eighth_grade_c_l168_168714

-- Define the conditions for the problem
def seventh_grade_scores : List ℕ := [99, 95, 95, 91, 100, 86, 77, 93, 85, 79]
def eighth_grade_scores : List ℕ := [99, 91, 97, 63, 96, 97, 100, 94, 87, 76]

-- Calculate the necessary values based on the conditions
def num_students_70_to_80 (scores : List ℕ) : ℕ := List.count (λ x => 70 < x ∧ x ≤ 80) scores

def median (scores : List ℕ) : ℕ :=
  let sorted_scores := scores.qsort (· < ·)
  let n := sorted_scores.length
  if n % 2 = 0 then (sorted_scores.get! (n / 2 - 1) + sorted_scores.get! (n / 2)) / 2 
  else sorted_scores.get! (n / 2)

def mode (scores : List ℕ) : ℕ :=
  scores.foldl (λ m x => if scores.count x > scores.count m then x else m) 0

-- Lean proof problem statements
theorem seventh_grade_a : num_students_70_to_80 seventh_grade_scores = 2 := by sorry
theorem seventh_grade_b : median seventh_grade_scores = 92 := by sorry
theorem eighth_grade_c : mode eighth_grade_scores = 97 := by sorry

end seventh_grade_a_seventh_grade_b_eighth_grade_c_l168_168714


namespace smallest_integer_solution_l168_168359

theorem smallest_integer_solution (x : ℝ) :
  x^4 - 40 * x^2 + 324 = 0 → x = -4 :=
begin
  sorry
end

end smallest_integer_solution_l168_168359


namespace Mary_has_more_apples_than_peaches_l168_168639

variable (JakePeaches JakeApples MaryApples MaryPeaches StevenApples StevenPeaches : ℕ)

theorem Mary_has_more_apples_than_peaches :
  StevenApples = 11 →
  StevenPeaches = 18 →
  JakePeaches = StevenPeaches - 8 →
  JakeApples = StevenApples + 10 →
  MaryApples = 2 * JakeApples →
  MaryPeaches = StevenPeaches / 2 →
  MaryPeaches - MaryApples = -33 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end Mary_has_more_apples_than_peaches_l168_168639


namespace weight_of_replaced_person_l168_168609

theorem weight_of_replaced_person (W : ℝ) (average_increase : ℝ) (new_person_weight : ℝ) 
(h1 : average_increase = 4) (h2 : new_person_weight = 87) :
  55 = W / 8 + 4 * 8 - 87 - W / 8 :=
begin
  sorry
end

end weight_of_replaced_person_l168_168609


namespace a_has_inverse_c_has_inverse_d_has_inverse_f_has_inverse_g_has_inverse_h_has_inverse_b_has_no_inverse_e_has_no_inverse_l168_168477

variable (x y : ℝ)

-- Function Definitions
def a (x : ℝ) : ℝ := Real.sqrt (3 - x)            -- Domain: (-∞, 3]
def b (x : ℝ) : ℝ := x^3 - 2 * x                  -- Domain: ℝ
def c (x : ℝ) : ℝ := x + 2 / x                    -- Domain: (0, ∞)
def d (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 8          -- Domain: [0, ∞)
def e (x : ℝ) : ℝ := abs (x - 3) + abs (x + 4)    -- Domain: ℝ
def f (x : ℝ) : ℝ := 4^x + 8^x                    -- Domain: ℝ
def g (x : ℝ) : ℝ := x - 2 / x                    -- Domain: (0, ∞)
def h (x : ℝ) : ℝ := x / 3                        -- Domain: [-3, 6)

-- Predicate to check if a function has an inverse
def has_inverse {α β : Type} (f : α → β) : Prop :=
  ∃ g : β → α, ∀ x : α, g (f x) = x

-- Lean statements to verify inverses
theorem a_has_inverse : has_inverse (a : ℝ → ℝ) := by sorry
theorem c_has_inverse : has_inverse (c : ℝ → ℝ) := by sorry
theorem d_has_inverse : has_inverse (d : ℝ → ℝ) := by sorry
theorem f_has_inverse : has_inverse (f : ℝ → ℝ) := by sorry
theorem g_has_inverse : has_inverse (g : ℝ → ℝ) := by sorry
theorem h_has_inverse : has_inverse (h : ℝ → ℝ) := by sorry

-- Proving that b and e do not have an inverse
lemma b_not_one_to_one : ¬(b x = b y → x = y) := by sorry
lemma e_not_one_to_one : ¬(e x = e y → x = y) := by sorry

theorem b_has_no_inverse : ¬has_inverse (b : ℝ → ℝ) := by
  intro h
  have : b x = b y → x = y := λ hxy => congr_fun (classical.some_spec h) _
  exact b_not_one_to_one this

theorem e_has_no_inverse : ¬has_inverse (e : ℝ → ℝ) := by
  intro h
  have : e x = e y → x = y := λ hxy => congr_fun (classical.some_spec h) _
  exact e_not_one_to_one this

end a_has_inverse_c_has_inverse_d_has_inverse_f_has_inverse_g_has_inverse_h_has_inverse_b_has_no_inverse_e_has_no_inverse_l168_168477


namespace cos_double_angle_l168_168176

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi - θ) = 1 / 3) : 
  Real.cos (2 * θ) = 7 / 9 :=
by 
  sorry

end cos_double_angle_l168_168176


namespace smallest_number_of_digits_to_append_l168_168807

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168807


namespace solve_absolute_value_equation_l168_168328

theorem solve_absolute_value_equation (x : ℝ) : x^2 - 3 * |x| - 4 = 0 ↔ x = 4 ∨ x = -4 :=
by
  sorry

end solve_absolute_value_equation_l168_168328


namespace soda_cost_l168_168292

theorem soda_cost (S P W : ℝ) (h1 : P = 3 * S) (h2 : W = 3 * P) (h3 : 3 * S + 2 * P + W = 18) : S = 1 :=
by
  sorry

end soda_cost_l168_168292


namespace john_total_money_l168_168715

-- Variables representing the prices and quantities.
def chip_price : ℝ := 2
def corn_chip_price : ℝ := 1.5
def chips_quantity : ℕ := 15
def corn_chips_quantity : ℕ := 10

-- Hypothesis representing the total money John has.
theorem john_total_money : 
    (chips_quantity * chip_price + corn_chips_quantity * corn_chip_price) = 45 := by
  sorry

end john_total_money_l168_168715


namespace number_of_women_in_luxury_compartment_l168_168293

theorem number_of_women_in_luxury_compartment (total_passengers : ℕ) (percent_women percent_in_luxury : ℝ) 
(h_total : total_passengers = 300) 
(h_percent_women : percent_women = 0.70)
(h_percent_in_luxury : percent_in_luxury = 0.15) : 
∃ women_in_luxury : ℕ, women_in_luxury = 32 :=
by 
  let total_women := total_passengers * percent_women 
  let women_in_luxury := total_women * percent_in_luxury
  have h1 : total_women = 210, from calc
    total_women = 300 * 0.70 : by rw [h_total, h_percent_women]
    ... = 210 : by norm_num,
  have h2 : women_in_luxury = 31.5, from calc
    women_in_luxury = 210 * 0.15 : by rw h1
    ... = 31.5 : by norm_num,
  use 32,
  sorry

end number_of_women_in_luxury_compartment_l168_168293


namespace find_range_of_m_l168_168163

variable (m : ℝ)

-- Definition of p: There exists x in ℝ such that mx^2 - mx + 1 < 0
def p : Prop := ∃ x : ℝ, m * x ^ 2 - m * x + 1 < 0

-- Definition of q: The curve of the equation (x^2)/(m-1) + (y^2)/(3-m) = 1 is a hyperbola
def q : Prop := (m - 1) * (3 - m) < 0

-- Given conditions
def proposition_and : Prop := ¬ (p m ∧ q m)
def proposition_or : Prop := p m ∨ q m

-- Final theorem statement
theorem find_range_of_m : proposition_and m ∧ proposition_or m → (0 < m ∧ m ≤ 1) ∨ (3 ≤ m ∧ m < 4) :=
sorry

end find_range_of_m_l168_168163


namespace product_of_nonreal_roots_eq_l168_168131

theorem product_of_nonreal_roots_eq :
  let p : Polynomial ℂ := Polynomial.C (-984) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 + 15 * Polynomial.X ^ 2 - 20 * Polynomial.X
  (r1 r2 : ℂ) (root_of_nonreal_roots : (Polynomial.roots p).erase 1 = [r1, r2]) (nonreal_r1 : r1.im ≠ 0) (nonreal_r2 : r2.im ≠ 0) :
  r1 * r2 = 4 - Real.sqrt 1000 :=
sorry

end product_of_nonreal_roots_eq_l168_168131


namespace base_eight_to_base_ten_l168_168353

theorem base_eight_to_base_ten {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 5) (h3 : d3 = 7) :
  d3 * 8^0 + d2 * 8^1 + d1 * 8^2 = 111 := 
by
  sorry

end base_eight_to_base_ten_l168_168353


namespace minimize_daily_average_cost_l168_168932

noncomputable def daily_average_cost (n : ℕ) : ℝ :=
  if h : n > 0 then
    (∑ i in finset.range n, (i + 1 + 49) / 10) / n + 32000 / n
  else 0

theorem minimize_daily_average_cost :
  ∃ n : ℕ, daily_average_cost n = 800 := sorry

end minimize_daily_average_cost_l168_168932


namespace Two_Sessions_Competition_l168_168712

open List

-- Definitions and conditions
def scores_7th : List ℕ := [99, 95, 95, 91, 100, 86, 77, 93, 85, 79]
def scores_8th : List ℕ := [99, 91, 97, 63, 96, 97, 100, 94, 87, 76]

-- Lean 4 statement
theorem Two_Sessions_Competition :
  let a := (scores_7th.filter (fun x => 70 < x ∧ x ≤ 80)).length in
  let b := (let sorted := scores_7th.qsort (≤) in (sorted.nth 4).getD 0 + (sorted.nth 5).getD 0) / 2 in
  let c := scores_8th.mode.getD 0 in
  a = 2 ∧ b = 92 ∧ c = 97 := by
  sorry

end Two_Sessions_Competition_l168_168712


namespace sum_of_valid_starting_integers_l168_168466

-- Define the machine operation
def machine (N : ℕ) : ℕ :=
  if N % 2 = 1 then 4 * N - 2 else N / 3

-- Define a function that recursively applies the machine operation
def machine_steps (N : ℕ) (steps : ℕ) : ℕ :=
  Nat.recOn steps N (λ _ res => machine res)

-- Main theorem to find all valid starting integers M that result in 5 after six steps
theorem sum_of_valid_starting_integers : 
  (∑ M in (finset.filter (λ n => machine_steps n 6 = 5) (finset.range 1000)), M) = 127 :=
  sorry

end sum_of_valid_starting_integers_l168_168466


namespace smallest_digits_to_append_l168_168831

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168831


namespace ratio_as_percent_l168_168325

-- Condition: The ratio given is 15:25
def ratio : ℝ := 15 / 25

-- The goal is to prove that the ratio expressed as a percent is equal to 60
theorem ratio_as_percent (h : ratio = 15 / 25) : (ratio * 100) = 60 :=
by
  sorry

end ratio_as_percent_l168_168325


namespace propositions_alpha_and_beta_true_l168_168286

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = -f (-x)

def strictly_increasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x < f y

def strictly_decreasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x > f y

def alpha (f : ℝ → ℝ) : Prop :=
∀ x, ∃ g h : ℝ → ℝ, even_function g ∧ odd_function h ∧ f x = g x + h x

def beta (f : ℝ → ℝ) : Prop :=
∀ x, strictly_increasing_function f → ∃ p q : ℝ → ℝ, 
  strictly_increasing_function p ∧ strictly_decreasing_function q ∧ f x = p x + q x

theorem propositions_alpha_and_beta_true (f : ℝ → ℝ) :
  alpha f ∧ beta f :=
by
  sorry

end propositions_alpha_and_beta_true_l168_168286


namespace Alyssa_spent_on_marbles_l168_168439

def total_spent_on_toys : ℝ := 12.30
def cost_of_football : ℝ := 5.71
def amount_spent_on_marbles : ℝ := 12.30 - 5.71

theorem Alyssa_spent_on_marbles :
  total_spent_on_toys - cost_of_football = amount_spent_on_marbles :=
by
  sorry

end Alyssa_spent_on_marbles_l168_168439


namespace train_passes_man_in_4_seconds_l168_168884

-- Define the speeds in km/h
def speed_train_kmh : ℝ := 90
def speed_man_kmh : ℝ := 9

-- Convert speeds to m/s
def kmh_to_ms (speed : ℝ) : ℝ := speed * (1000 / 3600)

def speed_train_ms : ℝ := kmh_to_ms speed_train_kmh
def speed_man_ms : ℝ := kmh_to_ms speed_man_kmh

-- Calculate the relative speed in m/s
def relative_speed_ms : ℝ := speed_train_ms + speed_man_ms

-- Define the length of the train in meters
def length_train : ℝ := 110

-- Calculate the time in seconds for the train to pass the man
def time_to_pass : ℝ := length_train / relative_speed_ms

-- The proof goal
theorem train_passes_man_in_4_seconds (h1 : speed_train_ms = 25) (h2 : speed_man_ms = 2.5) : time_to_pass = 4 :=
by 
  -- Proof steps go here
  sorry

end train_passes_man_in_4_seconds_l168_168884


namespace value_of_m_div_x_l168_168889

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5)

def x := a + 0.25 * a
def m := b - 0.40 * b

theorem value_of_m_div_x (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5) :
    m / x = 3 / 5 :=
by
  sorry

end value_of_m_div_x_l168_168889


namespace net_percentage_change_l168_168724

-- Definitions based on given conditions
variables (P : ℝ) (P_post_decrease : ℝ) (P_post_increase : ℝ)

-- Conditions
def decreased_by_5_percent : Prop := P_post_decrease = P * (1 - 0.05)
def increased_by_10_percent : Prop := P_post_increase = P_post_decrease * (1 + 0.10)

-- Proof problem
theorem net_percentage_change (h1 : decreased_by_5_percent P P_post_decrease) (h2 : increased_by_10_percent P_post_decrease P_post_increase) : 
  ((P_post_increase - P) / P) * 100 = 4.5 :=
by
  -- The proof would go here
  sorry

end net_percentage_change_l168_168724


namespace number_of_seven_digit_palindromes_l168_168348

theorem number_of_seven_digit_palindromes : 
  let choices := {5, 6, 7} in
  let num_choices := (choices.card : ℕ) in
  (num_choices * num_choices * num_choices * num_choices) = 81 :=
by
  let choices := {5, 6, 7}
  let num_choices := (choices.card : ℕ)
  have h : (num_choices = 3) := rfl
  rw [h]
  norm_num

end number_of_seven_digit_palindromes_l168_168348


namespace odd_even_subsets_count_equal_odd_even_subset_capacities_sum_equal_odd_subset_capacities_sum_l168_168392

-- Definitions from the problem
def Sn (n : Nat) : Set Nat := { k : Nat | 1 ≤ k ∧ k ≤ n }
def capacity (X : Set Nat) : Nat := X.toList.sum

def is_odd (n : Nat) : Prop := n % 2 = 1
def is_even (n : Nat) : Prop := n % 2 = 0

def is_odd_subset (X : Set Nat) : Prop := is_odd (capacity X)
def is_even_subset (X : Set Nat) : Prop := is_even (capacity X)

-- Problem 1: 
theorem odd_even_subsets_count_equal (n : Nat) : 
  (Finset.filter is_odd_subset (Finset.powerset (Finset.range (n + 1)))).card =
  (Finset.filter is_even_subset (Finset.powerset (Finset.range (n + 1)))).card :=
sorry

-- Problem 2:
theorem odd_even_subset_capacities_sum_equal (n : Nat) (h : 3 ≤ n) : 
  (Finset.filter is_odd_subset (Finset.powerset (Finset.range (n + 1)))).toList.sum capacity = 
  (Finset.filter is_even_subset (Finset.powerset (Finset.range (n + 1)))).toList.sum capacity :=
sorry

-- Problem 3:
theorem odd_subset_capacities_sum (n : Nat) (h : 3 ≤ n) : 
  (Finset.filter is_odd_subset (Finset.powerset (Finset.range (n + 1)))).toList.sum capacity = 
  2 ^ (n - 3) * n * (n + 1) :=
sorry

end odd_even_subsets_count_equal_odd_even_subset_capacities_sum_equal_odd_subset_capacities_sum_l168_168392


namespace smallest_digits_to_append_l168_168869

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168869


namespace min_value_expression_for_real_numbers_l168_168157

def min_expression_value (x : ℝ) (hx : x > 4) : Prop :=
  ∀ (y : ℝ), y = (x - 4) → y > 0 → (x + 18) / (y) = 2 * Real.sqrt 22

theorem min_value_expression_for_real_numbers (x : ℝ) (hx : x > 4) :
  ∃ (y : ℝ), y > 0 ∧ y = Real.sqrt (x - 4) ∧ ((x + 18) / y) = 2 * Real.sqrt 22 :=
begin
  sorry
end

end min_value_expression_for_real_numbers_l168_168157


namespace Two_Sessions_Competition_l168_168711

open List

-- Definitions and conditions
def scores_7th : List ℕ := [99, 95, 95, 91, 100, 86, 77, 93, 85, 79]
def scores_8th : List ℕ := [99, 91, 97, 63, 96, 97, 100, 94, 87, 76]

-- Lean 4 statement
theorem Two_Sessions_Competition :
  let a := (scores_7th.filter (fun x => 70 < x ∧ x ≤ 80)).length in
  let b := (let sorted := scores_7th.qsort (≤) in (sorted.nth 4).getD 0 + (sorted.nth 5).getD 0) / 2 in
  let c := scores_8th.mode.getD 0 in
  a = 2 ∧ b = 92 ∧ c = 97 := by
  sorry

end Two_Sessions_Competition_l168_168711


namespace equilateral_triangle_CE_length_l168_168023

theorem equilateral_triangle_CE_length
  (a : ℝ)
  (A B C D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  [EquilateralTriangle A B C]
  (h : side_length A B = a)
  (h₁ : lies_on D (line_segment B C))
  (h₂ : lies_on E (line_segment A B))
  (h₃ : distance B D = (1/3) * a)
  (h₄ : distance A E = distance D E) :
  distance C E = (sqrt 37 / 3) * a :=
by
  sorry

end equilateral_triangle_CE_length_l168_168023


namespace jemma_grasshoppers_l168_168252

-- Definitions corresponding to the conditions
def grasshoppers_on_plant : ℕ := 7
def baby_grasshoppers : ℕ := 2 * 12

-- Theorem statement equivalent to the problem
theorem jemma_grasshoppers : grasshoppers_on_plant + baby_grasshoppers = 31 :=
by
  sorry

end jemma_grasshoppers_l168_168252


namespace no_consecutive_identical_digits_total_l168_168583

/-- 
  Define the set of numbers from 0 to 999999 and count how many of these numbers
  do not have two consecutive digits that are the same.
 -/
def count_no_consecutive_identical_digits : ℕ :=
  (∑ n in finset.range 1.succ 7, 9^n) + 1

/-- 
  Statement to prove that the number of integers from 0 to 999999 that do not
  have two identical consecutive digits in their decimal representation is 597880.  
 -/
theorem no_consecutive_identical_digits_total :
  count_no_consecutive_identical_digits = 597880 :=
by sorry

end no_consecutive_identical_digits_total_l168_168583


namespace sqrt_equation_solution_l168_168589

theorem sqrt_equation_solution (x : ℝ) (h : sqrt (3 + sqrt (2 * x - 1)) = 4) : x = 85 := by
  sorry

end sqrt_equation_solution_l168_168589


namespace combined_ticket_cost_l168_168686

variables (S K : ℕ)

theorem combined_ticket_cost (total_budget : ℕ) (samuel_food_drink : ℕ) (kevin_food : ℕ) (kevin_drink : ℕ) :
  total_budget = 20 →
  samuel_food_drink = 6 →
  kevin_food = 4 →
  kevin_drink = 2 →
  S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget →
  S + K = 8 :=
by
  intros h_total_budget h_samuel_food_drink h_kevin_food h_kevin_drink h_total_spent
  /-
  We have the following conditions:
  1. total_budget = 20
  2. samuel_food_drink = 6
  3. kevin_food = 4
  4. kevin_drink = 2
  5. S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget

  We need to prove that S + K = 8. We can use the conditions to derive this.
  -/
  rw [h_total_budget, h_samuel_food_drink, h_kevin_food, h_kevin_drink] at h_total_spent
  exact sorry

end combined_ticket_cost_l168_168686


namespace jacket_final_price_l168_168044

-- Definitions
def initial_price : ℝ := 20
def first_discount_rate : ℝ := 0.4
def second_discount_rate : ℝ := 0.25

-- Proof problem statement
theorem jacket_final_price :
  let first_discount := initial_price * first_discount_rate,
      first_new_price := initial_price - first_discount,
      second_discount := first_new_price * second_discount_rate,
      final_price := first_new_price - second_discount
  in final_price = 9 := by
  sorry

end jacket_final_price_l168_168044


namespace find_number_of_persons_l168_168063

-- Definitions of the given conditions
def total_amount : ℕ := 42900
def amount_per_person : ℕ := 1950

-- The statement to prove
theorem find_number_of_persons (n : ℕ) (h : total_amount = n * amount_per_person) : n = 22 :=
sorry

end find_number_of_persons_l168_168063


namespace max_value_of_function_attain_max_value_of_function_l168_168128

theorem max_value_of_function (x : ℝ) (hx : x < 0) : x + 4 / x ≤ -4 :=
sorry

theorem attain_max_value_of_function : ∃ x < 0, x + 4 / x = -4 :=
begin
  use -2,
  split,
  { linarith, },
  { norm_num, },
end

end max_value_of_function_attain_max_value_of_function_l168_168128


namespace min_distance_A_B_l168_168658

noncomputable def minimum_distance : ℝ :=
  1 + (1/4) * Real.log 3

theorem min_distance_A_B :
  ∀ (A B : ℝ × ℝ), 
  (∃ x y : ℝ, A = (x, y) ∧ (√3) * x - y + 1 = 0) ∧
  (∃ u : ℝ, B = (u, Real.log u)) →
  ∃ P : ℝ × ℝ, 
  ∀ (x y : ℝ), 
  (∃ x : ℝ, P = (x, Real.log x)) ∧
  (A.1 - P.1 + √3 * P.2 - √3 * A.2 + 1 / 2 * Real.log 3 = 0) →
  Real.dist A B = minimum_distance := sorry

end min_distance_A_B_l168_168658


namespace angle_same_terminal_side_210_l168_168623

theorem angle_same_terminal_side_210 (n : ℤ) : 
  ∃ k : ℤ, 210 = -510 + k * 360 ∧ 0 ≤ 210 ∧ 210 < 360 :=
by
  use 2
  -- proof steps will go here
  sorry

end angle_same_terminal_side_210_l168_168623


namespace sum_of_solutions_tan_sin_l168_168186

theorem sum_of_solutions_tan_sin :
  (∃ ω φ : ℝ, 0 < |φ| ∧ |φ| < π / 2 ∧ ω > 0 ∧
    ∃ f : ℝ → ℝ, f = (λ x, tan (ω * x + φ)) ∧
    (∀ x, f (x + π / ω) = f x) ∧
    f (π / 3) = 0 ∧
    ∃ k_x k_ω k_φ : ℝ, k_ω = 2 ∧ k_φ = π / 3 ∧
    (∑ x in (filter (λ x, 0 ≤ x ∧ x ≤ π) (set.range (λ k : ℤ, (k * π / 2 - π / 6) : ℝ))),
    x = (π / 3 + 5 * π / 6) : ℝ)
  ) := sorry

end sum_of_solutions_tan_sin_l168_168186


namespace minimum_value_f_l168_168710

noncomputable def f (x : ℝ) : ℝ :=
  x^3 + (1/2) * x^2 - 4 * x

theorem minimum_value_f :
  ∃ x : ℝ, f x = -5/2 :=
by
  use 1
  have : f 1 = 1^3 + (1/2) * 1^2 - 4 * 1, by rfl
  rw this
  norm_num

end minimum_value_f_l168_168710


namespace probability_of_rain_given_east_wind_l168_168057

-- Definitions of probabilities
def P (X : Set) : ℚ := sorry
def A : Set := sorry
def B : Set := sorry

axiom PA : P A = 3 / 10
axiom PAB : P (A ∩ B) = 4 / 15

-- Conditional probability of B given A
def P_cond (B A : Set) : ℚ := P (A ∩ B) / P A

theorem probability_of_rain_given_east_wind :
  P_cond B A = 8 / 9 :=
by
  rw [P_cond, PA, PAB]
  sorry

end probability_of_rain_given_east_wind_l168_168057


namespace smallest_k_l168_168206

theorem smallest_k (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 * max m n + min m n - 1 ∧ 
  (∀ (persons : Finset ℕ),
    persons.card ≥ k →
    (∃ (acquainted : Finset (ℕ × ℕ)), acquainted.card = m ∧ 
      (∀ (x y : ℕ), (x, y) ∈ acquainted → (x ∈ persons ∧ y ∈ persons))) ∨
    (∃ (unacquainted : Finset (ℕ × ℕ)), unacquainted.card = n ∧ 
      (∀ (x y : ℕ), (x, y) ∈ unacquainted → (x ∈ persons ∧ y ∈ persons ∧ x ≠ y)))) :=
sorry

end smallest_k_l168_168206


namespace interval_contains_zero_l168_168124

def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem interval_contains_zero :
  ∃ c ∈ Ioo (2 : ℝ) (3 : ℝ), f c = 0 :=
by
  have h1 : f 2 < 0 :=
  by
    calc
      f 2 = Real.log 2 + 2 * 2 - 6 : rfl
      ... < 1 + 4 - 6 : by sorry -- we know Real.log 2 < 1
      ... = -1 : by norm_num
  have h2 : f 3 > 0 := 
  by
    calc
      f 3 = Real.log 3 + 2 * 3 - 6 : rfl
      ... > 0 + 6 - 6 : by sorry -- we know Real.log 3 > 0 
      ... = 0 : by norm_num
  exact IntermediateValueTheorem f 2 3 h1 h2

end interval_contains_zero_l168_168124


namespace exists_perpendicular_intersecting_circles_l168_168472

open EuclideanGeometry

variables {A B C : Point}
variables (k_A k_B k_C : Circle)

theorem exists_perpendicular_intersecting_circles (h_acute : is_acute_triangle A B C) :
  ∃ k_A k_B k_C, 
    (circle.contains k_A A) ∧ (circle.contains k_B B) ∧ (circle.contains k_C C) ∧
    circles_intersect_perpendicularly k_A k_B ∧
    circles_intersect_perpendicularly k_B k_C ∧
    circles_intersect_perpendicularly k_C k_A :=
sorry

end exists_perpendicular_intersecting_circles_l168_168472


namespace position_T_unreachable_l168_168414

variables {Point : Type} (S T : Point)
variables (moveThree : Point → Point → Prop)
variables (moveTwoPerpendicular : Point → Point → Prop)

def Reachable (start finish : Point) : Prop :=
  ∃ p1, moveThree start p1 ∧ moveTwoPerpendicular p1 finish

theorem position_T_unreachable : ¬ Reachable S T :=
  sorry

end position_T_unreachable_l168_168414


namespace grasshoppers_total_l168_168253

theorem grasshoppers_total (grasshoppers_on_plant : ℕ) (dozens_of_baby_grasshoppers : ℕ) (dozen_value : ℕ) : 
  grasshoppers_on_plant = 7 → dozens_of_baby_grasshoppers = 2 → dozen_value = 12 → 
  grasshoppers_on_plant + dozens_of_baby_grasshoppers * dozen_value = 31 :=
by
  intros h1 h2 h3
  sorry

end grasshoppers_total_l168_168253


namespace solve_inequalities_l168_168112

theorem solve_inequalities (x : ℝ) : 
  (x^2 - x ≤ 4 ∧ x > 1 - 2 * x) ↔ (x ∈ Ioc (1 / 3) 4) := 
by
  sorry

end solve_inequalities_l168_168112


namespace crate_minimum_dimension_l168_168408

theorem crate_minimum_dimension (a : ℕ) (h1 : a ≥ 12) :
  min a (min 8 12) = 8 :=
by
  sorry

end crate_minimum_dimension_l168_168408


namespace number_of_correct_propositions_is_zero_l168_168743

-- Define the four propositions
def prop1 := ∀ (A B C : Type) (f : A → B → C),
  (∃ a b c : A, ¬collinear a b c) → (∃ P : A → A → A → Prop, P a b c)

def prop2 := ∀ (Q : Type) (a b c d : Q),
  (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧ eq_sides a b c d) → is_rhombus Q a b c d

def prop3 := ∀ (T : Type) (a b c : T),
  (is_equilateral_triangle T a b c ∧ isosceles_lateral_faces T a b c) → is_regular_tetrahedron T a b c

def prop4 := ∀ (S : Type) (p1 p2 : S) (great_circle : S → S → S),
  (distinct p1 p2 ∧ on_sphere S) → unique_great_circle S p1 p2 great_circle

theorem number_of_correct_propositions_is_zero 
  (P1 P2 P3 P4 : Prop) 
  (h1 : P1 = prop1) 
  (h2 : P2 = prop2) 
  (h3 : P3 = prop3) 
  (h4 : P4 = prop4) 
  (correct_propositions_count : ℕ) 
  : correct_propositions_count = 0 := 
sorry

end number_of_correct_propositions_is_zero_l168_168743


namespace find_m_n_l168_168568

theorem find_m_n (a n m : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x, f x = a^(2 * x - 4) + n)
  (h4 : f m = 2) : m + n = 3 := 
by sorry

end find_m_n_l168_168568


namespace cooking_oil_distribution_l168_168882

theorem cooking_oil_distribution (total_oil : ℝ) (oil_A : ℝ) (oil_B : ℝ) (oil_C : ℝ)
    (h_total_oil : total_oil = 3 * 1000) -- Total oil is 3000 milliliters
    (h_A_B : oil_A = oil_B + 200) -- A receives 200 milliliters more than B
    (h_B_C : oil_B = oil_C + 200) -- B receives 200 milliliters more than C
    : oil_B = 1000 :=              -- We need to prove B receives 1000 milliliters
by
  sorry

end cooking_oil_distribution_l168_168882


namespace trees_divisible_by_43_l168_168608

section
variables (m s k l : ℕ)

/--
In a forest, there were pine trees, cedar trees, and larch trees, each having an equal 
number of cones. A light breeze caused a few cones to fall to the ground. It turned out 
that 11% of the cones fell from each pine tree, 54% from each cedar tree, and 97% from 
each larch tree. Overall, exactly 30% of all the cones on the trees fell. Prove that the 
total number of trees in the forest is divisible by 43.
-/
theorem trees_divisible_by_43 
    (h1 : 0.11 * m * s ∈ ℕ)
    (h2 : 0.54 * m * k ∈ ℕ)
    (h3 : 0.97 * m * l ∈ ℕ)
    (h4 : 0.3 * m * (s + k + l) ∈ ℕ)
    (h5 : 0.3 * m * (s + k + l) = 0.11 * m * s + 0.54 * m * k + 0.97 * m * l)
    : 43 ∣ (s + k + l) :=
sorry

end

end trees_divisible_by_43_l168_168608


namespace find_integer_pairs_l168_168967

theorem find_integer_pairs : 
  ∀ (x y : Int), x^3 = y^3 + 2 * y^2 + 1 ↔ (x, y) = (1, 0) ∨ (x, y) = (1, -2) ∨ (x, y) = (-2, -3) :=
by
  intros x y
  sorry

end find_integer_pairs_l168_168967


namespace sum_of_integers_divisible_by_3_between_2_and_20_l168_168008

theorem sum_of_integers_divisible_by_3_between_2_and_20 : 
  ∑ (i : ℕ) in finset.filter (λ x => x % 3 = 0 ∧ x > 2 ∧ x < 20) (finset.range 20) = 63 :=
begin
  sorry
end

end sum_of_integers_divisible_by_3_between_2_and_20_l168_168008


namespace sum_of_squares_of_medians_l168_168368

/-- Define the sides of the triangle -/
def side_lengths := (13, 13, 10)

/-- Define the triangle as isosceles with given side lengths -/
def isosceles_triangle (a b c : ℝ) := (a = b) ∧ (a ≠ c)

/-- Define the median lengths calculation -/
noncomputable def median_length (a b c : ℝ) : ℝ :=
  if h : isosceles_triangle a b c then
    let AD := Real.sqrt (a ^ 2 - (c / 2) ^ 2) in
    let BE_CF := Real.sqrt ((a / 2) ^ 2 + (3 / 4) * c ^ 2) in
    AD^2 + BE_CF^2 + BE_CF^2
  else
    0

/-- The sum of the squares of the lengths of the medians for the given triangle is -/
theorem sum_of_squares_of_medians : median_length 13 13 10 = 447.5 := by
  sorry

end sum_of_squares_of_medians_l168_168368


namespace silvia_saves_50_l168_168690

variables
  (price : ℕ := 1000) -- Suggested retail price of the guitar
  (gc_discount : ℕ := 15) -- Guitar Center discount in percentage
  (gc_shipping : ℕ := 100) -- Guitar Center shipping fee
  (sw_discount : ℕ := 10) -- Sweetwater discount in percentage
  (sw_shipping : ℕ := 0) -- Sweetwater shipping fee

def guitar_price_after_discount (price : ℕ) (discount : ℕ) : ℕ := 
  price - (price * discount / 100)

def total_cost (price : ℕ) (discount : ℕ) (shipping : ℕ) : ℕ :=
  (guitar_price_after_discount price discount) + shipping

def savings (price : ℕ) (gc_discount : ℕ) (gc_shipping : ℕ) (sw_discount : ℕ) (sw_shipping : ℕ) : ℕ :=
  total_cost price gc_discount gc_shipping - total_cost price sw_discount sw_shipping

theorem silvia_saves_50 (price : ℕ) (gc_discount : ℕ) (gc_shipping : ℕ) (sw_discount : ℕ) (sw_shipping : ℕ) :
  savings price gc_discount gc_shipping sw_discount sw_shipping = 50 := by
begin
  sorry
end

end silvia_saves_50_l168_168690


namespace four_points_concyclic_l168_168218

open EuclideanGeometry

variables {A B C E F : Point}

-- Conditions: BE and CF are the altitudes from B and C to the sides AC and AB of triangle ABC respectively.
def is_altitude (P Q R : Point) (PQ PR : Line) : Prop := 
  PQ ⊥ line_through Q R ∧ PR ⊥ line_through P R

def is_triangle (A B C : Point) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Problem statement: Prove that B, E, C, and F are concyclic.
theorem four_points_concyclic
  (htriangle: is_triangle A B C)
  (hBE : is_altitude B E (side AC) (side BE))
  (hCF : is_altitude C F (side AB) (side CF)) :
  concyclic {B, E, C, F} :=
by
  sorry

end four_points_concyclic_l168_168218


namespace smallest_append_digits_l168_168835

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168835


namespace smallest_digits_to_append_l168_168843

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168843


namespace smallest_number_append_l168_168819

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168819


namespace colton_share_l168_168074

-- Definitions
def footToInch (foot : ℕ) : ℕ := 12 * foot -- 1 foot equals 12 inches

-- Problem conditions
def coltonBurgerLength := footToInch 1 -- Colton bought a foot long burger
def sharedBurger (length : ℕ) : ℕ := length / 2 -- shared half with his brother

-- Equivalent proof problem statement
theorem colton_share : sharedBurger coltonBurgerLength = 6 := 
by sorry

end colton_share_l168_168074


namespace numbers_from_1_to_20_with_five_eights_l168_168762

noncomputable def can_form_numbers (n : ℕ) (digits : ℕ) : Prop :=
  ∃ f : fin 5 → ℕ, (∀ i, f i = 8) ∧ (∃ op : ℕ → ℕ, op (f 0, f 1, f 2, f 3, f 4) = n)

theorem numbers_from_1_to_20_with_five_eights :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 20 → can_form_numbers n 8 :=
by
  intros n hn
  cases n
  { simp at hn }
  case dp.eight :

  sorry

end numbers_from_1_to_20_with_five_eights_l168_168762


namespace cyclic_circumcircle_intersect_l168_168259

noncomputable def cyclic {A B C : Type} [CommRing A] (circle : set (B × C)) (P Q R : B) : Prop :=
  ∃ k, k ∈ circle ∧ P ∈ circle ∧ Q ∈ circle ∧ R ∈ circle

variables {ABC P Q R X : Type} [Triangle ABC] [Point P] [Point Q] [Point R] [Point X]

/-- Given a triangle ABC, points P, Q, and R on segments [BC], [CA], and [AB] respectively. 
    The circumcircles of triangles AQR and BRP intersect at a second point X.
    Prove that X is also on the circumcircle of triangle CQP. -/
theorem cyclic_circumcircle_intersect (hP : P ∈ segment BC) (hQ : Q ∈ segment CA) (hR : R ∈ segment AB)
  (h_circ_1 : cyclic AQR X) (h_circ_2 : cyclic BRP X) : cyclic CQP X :=
sorry

end cyclic_circumcircle_intersect_l168_168259


namespace min_diff_composite_sum_101_l168_168900

def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, (p > 1 ∧ q > 1 ∧ p * q = n)

theorem min_diff_composite_sum_101 : ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = 101 ∧ (∀ x y : ℕ, is_composite x → is_composite y → x + y = 101 → |x - y| ≥ 1) :=
begin
  sorry
end

end min_diff_composite_sum_101_l168_168900


namespace probability_is_correct_l168_168923

-- Define the properties of the islands
def hasTreasureAndNoTraps : ℚ := 1 / 3
def hasTrapsAndNoTreasure : ℚ := 1 / 6
def hasNeither : ℚ := 1 / 2

-- Define the total number of islands
def totalIslands : ℕ := 7

-- Define the probability calculation function
noncomputable def probabilityOfExactlyFourTreasures : ℚ :=
  let combinations := Nat.choose totalIslands 4 in
  let probFourTreasures := (hasTreasureAndNoTraps ^ 4) in
  let probThreeNoTrapsNoTreasure := (hasNeither ^ 3) in
  combinations * probFourTreasures * probThreeNoTrapsNoTreasure

-- The statement that needs to be proven
theorem probability_is_correct :
  probabilityOfExactlyFourTreasures = 35 / 648 :=
sorry

end probability_is_correct_l168_168923


namespace proof_base_5_conversion_and_addition_l168_168202

-- Define the given numbers in decimal (base 10)
def n₁ := 45
def n₂ := 25

-- Base 5 conversion function and proofs of correctness
def to_base_5 (n : ℕ) : ℕ := sorry
def from_base_5 (n : ℕ) : ℕ := sorry

-- Converted values to base 5
def a₅ : ℕ := to_base_5 n₁
def b₅ : ℕ := to_base_5 n₂

-- Sum in base 5
def c₅ : ℕ := a₅ + b₅  -- addition in base 5

-- Convert the final sum back to decimal base 10
def d₁₀ : ℕ := from_base_5 c₅

theorem proof_base_5_conversion_and_addition :
  d₁₀ = 65 ∧ to_base_5 65 = 230 :=
by sorry

end proof_base_5_conversion_and_addition_l168_168202


namespace polynomial_p_l168_168130

def p (x : ℝ) : ℝ := x^2 + 1

theorem polynomial_p
  (p : ℝ → ℝ)
  (h1 : p 3 = 10)
  (h2 : p 4 = 17)
  (h3 : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) :
  p = λ x, x^2 + 1 :=
by
  sorry

end polynomial_p_l168_168130


namespace find_roots_l168_168120

def poly (x : ℝ) : ℝ := x^3 + 2 * x^2 - 5 * x - 6

theorem find_roots :
  polynomial.eval (-1) (polynomial.C ∘ poly) = 0 ∧
  polynomial.eval (2) (polynomial.C ∘ poly) = 0 ∧
  polynomial.eval (-3) (polynomial.C ∘ poly) = 0 ∧
  ∀ x : ℝ, (polynomial.eval x (polynomial.C ∘ poly) = 0) → (x = -1 ∨ x = 2 ∨ x = -3) :=
by
  sorry

end find_roots_l168_168120


namespace triangle_similarity_length_CE_l168_168689

-- Definitions based on conditions in problem statement
variables {A B C D E F : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (triangle_ABC : set A) (line_DEF : set D)
variables (AB : ℝ) (DE : ℝ) (CE : ℝ) (AE_bisects_FED : Prop)
variables (parallel_DEF_AB : Prop)

-- Problem translated properly into Lean statement
theorem triangle_similarity_length_CE
  (h1 : AB = 10)
  (h2 : DE = 6)
  (h3 : parallel_DEF_AB)
  (h4 : AE_bisects_FED) :
  CE = 15 :=
by sorry

end triangle_similarity_length_CE_l168_168689


namespace two_squares_always_similar_l168_168877

-- Define geometric shapes and their properties
inductive Shape
| Rectangle : Shape
| Rhombus   : Shape
| Square    : Shape
| RightAngledTriangle : Shape

-- Define similarity condition
def similar (s1 s2 : Shape) : Prop :=
  match s1, s2 with
  | Shape.Square, Shape.Square => true
  | _, _ => false

-- Prove that two squares are always similar
theorem two_squares_always_similar : similar Shape.Square Shape.Square = true :=
by
  sorry

end two_squares_always_similar_l168_168877


namespace evaluate_f_function_l168_168524

theorem evaluate_f_function :
  ∀ (a θ : ℝ), 0 < a ∧ a < sqrt 3 * sin θ ∧ (θ ∈ set.Icc (π / 6) (Real.arcsin (Real.sqrt 3 ^ 3 / 2))) →
  (sin θ) ^ 3 + 4 / (3 * a * (sin θ) ^ 2 - a ^ 3) = 3 * sqrt 3 :=
by sorry

end evaluate_f_function_l168_168524


namespace cone_height_l168_168553

theorem cone_height {r : ℝ} (h1 : r = 2) : ∃ h : ℝ, h = sqrt 3 :=
by
  sorry

end cone_height_l168_168553


namespace sum_ceil_floor_diff_log_eq_499407_l168_168459

noncomputable def log_sqrt3 (k : ℕ) : ℝ := real.log k / real.log (real.sqrt 3)

def ceil_floor_diff (x : ℝ) : ℝ :=
  if x ≠ real.floor x then 1 else 0

def ceil_log_sqrt3_sub_floor_log_sqrt3 (k : ℕ) : ℝ :=
  ceil_floor_diff (log_sqrt3 k)

def sum_k_ceil_log_sqrt3_sub_floor_log_sqrt3 (n : ℕ) : ℝ :=
  ∑ k in finset.range n, k * ceil_log_sqrt3_sub_floor_log_sqrt3 k

theorem sum_ceil_floor_diff_log_eq_499407 : sum_k_ceil_log_sqrt3_sub_floor_log_sqrt3 1001 = 499407 := by
  sorry

end sum_ceil_floor_diff_log_eq_499407_l168_168459


namespace smallest_append_digits_l168_168837

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168837


namespace numbers_from_1_to_20_with_five_eights_l168_168763

noncomputable def can_form_numbers (n : ℕ) (digits : ℕ) : Prop :=
  ∃ f : fin 5 → ℕ, (∀ i, f i = 8) ∧ (∃ op : ℕ → ℕ, op (f 0, f 1, f 2, f 3, f 4) = n)

theorem numbers_from_1_to_20_with_five_eights :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 20 → can_form_numbers n 8 :=
by
  intros n hn
  cases n
  { simp at hn }
  case dp.eight :

  sorry

end numbers_from_1_to_20_with_five_eights_l168_168763


namespace find_four_coprime_l168_168237

theorem find_four_coprime (A : Finset ℕ) (hA1 : A.card = 133) 
  (hA2 : (Finset.filter (λ (x : (ℕ × ℕ)), Nat.coprime x.fst x.snd) (A.product A)).card / 2 ≥ 799) :
  ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd d a = 1 :=
sorry

end find_four_coprime_l168_168237


namespace distance_between_Tolya_and_Anton_l168_168749

theorem distance_between_Tolya_and_Anton (v_A v_S v_T : ℝ) (h1 : v_S = 0.9 * v_A) (h2 : v_T = 0.81 * v_A) :
  let t_A := 100 / v_A in
  let d_T := v_T * t_A in
  100 - d_T = 19 :=
by
  sorry

end distance_between_Tolya_and_Anton_l168_168749


namespace part1_area_quadrilateral_part2_maximized_line_equation_l168_168204

noncomputable def area_MA_NB (α : ℝ) : ℝ :=
  (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)

theorem part1_area_quadrilateral (α : ℝ) :
  area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2) :=
by sorry

theorem part2_maximized_line_equation :
  ∃ α : ℝ, area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)
    ∧ (Real.tan α = -1 / 2) ∧ (∀ x : ℝ, x = -1 / 2 * y + Real.sqrt 5 / 2) :=
by sorry

end part1_area_quadrilateral_part2_maximized_line_equation_l168_168204


namespace at_least_n_distinct_lines_l168_168287

theorem at_least_n_distinct_lines {n : ℕ} (h : ∀ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →  ¬ collinear {p1, p2, p3}) :
  ∃ (L : set (set ℕ)), L.card ≥ n ∧ (∀ l ∈ L, ∃ p1 p2, p1 ≠ p2 ∧ l = {p1, p2}) :=
sorry

end at_least_n_distinct_lines_l168_168287


namespace equation_of_line_AB_l168_168407

noncomputable def circle_center : ℝ × ℝ := (1, 0)  -- center of the circle (x-1)^2 + y^2 = 1
noncomputable def circle_radius : ℝ := 1          -- radius of the circle (x-1)^2 + y^2 = 1
noncomputable def point_P : ℝ × ℝ := (3, 1)       -- point P(3,1)

theorem equation_of_line_AB :
  ∃ (AB : ℝ → ℝ → Prop),
    (∀ x y, AB x y ↔ (2 * x + y - 3 = 0)) := sorry

end equation_of_line_AB_l168_168407


namespace jemma_grasshoppers_l168_168251

-- Definitions corresponding to the conditions
def grasshoppers_on_plant : ℕ := 7
def baby_grasshoppers : ℕ := 2 * 12

-- Theorem statement equivalent to the problem
theorem jemma_grasshoppers : grasshoppers_on_plant + baby_grasshoppers = 31 :=
by
  sorry

end jemma_grasshoppers_l168_168251


namespace max_length_third_side_l168_168696

open Real

theorem max_length_third_side (A B C : ℝ) (a b c : ℝ) 
  (h1 : cos (2 * A) + cos (2 * B) + cos (2 * C) = 1)
  (h2 : a = 9) 
  (h3 : b = 12)
  (h4 : a^2 + b^2 = c^2) : 
  c = 15 := 
sorry

end max_length_third_side_l168_168696


namespace isosceles_triangle_proof_l168_168436

-- Defining the main structures and conditions in Lean
structure IsoscelesTriangle (A B C D E M : Type) [Field A] [Field B] [Field C] [Field D] [Field E] [Field M] :=
  (isosceles : AB = AC)
  (midpoint : M = (A + C) / 2)
  (D_on_circumcircle : D ∈ circumcircle(B, M, C))
  (D_not_including_M : D ∉ arc_BMC_not_including_M)
  (BD_intersect_AC_at_E : (BD).intersect(AC) = E)
  (DE_eq_MC : DE = MC)

-- Statement to prove MD^2 = AC * CE and CE^2 = (BC * MD) / 2
theorem isosceles_triangle_proof (A B C D E M : Type) [Field A] [Field B] [Field C] [Field D] [Field E] [Field M]
  (t : IsoscelesTriangle A B C D E M) :
  (MD ^ 2 = AC * CE) ∧ (CE ^ 2 = (BC * MD) / 2) :=
sorry

end isosceles_triangle_proof_l168_168436


namespace train_speed_is_59_5_kmh_l168_168051

-- Definitions for the conditions in the problem
def train_length : ℝ := 150  -- Train length in meters
def man_speed_kmh : ℝ := 8   -- Man's speed in km/h
def passing_time : ℝ := 8    -- Time to pass the man in seconds

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)
def man_speed_ms : ℝ := kmh_to_ms man_speed_kmh

-- Relative speed when train passes the man
def relative_speed : ℝ := train_length / passing_time

-- The speed of the train in m/s
def train_speed_ms : ℝ := relative_speed - man_speed_ms

-- Conversion factor from m/s to km/h
def ms_to_kmh (speed_ms : ℝ) : ℝ := speed_ms * (3600 / 1000)
def train_speed_kmh : ℝ := ms_to_kmh train_speed_ms

-- Main theorem stating the speed of the train given the conditions
theorem train_speed_is_59_5_kmh :
  train_speed_kmh = 59.5 := sorry

end train_speed_is_59_5_kmh_l168_168051


namespace volume_of_tetrahedron_C1LMN_l168_168031

noncomputable def volume_tetrahedron (A B D A1 C1 L M N : ℝ × ℝ × ℝ) : ℝ :=
  (1/6) * abs ((B.1 - A.1) * ((D.2 - A.2) * (A1.3 - A.3) - (A1.2 - A.2) * (D.3 - A.3))
            - (D.1 - A.1) * ((B.2 - A.2) * (A1.3 - A.3) - (A1.2 - A.2) * (B.3 - A.3))
            + (A1.1 - A.1) * ((B.2 - A.2) * (D.3 - A.3) - (D.2 - A.2) * (B.3 - A.3)))

theorem volume_of_tetrahedron_C1LMN :
  let A        := (0, 0, 0)
  let B        := (251, 0, 0)
  let D        := (0, 3, 0)
  let A1       := (0, 0, 2)
  let C1       := (251, 3, 2)
  let plane_eq := (6, -1004, 753)

  ∃ (L M N : ℝ × ℝ × ℝ),
    volume_tetrahedron A B D A1 C1 L M N = (tbd)
  :=
  sorry

end volume_of_tetrahedron_C1LMN_l168_168031


namespace mike_books_eq_total_bobby_kristi_l168_168942

theorem mike_books_eq_total_bobby_kristi (bobby_books kristi_books : ℕ) 
  (hb : bobby_books = 142) (hk : kristi_books = 78) : 
  ∃ mike_books : ℕ, mike_books = bobby_books + kristi_books ∧ mike_books = 220 :=
by
  use 220
  rw [hb, hk]
  exact Nat.add_eq_of_certain_sum 142 78
  sorry

end mike_books_eq_total_bobby_kristi_l168_168942


namespace consecutive_integer_sets_sum_100_l168_168318

theorem consecutive_integer_sets_sum_100 :
  ∃ s : Finset (Finset ℕ), 
    (∀ seq ∈ s, (∀ x ∈ seq, x > 0) ∧ (seq.sum id = 100)) ∧
    (s.card = 2) :=
sorry

end consecutive_integer_sets_sum_100_l168_168318


namespace pizza_total_cost_l168_168256

theorem pizza_total_cost (pizzas : ℕ) (slices_per_pizza : ℕ) (cost_per_5_slices : ℕ) :
  pizzas = 3 ∧ slices_per_pizza = 12 ∧ cost_per_5_slices = 10 →
  (5 * (cost_per_5_slices / 5) * (slices_per_pizza * pizzas) / slices_per_pizza) = 72 :=
by
  intro h
  cases h with hpizzas hrest
  cases hrest with hslices hcost
  have hp1 : pizzas = 3 := hpizzas
  have hs1 : slices_per_pizza = 12 := hslices
  have hc1 : cost_per_5_slices = 10 := hcost
  sorry

end pizza_total_cost_l168_168256


namespace find_f_k_l_l168_168646

noncomputable
def f : ℕ → ℕ := sorry

axiom f_condition_1 : f 1 = 1
axiom f_condition_2 : ∀ n : ℕ, 3 * f n * f (2 * n + 1) = f (2 * n) * (1 + 3 * f n)
axiom f_condition_3 : ∀ n : ℕ, f (2 * n) < 6 * f n

theorem find_f_k_l (k l : ℕ) (h : k < l) : 
  (f k + f l = 293) ↔ 
  ((k = 121 ∧ l = 4) ∨ (k = 118 ∧ l = 4) ∨ 
   (k = 109 ∧ l = 16) ∨ (k = 16 ∧ l = 109)) := 
by 
  sorry

end find_f_k_l_l168_168646


namespace smallest_digits_to_append_l168_168859

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168859


namespace value_of_expression_l168_168591

theorem value_of_expression (x y : ℤ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 :=
by
  -- Substitute the given values into the expression and calculate
  sorry

end value_of_expression_l168_168591


namespace num_solutions_eq_five_l168_168211

theorem num_solutions_eq_five :
  {x : ℤ | (x^2 - x - 2)^(x + 3) = 1}.card = 5 :=
by
  sorry

end num_solutions_eq_five_l168_168211


namespace polynomial_reformulation_l168_168966

theorem polynomial_reformulation : 
  ∃ (a h k : ℝ), (∀ x : ℝ, 6 * x ^ 2 + 12 * x + 8 = a * (x - h) ^ 2 + k) ∧ a + h + k = 9 :=
begin
  sorry
end

end polynomial_reformulation_l168_168966


namespace parallel_transitivity_l168_168543

-- Definitions of the data
variables {l m n : Type} [line l] [line m] [line n]

-- Parallel relation for lines
def parallel (x y : Type) [line x] [line y] : Prop := sorry

-- Given conditions
variables (h1 : parallel m l) (h2 : parallel n l)

-- The theorem statement
theorem parallel_transitivity : parallel m n :=
by {
  -- The proof would go here
  sorry
}

end parallel_transitivity_l168_168543


namespace correct_propositions_l168_168192

-- Definitions
def prop1 (L1 L2 : Plane) (P3 : Plane) : Prop :=
  (∀ p1 p2 : Line, p1 ∈ L1 → p2 ∈ L2 → 
  (p1 ∥ P3) → (p2 ∥ P3) →  p1 ∩ p2 ≠ ∅ → L1 ∥ L2)

def prop2 (L1 : Plane) (L2 : Plane) (l : Line) : Prop :=
  (l ⊥ L1) → (l ∈ L2) → (L1 ⊥ L2)

def prop3 (l1 l2 l3 : Line) : Prop :=
  (l1 ⊥ l3) → (l2 ⊥ l3) → (l1 ∥ l2)

def prop4 (L1 L2 : Plane) (l : Line) : Prop :=
  (L1 ⊥ L2) → (l ⊥ (L1 ∩ L2)) → (l ∉ L1) → (l ⊥ L2)

-- Main theorem stating the correctness of propositions 2 and 4 while invalidating propositions 1 and 3
theorem correct_propositions: 
  prop2 ∧ prop4 ∧ ¬ prop1 ∧ ¬ prop3 := 
sorry

end correct_propositions_l168_168192


namespace find_pairs_l168_168116

theorem find_pairs (n k : ℕ) : (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) := by
  sorry

end find_pairs_l168_168116


namespace number_is_3034_l168_168922

theorem number_is_3034 (number : ℝ) (h : number - 1002 / 20.04 = 2984) : number = 3034 :=
sorry

end number_is_3034_l168_168922


namespace ages_of_boys_l168_168729

theorem ages_of_boys (a b c : ℕ) (h : a + b + c = 29) (h₁ : a = b) (h₂ : c = 11) : a = 9 ∧ b = 9 := 
by
  sorry

end ages_of_boys_l168_168729


namespace find_m_l168_168475

-- Define the operation a * b
def star (a b : ℝ) : ℝ := a * b + a - 2 * b

theorem find_m (m : ℝ) (h : star 3 m = 17) : m = 14 :=
by
  -- Placeholder for the proof
  sorry

end find_m_l168_168475


namespace first_ship_rescued_boy_l168_168416

noncomputable def river_speed : ℝ := 3 -- River speed is 3 km/h

-- Define the speeds of the ships
def ship1_speed_upstream : ℝ := 4 
def ship2_speed_upstream : ℝ := 6 
def ship3_speed_upstream : ℝ := 10 

-- Define the distance downstream where the boy was found
def boy_distance_from_bridge : ℝ := 6

-- Define the equation for the first ship
def first_ship_equation (c : ℝ) : Prop := (10 - c) / (4 + c) = 1 + 6 / c

-- The problem to prove:
theorem first_ship_rescued_boy : first_ship_equation river_speed :=
by sorry

end first_ship_rescued_boy_l168_168416


namespace collinear_points_sum_l168_168586

theorem collinear_points_sum (a b : ℝ) :
  let A := (1, 5, -1)
  let B := (2, 4, 1)
  let C := (a, 3, b + 2)
  (C - B) = (λ * (B - A)) → a + b = 4 :=
by
  sorry

end collinear_points_sum_l168_168586


namespace geometric_sequence_term_l168_168528

open_locale big_operators

noncomputable def a_n (n : ℕ) : ℚ :=
  if n = 0 then 0 else 1 / 3^n

noncomputable def S_n (n : ℕ) : ℚ := 
  ∑ i in range(n + 1), a_n i

theorem geometric_sequence_term (n : ℕ) 
  (h1 : a_n 1 = 1/3)
  (h2 : ∀ n, a_n n = (1 / 3) * (1 / 3)^(n-1))
  (h3 : (S_n 1), (2 * S_n 2), (3 * S_n 3) form arithmetic_sequence) :
  a_n = λ n, 1 / 3^n :=
sorry

end geometric_sequence_term_l168_168528


namespace brokerage_percentage_l168_168928

theorem brokerage_percentage {face_value discount_percentage cost_price : ℝ} 
  (h_face_value : face_value = 100)
  (h_discount_percentage : discount_percentage = 7)
  (h_cost_price : cost_price = 93.2) :
  let discount := face_value * (discount_percentage / 100)
  let price_before_brokerage := face_value - discount
  let brokerage_fee := cost_price - price_before_brokerage
  let brokerage_percentage := (brokerage_fee / price_before_brokerage) * 100
  brokerage_percentage ≈ 0.2151 :=
by
  sorry

end brokerage_percentage_l168_168928


namespace find_principal_amount_l168_168929

-- Given conditions
def SI : ℝ := 4016.25
def R : ℝ := 0.14
def T : ℕ := 5

-- Question: What is the principal amount P?
theorem find_principal_amount : (SI / (R * T) = 5737.5) :=
sorry

end find_principal_amount_l168_168929


namespace train_pass_time_l168_168930

-- Definitions based on conditions
def length_of_train : ℝ := 300
def speed_of_train_kmh : ℝ := 90
def speed_of_man_kmh : ℝ := 15
def wind_resistance : ℝ := 0.05

-- Derived definitions
def speed_of_train_ms : ℝ := (speed_of_train_kmh * 1000) / 3600
def speed_of_man_ms : ℝ := (speed_of_man_kmh * 1000) / 3600
def decrease_due_to_wind_resistance : ℝ := wind_resistance * speed_of_train_ms
def effective_speed_of_train : ℝ := speed_of_train_ms - decrease_due_to_wind_resistance
def relative_speed : ℝ := effective_speed_of_train + speed_of_man_ms

-- Theorem to prove
theorem train_pass_time : length_of_train / relative_speed ≈ 10.75 := 
by
  -- This is where you would write the proof, but it's omitted in this task.
  sorry

end train_pass_time_l168_168930


namespace log_seq_none_of_these_l168_168695

open Real

theorem log_seq_none_of_these
  (x y z : ℝ) (m : ℕ)
  (hx : x = y^(1/3))
  (hz : z = y^(2/3))
  (hm1 : 2 < m)
  (hm2 : (m:ℝ) < y) :
  ¬ (∃ c: ℝ, ∀ n, (log x m, log y m, log z m) = (c^(n-1), c^n, c^(n+1))) ∧
  ¬ (∃ a d: ℝ, (log x m, log y m, log z m) = (a, a + d, a + 2 * d)) ∧
  ¬ (∃ a b: ℝ, (1/log x m, 1/log y m, 1/log z m) = (a, a + b, a + 2 * b)) ∧
  true :=
sorry

end log_seq_none_of_these_l168_168695


namespace vector_subtraction_l168_168207

variables (a b : ℝ × ℝ)
def v1 : ℝ × ℝ := (3, 2)
def v2 : ℝ × ℝ := (0, -1)

theorem vector_subtraction : (let a := v1; let b := v2 in (3 : ℝ) • b - a) = (-3, -5) :=
by sorry

end vector_subtraction_l168_168207


namespace quadrilateral_CD_l168_168238

theorem quadrilateral_CD (AB BD BC CD : ℝ) 
    (h1: ∠ BAD = ∠ ADC) 
    (h2: ∠ ABD = ∠ BCD)
    (hAB: AB = 7) 
    (hBD: BD = 9) 
    (hBC: BC = 5) 
    (hCD: CD = 7) :
  let m := 7 in let n := 1 in m + n = 8 := 
by
  sorry

end quadrilateral_CD_l168_168238


namespace find_S9_l168_168151

-- Defining the sequence and given sums
def geometric_sequence_sum (S : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ a r, (∀ n, S n = a * (r^n - 1) / (r - 1)) → 
         S 3 = 3 ∧ S 6 = 15

theorem find_S9 (S : ℕ → ℝ) : geometric_sequence_sum S 9 → S 9 = 63 :=
by {
  intros h,
  sorry
}

end find_S9_l168_168151


namespace number_of_non_congruent_rectangles_with_perimeter_60_l168_168041

open Nat

theorem number_of_non_congruent_rectangles_with_perimeter_60 
  (w h : ℕ) (hp : 2 * (w + h) = 60) : 15 :=
sorry

end number_of_non_congruent_rectangles_with_perimeter_60_l168_168041


namespace largest_mystical_integer_is_81649_l168_168423

/-- A positive integer is called mystical if it has at least two digits and every pair
    of two consecutive digits, read from left to right, forms a perfect square. -/
def is_mystical (n : ℕ) : Prop :=
  n ≥ 10 ∧ (∀ i, (10 ≤ (n / (10^i) % 100) ∧ (√(n / (10^i) % 100))^2 = n / (10^i) % 100) 
  ∨ (n / (10^i) % 100 = 25)
  ∨ (n / (10^i) % 100 = 36)
  ∨ (n / (10^i) % 100 = 49)
  ∨ (n / (10^i) % 100 = 64)
  ∨ (n / (10^i) % 100 = 81))

/-- The largest mystical integer is 81649. -/
theorem largest_mystical_integer_is_81649 : ∀ n : ℕ, is_mystical n → n ≤ 81649 :=
by { sorry }

end largest_mystical_integer_is_81649_l168_168423


namespace ella_share_fraction_l168_168448

theorem ella_share_fraction (l : ℕ) (h : l > 0) :
  let ella := 2 * l,
      connor := 4 * l,
      total := l + ella + connor,
      per_person := total / 3 in
  (per_person = 7 * l / 3) →
  (ella - per_person) / ella = 1 / 6 :=
by
  intros
  sorry

end ella_share_fraction_l168_168448


namespace probability_proofs_l168_168986

noncomputable def second_quality_probability (p : ℝ) : Prop :=
  (1 - p)^2 + 2 * p * (1 - p) = 0.96

noncomputable def at_least_one_second_quality_probability (total_pieces : ℕ) (second_quality_pieces : ℕ) : Prop :=
  let none_second_quality := (total_pieces - second_quality_pieces).choose 2 / total_pieces.choose 2
  in 1 - none_second_quality = 179 / 495

theorem probability_proofs (p : ℝ) (total_pieces : ℕ) (second_quality_pieces : ℕ)
  (h1 : second_quality_probability p)
  (h2 : total_pieces = 100)
  (h3 : second_quality_pieces = 20) : 
  p = 0.2 ∧ at_least_one_second_quality_probability total_pieces second_quality_pieces :=
by {
  sorry
}

end probability_proofs_l168_168986


namespace job_completion_time_l168_168592

theorem job_completion_time (initial_men : ℕ) (initial_days : ℕ) (extra_men : ℕ) (interval_days : ℕ) (total_days : ℕ) : 
  initial_men = 20 → 
  initial_days = 15 → 
  extra_men = 10 → 
  interval_days = 5 → 
  total_days = 12 → 
  ∀ n, (20 * 5 + (20 + 10) * 5 + (20 + 10 + 10) * n.succ = 300 → n + 10 + n.succ = 12) :=
by
  intro h1 h2 h3 h4 h5
  sorry

end job_completion_time_l168_168592


namespace sufficient_but_not_necessary_l168_168200

theorem sufficient_but_not_necessary (x : ℝ) : 
  (1 < x ∧ x < 2) → (x > 0) ∧ ¬((x > 0) → (1 < x ∧ x < 2)) := 
by 
  sorry

end sufficient_but_not_necessary_l168_168200


namespace value_of_f_at_6_l168_168173

variable {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry

-- Conditions
axiom odd_function (x : R) : f (-x) = -f x
axiom periodicity (x : R) : f (x + 2) = -f x

-- Theorem to prove
theorem value_of_f_at_6 : f 6 = 0 := by sorry

end value_of_f_at_6_l168_168173


namespace append_digits_divisible_by_all_less_than_10_l168_168796

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168796


namespace sin_double_angle_l168_168165

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Ioo (-π / 2) 0) (h2 : tan (π / 4 - α) = 3 * cos (2 * α)) :
  sin (2 * α) = -2 / 3 := 
sorry

end sin_double_angle_l168_168165


namespace cupcakes_left_l168_168290

theorem cupcakes_left (initial_cupcakes : ℕ)
  (students_delmont : ℕ) (ms_delmont : ℕ)
  (students_donnelly : ℕ) (mrs_donnelly : ℕ)
  (school_nurse : ℕ) (school_principal : ℕ) (school_custodians : ℕ)
  (favorite_teachers : ℕ) (cupcakes_per_favorite_teacher : ℕ)
  (other_classmates : ℕ) :
  initial_cupcakes = 80 →
  students_delmont = 18 → ms_delmont = 1 →
  students_donnelly = 16 → mrs_donnelly = 1 →
  school_nurse = 1 → school_principal = 1 → school_custodians = 3 →
  favorite_teachers = 5 → cupcakes_per_favorite_teacher = 2 → 
  other_classmates = 10 →
  initial_cupcakes - (students_delmont + ms_delmont +
                      students_donnelly + mrs_donnelly +
                      school_nurse + school_principal + school_custodians +
                      favorite_teachers * cupcakes_per_favorite_teacher +
                      other_classmates) = 19 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _
  sorry

end cupcakes_left_l168_168290


namespace simplify_and_find_ab_ratio_l168_168509

-- Given conditions
def given_expression (k : ℤ) : ℤ := 10 * k + 15

-- Simplified form
def simplified_form (k : ℤ) : ℤ := 2 * k + 3

-- Proof problem statement
theorem simplify_and_find_ab_ratio
  (k : ℤ) :
  let a := 2
  let b := 3
  (10 * k + 15) / 5 = 2 * k + 3 → 
  (a:ℚ) / (b:ℚ) = 2 / 3 := sorry

end simplify_and_find_ab_ratio_l168_168509


namespace xy_value_l168_168758

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : xy = 21 :=
sorry

end xy_value_l168_168758


namespace sum_of_roots_l168_168144

theorem sum_of_roots (r s t : ℝ) (hroots : 3 * (r^3 + s^3 + t^3) + 9 * (r^2 + s^2 + t^2) - 36 * (r + s + t) + 12 = 0) :
  r + s + t = -3 :=
sorry

end sum_of_roots_l168_168144


namespace product_of_solutions_l168_168138

noncomputable def absolute_value (x : ℝ) : ℝ :=
  if x >= 0 then x else -x

theorem product_of_solutions : (∏ x in ({3, -3} : finset ℝ), x) = -9 :=
by 
  have hsol : {x : ℝ | absolute_value x = 3 * (absolute_value x - 2)} = {3, -3},
  { sorry },  -- Proof that the solutions set is exactly {3, -3}
  rw finset.prod_eq_mul,
  norm_num
  sorry

end product_of_solutions_l168_168138


namespace proposition1_proposition2_proposition3_proposition4_l168_168938

theorem proposition1 : ¬(∀ (T : Triangle), (∃ (a b c : ℝ), a > b → ∠T a > ∠T b) ↔ (∃ (a b c : ℝ), ∠T b < ∠T a → b > a)) :=
sorry  -- Negating the statement which is true.

theorem proposition2 (m : ℝ) : (∃ x : ℝ, x^2 - 2 * x + m = 0) ↔ (m ≤ 1) ∧ ¬(m < 2 → ∃ x : ℝ, x^2 - 2 * x + m = 0) :=
sorry  -- Showing the discriminant fails for m < 2.

theorem proposition3 (A B : Set) : (A ∩ B = B ↔ A ∪ B = A) :=
sorry  -- Showing the truth of the set property.

theorem proposition4 (x y : ℝ) : ((x + 3)^2 + (y - 4)^2 = 0) → ((x + 3)*(x - 4) = 0 → False) :=
sorry  -- Demonstrating the lack of necessary condition.

end proposition1_proposition2_proposition3_proposition4_l168_168938


namespace smallest_number_append_l168_168821

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168821


namespace find_angle_C_l168_168229

theorem find_angle_C
  {a b c : ℝ}
  (h1 : b * sin (C - π / 3) - c * sin B = 0) :
  C = 2 * π / 3 := 
sorry

noncomputable def area_of_triangle
  (a b c : ℝ)
  (h1 : a = 4)
  (h2 : c = 2 * sqrt 7)
  (C : ℝ)
  (h3 : C = 2 * π / 3) : 
  ℝ :=
1 / 2 * a * b * sin C

example (a b c : ℝ)
  (h1 : a = 4)
  (h2 : c = 2 * sqrt 7)
  (h3 : C = 2 * π / 3)
  (hb : b = 2)
  (hangle : h1 + h2 + h3 = 0) :
  area_of_triangle a b c = 2 * sqrt 3 :=
sorry

end find_angle_C_l168_168229


namespace find_pairs_l168_168115

theorem find_pairs (n k : ℕ) : (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) := by
  sorry

end find_pairs_l168_168115


namespace sum_of_digits_of_k_l168_168721

theorem sum_of_digits_of_k : 
  ∃ (k : ℕ), (0 < k) ∧ (k / 12) / (15 / k) = 20 ∧ (Nat.digits 10 k).sum = 6 := 
by
  sorry

end sum_of_digits_of_k_l168_168721


namespace latest_start_time_is_correct_l168_168250

noncomputable def doughComingToRoomTemp : ℕ := 1  -- 1 hour
noncomputable def shapingDough : ℕ := 15         -- 15 minutes
noncomputable def proofingDough : ℕ := 2         -- 2 hours
noncomputable def bakingBread : ℕ := 30          -- 30 minutes
noncomputable def coolingBread : ℕ := 15         -- 15 minutes
noncomputable def bakeryOpeningTime : ℕ := 6     -- 6:00 am

-- Total preparation time in minutes
noncomputable def totalPreparationTimeInMinutes : ℕ :=
  (doughComingToRoomTemp * 60) + shapingDough + (proofingDough * 60) + bakingBread + coolingBread

-- Total preparation time in hours
noncomputable def totalPreparationTimeInHours : ℕ :=
  totalPreparationTimeInMinutes / 60

-- Latest time the baker can start working
noncomputable def latestTimeBakerCanStart : ℕ :=
  if (bakeryOpeningTime - totalPreparationTimeInHours) < 0 then 24 + (bakeryOpeningTime - totalPreparationTimeInHours)
  else bakeryOpeningTime - totalPreparationTimeInHours

theorem latest_start_time_is_correct : latestTimeBakerCanStart = 2 := by
  sorry

end latest_start_time_is_correct_l168_168250


namespace jackson_total_souvenirs_l168_168635

theorem jackson_total_souvenirs 
  (num_hermit_crabs : ℕ)
  (spiral_shells_per_hermit_crab : ℕ) 
  (starfish_per_spiral_shell : ℕ) :
  (num_hermit_crabs = 45) → 
  (spiral_shells_per_hermit_crab = 3) → 
  (starfish_per_spiral_shell = 2) →
  (45 + 45 * 3 + 45 * 3 * 2 = 450) :=
by
  intros h0 h1 h2
  rw [h0, h1, h2]
  rfl

end jackson_total_souvenirs_l168_168635


namespace smallest_number_append_l168_168817

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168817


namespace mean_age_euler_family_l168_168305

theorem mean_age_euler_family : 
  let ages := [5, 8, 8, 8, 12, 12] in
  let num_children := 6 in
  (list.sum ages) / num_children = 53 / 6 :=
by
  sorry

end mean_age_euler_family_l168_168305


namespace calc_value_l168_168072

theorem calc_value : 500 * 3986 * 0.3986 * 20 = 3986^2 := by
  -- Rearranging and combining terms
  have h1 : 500 * 20 = 10000 := by linarith
  -- Expressing 0.3986 in terms of 3986
  have h2 : 0.3986 = 3986 * 10^(-4) := by norm_num
  -- Substituting and simplifying
  calc
    500 * 3986 * 0.3986 * 20
        = 10000 * 3986 * (3986 * 10^(-4)) : by rw [h1, h2]
    ... = (10000 * 10^(-4)) * (3986 * 3986) : by ring
    ... = 1 * 3986^2 : by norm_num
    ... = 3986^2 : by norm_num

end calc_value_l168_168072


namespace no_winning_strategy_for_A_l168_168912

/-
We define the problem including the operations of painting by two players and the winning condition.
-/

inductive Player
| A
| B

def paint_turn (remaining_edges : Nat) (painted_edges : Nat) : Nat :=
  if remaining_edges >= 3 then remaining_edges - 3 else remaining_edges 

def game_result (player_A_edges : Nat) (player_B_edges : Nat) : Bool :=
  if player_A_edges = 4 * 3 then false else true

theorem no_winning_strategy_for_A : ∀ (remaining_edges : Nat) (player_A_edges : Nat) (player_B_edges : Nat),
  player_A_edges + player_B_edges = 12 → 
  paint_turn remaining_edges player_A_edges = 6 → 
  player_A_edges < 4 →
  game_result player_A_edges player_B_edges = true := 
by {
  intros,
  -- actual proof steps are omitted
  sorry
}

end no_winning_strategy_for_A_l168_168912


namespace emberly_total_miles_l168_168107

noncomputable def totalMilesWalkedInMarch : ℕ :=
  let daysInMarch := 31
  let daysNotWalked := 4
  let milesPerDay := 4
  (daysInMarch - daysNotWalked) * milesPerDay

theorem emberly_total_miles : totalMilesWalkedInMarch = 108 :=
by
  sorry

end emberly_total_miles_l168_168107


namespace max_volume_prism_l168_168612

theorem max_volume_prism :
  ∃ (a b h : ℝ) (θ : ℝ), 
    let A := a * h,
        B := b * h,
        C := (1/2) * a * b * Real.sin θ,
        V := (1/2) * a * b * h * Real.sin θ,
    A + B + C = 36 ∧ θ = Real.pi / 2 ∧ V = 27 :=
begin
  sorry
end

end max_volume_prism_l168_168612


namespace smallest_sum_of_consecutive_integers_gt_420_l168_168872

theorem smallest_sum_of_consecutive_integers_gt_420 : 
  ∃ n : ℕ, (n * (n + 1) > 420) ∧ (n + (n + 1) = 43) := sorry

end smallest_sum_of_consecutive_integers_gt_420_l168_168872


namespace append_digits_divisible_by_all_less_than_10_l168_168800

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168800


namespace c_minus_3d_eq_2_plus_6i_l168_168770

def c : ℂ := 5 + 3 * Complex.i
def d : ℂ := 1 - Complex.i

theorem c_minus_3d_eq_2_plus_6i : c - 3 * d = 2 + 6 * Complex.i :=
by
  sorry

end c_minus_3d_eq_2_plus_6i_l168_168770


namespace find_length_of_side_y_l168_168126

-- Define the given conditions
def triangle_AOC (OA OC y : ℝ) : Prop :=
  OA = 4 ∧ OC = 6 ∧ 
  (∃ (θ : ℝ ), (0 ≤ θ ∧ θ ≤ π))

def triangle_BOD (OB OD BD : ℝ) : Prop :=
  OB = 7 ∧ OD = 4 ∧ BD = 8 ∧ 
  (∃ (θ : ℝ ), (0 ≤ θ ∧ θ ≤ π ∧ θ = ∠ (OB, OD)))

noncomputable def length_of_side_y {OA OC OB OD BD y : ℝ} (h1 : triangle_AOC OA OC y) (h2 : triangle_BOD OB OD BD) : ℝ :=
  y

-- Proof goal
theorem find_length_of_side_y {OA OC OB OD BD y : ℝ} (h1 : triangle_AOC OA OC y) (h2 : triangle_BOD OB OD BD) :
  length_of_side_y h1 h2 = (2 * Real.sqrt 889) / 7 :=
by
  sorry

end find_length_of_side_y_l168_168126


namespace smallest_number_of_digits_to_append_l168_168805

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168805


namespace original_price_l168_168722

variable (q r : ℝ)

theorem original_price (x : ℝ) (h : x * (1 + q / 100) * (1 - r / 100) = 1) :
  x = 1 / ((1 + q / 100) * (1 - r / 100)) :=
sorry

end original_price_l168_168722


namespace silvia_saves_50_l168_168691

variables
  (price : ℕ := 1000) -- Suggested retail price of the guitar
  (gc_discount : ℕ := 15) -- Guitar Center discount in percentage
  (gc_shipping : ℕ := 100) -- Guitar Center shipping fee
  (sw_discount : ℕ := 10) -- Sweetwater discount in percentage
  (sw_shipping : ℕ := 0) -- Sweetwater shipping fee

def guitar_price_after_discount (price : ℕ) (discount : ℕ) : ℕ := 
  price - (price * discount / 100)

def total_cost (price : ℕ) (discount : ℕ) (shipping : ℕ) : ℕ :=
  (guitar_price_after_discount price discount) + shipping

def savings (price : ℕ) (gc_discount : ℕ) (gc_shipping : ℕ) (sw_discount : ℕ) (sw_shipping : ℕ) : ℕ :=
  total_cost price gc_discount gc_shipping - total_cost price sw_discount sw_shipping

theorem silvia_saves_50 (price : ℕ) (gc_discount : ℕ) (gc_shipping : ℕ) (sw_discount : ℕ) (sw_shipping : ℕ) :
  savings price gc_discount gc_shipping sw_discount sw_shipping = 50 := by
begin
  sorry
end

end silvia_saves_50_l168_168691


namespace least_positive_multiple_of_primes_l168_168005

theorem least_positive_multiple_of_primes :
  11 * 13 * 17 * 19 = 46189 :=
by
  sorry

end least_positive_multiple_of_primes_l168_168005


namespace average_weight_correct_l168_168372

-- Define the weights of the 7 students
def student_weights : List ℝ := [35.1, 41.3, 38.6, 40.2, 39.0, 43.7, 38.4]

-- Define the condition of being greater than or equal to 39 kg
def meets_criterion (w : ℝ) : Prop := w ≥ 39.0

-- Filter the weights that meet the criterion
def qualifying_weights : List ℝ := student_weights.filter meets_criterion

-- Calculate the sum of the qualifying weights
def sum_qualifying_weights : ℝ := List.sum qualifying_weights

-- Calculate the number of qualifying weights
def count_qualifying_weights : ℕ := List.length qualifying_weights

-- Calculate the average weight of the qualifying students
noncomputable def average_weight : ℝ := sum_qualifying_weights / count_qualifying_weights

-- The lean statement that asserts the average weight is 41.05
theorem average_weight_correct : average_weight = 41.05 := by
  sorry

end average_weight_correct_l168_168372


namespace y_value_for_equations_l168_168468

theorem y_value_for_equations (x y : ℝ) (h1 : x^2 + y^2 = 25) (h2 : x^2 + y = 10) :
  y = (1 - Real.sqrt 61) / 2 := by
  sorry

end y_value_for_equations_l168_168468


namespace jackson_total_souvenirs_l168_168636

theorem jackson_total_souvenirs 
  (num_hermit_crabs : ℕ)
  (spiral_shells_per_hermit_crab : ℕ) 
  (starfish_per_spiral_shell : ℕ) :
  (num_hermit_crabs = 45) → 
  (spiral_shells_per_hermit_crab = 3) → 
  (starfish_per_spiral_shell = 2) →
  (45 + 45 * 3 + 45 * 3 * 2 = 450) :=
by
  intros h0 h1 h2
  rw [h0, h1, h2]
  rfl

end jackson_total_souvenirs_l168_168636


namespace solution_set_of_x_squared_geq_four_l168_168979

theorem solution_set_of_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
sorry

end solution_set_of_x_squared_geq_four_l168_168979


namespace maximize_area_triangle_PAB_l168_168320

def parabola (x : ℝ) : ℝ := 4 - x^2
def line (x : ℝ) : ℝ := 4 * x
def slope_tangent_at (x : ℝ) : ℝ := -2 * x
def tangent_parallel_to_line (x : ℝ) : Prop := slope_tangent_at x = 4

noncomputable def point_P_coordinates : ℝ × ℝ :=
  let x := -2 in
  (x, parabola x)

theorem maximize_area_triangle_PAB :
  ∃ (P : ℝ × ℝ), P = (-2, 0) ∧
    (∃ A B : ℝ × ℝ,
      A ≠ B ∧
      A.2 = 4 - A.1^2 ∧
      B.2 = 4 - B.1^2 ∧
      A.2 = 4 * A.1 ∧ 
      B.2 = 4 * B.1 ∧
      point_P_coordinates = P
    ) :=
by
  let P := point_P_coordinates
  use P
  use (-1, 3)
  use (0, 4)
  split
  { -- P = (-2, 0)
    sorry
  }
  split
  { -- A ≠ B
    sorry
  }
  split
  { -- A on parabola
    sorry
  }
  split
  { -- B on parabola
    sorry
  }
  split
  { -- A on line
    sorry
  }
  { -- B on line
    sorry
  }

end maximize_area_triangle_PAB_l168_168320


namespace part_I_min_value_part_II_nonexistence_l168_168525

theorem part_I_min_value (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : a^2 + 16 * b^2 ≥ 32 :=
by
  sorry

theorem part_II_nonexistence (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 6 :=
by
  sorry

end part_I_min_value_part_II_nonexistence_l168_168525


namespace triangle_property_l168_168308

open EuclideanGeometry

variables {A B C U V W T : Point} -- Points in Euclidean plane
variables {circumcircle : Circle} -- Circumcircle of the triangle ABC

-- Given: A, B, and C form a triangle such that angle at A is the smallest angle.
-- U lies on the arc BC that does not contain A.
-- V and W are points on line AU such that V and W are on the perpendicular bisectors of AB and AC respectively.
-- T is the intersection of lines BV and CW.

theorem triangle_property 
  (hABC : Triangle A B C) 
  (hsmallest_angle : ∀ β γ, β = ∠B ∧ γ = ∠C → ∠A < β ∧ ∠A < γ)
  (hU_arc : U ∈ int_arc B C circumcircle ∧ ¬A ∈ int_arc B C circumcircle)
  (hV_W_bisectors : on_line V (perpendicular_bisector A B) ∧ on_line W (perpendicular_bisector A C) ∧ V, W ∈ line A U)
  (hT_intersection : intersection_point (line B V) (line C W) = T) :
  dist A U = dist T B + dist T C := sorry

end triangle_property_l168_168308


namespace triangle_AEF_equilateral_l168_168168

variables {A B C D E F : Type*}
variables [AffineSpace ℝ (A B C D E F)]

-- Assume we have a rectangle ABCD
variable (h_rect : ∀ (P Q R S : A B C D), parallelogram P Q R S)

-- Assume triangles BEC and CFD are equilateral and each shares only one side with the rectangle ABCD
variable (h_eq_BEC : ∀ (B E C : A B C D), equilateral_triangle B E C ∧ side_shares_with_rectangle B E C)
variable (h_eq_CFD : ∀ (C F D : A B C D), equilateral_triangle C F D ∧ side_shares_with_rectangle C F D)

-- Goal: Prove that triangle AEF is equilateral
theorem triangle_AEF_equilateral (h_rect : parallelogram A B C D) 
    (h_eq_BEC : equilateral_triangle B E C ∧ side_shares_with_rectangle B E C)
    (h_eq_CFD : equilateral_triangle C F D ∧ side_shares_with_rectangle C F D)
    : equilateral_triangle A E F :=
sorry

end triangle_AEF_equilateral_l168_168168


namespace circle_properties_l168_168400

theorem circle_properties (D r C A : ℝ) (h1 : D = 15)
  (h2 : r = 7.5)
  (h3 : C = 15 * Real.pi)
  (h4 : A = 56.25 * Real.pi) :
  (9 ^ 2 + 12 ^ 2 = D ^ 2) ∧ (D = 2 * r) ∧ (C = Real.pi * D) ∧ (A = Real.pi * r ^ 2) :=
by
  sorry

end circle_properties_l168_168400


namespace log_sum_geometric_sequence_l168_168733

theorem log_sum_geometric_sequence (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ m n, a (m + n) = a m * a n / a 0) (h_extreme : a 5 = 4 ∧ a 6 = 2) :
  ∑ i in finset.range 10, real.logb 2 (a (i + 1)) = 15 :=
by
  have h_mult : a 5 * a 6 = 8 := by sorry
  have h_product : ∏ i in finset.range 10, a (i + 1) = (a 5 * a 6) ^ 5 := by sorry
  have h_log_prod : real.logb 2 ((a 5 * a 6) ^ 5) = 15 := by sorry
  rw [finset.prod_log_of_pos h_pos, h_product, h_log_prod]
  sorry

end log_sum_geometric_sequence_l168_168733


namespace division_by_fraction_l168_168110

theorem division_by_fraction :
  5 / (8 / 13) = 65 / 8 :=
sorry

end division_by_fraction_l168_168110


namespace existence_of_alpha_beta_M_l168_168326

def c : ℕ → ℤ 
| 0       := 1
| 1       := 0
| (n + 2) := c (n + 1) + c n

def S : set (ℕ × ℕ) :=
  {p | ∃ J : finset ℕ, (∀ j ∈ J, 0 < j) ∧ p.1 = J.sum (λ j, c j) ∧ p.2 = J.sum (λ j, c (j - 1))}

theorem existence_of_alpha_beta_M :
  ∃ (α β M : ℝ) (m : ℝ), ∀ (x y : ℕ), (m < α * x + β * y ∧ α * x + β * y < M) ↔ (x, y) ∈ S :=
sorry

end existence_of_alpha_beta_M_l168_168326


namespace smallest_number_append_l168_168815

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168815


namespace trig_identity_solutions_l168_168025

open Real

theorem trig_identity_solutions (x : ℝ) (k n : ℤ) :
  (4 * sin x * cos (π / 2 - x) + 4 * sin (π + x) * cos x + 2 * sin (3 * π / 2 - x) * cos (π + x) = 1) ↔ 
  (∃ k : ℤ, x = arctan (1 / 3) + π * k) ∨ (∃ n : ℤ, x = π / 4 + π * n) := 
sorry

end trig_identity_solutions_l168_168025


namespace sum_less_than_four_l168_168656

theorem sum_less_than_four (n : ℕ) (h : 2 ≤ n) : 
  ∑ k in finset.range (n - 1), (n : ℝ) / (n - k) * (1 / 2)^(k - 1) < 4 := 
sorry

end sum_less_than_four_l168_168656


namespace behavior_of_g_as_x_tends_to_infinity_l168_168951

def g (x : ℝ) : ℝ := -3 * x^3 - 2 * x^2 + x + 10

theorem behavior_of_g_as_x_tends_to_infinity :
  (filter.tendsto g filter.at_top filter.at_bot) ∧ (filter.tendsto g filter.at_bot filter.at_top) :=
  by
    sorry

end behavior_of_g_as_x_tends_to_infinity_l168_168951


namespace find_three_digit_number_l168_168508

def digits_to_num (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem find_three_digit_number (a b c : ℕ) (h1 : 8 * a + 5 * b + c = 100) (h2 : a + b + c = 20) :
  digits_to_num a b c = 866 :=
by 
  sorry

end find_three_digit_number_l168_168508


namespace builder_used_total_bolts_and_nuts_for_project_l168_168153

variable (boxes_bolts : ℕ) (bolts_per_box : ℕ) (boxes_nuts : ℕ) (nuts_per_box : ℕ)
variable (leftover_bolts : ℕ) (leftover_nuts : ℕ)

-- Conditions in a)
def total_bolts_purchased : ℕ := boxes_bolts * bolts_per_box
def total_nuts_purchased : ℕ := boxes_nuts * nuts_per_box

def bolts_used : ℕ := total_bolts_purchased - leftover_bolts
def nuts_used : ℕ := total_nuts_purchased - leftover_nuts

def total_used : ℕ := bolts_used + nuts_used

-- Question in c) as theorem
theorem builder_used_total_bolts_and_nuts_for_project :
  boxes_bolts = 7 →
  bolts_per_box = 11 →
  boxes_nuts = 3 →
  nuts_per_box = 15 →
  leftover_bolts = 3 →
  leftover_nuts = 6 →
  total_used = 113 :=
by sorry

end builder_used_total_bolts_and_nuts_for_project_l168_168153


namespace two_colorable_graph_majority_l168_168649

-- Definition of a graph
structure Graph (V : Type) :=
  (E : V → V → Prop) -- Edge relation

-- Two-colorable property definition
def two_colorable (G : Graph V) : Prop :=
  ∃ (color : V → bool), ∀ v : V, ∃ opposites : set V, (G.E v = opposites) ∧
    (∀ u ∈ opposites, color u ≠ color v) ∧ (opposites.size ≥ (neighbors.size / 2))

theorem two_colorable_graph_majority (G : Graph V) : 
  two_colorable G :=
by 
  sorry

end two_colorable_graph_majority_l168_168649


namespace number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l168_168764

/- Definitions for each number's expression using five eights -/
def number1 : Int := (8 / 8) ^ (8 / 8) * (8 / 8)
def number2 : Int := 8 / 8 + 8 / 8
def number3 : Int := (8 + 8 + 8) / 8
def number4 : Int := 8 / 8 + 8 / 8 + 8 / 8 + 8 / 8
def number5 : Int := (8 * 8 - 8) / 8 + 8 / 8

/- Theorem statements to be proven -/
theorem number1_is_1 : number1 = 1 := by
  sorry

theorem number2_is_2 : number2 = 2 := by
  sorry

theorem number3_is_3 : number3 = 3 := by
  sorry

theorem number4_is_4 : number4 = 4 := by
  sorry

theorem number5_is_5 : number5 = 5 := by
  sorry

end number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l168_168764


namespace locus_of_P_is_ellipse_l168_168949

-- Define points A, B, C, D, and P
variable {A B C D P : Type}

-- Define the various lines and constructions
variable (AB e AD BD AP : A → B → C)

-- Define the perpendicular constructions and intersection
variable (perp_AD_P : ∀ {A D}, A ≠ D → AD ⊥ A ⟶ P)

-- Define the coordinates of the points in the space
variable (coords_A coords_B coords_C coords_D coords_P : ℝ × ℝ)

-- Define the conditions and hypotheses
variable (ext_AB_C : ∃ (x : ℝ), x < 0) -- C is on the extension of AB beyond A
variable (line_e_perp_AB : e ⊥ AB) -- Perpendicular line e to AB at C
variable (D_on_e : ∃ d, D ∈ e) -- D is arbitrary on e
variable (perp_A_AD : ∀ {x : ℝ}, ∃ A D, ⊥ AD) -- Perpendicular from A to AD
variable (intersection_P_BD : P ∈ BD) -- P is the intersection of the perpendicular with BD

-- Define the goal: The locus of points P forms an ellipse excluding point B
theorem locus_of_P_is_ellipse :
  ∀ (A B C D P : Type) (AB e AD BD AP : A → B → C)
    (coords_A coords_B coords_C coords_D coords_P : ℝ × ℝ),
    ext_AB_C → line_e_perp_AB → D_on_e → perp_A_AD → intersection_P_BD →
    locus P ≠ B :=
sorry

end locus_of_P_is_ellipse_l168_168949


namespace maximum_delta_value_l168_168574

-- Definition of the sequence a 
def a (n : ℕ) : ℕ := 1 + n^3

-- Definition of δ_n as the gcd of consecutive terms in the sequence a
def delta (n : ℕ) : ℕ := Nat.gcd (a (n + 1)) (a n)

-- Main theorem statement
theorem maximum_delta_value : ∃ n, delta n = 7 :=
by
  -- Insert the proof later
  sorry

end maximum_delta_value_l168_168574


namespace largest_diagonal_BD_l168_168260

theorem largest_diagonal_BD (a b c d : ℕ) (BD : ℝ) (area : ℝ) 
  (h₁ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h₂ : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10)
  (h₃ : a * d = b * c)
  (h₄ : area = 30)
  (h_cyclic : cyclic_quad a b c d) :
  BD = real.sqrt 54 := 
sorry

end largest_diagonal_BD_l168_168260


namespace sum_of_two_digit_divisors_l168_168268

theorem sum_of_two_digit_divisors (d : ℕ) (h₁ : d > 0) (h₂ : 144 % d = 9) :
  d = 15 ∨ d = 27 ∨ d = 45 → ∑ x in {15, 27, 45}, x = 87 :=
by 
  sorry

end sum_of_two_digit_divisors_l168_168268


namespace cone_height_l168_168907

theorem cone_height (r_sphere : ℝ) (r_cone : ℝ) (waste_percentage : ℝ) 
  (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) : 
  r_sphere = 9 → r_cone = 9 → waste_percentage = 0.75 → 
  V_sphere = (4 / 3) * Real.pi * r_sphere^3 → 
  V_cone = (1 / 3) * Real.pi * r_cone^2 * h → 
  V_cone = waste_percentage * V_sphere → 
  h = 27 :=
by
  intros r_sphere_eq r_cone_eq waste_eq V_sphere_eq V_cone_eq V_cone_waste_eq
  sorry

end cone_height_l168_168907


namespace min_value_expression_for_real_numbers_l168_168156

def min_expression_value (x : ℝ) (hx : x > 4) : Prop :=
  ∀ (y : ℝ), y = (x - 4) → y > 0 → (x + 18) / (y) = 2 * Real.sqrt 22

theorem min_value_expression_for_real_numbers (x : ℝ) (hx : x > 4) :
  ∃ (y : ℝ), y > 0 ∧ y = Real.sqrt (x - 4) ∧ ((x + 18) / y) = 2 * Real.sqrt 22 :=
begin
  sorry
end

end min_value_expression_for_real_numbers_l168_168156


namespace purely_imaginary_complex_number_l168_168526

theorem purely_imaginary_complex_number (a : ℝ) (i : ℂ)
  (h₁ : i * i = -1)
  (h₂ : ∃ z : ℂ, z = (a + i) / (1 - i) ∧ z.im ≠ 0 ∧ z.re = 0) :
  a = 1 :=
sorry

end purely_imaginary_complex_number_l168_168526


namespace eq_infinite_solutions_function_satisfies_identity_l168_168288

-- First Part: Proving the equation has infinitely many positive integer solutions
theorem eq_infinite_solutions : ∃ (x y z t : ℕ), ∀ n : ℕ, x^2 + 2 * y^2 = z^2 + 2 * t^2 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 := 
sorry

-- Second Part: Finding and proving the function f
def f (n : ℕ) : ℕ := n

theorem function_satisfies_identity (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (f n^2 + 2 * f m^2) = n^2 + 2 * m^2) : ∀ k : ℕ, f k = k :=
sorry

end eq_infinite_solutions_function_satisfies_identity_l168_168288


namespace find_sale_in_second_month_l168_168914

def sale_in_second_month (sale1 sale3 sale4 sale5 sale6 target_average : ℕ) (S : ℕ) : Prop :=
  sale1 + S + sale3 + sale4 + sale5 + sale6 = target_average * 6

theorem find_sale_in_second_month :
  sale_in_second_month 5420 6200 6350 6500 7070 6200 5660 :=
by
  sorry

end find_sale_in_second_month_l168_168914


namespace Addison_High_School_college_attendance_l168_168935

theorem Addison_High_School_college_attendance:
  ∀ (G B : ℕ) (pG_not_college p_total_college : ℚ),
  G = 200 →
  B = 160 →
  pG_not_college = 0.4 →
  p_total_college = 0.6667 →
  ((B * 100) / 160) = 75 := 
by
  intro G B pG_not_college p_total_college G_eq B_eq pG_not_college_eq p_total_college_eq
  -- skipped proof
  sorry

end Addison_High_School_college_attendance_l168_168935


namespace dave_final_tickets_l168_168394

-- Define the initial number of tickets and operations
def initial_tickets : ℕ := 25
def tickets_spent_on_beanie : ℕ := 22
def tickets_won_after : ℕ := 15

-- Define the final number of tickets function
def final_tickets (initial : ℕ) (spent : ℕ) (won : ℕ) : ℕ :=
  initial - spent + won

-- Theorem stating that Dave would end up with 18 tickets given the conditions
theorem dave_final_tickets : final_tickets initial_tickets tickets_spent_on_beanie tickets_won_after = 18 :=
by
  -- Proof to be filled in
  sorry

end dave_final_tickets_l168_168394


namespace find_roots_l168_168121

def poly (x : ℝ) : ℝ := x^3 + 2 * x^2 - 5 * x - 6

theorem find_roots :
  polynomial.eval (-1) (polynomial.C ∘ poly) = 0 ∧
  polynomial.eval (2) (polynomial.C ∘ poly) = 0 ∧
  polynomial.eval (-3) (polynomial.C ∘ poly) = 0 ∧
  ∀ x : ℝ, (polynomial.eval x (polynomial.C ∘ poly) = 0) → (x = -1 ∨ x = 2 ∨ x = -3) :=
by
  sorry

end find_roots_l168_168121


namespace pure_imaginary_condition_l168_168225

theorem pure_imaginary_condition (m : ℝ) (z : ℂ) (h1 : z = (m^2 - m - 2 : ℝ) + (m + 1 : ℝ) * complex.I) (h2 : ∃ a : ℝ, z = a * complex.I ∧ a ≠ 0) : m = 2 :=
sorry

end pure_imaginary_condition_l168_168225


namespace grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l168_168011

-- Definition of the conditions
def grandfather_age (gm_age dad_age : ℕ) := gm_age = 2 * dad_age
def dad_age_eight_times_xiaoming (dad_age xm_age : ℕ) := dad_age = 8 * xm_age
def grandfather_age_61 (gm_age : ℕ) := gm_age = 61
def twenty_times_xiaoming (gm_age xm_age : ℕ) := gm_age = 20 * xm_age

-- Question 1: Proof that Grandpa is 57 years older than Xiaoming 
theorem grandfather_older_than_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : grandfather_age gm_age dad_age) (h2 : dad_age_eight_times_xiaoming dad_age xm_age)
  (h3 : grandfather_age_61 gm_age)
  : gm_age - xm_age = 57 := 
sorry

-- Question 2: Proof that Dad is 31 years old when Grandpa's age is twenty times Xiaoming's age
theorem dad_age_when_twenty_times_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : twenty_times_xiaoming gm_age xm_age)
  (hm : grandfather_age gm_age dad_age)
  : dad_age = 31 :=
sorry

end grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l168_168011


namespace petya_mistake_no_double_l168_168024

theorem petya_mistake_no_double (x y z : ℕ) (h₁ : x + y + z = 100) (h₂ : 201.2 * x + 2.012 * y + 20.12 * z = 4024) :
  False := by
  sorry

end petya_mistake_no_double_l168_168024


namespace geometric_sequence_a10_a11_l168_168664

noncomputable def a (n : ℕ) : ℝ := sorry  -- define the geometric sequence {a_n}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q^m

variables (a : ℕ → ℝ) (q : ℝ)

-- Conditions given in the problem
axiom h1 : a 1 + a 5 = 5
axiom h2 : a 4 + a 5 = 15
axiom geom_seq : is_geometric_sequence a q

theorem geometric_sequence_a10_a11 : a 10 + a 11 = 135 :=
by {
  sorry
}

end geometric_sequence_a10_a11_l168_168664


namespace pentomino_symmetry_count_l168_168212

def is_pentomino (shape : Type) : Prop :=
  -- Define the property of being a pentomino as composed of five squares edge to edge
  sorry

def has_reflectional_symmetry (shape : Type) : Prop :=
  -- Define the property of having at least one line of reflectional symmetry
  sorry

def has_rotational_symmetry_of_order_2 (shape : Type) : Prop :=
  -- Define the property of having rotational symmetry of order 2 (180 degrees rotation results in the same shape)
  sorry

noncomputable def count_valid_pentominoes : Nat :=
  -- Assume that we have a list of 18 pentominoes
  -- Count the number of pentominoes that meet both criteria
  sorry

theorem pentomino_symmetry_count :
  count_valid_pentominoes = 4 :=
sorry

end pentomino_symmetry_count_l168_168212


namespace least_number_to_make_divisible_l168_168891

def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem least_number_to_make_divisible :
  ∃ x : ℕ, x + 2496 = 2500 ∧ divisible_by_5 (x + 2496) := by
  exists 4
  constructor
  · rfl
  · unfold divisible_by_5
    norm_num

end least_number_to_make_divisible_l168_168891


namespace ratio_areas_l168_168410

theorem ratio_areas (H : ℝ) (L : ℝ) (r : ℝ) (A_rectangle : ℝ) (A_circle : ℝ) :
  H = 45 ∧ (L / H = 4 / 3) ∧ r = H / 2 ∧ A_rectangle = L * H ∧ A_circle = π * r^2 →
  (A_rectangle / A_circle = 17 / π) :=
by
  sorry

end ratio_areas_l168_168410


namespace angle_measure_l168_168708

theorem angle_measure (x : ℝ) : 
  (3 * x - 8 = 90 - x) → (x = 24.5) :=
by
  intro h₁
  have h₂ : (4 * x - 8 = 90) := by linarith
  have h₃ : (4 * x = 98) := by linarith
  have h₄ : (x = 24.5) := by { field_simp, linarith }
  exact h₄

end angle_measure_l168_168708


namespace average_age_of_9_l168_168309

theorem average_age_of_9 : 
  ∀ (avg_20 avg_5 age_15 : ℝ),
  avg_20 = 15 →
  avg_5 = 14 →
  age_15 = 86 →
  (9 * (69/9)) = 7.67 :=
by
  intros avg_20 avg_5 age_15 avg_20_val avg_5_val age_15_val
  -- The proof is skipped
  sorry

end average_age_of_9_l168_168309


namespace bug_visits_tiles_l168_168424

theorem bug_visits_tiles (width length : ℕ) (h_w : width = 15) (h_l : length = 25) :
  width + length - Nat.gcd width length = 35 :=
by
  rw [h_w, h_l]
  rfl

end bug_visits_tiles_l168_168424


namespace find_number_l168_168920

-- Define a constant to represent the number
def c : ℝ := 1002 / 20.04

-- Define the main theorem
theorem find_number (x : ℝ) (h : x - c = 2984) : x = 3034 := by
  -- The proof will be placed here
  sorry

end find_number_l168_168920


namespace smallest_digits_to_append_l168_168823

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168823


namespace find_k_l168_168224

theorem find_k (k : ℝ) : (∃ x : ℝ, k * x^2 - 9 * x + 8 = 0 ∧ x = 1) → k = 1 :=
sorry

end find_k_l168_168224


namespace initial_children_on_bus_l168_168341

-- Definitions based on conditions
variable (x : ℕ) -- number of children who got off the bus
variable (y : ℕ) -- initial number of children on the bus
variable (after_exchange : ℕ := 30) -- number of children on the bus after exchange
variable (got_on : ℕ := 82) -- number of children who got on the bus
variable (extra_on : ℕ := 2) -- extra children who got on compared to got off

-- Problem translated to Lean 4 statement
theorem initial_children_on_bus (h : got_on = x + extra_on) (hx : y + got_on - x = after_exchange) : y = 28 :=
by
  sorry

end initial_children_on_bus_l168_168341


namespace darnell_texts_l168_168474

theorem darnell_texts (T : ℕ) (unlimited_plan_cost alternative_text_cost alternative_call_cost : ℕ) 
    (call_minutes : ℕ) (cost_difference : ℕ) :
    unlimited_plan_cost = 12 →
    alternative_text_cost = 1 →
    alternative_call_cost = 3 →
    call_minutes = 60 →
    cost_difference = 1 →
    (alternative_text_cost * T / 30 + alternative_call_cost * call_minutes / 20) = 
      unlimited_plan_cost - cost_difference →
    T = 60 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end darnell_texts_l168_168474


namespace count_special_divisors_of_180_l168_168213

theorem count_special_divisors_of_180 :
  let n := 180 in
  let factored := n = 2^2 * 3^2 * 5 in
  let divisors := {d : ℕ | d ∣ n ∧ ¬ (3 ∣ d) ∧ ¬ (5 ∣ d)} in
  factored → finset.card (finset.filter (λ d, d ∈ divisors) (finset.range (n + 1))) = 3 :=
by
  sorry

end count_special_divisors_of_180_l168_168213


namespace sum_of_ages_l168_168026

variable (S F : ℕ)

-- Conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F + 6 = 2 * (S + 6)

-- Theorem Statement
theorem sum_of_ages (h1 : condition1 S F) (h2 : condition2 S F) : S + 6 + (F + 6) = 36 := by
  sorry

end sum_of_ages_l168_168026


namespace no_partition_with_equal_products_l168_168968

theorem no_partition_with_equal_products (n : ℕ) (hn : 0 < n) :
  ¬∃ (A B : Finset ℕ), 
    A ≠ ∅ ∧ B ≠ ∅ ∧
    (A ∩ B = ∅) ∧ 
    (A ∪ B = Finset.range 6).map (Finset.singleton n + Finset.singleton (λ x, n + x)) ∧
    (A.prod id = B.prod id) :=
by
  sorry

end no_partition_with_equal_products_l168_168968


namespace number_divisible_l168_168787

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168787


namespace product_of_divisor_and_dividend_l168_168185

theorem product_of_divisor_and_dividend (d D : ℕ) (q : ℕ := 6) (r : ℕ := 3) 
  (h₁ : D = d + 78) 
  (h₂ : D = d * q + r) : 
  D * d = 1395 :=
by 
  sorry

end product_of_divisor_and_dividend_l168_168185


namespace intersection_A_B_l168_168576

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 3^x}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2^(-x)}

theorem intersection_A_B :
  A ∩ B = {p | p = (0, 1)} :=
by
  sorry

end intersection_A_B_l168_168576


namespace pentagon_perimeter_l168_168419

/-- A pentagon is formed by joining the points (0,0), (1,3), (3,3), (4,0), and (2,-1), and back to (0,0). 
The perimeter of the pentagon can be written in the form p + q*sqrt(10) + r*sqrt(5), 
where p, q, and r are whole numbers. Prove that p + q + r = 6. -/
theorem pentagon_perimeter :
  ∃ (p q r : ℕ), 
  let P := [(0, 0), (1, 3), (3, 3), (4, 0), (2, -1), (0, 0)] in
  let d := λ (a b : ℤ × ℤ), ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2 : ℕ) in
  let distances := [d (0, 0) (1, 3), d (1, 3) (3, 3), d (3, 3) (4, 0), d (4, 0) (2, -1), d (2, -1) (0, 0)] in
  let perimeter := (distances.map (λ x => if x == 10 then 1 else if x == 5 then 2 else x)).sum in
  perimeter = p + q * (Real.sqrt 10) + r * (Real.sqrt 5) ∧ p + q + r = 6 :=
sorry

end pentagon_perimeter_l168_168419


namespace spherical_coordinates_correct_l168_168473

variables {x y z : ℝ}
variables (ρ θ φ : ℝ)

def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := Real.atan2 y x
  let φ := Real.acos (z / ρ)
  (ρ, θ, φ)

def target_answer : ℝ × ℝ × ℝ :=
  (Real.sqrt 30, Real.pi / 6, Real.acos (-Real.sqrt 30 / 10))

theorem spherical_coordinates_correct :
  rectangular_to_spherical (Real.sqrt 12) 3 (-3) = target_answer :=
by {
  -- Here, the proof should be provided, which will show that the transformation calculation is equivalent
  -- to the target spherical coordinates
  sorry
}

end spherical_coordinates_correct_l168_168473


namespace smallest_number_of_digits_to_append_l168_168810

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168810


namespace regression_estimate_l168_168984

theorem regression_estimate:
  ∀ (x : ℝ), (1.43 * x + 257 = 400) → x = 100 :=
by
  intro x
  intro h
  sorry

end regression_estimate_l168_168984


namespace club_members_with_all_uncool_relatives_l168_168741

variables (U : Type) (club : Finset U)
variables (D M S : Finset U) -- Sets of people with cool dads, cool moms, and cool siblings

noncomputable def number_of_uncool_relatives : ℕ :=
  let total := 50 in
  let numD := 25 in
  let numM := 28 in
  let numS := 10 in
  let numDM := 15 in
  let numDS := 5 in
  let numMS := 7 in
  let numDMS := 3 in
  let num_one_rel := numD + numM + numS in
  let num_two_rel := numDM + numDS + numMS - 3 * numDMS in
  let num_at_least_one_rel := numD + numM + numS - num_two_rel in
  total - num_at_least_one_rel

theorem club_members_with_all_uncool_relatives
    (h_club : club.card = 50)
    (h_D : D.card = 25)
    (h_M : M.card = 28)
    (h_S : S.card = 10)
    (h_DM : (D ∩ M).card = 15)
    (h_DS : (D ∩ S).card = 5)
    (h_MS : (M ∩ S).card = 7)
    (h_DMS : (D ∩ M ∩ S).card = 3) :
  number_of_uncool_relatives U club D M S = 11 :=
sorry

end club_members_with_all_uncool_relatives_l168_168741


namespace range_of_function_l168_168897

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem range_of_function :
  ∀ (x : ℝ), 1 ≤ x ∧ x < 5 → -4 ≤ f x ∧ f x < 5 :=
by
  intro x hx
  sorry

end range_of_function_l168_168897


namespace minimum_days_to_owe_double_l168_168255

/-- Kim borrows $100$ dollars from Sam with a simple interest rate of $10\%$ per day.
    There's a one-time borrowing fee of $10$ dollars that is added to the debt immediately.
    We need to prove that the least integer number of days after which Kim will owe 
    Sam at least twice as much as she borrowed is 9 days.
-/
theorem minimum_days_to_owe_double :
  ∀ (x : ℕ), 100 + 10 + 10 * x ≥ 200 → x ≥ 9 :=
by
  intros x h
  sorry

end minimum_days_to_owe_double_l168_168255


namespace maximum_value_A_l168_168148

theorem maximum_value_A
  (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  let A := (x - y) * Real.sqrt (x ^ 2 + y ^ 2) +
           (y - z) * Real.sqrt (y ^ 2 + z ^ 2) +
           (z - x) * Real.sqrt (z ^ 2 + x ^ 2) +
           Real.sqrt 2,
      B := (x - y) ^ 2 + (y - z) ^ 2 + (z - x) ^ 2 + 2 in
  A / B ≤ 1 / Real.sqrt 2 :=
sorry

end maximum_value_A_l168_168148


namespace no_integer_solutions_for_eq_l168_168479

theorem no_integer_solutions_for_eq : ∀ x y : ℤ, 2 ^ (2 * x) - 3 ^ (2 * y) ≠ 89 := by
  intro x y
  sorry

end no_integer_solutions_for_eq_l168_168479


namespace emberly_total_miles_l168_168106

noncomputable def totalMilesWalkedInMarch : ℕ :=
  let daysInMarch := 31
  let daysNotWalked := 4
  let milesPerDay := 4
  (daysInMarch - daysNotWalked) * milesPerDay

theorem emberly_total_miles : totalMilesWalkedInMarch = 108 :=
by
  sorry

end emberly_total_miles_l168_168106


namespace distance_inequality_l168_168272

theorem distance_inequality
  {l l' : Type*} 
  (A B C : l) 
  (ha : A = B + B)
  (hb : B = (A + C) / 2)
  (a b c : ℝ)
  (dist_A : ∀ (P : l), P = A → ∃ d : ℝ, d = a)
  (dist_B : ∀ (P : l), P = B → ∃ d : ℝ, d = b)
  (dist_C : ∀ (P : l), P = C → ∃ d : ℝ, d = c) :
  b ≤ sqrt ((a^2 + c^2) / 2) ∧ (b = sqrt ((a^2 + c^2) / 2) ↔ is_parallel l l') :=
by
  sorry

end distance_inequality_l168_168272


namespace emily_points_l168_168108

theorem emily_points (r1 r2 r3 r4 r5 m4 m5 l : ℤ)
  (h1 : r1 = 16)
  (h2 : r2 = 33)
  (h3 : r3 = 21)
  (h4 : r4 = 10)
  (h5 : r5 = 4)
  (hm4 : m4 = 2)
  (hm5 : m5 = 3)
  (hl : l = 48) :
  r1 + r2 + r3 + r4 * m4 + r5 * m5 - l = 54 := by
  sorry

end emily_points_l168_168108


namespace append_digits_divisible_by_all_less_than_10_l168_168799

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168799


namespace first_discount_percentage_l168_168909

theorem first_discount_percentage (d : ℝ) (h : d > 0) :
  (∃ x : ℝ, (0 < x) ∧ (x < 100) ∧ 0.6 * d = (d * (1 - x / 100)) * 0.8) → x = 25 :=
by
  sorry

end first_discount_percentage_l168_168909


namespace find_angle_A_l168_168246

theorem find_angle_A (A B C I : Type) [IncenterTriangle A B C I] 
  (h : ∠ BIC = 100) : ∠ A = 20 :=
sorry

end find_angle_A_l168_168246


namespace at_least_one_circumradius_le_sqrtR_l168_168189

-- Definitions of the given quantities
variables {A B C O : Point} -- Points for the triangle
variables {R R1 R2 R3 : ℝ} -- Circumradii
variables {dA dB dC : ℝ} -- Distances from the incenter O to vertices A, B, C

-- Condition: Sum of distances from incenter O to vertices of the triangle
axiom sum_distances (h1 : dA + dB + dC = 3)

-- Condition: Circumradius definitions for triangles ABC, OBC, OCA, OAB
axiom circumradii_def (hR : circumradius ABC R)
axiom circumradii_1_def (hR1 : circumradius OBC R1)
axiom circumradii_2_def (hR2 : circumradius OCA R2)
axiom circumradii_3_def (hR3 : circumradius OAB R3)

-- Theorem: At least one of R1, R2, or R3 is less than or equal to sqrt(R)
theorem at_least_one_circumradius_le_sqrtR : ∃ i ∈ {R1, R2, R3}, i ≤ sqrt R :=
by {
  sorry
}

end at_least_one_circumradius_le_sqrtR_l168_168189


namespace wuxi_GDP_scientific_notation_l168_168934

theorem wuxi_GDP_scientific_notation :
  14800 = 1.48 * 10^4 :=
sorry

end wuxi_GDP_scientific_notation_l168_168934


namespace train_cross_time_l168_168049

-- Define the given conditions
def length_of_train : ℕ := 110
def length_of_bridge : ℕ := 265
def speed_kmh : ℕ := 45

-- Convert speed to m/s
def speed_ms : ℝ := speed_kmh * 1000 / 3600

-- Define the total distance the train needs to travel
def total_distance : ℕ := length_of_train + length_of_bridge

-- Calculate the time it takes to cross the bridge
noncomputable def time_to_cross : ℝ := total_distance / speed_ms

-- State the theorem
theorem train_cross_time : time_to_cross = 30 := by sorry

end train_cross_time_l168_168049


namespace winning_rules_l168_168767

-- Define sets of cards A and B
def cards := Fin 100 → ℝ

-- Define the winner rule: A beats B if there is a rule R_i such that A's i-th card beats B's i-th card
def beats (A B : cards) (i : Fin 100) : Prop := ∀ i, A i > B i

-- The theorem statement
theorem winning_rules :
  ∃ (R : Set (cards → cards → Prop)), 
    (∀ (A B : cards), (∀ i : Fin 100, beats A B i ∨ beats B A i)) ∧
    (∀ (A B C : cards), 
      (∀ i : Fin 100, beats A B i → beats B C i → beats A C i)) ∧
    (R.card = 100) :=
sorry

end winning_rules_l168_168767


namespace trout_weight_l168_168959

theorem trout_weight :
  ∃ T : ℝ, (4 * T) + (3 * 1.5) + (5 * 2.5) = 25 ∧ T = 2 :=
begin
  use 2,
  dsimp,
  linarith,
end

end trout_weight_l168_168959


namespace trig_identity_1_trig_identity_2_l168_168289

noncomputable theory

open real

-- The first trigonometric identity
theorem trig_identity_1 (n : ℕ) (h : 2 ≤ n) :
  (∏ k in finset.range (n - 1) + 1, sin (π * ↑k / (2 * n))) = sqrt n / 2 ^ (n - 1) :=
sorry

-- The second trigonometric identity
theorem trig_identity_2 (n : ℕ) :
  (∏ k in finset.range (2 * n) | k.bodd, sin (π * ↑k / (4 * n))) = sqrt 2 / 2 ^ n :=
sorry

end trig_identity_1_trig_identity_2_l168_168289


namespace zero_of_function_l168_168735

theorem zero_of_function : ∃ x : Real, 4 * x - 2 = 0 ∧ x = 1 / 2 :=
by
  sorry

end zero_of_function_l168_168735


namespace find_zero_of_exponential_function_l168_168980

noncomputable def zero_of_exponential_function : Prop :=
  (∀ x : ℝ, exp(2 * x) - 1 = 0 ↔ x = 0)

theorem find_zero_of_exponential_function : zero_of_exponential_function := 
by
  sorry

end find_zero_of_exponential_function_l168_168980


namespace determine_b_l168_168300

noncomputable def f (x b : ℝ) : ℝ := 1 / (3 * x + b)

noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, f (f_inv x) b = x) -> b = 3 :=
by
  intro h
  sorry

end determine_b_l168_168300


namespace smallest_digits_to_append_l168_168853

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168853


namespace planA_equals_planB_at_3_l168_168406

def planA_charge_for_first_9_minutes : ℝ := 0.24
def planA_charge (X: ℝ) (minutes: ℕ) : ℝ := if minutes <= 9 then X else X + 0.06 * (minutes - 9)
def planB_charge (minutes: ℕ) : ℝ := 0.08 * minutes

theorem planA_equals_planB_at_3 : planA_charge planA_charge_for_first_9_minutes 3 = planB_charge 3 :=
by sorry

end planA_equals_planB_at_3_l168_168406


namespace solve_for_lambda_l168_168578

variable (λ : ℝ)
def vector_m := (λ + 1, 1)
def vector_n := (4, -2)

theorem solve_for_lambda
  (h : vector_m λ).2 * (vector_n).1 - (vector_m λ).1 * (vector_n).2 = 0 :
  λ = -3 :=
by
  sorry

end solve_for_lambda_l168_168578


namespace sum_arithmetic_sequence_l168_168071

theorem sum_arithmetic_sequence :
  let a : ℤ := -25
  let d : ℤ := 4
  let a_n : ℤ := 19
  let n : ℤ := (a_n - a) / d + 1
  let S : ℤ := n * (a + a_n) / 2
  S = -36 :=
by 
  let a := -25
  let d := 4
  let a_n := 19
  let n := (a_n - a) / d + 1
  let S := n * (a + a_n) / 2
  show S = -36
  sorry

end sum_arithmetic_sequence_l168_168071


namespace num_subsets_A_l168_168718

theorem num_subsets_A : (A : Set Int) = {-1, 1} → Finset.card (Finset.powerset (Finset.fromSet A.toFinset)) = 4 :=
by sorry

end num_subsets_A_l168_168718


namespace O₁O₂O₃O₄_is_rectangle_H₁H₂H₃H₄_is_congruent_to_ABCD_l168_168682

-- Definitions for the conditions
variable (A B C D : Point)
variable (O₁ O₂ O₃ O₄ H₁ H₂ H₃ H₄ : Point)
variable (circleABCD : Circle A B C D)
variable (incenter : Triangle → Point)
variable (orthocenter : Triangle → Point)

-- Conditions assumptions
axiom incenter_O₁ : O₁ = incenter (Triangle.mk A B C)
axiom incenter_O₂ : O₂ = incenter (Triangle.mk B C D)
axiom incenter_O₃ : O₃ = incenter (Triangle.mk C D A)
axiom incenter_O₄ : O₄ = incenter (Triangle.mk D A B)

axiom orthocenter_H₁ : H₁ = orthocenter (Triangle.mk A B C)
axiom orthocenter_H₂ : H₂ = orthocenter (Triangle.mk B C D)
axiom orthocenter_H₃ : H₃ = orthocenter (Triangle.mk C D A)
axiom orthocenter_H₄ : H₄ = orthocenter (Triangle.mk D A B)

theorem O₁O₂O₃O₄_is_rectangle : 
  ∀ (A B C D O₁ O₂ O₃ O₄: Point), (is_inscribed A B C D) →
    (O₁ = incenter (Triangle.mk A B C)) →
    (O₂ = incenter (Triangle.mk B C D)) →
    (O₃ = incenter (Triangle.mk C D A)) →
    (O₄ = incenter (Triangle.mk D A B)) →
    is_rectangle O₁ O₂ O₃ O₄ := 
by
  sorry

theorem H₁H₂H₃H₄_is_congruent_to_ABCD : 
  ∀ (A B C D H₁ H₂ H₃ H₄: Point), (is_inscribed A B C D) →
    (H₁ = orthocenter (Triangle.mk A B C)) →
    (H₂ = orthocenter (Triangle.mk B C D)) →
    (H₃ = orthocenter (Triangle.mk C D A)) →
    (H₄ = orthocenter (Triangle.mk D A B)) →
    is_congruent (Quadrilateral.mk H₁ H₂ H₃ H₄) (Quadrilateral.mk A B C D) :=
by
  sorry

end O₁O₂O₃O₄_is_rectangle_H₁H₂H₃H₄_is_congruent_to_ABCD_l168_168682


namespace proof_problem_l168_168546

open Nat

theorem proof_problem 
  (p : ℕ) (m : ℕ) (n : ℕ)
  (hp_prime : Prime p) 
  (hp_odd : p % 2 = 1)
  (hm_gt_one : m > 1)
  (hn_pos : n > 0)
  (h_prime_quotient : Prime (m^(p*n) - 1) / (m^n - 1)) :
  p * n ∣ ((p - 1)^n + 1) :=
sorry

end proof_problem_l168_168546


namespace find_a2017_l168_168662

noncomputable def a : ℕ → ℚ
| 0       := 0 -- We define a_0 to be 0 as it is not used
| 1       := 2/3
| (n + 2) := a n + 2^n
| (n + 4) := a n + 5*2^n

lemma sequence_upper_bound : ∀ n : ℕ, a (n + 2) - a n ≤ 2^n := 
sorry

lemma sequence_lower_bound : ∀ n : ℕ, a (n + 4) - a n ≥ 5 * 2^n := 
sorry

theorem find_a2017 : a 2017 = 2^(2017) / 3 :=
sorry

end find_a2017_l168_168662


namespace pencil_case_probability_l168_168046

/-- A student has a pencil case with 6 different ballpoint pens: 3 black, 2 red, and 1 blue.
If 2 pens are randomly selected from the case, prove that the probability of both pens being black
is 1/5 and the probability of one pen being black and one pen being blue is also 1/5. -/
theorem pencil_case_probability :
  let total_pens := 6
  let total_combinations := nat.choose 6 2
  let black_pens := 3
  let red_pens := 2
  let blue_pens := 1
  (nat.choose black_pens 2) / total_combinations = 1 / 5 ∧
  (black_pens * blue_pens) / total_combinations = 1 / 5 :=
by
  intros
  have total_combinations := nat.choose 6 2
  have black_combinations := nat.choose 3 2
  have black_blue_combinations := 3 * 1
  split
  case left =>
    calc
      (black_combinations : ℚ) / total_combinations
          = 3 / 15 : by norm_num [total_combinations, black_combinations]
      ... = 1 / 5   : by norm_num
  case right =>
    calc
      (black_blue_combinations : ℚ) / total_combinations
          = 3 / 15 : by norm_num [total_combinations, black_blue_combinations]
      ... = 1 / 5   : by norm_num

end pencil_case_probability_l168_168046


namespace B_k_maximized_at_45_l168_168965

theorem B_k_maximized_at_45 : 
  (∃ k : ℕ, k ≤ 500 ∧ (∀ n : ℕ, n ≤ 500 → (B n ≤ B k)) ∧ k = 45) :=
by
  let B : ℕ → ℝ := λ k, (nat.choose 500 k) * (0.1 ^ k)
  use 45
  split
  { linarith }
  split
  { intro n
    intro hn : n ≤ 500
    /- We need to prove that B(n) ≤ B(45) -/
    sorry }
  { rfl }

end B_k_maximized_at_45_l168_168965


namespace range_a_l168_168196

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

theorem range_a (H : ∀ x1 x2 : ℝ, 2 < x1 → 2 < x2 → (f x1 a - f x2 a) / (x1 - x2) > 0) : a ≤ 2 := 
sorry

end range_a_l168_168196


namespace smallest_digits_to_append_l168_168779

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168779


namespace geometric_sequence_condition_l168_168989

theorem geometric_sequence_condition (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) → (a * d = b * c) ∧ 
  ¬ (∀ a b c d : ℝ, a * d = b * c → ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) := 
by
  sorry

end geometric_sequence_condition_l168_168989


namespace expected_value_proof_l168_168066

variable (n : ℕ) (h_pos : n > 0)
variable (X : ℕ → ℕ)
variable (D : ℕ → ℕ)
variable h_var : D(X) = 1

-- Define the binomial distribution parameters
def p : ℚ := 5 / (n + 5 : ℚ)
def trials : ℕ := 4

-- Define the expected value for the binomial distribution
def E : ℚ := trials * p

noncomputable def proof_statement : Prop :=
  E = 2

theorem expected_value_proof (h_var : D(X) = 1) : proof_statement :=
by
  sorry

end expected_value_proof_l168_168066


namespace smallest_digits_to_append_l168_168845

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168845


namespace no_linear_term_l168_168228

theorem no_linear_term (m : ℝ) (x : ℝ) : 
  (x + m) * (x + 3) - (x * x + 3 * m) = 0 → m = -3 :=
by
  sorry

end no_linear_term_l168_168228


namespace numbers_whose_triples_plus_1_are_primes_l168_168350

def is_prime (n : ℕ) : Prop := Nat.Prime n

def in_prime_range (n : ℕ) : Prop := 
  is_prime n ∧ 70 ≤ n ∧ n ≤ 110

def transformed_by_3_and_1 (x : ℕ) : ℕ := 3 * x + 1

theorem numbers_whose_triples_plus_1_are_primes :
  { x : ℕ | in_prime_range (transformed_by_3_and_1 x) } = {24, 26, 32, 34, 36} :=
by
  sorry

end numbers_whose_triples_plus_1_are_primes_l168_168350


namespace substitutions_modulo_1000_equals_301_l168_168027

noncomputable def num_substitutions_modulo_1000 : ℕ :=
  let b : ℕ → ℕ
  | 0       => 1
  | 1       => 50
  | 2       => 2250
  | 3       => 90000
  | 4       => 3150000
  | 5       => 94500000
  | _       => 0

  (b 0 + b 1 + (b 2 % 1000)) % 1000

theorem substitutions_modulo_1000_equals_301 : num_substitutions_modulo_1000 = 301 :=
by
  unfold num_substitutions_modulo_1000
  rw [Nat.add_assoc, Nat.add_mod, Nat.add_mod, Nat.add_mod, Nat.mod_eq_of_lt (show 1 + 50 + 250 < 1000 by norm_num)]
  norm_num
  sorry  -- The detailed proof is omitted, but norm_num would solve it practically

end substitutions_modulo_1000_equals_301_l168_168027


namespace append_digits_divisible_by_all_less_than_10_l168_168795

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168795


namespace polynomial_standard_form_1_polynomial_standard_form_2_polynomial_standard_form_3_l168_168954

-- Problem 1
theorem polynomial_standard_form_1 (a b : ℝ) :
  (a - b) * (a + b) * (a^2 + ab + b^2) * (a^2 - ab + b^2) = a^6 - b^6 :=
by sorry

-- Problem 2
theorem polynomial_standard_form_2 (x : ℝ) :
  (x - 1)^3 * (x + 1)^2 * (x^2 + 1) * (x^2 + x + 1) = 
  x^9 - x^7 - ... - x^8 - x^5 + x^4 + x^3 + x^2 - 1 :=
by sorry

-- Problem 3
theorem polynomial_standard_form_3 (x : ℝ) :
  (x^4 - x^2 + 1) * (x^2 - x + 1) * (x^2 + x + 1) = x^8 + x^4 + 1 :=
by sorry

end polynomial_standard_form_1_polynomial_standard_form_2_polynomial_standard_form_3_l168_168954


namespace propositions_correct_l168_168376

noncomputable section

variables (a b c v : Vector3)

-- Proposition 1
def prop1_cond : Prop := ∀ v, ¬ LinearIndependent ℝ ![a, b, v]
def prop1 : Prop := Parallel a b

-- Proposition 2
def prop2_cond : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ Ortho a b ∧ Ortho b c
def prop2 : Prop := Parallel a c

-- Proposition 3
variables (OA OB OC OD : Vector3)
def prop3_cond : Prop := LinearIndependent ℝ ![OA, OB, OC] ∧ OD = (1/3 : ℝ) • OA + (1/3 : ℝ) • OB + (1/3 : ℝ) • OC
def prop3 : Prop := Coplanar ![OA, OB, OC, OD]

-- Proposition 4
def prop4_cond : Prop := LinearIndependent ℝ ![a + b, b + c, c + a]
def prop4 : Prop := LinearIndependent ℝ ![a, b, c]

theorem propositions_correct :
  (prop1_cond → prop1) ∧
  (prop2_cond → ¬ prop2) ∧
  (prop3_cond → prop3) ∧
  (prop4_cond → prop4) := by
  sorry

end propositions_correct_l168_168376


namespace num_pos_3_digit_multiples_of_25_not_40_l168_168584

theorem num_pos_3_digit_multiples_of_25_not_40 : 
  let three_digit_numbers := {n : ℤ | 100 ≤ n ∧ n ≤ 999}
  let multiples_of_25 := {n : ℤ | ∃ k : ℤ, n = 25 * k}
  let multiples_of_40 := {n : ℤ | ∃ k : ℤ, n = 40 * k}
  let valid_numbers := (three_digit_numbers ∩ multiples_of_25) \ multiples_of_40
  in valid_numbers.card = 32 :=
by
  sorry

end num_pos_3_digit_multiples_of_25_not_40_l168_168584


namespace binary_to_decimal_110011_l168_168090

theorem binary_to_decimal_110011 : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end binary_to_decimal_110011_l168_168090


namespace number_of_newborns_approx_150_l168_168606

noncomputable def probability_survival_per_month : ℝ := 9 / 10
noncomputable def total_months : ℝ := 3
noncomputable def expected_survivors : ℝ := 109.35

theorem number_of_newborns_approx_150 :
  let N := expected_survivors / (probability_survival_per_month ^ total_months) in
  abs (N - 150) < 1 :=
by
  sorry

end number_of_newborns_approx_150_l168_168606


namespace rectangles_in_square_rectangles_in_three_squares_l168_168391

-- Given conditions as definitions
def positive_integer (n : ℕ) : Prop := n > 0

-- Part a
theorem rectangles_in_square (n : ℕ) (h : positive_integer n) :
  (n * (n + 1) / 2) ^ 2 = (n * (n + 1) / 2) ^ 2 :=
by sorry

-- Part b
theorem rectangles_in_three_squares (n : ℕ) (h : positive_integer n) :
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 = 
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 :=
by sorry

end rectangles_in_square_rectangles_in_three_squares_l168_168391


namespace sum_of_interiors_l168_168312

theorem sum_of_interiors (n : ℕ) (h : 180 * (n - 2) = 1620) : 180 * ((n + 3) - 2) = 2160 :=
by sorry

end sum_of_interiors_l168_168312


namespace positive_integer_condition_l168_168899

theorem positive_integer_condition (n : ℕ) (h : 15 * n = n^2 + 56) : n = 8 :=
sorry

end positive_integer_condition_l168_168899


namespace probability_six_players_from_different_classes_l168_168028

noncomputable theory

open Nat

def combination (n k : ℕ) : ℕ := nat.choose n k

theorem probability_six_players_from_different_classes :
  let total_ways := combination 12 6 in
  let favorable_ways := combination 2 1 * combination 2 1 * combination 8 4 in
  (favorable_ways : ℚ) / total_ways = 10 / 33 :=
by
  sorry

end probability_six_players_from_different_classes_l168_168028


namespace probability_grace_reaches_pad_7_without_predators_l168_168339

def lily_pads := ℕ
def is_predator_lily (n : lily_pads) : Prop := n = 2 ∨ n = 5 ∨ n = 6
def has_snack (n : lily_pads) : Prop := n = 7
def start_pad := 0

noncomputable def probability_reach_pad (current : lily_pads) (target : lily_pads) : ℚ :=
  if is_predator_lily current then 0
  else if current = target then 1
  else 1 / 2 * probability_reach_pad (current + 1) target +
       1 / 2 * probability_reach_pad (current + 2) target

theorem probability_grace_reaches_pad_7_without_predators :
  probability_reach_pad start_pad 7 = 1 / 16 := 
sorry

end probability_grace_reaches_pad_7_without_predators_l168_168339


namespace number_divisible_l168_168789

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168789


namespace smallest_digits_to_append_l168_168842

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168842


namespace problem_statement_l168_168058

-- Define the set of numbers
def num_set := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_multiple (a b : ℕ) : Prop := b ∣ a

-- Problem statement
theorem problem_statement (al bill cal : ℕ) (h_al : al ∈ num_set) (h_bill : bill ∈ num_set) (h_cal : cal ∈ num_set) (h_distinct: distinct al bill cal) : 
  (is_multiple al bill) ∧ (is_multiple bill cal) →
  ∃ (p : ℚ), p = 1 / 190 :=
sorry

end problem_statement_l168_168058


namespace isabelle_weeks_needed_l168_168633

def total_ticket_cost : ℕ := 20 + 10 + 10
def total_savings : ℕ := 5 + 5
def weekly_earnings : ℕ := 3
def amount_needed : ℕ := total_ticket_cost - total_savings
def weeks_needed : ℕ := amount_needed / weekly_earnings

theorem isabelle_weeks_needed 
  (ticket_cost_isabelle : ℕ := 20)
  (ticket_cost_brother : ℕ := 10)
  (savings_brothers : ℕ := 5)
  (savings_isabelle : ℕ := 5)
  (earnings_weekly : ℕ := 3)
  (total_cost := ticket_cost_isabelle + 2 * ticket_cost_brother)
  (total_savings := savings_brothers + savings_isabelle)
  (needed_amount := total_cost - total_savings)
  (weeks := needed_amount / earnings_weekly) :
  weeks = 10 :=
  by
  sorry

end isabelle_weeks_needed_l168_168633


namespace asymptote_of_hyperbola_with_parabola_focus_l168_168199

-- Define the equations and conditions for the hyperbola and parabola
def hyperbola_eqn (a : ℝ) : Prop := ∀ x y : ℝ, (x^2 / a^2) - y^2 = 1
def parabola_eqn : Prop := ∀ x y : ℝ, y^2 = 8 * x

-- Define the focus of the parabola
def focus_parabola : (ℝ × ℝ) := (2, 0)

-- Define the right focus of the hyperbola coinciding with the focus of the parabola
def right_focus_hyperbola_coincides_with_parabola (a : ℝ) (x y : ℝ) : Prop :=
  x = 2 ∧ y = 0

-- Asymptote of the hyperbola to be proved
def asymptote_eqn (x y : ℝ) : Prop := y = (√3 / 3) * x ∨ y = - (√3 / 3) * x

-- The theorem to prove
theorem asymptote_of_hyperbola_with_parabola_focus (a : ℝ) (x y : ℝ) (h1 : hyperbola_eqn a) (h2 : parabola_eqn) (h3 : right_focus_hyperbola_coincides_with_parabola a x y) :
    asymptote_eqn x y :=
sorry

end asymptote_of_hyperbola_with_parabola_focus_l168_168199


namespace bus_passengers_final_count_l168_168306

theorem bus_passengers_final_count :
  let initial_passengers := 15
  let changes := [(3, -6), (-2, 4), (-7, 2), (3, -5)]
  let apply_change (acc : Int) (change : Int × Int) : Int :=
    acc + change.1 + change.2
  initial_passengers + changes.foldl apply_change 0 = 7 :=
by
  intros
  sorry

end bus_passengers_final_count_l168_168306


namespace shiela_paintings_l168_168447

theorem shiela_paintings (h1 : 18 % 2 = 0) : 18 / 2 = 9 := 
by sorry

end shiela_paintings_l168_168447


namespace smallest_integer_quad_ineq_l168_168007

-- Definition of the condition
def quad_ineq (n : ℤ) := n^2 - 14 * n + 45 > 0

-- Lean 4 statement of the math proof problem
theorem smallest_integer_quad_ineq : ∃ n : ℤ, quad_ineq n ∧ ∀ m : ℤ, quad_ineq m → n ≤ m :=
  by
    existsi 10
    sorry

end smallest_integer_quad_ineq_l168_168007


namespace smallest_number_append_l168_168818

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168818


namespace smallest_digits_to_append_l168_168861

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168861


namespace exists_b_l168_168273

theorem exists_b (n m : ℕ) (hn : n > 1) (hm : m > 1) (a : ℕ → ℕ)
  (ha : ∀ i ∈ Finset.range n, 1 ≤ a i ∧ a i < n^(2*m-1) - 2) :
  ∃ b : ℕ → ℕ, (∀ i ∈ Finset.range m, b i ∈ {0, 1}) ∧ 
               ∀ i ∈ Finset.range m, a i + b i < n :=
begin
  sorry
end

end exists_b_l168_168273


namespace regular_polygon_hexagon_l168_168599

theorem regular_polygon_hexagon (n : ℕ) (r : ℝ) (h : r > 0) (polygon_regular : regular_polygon n r) (side_eq_radius : polygon_regular.side_length = r) : n = 6 := by
  sorry

end regular_polygon_hexagon_l168_168599


namespace convert_fraction_to_decimal_l168_168088

theorem convert_fraction_to_decimal : (3 / 40 : ℝ) = 0.075 := 
by
  sorry

end convert_fraction_to_decimal_l168_168088


namespace smallest_number_of_digits_to_append_l168_168811

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168811


namespace area_ratio_triangle_l168_168534

theorem area_ratio_triangle {A B C G P Q R : Type*} [Triangle A B C] 
  (medians : Medians A B C) (angle_bisectors : AngleBisectors A B C)
  (P_def : P = intersect angle_bisectors.w_a medians.m_b)
  (Q_def : Q = intersect angle_bisectors.w_b medians.m_c)
  (R_def : R = intersect angle_bisectors.w_c medians.m_a)
  (delta F : ℝ) (area_def : delta = triangle_area P Q R)
  (triangle_area_def: F = triangle_area A B C) :
  delta / F < 1 / 6 :=
sorry

end area_ratio_triangle_l168_168534


namespace possible_values_f1_l168_168314

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add : ∀ x y : ℝ, f(x + y) = f(x) * f(y)
axiom f_non_zero_function : ¬ ∀ x : ℝ, f(x) = 0
axiom exists_a : ∃ a : ℝ, a ≠ 0 ∧ f(a) = 2

theorem possible_values_f1 : f(1) = Real.sqrt 2 ∨ f(1) = -Real.sqrt 2 := 
by
  sorry

end possible_values_f1_l168_168314


namespace B_days_to_complete_job_alone_l168_168402

theorem B_days_to_complete_job_alone (x : ℝ) : 
  (1 / 15 + 1 / x) * 4 = 0.4666666666666667 → x = 20 :=
by
  intro h
  -- Note: The proof is omitted as we only need the statement here.
  sorry

end B_days_to_complete_job_alone_l168_168402


namespace find_theta_l168_168495

def F (x θ : ℝ) : ℝ :=
  x^2 * Real.cos θ + x * (1 + x) - (1 + x)^2 * Real.sin θ

theorem find_theta (θ : ℝ) (hθ1 : 0 ≤ θ) (hθ2 : θ ≤ π) :
  (∀ x : ℝ, -1 ≤ x → x ≤ 1 → F x θ < 0) ↔ (θ > π/2 ∧ θ < π) :=
by
  sorry

end find_theta_l168_168495


namespace average_speed_trip_l168_168411

theorem average_speed_trip :
  let d1 := 40 -- distance of the first segment in kilometers
  let s1 := 50 -- speed of the first segment in kilometers per hour
  let d2 := 35 -- distance of the second segment in kilometers
  let s2 := 30 -- speed of the second segment in kilometers per hour
  let d3 := 25 -- distance of the third segment in kilometers
  let s3 := 60 -- speed of the third segment in kilometers per hour
  let total_distance := d1 + d2 + d3
  let t1 := d1 / s1 -- time for the first segment in hours
  let t2 := d2 / s2 -- time for the second segment in hours
  let t3 := d3 / s3 -- time for the third segment in hours
  let total_time := t1 + t2 + t3
  total_distance = 100 →
  abs (total_distance / total_time - 41.96) < 0.01 := sorry

end average_speed_trip_l168_168411


namespace train_crosses_bridge_in_30_seconds_l168_168047

theorem train_crosses_bridge_in_30_seconds
    (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ)
    (h1 : train_length = 110)
    (h2 : train_speed_kmh = 45)
    (h3 : bridge_length = 265) : 
    (train_length + bridge_length) / (train_speed_kmh * (1000 / 3600)) = 30 := 
by
  sorry

end train_crosses_bridge_in_30_seconds_l168_168047


namespace circumcenter_ABD_AC_l168_168643

open EuclideanGeometry

variables {A B C D M O P : Point}

theorem circumcenter_ABD_AC
  (h_trapezoid : Trapezoid A B C D)
  (O_circumABC : Cir_ring O A B C)
  (O_BD : Lies_on O (Line B D)) :
  ∃ Q, Cir_ring Q A B D ∧ Lies_on Q (Line A C) :=
by 
  sorry

end circumcenter_ABD_AC_l168_168643


namespace range_of_g_l168_168480

theorem range_of_g (A : ℝ) (h : A ≠ n * π) :
  let g := λ A : ℝ, (cos A * (2 * sin A ^ 2 + 3 * sin A ^ 4 + 2 * cos A ^ 2 + (cos A ^ 2) * (sin A ^ 2))) /
                    (cot A * (csc A - cos A * cot A))
  in (2 ≤ g A) ∧ (g A ≤ 5) :=
by
  let g := λ A : ℝ, (cos A * (2 * sin A ^ 2 + 3 * sin A ^ 4 + 2 * cos A ^ 2 + (cos A ^ 2) * (sin A ^ 2))) /
                    (cot A * (csc A - cos A * cot A))
  sorry

end range_of_g_l168_168480


namespace ratio_of_segments_l168_168947

theorem ratio_of_segments (O P A B C D L : Point)
    (circle : Circle O)
    (AB CD : Line)
    (on_circle_AB : chord_length circle AB = 5)
    (on_circle_CD : chord_length circle CD = 5)
    (extension_intersect : ∃ (extend_P : Line), extension_intersect AB CD P)
    (DP_length : segment_length D P = 13)
    (PO_intersect_AC : ∃ (P O inter : Line), line_intersects AC PO L) :
    (AL/LC) = (13/18) :=
sorry

end ratio_of_segments_l168_168947


namespace solve_system_l168_168297

theorem solve_system :
  ∀ x y z : ℝ,
  (y + z = x * y * z) →
  (z + x = x * y * z) →
  (x + y = x * y * z) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = sqrt 2 ∧ y = sqrt 2 ∧ z = sqrt 2) ∨
  (x = -sqrt 2 ∧ y = -sqrt 2 ∧ z = -sqrt 2) :=
by
  intro x y z
  intro h1 h2 h3
  sorry

end solve_system_l168_168297


namespace allocation_ways_l168_168880

theorem allocation_ways (programs : Finset ℕ) (grades : Finset ℕ) (h_programs : programs.card = 6) (h_grades : grades.card = 4) : 
  ∃ ways : ℕ, ways = 1080 := 
by 
  sorry

end allocation_ways_l168_168880


namespace area_of_EFGH_l168_168298

noncomputable def radius_of_circle (side_length : ℝ) : ℝ :=
  let diagonal := Real.sqrt (2 * side_length^2) in
  diagonal / 2

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length^2

theorem area_of_EFGH (ABCD_area : ℝ)
  (EF_on_AB : Bool) (GH_on_circle : Bool) (EFGH_side_on_square_ABCD : Bool)
  (circumscribed_square_side : ℝ) :
  area_of_square (circumscribed_square_side / Real.sqrt 2) = 8 :=
by
  sorry

end area_of_EFGH_l168_168298


namespace t_n_minus_n_even_l168_168644

noncomputable def number_of_nonempty_subsets_with_integer_average (n : ℕ) : ℕ := 
  sorry

theorem t_n_minus_n_even (N : ℕ) (hN : N > 1) :
  ∃ T_n, T_n = number_of_nonempty_subsets_with_integer_average N ∧ (T_n - N) % 2 = 0 :=
by
  sorry

end t_n_minus_n_even_l168_168644


namespace probability_odd_sum_subsets_l168_168323

theorem probability_odd_sum_subsets :
  let n := 2017
  let total_subsets := 2^n - 1
  let odd_sum_subsets := 2^(n - 1)
  next
    (odd_sum_subsets * 2 = total_subsets) -- Verifies that half of the subsets have odd sums and half have even sums
  in
    (odd_sum_subsets / total_subsets) = (2^(n - 1)) / (2^n - 1) :=
by
  let n := 2017
  let total_subsets := 2^n - 1
  let odd_sum_subsets := 2^(n - 1)
  have h1 : odd_sum_subsets * 2 = total_subsets, from sorry
  have h2 : (odd_sum_subsets / total_subsets) = (2^(n - 1)) / (2^n - 1), from sorry
  exact h2

end probability_odd_sum_subsets_l168_168323


namespace error_in_major_premise_l168_168249

/-- 
  To identify the error in the reasoning process of the statement:
  - Major premise: The sum of two irrational numbers is always an irrational number.
  - Minor premise: sqrt(2) and sqrt(3) are irrational numbers.
  - Conclusion: Therefore, sqrt(2) + sqrt(3) is also an irrational number.
- The correct answer is that the error lies in the major premise.
--/
theorem error_in_major_premise : 
  ∀ (irr_irr_sum_irrational : ∀ x y : ℝ, irrational x → irrational y → irrational (x + y)),
  (irrational (Real.sqrt 2) ∧ irrational (Real.sqrt 3))
  → ∃ x y : ℝ, irrational x ∧ irrational y ∧ ¬ irrational (x + y) :=
by
  sorry

end error_in_major_premise_l168_168249


namespace slope_angle_of_tangent_line_l168_168327

theorem slope_angle_of_tangent_line (x y : ℝ) (h : y = x^2 - x) (h_point : (x, y) = (1, 0)) :
  real.arctan (2 * 1 - 1) = real.pi / 4 := 
sorry

end slope_angle_of_tangent_line_l168_168327


namespace old_price_per_kwh_l168_168723

theorem old_price_per_kwh (power_old_computer_watts : ℕ) (time_hours : ℕ) (total_cost : ℝ) (power_conversion : power_old_computer_watts = 800): 
  (time_hours = 50) → (total_cost = 9) → old_price : ℝ , old_price = 0.225 → old_price = total_cost / (power_old_computer_watts.to_real / 1000 * time_hours) :=
by
  sorry

end old_price_per_kwh_l168_168723


namespace sum_of_squares_of_medians_l168_168365

noncomputable def mAD (a b c : ℝ) := real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2
noncomputable def mBE (a b c : ℝ) := real.sqrt (2 * a^2 + 2 * c^2 - b^2) / 2
noncomputable def mCF (a b c : ℝ) := real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2

theorem sum_of_squares_of_medians (a b c : ℝ) (h₁ : a = 13) (h₂ : b = 13) (h₃ : c = 10) :
  (mAD a b c)^2 + (mBE a b c)^2 + (mCF a b c)^2 = 244 :=
by sorry

end sum_of_squares_of_medians_l168_168365


namespace quotient_larger_than_dividend_l168_168918

-- Define the problem conditions
variables {a b : ℝ}

-- State the theorem corresponding to the problem
theorem quotient_larger_than_dividend (h : b ≠ 0) : ¬ (∀ a : ℝ, ∀ b : ℝ, (a / b > a) ) :=
by
  sorry

end quotient_larger_than_dividend_l168_168918


namespace coefficient_of_x_l168_168095

theorem coefficient_of_x :
  let expr := (5 * (x - 6)) + (6 * (9 - 3 * x ^ 2 + 3 * x)) - (9 * (5 * x - 4))
  (expr : ℝ) → 
  let expr' := 5 * x - 30 + 54 - 18 * x ^ 2 + 18 * x - 45 * x + 36
  (expr' : ℝ) → 
  let coeff_x := 5 + 18 - 45
  coeff_x = -22 :=
by
  sorry

end coefficient_of_x_l168_168095


namespace square_division_possible_square_division_impossible_n_2_square_division_impossible_n_3_l168_168295

theorem square_division_possible (n : ℕ) (h : n > 5) : 
  ∃ squares : list (set (ℝ × ℝ)), 
    (∀ s ∈ squares, ∃ a b : ℝ, is_square s a b) ∧  -- each smaller set is a square
    disjoint_union squares ∧  -- all smaller squares are disjoint
    union_of_squares squares = original_square ∧  -- they together form the original square
    squares.length = n := 
sorry

theorem square_division_impossible_n_2 :
  ¬ ∃ squares : list (set (ℝ × ℝ)),
    (∀ s ∈ squares, ∃ a b : ℝ, is_square s a b) ∧
    disjoint_union squares ∧
    union_of_squares squares = original_square ∧
    squares.length = 2 := 
sorry

theorem square_division_impossible_n_3 :
  ¬ ∃ squares : list (set (ℝ × ℝ)),
    (∀ s ∈ squares, ∃ a b : ℝ, is_square s a b) ∧
    disjoint_union squares ∧
    union_of_squares squares = original_square ∧
    squares.length = 3 := 
sorry

end square_division_possible_square_division_impossible_n_2_square_division_impossible_n_3_l168_168295


namespace xiaoming_total_savings_l168_168380

-- Declare the conditions as hypotheses
variables (x y z : ℕ) -- number of 2-cent and 5-cent coins in respective piles
variables (total_cents : ℕ) (total_yuan : ℝ)
hypothesis h1 : 2 * y = 5 * z
hypothesis h2 : 7 * x = 4 * y
hypothesis h3 : total_cents = 100 * 5 + 100 * 1.4

-- Prove that the total amount of money Xiaoming saved is 5.6 yuan
theorem xiaoming_total_savings :
  total_cents = 560 ∧ total_yuan = 5.6 :=
by
  sorry

end xiaoming_total_savings_l168_168380


namespace solution_l168_168277

noncomputable def f (a x : ℝ) : ℝ :=
  a / x - x

def condition (a : ℝ) : Prop :=
  ∀ x ∈ set.Ioo (1 / 4 : ℝ) 1, abs ((a / x - x) * (x - 1 / 2)) ≤ 1

theorem solution (a : ℝ) : condition a → a ≤ 17 / 16 :=
begin
  sorry
end

end solution_l168_168277


namespace ellipse_sum_l168_168263

theorem ellipse_sum (F1 F2 : ℝ × ℝ) (h k a b : ℝ) 
  (hf1 : F1 = (0, 0)) (hf2 : F2 = (6, 0))
  (h_eqn : ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = 10) :
  h + k + a + b = 12 :=
by
  sorry

end ellipse_sum_l168_168263


namespace jonah_walked_8_miles_l168_168640

def speed : ℝ := 4
def time : ℝ := 2
def distance (s t : ℝ) : ℝ := s * t

theorem jonah_walked_8_miles : distance speed time = 8 := sorry

end jonah_walked_8_miles_l168_168640


namespace find_n_l168_168123

theorem find_n (n : ℤ) (hn_range : -150 < n ∧ n < 150) (h_tan : Real.tan (n * Real.pi / 180) = Real.tan (286 * Real.pi / 180)) : 
  n = -74 :=
sorry

end find_n_l168_168123


namespace gcf_lcm_sum_l168_168262

def gcf (a b c : ℕ) : ℕ := 
  Nat.gcd a (Nat.gcd b c)

def lcm (a b c : ℕ) : ℕ := 
  Nat.lcm a (Nat.lcm b c)

theorem gcf_lcm_sum :
  let C := gcf 9 15 45 in
  let D := lcm 9 15 45 in
  C + D = 60 :=
by
  sorry

end gcf_lcm_sum_l168_168262


namespace max_leftover_apples_l168_168903

theorem max_leftover_apples (a : ℕ) : (a % 7 < 7) ∧ (a % 7 = 6) :=
begin
  sorry
end

end max_leftover_apples_l168_168903


namespace rectangle_area_in_cm_l168_168997

theorem rectangle_area_in_cm (length_in_m : ℝ) (width_in_m : ℝ) 
  (h_length : length_in_m = 0.5) (h_width : width_in_m = 0.36) : 
  (100 * length_in_m) * (100 * width_in_m) = 1800 :=
by
  -- We skip the proof for now
  sorry

end rectangle_area_in_cm_l168_168997


namespace find_x_values_l168_168511

theorem find_x_values (x : ℝ) :
  (1/2 * x^2 = 5 ∧ (x - 1)^2 = 16) →
  (x = sqrt 10 ∨ x = -sqrt 10 ∨ x = 5 ∨ x = -3) :=
by
  intros h
  cases h
  sorry

end find_x_values_l168_168511


namespace probability_shooter_correct_statements_l168_168043

-- Define the probability of hitting the target in a single shot
def p_hit : ℝ := 0.9

-- Define the probability of missing the target in a single shot
def p_miss : ℝ := 0.1

-- Define the probability of hitting the target exactly 3 times out of 4 shots
def prob_hit_exactly_three_times : ℝ := (nat.choose 4 3) * (p_hit ^ 3) * p_miss

-- Define the probability of hitting the target at least once in 4 shots
def prob_hit_at_least_once : ℝ := 1 - (p_miss ^ 4)

-- Define the statements
def statement_1 : Prop := p_hit = 0.9
def statement_2 : Prop := prob_hit_exactly_three_times = 0.9^3 * 0.1
def statement_3 : Prop := prob_hit_at_least_once = 1 - 0.1^4

-- Define the number of correct statements
def num_correct_statements : ℕ := if statement_1 ∧ ¬statement_2 ∧ statement_3 then 2 else 0

-- The theorem statement
theorem probability_shooter_correct_statements : 
  num_correct_statements = 2 :=
by
  simp only [statement_1, statement_2, statement_3, prob_hit_exactly_three_times, prob_hit_at_least_once]
  sorry

end probability_shooter_correct_statements_l168_168043


namespace sum_of_six_terms_l168_168529

variable (a₁ a₂ a₃ a₄ a₅ a₆ q : ℝ)

-- Conditions
def geom_seq := a₂ = q * a₁ ∧ a₃ = q * a₂ ∧ a₄ = q * a₃ ∧ a₅ = q * a₄ ∧ a₆ = q * a₅
def cond₁ : Prop := a₁ + a₃ = 5 / 2
def cond₂ : Prop := a₂ + a₄ = 5 / 4

-- Problem Statement
theorem sum_of_six_terms : geom_seq a₁ a₂ a₃ a₄ a₅ a₆ q → cond₁ a₁ a₃ → cond₂ a₂ a₄ → 
  (a₁ * (1 - q^6) / (1 - q) = 63 / 16) := 
by 
  sorry

end sum_of_six_terms_l168_168529


namespace explicit_formula_for_f_range_of_f_on_interval_l168_168996

-- Define the quadratic function f with given conditions
def f (x : ℝ) : ℝ := x^2 - x + 1

-- State and prove the explicit formula for f
theorem explicit_formula_for_f :
  (f 0 = 1) ∧ (∀ x : ℝ, f (x + 1) - f x = 2 * x) →
  (∀ x : ℝ, f x = x^2 - x + 1) := by
  sorry

-- Define the function and consider the interval [-1, 1]
noncomputable def interval_range_f := set.range (λ x, f x) ∩ (set.Icc (-1 : ℝ) (1 : ℝ))

-- State and prove the range of f on the interval [-1, 1]
theorem range_of_f_on_interval :
  interval_range_f = set.Icc (3 / 4) 3 := by
  sorry

end explicit_formula_for_f_range_of_f_on_interval_l168_168996


namespace interest_percentage_approx_to_nearest_tenth_l168_168019

noncomputable def purchase_price : ℝ := 112
noncomputable def down_payment : ℝ := 12
noncomputable def monthly_payment : ℝ := 10
noncomputable def monthly_payment_count : ℝ := 12

noncomputable def total_monthly_payments : ℝ := monthly_payment_count * monthly_payment
noncomputable def total_amount_paid : ℝ := total_monthly_payments + down_payment
noncomputable def interest_paid : ℝ := total_amount_paid - purchase_price
noncomputable def interest_percent : ℝ := (interest_paid / purchase_price) * 100

theorem interest_percentage_approx_to_nearest_tenth :
  Float.round (interest_percent * 10) / 10 = 17.9 := sorry

end interest_percentage_approx_to_nearest_tenth_l168_168019


namespace sum_of_squares_of_medians_l168_168362

theorem sum_of_squares_of_medians (a b c : ℝ) (h1 : a = 13) (h2 : b = 13) (h3 : c = 10) :
  let m₁ := (1/2 : ℝ) * math.sqrt(2 * b^2 + 2 * c^2 - a^2);
      m₂ := (1/2 : ℝ) * math.sqrt(2 * a^2 + 2 * c^2 - b^2);
      m₃ := (1/2 : ℝ) * math.sqrt(2 * a^2 + 2 * b^2 - c^2)
  in m₁^2 + m₂^2 + m₃^2 = 278.5 := by
  -- Proof skeleton here
  sorry

end sum_of_squares_of_medians_l168_168362


namespace part1_part2_l168_168580

noncomputable def a : ℝ × ℝ := (real.sqrt 3, 1)
def a_dot_b (b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem part1 (b : ℝ × ℝ) (hb : magnitude b = 4) (hab : a_dot_b b = 4) :
  magnitude (a.1 + b.1, a.2 + b.2) = 2 * real.sqrt 7 :=
by sorry

theorem part2 (b : ℝ × ℝ) (hab : a_dot_b b = 4) :
  ∃ (m : ℝ), magnitude b = m ∧ m = 2 ∧ real.arccos (a_dot_b b / (magnitude a * m)) = 0 :=
by sorry

end part1_part2_l168_168580


namespace line_passes_through_fixed_point_line_perpendicular_line_passes_first_quadrant_when_a_lt_0_line_passes_second_third_fourth_quadrant_when_a_gt_0_l168_168291

theorem line_passes_through_fixed_point (a : ℝ) : (∀ x y, ax + y + a = 0 → (x = -1 ∧ y = 0)) :=
by
  intro x y h
  have : a * (-1) + 0 + a = 0 := by linarith
  exact ⟨rfl, rfl⟩

theorem line_perpendicular (a : ℝ) (h : a = -1) : (∀ x y, ax + y + a = 0 → ∃ x' y', x' + y' - 2 = 0 ∧ (y = x + 1)) :=
by
  intro x y h1
  use [x, y]
  rw [h, mul_neg_one] at h1
  exact ⟨by linarith, by linarith⟩

theorem line_passes_first_quadrant_when_a_lt_0 (a : ℝ) : (a < 0 → ∀ x y, ax + y + a = 0 → x > 0 ∧ y > 0 ∧ y < ( -a * x - a)) :=
by
  intro h1 x y h2
  have h3 : y = -a * x - a := by linarith
  exact ⟨by linarith [h1, h2, h3], by linarith [h1, h2, h3]⟩

theorem line_passes_second_third_fourth_quadrant_when_a_gt_0 (a : ℝ) : (a > 0 → ∀ x y, ax + y + a = 0 → x < 0 ∧ y < 0 ∧ y > ( -a * x - a)) :=
by
  intro h1 x y h2
  have h3 : y = -a * x - a := by linarith
  exact ⟨by linarith [h1, h2, h3], by linarith [h1, h2, h3]⟩

-- Place a sorry to skip the proofs
sorry

end line_passes_through_fixed_point_line_perpendicular_line_passes_first_quadrant_when_a_lt_0_line_passes_second_third_fourth_quadrant_when_a_gt_0_l168_168291


namespace largest_perimeter_is_34_l168_168342

noncomputable def largest_perimeter_of_polygon : ℕ :=
  let side_length := 2
  let square_sides := 4
  let congruent_sides := 8 -- from solving interior angle equation
  let internal_angle_sum := 360
  let perimeter := 2 * (2 * side_length * 7 + side_length * 3)
  perimeter

theorem largest_perimeter_is_34 
  (side_length : ℕ) (square_sides : ℕ) (congruent_sides : ℕ) (internal_angle_sum : ℕ) :
  side_length = 2 → square_sides = 4 → congruent_sides = 8 → internal_angle_sum = 360 →
  largest_perimeter_of_polygon = 34 := by
  intros
  rw [side_length, square_sides, congruent_sides, internal_angle_sum]
  have h : largest_perimeter_of_polygon = 34 := rfl
  exact h

end largest_perimeter_is_34_l168_168342


namespace tangent_line_circumcircle_l168_168535

variables {A B C O P : Point}
variables {O_A O_B O_C : Point}
variables {l_A l_B l_C : Line}
variables {A_prime B_prime C_prime : Point}

-- Assuming basic definitions of points and lines
-- Definitions for circumcenter, circumcircle, and perpendicularity are also assumed

-- Conditions
def acute_triangle (A B C : Point) : Prop := sorry
def on_circumcircle (P : Point) : Prop := sorry
def circumcenter (A B C : Point) : Point := sorry
def perpendicular_to (l : Line) (B C : Point) : Prop := sorry
def intersection_point (L1 L2 : Line) : Point := sorry
def tangent (L : Line) (circ : Circle) : Prop := sorry

-- Given
axiom h1: acute_triangle A B C
axiom h2: on_circumcircle P
axiom h3: P ≠ A ∧ P ≠ B ∧ P ≠ C
axiom h4: P ≠ antipode(A, O) ∧ P ≠ antipode(B, O) ∧ P ≠ antipode(C, O)
axiom h5: O_A = circumcenter A O P
axiom h6: O_B = circumcenter B O P
axiom h7: O_C = circumcenter C O P
axiom h8: perpendicular_to l_A B C
axiom h9: perpendicular_to l_B C A
axiom h10: perpendicular_to l_C A B

-- Intersections
def A_prime := intersection_point OP l_A
def B_prime := intersection_point OP l_B
def C_prime := intersection_point OP l_C

-- Prove tangency
theorem tangent_line_circumcircle : tangent OP (circumcircle A_prime B_prime C_prime) := sorry

end tangent_line_circumcircle_l168_168535


namespace cos_alpha_minus_beta_l168_168164

-- Given conditions
variables (α β : ℝ)
hypothesis h1 : α ∈ Ioc (-(π / 4)) (π / 4)
hypothesis h2 : β ∈ Ioc (-(π / 4)) (π / 4)
hypothesis h_cos_sum : cos (2 * α + 2 * β) = -7 / 9
hypothesis h_sin_product : sin α * sin β = 1 / 4

-- Goal statement
theorem cos_alpha_minus_beta : cos (α - β) = 5 / 6 :=
by
  sorry

end cos_alpha_minus_beta_l168_168164


namespace probability_of_diff_colors_l168_168605

noncomputable def probability_different_colors 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (yellow_balls : ℕ) : ℚ :=
  if total_balls = 4 ∧ red_balls = 2 ∧ yellow_balls = 2 then
    (C 4 2 * (C 2 1 * C 2 1)) / C 4 2
  else 0

theorem probability_of_diff_colors : probability_different_colors 4 2 2 = 2 / 3 := 
  sorry

end probability_of_diff_colors_l168_168605


namespace range_of_a_l168_168601

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x + 6 < 2 + 3x → (a + x) / 4 > x) ∧ (∃! i : ℤ, ∃! j : ℤ, ∃! k : ℤ, 2 < i ∧ i < a / 3 ∧ 2 < j ∧ j < a / 3 ∧ 2 < k ∧ k < a / 3) → 15 < a ∧ a ≤ 18 :=
by
  sorry

end range_of_a_l168_168601


namespace divide_composite_products_l168_168486

def first_eight_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List ℕ := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem divide_composite_products :
  product first_eight_composites * 3120 = product next_eight_composites :=
by
  -- This would be the place for the proof solution
  sorry

end divide_composite_products_l168_168486


namespace range_of_a_l168_168570

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0)
  ↔ a ∈ Icc (-3/5 : ℝ) 1 :=
begin
  sorry
end

end range_of_a_l168_168570


namespace harvest_bushels_l168_168452

def num_rows : ℕ := 5
def stalks_per_row : ℕ := 80
def stalks_per_bushel : ℕ := 8

theorem harvest_bushels : (num_rows * stalks_per_row) / stalks_per_bushel = 50 := by
  sorry

end harvest_bushels_l168_168452


namespace find_intersection_point_l168_168974

theorem find_intersection_point :
  ∃ (x y z : ℝ), 
    ((∃ t : ℝ, x = 1 + 2 * t ∧ y = 1 - t ∧ z = -2 + 3 * t) ∧ 
    (4 * x + 2 * y - z - 11 = 0)) ∧ 
    (x = 3 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end find_intersection_point_l168_168974


namespace find_normal_price_l168_168358

theorem find_normal_price (P : ℝ) (S : ℝ) (d1 d2 d3 : ℝ) : 
  (P * (1 - d1) * (1 - d2) * (1 - d3) = S) → S = 144 → d1 = 0.12 → d2 = 0.22 → d3 = 0.15 → P = 246.81 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_normal_price_l168_168358


namespace base_eight_to_base_ten_l168_168352

theorem base_eight_to_base_ten {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 5) (h3 : d3 = 7) :
  d3 * 8^0 + d2 * 8^1 + d1 * 8^2 = 111 := 
by
  sorry

end base_eight_to_base_ten_l168_168352


namespace sin_neg_765_eq_neg_sqrt2_div_2_l168_168734

theorem sin_neg_765_eq_neg_sqrt2_div_2 :
  Real.sin (-765 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_765_eq_neg_sqrt2_div_2_l168_168734


namespace fixed_point_through_ellipse_l168_168191

-- Define the ellipse and the points
def C (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def P2 : ℝ × ℝ := (0, 1)

-- Define the condition for a line not passing through P2 and intersecting the ellipse
def line_l_intersects_ellipse (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∃ (x1 x2 b k : ℝ), l (x1, k * x1 + b) ∧ l (x2, k * x2 + b) ∧
  (C x1 (k * x1 + b)) ∧ (C x2 (k * x2 + b)) ∧
  ((x1, k * x1 + b) ≠ P2 ∧ (x2, k * x2 + b) ≠ P2) ∧
  ((k * x1 + b ≠ 1) ∧ (k * x2 + b ≠ 1)) ∧ 
  (∃ (kA kB : ℝ), kA = (k * x1 + b - 1) / x1 ∧ kB = (k * x2 + b - 1) / x2 ∧ kA + kB = -1)

-- Prove there exists a fixed point (2, -1) through which all such lines must pass
theorem fixed_point_through_ellipse (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  line_l_intersects_ellipse A B l → l (2, -1) :=
sorry

end fixed_point_through_ellipse_l168_168191


namespace sum_of_squares_of_medians_l168_168363

theorem sum_of_squares_of_medians (a b c : ℝ) (h1 : a = 13) (h2 : b = 13) (h3 : c = 10) :
  let m₁ := (1/2 : ℝ) * math.sqrt(2 * b^2 + 2 * c^2 - a^2);
      m₂ := (1/2 : ℝ) * math.sqrt(2 * a^2 + 2 * c^2 - b^2);
      m₃ := (1/2 : ℝ) * math.sqrt(2 * a^2 + 2 * b^2 - c^2)
  in m₁^2 + m₂^2 + m₃^2 = 278.5 := by
  -- Proof skeleton here
  sorry

end sum_of_squares_of_medians_l168_168363


namespace competition_duration_and_medals_l168_168726

theorem competition_duration_and_medals (n m : ℕ) (h1 : n > 1) 
    (h2 : m > 0)
    (h3 : ∀ k, 1 ≤ k ∧ k ≤ n → 
         (if k = 1 then 1 + (m - 1) / 7
          else k + (m - (∑ i in range (k - 1), i) - k) / 7) = 
         (if k = 1 then (m + 6) / 7
          else (6 * (k - 1) + m - 36) / 7 + 6)) : n = 6 ∧ m = 36 := 
sorry

end competition_duration_and_medals_l168_168726


namespace incident_reflected_eqs_l168_168547

theorem incident_reflected_eqs {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), A = (2, 3) ∧ B = (1, 1) ∧ 
   (∀ (P : ℝ × ℝ), (P = A ∨ P = B → (P.1 + P.2 + 1 = 0) → false)) ∧
   (∃ (line_inc line_ref : ℝ × ℝ × ℝ),
     line_inc = (5, -4, 2) ∧
     line_ref = (4, -5, 1))) :=
sorry

end incident_reflected_eqs_l168_168547


namespace verify_digits_l168_168892

theorem verify_digits :
  ∀ (a b c d e f g h : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 →
  (10 * a + b) - (10 * c + d) = 10 * e + d →
  e * f = 10 * d + c →
  (10 * g + d) + (10 * g + b) = 10 * h + c →
  a = 9 ∧ b = 8 ∧ c = 2 ∧ d = 4 ∧ e = 7 ∧ f = 6 ∧ g = 1 ∧ h = 3 :=
by
  intros a b c d e f g h
  intros h1 h2 h3
  sorry

end verify_digits_l168_168892


namespace people_in_group_l168_168704

-- Define the conditions as Lean definitions
def avg_weight_increase := 2.5
def replaced_weight := 45
def new_weight := 65
def weight_difference := new_weight - replaced_weight -- 20 kg

-- State the problem as a Lean theorem
theorem people_in_group :
  ∀ n : ℕ, avg_weight_increase * n = weight_difference → n = 8 :=
by
  intros n h
  sorry

end people_in_group_l168_168704


namespace ratio_of_areas_l168_168230

variables {P Q R T : Type}

noncomputable def ratio_areas_of_triangles (P Q R T : Point) (QT TR : ℝ) 
  (hQT : QT = 6) (hTR : TR = 10)
  (h_point_on_side : point_on_side T Q R) 
  : ℕ × ℕ :=
(3, 5)

theorem ratio_of_areas (P Q R T : Point) (hQT : QT = 6) (hTR : TR = 10)
  (h_point_on_side : point_on_side T Q R)
  : ratio_areas_of_triangles P Q R T QT TR hQT hTR h_point_on_side = (3, 5) :=
sorry

end ratio_of_areas_l168_168230


namespace sum_of_three_numbers_l168_168373

theorem sum_of_three_numbers (x y z : ℝ) (h1 : x + y = 31) (h2 : y + z = 41) (h3 : z + x = 55) :
  x + y + z = 63.5 :=
by
  sorry

end sum_of_three_numbers_l168_168373


namespace append_digits_divisible_by_all_less_than_10_l168_168792

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168792


namespace smallest_digits_to_append_l168_168857

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168857


namespace tetrahedron_volume_relation_l168_168654

theorem tetrahedron_volume_relation 
  (V : ℝ) 
  (S : Fin 4 → ℝ) 
  (R l : Fin 4 → ℝ) 
  (h : ∀ i, S i * (l i ^ 2 - R i ^ 2) = 18 * V ^ 2) : 
  18 * V ^ 2 = ∑ i, S i ^ 2 * (l i ^ 2 - R i ^ 2) :=
begin
  sorry
end

end tetrahedron_volume_relation_l168_168654


namespace polynomial_form_l168_168496

variable (p : ℝ → ℝ)

theorem polynomial_form (h : ∀ a b c : ℝ, 
  p (a + b - 2 * c) + p (b + c - 2 * a) + p (c + a - 2 * b) = 
  3 * p (a - b) + 3 * p (b - c) + 3 * p (c - a)) :
  ∃ λ μ : ℝ, ∀ x, p x = λ * x^2 + μ * x :=
by
  sorry

end polynomial_form_l168_168496


namespace turtle_distance_in_six_minutes_l168_168054

theorem turtle_distance_in_six_minutes 
  (observers : ℕ)
  (time_interval : ℕ)
  (distance_seen : ℕ)
  (total_time : ℕ)
  (total_distance : ℕ)
  (observation_per_minute : ∀ t ≤ total_time, ∃ n : ℕ, n ≤ observers ∧ (∃ interval : ℕ, interval ≤ time_interval ∧ distance_seen = 1)) :
  total_distance = 10 :=
sorry

end turtle_distance_in_six_minutes_l168_168054


namespace megan_popsicles_l168_168673

def popsicles_consumed (start_time end_time duration_per_popsicle : ℕ) : ℕ :=
  let total_time := (end_time - start_time)
  total_time / duration_per_popsicle

theorem megan_popsicles : popsicles_consumed 60*13 60*18+20 20 = 16 := by
  -- Definition of start and end time in minutes since the start of the day
  let start_time := 60 * 13   -- 1:00 PM is the 13th hour
  let end_time := 60 * 18 + 20 -- 6:20 PM is the 18th hour + 20 minutes
  let duration_per_popsicle := 20 -- 20 minutes for each popsicle
  -- Calculate the total time in minutes
  have total_time : ℕ := end_time - start_time
  -- Check if total_time is 320
  have total_time_eq : total_time = 320 := by
    calc total_time
        = (60 * 18 + 20) - (60 * 13) : by sorry -- calculation grosso modo
    ... = 320 : by sorry
  -- Check the division of total_time by duration_per_popsicle
  have division_eq : total_time / duration_per_popsicle = 16 := by
    rw total_time_eq
    exact nat.div_eq_of_eq_mul_left (by decide) rfl
  exact division_eq

end megan_popsicles_l168_168673


namespace percent_increase_stock_l168_168449

theorem percent_increase_stock (P_open P_close: ℝ) (h1: P_open = 30) (h2: P_close = 45):
  (P_close - P_open) / P_open * 100 = 50 :=
by
  sorry

end percent_increase_stock_l168_168449


namespace sum_of_tangents_is_product_l168_168330

open Real

theorem sum_of_tangents_is_product :
  tan 117° + tan 118° + tan 125° = tan 117° * tan 118° * tan 125° :=
by sorry

end sum_of_tangents_is_product_l168_168330


namespace product_of_solutions_l168_168140

noncomputable def absolute_value (x : ℝ) : ℝ :=
  if x >= 0 then x else -x

theorem product_of_solutions : (∏ x in ({3, -3} : finset ℝ), x) = -9 :=
by 
  have hsol : {x : ℝ | absolute_value x = 3 * (absolute_value x - 2)} = {3, -3},
  { sorry },  -- Proof that the solutions set is exactly {3, -3}
  rw finset.prod_eq_mul,
  norm_num
  sorry

end product_of_solutions_l168_168140


namespace each_person_receives_equal_share_l168_168685

theorem each_person_receives_equal_share :
  (527500 / 5) = 105500 :=
by 
  -- Total amount of money
  let total_money : ℕ := 527500
  -- Number of people inheriting the money
  let number_of_people : ℕ := 5
  -- Money each person receives
  let each_person_gets : ℕ := total_money / number_of_people
  show each_person_gets = 105500, from sorry

end each_person_receives_equal_share_l168_168685


namespace find_a2017_l168_168725

def seq : ℕ → ℝ 
| 0       := 2
| (n + 1) := 1 / (1 - seq n)

theorem find_a2017 : seq 2017 = 2 := sorry

end find_a2017_l168_168725


namespace proof_problem_l168_168655

variable (a : ℝ) (f : ℝ → ℝ)

-- Given conditions
def conditions : Prop :=
  (0 < a ∧ a < 1) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x > f y ) ∧
  (f (1/2) = 0) ∧
  (∀ x, f (log a x) > 0)

-- Range of x to prove
def correct_range (x : ℝ) : Prop :=
  x > 1 / Real.sqrt a ∨ Real.sqrt a < x ∧ x < 1

theorem proof_problem (h : conditions a f) : ∀ x, f (log a x) > 0 → correct_range a x :=
begin
  intros x hx,
  sorry
end

end proof_problem_l168_168655


namespace cone_central_angle_l168_168221

/--
Given a cone with a base radius of 4 cm and a slant height of 9 cm,
prove that the central angle of the unfolded side of the cone is 160 degrees.
-/
theorem cone_central_angle (r l : ℝ) (h_r : r = 4) (h_l : l = 9) : 
  let circumference := 2 * Real.pi * r in
  let n := (circumference * 180) / (Real.pi * l) in
  n = 160 :=
by
  intro r l h_r h_l
  dsimp [circumference, n]
  rw [h_r, h_l]
  norm_num
  sorry

end cone_central_angle_l168_168221


namespace dot_product_computation_l168_168266

variable {E : Type*} [InnerProductSpace ℝ E]

theorem dot_product_computation
  (u v w : E)
  (hu : ‖u‖ = 2)
  (hv : ‖v‖ = 5)
  (hw : ‖w‖ = 6)
  (h_sum : u + v + w = 0)
  (h_orth : ⟪u, v⟫ = 0) :
  ⟪u, v⟫ + ⟪u, w⟫ + ⟪v, w⟫ = - (65 / 2) :=
by {
  -- sorry is used to skip the proof
  sorry
}

end dot_product_computation_l168_168266


namespace min_value_of_inverse_sum_l168_168598

noncomputable def min_value (a b : ℝ) := ¬(1 ≤ a + 2*b)

theorem min_value_of_inverse_sum (a b : ℝ) (h : a + 2 * b = 1) (h_nonneg : 0 < a ∧ 0 < b) :
  (1 / a + 2 / b) ≥ 9 :=
sorry

end min_value_of_inverse_sum_l168_168598


namespace smallest_digits_to_append_l168_168864

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168864


namespace smallest_append_digits_l168_168834

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168834


namespace expected_value_X_squared_l168_168908

noncomputable def p (x : ℝ) : ℝ := 
  if (0 < x ∧ x < π / 2) then cos x else 0

theorem expected_value_X_squared :
  (∫ x in 0..π / 2, x^2 * p x) = (π^2 / 4 - 2) :=
by
  sorry

end expected_value_X_squared_l168_168908


namespace correct_statement_proof_l168_168876

-- Definitions based on the conditions from part a)
def sampling_method_for_light_bulbs : Prop := 
  "To understand the service life of a batch of light bulbs, a sampling method should be used."

def drawing_red_ball_certain : Prop := 
  "Drawing a red ball from a bag containing only white and red balls is a certain event."

def lottery_probability : Prop := 
  "The probability of winning a certain lottery is 1/1000, so buying 1000 of these lottery tickets will definitely win."

def frequency_estimation : Prop := 
  "When a certain event occurs with a frequency stable around 0.6 in a large number of repeated experiments, estimating the probability of the event occurring as 0.6."

-- The correct answer based on the solution from part b)
def correct_statement := 
  "When a certain event occurs with a frequency stable around 0.6 in a large number of repeated experiments, estimating the probability of the event occurring as 0.6"

theorem correct_statement_proof :
  frequency_estimation = correct_statement :=
by 
  -- Skip the proof steps
  sorry

end correct_statement_proof_l168_168876


namespace distinct_arrangements_in_grid_l168_168381

theorem distinct_arrangements_in_grid : 
  let digits := [0, 1, 2, 3] in
  let boxes := (fin 2).product (fin 2) in
  (digits.permutations.length = 24) :=
by
  sorry

end distinct_arrangements_in_grid_l168_168381


namespace evaluate_fraction_l168_168962

theorem evaluate_fraction : ∃ p q : ℤ, gcd p q = 1 ∧ (2023 : ℤ) / (2022 : ℤ) - 2 * (2022 : ℤ) / (2023 : ℤ) = (p : ℚ) / (q : ℚ) ∧ p = -(2022^2 : ℤ) + 4045 :=
by
  sorry

end evaluate_fraction_l168_168962


namespace smallest_append_digits_l168_168838

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168838


namespace smallest_digits_to_append_l168_168830

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168830


namespace percentage_increase_l168_168021

def old_price : ℝ := 300
def new_price : ℝ := 330

theorem percentage_increase : ((new_price - old_price) / old_price) * 100 = 10 := by
  sorry

end percentage_increase_l168_168021


namespace polynomial_simplification_l168_168086

theorem polynomial_simplification (x : ℝ) :
  x * (x * (x * (3 - x) - 6) + 12) + 2 = -x^4 + 3*x^3 - 6*x^2 + 12*x + 2 := 
by
  sorry

end polynomial_simplification_l168_168086


namespace relationship_between_y_values_l168_168174

variables (a c y1 y2 y3 : ℝ)
-- Definition of the parabola function
def parabola (x : ℝ) : ℝ := a * x ^ 2 - 4 * a * x + c

-- Points on the parabola with given coordinates
def A := (2, y1)
def B := (3, y2)
def C := (-1, y3)

-- Ensure that A, B, and C lie on the parabola
def A_on_parabola : Prop := parabola 2 = y1
def B_on_parabola : Prop := parabola 3 = y2
def C_on_parabola : Prop := parabola (-1) = y3

theorem relationship_between_y_values (ha : a > 0) 
  (hA : A_on_parabola) 
  (hB : B_on_parabola)
  (hC : C_on_parabola) :
  y1 < y2 ∧ y2 < y3 :=
by sorry

end relationship_between_y_values_l168_168174


namespace ab_leq_one_fraction_inequality_l168_168162

-- Part 1: Prove that ab ≤ 1
theorem ab_leq_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) : a * b ≤ 1 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

-- Part 2: Prove that (1/a^3 - 1/b^3) > 3 * (1/a - 1/b) given b > a
theorem fraction_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) (h4 : b > a) :
  1/(a^3) - 1/(b^3) > 3 * (1/a - 1/b) :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end ab_leq_one_fraction_inequality_l168_168162


namespace total_number_of_possible_outcomes_l168_168374

-- Define the conditions
def num_faces_per_die : ℕ := 6
def num_dice : ℕ := 2

-- Define the question as a hypothesis and the answer as the conclusion
theorem total_number_of_possible_outcomes :
  (num_faces_per_die * num_faces_per_die) = 36 := 
by
  -- Provide a proof outline, this is used to skip the actual proof
  sorry

end total_number_of_possible_outcomes_l168_168374


namespace number_divisible_l168_168786

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168786


namespace find_a3_l168_168992

noncomputable def geometric_term (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n-1)

noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (q^n - 1) / (q - 1)

theorem find_a3 (a : ℝ) (q : ℝ) (h_q : q = 3)
  (h_sum : geometric_sum a q 3 + geometric_sum a q 4 = 53 / 3) :
  geometric_term a q 3 = 3 :=
by
  sorry

end find_a3_l168_168992


namespace probability_first_prize_both_distribution_of_X_l168_168607

-- Definitions for the conditions
def total_students : ℕ := 500
def male_students : ℕ := 200
def female_students : ℕ := 300

def male_first_prize : ℕ := 10
def female_first_prize : ℕ := 25

def male_second_prize : ℕ := 15
def female_second_prize : ℕ := 25

def male_third_prize : ℕ := 15
def female_third_prize : ℕ := 40

-- Part (1): Prove the probability that both selected students receive the first prize is 1/240.
theorem probability_first_prize_both :
  (male_first_prize / male_students : ℚ) * (female_first_prize / female_students : ℚ) = 1 / 240 := 
sorry

-- Part (2): Prove the distribution of X.
def P_male_award : ℚ := (male_first_prize + male_second_prize + male_third_prize) / male_students
def P_female_award : ℚ := (female_first_prize + female_second_prize + female_third_prize) / female_students

theorem distribution_of_X :
  ∀ X : ℕ, X = 0 ∧ ((1 - P_male_award) * (1 - P_female_award) = 28 / 50) ∨ 
           X = 1 ∧ ((1 - P_male_award) * (1 - P_female_award) + (P_male_award * (1 - P_female_award)) + ((1 - P_male_award) * P_female_award) = 19 / 50) ∨ 
           X = 2 ∧ (P_male_award * P_female_award = 3 / 50) :=
sorry

end probability_first_prize_both_distribution_of_X_l168_168607


namespace range_of_a_l168_168594

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (a + 3) * x + Real.log x

def f_deriv (x : ℝ) (a : ℝ) : ℝ :=
  2 * x + (a + 3) + 1 / x

-- Prove that if f has a unique extremum point in the interval (1, 2),
-- then the range of a is (-15/2, -6)
theorem range_of_a (a : ℝ) :
  (f_deriv 1 a) * (f_deriv 2 a) < 0 ↔ - (15 / 2) < a ∧ a < -6 :=
by
  sorry

end range_of_a_l168_168594


namespace geralds_apples_l168_168284

theorem geralds_apples (P G : ℕ) (h1 : P = 10) (h2 : ∀ i, P = 3 * G) (h3 : 30 * G = 1200) : G = 40 :=
by sorry

end geralds_apples_l168_168284


namespace volume_of_sphere_l168_168556

theorem volume_of_sphere (r : ℝ) (h : r = 3) : (4 / 3) * π * r ^ 3 = 36 * π := 
by
  sorry

end volume_of_sphere_l168_168556


namespace smallest_digits_to_append_l168_168858

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168858


namespace lilly_and_rosy_total_fish_l168_168667

def total_fish (lilly_fish rosy_fish : ℕ) := lilly_fish + rosy_fish

theorem lilly_and_rosy_total_fish :
  total_fish 10 11 = 21 :=
by simp [total_fish]; sorry

end lilly_and_rosy_total_fish_l168_168667


namespace solve_for_w_l168_168111

theorem solve_for_w (w : ℤ) : 5^6 * 5^w = 25 → w = -4 :=
by
  intro h
  sorry

end solve_for_w_l168_168111


namespace emberly_walks_miles_in_march_l168_168105

theorem emberly_walks_miles_in_march (days_not_walked : ℕ) (total_days_in_march : ℕ) (miles_per_hour : ℕ) (hours_per_walk : ℕ) :
  total_days_in_march = 31 → 
  days_not_walked = 4 → 
  hours_per_walk = 1 → 
  miles_per_hour = 4 → 
  let days_walked := total_days_in_march - days_not_walked in
  let total_hours_walked := days_walked * hours_per_walk in
  let total_miles_walked := total_hours_walked * miles_per_hour in
  total_miles_walked = 108 :=
by {
  intros h1 h2 h3 h4,
  have days_walked_def : days_walked = 31 - 4 := by rw [h1, h2],
  have hours_walked_def : total_hours_walked = (31 - 4) * 1 := by rw [days_walked_def, h3],
  have miles_walked_def : total_miles_walked = (27) * 4 := by rw [hours_walked_def, h4],
  have final_result : total_miles_walked = 108 := by rw miles_walked_def,
  exact final_result,
}

end emberly_walks_miles_in_march_l168_168105


namespace problem_l168_168399

noncomputable def area_of_S (R r : ℝ) (A B C D : ℝ) : ℝ :=
  let d := Math.sqrt(1^2 + 1^2)
  let R := d / 2
  let A_m := Real.pi * (R^2)
  let A_n := Real.pi * (1^2)
  let A_quarter_n := A_n / 4
  A_m - A_quarter_n

theorem problem
  (A B C D : ℝ)
  (m : set Point)
  (n : set Point)
  (h_square : is_inscribed_square (A B C D) m)
  (h_n_radius : radius n = 1)
  (h_n_center : center n = A) :
  area_of_S R r A B C D = Real.pi / 4 :=
by
  sorry

end problem_l168_168399


namespace base_eight_to_base_ten_l168_168357

theorem base_eight_to_base_ten (a b c : ℕ) (ha : a = 1) (hb : b = 5) (hc : c = 7) :
  a * 8^2 + b * 8^1 + c * 8^0 = 111 :=
by 
  -- rest of the proof will go here
  sorry

end base_eight_to_base_ten_l168_168357


namespace correct_statement_for_ellipse_l168_168982

def ellipse := ∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1

theorem correct_statement_for_ellipse :
  let e := 1 / 2 in
  (∀ e, e = 1 / 2) :=
sorry

end correct_statement_for_ellipse_l168_168982


namespace smallest_digits_to_append_l168_168846

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168846


namespace valid_angles_count_l168_168097

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def is_valid_angle (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 2 * Real.pi ∧ ¬ ∃ k : ℤ, θ = k * Real.pi / 3

noncomputable def number_of_valid_angles : ℕ :=
  {θ : ℝ | is_valid_angle θ ∧ (is_arithmetic_sequence (Real.sin θ) (Real.cos θ) (Real.tan θ) ∨
                                is_arithmetic_sequence (Real.cos θ) (Real.sin θ) (Real.tan θ) ∨
                                is_arithmetic_sequence (Real.sin θ) (Real.tan θ) (Real.cos θ))}.to_finset.card

theorem valid_angles_count : number_of_valid_angles = 6 := by sorry

end valid_angles_count_l168_168097


namespace lulu_spent_on_ice_cream_l168_168669

theorem lulu_spent_on_ice_cream : 
  ∃ (x : ℝ), 
  let initial := 65 in 
  let tshirt := (1 / 2) * (initial - x) in 
  let bank := (4 / 5) * tshirt in 
  bank = 24 →
  x = 5 :=
begin
  sorry
end

end lulu_spent_on_ice_cream_l168_168669


namespace product_of_solutions_neg_9_l168_168135

-- Define the condition as an equation
def equation (x : ℝ) := abs x = 3 * (abs x - 2)

-- State the theorem that the product of solutions is -9.
theorem product_of_solutions_neg_9 : 
  let sols := {x | equation x} in
  (∃ a b, a ∈ sols ∧ b ∈ sols ∧ a ≠ b ∧ a * b = -9) :=
begin
  sorry
end

end product_of_solutions_neg_9_l168_168135


namespace real_roots_quadratic_l168_168983

theorem real_roots_quadratic (k : ℝ) : 
  real_roots ((k - 3) * x^2 - 4 * x + 2) ↔ k ≤ 5 :=
begin
  sorry
end

end real_roots_quadratic_l168_168983


namespace solution_l168_168648

def mapping (x : ℝ) : ℝ := x^2

theorem solution (x : ℝ) : mapping x = 4 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end solution_l168_168648


namespace intersection_segment_length_l168_168243

-- Define curve C1: ρ cos θ = 1 in polar coordinates
def C1 (ρ θ : ℝ) : Prop := ρ * cos θ = 1

-- Define curve C2: ρ = 4 cos θ in polar coordinates
def C2 (ρ θ : ℝ) : Prop := ρ = 4 * cos θ

-- Define the length of segment AB where A and B are the intersection points of C1 and C2
def length_AB : ℝ := 2 * sqrt 3

-- The proof problem statement
theorem intersection_segment_length :
  (∀ ρ θ : ℝ, C1 ρ θ ∧ C2 ρ θ) →
  length_AB = 2 * sqrt 3 :=
sorry

end intersection_segment_length_l168_168243


namespace distinct_coloring_schemes_l168_168761

noncomputable def number_of_coloring_schemes (n m : ℕ) : ℕ :=
  if h0 : n % 4 = 0 then
    m * ((m - 1)^(n / 4) + (-1)^(n / 4) * (m - 1))^4
  else if h1 : n % 4 = 1 ∨ n % 4 = 3 then
    m * ((m - 1)^n + (-1)^n * (m - 1))
  else if h2 : n % 4 = 2 then
    m * ((m - 1)^(n / 2) + (-1)^(n / 2) * (m - 1))^2
  else
    0

theorem distinct_coloring_schemes (n m : ℕ) (hn : n ≥ 4) (hm : m ≥ 2) :
  number_of_coloring_schemes n m = 
  if h0 : n % 4 = 0 then
    m * ((m - 1)^(n / 4) + (-1)^(n / 4) * (m - 1))^4
  else if h1 : n % 4 = 1 ∨ n % 4 = 3 then
    m * ((m - 1)^n + (-1)^n * (m - 1))
  else if h2 : n % 4 = 2 then
    m * ((m - 1)^(n / 2) + (-1)^(n / 2) * (m - 1))^2
  else
    0 :=
by
  sorry

end distinct_coloring_schemes_l168_168761


namespace smallest_multiple_of_8_and_9_l168_168360

theorem smallest_multiple_of_8_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 9 = 0) ∧ (∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 9 = 0) → n ≤ m) ∧ n = 72 :=
by
  sorry

end smallest_multiple_of_8_and_9_l168_168360


namespace painting_three_cars_l168_168016

noncomputable def painting_rates := 
  let expert_rate := 1 
  let amateur_rate := 1 / 2 
  let beginner_rate := 1 / 3 
  (expert_rate, amateur_rate, beginner_rate)

theorem painting_three_cars (expert_rate amateur_rate beginner_rate : ℚ): 
  expert_rate = 1 → amateur_rate = 1 / 2 → beginner_rate = 1 / 3 → 
  let combined_rate := expert_rate + amateur_rate + beginner_rate in
  let days := (3 : ℚ) / combined_rate in
  let m := 18 in
  let n := 11 in
  nat.gcd m n = 1 → 
  m + n = 29 := by 
  intros h1 h2 h3
  sorry

end painting_three_cars_l168_168016


namespace wilson_pays_total_l168_168379

def hamburger_price : ℝ := 5
def cola_price : ℝ := 2
def fries_price : ℝ := 3
def sundae_price : ℝ := 4
def discount_coupon : ℝ := 4
def loyalty_discount : ℝ := 0.10

def total_cost_before_discounts : ℝ :=
  2 * hamburger_price + 3 * cola_price + fries_price + sundae_price

def total_cost_after_coupon : ℝ :=
  total_cost_before_discounts - discount_coupon

def loyalty_discount_amount : ℝ :=
  loyalty_discount * total_cost_after_coupon

def total_cost_after_all_discounts : ℝ :=
  total_cost_after_coupon - loyalty_discount_amount

theorem wilson_pays_total : total_cost_after_all_discounts = 17.10 :=
  sorry

end wilson_pays_total_l168_168379


namespace billy_apples_l168_168450

def num_apples_eaten (monday_apples tuesday_apples wednesday_apples thursday_apples friday_apples total_apples : ℕ) : Prop :=
  monday_apples = 2 ∧
  tuesday_apples = 2 * monday_apples ∧
  wednesday_apples = 9 ∧
  friday_apples = monday_apples / 2 ∧
  thursday_apples = 4 * friday_apples ∧
  total_apples = monday_apples + tuesday_apples + wednesday_apples + thursday_apples + friday_apples

theorem billy_apples : num_apples_eaten 2 4 9 4 1 20 := 
by
  unfold num_apples_eaten
  sorry

end billy_apples_l168_168450


namespace tangent_to_x_axis_two_zero_points_l168_168567

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 2) * log x + a * x - 1

theorem tangent_to_x_axis (a : ℝ) (h : a = 1) : 
  ∃ x : ℝ, f x 1 = 0 ∧ x = 1 := 
sorry

theorem two_zero_points (a : ℝ) (h : a < 1) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0 := 
sorry

end tangent_to_x_axis_two_zero_points_l168_168567


namespace locus_of_midpoints_of_segments_locus_of_points_dividing_segments_l168_168030

-- (a) Locus of midpoints of segments XY where X is any point on segment AC and Y is any point on segment B'D'
theorem locus_of_midpoints_of_segments {A C B'D'} (X : ℝ³) (Y : ℝ³) :
  (X ∈ AC) ∧ (Y ∈ B'D') ↔ 
  ∃ (M : ℝ³), (M = (X + Y) / 2) ∧
  (M.z = 1/2) ∧ 
  (0 ≤ M.x) ∧ (M.x ≤ 1) ∧ 
  (0 ≤ M.y) ∧ (M.y ≤ 1) ∧ 
  (M = (1 / 2, 1 / 2, 1 / 2)) := sorry

-- (b) Locus of points Z on segments XY such that ∇ZY = 2∇XZ
theorem locus_of_points_dividing_segments {X Y Z : ℝ³} :
  (Z ∈ (segment X Y)) ∧ (overrightarrow Z Y = 2 • (overrightarrow X Z)) ↔
  (Z = (1 - 2/3) * X + 2/3 * Y) ∧ 
  (0 ≤ Z.x) ∧ (Z.x ≤ 2/3) ∧ 
  (0 ≤ Z.y) ∧ (Z.y ≤ 1) ∧ 
  (Z.z = 2/3) ∧ 
  (Z = (1/3 * t + 2/3 * (1 - u), 1/3 * t + 2/3 * u, 2/3)) := sorry

end locus_of_midpoints_of_segments_locus_of_points_dividing_segments_l168_168030


namespace coefficient_x5_y2_l168_168188

theorem coefficient_x5_y2 (n : ℕ) :
  (∑ k in finset.range (n + 1), binomial n k ) = 32 →
  n = 5 →
  (2 * (n.choose 2) * (3.choose 1) * 2^2) = 120 :=
by
  intro h1 h2
  rw h2 at *
  have h3 : ∑ k in finset.range (6), binomial 5 k = 32 := by
    dsimp
    rw finset.sum_range_succ
    rw finset.sum_range_succ
    rw finset.sum_range_succ
    rw finset.sum_range_succ
    rw finset.sum_range_succ
    rw finset.sum_singleton
    norm_num
  sorry  -- Placeholder for the remaining proof

end coefficient_x5_y2_l168_168188


namespace x_cubed_inverse_cubed_l168_168216

theorem x_cubed_inverse_cubed (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 :=
by sorry

end x_cubed_inverse_cubed_l168_168216


namespace distance_sum_line_parabola_l168_168554

theorem distance_sum_line_parabola :
  let l := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 1 + 1/2 * t ∧ p.2 = (Real.sqrt 3) / 2 * t}
  let C := {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1}
  let M := (1 : ℝ, 0 : ℝ)
  ∃ A B : ℝ × ℝ, A ∈ l ∧ B ∈ l ∧ A ∈ C ∧ B ∈ C ∧
  |(M.1 - A.1)^2 + (M.2 - A.2)^2| + |(M.1 - B.1)^2 + (M.2 - B.2)^2| = 16 / 3 :=
by
  sorry

end distance_sum_line_parabola_l168_168554


namespace speed_of_man_in_still_water_l168_168916

variables {v_m v_s : ℝ}

-- The conditions given in the problem
def downstream_condition : Prop := (v_m + v_s = 24 / 6)
def upstream_condition : Prop := (v_m - v_s = 12 / 6)

-- The mathematical goal derived from the problem
theorem speed_of_man_in_still_water (h1 : downstream_condition) (h2 : upstream_condition) : v_m = 3 := sorry

end speed_of_man_in_still_water_l168_168916


namespace desired_interest_rate_l168_168037

theorem desired_interest_rate 
  (F : ℝ) -- Face value of each share
  (D : ℝ) -- Dividend rate
  (M : ℝ) -- Market value of each share
  (annual_dividend : ℝ := (D / 100) * F) -- Annual dividend per share
  (desired_interest_rate : ℝ := (annual_dividend / M) * 100) -- Desired interest rate
  (F_eq : F = 44) -- Given Face value
  (D_eq : D = 9) -- Given Dividend rate
  (M_eq : M = 33) -- Given Market value
  : desired_interest_rate = 12 := 
by
  sorry

end desired_interest_rate_l168_168037


namespace min_distance_AB_l168_168661

theorem min_distance_AB :
  let A ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, sqrt 3 * x + 1)} in
  let B ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, Real.log x)} in
  let min_dist := 1 + (1 / 4) * Real.log 3 in
  min_dist = (let x := sqrt 3 / 3 in
              let y := - (1 / 2) * Real.log 3 in
              let P := (x, y) in
              abs (sqrt 3 * x - y + 1) / sqrt (3 + 1)) :=
by sorry

end min_distance_AB_l168_168661


namespace sequence_terms_and_sum_l168_168530

noncomputable theory

-- Definitions for sequences and conditions
def a_n (n : ℕ) : ℤ := 2 * n - 1
def b_n (n : ℕ) : ℤ := 2 ^ n
def c_n (n : ℕ) : ℤ := a_n n * b_n n

-- Sum of first n terms of the sequence {c_n}
def S_n (n : ℕ) : ℤ := (Finset.range n).sum (λ k, c_n (k + 1))

-- Lean 4 statement for the proof problem
theorem sequence_terms_and_sum (a_n b_n : ℕ → ℤ) (c_n S_n : ℕ → ℤ)
  (h1 : b_n 1 = a_n 1 + 1) (h2 : b_n 1 = 2) 
  (h3 : b_n 2 = a_n 2 + 1) (h4 : b_n 3 = a_n 4 + 1)
  (a_n_def : ∀ n : ℕ, a_n n = 2 * n - 1)
  (b_n_def : ∀ n : ℕ, b_n n = 2 ^ n)
  (c_n_def : ∀ n : ℕ, c_n n = a_n n * b_n n)
  (S_n_def : ∀ n : ℕ, S_n n = ∑ k in Finset.range n, c_n (k + 1)) :
  S_n n = 6 + (2 * n - 3) * 2^(n + 1) := 
sorry

end sequence_terms_and_sum_l168_168530


namespace geometric_series_common_ratio_l168_168442

theorem geometric_series_common_ratio
    (a : ℝ) (S : ℝ) (r : ℝ)
    (h_a : a = 512)
    (h_S : S = 3072)
    (h_sum : S = a / (1 - r)) : 
    r = 5 / 6 :=
by 
  rw [h_a] at h_sum
  rw [h_S] at h_sum
  sorry

end geometric_series_common_ratio_l168_168442


namespace new_concentration_mixture_l168_168435

theorem new_concentration_mixture :
  ∃ concentration, 
  let vessel1_capacity := 2,
      vessel1_alcohol_percentage := 0.4,
      vessel2_capacity := 6,
      vessel2_alcohol_percentage := 0.6,
      total_vessel_capacity := 10,
      total_initial_liquid := vessel1_capacity + vessel2_capacity,
      remaining_volume := total_vessel_capacity - total_initial_liquid,
      total_alcohol := (vessel1_capacity * vessel1_alcohol_percentage) + (vessel2_capacity * vessel2_alcohol_percentage),
      total_volume := total_vessel_capacity
  in concentration = (total_alcohol / total_volume) * 100 → concentration = 44 := 
by
  sorry

end new_concentration_mixture_l168_168435


namespace p_implies_q_not_q_implies_p_p_sufficient_but_not_necessary_for_q_l168_168579

variables (x : ℝ)

def p := 1 < x ∧ x < 2
def q := 2^x > 1

theorem p_implies_q : p x → q x := sorry
theorem not_q_implies_p : ¬(q x → p x) := sorry

theorem p_sufficient_but_not_necessary_for_q : (p x → q x) ∧ ¬(q x → p x) := sorry

end p_implies_q_not_q_implies_p_p_sufficient_but_not_necessary_for_q_l168_168579


namespace smallest_number_append_l168_168820

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168820


namespace rope_cylinder_solution_l168_168910

def rope_cylinder_problem : Prop :=
  ∃ j k l : ℕ, l.prime ∧
  (let len := 30
     radius := 10
     height := 5
     distance := 5
     rope_length_touch := (j - real.sqrt k) / l
   in rope_length_touch = (30 - real.sqrt 5) / 5) ∧
  j + k + l = 40

theorem rope_cylinder_solution : rope_cylinder_problem :=
by
  sorry

end rope_cylinder_solution_l168_168910


namespace cosine_of_C_l168_168555

theorem cosine_of_C (A B C : ℝ) (a b c : ℝ) 
  (h_perimeter : a + b + c = 9)
  (h_ratio_sines : sin A / sin B = 3 / 2 ∧ sin A / sin C = 3 / 4) :
  cos C = -1 / 4 := 
by
  sorry

end cosine_of_C_l168_168555


namespace find_t_eq_twenty_over_three_l168_168494

-- Define the slope calculation as a function of two points.
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the points involved in the problem.
def p1 : ℝ × ℝ := (0, 2)
def p2 : ℝ × ℝ := (-4, -1)
def p3 (t : ℝ) : ℝ × ℝ := (t, 7)

-- Define the line relationship we need to prove.
theorem find_t_eq_twenty_over_three (t : ℝ) :
  slope p1 p2 = slope p1 (p3 t) → t = 20 / 3 :=
by
  sorry

end find_t_eq_twenty_over_three_l168_168494


namespace trig_identity_l168_168991

theorem trig_identity (θ : ℝ) 
  (h : sin θ + cos θ = -((sqrt 5) / 3)) : 
  cos (2 * θ - 7 * π / 2) = 4 / 9 := 
by 
  sorry

end trig_identity_l168_168991


namespace smallest_digits_to_append_l168_168772

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168772


namespace area_JKLMNO_l168_168677

def JK : ℝ := 8
def KL : ℝ := 10
def OP : ℝ := 6
def PM : ℝ := 3

theorem area_JKLMNO :
  let area_JKLMNP := JK * KL
  let area_PMNO := PM * OP
  let area_JKLMNO := area_JKLMNP - area_PMNO
  area_JKLMNO = 62 :=
by
  let area_JKLMNP := JK * KL
  let area_PMNO := PM * OP
  let area_JKLMNO := area_JKLMNP - area_PMNO
  show area_JKLMNO = 62
  rfl

end area_JKLMNO_l168_168677


namespace combined_initial_weight_l168_168926

variable (B P : ℝ)
variable (final_beef_weight final_pork_weight : ℝ)
variable (B_weight_loss_stages P_weight_loss_stages : List ℝ)

-- Definitions of the final weights post-processing:
def beef_weight_after_stages : ℝ := B * List.prod B_weight_loss_stages
def pork_weight_after_stages : ℝ := P * List.prod P_weight_loss_stages

-- Conditions as hypotheses:
axiom Beef_final : B_weight_loss_stages = [0.90, 0.85, 0.80, 0.75, 0.70] ∧ beef_weight_after_stages B_weight_loss_stages B = 295.88
axiom Pork_final : P_weight_loss_stages = [0.95, 0.90, 0.80, 0.70, 0.75] ∧ pork_weight_after_stages P_weight_loss_stages P = 204.66

theorem combined_initial_weight :
  B + P = 1506 := by
  sorry

end combined_initial_weight_l168_168926


namespace sum_over_triples_l168_168460

theorem sum_over_triples :
  (∑ a in (range 1000).filter (λ a, 1 ≤ a), 
     ∑ b in (range 1000).filter (λ b, a < b),
       ∑ c in (range 1000).filter (λ c, b < c), 
         (1 / (3^a * 5^b * 7^c))) = 1 / 21216 := 
by
  sorry

end sum_over_triples_l168_168460


namespace smallest_lambda_exists_l168_168504

open_locale big_operators

theorem smallest_lambda_exists :
  ∃ (λ : ℝ), (λ = 2 / 2024) ∧ ∀ (n : ℕ),
    n > 0 →
    ∃ (x : fin 2023 → ℕ),
      (∏ i, x i = n) ∧
      (∀ i, prime (x i) ∨ (1 ≤ x i ∧ x i ≤ ⌊n^λ⌋)) :=
sorry

end smallest_lambda_exists_l168_168504


namespace equilateral_triangle_angle_bisector_l168_168894

theorem equilateral_triangle_angle_bisector (A B C K M N : Point)
  (h1 : is_equilateral_triangle A B C)
  (h2 : K ∈ segment A B)
  (h3 : M ∈ segment B C)
  (h4 : N ∈ segment A C)
  (h5 : ∠M K B = ∠M N C)
  (h6 : ∠K M B = ∠K N A) :
  is_angle_bisector N B (∠M N K) :=
sorry

end equilateral_triangle_angle_bisector_l168_168894


namespace cos_arith_seq_a3_a7_l168_168540

theorem cos_arith_seq_a3_a7 (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum : a 1 + a 5 + a 9 = 8 * real.pi) :
  real.cos (a 3 + a 7) = -1 / 2 :=
by 
  sorry

end cos_arith_seq_a3_a7_l168_168540


namespace ages_of_boys_l168_168728

theorem ages_of_boys (a b c : ℕ) (h : a + b + c = 29) (h₁ : a = b) (h₂ : c = 11) : a = 9 ∧ b = 9 := 
by
  sorry

end ages_of_boys_l168_168728


namespace find_a4_l168_168239

variable {α : Type*} [AddCommGroupWithOne α]

-- Define the arithmetic sequence based on its properties
def arithmetic_sequence (a d : α) (n : ℕ) : α := a + n * d

noncomputable def a_n (a d : α) := λ n, arithmetic_sequence a d (n - 1)

-- The condition given in the problem
axiom condition (a d : α) : a_n a d 3 + a_n a d 5 = 2

-- The target to prove
theorem find_a4 (a d : α) (h : a_n a d 3 + a_n a d 5 = 2) : a_n a d 4 = 1 :=
by
  sorry -- Proof to be filled in


end find_a4_l168_168239


namespace CD_eq_AB_l168_168002

theorem CD_eq_AB (x1 y1 x2 y2 x3 y3 : ℤ) (h1 : x2 = x1 + 2) (h2 : y2 = y1 + 3) :
  let A := (x1, y1), B := (x2, y2), C := (x3, y3), D := (x3 + 2, y3 + 3) in
  (x3 + 2 - x3) = (x2 - x1) ∧ (y3 + 3 - y3) = (y2 - y1) ∧
  (abs ((x2 - x1)^2 + (y2 - y1)^2) = abs ((x3 + 2 - x3)^2 + (y3 + 3 - y3)^2)) :=
begin
  intros,
  split,
  { exact eq.trans (sub_self x3) (eq.trans h1 (add_sub_cancel' 2 x1).symm) },
  split,
  { exact eq.trans (sub_self y3) (eq.trans h2 (add_sub_cancel' 3 y1).symm) },
  { rw [← add_sub_cancel' (x2 - x1)^2 (y2 - y1)^2, ← abs_eq],
    apply congr_arg,
    exact eq.trans (add_comm _ (y2 - y1)^2) (eq.trans h1 h2).symm }
end

end CD_eq_AB_l168_168002


namespace value_of_a_minus_b_l168_168217

theorem value_of_a_minus_b (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = a + b) :
  a - b = 2 ∨ a - b = 6 :=
sorry

end value_of_a_minus_b_l168_168217


namespace max_value_A_l168_168150

theorem max_value_A (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( ( (x - y) * Real.sqrt (x^2 + y^2) + (y - z) * Real.sqrt (y^2 + z^2) + (z - x) * Real.sqrt (z^2 + x^2) + Real.sqrt 2 ) / 
    ( (x - y)^2 + (y - z)^2 + (z - x)^2 + 2 ) ) ≤ 1 / Real.sqrt 2 :=
sorry

end max_value_A_l168_168150


namespace pairs_nat_eq_l168_168113

theorem pairs_nat_eq (n k : ℕ) :
  (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) :=
by
  sorry

end pairs_nat_eq_l168_168113


namespace wage_tax_deduction_l168_168936

variable (hourlyWage : ℕ) (taxRate : ℚ)
variable (centsPerDollar : ℕ := 100)

theorem wage_tax_deduction
  (h1: hourlyWage = 2500)
  (h2: taxRate = 0.018) :
  hourlyWage * (taxRate.toRational) = 45 := 
by
  rw [h1, h2]
  norm_num
  sorry

end wage_tax_deduction_l168_168936


namespace common_difference_is_2_l168_168331

variable {a_n : ℕ → ℝ}
variable {d : ℝ}

-- Define the conditions as given in the problem
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) - a_n n = d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a_n i

-- The problem conditions
axiom a1_condition : a_n 1 + a_n 5 = 10
axiom s4_condition : sum_of_first_n_terms a_n 4 = 16

-- The statement to prove
theorem common_difference_is_2 : is_arithmetic_sequence a_n d → d = 2 :=
by
  sorry

end common_difference_is_2_l168_168331


namespace number_of_students_playing_sports_students_not_playing_any_sports_l168_168231

def total_students : ℕ := 40
def students_basketball : ℕ := 15
def students_cricket : ℕ := 20
def students_baseball : ℕ := 12
def students_basketball_and_cricket : ℕ := 5
def students_cricket_and_baseball : ℕ := 7
def students_basketball_and_baseball : ℕ := 3
def students_all_three : ℕ := 2

theorem number_of_students_playing_sports (total students_b students_c students_ba students_bc students_cba students_bba students_bcb : ℕ) :
  students_b + students_c + students_ba - students_bc - students_cba - students_bba + students_bcb = 32 :=
begin
  have total := students_b + students_c + students_ba - students_bc - students_cba - students_bba + students_bcb,
  have heq : total = 32,
  sorry
end

theorem students_not_playing_any_sports (total students_b students_c students_ba students_bc students_cba students_bba students_bcb : ℕ) :
  total - (students_b + students_c + students_ba - students_bc - students_cba - students_bba + students_bcb) = 8 :=
begin
  have students_playing := students_b + students_c + students_ba - students_bc - students_cba - students_bba + students_bcb,
  have heq : total - students_playing = 8,
  sorry
end

end number_of_students_playing_sports_students_not_playing_any_sports_l168_168231


namespace perpendicular_vectors_m_eq_0_or_neg2_l168_168581

theorem perpendicular_vectors_m_eq_0_or_neg2
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (1, m - 1))
  (h : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) :
  m = 0 ∨ m = -2 := sorry

end perpendicular_vectors_m_eq_0_or_neg2_l168_168581


namespace quad_common_root_l168_168129

theorem quad_common_root (a b c d : ℝ) :
  (∃ α : ℝ, α^2 + a * α + b = 0 ∧ α^2 + c * α + d = 0) ↔ (a * d - b * c) * (c - a) = (b - d)^2 ∧ (a ≠ c) := 
sorry

end quad_common_root_l168_168129


namespace find_new_spherical_coordinates_l168_168038

noncomputable def original_spherical_coordinates : (ℝ × ℝ × ℝ) := (3, 6 * Real.pi / 5, Real.pi / 4)

def original_rectangular_coordinates (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

def transformed_rectangular_coordinates (x y z : ℝ) : (ℝ × ℝ × ℝ) :=
  (-y, x, -z)

def spherical_coordinates_of_transformed (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  let (x, y, z) := original_rectangular_coordinates ρ θ φ in
  let (tx, ty, tz) := transformed_rectangular_coordinates x y z in
  (Real.sqrt (tx^2 + ty^2 + tz^2),
   Real.arctan2 ty tx,
   Real.acos (tz / ρ))

theorem find_new_spherical_coordinates :
  spherical_coordinates_of_transformed 3 (6 * Real.pi / 5) (Real.pi / 4) =
  (3, 11 * Real.pi / 10, 3 * Real.pi / 4) :=
by
  sorry

end find_new_spherical_coordinates_l168_168038


namespace max_value_A_l168_168149

theorem max_value_A (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( ( (x - y) * Real.sqrt (x^2 + y^2) + (y - z) * Real.sqrt (y^2 + z^2) + (z - x) * Real.sqrt (z^2 + x^2) + Real.sqrt 2 ) / 
    ( (x - y)^2 + (y - z)^2 + (z - x)^2 + 2 ) ) ≤ 1 / Real.sqrt 2 :=
sorry

end max_value_A_l168_168149


namespace midpoint_polar_coordinates_l168_168611

theorem midpoint_polar_coordinates :
  ∃ (r θ : ℝ), 0 < r ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 
    (let A : ℝ × ℝ := (10, π / 4)
     let B : ℝ × ℝ := (10, 3 * π / 4)
     let Ax := 10 * Real.cos (π / 4)
     let Ay := 10 * Real.sin (π / 4)
     let Bx := 10 * Real.cos (3 * π / 4)
     let By := 10 * Real.sin (3 * π / 4)
     let Mx := (Ax + Bx) / 2
     let My := (Ay + By) / 2
     let Mr := Real.sqrt (Mx^2 + My^2)
     let Mθ := Real.arctan2 My Mx 
     Mr = r ∧ Mθ = θ) :=
sorry

end midpoint_polar_coordinates_l168_168611


namespace smallest_number_of_digits_to_append_l168_168809

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168809


namespace find_f_log3_54_l168_168719

noncomputable def f (x : ℝ) : ℝ :=  if h : (0 < x ∧ x < 1) then 3^x + 1/2 else sorry

theorem find_f_log3_54 (x : ℝ) (h₁ : x ∈ (0, 1)) (h₂ : ∀ x, f (x + 2) = -f x) (h₃ : f (log 3 54) = -2): 
  f (log 3 54) = -2 := by
  sorry

end find_f_log3_54_l168_168719


namespace Emily_total_points_l168_168960

-- Definitions of the points scored in each round
def round1_points := 16
def round2_points := 32
def round3_points := -27
def round4_points := 92
def round5_points := 4

-- Total points calculation in Lean
def total_points := round1_points + round2_points + round3_points + round4_points + round5_points

-- Lean statement to prove total points at the end of the game
theorem Emily_total_points : total_points = 117 :=
by 
  -- Unfold the definition of total_points and simplify
  unfold total_points round1_points round2_points round3_points round4_points round5_points
  -- Simplify the expression
  sorry

end Emily_total_points_l168_168960


namespace sin_210_eq_neg_half_l168_168896

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 :=
by 
  sorry

end sin_210_eq_neg_half_l168_168896


namespace lowest_score_56_l168_168701

theorem lowest_score_56
  (s : Fin 15 → ℝ)
  (mean_fifteen : (∑ i, s i) / 15 = 90)
  (mean_thirteen : (∑ i in Finset.filter (λ i, s i ≠ max (Finset.univ.map s)) s) / 13 = 92)
  (max_score : max (Finset.univ.map s) = 98) :
  min (Finset.univ.map s) = 56 :=   
sorry

end lowest_score_56_l168_168701


namespace coeff_x2_expansion_l168_168622

theorem coeff_x2_expansion : 
  (∃ c : ℝ, c * x ^ 2 = (x * (x - 1) * (x + 1) ^ 4).as_poly.coeff 2) ∧ 
  c = 5 :=
by sorry

end coeff_x2_expansion_l168_168622


namespace symmetric_point_xOy_plane_l168_168146

def symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

theorem symmetric_point_xOy_plane : 
  symmetric_point (2, 3, 4) = (2, 3, -4) :=
by
  -- We define the symmetric point function and prove the statement.
  rfl

end symmetric_point_xOy_plane_l168_168146


namespace part_a_part_b_part_c_l168_168476

noncomputable def f : ℕ → ℕ
| 1          := 1
| (n + 2) := if f (n + 1) > n + 2 then f (n + 1) - (n + 2) else f (n + 1) + (n + 2)

def S : Set ℕ := { n | f n = 1993 }

theorem part_a : Set.Infinite S :=
sorry

theorem part_b : ∃ n ∈ S, ∀ m ∈ S, n ≤ m :=
sorry

theorem part_c : tendsto (λ i, (n i.succ) / (n i).to_rat) at_top (𝓝 3) :=
sorry

end part_a_part_b_part_c_l168_168476


namespace proof_problem_l168_168278

open Set

variable (U A B : Set ℕ)
variable [DecidableEq ℕ]

noncomputable def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem proof_problem :
  U = {1, 2, 3, 4, 5, 6, 7, 8} →
  (A ∩ complement_U B) = {1, 3} →
  complement_U (A ∪ B) = {2, 4} →
  B = {5, 6, 7, 8} :=
by
  intros hU hA hB
  sorry

end proof_problem_l168_168278


namespace greatest_ln_2_l168_168013

theorem greatest_ln_2 (x1 x2 x3 x4 : ℝ) (h1 : x1 = (Real.log 2) ^ 2) (h2 : x2 = Real.log (Real.log 2)) (h3 : x3 = Real.log (Real.sqrt 2)) (h4 : x4 = Real.log 2) 
  (h5 : Real.log 2 < 1) : 
  x4 = max x1 (max x2 (max x3 x4)) := by 
  sorry

end greatest_ln_2_l168_168013


namespace silvia_savings_l168_168692

def retail_price : ℝ := 1000
def guitar_center_discount_rate : ℝ := 0.15
def sweetwater_discount_rate : ℝ := 0.10
def guitar_center_shipping_fee : ℝ := 100
def sweetwater_shipping_fee : ℝ := 0

def guitar_center_cost : ℝ := retail_price * (1 - guitar_center_discount_rate) + guitar_center_shipping_fee
def sweetwater_cost : ℝ := retail_price * (1 - sweetwater_discount_rate) + sweetwater_shipping_fee

theorem silvia_savings : guitar_center_cost - sweetwater_cost = 50 := by
  sorry

end silvia_savings_l168_168692


namespace smallest_digits_to_append_l168_168855

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168855


namespace basketball_weight_calc_l168_168488

-- Define the variables and conditions
variable (weight_basketball weight_watermelon : ℕ)
variable (h1 : 8 * weight_basketball = 4 * weight_watermelon)
variable (h2 : weight_watermelon = 32)

-- Statement to prove
theorem basketball_weight_calc : weight_basketball = 16 :=
by
  sorry

end basketball_weight_calc_l168_168488


namespace sum_of_n_l168_168507

theorem sum_of_n (sum_n : ℕ) :
  sum_n = ∑ n in (Finset.filter (λ n, 
    let y_squared := n^2 - 18 * n + 80 in 
    let y := Int.sqrt y_squared ^ 2 in
    y_squared = y^2 ∧ 15 % n = 0) (Finset.Icc 1 15)), 
 0 :=
by
  sorry

end sum_of_n_l168_168507


namespace zero_point_in_interval_l168_168315

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + 2 * x - 1

theorem zero_point_in_interval :
  (∃ x : ℝ, f x = 0 ∧ (1 / 2) < x ∧ x < 1) :=
by {
  -- Definitions and conditions provided in a)
  let a := 1 / 2,
  let b := 1,

  -- Assertions based on values provided in b)
  have h1 : f (1 / 2) < 0,
  { exact (by norm_num : f (1 / 2) = -1), },

  have h2 : f 1 > 0,
  { exact (by norm_num : f 1 = 1), },

  -- Applying the intermediate value theorem for continuity and monotonic increase
  exact exists_zero_of_continuous_increasing h1 h2,
}

end zero_point_in_interval_l168_168315


namespace circumcenter_iff_perimeter_l168_168650

variables 
  (O A B C A₁ B₁ C₁ : Type) 
  [euclidean_geometry.basic A B C] 
  [euclidean_geometry.basic A₁ B₁ C₁] 
  [incidence.is_interior O (triangle ABC)]
  (h₁ : is_perpendicular (line_through O A₁) (line_through B C))
  (h₂ : is_perpendicular (line_through O B₁) (line_through C A))
  (h₃ : is_perpendicular (line_through O C₁) (line_through A B))

-- Prove that O is the circumcenter of triangle ABC if and only if the perimeter of triangle A₁ B₁ C₁
-- is not less than the perimeter of any of the triangles AB₁ C₁, BC₁ A₁, and CA₁ B₁.
theorem circumcenter_iff_perimeter 
  (h₁ : is_perpendicular (line_through O A₁) (line_through B C))
  (h₂ : is_perpendicular (line_through O B₁) (line_through C A))
  (h₃ : is_perpendicular (line_through O C₁) (line_through A B)) :
  is_circumcenter O (triangle ABC) ↔ 
    (perimeter (triangle A₁ B₁ C₁) ≥ perimeter (triangle A B₁ C₁) ∧
     perimeter (triangle A₁ B₁ C₁) ≥ perimeter (triangle B C₁ A₁) ∧
     perimeter (triangle A₁ B₁ C₁) ≥ perimeter (triangle C A₁ B₁)) :=
sorry

end circumcenter_iff_perimeter_l168_168650


namespace number_of_sequences_l168_168172

def a₁ : ℤ := 2019
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n = a₁ + n * d
def satisfies_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, a₁ + (finset.range n).sum a = a m

theorem number_of_sequences :
  ∃ a : (ℕ → ℤ), is_arithmetic_sequence a ∧ satisfies_condition a ∧ (∃! k, k = 5) :=
sorry

end number_of_sequences_l168_168172


namespace smallest_append_digits_l168_168832

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168832


namespace volume_of_cone_l168_168943

theorem volume_of_cone (d h : ℝ) (d_eq : d = 12) (h_eq : h = 9) : 
  (1 / 3) * π * (d / 2)^2 * h = 108 * π := 
by 
  rw [d_eq, h_eq] 
  sorry

end volume_of_cone_l168_168943


namespace y1_gt_y2_of_points_on_linear_function_l168_168597

theorem y1_gt_y2_of_points_on_linear_function :
  (∀ (y1 y2 : ℝ), y1 = -(-2) + 1 ∧ y2 = -(2) + 1 → y1 > y2) :=
by
  intros y1 y2 h
  cases h with h1 h2
  rw [h1, h2]
  norm_num
  sorry

end y1_gt_y2_of_points_on_linear_function_l168_168597


namespace smallest_digits_to_append_l168_168774

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168774


namespace constant_a_unique_max_l168_168227

def f (a x : ℝ) : ℝ := (2 + a) * Real.sin (x + Real.pi / 4)

theorem constant_a_unique_max (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 3) ∧ (∃ x : ℝ, f a x = 3) →
  a = 1 := by
  sorry

end constant_a_unique_max_l168_168227


namespace find_b_card_l168_168744

def card_a_card_b_same (a b : ℕ × ℕ) : Prop :=
a.1 = b.1 ∨ a.1 = b.2 ∨ a.2 = b.1 ∨ a.2 = b.2

def card_b_card_c_same (b c : ℕ × ℕ) : Prop :=
b.1 = c.1 ∨ b.1 = c.2 ∨ b.2 = c.1 ∨ b.2 = c.2

def sum_cards_not_five (c : ℕ × ℕ) : Prop :=
c.1 + c.2 ≠ 5

def cards : list (ℕ × ℕ) := [(1,2), (1,3), (2,3)]

theorem find_b_card (a_card b_card c_card : ℕ × ℕ)
    (h_cards : a_card ∈ cards ∧ b_card ∈ cards ∧ c_card ∈ cards)
    (h_diff_ab : ¬card_a_card_b_same a_card b_card → ¬card_a_card_b_same b_card a_card)
    (h_diff_bc : ¬card_b_card_c_same b_card c_card → ¬card_b_card_c_same c_card b_card)
    (h_no_five_sum : sum_cards_not_five c_card) :
  b_card = (2,3) :=
by
  sorry

end find_b_card_l168_168744


namespace tangent_line_circle_intersect_line_circle_l168_168226

-- Prove that if the line 4x + 3y + a = 0 is tangent to the circle x^2 + y^2 = 4, then a = ±10
theorem tangent_line_circle (a : ℝ) :
  (∀ (x y : ℝ), (4 * x + 3 * y + a = 0)) ∧ (∀ (x y : ℝ), (x^2 + y^2 = 4)) →
  a = 10 ∨ a = -10 :=
  sorry

-- Prove that if the line 4x + 3y + a = 0 intersects the circle x^2 + y^2 = 4 at points A and B
-- and |AB| = 2√3, then a = ±5
theorem intersect_line_circle (a : ℝ) :
  (∀ (x y : ℝ), (4 * x + 3 * y + a = 0)) ∧ (∀ (x y : ℝ), (x^2 + y^2 = 4)) ∧
  (∃ (A B : ℝ × ℝ), A ≠ B ∧ (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧
    (4 * A.1 + 3 * A.2 + a = 0) ∧ (4 * B.1 + 3 * B.2 + a = 0) ∧
    (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * real.sqrt 3)) →
  a = 5 ∨ a = -5 :=
  sorry

end tangent_line_circle_intersect_line_circle_l168_168226


namespace total_shirts_l168_168520

theorem total_shirts (blue_shirts green_shirts total_shirts : ℕ) (h₁ : blue_shirts = 6) (h₂ : green_shirts = 17) (h₃ : total_shirts = blue_shirts + green_shirts) : total_shirts = 23 :=
by
  rw [h₁, h₂] at h₃ -- Replace blue_shirts and green_shirts with their values in the expression total_shirts = blue_shirts + green_shirts
  exact h₃.trans (nat.add_comm _ _).symm -- Final conclusion by transitivity and commutativity

end total_shirts_l168_168520


namespace maximize_product_of_first_n_terms_l168_168993

namespace GeometricSequence

def first_term : ℕ → ℝ := λ n, 3 * (2 / 5) ^ (n - 1)

def product_of_first_n_terms (n : ℕ) : ℝ :=
  ∏ i in finset.range n, first_term (i + 1)

theorem maximize_product_of_first_n_terms :
  argmax (product_of_first_n_terms) = 3 :=
sorry

end GeometricSequence

end maximize_product_of_first_n_terms_l168_168993


namespace sequence_length_l168_168456

theorem sequence_length 
  (a : ℕ)
  (b : ℕ)
  (d : ℕ)
  (steps : ℕ)
  (h1 : a = 160)
  (h2 : b = 28)
  (h3 : d = 4)
  (h4 : (28:ℕ) = (160:ℕ) - steps * 4) :
  steps + 1 = 34 :=
by
  sorry

end sequence_length_l168_168456


namespace smallest_digits_to_append_l168_168870

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168870


namespace smallest_digits_to_append_l168_168850

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168850


namespace smallest_digits_to_append_l168_168856

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168856


namespace calculate_total_calories_l168_168032

-- Definition of variables and conditions
def total_calories (C : ℝ) : Prop :=
  let FDA_recommended_intake := 25
  let consumed_calories := FDA_recommended_intake + 5
  (3 / 4) * C = consumed_calories

-- Theorem statement
theorem calculate_total_calories : ∃ C : ℝ, total_calories C ∧ C = 40 :=
by
  sorry  -- Proof will be provided here

end calculate_total_calories_l168_168032


namespace series_sum_is_correct_l168_168143

noncomputable def series_sum : ℝ := ∑' k, 5^((2 : ℕ)^k) / (25^((2 : ℕ)^k) - 1)

theorem series_sum_is_correct : series_sum = 1 / (Real.sqrt 5 - 1) := 
by
  sorry

end series_sum_is_correct_l168_168143


namespace product_of_solutions_neg_9_l168_168136

-- Define the condition as an equation
def equation (x : ℝ) := abs x = 3 * (abs x - 2)

-- State the theorem that the product of solutions is -9.
theorem product_of_solutions_neg_9 : 
  let sols := {x | equation x} in
  (∃ a b, a ∈ sols ∧ b ∈ sols ∧ a ≠ b ∧ a * b = -9) :=
begin
  sorry
end

end product_of_solutions_neg_9_l168_168136


namespace tangent_BD_circumcircle_ADZ_l168_168056

noncomputable section

-- Define the setup
variable {A B C D E X Y Z : Point} -- Points in the plane
variable {circumcircle : Circle} -- Circumcircle of triangle ABC

-- Define the conditions
variables (h₁ : ∠A = 90)
           (h₂ : isTangentAt A circumcircle line_BC D)
           (h₃ : reflection_line_BC A = E)
           (h₄ : foot_perpendicular A line_BE = X)
           (h₅ : midpoint AX = Y)
           (h₆ : line_BY_meets_circumcircle_at_B_again Z)

-- The statement to be proved
theorem tangent_BD_circumcircle_ADZ
  (h₁ : ∠A = 90)
  (h₂ : isTangentAt A circumcircle line_BC D)
  (h₃ : reflection_line_BC A = E)
  (h₄ : foot_perpendicular A line_BE = X)
  (h₅ : midpoint AX = Y)
  (h₆ : line_BY_meets_circumcircle_at_B_again Z) :
  isTangentAt D (circumcircle_triangle AD Z) line_BD :=
sorry

end tangent_BD_circumcircle_ADZ_l168_168056


namespace systematic_sampling_twentieth_number_l168_168610

theorem systematic_sampling_twentieth_number (n m k first_num twentieth_num : ℕ) 
(h1 : n = 1000) (h2 : m = 50) (h3 : k = n / m) 
(h4 : first_num = 15) (h5 : twentieth_num = 395) :
twentieth_num = first_num + 19 * k :=
by {
  rw [h1, h2, h3, h4],
  norm_num,
  exact h5,
}

end systematic_sampling_twentieth_number_l168_168610


namespace sum_of_coordinates_of_other_endpoint_l168_168674

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (h1 : (1 + x) / 2 = 5)
  (h2 : (2 + y) / 2 = 6) :
  x + y = 19 :=
by
  sorry

end sum_of_coordinates_of_other_endpoint_l168_168674


namespace ages_of_boys_l168_168730

theorem ages_of_boys (a b c : ℕ) (h1 : a + b + c = 29) (h2 : a = b) (h3 : c = 11) : a = 9 :=
by
  sorry

end ages_of_boys_l168_168730


namespace product_of_solutions_neg_9_l168_168137

-- Define the condition as an equation
def equation (x : ℝ) := abs x = 3 * (abs x - 2)

-- State the theorem that the product of solutions is -9.
theorem product_of_solutions_neg_9 : 
  let sols := {x | equation x} in
  (∃ a b, a ∈ sols ∧ b ∈ sols ∧ a ≠ b ∧ a * b = -9) :=
begin
  sorry
end

end product_of_solutions_neg_9_l168_168137


namespace sum_distinct_x_values_l168_168566

def g (x : ℝ) : ℝ := -x^2 + 6 * x - 8

theorem sum_distinct_x_values (x1 x2 : ℝ) (h1 : g(g(g(x1))) = 2) (h2 : g(g(g(x2))) = 2) :
  x1 = 1 ∨ x1 = 5 ∧ x2 = 1 ∨ x2 = 5 → x1 ≠ x2 → x1 + x2 = 6 :=
by
  -- complete proof skipped
  sorry

end sum_distinct_x_values_l168_168566


namespace theta_sum_of_complex_numbers_l168_168742

theorem theta_sum_of_complex_numbers (n : ℕ) (h1 : ∀ z : ℂ, z^24 - z^12 - 1 = 0 ∧ |z| = 1) (h2 : ∀ m : ℕ, 0 ≤ m ∧ m < 2 * n) :
  ∃ (θ : Fin (2 * n) → ℝ), (∀ m, z_m = Complex.exp (θ m * Complex.i)) ∧
                         (0 ≤ θ (↑m.cast_lt (by linarith)) < 360) ∧
                         ((Σ m in Finset.Ico 0 n, θ (bit0 m+1)) = 960) :=
sorry

end theta_sum_of_complex_numbers_l168_168742


namespace smallest_digits_to_append_l168_168863

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168863


namespace smallest_digits_to_append_l168_168862

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168862


namespace sequence_periodic_9_l168_168417

noncomputable def recurrence_f : ℤ × ℤ → ℤ × ℤ := λ p, (|p.1| - p.2, p.1)

theorem sequence_periodic_9 (x0 x1 : ℤ) (x : ℕ → ℤ)
  (h : ∀ n > 1, x (n + 1) = |x n| - x (n - 1)) :
  ∃ p : ℕ, p = 9 ∧ ∀ n, x (n + 9) = x n :=
sorry

end sequence_periodic_9_l168_168417


namespace u_series_sum_l168_168768

noncomputable def u0 : ℝ × ℝ := (2, 2)
noncomputable def z0 : ℝ × ℝ := (3, 1)
noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ := 
  let dot := v.1 * w.1 + v.2 * w.2 in
  let normsq := w.1 * w.1 + w.2 * w.2 in
  (dot / normsq * w.1, dot / normsq * w.2)

noncomputable def infinite_sum : ℝ × ℝ := 
  let rec u (n : ℕ) : ℝ×ℝ :=
    if n = 0 then u0 else proj (z (n-1)) u0
  and z (n : ℕ) : ℝ×ℝ :=
    if n = 0 then z0 else proj (u n) z0 in
  let rec sum_aux (n : ℕ) (partial_sum : ℝ × ℝ) : ℝ × ℝ :=
    let current_u := u n in
    let new_partial_sum := (partial_sum.1 + current_u.1, partial_sum.2 + current_u.2) in
    if n = 1 then new_partial_sum else sum_aux (n-1) new_partial_sum in
  sum_aux 100 (0, 0) -- using an approximation with 100 terms

theorem u_series_sum : infinite_sum = (10, 10) :=
sorry

end u_series_sum_l168_168768


namespace train_cross_time_l168_168050

-- Define the given conditions
def length_of_train : ℕ := 110
def length_of_bridge : ℕ := 265
def speed_kmh : ℕ := 45

-- Convert speed to m/s
def speed_ms : ℝ := speed_kmh * 1000 / 3600

-- Define the total distance the train needs to travel
def total_distance : ℕ := length_of_train + length_of_bridge

-- Calculate the time it takes to cross the bridge
noncomputable def time_to_cross : ℝ := total_distance / speed_ms

-- State the theorem
theorem train_cross_time : time_to_cross = 30 := by sorry

end train_cross_time_l168_168050


namespace find_litres_acai_berry_juice_l168_168316

noncomputable def cost_mixed_fruit_juice := 262.85
noncomputable def cost_acai_berry_juice := 3104.35
noncomputable def cost_cocktail := 1399.45
noncomputable def litres_mixed_fruit_juice := 37
noncomputable def total_cost_mixed_fruit := litres_mixed_fruit_juice * cost_mixed_fruit_juice

def litres_acai_berry_juice_to_add (x : ℝ) : Prop :=
  total_cost_mixed_fruit + x * cost_acai_berry_juice = (litres_mixed_fruit_juice + x) * cost_cocktail

theorem find_litres_acai_berry_juice : ∃ x : ℝ, x ≈ 24.68 ∧ litres_acai_berry_juice_to_add x :=
sorry

end find_litres_acai_berry_juice_l168_168316


namespace extremely_powerful_count_l168_168945

def is_extremely_powerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ b % 2 = 1 ∧ a^b = n

noncomputable def count_extremely_powerful_below (m : ℕ) : ℕ :=
  Nat.card { n : ℕ | is_extremely_powerful n ∧ n < m }

theorem extremely_powerful_count : count_extremely_powerful_below 5000 = 19 :=
by
  sorry

end extremely_powerful_count_l168_168945


namespace minimum_degree_P_l168_168995

noncomputable def P (n : ℕ) (x y : ℝ) : ℝ := 1 / (1 + x + y)

theorem minimum_degree_P (n : ℕ) (hx : ∀ x : ℝ, 0 ≤ x ∧ x ≤ n)
  (hy : ∀ y : ℝ, 0 ≤ y ∧ y ≤ n):
  let P (x y : ℝ) := 1 / (1 + x + y) in
  ∀ P : ℝ → ℝ → ℝ, (∀ x y, x ∈ ({0, 1, 2, ..., n}: Set ℕ) ∧ y ∈ ({0, 1, 2, ..., n}: Set ℕ) → 
    P x y = 1 / (1 + x + y)) → 
    ∃ k ≤ 2 * n, ∀ a b, a + b = k →
    P x y = A_n(x) * y^n + A_{n-1}(x) * y^(n-1) + ... + A_0(x) :=
sorry

end minimum_degree_P_l168_168995


namespace smallest_digits_to_append_l168_168775

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168775


namespace length_of_floor_is_10_l168_168042

variable (L : ℝ) -- Declare the variable representing the length of the floor

-- Conditions as definitions
def width_of_floor := 8
def strip_width := 2
def area_of_rug := 24
def rug_length := L - 2 * strip_width
def rug_width := width_of_floor - 2 * strip_width

-- Math proof problem statement
theorem length_of_floor_is_10
  (h1 : rug_length * rug_width = area_of_rug)
  (h2 : width_of_floor = 8)
  (h3 : strip_width = 2) :
  L = 10 :=
by
  -- Placeholder for the actual proof
  sorry

end length_of_floor_is_10_l168_168042


namespace box_volume_of_pyramid_l168_168514

/-- A theorem to prove the volume of the smallest cube-shaped box that can house the given rectangular pyramid. -/
theorem box_volume_of_pyramid :
  (∀ (h l w : ℕ), h = 15 ∧ l = 8 ∧ w = 12 → (∀ (v : ℕ), v = (max h (max l w)) ^ 3 → v = 3375)) :=
by
  intros h l w h_condition v v_def
  sorry

end box_volume_of_pyramid_l168_168514


namespace smallest_digits_to_append_l168_168851

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168851


namespace ellipse_CD_distance_l168_168082

theorem ellipse_CD_distance :
  (sqrt ((4 : ℝ)^2 + (2 : ℝ)^2)) = 2 * sqrt 5 := 
by {
  sorry
}

end ellipse_CD_distance_l168_168082


namespace min_distance_A_B_l168_168659

noncomputable def minimum_distance : ℝ :=
  1 + (1/4) * Real.log 3

theorem min_distance_A_B :
  ∀ (A B : ℝ × ℝ), 
  (∃ x y : ℝ, A = (x, y) ∧ (√3) * x - y + 1 = 0) ∧
  (∃ u : ℝ, B = (u, Real.log u)) →
  ∃ P : ℝ × ℝ, 
  ∀ (x y : ℝ), 
  (∃ x : ℝ, P = (x, Real.log x)) ∧
  (A.1 - P.1 + √3 * P.2 - √3 * A.2 + 1 / 2 * Real.log 3 = 0) →
  Real.dist A B = minimum_distance := sorry

end min_distance_A_B_l168_168659


namespace even_heads_probability_is_17_over_25_l168_168427

-- Definition of the probabilities of heads and tails
def prob_tails : ℚ := 1 / 5
def prob_heads : ℚ := 4 * prob_tails

-- Definition of the probability of getting an even number of heads in two flips
def even_heads_prob (p_heads p_tails : ℚ) : ℚ :=
  p_tails * p_tails + p_heads * p_heads

-- Theorem statement
theorem even_heads_probability_is_17_over_25 :
  even_heads_prob prob_heads prob_tails = 17 / 25 := by
  sorry

end even_heads_probability_is_17_over_25_l168_168427


namespace solve_problem_l168_168100

noncomputable def exists_special_integers : Prop :=
  ∃ (A : Fin 10 → ℕ) (p : Fin 10 → ℕ), 
  (∀ i, prime (p i)) ∧ 
  (∀ i, let Ai := (∏ j, if i = j then (p j) ^ 2 else p j) in 
          A i = Ai ∧ 
          (∀ j, i ≠ j → ¬(Ai ∣ A j)) ∧ 
          (∀ j, i ≠ j → A j ∣ (Ai ^ 2)))

theorem solve_problem : exists_special_integers := 
sorry

end solve_problem_l168_168100


namespace value_of_expression_l168_168873

-- Define the conditions
def x := -2
def y := 1
def z := 1
def w := 3

-- The main theorem statement
theorem value_of_expression : 
  (x^2 * y^2 * z^2) - (x^2 * y * z^2) + (y / w) * Real.sin (x * z) = - (1 / 3) * Real.sin 2 := by
  sorry

end value_of_expression_l168_168873


namespace sqrt_23_range_l168_168961

theorem sqrt_23_range : 4.5 < Real.sqrt 23 ∧ Real.sqrt 23 < 5 := by
  sorry

end sqrt_23_range_l168_168961


namespace smallest_positive_debt_resolves_l168_168000

theorem smallest_positive_debt_resolves :
  ∃ (c t : ℤ), (240 * c + 180 * t = 60) ∧ (60 > 0) :=
by
  sorry

end smallest_positive_debt_resolves_l168_168000


namespace arithmetic_sequence_sum_mod_l168_168099

theorem arithmetic_sequence_sum_mod (a d l k S n : ℕ) 
  (h_seq_start : a = 3)
  (h_common_difference : d = 5)
  (h_last_term : l = 103)
  (h_sum_formula : S = (k * (3 + 103)) / 2)
  (h_term_count : k = 21)
  (h_mod_condition : 1113 % 17 = n)
  (h_range_condition : 0 ≤ n ∧ n < 17) : 
  n = 8 :=
by
  sorry

end arithmetic_sequence_sum_mod_l168_168099


namespace cos_sum_formula_l168_168931

open Real

theorem cos_sum_formula (A B C : ℝ) (h1 : sin A + sin B + sin C = 0) (h2 : cos A + cos B + cos C = 0) :
  cos (A - B) + cos (B - C) + cos (C - A) = -3 / 2 :=
by
  sorry

end cos_sum_formula_l168_168931


namespace differentiated_roles_grouping_non_differentiated_roles_grouping_l168_168075

section soldier_grouping

variables (n : ℕ)

-- Case 1: Number of ways to form n groups of 2 with differentiated roles (main shooter and assistant shooter)
theorem differentiated_roles_grouping : 
  (nat.factorial (2 * n)) / (nat.factorial n) = (nat.choose (2 * n) n * nat.factorial n) :=
by sorry

-- Case 2: Number of ways to form n groups of 2 without differentiating roles
theorem non_differentiated_roles_grouping : 
  (nat.factorial (2 * n)) / ((2 ^ n) * nat.factorial n) = ((nat.factorial (2 * n)) / (nat.factorial (2 * n))) :=
by sorry

end soldier_grouping

end differentiated_roles_grouping_non_differentiated_roles_grouping_l168_168075


namespace negation_proposition_l168_168301

variable (U : Type) [DecidableEq U]

-- Conditions
variable (A B : Set U)
variable (p : U → Prop)

-- Assume 2011 is in A ∩ B
noncomputable def propositionP := (2011 ∈ A ∩ B)

-- Provide the negation that we need to prove valid
theorem negation_proposition :
  propositionP U A B 2011 → 2011 ∈ (U \ A) ∪ (U \ B) :=
  sorry

end negation_proposition_l168_168301


namespace number_of_palindromic_integers_l168_168003

open Finset

-- Definitions based on conditions
def D : Finset ℕ := {5, 8, 9}

-- Problem translated into Lean statement (tuple of condition, question, and answer)
theorem number_of_palindromic_integers :
  (D.card * D.card * D.card) = 27 :=
by
  sorry

end number_of_palindromic_integers_l168_168003


namespace division_correct_multiplication_correct_l168_168096

theorem division_correct : 400 / 5 = 80 := by
  sorry

theorem multiplication_correct : 230 * 3 = 690 := by
  sorry

end division_correct_multiplication_correct_l168_168096


namespace part1_proof_part2_proof_l168_168247

-- Given the conditions for part (1)
def part1_find_angle_B (A B: Real) (a b: Real) (h1: 0 < A ∧ A < Real.pi) (h2: 0 < B ∧ B < Real.pi)
  (h3: a ≠ 0) (h4: b ≠ 0) (eq : √3 * b * Real.sin A = a * (1 + Real.cos B)) : Prop :=
  B = Real.pi / 3

-- Given the conditions for part (2)
def part2_max_area_position_M (a b : Real) (A B C x y : Real) (M_AC : ∀ M: Real, M ∈ Set.Icc 0 1)
  (A_pos: a = 1) (B_sqrt3: b = √3) (h1 : Real.sin A = 1 / 2) 
  (h2: 2 * (Real.sin (B - Real.pi / 6) = 1)) 
  (h3 : A = Real.pi / 6)
  (h4 : B = Real.pi / 3)
  (h5 : C = Real.pi / 2)
  (h6 : x ≠ 0) (h7 : y ≠ 0) 
  (angle_ADC : ∠ A D C = 2 * Real.pi / 3) 
  (cosine_law: x^2 + y^2 + x * y = 4) : Prop :=
  let S_max := (sqrt 3 / 4 : Real)
  ∧ ∀ point_M: Real, point_M ∈ Set.Icc 0 1 → abs (point_M - 2/3) < 1/3

-- Lean code for defining the proof problem
theorem part1_proof (A B: Real) (a b: Real) (h1: 0 < A ∧ A < Real.pi) (h2: 0 < B ∧ B < Real.pi)
  (h3: a ≠ 0) (h4: b ≠ 0) (eq : √3 * b * Real.sin A = a * (1 + Real.cos B)) :
  part1_find_angle_B A B a b h1 h2 h3 h4 eq :=
sorry

theorem part2_proof (a b A B C x y: Real) (M_AC : ∀ M: Real, M ∈ Set.Icc 0 1)
  (A_pos : a = 1) (B_sqrt3 : b = √3) (h1 : Real.sin A = 1 / 2) 
  (h2: 2 * (Real.sin (B - Real.pi / 6) = 1))
  (h3 : A = Real.pi / 6)
  (h4 : B = Real.pi / 3)
  (h5 : C = Real.pi / 2)
  (h6 : x ≠ 0) (h7 : y ≠ 0)
  (angle_ADC : ∠ A D C = 2 * Real.pi / 3) 
  (cosine_law: x^2 + y^2 + x * y = 4) :
  part2_max_area_position_M a b A B C x y M_AC A_pos B_sqrt3 h1 h2 h3 h4 h5 h6 h7 angle_ADC cosine_law :=
sorry

end part1_proof_part2_proof_l168_168247


namespace harvest_bushels_l168_168451

def num_rows : ℕ := 5
def stalks_per_row : ℕ := 80
def stalks_per_bushel : ℕ := 8

theorem harvest_bushels : (num_rows * stalks_per_row) / stalks_per_bushel = 50 := by
  sorry

end harvest_bushels_l168_168451


namespace f_one_one_f_not_decreasing_f_increasing_eq_identity_l168_168645

variable {R : Type*} [LinearOrderedField R]

-- Condition
variable (f : R → R)
axiom f_property : ∀ x : R, f (f (f x)) = x

-- Part (i): Show that f is one-one
theorem f_one_one : ∀ x y : R, f x = f y → x = y :=
by
  intros x y h
  have : f (f (f x)) = f (f (f y)), by rw h
  rw [f_property x, f_property y] at this
  exact this

-- Part (ii): Show that f cannot be strictly decreasing
theorem f_not_decreasing : ¬StrictMonoDecr f :=
by
  intro hf
  have hc : ∀ x : R, f(f(f x)) = x := f_property
  specialize hf 0 1 (by norm_num)
  rw [hc 0, hc 1] at hf
  exact hf.ne rfl

-- Part (iii): If f is strictly increasing, then show that f(x) = x for all x ∈ R
theorem f_increasing_eq_identity (h_inc : StrictMono f) : ∀ x : R, f x = x :=
by
  intro x
  have hc : ∀ x : R, f(f(f x)) = x := f_property
  specialize h_inc (f x) x
  have : x ≤ f x, from le_of_lt (by rwa [← hc x])
  have h1 := by { specialize h_inc this, rwa ← hc x at h_inc }
  have : f x ≤ x := le_of_lt (by rwa h1)
  exact le_antisymm this this

end f_one_one_f_not_decreasing_f_increasing_eq_identity_l168_168645


namespace area_of_set_S_l168_168950

theorem area_of_set_S :
  let octagon_side_length := 2 * Real.sqrt(2 - Real.sqrt(2)),
  let octagon_center := (0 : ℂ),
  let real_axis_parallel := true,
  let R := { z : ℂ | ∀ w ∈ (octagon_center + octagon_side_length * ℂ.Im).continuity, z ≠ w},
  let S := { 1 / z | z ∈ R },
  area S = π / 8 :=
sorry

end area_of_set_S_l168_168950


namespace triangle_problem_l168_168629

noncomputable theory

def triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  ∃ x y : ℝ, x = dist A B ∧ y = dist A C

theorem triangle_problem (A B C : ℝ) (hA : angle A = 90) (hBC : dist B C = 10) (h_tanC_cosB : tan (angle C) = 3 * cos (angle B)) :
  dist A B = 20 * Real.sqrt 2 / 3 :=
sorry

end triangle_problem_l168_168629


namespace fraction_not_on_time_l168_168067

theorem fraction_not_on_time (total_attendees : ℕ) (h1 : 2 * total_attendees % 3 = 0) (h2 : 3 * (2 * total_attendees / 3) % 4 = 0) (h3 : 5 * (total_attendees - 2 * total_attendees / 3) % 6 = 0) : 
  (total_attendees - (3 * (2 * total_attendees / 3) / 4) - (5 * (total_attendees - 2 * total_attendees / 3) / 6)) / total_attendees = 1 / 4 := 
begin
  sorry
end

end fraction_not_on_time_l168_168067


namespace tangent_line_at_1_1_is_5x_plus_y_minus_6_l168_168499

noncomputable def f : ℝ → ℝ :=
  λ x => x^3 - 4*x^2 + 4

def tangent_line_equation (x₀ y₀ m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y - y₀ = m * (x - x₀)

theorem tangent_line_at_1_1_is_5x_plus_y_minus_6 : 
  tangent_line_equation 1 1 (-5) = (λ x y => 5 * x + y - 6 = 0) := 
by
  sorry

end tangent_line_at_1_1_is_5x_plus_y_minus_6_l168_168499


namespace second_caterer_cheaper_l168_168282

theorem second_caterer_cheaper (x : ℕ) (h : x > 33) : 200 + 12 * x < 100 + 15 * x := 
by
  sorry

end second_caterer_cheaper_l168_168282


namespace salary_reduction_percentage_l168_168420

theorem salary_reduction_percentage
  (S : ℝ) 
  (h : S * (1 - R / 100) = S / 1.388888888888889): R = 28 :=
sorry

end salary_reduction_percentage_l168_168420


namespace no_consecutive_positive_integers_have_sum_75_l168_168717

theorem no_consecutive_positive_integers_have_sum_75 :
  ∀ n a : ℕ, (n ≥ 2) → (a ≥ 1) → (n * (2 * a + n - 1) = 150) → False :=
by
  intros n a hn ha hsum
  sorry

end no_consecutive_positive_integers_have_sum_75_l168_168717


namespace num_even_five_digit_numbers_position_of_35214_num_divisible_by_6_five_digit_numbers_l168_168766

/-- First statement: Number of distinct five-digit even numbers greater than 20000. -/
theorem num_even_five_digit_numbers : 
  let digits := {0, 1, 2, 3, 4, 5}
  even_five_digit_numbers_gt_20000 : 
  (count (λ n, even_number n ∧ 20000 < n ∧ (has_distinct_digits n digits) ∧ (num_digits n = 5)) (all_numbers digits)) = 240
by sorry

/-- Second statement: Position of the number 35214 in ascending order. -/
theorem position_of_35214 : 
  let digits := {0, 1, 2, 3, 4, 5}
  position_in_ascending_order : 
  (find_position 35214 (ascending_order_list (all_five_digit_numbers digits))) = 351
by sorry

/-- Third statement: Number of distinct five-digit numbers divisible by 6. -/
theorem num_divisible_by_6_five_digit_numbers : 
  let digits := {0, 1, 2, 3, 4, 5}
  divisible_by_6_five_digit_numbers : 
  (count (λ n, divisible_by_6 n ∧ (has_distinct_digits n digits) ∧ (num_digits n = 5)) (all_numbers digits)) = 108
by sorry

end num_even_five_digit_numbers_position_of_35214_num_divisible_by_6_five_digit_numbers_l168_168766


namespace max_value_yx_l168_168167

noncomputable def max_ratio_yx (x y : ℝ) (h1 : x ≠ 0) (h2 : ∥(x - 2, y)∥ = sqrt 3) : ℝ :=
  sqrt 3

theorem max_value_yx (x y : ℝ) (h1 : x ≠ 0) (h2 : ∥(x - 2 : ℂ) - 2∥ = sqrt 3) :
    max_ratio_yx x y h1 h2 = sqrt 3 :=
sorry

end max_value_yx_l168_168167


namespace find_angle_y_l168_168620

-- Define the conditions
def parallel_lines (m n : ℝ) : Prop := m = n

def supplementary_angles (a b : ℝ) : Prop := a + b = 180

-- State the problem
theorem find_angle_y (m n : ℝ) (H1 : parallel_lines m n) (H2 : ∠ D B I = 40) (H3 : ∠ D B I + ∠ y = 180) :
  ∠ y = 140 := 
sorry

end find_angle_y_l168_168620


namespace no_real_solutions_eqn_l168_168214

theorem no_real_solutions_eqn : ∀ x : ℝ, (2 * x - 4 * x + 7)^2 + 1 ≠ -|x^2 - 1| :=
by
  intro x
  sorry

end no_real_solutions_eqn_l168_168214


namespace people_needed_to_mow_lawn_l168_168489

theorem people_needed_to_mow_lawn
  (p1 t1 t2 : ℕ)
  (h1 : p1 = 8)
  (h2 : t1 = 8)
  (h3 : t2 = 2)
  (constant_product : p1 * t1 = 64) :
  let p2 := 64 / t2 in
  (p2 - p1) = 24 :=
by
  sorry

end people_needed_to_mow_lawn_l168_168489


namespace sum_of_n_with_perfect_square_condition_l168_168651

theorem sum_of_n_with_perfect_square_condition :
  let T := (∑ n in (Finset.filter (λ n, ∃ k : ℤ, n^2 + 12 * n - 208 = k^2) (Finset.range 100)), n)
  in T % 100 = 56 := by
  sorry

end sum_of_n_with_perfect_square_condition_l168_168651


namespace smallest_n_7n_eq_n7_mod_3_l168_168481

theorem smallest_n_7n_eq_n7_mod_3 : ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 3]) ∧ ∀ m : ℕ, m > 0 → (7^m ≡ m^7 [MOD 3] → m ≥ n) :=
by
  sorry

end smallest_n_7n_eq_n7_mod_3_l168_168481


namespace revisit_origin_if_polynomials_cannot_revisit_origin_if_arbitrary_l168_168274

-- Defining the problem context
def is_polynomial (f : ℕ → ℕ) : Prop := ∃ p : Polynomial ℤ, ∀ n, f n = p.eval n

-- Main theorem statements
theorem revisit_origin_if_polynomials : 
  ∀ (f g : ℕ → ℕ), (is_polynomial f) → (is_polynomial g) → 
  (∃ (infinitely_often : ℕ → (ℤ × ℤ)), ∀ n, infinitely_often n = (0, 0)) :=
begin
  sorry -- The proof is omitted as per instructions
end

theorem cannot_revisit_origin_if_arbitrary : 
  ∀ (f g : ℕ → ℕ), ¬(∃ (infinitely_often : ℕ → (ℤ × ℤ)), ∀ n, infinitely_often n = (0, 0)) :=
begin
  sorry -- The proof is omitted as per instructions
end

end revisit_origin_if_polynomials_cannot_revisit_origin_if_arbitrary_l168_168274


namespace call_fee_correct_l168_168946

noncomputable def calculate_fee (t : ℝ) : ℝ :=
  if t <= 3 then
    0.22
  else
    let excess := t - 3 in
    if excess.floor = excess then
      0.22 + 0.1 * excess
    else
      0.22 + 0.1 * (excess.floor + 1)

theorem call_fee_correct (t : ℝ) : 
  (t <= 3 → calculate_fee t = 0.22) ∧ 
  (t > 3 → (t.floor = t → calculate_fee t = 0.22 + 0.1 * (t - 3)) ∧ 
             (t.floor ≠ t → calculate_fee t = 0.22 + 0.1 * ((t - 3).floor + 1))) :=
sorry

end call_fee_correct_l168_168946


namespace range_of_a_l168_168600

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x + 6 < 2 + 3x → (a + x) / 4 > x) ∧ (∃! i : ℤ, ∃! j : ℤ, ∃! k : ℤ, 2 < i ∧ i < a / 3 ∧ 2 < j ∧ j < a / 3 ∧ 2 < k ∧ k < a / 3) → 15 < a ∧ a ≤ 18 :=
by
  sorry

end range_of_a_l168_168600


namespace lcm_4_6_9_l168_168500

/-- The least common multiple (LCM) of 4, 6, and 9 is 36 -/
theorem lcm_4_6_9 : Nat.lcm (Nat.lcm 4 6) 9 = 36 :=
by
  -- sorry replaces the actual proof steps
  sorry

end lcm_4_6_9_l168_168500


namespace not_linear_eq_l168_168012

-- Representing the given equations
def eq1 (x : ℝ) : Prop := 5 * x + 3 = 3 * x - 7
def eq2 (x : ℝ) : Prop := 1 + 2 * x = 3
def eq4 (x : ℝ) : Prop := x - 7 = 0

-- The equation to verify if it's not linear
def eq3 (x : ℝ) : Prop := abs (2 * x) / 3 + 5 / x = 3

-- Stating the Lean statement to be proved
theorem not_linear_eq : ¬ (eq3 x) := by
  sorry

end not_linear_eq_l168_168012


namespace infinitely_many_composite_among_sequence_l168_168680

theorem infinitely_many_composite_among_sequence :
  ∃∞ k, ∃ m ≥ k, ∃ n > 1, (n ∣ (Int.floor (Real.sqrt 2 * 2^m)) ∧ ¬ (Nat.prime n)) :=
begin
  sorry
end

end infinitely_many_composite_among_sequence_l168_168680


namespace angle_FCG_l168_168280

theorem angle_FCG {A B C D E F G : Type} [incircle : Circle]
  (diam_AE : diameter AE)
  (angle_ABF : angle A B F = 81)
  (angle_EDG : angle E D G = 76) :
  angle F C G = 67 :=
begin
  sorry
end

end angle_FCG_l168_168280


namespace cuboid_surface_area_l168_168596

def length := 90 -- cm
def breadth := 75 -- cm
def height := 60 -- cm

def ratio (a b c : Nat) : Prop := a / b = 6 / 5 ∧ b / c = 5 / 4

theorem cuboid_surface_area (l b h : Nat) (hr : ratio l b h) : 
  2 * (l * b + b * h + h * l) = 33300 :=
by
  rw [length, breadth, height] at *
  sorry

end cuboid_surface_area_l168_168596


namespace number_divisible_l168_168791

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168791


namespace find_ordered_pair_l168_168502

theorem find_ordered_pair :
  ∃ (x y : ℚ), (x + y + 1 = (6 - x) + (6 - y)) ∧ (x - y + 2 = (x - 2) + (y - 2)) ∧ x = 5 / 2 ∧ y = 3 := by
sorry

end find_ordered_pair_l168_168502


namespace complement_union_eq_l168_168159

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_union_eq :
  U = {1, 2, 3, 4, 5, 6, 7, 8} →
  A = {1, 3, 5, 7} →
  B = {2, 4, 5} →
  U \ (A ∪ B) = {6, 8} :=
by
  intros hU hA hB
  -- Proof goes here
  sorry

end complement_union_eq_l168_168159


namespace graph_shift_cos_function_l168_168548

theorem graph_shift_cos_function (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = 2 * Real.cos (π * x / 3 + φ)) ∧ 
  (∃ x, f x = 0 ∧ x = 2) ∧ 
  (f 1 > f 3) →
  (∀ x, f x = 2 * Real.cos (π * (x - 1/2) / 3)) :=
by
  sorry

end graph_shift_cos_function_l168_168548


namespace reflected_ray_equation_l168_168040

noncomputable def problem_statement : Prop :=
  let emitted_point := (2 : ℝ, 3 : ℝ)
  let travel_line := λ p : ℝ × ℝ, p.1 - 2 * p.2 = 0
  let reflected_line := λ p : ℝ × ℝ, p.1 + 2 * p.2 - 4 = 0
  ∃ p : ℝ × ℝ, (travel_line p) ∧ (reflected_line p)

theorem reflected_ray_equation:
  problem_statement := sorry

end reflected_ray_equation_l168_168040


namespace product_of_two_numbers_l168_168333

theorem product_of_two_numbers (x y : ℝ)
  (h1 : x + y = 25)
  (h2 : x - y = 3)
  : x * y = 154 := by
  sorry

end product_of_two_numbers_l168_168333


namespace rotation_x_coordinate_l168_168180

theorem rotation_x_coordinate :
  ∀ (P Q : ℝ × ℝ), 
    P = (4 * Real.sqrt 3, 1) → 
    (∃ θ : ℝ, θ = Real.arctan (1 / (4 * Real.sqrt 3))) → 
    Q = (7 * (Real.cos (θ - Real.pi / 3)), 7 * (Real.sin (θ - Real.pi / 3))) → 
    (fst Q) = (5 * Real.sqrt 3 / 2) :=
sorry

end rotation_x_coordinate_l168_168180


namespace kelly_baking_powder_difference_l168_168015

theorem kelly_baking_powder_difference :
  let amount_yesterday := 0.4
  let amount_now := 0.3
  amount_yesterday - amount_now = 0.1 :=
by
  -- Definitions for amounts 
  let amount_yesterday := 0.4
  let amount_now := 0.3
  
  -- Applying definitions in the computation
  show amount_yesterday - amount_now = 0.1
  sorry

end kelly_baking_powder_difference_l168_168015


namespace min_value_l168_168542

theorem min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ x, x = a^2 + 1 / (b * (a - b)) ∧ x ≥ 4 :=
by
  sorry

end min_value_l168_168542


namespace tangent_line_parabola_l168_168512

theorem tangent_line_parabola (k : ℝ) (tangent : ∀ y : ℝ, ∃ x : ℝ, 4 * x + 3 * y + k = 0 ∧ y^2 = 12 * x) : 
  k = 27 / 4 :=
sorry

end tangent_line_parabola_l168_168512


namespace count_three_digit_numbers_with_double_sum_two_l168_168515

-- Definitions for the sum of digits and double sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def double_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (sum_of_digits n)

-- Predicate to check if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Main theorem statement
theorem count_three_digit_numbers_with_double_sum_two :
  (finset.filter (λ n, double_sum_of_digits n = 2) (finset.Icc 100 999)).card = 100 :=
by
  sorry

end count_three_digit_numbers_with_double_sum_two_l168_168515


namespace inequality_chain_l168_168161

theorem inequality_chain (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b) (h₃ : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_chain_l168_168161


namespace necessary_and_sufficient_condition_l168_168501

theorem necessary_and_sufficient_condition 
  (n : ℕ) 
  (x : Fin (n + 1) → ℝ) 
  (y : Fin (n + 1) → ℝ) 
  (h₀ : 2 ≤ n) :
  (∃ y : Fin (n + 1) → ℝ, 
    (x 0 + complex.I * y 0)^2 = 
    ∑ k in Finset.range n, (x (Fin.succ k) + complex.I * y (Fin.succ k))^2) ↔
  (x 0 ^ 2 ≤ ∑ k in Finset.range n, (x (Fin.succ k))^2) := 
sorry

end necessary_and_sufficient_condition_l168_168501


namespace max_value_l168_168294

open Real

-- Define f(x) as sin(x)
def f (x : ℝ) : ℝ := sin x

-- Define g(x) as the right-shift of f(x) by π/3
def g (x : ℝ) : ℝ := sin (x - π / 3)

-- Define the function y = f(x) + g(x)
def y (x : ℝ) : ℝ := f x + g x

-- Statement of the problem
theorem max_value : ∃ x, y x = sqrt 3 :=
sorry

end max_value_l168_168294


namespace calculate_expression_l168_168073

theorem calculate_expression : 
  -3^2 + Real.sqrt ((-2)^4) - (-27)^(1/3 : ℝ) = -2 := 
by
  sorry

end calculate_expression_l168_168073


namespace integral_value_l168_168184

noncomputable def constant_term : ℝ := -12
noncomputable def a : ℝ := -1

theorem integral_value : 
  (∫ x in a..2, sin(π / 2 * x)) = 2 / π := 
by
  sorry

end integral_value_l168_168184


namespace smallest_digits_to_append_l168_168826

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168826


namespace max_log_expression_is_zero_l168_168593

-- Define the conditions
variables (a b : Real) (h1 : a^2 ≥ b^2) (h2 : b^2 > 1)

-- Define the logarithmic expression we want to find the maximum of
noncomputable def log_expr := log (a^2 / b^2) / log (a^2) + log (b^2 / a^2) / log (b^2)

-- The theorem statement that expresses the maximum value is 0
theorem max_log_expression_is_zero (h1 : a^2 ≥ b^2) (h2 : b^2 > 1) : 
  ∃ a b : Real, log_expr a b h1 h2 = 0 :=
sorry

end max_log_expression_is_zero_l168_168593


namespace min_expression_value_l168_168478

open Real

theorem min_expression_value : ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 := by
  sorry

end min_expression_value_l168_168478


namespace pairs_nat_eq_l168_168114

theorem pairs_nat_eq (n k : ℕ) :
  (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) :=
by
  sorry

end pairs_nat_eq_l168_168114


namespace minimum_neighboring_pairs_l168_168446

def infinite_sheet : Type := ℕ × ℕ -- Representing the infinite grid as pairs of natural numbers

def color : Type := Fin 9 -- Representing the 9 colors (0 to 8)

def painting (f : infinite_sheet → color) := True -- A painting is a function from the grid to the colors

def neighboring (c1 c2 : color) (f : infinite_sheet → color) : Prop :=
  ∃ (x1 x2 y1 y2 : ℕ), 
    (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 + 1 = y2) ∨ 
     y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 + 1 = x2)) ∧ 
    f (x1, y1) = c1 ∧ f (x2, y2) = c2

theorem minimum_neighboring_pairs :
  ∃ f : infinite_sheet → color, 
    -- There is a painting such that each color is used at least once
    (∀ c : color, ∃ x y : ℕ, f (x, y) = c) ∧ 
    -- And the number of neighboring pairs is minimal
    let pairs := {p : color × color // neighboring p.1 p.2 f} in 
    Fintype.card pairs = 8 := 
sorry

end minimum_neighboring_pairs_l168_168446


namespace equalize_pressures_40_cylinders_l168_168336

theorem equalize_pressures_40_cylinders (k n : ℕ) (h: n = 40) 
 (connect_cylinders: ∀ (k: ℕ), k ≤ 5 → Prop) 
 (equalize_pressure: ∀ pressures: (fin n → ℝ), 
  ∃ steps: ℕ, n > 0 → 
  (∀ i: fin n, pressures i = pressures 0) → 
  (∀ (s : ℕ), s < steps → 
  ∀ (subset: set (fin n)), finset.card subset ≤ k → 
  ∀ i ∈ subset, pressures i = (finset.sum subset pressures) / (finset.card subset)) 
 ) :
 ∃ k, k = 5 :=
begin
  sorry
end

end equalize_pressures_40_cylinders_l168_168336


namespace tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l168_168990

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.log x - x
noncomputable def g (x m : ℝ) : ℝ := f x + m * x^2
noncomputable def tangentLineEq (x y : ℝ) : Prop := x + 2 * y + 1 = 0
noncomputable def rangeCondition (x₁ x₂ m : ℝ) : Prop := g x₁ m + g x₂ m < -3 / 2

theorem tangent_line_eq_at_x_is_1 :
  tangentLineEq 1 (f 1) := 
sorry

theorem range_of_sum_extreme_values (h : 0 < m ∧ m < 1 / 4) (x₁ x₂ : ℝ) :
  rangeCondition x₁ x₂ m := 
sorry

end tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l168_168990


namespace each_monkey_gets_banana_l168_168337

noncomputable def monkey_banana_problem : Prop :=
  ∀ (monkeys ladders : Fin 5)
    (rope_connections : Fin 5 → Fin 5 → Bool),
    (∀ i, ∃! j, monkeys i = ladders j) ∧  -- each monkey starts at a different ladder
    (∀ i j, rope_connections i j = true ↔ rope_connections j i = true) ∧  -- symmetric rope connections
    (∀ i j k, rope_connections i j = true → rope_connections j k = true → i ≠ k)  -- unique rope connections per rung
    → 
    ∃ bananas : Fin 5,
      ∀ i, ∃! j, monkeys i = ladders j ∧ 
                 ladders j = bananas j  -- each monkey gets a banana

-- Lean statement for proof obligation
theorem each_monkey_gets_banana : monkey_banana_problem :=
  sorry

end each_monkey_gets_banana_l168_168337


namespace smallest_digits_to_append_l168_168852

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168852


namespace constant_term_expansion_l168_168241

theorem constant_term_expansion (r : Nat) (h : 12 - 3 * r = 0) :
  (Nat.choose 6 r) * 2^r = 240 :=
sorry

end constant_term_expansion_l168_168241


namespace jackson_total_souvenirs_l168_168638

-- Define the conditions
def num_hermit_crabs : ℕ := 45
def spiral_shells_per_hermit_crab : ℕ := 3
def starfish_per_spiral_shell : ℕ := 2

-- Define the calculation based on the conditions
def num_spiral_shells := num_hermit_crabs * spiral_shells_per_hermit_crab
def num_starfish := num_spiral_shells * starfish_per_spiral_shell
def total_souvenirs := num_hermit_crabs + num_spiral_shells + num_starfish

-- Prove that the total number of souvenirs is 450
theorem jackson_total_souvenirs : total_souvenirs = 450 :=
by
  sorry

end jackson_total_souvenirs_l168_168638


namespace factorize_expression_l168_168109

variable (x y : ℝ)

theorem factorize_expression :
  x^2 - 1 + 2 * x * y + y^2 = (x + y + 1) * (x + y - 1) :=
by
  sorry

end factorize_expression_l168_168109


namespace projection_n4_projection_n6_projection_n10_l168_168281

section regular_polygon_projections

-- Define the central conditions: Regular n-gon with specific projections

-- The type of regular n-gon and relevant projections
variable (n : ℕ) (A1 A2 A3 ... An : Type) (M1 : Type)  -- placeholder types for points
variable (OnSide : A1 -> A2 -> M1 -> Prop) -- Placeholder for condition about points being on the side
variable (Project : A1 -> A2 -> A3 -> A4 -> M1 -> M1 -> Prop) -- Placeholder projection condition
variable (coincidesWith : M1 -> M1 -> Prop) -- Placeholder for points coincidence

-- Assuming some basic properties for our types
variable [PointLineProperty : ∀ (x y z: Type), Prop]  -- Placeholder for property of lines through points

-- Define the theorem statements
theorem projection_n4 (M13 : M1): OnSide A1 A2 M1 -> (∃ M2 M3 ... M12, Project A1 ... A4 M1 M2) -> (coin - ...
  coincidesWith M13 M1 := sorry 

theorem projection_n6 (M13 : M1): OnSide A1 A2 M1 -> (∃ M2 M3 ... M12, Project A1 ... A4 M1 M2) -> (coin - ...
  coincidesWith M13 M1 := sorry 

theorem projection_n10 (M11 : M1): OnSide A1 A2 M1 -> (∃ M2 M3 ... M10, Project A1 ... A4 M1 M2) -> (coin - 
  coincidesWith M11 M1 := sorry 

end regular_polygon_projections

end projection_n4_projection_n6_projection_n10_l168_168281


namespace people_in_group_l168_168705

-- Define the conditions as Lean definitions
def avg_weight_increase := 2.5
def replaced_weight := 45
def new_weight := 65
def weight_difference := new_weight - replaced_weight -- 20 kg

-- State the problem as a Lean theorem
theorem people_in_group :
  ∀ n : ℕ, avg_weight_increase * n = weight_difference → n = 8 :=
by
  intros n h
  sorry

end people_in_group_l168_168705


namespace distance_between_first_and_second_points_l168_168490

theorem distance_between_first_and_second_points
  (points : Fin 11 → ℝ)
  (h1 : ∑ i in Finset.filter (λ i, 0 < i) (Finset.univ : Finset (Fin 11)), (points i - points 0).abs = 2018)
  (h2 : ∑ i in Finset.filter (λ i, 1 ≠ i) (Finset.univ : Finset (Fin 11)), (points i - points 1).abs = 2000) :
  (points 1 - points 0).abs = 2 := by
  sorry

end distance_between_first_and_second_points_l168_168490


namespace tan_beta_add_pi_over_6_l168_168539

variable (α β : ℝ)

theorem tan_beta_add_pi_over_6 :
  (tan (α - π / 6) = 2) →
  (tan (α + β) = -3) →
  (tan (β + π / 6) = 1) := 
by
  intro h1 h2,
  sorry

end tan_beta_add_pi_over_6_l168_168539


namespace polynomial_multiplication_l168_168078

-- Define the variable a
variable (a : ℝ)

-- The mathematical proof problem in Lean 4 statement
theorem polynomial_multiplication :
  (a + 2) * (2a - 3) = 2 * a ^ 2 + a - 6 :=
sorry

end polynomial_multiplication_l168_168078


namespace Yan_ratio_distance_l168_168017

theorem Yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq : y/w = x/w + (x + y)/(5 * w)) : x/y = 2/3 := by
  sorry

end Yan_ratio_distance_l168_168017


namespace polynomial_count_l168_168422

noncomputable def number_of_polynomials : ℕ := 528

theorem polynomial_count :
  (∀ (f : ℂ → ℂ),
    (∃ n : ℕ, ∃ (c : fin n → ℕ),
      f = (λ z, z^n + c (fin n - 1) * z^(n-1) +
        ∑ (i : fin (n-2)), c i * z^(i + 1) + 3 * c 0 * z + 50) ∧
      (∀ z : ℂ, f z = 0 → ∃ a b : ℤ, z = a + b * complex.I)) →
  ∃ k : ℕ, k = 528) :=
by
  intro f h
  have h_count : ∃ n : ℕ, f = (λ z, z^n + ... + 50) → true,
  {
    sorry
  }
  existsi 528
  exact h_count
  sorry

end polynomial_count_l168_168422


namespace length_of_plot_l168_168769

variables (width_ft : ℝ) (total_rabbits : ℕ) (area_per_rabbit_per_day : ℕ) (days : ℕ)
variables (width_yd length_yd total_area_yd2 : ℝ)

-- Provided conditions
def width_ft := 200
def total_rabbits := 100
def area_per_rabbit_per_day := 10
def days := 20

-- Conversion from feet to yards
def width_yd := width_ft / 3

-- Total area calculation
def total_area_yd2 := total_rabbits * area_per_rabbit_per_day * days

-- The question is to find the length of the plot of land such that width_yd * length_yd = total_area_yd2.
theorem length_of_plot : length_yd = 300 :=
by
  have : length_yd = total_area_yd2 / width_yd,
  { sorry },
  rw this,
  -- Approximate values to match the problem statement
  have h1 : width_yd = 200 / 3 := by sorry,
  have h2 : total_area_yd2 = 100 * 10 * 20 := by sorry,
  rw [h1, h2],
  norm_num,
  sorry

end length_of_plot_l168_168769


namespace students_selected_are_three_l168_168737

-- Definitions of the conditions 
variables (boys girls ways : ℕ)
variables (selection_ways : ℕ)

-- Given conditions
def boys_in_class : Prop := boys = 15
def girls_in_class : Prop := girls = 10
def ways_to_select : Prop := selection_ways = 1050

-- Define the problem statement
theorem students_selected_are_three 
  (hb : boys_in_class boys) 
  (hg : girls_in_class girls)
  (hw : ways_to_select 1050) :
  ∃ n, n = 3 := 
sorry

end students_selected_are_three_l168_168737


namespace matrix_multiplication_correct_l168_168948

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 0, 4],
    ![2, -1, 0],
    ![0, 1, -2]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, -3, 0],
    ![0, 2, 1],
    ![4, 0, -2]]

def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![31, -9, -8],
    ![10, -8, -1],
    ![-8, 2, 5]]

theorem matrix_multiplication_correct : A ⬝ B = C :=
  sorry

end matrix_multiplication_correct_l168_168948


namespace correct_proposition_l168_168465

-- Definitions of basic geometric relationships
variables {Point Line Plane : Type}
variables (l : Line) (alpha beta : Plane)

-- Conditions from the problem
variable parallel_to_plane : Line → Plane → Prop
variable perpendicular_to_plane : Line → Plane → Prop
variable planes_parallel : Plane → Plane → Prop
variable planes_perpendicular : Plane → Plane → Prop

-- Correct proposition to verify
theorem correct_proposition (hl_alpha : parallel_to_plane l alpha)
                           (hl_beta : perpendicular_to_plane l beta) :
  planes_perpendicular alpha beta := sorry

end correct_proposition_l168_168465


namespace ellipse_equation_max_area_line_equations_l168_168537

-- Definitions for the given conditions
def f1 : ℝ × ℝ := (-real.sqrt 3, 0)
def f2 : ℝ × ℝ := (real.sqrt 3, 0)
def point_p : ℝ × ℝ := (1, real.sqrt 3 / 2)

-- Ellipse parameters
variables {a b : ℝ}
axiom ellipse_form : a > b ∧ b > 0

-- Definitions for the equations to be proven
def ellipse_eq (x y : ℝ) := (x^2 / 4 + y^2 = 1)
def line_eq1 (x y : ℝ) := (x - real.sqrt 2 * y + real.sqrt 3 = 0)
def line_eq2 (x y : ℝ) := (x + real.sqrt 2 * y + real.sqrt 3 = 0)

-- Proof statement for the ellipse equation
theorem ellipse_equation : 
  (∀ x y : ℝ, point_p = (x, y) → ellipse_eq x y) ∧ 
  (f1 ≠ f2 ∧ (f1.1 = -real.sqrt 3 ∧ f1.2 = 0) ∧ (f2.1 = real.sqrt 3 ∧ f2.2 = 0)) → 
  (a^2 = 4 ∧ b^2 = 1) → 
  (∀ x y : ℝ, ellipse_eq x y) := 
sorry

-- Proof statement for the line equations
theorem max_area_line_equations : 
  (∀ x y : ℝ, point_p = (x, y) → ellipse_eq x y) ∧ 
  (f1 ≠ f2 ∧ (f1.1 = -real.sqrt 3 ∧ f1.2 = 0) ∧ (f2.1 = real.sqrt 3 ∧ f2.2 = 0)) → 
  (a^2 = 4 ∧ b^2 = 1) → 
  (∀ x y : ℝ, (line_eq1 x y ∨ line_eq2 x y)) := 
sorry

end ellipse_equation_max_area_line_equations_l168_168537


namespace smallest_number_l168_168061

-- Definitions of the numbers in their respective bases
def num1 := 5 * 9^0 + 8 * 9^1 -- 85_9
def num2 := 0 * 6^0 + 1 * 6^1 + 2 * 6^2 -- 210_6
def num3 := 0 * 4^0 + 0 * 4^1 + 0 * 4^2 + 1 * 4^3 -- 1000_4
def num4 := 1 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5 -- 111111_2

-- Assert that num4 is the smallest
theorem smallest_number : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 :=
by 
  sorry

end smallest_number_l168_168061


namespace probability_join_same_group_l168_168756

-- Define the study groups
inductive StudyGroup
| A
| B
| C

-- Define the people
inductive Person
| 甲
| 乙

-- Define the probability that both 甲 and 乙 join the same study group
def probability_same_group : ℚ :=
  have total_ways : ℚ := 3 * 3,
  have favorable_ways : ℚ := 3,
  favorable_ways / total_ways

theorem probability_join_same_group : probability_same_group = 1 / 3 := 
  sorry

end probability_join_same_group_l168_168756


namespace train_crosses_bridge_in_30_seconds_l168_168048

theorem train_crosses_bridge_in_30_seconds
    (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ)
    (h1 : train_length = 110)
    (h2 : train_speed_kmh = 45)
    (h3 : bridge_length = 265) : 
    (train_length + bridge_length) / (train_speed_kmh * (1000 / 3600)) = 30 := 
by
  sorry

end train_crosses_bridge_in_30_seconds_l168_168048


namespace number_divisible_l168_168782

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168782


namespace defined_interval_l168_168985

theorem defined_interval (x : ℝ) : (1 ≤ x ∧ x < 5) ↔ (∃ y : ℝ, sqrt (2 * x - 2) = y ∧ log (5 - x) ∈ ℝ) :=
by
  sorry

end defined_interval_l168_168985


namespace number_divisible_l168_168790

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168790


namespace circles_internally_tangent_l168_168720

-- Define the first circle
def circle1_center := (0 : ℝ, 0 : ℝ)
def circle1_radius := 2

-- Define the second circle
def circle2_center := (3 : ℝ, -4 : ℝ)
def circle2_radius := 7

-- Define a helper function to find distance between centers
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
    Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define the main statement
theorem circles_internally_tangent :
  distance circle1_center circle2_center = circle2_radius - circle1_radius :=
begin
   sorry
end

end circles_internally_tangent_l168_168720


namespace quadratic_functions_x4_minus_x1_eq_752_l168_168383

theorem quadratic_functions_x4_minus_x1_eq_752
  (f g : ℝ → ℝ)
  (hf : ∀ a b c : ℝ, f x = a * x^2 + b * x + c)
  (hg : ∀ x : ℝ, g x = - f (100 - x))
  (h_vertex : ∃ v : ℝ, g v = f v)
  (x1 x2 x3 x4 : ℝ)
  (hx_asc : x1 < x2 ∧ x2 < x3 ∧ x3 < x4)
  (hx_diff : x3 - x2 = 150)
  (m n p : ℕ)
  (hp_prime_square : ∀ k : ℕ, k * k ∣ p → k = 1)
  (hx4_x1 : x4 - x1 = m + n * sqrt p) :
  m + n + p = 752 :=
sorry

end quadratic_functions_x4_minus_x1_eq_752_l168_168383


namespace simplify_expression_l168_168296

-- Define the rational functions
def f1 (x : ℝ) := (x^2 - 4*x + 3) / (x^2 - 6*x + 9)
def f2 (x : ℝ) := (x^2 - 6*x + 8) / (x^2 - 8*x + 15)

-- Define the simplified rational function
def target (x : ℝ) := ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2))

-- State the theorem
theorem simplify_expression (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 2) (h₄ : x ≠ 5) : f1 x / f2 x = target x := by
  sorry

end simplify_expression_l168_168296


namespace Eliane_schedule_combinations_l168_168491

def valid_schedule_combinations : ℕ :=
  let mornings := 6 * 3 -- 6 days (Monday to Saturday) each with 3 time slots
  let afternoons := 5 * 2 -- 5 days (Monday to Friday) each with 2 time slots
  let mon_or_fri_comb := 2 * 3 * 3 * 2 -- Morning on Monday or Friday
  let sat_comb := 1 * 3 * 4 * 2 -- Morning on Saturday
  let tue_wed_thu_comb := 3 * 3 * 2 * 2 -- Morning on Tuesday, Wednesday, or Thursday
  mon_or_fri_comb + sat_comb + tue_wed_thu_comb

theorem Eliane_schedule_combinations :
  valid_schedule_combinations = 96 := by
  sorry

end Eliane_schedule_combinations_l168_168491


namespace segments_in_proportion_l168_168874

theorem segments_in_proportion :
  ∀ (a b c d : ℝ), (a = 2) → (b = ℝ.sqrt 5) → (c = 2 * ℝ.sqrt 3) → (d = ℝ.sqrt 15) → (a * d = b * c) :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end segments_in_proportion_l168_168874


namespace base_eight_to_base_ten_l168_168355

theorem base_eight_to_base_ten (a b c : ℕ) (ha : a = 1) (hb : b = 5) (hc : c = 7) :
  a * 8^2 + b * 8^1 + c * 8^0 = 111 :=
by 
  -- rest of the proof will go here
  sorry

end base_eight_to_base_ten_l168_168355


namespace jackson_total_souvenirs_l168_168637

-- Define the conditions
def num_hermit_crabs : ℕ := 45
def spiral_shells_per_hermit_crab : ℕ := 3
def starfish_per_spiral_shell : ℕ := 2

-- Define the calculation based on the conditions
def num_spiral_shells := num_hermit_crabs * spiral_shells_per_hermit_crab
def num_starfish := num_spiral_shells * starfish_per_spiral_shell
def total_souvenirs := num_hermit_crabs + num_spiral_shells + num_starfish

-- Prove that the total number of souvenirs is 450
theorem jackson_total_souvenirs : total_souvenirs = 450 :=
by
  sorry

end jackson_total_souvenirs_l168_168637


namespace find_angle_DSG_l168_168631

-- Define the main problem conditions and proof statement in Lean 4

variables {DOG DGO OGD DGS GDS : ℝ}

-- The conditions
def angle_DOG_eq_angle_DGO (h1 : DOG = DGO) : Prop := h1
def angle_OGD_45 (h2 : OGD = 45) : Prop := h2
def GS_bisects_angle_DGO (h3 : DGO / 2 = DGS) : Prop := h3

-- The proof goal
theorem find_angle_DSG
  (h1 : DOG = DGO)
  (h2 : OGD = 45)
  (h3 : DGO / 2 = DGS)
  (h4 : GDS = DOG)
  : ∃ (DSG : ℝ), DSG = 180 - DGS - GDS := 
   sorry

end find_angle_DSG_l168_168631


namespace sum_of_coeffs_eq_neg30_l168_168964

-- Define the expression based on the given condition
def expr (c : ℤ) : ℤ := -(5 - 2 * c) * (c + 3 * (5 - 2 * c))

-- State the theorem that the sum of the coefficients of the expanded form is -30
theorem sum_of_coeffs_eq_neg30 (c : ℤ) : 
  let expanded_expr := -10 * c^2 + 55 * c - 75 in 
  let sum_of_coeffs := -10 + 55 - 75 in 
  expr c = expanded_expr ∧ sum_of_coeffs = -30 :=
by
  sorry

end sum_of_coeffs_eq_neg30_l168_168964


namespace salmon_conditional_probability_l168_168426

noncomputable def conditional_probability (p q : ℝ) (n : ℕ) : ℝ :=
  if q ≠ 1 ∧ p > 0 ∧ q > 0 then
    (p - q) * ((1 - p) / (1 - q))^n / (p - q * ((1 - p) / (1 - q))^n)
  else 0

theorem salmon_conditional_probability (p q : ℝ) (n : ℕ) (h_p_pos : p > 0) (h_q_pos : q > 0) (h_ind : q ≠ 1) :
  conditional_probability p q n = \frac{(p-q)\left(\frac{1-p}{1-q}\right)^n}{p-q\left(\frac{1-p}{1-q}\right)^n} :=
sorry

end salmon_conditional_probability_l168_168426


namespace smallest_append_digits_l168_168841

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168841


namespace rectangle_length_l168_168434

variable (base_triangle : ℝ) (height_triangle : ℝ) (width_rectangle : ℝ) (length_rectangle : ℝ)

-- Define the areas of triangle and rectangle
def area_triangle (base height : ℝ) : ℝ := (base * height) / 2
def area_rectangle (width length : ℝ) : ℝ := width * length

-- Given the conditions
def base_triangle_condition : base_triangle = 7.2 := rfl
def height_triangle_condition : height_triangle = 7 := rfl
def width_rectangle_condition : width_rectangle = 4 := rfl

-- State the goal
theorem rectangle_length :
  area_triangle base_triangle height_triangle = area_rectangle width_rectangle length_rectangle →
  length_rectangle = 6.3 :=
sorry

end rectangle_length_l168_168434


namespace smallest_n_Tn_gt_10e6_l168_168516

def f (x : ℕ) : ℕ := 2 ^ (Nat.find (exists_dvd_and_not_greater 2 x))

def exists_dvd_and_not_greater (b n : ℕ) : ∃ j : ℕ, b ^ j ∣ n ∧ ¬ (b ^ (j + 1)) ∣ n :=
begin
  sorry
end

def T (n : ℕ) : ℕ :=
  ∑ k in Finset.range (2^n + 1), f k

theorem smallest_n_Tn_gt_10e6 : ∃ n : ℕ, T n > 10^6 ∧ ∀ m : ℕ, T m > 10^6 → n ≤ m :=
begin
  existsi 20,
  split,
  {
    -- Prove that T 20 > 10^6
    sorry
  },
  {
    -- Prove that for all m if T m > 10^6 then 20 ≤ m
    sorry
  }
end

end smallest_n_Tn_gt_10e6_l168_168516


namespace div_remainder_example_l168_168976

theorem div_remainder_example :
  ∃ q r, 256 = 13 * q + r ∧ 0 ≤ r ∧ r < 13 ∧ r = 9 :=
by {
  use 19,
  use 9,
  sorry
}

end div_remainder_example_l168_168976


namespace tan_75_proof_l168_168076

theorem tan_75_proof : Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  have tan_add : ∀ a b: ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b),
    from Real.tan_add
  have tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3, from Real.tan_pi_div_three
  have tan_15 : Real.tan (15 * Real.pi / 180) = 2 - Real.sqrt 3, from sorry
  calc
    Real.tan (75 * Real.pi / 180)
        = Real.tan ((60 + 15) * Real.pi / 180) : by rw [add_mul, ←two_mul]
    ... = (Real.tan (60 * Real.pi / 180) + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (60 * Real.pi / 180) * Real.tan (15 * Real.pi / 180)) : by rw [tan_add]
    ... = (Real.sqrt 3 + (2 - Real.sqrt 3)) / (1 - Real.sqrt 3 * (2 - Real.sqrt 3)) : by rw [tan_60, tan_15]
    ... = (2 : ℝ) / (1 - (2 * Real.sqrt 3 - (Real.sqrt 3 ^ 2))) : by rw [sq]
    ... = (2 : ℝ) / (4 - (2 * Real.sqrt 3)) : by norm_num
    ... = 2 / (2 - Real.sqrt 3)  : by ring_nf
    ... = 2 + Real.sqrt 3 : sorry

end tan_75_proof_l168_168076


namespace monotonically_decreasing_interval_inequality_proof_minimum_integer_a_l168_168198

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x - x^2 + a * x

-- (1) Prove that the monotonically decreasing interval of f(x) with a = 1 is (1, +∞)
theorem monotonically_decreasing_interval : 
  ∀ x : ℝ, 1 < x -> (deriv (λ x : ℝ, Real.log x - x^2 + x) x < 0) :=
begin
  sorry
end

-- (2) Prove that for n ≥ 2, ∑_{k=2}^n 1/ln(k) > 1 - 1/n
theorem inequality_proof (n : ℕ) (h : n ≥ 2) : 
  (∑ k in Finset.range n \ Finset.range 1, 1 / Real.log (k + 1) > 1 - 1 / n) :=
begin
  sorry
end

-- (3) Prove that the minimum value of the integer a such that 
-- f(x) ≤ (1/2 a - 1) x^2 + (2a - 1) x - 1 always holds is 2
theorem minimum_integer_a : 
  ∃ a : ℤ, ∀ x : ℝ, x > 0 -> (Real.log x - x^2 + (a : ℝ) * x ≤ (1/2 * a - 1) * x^2 + (2 * a - 1) * x - 1) ∧ a = 2 :=
begin
  sorry
end

end monotonically_decreasing_interval_inequality_proof_minimum_integer_a_l168_168198


namespace circular_number_divisible_by_27_l168_168179

theorem circular_number_divisible_by_27 
  (digits: Fin 1953 -> ℕ) 
  (A: ℕ) 
  (hA: A = ∑ i in Finset.range 1953, digits i * 10 ^ (1953 - 1 - i) ∧ 27 ∣ A) 
  (n: Fin 1953): 
  27 ∣ (∑ i in Finset.range 1953, digits ((i + n) % 1953) * 10 ^ (1953 - 1 - i)) :=
sorry

end circular_number_divisible_by_27_l168_168179


namespace original_worth_of_goods_l168_168688

section
variables (P Rp R St Total : ℝ)
hypotheses
  (h1 : R = 0.06 * P)
  (h2 : Rp = P - R)
  (h3 : St = 0.10 * Rp)
  (h4 : Total = Rp + St)
  (h5 : Total = 6876.1)

theorem original_worth_of_goods : P = 6650 :=
by
  rw [h1, h2, h3, h4] at h5
  sorry
end

end original_worth_of_goods_l168_168688


namespace intersection_points_l168_168317

-- Parametric equations of the line
def line_x (t : ℝ) : ℝ := t - 1
def line_y (t : ℝ) : ℝ := 2 - t

-- Parametric equations of the curve
def curve_x (θ : ℝ) : ℝ := 3 * Real.cos θ
def curve_y (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Implicit form of the line
def line_implicit (x y : ℝ) : Prop := x + y - 1 = 0

-- Implicit form of the ellipse
def ellipse_implicit (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1

-- To prove: Number of intersection points between the line and the curve is 2
theorem intersection_points : ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ line_implicit (line_x t₁) (line_y t₁) ∧ ellipse_implicit (line_x t₁) (line_y t₁) ∧ line_implicit (line_x t₂) (line_y t₂) ∧ ellipse_implicit (line_x t₂) (line_y t₂) :=
by sorry

end intersection_points_l168_168317


namespace tshirt_cost_l168_168378

variables (T : ℕ)

-- Given conditions
theorem tshirt_cost :
    let initial_amount := 74 in
    let sweater_cost := 9 in
    let shoes_cost := 30 in
    let refund_rate := 0.90 in
    let amount_left := 51 in
    ∃ T : ℕ, 
    initial_amount - (sweater_cost + T + shoes_cost) + (refund_rate * shoes_cost).toNat = amount_left ∧
    T = 14 := 
by
    sorry

end tshirt_cost_l168_168378


namespace maximum_probability_replanting_expectation_X_for_n_5_l168_168404

-- Defining the conditions for the problem
def probability_of_replanting_single_pit : ℚ := 1 / 2

-- The first part: Determine the maximum probability of needing replanting in exactly 4 pits
theorem maximum_probability_replanting (n : ℕ) (h1 : 7 ≤ n) (h2 : n ≤ 8) :
  combinatorial.choose n 4 * (1 / 2)^n = 35 / 128 :=
sorry

-- The second part: Finding the expected value when n = 5
def probability_distribution_X (X : ℕ) : ℚ :=
  if X = 0 then 1 / 32 else
  if X = 1 then 5 / 32 else
  if X = 2 then 5 / 16 else
  if X = 3 then 5 / 16 else
  if X = 4 then 5 / 32 else 1 / 32

theorem expectation_X_for_n_5:
  let n := 5
  let E_X := ∑ X in {0, 1, 2, 3, 4, 5}, X * probability_distribution_X X
  E_X = 5 / 2 :=
sorry

end maximum_probability_replanting_expectation_X_for_n_5_l168_168404


namespace no_prime_pairs_divide_exp_l168_168117

/-- 
  There are no pairs of prime numbers (p, q) such that pq divides (2^p - 1)(2^q - 1).
-/
theorem no_prime_pairs_divide_exp :
  ∀ p q : ℕ, prime p → prime q → ¬ (p * q ∣ (2^p - 1) * (2^q - 1)) := 
by
  intros
  sorry

end no_prime_pairs_divide_exp_l168_168117


namespace meet_for_the_first_time_in_108_seconds_l168_168022

noncomputable def speed_kmh_to_mps (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000) / 3600

theorem meet_for_the_first_time_in_108_seconds :
  let track_length := 4800 -- meters
      speed1 := speed_kmh_to_mps 60 -- first boy's speed in m/s
      speed2 := speed_kmh_to_mps 100 -- second boy's speed in m/s
      relative_speed := speed1 + speed2 -- relative speed when moving in opposite directions
      time_to_meet := track_length / relative_speed
  in time_to_meet = 108 := 
by
  sorry

end meet_for_the_first_time_in_108_seconds_l168_168022


namespace base_eight_to_base_ten_l168_168356

theorem base_eight_to_base_ten (a b c : ℕ) (ha : a = 1) (hb : b = 5) (hc : c = 7) :
  a * 8^2 + b * 8^1 + c * 8^0 = 111 :=
by 
  -- rest of the proof will go here
  sorry

end base_eight_to_base_ten_l168_168356


namespace new_person_weight_l168_168703

-- Define the conditions
def average_weight_increase (weight_increase : ℕ) (num_persons : ℕ) : ℕ :=
  weight_increase * num_persons

def weight_of_new_person (original_weight : ℕ) (total_increase : ℕ) : ℕ :=
  original_weight + total_increase

-- Given conditions
def avg_incr := average_weight_increase 2.5 8
def old_weight := 45

-- Theorem statement
theorem new_person_weight :
  weight_of_new_person old_weight avg_incr = 65 :=
  by sorry

end new_person_weight_l168_168703


namespace ages_of_boys_l168_168731

theorem ages_of_boys (a b c : ℕ) (h1 : a + b + c = 29) (h2 : a = b) (h3 : c = 11) : a = 9 :=
by
  sorry

end ages_of_boys_l168_168731


namespace total_gross_profit_l168_168045

noncomputable def SP_A := 91
noncomputable def SP_B := 110
noncomputable def SP_C := 240

noncomputable def GP_rate := 1.60
noncomputable def d_A := 0.10
noncomputable def d_B := 0.05
noncomputable def d_C := 0.12

noncomputable def CP_A := SP_A / (1 + GP_rate)
noncomputable def CP_B := SP_B / (1 + GP_rate)
noncomputable def CP_C := SP_C / (1 + GP_rate)

noncomputable def DP_A := SP_A * (1 - d_A)
noncomputable def DP_B := SP_B * (1 - d_B)
noncomputable def DP_C := SP_C * (1 - d_C)

noncomputable def GP_A := DP_A - CP_A
noncomputable def GP_B := DP_B - CP_B
noncomputable def GP_C := DP_C - CP_C

noncomputable def TotalGP := GP_A + GP_B + GP_C

theorem total_gross_profit : TotalGP = 227.98 := by
  sorry

end total_gross_profit_l168_168045


namespace emberly_walks_miles_in_march_l168_168104

theorem emberly_walks_miles_in_march (days_not_walked : ℕ) (total_days_in_march : ℕ) (miles_per_hour : ℕ) (hours_per_walk : ℕ) :
  total_days_in_march = 31 → 
  days_not_walked = 4 → 
  hours_per_walk = 1 → 
  miles_per_hour = 4 → 
  let days_walked := total_days_in_march - days_not_walked in
  let total_hours_walked := days_walked * hours_per_walk in
  let total_miles_walked := total_hours_walked * miles_per_hour in
  total_miles_walked = 108 :=
by {
  intros h1 h2 h3 h4,
  have days_walked_def : days_walked = 31 - 4 := by rw [h1, h2],
  have hours_walked_def : total_hours_walked = (31 - 4) * 1 := by rw [days_walked_def, h3],
  have miles_walked_def : total_miles_walked = (27) * 4 := by rw [hours_walked_def, h4],
  have final_result : total_miles_walked = 108 := by rw miles_walked_def,
  exact final_result,
}

end emberly_walks_miles_in_march_l168_168104


namespace cistern_fill_time_l168_168384

noncomputable def fill_rate := 1 / 4 -- cisterns per hour
noncomputable def empty_rate := 1 / 9 -- cisterns per hour
noncomputable def net_rate := fill_rate - empty_rate -- net rate of filling cistern

theorem cistern_fill_time :
  let time_to_fill := 1 / net_rate in
  time_to_fill ≈ 7.2 := 
by
  sorry

end cistern_fill_time_l168_168384


namespace triangle_ab_length_triangle_roots_quadratic_l168_168998

open Real

noncomputable def right_angled_triangle_length_ab (p s : ℝ) : ℝ :=
  (p / 2) - sqrt ((p / 2)^2 - 2 * s)

noncomputable def right_angled_triangle_quadratic (p s : ℝ) : Polynomial ℝ :=
  Polynomial.X^2 - Polynomial.C ((p / 2) + sqrt ((p / 2)^2 - 2 * s)) * Polynomial.X
    + Polynomial.C (2 * s)

theorem triangle_ab_length (p s : ℝ) :
  ∃ (AB : ℝ), AB = right_angled_triangle_length_ab p s ∧
    ∃ (AC BC : ℝ), (AC + BC + AB = p) ∧ (1 / 2 * BC * AC = s) :=
by
  use right_angled_triangle_length_ab p s
  sorry

theorem triangle_roots_quadratic (p s : ℝ) :
  ∃ (AC BC : ℝ), AC + BC = (p / 2) + sqrt ((p / 2)^2 - 2 * s) ∧
    AC * BC = 2 * s ∧
    (Polynomial.aeval AC (right_angled_triangle_quadratic p s) = 0) ∧
    (Polynomial.aeval BC (right_angled_triangle_quadratic p s) = 0) :=
by
  sorry

end triangle_ab_length_triangle_roots_quadratic_l168_168998


namespace train_pass_time_l168_168433

theorem train_pass_time :
  ∀ (L : ℝ) (speed : ℝ) (platform_length : ℝ) (time_pass_platform : ℝ),
  platform_length = 300.024 →
  speed = 15 →
  time_pass_platform = 40 →
  L + platform_length = speed * time_pass_platform →
  (L / speed) ≈ 20 := by
  intro L speed platform_length time_pass_platform
  assume h1 h2 h3 h4
  have L_val : L = 600 - 300.024 := by sorry
  have time_pass_man : L / speed = 299.976 / 15 := by sorry
  have approx_time : (299.976 / 15) ≈ 20 := by sorry
  exact approx_time

end train_pass_time_l168_168433


namespace smallest_digits_to_append_l168_168825

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168825


namespace find_cos_C_l168_168604

variables (A B C : ℝ)
variables (cos_A sin_B : ℝ)
hypothesis h1 : cos_A = 5 / 13
hypothesis h2 : sin_B = 4 / 5

theorem find_cos_C : 
  ∃ (cos_C : ℝ), cos_C = 33 / 65 :=
begin
  sorry
end

end find_cos_C_l168_168604


namespace range_of_t_l168_168270

noncomputable theory

-- Definition of the odd function f
def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 else -x^2

-- Statement of the problem
theorem range_of_t (t : ℝ) :
  (∀ x ∈ set.Icc t (t + 2), f(x + t) ≥ 2 * f(x)) ↔ t ≥ real.sqrt 2 :=
sorry

end range_of_t_l168_168270


namespace silvia_savings_l168_168693

def retail_price : ℝ := 1000
def guitar_center_discount_rate : ℝ := 0.15
def sweetwater_discount_rate : ℝ := 0.10
def guitar_center_shipping_fee : ℝ := 100
def sweetwater_shipping_fee : ℝ := 0

def guitar_center_cost : ℝ := retail_price * (1 - guitar_center_discount_rate) + guitar_center_shipping_fee
def sweetwater_cost : ℝ := retail_price * (1 - sweetwater_discount_rate) + sweetwater_shipping_fee

theorem silvia_savings : guitar_center_cost - sweetwater_cost = 50 := by
  sorry

end silvia_savings_l168_168693


namespace train_length_calc_l168_168052

theorem train_length_calc 
  (speed_kmh : ℝ) (time_sec : ℝ)
  (h_speed : speed_kmh = 108)
  (h_time : time_sec = 50) : 
  let speed_ms := speed_kmh * (5 / 18) in
  let train_length := speed_ms * time_sec in
  train_length = 1500 := 
by
  sorry

end train_length_calc_l168_168052


namespace sin_210_eq_neg_half_l168_168482

theorem sin_210_eq_neg_half :
  sin (210 : ℝ) = -1 / 2 :=
by 
  -- conditions
  have h1 : 210 = 180 + 30 := by norm_num
  have h2 : ∀ θ : ℝ, sin (180 + θ) = -sin θ := λ θ, by norm_num [sin_add]
  have h3 : sin (30 : ℝ) = 1 / 2 := by norm_num [sin_of_real]

  -- proof goals
  rw [h1]
  rw [h2]
  rw [h3]
  norm_num

end sin_210_eq_neg_half_l168_168482


namespace jackie_phil_same_heads_l168_168634

theorem jackie_phil_same_heads:
  let fair_coin := (1 + x),
      biased_coin := (3 + 2 * x),
      generating_function := (fair_coin^2) * biased_coin,
      sum_of_coefficients := 20,  -- Sum of coefficients in generating_function
      coefficient_squares := 3^2 + 8^2 + 7^2 + 2^2,  -- Sum of the squares of coefficients
      total_ways := sum_of_coefficients^2,  -- Number of total possible outcomes
      successful_ways := coefficient_squares,
      probability := successful_ways / total_ways,
      p := 63,
      q := 200 in
  p + q = 263 := 
sorry

end jackie_phil_same_heads_l168_168634


namespace f_zero_f_monotonic_inequality_solution_l168_168527

-- Define the function f and the conditions on f

axiom f : ℝ → ℝ
axiom f_add : ∀ x y : ℝ, f(x + y) = f(x) + f(y) + 1
axiom f_pos : ∀ x : ℝ, x > 0 → f(x) > -1

/-- Part (I) - f(0) = -1 and f is monotonically increasing on ℝ -/

theorem f_zero : f(0) = -1 :=
sorry

theorem f_monotonic : ∀ x₁ x₂ : ℝ, x₁ > x₂ → f(x₁) > f(x₂) :=
sorry

/-- Part (II) - Inequality solution --/

axiom f_one : f(1) = 1

theorem inequality_solution (x : ℝ) : f(x^2 + 2x) + f(1 - x) > 4 ↔ x < -2 ∨ x > 1 :=
sorry

end f_zero_f_monotonic_inequality_solution_l168_168527


namespace boxes_have_n_plus_one_marbles_l168_168396

def number_of_marbles (n : ℕ) (box_idx : ℕ) : ℕ :=
  (box_idx + 1) + ∑ k in (finset.range (n)).filter (λ k, (box_idx + 1) % (k + 1) = 0), 1

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ (∀ m, m ∣ p → m = 1 ∨ m = p)

theorem boxes_have_n_plus_one_marbles (n : ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ n → number_of_marbles n i = n + 1) ↔ is_prime (n + 1) :=
sorry

end boxes_have_n_plus_one_marbles_l168_168396


namespace find_b_l168_168201

noncomputable def curve (a b x : ℝ) : ℝ := x^3 + a*x + b
noncomputable def line (k x : ℝ) : ℝ := k*x + 1

theorem find_b :
  ∃ (a b : ℝ),
    -- The slope k is 2
    let k := 2 in
    -- The point of tangency is (1, 3)
    (curve a b 1 = 3) ∧ (line k 1 = 3) ∧
    -- The slope of the line is equal to the slope of the curve at x = 1
    (deriv (λ x, curve a b x) 1 = k) ∧
    -- We need to prove b = 3 under these conditions
    b = 3 :=
begin
  use [-1, 3],
  split,
  { simp [curve], },
  split,
  { simp [line], },
  split,
  { simp [curve, deriv],
    sorry, -- Here you would normally provide the actual proof steps
  },
  { refl, }
end

end find_b_l168_168201


namespace opposite_of_abs_frac_l168_168319

theorem opposite_of_abs_frac (h : 0 < (1 : ℝ) / 2023) : -|((1 : ℝ) / 2023)| = -(1 / 2023) := by
  sorry

end opposite_of_abs_frac_l168_168319


namespace f_composite_l168_168560

def f (a : ℝ) : ℝ := ∫ x in 0..a, sin x

theorem f_composite (π_div_2 : ℝ) (cos_val : ℝ) : (π_div_2 = Real.pi / 2) → (cos_val = cos 1) → f(f(π_div_2)) = 1 - cos_val :=
by
  intros h1 h2
  rw [←h1]
  -- skipped: proof steps according to solution
  rw [←h2]
  sorry

end f_composite_l168_168560


namespace myOperation_identity_l168_168981

variable {R : Type*} [LinearOrderedField R]

def myOperation (a b : R) : R := (a - b) ^ 2

theorem myOperation_identity (x y : R) : myOperation ((x - y) ^ 2) ((y - x) ^ 2) = 0 := 
by 
  sorry

end myOperation_identity_l168_168981


namespace equivalent_discount_l168_168694

theorem equivalent_discount {x : ℝ} (h₀ : x > 0) :
    let first_discount := 0.10
    let second_discount := 0.20
    let single_discount := 0.28
    (1 - (1 - first_discount) * (1 - second_discount)) = single_discount := by
    sorry

end equivalent_discount_l168_168694


namespace imaginary_unit_powers_l168_168881

-- Define the properties of i
def is_imaginary_unit (i : ℂ) : Prop :=
  i^2 = -1

-- Define the necessary conditions
variables (i : ℂ) (n : ℕ)
hypothesis (h_i_pow2 : i^2 = -1)
hypothesis (h_i_pow3 : i^3 = -i)
hypothesis (h_i_pow4 : i^4 = 1)

-- Prove the required equalities
theorem imaginary_unit_powers :
  i^(4*n + 1) = i ∧
  i^(4*n + 2) = -1 ∧
  i^(4*n + 3) = -i :=
sorry

end imaginary_unit_powers_l168_168881


namespace spatial_geometric_figures_l168_168875

theorem spatial_geometric_figures :
  (∀ (P : Prism), is_lateral_face_parallelogram P) ∧
  (∀ (T : Triangle), rotates_around_side_correct T → is_cone_formed T = false) ∧
  (∀ (Q : QuadrilateralPrism), is_regular_quadrilateral_prism Q → is_rectangular_prism Q) ∧
  (∀ (Py : Pyramid), intersects_not_parallel_to_base Py → is_frustum Py = false) := sorry

-- Definitions for the geometric objects and properties used in the theorem
structure Prism := (lateral_faces : List Face) 
def is_lateral_face_parallelogram (P : Prism) : Prop := ∀ face ∈ P.lateral_faces, is_parallelogram face

structure Triangle := (side1 side2 side3 : ℝ)
def rotates_around_side_correct (T : Triangle) : Prop := sorry  -- this would depend on further geometric properties
def is_cone_formed (T : Triangle) : Boolean := sorry  -- and computing the resulting solid

structure QuadrilateralPrism := (base : Quadrilateral) (height : ℝ)
def is_regular_quadrilateral_prism (Q : QuadrilateralPrism) : Prop := sorry  -- requires base to be square, etc.
def is_rectangular_prism (Q : QuadrilateralPrism) : Prop := sorry  -- requires definition of rectangular prism

structure Pyramid := (base : Polygon) (height : ℝ)
def intersects_not_parallel_to_base (Py : Pyramid) : Prop := sorry  -- denotes the geometric condition
def is_frustum (Py : Pyramid) : Boolean := sorry  -- if intersected part forms a frustum

end spatial_geometric_figures_l168_168875


namespace minimum_area_AMBN_l168_168617

/-
We define the line l passing through point M(1,0) with an angle α
We define the polar equation of the curve C and convert it to Cartesian coordinates.
We prove the parametric equation of the line l, the Cartesian equation of the curve C,
and the minimum area of the quadrilateral AMBN.
-/

noncomputable def line_through_point_makes_angle (M : ℝ × ℝ) (α : ℝ) : ℝ × ℝ :=
  (1 + cos α, sin α)

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

lemma parametric_equation_of_line (α t : ℝ) :
  (1 + t * cos α, t * sin α) = line_through_point_makes_angle (1, 0) α :=
begin
  sorry,
end

lemma cartesian_equation_of_curve (ρ θ : ℝ) :
  ρ^2 * (sin θ)^2 - 4 * ρ * (cos θ) = 0 ↔ (polar_to_cartesian ρ θ).snd^2 = 4 * (polar_to_cartesian ρ θ).fst :=
begin
  sorry,
end

theorem minimum_area_AMBN (α : ℝ) :
  (∀ α : ℝ, α ≠ 0 → α ≠ π / 2 →
  ∃ t1 t2 : ℝ, t1 + t2 = 4 * cos α / (sin α)^2 ∧
  t1 * t2 = -4 / (sin α)^2) →
  ∃ area : ℝ, area = 32 :=
begin
  sorry,
end

end minimum_area_AMBN_l168_168617


namespace buy_tshirts_l168_168321

theorem buy_tshirts
  (P T : ℕ)
  (h1 : 3 * P + 6 * T = 1500)
  (h2 : P + 12 * T = 1500)
  (budget : ℕ)
  (budget_eq : budget = 800) :
  (budget / T) = 8 := by
  sorry

end buy_tshirts_l168_168321


namespace earnings_per_puppy_l168_168957

def daily_pay : ℝ := 40
def total_earnings : ℝ := 76
def num_puppies : ℕ := 16

theorem earnings_per_puppy : (total_earnings - daily_pay) / num_puppies = 2.25 := by
  sorry

end earnings_per_puppy_l168_168957


namespace ellipse_CD_distance_l168_168083

theorem ellipse_CD_distance :
  (sqrt ((4 : ℝ)^2 + (2 : ℝ)^2)) = 2 * sqrt 5 := 
by {
  sorry
}

end ellipse_CD_distance_l168_168083


namespace corn_harvest_l168_168453

theorem corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) (total_bushels : ℕ) :
  rows = 5 →
  stalks_per_row = 80 →
  stalks_per_bushel = 8 →
  total_bushels = (rows * stalks_per_row) / stalks_per_bushel →
  total_bushels = 50 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, mul_comm 5 80] at h4
  norm_num at h4
  exact h4

end corn_harvest_l168_168453


namespace exists_unique_real_l168_168533

-- Define the conditions of the sequence
def satisfies_conditions (a : ℕ → ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ 1 ≤ j ∧ i + j ≤ 1997 → a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1

-- Define the existence and uniqueness of the real number x
theorem exists_unique_real (a : ℕ → ℕ) (h : satisfies_conditions a) :
  ∃! x : ℝ, ∀ n, 1 ≤ n ∧ n ≤ 1997 → a n = int.floor (n * x) :=
sorry

end exists_unique_real_l168_168533


namespace sum_of_squares_of_medians_l168_168367

/-- Define the sides of the triangle -/
def side_lengths := (13, 13, 10)

/-- Define the triangle as isosceles with given side lengths -/
def isosceles_triangle (a b c : ℝ) := (a = b) ∧ (a ≠ c)

/-- Define the median lengths calculation -/
noncomputable def median_length (a b c : ℝ) : ℝ :=
  if h : isosceles_triangle a b c then
    let AD := Real.sqrt (a ^ 2 - (c / 2) ^ 2) in
    let BE_CF := Real.sqrt ((a / 2) ^ 2 + (3 / 4) * c ^ 2) in
    AD^2 + BE_CF^2 + BE_CF^2
  else
    0

/-- The sum of the squares of the lengths of the medians for the given triangle is -/
theorem sum_of_squares_of_medians : median_length 13 13 10 = 447.5 := by
  sorry

end sum_of_squares_of_medians_l168_168367


namespace figure_can_be_cut_and_reassembled_into_square_l168_168457

-- Define the conditions
def is_square_area (n: ℕ) : Prop := ∃ k: ℕ, k * k = n

def can_form_square (area: ℕ) : Prop :=
area = 18 ∧ ¬ is_square_area area

-- The proof statement
theorem figure_can_be_cut_and_reassembled_into_square (area: ℕ) (hf: area = 18): 
  can_form_square area → ∃ (part1 part2 part3: Set (ℕ × ℕ)), true :=
by
  sorry

end figure_can_be_cut_and_reassembled_into_square_l168_168457


namespace fraction_of_garden_occupied_is_correct_l168_168924

-- Definitions
def garden_length : ℕ := 40
def garden_width : ℕ := 8
def triangle_leg_length : ℕ := 5
def sandbox_side_length : ℕ := 5

-- Calculations
def triangle_area : ℕ := (triangle_leg_length * triangle_leg_length) / 2
def total_triangle_area : ℕ := 2 * triangle_area
def sandbox_area : ℕ := sandbox_side_length * sandbox_side_length
def total_occupied_area : ℕ := total_triangle_area + sandbox_area
def garden_area : ℕ := garden_length * garden_width
def occupied_fraction := (total_occupied_area.to_rat / garden_area.to_rat)

-- Proof statement (the final theorem to verify the result)
theorem fraction_of_garden_occupied_is_correct : occupied_fraction = 5 / 32 := by
  sorry

end fraction_of_garden_occupied_is_correct_l168_168924


namespace cube_sum_div_by_9_implies_prod_div_by_3_l168_168681

theorem cube_sum_div_by_9_implies_prod_div_by_3 
  {a1 a2 a3 a4 a5 : ℤ} 
  (h : 9 ∣ a1^3 + a2^3 + a3^3 + a4^3 + a5^3) : 
  3 ∣ a1 * a2 * a3 * a4 * a5 := by
  sorry

end cube_sum_div_by_9_implies_prod_div_by_3_l168_168681


namespace smallest_digits_to_append_l168_168867

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168867


namespace z_is_real_iff_z_is_imaginary_iff_z_is_pure_imaginary_iff_l168_168519

noncomputable def z (m : ℝ) : ℂ := (m^2 + m - 6) / m + (m^2 - 2 * m) * complex.I

def is_real (z : ℂ) : Prop := z.im = 0
def is_imaginary (z : ℂ) : Prop := z.re = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem z_is_real_iff (m : ℝ) : is_real (z m) ↔ m = 2 :=
by sorry

theorem z_is_imaginary_iff (m : ℝ) : is_imaginary (z m) ↔ m ≠ 2 ∧ m ≠ 0 :=
by sorry

theorem z_is_pure_imaginary_iff (m : ℝ) : is_pure_imaginary (z m) ↔ m = -3 :=
by sorry

end z_is_real_iff_z_is_imaginary_iff_z_is_pure_imaginary_iff_l168_168519


namespace proposition_p_is_false_iff_l168_168564

def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 3)

def p (a : ℝ) : Prop := ∃ x : ℝ, f x < a

theorem proposition_p_is_false_iff (a : ℝ) : (¬p a) ↔ (a < 5) :=
by sorry

end proposition_p_is_false_iff_l168_168564


namespace smallest_digits_to_append_l168_168866

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168866


namespace smallest_number_of_digits_to_append_l168_168803

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168803


namespace firetruck_reachable_area_l168_168068

-- Defining the conditions of the problem
def firetruck_speed_highway := 60 -- miles per hour
def firetruck_speed_prairie := 15 -- miles per hour
def time_available := (5 : ℕ) / 60 -- hours

-- Calculate the area reachable by the firetruck
theorem firetruck_reachable_area :
  let distance_highway := firetruck_speed_highway * time_available,
      distance_prairie (x : ℝ) := firetruck_speed_prairie * (time_available - x / firetruck_speed_highway),
      integrand (x : ℝ) := (π * (distance_prairie x / 4)^2) / 16,
      total_area := 4 * ∫ x in (0 : ℝ)..distance_highway, integrand x
  in m + n = 137 :=
by
  have distance_highway := firetruck_speed_highway * time_available,
  have distance_prairie (x : ℝ) := firetruck_speed_prairie * (time_available - x / firetruck_speed_highway),
  have integrand (x : ℝ) := (π * (distance_prairie x / 4)^2) / 16,
  have total_area := 4 * ∫ x in (0 : ℝ)..distance_highway, integrand x,
  sorry

end firetruck_reachable_area_l168_168068


namespace coloring_problem_minimum_number_of_colors_l168_168970

-- Define the problem of coloring integers with desired properties
def minColors : ℕ :=
  6

theorem coloring_problem (a b c : ℕ) (h₀ : a < b) (h₁ : b < c) (h₂ : a ∣ b) (h₃ : b ∣ c) : 
(a ≠ b ∧ a ≠ c ∧ b ≠ c → a ∣ b ∨ b ∣ c → false) :=
sorry

theorem minimum_number_of_colors : 
  ∃ (colors : ℕ), 
    (colors = minColors ∧
     ∀ (a b c : ℕ), 
       (1 ≤ a ∧ a ≤ 2007) → 
       (1 ≤ b ∧ b ≤ 2007) → 
       (1 ≤ c ∧ c ≤ 2007) → 
       a ≠ b → a ≠ c → b ≠ c →
       ¬ ( color a = color b ∧ color b = color c → a ∣ b ∧ b ∣ c )
    ) :=
begin
  use minColors,
  split,
  { refl, },
  intros a a_pos a_le b b_pos b_le c c_pos c_le a_ne_b a_ne_c b_ne_c color_same,
  sorry
end

end coloring_problem_minimum_number_of_colors_l168_168970


namespace recommended_screen_time_l168_168304

def morning_minutes : ℕ := 45
def evening_minutes : ℕ := 75
def total_minutes : ℕ := morning_minutes + evening_minutes
def minutes_to_hours (m : ℕ) : ℕ := m / 60

theorem recommended_screen_time : minutes_to_hours total_minutes = 2 := 
by 
  simp [total_minutes, morning_minutes, evening_minutes, minutes_to_hours]
  sorry

end recommended_screen_time_l168_168304


namespace product_of_primes_impossible_l168_168258

theorem product_of_primes_impossible (q : ℕ) (hq1 : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬ ∀ i ∈ Finset.range (q-1), ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ (i^2 + i + q = p1 * p2) :=
sorry

end product_of_primes_impossible_l168_168258


namespace num_palindromic_seven_digit_integers_l168_168347

theorem num_palindromic_seven_digit_integers : 
  let digits := {5, 6, 7}
  let count_palindromes := (λ a b c d : ℕ, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits) in
  ∃ (num_palindromes : ℕ), num_palindromes = 3^4 ∧ count_palindromes = (num_palindromes) :=
sorry

end num_palindromic_seven_digit_integers_l168_168347


namespace fraction_sum_equals_l168_168492

theorem fraction_sum_equals :
  (1 / 20 : ℝ) + (2 / 10 : ℝ) + (4 / 40 : ℝ) = 0.35 :=
by
  sorry

end fraction_sum_equals_l168_168492


namespace volume_problem_l168_168087

theorem volume_problem 
  (m n p : ℕ) 
  (hb_rel_prime : nat.coprime n p)
  (volume_eq : (m + n * Real.pi) / p = (462 + 40 * Real.pi) / 3) : 
  m + n + p = 505 := 
by 
  sorry

end volume_problem_l168_168087


namespace product_diverges_to_infinity_l168_168070

noncomputable def infinite_product := ∏ n in (finset.range 100), real.pow 3 ((n : ℝ)/(n + 2))

theorem product_diverges_to_infinity : real.log (infinite_product) = ∞ := 
by 
  sorry

end product_diverges_to_infinity_l168_168070


namespace number_of_solutions_l168_168154

def fractional_part (r : ℝ) : ℝ := r - r.floor

theorem number_of_solutions :
  (∃ x : ℝ, 1 ≤ x ∧ x < 2 ∧ fractional_part (x^2018) = fractional_part (x^2017)) →
  ∃ count : ℕ, count = 2^2017 := sorry

end number_of_solutions_l168_168154


namespace emma_telephone_numbers_count_l168_168438

/-- Given the form of Emma's telephone numbers and the restrictions on the digits,
    prove that there are 0 possible telephone numbers she can have. -/
theorem emma_telephone_numbers_count : 
  ∀ (a b c d e f g h : ℕ), (a ∈ {2, 3, 6, 7, 8, 9} ∧ 
  b ∈ {2, 3, 6, 7, 8, 9} ∧ 
  c ∈ {2, 3, 6, 7, 8, 9} ∧ 
  d ∈ {2, 3, 6, 7, 8, 9} ∧ 
  e ∈ {2, 3, 6, 7, 8, 9} ∧ 
  f ∈ {2, 3, 6, 7, 8, 9} ∧ 
  g ∈ {2, 3, 6, 7, 8, 9} ∧ 
  h ∈ {2, 3, 6, 7, 8, 9}) ∧ 
  (a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧ g < h) 
  → false :=
by 
  sorry

end emma_telephone_numbers_count_l168_168438


namespace max_distance_on_ellipse_l168_168549

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) 

theorem max_distance_on_ellipse :
  let A := (0, 2)
  ∃ B : ℝ × ℝ, (B.1^2 + 6 * B.2^2 = 6) ∧ 
  ∀ B', (B'.1^2 + 6 * B'.2^2 = 6) → distance A B' ≤ distance A B ∧ 
  distance A B = 3 * real.sqrt 30 / 5 :=
by
  sorry

end max_distance_on_ellipse_l168_168549


namespace find_f_expr_l168_168590

-- Given function definition and the constraint
def f (x : ℝ) : ℝ := sorry

-- Define the given condition
axiom f_condition (x : ℝ) (hx : x ≥ 0) : f (real.sqrt x + 1) = x + 3

-- Prove the target property
theorem find_f_expr : ∀ x : ℝ, x ≥ 1 → f x = x^2 - 2*x + 4 :=
by
  intros x hx
  sorry

end find_f_expr_l168_168590


namespace initial_percentage_reduction_l168_168322
noncomputable section

variable (P : ℝ) (x : ℝ)

theorem initial_percentage_reduction :
  let reduced_price := P * (1 - x) * 0.7 in
  reduced_price * 1.9047619047619048 = P →
  x = 0.25 :=
by
  intro h
  sorry

end initial_percentage_reduction_l168_168322


namespace polynomial_product_expansion_l168_168493

theorem polynomial_product_expansion (x : ℝ) : (x^2 + 3 * x + 3) * (x^2 - 3 * x + 3) = x^4 - 3 * x^2 + 9 := 
by sorry

end polynomial_product_expansion_l168_168493


namespace cauchy_problem_solution_l168_168978

noncomputable def y_solution (x : ℝ) := (2 - 3 * x) * Real.exp (2 * x)

theorem cauchy_problem_solution :
  ∀ (x : ℝ), ∃ (y : ℝ → ℝ), 
    (∀ x, (y'' x) - 4 * (y' x) + 4 * (y x) = 0) ∧
    y 0 = 2 ∧ 
    deriv y 0 = 1 ∧
    y = y_solution :=
by
  sorry

end cauchy_problem_solution_l168_168978


namespace points_on_circle_l168_168171

open EuclideanGeometry

variables {A B C I S T : Point}

theorem points_on_circle (h₁ : AB < AC)
  (h₂ : incenter A B C I)
  (h₃ : perp_bisector B C ∩ angle_bisector A B C S)
  (h₄ : angle_bisector B A C ∩ angle_bisector C A B T) :
  cyclic [C, I, S, T] :=
begin
  sorry
end

end points_on_circle_l168_168171


namespace painted_cubes_l168_168034

theorem painted_cubes {n : ℕ} (h₁ : n = 4) 
  (h₂ : ∀ (i : ℕ), i ∈ {0, 1, 2, 3}) : ∃ (count : ℕ), count = 20 :=
by
  sorry

end painted_cubes_l168_168034


namespace todd_initial_gum_l168_168751

theorem todd_initial_gum (x : ℝ)
(h1 : 150 = 0.25 * x)
(h2 : x + 150 = 890) :
x = 712 :=
by
  -- Here "by" is used to denote the beginning of proof block
  sorry -- Proof will be filled in later.

end todd_initial_gum_l168_168751


namespace maximum_value_A_l168_168147

theorem maximum_value_A
  (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  let A := (x - y) * Real.sqrt (x ^ 2 + y ^ 2) +
           (y - z) * Real.sqrt (y ^ 2 + z ^ 2) +
           (z - x) * Real.sqrt (z ^ 2 + x ^ 2) +
           Real.sqrt 2,
      B := (x - y) ^ 2 + (y - z) ^ 2 + (z - x) ^ 2 + 2 in
  A / B ≤ 1 / Real.sqrt 2 :=
sorry

end maximum_value_A_l168_168147


namespace infinite_series_sum_l168_168469

theorem infinite_series_sum :
  let S := 1 - (1/3) - (1/6) + (1/12) - (1/24) - (1/48) + (1/96) - (1/192) - ...
  ∑' (n : ℕ), S[n]
  T = 25 / 48 :=
sorry

end infinite_series_sum_l168_168469


namespace isosceles_triangle_area_l168_168345

theorem isosceles_triangle_area {a b h : ℝ} (h1 : a = 13) (h2 : b = 13) (h3 : h = 10) :
  ∃ (A : ℝ), A = 60 ∧ A = (1 / 2) * h * 12 :=
by
  sorry

end isosceles_triangle_area_l168_168345


namespace square_distance_from_B_to_center_l168_168036

noncomputable def distance_squared (a b : ℝ) : ℝ := a^2 + b^2

theorem square_distance_from_B_to_center :
  ∀ (a b : ℝ),
    (a^2 + (b + 8)^2 = 75) →
    ((a + 2)^2 + b^2 = 75) →
    distance_squared a b = 122 :=
by
  intros a b h1 h2
  sorry

end square_distance_from_B_to_center_l168_168036


namespace exists_pos_real_alpha_l168_168487

noncomputable def exists_alpha : Prop :=
  ∃ α : ℝ, α > 0 ∧ (∀ n : ℕ, n > 0 → (⌊α * (n : ℝ)⌋ : ℤ) - (n : ℤ)) % 2 = 0

theorem exists_pos_real_alpha : exists_alpha := sorry

end exists_pos_real_alpha_l168_168487


namespace route_y_is_quicker_l168_168670

def distance_X : ℝ := 8
def speed_X : ℝ := 35
def time_X : ℝ := (distance_X / speed_X) * 60

def distance_Y_total : ℝ := 7
def distance_Y_normal : ℝ := 6
def speed_Y_normal : ℝ := 45
def time_Y_normal : ℝ := (distance_Y_normal / speed_Y_normal) * 60

def distance_Y_construction : ℝ := 1
def speed_Y_construction : ℝ := 15
def time_Y_construction : ℝ := (distance_Y_construction / speed_Y_construction) * 60

def time_Y : ℝ := time_Y_normal + time_Y_construction

theorem route_y_is_quicker : (time_X - time_Y ≈ 1.71) :=
by
  sorry

end route_y_is_quicker_l168_168670


namespace sin_cos_identity_l168_168510

theorem sin_cos_identity :
  sin (21 * Real.pi / 180) * cos (39 * Real.pi / 180) + cos (21 * Real.pi / 180) * sin (39 * Real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end sin_cos_identity_l168_168510


namespace max_product_913_l168_168760

-- Define the condition that ensures the digits are from the set {3, 5, 8, 9, 1}
def valid_digits (digits : List ℕ) : Prop :=
  digits = [3, 5, 8, 9, 1]

-- Define the predicate for a valid three-digit and two-digit integer
def valid_numbers (a b c d e : ℕ) : Prop :=
  valid_digits [a, b, c, d, e] ∧
  ∃ x y, 100 * x + 10 * 1 + y = 10 * d + e ∧ d ≠ 1 ∧ a ≠ 1

-- Define the product function
def product (a b c d e : ℕ) : ℕ :=
  (100 * a + 10 * b + c) * (10 * d + e)

-- State the theorem
theorem max_product_913 : ∀ (a b c d e : ℕ), valid_numbers a b c d e → 
(product a b c d e) ≤ (product 9 1 3 8 5) :=
by
  intros a b c d e
  unfold valid_numbers product 
  sorry

end max_product_913_l168_168760


namespace regular_octahedron_faces_l168_168973

-- Assume the regular octahedron as the condition
def is_regular_octahedron (P : Type) [Polyhedron P] : Prop :=
  (is_platonic_solid P) ∧ (faces_count P = 8)

-- We need to show that the number of faces is indeed 8 for a regular octahedron
theorem regular_octahedron_faces :
  ∀ (P : Type) [Polyhedron P], is_regular_octahedron P → faces_count P = 8
by
  intros P hP octa
  cases octa
  sorry

end regular_octahedron_faces_l168_168973


namespace isosceles_triangle_incircle_tangency_points_distance_l168_168615

theorem isosceles_triangle_incircle_tangency_points_distance
  (A B C P Q : Point ℝ) (AC BC : Real)
  (h_isosceles : AC = BC)
  (r : Real := 3)
  (h_incircle_radius : distance (incircle_center A B C) P = r)
  (h_PQ_touches_AC : touches_ac A P)
  (h_PQ_touches_BC : touches_bc B Q)
  (L : Line ℝ) (h_l_tangent : tangent_to_incircle L r)
  (h_l_parallel : parallel L (Segment A C))
  (h_B_l_distance : distance B L = 3) :
  distance P Q = 3 * Real.sqrt 3 := sorry

end isosceles_triangle_incircle_tangency_points_distance_l168_168615


namespace general_formula_a_n_sum_of_b_n_l168_168265

def a_n (n : ℕ) : ℝ := 4 * n - 2

def S_n (n : ℕ) : ℝ := (n : ℝ) * (a_n n) / 2

def b_n (n : ℕ) : ℝ := 4 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℝ := ∑ k in Finset.range n, b_n k

theorem general_formula_a_n (n : ℕ) : a_n n = 4 * n - 2 :=
  by sorry

theorem sum_of_b_n (n : ℕ) : T_n n = n / (2 * n + 1) :=
  by sorry

end general_formula_a_n_sum_of_b_n_l168_168265


namespace group_D_forms_a_definite_set_l168_168060

theorem group_D_forms_a_definite_set : 
  ∃ (S : Set ℝ), S = { x : ℝ | x = 1 ∨ x = -1 } :=
by
  sorry

end group_D_forms_a_definite_set_l168_168060


namespace exists_circle_containing_points_l168_168895

-- Definitions for our conditions
variables {C : Type} [metric_space C] {O : C} {R : ℝ}
variables {A B : C}

-- Conditions
def is_inside_circle (O : C) (R : ℝ) (P : C) : Prop := dist O P < R

-- Main statement
theorem exists_circle_containing_points (h_circle : is_inside_circle O R A) (h_circle2 : is_inside_circle O R B) :
  ∃ (P : C) (r : ℝ), is_inside_circle O R P ∧ dist P A = r ∧ dist P B = r ∧ r > 0 ∧ ∀ (Q : C), dist P Q = r → dist O Q < R :=
sorry

end exists_circle_containing_points_l168_168895


namespace passes_through_fixed_point_l168_168010

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a^(x-2) - 3

theorem passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -2 :=
by
  sorry

end passes_through_fixed_point_l168_168010


namespace number_of_seven_digit_palindromes_l168_168349

theorem number_of_seven_digit_palindromes : 
  let choices := {5, 6, 7} in
  let num_choices := (choices.card : ℕ) in
  (num_choices * num_choices * num_choices * num_choices) = 81 :=
by
  let choices := {5, 6, 7}
  let num_choices := (choices.card : ℕ)
  have h : (num_choices = 3) := rfl
  rw [h]
  norm_num

end number_of_seven_digit_palindromes_l168_168349


namespace initial_investment_l168_168222

noncomputable def doubling_period (r : ℝ) : ℝ := 70 / r
noncomputable def investment_after_doubling (P : ℝ) (n : ℝ) : ℝ := P * (2 ^ n)

theorem initial_investment (total_amount : ℝ) (years : ℝ) (rate : ℝ) (initial : ℝ) :
  rate = 8 → total_amount = 28000 → years = 18 → 
  initial = total_amount / (2 ^ (years / (doubling_period rate))) :=
by
  intros hrate htotal hyears
  simp [doubling_period, investment_after_doubling] at *
  rw [hrate, htotal, hyears]
  norm_num
  sorry

end initial_investment_l168_168222


namespace chord_length_of_circle_l168_168127

theorem chord_length_of_circle (x y : ℝ) (h1 : (x - 0)^2 + (y - 2)^2 = 4) (h2 : y = x) : 
  length_of_chord_intercepted_by_line_eq_2sqrt2 :=
sorry

end chord_length_of_circle_l168_168127


namespace smallest_digits_to_append_l168_168868

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168868


namespace equilateral_triangle_of_equal_heights_and_inradius_l168_168679

theorem equilateral_triangle_of_equal_heights_and_inradius 
  {a b c h1 h2 h3 r : ℝ} (h1_eq : h1 = 2 * r * (a * b * c) / a) 
  (h2_eq : h2 = 2 * r * (a * b * c) / b) 
  (h3_eq : h3 = 2 * r * (a * b * c) / c) 
  (sum_heights_eq : h1 + h2 + h3 = 9 * r) : a = b ∧ b = c ∧ c = a :=
by
  sorry

end equilateral_triangle_of_equal_heights_and_inradius_l168_168679


namespace AmandaWillSpend_l168_168440

/--
Amanda goes shopping and sees a sale where different items have different discounts.
She wants to buy a dress for $50 with a 30% discount, a pair of shoes for $75 with a 25% discount,
and a handbag for $100 with a 40% discount.
After applying the discounts, a 5% tax is added to the final price.
Prove that Amanda will spend $158.81 to buy all three items after the discounts and tax have been applied.
-/
noncomputable def totalAmount : ℝ :=
  let dressPrice := 50
  let dressDiscount := 0.30
  let shoesPrice := 75
  let shoesDiscount := 0.25
  let handbagPrice := 100
  let handbagDiscount := 0.40
  let taxRate := 0.05
  let dressFinalPrice := dressPrice * (1 - dressDiscount)
  let shoesFinalPrice := shoesPrice * (1 - shoesDiscount)
  let handbagFinalPrice := handbagPrice * (1 - handbagDiscount)
  let subtotal := dressFinalPrice + shoesFinalPrice + handbagFinalPrice
  let tax := subtotal * taxRate
  let totalAmount := subtotal + tax
  totalAmount

theorem AmandaWillSpend : totalAmount = 158.81 :=
by
  -- proof goes here
  sorry

end AmandaWillSpend_l168_168440


namespace correct_assignment_statement_l168_168375

theorem correct_assignment_statement (a b : ℕ) : 
  (2 = a → False) ∧ 
  (a = a + 1 → True) ∧ 
  (a * b = 2 → False) ∧ 
  (a + 1 = a → False) :=
by {
  sorry
}

end correct_assignment_statement_l168_168375


namespace ones_digit_of_73_pow_351_l168_168386

-- Definition of the problem in Lean 4
theorem ones_digit_of_73_pow_351 : (73 ^ 351) % 10 = 7 := by
  sorry

end ones_digit_of_73_pow_351_l168_168386


namespace alpha_winning_strategy_l168_168757

-- Definitions based on the game conditions
def initial_chessboard (n : ℕ) : matrix (fin n) (fin n) bool := sorry

def is_white (board : matrix (fin n) (fin n) bool) (i j : fin n) : bool := sorry

def valid_move (board : matrix (fin n) (fin n) bool) (start finish : fin n × fin n) : bool := sorry

def move_rook (board : matrix (fin n) (fin n) bool) (start finish : fin n × fin n) : matrix (fin n) (fin n) bool := sorry

-- Proof problem statement
theorem alpha_winning_strategy (n : ℕ) (board : matrix (fin n) (fin n) bool) 
    (initial_black : board 0 0 = ff)
    (rook : fin n × fin n)
    (valid_move_alpha : ∀ (start finish : fin n × fin n), valid_move board start finish → move_rook board start finish = sorry) 
  : sorry :=
sorry

end alpha_winning_strategy_l168_168757


namespace hyperbola_eccentricity_correct_l168_168994

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
    (h_asymp : ∀ x y : ℝ, (x, y) = (0, 2) → abs (2 * b / sqrt (a^2 + b^2)) = 1) 
    : ℝ := sorry

theorem hyperbola_eccentricity_correct (a b : ℝ) (ha : a > 0) (hb : b > 0)
    (h_asymp : ∀ x y : ℝ, (x, y) = (0, 2) → abs (2 * b / sqrt (a^2 + b^2)) = 1) 
    : hyperbola_eccentricity a b ha hb h_asymp = 2 * sqrt 3 / 3 := sorry

end hyperbola_eccentricity_correct_l168_168994


namespace difference_between_local_and_face_value_of_7_in_65793_l168_168890

theorem difference_between_local_and_face_value_of_7_in_65793 :
  let numeral := 65793
  let digit := 7
  let place := 100
  let local_value := digit * place
  let face_value := digit
  local_value - face_value = 693 := 
by
  sorry

end difference_between_local_and_face_value_of_7_in_65793_l168_168890


namespace domain_of_f_l168_168956

def f (x : ℝ) : ℝ := real.sqrt (2 + x) + real.sqrt (1 - x)

theorem domain_of_f :
  ∀ x, (2 + x ≥ 0) ∧ (1 - x ≥ 0) ↔ x ∈ set.Icc (-2 : ℝ) 1 :=
by
  sorry

end domain_of_f_l168_168956


namespace khalil_total_payment_l168_168279

def cost_dog := 60
def cost_cat := 40
def cost_parrot := 70
def cost_rabbit := 50

def num_dogs := 25
def num_cats := 45
def num_parrots := 15
def num_rabbits := 10

def total_cost := num_dogs * cost_dog + num_cats * cost_cat + num_parrots * cost_parrot + num_rabbits * cost_rabbit

theorem khalil_total_payment : total_cost = 4850 := by
  sorry

end khalil_total_payment_l168_168279


namespace unit_square_divisible_l168_168531

theorem unit_square_divisible (n : ℕ) (h: n ≥ 6) : ∃ squares : ℕ, squares = n :=
by
  sorry

end unit_square_divisible_l168_168531


namespace max_different_ages_l168_168702

def average_age : ℝ := 10
def std_deviation : ℝ := 8
def lower_limit : ℤ := ⌊average_age - std_deviation⌋.toInt
def upper_limit : ℤ := ⌈average_age + std_deviation⌉.toInt

theorem max_different_ages : (upper_limit - lower_limit + 1) = 17 := 
by 
  -- definitions from conditions
  have avg : ℝ := 10
  have std_dev : ℝ := 8
  have low_lim : ℤ := ⌊avg - std_dev⌋.toInt
  have upp_lim : ℤ := ⌈avg + std_dev⌉.toInt
  -- intermediate results
  have lower_calc : low_lim = 2 := by sorry  -- calculation for lower limit
  have upper_calc : upp_lim = 18 := by sorry -- calculation for upper limit
  -- final calculation
  show (upp_lim - low_lim + 1) = 17, from by {
    rw [lower_calc, upper_calc],  -- substitute calculated limits
    calc
      18 - 2 + 1 = 17 : by norm_num
  }
  sorry

end max_different_ages_l168_168702


namespace roots_are_negative_one_l168_168470

-- Problem statement in Lean 4:
theorem roots_are_negative_one {n : ℕ} {a : Fin n → ℝ} (h1 : ∀ i, (polynomial.roots (polynomial.X ^ n + n • polynomial.X ^ (n - 1) + polynomial.of_finsupp (finset.univ.sum (λ i, (a i) • polynomial.X ^ (n - i - 2))) * polynomial.C (a n))).count_real = n)
  (h2 : ∑ i, (polynomial.roots (polynomial.X ^ n + n • polynomial.X ^ (n - 1) + polynomial.of_finsupp (finset.univ.sum (λ i, (a i) • polynomial.X ^ (n - i - 2))) * polynomial.C (a n))) i ^ 16 = n) :
  ∀ i, (polynomial.roots (polynomial.X ^ n + n • polynomial.X ^ (n - 1) + polynomial.of_finsupp (finset.univ.sum (λ i, (a i) • polynomial.X ^ (n - i - 2))) * polynomial.C (a n))) i = -1 := 
sorry

end roots_are_negative_one_l168_168470


namespace part_I_part_II_l168_168999

variables {a b c : ℝ}
variables {A B C : ℝ}

-- Part I
theorem part_I (h1 : A = 2 * C)
  (h2 : A > 0 ∧ A < π / 2) 
  (h3 : B > 0 ∧ B < π / 2) 
  (h4 : C > 0 ∧ C < π / 2) :
  sqrt 2 < a / c ∧ a / c < sqrt 3 :=
sorry

-- Part II
theorem part_II (h1 : A = 2 * C) 
  (h2 : b = 1) 
  (h3 : c = 3) : 
  let cos_C := sqrt 3 / 3 in
  let sin_C := sqrt 6 / 3 in
  let a := 6 * cos_C in
  let area := 1/2 * a * b * sin_C in
  area = sqrt 2 :=
sorry

end part_I_part_II_l168_168999


namespace quadratic_other_x_intercept_l168_168158

theorem quadratic_other_x_intercept(
  (a b c : ℝ)
  (vertex : (5, -3))
  (x_intercept1 : (0, 0))
) : ∃ x : ℝ, x ≠ 0 ∧ x * (a * x + b) + c = 0 ∧ x = 10 := by
  sorry

end quadratic_other_x_intercept_l168_168158


namespace maximum_possible_product_l168_168671

-- Define the initial number and the digits used
def initial_number : ℕ := 2015
def digits : list ℕ := [2, 0, 1, 5]

-- Define a predicate for non-zero start in a number
def non_zero_start (n : ℕ) : Prop := n / 10^(nat.log10 n) ≠ 0

-- Define the maximum product obtained under given conditions
def maximum_product : ℕ :=
  list.maximum (list.map 
    (λ nums, nums.head * nums.tail.head) 
    [ [52, 10], [51, 20], [21, 50] ])

-- These two conditions can actually be inferred from the problem description.
-- However, we can add them for completeness.
axiom cond1 : ∀ n ∈ digits, non_zero_start n
axiom cond2 : ∀ n ∈ digits, (n ≤ initial_number)

theorem maximum_possible_product : maximum_product = 1050 :=
by
  sorry

end maximum_possible_product_l168_168671


namespace geometric_prog_real_roots_a_unique_value_l168_168093

theorem geometric_prog_real_roots_a_unique_value (a : ℝ) :
  (∃ x q : ℝ, x ≠ 0 ∧ q ≠ 1 ∧ 
  {x, q*x, q^2*x, q^3*x}.card = 4 ∧ 
  ∀ y : ℝ, y ∈ {x, q*x, q^2*x, q^3*x} → 
  (16 * y^4 - a * y^3 + (2 * a + 17) * y^2 - a * y + 16 = 0)) ↔ a = 170 :=
sorry

end geometric_prog_real_roots_a_unique_value_l168_168093


namespace average_people_per_hour_is_31_l168_168242

noncomputable def average_people_per_hour (total_people moving_to_texas: ℕ) (days: ℕ) (hours_per_day: ℕ) : ℕ :=
  Nat.round ((total_people:m ℝ / (days * hours_per_day) : ℝ))

theorem average_people_per_hour_is_31 :
  average_people_per_hour 3000 4 24 = 31 :=
by
  sorry

end average_people_per_hour_is_31_l168_168242


namespace range_of_m_l168_168203

def p (m : ℝ) : Prop := ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0)
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1 ∨ ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0))
  ∧ (¬ (∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0) → ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1)) ↔
  (-1 ≤ m ∧ m ≤ 0) ∨ (1 < m ∧ m < 2) :=
  sorry

end range_of_m_l168_168203


namespace find_d_and_a11_l168_168618

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_d_and_a11 (a : ℕ → ℤ) (d : ℤ) :
  arithmetic_sequence a d →
  a 5 = 6 →
  a 8 = 15 →
  d = 3 ∧ a 11 = 24 :=
by
  intros h_seq h_a5 h_a8
  sorry

end find_d_and_a11_l168_168618


namespace bugs_meeting_distance_l168_168752

/-- Triangle PQR has side lengths PQ = 6, QR = 8, and PR = 9. Two bugs start simultaneously 
from P and crawl along the perimeter of the triangle in opposite directions at the same speed.
They meet at point S. Prove that QS = 5.5. -/
theorem bugs_meeting_distance :
  ∀ (P Q R S : Type) (PQ QR PR : ℝ),
    PQ = 6 → QR = 8 → PR = 9 →
    (∀ t : ℝ, 0 ≤ t / 23.0 ≤ 1 → 
      (∃ a b : ℝ, a + b = t ∧ a ≤ PQ + PR ∧ b ≤ PR + QR 
                  → (Q - S).dist Q R = 5.5)) :=
begin
  intros P Q R S PQ QR PR hPQ hQR hPR t ht,
  use [6, 17],
  sorry
end

end bugs_meeting_distance_l168_168752


namespace product_of_solutions_l168_168133

theorem product_of_solutions : 
  let S := {x : ℝ | abs x = 3 * (abs x - 2)} in
  ∀ x y ∈ S, x ≠ y → (x * y) = -9 :=
by
  introv hx hy hxy
  have hS : S = {3, -3} := sorry -- Showing that S is exactly the set {3, -3}
  have hx' : x = 3 ∨ x = -3 := by rwa [hS] at hx
  have hy' : y = 3 ∨ y = -3 := by rwa [hS] at hy
  cases hx'; cases hy'; contradiction <|> norm_num

end product_of_solutions_l168_168133


namespace equation_of_line_AB_l168_168550

open Real

theorem equation_of_line_AB
  (A B : Point)
  (circle_eq : ∀ (P : Point), (P = A ∨ P = B) → (P.x ^ 2 + P.y ^ 2 = 4))
  (midpoint_eq : (A.x + B.x) / 2 = 1 ∧ (A.y + B.y) / 2 = 1) :
  ∃ (k b : ℝ), (k = -1) ∧ (b = 2) ∧ ∀ (P : Point), (P = A ∨ P = B) → (P.y = k * P.x + b) := sorry

end equation_of_line_AB_l168_168550


namespace find_x_plus_inv_x_l168_168544

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + x⁻³ = 110) : x + x⁻¹ = 5 :=
sorry

end find_x_plus_inv_x_l168_168544


namespace darry_full_ladder_climbs_l168_168091

-- Definitions and conditions
def full_ladder_steps : ℕ := 11
def smaller_ladder_steps : ℕ := 6
def smaller_ladder_climbs : ℕ := 7
def total_steps_climbed_today : ℕ := 152

-- Question: How many times did Darry climb his full ladder?
theorem darry_full_ladder_climbs (x : ℕ) 
  (H : 11 * x + smaller_ladder_steps * 7 = total_steps_climbed_today) : 
  x = 10 := by
  -- proof steps omitted, so we write
  sorry

end darry_full_ladder_climbs_l168_168091


namespace equivalent_fraction_div_calc_l168_168343

theorem equivalent_fraction_div_calc : 
  ∀ (a b c : ℕ), a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 ∧ 
  (3 * a + 2 = 4 * b + 3) ∧ (3 * a + 2 = 5 * c + 3) →
  (2 * a + b) / c = 4.75 :=
by
  intro a b c h,
  sorry

end equivalent_fraction_div_calc_l168_168343


namespace number_of_people_l168_168706

def avg_weight_increase : ℝ := 2.5
def old_person_weight : ℝ := 45
def new_person_weight : ℝ := 65

theorem number_of_people (n : ℕ) 
  (h1 : avg_weight_increase = 2.5) 
  (h2 : old_person_weight = 45) 
  (h3 : new_person_weight = 65) :
  n = 8 :=
  sorry

end number_of_people_l168_168706


namespace number_divisible_l168_168784

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168784


namespace angle_turned_by_hour_hand_l168_168700

theorem angle_turned_by_hour_hand (rotation_degrees_per_hour : ℝ) (total_degrees_per_rotation : ℝ) :
  rotation_degrees_per_hour * 1 = -30 :=
by
  have rotation_degrees_per_hour := - total_degrees_per_rotation / 12
  have total_degrees_per_rotation := 360
  sorry

end angle_turned_by_hour_hand_l168_168700


namespace calorie_allowance_correct_l168_168603

-- Definitions based on the problem's conditions
def daily_calorie_allowance : ℕ := 2000
def weekly_calorie_allowance : ℕ := 10500
def days_in_week : ℕ := 7

-- The statement to be proven
theorem calorie_allowance_correct :
  daily_calorie_allowance * days_in_week = weekly_calorie_allowance :=
by
  sorry

end calorie_allowance_correct_l168_168603


namespace geometric_series_common_ratio_l168_168445

theorem geometric_series_common_ratio (a S r : ℝ) (h1 : a = 512) (h2 : S = 3072) 
(h3 : S = a / (1 - r)) : r = 5/6 := 
sorry

end geometric_series_common_ratio_l168_168445


namespace projectiles_meet_in_72_minutes_l168_168886

theorem projectiles_meet_in_72_minutes :
  ∀ (distance speed1 speed2 : ℝ),
    distance = 1182 ∧ speed1 = 460 ∧ speed2 = 525 →
    (distance / (speed1 + speed2)) * 60 = 72 :=
by
  intros distance speed1 speed2 h
  cases h with h_dist h_speeds
  cases h_speeds with h_speed1 h_speed2
  rw [h_dist, h_speed1, h_speed2]
  sorry

end projectiles_meet_in_72_minutes_l168_168886


namespace domain_f_l168_168077

noncomputable def domain_of_function (f : ℝ → ℝ) := {x : ℝ | ∃ y : ℝ, f y = x}

def f (x : ℝ) : ℝ := real.sqrt (5 - real.sqrt (9 - real.sqrt x))

theorem domain_f :
  domain_of_function f = set.Icc 0 81 := by
sorry

end domain_f_l168_168077


namespace man_owns_fraction_of_business_l168_168915

theorem man_owns_fraction_of_business (x : ℝ) (h1 : 3/4 * x * 150000 = 75000) 
    (h2 : 150000 ≠ 0) : x = 2/3 :=
by
  have h : 3/4 * x = 75000 / 150000 := by sorry
  have h' : 3/4 * x = 1/2 := by sorry
  have x_value : x = (1/2) * (4/3) := by sorry
  show x = 2/3 from by sorry

end man_owns_fraction_of_business_l168_168915


namespace first_quadrant_solution_l168_168267

theorem first_quadrant_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 2 ∧ c * x + y = 3 ∧ 0 < x ∧ 0 < y) ↔ -1 < c ∧ c < 3 / 2 :=
by
  sorry

end first_quadrant_solution_l168_168267


namespace prob_heart_and_club_not_king_l168_168753

-- Definitions for the deck, cards, and dealing process
def deck : finset (Σ x : ℕ, x < 52) := sorry
def is_heart (card : Σ x : ℕ, x < 52) : Prop := sorry
def is_club (card : Σ x : ℕ, x < 52) : Prop := sorry
def is_king (card : Σ x : ℕ, x < 52) : Prop := sorry

-- The probability of an event
def probability {α : Type*} (s : finset α) (p : α → Prop) [decidable_pred p] : ℝ := 
  (s.filter p).card / s.card

-- The event of dealing two cards with the given conditions
def first_is_heart (cards : list (Σ x : ℕ, x < 52)) : Prop := is_heart (cards.head)
def second_is_club_not_king (cards : list (Σ x : ℕ, x < 52)) : Prop := 
  ¬ (is_king (cards.nth 1).get_or_else (cards.head)) ∧ is_club (cards.nth 1).get_or_else (cards.head)

-- The probability of the specific chained event 
def event_prob := 
  probability deck (λ card1, is_heart card1) * 
  probability (deck.erase (deck.filter is_heart).choose sorry) (λ card2, is_club card2 ∧ ¬ is_king card2)

theorem prob_heart_and_club_not_king : event_prob = 1 / 17 :=
sorry

end prob_heart_and_club_not_king_l168_168753


namespace smallest_digits_to_append_l168_168780

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168780


namespace ibrahim_lacks_129_euros_l168_168585

def mp3_player_cost := 135
def cd_cost := 25
def headphones_cost := 50
def carrying_case_cost := 30
def ibrahim_savings := 55
def father_contribution := 20
def discount_threshold := 150
def discount_rate := 0.15

def total_items_cost := mp3_player_cost + cd_cost + headphones_cost + carrying_case_cost
def qualifies_for_discount := total_items_cost > discount_threshold
def discount_amount := discount_rate * total_items_cost
def total_cost_after_discount := total_items_cost - (if qualifies_for_discount then discount_amount else 0)
def total_money := ibrahim_savings + father_contribution
def money_lacking := total_cost_after_discount - total_money

theorem ibrahim_lacks_129_euros :
  money_lacking = 129 := sorry

end ibrahim_lacks_129_euros_l168_168585


namespace probability_of_above_parabola_is_zero_l168_168299

theorem probability_of_above_parabola_is_zero :
  let S := {a | a ∈ {2, 4, 6, 8}} × {b | b ∈ {2, 4, 6, 8}} in
  let valid_points := {p : ℕ × ℕ | p ∈ S ∧ (∀ (x : ℕ), (p.2 : ℤ) > (p.1 : ℤ) * (x : ℤ)^2 + (p.1 : ℤ) * (x : ℤ))} in
  fintype.card valid_points = 0 :=
by
  sorry

end probability_of_above_parabola_is_zero_l168_168299


namespace calculate_xy_l168_168429

theorem calculate_xy :
  ∃ (x y : ℕ), 
  let xy := 10 * x + y in
  (0 ≤ x ∧ x ≤ 9) ∧ (0 ≤ y ∧ y ≤ 9) ∧ 
  45 * (2 + xy / 99) - 45 * (2 + xy / 100) = 1.35 ∧ 
  xy = 30 :=
begin
  sorry
end

end calculate_xy_l168_168429


namespace distance_between_axes_endpoints_l168_168084

theorem distance_between_axes_endpoints :
  ∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 → ∃ (C D : ℝ × ℝ), 
    (C = (-2, 2) ∨ C = (-2, -2)) ∧ 
    (D = (-6, 0) ∨ D = (2, 0)) ∧
    dist C D = 2 * Real.sqrt 5 :=
begin
  sorry
end

end distance_between_axes_endpoints_l168_168084


namespace product_equals_permutation_l168_168395

theorem product_equals_permutation :
  (∏ i in finset.Icc 89 100, i) = (100.perm 12) :=
by
  sorry

end product_equals_permutation_l168_168395


namespace circumscribed_triangle_cross_ratio_l168_168102

theorem circumscribed_triangle_cross_ratio
  (A B C P Q R B' C' : Point)
  (circumcircle : Circle)
  (H1 : is_circumscribed_triangle circumcircle A B C)
  (H2 : is_tangent_circle_at circumcircle B C P)
  (H3 : lies_on_line P C' B' AC)
  (H4 : lies_on_line P B' C' AB)
  (H5 : intersects_circle P Q R circumcircle) :
  ratio (dist C' Q) (dist B' Q) = ratio (dist C' R) (-dist B' R) :=
  sorry

end circumscribed_triangle_cross_ratio_l168_168102


namespace exists_polynomial_positive_powers_l168_168101

/- A definition for a polynomial with real coefficients that has at least one negative coefficient. -/
noncomputable def polynomial_with_negative_coeff : Π {R : Type*} [CommRing R], Polynomial R :=
  Polynomial.mk (λ n, if n = 2 then -1 else if n = 1 ∨ n = 3 ∨ n = 4 then 10 else if n = 0 then 10 else 0)

/- Proof statement for the existence of a polynomial with the required properties -/
theorem exists_polynomial_positive_powers :
  ∃ (p : Polynomial ℝ), 
    (∃ (n : ℕ), Polynomial.coeff p n < 0) ∧ 
    ∀ (n : ℕ), n > 1 → ∀ (m : ℕ), Polynomial.coeff (p^n) m > 0 :=
sorry

end exists_polynomial_positive_powers_l168_168101


namespace solve_system_l168_168303

noncomputable def quadratic_solutions {x y: ℂ} (h1: x^2 + y^2 = x * y) (h2: x + y = x * y) : Prop :=
  (x = 0 ∧ y = 0) ∨
  (x = (3 + complex.I * complex.sqrt 3) / 2 ∧ y = (3 - complex.I * complex.sqrt 3) / 2) ∨
  (x = (3 - complex.I * complex.sqrt 3) / 2 ∧ y = (3 + complex.I * complex.sqrt 3) / 2)

theorem solve_system : ∃ x y : ℂ, x^2 + y^2 = x * y ∧ x + y = x * y ∧ quadratic_solutions x y :=
by
  sorry

end solve_system_l168_168303


namespace grasshoppers_total_l168_168254

theorem grasshoppers_total (grasshoppers_on_plant : ℕ) (dozens_of_baby_grasshoppers : ℕ) (dozen_value : ℕ) : 
  grasshoppers_on_plant = 7 → dozens_of_baby_grasshoppers = 2 → dozen_value = 12 → 
  grasshoppers_on_plant + dozens_of_baby_grasshoppers * dozen_value = 31 :=
by
  intros h1 h2 h3
  sorry

end grasshoppers_total_l168_168254


namespace midpoint_of_YZ_l168_168754

open EuclideanGeometry

variables {ℝ : Type} [LinearOrderedField ℝ]
variables (C1 C2 : Circle ℝ) (P Q A B X Y Z : Point ℝ)
variables (hPQ : P ∈ C1 ∧ P ∈ C2 ∧ Q ∈ C1 ∧ Q ∈ C2)
variables (hPA: P ≠ A ∧ line_through P A ∩ C1 = {P, A})
variables (hPB: P ≠ B ∧ line_through P B ∩ C2 = {P, B})
variables (hMidpointAB : midpoint A B X)
variables (hQX: (Q ≠ X) ∧ line_through Q X ∩ C1 = {Q, Y})
variables (hQZ: (Q ≠ Z) ∧ line_through Q X ∩ C2 = {Q, Z})

theorem midpoint_of_YZ :
  midpoint Y Z X :=
sorry

end midpoint_of_YZ_l168_168754


namespace volume_displaced_cube_displaced_volume_l168_168463

def radius_cylinder := 3
def height_cylinder := 12
def side_length_cube := 6
def submerged_height_cube := side_length_cube / 2
def inscribed_square_side := radius_cylinder * Real.sqrt 2
def area_inscribed_square := inscribed_square_side ^ 2
def submerged_volume := area_inscribed_square * submerged_height_cube

theorem volume_displaced (r h s : ℝ) (h_r : r = 3) (h_h : h = 12) (h_s : s = 6) : 
  submerged_volume = 54 :=
by 
  rw [h_r, h_h, h_s]
  sorry

theorem cube_displaced_volume (v : ℝ) (h_v : v = 54) : v^3 = 157464 := 
by 
  rw [h_v]
  sorry

end volume_displaced_cube_displaced_volume_l168_168463


namespace shift_graph_to_right_l168_168344

theorem shift_graph_to_right (x : ℝ) : 
  4 * Real.cos (2 * x + π / 4) = 4 * Real.cos (2 * (x - π / 8) + π / 4) :=
by 
  -- sketch of the intended proof without actual steps for clarity
  sorry

end shift_graph_to_right_l168_168344


namespace percentage_of_360_is_120_l168_168371

theorem percentage_of_360_is_120 (part whole : ℝ) (h1 : part = 120) (h2 : whole = 360) : 
  ((part / whole) * 100 = 33.33) :=
by
  sorry

end percentage_of_360_is_120_l168_168371


namespace cartesian_eqn_of_polar_max_of_3x_plus_4y_l168_168187

-- Given polar equation of the curve C
def polar_eqn (ρ θ : ℝ) := ρ^2 = 36 / (4 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2)

-- Convert to Cartesian coordinates and prove equivalent Cartesian equation
theorem cartesian_eqn_of_polar :
  (∀ ρ θ, polar_eqn ρ θ → (∃ x y, (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ (x^2 / 9 + y^2 / 4 = 1))) :=
by
  intros ρ θ hρ
  use [ρ * Real.cos θ, ρ * Real.sin θ]
  split
  { exact ⟨rfl, rfl⟩ }
  sorry

-- Prove the maximum value of 3x + 4y for points (x, y) on the curve
theorem max_of_3x_plus_4y :
  (∀ x y, (x^2 / 9 + y^2 / 4 = 1) → ∃ θ, (x = 3 * Real.cos θ ∧ y = 2 * Real.sin θ) → (3 * x + 4 * y ≤ Real.sqrt 145)) :=
by
  intros x y hxy
  use [Real.atan2 y x]
  intro hθ
  rw [hθ.left, hθ.right]
  have : 9 * Real.cos (Real.atan2 y x) + 8 * Real.sin (Real.atan2 y x) ≤ Real.sqrt(145) * 1,
  { sorry }
  have : ∀ t, t ≤ t + 0, from λ t, by linarith,
  exact this _

end cartesian_eqn_of_polar_max_of_3x_plus_4y_l168_168187


namespace cube_side_length_l168_168497

theorem cube_side_length (base_area : ℝ) (height : ℝ) (volume_diff : ℝ) (s : ℝ) 
  (h_base_area : base_area = 10) 
  (h_height : height = 73)
  (h_volume_diff : volume_diff = 1) 
  (volume_cuboid : ℝ) 
  (h_volume_cuboid : volume_cuboid = base_area * height) 
  (volume_cube : ℝ) 
  (h_volume_cube : volume_cube = volume_cuboid - volume_diff) 
  (h_cube_side : s^3 = volume_cube) :
  s = 9 := 
begin
  sorry
end

end cube_side_length_l168_168497


namespace circle_center_image_l168_168458

def C_initial : (ℕ × ℕ) := (3, -4)
def reflect_x (p : ℕ × ℕ) : ℕ × ℕ := (p.1, -p.2)
def translate_right (p : ℕ × ℕ) (d : ℕ) : ℕ × ℕ := (p.1 + d, p.2)

theorem circle_center_image 
  (initial : ℕ × ℕ)
  (rfl_x : (ℕ × ℕ) → ℕ × ℕ)
  (tr_right : (ℕ × ℕ) → ℕ → ℕ × ℕ) :
  (translate_right (reflect_x initial) 5) = (8, 4) :=
by
  let initial := C_initial
  let rfl_x := reflect_x
  let tr_right := translate_right
  sorry

end circle_center_image_l168_168458


namespace integer_solutions_abs_sum_lt_n_l168_168210

theorem integer_solutions_abs_sum_lt_n (n : ℕ) : 
  ∃ (s : ℕ), s = 2 * n^2 - 2 * n + 1 ∧
  (∀ (x y : ℤ), |x| + |y| < n → ((x, y) ∈ set_of_accepted_solutions)) :=
by
  sorry

end integer_solutions_abs_sum_lt_n_l168_168210


namespace largest_divisor_of_square_l168_168387

theorem largest_divisor_of_square (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n ^ 2) : 12 ∣ n := 
sorry

end largest_divisor_of_square_l168_168387


namespace woman_l168_168933

-- Define the variables and given conditions
variables (W S X : ℕ)
axiom s_eq : S = 27
axiom sum_eq : W + S = 84
axiom w_eq : W = 2 * S + X

theorem woman's_age_more_years : X = 3 :=
by
  -- Proof goes here
  sorry

end woman_l168_168933


namespace find_marks_in_mathematics_l168_168955

theorem find_marks_in_mathematics : 
  ∀ (E P C B A M : ℕ), 
    E = 36 → P = 42 → C = 57 → B = 55 → A = 45 → M = (A * 5) - (E + P + C + B) → 
    M = 35 :=
by {
  intros E P C B A M hE hP hC hB hA hM,
  sorry
}


end find_marks_in_mathematics_l168_168955


namespace quadratic_function_solution_l168_168271

theorem quadratic_function_solution :
  (∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ f'(x) = 2 * x + 2) ∧ (b^2 - 4 * a * c = 0) ∧ (f = λ x, a * x^2 + b * x + c)) → (f = λ x, x^2 + 2 * x + 1) :=
by
  sorry

end quadratic_function_solution_l168_168271


namespace value_of_f_at_2_and_neg_log2_3_l168_168563

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 2 else 2^(-x)

theorem value_of_f_at_2_and_neg_log2_3 :
  f 2 * f (-Real.log 3 / Real.log 2) = 3 := by
  sorry

end value_of_f_at_2_and_neg_log2_3_l168_168563


namespace find_x_l168_168142

theorem find_x (x : ℝ) (h : 8^(x-1) / 2^(x-1) = 64^(2*x)) : x = -1/5 := by
  sorry

end find_x_l168_168142


namespace sum_of_roots_of_tan_quadratic_l168_168145

theorem sum_of_roots_of_tan_quadratic :
  (∑ x in {x ∈ Ico 0 (2 * Real.pi) | tan x * tan x - 9 * tan x + 1 = 0}, x) = 3 * Real.pi :=
sorry

end sum_of_roots_of_tan_quadratic_l168_168145


namespace frog_jumps_final_position_l168_168412

noncomputable def frog_jump_probability : ℝ :=
  let p := 1 / 10 in
  p

theorem frog_jumps_final_position :
  ∀ {f : ℕ → ℝ} {d : ℝ},
    (∀ n, f 1 = 1 ∧ f 2 = 2 ∧ f 3 = 3 ∧ f 4 = 4) →
    (∑ n in {1, 2, 3, 4}, (direction n * f n) = d) →
    d ≤ 2 →
    frog_jump_probability = 1 / 10 :=
sorry

end frog_jumps_final_position_l168_168412


namespace sphere_segment_area_l168_168755

theorem sphere_segment_area (R h : ℝ) (h_pos : 0 < h) (R_pos : 0 < R)
  (planes_do_not_intersect : h < 2 * R) : 
  surface_area_segment R h = 2 * π * R * h := 
sorry

end sphere_segment_area_l168_168755


namespace left_handed_like_jazz_l168_168740

theorem left_handed_like_jazz (total_people left_handed like_jazz right_handed_dislike_jazz : ℕ)
    (h1 : total_people = 30)
    (h2 : left_handed = 12)
    (h3 : like_jazz = 20)
    (h4 : right_handed_dislike_jazz = 3)
    (h5 : ∀ p, p = total_people - left_handed ∧ p = total_people - (left_handed + right_handed_dislike_jazz)) :
    ∃ x, x = 5 := by
  sorry

end left_handed_like_jazz_l168_168740


namespace average_score_correct_l168_168698

-- Define the conditions
def simplified_scores : List Int := [10, -5, 0, 8, -3]
def base_score : Int := 90

-- Translate simplified score to actual score
def actual_score (s : Int) : Int :=
  base_score + s

-- Calculate the average of the actual scores
def average_score : Int :=
  (simplified_scores.map actual_score).sum / simplified_scores.length

-- The proof statement
theorem average_score_correct : average_score = 92 := 
by 
  -- Steps to compute the average score
  -- sorry is used since the proof steps are not required
  sorry

end average_score_correct_l168_168698


namespace relationship_between_M_and_N_l168_168523

variable (x y : ℝ)

theorem relationship_between_M_and_N (h1 : x ≠ 3) (h2 : y ≠ -2)
  (M : ℝ) (hm : M = x^2 + y^2 - 6 * x + 4 * y)
  (N : ℝ) (hn : N = -13) : M > N :=
by
  sorry

end relationship_between_M_and_N_l168_168523


namespace two_trucks_carry_2_tons_l168_168927

theorem two_trucks_carry_2_tons :
  ∀ (truck_capacity : ℕ), truck_capacity = 999 →
  (truck_capacity * 2) / 1000 = 2 :=
by
  intros truck_capacity h_capacity
  rw [h_capacity]
  exact sorry

end two_trucks_carry_2_tons_l168_168927


namespace smallest_append_digits_l168_168839

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168839


namespace rate_of_current_is_5_l168_168329

theorem rate_of_current_is_5 
  (speed_still_water : ℕ)
  (distance_travelled : ℕ)
  (time_travelled : ℚ) 
  (effective_speed_with_current : ℚ) : 
  speed_still_water = 20 ∧ distance_travelled = 5 ∧ time_travelled = 1/5 ∧ 
  effective_speed_with_current = (speed_still_water + 5) →
  effective_speed_with_current * time_travelled = distance_travelled :=
by
  sorry

end rate_of_current_is_5_l168_168329


namespace min_y_value_l168_168971

-- Define the function y
def y (x : ℝ) : ℝ := 3 * x ^ 2 + 6 / (x ^ 2 + 1)

-- Define the minimum value to be proved
def min_val_y : ℝ := 6 * Real.sqrt 2 - 3

-- Theorem statement to prove
theorem min_y_value : ∃ x : ℝ, y x = min_val_y := sorry

end min_y_value_l168_168971


namespace board_painting_max_1x2_tiles_l168_168081

-- Define the properties of the 9x9 board
def board_size : ℕ := 9
def odd_rows : List ℕ := [1, 3, 5, 7, 9]
def even_rows : List ℕ := [2, 4, 6, 8]
def row_length : ℕ := 9
def blue_tiles_per_odd_row : ℕ := 5
def white_tiles_per_odd_row : ℕ := 4
def gray_tiles_per_even_row : ℕ := 5
def red_tiles_per_even_row : ℕ := 4

theorem board_painting :
  let total_blue := (odd_rows.length * blue_tiles_per_odd_row)
  let total_white := (odd_rows.length * white_tiles_per_odd_row)
  let total_gray := (even_rows.length * gray_tiles_per_even_row)
  let total_red := (even_rows.length * red_tiles_per_even_row)
  total_blue = 25 ∧ total_white = 20 ∧ total_gray = 20 ∧ total_red = 16 :=
by
  let total_blue := (odd_rows.length * blue_tiles_per_odd_row)
  let total_white := (odd_rows.length * white_tiles_per_odd_row)
  let total_gray := (even_rows.length * gray_tiles_per_even_row)
  let total_red := (even_rows.length * red_tiles_per_even_row)
  show total_blue = 25 ∧ total_white = 20 ∧ total_gray = 20 ∧ total_red = 16
  sorry

theorem max_1x2_tiles :
  ∃ max_tiles, max_tiles = 16 :=
by
  let num_red_tiles := (even_rows.length * red_tiles_per_even_row)
  show ∃ max_tiles, max_tiles = 16
  from Exists.intro num_red_tiles (by refl)

end board_painting_max_1x2_tiles_l168_168081


namespace sum_of_squares_of_medians_l168_168361

theorem sum_of_squares_of_medians (a b c : ℝ) (h1 : a = 13) (h2 : b = 13) (h3 : c = 10) :
  let m₁ := (1/2 : ℝ) * math.sqrt(2 * b^2 + 2 * c^2 - a^2);
      m₂ := (1/2 : ℝ) * math.sqrt(2 * a^2 + 2 * c^2 - b^2);
      m₃ := (1/2 : ℝ) * math.sqrt(2 * a^2 + 2 * b^2 - c^2)
  in m₁^2 + m₂^2 + m₃^2 = 278.5 := by
  -- Proof skeleton here
  sorry

end sum_of_squares_of_medians_l168_168361


namespace tuple_bound_l168_168390

noncomputable def max_cardinality_of_tuples {n : ℕ} (x : fin n → ℝ) (a : ℝ) : Prop :=
  (∀ i : fin n, abs (x i) ≥ 1) →
  ∃ (S : set (fin n → ℤ)),
    (∀ v, v ∈ S ↔ (∀ i, v i = 1 ∨ v i = -1) ∧ (a ≤ ∑ i, (v i) * (x i)) ∧ (∑ i, (v i) * (x i) < a + 2)) ∧
    S.card ≤ nat.choose n (nat.ceil (n / 2))

theorem tuple_bound (n : ℕ) (x : fin n → ℝ) (a : ℝ) :
  ∀ i : fin n, abs (x i) ≥ 1 → ∃ S : set (fin n → ℤ), (∀ v, v ∈ S ↔ (∀ i, v i = 1 ∨ v i = -1) ∧ (a ≤ ∑ i, v i * x i) ∧ (∑ i, v i * x i < a + 2)) ∧
  S.card ≤ (nat.choose n (nat.ceil (n / 2))) :=
sorry

end tuple_bound_l168_168390


namespace ratio_Polly_Willy_l168_168388

theorem ratio_Polly_Willy (P S W : ℝ) (h1 : P / S = 4 / 5) (h2 : S / W = 5 / 2) :
  P / W = 2 :=
by sorry

end ratio_Polly_Willy_l168_168388


namespace distance_between_axes_endpoints_l168_168085

theorem distance_between_axes_endpoints :
  ∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 → ∃ (C D : ℝ × ℝ), 
    (C = (-2, 2) ∨ C = (-2, -2)) ∧ 
    (D = (-6, 0) ∨ D = (2, 0)) ∧
    dist C D = 2 * Real.sqrt 5 :=
begin
  sorry
end

end distance_between_axes_endpoints_l168_168085


namespace percentage_of_students_passed_l168_168236

def total_students : ℕ := 740
def failed_students : ℕ := 481
def passed_students : ℕ := total_students - failed_students
def pass_percentage : ℚ := (passed_students / total_students) * 100

theorem percentage_of_students_passed : pass_percentage = 35 := by
  sorry

end percentage_of_students_passed_l168_168236


namespace range_f_range_g_l168_168141

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem range_f : set.range f = set.Ici (-1) := 
sorry

-- Definition of g and range theorem
noncomputable def g (x : ℝ) (hx : 1 ≤ x ∧ x < 3) : ℝ := 1 / x

theorem range_g (h : 1 ≤ x ∧ x < 3) : set.range (λ x, g x h) = set.Ioc (1 / 3) 1 :=
sorry

end range_f_range_g_l168_168141


namespace pi_expression_value_l168_168958

theorem pi_expression_value :
  let π := 4 * Real.sin (52 * Real.pi / 180) in
  (1 - 2 * Real.cos(7 * Real.pi / 180)^2) / (π * Real.sqrt(16 - π^2)) = -1 / 8 := 
by
  sorry

end pi_expression_value_l168_168958


namespace smallest_digits_to_append_l168_168822

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168822


namespace sum_distinct_x_f_ff_f_ff_f_x_1_l168_168569

def f (x : ℝ) : ℝ := x^2 / 4 - x - 1

theorem sum_distinct_x_f_ff_f_ff_f_x_1 :
  let X := {x : ℝ | f (f (f x)) = 1}
  in ∑ x in finset.filter (λ x, x ∈ X) (finset.Icc (-10 : ℝ) (10 : ℝ)) = 10 :=
by
  sorry

end sum_distinct_x_f_ff_f_ff_f_x_1_l168_168569


namespace bird_families_difference_l168_168878

-- Define the conditions
def bird_families_to_africa : ℕ := 47
def bird_families_to_asia : ℕ := 94

-- The proof statement
theorem bird_families_difference : (bird_families_to_asia - bird_families_to_africa = 47) :=
by
  sorry

end bird_families_difference_l168_168878


namespace solve_for_r_l168_168092

def E (a b c : ℝ) : ℝ := a * b^c

theorem solve_for_r : ∃ r : ℝ, r > 0 ∧ E r r 4 = 256 ∧ r = 2^(8/5) :=
by
  sorry

end solve_for_r_l168_168092


namespace smallest_digits_to_append_l168_168871

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168871


namespace slower_pipe_filling_time_l168_168020

-- Definitions based on conditions
def faster_pipe_rate (S : ℝ) : ℝ := 3 * S
def combined_rate (S : ℝ) : ℝ := (faster_pipe_rate S) + S

-- Statement of what needs to be proved 
theorem slower_pipe_filling_time :
  (∀ S : ℝ, combined_rate S * 40 = 1) →
  ∃ t : ℝ, t = 160 :=
by
  intro h
  sorry

end slower_pipe_filling_time_l168_168020


namespace smallest_number_of_digits_to_append_l168_168802

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168802


namespace min_value_expr_l168_168506

variable (a b c d : ℝ) 

noncomputable def expr := (a + b)^2 + (b - c)^2 + (c - d)^2 + (d - a)^2) / b^2

theorem min_value_expr 
  (h1 : b > c) (h2 : c > d) (h3 : d > a) (h4 : b ≠ 0): 
  ∃ m, m = 1 ∧ ∀ a b c d, expr a b c d ≥ m := 
sorry

end min_value_expr_l168_168506


namespace smallest_digits_to_append_l168_168829

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168829


namespace disk_area_l168_168421

theorem disk_area (r : ℝ) (h : r = 1) (A1 A2 : ℝ) 
  (h₁ : A1 = 3 * A2) (h₂ : A1 + A2 = 4 * π) : 
  disk_area = 3 * π / 4 :=
by
  -- The proof will be implemented here
  sorry

end disk_area_l168_168421


namespace connie_correct_answer_l168_168080

theorem connie_correct_answer 
  (x : ℝ) 
  (h1 : 2 * x = 80) 
  (correct_ans : ℝ := x / 3) :
  correct_ans = 40 / 3 :=
by
  sorry

end connie_correct_answer_l168_168080


namespace append_digits_divisible_by_all_less_than_10_l168_168793

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168793


namespace maximize_product_l168_168879

-- Define 22 as a constant natural number
def N : ℕ := 22

-- Define what it means for sums of distinct natural numbers to sum to 22
def is_valid_split (ns : List ℕ) : Prop :=
  (ns.sum = N) ∧ (∀ (x y : ℕ), x ∈ ns → y ∈ ns → x ≠ y → nat.gcd x y = 1)

-- State the main theorem
theorem maximize_product : ∃ (ns : List ℕ), is_valid_split ns ∧ ns.prod = 1008 :=
sorry  -- proof to be filled

end maximize_product_l168_168879


namespace polar_distance_diametrically_opposite_l168_168969

theorem polar_distance_diametrically_opposite (r1 r2 θ1 θ2 : ℝ) (hA : r1 = 5) (θA_eq : θ1 = 7 * π / 36) (hB : r2 = 12) (θB_eq : θ2 = 43 * π / 36) :
    θ2 - θ1 = π → (A = (r1, θ1) ∧ B = (r2, θ2)) → dist (r1, θ1) (r2, θ2) = sqrt (r1^2 + r2^2 - 2 * r1 * r2 * cos (π)) :=
begin
  sorry
end

end polar_distance_diametrically_opposite_l168_168969


namespace jialingRiver_increase_l168_168602

-- Definition of Jialing River water level notation
def JialingRiver :=
  { waterLevel : ℤ // waterLevel < 0 → "decrease by (-waterLevel) meters" = true ∧ waterLevel > 0 → "increase by (waterLevel) meters" = true }

-- Proof statement
theorem jialingRiver_increase :
  ∀ (waterLevel : JialingRiver), waterLevel = 8 → "increase by 8 meters" = true :=
by
  sorry

end jialingRiver_increase_l168_168602


namespace probability_two_female_one_male_l168_168672

-- Define basic conditions
def total_contestants : Nat := 7
def female_contestants : Nat := 4
def male_contestants : Nat := 3
def choose_count : Nat := 3

-- Calculate combinations (binomial coefficients)
def comb (n k : Nat) : Nat := Nat.choose n k

-- Define the probability calculation steps in Lean
def total_ways := comb total_contestants choose_count
def favorable_ways_female := comb female_contestants 2
def favorable_ways_male := comb male_contestants 1
def favorable_ways := favorable_ways_female * favorable_ways_male

theorem probability_two_female_one_male :
  (favorable_ways : ℚ) / (total_ways : ℚ) = 18 / 35 := by
  sorry

end probability_two_female_one_male_l168_168672


namespace sum_series_eq_two_l168_168079

noncomputable def series_term (n : ℕ) : ℚ := (3 * n - 2) / (n * (n + 1) * (n + 2))

theorem sum_series_eq_two :
  ∑' n : ℕ, series_term (n + 1) = 2 :=
sorry

end sum_series_eq_two_l168_168079


namespace find_a_of_ellipse_foci_l168_168223

theorem find_a_of_ellipse_foci (a : ℝ) :
  (∀ x y : ℝ, a^2 * x^2 - (a / 2) * y^2 = 1) →
  (a^2 - (2 / a) = 4) →
  a = (1 - Real.sqrt 5) / 4 :=
by 
  intros h1 h2
  sorry

end find_a_of_ellipse_foci_l168_168223


namespace solution_set_of_inequality_l168_168557

noncomputable def problem_solution_set (b : ℝ) (x : ℝ) : set ℝ :=
{y | (y + b) / ((y - 6) * (y + 1)) > 0}

theorem solution_set_of_inequality :
  (∀ x : ℝ, x + b > 0 ↔ x > 2) →
  problem_solution_set (-2) x = {x | (-1 < x ∧ x < 2) ∨ 6 < x} :=
by
  intro h
  have hb_eq_neg2 : b = -2 := sorry
  subst hb_eq_neg2
  sorry

end solution_set_of_inequality_l168_168557


namespace inequality_solution_range_l168_168397

theorem inequality_solution_range (a : ℝ) :
  (∃ (x : ℝ), |x + 1| - |x - 2| < a^2 - 4 * a) → (a > 3 ∨ a < 1) :=
by
  sorry

end inequality_solution_range_l168_168397


namespace sum_of_binomial_coeffs_expansion_l168_168732

open Finset

-- Definition of binomial expansion
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of binomial expansion sum of coefficients
def binomial_expansion_sum (a b : ℕ) (n : ℕ) : ℕ :=
  ∑ k in range (n + 1), binomial_coeff n k * a^(n - k) * b^k

-- The theorem we need to prove
theorem sum_of_binomial_coeffs_expansion (a b : ℕ) (n : ℕ) :
  (a - b) ^ n = 0 → binomial_expansion_sum 1 1 n = 0 :=
by
  sorry

end sum_of_binomial_coeffs_expansion_l168_168732


namespace min_distance_AB_l168_168660

theorem min_distance_AB :
  let A ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, sqrt 3 * x + 1)} in
  let B ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, Real.log x)} in
  let min_dist := 1 + (1 / 4) * Real.log 3 in
  min_dist = (let x := sqrt 3 / 3 in
              let y := - (1 / 2) * Real.log 3 in
              let P := (x, y) in
              abs (sqrt 3 * x - y + 1) / sqrt (3 + 1)) :=
by sorry

end min_distance_AB_l168_168660


namespace smallest_append_digits_l168_168836

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168836


namespace roots_of_polynomial_l168_168118

def polynomial : Polynomial ℝ := X^3 + 2*X^2 - 5*X - 6

theorem roots_of_polynomial :
  (Polynomial.roots polynomial).toFinset = {-1, 2, -3} :=
by
  sorry

end roots_of_polynomial_l168_168118


namespace shortest_segment_halving_area_of_345_triangle_l168_168771

noncomputable def herons_formula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem shortest_segment_halving_area_of_345_triangle :
  ∃ (PQ : ℝ), PQ = 2 ∧ ∀ s1 s2 : ℝ, herons_formula 3 4 5 / 2 = herons_formula s1 s2 PQ / 2 :=
begin
  sorry
end

end shortest_segment_halving_area_of_345_triangle_l168_168771


namespace exam_score_probability_l168_168736

open ProbabilityTheory

noncomputable def num_people_scoring_at_least_139 
  (num_people : ℝ) (μ : ℝ) (σ : ℝ) (p_interval : ℝ) :=
  let probability_ge_139 := (1 - p_interval) / 2 in
  num_people * probability_ge_139

theorem exam_score_probability
  (num_people : ℝ)
  (μ σ : ℝ)
  (p_interval : ℝ)
  (h : p_interval = 0.997)
  (h_norm : ∀ X, X ~ Normal μ σ^2) :
  num_people_scoring_at_least_139 num_people μ σ p_interval = 15 :=
by
  dsimp [num_people_scoring_at_least_139]
  rw [h]
  norm_num
  apply mul_eq_mul_right_iff.mpr
  right
  norm_num
  sorry

end exam_score_probability_l168_168736


namespace problem_statement_l168_168678

variable (a b : ℝ)

-- Conditions
variable (h1 : a > 0) (h2 : b > 0) (h3 : ∃ x, x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a)))

-- The Lean theorem statement for the problem
theorem problem_statement : 
  ∀ x, (x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))) →
  (2 * a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := 
sorry


end problem_statement_l168_168678


namespace find_AB_l168_168628

-- Definitions based on the problem's statement
variables (ABC : Type) [linear_ordered_field ABC] (A B C : ABC)

-- Problem conditions
def angle_A_90 : Prop := angle A = 90
def BC_eq_10 : Prop := BC = 10
def tan_C_eq_3_cos_B : Prop := tan C = 3 * cos B

-- The goal is to prove AB = 20 * sqrt 2 / 3
theorem find_AB (h1 : angle_A_90) (h2 : BC_eq_10) (h3 : tan_C_eq_3_cos_B) : AB = 20 * sqrt 2 / 3 := 
by sorry

end find_AB_l168_168628


namespace correct_option_is_D_l168_168441

-- Definition of the expressions
def expr_A (m : ℝ) : ℝ := m^2 * m^5
def expr_B (m : ℝ) : ℝ := m^5 + m^5
def expr_C (m : ℝ) : ℝ := m^20 / m^2
def expr_D (m : ℝ) : ℝ := (m^2)^5

-- The statement of the theorem
theorem correct_option_is_D (m : ℝ) : expr_D m = m^10 ∧ expr_A m ≠ m^10 ∧ expr_B m ≠ m^10 ∧ expr_C m ≠ m^10 :=
by sorry

end correct_option_is_D_l168_168441


namespace distance_from_focus_to_line_line_parallel_to_x_axis_isosceles_right_triangle_PAB_l168_168551

noncomputable def hyperbola_eq (x y : ℝ) : Prop := (x ^ 2 / 2) - y ^ 2 = 1

noncomputable def line_eq (k x y : ℝ) : Prop := y = k * x + 1

noncomputable def dist_point_to_line (A B C x y : ℝ) : ℝ := 
  (abs (A * x + B * y + C) / real.sqrt (A ^ 2 + B ^ 2))

theorem distance_from_focus_to_line 
(F1_x F1_y : ℝ) (F2_x F2_y : ℝ) (k : ℝ) 
(h1 : F1_x = -real.sqrt 3) 
(h2 : F1_y = 0) 
(h3 : F2_x = real.sqrt 3) 
(h4 : F2_y = 0)
(h5 : line_eq k (-real.sqrt 3) 0) :
dist_point_to_line (k * real.sqrt 3) (-1) 1 (real.sqrt 3) 0 = real.sqrt 3 :=
sorry

theorem line_parallel_to_x_axis
(O_x O_y : ℝ) (C_x C_y D_x D_y : ℝ) (k : ℝ) 
(h1 : O_x = 0) (h2 : O_y = 0) 
(h3 : line_eq k C_x C_y) 
(h4 : line_eq k D_x D_y) 
(h5 : hyperbola_eq C_x C_y) 
(h6 : hyperbola_eq D_x D_y) :
let area_COD := real.abs ((C_x * D_y - C_y * D_x) / 2) in
  (area_COD = real.sqrt 2 → k = 0) :=
sorry

theorem isosceles_right_triangle_PAB
(A_x A_y B_x B_y P_x P_y k : ℝ)
(h1 : line_eq k A_x A_y) 
(h2 : line_eq k B_x B_y) 
(h3 : hyperbola_eq A_x A_y) 
(h4 : hyperbola_eq B_x B_y) 
(h5 : P_y = 0) 
(h6 : (P_x * P_x + A_y * A_y = B_y * B_y -> Δ_A_P_B))
.exists (k : ℝ) such that (∀ A B P : (P_x = -3 * real.sqrt 3) ∧ P_y = 0 := 
sorry

end distance_from_focus_to_line_line_parallel_to_x_axis_isosceles_right_triangle_PAB_l168_168551


namespace rational_solutions_k_values_l168_168094

theorem rational_solutions_k_values (k : ℕ) (h₁ : k > 0) 
    (h₂ : ∃ (m : ℤ), 900 - 4 * (k:ℤ)^2 = m^2) : k = 9 ∨ k = 15 := 
by
  sorry

end rational_solutions_k_values_l168_168094


namespace arctan_maclaurin_series_l168_168503

theorem arctan_maclaurin_series (x : ℝ) (h : abs x < 1) : 
  ∃ (s : ℕ → ℝ), 
  (∀ n, s n = (-1)^n * (x^(2*n+1)) / (2*n + 1)) ∧ 
  has_sum s (arctan x) :=
by sorry

end arctan_maclaurin_series_l168_168503


namespace smallest_number_is_C_l168_168745

-- Define the conditions
def A := 18 + 38
def B := A - 26
def C := B / 3

-- Proof statement: C is the smallest number among A, B, and C
theorem smallest_number_is_C : C = min A (min B C) :=
by
  sorry

end smallest_number_is_C_l168_168745


namespace value_of_a_l168_168215

theorem value_of_a (m : ℝ) (f : ℝ → ℝ) (h : f = fun x => (1/3)^x + m - 1/3) 
  (h_m : ∀ x, f x ≥ 0 ↔ m ≥ -2/3) : m ≥ -2/3 :=
by
  sorry

end value_of_a_l168_168215


namespace units_digit_of_sum_of_factorials_l168_168587

theorem units_digit_of_sum_of_factorials : 
  (3! + 4! + (∑ i in finset.range (101 - 4) + 5, i!)) % 10 = 0 :=
by
  sorry

end units_digit_of_sum_of_factorials_l168_168587


namespace find_n_and_P_l168_168893

noncomputable def P (x : ℝ) : ℝ := sorry

theorem find_n_and_P :
  (∃ (n : ℕ) (P : ℝ → ℝ), degree P = 2 * n ∧
  (∀ i : ℕ, i ≤ 2 * n → P (2 * i) = 0) ∧
  (∀ j : ℕ, j ≤ 2 * n - 1 → P (2 * j + 1) = 2) ∧
  P (2 * n + 1) = -6 ∧
  n = 1 ∧ P = (λ x, -2 * x^2 + 4 * x)) :=
begin
  sorry
end

end find_n_and_P_l168_168893


namespace charlie_max_success_ratio_l168_168614

-- Given:
-- Alpha scored 180 points out of 360 attempted on day one.
-- Alpha scored 120 points out of 240 attempted on day two.
-- Charlie did not attempt 360 points on the first day.
-- Charlie's success ratio on each day was less than Alpha’s.
-- Total points attempted by Charlie on both days are 600.
-- Alpha's two-day success ratio is 300/600 = 1/2.
-- Find the largest possible two-day success ratio that Charlie could have achieved.

theorem charlie_max_success_ratio:
  ∀ (x y z w : ℕ),
  0 < x ∧ 0 < z ∧ 0 < y ∧ 0 < w ∧
  y + w = 600 ∧
  (2 * x < y) ∧ (2 * z < w) ∧
  (x + z < 300) -> (299 / 600 = 299 / 600) :=
by
  sorry

end charlie_max_success_ratio_l168_168614


namespace regular_polygon_area_l168_168925

theorem regular_polygon_area (n : ℕ) (R : ℝ)
  (hR : R > 0)
  (hA : (1 / 2) * n * R^2 * real.sin (360 / n * real.pi / 180) = 3 * R^2) :
  n = 12 :=
sorry

end regular_polygon_area_l168_168925


namespace length_segment_PR_l168_168285

theorem length_segment_PR (P Q R : Point) (O : Point) (r : ℝ) 
  (h1 : distance O P = 7) (h2 : distance O Q = 7)
  (h3 : distance P Q = 8) (h4 : midpoint_arc R P Q O) :
  distance P R = sqrt (98 - 14 * sqrt 33) :=
  sorry

end length_segment_PR_l168_168285


namespace cos_double_angle_sin_angle_sum_tan_double_angle_l168_168160

variable (α : ℝ)

axiom sin_alpha : sin α = 1 / 3
axiom alpha_in_I : α ∈ set.Ioo 0 (Real.pi / 2)

theorem cos_double_angle : cos (2 * α) = 7 / 9 :=
by
  have hα : sin α = 1 / 3 := sin_alpha
  have hα_in_I : α ∈ set.Ioo 0 (Real.pi / 2) := alpha_in_I
  sorry

theorem sin_angle_sum : sin (2 * α + (Real.pi / 3)) = (4 * Real.sqrt 2 + 7 * Real.sqrt 3) / 18 :=
by
  have hα : sin α = 1 / 3 := sin_alpha
  have hα_in_I : α ∈ set.Ioo 0 (Real.pi / 2) := alpha_in_I
  sorry

theorem tan_double_angle : tan (2 * α) = 4 * Real.sqrt 2 / 7 :=
by
  have hα : sin α = 1 / 3 := sin_alpha
  have hα_in_I : α ∈ set.Ioo 0 (Real.pi / 2) := alpha_in_I
  sorry

end cos_double_angle_sin_angle_sum_tan_double_angle_l168_168160


namespace sampling_method_is_systematic_l168_168033

-- Definitions corresponding to the conditions
def interval_sampling (interval: Nat) (time: Nat) : Prop := time % interval = 0
def large_population (size: Nat) : Prop := size > 100 -- assuming large means greater than 100

-- The proof problem statement
theorem sampling_method_is_systematic :
  ∀ (time: Nat) (interval: Nat) (size: Nat),
  interval_sampling interval time →
  large_population size →
  interval = 10 →
  "Systematic Sampling" := by
  sorry

end sampling_method_is_systematic_l168_168033


namespace seafood_noodles_l168_168029

theorem seafood_noodles (total_plates lobster_rolls spicy_hot_noodles : ℕ)
  (h_total : total_plates = 55)
  (h_lobster : lobster_rolls = 25)
  (h_spicy : spicy_hot_noodles = 14) :
  total_plates - (lobster_rolls + spicy_hot_noodles) = 16 :=
by
  sorry

end seafood_noodles_l168_168029


namespace smallest_rational_in_set_l168_168940

theorem smallest_rational_in_set : 
  ∀ (a b c d : ℚ), 
    a = -2/3 → b = -1 → c = 0 → d = 1 → 
    (a > b ∧ b < c ∧ c < d) → b = -1 := 
by
  intros a b c d ha hb hc hd h
  sorry

end smallest_rational_in_set_l168_168940


namespace prove_q_l168_168220

theorem prove_q 
  (p q : ℝ)
  (h : (∀ x, (x + 3) * (x + p) = x^2 + q * x + 12)) : 
  q = 7 :=
sorry

end prove_q_l168_168220


namespace sum_of_extremes_is_correct_l168_168952

def given_set := {0.34, 0.304, 0.034, 0.43}

def smallest_number (s : Set ℝ) : ℝ := 0.034
def largest_number (s : Set ℝ) : ℝ := 0.43
def sum_smallest_largest (s : Set ℝ) : ℝ := smallest_number s + largest_number s

theorem sum_of_extremes_is_correct : sum_smallest_largest given_set = 0.464 :=
by 
  have smallest := smallest_number given_set
  have largest := largest_number given_set
  have sum := smallest + largest
  sorry

end sum_of_extremes_is_correct_l168_168952


namespace smallest_digits_to_append_l168_168844

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168844


namespace range_of_m_l168_168483

noncomputable def f' (x : ℝ) : ℝ := (a - x) / Real.exp x

def g (x : ℝ) : ℝ := (Real.exp x) / x - 3 * x + 6

theorem range_of_m (a b m : ℝ) (h0 : ∀ x ∈ Ioo (1/2 : ℝ) (3/2), f x < 1 / (6 * x - 3 * x^2))
  (h1 : ∀ x ∈ Ioo (1/2 : ℝ) (3/2), f' x > 0)
  (h2 : ∀ x ∈ Ioo (1/2 : ℝ) (3/2), b < 3) :
  m ∈ Icc (-9/4) (Real.exp 1 - 3) :=
sorry

end range_of_m_l168_168483


namespace student_marks_l168_168235

theorem student_marks 
    (correct: ℕ) 
    (attempted: ℕ) 
    (marks_per_correct: ℕ) 
    (marks_per_incorrect: ℤ) 
    (correct_answers: correct = 27)
    (attempted_questions: attempted = 70)
    (marks_per_correct_condition: marks_per_correct = 3)
    (marks_per_incorrect_condition: marks_per_incorrect = -1): 
    (correct * marks_per_correct + (attempted - correct) * marks_per_incorrect) = 38 :=
by
    sorry

end student_marks_l168_168235


namespace least_odd_prime_factor_2023_pow_10_plus_1_l168_168125

open Nat

theorem least_odd_prime_factor_2023_pow_10_plus_1 :
  ∃ p : ℕ, Prime p ∧ p ≠ 2 ∧ p ∣ (2023^10 + 1) ∧ p = 41 :=
by
  have h1 : 2023^10 % 41 = (2023 % 41)^10 % 41,
  -- Computation details and further intermediate steps will go here.
  -- These steps are skipped with 'sorry' as the detailed calculation and verification parts,
  -- which include computational checks and modular arithmetic proofs.
  sorry

end least_odd_prime_factor_2023_pow_10_plus_1_l168_168125


namespace smallest_percent_increase_14_to_15_l168_168233

-- Definitions based on conditions:
def dollar_value : ℕ → ℕ
| n := if n < 10 then 100 * 2^n else 51200 + 100000 * (n - 9)

def percent_increase (n : ℕ) : ℚ :=
  (dollar_value (n + 1) - dollar_value n : ℚ) / dollar_value n * 100

-- Proof that the smallest percent increase occurs from question 14 to 15:
theorem smallest_percent_increase_14_to_15 :
  ∀ n m, (n ≠ 13 ∨ m ≠ 14) → percent_increase n ≥ percent_increase m →
  percent_increase 14 < percent_increase 13 := sorry

end smallest_percent_increase_14_to_15_l168_168233


namespace rectangle_problem_l168_168616

variable (A B C D F : Type)
variable [point : euclidean_geometry.point_geom](A B C D F)

noncomputable def AB_length : ℝ := 30
noncomputable def BC_length : ℝ := 15
noncomputable def ∠CBF : ℝ := 30 * real.pi / 180 -- converting degrees to radians for Lean compatibility

noncomputable def AF_length : ℝ := real.sqrt (1200 - 300 * real.sqrt 3)
noncomputable def area_ABF : ℝ := 75 * real.sqrt 3 / 2

theorem rectangle_problem (rect : euclidean_geometry.rectangle_geom A B C D) 
(F_on_CD : F ∈ line_segment CD) 
(angle_CBF : euclidean_geometry.measure_angle C B F = 30 * real.pi / 180) : 
euclidean_geometry.distance A F = real.sqrt (1200 -300 * real.sqrt 3) 
∧ euclidean_geometry.triangle_area A B F = 75 * real.sqrt 3 / 2 := 
by 
  sorry

end rectangle_problem_l168_168616


namespace complement_union_eq_l168_168205

variable (U : Set ℝ) (M N : Set ℝ)

noncomputable def complement_union (U M N : Set ℝ) : Set ℝ :=
  U \ (M ∪ N)

theorem complement_union_eq :
  U = Set.univ → 
  M = {x | |x| < 1} → 
  N = {y | ∃ x, y = 2^x} → 
  complement_union U M N = {x | x ≤ -1} :=
by
  intros hU hM hN
  unfold complement_union
  sorry

end complement_union_eq_l168_168205


namespace angle_comparison_l168_168437

theorem angle_comparison :
  (∀ x : ℝ, ∃ x1 x2 : ℝ, x1 = 60 ∧ x2 = 50 ∧ sin (x1.to_real) - sin (x2.to_real) = 0.1000) →
  (∃ x : ℝ, sin (x.to_real) = (Math.sqrt 3) / 2 - (1 / 10) → x < 50) :=
sorry

end angle_comparison_l168_168437


namespace smallest_digits_to_append_l168_168773

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168773


namespace ab_cannot_be_specific_values_l168_168595

theorem ab_cannot_be_specific_values (a b : ℝ) (h1 : ∀ x : ℝ, x > 0 → (ln (a * x) - 1) * (exp x - b) ≥ 0) : 
  ab ≠ e ∧ ab ≠ (25 / 4) :=
  sorry

end ab_cannot_be_specific_values_l168_168595


namespace question_eq_answer_l168_168883

theorem question_eq_answer 
  (x : ℝ) 
  (hx : x ≥ 1/6) : 
  ∃ a : ℝ, 
    2.344 * a = 
    sqrt((1 / 6) * ((3 * x + sqrt(6 * x - 1))⁻¹ + (3 * x - sqrt(6 * x - 1))⁻¹)) * 
    |x - 1| * x⁻¹⁄² ∧
    ((1/6 ≤ x ∧ x < 1/3 ∨ x ≥ 1) → a = (x - 1) / (3 * x - 1)) ∧
    (1/3 < x ∧ x < 1 → a = (1 - x) / (3 * x - 1)) :=
begin
  sorry
end

end question_eq_answer_l168_168883


namespace length_angle_bisector_XD_l168_168245

/-- In triangle XYZ, given XY = 4, XZ = 8, and cos(angle X) = 1/10, the length of angle bisector XD is approximately 2.907. -/
theorem length_angle_bisector_XD :
  ∀ (X Y Z : Type) [metric_space X] [metric_space Y] [metric_space Z] (XY XZ : ℝ) (cos_X : ℝ),
    XY = 4 ∧ XZ = 8 ∧ cos_X = 1 / 10 →
      ∃ (XD : ℝ), XD ≈ 2.907 :=
begin
  sorry
end

end length_angle_bisector_XD_l168_168245


namespace correct_relations_count_l168_168098

def rational (x : ℝ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ x = p / q
def positive_integers (x : ℕ) : Prop := x > 0

theorem correct_relations_count : 
  ¬(1/2 ∉ ℝ) ∧  (¬rational (real.sqrt 2)) ∧ 
  ¬(3 ∉ {x : ℕ | positive_integers x}) ∧ ¬rational (real.sqrt 3) → 
  ∃ n : ℕ, n = 1 :=
by
  intro h
  use 1
  sorry

end correct_relations_count_l168_168098


namespace circle_equation_and_line_l168_168552

noncomputable def x1 := 2
noncomputable def y1 := 2
noncomputable def qx := 2
noncomputable def qy := 3
noncomputable def l1 := 2
noncomputable def l2 := 6

theorem circle_equation_and_line :
  (∀ C : ℝ × ℝ,
     (C = (C.1, C.1 - 1)) → 
     (∃ r : ℝ, r = ∥(C.1 - x1) + (C.2 - y1)∥ ∧ 
     r = sqrt 13 → 
     (x^2 + (y + 1) ^ 2 = 13 ∨
     x = l1 ∨ 3 * x - 4 * y + l2 = 0))) :=
begin
  sorry
end

end circle_equation_and_line_l168_168552


namespace triangle_ab_value_l168_168169

theorem triangle_ab_value (a b c : ℝ) (A B C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by
  sorry

end triangle_ab_value_l168_168169


namespace conner_bonus_needed_l168_168302

/-- Sydney and Conner's rock collecting contest over five days -/
def sydney_day_one (s₀ : ℕ) : ℕ := s₀ + 4
def conner_day_one (c₀ : ℕ) (s_day1 : ℕ) : ℕ := 
  let collected := 8 * (s_day1 - s₀); in c₀ + collected / 2

def sydney_day_two (s_day1 : ℕ) : ℕ := s_day1 - s_day1 / 10
def conner_day_two (c_day1 : ℕ) (collected : ℕ) : ℕ := 
  c_day1 + 2 * collected - 7

def sydney_day_three (s_day2 : ℕ) (collected : ℕ) : ℕ := s_day2 + 2 * collected / 2
def conner_day_three (c_day2 : ℕ) (collected : ℕ) : ℕ := 
  c_day2 - collected / 4

def sydney_day_four (s_day3 : ℕ) (collected : ℕ) : ℕ :=
  s_day3 + 3 * collected - 5
def conner_day_four (c_day3 : ℕ) (collected : ℕ) : ℕ := 
  c_day3 + (3 * collected) / 2 - 9

noncomputable def sydney_day_five (s_day4 : ℕ) : ℕ := 
  s_day4 + Nat.round (Real.cbrt s_day4)
noncomputable def conner_day_five (c_day4 : ℕ) (collected : ℕ) : ℕ := 
  c_day4 + Nat.round (Real.sqrt collected) + 1

-- The final day 5 collections without any bonus
def final_sydney_collection := 
  sydney_day_five (sydney_day_four 789 32)
def final_conner_collection :=
  conner_day_five 794 32

-- The problem statement to prove:
theorem conner_bonus_needed (c₀ s₀ : ℕ) (s_day1 s_day2 s_day3 s_day4 : ℕ)
  (c_day1 c_day2 c_day3 c_day4 : ℕ) (collected : ℕ) :
  final_sydney_collection - final_conner_collection = 96 :=
sorry

end conner_bonus_needed_l168_168302


namespace count_valid_triples_l168_168152

def S (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def satisfies_conditions (a b c : ℕ) : Prop :=
  is_three_digit a ∧ is_three_digit b ∧ is_three_digit c ∧ 
  (a + b + c = 2005) ∧ (S a + S b + S c = 61)

def number_of_valid_triples : ℕ := sorry

theorem count_valid_triples : number_of_valid_triples = 17160 :=
sorry

end count_valid_triples_l168_168152


namespace sum_of_squares_of_medians_l168_168366

noncomputable def mAD (a b c : ℝ) := real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2
noncomputable def mBE (a b c : ℝ) := real.sqrt (2 * a^2 + 2 * c^2 - b^2) / 2
noncomputable def mCF (a b c : ℝ) := real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2

theorem sum_of_squares_of_medians (a b c : ℝ) (h₁ : a = 13) (h₂ : b = 13) (h₃ : c = 10) :
  (mAD a b c)^2 + (mBE a b c)^2 + (mCF a b c)^2 = 244 :=
by sorry

end sum_of_squares_of_medians_l168_168366


namespace second_integer_value_l168_168332

theorem second_integer_value (n : ℚ) (h : (n - 1) + (n + 1) + (n + 2) = 175) : n = 57 + 2 / 3 :=
by
  sorry

end second_integer_value_l168_168332


namespace max_planes_determined_by_points_l168_168275
-- Import the entire Mathlib library to ensure all necessary components are available

-- Define the problem statement
theorem max_planes_determined_by_points 
  (alpha beta : Type) -- Representing the two planes
  [is_plane alpha]   -- Assuming alpha is a plane
  [is_plane beta]    -- Assuming beta is a plane
  [parallel_planes alpha beta] -- alpha and beta are parallel
  (points_alpha : set alpha) 
  (points_beta : set beta)
  (h1 : points_alpha.card = 4) -- 4 points on plane alpha
  (h2 : points_beta.card = 5) -- 5 points on plane beta :
  (h3 : ∀ (p : alpha), p ∈ points_alpha) -- Each point from alpha belongs to points_alpha set
  (h4 : ∀ (p : beta), p ∈ points_beta) -- Each point from beta belongs to points_beta set 
  (h5 : ∀ (p1 p2 : alpha), p1 ≠ p2)    -- All points in points_alpha are distinct
  (h6 : ∀ (p1 p2 : beta), p1 ≠ p2)     -- All points in points_beta are distinct :
  ∃ (n : ℕ), n = 72 :=
sorry

end max_planes_determined_by_points_l168_168275


namespace parity_of_pq_l168_168178

theorem parity_of_pq (x y m n p q : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 0)
    (hx : x = p) (hy : y = q) (h1 : x - 1998 * y = n) (h2 : 1999 * x + 3 * y = m) :
    p % 2 = 0 ∧ q % 2 = 1 :=
by
  sorry

end parity_of_pq_l168_168178


namespace find_k_l168_168208

def vector (α : Type*) := (α × α)

def dot_product {α : Type*} [Add α] [Mul α] (v1 v2 : vector α) : α :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k (k : ℝ) : dot_product (2, 1) (-1, k) = 0 → k = 2 :=
by
  intro h
  have h' : 2 * (-1) + 1 * k = -2 + k := rfl
  rw h' at h
  linarith

end find_k_l168_168208


namespace quadratic_nonnegative_l168_168518

theorem quadratic_nonnegative (x y : ℝ) : x^2 + x * y + y^2 ≥ 0 :=
by sorry

end quadratic_nonnegative_l168_168518


namespace positive_difference_prime_factors_l168_168006

theorem positive_difference_prime_factors (n : ℕ) (h_n : n = 278459)
  (h_prime_factors : ∃ a b c : ℕ, Prime a ∧ Prime b ∧ Prime c ∧ a * b * c = n ∧ {a, b, c} = {3, 19, 257}) : 
  257 - 3 = 254 := by
  sorry

end positive_difference_prime_factors_l168_168006


namespace triangle_angles_and_area_l168_168244

theorem triangle_angles_and_area (A B C : ℝ) (AM : ℝ) 
  (h1 : sin A = sin B)
  (h2 : sin A = -cos C)
  (h3 : A + B + C = π)
  (h4 : AM = sqrt 7) :
  A = π / 6 ∧ B = π / 6 ∧ C = 2 * π / 3 ∧ 
  let x := 2 in
  let area := 1 / 2 * x * x * sin C in
  area = 2 * sqrt 3 :=
by
  sorry

end triangle_angles_and_area_l168_168244


namespace minimum_value_theta_l168_168561

open Real

def f (x : ℝ) : ℝ := 2 * sin x * cos x - sin x ^ 2 + 1

theorem minimum_value_theta (θ : ℝ) (h : ∀ x : ℝ, f x ≥ f θ) :
  (sin (2 * θ) + cos (2 * θ)) / (sin (2 * θ) - cos (2 * θ)) = 3 := by
sorry

end minimum_value_theta_l168_168561


namespace smallest_digits_to_append_l168_168776

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168776


namespace determine_a_range_l168_168697

variable (a : ℝ)

-- Define proposition p as a function
def p : Prop := ∀ x : ℝ, x^2 + x > a

-- Negation of Proposition q
def not_q : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 2 - a ≠ 0

-- The main theorem to be stated, proving the range of 'a'
theorem determine_a_range (h₁ : p a) (h₂ : not_q a) : -2 < a ∧ a < -1 / 4 := sorry

end determine_a_range_l168_168697


namespace sufficient_not_necessary_condition_arithmetic_l168_168190

noncomputable def sequence_condition {α : Type*} (a : α) (b : α) (n : ℕ) (S : ℕ → α) (a_n : ℕ → α) 
  [HasAdd α] [HasPow α ℕ] [Mul α] [HasAdd α] [HasSmul ℕ α] [NeZero a] : Prop :=
  ∀ n, S n = (a_n n) ^ 2 + b • n

theorem sufficient_not_necessary_condition_arithmetic {α : Type*} [LinearOrderedField α] 
  (a b : α) (S : ℕ → α) (a_n : ℕ → α) [NeZero a] :
  (∀ n, S n = (a_n n) ^ 2 + b • n) → 
  (sufficient_condition (is_arithmetic_sequence a_n) S) ∧ 
  ¬ (necessary_condition (is_arithmetic_sequence a_n) S) :=
by
  sorry

def is_arithmetic_sequence {α : Type*} (a_n : ℕ → α) [HasAdd α] [HasSub α] [HasMul α] : Prop :=
  ∃ d, ∀ n, a_n (n + 1) = a_n n + d

def sufficient_condition {α : Prop} (P Q : α) := (P → Q)
def necessary_condition {α : Prop} (P Q : α) := (Q → P)

end sufficient_not_necessary_condition_arithmetic_l168_168190


namespace measure_W_l168_168668

-- Definition of the problem
variables (p q : Line) (r : Line) (Z : Point) 
          (X Y W : Angle)
          (m_X m_Y m_Z m_W : ℝ) -- Measures of angles in degrees

-- Conditions
axiom parallel_pq : Parallel p q
axiom measure_X : m_X = 100
axiom measure_Y : m_Y = 130
axiom measure_Z : m_Z = 70
axiom intersection_Z : r.IntersectsAt p Z

-- Goal
theorem measure_W : m_W = 120 := 
by
  -- Use conditions to establish the measure of W
  sorry

end measure_W_l168_168668


namespace find_AB_l168_168627

-- Definitions based on the problem's statement
variables (ABC : Type) [linear_ordered_field ABC] (A B C : ABC)

-- Problem conditions
def angle_A_90 : Prop := angle A = 90
def BC_eq_10 : Prop := BC = 10
def tan_C_eq_3_cos_B : Prop := tan C = 3 * cos B

-- The goal is to prove AB = 20 * sqrt 2 / 3
theorem find_AB (h1 : angle_A_90) (h2 : BC_eq_10) (h3 : tan_C_eq_3_cos_B) : AB = 20 * sqrt 2 / 3 := 
by sorry

end find_AB_l168_168627


namespace smallest_digits_to_append_l168_168778

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168778


namespace part1_part2_l168_168197

noncomputable def f (a b x : ℝ) : ℝ := x^3 + (1 - a) * x^2 - a * (a + 2) * x + b
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 + 2 * (1 - a) * x - a * (a + 2)

-- Assertions for part (I)
theorem part1 (a b : ℝ) (h_origin : f a b 0 = 0) (h_slope : f' a 0 = -3) :
  b = 0 ∧ (a = -3 ∨ a = 1) :=
by
  sorry

-- Assertions for part (II)
theorem part2 (a : ℝ) (h_non_monotonic : ∃ (x y : ℝ), x ≠ y ∧ -1 < x ∧ x < 1 ∧ f' a x = 0 ∧ -1 < y ∧ y < 1 ∧ f' a y = 0) :
  a ∈ Ioo (-5 : ℝ) (-1 / 2) ∨ a ∈ Ioo (-1 / 2) 1 :=
by
  sorry

end part1_part2_l168_168197


namespace distinct_values_count_l168_168953

theorem distinct_values_count :
  ∃ (S : Finset ℕ), S.card = 104 ∧
    (∀ p q : ℕ, p ∈ Finset.range 16 → p > 0 → q ∈ Finset.range 16 → q > 0 → (p * q + p + q) ∈ S) := 
sorry

end distinct_values_count_l168_168953


namespace maximum_students_l168_168232

-- Define the number of desks in each row based on the given recurrence relation
def desks_in_row : ℕ → ℕ
| 0       := 10  -- First row has 10 desks
| (n + 1) := desks_in_row n + (n + 1)

-- Define the number of seats in each row based on the 75% occupancy rule
def seats_in_row (n : ℕ) : ℕ := (desks_in_row n * 3) / 4

-- Define the total number of seats according to the problem's conditions
def total_seats : ℕ := (List.range 8).sum seats_in_row

theorem maximum_students : total_seats = 141 :=
by
  -- Proof omitted
  sorry

end maximum_students_l168_168232


namespace probability_of_matching_pair_l168_168963

theorem probability_of_matching_pair 
  (blue_socks gray_socks white_socks : ℕ) 
  (h_blue_socks : blue_socks = 12) 
  (h_gray_socks : gray_socks = 10) 
  (h_white_socks : white_socks = 8) :
  let total_socks := blue_socks + gray_socks + white_socks in
  let total_pairs := (Nat.choose total_socks 2) in
  let blue_pairs := (Nat.choose blue_socks 2) in
  let gray_pairs := (Nat.choose gray_socks 2) in
  let white_pairs := (Nat.choose white_socks 2) in
  (blue_pairs + gray_pairs + white_pairs : ℚ) / total_pairs = 139 / 435 :=
by {
  sorry
}

end probability_of_matching_pair_l168_168963


namespace product_of_solutions_l168_168132

theorem product_of_solutions : 
  let S := {x : ℝ | abs x = 3 * (abs x - 2)} in
  ∀ x y ∈ S, x ≠ y → (x * y) = -9 :=
by
  introv hx hy hxy
  have hS : S = {3, -3} := sorry -- Showing that S is exactly the set {3, -3}
  have hx' : x = 3 ∨ x = -3 := by rwa [hS] at hx
  have hy' : y = 3 ∨ y = -3 := by rwa [hS] at hy
  cases hx'; cases hy'; contradiction <|> norm_num

end product_of_solutions_l168_168132


namespace zeus_max_10gram_l168_168382

def zeus_can_find_9_gram_coin (N M : ℕ) : Prop :=
  ∃ strategy : (fin 16 → fin 16), 
    ∀ w : fin 16 → fin 2, 
      ∀ k < 4, -- number of weighings
        ∃ i : fin 16, w i = 1  -- at least one 9-gram coin is found

theorem zeus_max_10gram (N M : ℕ) : zeus_can_find_9_gram_coin N M → N ≤ 15 :=
sorry

end zeus_max_10gram_l168_168382


namespace flour_maximum_weight_l168_168904

/-- Given that the bag of flour is marked with 25kg + 50g, prove that the maximum weight of the flour is 25.05kg. -/
theorem flour_maximum_weight :
  let weight_kg := 25
  let weight_g := 50
  (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 :=
by 
  -- provide definitions
  let weight_kg := 25
  let weight_g := 50
  have : (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 := sorry
  exact this

end flour_maximum_weight_l168_168904


namespace tangent_to_circle_exists_l168_168175

open_locale classical

variables {W : Type*} [metric_space W] [normed_space ℝ W] {A B C M D P Q R : W}
variable {l : set W}

-- Definitions based on the conditions
def diameter (A B : W) (W : set W) : Prop :=
  ∃ (O : W) (r : ℝ), W = {X : W | dist O X = r} ∧ dist O A = r ∧ dist O B = r ∧ dist A B = 2 * r

def tangent_at (A : W) (l : set W) (W : set W) : Prop :=
  ∃ (O : W) (r : ℝ), W = {X : W | dist O X = r} ∧ A ∈ l ∧ ∀ (X : W), X ∈ l → X ≠ A → dist O X > r

def points_on_line (C M D : W) (l : set W) : Prop :=
  C ∈ l ∧ M ∈ l ∧ D ∈ l

def points_on_same_line (CM MD : ℝ) (C M D : W) : Prop :=
  dist C M = CM ∧ dist M D = MD ∧ CM = MD

def intersection (BC BD P Q : W) (W : set W) : Prop :=
  ∃ (P' Q' : W), P' ∈ intersect_line_circle B C W ∧ Q' ∈ intersect_line_circle B D W ∧ P' = P ∧ Q' = Q

-- The proof problem statement
theorem tangent_to_circle_exists
  (h1 : diameter A B W) 
  (h2 : tangent_at A l W) 
  (h3 : points_on_line C M D l) 
  (h4 : points_on_same_line (dist C M) (dist M D) C M D) 
  (h5 : intersection B C P W) 
  (h6 : intersection B D Q W) :
  ∃ (R : W), R ∈ line_through B M ∧ tangent_to_circle R P W ∧ tangent_to_circle R Q W :=
sorry

end tangent_to_circle_exists_l168_168175


namespace min_abs_alpha_gamma_l168_168413

open Complex

-- Define the function f
def f (α γ z : ℂ) : ℂ := (5 + I) * z^2 + α * z + γ

-- Define the main theorem
theorem min_abs_alpha_gamma (α γ : ℂ) :
  (∀ z : ℂ, f α γ 1 ∈ ℝ ∧ f α γ I ∈ ℝ) →
  abs α + abs γ ≥ real.sqrt 2 :=
by
  -- Skip proof
  sorry

end min_abs_alpha_gamma_l168_168413


namespace locus_of_circumcenter_is_arc_l168_168004

variables {k : Type*} [linear_ordered_field k]
variables (r : ℝ)
-- Vertices A and C are fixed points on a circle with radius r
def is_on_circle (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 = r^2

def A : ℝ × ℝ := (-r / real.sqrt 2, -r / real.sqrt 2)
def C : ℝ × ℝ := (r, 0)

-- Vertices B and D are moving points on the circle such that BC = CD
variables (α : ℝ)
def B : ℝ × ℝ := (r * cos α, r * sin α)
def D : ℝ × ℝ := (r * cos α, -r * sin α)

-- M is the intersection point of AC and BD
def M : ℝ × ℝ := (r * cos α, 0)

-- F is the circumcenter of triangle ABM
def is_circumcenter (F : ℝ × ℝ) (A B M : ℝ × ℝ) : Prop :=
  dist A F = dist B F ∧ dist B F = dist M F

-- The locus of F forms an arc of a circle
def is_on_arc (F : ℝ × ℝ) : Prop :=
  sorry -- Skipping the formal characterization and proof of an arc

theorem locus_of_circumcenter_is_arc : ∀ (α : ℝ), 
  is_on_circle (A) r ∧ is_on_circle (C) r ∧ 
  (∀ α, is_on_circle (B α) r ∧ is_on_circle (D α) r ∧ 
        dist (B α) (C) = dist (D α) (C)) ∧
  (∀ α, let M := M α in ∃ F, is_circumcenter F A (B α) M → is_on_arc F) :=
sorry

end locus_of_circumcenter_is_arc_l168_168004


namespace smallest_number_of_digits_to_append_l168_168804

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168804


namespace smallest_digits_to_append_l168_168777

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168777


namespace smallest_digits_to_append_l168_168854

def lcm_of_1_to_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

theorem smallest_digits_to_append (n : ℕ) : lcm_of_1_to_9 = 2520 ∧ (20140000 ≤ 2014 * 10^n ≤ 20149999) → n = 4 :=
by
  unfold lcm_of_1_to_9
  sorry

end smallest_digits_to_append_l168_168854


namespace division_result_l168_168009

theorem division_result : (5 * 6 + 4) / 8 = 4.25 :=
by
  sorry

end division_result_l168_168009


namespace probability_of_red_ball_l168_168902

theorem probability_of_red_ball (red_balls yellow_balls : ℕ) (h1 : red_balls = 3) (h2 : yellow_balls = 2) :
  (red_balls.toReal / (red_balls + yellow_balls).toReal) = 3 / 5 :=
by
  sorry

end probability_of_red_ball_l168_168902


namespace smallest_digits_to_append_l168_168848

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168848


namespace smallest_four_digit_solution_l168_168505

theorem smallest_four_digit_solution :
  ∃ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧
  (3 * x ≡ 6 [MOD 12]) ∧
  (5 * x + 20 ≡ 25 [MOD 15]) ∧
  (3 * x - 2 ≡ 2 * x [MOD 35]) ∧
  x = 1274 :=
by
  sorry

end smallest_four_digit_solution_l168_168505


namespace distance_to_line_eq_l168_168987

theorem distance_to_line_eq {a : ℝ} :
  let A := (-2: ℝ, 0: ℝ)
  let B := (4: ℝ, a)
  let l := (3: ℝ, -4: ℝ, 1: ℝ)
  let dist := λ x y l => |l.1 * x + l.2 * y + l.3| / real.sqrt (l.1 ^ 2 + l.2 ^ 2)
  dist A.1 A.2 l = dist B.1 B.2 l ↔ a = 2 ∨ a = 4.5 :=
by
  sorry

end distance_to_line_eq_l168_168987


namespace geometric_series_common_ratio_l168_168444

theorem geometric_series_common_ratio (a S r : ℝ) (h1 : a = 512) (h2 : S = 3072) 
(h3 : S = a / (1 - r)) : r = 5/6 := 
sorry

end geometric_series_common_ratio_l168_168444


namespace pond_length_l168_168716

theorem pond_length (
    W L P : ℝ) 
    (h1 : L = 2 * W) 
    (h2 : L = 32) 
    (h3 : (L * W) / 8 = P^2) : 
  P = 8 := 
by 
  sorry

end pond_length_l168_168716


namespace problem1_problem2_l168_168944

variable (x y a b c d : ℝ)
variable (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)

-- Problem 1: Prove (x + y) * (x^2 - x * y + y^2) = x^3 + y^3
theorem problem1 : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

-- Problem 2: Prove ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6)
theorem problem2 (a b c d : ℝ) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) : 
  ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6) := 
  sorry

end problem1_problem2_l168_168944


namespace number_of_12_digit_integers_with_consecutive_zeros_is_3719_l168_168209

def number_of_12_digit_integers_with_consecutive_zeros : ℕ :=
  let total := 2^12 in
  let fib := λ n =>
    let rec fib_aux (a b : ℕ) (n : ℕ) :=
      match n with
      | 0 => a
      | n + 1 => fib_aux b (a + b) n
    fib_aux 2 3 (n - 2)
  total - fib 12

theorem number_of_12_digit_integers_with_consecutive_zeros_is_3719 :
  number_of_12_digit_integers_with_consecutive_zeros = 3719 :=
by
  sorry

end number_of_12_digit_integers_with_consecutive_zeros_is_3719_l168_168209


namespace probability_457_more_ones_than_sixes_l168_168219

noncomputable def probability_of_more_ones_than_sixes : ℚ :=
  let total_ways := 6^6 in
  let same_number_ways :=
    (4^6) + (Nat.choose 6 1 * Nat.choose 5 1 * 4^4) + (Nat.choose 6 2 * Nat.choose 4 2 * 4^2) + (Nat.choose 6 3 * Nat.choose 3 3) in
  let desired_probability := (1/2) * (1 - (same_number_ways / total_ways)) in
  let simplified_probability := 8355 / 23328 in
  simplified_probability

theorem probability_457_more_ones_than_sixes :
  probability_of_more_ones_than_sixes = 8355 / 23328 := sorry

end probability_457_more_ones_than_sixes_l168_168219


namespace rankings_count_l168_168338

theorem rankings_count (students : Fin 5) (scores : Fin 5 -> ℕ):
  (∀ i j : Fin 5, i ≠ j → scores i ≠ scores j) →
  (∀ i : Fin 5, scores i ≠ max (scores <$> Finset.univ)) →
  (∀ i : Fin 5, scores i ≠ min (scores <$> Finset.univ)) →
  (∑ i : Finset.perm (Fin 5), 1) = 78 :=
begin
  sorry
end

end rankings_count_l168_168338


namespace minimum_participants_round_robin_l168_168234

theorem minimum_participants_round_robin : 
  ∃ (k : ℕ), (∀ w : ℕ, 0.68 * k < w ∧ w < 0.69 * k) → k + 1 = 17 :=
by {
  use 16,
  intros w hw,
  have h1 : 0.68 * 16 < w,
  by linarith [hw.1],
  have h2 : w < 0.69 * 16,
  by linarith [hw.2],
  split,
  exacts [h1, h2]
}
sorry

end minimum_participants_round_robin_l168_168234


namespace algorithm_is_determinate_l168_168727

-- Definitions from the conditions
def finite_steps {α : Type*} (algorithm : list α) : Prop := 
  ∃ n, algorithm.length = n

def precise_steps {α : Type*} (algorithm : list α) : Prop :=
  ∀ step ∈ algorithm, step ≠ none

def effectively_executed {α : Type*} (algorithm : list α) : Prop :=
  ∀ step ∈ algorithm, execute step ≠ none

def yields_determinate_result {α : Type*} (algorithm : list α) : Prop :=
  ∃ result, run algorithm = result

def no_ambiguity {α : Type*} (algorithm : list α) : Prop :=
  ∀ step1 step2 ∈ algorithm, (step1 = step2) ∨ (step1 ≠ step2)

-- The main theorem statement
theorem algorithm_is_determinate (algorithm : list (option (fin 100))) :
  finite_steps algorithm →
  precise_steps algorithm →
  effectively_executed algorithm →
  yields_determinate_result algorithm →
  no_ambiguity algorithm →
  ∀ step ∈ algorithm, step = determinate :=
sorry

end algorithm_is_determinate_l168_168727


namespace boatman_current_speed_and_upstream_time_l168_168901

variables (v : ℝ) (v_T : ℝ) (t_up : ℝ) (t_total : ℝ) (dist : ℝ) (d1 : ℝ) (d2 : ℝ)

theorem boatman_current_speed_and_upstream_time
  (h1 : dist = 12.5)
  (h2 : d1 = 3)
  (h3 : d2 = 5)
  (h4 : t_total = 8)
  (h5 : ∀ t, t = d1 / (v - v_T))
  (h6 : ∀ t, t = d2 / (v + v_T))
  (h7 : dist / (v - v_T) + dist / (v + v_T) = t_total) :
  v_T = 5 / 6 ∧ t_up = 5 := by
  sorry

end boatman_current_speed_and_upstream_time_l168_168901


namespace exists_root_in_interval_l168_168269

noncomputable def f (x : ℝ) := 3^x + 3 * x - 8

theorem exists_root_in_interval :
  f 1 < 0 → f 1.5 > 0 → f 1.25 < 0 → ∃ x ∈ (Set.Ioo 1.25 1.5), f x = 0 :=
by
  intros h1 h2 h3
  sorry

end exists_root_in_interval_l168_168269


namespace find_number_l168_168919

-- Define a constant to represent the number
def c : ℝ := 1002 / 20.04

-- Define the main theorem
theorem find_number (x : ℝ) (h : x - c = 2984) : x = 3034 := by
  -- The proof will be placed here
  sorry

end find_number_l168_168919


namespace seventh_grade_a_seventh_grade_b_eighth_grade_c_l168_168713

-- Define the conditions for the problem
def seventh_grade_scores : List ℕ := [99, 95, 95, 91, 100, 86, 77, 93, 85, 79]
def eighth_grade_scores : List ℕ := [99, 91, 97, 63, 96, 97, 100, 94, 87, 76]

-- Calculate the necessary values based on the conditions
def num_students_70_to_80 (scores : List ℕ) : ℕ := List.count (λ x => 70 < x ∧ x ≤ 80) scores

def median (scores : List ℕ) : ℕ :=
  let sorted_scores := scores.qsort (· < ·)
  let n := sorted_scores.length
  if n % 2 = 0 then (sorted_scores.get! (n / 2 - 1) + sorted_scores.get! (n / 2)) / 2 
  else sorted_scores.get! (n / 2)

def mode (scores : List ℕ) : ℕ :=
  scores.foldl (λ m x => if scores.count x > scores.count m then x else m) 0

-- Lean proof problem statements
theorem seventh_grade_a : num_students_70_to_80 seventh_grade_scores = 2 := by sorry
theorem seventh_grade_b : median seventh_grade_scores = 92 := by sorry
theorem eighth_grade_c : mode eighth_grade_scores = 97 := by sorry

end seventh_grade_a_seventh_grade_b_eighth_grade_c_l168_168713


namespace num_palindromic_seven_digit_integers_l168_168346

theorem num_palindromic_seven_digit_integers : 
  let digits := {5, 6, 7}
  let count_palindromes := (λ a b c d : ℕ, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits) in
  ∃ (num_palindromes : ℕ), num_palindromes = 3^4 ∧ count_palindromes = (num_palindromes) :=
sorry

end num_palindromic_seven_digit_integers_l168_168346


namespace area_AEF_l168_168626

variables (A B C D E F : Type)

/- Definitions representing the geometric properties -/
def is_midpoint (M P Q : Type) : Prop :=
  ∃ (m n : ℝ), m + n = 1 ∧ M = m • P + n • Q

def area_of_triangle (A B C : Type) : ℝ :=  -- Placeholder definition
sorry

/- Given Conditions -/
variables (area_ABC : ℝ)
variables (is_mid_D_AB : is_midpoint D A B)
variables (is_mid_E_DB : is_midpoint E D B)
variables (is_mid_F_BC : is_midpoint F B C)

-- The proof goal
theorem area_AEF (h : area_of_triangle A B C = 96)
  (hD : is_midpoint D A B) (hE : is_midpoint E D B) (hF : is_midpoint F B C) :
  area_of_triangle A E F = 36 :=
sorry

end area_AEF_l168_168626


namespace train_b_time_l168_168759

variable (x t: ℝ)
variable trainA_speed trainB_speed trainA_time : ℝ
variable total_distance : ℝ

namespace TrainTrip

theorem train_b_time : 
  ∀ (trainA_speed trainB_speed trainA_time: ℝ), 
  trainA_speed = 120 → 
  trainB_speed = 160 → 
  trainA_time = 16 → 
  (120 * 16) + (160 * (x + t)) = (120 * (x + 16)) ∧ 
  (120 * (x + 16) = 160 * (x + t)) → 
  t = 8 :=
by
  intros trainA_speed trainB_speed trainA_time hsA hsB htA
  simp [hsA, hsB, htA]
  sorry

end TrainTrip

end train_b_time_l168_168759


namespace smallest_digits_to_append_l168_168849

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l168_168849


namespace middle_school_students_count_l168_168405

variable (total_students : ℕ) (sample_size : ℕ) (middle_sample : ℕ) (middle_students : ℕ)

theorem middle_school_students_count :
  total_students = 2000 → sample_size = 400 → middle_sample = 180 → middle_students = (2000 * 180) / 400 → middle_students = 900 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have h5 : 2000 * 180 = 360000 := by norm_num
  have h6 : 360000 / 400 = 900 := by norm_num
  rw [h5, h6] at h4
  exact h4

end middle_school_students_count_l168_168405


namespace determine_all_numbers_with_two_questions_l168_168283

noncomputable def determine_numbers (x : Fin 10 → ℕ) (S : ℕ) (S' : ℕ) (n : ℕ) : Prop :=
  S = ∑ i, x i ∧
  S' = ∑ i, 10^(n * i.cast) * x i ∧
  (10^n > S)

theorem determine_all_numbers_with_two_questions {x : Fin 10 → ℕ} :
  ∃ S S' n, determine_numbers x S S' n :=
begin
  -- The structure of the proof would go here, but it is omitted.
  -- The problem translation required the theorem statement only.
  sorry
end

end determine_all_numbers_with_two_questions_l168_168283


namespace critical_point_of_x_cubed_l168_168311

noncomputable def f (x : ℝ) : ℝ := x^3

theorem critical_point_of_x_cubed :
  ∃ x : ℝ, (∂ f / ∂ x) x = 0 ↔ x = 0 :=
by
  sorry

end critical_point_of_x_cubed_l168_168311


namespace smallest_append_digits_l168_168840

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168840


namespace lewis_weekly_earning_l168_168666

theorem lewis_weekly_earning
  (weeks : ℕ)
  (weekly_rent : ℤ)
  (total_savings : ℤ)
  (h1 : weeks = 1181)
  (h2 : weekly_rent = 216)
  (h3 : total_savings = 324775)
  : ∃ (E : ℤ), E = 49075 / 100 :=
by
  let E := 49075 / 100
  use E
  sorry -- The proof would go here

end lewis_weekly_earning_l168_168666


namespace sum_of_factorable_polynomial_is_zero_l168_168264

theorem sum_of_factorable_polynomial_is_zero :
  ∃ T : ℤ, 
    (T = ∑ d in {d | ∃ u v : ℤ, d ≠ 0 ∧ u + v = -d ∧ u * v = 2016 * d }, d) ∧ |T| = 0 :=
by
  sorry

end sum_of_factorable_polynomial_is_zero_l168_168264


namespace smallest_digits_to_append_l168_168828

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168828


namespace speed_of_goods_train_l168_168913

theorem speed_of_goods_train
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_crossing : ℕ)
  (h_length_train : length_train = 240)
  (h_length_platform : length_platform = 280)
  (h_time_crossing : time_crossing = 26)
  : (length_train + length_platform) / time_crossing * (3600 / 1000) = 72 := 
by sorry

end speed_of_goods_train_l168_168913


namespace sequence_general_formula_and_max_n_l168_168536

theorem sequence_general_formula_and_max_n {a : ℕ → ℝ} {S : ℕ → ℝ} {T : ℕ → ℝ}
  (hS2 : S 2 = (3 / 2) * a 2 - 1) 
  (hS3 : S 3 = (3 / 2) * a 3 - 1) :
  (∀ n, a n = 2 * 3^(n - 1)) ∧ 
  (∃ n : ℕ, (8 / 5) * T n + n / (5 * 3 ^ (n - 1)) ≤ 40 / 27 ∧ ∀ k > n, 
    (8 / 5) * T k + k / (5 * 3 ^ (k - 1)) > 40 / 27) :=
by
  sorry

end sequence_general_formula_and_max_n_l168_168536


namespace infinite_set_of_n_l168_168653

def largest_prime_divisor (n : ℕ) : ℕ :=
  if h : n ≥ 2 then some (classical.some_spec (nat.exists_prime_and_dvd (by linarith)))
  else 2  -- placeholder, shouldn't reach here if n ≥ 2

theorem infinite_set_of_n (h : Π n : ℕ, n ≥ 2 → ℕ) :
  ∃ S : set ℕ, set.infinite S ∧ ∀ n ∈ S, h n < h (n+1) ∧ h (n+1) < h (n+2) :=
sorry

end infinite_set_of_n_l168_168653


namespace only_negative_number_is_minus_2023_l168_168062

theorem only_negative_number_is_minus_2023 :
  ∀ (num : ℤ), (num = 2023 ∨ num = -2023 ∨ num = (1 : ℚ) / 2023 ∨ num = 0) →
  (num < 0 ↔ num = -2023) :=
by
  intros num h
  cases h
  case inl h₁ => -- num = 2023
    rw h₁
    exact not_lt_zero 2023
  case inr h₂ =>
    cases h₂
    case inl h₃ => -- num = -2023
      rw h₃
      simp -- -2023 < 0
    case inr h₄ =>
      cases h₄
      case inl h₅ => -- num = (1 : ℚ) / 2023
        rw h₅
        apply not_lt_zero
        exact div_pos one_pos (by norm_num)
        -- To show (1 : ℚ) / 2023 > 0, need that 2023 > 0 which can be solved by norm_num
      case inr h₆ => -- num = 0
        rw h₆
        exact iff_mp (eq_refl 0) (by refl)
        -- To show 0 < 0 is false, so 0 is not negative

end only_negative_number_is_minus_2023_l168_168062


namespace max_value_of_P_l168_168155

noncomputable def P (a : ℝ) : ℝ :=
  let x : ℝ := (Set.Icc 0 (a^2)).some -- picking random x in [0, a^2]
  let y : ℝ := (Set.Icc 0 a).some -- picking random y in [0, a]
  if (Math.cos (π * x))^2 + (Math.cos (π * y))^2 > 1 then 1 else 0

theorem max_value_of_P :
  ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ P a = 1/2 :=
sorry

end max_value_of_P_l168_168155


namespace probability_sum_is_odd_l168_168401

-- Define the problem
noncomputable def probability_odd_sum (n k : ℕ) (balls : Fin n) : ℝ :=
  let odd_balls := (Finset.filter (λ x : Fin n, ↑x % 2 = 1) Finset.univ).card
  let even_balls := (Finset.filter (λ x : Fin n, ↑x % 2 = 0) Finset.univ).card
  let favorable_cases := ((Finset.card (Finset.filter (λ x, ↑x % 2 = 1) (Finset.range 12).choose 5)) *
                          (Finset.card (Finset.filter (λ x, ↑x % 2 = 0) (Finset.range 12).choose 2))
                          + (Finset.card (Finset.filter (λ x, ↑x % 2 = 1) (Finset.range 12).choose 3)) *
                          (Finset.card (Finset.filter (λ x, ↑x % 2 = 0) (Finset.range 12).choose 4))
                          + (Finset.card (Finset.filter (λ x, ↑x % 2 = 1) (Finset.range 12).choose 1)) *
                          (Finset.card (Finset.filter (λ x, ↑x % 2 = 0) (Finset.range 12).choose 6)))
                          .to_nat
  let total_outcomes := (Finset.range 12).choose 7
  favorable_cases / total_outcomes.to_nat

theorem probability_sum_is_odd (n k : ℕ) (balls : Fin n) :
  probability_odd_sum 12 7 balls = 1/2 := by
  -- Provide conditions and skipped proof
  sorry

end probability_sum_is_odd_l168_168401


namespace liquid_x_percentage_combined_l168_168455

-- Define the given conditions
def solutionA : ℝ := 500
def solutionB : ℝ := 700
def percentageXInA : ℝ := 0.8 / 100
def percentageXInB : ℝ := 1.8 / 100

-- Statement to prove
theorem liquid_x_percentage_combined :
  ( ((percentageXInA * solutionA) + (percentageXInB * solutionB)) / (solutionA + solutionB) ) * 100 = 1.3833 :=
by
  sorry

end liquid_x_percentage_combined_l168_168455


namespace problem_l168_168937

def same_type (a b : ℝ) : Prop := sqrt a = sqrt b ∨ sqrt a = 1 / sqrt b

theorem problem :
    ¬same_type 12 (1 / 2) ∧
    ¬same_type 18 27 ∧
    same_type 3 (1 / 3) ∧
    ¬same_type 45 54 :=
by sorry

end problem_l168_168937


namespace intersection_of_A_and_B_l168_168467

open Set Real

-- Define the universal set U as the set of all real numbers
def U := ℝ

-- Define set A
def A : Set ℝ := {x | 2^x > 4}

-- Define set B
def B : Set ℝ := {x | log 3 x < 1}

-- The statement to be proven
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x < 3} := 
by sorry

end intersection_of_A_and_B_l168_168467


namespace find_a_from_inequality_l168_168571

theorem find_a_from_inequality (a : ℝ) :
  (∀ x : ℝ, (ax - 1) / (x + 1) < 0 ↔ x < -1 ∨ x > -1 / 2) → a = -2 :=
by
  intro h
  have h1 : (∀ x : ℝ, x < -1 ∨ x > -1 / 2 ↔ (ax - 1) / (x + 1) < 0) := h
  sorry

end find_a_from_inequality_l168_168571


namespace inclination_angle_l168_168699

theorem inclination_angle (θ : ℝ) : 
  (∃ (x y : ℝ), x + y - 3 = 0) → θ = 3 * Real.pi / 4 := 
sorry

end inclination_angle_l168_168699


namespace part_a_proof_part_b_proof_l168_168898

noncomputable def has_memoryless_property (X : ℝ → [0, ∞]) :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → P (X > x + y ∣ X > x) = P (X > y)

noncomputable def part_a_statement (X : ℝ → [0, ∞]) :=
  has_memoryless_property X → 
  (P (X = 0) = 1 ∨ P (X = ∞) = 1 ∨ (∃ λ > 0, ∀ x, P (X > x) = exp (-λ * x)))

noncomputable def part_b_statement (X : ℝ → [0, ∞]) (f : ℝ → ℝ) :=
  (∀ x ≥ 0, P (X > x) > 0) →
  (∀ (y : ℝ), 0 ≤ y →
    (∃ f_l : ℝ → ℝ, ∀ x, (x → ∞) → P (X > x + y ∣ X > x) → f_l(y)) → 
    (∃ λ ≥ 0, ∀ y, f(y) = exp (-λ * y)) ∧ 
    (∀ x, P (X > x) = exp (-λ * x + o(x))))

-- Using sorry to leave out the proof
theorem part_a_proof (X : ℝ → [0, ∞]) : part_a_statement X := sorry

theorem part_b_proof (X : ℝ → [0, ∞]) (f : ℝ → ℝ) : part_b_statement X f := sorry

end part_a_proof_part_b_proof_l168_168898


namespace max_combined_weight_l168_168432

noncomputable def crate_weight := 250

theorem max_combined_weight (w : ℕ → ℝ) 
  (hw : ∀ i, w i ∈ set.Icc 150 250)
  (hc1 : ∑ i in finset.range 8, w i ≤ 1300)
  (hc2 : ∑ i in finset.range 12, w i ≤ 2100):
  ∑ i in finset.range 35, w i = 8750 :=
by
  have h1 : ∑ i in finset.range 8, 250 = 2000 := sorry
  have h2 : ∑ i in finset.range 12, 250 = 3000 := sorry
  have h3 : ∑ i in finset.range 15, 250 = 3750 := sorry
  have h4 : 2000 + 3000 + 3750 = 8750 := sorry
  exact h4

end max_combined_weight_l168_168432


namespace regular_square_pyramid_side_length_is_5_l168_168181

noncomputable def regular_square_pyramid_side_length 
  (base_edge_length : ℝ) (volume : ℝ) : ℝ :=
  let base_area : ℝ := base_edge_length ^ 2 in
  let height : ℝ := (3 * volume) / base_area in
  let diagonal : ℝ := real.sqrt (2 * base_edge_length ^ 2) in
  let side_edge : ℝ := real.sqrt (height ^ 2 + (diagonal / 2) ^ 2) in
  side_edge

theorem regular_square_pyramid_side_length_is_5 :
  regular_square_pyramid_side_length (4 * real.sqrt 2) 32 = 5 :=
by sorry

end regular_square_pyramid_side_length_is_5_l168_168181


namespace problem_I_problem_II_l168_168194

noncomputable def f (x a : ℝ) : ℝ := (exp x / x) + (exp 1) * log x - a * x
def f_der (x a : ℝ) : ℝ := ((exp x * (x - 1)) / (x ^ 2)) + (x / exp 1) - a

theorem problem_I (a : ℝ) (h : f_der 1 a = 0) : a = exp 1 := by
  sorry

theorem problem_II (a : ℝ) (h : a = exp 1) : ∀ x : ℝ, x > 0 → f x a ≥ 0 := by
  sorry

end problem_I_problem_II_l168_168194


namespace append_digits_divisible_by_all_less_than_10_l168_168797

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168797


namespace smallest_root_of_g_l168_168977

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

-- The main statement: proving the smallest root of g(x) is -sqrt(7/5)
theorem smallest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → x ≤ y := 
sorry

end smallest_root_of_g_l168_168977


namespace smallest_number_of_digits_to_append_l168_168806

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l168_168806


namespace roots_of_polynomial_l168_168119

def polynomial : Polynomial ℝ := X^3 + 2*X^2 - 5*X - 6

theorem roots_of_polynomial :
  (Polynomial.roots polynomial).toFinset = {-1, 2, -3} :=
by
  sorry

end roots_of_polynomial_l168_168119


namespace counterexample_l168_168014

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem counterexample (n : ℕ) (h₁ : n = 33) : ¬ is_prime n ∧ is_prime (n - 2) := 
by {
  -- condition that n = 33
  have h₁ := h₁,
  -- conditions that n is not prime and n-2 is prime
  have h₂ : ¬ is_prime n, sorry,
  have h₃ : is_prime (n - 2), sorry,
  exact ⟨h₂, h₃⟩
}

end counterexample_l168_168014


namespace smallest_solution_l168_168351

theorem smallest_solution (x : ℕ) (h1 : 6 * x ≡ 17 [MOD 31]) (h2 : x ≡ 3 [MOD 7]) : x = 24 := 
by 
  sorry

end smallest_solution_l168_168351


namespace find_eccentricity_l168_168464

-- Definitions and conditions
def hyperbola (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def circle (a x y : ℝ) : Prop :=
  a > 0 ∧ (x^2 + y^2 = a^2)

def parabola (c x y : ℕ) : Prop :=
  c > 0 ∧ (y^2 = 4 * c * x)

def focus (c : ℝ) : ℝ × ℝ :=
  (-c, 0)

def eccentricity (a b c : ℝ) : ℝ :=
  sqrt(1 + b^2/a^2)

-- Theorem
theorem find_eccentricity (a b c : ℝ) (x y : ℝ) :
  hyperbola a b x y →
  circle a x y →
  parabola c x y →
  focus c = (-c, 0) →
  E : ℝ × ℝ →
  E = (x + c, y) →
  (∃ P : ℝ × ℝ, P.1 = 2 * a - c ∧ P.2 = y ∧ E = (P.1 + (-c)) / 2 ∧ E = (P.2 + 0) / 2) →
  eccentricity a b c = (sqrt 5 + 1) / 2 :=
sorry

end find_eccentricity_l168_168464


namespace max_pieces_with_three_cuts_l168_168018

def cake := Type

noncomputable def max_identical_pieces (cuts : ℕ) (max_cuts : ℕ) : ℕ :=
  if cuts = 3 ∧ max_cuts = 3 then 8 else sorry

theorem max_pieces_with_three_cuts : ∀ (c : cake), max_identical_pieces 3 3 = 8 :=
by
  intro c
  sorry

end max_pieces_with_three_cuts_l168_168018


namespace parallelogram_area_l168_168261

theorem parallelogram_area (ABCD : Parallelogram) (A X Y C : Point) 
    (hAXD : right_angle A X D) (hCYB : right_angle C Y B)
    (h_parallelogram : parallelogram_properties A B C D)
    (hX_between_A_Y : between X A Y)
    (hAX : dist A X = 4) (hXY : dist X Y = 3) (hYC : dist Y C = 5)
    : area ABCD = 36 := 
sorry

end parallelogram_area_l168_168261


namespace train_length_is_correct_l168_168053

noncomputable def length_of_train 
  (crossing_time : ℚ)  -- 6 seconds
  (man_speed_kmph : ℚ)  -- 5 kmph
  (train_speed_kmph : ℚ)  -- 114.99 kmph
  (conversion_factor_kmph_to_mps : ℚ) := -- 1 kmph = 1000 meters / 3600 seconds

let relative_speed_kmph := train_speed_kmph + man_speed_kmph in
let relative_speed_mps := relative_speed_kmph * conversion_factor_kmph_to_mps in
let train_length_m := relative_speed_mps * crossing_time in
train_length_m

theorem train_length_is_correct :
  length_of_train 6 5 114.99 (1000 / 3600) = 199.98 :=
by
  sorry

end train_length_is_correct_l168_168053


namespace fair_division_proof_l168_168746

noncomputable def fair_division (A B C: ℝ) (bandit1_perception: ℝ → ℝ) (bandit2_perception: ℝ → ℝ) (bandit3_perception: ℝ → ℝ) : Prop :=
  let parts := [A, B, C] in
  let bandit1_perceived_equal := bandit1_perception A = bandit1_perception B ∧ bandit1_perception B = bandit1_perception C in
  let bandit2_choice := parts.maxBy bandit2_perception in
  let bandit3_choice := parts.maxBy bandit3_perception in
  if bandit2_choice ≠ bandit3_choice then 
    bandit1_perception (parts.remove bandit2_choice).head! ≥ A/3 ∧ 
    bandit2_perception bandit2_choice ≥ B/3 ∧ 
    bandit3_perception bandit3_choice ≥ C/3
  else
    let remaining_part := parts.remove bandit2_choice in
    bandit1_perception remaining_part.head! ≥ A/3 ∨ 
    bandit2_perception bandit2_choice ≥ B/3 ∨
    bandit3_perception bandit3_choice ≥ C/3

theorem fair_division_proof :
  ∀ (A B C: ℝ) (bandit1_perception: ℝ → ℝ) (bandit2_perception: ℝ → ℝ) (bandit3_perception: ℝ → ℝ), 
  fair_division A B C bandit1_perception bandit2_perception bandit3_perception :=
  by sorry

end fair_division_proof_l168_168746


namespace smallest_n_for_sqrt4_floor_l168_168462

theorem smallest_n_for_sqrt4_floor (n : ℕ) :
  ∀ n, 0 < (real.sqrt (real.sqrt n)) - real.floor (real.sqrt (real.sqrt n)) → (real.sqrt (real.sqrt n)) - real.floor (real.sqrt (real.sqrt n)) < 1/2015 → n = 4097 :=
by
  sorry

end smallest_n_for_sqrt4_floor_l168_168462


namespace number_divisible_l168_168788

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168788


namespace not_parallelogram_l168_168676

structure Midpoint (P Q M : Type) :=
(midpoint : M = (P + Q) / 2)

variables (A B C D M N : Type)
variables [Add A] [Add B] [Add C] [Add D] [HasDiv A] [HasDiv C]
variables [Sub A] [Sub B] [HasEq A] [HasEq B] [HasEq C] [HasEq D]

axiom midpoint_AB : Midpoint A B M
axiom midpoint_CD : Midpoint C D N
axiom parallel_BC_AD : ∀ A B C D, C ∥ A
axiom AN_eq_CM : ∀ A N C M, (A - N) = (C - M)

theorem not_parallelogram (A B C D M N : Type) [Add A] [Add B] [Add C] [Add D] [HasDiv A] [HasDiv C] [Sub A] [Sub B] [HasEq A] [HasEq B] [HasEq C] [HasEq D]
    (midpoint_AB : Midpoint A B M) (midpoint_CD : Midpoint C D N)
    (parallel_BC_AD : ∀ A B C D, C ∥ A)
    (AN_eq_CM : ∀ A N C M, (A - N) = (C - M)) :
    ¬ (A ∥ C ∧ B ∥ D) := 
sorry

end not_parallelogram_l168_168676


namespace smallest_number_append_l168_168813

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168813


namespace number_divisible_l168_168785

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168785


namespace tennis_tournament_possible_l168_168613

theorem tennis_tournament_possible (n : ℕ) : 
  (∃ k : ℕ, n = 8 * k + 1) ↔ (∀ i j : ℕ, i < j → j ≤ n → ∃ a b c d : ℕ, {i, j} = {a, b} ∨ {i, j} = {c, d} ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
sorry

end tennis_tournament_possible_l168_168613


namespace volume_of_regular_quadrilateral_pyramid_l168_168513

open Real

variables (b R : ℝ)
def regular_quadrilateral_pyramid (b R : ℝ) := true -- Placeholder definition

theorem volume_of_regular_quadrilateral_pyramid
  (h1: b > 0)
  (h2: R > 0):
  (3:ℝ) ≠ 0 → -- Ensure division by non-zero in the formula
  ∀ b R: ℝ,
  volume_of_regular_quadrilateral_pyramid b R = (b^4) / (6 * R) :=
by
  intro h1 h2 h
  sorry

end volume_of_regular_quadrilateral_pyramid_l168_168513


namespace append_digits_divisible_by_all_less_than_10_l168_168794

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168794


namespace geometric_series_common_ratio_l168_168443

theorem geometric_series_common_ratio
    (a : ℝ) (S : ℝ) (r : ℝ)
    (h_a : a = 512)
    (h_S : S = 3072)
    (h_sum : S = a / (1 - r)) : 
    r = 5 / 6 :=
by 
  rw [h_a] at h_sum
  rw [h_S] at h_sum
  sorry

end geometric_series_common_ratio_l168_168443


namespace find_angle_FYD_l168_168619

variables (A B C D X Y F : Type*) [affine_space A B C D] [affine_space A C] [affine_space C D] [affine_space X F]
variables (parallel_AB : A.parallel B) (parallel_CD : C.parallel D)
variables (angle_AXF : real) (angle_FYD : real)

axiom angle_AXF_125 : angle_AXF = 125

theorem find_angle_FYD :
  parallel_AB → parallel_CD → angle_AXF = 125 → angle_FYD = 180 - angle_AXF :=
begin
  intros hAB hCD hAXF,
  rw hAXF,
  norm_num,
end

end find_angle_FYD_l168_168619


namespace find_ff3_l168_168565

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(x + 1) else 1 - Real.log2 x

theorem find_ff3 : f (f 3) = 4/3 := by
  sorry

end find_ff3_l168_168565


namespace triangle_max_perimeter_has_incenter_l168_168675

-- Definitions of distances given by the conditions
variable (P A B C : Type) [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable [has_dist P A] [has_dist P B] [has_dist P C]
variable (pa : dist P A = 3) (pb : dist P B = 5) (pc : dist P C = 7)

-- Statement asserting P is the incenter for maximizing the perimeter
theorem triangle_max_perimeter_has_incenter :
  (∀ (A B C : Type), dist P A = 3 → dist P B = 5 → dist P C = 7 → is_incenter P A B C) :=
  by
    intros A B C pa pb pc
    sorry

end triangle_max_perimeter_has_incenter_l168_168675


namespace percentage_runs_made_by_running_is_correct_l168_168409

-- Define the total score and number of boundaries and sixes
def total_score : ℕ := 142
def boundaries : ℕ := 12
def sixes : ℕ := 2

-- Calculate runs from boundaries and sixes
def runs_from_boundaries : ℕ := boundaries * 4
def runs_from_sixes : ℕ := sixes * 6
def total_runs_from_boundaries_and_sixes : ℕ := runs_from_boundaries + runs_from_sixes

-- Calculate runs made by running between the wickets
def runs_by_running : ℕ := total_score - total_runs_from_boundaries_and_sixes

-- Calculate percentage of runs made by running
def percentage_runs_by_running : ℚ := (runs_by_running.to_rat / total_score.to_rat) * 100

theorem percentage_runs_made_by_running_is_correct : percentage_runs_by_running ≈ 57.75 :=
by
  sorry

end percentage_runs_made_by_running_is_correct_l168_168409


namespace base_eight_to_base_ten_l168_168354

theorem base_eight_to_base_ten {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 5) (h3 : d3 = 7) :
  d3 * 8^0 + d2 * 8^1 + d1 * 8^2 = 111 := 
by
  sorry

end base_eight_to_base_ten_l168_168354


namespace smallest_digits_to_append_l168_168827

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l168_168827


namespace largest_among_five_l168_168059

noncomputable def A := 2009 ^ 2010
noncomputable def B := 2009 * 2010 ^ 2
noncomputable def C := 2010 ^ 2009
noncomputable def D (a : ℕ) := 3 ^ (3 ^ (a ^ 3))
noncomputable def E := ∑ i in (finset.range 2011).filter (λ n, n % 2 = 0), n ^ 10

theorem largest_among_five : 
  ∀ a : ℕ, D a > A ∧ D a > B ∧ D a > C ∧ D a > E :=
begin
  sorry
end

end largest_among_five_l168_168059


namespace number_is_3034_l168_168921

theorem number_is_3034 (number : ℝ) (h : number - 1002 / 20.04 = 2984) : number = 3034 :=
sorry

end number_is_3034_l168_168921


namespace area_of_circle_with_radius_4_l168_168905

theorem area_of_circle_with_radius_4 :
  ∃ (A : ℝ), A ≈ 50.27 ∧ (A = π * (4 : ℝ) ^ 2) :=
sorry

end area_of_circle_with_radius_4_l168_168905


namespace store_earnings_l168_168428

theorem store_earnings (num_pencils : ℕ) (num_erasers : ℕ) (price_eraser : ℝ) 
  (multiplier : ℝ) (price_pencil : ℝ) (total_earnings : ℝ) :
  num_pencils = 20 →
  price_eraser = 1 →
  num_erasers = num_pencils * 2 →
  price_pencil = (price_eraser * num_erasers) * multiplier →
  multiplier = 2 →
  total_earnings = num_pencils * price_pencil + num_erasers * price_eraser →
  total_earnings = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end store_earnings_l168_168428


namespace total_dots_on_left_faces_l168_168393

-- Define the number of dots on the faces A, B, C, and D
def d_A : ℕ := 3
def d_B : ℕ := 5
def d_C : ℕ := 6
def d_D : ℕ := 5

-- The statement we need to prove
theorem total_dots_on_left_faces : d_A + d_B + d_C + d_D = 19 := by
  sorry

end total_dots_on_left_faces_l168_168393


namespace number_of_people_l168_168707

def avg_weight_increase : ℝ := 2.5
def old_person_weight : ℝ := 45
def new_person_weight : ℝ := 65

theorem number_of_people (n : ℕ) 
  (h1 : avg_weight_increase = 2.5) 
  (h2 : old_person_weight = 45) 
  (h3 : new_person_weight = 65) :
  n = 8 :=
  sorry

end number_of_people_l168_168707


namespace diameter_computation_l168_168461

-- Definitions used in problem conditions
def radius_1 : ℝ := 12
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
def volume_3x := 3 * volume_sphere radius_1

-- Prove the tuple result (diameter in the form of a√(3){b}) 
noncomputable def diameter_in_form : ℝ := 24 * Real.root 3 3

-- Lean statement
theorem diameter_computation :
  diameter_in_form = 24 * Real.root 3 3 ∧ 
  let a := 24 in let b := 3 in a + b = 27 :=
by
  sorry

end diameter_computation_l168_168461


namespace product_of_solutions_l168_168134

theorem product_of_solutions : 
  let S := {x : ℝ | abs x = 3 * (abs x - 2)} in
  ∀ x y ∈ S, x ≠ y → (x * y) = -9 :=
by
  introv hx hy hxy
  have hS : S = {3, -3} := sorry -- Showing that S is exactly the set {3, -3}
  have hx' : x = 3 ∨ x = -3 := by rwa [hS] at hx
  have hy' : y = 3 ∨ y = -3 := by rwa [hS] at hy
  cases hx'; cases hy'; contradiction <|> norm_num

end product_of_solutions_l168_168134


namespace same_type_as_sqrt2_l168_168939

theorem same_type_as_sqrt2: 
  (∃ x, x = sqrt 18 ∧ (∃ y, y = sqrt 2 ∧ y = x / (9 * sqrt 2)))

end same_type_as_sqrt2_l168_168939


namespace sum_of_squares_of_medians_l168_168369

/-- Define the sides of the triangle -/
def side_lengths := (13, 13, 10)

/-- Define the triangle as isosceles with given side lengths -/
def isosceles_triangle (a b c : ℝ) := (a = b) ∧ (a ≠ c)

/-- Define the median lengths calculation -/
noncomputable def median_length (a b c : ℝ) : ℝ :=
  if h : isosceles_triangle a b c then
    let AD := Real.sqrt (a ^ 2 - (c / 2) ^ 2) in
    let BE_CF := Real.sqrt ((a / 2) ^ 2 + (3 / 4) * c ^ 2) in
    AD^2 + BE_CF^2 + BE_CF^2
  else
    0

/-- The sum of the squares of the lengths of the medians for the given triangle is -/
theorem sum_of_squares_of_medians : median_length 13 13 10 = 447.5 := by
  sorry

end sum_of_squares_of_medians_l168_168369


namespace rug_overlap_area_l168_168748

theorem rug_overlap_area (A S S2 S3 : ℝ) 
  (hA : A = 200)
  (hS : S = 138)
  (hS2 : S2 = 24)
  (h1 : ∃ (S1 : ℝ), S1 + S2 + S3 = S)
  (h2 : ∃ (S1 : ℝ), S1 + 2 * S2 + 3 * S3 = A) : S3 = 19 :=
by
  sorry

end rug_overlap_area_l168_168748


namespace slope_ratio_equiv_l168_168573

noncomputable def parabola : Type := { p : ℝ × ℝ // p.snd ^ 2 = 4 * p.fst }

def focus : (ℝ × ℝ) := (1, 0)

def point_P : (ℝ × ℝ) := (2, 0)

def intersect_parabola (l : (ℝ × ℝ) → Prop) : Prop :=
  ∃ A B : parabola, l (A.1) ∧ l (B.1)

def line_through_focus (A : parabola) : (ℝ × ℝ) → Prop := 
  λ P, ∃ k : ℝ, P.snd = k * (P.fst - focus.fst)

def intersect_again (A : parabola) : parabola :=
  -- Assume we can always find another intersection point
  sorry

def slope (P Q : (ℝ × ℝ)) : ℝ := 
  (Q.snd - P.snd) / (Q.fst - P.fst)

def slope_lines (line1 line2 : (ℝ × ℝ) → Prop) (x1 x2 y1 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

theorem slope_ratio_equiv (A B C D : parabola) (k1 k2 : ℝ) :
  slope_lines (line_through_focus A) (line_through_focus B) A.1.fst B.1.fst A.1.snd B.1.snd = k1 →
  slope_lines (line_through_focus C) (line_through_focus D) C.1.fst D.1.fst C.1.snd D.1.snd = k2 →
  k2 = 2 * k1 →
  k1 / k2 = 1 / 2 :=
by
  intros h1 h2 h3
  rw [h3]
  field_simp
  sorry

end slope_ratio_equiv_l168_168573


namespace AlyssaBottleCaps_l168_168642

def bottleCapsKatherine := 34
def bottleCapsGivenAway (bottleCaps: ℕ) := bottleCaps / 2
def bottleCapsLost (bottleCaps: ℕ) := bottleCaps - 8

theorem AlyssaBottleCaps : bottleCapsLost (bottleCapsGivenAway bottleCapsKatherine) = 9 := 
  by 
  sorry

end AlyssaBottleCaps_l168_168642


namespace ant_opposite_vertex_probability_l168_168064

noncomputable def probability_ant_at_opposite_vertex (start_vertex end_vertex : ℕ) : ℚ :=
  if start_vertex = end_vertex then 0 else 1 / 128

theorem ant_opposite_vertex_probability :
  ∀ (start_vertex end_vertex : ℕ), 
    start_vertex ≠ end_vertex →
    -- conditions
    (∃ (vertices edges : Finset ℕ), 
      vertices.card = 6 ∧ edges.card = 12 ∧
      (∀ (v : ℕ), v ∈ vertices → ∃ e ∈ edges, e = v ∧ -- edge connects vertex here for simplicity, could define precise conditions
      (∀ (v : ℕ), v ∈ vertices →  ∃ edge_choice_set : Finset ℕ, edge_choice_set.card = 4 ∧ 
        -- Define the independence and move choices logic here, essentially proving existence of 4 choices from each vertex
        True)) →
    -- Proofing part
    probability_ant_at_opposite_vertex start_vertex end_vertex = 1 / 128 :=
by
  intros start_vertex end_vertex h_ne vertices edges h_conditions
  simp only [probability_ant_at_opposite_vertex, h_ne] -- placeholder for logical derivations
  -- skip the detailed combinatorial proof steps which are expected to be context-specific to the ant-path conditions
  sorry

end ant_opposite_vertex_probability_l168_168064


namespace exponential_function_a_eq_2_l168_168370

theorem exponential_function_a_eq_2
  (a : ℝ) (h1 : a ≠ 1) (h2 : 0 < a) (h3 : ∀ x : ℝ, ∃ b : ℝ, b > 0 ∧ b ≠ 1 ∧ f(x) = b^x) 
  (h4 : ∀ x : ℝ, f(x) = (a^2 - 3a + 3) * a^x) : a = 2 :=
by
  sorry

end exponential_function_a_eq_2_l168_168370


namespace sector_to_circle_inscribed_ratio_l168_168975

noncomputable def sector_to_circle_ratio (α R r : ℝ) : ℝ :=
  let S_sector := (1 / 2) * R^2 * α
  let S_circle := π * r^2
  S_sector / S_circle

theorem sector_to_circle_inscribed_ratio (α R : ℝ)
  (h_r : r = R * sin (α / 2) / (1 + sin (α / 2))) :
  sector_to_circle_ratio α R r = 2 * α * (cos (π / 4 - α / 4))^2 / (π * (sin (α / 2))^2) :=
by
  sorry

end sector_to_circle_inscribed_ratio_l168_168975


namespace prism_volume_l168_168624

-- Define the right triangular prism conditions

variables (AB BC AC : ℝ)
variable (S : ℝ)
variable (volume : ℝ)

-- Given conditions
axiom AB_eq_2 : AB = 2
axiom BC_eq_2 : BC = 2
axiom AC_eq_2sqrt3 : AC = 2 * Real.sqrt 3
axiom circumscribed_sphere_surface_area : S = 32 * Real.pi

-- Statement to prove
theorem prism_volume : volume = 4 * Real.sqrt 3 :=
sorry

end prism_volume_l168_168624


namespace new_average_of_adjusted_sequence_l168_168887

open Nat

theorem new_average_of_adjusted_sequence (a : ℕ → ℤ) (n : ℕ) (h_avg : (finset.range 15).sum a / 15 = 25)
  (h_adjust : ∀ i < 7, a i - 2 * i - 14 + (a (i + 1) + 2 * i + 12) = 0) :
  ((finset.range 15).sum (λ i, if i % 2 = 0 then a i - (14 - 2 * (i / 2)) else a i + (12 - 2 * (i / 2)))) / 15 = 25 := 
sorry

end new_average_of_adjusted_sequence_l168_168887


namespace find_alpha_plus_beta_l168_168588

open Real

theorem find_alpha_plus_beta 
  (α β : ℝ)
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin β = sqrt 10 / 10)
  (h3 : π / 2 < α ∧ α < π)
  (h4 : π / 2 < β ∧ β < π) :
  α + β = 7 * π / 4 :=
sorry

end find_alpha_plus_beta_l168_168588


namespace repeated_mul_eq_pow_l168_168307

-- Define the repeated multiplication of 2, n times
def repeated_mul (n : ℕ) : ℕ :=
  (List.replicate n 2).prod

-- State the theorem to prove
theorem repeated_mul_eq_pow (n : ℕ) : repeated_mul n = 2 ^ n :=
by
  sorry

end repeated_mul_eq_pow_l168_168307


namespace smallest_number_append_l168_168814

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168814


namespace smallest_append_digits_l168_168833

theorem smallest_append_digits (a b : ℕ) (h : b = 2520) (n : ℕ) (hn : n < 10) :
  ∃ x, ∀ y, (2014 + x) % b = 0 ∧ (2014 + x) = y * b :=
begin
  use 506,
  intros y,
  split,
  { -- Proof that (2014 + 506) % 2520 = 0
    sorry },
  { -- Proof that (2014 + 506) = y * 2520 for some y
    sorry }
end

end smallest_append_digits_l168_168833


namespace bag_sums_l168_168069

theorem bag_sums :
  let bag_A := {1, 3, 5, 7}
  let bag_B := {2, 4, 6, 8}
  ∃ (sums : Finset ℕ), 
    (∀ a ∈ bag_A, ∀ b ∈ bag_B, ∃ s ∈ sums, s = a + b) ∧ 
    (sums.card = 7) :=
by
  let bag_A := {1, 3, 5, 7}
  let bag_B := {2, 4, 6, 8}
  use {3, 5, 7, 9, 11, 13, 15}
  split
  sorry

end bag_sums_l168_168069


namespace product_of_solutions_l168_168139

noncomputable def absolute_value (x : ℝ) : ℝ :=
  if x >= 0 then x else -x

theorem product_of_solutions : (∏ x in ({3, -3} : finset ℝ), x) = -9 :=
by 
  have hsol : {x : ℝ | absolute_value x = 3 * (absolute_value x - 2)} = {3, -3},
  { sorry },  -- Proof that the solutions set is exactly {3, -3}
  rw finset.prod_eq_mul,
  norm_num
  sorry

end product_of_solutions_l168_168139


namespace employees_percentage_l168_168334

theorem employees_percentage (y : ℕ) :
  let T := (4*y + 6*y + 5*y + 2*y + 3*y + 1*y + 1*y + 2*y + 2*y + 1*y + 1*y) in
  let E := (2*y + 2*y + 1*y + 1*y) in
  (E * 100) / T = 21.43 :=
by {
  -- The proof goes here.
  sorry
}

end employees_percentage_l168_168334


namespace fountain_pen_price_l168_168398

theorem fountain_pen_price
  (n_fpens : ℕ) (n_mpens : ℕ) (total_cost : ℕ) (avg_cost_mpens : ℝ)
  (hpens : n_fpens = 450) (mpens : n_mpens = 3750) 
  (htotal : total_cost = 11250) (havg_mpens : avg_cost_mpens = 2.25) : 
  (total_cost - n_mpens * avg_cost_mpens) / n_fpens = 6.25 :=
by
  sorry

end fountain_pen_price_l168_168398


namespace smallest_digits_to_append_l168_168781

theorem smallest_digits_to_append : ∃ d ∈ (finset.range 10).filter (λ n : ℕ, n ≥ 1), 
  (10 ^ d * 2014 % Nat.lcm (finset.range 1 10) = 0 ∧ (∀ d' ∈ (finset.range d), 10 ^ d' * 2014 % Nat.lcm (finset.range 1 10) ≠ 0) :=
begin
  sorry
end

end smallest_digits_to_append_l168_168781


namespace sum_of_perfectly_paintable_numbers_l168_168641

def perfectlyPaintable (j k l : ℕ) : Prop :=
  ∀ n : ℕ, (n % j = 1 ∨ n % k = 2 ∨ n % l = 3) ∧
  ∀ m n : ℕ, (m ≠ n) → ((m % j = 1 ∧ n % j ≠ 1) 
                          ∧ (m % k = 2 ∧ n % k ≠ 2) 
                          ∧ (m % l = 3 ∧ n % l ≠ 3))

theorem sum_of_perfectly_paintable_numbers : ℕ :=
  let perfectlyPaintableValues := [100 * 3 + 10 * 3 + 3, 100 * 4 + 10 * 4 + 4] in
  perfectlyPaintableValues.sum = 777

#eval sum_of_perfectly_paintable_numbers

end sum_of_perfectly_paintable_numbers_l168_168641


namespace penguins_count_l168_168335

theorem penguins_count (fish_total penguins_fed penguins_require : ℕ) (h1 : fish_total = 68) (h2 : penguins_fed = 19) (h3 : penguins_require = 17) : penguins_fed + penguins_require = 36 :=
by
  sorry

end penguins_count_l168_168335


namespace union_cardinality_is_4_l168_168663

def setA : Set ℕ := {1, 2, 3}
def setB : Set ℕ := {0, 1, 2}

theorem union_cardinality_is_4 : (setA ∪ setB).card = 4 := by
  sorry

end union_cardinality_is_4_l168_168663


namespace milk_price_same_after_reductions_l168_168377

theorem milk_price_same_after_reductions (x : ℝ) (h1 : 0 < x) :
  (x - 0.4 * x) = ((x - 0.2 * x) - 0.25 * (x - 0.2 * x)) :=
by
  sorry

end milk_price_same_after_reductions_l168_168377


namespace model_distance_comparison_l168_168389

theorem model_distance_comparison (m h c x y z : ℝ) (hm : 0 < m) (hh : 0 < h) (hc : 0 < c) (hz : 0 < z) (hx : 0 < x) (hy : 0 < y)
    (h_eq : (x - c) * z = (y - c) * (z + m) + h) :
    (if h > c * m then (x * z > y * (z + m))
     else if h < c * m then (x * z < y * (z + m))
     else (h = c * m → x * z = y * (z + m))) :=
by
  sorry

end model_distance_comparison_l168_168389


namespace f_eval_l168_168562

def f : ℝ → ℝ 
| x => if x > 0 then log x / log 9
       else (4 ^ (-x)) + 3 / 2

theorem f_eval : f 27 + f (- (Real.log 3 / Real.log 4)) = 6 :=
by
  sorry

end f_eval_l168_168562


namespace area_of_quadrilateral_l168_168532

-- Defining a quadrilateral and points on its sides.
variables {A B C D M N K L : Type} [affine_space A] [affine_space B] 
          [affine_space C] [affine_space D] [affine_space M] 
          [affine_space N] [affine_space K] [affine_space L] 

-- Conditions
variables (AM MB CN ND : ℝ)
variable (r : ℝ)
variable (h1 h2 h : ℝ)
variable (AB CD : ℝ)

variables (p h1' h2' : ℝ)

axiom AM_MB_ratio : r = AM / AB
axiom CN_ND_ratio : r = CN / CD

axiom height_expression : h = p * h2' + (1 - p) * h1'

-- Areas
variable (S_KMLN S_ADK S_BCL : ℝ)

axiom area_expression : S_KMLN = S_ADK + S_BCL

-- Statement
theorem area_of_quadrilateral (AM_MB_ratio: AM / AB = CN / CD)
(height_expression: h = r * h2' + (1 - r) * h1')
(area: S_KMLN = S_ADK + S_BCL) : 
S_KMLN = S_ADK + S_BCL := 
sorry


end area_of_quadrilateral_l168_168532


namespace length_XY_l168_168625

variables (P Q R S T X Y : Point)
-- The trapezoid condition and parallel sides
variable (h1: IsTrapezoid P Q R S)
variables (h2: parallel QR PS : Segment)
-- Length conditions
variable (h3: QR.length = 800)
variable (h4: PS.length = 1600)
-- Angle conditions
variable (h5: angle P = 45)
variable (h6: angle S = 45)
-- Midpoint conditions
variable (h7: X = midpoint QR)
variable (h8: Y = midpoint PS)

-- The theorem we're trying to prove
theorem length_XY : XY.length = 400 := by
  sorry

end length_XY_l168_168625


namespace count_prime_boring_lt_10000_l168_168485

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_boring (n : ℕ) : Prop := 
  let digits := n.digits 10
  match digits with
  | [] => false
  | (d::ds) => ds.all (fun x => x = d)

theorem count_prime_boring_lt_10000 : 
  ∃! n, is_prime n ∧ is_boring n ∧ n < 10000 := 
by 
  sorry

end count_prime_boring_lt_10000_l168_168485


namespace sum_of_interior_angles_n_plus_4_l168_168709

theorem sum_of_interior_angles_n_plus_4 (n : ℕ) (h : 180 * (n - 2) = 3240) :
  180 * ((n + 4) - 2) = 3960 :=
by
  have h1 : n - 2 = 18 := by
    calc
      n - 2 = 3240 / 180 := eq.symm (Int.ediv_eq_of_eq_mul_left (by norm_num) h)
           ... = 18       := by norm_num
  have h2 : n = 20 := by linarith
  sorry

end sum_of_interior_angles_n_plus_4_l168_168709


namespace find_length_of_BC_l168_168385

-- Define the points of the triangle and relevant segments
variables {A B C D E : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (AB AC BC DE : ℝ)
variables (Area_ADE Area_DECB : ℝ)

-- Conditions provided in the problem
def condition_1 : Prop := DE = 13
def condition_2 : Prop := BC = 169 / 12
def condition_3 : Prop := Area_ADE / Area_DECB = 144 / 25

-- Question to prove
theorem find_length_of_BC :
  DE = 13 ∧ (Area_ADE / Area_DECB = 144 / 25) ∧ (DE / (169 / 12) ^ 2 = 144 / 169) → BC = 14.08 :=
by sorry

end find_length_of_BC_l168_168385


namespace triangle_no_tangent_circles_l168_168170

theorem triangle_no_tangent_circles (ABC : Triangle) :
  (∃ A B C : ℝ, ∠A < 38.95 ∧ ∠B < 38.95 ∧ ∠C < 38.95 → 
        ¬(∃ (k₁ k₂: Circle),
              (k₁.is_tangent_to $ ABC.side_A ∧ k₁.is_tangent_to $ ABC.side_B) ∧ 
              (k₂.is_tangent_to $ ABC.side_B ∧ k₂.is_tangent_to $ ABC.side_C) ∧ 
              (tangent k₁ k₂) ∧ (radius k₁ / radius k₂ = 1/2))) :=
sorry

end triangle_no_tangent_circles_l168_168170


namespace share_of_B_is_1400_l168_168055

-- Definitions based on conditions
variables (A B C : ℝ) -- Assuming investments as real numbers
variable total_profit : ℝ

-- Conditions
axiom A_investment : A = 3 * B
axiom B_investment : B = (2 / 3) * C
axiom total_profit_is : total_profit = 7700

-- The statement to prove
theorem share_of_B_is_1400 :
  let total_investment := A + B + C in
  let B_share := (B / total_investment) * total_profit in
  B_share = 1400 :=
by
  -- Proof is not required
  sorry

end share_of_B_is_1400_l168_168055


namespace calc_f_at_3_l168_168522

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem calc_f_at_3 : f 3 = 328 := 
sorry

end calc_f_at_3_l168_168522


namespace textbook_profit_l168_168430

theorem textbook_profit (cost_price selling_price : ℕ) (h1 : cost_price = 44) (h2 : selling_price = 55) :
  (selling_price - cost_price) = 11 := by
  sorry

end textbook_profit_l168_168430


namespace prove_math_proof_problem_l168_168065

variable (A B C D M N : Point)
variable (k : Real)

-- Assuming the necessary geometric definitions and properties hold
-- isosceles_trapezoid is a placeholder for the actual geometric condition in Lean
-- circumscribes means the circle touches all sides of the trapezoid
-- intersect_at means line AM intersects the circle at N and touches BC at M

def math_proof_problem : Prop :=
  isosceles_trapezoid A B C D ∧
  circumscribes (circle A B C D) ∧
  intersects (line A M) (circle A B C D) M N ∧
  (MN : Real) = distance M N ∧
  (AN : Real) = distance A N ∧
  MN / AN = k →
  AB / CD = (8 - k) / k

theorem prove_math_proof_problem : math_proof_problem A B C D M N k := sorry

end prove_math_proof_problem_l168_168065


namespace second_class_students_l168_168310

-- Define the conditions
variables (x : ℕ)
variable (sum_marks_first_class : ℕ := 35 * 40)
variable (sum_marks_second_class : ℕ := x * 60)
variable (total_students : ℕ := 35 + x)
variable (total_marks_all_students : ℕ := total_students * 5125 / 100)

-- The theorem to prove
theorem second_class_students : 
  1400 + (x * 60) = (35 + x) * 5125 / 100 →
  x = 45 :=
by
  sorry

end second_class_students_l168_168310


namespace number_divisible_l168_168783

-- Define the given number
def base_number : ℕ := 2014

-- Define the range of natural numbers
def natural_numbers_below_10 := {n // n < 10 ∧ n > 0}

-- Helper function to calculate LCM
def lcm (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

-- Calculate the LCM of numbers from 1 to 9
def lcm_1_to_9 : ℕ := (List.foldl lcm 1 [2,3,4,5,6,7,8,9])

-- Define the resulting number by appending digits to 2014
def resulting_number : ℕ := 2014506

-- Proof that the resulting number is divisible by the LCM of numbers from 1 to 9
theorem number_divisible : resulting_number % lcm_1_to_9 = 0 :=
sorry

end number_divisible_l168_168783


namespace sin_identity_l168_168538

theorem sin_identity (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) :
  Real.sin ((3 * π / 4) - α) = 3 / 5 :=
by
  sorry

end sin_identity_l168_168538


namespace max_cos_a_value_l168_168657

theorem max_cos_a_value (a b : ℝ) (h : cos (a - b) = cos a - cos b) : 
  (cos a) ≤ sqrt ((3 + sqrt 5) / 2) :=
by sorry

end max_cos_a_value_l168_168657


namespace number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l168_168765

/- Definitions for each number's expression using five eights -/
def number1 : Int := (8 / 8) ^ (8 / 8) * (8 / 8)
def number2 : Int := 8 / 8 + 8 / 8
def number3 : Int := (8 + 8 + 8) / 8
def number4 : Int := 8 / 8 + 8 / 8 + 8 / 8 + 8 / 8
def number5 : Int := (8 * 8 - 8) / 8 + 8 / 8

/- Theorem statements to be proven -/
theorem number1_is_1 : number1 = 1 := by
  sorry

theorem number2_is_2 : number2 = 2 := by
  sorry

theorem number3_is_3 : number3 = 3 := by
  sorry

theorem number4_is_4 : number4 = 4 := by
  sorry

theorem number5_is_5 : number5 = 5 := by
  sorry

end number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l168_168765


namespace value_of_n_l168_168888

def is_3_digit_integer (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

def not_divisible_by (n k : ℕ) : Prop := ¬ (k ∣ n)

def least_common_multiple (a b c : ℕ) : Prop := Nat.lcm a b = c

theorem value_of_n (d n : ℕ) (h1 : least_common_multiple d n 690) 
  (h2 : not_divisible_by n 3) (h3 : not_divisible_by d 2) (h4 : is_3_digit_integer n) : n = 230 :=
by
  sorry

end value_of_n_l168_168888


namespace max_value_2a_l168_168183

-- Define the function f(x)
def f (a x : ℝ) : ℝ := -x^2 - 2*a*x + 1

-- The theorem statement
theorem max_value_2a (a : ℝ) (h : a > 1) : 
  ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ ∀ y ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ f a y :=
begin
  use -1,
  split,
  { norm_num, },
  { intro y,
    intro hy,
    have : y = -1 ∨ y ∈ Icc (-1) 1 ∧ (-1 < y),
    { refine or_iff_not_imp_right.mpr (λ hy1, _),
      rw [Set.mem_Icc] at hy1 ⊢,
      linarith, },
    cases this,
    { simp [this] },
    { have : f a (-1) = 2 * a, 
      { unfold f, ring },
      rw this,
      suffices : ∀ y ∈ Icc (-1) 1, f a y ≤ 2 * a,
      { exact this y hy },
      intros y hy,
      unfold f,
      linarith }
  }
end

end max_value_2a_l168_168183


namespace angle_PTV_60_l168_168621

variables (m n TV TPV PTV : ℝ)

-- We state the conditions
axiom parallel_lines : m = n
axiom angle_TPV : TPV = 150
axiom angle_TVP_perpendicular : TV = 90

-- The goal statement to prove
theorem angle_PTV_60 : PTV = 60 :=
by
  sorry

end angle_PTV_60_l168_168621


namespace problem_solution_l168_168340

theorem problem_solution :
  ∃ n : ℕ, 50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ n = 58 :=
by
  -- Lean code to prove the statement
  sorry

end problem_solution_l168_168340


namespace exists_disjoint_subset_l168_168911

open Set

variable {α : Type*} [MetricSpace α]

def circle (c : α) (r : ℝ) : Set α :=
  {p | dist p c = r}

-- Given finite set of unit circles
variable (U : Set (Set α)) (S : ℝ)
hypothesis union_circles : Union U

-- Given condition: area of their union is S
noncomputable def area (s : Set α) : ℝ := sorry

axiom area_union_circles : area union_circles = S

theorem exists_disjoint_subset (U : Set (Set α)) (S : ℝ)
  (h1 : ∀ c ∈ U, ∃ r, circle c 1)
  (h2 : area (Union U) = S) :
  ∃ T ⊆ U, PairwiseDisjoint T ∧ area (Union T) > (2 * S / 9) :=
begin
  sorry
end

end exists_disjoint_subset_l168_168911


namespace triangle_area_ratio_l168_168632

noncomputable def triangleAreasRatio (P Q R S : Point) (PQ PR QR : ℝ) (angle_bisector : Line)
  (hPQ : dist P Q = 20) (hPR : dist P R = 30) (hQR : dist Q R = 26)
  (hBisector : angle_bisector = angleBisector P Q R S) : ℝ := 
    ((area P Q S) / (area P R S))

theorem triangle_area_ratio (P Q R S : Point) (PQ PR QR : ℝ) (angle_bisector : Line)
  (hPQ : dist P Q = 20) (hPR : dist P R = 30) (hQR : dist Q R = 26)
  (hBisector : angle_bisector = angleBisector P Q R S) : 
  triangleAreasRatio P Q R S PQ PR QR angle_bisector hPQ hPR hQR hBisector = 2 / 3 := 
sorry

end triangle_area_ratio_l168_168632


namespace integer_points_on_circle_l168_168885

theorem integer_points_on_circle : 
  ∃ (k : ℕ), k = 12 ∧ (∀ (x y : ℤ), x^2 + y^2 = 25 → (x, y) = (0, 5) ∨ (x, y) = (0, -5) ∨ (x, y) = (3, 4) ∨ (x, y) = (3, -4) ∨ (x, y) = (-3, 4) ∨ (x, y) = (-3, -4) ∨ (x, y) = (4, 3) ∨ (x, y) = (4, -3) ∨ (x, y) = (-4, 3) ∨ (x, y) = (-4, -3) ∨ (x, y) = (5, 0) ∨ (x, y) = (-5, 0)) :=
begin
  sorry
end

end integer_points_on_circle_l168_168885


namespace geometric_sequence_property_a_seq_formula_terms_sum_correct_l168_168652

noncomputable def a_seq : ℕ → ℝ
| 0     := 3 / 7
| (n+1) := 3 * (a_seq n) / (4 * (a_seq n) + 1)

theorem geometric_sequence_property (n : ℕ) :
  (1 / a_seq (n + 1) - 2 = (1 / 3) * (1 / a_seq n - 2)) :=
sorry

theorem a_seq_formula (n : ℕ) :
  a_seq (n + 1) = 3^n / (2 * 3^n + 1) :=
sorry

noncomputable def terms_sum (n : ℕ) : ℝ :=
  ∑ k in range n, k / a_seq k

theorem terms_sum_correct (n : ℕ) :
  terms_sum n = (3 / 4) - (3 + 2 * n) / (4 * 3^n) + n^2 + n :=
sorry

end geometric_sequence_property_a_seq_formula_terms_sum_correct_l168_168652


namespace garden_width_l168_168484

theorem garden_width (L P : ℝ) (hL : L = 1.2) (hP : P = 8.4) :
  ∃ W : ℝ, W = 3.0 ∧ P = 2 * (L + W) := 
by
  exists 3.0
  split
  { rfl }
  { rw [←hL, ←hP] 
    norm_num
    sorry }

end garden_width_l168_168484


namespace smallest_number_append_l168_168812

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l168_168812


namespace append_digits_divisible_by_all_less_than_10_l168_168798

-- Defining the conditions and functions needed
def LCM_of_1_to_9 : ℕ := lcm (lcm 1 2) (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9))))))

theorem append_digits_divisible_by_all_less_than_10 :
  ∃ d : ℕ, (2014 * 10 ^ (nat.log10 d + 1) + d) % LCM_of_1_to_9 = 0 ∧ nat.log10 d + 1 < 10 := 
by {
  have h_lcm : LCM_of_1_to_9 = 2520 := by sorry, -- Computing LCM of numbers 1 to 9 as condition
  sorry
}

end append_digits_divisible_by_all_less_than_10_l168_168798


namespace smallest_digits_to_append_l168_168865

theorem smallest_digits_to_append (n : ℕ) (d : ℕ) (m : ℕ) :
    (∀ m, 0 ≤ d ∧ d < 10^m ∧ m ≥ 4 → ∃ k, 2014 * 10^m + d + k * 10^m = 0 [MOD 2520]) := 
sorry

end smallest_digits_to_append_l168_168865


namespace B_investment_l168_168415

theorem B_investment (B : ℝ) : 
  let A_investment := 150 * 12 in 
  let B_investment := B * 6 in
  let total_investment := A_investment + B_investment in
  let A_share := (A_investment / total_investment) * 100 in
  let total_profit := 100 in
  let A_profit := 60 in
  A_share = A_profit → B = 200 :=
by
  sorry

end B_investment_l168_168415


namespace max_value_of_expression_l168_168647

theorem max_value_of_expression (a : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) (b : ℝ := 1) (c : ℝ := 0) : 
  ( √(a * b * c) + √((1 - a) * (1 - b) * (1 - c)) ) ≤ 0 :=
by 
  sorry

end max_value_of_expression_l168_168647


namespace sum_WY_eq_3_l168_168103

theorem sum_WY_eq_3 (W X Y Z : ℕ) (hn : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z)
  (hW : W ∈ {1, 2, 3, 4}) (hX : X ∈ {1, 2, 3, 4}) (hY : Y ∈ {1, 2, 3, 4}) (hZ : Z ∈ {1, 2, 3, 4})
  (heq : (W : ℚ) / X + (Y : ℚ) / Z = 1) : W + Y = 3 :=
sorry

end sum_WY_eq_3_l168_168103


namespace solve_cookies_given_to_Tim_l168_168001

def cookies_given_to_Tim (T : ℕ) : Prop :=
  let total_cookies := 256
  let cookies_to_Mike := 23
  let cookies_in_fridge := 188
  let cookies_to_Anna := 2 * T
  T + cookies_to_Mike + cookies_in_fridge + cookies_to_Anna = total_cookies

theorem solve_cookies_given_to_Tim : ∃ T : ℕ, cookies_given_to_Tim T ∧ T = 15 :=
by { existsi 15, sorry }

end solve_cookies_given_to_Tim_l168_168001


namespace interval_monotonicity_sin_2alpha_l168_168195

section

-- Conditions
variable {ω : ℝ} (hω : ω > 0)
              (symmetry_dist : ℝ := π / 2)
              (ω_value : ω = 2)

-- Given function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (ω * x)

-- Ⅰ: Intervals of monotonicity for f(x+π/6) in [-π/6, 2π/3]
theorem interval_monotonicity (x : ℝ) (h1 : x ∈ [-π/6, 2*π/3]) :
  (∀ x ∈ [-π/6, π/3], f(x + π/6) is_decreasing_on) ∧
  (∀ x ∈ [π/3, 2*π/3], f(x + π/6) is_increasing_on) :=
  sorry

-- Ⅱ: Given conditions for alpha and finding sin 2α
theorem sin_2alpha (α : ℝ) (hα1 : α ∈ (5*π/12, π/2))
                    (hα2 : f(α + π/3) = 1/3) :
  Real.sin (2 * α) = (2 * Real.sqrt 2 - Real.sqrt 3) / 6 :=
  sorry

end

end interval_monotonicity_sin_2alpha_l168_168195


namespace number_of_correct_statements_l168_168240

def I : Set (ℝ × ℝ) → Prop := 
  λ l, ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ ∀ x y : ℝ, ((a * x) + (b * y) + c = 0) ↔ (x, y) ∈ l

def M : Set (ℝ × ℝ) → Prop := 
  λ l, ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ ∃! (x y : ℤ), ((a * (x:ℝ)) + (b * (y:ℝ)) + c = 0) 

def N : Set (ℝ × ℝ) → Prop := 
  λ l, ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ ∀ (x y : ℤ), ((a * (x:ℝ)) + (b * (y:ℝ)) + c ≠ 0)

def P : Set (ℝ × ℝ) → Prop := 
  λ l, ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ (∀ (x y : ℤ), ((a * x) + (b * y) + c = 0) ↔ (x,y) ∈ l) 
    ∧ ∃ x y : ℤ, (x, y) ∈ l

theorem number_of_correct_statements : 4 = (
  if (M ∪ N ∪ P = I 
    ∧ (∃ l, N l) 
    ∧ (∃ l, M l) 
    ∧ (∃ l, P l)
  ) then 4 else 0
) :=
sorry

end number_of_correct_statements_l168_168240


namespace unique_assignment_l168_168582

-- Define the chessboard.
def chessboard := fin 8 × fin 8  -- 8x8 chessboard

-- Define a function that assigns the numbers 1 to 64 to the chessboard cells.
def assignment := chessboard → ℕ

-- Conditions are the sums of the numbers in each 1x2 rectangle.
def valid_rectangles (f: assignment) : Prop :=
  ∀ (x y: fin 8), if y.val = 7 then true else
    let r1: ℕ := f (x, y);
    let r2: ℕ := f (x, y + 1) in
    r1 + r2 ≤ 64 ∧ r1 + r2 > 1

-- Condition that 1 and 64 lie on the same diagonal.
def on_same_diagonal (f: assignment) : Prop :=
  ∃ d: ℕ, d < 16 ∧ (∃ dx: fin 8, dx.val = d ∧ (f (dx, dx) = 1 ∧ f (7 - dx, 7 - dx) = 64))

-- Define the main theorem: Given the conditions, Lyosha can determine the chessboard uniquely.
theorem unique_assignment (f: assignment) :
  valid_rectangles f ∧ on_same_diagonal f →
  ∃! f': assignment, valid_rectangles f' ∧ on_same_diagonal f' :=
sorry

end unique_assignment_l168_168582
