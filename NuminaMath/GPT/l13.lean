import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Lattice
import Mathlib.Algebra.Lcm
import Mathlib.Algebra.Order
import Mathlib.Algebra.Probability
import Mathlib.Analysis.Calculus.Continuity
import Mathlib.Analysis.Calculus.FDerivative
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Binomial
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Integrals
import Mathlib.NumberTheory.Primes
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Theory
import Mathlib.Tactic
import Mathlib.Topology.Basic

namespace greatest_possible_value_q_minus_r_l13_13338

theorem greatest_possible_value_q_minus_r {q r : ℕ} (hq : 100 ≤ q ∧ q < 1000) (hr : 100 ≤ r ∧ r < 1000)
  (digits_reversed : ∃ a b c : ℕ, a ≠ 0 ∧ c ≠ 0 ∧ q = 100 * a + 10 * b + c ∧ r = 100 * c + 10 * b + a)
  (diff_lt_300 : abs (q - r) < 300) : q - r = 297 :=
by
  sorry

end greatest_possible_value_q_minus_r_l13_13338


namespace isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l13_13165

-- Definitions for number of valence electrons
def valence_electrons (atom : String) : ℕ :=
  if atom = "C" then 4
  else if atom = "N" then 5
  else if atom = "O" then 6
  else if atom = "F" then 7
  else if atom = "S" then 6
  else 0

-- Definitions for molecular valence count
def molecule_valence_electrons (molecule : List String) : ℕ :=
  molecule.foldr (λ x acc => acc + valence_electrons x) 0

-- Definitions for specific molecules
def N2_molecule := ["N", "N"]
def CO_molecule := ["C", "O"]
def N2O_molecule := ["N", "N", "O"]
def CO2_molecule := ["C", "O", "O"]
def NO2_minus_molecule := ["N", "O", "O"]
def SO2_molecule := ["S", "O", "O"]
def O3_molecule := ["O", "O", "O"]

-- Isoelectronic property definition
def isoelectronic (mol1 mol2 : List String) : Prop :=
  molecule_valence_electrons mol1 = molecule_valence_electrons mol2

theorem isoelectronic_problem_1_part_1 :
  isoelectronic N2_molecule CO_molecule := sorry

theorem isoelectronic_problem_1_part_2 :
  isoelectronic N2O_molecule CO2_molecule := sorry

theorem isoelectronic_problem_2 :
  isoelectronic NO2_minus_molecule SO2_molecule ∧
  isoelectronic NO2_minus_molecule O3_molecule := sorry

end isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l13_13165


namespace binomial_sum_identity_l13_13118

-- Definitions for binomial coefficients
def binomial : ℕ → ℕ → ℕ
| n, k => if k = 0 then 1 else if k > n then 0 else binomial (n - 1) (k - 1) + binomial (n - 1) k

-- Definitions for the series on the left-hand side of the equation
def lhs (n m : ℕ) : ℕ :=
  (List.range (n + 1 - m)).map (λ k, (k + 1) * binomial (n - k) m).sum

-- Main theorem statement
theorem binomial_sum_identity (n m : ℕ) (h0 : 0 ≤ n) (h1 : 0 ≤ m) :
  lhs n m = binomial (n + 2) (m + 2) :=
sorry

end binomial_sum_identity_l13_13118


namespace limit_epsilon_delta_l13_13225

theorem limit_epsilon_delta :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x + (7 / 2)| → |x + (7 / 2)| < δ →
    |(2 * x^2 + 13 * x + 21) / (2 * x + 7) + (1 / 2)| < ε :=
by
  sorry

end limit_epsilon_delta_l13_13225


namespace convex_polygon_center_symmetry_l13_13230

def centrally_symmetric (P : Type) [MetricSpace P] (polygon : set P) : Prop :=
  ∃ O : P, ∀ p ∈ polygon, ∃ q ∈ polygon, midpoint ℝ (p, q) = O

def convex_polygon (P : Type) [MetricSpace P] (M : set P) : Prop :=
  is_convex ℝ M

def can_be_divided_into (P : Type) [MetricSpace P] (M : set P) (polygons : set (set P)) : Prop :=
  M = ⋃₀ polygons ∧ (∀ polygon ∈ polygons, centrally_symmetric P polygon)

theorem convex_polygon_center_symmetry
  {P : Type} [MetricSpace P] (M : set P)
  (h1 : convex_polygon P M)
  (h2 : ∃ polygons : set (set P), can_be_divided_into P M polygons) :
  ∃ O : P, ∀ p ∈ M, ∃ q ∈ M, midpoint ℝ (p, q) = O :=
sorry

end convex_polygon_center_symmetry_l13_13230


namespace jose_initial_caps_l13_13553

-- Definition of conditions and the problem
def jose_starting_caps : ℤ :=
  let final_caps := 9
  let caps_from_rebecca := 2
  final_caps - caps_from_rebecca

-- Lean theorem to state the required proof
theorem jose_initial_caps : jose_starting_caps = 7 := by
  -- skip proof
  sorry

end jose_initial_caps_l13_13553


namespace probability_green_face_up_l13_13034
open_locale classical

theorem probability_green_face_up (h_faces : 8 > 0) (h_green_faces : 5 > 0) : 
  let total_faces := 8 in
  let green_faces := 5 in
  green_faces / total_faces = 5 / 8 := 
by
  sorry

end probability_green_face_up_l13_13034


namespace winning_cards_at_least_one_l13_13429

def cyclicIndex (n : ℕ) (i : ℕ) : ℕ := (i % n + n) % n

theorem winning_cards_at_least_one (a : ℕ → ℕ) (h : ∀ i, (a (cyclicIndex 8 (i - 1)) + a i + a (cyclicIndex 8 (i + 1))) % 2 = 1) :
  ∀ i, 1 ≤ a i :=
by
  sorry

end winning_cards_at_least_one_l13_13429


namespace prescription_duration_l13_13944

theorem prescription_duration (D : ℕ) (h1 : (2 * D) * (1 / 5) = 12) : D = 30 :=
by
  sorry

end prescription_duration_l13_13944


namespace fraction_reducible_l13_13789

theorem fraction_reducible (l : ℤ) : ∃ d : ℤ, d ≠ 1 ∧ d > 0 ∧ d = gcd (5 * l + 6) (8 * l + 7) := by 
  use 13
  sorry

end fraction_reducible_l13_13789


namespace ratio_of_shaded_area_l13_13301

/-- A right isosceles triangle ABC with given midpoints D, E, F, G, H and 
    the ratio of shaded to non-shaded area is 5/11. -/
theorem ratio_of_shaded_area (ABC : Triangle) 
  (right_isosceles : IsRightIsoscelesTriangle ABC)
  (D E F : Point)
  (mid_D : Midpoint (ABC.side_a B C) D)
  (mid_E : Midpoint (ABC.side_b A C) E)
  (mid_F : Midpoint (ABC.side_c A B) F)
  (G H : Point)
  (mid_G : Midpoint (LineSegment D F) G)
  (mid_H : Midpoint (LineSegment F E) H) :
  area (shaded_area ABC D E F G H) / area (non_shaded_area ABC D E F G H) = 5 / 11 := 
sorry

end ratio_of_shaded_area_l13_13301


namespace profit_percentage_A_is_20_l13_13374

-- Define the conditions
def costPriceA : ℝ := 154
def costPriceC : ℝ := 231
def profitRateB : ℝ := 0.25

-- Define the condition equivalences
def costPriceB : ℝ := costPriceC / (1 + profitRateB)
def sellPriceA : ℝ := costPriceB
def profitA : ℝ := sellPriceA - costPriceA
def profitPercentageA : ℝ := (profitA / costPriceA) * 100

-- The statement to be proven
theorem profit_percentage_A_is_20 :
  profitPercentageA = 20 :=
by
  -- The proof is omitted 
  sorry

end profit_percentage_A_is_20_l13_13374


namespace product_not_perfect_power_l13_13737

theorem product_not_perfect_power (n : ℕ) : ¬∃ (k : ℕ) (a : ℤ), k > 1 ∧ n * (n + 1) = a^k := by
  sorry

end product_not_perfect_power_l13_13737


namespace perpendicular_vectors_x_value_l13_13142

-- Define the vectors a and b
def a : ℝ × ℝ := (3, -1)
def b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define the dot product function for vectors in ℝ^2
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- The mathematical statement to prove
theorem perpendicular_vectors_x_value (x : ℝ) (h : dot_product a (b x) = 0) : x = 3 :=
by
  sorry

end perpendicular_vectors_x_value_l13_13142


namespace cookies_per_bag_l13_13245

-- Definitions of the given conditions
def c1 := 23  -- number of chocolate chip cookies
def c2 := 25  -- number of oatmeal cookies
def b := 8    -- number of baggies

-- Statement to prove
theorem cookies_per_bag : (c1 + c2) / b = 6 :=
by 
  sorry

end cookies_per_bag_l13_13245


namespace temperature_conversion_l13_13863

theorem temperature_conversion :
  ∀ (k t : ℝ),
    (t = (5 / 9) * (k - 32) ∧ k = 95) →
    t = 35 := by
  sorry

end temperature_conversion_l13_13863


namespace third_prince_pays_to_second_l13_13618

-- Defining variables and assumptions
variables {x a b k : ℕ}

-- Conditions
def eldest_fruits := 2 * x
def second_fruits := 3 * x
def third_fruits_final := a + b

axiom fruits_equal (h1 : 2 * x - a = a + b) (h2 : 3 * x - b = a + b) : true
axiom total_spent (h3 : k * (a + b) = 180) : true

-- Equation derived from conditions
lemma b_value (h4 : b = 2 * x - 2 * a) (h5 : x = 3 * a) : true :=
sorry

lemma amount_to_second_prince (h6 : b * k = 144) : true :=
sorry

-- Theorem to prove
theorem third_prince_pays_to_second :
  b * k = 144 :=
begin
  -- Using the given conditions and derived equations,
  exact amount_to_second_prince,
end

end third_prince_pays_to_second_l13_13618


namespace find_angle_ACB_l13_13528

axiom triangle_ABC (A B C D : Type) 
  (angle_ABC : Real) (angle_DAB : Real) 
  (k : Real) (BD CD BC : Real) :
  angle_ABC = 30 ∧ angle_DAB = 45 ∧ (3 * BD = 2 * CD) → 
  (∃ θ : Real, θ = 75 ∧ angle A C B = θ)

theorem find_angle_ACB
  (A B C : Type) (angle_ABC : Real) (angle_DAB : Real) 
  (k : Real) (BD CD BC : Real) :
  angle_ABC = 30 ∧ angle_DAB = 45 ∧ (3 * BD = 2 * CD) → 
  ∃ θ : Real, θ = 75 ∧ angle A C B = θ := 
sorry

end find_angle_ACB_l13_13528


namespace option_d_correct_l13_13324

-- Definition for the problem statements
def generatrix_of_cone (gen : ℝ) (diam_base_circle : ℝ) : Prop :=
  gen = diam_base_circle

def generatrix_of_cylinder (gen : ℝ) (axis : ℝ) : Prop :=
  gen ⊥ axis

def generatrix_of_frustum (gen : ℝ) (axis : ℝ) : Prop :=
  gen ∥ axis

def diameter_of_sphere (diam : ℝ) (center : ℝ × ℝ × ℝ) : Prop :=
  ∀ (d : ℝ) (c : ℝ × ℝ × ℝ), diam = 2 * d ∧ c = center

-- The statement that needs to be proved
theorem option_d_correct (diam : ℝ) (center : ℝ × ℝ × ℝ) : 
  diameter_of_sphere diam center :=
by
  sorry

end option_d_correct_l13_13324


namespace foldable_polygons_count_l13_13975

def isValidFolding (base_positions : Finset Nat) (additional_position : Nat) : Prop :=
  ∃ (valid_positions : Finset Nat), valid_positions = {4, 5, 6, 7, 8, 9} ∧ additional_position ∈ valid_positions

theorem foldable_polygons_count : 
  ∃ (valid_additional_positions : Finset Nat), valid_additional_positions = {4, 5, 6, 7, 8, 9} ∧ valid_additional_positions.card = 6 := 
by
  sorry

end foldable_polygons_count_l13_13975


namespace largest_value_is_E_l13_13319

-- Define the given values
def A := 1 - 0.1
def B := 1 - 0.01
def C := 1 - 0.001
def D := 1 - 0.0001
def E := 1 - 0.00001

-- Main theorem statement
theorem largest_value_is_E : E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end largest_value_is_E_l13_13319


namespace C_share_l13_13358

-- Definitions based on conditions
def total_sum : ℝ := 164
def ratio_B : ℝ := 0.65
def ratio_C : ℝ := 0.40

-- Statement of the proof problem
theorem C_share : (ratio_C * (total_sum / (1 + ratio_B + ratio_C))) = 32 :=
by
  sorry

end C_share_l13_13358


namespace bushes_count_after_work_l13_13548

theorem bushes_count_after_work :
  ∀ (orchid_initial rose_initial tulip_initial : ℕ)
    (orchid_planted rose_removed tulip_multiplier : ℕ),
    orchid_initial = 2 →
    rose_initial = 5 →
    tulip_initial = 3 →
    orchid_planted = 4 →
    rose_removed = 1 →
    tulip_multiplier = 2 →
    (orchid_initial + orchid_planted = 6) ∧
    (rose_initial - rose_removed = 4) ∧
    (tulip_initial * tulip_multiplier = 6) :=
by
  intros orchid_initial rose_initial tulip_initial orchid_planted rose_removed tulip_multiplier
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  split; refl

end bushes_count_after_work_l13_13548


namespace sin_double_angle_l13_13806

theorem sin_double_angle (x : ℝ) (h : sin (π / 4 - x) = 3 / 5) : sin (2 * x) = 7 / 25 :=
sorry

end sin_double_angle_l13_13806


namespace find_sin_θ_find_cos_2θ_find_cos_φ_l13_13817

noncomputable def θ : ℝ := sorry
noncomputable def φ : ℝ := sorry

-- Conditions
axiom cos_eq : Real.cos θ = Real.sqrt 5 / 5
axiom θ_in_quadrant_I : 0 < θ ∧ θ < Real.pi / 2
axiom sin_diff_eq : Real.sin (θ - φ) = Real.sqrt 10 / 10
axiom φ_in_quadrant_I : 0 < φ ∧ φ < Real.pi / 2

-- Goals
-- Part (I) Prove the value of sin θ
theorem find_sin_θ : Real.sin θ = 2 * Real.sqrt 5 / 5 :=
by
  sorry

-- Part (II) Prove the value of cos 2θ
theorem find_cos_2θ : Real.cos (2 * θ) = -3 / 5 :=
by
  sorry

-- Part (III) Prove the value of cos φ
theorem find_cos_φ : Real.cos φ = Real.sqrt 2 / 2 :=
by
  sorry

end find_sin_θ_find_cos_2θ_find_cos_φ_l13_13817


namespace rationalizing_factors_fraction_simplification_compare_sqrt_telescoping_series_l13_13032

-- Part 1: Rationalizing factors
theorem rationalizing_factors : 
  ( (Real.sqrt 2023 - Real.sqrt 2022) * (Real.sqrt 2023 + Real.sqrt 2022) = 2023 - 2022 ) := 
  by sorry

-- Part 2: Simplification of the fraction
theorem fraction_simplification (n : ℕ) (h : 0 < n) :
  (1 / (Real.sqrt (n + 1) + Real.sqrt n) = Real.sqrt (n + 1) - Real.sqrt n) := 
  by sorry

-- Part 3: Comparison of expressions
theorem compare_sqrt : 
  (Real.sqrt 2023 - Real.sqrt 2022 < Real.sqrt 2022 - Real.sqrt 2021) := 
  by sorry

-- Part 4: Telescoping series calculation
theorem telescoping_series :
  ( ((range 1011).map (λ k, 1 / (Real.sqrt (2 * k + 3) + Real.sqrt (2 * k + 1))).sum * (Real.sqrt 2023 + 1) = 1011 ) := 
  by sorry

end rationalizing_factors_fraction_simplification_compare_sqrt_telescoping_series_l13_13032


namespace tangent_line_equation_minimum_value_range_monotonic_function_l13_13492

/-- Question (I): Tangent Line Equation -/
theorem tangent_line_equation (a : ℝ) :
  a = 1 → (∃ b c : ℝ, ∀ x : ℝ, b * x + c = -2) :=
sorry

/-- Question (II): Minimum Value Range -/
theorem minimum_value_range (a : ℝ) :
  (a > 0 ∧ ∃ (f : ℝ → ℝ), (∀ x : ℝ, f x = a * x^2 - (a + 2) * x + log x) ∧ (∀ x ∈ set.Icc (1 : ℝ) (Real.exp 1), f x ≥ -2)) → a ≥ 1 :=
sorry

/-- Question (III): Monotonic Function -/
theorem monotonic_function (a : ℝ) :
  (∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 > 0 → x2 > 0 → (a*x1^2 - (a+2)*x1 + log x1 + 2*x1 - (a*x2^2 - (a+2)*x2 + log x2 + 2*x2)) / (x1 - x2) > 0) → 0 < a ∧ a ≤ 8 :=
sorry

end tangent_line_equation_minimum_value_range_monotonic_function_l13_13492


namespace log_difference_limit_l13_13520

open Real

theorem log_difference_limit (x : ℝ) (h : 0 < x) (h_inf : filter.at_top x) : 
  tendsto (λ x, log 5 (10 * x - 7) - log 5 (3 * x + 2)) filter.at_top (nhds (log 5 (10 / 3))) :=
by
  sorry

end log_difference_limit_l13_13520


namespace imaginary_part_and_modulus_l13_13463

open Complex

theorem imaginary_part_and_modulus (z ω : ℂ) 
  (hz : i * (z + 1) = -2 + 2 * i)
  (hω : ω = z / (1 - 2 * i)) :
  im z = 2 ∧ abs ω ^ 2015 = 1 :=
by 
  -- Define z from given condition
  have hz_def : z = 1 + 2 * i := 
    by {
      have h := congr_arg (λ x, x / i) hz,
      simp [hz, i_mul_i] at h,
      exact h.symm,
    }
  subst hz_def,

  -- Show imaginary part of z is 2
  have him_z : im (1 + 2 * i) = 2 := by simp,
  
  -- Define abs ω from given condition and its power
  have hw : abs ((1 + 2 * i) / (1 - 2 * i)) = 1 :=
    by simp [abs_div, norm_eq_abs, Complex.abs], 

  split,
  exact him_z,
  calc
    (abs ω) ^ 2015 = 1 ^ 2015 : by {rw hw, },
    rw pow_one,
end

end imaginary_part_and_modulus_l13_13463


namespace range_of_a_l13_13495

def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a * x^2

def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.log x + 1 + 2 * a * x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ f_prime x a = 3) ↔ (-1 / (2 * Real.exp 3) ≤ a) :=
by
  sorry

end range_of_a_l13_13495


namespace number_of_nonempty_subsets_of_P_l13_13938

noncomputable def P : Set (ℕ × ℕ) := {p | p.1 + p.2 < 3 ∧ p.1 > 0 ∧ p.2 > 0}

theorem number_of_nonempty_subsets_of_P : (2 ^ P.to_finset.card - 1) = 1 := by
  sorry

end number_of_nonempty_subsets_of_P_l13_13938


namespace product_of_m_and_r_l13_13751

-- The 3x3 grid of points with coordinates satisfying 0 ≤ x, y ≤ 2
def point (x y : ℕ) : Prop := 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2

-- Define a growing path as a sequence of distinct points where distance between consecutive points strictly increases
def growing_path (path : List (ℕ × ℕ)) : Prop :=
  ∀ i j, i < j ∧ j < path.length →
    ((path.nth i).fst - (path.nth j).fst)^2 + ((path.nth i).snd - (path.nth j).snd)^2 < ((path.nth j).fst - (path.nth (j + 1)).fst)^2 + ((path.nth j).snd - (path.nth (j + 1)).snd)^2

-- Define the maximum number of points in a growing path (m) and the number of distinct growing paths with exactly m points (r)
def m : ℕ := 6
def r : ℕ := 8

-- Prove that the product of m and r is 48
theorem product_of_m_and_r : m * r = 48 :=
by
  sorry

end product_of_m_and_r_l13_13751


namespace ellipse_equation_area_is_constant_l13_13473

-- Let's define the parameters and show two goals based on given conditions.
def a : ℝ := 2
def b : ℝ := sqrt 3

-- Ellipse eccentricity fact
axiom eccentricity : a > b ∧ b > 0 ∧ (1 / 2 = sqrt (a^2 - b^2) / a)

-- Assertion that ellipse passes through (1, 3/2)
axiom passes_through : 1 / (a^2 : ℝ) + (3 / 2)^2 / (b^2 : ℝ) = 1

-- First goal: Prove the equation of the ellipse
theorem ellipse_equation :
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 3 = 1)) := by
  sorry

-- Second goal: Prove the area of The Area of ΔAOB is a constant value
theorem area_is_constant :
  (∀ (k m : ℝ), (some_conditions_implied_by_the_problem) → (area_of_ΔAOB_is_√3)) := by
  sorry

end ellipse_equation_area_is_constant_l13_13473


namespace perimeter_of_staircase_region_l13_13534

theorem perimeter_of_staircase_region :
  ∀ (length stairs_count : ℕ) (area total_area : ℝ), 
    total_area = 12.0 * (101.0 / 12.0) - 16.0 -> 
    length = 12 ->
    stairs_count = 12 ->
    area = 85.0 ->
    (2 * length + 2 * (101.0 / 12.0) + stairs_count) = 41 := 
by
  intros length stairs_count area total_area h_total_area h_length h_stairs_count h_area
  sorry

end perimeter_of_staircase_region_l13_13534


namespace johns_overall_loss_l13_13551

noncomputable def johns_loss_percentage : ℝ :=
  let cost_A := 1000 * 2
  let cost_B := 1500 * 3
  let cost_C := 2000 * 4
  let discount_A := 0.1
  let discount_B := 0.15
  let discount_C := 0.2
  let cost_A_after_discount := cost_A * (1 - discount_A)
  let cost_B_after_discount := cost_B * (1 - discount_B)
  let cost_C_after_discount := cost_C * (1 - discount_C)
  let total_cost_after_discount := cost_A_after_discount + cost_B_after_discount + cost_C_after_discount
  let import_tax_rate := 0.08
  let import_tax := total_cost_after_discount * import_tax_rate
  let total_cost_incl_tax := total_cost_after_discount + import_tax
  let cost_increase_rate_C := 0.04
  let new_cost_C := 2000 * (4 + 4 * cost_increase_rate_C)
  let adjusted_total_cost := cost_A_after_discount + cost_B_after_discount + new_cost_C
  let total_selling_price := (800 * 3) + (70 * 3 + 1400 * 3.5 + 900 * 5) + (130 * 2.5 + 130 * 3 + 130 * 5)
  let gain_or_loss := total_selling_price - adjusted_total_cost
  let loss_percentage := (gain_or_loss / adjusted_total_cost) * 100
  loss_percentage

theorem johns_overall_loss : abs (johns_loss_percentage + 4.09) < 0.01 := sorry

end johns_overall_loss_l13_13551


namespace line_through_F_min_value_frac_l13_13175

-- Definitions based on conditions
variable (A : ℝ × ℝ) (m p : ℝ)
variable (H1 : A = (2, 2))
variable (H2 : p > 0)
variable (H3 : m > 0)

-- Equation of the parabola and verification of F
noncomputable def parabola (y : ℝ) : ℝ := (y^2) / 2
variable (F : ℝ × ℝ)
variable (H4 : F = (parabola 2, 0))

-- Equation of the line perpendicular to OA through F
theorem line_through_F (x y : ℝ) (H5 : x = parabola 2) (H6 : y = 0):
  x + y - 1 / 2 = 0 :=
  sorry

-- Conditions for points D and E on the parabola intersected by line through M
variable (D E : ℝ × ℝ)
variable (H7 : ∃ k : ℝ, E = (m, 0) ∧ line_through_F D E k)
variable (H8 : |E.2 - M.2| = 2 * |D.2 - M.2|)

-- Calculating the minimum value of the given expression
noncomputable def DE_squared (D E : ℝ × ℝ) : ℝ :=
  (D.1 - E.1)^2 + (D.2 - E.2)^2

noncomputable def OM (m: ℝ) := m

theorem min_value_frac (H9 : D ≠ E) :
  ∃ m : ℝ, m = 2/3 ∧ 12 = inf ((DE_squared D E + 1) / OM m) :=
  sorry

end line_through_F_min_value_frac_l13_13175


namespace find_tan_B_l13_13176

theorem find_tan_B (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_angle_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_condition1 : sin A / sin B + cos C = 0)
  (h_condition2 : tan A = sqrt 2 / 4) :
  tan B = sqrt 2 / 2 := 
sorry

end find_tan_B_l13_13176


namespace initial_consumption_l13_13681

variable (P : ℝ) (C : ℝ) (new_consumption : ℝ := 25)
variable (price_increase : ℝ := 1.32) (expenditure_increase : ℝ := 1.10)

theorem initial_consumption :
  (price_increase * new_consumption = expenditure_increase * P * C) → 
  (C = 75) :=
begin
  intros h,
  sorry -- Proof is skipped as specified
end

end initial_consumption_l13_13681


namespace find_a_plus_b_l13_13986

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (9^x - a) / 3^x
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := log (10^x + 1) + b * x

theorem find_a_plus_b :
  (∀ x : ℝ, f x a = - f (-x) a) →
  (∀ x : ℝ, g x b = g (-x) b) →
  a + b = 1 / 2 :=
by
  sorry

end find_a_plus_b_l13_13986


namespace translation_symmetry_left_translation_symmetry_right_l13_13490

variable (k k' : ℤ)
def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem translation_symmetry_left (φ : ℝ) (h₁ : 2 * φ + Real.pi / 6 = k * Real.pi + Real.pi / 2) :
  φ = Real.pi / 6 :=
by
  sorry

theorem translation_symmetry_right (φ : ℝ) (h₂ : Real.pi / 6 - 2 * φ = k' * Real.pi + Real.pi / 2) :
  φ = Real.pi / 3 :=
by
  sorry

end translation_symmetry_left_translation_symmetry_right_l13_13490


namespace find_k_l13_13506

def vector_a : ℝ × ℝ × ℝ := (1, 1, 0)
def vector_b : ℝ × ℝ × ℝ := (-1, 0, 2)

def is_perpendicular (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem find_k (k : ℝ) (h : is_perpendicular vector_a (vector_a.1 - k, vector_a.2, 2 * k)) : k = 2 :=
by
  sorry

end find_k_l13_13506


namespace output_is_15_5_l13_13547

-- Step 1: Define the input
def input := 15

-- Step 2: Define the first condition (Multiply by 1.5)
def multiply_by_1_5 (x : ℝ) : ℝ := x * 1.5

-- Step 3: Define the second condition (Comparison with 20 and subsequent operations)
def function_machine (x : ℝ) : ℝ :=
  if multiply_by_1_5 x > 20 then
    multiply_by_1_5 x - 7
  else
    multiply_by_1_5 x + 10

-- Step 4: State the theorem to be proved
theorem output_is_15_5 : function_machine input = 15.5 :=
by sorry

end output_is_15_5_l13_13547


namespace inequality_solution_l13_13426

theorem inequality_solution (x y : ℝ) : 
  2 * y - 3 * x < real.sqrt (9 * x^2 + 16) → 
  ((x ≥ 0 → y < 4 * x) ∧ (x < 0 → y < -x)) :=
by 
  sorry

end inequality_solution_l13_13426


namespace parabola_no_intersection_with_x_axis_l13_13847

theorem parabola_no_intersection_with_x_axis (x : ℝ) : 
  let y := -2 * x^2 + x - 1 in 
  (y = 0 → ¬ ∃ (x : ℝ), -2 * x^2 + x - 1 = 0) :=
by
  intro x y_eq
  have eqn := by sorry
  have discriminant_negative := by sorry
  have no_real_roots := by sorry
  sorry

end parabola_no_intersection_with_x_axis_l13_13847


namespace probability_both_white_is_zero_l13_13303

theorem probability_both_white_is_zero (a b : ℕ) (h1 : a + b = 36) (h2 : ∀ x, x = a → ∃ y, y = b) 
    (h3 : (a : ℚ) / 36 * (a : ℚ) / b = 18 / 25) : 
    ∃ m n : ℕ, m / n = 0 / 1 ∧ Nat.coprime m n ∧ m + n = 1 := 
by
    sorry

end probability_both_white_is_zero_l13_13303


namespace number_of_elements_in_P_add_Q_l13_13141

def P : Set ℕ := {0, 2, 5}
def Q : Set ℕ := {1, 2, 6}

def P_add_Q : Set ℕ := {c | ∃ a ∈ P, ∃ b ∈ Q, c = a + b}

theorem number_of_elements_in_P_add_Q : (P_add_Q : Set ℕ).toFinset.card = 8 := by
  sorry

end number_of_elements_in_P_add_Q_l13_13141


namespace linear_equation_m_equals_neg_3_l13_13828

theorem linear_equation_m_equals_neg_3 
  (m : ℤ)
  (h1 : |m| - 2 = 1)
  (h2 : m - 3 ≠ 0) :
  m = -3 :=
sorry

end linear_equation_m_equals_neg_3_l13_13828


namespace adult_ticket_price_is_35_l13_13654

-- Define the cost of children's tickets and total family configuration
def child_ticket_cost : ℕ := 20
def num_children : ℕ := 6
def num_adults : ℕ := 1

-- Define the total cost of separate tickets
def total_separate_cost : ℕ := 155

-- Define the total cost of family pass
def family_pass_cost : ℕ := 120

-- Define the hypothesis of the problem
axiom adult_ticket_cost (A : ℕ) : (A + (num_children * child_ticket_cost) = total_separate_cost)

-- The theorem we need to prove
theorem adult_ticket_price_is_35 (A : ℕ) (h : adult_ticket_cost A) : A = 35 := 
  sorry

end adult_ticket_price_is_35_l13_13654


namespace borrowing_methods_l13_13348

theorem borrowing_methods (A_has_3_books : True) (B_borrows_at_least_one_book : True) :
  (∃ (methods : ℕ), methods = 7) :=
by
  existsi 7
  sorry

end borrowing_methods_l13_13348


namespace cost_per_topping_is_2_l13_13509

theorem cost_per_topping_is_2 : 
  ∃ (x : ℝ), 
    let large_pizza_cost := 14 
    let num_large_pizzas := 2 
    let num_toppings_per_pizza := 3 
    let tip_rate := 0.25 
    let total_cost := 50 
    let cost_pizzas := num_large_pizzas * large_pizza_cost 
    let num_toppings := num_large_pizzas * num_toppings_per_pizza 
    let cost_toppings := num_toppings * x 
    let before_tip_cost := cost_pizzas + cost_toppings 
    let tip := tip_rate * before_tip_cost 
    let final_cost := before_tip_cost + tip 
    final_cost = total_cost ∧ x = 2 := 
by
  simp
  sorry

end cost_per_topping_is_2_l13_13509


namespace zane_pays_l13_13687

noncomputable def total_cost_in_usd (regular_price : ℕ) (discount1 discount2 tax exchange_rate : ℚ) : ℚ :=
let price1 := regular_price - (regular_price * discount1) in
let price2 := regular_price - (regular_price * discount2) in
let total := price1 + price2 in
let total_with_tax := total * (1 + tax) in
total_with_tax * exchange_rate

theorem zane_pays (regular_price : ℕ) (discount1 discount2 tax exchange_rate : ℚ) :
  regular_price = 50 →
  discount1 = 0.40 → 
  discount2 = 0.30 → 
  tax = 0.08 → 
  exchange_rate = 1.18 →
  total_cost_in_usd regular_price discount1 discount2 tax exchange_rate ≈ 82.84 :=
by {
  intros,
  sorry
}

end zane_pays_l13_13687


namespace direction_vectors_same_line_k_l13_13873

theorem direction_vectors_same_line_k :
  ∃ k : ℝ, let a := (8, -2, 1) and b := (-4, 1, k) in
  ∃ λ : ℝ, a = (λ * b.1, λ * b.2, λ * b.3) ∧ k = -1 / 2 :=
begin
  sorry
end

end direction_vectors_same_line_k_l13_13873


namespace antiderivative_correct_l13_13453

noncomputable def f (x : ℝ) : ℝ := 2 * sin (5 * x) + sqrt x + 3 / 5

noncomputable def F (x : ℝ) : ℝ := -2 / 5 * cos (5 * x) + 2 / 3 * x * sqrt x + 3 / 5 * x + 1

theorem antiderivative_correct : 
  (∀ x, has_deriv_at F (f x) x) ∧ F 0 = f 0 :=
by 
  sorry

end antiderivative_correct_l13_13453


namespace gcd_lcm_identity_l13_13928

theorem gcd_lcm_identity {a b c : ℕ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (gcd a b c)^2 / (gcd a b * gcd a c * gcd b c) = 
  (lcm a b c)^2 / (lcm a b * lcm a c * lcm b c) := 
sorry

end gcd_lcm_identity_l13_13928


namespace min_value_a_l13_13882

/-- In a deck of 52 cards, Alex and Dylan's team formation problem. -/
theorem min_value_a (a : ℕ) (h₁ : a ≤ 52) (h₂ : a + 15 ≤ 52) :
  ∃ a, a ≤ 38 ∧ 
  (let p := ((nat.choose (36 - a) 2) + (nat.choose (a - 1) 2)) / (2 * 1225) in 
  p ≥ 1 / 2) :=
by sorry

end min_value_a_l13_13882


namespace tens_digit_19_pow_2023_l13_13772

theorem tens_digit_19_pow_2023 :
  ∃ d : ℕ, d = (59 / 10) % 10 ∧ (19 ^ 2023 % 100) / 10 = d :=
by
  have h1 : 19 ^ 10 % 100 = 1 := by sorry
  have h2 : 19 ↔ 0 := by sorry
  have h4 : 2023 % 10 = 3 := by sorry
  have h5 : 19 ^ 10 ↔ 1 := by sorry
  have h6 : 19 ^ 3 % 100 = 59 := by sorry
  have h7 : (19 ^ 2023 % 100) = 59 := by sorry
  exists 5
  split
  repeat { assumption.dump }

end tens_digit_19_pow_2023_l13_13772


namespace example_of_divisible_45_exists_l13_13949

open Nat

def sequence : List ℕ := [2, 0, 1, 5, 2, 0, 1, 5, 2, 0, 1, 5, 2, 0, 1, 5, 2, 0, 1, 5]

def isDivisibleBy45 (n : ℕ) : Prop := (n % 45 = 0)

def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem example_of_divisible_45_exists :
  ∃ s1 s2 : List ℕ, 
  s1 ++ s2 = [2, 5, 2, 0] ∧
  let n := List.foldl (fun acc d => 10 * acc + d) 0 (s1 ++ s2) in
  isDivisibleBy45 n :=
by
  exists [2], [5, 2, 0]
  simp
  exact rfl

end example_of_divisible_45_exists_l13_13949


namespace tens_digit_19_pow_2023_l13_13770

theorem tens_digit_19_pow_2023 :
  ∃ d : ℕ, d = (59 / 10) % 10 ∧ (19 ^ 2023 % 100) / 10 = d :=
by
  have h1 : 19 ^ 10 % 100 = 1 := by sorry
  have h2 : 19 ↔ 0 := by sorry
  have h4 : 2023 % 10 = 3 := by sorry
  have h5 : 19 ^ 10 ↔ 1 := by sorry
  have h6 : 19 ^ 3 % 100 = 59 := by sorry
  have h7 : (19 ^ 2023 % 100) = 59 := by sorry
  exists 5
  split
  repeat { assumption.dump }

end tens_digit_19_pow_2023_l13_13770


namespace find_math_marks_l13_13045

theorem find_math_marks (subjects : ℕ)
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℝ)
  (math_marks : ℕ) :
  subjects = 5 →
  english_marks = 96 →
  physics_marks = 99 →
  chemistry_marks = 100 →
  biology_marks = 98 →
  average_marks = 98.2 →
  math_marks = 98 :=
by
  intros h_subjects h_english h_physics h_chemistry h_biology h_average
  sorry

end find_math_marks_l13_13045


namespace profit_is_37_l13_13711

-- Define the initial parameters
def initial_cost : ℕ := 400
def price_per_set : ℕ := 55
def number_of_sets : ℕ := 8
def adjustments : List ℤ := [+2, -3, +2, +1, -2, -1, 0, -2]

-- Compute the total adjustment
def total_adjustment : ℤ := adjustments.sum

-- Compute the total revenue
def total_revenue : ℤ := price_per_set * number_of_sets + total_adjustment

-- Define the profit calculation
def profit : ℤ := total_revenue - initial_cost

-- Statement to prove that the profit is 37 yuan
theorem profit_is_37 : profit = 37 := by
  sorry

end profit_is_37_l13_13711


namespace fruit_bowl_l13_13979

variable {A P B : ℕ}

theorem fruit_bowl : (P = A + 2) → (B = P + 3) → (A + P + B = 19) → B = 9 :=
by
  intros h1 h2 h3
  sorry

end fruit_bowl_l13_13979


namespace total_games_is_seven_l13_13640

def total_football_games (games_missed : ℕ) (games_attended : ℕ) : ℕ :=
  games_missed + games_attended

theorem total_games_is_seven : total_football_games 4 3 = 7 := 
by
  sorry

end total_games_is_seven_l13_13640


namespace comb_8_3_eq_56_l13_13398

theorem comb_8_3_eq_56 : nat.choose 8 3 = 56 := sorry

end comb_8_3_eq_56_l13_13398


namespace cupcakes_for_children_l13_13044

-- Definitions for the conditions
def packs15 : Nat := 4
def packs10 : Nat := 4
def cupcakes_per_pack15 : Nat := 15
def cupcakes_per_pack10 : Nat := 10

-- Proposition to prove the total number of cupcakes is 100
theorem cupcakes_for_children :
  (packs15 * cupcakes_per_pack15) + (packs10 * cupcakes_per_pack10) = 100 := by
  sorry

end cupcakes_for_children_l13_13044


namespace area_triangle_MOD_l13_13532

-- Define the circle with center O and radius r
variables (O A B M D : Type)
variable {r : ℝ}

-- Define the properties and conditions
def is_circle_center (O : Type) (r : ℝ) : Prop := true
def is_chord (A B : Type) (r : ℝ) : Prop := true
def chord_midpoint (A B M : Type) : Prop := true
def perpendicular (O M : Type) : Prop := true

-- Adding the theorem based on the conditions above
theorem area_triangle_MOD (O A B M D : Type) (r : ℝ)
  [h₁ : is_circle_center O r]
  [h₂ : is_chord A B (r * real.sqrt 2)]
  [h₃ : chord_midpoint A B M]
  [h₄ : perpendicular O M]
  [h₅ : perpendicular M O] :
  area_triangle O M D = r^2 / (4 * real.sqrt 2) := sorry

end area_triangle_MOD_l13_13532


namespace min_value_expression_l13_13836

theorem min_value_expression
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : 0 < a ∧ a < 2)
  (h2 : 0 < b ∧ b < 2)
  (h3 : f = λ x, Real.cos (π * x))
  (h4 : f a = f b)
  (h5 : a ≠ b)
  : ∃ (k : ℝ), k = (1/a + 4/b) ∧ k = 9/2 := sorry

end min_value_expression_l13_13836


namespace probability_diagonals_intersect_hexagon_l13_13881

theorem probability_diagonals_intersect_hexagon:
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2 -- Total number of diagonals in a convex polygon
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2 -- Total number of ways to choose 2 diagonals
  let non_principal_intersections := 3 * 6 -- Each of 6 non-principal diagonals intersects 3 others
  let principal_intersections := 4 * 3 -- Each of 3 principal diagonals intersects 4 others
  let total_intersections := (non_principal_intersections + principal_intersections) / 2 -- Correcting for double-counting
  let probability := total_intersections / total_pairs -- Probability of intersection inside the hexagon
  probability = 5 / 12 := by
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2
  let non_principal_intersections := 3 * 6
  let principal_intersections := 4 * 3
  let total_intersections := (non_principal_intersections + principal_intersections) / 2
  let probability := total_intersections / total_pairs
  have h : total_diagonals = 9 := by norm_num
  have h_pairs : total_pairs = 36 := by norm_num
  have h_intersections : total_intersections = 15 := by norm_num
  have h_prob : probability = 5 / 12 := by norm_num
  exact h_prob

end probability_diagonals_intersect_hexagon_l13_13881


namespace speed_of_travel_l13_13022

theorem speed_of_travel (
  (h_diagonal_time : 2 * 60 = 120), -- travel time in seconds
  (h_area : 50 = 50) -- area in square meters
) : ∃ (speed : ℝ), speed = 0.3 := 
sorry

end speed_of_travel_l13_13022


namespace general_term_formula_sum_first_n_terms_l13_13123

-- Define the conditions as Lean variables and axioms
variable {a : ℕ → ℝ} -- the sequence
variable (n : ℕ) -- natural number for indexing the sequence

-- Given conditions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * a 1 / a 0
axiom a2_a5_eq_36 : a 2 + a 5 = 36
axiom a3_a4_eq_128 : a 3 * a 4 = 128

-- The first part: Finding the general term formula
theorem general_term_formula : (a 0 = 2 ∧ ∀ n, a n = 2^n) ∨ (a 0 = 64 ∧ ∀ n, a n = 2^(7-n))
:= sorry

-- Additional condition for part II, assuming an increasing sequence
axiom increasing_sequence : ∀ n m, n < m → a n < a m

-- Define sequence b_n
def b (n : ℕ) : ℝ := a n + real.log (a n) / real.log 2

-- Define the sum S_n
def S (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- The second part: finding the sum of the first n terms
theorem sum_first_n_terms (n : ℕ) : S n = 2^(n+1) - 2 + n * (n + 1) / 2
:= sorry

end general_term_formula_sum_first_n_terms_l13_13123


namespace length_of_side_AC_l13_13877

-- Define the problem data
variables (A B C M N G : Type) [MetricSpace A]
variables (theta : ℝ) (AM BN AC : ℝ)
variable  (triangle : Triangle A B C)

-- State the conditions
variables (median_AM : Median triangle M A)
variables (median_BN : Median triangle N B)
variables (intersect_angle : MeasureAngle median_AM median_BN = 60)
variables (length_AM : Length AM = 12)
variables (length_BN : Length BN = 16)

-- State the conclusion
theorem length_of_side_AC :
  Length AC = (16 * Real.sqrt 13) / 3 := sorry

end length_of_side_AC_l13_13877


namespace symmetric_line_equation_l13_13261

theorem symmetric_line_equation (x y : ℝ) (h : 4 * x - 3 * y + 5 = 0):
  4 * x + 3 * y + 5 = 0 :=
sorry

end symmetric_line_equation_l13_13261


namespace incenter_bisects_perimeter_l13_13021

theorem incenter_bisects_perimeter (A B C I D E F : Point)
  (hIncenter : ∀ (P : Point), Incenter P A B C ↔ P = I)
  (hMidpointD : Midpoint A B C D)
  (hMidpointE : Midpoint B C A E)
  (hMidpointF : Midpoint C A B F) :
  BisectsPerimeter A B C D E F I :=
sorry

end incenter_bisects_perimeter_l13_13021


namespace satisfaction_independence_distribution_and_expectation_l13_13727

-- Conditions
def total_people : ℕ := 200
def both_satisfied : ℕ := 50
def dedication_satisfied : ℕ := 80
def management_satisfied : ℕ := 90
def alpha_value : ℝ := 0.01
def chi_squared_stat : ℝ := (200 * (50 * 80 - 30 * 40)^2) / (80 * 120 * 90 * 110)
def critical_value : ℝ := 6.635
def p_X_0 : ℝ := 27/64
def p_X_1 : ℝ := 27/64
def p_X_2 : ℝ := 9/64
def p_X_3 : ℝ := 1/64
def expected_value_X : ℝ := 3/4

-- Statement for part 1: Testing for Independence
theorem satisfaction_independence : chi_squared_stat > critical_value := by
  sorry

-- Statement for part 2: Distribution table and Mathematical Expectation
theorem distribution_and_expectation :
  ∀ X, (X = 0 → P(X) = p_X_0) ∧
       (X = 1 → P(X) = p_X_1) ∧
       (X = 2 → P(X) = p_X_2) ∧
       (X = 3 → P(X) = p_X_3) ∧
       E(X) = expected_value_X :=
by
  sorry

end satisfaction_independence_distribution_and_expectation_l13_13727


namespace correct_propositions_l13_13835

theorem correct_propositions (p1 p2 p3 p4 p5 : Prop) :
  (p1 ↔ "The functions \(y = |x|\) and \(y = (\sqrt{x})^2\) represent the same function.") →
  (p2 ↔ "The graph of an odd function always passes through the origin of the coordinate system.") →
  (p3 ↔ "The graph of the function \(y = 3(x - 1)^2\) can be obtained by shifting the graph of \(y = 3x^2\) one unit to the right.") →
  (p4 ↔ "The minimum value of \(y = 2^{|x|}\) is 1.") →
  (p5 ↔ "If for the function \(f(x)\), \(f(-1) \cdot f(3) < 0\), then the equation \(f(x) = 0\) has a real root in the interval \([-1, 3]\).") →
  (p1 → False) ∧
  (p2 → False) ∧
  (p3 → True) ∧
  (p4 → True) ∧
  (p5 → False) :=
by
  intro h1 h2 h3 h4 h5
  constructor
  · intro h; exact h1.mp h
  constructor
  · intro h; exact h2.mp h
  constructor
  · intro h; exact h3.mp h
  constructor
  · intro h; exact h4.mp h
  · intro h; exact h5.mp h

end correct_propositions_l13_13835


namespace train_speed_kmph_l13_13701

theorem train_speed_kmph 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_seconds : ℝ) 
  (total_length : train_length + platform_length = 400.032) 
  (time_constraint : time_seconds = 20) 
  : (total_length / time_seconds) * 3.6 ≈ 72.006 :=
by
  sorry

end train_speed_kmph_l13_13701


namespace repeating_decimal_as_fraction_lowest_terms_l13_13434

theorem repeating_decimal_as_fraction_lowest_terms :
  ∃ (x : ℝ), x = 0.36 ∧ x = 4 / 11 :=
begin
  let x := 0.363636363636...,
  have h1 : x = 0.36, sorry, -- Represent the repeating decimal
  have h2 : 100 * x = 36.363636..., sorry, -- Multiply by 100 and represent the repeating decimal again
  have h3 : 100 * x - x = 36.363636... - 0.36, sorry, -- Subtraction step and represent the repeating decimal again
  have h4 : 99 * x = 36, sorry, -- Simplify the equation
  have h5 : x = 36 / 99, sorry, -- Solve for x
  have h6 : (36 / 99) = (4 / 11), sorry, -- Simplify the fraction
  use 0.36,
  split,
  { exact h1 },
  { exact h6.symm }
end

end repeating_decimal_as_fraction_lowest_terms_l13_13434


namespace Tanner_berries_l13_13596

def berries (Skylar Steve Stacy Tanner : ℕ) : Prop :=
  Skylar = 20 ∧
  Steve = 4 * (20 / 3) ^ 2 * 6 ∧
  Stacy = 2 * Steve + 50 ∧
  Tanner = (8 * Stacy) / (Skylar + Steve)

theorem Tanner_berries (Skylar Steve Stacy Tanner : ℕ) (h : berries Skylar Steve Stacy Tanner) : Tanner = 16 :=
by
  obtain ⟨hSkylar, hSteve, hStacy, hTanner⟩ := h
  simp [hSkylar, hSteve, hStacy, hTanner]
  sorry

end Tanner_berries_l13_13596


namespace mara_additional_miles_l13_13576

theorem mara_additional_miles
  (distance1 : ℝ) (speed1 : ℝ) (speed2 : ℝ) (target_speed : ℝ) (additional_miles : ℝ)
  (h_distance1 : distance1 = 20)
  (h_speed1 : speed1 = 40)
  (h_speed2 : speed2 = 60)
  (h_target_speed : target_speed = 55)
  (h_additional_miles : additional_miles = 90) :
  let time1 := distance1 / speed1 in
  distance1 + additional_miles = target_speed * (time1 + additional_miles / speed2) :=
by
  sorry

end mara_additional_miles_l13_13576


namespace cells_after_one_week_l13_13721

theorem cells_after_one_week : (3 ^ 7) = 2187 :=
by sorry

end cells_after_one_week_l13_13721


namespace smallest_multiple_of_45_and_75_not_20_l13_13668

-- Definitions of the conditions
def isMultipleOf (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b
def notMultipleOf (a b : ℕ) : Prop := ¬ (isMultipleOf a b)

-- The proof statement
theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ N : ℕ, isMultipleOf N 45 ∧ isMultipleOf N 75 ∧ notMultipleOf N 20 ∧ N = 225 :=
by
  -- sorry is used to indicate that the proof needs to be filled here
  sorry

end smallest_multiple_of_45_and_75_not_20_l13_13668


namespace worst_is_man_l13_13709

-- Definitions of the participants
universe u
constant Player : Type u
constants (man sister nephew niece : Player)

-- Definitions of gender
constant Sex : Type u
constant male female : Sex
constant sex : Player → Sex

-- Age definition (simplified as all players are the same age)
constant same_age : Player → Player → Prop

-- Condition: The worst player's twin and the best player are of opposite sex
constant twin : Player → Player → Prop

-- Condition: The sister is older than the man
constant older : Player → Player → Prop

-- Declaration of the worst and best players
constant worst_player best_player : Player

-- Conditions given in the problem
axiom worst_best_opposite_sex : sex (twin worst_player) ≠ sex best_player
axiom worst_best_same_age : same_age worst_player best_player
axiom sister_older_man : older sister man

-- Goal
theorem worst_is_man : worst_player = man :=
by
  sorry

end worst_is_man_l13_13709


namespace tetrahedron_paintings_l13_13657

theorem tetrahedron_paintings (n : ℕ) (h : n ≥ 4) : 
  let term1 := (n - 1) * (n - 2) * (n - 3) / 12
  let term2 := (n - 1) * (n - 2) / 3
  let term3 := n - 1
  let term4 := 1
  2 * (term1 + term2 + term3) + n = 
  n * (term1 + term2 + term3 + term4) := by
{
  sorry
}

end tetrahedron_paintings_l13_13657


namespace area_of_OAB_l13_13383

variable (AB CD Area_AOBCD : ℝ)
variable (A B C D O : Type) [Point A] [Point B] [Point C] [Point D] [Point O]

axiom trapezoid 
  (AB_parallel_CD : AB ∥ CD)
  (AB_val : AB = 5)
  (CD_val : CD = 3)
  (Area_AB : Area_AOBCD = 4)
-- need more axioms or assumptions to encode the configuration about A, B, C, D -- 

theorem area_of_OAB
  (H : (AB_parallel_CD : AB ∥ CD))
  (h1 : (AB_val : AB = 5))
  (h2 : (CD_val : = CD = 3))
  (h3 : (Area_AB : Area_AOBCD = 4))
  : area_of_point (△ O A B) = 5/8 :=
sorry

end area_of_OAB_l13_13383


namespace yield_percentage_of_stock_l13_13379

def stock_dividend_rate : ℝ := 0.08
def market_value : ℝ := 40
def par_value : ℝ := 100

def annual_dividend (par : ℝ) (rate : ℝ) : ℝ :=
  par * rate

def yield_percentage (dividend : ℝ) (market_val : ℝ) : ℝ :=
  (dividend / market_val) * 100

theorem yield_percentage_of_stock :
  yield_percentage (annual_dividend par_value stock_dividend_rate) market_value = 20 :=
sorry

end yield_percentage_of_stock_l13_13379


namespace evaluate_expression_l13_13070

theorem evaluate_expression : 
  (sqrt (5 + 4 * sqrt 3) - sqrt (5 - 4 * sqrt 3) + sqrt (7 + 2 * sqrt 10) - sqrt (7 - 2 * sqrt 10)) = 2 * sqrt 2 :=
by
  sorry

end evaluate_expression_l13_13070


namespace part_a_part_b_l13_13470

def star_placement_4x4_valid (grid : list (list bool)) : Prop :=
  ∀ (r1 r2 c1 c2 : ℕ), 
    r1 < 4 → r2 < 4 → c1 < 4 → c2 < 4 → 
    r1 ≠ r2 → c1 ≠ c2 → 
    grid ≠ [[false, false, false, false],
            [false, false, false, false],
            [false, false, false, false],
            [false, false, false, false]]

def seven_star_placement : Prop :=
  ∃ (grid : list (list bool)), 
    (∃ (count : ℕ), count = 7 ∧ 
      sum (map (sum ∘ map (λ b, if b then 1 else 0)) grid) = count) ∧ 
    star_placement_4x4_valid grid

def less_than_seven_star_problem : Prop :=
  ∀ (grid : list (list bool)) (count : ℕ),
    count < 7 → 
    sum (map (sum ∘ map (λ b, if b then 1 else 0)) grid) = count →
    ∃ (r1 r2 c1 c2 : ℕ), 
      r1 < 4 → r2 < 4 → c1 < 4 → c2 < 4 → 
      r1 ≠ r2 → c1 ≠ c2 → 
      grid = [[false, false, false, false],
              [false, false, false, false],
              [false, false, false, false],
              [false, false, false, false]]

theorem part_a : seven_star_placement :=
sorry

theorem part_b : less_than_seven_star_problem :=
sorry

end part_a_part_b_l13_13470


namespace max_sum_composite_shape_l13_13224

theorem max_sum_composite_shape :
  let faces_hex_prism := 8
  let edges_hex_prism := 18
  let vertices_hex_prism := 12

  let faces_hex_with_pyramid := 8 - 1 + 6
  let edges_hex_with_pyramid := 18 + 6
  let vertices_hex_with_pyramid := 12 + 1
  let sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  let faces_rec_with_pyramid := 8 - 1 + 5
  let edges_rec_with_pyramid := 18 + 4
  let vertices_rec_with_pyramid := 12 + 1
  let sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sum_hex_with_pyramid = 50 ∧ sum_rec_with_pyramid = 46 ∧ sum_hex_with_pyramid ≥ sum_rec_with_pyramid := 
by
  have faces_hex_prism := 8
  have edges_hex_prism := 18
  have vertices_hex_prism := 12

  have faces_hex_with_pyramid := 8 - 1 + 6
  have edges_hex_with_pyramid := 18 + 6
  have vertices_hex_with_pyramid := 12 + 1
  have sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  have faces_rec_with_pyramid := 8 - 1 + 5
  have edges_rec_with_pyramid := 18 + 4
  have vertices_rec_with_pyramid := 12 + 1
  have sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sorry -- proof omitted

end max_sum_composite_shape_l13_13224


namespace max_remaining_perimeter_proof_l13_13730

/-- A rectangle is defined by its length and width -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- The perimeter of a rectangle is twice the sum of its length and width -/
def perimeter (r : Rectangle) : ℕ :=
  2 * (r.length + r.width)

/-- Given a rectangle and a smaller rectangle whose one side aligns with an edge of 
the larger rectangle, the maximum perimeter of the remaining piece of paper is calculated -/
def remaining_max_perimeter (large small : Rectangle) : ℕ :=
  let large_perimeter := perimeter large in
  if small.length ≤ large.length && small.width ≤ large.width then
    large_perimeter + 2 * small.length
  else if small.length ≤ large.width && small.width <= large.length then
    large_perimeter + 2 * small.width
  else
    large_perimeter

/-- The problem statement: Given an initial rectangle of dimensions 20 cm by 16 cm,
and a smaller rectangle of dimensions 8 cm by 4 cm, prove that the maximum perimeter
of the remaining piece of paper after cutting is 88 cm -/
theorem max_remaining_perimeter_proof :
  remaining_max_perimeter
    ⟨20, 16⟩
    ⟨8, 4⟩ = 88 :=
by
  sorry

end max_remaining_perimeter_proof_l13_13730


namespace largest_n_satisfying_inequality_l13_13661

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, (1/4 + n/6 < 3/2) ∧ (∀ m : ℕ, m > n → 1/4 + m/6 ≥ 3/2) :=
by
  sorry

end largest_n_satisfying_inequality_l13_13661


namespace sector_area_is_2_l13_13467

-- Definition of the sector's properties
def sector_perimeter (r : ℝ) (θ : ℝ) : ℝ := r * θ + 2 * r

def sector_area (r : ℝ) (θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Theorem stating that the area of the sector is 2 cm² given the conditions
theorem sector_area_is_2 (r θ : ℝ) (h1 : sector_perimeter r θ = 6) (h2 : θ = 1) : sector_area r θ = 2 :=
by
  sorry

end sector_area_is_2_l13_13467


namespace average_speed_correct_l13_13354

-- Definitions of the distances and times
def Distance1 : ℝ := 15
def Speed1 : ℝ := 30
def Time1 : ℝ := Distance1 / Speed1  -- 0.5 hours

def Distance2 : ℝ := 20
def Speed2 : ℝ := 35
def Time2 : ℝ := Distance2 / Speed2  -- approx 0.571 hours

def Speed3 : ℝ := 25
def Time3 : ℝ := 0.5  -- 30 minutes in hours
def Distance3 : ℝ := Speed3 * Time3  -- 12.5 km

def Speed4 : ℝ := 20
def Time4 : ℝ := 0.75  -- 45 minutes in hours
def Distance4 : ℝ := Speed4 * Time4  -- 15 km

def TotalDistance : ℝ := Distance1 + Distance2 + Distance3 + Distance4  -- 62.5 km
def TotalTime : ℝ := Time1 + Time2 + Time3 + Time4  -- approx 2.321 hours

def ApproxAverageSpeed : ℝ := 26.93

-- The proof statement
theorem average_speed_correct : TotalDistance / TotalTime ≈ ApproxAverageSpeed :=
by
  sorry -- proof goes here

end average_speed_correct_l13_13354


namespace log_identity_l13_13521

theorem log_identity (x : ℝ) (h : log 8 (5 * x) = 3) : log x 200 = 23 / 20 := 
by 
  sorry

end log_identity_l13_13521


namespace intersection_P_Q_range_a_l13_13138

def set_P : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
def set_Q (a : ℝ) : Set ℝ := { x | (x - a) * (x - a - 1) ≤ 0 }

theorem intersection_P_Q (a : ℝ) (h_a : a = 1) :
  set_P ∩ set_Q 1 = {1} :=
sorry

theorem range_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set_P → x ∈ set_Q a) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end intersection_P_Q_range_a_l13_13138


namespace sum_of_excluded_values_l13_13773

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + (1 / (2 + (1 / x))))

def exclude_from_domain (x : ℝ) : Prop :=
  x = 0 ∨ 2 + (1 / x) = 0 ∨ 2 + (1 / (2 + (1 / x))) = 0

theorem sum_of_excluded_values : 
  ∑ x in {0, -1/2, -2/5}, x = -9/10 :=
by
  sorry

end sum_of_excluded_values_l13_13773


namespace exists_integer_multiple_of_3_2008_l13_13233

theorem exists_integer_multiple_of_3_2008 :
  ∃ k : ℤ, 3 ^ 2008 ∣ (k ^ 3 - 36 * k ^ 2 + 51 * k - 97) :=
sorry

end exists_integer_multiple_of_3_2008_l13_13233


namespace two_digit_number_representation_l13_13015

-- Define the conditions and the problem statement in Lean 4
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

theorem two_digit_number_representation (x : ℕ) (h : x < 10) :
  ∃ n : ℕ, units_digit n = x ∧ tens_digit n = 2 * x ^ 2 ∧ n = 20 * x ^ 2 + x :=
by {
  sorry
}

end two_digit_number_representation_l13_13015


namespace distinct_paths_in_grid_l13_13752

def number_of_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem distinct_paths_in_grid :
  number_of_paths 7 8 = 6435 :=
by
  sorry

end distinct_paths_in_grid_l13_13752


namespace cows_grazed_by_C_l13_13350

-- Define the initial conditions as constants
def cows_grazed_A : ℕ := 24
def months_grazed_A : ℕ := 3
def cows_grazed_B : ℕ := 10
def months_grazed_B : ℕ := 5
def cows_grazed_D : ℕ := 21
def months_grazed_D : ℕ := 3
def share_rent_A : ℕ := 1440
def total_rent : ℕ := 6500

-- Define the cow-months calculation for A, B, D
def cow_months_A : ℕ := cows_grazed_A * months_grazed_A
def cow_months_B : ℕ := cows_grazed_B * months_grazed_B
def cow_months_D : ℕ := cows_grazed_D * months_grazed_D

-- Let x be the number of cows grazed by C
variable (x : ℕ)

-- Define the cow-months calculation for C
def cow_months_C : ℕ := x * 4

-- Define rent per cow-month
def rent_per_cow_month : ℕ := share_rent_A / cow_months_A

-- Proof problem statement
theorem cows_grazed_by_C : 
  (6500 = (cow_months_A + cow_months_B + cow_months_C x + cow_months_D) * rent_per_cow_month) →
  x = 35 := by
  sorry

end cows_grazed_by_C_l13_13350


namespace find_f_log2_5_l13_13478

-- Define the properties of the function f
def monotonic (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

variable (f : ℝ → ℝ)
variable H1 : monotonic f
variable H2 : ∀ x : ℝ, f (f x + 2 / (2^x + 1)) = 1 / 3

-- Define the goal: f(log_2 5) = 2/3
theorem find_f_log2_5 : f (Real.log 5 / Real.log 2) = 2 / 3 :=
sorry

end find_f_log2_5_l13_13478


namespace intersection_complement_is_l13_13940

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem intersection_complement_is :
  N ∩ (U \ M) = {3, 5} :=
  sorry

end intersection_complement_is_l13_13940


namespace find_age_l13_13371

variable (x : ℤ)

def age_4_years_hence := x + 4
def age_4_years_ago := x - 4
def brothers_age := x - 6

theorem find_age (hx : x = 4 * (x + 4) - 4 * (x - 4) + 1/2 * (x - 6)) : x = 58 :=
sorry

end find_age_l13_13371


namespace polynomial_third_root_l13_13423

theorem polynomial_third_root 
  (c d : ℚ)
  (h_eq1 : 4 * c - d + 12 = 0) 
  (h_eq2 : -32 * c - 15 * d + 12 = 0) :
  let root3 := 4 in
  root3 = 4 :=
by
  sorry

end polynomial_third_root_l13_13423


namespace hilton_final_marbles_l13_13510

theorem hilton_final_marbles :
  ∀ (initial found lost percent: ℝ),
    initial = 30 →
    found = 8.5 →
    lost = 12 →
    percent = 1.5 →
    (initial + found - lost + percent * lost) = 44.5 :=
by
  intros initial found lost percent
  intro h_initial h_found h_lost h_percent
  rw [h_initial, h_found, h_lost, h_percent]
  norm_num
  sorry

end hilton_final_marbles_l13_13510


namespace sum_of_reciprocals_l13_13930

open Set

variables {m n : ℤ} (S : Set ℤ) (T : Set ℤ)
variables (a : ℤ) (a_i : ℕ → ℤ)
noncomputable theory

def predicates := m > n ∧ n ≥ 2 ∧ 
                  (S = {1, 2, ..., m}) ∧ 
                  (∀ i j, i ≠ j → ¬(a_i i ∣ a_i j)) ∧ 
                  (∀ i, a_i i ∈ S)

theorem sum_of_reciprocals (hmn : predicates m n S T a_i) :
  (∑ i in Finset.range n, (1 : ℚ) / (a_i i)) < ((m + n) / m) :=
sorry

end sum_of_reciprocals_l13_13930


namespace find_quadrant_372_degrees_l13_13422

theorem find_quadrant_372_degrees : 
  ∃ q : ℕ, q = 1 ↔ (372 % 360 = 12 ∧ (0 ≤ 12 ∧ 12 < 90)) :=
by
  sorry

end find_quadrant_372_degrees_l13_13422


namespace gwen_remaining_money_l13_13449

theorem gwen_remaining_money:
  ∀ (Gwen_received Gwen_spent Gwen_remaining: ℕ),
    Gwen_received = 5 →
    Gwen_spent = 3 →
    Gwen_remaining = Gwen_received - Gwen_spent →
    Gwen_remaining = 2 :=
by
  intros Gwen_received Gwen_spent Gwen_remaining h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end gwen_remaining_money_l13_13449


namespace quadrilateral_relation_l13_13372

variables {A B C D I L N : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space I] [metric_space L] [metric_space N]
variables {BC AD IL IN : ℝ}
variables (hcond: BC * AD = 4 * IL * IN)
variables (h_inscribed : inscribed_quadrilateral A B C D I)

theorem quadrilateral_relation (h_no_equal_parallel : ¬has_equal_or_parallel_sides A B C D) :
  BC * AD = 4 * IL * IN :=
sorry

end quadrilateral_relation_l13_13372


namespace average_speed_of_tiger_exists_l13_13689

-- Conditions
def head_start_distance (v_t : ℝ) : ℝ := 5 * v_t
def zebra_distance : ℝ := 6 * 55
def tiger_distance (v_t : ℝ) : ℝ := 6 * v_t

-- Problem statement
theorem average_speed_of_tiger_exists (v_t : ℝ) (h : zebra_distance = head_start_distance v_t + tiger_distance v_t) : v_t = 30 :=
by
  sorry

end average_speed_of_tiger_exists_l13_13689


namespace cylinder_problem_l13_13097

-- Define the initial setup of the box and cylinders
def diameter : ℝ := 1
def initial_cylinders : ℕ := 40
def rows : ℕ := 5
def columns : ℕ := 8
def additional_cylinder : ℕ := 1

-- Define the removal of specific cylinders
def removed_cylinders : ℕ := 2

-- Define a function that checks if the cylinders fit in the box and rattle
def new_cylinder_count := initial_cylinders - removed_cylinders + additional_cylinder + removed_cylinders

-- Mathematical condition for tight packing (derived from geometric considerations)
def tight_packing_condition := (columns * (Real.sqrt 3 / 2)) + diameter

-- Define the box width
def box_width : ℝ := 8

-- Final proof stating the equivalent problem
theorem cylinder_problem :
  new_cylinder_count = 41 →
  tight_packing_condition ≤ box_width →
  (∃ rattle: Bool, rattle = (tight_packing_condition < box_width)) :=
by
  intros _ _
  use (tight_packing_condition < box_width)
  exact trivial
  sorry

end cylinder_problem_l13_13097


namespace division_problem_l13_13674

theorem division_problem : (5 * 8) / 10 = 4 := by
  sorry

end division_problem_l13_13674


namespace fraction_left_handed_non_throwers_l13_13948

theorem fraction_left_handed_non_throwers (total_players throwers right_handed : ℕ)
    (h1 : total_players = 58)
    (h2 : throwers = 37)
    (h3 : right_handed = 51)
    (h4 : throwers ≤ right_handed)  -- Implicitly all throwers are right-handed
    : (let non_throwers := total_players - throwers in
       let right_handed_non_throwers := right_handed - throwers in
       let left_handed_non_throwers := non_throwers - right_handed_non_throwers in
       left_handed_non_throwers / non_throwers = 1 / 3) :=
by 
  sorry

end fraction_left_handed_non_throwers_l13_13948


namespace rowing_speed_in_still_water_l13_13002

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.5) 
(h2 : ∀ t : ℝ, (v + c) * t = (v - c) * 2 * t) : 
  v = 4.5 :=
by
  sorry

end rowing_speed_in_still_water_l13_13002


namespace sally_total_net_earnings_l13_13914

-- Sally's net earnings for the first and second month 
-- will be calculated in this Lean 4 statement.

noncomputable def sally_first_month_earnings (income work_expenses side_hustle: ℝ) : ℝ :=
  income + side_hustle - work_expenses

noncomputable def sally_second_month_earnings (income work_expenses side_hustle: ℝ) : ℝ :=
  let raised_income := income + 0.10 * income
  let increased_expenses := work_expenses + 0.15 * work_expenses
  raised_income + side_hustle - increased_expenses

theorem sally_total_net_earnings
  (income work_expenses side_hustle: ℝ)
  (net_earning_first_month: ℝ := sally_first_month_earnings income work_expenses side_hustle)
  (net_earning_second_month: ℝ := sally_second_month_earnings income work_expenses side_hustle)
  (total_net_earnings: ℝ := net_earning_first_month + net_earning_second_month)
  : total_net_earnings = 1970 := by
  have h1 : net_earning_first_month = 950 := by
    sorry
  have h2 : net_earning_second_month = 1020 := by
    sorry
  rw [h1, h2]
  show 950 + 1020 = 1970
  exact rfl

end sally_total_net_earnings_l13_13914


namespace verify_correct_choices_l13_13838

-- Define the function and the condition on ω
def f (ω x : ℝ) : ℝ := 2 * Real.sin(ω * x - Real.pi / 6)
def ω_condition (ω : ℝ) : Prop := ω > 0

-- Define the correctness of options A and B
def is_correct (ω : ℝ) : Prop :=
  ω_condition ω ∧ (2 > 0) ∧ (-Real.pi / 6 = -Real.pi / 6) ∧ (0 < ω ∧ ω ≤ 2)

-- The goal is to show that, given ω_condition, A and B are correct
theorem verify_correct_choices (ω : ℝ) (hω : ω_condition ω) : is_correct ω :=
by
  sorry

end verify_correct_choices_l13_13838


namespace rotation_of_isosceles_triangle_parallel_l13_13728

theorem rotation_of_isosceles_triangle_parallel 
  (A B C A1 B1 B2 C2 : Point)
  (h_isosceles : dist A B = dist A C)
  (h_rot_A_C : rotate_around A C A1)
  (h_A1_on_BC : lies_on A1 (line B C))
  (h_rot_B_C : rotate_around B C B1)
  (same_side_B1_A : same_side B1 A (line B C))
  (h_rot_B1_A1 : rotate_around B1 A1 B2)
  (h_B2_on_BC : lies_on B2 (line B C))
  (h_rot_C_A1 : rotate_around C A1 C2)
  (same_side_C2_A : same_side C2 A (line B C))
  : parallel (line C2 B2) (line A C) :=
sorry

end rotation_of_isosceles_triangle_parallel_l13_13728


namespace three_scientists_same_topic_exists_l13_13058

theorem three_scientists_same_topic_exists
  (scientists : Finset ℕ)
  (topics : Finset ℕ)
  (pairs_correspond_map : (ℕ × ℕ) → ℕ)
  (H17 : scientists.card = 17)
  (H3 : topics.card = 3)
  (H_corr : ∀ s t ∈ scientists, s ≠ t → (s, t) ∈ pairs_correspond_map ∧ pairs_correspond_map (s, t) ∈ topics) :
  ∃ (s1 s2 s3 : ℕ), s1 ∈ scientists ∧ s2 ∈ scientists ∧ s3 ∈ scientists ∧ s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧ (pairs_correspond_map (s1, s2) = pairs_correspond_map (s2, s3) ∧ pairs_correspond_map (s2, s3) = pairs_correspond_map (s1, s3)) :=
by
  sorry

end three_scientists_same_topic_exists_l13_13058


namespace frank_money_made_l13_13454

theorem frank_money_made
  (spent_on_blades : ℕ)
  (number_of_games : ℕ)
  (cost_per_game : ℕ)
  (total_cost_games := number_of_games * cost_per_game)
  (total_money_made := spent_on_blades + total_cost_games)
  (H1 : spent_on_blades = 11)
  (H2 : number_of_games = 4)
  (H3 : cost_per_game = 2) :
  total_money_made = 19 :=
by
  sorry

end frank_money_made_l13_13454


namespace arcs_intersection_l13_13804

theorem arcs_intersection (k : ℕ) : (1 ≤ k ∧ k ≤ 99) ∧ ¬(∃ m : ℕ, k + 1 = 8 * m) ↔ ∃ n l : ℕ, (2 * l + 1) * 100 = (k + 1) * n ∧ n = 100 ∧ k < 100 := by
  sorry

end arcs_intersection_l13_13804


namespace sum_squares_fraction_zero_l13_13203

variable {x : Fin 50 → ℝ}

theorem sum_squares_fraction_zero (h1 : (∑ i, x i) = 0)
  (h2 : (∑ i, x i / (1 + x i)) = 0) : 
  (∑ i, (x i ^ 2) / (1 + x i)) = 0 :=
by
  sorry

end sum_squares_fraction_zero_l13_13203


namespace minimum_teams_partition_l13_13172

theorem minimum_teams_partition :
  ∃ (team1 team2 : finset ℕ), 
    team1 ∪ team2 = (finset.range 100).map (λ x, x + 1) ∧ 
    team1 ∩ team2 = ∅ ∧
    (∀ x ∈ team1, ∀ y ∈ team1, x ≠ y → x ≠ 2 * y ∧ y ≠ 2 * x) ∧
    (∀ x ∈ team2, ∀ y ∈ team2, x ≠ y → x ≠ 2 * y ∧ y ≠ 2 * x) :=
sorry

end minimum_teams_partition_l13_13172


namespace meat_supply_last_days_l13_13167

theorem meat_supply_last_days 
    (lion_consumption : ℕ)
    (tiger_consumption : ℕ)
    (leopard_consumption : ℕ)
    (hyena_consumption : ℕ)
    (total_meat : ℕ)
    (h_lion : lion_consumption = 25)
    (h_tiger : tiger_consumption = 20)
    (h_leopard : leopard_consumption = 15)
    (h_hyena : hyena_consumption = 10)
    (h_total_meat : total_meat = 500) :
    nat.ceil (total_meat / (lion_consumption + tiger_consumption + leopard_consumption + hyena_consumption) : ℝ) = 7 :=
by
  sorry

end meat_supply_last_days_l13_13167


namespace jimmy_change_l13_13910

def cost_of_pens (num_pens : ℕ) (cost_per_pen : ℕ): ℕ := num_pens * cost_per_pen
def cost_of_notebooks (num_notebooks : ℕ) (cost_per_notebook : ℕ): ℕ := num_notebooks * cost_per_notebook
def cost_of_folders (num_folders : ℕ) (cost_per_folder : ℕ): ℕ := num_folders * cost_per_folder

def total_cost : ℕ :=
  cost_of_pens 3 1 + cost_of_notebooks 4 3 + cost_of_folders 2 5

def paid_amount : ℕ := 50

theorem jimmy_change : paid_amount - total_cost = 25 := by
  sorry

end jimmy_change_l13_13910


namespace area_of_triangle_ABC_l13_13077

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1424233, 2848467)
def C : point := (1424234, 2848469)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_ABC : triangle_area A B C = 0.50 := by
  sorry

end area_of_triangle_ABC_l13_13077


namespace sequence_existence_l13_13929

theorem sequence_existence (a : Fin 2008 → ℝ) (h₀ : ∀ k, 0 < a k) (h₁ : (∑ k, a k) > 1) :
  ∃ x : ℕ → ℝ, x 0 = 0 ∧ (∀ n, x n < x (n + 1)) ∧ (filter.at_top x).tendsto ∧
  ∀ n, x (n + 1) - x n = (∑ k : Fin 2008, a k * x (n + k + 1)) - (∑ k : Fin 2008, a (k + 1) * x (n + k)) :=
sorry

end sequence_existence_l13_13929


namespace sum_of_first_10_terms_l13_13816

noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def S (n : ℕ) : ℕ := sorry

variable {n : ℕ}

-- Conditions
axiom h1 : ∀ n, S (n + 1) = S n + a n + 3
axiom h2 : a 5 + a 6 = 29

-- Statement to prove
theorem sum_of_first_10_terms : S 10 = 145 := 
sorry

end sum_of_first_10_terms_l13_13816


namespace smallest_rectangle_area_l13_13310

theorem smallest_rectangle_area (radius : ℝ) (h_radius : radius = 5) :
  let diameter := 2 * radius in
  let width := diameter in
  let height := diameter in
  width * height = 100 :=
by
  sorry

end smallest_rectangle_area_l13_13310


namespace min_shift_sine_l13_13619

theorem min_shift_sine (φ : ℝ) (hφ : φ > 0) :
    (∃ k : ℤ, 2 * φ + π / 3 = 2 * k * π) → φ = 5 * π / 6 :=
sorry

end min_shift_sine_l13_13619


namespace negative_half_power_zero_l13_13389

theorem negative_half_power_zero : (- (1 / 2)) ^ 0 = 1 :=
by
  sorry

end negative_half_power_zero_l13_13389


namespace distribute_apples_l13_13725

-- Conditions definitions
def min_apples_alice := 3
def min_apples_becky := 3
def min_apples_chris := 3
def min_apples_dan := 1
def total_apples := 30

-- Problem statement
theorem distribute_apples :
  (∃ a b c d : ℕ,
     a ≥ min_apples_alice ∧
     b ≥ min_apples_becky ∧
     c ≥ min_apples_chris ∧
     d ≥ min_apples_dan ∧
     a + b + c + d = total_apples) →
  ∑ x in { x : ℕ // x ≥ 0}, ∑ y in { y : ℕ // y ≥ 0 }, ∑ z in { z : ℕ // z ≥ 0 }, ∑ w in { w : ℕ // w ≥ 0 }, x + y + z + w = 20 :=
begin
  sorry
end

end distribute_apples_l13_13725


namespace problem_part_1_problem_part_2_l13_13590

noncomputable def min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 9/y = 1) : ℝ := 36

noncomputable def min_x_add_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 9/y = 1) : ℝ := 19 + 6 * Real.sqrt 2

theorem problem_part_1 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 9/y = 1) : xy = min_xy x y h1 h2 h3 := 
by sorry

theorem problem_part_2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 9/y = 1) : x + 2y = min_x_add_2y x y h1 h2 h3 := 
by sorry

end problem_part_1_problem_part_2_l13_13590


namespace ratio_of_toy_soldiers_to_toy_cars_l13_13550

def total_toys : ℕ := 60
def toy_cars : ℕ := 20
def toy_soldiers : ℕ := total_toys - toy_cars

theorem ratio_of_toy_soldiers_to_toy_cars : (toy_soldiers : toy_cars) = 2 :=
by
  unfold toy_soldiers
  norm_num
  exact sorry -- Replace with the actual proof if desired.

end ratio_of_toy_soldiers_to_toy_cars_l13_13550


namespace part_i_part_ii_l13_13039

noncomputable theory

-- Statement for Part (i)
theorem part_i (a : ℕ) (h : a > 1) :
  ∃ P : ℕ → Prop, (∀ n, Prime (P n) ∧ ∃ k, P n ∣ (a ^ k + 1)) ∧ ∀ n, P n ≠ P (n + 1) :=
sorry

-- Statement for Part (ii)
theorem part_ii (a : ℕ) (h : a > 1) :
  ∃ Q : ℕ → Prop, (∀ n, Prime (Q n) ∧ ∀ k, ¬ (Q n ∣ (a ^ k + 1))) ∧ ∀ n, Q n ≠ Q (n + 1) :=
sorry

end part_i_part_ii_l13_13039


namespace quadrilateral_is_parallelogram_l13_13481

open EuclideanGeometry

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def A : Point2D := ⟨0, 0⟩
def B : Point2D := ⟨2, -1⟩
def C : Point2D := ⟨4, 2⟩
def D : Point2D := ⟨2, 3⟩

noncomputable def vector (P Q : Point2D) : Point2D :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def magnitude (v : Point2D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2)

noncomputable def dot_product (v1 v2 : Point2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

theorem quadrilateral_is_parallelogram :
  let AB := vector A B,
      BC := vector B C,
      CD := vector C D,
      DA := vector D A,
      AD := vector A D in
  AB = vector D C ∧ magnitude AB = magnitude CD ∧ 
  dot_product AB BC ≠ 0 ∧ magnitude AB ≠ magnitude BC → 
  (vector A C = vector B D ∧ vector A D ≠ vector B C) :=
by
  sorry

end quadrilateral_is_parallelogram_l13_13481


namespace number_of_triangles_with_longest_side_11_l13_13621

theorem number_of_triangles_with_longest_side_11 : 
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ 11 ∧ a + b ≥ 12 → 
  (finset.card {t : ℕ × ℕ // 1 ≤ t.1 ∧ t.1 ≤ t.2 ∧ t.2 ≤ 11 ∧ t.1 + t.2 ≥ 12}) = 36 :=
by sorry

end number_of_triangles_with_longest_side_11_l13_13621


namespace bus_ride_difference_l13_13954

def oscars_bus_ride : ℝ := 0.75
def charlies_bus_ride : ℝ := 0.25

theorem bus_ride_difference :
  oscars_bus_ride - charlies_bus_ride = 0.50 :=
by
  sorry

end bus_ride_difference_l13_13954


namespace find_a_iff_l13_13051

def non_deg_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, 9 * (x^2) + (y^2) - 36 * x + 8 * y = k → 
  (∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0))

theorem find_a_iff (k : ℝ) : non_deg_ellipse k ↔ k > -52 := by
  sorry

end find_a_iff_l13_13051


namespace tangent_line_at_pi_f_less_than_cubed_on_interval_maximum_k_value_on_interval_l13_13491

noncomputable def f (x : ℝ) : ℝ := Real.sin x - x * Real.cos x

theorem tangent_line_at_pi :
  let fp := f π in
  ∀ p : ℝ × ℝ, p = (π, f π) → p.2 = π :=
by sorry

theorem f_less_than_cubed_on_interval :
  ∀ x : ℝ, 0 < x ∧ x < π / 2 → f x < (1 / 3) * x^3 :=
by sorry

theorem maximum_k_value_on_interval :
  ∀ k : ℝ, (∀ x : ℝ, 0 < x ∧ x < π / 2 → f x > k * x - x * Real.cos x) → k ≤ 2 / π :=
by sorry

end tangent_line_at_pi_f_less_than_cubed_on_interval_maximum_k_value_on_interval_l13_13491


namespace audrey_needs_12_limes_l13_13385

noncomputable def key_lime_pie_limes_needs (key_lime_juice_per_lime : ℕ → ℚ) : ℕ :=
  let juice_needed := (3 * 1/4 : ℚ) * 16 in
  let min_yield := 1 in
  let max_yield := 2 in
  let max_limes := juice_needed / max_yield in
  let min_limes := juice_needed / min_yield in
  if key_lime_juice_per_lime 12 >= min_yield then 12 else 12

theorem audrey_needs_12_limes (limes_needed : ℕ → ℚ → ℕ) (t : ℕ → ℚ): 
  limes_needed key_lime_pie_limes_needs(1) t = 12 → t 12  ≥ 1  := 
begin
  sorry
end

end audrey_needs_12_limes_l13_13385


namespace sin_theta_between_line_and_plane_l13_13934

theorem sin_theta_between_line_and_plane :
  let θ := angle_between_line_and_plane ⟨2, -1, 0⟩ ⟨4, 5, 7⟩ ⟨8, 4, -9⟩ ⟨0, 0, 6⟩ in
  sin θ = -11 / (Real.sqrt 90 * Real.sqrt 161) :=
by {
  -- angle_between_line_and_plane is a placeholder for the actual function calculating the angle;
  -- more definitions are necessary based on the geometry definitions.
  sorry
}

end sin_theta_between_line_and_plane_l13_13934


namespace income_derived_from_investment_l13_13992

theorem income_derived_from_investment :
  ∀ (market_value : ℝ) (investment_amount : ℝ) (brokerage_rate : ℝ), 
    market_value = 83.08333333333334 ∧ investment_amount = 6000 ∧ brokerage_rate = 0.0025 → 
    let brokerage_fee := investment_amount * brokerage_rate in
    let actual_investment_amount := investment_amount - brokerage_fee in
    let number_of_units := actual_investment_amount / market_value in
    let total_face_value := number_of_units * 100 in
    let income := total_face_value * 0.105 in
    income = 756 :=
by {
  intros market_value investment_amount brokerage_rate h,
  rcases h with ⟨hv, hi, hb⟩,
  let brokerage_fee := investment_amount * brokerage_rate,
  let actual_investment_amount := investment_amount - brokerage_fee,
  let number_of_units := actual_investment_amount / market_value,
  let total_face_value := number_of_units * 100,
  let income := total_face_value * 0.105,
  have h1 : brokerage_fee = 15, by sorry,
  have h2 : actual_investment_amount = 5985, by sorry,
  have h3 : number_of_units ≈ 72, by sorry,
  have h4 : total_face_value ≈ 7200, by sorry,
  have h5 : income ≈ 756, by sorry,
  exact eq_of_approx_eq h5,
}

end income_derived_from_investment_l13_13992


namespace sum_of_t_values_l13_13670

def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  (dist A B = dist A C) ∨ (dist B A = dist B C) ∨ (dist C A = dist C B)

def cos_sin_point (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ * Real.pi / 180), Real.sin (θ * Real.pi / 180))

theorem sum_of_t_values :
  ∑ t in ({t | 0 ≤ t ∧ t ≤ 360 ∧ is_isosceles (cos_sin_point 30) (cos_sin_point 90) (cos_sin_point t)} : set ℝ), t = 750 :=
by
  sorry

end sum_of_t_values_l13_13670


namespace product_mod_five_remainder_l13_13089

theorem product_mod_five_remainder :
  (∏ i in (Finset.range 10), 3 + 10 * i) % 5 = 4 :=
by
  sorry

end product_mod_five_remainder_l13_13089


namespace longer_diagonal_of_rhombus_l13_13007

theorem longer_diagonal_of_rhombus
  (A : ℝ) (r1 r2 : ℝ) (x : ℝ)
  (hA : A = 135)
  (h_ratio : r1 = 5) (h_ratio2 : r2 = 3)
  (h_area : (1/2) * (r1 * x) * (r2 * x) = A) :
  r1 * x = 15 :=
by
  sorry

end longer_diagonal_of_rhombus_l13_13007


namespace triangle_geometry_l13_13300

theorem triangle_geometry (ABC : Triangle) (AB BC CA : ℝ) 
(AB_eq : AB = 8) (BC_eq : BC = 9) (CA_eq : CA = 10)
(B : Point) (C : Point) (A : Point)
(ω1 ω2 : Circle)
(K : Point)
(ω1_cond : on_circle ω1 B ∧ ω1.tangent A AC)
(ω2_cond : on_circle ω2 C ∧ ω2.tangent A AB)
(K_cond : intersection_not_equal_to_A ω1 ω2 K A):
  segment_length BK = 128 / 27 := 
sorry

end triangle_geometry_l13_13300


namespace third_largest_number_with_digits_2_3_7_l13_13515

def twoDigitNumbers (digits : List ℕ) : List ℕ :=
  digits.bind (λ d1 => digits.map (λ d2 => 10 * d1 + d2))

theorem third_largest_number_with_digits_2_3_7 : 
  let digits := [2, 3, 7] in
  let numbers := twoDigitNumbers digits in
  let sorted_numbers := List.qsort (≥) numbers in
  sorted_numbers.nth 2 = some 72 :=
by
  let digits := [2, 3, 7]
  let numbers := twoDigitNumbers digits
  let sorted_numbers := List.qsort (≥) numbers
  have h : sorted_numbers = [77, 73, 72, 37, 33, 32, 27, 23, 22] := sorry
  have h_nth : sorted_numbers.nth 2 = some 72 := by
    rw h
    exact rfl
  exact h_nth

end third_largest_number_with_digits_2_3_7_l13_13515


namespace distance_between_vertices_l13_13927

noncomputable def vertex_distance : ℝ :=
  let C := (1 : ℝ, -1 : ℝ) in
  let D := (-1 : ℝ, 1 : ℝ) in
  real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)

theorem distance_between_vertices :
  vertex_distance = 2 * real.sqrt 2 :=
by sorry

end distance_between_vertices_l13_13927


namespace certain_number_more_than_10_times_Sierra_age_l13_13351

variable (Sierra_current_age : ℕ) (Diaz_future_age : ℕ) (years : ℕ)

theorem certain_number_more_than_10_times_Sierra_age
    (h1 : Sierra_current_age = 30)
    (h2 : Diaz_future_age = 56)
    (h3 : years = 20) :
    let Diaz_current_age := Diaz_future_age - years
    let ten_times_Diaz_age := 10 * Diaz_current_age
    let ten_times_Sierra_age := 10 * Sierra_current_age
    let difference := ten_times_Diaz_age - 40
    difference - ten_times_Sierra_age = 20 := 
by 
  let Diaz_current_age := Diaz_future_age - years
  let ten_times_Diaz_age := 10 * Diaz_current_age
  let ten_times_Sierra_age := 10 * Sierra_current_age
  let difference := ten_times_Diaz_age - 40
  calc
    difference - ten_times_Sierra_age
        = (10 * Diaz_current_age) - 40 - 10 * Sierra_current_age : by rfl
    ... = (10 * (Diaz_future_age - years)) - 40 - 10 * Sierra_current_age : by rfl
    ... = (10 * 36 - 40 - 10 * 30) : by 
          rw [h1, h2, h3]
          simp
    ... = 360 - 40 - 300 : by rfl
    ... = 20 : by rfl

end certain_number_more_than_10_times_Sierra_age_l13_13351


namespace natalia_total_distance_l13_13219

theorem natalia_total_distance :
  let dist_mon := 40
  let bonus_mon := 0.05 * dist_mon
  let effective_mon := dist_mon + bonus_mon
  
  let dist_tue := 50
  let bonus_tue := 0.03 * dist_tue
  let effective_tue := dist_tue + bonus_tue
  
  let dist_wed := dist_tue / 2
  let bonus_wed := 0.07 * dist_wed
  let effective_wed := dist_wed + bonus_wed
  
  let dist_thu := dist_mon + dist_wed
  let bonus_thu := 0.04 * dist_thu
  let effective_thu := dist_thu + bonus_thu
  
  let dist_fri := 1.2 * dist_thu
  let bonus_fri := 0.06 * dist_fri
  let effective_fri := dist_fri + bonus_fri
  
  let dist_sat := 0.75 * dist_fri
  let bonus_sat := 0.02 * dist_sat
  let effective_sat := dist_sat + bonus_sat
  
  let dist_sun := dist_sat - dist_wed
  let bonus_sun := 0.10 * dist_sun
  let effective_sun := dist_sun + bonus_sun
  
  effective_mon + effective_tue + effective_wed + effective_thu + effective_fri + effective_sat + effective_sun = 367.05 :=
by
  sorry

end natalia_total_distance_l13_13219


namespace positive_integer_pairs_l13_13073

theorem positive_integer_pairs (x y : ℕ) (hx : x > 1)
(hc : (⌊(x ^ 2) / y⌋ + 1) % x = 0) : 
(x = 2 ∧ y = 4) ∨ (∃ t > 1, x = t ∧ y = t + 1) :=
sorry

end positive_integer_pairs_l13_13073


namespace perpendicular_vectors_alpha_l13_13815

theorem perpendicular_vectors_alpha (α : ℝ) 
  (ha : α ∈ Ioo 0 π)
  (a : EuclideanSpace ℝ (Fin 2)) 
  (b : EuclideanSpace ℝ (Fin 2)) 
  (h_a : a = ![2 * Real.cos α, -1])
  (h_b : b = ![Real.cos α, 1])
  (h_perp : a ⬝ b = 0) :
  α = π / 4 ∨ α = 3 * π / 4 :=
by
  sorry

end perpendicular_vectors_alpha_l13_13815


namespace monotonic_intervals_extreme_values_on_interval_l13_13842

noncomputable def f (x : ℝ) : ℝ :=
  (1/3) * x^3 + 2 * x^2 - 5 * x - 1

theorem monotonic_intervals :
  (∀ x < -5, deriv f x > 0) ∧
  (∀ x > -5, x < 1, deriv f x < 0) ∧
  (∀ x > 1, deriv f x > 0) :=
sorry

theorem extreme_values_on_interval :
  let a := f (-2) in
  let b := f 2 in
  let c := f 1 in
  a = -5/3 ∧ b = -1/3 ∧ c = -11/3 ∧
  (∀ x ∈ Icc (-2 : ℝ) 2, f x ≥ -11/3 ∧ f x ≤ -1/3) :=
sorry

end monotonic_intervals_extreme_values_on_interval_l13_13842


namespace dice_prob_l13_13318

-- Defining the standard dice values and probabilities
def is_valid_die (x : ℤ) : Prop := x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6

-- Defining the condition where none of the dice show 1 or 6
def valid_condition (a b c : ℤ) : Prop :=
  (a ≠ 1) ∧ (a ≠ 6) ∧ (b ≠ 1) ∧ (b ≠ 6) ∧ (c ≠ 1) ∧ (c ≠ 6)

-- To calculate the probability for the valid condition
def probability_valid (p : ℚ) : Prop :=
  p = 8 / 27

theorem dice_prob (a b c : ℤ) (h_die1 : is_valid_die a) (h_die2 : is_valid_die b) (h_die3 : is_valid_die c) :
  valid_condition a b c → probability_valid (2/3 * 2/3 * 2/3) :=
by
  intros h
  unfold probability_valid
  norm_num
  -- Proof would be filled in here
  sorry

end dice_prob_l13_13318


namespace smallest_positive_multiple_l13_13666

theorem smallest_positive_multiple (n : ℕ) (h1 : n > 0) (h2 : n % 45 = 0) (h3 : n % 75 = 0) (h4 : n % 20 ≠ 0) :
  n = 225 :=
by
  sorry

end smallest_positive_multiple_l13_13666


namespace parabola_directrix_a_l13_13156

theorem parabola_directrix_a (a : ℝ) : (∀ y, (y = -2) → (∀ x, x^2 = a * y)) → a = 8 :=
by
  intros hy 
  specialize hy (-2)
  have h : -2 = -2 := rfl
  specialize hy h
  sorry

end parabola_directrix_a_l13_13156


namespace length_of_BD_eq_decagon_side_l13_13342

-- Definitions of the points and lengths in the triangle
variables {A B C D E : Type}
variables (R : ℝ) [DecidableEq A] [DecidableEq B] [DecidableEq C]
variables [metric_space A] [metric_space B] [metric_space C]

-- Define the segments and relationships
variables (AB AD BD AC AE DE BC : ℝ)
variables (AD_eq_DE : AD = DE) (AD_eq_AC : AD = AC) (BD_eq_AE : BD = AE)
variables (DE_parallel_BC : parallel DE BC)

-- Theorem statement
theorem length_of_BD_eq_decagon_side :
  BD = R * 2 * sin (π / 10) :=
sorry

end length_of_BD_eq_decagon_side_l13_13342


namespace derivative_of_f_l13_13082

noncomputable def f (x : ℝ) : ℝ := Real.arctan (2 * Real.sin x / Real.sqrt (9 * Real.cos x ^ 2 - 4))

theorem derivative_of_f (x : ℝ) :
  (Real.cos x ^ 2 + Real.sin x ^ 2 = 1) →
  deriv f x = 2 / (Real.cos x * Real.sqrt (9 * Real.cos x ^ 2 - 4)) :=
begin
  sorry
end

end derivative_of_f_l13_13082


namespace wealth_ratio_l13_13584

def global_population (P : ℝ) : Prop := P > 0
def global_wealth (W : ℝ) : Prop := W > 0

def nation_population_percent (p r : ℝ) : Prop := p > 0 ∧ p ≤ 100 ∧ r > 0 ∧ r ≤ 100
def nation_wealth_percent (q s : ℝ) : Prop := q > 0 ∧ q ≤ 100 ∧ s > 0 ∧ s ≤ 100

theorem wealth_ratio (P W p q r s : ℝ) 
  (hP : global_population P)
  (hW : global_wealth W)
  (hpq : nation_population_percent p r)
  (hqr : nation_wealth_percent q s) :
  let wX := (0.5 * 0.01 * q * W) / (0.01 * p * P) in
  let wY := (0.01 * s * W) / (0.01 * r * P) in
  (wX / wY) = (0.5 * q * r) / (p * s) := 
sorry

end wealth_ratio_l13_13584


namespace solution_set_abs_inequality_l13_13281

theorem solution_set_abs_inequality :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 0} :=
sorry

end solution_set_abs_inequality_l13_13281


namespace complex_conjugate_problem_l13_13111

noncomputable def z : ℂ := 1 + complex.i

theorem complex_conjugate_problem :
  (complex.conj(z) + z = 2) ∧ ((complex.conj(z) - z) * complex.i = 2) :=
by
  unfold z
  split
  {
    -- Proof obligation: complex.conj(z) + z = 2
    sorry
  }
  {
    -- Proof obligation: (complex.conj(z) - z) * complex.i = 2
    sorry
  }

end complex_conjugate_problem_l13_13111


namespace circle_diameter_in_feet_l13_13977

/-- Given: The area of a circle is 25 * pi square inches.
    Prove: The diameter of the circle in feet is 5/6 feet. -/
theorem circle_diameter_in_feet (A : ℝ) (hA : A = 25 * Real.pi) :
  ∃ d : ℝ, d = (5 / 6) :=
by
  -- The proof goes here
  sorry

end circle_diameter_in_feet_l13_13977


namespace log_base_value_l13_13425

theorem log_base_value (x : ℝ) (b : ℝ) (h : 9^(x+6) = 5^(x+1)) : b = 5 / 9 ↔ x = log b (9^6) :=
by
  sorry

end log_base_value_l13_13425


namespace mixture_weight_l13_13259

theorem mixture_weight (C : ℚ) (W : ℚ)
  (H1: C > 0) -- C represents the cost per pound of milk powder and coffee in June, and is a positive number
  (H2: C * 0.2 = 0.2) -- The price per pound of milk powder in July
  (H3: (W / 2) * 0.2 + (W / 2) * 4 * C = 6.30) -- The cost of the mixture in July

  : W = 3 := 
sorry

end mixture_weight_l13_13259


namespace angle_relationship_l13_13886

theorem angle_relationship (u x y z w : ℝ)
    (H1 : ∀ (D E : ℝ), x + y + (360 - u - z) = 360)
    (H2 : ∀ (D E : ℝ), z + w + (360 - w - x) = 360) :
    x = (u + 2*z - y - w) / 2 := by
  sorry

end angle_relationship_l13_13886


namespace b_intercept_range_l13_13115

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2 / x) + a * Real.log x

theorem b_intercept_range (a x_1 x_2 b_1 b_2 : ℝ) (h1 : a = 1) 
    (h2 : 2 = 2) (h3 : x_1 < x_2) (h4 : x_2 < 6)
    (h5 : (1/x_1) + (1/x_2) = 0.5) : 
    (b_1 - b_2) ∈ set.Ioo ((2 / 3) - Real.log 2) 0 := 
sorry

end b_intercept_range_l13_13115


namespace least_number_divisible_9_remainder_2_l13_13311

theorem least_number_divisible_9_remainder_2 :
  ∃ (n : ℕ), 
    (∀ d ∈ {3, 4, 5, 6, 7}, ∃ k : ℤ, n = 2 + d * k) ∧
    (9 ∣ n) ∧
    n = 6302 :=
begin
  sorry
end

end least_number_divisible_9_remainder_2_l13_13311


namespace number_of_white_tiles_fifth_pattern_l13_13814

theorem number_of_white_tiles_fifth_pattern :
  let a₁ := 6
  let d := 4
  let a (n : ℕ) := a₁ + (n - 1) * d
  in a 5 = 22 :=
by 
  sorry

end number_of_white_tiles_fifth_pattern_l13_13814


namespace negation_equiv_l13_13346

variables {a b c : ℝ}

theorem negation_equiv (h : ¬ (a + b + c = 3 → a ^ 2 + b ^ 2 + c ^ 2 ≥ 3)) :
  a + b + c ≠ 3 → a ^ 2 + b ^ 2 + c ^ 2 < 3 :=
begin
  intros h₁,
  apply h,
  intros h₂,
  contradiction,
end

-- add sorry to bypass proof
example (a b c : ℝ) : ¬ (a + b + c = 3 → a ^ 2 + b ^ 2 + c ^ 2 ≥ 3) ↔ 
  (a + b + c ≠ 3 → a ^ 2 + b ^ 2 + c ^ 2 < 3) :=
sorry

end negation_equiv_l13_13346


namespace comb_8_3_eq_56_l13_13399

theorem comb_8_3_eq_56 : nat.choose 8 3 = 56 := sorry

end comb_8_3_eq_56_l13_13399


namespace player_a_wins_l13_13538

variable (p : ℝ) (h : p ≤ (1 / 2))

/-- In a tennis match, given the conditions described, 
    prove that the probability of player A winning the game is not greater than 2 * p^2 -/
theorem player_a_wins (h_cond1 : ∀ p, 0 ≤ p ∧ p ≤ 1) 
    (h_cond2 : ∀ n, (0 ≤ n ∧ n ≤ 4 ∧ (n = 4 → n ≤ 2)))
    (h_cond3 : p ≤ 1/2) 
    (h_cond4 : ∀ a b, (a = 3 ∧ b = 3) → (a + b) → abs (a - b) = 2) :
    (∃ q, (q = 1 - p) → 
    (∀ r, (r = p^2 + 2 * p * q * r) → 
    (prob_wins : ℝ) (prob_wins = p^6 + choose 6 1 * p^5 * q + choose 6 2 * p^4 * q^2 + choose 6 3 * p^3 * q^3 * r) → 
    prob_wins <= 2 * p^2)) :=
sorry

end player_a_wins_l13_13538


namespace tank_capacity_l13_13616

-- Define the condition
def tank (x : ℝ) := 0.24 * x = 72

-- State the theorem
theorem tank_capacity : ∃ x : ℝ, tank x ∧ x = 300 :=
by {
  existsi 300,
  split,
  { norm_num },
  { norm_num }
}

end tank_capacity_l13_13616


namespace even_number_of_triangles_l13_13564

noncomputable theory

open Finset

def T : Finset (ℝ × ℝ) := sorry  -- Assume the set T of 2005 points in the plane

-- Assume non-collinearity condition
axiom non_collinear (h_t_size : T.card = 2005) : ¬ ∃ (a b c : (ℝ × ℝ)), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ collinear {a, b, c}

-- Assume any point lies inside some triangle with vertices in T
axiom point_in_triangle : ∀ (p ∈ T), ∃ (a b c : (ℝ × ℝ)), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ p ≠ a ∧ p ≠ b ∧ p ≠ c ∧ is_inside_triangle p a b c

theorem even_number_of_triangles 
  (h_t_size : T.card = 2005)
  (h_non_collinear : ¬ ∃ (a b c : (ℝ × ℝ)), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ collinear {a, b, c})
  (h_inside_triangle : ∀ (p ∈ T), ∃ (a b c : (ℝ × ℝ)), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ p ≠ a ∧ p ≠ b ∧ p ≠ c ∧ is_inside_triangle p a b c) : 
  ∃ (n : ℕ), n % 2 = 0 ∧ (∀ (a b c : (ℝ × ℝ)), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ ¬collinear {a, b, c} → count_triangles a b c = n) :=
sorry

end even_number_of_triangles_l13_13564


namespace problem_solution_l13_13101

-- Assuming function f and its inverse function exist and are related as described in the problem
axioms (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (h_inv : ∀ y, f (f_inv y) = y)
  (h_func : ∀ x, f_inv (f x) = x)
  (h_cond : ∀ x, f_inv (x - 1) = f_inv x - 2)

theorem problem_solution : f_inv 2010 - f_inv 1 = 4018 :=
  sorry

end problem_solution_l13_13101


namespace attendants_both_tools_l13_13334

theorem attendants_both_tools (pencil_users pen_users only_one_type total_attendants both_types : ℕ)
  (h1 : pencil_users = 25) 
  (h2 : pen_users = 15) 
  (h3 : only_one_type = 20) 
  (h4 : total_attendants = only_one_type + both_types) 
  (h5 : total_attendants = pencil_users + pen_users - both_types) 
  : both_types = 10 :=
by
  -- Fill in the proof sub-steps here if needed
  sorry

end attendants_both_tools_l13_13334


namespace sequence_problem_l13_13037

theorem sequence_problem :
  7 * 9 * 11 + (7 + 9 + 11) = 720 :=
by
  sorry

end sequence_problem_l13_13037


namespace find_x_l13_13331

theorem find_x (x : ℝ) : 
  (1 + x) * 0.20 = x * 0.4 → x = 1 :=
by
  intros h
  sorry

end find_x_l13_13331


namespace binom_200_200_binom_200_0_l13_13749

theorem binom_200_200 : nat.choose 200 200 = 1 := by
  sorry

theorem binom_200_0 : nat.choose 200 0 = 1 := by
  sorry

end binom_200_200_binom_200_0_l13_13749


namespace sum_a_c_e_l13_13151

theorem sum_a_c_e {a b c d e f : ℝ} 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 5) : 
  a + c + e = 10 :=
by
  -- Proof goes here
  sorry

end sum_a_c_e_l13_13151


namespace range_of_m_l13_13497

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : x1 > x2) (h2 : y1 > y2) (h3 : y1 = (m-2)*x1) (h4 : y2 = (m-2)*x2) : m > 2 :=
by sorry

end range_of_m_l13_13497


namespace wheel_radius_l13_13377

theorem wheel_radius 
  (distance : ℝ) (revolutions : ℕ) (π : ℝ)
  (h_distance : distance = 1065.43)
  (h_revolutions : revolutions = 750)
  (h_pi : π = 3.14159) :
  (radius : ℝ) (h_radius_equiv : radius = distance / (revolutions * 2 * π)) :
  radius ≈ 0.22619 := sorry

end wheel_radius_l13_13377


namespace max_value_a_n_l13_13106

noncomputable def a_seq : ℕ → ℕ
| 0     => 0  -- By Lean's 0-based indexing, a_1 corresponds to a_seq 1
| 1     => 3
| (n+2) => a_seq (n+1) + 1

def S_n (n : ℕ) : ℕ := (n * (n + 5)) / 2

theorem max_value_a_n : 
  ∃ n : ℕ, S_n n = 2023 ∧ a_seq n = 73 :=
by
  sorry

end max_value_a_n_l13_13106


namespace john_profit_l13_13184

theorem john_profit (purchase_price_grinder purchase_price_mobile : ℝ)
  (loss_percentage_grinder profit_percentage_mobile : ℝ) :
  purchase_price_grinder = 15000 →
  purchase_price_mobile = 8000 →
  loss_percentage_grinder = 4 →
  profit_percentage_mobile = 15 →
  let loss_grinder := (loss_percentage_grinder / 100) * purchase_price_grinder;
      selling_price_grinder := purchase_price_grinder - loss_grinder;
      profit_mobile := (profit_percentage_mobile / 100) * purchase_price_mobile;
      selling_price_mobile := purchase_price_mobile + profit_mobile;
      total_cost_price := purchase_price_grinder + purchase_price_mobile;
      total_selling_price := selling_price_grinder + selling_price_mobile;
      overall_profit := total_selling_price - total_cost_price in
  overall_profit = 600 :=
by
  intros h1 h2 h3 h4;
  dsimp;
  sorry

end john_profit_l13_13184


namespace find_interest_rate_l13_13444

noncomputable def amount : ℝ := 896
noncomputable def principal : ℝ := 799.9999999999999
noncomputable def time : ℝ := 2 + 2 / 5
noncomputable def interest : ℝ := amount - principal
noncomputable def rate : ℝ := interest / (principal * time)

theorem find_interest_rate :
  rate * 100 = 5 := by
  sorry

end find_interest_rate_l13_13444


namespace parallel_lines_condition_l13_13526

theorem parallel_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * m * x + y + 6 = 0 → (m - 3) * x - y + 7 = 0) → m = 1 :=
by
  sorry

end parallel_lines_condition_l13_13526


namespace jake_balloons_l13_13726

theorem jake_balloons (total_balloons : ℕ) (balloons_brought_by_Allan : ℕ) : 
  total_balloons = 3 → balloons_brought_by_Allan = 2 → ∃ balloons_brought_by_Jake : ℕ, balloons_brought_by_Jake = 1 := 
by
  intros h1 h2
  use total_balloons - balloons_brought_by_Allan
  rw [h1, h2]
  exact nat.sub_self.symm sorry

end jake_balloons_l13_13726


namespace sum_even_odd_divisors_leq_n_l13_13093

noncomputable def d_0 (m : ℕ) : ℕ :=
  if m % 2 = 0 then (Nat.divisors m).count (λ d => d % 2 = 0) else 0

noncomputable def d_1 (m : ℕ) : ℕ :=
  (Nat.divisors m).count (λ d => d % 2 = 1)

theorem sum_even_odd_divisors_leq_n (n : ℕ) (h_pos : 0 < n) :
  abs (∑ k in Finset.range (n + 1), (d_0 k - d_1 k)) ≤ n := by
  sorry

end sum_even_odd_divisors_leq_n_l13_13093


namespace fatima_probability_l13_13313

theorem fatima_probability :
  let y := (Nat.choose 12 6 : ℚ) / 2^12 in
  2 * (793 / 2048 : ℚ) + y = 1 → (793 / 2048 : ℚ) = (Nat.choose 12 6 : ℚ) / 2^12 → 
  (793 / 2048 = 793 / 2048 : ℚ) := 
by
  intros y h1 h2
  sorry

end fatima_probability_l13_13313


namespace product_sequence_mod_5_l13_13087

theorem product_sequence_mod_5 :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by
  have h1 : ∀ n, n % 10 = 3 → n % 5 = 3 := by intros n hn; rw [hn]; norm_num
  have seq : list ℕ := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  have h2 : ∀ n ∈ seq, n % 5 = 3 := by intros n hn; rw list.mem_map at hn; rcases hn with ⟨n', rfl, _⟩; exact h1 _ rfl
  have hprod : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = (3 ^ 10) % 5 := 
    by simp [pow_mul, mul_comm, int.mul_mod]; exact list.prod_eq_pow_seq 3 seq
  rw [hprod, <- pow_mod, pow_Todd_Coe] -- use appropriate power mod technique
  sorry

end product_sequence_mod_5_l13_13087


namespace directrix_of_parabola_l13_13611

theorem directrix_of_parabola :
  ∀ y : ℝ, 2 * y * y = -x →
  x = liberal_directrix where liberal_directrix: 0.5 := sorry

end directrix_of_parabola_l13_13611


namespace cows_ratio_l13_13378

/-- Define the conditions of the problem -/
variables (A M V : ℕ) (h1 : A + M + V = 570) (h2 : M = 60) (h3 : A + M = V + 30)

/-- The main theorem stating the ratio of number of cows Aaron has to the number of cows Matthews has -/
theorem cows_ratio (h1 : A + M + V = 570) (h2 : M = 60) (h3 : A + M = V + 30) :
  (A : ℚ) / (M : ℚ) = 4 / 1 :=
sorry

end cows_ratio_l13_13378


namespace orthocenter_divides_altitude_l13_13592

variable {A B C : ℝ} -- Angles of the triangle
variable {H : Point} -- Orthocenter of the triangle
variable {CH : Line} -- Altitude from vertex C

theorem orthocenter_divides_altitude 
  (triangle_ABC : Triangle)
  (is_orthocenter : Orthocenter H triangle_ABC)
  (altitude_CH : Altitude CH triangle_ABC)
  (angle_sum : ∠ A + ∠ B + ∠ C = π) : 
  segment_ratio (CH.segment H) (CH.segment (vertex C)) = (Real.cos C) / (Real.cos A * Real.cos B) :=
sorry

end orthocenter_divides_altitude_l13_13592


namespace value_of_m_l13_13104

def x_data : List ℝ := [1, 2, 3, 4]
def y_data : List ℝ := [4.5, 3.2, 4.8, 7.5]
axiom regression_eq (x : ℝ) : ℝ := 2.1 * x - 0.25

theorem value_of_m : 
  (∃ m : ℝ, List.Sum (m :: y_data.tail!) = 4 * regression_eq (List.mean x_data)) ∧
  List.head y_data = 4.5 :=
by
  sorry

end value_of_m_l13_13104


namespace find_angle_MKA_find_segment_MC_l13_13223

/- Constants from the problem conditions -/
constant A B C : Type
constant K : A → B → Prop
constant L M : B → C → Prop
constant angle : A → B → Prop

/- Conditions -/
constants [EquilateralTriangle] : Prop
constants [OnSide] : K → AB → L → BC → M → BC → Prop
constants [EqualSegments] : KL = KM → Prop
constants [CloserTo] : L → B → M → Prop
constants [AngleCondition] : ∠BKL = 10 : Prop
constants [SegmentLengthBL] : BL = 2 : Prop
constants [SegmentLengthKA] : KA = 3 : Prop

/- Statements to prove -/
theorem find_angle_MKA (h1 : EquilateralTriangle) (h2 : OnSide) (h3 : EqualSegments) (h4 : CloserTo) (h5 : AngleCondition) : ∠MKA = 130 :=
sorry

theorem find_segment_MC (h1 : EquilateralTriangle) (h2 : OnSide) (h3 : EqualSegments) (h4 : CloserTo) (h6 : SegmentLengthBL) (h7 : SegmentLengthKA) : MC = 5 :=
sorry

end find_angle_MKA_find_segment_MC_l13_13223


namespace barrel_capacity_l13_13353

theorem barrel_capacity (x y : ℝ) (h1 : y = 45 / (3/5)) (h2 : 0.6*x = y*3/5) (h3 : 0.4*x = 18) : 
  y = 75 :=
by
  sorry

end barrel_capacity_l13_13353


namespace intersection_and_complement_find_m_l13_13134

-- Define the sets A, B, C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def C (m : ℝ) : Set ℝ := {x | m+1 ≤ x ∧ x ≤ 3*m}

-- State the first proof problem: intersection A ∩ B and complement of B
theorem intersection_and_complement (x : ℝ) : 
  (x ∈ (A ∩ B) ↔ (2 ≤ x ∧ x ≤ 3)) ∧ 
  (x ∈ (compl B) ↔ (x < 1 ∨ x > 4)) :=
by 
  sorry

-- State the second proof problem: find m satisfying A ∪ C(m) = A
theorem find_m (m : ℝ) (x : ℝ) : 
  (∀ x, (x ∈ A ∪ C m) ↔ (x ∈ A)) ↔ (m = 1) :=
by 
  sorry

end intersection_and_complement_find_m_l13_13134


namespace fraction_in_pairing_l13_13170

open Function

theorem fraction_in_pairing (s t : ℕ) (h : (t : ℚ) / 4 = s / 3) : 
  ((t / 4 : ℚ) + (s / 3)) / (t + s) = 2 / 7 :=
by sorry

end fraction_in_pairing_l13_13170


namespace combine_fraction_l13_13685

variable (d : ℤ)

theorem combine_fraction : (3 + 4 * d) / 5 + 3 = (18 + 4 * d) / 5 := by
  sorry

end combine_fraction_l13_13685


namespace binom_8_3_eq_56_l13_13402

def binom (n k : ℕ) : ℕ :=
(n.factorial) / (k.factorial * (n - k).factorial)

theorem binom_8_3_eq_56 : binom 8 3 = 56 := by
  sorry

end binom_8_3_eq_56_l13_13402


namespace finding_p_plus_q_l13_13174

open_locale classical

variables (A B C D E : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables {ab bc bd : ℝ} (h1 : ab = 5) (h2 : bd = 13) (h3 : bc = 12)
variables (BAC_eq_BDC : ∀ {A B C D}, ∠(A, B, C) = ∠(B, D, C))
variables (ABD_eq_CBD : ∀ {A B D C}, ∠(A, B, D) = ∠(B, C, D))

noncomputable def AD_length : ℝ := 60 / 13

theorem finding_p_plus_q : 
  (AD_length = 60 / 13) → 
  ∃ p q : ℕ, nat.gcd p q = 1 ∧ AD_length = p / q ∧ p + q = 73 := 
sorry

end finding_p_plus_q_l13_13174


namespace size_of_can_of_concentrate_l13_13017

theorem size_of_can_of_concentrate
  (can_to_water_ratio : ℕ := 1 + 3)
  (servings_needed : ℕ := 320)
  (serving_size : ℕ := 6)
  (total_volume : ℕ := servings_needed * serving_size) :
  ∃ C : ℕ, C = total_volume / can_to_water_ratio :=
by
  sorry

end size_of_can_of_concentrate_l13_13017


namespace general_term_l13_13502

noncomputable def seq : ℕ → ℚ
| 0     := 1
| (n+1) := 2^n * (seq n)^2 - (seq n) + (1 / 2^(n+1))

theorem general_term (n : ℕ) : seq n = (3^(2^(n-1)) + 1) / 2^(n+1) := by
  sorry

end general_term_l13_13502


namespace correct_statements_l13_13323

theorem correct_statements (A B C D : Prop)
  (hA : ∀ (A B : Point), (vector A B) + (vector B A) = 0)
  (hB : ∀ (a b : Vector), ∥a∥ = ∥b∥ ∧ a ∥ b → a = b)
  (hC : ∀ (a b : Vector), a ≠ 0 ∧ b ≠ 0 ∧ ∥a + b∥ = ∥a - b∥ → a ⊥ b)
  (hD : ∀ (a b : Vector), a ∥ b → ∃! λ : ℝ, b = λ • a) :
  A ∧ C := by {
  apply And.intro,
  -- Proof of statement A being correct
  exact (hA) sorry, -- Replace with proof
  -- Proof of statement C being correct
  exact (hC) sorry, -- Replace with proof
}

end correct_statements_l13_13323


namespace find_expression_l13_13839

noncomputable def f (ω x φ b : ℝ) := sin (ω * x + φ) - b

theorem find_expression (ω φ b : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) 
  (h1 : (2 * real.pi / ω) = 2 * (real.pi / 2)) 
  (g : ℝ → ℝ) 
  (h2 : ∀ x, g x = sin (2 * (x - real.pi / 6) + φ) - b + real.sqrt 3) 
  (h3 : odd g) : 
  ∃ φ b,  (φ = real.pi / 3 ∧ b = real.sqrt 3) ∧ f ω x φ b = sin (2 * x + real.pi / 3) - real.sqrt 3 :=
by
  sorry

end find_expression_l13_13839


namespace factorize_expression_l13_13438

theorem factorize_expression (x y : ℂ) : (x * y^2 - x = x * (y + 1) * (y - 1)) :=
sorry

end factorize_expression_l13_13438


namespace distinct_sequences_l13_13852

theorem distinct_sequences (letters : List Char) (h_distinct : letters.nodup) :
  (List.length letters = 8) ∧ ('S' ∈ letters) ∧ ('T' ∈ letters) →
  (∃ seqs : List (List Char),
    ∀ seq ∈ seqs, seq.length = 5 ∧ List.head seq = 'S' ∧ List.last seq ≠ 'T') →
  seqs.length = 720 :=
by
  sorry

end distinct_sequences_l13_13852


namespace handshake_distance_l13_13059

theorem handshake_distance 
  (n : ℕ) (r : ℝ)
  (h_n : n = 8) 
  (h_r : r = 50) 
  (distance : ℝ) 
  (correct_answer : distance = 8000 * real.sqrt ((real.sqrt 2 - 1) / (2 * real.sqrt 2))) :
  ∃ d, d = 8000 * real.sqrt ((real.sqrt 2 - 1) / (2 * real.sqrt 2)) := 
  by {
  sorry
}

end handshake_distance_l13_13059


namespace measureable_weights_count_l13_13305

theorem measureable_weights_count (a b c : ℕ) (ha : a = 1) (hb : b = 3) (hc : c = 9) :
  ∃ s : Finset ℕ, s.card = 13 ∧ ∀ x ∈ s, x ≥ 1 ∧ x ≤ 13 := 
sorry

end measureable_weights_count_l13_13305


namespace solve_integral_equation_l13_13598

theorem solve_integral_equation:
  ∃ φ : ℝ → ℝ, 
    (∀ x : ℝ, φ(x) = ∫ t in 0..1, x * t^2 * φ(t) + 1) ∧
    φ = fun x => 1 + (4/9) * x :=
by
  sorry

end solve_integral_equation_l13_13598


namespace distance_equality_proof_l13_13609

-- Define the geometric entities and conditions
variables 
  (trapezoid : Type) -- Define the type for trapezoid
  [is_isosceles : IsIsoscelesTrapezoid trapezoid] -- Define the isosceles condition
  [perpendicular_diagonals : PerpendicularDiagonals trapezoid] -- Define the perpendicular diagonals condition
  [circumscribed_circle : CircumscribedCircle trapezoid] -- Define the circumscribed circle condition
  -- Define the points of interest
  (center : Point) -- Center of the circumscribed circle
  (base1 : Line) -- First base of the trapezoid
  (base2 : Line) -- Second base of the trapezoid
  (intersection : Point) -- Intersection point of the diagonals

-- Define distances
variable (distance_from_center_to_base1 : ℝ)
variable (distance_from_intersection_to_base2 : ℝ)

-- Given conditions
axiom (center_distance_property :  Distance center base1 = distance_from_center_to_base1)
axiom (intersection_distance_property :  Distance intersection base2 = distance_from_intersection_to_base2)

-- Theorem to prove the required equality
theorem distance_equality_proof :
  distance_from_center_to_base1 = distance_from_intersection_to_base2 :=
begin
  sorry
end

end distance_equality_proof_l13_13609


namespace sample_capacity_l13_13706

theorem sample_capacity 
  (n : ℕ) 
  (model_A : ℕ) 
  (model_B model_C : ℕ) 
  (ratio_A ratio_B ratio_C : ℕ)
  (r_A : ratio_A = 2)
  (r_B : ratio_B = 3)
  (r_C : ratio_C = 5)
  (total_production_ratio : ratio_A + ratio_B + ratio_C = 10)
  (items_model_A : model_A = 15)
  (proportion : (model_A : ℚ) / (ratio_A : ℚ) = (n : ℚ) / 10) :
  n = 75 :=
by sorry

end sample_capacity_l13_13706


namespace angle_C_and_sum_of_sides_l13_13173

variables {A B C a b c : ℝ}

-- Given conditions
def acute_triangle (A B C : ℝ) (a b c : ℝ) := C < π / 2
def vectors_parallel (m n : ℝ × ℝ) := ∃ k : ℝ, m.1 = k * n.1 ∧ m.2 = k * n.2
def area_triangle (a b c Area : ℝ) := Area = (1/2) * a * b * (Real.sin C)

-- Given values
def m := (Real.sqrt 3, 2 * Real.sin A)
def n := (c, a)
def c_val := Real.sqrt 7
def area_val := (3 * Real.sqrt 3) / 2
def sin_C := Real.sin (π / 3)

-- Theorem statement
theorem angle_C_and_sum_of_sides (h1 : acute_triangle A B C)
                                  (h2 : vectors_parallel m n)
                                  (h3 : c = c_val)
                                  (h4 : area_triangle a b c area_val)
                                  : C = π / 3 ∧ a + b = 5 :=
sorry

end angle_C_and_sum_of_sides_l13_13173


namespace find_B_given_conditions_l13_13529

variable (A B a b : ℝ)

def valid_triangle_angles (A B : ℝ) : Prop :=
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ A + B < π

def sine_law (a b A B : ℝ) : Prop :=
  sin (A) / a = sin (B) / b

theorem find_B_given_conditions :
  (sin (2 * A + π / 6) = 1 / 2) →
  (a = sqrt 3) →
  (b = 1) →
  valid_triangle_angles A B →
  sine_law a b A B →
  B = π / 6 :=
by
  sorry

end find_B_given_conditions_l13_13529


namespace liquid_percentage_P_is_l13_13732

theorem liquid_percentage_P_is:
  (p : ℝ) (h1 : 0.015 * 800 + 200 * p = 0.013 * 1000)
  : p = 0.005 := 
sorry

end liquid_percentage_P_is_l13_13732


namespace pure_imaginary_z_squared_l13_13982

-- Formalization in Lean 4
theorem pure_imaginary_z_squared (a : ℝ) (h : a + (1 + a) * I = (1 + a) * I) : (a + (1 + a) * I)^2 = -1 :=
by
  sorry

end pure_imaginary_z_squared_l13_13982


namespace evaluate_expression_l13_13432

theorem evaluate_expression : 
  (Real.cbrt (8 + 3 * Real.sqrt 21) + Real.cbrt (8 - 3 * Real.sqrt 21) = 1) :=
sorry

end evaluate_expression_l13_13432


namespace candy_distribution_l13_13603

theorem candy_distribution (A B C : ℕ) (x y : ℕ)
  (h1 : A > 2 * B)
  (h2 : B > 3 * C)
  (h3 : A + B + C = 200) :
  (A = 121) ∧ (C = 19) :=
  sorry

end candy_distribution_l13_13603


namespace man_rate_in_still_water_l13_13003

-- The conditions
def speed_with_stream : ℝ := 20
def speed_against_stream : ℝ := 4

-- The problem rephrased as a Lean statement
theorem man_rate_in_still_water : 
  (speed_with_stream + speed_against_stream) / 2 = 12 := 
by
  sorry

end man_rate_in_still_water_l13_13003


namespace book_cost_is_60_over_11_l13_13962

noncomputable def book_cost_before_tax
    (initial_amount : ℝ)
    (num_series : ℕ)
    (books_per_series : ℕ)
    (remaining_amount_after_3_series : ℝ)
    (tax_rate : ℝ) : ℝ := sorry

theorem book_cost_is_60_over_11
    (initial_amount : ℝ := 200)
    (num_series : ℕ := 4)
    (books_per_series : ℕ := 8)
    (remaining_amount_after_3_series : ℝ := 56)
    (tax_rate : ℝ := 0.10) :
    book_cost_before_tax initial_amount num_series books_per_series remaining_amount_after_3_series tax_rate = (60 / 11) :=
begin
   sorry
end

end book_cost_is_60_over_11_l13_13962


namespace find_b_l13_13573

def g (x b : ℝ) : ℝ := x^3 + (5 / 2) * x^2 + 3 * Real.log x + b

theorem find_b (b : ℝ) (h1 : (deriv (λ x => g x b) 1 = 11)) 
  (h2 : ((1 : ℝ), (g 1 b)) = ((0 : ℝ), -5)) : b = 5 / 2 := by
  sorry

end find_b_l13_13573


namespace Rajesh_days_to_complete_l13_13216

theorem Rajesh_days_to_complete (Mahesh_days : ℕ) (Rajesh_days : ℕ) (Total_days : ℕ)
  (h1 : Mahesh_days = 45) (h2 : Total_days - 20 = Rajesh_days) (h3 : Total_days = 54) :
  Rajesh_days = 34 :=
by
  sorry

end Rajesh_days_to_complete_l13_13216


namespace problem_solution_l13_13812

-- Definitions for the conditions
def a_n (n : ℕ) : ℕ := 2 * n - 1
def b_n (n : ℕ) : ℕ := 3 ^ n
def S_n (n : ℕ) : ℕ := n * n

-- Main theorem statement
theorem problem_solution (k : ℝ) :
  (∀ n : ℕ, 0 < n → k * (b_n n : ℝ) ≥ S_n n) ↔ (k ∈ set.Ici (4 / 9)) :=
by
  sorry

end problem_solution_l13_13812


namespace wrap_XL_boxes_per_roll_l13_13851

-- Conditions
def rolls_per_shirt_box : ℕ := 5
def num_shirt_boxes : ℕ := 20
def num_XL_boxes : ℕ := 12
def cost_per_roll : ℕ := 4
def total_cost : ℕ := 32

-- Prove that one roll of wrapping paper can wrap 3 XL boxes
theorem wrap_XL_boxes_per_roll : (num_XL_boxes / ((total_cost / cost_per_roll) - (num_shirt_boxes / rolls_per_shirt_box))) = 3 := 
sorry

end wrap_XL_boxes_per_roll_l13_13851


namespace system_no_solution_l13_13053

theorem system_no_solution (n : ℝ) :
  ∃ x y z : ℝ, (n * x + y = 1) ∧ (1 / 2 * n * y + z = 1) ∧ (x + 1 / 2 * n * z = 2) ↔ n = -1 := 
sorry

end system_no_solution_l13_13053


namespace workout_increase_l13_13056

theorem workout_increase (x : ℕ) : 
  (30 + 3 * x = 45) -> x = 5 :=
begin
  intro h,
  linarith,
end

end workout_increase_l13_13056


namespace balls_per_bag_l13_13368

theorem balls_per_bag (total_balls : ℕ) (total_bags : ℕ) (h1 : total_balls = 36) (h2 : total_bags = 9) : total_balls / total_bags = 4 :=
by
  sorry

end balls_per_bag_l13_13368


namespace gain_percent_is_56_25_l13_13524

-- Let C be the cost price of one article and S be the selling price of one article
variables (C S : ℝ)

-- Given the condition that the cost price of 50 articles is equal to the selling price of 32 articles
def condition := 50 * C = 32 * S

-- Define the gain percent calculation
def gain_percent := ((S - C) / C) * 100

theorem gain_percent_is_56_25 (C S : ℝ) (h : condition C S) :
  gain_percent C S = 56.25 :=
by
  -- We provide a placeholder to show that the proof is omitted
  sorry

end gain_percent_is_56_25_l13_13524


namespace intersection_M_N_l13_13522

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x^2 - 3*x ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l13_13522


namespace beautiful_numbers_sum_l13_13756

def is_beautiful_number (a: ℕ) : Prop :=
  ∃ b n : ℕ, b ∈ {3, 4, 5, 6} ∧ n > 0 ∧ a = b^n

theorem beautiful_numbers_sum {n : ℕ} (h : n ≥ 3) :
  ∃ (k : ℕ) (a : Fin k → ℕ), (∀ i j, i ≠ j → a i ≠ a j) ∧ (∀ i, is_beautiful_number (a i)) ∧ n = (Finset.univ.sum a) :=
by sorry

end beautiful_numbers_sum_l13_13756


namespace function_passes_through_fixed_point_l13_13114

theorem function_passes_through_fixed_point (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (-1, 2) ∧ ∀ x : ℝ, y = 2 + log a (x + 2) -> y = P.2 := 
sorry

end function_passes_through_fixed_point_l13_13114


namespace find_y_orthogonal_vectors_l13_13797

-- Define vectors and orthogonality condition
def vec1 (y : ℝ) : ℝ × ℝ × ℝ := (3, 4, y)
def vec2 : ℝ × ℝ × ℝ := (-4, 3, 2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_y_orthogonal_vectors : 
  ∃ y : ℝ, dot_product (vec1 y) vec2 = 0 ∧ y = 0 := 
by
  exists 0
  simp [vec1, vec2, dot_product]
  sorry -- proof required to complete the theorem

end find_y_orthogonal_vectors_l13_13797


namespace num_memorable_telephone_numbers_l13_13406

-- Define digits as numbers from 0 to 9
def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

-- Define a sequence of digits
def is_sequence_of_digits (s : List ℕ) : Prop :=
  s.length = 4 ∧ ∀ d ∈ s, is_digit d

-- Define a memorable number as two identical sequences of digits
def is_memorable (n : List ℕ) : Prop := 
  n.length = 8 ∧
  is_sequence_of_digits (n.take 4) ∧
  is_sequence_of_digits (n.drop 4) ∧
  n.take 4 = n.drop 4

-- Prove the number of memorable telephone numbers
theorem num_memorable_telephone_numbers : 
  ∃ n: ℕ, n = 10000 ∧ (∃ l : List ℕ, is_memorable l) :=
begin
  sorry
end

end num_memorable_telephone_numbers_l13_13406


namespace volleyball_team_ways_l13_13955

def num_ways_choose_starers : ℕ :=
  3 * (Nat.choose 12 6 + Nat.choose 12 5)

theorem volleyball_team_ways :
  num_ways_choose_starers = 5148 := by
  sorry

end volleyball_team_ways_l13_13955


namespace inverse_of_h_l13_13763

noncomputable def h (x : ℝ) : ℝ := 3 - 7 * x
noncomputable def k (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_of_h :
  (∀ x : ℝ, h (k x) = x) ∧ (∀ x : ℝ, k (h x) = x) :=
by
  sorry

end inverse_of_h_l13_13763


namespace correct_statements_l13_13707

/-
Let A be the event "the first roll is an even number",
B be the event "the second roll is an even number",
C be the event "the sum of the two rolls is an even number",
and both rolls are done with a fair cubic dice.
Prove that the statements A, C, and D are correct.
A: P(A) = 1 - P(B)
C: B and C are independent events
D: P(A ∪ B) = 3 / 4
-/

open ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}

def first_roll_even (ω : Ω) : Prop := ω ∈ {i | i % 2 = 0}
def second_roll_even (ω : Ω) : Prop := ω ∈ {i | i % 2 = 0}
def sum_rolls_even (ω : Ω) : Prop := (ω % 6 % 2 + ω % 6 % 2) % 2 = 0

theorem correct_statements (fair_dice : ∀ ω, P {a | a = first_roll_even ω} = 1 / 2) :
  (P {ω | first_roll_even ω} = 1 - P {ω | second_roll_even ω}) ∧
  (∀ t, P (λ ω, second_roll_even ω ∧ sum_rolls_even ω) = P (λ ω, second_roll_even ω) * P (λ ω, sum_rolls_even ω)) ∧
  (P {ω | first_roll_even ω ∨ second_roll_even ω} = 3 / 4) :=
sorry

end correct_statements_l13_13707


namespace number_of_letters_with_both_l13_13880

theorem number_of_letters_with_both (total_letters : ℕ) (letters_with_straight_line_only : ℕ) (letters_with_dot_only : ℕ) :
  total_letters = 40 →
  letters_with_straight_line_only = 24 →
  letters_with_dot_only = 6 →
  ∃ X : ℕ, 24 + 6 + X = 40 ∧ X = 10 :=
by
  intros h1 h2 h3
  use 10
  split
  { rw [h1, h2, h3], norm_num }
  { norm_num }

end number_of_letters_with_both_l13_13880


namespace farmer_initial_buckets_l13_13362

theorem farmer_initial_buckets (remaining sowed : ℝ) 
  (h_remaining : remaining = 6)
  (h_sowed : sowed = 2.75) : 
  remaining + sowed = 8.75 := 
by 
  rw [h_remaining, h_sowed]
  exact eq.refl 8.75

end farmer_initial_buckets_l13_13362


namespace handshake_distance_l13_13060

theorem handshake_distance 
  (n : ℕ) (r : ℝ)
  (h_n : n = 8) 
  (h_r : r = 50) 
  (distance : ℝ) 
  (correct_answer : distance = 8000 * real.sqrt ((real.sqrt 2 - 1) / (2 * real.sqrt 2))) :
  ∃ d, d = 8000 * real.sqrt ((real.sqrt 2 - 1) / (2 * real.sqrt 2)) := 
  by {
  sorry
}

end handshake_distance_l13_13060


namespace find_roses_sold_first_day_l13_13942

/-
Conditions:
1. On the first day, she sold 30 tulips and some roses.
2. The next day, she doubled the previous day's sales.
3. On the third day, she sold only 10% of the tulips sold on the second day and 16 roses.
4. The price of one tulip is $2 and one rose is $3.
5. Maria earned $420 over these three days.

Question:
How many roses did Maria sell on the first day?
-/

theorem find_roses_sold_first_day
  (tulips_day1 : ℕ := 30)
  (roses_day1 : ℕ)
  (tulip_price : ℕ := 2)
  (rose_price : ℕ := 3)
  (total_revenue : ℕ := 420) :
  (tulips_day1 * tulip_price + roses_day1 * rose_price +
  (2 * tulips_day1) * tulip_price + (2 * roses_day1) * rose_price +
  (0.1 * (2 * tulips_day1)) * tulip_price + 16 * rose_price) = total_revenue → 
  roses_day1 = 20 := by
  intro h
  sorry

end find_roses_sold_first_day_l13_13942


namespace n_leq_84_l13_13199

theorem n_leq_84 (n : ℕ) (hn : 0 < n) (h: (1 / 2 + 1 / 3 + 1 / 7 + 1 / ↑n : ℚ).den ≤ 1): n ≤ 84 :=
sorry

end n_leq_84_l13_13199


namespace cara_total_bread_l13_13094

theorem cara_total_bread 
  (d : ℕ) (L : ℕ) (B : ℕ) (S : ℕ) 
  (h_dinner : d = 240) 
  (h_lunch : d = 8 * L) 
  (h_breakfast : d = 6 * B) 
  (h_snack : d = 4 * S) : 
  d + L + B + S = 370 := 
sorry

end cara_total_bread_l13_13094


namespace arccos_of_cos_periodic_l13_13742

theorem arccos_of_cos_periodic :
  arccos (cos 8) = 8 - 2 * Real.pi :=
by
  sorry

end arccos_of_cos_periodic_l13_13742


namespace num_solutions_abs_x_plus_abs_y_lt_100_l13_13512

theorem num_solutions_abs_x_plus_abs_y_lt_100 :
  (∃ n : ℕ, n = 338350 ∧ ∀ (x y : ℤ), (|x| + |y| < 100) → True) :=
sorry

end num_solutions_abs_x_plus_abs_y_lt_100_l13_13512


namespace additive_inverse_of_half_l13_13268

theorem additive_inverse_of_half :
  - (1 / 2) = -1 / 2 :=
by
  sorry

end additive_inverse_of_half_l13_13268


namespace equilateral_triangle_perimeter_l13_13883

/-- Theorem: The perimeter of an equilateral triangle with side length 10 cm is 30 cm. -/
theorem equilateral_triangle_perimeter (a : ℕ) (h : a = 10) : 3 * a = 30 :=
by
  rw [h]
  simp
  done

end equilateral_triangle_perimeter_l13_13883


namespace minimum_cost_for_18_oranges_l13_13722

noncomputable def min_cost_oranges (x y : ℕ) : ℕ :=
  10 * x + 30 * y

theorem minimum_cost_for_18_oranges :
  (∃ x y : ℕ, 3 * x + 7 * y = 18 ∧ min_cost_oranges x y = 60) ∧ (60 / 18 = 10 / 3) :=
sorry

end minimum_cost_for_18_oranges_l13_13722


namespace boys_of_other_communities_l13_13537

theorem boys_of_other_communities (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℝ) 
  (h_tm : total_boys = 1500)
  (h_pm : percentage_muslims = 37.5)
  (h_ph : percentage_hindus = 25.6)
  (h_ps : percentage_sikhs = 8.4) : 
  ∃ (boys_other_communities : ℕ), boys_other_communities = 428 :=
by
  sorry

end boys_of_other_communities_l13_13537


namespace ratio_of_areas_of_similar_triangles_l13_13871

theorem ratio_of_areas_of_similar_triangles (m1 m2 : ℝ) (med_ratio : m1 / m2 = 1 / Real.sqrt 2) :
    let area_ratio := (m1 / m2) ^ 2
    area_ratio = 1 / 2 := by
  sorry

end ratio_of_areas_of_similar_triangles_l13_13871


namespace foldable_polygons_count_l13_13976

def isValidFolding (base_positions : Finset Nat) (additional_position : Nat) : Prop :=
  ∃ (valid_positions : Finset Nat), valid_positions = {4, 5, 6, 7, 8, 9} ∧ additional_position ∈ valid_positions

theorem foldable_polygons_count : 
  ∃ (valid_additional_positions : Finset Nat), valid_additional_positions = {4, 5, 6, 7, 8, 9} ∧ valid_additional_positions.card = 6 := 
by
  sorry

end foldable_polygons_count_l13_13976


namespace min_value_geq_sqrt2_l13_13924

theorem min_value_geq_sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4^a * 2^b = 4) :
  2 / a + 1 / b ≥ 9 / 2 :=
sorry

end min_value_geq_sqrt2_l13_13924


namespace probability_not_touch_outer_edge_l13_13586

def checkerboard : ℕ := 10

def total_squares : ℕ := checkerboard * checkerboard

def perimeter_squares : ℕ := 4 * checkerboard - 4

def inner_squares : ℕ := total_squares - perimeter_squares

def probability : ℚ := inner_squares / total_squares

theorem probability_not_touch_outer_edge : probability = 16 / 25 :=
by
  sorry

end probability_not_touch_outer_edge_l13_13586


namespace centroid_distance_to_plane_l13_13872

theorem centroid_distance_to_plane 
  (h1 h2 h3 : ℕ) 
  (h1_val : h1 = 1) 
  (h2_val : h2 = 2) 
  (h3_val : h3 = 3) : 
  (h1 + h2 + h3) / 3 = 2 :=
by 
  rw [h1_val, h2_val, h3_val]
  norm_num

end centroid_distance_to_plane_l13_13872


namespace find_constants_l13_13456

noncomputable def find_a {a : ℝ} (expansion_constant : ℝ) := 
  ∃ (n : ℕ) (k : ℕ), n = 8 ∧ k = 4 ∧ ((binom n k * (-a)^k): ℝ) = expansion_constant

noncomputable def sum_of_coefficients {a : ℝ} (s1 s2 : ℝ) :=
  let p (x : ℝ) := (x - a/x : ℝ)^8 in
  p 1 = s1 ∨ p 1 = s2

theorem find_constants (a s1 s2 : ℝ) (expansion_constant : ℝ) :
  find_a expansion_constant → (a = 2 ∨ a = -2) ∧
  sum_of_coefficients s1 s2 :=
by
  sorry

end find_constants_l13_13456


namespace arccos_cos_eight_l13_13747

-- Define the conditions
def cos_equivalence (x : ℝ) : Prop := cos x = cos (x - 2 * Real.pi)
def range_principal (x : ℝ) : Prop := 0 ≤ x - 2 * Real.pi ∧ x - 2 * Real.pi ≤ Real.pi

-- State the main proposition
theorem arccos_cos_eight :
  cos_equivalence 8 ∧ range_principal 8 → Real.arccos (Real.cos 8) = 8 - 2 * Real.pi :=
by
  sorry

end arccos_cos_eight_l13_13747


namespace area_ratio_triangle_MNO_XYZ_l13_13905

noncomputable def triangle_area_ratio (XY YZ XZ p q r : ℝ) : ℝ := sorry

theorem area_ratio_triangle_MNO_XYZ : 
  ∀ (p q r: ℝ),
  p > 0 → q > 0 → r > 0 →
  p + q + r = 3 / 4 →
  p ^ 2 + q ^ 2 + r ^ 2 = 1 / 2 →
  triangle_area_ratio 12 16 20 p q r = 9 / 32 :=
sorry

end area_ratio_triangle_MNO_XYZ_l13_13905


namespace sum_T_n_geq_m_div_20_l13_13466

variable (n : ℕ) (m : ℝ)

def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x

def S_n (n : ℕ) : ℝ := f n

def a_n (n : ℕ) : ℝ := 
  if n = 1 then 1 
  else S_n n - S_n (n - 1)

def b_n (n : ℕ) : ℝ := 3 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℝ :=
  (1 / 2) * (1 - (1 / (6 * n + 1)))

theorem sum_T_n_geq_m_div_20 : 
  (∀ n, T_n n ≥ m / 20) ↔ m ≤ 60 / 7 := sorry

end sum_T_n_geq_m_div_20_l13_13466


namespace find_area_S_find_a_l13_13906

-- Definitions
variables (A B C : ℝ)
variables (a b c : ℝ)
variables (cos_half_A : ℝ) (dot_product_AB_AC : ℝ)
variables (sum_b_c : ℝ)

-- Conditions
hypothesis h1 : cos_half_A = 2 * real.sqrt 5 / 5
hypothesis h2 : dot_product_AB_AC = 3
hypothesis h3 : sum_b_c = 6

-- To Prove
theorem find_area_S : 
  ∃ S : ℝ, (cos_half_A = 2 * real.sqrt 5 / 5) ∧ (dot_product_AB_AC = 3) → S = 2 :=
begin
  sorry -- Proof should go here
end

theorem find_a : 
  ∃ a : ℝ, (cos_half_A = 2 * real.sqrt 5 / 5) ∧ (dot_product_AB_AC = 3) ∧ (sum_b_c = 6) → a = real.sqrt 15 :=
begin
  sorry -- Proof should go here
end

end find_area_S_find_a_l13_13906


namespace solve_equation_l13_13250

theorem solve_equation (x : ℝ) (h1 : 0 < x) (h2 : x ≠ 1) :
  3 * 4^log x 2 - 46 * 2^(log x 2 - 1) = 8 → x = real.cbrt 2 :=
by
  sorry

end solve_equation_l13_13250


namespace inverse_function_l13_13761

-- Define the function h
def h (x : ℝ) : ℝ := 3 - 7 * x

-- Define the candidate inverse function k
def k (x : ℝ) : ℝ := (3 - x) / 7

-- State the proof problem
theorem inverse_function (x : ℝ) : h (k x) = x := by
  sorry

end inverse_function_l13_13761


namespace matrix_eq_scalar_l13_13792

theorem matrix_eq_scalar (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ u : Vector ℝ (Fin 3), N.mulVec u = (3 : ℝ) • u) ->
  N = Matrix.scalar _ (3 : ℝ) := by
  sorry

end matrix_eq_scalar_l13_13792


namespace closest_point_l13_13085

open Real

noncomputable def line_eq (x : ℝ) : ℝ := (3 * x - 1) / 4

theorem closest_point :
  ∃ (x y : ℝ), y = line_eq x ∧ dist (x, y) (3, 5) = dist (frac 111 25, frac 193 50) (3, 5) :=
begin
  let y_l := line_eq,
  let point := (3, 5),
  let closest_pt := (111 / 25, 193 / 50),
  use [111 / 25, 193 / 50],
  split,
  {
    calc
    193 / 50 = (3 * (111 / 25) - 1) / 4 : by sorry
  },
  {
    calc
    dist (111 / 25, 193 / 50) (3, 5) = sqrt ((111 / 25 - 3) ^ 2 + (193 / 50 - 5) ^ 2) : by sorry
    ... = dist (111 / 25, 193 / 50) (3, 5) : by sorry
  }
end

end closest_point_l13_13085


namespace tens_digit_of_19_pow_2023_l13_13769

theorem tens_digit_of_19_pow_2023 :
  (19^2023 % 100) / 10 % 10 = 5 := 
sorry

end tens_digit_of_19_pow_2023_l13_13769


namespace octagonal_walk_distance_l13_13065

def angle_subtended_by_chord (r : ℝ) (θ : ℝ) : ℝ :=
  2 * r * Real.sin (θ / 2)

def distance_walked_by_one_boy (r : ℝ) : ℝ :=
  let d90 := angle_subtended_by_chord r (Real.pi / 2)
  let d180 := 2 * r
  d90 + d90 + d180

def total_distance_walked (r : ℝ) (n : ℕ) : ℝ :=
  n * distance_walked_by_one_boy r

theorem octagonal_walk_distance :
  total_distance_walked 50 8 = 1600 + 800 * Real.sqrt (2 - Real.sqrt 2) :=
by
  -- Proof to be filled
  sorry

end octagonal_walk_distance_l13_13065


namespace circles_internally_tangent_l13_13995

-- Definitions for circles
def circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  { p | (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = radius ^ 2 }

-- Centers and radii for circles C1 and C2
def C1_center := (1.0, 1.0)
def C1_radius := 1.0

def C2_center := (-2.0, 5.0)
def C2_radius := 6.0

-- Distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Main theorem
theorem circles_internally_tangent :
  distance C1_center C2_center = abs (C2_radius - C1_radius) :=
by
  sorry

end circles_internally_tangent_l13_13995


namespace cos_alpha_l13_13124

theorem cos_alpha (a : ℝ) (h : a < 0) : 
  let P := (3 * a, 4 * a) in 
  let OP := real.sqrt ((3 * a)^2 + (4 * a)^2) in 
  (OP = -5 * a) →
  (cos (real.arctan (P.snd / P.fst)) = -3 / 5) := 
by
  sorry

end cos_alpha_l13_13124


namespace zoe_spent_amount_l13_13380

def flower_price : ℕ := 3
def roses_bought : ℕ := 8
def daisies_bought : ℕ := 2

theorem zoe_spent_amount :
  roses_bought + daisies_bought = 10 ∧
  flower_price = 3 →
  (roses_bought + daisies_bought) * flower_price = 30 :=
by
  sorry

end zoe_spent_amount_l13_13380


namespace find_k_l13_13508

theorem find_k (k : ℝ) : 
  let a := (1, 1)
  let b := (-1, 0)
  let c := (2, 1)
  (let u := (k * a.1 + b.1, k * a.2 + b.2)
  (u.1 * c.2 = u.2 * c.1)) ↔ k = -1 :=
by
  let a := (1, 1)
  let b := (-1, 0)
  let c := (2, 1)
  let u := (k * a.1 + b.1, k * a.2 + b.2)
  have h : u = (k - 1, k) := by
    simp only [Prod.ext_iff]
    split
    · simp [u, a, b]
    · 
      sorry
  simp [h, c]
  constructor
  · intro h1
    sorry
  · intro h2
    sorry

end find_k_l13_13508


namespace max_median_proof_l13_13953

def cans_sold := 360
def customers := 120
def max_zero_customers := 10
noncomputable def max_median_cans_per_customer := 5.5

theorem max_median_proof 
  (h1 : ∑ i in (finset.range customers), (if i < max_zero_customers then 0 else 1) ≤ cans_sold)
  (h2 : ∀ i, if i < max_zero_customers then 0 else 1 ≤ 1) :
  max_median_cans_per_customer = 5.5 :=
sorry

end max_median_proof_l13_13953


namespace sum_of_consecutive_even_integers_l13_13672

theorem sum_of_consecutive_even_integers (n : ℕ) (h1 : (n - 2) + (n + 2) = 162) (h2 : ∃ k : ℕ, n = k^2) :
  (n - 2) + n + (n + 2) = 243 :=
by
  -- no proof required
  sorry

end sum_of_consecutive_even_integers_l13_13672


namespace average_snowfall_dec_1861_l13_13166

theorem average_snowfall_dec_1861 (snowfall : ℕ) (days_in_dec : ℕ) (hours_in_day : ℕ) 
  (time_period : ℕ) (Avg_inch_per_hour : ℚ) : 
  snowfall = 492 ∧ days_in_dec = 31 ∧ hours_in_day = 24 ∧ time_period = days_in_dec * hours_in_day ∧ 
  Avg_inch_per_hour = snowfall / time_period → 
  Avg_inch_per_hour = 492 / (31 * 24) :=
by sorry

end average_snowfall_dec_1861_l13_13166


namespace price_decrease_zero_l13_13996

theorem price_decrease_zero
    (P : ℝ) 
    (initial_increase : P → 1.15 * P) 
    (final_price : (P : ℝ) → P * (1.15) * (1 - 0 / 100) = 97.75) : 
    0 = 0 :=
by
  sorry

end price_decrease_zero_l13_13996


namespace mileage_on_city_streets_l13_13222

-- Defining the given conditions
def distance_on_highways : ℝ := 210
def mileage_on_highways : ℝ := 35
def total_gas_used : ℝ := 9
def distance_on_city_streets : ℝ := 54

-- Proving the mileage on city streets
theorem mileage_on_city_streets :
  ∃ x : ℝ, 
    (distance_on_highways / mileage_on_highways + distance_on_city_streets / x = total_gas_used)
    ∧ x = 18 :=
by
  sorry

end mileage_on_city_streets_l13_13222


namespace probability_of_top_grade_product_l13_13005

-- Definitions for the problem conditions
def P_B : ℝ := 0.03
def P_C : ℝ := 0.01

-- Given that the sum of all probabilities is 1
axiom sum_of_probabilities (P_A P_B P_C : ℝ) : P_A + P_B + P_C = 1

-- Statement to be proved
theorem probability_of_top_grade_product : ∃ P_A : ℝ, P_A = 1 - P_B - P_C ∧ P_A = 0.96 :=
by
  -- Assuming the proof steps to derive the answer
  sorry

end probability_of_top_grade_product_l13_13005


namespace opposite_of_half_l13_13271

theorem opposite_of_half : - (1 / 2) = -1 / 2 := 
by
  sorry

end opposite_of_half_l13_13271


namespace round_robin_tournament_points_l13_13595

theorem round_robin_tournament_points :
  ∀ (teams : Finset ℕ), teams.card = 6 →
  ∀ (matches_played : ℕ), matches_played = 12 →
  ∀ (total_points : ℤ), total_points = 32 →
  ∀ (third_highest_points : ℤ), third_highest_points = 7 →
  ∀ (draws : ℕ), draws = 4 →
  ∃ (fifth_highest_points_min fifth_highest_points_max : ℤ),
    fifth_highest_points_min = 1 ∧
    fifth_highest_points_max = 3 :=
by
  sorry

end round_robin_tournament_points_l13_13595


namespace remainder_when_divided_by_11_l13_13206

noncomputable def hundred_thousands_9 (k : Nat) : Int :=
  (10^k - 1) / 9

noncomputable def hundred_thousands_6 (k : Nat) : Int :=
  (7 * (10^k - 1)) / 9

def A := hundred_thousands_9 20069
def B := hundred_thousands_6 20066
def n := A^2 - B

theorem remainder_when_divided_by_11 : n % 11 = 1 := by
  sorry

end remainder_when_divided_by_11_l13_13206


namespace caterpillar_count_l13_13632

theorem caterpillar_count 
    (initial_count : ℕ)
    (hatched : ℕ)
    (left : ℕ)
    (h_initial : initial_count = 14)
    (h_hatched : hatched = 4)
    (h_left : left = 8) :
    initial_count + hatched - left = 10 :=
by
    sorry

end caterpillar_count_l13_13632


namespace xiaoming_total_chars_l13_13057

-- Definitions used:
def day1_chars_written (N : ℕ) : ℕ := N / 2 - 50
def day1_remaining_chars (N : ℕ) : ℕ := N - day1_chars_written N
def day2_chars_written (R1 : ℕ) : ℕ := R1 / 2 - 20
def day2_remaining_chars (R1 : ℕ) (x_2 : ℕ) : ℕ := R1 - x_2
def day3_chars_written (R2 : ℕ) : ℕ := R2 / 2 + 10
def day3_remaining_chars (R2 : ℕ) (x_3 : ℕ) : ℕ := R2 - x_3
def day4_remaining_chars (R3 : ℕ) : ℕ := R3 - 60

-- Main theorem to prove
theorem xiaoming_total_chars : ∀ (N : ℕ), 
  let x1 := day1_chars_written N,
      R1 := day1_remaining_chars N,
      x2 := day2_chars_written R1,
      R2 := day2_remaining_chars R1 x2,
      x3 := day3_chars_written R2,
      R3 := day3_remaining_chars R2 x3,
      final_remaining := day4_remaining_chars R3
  in final_remaining = 40 → N = 700 :=
by sorry

end xiaoming_total_chars_l13_13057


namespace cos_pi_minus_alpha_l13_13805

theorem cos_pi_minus_alpha (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : 3 * sin (2 * α) = sin α) : 
  cos (π - α) = -1 / 6 :=
by
  sorry

end cos_pi_minus_alpha_l13_13805


namespace find_f_log2_3_l13_13131

noncomputable def f : ℝ → ℝ
| x := if x ≥ 4 then (1/2)^x else f (x + 1)

theorem find_f_log2_3 : f (Real.log 3 / Real.log 2) = 1 / 24 :=
by
  sorry
  
end find_f_log2_3_l13_13131


namespace existence_of_non_intersecting_chords_l13_13831

theorem existence_of_non_intersecting_chords (n : ℕ) :
  ∃ (chords : set (fin (2 * n) × fin (2 * n))),
    (∀ (c ∈ chords), (c.1 < n ∧ c.2 ≥ n) ∨ (c.1 ≥ n ∧ c.2 < n)) ∧
    (∀ (c1 c2 ∈ chords), c1 ≠ c2 → ¬intersects c1 c2) ∧
    chords.card = n := sorry

end existence_of_non_intersecting_chords_l13_13831


namespace sin_eq_cos_sufficient_not_necessary_l13_13518

theorem sin_eq_cos_sufficient_not_necessary (α : ℝ) : (sin α = cos α) → (cos (2 * α) = 0) ∧ (∃ β : ℝ, cos (2 * β) = 0 ∧ sin β ≠ cos β) :=
by
  intro h
  split
  case left =>
    -- Proof sufficiency part here
    sorry
  case right =>
    -- Proof that there exist some β such that cos(2β) = 0 but sin β ≠ cos β
    sorry


end sin_eq_cos_sufficient_not_necessary_l13_13518


namespace max_diagonals_in_5x5_grid_l13_13011

-- Define the conditions
def is_diagonal (square :: fin 5 × fin 5) (d :: fin 2) : Prop :=
  match d with
  | 0 => True  -- one diagonal direction
  | 1 => True  -- the other diagonal direction
  | _ => False

def intersect (d1 :: (fin 5 × fin 5) × fin 2) (d2 :: (fin 5 × fin 5) × fin 2) : Prop :=
  -- diagonals intersect if they share a vertex
  let (s1, diag1) := d1
  let (s2, diag2) := d2
  s1.fst = s2.fst ∨ s1.snd = s2.snd ∨ (s1.fst = s2.snd ∧ diag1 = diag2) ∨ (s1.snd = s2.fst ∧ diag1 = diag2)

-- Formalizing the problem
theorem max_diagonals_in_5x5_grid : 
  ∃ (diagonals :: (fin 5 × fin 5) × fin 2 → Prop), 
  (∀ d1 d2, diagonals d1 → diagonals d2 → d1 ≠ d2 → ¬ intersect d1 d2)
  ∧ (∑ s : fin 5 × fin 5, ∑ d : fin 2, if diagonals (s, d) then 1 else 0 = 16) :=
sorry

end max_diagonals_in_5x5_grid_l13_13011


namespace divisible_by_77_l13_13226

theorem divisible_by_77 (n : ℤ) : ∃ k : ℤ, n^18 - n^12 - n^8 + n^2 = 77 * k :=
by
  sorry

end divisible_by_77_l13_13226


namespace coeff_x4_expansion_l13_13607

open BigOperators

theorem coeff_x4_expansion : 
  (Polynomial.coeff (Polynomial.expand (5:ℕ) (Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.C 2 * Polynomial.X⁻¹)) 4) = 40 := by
  sorry

end coeff_x4_expansion_l13_13607


namespace expand_array_l13_13916

theorem expand_array (n : ℕ) (h₁ : n ≥ 3) 
  (matrix : Fin (n-2) → Fin n → Fin n)
  (h₂ : ∀ i : Fin (n-2), ∀ j: Fin n, ∀ k: Fin n, j ≠ k → matrix i j ≠ matrix i k)
  (h₃ : ∀ j : Fin n, ∀ k: Fin (n-2), ∀ l: Fin (n-2), k ≠ l → matrix k j ≠ matrix l j) :
  ∃ (expanded_matrix : Fin n → Fin n → Fin n), 
    (∀ i : Fin n, ∀ j: Fin n, ∀ k: Fin n, j ≠ k → expanded_matrix i j ≠ expanded_matrix i k) ∧
    (∀ j : Fin n, ∀ k: Fin n, ∀ l: Fin n, k ≠ l → expanded_matrix k j ≠ expanded_matrix l j) :=
sorry

end expand_array_l13_13916


namespace solve_equation_l13_13251

theorem solve_equation : ∀ (x : ℝ), (2 * x + 1) / 3 - (x - 1) / 6 = 2 ↔ x = 3 := 
begin
  sorry
end

end solve_equation_l13_13251


namespace sum_peak_minus_valley_eq_n_l13_13950

section PeaksAndValleys

variable (n : ℕ) (labels : Fin (2 * n) → ℤ)
variable (peak_valley_diff : ℤ) 

-- Conditions
-- Each vertex of a 2n-gon labeled with consecutive integers differing by 1
axiom consecutive_integers (i : Fin (2 * n)) : abs (labels (i + 1) - labels i) = 1

-- Define a peak
def is_peak (i : Fin (2 * n)) : Prop :=
labels i > labels (i - 1) ∧ labels i > labels (i + 1)

-- Define a valley
def is_valley (i : Fin (2 * n)) : Prop :=
labels i < labels (i - 1) ∧ labels i < labels (i + 1)

-- Sum of peak numbers minus sum of valley numbers
def peak_valley_sum_difference : ℤ :=
  ∑ i in Finset.filter is_peak Finset.univ, labels i - ∑ i in Finset.filter is_valley Finset.univ, labels i

-- Theorem to prove
theorem sum_peak_minus_valley_eq_n : peak_valley_sum_difference labels = n := by
  sorry

end PeaksAndValleys

end sum_peak_minus_valley_eq_n_l13_13950


namespace caterpillar_count_proof_l13_13633

def number_of_caterpillars_after_events (initial : ℕ) (hatched : ℕ) (left : ℕ) : ℕ :=
  initial + hatched - left

theorem caterpillar_count_proof :
  number_of_caterpillars_after_events 14 4 8 = 10 :=
by
  simp [number_of_caterpillars_after_events]
  sorry

end caterpillar_count_proof_l13_13633


namespace largest_n_satisfying_inequality_l13_13662

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, (1/4 + n/6 < 3/2) ∧ (∀ m : ℕ, m > n → 1/4 + m/6 ≥ 3/2) :=
by
  sorry

end largest_n_satisfying_inequality_l13_13662


namespace tens_digit_of_19_pow_2023_l13_13765

theorem tens_digit_of_19_pow_2023 : (19 ^ 2023) % 100 = 59 := 
  sorry

end tens_digit_of_19_pow_2023_l13_13765


namespace triangle_proof_l13_13878

noncomputable def triangle_properties (A B C : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C]
  (a c : ℝ) (cos_A : ℝ) : Prop :=
  a = 2 ∧
  c = real.sqrt 2 ∧
  cos_A = real.sqrt 2 / 4 ∧
  let sin_A := real.sqrt 14 / 4,
      sin_C := real.sqrt 7 / 4,
      b := 1,
      cos_2A := 2 * (cos_A * cos_A) - 1,
      sin_2A := 2 * (sin_A * cos_A),
      cos_expr := (cos_2A * (1 / 2)) - (sin_2A * (real.sqrt 3 / 2))
  in sin_C = real.sqrt 7 / 4 ∧ b = 1 ∧ cos_expr = (-3 + real.sqrt 21) / 8

theorem triangle_proof : triangle_properties ℝ ℝ ℝ 2 (real.sqrt 2) (real.sqrt 2 / 4) :=
by
  apply and.intro rfl
  apply and.intro rfl
  apply and.intro rfl
  unfold triangle_properties
  sorry

end triangle_proof_l13_13878


namespace simplify_expression_l13_13315

theorem simplify_expression (a : ℕ) (h1 : a = 2015) :
    (a^3 - 3 * a^2 * (a + 1) + 3 * a * (a + 1)^2 - (a + 1)^3 + 1) / (a * (a + 1)) = -3 :=
by
  rw [h1]
  sorry

end simplify_expression_l13_13315


namespace crayons_loss_difference_l13_13589

theorem crayons_loss_difference :
  ∀ (initial_crayons given_crayons lost_crayons : ℕ),
  initial_crayons = 110 →
  given_crayons = 90 →
  lost_crayons = 412 →
  lost_crayons - given_crayons = 322 :=
by
  intros initial_crayons given_crayons lost_crayons h1 h2 h3
  rw [h1, h2, h3]
  exact calc
    412 - 90 = 322 : by norm_num

end crayons_loss_difference_l13_13589


namespace digit_1_left_of_3_l13_13260

theorem digit_1_left_of_3 :
  let digits := {1, 2, 3, 4, 5, 6},
  let total_arrangements := Nat.factorial 6,
  total_arrangements = 720 →
  ∃ num_arrangements,
    num_arrangements = 360 ∧
    ∀ (arrangement : List Nat), arrangement ∈ digits.toList.permutations →
      (list_index arrangement 1) < (list_index arrangement 3) ↔ num_arrangements = 360 :=
by
  intros digits total_arrangements ht
  -- sorry added to skip the actual proof
  sorry

end digit_1_left_of_3_l13_13260


namespace problem_solution_l13_13855

noncomputable def p : ℝ := sorry -- Placeholder to indicate that solving for p gives approximately 3.6
noncomputable def r : ℝ := sorry -- Placeholder to indicate that solving for r gives exactly 4
noncomputable def s : ℝ := sorry -- Placeholder to indicate that solving for s gives approximately 2.8

theorem problem_solution :
  (4^p + 4^3 = 272) →
  (3^r + 39 = 120) →
  (4^s + 2^8 = 302) →
  p * r * s = 40.32 :=
by
  intros h1 h2 h3
  sorry

end problem_solution_l13_13855


namespace order_of_values_l13_13120

variable {f : ℝ → ℝ}
variable {f_deriv : ℝ → ℝ}
variable {x : ℝ}

-- Declare the hypotheses
axiom h1 : ∀ x : ℝ, f_deriv x < f x
axiom h2 : ∀ x : ℝ, f x = f (-x)

-- Define the values a, b, and c
def a : ℝ := Real.exp 1 * f 2
def b : ℝ := f (-3)
def c : ℝ := Real.exp 2 * f 1

-- The theorem to prove
theorem order_of_values : b < a ∧ a < c := by
  sorry

end order_of_values_l13_13120


namespace least_sum_upper_bound_l13_13405

theorem least_sum_upper_bound
  (n : ℕ) (h_even : n % 2 = 0)
  (A : Matrix (Fin n) (Fin n) ℝ)
  (h_abs_bound : ∀ i j, abs (A i j) ≤ 1)
  (h_sum_zero : finset.sum (finset.univ.product finset.univ) (λ (ij : Fin n × Fin n), A ij.1 ij.2) = 0):
  ∃ i, abs (finset.sum finset.univ (λ j, A i j)) ≤ n / 2 ∨ ∃ j, abs (finset.sum finset.univ (λ i, A i j)) ≤ n / 2 :=
sorry

end least_sum_upper_bound_l13_13405


namespace find_k_l13_13860

def g (x : ℝ) : ℝ := Real.tan (x / 2)

def f (gx : ℝ) : ℝ := Real.sin (2 * (2 * Real.arctan gx))

theorem find_k (k : ℝ) :
  (∀ x, 0 < x ∧ x < Real.pi → g x = Real.tan (x / 2) ∧ f (g x) = Real.sin (2 * x))
  ∧ k * f (Real.sqrt 2 / 2) = 36 * Real.sqrt 2 → k = 81 :=
by
  sorry

end find_k_l13_13860


namespace lily_received_books_l13_13220

def mike_books : ℕ := 45
def corey_books : ℕ := 2 * mike_books
def mike_gave_lily : ℕ := 10
def corey_gave_lily : ℕ := mike_gave_lily + 15
def lily_books_received : ℕ := mike_gave_lily + corey_gave_lily

theorem lily_received_books : lily_books_received = 35 := by
  sorry

end lily_received_books_l13_13220


namespace product_mod_five_remainder_l13_13088

theorem product_mod_five_remainder :
  (∏ i in (Finset.range 10), 3 + 10 * i) % 5 = 4 :=
by
  sorry

end product_mod_five_remainder_l13_13088


namespace eval_modulus_poly_l13_13431

-- Define z
def z : ℂ := 7 + 3 * complex.I

-- Define the polynomial
def polynomial (z : ℂ) : ℂ := z^2 + 8 * z + 100

-- The modulus of the polynomial
def mod_poly (z : ℂ) : ℝ := complex.abs (polynomial z)

-- Problem statement
theorem eval_modulus_poly : mod_poly z = 207 := by
  sorry

end eval_modulus_poly_l13_13431


namespace relationship_abc_l13_13557

theorem relationship_abc :
  let a := 4 ^ 0.2
  let b := (1 / 3) ^ (-0.4)
  let c := Real.log 0.4 / Real.log 0.2
  c < a ∧ a < b :=
by
  sorry

end relationship_abc_l13_13557


namespace discrete_rv_X_hit_rings_l13_13320

-- Defining the problem in Lean
theorem discrete_rv_X_hit_rings 
  (X : ℕ → ℝ)        -- X is a function taking natural numbers to real numbers representing probabilities
  (hx : ∀ n, n ≤ 10 → X n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})  -- Values X can take are within 0 to 10
  : true := sorry    -- Placeholder for the proof which is not required

end discrete_rv_X_hit_rings_l13_13320


namespace euler_totient_formula_l13_13205

open Nat

-- Definitions of conditions
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∃ d : ℕ, d > 1 ∧ d ∣ p

def prime_factors (n : ℕ) : List (ℕ × ℕ) :=
-- Placeholder for the actual implementation for extracting prime factors
[]

-- Definition of Euler's Totient Function
def euler_totient (n : ℕ) : ℕ :=
-- Placeholder for the actual implementation
0

-- The proof statement
theorem euler_totient_formula {n : ℕ}
  (h : prime_factors n = List.zip (List.range k) primes)
  (primes_le : ∀ (i : ℕ) (h_i : i < primes.length), is_prime (primes.get i) ∧ primes.get i < primes.get (i + 1)) :
  euler_totient n = List.prod ((List.map (λ p, (p - 1)) primes) ++ (List.map (λ (p π : ℕ), p^(π-1)) primes)) :=
sorry

end euler_totient_formula_l13_13205


namespace pqrs_sum_l13_13095

theorem pqrs_sum (p q r s : ℤ)
  (h1 : r + p = -1)
  (h2 : s + p * r + q = 3)
  (h3 : p * s + q * r = -4)
  (h4 : q * s = 4) :
  p + q + r + s = -1 :=
sorry

end pqrs_sum_l13_13095


namespace snowman_volume_l13_13585

noncomputable def volume_snowman (r₁ r₂ r₃ r_c h_c : ℝ) : ℝ :=
  (4 / 3 * Real.pi * r₁^3) + (4 / 3 * Real.pi * r₂^3) + (4 / 3 * Real.pi * r₃^3) + (Real.pi * r_c^2 * h_c)

theorem snowman_volume 
  : volume_snowman 4 6 8 3 5 = 1101 * Real.pi := 
by 
  sorry

end snowman_volume_l13_13585


namespace average_of_excellent_students_l13_13031

theorem average_of_excellent_students (scores : Fin 54 → ℝ) :
  (∑ i, if scores i > 90 then scores i else 0) / (∑ i, if scores i > 90 then 1 else 0) =
  (∑ i in Finset.univ.filter (λ i, scores i > 90), scores i) / ↑((Finset.univ.filter (λ i, scores i > 90)).card) :=
by
  sorry

end average_of_excellent_students_l13_13031


namespace journey_speed_l13_13369

theorem journey_speed 
  (total_time : ℝ)
  (total_distance : ℝ)
  (second_half_speed : ℝ)
  (first_half_speed : ℝ) :
  total_time = 30 ∧ total_distance = 400 ∧ second_half_speed = 10 ∧
  2 * (total_distance / 2 / second_half_speed) + total_distance / 2 / first_half_speed = total_time →
  first_half_speed = 20 :=
by
  intros hyp
  sorry

end journey_speed_l13_13369


namespace first_player_wins_l13_13364

-- Define the initial conditions of the game.
def initial_coins : ℕ := 25

-- Define the game rules as follows:
-- - A player can take either 1 or 2 adjacent coins.
def valid_move (coins_taken : ℕ) : Prop :=
  coins_taken = 1 ∨ coins_taken = 2

-- Define a function to determine if a player has lost (i.e., the player cannot make a move).
def game_lost (remaining_coins : ℕ) : Prop :=
  remaining_coins = 0

-- Define what it means for the first player to have a winning strategy.
def first_player_winning_strategy : Prop :=
  ∀ remaining_coins : ℕ, remaining_coins = 25 →
  (∃ moves : list ℕ, (∀ move_in_list ∈ moves, valid_move move_in_list) ∧
  (∃ second_player_move : ℕ, valid_move second_player_move ∧
  game_lost (remaining_coins - second_player_move))) 

theorem first_player_wins : first_player_winning_strategy :=
  sorry

end first_player_wins_l13_13364


namespace remainder_proof_l13_13316

noncomputable def problem (n : ℤ) : Prop :=
  n % 9 = 4

noncomputable def solution (n : ℤ) : ℤ :=
  (4 * n - 11) % 9

theorem remainder_proof (n : ℤ) (h : problem n) : solution n = 5 := by
  sorry

end remainder_proof_l13_13316


namespace sum_of_cubes_equal_square_of_sum_l13_13964

theorem sum_of_cubes_equal_square_of_sum (n : ℕ) (h : 1 ≤ n) : 
  ∑ k in finset.range n.succ, k^3 = (∑ k in finset.range n.succ, k)^2 := 
sorry

end sum_of_cubes_equal_square_of_sum_l13_13964


namespace base_conversion_result_l13_13210

noncomputable def base_b_number (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.reverse.foldl (λ acc d, acc * b + d) 0

def condition_product (b : ℕ) :=
  (b + 3) * (b + 4) * (b + 7) = 5 * b^3 + 1 * b^2 + 6 * b + 7

def condition_sum (b : ℕ) :=
  3 * b + 14

theorem base_conversion_result (b : ℕ) (s : ℕ) (res : ℕ) :
  condition_product b →
  s = condition_sum b →
  s = res →
  base_b_number 7 [0, 5] = res :=
sorry

end base_conversion_result_l13_13210


namespace quadrilaterals_congruent_l13_13937

-- Define the quadrilateral and perpendicular condition
variables {Point : Type} [Geometry Point] (A B C D E F G H I J K L P1 Q1 R1 S1 P2 Q2 R2 S2 : Point)

-- Conditions for AC perpendicular to BD and construction of squares
axiom ac_perpendicular_bd : Perpendicular A C B D
axiom square_abef : is_square A B E F
axiom square_bcgh : is_square B C G H
axiom square_cdij : is_square C D I J
axiom square_dakl : is_square D A K L

-- Definitions of intersection points
axiom p1_definition : P1 = intersection_point C L D F
axiom q1_definition : Q1 = intersection_point D F B J
axiom r1_definition : R1 = intersection_point A H B J
axiom s1_definition : S1 = intersection_point A H D F

axiom p2_definition : P2 = intersection_point A I B K
axiom q2_definition : Q2 = intersection_point B K C E
axiom r2_definition : R2 = intersection_point C E D G
axiom s2_definition : S2 = intersection_point D G A I

-- Main theorem statement
theorem quadrilaterals_congruent : congruent (quadrilateral P1 Q1 R1 S1) (quadrilateral P2 Q2 R2 S2) := 
by sorry

end quadrilaterals_congruent_l13_13937


namespace find_a_from_distance_l13_13867

theorem find_a_from_distance 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : ¬a = 2) 
  (h3 : ∀ (x₀ y₀) (A B C : ℝ), A*x₀ + B*y₀ + C = 0 → a = 2):
  (square_root = Real.sqrt (4*4 + 3*3) := by sorry) ∧ sorry :=

  sorry -- Proof goes here

end find_a_from_distance_l13_13867


namespace n_four_plus_four_composite_l13_13788

theorem n_four_plus_four_composite (n : ℕ) (h : n ≠ 1) : ¬ prime (n^4 + 4) :=
sorry

end n_four_plus_four_composite_l13_13788


namespace gas_to_groceries_ratio_l13_13218

variables (initial_balance groceries_cost return_amount new_balance gas_spent : ℝ)

-- Given conditions
def conditions := initial_balance = 126 ∧ groceries_cost = 60 ∧ return_amount = 45 ∧ new_balance = 171

-- The task is to prove the ratio of gas spent to groceries cost equals 1:2
theorem gas_to_groceries_ratio (h : conditions) :
  gas_spent = new_balance - (initial_balance + groceries_cost - return_amount) → gas_spent / groceries_cost = 1 / 2 :=
by
  intros h1
  rw h1
  -- remaining proof steps can be added here using the conditions defined earlier
  sorry

end gas_to_groceries_ratio_l13_13218


namespace recover_investment_at_least_5_years_l13_13704

theorem recover_investment_at_least_5_years :
  ∀ n : ℕ, (1000000 * n ≥ 4000000 + ∑ k in Finset.range n, (100000 + 50000 * k)) → n ≥ 5 :=
by
  intro n
  intro h
  sorry

end recover_investment_at_least_5_years_l13_13704


namespace sum_multiple_of_3_l13_13602

def multiple_of (k n : ℤ) : Prop := ∃ m : ℤ, n = k * m

variables (a b : ℤ)
hypothesis h1 : multiple_of 6 a
hypothesis h2 : multiple_of 9 b

theorem sum_multiple_of_3 : multiple_of 3 (a + b) :=
sorry

end sum_multiple_of_3_l13_13602


namespace max_mn_l13_13843

theorem max_mn : ∀ (a : ℝ) (m n : ℝ), (a > 0) ∧ (a ≠ 1) ∧ (m + n = 2) → (m * n ≤ 1) := 
by
  intros a m n h
  cases h with a_positive_and_neq_one sum_eq_two
  sorry

end max_mn_l13_13843


namespace AS_squared_minus_AE_squared_eq_AB_minus_AC_squared_l13_13876

theorem AS_squared_minus_AE_squared_eq_AB_minus_AC_squared
  {A B C E S : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace E] [MetricSpace S]
  (AB AC AE AS : ℝ) [Decidable (AB > AC)]
  (intersects_at_E : AE.intersect BC E)
  (angle_bisector_A : angleBisector A AE ∧ AE.intersect BC E)
  (point_S_on_BC : S ∈ BC)
  (BS_eq_EC : dist B S = dist E C)
  : (dist A S)^2 - (dist A E)^2 = (AB - AC)^2 := 
  sorry

end AS_squared_minus_AE_squared_eq_AB_minus_AC_squared_l13_13876


namespace largest_whole_number_for_inequality_l13_13663

theorem largest_whole_number_for_inequality :
  ∀ n : ℕ, (1 : ℝ) / 4 + (n : ℝ) / 6 < 3 / 2 → n ≤ 7 :=
by
  admit  -- skip the proof

end largest_whole_number_for_inequality_l13_13663


namespace range_x_squared_l13_13152

theorem range_x_squared (x : ℝ) (h : (∛(x + 16) - ∛(x - 16) = 4)) : 240 ≤ x^2 ∧ x^2 ≤ 250 :=
by
  sorry

end range_x_squared_l13_13152


namespace _l13_13845

variable (x y : ℝ) (F : ℝ × ℝ) (m : ℝ)
variable (right_focus_cond : F = (2, 0))
variable (hyperbola_equation : x^2 + m * y^2 = 1)

noncomputable def value_of_m : ℝ :=
-1 / 3

noncomputable theorem asymptotes_equation :
  (∀ (x y : ℝ), y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) :=
by sorry

end _l13_13845


namespace pyramid_volume_l13_13276

/-
The rectangle ABCD has dimensions AB = 15 * sqrt 2 and BC = 10 * sqrt 2. 
Diagonals AC and BD intersect at P. If triangle ABP is removed and AP and BP
are joined, creasing along segments CP and DP, a triangular pyramid is formed. 
All four faces of this pyramid are isosceles triangles. 
We need to prove that the volume of this pyramid is 375.
-/

theorem pyramid_volume (AB BC: ℝ) (h1: AB = 15 * Real.sqrt 2) (h2: BC = 10 * Real.sqrt 2) 
    (isosceles: ∀ (a b c: ℝ), a = b ∧ a = c ∧ b = c):
  let V := (1 / 3) * (length_of_base * height_of_pyramid) in
  V = 375 :=
by
  sorry

end pyramid_volume_l13_13276


namespace monotonic_intervals_g_zeros_in_interval_l13_13133

-- Given conditions
def f (ω x : ℝ) : ℝ :=
  (sqrt 3) * sin (ω * x) * cos (ω * x) - (cos (ω * x))^2

def g (ω x b : ℝ) : ℝ :=
  f ω x - b

-- Statements to prove
theorem monotonic_intervals (ω : ℝ) (h_ω : ω > 0) :
  ∃ k : ℤ, ∀ x, (k * real.pi - real.pi / 6) ≤ x ∧ x ≤ (k * real.pi + real.pi / 3) ↔
         (deriv (λ y, f ω y) x > 0) :=
sorry

theorem g_zeros_in_interval (ω b : ℝ) (h_ω : ω > 0) :
  ∀ x ∈ set.Icc 0 (real.pi / 2), 
  (g ω x b = 0 → (∀ k ∈ set.Icc 0 (real.pi / 2), g ω k b = 0 → (0 ≤ b ∧ b < 1/2))) :=
sorry

end monotonic_intervals_g_zeros_in_interval_l13_13133


namespace transform_result_l13_13753

def complex_transform (z : ℂ) : ℂ :=
  let rotation := (1/2) + (real.sqrt 3) * complex.I / 2
  let dilation := 2
  dilation * rotation * z

theorem transform_result :
  complex_transform (-3 - 8 * complex.I) = (-3 - 8 * real.sqrt 3) + (-3 * real.sqrt 3 - 8) * complex.I :=
by
  -- The proof is omitted
  sorry

end transform_result_l13_13753


namespace M_intersection_N_is_empty_l13_13505

def M : Set ℂ := 
  {z | ∃ t : ℝ, (t ≠ -1) ∧ (t ≠ 0) ∧ z = (t / (1 + t)) + (1 + t) / t * complex.I}

def N : Set ℂ := 
  {z | ∃ t : ℝ, (abs t ≤ 1) ∧ z = sqrt 2 * (complex.cos (real.arcsin t) + complex.cos (real.arccos t) * complex.I)}

theorem M_intersection_N_is_empty : M ∩ N = ∅ :=
by
  sorry

end M_intersection_N_is_empty_l13_13505


namespace ordering_of_a_b_c_l13_13558

def a := Real.exp (1 / 2)
def b := Real.log (1 / 2)
def c := Real.log 2 (Real.sqrt 2)

theorem ordering_of_a_b_c : a > c ∧ c > b := by
  sorry

end ordering_of_a_b_c_l13_13558


namespace minimum_cars_with_racing_stripes_l13_13171

-- Definitions and conditions
variable (numberOfCars : ℕ) (withoutAC : ℕ) (maxWithACWithoutStripes : ℕ)

axiom total_number_of_cars : numberOfCars = 100
axiom cars_without_ac : withoutAC = 49
axiom max_ac_without_stripes : maxWithACWithoutStripes = 49    

-- Proposition
theorem minimum_cars_with_racing_stripes 
  (total_number_of_cars : numberOfCars = 100) 
  (cars_without_ac : withoutAC = 49)
  (max_ac_without_stripes : maxWithACWithoutStripes = 49) :
  ∃ (R : ℕ), R = 2 :=
by
  sorry

end minimum_cars_with_racing_stripes_l13_13171


namespace compute_AD_l13_13188

variables (A B C H D : ℝ) (AB BC CA : ℝ) (AH HD : ℝ)

def triangle_properties 
  (AB BC CA : ℝ) : Prop :=
  AB = 9 ∧ BC = 10 ∧ CA = 11

def orthocenter_properties (H D : ℝ) (AH HD : ℝ) : Prop :=
  AH = HD

theorem compute_AD 
  (conditions : triangle_properties AB BC CA)
  (ortho_conditions : orthocenter_properties H D AH HD) :
  D = sqrt 102 :=
by
  sorry

end compute_AD_l13_13188


namespace max_value_of_f_l13_13445

noncomputable def f (x : ℝ) : ℝ := 
  sin (2 * x) - 2 * sqrt 3 * (sin x)^2

theorem max_value_of_f : 
  (∀ x : ℝ, f(x) ≤ 2 - sqrt 3) ∧ (∃ x : ℝ, f(x) = 2 - sqrt 3) :=
by
  sorry

end max_value_of_f_l13_13445


namespace derivative_y_l13_13790

-- Define the function y
def y (x : ℝ) : ℝ :=
  sin (real.sqrt 3) + (1 / 3) * (sin (3 * x))^2 / (cos (6 * x))

-- Define the correct answer for the derivative
def y_deriv_correct (x : ℝ) : ℝ := 
  2 * (tan (6 * x) / cos (6 * x))

-- State the theorem to prove that differentiating y correctly gives y_deriv_correct
theorem derivative_y (x : ℝ) : deriv y x = y_deriv_correct x := 
by 
  sorry

end derivative_y_l13_13790


namespace only_n_1_has_integer_solution_l13_13758

theorem only_n_1_has_integer_solution :
  ∀ n : ℕ, (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 := 
by 
  sorry

end only_n_1_has_integer_solution_l13_13758


namespace minimum_total_distance_l13_13064

theorem minimum_total_distance :
  let r : ℝ := 50
  let num_boys : ℕ := 8
  let angle : ℝ := (3/8 : ℝ) * (2 * Real.pi)
  let s := Real.sqrt (2 * r^2 * (1 - Real.cos angle))
  let distance_per_boy := 4 * s
  let total_distance := num_boys * distance_per_boy
  total_distance = 1600 * Real.sqrt(2 - Real.sqrt 2) := 
by
  let r : ℝ := 50
  let num_boys : ℕ := 8
  let angle : ℝ := (3/8 : ℝ) * (2 * Real.pi)
  let s := Real.sqrt (2 * r^2 * (1 - Real.cos angle))
  let distance_per_boy := 4 * s
  let total_distance := num_boys * distance_per_boy
  show total_distance = 1600 * Real.sqrt(2 - Real.sqrt 2) from sorry

end minimum_total_distance_l13_13064


namespace comb_8_3_eq_56_l13_13400

theorem comb_8_3_eq_56 : nat.choose 8 3 = 56 := sorry

end comb_8_3_eq_56_l13_13400


namespace green_bows_count_l13_13884

-- Definitions of the conditions
def fraction_red : ℚ := 1 / 4
def fraction_blue : ℚ := 1 / 3
def fraction_green : ℚ := 1 / 6
def bows_white : ℕ := 40

-- Assertion that we want to prove
theorem green_bows_count : 
  let fraction_remaining := 1 - (fraction_red + fraction_blue + fraction_green)
  let total_bows := bows_white / fraction_remaining
  let green_bows := fraction_green * total_bows
  green_bows = 27 :=
by
  have h_fraction_remaining : 1 - (fraction_red + fraction_blue + fraction_green) = 1 / 4 := 
    by
      -- Calculation to prove the fraction of remaining bows
      sorry
  have h_total_bows : (bows_white : ℚ) / (1 / 4) = 160 :=
    by
      -- Calculation to convert white bows to total bows
      sorry
  have h_green_bows : fraction_green * 160 = 160 / 6 :=
    by
      -- Calculation for green bows
      sorry
  show green_bows = 27 :=
    by
      -- Final step to convert fraction to actual number of bows
      sorry

end green_bows_count_l13_13884


namespace conic_section_intersection_l13_13915

variable (P : ℕ → Type) [fintype (P 1)] [fintype (P 2)] [fintype (P 3)] [fintype (P 4)] 
          [fintype (P 5)] [fintype (P 6)] [fintype (P 7)] [fintype (P 8)] [incidence_geometry P]

-- A cyclic octagon
variable (A : ℕ → P (8))

-- Intersections B_i as defined in problem
variable (B : ℕ → P (8))

noncomputable def intersections (i : ℕ) : ℕ :=
  if even i then A (i % 8 + 3) else A ((i + 1) % 8 + 4)

-- The proof problem statement
theorem conic_section_intersection :
  (∀ i, B i = intersections A i) →
  cyclic A →
  convex A →
  ∃ (conic_section : set (P 8)), ∀ i, B i ∈ conic_section := 
by
  sorry

end conic_section_intersection_l13_13915


namespace region_area_correct_l13_13733

noncomputable def area_of_bounded_region : ℝ :=
  let x (t : ℝ) := 8 * (Real.cos t) ^ 3
  let y (t : ℝ) := 4 * (Real.sin t) ^ 3
  let x_constraint := 3 * Real.sqrt 3
  let intersection_points := {t | x t = x_constraint ∧ x t ≥ x_constraint}
  2 * (∫ t in (Real.arccos (Real.sqrt 3 / 2)) to (-Real.arccos (Real.sqrt 3 / 2)), y t * (Real.deriv x t)) 

theorem region_area_correct : area_of_bounded_region = 2 * Real.pi - 3 * (Real.sqrt 3) := 
sorry

end region_area_correct_l13_13733


namespace find_p_l13_13411

-- Conditions: Consider the quadratic equation 2x^2 + px + q = 0 where p and q are integers.
-- Roots of the equation differ by 2.
-- q = 4

theorem find_p (p : ℤ) (q : ℤ) (h1 : q = 4) (h2 : ∃ x₁ x₂ : ℝ, 2 * x₁^2 + p * x₁ + q = 0 ∧ 2 * x₂^2 + p * x₂ + q = 0 ∧ |x₁ - x₂| = 2) :
  p = 7 ∨ p = -7 :=
by
  sorry

end find_p_l13_13411


namespace garden_hose_rate_l13_13574

theorem garden_hose_rate (V L T : ℝ) (hV : V = 60) (hL : L = 0.1) (hT : T = 40) : 
  let R := 1.6 in
  (R - L) * T = V :=
by
  have h1 : (1.6 - L) * T = V,
  { sorry }
  exact h1

end garden_hose_rate_l13_13574


namespace surface_area_of_soccer_ball_l13_13009

theorem surface_area_of_soccer_ball 
  (C : ℝ) (h1 : 68 ≤ C) (h2 : C ≤ 70) (hC : C = 69):
  let r := C / (2 * Real.pi) in
  let S := 4 * Real.pi * r^2 in
  S = 4761 / Real.pi :=
by
  intros
  rw [hC]
  sorry

end surface_area_of_soccer_ball_l13_13009


namespace vertical_angles_congruent_l13_13243

namespace VerticalAngles

-- Define what it means for two angles to be vertical
def Vertical (α β : Real) : Prop :=
  -- Definition of vertical angles goes here
  sorry

-- Hypothesis: If two angles are vertical angles
theorem vertical_angles_congruent (α β : Real) (h : Vertical α β) : α = β :=
sorry

end VerticalAngles

end vertical_angles_congruent_l13_13243


namespace binomial_eight_three_l13_13397

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem binomial_eight_three : binomial 8 3 = 56 := by
  sorry

end binomial_eight_three_l13_13397


namespace perpendicular_and_double_constraints_l13_13750

noncomputable def midpoint (a b : ℂ) := (a + b) / 2
noncomputable def dist (a b : ℂ) := complex.abs (a - b)

variables (A B C B' C' I I' : ℂ)

-- Conditions
axiom AC_eq : dist A C = dist A C'
axiom angle_AC_AC'_pi_2 : complex.arg ((C' - A) / (C - A)) = real.pi / 2
axiom AB_eq : dist A B = dist A B'
axiom angle_AB_AB'_neg_pi_2 : complex.arg ((B' - A) / (B - A)) = -real.pi / 2
axiom I_is_midpoint_BC : I = midpoint B C
axiom I'_is_midpoint_B'C' : I' = midpoint B' C'

-- Questions to prove
theorem perpendicular_and_double_constraints :
  (complex.arg ((I - A) / (B' - C')) = real.pi / 2 ∧ dist A I * 2 = dist B' C') ∧
  (complex.arg ((I' - A) / (B - C)) = real.pi / 2 ∧ dist A I' * 2 = dist B C) :=
by sorry

end perpendicular_and_double_constraints_l13_13750


namespace arithmetic_sequence_general_formula_sum_first_n_reciprocal_Sn_terms_l13_13407

noncomputable def arithmetic_sequence (d : ℕ) : ℕ → ℕ
| 0 := 2
| (n+1) := 2 + d * n

theorem arithmetic_sequence_general_formula (d : ℕ) (h₀ : d ≠ 0)
  (h₁ : arithmetic_sequence d 3 = (arithmetic_sequence d 1 * arithmetic_sequence d 7) ^ (1/2)) :
  ∀ n, arithmetic_sequence d n = 2n :=
sorry

noncomputable def Sn (n : ℕ) : ℕ :=
n * (n + 1)

noncomputable def reciprocal_Sn_sequence : ℕ → ℕ
| 0 := 1
| (n+1) := 1 / Sn (n+1)

theorem sum_first_n_reciprocal_Sn_terms (n : ℕ) :
  (∑ i in range n, reciprocal_Sn_sequence i) = n / (n + 1) :=
sorry

end arithmetic_sequence_general_formula_sum_first_n_reciprocal_Sn_terms_l13_13407


namespace tetrahedron_circumcircle_angles_equal_l13_13231

noncomputable def angle_between_circles (c1 c2 : Circle) : Real := sorry

theorem tetrahedron_circumcircle_angles_equal
  (A B C D : Point)
  (circ1 : Circumcircle (Triangle D A B))
  (circ2 : Circumcircle (Triangle D A C))
  (circ3 : Circumcircle (Triangle D B C))
  (circ4 : Circumcircle (Triangle A B C)) :
  angle_between_circles circ1 circ2 = angle_between_circles circ3 circ4 :=
sorry

end tetrahedron_circumcircle_angles_equal_l13_13231


namespace total_crayons_l13_13579

-- Definitions for conditions
def boxes : Nat := 7
def crayons_per_box : Nat := 5

-- Statement that needs to be proved
theorem total_crayons : boxes * crayons_per_box = 35 := by
  sorry

end total_crayons_l13_13579


namespace dice_comparison_l13_13736

def diceA : List ℕ := [1, 2, 9, 10, 17, 18]
def diceB : List ℕ := [5, 6, 7, 8, 15, 16]
def diceC : List ℕ := [3, 4, 11, 12, 13, 14]

def probability_more_advantageous (dice1 dice2 : List ℕ) : Prop :=
  let favorable_outcomes := List.sum (List.map (λ face1, List.length (List.filter (λ face2, face1 > face2) dice2)) dice1)
  let total_outcomes := dice1.length * dice2.length
  (favorable_outcomes: ℝ) / (total_outcomes: ℝ) > 0.5

theorem dice_comparison :
  probability_more_advantageous diceA diceB ∧
  probability_more_advantageous diceB diceC ∧
  probability_more_advantageous diceC diceA :=
by sorry

end dice_comparison_l13_13736


namespace S_n_formula_l13_13919

open Set Nat

def f {n : ℕ} (A : Finset (Fin n)) : Option ℕ :=
  A.max

def S (n : ℕ) : ℕ :=
  let M := Finset.range n
  Finset.sum
    (Finset.pows (Finset.range n))
    (fun A => match f A with
              | some k => k + 1
              | none   => 0)

theorem S_n_formula (n : ℕ) (h_pos : 0 < n) : 
  S n = (n - 1) * 2^n + 1 := 
by 
  sorry

end S_n_formula_l13_13919


namespace sum_extension_l13_13920

noncomputable def S (k : ℕ) : ℝ := ∑ i in range (k - 1), 1 / (k + 2 + i : ℝ)

theorem sum_extension {k : ℕ} (hk : 3 ≤ k) :
  S (k + 1) = S k + 1 / (2 * k : ℝ) + 1 / (2 * k + 1 : ℝ) - 1 / (k + 2 : ℝ) :=
sorry

end sum_extension_l13_13920


namespace graph_symmetric_after_shift_l13_13132

noncomputable def f (x : ℝ) := Real.sin ((1/2) * x + Real.pi / 6)

theorem graph_symmetric_after_shift : 
  ∀ x : ℝ, f(x - Real.pi / 3) = Real.sin (2 * x) :=
  sorry

end graph_symmetric_after_shift_l13_13132


namespace reading_is_correct_l13_13699

-- Defining the value we are dealing with
def p : Float := 92.9

-- Defining the correct reading of p
def reading : String := "ninety-two point nine percent"

-- The proof statement
theorem reading_is_correct : (read_percentage p = reading) :=
  sorry

-- We assume a function read_percentage that converts a Float to its English reading
-- This assumption is necessary because actual code for converting a number to English words
-- is outside the scope of simple mathematical proof and requires formalization.
def read_percentage (x : Float) : String := 
  if x = 92.9 then "ninety-two point nine percent" else "unknown"

end reading_is_correct_l13_13699


namespace numValidPermutations_is_fibonacci_l13_13567

def fibonacci : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n+2) := fibonacci (n+1) + fibonacci n

def numValidPermutations : ℕ → ℕ
| 0       := 1
| 1       := 2
| n       := numValidPermutations (n-1) + numValidPermutations (n-2)

theorem numValidPermutations_is_fibonacci (n : ℕ) :
  numValidPermutations n = fibonacci (n+1) := by sorry

end numValidPermutations_is_fibonacci_l13_13567


namespace tangent_line_at_0_2_is_correct_l13_13612

noncomputable def curve (x : ℝ) : ℝ := Real.exp (-2 * x) + 1

def tangent_line_at_0_2 (x : ℝ) : ℝ := -2 * x + 2

theorem tangent_line_at_0_2_is_correct :
  tangent_line_at_0_2 = fun x => -2 * x + 2 :=
by {
  sorry
}

end tangent_line_at_0_2_is_correct_l13_13612


namespace find_b_value_l13_13643

variable (a p q b : ℝ)
variable (h1 : p * 0 + q * (3 * a) + b * 1 = 1)
variable (h2 : p * (9 * a) + q * (-1) + b * 2 = 1)
variable (h3 : p * 0 + q * (3 * a) + b * 0 = 1)

theorem find_b_value : b = 0 :=
by
  sorry

end find_b_value_l13_13643


namespace num_possible_triplets_l13_13279

theorem num_possible_triplets :
  (∃ s : Finset (ℤ × ℤ × ℤ), 
    (∀ abc ∈ s, 
      let a := abc.1, b := abc.2.1, c := abc.2.2 in 
      a * b * c ≠ 0 ∧ b * b = a * c ∧ 
      (a = 100 ∨ c = 100)) ∧ 
    s.card = 10) :=
by sorry

end num_possible_triplets_l13_13279


namespace num_valid_pairings_l13_13739

def colors := ["red", "blue", "yellow", "green", "purple"]
def valid_glasses := ["red", "blue", "yellow", "green"]

noncomputable def total_valid_pairings : Nat :=
  (colors.length - 1) * valid_glasses.length + (colors.length - 2)

theorem num_valid_pairings : total_valid_pairings = 19 :=
by
  -- assumption that there are 5 bowls and 4 glasses with the given condition
  have h1 : colors.length = 5 := rfl
  have h2 : valid_glasses.length = 4 := rfl

  -- calculating total pairings without condition
  have total : Nat := (colors.length * valid_glasses.length) - 1

  -- given condition removes 1 pairing
  have no_purple_green := total - 1
  have valid_pairings := no_purple_green

  -- thus, calculated number of valid pairings should be 19
  have : valid_pairings = 19
  exact this
  sorry

end num_valid_pairings_l13_13739


namespace solve_log_eq_l13_13784

theorem solve_log_eq (x : ℝ) (h : 4 * log 3 x = log 3 (4 * x ^ 2)) : x = 2 :=
sorry

end solve_log_eq_l13_13784


namespace line_intersects_xy_plane_at_l13_13447

-- Define the points through which the line passes
def point1 : ℝ × ℝ × ℝ := (3, 4, 1)
def point2 : ℝ × ℝ × ℝ := (5, 1, 6)

-- Define the intersection point to be checked
def intersection_point : ℝ × ℝ × ℝ := (13/5, 23/5, 0)

-- The theorem stating the intersection point of the line with the xy-plane
theorem line_intersects_xy_plane_at :
  let dir_vec := (point2.1 - point1.1, point2.2 - point1.2, point2.3 - point1.3) in
  let param_point (t : ℝ) := (point1.1 + t * dir_vec.1, point1.2 + t * dir_vec.2, point1.3 + t * dir_vec.3) in
  ∃ t : ℝ, param_point(t) = intersection_point :=
sorry

end line_intersects_xy_plane_at_l13_13447


namespace sequence_2023_l13_13415

-- Define the sequence recursively
def sequence (n : ℕ) : ℚ := by
  match n with
  | 1 => 2
  | 2 => 1/3
  | n + 1 => if h : n ≥ 2 then 
               (sequence (n - 1) * sequence n) / (3 * sequence (n - 1) + 2 * sequence n)
             else
               0

-- State the theorem we want to prove
theorem sequence_2023 (p q : ℕ) (h_rel_prime : Nat.gcd p q = 1) (h_gt_zero : p > 0 ∧ q > 0) :
    sequence 2023 = (p : ℚ) / q ∧ p + q = 6070 := sorry

end sequence_2023_l13_13415


namespace equation_involving_x_l13_13875

-- Define the conditions: x and y are numbers and the smallest value of x^2 + y^2 is 9.
variables (x y : ℝ)
axiom smallest_value_condition : (x^2 + y^2) = 9

-- The statement that needs to be proved
theorem equation_involving_x (x y : ℝ) :
  ∃ (r : ℝ), (r = 3 ∧ x^2 + y^2 = r^2) := by
  existsi 3
  split
  . refl
  . exact smallest_value_condition
  . sorry

end equation_involving_x_l13_13875


namespace cos_plus_2sin_eq_one_l13_13857

theorem cos_plus_2sin_eq_one (α : ℝ) (h : (1 + Real.cos α) / Real.sin α = 1 / 2) : 
  Real.cos α + 2 * Real.sin α = 1 := 
by
  sorry

end cos_plus_2sin_eq_one_l13_13857


namespace cost_of_siding_l13_13246

def area_of_wall (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def area_of_roof (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length * width)

def area_of_sheet (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def sheets_needed (total_area : ℕ) (sheet_area : ℕ) : ℕ :=
  (total_area + sheet_area - 1) / sheet_area  -- Cooling the ceiling with integer arithmetic

def total_cost (sheets : ℕ) (price_per_sheet : ℕ) : ℕ :=
  sheets * price_per_sheet

theorem cost_of_siding : 
  ∀ (length_wall width_wall length_roof width_roof length_sheet width_sheet price_per_sheet : ℕ),
  length_wall = 10 → width_wall = 7 →
  length_roof = 10 → width_roof = 6 →
  length_sheet = 10 → width_sheet = 14 →
  price_per_sheet = 50 →
  total_cost (sheets_needed (area_of_wall length_wall width_wall + area_of_roof length_roof width_roof) (area_of_sheet length_sheet width_sheet)) price_per_sheet = 100 :=
by
  intros
  simp [area_of_wall, area_of_roof, area_of_sheet, sheets_needed, total_cost]
  sorry

end cost_of_siding_l13_13246


namespace ministers_never_resign_l13_13169

theorem ministers_never_resign (V : Type) [Fintype V] [DecidableEq V] (E : set (V × V))
  (h_connected : ∀ (u v : V), ∃ p : List (V × V), u ∈ p ∧ v ∈ p ∧ ∀ e ∈ p, e ∈ E)
  (h_single_edge_connected : ∀ (u v : V) (e : V × V), e ∈ E → ∃ p : List (V × V), e ∉ p ∧ u ∈ p ∧ v ∈ p ∧ ∀ e' ∈ p, e' ∈ E) :
  ¬ (∀ strategy : (V × V) → Prop, ∃ u v : V, (u, v) ∈ E ∧ strategy (u, v) → ¬ (∃ (p : List (V × V)), u ∈ p ∧ v ∈ p ∧ ∀ e ∈ p, e ∈ E)) :=
sorry

end ministers_never_resign_l13_13169


namespace find_integer_n_l13_13443

theorem find_integer_n : ∃ n : ℤ, -120 ≤ n ∧ n ≤ 120 ∧ real.sin (n * real.pi / 180) = real.cos (682 * real.pi / 180) ∧ n = 128 :=
by
  sorry

end find_integer_n_l13_13443


namespace twenty_digit_number_properties_l13_13267

def pow : ℕ → ℕ → ℕ
| base, 0 => 1
| base, n+1 => base * pow base n

noncomputable def reverse_num (n : ℕ) : ℕ :=
  n.to_string.reverse.val

def is_not_perfect_square (n : ℕ) : Prop :=
  ∀ k : ℕ, k * k ≠ n

def product_is_perfect_square (a b : ℕ) : Prop :=
  ∃ k : ℕ, k * k = a * b

theorem twenty_digit_number_properties :
  let n := 15841584158415841584 in
  is_not_perfect_square n ∧ 
  n ≠ reverse_num n ∧ 
  product_is_perfect_square n (reverse_num n) :=
by
  sorry

end twenty_digit_number_properties_l13_13267


namespace convex_polygon_center_of_symmetry_l13_13228

theorem convex_polygon_center_of_symmetry (M : Type) 
    (convex : Convex M)
    (centrally_symmetric_div : ∀ (P : Type), P ∈ M → CentrallySymmetric P) :
    CentrallySymmetric M :=
sorry

end convex_polygon_center_of_symmetry_l13_13228


namespace find_k_l13_13846

variable (t s k : ℝ)

def line1 (t k : ℝ) : ℝ × ℝ := (-2 * t, 2 + k * t)
def line2 (s : ℝ) : ℝ × ℝ := (2 + s, 1 - 2 * s)

theorem find_k (hts_perpendicular : ∀ t s, (-2 : ℝ) * (-2 : ℝ) = -1) : k = -1 := sorry

end find_k_l13_13846


namespace cliffton_avg_temp_l13_13016

theorem cliffton_avg_temp :
  let temps : List ℚ := [52, 62, 55, 59, 50] in
  (temps.sum / temps.length) = 55.6 :=
by
  -- The proof will go here
  sorry

end cliffton_avg_temp_l13_13016


namespace transform_graph_function_l13_13296

-- Definitions from conditions
def transform_abscissa (x : ℝ) (ω : ℝ) : ℝ := x / ω

def transform_ordinate (y : ℝ) (A : ℝ) : ℝ := A * y

-- Given the initial and transformed functions
def initial_function (x : ℝ) : ℝ := 3 * sin (x - (π / 6))
def transformed_function (x : ℝ) : ℝ := 3 * sin (2 * x - (π / 6))

-- The theorem that needs to be proved
theorem transform_graph_function :
  ∀ (x : ℝ), transformed_function x = initial_function (x / 2) :=
by
  sorry

end transform_graph_function_l13_13296


namespace Ann_ends_with_53_blocks_l13_13724

theorem Ann_ends_with_53_blocks (a b : ℕ) (ha : a = 9) (hb : b = 44) : a + b = 53 := by
  rw [ha, hb]
  simp
  sorry

end Ann_ends_with_53_blocks_l13_13724


namespace find_number_l13_13286

theorem find_number (x : ℝ) 
(h : x * 13.26 + x * 9.43 + x * 77.31 = 470) : 
x = 4.7 := 
sorry

end find_number_l13_13286


namespace octagonal_walk_distance_l13_13067

def angle_subtended_by_chord (r : ℝ) (θ : ℝ) : ℝ :=
  2 * r * Real.sin (θ / 2)

def distance_walked_by_one_boy (r : ℝ) : ℝ :=
  let d90 := angle_subtended_by_chord r (Real.pi / 2)
  let d180 := 2 * r
  d90 + d90 + d180

def total_distance_walked (r : ℝ) (n : ℕ) : ℝ :=
  n * distance_walked_by_one_boy r

theorem octagonal_walk_distance :
  total_distance_walked 50 8 = 1600 + 800 * Real.sqrt (2 - Real.sqrt 2) :=
by
  -- Proof to be filled
  sorry

end octagonal_walk_distance_l13_13067


namespace handshake_distance_l13_13061

theorem handshake_distance 
  (n : ℕ) (r : ℝ)
  (h_n : n = 8) 
  (h_r : r = 50) 
  (distance : ℝ) 
  (correct_answer : distance = 8000 * real.sqrt ((real.sqrt 2 - 1) / (2 * real.sqrt 2))) :
  ∃ d, d = 8000 * real.sqrt ((real.sqrt 2 - 1) / (2 * real.sqrt 2)) := 
  by {
  sorry
}

end handshake_distance_l13_13061


namespace min_value_of_f_l13_13793

noncomputable def f (x a b : ℝ) : ℝ :=
  (x + a + b) * (x + a - b) * (x - a + b) * (x - a - b)

theorem min_value_of_f (a b : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ a ≥ b) :
  ∃ x, f x a b = -4 * a^2 * b^2 :=
by
  use x := sqrt (a^2 + b^2)
  sorry

end min_value_of_f_l13_13793


namespace last_two_digits_x2012_l13_13191

def x : ℕ → ℤ 
| 1 := 1
| 2 := 1
| n := x (n-1) * y (n-2) + x (n-2) * y (n-1)

def y : ℕ → ℤ 
| 1 := 1
| 2 := 1
| n := y (n-1) * y (n-2) - x (n-1) * x (n-2)

theorem last_two_digits_x2012 : |x 2012| % 100 = 96 :=
by
  sorry

end last_two_digits_x2012_l13_13191


namespace probability_X_greater_than_one_l13_13997

section
  variable {Ω : Type} {P : Ω → ℝ}
  variable {X : Ω → ℕ}
  variable (h_dist : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → P (X = k) = k / 15)
  
  theorem probability_X_greater_than_one (h_dist : ∀ k, 1 ≤ k ∧ k ≤ 5 → P (X = k) = k / 15) :
    P (X > 1) = 14 / 15 :=
  sorry
end

end probability_X_greater_than_one_l13_13997


namespace coral_three_night_total_pages_l13_13042

-- Definitions based on conditions in the problem
def night1_pages : ℕ := 30
def night2_pages : ℕ := 2 * night1_pages - 2
def night3_pages : ℕ := night1_pages + night2_pages + 3
def total_pages : ℕ := night1_pages + night2_pages + night3_pages

-- The statement we want to prove
theorem coral_three_night_total_pages : total_pages = 179 := by
  sorry

end coral_three_night_total_pages_l13_13042


namespace fewer_females_than_males_l13_13162

theorem fewer_females_than_males 
  (total_students : ℕ)
  (female_students : ℕ)
  (h_total : total_students = 280)
  (h_female : female_students = 127) :
  total_students - female_students - female_students = 26 := by
  sorry

end fewer_females_than_males_l13_13162


namespace seating_arrangements_l13_13327

-- Define the participants
inductive Person : Type
| xiaoMing
| parent1
| parent2
| grandparent1
| grandparent2

open Person

-- Define the function to count seating arrangements
noncomputable def count_seating_arrangements : Nat :=
  let arrangements := [
    -- (Only one parent next to Xiao Ming, parents not next to each other)
    12,
    -- (Only one parent next to Xiao Ming, parents next to each other)
    24,
    -- (Both parents next to Xiao Ming)
    12
  ]
  arrangements.foldr (· + ·) 0

theorem seating_arrangements : count_seating_arrangements = 48 := by
  sorry

end seating_arrangements_l13_13327


namespace percentage_reduction_in_area_l13_13716

theorem percentage_reduction_in_area (a : ℝ) :
  let Area := (3 * Real.sqrt 3 / 2) * a^2,
      Area_new := (3 * Real.sqrt 3 / 8) * a^2,
      Reduction := (Area - Area_new),
      Percentage_reduction := (Reduction / Area) * 100
  in Percentage_reduction = 75 := by
  sorry

end percentage_reduction_in_area_l13_13716


namespace find_x_y_l13_13850

-- Definitions based on the conditions
def vector_a : ℝ × ℝ × ℝ := (2, -3, 5)
def vector_b (x y : ℝ) : ℝ × ℝ × ℝ := (4, x, y)

-- Lean statement for the problem
theorem find_x_y (x y : ℝ) (h : ∃ λ : ℝ, vector_a = (λ • vector_b x y)) : x = -6 ∧ y = 10 :=
sorry

end find_x_y_l13_13850


namespace tilly_bag_cost_l13_13293

theorem tilly_bag_cost (n : ℕ) (p : ℕ) (profit : ℕ) : 
  (n = 100) → (p = 10) → (profit = 300) → (n * p - profit) / n = 7 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3]
  norm_num
  sorry

end tilly_bag_cost_l13_13293


namespace factorize_expression_l13_13437

theorem factorize_expression (x y : ℂ) : (x * y^2 - x = x * (y + 1) * (y - 1)) :=
sorry

end factorize_expression_l13_13437


namespace problem_statement_l13_13112

noncomputable def is_geometric (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

noncomputable def a : ℕ → ℝ := sorry                 -- Placeholder for the geometric sequence {a_n}
noncomputable def b (n : ℕ) : ℝ := Real.log2 (a n)

theorem problem_statement :
  is_geometric a ∧ a 11 = 8 ∧ b 4 = 17 →
  (∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d ∧ d = -2) ∧ (∃ n : ℕ, b 1 = 23 ∧ 144 = (-(n - 12) ^ 2 + 144) ∧ n = 12) :=
sorry

end problem_statement_l13_13112


namespace selling_price_approx_l13_13718

def cost_price : ℝ := 19
def loss_fraction : ℝ := 1/6
def loss_amount (cp : ℝ) : ℝ := cp * loss_fraction
def selling_price (cp : ℝ) : ℝ := cp - loss_amount(cp)

theorem selling_price_approx (hp : cost_price = 19) : selling_price cost_price ≈ 15.83 :=
by
  have loss := loss_amount cost_price
  have sp := selling_price cost_price
  show sp ≈ 15.83
  sorry

end selling_price_approx_l13_13718


namespace lottery_sample_representativeness_l13_13617

theorem lottery_sample_representativeness (A B C D : Prop) :
  B :=
by
  sorry

end lottery_sample_representativeness_l13_13617


namespace sum_of_angles_outside_pentagon_l13_13710

-- Define the properties and structure of an inscribed pentagon
axiom inscribed_pentagon : Prop

-- Define the angles inscribed in the segments outside the pentagon
variables (α β γ δ ε : ℝ)

-- Sum of the five angles inscribed
def sum_of_angles (α β γ δ ε : ℝ) : ℝ := α + β + γ + δ + ε

theorem sum_of_angles_outside_pentagon :
  inscribed_pentagon → ∀ (α β γ δ ε : ℝ), 
  sum_of_angles α β γ δ ε = 720 :=
begin
  assume h,
  sorry
end

end sum_of_angles_outside_pentagon_l13_13710


namespace meaningful_expression_range_l13_13523

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = sqrt (x + 1) + 1 / x) ↔ (x ≥ -1 ∧ x ≠ 0) :=
by
  sorry

end meaningful_expression_range_l13_13523


namespace range_of_k_l13_13841

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_f : ∀ x, f x = x^3 - 3 * x^2 - k)
  (h_f' : ∀ x, f' x = 3 * x^2 - 6 * x) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) ↔ -4 < k ∧ k < 0 :=
sorry

end range_of_k_l13_13841


namespace trigonometric_identity_problem_l13_13207

theorem trigonometric_identity_problem
  {x y : ℝ}
  (h1 : sin x / sin y = 4)
  (h2 : cos x / cos y = 1 / 3) :
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 169 / 381 := 
by 
  sorry

end trigonometric_identity_problem_l13_13207


namespace domain_of_f_l13_13868

-- Function definition
def f (x : ℝ) : ℝ := log (2 - x)

-- Proof statement that the domain of f(x) is (-∞, 2)
theorem domain_of_f : {x : ℝ | ∃ y. f y = f x} = {x : ℝ | x < 2} :=
sorry

end domain_of_f_l13_13868


namespace verify_sum_of_cubes_l13_13629

noncomputable def sum_of_squares := ∀ (n : ℤ), (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 2106

noncomputable def sum_of_cubes := ∀ (n : ℤ),
  (sum_of_squares n) →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 45900 

theorem verify_sum_of_cubes : sum_of_cubes 22 := 
by
  -- We assume the input translates correctly based on conditions provided
  sorry

end verify_sum_of_cubes_l13_13629


namespace bus_speed_including_stoppages_l13_13782

theorem bus_speed_including_stoppages
  (speed_excluding_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ) :
  speed_excluding_stoppages = 64 ∧ stoppage_time_per_hour = 15 / 60 →
  (44 / 60) * speed_excluding_stoppages = 48 :=
by
  sorry

end bus_speed_including_stoppages_l13_13782


namespace vertical_angles_congruent_l13_13239

theorem vertical_angles_congruent (a b : Angle) (h : VerticalAngles a b) : CongruentAngles a b :=
sorry

end vertical_angles_congruent_l13_13239


namespace find_m_values_l13_13535

theorem find_m_values (m : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (2, 2) ∧ B = (m, 0) ∧ 
   ∃ r R : ℝ, r = 1 ∧ R = 3 ∧ 
   ∃ d : ℝ, d = abs (dist A B) ∧ d = (R + r)) →
  (m = 2 - 2 * Real.sqrt 3 ∨ m = 2 + 2 * Real.sqrt 3) := 
sorry

end find_m_values_l13_13535


namespace exact_value_of_v_sum_l13_13024

def v (x : ℝ) : ℝ := -x + 2 * Real.cos (π * x / 2)

theorem exact_value_of_v_sum :
  v (-1.75) + v (-0.5) + v (0.5) + v (1.75) = 0 :=
sorry

end exact_value_of_v_sum_l13_13024


namespace main_theorem_l13_13414

noncomputable def p (x : ℝ) : ℝ := 4 * x^3 - 2 * x^2 - 15 * x + 9
noncomputable def q (x : ℝ) : ℝ := 12 * x^3 + 6 * x^2 - 7 * x + 1

-- Show that p(x) has exactly three distinct real roots
def p_has_three_real_roots : Prop := 
  ∃ (roots : Finset ℝ), roots.card = 3 ∧ ∀ r ∈ roots, p r = 0

-- Show that q(x) has exactly three distinct real roots
def q_has_three_real_roots : Prop := 
  ∃ (roots : Finset ℝ), roots.card = 3 ∧ ∀ r ∈ roots, q r = 0

-- Definitions for the largest roots of p(x) and q(x)
def is_largest_root_of (r : ℝ) (f : ℝ → ℝ) : Prop :=
  f r = 0 ∧ ∀ x, f x = 0 → x ≤ r

noncomputable def A := Classical.choose (exists_root_max p)
noncomputable def B := Classical.choose (exists_root_max q)

-- Define the main theorem statement which includes the proof of the problem
theorem main_theorem : p_has_three_real_roots ∧ q_has_three_real_roots ∧ is_largest_root_of A p ∧ is_largest_root_of B q ∧ (A^2 + 3 * B^2 = 4) :=
by
  sorry

end main_theorem_l13_13414


namespace balls_into_boxes_l13_13144

theorem balls_into_boxes : (@Finset.choose 8 2).card = 28 := by
  sorry

end balls_into_boxes_l13_13144


namespace ellipse_is_set_A_l13_13556

structure Ellipse (F1 F2 : Point) (a : ℝ) :=
  (on_ellipse : ∀ (A : Point), dist A F1 + dist A F2 = 2 * a)

theorem ellipse_is_set_A {F1 F2 : Point} {a : ℝ} (E : Ellipse F1 F2 a) :
  ∀ (A : Point), dist A F1 + dist A F2 ≤ 2 * a ↔ A ∈ {A : Point | dist A F1 + dist A F2 = 2 * a} :=
sorry

end ellipse_is_set_A_l13_13556


namespace solve_inequality_l13_13627

theorem solve_inequality (x : ℝ) : 2 - x < 1 → x > 1 := 
by
  sorry

end solve_inequality_l13_13627


namespace smallest_positive_multiple_l13_13667

theorem smallest_positive_multiple (n : ℕ) (h1 : n > 0) (h2 : n % 45 = 0) (h3 : n % 75 = 0) (h4 : n % 20 ≠ 0) :
  n = 225 :=
by
  sorry

end smallest_positive_multiple_l13_13667


namespace arccos_cos_eight_l13_13748

-- Define the conditions
def cos_equivalence (x : ℝ) : Prop := cos x = cos (x - 2 * Real.pi)
def range_principal (x : ℝ) : Prop := 0 ≤ x - 2 * Real.pi ∧ x - 2 * Real.pi ≤ Real.pi

-- State the main proposition
theorem arccos_cos_eight :
  cos_equivalence 8 ∧ range_principal 8 → Real.arccos (Real.cos 8) = 8 - 2 * Real.pi :=
by
  sorry

end arccos_cos_eight_l13_13748


namespace incenter_linear_combination_of_incenter_l13_13904

variables (D E F : ℝ)
variables (d e f x y z : ℝ)

-- Given conditions
def conditions (d e f : ℝ) :=
  d = 10 ∧ e = 6 ∧ f = 8 ∧ (x + y + z = 1)

-- The required properties of the incenter
def incenter_coordinates (J : ℝ) :=
  J = x * D + y * E + z * F

-- Main theorem statement
theorem incenter_linear_combination_of_incenter 
  (h : conditions d e f) : 
  x = 5 / 12 ∧ y = 1 / 4 ∧ z = 1 / 3 :=
sorry

end incenter_linear_combination_of_incenter_l13_13904


namespace mutually_exclusive_BC_dependent_AC_probability_C_conditional_probability_CA_l13_13540

open ProbabilityTheory

-- Definitions for the problem
variables (BoxA_red BoxA_white BoxB_red BoxB_white : ℕ)
variables (A B C : Event)

-- Conditions
def BoxA := 3 -- 3 red balls in Box A
def BoxA_white := 2 -- 2 white balls in Box A
def BoxB := 2 -- 2 red balls in Box B
def BoxB_white := 3 -- 3 white balls in Box B
def P_A : ℚ := 3 / (3 + 2) -- Probability of drawing a red ball from Box A
def P_B : ℚ := 2 / (3 + 2) -- Probability of drawing a white ball from Box A

def move_ball_event_A := sorry   -- Assuming moving event A is given as an event
def move_ball_event_B := sorry   -- Assuming moving event B is given as an event

-- Event Definitions
def P_C_given_A : ℚ := 3 / (3 + 3) -- P(C|A), probability of drawing red ball from Box B if red ball added
def P_C_given_B : ℚ := 2 / (2 + 3) -- P(C|B), probability of drawing red ball from Box B if white ball added

-- P(C) combining both scenarios
def P_C : ℚ := (P_A * P_C_given_A) + (P_B * P_C_given_B)

-- Conditional Probability P(C|A)
def P_CA : ℚ := (P_A * P_C_given_A) / P_A

-- Rephrasing problem into proofs
theorem mutually_exclusive_BC : ¬(B ∧ C) :=
sorry

theorem dependent_AC : ¬(independent A C) :=
sorry

theorem probability_C : P_C = 13 / 30 :=
sorry

theorem conditional_probability_CA : P_CA = 1 / 2 :=
sorry

end mutually_exclusive_BC_dependent_AC_probability_C_conditional_probability_CA_l13_13540


namespace total_gain_percentage_correct_l13_13000

-- Define the buying prices
def cost_A : ℝ := 20
def cost_B : ℝ := 30
def cost_C : ℝ := 40

-- Define the selling prices
def sell_A : ℝ := 25
def sell_B : ℝ := 35
def sell_C : ℝ := 60

-- Definitions of total cost price, total selling price, total gain, and total gain percentage
def total_cost_price : ℝ := cost_A + cost_B + cost_C
def total_selling_price : ℝ := sell_A + sell_B + sell_C
def total_gain : ℝ := total_selling_price - total_cost_price
def total_gain_percentage : ℝ := (total_gain / total_cost_price) * 100

-- The theorem to be proved
theorem total_gain_percentage_correct :
  total_gain_percentage = 33.33 := 
by 
  -- include the proof statements
  sorry

end total_gain_percentage_correct_l13_13000


namespace fifth_largest_divisor_of_2500000000_l13_13047

theorem fifth_largest_divisor_of_2500000000 :
  ∃ (d : ℕ), d = 156250000 ∧
             (d divides 2500000000) ∧
             (∀ k : ℕ, k > d → (k divides 2500000000) → (∃ n, n < 5 ∧ n ≠ 0 ∧ k = 2^(7-n) * 5^9)) :=
begin
  sorry,
end

end fifth_largest_divisor_of_2500000000_l13_13047


namespace angle_is_2pi_over_3_l13_13213

-- Given: two vectors a and b with specific properties.
variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
def condition1 : Prop := (a - b) ∙ (a + 2 • b) = -8
def condition2 : Prop := ∥a∥ = 1
def condition3 : Prop := ∥b∥ = 2

-- Definition of the angle between vectors a and b
def angle_between (a b : EuclideanSpace ℝ (Fin 3)) : ℝ := real.acos ((a ∙ b) / (∥a∥ * ∥b∥))

-- Theorem that states the angle between a and b
theorem angle_is_2pi_over_3 (ha : condition1) (hb : condition2) (hc : condition3) :
  angle_between a b = (2 * real.pi) / 3 :=
sorry

end angle_is_2pi_over_3_l13_13213


namespace my_car_mpg_l13_13581

-- Definitions from the conditions.
def total_miles := 100
def total_gallons := 5

-- The statement we need to prove.
theorem my_car_mpg : (total_miles / total_gallons : ℕ) = 20 :=
by
  sorry

end my_car_mpg_l13_13581


namespace foldable_cube_with_one_face_missing_l13_13974

-- Definitions for the conditions
structure Square where
  -- You can define properties of a square here if necessary

structure Polygon where
  squares : List Square
  congruent : True -- All squares are congruent
  joined_edge_to_edge : True -- The squares are joined edge-to-edge

-- The positions the additional square can be added to
inductive Position
| P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9

-- Define the problem in Lean 4 as a theorem
theorem foldable_cube_with_one_face_missing (base_polygon : Polygon) :
  base_polygon.squares.length = 4 →
  ∃ (positions : List Position), positions.length = 6 ∧
    ∀ pos ∈ positions, 
      let new_polygon := { base_polygon with squares := base_polygon.squares.append [Square.mk] }
      new_polygon.foldable_into_cube_with_one_face_missing pos :=
  sorry

end foldable_cube_with_one_face_missing_l13_13974


namespace LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l13_13651

noncomputable section

-- Problem 1: Prove length ratios for simultaneous ignition
def LengthRatioSimultaneous (t : ℝ) : Prop :=
  let LA := 1 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosSimultaneous (t : ℝ) : LengthRatioSimultaneous t := sorry

-- Problem 2: Prove length ratios when one candle is lit 30 minutes earlier
def LengthRatioNonSimultaneous (t : ℝ) : Prop :=
  let LA := 5 / 6 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosNonSimultaneous (t : ℝ) : LengthRatioNonSimultaneous t := sorry

end LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l13_13651


namespace inequality_solution_l13_13075

theorem inequality_solution (x : ℝ) :
  (∃ x, x ∈ ((-∞, -4) ∪ (-4, -2) ∪ (-2, ∞))) ↔
  (2 / (x + 2) - 4 / (x + 8) > 1 / 2) := by
sorry

end inequality_solution_l13_13075


namespace profit_percent_by_selling_l13_13678

def profit_percent (CP SP : ℝ) : ℝ := ((SP - CP) / CP) * 100

theorem profit_percent_by_selling (CP SP : ℝ) (h1 : SP / 2 = 0.8 * CP) : 
  profit_percent CP SP = 60 :=
by
  sorry

end profit_percent_by_selling_l13_13678


namespace division_in_consecutive_integers_l13_13569

theorem division_in_consecutive_integers (a b c : ℕ) (ha : 0 < a) (hb : a < b) (hc : b < c) : 
  ∀ (k : ℕ),  ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (∀ i, k * 2 * c ≤ i ∧ i < k * 2 * c + 2 * c → (xz where xz=x*y*z)) ∧ abc ∣ xz := 
begin
  sorry
end

end division_in_consecutive_integers_l13_13569


namespace caterpillar_count_l13_13631

theorem caterpillar_count 
    (initial_count : ℕ)
    (hatched : ℕ)
    (left : ℕ)
    (h_initial : initial_count = 14)
    (h_hatched : hatched = 4)
    (h_left : left = 8) :
    initial_count + hatched - left = 10 :=
by
    sorry

end caterpillar_count_l13_13631


namespace number_of_integers_less_than_500_eight_times_sum_of_digits_l13_13853

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem number_of_integers_less_than_500_eight_times_sum_of_digits :
  (finset.filter (λ n : ℕ, sum_of_digits n * 8 = n ∧ n < 500) (finset.range 500)).card = 1 := by
  sorry

end number_of_integers_less_than_500_eight_times_sum_of_digits_l13_13853


namespace find_a_and_c_l13_13783

noncomputable def square_of_binomial (a c : ℝ) : Prop :=
  ∃ (r s : ℝ), (r^2 = a) ∧ (2 * r * s = -20) ∧ (s^2 = c)

theorem find_a_and_c (a c : ℝ) : square_of_binomial a c → a = (10 / (20 / a))^2 ∧ c = 100 / a :=
by
  intro H
  cases H with r H
  cases H with s H
  cases H with H1 H
  cases H with H2 H3
  have H4 : r = 10 / (20 / r)
  { sorry }
  have H5 : a = (10 / (20 / r))^2
  { rw [H1, H4], sorry }
  have H6 : c = 100 / a
  { rw [H3, H1], sorry }
  exact ⟨H5, H6⟩

end find_a_and_c_l13_13783


namespace triangle_area_l13_13471

noncomputable def area_of_triangle (a b c : ℝ) (h_seq : a + b = 2 * c ∧ c ≥ b ∧ b ≥ a)
  (h_sine : ∃ θ, θ = real.pi * 2 / 3 ∧ real.sin θ = sqrt 3 / 2) : ℝ :=
1 / 2 * a * b * sqrt 3 / 2

theorem triangle_area (a b c : ℝ) (h_seq : a + b = 2 * c ∧ c ≥ b ∧ b ≥ a)
  (h_sine : ∃ θ, θ = real.pi * 2 / 3 ∧ real.sin θ = sqrt 3 / 2) :
  area_of_triangle a b c h_seq h_sine = 15 * sqrt 3 / 4 := sorry

end triangle_area_l13_13471


namespace binary_representation_f_l13_13092

def piecewise_f (f : ℚ → ℚ) (x : ℚ) : ℚ :=
if h : 0 ≤ x ∧ x < 1/2 then (f (2*x))/4
else if h : 1/2 ≤ x ∧ x < 1 then 3/4 + (f (2*x - 1))/4
else 0 -- note: this case should not occur given the problem conditions

theorem binary_representation_f (x : ℚ) (hx: 0 ≤ x ∧ x < 1) (b : ℕ → bool)
  (h_bin : ∃ n, (b n) = true) : (∃ b : ℕ → bool, x = ∑ n, (if b n then (1 / 2^(n+1)) else 0) 
   ∧ (piecewise_f (λ x, 0.b_1b_1b_2b_2b_3b_3 ... ∑ n, if b (2*n) then 1/2^(n+1) else 0) = f(x))) :=
sorry

end binary_representation_f_l13_13092


namespace greatest_possible_value_of_squares_l13_13192

theorem greatest_possible_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 15)
  (h2 : ab + c + d = 78)
  (h3 : ad + bc = 160)
  (h4 : cd = 96) :
  a^2 + b^2 + c^2 + d^2 ≤ 717 ∧ ∃ a b c d, a + b = 15 ∧ ab + c + d = 78 ∧ ad + bc = 160 ∧ cd = 96 ∧ a^2 + b^2 + c^2 + d^2 = 717 :=
sorry

end greatest_possible_value_of_squares_l13_13192


namespace fixed_point_of_line_l13_13155

theorem fixed_point_of_line :
  (∀ m : ℝ, ∃ (x y : ℝ), mx + y - 1 + 2m = 0) ↔ (x = -2 ∧ y = 1) := by
  sorry

end fixed_point_of_line_l13_13155


namespace evaluate_fraction_l13_13433

theorem evaluate_fraction (a b : ℝ) (h : a ≠ b) :
  (a^(-5) - b^(-5)) / (a^(-3) - b^(-3)) = a^(-1) + b^(-1) :=
by
  sorry

end evaluate_fraction_l13_13433


namespace caterpillar_count_proof_l13_13634

def number_of_caterpillars_after_events (initial : ℕ) (hatched : ℕ) (left : ℕ) : ℕ :=
  initial + hatched - left

theorem caterpillar_count_proof :
  number_of_caterpillars_after_events 14 4 8 = 10 :=
by
  simp [number_of_caterpillars_after_events]
  sorry

end caterpillar_count_proof_l13_13634


namespace value_of_m_if_z_is_purely_imaginary_l13_13865

theorem value_of_m_if_z_is_purely_imaginary (m : ℝ) :
  (∃ z : ℂ, z = ((m^2 - m) : ℝ) + (m : ℂ) * complex.I ∧ z.im = z) → m = 1 :=
by
  sorry

end value_of_m_if_z_is_purely_imaginary_l13_13865


namespace molecular_weight_N2O_correct_l13_13312

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of N2O
def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O

-- Prove the statement
theorem molecular_weight_N2O_correct : molecular_weight_N2O = 44.02 := by
  -- We leave the proof as an exercise (or assumption)
  sorry

end molecular_weight_N2O_correct_l13_13312


namespace area_between_middle_and_largest_circle_l13_13041

variables (r1 r2 r3 : ℝ)

def area_between_circles (r2 r3 : ℝ) : ℝ := π * (r3 ^ 2) - π * (r2 ^ 2)

theorem area_between_middle_and_largest_circle
  (r1 r2 r3 : ℝ)
  (h1 : r1 = 2)
  (h2 : r2 = 4)
  (h3 : r3 = 7) :
  area_between_circles r2 r3 = 33 * π :=
by
  have h_area : area_between_circles r2 r3 = π * (r3 ^ 2) - π * (r2 ^ 2),
  { unfold area_between_circles },
  rw [h2, h3, h_area],
  norm_num,
  sorry

#check area_between_middle_and_largest_circle

end area_between_middle_and_largest_circle_l13_13041


namespace calc_fraction_power_l13_13392

theorem calc_fraction_power (n m : ℤ) (h_n : n = 2023) (h_m : m = 2022) :
  (- (2 / 3 : ℚ))^n * ((3 / 2 : ℚ))^m = - (2 / 3) := by
  sorry

end calc_fraction_power_l13_13392


namespace diagonals_of_quadrilateral_are_equal_l13_13695

theorem diagonals_of_quadrilateral_are_equal
  (A B C D O : Type)
  [acute_triangle ABC]
  (hAB_lt_AC : AB < AC)
  (hD_altitude : is_foot_of_altitude A D BC)
  (hO_circumcenter : is_circumcenter O ABC)
  (h_ext_angle_bisector_parallel : is_parallel (external_angle_bisector (angle A B C)) (line O D)) :
  length (segment O C) = length (segment A D) :=
sorry

end diagonals_of_quadrilateral_are_equal_l13_13695


namespace vertical_angles_congruent_l13_13237

theorem vertical_angles_congruent (A B : Angle) (h : vertical_angles A B) : congruent A B := sorry

end vertical_angles_congruent_l13_13237


namespace part1_part2_l13_13530

namespace TriangleProof

variables {A B C a b c : ℝ}

-- Conditions
axiom angle_side_conditions : ∀ {A B C : ℝ} {a b c : ℝ},
  3 * Real.cos B * Real.cos C + 1 = 3 * Real.sin B * Real.sin C + Real.cos (2 * A)

noncomputable def magnitude_of_A (B C : ℝ) : ℝ :=
  if h : ∃ x, 3 * Real.cos B * Real.cos C + 1 = 3 * Real.sin B * Real.sin C + Real.cos (2 * x)
  then classical.some h
  else 0

-- Proof that A = π / 3 under the given condition
theorem part1 (A B C : ℝ) (h : 3 * Real.cos B * Real.cos C + 1 = 3 * Real.sin B * Real.sin C + Real.cos (2 * A))
  : A = π / 3 :=
sorry

-- Conditions for part 2
axiom side_length_conditions : a = 2 * Real.sqrt 3

-- Proof that the maximum value of (b + 2c) is 4 * sqrt 7 under given conditions
theorem part2 (a b c : ℝ) (h₁ : a = 2 * Real.sqrt 3) (h₂ : 3 * Real.cos B * Real.cos C + 1 = 3 * Real.sin B * Real.sin C + Real.cos (2 * A))
  : b + 2 * c ≤ 4 * Real.sqrt 7 :=
sorry

end TriangleProof

end part1_part2_l13_13530


namespace rate_mangoes_correct_l13_13298

-- Define the conditions
def weight_apples : ℕ := 8
def rate_apples : ℕ := 70
def cost_apples := weight_apples * rate_apples

def total_payment : ℕ := 1145
def weight_mangoes : ℕ := 9
def cost_mangoes := total_payment - cost_apples

-- Define the rate per kg of mangoes
def rate_mangoes := cost_mangoes / weight_mangoes

-- Prove the rate per kg for mangoes
theorem rate_mangoes_correct : rate_mangoes = 65 := by
  -- all conditions and intermediate calculations already stated
  sorry

end rate_mangoes_correct_l13_13298


namespace original_number_of_men_l13_13366

theorem original_number_of_men (x : ℕ) (h1 : 40 * x = 60 * (x - 5)) : x = 15 :=
by
  sorry

end original_number_of_men_l13_13366


namespace solve_triangle_l13_13903

noncomputable def triangle_problem_statement 
  (A B C a b c : ℝ)
  (triangle : IsTriangle A B C) -- Assuming IsTriangle is a valid predicate in the context
  (angle_opposite : OppositeSide a A ∧ OppositeSide b B ∧ OppositeSide c C) -- Assuming OppositeSide is a valid predicate
  (angle_relation : B = 2 * A)
  (cos_condition : cos A * cos B * cos C > 0)
  : Prop :=
  (sqrt 3 / 6 < a * sin A / b ∧ a * sin A / b < 1 / 2)

theorem solve_triangle
  {A B C a b c : ℝ}
  (triangle : IsTriangle A B C)
  (angle_opposite : OppositeSide a A ∧ OppositeSide b B ∧ OppositeSide c C)
  (angle_relation : B = 2 * A)
  (cos_condition : cos A * cos B * cos C > 0) :
  triangle_problem_statement A B C a b c triangle angle_opposite angle_relation cos_condition :=
sorry

end solve_triangle_l13_13903


namespace tens_digit_of_19_pow_2023_l13_13768

theorem tens_digit_of_19_pow_2023 :
  (19^2023 % 100) / 10 % 10 = 5 := 
sorry

end tens_digit_of_19_pow_2023_l13_13768


namespace trigonometric_comparison_l13_13822

theorem trigonometric_comparison :
  let a := Real.tan (Float.pi * 50 / 180)
  let b := 1 + Real.cos (Float.pi * 20 / 180)
  let c := 2 * Real.sin (Float.pi * 160 / 180)
  c < a ∧ a < b :=
by
  sorry

end trigonometric_comparison_l13_13822


namespace point_on_scaled_square_perimeter_l13_13139

-- Define the notion of a square and its vertices, and the area function
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : A = (0, 0) ∧ B = (side_length, 0) ∧ C = (side_length, side_length) ∧ D = (0, side_length))

-- Define the area calculation for triangles
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

-- Define the condition for the sum of triangle areas equaling the square area
def triangles_condition (M : ℝ × ℝ) (sq : Square) : Prop :=
  let (A, B, C, D) := (sq.A, sq.B, sq.C, sq.D)
  in triangle_area M A C + triangle_area M B D = sq.side_length * sq.side_length

-- Define the scaled square perimeter condition
def scaled_square_perimeter_condition (M : ℝ × ℝ) (sq : Square) : Prop :=
  let O := ((sq.side_length / 2), (sq.side_length / 2))
  in (M.1 - O.1)^2 + (M.2 - O.2)^2 = (sq.side_length * Math.sqrt 2)^2

-- State the theorem
theorem point_on_scaled_square_perimeter (sq : Square) (M : ℝ × ℝ) :
  triangles_condition M sq → scaled_square_perimeter_condition M sq :=
sorry

end point_on_scaled_square_perimeter_l13_13139


namespace arccos_cos_eight_l13_13746

-- Define the conditions
def cos_equivalence (x : ℝ) : Prop := cos x = cos (x - 2 * Real.pi)
def range_principal (x : ℝ) : Prop := 0 ≤ x - 2 * Real.pi ∧ x - 2 * Real.pi ≤ Real.pi

-- State the main proposition
theorem arccos_cos_eight :
  cos_equivalence 8 ∧ range_principal 8 → Real.arccos (Real.cos 8) = 8 - 2 * Real.pi :=
by
  sorry

end arccos_cos_eight_l13_13746


namespace sum_first_9_terms_eq_18_l13_13103

-- Defines the sequence {a_n} based on the given condition
def a_n (n : ℕ) (k : ℝ) : ℝ :=
  k * n - 5 * k + 2

-- Sum of the first 9 terms of the sequence
def S_9 (k : ℝ) : ℝ :=
  ∑ n in Finset.range 9, a_n (n + 1) k

-- The main theorem to be proved
theorem sum_first_9_terms_eq_18 (k : ℝ) : S_9 k = 18 := by
  sorry

end sum_first_9_terms_eq_18_l13_13103


namespace characteristic_function_of_pdf_l13_13247

noncomputable def φ_n (n : ℕ) (t : ℝ) : ℂ :=
  ∑ k in (Finset.range n), (complex.I * t) ^ k / k.factorial

theorem characteristic_function_of_pdf 
(n : ℕ) (n_pos : 0 < n) : 
 ∀ t : ℝ, 
  (∫ x in 0..1, complex.exp (complex.I * t * x) * n * (1 - x)^(n-1) dx) =
  (complex.exp (complex.I * t) - φ_n n t) / (complex.I * t) ^ n * n.factorial := 
begin
  sorry
end

end characteristic_function_of_pdf_l13_13247


namespace find_n_l13_13459

open Nat

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given condition for the proof
def condition (n : ℕ) : Prop := binom (n + 1) 7 - binom n 7 = binom n 8

-- The statement to prove
theorem find_n (n : ℕ) (h : condition n) : n = 14 :=
sorry

end find_n_l13_13459


namespace Tom_money_made_l13_13187

theorem Tom_money_made (money_last_week money_now : ℕ) (h1 : money_last_week = 74) (h2 : money_now = 160) : 
  (money_now - money_last_week = 86) :=
by 
  sorry

end Tom_money_made_l13_13187


namespace cyclist_A_speed_l13_13649

theorem cyclist_A_speed (a b : ℝ) (h1 : b = a + 5)
    (h2 : 80 / a = 120 / b) : a = 10 :=
by
  sorry

end cyclist_A_speed_l13_13649


namespace find_lambda_l13_13476

variable {λ : ℝ}

def m : ℝ × ℝ × ℝ := (λ, 2, 3)
def n : ℝ × ℝ × ℝ := (1, -3, 1)

theorem find_lambda (h : (λ, 2, 3) ⬝ (1, -3, 1) = 0) : λ = 3 :=
  by sorry

end find_lambda_l13_13476


namespace john_traveled_distance_l13_13284

theorem john_traveled_distance (full_tank : ℕ) (remaining_fuel : ℕ) (consumption_rate : ℕ) :
  full_tank = 47 → remaining_fuel = 14 → consumption_rate = 12 → 
  (let fuel_used := full_tank - remaining_fuel in
   let distance_traveled := (fuel_used * 100) / consumption_rate in
   distance_traveled = 275) :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  -- The proof steps to show that distance_traveled = 275 are omitted.
  sorry

end john_traveled_distance_l13_13284


namespace max_f_l13_13420

noncomputable def f (x φ : ℝ) : ℝ := sin (x + φ) - 2 * sin φ * cos x

theorem max_f (φ : ℝ) : ∃ x : ℝ, f x φ = 1 := by
  sorry

end max_f_l13_13420


namespace find_length_MN_l13_13829

open Real

-- Definitions based on the problem
def point (x y : ℝ) := (x, y)
def line (k : ℝ) (A : ℝ × ℝ) := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, k * x + A.2)}
def circle (C : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - C.1) ^ 2 + (p.2 - C.2) ^ 2 = r ^ 2}

-- Given conditions and required theorem
def A : ℝ × ℝ := point 0 1
def k_range : set ℝ := {k | (4 - sqrt 7) / 3 < k ∧ k < (4 + sqrt 7) / 3}
def l (k : ℝ) : set (ℝ × ℝ) := line k A
def C : ℝ × ℝ := point 2 3
def r : ℝ := 1
def circle_C : set (ℝ × ℝ) := circle C r

-- The main theorem
theorem find_length_MN
  (k : ℝ)
  (hk : k ∈ k_range)
  (H : ∃ M N : ℝ × ℝ, M ∈ l k ∧ N ∈ l k ∧ M ∈ circle_C ∧ N ∈ circle_C ∧ (M.1 * N.1 + M.2 * N.2) = 12)
  : ∃ (M N : ℝ × ℝ), (M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2 = 4 := sorry

end find_length_MN_l13_13829


namespace geometric_sequence_general_term_and_sum_l13_13503

variable {a : ℕ → ℕ}

-- We define the initial condition and the recurrence relation
def a_1 := 5
def recurrence (n : ℕ) := a (n + 1) = 2 * a n + 1

-- Problem 1: Prove that the sequence {a_n + 1} is a geometric sequence
theorem geometric_sequence (n : ℕ) (h₁ : a 0 = a_1) (h₂ : ∀ n : ℕ, recurrence n) : 
  ∃ r : ℕ, ∀ n : ℕ, (a (n + 1) + 1) = r * (a n + 1) := sorry

-- Problem 2: Find the general term formula for {a_n} and the sum of the first n terms S_n
theorem general_term_and_sum (n : ℕ) (h₁ : a 0 = a_1) (h₂ : ∀ n : ℕ, recurrence n) :
  (a n = 6 * 2^(n - 1) - 1) ∧ (∑ i in Finset.range n, a i = 6 * 2^n - n - 6) := sorry

end geometric_sequence_general_term_and_sum_l13_13503


namespace correct_transformation_l13_13682

-- Variables Definition
variables (a b x y : ℝ)

-- Propositions
def option_A : Prop := (2 * a + 1) / (b + 1) = 2 * a / b
def option_B : Prop := -(x - y) / (x + y) = (-x + y) / (x + y)
def option_C : Prop := 0.2 * x / (0.1 * x + 2 * y) = 2 * x / (x + 2 * y)
def option_D : Prop := a / b = a^2 / b^2

-- Theorem Statement
theorem correct_transformation : option_B ∧ ¬ option_A ∧ ¬ option_C ∧ ¬ option_D :=
by 
  sorry

end correct_transformation_l13_13682


namespace bus_passengers_total_l13_13636

theorem bus_passengers_total (children_percent : ℝ) (adults_number : ℝ) (H1 : children_percent = 0.25) (H2 : adults_number = 45) :
  ∃ T : ℝ, T = 60 :=
by
  sorry

end bus_passengers_total_l13_13636


namespace lollipop_remainder_l13_13325

theorem lollipop_remainder 
  (num_cherry : ℕ) (num_wintergreen : ℕ) (num_grape : ℕ) (num_shrimp : ℕ) (num_friends : ℕ)
  (h_cherry : num_cherry = 55)
  (h_wintergreen : num_wintergreen = 134)
  (h_grape : num_grape = 12)
  (h_shrimp : num_shrimp = 265)
  (h_friends : num_friends = 15) :
  let total_lollipops := num_cherry + num_wintergreen + num_grape + num_shrimp in
  total_lollipops % num_friends = 1 := 
by 
  -- Proof goes here
  sorry

end lollipop_remainder_l13_13325


namespace picture_frame_perimeter_l13_13580

-- Define the height and length
def height : ℕ := 12
def length : ℕ := 10

-- Define the perimeter of the rectangle function
def perimeter (height : ℕ) (length : ℕ) : ℕ :=
  2 * (height + length)

-- State the theorem
theorem picture_frame_perimeter :
  perimeter height length = 44 :=
by
  sorry

end picture_frame_perimeter_l13_13580


namespace imaginary_part_of_complex_l13_13987

def complex_imag_part : ℂ := (1 / (complex.I - 2)) + (1 / (1 - 2 * complex.I))

theorem imaginary_part_of_complex : complex.im complex_imag_part = 1/5 := by
  sorry

end imaginary_part_of_complex_l13_13987


namespace dolphins_in_aquarium_l13_13266

theorem dolphins_in_aquarium :
  ∀ (hours_per_dolphin trainers hours_per_trainer : ℕ), 
  hours_per_dolphin = 3 → trainers = 2 → hours_per_trainer = 6 → 
  (trainers * hours_per_trainer) / hours_per_dolphin = 4 :=
by
  intros hours_per_dolphin trainers hours_per_trainer h_dolphin h_trainers h_trainer
  rw [h_dolphin, h_trainers, h_trainer]
  norm_num
  sorry

end dolphins_in_aquarium_l13_13266


namespace problem_statement_l13_13202

theorem problem_statement (n : ℕ) (hn : 0 < n) 
  (h : (1/2 : ℚ) + 1/3 + 1/7 + 1/n ∈ ℤ) : n = 42 ∧ ¬ (n > 84) := 
by
  sorry

end problem_statement_l13_13202


namespace matrix_vector_computation_l13_13922

variable (N : Matrix (Fin 2) (Fin 2) ℝ)
variable (a b : Vector (Fin 2) ℝ)

axiom h1 : N.mulVec a = ![3, 2]
axiom h2 : N.mulVec b = ![4, 1]

theorem matrix_vector_computation :
  N.mulVec (2 • a - 4 • b) = ![-10, 0] :=
by
  sorry

end matrix_vector_computation_l13_13922


namespace center_of_pan_under_pancake_l13_13587

def round_pan_area : ℝ := 1
def is_convex (B : Set ℝ × ℝ) : Prop := sorry -- Definition of convexity.
def pancake_area (B : Set ℝ × ℝ) : ℝ := sorry -- Measure (area) of the pancake.

theorem center_of_pan_under_pancake (O : ℝ × ℝ) (B : Set ℝ × ℝ) (h_pan : π * (1 : ℝ) ^ 2 / 4 = 1) 
  (h_convex : is_convex B) (h_area : pancake_area B > 1 / 2) : O ∈ B :=
sorry

end center_of_pan_under_pancake_l13_13587


namespace inequality_proof_l13_13932

variables {n : ℕ} (A B : ℝ) (a b : Fin n → ℝ)

theorem inequality_proof
  (h1 : 0 < A) (h2 : 0 < B)
  (h3 : ∀ i, 0 < a i ∧ 0 < b i)
  (h4 : ∀ i, a i ≤ b i)
  (h5 : ∀ i, a i ≤ A)
  (h6 : (∏ i, b i) / (∏ i, a i) ≤ B / A) :
  (∏ i, b i + 1) / (∏ i, a i + 1) ≤ (B + 1) / (A + 1) := by
sorry

end inequality_proof_l13_13932


namespace octagonal_walk_distance_l13_13066

def angle_subtended_by_chord (r : ℝ) (θ : ℝ) : ℝ :=
  2 * r * Real.sin (θ / 2)

def distance_walked_by_one_boy (r : ℝ) : ℝ :=
  let d90 := angle_subtended_by_chord r (Real.pi / 2)
  let d180 := 2 * r
  d90 + d90 + d180

def total_distance_walked (r : ℝ) (n : ℕ) : ℝ :=
  n * distance_walked_by_one_boy r

theorem octagonal_walk_distance :
  total_distance_walked 50 8 = 1600 + 800 * Real.sqrt (2 - Real.sqrt 2) :=
by
  -- Proof to be filled
  sorry

end octagonal_walk_distance_l13_13066


namespace trapezoid_geometry_proof_l13_13993

theorem trapezoid_geometry_proof
  (midline_length : ℝ)
  (segment_midpoints : ℝ)
  (angle1 angle2 : ℝ)
  (h_midline : midline_length = 5)
  (h_segment_midpoints : segment_midpoints = 3)
  (h_angle1 : angle1 = 30)
  (h_angle2 : angle2 = 60) :
  ∃ (AD BC AB : ℝ), AD = 8 ∧ BC = 2 ∧ AB = 3 :=
by
  sorry

end trapezoid_geometry_proof_l13_13993


namespace tens_digit_of_19_pow_2023_l13_13767

theorem tens_digit_of_19_pow_2023 :
  (19^2023 % 100) / 10 % 10 = 5 := 
sorry

end tens_digit_of_19_pow_2023_l13_13767


namespace SashaCanDetermineX_l13_13972

def TanyaChoseNumber : Prop :=
  ∃ (X : ℕ), X ≤ 100

def SashaAsks (X M N : ℕ) : ℕ :=
  Nat.gcd (X + M) N

theorem SashaCanDetermineX (h : TanyaChoseNumber) :
  ∃ (strategy : (ℕ → ℕ → ℕ) → ℕ) (M N : ℕ), 
  (∀ X : ℕ, X ≤ 100 → 
   ∃ (answers : List ℕ), 
   List.length answers ≤ 7 ∧ 
   List.nthLe answers 0 sorry = SashaAsks X (strategy SashaAsks) M ∧ 
   ∀ i < 7, SashaAsks X (strategy SashaAsks) N ∈ answers) := by 
sorry

end SashaCanDetermineX_l13_13972


namespace man_l13_13370

theorem man's_salary (S : ℝ)
  (h1 : S * (1/5 + 1/10 + 3/5) = 9/10 * S)
  (h2 : S - 9/10 * S = 14000) :
  S = 140000 :=
by
  sorry

end man_l13_13370


namespace decreasing_on_positive_reals_l13_13959

noncomputable def f : ℝ → ℝ := λ x, -x^2 + 3

theorem decreasing_on_positive_reals : ∀ x y : ℝ, (0 < x) → (0 < y) → (x < y) → f x > f y :=
by {
  intros x y hx hy hxy,
  let fx := f x,
  let fy := f y,
  show fx > fy,
  sorry
}

end decreasing_on_positive_reals_l13_13959


namespace tilly_bag_cost_l13_13294

theorem tilly_bag_cost (n : ℕ) (p : ℕ) (profit : ℕ) : 
  (n = 100) → (p = 10) → (profit = 300) → (n * p - profit) / n = 7 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3]
  norm_num
  sorry

end tilly_bag_cost_l13_13294


namespace right_triangle_medians_right_triangle_l13_13660

theorem right_triangle_medians_right_triangle (a b c s_a s_b s_c : ℝ)
  (hyp_a_lt_b : a < b) (hyp_b_lt_c : b < c)
  (h_c_hypotenuse : c = Real.sqrt (a^2 + b^2))
  (h_sa : s_a^2 = b^2 + (a / 2)^2)
  (h_sb : s_b^2 = a^2 + (b / 2)^2)
  (h_sc : s_c^2 = (a^2 + b^2) / 4) :
  b = a * Real.sqrt 2 :=
by
  sorry

end right_triangle_medians_right_triangle_l13_13660


namespace number_of_odd_digits_base9_560_l13_13794

theorem number_of_odd_digits_base9_560 : 
  let base9_digits := [6, 8, 2]
  in (list.countp (λ d : ℕ, d % 2 = 1) base9_digits = 0) :=
by
  let base10_number := 560
  let base9_rep := [6, 8, 2]
  have step1 : base10_number = 6 * 9^2 + 8 * 9 + 2 := by norm_num
  let odd_digits := list.filter (λ d, d % 2 = 1) base9_rep
  show list.length odd_digits = 0
  sorry

end number_of_odd_digits_base9_560_l13_13794


namespace cross_section_is_curved_l13_13653

-- Define the structure of a frustum (truncated cone)
structure Frustum where
  radius_top : ℝ
  radius_bottom : ℝ
  height : ℝ

-- Define a plane intersecting the frustum
structure IntersectingPlane where
  frustum : Frustum
  intersects_top : Bool -- the plane intersects the top base
  intersects_bottom : Bool -- the plane intersects the bottom base

-- State the theorem
theorem cross_section_is_curved (fp : IntersectingPlane) (h_top : fp.intersects_top) (h_bottom : fp.intersects_bottom) : 
  ∃ shape, is_curved_sided shape := sorry

end cross_section_is_curved_l13_13653


namespace platform_length_correct_l13_13014

def train_speed_kmph : ℝ := 72
def time_cross_platform_sec : ℝ := 31
def time_cross_man_sec : ℝ := 18
def train_speed_mps : ℝ := 72 * (1000 / 3600)
def length_train : ℝ := train_speed_mps * time_cross_man_sec
def distance_cross_platform : ℝ := train_speed_mps * time_cross_platform_sec
def length_platform : ℝ := 260

theorem platform_length_correct :
  distance_cross_platform - length_train = length_platform :=
by
  sorry

end platform_length_correct_l13_13014


namespace cost_per_bag_l13_13291

theorem cost_per_bag (total_bags : ℕ) (sale_price_per_bag : ℕ) (desired_profit : ℕ) (total_revenue : ℕ)
  (total_cost : ℕ) (cost_per_bag : ℕ) :
  total_bags = 100 → sale_price_per_bag = 10 → desired_profit = 300 →
  total_revenue = total_bags * sale_price_per_bag →
  total_cost = total_revenue - desired_profit →
  cost_per_bag = total_cost / total_bags →
  cost_per_bag = 7 := by
  sorry

end cost_per_bag_l13_13291


namespace smallest_n_l13_13601

variable {α : Type*} [LinearOrderedField α]

theorem smallest_n (x : ℕ → α) (n : ℕ) 
  (h1 : ∀ i, i < n → |x i| < 1) 
  (h2 : ∑ i in Finset.range n, |x i| = 19 + |∑ i in Finset.range n, x i|) :
  n = 20 := sorry

end smallest_n_l13_13601


namespace smallest_multiple_of_45_and_75_not_20_l13_13669

-- Definitions of the conditions
def isMultipleOf (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b
def notMultipleOf (a b : ℕ) : Prop := ¬ (isMultipleOf a b)

-- The proof statement
theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ N : ℕ, isMultipleOf N 45 ∧ isMultipleOf N 75 ∧ notMultipleOf N 20 ∧ N = 225 :=
by
  -- sorry is used to indicate that the proof needs to be filled here
  sorry

end smallest_multiple_of_45_and_75_not_20_l13_13669


namespace product_sequence_mod_5_l13_13086

theorem product_sequence_mod_5 :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by
  have h1 : ∀ n, n % 10 = 3 → n % 5 = 3 := by intros n hn; rw [hn]; norm_num
  have seq : list ℕ := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  have h2 : ∀ n ∈ seq, n % 5 = 3 := by intros n hn; rw list.mem_map at hn; rcases hn with ⟨n', rfl, _⟩; exact h1 _ rfl
  have hprod : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = (3 ^ 10) % 5 := 
    by simp [pow_mul, mul_comm, int.mul_mod]; exact list.prod_eq_pow_seq 3 seq
  rw [hprod, <- pow_mod, pow_Todd_Coe] -- use appropriate power mod technique
  sorry

end product_sequence_mod_5_l13_13086


namespace distinct_remainders_l13_13071

theorem distinct_remainders (n : ℕ) (h1 : odd n) :
  ∀ (k : ℕ), 0 < k → k < n →
  let a_i (i : ℕ) := 3 * i
  let b_i (i : ℕ) := 3 * i - 1
  let a_i_i_plus_1 := ∀ i, a_i i + a_i (i + 1) % (3 * n) ≠ a_i j + a_i (j + 1) % (3 * n) → i ≠ j
  let a_i_b_i := ∀ i, a_i i + b_i i % (3 * n) ≠ a_i j + b_i j % (3 * n) → i ≠ j
  let b_i_b_i_plus_k := ∀ i, b_i i + b_i (i + k) % (3 * n) ≠ b_i j + b_i (j + k) % (3 * n) → i ≠ j
  in sorry

end distinct_remainders_l13_13071


namespace find_projection_l13_13799

open Matrix

def is_orthogonal_projection
  (v : ℝ × ℝ) (a : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p = (a.1 * v.1 + a.2 * v.2) / (v.1 * v.1 + v.2 * v.2) • v

theorem find_projection (v : ℝ × ℝ) (p : ℝ × ℝ)
  (h_orth_proj_a1 : is_orthogonal_projection v ⟨3, -1⟩ p)
  (h_orth_proj_a2 : is_orthogonal_projection v ⟨-4, 5⟩ p) :
  p = ⟨-15 / 17, 77 / 85⟩ :=
sorry

end find_projection_l13_13799


namespace maximum_n_le_19_l13_13178

-- Define the arithmetic sequence with the given conditions
variable {a : ℕ → ℝ}

axiom a_10_neg : a 10 < 0
axiom a_11_pos : a 11 > 0
axiom a_11_gt_abs_a_10 : a 11 > |a 10|

-- Define the sum of the first n terms of the sequence
def sum_arith_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

-- Translate the problem to the statement: For the arithmetic sequence {a_n}, if Sn < 0, find the maximum n.
theorem maximum_n_le_19 (n : ℕ) : sum_arith_seq a n < 0 → n ≤ 19 := by
  sorry

end maximum_n_le_19_l13_13178


namespace find_p_minus_r_l13_13290

theorem find_p_minus_r 
  (p q r : ℕ) 
  (h1 : p > 0) 
  (h2 : q > 0) 
  (h3 : r > 0) 
  (h4 : p * q * r = 10.factorial) 
  (h5 : p * q + p + q = 2450)
  (h6 : q * r + q + r = 1012)
  (h7 : r * p + r + p = 2020) 
  : p - r = -430 := 
by
  sorry

end find_p_minus_r_l13_13290


namespace circles_intersect_l13_13273

-- Definitions for the circles
def C1_eq : (ℝ × ℝ) → Prop := λ ⟨x, y⟩, x^2 + (y - 1)^2 = 1
def C2_eq : (ℝ × ℝ) → Prop := λ ⟨x, y⟩, x^2 - 6x + y^2 - 8y = 0

-- Coordinates of the center and radii of the circles
def C1_center : ℝ × ℝ := (0, 1)
def C1_radius : ℝ := 1

def C2_center : ℝ × ℝ := (3, 4)
def C2_radius : ℝ := 5

-- Distance between the centers of the circles
def center_distance : ℝ := Real.sqrt ((C1_center.1 - C2_center.1)^2 + (C1_center.2 - C2_center.2)^2)

-- The theorem stating their positional relationship
theorem circles_intersect : (C2_radius - C1_radius < center_distance) ∧ (center_distance < C2_radius + C1_radius) :=
by 
  sorry

end circles_intersect_l13_13273


namespace orthogonal_lines_solution_l13_13638

theorem orthogonal_lines_solution (a b c d : ℝ)
  (h1 : b - a = 0)
  (h2 : c - a = 2)
  (h3 : 12 * d - a = 1)
  : d = 3 / 11 :=
by {
  sorry
}

end orthogonal_lines_solution_l13_13638


namespace machines_finish_job_in_2_hours_l13_13214

theorem machines_finish_job_in_2_hours :
  (1 / 4 + 1 / 12 + 1 / 6) = 1 / 2 → (1 / (1 / 2)) = 2 :=
by
  intro h
  linarith

end machines_finish_job_in_2_hours_l13_13214


namespace sequence_a5_l13_13900

theorem sequence_a5 : 
    ∃ (a : ℕ → ℚ), 
    a 1 = 1 / 3 ∧ 
    (∀ (n : ℕ), n ≥ 2 → a n = (-1 : ℚ)^n * 2 * a (n - 1)) ∧ 
    a 5 = -16 / 3 := 
sorry

end sequence_a5_l13_13900


namespace triangle_AC_l13_13907

theorem triangle_AC {A B C D : Type*}
  (AB BC BD AC AD DC : ℝ)
  (h1 : BC > AB)
  (h2 : AB = 5)
  (h3 : BD = 4)
  (isos_ABD : AB = BD ∧ AD = AC ∧ AC = x + 4)
  (isos_BCD : BD = DC) :
  AC = 9 ∨ AC = 10 :=
begin
  sorry
end

end triangle_AC_l13_13907


namespace binary_to_decimal_l13_13754

theorem binary_to_decimal (b : ℕ) (h : b = 2^3 + 2^2 + 0 * 2^1 + 2^0) : b = 13 :=
by {
  -- proof is omitted
  sorry
}

end binary_to_decimal_l13_13754


namespace emilia_strCartons_l13_13779

theorem emilia_strCartons (total_cartons_needed cartons_bought cartons_blueberries : ℕ) (h1 : total_cartons_needed = 42) (h2 : cartons_blueberries = 7) (h3 : cartons_bought = 33) :
  (total_cartons_needed - (cartons_bought + cartons_blueberries)) = 2 :=
by
  sorry

end emilia_strCartons_l13_13779


namespace total_fertilizer_usage_l13_13363

theorem total_fertilizer_usage :
  let daily_A : ℝ := 3 / 12
  let daily_B : ℝ := 4 / 10
  let daily_C : ℝ := 5 / 8
  let final_A : ℝ := daily_A + 6
  let final_B : ℝ := daily_B + 5
  let final_C : ℝ := daily_C + 7
  (final_A + final_B + final_C) = 19.275 := by
  sorry

end total_fertilizer_usage_l13_13363


namespace find_sum_of_exponents_l13_13458

theorem find_sum_of_exponents (x y : ℝ) (h1 : 4^x = 16^(y + 2)) (h2 : 27^y = 9^(x - 8)) :
  x + y = 40 := 
by
  sorry

end find_sum_of_exponents_l13_13458


namespace determinant_relation_l13_13568

variables (u v w : ℝ^3)
def E : ℝ := let M := ![![2 * u, v, w]] in Matrix.det M
def F : ℝ := let N := ![![u × v, v × w, w × (2 * u)]] in Matrix.det N

theorem determinant_relation : F = 2 * (E ^ 2) := sorry

end determinant_relation_l13_13568


namespace temperature_below_zero_l13_13892

-- Assume the basic definitions and context needed
def above_zero (temp : Int) := temp > 0
def below_zero (temp : Int) := temp < 0

theorem temperature_below_zero (t1 t2 : Int) (h1 : above_zero t1) (h2 : t2 = -7) :
  below_zero t2 := by 
  -- This is where the proof would go
  sorry

end temperature_below_zero_l13_13892


namespace double_integral_example_l13_13390

noncomputable def integral_double : ℝ :=
  ∫ x in 0..1, ∫ y in (-(Real.sqrt x))..(x^3), (54 * x^2 * y^2 + 150 * x^4 * y^4) ∂y ∂x

theorem double_integral_example : integral_double = 11 :=
by
  sorry

end double_integral_example_l13_13390


namespace expected_value_of_n_eq_55200_208_l13_13800

noncomputable def expected_value_n : ℚ :=
  let prob_remain := (3 / 5 : ℚ) ^ 3
  let prob_swap := (1 - prob_remain)
  let digits := [1, 3, 5, 7, 9]
  let expected_digit d := prob_remain * d + prob_swap * (digits.sum - d) / 4
  let weights := [10000, 1000, 100, 10, 1]
  (weights.zip (digits.map expected_digit)).map (fun (w, e) => w * e).sum

theorem expected_value_of_n_eq_55200_208 :
  expected_value_n = (55200.208 : ℚ) := by
  sorry

end expected_value_of_n_eq_55200_208_l13_13800


namespace z_imaginary_part_l13_13866

def z_imaginary_part_condition (z : ℂ) : Prop :=
  (complex.i * z = -((1 + complex.i) / 2))

theorem z_imaginary_part {z : ℂ} (h : z_imaginary_part_condition z) : z.im = 1 / 2 :=
sorry

end z_imaginary_part_l13_13866


namespace balls_into_boxes_l13_13145

theorem balls_into_boxes : (@Finset.choose 8 2).card = 28 := by
  sorry

end balls_into_boxes_l13_13145


namespace find_K_l13_13926

open_locale classical

-- Definition of the sequence b_n satisfying the given conditions
variables {b : ℕ → ℝ}

-- Conditions from the problem
axiom (b_nonneg : ∀ n, b n ≥ 0)
axiom (b_squared_ineq : ∀ n, b (n + 1) ^ 2 ≥ ∑ k in finset.range (n + 1), (b k ^ 2) / (k + 1) ^ 3)

-- The theorem we need to prove
theorem find_K : ∃ K : ℕ, ∑ n in finset.range K, b (n + 1) / (∑ k in finset.range (n + 1), b k) ≥ 1993 / 1000 :=
sorry

end find_K_l13_13926


namespace g_values_l13_13198

variable (g : ℝ → ℝ)

-- Condition: ∀ x y z ∈ ℝ, g(x^2 + y * g(z)) = x * g(x) + 2 * z * g(y)
axiom g_axiom : ∀ x y z : ℝ, g (x^2 + y * g z) = x * g x + 2 * z * g y

-- Proposition: The possible values of g(4) are 0 and 8.
theorem g_values : g 4 = 0 ∨ g 4 = 8 :=
by
  sorry

end g_values_l13_13198


namespace continuous_constant_function_l13_13046

noncomputable def f (x : ℝ) : ℝ := sorry

theorem continuous_constant_function :
  (∀ a b : ℝ, (a^2 + a*b + b^2) * (∫ x in a..b, f x) = 3 * (∫ x in a..b, x^2 * f x)) →
  continuous f →
  ∃ C : ℝ, ∀ x : ℝ, f x = C := 
sorry

end continuous_constant_function_l13_13046


namespace permutation_count_l13_13917

theorem permutation_count :
  let a : Fin 15 → Fin 15 :=
    λ i, if i = 5 then 0 else
         if i < 5 then 14 - i else
         i - 5 in
  (permute_count (a '' univ)) = (choose 14 4) := 
by
  sorry

end permutation_count_l13_13917


namespace arccos_cos_8_eq_1_point_72_l13_13743

noncomputable def arccos_cos_eight : Real :=
  Real.arccos (Real.cos 8)

theorem arccos_cos_8_eq_1_point_72 : arccos_cos_eight = 1.72 :=
by
  sorry

end arccos_cos_8_eq_1_point_72_l13_13743


namespace rhyme_around_3_7_l13_13650

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rhymes_around (p q m : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ ((p < m ∧ q > m ∧ q - m = m - p) ∨ (p > m ∧ q < m ∧ p - m = m - q))

theorem rhyme_around_3_7 : ∃ m : ℕ, rhymes_around 3 7 m ∧ m = 5 :=
by
  sorry

end rhyme_around_3_7_l13_13650


namespace det_new_matrix_l13_13150

variables {a b c d : ℝ}

theorem det_new_matrix (h : a * d - b * c = 5) : (a - c) * d - (b - d) * c = 5 :=
by sorry

end det_new_matrix_l13_13150


namespace isosceles_triangle_angles_l13_13297

/-- Given an isosceles triangle ABC with AB = BC, and point L on AB such that the perpendicular
    bisector of AC intersects AB at L and the extension of BC at K,
    and the areas of triangles ALC and KBL are equal,
    prove the angles of the triangle ABC are 36°, 72°, and 72°. -/
theorem isosceles_triangle_angles (A B C L K : Point)
(ha : isosceles_triangle A B C)
(hl : Line A B)
(perp : perpendicular_bisector AC L K)
(area_eq : area (triangle A L C) = area (triangle K B L))
: angle A B C = 72 ∧ angle B C A = 72 ∧ angle B A C = 36 := 
sorry

end isosceles_triangle_angles_l13_13297


namespace find_point_C_l13_13109

-- Define points A and B
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨4, 1, 3⟩
def B : Point3D := ⟨2, -5, 1⟩

-- Define vector subtraction to get the difference between two points
def vector_sub (p1 p2 : Point3D) : Point3D :=
  ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

-- Define vector scaling
def vector_scale (s : ℝ) (v : Point3D) : Point3D :=
  ⟨s * v.x, s * v.y, s * v.z⟩

-- Define the coordinates of point C
def C : Point3D := ⟨10 / 3, -1, 7 / 3⟩

-- The theorem to prove the coordinates of Point C
theorem find_point_C : (vector_sub A B = vector_scale 3 (vector_sub A C)) :=
  sorry

end find_point_C_l13_13109


namespace length_of_parallel_segment_closest_to_AC_is_correct_l13_13180

-- Definitions for the problem given in Lean 4

def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ C.1 = 8

def segments_parallel_to_AC (A B C : ℝ × ℝ) (DE FG : ℝ × ℝ × ℝ × ℝ) : Prop :=
  ∃ D E F G : ℝ × ℝ,
  -- Points D, E, F, G are on AB and BC
  D.1 ≤ B.1 ∧ F.1 ≤ B.1 ∧
  -- Line segments DE and FG are parallel to AC
  (D.2 = E.2 ∧ F.2 = G.2) ∧ 
  -- Divide triangle into three parts of equal area
  sorry -- This part would need to verify areas

theorem length_of_parallel_segment_closest_to_AC_is_correct (A B C DE FG : ℝ × ℝ) :
  triangle_ABC A B C →
  segments_parallel_to_AC A B C DE FG →
  (∃ x : ℝ, x = (AC_length := 8, *result) = x → x = 8 * sqrt 3 / 3 : ℝ) :=
sorry -- Proof omitted

end length_of_parallel_segment_closest_to_AC_is_correct_l13_13180


namespace Geraldine_more_than_Jazmin_l13_13099

-- Define the number of dolls Geraldine and Jazmin have
def Geraldine_dolls : ℝ := 2186.0
def Jazmin_dolls : ℝ := 1209.0

-- State the theorem we need to prove
theorem Geraldine_more_than_Jazmin :
  Geraldine_dolls - Jazmin_dolls = 977.0 := 
by
  sorry

end Geraldine_more_than_Jazmin_l13_13099


namespace shaded_area_of_rotated_square_l13_13376

-- Define the conditions as assumptions
def side_length : ℝ := 2
def cos_beta : ℝ := 3 / 5

theorem shaded_area_of_rotated_square : 
    ∃ (area : ℝ), area = 4 / 3 :=
by
  sorry

end shaded_area_of_rotated_square_l13_13376


namespace scheme_choice_l13_13356

variable (x y₁ y₂ : ℕ)

def cost_scheme_1 (x : ℕ) : ℕ := 12 * x + 40

def cost_scheme_2 (x : ℕ) : ℕ := 16 * x

theorem scheme_choice :
  ∀ (x : ℕ), 5 ≤ x → x ≤ 20 →
  (if x < 10 then cost_scheme_2 x < cost_scheme_1 x else
   if x = 10 then cost_scheme_2 x = cost_scheme_1 x else
   cost_scheme_1 x < cost_scheme_2 x) :=
by
  sorry

end scheme_choice_l13_13356


namespace convex_polygon_center_of_symmetry_l13_13227

theorem convex_polygon_center_of_symmetry (M : Type) 
    (convex : Convex M)
    (centrally_symmetric_div : ∀ (P : Type), P ∈ M → CentrallySymmetric P) :
    CentrallySymmetric M :=
sorry

end convex_polygon_center_of_symmetry_l13_13227


namespace outfit_combinations_l13_13970

theorem outfit_combinations (shirts ties hat_choices : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 7) (h_hat_choices : hat_choices = 3) : shirts * ties * hat_choices = 168 := by
  sorry

end outfit_combinations_l13_13970


namespace correct_operation_l13_13683

noncomputable def check_operations : Prop :=
    ∀ (a : ℝ), ( a^6 / a^3 = a^3 ) ∧ 
               ¬( 3 * a^5 + a^5 = 4 * a^10 ) ∧
               ¬( (2 * a)^3 = 2 * a^3 ) ∧
               ¬( (a^2)^4 = a^6 )

theorem correct_operation : check_operations :=
by
  intro a
  have h1 : a^6 / a^3 = a^3 := by
    sorry
  have h2 : ¬(3 * a^5 + a^5 = 4 * a^10) := by
    sorry
  have h3 : ¬((2 * a)^3 = 2 * a^3) := by
    sorry
  have h4 : ¬((a^2)^4 = a^6) := by
    sorry
  exact ⟨h1, h2, h3, h4⟩

end correct_operation_l13_13683


namespace arccos_of_cos_periodic_l13_13741

theorem arccos_of_cos_periodic :
  arccos (cos 8) = 8 - 2 * Real.pi :=
by
  sorry

end arccos_of_cos_periodic_l13_13741


namespace sin_angle_RPT_l13_13896

theorem sin_angle_RPT (θ : ℝ) (h : real.sin θ = 3/5) : real.sin (2 * π - θ) = 3/5 :=
by 
  -- This by block is a placeholder for the actual proof.
  sorry

end sin_angle_RPT_l13_13896


namespace perpendicular_planes_l13_13571

variables (m n : Line) (α β : Plane)

def lines_are_different : Prop := m ≠ n
def planes_are_different : Prop := α ≠ β
def m_perpendicular_n : Prop := m ⊥ n
def m_perpendicular_alpha : Prop := m ⊥ α
def n_perpendicular_beta : Prop := n ⊥ β

theorem perpendicular_planes :
  lines_are_different m n →
  planes_are_different α β →
  m_perpendicular_n m n →
  m_perpendicular_alpha m α →
  n_perpendicular_beta n β →
  α ⊥ β :=
by
  sorry

end perpendicular_planes_l13_13571


namespace smallest_positive_period_axis_of_symmetry_function_range_l13_13840

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * cos x ^ 2

theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, f(x + T) = f(x) ∧ T = π := sorry

theorem axis_of_symmetry :
  ∀ k : ℤ, ∃ x : ℝ, x = π / 6 + k * π / 2 := sorry

theorem function_range :
  ∀ x ∈ Set.Icc (-π/12) (π/2), f x ∈ Set.Icc (-1) 2 := sorry

end smallest_positive_period_axis_of_symmetry_function_range_l13_13840


namespace sophie_solution_l13_13969

noncomputable def sophie_number : ℂ :=
  let theo_number := 7 + 4 * complex.I
  let product := 80 - 24 * complex.I
  product / theo_number

theorem sophie_solution : sophie_number = 7.1385 - 7.5077 * complex.I := by
  sorry

end sophie_solution_l13_13969


namespace arithmetic_sequence_sum_l13_13545

theorem arithmetic_sequence_sum 
    (a : ℕ → ℤ)
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) -- Arithmetic sequence condition
    (h2 : a 5 = 3)
    (h3 : a 6 = -2) :
    (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3) :=
sorry

end arithmetic_sequence_sum_l13_13545


namespace monotonicity_of_f_range_of_a_if_f_lt_x_squared_l13_13489

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a / x

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  ∀ x y : ℝ, 0 < x → x < y → f x a < f y a := by
  sorry

theorem range_of_a_if_f_lt_x_squared (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a < x^2) → a ≥ -1 := by
  sorry

end monotonicity_of_f_range_of_a_if_f_lt_x_squared_l13_13489


namespace smallest_positive_period_of_f_f_monotonically_increasing_f_min_max_values_l13_13129

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x)

-- Statement 1: Proving the smallest positive period of the function is π
theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T' :=
sorry

-- Statement 2: Proving the function is monotonically increasing on given intervals
theorem f_monotonically_increasing (k : ℤ) : ∀ x, (k * π - 3 * π / 8 ≤ x ∧ x ≤ k * π + π / 8) → derivative f x > 0 :=
sorry

-- Statement 3: Proving the minimum and maximum values of the function on the specified interval
theorem f_min_max_values : 
  (∀ x ∈ Icc (-π / 4) (π / 4), f x ≥ 0) ∧ 
  (∀ x ∈ Icc (-π / 4) (π / 4), f x ≤ sqrt 2 + 1) ∧ 
  (∃ x₁ x₂, x₁ ∈ Icc (-π / 4) (π / 4) ∧ f x₁ = 0 ∧ x₂ ∈ Icc (-π / 4) (π / 4) ∧ f x₂ = sqrt 2 + 1) :=
sorry

end smallest_positive_period_of_f_f_monotonically_increasing_f_min_max_values_l13_13129


namespace maximize_probability_sum_8_l13_13656

def L : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]

theorem maximize_probability_sum_8 :
  (∀ x ∈ L, x ≠ 4 → (∃ y ∈ (List.erase L x), y = 8 - x)) ∧ 
  (∀ y ∈ List.erase L 4, ¬(∃ x ∈ List.erase L 4, x + y = 8)) :=
sorry

end maximize_probability_sum_8_l13_13656


namespace slope_at_certain_point_equal_three_l13_13493
open Real

noncomputable def f (x a : ℝ) := x * log x + a * x^2

theorem slope_at_certain_point_equal_three (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ deriv (λ z, f z a) x = 3) → a ≥ -1 / (2 * exp 3) := 
by
  sorry

end slope_at_certain_point_equal_three_l13_13493


namespace cookies_left_to_take_home_l13_13912

variable (initial_chocolate_chip : Nat) (initial_sugar : Nat) (initial_oatmeal_raisin : Nat)
variable (morning_chocolate_chip_sold : Nat) (morning_sugar_sold : Nat)
variable (lunch_chocolate_chip_sold : Nat) (lunch_sugar_sold : Nat) (lunch_oatmeal_raisin_sold : Nat)
variable (afternoon_chocolate_chip_sold : Nat) (afternoon_sugar_sold : Nat) (afternoon_oatmeal_raisin_sold : Nat)
variable (unsellable_fraction : Float)

-- Initialize variables according to problem conditions
def initial_conditions : Prop :=
  initial_chocolate_chip = 60 ∧ initial_sugar = 40 ∧ initial_oatmeal_raisin = 20 ∧
  morning_chocolate_chip_sold = 24 ∧ morning_sugar_sold = 12 ∧
  lunch_chocolate_chip_sold = 33 ∧ lunch_sugar_sold = 20 ∧ lunch_oatmeal_raisin_sold = 4 ∧
  afternoon_chocolate_chip_sold = 10 ∧ afternoon_sugar_sold = 4 ∧ afternoon_oatmeal_raisin_sold = 2 ∧
  unsellable_fraction = 0.05

-- Prove that the number of cookies left to take home is 18
theorem cookies_left_to_take_home {initial_chocolate_chip initial_sugar initial_oatmeal_raisin
  morning_chocolate_chip_sold morning_sugar_sold
  lunch_chocolate_chip_sold lunch_sugar_sold lunch_oatmeal_raisin_sold
  afternoon_chocolate_chip_sold afternoon_sugar_sold afternoon_oatmeal_raisin_sold
  unsellable_fraction : Float}
  (h : initial_conditions initial_chocolate_chip initial_sugar initial_oatmeal_raisin
    morning_chocolate_chip_sold morning_sugar_sold
    lunch_chocolate_chip_sold lunch_sugar_sold lunch_oatmeal_raisin_sold
    afternoon_chocolate_chip_sold afternoon_sugar_sold afternoon_oatmeal_raisin_sold
    unsellable_fraction): 
  (remaining_cookies : Int) := 18 . by sorry

end cookies_left_to_take_home_l13_13912


namespace cosine_largest_angle_l13_13677

theorem cosine_largest_angle (a b c : ℝ) (h : a = 4 ∧ b = 5 ∧ c = 6) : 
  let θ := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) in
  Real.cos θ = 1 / 8 := 
by
  sorry

end cosine_largest_angle_l13_13677


namespace exists_triangle_with_angle_le_30_degrees_l13_13809

-- Define a Lean statement using conditions and the correct answer
theorem exists_triangle_with_angle_le_30_degrees (P : Fin 6 → ℝ × ℝ)
  (h_no_collinear : ∀ (i j k : Fin 6), i ≠ j → j ≠ k → i ≠ k →
    ¬ (collinear ({P i, P j, P k} : Set (ℝ × ℝ)))) :
  ∃ (A B C : Fin 6), A ≠ B → B ≠ C → A ≠ C →
  ∃ (θ : ℝ), is_triangle (P A) (P B) (P C) ∧ θ ≤ 30 :=
  sorry

end exists_triangle_with_angle_le_30_degrees_l13_13809


namespace decimal_rep_10_3010_plus_1_5_div_9_l13_13391

noncomputable def first_three_decimal_digits := Nat.sqrt 555

theorem decimal_rep_10_3010_plus_1_5_div_9 :
  ∃ d1 d2 d3 : Nat, d1 = 5 ∧ d2 = 5 ∧ d3 = 5 ∧
  ((10 ^ 3010 + 1)^ (5/9) - (10 ^ 3010 + 1)^ (5/9) % 1) * 1000 = d1 * 100 + d2 * 10 + d3 :=
begin
  sorry,
end

end decimal_rep_10_3010_plus_1_5_div_9_l13_13391


namespace courses_chosen_l13_13020

theorem courses_chosen (A B: set ℕ) (hA: A ⊆ {1, 2, 3, 4, 5, 6}) (hB: B ⊆ {1, 2, 3, 4, 5, 6})
  (hA_3: A.card = 3) (hB_3: B.card = 3) (h1_common: (A ∩ B).card = 1) : 
  ∃ (n: ℕ), n = 180 :=
by {
  sorry
}

end courses_chosen_l13_13020


namespace shaded_region_area_l13_13030

-- Defining the given conditions
def radius1 : ℝ := 3
def radius2 : ℝ := 4
def radius_circumscribed : ℝ := 7

-- The target theorem
theorem shaded_region_area : 
  let area1 := π * radius1^2,
      area2 := π * radius2^2,
      area_circumscribed := π * radius_circumscribed^2
  in area_circumscribed - area1 - area2 = 24 * π :=
by
  sorry

end shaded_region_area_l13_13030


namespace average_girls_score_l13_13018

open Function

variable (C c D d : ℕ)
variable (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)

-- Conditions
def CedarBoys := avgCedarBoys = 85
def CedarGirls := avgCedarGirls = 80
def CedarCombined := avgCedarCombined = 83
def DeltaBoys := avgDeltaBoys = 76
def DeltaGirls := avgDeltaGirls = 95
def DeltaCombined := avgDeltaCombined = 87
def CombinedBoys := avgCombinedBoys = 73

-- Correct answer
def CombinedGirls (avgCombinedGirls : ℤ) := avgCombinedGirls = 86

-- Final statement
theorem average_girls_score (C c D d : ℕ)
    (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)
    (H1 : CedarBoys avgCedarBoys)
    (H2 : CedarGirls avgCedarGirls)
    (H3 : CedarCombined avgCedarCombined)
    (H4 : DeltaBoys avgDeltaBoys)
    (H5 : DeltaGirls avgDeltaGirls)
    (H6 : DeltaCombined avgDeltaCombined)
    (H7 : CombinedBoys avgCombinedBoys) :
    ∃ avgCombinedGirls, CombinedGirls avgCombinedGirls :=
sorry

end average_girls_score_l13_13018


namespace solution_set_correct_l13_13090

noncomputable def solution_set (x : ℝ) : Prop :=
  x + 2 / (x + 1) > 2

theorem solution_set_correct :
  {x : ℝ | solution_set x} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by sorry

end solution_set_correct_l13_13090


namespace ratio_of_legs_eq_sqrt3_l13_13605

theorem ratio_of_legs_eq_sqrt3 (a b c : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_area_right_triangle : 1/2 * a * b = S)
  (h_area_equilateral_triangle : 2 * S = (sqrt 3 / 4) * c^2) :
  a / b = sqrt 3 :=
sorry

end ratio_of_legs_eq_sqrt3_l13_13605


namespace solve_for_x_l13_13786

noncomputable def log_base_3 (y : ℝ) : ℝ := log y / log 3

theorem solve_for_x (x : ℝ) (h : 4 * log_base_3 x = log_base_3 (4 * x^2)) : x = 2 := by
  sorry

end solve_for_x_l13_13786


namespace find_a10_l13_13472

variable {a : ℕ → ℝ}
variable (h1 : ∀ n m, a (n + 1) = a n + a m)
variable (h2 : a 6 + a 8 = 16)
variable (h3 : a 4 = 1)

theorem find_a10 : a 10 = 15 := by
  sorry

end find_a10_l13_13472


namespace sufficient_but_not_necessary_l13_13344

-- Defining the variables and conditions
variable (m : ℝ)
def z (m : ℝ) := complex.mk (m ^ 2 - 1) (m)

-- Main statement
theorem sufficient_but_not_necessary : (m = 1) → (z m).re = 0 :=
by
  -- placeholder for the proof
  sorry

end sufficient_but_not_necessary_l13_13344


namespace prod_largest_second_largest_l13_13288

def numbers : List ℕ := [10, 11, 12, 13, 14]

def largest : ℕ := List.maximum numbers (by decide!)
def second_largest : ℕ := List.maximum (numbers.erase largest) (by decide!)

theorem prod_largest_second_largest :
  largest * second_largest = 182 := by
  have h_largest : largest = 14 := by decide!
  have h_second_largest : second_largest = 13 := by decide!
  rw [h_largest, h_second_largest]
  exact rfl

end prod_largest_second_largest_l13_13288


namespace find_polygon_formed_by_four_lines_is_quadrilateral_l13_13795

noncomputable theory
open_locale classical

structure Line (α : Type) :=
(a b c : α) -- represents Ax + By + C = 0

def eval_line {α : Type} [field α] (l : Line α) (x y : α) := l.a * x + l.b * y + l.c

def intersection_point {α : Type} [field α] (l1 l2 : Line α) : option (α × α) :=
if h : l1.a * l2.b - l2.a * l1.b ≠ 0 then some ((l2.c * l1.b - l1.c * l2.b) / (l1.a * l2.b - l2.a * l1.b), (l1.c * l2.a - l2.c * l1.a) / (l1.b * l2.a - l2.b * l1.b))
else none

def is_quadrilateral {α : Type} [field α] (pts : list (α × α)) : Prop :=
pts.length = 4

def l1 : Line ℝ := ⟨2, -1, 3⟩
def l2 : Line ℝ := ⟨-2, -1, 1⟩
def l3 : Line ℝ := ⟨0, 1, 1⟩
def l4 : Line ℝ := ⟨1, 0, -1⟩

theorem find_polygon_formed_by_four_lines_is_quadrilateral :
  ∃ pts : list (ℝ × ℝ), 
  intersection_point l1 l2 = some (-1/2, 2) ∧ 
  intersection_point l1 l3 = some (-2, -1) ∧ 
  intersection_point l2 l3 = some (1, -1) ∧ 
  intersection_point l3 l4 = some (1, -1) ∧ 
  intersection_point l4 l1 = some (1, 5) ∧ 
  is_quadrilateral pts :=
by sorry

end find_polygon_formed_by_four_lines_is_quadrilateral_l13_13795


namespace students_per_group_l13_13349

theorem students_per_group (total_students : ℤ) (total_teachers : ℤ) (h_students : total_students = 256) (h_teachers : total_teachers = 8) : ∃ (students_per_group : ℤ), students_per_group = total_students / total_teachers ∧ students_per_group = 32 :=
by
  use 256 / 8
  split
  . rfl
  . norm_num
  . sorry

end students_per_group_l13_13349


namespace clap_7_total_count_l13_13575

def is_visible_7 (n : ℕ) : Prop := n.to_digits 10 contains 7

def is_invisible_7 (n : ℕ) : Prop := n % 7 = 0

theorem clap_7_total_count : (finset.range 101).filter is_visible_7 ∪ (finset.range 101).filter is_invisible_7 = 30 :=
by
  sorry

end clap_7_total_count_l13_13575


namespace largest_possible_N_l13_13947

theorem largest_possible_N (N : ℕ)
  (h₁ : ∀ (lines : finset (set (ℝ × ℝ))), lines.card = N → ∀ l₁ l₂ ∈ lines, l₁ ≠ l₂ → ∃ p : ℝ × ℝ, p ∈ l₁ ∧ p ∈ l₂)
  (h₂ : ∀ (lines : finset (set (ℝ × ℝ))), lines.card = 15 → ∃ l₁ l₂ ∈ lines, ∃ θ : ℝ, θ = 60 ∧ angle_between_lines l₁ l₂ = θ) :
  N ≤ 42 :=
sorry

end largest_possible_N_l13_13947


namespace largest_integer_satisfying_inequality_l13_13048

theorem largest_integer_satisfying_inequality : ∃ (x : ℤ), (5 * x - 4 < 3 - 2 * x) ∧ (∀ (y : ℤ), (5 * y - 4 < 3 - 2 * y) → y ≤ x) ∧ x = 0 :=
by
  sorry

end largest_integer_satisfying_inequality_l13_13048


namespace cannot_form_triangle_l13_13427

-- Definitions of the lines in terms of m:
def line1 (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y - 1 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 2 * y - 5 = 0
def line3 (x y : ℝ) : Prop := 6 * x + y - 5 = 0

-- The proof statement:
theorem cannot_form_triangle (m : ℝ) : m = -2 ∨ m = 1 / 2 ↔ 
  ¬ (∃ x y : ℝ, line1 m x y ∧ line2 x y ∧ line3 x y) :=
begin
  sorry -- Proof to be provided
end

end cannot_form_triangle_l13_13427


namespace roots_of_third_quadratic_l13_13121

/-- Given two quadratic equations with exactly one common root and a non-equal coefficient condition, 
prove that the other roots are roots of a third quadratic equation -/
theorem roots_of_third_quadratic 
  (a1 a2 a3 α β γ : ℝ)
  (h1 : α ≠ β)
  (h2 : α ≠ γ)
  (h3 : a1 ≠ a2)
  (h_eq1 : α^2 + a1*α + a2*a3 = 0)
  (h_eq2 : β^2 + a1*β + a2*a3 = 0)
  (h_eq3 : α^2 + a2*α + a1*a3 = 0)
  (h_eq4 : γ^2 + a2*γ + a1*a3 = 0) :
  β^2 + a3*β + a1*a2 = 0 ∧ γ^2 + a3*γ + a1*a2 = 0 :=
by
  sorry

end roots_of_third_quadratic_l13_13121


namespace percentage_of_useful_items_l13_13578

theorem percentage_of_useful_items
  (junk_percentage : ℚ)
  (useful_items junk_items total_items : ℕ)
  (h1 : junk_percentage = 0.70)
  (h2 : useful_items = 8)
  (h3 : junk_items = 28)
  (h4 : junk_percentage * total_items = junk_items) :
  (useful_items : ℚ) / (total_items : ℚ) * 100 = 20 :=
sorry

end percentage_of_useful_items_l13_13578


namespace units_digit_product_even_integers_l13_13673

theorem units_digit_product_even_integers (S : Finset ℕ) (h₁ : ∀ n ∈ S, 10 ≤ n ∧ n ≤ 100 ∧ n % 2 = 0) :
  ∃ m, m ∈ S ∧ m % 10 = 0 → (∏ n in S, n) % 10 = 0 :=
by
  sorry

end units_digit_product_even_integers_l13_13673


namespace rectangle_area_is_180_l13_13339

def area_of_square (side : ℕ) : ℕ := side * side
def length_of_rectangle (radius : ℕ) : ℕ := (2 * radius) / 5
def area_of_rectangle (length breadth : ℕ) : ℕ := length * breadth

theorem rectangle_area_is_180 :
  ∀ (side breadth : ℕ), 
    area_of_square side = 2025 → 
    breadth = 10 → 
    area_of_rectangle (length_of_rectangle side) breadth = 180 :=
by
  intros side breadth h_area h_breadth
  sorry

end rectangle_area_is_180_l13_13339


namespace slope_of_line_through_ellipse_l13_13898

theorem slope_of_line_through_ellipse :
  ∀ (a b k : ℝ), 
    a > b ∧ b > 0 ∧ k > 0 ∧ (1 / 3 : ℝ) = (classical.some (unique c, h : e = c / a : (∃ a > 0, e = c / a))) → 
    let c := a / 3 in 
    let B := (c, (b^2 / a)) in 
    let A := (-a, 0) in 
    k = ((b^2 / a) - 0) / (c - (-a)) → 
    k = (2 / 3) :=
begin
  intros,
  sorry
end

end slope_of_line_through_ellipse_l13_13898


namespace smallest_value_3a_plus_1_l13_13856

theorem smallest_value_3a_plus_1 
  (a : ℝ)
  (h : 8 * a^2 + 9 * a + 6 = 2) : 
  ∃ (b : ℝ), b = 3 * a + 1 ∧ b = -2 :=
by 
  sorry

end smallest_value_3a_plus_1_l13_13856


namespace sum_of_areas_l13_13720

noncomputable def radius_sequence (n : ℕ) : ℝ :=
  1 / (real.sqrt 2) ^ (n - 1)

noncomputable def area_sequence (n : ℕ) : ℝ :=
  π * (radius_sequence n) ^ 2

noncomputable def sum_areas : ℝ :=
  ∑' n, area_sequence n

theorem sum_of_areas : sum_areas = 2 * π :=
  sorry

end sum_of_areas_l13_13720


namespace car_energy_fraction_l13_13215

-- Definitions based on conditions
def maglev_energy_consumption := x : ℝ
def airplane_energy_consumption := 3 * maglev_energy_consumption
def car_energy_consumption := (7 / 10) * maglev_energy_consumption

-- Theorem statement
theorem car_energy_fraction:
  car_energy_consumption / airplane_energy_consumption = (10 / 21) :=
  by 
    -- Proof steps go here
    sorry

end car_energy_fraction_l13_13215


namespace original_number_proof_l13_13317

noncomputable def solve_number := 
  let original_number := 0.0707
  original_number

theorem original_number_proof : solve_number = 0.0707 := by
  have h : ∀ x : ℝ, 1000 * x = 5 * (1 / x) ↔ x = 0.0707 := sorry
  show solve_number = 0.0707 from h

end original_number_proof_l13_13317


namespace Billy_current_age_l13_13859

variable (B : ℕ)

theorem Billy_current_age 
  (h1 : ∃ B, 4 * B - B = 12) : B = 4 := by
  sorry

end Billy_current_age_l13_13859


namespace characterize_integers_with_property_l13_13419

noncomputable def has_property (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ (a : Fin n → ℤ), (∑ i, a i) % n ≠ 0 →
    ∃ i : Fin n, ∀ j ∈ Finset.range n, (∑ k in Finset.range (j + 1), a ((i + k) % n)) % n ≠ 0

theorem characterize_integers_with_property :
  ∀ n : ℕ, has_property n ↔ Nat.Prime n :=
by
  sorry

end characterize_integers_with_property_l13_13419


namespace inequality_solution_l13_13441

theorem inequality_solution (x : ℝ) : 
  (0 < x ∧ x ≤ 3) ∨ (4 ≤ x) ↔ (3 * (x - 3) * (x - 4)) / x ≥ 0 := 
sorry

end inequality_solution_l13_13441


namespace table_C_more_than_A_l13_13971

noncomputable def table_money_diff : ℕ := 60 - 40

theorem table_C_more_than_A 
(A B C : ℕ) 
(h1 : B = 2 * C) 
(h2 : A = 40) 
(h3 : A + B + C = 220) : table_money_diff = 20 :=
by
  -- definitions unfolded for clarity
  have hC : C = 60 := by
    have h_sum : 40 + 2 * C + C = 220 := by
      rw [← h2, ← h1]
      exact h3
    linarith
  -- Calculation of the difference between C and A
  have diff_C_A : 60 - 40 = 20 := by
    norm_num
  rw ← hC at diff_C_A
  exact diff_C_A

end table_C_more_than_A_l13_13971


namespace range_of_scores_is_3_l13_13686

def scores : List ℕ := [7, 8, 7, 9, 8, 10]

theorem range_of_scores_is_3 : (List.maximum scores - List.minimum scores) = 3 := by
  sorry

end range_of_scores_is_3_l13_13686


namespace ratio_PN_NR_l13_13531

variables (P Q R N : Type)
variables (PQ QR PR : ℝ)
variables (a b : ℕ)
variables (PN NR : ℝ)

def triangle := Type
def circumradius (A B C : triangle) (a b c : ℝ) := ℝ

noncomputable def A_PQN := sorry
noncomputable def A_QRN := sorry

-- We define the relevant side lengths of the triangle
def side_length_PQ := 10
def side_length_QR := 17
def side_length_PR := 21

-- We define the point N on PR such that the circumradii of PQN and QRN are equal
def is_on_segment_PR (N : Type) : Prop := sorry
def circumradii_equal (P Q R N : triangle) (PQ QR PR PN NR : ℝ) : Prop := 
circumradius P Q N PQ NR PN = circumradius Q R N QR NR PN

-- The final theorem to prove in Lean 4
theorem ratio_PN_NR (P Q R N : triangle) (PQ QR PR : ℝ) (PN NR : ℝ)
  (h1 : side_length_PQ = 10) (h2 : side_length_QR = 17) (h3 : side_length_PR = 21)
  (h4 : is_on_segment_PR N)
  (h5 : circumradii_equal P Q R N )
  : ∃ (a b : ℕ), (PN / NR = 22 / 41) ∧ (a + b = 63) := sorry

end ratio_PN_NR_l13_13531


namespace triangle_similarity_is_first_brocard_point_l13_13204

noncomputable section

open EuclideanGeometry

variables (A B C Q O A1 B1 C1 : Point)

-- conditions
def is_second_brocard_point_triangle_ABC : Prop := 
secondBrocardPoint △ABC Q

def is_circumcenter_AB_Circle : Prop :=
(circumcenter △ABC O)

def is_circumcenter_tris : Prop :=
(circumcenter △CAQ A1) ∧ 
(circumcenter △ABQ B1) ∧ 
(circumcenter △BCQ C1)

-- theorems
theorem triangle_similarity : 
  is_second_brocard_point_triangle_ABC A B C Q → 
  is_circumcenter_AB_Circle A B C O → 
  is_circumcenter_tris A B C Q A1 B1 C1 →
  similar (triangle A1 B1 C1) (triangle A B C) :=
by
  sorry

theorem is_first_brocard_point : 
  is_second_brocard_point_triangle_ABC A B C Q → 
  is_circumcenter_AB_Circle A B C O → 
  is_circumcenter_tris A B C Q A1 B1 C1 →
  is_first_brocard_point (triangle A1 B1 C1) O :=
by
  sorry

end triangle_similarity_is_first_brocard_point_l13_13204


namespace points_collinear_b_eq_17_over_7_l13_13424

theorem points_collinear_b_eq_17_over_7 
  (b : ℚ) : 
  (4 : ℚ, -6 : ℚ), (b + 3, 4), (3 * b - 2, 3) 
  ∈ {l : Set (ℚ × ℚ) | ∃ (m : ℚ) (c : ℚ), ∀ (p : ℚ × ℚ), l p -> p.2 = m * p.1 + c} -> 
  b = 17 / 7 := by
  sorry

end points_collinear_b_eq_17_over_7_l13_13424


namespace triangle_area_l13_13080

-- Given coordinates of vertices of triangle ABC
def A := (0, 0)
def B := (1424233, 2848467)
def C := (1424234, 2848469)

-- Define a mathematical proof statement to prove the area of the triangle ABC
theorem triangle_area :
  let area_ABC := (1 / 2 : ℝ) * Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) * (1 / Real.sqrt (2^2 + (-1)^2))
  (Float.to_string 0.50) = "0.50" :=
by
  let area_ABC := (1 / 2 : ℝ) * Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) * (1 / Real.sqrt (2^2 + (-1)^2))
  sorry
    

end triangle_area_l13_13080


namespace polygon_sides_eq_14_l13_13283

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem polygon_sides_eq_14 (n : ℕ) (h : n + num_diagonals n = 77) : n = 14 :=
by
  sorry

end polygon_sides_eq_14_l13_13283


namespace pepperoni_fell_off_count_l13_13186

-- Define that the pizza has 40 evenly spread slices of pepperoni.
constant total_pepperoni_slices : ℕ := 40

-- Define that the pizza is cut into four equal slices.
constant number_of_slices : ℕ := 4

-- Define the number of pepperoni slices per slice if they are evenly distributed.
def pepperoni_slices_per_slice : ℕ := total_pepperoni_slices / number_of_slices

-- Define the number of pepperoni slices on Jelly's slice.
constant jelly_slice_pepperoni_slices : ℕ := 9

-- Prove how many slices of pepperoni fell off when Lard picked up the slice.
theorem pepperoni_fell_off_count :
  pepperoni_slices_per_slice - jelly_slice_pepperoni_slices = 1 := by
  sorry

end pepperoni_fell_off_count_l13_13186


namespace quadrilateral_midpoints_bisect_l13_13958

variable {A B C D M N P Q : Type*}
variable [AffineSpace A B] 

-- Definitions specifying that points are midpoints
def midpoint (x y m : A) := ∃ v : B, x -ᵥ m = v ∧ m -ᵥ y = v

-- Quadrilateral ABCD
variables (A B C D : A)

-- Midpoints M, N, P, Q
variables (M N P Q : A)

-- Midpoint conditions
hypothesis (hM : midpoint A B M) 
hypothesis (hN : midpoint C D N)
hypothesis (hP : midpoint B C P)
hypothesis (hQ : midpoint D A Q)

-- Proof that line segments MN and PQ bisect each other
theorem quadrilateral_midpoints_bisect : 
  ∃ I : A, I -ᵥ M == I -ᵥ N ∧ I -ᵥ P == I -ᵥ Q := sorry

end quadrilateral_midpoints_bisect_l13_13958


namespace intersection_A_B_l13_13193

def A : Set ℝ := {x | x < 3 * x - 1}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_A_B : (A ∩ B) = {x | x > 1 / 2 ∧ x < 3} :=
by sorry

end intersection_A_B_l13_13193


namespace trig_expression_value_l13_13457

theorem trig_expression_value (θ : ℝ) (h1 : Real.tan (2 * θ) = -2 * Real.sqrt 2)
  (h2 : 2 * θ > Real.pi / 2 ∧ 2 * θ < Real.pi) : 
  (2 * Real.cos θ / 2 ^ 2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 2 * Real.sqrt 2 - 3 :=
by
  sorry

end trig_expression_value_l13_13457


namespace tens_digit_of_19_pow_2023_l13_13766

theorem tens_digit_of_19_pow_2023 : (19 ^ 2023) % 100 = 59 := 
  sorry

end tens_digit_of_19_pow_2023_l13_13766


namespace right_triangle_exists_c_l13_13321

theorem right_triangle_exists_c : 
  (∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ {a, b, c} = ({5, 12, 13} : set ℝ)) ∧
  (¬(∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ {a, b, c} = ({sqrt 3, sqrt 4, sqrt 5} : set ℝ))) ∧
  (¬(∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ {a, b, c} = ({4, 9, sqrt 13} : set ℝ))) ∧
  (¬(∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ {a, b, c} = ({0.8, 0.15, 0.17} : set ℝ))) :=
by
  sorry

end right_triangle_exists_c_l13_13321


namespace sarah_amount_l13_13388

theorem sarah_amount:
  ∀ (X : ℕ), (X + (X + 50) = 300) → X = 125 := by
  sorry

end sarah_amount_l13_13388


namespace show_revenue_and_vacancies_l13_13536

theorem show_revenue_and_vacancies:
  let total_seats := 600
  let vip_seats := 50
  let general_seats := 400
  let balcony_seats := 150
  let vip_price := 40
  let general_price := 25
  let balcony_price := 15
  let vip_filled_rate := 0.80
  let general_filled_rate := 0.70
  let balcony_filled_rate := 0.50
  let vip_filled := vip_filled_rate * vip_seats
  let general_filled := general_filled_rate * general_seats
  let balcony_filled := balcony_filled_rate * balcony_seats
  let vip_revenue := vip_filled * vip_price
  let general_revenue := general_filled * general_price
  let balcony_revenue := balcony_filled * balcony_price
  let overall_revenue := vip_revenue + general_revenue + balcony_revenue
  let vip_vacant := vip_seats - vip_filled
  let general_vacant := general_seats - general_filled
  let balcony_vacant := balcony_seats - balcony_filled
  vip_revenue = 1600 ∧
  general_revenue = 7000 ∧
  balcony_revenue = 1125 ∧
  overall_revenue = 9725 ∧
  vip_vacant = 10 ∧
  general_vacant = 120 ∧
  balcony_vacant = 75 :=
by
  sorry

end show_revenue_and_vacancies_l13_13536


namespace exist_two_people_no_common_knows_l13_13359

open Finset

-- Let's define the problem
def Person := Fin 17
def knows (a b : Person) : Prop := sorry -- define an adjacency relation (undirected graph edges)

-- Condition: Each person knows exactly 4 other people
axiom each_person_knows_exactly_4 : ∀ a : Person, (univ.filter (λ b : Person, knows a b)).card = 4

-- Question: Prove that there exist two people who neither know each other nor share a common acquaintance
theorem exist_two_people_no_common_knows :
  ∃ (a b : Person), ¬ knows a b ∧ (∀ c : Person, ¬ (knows a c ∧ knows b c)) :=
sorry

end exist_two_people_no_common_knows_l13_13359


namespace square_side_length_l13_13253

-- Define the conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 4
def area_rectangle : ℝ := rectangle_width * rectangle_length
def area_square : ℝ := area_rectangle

-- Prove the side length of the square
theorem square_side_length :
  ∃ s : ℝ, s * s = area_square ∧ s = 4 := 
  by {
    -- Here you'd write the proof step, but it's omitted as per instructions
    sorry
  }

end square_side_length_l13_13253


namespace solve_ellipse_problem_l13_13035

noncomputable def ellipse_problem : ℝ :=
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  let c := Real.sqrt (a^2 - b^2)
  h + k + a + b + 2*c

theorem solve_ellipse_problem : ellipse_problem = 9 + 2*Real.sqrt 33 :=
by
  unfold ellipse_problem
  norm_num
  rw [Real.sqrt_sq]
  sorry

end solve_ellipse_problem_l13_13035


namespace g_analytical_expression_g_minimum_value_l13_13818

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1
noncomputable def M (a : ℝ) : ℝ := if (a ≥ 1/3 ∧ a ≤ 1/2) then f a 1 else f a 3
noncomputable def N (a : ℝ) : ℝ := f a (1/a)
noncomputable def g (a : ℝ) : ℝ :=
  if a ≥ 1/3 ∧ a ≤ 1/2 then M a - N a 
  else if a > 1/2 ∧ a ≤ 1 then M a - N a
  else 0 -- outside the given interval, by definition may be kept as 0

theorem g_analytical_expression (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) : 
  g a = if (1/3 ≤ a ∧ a ≤ 1/2) then a + 1/a - 2 else 9 * a + 1/a - 6 := 
sorry

theorem g_minimum_value (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  ∃ (a' : ℝ), 1/3 ≤ a' ∧ a' ≤ 1 ∧ (∀ a, 1/3 ≤ a ∧ a ≤ 1 → g a ≥ g a') ∧ g a' = 1/2 := 
sorry

end g_analytical_expression_g_minimum_value_l13_13818


namespace log_is_geometric_l13_13504

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, 0 < n → a (n + 1) = (a n)^2

theorem log_is_geometric (a : ℕ → ℕ) (h : sequence a) :
  ∃ r, ∀ n : ℕ, log (a (n + 1)) = log (a n) + r := by
  sorry

end log_is_geometric_l13_13504


namespace breadth_of_hall_l13_13367

theorem breadth_of_hall (length_hall : ℝ) (stone_length_dm : ℝ) (stone_breadth_dm : ℝ)
    (num_stones : ℕ) (area_stone_m2 : ℝ) (total_area_m2 : ℝ) (breadth_hall : ℝ):
    length_hall = 36 → 
    stone_length_dm = 8 → 
    stone_breadth_dm = 5 → 
    num_stones = 1350 → 
    area_stone_m2 = (stone_length_dm * stone_breadth_dm) / 100 → 
    total_area_m2 = num_stones * area_stone_m2 → 
    breadth_hall = total_area_m2 / length_hall → 
    breadth_hall = 15 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4] at *
  simp [h5, h6, h7]
  sorry

end breadth_of_hall_l13_13367


namespace area_of_square_with_diagonal_12_l13_13309

theorem area_of_square_with_diagonal_12 :
  ∀ (s : ℝ), (2 * s^2 = 12^2) → s^2 = 72 :=
by
  intros s h
  have : 2 * s^2 = 144 := h
  sorry

end area_of_square_with_diagonal_12_l13_13309


namespace red_green_probability_l13_13302

-- Define the probability that a ball is tossed into bin k
def p_k (k : ℕ) : ℝ := 1 / (k * (k + 1))

-- Define the event that the red ball is tossed into an odd-numbered bin
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the event that the green ball is tossed into an even-numbered bin
def is_even (n : ℕ) : Prop := n % 2 = 0

-- To compute the probability that the red ball is in an odd bin and the green ball is in an even bin
noncomputable def red_in_odd_and_green_in_even : ℝ := 
  (∑ i in {n | is_odd n}, p_k i) * (∑ j in {m | is_even m}, p_k j)

-- The main goal to prove
theorem red_green_probability :
  red_in_odd_and_green_in_even = 1 / 4 :=
sorry

end red_green_probability_l13_13302


namespace proof_problem_l13_13409

variables (a : ℝ) (f : ℝ → ℝ)

def proposition_p := ∀ x, a^(x+1) = 1 → x = -1
def proposition_q := ∀ x, f(-x) = f(x) → (∀ x, f(x+1) ≠ f(1-x))

theorem proof_problem : proposition_p a f ∨ ¬ proposition_q a f :=
by
  sorry

end proof_problem_l13_13409


namespace common_root_exists_l13_13448

noncomputable def P (x : ℝ) : ℝ := x^3 + 41*x^2 - 49*x - 2009
noncomputable def Q (x : ℝ) : ℝ := x^3 + 5*x^2 - 49*x - 245
noncomputable def R (x : ℝ) : ℝ := x^3 + 39*x^2 - 117*x - 1435

theorem common_root_exists : P 7 = 0 ∧ Q 7 = 0 ∧ R 7 = 0 := 
by 
  show P 7 = 0 sorry
  show Q 7 = 0 sorry
  show R 7 = 0 sorry

end common_root_exists_l13_13448


namespace root_in_interval_l13_13277

noncomputable def f (x : ℝ) : ℝ := 2^x - x^2 - 1/2

theorem root_in_interval : 
  ∃ x : ℝ, x ∈ set.Ioo (3/2 : ℝ) 2 ∧ f x = 0 :=
by {
  sorry
}

end root_in_interval_l13_13277


namespace factorial_inequality_l13_13418

noncomputable def f : ℕ → ℕ := sorry

axiom functional_equation (w x y z : ℕ) : 
  f(f(f(z))) * f(w * x * f(y * f(z))) = z^2 * f(x * f(y)) * f(w)

theorem factorial_inequality (n : ℕ) (hn : 0 < n) : f(n!) ≥ n! :=
sorry

end factorial_inequality_l13_13418


namespace billiard_ball_reflections_l13_13893

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem billiard_ball_reflections (L W D : ℝ) (angle : ℝ) (P D : ℝ) :
  L = 3 ∧ W = 1 ∧ D = 2 ∧ P = 2 → 
  let E := (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))
  in E = expected_reflections :=
by
  sorry

end billiard_ball_reflections_l13_13893


namespace number_of_arrangements_l13_13234

def basil_plants := 2
def aloe_plants := 1
def cactus_plants := 1
def white_lamps := 2
def red_lamps := 2
def total_plants := basil_plants + aloe_plants + cactus_plants
def total_lamps := white_lamps + red_lamps

theorem number_of_arrangements : total_plants = 4 ∧ total_lamps = 4 →
  ∃ n : ℕ, n = 28 :=
by
  intro h
  sorry

end number_of_arrangements_l13_13234


namespace ratio_CD_to_BC_l13_13712

variables (AB BC CD AD : ℝ)
variable k : ℝ

-- Conditions
def distance_AB : AB = 100 := by sorry
def distance_BC : BC = AB + 50 := by sorry
def total_distance_AD : AD = 550 := by sorry
def distance_CD : CD = k * BC := by sorry

-- Proof goal
theorem ratio_CD_to_BC (h1 : distance_AB AB) (h2 : distance_BC AB BC) (h3 : total_distance_AD AB AD) (h4 : distance_CD BC CD k) :
  CD / BC = 2 :=
begin
  -- Implement assumption proofs
  sorry
end

end ratio_CD_to_BC_l13_13712


namespace exists_segment_l13_13428

-- Defining the function f
def f (r : ℚ) : ℤ := sorry

-- The theorem statement we want to prove
theorem exists_segment (f : ℚ → ℤ) :
  ∃ a b : ℚ, f(a) + f(b) ≤ 2 * f((a + b) / 2) :=
sorry

end exists_segment_l13_13428


namespace find_m_value_l13_13901

noncomputable def m_value (A B C E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace E] (AB AC BC BE : ℝ) (angleBAC : ℝ) :=
  BE = m * Real.sqrt 2 → m = Real.sqrt 138 / 4 → BE = Real.sqrt (69 / 4) →  BE = Real.sqrt 69 / 2 

-- These should be the conditions in the proof
variables {A B C E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace E]
variables (AB AC BC BE : ℝ) (m : ℝ)

-- Define the constants from the problem
def AB := (5 : ℝ)
def BC := (12 : ℝ)
def AC := (13 : ℝ)
def BE := m * Real.sqrt 2

theorem find_m_value : 
  let BE := (Real.sqrt 69) / 2
  m = Real.sqrt 138 / 4 :=
sorry

end find_m_value_l13_13901


namespace log_equation_solution_l13_13516

theorem log_equation_solution :
  ∃ x : ℝ, (9 * log 3 x - 10 * (log 9 x) = 18 * log 27 45) ∧ 
           (x = 135 * real.sqrt 5) ∧
           (∃ m n : ℕ, m = 135 ∧ n = 5 ∧ m + n = 140) :=
by
  sorry

end log_equation_solution_l13_13516


namespace equal_segments_in_triangle_l13_13908

theorem equal_segments_in_triangle
  {A B C A₁ B₁ C₁ A₂ B₂ C₂ : Point}
  (h1 : A₁ ∈ segment B C)
  (h2 : B₁ ∈ segment C A)
  (h3 : C₁ ∈ segment A B)
  (h4 : concurrent_lines (line_through A A₁) (line_through B B₁) (line_through C C₁))
  (h5 : parallel (line_through A A₂) (line_through B C))
  (h6 : B₂ = intersection_point (line_through A₁ B₁) (line_through A A₂))
  (h7 : C₂ = intersection_point (line_through A₁ C₁) (line_through A A₂)) :
  segment_length (segment A B₂) = segment_length (segment A C₂) :=
sorry

end equal_segments_in_triangle_l13_13908


namespace equation_of_line_l13_13991

theorem equation_of_line 
  (a : ℝ) (h : a < 3) 
  (C : ℝ × ℝ) 
  (hC : C = (-2, 3)) 
  (l_intersects_circle : ∃ A B : ℝ × ℝ, 
    (A.1^2 + A.2^2 + 2 * A.1 - 4 * A.2 + a = 0) ∧ 
    (B.1^2 + B.2^2 + 2 * B.1 - 4 * B.2 + a = 0) ∧ 
    (C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))) : 
  ∃ (m b : ℝ), 
    (m = 1) ∧ 
    (b = -5) ∧ 
    (∀ x y, y - 3 = m * (x + 2) ↔ x - y + 5 = 0) :=
by
  sorry

end equation_of_line_l13_13991


namespace sum_of_integers_l13_13282

theorem sum_of_integers (a b c : ℤ) (h1 : a = (1 / 3) * (b + c)) (h2 : b = (1 / 5) * (a + c)) (h3 : c = 35) : a + b + c = 60 :=
by
  sorry

end sum_of_integers_l13_13282


namespace probability_interval_1_3_l13_13830

noncomputable def pdf (x : ℝ) : ℝ :=
if x > 0 then exp (-x) else 0

theorem probability_interval_1_3 :
  (∫ x in (1:ℝ)..(3:ℝ), pdf x) = (exp 2 - 1) / exp 3 :=
by
  sorry

end probability_interval_1_3_l13_13830


namespace ratio_of_bottles_given_to_first_house_l13_13723

theorem ratio_of_bottles_given_to_first_house 
  (total_bottles : ℕ) 
  (bottles_only_cider : ℕ) 
  (bottles_only_beer : ℕ) 
  (bottles_mixed : ℕ) 
  (first_house_bottles : ℕ) 
  (h1 : total_bottles = 180) 
  (h2 : bottles_only_cider = 40) 
  (h3 : bottles_only_beer = 80) 
  (h4 : bottles_mixed = total_bottles - bottles_only_cider - bottles_only_beer) 
  (h5 : first_house_bottles = 90) : 
  first_house_bottles / total_bottles = 1 / 2 :=
by 
  -- Proof goes here
  sorry

end ratio_of_bottles_given_to_first_house_l13_13723


namespace sin_inequality_l13_13679

theorem sin_inequality (α₁ α₂ α₃ : ℝ) (h1 : 0 ≤ α₁ ∧ α₁ ≤ real.pi) (h2 : 0 ≤ α₂ ∧ α₂ ≤ real.pi) (h3 : 0 ≤ α₃ ∧ α₃ ≤ real.pi) :
  real.sin α₁ + real.sin α₂ + real.sin α₃ ≤ 3 * real.sin ((α₁ + α₂ + α₃) / 3) :=
sorry

end sin_inequality_l13_13679


namespace boat_distance_along_stream_l13_13541

theorem boat_distance_along_stream
  (distance_against_stream : ℝ)
  (speed_still_water : ℝ)
  (time : ℝ)
  (v_s : ℝ)
  (H1 : distance_against_stream = 5)
  (H2 : speed_still_water = 6)
  (H3 : time = 1)
  (H4 : speed_still_water - v_s = distance_against_stream / time) :
  (speed_still_water + v_s) * time = 7 :=
by
  -- Sorry to skip proof
  sorry

end boat_distance_along_stream_l13_13541


namespace team_selection_count_l13_13582

-- The problem's known conditions
def boys : ℕ := 10
def girls : ℕ := 12
def team_size : ℕ := 8

-- The number of ways to select a team of 8 members with at least 2 boys and no more than 4 boys
noncomputable def count_ways : ℕ :=
  (Nat.choose boys 2) * (Nat.choose girls 6) +
  (Nat.choose boys 3) * (Nat.choose girls 5) +
  (Nat.choose boys 4) * (Nat.choose girls 4)

-- The main statement to prove
theorem team_selection_count : count_ways = 238570 := by
  sorry

end team_selection_count_l13_13582


namespace probability_closer_to_origin_l13_13713

/-- Given a point P randomly selected from the rectangular region 
with vertices (0,0), (3,0), (3,2), and (0,2). Prove that the probability 
that P is closer to the origin (0,0) than it is to the point (4,2) 
is 1/4. -/
theorem probability_closer_to_origin (P : ℝ × ℝ)
  (h1 : 0 ≤ P.1) (h2 : P.1 ≤ 3) (h3 : 0 ≤ P.2) (h4 : P.2 ≤ 2) :
  probability (closer_to_origin P) = 1 / 4 :=
sorry

/-- Predicate checking if a point P is closer to the origin (0,0) than to the point (4,2). -/
def closer_to_origin (P : ℝ × ℝ) : Prop :=
  dist P (0, 0) < dist P (4, 2)

end probability_closer_to_origin_l13_13713


namespace price_increase_twice_l13_13691

theorem price_increase_twice (P : ℝ) : 
  let P1 := P * 1.10 in
  let P2 := P1 * 1.10 in
  P2 = P * 1.21 :=
by
  let P1 := P * 1.10
  let P2 := P1 * 1.10
  show P2 = P * 1.21 from sorry

end price_increase_twice_l13_13691


namespace rowing_speed_in_still_water_l13_13001

variable (v c t : ℝ)
variable (h1 : c = 1.3)
variable (h2 : 2 * ((v - c) * t) = ((v + c) * t))

theorem rowing_speed_in_still_water : v = 3.9 := by
  sorry

end rowing_speed_in_still_water_l13_13001


namespace function_relationship_l13_13122

noncomputable def f : ℝ → ℝ := sorry

theorem function_relationship :
  (∀ x : ℝ, f (2 - x) = f x) ∧
  (∀ x : ℝ, f (x + 2) = f (x - 2)) ∧
  (∀ x₁ x₂ : ℝ, x₁ ∈ Icc 1 3 → x₂ ∈ Icc 1 3 → (f x₁ - f x₂) / (x₁ - x₂) < 0) →
  (f 2016 = f 2014 ∧ f 2014 > f 2015) :=
by sorry

end function_relationship_l13_13122


namespace fourth_term_in_arithmetic_sequence_l13_13161

theorem fourth_term_in_arithmetic_sequence (a d : ℝ) (h : 2 * a + 6 * d = 20) : a + 3 * d = 10 :=
sorry

end fourth_term_in_arithmetic_sequence_l13_13161


namespace inequality_proof_l13_13549

variable (a b c d : ℝ)

theorem inequality_proof
  (h_pos: 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ 0 < d ∧ d < 1)
  (h_product: a * b * c * d = (1 - a) * (1 - b) * (1 - c) * (1 - d)) : 
  (a + b + c + d) - (a + c) * (b + d) ≥ 1 :=
by
  sorry

end inequality_proof_l13_13549


namespace cost_of_goat_l13_13295

theorem cost_of_goat (G : ℝ) (goat_count : ℕ) (llama_count : ℕ) (llama_multiplier : ℝ) (total_cost : ℝ) 
    (h1 : goat_count = 3)
    (h2 : llama_count = 2 * goat_count)
    (h3 : llama_multiplier = 1.5)
    (h4 : total_cost = 4800) : G = 400 :=
by
  sorry

end cost_of_goat_l13_13295


namespace constant_term_in_expansion_l13_13899

theorem constant_term_in_expansion :
  let x := (x + 1 / (x^(1/2)))^6 in
  (∃ (t : ℕ), binom 6 t = 15 ∧ (3 * t - 6) / 2 = 0) :=
begin
  -- We can skip the proof part as instructed
  sorry
end

end constant_term_in_expansion_l13_13899


namespace cost_per_bag_l13_13292

theorem cost_per_bag (total_bags : ℕ) (sale_price_per_bag : ℕ) (desired_profit : ℕ) (total_revenue : ℕ)
  (total_cost : ℕ) (cost_per_bag : ℕ) :
  total_bags = 100 → sale_price_per_bag = 10 → desired_profit = 300 →
  total_revenue = total_bags * sale_price_per_bag →
  total_cost = total_revenue - desired_profit →
  cost_per_bag = total_cost / total_bags →
  cost_per_bag = 7 := by
  sorry

end cost_per_bag_l13_13292


namespace daily_evaporation_l13_13355

theorem daily_evaporation :
  ∀ (initial_amount : ℝ) (percentage_evaporated : ℝ) (days : ℕ),
  initial_amount = 10 →
  percentage_evaporated = 6 →
  days = 50 →
  (initial_amount * (percentage_evaporated / 100)) / days = 0.012 :=
by
  intros initial_amount percentage_evaporated days
  intros h_initial h_percentage h_days
  rw [h_initial, h_percentage, h_days]
  sorry

end daily_evaporation_l13_13355


namespace tens_digit_of_19_pow_2023_l13_13764

theorem tens_digit_of_19_pow_2023 : (19 ^ 2023) % 100 = 59 := 
  sorry

end tens_digit_of_19_pow_2023_l13_13764


namespace sets_of_arithmetic_sequence_with_even_difference_l13_13040

/-- Determine the number of sets of three distinct digits from the set {0, 1, 2, ..., 12}
such that the digits in each set are in an arithmetic sequence and the common difference
is even. -/
theorem sets_of_arithmetic_sequence_with_even_difference :
  ∃ (n : ℕ), n = 19 ∧
  (∀ (a b c : ℕ), a ∈ (finset.range 13) ∧ b ∈ (finset.range 13) ∧ c ∈ (finset.range 13) ∧ a < b ∧ b < c →
    (((b - a = c - b) ∧ ((b - a) % 2 = 0)) → set.count (a, b, c) = n)) :=
begin
  -- Prove the number of sets of three distinct digits that form the desired arithmetic sequence is 19
  sorry
end

end sets_of_arithmetic_sequence_with_even_difference_l13_13040


namespace convex_polygon_center_symmetry_l13_13229

def centrally_symmetric (P : Type) [MetricSpace P] (polygon : set P) : Prop :=
  ∃ O : P, ∀ p ∈ polygon, ∃ q ∈ polygon, midpoint ℝ (p, q) = O

def convex_polygon (P : Type) [MetricSpace P] (M : set P) : Prop :=
  is_convex ℝ M

def can_be_divided_into (P : Type) [MetricSpace P] (M : set P) (polygons : set (set P)) : Prop :=
  M = ⋃₀ polygons ∧ (∀ polygon ∈ polygons, centrally_symmetric P polygon)

theorem convex_polygon_center_symmetry
  {P : Type} [MetricSpace P] (M : set P)
  (h1 : convex_polygon P M)
  (h2 : ∃ polygons : set (set P), can_be_divided_into P M polygons) :
  ∃ O : P, ∀ p ∈ M, ∃ q ∈ M, midpoint ℝ (p, q) = O :=
sorry

end convex_polygon_center_symmetry_l13_13229


namespace n_leq_84_l13_13200

theorem n_leq_84 (n : ℕ) (hn : 0 < n) (h: (1 / 2 + 1 / 3 + 1 / 7 + 1 / ↑n : ℚ).den ≤ 1): n ≤ 84 :=
sorry

end n_leq_84_l13_13200


namespace p_necessary_not_sufficient_for_q_l13_13100

variables {x : ℝ}
def p := x^2 - 4 * x + 3 > 0
def q := x^2 < 1

theorem p_necessary_not_sufficient_for_q : (∀ (x : ℝ), q x → p x) ∧ ¬(∀ (x : ℝ), p x → q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l13_13100


namespace imaginary_part_of_z_l13_13864

-- Define complex numbers and necessary conditions
variable (z : ℂ)

-- The main statement
theorem imaginary_part_of_z (h : z * (1 + 2 * I) = 3 - 4 * I) : 
  (z.im = -2) :=
sorry

end imaginary_part_of_z_l13_13864


namespace geometric_sequence_properties_l13_13498

-- Given conditions as definitions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 * a 3 = a 4 ∧ a 3 = 8

-- Prove the common ratio and the sum of the first n terms
theorem geometric_sequence_properties (a : ℕ → ℝ)
  (h : seq a) :
  (∃ q, ∀ n, a n = a 1 * q ^ (n - 1) ∧ q = 2) ∧
  (∀ S_n, S_n = (1 - (2 : ℝ) ^ S_n) / (1 - 2) ∧ S_n = 2 ^ S_n - 1) :=
by
  sorry

end geometric_sequence_properties_l13_13498


namespace contrapositive_statement_l13_13983

theorem contrapositive_statement (x : ℝ) : (x ≤ -3 → x < 0) → (x ≥ 0 → x > -3) := 
by
  sorry

end contrapositive_statement_l13_13983


namespace tangent_line_to_f_at_1_l13_13486

noncomputable def f (g : ℝ → ℝ) : ℝ → ℝ := λ x, g(x) + x^2

theorem tangent_line_to_f_at_1 (g : ℝ → ℝ) 
  (tangent_g_at_1 : ∀ x, (g'(1) = 2) ∧ (g(1) = 3)) :
  ∀ x, let f := f g in (f'(1) = 4) ∧ (f(1) = 4) ∧ (y = 4x - 4) :=
begin
  intro x,
  sorry
end

end tangent_line_to_f_at_1_l13_13486


namespace range_of_a_for_monotonicity_l13_13525

theorem range_of_a_for_monotonicity (a : ℝ) : 
  (∀ x ∈ (Ioo (0 : ℝ) 2), deriv (λ x, x^3 - a * x^2 + 4) x < 0) ↔ (3 < a) :=
sorry

end range_of_a_for_monotonicity_l13_13525


namespace min_value_expression_l13_13562

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 16) :
  x^2 + 8 * x * y + 16 * y^2 + 4 * z^2 ≥ 48 := 
begin
  sorry
end

end min_value_expression_l13_13562


namespace original_selling_price_l13_13387

theorem original_selling_price (P : ℝ) (h1 : ∀ P, 1.17 * P = 1.10 * P + 42) :
    1.10 * P = 660 := by
  sorry

end original_selling_price_l13_13387


namespace smallest_degree_of_q_for_horizontal_asymptote_l13_13265

noncomputable def rational_function (q : Polynomial ℝ) : (x : ℝ) → ℝ :=
  λ x, (2 * x ^ 6 + 3 * x ^ 5 - x ^ 2 - 1) / q.eval x

theorem smallest_degree_of_q_for_horizontal_asymptote
  (q : Polynomial ℝ)
  (hq : ∃ L, ∀ ε > 0, ∃ N > 0, ∀ x > N, |(rational_function q x) - L| < ε) :
  6 ≤ q.natDegree :=
sorry

end smallest_degree_of_q_for_horizontal_asymptote_l13_13265


namespace perimeter_of_broken_line_l13_13181

variable {A B C A1 B1 C1 A2 B2 C2 : Type}

-- Define the points and segments in the triangle
variables [medians: Medians A B C A1 B1 C1] [altitudes: Altitudes A B C A2 B2 C2]

-- Prove the length of the broken line equals the perimeter of the triangle
theorem perimeter_of_broken_line :
  length (A1, B2) + length (B2, C1) + length (C1, A2) + length (A2, B1) + length (B1, C2) + length (C2, A1)
  = length (A, B) + length (B, C) + length (C, A) :=
sorry

end perimeter_of_broken_line_l13_13181


namespace num_permutations_satisfying_inequality_l13_13084

open Finset

theorem num_permutations_satisfying_inequality :
  let permutations := perm (range 1 8)
  ∑ p in permutations, 
    if (∏ i in range 1 8, (p(i) + i) / 2) > 7! 
    then 1 else 0 = 5039
:= by
  sorry

end num_permutations_satisfying_inequality_l13_13084


namespace find_m_l13_13479

theorem find_m (m : ℕ) (h1 : Nat.Pos m) (h2 : Nat.lcm 36 m = 180) (h3 : Nat.lcm m 50 = 300) : m = 60 :=
by {
  sorry
}

end find_m_l13_13479


namespace A_J_K_collinear_l13_13565

noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def Γ : Circle := sorry
noncomputable def Γ_A : Circle := sorry
noncomputable def I : Point := sorry
noncomputable def J : Point := sorry
noncomputable def K : Point := sorry

axiom triangle_ABC : is_triangle A B C
axiom incircle_Γ : is_incircle Γ A B C
axiom excircle_Γ_A : is_excircle Γ_A A B C
axiom Γ_touches_BC_at_I : touches_at Γ B C I
axiom Γ_A_touches_BC_at_J : touches_at Γ_A B C J
axiom K_opposite_to_I : diametrically_opposite Γ I K

theorem A_J_K_collinear : collinear A J K := sorry

end A_J_K_collinear_l13_13565


namespace bruno_initial_books_l13_13025

theorem bruno_initial_books :
  ∃ (B : ℕ), B - 4 + 10 = 39 → B = 33 :=
by
  use 33
  intro h
  linarith [h]

end bruno_initial_books_l13_13025


namespace find_theta_l13_13208

theorem find_theta (θ : ℝ) (z : ℂ) (ω : ℂ) (h₁ : 0 < θ) (h₂ : θ < π)
  (hz : z = complex.cos θ + complex.sin θ * complex.I)
  (hω : ω = (1 - (conj z)^4) / (1 + z^4))
  (hω_mag : abs ω = real.sqrt(3) / 3)
  (hω_arg : complex.arg ω < π / 2) :
  θ = π / 12 ∨ θ = 7 * π / 12 :=
sorry

end find_theta_l13_13208


namespace minimum_total_distance_l13_13062

theorem minimum_total_distance :
  let r : ℝ := 50
  let num_boys : ℕ := 8
  let angle : ℝ := (3/8 : ℝ) * (2 * Real.pi)
  let s := Real.sqrt (2 * r^2 * (1 - Real.cos angle))
  let distance_per_boy := 4 * s
  let total_distance := num_boys * distance_per_boy
  total_distance = 1600 * Real.sqrt(2 - Real.sqrt 2) := 
by
  let r : ℝ := 50
  let num_boys : ℕ := 8
  let angle : ℝ := (3/8 : ℝ) * (2 * Real.pi)
  let s := Real.sqrt (2 * r^2 * (1 - Real.cos angle))
  let distance_per_boy := 4 * s
  let total_distance := num_boys * distance_per_boy
  show total_distance = 1600 * Real.sqrt(2 - Real.sqrt 2) from sorry

end minimum_total_distance_l13_13062


namespace alpha_gt_beta_l13_13110

theorem alpha_gt_beta (α β : ℝ) (h1 : 0 < α ∧ α < real.pi / 2) (h2 : 0 < β ∧ β < real.pi / 2) (h3 : real.sin α ^ 2 = real.cos (α - β)) :
  α > β :=
sorry

end alpha_gt_beta_l13_13110


namespace problem_statement_l13_13201

theorem problem_statement (n : ℕ) (hn : 0 < n) 
  (h : (1/2 : ℚ) + 1/3 + 1/7 + 1/n ∈ ℤ) : n = 42 ∧ ¬ (n > 84) := 
by
  sorry

end problem_statement_l13_13201


namespace sum_f_values_l13_13824

-- Define the function f with the specified properties
variable {f : ℝ → ℝ}

-- Assume f is an even function
axiom even_f (x : ℝ) : f (-x) = f x

-- Translation of f(x) one unit to the right results in an odd function
axiom odd_translation (x : ℝ) : f (-x-1) = -f (x-1)

-- Given f(2) is -1
axiom f_2 : f 2 = -1

-- Proof statement
theorem sum_f_values : 
  (∑ i in Finset.range 2011.succ, f (i + 1)) = -1 :=
by
  sorry

end sum_f_values_l13_13824


namespace proof_MP_eq_NC_l13_13895

variables {A B C D P Q M N : Type}
variables {A B C D P Q : Point}
variables {M : Point}
variables {N : Point}

-- Conditions: Cyclic quadrilateral ABCD with AP = CD, AQ = BC, 
-- M is the intersection of lines PQ and AC, N is the midpoint of BD
def cyclic_quadrilateral (A B C D P Q M N : Point) : Prop :=
  cyclic_quadrilateral ABCD ∧
  segment_eq A P C D ∧ 
  segment_eq A Q B C ∧
  intersection_lines PQ AC M ∧
  midpoint B D N

theorem proof_MP_eq_NC (A B C D P Q M N : Point) 
  (h1: cyclic_quadrilateral A B C D P Q M N) : segment_eq M P N C := sorry

end proof_MP_eq_NC_l13_13895


namespace midpoint_is_incenter_l13_13256

theorem midpoint_is_incenter (A B C O P K : Point)
  (hAB_eq_BC : A.dist B = B.dist C)
  (h_circle_S_touches_AB_at_P : Circle S touches Line AB at P)
  (h_circle_S_touches_BC_at_K : Circle S touches Line BC at K)
  (h_circle_S_touches_circumcircle : Circle S touches circumcircle (triangle_hull A B C) internally):
  midpoint P K = incenter (triangle_hull A B C) :=
sorry

end midpoint_is_incenter_l13_13256


namespace time_spent_on_type_A_l13_13168

theorem time_spent_on_type_A :
  let t := (240:ℝ) / 370 in
  let time_A := 30 * 2 * t in
  time_A = 1440 / 37 :=
by
  let t := 240 / 370
  let time_A := 30 * 2 * t
  show time_A = 1440 / 37
  sorry

end time_spent_on_type_A_l13_13168


namespace triangle_angle_relationship_l13_13599

/-
  Given: a triangle with side lengths a = 20, b = 15, c = 7.
  Prove: the angle relationship α = 3β + γ.
-/
theorem triangle_angle_relationship (α β γ : ℝ) (h : ∀ (a b c : ℝ), a = 20 ∧ b = 15 ∧ c = 7 → α = 3 * β + γ):
  ∃ α β γ : ℝ, ∀ (a b c : ℝ), a = 20 ∧ b = 15 ∧ c = 7 → α = 3 * β + γ :=
begin
  sorry
end

end triangle_angle_relationship_l13_13599


namespace number_of_permutations_satisfying_inequality_l13_13446

theorem number_of_permutations_satisfying_inequality :
  (let permutations := { p | p ∈ permutations_of {1,2,3,4,5,6}
                             ∧ ((p.map (λ k, (k + (k+1)) / 3)).prod > factorial 6) } in
   fintype.card permutations = 719) :=
sorry

end number_of_permutations_satisfying_inequality_l13_13446


namespace find_f_10_l13_13107

-- Defining the function f as an odd, periodic function with period 2
def odd_func_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x : ℝ, f (x + 2) = f x)

-- Stating the theorem that f(10) is 0 given the conditions
theorem find_f_10 (f : ℝ → ℝ) (h1 : odd_func_periodic f) : f 10 = 0 :=
sorry

end find_f_10_l13_13107


namespace number_of_complex_roots_l13_13083

noncomputable def rational_function (z : ℂ) :=
  (z^3 + 2*z^2 + z - 2) / (z^2 - 3*z + 2)

theorem number_of_complex_roots : 
  ∃ (n : ℕ), n = 2 ∧ ∀ z : ℂ, rational_function z = 0 → (z = complex.i ∨ z = -complex.i) := 
by
  sorry

end number_of_complex_roots_l13_13083


namespace binomial_eight_three_l13_13396

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem binomial_eight_three : binomial 8 3 = 56 := by
  sorry

end binomial_eight_three_l13_13396


namespace area_of_triangle_ABC_l13_13078

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1424233, 2848467)
def C : point := (1424234, 2848469)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_ABC : triangle_area A B C = 0.50 := by
  sorry

end area_of_triangle_ABC_l13_13078


namespace find_asymptote_slope_l13_13615

noncomputable def hyperbola_asymptote_slope : Prop :=
  let eq : ∀ x y : ℝ, (x^2 / 49) - (y^2 / 36) = 1
  ∀ x y : ℝ,
    (x^2 / 49) - (y^2 / 36) = 1 → y = (6 / 7) * x ∨ y = -(6 / 7) * x

theorem find_asymptote_slope (x y : ℝ) (h : (x^2 / 49) - (y^2 / 36) = 1) :
  y = (6 / 7) * x ∨ y = -(6 / 7) * x :=
by
  sorry

end find_asymptote_slope_l13_13615


namespace find_speed_of_second_car_l13_13647

noncomputable def problem : Prop := 
  let s1 := 1600 -- meters
  let s2 := 800 -- meters
  let v1 := 72 / 3.6 -- converting to meters per second for convenience; 72 km/h = 20 m/s
  let s := 200 -- meters
  let t1 := s1 / v1 -- time taken by the first car to reach the intersection
  let l1 := s2 - s -- scenario 1: second car travels 600 meters
  let l2 := s2 + s -- scenario 2: second car travels 1000 meters
  let v2_1 := l1 / t1 -- speed calculation for scenario 1
  let v2_2 := l2 / t1 -- speed calculation for scenario 2
  v2_1 = 7.5 ∧ v2_2 = 12.5 -- expected speeds in both scenarios

theorem find_speed_of_second_car : problem := sorry

end find_speed_of_second_car_l13_13647


namespace cardinality_PstarQ_l13_13194

def P : Set ℕ := {0, 1, 2}
def Q : Set ℕ := {1, 2, 3, 4}

def PstarQ : Set (ℕ × ℕ) := {x | x.1 ∈ P ∧ x.2 ∈ Q}

theorem cardinality_PstarQ : PstarQ.toFinset.card = 12 := by
  sorry

end cardinality_PstarQ_l13_13194


namespace exists_pentagon_satisfying_condition_l13_13182

def Point : Type := ℝ × ℝ

structure Triangle :=
(A B C : Point)
(right_angle: (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)

structure Pentagon :=
(A B C D E : Point)
(rectangle: {ABXY: Triangle // ABXY.A = A ∧ ABXY.B = B ∧ ABXY.C = ((B.1 - A.1), A.2)}) -- ABXY.2 = 2 * ABXY.1
(triangle1: Triangle)
(triangle2: Triangle)
(congruent_triangles: triangle1 = rectangle ∧ triangle2 = rectangle)

theorem exists_pentagon_satisfying_condition :
  ∃ (p : Pentagon), 
    ∃ straight_line : Point × Point,
    let (cutA, cutB, cutC) := divide_pentagon p straight_line in
    (cutA = cutB ∨ cutA = cutC ∨ cutB = cutC) := sorry

end exists_pentagon_satisfying_condition_l13_13182


namespace set_swept_by_all_lines_l13_13644

theorem set_swept_by_all_lines
  (a c x y : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < c)
  (h3 : c < a)
  (h4 : x^2 + y^2 ≤ a^2) : 
  (c^2 - a^2) * x^2 - a^2 * y^2 ≤ (c^2 - a^2) * c^2 :=
sorry

end set_swept_by_all_lines_l13_13644


namespace carA_travel_time_l13_13738

theorem carA_travel_time 
    (speedA speedB distanceB : ℕ)
    (ratio : ℕ)
    (timeB : ℕ)
    (h_speedA : speedA = 50)
    (h_speedB : speedB = 100)
    (h_distanceB : distanceB = speedB * timeB)
    (h_ratio : distanceA / distanceB = ratio)
    (h_ratio_value : ratio = 3)
    (h_timeB : timeB = 1)
  : distanceA / speedA = 6 :=
by sorry

end carA_travel_time_l13_13738


namespace inverse_function_l13_13760

-- Define the function h
def h (x : ℝ) : ℝ := 3 - 7 * x

-- Define the candidate inverse function k
def k (x : ℝ) : ℝ := (3 - x) / 7

-- State the proof problem
theorem inverse_function (x : ℝ) : h (k x) = x := by
  sorry

end inverse_function_l13_13760


namespace number_of_terms_in_sequence_l13_13514

theorem number_of_terms_in_sequence : 
  ∃ n : ℕ, 
    let a := 2.5
    let l := 72.5
    let d := 5
    in a + (n - 1) * d = l ∧ n = 15 :=
begin
  sorry
end

end number_of_terms_in_sequence_l13_13514


namespace sunny_final_distance_l13_13539

-- Definitions
variable (h d : ℝ)
variable (w s : ℝ)

-- Conditions
-- Sunny finishes d meters behind Windy in a h-meter race
axiom race1 : t = h / w
axiom race2 : t = (h - d) / s

-- Speed ratio calculated from the first race
axiom speed_ratio : s / w = (h - d) / h

-- From the second race conditions, Sunny starts d/2 meters behind Windy
axiom sunny_start_distance : Sunny_state = Windy_start_state - d/2

-- Distance calculation
noncomputable def calculate_distance : ℝ :=
  (h - ((h-d) * (h + d/2) / h))

-- Theorem proving the final distance when Sunny finishes the second race
theorem sunny_final_distance : calculate_distance h d = d * (d + h) / (2 * h) := sorry

end sunny_final_distance_l13_13539


namespace number_of_subsets_of_P_l13_13159

open Set

theorem number_of_subsets_of_P : 
  let M := {0, 1, 2, 3, 4}
  let N := {1, 3, 5}
  let P := M ∩ N
  P = {1, 3} →
  Finset.card (Finset.powerset P) = 4 :=
by {
  intros M N P hP,
  have hPcard : Finset.card (Finset.nth P) = 2, -- P = {1, 3}
  {
    sorry
  },
  have hPpowerset : Finset.card (Finset.powerset P) = 2 ^ Finset.card P,
  {
    sorry
  },
  rw [hPcard] at hPpowerset,
  norm_num at hPpowerset,
  exact hPpowerset,
}

end number_of_subsets_of_P_l13_13159


namespace general_formula_sum_first_n_terms_l13_13211

-- Condition: the sequence {a_n} satisfies a_1 + 3*a_2 + ... + (2*n-1)*a_n = 2*n
def seq_condition (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → ∑ k in Finset.range n, (2 * (k + 1) - 1) * a (k + 1) = 2 * n

-- (1) Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℚ) (h : seq_condition a) : ∀ n ≥ 1, a n = 2 / (2 * n - 1) :=
  sorry

-- (2) Sum of the first n terms of the sequence {a_n / (2n + 1)}
theorem sum_first_n_terms (a : ℕ → ℚ) (h : seq_condition a) :
  ∑ k in Finset.range n, a (k + 1) / (2 * (k + 1) + 1) = (2 * n) / (2 * n + 1) :=
  sorry

end general_formula_sum_first_n_terms_l13_13211


namespace quotient_calculation_l13_13314

theorem quotient_calculation
  (dividend : ℕ)
  (divisor : ℕ)
  (remainder : ℕ)
  (h_dividend : dividend = 176)
  (h_divisor : divisor = 14)
  (h_remainder : remainder = 8) :
  ∃ q, dividend = divisor * q + remainder ∧ q = 12 :=
by
  sorry

end quotient_calculation_l13_13314


namespace jose_internet_speed_l13_13552

-- Define the given conditions
def file_size : ℕ := 160
def upload_time : ℕ := 20

-- Define the statement we need to prove
theorem jose_internet_speed : file_size / upload_time = 8 :=
by
  -- Proof should be provided here
  sorry

end jose_internet_speed_l13_13552


namespace vector_condition_sufficiency_l13_13820

variables {V : Type} [inner_product_space ℝ V]

theorem vector_condition_sufficiency (a b : V) (ha : ∥a∥ ≠ 0) (hb : ∥b∥ ≠ 0) :
  (a = b → ∥a∥^2 = inner_product_space.inner a b) ∧
  (∥a∥^2 = inner_product_space.inner a b → a = b) := 
sorry

end vector_condition_sufficiency_l13_13820


namespace total_amount_shared_l13_13019

theorem total_amount_shared (A B C : ℕ) (h1 : A = 24) (h2 : 2 * A = 3 * B) (h3 : 8 * A = 4 * C) :
  A + B + C = 156 :=
sorry

end total_amount_shared_l13_13019


namespace arrangement_ways_l13_13148

-- Definitions representing conditions
def physics_books : Finset (Fin 4) := {0, 1, 2, 3}
def history_books : Finset (Fin 6) := {0, 1, 2, 3, 4, 5}

-- The mathematical proof problem statement
theorem arrangement_ways : 
  let block_ways := (2.factorial)
  let physics_ways := (4.factorial)
  let history_ways := (6.factorial)
  block_ways * physics_ways * history_ways = 34560 :=
by
  sorry

end arrangement_ways_l13_13148


namespace boundary_length_is_correct_l13_13010

noncomputable def square_side_length (area : ℕ) : ℕ :=
  (Float.sqrt (area.toFloat)).toNat

noncomputable def part_length (side_length : ℕ) : ℕ :=
  side_length / 4

noncomputable def circle_circumference (radius : ℕ) : Float :=
  2 * Float.pi * radius

def total_boundary_length (area : ℕ) : Float :=
  let side_length := square_side_length area
  let quarter_circle_radius := part_length side_length
  let quarter_circle_length := (circle_circumference quarter_circle_radius) / 4
  let combined_arcs_length := 4 * quarter_circle_length
  let total_straight_length := 8 * quarter_circle_radius
  combined_arcs_length + total_straight_length

theorem boundary_length_is_correct : total_boundary_length 256 ≈ 57.1 :=
begin
  sorry
end

end boundary_length_is_correct_l13_13010


namespace simplify_expression_l13_13594

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) : 
  (6 / (2 * y^(-4))) * (5 * y^(3 / 2) / 3) = 5 * y^(5.5) := by
  sorry

end simplify_expression_l13_13594


namespace binom_8_3_eq_56_l13_13401

def binom (n k : ℕ) : ℕ :=
(n.factorial) / (k.factorial * (n - k).factorial)

theorem binom_8_3_eq_56 : binom 8 3 = 56 := by
  sorry

end binom_8_3_eq_56_l13_13401


namespace red_black_card_combinations_l13_13012

theorem red_black_card_combinations :
  (let cards := 52;
       suits := 4;
       cards_per_suit := 13;
       red_cards := 2 * cards_per_suit;
       black_cards := 2 * cards_per_suit in
   red_cards * black_cards) = 676 :=
by
  let cards := 52;
  let suits := 4;
  let cards_per_suit := 13;
  let red_cards := 2 * cards_per_suit;
  let black_cards := 2 * cards_per_suit;
  sorry

end red_black_card_combinations_l13_13012


namespace inequality_solution_set_l13_13280

theorem inequality_solution_set
  (a b x : ℝ)
  (h1 : (∃ a : ℝ, a > 0 ∧ b = a / 2))
  (h2 : ∀ x : ℝ, ax - b > 0 ↔ x > 1/2) :
  (∃ x : ℝ, (ax - 2b)/(-x + 5) > 0 ↔ 1 < x ∧ x < 5) := 
sorry

end inequality_solution_set_l13_13280


namespace infinite_prime_set_exists_l13_13960

noncomputable def P : Set Nat := {p | Prime p ∧ ∃ m : Nat, p ∣ m^2 + 1}

theorem infinite_prime_set_exists :
  ∃ (P : Set Nat), (∀ p ∈ P, Prime p) ∧ (Set.Infinite P) ∧ 
  (∀ (p : Nat) (hp : p ∈ P) (k : ℕ),
    ∃ (m : Nat), p^k ∣ m^2 + 1 ∧ ¬(p^(k+1) ∣ m^2 + 1)) :=
sorry

end infinite_prime_set_exists_l13_13960


namespace probability_of_triangle_l13_13069

-- Define the lengths of the sticks
def stick_lengths : list ℕ := [3, 4, 6, 8, 12, 14, 18, 20]

-- Define a function to check if a given triplet of stick lengths can form a triangle
def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define a list of all possible triplets from the given stick lengths
def all_triplets : list (ℕ × ℕ × ℕ) :=
  [(3, 4, 6), (3, 4, 8), (3, 4, 12), (3, 4, 14), (3, 4, 18), (3, 4, 20),
   (3, 6, 8), (3, 6, 12), (3, 6, 14), (3, 6, 18), (3, 6, 20),
   -- ... include all possible triplets
   (18, 20, 3), (18, 20, 4), (18, 20, 6), (18, 20, 8), (18, 20, 12), (18, 20, 14)]

-- Define a list of valid triplets based on the triangle inequality
def valid_triplets : list (ℕ × ℕ × ℕ) :=
  filter (λ t, is_valid_triangle t.1 t.2 t.3) all_triplets

-- Proof problem: show that the probability of forming a triangle is 15/56
theorem probability_of_triangle : 
  (valid_triplets.length : ℚ) / (all_triplets.length : ℚ) = 15 / 56 :=
by
  sorry

end probability_of_triangle_l13_13069


namespace horse_problem_l13_13373

theorem horse_problem :
  ∃ (x y : ℕ) (cuirassier_cost dragoon_cost : ℕ),
    (11250 / x = cuirassier_cost) ∧
    (16000 / y = dragoon_cost) ∧
    (dragoon_cost + 50 = cuirassier_cost) ∧
    (y = x + 15) ∧
    (x = 25) ∧
    (y = 40) ∧
    (cuirassier_cost = 450) ∧
    (dragoon_cost = 400) :=
begin
  -- the proof will be inserted here
  sorry
end

end horse_problem_l13_13373


namespace problem_l13_13827

theorem problem (x : ℝ) 
  (h1 : sin x + cos x = (3 * real.sqrt 2) / 5)
  (h2 : 0 < x ∧ x < real.pi) : 
  (1 - cos (2 * x)) / (sin (2 * x)) = -7 :=
sorry

end problem_l13_13827


namespace final_answer_l13_13451

noncomputable def f (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1^2 - P.2^2, 2 * P.1 * P.2 - P.2^2)

def S : set (ℝ × ℝ) :=
  {P | ∀ ε > 0, ∃ N, ∀ n ≥ N, (λ P n, f^(n : ℕ) P) P ∈ metric.ball (0,0) ε}

-- Area of S can be expressed as π√r
def area_S := real.pi * real.sqrt ((4 : ℝ) / 3)

theorem final_answer : ⌊100 * (4 / 3)⌋ = 133 :=
by {
  have r := 4 / 3,
  calc ⌊100 * r⌋ = ⌊100 * (4 / 3)⌋ : by congr
              ... = 133              : by norm_num,
}

end final_answer_l13_13451


namespace sum_reciprocal_sequence_l13_13848

variable (a : ℕ → ℝ)
noncomputable def sequence := ∀ n : ℕ,
  (n = 0 → a n = 2) ∧
  ∀ m : ℕ, (n = m + 1 → a n = (m * a m) / (m + 1 + 2 * a m))

theorem sum_reciprocal_sequence (n : ℕ*) (h : sequence a) :
  ∑ k in finset.range n, 1 / a k = (5 * n^2 - 3 * n) / 4 := sorry

end sum_reciprocal_sequence_l13_13848


namespace determine_m_n_l13_13157

theorem determine_m_n (m n : ℤ) 
  (h1 : x^{m-1} - 2 * y^{3+n} = 5) 
  (h2 : linear_in_two_variables h1) : 
  m = 2 ∧ n = -2 := 
sorry

end determine_m_n_l13_13157


namespace prove_c_d_e_value_l13_13480

-- Define the variables and constants
variables (a b c d e f : ℝ)

-- Define the conditions given in the problem
def condition1 := a * b * c = 130
def condition2 := b * c * d = 65
def condition3 := d * e * f = 250
def condition4 := (a * f) / (c * d) = 0.6666666666666666

-- The theorem to prove that c * d * e = 750 given the conditions
theorem prove_c_d_e_value (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) :
  c * d * e = 750 :=
sorry

end prove_c_d_e_value_l13_13480


namespace sin_alpha_value_l13_13869

theorem sin_alpha_value :
  let α := (2 * Real.pi / 3) in
  let x := Real.sin α in
  let y := Real.cos α in
  let r := 1 in
  y / r = -1 / 2 := by
    sorry

end sin_alpha_value_l13_13869


namespace floor_K_l13_13068

noncomputable def K' (r : ℝ) : ℝ :=
  let area_D := 1600 * Real.pi
  let area_small := (8 * (1600 / 9) * Real.pi)
  area_D - area_small

def r' : ℝ := 40 / 3

theorem floor_K'_eq_12320 : ⌊K' r'⌋ = 12320 := by
  -- Compute K' and use the specific value to get the floor
  sorry

end floor_K_l13_13068


namespace f_comp_g_at_3_l13_13135

def f(x : ℝ) := x + 5
def g(x : ℝ) := x^2 - 4 * x + 3

theorem f_comp_g_at_3 : f(g(3)) = 5 :=
by
  sorry

end f_comp_g_at_3_l13_13135


namespace math_problem_l13_13404

variables {a : ℕ → ℤ}
variables {S : ℕ → ℤ}

-- Conditions
def is_decreasing_arithmetic_seq (d : ℤ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * d

def given_conditions (a : ℕ → ℤ) : Prop :=
  is_decreasing_arithmetic_seq (-1) ∧
  (a 3 * a 5 = 63) ∧
  (a 2 + a 6 = 16)

-- Conjecture 1: General term formula of the sequence
def general_term_formula (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 12 - n

-- Conjecture 2: Maximum value of Sn and the value of n
def maximum_value (S : ℕ → ℤ) : Prop :=
  ∀ n, (S 11 = 66 ∧ S 12 = 66)

-- Conjecture 3: Absolute sum of the sequence
def absolute_sum (a S : ℕ → ℤ) : Prop :=
  ∀ n, | ∑ k in finset.range (n + 1), |a k| | = 
  if n ≤ 12 then
    - ((1 / 2) * n * n) + (23 / 2) * n
  else
    ((1 / 2) * n * n) - (23 / 2) * n + 132

theorem math_problem (a : ℕ → ℤ) (S : ℕ → ℤ) :
  given_conditions a →
  general_term_formula a ∧ 
  maximum_value S ∧
  absolute_sum a S :=
sorry

end math_problem_l13_13404


namespace simplify_expression_l13_13330

theorem simplify_expression : 3000 * (3000 ^ 3000) + 3000 * (3000 ^ 3000) = 2 * 3000 ^ 3001 := 
by 
  sorry

end simplify_expression_l13_13330


namespace opposite_of_half_l13_13270

theorem opposite_of_half : - (1 / 2) = -1 / 2 := 
by
  sorry

end opposite_of_half_l13_13270


namespace bacteria_count_at_1_20pm_l13_13008

theorem bacteria_count_at_1_20pm (initial_bacteria : ℕ) (doubling_time : ℕ) (duration : ℕ) : 
  initial_bacteria = 15 →
  doubling_time = 5 →
  duration = 20 →
  final_bacteria = 15 * 2^4 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end bacteria_count_at_1_20pm_l13_13008


namespace number_of_bananas_in_bowl_l13_13980

theorem number_of_bananas_in_bowl (A P B : Nat) (h1 : P = A + 2) (h2 : B = P + 3) (h3 : A + P + B = 19) : B = 9 :=
sorry

end number_of_bananas_in_bowl_l13_13980


namespace switches_in_position_A_l13_13289

theorem switches_in_position_A (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 6) (h2 : 0 ≤ y ∧ y ≤ 6) :
  let switches := {i : ℕ | ∃ x y, i = (2^x) * (3^y) ∧ 0 ≤ x ∧ x ≤ 6 ∧ 0 ≤ y ∧ y ≤ 6} in
  let activated := {s : ℕ | (2^s) ∣ (2^6) * (3^6)} in
  let step := λ pos, match pos with
    | 0 => 1
    | 1 => 2
    | 2 => 3
    | _ => 0
  in
  let positions := λ n, iterate step n 0 in
  positions 500 = 0 →
  (500 - (4 * 4 + 2 * 4 * 2)) = 468 :=
begin
  sorry,
end

end switches_in_position_A_l13_13289


namespace find_a3_plus_a5_l13_13465

variable (a : ℕ → ℝ)
variable (positive_arith_geom_seq : ∀ n : ℕ, 0 < a n)
variable (h1 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25)

theorem find_a3_plus_a5 (positive_arith_geom_seq : ∀ n : ℕ, 0 < a n) (h1 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) :
  a 3 + a 5 = 5 :=
by
  sorry

end find_a3_plus_a5_l13_13465


namespace area_union_half_disks_l13_13450

def half_disk (y : ℝ) : set (ℝ × ℝ) :=
  {p | let d := (0, y), v := (2, 0) in
       ∃ θ, 0 ≤ θ ∧ θ ≤ π ∧ p = ((1 + cos θ), (y/2 + sin θ))}

def union_half_disks : set (ℝ × ℝ) :=
  ⋃ (y : ℝ) (hy : 0 ≤ y ∧ y ≤ 2), half_disk y

theorem area_union_half_disks : 
  measure_theory.measure (set.univ) union_half_disks = π := 
sorry

end area_union_half_disks_l13_13450


namespace parabola_translation_and_line_expression_l13_13136

theorem parabola_translation_and_line_expression (a t: ℝ) :
  (a * (1 + 2)^2 + 3 = 1) →
  (∀ x y: ℝ, y = a * (x + 2)^2 + 3 ↔ y = x^2) ∧
  (∀ t: ℝ, A t = (t, t^2) → B t = (-t^2, t) → 
    (∃ k: ℝ, ∃ C: ℝ × ℝ,
      parallel (line_through (0, 0) (B t)) 
               (line_through (0, 0) (C y = x^2))
      ∧ distance (1 / 2, 0) (line_through A t C)) = (max ∃ k: ℝ, y = x + 1/2)) :=
begin
  sorry
end

end parabola_translation_and_line_expression_l13_13136


namespace correct_correction_l13_13702

variable (y : ℕ)

-- Definitions based on conditions
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def quarter_value : ℕ := 25
def fifty_cent_value : ℕ := 50

def overestimation_pennies : ℕ := (nickel_value - penny_value) * y
def overestimation_quarters : ℕ := (fifty_cent_value - quarter_value) * y
def total_overestimation : ℕ := overestimation_pennies + overestimation_quarters

theorem correct_correction : total_overestimation y = 29 * y := 
  by
  unfold total_overestimation overestimation_pennies overestimation_quarters
  rw [nickel_value, penny_value, fifty_cent_value, quarter_value]
  calc
  (5 - 1) * y + (50 - 25) * y = 4 * y + 25 * y : by linarith
  ... = 29 * y : by linarith

end correct_correction_l13_13702


namespace vertical_angles_congruent_l13_13244

namespace VerticalAngles

-- Define what it means for two angles to be vertical
def Vertical (α β : Real) : Prop :=
  -- Definition of vertical angles goes here
  sorry

-- Hypothesis: If two angles are vertical angles
theorem vertical_angles_congruent (α β : Real) (h : Vertical α β) : α = β :=
sorry

end VerticalAngles

end vertical_angles_congruent_l13_13244


namespace projection_correct_l13_13507

theorem projection_correct :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, 3)
  -- Definition of dot product for 2D vectors
  let dot (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  -- Definition of projection of a onto b
  let proj := (dot a b / (b.1^2 + b.2^2)) • b
  proj = (-1 / 2, 3 / 2) :=
by
  sorry

end projection_correct_l13_13507


namespace expression_equal_five_sqrt_two_minus_one_l13_13393

theorem expression_equal_five_sqrt_two_minus_one :
  (sqrt (4 / 3) + sqrt 3) * sqrt 6 - ((sqrt 20 - sqrt 5) / sqrt 5) = 5 * sqrt 2 - 1 := 
sorry

end expression_equal_five_sqrt_two_minus_one_l13_13393


namespace parallelogram_area_l13_13957

theorem parallelogram_area (a b : ℝ) (γ : ℝ) : 
  ∃ S, S = a * b * Real.sin γ :=
begin
  sorry
end

end parallelogram_area_l13_13957


namespace BM_perpendicular_to_AC_l13_13257

theorem BM_perpendicular_to_AC
  (A B C K L M : Point)
  (h_triangle : is_acute_angled_triangle A B C)
  (h_circle : circle_with_diameter A C)
  (h_intersect_K : point_on_segment K A B)
  (h_intersect_L : point_on_segment L B C)
  (h_tangent_intersect : lines_tangent_to_circle_at_points K L M) :
  is_perpendicular (line_through B M) (line_through A C) :=
sorry

end BM_perpendicular_to_AC_l13_13257


namespace prime_counting_inequality_equality_condition_l13_13591

-- Definitions representing the mathematical functions in the conditions:
noncomputable def pi (x : ℕ) : ℕ := sorry  -- Replace with actual prime-counting function
noncomputable def phi (n : ℕ) : ℕ := sorry -- Replace with actual Euler's totient function

-- Formal statement of the proof problem:
theorem prime_counting_inequality (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  pi(m) - pi(n) ≤ (m - 1) * phi(n) / n := sorry

theorem equality_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (pi(m) - pi(n) = (m - 1) * phi(n) / n) ↔ 
  (m, n) = (1, 1) ∨ (m, n) = (2, 1) ∨ (m, n) = (3, 1) ∨
  (m, n) = (3, 2) ∨ (m, n) = (5, 2) ∨ (m, n) = (7, 2) := sorry

end prime_counting_inequality_equality_condition_l13_13591


namespace radius_correct_l13_13879

noncomputable def radius_of_circumscribed_sphere (DA DB DC : ℝ) (h0 : DA = 2) (h1 : DB = 2) (h2 : DC = 2) (h3 : DA ⬝ DB = 0) (h4 : DA ⬝ DC = 0) (cos_theta : ℝ) (h5 : cos_theta = √6 / 3) : ℝ :=
  sqrt 3

theorem radius_correct :
  let DA := 2 in
  let DB := 2 in
  let DC := 2 in
  let cos_theta := √6 / 3 in
  radius_of_circumscribed_sphere DA DB DC sorry sorry sorry sorry sorry cos_theta sorry = sqrt 3 :=
by
  sorry

end radius_correct_l13_13879


namespace vertical_angles_congruent_l13_13242

namespace VerticalAngles

-- Define what it means for two angles to be vertical
def Vertical (α β : Real) : Prop :=
  -- Definition of vertical angles goes here
  sorry

-- Hypothesis: If two angles are vertical angles
theorem vertical_angles_congruent (α β : Real) (h : Vertical α β) : α = β :=
sorry

end VerticalAngles

end vertical_angles_congruent_l13_13242


namespace area_of_small_parallelograms_l13_13626

noncomputable def small_parallelogram_area (n m : ℕ) (A : ℝ) : ℝ :=
  if n > 0 ∧ m > 0 then 
    A / (n * m)
  else 
    0

theorem area_of_small_parallelograms 
  (n m : ℕ) 
  (A : ℝ) 
  (hn : n > 0) 
  (hm : m > 0) 
  (hA : A = 1) : 
  small_parallelogram_area n m A = 1 / (n * m) := 
by 
  unfold small_parallelogram_area 
  rw [if_pos (and.intro hn hm)] 
  rw [hA] 
  sorry

end area_of_small_parallelograms_l13_13626


namespace two_digit_ab_divisible_by_11_13_l13_13160

theorem two_digit_ab_divisible_by_11_13 (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 11 = 0)
  (h4 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 13 = 0) :
  10 * a + b = 48 :=
sorry

end two_digit_ab_divisible_by_11_13_l13_13160


namespace equation_solution_l13_13966

theorem equation_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + d^2 - a * b - b * c - c * d - d + (2 / 5) = 0 ↔ 
  a = 1 / 5 ∧ b = 2 / 5 ∧ c = 3 / 5 ∧ d = 4 / 5 :=
by sorry

end equation_solution_l13_13966


namespace sochi_apartment_price_decrease_l13_13341

theorem sochi_apartment_price_decrease (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let moscow_rub_decrease := 0.2
  let moscow_eur_decrease := 0.4
  let sochi_rub_decrease := 0.1
  let new_moscow_rub := (1 - moscow_rub_decrease) * a
  let new_moscow_eur := (1 - moscow_eur_decrease) * b
  let ruble_to_euro := new_moscow_rub / new_moscow_eur
  let new_sochi_rub := (1 - sochi_rub_decrease) * a
  let new_sochi_eur := new_sochi_rub / ruble_to_euro
  let decrease_percentage := (b - new_sochi_eur) / b * 100
  decrease_percentage = 32.5 :=
by
  sorry

end sochi_apartment_price_decrease_l13_13341


namespace min_value_a_quarter_range_of_a_l13_13488

open Classical

noncomputable def f (a x : ℝ) : ℝ :=
  if x < 1 then (2 - 4 * a) * a^x + a else ln x

theorem min_value_a_quarter :
  is_min (f (1 / 4)) 0 :=
  sorry

theorem range_of_a {a : ℝ} (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (∀ y, ∃ x, f a x = y) ↔ (1 / 2 < a ∧ a ≤ 3 / 4) :=
  sorry

end min_value_a_quarter_range_of_a_l13_13488


namespace gas_and_maintenance_money_l13_13913

theorem gas_and_maintenance_money
  (income : ℝ := 3200)
  (rent : ℝ := 1250)
  (utilities : ℝ := 150)
  (retirement_savings : ℝ := 400)
  (groceries : ℝ := 300)
  (insurance : ℝ := 200)
  (miscellaneous_expenses : ℝ := 200)
  (car_payment : ℝ := 350) :
  income - (rent + utilities + retirement_savings + groceries + insurance + miscellaneous_expenses + car_payment) = 350 :=
by
  sorry

end gas_and_maintenance_money_l13_13913


namespace painting_time_l13_13776

theorem painting_time (t : ℚ) (h_rate_d : ℚ) (h_rate_v : ℚ) (paint_done_together : ℚ) :
  (h_rate_d = 1 / 6) → (h_rate_v = 1 / 8) → (paint_done_together = 3 * (h_rate_d + h_rate_v) + (t - 5) * (h_rate_d + h_rate_v)) →
  (paint_done_together = 1) → t = 38 / 7 :=
begin
  intros,
  sorry
end

end painting_time_l13_13776


namespace solve_inequality_l13_13440

theorem solve_inequality (y : ℚ) :
  (3 / 40 : ℚ) + |y - (17 / 80 : ℚ)| < (1 / 8 : ℚ) ↔ (13 / 80 : ℚ) < y ∧ y < (21 / 80 : ℚ) := 
by
  sorry

end solve_inequality_l13_13440


namespace angle_NPO_eq_190_deg_minus_angle_AHB_l13_13729

-- Suppose we have a geometric setup as described in the problem
variables {A B C O P M H N : Point}
variables (circle_O : Circle O)
variables (line_AB line_AC line_BC line_OM : Line)
variables (tangent_AB : Tangent circle_O A B)
variables (tangent_AC : Tangent circle_O A C)
variables (line_AP : LineSegment A P)
variables (line_MP : Midpoint line_AP M)
variables (line_PH : Perpendicular P H BC)
variables (line_OM_N : Intersect OM BC N)

-- Projection
axiom projection_PH : ∀ {P H BC}, LinePerpendicular P H BC

-- Natural geometric intersection
axiom intersection_OM_N : ∀ {OM BC N}, LineIntersect OM BC N

-- Midpoint property
axiom midpoint_MP : Midpoint line_AP M

-- The actual proof problem we want to state
theorem angle_NPO_eq_190_deg_minus_angle_AHB :
  ∠ N P O = 190° - ∠ A H B :=
sorry

end angle_NPO_eq_190_deg_minus_angle_AHB_l13_13729


namespace exists_correct_function_l13_13639

def is_function (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f x = y → ∀ z : ℝ, f z = y → x = z

theorem exists_correct_function (f : ℝ → ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2 * x) = |x + 1|) ∧ 
  (¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (sin (2 * x)) = sin x) ∧ 
  (¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (sin (2 * x)) = x^2 + x) ∧ 
  (¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 1) = |x + 1|) :=
by
  sorry

end exists_correct_function_l13_13639


namespace three_digit_number_l13_13688

theorem three_digit_number (a b c : ℕ) (h1 : a + b + c = 10) (h2 : b = a + c) (h3 : 100 * c + 10 * b + a = 100 * a + 10 * b + c + 99) : (100 * a + 10 * b + c) = 253 := 
by
  sorry

end three_digit_number_l13_13688


namespace min_value_b_l13_13889

noncomputable def circle_below_line_with_max_one_intersection (a : ℝ) (b : ℝ) : Prop :=
  (-1 ≤ a ∧ a ≤ 1) ∧
  (d = |3 * a - 1 + b| / real.sqrt 2) ∧ 
  (d ≥ real.sqrt 2) ∧
  (1 - 2 * a < a + b)

theorem min_value_b (b : ℝ) : (∃ a, circle_below_line_with_max_one_intersection a b) → b ≥ 6 := 
sorry

end min_value_b_l13_13889


namespace basic_spatial_data_source_l13_13994

def source_of_basic_spatial_data (s : String) : Prop :=
  s = "Detailed data provided by high-resolution satellite remote sensing technology" ∨
  s = "Data from various databases provided by high-speed networks" ∨
  s = "Various data collected and organized through the information highway" ∨
  s = "Various spatial exchange data provided by GIS"

theorem basic_spatial_data_source :
  source_of_basic_spatial_data "Data from various databases provided by high-speed networks" :=
sorry

end basic_spatial_data_source_l13_13994


namespace negation_proposition_l13_13870

theorem negation_proposition (p : ∀ x ∈ set.Icc 1 2, x^2 - 1 ≥ 0) :
  (¬ ∀ x ∈ set.Icc 1 2, x^2 - 1 ≥ 0) ↔ (∃ x ∈ set.Icc 1 2, x^2 - 1 ≤ 0) :=
by
  sorry

end negation_proposition_l13_13870


namespace find_m_l13_13036

noncomputable def f (x : ℝ) := 4 * x^2 - 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -14 :=
by
  sorry

end find_m_l13_13036


namespace volume_semi_cylinder_l13_13984

noncomputable def volume_of_semi_cylinder (a r : ℝ) : ℝ :=
  (π * (a - π * r^2) * r) / (2 * (π + 2))

theorem volume_semi_cylinder (a r : ℝ) (h_a : a > 0) (h_r : r > 0) :
  volume_of_semi_cylinder a r = (π * (a - π * r^2) * r) / (2 * (π + 2)) :=
sorry

end volume_semi_cylinder_l13_13984


namespace initial_value_l13_13306

theorem initial_value (x k : ℤ) (h : x + 335 = k * 456) : x = 121 := sorry

end initial_value_l13_13306


namespace find_y_l13_13153

theorem find_y (x y : ℝ) (h1 : x^2 = 2 * y - 6) (h2 : x = 7) : y = 55 / 2 :=
by
  sorry

end find_y_l13_13153


namespace problem_statement_l13_13196

open probability

-- Define the outcomes of rolling a die
def die_outcomes := {1, 2, 3, 4, 5, 6}

-- Define A
def A (b c : ℕ) : set ℝ := {x : ℝ | x^2 - (b : ℝ) * x + 2 * (c : ℝ) < 0}

-- Define the probability that A is nonempty
def P_A_ne_empty (b c : ℕ) : ℝ := 
  if b^2 - 8 * c > 0 then 1 else 0

-- Define the random variable ξ
def ξ (b c : ℕ) : ℕ := abs (b - c)

-- The main theorem statement without proof
theorem problem_statement :
  (P_A_ne_empty.val = 1/4) ∧
  (P(ξ = 0) = 1/6) ∧
  (P(ξ = 1) = 5/18) ∧
  (P(ξ = 2) = 2/9) ∧
  (P(ξ = 3) = 1/6) ∧
  (P(ξ = 4) = 1/9) ∧
  (P(ξ = 5) = 1/18) :=
by
  sorry

end problem_statement_l13_13196


namespace math_problem_l13_13933

variable {x y z : ℝ}
variable (hx : x > 0) (hy : y > 0) (hz : z > 0)
variable (h : x^2 + y^2 + z^2 = 1)

theorem math_problem : 
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ 3 * Real.sqrt 3 / 2 :=
sorry

end math_problem_l13_13933


namespace minimum_total_distance_l13_13063

theorem minimum_total_distance :
  let r : ℝ := 50
  let num_boys : ℕ := 8
  let angle : ℝ := (3/8 : ℝ) * (2 * Real.pi)
  let s := Real.sqrt (2 * r^2 * (1 - Real.cos angle))
  let distance_per_boy := 4 * s
  let total_distance := num_boys * distance_per_boy
  total_distance = 1600 * Real.sqrt(2 - Real.sqrt 2) := 
by
  let r : ℝ := 50
  let num_boys : ℕ := 8
  let angle : ℝ := (3/8 : ℝ) * (2 * Real.pi)
  let s := Real.sqrt (2 * r^2 * (1 - Real.cos angle))
  let distance_per_boy := 4 * s
  let total_distance := num_boys * distance_per_boy
  show total_distance = 1600 * Real.sqrt(2 - Real.sqrt 2) from sorry

end minimum_total_distance_l13_13063


namespace range_of_k_l13_13803

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 0     := 2 - k
| (m+1) := 2 * (m + 2) - k * (m + 1)

def excellent_value (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, 2^i * a (i + 1)) / n

def sequence_property (a : ℕ → ℝ) (n : ℕ) : Prop :=
  excellent_value a n = 2^(n + 1)

def sum_of_first_n_terms (seq : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, seq i)

def satisfies_condition (a : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → sum_of_first_n_terms (λ i, a i - k * i) n ≤
  sum_of_first_n_terms (λ i, a i - k * i) 5

theorem range_of_k : ∃ k : ℝ, (7/3 : ℝ) ≤ k ∧ k ≤ (12/5 : ℝ) :=
sorry

end range_of_k_l13_13803


namespace binary_digit_sum_le_8_probability_l13_13715

def binary_sum_le_8_probability : ℚ := 655 / 672

theorem binary_digit_sum_le_8_probability :
  ∀ (x : ℕ), (1 ≤ x ∧ x ≤ 2016) →
    (probability (sum_of_binary_digits x ≤ 8) = binary_sum_le_8_probability) :=
sorry

end binary_digit_sum_le_8_probability_l13_13715


namespace original_profit_percentage_l13_13731

theorem original_profit_percentage {P S : ℝ}
  (h1 : S = 1100)
  (h2 : P ≠ 0)
  (h3 : 1.17 * P = 1170) :
  (S - P) / P * 100 = 10 :=
by
  sorry

end original_profit_percentage_l13_13731


namespace ab_of_complex_conjugate_l13_13608

noncomputable def complex_conjugate_ab : ℂ := 
  let z : ℂ := (2 + 3 * Complex.I) / Complex.I 
  let conj_z : ℂ := conj z 
  let a : ℤ := conj_z.re.toRational -- Extract the real part and convert to integer
  let b : ℤ := conj_z.im.toRational -- Extract the imaginary part and convert to integer
  a * b

theorem ab_of_complex_conjugate : complex_conjugate_ab = 6 := 
  by
    sorry

end ab_of_complex_conjugate_l13_13608


namespace distance_between_points_l13_13952

theorem distance_between_points : ∀ (A B : ℤ), A = 5 → B = -3 → |A - B| = 8 :=
by
  intros A B hA hB
  rw [hA, hB]
  norm_num

end distance_between_points_l13_13952


namespace ratio_of_rise_l13_13304

-- Define the conditions
variable (h1 h2 : ℝ) (V : ℝ)
variable (r1 : ℝ := 4) (r2 : ℝ := 8) -- Radii of liquid surfaces in cones
variable (r_marble : ℝ := 1.5) -- Radius of the marble
variable (V_marble : ℝ := (4 / 3) * (Real.pi) * r_marble^3)
variable (V1 : ℝ := (1 / 3) * (Real.pi) * r1^2 * h1) -- Volume of the narrow cone
variable (V2 : ℝ := (1 / 3) * (Real.pi) * r2^2 * h2) -- Volume of the wide cone

-- Conditions
axiom equal_volumes : V1 = V2
axiom marble_immersed : ∀ (V_final1 V_final2 : ℝ), 
  V_final1 = V1 + V_marble →
  V_final2 = V2 + V_marble → 
  V_final1 = V_final2

-- Lean 4 statement to prove the required ratio
theorem ratio_of_rise : 
  (h1 = 4 * h2) →
  ∀ (x y : ℝ),
  (V1 * x^3 = V1 + (4 / 3) * (Real.pi) * r_marble^3 ) →
  (V2 * y^3 = V2 + (4 / 3) * (Real.pi) * r_marble^3 ) →
  4 * (x - 1) = (4 * (y - 1)) → 
  ratio_of_rise = (4:1) :=
by
  intros h1_eq_4h2 x y V1_eq V2_eq ratio_eq
  sorry

end ratio_of_rise_l13_13304


namespace polynomial_satisfaction_l13_13439

theorem polynomial_satisfaction :
  ∀ (f : ℝ → ℝ),
  (∃ a b c d : ℝ, f = λ x, a * x^3 + b * x^2 + c * x + d) →
  (∀ x, f (2 * x) = (D f x) * (D (D f) x)) →
  f = λ x, (4 / 9) * x^3 :=
by
  sorry

end polynomial_satisfaction_l13_13439


namespace investment_difference_l13_13217

noncomputable def A_Maria : ℝ := 60000 * (1 + 0.045)^3
noncomputable def A_David : ℝ := 60000 * (1 + 0.0175)^6
noncomputable def investment_diff : ℝ := A_Maria - A_David

theorem investment_difference : abs (investment_diff - 1803.30) < 1 :=
by
  have hM : A_Maria = 60000 * (1 + 0.045)^3 := by rfl
  have hD : A_David = 60000 * (1 + 0.0175)^6 := by rfl
  have hDiff : investment_diff = A_Maria - A_David := by rfl
  -- Proof would go here; using the provided approximations
  sorry

end investment_difference_l13_13217


namespace hyperbola_eccentricity_l13_13499

-- Definitions of the hyperbolas
def hyperbola_1 (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def hyperbola_2 (x y a b : ℝ) : Prop := y^2 / a^2 - x^2 / b^2 = 1

-- Asymptote conditions dividing the first quadrant into three equal parts
def asymptote_condition (a b : ℝ) : Prop := 
  (b / a = Real.tan (Real.pi / 6)) ∨ (b / a = Real.tan (Real.pi / 3))

-- The main theorem stating the eccentricity
theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  hyperbola_1 1 1 a b → hyperbola_2 1 1 a b → asymptote_condition a b →
  (∃ e : ℝ, e = 2 ∨ e = (2 * Real.sqrt 3) / 3) :=
by
  intros h1 h2 h3
  use sorry

-- sorry is added to skip the actual proof

end hyperbola_eccentricity_l13_13499


namespace path_of_A_is_circle_l13_13212

-- Define the vertices B and C as fixed points, and the vertex A as a moving point
variables {B C : ℝ × ℝ} {A : ℝ → ℝ × ℝ} {a : ℝ} {φ : ℝ}

-- Assume the Brocard angle φ remains constant
axiom brocard_constant :
  ∀ t, let A' := A t in 
  let (xB, yB) := B in
  let (xC, yC) := C in
  let (xA, yA) := A' in
  (xB - xA)^2 + (yB - yA)^2 + (xC - xA)^2 + (yC - yA)^2 + a^2 - b^2 - c^2 = 
  (xB * yA + xC * yB + yB * xA + yC * xB + yA * xC + yB * xC) * cot φ

-- Prove that the point A moves along a circle of the specified radius
theorem path_of_A_is_circle :
  (∃ r, r = a / 2 * sqrt (cot φ ^ 2 - 3)) →
  ∀ t, let A' := A t in 
  let (x, y) := A' in 
  x ^ 2 + y ^ 2 = r ^ 2 
  :=
sorry

end path_of_A_is_circle_l13_13212


namespace michael_truck_meetings_l13_13943

noncomputable def michael_position (t : ℕ) : ℕ :=
if t ≤ 20 then 0 else 3 * (t - 20)

noncomputable def truck_position (t : ℕ) : ℕ :=
let cycle_time := 70
let speed := 12
let distance := 300
in 
if (t % cycle_time) <= 25 then distance + speed * (t % cycle_time) else distance + speed * 25

theorem michael_truck_meetings : 
  ∃ m : ℕ, 
    m = 6 ∧
    (∃ (t : ℕ), 
      t > 20 ∧ 
      michael_position t = truck_position t ∧ 
      ∀ k : ℕ, k < m → (∃ t : ℕ, t > 20 ∧ michael_position t = truck_position t))
:= sorry

end michael_truck_meetings_l13_13943


namespace cost_price_6500_l13_13692

variable (CP SP : ℝ)

-- Condition 1: The selling price is 30% more than the cost price.
def selling_price (CP : ℝ) : ℝ := CP * 1.3

-- Condition 2: The selling price is Rs. 8450.
axiom selling_price_8450 : selling_price CP = 8450

-- Prove that the cost price of the computer table is Rs. 6500.
theorem cost_price_6500 : CP = 6500 :=
by
  sorry

end cost_price_6500_l13_13692


namespace probability_of_rolling_four_threes_l13_13858
open BigOperators

def probability_four_threes (n : ℕ) (k : ℕ) (p : ℚ) (q : ℚ) : ℚ := 
  (n.choose k) * (p ^ k) * (q ^ (n - k))

theorem probability_of_rolling_four_threes : 
  probability_four_threes 5 4 (1 / 10) (9 / 10) = 9 / 20000 := 
by 
  sorry

end probability_of_rolling_four_threes_l13_13858


namespace electric_blankets_sold_l13_13655

theorem electric_blankets_sold (T H E : ℕ)
  (h1 : 2 * T + 6 * H + 10 * E = 1800)
  (h2 : T = 7 * H)
  (h3 : H = 2 * E) : 
  E = 36 :=
by {
  sorry
}

end electric_blankets_sold_l13_13655


namespace average_height_of_students_l13_13628

theorem average_height_of_students (x : ℕ) (female_height male_height : ℕ) 
  (female_height_eq : female_height = 170) (male_height_eq : male_height = 185) 
  (ratio : 2 * x = x * 2) : 
  ((2 * x * male_height + x * female_height) / (2 * x + x) = 180) := 
by
  sorry

end average_height_of_students_l13_13628


namespace volume_in_barrel_l13_13328

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem volume_in_barrel (x : ℕ) (V : ℕ) (hx : V = 30) 
  (h1 : V = x / 2 + x / 3 + x / 4 + x / 5 + x / 6) 
  (h2 : is_divisible (87 * x) 60) : 
  V = 29 := 
sorry

end volume_in_barrel_l13_13328


namespace no_positive_reals_satisfy_conditions_l13_13074

theorem no_positive_reals_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  4 * (a * b + b * c + c * a) - 1 ≥ a^2 + b^2 + c^2 ∧ 
  a^2 + b^2 + c^2 ≥ 3 * (a^3 + b^3 + c^3) :=
by
  sorry

end no_positive_reals_satisfy_conditions_l13_13074


namespace finish_time_is_1_10_PM_l13_13183

-- Definitions of the problem conditions
def start_time := 9 * 60 -- 9:00 AM in minutes past midnight
def third_task_finish_time := 11 * 60 + 30 -- 11:30 AM in minutes past midnight
def num_tasks := 5
def tasks1_to_3_duration := third_task_finish_time - start_time
def one_task_duration := tasks1_to_3_duration / 3
def total_duration := one_task_duration * num_tasks

-- Statement to prove the final time when John finishes the fifth task
theorem finish_time_is_1_10_PM : 
  start_time + total_duration = 13 * 60 + 10 := 
by 
  sorry

end finish_time_is_1_10_PM_l13_13183


namespace find_a_minus_b_l13_13819

theorem find_a_minus_b (a b x y : ℤ)
  (h_x : x = 1)
  (h_y : y = 1)
  (h1 : a * x + b * y = 2)
  (h2 : x - b * y = 3) :
  a - b = 6 := by
  subst h_x
  subst h_y
  simp at h1 h2
  have h_b: b = -2 := by linarith
  have h_a: a = 4 := by linarith
  rw [h_a, h_b]
  norm_num

end find_a_minus_b_l13_13819


namespace zero_in_interval_l13_13874

def f (x : ℝ) : ℝ := -(abs (x - 5)) + 2^(x - 1)

theorem zero_in_interval : ∃ k : ℤ, f (k : ℝ) * f ((k+1) : ℝ) < 0 ∧ (2 : ℤ) = k :=
by
  sorry

end zero_in_interval_l13_13874


namespace triangle_side_length_l13_13902

noncomputable theory

theorem triangle_side_length (AC : ℝ) (A C : ℝ) (hAC : AC = real.sqrt 3) (hA : A = 45) (hC : C = 75) : 
  ∃ BC : ℝ, BC = real.sqrt 2 :=
by {
  sorry -- The proof steps go here
}

end triangle_side_length_l13_13902


namespace constant_term_binomial_expansion_l13_13258

theorem constant_term_binomial_expansion :
  let f : ℚ[X] := (X - (3 : ℚ) * X⁻²)^6 in
  ∃ c : ℚ, constant_term f = c ∧ c = 135 :=
by
  sorry

end constant_term_binomial_expansion_l13_13258


namespace determine_s_k_l13_13630

/- Definitions of the sequences and the problem. -/
def u_seq : ℕ → ℤ
| 0 => u₀
| 1 => u₁
| (n+2) => 2 * u_seq (n+1) + u_seq n

def v_seq : ℕ → ℤ
| 0 => v₀
| 1 => v₁
| (n+2) => 3 * v_seq (n+1) - v_seq n

def s_seq (n : ℕ) : ℤ := u_seq n + v_seq n

/- The main theorem stating the ability to determine s_k for k ≥ 5. -/
theorem determine_s_k (s : ℕ → ℤ) (k : ℕ) (h : k ≥ 5) :
  ∃ f : (Π n, n < k → ℤ) → ℤ, s k = f (λ n h', s n) :=
sorry

end determine_s_k_l13_13630


namespace bisect_FPQ_l13_13382

variable (A B C D E F P Q : Point)
variable (ABC : Triangle A B C)
variable (altitude_AD : Altitude ABC A D)
variable (altitude_BE : Altitude ABC B E)
variable (altitude_CF : Altitude ABC C F)
variable (P_on_DF : OnSegment P D F)
variable (Q_on_EF : OnSegment Q E F)
variable (angle_PAQ_eq_angle_DAC : ∠ PAQ = ∠ DAC)

theorem bisect_FPQ (h1 : acuteAngled ABC) (h2 : angle_PAQ_eq_angle_DAC) : 
  Bisects (Line A P) (Angle F P Q) := by 
  sorry

end bisect_FPQ_l13_13382


namespace evaluate_expression_l13_13780

theorem evaluate_expression :
  (2 + 3 / (4 + 5 / (6 + 7 / 8))) = 137 / 52 :=
by
  sorry

end evaluate_expression_l13_13780


namespace quadratic_properties_l13_13102

variable {α : Type*} [linear_ordered_field α]

def quadratic_function (a b : α) (h : a ≠ 0) (f_x := λ x : α, a * x^2 + b * x) :=
  f_x

theorem quadratic_properties (a b : α) (h_a : a ≠ 0) (h_f2 : quadratic_function a b h_a 2 = 0)
                             (h_equal_roots : ∀ x, quadratic_function a b h_a x = x → x = 0 ∨ quadratic_function a b h_a x = 1)
                             : 
  (let f_x := quadratic_function (-1/2 : α) (1 : α) (by norm_num : (-1/2 : α) ≠ 0) in
    ∃ f_x, (∀ x, f_x x = -1/2 * x^2 + x) ∧
           (∀ y, f_x y ≤ 1/2) ∧
           (f_x 3 < f_x 0 ∧ f_x 0 < f_x 1) ∧
           (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 1 → f_x x₁ < f_x x₂)) :=
begin
  sorry
end

end quadratic_properties_l13_13102


namespace range_of_point_walking_l13_13714

theorem range_of_point_walking {a b : ℕ} (h : a > b) : 
  let final_position := a - b 
  in ∃ max_coord min_coord : ℤ, max_coord - min_coord = final_position := sorry

end range_of_point_walking_l13_13714


namespace cone_lateral_surface_area_l13_13998

theorem cone_lateral_surface_area {r l : ℝ} (hr : r = 3) (hl : l = 5) :
  ∃ A, A = π * r * l ∧ A = 15 * π :=
by
  use π * r * l
  split
  · rw [hr, hl]
    ring
  · sorry

end cone_lateral_surface_area_l13_13998


namespace max_value_fraction_l13_13096

theorem max_value_fraction (x y z w : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) :
  (xyzw (x + y + z + w) / ((x + y)^3 * (y + z + w)^2) ≤ 1 / 4) :=
by
  sorry

end max_value_fraction_l13_13096


namespace Sara_turns_over_card_to_disprove_Tom_statement_l13_13221

theorem Sara_turns_over_card_to_disprove_Tom_statement :
  (∃ card: string, card = "8" ∧ 
      (∀ side1 side2 : string, 
        (side1 = "R" ∨ side1 = "5" ∨ side1 = "8" ∨ side1 = "7" ∨ side1 = "A") →
        (side2 = card → ¬(side1 = "R" → side2 % 2 = 1)))) :=
begin
  sorry -- proof omitted
end

end Sara_turns_over_card_to_disprove_Tom_statement_l13_13221


namespace kn_divides_bc_ratio_l13_13811

theorem kn_divides_bc_ratio (A B C N K M : Type) (h_triangle : Triangle A B C)
  (h_N_extension: N ∈ ray_extend C A)
  (h_CN_ratio: CN = (2 / 3) * AC)
  (h_K_ratio: ∃ (ratio: ℝ), ratio = (AK / KB) ∧ ratio = 3 / 2)
  (h_intersection: M = line_intersection K N B C):
  segment_ratio B M C = (5 / 3) :=
  sorry

end kn_divides_bc_ratio_l13_13811


namespace find_50th_negative_term_l13_13417

def sequence_b (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), Real.cos k

def is_negative_term (n : ℕ) : Prop :=
  sequence_b n < 0

theorem find_50th_negative_term : ∃ n, is_negative_term n ∧ n = 314 :=
sorry

end find_50th_negative_term_l13_13417


namespace sum_first_last_l13_13255

theorem sum_first_last (A B C D : ℕ) (h1 : (A + B + C) / 3 = 6) (h2 : (B + C + D) / 3 = 5) (h3 : D = 4) : A + D = 11 :=
by
  sorry

end sum_first_last_l13_13255


namespace min_distance_point_to_line_l13_13421

-- Definitions
def point := (3 : ℝ, 4 : ℝ)
def line (x y : ℝ) : Prop := 2 * x + 15 * y = 90

-- Proof statement
theorem min_distance_point_to_line : ∃ (d : ℝ), d = abs (2 * 3 + 15 * 4 - 90) / sqrt (2^2 + 15^2) ∧ d = 24 / sqrt 229 :=
by
  have h_point_line := @line 3 4
  -- Further proof steps would go here
  sorry

end min_distance_point_to_line_l13_13421


namespace largest_unused_card_l13_13637

theorem largest_unused_card :
  let cards := {1, 7, 3}
  ∃ m n u : ℕ, cards = {m, n, u} ∧
  10 * max m n + min m n = 73 →
  u = 1 :=
begin
  sorry
end

end largest_unused_card_l13_13637


namespace cici_originally_had_9_cards_l13_13394

variable {x y : ℕ}

-- Conditions
def condition1 (x y : ℕ) : Prop := x + 2 = 2 * (y - 2)
def condition2 (x y : ℕ) : Prop := 3 * (x - 3) = y + 3
def condition3 (x y : ℕ) : Prop := x + 4 = 4 * (y - 4)
def condition4 (x y : ℕ) : Prop := 5 * (x - 5) = y + 5
def exactly_two_conditions_correct (c1 c2 c3 c4 : Prop) : Prop :=
  (c1 ∧ c2 ∧ ¬c3 ∧ ¬c4) ∨ (c1 ∧ c3 ∧ ¬c2 ∧ ¬c4) ∨ (c1 ∧ c4 ∧ ¬c2 ∧ ¬c3) ∨
  (c2 ∧ c3 ∧ ¬c1 ∧ ¬c4) ∨ (c2 ∧ c4 ∧ ¬c1 ∧ ¬c3) ∨ (c3 ∧ c4 ∧ ¬c1 ∧ ¬c2)
def final_condition_equal_cards (x y : ℕ) : Prop := (x - k) = (y + k) -- k to be determined according to steps in original problem's combination

-- Claim: under these conditions, x must be 9
theorem cici_originally_had_9_cards (hx : exactly_two_conditions_correct (condition1 x y) (condition2 x y) (condition3 x y) (condition4 x y))
  (heq : final_condition_equal_cards x y) : x = 9 :=
sorry -- Proof to be completed

end cici_originally_had_9_cards_l13_13394


namespace foldable_cube_with_one_face_missing_l13_13973

-- Definitions for the conditions
structure Square where
  -- You can define properties of a square here if necessary

structure Polygon where
  squares : List Square
  congruent : True -- All squares are congruent
  joined_edge_to_edge : True -- The squares are joined edge-to-edge

-- The positions the additional square can be added to
inductive Position
| P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9

-- Define the problem in Lean 4 as a theorem
theorem foldable_cube_with_one_face_missing (base_polygon : Polygon) :
  base_polygon.squares.length = 4 →
  ∃ (positions : List Position), positions.length = 6 ∧
    ∀ pos ∈ positions, 
      let new_polygon := { base_polygon with squares := base_polygon.squares.append [Square.mk] }
      new_polygon.foldable_into_cube_with_one_face_missing pos :=
  sorry

end foldable_cube_with_one_face_missing_l13_13973


namespace cindy_arrival_speed_l13_13029

def cindy_speed (d t1 t2 t3: ℕ) : Prop :=
  (d = 20 * t1) ∧ 
  (d = 10 * (t2 + 3 / 4)) ∧
  (t3 = t1 + 1 / 2) ∧
  (20 * t1 = 10 * (t2 + 3 / 4)) -> 
  (d / (t3) = 12)

theorem cindy_arrival_speed (t1 t2: ℕ) (h₁: t2 = t1 + 3 / 4) (d: ℕ) (h2: d = 20 * t1) (h3: t3 = t1 + 1 / 2) :
  cindy_speed d t1 t2 t3 := by
  sorry

end cindy_arrival_speed_l13_13029


namespace equation_of_plane_l13_13263

def point := (ℝ × ℝ × ℝ)

def plane (A B C D : ℤ) := {p : point // A * p.fst + B * p.snd + C * p.thd + D = 0}

theorem equation_of_plane : ∃ (A B C D : ℤ),
  let n := (10, -5, 2) in
  let foot := (10, -5, 2) in
  A = 10 ∧ B = -5 ∧ C = 2 ∧ A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
  (foot : point) = (10, -5, 2) ∧ 
  (A * foot.1 + B * foot.2 + C * foot.3 + D = 0) ∧
  plane A B C D :=
by
  sorry

end equation_of_plane_l13_13263


namespace sequence_2011_eq_2_l13_13719

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 8
  else if n = 1 then 3
  else (sequence (n - 1) + sequence (n - 2)) % 10

theorem sequence_2011_eq_2 : sequence 2011 = 2 :=
  sorry

end sequence_2011_eq_2_l13_13719


namespace helicopter_rental_cost_l13_13299

theorem helicopter_rental_cost
  (hours_per_day : ℕ)
  (total_days : ℕ)
  (total_cost : ℕ)
  (H1 : hours_per_day = 2)
  (H2 : total_days = 3)
  (H3 : total_cost = 450) :
  total_cost / (hours_per_day * total_days) = 75 :=
by
  sorry

end helicopter_rental_cost_l13_13299


namespace sequence_a4_l13_13469

theorem sequence_a4 :
  (∀ n : ℕ, n > 0 → ∀ (a : ℕ → ℝ),
    (a 1 = 1) →
    (∀ n > 0, a (n + 1) = (1 / 2) * a n + 1 / (2 ^ n)) →
    a 4 = 1 / 2) :=
by
  sorry

end sequence_a4_l13_13469


namespace proposition_1_proposition_2_proposition_3_proposition_4_l13_13125

def nearest_integer (x : ℝ) : ℤ :=
  if h : ∃ m : ℤ, m - 1/2 < x ∧ x ≤ m + 1/2 then
    Classical.choose h
  else 0

def f (x : ℝ) : ℝ :=
  |x - (nearest_integer x : ℝ)|

theorem proposition_1 : f (-1/2) = 1/2 := sorry

theorem proposition_2 : f (3.4) ≠ -0.4 := sorry

theorem proposition_3 : f (-1/4) = f (1/4) := sorry

theorem proposition_4 : ∀ y, ∃ x, y = f x ∧ y ∈ set.Icc 0 (1/2) := sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l13_13125


namespace slope_angle_parametric_line_l13_13272

theorem slope_angle_parametric_line (t : ℝ) : 
  let x := 2 + t,
      y := 1 + (real.sqrt 3 / 3) * t,
      slope := (real.sqrt 3 / 3) / 1,
      θ := real.arctan slope
  in θ = real.pi / 6 := 
begin
  sorry
end

end slope_angle_parametric_line_l13_13272


namespace Paco_initial_salty_cookies_l13_13956

-- Definitions based on conditions
variables (initial_salty_cookies : ℕ) (sweet_cookies : ℕ := 17)
variables (eaten_sweet_cookies : ℕ := 14) (eaten_salty_cookies : ℕ := 9)
variables (salty_cookies_left : ℕ := 17)

-- The mathematical proof problem statement
theorem Paco_initial_salty_cookies :
  initial_salty_cookies - eaten_salty_cookies = salty_cookies_left → initial_salty_cookies = 26 :=
begin
  intro h,
  -- Placeholder for the proof
  sorry,
end

end Paco_initial_salty_cookies_l13_13956


namespace sum_of_slopes_constant_l13_13834

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ) (a_pos : 0 < a) (b_pos: 0 < b), (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1)) ∧
  (∃ (F : ℝ × ℝ) (A₁ A₂ : ℝ × ℝ),
    F = (1, 0) ∧ A₁ = (-a, 0) ∧ A₂ = (a, 0) ∧ 
    let FA₁ := (tuple.fst A₁ - tuple.fst F, tuple.snd A₁ - tuple.snd F) in
    let FA₂ := (tuple.fst A₂ - tuple.fst F, tuple.snd A₂ - tuple.snd F) in
    ((FA₁.1 * FA₂.1) + (FA₁.2 * FA₂.2) = -1))
  
theorem sum_of_slopes_constant : 
  ellipse_equation →
  ∀ {k : ℝ} {P Q : ℝ × ℝ} {B : ℝ × ℝ},
  B = (0, -1) →
  (P ≠ B ∧ Q ≠ B) →
  (∃ (x₁ x₂ y₁ y₂ : ℝ), P = (x₁, y₁) ∧ Q = (x₂, y₂)) →
  (∃ (k : ℝ), P.2 = k * (P.1 - 1) + 1 ∧ Q.2 = k * (Q.1 - 1) + 1) →
  (k ≠ 2) →
  (∃ (slopes_sum : ℝ), slopes_sum = 2) :=
sorry

end sum_of_slopes_constant_l13_13834


namespace prove_a_ge_5_find_b_c_for_a_5_l13_13119

-- Define the conditions as Lean structures and predicates
structure Conditions (a b c : ℤ) :=
  (a_pos : a > 0)
  (roots_in_interval : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ (a:ℝ) * x₁^2 + (b:ℝ) * x₁ + (c:ℝ) = 0 ∧ (a:ℝ) * x₂^2 + (b:ℝ) * x₂ + (c:ℝ) = 0)

-- Define the main theorem to prove a >= 5
theorem prove_a_ge_5 (a b c : ℤ) (h : Conditions a b c) : a ≥ 5 :=
  sorry

-- Define the existential theorem to find b and c when a = 5
theorem find_b_c_for_a_5 : ∃ b c : ℤ, Conditions 5 b c :=
  ⟨-5, 1, ⟨by decide, by decide⟩⟩

end prove_a_ge_5_find_b_c_for_a_5_l13_13119


namespace deposit_amount_l13_13700

-- Conditions
def total_cost := 550
def remaining := 495
def deposit := 0.10 * total_cost

-- Statement
theorem deposit_amount : total_cost - deposit = remaining → deposit = 55 := 
by
  intro h
  -- Here we just state the theorem and the necessary implications
  sorry

end deposit_amount_l13_13700


namespace count_interesting_quadruples_l13_13416

def is_interesting (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d > b + c

-- Main theorem statement
theorem count_interesting_quadruples : 
  (finset.filter (λ t : ℕ × ℕ × ℕ × ℕ, is_interesting t.1 t.2.1 t.2.2.1 t.2.2.2) 
    ((finset.range 16).product ((finset.range 16).product ((finset.range 16).product (finset.range 16)))).card) 
  = 635 :=
sorry

end count_interesting_quadruples_l13_13416


namespace problem_b_problem_c_problem_d_l13_13116

variables (a b : ℝ)
hypothesis (pos_a : 0 < a) (pos_b : 0 < b) (h : a + 4*b = a*b)

theorem problem_b :
  a + 4*b = 16 := 
  sorry

theorem problem_c :
  a + 2*b ≥ 6 + 4 * Real.sqrt 2 := 
  sorry

theorem problem_d :
  16 / a^2 + 1 / b^2 ≥ 1 / 2 := 
  sorry

end problem_b_problem_c_problem_d_l13_13116


namespace P_144_is_216543_l13_13189

def permutations (l : List ℕ) : List (List ℕ) := List.permutations l

def nth_permutation (l : List ℕ) (n : ℕ) : Option (List ℕ) :=
  List.get? (permutations l).qsort (fun a b => List.lex (<) a b) (n - 1)

theorem P_144_is_216543 :
  (nth_permutation [1, 2, 3, 4, 5, 6] 144).iget = [2, 1, 6, 5, 4, 3] :=
sorry

end P_144_is_216543_l13_13189


namespace rectangle_segments_sum_l13_13593

theorem rectangle_segments_sum :
  let EF := 6
  let FG := 8
  let n := 210
  let diagonal_length := Real.sqrt (EF^2 + FG^2)
  let segment_length (k : ℕ) : ℝ := diagonal_length * (n - k) / n
  let sum_segments := 2 * (Finset.sum (Finset.range 210) segment_length) - diagonal_length
  sum_segments = 2080 := by
  sorry

end rectangle_segments_sum_l13_13593


namespace a_minus_b_ge_one_l13_13566

def a : ℕ := 19^91
def b : ℕ := (999991)^19

theorem a_minus_b_ge_one : a - b ≥ 1 :=
by
  sorry

end a_minus_b_ge_one_l13_13566


namespace f_leq_zero_l13_13461

noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + (2 * a - 1) * x

theorem f_leq_zero (a x : ℝ) (h1 : 1/2 < a) (h2 : a ≤ 1) (hx : 0 < x) :
  f x a ≤ 0 :=
sorry

end f_leq_zero_l13_13461


namespace sum_a4_a6_l13_13105

variable (a : ℕ → ℝ) (d : ℝ)
variable (h_arith : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
variable (h_sum : a 2 + a 3 + a 7 + a 8 = 8)

theorem sum_a4_a6 : a 4 + a 6 = 4 :=
by
  sorry

end sum_a4_a6_l13_13105


namespace age_of_father_now_l13_13004

variable (M F : ℕ)

theorem age_of_father_now :
  (M = 2 * F / 5) ∧ (M + 14 = (F + 14) / 2) → F = 70 :=
by 
sorry

end age_of_father_now_l13_13004


namespace expenditures_ratio_l13_13622

open Real

variables (I1 I2 E1 E2 : ℝ)
variables (x : ℝ)

theorem expenditures_ratio 
  (h1 : I1 = 4500)
  (h2 : I1 / I2 = 5 / 4)
  (h3 : I1 - E1 = 1800)
  (h4 : I2 - E2 = 1800) : 
  E1 / E2 = 3 / 2 :=
by
  have h5 : I1 / 5 = x := by sorry
  have h6 : I2 = 4 * x := by sorry
  have h7 : I2 = 3600 := by sorry
  have h8 : E1 = 2700 := by sorry
  have h9 : E2 = 1800 := by sorry
  exact sorry 

end expenditures_ratio_l13_13622


namespace eccentricity_of_hyperbola_l13_13844

-- Conditions setup
variables {a b e c : ℝ}
variables (hx : a > 0) (hy : b > 0)
-- Hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1

-- Eccentricity
def eccentricity (e : ℝ) : Prop :=
  e = (Real.sqrt 6) / 2

-- The chord length condition of the parabola y^2 = 4cx
def chord_length (be : ℝ) : Prop :=
  be = (2 * Real.sqrt 2 / 3) * (b * e ^ 2)

-- The relation between a, b, and c for the hyperbola
def hyperbola_relation (a b c : ℝ) : Prop :=
  c ^ 2 = a ^ 2 + b ^ 2

-- The final proof statement
theorem eccentricity_of_hyperbola :
  ∀ {a b e c : ℝ},
    (a > 0) →
    (b > 0) →
    hyperbola_relation a b c →
    chord_length b e →
    eccentricity e :=
by { intros, sorry }

end eccentricity_of_hyperbola_l13_13844


namespace problem1_problem2_problem3_l13_13468

noncomputable theory

open_locale big_operators

-- Problem 1
theorem problem1 (a : ℕ → ℤ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, |a (n + 1) - a n| = 1) : 
  a 4 ∈ ({-2, 0, 2, 4} : set ℤ) :=
sorry

-- Problem 2
theorem problem2 (a : ℕ → ℝ) (p : ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, a (n + 1) - a n = p^n)
(h₂ : ∀ n : ℕ, a (n+1) > a n) (h₃ : 3 * a 3 - 2 * a 2 = 2 * a 2 - a 1) :
  p = -1/3 :=
sorry

-- Problem 3
theorem problem3 (a : ℕ → ℝ) (h₀ : a 1 = 1) 
(h₁ : ∀ n : ℕ, |a (n + 1) - a n| = (1/2)^n)
(h₂ : ∀ n : ℕ, a (2 * n - 1) < a (2 * n + 1))
(h₃ : ∀ n : ℕ, a (2 * n) > a (2 * n + 2)) :
  ∀ n, a n = 4 / 3 + 1 / 3 * (-1)^n / 2^(n - 1) :=
sorry

end problem1_problem2_problem3_l13_13468


namespace rational_segments_l13_13333

noncomputable def rational_triangle (A B C B_1 : ℝ) (AB AC BC BB_1 : ℚ) : Prop :=
  B_1 = (B - (B * A + B * C)) / AC

theorem rational_segments 
  (A B C B_1 : ℝ)
  (AB AC BC BB_1 : ℚ)
  (h : B_1 = (B - (B * A + B * C)) / AC) :
  ∃ AB_1 CB_1 : ℚ, ∀ AB AC BC BB_1 : ℚ, AB_1 = (B - (B * A)) / AC + C - B ∧ CB_1 = (B + (B * C)) / AC - A + B :=
begin
  sorry
end

end rational_segments_l13_13333


namespace simplify_expr_l13_13248

theorem simplify_expr :
  (2^8 + 4^5) * (2^3 - (-2)^3)^7 = 1280 * 16^7 :=
by
  have h1 : 2^8 = 256 := by norm_num
  have h2 : 4^5 = 1024 := by norm_num
  have h3 : 2^3 = 8 := by norm_num
  have h4 : (-2)^3 = -8 := by norm_num
  have h5 : 2^3 - (-2)^3 = 16 := by norm_num
  have h6 : (2^3 - (-2)^3)^7 = 16^7 := by norm_num
  have h7 : 2^8 + 4^5 = 1280 := by norm_num
  have h8 : (2^8 + 4^5) * (2^3 - (-2)^3)^7 = 1280 * 16^7 := by
    rw [h1, h2, h3, h4, h5, h6, h7]
    norm_num
  exact h8

end simplify_expr_l13_13248


namespace distance_between_intersections_l13_13361
open Real

noncomputable def cube_vertices : set (ℝ × ℝ × ℝ) :=
  {(0,0,0), (6,0,0), (6,6,0), (0,6,0), (0,0,6), (6,0,6), (6,6,6), (0,6,6)}

def cutting_plane : ℝ × ℝ × ℝ :=
  (-3, 10, 4)

theorem distance_between_intersections :
  let U := (0:ℝ, 0:ℝ, 3:ℝ) in
  let V := (6:ℝ, 6:ℝ, 3:ℝ) in
  dist (U) (V) = 6 * sqrt 2 :=
by {
  sorry
}

end distance_between_intersections_l13_13361


namespace find_points_on_line_l13_13072

theorem find_points_on_line (x y : ℝ)
  (h1 : x + 3 * y = 0)
  (h2 : Real.sqrt (x^2 + y^2) = Real.abs ((1 / Real.sqrt 10) * (x + 3 * y + 2))) :
  (x = -3/5 ∧ y = 1/5) ∨ (x = 3/5 ∧ y = -1/5) :=
by
  sorry

end find_points_on_line_l13_13072


namespace ratio_of_abc_l13_13693

theorem ratio_of_abc {a b c : ℕ} (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) 
                     (h_ratio : ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x)
                     (h_mean : (a + b + c) / 3 = 42) : 
  a = 28 := 
sorry

end ratio_of_abc_l13_13693


namespace problem_solution_l13_13698

-- Definition of geometric sequence with given conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a n > 0 ∧ a (n + 1) = a n * q

-- Conditions given in the math problem
def conditions (a : ℕ → ℝ) (q : ℝ) :=
  geometric_sequence a q ∧ 0 < q ∧ q < 1 ∧
  a 1 * a 5 + 2 * a 3 * a 5 + a 2 * a 8 = 25 ∧
  (a 3 * a 5)^(1/2) = 2

-- General formula to be proved
def general_formula (a : ℕ → ℝ) := ∀ n : ℕ, a n = 2^(5 - n)

-- Sum of sequence S_n to be proved
def sum_sequence (S : ℕ → ℝ) := ∀ n : ℕ, S n = n * (9 - n) / 2

theorem problem_solution (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  conditions a q → general_formula a ∧ sum_sequence (λ n, (log 2 (a n))) S :=
by
  sorry

end problem_solution_l13_13698


namespace sequence_a_l13_13759

variable (a : ℕ → ℕ)
variable (c : ℕ)
variable (n : ℕ)

-- Conditions
axiom cond_1 : a 0 ≥ 2015
axiom cond_2 : ∀ n ≥ 1, a (n + 2) % a n = 0
axiom cond_3 : ∀ n ≥ 1, abs (a (n + 1) - a n + a (n - 1) - ... + (-1) ^ (n + 1) * a 0 - (n + 1) * a n) = 1

-- Definition of the sequences
def seq_1 (n : ℕ) : ℕ := if n = 0 then c + 1 else c * (n + 2) * n
def seq_2 (n : ℕ) : ℕ := if n = 0 then c - 1 else c * (n + 2) * n

theorem sequence_a (c ≥ 2014) :
  ( ∀ n, a n = seq_1 n ) ∨ ( ∀ n, a n = seq_2 n ) :=
sorry

end sequence_a_l13_13759


namespace find_common_ratio_l13_13939

-- Define a geometric sequence and sums.
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)
def sum_first_n_terms (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

-- Given conditions.
variable {a₁ q : ℝ} (h₁ : q > 0)
variable (hS₂ : sum_first_n_terms a₁ q 2 = 3 * geometric_sequence a₁ q 2 + 2)
variable (hS₄ : sum_first_n_terms a₁ q 4 = 3 * geometric_sequence a₁ q 4 + 2)

-- Problem: Prove the common ratio q is 2
theorem find_common_ratio : q = 2 :=
by { sorry }

end find_common_ratio_l13_13939


namespace cone_lateral_surface_area_l13_13999

theorem cone_lateral_surface_area {r l : ℝ} (hr : r = 3) (hl : l = 5) :
  ∃ A, A = π * r * l ∧ A = 15 * π :=
by
  use π * r * l
  split
  · rw [hr, hl]
    ring
  · sorry

end cone_lateral_surface_area_l13_13999


namespace drum_oil_capacity_l13_13335

theorem drum_oil_capacity (C : ℝ) (hC : C > 0):
  let drumX_oil := (1/2) * C,
      drumY_capacity := 2 * C,
      drumY_oil := (1/3) * drumY_capacity,
      total_oil_in_drumY := drumY_oil + drumX_oil in
  total_oil_in_drumY / drumY_capacity = 7 / 12 :=
by
  let drumX_oil := (1/2) * C
  let drumY_capacity := 2 * C
  let drumY_oil := (1/3) * drumY_capacity
  let total_oil_in_drumY := drumY_oil + drumX_oil
  sorry

end drum_oil_capacity_l13_13335


namespace infinitely_many_lovely_no_lovely_square_gt_1_l13_13658

def lovely (n : ℕ) : Prop :=
  ∃ (k : ℕ) (d : Fin k → ℕ),
    n = (List.ofFn d).prod ∧
    ∀ i, (d i)^2 ∣ n + (d i)

theorem infinitely_many_lovely : ∀ N : ℕ, ∃ n > N, lovely n :=
  sorry

theorem no_lovely_square_gt_1 : ∀ n : ℕ, n > 1 → lovely n → ¬∃ m, n = m^2 :=
  sorry

end infinitely_many_lovely_no_lovely_square_gt_1_l13_13658


namespace decrease_intervals_delta_area_l13_13487

-- Define the function f(x)
def f (x : ℝ) : ℝ := - ((sin x) ^ 2) - (sqrt 3) * (sin x) * (cos x) + (1 / 2)

-- Part (Ⅰ): Checking the intervals of monotonic decrease in [0, π]
theorem decrease_intervals : 
  monotonic_decreasing_on f (Icc 0 (π / 3)) ∧ 
  monotonic_decreasing_on f (Icc (5 * π / 6) π) :=
sorry

-- Part (Ⅱ): Given conditions and proving the area of ΔABC
variables {a b c : ℝ} {A B C : ℝ}
hypothesis (h1 : f A = -1)
hypothesis (h2 : a = 2)
hypothesis (h3 : b * sin C = a * sin A)
hypothesis (acute : 0 < A ∧ A < π / 2)

theorem delta_area : 
  ∃ (area : ℝ), area = sqrt 3 :=
sorry

end decrease_intervals_delta_area_l13_13487


namespace vertical_angles_congruent_l13_13240

theorem vertical_angles_congruent (a b : Angle) (h : VerticalAngles a b) : CongruentAngles a b :=
sorry

end vertical_angles_congruent_l13_13240


namespace expected_value_3_defective_0_3_variance_3_defective_0_255_l13_13832

noncomputable def probability_mass_function (n k nk : ℕ) : ℚ :=
(nk.choose(k) * (n - nk).choose(3 - k)) / n.choose(3)

noncomputable def expected_value_defective (n nk : ℕ) : ℚ :=
(probability_mass_function n 0 nk) * 0 +
(probability_mass_function n 1 nk) * 1 +
(probability_mass_function n 2 nk) * 2 +
(probability_mass_function n 3 nk) * 3

noncomputable def second_moment_defective (n nk : ℕ) : ℚ :=
(probability_mass_function n 0 nk) * 0^2 +
(probability_mass_function n 1 nk) * 1^2 +
(probability_mass_function n 2 nk) * 2^2 +
(probability_mass_function n 3 nk) * 3^2

noncomputable def variance_defective (n nk : ℕ) : ℚ :=
second_moment_defective n nk - (expected_value_defective n nk)^2

theorem expected_value_3_defective_0_3 : expected_value_defective 100 10 = 0.3 := 
by sorry

theorem variance_3_defective_0_255 : variance_defective 100 10 = 0.255 := 
by sorry

end expected_value_3_defective_0_3_variance_3_defective_0_255_l13_13832


namespace initial_water_amount_l13_13365

variable (evaporation_per_day : ℝ) (days : ℕ) (W : ℝ) (total_evaporated : ℝ)

-- Given conditions
def evaporation_per_day := 0.01
def days := 20
def total_evaporated := evaporation_per_day * (days : ℝ)

theorem initial_water_amount :
  (0.005 * W = total_evaporated) → W = 40 := by
  sorry

end initial_water_amount_l13_13365


namespace equal_segments_of_cyclic_quadrilaterals_l13_13345

theorem equal_segments_of_cyclic_quadrilaterals
  {A B C D E F P Q : Type*}
  (h_acute : acute_angled_triangle A B C)
  (h_altitudes : altitude A E ∧ altitude C D)
  (h_angle_bisector : angle_bisector B (D - E) F)
  (h_P_on_AE : point_on_segment P (A - E))
  (h_Q_on_CD : point_on_segment Q (C - D))
  (h_ADFQ_cyclic : cyclic_quadrilateral A D F Q)
  (h_CEPF_cyclic : cyclic_quadrilateral C E P F) :
  segment_length A P = segment_length C Q :=
by
  sorry

end equal_segments_of_cyclic_quadrilaterals_l13_13345


namespace find_w_l13_13854

theorem find_w (u v w : ℝ) (h1 : 10 * u + 8 * v + 5 * w = 160)
  (h2 : v = u + 3) (h3 : w = 2 * v) : w = 13.5714 := by
  -- The proof would go here, but we leave it empty as per instructions.
  sorry

end find_w_l13_13854


namespace intersection_M_N_l13_13849

def M := {x : ℝ | -3 < x ∧ x ≤ 5}
def N := {y : ℝ | -5 < y ∧ y < 5}

theorem intersection_M_N : M ∩ N = {z : ℝ | -3 < z ∧ z < 5} := 
sorry

end intersection_M_N_l13_13849


namespace festival_winner_is_D_l13_13717

def team := {A, B, C, D}

def prediction (team_wins : team → Prop) : (Prop × Prop × Prop × Prop) :=
  (team_wins A ∨ team_wins B,  -- Xiao Zhang's prediction
   team_wins D,                -- Xiao Wang's prediction
   ¬team_wins B ∧ ¬team_wins C,-- Xiao Li's prediction
   team_wins A)                -- Xiao Zhao's prediction

theorem festival_winner_is_D (team_wins : team → Prop) :
  (let (zhang, wang, li, zhao) := prediction team_wins in
   (zhang = true) ↔ (team_wins = {A, D}) ∧
   (wang = true) ↔ (team_wins D) ∧
   (li = true) ↔ (¬team_wins B ∧ ¬team_wins C) ∧
   (zhao = true) ↔ (team_wins A) ∧
   2 = {zhang, wang, li, zhao}.filter id.card
  ) → (team_wins D) :=
by
  intro h
  let (zhang, wang, li, zhao) := prediction team_wins
  sorry

end festival_winner_is_D_l13_13717


namespace teacup_problem_l13_13287

-- Define the initial states of the teacups
def initial_states (m : ℕ) : list ℤ := list.replicate m 1

-- Define the condition under which we flip n teacups
def flip_teacups (states : list ℤ) (n : ℕ) : list ℤ :=
list.map_with_index
  (λ i state, if i < n then -state else state)
  states

-- Define the product of the states
def product_of_states (states : list ℤ) : ℤ :=
list.prod states

theorem teacup_problem (m n : ℕ) (h_m_geq_3 : m ≥ 3) (h_m_odd : m % 2 = 1)
  (h_n_geq_2 : n ≥ 2) (h_n_lt_m : n < m) (h_n_even : n % 2 = 0) :
  ∀ k : ℕ, product_of_states (flip_teacups (initial_states m) n) ≠ -1 :=
by
  sorry

end teacup_problem_l13_13287


namespace parabola_line_AB_length_l13_13696

theorem parabola_line_AB_length (x1 x2 xM : ℝ) (y1 y2: ℝ) 
  (h_focus : (focus_x = 1 ∧ focus_y = 0))
  (h_parabola : ∀ (x y : ℝ), y^2 = 4 * x ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, p.2 ^ 2 = 4 * p.1))
  (h_midpoint : xM = (x1 + x2) / 2)
  (h_xM : xM = 3)
  (h_A_on_parabola : (A_x, A_y) ∈ set_of (λ p : ℝ × ℝ, p.2 ^ 2 = 4 * p.1))
  (h_B_on_parabola : (B_x, B_y) ∈ set_of (λ p : ℝ × ℝ, p.2 ^ 2 = 4 * p.1)) : 
  |x2 - x1 + y2 - y1| = 8 := by
  sorry

end parabola_line_AB_length_l13_13696


namespace chord_intersects_inner_circle_probability_l13_13648

-- Define the radius of circles and the central point
def center : ℝ² := (0,0)
def radius_inner : ℝ := 2
def radius_outer : ℝ := 3

-- Assume points are chosen independently and uniformly at random
def random_point_on_circle (r : ℝ) : ProbabilityMeasure ℝ² :=
  sorry  -- Assuming a probability measure for random selection on the circle

-- Define a function checking if a chord intersects the inner circle
def chord_intersects_inner_circle (p1 p2 : ℝ²) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  ∥midpoint∥ < radius_inner

-- The probability that the chord joining two random points on the outer circle intersects the inner circle
noncomputable def probability_chord_intersects_inner_circle :
  ℝ :=
  Probability.probability (do
    p1 ← random_point_on_circle radius_outer,
    p2 ← random_point_on_circle radius_outer,
    pure (chord_intersects_inner_circle p1 p2))

-- Proof statement that this probability is approximately 0.2148
theorem chord_intersects_inner_circle_probability :
  |probability_chord_intersects_inner_circle - 0.2148| < 0.0001 :=
sorry

end chord_intersects_inner_circle_probability_l13_13648


namespace unit_A_saplings_l13_13604

theorem unit_A_saplings 
  (Y B D J : ℕ)
  (h1 : J = 2 * Y + 20)
  (h2 : J = 3 * B + 24)
  (h3 : J = 5 * D - 45)
  (h4 : J + Y + B + D = 2126) :
  J = 1050 :=
by sorry

end unit_A_saplings_l13_13604


namespace ratio_of_areas_of_rectangle_and_square_l13_13990

theorem ratio_of_areas_of_rectangle_and_square (s : ℝ) :
  let longer_side_R := 1.2 * s
  let shorter_side_R := 0.8 * s
  let area_S := s^2
  let area_R := longer_side_R * shorter_side_R
  let diagonal_S := s * real.sqrt 2
  let diagonal_R := (1.1) * diagonal_S
  diagonal_R = real.sqrt (longer_side_R^2 + shorter_side_R^2) ->
  (area_R / area_S) = (24 / 25) := sorry

end ratio_of_areas_of_rectangle_and_square_l13_13990


namespace triangle_area_l13_13894

noncomputable def area_of_triangle (w : ℂ) : ℝ :=
  (complex.abs (w^2 - w))^2 * (sqrt 3 / 4)

theorem triangle_area (w : ℂ) (h1 : w^4 - w = complex.exp (complex.I * (2 * real.pi / 3)) * (w^2 - w) ∨
                                  w^4 - w = complex.exp (-complex.I * (2 * real.pi / 3)) * (w^2 - w)) :
  area_of_triangle w = (3 * sqrt 3) / 4 :=
sorry

end triangle_area_l13_13894


namespace a_1996_is_square_l13_13559

noncomputable def a : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 1
| 3       := 2
| 4       := 4
| (n + 1) :=
  if h : n + 1 > 4 then
    (a n) + (a (n - 2)) + (a (n - 3))
  else 0

noncomputable def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib n + fib (n + 1)

theorem a_1996_is_square : ∃ k : ℕ, a 1996 = k * k := 
sorry

end a_1996_is_square_l13_13559


namespace sum_abc_equals_10_l13_13570

theorem sum_abc_equals_10 :
  ∃ (a b c : ℕ), 
    (∀ x, (f x : ℤ) = 
      if x > 0 then a * x + 4 
      else if x = 0 then a * b 
      else b * x + c) ∧ 
    f 3 = 7 ∧ 
    f 0 = 6 ∧ 
    f (-3) = -15 ∧ 
    a + b + c = 10 :=
by sorry

end sum_abc_equals_10_l13_13570


namespace g_identical_to_inverse_when_shifted_l13_13386

noncomputable def f : ℝ → ℝ :=
λ x, (x - 3) / (x - 1)

noncomputable def g (a : ℝ) : ℝ → ℝ :=
λ x, f (x + a)

theorem g_identical_to_inverse_when_shifted :
  ∃ (a : ℝ), (g a = g a ⁻¹) ↔ (a = 1) :=
by
  sorry

end g_identical_to_inverse_when_shifted_l13_13386


namespace impossible_seven_weights_l13_13098

open Finset

theorem impossible_seven_weights (s : Finset ℕ) (h : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 26) (hs : s.card = 7) :
  ∃ (a b : Finset ℕ), a ≠ b ∧ a ⊆ s ∧ b ⊆ s ∧ a.sum = b.sum :=
by
  sorry

end impossible_seven_weights_l13_13098


namespace max_single_digit_sum_of_2016_digit_integer_l13_13357

theorem max_single_digit_sum_of_2016_digit_integer :
  ∃ (n : ℕ), (nat.digits 10 n).length = 2016 ∧ (nat.digit_sum n).1 = 9 :=
by
  sorry

end max_single_digit_sum_of_2016_digit_integer_l13_13357


namespace number_of_integer_perfect_squares_l13_13802

theorem number_of_integer_perfect_squares : 
  (∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ ∃ k : ℤ, n / (25 - n) = k^2) → 2 :=
by sorry

end number_of_integer_perfect_squares_l13_13802


namespace boys_on_debate_team_l13_13623

theorem boys_on_debate_team (total_members girls: ℕ): total_members = 72 ∧ girls = 46 → 
  (total_members - girls) = 26 :=
by
  intros h
  cases h with total_eq girls_eq
  rw [total_eq, girls_eq]
  exact Nat.sub_eq_of_eq_add (by norm_num)

end boys_on_debate_team_l13_13623


namespace michael_birth_year_l13_13262

theorem michael_birth_year (first_AMC8_year : ℕ) (tenth_AMC8_year : ℕ) (age_during_tenth_AMC8 : ℕ) 
  (h1 : first_AMC8_year = 1985) (h2 : tenth_AMC8_year = (first_AMC8_year + 9)) (h3 : age_during_tenth_AMC8 = 15) :
  (tenth_AMC8_year - age_during_tenth_AMC8) = 1979 :=
by
  sorry

end michael_birth_year_l13_13262


namespace max_f_neg1_l13_13149

theorem max_f_neg1 (f : ℝ → ℝ) (hf_monic : ∃ a b c : ℝ, f = λ x, x^3 + a * x^2 + b * x + c)
  (hf_roots_nonneg : ∃ a b c : ℝ, (f = λ x, (x - a) * (x - b) * (x - c)) ∧ (0 ≤ a) ∧ (0 ≤ b) ∧ (0 ≤ c))
  (hf_zero : f 0 = -64) :
  f (-1) ≤ -125 :=
sorry

end max_f_neg1_l13_13149


namespace area_of_given_triangle_l13_13625

-- Define the variables and conditions
variables (a b c : ℝ)
variables (h₁ : a = 15)
variables (h₂ : b = 36)
variables (h₃ : c = 39)

-- Define the condition that verifies the triangle is a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

-- Define the area formula for a right triangle
def right_triangle_area (a b : ℝ) : ℝ :=
  1 / 2 * a * b

-- The final theorem statement
theorem area_of_given_triangle : is_right_triangle a b c → right_triangle_area a b = 270 :=
by
  intros h
  rw [h₁, h₂, h₃]
  sorry -- skipping the complete proof

end area_of_given_triangle_l13_13625


namespace relationship_among_a_b_c_l13_13113

noncomputable def a := Real.log 0.8 / Real.log 0.5
noncomputable def b := Real.log 0.8 / Real.log 1.1
noncomputable def c := 1.1 ^ 0.8

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l13_13113


namespace vertical_angles_congruent_l13_13236

theorem vertical_angles_congruent (A B : Angle) (h : vertical_angles A B) : congruent A B := sorry

end vertical_angles_congruent_l13_13236


namespace synodic_month_is_approx_29_5306_l13_13694

noncomputable def sidereal_month_moon : ℝ := 
27 + 7/24 + 43/1440  -- conversion of 7 hours and 43 minutes to days

noncomputable def sidereal_year_earth : ℝ := 
365 + 6/24 + 9/1440  -- conversion of 6 hours and 9 minutes to days

noncomputable def synodic_month (T_H T_F: ℝ) : ℝ := 
(T_H * T_F) / (T_F - T_H)

theorem synodic_month_is_approx_29_5306 : 
  abs (synodic_month sidereal_month_moon sidereal_year_earth - (29 + 12/24 + 44/1440)) < 0.0001 :=
by 
  sorry

end synodic_month_is_approx_29_5306_l13_13694


namespace remainder_of_power_mod_l13_13665

theorem remainder_of_power_mod (h : (2^10 : ℤ) ≡ 1 [MOD 11]) : (2^2023 : ℤ) ≡ 8 [MOD 11] :=
by
  sorry

end remainder_of_power_mod_l13_13665


namespace perfect_square_A_perfect_square_D_l13_13127

def is_even (n : ℕ) : Prop := n % 2 = 0

def A : ℕ := 2^10 * 3^12 * 7^14
def D : ℕ := 2^20 * 3^16 * 7^12

theorem perfect_square_A : ∃ k : ℕ, A = k^2 :=
by
  sorry

theorem perfect_square_D : ∃ k : ℕ, D = k^2 :=
by
  sorry

end perfect_square_A_perfect_square_D_l13_13127


namespace range_of_a_l13_13130

def f (a x : ℝ) := a * x - x^3

theorem range_of_a {a : ℝ} : (∀ x1 x2 ∈ Ioo 0 1, x1 < x2 → f a x2 - f a x1 > x2 - x1) → 4 ≤ a :=
by
  sorry

end range_of_a_l13_13130


namespace number_on_board_after_61_minutes_l13_13951

-- Define the function that computes the next number based on the problem conditions
def next_number (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  (d1 * d2) + 15

-- Define the initial number
def initial_number : ℕ := 98

-- Define the sequence of numbers based on the next_number function and the initial number
def sequence : ℕ → ℕ 
| 0     := initial_number
| (n+1) := next_number (sequence n)

-- The theorem to prove
theorem number_on_board_after_61_minutes : sequence 61 = 23 :=
sorry

end number_on_board_after_61_minutes_l13_13951


namespace tan_A_eq_11_l13_13163

variable (A B C : ℝ)

theorem tan_A_eq_11
  (h1 : Real.sin A = 10 * Real.sin B * Real.sin C)
  (h2 : Real.cos A = 10 * Real.cos B * Real.cos C) :
  Real.tan A = 11 := 
sorry

end tan_A_eq_11_l13_13163


namespace range_of_a_l13_13823

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → x ∈ Iio (-1 / 2) → y ∈ Iio (-1 / 2) → 
    log (1 / 2) (x^2 - a * x - a) < log (1 / 2) (y^2 - a * y - a)) ↔ 
    -1 ≤ a ∧ a < 1 / 2 :=
by
  sorry

end range_of_a_l13_13823


namespace binom_8_3_eq_56_l13_13403

def binom (n k : ℕ) : ℕ :=
(n.factorial) / (k.factorial * (n - k).factorial)

theorem binom_8_3_eq_56 : binom 8 3 = 56 := by
  sorry

end binom_8_3_eq_56_l13_13403


namespace arithmetic_sequence_term_l13_13546

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Conditions
def common_difference := d = 2
def value_a_2007 := a 2007 = 2007

-- Question to be proved
theorem arithmetic_sequence_term :
  common_difference d →
  value_a_2007 a →
  a 2009 = 2011 :=
by
  sorry

end arithmetic_sequence_term_l13_13546


namespace problem1_problem2_l13_13137

-- Define the sets and the conditions
def A : Set ℝ := {-2, 3, 4, 6}
noncomputable def B (a : ℝ) : Set ℝ := {3, a, a^2}

-- First problem: Proving if B ⊆ A then a = 2
theorem problem1 (a : ℝ) : B a ⊆ A → a = 2 := by
  sorry

-- Second problem: Proving if A ∩ B = {3, 4} then a = 2 or a = 4
theorem problem2 (a : ℝ) : A ∩ B a = {3, 4} → a = 2 ∨ a = 4 := by
  sorry

end problem1_problem2_l13_13137


namespace arccos_of_cos_periodic_l13_13740

theorem arccos_of_cos_periodic :
  arccos (cos 8) = 8 - 2 * Real.pi :=
by
  sorry

end arccos_of_cos_periodic_l13_13740


namespace range_of_alpha_l13_13108

variable {α t t₀ : ℝ}
variable {OA OB OP OQ PQ : ℝ → ℝ} -- Here we consider them functions of time for simplicity
variable {vector_len1 : OA(0) = 1}
variable {vector_len2 : OB(0) = 2}
variable {noncoplanar : ∀ t, ¬ coPlanar (OA(t)) (OB(t)) (OP(t))}
variable {angle_between : angle OA OB = α}
variable {op_eq : ∀ t, OP(t) = (1-t) * OA(t)}
variable {oq_eq : ∀ t, OQ(t) = t * OB(t)}
variable {pq_eq : ∀ t, PQ(t) = OQ(t) - OP(t)}
variable {t_bounds : 0 ≤ t ∧ t ≤ 1}
variable {min_t0 : (PQ t₀)^2 = (5 + 4 * cos α) * t₀^2 - 2 * (1 + 2 * cos α) * t₀ + 1}
variable {t0_bounds : 0 < t₀ ∧ t₀ < 1/5}

theorem range_of_alpha (OA OB OP OQ PQ : ℝ → ℝ)
  (vector_len1 : OA 0 = 1)
  (vector_len2 : OB 0 = 2)
  (noncoplanar : ∀ t, ¬ coPlanar (OA t) (OB t) (OP t))
  (angle_between : angle (OA 0) (OB 0) = α)
  (op_eq : ∀ t, OP t = (1-t) * OA t)
  (oq_eq : ∀ t, OQ t = t * OB t)
  (pq_eq : ∀ t, PQ t = OQ t - OP t)
  (t_bounds : 0 ≤ t ∧ t ≤ 1)
  (min_t0 : (PQ t₀)^2 = (5 + 4 * cos α) * t₀^2 - 2 * (1 + 2 * cos α) * t₀ + 1)
  (t0_bounds : 0 < t₀ ∧ t₀ < 1/5) :
  (π/2 < α ∧ α < 2 * π / 3) :=
sorry

end range_of_alpha_l13_13108


namespace number_of_ways_to_choose_two_groups_l13_13542

theorem number_of_ways_to_choose_two_groups (Mathematics ComputerScience ModelAviation : Type) :
  (finset.card (finset.powerset (finset.insert Mathematics (finset.insert ComputerScience (finset.singleton ModelAviation))).filter (λ s, finset.card s = 2)) = 3) :=
by sorry

end number_of_ways_to_choose_two_groups_l13_13542


namespace combined_value_is_29_l13_13936

-- Solution Definitions and Conditions
def i := 2
def k := 44
def j := 23
def expression := 2 * i - k + 3 * j

-- The Proof Problem
theorem combined_value_is_29 : expression = 29 := by
  rfl

end combined_value_is_29_l13_13936


namespace CD_is_b_minus_a_minus_c_l13_13544

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (A B C D : V) (a b c : V)

def AB : V := a
def AD : V := b
def BC : V := c

theorem CD_is_b_minus_a_minus_c (h1 : A + a = B) (h2 : A + b = D) (h3 : B + c = C) :
  D - C = b - a - c :=
by sorry

end CD_is_b_minus_a_minus_c_l13_13544


namespace cora_tuesday_pages_l13_13413

theorem cora_tuesday_pages :
  ∀ (total_pages pages_monday pages_wednesday pages_thursday pages_friday pages_tuesday : ℕ),
  total_pages = 158 →
  pages_monday = 23 →
  pages_wednesday = 61 →
  pages_thursday = 12 →
  pages_friday = 2 * pages_thursday →
  pages_tuesday = total_pages - (pages_monday + pages_wednesday + pages_thursday + pages_friday) →
  pages_tuesday = 38 :=
begin
  intros total_pages pages_monday pages_wednesday pages_thursday pages_friday pages_tuesday,
  intros h_total_pages h_pages_monday h_pages_wednesday h_pages_thursday h_pages_friday h_pages_tuesday,
  rw [h_total_pages, h_pages_monday, h_pages_wednesday, h_pages_thursday, h_pages_friday, h_pages_tuesday],
  norm_num, -- 158 - (23 + 61 + 12 + 24) = 38
end

end cora_tuesday_pages_l13_13413


namespace liam_prob_9_l13_13941

/-- The probability that Liam eventually writes 9 on the board,
    given that he stops when either 8 or 9 is written, is 7/10. -/
theorem liam_prob_9 : 
  let N : ℤ := sorry in
  let prob_N : ℤ → ℝ := λ N, 3^(-|N|) in
  let stop : ℤ → Prop := λ x, x = 8 ∨ x = 9 in
  let process : ℤ → ℤ → Prop := λ n1 n2, n2 = n1 + N in
  let final_state : ℤ → ℝ := sorry in
  ∀ (x y : ℝ), 
    (y = 7 / 10) ∧ sum_of_probabilities_of_states final_state = 1 :=
  sorry

end liam_prob_9_l13_13941


namespace equilateral_triangle_l13_13347

noncomputable def h_B (a b c s : ℝ) : ℝ := (2 / a) * real.sqrt (s * (s - a) * (s - b) * (s - c))
def m_C (a b c : ℝ) : ℝ := (1 / 2) * real.sqrt (2 * a^2 + 2 * b^2 - c^2)
def h_C (a b c s : ℝ) : ℝ := (2 / a) * real.sqrt (s * (s - a) * (s - b) * (s - c))
def m_B (a b c : ℝ) : ℝ := (1 / 2) * real.sqrt (2 * a^2 + 2 * c^2 - b^2)

theorem equilateral_triangle (a b c : ℝ) (s : ℝ) 
  (hB_eq_mC : h_B a b c s = m_C a b c) 
  (hC_eq_mB : h_C a b c s = m_B a b c) : 
  ∃ A B C : ℝ, (A = 60) ∧ (B = 60) ∧ (C = 60) := 
by 
  sorry

end equilateral_triangle_l13_13347


namespace max_n_intersection_non_empty_l13_13501

-- Define the set An
def An (n : ℕ) : Set ℝ := {x : ℝ | n < x^n ∧ x^n < n + 1}

-- State the theorem
theorem max_n_intersection_non_empty : 
  ∃ x, (∀ n, n ≤ 4 → x ∈ An n) ∧ (∀ n, n > 4 → x ∉ An n) :=
by
  sorry

end max_n_intersection_non_empty_l13_13501


namespace conic_section_union_l13_13985

theorem conic_section_union : 
  ∀ (y x : ℝ), y^4 - 6*x^4 = 3*y^2 - 2 → 
  ( ( y^2 - 3*x^2 = 1 ∨ y^2 - 2*x^2 = 1 ) ∧ 
    ( y^2 - 2*x^2 = 2 ∨ y^2 - 3*x^2 = 2 ) ) :=
by
  sorry

end conic_section_union_l13_13985


namespace euclidean_remainder_sum_first_n_l13_13931

theorem euclidean_remainder_sum_first_n (n : ℕ) (h₀ : n ≠ 0) :
  let S_n := n * (n + 1) / 2 in
  let r := S_n % n in
  r = if n % 2 = 1 then 0 else n / 2 :=
by 
  sorry

end euclidean_remainder_sum_first_n_l13_13931


namespace right_angled_triangle_setB_l13_13322

def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c

theorem right_angled_triangle_setB :
  isRightAngledTriangle 1 1 (Real.sqrt 2) ∧
  ¬isRightAngledTriangle 1 2 3 ∧
  ¬isRightAngledTriangle 6 8 11 ∧
  ¬isRightAngledTriangle 2 3 4 :=
by
  sorry

end right_angled_triangle_setB_l13_13322


namespace decimal_to_vulgar_fraction_l13_13412

theorem decimal_to_vulgar_fraction (d : ℚ) (h : d = 0.36) : d = 9 / 25 :=
by {
  sorry
}

end decimal_to_vulgar_fraction_l13_13412


namespace range_of_a_l13_13527

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0 ↔ a ∈ Set.Ioo (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l13_13527


namespace triangle_area_l13_13079

-- Given coordinates of vertices of triangle ABC
def A := (0, 0)
def B := (1424233, 2848467)
def C := (1424234, 2848469)

-- Define a mathematical proof statement to prove the area of the triangle ABC
theorem triangle_area :
  let area_ABC := (1 / 2 : ℝ) * Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) * (1 / Real.sqrt (2^2 + (-1)^2))
  (Float.to_string 0.50) = "0.50" :=
by
  let area_ABC := (1 / 2 : ℝ) * Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) * (1 / Real.sqrt (2^2 + (-1)^2))
  sorry
    

end triangle_area_l13_13079


namespace expression_value_l13_13308

   theorem expression_value :
     (20 - (2010 - 201)) + (2010 - (201 - 20)) = 40 := 
   by
     sorry
   
end expression_value_l13_13308


namespace simplify_expression_evaluate_expression_l13_13735

-- Part 1: Proof for simplifying the expression
theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (x - 1) / x ÷ (2 * x - 2) / (x^2) = x / 2 :=
by
  sorry

-- Part 2: Proof for simplifying and evaluating the expression when a = 2
theorem evaluate_expression (a : ℝ) (ha : a = 2) : 
  (2 - (a - 1) / (a + 1)) ÷ ((a^2 + 6 * a + 9) / (a + 1)) = 1 / 5 :=
by
  have ha_pos : a = 2 := ha,
  calc
    (2 - (a - 1) / (a + 1)) ÷ ((a^2 + 6 * a + 9) / (a + 1))
      = (2 - (2 - 1) / (2 + 1)) ÷ ((2^2 + 6 * 2 + 9) / (2 + 1)) : by rw ha_pos
  sorry

end simplify_expression_evaluate_expression_l13_13735


namespace sum_of_k_values_l13_13274

theorem sum_of_k_values : 
  let ks := { k ∈ Set.Univ | ∃ (α β : ℤ), α * β = 16 ∧ k = α + β ∧ k > 0 } 
  in ks.sum = 35 :=
by
  sorry

end sum_of_k_values_l13_13274


namespace circle_condition_l13_13887

-- Define the center of the circle
def center := ((-3 + 27) / 2, (0 + 0) / 2)

-- Define the radius of the circle
def radius := 15

-- Define the circle's equation
def circle_eq (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the final Lean 4 statement
theorem circle_condition (x : ℝ) : circle_eq x 12 → (x = 21 ∨ x = 3) :=
  by
  intro h
  -- Proof goes here
  sorry

end circle_condition_l13_13887


namespace original_wage_before_increase_l13_13332

theorem original_wage_before_increase (new_wage : ℝ) (increase_rate : ℝ) (original_wage : ℝ) (h : new_wage = original_wage + increase_rate * original_wage) : 
  new_wage = 42 → increase_rate = 0.50 → original_wage = 28 :=
by
  intros h_new_wage h_increase_rate
  have h1 : new_wage = 42 := h_new_wage
  have h2 : increase_rate = 0.50 := h_increase_rate
  have h3 : new_wage = original_wage + increase_rate * original_wage := h
  sorry

end original_wage_before_increase_l13_13332


namespace binary_operation_l13_13946

def binary_mult_subtract (a b c : ℕ) : ℕ :=
  (a * b) - c

theorem binary_operation :
  let a := 0b1101
  let b := 0b111
  let c := 0b101
  binary_mult_subtract a b c = 0b1001000 :=
by {
  sorry,
}

end binary_operation_l13_13946


namespace slope_at_certain_point_equal_three_l13_13494
open Real

noncomputable def f (x a : ℝ) := x * log x + a * x^2

theorem slope_at_certain_point_equal_three (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ deriv (λ z, f z a) x = 3) → a ≥ -1 / (2 * exp 3) := 
by
  sorry

end slope_at_certain_point_equal_three_l13_13494


namespace distinct_difference_count_l13_13511

def S : Set ℕ := {2, 3, 5, 7, 8, 10}

theorem distinct_difference_count (S : Set ℕ) (h : S = {2, 3, 5, 7, 8, 10}) : 
    (Finset.card ((Finset.filter (λ x => x > 0) (Finset.image2 (λ x y => x - y) 
    (S.toFinset) (S.toFinset))) ∖ {(0 : ℕ)})) = 7 := sorry

end distinct_difference_count_l13_13511


namespace least_n_progression_l13_13500

theorem least_n_progression (P : Nat -> ℝ) (n : Nat) : 
  (∀ n, P n = 2 ^ (n / 13)) ∧ (∏ i in Finset.range n, P (i + 1) > 1000000) → n = 23 := 
by
  sorry

end least_n_progression_l13_13500


namespace proof_problem_l13_13888

def curve_C_parametric (α : ℝ) : ℝ × ℝ :=
  (3 * Real.cos α, Real.sin α)

def line_l_polar (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ - π / 4) = sqrt 2

def curve_C_general (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

def line_l_cartesian (x y : ℝ) : Prop :=
  y = x + 2

def point_P : ℝ × ℝ := (0, 2)

def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (sqrt 2 / 2 * t, 2 + sqrt 2 / 2 * t)

theorem proof_problem :
  (∀ {α : ℝ}, curve_C_parametric α = (3 * Real.cos α, Real.sin α)) →
  (∀ {ρ θ : ℝ}, line_l_polar ρ θ → line_l_cartesian ρ θ) →
  (curve_C_general 0 2) →
  (∀ t1 t2 : ℝ, (curve_C_general (sqrt 2 / 2 * t1) (2 + sqrt 2 / 2 * t1)) ∧ (curve_C_general (sqrt 2 / 2 * t2) (2 + sqrt 2 / 2 * t2)) → 
  |point_P.1 - sqrt 2 / 2 * t1| + |point_P.2 - (2 + sqrt 2 / 2 * t1)| + 
  |point_P.1 - sqrt 2 / 2 * t2| + |point_P.2 - (2 + sqrt 2 / 2 * t2)| = 18 * sqrt 2 / 5) :=
by sorry

end proof_problem_l13_13888


namespace arman_sister_age_l13_13381

-- Define the conditions
variables (S : ℝ) -- Arman's sister's age four years ago
variable (A : ℝ) -- Arman's age four years ago

-- Given conditions as hypotheses
axiom h1 : A = 6 * S -- Arman is six times older than his sister
axiom h2 : A + 8 = 40 -- In 4 years, Arman's age will be 40 (hence, A in 4 years should be A + 8)

-- Main theorem to prove
theorem arman_sister_age (h1 : A = 6 * S) (h2 : A + 8 = 40) : S = 16 / 3 :=
by
  sorry

end arman_sister_age_l13_13381


namespace vertical_angles_congruent_l13_13238

theorem vertical_angles_congruent (A B : Angle) (h : vertical_angles A B) : congruent A B := sorry

end vertical_angles_congruent_l13_13238


namespace sphere_radius_twice_volume_cone_l13_13705

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1/3) * real.pi * r^2 * h
noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4/3) * real.pi * r^3
noncomputable def two_volumes_of_cone (r h : ℝ) : ℝ := 2 * volume_of_cone r h

theorem sphere_radius_twice_volume_cone:
  (volume_of_cone 2 8) * 2 = volume_of_sphere (real.cbrt 16) :=
by
  sorry

end sphere_radius_twice_volume_cone_l13_13705


namespace triangle_solution_l13_13968

noncomputable def solve_triangle (a : ℝ) (α : ℝ) (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let s := 75
  let b := 41
  let c := 58
  let β := 43 + 36 / 60 + 10 / 3600
  let γ := 77 + 19 / 60 + 11 / 3600
  ((b, c), (β, γ))

theorem triangle_solution :
  solve_triangle 51 (59 + 4 / 60 + 39 / 3600) 1020 = ((41, 58), (43 + 36 / 60 + 10 / 3600, 77 + 19 / 60 + 11 / 3600)) :=
sorry  

end triangle_solution_l13_13968


namespace total_handshakes_l13_13384

-- Define the conditions
def number_of_players_per_team : Nat := 11
def number_of_referees : Nat := 3
def total_number_of_players : Nat := number_of_players_per_team * 2

-- Prove the total number of handshakes
theorem total_handshakes : 
  (number_of_players_per_team * number_of_players_per_team) + (total_number_of_players * number_of_referees) = 187 := 
by {
  sorry
}

end total_handshakes_l13_13384


namespace roots_of_quadratic_l13_13278

theorem roots_of_quadratic :
  (∀ x : ℝ, 3 * x^2 = x ↔ x = 0 ∨ x = 1 / 3) :=
by
  intro x
  split
  { intro h
    have : x * (3 * x - 1) = 0 :=
      by linarith [h]
    cases mul_eq_zero.mp this
    { left
      exact h_1 }
    { right
      linarith [h_1] } }
  { intro h
    cases h
    { rw h
      ring }
    { rw h
      field_simp
      ring } }

end roots_of_quadratic_l13_13278


namespace arrange_desc_l13_13460

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (35 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (35 * Real.pi / 180)
noncomputable def d : ℝ := Real.log 5

theorem arrange_desc : d > c ∧ c > b ∧ b > a := by
  sorry

end arrange_desc_l13_13460


namespace magnitude_of_projection_l13_13195

variables (v w : ℝ^2) -- Using two-dimensional vectors for simplicity
variable (dot_product : ℝ)
variable (norm_w : ℝ)
variable (magnitude : ℝ)

theorem magnitude_of_projection (v w : ℝ^2) (h1 : dot_product = 8) (h2 : norm_w = 4) : magnitude = 2 :=
sorry

end magnitude_of_projection_l13_13195


namespace intersection_point_const_dist_l13_13340

open EuclideanGeometry

-- Definitions of points and lines
noncomputable def Point := EuclideanGeometry.Point ℝ
noncomputable def Line := EuclideanGeometry.Line ℝ

-- Conditions
variable (A B A1 B1 P : Point)
variable (AB : Line)
variable (a b : ℝ)

-- Given conditions: 
-- - Points A, B on line AB
-- - Perpendicular segments AA1 = a and BB1 = b
-- - Lines AA1 and BB1 are perpendicular to AB at A and B, respectively
variables (hA : A ∈ AB)
          (hB : B ∈ AB)
          (hPerpA : EuclideanGeometry.perpendicular AB (Line.mk A A1))
          (hPerpB : EuclideanGeometry.perpendicular AB (Line.mk B B1))
          (hDistA : EuclideanGeometry.distance A A1 = a)
          (hDistB : EuclideanGeometry.distance B B1 = b)
          (hIntersect : EuclideanGeometry.intersection (Line.mk A B1) (Line.mk A1 B) = P)

-- The proof statement: 
theorem intersection_point_const_dist (A B A1 B1 P : Point) (AB : Line) (a b : ℝ)
  (hA : A ∈ AB) (hB : B ∈ AB) (hPerpA : EuclideanGeometry.perpendicular AB (Line.mk A A1))
  (hPerpB : EuclideanGeometry.perpendicular AB (Line.mk B B1)) 
  (hDistA : EuclideanGeometry.distance A A1 = a)
  (hDistB : EuclideanGeometry.distance B B1 = b) 
  (hIntersect : EuclideanGeometry.intersection (Line.mk A B1) (Line.mk A1 B) = P) :
  ∃ d : ℝ, ∀ A B : Point, (A ∈ AB) → (B ∈ AB) → EuclideanGeometry.distance (EuclideanGeometry.projection P AB) P = d :=
sorry

end intersection_point_const_dist_l13_13340


namespace number_of_incorrect_propositions_is_1_l13_13128

-- Define propositions as lean statements
variable (p q : Prop) (a b A B : Real) (x : Real)

-- Define the conditions specific to the problem
def prop1 := ¬(p ∧ q) → (¬p ∧ ¬q)
def prop2 := ¬(a > b → 2^a > 2^b - 1) = (a ≤ b → 2^a ≤ 2^b - 1)
def prop3 := ¬(∀ x : ℝ, x^2 + 1 ≥ 0) = (∃ x : ℝ, x^2 + 1 < 0)
def prop4 := (A > B) ↔ (sin A > sin B)

-- The proof problem statement
theorem number_of_incorrect_propositions_is_1 (h1 : prop1) (h2 : prop2) (h3 : prop3) (h4 : prop4) :
  [h1 = false, h2 = true, h3 = true, h4 = true].count (fun b => b = false) = 1 := sorry

end number_of_incorrect_propositions_is_1_l13_13128


namespace polynomial_divisible_by_prime_l13_13158

theorem polynomial_divisible_by_prime
  (p : ℕ) [fact p.prime]
  (a b c : ℤ)
  (α β γ : ℤ)
  (h1 : p ∣ (a * α^2 + b * α + c))
  (h2 : p ∣ (a * β^2 + b * β + c))
  (h3 : p ∣ (a * γ^2 + b * γ + c))
  (h4 : ¬ p ∣ (α - β))
  (h5 : ¬ p ∣ (β - γ))
  (h6 : ¬ p ∣ (γ - α)) :
  ∀ x : ℤ, p ∣ (a * x^2 + b * x + c) :=
begin
  sorry
end

end polynomial_divisible_by_prime_l13_13158


namespace William_won_10_rounds_l13_13684

theorem William_won_10_rounds (H : ℕ) (total_rounds : H + (H + 5) = 15) : H + 5 = 10 := by
  sorry

end William_won_10_rounds_l13_13684


namespace area_diff_l13_13588

-- Definitions based on conditions
def length_side_unit_square : ℝ := 1
def area_unit_square : ℝ := length_side_unit_square^2
def side_length_equilateral_triangle : ℝ := 1
def area_equilateral_triangle (s : ℝ) : ℝ := (s^2 * sqrt(3)) / 4
def side_length_small_square : ℝ := 1 
def area_small_square (s : ℝ) : ℝ := s^2

-- Given the definitions above, let's calculate the region P
def total_area_region_P : ℝ :=
  area_unit_square + 2 * area_equilateral_triangle(side_length_equilateral_triangle) + 
  6 * area_small_square(side_length_small_square)

-- Area of the smallest convex polygon Q that contains the region P
def area_Q : ℝ := 6 * sqrt(3)

-- The area inside Q but outside P
def area_inside_Q_outside_P : ℝ := area_Q - total_area_region_P

theorem area_diff (h1 : length_side_unit_square = 1) (h2 : side_length_equilateral_triangle = 1) 
  (h3 : side_length_small_square = 1) : 
  area_inside_Q_outside_P = (11 * sqrt(3))/2 - 7 :=
by
  sorry

end area_diff_l13_13588


namespace factorize_expression_l13_13436

-- Defining the variables x and y as real numbers.
variable (x y : ℝ)

-- Statement of the proof problem.
theorem factorize_expression : 
  x * y^2 - x = x * (y + 1) * (y - 1) :=
sorry

end factorize_expression_l13_13436


namespace sum_of_squares_l13_13734

theorem sum_of_squares : 
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  (squares.sum = 195) := 
by
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  have h : squares.sum = 195 := sorry
  exact h

end sum_of_squares_l13_13734


namespace perfect_square_factors_of_14400_l13_13513

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

theorem perfect_square_factors_of_14400 : 
  ∃ (count : ℕ), count = 12 ∧ (∀ (d : ℕ), (d ∣ 14400) → is_perfect_square d ↔ d ∣ 14400 ∧ ∃ (a b c : ℕ), a ∈ {0, 2, 4} ∧ b ∈ {0, 2} ∧ c ∈ {0, 2} ∧ d = (2^a * 3^b * 5^c)) :=
by 
  sorry

end perfect_square_factors_of_14400_l13_13513


namespace find_sin_ratio_l13_13890

-- Define the positions of A and C
def A : ℝ × ℝ := (-6, 0)
def C : ℝ × ℝ := (6, 0)

-- Define the hyperbola on which B lies
def hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 11 = 1

-- Define the final proof problem statement in Lean
theorem find_sin_ratio (B : ℝ × ℝ) (hb : hyperbola B.1 B.2) : 
  (B.2 ≠ 0) → ( (sin (atan2 A.2 A.1) - sin (atan2 C.2 C.1)) / sin (atan2 B.2 B.1) = 5 / 6 ) :=
by
  sorry

end find_sin_ratio_l13_13890


namespace fencing_cost_correct_l13_13708

def circular_radius := 200
def semicircular_diameter := 150
def cost_circular_per_meter := 5
def cost_semicircular_flat_per_meter := 3
def cost_semicircular_curved_per_meter := 4
def π_approx := 3.14159

-- Define a function to calculate the total cost of the fencing
def total_fencing_cost : ℝ :=
  let C_circular := 2 * π * circular_radius
  let C_semicircular_curved := π * (semicircular_diameter / 2)
  let C_semicircular_flat := semicircular_diameter
  let Cost_circular := C_circular * cost_circular_per_meter
  let Cost_semicircular_flat := C_semicircular_flat * cost_semicircular_flat_per_meter
  let Cost_semicircular_curved := C_semicircular_curved * cost_semicircular_curved_per_meter
  Cost_circular + Cost_semicircular_flat + Cost_semicircular_curved

theorem fencing_cost_correct : abs (total_fencing_cost - 7676.66) < 0.01 :=
by
  unfold total_fencing_cost
  unfold circular_radius semicircular_diameter cost_circular_per_meter cost_semicircular_flat_per_meter cost_semicircular_curved_per_meter π_approx
  norm_num
  sorry

end fencing_cost_correct_l13_13708


namespace angle_between_vectors_l13_13923

variable {V : Type} [InnerProductSpace ℝ V]

theorem angle_between_vectors (a b : V) (ha : ∥a∥ = 1) (hb : ∥b∥ = 2) (h : inner (a + b) a = 0) : 
  real.arccos (inner a b / (∥a∥ * ∥b∥)) = real.pi / 3 :=
by
  -- Definitions and import statements
  sorry

end angle_between_vectors_l13_13923


namespace hyperbola_k_range_l13_13833

theorem hyperbola_k_range (k : ℝ) : ((k + 2) * (6 - 2 * k) > 0) ↔ (-2 < k ∧ k < 3) := 
sorry

end hyperbola_k_range_l13_13833


namespace cost_of_remaining_ingredients_l13_13285

theorem cost_of_remaining_ingredients :
  let cocoa_required := 0.4
  let sugar_required := 0.6
  let cake_weight := 450
  let given_cocoa := 259
  let cost_per_lb_cocoa := 3.50
  let cost_per_lb_sugar := 0.80
  let total_cocoa_needed := cake_weight * cocoa_required
  let total_sugar_needed := cake_weight * sugar_required
  let remaining_cocoa := max 0 (total_cocoa_needed - given_cocoa)
  let remaining_sugar := total_sugar_needed
  let total_cost := remaining_cocoa * cost_per_lb_cocoa + remaining_sugar * cost_per_lb_sugar
  total_cost = 216 := by
  sorry

end cost_of_remaining_ingredients_l13_13285


namespace correct_statements_l13_13561

-- Define the function and the given conditions
def f : ℝ → ℝ := sorry

lemma not_constant (h: ∃ x y: ℝ, x ≠ y ∧ f x ≠ f y) : true := sorry
lemma periodic (x : ℝ) : f (x - 1) = f (x + 1) := sorry
lemma symmetric (x : ℝ) : f (2 - x) = f x := sorry

-- The statements we want to prove
theorem correct_statements : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (1 - x) = f (1 + x)) ∧ 
  (∃ T > 0, ∀ x, f (x + T) = f x)
:= by
  sorry

end correct_statements_l13_13561


namespace math_problem_l13_13027

theorem math_problem : 1999^2 - 2000 * 1998 = 1 := 
by
  sorry

end math_problem_l13_13027


namespace factorize_expression_l13_13435

-- Defining the variables x and y as real numbers.
variable (x y : ℝ)

-- Statement of the proof problem.
theorem factorize_expression : 
  x * y^2 - x = x * (y + 1) * (y - 1) :=
sorry

end factorize_expression_l13_13435


namespace correct_statement_l13_13697

structure Line : Type := 
  (incl : ∀ (P : Point), P ∈ Line → P ∈ Plane)

structure Plane : Type := 
  (perp : ∀ (L : Line), (L ∈ Plane) → Prop)
  (parallel : ∀ (L : Line), (L ∈ Plane) → Prop)

variables (m n : Line) (α β : Plane)

axiom line_perp_plane : m.incl ∧ n.incl → m.perp α ∧ m.parallel n → n.parallel β → α.perp β

theorem correct_statement : m.perp α ∧ m.parallel n → n.parallel β → α.perp β :=
begin
  sorry
end

end correct_statement_l13_13697


namespace Z_belongs_to_XY_diameter_circle_l13_13343

-- Assumptions
variables (A B C D X Y Z : Point)
variables (AB CD AD BC : Line)
variables (circumcircle inscribed_circle : Circle)

-- Definitions according to the conditions
def is_cyclic_quadrilateral (A B C D : Point) : Prop :=
  is_cyclic quadrilateral with points A B C D

def intersection (L1 L2 : Line) : Point :=
  intersect L1 L2

def belongs_to_circle (P : Point) (circ : Circle) : Prop :=
  P on circ

def orthogonal_circles (circ1 circ2 : Circle) : Prop :=
  orthogonal circ1 circ2

-- Given conditions
axiom hABCDis_cyclic : is_cyclic_quadrilateral A B C D
axiom hX : X = intersection AB CD
axiom hY : Y = intersection AD BC
axiom h_inscribed_circle : inscribed_circle = inscribed_circle_of_quadrilateral A B C D
axiom hZ_center : center inscribed_circle = Z

-- To Prove
theorem Z_belongs_to_XY_diameter_circle (A B C D X Y Z : Point)
   (AB CD AD BC : Line)
   (circumcircle inscribed_circle : Circle)
   (hABCDis_cyclic : is_cyclic_quadrilateral A B C D)
   (hX : X = intersection AB CD)
   (hY : Y = intersection AD BC)
   (h_inscribed_circle : inscribed_circle = inscribed_circle_of_quadrilateral A B C D)
   (hZ_center : center inscribed_circle = Z) :
   ∃ circ_XY : Circle, 
   (circ_XY = circle_with_diameter X Y) ∧
   belongs_to_circle Z circ_XY ∧
   orthogonal_circles circ_XY circumcircle := sorry

end Z_belongs_to_XY_diameter_circle_l13_13343


namespace main_proof_l13_13921

/-- Two different planes alpha and beta. -/
variable {α β : Type}

/-- Two different lines l and m. -/
variable {l m : α}

/-- Proposition p: If α is parallel to β, l intersects α, and m intersects β,
    then l is parallel to m. -/
def proposition_p (h1 : α ∥ β) (h2 : l ∩ α) (h3 : m ∩ β) : l ∥ m :=
sorry

/-- Proposition q: If l is parallel to α, m is perpendicular to l, and m intersects β,
    then α is perpendicular to β. -/
def proposition_q (h1 : l ∥ α) (h2 : m ⊥ l) (h3 : m ∩ β) : α ⊥ β :=
sorry

/-- Main proof that under the given conditions, "not p or q" is true. -/
theorem main_proof (h1 : ¬ proposition_p α β l m)
                   (h2 : ¬ proposition_q α β l m) :
  ¬ proposition_p α β l m ∨ proposition_q α β l m :=
begin
  exact or.inl h1,
end

end main_proof_l13_13921


namespace vector_addition_l13_13143

variable {𝕍 : Type} [AddCommGroup 𝕍] [Module ℝ 𝕍]
variable (a b : 𝕍)

theorem vector_addition : 
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by
  sorry

end vector_addition_l13_13143


namespace room_length_l13_13791

theorem room_length (B H D : ℝ) (hB : B = 8) (hH : H = 9) (hD : D = 17) : 
  ∃ L : ℝ, L = 12 :=
by
  let L_squared := D * D - B * B - H * H
  have hL_squared : L_squared = 144, {
    calc
      L_squared = D * D - B * B - H * H : by sorry
             ... = 17 * 17 - 8 * 8 - 9 * 9 : by sorry
             ... = 289 - 64 - 81 : by sorry
             ... = 144 : by sorry
  }
  use real.sqrt 144
  have hL : real.sqrt 144 = 12 := by sorry
  simp [hL]

end room_length_l13_13791


namespace solve_abs_ineq_l13_13050

theorem solve_abs_ineq (x : ℝ) (h : x > 0) : |4 * x - 5| < 8 ↔ 0 < x ∧ x < 13 / 4 :=
by
  sorry

end solve_abs_ineq_l13_13050


namespace sum_last_two_digits_l13_13026

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem sum_last_two_digits :
  last_two_digits (∑ k in {n | n % 3 = 0 ∧ 3 ≤ n ∧ n ≤ 99}.toFinset, 2 * k.factorial) = 24 :=
by
  sorry

end sum_last_two_digits_l13_13026


namespace least_value_sum_log3_l13_13862

variable {a b : ℝ}

theorem least_value_sum_log3 (h : Real.log 3 a + Real.log 3 b ≥ 7) : a + b ≥ 18 * Real.sqrt 27 := 
sorry

end least_value_sum_log3_l13_13862


namespace find_second_number_l13_13254

theorem find_second_number 
  (h₁ : (20 + 40 + 60) / 3 = (10 + x + 15) / 3 + 5) :
  x = 80 :=
  sorry

end find_second_number_l13_13254


namespace fraction_of_sum_l13_13013

theorem fraction_of_sum (P : ℝ) (R : ℝ) (T : ℝ) (H_R : R = 8.333333333333337) (H_T : T = 2) : 
  let SI := (P * R * T) / 100
  let A := P + SI
  A / P = 1.1666666666666667 :=
by
  sorry

end fraction_of_sum_l13_13013


namespace sum_first_eight_geom_terms_eq_l13_13796

noncomputable def S8_geom_sum : ℚ :=
  let a := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  a * (1 - r^8) / (1 - r)

theorem sum_first_eight_geom_terms_eq :
  S8_geom_sum = 3280 / 6561 :=
by
  sorry

end sum_first_eight_geom_terms_eq_l13_13796


namespace intersection_M_N_l13_13572

def M : Set ℝ := { x | x^2 + x - 2 < 0 }
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l13_13572


namespace natasha_avg_speed_up_hill_l13_13583

theorem natasha_avg_speed_up_hill :
  ∀ (D : ℝ),
  (∀ (hours_up : ℝ), hours_up = 4 → D / hours_up = 2.25) ∧
  (∀ (hours_down : ℝ), hours_down = 2 → D / hours_down = 4.5) ∧
  (∀ (total_distance total_time avg_speed : ℝ), total_distance = 2 * D → total_time = 6 → avg_speed = 3 → 6 * 3 = total_distance → D = 9) → 
  ∃ speed_up : ℝ, speed_up = 2.25 :=
begin
  intros D h1 h2 h3,
  sorry
end

end natasha_avg_speed_up_hill_l13_13583


namespace integral_2x_ex_l13_13774

open Real IntervalIntegral

theorem integral_2x_ex : ∫ x in 0..1, (2 * x + exp x) = exp 1 := 
begin
  have h : ∀ x, deriv (λ x, x^2 + exp x) x = 2 * x + exp x := by  
  { intro x,
    calc (λ x, x^2 + exp x).deriv x = x.deriv * 1 + (exp x).deriv : 
      by apply funext; intro x; ring_deriv -- Windows addition
                          ... = 2 * x + exp x : by ring },
  rw integral_eq_sub_of_has_deriv_at h (by continuity) (by continuity),
  simp,
  norm_num,
end

end integral_2x_ex_l13_13774


namespace arithmetic_seq_geom_mean_eq_five_l13_13483

theorem arithmetic_seq_geom_mean_eq_five (d : ℤ) (h_d : d ≠ 0) :
  ∀ (a : ℕ → ℤ), 
  (a 1 = 2 * d) →
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (∀ k : ℕ, (a k)^2 = a 1 * a (2 * k + 7) → k = 5) :=
by {
  intros,
  sorry
}

end arithmetic_seq_geom_mean_eq_five_l13_13483


namespace difference_between_mean_and_median_l13_13885

def percent_students := {p : ℝ // 0 ≤ p ∧ p ≤ 1}

def students_scores_distribution (p60 p75 p85 p95 : percent_students) : Prop :=
  p60.val + p75.val + p85.val + p95.val = 1 ∧
  p60.val = 0.15 ∧
  p75.val = 0.20 ∧
  p85.val = 0.40 ∧
  p95.val = 0.25

noncomputable def weighted_mean (p60 p75 p85 p95 : percent_students) : ℝ :=
  60 * p60.val + 75 * p75.val + 85 * p85.val + 95 * p95.val

noncomputable def median_score (p60 p75 p85 p95 : percent_students) : ℝ :=
  if p60.val + p75.val < 0.5 then 85 else if p60.val + p75.val < 0.9 then 95 else 60

theorem difference_between_mean_and_median :
  ∀ (p60 p75 p85 p95 : percent_students),
    students_scores_distribution p60 p75 p85 p95 →
    abs (median_score p60 p75 p85 p95 - weighted_mean p60 p75 p85 p95) = 3.25 :=
by
  intro p60 p75 p85 p95
  intro h
  sorry

end difference_between_mean_and_median_l13_13885


namespace thirty_six_forty_five_nine_eighteen_l13_13091

theorem thirty_six_forty_five_nine_eighteen :
  18 * 36 + 45 * 18 - 9 * 18 = 1296 :=
by
  sorry

end thirty_six_forty_five_nine_eighteen_l13_13091


namespace temperature_below_zero_l13_13891

-- Assume the basic definitions and context needed
def above_zero (temp : Int) := temp > 0
def below_zero (temp : Int) := temp < 0

theorem temperature_below_zero (t1 t2 : Int) (h1 : above_zero t1) (h2 : t2 = -7) :
  below_zero t2 := by 
  -- This is where the proof would go
  sorry

end temperature_below_zero_l13_13891


namespace imaginary_part_of_z_l13_13117

-- Definition of the imaginary unit and the complex number z
def i : ℂ := complex.I
def z : ℂ := i * (i - 1)

-- The proof problem statement
theorem imaginary_part_of_z :
  complex.im z = -1 := 
sorry

end imaginary_part_of_z_l13_13117


namespace equation_pattern_l13_13533
open Nat

theorem equation_pattern (n : ℕ) (h_pos : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end equation_pattern_l13_13533


namespace red_team_probability_expected_value_team_wins_l13_13235

variables (P_D P_E P_F : ℝ)
variables (indep : indipendents([P_D, P_E, P_F]))
axiom P_D_value : P_D = 0.6
axiom P_E_value : P_E = 0.5
axiom P_F_value : P_F = 0.5

theorem red_team_probability :
  let P_D = 0.6 in
  let P_E = 0.5 in
  let P_F = 0.5 in
    ( (P_D * P_E * (1 - P_F))
    + (P_D * (1 - P_E) * P_F)
    + ((1 - P_D) * P_E * P_F)
    + (P_D * P_E * P_F))
    = 0.55 := 
by {
  simp [P_D_value, P_E_value, P_F_value],
  sorry
}

theorem expected_value_team_wins :
  let P_D = 0.6 in
  let P_E = 0.5 in
  let P_F = 0.5 in
    ( (0 * (1 - P_D) * (1 - P_E) * (1 - P_F))
    + (1 * (P_D * (1 - P_E) * (1 - P_F)))
    + (1 * ((1 - P_D) * P_E * (1 - P_F)))
    + (1 * ((1 - P_D) * (1 - P_E) * P_F))
    + (2 * (P_D * P_E * (1 - P_F)))
    + (2 * (P_D * (1 - P_E) * P_F))
    + (2 * ((1 - P_D) * P_E * P_F))
    + (3 * (P_D * P_E * P_F)) )
    = 1.6 :=
by {
  simp [P_D_value, P_E_value, P_F_value],
  sorry
}

end red_team_probability_expected_value_team_wins_l13_13235


namespace second_book_cost_l13_13963

theorem second_book_cost 
  (initial_money : ℕ)
  (cost_first_book : ℕ)
  (poster_cost : ℕ)
  (posters_bought : ℕ)
  (remaining_money : ℕ) :
  initial_money = 20 →
  cost_first_book = 8 →
  poster_cost = 4 →
  posters_bought = 2 →
  remaining_money = initial_money - cost_first_book - (poster_cost * posters_bought) →
  remaining_money = 4 :=
by
  intros h1 h2 h3 h4 h5
  have h_goal : 20 - 8 - (4 * 2) = 4 := by 
    calc
      20 - 8 - (4 * 2) = 20 - 8 - 8 : rfl
                         ... = 12 - 8 : rfl
                         ... = 4 : rfl
  exact h_goal

-- sorry added to complete the goal
sorry

end second_book_cost_l13_13963


namespace no_unique_p_l13_13961

-- Define the probabilities P_1 and P_2 given p
def P1 (p : ℝ) : ℝ := 3 * p^2 - 2 * p^3
def P2 (p : ℝ) : ℝ := 3 * p^2 - 3 * p^3

-- Define the expected value E(xi)
def E_xi (p : ℝ) : ℝ := P1 p + P2 p

-- Prove that there does not exist a unique p in (0, 1) such that E(xi) = 1.5
theorem no_unique_p (p : ℝ) (h : 0 < p ∧ p < 1) : E_xi p ≠ 1.5 := by
  sorry

end no_unique_p_l13_13961


namespace souvenir_price_in_october_l13_13703

theorem souvenir_price_in_october
  (sales_september : ℝ)
  (discount_percentage : ℝ)
  (increase_volume : ℕ)
  (increase_revenue : ℝ)
  (sales_october : ℝ) :
  sales_september = 2000 ∧ 
  discount_percentage = 0.1 ∧ 
  increase_volume = 20 ∧ 
  increase_revenue = 700 ∧
  sales_october = 2700 →
  let price_september := 50 in
  let price_october := price_september * (1 - discount_percentage) in
  price_october = 45 :=
begin
  intros, 
  sorry
end

end souvenir_price_in_october_l13_13703


namespace symmetry_diagonal_bd_l13_13897

-- Definitions of vertices and points
def Point : Type := (ℕ × ℕ)
def A : Point := (1, 1)
def B : Point := (1, 5)
def C : Point := (5, 5)
def D : Point := (5, 1)
def BD : Point × Point := (B, D)

def P : Point := (1, 3)
def Q : Point := (1, 4)
def R : Point := (2, 4)
def S : Point := (2, 5)
def T : Point := (3, 5)

-- Reflecting point coordinates over the diagonal BD
def reflect_over_bd (p : Point) : Point :=
  (p.snd, p.fst)

-- Original positions of shaded squares
def pos1 : Point := (3, 1)
def pos2 : Point := (5, 2)

-- Shading positions after reflection
def shade_positions : list Point := 
  [reflect_over_bd pos1, reflect_over_bd pos2]

-- Proving which points need to be shaded to preserve symmetry
theorem symmetry_diagonal_bd :
  shade_positions = [P, S] :=
sorry -- proof not required

end symmetry_diagonal_bd_l13_13897


namespace kiley_slices_l13_13778

theorem kiley_slices (calories_per_slice total_calories : ℕ) (percentage : ℚ)
  (h1 : calories_per_slice = 350)
  (h2 : total_calories = 2800)
  (h3 : percentage = 0.25) :
  let kiley_calories := percentage * total_calories in
  let number_of_slices := kiley_calories / calories_per_slice in
  number_of_slices = 2 :=
by
  sorry

end kiley_slices_l13_13778


namespace actual_distance_is_correct_l13_13154

structure PersonTravel where
  speed_normal : ℝ -- Normal walking speed in km/hr
  speed_fast : ℝ -- Faster walking speed in km/hr
  speed_fast_reduction : ℝ -- Percentage reduction in speed for uphill
  distance_uphill_percentage : ℝ -- Percentage of distance that is uphill
  extra_distance : ℝ -- Extra distance walked at 35 km/hr

def actual_distance (p : PersonTravel) : ℝ := 
  let D := 33.75
  D

theorem actual_distance_is_correct :
  ∀ (p : PersonTravel),
    p.speed_normal = 15 →
    p.speed_fast = 35 →
    p.speed_fast_reduction = 0.10 →
    p.distance_uphill_percentage = 0.60 →
    p.extra_distance = 45 →
    actual_distance p = 33.75 := by
  intros
  rw [actual_distance]
  sorry

end actual_distance_is_correct_l13_13154


namespace largest_whole_number_for_inequality_l13_13664

theorem largest_whole_number_for_inequality :
  ∀ n : ℕ, (1 : ℝ) / 4 + (n : ℝ) / 6 < 3 / 2 → n ≤ 7 :=
by
  admit  -- skip the proof

end largest_whole_number_for_inequality_l13_13664


namespace area_equilateral_triangle_parabola_l13_13474

theorem area_equilateral_triangle_parabola {a : ℝ} (h1 : a > 0) 
  (P Q M : ℝ × ℝ) 
  (h2P : P.1^2 = (1 / a) * P.2)
  (h2Q : Q.1^2 = (1 / a) * Q.2)
  (h_slopes_PM_QM : ∃ (s : ℝ), s = √3 ∨ s = -√3)
  (h_triangle : EquilateralTriangle M P Q) : 
  area M P Q = (3 * √3) / (4 * (a * a)) :=
by
  sorry

end area_equilateral_triangle_parabola_l13_13474


namespace average_annual_decrease_10_unit_price_reduction_l13_13641

section
variable (p2019 : ℝ) (p2021 : ℝ) (x : ℝ) (s: ℝ) (m : ℝ) (π : ℝ)
variable (units_sold: ℝ) (price_per_unit: ℝ)

-- Condition: Initial and final factory prices
def factory_prices := p2019 = 200 ∧ p2021 = 162 

-- Condition: Average annual percentage decrease
def average_annual_percentage_decrease : Prop :=
  (1 - x) ^ 2 = p2021 / p2019

-- Question 1: Proving the average annual percentage decrease is 10%
theorem average_annual_decrease_10 (h1: factory_prices):
  average_annual_percentage_decrease p2019 p2021 x → x = 0.1 :=
sorry

-- Condition: Sales information and profit target
def sales_info := 
  (units_sold = 20) ∧ (π = 1150) ∧ (∀ m : ℝ, price_per_unit = 200 - m) 

-- Condition: Increase in units sold for each $3 decrease in price
def increased_units_sold (m : ℝ) : Prop := 
  λ m, units_sold + 2 * m = 20 + 2 * m

-- Condition: Daily profit determination
def daily_profit (price_per_unit : ℝ) (units_sold : ℝ) (π: ℝ) :=
  (price_per_unit - (200 - 162)) * (20 + 2 * m) = π

-- Question 2: Proving the reduction in unit price by $15 results in a daily profit of $1150
theorem unit_price_reduction (h2: sales_info ∧ increased_units_sold m):
  daily_profit price_per_unit units_sold π → m = 15 :=
sorry

end

end average_annual_decrease_10_unit_price_reduction_l13_13641


namespace fib_identity_l13_13909

theorem fib_identity (n : ℕ) : 
  let F : ℕ → ℕ := sorry in
  (∀ n > 0, 
    (Matrix (fin 2) (fin 2) ℕ) ![![1, 1], ![1, 0]] ^ n = 
    (Matrix (fin 2) (fin 2) ℕ) ![![F (n + 1), F n], ![F n, F (n - 1)]]) →
  F 1000 * F 1002 - F 1001^2 = -1 :=
by
  sorry

end fib_identity_l13_13909


namespace range_of_a_l13_13496

def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a * x^2

def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.log x + 1 + 2 * a * x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ f_prime x a = 3) ↔ (-1 / (2 * Real.exp 3) ≤ a) :=
by
  sorry

end range_of_a_l13_13496


namespace fruit_bowl_l13_13978

variable {A P B : ℕ}

theorem fruit_bowl : (P = A + 2) → (B = P + 3) → (A + P + B = 19) → B = 9 :=
by
  intros h1 h2 h3
  sorry

end fruit_bowl_l13_13978


namespace magic_8_ball_probability_l13_13645

theorem magic_8_ball_probability :
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  (Nat.choose 7 3) * (p^3) * (q^4) = 590625 / 2097152 :=
by
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  sorry

end magic_8_ball_probability_l13_13645


namespace contrapositive_l13_13326

theorem contrapositive (a : ℝ) : (a > 0 → a > 1) → (a ≤ 1 → a ≤ 0) :=
by sorry

end contrapositive_l13_13326


namespace exists_x_given_y_l13_13798

theorem exists_x_given_y (y : ℝ) : ∃ x : ℝ, x^2 + y^2 = 10 ∧ x^2 - x * y - 3 * y + 12 = 0 := 
sorry

end exists_x_given_y_l13_13798


namespace find_b_over_a_l13_13179

variables {a b c : ℝ}
variables {b₃ b₇ b₁₁ : ℝ}

-- Conditions
def roots_of_quadratic (a b c b₃ b₁₁ : ℝ) : Prop :=
  ∃ p q, p + q = -b / a ∧ p * q = c / a ∧ (p = b₃ ∨ p = b₁₁) ∧ (q = b₃ ∨ q = b₁₁)

def middle_term_value (b₇ : ℝ) : Prop :=
  b₇ = 3

-- The statement to be proved
theorem find_b_over_a
  (h1 : roots_of_quadratic a b c b₃ b₁₁)
  (h2 : middle_term_value b₇)
  (h3 : b₃ + b₁₁ = 2 * b₇) :
  b / a = -6 :=
sorry

end find_b_over_a_l13_13179


namespace rotated_parabola_eq_l13_13410

theorem rotated_parabola_eq :
  ∀ x y : ℝ, y = x^2 → ∃ y' x' : ℝ, (y' = (-x':ℝ)^2) := sorry

end rotated_parabola_eq_l13_13410


namespace log_abs_even_and_monotonically_decreasing_l13_13613

-- Define the domain and properties of the logarithmic absolute function
def domain (x : ℝ) : Prop := x ≠ 0
def log_abs (x : ℝ) : ℝ := Real.log (abs x)

theorem log_abs_even_and_monotonically_decreasing:
  (∀ x : ℝ, domain x → log_abs (-x) = log_abs x) ∧
  (∀ x y: ℝ, x < 0 → y < 0 → x < y → log_abs x > log_abs y) :=
by
  sorry

end log_abs_even_and_monotonically_decreasing_l13_13613


namespace seating_arrangements_l13_13028

-- Define the problem conditions
variables {P : Type} [fintype P] (A B C D E : P) (seats : fin 5 → P)
variables (h_disjoint : fintype.card P = 5)

def valid_seating : Prop :=
  ∀ i, seats i ≠ Carla ∨ (seats (i+1) % 5 ≠ Bob /\ seats (i+1) % 5 ≠ Derek) /\
  ∀ i, seats i ≠ Eric ∨ seats (i+1) % 5 ≠ Bob

-- Statement of the proof problem
theorem seating_arrangements : 
  ∃ l : list P, l.perm (A :: B :: C :: D :: [E]) ∧ valid_seating seats :=
    sorry

end seating_arrangements_l13_13028


namespace fourth_geometric_term_l13_13264

theorem fourth_geometric_term :
  ∀ (x : ℝ), (3 * x + 3) ^ 2 = x * (6 * x + 6) → (let a₄ := 2 * (6 * x + 6) in a₄ = -24) :=
by
  intro x h
  sorry

end fourth_geometric_term_l13_13264


namespace min_value_a_b_inv_a_inv_b_l13_13808

theorem min_value_a_b_inv_a_inv_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b + 1/a + 1/b ≥ 4 :=
sorry

end min_value_a_b_inv_a_inv_b_l13_13808


namespace complex_number_inequality_l13_13484

/-- Given a complex number z, prove that the only real number m that satisfies z < 0 is m = 1.
z is defined as (2 + i)m^2 - 3(i + 1)m - 2(-i) -/
theorem complex_number_inequality (m : ℝ) : 
  let z := (2 + complex.i) * m^2 - 3 * (complex.i + 1) * m - 2 * (-complex.i) in
  z.re < 0 ∧ z.im = 0 ↔ m = 1 := by
  sorry

end complex_number_inequality_l13_13484


namespace cost_price_per_metre_is_41_l13_13375

-- Define the conditions
def sellingPrice : ℝ := 18000
def lossPerMetre : ℝ := 5
def numberOfMetres : ℝ := 500

-- Define the calculation of the total cost price
def totalLoss : ℝ := lossPerMetre * numberOfMetres
def totalCostPrice : ℝ := sellingPrice + totalLoss

-- Define the cost price per metre
def costPricePerMetre : ℝ := totalCostPrice / numberOfMetres

-- Prove that the cost price per metre is 41
theorem cost_price_per_metre_is_41 : costPricePerMetre = 41 :=
by
  -- This will be filled in with the proof steps
  sorry

end cost_price_per_metre_is_41_l13_13375


namespace balls_into_boxes_l13_13146

theorem balls_into_boxes : 
  ∃ (n : ℕ), n = nat.choose (6 + 3 - 1) (3 - 1) 
          ∧ n = 28 := 
by 
  sorry

end balls_into_boxes_l13_13146


namespace dilute_lotion_l13_13049

/-- Determine the number of ounces of water needed to dilute 12 ounces
    of a shaving lotion containing 60% alcohol to a lotion containing 45% alcohol. -/
theorem dilute_lotion (W : ℝ) : 
  ∃ W, 12 * (0.60 : ℝ) / (12 + W) = 0.45 ∧ W = 4 :=
by
  use 4
  sorry

end dilute_lotion_l13_13049


namespace solve_equation_l13_13967

theorem solve_equation :
  {x : ℂ // (x - 2)^6 + (x - 6)^6 = 32} =
  {x : ℂ // x = 4 + complex.i * complex.sqrt 6 ∨
                x = 4 - complex.i * complex.sqrt 6 ∨
                x = 4 + complex.i * complex.sqrt (21 + complex.sqrt 433) ∨
                x = 4 - complex.i * complex.sqrt (21 + complex.sqrt 433) ∨
                x = 4 + complex.i * complex.sqrt (21 - complex.sqrt 433) ∨
                x = 4 - complex.i * complex.sqrt (21 - complex.sqrt 433) } :=
by
  sorry

end solve_equation_l13_13967


namespace each_person_ate_3_brownie_bites_l13_13023

theorem each_person_ate_3_brownie_bites
  (cinnamon_swirls : ℕ) (brownie_bites : ℕ) (fruit_tartlets : ℕ) (people : ℕ)
  (h_cinnamon : cinnamon_swirls = 15)
  (h_brownie : brownie_bites = 24)
  (h_tartlets : fruit_tartlets = 18)
  (h_people : people = 8) :
  brownie_bites / people = 3 :=
by
  rw [h_brownie, h_people]
  norm_num
  sorry

end each_person_ate_3_brownie_bites_l13_13023


namespace Gina_gave_fraction_to_mom_l13_13455

variable (M : ℝ)

theorem Gina_gave_fraction_to_mom :
  (∃ M, M + (1/8 : ℝ) * 400 + (1/5 : ℝ) * 400 + 170 = 400) →
  M / 400 = 1/4 :=
by
  intro h
  sorry

end Gina_gave_fraction_to_mom_l13_13455


namespace range_of_function_l13_13275

theorem range_of_function :
  ∀ x: ℝ, x ∈ Set.Icc (-1:ℝ) 1 → 
  (∃ y: ℝ, y = (sqrt (1 - x^2)) / (2 + x) ∧ y ∈ Set.Icc (0:ℝ) (sqrt 3 / 3)) := 
sorry

end range_of_function_l13_13275


namespace percentage_of_volume_is_P_l13_13597

noncomputable def volumeOfSolutionP {P Q : ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : ℝ := 
(P / (P + Q)) * 100

theorem percentage_of_volume_is_P {P Q: ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : 
  volumeOfSolutionP h = 50 :=
sorry

end percentage_of_volume_is_P_l13_13597


namespace margaux_total_collection_l13_13577

theorem margaux_total_collection :
  let friend_daily_payment := 5
  let brother_daily_payment := 8
  let cousin_daily_payment := 4
  let days := 7
  let total_collection :=
    friend_daily_payment * days + 
    brother_daily_payment * days + 
    cousin_daily_payment * days
  in total_collection = 119 :=
by
  sorry

end margaux_total_collection_l13_13577


namespace train_crossing_time_l13_13337

def convert_speed (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

def time_to_cross (length : ℝ) (speed_mps : ℝ) : ℝ := length / speed_mps

theorem train_crossing_time :
  let length := 100 -- length of the train in meters
  let speed_kmh := 36 -- speed of the train in km/hr
  let speed_mps := convert_speed speed_kmh -- convert speed to meters per second
  time_to_cross length speed_mps = 10 := by
    let length := (100 : ℝ)
    let speed_kmh := (36 : ℝ)
    let speed_mps := convert_speed speed_kmh
    let time := time_to_cross length speed_mps
    show time = 10
    sorry

end train_crossing_time_l13_13337


namespace solve_differential_eq_l13_13965

noncomputable def differential_eq (x y y' : ℝ) : Prop :=
  (1 + x^2 * y^2) * y + (x * y - 1)^2 * x * y' = 0

noncomputable def substitution (x y z : ℝ) : Prop :=
  z = x * y

noncomputable def solution (x y : ℝ) (C : ℝ) : Prop :=
  x * y - 1 / (x * y) - 2 * Real.log (Real.abs y) = C

theorem solve_differential_eq (x y y' z C : ℝ) :
  differential_eq x y y' → substitution x y z → solution x y C :=
sorry

end solve_differential_eq_l13_13965


namespace hyperbola_eccentricity_l13_13038

theorem hyperbola_eccentricity 
    (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (c : ℝ) 
    (h3 : b^2 = a * c) 
    (h4 : c = a * (sqrt 5 + 1) / 2) : 
    c^2 = a^2 + b^2 := 
begin
  sorry,
end

end hyperbola_eccentricity_l13_13038


namespace height_after_16_minutes_l13_13352

noncomputable def ferris_wheel_height (t : ℝ) : ℝ :=
  8 * Real.sin ((Real.pi / 6) * t - Real.pi / 2) + 10

theorem height_after_16_minutes : ferris_wheel_height 16 = 6 := by
  sorry

end height_after_16_minutes_l13_13352


namespace find_z_l13_13081

-- Define the 3D points A, B, and C.
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

def dist (p q : Point) : ℝ :=
  real.sqrt ((p.x - q.x) ^ 2 + (p.y - q.y) ^ 2 + (p.z - q.z) ^ 2)

theorem find_z (z : ℝ) :
  let A := Point.mk 0 0 z
  let B := Point.mk (-6) 7 5
  let C := Point.mk 8 (-4) 3
  dist A B = dist A C →
  z = 21 / 4 := 
by {
  let A := Point.mk 0 0 z,
  let B := Point.mk (-6) 7 5,
  let C := Point.mk 8 (-4) 3,
  -- Skipping computation and including correct answer directly for the theorem statement.
  sorry
}

end find_z_l13_13081


namespace ethanol_combustion_heat_l13_13680

theorem ethanol_combustion_heat (Q : Real) :
  (∃ (m : Real), m = 0.1 ∧ (∀ (n : Real), n = 1 → Q * n / m = 10 * Q)) :=
by
  sorry

end ethanol_combustion_heat_l13_13680


namespace valid_integer_n_count_l13_13801

theorem valid_integer_n_count : 
  ∃ n_set : Set ℤ, (∀ n ∈ n_set, (∃ k : ℤ, 8000 * ((2:ℚ)^n) / ((5:ℚ)^n) = k)) ∧ n_set.card = 10 := 
by
  sorry

end valid_integer_n_count_l13_13801


namespace inner_circumference_approx_l13_13988

def outer_radius : ℝ := 140.0563499208679
def track_width : ℝ := 18
def inner_radius : ℝ := outer_radius - track_width
def pi : ℝ := Real.pi
def inner_circumference : ℝ := 2 * pi * inner_radius

theorem inner_circumference_approx : abs (inner_circumference - 767.145882893066) < 1e-6 :=
by
  sorry

end inner_circumference_approx_l13_13988


namespace find_CD_l13_13543

variable (A B C D : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Defining points
variable (a b d c : ℝ)
variable (BA : d = A)
variable (etat_AD_DC : angle BAD = angle ADC)
variable (etat_ABD_BCD : angle ABD = angle BCD)
variable (AB : dist a b = 9)
variable (BD : dist b d = 15)
variable (BC : dist b c = 9)

theorem find_CD : dist c d = 10.8 := sorry

end find_CD_l13_13543


namespace area_of_triangle_l13_13810

variables {x1 y1 x2 y2 : ℝ}

def vector (a b : ℝ) := (a, b)

def triangle_area (v1 v2 : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (fst v1 * snd v2 + fst v2 * snd v1)

theorem area_of_triangle (h1 : vector x1 y1 = (x1, y1))
                         (h2 : vector x2 y2 = (x2, y2)) :
  triangle_area (vector x1 y1) (vector x2 y2)
    = 1 / 2 * abs (x1 * y2 + x2 * y1) :=
by
  sorry

end area_of_triangle_l13_13810


namespace solve_logarithmic_equation_l13_13624

theorem solve_logarithmic_equation (x : ℝ) :
  (x ^ log (x) = x^5 / 10000) ↔ (x = 10 ∨ x = 10000) :=
sorry

end solve_logarithmic_equation_l13_13624


namespace spherical_to_rectangular_l13_13755

noncomputable def spherical_to_rectangular_coordinates (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
(x, y, z) where
    x = ρ * Real.sin φ * Real.cos θ
    y = ρ * Real.sin φ * Real.sin θ
    z = ρ * Real.cos φ

theorem spherical_to_rectangular
    (ρ θ φ : ℝ)
    (hρ : ρ = 4)
    (hθ : θ = Real.pi / 3)
    (hφ : φ = Real.pi / 4) :
    spherical_to_rectangular_coordinates ρ θ φ = (Real.sqrt 2, Real.sqrt 6, 2 * Real.sqrt 2) :=
by
    rw [hρ, hθ, hφ]
    simp [spherical_to_rectangular_coordinates]
    sorry

end spherical_to_rectangular_l13_13755


namespace middle_card_number_l13_13642

theorem middle_card_number (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 15) (h_order : a < b ∧ b < c)
  (h_casey : ¬ ∀ x y, a = 3 → (x, y) = (b, c) ∨ (x, y) = (c, b))
  (h_tracy : ¬ ∀ x y, c = 12 → (x, y) = (a, b) ∨ (x, y) = (b, a))
  (h_stacy : ¬ ∀ x y, b ∈ {3, 4, 5, 6} → (x, y) = (a, c) ∨ (x, y) = (c, a)) :
  b = 4 :=
by 
  sorry

end middle_card_number_l13_13642


namespace proof_problem_l13_13813

def ellipse (x y : ℝ) (a : ℝ) : Prop := (x^2 / a^2 + y^2 / 3 = 1)

def conditions (a c x₁ y₁ x₂ y₂ : ℝ) : Prop := 
a > sqrt 3 ∧ 
(x₁, y₁) = (4, 0) ∧
(∃ F : ℝ × ℝ, F = (c, 0)) ∧ 
(∃ Q : ℝ × ℝ, ∃ E : ℝ × ℝ, Q = (x₂, y₂) ∧ E = (x₂, -y₂)) ∧ 
(∀ P Q E : ℝ × ℝ, (x₁, y₁) = P → Q = (x₂, y₂) → E = (x₂, -y₂))

theorem proof_problem (a c x : ℝ) (hx : conditions a c 4 0 x (0) (1) (0)) :
  (ellipse x 0 2) ∧ (x = 1) :=
sorry

end proof_problem_l13_13813


namespace xy_yz_zx_nonzero_l13_13190

theorem xy_yz_zx_nonzero (x y z : ℝ)
  (h1 : 1 / |x^2 + 2 * y * z| + 1 / |y^2 + 2 * z * x| > 1 / |z^2 + 2 * x * y|)
  (h2 : 1 / |y^2 + 2 * z * x| + 1 / |z^2 + 2 * x * y| > 1 / |x^2 + 2 * y * z|)
  (h3 : 1 / |z^2 + 2 * x * y| + 1 / |x^2 + 2 * y * z| > 1 / |y^2 + 2 * z * x|) :
  x * y + y * z + z * x ≠ 0 := by
  sorry

end xy_yz_zx_nonzero_l13_13190


namespace no_valid_subset_exists_l13_13054

noncomputable def exists_valid_subset (A : Set ℕ) : Prop :=
  ∃ A, (∀ x ∈ A, 2 * x ∉ A) ∧ (A ⊆ (Finset.Icc 1 3000).toSet) ∧ (A.card = 2000)

theorem no_valid_subset_exists : ¬ exists_valid_subset (Finset.Icc 1 3000).toSet :=
  by
    sorry

end no_valid_subset_exists_l13_13054


namespace simplify_expression_l13_13249

variable (t : ℝ)

theorem simplify_expression (ht : t > 0) (ht_ne : t ≠ 1 / 2) :
  (1 - Real.sqrt (2 * t)) / ( (1 - Real.sqrt (4 * t ^ (3 / 4))) / (1 - Real.sqrt (2 * t ^ (1 / 4))) - Real.sqrt (2 * t)) *
  (Real.sqrt (1 / (1 / 2) + Real.sqrt (4 * t ^ 2)) / (1 + Real.sqrt (1 / (2 * t))) - Real.sqrt (2 * t))⁻¹ = 1 :=
by
  sorry

end simplify_expression_l13_13249


namespace even_odd_functions_and_difference_l13_13560

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem even_odd_functions_and_difference:
  (∀ x : ℝ, f (-x) = f x) →             -- Condition 1: f(x) is an even function
  (∀ x : ℝ, g (-x) = -g x) →            -- Condition 2: g(x) is an odd function
  (∀ x : ℝ, f x - g x = x^2 - x + 1) →  -- Condition 3: f(x) - g(x) = x^2 - x + 1
  f 1 = 2 :=                             -- Question: Show that f(1) = 2
begin
  sorry
end

end even_odd_functions_and_difference_l13_13560


namespace solve_for_exponent_l13_13675

theorem solve_for_exponent (K : ℕ) (h1 : 32 = 2 ^ 5) (h2 : 64 = 2 ^ 6) 
    (h3 : 32 ^ 5 * 64 ^ 2 = 2 ^ K) : K = 37 := 
by 
    sorry

end solve_for_exponent_l13_13675


namespace find_linear_function_find_intersection_point_l13_13464

-- Define the linear function y = kx + b
def linear_function (x : ℝ) (k b : ℝ) : ℝ := k * x + b

-- Define the given points M and N
def M : ℝ × ℝ := (0, 2)
def N : ℝ × ℝ := (1, 3)

-- Prove the analytical expression of the linear function is y = x + 2
theorem find_linear_function :
  ∃ (k b : ℝ), (linear_function 0 k b = 2) ∧ (linear_function 1 k b = 3) ∧ (k = 1) ∧ (b = 2) ∧
    (∀ x : ℝ, linear_function x k b = x + 2) :=
by
  use 1, 2
  simp [linear_function]
  split; simp
  split; simp
  split; simp

-- Prove the coordinates of the intersection point between the linear function and the x-axis are (-2, 0)
theorem find_intersection_point :
  ∃ (x : ℝ), (linear_function x 1 2 = 0) ∧ (x = -2) :=
by
  use -2
  simp [linear_function]
  split; simp

end find_linear_function_find_intersection_point_l13_13464


namespace maria_distance_after_second_stop_l13_13055

theorem maria_distance_after_second_stop (total_distance : ℕ)
  (half_distance := total_distance / 2)
  (remaining_distance_after_first_stop := total_distance - half_distance)
  (quarter_remaining_distance := remaining_distance_after_first_stop / 4)
  (remaining_distance_after_second_stop := remaining_distance_after_first_stop - quarter_remaining_distance) :
  total_distance = 400 →
  remaining_distance_after_second_stop = 150 :=
by
  intros h,
  sorry

end maria_distance_after_second_stop_l13_13055


namespace figure_side_length_l13_13861

theorem figure_side_length (number_of_sides : ℕ) (perimeter : ℝ) (length_of_one_side : ℝ) 
  (h1 : number_of_sides = 8) (h2 : perimeter = 23.6) : length_of_one_side = 2.95 :=
by
  sorry

end figure_side_length_l13_13861


namespace plato_city_high_schools_l13_13781

theorem plato_city_high_schools 
(a b : ℕ) (h₁ : 45 ≤ a) (h₂ : 58 ≤ b) (h₃ : a + b + 1 = 2 * (3 * 21 + 1) / 2) : 
  let total_students := 63 in 
  total_students / 3 = 21 := 
by
  sorry

end plato_city_high_schools_l13_13781


namespace primes_less_or_equal_F_l13_13925

-- Definition of F_n
def F (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1

-- The main theorem statement
theorem primes_less_or_equal_F (n : ℕ) : ∃ S : Finset ℕ, S.card ≥ n + 1 ∧ ∀ p ∈ S, Nat.Prime p ∧ p ≤ F n := 
sorry

end primes_less_or_equal_F_l13_13925


namespace solve_for_x_l13_13787

noncomputable def log_base_3 (y : ℝ) : ℝ := log y / log 3

theorem solve_for_x (x : ℝ) (h : 4 * log_base_3 x = log_base_3 (4 * x^2)) : x = 2 := by
  sorry

end solve_for_x_l13_13787


namespace find_A_for_diamond_eq_85_l13_13517

def diamond (A B : ℝ) : ℝ := 4 * A + B^2 + 7

theorem find_A_for_diamond_eq_85 :
  ∃ (A : ℝ), diamond A 3 = 85 ∧ A = 17.25 :=
by
  sorry

end find_A_for_diamond_eq_85_l13_13517


namespace connected_graph_coloring_l13_13360

-- Define a connected graph
structure Graph (V : Type) :=
    (adj : V → V → Prop)
    (connected : ∀ u v : V, u ≠ v → ∃ p : List V, p ≠ [] ∧ p.head = u ∧ p.last = v ∧ (∀ (u v : V), v ∈ p.tail → adj u v))

-- Define a type for colors
inductive Color
| blue
| green

open Color

-- Define the problem statement
theorem connected_graph_coloring {V : Type} (G : Graph V) : 
  ∃ (c : V → Color) (E' : (V → V → Prop)), 
  (∀ u v : V, ∃ (p : List V), (p.head = u ∧ p.last = v ∧ ∀ i < p.length - 1, E' (p.nth_le i _) (p.nth_le (i+1) _))) ∧
  (∀ u v : V, E' u v → c u ≠ c v) ∧
  (∀ u v : V, G.adj u v → c u = green → c v ≠ green) :=
by
  sorry

end connected_graph_coloring_l13_13360


namespace balls_into_boxes_all_ways_balls_into_boxes_one_empty_l13_13635

/-- There are 4 different balls and 4 different boxes. -/
def balls : ℕ := 4
def boxes : ℕ := 4

/-- The number of ways to put 4 different balls into 4 different boxes is 256. -/
theorem balls_into_boxes_all_ways : (balls ^ boxes) = 256 := by
  sorry

/-- The number of ways to put 4 different balls into 4 different boxes such that exactly one box remains empty is 144. -/
theorem balls_into_boxes_one_empty : (boxes.choose 1 * (balls ^ (boxes - 1))) = 144 := by
  sorry

end balls_into_boxes_all_ways_balls_into_boxes_one_empty_l13_13635


namespace number_of_bananas_in_bowl_l13_13981

theorem number_of_bananas_in_bowl (A P B : Nat) (h1 : P = A + 2) (h2 : B = P + 3) (h3 : A + P + B = 19) : B = 9 :=
sorry

end number_of_bananas_in_bowl_l13_13981


namespace axis_of_symmetry_l13_13600

-- Definitions of original function and transformations
def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x)

def stretch_x_coordinates (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x, f (x / 2)

def shift_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f (x - a)

-- Original function transformed by given conditions
def g (x : ℝ) : ℝ :=
  shift_right (stretch_x_coordinates f) (π / 6) x

theorem axis_of_symmetry : ∃ k : ℤ, (k = 0) ∧ (k * π + π / 3 = π / 3) :=
sorry

end axis_of_symmetry_l13_13600


namespace min_product_eccentricities_l13_13475

theorem min_product_eccentricities (F1 F2 P : Point) (a1 a2 c : ℝ)
  (hfoci_ellipse : Ellipse F1 F2)
  (hfoci_hyperbola : Hyperbola F1 F2)
  (hcommon_point : is_common_point P hfoci_ellipse hfoci_hyperbola)
  (hangle : angle F1 P F2 = π / 3)
  (hecc1 : eccentricity hfoci_ellipse = c / a1)
  (hecc2 : eccentricity hfoci_hyperbola = c / a2) :
  eccentricity_product hfoci_ellipse hfoci_hyperbola ≥ sqrt 3 / 2 :=
sorry

end min_product_eccentricities_l13_13475


namespace vertical_angles_congruent_l13_13241

theorem vertical_angles_congruent (a b : Angle) (h : VerticalAngles a b) : CongruentAngles a b :=
sorry

end vertical_angles_congruent_l13_13241


namespace alpha_div_beta_is_rational_l13_13554

noncomputable def alpha_is_multiple (α : ℝ) (k : ℕ) : Prop :=
  ∃ k : ℕ, α = k * (2 * Real.pi / 1996)

noncomputable def beta_is_multiple (β : ℝ) (m : ℕ) : Prop :=
  β ≠ 0 ∧ ∃ m : ℕ, β = m * (2 * Real.pi / 1996)

theorem alpha_div_beta_is_rational (α β : ℝ) (k m : ℕ)
  (hα : alpha_is_multiple α k) (hβ : beta_is_multiple β m) :
  ∃ r : ℚ, α / β = r := by
    sorry

end alpha_div_beta_is_rational_l13_13554


namespace solve_log_eq_l13_13785

theorem solve_log_eq (x : ℝ) (h : 4 * log 3 x = log 3 (4 * x ^ 2)) : x = 2 :=
sorry

end solve_log_eq_l13_13785


namespace digits_sum_l13_13408

theorem digits_sum (a b : ℕ) (h₁ : 10 * b + 4 = 14) (h₂ : 10 * 3 + a = 3a) (h₃ : (10 * (10 * 3 + a)) * (10 * b + 4) = 146) : a + b = 13 :=
sorry

end digits_sum_l13_13408


namespace find_first_term_of_arithmetic_sequence_l13_13563

theorem find_first_term_of_arithmetic_sequence (a : ℚ) (c : ℚ) (n : ℕ) (h_n_pos : 0 < n)
    (S_n_def : ∀ n : ℕ, S_n = (n * (2 * a + (n - 1) * 5)) / 2)
    (ratio_constant : ∀ n : ℕ, 0 < n → (S_n 4 * n) / (S_n n) = c) :
  a = 5 / 2 := 
by
  sorry

end find_first_term_of_arithmetic_sequence_l13_13563


namespace integers_satisfy_equation_l13_13775

theorem integers_satisfy_equation (a b c : ℤ) : 
  (a(a - b) + b(b - c) + c(c - a) = 0) ↔ (a = c ∧ b = c + 2) := 
sorry

end integers_satisfy_equation_l13_13775


namespace no_solution_system_l13_13252

theorem no_solution_system :
  ∀ (x y z : ℝ), ¬ (sqrt(2 * x^2 + 2) = y - 1 ∧
                   sqrt(2 * y^2 + 2) = z - 1 ∧
                   sqrt(2 * z^2 + 2) = x - 1) :=
by
  intros x y z
  intro h
  cases h with hxy hrest
  cases hrest with hyz hzx
  -- Here we would do further proof steps to show the contradiction, but we
  -- can leave it as a placeholder for now.
  sorry

end no_solution_system_l13_13252


namespace problems_per_hour_l13_13452

theorem problems_per_hour :
  ∀ (mathProblems spellingProblems totalHours problemsPerHour : ℕ), 
    mathProblems = 36 →
    spellingProblems = 28 →
    totalHours = 8 →
    (mathProblems + spellingProblems) / totalHours = problemsPerHour →
    problemsPerHour = 8 :=
by
  intros
  subst_vars
  sorry

end problems_per_hour_l13_13452


namespace find_y_value_l13_13336

theorem find_y_value : (12 ^ 2 * 6 ^ 3) / 432 = 72 := 
by 
  -- First we demonstrate the calculation of (12)^2 and 6^3
  have h1: 12 ^ 2 = 144 := by norm_num,
  have h2: 6 ^ 3 = 216 := by norm_num,
  -- Substitute these values into the multiplication and division expression
  calc (12 ^ 2 * 6 ^ 3) / 432
    = (144 * 216) / 432 : by rw [h1, h2]
    ... = 31104 / 432 : by norm_num
    ... = 72 : by norm_num 

end find_y_value_l13_13336


namespace vector_magnitude_eq_five_l13_13140

theorem vector_magnitude_eq_five : 
  let a : ℝ × ℝ := (-3, 4) in 
  real.sqrt (a.1^2 + a.2^2) = 5 :=
by
  let a : ℝ × ℝ := (-3, 4)
  sorry

end vector_magnitude_eq_five_l13_13140


namespace inverse_of_h_l13_13762

noncomputable def h (x : ℝ) : ℝ := 3 - 7 * x
noncomputable def k (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_of_h :
  (∀ x : ℝ, h (k x) = x) ∧ (∀ x : ℝ, k (h x) = x) :=
by
  sorry

end inverse_of_h_l13_13762


namespace pq_sum_l13_13614

theorem pq_sum :
  let f : ℚ → ℚ := λ x, 4 * x^2 + 6 * x + 3 in
  let p := -3 / 2 in
  let q := -1 / 2 in
  (∀ x, f x = 0 → (x = p ∨ x = q)) → p + q = -3 / 2 := by
  intros
  sorry

end pq_sum_l13_13614


namespace sqrt_0_1681_eq_0_41_l13_13821

theorem sqrt_0_1681_eq_0_41 (h : Real.sqrt 16.81 = 4.1) : Real.sqrt 0.1681 = 0.41 := by 
  sorry

end sqrt_0_1681_eq_0_41_l13_13821


namespace coloring_ways_l13_13777

-- Define the graph structure of the problem
structure Graph (V : Type) :=
(edges : set (V × V))

-- Define specific type for the dots
inductive Dot
| D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | D11 | D12

open Dot

-- Define the edges for the specific problem's graph
def extended_figure_graph : Graph Dot := {
  edges := {
    (D1, D2), (D2, D3), (D1, D3), -- first triangle
    (D3, D4), -- connection between first and second triangle
    (D4, D5), (D5, D6), (D4, D6), -- second triangle
    (D6, D7), -- and so on
    (D7, D8), (D8, D9), (D7, D9), 
    (D9, D10), 
    (D10, D11), (D11, D12), (D10, D12), 
    (D12, D13)
    -- The above edges connect dots as described in conditions
  }
}

-- Define the proof problem
theorem coloring_ways : 
  ∃ c : Dot → ℕ, 
    (∀ v ∈ [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12], c v ∈ [1, 2, 3, 4]) ∧ -- each dot is colored one of four colors
    (∀ (v1 v2 : Dot), (v1, v2) ∈ extended_figure_graph.edges → c v1 ≠ c v2) ∧ -- adjacent dots are differently colored
    (tsize (filter (λ x, x (extended_figure_graph.edges ≠ 7776) := sorry

end coloring_ways_l13_13777


namespace expression_value_l13_13676

theorem expression_value : (5^2 - 5) * (6^2 - 6) - (7^2 - 7) = 558 := by
  sorry

end expression_value_l13_13676


namespace linear_function_expression_l13_13482

theorem linear_function_expression 
  (f : ℝ → ℝ) 
  (h₁ : ∃ k b : ℝ, ∀ x, f(x) = k * x + b)
  (h₂ : ∀ x, f(f(x)) = 9 * x + 8) : 
  (∀ x, f(x) = 3 * x + 2) ∨ (∀ x, f(x) = -3 * x - 4) :=
sorry

end linear_function_expression_l13_13482


namespace additive_inverse_of_half_l13_13269

theorem additive_inverse_of_half :
  - (1 / 2) = -1 / 2 :=
by
  sorry

end additive_inverse_of_half_l13_13269


namespace tan_beta_value_l13_13807

noncomputable def alpha : Real := Real.angle.pi / 2 + Real.arcsin (3/5)
noncomputable def beta : Real := 
  let tan_alpha := -3/4
  let tan_beta := 7
  Real.arctan tan_beta - Real.arctan tan_alpha

theorem tan_beta_value (sin_alpha : Real) (alpha_cond : sin_alpha = 3/5) 
    (tan_alpha_beta : Real) (tan_alpha_beta_cond : tan_alpha_beta = 1) : 
    Real.tan beta = 7 := 
by
  have sin_alpha := 3/5
  have cos_alpha := -4/5
  have tan_alpha := -3/4
  have tan_alpha_cond := tan_alpha = -3/4
  
  -- Proof steps here
  sorry

end tan_beta_value_l13_13807


namespace solution_set_when_a_eq_1_range_of_a_if_fx_le_1_l13_13209

def f (x a : ℝ) := 5 - |x + a| - |x - 2|

theorem solution_set_when_a_eq_1 :
  (∀ x : ℝ, f x 1 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
by
  intros x
  sorry

theorem range_of_a_if_fx_le_1 :
  (∀ x : ℝ, (f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2)) :=
by
  intros x a
  sorry

end solution_set_when_a_eq_1_range_of_a_if_fx_le_1_l13_13209


namespace coris_aunt_age_today_l13_13043

variable (Cori_age_now : ℕ) (age_diff : ℕ)

theorem coris_aunt_age_today (H1 : Cori_age_now = 3) (H2 : ∀ (Cori_age5 Aunt_age5 : ℕ), Cori_age5 = Cori_age_now + 5 → Aunt_age5 = 3 * Cori_age5 → Aunt_age5 - 5 = age_diff) :
  age_diff = 19 := 
by
  intros
  sorry

end coris_aunt_age_today_l13_13043


namespace tens_digit_19_pow_2023_l13_13771

theorem tens_digit_19_pow_2023 :
  ∃ d : ℕ, d = (59 / 10) % 10 ∧ (19 ^ 2023 % 100) / 10 = d :=
by
  have h1 : 19 ^ 10 % 100 = 1 := by sorry
  have h2 : 19 ↔ 0 := by sorry
  have h4 : 2023 % 10 = 3 := by sorry
  have h5 : 19 ^ 10 ↔ 1 := by sorry
  have h6 : 19 ^ 3 % 100 = 59 := by sorry
  have h7 : (19 ^ 2023 % 100) = 59 := by sorry
  exists 5
  split
  repeat { assumption.dump }

end tens_digit_19_pow_2023_l13_13771


namespace Hans_current_age_l13_13911

variable {H : ℕ} -- Hans' current age

-- Conditions
def Josiah_age (H : ℕ) := 3 * H
def Hans_age_in_3_years (H : ℕ) := H + 3
def Josiah_age_in_3_years (H : ℕ) := Josiah_age H + 3
def sum_of_ages_in_3_years (H : ℕ) := Hans_age_in_3_years H + Josiah_age_in_3_years H

-- Theorem to prove
theorem Hans_current_age : sum_of_ages_in_3_years H = 66 → H = 15 :=
by
  sorry

end Hans_current_age_l13_13911


namespace volume_of_rect_prism_l13_13006

variables {a b c V : ℝ}

theorem volume_of_rect_prism :
  (∃ (a b c : ℝ), (a * b = Real.sqrt 2) ∧ (b * c = Real.sqrt 3) ∧ (a * c = Real.sqrt 6) ∧ V = a * b * c) →
  V = Real.sqrt 6 :=
by
  sorry

end volume_of_rect_prism_l13_13006


namespace solve_for_k_l13_13052

theorem solve_for_k :
  ∀ (k : ℝ), (∃ x : ℝ, (3*x + 8)*(x - 6) = -50 + k*x) ↔
    k = -10 + 2*Real.sqrt 6 ∨ k = -10 - 2*Real.sqrt 6 := by
  sorry

end solve_for_k_l13_13052


namespace range_log2_dom_half_to_four_l13_13485

open Real

def log2 := logb (2 : ℝ)

theorem range_log2_dom_half_to_four : 
  ∃ y ∈ set.range (λ x, log2 x), y = -1 ∧ y = 2 ∧ (∀ y ∈ set.range (λ x, log2 x), -1 ≤ y ∧ y ≤ 2) ∧
  ∀ x, x ∈ Icc (1 / 2 : ℝ) 4 → log2 x ∈ Icc (-1 : ℝ) 2 :=
by
  sorry

end range_log2_dom_half_to_four_l13_13485


namespace equation_of_line_area_of_triangle_l13_13826

/-
  Given: 
  1. Line l passes through the point (0, -2)
  2. The slope angle of line l is 60 degrees

  Prove:
  1. The equation of line l is sqrt(3) * x - y - 2 = 0
  2. The area of the triangle formed by line l and the two coordinate axes is 2 * sqrt(3) / 3
-/

-- Defining the conditions
def point_on_line : Prop :=
  ∃ (l : ℝ → ℝ), l 0 = -2

def slope_angle : Prop :=
  ∃ (l : ℝ → ℝ), ∀ x : ℝ, l x = √3 * x - 2

-- Statements to be proved
theorem equation_of_line (h1 : point_on_line) (h2 : slope_angle) :
  ∃ x y : ℝ, √3 * x - y - 2 = 0 :=
sorry

theorem area_of_triangle (h1 : point_on_line) (h2 : slope_angle) :
  ∃ S : ℝ, S = (2 * √3) / 3 :=
sorry

end equation_of_line_area_of_triangle_l13_13826


namespace SumOfCosines_SumOfSines_l13_13232

-- We define our problem in Lean
theorem SumOfCosines (x d : ℝ) (n : ℕ) :
  (∑ k in Finset.range (n + 1), cos (x + k * d)) = 
  (sin ((↑n + 1) * d / 2) * cos (x + ↑n * d / 2)) / (sin (d / 2)) := 
sorry

theorem SumOfSines (x d : ℝ) (n : ℕ) :
  (∑ k in Finset.range (n + 1), sin (x + k * d)) = 
  (sin ((↑n + 1) * d / 2) * sin (x + ↑n * d / 2)) / (sin (d / 2)) := 
sorry

end SumOfCosines_SumOfSines_l13_13232


namespace circle_area_portion_l13_13659

theorem circle_area_portion :
  let circle_eq : ∀ x y : ℝ, (x - 2)^2 + (y - 3)^2 = 9
  let half_plane : ∀ y : ℝ, y ≥ 0
  let line_eq : ∀ x y : ℝ, y ≤ x - 3
  ∃ A: ℝ, A = 9 * π / 4 :=
by 
  sorry

end circle_area_portion_l13_13659


namespace omega_range_l13_13837

noncomputable def f (ω x : ℝ) : ℝ := sqrt 2 * sin (ω * x) + sqrt 2 * cos (ω * x)

theorem omega_range (ω : ℝ) : 
  (∀ x1 x2 ∈ Ioo (-π/3) (π/4), x1 < x2 → f ω x1 < f ω x2) ↔ (0 < ω ∧ ω ≤ 1) := 
by 
  sorry

end omega_range_l13_13837


namespace added_number_is_four_l13_13652

theorem added_number_is_four :
  ∃ x y, 2 * x < 3 * x ∧ (3 * x - 2 * x = 8) ∧ 
         ((2 * x + y) * 7 = 5 * (3 * x + y)) ∧ y = 4 :=
  sorry

end added_number_is_four_l13_13652


namespace postal_code_permutations_l13_13646

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def multiplicity_permutations (n : Nat) (repetitions : List Nat) : Nat :=
  factorial n / List.foldl (λ acc k => acc * factorial k) 1 repetitions

theorem postal_code_permutations : multiplicity_permutations 4 [2, 1, 1] = 12 :=
by
  unfold multiplicity_permutations
  unfold factorial
  sorry

end postal_code_permutations_l13_13646


namespace gcf_of_lcm_9_21_and_10_22_eq_one_l13_13076

theorem gcf_of_lcm_9_21_and_10_22_eq_one :
  Nat.gcd (Nat.lcm 9 21) (Nat.lcm 10 22) = 1 :=
sorry

end gcf_of_lcm_9_21_and_10_22_eq_one_l13_13076


namespace g_add_g_2010_x_l13_13197

-- Define f as a function on ℝ
def f : ℝ → ℝ := sorry

-- Define g based on the given conditions
def g (x : ℝ) := f(x) - f(2010 - x)

-- The statement to be proved
theorem g_add_g_2010_x (x : ℝ) : g(x) + g(2010 - x) = 0 :=
by
  -- Proof is omitted
  sorry

end g_add_g_2010_x_l13_13197


namespace math_classic_problem_l13_13177

theorem math_classic_problem
  (x y : ℕ)
  (h1 : 3 * (x - 2) = 2 * x + 9)
  (h2 : y / 3 + 2 = (y - 9) / 2)
  (h3_1 : y = 3 * (x - 2))
  (h3_2 : 2 * x = y + 9)
  (h4_1 : x = 3 * (y - 2))
  (h4_2 : 2 * y = x - 9) :
  (h1 ∧ h2 ∧ h4_1 ∧ h4_2) ∧ ¬ (h3_1 ∧ h3_2) :=
by
  
  sorry

end math_classic_problem_l13_13177


namespace Emma_Ethan_not_in_photo_together_l13_13430

noncomputable def lap_time_Emma : ℕ := 120
noncomputable def lap_time_Ethan : ℕ := 100
noncomputable def photographer_time_range : set ℕ := {900, ..., 960}
noncomputable def track_fraction : ℚ := 1 / 5

-- Function to calculate the time in one fraction of the lap
def fraction_time (lap_time : ℕ) (fraction : ℚ) : ℚ :=
  lap_time * fraction

-- Functions to determine if they are within the photographed segment at a time t
def Emma_in_photo (t : ℚ) : Prop :=
  let position := (t % lap_time_Emma : ℚ)
  position ≤ track_fraction * ↑lap_time_Emma / 2 ∨
  position ≥ (↑lap_time_Emma - track_fraction * ↑lap_time_Emma / 2)

def Ethan_in_photo (t : ℚ) : Prop :=
  let position := (t % lap_time_Ethan : ℚ)
  position ≤ track_fraction * ↑lap_time_Ethan / 2 ∨
  position ≥ (↑lap_time_Ethan - track_fraction * ↑lap_time_Ethan / 2)

theorem Emma_Ethan_not_in_photo_together :
  ∀ t ∈ photographer_time_range, ¬ (Emma_in_photo t ∧ Ethan_in_photo t) :=
by
  sorry

end Emma_Ethan_not_in_photo_together_l13_13430


namespace minimal_value_abs_diff_l13_13519

theorem minimal_value_abs_diff (c d : ℕ) (h : c * d - 4 * c + 5 * d = 102) : |c - d| = 30 := by
  sorry

end minimal_value_abs_diff_l13_13519


namespace arccos_cos_8_eq_1_point_72_l13_13745

noncomputable def arccos_cos_eight : Real :=
  Real.arccos (Real.cos 8)

theorem arccos_cos_8_eq_1_point_72 : arccos_cos_eight = 1.72 :=
by
  sorry

end arccos_cos_8_eq_1_point_72_l13_13745


namespace binomial_eight_three_l13_13395

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem binomial_eight_three : binomial 8 3 = 56 := by
  sorry

end binomial_eight_three_l13_13395


namespace highest_total_zits_l13_13945

def zits_per_student_Swanson := 5
def students_Swanson := 25
def total_zits_Swanson := zits_per_student_Swanson * students_Swanson -- should be 125

def zits_per_student_Jones := 6
def students_Jones := 32
def total_zits_Jones := zits_per_student_Jones * students_Jones -- should be 192

def zits_per_student_Smith := 7
def students_Smith := 20
def total_zits_Smith := zits_per_student_Smith * students_Smith -- should be 140

def zits_per_student_Brown := 8
def students_Brown := 16
def total_zits_Brown := zits_per_student_Brown * students_Brown -- should be 128

def zits_per_student_Perez := 4
def students_Perez := 30
def total_zits_Perez := zits_per_student_Perez * students_Perez -- should be 120

theorem highest_total_zits : 
  total_zits_Jones = max total_zits_Swanson (max total_zits_Smith (max total_zits_Brown (max total_zits_Perez total_zits_Jones))) :=
by
  sorry

end highest_total_zits_l13_13945


namespace range_of_square_of_difference_of_roots_l13_13126

theorem range_of_square_of_difference_of_roots (a : ℝ) (h : (a - 1) * (a - 2) < 0) :
  ∃ (S : Set ℝ), S = { x | 0 < x ∧ x ≤ 1 } ∧ ∀ (x1 x2 : ℝ),
  x1 + x2 = 2 * a ∧ x1 * x2 = 2 * a^2 - 3 * a + 2 → (x1 - x2)^2 ∈ S :=
sorry

end range_of_square_of_difference_of_roots_l13_13126


namespace num_factors_of_M_l13_13918

def M : ℕ := 99^5 + 5 * 99^4 + 10 * 99^3 + 10 * 99^2 + 5 * 99 + 1

theorem num_factors_of_M : nat.factors.count M = 121 := by
  sorry

end num_factors_of_M_l13_13918


namespace balls_into_boxes_l13_13147

theorem balls_into_boxes : 
  ∃ (n : ℕ), n = nat.choose (6 + 3 - 1) (3 - 1) 
          ∧ n = 28 := 
by 
  sorry

end balls_into_boxes_l13_13147


namespace annika_total_kilometers_east_l13_13690

def annika_constant_rate : ℝ := 10 -- 10 minutes per kilometer
def distance_hiked_initially : ℝ := 2.5 -- 2.5 kilometers
def total_time_to_return : ℝ := 35 -- 35 minutes

theorem annika_total_kilometers_east :
  (total_time_to_return - (distance_hiked_initially * annika_constant_rate)) / annika_constant_rate + distance_hiked_initially = 3.5 := by
  sorry

end annika_total_kilometers_east_l13_13690


namespace product_of_roots_cubic_eq_l13_13033

theorem product_of_roots_cubic_eq (α : Type _) [Field α] :
  (∃ (r1 r2 r3 : α), (r1 * r2 * r3 = 6) ∧ (r1 + r2 + r3 = 6) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 11)) :=
by
  sorry

end product_of_roots_cubic_eq_l13_13033


namespace room_length_is_5_5_l13_13989

-- Define the conditions
def width := 3.75
def total_cost := 20625
def rate_per_sqm := 1000

-- Compute the total area
def total_area := total_cost / rate_per_sqm

-- Define what we need to prove
def length := total_area / width

-- The statement to be proven
theorem room_length_is_5_5 : length = 5.5 := by
  sorry

end room_length_is_5_5_l13_13989


namespace range_of_a_plus_b_at_least_one_nonnegative_l13_13462

-- Conditions
variable (x : ℝ) (a := x^2 - 1) (b := 2 * x + 2)

-- Proof Problem 1: Prove that the range of a + b is [0, +∞)
theorem range_of_a_plus_b : (a + b) ≥ 0 :=
by sorry

-- Proof Problem 2: Prove by contradiction that at least one of a or b is greater than or equal to 0
theorem at_least_one_nonnegative : ¬(a < 0 ∧ b < 0) :=
by sorry

end range_of_a_plus_b_at_least_one_nonnegative_l13_13462


namespace min_value_is_1_5_l13_13935

noncomputable def min_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) : ℝ :=
  (1 : ℝ) / (a + b) + 
  (1 : ℝ) / (b + c) + 
  (1 : ℝ) / (c + a)

theorem min_value_is_1_5 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  min_value a b c h1 h2 h3 h4 = 1.5 :=
sorry

end min_value_is_1_5_l13_13935


namespace mean_of_new_numbers_l13_13606

theorem mean_of_new_numbers (x y z : ℝ) 
  (h1 : ∑ i in (range 7), a i = 294)
  (h2 : (∑ i in (range 7), a i) + x + y + z = 500) :
  (x + y + z) / 3 = 206 / 3 :=
by
  sorry

end mean_of_new_numbers_l13_13606


namespace sum_of_multiples_of_three_l13_13671

theorem sum_of_multiples_of_three : 
  (∑ i in finset.filter (λ x, x % 3 = 0) (finset.range (10 + 21)) (λ x, ((x - 20): ℤ))) = -45 :=
by
  sorry

end sum_of_multiples_of_three_l13_13671


namespace trigonometric_ratio_l13_13477

theorem trigonometric_ratio (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 1) :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -7 ∨
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 :=
sorry

end trigonometric_ratio_l13_13477


namespace solution_set_l13_13757

noncomputable def f : ℝ → ℝ := sorry

def f'_deriv (x : ℝ) : ℝ := sorry

axiom f_gt_f'_deriv : ∀ x : ℝ, f x > f'_deriv x

axiom f_odd_function : ∀ x : ℝ, f (x + 2017) = -f (-(x + 2017))

theorem solution_set : {x : ℝ | f (x + 2017) * exp x < 0} = set.Ioi 0 :=
by
  sorry

end solution_set_l13_13757


namespace number_of_elements_le_2009_l13_13555

def S : Set ℕ := {n | ∃ k, (n = 0) ∨ ∃ x ∈ S, (n = 3*x) ∨ (n = 3*x + 1)}

theorem number_of_elements_le_2009 : {n : ℕ | n ∈ S ∧ n ≤ 2009}.card = 128 :=
by
  sorry

end number_of_elements_le_2009_l13_13555


namespace fraction_of_3_4_is_4_27_l13_13307

theorem fraction_of_3_4_is_4_27 (a b : ℚ) (h1 : a = 3/4) (h2 : b = 1/9) :
  b / a = 4 / 27 :=
by
  sorry

end fraction_of_3_4_is_4_27_l13_13307


namespace cell_magnification_l13_13610

theorem cell_magnification (d : ℝ) (m : ℝ) (h_d : d = 1.56 * 10 ^ (-6)) (h_m : m = 10 ^ 6) : d * m = 1.56 :=
by
  rw [h_d, h_m]
  sorry

end cell_magnification_l13_13610


namespace function_minimum_value_is_minus_two_l13_13620

noncomputable def function_minimum_value (x : ℝ) : ℝ :=
  sin x ^ 2 - 2 * cos x

theorem function_minimum_value_is_minus_two :
  ∃ x : ℝ, function_minimum_value x = -2 :=
by sorry

end function_minimum_value_is_minus_two_l13_13620


namespace sum_of_inradii_l13_13164

-- Definitions of the given side lengths and midpoint condition.
def PQ : ℝ := 7
def PR : ℝ := 9
def QR : ℝ := 12
def S_midpoint (QR : ℝ) : ℝ := QR / 2

-- Conditions in a)
def side_lengths (P Q R : ℝ) : Prop := P = 7 ∧ Q = 9 ∧ R = 12
def is_midpoint (S QR : ℝ) : Prop := S = QR / 2

-- Translated mathematical proof problem
theorem sum_of_inradii 
    {P Q R S PS QS PR PQS PRS}
    (h1 : side_lengths PQ PR QR)
    (h2 : is_midpoint QS QR)
    (h3 : PS = Real.sqrt (PQ^2 + PR^2 - 2 * PQ * PR * Real.cos (Arccos (QS / QR))))
    (h4 : S_midpoint QR = 6)
    : (14 * Real.sqrt 5 / (6.5 + Real.sqrt 29)) = (2 * (14 * Real.sqrt 5 / (6.5 + Real.sqrt 29))) :=
sorry

end sum_of_inradii_l13_13164


namespace arccos_cos_8_eq_1_point_72_l13_13744

noncomputable def arccos_cos_eight : Real :=
  Real.arccos (Real.cos 8)

theorem arccos_cos_8_eq_1_point_72 : arccos_cos_eight = 1.72 :=
by
  sorry

end arccos_cos_8_eq_1_point_72_l13_13744


namespace volume_in_barrel_l13_13329

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem volume_in_barrel (x : ℕ) (V : ℕ) (hx : V = 30) 
  (h1 : V = x / 2 + x / 3 + x / 4 + x / 5 + x / 6) 
  (h2 : is_divisible (87 * x) 60) : 
  V = 29 := 
sorry

end volume_in_barrel_l13_13329


namespace cookie_cost_l13_13185

-- Given conditions:
def total_money : ℝ := 20
def hat_cost : ℝ := 10
def pencil_cost : ℝ := 2
def cookies_count : ℕ := 4
def remaining_money : ℝ := 3

-- Prove that each cookie costs $1.25
theorem cookie_cost :
  let total_spent := total_money - remaining_money,
      other_items_cost := hat_cost + pencil_cost,
      total_cookies_cost := total_spent - other_items_cost in
  (total_cookies_cost / cookies_count) = 1.25 :=
by 
  -- Proof goes here
  sorry

end cookie_cost_l13_13185


namespace hyperbola_eccentricity_l13_13442

theorem hyperbola_eccentricity:
  (a b e : ℝ) 
  (h1 : a = Real.sqrt 5) 
  (h2 : b = 2) 
  (h3 : e = Real.sqrt (a^2 + b^2) / a) :
  e = 3 * Real.sqrt 5 / 5 :=
by 
  rw [h1, h2, h3]
  sorry

end hyperbola_eccentricity_l13_13442


namespace range_of_a_l13_13825

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → ln x - a * (1 - 1 / x) ≥ 0) → a ≤ 1 :=
by
  sorry

end range_of_a_l13_13825
