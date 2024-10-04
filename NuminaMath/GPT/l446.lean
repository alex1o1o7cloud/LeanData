import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Floor
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.OrdField
import Mathlib.Algebra.Order
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Prob
import Mathlib.Algebra.Series
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.RealBasic
import Mathlib.Set_theory.Basic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Algebra.Polynomial
import Mathlib.Topology.Basic

namespace dispersion_measured_by_std_dev_and_range_l446_446176

variables {α : Type*} [linear_order α] (x : list α)

def standard_deviation (l : list ℝ) : ℝ := sorry -- definition of standard_deviation
def median (l : list α) : α := sorry -- definition of median
def range (l : list ℝ) : ℝ := sorry -- definition of range
def mean (l : list ℝ) : ℝ := sorry -- definition of mean

theorem dispersion_measured_by_std_dev_and_range :
  (standard_deviation (map (λ x, (x : ℝ)) (x : list α)) > 0 ∨ range (map (λ x, (x : ℝ)) (x : list α)) > 0) →
  (∀ x, x ∈ [standard_deviation (map (λ x, (x : ℝ)) (x : list α)), range (map (λ x, (x : ℝ)) (x : list α))]) :=
begin
  sorry
end

end dispersion_measured_by_std_dev_and_range_l446_446176


namespace area_closed_figure_l446_446835

noncomputable def closedFigureArea : ℝ :=
  ∫ x in 1..3, x - 1/x

theorem area_closed_figure : closedFigureArea = 4 - Real.log 3 := 
  sorry

end area_closed_figure_l446_446835


namespace local_max_at_x_eq_1_l446_446401

def f (x : ℝ) : ℝ := (sin (x - 1))^2 - x^2 + 2 * x

theorem local_max_at_x_eq_1 :
  let y := f;
  let y' := λ x => deriv y x;
  let y'' := λ x => deriv y' x;
  let y''' := λ x => deriv y'' x;
  let y'''' := λ x => deriv y''' x;
  y 1 = (sin (0))^2 - 1 + 2 ∧
  y'(1) = 0 ∧
  y''(1) = 2 ∧
  y'''(1) = 0 ∧
  y''''(1) = -16 ∧
  y''''(1) < 0 →
  ∃ ε > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → y x ≤ y 1 :=
  by
    intros
    sorry

end local_max_at_x_eq_1_l446_446401


namespace number_of_good_points_l446_446805

-- Definition of a point in a 2D grid
structure Point where
  x : Int
  y : Int

-- Definition of the good point property
def good_point (P A B C : Point) : Bool :=
  let area_PAB := (P.x * (A.y - B.y) + A.x * (B.y - P.y) + B.x * (P.y - A.y)).natAbs
  let area_PAC := (P.x * (A.y - C.y) + A.x * (C.y - P.y) + C.x * (P.y - A.y)).natAbs
  area_PAB = area_PAC

-- Statement of the problem
theorem number_of_good_points : 
  let grid_size := 9
  let A := Point.mk 0 0
  let B := Point.mk a 0
  let C := Point.mk 0 b
  (good_points_count : Int),
  good_points_count = 6
  sorry

end number_of_good_points_l446_446805


namespace work_done_l446_446206

def force1 : ℝ × ℝ := (Real.log 2, Real.log 2)
def force2 : ℝ × ℝ := (Real.log 5, Real.log 2)
def displacement : ℝ × ℝ := (2 * Real.log 5, 1)
def resultant_force : ℝ × ℝ := (force1.1 + force2.1, force1.2 + force2.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem work_done : dot_product resultant_force displacement = 2 := by
  have h1 : resultant_force = (1, 2 * Real.log 2) := by
    simp [force1, force2]
    rw [Real.log_mul (show 2 > 0 from by norm_num) (show 5 > 0 from by norm_num)]
    simp
    
  have h2 : dot_product resultant_force displacement = 2 * (Real.log 5 + Real.log 2) := by
    rw [h1, displacement]
    simp
    rw [mul_comm]
    
  rw [h2]
  simp
  rw [Real.log_mul (show 2 > 0 from by norm_num) (show 5 > 0 from by norm_num)]
  admit

end work_done_l446_446206


namespace probability_calculation_correct_l446_446094

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
let total_ways := (Nat.choose 24 5),
    all_boys_or_all_girls := 2 * (Nat.choose 12 5) in
1 - (all_boys_or_all_girls / total_ways)

theorem probability_calculation_correct :
  probability_at_least_one_boy_and_one_girl = 5115 / 5313 := by
sorry

end probability_calculation_correct_l446_446094


namespace parabola_intersections_l446_446921

noncomputable def intersection_points : set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), (y = 3*x^2 - 12*x - 15) ∧ (y = x^2 - 6*x + 11) ∧ p = (x, y)}

theorem parabola_intersections :
  intersection_points = 
    { ( (3 + Real.sqrt 61) / 2, 3 * ((3 + Real.sqrt 61) / 2)^2 - 12 * ((3 + Real.sqrt 61) / 2) - 15 ),
      ( (3 - Real.sqrt 61) / 2, 3 * ((3 - Real.sqrt 61) / 2)^2 - 12 * ((3 - Real.sqrt 61) / 2) - 15 ) } :=
by
  sorry

end parabola_intersections_l446_446921


namespace negation_exists_l446_446853

theorem negation_exists:
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by
  sorry

end negation_exists_l446_446853


namespace tangent_half_angle_identity_l446_446885

variable (a b c d s α : ℝ)

-- Assume the conditions:
-- 1. The sides of the cyclic quadrilateral.
-- 2. The perimeter is 2s. 
-- 3. The angle between sides a and b is α.
axiom sides_a_b_c_d : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
axiom perimeter : a + b + c + d = 2 * s 
axiom cyclic_quad_angle : 0 < α ∧ α < π

-- The goal is to show the given formula.
theorem tangent_half_angle_identity : 
  tan (α / 2) ^ 2 = ((s - a) * (s - b)) / ((s - c) * (s - d)) := 
sorry

end tangent_half_angle_identity_l446_446885


namespace find_c_l446_446712

theorem find_c (c : ℝ) (h : ∀ x : ℝ, ∃ a : ℝ, (x + a)^2 = x^2 + 200 * x + c) : c = 10000 :=
sorry

end find_c_l446_446712


namespace product_remainder_mod_7_l446_446868

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l446_446868


namespace product_of_divisors_eq_1024_l446_446519

theorem product_of_divisors_eq_1024 (n : ℕ) (h1 : 0 < n) (h2 : ∏ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446519


namespace sum_positive_integers_not_in_S_l446_446776

def set_S (m n : ℕ) : ℤ := 50 * m + 3 * n

theorem sum_positive_integers_not_in_S :
  ∑ k in {k : ℕ | k > 0 ∧ ¬ ∃ m n : ℕ, set_S m n = k}, k = 2009 := sorry

end sum_positive_integers_not_in_S_l446_446776


namespace ellipse_a_value_l446_446318

theorem ellipse_a_value (a : ℝ) (hpos : a > 0)
  (heq : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1 ↔ a = 3)
  (eccentricity : 2 / 3 = sqrt(a^2 - 5) / a) :
  a = 3 :=
by
  sorry

end ellipse_a_value_l446_446318


namespace four_digit_numbers_count_l446_446635

theorem four_digit_numbers_count : 
  let thousand_place_digits := {d : ℕ | 1 ≤ d ∧ d ≤ 9},
      hundred_place_digits := {d : ℕ | 0 ≤ d ∧ d ≤ 9},
      ten_place_digits := {d : ℕ | 0 ≤ d ∧ d ≤ 8},
      valid_unit_place d := d + 1 in
      (thousand_place_digits.card * hundred_place_digits.card * ten_place_digits.card) = 810 :=
by sorry

end four_digit_numbers_count_l446_446635


namespace area_of_region_enclosed_by_equation_l446_446275

theorem area_of_region_enclosed_by_equation :
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 4*y = -9) → 
  real.pi * 2^2 = 4 * real.pi :=
by
  sorry

end area_of_region_enclosed_by_equation_l446_446275


namespace proof_problem_l446_446321

def p : Prop := ∃ x : ℝ, x - 2 > log 10 x
def q : Prop := ∀ x : ℝ, x^2 > 0

theorem proof_problem : p ∧ ¬ q := by
  sorry

end proof_problem_l446_446321


namespace part1_part2_l446_446771

variables (a b c : ℝ) (λ : ℝ)

def f (λ a b c : ℝ) : ℝ :=
  (a / (λ * a + b + c)) + (b / (λ * b + c + a)) + (c / (λ * c + a + b))

theorem part1 (h_neg1_lt_lambda: -1 < λ) (h_lambda_lt_1 : λ < 1) :
  (3 / (λ + 2) ≤ f λ a b c) ∧ (f λ a b c < 2 / (λ + 1)) := 
begin
  sorry
end

theorem part2 (h_lambda_gt_1 : λ > 1) :
  (2 / (λ + 1) < f λ a b c) ∧ (f λ a b c ≤ 3 / (λ + 2)) :=
begin
  sorry
end

end part1_part2_l446_446771


namespace find_x_of_spherical_surface_l446_446683

theorem find_x_of_spherical_surface
  (x : ℝ)
  (h1 : 2^2 + x^2 + 5^2 = 29 + x^2)
  (h2 : 4 * π * ((sqrt (29 + x^2)) / 2)^2 = 38 * π) :
  x = 3 :=
by
  sorry

end find_x_of_spherical_surface_l446_446683


namespace sequences_sum_divisible_by_13_l446_446313

-- Define the sequence of 26 non-zero digits
def sequence := List (Fin 10)

-- The definition of divisibility by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- The proof problem statement
theorem sequences_sum_divisible_by_13 (s : sequence)
  (h : s.length = 26): ∃ segments : List (List ℕ), 
  (∀ seg ∈ segments, ¬seg.empty) ∧ 
  (∀ seg ∈ segments, (∑ n in seg, n) % 13 = 0) ∧ 
  (s = segments.join) :=
sorry

end sequences_sum_divisible_by_13_l446_446313


namespace calculate_ratio_l446_446314

variables {α : Type*} [linear_ordered_field α]

structure Triangle (α : Type*) :=
(A B C D : α)
(acute : α)
(angle_ADB_ACB_90 : A - D = C - B - acute)
(prod_equal : A * C * B = D * B * A)

theorem calculate_ratio (T : Triangle α) (angle_ADB_ACB_90_deg : T.angle_ADB_ACB_90)
(prod_equal_cond : T.prod_equal) :
  T.acute ∃ r, r = (A * B * C) / (A * D * B) ∨ r = sqrt 2 :=
by {
  sorry
}

end calculate_ratio_l446_446314


namespace reciprocal_of_neg_two_thirds_l446_446857

theorem reciprocal_of_neg_two_thirds : ∃ b : ℚ, (-2 / 3) * b = 1 ∧ b = -3 / 2 :=
by
  use -3 / 2
  split
  {
    -- Prove that (-2 / 3) * (-3 / 2) = 1
    calc (-2 / 3) * (-3 / 2) = (2 / 3) * (3 / 2)   : by ring
                       ... = (2 * 3) / (3 * 2)  : by rw mul_div_assoc'
                       ... = 6 / 6             : by ring
                       ... = 1                 : by norm_num
  }
  {
    -- Prove that our choice of b is indeed -3 / 2
    rfl
  }

end reciprocal_of_neg_two_thirds_l446_446857


namespace equal_points_probability_not_always_increasing_l446_446908

theorem equal_points_probability_not_always_increasing 
  (p q : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ q) (h₂ : q ≤ 1) :
  ¬ ∀ p₀ p₁, 0 ≤ p₀ ∧ p₀ ≤ p₁ ∧ p₁ ≤ 1 → 
    let f := λ x : ℝ, (3 * x^2 - 2 * x + 1) / 4 in
    f p₀ ≤ f p₁ := by
    sorry

end equal_points_probability_not_always_increasing_l446_446908


namespace sum_of_possible_values_of_x_l446_446079

theorem sum_of_possible_values_of_x : 
  (∑ x in ({x : ℝ | 4^(x^2 + 5*x + 6) = 16^(x + 3)}, (λ x, x))) = -3 :=
by
  sorry

end sum_of_possible_values_of_x_l446_446079


namespace original_class_strength_l446_446480

variable (x : ℕ)

/-- The average age of an adult class is 40 years.
  18 new students with an average age of 32 years join the class, 
  therefore decreasing the average by 4 years.
  Find the original strength of the class.
-/
theorem original_class_strength (h1 : 40 * x + 18 * 32 = (x + 18) * 36) : x = 18 := 
by sorry

end original_class_strength_l446_446480


namespace logo_design_proof_l446_446111

noncomputable def side_length : ℝ := 30
noncomputable def radius : ℝ := side_length / 4
noncomputable def square_area : ℝ := side_length ^ 2
noncomputable def circle_area : ℝ := π * (radius ^ 2)
noncomputable def total_circle_area : ℝ := 4 * circle_area
noncomputable def shaded_area : ℝ := square_area - total_circle_area
noncomputable def circle_circumference : ℝ := 2 * π * radius
noncomputable def total_circumference : ℝ := 4 * circle_circumference

theorem logo_design_proof :
  shaded_area = 900 - 225 * π ∧ total_circumference = 60 * π := sorry

end logo_design_proof_l446_446111


namespace dave_more_than_jerry_games_l446_446035

variable (K D J : ℕ)  -- Declaring the variables for Ken, Dave, and Jerry respectively

-- Defining the conditions
def ken_more_games := K = D + 5
def dave_more_than_jerry := D > 7
def jerry_games := J = 7
def total_games := K + D + 7 = 32

-- Defining the proof problem
theorem dave_more_than_jerry_games (hK : ken_more_games K D) (hD : dave_more_than_jerry D) (hJ : jerry_games J) (hT : total_games K D) : D - 7 = 3 :=
by
  sorry

end dave_more_than_jerry_games_l446_446035


namespace sin_inequality_l446_446072

theorem sin_inequality (K : ℝ) (x : ℝ) 
  (hK : 1 < K) (hx1 : 0 < x) (hx2 : x < π / K):
  (sin (K * x)) / (sin x) < K * real.exp (-((K^2 - 1) / 6) * x^2) := 
begin
  sorry,
end

end sin_inequality_l446_446072


namespace XiaoMing_strategy_l446_446969

noncomputable def prob_A_correct : ℝ := 0.8
noncomputable def prob_B_correct : ℝ := 0.6

def points_A_correct : ℝ := 20
def points_B_correct : ℝ := 80

def prob_XA_0 : ℝ := 1 - prob_A_correct
def prob_XA_20 : ℝ := prob_A_correct * (1 - prob_B_correct)
def prob_XA_100 : ℝ := prob_A_correct * prob_B_correct

def expected_XA : ℝ := 0 * prob_XA_0 + points_A_correct * prob_XA_20 + (points_A_correct + points_B_correct) * prob_XA_100

def prob_YB_0 : ℝ := 1 - prob_B_correct
def prob_YB_80 : ℝ := prob_B_correct * (1 - prob_A_correct)
def prob_YB_100 : ℝ := prob_B_correct * prob_A_correct

def expected_YB : ℝ := 0 * prob_YB_0 + points_B_correct * prob_YB_80 + (points_A_correct + points_B_correct) * prob_YB_100

def distribution_A_is_correct : Prop :=
  prob_XA_0 = 0.2 ∧ prob_XA_20 = 0.32 ∧ prob_XA_100 = 0.48

def choose_B_first : Prop :=
  expected_YB > expected_XA

theorem XiaoMing_strategy :
  distribution_A_is_correct ∧ choose_B_first :=
by
  sorry

end XiaoMing_strategy_l446_446969


namespace cost_of_gas_trip_l446_446473

-- Definitions derived from given conditions
def initial_odometer : ℕ := 92150
def final_odometer : ℕ := 92185
def mileage : ℕ := 25
def gas_price_per_gallon : ℝ := 4.25

-- Lean statement to prove the cost of gas used for the trip
theorem cost_of_gas_trip : 
  let distance_traveled := (final_odometer - initial_odometer : ℕ)
      gallons_used := (distance_traveled : ℝ) / (mileage : ℝ)
      cost := gallons_used * gas_price_per_gallon
  in
  cost = 5.95 :=
  by
  sorry

end cost_of_gas_trip_l446_446473


namespace product_of_divisors_eq_1024_l446_446515

theorem product_of_divisors_eq_1024 (n : ℕ) (h : n > 0) (hp : ∏ d in (finset.filter (λ x, x ∣ n) (finset.range (n+1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446515


namespace smallest_x_condition_l446_446145

theorem smallest_x_condition (x : ℕ) : (∃ x > 0, (3 * x + 28)^2 % 53 = 0) -> x = 26 := 
by
  sorry

end smallest_x_condition_l446_446145


namespace card_arrangement_exists_l446_446807

theorem card_arrangement_exists :
  ∃ (a b c d e f g h : ℕ),
    (a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ g ≠ h) ∧
    List.nodup [a, b, c, d, e, f, g, h] ∧
    0 ≤ a ∧ a < 8 ∧ 0 ≤ b ∧ b < 8 ∧
    0 ≤ c ∧ c < 8 ∧ 0 ≤ d ∧ d < 8 ∧
    0 ≤ e ∧ e < 8 ∧ 0 ≤ f ∧ f < 8 ∧
    0 ≤ g ∧ g < 8 ∧ 0 ≤ h ∧ h < 8 ∧
    (|a - b| = 2) ∧
    (|c - d| = 3) ∧
    (|e - f| = 4) ∧
    (|g - h| = 5) :=
by
  sorry

end card_arrangement_exists_l446_446807


namespace butterfly_theorem_l446_446775

noncomputable theory
open_locale classical

variables {S : Type*} [circle S]
variables (A B M N P Q E F O : S)
variables (hO : is_midpoint O A B)
variables (hPASS_O_MN : passes_through O M N)
variables (hPASS_O_PQ : passes_through O P Q)
variables (h_same_side : same_side_of_chord P N A B)
variables (h_int_E : intersection_chord E A B M P)
variables (h_int_F : intersection_chord F A B N Q)

theorem butterfly_theorem :
  is_midpoint O E F :=
sorry

end butterfly_theorem_l446_446775


namespace triangle_is_obtuse_l446_446229
-- Import necessary libraries

-- Definitions for altitudes
def altitude_a (a : ℝ) := 1 / 13
def altitude_b (b : ℝ) := 1 / 11
def altitude_c (c : ℝ) := 1 / 5

-- Define the triangle sides ratio condition
def sides_ratio (a b c : ℝ) := (a / b = 13 / 11) ∧ (a / c = 13 / 5) ∧ (b / c = 11 / 5)

-- Define obtuse triangle predicate
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  let cos_A := (b^2 + c^2 - a^2) / (2 * b * c) in
  cos_A < 0

-- The statement to prove
theorem triangle_is_obtuse (a b c : ℝ) (ha : altitude_a a) (hb : altitude_b b) (hc : altitude_c c) (ratio : sides_ratio a b c) : is_obtuse_triangle a b c :=
sorry

end triangle_is_obtuse_l446_446229


namespace product_of_divisors_eq_1024_l446_446522

theorem product_of_divisors_eq_1024 (n : ℕ) (h1 : 0 < n) (h2 : ∏ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446522


namespace choose_and_assign_members_l446_446320

theorem choose_and_assign_members 
  (members : Finset String) 
  (roles : Finset String) 
  (h_mem_count : members.card = 4)
  (h_roles_count : roles.card = 3) : 
  ∃ ways : ℕ, ways = 24 := 
by 
  have alice_mem : "Alice" ∈ members := sorry
  have bob_mem : "Bob" ∈ members := sorry
  have carol_mem : "Carol" ∈ members := sorry
  have dave_mem : "Dave" ∈ members := sorry
  have h_choose := Nat.choose_eq (4 : ℕ) (3 : ℕ) 
  have h_comb_choose : 4 = 4 := rfl
  have perm_roles : (Nat.factorial 3) = 6 := by simp
  existsi (4 * 6)
  rw [h_comb_choose, perm_roles]
  norm_num
  exact rfl

end choose_and_assign_members_l446_446320


namespace not_basic_structure_l446_446738

def sequential_structure : Prop := true
def selection_structure : Prop := true
def loop_structure : Prop := true
def judgment_structure : Prop := true

def basic_structures : list Prop := [sequential_structure, selection_structure, loop_structure]

theorem not_basic_structure : judgment_structure ∉ basic_structures := by
  sorry

end not_basic_structure_l446_446738


namespace savings_for_mother_l446_446797

theorem savings_for_mother (Liam_oranges Claire_oranges: ℕ) (Liam_price_per_pair Claire_price_per_orange: ℝ) :
  Liam_oranges = 40 ∧ Liam_price_per_pair = 2.50 ∧
  Claire_oranges = 30 ∧ Claire_price_per_orange = 1.20 →
  ((Liam_oranges / 2) * Liam_price_per_pair + Claire_oranges * Claire_price_per_orange) = 86 :=
by
  intros
  sorry

end savings_for_mother_l446_446797


namespace sum_fourth_powers_below_500_is_354_l446_446153

noncomputable def sum_fourth_powers_below_500 : ℕ :=
  ∑ n in Finset.Icc 1 4, n^4

theorem sum_fourth_powers_below_500_is_354 : sum_fourth_powers_below_500 = 354 :=
by
  sorry

end sum_fourth_powers_below_500_is_354_l446_446153


namespace graph_function_quadrant_l446_446494

theorem graph_function_quadrant (x y : ℝ): 
  (∀ x : ℝ, y = -x + 2 → (x < 0 → y ≠ -3 + - x)) := 
sorry

end graph_function_quadrant_l446_446494


namespace salary_calculation_l446_446524

variable {A B : ℝ}

theorem salary_calculation (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) : A = 4500 :=
by
  sorry

end salary_calculation_l446_446524


namespace sum_of_coefficients_l446_446607

theorem sum_of_coefficients : 
  (x y : ℂ) → (∑ a b c, (8.choose a).choose b * (a + b = 8) * (x^2)^a * (-3*x*y)^b * (y^2)^c := 
  ∑ i in (finset.range (8 + 1)),
  (8.choose i) * (1 : ℂ)^i * (-3 : ℂ) * (1 : ℂ)^(8 - i) = 1 :=
begin 
... , 
end 

end sum_of_coefficients_l446_446607


namespace dispersion_measures_l446_446167
-- Definitions for statistical measures (for clarity, too simplistic)
def standard_deviation (x : List ℝ) : ℝ := 
  let mean := (x.sum / x.length)
  Math.sqrt ((x.map (λ xi => (xi - mean)^2)).sum / (x.length - 1))

def median (x : List ℝ) : ℝ := 
  let sorted := x.qsort (≤)
  if h : sorted.length % 2 = 1 then (sorted.sorted.nth (sorted.length / 2))
  else ((sorted.nth (sorted.length / 2 - 1) + sorted.nth (sorted.length / 2)) / 2)

def range (x : List ℝ) : ℝ := x.maximum - x.minimum

def mean (x : List ℝ) : ℝ := x.sum / x.length

-- Statement to prove
theorem dispersion_measures (x : List ℝ) : 
  (standard_deviation x ∈ {standard_deviation x, range x}) ∧ 
  (range x ∈ {standard_deviation x, range x}) ∧
  ¬ (median x ∈ {standard_deviation x, range x})  ∧
  ¬ (mean x ∈ {standard_deviation x, range x}) := 
sorry

end dispersion_measures_l446_446167


namespace find_x_l446_446283

theorem find_x (x : ℝ) (h : ⌊x⌋ + x = 15/4) : x = 7/4 :=
sorry

end find_x_l446_446283


namespace probability_A2_equals_zero_matrix_l446_446058

noncomputable def probability_A2_zero (n : ℕ) (hn : n ≥ 2) : ℚ :=
  let numerator := (n - 1) * (n - 2)
  let denominator := n * (n - 1)
  numerator / denominator

theorem probability_A2_equals_zero_matrix (n : ℕ) (hn : n ≥ 2) :
  probability_A2_zero n hn = ((n - 1) * (n - 2) / (n * (n - 1))) := by
  sorry

end probability_A2_equals_zero_matrix_l446_446058


namespace range_of_a_with_common_tangent_l446_446363

noncomputable def has_common_tangent (a x₁ x₂ : ℝ) (pos_x1 : x₁ > 0) (pos_x2 : x₂ > 0) :=
  (-a / (x₁ ^ 2) = 2 / x₂) ∧ (2 * a / x₁ = 2 * (ln x₂) - 2)

theorem range_of_a_with_common_tangent : 
  ∃ (a : ℝ), ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → has_common_tangent a x₁ x₂ x₁ x₂ → (-2 / real.exp(1)) ≤ a ∧ a < 0 :=
sorry

end range_of_a_with_common_tangent_l446_446363


namespace ratio_male_hamsters_l446_446260

-- Definitions based on conditions
def total_pets : ℕ := 92
def total_gerbils : ℕ := 68
def total_males : ℕ := 25
def fraction_male_gerbils : ℚ := 1 / 4

-- Using these definitions, prove the ratio of male hamsters to total hamsters is 1:3
theorem ratio_male_hamsters (total_pets = 92) (total_gerbils = 68)
                        (total_males = 25) (fraction_male_gerbils = 1 / 4) :
    let male_gerbils := fraction_male_gerbils * total_gerbils
    let male_hamsters := total_males - male_gerbils
    let total_hamsters := total_pets - total_gerbils
    (male_hamsters / total_hamsters) = 1 / 3 := 
by
    sorry

end ratio_male_hamsters_l446_446260


namespace length_of_EF_l446_446014

-- Definitions and conditions
variables (A B C D E F X Y : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space X] [metric_space Y]
variable (d_AC : ℝ) (d_BD : ℝ) (d_AD : ℝ) (d_BC : ℝ) (d_AB : ℝ) (d_DC : ℝ)

-- Given the isosceles trapezoid condition
def is_isosceles_trapezoid (A B C D: Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] : Prop :=
  d_AD = d_BC ∧ d_AB = 6 ∧ d_DC = 14

-- Given point E on DF and B is the midpoint of DE
def midpoint (A E B: Type) [metric_space A] [metric_space E] [metric_space B] : Prop :=
  dist A E = dist E B

-- Given lengths
def length_AD (AD: Type) [metric_space AD] : ℝ := 7
def length_BC (BC: Type) [metric_space BC] : ℝ := 7
def length_AB (AB: Type) [metric_space AB] : ℝ := 6
def length_DC (DC: Type) [metric_space DC] : ℝ := 14

-- Prove that the length of EF is 20
theorem length_of_EF (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
  (h1: is_isosceles_trapezoid A B C D)
  (h2: midpoint D E B)
  (h3: d_AD = 7)
  (h4: d_BC = 7)
  (h5: d_AB = 6)
  (h6: d_DC = 14) :
  dist E F = 20 :=
sorry

end length_of_EF_l446_446014


namespace number_of_lines_at_specified_distances_l446_446388

def point := (ℝ × ℝ)

def distance_from_point_to_line (p : point) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / sqrt (a ^ 2 + b ^ 2)

def A : point := (1, 2)
def B : point := (3, 1)

theorem number_of_lines_at_specified_distances :
  ∃ l : ℝ × ℝ × ℝ, (distance_from_point_to_line A l.1 l.2 l.3 = 1) 
                    ∧ (distance_from_point_to_line B l.1 l.2 l.3 = 2)
                    ∧ (∃! l1 l2 l3 : ℝ, ∃! k : ℝ × ℝ × ℝ, k = (l1, l2, l3)) :=
by
  sorry

end number_of_lines_at_specified_distances_l446_446388


namespace total_problems_l446_446239

theorem total_problems (C W : ℕ) (h1 : 3 * C + 5 * W = 110) (h2 : C = 20) : C + W = 30 :=
by {
  sorry
}

end total_problems_l446_446239


namespace savings_for_mother_l446_446796

theorem savings_for_mother (Liam_oranges Claire_oranges: ℕ) (Liam_price_per_pair Claire_price_per_orange: ℝ) :
  Liam_oranges = 40 ∧ Liam_price_per_pair = 2.50 ∧
  Claire_oranges = 30 ∧ Claire_price_per_orange = 1.20 →
  ((Liam_oranges / 2) * Liam_price_per_pair + Claire_oranges * Claire_price_per_orange) = 86 :=
by
  intros
  sorry

end savings_for_mother_l446_446796


namespace max_distance_city_l446_446595

theorem max_distance_city (highway_mpg : ℝ) (city_mpg : ℝ) (gallons : ℝ) (max_distance_highway : ℝ) :
  highway_mpg = 12.2 → city_mpg = 7.6 → gallons = 22 → max_distance_highway = highway_mpg * gallons →
  city_mpg * gallons = 167.2 :=
by
  intros h_highway_mpg h_city_mpg h_gallons h_max_distance_highway
  rw [h_city_mpg, h_gallons]
  norm_num
  sorry

end max_distance_city_l446_446595


namespace rounding_hexagon_l446_446396

-- Let a_1, a_2, a_3, a_4, a_5, a_6 be the values assigned to the vertices
-- These values are represented by real numbers
variables {a_1 a_2 a_3 a_4 a_5 a_6 : ℝ}

-- Define rounding functions
def round_up (r : ℝ) : ℤ := int.ceil r
def round_down (r : ℝ) : ℤ := int.floor r

-- Prove that there exists a suitable rounding that maintains the sum condition
theorem rounding_hexagon (h1 : a_1 + a_2 = s_12) (h2 : a_2 + a_3 = s_23) 
                         (h3 : a_3 + a_4 = s_34) (h4 : a_4 + a_5 = s_45) 
                         (h5 : a_5 + a_6 = s_56) (h6 : a_6 + a_1 = s_61) :
  ∃ (b_1 b_2 b_3 b_4 b_5 b_6 : ℤ),
    b_1 = round_down a_1 ∧ b_2 = round_up a_2 ∧ 
    b_3 = round_down a_3 ∧ b_4 = round_up a_4 ∧ 
    b_5 = round_down a_5 ∧ b_6 = round_up a_6 ∧
    b_1 + b_2 = s_12 ∧ b_2 + b_3 = s_23 ∧ 
    b_3 + b_4 = s_34 ∧ b_4 + b_5 = s_45 ∧ 
    b_5 + b_6 = s_56 ∧ b_6 + b_1 = s_61 := sorry

end rounding_hexagon_l446_446396


namespace denominator_of_second_fraction_l446_446937

theorem denominator_of_second_fraction :
  let a := 2007
  let b := 2999
  let c := 8001
  let d := 2001
  let e := 3999
  let sum := 3.0035428163476343
  let first_fraction := (2007 : ℝ) / 2999
  let third_fraction := (2001 : ℝ) / 3999
  ∃ x : ℤ, (first_fraction + (8001 : ℝ) / x + third_fraction) = 3.0035428163476343 ∧ x = 4362 := 
by
  sorry

end denominator_of_second_fraction_l446_446937


namespace tangent_at_origin_interval_of_increase_l446_446787

open Real

noncomputable def f (x : ℝ) : ℝ := x * exp x

theorem tangent_at_origin : 
    let x0 := 0 in
    let y0 := f 0 in
    let f' (x : ℝ) := (1 + x) * exp x in
    let slope := f' x0 in
    ∀ x, (f x - y0) = slope * (x - x0) → (y0 = 0 ∧ slope = 1 ∧ ∀ x, f x = slope * x) := 
begin
    sorry
end

theorem interval_of_increase :
    (∀ x ∈ Ioc (-1: ℝ) (∞), (1 + x) * exp x > 0) := 
begin
    sorry
end

end tangent_at_origin_interval_of_increase_l446_446787


namespace cyclic_sum_inequality_l446_446660

variable {a b c x y z : ℝ}

-- Define the conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : x = a + 1 / b - 1
axiom h5 : y = b + 1 / c - 1
axiom h6 : z = c + 1 / a - 1
axiom h7 : x > 0
axiom h8 : y > 0
axiom h9 : z > 0

-- The statement we need to prove
theorem cyclic_sum_inequality : (x * y) / (Real.sqrt (x * y) + 2) + (y * z) / (Real.sqrt (y * z) + 2) + (z * x) / (Real.sqrt (z * x) + 2) ≥ 1 :=
sorry

end cyclic_sum_inequality_l446_446660


namespace find_larger_number_l446_446718

theorem find_larger_number (x y : ℝ) (h1 : 4 * y = 6 * x) (h2 : x + y = 36) : y = 21.6 :=
by
  sorry

end find_larger_number_l446_446718


namespace angle_bisector_slope_l446_446478

theorem angle_bisector_slope :
  ∃ k : ℝ, let m1 := 1, m2 := 4 in
  k = -5 / 3 + Real.sqrt 2 ∧
  (∃ (θ : ℝ), θ = Real.arctan (abs (m2 - m1) / (1 + m1 * m2))) ∧
  (∃ b : ℝ, b = (m1 + m2 + Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)) ∧
  k = b := 
begin
  sorry
end

end angle_bisector_slope_l446_446478


namespace dispersion_measures_l446_446169
-- Definitions for statistical measures (for clarity, too simplistic)
def standard_deviation (x : List ℝ) : ℝ := 
  let mean := (x.sum / x.length)
  Math.sqrt ((x.map (λ xi => (xi - mean)^2)).sum / (x.length - 1))

def median (x : List ℝ) : ℝ := 
  let sorted := x.qsort (≤)
  if h : sorted.length % 2 = 1 then (sorted.sorted.nth (sorted.length / 2))
  else ((sorted.nth (sorted.length / 2 - 1) + sorted.nth (sorted.length / 2)) / 2)

def range (x : List ℝ) : ℝ := x.maximum - x.minimum

def mean (x : List ℝ) : ℝ := x.sum / x.length

-- Statement to prove
theorem dispersion_measures (x : List ℝ) : 
  (standard_deviation x ∈ {standard_deviation x, range x}) ∧ 
  (range x ∈ {standard_deviation x, range x}) ∧
  ¬ (median x ∈ {standard_deviation x, range x})  ∧
  ¬ (mean x ∈ {standard_deviation x, range x}) := 
sorry

end dispersion_measures_l446_446169


namespace Joan_seashells_l446_446409

theorem Joan_seashells (J_J : ℕ) (J : ℕ) (h : J + J_J = 14) (hJJ : J_J = 8) : J = 6 :=
by
  sorry

end Joan_seashells_l446_446409


namespace find_abc_l446_446049

theorem find_abc :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a + b + c = 30 ∧
  (1/a + 1/b + 1/c + 450/(a*b*c) = 1) ∧ 
  a*b*c = 1912 :=
sorry

end find_abc_l446_446049


namespace find_n_l446_446511

-- Define that n is a positive integer
def positive_integer (n : ℕ) : Prop := n > 0

-- Define number of divisors
def num_divisors (n : ℕ) : ℕ := (finset.range (n+1)).filter (λ d, n % d = 0).card

-- Define the product of divisors function
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
(finset.range (n+1)).filter (λ d, n % d = 0).prod id

-- The final theorem statement to be proven
theorem find_n (n : ℕ) (hn : positive_integer n) :
  prod_of_divisors n = 1024 → n = 16 :=
by { sorry }

end find_n_l446_446511


namespace find_SD_l446_446745

theorem find_SD 
  (ABCD : Type) [rectangle ABCD]
  (A B C D P T S Q R : Point) 
  (hAPD : angle A P D = 90) 
  (hBP_PT : BP = PT) 
  (hTS_perpendicular : perpendicular T S B C)
  (hPD_intersects_TS : P D intersects T S at Q)
  (hRA_passes_Q : R on CD ∧ R A passes through Q)
  (PA AQ QP : ℝ)
  (hPA : PA = 18) 
  (hAQ : AQ = 15)
  (hQP : QP = 9) :
  SD = 42 / 5 := 
by 
  sorry

end find_SD_l446_446745


namespace solve_system_of_equations_l446_446826

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (3 * x + y = 8) ∧ (2 * x - y = 7) ∧ x = 3 ∧ y = -1 :=
by {
  -- Variables
  let x := 3,
  let y := -1,
  use [x, y],
  -- Show conditions are satisfied
  split,
  {
    calc 3 * x + y = 3 * 3 + (-1) : by sorry
           ...    = 8            : by sorry,
  },
  {
    split,
    {
      calc 2 * x - y = 2 * 3 - (-1) : by sorry
             ...    = 7            : by sorry,
    },
    {
      split; { trivial }
    }
  }
}

end solve_system_of_equations_l446_446826


namespace joe_total_time_l446_446760

def joe_walk_run_time (t_w t_r r_w r_r d : ℝ) : ℝ :=
  if r_r = 2 * r_w ∧ t_w = 9 ∧ (r_w * t_w + r_r * t_r = d) then t_w + t_r else 0

theorem joe_total_time :
  let t_w := 9 in
  let r_w := d / 27 in
  let r_r := 2 * r_w in
  let t_r := 9 in
  t_w + t_r = 18 :=
by
  sorry

end joe_total_time_l446_446760


namespace dispersion_measures_correct_l446_446186

-- Define a sample data set
variable {x : ℕ → ℝ}
variable {n : ℕ}

-- Definitions of the four statistics
def standard_deviation (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def median (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def range (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def mean (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry

-- Definition of measures_dispersion function
def measures_dispersion (stat : ℕ → ℝ → ℝ) (x : ℕ → ℝ) (n : ℕ) : Prop :=
  sorry -- Define what it means for a statistic to measure dispersion

-- Problem statement in Lean
theorem dispersion_measures_correct :
  measures_dispersion standard_deviation x n ∧
  measures_dispersion range x n ∧
  ¬measures_dispersion median x n ∧
  ¬measures_dispersion mean x n :=
by sorry

end dispersion_measures_correct_l446_446186


namespace problem_1_problem_2_problem_3_l446_446062

-- The sequence S_n and its given condition
def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - 2 * n

-- Definitions for a_1, a_2, and a_3 based on S_n conditions
theorem problem_1 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  a 1 = 2 ∧ a 2 = 6 ∧ a 3 = 14 :=
sorry

-- Definition of sequence b_n and its property of being geometric
def b (n : ℕ) (a : ℕ → ℕ) : ℕ := a n + 2

theorem problem_2 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n ≥ 1, b n a = 2 * b (n - 1) a :=
sorry

-- The sum of the first n terms of the sequence {na_n}, denoted by T_n
def T (n : ℕ) (a : ℕ → ℕ) : ℕ := (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1)

theorem problem_3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n, T n a = (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1) :=
sorry

end problem_1_problem_2_problem_3_l446_446062


namespace sum_fourth_powers_below_500_is_354_l446_446154

noncomputable def sum_fourth_powers_below_500 : ℕ :=
  ∑ n in Finset.Icc 1 4, n^4

theorem sum_fourth_powers_below_500_is_354 : sum_fourth_powers_below_500 = 354 :=
by
  sorry

end sum_fourth_powers_below_500_is_354_l446_446154


namespace average_side_lengths_of_squares_l446_446096

theorem average_side_lengths_of_squares:
  let a₁ := 25
  let a₂ := 36
  let a₃ := 64

  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃

  (s₁ + s₂ + s₃) / 3 = 19 / 3 :=
by 
  sorry

end average_side_lengths_of_squares_l446_446096


namespace divide_sqrt_81_by_3_l446_446159

theorem divide_sqrt_81_by_3 : (sqrt 81 = 9) -> 9 / 3 = 3 :=
by
  intro h
  rw h
  norm_num
  sorry

end divide_sqrt_81_by_3_l446_446159


namespace find_cos_squared_y_l446_446202

noncomputable def α : ℝ := Real.arccos (-3 / 7)

def arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

def transformed_arithmetic_progression (a b c : ℝ) : Prop :=
  14 / Real.cos b = 1 / Real.cos a + 1 / Real.cos c

theorem find_cos_squared_y (x y z : ℝ)
  (h1 : arithmetic_progression x y z)
  (h2 : transformed_arithmetic_progression x y z)
  (hα : 2 * α = z - x) : Real.cos y ^ 2 = 10 / 13 :=
by
  sorry

end find_cos_squared_y_l446_446202


namespace fraction_of_income_from_tips_l446_446559

theorem fraction_of_income_from_tips (S T I : ℚ) (h1 : T = (5/3) * S) (h2 : I = S + T) :
  T / I = 5 / 8 :=
by
  -- We're only required to state the theorem, not prove it.
  sorry

end fraction_of_income_from_tips_l446_446559


namespace table_coloring_l446_446484

theorem table_coloring (m n : ℕ) (h_m : m ≥ 5) (h_n : n ≥ 5)
  (colors : m × n → Fin 3)
  (h_adj : ∀ i j, i < m → j < n →
    let ci := colors (i, j) in
    ((if i > 0 then [colors (i - 1, j)] else []) ++
     (if i < m - 1 then [colors (i + 1, j)] else []) ++
     (if j > 0 then [colors (i, j - 1)] else []) ++
     (if j < n - 1 then [colors (i, j + 1)] else [])).count ci = 0)
  (h_corners : ∀ (i j), 
    (i = 0 ∨ i = m - 1) → 
    (j = 0 ∨ j = n - 1) →
    let ci := colors (i, j) in
    (if i > 0 then [colors (i - 1, j)] else []) ++
    (if i < m - 1 then [colors (i + 1, j)] else []) ++
    (if j > 0 then [colors (i, j - 1)] else []) ++
    (if j < n - 1 then [colors (i, j + 1)] else [])
    ).count ci = 0 := 
(m = 2 * i ∧ n = 3 * j ∨ m = 3 * k ∧ n = 2 * l :=
by
  ∃ i j k l, 3 ≤ i ∧ 2 ≤ j ∧ 3 ≤ k ∧ 2 ≤ l

end table_coloring_l446_446484


namespace valid_sandwiches_bob_can_order_l446_446841

def total_breads := 5
def total_meats := 7
def total_cheeses := 6

def undesired_combinations_count : Nat :=
  let turkey_swiss := total_breads
  let roastbeef_rye := total_cheeses
  let roastbeef_swiss := total_breads
  turkey_swiss + roastbeef_rye + roastbeef_swiss

def total_sandwiches : Nat :=
  total_breads * total_meats * total_cheeses

def valid_sandwiches_count : Nat :=
  total_sandwiches - undesired_combinations_count

theorem valid_sandwiches_bob_can_order : valid_sandwiches_count = 194 := by
  sorry

end valid_sandwiches_bob_can_order_l446_446841


namespace giraffes_not_pandas_l446_446282

theorem giraffes_not_pandas :
  ∀ (total children_pandas children_giraffes children_only_pandas : ℕ),
    total = 50 →
    children_pandas = 36 →
    children_giraffes = 28 →
    children_only_pandas = 15 →
    (children_giraffes - (children_pandas - children_only_pandas)) = 7 :=
by
  intros total children_pandas children_giraffes children_only_pandas h_total h_pandas h_giraffes h_only_pandas
  rw [h_total, h_pandas, h_giraffes, h_only_pandas]
  sorry

end giraffes_not_pandas_l446_446282


namespace distance_from_Idaho_to_Nevada_l446_446615

theorem distance_from_Idaho_to_Nevada (d1 d2 s1 s2 t total_time : ℝ) 
  (h1 : d1 = 640)
  (h2 : s1 = 80)
  (h3 : s2 = 50)
  (h4 : total_time = 19)
  (h5 : t = total_time - (d1 / s1)) :
  d2 = s2 * t :=
by
  sorry

end distance_from_Idaho_to_Nevada_l446_446615


namespace product_remainder_mod_7_l446_446877

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l446_446877


namespace bottle_caps_per_car_l446_446413

def total_bottle_caps := 100
def truck_price := 6
def trucks_bought := 10
def cars_bought := 16 - trucks_bought
def percent_spent_on_cars := 0.75

theorem bottle_caps_per_car :
  (total_bottle_caps - (truck_price * trucks_bought)) * percent_spent_on_cars / cars_bought = 5 :=
by
  sorry

end bottle_caps_per_car_l446_446413


namespace unit_prices_minimum_cost_l446_446574

open Locale.Matrix

variables (price_A price_B units_A units_B : ℕ)

-- Adding conditions as hypotheses
axiom h1 : price_A = price_B + 200
axiom h2 : 2000 / price_A = 1200 / price_B
axiom h3 : units_A + units_B = 40
axiom h4 : units_B ≤ 3 * units_A
axiom h5 : price_A = 500
axiom h6 : price_B = 300

-- Define the total cost with a 20% discount
noncomputable def total_cost : ℕ :=
  4 * (price_A * units_A) + 4 * (price_B * units_B)

-- The proof problem statements
theorem unit_prices : price_A = 500 ∧ price_B = 300 := by sorry

theorem minimum_cost : units_A = 10 ∧ units_B = 30 ∧ total_cost price_A price_B units_A units_B = 11200 := by sorry

end unit_prices_minimum_cost_l446_446574


namespace count_factorable_polynomials_l446_446295

-- Definition of the polynomial and conditions
def is_factorable_polynomial (n : ℕ) : Prop :=
  ∃ b : ℤ, n = -(b^2 + b) ∧ 1 ≤ n ∧ n ≤ 2000

-- Statement we need to prove
theorem count_factorable_polynomials : 
  (∃ (count : ℕ), count = nat.card {n | is_factorable_polynomial n} ∧ count = 89) :=
sorry

end count_factorable_polynomials_l446_446295


namespace bubble_pass_pos25_to_pos40_l446_446618

open Nat

-- Define the sequence of distinct real numbers
def seq (n : ℕ) : Type := { s : ℕ → ℝ // ∀ i j : ℕ, i ≠ j → s i ≠ s j }

-- Define the bubble pass operation
def bubble_pass {n : ℕ} (s : seq n) : seq n :=
  ⟨λ i, if i < n-1 ∧ s.val i > s.val (i+1) then s.val (i+1) else s.val i,
   λ i j h, by
     split_ifs; 
     dsimp; 
     finish⟩

-- Given conditions
noncomputable def initial_seq : seq 50 := sorry

-- State the theorem
theorem bubble_pass_pos25_to_pos40 (s : seq 50) :
  ∃ p q : ℕ, p + q = 1641 ∧
  let s' := bubble_pass s in
  (∀ i, s.val i ≠ s.val (25 - 1) ∨ i < 25 - 1 ∨ i > 25 - 1) ∧
  (∀ i, s.val i ≠ s.val (40 - 1) ∨ i < 40 - 1 ∨ i > 40 - 1) ∧
  (s.val (25 - 1) = s'.val (40 - 1)) ∧
  ( ∃ r1 r2 : ℚ, r1 = 1 ∧ r2 = 1640 ∧ r1 / r2 = (39.factorial : ℚ) / (41.factorial : ℚ) ) := sorry

end bubble_pass_pos25_to_pos40_l446_446618


namespace find_x_l446_446643

-- Define the problem statement
theorem find_x (x : ℝ) : 0 < x ∧ x < 360 ∧ 
  tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) → 
  x = 100 ∨ x = 220 :=
by
  sorry

end find_x_l446_446643


namespace shelf_distance_from_end_l446_446994

-- The width of the wall
def wall_width : ℝ := 26

-- The width of the picture
def picture_width : ℝ := 4

-- The picture is centered on the wall
def picture_distance_from_edge : ℝ := (wall_width - picture_width) / 2

-- The left edge of the shelf aligns with the right edge of the picture
def shelf_position_from_edge : ℝ := picture_distance_from_edge + picture_width

-- Statement to prove
theorem shelf_distance_from_end : shelf_position_from_edge = 15 :=
by
  -- Proof is omitted as per instructions
  sorry

end shelf_distance_from_end_l446_446994


namespace sum_of_squared_distances_constant_l446_446698

-- Define the conditions of the problem
variables {R a : ℝ} -- R is the radius of the circle, a is half the side length of the square
variables {P : ℝ × ℝ} -- P is a point on the circle, represented as a coordinate pair
variables {x y : ℝ} -- Coordinates of point P
variables (hP : x^2 + y^2 = R^2) -- P lies on the circumference of the circle

-- Define the coordinates of vertices of the square centered at the origin
def A : ℝ × ℝ := (-a, a)
def B : ℝ × ℝ := (a, a)
def C : ℝ × ℝ := (a, -a)
def D : ℝ × ℝ := (-a, -a)

-- Define squared distances: PA^2, PB^2, PC^2, PD^2
def PA_sq (P : ℝ × ℝ) : ℝ := (P.1 + a)^2 + (P.2 - a)^2
def PB_sq (P : ℝ × ℝ) : ℝ := (P.1 - a)^2 + (P.2 - a)^2
def PC_sq (P : ℝ × ℝ) : ℝ := (P.1 - a)^2 + (P.2 + a)^2
def PD_sq (P : ℝ × ℝ) : ℝ := (P.1 + a)^2 + (P.2 + a)^2

-- The theorem to prove the sum of squared distances is constant
theorem sum_of_squared_distances_constant (hP : x^2 + y^2 = R^2) : 
  PA_sq (x, y) + PB_sq (x, y) + PC_sq (x, y) + PD_sq (x, y) = 4 * R^2 + 8 * a^2 := 
by {
  sorry
}

end sum_of_squared_distances_constant_l446_446698


namespace prove_BC_gt_AD_l446_446106

variables {A B C D E : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (distance : A → B → ℝ) (angle : C → D → E → ℝ)
variables {quadrilateral : Prop}

-- Conditions
def intersect_at (AC BD E : Prop) := true
def equal_length (AB CE : ℝ) := AB = CE
def equal_length2 (BE AD : ℝ) := BE = AD
def equal_angle (AED BAD : ℝ) := AED = BAD

-- Question
def to_prove (BC AD : ℝ) := BC > AD

-- Full proof statement
theorem prove_BC_gt_AD {A B C D E : Type*} [metric_space A] [metric_space B] [metric_space C] 
  [metric_space D] [metric_space E]
  (distance : A → B → ℝ) (angle : C → D → E → ℝ)
  (quadrilateral : Prop)
  (AC BD E : Prop) (AB CE BE AD BC : ℝ) (AED BAD : ℝ)
  (h_intersect : intersect_at AC BD E)
  (h_AB_CE : equal_length AB CE)
  (h_BE_AD : equal_length2 BE AD)
  (h_AED_BAD : equal_angle AED BAD) : to_prove BC AD :=
sorry

end prove_BC_gt_AD_l446_446106


namespace cumulative_distribution_X_maximized_expected_score_l446_446975

noncomputable def distribution_X (p_A : ℝ) (p_B : ℝ) : (ℝ × ℝ × ℝ) :=
(1 - p_A, p_A * (1 - p_B), p_A * p_B)

def expected_score (p_A : ℝ) (p_B : ℝ) (s_A : ℝ) (s_B : ℝ) : ℝ :=
0 * (1 - p_A) + s_A * (p_A * (1 - p_B)) + (s_A + s_B) * (p_A * p_B)

theorem cumulative_distribution_X :
  distribution_X 0.8 0.6 = (0.2, 0.32, 0.48) :=
sorry

theorem maximized_expected_score :
  expected_score 0.8 0.6 20 80 < expected_score 0.6 0.8 80 20 :=
sorry

end cumulative_distribution_X_maximized_expected_score_l446_446975


namespace range_of_m_l446_446724

theorem range_of_m (m : ℝ) : (∃ x1 x2 x3 : ℝ, 
    (x1 - 1) * (x1^2 - 2*x1 + m) = 0 ∧ 
    (x2 - 1) * (x2^2 - 2*x2 + m) = 0 ∧ 
    (x3 - 1) * (x3^2 - 2*x3 + m) = 0 ∧ 
    x1 = 1 ∧ 
    x2^2 - 2*x2 + m = 0 ∧ 
    x3^2 - 2*x3 + m = 0 ∧ 
    x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1 ∧ 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0) ↔ 3 / 4 < m ∧ m ≤ 1 := 
by
  sorry

end range_of_m_l446_446724


namespace cos_neg_75_eq_l446_446608

noncomputable def cos_75_degrees : Real := (Real.sqrt 6 - Real.sqrt 2) / 4

theorem cos_neg_75_eq : Real.cos (-(75 * Real.pi / 180)) = cos_75_degrees := by
  sorry

end cos_neg_75_eq_l446_446608


namespace problem_solution_l446_446357

theorem problem_solution :
  ∃ (m n : ℕ), m.coprime n ∧ (2013 * 2013) / (2014 * 2014 + 2012) = (n : ℚ) / m ∧ m + n = 1343 :=
begin
  sorry
end

end problem_solution_l446_446357


namespace sin_A_plus_B_lt_sin_A_add_sin_B_l446_446398

variable {A B : ℝ}
variable (A_pos : 0 < A)
variable (B_pos : 0 < B)
variable (AB_sum_pi : A + B < π)

theorem sin_A_plus_B_lt_sin_A_add_sin_B (a b : ℝ) (h1 : a = Real.sin (A + B)) (h2 : b = Real.sin A + Real.sin B) : 
  a < b := by
  sorry

end sin_A_plus_B_lt_sin_A_add_sin_B_l446_446398


namespace winnie_ate_l446_446554

def side_length := 3 * a

-- Area of regular hexagon
def total_area_hex (a : ℝ) := (27 * real.sqrt 3 / 2) * a^2

-- Area of one triangle
def area_triangle (a : ℝ) := (real.sqrt 3 / 4) * a^2

-- Total area of 6 triangles
def area_total_cut (a : ℝ) := 6 * (real.sqrt 3 / 4) * a^2

-- Remaining cake weight
def remaining_weight : ℝ := 900

-- Relate weights to find the weight of the eaten part
theorem winnie_ate (a : ℝ) : 
  (9 / 8) * remaining_weight - remaining_weight = 112.5 :=
by
  sorry

end winnie_ate_l446_446554


namespace picnic_attendance_l446_446525

theorem picnic_attendance (L x : ℕ) (h1 : L + x = 2015) (h2 : L - (x - 1) = 4) : x = 1006 := 
by
  sorry

end picnic_attendance_l446_446525


namespace number_of_daisies_is_two_l446_446464

theorem number_of_daisies_is_two :
  ∀ (total_flowers daisies tulips sunflowers remaining_flowers : ℕ), 
    total_flowers = 12 →
    sunflowers = 4 →
    (3 / 5) * remaining_flowers = tulips →
    (2 / 5) * remaining_flowers = sunflowers →
    remaining_flowers = total_flowers - daisies - sunflowers →
    daisies = 2 :=
by
  intros total_flowers daisies tulips sunflowers remaining_flowers 
  sorry

end number_of_daisies_is_two_l446_446464


namespace percentage_of_income_from_large_and_medium_airliners_l446_446941

def small_airliners := 150
def medium_airliners := 75
def large_airliners := 60

def price_small := 125
def price_medium := 175
def price_large := 220

def income_small := small_airliners * price_small
def income_medium := medium_airliners * price_medium
def income_large := large_airliners * price_large
def income_total := income_small + income_medium + income_large
def income_medium_large := income_medium + income_large

def percentage_income_medium_large := (income_medium_large / income_total) * 100

theorem percentage_of_income_from_large_and_medium_airliners :
  percentage_income_medium_large ≈ 58.39 :=
sorry

end percentage_of_income_from_large_and_medium_airliners_l446_446941


namespace sam_distinct_meals_count_l446_446251

-- Definitions based on conditions
def main_dishes := ["Burger", "Pasta", "Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Chips", "Cookie", "Apple"]

-- Definition to exclude invalid combinations
def is_valid_combination (main : String) (beverage : String) : Bool :=
  if main = "Burger" && beverage = "Soda" then false else true

-- Number of valid combinations
def count_valid_meals : Nat :=
  main_dishes.length * beverages.length * snacks.length - snacks.length

theorem sam_distinct_meals_count : count_valid_meals = 15 := 
  sorry

end sam_distinct_meals_count_l446_446251


namespace middle_number_of_five_consecutive_numbers_l446_446121

theorem middle_number_of_five_consecutive_numbers (n : ℕ) 
  (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 60) : n = 12 :=
by
  sorry

end middle_number_of_five_consecutive_numbers_l446_446121


namespace second_machine_time_l446_446992

theorem second_machine_time (r : ℝ) :
  let first_machine_rate := 800 / 10,
      combined_rate := 800 / 3,
      second_machine_avg_rate := (r + 0.9*r + 0.81*r) / 3 in
  (first_machine_rate + second_machine_avg_rate = combined_rate) →
  800 / r ≈ 3.87 :=
by
  intros h1
  have h2 : first_machine_rate = 80 := by norm_num
  have h3 : combined_rate = 266.67 := by norm_num
  have h4 : second_machine_avg_rate = 0.9033 * r := by norm_num
  sorry

end second_machine_time_l446_446992


namespace annual_income_before_tax_l446_446725

-- Definitions based on conditions
def originalTaxRate : ℝ := 0.46
def newTaxRate : ℝ := 0.32
def differentialSavings : ℝ := 5040

-- Define the annual income before tax as a variable
variable (I : ℝ)

-- The statement to prove
theorem annual_income_before_tax :
  originalTaxRate * I - newTaxRate * I = differentialSavings → I = 36000 :=
by
  -- Assume the condition given in the problem
  intro h,
  -- Simplify the equation to solve for I
  have h1 : (originalTaxRate - newTaxRate) * I = differentialSavings,
  { linarith },
  -- Further simplification
  have h2 : 0.14 * I = differentialSavings,
  { rw [sub_eq_add_neg, ← sub_add_eq_sub_sub, zero_sub, neg_mul_eq_mul_neg, add_comm, mul_neg_eq_neg_mul_symm, sub_eq_add_neg, ← h1],
    norm_num [originalTaxRate, newTaxRate] },
  -- Solve for I
  rw [mul_comm] at h2,
  exact eq_of_mul_eq_mul_right (by linarith) h2

end annual_income_before_tax_l446_446725


namespace solution_set_inequality_l446_446328

variable (f : ℝ → ℝ)

noncomputable def f_def (x : ℝ) : ℝ :=
  if (x > 0) then (1 - x) * Real.exp x else
  if (x < 0) then (1 + x) * Real.exp (-x) else 0

lemma odd_function (x : ℝ) : f (-x) + f x = 0 :=
sorry  -- given condition: f(-x) + f(x) = 0

lemma domain (x : ℝ) : x ≠ 0 → x ∈ (-∞, 0) ∪ (0, ∞) :=
sorry  -- domain of the function

lemma f_when_positive (x : ℝ) (h : x > 0) : f x = (1 - x) * Real.exp x :=
sorry  -- f(x) = (1-x)e^x when x > 0

theorem solution_set_inequality : ∀ x, x*f x > 0 ↔ (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) :=
sorry  -- prove that the inequality xf(x) > 0 holds when x in (-1,0) ∪ (0,1)

end solution_set_inequality_l446_446328


namespace ratio_of_volumes_of_tetrahedrons_l446_446751

theorem ratio_of_volumes_of_tetrahedrons (a : ℝ) :
  let V1 = (a ^ 3 * Real.sqrt 2) / 12
  let V2 = ((2 * a) ^ 3 * Real.sqrt 2) / 12
  (V2 / V1) = 8 :=
by
  let V1 := (a ^ 3 * Real.sqrt 2) / 12
  let V2 := ((2 * a) ^ 3 * Real.sqrt 2) / 12
  have h : V2 / V1 = 8 := by sorry
  exact h

end ratio_of_volumes_of_tetrahedrons_l446_446751


namespace jeff_total_cabinets_l446_446758

theorem jeff_total_cabinets : 
  let initial_cabinets := 3
  let additional_per_counter := 3 * 2
  let num_counters := 3
  let additional_total := additional_per_counter * num_counters
  let final_cabinets := additional_total + 5
in initial_cabinets + final_cabinets = 26 :=
by
  -- Proof omitted
  sorry

end jeff_total_cabinets_l446_446758


namespace length_AB_l446_446384

-- Conditions Given in the problem
variables (P B C D A Q : Point) 
variables (BP CP PQ APQ DPQ APD : ℝ)
variables (sin_APD: sin APD = 5/13)
variables (BP_val : BP = 9)
variables (CP_val : CP = 15)

-- The proof problem statement: length of side AB is 24
theorem length_AB : ∃ x : ℝ, (PQ = x ∧ BP = 9 ∧ CP = 15 ∧ sin_APD = 5/13) → x = 24 :=
begin
  sorry
end

end length_AB_l446_446384


namespace set_intersection_complement_l446_446064

open Set

variable (U M N : Set ℕ)
variable [DecidableEq ℕ]

theorem set_intersection_complement (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 4}) (hN : N = {1, 3, 5}) :
  N ∩ (U \ M) = {3, 5} :=
by
  rw [hU, hM, hN]
  conv_rhs { rw [←@SetOf ℕ (fun x => x = 3 ∨ x = 5)] }
  rw [inter_diff_comm, inter, Set.ext_iff]
  intro x
  split
  · intro hx
    simp at hx ⊢
    cases hx
    · right
      exact hx.2
    · left
      exact hx.2
  · intro hx
    simp
    exact hx

end set_intersection_complement_l446_446064


namespace product_mod_7_l446_446863

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l446_446863


namespace triangle_AB_length_15_l446_446400

-- Define the conditions
variables {A B C D F E : Type} [HasLen A] [HasLen B] [HasLen C] [HasLen D] [HasLen F] [HasLen E]
variables [Triangle A B C]
variables [Median B D]
variables [Intersection CF BD E]
variables [Equality BE ED]
variables [OnLine F AB]
variables [Length BF (5 : ℝ)]

-- State theorem to prove
theorem triangle_AB_length_15 :
  length (A, B) = 15 := sorry

end triangle_AB_length_15_l446_446400


namespace intersecting_diagonals_prob_nonagon_l446_446957

theorem intersecting_diagonals_prob_nonagon :
  let num_vertices := 9
  let num_diagonals := (nat.choose num_vertices 2) - num_vertices
  let pairs_of_diagonals := nat.choose num_diagonals 2
  let num_intersecting_diagonals := nat.choose num_vertices 4
  (num_intersecting_diagonals / pairs_of_diagonals) = 6 / 13 :=
by
  sorry

end intersecting_diagonals_prob_nonagon_l446_446957


namespace tower_count_mod_1000_l446_446220

def num_towers (cubes : List ℕ) : ℕ := 
  have T_2 := 2
  have T_3 := 4 * T_2
  have T_4 := 4 * T_3
  have T_5 := 4 * T_4
  have T_6 := 4 * T_5
  have T_7 := 4 * T_6
  T_7

def cube_length_constraints (cubes : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → cubes.get i ≤ cubes.get j + 2

theorem tower_count_mod_1000 (cubes : List ℕ) (h : cube_length_constraints cubes ∧ List.length cubes = 7) :
  num_towers cubes % 1000 = 48 :=
  sorry

end tower_count_mod_1000_l446_446220


namespace sum_third_fifth_l446_446727

theorem sum_third_fifth (a x : ℕ) (h : 2 * a + 2 * x = 12) :
  (a + x) + (a + 3 * x) = 12 :=
by
  have h₁ : a + x = 6,
  { linarith, }
  have h₂ : 2 * (a + x) = 12,
  { linarith, }
  have h₃ : 2 * a + 4 * x = 12,
  { linarith, }
  linarith

end sum_third_fifth_l446_446727


namespace annual_haircut_cost_l446_446031

-- Define conditions as Lean definitions
def hair_growth_rate_per_month : ℝ := 1.5
def initial_hair_length : ℝ := 9
def post_haircut_length : ℝ := 6
def haircut_cost : ℝ := 45
def tip_percentage : ℝ := 0.2

-- The question to be answered as a theorem
theorem annual_haircut_cost :
  (hair_growth_rate_per_month > 0) →
  (initial_hair_length > post_haircut_length) →
  let length_cut := initial_hair_length - post_haircut_length in
  let months_between_haircuts := length_cut / hair_growth_rate_per_month in
  let haircuts_per_year := 12 / months_between_haircuts in
  let tip_amount := haircut_cost * tip_percentage in
  let cost_per_haircut := haircut_cost + tip_amount in
  haircuts_per_year * cost_per_haircut = 324 := 
by 
  -- Skipping the proof
  sorry

end annual_haircut_cost_l446_446031


namespace sum_of_digits_of_greatest_prime_divisor_of_8191_is_10_l446_446156

-- Definitions derived directly from the conditions
def is_prime (n : ℕ) : Prop := nat.prime n

def sum_of_digits (n : ℕ) : ℕ :=
(n.digits 10).sum

def greatest_prime_divisor (n : ℕ) : ℕ :=
nat.gcd n (nat.min_fac n)

-- The theorem to be proved
theorem sum_of_digits_of_greatest_prime_divisor_of_8191_is_10 :
  sum_of_digits (greatest_prime_divisor 8191) = 10 :=
sorry

end sum_of_digits_of_greatest_prime_divisor_of_8191_is_10_l446_446156


namespace complete_the_square_d_l446_446160

theorem complete_the_square_d (x : ℝ) : (∃ c d : ℝ, x^2 + 6 * x - 4 = 0 → (x + c)^2 = d) ∧ d = 13 :=
by
  sorry

end complete_the_square_d_l446_446160


namespace limits_do_not_exist_l446_446039

noncomputable def f (x : ℝ) : ℝ := 1 / (⌊|x|⌋)
noncomputable def g (x : ℝ) : ℝ := 1 / (|⌊x⌋|)

theorem limits_do_not_exist :
  ¬(∃ L : ℝ, filter.tendsto f (nhds_within (-1 : ℝ) (set.Ioi (-1))) (nhds L)) ∧
  ¬(∃ L : ℝ, filter.tendsto g (nhds_within (1 : ℝ) (set.Iio 1)) (nhds L)) :=
sorry

end limits_do_not_exist_l446_446039


namespace jeff_total_cabinets_l446_446756

def initial_cabinets : ℕ := 3
def cabinets_per_counter : ℕ := 2 * initial_cabinets
def total_cabinets_installed : ℕ := 3 * cabinets_per_counter + 5
def total_cabinets (initial : ℕ) (installed : ℕ) : ℕ := initial + installed

theorem jeff_total_cabinets : total_cabinets initial_cabinets total_cabinets_installed = 26 :=
by
  sorry

end jeff_total_cabinets_l446_446756


namespace triangle_XYZ_area_l446_446593

theorem triangle_XYZ_area :
  let X := (0, 0)
  let Y := (4, 0)
  let Z := (4, -2)
  let XY := 4
  let YZ := 2
  let XZ := 5
  let right_triangle := ∃ x y z : ℝ, (sqrt ((x - 0)^2 + (y - 0)^2) = XY) ∧
                                          (sqrt ((x - 4)^2 + (y - (-2))^2) = YZ) ∧
                                          (sqrt ((4 - 0)^2 + ((-2) - 0)^2) = XZ)
  in right_triangle → (1/2) * XY * YZ = 4 := sorry

end triangle_XYZ_area_l446_446593


namespace egg_processing_plant_l446_446004

-- Definitions based on the conditions
def original_ratio (E : ℕ) : ℕ × ℕ := (24 * E / 25, E / 25)
def new_ratio (E : ℕ) : ℕ × ℕ := (99 * E / 100, E / 100)

-- The mathematical proof problem
theorem egg_processing_plant (E : ℕ) (h : new_ratio E = (original_ratio E).fst + 12, (original_ratio E).snd) : E = 400 := 
  sorry

end egg_processing_plant_l446_446004


namespace radius_of_xz_intersection_l446_446589

-- Definitions based on conditions
def sphere_center : ℝ × ℝ × ℝ := (3, 5, -6)
def radius_in_xy_plane : ℝ := 3
def radius_in_xz_plane : ℝ := 6

-- Mathematically equivalent Lean 4 statement of the problem
theorem radius_of_xz_intersection :
  let R := real.sqrt (3^2 + 6^2)
  (R = 3 * real.sqrt 5) →
  radius_in_xz_plane = real.sqrt (R^2 - 3^2) →
  radius_in_xz_plane = 6 :=
by
  sorry

end radius_of_xz_intersection_l446_446589


namespace card_arrangement_l446_446810

theorem card_arrangement :
  ∃ (a b c d e f g h : ℕ), 
    a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ g ≠ h ∧ 
    {a, b, c, d, e, f, g, h} = {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
    abs (a - b) = 2 ∧ 
    abs (c - d) = 3 ∧ 
    abs (e - f) = 4 ∧ 
    abs (g - h) = 5 :=
by
  sorry

end card_arrangement_l446_446810


namespace dispersion_measures_l446_446166
-- Definitions for statistical measures (for clarity, too simplistic)
def standard_deviation (x : List ℝ) : ℝ := 
  let mean := (x.sum / x.length)
  Math.sqrt ((x.map (λ xi => (xi - mean)^2)).sum / (x.length - 1))

def median (x : List ℝ) : ℝ := 
  let sorted := x.qsort (≤)
  if h : sorted.length % 2 = 1 then (sorted.sorted.nth (sorted.length / 2))
  else ((sorted.nth (sorted.length / 2 - 1) + sorted.nth (sorted.length / 2)) / 2)

def range (x : List ℝ) : ℝ := x.maximum - x.minimum

def mean (x : List ℝ) : ℝ := x.sum / x.length

-- Statement to prove
theorem dispersion_measures (x : List ℝ) : 
  (standard_deviation x ∈ {standard_deviation x, range x}) ∧ 
  (range x ∈ {standard_deviation x, range x}) ∧
  ¬ (median x ∈ {standard_deviation x, range x})  ∧
  ¬ (mean x ∈ {standard_deviation x, range x}) := 
sorry

end dispersion_measures_l446_446166


namespace exercise_problem_l446_446322

variables (y : ℝ → ℝ) (a : ℝ) (f : ℝ → ℝ)

def proposition_p : Prop := ∀ (x : ℝ), y = 2 - a^(x+1) → y (1) = 2
def proposition_q : Prop := (∀ (x : ℝ), f (x-1) = f (-x+1)) → (∀ (x : ℝ), f x = f (-(x-1)))

theorem exercise_problem : ¬ (proposition_p y a) ∧ ¬ (proposition_q f) :=
by
  sorry

end exercise_problem_l446_446322


namespace sum_of_digits_unique_n_l446_446642

theorem sum_of_digits_unique_n :
  ∃ n : ℕ, (log 3 (log 27 n) = log 9 (log 9 n)) ∧ (∑ c in (to_digits 10 n), c = 11) :=
sorry

end sum_of_digits_unique_n_l446_446642


namespace range_of_expression_meaningful_l446_446359

theorem range_of_expression_meaningful (x : ℝ) : 
  (x >= -2 ∧ x ≠ 0) ↔ (∃ x, (sqrt (x + 2)) + (1 / x) ∈ ℝ) :=
by
  sorry

end range_of_expression_meaningful_l446_446359


namespace min_value_abs_b_minus_c_l446_446323

-- Define the problem conditions
def condition1 (a b c : ℝ) : Prop :=
  (a - 2 * b - 1)^2 + (a - c - Real.log c)^2 = 0

-- Define the theorem to be proved
theorem min_value_abs_b_minus_c {a b c : ℝ} (h : condition1 a b c) : |b - c| = 1 :=
sorry

end min_value_abs_b_minus_c_l446_446323


namespace jonathans_sister_first_name_letters_l446_446412

theorem jonathans_sister_first_name_letters
  (jonathan_fname_letters : Nat)
  (jonathan_sname_letters : Nat)
  (sister_sname_letters : Nat)
  (total_letters : Nat) :
  jonathan_fname_letters = 8 →
  jonathan_sname_letters = 10 →
  sister_sname_letters = 10 →
  total_letters = 33 →
  ∃ (sister_fname_letters : Nat), jonathan_fname_letters + jonathan_sname_letters + sister_fname_letters + sister_sname_letters = total_letters ∧ sister_fname_letters = 5 :=
by
  intros h1 h2 h3 h4
  have h_jonathan : jonathan_fname_letters + jonathan_sname_letters = 18 := by
    rw [h1, h2]
    exact rfl
  have h_sister_total : ∀ (sister_fname_letters : Nat), 18 + (sister_fname_letters + sister_sname_letters) = total_letters → sister_fname_letters = 5 := by
    intros x h
    have h_eq : x + 28 = total_letters := by
      rw [← add_assoc, add_assoc 18, add_comm 10]
      exact h
    have h_final : x = 5 := by
      rw h4 at h_eq
      exact Nat.sub_eq_iff_eq_add.mpr (eq.symm h_eq)
    rw h4
    exact h_final
  apply Exists.intro 5
  constructor
  · have : 18 + (5 + 10) = 33 := by
      norm_num
    exact this
  · exact rfl


end jonathans_sister_first_name_letters_l446_446412


namespace returns_to_starting_point_after_7th_passenger_final_position_after_last_passenger_l446_446831

def lao_yao_distances : List ℤ := [+5, -3, +6, -7, +6, -2, -5, +4, +6, -8]

/-- Prove that Lao Yao returns to the starting point after dropping off the 7th passenger -/
theorem returns_to_starting_point_after_7th_passenger :
  (lao_yao_distances.take 7).sum = 0 :=
by
  -- Proof to be provided
  sorry

/-- Prove that Lao Yao is 2 km to the east of the starting point after dropping off the last passenger -/
theorem final_position_after_last_passenger :
  lao_yao_distances.sum = 2 :=
by
  -- Proof to be provided
  sorry

end returns_to_starting_point_after_7th_passenger_final_position_after_last_passenger_l446_446831


namespace find_n_of_divisors_product_l446_446504

theorem find_n_of_divisors_product (n : ℕ) (h1 : 0 < n)
  (h2 : ∏ k in (finset.filter (λ d : ℕ, d ∣ n) (finset.range (n + 1))), d = 1024) :
  n = 16 :=
sorry

end find_n_of_divisors_product_l446_446504


namespace solve_equations_l446_446080

theorem solve_equations :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧ (x1^2 - 4 * x1 - 1 = 0) ∧ (x2^2 - 4 * x2 - 1 = 0)) ∧
  (∃ y1 y2 : ℝ, y1 = -4 ∧ y2 = 1 ∧ ((y1 + 4)^2 = 5 * (y1 + 4)) ∧ ((y2 + 4)^2 = 5 * (y2 + 4))) :=
by
  sorry

end solve_equations_l446_446080


namespace num_int_satisfying_conditions_l446_446348

theorem num_int_satisfying_conditions : 
  let count_n := (∀ n : ℤ, 150 < n ∧ n < 300 ∧ (n % 7 = n % 9) → (∃! k : ℤ, k ∈ {3, 4} ∧ ∀ r :ℤ, 0 ≤ r ∧ r ≤ 6 → n = 63 * k + r)) in
  count_n = 14 :=
sorry

end num_int_satisfying_conditions_l446_446348


namespace find_n_l446_446512

-- Define that n is a positive integer
def positive_integer (n : ℕ) : Prop := n > 0

-- Define number of divisors
def num_divisors (n : ℕ) : ℕ := (finset.range (n+1)).filter (λ d, n % d = 0).card

-- Define the product of divisors function
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
(finset.range (n+1)).filter (λ d, n % d = 0).prod id

-- The final theorem statement to be proven
theorem find_n (n : ℕ) (hn : positive_integer n) :
  prod_of_divisors n = 1024 → n = 16 :=
by { sorry }

end find_n_l446_446512


namespace systematic_sampling_draw_l446_446543

theorem systematic_sampling_draw
  (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 8)
  (h2 : 160 ≥ 8 * 20)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 → 
    160 ≥ ((k - 1) * 8 + 1 + 7))
  (h4 : ∀ y : ℕ, y = 1 + (15 * 8) → y = 126)
: x = 6 := 
sorry

end systematic_sampling_draw_l446_446543


namespace sum_of_smallest_elements_in_intersection_l446_446417

theorem sum_of_smallest_elements_in_intersection :
  let A := { x | ∃ n : ℕ+, x = n * (n + 1) }
  let B := { y | ∃ m : ℕ+, y = m * (m + 1) * (m + 2) }
  let C := A ∩ B
  C.nonempty → (∃ a b : ℕ, a ∈ C ∧ b ∈ C ∧ a < b ∧ (∀ c ∈ C, c = a ∨ c = b) ∧ a + b = 216) :=
by
  let A := { x | ∃ n : ℕ+, x = n * (n + 1) }
  let B := { y | ∃ m : ℕ+, y = m * (m + 1) * (m + 2) }
  let C := A ∩ B
  sorry

end sum_of_smallest_elements_in_intersection_l446_446417


namespace correct_factorization_l446_446163

theorem correct_factorization :
  (∀ (x y : ℝ), x^2 + y^2 ≠ (x + y)^2) ∧
  (∀ (x y : ℝ), x^2 + 2*x*y + y^2 ≠ (x - y)^2) ∧
  (∀ (x : ℝ), x^2 + x ≠ x * (x - 1)) ∧
  (∀ (x y : ℝ), x^2 - y^2 = (x + y) * (x - y)) :=
by 
  sorry

end correct_factorization_l446_446163


namespace find_a_l446_446691

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, x - 2 * a * y - 3 = 0 ∧ x^2 + y^2 - 2 * x + 2 * y - 3 = 0) → a = 1 :=
by
  sorry

end find_a_l446_446691


namespace value_of_a_l446_446042

def A := { x : ℝ | x^2 - 8*x + 15 = 0 }
def B (a : ℝ) := { x : ℝ | x * a - 1 = 0 }

theorem value_of_a (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end value_of_a_l446_446042


namespace correct_factorization_l446_446164

theorem correct_factorization :
  (∀ (x y : ℝ), x^2 + y^2 ≠ (x + y)^2) ∧
  (∀ (x y : ℝ), x^2 + 2*x*y + y^2 ≠ (x - y)^2) ∧
  (∀ (x : ℝ), x^2 + x ≠ x * (x - 1)) ∧
  (∀ (x y : ℝ), x^2 - y^2 = (x + y) * (x - y)) :=
by 
  sorry

end correct_factorization_l446_446164


namespace complex_product_polar_form_l446_446280

def cis (θ : ℝ) : ℂ := complex.exp (complex.I * θ)

theorem complex_product_polar_form :
  let z1 := 5 * cis (real.pi / 4)
  let z2 := -3 * cis (real.pi / 3)
  let z := z1 * z2
  let r := complex.abs z
  let θ := complex.arg z
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * real.pi) →
  (r = 15 ∧ θ = 19 * real.pi / 12) :=
by
  sorry

end complex_product_polar_form_l446_446280


namespace unique_rational_solution_l446_446203

theorem unique_rational_solution (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 := 
by {
  sorry
}

end unique_rational_solution_l446_446203


namespace find_n_of_divisors_product_l446_446508

theorem find_n_of_divisors_product (n : ℕ) (h1 : 0 < n)
  (h2 : ∏ k in (finset.filter (λ d : ℕ, d ∣ n) (finset.range (n + 1))), d = 1024) :
  n = 16 :=
sorry

end find_n_of_divisors_product_l446_446508


namespace find_g_53_l446_446847

variable (g : ℝ → ℝ)

axiom functional_eq (x y : ℝ) : g (x * y) = y * g x
axiom g_one : g 1 = 10

theorem find_g_53 : g 53 = 530 :=
by
  sorry

end find_g_53_l446_446847


namespace product_remainder_mod_7_l446_446875

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l446_446875


namespace minimize_sum_of_squares_l446_446432

theorem minimize_sum_of_squares (x1 x2 x3 : ℝ) (hpos1 : 0 < x1) (hpos2 : 0 < x2) (hpos3 : 0 < x3)
  (h_eq : x1 + 3 * x2 + 5 * x3 = 100) : x1^2 + x2^2 + x3^2 = 2000 / 7 := 
sorry

end minimize_sum_of_squares_l446_446432


namespace don_walking_speed_l446_446256

theorem don_walking_speed 
  (distance_between_homes : ℝ)
  (cara_walking_speed : ℝ)
  (cara_distance_before_meeting : ℝ)
  (time_don_starts_after_cara : ℝ)
  (total_distance : distance_between_homes = 45)
  (cara_speed : cara_walking_speed = 6)
  (cara_distance : cara_distance_before_meeting = 30)
  (time_after_cara : time_don_starts_after_cara = 2) :
  ∃ (v : ℝ), v = 5 := by
    sorry

end don_walking_speed_l446_446256


namespace product_remainder_mod_7_l446_446874

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l446_446874


namespace probability_relatively_prime_to_50_l446_446930

open Nat

def count_relatively_prime (n : ℕ) (m : ℕ) : ℕ :=
  (range m).filter (λ x => x.gcd n = 1).length

theorem probability_relatively_prime_to_50 : 
  (count_relatively_prime 50 50) / 50 = 2 / 5 :=
by
  sorry

end probability_relatively_prime_to_50_l446_446930


namespace divisible_by_eight_l446_446542

def expr (n : ℕ) : ℕ := 3^(4*n + 1) + 5^(2*n + 1)

theorem divisible_by_eight (n : ℕ) : expr n % 8 = 0 :=
  sorry

end divisible_by_eight_l446_446542


namespace sum_of_squares_of_perfect_squares_less_than_500_l446_446147

theorem sum_of_squares_of_perfect_squares_less_than_500 :
  (∑ n in {1, 16, 81, 256}, n) = 354 :=
by {
  sorry
}

end sum_of_squares_of_perfect_squares_less_than_500_l446_446147


namespace max_standing_people_l446_446246

def num_people := 55
def num_barons := 25
def num_counts := 20
def num_marquises := 10
def standing_condition (left middle right : String) := (left = right)

theorem max_standing_people (a : Fin num_people) (title : Fin num_people → String)
  (H1 : ∃! (i j k : Fin num_people), standing_condition (title i) (title j) (title k) ∧ i ≠ j ∧ j ≠ k ∧ k ≠ i):
  (∑ (i : Fin num_people), if standing_condition (title (i - 1)) (title i) (title (i + 1)) then 1 else 0) = 52 := 
sorry

end max_standing_people_l446_446246


namespace sequence_general_term_sum_of_sequence_bounded_l446_446661

theorem sequence_general_term (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : S 1 = (1 / 2) * a 1 * a 2) (h2 : ∀ n : ℕ, n ≥ 1 → S (n + 1) = (1 / 2) * a (n + 1) * a (n + 2)) (h3 : a 1 = 1) :
  ∀ n : ℕ, n ≥ 1 → a n = n :=
sorry

theorem sum_of_sequence_bounded (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n ≥ 1 → a n = n) :
  ∀ n : ℕ, (n > 0 → ∑ i in finset.range (n + 1), (1 / (a (i + 1))^2) < 7 / 4) :=
sorry

end sequence_general_term_sum_of_sequence_bounded_l446_446661


namespace volume_of_solid_is_37_pi_l446_446619

/-- Definitions for the conditions given in the problem --/
def part1_cylinder_radius := 5
def part1_cylinder_height := 1
def part2_cylinder_radius := 2
def part2_cylinder_height := 3

/-- Calculations based on the given conditions --/
def part1_volume := Real.pi * (part1_cylinder_radius ^ 2) * part1_cylinder_height
def part2_volume := Real.pi * (part2_cylinder_radius ^ 2) * part2_cylinder_height

/-- The total volume by summing the volumes of the two parts --/
def total_volume := part1_volume + part2_volume

/-- The theorem we need to prove --/
theorem volume_of_solid_is_37_pi : total_volume = 37 * Real.pi := by
  sorry

end volume_of_solid_is_37_pi_l446_446619


namespace smallest_twice_covered_area_l446_446925

-- Define the conditions of the problem
def paper_strip_width : Real := 5

-- The property we aim to prove
theorem smallest_twice_covered_area : ∃ (area : Real), area = 12.5 := by
  -- The height of the triangle formed
  let height := paper_strip_width
  -- The base of the triangle formed in the minimal configuration
  let base := paper_strip_width
  -- Calculate the area using the formula for the area of a triangle
  let area := (1 / 2) * base * height
  -- Simplify and verify it is as expected
  have h : area = 12.5 := by
    calc
      area = (1 / 2) * base * height : by rfl
          ... = (1 / 2) * 5 * 5        : by rfl
          ... = 12.5                   : by norm_num
  exact ⟨area, h⟩

end smallest_twice_covered_area_l446_446925


namespace find_a2_plus_b2_l446_446048

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) : a^2 + b^2 = 68 :=
sorry

end find_a2_plus_b2_l446_446048


namespace probability_diagonals_intersecting_in_nonagon_l446_446950

theorem probability_diagonals_intersecting_in_nonagon :
  let num_diagonals := (finset.card (finset.univ : finset (fin (nat.choose 9 2))) - 9) in
  let num_intersections := (nat.choose 9 4) in
  let total_diagonal_pairs := (nat.choose num_diagonals 2) in
  (num_intersections.to_rat / total_diagonal_pairs.to_rat) = (14/39 : ℚ) :=
by
  sorry

end probability_diagonals_intersecting_in_nonagon_l446_446950


namespace total_rotated_volume_l446_446264

-- Definitions for the conditions.
def rectangle_dimensions := (2, 3)
def square_dimensions := 2

-- Volumes of the resulting solids when the region is rotated about the y-axis.
def volume_cylinder (radius height : ℝ) : ℝ :=
  π * radius^2 * height

-- Given conditions translated into Lean.
def volume_rectangle_rotated :=
  volume_cylinder 3 2

def volume_square_rotated :=
  volume_cylinder 2 2

-- Problem statement: Prove that the total volume is 26π cubic units.
theorem total_rotated_volume :
  volume_rectangle_rotated + volume_square_rotated = 26 * π :=
by
  sorry

end total_rotated_volume_l446_446264


namespace dispersion_statistics_l446_446184

-- Define the variables and possible statistics options
def sample (n : ℕ) := fin n → ℝ

inductive Statistics
| StandardDeviation : Statistics
| Median : Statistics
| Range : Statistics
| Mean : Statistics

-- Define a function that returns if a statistic measures dispersion
def measures_dispersion : Statistics → Prop
| Statistics.StandardDeviation := true
| Statistics.Median := false
| Statistics.Range := true
| Statistics.Mean := false

-- Prove that StandardDeviation and Range measure dispersion
theorem dispersion_statistics (x : sample n) :
  measures_dispersion Statistics.StandardDeviation ∧
  measures_dispersion Statistics.Range :=
by
  split;
  exact trivial

end dispersion_statistics_l446_446184


namespace min_value_of_CO_l446_446737

variables (A B C O : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O]
variables (AC BC : ℝ)
variables (x y : ℝ)
variables (CA CB CO : Type) [Inhabited CA] [Inhabited CB] [Inhabited CO]
variables (f : ℝ → ℝ) 

noncomputable def min_value_CO (AC BC : ℝ) (x y : ℝ) (CA CB : Type) [Inhabited CA] [Inhabited CB] (f : ℝ → ℝ) : ℝ :=
  if AC = 2 ∧ BC = 2 ∧ x + y = 1 ∧ ∀ λ, f(λ) = _root_.abs (2 * λ * _root_.cos (real.pi / 3)) 
  then sqrt 3 
  else 0

theorem min_value_of_CO :
  min_value_CO 2 2 x y CA CB f = sqrt 3 := sorry

end min_value_of_CO_l446_446737


namespace sandy_correct_sums_l446_446465

theorem sandy_correct_sums :
  ∃ x y: ℕ, x + y = 30 ∧ 3 * x - 2 * y = 65 ∧ x = 25 :=
by
  use [25, 5] -- Here we provide the values directly to validate the proof structure
  split
  · exact rfl
  split
  · exact rfl
  · exact rfl

end sandy_correct_sums_l446_446465


namespace lifestyle_risk_factors_l446_446800

theorem lifestyle_risk_factors (p : ℚ) (q : ℚ) (cond : ℚ) (s t : ℕ) (h_prime : Nat.coprime s t) :
  p = 0.05 ∧ q = 0.07 ∧ cond = 1 / 4 ∧ (200 - 3 * 10 - 3 * 14 - (1 / 3 * 28 : ℚ)) / (200 - (10 + 2 * 14 + (1 / 3 * 28 : ℚ))) = s / t →
  s + t = 276 :=
begin
  sorry
end

end lifestyle_risk_factors_l446_446800


namespace modulus_of_complex_number_l446_446851

open Complex

def complex_number : ℂ := (2 - I) / (1 + I)

def modulus_is_sqrt_ten_div_two : Prop :=
  abs complex_number = Real.sqrt 10 / 2

theorem modulus_of_complex_number : modulus_is_sqrt_ten_div_two :=
  by
  sorry

end modulus_of_complex_number_l446_446851


namespace sum_f_k_l446_446496

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 (x : ℝ) : f x = - f (x + 3 / 2)

axiom condition2 : f (-1) = 1

axiom condition3 : f 0 = -2

theorem sum_f_k : (∑ k in Finset.range 2024, f (k + 1)) = 2 :=
by
  sorry

end sum_f_k_l446_446496


namespace dispersion_measured_by_std_dev_and_range_l446_446179

variables {α : Type*} [linear_order α] (x : list α)

def standard_deviation (l : list ℝ) : ℝ := sorry -- definition of standard_deviation
def median (l : list α) : α := sorry -- definition of median
def range (l : list ℝ) : ℝ := sorry -- definition of range
def mean (l : list ℝ) : ℝ := sorry -- definition of mean

theorem dispersion_measured_by_std_dev_and_range :
  (standard_deviation (map (λ x, (x : ℝ)) (x : list α)) > 0 ∨ range (map (λ x, (x : ℝ)) (x : list α)) > 0) →
  (∀ x, x ∈ [standard_deviation (map (λ x, (x : ℝ)) (x : list α)), range (map (λ x, (x : ℝ)) (x : list α))]) :=
begin
  sorry
end

end dispersion_measured_by_std_dev_and_range_l446_446179


namespace exists_product_sum_20000_l446_446138

theorem exists_product_sum_20000 :
  ∃ (k m : ℕ), 1 ≤ k ∧ k ≤ 999 ∧ 1 ≤ m ∧ m ≤ 999 ∧ k * (k + 1) + m * (m + 1) = 20000 :=
by 
  sorry

end exists_product_sum_20000_l446_446138


namespace trapezoid_area_bc_l446_446479

theorem trapezoid_area_bc 
  (area_abcd : ℝ)
  (altitude : ℝ)
  (ab : ℝ)
  (cd : ℝ)
  (h1 : area_abcd = 180)
  (h2 : altitude = 8)
  (h3 : ab = 14)
  (h4 : cd = 20) :
  let bc := (180 - 8 * real.sqrt 33 - 16 * real.sqrt 21) / 8 in
  bc = 22.5 - real.sqrt 33 - 2 * real.sqrt 21 := 
by
  sorry

end trapezoid_area_bc_l446_446479


namespace product_of_divisors_eq_1024_l446_446516

theorem product_of_divisors_eq_1024 (n : ℕ) (h : n > 0) (hp : ∏ d in (finset.filter (λ x, x ∣ n) (finset.range (n+1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446516


namespace kate_needs_more_money_l446_446414

theorem kate_needs_more_money
  (pen_price : ℝ)
  (notebook_price : ℝ)
  (artset_price : ℝ)
  (kate_pen_money_fraction : ℝ)
  (notebook_discount : ℝ)
  (artset_discount : ℝ)
  (kate_artset_money : ℝ) :
  pen_price = 30 →
  notebook_price = 20 →
  artset_price = 50 →
  kate_pen_money_fraction = 1/3 →
  notebook_discount = 0.15 →
  artset_discount = 0.4 →
  kate_artset_money = 10 →
  (pen_price - kate_pen_money_fraction * pen_price) +
  (notebook_price * (1 - notebook_discount)) +
  (artset_price * (1 - artset_discount) - kate_artset_money) = 57 :=
by
  sorry

end kate_needs_more_money_l446_446414


namespace probability_relatively_prime_to_50_l446_446929

open Nat

def count_relatively_prime (n : ℕ) (m : ℕ) : ℕ :=
  (range m).filter (λ x => x.gcd n = 1).length

theorem probability_relatively_prime_to_50 : 
  (count_relatively_prime 50 50) / 50 = 2 / 5 :=
by
  sorry

end probability_relatively_prime_to_50_l446_446929


namespace oranges_savings_l446_446790

-- Definitions for the conditions
def liam_oranges : Nat := 40
def liam_price_per_set : Real := 2.50
def oranges_per_set : Nat := 2

def claire_oranges : Nat := 30
def claire_price_per_orange : Real := 1.20

-- Statement of the problem to be proven
theorem oranges_savings : 
  liam_oranges / oranges_per_set * liam_price_per_set + 
  claire_oranges * claire_price_per_orange = 86 := 
by 
  sorry

end oranges_savings_l446_446790


namespace jim_total_cost_l446_446192

noncomputable def total_cost : ℝ :=
let
  -- Prices before discount
  lamp_cost := 12
  bulb_cost := lamp_cost - 4
  bedside_table_cost := 25
  decorative_item_cost := 10
  
  -- Quantity bought
  lamp_qty := 2
  bulb_qty := 6
  bedside_table_qty := 3
  decorative_item_qty := 4
  
  -- Discounts
  lamp_discount := 0.20
  bulb_discount := 0.30
  decorative_item_discount := 0.15
  
  -- Tax rates
  lamp_bulb_tax := 0.05
  bedside_table_tax := 0.06
  decorative_item_tax := 0.04
  
  -- Calculate discounted prices
  lamp_total := lamp_qty * lamp_cost * (1 - lamp_discount)
  bulb_total := bulb_qty * bulb_cost * (1 - bulb_discount)
  bedside_table_total := bedside_table_qty * bedside_table_cost
  decorative_item_total := decorative_item_qty * decorative_item_cost * (1 - decorative_item_discount)
  
  -- Apply taxes
  lamp_total_with_tax := lamp_total * (1 + lamp_bulb_tax)
  bulb_total_with_tax := bulb_total * (1 + lamp_bulb_tax)
  bedside_table_total_with_tax := bedside_table_total * (1 + bedside_table_tax)
  decorative_item_total_with_tax := decorative_item_total * (1 + decorative_item_tax)
  
  -- Total cost sum
  total := lamp_total_with_tax + bulb_total_with_tax + bedside_table_total_with_tax + decorative_item_total_with_tax
in total

theorem jim_total_cost : total_cost = 170.30 :=
by
  sorry
  -- If we filled this in, it would follow the method to show total_cost = 170.30

end jim_total_cost_l446_446192


namespace percent_calculation_l446_446716

theorem percent_calculation (x : ℕ) (p : ℕ) (h1 : x = 180) (h2 : 0.25 * x = (p / 100) * 500 - 5) : p = 10 :=
by
  sorry

end percent_calculation_l446_446716


namespace dodecahedron_interior_diagonals_l446_446708

-- Define the conditions for a dodecahedron:
def dodecahedron : Type := 
{ faces : ℕ, vertices : ℕ, vertices_per_face : ℕ, faces_per_vertex : ℕ } 

-- Define the number of interior diagonals in a dodecahedron:
def number_of_interior_diagonals (d : dodecahedron) : ℕ :=
  let total_vertex_pairs := (d.vertices.choose 2)
  let edges := 30
  let pairs_sharing_face := 12 * (d.vertices_per_face.choose 2)
  total_vertex_pairs - edges - pairs_sharing_face

-- Define a specific dodecahedron instance:
def specific_dodecahedron : dodecahedron := {
  faces := 12,
  vertices := 20,
  vertices_per_face := 5,
  faces_per_vertex := 3
}

-- Statement of the problem in Lean 4
theorem dodecahedron_interior_diagonals :
  number_of_interior_diagonals specific_dodecahedron = 40 :=
by {
  unfold number_of_interior_diagonals,
  sorry
}

end dodecahedron_interior_diagonals_l446_446708


namespace range_of_t_l446_446672

noncomputable def f (x : ℝ) : ℝ := log 3 (3^x + 1) - (1/2) * x

def g (x : ℝ) : ℝ := f x + (1/2) * x

def h (x t : ℝ) : ℝ := x^2 - 2 * t * x + 5

theorem range_of_t (t : ℝ) :
  (∃ x1 ∈ Set.Icc (real.log 2 / real.log 3) 8, ∀ x2 ∈ Set.Icc 1 4, g x1 ≤ h x2 t) →
  t ≤ 2 :=
sorry

end range_of_t_l446_446672


namespace second_candidate_extra_marks_l446_446967

theorem second_candidate_extra_marks (T : ℝ) (marks_40_percent : ℝ) (marks_passing : ℝ) (marks_60_percent : ℝ) 
  (h1 : marks_40_percent = 0.40 * T)
  (h2 : marks_passing = 160)
  (h3 : marks_60_percent = 0.60 * T)
  (h4 : marks_passing = marks_40_percent + 40) :
  (marks_60_percent - marks_passing) = 20 :=
by
  sorry

end second_candidate_extra_marks_l446_446967


namespace product_remainder_mod_7_l446_446872

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l446_446872


namespace expected_value_monicas_winnings_l446_446802

noncomputable def prob_roll_odd_sum : ℝ := 1 / 2
noncomputable def prob_roll_even_sum : ℝ := 1 / 2
noncomputable def prob_roll_doubles : ℝ := 1 / 6
noncomputable def prob_roll_even_non_doubles : ℝ := 1 / 3
noncomputable def expected_sum_two_dice : ℝ := 7
noncomputable def expected_winnings_odd_sum : ℝ := 4 * expected_sum_two_dice
noncomputable def expected_winnings_even_doubles : ℝ := 2 * expected_sum_two_dice
noncomputable def loss_even_non_doubles : ℝ := -6

theorem expected_value_monicas_winnings : 
  (prob_roll_odd_sum * expected_winnings_odd_sum + 
  prob_roll_doubles * expected_winnings_even_doubles + 
  prob_roll_even_non_doubles * loss_even_non_doubles) = 14.33 :=
by 
  -- Proof goes here
  sorry

end expected_value_monicas_winnings_l446_446802


namespace number_of_squares_l446_446423

-- Define the isosceles right triangle with the given conditions
structure IsoscelesRightTriangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
(angle_B_eq_90 : ∠ B = 90)
(length_AB_eq_BC : dist A B = dist B C)

-- Main theorem statement
theorem number_of_squares (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (t : IsoscelesRightTriangle A B C) : 
  ∃ (count : ℕ), count = 4 :=
begin
  use 4,
  sorry
end

end number_of_squares_l446_446423


namespace dot_product_magnitude_l446_446425

variable {V : Type*} [inner_product_space ℝ V]

-- Conditions
variables (a b : V) (ha : ∥a∥ = 3) (hb : ∥b∥ = 7) (h_cross : ∥a × b∥ = 18)

theorem dot_product_magnitude :
  |(inner_product_space.inn a b)| = 3 * Real.sqrt 13 :=
by
  -- We are required to prove the statement. The exact proof is reserved for later
  sorry

end dot_product_magnitude_l446_446425


namespace area_of_inscribed_rectangle_l446_446815

theorem area_of_inscribed_rectangle 
  (A B C D E F : ℝ) 
  (AD : ℝ) 
  (ED : ℝ)
  (AF : ℝ)
  (diameter_EF : ℝ) 
  (h1 : AD = 20)
  (h2 : ED = 6)
  (h3 : AF = 6) 
  (h4 : diameter_EF = 32)
  (h_inscribed : A = E ∧ D = F) :

  let radius := diameter_EF / 2 in 
  let OC    := √(radius^2 - (AD / 2)^2) in
  AD * OC = 20 * √156 :=
by 
  sorry

end area_of_inscribed_rectangle_l446_446815


namespace germs_per_dish_calc_l446_446561

theorem germs_per_dish_calc :
    let total_germs := 0.036 * 10^5
    let total_dishes := 36000 * 10^(-3)
    (total_germs / total_dishes) = 100 := by
    sorry

end germs_per_dish_calc_l446_446561


namespace semi_minor_axis_of_ellipse_l446_446735

theorem semi_minor_axis_of_ellipse :
  ∀ (center focus semi_major_axis_endpoint : ℝ×ℝ),
  center = (2, -3) →
  focus = (2, -5) →
  semi_major_axis_endpoint = (2, 0) →
  let c := abs (center.snd - focus.snd) in
  let a := abs (center.snd - semi_major_axis_endpoint.snd) in
  let b := real.sqrt (a^2 - c^2) in
  b = real.sqrt 5 :=
begin
  intros center focus semi_major_axis_endpoint,
  intros h_center h_focus h_semi_major_axis_endpoint,
  rw [h_center, h_focus, h_semi_major_axis_endpoint],
  let c := abs (-3 - (-5)),
  let a := abs (-3 - 0),
  let b := real.sqrt (a^2 - c^2),
  have hc : c = 2, by norm_num,
  have ha : a = 3, by norm_num,
  rw [hc, ha],
  have h_b : b = real.sqrt 5, by norm_num,
  exact h_b,
end

end semi_minor_axis_of_ellipse_l446_446735


namespace largest_subset_size_l446_446767

def is_valid_subset (S : Set ℤ) : Prop :=
  ∀ a b ∈ S, a ≠ b → (|a - b| ≠ 4 ∧ |a - b| ≠ 7)

theorem largest_subset_size : 
  ∃ S : Set ℤ, S ⊆ {n : ℤ | 1 ≤ n ∧ n ≤ 1989} ∧ is_valid_subset S ∧ S.card = 905 :=
sorry

end largest_subset_size_l446_446767


namespace area_triangle_QCA_l446_446621

-- Definition of the given points Q, A, and C
def Q := (0, 15)
def A := (3, 15)
def C (p : ℝ) := (0, p)

-- Proof statement for the area of triangle QCA
theorem area_triangle_QCA (p : ℝ) : 
  let QA := abs (A.1 - Q.1),
      QC := abs (Q.2 - C(p).2),
      Area := (1/2:ℝ) * QA * QC
  in Area = (45 - 3 * p) / 2 := 
by
  let QA := abs (A.1 - Q.1)
  let QC := abs (Q.2 - C(p).2)
  let Area := (1/2:ℝ) * QA * QC
  show Area = (45 - 3 * p) / 2, from sorry

end area_triangle_QCA_l446_446621


namespace sales_fraction_l446_446415

theorem sales_fraction (A D : ℝ) (h : D = 2 * A) : D / (11 * A + D) = 2 / 13 :=
by
  sorry

end sales_fraction_l446_446415


namespace find_AP2_AQ2_PQ2_l446_446011

variables {m n : ℝ} {A B C O P Q : ℝ}
variables (h1 : n < m / 2) 
variables (h2 : BC = m)
variables (h3 : O = (B + C) / 2)
variables (h4 : dist O P = n / 2) (h5 : dist O Q = n / 2)
variables (h6 : dist A O = m / 2)
variables (h7 : ∠AOP + ∠AOQ = 180)

theorem find_AP2_AQ2_PQ2 :
  dist A P ^ 2 + dist A Q ^ 2 + dist P Q ^ 2 = m^2 / 2 + 6 * n^2 :=
sorry

end find_AP2_AQ2_PQ2_l446_446011


namespace production_equation_correct_l446_446191

variable (x : ℝ) -- x is the production rate in millions of products per day before the update

-- Definition of the production rates
def production_rate_before_update : ℝ := x
def production_rate_after_update : ℝ := x + 30

-- Definition of the times to produce the products
def time_to_produce_4_million_before_update : ℝ := 400 / production_rate_before_update
def time_to_produce_5_million_after_update : ℝ := 500 / production_rate_after_update

-- The theorem to be proven
theorem production_equation_correct 
    (h: production_rate_before_update = x ∧ production_rate_after_update = x + 30): 
    time_to_produce_4_million_before_update = time_to_produce_5_million_after_update := 
by
    sorry

end production_equation_correct_l446_446191


namespace james_car_new_speed_l446_446404

-- Define the conditions and the statement to prove
variable (original_speed supercharge_increase weight_reduction : ℝ)
variable (original_speed_gt_zero : 0 < original_speed)

theorem james_car_new_speed :
  original_speed = 150 →
  supercharge_increase = 0.3 →
  weight_reduction = 10 →
  original_speed * (1 + supercharge_increase) + weight_reduction = 205 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- Calculate the speed after supercharging
  have supercharged_speed : ℝ := 150 * (1 + 0.3)
  calc
    150 * (1 + 0.3) + 10 = 195 + 10 : by norm_num
                       ... = 205 : by norm_num
  sorry

end james_car_new_speed_l446_446404


namespace pizza_slices_left_over_l446_446588

def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8
def small_pizzas_purchased : ℕ := 3
def large_pizzas_purchased : ℕ := 2
def george_slices : ℕ := 3
def bob_slices : ℕ := george_slices + 1
def susie_slices : ℕ := bob_slices / 2
def bill_slices : ℕ := 3
def fred_slices : ℕ := 3
def mark_slices : ℕ := 3

def total_pizza_slices : ℕ := (small_pizzas_purchased * small_pizza_slices) + (large_pizzas_purchased * large_pizza_slices)
def total_slices_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

theorem pizza_slices_left_over : total_pizza_slices - total_slices_eaten = 10 :=
by sorry

end pizza_slices_left_over_l446_446588


namespace wall_area_in_square_meters_l446_446728

variable {W H : ℤ} -- We treat W and H as integers referring to centimeters

theorem wall_area_in_square_meters 
  (h₁ : W / 30 = 8) 
  (h₂ : H / 30 = 5) : 
  (W / 100) * (H / 100) = 360 / 100 :=
by 
  sorry

end wall_area_in_square_meters_l446_446728


namespace max_area_ratio_triangle_l446_446061

theorem max_area_ratio_triangle {A B C I P : Type*} [nonempty A] [nonempty B] [nonempty C] [nonempty I] [nonempty P]
  (h1 : ∀ {A B C I : Type*}, ∃ (radius : ℝ), radius = 2)
  (h2 : ∀ {I P : Type*}, ∃ (distance : ℝ), distance = 1) :
  ∃ (max_ratio : ℝ), max_ratio = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end max_area_ratio_triangle_l446_446061


namespace johns_annual_haircut_cost_l446_446027

-- Define constants given in the problem
def hair_growth_rate : ℝ := 1.5 -- inches per month
def initial_hair_length_before_cut : ℝ := 9 -- inches
def initial_hair_length_after_cut : ℝ := 6 -- inches
def haircut_cost : ℝ := 45 -- dollars
def tip_percentage : ℝ := 0.2 -- 20%

-- Calculate the number of haircuts per year and the total cost including tips
def monthly_hair_growth := initial_hair_length_before_cut - initial_hair_length_after_cut
def haircut_interval := monthly_hair_growth / hair_growth_rate
def number_of_haircuts_per_year := 12 / haircut_interval
def tip_amount_per_haircut := haircut_cost * tip_percentage
def total_cost_per_haircut := haircut_cost + tip_amount_per_haircut
def total_annual_cost := number_of_haircuts_per_year * total_cost_per_haircut

-- Prove the total annual cost to be $324
theorem johns_annual_haircut_cost : total_annual_cost = 324 := by
  sorry

end johns_annual_haircut_cost_l446_446027


namespace find_x_l446_446706

-- Define the vectors and the conditions
noncomputable def vector_a : ℝ × ℝ × ℝ := (2, -1, 4)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)
noncomputable def vector_c (x : ℝ) : ℝ × ℝ × ℝ := (1, x, 2)

-- Define the dot product of vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the condition that vectors are perpendicular
def are_perpendicular (u v : ℝ × ℝ × ℝ) : Prop :=
dot_product u v = 0

-- State the theorem to be proved
theorem find_x (x : ℝ) : 
  let vector_sum := (vector_a.1 + vector_b(x).1, vector_a.2 + vector_b(x).2, vector_a.3 + vector_b(x).3) in
  are_perpendicular vector_sum (vector_c x) → x = -2 :=
by sorry

end find_x_l446_446706


namespace walnut_trees_orange_trees_count_l446_446128

def walnut_problem : ℕ → ℕ → Prop :=
λ initial remaining, initial - 13 = remaining

theorem walnut_trees :
  walnut_problem 42 29 :=
by
  sorry

def orange_trees_problem : Prop :=
  ∃ (orange_trees : ℕ), true

theorem orange_trees_count : ¬ orange_trees_problem :=
by
  sorry

end walnut_trees_orange_trees_count_l446_446128


namespace smallest_prime_divisor_of_sum_of_powers_l446_446146

theorem smallest_prime_divisor_of_sum_of_powers :
  let a := 5
  let b := 7
  let n := 23
  let m := 17
  Nat.minFac (a^n + b^m) = 2 := by
  sorry

end smallest_prime_divisor_of_sum_of_powers_l446_446146


namespace cotangent_half_angle_sum_eq_l446_446665

theorem cotangent_half_angle_sum_eq (a b c : ℝ) (A B C S : ℝ) (hA: a = 2*S / (b * sin B)) (hB: b = 2*S / (a * sin A)) (hC: c = 2*S / (a * sin A)) (hS: S = (1/2) * a * b * sin C):
  (Real.cot (A/2) + Real.cot (B/2) + Real.cot (C/2) = (a+b+c)^2 / (4 * S)) := 
by
  sorry

end cotangent_half_angle_sum_eq_l446_446665


namespace monotonic_intervals_range_of_m_l446_446657

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 3 - 2 * x)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 2 * x + m - 3

theorem monotonic_intervals :
  ∀ k : ℤ,
    (
      (∀ x, -Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 5 * Real.pi / 12 + k * Real.pi → ∃ (d : ℝ), f x = d)
      ∧
      (∀ x, 5 * Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 11 * Real.pi / 12 + k * Real.pi → ∃ (i : ℝ), f x = i)
    ) := sorry

theorem range_of_m (m : ℝ) :
  (∀ x1 : ℝ, Real.pi / 12 ≤ x1 ∧ x1 ≤ Real.pi / 2 → ∃ x2 : ℝ, -2 ≤ x2 ∧ x2 ≤ m ∧ f x1 = g x2 m) ↔ -1 ≤ m ∧ m ≤ 3 := sorry

end monotonic_intervals_range_of_m_l446_446657


namespace heptagon_folding_to_quadrilateral_l446_446475

/-- 
Given a convex heptagon (7-sided polygon), 
prove that there exists a folding process that results in a two-layer quadrilateral.
-/
theorem heptagon_folding_to_quadrilateral (P : Polygon) (h : P.is_convex ∧ P.num_sides = 7) :
  ∃ Q : Polygon, Q.num_sides = 4 ∧ Q.layers = 2 :=
by
  sorry

end heptagon_folding_to_quadrilateral_l446_446475


namespace percent_decrease_correct_l446_446245

def original_price_per_pack : ℚ := 7 / 3
def promotional_price_per_pack : ℚ := 8 / 4
def percent_decrease_in_price (old_price new_price : ℚ) : ℚ := 
  ((old_price - new_price) / old_price) * 100

theorem percent_decrease_correct :
  percent_decrease_in_price original_price_per_pack promotional_price_per_pack = 14 := by
  sorry

end percent_decrease_correct_l446_446245


namespace least_sum_of_bases_l446_446503

theorem least_sum_of_bases :
  ∃ (c d : ℕ), 0 < c ∧ 0 < d ∧ (5 * c + 8 = 8 * d + 5) ∧ (c + d = 15) :=
by
  -- Definitions for conditions
  let c := 9
  let d := 6
  have h1 : 0 < c := by simp [c, nat.succ_pos']
  have h2 : 0 < d := by simp [d, nat.succ_pos']
  have h3 : 5 * c + 8 = 8 * d + 5 := by calc
    5 * c + 8 = 5 * 9 + 8 : by simpa [c]
    ... = 45 + 8 : by simp
    ... = 53 : by simp
    ... = 48 + 5 : by simp
    ... = 8 * d + 5 : by simpa [d]
  have h4 : c + d = 15 := by calc
    c + d = 9 + 6 : by simpa [c, d]
    ... = 15 : by simp
  exact ⟨c, d, h1, h2, h3, h4⟩

end least_sum_of_bases_l446_446503


namespace pendulum_proportional_l446_446281

theorem pendulum_proportional (t l g : ℝ) (h : t = 2 * Real.pi * Real.sqrt (l / g)) : t ∝ Real.sqrt l :=
sorry

end pendulum_proportional_l446_446281


namespace min_tetrominoes_required_l446_446891

-- We have definitions for Tetromino and a 5x5 Grid
structure Tetromino :=
(square_count : ℕ)
(rotation_flip : ℕ) -- Number of rotation and flip configurations

def grid_size : ℕ := 5 * 5

def tetromino_square_count (t : Tetromino) : ℕ := t.square_count

-- Define a property that a placement leaves no space for additional tetrominoes
def no_additional_tetromino (placements : list Tetromino) : Prop :=
  grid_size - (list.sum (list.map tetromino_square_count placements)) < 4

-- Define the main theorem
theorem min_tetrominoes_required : ∃ n : ℕ, n = 3 ∧
  (∀ placements : list Tetromino, list.length placements = n → no_additional_tetromino placements) :=
sorry

end min_tetrominoes_required_l446_446891


namespace simson_line_l446_446813

-- Definitions of the objects involved in the problem
variables {A B C P D E F : Type}

-- Triangle with vertices A, B, and C
variables (triangle_ABC : affine_plane) 

-- Circumcircle of triangle ABC denoted as Γ
variables (Γ : circle triangle_ABC)

-- Point P lies on the circumcircle Γ
variable (hyp_P_on_Γ : P ∈ Γ)

-- D, E, and F are feet of the perpendiculars from P to the sides of the triangle ABC
variable (perpendiculars : (foot_of_perpendicular P A B) = D ∧
                          (foot_of_perpendicular P B C) = E ∧
                          (foot_of_perpendicular P C A) = F)

-- To prove: The points D, E, and F are collinear (Simson line)
theorem simson_line (triangle_ABC : affine_plane)
  (Γ : circle triangle_ABC) 
  (P : point) 
  (hyp_P_on_Γ : P ∈ Γ)
  (D E F : point) 
  (perpendiculars : (foot_of_perpendicular P (side A B triangle_ABC) = D ∧
                     foot_of_perpendicular P (side B C triangle_ABC) = E ∧
                     foot_of_perpendicular P (side C A triangle_ABC) = F)) :
collinear {D, E, F} :=
sorry

end simson_line_l446_446813


namespace book_increased_vocabulary_by_50_percent_l446_446537

-- Definition of the main parameters
def words_learned_per_day : ℕ := 10
def days_per_year : ℕ := 365
def years : ℕ := 2
def original_words : ℕ := 14600

-- Derived definitions based on problem conditions
def total_days : ℕ := days_per_year * years
def words_learned_in_2_years : ℕ := words_learned_per_day * total_days
def total_words_after_2_years : ℕ := original_words + words_learned_in_2_years

-- Percentage increase calculation
def percentage_increase (original new : ℕ) : ℚ :=
  ((new - original).toRat / original.toRat) * 100

-- The theorem we need to prove
theorem book_increased_vocabulary_by_50_percent 
  (h1 : words_learned_per_day = 10)
  (h2 : days_per_year = 365)
  (h3 : years = 2)
  (h4 : original_words = 14600) :
  percentage_increase original_words total_words_after_2_years = 50 :=
by
  sorry

end book_increased_vocabulary_by_50_percent_l446_446537


namespace ratio_of_areas_l446_446219

theorem ratio_of_areas (r : ℝ) (h_r : r = 3) : 
  let A_circle := π * r ^ 2 in 
  let l := 3 * r in
  let w := 2 * r in
  let A_rectangle := l * w in
  A_rectangle / A_circle = 6 / π := 
by
  sorry

end ratio_of_areas_l446_446219


namespace parabola_intersections_l446_446922

open Real

-- Definition of the two parabolas
def parabola1 (x : ℝ) : ℝ := 3*x^2 - 6*x + 2
def parabola2 (x : ℝ) : ℝ := 9*x^2 - 4*x - 5

-- Theorem stating the intersections are (-7/3, 9) and (0.5, -0.25)
theorem parabola_intersections : 
  {p : ℝ × ℝ | parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2} =
  {(-7/3, 9), (0.5, -0.25)} :=
by 
  sorry

end parabola_intersections_l446_446922


namespace solve_for_sum_l446_446699

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := -1
noncomputable def c : ℝ := Real.sqrt 26

theorem solve_for_sum :
  (a * (a - 4) = 5) ∧ (b * (b - 4) = 5) ∧ (c * (c - 4) = 5) ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a^2 + b^2 = c^2) → (a + b + c = 4 + Real.sqrt 26) :=
by
  sorry

end solve_for_sum_l446_446699


namespace inequality_solution_set_l446_446641

theorem inequality_solution_set (x : ℝ) :
  2 * x^2 - x ≤ 0 → 0 ≤ x ∧ x ≤ 1 / 2 :=
sorry

end inequality_solution_set_l446_446641


namespace solve_system_of_equations_l446_446828
noncomputable theory

theorem solve_system_of_equations 
  (x y : ℝ)
  (h1 : 3 * x + y = 8)
  (h2 : 2 * x - y = 7) :
  x = 3 ∧ y = -1 :=
sorry

end solve_system_of_equations_l446_446828


namespace equal_points_probability_l446_446902

theorem equal_points_probability (p : ℝ) (prob_draw_increases : 0 ≤ p ∧ p ≤ 1) :
  (∀ q : ℝ, (0 ≤ q ∧ q < p) → (q^2 + (1 - q)^2 / 2 < p^2 + (1 - p)^2 / 2)) → False :=
begin
  sorry
end

end equal_points_probability_l446_446902


namespace find_n_l446_446510

-- Define that n is a positive integer
def positive_integer (n : ℕ) : Prop := n > 0

-- Define number of divisors
def num_divisors (n : ℕ) : ℕ := (finset.range (n+1)).filter (λ d, n % d = 0).card

-- Define the product of divisors function
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
(finset.range (n+1)).filter (λ d, n % d = 0).prod id

-- The final theorem statement to be proven
theorem find_n (n : ℕ) (hn : positive_integer n) :
  prod_of_divisors n = 1024 → n = 16 :=
by { sorry }

end find_n_l446_446510


namespace solve_door_dimensions_l446_446842

theorem solve_door_dimensions :
  ∀ (frame_width : ℝ) (inner_area : ℝ) (shorter_squares longer_squares : ℕ), 
  frame_width = 5 →
  inner_area = 432 →
  shorter_squares = 3 →
  longer_squares = 4 →
  let a := (inner_area / (shorter_squares * longer_squares)) ^ (1 / 2) in
  let inner_width := shorter_squares * a in
  let inner_height := longer_squares * a in
  let external_width := inner_width + 2 * frame_width in
  let external_height := inner_height + 2 * frame_width in
  external_width = 28 ∧ external_height = 34 :=
by
  intros frame_width inner_area shorter_squares longer_squares
  intro hfw
  intro hiw
  intro hss
  intro hls
  let a := (inner_area / (shorter_squares * longer_squares)) ^ (1 / 2)
  let inner_width := shorter_squares * a
  let inner_height := longer_squares * a
  let external_width := inner_width + 2 * frame_width
  let external_height := inner_height + 2 * frame_width
  have h1 : a = 6 := sorry
  have h2 : inner_width = 18 := sorry
  have h3 : inner_height = 24 := sorry
  have h4 : external_width = 28 := by sorry
  have h5 : external_height = 34 := by sorry
  exact ⟨h4, h5⟩

end solve_door_dimensions_l446_446842


namespace identical_remainders_l446_446060

theorem identical_remainders (a : Fin 11 → Fin 11) (h_perm : ∀ n, ∃ m, a m = n) :
  ∃ (i j : Fin 11), i ≠ j ∧ (i * a i) % 11 = (j * a j) % 11 :=
by 
  sorry

end identical_remainders_l446_446060


namespace concurrent_lines_l446_446054

variables {A B C D X Y P M N : Point}
variables (h_collinear : collinear A B C D)
variables (h_diameter_1 : diameter_circle A C)
variables (h_diameter_2 : diameter_circle B D)
variables (h_intersection_1 : circle_intersect_points (diameter_circle A C) (diameter_circle B D) X Y)
variables (h_P_on_XY : lies_on P (line_through X Y))
variables (h_M_intersect : line_circle_intersect (line_through C P) (diameter_circle B D) M)
variables (h_N_intersect : line_circle_intersect (line_through B P) (diameter_circle A C) N)

theorem concurrent_lines :
  concurrent (line_through A M) (line_through B N) (line_through X Y) :=
sorry

end concurrent_lines_l446_446054


namespace intersection_of_segments_is_union_of_at_most_9901_disjoint_segments_l446_446037

noncomputable def segment := { S : set ℝ // ∃ x y : ℝ, x ≤ y ∧ S = set.Icc x y } 

def A (i : ℕ) (h : 1 ≤ i ∧ i ≤ 100) : set (set ℝ) :=
  { S | ∃ j : ℕ, (1 ≤ j ∧ j ≤ 100) ∧ S ∈ segment }

theorem intersection_of_segments_is_union_of_at_most_9901_disjoint_segments 
  (A : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → set (set ℝ)) 
  (hA : ∀ i (h : 1 ≤ i ∧ i ≤ 100), ∀ S ∈ A i h, ∃ j : ℕ, 1 ≤ j ∧ j ≤ 100 ∧ S ∈ segment 
  ) :
  ∃ I : set (set ℝ), I.countable ∧ (∀ S ∈ I, ∃ x y : ℝ, x ≤ y ∧ S = set.Icc x y) ∧ I.size ≤ 9901 ∧
  (⋂ i (h : 1 ≤ i ∧ i ≤ 100), ⋃ S ∈ A i h, S) = ⋃ S ∈ I, S :=
sorry

end intersection_of_segments_is_union_of_at_most_9901_disjoint_segments_l446_446037


namespace power_of_point_l446_446449

open EuclideanGeometry

namespace CircleProducts

variables {S : Circle} {P A B : Point}
-- Assuming P is outside the circle S
axiom point_outside_circle (hP : ¬(P ∈ S))

-- Define the condition for the line passing through point P and intersecting circle at points A and B
def secant_line_through_point (L : Line) (P : Point) :=
  P ∈ L ∧ ∃ A B : Point, A ∈ S ∧ B ∈ S ∧ A ≠ B ∧ A ∈ L ∧ B ∈ L

theorem power_of_point (L : Line) (hL : secant_line_through_point L P) :
    (dist P A) * (dist P B) = (dist P (L.intersec S).fst) * (dist P (L.intersec S).snd) := 
by
  sorry

end CircleProducts

end power_of_point_l446_446449


namespace gain_percentage_is_30_l446_446942

-- Definitions based on the conditions
def selling_price : ℕ := 195
def gain : ℕ := 45
def cost_price : ℕ := selling_price - gain
def gain_percentage : ℕ := (gain * 100) / cost_price

-- The statement to prove the gain percentage
theorem gain_percentage_is_30 : gain_percentage = 30 := 
by 
  -- Allow usage of fictive sorry for incomplete proof
  sorry

end gain_percentage_is_30_l446_446942


namespace average_inversions_correct_l446_446233

def average_inversions (n : ℕ) : ℕ :=
  n * (n - 1) / 4

theorem average_inversions_correct (n : ℕ) : (expected_inversions n) = average_inversions n := 
by 
  sorry

end average_inversions_correct_l446_446233


namespace shopkeeper_gain_percent_l446_446819

def gain_percent_during_sale : ℝ :=
  let marked_price := 30
  let gain_percent := 0.15
  let discount_percent := 0.10
  let sales_tax_percent := 0.05
  let additional_discount_percent := 0.07
  
  -- Step 1: Calculate Cost Price (CP)
  let cp := marked_price / (1 + gain_percent)
  
  -- Step 2: Discounted Price after Initial 10% Discount
  let discounted_price := marked_price * (1 - discount_percent)
  
  -- Step 3: Price after 5% Sales Tax
  let price_after_tax := discounted_price * (1 + sales_tax_percent)
  
  -- Step 4: Price after Additional 7% Clearance Discount
  let final_price := price_after_tax * (1 - additional_discount_percent)
  
  -- Step 5: Calculate Gain Percent
  let gain := final_price - cp
  (gain / cp) * 100

theorem shopkeeper_gain_percent : gain_percent_during_sale = 1.07 := sorry

end shopkeeper_gain_percent_l446_446819


namespace star_larger_emilio_l446_446081

-- Define the replacement function for Emilio's modification
def replace_3_with_2 (n : ℕ) : ℕ :=
  let s := n.toString in
  let s' := s.map (fun c => if c = '3' then '2' else c) in
  s'.toNat

-- Sum of numbers from 1 to 30 as listed by Star
def star_sum : ℕ :=
  ∑ i in (Finset.range 31).filter (fun x => x ≠ 0), id i

-- Sum of numbers from 1 to 30 as modified by Emilio
def emilio_sum : ℕ :=
  ∑ i in (Finset.range 31).filter (fun x => x ≠ 0), replace_3_with_2 i

-- Prove that Star's sum is 13 larger than Emilio's sum
theorem star_larger_emilio : star_sum - emilio_sum = 13 := by
  sorry

end star_larger_emilio_l446_446081


namespace product_of_divisors_eq_1024_l446_446518

theorem product_of_divisors_eq_1024 (n : ℕ) (h : n > 0) (hp : ∏ d in (finset.filter (λ x, x ∣ n) (finset.range (n+1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446518


namespace cube_unpainted_unit_cubes_l446_446215

theorem cube_unpainted_unit_cubes : 
  let cube_side := 6
  let unit_cubes := 216
  let faces := 6
  let strips_per_face := 2
  let strip_width_small := 2
  let strip_length_large := 6
  let overlap_factor := 4
  let corners := 8
  let edges := 12
  let edge_per_face_share := 4
  in 
    unit_cubes - (faces * (strips_per_face * strip_length_large + strips_per_face * strip_length_large - overlap_factor * overlap_factor)
             - (edges * edge_per_face_share - corners)) = 160 :=
by
  sorry

end cube_unpainted_unit_cubes_l446_446215


namespace dartboard_probability_odd_score_l446_446453

theorem dartboard_probability_odd_score :
  let r1 := 3
  let r2 := 6
  let inner_regions := [1, 2, 2]
  let outer_regions := [2, 1, 1]
  let area_inner_region := λ r : ℝ, (r1 * r1 * π) / 3
  let area_outer_region := λ r : ℝ, ((r2 * r2 * π) - (r1 * r1 * π)) / 3
  let prob_inner_region := λ p : ℝ, (area_inner_region r1 * p) / (area_inner_region r1 * 3 + area_outer_region r2 * 3)
  let prob_outer_region := λ p : ℝ, (area_outer_region r2 * p) / (area_inner_region r1 * 3 + area_outer_region r2 * 3)
  let prob_odd_then_even := (prob_inner_region 1 + prob_outer_region 1) * (prob_inner_region 2 + prob_outer_region 2)
  let prob_even_then_odd := (prob_inner_region 2 + prob_outer_region 2) * (prob_inner_region 1 + prob_outer_region 1)
  let total_prob_odd := prob_odd_then_even + prob_even_then_odd
  total_prob_odd = 35 / 72 := by
  sorry

end dartboard_probability_odd_score_l446_446453


namespace jerusha_earnings_l446_446759

theorem jerusha_earnings (L : ℕ) (h1 : 5 * L = 85) : 4 * L = 68 := 
by
  sorry

end jerusha_earnings_l446_446759


namespace average_height_of_trees_is_10_l446_446581

theorem average_height_of_trees_is_10 :
  ∃ (h₁ h₃ : ℕ), (h₁ = 5 ∨ h₁ = 45) ∧ (h₃ = 15 ∨ h₃ = 5 / 3) ∧
  let h₂ := 15
  let h₄ := 5
  in (h₁ + h₂ + h₃ + h₄) / 4 = 10 :=
sorry

end average_height_of_trees_is_10_l446_446581


namespace calculation_equals_106_25_l446_446255

noncomputable def calculation : ℝ := 2.5 * 8.5 * (5.2 - 0.2)

theorem calculation_equals_106_25 : calculation = 106.25 := 
by
  sorry

end calculation_equals_106_25_l446_446255


namespace least_possible_M_l446_446040

def k_lt_lt (k n : ℕ) : Prop := k < n ∧ k ∣ n
def f_lt_lt (f : ℕ → ℕ) (k n : ℕ) : Prop := f k < f n ∧ f k ∣ f n

theorem least_possible_M :
  ∃ (f : ℕ → ℕ) (M : ℕ), (∀ n k, n ≤ 2013 → k_lt_lt k n → f_lt_lt f k n) ∧ M = 1024 :=
begin
  let f := λ n : ℕ, nat.factorial n,
  let M := 1024,
  use [f, M],
  sorry -- we only need the statement, not the proof
end

end least_possible_M_l446_446040


namespace total_journey_time_l446_446991

def distance_to_post_office : ℝ := 19.999999999999996
def speed_to_post_office : ℝ := 25
def speed_back : ℝ := 4

theorem total_journey_time : 
  (distance_to_post_office / speed_to_post_office) + (distance_to_post_office / speed_back) = 5.8 :=
by
  sorry

end total_journey_time_l446_446991


namespace angle_ABM_eq_angle_ABN_l446_446785

noncomputable def parabola : Set (ℝ × ℝ) :=
  {p | ∃ x y : ℝ, p = (x, y) ∧ y^2 = 2 * x}

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (-2, 0)

def line_through_A (l : ℝ × ℝ → Prop) : Prop :=
  ∀ (p : ℝ × ℝ), l p → p.2 = p.1 - A.1

def intersect_parabola (p : ℝ × ℝ) : Prop :=
  p ∈ parabola ∧ ∃ l : ℝ × ℝ → Prop, line_through_A l ∧ l p

theorem angle_ABM_eq_angle_ABN (M N : ℝ × ℝ)
  (h₁ : intersect_parabola M)
  (h₂ : intersect_parabola N)
  (h₃ : line_through_A (λ p, p = M))
  (h₄ : line_through_A (λ p, p = N)):
  ∠ (A, B, M) = ∠ (A, B, N) := 
sorry

end angle_ABM_eq_angle_ABN_l446_446785


namespace sarah_min_correct_answers_l446_446476

-- Conditions as definitions
def total_problems : ℕ := 30
def correct_points : ℕ := 7
def incorrect_points : ℕ := 0
def unanswered_points : ℕ := 2
def problems_attempted : ℕ := 27
def problems_unanswered : ℕ := total_problems - problems_attempted
def required_min_score : ℕ := 150

-- Question restated as a Lean theorem
theorem sarah_min_correct_answers : exists n : ℕ,
  (n ≤ problems_attempted) ∧ ((problems_unanswered * unanswered_points) + (n * correct_points) ≥ required_min_score) := 
begin
  sorry
end

end sarah_min_correct_answers_l446_446476


namespace profit_in_terms_of_selling_price_l446_446372

variables (m n C S : ℝ)

-- Given conditions
def cost := C
def selling_price := S
def profit := P = (1 / m) * S - (1 / n) * C

-- Goal: Prove the profit in terms of selling price S
theorem profit_in_terms_of_selling_price (P : ℝ) :
  (1 / m) * S - (1 / n) * C = (m - n) / (m * n) * S :=
sorry

end profit_in_terms_of_selling_price_l446_446372


namespace find_ellipse_equation_find_k_value_QA_dot_QB_constant_l446_446317

-- Definitions for constant values based on given conditions
def e : ℝ := real.sqrt 2 / 2
def a : ℝ := 2
def b : ℝ := real.sqrt 2

-- Main theorem statements
theorem find_ellipse_equation : 
  (∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1 ↔ (x^2) / 4 + (y^2) / 2 = 1) :=
sorry

theorem find_k_value (k : ℝ) (hk : k > 0) :
  (∀ x1 x2 : ℝ, (x1 + x2 = 1 → k = real.sqrt 2 / 2)) :=
sorry

theorem QA_dot_QB_constant (A B Q : ℝ × ℝ):
  (Q = (7/4, 0) → (Q.fst - A.fst) * (Q.snd - A.snd) + (Q.fst - B.fst) * (Q.snd - B.snd) = -15/16) :=
sorry

end find_ellipse_equation_find_k_value_QA_dot_QB_constant_l446_446317


namespace length_of_bridge_is_correct_l446_446240

noncomputable def length_of_bridge (length_of_train : ℕ) (time_in_seconds : ℕ) (speed_in_kmph : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmph * (1000 / 3600)
  time_in_seconds * speed_in_mps - length_of_train

theorem length_of_bridge_is_correct :
  length_of_bridge 150 40 42.3 = 320 := by
  sorry

end length_of_bridge_is_correct_l446_446240


namespace graph_intersections_l446_446580

open Real

-- Definitions for the parametric equations
def x (t : ℝ) : ℝ := 2 * cos t + t / 3
def y (t : ℝ) : ℝ := 3 * sin t

theorem graph_intersections :
  (∀ t1 t2 : ℝ, x t1 = x t2 ∧ y t1 = y t2 → t1 ≠ t2)
  ∧ ∃n, n = 27 :=
by
  sorry

end graph_intersections_l446_446580


namespace x4_y4_value_l446_446670

theorem x4_y4_value (x y : ℝ) (h1 : x^4 + x^2 = 3) (h2 : y^4 - y^2 = 3) : x^4 + y^4 = 7 := by
  sorry

end x4_y4_value_l446_446670


namespace sum_of_digits_1_to_9999_l446_446254

-- Proving the sum of digits of all numbers from 1 to 9999 is 474090
theorem sum_of_digits_1_to_9999 : 
  ∑ k in (Finset.range 10000), (to_digits 10 k).sum = 474090 := by
  sorry

end sum_of_digits_1_to_9999_l446_446254


namespace lines_parallel_to_one_plane_l446_446560

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c : V)
variables (α β γ : ℝ)  -- These are the angles that are not right angles

theorem lines_parallel_to_one_plane 
  (h1 : α ≠ π / 2) 
  (h2 : β ≠ π / 2) 
  (h3 : γ ≠ π / 2) 
  (ha_perp : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  let l1 := (a, b) • c - (a, c) • b,
      l2 := (b, c) • a - (b, a) • c,
      l3 := (c, a) • b - (c, b) • a in
  ∃ plane : Submodule ℝ V, l1 ∈ plane ∧ l2 ∈ plane ∧ l3 ∈ plane :=
by sorry

end lines_parallel_to_one_plane_l446_446560


namespace cos_B_value_l446_446271

-- Define the sides of the triangle
def AB : ℝ := 8
def AC : ℝ := 10
def right_angle_at_A : Prop := true

-- Define the cosine function within the context of the given triangle
noncomputable def cos_B : ℝ := AB / AC

-- The proof statement asserting the condition
theorem cos_B_value : cos_B = 4 / 5 :=
by
  -- Given conditions
  have h1 : AB = 8 := rfl
  have h2 : AC = 10 := rfl
  -- Direct computation
  sorry

end cos_B_value_l446_446271


namespace car_new_speed_l446_446405

theorem car_new_speed (original_speed : ℝ) (supercharge_percent : ℝ) (weight_cut_speed_increase : ℝ) :
  original_speed = 150 → supercharge_percent = 0.30 → weight_cut_speed_increase = 10 → 
  original_speed * (1 + supercharge_percent) + weight_cut_speed_increase = 205 :=
by
  intros h_orig h_supercharge h_weight
  rw [h_orig, h_supercharge]
  sorry

end car_new_speed_l446_446405


namespace dot_product_equals_6_l446_446705

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication and addition
def scaled_added_vector : ℝ × ℝ := (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2)

-- Define the dot product
def dot_product : ℝ := scaled_added_vector.1 * vec_a.1 + scaled_added_vector.2 * vec_a.2

-- Assertion that the dot product is equal to 6
theorem dot_product_equals_6 : dot_product = 6 :=
by
  sorry

end dot_product_equals_6_l446_446705


namespace remaining_hours_needed_l446_446897

noncomputable
def hours_needed_to_finish (x : ℚ) : Prop :=
  (1/5 : ℚ) * (2 + x) + (1/8 : ℚ) * x = 1

theorem remaining_hours_needed :
  ∃ x : ℚ, hours_needed_to_finish x ∧ x = 24/13 :=
by
  use 24/13
  sorry

end remaining_hours_needed_l446_446897


namespace count_four_digit_numbers_with_thousandth_place_2_greater_than_unit_place_no_repeats_l446_446243

-- Definitions corresponding to conditions in a)
def four_digit_number_no_repeats (a b c d : ℕ) : Prop :=
  (1000 ≤ 1000 * a + 100 * b + 10 * c + d) ∧
  (1000 * a + 100 * b + 10 * c + d < 10000) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

def thousandth_place_2_greater_than_unit_place (a d : ℕ) : Prop :=
  a = d + 2

-- The equivalent proof problem statement
theorem count_four_digit_numbers_with_thousandth_place_2_greater_than_unit_place_no_repeats :
  ∃ n : ℕ, n = 448 ∧ 
  (n = (finset.univ.filter (λ x : ℕ, 
     ∃ a b c d : ℕ, 
       1000 * a + 100 * b + 10 * c + d = x ∧ 
       four_digit_number_no_repeats a b c d ∧ 
       thousandth_place_2_greater_than_unit_place a d)).card) := 
sorry

end count_four_digit_numbers_with_thousandth_place_2_greater_than_unit_place_no_repeats_l446_446243


namespace trigonometric_identity_l446_446493

theorem trigonometric_identity : 
  (3 - Real.sin (70 * Real.pi / 180)) / (2 - Real.cos (10 * Real.pi / 180) ^ 2) = 2 :=
by
  have h1 : Real.sin (70 * Real.pi / 180) = Real.cos (20 * Real.pi / 180) := sorry
  have h2 : Real.cos (2 * (10 * Real.pi / 180)) = 2 * Real.cos (10 * Real.pi / 180) ^ 2 - 1 := sorry
  rw [h1, h2]
  sorry

end trigonometric_identity_l446_446493


namespace not_possible_2018_people_in_2019_minutes_l446_446740

-- Definitions based on conditions
def initial_people (t : ℕ) : ℕ := 0
def changed_people (x y : ℕ) : ℕ := 2 * x - y

theorem not_possible_2018_people_in_2019_minutes :
  ¬ ∃ (x y : ℕ), (x + y = 2019) ∧ (2 * x - y = 2018) :=
by
  sorry

end not_possible_2018_people_in_2019_minutes_l446_446740


namespace cone_base_radius_l446_446025

theorem cone_base_radius (r_paper : ℝ) (n_parts : ℕ) (r_cone_base : ℝ) 
  (h_radius_paper : r_paper = 16)
  (h_n_parts : n_parts = 4)
  (h_cone_part : r_cone_base = r_paper / n_parts) : r_cone_base = 4 := by
  sorry

end cone_base_radius_l446_446025


namespace hexahedron_octahedron_ratio_l446_446626

open Real

theorem hexahedron_octahedron_ratio (a : ℝ) (h_a_pos : 0 < a) :
  let r1 := (sqrt 6 * a / 9)
  let r2 := (sqrt 6 * a / 6)
  let ratio := r1 / r2
  ∃ m n : ℕ, gcd m n = 1 ∧ (ratio = (m : ℝ) / (n : ℝ)) ∧ (m * n = 6) :=
by {
  sorry
}

end hexahedron_octahedron_ratio_l446_446626


namespace minimize_quadratic_function_l446_446934

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := 5 * x^2 - 20 * x + 7

-- The theorem statement
theorem minimize_quadratic_function : ∃ x : ℝ, (∀ y : ℝ, quadratic_function y ≥ quadratic_function x) ∧ x = 2 :=
by
  use 2
  split
  · intro y
    dsimp [quadratic_function]
    calc
      5 * y^2 - 20 * y + 7
          = 5 * (y - 2)^2 - 13 : by sorry -- Complete the square step
      ... ≥ -13 : by sorry -- Non-negativity of the square term
  · rfl

end minimize_quadratic_function_l446_446934


namespace quadratic_polynomial_correct_l446_446638

noncomputable def quadratic_polynomial (x : ℝ) : ℂ := 3 * x^2 - 24 * x + 60

theorem quadratic_polynomial_correct :
  ∃ (p : ℂ → ℂ), 
  (∀ x : ℂ, p(x) = 3 * x^2 - 24 * x + 60) ∧ 
  (p (4 + 2 * complex.I) = 0) ∧ 
  (p (4 - 2 * complex.I) = 0) ∧ 
  (∀ (a b c : ℝ), p = (λ x : ℂ, a * x^2 + b * x + c) → a = 3) :=
by
  sorry

end quadratic_polynomial_correct_l446_446638


namespace greatest_difference_of_units_digit_l446_446886

theorem greatest_difference_of_units_digit (x y : ℕ) (h_range_x : 0 ≤ x ∧ x ≤ 9) (h_range_y : 0 ≤ y ∧ y ≤ 9)
  (h_mult_3_x : (13 + x) % 3 = 0) (h_mult_3_y : (13 + y) % 3 = 0) :
  x ≠ y → (abs (x - y) ≤ 6) ∧ (abs (x - y) = 6 ∨ abs (x - y) < 6) :=
by
  sorry

end greatest_difference_of_units_digit_l446_446886


namespace factory_output_l446_446416

variable (a : ℝ)
variable (n : ℕ)
variable (r : ℝ)

-- Initial condition: the output value increases by 10% each year for 5 years
def annual_growth (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Theorem statement
theorem factory_output (a : ℝ) : annual_growth a 1.1 5 = 1.1^5 * a :=
by
  sorry

end factory_output_l446_446416


namespace milk_production_l446_446714

theorem milk_production (y : ℕ) (hcows : y > 0) (hcans : y + 2 > 0) (hdays : y + 3 > 0) :
  let daily_production_per_cow := (y + 2 : ℕ) / (y * (y + 3) : ℕ)
  let total_daily_production := (y + 4 : ℕ) * daily_production_per_cow
  let required_days := (y + 6 : ℕ) / total_daily_production
  required_days = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) :=
by
  sorry

end milk_production_l446_446714


namespace max_non_overlapping_triangles_l446_446789

/-- Given 6050 points in a plane with no three points collinear, 
the maximum number of non-overlapping triangles without common vertices is 2016. -/
theorem max_non_overlapping_triangles (points : Finset ℝ × ℝ) 
  (h₁ : points.card = 6050) 
  (h₂ : ∀ (a b c : ℝ × ℝ), a ≠ b → b ≠ c → c ≠ a → a ∉ ({b, c} : Finset _)) : 
  ∃ k : ℕ, k = 2016 ∧ 
  ∃ T : Finset (Finset (ℝ × ℝ)), 
    (∀ t ∈ T, t.card = 3 ∧ 
    (∀ v ∈ t, ∀ t' ∈ T, t ≠ t' → v ∉ t') ) ∧ 
    T.card = k :=
by
  sorry

end max_non_overlapping_triangles_l446_446789


namespace product_of_divisors_eq_1024_l446_446517

theorem product_of_divisors_eq_1024 (n : ℕ) (h : n > 0) (hp : ∏ d in (finset.filter (λ x, x ∣ n) (finset.range (n+1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446517


namespace number_of_moves_l446_446126

theorem number_of_moves (n m : ℕ) (h₁ : n ≥ 2) (h₂ : ∃ heads, heads = m - 1) :
  ∃ moves, moves = ⌊ 2^m / 3 ⌋ := sorry

end number_of_moves_l446_446126


namespace sum_of_numbers_l446_446943

theorem sum_of_numbers (a b c : ℝ) :
  a^2 + b^2 + c^2 = 138 → ab + bc + ca = 131 → a + b + c = 20 :=
by
  sorry

end sum_of_numbers_l446_446943


namespace sum_mnp_is_405_l446_446614

theorem sum_mnp_is_405 :
  let C1_radius := 4
  let C2_radius := 10
  let C3_radius := C1_radius + C2_radius
  let chord_length := (8 * Real.sqrt 390) / 7
  ∃ m n p : ℕ,
    m * Real.sqrt n / p = chord_length ∧
    m.gcd p = 1 ∧
    (∀ k : ℕ, k^2 ∣ n → k = 1) ∧
    m + n + p = 405 :=
by
  sorry

end sum_mnp_is_405_l446_446614


namespace equal_points_probability_not_always_increasing_l446_446905

theorem equal_points_probability_not_always_increasing 
  (p q : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ q) (h₂ : q ≤ 1) :
  ¬ ∀ p₀ p₁, 0 ≤ p₀ ∧ p₀ ≤ p₁ ∧ p₁ ≤ 1 → 
    let f := λ x : ℝ, (3 * x^2 - 2 * x + 1) / 4 in
    f p₀ ≤ f p₁ := by
    sorry

end equal_points_probability_not_always_increasing_l446_446905


namespace savings_for_mother_l446_446798

theorem savings_for_mother (Liam_oranges Claire_oranges: ℕ) (Liam_price_per_pair Claire_price_per_orange: ℝ) :
  Liam_oranges = 40 ∧ Liam_price_per_pair = 2.50 ∧
  Claire_oranges = 30 ∧ Claire_price_per_orange = 1.20 →
  ((Liam_oranges / 2) * Liam_price_per_pair + Claire_oranges * Claire_price_per_orange) = 86 :=
by
  intros
  sorry

end savings_for_mother_l446_446798


namespace maximum_fraction_l446_446232

theorem maximum_fraction (a b h : ℝ) (d : ℝ) (h_d_def : d = Real.sqrt (a^2 + b^2 + h^2)) :
  (a + b + h) / d ≤ Real.sqrt 3 :=
sorry

end maximum_fraction_l446_446232


namespace baseball_card_decrease_l446_446557

theorem baseball_card_decrease (V₀ : ℝ) (V₁ V₂ : ℝ)
  (h₁: V₁ = V₀ * (1 - 0.20))
  (h₂: V₂ = V₁ * (1 - 0.20)) :
  ((V₀ - V₂) / V₀) * 100 = 36 :=
by
  sorry

end baseball_card_decrease_l446_446557


namespace max_value_trig_expr_l446_446427

noncomputable def max_trig_expr (a b φ: ℝ): ℝ :=
  √(a^2 + b^2)

theorem max_value_trig_expr (a b φ: ℝ) :
  ∃ θ : ℝ, a * cos (θ + φ) + b * sin (θ + φ) = max_trig_expr a b φ := by
  sorry

end max_value_trig_expr_l446_446427


namespace segment_weight_2_to_7_l446_446993

noncomputable def density : ℝ → ℝ := λ x, 10 + x

noncomputable def rod_length : ℝ := 11.25

noncomputable def total_weight : ℝ := 42.75

noncomputable def segment_weight (a b : ℝ) : ℝ :=
  ∫ x in a..b, density x

theorem segment_weight_2_to_7 :
  segment_weight 2 7 = 72.5 := by
  sorry

end segment_weight_2_to_7_l446_446993


namespace average_score_of_the_class_l446_446591

theorem average_score_of_the_class :
  let points := [3, 2, 1, 0]
  let proportions := [0.30, 0.50, 0.10, 0.10]
  (points.zip proportions).sum (λ (p : ℕ × ℝ), p.1 * p.2) = 2 :=
by
  sorry

end average_score_of_the_class_l446_446591


namespace circumcenter_not_lattice_point_of_minimal_triangle_l446_446530

theorem circumcenter_not_lattice_point_of_minimal_triangle 
  (A B C : ℤ × ℤ) 
  (hA : A = (0,0)) 
  (hB : B.fst ≠ 0 ∨ B.snd ≠ 0) 
  (hC : C.fst ≠ 0 ∨ C.snd ≠ 0) 
  (hMin : ∀ D E F : ℤ × ℤ, 
            is_similar (triangle A B C) (triangle D E F) → 
            triangle_area (triangle D E F) < triangle_area (triangle A B C) → 
            false) 
  : ¬is_lattice_point (circumcenter (triangle A B C)) :=
by
  sorry

end circumcenter_not_lattice_point_of_minimal_triangle_l446_446530


namespace not_product_of_consecutive_integers_l446_446460

theorem not_product_of_consecutive_integers (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  ∀ (m : ℕ), 2 * (n ^ k) ^ 3 + 4 * (n ^ k) + 10 ≠ m * (m + 1) := by
sorry

end not_product_of_consecutive_integers_l446_446460


namespace closest_area_shaded_l446_446749

noncomputable def pi := Real.pi

def radius_large := 1
def radius_small := 1 / 2
def area_small_circle := pi * (radius_small ^ 2)

def angle_sector := pi / 3  -- 120 degrees in radians
def area_sector := angle_sector * (radius_large ^ 2)
def area_triangle := (1 / 2) * 1 * (sqrt 3 / 2)  -- Area of the equilateral triangle

def area_overlap := 2 * (area_sector - area_triangle)
def area_shaded := area_overlap - area_small_circle

theorem closest_area_shaded : 
  |area_shaded - 0.44| < min (|area_shaded - 0.42|) (min (|area_shaded - 0.40|) (min (|area_shaded - 0.38|) (|area_shaded - 0.36|))) :=
sorry

end closest_area_shaded_l446_446749


namespace freezing_point_depression_of_NH4Br_solution_l446_446276

section FreezingPointDepression

def freezing_point_depression
  (vanthoff_factor : ℕ)
  (kf : ℝ)
  (moles_solute : ℝ)
  (mass_solvent_kg : ℝ) : ℝ :=
  vanthoff_factor * kf * (moles_solute / mass_solvent_kg)

theorem freezing_point_depression_of_NH4Br_solution :
  freezing_point_depression 2 1.86 5 0.5 = 37.2 := by
  sorry

end FreezingPointDepression

end freezing_point_depression_of_NH4Br_solution_l446_446276


namespace train_pass_platform_time_l446_446961

-- Definitions for the conditions in the problem
def train_length : ℝ := 1400
def tree_crossing_time : ℝ := 100
def platform_length : ℝ := 700

-- Calculated speed of the train
def train_speed : ℝ := train_length / tree_crossing_time

-- Combined length of the train and the platform
def combined_length : ℝ := train_length + platform_length

-- Time required to pass the platform
def time_to_pass_platform : ℝ := combined_length / train_speed

-- The proof statement
theorem train_pass_platform_time :
  time_to_pass_platform = 150 := 
by
  sorry

end train_pass_platform_time_l446_446961


namespace seating_arrangement_count_l446_446890

/-- 
There are 8 seats in a row. If each seat can only be occupied by 1 person, and 4 people are seated, 
prove the number of different ways to arrange the seating such that exactly two empty seats are adjacent is 720. 
-/
theorem seating_arrangement_count 
  (total_seats : ℕ) 
  (occupied_seats : ℕ) 
  (adjacent_empty_seats : ℕ): 
  total_seats = 8 → 
  occupied_seats = 4 → 
  adjacent_empty_seats = 2 → 
  ∃ count : ℕ, count = 720 := 
by 
  intros h1 h2 h3; 
  use 720; 
  sorry

end seating_arrangement_count_l446_446890


namespace number_of_incorrect_statements_l446_446385

noncomputable def fine_horse_distance (n : ℕ) : ℕ :=
  193 + (n - 1) * 13

noncomputable def fine_horse_total_distance (n : ℕ) : ℕ :=
  n * 193 + (n * (n - 1) * 13) / 2

noncomputable def nag_horse_distance (n : ℕ) : ℕ :=
  97 + (n - 1) * (-1/2 : ℝ)

noncomputable def nag_horse_total_distance (n : ℕ) : ℝ :=
  n * 97 + (n * (n - 1) * (-1/2 : ℝ)) / 2

def statements_correctness : ℕ :=
  let statement1 := nag_horse_distance 9 = 93
  let statement2 := fine_horse_total_distance 5 = 1095
  let (min_days, _) := min_days_to_meet 3000
  let statement3 := min_days = 21
  nat.bodd (nat.bodd statement1 + nat.bodd statement2 + nat.bodd statement3)

theorem number_of_incorrect_statements :
  statements_correctness = 1 :=
sorry

end number_of_incorrect_statements_l446_446385


namespace rectangle_area_ratio_l446_446488

def isRectangle (A B C D : Point) : Prop := sorry -- Definition of a rectangle with vertices A, B, C, D
def isRegularDecagon (vertices : List Point) : Prop := sorry -- Definition of a regular decagon given its vertices
def dividesIntoTriangles (O : Point) (vertices : List Point) (n : Nat) : Prop := sorry -- Property that the decagon can be divided into n congruent isosceles triangles by drawing radii from O

theorem rectangle_area_ratio
    (A E F J : Point)
    (vertices : List Point)
    (O : Point)
    (decagon_cond : isRegularDecagon vertices)
    (rectangle_cond : isRectangle A E F J)
    (intersection_cond : O ∈ (lineSegment A E ∩ lineSegment F J))
    (division_cond : dividesIntoTriangles O vertices 10) :
    -- Ratio of the area of rectangle AEFJ to the area of the decagon ABCDEFGHIJ is 2:5
    (areaOfRectangle A E F J) / (areaOfDecagon vertices) = 2 / 5 := sorry

end rectangle_area_ratio_l446_446488


namespace round_trip_in_first_trip_l446_446448

def percentage_rt_trip_first_trip := 0.3 -- 30%
def percentage_2t_trip_second_trip := 0.6 -- 60%
def percentage_ow_trip_third_trip := 0.45 -- 45%

theorem round_trip_in_first_trip (P1 P2 P3: ℝ) (C1 C2 C3: ℝ) 
  (h1 : P1 = 0.3) 
  (h2 : 0 < P1 ∧ P1 < 1) 
  (h3 : P2 = 0.6) 
  (h4 : 0 < P2 ∧ P2 < 1) 
  (h5 : P3 = 0.45) 
  (h6 : 0 < P3 ∧ P3 < 1) 
  (h7 : C1 + C2 + C3 = 1) 
  (h8 : (C1 = (1 - P1) * 0.15)) 
  (h9 : C2 = 0.2 * P2) 
  (h10 : C3 = 0.1 * P3) :
  P1 = 0.3 := by
  sorry

end round_trip_in_first_trip_l446_446448


namespace increase_p_does_not_always_increase_equal_points_l446_446915

-- Define the function representing equal points probability
def equal_points_probability (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

-- The main theorem states that increasing the probability 'p' of a draw 
-- does not necessarily increase the probability of the teams acquiring equal points.
theorem increase_p_does_not_always_increase_equal_points :
  ∃ p₁ p₂ : ℝ, 0 ≤ p₁ ∧ p₁ < p₂ ∧ p₂ ≤ 1 ∧ equal_points_probability p₁ ≥ equal_points_probability p₂ :=
by
  sorry

end increase_p_does_not_always_increase_equal_points_l446_446915


namespace measure_of_angle_C_range_of_t_l446_446063

-- Define the geometric setting of the triangle and the given equation.
variables {A B C : RealAngle}
variables {a b c : RealLen}
hypothesis h1 : (a - c) * (Real.sin A + Real.sin C) = (a - b) * Real.sin B

-- Problem 1: Prove the measure of angle C
theorem measure_of_angle_C (h1) : C = π / 3 :=
sorry

-- Define the trisecting point D and the relationship CD = tAD
variables {D : Point}
variables {AD : RealLen}  {BD : RealLen}
variables {t : Real}
hypothesis h2 : CD = t * AD

-- Problem 2: Prove the range of t
theorem range_of_t (h1 h2) : 1 < t ∧ t <= √3 + 1 :=
sorry

end measure_of_angle_C_range_of_t_l446_446063


namespace area_outside_circle_of_equilateral_triangle_l446_446118

noncomputable def equilateral_triangle_area_outside_circle {a : ℝ} (h : a > 0) : ℝ :=
  let S1 := a^2 * Real.sqrt 3 / 4
  let S2 := Real.pi * (a / 3)^2
  let S3 := (Real.pi * (a / 3)^2 / 6) - (a^2 * Real.sqrt 3 / 36)
  S1 - S2 + 3 * S3

theorem area_outside_circle_of_equilateral_triangle
  (a : ℝ) (h : a > 0) :
  equilateral_triangle_area_outside_circle h = a^2 * (3 * Real.sqrt 3 - Real.pi) / 18 :=
sorry

end area_outside_circle_of_equilateral_triangle_l446_446118


namespace ab_ac_sum_l446_446610

variable {r : ℝ}
variable {t : ℝ}
variable {oa : ℝ}
variable {bc : ℝ}

noncomputable def AB_AC_addition (r : ℝ) (t : ℝ) (oa : ℝ) (bc : ℝ) : Prop :=
  OA = 17 ∧ r = 7 ∧ BC = 9 → AB + AC = 8 * real.sqrt 15

theorem ab_ac_sum : AB_AC_addition 7 17 9 :=
by
  sorry

end ab_ac_sum_l446_446610


namespace brazil_avg_sq_feet_per_person_approx_l446_446116

noncomputable def population_of_brazil : ℕ := 195000000
noncomputable def area_of_brazil_sq_miles : ℕ := 3288000
noncomputable def sq_feet_per_sq_mile : ℕ := 27878400
noncomputable def total_area_sq_feet : ℕ := area_of_brazil_sq_miles * sq_feet_per_sq_mile
noncomputable def avg_sq_feet_per_person : ℕ := total_area_sq_feet / population_of_brazil

theorem brazil_avg_sq_feet_per_person_approx :
  avg_sq_feet_per_person ≈ 470000 :=
sorry

end brazil_avg_sq_feet_per_person_approx_l446_446116


namespace simplify_expression_l446_446548

theorem simplify_expression (x : ℝ) : 
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3 - 4 * x^3 + 6 * x^3 = 
  4 * x^3 - x^2 + 23 * x - 3 :=
by -- proof steps are omitted
  sorry

end simplify_expression_l446_446548


namespace increase_p_does_not_always_increase_equal_points_l446_446919

-- Define the function representing equal points probability
def equal_points_probability (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

-- The main theorem states that increasing the probability 'p' of a draw 
-- does not necessarily increase the probability of the teams acquiring equal points.
theorem increase_p_does_not_always_increase_equal_points :
  ∃ p₁ p₂ : ℝ, 0 ≤ p₁ ∧ p₁ < p₂ ∧ p₂ ≤ 1 ∧ equal_points_probability p₁ ≥ equal_points_probability p₂ :=
by
  sorry

end increase_p_does_not_always_increase_equal_points_l446_446919


namespace loss_completeness_l446_446023

-- Definition of compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) (n : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

-- Definition of simple interest formula
def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

-- Definitions for given conditions
def P : ℝ := 7500
def r : ℝ := 0.04
def t : ℕ := 2
def n : ℕ := 1

-- Calculate the future value using compound interest
def A : ℝ := compound_interest P r t n

-- Calculate the simple interest
def SI : ℝ := simple_interest P r t

-- Calculate the loss
def loss : ℝ := (A - P) - SI

-- The theorem to prove the loss is equal to Rs. 9.60
theorem loss_completeness : loss = 9.60 := by 
  -- Proof goes here
  sorry

end loss_completeness_l446_446023


namespace unique_positive_integer_satisfies_condition_l446_446489

def is_positive_integer (n : ℕ) : Prop := n > 0

def condition (n : ℕ) : Prop := 20 - 5 * n ≥ 15

theorem unique_positive_integer_satisfies_condition :
  ∃! n : ℕ, is_positive_integer n ∧ condition n :=
by
  sorry

end unique_positive_integer_satisfies_condition_l446_446489


namespace dispersion_measured_by_std_dev_and_range_l446_446178

variables {α : Type*} [linear_order α] (x : list α)

def standard_deviation (l : list ℝ) : ℝ := sorry -- definition of standard_deviation
def median (l : list α) : α := sorry -- definition of median
def range (l : list ℝ) : ℝ := sorry -- definition of range
def mean (l : list ℝ) : ℝ := sorry -- definition of mean

theorem dispersion_measured_by_std_dev_and_range :
  (standard_deviation (map (λ x, (x : ℝ)) (x : list α)) > 0 ∨ range (map (λ x, (x : ℝ)) (x : list α)) > 0) →
  (∀ x, x ∈ [standard_deviation (map (λ x, (x : ℝ)) (x : list α)), range (map (λ x, (x : ℝ)) (x : list α))]) :=
begin
  sorry
end

end dispersion_measured_by_std_dev_and_range_l446_446178


namespace negation_exponential_l446_446112

theorem negation_exponential (P : ∃ x0 : ℝ, 2 ^ x0 ≤ 0) : (∀ x : ℝ, 2 ^ x > 0) :=
by
  sorry

end negation_exponential_l446_446112


namespace sqrt_abc_abc_sum_eq_231_l446_446428

-- Defining the conditions
variables (a b c : ℝ) -- a, b, c are real numbers
variable h1 : b + c = 16
variable h2 : c + a = 18
variable h3 : a + b = 20

-- The statement to prove
theorem sqrt_abc_abc_sum_eq_231
(h1 : b + c = 16)
(h2 : c + a = 18)
(h3 : a + b = 20)
: sqrt (a * b * c * (a + b + c)) = 231 :=
sorry -- Proof to be filled in

end sqrt_abc_abc_sum_eq_231_l446_446428


namespace cyclic_quadrilateral_LA_MB_LM_l446_446437

-- Lean 4 Statement
theorem cyclic_quadrilateral_LA_MB_LM (A B C D L M E : Point)
[cyclic_quad : cyclic Quadrilateral ABCD]
[angle_bis_A : angle_bisector A E]
[angle_bis_B : angle_bisector B E]
(parallel_E : parallel (line_through E) (line_through C D))
(line_inter_L : intersection (line_through E parallel_to CD) (line_through A D) L)
(line_inter_M : intersection (line_through E parallel_to CD) (line_through B C) M) :
  LA + MB = LM := by
  sorry

end cyclic_quadrilateral_LA_MB_LM_l446_446437


namespace kelly_games_giveaway_l446_446946

theorem kelly_games_giveaway (n m g : ℕ) (h_current: n = 50) (h_left: m = 35) : g = n - m :=
by
  sorry

end kelly_games_giveaway_l446_446946


namespace card_arrangement_l446_446809

theorem card_arrangement :
  ∃ (a b c d e f g h : ℕ), 
    a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ g ≠ h ∧ 
    {a, b, c, d, e, f, g, h} = {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
    abs (a - b) = 2 ∧ 
    abs (c - d) = 3 ∧ 
    abs (e - f) = 4 ∧ 
    abs (g - h) = 5 :=
by
  sorry

end card_arrangement_l446_446809


namespace volume_removed_percentage_l446_446235

noncomputable def volume_of_box (length width height : ℝ) : ℝ := 
  length * width * height

noncomputable def volume_of_cube (side : ℝ) : ℝ := 
  side ^ 3

noncomputable def volume_removed (length width height side : ℝ) : ℝ :=
  8 * (volume_of_cube side)

noncomputable def percentage_removed (length width height side : ℝ) : ℝ :=
  (volume_removed length width height side) / (volume_of_box length width height) * 100

theorem volume_removed_percentage :
  percentage_removed 20 15 12 4 = 14.22 := 
by
  sorry

end volume_removed_percentage_l446_446235


namespace XiaoMing_strategy_l446_446970

noncomputable def prob_A_correct : ℝ := 0.8
noncomputable def prob_B_correct : ℝ := 0.6

def points_A_correct : ℝ := 20
def points_B_correct : ℝ := 80

def prob_XA_0 : ℝ := 1 - prob_A_correct
def prob_XA_20 : ℝ := prob_A_correct * (1 - prob_B_correct)
def prob_XA_100 : ℝ := prob_A_correct * prob_B_correct

def expected_XA : ℝ := 0 * prob_XA_0 + points_A_correct * prob_XA_20 + (points_A_correct + points_B_correct) * prob_XA_100

def prob_YB_0 : ℝ := 1 - prob_B_correct
def prob_YB_80 : ℝ := prob_B_correct * (1 - prob_A_correct)
def prob_YB_100 : ℝ := prob_B_correct * prob_A_correct

def expected_YB : ℝ := 0 * prob_YB_0 + points_B_correct * prob_YB_80 + (points_A_correct + points_B_correct) * prob_YB_100

def distribution_A_is_correct : Prop :=
  prob_XA_0 = 0.2 ∧ prob_XA_20 = 0.32 ∧ prob_XA_100 = 0.48

def choose_B_first : Prop :=
  expected_YB > expected_XA

theorem XiaoMing_strategy :
  distribution_A_is_correct ∧ choose_B_first :=
by
  sorry

end XiaoMing_strategy_l446_446970


namespace total_savings_l446_446793

def liam_oranges : ℕ := 40
def liam_price_per_two : ℝ := 2.50
def claire_oranges : ℕ := 30
def claire_price_per_one : ℝ := 1.20

theorem total_savings : (liam_oranges / 2 * liam_price_per_two) + (claire_oranges * claire_price_per_one) = 86 := by
  sorry

end total_savings_l446_446793


namespace Sara_money_left_l446_446467

/--
Sara worked for 40 hours a week at a rate of $11.50 per hour for two weeks.
She bought a set of tires for $410.
Prove that she was left with $510 after buying the tires.
-/
theorem Sara_money_left (hours_per_week : ℕ) (hourly_wage : ℝ) (weeks_worked : ℕ) (cost_of_tires : ℝ) : 
  hours_per_week = 40 → hourly_wage = 11.50 → weeks_worked = 2 → cost_of_tires = 410 → 
  let total_earnings := hours_per_week * hourly_wage * weeks_worked in
  let money_left := total_earnings - cost_of_tires in
  money_left = 510 :=
by {
  intros,
  subst hours_per_week,
  subst hourly_wage,
  subst weeks_worked,
  subst cost_of_tires,
  dsimp [total_earnings, money_left],
  norm_num,
  sorry
}

end Sara_money_left_l446_446467


namespace sum_mnp_is_405_l446_446613

theorem sum_mnp_is_405 :
  let C1_radius := 4
  let C2_radius := 10
  let C3_radius := C1_radius + C2_radius
  let chord_length := (8 * Real.sqrt 390) / 7
  ∃ m n p : ℕ,
    m * Real.sqrt n / p = chord_length ∧
    m.gcd p = 1 ∧
    (∀ k : ℕ, k^2 ∣ n → k = 1) ∧
    m + n + p = 405 :=
by
  sorry

end sum_mnp_is_405_l446_446613


namespace area_of_inscribed_square_l446_446978

noncomputable def circle_eq (x y : ℝ) : Prop := 
  3*x^2 + 3*y^2 - 15*x + 9*y + 27 = 0

theorem area_of_inscribed_square :
  (∃ x y : ℝ, circle_eq x y) →
  ∃ s : ℝ, s^2 = 25 :=
by
  sorry

end area_of_inscribed_square_l446_446978


namespace log_simplification_l446_446935

theorem log_simplification :
  let a := 16
  let b := 4
  let c := (1 / 4 : ℝ)
  log b a / log a c = -4 :=
by
  have h1 : a = b^2 := sorry
  have h2 : c = a ^ (-1 / 2) := sorry
  calc
    log b a / log a c = log b (b^2) / log a (a ^ (-1 / 2)) : by rw [h1, h2]
                   ... = 2 / (-1 / 2) : by rw [log_base_change, log_base_change]
                   ... = -4 : by norm_num

end log_simplification_l446_446935


namespace f_zero_one_range_of_a_l446_446439

section
variable {f : ℝ → ℝ}
variable {a : ℝ}

/-- Problem (1) -/
theorem f_zero_one (h1 : ∀ m n : ℝ, f (m + n) = f m * f n)
    (h2 : ∀ x : ℝ, 0 < x → 0 < f x ∧ f x < 1) : f 0 = 1 ∧ (∀ x : ℝ, x < 0 → 1 < f x) :=
sorry

/-- Problem (2) -/
theorem range_of_a (h1 : ∀ m n : ℝ, f (m + n) = f m * f n)
    (h2 : ∀ x : ℝ, 0 < x → 0 < f x ∧ f x < 1)
    (h3 : ∀ x y : ℝ, f (x^2 + y^2) > f 1)
    (h4 : ∀ x y : ℝ, f (a * x - y + 2) = 1) : -real.sqrt 3 ≤ a ∧ a ≤ real.sqrt 3 :=
sorry

end

end f_zero_one_range_of_a_l446_446439


namespace relationship_among_abc_l446_446308

noncomputable def a := 3 ^ 0.1
noncomputable def b := Real.logBase π 2
noncomputable def c := Real.logBase 2 (Real.sin (2 * Real.pi / 3))

theorem relationship_among_abc : a > b ∧ b > c :=
by
  -- Definitions as provided in the conditions.
  sorry

end relationship_among_abc_l446_446308


namespace percentage_volume_removed_l446_446237

/-- A solid box has dimensions 20 cm by 15 cm by 12 cm. 
A new solid is formed by removing a cube of 4 cm on a side from each of the eight corners. 
We need to prove that the percentage of the original volume removed is approximately 14.22%. -/
theorem percentage_volume_removed :
  let volume_original_box := 20 * 15 * 12
  let volume_one_cube := 4^3
  let total_volume_removed := 8 * volume_one_cube
  let percentage_removed := (total_volume_removed : ℚ) / volume_original_box * 100
  percentage_removed ≈ 14.22 := sorry

end percentage_volume_removed_l446_446237


namespace sam_shooting_stars_l446_446252

def shooting_stars_proof : Prop :=
  let bridget_count := 14
  let reginald_count := bridget_count - 2
  ∃ (sam_count : ℕ), 
    (sam_count = (bridget_count + reginald_count + sam_count) / 3 + 2) ∧
    (sam_count - reginald_count = 4)

theorem sam_shooting_stars :
  shooting_stars_proof :=
by
  let bridget_count := 14
  let reginald_count := bridget_count - 2
  use 16
  split
  sorry
  sorry

end sam_shooting_stars_l446_446252


namespace triangle_perimeter_l446_446114

theorem triangle_perimeter (r A : ℝ) (h_r : r = 2.5) (h_A : A = 50) : 
  ∃ p : ℝ, p = 40 :=
by
  sorry

end triangle_perimeter_l446_446114


namespace fibonacci_square_l446_446832

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem fibonacci_square (n : ℕ) : fibonacci n = n^2 ↔ n = 12 := by
  sorry

end fibonacci_square_l446_446832


namespace total_savings_l446_446794

def liam_oranges : ℕ := 40
def liam_price_per_two : ℝ := 2.50
def claire_oranges : ℕ := 30
def claire_price_per_one : ℝ := 1.20

theorem total_savings : (liam_oranges / 2 * liam_price_per_two) + (claire_oranges * claire_price_per_one) = 86 := by
  sorry

end total_savings_l446_446794


namespace arithmetic_mean_of_remaining_set_is_60_l446_446097

theorem arithmetic_mean_of_remaining_set_is_60
  (original_mean : ℝ)
  (num_elements : ℕ)
  (remaining_elements : ℕ)
  (removed_num1 removed_num2 : ℝ)
  (h1 : original_mean = 60)
  (h2 : num_elements = 75)
  (h3 : remaining_elements = 73)
  (h4 : removed_num1 = 72)
  (h5 : removed_num2 = 48) :
  let original_sum := original_mean * num_elements
  let new_sum := original_sum - removed_num1 - removed_num2
  let new_mean := new_sum / remaining_elements in
  new_mean = 60 := by
    sorry

end arithmetic_mean_of_remaining_set_is_60_l446_446097


namespace product_remainder_mod_7_l446_446859

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l446_446859


namespace value_of_k_l446_446671

theorem value_of_k
  (a b c : ℝ) 
  (h1 : 2^a = 3^b) 
  (h2 : 3^b = 6^c) 
  (h3 : (a + b) / c ∈ set.Ioo k (k + 1)) :
  k = 4 :=
sorry

end value_of_k_l446_446671


namespace number_of_liars_l446_446566

/-- There are 25 people in line, each of whom either tells the truth or lies.
The person at the front of the line says: "Everyone behind me is lying."
Everyone else says: "The person directly in front of me is lying."
Prove that the number of liars among these 25 people is 13. -/
theorem number_of_liars : 
  ∀ (persons : Fin 25 → Prop), 
    (persons 0 → ∀ n > 0, ¬persons n) →
    (∀ n : Nat, (1 ≤ n → n < 25 → persons n ↔ ¬persons (n - 1))) →
    (∃ l, l = 13 ∧ ∀ n : Nat, (0 ≤ n → n < 25 → persons n ↔ (n % 2 = 0))) :=
by
  sorry

end number_of_liars_l446_446566


namespace circle_through_A_B_and_tangent_to_m_l446_446686

noncomputable def circle_equation (x y : ℚ) : Prop :=
  x^2 + (y - 1/3)^2 = 16/9

theorem circle_through_A_B_and_tangent_to_m :
  ∃ (c : ℚ × ℚ) (r : ℚ),
    (c = (0, 1/3)) ∧
    (r = 4/3) ∧
    (∀ (x y : ℚ),
      (x = 0 ∧ y = -1 ∨ x = 4/3 ∧ y = 1/3 → (x^2 + (y - 1/3)^2 = 16/9)) ∧
      (x = 4/3 → x = r)) :=
by
  sorry

end circle_through_A_B_and_tangent_to_m_l446_446686


namespace min_n_for_constant_term_l446_446369

theorem min_n_for_constant_term (n : ℕ) :
  (∃ n : ℕ, ∀ r : ℕ, 
    6 * n - (15 / 2) * r = 0 ∧ (n > 0) → n = 5) :=
by
  -- Let n be a positive integer such that the term with zero exponent exists
  use 5
  intros r hr
  cases hr with hzero hpos
  have h := eq_zero_of_mul_eq_zero (6 * n - 15 / 2 * r) hzero
  -- Assuming r = 4
  sorry

end min_n_for_constant_term_l446_446369


namespace child_ticket_cost_l446_446241

theorem child_ticket_cost 
    (total_people : ℕ) 
    (total_money_collected : ℤ) 
    (adult_ticket_price : ℤ) 
    (children_attended : ℕ) 
    (adults_count : ℕ) 
    (total_adult_cost : ℤ) 
    (total_child_cost : ℤ) 
    (c : ℤ)
    (total_people_eq : total_people = 22)
    (total_money_collected_eq : total_money_collected = 50)
    (adult_ticket_price_eq : adult_ticket_price = 8)
    (children_attended_eq : children_attended = 18)
    (adults_count_eq : adults_count = total_people - children_attended)
    (total_adult_cost_eq : total_adult_cost = adults_count * adult_ticket_price)
    (total_child_cost_eq : total_child_cost = children_attended * c)
    (money_collected_eq : total_money_collected = total_adult_cost + total_child_cost) 
  : c = 1 := 
  by
    sorry

end child_ticket_cost_l446_446241


namespace correct_equation_l446_446624

theorem correct_equation (x : ℝ) : 3 * x + 20 = 4 * x - 25 :=
by sorry

end correct_equation_l446_446624


namespace volume_removed_percentage_l446_446236

noncomputable def volume_of_box (length width height : ℝ) : ℝ := 
  length * width * height

noncomputable def volume_of_cube (side : ℝ) : ℝ := 
  side ^ 3

noncomputable def volume_removed (length width height side : ℝ) : ℝ :=
  8 * (volume_of_cube side)

noncomputable def percentage_removed (length width height side : ℝ) : ℝ :=
  (volume_removed length width height side) / (volume_of_box length width height) * 100

theorem volume_removed_percentage :
  percentage_removed 20 15 12 4 = 14.22 := 
by
  sorry

end volume_removed_percentage_l446_446236


namespace problem_statement_l446_446433

noncomputable def h (y : ℂ) : ℂ := y^5 - y^3 + 1
noncomputable def p (y : ℂ) : ℂ := y^2 - 3

theorem problem_statement (y_1 y_2 y_3 y_4 y_5 : ℂ) (hroots : ∀ y, h y = 0 ↔ y = y_1 ∨ y = y_2 ∨ y = y_3 ∨ y = y_4 ∨ y = y_5) :
  (p y_1) * (p y_2) * (p y_3) * (p y_4) * (p y_5) = 22 :=
by
  sorry

end problem_statement_l446_446433


namespace find_a_c_l446_446855

theorem find_a_c (a c : ℝ) (h_discriminant : ∀ x : ℝ, a * x^2 + 10 * x + c = 0 → ∃ k : ℝ, a * k^2 + 10 * k + c = 0 ∧ (a * x^2 + 10 * k + c = 0 → x = k))
  (h_sum : a + c = 12) (h_lt : a < c) : (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end find_a_c_l446_446855


namespace parabola_focus_directrix_l446_446723

-- Definitions and conditions
def parabola (y a x : ℝ) : Prop := y^2 = a * x
def distance_from_focus_to_directrix (d : ℝ) : Prop := d = 2

-- Statement of the problem
theorem parabola_focus_directrix {a : ℝ} (h : parabola y a x) (h2 : distance_from_focus_to_directrix d) : 
  a = 4 ∨ a = -4 :=
sorry

end parabola_focus_directrix_l446_446723


namespace multiples_of_231_l446_446349

theorem multiples_of_231 (h : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ 99 → i % 2 = 1 → 231 ∣ 10^j - 10^i) :
  ∃ n, n = 416 :=
by sorry

end multiples_of_231_l446_446349


namespace find_coprime_ratio_l446_446960

noncomputable def tuple (x : Fin 39 → ℝ) : Prop :=
  (2 * ∑ i in Finset.univ, Real.sin (x i) = -34) ∧
  (∑ i in Finset.univ, Real.cos (x i) = -34)

theorem find_coprime_ratio (x : Fin 39 → ℝ) (hx : tuple x) :
  let max_cos : ℝ := Finset.univ.sup (λ i, Real.cos (x i)),
      max_sin : ℝ := Finset.univ.sup (λ i, Real.sin (x i)),
      a : ℕ := 12,
      b : ℕ := 25
  in (Nat.coprime a b) ∧ (a + b = 37) :=
sorry

end find_coprime_ratio_l446_446960


namespace inequality_problem_l446_446310

theorem inequality_problem (x y a b : ℝ) (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < b) (h5 : b < 1) : (a ^ x < b ^ y) :=
by 
  sorry

end inequality_problem_l446_446310


namespace probability_function_meaningful_l446_446463

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

def is_meaningful (x : ℝ) : Prop := 1 - x^2 > 0

def measure_interval (a b : ℝ) : ℝ := b - a

theorem probability_function_meaningful:
  let interval_a := -2
  let interval_b := 1
  let meaningful_a := -1
  let meaningful_b := 1
  let total_interval := measure_interval interval_a interval_b
  let meaningful_interval := measure_interval meaningful_a meaningful_b
  let P := meaningful_interval / total_interval
  (P = (2/3)) :=
by
  sorry

end probability_function_meaningful_l446_446463


namespace remainder_of_product_mod_7_l446_446880

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l446_446880


namespace count_values_n_l446_446050

-- Define a function to calculate sum of divisors
def sum_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum id

-- Define a function to check if a number is prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- The main theorem statement
theorem count_values_n (count : ℕ) : count = 5 :=
  let primes_in_range := 
    (Finset.range (30 + 1)).filter (λ n, is_prime (sum_divisors n)) in
  primes_in_range.card = count := sorry

end count_values_n_l446_446050


namespace optimal_strategy_l446_446974

noncomputable def prob_A_correct : ℝ := 0.8
noncomputable def prob_B_correct : ℝ := 0.6
noncomputable def score_A_correct : ℝ := 20
noncomputable def score_B_correct : ℝ := 80

def X_distribution : (ℝ → ℝ) :=
λ x, if x = 0 then 1 - prob_A_correct
     else if x = 20 then prob_A_correct * (1 - prob_B_correct)
     else if x = 100 then prob_A_correct * prob_B_correct
     else 0

noncomputable def E_X : ℝ :=
(0 * (1 - prob_A_correct)) + (20 * (prob_A_correct * (1 - prob_B_correct))) + (100 * (prob_A_correct * prob_B_correct))

noncomputable def E_Y : ℝ :=
(0 * (1 - prob_B_correct)) + (80 * (prob_B_correct * (1 - prob_A_correct))) + (100 * (prob_B_correct * prob_A_correct))

theorem optimal_strategy : E_X = 54.4 ∧ E_Y = 57.6 → (57.6 > 54.4) :=
by {
  sorry 
}

end optimal_strategy_l446_446974


namespace find_n_l446_446513

-- Define that n is a positive integer
def positive_integer (n : ℕ) : Prop := n > 0

-- Define number of divisors
def num_divisors (n : ℕ) : ℕ := (finset.range (n+1)).filter (λ d, n % d = 0).card

-- Define the product of divisors function
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
(finset.range (n+1)).filter (λ d, n % d = 0).prod id

-- The final theorem statement to be proven
theorem find_n (n : ℕ) (hn : positive_integer n) :
  prod_of_divisors n = 1024 → n = 16 :=
by { sorry }

end find_n_l446_446513


namespace smallest_nominal_value_l446_446639

theorem smallest_nominal_value :
  let a₁ := 1993 ^ (1994 ^ 1995)
  ∃ n, exists_smallest_nominal (fun (a : ℕ) => 
    (n : ℕ) -> a n = a₁ ∧
    (∀ n, a (n+1) = if a n % 2 = 0 then a n / 2 else a n + 7)) 
  1 := sorry

-- exists_smallest_nominal is a helper function to find the smallest 
-- nominal value in the sequence.
def exists_smallest_nominal (seq : ℕ → ℕ) (nominal : ℕ) : Prop :=
  ∃ n, seq n = nominal ∧ ∀ m, seq m ≤ seq n → seq m = nominal := sorry

end smallest_nominal_value_l446_446639


namespace compute_div_mul_l446_446262

theorem compute_div_mul (x y z : Int) (h : y ≠ 0) (hx : x = -100) (hy : y = -25) (hz : z = -6) :
  (((-x) / (-y)) * -z) = -24 := by
  sorry

end compute_div_mul_l446_446262


namespace eggs_per_group_l446_446073

-- Conditions
def total_eggs : ℕ := 9
def total_groups : ℕ := 3

-- Theorem statement
theorem eggs_per_group : total_eggs / total_groups = 3 :=
sorry

end eggs_per_group_l446_446073


namespace line_through_points_l446_446845

theorem line_through_points (x1 y1 x2 y2 : ℕ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (3, 8)) : 
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = 2 := 
by
  sorry

end line_through_points_l446_446845


namespace wire_total_length_l446_446988

theorem wire_total_length (a b c total_length : ℕ) (h1 : a = 7) (h2 : b = 3) (h3 : c = 2) (h4 : c * 16 = 32) :
  total_length = (a + b + c) * (16 / c) :=
by
  have h5 : c = 2 := by rw [←nat.add_assoc, add_comm, h3]
  have h6 : total_length = (a + b + c) * 8 := sorry
  exact h6

end wire_total_length_l446_446988


namespace eval_expr_l446_446630

theorem eval_expr : (2/5) + (3/8) - (1/10) = 27/40 :=
by
  sorry

end eval_expr_l446_446630


namespace find_x_triangle_area_l446_446302

theorem find_x_triangle_area (x : ℝ) (h1 : x > 0) (h2 : (1/2) * x * (3 * x) = 72) : x = 4 * real.sqrt 3 :=
begin
  sorry
end

end find_x_triangle_area_l446_446302


namespace segment_length_polar_coordinates_l446_446015

theorem segment_length_polar_coordinates :
  ∀ (ρ θ: ℝ), ρ = 1 ∧ ρ * sin θ - ρ * cos θ = 1 → length_segment_polar ρ θ = sqrt 2 := 
by
  intros ρ θ h
  sorry

end segment_length_polar_coordinates_l446_446015


namespace tamia_total_slices_and_pieces_l446_446087

-- Define the conditions
def num_bell_peppers : ℕ := 5
def slices_per_pepper : ℕ := 20
def num_large_slices : ℕ := num_bell_peppers * slices_per_pepper
def num_half_slices : ℕ := num_large_slices / 2
def small_pieces_per_slice : ℕ := 3
def num_small_pieces : ℕ := num_half_slices * small_pieces_per_slice
def num_uncut_slices : ℕ := num_half_slices

-- Define the total number of pieces and slices
def total_pieces_and_slices : ℕ := num_uncut_slices + num_small_pieces

-- State the theorem and provide a placeholder for the proof
theorem tamia_total_slices_and_pieces : total_pieces_and_slices = 200 :=
by
  sorry

end tamia_total_slices_and_pieces_l446_446087


namespace bunyakovsky_more_likely_same_sector_l446_446609

theorem bunyakovsky_more_likely_same_sector (n : ℕ) (p : Fin n → ℝ) (hne : ∃ i j : Fin n, i ≠ j ∧ p i ≠ p j)
  (hprob : ∑ i, p i = 1) : 
  ∑ i, (p i)^2 ≥ ∑ i, p i * p ((i + 1) % n) :=
by
  sorry

end bunyakovsky_more_likely_same_sector_l446_446609


namespace circle_area_increase_l446_446834

theorem circle_area_increase (r : ℝ) :
  let R := 2 * r,
      A1 := Real.pi * r^2,
      A2 := Real.pi * R^2 in
  ((A2 - A1) / A1) * 100 = 300 :=
by
  sorry

end circle_area_increase_l446_446834


namespace cistern_total_wet_surface_area_l446_446979

/-- Given a cistern with length 6 meters, width 4 meters, and water depth 1.25 meters,
    the total area of the wet surface is 49 square meters. -/
theorem cistern_total_wet_surface_area
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 6) (h_width : width = 4) (h_depth : depth = 1.25) :
  (length * width) + 2 * (length * depth) + 2 * (width * depth) = 49 :=
by {
  -- Proof goes here
  sorry
}

end cistern_total_wet_surface_area_l446_446979


namespace tangent_line_equation_l446_446491

theorem tangent_line_equation :
  let y := λ x : ℝ, -x^3 + 3 * x^2 in
  let dydx := λ x : ℝ, -3 * x^2 + 6 * x in
  let point := (1 : ℝ, 2 : ℝ) in
  let slope_at_point := dydx 1 in
  slope_at_point = 3 →
  ∀ x : ℝ, (y x - 2 = 3 * (x - 1)) → y = λ x, 3 * x - 1 :=
by {
  intros y dydx point slope_at_point h_slope_proof x h_tangent_line,
  sorry
}

end tangent_line_equation_l446_446491


namespace annual_haircut_cost_l446_446030

-- Define conditions as Lean definitions
def hair_growth_rate_per_month : ℝ := 1.5
def initial_hair_length : ℝ := 9
def post_haircut_length : ℝ := 6
def haircut_cost : ℝ := 45
def tip_percentage : ℝ := 0.2

-- The question to be answered as a theorem
theorem annual_haircut_cost :
  (hair_growth_rate_per_month > 0) →
  (initial_hair_length > post_haircut_length) →
  let length_cut := initial_hair_length - post_haircut_length in
  let months_between_haircuts := length_cut / hair_growth_rate_per_month in
  let haircuts_per_year := 12 / months_between_haircuts in
  let tip_amount := haircut_cost * tip_percentage in
  let cost_per_haircut := haircut_cost + tip_amount in
  haircuts_per_year * cost_per_haircut = 324 := 
by 
  -- Skipping the proof
  sorry

end annual_haircut_cost_l446_446030


namespace prime_condition_find_solution_l446_446565

theorem prime_condition (p : ℕ) (h_prime : p ≠ 1 ∧ (∀ d : ℕ, d ∣ p → d = 1 ∨ d = p))
  (h1 : ∀ n : ℕ, 4 * p^2 + 1 = n → (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n))
  (h2 : ∀ n : ℕ, 6 * p^2 + 1 = n → (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)) :
  p = 5 :=
sorry

theorem find_solution (x y z u : ℝ)
  (h_eq1 : x*y*z + x*y + y*z + z*x + x + y + z = 7)
  (h_eq2 : y*z*u + y*z + z*u + u*y + y + z + u = 10)
  (h_eq3 : z*u*x + z*u + u*x + x*z + z + u + x = 10)
  (h_eq4 : u*x*y + u*x + x*y + y*u + u + x + y = 10) :
  x = 1 ∧ y = 1 ∧ z = 1 ∧ u = 7/4 :=
sorry

end prime_condition_find_solution_l446_446565


namespace eval_sum_T_l446_446299

noncomputable def T (n : ℝ) : ℝ :=
  (cos (30 * (real.pi / 180) - n))^2 - 
  (cos (30 * (real.pi / 180) - n)) * (cos (30 * (real.pi / 180) + n)) + 
  (cos (30 * (real.pi / 180) + n))^2

theorem eval_sum_T : 
  (4 * ∑ n in finset.range 30, (n + 1) * T ((n + 1) * (real.pi / 180))) = 1395 :=
sorry

end eval_sum_T_l446_446299


namespace select_students_l446_446485

-- Definitions for the conditions
variables (A B C D E : Prop)

-- Conditions
def condition1 : Prop := A → B ∧ ¬E
def condition2 : Prop := (B ∨ E) → ¬D
def condition3 : Prop := C ∨ D

-- The main theorem
theorem select_students (hA : A) (h1 : condition1 A B E) (h2 : condition2 B E D) (h3 : condition3 C D) : B ∧ C :=
by 
  sorry

end select_students_l446_446485


namespace dispersion_statistics_l446_446182

-- Define the variables and possible statistics options
def sample (n : ℕ) := fin n → ℝ

inductive Statistics
| StandardDeviation : Statistics
| Median : Statistics
| Range : Statistics
| Mean : Statistics

-- Define a function that returns if a statistic measures dispersion
def measures_dispersion : Statistics → Prop
| Statistics.StandardDeviation := true
| Statistics.Median := false
| Statistics.Range := true
| Statistics.Mean := false

-- Prove that StandardDeviation and Range measure dispersion
theorem dispersion_statistics (x : sample n) :
  measures_dispersion Statistics.StandardDeviation ∧
  measures_dispersion Statistics.Range :=
by
  split;
  exact trivial

end dispersion_statistics_l446_446182


namespace increase_p_does_not_always_increase_equal_points_l446_446917

-- Define the function representing equal points probability
def equal_points_probability (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

-- The main theorem states that increasing the probability 'p' of a draw 
-- does not necessarily increase the probability of the teams acquiring equal points.
theorem increase_p_does_not_always_increase_equal_points :
  ∃ p₁ p₂ : ℝ, 0 ≤ p₁ ∧ p₁ < p₂ ∧ p₂ ≤ 1 ∧ equal_points_probability p₁ ≥ equal_points_probability p₂ :=
by
  sorry

end increase_p_does_not_always_increase_equal_points_l446_446917


namespace max_rooks_on_board_l446_446381

theorem max_rooks_on_board : 
  ∃ (k : ℕ), 
  k = 16 ∧
  ∀ (board : matrix (fin 10) (fin 10) bool)
    (cells_marked : fin 10 × fin 10 → bool) 
    (r : fin 10 × fin 10 → Prop)
    (c : fin 10 × fin 10 → fin 10 → Prop),
    (∀ (i j : fin 10), k i j → r (i, j)) →
    (∀ (i j : fin 10), cells_marked (i, j) ↔ (∃ (rook : fin 10 × fin 10), r rook ∧ (rook.1 = i ∨ rook.2 = j))) →
    ∀ rook_removed : fin 10 × fin 10, r rook_removed →
    ∃ (i j : fin 10), cells_marked (i, j) ∧ ¬ cells_marked (i, j) :=
begin
  sorry,
end

end max_rooks_on_board_l446_446381


namespace equal_points_probability_l446_446901

theorem equal_points_probability (p : ℝ) (prob_draw_increases : 0 ≤ p ∧ p ≤ 1) :
  (∀ q : ℝ, (0 ≤ q ∧ q < p) → (q^2 + (1 - q)^2 / 2 < p^2 + (1 - p)^2 / 2)) → False :=
begin
  sorry
end

end equal_points_probability_l446_446901


namespace value_of_a_l446_446041

def A := { x : ℝ | x^2 - 8*x + 15 = 0 }
def B (a : ℝ) := { x : ℝ | x * a - 1 = 0 }

theorem value_of_a (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end value_of_a_l446_446041


namespace common_tangent_range_of_a_l446_446361

theorem common_tangent_range_of_a 
  (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ 
    (-a / x1^2 = 2 / x2) ∧ 
    (2 * a / x1 = 2 * (log x2) - 2)) 
  ↔ (-2 / Real.exp 1 ≤ a ∧ a < 0) := by
  sorry

end common_tangent_range_of_a_l446_446361


namespace count_values_n_l446_446051

-- Define a function to calculate sum of divisors
def sum_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum id

-- Define a function to check if a number is prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- The main theorem statement
theorem count_values_n (count : ℕ) : count = 5 :=
  let primes_in_range := 
    (Finset.range (30 + 1)).filter (λ n, is_prime (sum_divisors n)) in
  primes_in_range.card = count := sorry

end count_values_n_l446_446051


namespace problem_correct_l446_446573

noncomputable def table_filled_correctly : Prop :=
  ∃ boys_physics boys_history girls_physics girls_history,
    boys_physics = 30 ∧ boys_history = 10 ∧ girls_physics = 20 ∧ girls_history = 40 ∧
    boys_physics + boys_history = 40 ∧ girls_physics + girls_history = 60 ∧
    boys_physics + girls_physics = 50 ∧ boys_history + girls_history = 50

noncomputable def chi_squared_value_correct (n a b c d : ℕ) : Prop :=
  let K2 := (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  n = 100 ∧ a = 30 ∧ b = 20 ∧ c = 10 ∧ d = 40 ∧ K2 ≥ 10.828

noncomputable def probability_one_common_subject : Prop :=
  (C(4, 1) * C(3, 1) * C(2, 1)) / (C(4, 2) * C(4, 2)) = 2 / 3

theorem problem_correct 
  (n a b c d : ℕ)
  [fact (n = 100)] [fact (a = 30)] [fact (b = 20)] [fact (c = 10)] [fact (d = 40)] :
  table_filled_correctly ∧ chi_squared_value_correct n a b c d ∧ probability_one_common_subject :=
begin
  sorry
end

end problem_correct_l446_446573


namespace compute_BP_PC_minus_AQ_QC_l446_446021

variable (ABC : Triangle) (H : Point) (A B C P Q : Point)
variable (alt_AP : Altitude ABC A P) (alt_BQ : Altitude ABC B Q)
variable (HP HQ : Real)
variable [HP_eq_7 : HP = 7] [HQ_eq_3 : HQ = 3]

theorem compute_BP_PC_minus_AQ_QC :
  (BP PC - AQ QC) = 40 := by 
  -- Given all the conditions and definitions, the rest of the proof goes here
  sorry

end compute_BP_PC_minus_AQ_QC_l446_446021


namespace correct_answer_l446_446165

def complementary (x y : ℝ) : Prop := x + y = 90
def supplementary (x y : ℝ) : Prop := x + y = 180

theorem correct_answer :
  let A := ¬(complementary 90 90 ∧ supplementary 180 180)
  let B := ¬∀ ∠1 ∠2 ∠3 : ℝ, (∠1 + ∠2 + ∠3 = 180) → (complementary ∠1 ∠2 ∧ complementary ∠2 ∠3 ∧ complementary ∠1 ∠3)
  let C := ∀ ∠1 ∠2 : ℝ, ∠1 = ∠2 → supplementary ∠1 180 = supplementary ∠2 180
  let D := ∀ ∠α ∠β : ℝ, ∠α > ∠β → (supplementary ∠α 180 > supplementary ∠β 180)
  C
sorry

end correct_answer_l446_446165


namespace integral_solution_l446_446253

noncomputable def integral_problem : ℝ :=
∫ x in 1..64, (2 + real.cbrt x) / ((real.sqrt (real.cbrt x)) + 2 * real.cbrt x + real.sqrt x) * (1 / real.sqrt x)

theorem integral_solution : integral_problem = 6 := 
by
  sorry

end integral_solution_l446_446253


namespace even_digit_count_in_base_7_of_789_l446_446293

theorem even_digit_count_in_base_7_of_789 : 
  let num := 789 
  let base := 7 
  let base_7_rep := [2, 2, 0, 5] -- This is the base-7 representation of 789
  (nat.count (λ x, x % 2 = 0) base_7_rep) = 3 := 
by sorry

end even_digit_count_in_base_7_of_789_l446_446293


namespace tax_rate_other_items_l446_446450

/-- A type synonym for representing percentages as real numbers between 0 and 100. -/
def Percentage := ℝ

def total_amount : ℝ := 100
def spent_on_clothing : ℝ := 0.5 * total_amount
def spent_on_food : ℝ := 0.2 * total_amount
def spent_on_other_items : ℝ := 0.3 * total_amount
def tax_rate_clothing : Percentage := 4
def tax_rate_food : Percentage := 0
def total_tax_rate_paid : Percentage := 5

/-- The tax paid on a particular item calculated as (tax_rate * amount) / 100 -/
def tax_paid (tax_rate : Percentage) (amount : ℝ) : ℝ :=
  (tax_rate * amount) / 100

def total_tax_paid : ℝ :=
  tax_paid tax_rate_clothing spent_on_clothing +
  tax_paid tax_rate_food spent_on_food

theorem tax_rate_other_items : 
  (total_tax_paid + tax_paid x spent_on_other_items = (total_tax_rate_paid * total_amount) / 100) →
  x = 10 :=
by
  sorry

end tax_rate_other_items_l446_446450


namespace onion_root_cell_division_l446_446627

theorem onion_root_cell_division :
  let initial_cells := 2 ^ 10
  let hours_per_division := 12
  let total_hours := 3 * 24
  let num_divisions := total_hours / hours_per_division
  initial_cells * 2 ^ num_divisions = 2 ^ 16 :=
by
  -- Definitions based on conditions provided.
  let initial_cells := 2 ^ 10
  let hours_per_division := 12
  let total_hours := 3 * 24
  let num_divisions := total_hours / hours_per_division
  -- Correct answer based on the problem-solving steps.
  have h : initial_cells * 2 ^ num_divisions = 2 ^ 16,
  { rw [initial_cells, total_hours, hours_per_division, num_divisions],
    norm_num, },
  exact h

end onion_root_cell_division_l446_446627


namespace inequality_valid_for_n_l446_446273

-- Define the inequality condition
def inequality_holds (n : ℕ) (x : Fin n → ℝ) : Prop :=
  ∑ i, x i ^ 2 ≥ (∑ i in Finset.range (n-1), x i) * x (n-1)

-- The theorem to prove
theorem inequality_valid_for_n (n : ℕ) (x : Fin n → ℝ) (hn : n ∈ {2, 3, 4, 5}) : 
  inequality_holds n x :=
by sorry

end inequality_valid_for_n_l446_446273


namespace donuts_left_l446_446351

theorem donuts_left (initial_donuts : ℕ) (missing_percentage : ℕ) (remaining_percentage : ℕ) (initial_donuts = 30) (missing_percentage = 70) (remaining_percentage = 30) :
  (initial_donuts * remaining_percentage / 100) = 9 :=
by
  sorry

end donuts_left_l446_446351


namespace find_distance_from_A_to_OB_l446_446899

noncomputable def distance_from_A_to_OB (c1 c2 : Circle) (O A B : Point) : ℝ :=
  if hc1 : radius c1 = 1 ∧ tangent c1 O ∧ ¬intersect c1 c2 ∧
  hc2 : radius c2 = 1 ∧ tangent c2 B ∧ ¬intersect c2 c1 ∧
  ∃t1 : Tangent, passes_tangent t1 O ∧ tangent_to_circles t1 c1 c2 ∧
  ∃t2 : Tangent, intersects_rays t2 A B ∧ equal_dist O A B then
    2
  else
    0

theorem find_distance_from_A_to_OB 
  (c1 c2 : Circle) (O A B : Point)
  (hc1 : radius c1 = 1 ∧ tangent c1 O ∧ ¬intersect c1 c2)
  (hc2 : radius c2 = 1 ∧ tangent c2 B ∧ ¬intersect c2 c1)
  (ht1 : ∃t : Tangent, passes_tangent t O ∧ tangent_to_circles t c1 c2)
  (ht2 : ∃t : Tangent, intersects_rays t A B ∧ equal_dist O A B) :
  distance_from_A_to_OB c1 c2 O A B = 2 := by 
  sorry

end find_distance_from_A_to_OB_l446_446899


namespace dispersion_statistics_l446_446185

-- Define the variables and possible statistics options
def sample (n : ℕ) := fin n → ℝ

inductive Statistics
| StandardDeviation : Statistics
| Median : Statistics
| Range : Statistics
| Mean : Statistics

-- Define a function that returns if a statistic measures dispersion
def measures_dispersion : Statistics → Prop
| Statistics.StandardDeviation := true
| Statistics.Median := false
| Statistics.Range := true
| Statistics.Mean := false

-- Prove that StandardDeviation and Range measure dispersion
theorem dispersion_statistics (x : sample n) :
  measures_dispersion Statistics.StandardDeviation ∧
  measures_dispersion Statistics.Range :=
by
  split;
  exact trivial

end dispersion_statistics_l446_446185


namespace slope_interval_non_intersect_l446_446045

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5

def Q : ℝ × ℝ := (10, 10)

theorem slope_interval_non_intersect (r s : ℝ) (h : ∀ m : ℝ,
  ¬∃ x : ℝ, parabola x = m * (x - 10) + 10 ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end slope_interval_non_intersect_l446_446045


namespace relationship_f_neg1_f2_l446_446492

theorem relationship_f_neg1_f2 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f(x + 1) = f(1 - x))
  (h2 : ∀ x y, 1 ≤ x → x < y → f(x) < f(y)) :
  f(-1) > f(2) :=
by
  sorry

end relationship_f_neg1_f2_l446_446492


namespace closest_furthest_times_l446_446697

noncomputable def angle_diff (h m : ℝ) : ℝ :=
  let minute_angle := 6 * m
  let hour_angle := (h + m / 60) * 30
  (abs (hour_angle - minute_angle)) % 360

theorem closest_furthest_times :
  let times := [(6, 30), (6, 31), (6, 32), (6, 33), (6, 34), (6, 35)]
  (times.argmin (λ p, angle_diff p.1 p.2) = (6, 33))
  ∧ (times.argmax (λ p, angle_diff p.1 p.2) = (6, 30)) :=
  by sorry

end closest_furthest_times_l446_446697


namespace shortest_wire_length_l446_446569

theorem shortest_wire_length (d1 d2 : ℝ) : d1 = 6 → d2 = 18 → 
  let r1 := d1 / 2
      r2 := d2 / 2
      straight_sections := 2 * (real.sqrt ((r1 + r2) ^ 2 - (r2 - r1) ^ 2))
      theta1 := real.pi / 3
      theta2 := 2 * real.pi / 3
      small_circle_arc := (theta1 / (2 * real.pi)) * 2 * real.pi * r1
      large_circle_arc := (theta2 / (2 * real.pi)) * 2 * real.pi * r2
      total_length := straight_sections + small_circle_arc + large_circle_arc
  in total_length = 12 * real.sqrt 3 + 14 * real.pi :=
by intros h1 h2; simp [h1, h2]; sorry

end shortest_wire_length_l446_446569


namespace cumulative_distribution_X_maximized_expected_score_l446_446976

noncomputable def distribution_X (p_A : ℝ) (p_B : ℝ) : (ℝ × ℝ × ℝ) :=
(1 - p_A, p_A * (1 - p_B), p_A * p_B)

def expected_score (p_A : ℝ) (p_B : ℝ) (s_A : ℝ) (s_B : ℝ) : ℝ :=
0 * (1 - p_A) + s_A * (p_A * (1 - p_B)) + (s_A + s_B) * (p_A * p_B)

theorem cumulative_distribution_X :
  distribution_X 0.8 0.6 = (0.2, 0.32, 0.48) :=
sorry

theorem maximized_expected_score :
  expected_score 0.8 0.6 20 80 < expected_score 0.6 0.8 80 20 :=
sorry

end cumulative_distribution_X_maximized_expected_score_l446_446976


namespace annual_interest_rate_l446_446754

theorem annual_interest_rate
  (principal : ℕ) (quarterly_interest : ℕ) (debenture_period_months : ℕ)
  (total_interest_earned : ℕ) (annual_interest_rate_decimal : ℚ)
  (principal_condition : principal = 10000)
  (quarterly_interest_condition : quarterly_interest = 237.5)
  (debenture_period_condition : debenture_period_months = 18)
  (total_interest_condition : total_interest_earned = 1425)
  (annual_rate_condition : annual_interest_rate_decimal = 0.095) :
  let quarters := debenture_period_months / 3 in
  let total_interest := quarters * quarterly_interest in
  let years := debenture_period_months / 12 in
  let computed_annual_interest_rate := total_interest_earned / (principal * years) in
  computed_annual_interest_rate * 100 = annual_interest_rate_decimal * 100 :=
by
  sorry

end annual_interest_rate_l446_446754


namespace general_term_of_b_sum_of_c_geq_l446_446694

-- Problem 1
theorem general_term_of_b (a_n : ℕ → ℚ) (b_n : ℕ → ℚ)
  (h1 : a_n 1 = 1 / 4)
  (h2 : ∀ n, a_n n + b_n n = 1)
  (h3 : ∀ n, b_n (n + 1) = b_n n / (1 - (a_n n)^2)) :
  ∀ n, b_n n = (n + 2) / (n + 3) :=
sorry

-- Problem 2
theorem sum_of_c_geq (a_n : ℕ → ℚ) (c_n : ℕ → ℚ) (S_n : ℕ → ℚ)
  (h1 : ∀ n, a_n n = 1 / (n + 3))
  (h2 : ∀ n, c_n n = (a_n n - (a_n n)^2) / (2^n * (1 - 2 * a_n n) * (1 - 3 * a_n n)))
  (h3 : ∀ n, S_n n = ∑ i in Finset.range n, c_n i) :
  ∀ n, S_n n ≥ 3 / 4 :=
sorry

end general_term_of_b_sum_of_c_geq_l446_446694


namespace primes_condition_count_l446_446298

theorem primes_condition_count :
  let primes := {p : ℕ | Nat.Prime p ∧ p < 100}
  let condition (p : ℕ) := (⌊(2 + Real.sqrt 5)^p⌋ - 2^(p + 1)) % p = 0
  (primes.filter condition).card = 24 :=
by sorry

end primes_condition_count_l446_446298


namespace point_inside_even_number_of_triangles_l446_446829

theorem point_inside_even_number_of_triangles {P : Point} {vertices : List Point} (h1 : is_convex (Polygon vertices)) (h2 : 2 ∣ length vertices) (h3 : length vertices = 2 * n) (h4 : n > 0) (h5 : P ∉ ∧ (Polygon.statistics (Polygon vertices)).diagonals P) :
      even (count_triangles_containing_P P vertices) :=
sorry

end point_inside_even_number_of_triangles_l446_446829


namespace length_segment_AB_maximum_area_triangle_PAB_l446_446386

open Real

def line_eq (x y : ℝ) : Prop := x - y - 2 = 0

def curve_x (θ : ℝ) : ℝ := 2 * sqrt 3 * cos θ
def curve_y (θ : ℝ) : ℝ := 2 * sin θ

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  (x^2 / 12) + (y^2 / 4) = 1

def intersection (x1 y1 x2 y2 : ℝ) : Prop := 
  ellipse_eq x1 y1 ∧ line_eq x1 y1 ∧
  ellipse_eq x2 y2 ∧ line_eq x2 y2 ∧
  x1 ≠ x2

theorem length_segment_AB :
  ∀ x1 y1 x2 y2 : ℝ,
    intersection x1 y1 x2 y2 →
    dist (x1, y1) (x2, y2) = 3 * sqrt 2 := sorry

theorem maximum_area_triangle_PAB :
  ∀ θ P_x P_y : ℝ,
    (P_x = curve_x θ) →
    (P_y = curve_y θ) →
    θ = 5 * π / 6 →
    P_x = -3 →
    P_y = 1 ∧ 
    9 = dist (3, sqrt 3) (0, 2) * dist (_ stands for half length {for max area}) := sorry

end length_segment_AB_maximum_area_triangle_PAB_l446_446386


namespace AnyaPalindromeAfterDay_l446_446244

def isPalindrome (s : String) : Prop := s = s.reverse

-- Define a function to append the word on another person's strip
def appendWord (a b : String) : String := a ++ b

-- Define an inductive proposition for the word on Anya's strip after n minutes
inductive Steps : Nat → String → Prop
| initial (a b : String) : a = "A" → b = "B" → Steps 0 a
| step (n : Nat) (a b : String) (ha : Steps n a) (hb : Steps n b) (appendToAnya : Bool) : 
  Steps (n+1) (if appendToAnya then appendWord b a else appendWord a b)

theorem AnyaPalindromeAfterDay : 
  ∀ {a : String}, Steps 1440 a → ∃ s t : String, isPalindrome (s ++ t) :=
by
  sorry

end AnyaPalindromeAfterDay_l446_446244


namespace increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446913

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446913


namespace dispersion_statistics_l446_446183

-- Define the variables and possible statistics options
def sample (n : ℕ) := fin n → ℝ

inductive Statistics
| StandardDeviation : Statistics
| Median : Statistics
| Range : Statistics
| Mean : Statistics

-- Define a function that returns if a statistic measures dispersion
def measures_dispersion : Statistics → Prop
| Statistics.StandardDeviation := true
| Statistics.Median := false
| Statistics.Range := true
| Statistics.Mean := false

-- Prove that StandardDeviation and Range measure dispersion
theorem dispersion_statistics (x : sample n) :
  measures_dispersion Statistics.StandardDeviation ∧
  measures_dispersion Statistics.Range :=
by
  split;
  exact trivial

end dispersion_statistics_l446_446183


namespace three_gatherers_collect_at_least_fifty_l446_446075

theorem three_gatherers_collect_at_least_fifty (a : ℕ → ℕ) (h_sum : ∑ i in finset.range 7, a i = 100)
  (h_distinct : ∀ i j : fin (7 + 1), i ≠ j → a i ≠ a j) :
  ∃ i j k : fin (7 + 1), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ a i + a j + a k ≥ 50 :=
by
  -- Proof to be filled here.
  sorry

end three_gatherers_collect_at_least_fifty_l446_446075


namespace donuts_ratio_l446_446269

noncomputable def donuts_problem : Prop :=
  let total_donuts := 40
  let delta_donuts := 8
  let gamma_donuts := 8
  let beta_donuts := total_donuts - delta_donuts - gamma_donuts
  (beta_donuts : ℝ) / (gamma_donuts : ℝ) = 3

theorem donuts_ratio : donuts_problem :=
by
  have total_donuts : ℝ := 40
  have delta_donuts : ℝ := 8
  have gamma_donuts : ℝ := 8
  have beta_donuts : ℝ := total_donuts - delta_donuts - gamma_donuts
  change ((total_donuts - delta_donuts - gamma_donuts) : ℝ) / (gamma_donuts : ℝ) = 3
  sorry

end donuts_ratio_l446_446269


namespace diane_owes_money_l446_446279

theorem diane_owes_money (initial_amount winnings total_losses : ℤ) (h_initial : initial_amount = 100) (h_winnings : winnings = 65) (h_losses : total_losses = 215) : 
  initial_amount + winnings - total_losses = -50 := by
  sorry

end diane_owes_money_l446_446279


namespace intersection_M_N_l446_446696

noncomputable def set_M : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - x^2)}
noncomputable def set_N : Set ℝ := {y | ∃ x, y = x^2 - 1}

theorem intersection_M_N :
  (set_M ∩ set_N) = { x | -1 ≤ x ∧ x ≤ Real.sqrt 2 } := sorry

end intersection_M_N_l446_446696


namespace b_value_for_roots_forming_parallelogram_l446_446274

-- Define the conditions of the problem
def polynomial (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 8*z^3 + (13 * b) * z^2 - 5 * (2 * b^2 + 4 * b - 4) * z + 4

-- Statement to prove
theorem b_value_for_roots_forming_parallelogram : ∀ b : ℝ,
  (∃ z1 z2 z3 z4 : ℂ, 
    polynomial b z1 = 0 ∧ polynomial b z2 = 0 ∧ polynomial b z3 = 0 ∧ polynomial b z4 = 0 ∧ 
    (z1 - z2) * (z3 - z4) = 0 ∧ (z1 + z2 + z3 + z4) / 4 = 2) →
  b = 1.5 :=
begin
  sorry
end

end b_value_for_roots_forming_parallelogram_l446_446274


namespace marble_pile_sum_l446_446531

theorem marble_pile_sum (n : ℕ) : ∃ S, S = n * (n - 1) / 2 :=
by {
  use (n * (n - 1) / 2),
  sorry,
}

end marble_pile_sum_l446_446531


namespace clean_time_per_room_l446_446965

variable (h : ℕ)

-- Conditions
def floors := 4
def rooms_per_floor := 10
def total_rooms := floors * rooms_per_floor
def hourly_wage := 15
def total_earnings := 3600

-- Question and condition mapping to conclusion
theorem clean_time_per_room (H1 : total_rooms = 40) 
                            (H2 : total_earnings = 240 * hourly_wage) 
                            (H3 : 240 = 40 * h) :
                            h = 6 :=
by {
  sorry
}

end clean_time_per_room_l446_446965


namespace product_mod_7_l446_446867

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l446_446867


namespace proposition1_proposition2_proposition3_proposition4_main_result_l446_446687

theorem proposition1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : ∃ x, x = 1 ∧ 4*x^3 - a*x^2 - 2*b*x + 2 has_extreme_value_at x = 1) : ab <= 9 :=
sorry

theorem proposition2 (x1 x2 : ℝ) (hx1 : - (Real.pi / 2) <= x1 ∧ x1 <= Real.pi / 2) (hx2 : - (Real.pi / 2) <= x2 ∧ x2 <= Real.pi / 2) (h : x1 <= x2) : |x1| > |x2| :=
sorry

theorem proposition3 (a : ℝ) (ha : a > 2) : ∃! x : ℝ, (0 < x) ∧ (x < 2) ∧ x^3 - 3*a*x^2 + 3 = 0 :=
sorry

theorem proposition4 : ∀ x y : ℝ, (y = 3*x - x^3) (x = 2) ∧ (y = -2) → (y + 9*x - 16) = 0 :=
sorry

theorem main_result : {p : ℕ | p ∈ {1, 2, 3, 4} ∧ (p = 1 ∨ p = 2 ∨ p = 3)} :=
sorry

end proposition1_proposition2_proposition3_proposition4_main_result_l446_446687


namespace johns_pants_cost_50_l446_446411

variable (P : ℝ)

theorem johns_pants_cost_50 (h1 : P + 1.60 * P = 130) : P = 50 := 
by
  sorry

end johns_pants_cost_50_l446_446411


namespace max_elements_in_A_l446_446884

noncomputable def max_possible_elements_in_A : ℕ :=
  Nat.choose 2022 1011

structure valid_function (f : ℕ → ℕ) : Prop :=
  (non_increasing : ∀ (x y : ℕ), (1 ≤ x ∧ x < y ∧ y ≤ 2023) → f x ≥ f y)
  (composition_property : ∀ (g : ℕ → ℕ) (x : ℕ), (1 ≤ x ∧ x ≤ 2023) →
    f (g x) = g (f (g x)))

def is_valid_A (A : Finset (ℕ → ℕ)) : Prop :=
  ∀ f ∈ A, valid_function f

theorem max_elements_in_A :
  ∃ (A : Finset (ℕ → ℕ)), is_valid_A A ∧ A.card = max_possible_elements_in_A := by
  sorry

end max_elements_in_A_l446_446884


namespace prove_inequality_l446_446420

noncomputable def problem_statement (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) : Prop :=
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1

theorem prove_inequality (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r) : 
  problem_statement p q r n h_pqr :=
by
  sorry

end prove_inequality_l446_446420


namespace solve_system_of_equations_l446_446825

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (3 * x + y = 8) ∧ (2 * x - y = 7) ∧ x = 3 ∧ y = -1 :=
by {
  -- Variables
  let x := 3,
  let y := -1,
  use [x, y],
  -- Show conditions are satisfied
  split,
  {
    calc 3 * x + y = 3 * 3 + (-1) : by sorry
           ...    = 8            : by sorry,
  },
  {
    split,
    {
      calc 2 * x - y = 2 * 3 - (-1) : by sorry
             ...    = 7            : by sorry,
    },
    {
      split; { trivial }
    }
  }
}

end solve_system_of_equations_l446_446825


namespace gcd_f_x_l446_446326

def f (x : ℤ) : ℤ := (5 * x + 3) * (11 * x + 2) * (14 * x + 7) * (3 * x + 8)

theorem gcd_f_x (x : ℤ) (hx : x % 3456 = 0) : Int.gcd (f x) x = 48 := by
  sorry

end gcd_f_x_l446_446326


namespace product_remainder_mod_7_l446_446873

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l446_446873


namespace ball_bounce_height_l446_446217

noncomputable def height_after_bounces (h₀ : ℝ) (r : ℝ) (b : ℕ) : ℝ :=
  h₀ * (r ^ b)

theorem ball_bounce_height
  (h₀ : ℝ) (r : ℝ) (hb : ℕ) (h₀_pos : h₀ > 0) (r_pos : 0 < r ∧ r < 1) (h₀_val : h₀ = 320) (r_val : r = 3 / 4) (height_limit : ℝ) (height_limit_val : height_limit = 40):
  (hb ≥ 6) ∧ height_after_bounces h₀ r hb < height_limit :=
by
  sorry

end ball_bounce_height_l446_446217


namespace find_n_l446_446509

-- Define that n is a positive integer
def positive_integer (n : ℕ) : Prop := n > 0

-- Define number of divisors
def num_divisors (n : ℕ) : ℕ := (finset.range (n+1)).filter (λ d, n % d = 0).card

-- Define the product of divisors function
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
(finset.range (n+1)).filter (λ d, n % d = 0).prod id

-- The final theorem statement to be proven
theorem find_n (n : ℕ) (hn : positive_integer n) :
  prod_of_divisors n = 1024 → n = 16 :=
by { sorry }

end find_n_l446_446509


namespace equation_has_more_than_one_solution_l446_446059

noncomputable def value_of_100a_plus_4b (a b : ℝ) : ℝ :=
if (1 - 4 * a = 0 ∧ 2 * b - 3 = 0) then 100 * a + 4 * b else 0

theorem equation_has_more_than_one_solution (a b : ℝ) (h1 : 1 - 4 * a = 0) (h2 : 2 * b - 3 = 0) :
  100 * a + 4 * b = 31 :=
by
  have ha : a = 1 / 4, from eq_of_sub_eq_zero h1,
  have hb : b = 3 / 2, from eq_of_sub_eq_zero h2,
  rw [ha, hb],
  norm_num

end equation_has_more_than_one_solution_l446_446059


namespace line_of_BC_eq_l446_446344

noncomputable def triangle_ABC : Type := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

def vertex_A : ℝ × ℝ := (1, 4)

def angle_bisector_B : ℝ → ℝ → Prop := λ x y, x + y - 1 = 0
def angle_bisector_C : ℝ → ℝ → Prop := λ x y, x - 2y = 0

theorem line_of_BC_eq :
  ∃ a b c : ℝ, (a = 4) ∧ (b = 17) ∧ (c = 12) ∧
  (∀ (x y : ℝ), angle_bisector_B x y → angle_bisector_C x y → (a * x + b * y + c = 0)) :=
sorry

end line_of_BC_eq_l446_446344


namespace product_of_divisors_eq_1024_l446_446520

theorem product_of_divisors_eq_1024 (n : ℕ) (h1 : 0 < n) (h2 : ∏ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446520


namespace average_speed_l446_446197

-- Define the average speed v
variable {v : ℝ}

-- Conditions
def day1_distance : ℝ := 160  -- 160 miles on the first day
def day2_distance : ℝ := 280  -- 280 miles on the second day
def time_difference : ℝ := 3  -- 3 hours difference

-- Theorem to prove the average speed
theorem average_speed (h1 : day1_distance / v + time_difference = day2_distance / v) : v = 40 := 
by 
  sorry  -- Proof is omitted

end average_speed_l446_446197


namespace represent_natural_number_l446_446268

def f (n : ℕ) : ℕ := 10 * n
def g (n : ℕ) : ℕ := 10 * n + 4
def h (n : ℕ) : ℕ := n / 2

theorem represent_natural_number (N : ℕ) (hN : N ≥ 4) : ∃ m : ℕ, ∃ f_gh_sequence : list ℕ → ℕ, f_gh_sequence [N] = m :=
by
  sorry

end represent_natural_number_l446_446268


namespace constant_term_in_expansion_l446_446390

theorem constant_term_in_expansion : 
  let f1 := (λ (x : ℝ), x^3 + 2)
      f2 := (λ (x : ℝ), (2*x - (1 / x^2))^6)
  in
  ∀ x : ℝ, x ≠ 0 →
  let exp := f1 x * f2 x
  in (∃ c : ℝ, ∀ y : ℝ, exp = c * y^0) → c = 320 :=
by
  -- Assume all necessary variables and imports are brought in
  -- Define the necessary terms and their expansions
  let f1 : ℝ → ℝ := λ x, x^3 + 2
  let f2 : ℝ → ℝ := λ x, (2 * x - 1 / x^2)^6
  assume x hx
  let exp := f1 x * f2 x
  
  -- Formalize the statement about the constant term
  have H : ∃ c : ℝ, ∀ y : ℝ, exp = c * y^0, from sorry,
  have Heq : c = 320, from sorry,
  exact Heq

end constant_term_in_expansion_l446_446390


namespace increasing_function_range_l446_446678

def f (a : ℝ) (x : ℝ) : ℝ :=
  if 1 < x then a ^ x else (2 - a / 2) * x + 2

theorem increasing_function_range (a : ℝ) :
  (∀ x > 1, f a x < f a (x + 1)) ∧ (∀ x ≤ 1, f a x ≤ f a (x + 1)) ∧ (f a 1 = a) ↔ (8/3 ≤ a ∧ a < 4) := 
sorry

end increasing_function_range_l446_446678


namespace product_remainder_mod_7_l446_446860

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l446_446860


namespace statistics_measuring_dispersion_l446_446173

-- Definition of standard deviation
def standard_deviation (X : List ℝ) : ℝ :=
  let mean := (X.sum) / (X.length)
  (X.map (λ x => (x - mean) ^ 2)).sum / X.length

-- Definition of range
def range (X : List ℝ) : ℝ :=
  X.maximum.get - X.minimum.get

-- Definition of median
noncomputable def median (X : List ℝ) : ℝ :=
  let sorted := List.sorted X
  if sorted.length % 2 == 1 then sorted.get (sorted.length / 2)
  else (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2

-- Definition of mean
def mean (X : List ℝ) : ℝ :=
  X.sum / X.length

-- The proof statement
theorem statistics_measuring_dispersion (X : List ℝ) :
  (standard_deviation X ≠ 0 ∧ range X ≠ 0) ∧
  (∀ x : ℝ, x ∈ X ↔ (median X = x ∨ mean X = x)) → true :=
  sorry

end statistics_measuring_dispersion_l446_446173


namespace time_in_rain_is_48_l446_446820

def speed_non_rainy : ℚ := 40 / 60
def speed_rainy : ℚ := 25 / 60
def stop_time : ℚ := 15
def total_time : ℚ := 75
def total_distance : ℚ := 28

theorem time_in_rain_is_48 (t_r : ℚ) 
  (h1 : t_r = 48)
  (h2 : ((total_time - stop_time) - t_r) = 60 - t_r)
  (h3 : (60 - t_r) * speed_non_rainy + t_r * speed_rainy = total_distance) :
  t_r = 48 :=
begin
  sorry
end

end time_in_rain_is_48_l446_446820


namespace problem1_l446_446209

theorem problem1 (x : ℝ) : let m := x^2 - 1 in let n := 2*x + 2 in m ≥ 0 ∨ n ≥ 0 := 
by
  let m := x^2 - 1
  let n := 2*x + 2
  sorry

end problem1_l446_446209


namespace max_value_abcd_l446_446332

-- Define the digits and constraints on them
def distinct_digits (a b c d e : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

-- Encode the given problem as a Lean theorem
theorem max_value_abcd (a b c d e : ℕ) 
  (h₀ : distinct_digits a b c d e)
  (h₁ : 0 ≤ a ∧ a ≤ 9) 
  (h₂ : 0 ≤ b ∧ b ≤ 9) 
  (h₃ : 0 ≤ c ∧ c ≤ 9) 
  (h₄ : 0 ≤ d ∧ d ≤ 9)
  (h₅ : 0 ≤ e ∧ e ≤ 9)
  (h₆ : e ≠ 0)
  (h₇ : a * 1000 + b * 100 + c * 10 + d = (a * 100 + a * 10 + d) * e) :
  a * 1000 + b * 100 + c * 10 + d = 3015 :=
by {
  sorry
}

end max_value_abcd_l446_446332


namespace bricks_required_to_pave_courtyard_l446_446193

theorem bricks_required_to_pave_courtyard :
  (20 * 100) * (16 * 100) / (20 * 10) = 16000 :=
by
  -- Given (conditions)
  let courtyard_length := 20 * 100  -- converting meters to centimeters
  let courtyard_width := 16 * 100   -- converting meters to centimeters
  let brick_length := 20
  let brick_width := 10

  -- Calculate areas
  let courtyard_area := courtyard_length * courtyard_width
  let brick_area := brick_length * brick_width

  -- Calculate number of bricks
  have h : courtyard_area / brick_area = 3200000 / 200 :=
    by simp [courtyard_area, brick_area]
  calc
    (20 * 100) * (16 * 100) / (20 * 10) = courtyard_area / brick_area : by simp
    ... = 16000 : by simp [courtyard_area, brick_area]

end bricks_required_to_pave_courtyard_l446_446193


namespace school_selection_theorem_l446_446843

-- Define the basic setup and conditions
def school_selection_problem : Prop :=
  let schools := ["A", "B", "C", "D"]
  let total_schools := 4
  let selected_schools := 2
  let combinations := Nat.choose total_schools selected_schools
  let favorable_outcomes := Nat.choose (total_schools - 1) (selected_schools - 1)
  let probability := (favorable_outcomes : ℚ) / (combinations : ℚ)
  probability = 1 / 2

-- Proof is yet to be provided
theorem school_selection_theorem : school_selection_problem := sorry

end school_selection_theorem_l446_446843


namespace cone_new_height_l446_446998

noncomputable def new_cone_height : ℝ := 6

theorem cone_new_height (r h V : ℝ) (circumference : 2 * Real.pi * r = 24 * Real.pi)
  (original_height : h = 40) (same_base_circumference : 2 * Real.pi * r = 24 * Real.pi)
  (volume : (1 / 3) * Real.pi * (r ^ 2) * new_cone_height = 288 * Real.pi) :
    new_cone_height = 6 := 
sorry

end cone_new_height_l446_446998


namespace find_T_n_sum_l446_446682

noncomputable def T_n (n : ℕ) : ℝ :=
  (1 / 8) * (1 / (Real.log2 ((6 / 17 : ℝ))^2) + 1 / (Real.log2 ((24 / 17 : ℝ))^2) -
             1 / (Real.log2 ((6 / 17 * 4^n : ℝ))^2) - 1 / (Real.log2 ((6 / 17 * 4^(n+1) : ℝ))^2))

theorem find_T_n_sum (n : ℕ) : 
  ∑ k in Finset.range n, (Real.log2 ((6 / 17 * 4^k : ℝ))^2 * (Real.log2 ((6 / 17 * 4^(k+1) : ℝ))^2))⁻¹ = T_n n :=
by sorry

end find_T_n_sum_l446_446682


namespace solve_abs_eq_l446_446551

theorem solve_abs_eq (x : ℝ) (h : |x + 2| = |x - 3|) : x = 1 / 2 :=
sorry

end solve_abs_eq_l446_446551


namespace fg_difference_l446_446655

def f (x : ℕ) : ℕ := x^2 - 3 * x + 4
def g (x : ℕ) : ℕ := x - 2

theorem fg_difference :
  f (g 5) - g (f 5) = -8 := by
  sorry

end fg_difference_l446_446655


namespace optimal_strategy_l446_446973

noncomputable def prob_A_correct : ℝ := 0.8
noncomputable def prob_B_correct : ℝ := 0.6
noncomputable def score_A_correct : ℝ := 20
noncomputable def score_B_correct : ℝ := 80

def X_distribution : (ℝ → ℝ) :=
λ x, if x = 0 then 1 - prob_A_correct
     else if x = 20 then prob_A_correct * (1 - prob_B_correct)
     else if x = 100 then prob_A_correct * prob_B_correct
     else 0

noncomputable def E_X : ℝ :=
(0 * (1 - prob_A_correct)) + (20 * (prob_A_correct * (1 - prob_B_correct))) + (100 * (prob_A_correct * prob_B_correct))

noncomputable def E_Y : ℝ :=
(0 * (1 - prob_B_correct)) + (80 * (prob_B_correct * (1 - prob_A_correct))) + (100 * (prob_B_correct * prob_A_correct))

theorem optimal_strategy : E_X = 54.4 ∧ E_Y = 57.6 → (57.6 > 54.4) :=
by {
  sorry 
}

end optimal_strategy_l446_446973


namespace enclosed_area_eq_one_third_l446_446836

noncomputable def area_between_curves : ℝ := ∫ x in set.Icc 0 1, (Real.sqrt x - x^2)

theorem enclosed_area_eq_one_third :
  area_between_curves = 1 / 3 :=
sorry

end enclosed_area_eq_one_third_l446_446836


namespace total_people_present_l446_446892

def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698
def total_people : ℕ := number_of_parents + number_of_pupils

theorem total_people_present : total_people = 803 :=
by
  sorry

end total_people_present_l446_446892


namespace dispersion_measures_l446_446168
-- Definitions for statistical measures (for clarity, too simplistic)
def standard_deviation (x : List ℝ) : ℝ := 
  let mean := (x.sum / x.length)
  Math.sqrt ((x.map (λ xi => (xi - mean)^2)).sum / (x.length - 1))

def median (x : List ℝ) : ℝ := 
  let sorted := x.qsort (≤)
  if h : sorted.length % 2 = 1 then (sorted.sorted.nth (sorted.length / 2))
  else ((sorted.nth (sorted.length / 2 - 1) + sorted.nth (sorted.length / 2)) / 2)

def range (x : List ℝ) : ℝ := x.maximum - x.minimum

def mean (x : List ℝ) : ℝ := x.sum / x.length

-- Statement to prove
theorem dispersion_measures (x : List ℝ) : 
  (standard_deviation x ∈ {standard_deviation x, range x}) ∧ 
  (range x ∈ {standard_deviation x, range x}) ∧
  ¬ (median x ∈ {standard_deviation x, range x})  ∧
  ¬ (mean x ∈ {standard_deviation x, range x}) := 
sorry

end dispersion_measures_l446_446168


namespace solution_set_a2_range_of_a_l446_446339

noncomputable def f (x a : ℝ) := abs (x - 2) + abs (x - a)

-- Proof statement for part (1)
theorem solution_set_a2 : 
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 4} :=
sorry

-- Proof statement for part (2)
theorem range_of_a :
  {a : ℝ | ∀ x : ℝ, f x a < 4 → x ∈ {1, 2, 3}} = {2} :=
sorry

end solution_set_a2_range_of_a_l446_446339


namespace equal_points_probability_l446_446900

theorem equal_points_probability (p : ℝ) (prob_draw_increases : 0 ≤ p ∧ p ≤ 1) :
  (∀ q : ℝ, (0 ≤ q ∧ q < p) → (q^2 + (1 - q)^2 / 2 < p^2 + (1 - p)^2 / 2)) → False :=
begin
  sorry
end

end equal_points_probability_l446_446900


namespace christina_transfer_amount_l446_446258

theorem christina_transfer_amount:
  ∀ (initial_balance remaining_balance : ℕ), initial_balance = 27004 → remaining_balance = 26935 → initial_balance - remaining_balance = 69 :=
by
  intros initial_balance remaining_balance h_initial h_remaining
  rw [h_initial, h_remaining]
  exact Nat.sub_self _


end christina_transfer_amount_l446_446258


namespace prob_rel_prime_to_50_l446_446931

theorem prob_rel_prime_to_50 : (∑ n in Finset.range 51, if Nat.gcd n 50 = 1 then 1 else 0) / 50 = 2 / 5 := sorry

end prob_rel_prime_to_50_l446_446931


namespace johnsYearlyHaircutExpenditure_l446_446033

-- Definitions based on conditions
def hairGrowthRate : ℝ := 1.5 -- inches per month
def hairCutLength : ℝ := 9 -- inches
def hairAfterCut : ℝ := 6 -- inches
def monthsBetweenCuts := (hairCutLength - hairAfterCut) / hairGrowthRate
def haircutCost : ℝ := 45 -- dollars
def tipPercent : ℝ := 0.20
def tipsPerHaircut := tipPercent * haircutCost

-- Number of haircuts in a year
def numHaircutsPerYear := 12 / monthsBetweenCuts

-- Total yearly expenditure
def yearlyHaircutExpenditure := numHaircutsPerYear * (haircutCost + tipsPerHaircut)

theorem johnsYearlyHaircutExpenditure : yearlyHaircutExpenditure = 324 := 
by
  sorry

end johnsYearlyHaircutExpenditure_l446_446033


namespace shortest_side_of_triangle_l446_446592

noncomputable def triangle_shortest_side (AB : ℝ) (AD : ℝ) (DB : ℝ) (radius : ℝ) : ℝ :=
  let x := 6
  let y := 5
  2 * y

theorem shortest_side_of_triangle :
  let AB := 16
  let AD := 7
  let DB := 9
  let radius := 5
  AB = AD + DB →
  (AD = 7) ∧ (DB = 9) ∧ (radius = 5) →
  triangle_shortest_side AB AD DB radius = 10 :=
by
  intros h1 h2
  -- proof goes here
  sorry

end shortest_side_of_triangle_l446_446592


namespace find_f_5_l446_446675

def f (x : ℝ) : ℝ := sorry -- we need to create a function under our condition

theorem find_f_5 : f 5 = 0 :=
sorry

end find_f_5_l446_446675


namespace intersection_parallel_l446_446594

-- Definitions of the geometric setup
structure Tetrahedron (P A B C : Type) extends Equilateral P A B C :=
  (right_angle_at_A : ∠ P A B = 90)
  (right_angle_at_B : ∠ P B C = 90)
  (right_angle_at_C : ∠ P C A = 90)

variables {P A B C : Type}
variables (T : Tetrahedron P A B C)

-- Definition of planes and intersections
variables {α β : Type} [Plane α] [Plane β]
variables (P_in_α : P ∈ α) (α_parallel_ABC : α ∥ Plane ABC)
variables (BC_in_β : BC ∈ β) (β_parallel_PA : β ∥ PA)
variables (m : Line) (m_intersection_PBC : m = α ∩ Plane PBC)
variables (n : Line) (n_intersection_α_β : n = α ∩ β)

-- Lean 4 statement for the proof problem
theorem intersection_parallel (h : ∀ m n, m ∥ BC → BC ∥ n → m ∥ n)
  : m ∥ n :=
by sorry

end intersection_parallel_l446_446594


namespace Beth_crayons_proof_l446_446605

def Beth_packs_of_crayons (packs_crayons : ℕ) (total_crayons extra_crayons : ℕ) : ℕ :=
  total_crayons - extra_crayons

theorem Beth_crayons_proof
  (packs_crayons : ℕ)
  (each_pack_contains total_crayons extra_crayons : ℕ)
  (h_each_pack : each_pack_contains = 10) 
  (h_extra : extra_crayons = 6)
  (h_total : total_crayons = 40) 
  (valid_packs : packs_crayons = (Beth_packs_of_crayons total_crayons extra_crayons / each_pack_contains)) :
  packs_crayons = 3 :=
by
  rw [h_each_pack, h_extra, h_total] at valid_packs
  sorry

end Beth_crayons_proof_l446_446605


namespace average_difference_l446_446458

theorem average_difference 
  (differences : List ℤ)
  (daily_differences : differences = [15, -5, 25, -10, 15, 5, 20])
  (days : ℕ)
  (num_days : days = 7) :
  (∑ x in differences, x) / days = 9 :=
by
  sorry

end average_difference_l446_446458


namespace simplify_expression_l446_446076

theorem simplify_expression (x : ℕ) : (5 * x^4)^3 = 125 * x^(12) := by
  sorry

end simplify_expression_l446_446076


namespace cross_sectional_area_volume_of_R_l446_446664

open Real

def semicircle (x : ℝ) := sqrt (1 - x^2)

def S (x : ℝ) : ℝ := (1 + 2 * x) / (1 + x) * sqrt (1 - x^2)

theorem cross_sectional_area (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  S(x) = (1 + 2 * x) / (1 + x) * sqrt (1 - x^2) := 
sorry

theorem volume_of_R :
  (∫ x in 0..1, S(x)) = 1 :=
sorry

end cross_sectional_area_volume_of_R_l446_446664


namespace tangent_line_and_area_l446_446265

noncomputable def tangent_line_equation (t : ℝ) : String := 
  "x + e^t * y - t - 1 = 0"

noncomputable def area_triangle_MON (t : ℝ) : ℝ :=
  (t + 1)^2 / (2 * Real.exp t)

theorem tangent_line_and_area (t : ℝ) (ht : t > 0) :
  tangent_line_equation t = "x + e^t * y - t - 1 = 0" ∧
  area_triangle_MON t = (t + 1)^2 / (2 * Real.exp t) := by
  sorry

end tangent_line_and_area_l446_446265


namespace cos_alpha_solution_l446_446324

theorem cos_alpha_solution (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 1 / 2) : 
  Real.cos α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_alpha_solution_l446_446324


namespace total_dots_not_visible_l446_446649

noncomputable def total_dots_on_die : ℕ :=
  1 + 2 + 3 + 4 + 5 + 6

noncomputable def total_dots_on_dice (n : ℕ) : ℕ :=
  n * total_dots_on_die

noncomputable def visible_faces_sum (visible : List ℕ) : ℕ :=
  visible.sum

theorem total_dots_not_visible :
  ∀ visible : List ℕ, visible = [1, 2, 2, 3, 4, 4, 5, 6, 6] →
  total_dots_on_dice 4 - visible_faces_sum visible = 51 :=
by
  intro visible h
  have h1 : total_dots_on_dice 4 = 84
    by rw [total_dots_on_dice, total_dots_on_die]; norm_num
  have h2 : visible_faces_sum visible = 33
    by simp [visible_faces_sum, h]; norm_num
  rw [h1, h2]
  norm_num

end total_dots_not_visible_l446_446649


namespace quadratic_interlaced_roots_l446_446945

theorem quadratic_interlaced_roots
  (p1 p2 q1 q2 : ℝ)
  (h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  ∃ (r1 r2 s1 s2 : ℝ),
    (r1^2 + p1 * r1 + q1 = 0) ∧
    (r2^2 + p1 * r2 + q1 = 0) ∧
    (s1^2 + p2 * s1 + q2 = 0) ∧
    (s2^2 + p2 * s2 + q2 = 0) ∧
    (r1 < s1 ∧ s1 < r2 ∨ s1 < r1 ∧ r1 < s2) :=
sorry

end quadratic_interlaced_roots_l446_446945


namespace symmetric_point_l446_446105

-- Definitions
def P : ℝ × ℝ := (5, -2)
def line (x y : ℝ) : Prop := x - y + 5 = 0

-- Statement 
theorem symmetric_point (a b : ℝ) 
  (symmetric_condition1 : ∀ x y, line x y → (b + 2)/(a - 5) * 1 = -1)
  (symmetric_condition2 : ∀ x y, line x y → (a + 5)/2 - (b - 2)/2 + 5 = 0) :
  (a, b) = (-7, 10) :=
sorry

end symmetric_point_l446_446105


namespace num_values_of_n_for_prime_g_l446_446053

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ i in Nat.divisors n, i

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∈ Finset.range (n-1).filter (λ d, d > 1), n % m ≠ 0

def g (n : ℕ) : ℕ :=
  sum_of_divisors n

theorem num_values_of_n_for_prime_g :
  (Finset.card ((Finset.range 31).filter (λ n, is_prime (g n)))) = 10 :=
by
  sorry

end num_values_of_n_for_prime_g_l446_446053


namespace no_valid_pairs_l446_446637

theorem no_valid_pairs : ∀ (x y : ℕ), x > 0 → y > 0 → x^2 + y^2 + 1 = x^3 → false := 
by
  intros x y hx hy h
  sorry

end no_valid_pairs_l446_446637


namespace ratio_hortense_olga_l446_446804

-- Definitions based on the conditions
def stripes_olga_per_shoe := 3
def shoes_per_person := 2
def stripes_rick_per_shoe := stripes_olga_per_shoe - 1
def combined_stripes := 22

-- Proof statement
theorem ratio_hortense_olga :
  let stripes_olga := stripes_olga_per_shoe * shoes_per_person in
  let stripes_rick := stripes_rick_per_shoe * shoes_per_person in
  let stripes_olga_rick := stripes_olga + stripes_rick in
  let stripes_hortense := combined_stripes - stripes_olga_rick in
  stripes_hortense / stripes_olga = 2 :=
by
  sorry

end ratio_hortense_olga_l446_446804


namespace point_P_on_AC_l446_446394

noncomputable def spatial_quadrilateral (A B C D E F G H P : Point) : Prop :=
  let ABCE := Line A B C E
  ∧ onLine E A B
  ∧ onLine F B C
  ∧ onLine G C D
  ∧ onLine H D A
  ∧ intersect EF GH P
  ∧ inPlane EF A B C
  ∧ inPlane GH A D C

theorem point_P_on_AC {A B C D E F G H P : Point} (h : spatial_quadrilateral A B C D E F G H P) :
  onLine P A C :=
sorry

end point_P_on_AC_l446_446394


namespace product_of_divisors_eq_1024_l446_446514

theorem product_of_divisors_eq_1024 (n : ℕ) (h : n > 0) (hp : ∏ d in (finset.filter (λ x, x ∣ n) (finset.range (n+1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446514


namespace sum_of_square_roots_l446_446933

theorem sum_of_square_roots : 
  (Real.sqrt 1) + (Real.sqrt (1 + 3)) + (Real.sqrt (1 + 3 + 5)) + (Real.sqrt (1 + 3 + 5 + 7)) = 10 := 
by 
  sorry

end sum_of_square_roots_l446_446933


namespace tamia_bell_pepper_pieces_l446_446090

def total_pieces (n k p : Nat) : Nat :=
  let slices := n * k
  let half_slices := slices / 2
  let smaller_pieces := half_slices * p
  let total := half_slices + smaller_pieces
  total

theorem tamia_bell_pepper_pieces :
  total_pieces 5 20 3 = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l446_446090


namespace part_a_part_b_l446_446766

variable (a : ℕ → ℝ)
variable (h₁ : ∀ n : ℕ, 1 ≤ n → 2 * (∑ i in Finset.range n, a i) = n * a (n + 1))
variable (h₂ : ∀ n : ℕ, ⌊∑ i in Finset.range n.succ, a i⌋ = ∑ i in Finset.range n.succ, ⌊a i⌋)

theorem part_a : ∃ c : ℝ, ∀ n : ℕ, a n = n * c :=
sorry

theorem part_b : (∀ n : ℕ, a n = n * a 0) → (∀ n : ℕ, a 0 = ⌊a 0⌋) → (∀ n : ℕ, a n = ⌊a n⌋) :=
sorry

end part_a_part_b_l446_446766


namespace sales_tax_difference_l446_446606

theorem sales_tax_difference :
  let item_price := 50
  let tax_rate1 := 0.075
  let tax_rate2 := 0.07
  let sales_tax1 := item_price * tax_rate1
  let sales_tax2 := item_price * tax_rate2
  sales_tax1 - sales_tax2 = 0.25 :=
by
  let item_price := 50
  let tax_rate1 := 0.075
  let tax_rate2 := 0.07
  let sales_tax1 := item_price * tax_rate1
  let sales_tax2 := item_price * tax_rate2
  show sales_tax1 - sales_tax2 = 0.25
  -- you can complete your proof here
  sorry

end sales_tax_difference_l446_446606


namespace problem_solution_l446_446435

-- Define the function f(x)
def f (x : ℝ) : ℝ := ∑' n : ℕ, (x / 2) ^ n

-- Define the conditions for x
def domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1

-- State the theorem
theorem problem_solution (x : ℝ) (h : domain x) :
  sqrt (∫ t in 0..1, f t) = sqrt (2 * log 2) := 
sorry

end problem_solution_l446_446435


namespace sock_problem_l446_446711

def sock_pair_count (total_socks : Nat) (socks_distribution : List (String × Nat)) (target_color : String) (different_color : String) : Nat :=
  if target_color = different_color then 0
  else match socks_distribution with
    | [] => 0
    | (color, count) :: tail =>
        if color = target_color then count * socks_distribution.foldl (λ acc (col_count : String × Nat) =>
          if col_count.fst ≠ target_color then acc + col_count.snd else acc) 0
        else sock_pair_count total_socks tail target_color different_color

theorem sock_problem : sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "white" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "brown" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "blue" =
                        48 :=
by sorry

end sock_problem_l446_446711


namespace find_difference_l446_446242

-- Define the initial amounts each person paid.
def Alex_paid : ℕ := 95
def Tom_paid : ℕ := 140
def Dorothy_paid : ℕ := 110
def Sammy_paid : ℕ := 155

-- Define the total spent and the share per person.
def total_spent : ℕ := Alex_paid + Tom_paid + Dorothy_paid + Sammy_paid
def share : ℕ := total_spent / 4

-- Define how much each person needs to pay or should receive.
def Alex_balance : ℤ := share - Alex_paid
def Tom_balance : ℤ := Tom_paid - share
def Dorothy_balance : ℤ := share - Dorothy_paid
def Sammy_balance : ℤ := Sammy_paid - share

-- Define the values of t and d.
def t : ℤ := 0
def d : ℤ := 15

-- The proof goal
theorem find_difference : t - d = -15 := by
  sorry

end find_difference_l446_446242


namespace prob_rel_prime_to_50_l446_446932

theorem prob_rel_prime_to_50 : (∑ n in Finset.range 51, if Nat.gcd n 50 = 1 then 1 else 0) / 50 = 2 / 5 := sorry

end prob_rel_prime_to_50_l446_446932


namespace valid_votes_polled_for_A_is_correct_l446_446739

variable (total_voters : ℕ) (percent_invalid_votes percent_A_votes : ℝ)
variable (total_votes : ℕ := 560000)
variable (percent_invalid : ℝ := 15 / 100)
variable (percent_A : ℝ := 80 / 100)

def valid_votes : ℕ := (1 - percent_invalid) * total_votes

def votes_for_A : ℕ := percent_A * valid_votes

theorem valid_votes_polled_for_A_is_correct :
  votes_for_A = 380800 := by
  sorry

end valid_votes_polled_for_A_is_correct_l446_446739


namespace acute_triangle_properties_l446_446009

-- Defining the problem setup
variables {A B C a b c : ℝ}
variable (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
variable (h_sides : a = sin A * c / sin C ∧ b = sin B * c / sin C ∧ c - b = 2 * b * cos A)

-- The proof problem statement
theorem acute_triangle_properties (h : c - b = 2 * b * cos A ∧ 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) :
  (A = 2 * B) ∧ (B ∈ (π / 6, π / 4)) ∧ ((a / b) ∈ (sqrt 2, sqrt 3)) ∧ 
  ( (1 / tan B - 1 / tan A + 2 * sin A) ∈ (5 * sqrt 3 / 3, 3) ) := sorry

end acute_triangle_properties_l446_446009


namespace equal_points_probability_l446_446904

theorem equal_points_probability (p : ℝ) (prob_draw_increases : 0 ≤ p ∧ p ≤ 1) :
  (∀ q : ℝ, (0 ≤ q ∧ q < p) → (q^2 + (1 - q)^2 / 2 < p^2 + (1 - p)^2 / 2)) → False :=
begin
  sorry
end

end equal_points_probability_l446_446904


namespace total_people_present_l446_446893

open Nat

def number_of_parents := 83
def number_of_pupils := 956
def number_of_teachers := 154
def number_of_staff := 27
def family_members_per_pupil_ratio := 2 / 6
def number_of_family_members := (number_of_pupils / 6) * 2

theorem total_people_present :
  number_of_parents + number_of_pupils + number_of_teachers + number_of_staff + number_of_family_members.floor =
  1538 := by
  sorry

end total_people_present_l446_446893


namespace swan_implications_l446_446358

variable (Swans White Birds : Type) 
variable (is_swan : Swans → Prop)
variable (is_white : Swans → Prop)
variable (is_bird : Birds → Prop)

axiom H1 : ∀ x : Swans, is_white x
axiom H2 : ∃ x : Birds, is_swan x

theorem swan_implications :
  (∀ x : Swans, is_white x) ∧ (∃ x : Birds, is_swan x) → 
  (∀ x : Swans, is_bird x) ∧ (∃ x : Birds, is_white x) ∧ ¬(∃ x : Swans, ¬is_bird x) :=
by
  intro H
  cases H with h_white h_bird_swan
  split
  -- I. All swans are birds
  { sorry }
  split
  -- II. Some birds are white
  { sorry }
  -- III. Some swans are not birds
  { sorry }

end swan_implications_l446_446358


namespace andy_paint_total_l446_446599

-- Define the given ratio condition and green paint usage
def paint_ratio (blue green white : ℕ) : Prop :=
  blue / green = 1 / 2 ∧ white / green = 5 / 2

def green_paint_used (green : ℕ) : Prop :=
  green = 6

-- Define the proof goal: total paint used
def total_paint_used (blue green white : ℕ) : ℕ :=
  blue + green + white

-- The statement to be proved
theorem andy_paint_total (blue green white : ℕ)
  (h_ratio : paint_ratio blue green white)
  (h_green : green_paint_used green) :
  total_paint_used blue green white = 24 :=
  sorry

end andy_paint_total_l446_446599


namespace face_value_of_6_in_product_l446_446142

theorem face_value_of_6_in_product :
  let num := 7098060 in
  let lv_6 := 6000 in
  let lv_8 := 80 in
  let product := lv_6 * lv_8 in
  let lv_6_in_product := 6000 in
  lv_6_in_product = 6000 → (product / lv_6_in_product) = 6 :=
by 
  intros
  sorry

end face_value_of_6_in_product_l446_446142


namespace min_sum_l446_446316

section ArithmeticSequence

variables {a_n : ℕ → ℤ} {S_n : ℕ → ℤ}
variables (d : ℤ) [HasZero d]
variables (a_1 : ℤ) (h_pos_diff : 0 < d)
variables (h_a3_a4 : a_3 * a_4 = 117)
variables (h_a2_a5 : a_2 + a_5 = -22)

-- Given the conditions, prove the general term and the minimum sum.
def general_term : Prop :=
  ∃ (d : ℤ) (a_1 : ℤ), 0 < d ∧
    (a_n = λ n, a_1 + (n - 1) * d) ∧
    (a_3 * a_4 = 117) ∧
    (a_2 + a_5 = -22) ∧
    (a_n = 4 * n - 25)

theorem min_sum (a_1 d : ℤ) : Prop :=
  2 * (λ n, n * n - (23 / 4) ^ 2) - 539 / 8 ≥ -66

end ArithmeticSequence

end min_sum_l446_446316


namespace max_acute_angles_l446_446143

structure Hexagon :=
  (angles : Fin 6 → ℝ) -- 6 angles
  (is_convex : (∀ i, angles i < 180.0) ∧ (∃ i, angles i = 90.0) ) -- convex and one right angle
  (sum_of_angles : ∑ i, angles i = 720.0) -- sum of interior angles of a hexagon

def is_acute (x : ℝ) : Prop := x < 90.0

def num_acute_angles (hex : Hexagon) : ℕ :=
  Finset.card (Finset.filter (λ i, is_acute (hex.angles i)) Finset.univ)

theorem max_acute_angles (hex : Hexagon) : num_acute_angles hex ≤ 3 :=
by
  sorry

end max_acute_angles_l446_446143


namespace probability_nonagon_diagonal_intersect_l446_446954

theorem probability_nonagon_diagonal_intersect (n : ℕ) (h_n : n = 9) :
  let diagonals := (n.choose 2) - n,
      total_diagonals_pairs := (diagonals.choose 2),
      intersecting_pairs := (n.choose 4)
  in (intersecting_pairs : ℚ) / total_diagonals_pairs = 14 / 39 :=
by {
  sorry
}

end probability_nonagon_diagonal_intersect_l446_446954


namespace M_intersection_N_equals_M_l446_446786

variable (x a : ℝ)

def M : Set ℝ := { y | ∃ x, y = x^2 + 1 }
def N : Set ℝ := { y | ∃ a, y = 2 * a^2 - 4 * a + 1 }

theorem M_intersection_N_equals_M : M ∩ N = M := by
  sorry

end M_intersection_N_equals_M_l446_446786


namespace jeff_total_cabinets_l446_446755

def initial_cabinets : ℕ := 3
def cabinets_per_counter : ℕ := 2 * initial_cabinets
def total_cabinets_installed : ℕ := 3 * cabinets_per_counter + 5
def total_cabinets (initial : ℕ) (installed : ℕ) : ℕ := initial + installed

theorem jeff_total_cabinets : total_cabinets initial_cabinets total_cabinets_installed = 26 :=
by
  sorry

end jeff_total_cabinets_l446_446755


namespace intersection_lines_l446_446500

theorem intersection_lines (c d : ℝ) :
    (∃ x y, x = (1/3) * y + c ∧ y = (1/3) * x + d ∧ x = 3 ∧ y = -1) →
    c + d = 4 / 3 :=
by
  sorry

end intersection_lines_l446_446500


namespace remainder_of_product_mod_7_l446_446879

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l446_446879


namespace non_intersecting_chords_count_l446_446211

-- Definitions based on conditions
def points_on_circle : ℕ := 12
def num_chords : ℕ := 6

-- Catalan number calculation
def catalan (n : ℕ) : ℕ := (1 / (n + 1)) * nat.choose (2 * n) n

theorem non_intersecting_chords_count :
  catalan num_chords = 132 :=
by {
  -- The proof goes here
  sorry
}

end non_intersecting_chords_count_l446_446211


namespace number_of_valid_functions_l446_446636

theorem number_of_valid_functions : 
  ∃ n : ℕ, n = 16 ∧ ∀ (a b c d : ℝ), 
    (f x = a * x^3 + b * x^2 + c * x + d) → 
    (f (-x) = -a * x^3 + b * x^2 - c * x + d) →
    (f x * f (-x) = f (x^2)) →
    n = 2^4 := 
sorry

end number_of_valid_functions_l446_446636


namespace work_completion_l446_446201

theorem work_completion 
  (x_work_days : ℕ) 
  (y_work_days : ℕ) 
  (y_worked_days : ℕ) 
  (x_rate := 1 / (x_work_days : ℚ)) 
  (y_rate := 1 / (y_work_days : ℚ)) 
  (work_remaining := 1 - y_rate * y_worked_days) 
  (remaining_work_days := work_remaining / x_rate) : 
  x_work_days = 18 → 
  y_work_days = 15 → 
  y_worked_days = 5 → 
  remaining_work_days = 12 := 
by
  intros
  sorry

end work_completion_l446_446201


namespace statistics_measuring_dispersion_l446_446174

-- Definition of standard deviation
def standard_deviation (X : List ℝ) : ℝ :=
  let mean := (X.sum) / (X.length)
  (X.map (λ x => (x - mean) ^ 2)).sum / X.length

-- Definition of range
def range (X : List ℝ) : ℝ :=
  X.maximum.get - X.minimum.get

-- Definition of median
noncomputable def median (X : List ℝ) : ℝ :=
  let sorted := List.sorted X
  if sorted.length % 2 == 1 then sorted.get (sorted.length / 2)
  else (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2

-- Definition of mean
def mean (X : List ℝ) : ℝ :=
  X.sum / X.length

-- The proof statement
theorem statistics_measuring_dispersion (X : List ℝ) :
  (standard_deviation X ≠ 0 ∧ range X ≠ 0) ∧
  (∀ x : ℝ, x ∈ X ↔ (median X = x ∨ mean X = x)) → true :=
  sorry

end statistics_measuring_dispersion_l446_446174


namespace equal_points_probability_not_always_increasing_l446_446906

theorem equal_points_probability_not_always_increasing 
  (p q : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ q) (h₂ : q ≤ 1) :
  ¬ ∀ p₀ p₁, 0 ≤ p₀ ∧ p₀ ≤ p₁ ∧ p₁ ≤ 1 → 
    let f := λ x : ℝ, (3 * x^2 - 2 * x + 1) / 4 in
    f p₀ ≤ f p₁ := by
    sorry

end equal_points_probability_not_always_increasing_l446_446906


namespace cake_heavier_than_bread_l446_446887

-- Definitions
def weight_of_7_cakes_eq_1950_grams (C : ℝ) := 7 * C = 1950
def weight_of_5_cakes_12_breads_eq_2750_grams (C B : ℝ) := 5 * C + 12 * B = 2750

-- Statement
theorem cake_heavier_than_bread (C B : ℝ)
  (h1 : weight_of_7_cakes_eq_1950_grams C)
  (h2 : weight_of_5_cakes_12_breads_eq_2750_grams C B) :
  C - B = 165.47 :=
by {
  sorry
}

end cake_heavier_than_bread_l446_446887


namespace potential_plants_total_l446_446821

def eggplants_per_packet : ℕ := 14
def sunflowers_per_packet : ℕ := 10
def tomatoes_per_packet : ℕ := 16
def peas_per_packet : ℕ := 20
def cucumbers_per_packet : ℕ := 18

def eggplant_packets : ℕ := 6
def sunflower_packets : ℕ := 8
def tomato_packets : ℕ := 7
def pea_packets : ℕ := 9
def cucumber_packets : ℕ := 5

def spring_fraction : ℝ := 0.60
def summer_fraction : ℝ := 0.70
def both_seasons_fraction : ℝ := 0.80

theorem potential_plants_total : 
    (eggplant_packets * eggplants_per_packet * spring_fraction).toInt.floor
  + (pea_packets * peas_per_packet * spring_fraction).toInt.floor
  + (sunflower_packets * sunflowers_per_packet * summer_fraction).toInt.floor
  + (cucumber_packets * cucumbers_per_packet * summer_fraction).toInt.floor
  + (tomato_packets * tomatoes_per_packet * both_seasons_fraction).toInt.floor
  = 366 := 
  sorry

end potential_plants_total_l446_446821


namespace geometric_series_problem_l446_446135

theorem geometric_series_problem
  (h1: ∃ r₁ r₂ : ℝ, 0 < r₁ ∧ r₁ < 1 ∧ 0 < r₂ ∧ r₂ < 1 ∧ r₁ ≠ r₂ ∧ (1 - r₁) + (1 - r₂) = 2 ∧ (1 - r₁) * r₁^2 = 1 / 8)
  (h2: ∃ (m n p : ℕ), (m > 0) ∧ (n > 0) ∧ (p > 0) ∧ (∀ k, prime k → ¬ (k^2 ∣ m)) ∧
                      (∃ r: ℝ, r = (1 + real.sqrt 5) / 4 ∨ r = (1 - real.sqrt 5) / 4) ∧
                      let second_term := (1 - r) * r in
                      second_term = (real.sqrt m - n) / p) :
  100 * 5 + 10 * 1 + 8 = 518 :=
by
  -- Given conditions within Lean, skipped proof
  sorry

end geometric_series_problem_l446_446135


namespace problem1_problem2_problem3_l446_446527

-- Define the sequence a_n with initial condition and recursive relation
def a : ℕ → ℝ
| 0 := 1
| (n+1) := (2^(n+1) * a n) / (a n + 2^n)

-- Problem 1: Prove that the sequence { 2^n / a_n } is arithmetic
theorem problem1 : ∃ d : ℝ, ∀ n : ℕ, (2^(n+1) / a (n+1)) - (2^n / a n) = d := by
  sorry

-- Problem 2: Find the general term formula for the sequence {a_n}
theorem problem2 : ∀ n : ℕ, a n = (2^n) / (n + 1) := by
  sorry

-- Problem 3: Sum of the first n terms of the sequence {b_n} where b_n = n(n+1)a_n
def b (n : ℕ) : ℝ := n * (n + 1) * a n

theorem problem3 (n : ℕ) : (∑ i in Finset.range n, b i) = (n-1) * 2^(n+1) + 2 := by
  sorry

end problem1_problem2_problem3_l446_446527


namespace new_pyramid_volume_l446_446231

-- Define the conditions
def initial_volume : ℝ := 60
def scale_length : ℝ := 3
def scale_width : ℝ := 4
def scale_height : ℝ := 2

-- Define the theorem statement
theorem new_pyramid_volume : 
  let initial_volume := initial_volume in
  let new_volume := scale_length * scale_width * scale_height * initial_volume in
  new_volume = 1440 :=
by
  sorry

end new_pyramid_volume_l446_446231


namespace convert_minutes_to_hours_l446_446289

namespace TimeConversions

def minutes_to_seconds (m : ℕ) : ℕ := m * 60
def seconds_to_hours (s : ℕ) : ℚ := s / 3600.0

theorem convert_minutes_to_hours (m : ℕ) (h : m = 15 ∨ m = 16) :
  seconds_to_hours (minutes_to_seconds 15.5) = 930 / 3600 := by 
  sorry
  
end TimeConversions

end convert_minutes_to_hours_l446_446289


namespace cyclists_travel_same_distance_l446_446131

-- Define constants for speeds
def v1 := 12   -- speed of the first cyclist in km/h
def v2 := 16   -- speed of the second cyclist in km/h
def v3 := 24   -- speed of the third cyclist in km/h

-- Define the known total time
def total_time := 3  -- total time in hours

-- Hypothesis: Prove that the distance traveled by each cyclist is 16 km
theorem cyclists_travel_same_distance (d : ℚ) : 
  (v1 * (total_time * 3 / 13)) = d ∧
  (v2 * (total_time * 4 / 13)) = d ∧
  (v3 * (total_time * 6 / 13)) = d ∧
  d = 16 :=
by
  sorry

end cyclists_travel_same_distance_l446_446131


namespace find_sum_l446_446602

theorem find_sum (A B : ℕ) (h1 : B = 278 + 365 * 3) (h2 : A = 20 * 100 + 87 * 10) : A + B = 4243 := by
    sorry

end find_sum_l446_446602


namespace solve_inequality_range_of_m_l446_446208

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := - abs (x + 3) + m

theorem solve_inequality (x a : ℝ) :
  (f x + a - 1 > 0) ↔
  (a = 1 → x ≠ 2) ∧
  (a > 1 → true) ∧
  (a < 1 → x < a + 1 ∨ x > 3 - a) := by sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x ≥ g x m) ↔ m < 5 := by sorry

end solve_inequality_range_of_m_l446_446208


namespace tamia_total_slices_and_pieces_l446_446085

-- Define the conditions
def num_bell_peppers : ℕ := 5
def slices_per_pepper : ℕ := 20
def num_large_slices : ℕ := num_bell_peppers * slices_per_pepper
def num_half_slices : ℕ := num_large_slices / 2
def small_pieces_per_slice : ℕ := 3
def num_small_pieces : ℕ := num_half_slices * small_pieces_per_slice
def num_uncut_slices : ℕ := num_half_slices

-- Define the total number of pieces and slices
def total_pieces_and_slices : ℕ := num_uncut_slices + num_small_pieces

-- State the theorem and provide a placeholder for the proof
theorem tamia_total_slices_and_pieces : total_pieces_and_slices = 200 :=
by
  sorry

end tamia_total_slices_and_pieces_l446_446085


namespace probability_diagonals_intersecting_in_nonagon_l446_446951

theorem probability_diagonals_intersecting_in_nonagon :
  let num_diagonals := (finset.card (finset.univ : finset (fin (nat.choose 9 2))) - 9) in
  let num_intersections := (nat.choose 9 4) in
  let total_diagonal_pairs := (nat.choose num_diagonals 2) in
  (num_intersections.to_rat / total_diagonal_pairs.to_rat) = (14/39 : ℚ) :=
by
  sorry

end probability_diagonals_intersecting_in_nonagon_l446_446951


namespace product_mod_7_l446_446865

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l446_446865


namespace domain_of_ln_function_l446_446949

-- We need to define the function and its domain condition.
def function_domain (x : ℝ) : Prop := x^2 - x > 0

theorem domain_of_ln_function : {x : ℝ | function_domain x} = set.Iio 0 ∪ set.Ioi 1 :=
by
  sorry

end domain_of_ln_function_l446_446949


namespace hexagon_vertex_reach_l446_446419

theorem hexagon_vertex_reach (n : ℤ) (h : n ≥ 2) :
  ∃ (k : ℕ), (∀ (O P : vertex), is_center(O) → is_on_perimeter(P) → 
  can_reach_in_k_moves(O, P, k)) ∧ k = 2 * n - 2 := 
sorry

structure vertex :=
(is_center : Prop)
(is_on_perimeter : Prop)

-- Placeholder for the function that checks if a piece can be moved from O to P in k moves.
-- This function must be defined in terms of the game's rules and arrows.
def can_reach_in_k_moves (O P : vertex) (k : ℕ) : Prop := sorry

end hexagon_vertex_reach_l446_446419


namespace juice_distribution_percent_per_cup_l446_446924

def pitcherCapacity (C : ℝ) (filledFraction : ℝ) (numCups : ℕ) : ℝ :=
  let juicePerCup := (filledFraction * C) / (numCups : ℝ)
  (juicePerCup / C) * 100

theorem juice_distribution_percent_per_cup
  (C : ℝ) (hC : C > 0)
  (filledFraction : ℝ) (hFraction : filledFraction = 2 / 3)
  (numCups : ℕ) (hCups : numCups = 6) :
  (pitcherCapacity C filledFraction numCups) = 100 / 9 :=
by
  sorry

end juice_distribution_percent_per_cup_l446_446924


namespace not_inverse_proportion_C_l446_446552

-- Definitions of the given functions
def fA (x : ℝ) : ℝ := 5 / x
def fB (x : ℝ) : ℝ := 3 * x^(-1)
def fC (x : ℝ) : ℝ := (x - 1) / 7
def fD (x : ℝ) : ℝ := if x = 0 then 0 else (3 / 2) * (1 / x)

-- Define what it means for a function to be an inverse proportion function
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

-- The theorem to be proved
theorem not_inverse_proportion_C : ¬ is_inverse_proportion fC :=
sorry

end not_inverse_proportion_C_l446_446552


namespace find_sum_squares_l446_446715

variables (x y : ℝ)

theorem find_sum_squares (h1 : y + 4 = (x - 2)^2) (h2 : x + 4 = (y - 2)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 15 :=
sorry

end find_sum_squares_l446_446715


namespace binomial_integral_val_l446_446101

theorem binomial_integral_val (a : ℝ) (h : a = 1 ∨ a = -1) : 
  ∫ x in -2 .. a, x^2 = 3 ∨ ∫ x in -2 .. a, x^2 = 7 / 3 := by
sorry

end binomial_integral_val_l446_446101


namespace g_four_times_odd_l446_446436

def odd_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = -g (x)

theorem g_four_times_odd (g : ℝ → ℝ) (h_odd : odd_function g) :
  odd_function (λ x, g (g (g (g x)))) :=
sorry

end g_four_times_odd_l446_446436


namespace string_length_is_correct_l446_446575

noncomputable def calculate_string_length (circumference height : ℝ) (loops : ℕ) : ℝ :=
  let vertical_distance_per_loop := height / loops
  let hypotenuse_length := Real.sqrt ((circumference ^ 2) + (vertical_distance_per_loop ^ 2))
  loops * hypotenuse_length

theorem string_length_is_correct : calculate_string_length 6 16 5 = 34 := 
  sorry

end string_length_is_correct_l446_446575


namespace area_trisect_quadrilateral_one_third_l446_446647

theorem area_trisect_quadrilateral_one_third (A B C D E F K J : Point) (AB DC : Line) :
  -- Condition: sides AB and DC are trisected at points E, F, and K, J respectively
  trisect AB E F ∧ trisect DC K J ∧ is_convex_quadrilateral A B C D →
  area (quadrilateral E F J K) = 1 / 3 * area (quadrilateral A B C D) :=
by
  sorry

end area_trisect_quadrilateral_one_third_l446_446647


namespace celine_total_cost_l446_446571

def daily_cost_literature := 0.50
def daily_cost_history := 0.50
def daily_cost_science := 0.75

def days_literature := 20
def days_history := 31
def days_science := 31

def cost_literature := daily_cost_literature * days_literature
def cost_history := daily_cost_history * days_history
def cost_science := daily_cost_science * days_science

def total_cost := cost_literature + cost_history + cost_science

theorem celine_total_cost : total_cost = 48.75 := by
  have h1 : cost_literature = 20 * 0.50 := rfl
  have h2 : cost_history = 31 * 0.50 := rfl
  have h3 : cost_science = 31 * 0.75 := rfl
  have h4 : total_cost = 10.00 + 15.50 + 23.25 := by
    rw [h1, h2, h3]
    norm_num
  exact h4

end celine_total_cost_l446_446571


namespace area_comparison_l446_446020

def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_comparison :
  let DE := 15
      DF := 20
      EF := 18
      DE' := 30
      DF' := 26
      EF' := 18
      A := triangle_area DE DF EF
      A' := triangle_area DE' DF' EF'
  in 2 * A < A' ∧ A' < 3 * A :=
  by sorry

end area_comparison_l446_446020


namespace find_f_of_2_l446_446673

theorem find_f_of_2 (f g : ℝ → ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) 
                    (h₂ : ∀ x : ℝ, g x = f x + 9) (h₃ : g (-2) = 3) :
                    f 2 = 6 :=
by
  sorry

end find_f_of_2_l446_446673


namespace part_a_part_b_l446_446451

-- Part (a)
theorem part_a (students : Fin 67) (answers : Fin 6 → Bool) :
  ∃ (s1 s2 : Fin 67), s1 ≠ s2 ∧ answers s1 = answers s2 := by
  sorry

-- Part (b)
theorem part_b (students : Fin 67) (points : Fin 6 → ℤ)
  (h_points : ∀ k, points k = k ∨ points k = -k) :
  ∃ (scores : Fin 67 → ℤ), ∃ (s1 s2 s3 s4 : Fin 67),
  s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
  scores s1 = scores s2 ∧ scores s1 = scores s3 ∧ scores s1 = scores s4 := by
  sorry

end part_a_part_b_l446_446451


namespace matrix_multiplication_problem_l446_446424

variable {A B : Matrix (Fin 2) (Fin 2) ℝ}

theorem matrix_multiplication_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = ![![5, 2], ![-2, 4]]) :
  B * A = ![![5, 2], ![-2, 4]] :=
sorry

end matrix_multiplication_problem_l446_446424


namespace find_z_and_m_range_l446_446325

def complex_number_properties (z : ℂ) : Prop :=
  (z / 4).im ≠ 0 ∧ (z / 4).re = 0 ∧ complex.abs z = 2 * real.sqrt 5

def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

def real_range_of_m (m : ℝ) (z : ℂ) : Prop :=
  fourth_quadrant (z + complex.I * m) ∧ -2 < m ∧ m < 2

theorem find_z_and_m_range :
  ∃ z : ℂ, complex_number_properties z ∧
  (fourth_quadrant ((z + complex.I * real.sqrt 5) ^ 2) →
    ∀ m : ℝ, real_range_of_m m z → -2 < m ∧ m < 2) :=
begin
  sorry
end

end find_z_and_m_range_l446_446325


namespace find_length_JM_l446_446019

-- Definitions of the problem
variables {DE DF EF : ℝ} 
variable {DEF : Triangle ℝ} 
variable {DG EH FI : Line ℝ} 
variable {J M : Point ℝ}

def centroid (DEF : Triangle ℝ) : Point ℝ := centroid(DEF)

def foot_of_altitude (J : Point ℝ) (EF : Line ℝ) : Point ℝ := foot_of_altitude(J, EF)

theorem find_length_JM :
  DE = 14 → DF = 15 → EF = 21 → 
  let G := centroid DEF,
      H := foot_of_altitude J EF,
      DN : ℝ := altitude_length DEF D EF,
      area_DEF : ℝ := herons_formula 14 15 21 
  in 
  G = J ∧ H = M ∧ DN ≠ 0 ∧ 
  84 = area DEF ∧ DN = (2 * 84) / 21 →
  JM = 8 / 3 :=
sorry

end find_length_JM_l446_446019


namespace jelly_beans_problem_l446_446455

/-- Mrs. Wonderful's jelly beans problem -/
theorem jelly_beans_problem : ∃ n_girls n_boys : ℕ, 
  (n_boys = n_girls + 2) ∧
  ((n_girls ^ 2) + ((n_girls + 2) ^ 2) = 394) ∧
  (n_girls + n_boys = 28) :=
by
  sorry

end jelly_beans_problem_l446_446455


namespace largest_n_arithmetic_sequences_l446_446600

open Int

noncomputable def a_n : ℕ → ℤ := λ n, 1 + (n - 1) * x
noncomputable def b_n : ℕ → ℤ := λ n, 1 + (n - 1) * y

theorem largest_n_arithmetic_sequences (x y : ℤ)
  (h_a : ∀ n, a_n n = 1 + (n - 1) * x)
  (h_b : ∀ n, b_n n = 1 + (n - 1) * y)
  (h1: 1 < a_n 2) (h2: a_n 2 ≤ b_n 2)
  (h3: ∃ n, a_n n * b_n n = 1764):
  ∃ n, 1 + (n - 1) * x = 1 + (44 - 1) * x ∧ 1 + (n - 1) * y = 1 + (44 - 1) * y ∧
       n = 44 := sorry

end largest_n_arithmetic_sequences_l446_446600


namespace find_coordinates_of_P_l446_446070

theorem find_coordinates_of_P (P : ℝ × ℝ) (hx : abs P.2 = 5) (hy : abs P.1 = 3) (hq : P.1 < 0 ∧ P.2 > 0) : 
  P = (-3, 5) := 
  sorry

end find_coordinates_of_P_l446_446070


namespace prime_case_composite_case_l446_446947

namespace Proofs

open Nat

-- Define gcd for convenience
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

/-- 
If 2n - 1 is a prime number, then for any n distinct positive integers a_1, a_2, ..., a_n,
there exist i, j in {1, 2, ..., n} such that (a_i + a_j) / gcd(a_i, a_j) ≥ 2n - 1.
-/
theorem prime_case {n : ℕ} (hp : Prime (2*n - 1)) 
  (a : Fin n → ℕ) (h_distinct : Function.Injective a) : 
  ∃ i j : Fin n, i ≠ j ∧ (a i + a j) / gcd (a i) (a j) ≥ 2 * n - 1 := 
by
  sorry

/-- 
If 2n - 1 is a composite number, then there exist n distinct positive integers a_1, a_2, ..., a_n 
such that for any i, j in {1, 2, ..., n}, (a_i + a_j) / gcd(a_i, a_j) < 2n - 1.
-/
theorem composite_case {n : ℕ} (hc : ¬Prime (2*n - 1)) : 
  ∃ a : Fin n → ℕ, Function.Injective a ∧
  ∀ i j : Fin n, i ≠ j → (a i + a j) / gcd (a i) (a j) < 2 * n - 1 := 
by
  sorry

end Proofs

end prime_case_composite_case_l446_446947


namespace constant_projection_vector_is_p_l446_446162

-- Define the line equation as a function of x to y coordinate representation.
def line_eq (a : ℝ) : ℝ := (3/2) * a + 3

-- Define the projection function of a vector onto another.
def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let c := w.1
  let d := w.2
  let norm_sq := c^2 + d^2
  let dot_product := v.1 * c + v.2 * d
  (dot_product / norm_sq * c, dot_product / norm_sq * d)

-- Define the vector p
def p : ℝ × ℝ := (-18/13, 12/13)

-- The proof problem statement
theorem constant_projection_vector_is_p:
  ∀ (a : ℝ) (w : ℝ × ℝ),
    let v := (a, line_eq a) in
    (c := -3/2 * w.2 -> proj v w = p) :=
sorry

end constant_projection_vector_is_p_l446_446162


namespace storks_equal_other_birds_l446_446601

-- Definitions of initial numbers of birds
def initial_sparrows := 2
def initial_crows := 1
def initial_storks := 3
def initial_egrets := 0

-- Birds arriving initially
def sparrows_arrived := 1
def crows_arrived := 3
def storks_arrived := 6
def egrets_arrived := 4

-- Birds leaving after 15 minutes
def sparrows_left := 2
def crows_left := 0
def storks_left := 0
def egrets_left := 1

-- Additional birds arriving after 30 minutes
def additional_sparrows := 0
def additional_crows := 4
def additional_storks := 3
def additional_egrets := 0

-- Final counts
def final_sparrows := initial_sparrows + sparrows_arrived - sparrows_left + additional_sparrows
def final_crows := initial_crows + crows_arrived - crows_left + additional_crows
def final_storks := initial_storks + storks_arrived - storks_left + additional_storks
def final_egrets := initial_egrets + egrets_arrived - egrets_left + additional_egrets

def total_other_birds := final_sparrows + final_crows + final_egrets

-- Theorem statement
theorem storks_equal_other_birds : final_storks - total_other_birds = 0 := by
  sorry

end storks_equal_other_birds_l446_446601


namespace coefficient_x2_in_binomial_expansion_l446_446622

theorem coefficient_x2_in_binomial_expansion :
  let a := 1
  let b := 2
  let n := 5
  binomial_expansion (1 + 2 * x) ^ 5 (x ^ 2) = 40 :=
by
  sorry

end coefficient_x2_in_binomial_expansion_l446_446622


namespace wire_length_l446_446483

def distance_between_poles := 20
def height_shorter_pole := 8
def height_taller_pole := 18

theorem wire_length :
  let d := distance_between_poles in
  let h1 := height_shorter_pole in
  let h2 := height_taller_pole in
  sqrt (d^2 + (h2 - h1)^2) = 10 * sqrt 5 :=
  sorry

end wire_length_l446_446483


namespace log_base_30_of_8_l446_446284

theorem log_base_30_of_8 (a b : ℝ) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
  Real.logb 30 8 = (3 * (1 - a)) / (1 + b) :=
by
  sorry

end log_base_30_of_8_l446_446284


namespace triangle_ABC_proof_l446_446022

open Classical
open Real
open EuclideanGeometry

/-- Proof Problem: In triangle ABC, given that:
1. ∠BAC = 2 * ∠ABC
2. AL is the angle bisector of ∠BAC
3. AK = CL where K is on ray AL
show that AK = CK.
--/
theorem triangle_ABC_proof (A B C L K : Point)
  (h1 : ∠BAC = 2 * ∠ABC )
  (h2 : IsAngleBisector(AL, ∠BAC))
  (h3 : AK = CL) :
  AK = CK :=
by
  sorry

end triangle_ABC_proof_l446_446022


namespace sum_first_23_natural_numbers_l446_446157

theorem sum_first_23_natural_numbers :
  (23 * (23 + 1)) / 2 = 276 := 
by
  sorry

end sum_first_23_natural_numbers_l446_446157


namespace dispersion_measures_correct_l446_446187

-- Define a sample data set
variable {x : ℕ → ℝ}
variable {n : ℕ}

-- Definitions of the four statistics
def standard_deviation (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def median (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def range (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def mean (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry

-- Definition of measures_dispersion function
def measures_dispersion (stat : ℕ → ℝ → ℝ) (x : ℕ → ℝ) (n : ℕ) : Prop :=
  sorry -- Define what it means for a statistic to measure dispersion

-- Problem statement in Lean
theorem dispersion_measures_correct :
  measures_dispersion standard_deviation x n ∧
  measures_dispersion range x n ∧
  ¬measures_dispersion median x n ∧
  ¬measures_dispersion mean x n :=
by sorry

end dispersion_measures_correct_l446_446187


namespace problem_I_problem_II_l446_446207

-- Problem I statement
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 := 
by
  sorry

-- Problem II statement
theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, abs (x - a) + abs (2 * x - 1) ≥ 2) →
  a ∈ Set.Iic (-3/2) ∪ Set.Ici (5/2) :=
by 
  sorry

end problem_I_problem_II_l446_446207


namespace find_250th_letter_in_pattern_l446_446139

def repeating_pattern : List Char := ['A', 'B', 'C', 'D']

theorem find_250th_letter_in_pattern : repeating_pattern[(250 % repeating_pattern.length)] = 'B' :=
by
  -- proof steps would go here
  sorry

end find_250th_letter_in_pattern_l446_446139


namespace tangent_line_at_2_l446_446291

open Real

noncomputable def curve (x : ℝ) : ℝ := 1 / x

def tangent_line_equation (p : ℝ × ℝ) : ℝ → ℝ := λ x, -(1 / 4) * (x - p.1) + p.2

theorem tangent_line_at_2 :
  ∀ x y : ℝ, curve 2 = 1 / 2 ∧ p = (2, 1 / 2) →
  y = tangent_line_equation p x →
  x + 4 * y - 4 = 0 :=
by
  intro x y h1 h2
  sorry

end tangent_line_at_2_l446_446291


namespace shaded_area_equals_22_l446_446130

-- Define the structures and conditions
structure Square :=
  (side_length : ℝ)

structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (hypotenuse : ℝ)

def square_ABCD : Square := { side_length := 6 }
def triangle_AEF : Triangle := { side1 := 4 * Real.sqrt 2, side2 := 4 * Real.sqrt 2, hypotenuse := 8 }

-- Lean 4 statement to prove the area of the shaded region
theorem shaded_area_equals_22 : 
  let area_square := square_ABCD.side_length ^ 2,
      area_triangle_BGE := 2,
      area_triangle_AEF := 1 / 2 * triangle_AEF.side1 * triangle_AEF.side2
  in area_square + area_triangle_BGE - area_triangle_AEF = 22 :=
by
  let area_square := square_ABCD.side_length ^ 2
  let area_triangle_BGE := 2
  let area_triangle_AEF := 1 / 2 * triangle_AEF.side1 * triangle_AEF.side2
  have : area_square + area_triangle_BGE - area_triangle_AEF = 22 := sorry
  exact this

end shaded_area_equals_22_l446_446130


namespace total_cost_of_umbrellas_l446_446762

theorem total_cost_of_umbrellas : 
  ∀ (h_umbrellas c_umbrellas cost_per_umbrella : ℕ),
  h_umbrellas = 2 → 
  c_umbrellas = 1 → 
  cost_per_umbrella = 8 → 
  (h_umbrellas + c_umbrellas) * cost_per_umbrella = 24 :=
by 
  intros h_umbrellas c_umbrellas cost_per_umbrella h_eq c_eq cost_eq
  rw [h_eq, c_eq, cost_eq]
  sorry

end total_cost_of_umbrellas_l446_446762


namespace range_of_a_with_common_tangent_l446_446365

noncomputable def has_common_tangent (a x₁ x₂ : ℝ) (pos_x1 : x₁ > 0) (pos_x2 : x₂ > 0) :=
  (-a / (x₁ ^ 2) = 2 / x₂) ∧ (2 * a / x₁ = 2 * (ln x₂) - 2)

theorem range_of_a_with_common_tangent : 
  ∃ (a : ℝ), ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → has_common_tangent a x₁ x₂ x₁ x₂ → (-2 / real.exp(1)) ≤ a ∧ a < 0 :=
sorry

end range_of_a_with_common_tangent_l446_446365


namespace increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446911

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446911


namespace card_arrangement_exists_l446_446808

theorem card_arrangement_exists :
  ∃ (a b c d e f g h : ℕ),
    (a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ g ≠ h) ∧
    List.nodup [a, b, c, d, e, f, g, h] ∧
    0 ≤ a ∧ a < 8 ∧ 0 ≤ b ∧ b < 8 ∧
    0 ≤ c ∧ c < 8 ∧ 0 ≤ d ∧ d < 8 ∧
    0 ≤ e ∧ e < 8 ∧ 0 ≤ f ∧ f < 8 ∧
    0 ≤ g ∧ g < 8 ∧ 0 ≤ h ∧ h < 8 ∧
    (|a - b| = 2) ∧
    (|c - d| = 3) ∧
    (|e - f| = 4) ∧
    (|g - h| = 5) :=
by
  sorry

end card_arrangement_exists_l446_446808


namespace rate_of_interest_l446_446195

-- Define variables
variables (P R T : ℝ)

-- Define simple interest formula
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- Define doubling condition
def doubles_in_8_years (P : ℝ) (simple_interest : ℝ) : Prop :=
  simple_interest = P

-- Theorem statement
theorem rate_of_interest (P : ℝ) (h : doubles_in_8_years P (simple_interest P R 8)) : R = 12.5 :=
by
  -- Note: Skip the proof using 'sorry'
  sorry

end rate_of_interest_l446_446195


namespace standard_equation_of_line_and_curve_l446_446387

theorem standard_equation_of_line_and_curve :
  let l_equation := ∀ (x y : ℝ), x + 2 * y = 10
  let curve_equation := ∀ (x y : ℝ), (x^2 / 9) + (y^2 / 4) = 1
  let M := (9 / 5, 8 / 5)
  let minimum_distance := sqrt 5
  l_equation ∧ curve_equation ∧ (dist M (λ (x y : ℝ), x + 2 * y - 10) = minimum_distance) :=
by
  sorry

end standard_equation_of_line_and_curve_l446_446387


namespace intersection_A_B_is_correct_solution_set_for_a_b_l446_446653

open Set

def A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 < 0}

theorem intersection_A_B_is_correct : A ∩ B = {x | -1 < x ∧ x < 2} :=
sorry

variables (a b : ℝ)
def C : Set ℝ := {x | x^2 + a * x + b < 0}

theorem solution_set_for_a_b :
  C = {x | -1 < x ∧ x < 2} →
  {x | x^2 + a * x - b < 0} = {x | x < -1 ∨ x > 2} :=
sorry

end intersection_A_B_is_correct_solution_set_for_a_b_l446_446653


namespace rotated_vector_is_correct_l446_446124

noncomputable def rotated_vector 
  (v : ℝ × ℝ × ℝ) 
  (orig : v = (1, 2, 2))
  (rot90 : ∀ (x y z : ℝ), v = (x * cos (π / 2) - y * sin (π / 2), x * sin (π / 2) + y * cos (π / 2), z)) 
  (on_x_axis : ∃ t : ℝ, v = (t, 0, 0)) : 
  ℝ × ℝ × ℝ :=
(2 * real.sqrt 2, -1 / real.sqrt 2, -1 / real.sqrt 2)

theorem rotated_vector_is_correct :
  rotated_vector (1, 2, 2) rfl sorry sorry = (2 * real.sqrt 2, -1 / real.sqrt 2, -1 / real.sqrt 2) :=
by { sorry }

end rotated_vector_is_correct_l446_446124


namespace proof_problem_l446_446309

def f : ℝ → ℝ
| x => if x >= 5 then x^2 else (1/2) * f (x + 1)

theorem proof_problem (x : ℝ) (h : x ∈ Icc 4 5) :
  (2 * f 4 = f 5) ∧ (f x = (x + 1)^2 / 2) := by
  sorry

end proof_problem_l446_446309


namespace simplification_incorrect_l446_446553

def optionA := -(-4.9) = 4.9
def optionB := -(4.9) = -4.9
def optionC := -(+(-4.9)) = 4.9
def optionD := +[-(4.9)] = 4.9

theorem simplification_incorrect : ¬optionD :=
by sorry

end simplification_incorrect_l446_446553


namespace area_of_hexagon_l446_446486

variable {α : Type*}

-- Definitions for angles and side lengths
variables {A B C D E F : α}
variables (angle_A angle_B angle_C angle_E : ℝ)
variables (AF AB BC CD DE EF : ℝ)

-- Conditions
def conditions : Prop :=
  angle_A = 120 ∧
  angle_B = 120 ∧
  AF = 3 ∧
  AB = 3 ∧
  BC = 3 ∧
  CD = 5 ∧
  DE = 5 ∧
  EF = 5

-- Theorem for the area of the convex hexagon
theorem area_of_hexagon (h : conditions) : ∃ area : ℝ, area = 17 * (Real.sqrt 3) / 2 :=
by
  sorry

end area_of_hexagon_l446_446486


namespace triangle_sine_ratio_l446_446753

theorem triangle_sine_ratio (A B C D : Point) (BC : Line)
  (hB : ∠ B = 45) (hC : ∠ C = 30)
  (hD : divides D BC in_ratio 1 2) :
  (sin (∠ BAD) / sin (∠ CAD)) = 1 / (4 * sqrt 2) :=
sorry

end triangle_sine_ratio_l446_446753


namespace sum_of_b_for_unique_solution_l446_446532

theorem sum_of_b_for_unique_solution :
  (∃ b1 b2, (3 * (0:ℝ)^2 + (b1 + 6) * 0 + 7 = 0 ∧ 3 * (0:ℝ)^2 + (b2 + 6) * 0 + 7 = 0) ∧ 
   ((b1 + 6)^2 - 4 * 3 * 7 = 0) ∧ ((b2 + 6)^2 - 4 * 3 * 7 = 0) ∧ 
   b1 + b2 = -12)  :=
by
  sorry

end sum_of_b_for_unique_solution_l446_446532


namespace abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l446_446812

theorem abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one 
  (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 1 / a + 1 / b + 1 / c) : 
  (a = 1) ∨ (b = 1) ∨ (c = 1) :=
by
  sorry

end abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l446_446812


namespace conditional_probability_l446_446733

-- Define the events A and B as abstract entities
variable {Ω : Type} (A B : Ω → Prop)

-- Define the probabilities P of the events
variables (prob : MeasureTheory.ProbabilityMeasure Ω)

-- Conditions of the problem
axiom hPA : prob {ω | A ω} = 2 / 15
axiom hPAB : prob {ω | A ω ∧ B ω} = 1 / 10

-- The conditional probability we need to prove
theorem conditional_probability :
  prob.cond {ω | A ω} {ω | B ω} = 3 / 4 :=
by
  -- The proof is omitted, only the statement is required
  sorry

end conditional_probability_l446_446733


namespace sin_B_correct_l446_446007

-- Define basic conditions of the right triangle
variables (B C D E : Type)
variables (BC BD CD DE : ℝ)

-- Assuming some initial values and conditions
axiom BC_value : BC = 13
axiom BD_value : BD = 5
axiom DE_value : DE = 2
axiom right_triangle_BCD : ∀ (BCD : Type), ∃ D : Type, ∠ BCD D = 90

-- Using the Pythagorean theorem to define CD
noncomputable def CD := real.sqrt (BC^2 - BD^2)

-- Calculate the sine of angle B
def sin_B := CD / BC

-- Main theorem to be proven
theorem sin_B_correct : sin_B = 12 / 13 := 
  sorry

end sin_B_correct_l446_446007


namespace common_ratio_q_compute_T_l446_446681

noncomputable def S (n : ℕ) (q : ℝ) : ℝ := 
  if q = 1 then n 
  else (1 - q ^ n) / (1 - q)

def a_n (n : ℕ) (q : ℝ) : ℝ := 
  if n = 1 then 1 
  else q^(n-1)

def b_n (n : ℕ) (q : ℝ) : ℝ := 
  a_n n q - 15

def T (n : ℕ) (q : ℝ) : ℝ := 
  (List.sum $ List.map (λ n, abs (b_n n q)) [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

theorem common_ratio_q (q : ℝ) (S2 S4 S6 : ℝ) (h1 : S 2 q = S2)
  (h2 : S 4 q = S4) (h3 : S 6 q = S6) :
  S2 + 4 * S4 = S6 → q = 2 :=
  sorry

theorem compute_T (q : ℝ) (h : q = 2) : T 10 q = 963 :=
  sorry

end common_ratio_q_compute_T_l446_446681


namespace bird_family_problem_l446_446129

def initial_bird_families (f s i : Nat) : Prop :=
  i = f + s

theorem bird_family_problem : initial_bird_families 32 35 67 :=
by
  -- Proof would go here
  sorry

end bird_family_problem_l446_446129


namespace sqrt_abc_sum_eq_357_l446_446773

variables (a b c : ℝ)

-- Assuming the conditions given in the problem
axiom h1 : b + c = 20
axiom h2 : c + a = 22
axiom h3 : a + b = 24

-- The goal is to prove the specified mathematical equivalence
theorem sqrt_abc_sum_eq_357 : sqrt (a * b * c * (a + b + c)) = 357 :=
by sorry

end sqrt_abc_sum_eq_357_l446_446773


namespace magnitude_two_a_sub_b_l446_446704

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (2, 2, 1)
def b : ℝ × ℝ × ℝ := (3, 5, 3)

-- Define the magnitude function
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Define the vector operation 2a - b
def two_a_sub_b : ℝ × ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2, 2 * a.3 - b.3)

-- State the theorem
theorem magnitude_two_a_sub_b : magnitude two_a_sub_b = real.sqrt 3 := by
  sorry

end magnitude_two_a_sub_b_l446_446704


namespace existThreeConfettiWithSmallArea_l446_446547

variable (table_length : ℝ := 2) (table_width : ℝ := 1) (num_confetti : ℝ := 500)
variable (max_tri_area : ℝ := 50 / 10000)  -- cm^2 to m^2

-- Assume a function that places the confetti on the table
noncomputable def placeConfetti (table_length : ℝ) (table_width : ℝ) (num_confetti : ℝ) : List (ℝ × ℝ) := sorry

-- Define the area function for a triangle given three points on a plane
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1 / 2) * |(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))|

-- Main theorem to be proved
theorem existThreeConfettiWithSmallArea :
  ∃ (p1 p2 p3 : ℝ × ℝ) in placeConfetti table_length table_width num_confetti, triangleArea p1 p2 p3 < max_tri_area := sorry

end existThreeConfettiWithSmallArea_l446_446547


namespace three_digit_number_108_l446_446285

theorem three_digit_number_108 (a b c : ℕ) (ha : a ≠ 0) (h₀ : a < 10) (h₁ : b < 10) (h₂ : c < 10) (h₃: 100*a + 10*b + c = 12*(a + b + c)) : 
  100*a + 10*b + c = 108 := 
by 
  sorry

end three_digit_number_108_l446_446285


namespace dispersion_measured_by_std_dev_and_range_l446_446177

variables {α : Type*} [linear_order α] (x : list α)

def standard_deviation (l : list ℝ) : ℝ := sorry -- definition of standard_deviation
def median (l : list α) : α := sorry -- definition of median
def range (l : list ℝ) : ℝ := sorry -- definition of range
def mean (l : list ℝ) : ℝ := sorry -- definition of mean

theorem dispersion_measured_by_std_dev_and_range :
  (standard_deviation (map (λ x, (x : ℝ)) (x : list α)) > 0 ∨ range (map (λ x, (x : ℝ)) (x : list α)) > 0) →
  (∀ x, x ∈ [standard_deviation (map (λ x, (x : ℝ)) (x : list α)), range (map (λ x, (x : ℝ)) (x : list α))]) :=
begin
  sorry
end

end dispersion_measured_by_std_dev_and_range_l446_446177


namespace real_roots_range_of_m_l446_446648

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem real_roots_range_of_m :
  (∃ x : ℝ, x^2 + 4 * m * x + 4 * m^2 + 2 * m + 3 = 0) ∨ 
  (∃ x : ℝ, x^2 + (2 * m + 1) * x + m^2 = 0) ↔ 
  m ≤ -3 / 2 ∨ m ≥ -1 / 4 :=
by
  sorry

end real_roots_range_of_m_l446_446648


namespace max_price_per_unit_min_sales_volume_l446_446477

-- Part (1): Maximum price per unit ensuring revenue is not less than original.
theorem max_price_per_unit
  (initial_price : ℝ) (initial_sales : ℝ) (decrease_rate : ℝ) (original_revenue : ℝ) (t : ℝ) :
  initial_price = 25 ∧ initial_sales = 80000 ∧ decrease_rate = 2000 ∧ original_revenue = 2000000 →
  (80000 - decrease_rate * (t - 25)) * t ≥ 2000000 →
  t ≤ 40 :=
by
  intros h₁ h₂
  sorry

-- Part (2): Minimum annual sales volume ensuring revenue + investment is not less than original revenue + total investment.
theorem min_sales_volume
  (original_revenue : ℝ) (fixed_costs : ℝ) (variable_cost_rate : ℝ) (initial_investment : ℝ → ℝ)
  (x : ℝ) (a : ℝ) :
  original_revenue = 2000000 ∧ fixed_costs = 50000000 ∧ variable_cost_rate = 0.2 ∧ (initial_investment = λ x, (x^2 - 600)/6) →
  x > 25 →
  a * x ≥ original_revenue + fixed_costs + (initial_investment x) + (variable_cost_rate * x) →
  a ≥ 10.2 ∧ x = 30 :=
by
  intros h₁ h₂ h₃
  sorry

end max_price_per_unit_min_sales_volume_l446_446477


namespace rotate_Q_coords_l446_446540

-- Given points
def P : ℝ × ℝ := (0, 0)
def R : ℝ × ℝ := (8, 0)

-- Given angles and conditions
def angle_QRP := 90
def angle_QPR := 45

-- Prove the coordinates of Q' after rotation
theorem rotate_Q_coords :
  ∃ Q : ℝ × ℝ, Q.1 > 0 ∧ Q.2 > 0 ∧
    angle_QRP = 90 ∧ angle_QPR = 45 ∧
    let Q' := (Q.1 * (-1 / 2) - Q.2 * (√3 / 2), Q.1 * (√3 / 2) + Q.2 * (-1 / 2)) in
    Q' = (-4 - 4 * √3, 4 * √3 - 4) :=
sorry

end rotate_Q_coords_l446_446540


namespace sum_g_eq_half_l446_446429

noncomputable def g (n : ℕ) : ℝ := ∑' k, if h : k ≥ 3 then (1 / (k : ℝ) ^ n) else 0

theorem sum_g_eq_half : (∑' n, if h : n ≥ 3 then g n else 0) = 1 / 2 := by
  sorry

end sum_g_eq_half_l446_446429


namespace measure_of_angle_l446_446123

theorem measure_of_angle (x : ℝ) (h1 : 3 * x - 10 = 180 - x) : x = 47.5 :=
by
  have h2 : 4 * x = 190 := by sorry  -- Simplify and rearrange h1 to get this equation
  exact eq_div_of_mul_eq h2 (ne_of_lt (by linarith : 4 > 0)).symm

end measure_of_angle_l446_446123


namespace dispersion_measured_by_std_dev_and_range_l446_446180

variables {α : Type*} [linear_order α] (x : list α)

def standard_deviation (l : list ℝ) : ℝ := sorry -- definition of standard_deviation
def median (l : list α) : α := sorry -- definition of median
def range (l : list ℝ) : ℝ := sorry -- definition of range
def mean (l : list ℝ) : ℝ := sorry -- definition of mean

theorem dispersion_measured_by_std_dev_and_range :
  (standard_deviation (map (λ x, (x : ℝ)) (x : list α)) > 0 ∨ range (map (λ x, (x : ℝ)) (x : list α)) > 0) →
  (∀ x, x ∈ [standard_deviation (map (λ x, (x : ℝ)) (x : list α)), range (map (λ x, (x : ℝ)) (x : list α))]) :=
begin
  sorry
end

end dispersion_measured_by_std_dev_and_range_l446_446180


namespace original_height_l446_446198

theorem original_height (total_travel : ℝ) (h : ℝ) (half: h/2 = (1/2 * h)): 
  (total_travel = h + 2 * (h / 2) + 2 * (h / 4)) → total_travel = 260 → h = 104 :=
by
  intro travel_eq
  intro travel_value
  sorry

end original_height_l446_446198


namespace range_of_h_l446_446296

noncomputable def h : ℝ → ℝ
| x => if x = -7 then 0 else 2 * (x - 3)

theorem range_of_h :
  (Set.range h) = Set.univ \ {-20} :=
sorry

end range_of_h_l446_446296


namespace angle_KNL_90_degrees_l446_446456

theorem angle_KNL_90_degrees
    (A B C K L M N : EuclideanGeometry.Point)
    (hAB_eq : distance A B = distance B C)
    (hK_on_AB : EuclideanGeometry.is_on_segment A B K)
    (hL_on_BC : EuclideanGeometry.is_on_segment B C L)
    (hAK_LC_KL : distance A K + distance L C = distance K L)
    (hM_midpoint_KL : EuclideanGeometry.Midpoint M K L)
    (hN_on_AC : EuclideanGeometry.is_on_segment A C N)
    (hMN_parallel_BC : EuclideanGeometry.is_parallel (EuclideanGeometry.Line.mk M N) (EuclideanGeometry.Line.mk B C)) :
    EuclideanGeometry.angle K N L = 90 := 
sorry

end angle_KNL_90_degrees_l446_446456


namespace find_theta_l446_446652

noncomputable def point_A (θ : ℝ) : ℝ × ℝ := (-1, Real.cos θ)
noncomputable def point_B (θ : ℝ) : ℝ × ℝ := (Real.sin θ, 1)

def vector_add (v1 v2: ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vector_sub (v1 v2: ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)
def vector_norm (v: ℝ × ℝ) : ℝ := Real.sqrt ((v.1)^2 + (v.2)^2)

theorem find_theta (θ : ℝ) (h : vector_norm (vector_add (point_A θ) (point_B θ)) = vector_norm (vector_sub (point_A θ) (point_B θ))) : 
  θ = Real.pi / 4 :=
by
  sorry

end find_theta_l446_446652


namespace correct_transformation_B_l446_446685

def curve := fun x : ℝ => (Real.sin (2 * x - Real.pi / 3))

def transform_right (f : ℝ → ℝ) (shift : ℝ) := fun x => f (x - shift)

def is_symmetric_about_y_axis (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x = f (-x)

theorem correct_transformation_B :
  is_symmetric_about_y_axis (transform_right curve (Real.pi / 12)) :=
by
  sorry

end correct_transformation_B_l446_446685


namespace decreasing_interval_l446_446141

theorem decreasing_interval (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 - 2 * x) :
  {x | deriv f x < 0} = {x | x < 1} :=
by
  sorry

end decreasing_interval_l446_446141


namespace divide_99_l446_446270

def reverse_digits (n : ℕ) : ℕ :=
  let s := toString n
  s.data.reverse.asString.toNat

theorem divide_99 (n : ℕ) (h : n > 0) : 99 ∣ n^4 - (reverse_digits n)^4 :=
  sorry

end divide_99_l446_446270


namespace range_of_f_l446_446856

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2 * x)

theorem range_of_f : Set.Ioo 0 2 ∪ {2} = {y : ℝ | ∃ x : ℝ, f(x) = y} :=
by
  sorry

end range_of_f_l446_446856


namespace intersection_M_N_l446_446695

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def N : Set ℝ := { y | 0 < y }

theorem intersection_M_N : (M ∩ N) = { z | 0 < z ∧ z ≤ 2 } :=
by
  -- proof to be completed
  sorry

end intersection_M_N_l446_446695


namespace harriet_return_speed_l446_446939

-- Definitions using given conditions
def v1 : ℝ := 105  -- speed from A-ville to B-town in km/h
def T : ℝ := 5  -- total trip time in hours
def t1 : ℝ := 174 / 60  -- time from A-ville to B-town in hours
def distance : ℝ := v1 * t1  -- distance between A-ville and B-town

-- Required to be proved: speed back from B-town to A-ville, which is "145 km/h"
theorem harriet_return_speed :
  let t2 := T - t1 in
  let v2 := distance / t2 in
  v2 = 145 := 
sorry

end harriet_return_speed_l446_446939


namespace umbrella_cost_l446_446763

theorem umbrella_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) (h1 : house_umbrellas = 2) (h2 : car_umbrellas = 1) (h3 : cost_per_umbrella = 8) : 
  (house_umbrellas + car_umbrellas) * cost_per_umbrella = 24 := 
by
  sorry

end umbrella_cost_l446_446763


namespace product_remainder_mod_7_l446_446870

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l446_446870


namespace johnsYearlyHaircutExpenditure_l446_446034

-- Definitions based on conditions
def hairGrowthRate : ℝ := 1.5 -- inches per month
def hairCutLength : ℝ := 9 -- inches
def hairAfterCut : ℝ := 6 -- inches
def monthsBetweenCuts := (hairCutLength - hairAfterCut) / hairGrowthRate
def haircutCost : ℝ := 45 -- dollars
def tipPercent : ℝ := 0.20
def tipsPerHaircut := tipPercent * haircutCost

-- Number of haircuts in a year
def numHaircutsPerYear := 12 / monthsBetweenCuts

-- Total yearly expenditure
def yearlyHaircutExpenditure := numHaircutsPerYear * (haircutCost + tipsPerHaircut)

theorem johnsYearlyHaircutExpenditure : yearlyHaircutExpenditure = 324 := 
by
  sorry

end johnsYearlyHaircutExpenditure_l446_446034


namespace sequence_is_periodic_l446_446883

open Nat

def is_periodic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ p > 0, ∀ i, a (i + p) = a i

theorem sequence_is_periodic (a : ℕ → ℕ)
  (h1 : ∀ n, a n < 1988)
  (h2 : ∀ m n, a m + a n ∣ a (m + n)) : is_periodic_sequence a :=
by
  sorry

end sequence_is_periodic_l446_446883


namespace num_of_convex_numbers_l446_446583

-- Define what it means for a sequence of digits to form a "convex number"
def is_convex_number (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  (digits.length = 9) ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 9 → digits.nth i < digits.nth j) ∧
  (∀ i j, 9 ≤ i ∧ i < j ∧ j ≤ 17 → digits.nth i > digits.nth j) ∧
  (1 ≤ digits.nth 9)

-- Define the set of all 9-digit numbers using the digits 1-9
def nine_digit_numbers :=
  { n : ℕ | let digits := n.digits 10 in (digits.length = 9) ∧ ( ∀d, d ∈ digits → d ∈ finset.range 1 10 )}

-- Prove the question
theorem num_of_convex_numbers : 
  (∃ n, n ∈ nine_digit_numbers ∧ is_convex_number n) = 254 := by
  sorry

end num_of_convex_numbers_l446_446583


namespace avg_speed_first_part_l446_446966

theorem avg_speed_first_part (v : ℝ) (total_distance : ℝ) (distance_first_part : ℝ) (distance_second_part : ℝ) (speed_second_part : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250) 
  (h2 : distance_first_part = 148)
  (h3 : distance_second_part = 102) 
  (h4 : speed_second_part = 60) 
  (h5 : total_time = 5.4) 
  (h6 : distance_first_part / v + distance_second_part / speed_second_part = total_time) : 
  v = 40 := 
begin
  sorry
end

end avg_speed_first_part_l446_446966


namespace quadratic_complete_square_l446_446117

open Real

theorem quadratic_complete_square (d e : ℝ) :
  (∀ x, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  intros h
  have h_eq := h 12
  sorry

end quadratic_complete_square_l446_446117


namespace AM_GM_inequality_l446_446057

theorem AM_GM_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a / b + b / c + c / d + d / a) ≥ 4 := 
sorry

end AM_GM_inequality_l446_446057


namespace possible_values_of_a_l446_446319

theorem possible_values_of_a :
  (Π (f: ℝ → ℝ), (∀ x, f x = f (-x)) →  (∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y →  f y ≤ f x) → 
  ∀ a, f (3 * a) ≥ f (2 * a - 1) → -1 ≤ a ∧ a ≤ 1/5) :=
begin
  intros f h_even h_decreasing a h_condition,
  sorry
end

end possible_values_of_a_l446_446319


namespace log_range_sum_l446_446277

-- Define the logarithms we are considering
def log1000 := log 1000
def log3799 := log 3799
def log10000 := log 10000

-- State the theorem
theorem log_range_sum : (3 : ℤ) < log10 3799 ∧ log10 3799 < (4 : ℤ) → (3 : ℤ) + (4 : ℤ) = 7 := 
by {
  intro h,
  exact rfl,
}

end log_range_sum_l446_446277


namespace smallest_solution_l446_446640

theorem smallest_solution (x : ℝ) (hx : floor x = 7 + 75 * (x - floor x)) : x = 7 :=
sorry

end smallest_solution_l446_446640


namespace log3_20_approx_l446_446538

noncomputable def log10_approx (x : ℝ) := 
  if x = 2 then 0.301
  else if x = 3 then 0.477
  else 0 -- Assumption for undefined cases to allow Lean to compile

theorem log3_20_approx : 
  abs ((log10_approx 20 / log10_approx 3) - 2.7) < 0.1 := 
by 
  unfold log10_approx
  -- Manually verified values
  have h1 : log10_approx 20 = 1.301 := by unfold log10_approx; sorry
  have h2 : log10_approx 3 = 0.477 := by unfold log10_approx; sorry
  rw [h1, h2]
  -- Approximate division manually calculated
  have h3 : 1.301 / 0.477 ≈ 2.72767 := by sorry
  have h4 : abs (2.72767 - 2.7) < 0.1 := by sorry
  assumption

end log3_20_approx_l446_446538


namespace max_value_expression_max_value_at_sqrt2_l446_446431

noncomputable def expression (x : ℝ) : ℝ := (x^2 + 4 - real.sqrt (x^4 + 16)) / x

theorem max_value_expression : 
  ∀ x : ℝ, 0 < x → expression x ≤ 2 * real.sqrt 2 - 2 :=
sorry

theorem max_value_at_sqrt2 :
  expression (real.sqrt 2) = 2 * real.sqrt 2 - 2 :=
sorry

end max_value_expression_max_value_at_sqrt2_l446_446431


namespace perpendicular_line_equation_l446_446290

theorem perpendicular_line_equation 
  (p : ℝ × ℝ)
  (L1 : ℝ → ℝ → Prop)
  (L2 : ℝ → ℝ → ℝ → Prop) 
  (hx : p = (1, -1)) 
  (hL1 : ∀ x y, L1 x y ↔ 3 * x - 2 * y = 0) 
  (hL2 : ∀ x y m, L2 x y m ↔ 2 * x + 3 * y + m = 0) :
  ∃ m : ℝ, L2 (p.1) (p.2) m ∧ 2 * p.1 + 3 * p.2 + m = 0 :=
by
  sorry

end perpendicular_line_equation_l446_446290


namespace transformed_roots_l446_446844

theorem transformed_roots (b c : ℝ) (h₁ : (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c).roots = {2, -3}) :
  (Polynomial.C 1 * (Polynomial.X - Polynomial.C 4)^2 + Polynomial.C b * (Polynomial.X - Polynomial.C 4) + Polynomial.C c).roots = {1, 6} :=
by
  sorry

end transformed_roots_l446_446844


namespace seq_general_formula_l446_446393

open Nat

def seq (a : ℕ+ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, a (n + 1) = 2 * a n / (2 + a n)

theorem seq_general_formula (a : ℕ+ → ℝ) (h : seq a) :
  ∀ n : ℕ+, a n = 2 / (n + 1) :=
by
  sorry

end seq_general_formula_l446_446393


namespace median_length_l446_446801

theorem median_length
  (DP EQ : ℝ)
  (perpendicular : ∃ D E F P Q : Type, ∃ DEF : triangle D E F, ∃ DP EQ : segment D E P Q DEF, DP ⟂ EQ)
  (DP_eq : DP = 15)
  (EQ_eq : EQ = 20) :
  DF = (20 * real.sqrt 13) / 3 :=
by
  sorry

end median_length_l446_446801


namespace frog_eggs_ratio_l446_446983

theorem frog_eggs_ratio
    (first_day : ℕ)
    (second_day : ℕ)
    (third_day : ℕ)
    (total_eggs : ℕ)
    (h1 : first_day = 50)
    (h2 : second_day = first_day * 2)
    (h3 : third_day = second_day + 20)
    (h4 : total_eggs = 810) :
    (total_eggs - (first_day + second_day + third_day)) / (first_day + second_day + third_day) = 2 :=
by
    sorry

end frog_eggs_ratio_l446_446983


namespace divide_subtract_result_l446_446472

theorem divide_subtract_result (x : ℕ) (h : (x - 26) / 2 = 37) : 48 - (x / 4) = 23 := 
by
  sorry

end divide_subtract_result_l446_446472


namespace action_figures_more_than_books_proof_l446_446408

-- Definitions for the conditions
def books := 3
def action_figures_initial := 4
def action_figures_added := 2

-- Definition for the total action figures
def action_figures_total := action_figures_initial + action_figures_added

-- Definition for the number difference
def action_figures_more_than_books := action_figures_total - books

-- Proof statement
theorem action_figures_more_than_books_proof : action_figures_more_than_books = 3 :=
by
  sorry

end action_figures_more_than_books_proof_l446_446408


namespace find_a_l446_446654

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 0 then log 10 x else x + ∫ t in 0..a, 3 * t^2

theorem find_a (a : ℝ) (h : f (f 1 a) a = 1) : a = 1 :=
by
  sorry

end find_a_l446_446654


namespace lyra_beef_purchase_l446_446068

theorem lyra_beef_purchase :
  ∃ (P : ℕ), ∀ (B C_f C_b R : ℕ),
    B = 80 ∧ C_f = 12 ∧ C_b = 3 ∧ R = 53 →
    B - R - C_f = P * C_b ∧ P = 5 :=
begin
  sorry
end

end lyra_beef_purchase_l446_446068


namespace line_through_A_B_with_y_intercept_l446_446633

noncomputable def midpoint (A B : (ℝ × ℝ)) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem line_through_A_B_with_y_intercept 
  (A B : (ℝ × ℝ)) (y_intercept : ℝ) 
  (mid : (ℝ × ℝ)) (m : ℝ) : 
  mid = midpoint A B → 
  mid = (4, 0) → 
  y_intercept = -3 → 
  m = 3 / 4 → 
  (3 : ℝ) * (x : ℝ) - (4 : ℝ) * (y : ℝ) - 12 = 0 :=
by
  sorry

end line_through_A_B_with_y_intercept_l446_446633


namespace monotonically_increasing_interval_l446_446852

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x + 3)

theorem monotonically_increasing_interval :
  ∀ x, f(x) ∈ Ioi (1 : ℝ) ↔ x ∈ Ioo (-∞) 1 := 
sorry

end monotonically_increasing_interval_l446_446852


namespace equivalence_condition_l446_446046

universe u

variables {U : Type u} (A B : Set U)

theorem equivalence_condition :
  (∃ (C : Set U), A ⊆ C ∧ B ⊆ Cᶜ) ↔ (A ∩ B = ∅) :=
sorry

end equivalence_condition_l446_446046


namespace cumulative_distribution_X_maximized_expected_score_l446_446977

noncomputable def distribution_X (p_A : ℝ) (p_B : ℝ) : (ℝ × ℝ × ℝ) :=
(1 - p_A, p_A * (1 - p_B), p_A * p_B)

def expected_score (p_A : ℝ) (p_B : ℝ) (s_A : ℝ) (s_B : ℝ) : ℝ :=
0 * (1 - p_A) + s_A * (p_A * (1 - p_B)) + (s_A + s_B) * (p_A * p_B)

theorem cumulative_distribution_X :
  distribution_X 0.8 0.6 = (0.2, 0.32, 0.48) :=
sorry

theorem maximized_expected_score :
  expected_score 0.8 0.6 20 80 < expected_score 0.6 0.8 80 20 :=
sorry

end cumulative_distribution_X_maximized_expected_score_l446_446977


namespace prime_factor_of_sum_of_four_consecutive_integers_l446_446272

theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by 
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_l446_446272


namespace coat_lifetime_15_l446_446765

noncomputable def coat_lifetime : ℕ :=
  let cost_coat_expensive := 300
  let cost_coat_cheap := 120
  let years_cheap := 5
  let year_saving := 120
  let duration_comparison := 30
  let yearly_cost_cheaper := cost_coat_cheap / years_cheap
  let yearly_savings := year_saving / duration_comparison
  let cost_savings := yearly_cost_cheaper * duration_comparison - cost_coat_expensive * duration_comparison / (yearly_savings + (cost_coat_expensive / cost_coat_cheap))
  cost_savings

theorem coat_lifetime_15 : coat_lifetime = 15 := by
  sorry

end coat_lifetime_15_l446_446765


namespace binomial_expansion_constant_fifth_term_l446_446370

theorem binomial_expansion_constant_fifth_term (n : ℕ) :
  let x := 0 in
  let fifth_term := (Nat.choose n 4) * (-2)^4 * x^((n - 3 * 4) / 2) in
  fifth_term = (Nat.choose n 4) * (-2)^4 → n = 12 :=
by
  intro x fifth_term h
  have := congr_arg (fun x => x ^ (2 / (n - 3 * 4))) h
  sorry

end binomial_expansion_constant_fifth_term_l446_446370


namespace dispersion_measures_correct_l446_446189

-- Define a sample data set
variable {x : ℕ → ℝ}
variable {n : ℕ}

-- Definitions of the four statistics
def standard_deviation (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def median (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def range (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def mean (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry

-- Definition of measures_dispersion function
def measures_dispersion (stat : ℕ → ℝ → ℝ) (x : ℕ → ℝ) (n : ℕ) : Prop :=
  sorry -- Define what it means for a statistic to measure dispersion

-- Problem statement in Lean
theorem dispersion_measures_correct :
  measures_dispersion standard_deviation x n ∧
  measures_dispersion range x n ∧
  ¬measures_dispersion median x n ∧
  ¬measures_dispersion mean x n :=
by sorry

end dispersion_measures_correct_l446_446189


namespace sum_of_midpoint_coordinates_l446_446490

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := -1
  let x2 := 11
  let y2 := 21
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 17 := by
  sorry

end sum_of_midpoint_coordinates_l446_446490


namespace find_percentage_error_l446_446597

noncomputable def percentage_error_in_side_length (E : ℝ) : Prop :=
  let S := 1.0 in  -- Assume S is some positive length (WLOG, take S = 1 for simplicity)
  let S' := S * (1 + E / 100) in
  (S'^2 - S^2) / S^2 * 100 = 8.16

theorem find_percentage_error :
  percentage_error_in_side_length 4 :=
sorry

end find_percentage_error_l446_446597


namespace statistics_measuring_dispersion_l446_446171

-- Definition of standard deviation
def standard_deviation (X : List ℝ) : ℝ :=
  let mean := (X.sum) / (X.length)
  (X.map (λ x => (x - mean) ^ 2)).sum / X.length

-- Definition of range
def range (X : List ℝ) : ℝ :=
  X.maximum.get - X.minimum.get

-- Definition of median
noncomputable def median (X : List ℝ) : ℝ :=
  let sorted := List.sorted X
  if sorted.length % 2 == 1 then sorted.get (sorted.length / 2)
  else (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2

-- Definition of mean
def mean (X : List ℝ) : ℝ :=
  X.sum / X.length

-- The proof statement
theorem statistics_measuring_dispersion (X : List ℝ) :
  (standard_deviation X ≠ 0 ∧ range X ≠ 0) ∧
  (∀ x : ℝ, x ∈ X ↔ (median X = x ∨ mean X = x)) → true :=
  sorry

end statistics_measuring_dispersion_l446_446171


namespace import_tax_calculation_l446_446161

def import_tax_rate : ℝ := 0.07
def excess_value_threshold : ℝ := 1000
def total_value_item : ℝ := 2610
def correct_import_tax : ℝ := 112.7

theorem import_tax_calculation :
  (total_value_item - excess_value_threshold) * import_tax_rate = correct_import_tax :=
by
  sorry

end import_tax_calculation_l446_446161


namespace center_of_symmetry_min_value_of_a_l446_446338

def f (x : ℝ) : ℝ := cos(x) ^ 2 - (sqrt 3 / 2) * sin(2 * x) - 1 / 2

theorem center_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f(x) = f(x + k * π / 2) := sorry

theorem min_value_of_a 
  (A : ℝ) (b c : ℝ)
  (h1 : f(A) + 1 = 0)
  (h2 : b + c = 2) :
  ∃ a : ℝ, a = 1 := sorry

end center_of_symmetry_min_value_of_a_l446_446338


namespace necessary_not_sufficient_condition_l446_446228

theorem necessary_not_sufficient_condition (x : ℝ) :
  ((-6 ≤ x ∧ x ≤ 3) → (-5 ≤ x ∧ x ≤ 3)) ∧
  (¬ ((-5 ≤ x ∧ x ≤ 3) → (-6 ≤ x ∧ x ≤ 3))) :=
by
  -- Need proof steps here
  sorry

end necessary_not_sufficient_condition_l446_446228


namespace face_opposite_A_is_F_l446_446617

structure Cube where
  adjacency : String → String → Prop
  exists_face : ∃ a b c d e f : String, True

variable 
  (C : Cube)
  (adjA_B : C.adjacency "A" "B")
  (adjA_C : C.adjacency "A" "C")
  (adjB_D : C.adjacency "B" "D")

theorem face_opposite_A_is_F : 
  ∃ f : String, f = "F" ∧ ∀ g : String, (C.adjacency "A" g → g ≠ "F") :=
by 
  sorry

end face_opposite_A_is_F_l446_446617


namespace sugar_percentage_after_additions_l446_446570

noncomputable def initial_solution_volume : ℝ := 440
noncomputable def initial_water_percentage : ℝ := 0.88
noncomputable def initial_kola_percentage : ℝ := 0.08
noncomputable def initial_sugar_percentage : ℝ := 1 - initial_water_percentage - initial_kola_percentage
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8

noncomputable def initial_sugar_amount := initial_sugar_percentage * initial_solution_volume
noncomputable def new_sugar_amount := initial_sugar_amount + sugar_added
noncomputable def new_solution_volume := initial_solution_volume + sugar_added + water_added + kola_added

noncomputable def final_sugar_percentage := (new_sugar_amount / new_solution_volume) * 100

theorem sugar_percentage_after_additions :
    final_sugar_percentage = 4.52 :=
by
    sorry

end sugar_percentage_after_additions_l446_446570


namespace probability_negative_product_of_random_selection_l446_446307

open_locale classical

-- Definitions of the conditions
def numbers : set ℤ := {-1, -2, 3, 4}
def pairs (s : set ℤ) : set (ℤ × ℤ) := {p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 ≠ p.2}
def negative_product (p : ℤ × ℤ) : Prop := p.1 * p.2 < 0

-- Statement of the problem
theorem probability_negative_product_of_random_selection : 
  (↑(finite.to_finset (pairs numbers)).filter negative_product).card / 
  ((finite.to_finset (pairs numbers)).card : ℝ) = 2 / 3 :=
sorry

end probability_negative_product_of_random_selection_l446_446307


namespace function_decreasing_on_interval_l446_446278

noncomputable def f (x : ℝ) : ℝ := log 0.5 (x^2 - 4)

theorem function_decreasing_on_interval :
  ∀ x y : ℝ, 2 < x ∧ 2 < y ∧ x < y → f(x) > f(y) :=
begin
  sorry
end

end function_decreasing_on_interval_l446_446278


namespace volume_in_cubic_meters_l446_446849

noncomputable def mass_condition : ℝ := 100 -- mass in kg
noncomputable def volume_per_gram : ℝ := 10 -- volume in cubic centimeters per gram
noncomputable def volume_per_kg : ℝ := volume_per_gram * 1000 -- volume in cubic centimeters per kg
noncomputable def mass_in_kg : ℝ := mass_condition

theorem volume_in_cubic_meters (h : mass_in_kg = 100)
    (v_per_kg : volume_per_kg = volume_per_gram * 1000) :
  (mass_in_kg * volume_per_kg) / 1000000 = 1 := by
  sorry

end volume_in_cubic_meters_l446_446849


namespace log2_a_div_b_squared_l446_446772

variable (a b : ℝ)
variable (ha_ne_1 : a ≠ 1) (hb_ne_1 : b ≠ 1)
variable (ha_pos : 0 < a) (hb_pos : 0 < b)
variable (h1 : 2 ^ (Real.log 32 / Real.log b) = a)
variable (h2 : a * b = 128)

theorem log2_a_div_b_squared :
  (Real.log ((a / b) : ℝ) / Real.log 2) ^ 2 = 29 + (49 / 4) :=
sorry

end log2_a_div_b_squared_l446_446772


namespace side_length_of_square_l446_446305

-- Define the problem conditions and question
constant ungrazed_area : ℝ := 851.7546894755278
constant π : ℝ := Real.pi

-- Hypothesis based on the problem statement
axiom h : ∀ (a : ℝ), a^2 * (1 + π) = ungrazed_area

-- The theorem we need to prove
theorem side_length_of_square (a : ℝ) (h : a^2 * (1 + π) = ungrazed_area) : a ≈ 14.343 :=
by
  sorry

end side_length_of_square_l446_446305


namespace find_speed_of_A_l446_446535

noncomputable def speed_of_A_is_7_5 (a : ℝ) : Prop :=
  -- Conditions
  ∃ (b : ℝ), b = a + 5 ∧ 
  (60 / a = 100 / b) → 
  -- Conclusion
  a = 7.5

-- Statement in Lean 4
theorem find_speed_of_A (a : ℝ) (h : speed_of_A_is_7_5 a) : a = 7.5 :=
  sorry

end find_speed_of_A_l446_446535


namespace james_car_new_speed_l446_446403

-- Define the conditions and the statement to prove
variable (original_speed supercharge_increase weight_reduction : ℝ)
variable (original_speed_gt_zero : 0 < original_speed)

theorem james_car_new_speed :
  original_speed = 150 →
  supercharge_increase = 0.3 →
  weight_reduction = 10 →
  original_speed * (1 + supercharge_increase) + weight_reduction = 205 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- Calculate the speed after supercharging
  have supercharged_speed : ℝ := 150 * (1 + 0.3)
  calc
    150 * (1 + 0.3) + 10 = 195 + 10 : by norm_num
                       ... = 205 : by norm_num
  sorry

end james_car_new_speed_l446_446403


namespace range_of_x_l446_446334

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

theorem range_of_x (x : ℝ) (h1 : f (x^2 - 4) < 2) : x ∈ Set.union (Set.Ioo (-Real.sqrt 5) (-2)) (Set.Ioo 2 (Real.sqrt 5)) :=
by
  have domain : ∀ x, x > 0 → f x = Real.log x + 2^x + x - 1 := by sorry
  have monotonic_increasing : ∀ x > 0, f' x > 0 := by
    intros
    have f_prime := (1 / x) + (2^x * Real.log 2) + 1
    show f_prime > 0
    sorry
  have f1_eq_2 : f 1 = 2 := by
    calc
      f 1 = Real.log 1 + 2^1 + 1 - 1 := by
        rw [f]
        ring
      _     = 2 := by
        norm_num
  show x ∈ (Set.union (Set.Ioo (-Real.sqrt 5) (-2)) (Set.Ioo 2 (Real.sqrt 5)))
    sorry

end range_of_x_l446_446334


namespace football_defense_stats_l446_446744

/-- Given:
1. Team 1 has an average of 1.5 goals conceded per match.
2. Team 1 has a standard deviation of 1.1 for the total number of goals conceded throughout the year.
3. Team 2 has an average of 2.1 goals conceded per match.
4. Team 2 has a standard deviation of 0.4 for the total number of goals conceded throughout the year.

Prove:
There are exactly 3 correct statements out of the 4 listed statements. -/
theorem football_defense_stats
  (avg_goals_team1 : ℝ := 1.5)
  (std_dev_team1 : ℝ := 1.1)
  (avg_goals_team2 : ℝ := 2.1)
  (std_dev_team2 : ℝ := 0.4) :
  ∃ correct_statements : ℕ, correct_statements = 3 := 
by
  sorry

end football_defense_stats_l446_446744


namespace wall_thickness_proof_l446_446896

-- Definitions from the conditions
def wall_length : ℝ := 8  -- in meters
def wall_height : ℝ := 1  -- in meters
def wall_area_cm2 : ℝ := (wall_length * 100) * (wall_height * 100)  -- converted to cm²

def brick_length : ℝ := 25 -- in cm
def brick_width : ℝ := 11  -- in cm
def brick_height : ℝ := 6  -- in cm
def brick_volume : ℝ := brick_length * brick_width * brick_height -- in cm³

def num_bricks : ℝ := 242.42424242424244
def total_brick_volume : ℝ := brick_volume * num_bricks -- in cm³

-- Thickness calculation
def wall_thickness : ℝ := total_brick_volume / wall_area_cm2 -- in cm

-- Proof statement
theorem wall_thickness_proof : wall_thickness = 5 := by
  -- Pre-filled proof would go here
  sorry

end wall_thickness_proof_l446_446896


namespace find_n_of_divisors_product_l446_446506

theorem find_n_of_divisors_product (n : ℕ) (h1 : 0 < n)
  (h2 : ∏ k in (finset.filter (λ d : ℕ, d ∣ n) (finset.range (n + 1))), d = 1024) :
  n = 16 :=
sorry

end find_n_of_divisors_product_l446_446506


namespace probability_composite_first_50_l446_446200

open Nat

def is_composite (n : ℕ) : Prop :=
  ¬ Prime n ∧ n ≠ 1

lemma first_50_composites_count : (Finset.filter is_composite (Finset.range 51)).card = 34 := 
sorry

theorem probability_composite_first_50 : 
  ((Finset.filter is_composite (Finset.range 51)).card : ℚ) / 50 = 17 / 25 :=
by 
  rw first_50_composites_count
  norm_cast
  exact (by norm_num : (34 : ℚ) / 50 = 17 / 25)

end probability_composite_first_50_l446_446200


namespace smallest_scrumptious_l446_446818

def is_scrumptious (B : ℤ) : Prop :=
  ∃ (n : ℤ), n > 0 ∧ B - n + 1 + B + n - 1 = 2030

theorem smallest_scrumptious :
  ∀ B, is_scrumptious B → B ≥ -2029 :=
begin
  sorry
end

end smallest_scrumptious_l446_446818


namespace slope_of_l3_l446_446067

/-- Given conditions about lines and points on a Cartesian plane, prove the slope of a specific line. -/
theorem slope_of_l3
    (A : ℝ × ℝ) (A_coords : A = (-1, -2))
    (l1 : ℝ → ℝ → Prop) (l1_eq : ∀ x y : ℝ, l1 x y ↔ 3 * x - 2 * y = 1)
    (l2 : ℝ → ℝ → Prop) (l2_eq : ∀ x y : ℝ, l2 x y ↔ y = 2)
    (B : ℝ × ℝ) (B_coords : B = (5/3, 2))
    (l3 : ℝ → ℝ → Prop)
    (meets_at : ∃ C : ℝ × ℝ, l3 C.1 C.2 ∧ l2 C.1 C.2)
    (area_triangle_ABC : 5) : 
    (∃ C : ℝ × ℝ, 
        C = (25 / 6, 2) ∧ (l3 A.1 A.2 ∧ l3 C.1 C.2) → 
        ∃ m : ℝ, m = (4 + 2) / (25 / 6 + 1) / 2 :=
sorry

end slope_of_l3_l446_446067


namespace percentage_volume_removed_l446_446238

/-- A solid box has dimensions 20 cm by 15 cm by 12 cm. 
A new solid is formed by removing a cube of 4 cm on a side from each of the eight corners. 
We need to prove that the percentage of the original volume removed is approximately 14.22%. -/
theorem percentage_volume_removed :
  let volume_original_box := 20 * 15 * 12
  let volume_one_cube := 4^3
  let total_volume_removed := 8 * volume_one_cube
  let percentage_removed := (total_volume_removed : ℚ) / volume_original_box * 100
  percentage_removed ≈ 14.22 := sorry

end percentage_volume_removed_l446_446238


namespace total_vertical_distance_l446_446578

theorem total_vertical_distance (thickness top_diameter bottom_diameter : ℕ) 
  (h_thickness : thickness = 2) 
  (h_top_diameter : top_diameter = 20) 
  (h_bottom_diameter : bottom_diameter = 4) 
  (h_outside_decrease : ∀ n : ℕ, n > 0 → (top_diameter - 2 * n) > bottom_diameter → (top_diameter - 2 * (n + 1)) = (top_diameter - 2 * n) - 2) :
  (let n := 9 in
   let a := (top_diameter - 4) in
   let d := -2 in
   ∑ i in finset.range n, a + i * d) = 72 := 
by 
  sorry

end total_vertical_distance_l446_446578


namespace asthma_distribution_l446_446920

noncomputable def total_children := 490
noncomputable def boys := 280
noncomputable def general_asthma_ratio := 2 / 7
noncomputable def boys_asthma_ratio := 1 / 9

noncomputable def total_children_with_asthma := general_asthma_ratio * total_children
noncomputable def boys_with_asthma := boys_asthma_ratio * boys
noncomputable def girls_with_asthma := total_children_with_asthma - boys_with_asthma

theorem asthma_distribution
  (h_general_asthma: general_asthma_ratio = 2 / 7)
  (h_total_children: total_children = 490)
  (h_boys: boys = 280)
  (h_boys_asthma: boys_asthma_ratio = 1 / 9):
  boys_with_asthma = 31 ∧ girls_with_asthma = 109 :=
by
  sorry

end asthma_distribution_l446_446920


namespace area_of_awesome_points_set_l446_446418

def triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2

def awesome_points (T : Finset (ℝ × ℝ)) : Finset (ℝ × ℝ) :=
  T.filter (λ P, ∃ A B C D ∈ T, (P.1, P.2) = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) 
                                ∧ (P.1, P.2) = ((B.1 + D.1) / 2, (B.2 + D.2) / 2))

def area (X : Finset (ℝ × ℝ)) : ℝ :=
  let vertices := X.val in
  0.5 * abs ((vertices.get 0 0).1 * ((vertices.get 1 1).2 - (vertices.get 2 2).2) 
           + (vertices.get 1 0).1 * ((vertices.get 2 1).2 - (vertices.get 0 2).2) 
           + (vertices.get 2 0).1 * ((vertices.get 0 1).2 - (vertices.get 1 2).2))

theorem area_of_awesome_points_set :
  ∀ (T : Finset (ℝ × ℝ)),
  triangle 3 4 5 →
  T = {(0, 0), (3, 0), (0, 4)} →
  area (awesome_points T) = 3 / 2 :=
by
  sorry -- Proof is not provided here

end area_of_awesome_points_set_l446_446418


namespace park_area_correct_l446_446499

def rectangular_park_area (w l : ℝ) (hw : l = 3 * w + 15) (hp : 2 * (w + l) = 800) : ℝ :=
  w * l

theorem park_area_correct : ∃ (w l : ℝ), (l = 3 * w + 15) ∧ (2 * (w + l) = 800) ∧ (rectangular_park_area w l (by sorry) (by sorry) = 29234.375) :=
begin
  sorry
end

end park_area_correct_l446_446499


namespace company_A_at_least_two_correct_company_A_higher_chance_l446_446811

-- Defining the condition where Company A can answer 4 out of the 6 questions correctly
def company_A_correct_questions := 4

-- Defining the condition where Company B can answer each question correctly with probability 2/3
def company_B_correct_prob := 2 / 3

-- Proving the probability that Company A answers at least 2 questions correctly
theorem company_A_at_least_two_correct : ProbMassFunction.prob (λ (x : ℕ), (company_A_correct_questions.choose x * (6 - company_A_correct_questions).choose (3 - x)) / 6.choose 3) (λ x, 2 ≤ x) = 4 / 5 := 
sorry

-- Defining expectations and variances for both companies
def expectation_variance_A :=
  let E_X := 2 in
  let D_X := 2/5 in
  (E_X, D_X)

def expectation_variance_B :=
  let E_Y := 2 in
  let D_Y := 2/3 in
  (E_Y, D_Y)

-- Proving that Company A has a higher chance of winning based on lower variance
theorem company_A_higher_chance : ∀ (E_X D_X E_Y D_Y : ℝ), 
  (E_X = 2 ∧ D_X = 2/5) ∧ (E_Y = 2 ∧ D_Y = 2/3) → D_X < D_Y → "Company A wins" := 
sorry

end company_A_at_least_two_correct_company_A_higher_chance_l446_446811


namespace colored_cube_l446_446214

theorem colored_cube (cube : Fin 10 × Fin 10 × Fin 10 → Bool) :
  (∃! i j k, cube ⟨i, j, k⟩ = tt ∧ cube ⟨i, j, k⟩ = ff) → 
  (finset.count
    (λ face : Fin 9 × Fin 10 × Fin 10 ∨ Fin 10 × Fin 9 × Fin 10 ∨ Fin 10 × Fin 10 × Fin 9, 
      ∃ (i j k : fin 10), 
      (face = inl (⟨i, j, k⟩) ∧ cube ⟨i, j, k⟩ = tt ∧ cube ⟨i, j, k.succ⟩ = ff) ∨ 
      (face = inr (⟨i, j, k⟩) ∧ cube ⟨i, j, k⟩ = tt ∧ cube ⟨i, j.succ, k⟩ = ff) ∨ 
      (face = inr (inr (⟨i, j, k⟩)) ∧ cube ⟨i, j, k⟩ = tt ∧ cube ⟨i.succ, j, k⟩ = ff))
    ≥ 100) :=
sorry

end colored_cube_l446_446214


namespace triangle_area_of_perimeter_and_inradius_l446_446199

theorem triangle_area_of_perimeter_and_inradius
  (P : ℝ) (r : ℝ) (s : ℝ) (A : ℝ)
  (hP : P = 32) 
  (hr : r = 2.5) 
  (hs : s = P / 2) 
  (hA : A = r * s) :
  A = 40 := by
  rw [hP, hr, hs, hA]
  norm_num

end triangle_area_of_perimeter_and_inradius_l446_446199


namespace dispersion_statistics_l446_446181

-- Define the variables and possible statistics options
def sample (n : ℕ) := fin n → ℝ

inductive Statistics
| StandardDeviation : Statistics
| Median : Statistics
| Range : Statistics
| Mean : Statistics

-- Define a function that returns if a statistic measures dispersion
def measures_dispersion : Statistics → Prop
| Statistics.StandardDeviation := true
| Statistics.Median := false
| Statistics.Range := true
| Statistics.Mean := false

-- Prove that StandardDeviation and Range measure dispersion
theorem dispersion_statistics (x : sample n) :
  measures_dispersion Statistics.StandardDeviation ∧
  measures_dispersion Statistics.Range :=
by
  split;
  exact trivial

end dispersion_statistics_l446_446181


namespace inequality_AM_GM_l446_446038

theorem inequality_AM_GM
  (a b c : ℝ)
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c)
  (habc : a + b + c = 1) : 
  (a + 2 * a * b + 2 * a * c + b * c) ^ a * 
  (b + 2 * b * c + 2 * b * a + c * a) ^ b * 
  (c + 2 * c * a + 2 * c * b + a * b) ^ c ≤ 1 :=
by
  sorry

end inequality_AM_GM_l446_446038


namespace common_tangent_range_of_a_l446_446360

theorem common_tangent_range_of_a 
  (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ 
    (-a / x1^2 = 2 / x2) ∧ 
    (2 * a / x1 = 2 * (log x2) - 2)) 
  ↔ (-2 / Real.exp 1 ≤ a ∧ a < 0) := by
  sorry

end common_tangent_range_of_a_l446_446360


namespace isosceles_triangle_perimeter_eq_70_l446_446115

-- Define the conditions
def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 60
def isosceles_triangle_base : ℕ := 30

-- Calculate the side of equilateral triangle
def equilateral_triangle_side : ℕ := equilateral_triangle_perimeter / 3

-- Lean 4 statement
theorem isosceles_triangle_perimeter_eq_70 :
  ∃ (a b c : ℕ), is_equilateral_triangle a b c ∧ 
  a + b + c = equilateral_triangle_perimeter →
  (is_isosceles_triangle a a isosceles_triangle_base) →
  a + a + isosceles_triangle_base = 70 :=
by
  sorry -- proof is omitted

end isosceles_triangle_perimeter_eq_70_l446_446115


namespace compound_interest_correct_amount_l446_446119

-- Define constants and conditions
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

def compound_interest (P R T : ℕ) : ℕ := P * ((1 + R / 100) ^ T - 1)

-- Given values and conditions
def P₁ : ℕ := 1750
def R₁ : ℕ := 8
def T₁ : ℕ := 3
def R₂ : ℕ := 10
def T₂ : ℕ := 2

def SI : ℕ := simple_interest P₁ R₁ T₁
def CI : ℕ := 2 * SI

def P₂ : ℕ := 4000

-- The statement to be proven
theorem compound_interest_correct_amount : 
  compound_interest P₂ R₂ T₂ = CI := 
by 
  sorry

end compound_interest_correct_amount_l446_446119


namespace find_angle_find_k_l446_446327

noncomputable def theta : ℝ := 60 * Real.pi / 180

theorem find_angle (a b : EuclideanSpace ℝ (Fin 3)) 
  (ha : ‖a‖ = 3) (hb : ‖b‖ = 5) (hab : ‖a + b‖ = 7) :
  Real.cos (theta) = 1 / 2 :=
by
  sorry

theorem find_k (a b : EuclideanSpace ℝ (Fin 3)) 
  (ha : ‖a‖ = 3) (hb : ‖b‖ = 5) (hab : ‖a + b‖ = 7) 
  (theta_proved : Real.cos theta = 1 / 2) :
  ∃ (k : ℝ), 
    (k * a + b) ⬝ (a - 2 * b) = 0 ∧ 
    k = -(85 : ℝ) / 12 :=
by
  sorry

end find_angle_find_k_l446_446327


namespace quadratic_radical_condition_l446_446103

theorem quadratic_radical_condition (x : ℝ) : (√(x + 3)).isReal ↔ x ≥ -3 :=
by
  sorry

end quadratic_radical_condition_l446_446103


namespace satisfy_equation_l446_446623

theorem satisfy_equation (x : ℝ) (h : x > 0) : 
  (x ^ (Real.log10 (x ^ 2)) = (x ^ 4) / 1000) → (x = 10 ∨ x = 31.6227766) :=
sorry

end satisfy_equation_l446_446623


namespace horse_speed_is_20_l446_446940

-- Define the conditions
def bullet_speed : ℝ := 400
def bullet_speed_difference : ℝ := 40

-- Defining the unknown speed of the horse
noncomputable def horse_speed : ℝ :=
  (40 / 2)

-- State the theorem
theorem horse_speed_is_20 :
  let v_h := horse_speed in
  (bullet_speed + v_h) = (bullet_speed - v_h) + bullet_speed_difference :=
by
  let v_h := horse_speed
  have h1 : bullet_speed + v_h = 440 - v_h := by sorry
  sorry

end horse_speed_is_20_l446_446940


namespace second_discount_percentage_l446_446572

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ) :
  original_price = 10000 →
  first_discount = 0.20 →
  final_price = 6840 →
  second_discount = 14.5 :=
by
  sorry

end second_discount_percentage_l446_446572


namespace find_angles_of_triangle_l446_446526

noncomputable def similarity_ratio := Real.sqrt 3

-- Define the angles
def angle_A := 30
def angle_B := 60
def angle_C := 90

-- Lean statement
theorem find_angles_of_triangle (A B C E : Point)
  (h1 : segment BE divides triangle ABC into two similar triangles ABE and BEC with similarity ratio similarity_ratio) :
  ∠BAC = angle_A ∧ ∠ABC = angle_C ∧ ∠BCA = angle_B :=
by
  sorry

end find_angles_of_triangle_l446_446526


namespace ratio_adult_child_l446_446107

theorem ratio_adult_child (total_fee adults_fee children_fee adults children : ℕ) 
  (h1 : adults ≥ 1) (h2 : children ≥ 1) 
  (h3 : adults_fee = 30) (h4 : children_fee = 15) 
  (h5 : total_fee = 2250) 
  (h6 : adults_fee * adults + children_fee * children = total_fee) :
  (2 : ℚ) = adults / children :=
sorry

end ratio_adult_child_l446_446107


namespace fatou_lemma_conditional_expectation_l446_446204

open MeasureTheory

variables {Ω : Type*} {F : MeasurableSpace Ω} {P : ProbabilityMeasure F}
variables {G : MeasurableSpace Ω} {ξ : ℕ → Ω → ℝ}
variables (h1 : is_probability_measure P)
variables (h2 : ∀ n, integrable (ξ n) P)
variables
  (h3 : ∀ k, 
    tendsto 
      (λ n, ∫ ω, (ξ n ω).neg * indicator {ω | (ξ n ω).neg ≥ k} ω dP) at_top (nhds 0))

theorem fatou_lemma_conditional_expectation :
  ∫ ω, (liminf (λ n, ξ n) ω) dP ≤ liminf (λ n, ∫ ω, ξ n ω dP) :=
  sorry

end fatou_lemma_conditional_expectation_l446_446204


namespace complement_union_and_complement_intersect_l446_446065

-- Definitions of sets according to the problem conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

-- The correct answers derived in the solution
def complement_union_A_B : Set ℝ := { x | x ≤ 2 ∨ 10 ≤ x }
def complement_A_intersect_B : Set ℝ := { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) }

-- Statement of the mathematically equivalent proof problem
theorem complement_union_and_complement_intersect:
  (Set.compl (A ∪ B) = complement_union_A_B) ∧ 
  ((Set.compl A) ∩ B = complement_A_intersect_B) :=
  by 
    sorry

end complement_union_and_complement_intersect_l446_446065


namespace chord_length_polar_coordinates_l446_446752

theorem chord_length_polar_coordinates :
  (∃ (ρ θ : ℝ), ρ = 3 ∧ ρ * sin (θ + π / 4) = 2) → 
  (∃ L : ℝ, L = 2 * sqrt 5) :=
sorry

end chord_length_polar_coordinates_l446_446752


namespace quarts_of_water_required_l446_446710

-- Define the ratio of water to juice
def ratio_water_to_juice : Nat := 5 / 3

-- Define the total punch to prepare in gallons
def total_punch_in_gallons : Nat := 2

-- Define the conversion factor from gallons to quarts
def quarts_per_gallon : Nat := 4

-- Define the total number of parts
def total_parts : Nat := 5 + 3

-- Define the total punch in quarts
def total_punch_in_quarts : Nat := total_punch_in_gallons * quarts_per_gallon

-- Define the amount of water per part
def quarts_per_part : Nat := total_punch_in_quarts / total_parts

-- Prove the required amount of water in quarts
theorem quarts_of_water_required : quarts_per_part * 5 = 5 := 
by
  -- Proof is omitted, represented by sorry
  sorry

end quarts_of_water_required_l446_446710


namespace largest_corner_sum_l446_446616

-- Define the cube and its properties
structure Cube :=
  (faces : ℕ → ℕ)
  (opposite_faces_sum_to_8 : ∀ i, faces i + faces (7 - i) = 8)

-- Prove that the largest sum of three numbers whose faces meet at one corner is 16
theorem largest_corner_sum (c : Cube) : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  (c.faces i + c.faces j + c.faces k = 16) :=
sorry

end largest_corner_sum_l446_446616


namespace common_tangent_exists_range_of_a_l446_446367

noncomputable def a_range : Set ℝ := Set.Ico (-2/Real.exp 1) 0

theorem common_tangent_exists_range_of_a
  (x1 x2 : ℝ)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (a : ℝ)
  (h_tangent : (∀ (x > 0), DifferentiableAt ℝ (λ x, a / x) x) ∧ 
               (∀ (x > 0), DifferentiableAt ℝ (λ x, 2 * Real.log x) x) ∧ 
               ∃ (x1 x2 : ℝ), a/x1^2 = -2/x2 ∧ (a/x1 = Real.log x2 - 1)) :
  a ∈ a_range :=
sorry

end common_tangent_exists_range_of_a_l446_446367


namespace determinant_equals_2_l446_446421

open Real -- Provide access to real number operations and properties
open Matrix -- Provide matrix operations

-- Define the angles A, B, C and the associated matrix
variables (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
(hABC : A + B + C = π) -- Angles of a non-right triangle sum to π radians (180 degrees)

-- Define the cotangent function
def cot (x: ℝ) : ℝ := 1 / tan x

-- Define the matrix
def M : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [cot A, 1, 1],
    [1, cot B, 1],
    [1, 1, cot C]
  ]

-- The statement to be proved
theorem determinant_equals_2 (h : ¬(A = π / 2 ∨ B = π / 2 ∨ C = π / 2)) : 
  M.det = 2 :=
by {
  sorry
}

end determinant_equals_2_l446_446421


namespace plane_ticket_price_l446_446134

theorem plane_ticket_price :
  ∀ (P : ℕ),
  (20 * 155) + 2900 = 30 * P →
  P = 200 := 
by
  sorry

end plane_ticket_price_l446_446134


namespace cartesian_curve_equation_line_through_fixed_point_line_cartesian_equation_l446_446012

section math_problem

variable (t : ℝ) (α : ℝ)
def line_parametric (t α : ℝ) := (2 + t * Real.cos α, t * Real.sin α)

def curve_polar (ρ θ : ℝ) := ρ ^ 2 * (Real.cos θ ^ 2 + 2 * Real.sin θ ^ 2) = 12

theorem cartesian_curve_equation (x y : ℝ) (ρ θ : ℝ) :
  x = ρ * Real.cos θ → y = ρ * Real.sin θ → curve_polar ρ θ →
  x^2 + 2 * y^2 = 12 :=
by
sorry

theorem line_through_fixed_point (α : ℝ) :
  ∀ t, (∃ x' y', line_parametric t α = (x', y')) → (2, 0) ∈ line_parametric t α :=
by
sorry

theorem line_cartesian_equation (x y : ℝ) :
  |(2 - 2)| * 6 = 6 → (∀ t, (2 + t * (1/3), t * (√3/3)) = (x, y)) →
  y = ± (√2 / 2) * (x - 2) :=
by
sorry

end math_problem

end cartesian_curve_equation_line_through_fixed_point_line_cartesian_equation_l446_446012


namespace negate_neg_abs_value_first_expression_second_expression_l446_446205

theorem negate_neg (x : ℝ) : -(-x) = x :=
by {
  sorry  -- Proof will go here
}

theorem abs_value (x : ℝ) : |x| = if x < 0 then -x else x :=
by {
  sorry  -- Proof will go here
}

theorem first_expression : -(-2) = 2 :=
by {
  exact negate_neg 2,
}

theorem second_expression : -| -2 | = -2 :=
by {
  rw abs_value,
  finish,  -- Simplify the absolute value and negation
}

end negate_neg_abs_value_first_expression_second_expression_l446_446205


namespace problem_1_problem_2_l446_446336

noncomputable def f (a x : ℝ) : ℝ := - (a / 2) * x^2 + (a - 1) * x + Real.log x

theorem problem_1 (a x : ℝ) (h_neg_one_lt_a : a > -1) :
  (a ≥ 0 → ∀ x > 0, ((0 < x ∧ x < 1) → (Real.derivative (f a)) x > 0) ∧
                (x > 1 → (Real.derivative (f a)) x < 0)) ∧
  ((-1 < a ∧ a < 0) → ∃ x1 x2, x1 = 1 ∧ x2 = -1/a ∧
       x1 < x2 ∧ ∀ x > 0, ((0 < x ∧ x < x1) → (Real.derivative (f a)) x > 0) ∧
              ((x1 < x ∧ x < x2) → (Real.derivative (f a)) x < 0) ∧
              (x > x2 → (Real.derivative (f a)) x > 0)) :=
sorry

theorem problem_2 (a : ℝ) (h_a_gt_1 : a > 1) :
  ∀ x > 0, (2 * a - 1) * f a x < 3 * Real.exp (a - 3) :=
sorry

end problem_1_problem_2_l446_446336


namespace choose_two_subjects_l446_446347

theorem choose_two_subjects (n k : ℕ) (h_n : n = 3) (h_k : k = 2) :
    nat.choose n k = 3 :=
by
    rw [h_n, h_k]
    exact nat.choose_succ_succ_eq _ _ (by norm_num) -- C(3, 2) = 3 as calculated

end choose_two_subjects_l446_446347


namespace exists_m_n_l446_446782

open Classical

noncomputable def a (n : ℕ) : ℕ :=
  n

noncomputable def b (n : ℕ) : ℝ :=
  if n = 1 then 1/2 else 1/2 * 1/3^(n-1)

noncomputable def S (n : ℕ) : ℝ :=
  Σ k in range n, b k

noncomputable def T (n : ℕ) : ℝ :=
  Σ i in range (n + 1), Σ j in range (i + 1), b j - b (j+1)/(i+1)

theorem exists_m_n : ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ T n = (a (m+1) : ℝ) / (2 * a m) :=
by
  sorry

end exists_m_n_l446_446782


namespace problem_solution_l446_446702

noncomputable def vector_a : (ℝ × ℝ × ℝ) := (-2, -3, 1)
noncomputable def vector_b : (ℝ × ℝ × ℝ) := (2, 0, 4)
noncomputable def vector_c : (ℝ × ℝ × ℝ) := (-4, -6, 2)

-- Definitions for dot product
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Definitions for parallelism and orthogonality
def parallel (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
∃ k : ℝ, v2 = (k * v1.1, k * v1.2, k * v1.3)

def orthogonal (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
dot_product v1 v2 = 0

theorem problem_solution :
  parallel vector_a vector_c ∧ orthogonal vector_a vector_b :=
by
  sorry

end problem_solution_l446_446702


namespace vector_magnitude_calculation_l446_446676

variable (a b : ℝ → ℝ → ℝ) -- This defines vector space over real numbers
variable (angle_ab : ℝ)
variable (norm_a : ℝ)
variable (norm_b : ℝ)

noncomputable def vector_magnitude {x y : ℝ → ℝ → ℝ} (v : ℝ → ℝ → ℝ) :=
  real.sqrt (v.1^2 + v.2^2)

theorem vector_magnitude_calculation 
  (h1 : angle_ab = real.pi / 3)
  (h2 : vector_magnitude a = 1)
  (h3 : vector_magnitude b = 2) :
  vector_magnitude (λ x y, 2 * a x y - b x y) = 2 := 
  sorry

end vector_magnitude_calculation_l446_446676


namespace wire_length_l446_446990

variables (L M S W : ℕ)

def ratio_condition (L M S : ℕ) : Prop :=
  L * 2 = 7 * S ∧ M * 2 = 3 * S

def total_length (L M S : ℕ) : ℕ :=
  L + M + S

theorem wire_length (h : ratio_condition L M 16) : total_length L M 16 = 96 :=
by sorry

end wire_length_l446_446990


namespace odd_function_exists_inequality_min_value_l446_446340

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (4^x + k) / (2^x)

theorem odd_function_exists :
  ∃ k : ℝ, ∀ x : ℝ, f (-x) k = - f x k :=
begin
  sorry
end

theorem inequality_min_value (a : ℝ) :
  (∀ x : ℝ, f x 1 ≥ a) ↔ a ≤ 2 :=
begin
  sorry
end

end odd_function_exists_inequality_min_value_l446_446340


namespace ball_returns_to_bella_after_13_throws_l446_446534

def girl_after_throws (start : ℕ) (throws : ℕ) : ℕ :=
  (start + throws * 5) % 13

theorem ball_returns_to_bella_after_13_throws :
  girl_after_throws 1 13 = 1 :=
sorry

end ball_returns_to_bella_after_13_throws_l446_446534


namespace count_rational_numbers_in_first_2016_terms_of_Sn_is_43_l446_446056

noncomputable def sequence (n : ℕ) : ℚ := 1 / ((n + 1) * real.sqrt n + n * real.sqrt (n + 1))

noncomputable def sum_sequence (n : ℕ) : ℚ :=
  (finset.range n).sum (λ k, sequence (k + 1))

theorem count_rational_numbers_in_first_2016_terms_of_Sn_is_43 :
  finset.card ((finset.range 2016).filter (λ n, (sum_sequence(n+1) : ℝ).is_rational)) = 43 :=
sorry

end count_rational_numbers_in_first_2016_terms_of_Sn_is_43_l446_446056


namespace sum_of_valid_fourth_powers_is_354_l446_446152

-- Given conditions
def is_valid_fourth_power (n : ℕ) : Prop := n^4 < 500

-- The main theorem we want to prove
theorem sum_of_valid_fourth_powers_is_354 :
  let valid_n := { n // is_valid_fourth_power n }
  let fourth_powers := { n : ℕ | is_valid_fourth_power n }.to_finset.val
  fourth_powers.sum = 354 :=
by
  -- Proof omitted
  sorry

end sum_of_valid_fourth_powers_is_354_l446_446152


namespace number_of_teachers_l446_446814

theorem number_of_teachers
    (students : ℕ)
    (classes_per_student : ℕ)
    (students_per_class : ℕ)
    (classes_per_teacher : ℕ)
    (students = 1500)
    (classes_per_student = 4)
    (students_per_class = 25)
    (classes_per_teacher = 5) :
    ∃ teachers, teachers = 48 :=
by
  sorry

end number_of_teachers_l446_446814


namespace min_total_time_l446_446536

-- Lean 4 statement
theorem min_total_time (tA tB tC : ℝ) (hA : tA = 1.5) (hB : tB = 0.5) (hC : tC = 1):
  ∃ (min_time : ℝ), min_time = 5 :=
by
  use 5
  sorry

end min_total_time_l446_446536


namespace ratio_MX_XC_l446_446099

noncomputable def base_parallelogram (A B C D : Type) [Parallelogram A B C D] := Parallelogram.mk A B C D
noncomputable def bisects_edge (D M K : Type) [Bisection D M K] := Bisection.mk D M K
noncomputable def point_ratio (B M P : Type) : BP_PMRatio B M P := sorry
noncomputable def intersects_plane (A P K M C X : Type) [PlaneIntersection A P K M C X] := PlaneIntersection.mk A P K M C X

theorem ratio_MX_XC (M A B C D K P X : Type) [Parallelogram A B C D] [Bisection D M K] 
  [BP_PMRatio B M P] [PlaneIntersection A P K M C X] : (MX_ratio M X C) = 3/4 :=
by sorry 

end ratio_MX_XC_l446_446099


namespace remainder_of_8_pow_6_plus_1_mod_7_l446_446550

theorem remainder_of_8_pow_6_plus_1_mod_7 :
  (8^6 + 1) % 7 = 2 := by
  sorry

end remainder_of_8_pow_6_plus_1_mod_7_l446_446550


namespace new_volume_is_correct_l446_446995

-- Define the original volume and scale factors
def original_volume : ℝ := 60
def length_scale_factor : ℝ := 4
def width_scale_factor : ℝ := 2
def height_scale_factor : ℝ := 0.75

-- Define the original dimensions and volume formula
variables (l w h : ℝ)

-- Define the expression for the original volume
def V : ℝ := (1/3) * l * w * h

-- We assume the volume is known to be 60 cubic inches
axiom original_V_eq_60 : V l w h = original_volume

-- Define the new dimensions
def new_length := length_scale_factor * l
def new_width := width_scale_factor * w
def new_height := height_scale_factor * h

-- Define the expression for the new volume
def V' : ℝ := (1/3) * new_length * new_width * new_height

-- Prove that the new volume is 360 cubic inches
theorem new_volume_is_correct (l w h : ℝ) : V' l w h = 360 := 
by 
  -- Here we will add proof steps
  sorry

end new_volume_is_correct_l446_446995


namespace shoe_matching_probability_l446_446558

theorem shoe_matching_probability :
  let total_pairs := 9
  let total_shoes := 2 * total_pairs
  let ways_to_select_two_shoes := total_shoes * (total_shoes - 1) / 2
  let ways_to_select_matching_pair := total_pairs
  let probability := (ways_to_select_matching_pair : ℚ) / (ways_to_select_two_shoes : ℚ)
in
  probability = (1 / 17 : ℚ) :=
by
  sorry

end shoe_matching_probability_l446_446558


namespace oranges_savings_l446_446792

-- Definitions for the conditions
def liam_oranges : Nat := 40
def liam_price_per_set : Real := 2.50
def oranges_per_set : Nat := 2

def claire_oranges : Nat := 30
def claire_price_per_orange : Real := 1.20

-- Statement of the problem to be proven
theorem oranges_savings : 
  liam_oranges / oranges_per_set * liam_price_per_set + 
  claire_oranges * claire_price_per_orange = 86 := 
by 
  sorry

end oranges_savings_l446_446792


namespace teal_more_blue_l446_446212

theorem teal_more_blue (total : ℕ) (green : ℕ) (both_green_blue : ℕ) (neither_green_blue : ℕ)
  (h1 : total = 150) (h2 : green = 90) (h3 : both_green_blue = 40) (h4 : neither_green_blue = 25) :
  ∃ (blue : ℕ), blue = 75 :=
by
  sorry

end teal_more_blue_l446_446212


namespace spend_on_rent_and_utilities_l446_446445

variable (P : ℝ) -- The percentage of her income she used to spend on rent and utilities
variable (I : ℝ) -- Her previous monthly income
variable (increase : ℝ) -- Her salary increase
variable (new_percentage : ℝ) -- The new percentage her rent and utilities amount to

noncomputable def initial_conditions : Prop :=
I = 1000 ∧ increase = 600 ∧ new_percentage = 0.25

theorem spend_on_rent_and_utilities (h : initial_conditions I increase new_percentage) :
    (P / 100) * I = 0.25 * (I + increase) → 
    P = 40 :=
by
  sorry

end spend_on_rent_and_utilities_l446_446445


namespace intersecting_diagonals_prob_nonagon_l446_446958

theorem intersecting_diagonals_prob_nonagon :
  let num_vertices := 9
  let num_diagonals := (nat.choose num_vertices 2) - num_vertices
  let pairs_of_diagonals := nat.choose num_diagonals 2
  let num_intersecting_diagonals := nat.choose num_vertices 4
  (num_intersecting_diagonals / pairs_of_diagonals) = 6 / 13 :=
by
  sorry

end intersecting_diagonals_prob_nonagon_l446_446958


namespace sum_a_n_lt_1_l446_446587

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
| 1     := 1 / 2
| (n+1) := if n = 0 then 1 / 2 else (2 * (n + 1) - 3) / (2 * (n + 1)) * a n

-- Define the sum of the sequence up to n
noncomputable def partial_sum (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, a (k + 1)

-- State the theorem
theorem sum_a_n_lt_1 (n : ℕ) : partial_sum n < 1 := 
sorry

end sum_a_n_lt_1_l446_446587


namespace sum_of_solutions_l446_446438

def f (x : ℝ) : ℝ :=
if x < -3 then
  3 * x + 9
else
  -x^2 - 2 * x + 2

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | f x = -6}, x) = -9 :=
by
  sorry

end sum_of_solutions_l446_446438


namespace cauchy_schwarz_equivalent_iag_l446_446646

theorem cauchy_schwarz_equivalent_iag (a b c d : ℝ) :
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → (Real.sqrt x * Real.sqrt y) ≤ (x + y) / 2) ↔
  ((a * c + b * d) ^ 2 ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2)) := by
  sorry

end cauchy_schwarz_equivalent_iag_l446_446646


namespace jacket_price_before_noon_l446_446981

theorem jacket_price_before_noon 
  (P : ℝ)
  (total_jackets : ℕ)
  (price_after_noon : ℝ)
  (jackets_sold_after_noon : ℕ)
  (total_receipts : ℝ)
  (total_jackets = 214)
  (price_after_noon = 18.95)
  (jackets_sold_after_noon = 133)
  (total_receipts = 5108.30) : 
  P = 31.98 :=
by
  have jackets_sold_before_noon : ℕ := 214 - 133
  have revenue_after_noon : ℝ := 133 * 18.95
  have revenue_before_noon : ℝ := 5108.30 - revenue_after_noon
  have P := revenue_before_noon / 81
  have P_rounded := Real.floor (P * 100 + 0.5) / 100
  guard_hyp P_rounded = 31.98
  sorry

end jacket_price_before_noon_l446_446981


namespace right_handed_players_total_l446_446447

theorem right_handed_players_total (total_players throwers : ℕ) (h1 : total_players = 70) (h2 : throwers = 31) (h3 : ∃ left_handed_non_throwers right_handed_non_throwers : ℕ, left_handed_non_throwers = right_handed_non_throwers / 3 + right_handed_non_throwers ∧ right_handed_non_throwers = (total_players - throwers) - left_handed_non_throwers) (h4 : ∀ t, t ∈ throwers → t.right_handed) : (31 + (39 - 13) = 57) :=
by {
  sorry
}

end right_handed_players_total_l446_446447


namespace probability_of_red_ball_l446_446741

theorem probability_of_red_ball (redBalls yellowBalls : ℕ) (h_redBalls : redBalls = 6) (h_yellowBalls : yellowBalls = 3) :
  (redBalls.toRat / (redBalls + yellowBalls).toRat) = (2 / 3) := by
  sorry

end probability_of_red_ball_l446_446741


namespace johns_annual_haircut_cost_l446_446028

-- Define constants given in the problem
def hair_growth_rate : ℝ := 1.5 -- inches per month
def initial_hair_length_before_cut : ℝ := 9 -- inches
def initial_hair_length_after_cut : ℝ := 6 -- inches
def haircut_cost : ℝ := 45 -- dollars
def tip_percentage : ℝ := 0.2 -- 20%

-- Calculate the number of haircuts per year and the total cost including tips
def monthly_hair_growth := initial_hair_length_before_cut - initial_hair_length_after_cut
def haircut_interval := monthly_hair_growth / hair_growth_rate
def number_of_haircuts_per_year := 12 / haircut_interval
def tip_amount_per_haircut := haircut_cost * tip_percentage
def total_cost_per_haircut := haircut_cost + tip_amount_per_haircut
def total_annual_cost := number_of_haircuts_per_year * total_cost_per_haircut

-- Prove the total annual cost to be $324
theorem johns_annual_haircut_cost : total_annual_cost = 324 := by
  sorry

end johns_annual_haircut_cost_l446_446028


namespace shaded_region_perimeter_l446_446013

def point : Type := ℝ × ℝ

def distance (p q : point) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

noncomputable def perimeter_shaded_region : ℝ :=
  let O : point := (0, 0)
  let P : point := (-7, 0)
  let Q : point := (0, -7)
  let radius := 7
  let arc_length := (5 / 6) * (2 * real.pi * radius)
  distance O P + distance O Q + arc_length

theorem shaded_region_perimeter :
  perimeter_shaded_region = 14 + (35 / 3) * real.pi :=
by
  -- Detailed proof omitted
  sorry

end shaded_region_perimeter_l446_446013


namespace common_tangent_exists_range_of_a_l446_446366

noncomputable def a_range : Set ℝ := Set.Ico (-2/Real.exp 1) 0

theorem common_tangent_exists_range_of_a
  (x1 x2 : ℝ)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (a : ℝ)
  (h_tangent : (∀ (x > 0), DifferentiableAt ℝ (λ x, a / x) x) ∧ 
               (∀ (x > 0), DifferentiableAt ℝ (λ x, 2 * Real.log x) x) ∧ 
               ∃ (x1 x2 : ℝ), a/x1^2 = -2/x2 ∧ (a/x1 = Real.log x2 - 1)) :
  a ∈ a_range :=
sorry

end common_tangent_exists_range_of_a_l446_446366


namespace remainder_of_product_mod_7_l446_446881

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l446_446881


namespace isosceles_triangle_l446_446888

theorem isosceles_triangle 
  (a b c : ℝ) 
  (A : ℝ) 
  (cos_A : ℝ) 
  (log_eq : real.log (a^2) = real.log (b^2) + real.log (c^2) - real.log (2 * b * c * cos_A)) :
  (a = b) ∨ (a = c) :=
sorry

end isosceles_triangle_l446_446888


namespace standard_deviation_of_numbers_l446_446850

theorem standard_deviation_of_numbers :
  let nums := [9.8, 9.8, 9.9, 9.9, 10.0, 10.0, 10.1, 10.5]
  let mean := 10
  87.5% of nums are within 1 standard deviation of the mean
  shows standard deviation nums = 0.5 :=
sorry

end standard_deviation_of_numbers_l446_446850


namespace jeff_total_cabinets_l446_446757

theorem jeff_total_cabinets : 
  let initial_cabinets := 3
  let additional_per_counter := 3 * 2
  let num_counters := 3
  let additional_total := additional_per_counter * num_counters
  let final_cabinets := additional_total + 5
in initial_cabinets + final_cabinets = 26 :=
by
  -- Proof omitted
  sorry

end jeff_total_cabinets_l446_446757


namespace tangent_line_at_0_l446_446846

section
variables (x : ℝ)
def f : ℝ → ℝ := λ x, x^2 + x - 2 * Real.exp x
def f' : ℝ → ℝ := λ x, 2 * x + 1 - 2 * Real.exp x

theorem tangent_line_at_0 : 
  ∃ y, (f x = x^2 + x - 2 * Real.exp x) →
  (f' x = 2 * x + 1 - 2 * Real.exp x) →
  x = 0 →
  (f 0 = -2) →
  (f' 0 = -1) →
  (y = -x - 2) :=
sorry
end

end tangent_line_at_0_l446_446846


namespace circles_tangent_length_of_chord_l446_446611

open Real

-- Definitions based on given conditions
def r₁ := 4
def r₂ := 10
def r₃ := 14
def length_of_chord : ℝ := (8 * real.sqrt 390) / 7

-- Main theorem statement
theorem circles_tangent_length_of_chord (m n p : ℕ) 
  (h₁ : ∃ m n p, length_of_chord = (m * real.sqrt n) / p)
  (h₂ : m.gcd p = 1)
  (h₃ : ∀ k : ℕ, k^2 ∣ n → k=1) 
  : m + n + p = 405 :=
by 
  use 8, 390, 7
  sorry

end circles_tangent_length_of_chord_l446_446611


namespace sum_of_floor_sqrt_and_neg_floor_sqrt_l446_446158

theorem sum_of_floor_sqrt_and_neg_floor_sqrt (n : ℕ) (h : n = 1989 * 1990) :
  (∑ x in Finset.range (n + 1), (Int.floor (Real.sqrt x) + Int.floor (-(Real.sqrt x)))) = -3956121 :=
by
  -- placeholder proof
  sorry

end sum_of_floor_sqrt_and_neg_floor_sqrt_l446_446158


namespace no_full_circle_l446_446959

-- Define the problem setup
def bead_positions (n : ℕ) := {θ : ℤ | -2 * n < θ ∧ θ < 0}

-- Define the condition of repositioning
def repositioned (n : ℕ) (θs : Set ℤ) (i : ℕ) : Prop :=
  ∀ i, θs i = if i = 1 then 1 / 2 * (θs 2 + θs n - 2 * n) 
              else if i = n then 1 / 2 * (θs 1 + θs (n-1) + 2 * n) 
              else 1 / 2 * (θs (i-1) + θs (i+1))

-- Define the proof of no bead completing a full circle
theorem no_full_circle (n : ℕ) (θs : Fin n → ℤ) 
  (h1 : 0 < n) 
  (h2 : ∀ θ ∈ θs, -2 * n < θ ∧ θ < 0) 
  (h3 : repositioned n θs 1) 
  (h4 : repositioned n θs n): 
  ¬(∃ (initial_positions : Fin n → ℤ)
      (sequence_of_moves : ℕ → Fin n → ℤ), 
      ∀ i, θs i = repositioned n initial_positions i) :=
sorry

end no_full_circle_l446_446959


namespace vector_subtraction_l446_446692

def a : ℝ × ℝ := (5, 3)
def b : ℝ × ℝ := (1, -2)
def scalar : ℝ := 2

theorem vector_subtraction :
  a.1 - scalar * b.1 = 3 ∧ a.2 - scalar * b.2 = 7 :=
by {
  -- here goes the proof
  sorry
}

end vector_subtraction_l446_446692


namespace no_consecutive_numbers_on_adjacent_faces_l446_446502

noncomputable def dodecahedron_probability : ℚ :=
  let total_permutations := factorial 12
  -- Function to count valid configurations — needs a detailed specification
  let valid_configurations := sorry -- assuming some counting function to be defined
  valid_configurations / total_permutations

theorem no_consecutive_numbers_on_adjacent_faces :
  dodecahedron_probability = m / n ∧ gcd m n = 1 → m + n = 2274 :=
  by sorry

end no_consecutive_numbers_on_adjacent_faces_l446_446502


namespace find_ages_l446_446816

-- Definitions based on the conditions
variables (x : ℝ)
def Sandy_age_now := 4 * x
def Molly_age_now := 3 * x
def Kim_age_now := 5 * x
def Sandy_age_8_years := Sandy_age_now x + 8

-- The given conditions
axiom condition1 : Sandy_age_8_years x = 74

-- Proving the required ages
theorem find_ages (hx : x = 16.5) :
  Molly_age_now x = 49.5 ∧ Kim_age_now x = 82.5 := by
  split;
  sorry

end find_ages_l446_446816


namespace part_a_least_moves_part_b_least_moves_l446_446546

def initial_position : Nat := 0
def total_combinations : Nat := 10^6
def excluded_combinations : List Nat := [0, 10^5, 2 * 10^5, 3 * 10^5, 4 * 10^5, 5 * 10^5, 6 * 10^5, 7 * 10^5, 8 * 10^5, 9 * 10^5]

theorem part_a_least_moves : total_combinations - 1 = 10^6 - 1 := by
  simp [total_combinations, Nat.pow]

theorem part_b_least_moves : total_combinations - excluded_combinations.length = 10^6 - 10 := by
  simp [total_combinations, excluded_combinations, Nat.pow, List.length]

end part_a_least_moves_part_b_least_moves_l446_446546


namespace ocean_depth_350_l446_446221

noncomputable def depth_of_ocean (total_height : ℝ) (volume_ratio_above_water : ℝ) : ℝ :=
  let volume_ratio_below_water := 1 - volume_ratio_above_water
  let height_below_water := (volume_ratio_below_water^(1 / 3)) * total_height
  total_height - height_below_water

theorem ocean_depth_350 :
  depth_of_ocean 10000 (1 / 10) = 350 :=
by
  sorry

end ocean_depth_350_l446_446221


namespace aluminum_phosphate_molecular_weight_l446_446928

theorem aluminum_phosphate_molecular_weight :
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  (Al + P + 4 * O) = 121.95 :=
by
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  sorry

end aluminum_phosphate_molecular_weight_l446_446928


namespace prime_exponent_50_factorial_5_l446_446016

def count_factors_in_factorial (n p : ℕ) : ℕ :=
  if p ≤ 1 then 0 else
    let rec count (k acc : ℕ) : ℕ :=
      if n < p^k then acc else count (k+1) (acc + n/(p^k))
    count 1 0

theorem prime_exponent_50_factorial_5 : count_factors_in_factorial 50 5 = 12 :=
  by
    sorry

end prime_exponent_50_factorial_5_l446_446016


namespace tamia_bell_pepper_pieces_l446_446089

def total_pieces (n k p : Nat) : Nat :=
  let slices := n * k
  let half_slices := slices / 2
  let smaller_pieces := half_slices * p
  let total := half_slices + smaller_pieces
  total

theorem tamia_bell_pepper_pieces :
  total_pieces 5 20 3 = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l446_446089


namespace product_mod_7_l446_446864

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l446_446864


namespace find_a_l446_446043

noncomputable def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x * a = 1}
axiom A_is_B (a : ℝ) : A ∩ B a = B a → (a = 0) ∨ (a = 1/3) ∨ (a = 1/5)

-- statement to prove
theorem find_a (a : ℝ) (h : A ∩ B a = B a) : (a = 0) ∨ (a = 1/3) ∨ (a = 1/5) :=
by 
  apply A_is_B
  assumption

end find_a_l446_446043


namespace count_of_numbers_with_digit_3_eq_71_l446_446709

-- Define the problem space
def count_numbers_without_digit_3 : ℕ := 729
def total_numbers : ℕ := 800
def count_numbers_with_digit_3 : ℕ := total_numbers - count_numbers_without_digit_3

-- Prove that the count of numbers from 1 to 800 containing at least one digit 3 is 71
theorem count_of_numbers_with_digit_3_eq_71 :
  count_numbers_with_digit_3 = 71 :=
by
  sorry

end count_of_numbers_with_digit_3_eq_71_l446_446709


namespace ratio_of_volumes_l446_446980

theorem ratio_of_volumes
  (R : ℝ) (α β : ℝ)
  (cone_volume : ℝ) (cylinder_volume : ℝ) :
  cone_volume = (1/3 : ℝ) * π * R^3 * Real.cot β →
  cylinder_volume = π * ((R * Real.cos (α + β) / (Real.cos α * Real.cos β))^2) * (R * Real.tan α) →
  cone_volume / cylinder_volume = (Real.cos α)^3 * (Real.cos β)^3 / (3 * Real.sin α * Real.sin β * (Real.cos (α + β))^2) :=
by
  intro h_cone_volume h_cylinder_volume
  rw [h_cone_volume, h_cylinder_volume]
  sorry

end ratio_of_volumes_l446_446980


namespace wire_length_l446_446989

variables (L M S W : ℕ)

def ratio_condition (L M S : ℕ) : Prop :=
  L * 2 = 7 * S ∧ M * 2 = 3 * S

def total_length (L M S : ℕ) : ℕ :=
  L + M + S

theorem wire_length (h : ratio_condition L M 16) : total_length L M 16 = 96 :=
by sorry

end wire_length_l446_446989


namespace five_letter_arrangements_l446_446707

theorem five_letter_arrangements (A B C D E F G H : Type) :
  let letters := {A, B, C, D, E, F, G, H} in
  ∃ n : ℕ,
  (n = 5) ∧                                  -- The length of the arrangement is 5
  (D ∈ letters) ∧                             -- D must be in the first position
  (A ∈ letters) ∧                             -- A must be in the last position
  (E ∈ letters) ∧                             -- E must be one of the letters
  (∀ x : letters, x ≠ x) →                     -- No letter can be used more than once
  n = 60 :=
by
  sorry

end five_letter_arrangements_l446_446707


namespace solve_expression_l446_446854

theorem solve_expression :
  (27 ^ (2 / 3) - 2 ^ (Real.log 3 / Real.log 2) * (Real.logb 2 (1 / 8)) +
    Real.logb 10 4 + Real.logb 10 25 = 20) :=
by
  sorry

end solve_expression_l446_446854


namespace gcd_108_45_eq_9_l446_446544

-- Definition of the problem conditions
def gcd_euclidean (a b : ℕ) : ℕ :=
  if b = 0 then a
  else gcd_euclidean b (a % b)

-- The final statement to prove gcd(108, 45) = 9
theorem gcd_108_45_eq_9 : gcd_euclidean 108 45 = 9 := 
by
  -- We state the expected value
  have eq1 : gcd_euclidean 108 45 = gcd_euclidean 45 18 := sorry
  have eq2 : gcd_euclidean 45 18 = gcd_euclidean 18 9 := sorry
  have eq3 : gcd_euclidean 18 9 = gcd_euclidean 9 0 := sorry

  -- Since gcd_euclidean 9 0 = 9, we conclude
  have final_eq : gcd_euclidean 9 0 = 9 := by rfl
  exact final_eq

end gcd_108_45_eq_9_l446_446544


namespace path_grid_property_l446_446311

theorem path_grid_property
  (m n : ℕ) (hm : m ≥ 4) (hn : n ≥ 4) 
  (P : set (ℕ × ℕ)) -- P is the set of points forming the path
  (closed_path : true) -- Assuming P is a closed path
  (no_intersect : true) -- Assuming P does not self-intersect
  (A : ℕ) -- Number of points on P that are not turns
  (B : ℕ) -- Number of squares P goes through two non-adjacent sides
  (C : ℕ) -- Number of squares with no side in P
  (H : A + ... -- Definition of A based on path P
       B + ... -- Definition of B based on path P
       C + ... -- Definition of C based on path P
  ) :
  A = B - C + m + n - 1 := 
    sorry

end path_grid_property_l446_446311


namespace correct_order_of_solutions_l446_446375

structure SolutionCondition where
  forms_precipitate_with : list Nat
  soluble_in : Option Nat := none
  precipitate_soluble_in_excess : Option Nat := none

def solutionOrder (order: list Nat) (conditions: list SolutionCondition) : Prop :=
  -- Dummy implementation as placeholder for the actual chemical condition checks
  sorry

theorem correct_order_of_solutions :
  let solutions := [1, 2, 3, 4, 5, 6, 7, 8] in
  let conditions := [
    SolutionCondition.mk [1, 2, 3, 4] (some 7), -- For Solution 6
    SolutionCondition.mk [1, 2, 3, 4] (some 7), -- For Solution 8
    SolutionCondition.mk [4] (some 4) (some 8), -- For Solution 4 with 6, 8
    SolutionCondition.mk [1, 2] (some 1) (some 6), -- Precipitates of 1, 2 with 6, 8
    SolutionCondition.mk [5] none none, -- Solution 1 forms precipitate with 5, not soluble even in excess 7
    SolutionCondition.mk [4, 5] none (some 6) -- Solution 3 forms weak electrolyte with 4 and 5, 6 produces white precipitate soluble in 7
  ] in
  solutionOrder solutions conditions :=
  sorry

end correct_order_of_solutions_l446_446375


namespace hcl_combined_moles_l446_446294

theorem hcl_combined_moles (KOH KCl H2O : Type) [comm_ring KOH] [comm_ring KCl] [comm_ring H2O] :
  (∃ (HCl : Type) [comm_ring HCl], (HCl = KOH ∧ HCl = KCl ∧ KOH = KCl ∧ H2O).prod = HCl.prod) → 
  HCl.prod = KCl.prod :=
begin
  -- The proof will go here.
  sorry
end

end hcl_combined_moles_l446_446294


namespace disjoint_subsets_exist_l446_446434

open Set

variables {X : Type} 
variable [Fintype X]
variable (f : Set X → ℝ)
variable (D : Set X)

theorem disjoint_subsets_exist (hD : IsEven D ∧ f(D) > 1990)
(hf : ∀ A B : Set X, Disjoint A B ∧ IsEven A ∧ IsEven B → f(A ∪ B) = f(A) + f(B) - 1990) :
  ∃ P Q : Set X, Disjoint P Q ∧ P ∪ Q = univ ∧ 
  (∀ S, IsEven S ∧ S ≠ ∅ → f S > 1990) ∧ 
  (∀ T, IsEven T → f T ≤ 1990) :=
sorry

end disjoint_subsets_exist_l446_446434


namespace find_three_digit_number_l446_446288

-- Define digits a, b, c where a is non-zero for the three-digit number
variables (a b c : ℕ)
-- Conditions for digits
variables (ha : a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (hb : b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (hc : c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

-- Define the three-digit number
def number := 100 * a + 10 * b + c

-- Define the sum of the digits
def digit_sum := a + b + c

-- Theorem to prove the characterization of the number
theorem find_three_digit_number (h : number a b c = 12 * digit_sum a b c) :
  number a b c = 108 :=
by {
  sorry
}

end find_three_digit_number_l446_446288


namespace tom_books_l446_446539

theorem tom_books (books_may books_june books_july : ℕ) (h_may : books_may = 2) (h_june : books_june = 6) (h_july : books_july = 10) : 
books_may + books_june + books_july = 18 := by
sorry

end tom_books_l446_446539


namespace andy_paint_total_l446_446598

-- Define the given ratio condition and green paint usage
def paint_ratio (blue green white : ℕ) : Prop :=
  blue / green = 1 / 2 ∧ white / green = 5 / 2

def green_paint_used (green : ℕ) : Prop :=
  green = 6

-- Define the proof goal: total paint used
def total_paint_used (blue green white : ℕ) : ℕ :=
  blue + green + white

-- The statement to be proved
theorem andy_paint_total (blue green white : ℕ)
  (h_ratio : paint_ratio blue green white)
  (h_green : green_paint_used green) :
  total_paint_used blue green white = 24 :=
  sorry

end andy_paint_total_l446_446598


namespace inequality_solution_l446_446688

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 3
  else (1/3)^x - 2

def solution_set : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}

theorem inequality_solution :
  { x : ℝ | f x ≥ 1 } = solution_set :=
by
  sorry

end inequality_solution_l446_446688


namespace area_of_quadrilateral_l446_446018

noncomputable def C1 (a : ℝ) : (ℝ × ℝ) :=
  (sqrt(3) * cos a, sin a)

noncomputable def C2 : (ℝ × ℝ) → (ℝ × ℝ) :=
  λ (ρ θ : ℝ), (ρ * cos θ, ρ * sin θ)

noncomputable def C3 : (ℝ × ℝ) → (ℝ × ℝ) :=
  λ (ρ θ : ℝ), (ρ * cos (θ + π / 2), ρ * sin (θ + π / 2))

theorem area_of_quadrilateral :
  ∀ (a : ℝ), 3 ≤ 
    (λ (|AB| |CD|), 
      |AB| = 2 * sqrt (3 / (1 + 2 * sin a ^ 2)) ∧
      |CD| = 2 * sqrt (3 / (1 + 2 * cos a ^ 2)) ∧
      S := (1 / 2) * |AB| * |CD|)
    S ∧ S ≤ 2 * sqrt 3 :=
sorry

end area_of_quadrilateral_l446_446018


namespace johnsYearlyHaircutExpenditure_l446_446032

-- Definitions based on conditions
def hairGrowthRate : ℝ := 1.5 -- inches per month
def hairCutLength : ℝ := 9 -- inches
def hairAfterCut : ℝ := 6 -- inches
def monthsBetweenCuts := (hairCutLength - hairAfterCut) / hairGrowthRate
def haircutCost : ℝ := 45 -- dollars
def tipPercent : ℝ := 0.20
def tipsPerHaircut := tipPercent * haircutCost

-- Number of haircuts in a year
def numHaircutsPerYear := 12 / monthsBetweenCuts

-- Total yearly expenditure
def yearlyHaircutExpenditure := numHaircutsPerYear * (haircutCost + tipsPerHaircut)

theorem johnsYearlyHaircutExpenditure : yearlyHaircutExpenditure = 324 := 
by
  sorry

end johnsYearlyHaircutExpenditure_l446_446032


namespace contrapositive_is_false_l446_446104

-- Define the property of collinearity between two vectors
def collinear (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, a = k • b

-- Define the property of vectors having the same direction
def same_direction (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, k > 0 ∧ a = k • b

-- Original proposition in Lean statement
def original_proposition (a b : ℝ × ℝ) : Prop :=
  collinear a b → same_direction a b

-- Contrapositive of the original proposition
def contrapositive_proposition (a b : ℝ × ℝ) : Prop :=
  ¬ same_direction a b → ¬ collinear a b

-- The proof goal that the contrapositive is false
theorem contrapositive_is_false (a b : ℝ × ℝ) :
  (contrapositive_proposition a b) = false :=
sorry

end contrapositive_is_false_l446_446104


namespace lorry_sand_capacity_l446_446806

def cost_cement (bags : ℕ) (cost_per_bag : ℕ) : ℕ := bags * cost_per_bag
def total_cost (cement_cost : ℕ) (sand_cost : ℕ) : ℕ := cement_cost + sand_cost
def total_sand (sand_cost : ℕ) (cost_per_ton : ℕ) : ℕ := sand_cost / cost_per_ton
def sand_per_lorry (total_sand : ℕ) (lorries : ℕ) : ℕ := total_sand / lorries

theorem lorry_sand_capacity : 
  cost_cement 500 10 + (total_cost 5000 (total_sand 8000 40)) = 13000 ∧
  total_cost 5000 8000 = 13000 ∧
  total_sand 8000 40 = 200 ∧
  sand_per_lorry 200 20 = 10 :=
by
  sorry

end lorry_sand_capacity_l446_446806


namespace emily_remaining_flight_time_l446_446629

def flight_time_remaining 
  (flight_duration_hours : ℕ)
  (tv_episodes : ℕ) 
  (tv_episode_duration_minutes : ℕ)
  (sleeping_hours : ℕ)
  (sleeping_fraction : ℚ)
  (movie_count : ℕ)
  (movie_duration_minutes : ℕ)
  : ℕ :=
  let flight_duration_minutes := flight_duration_hours * 60
  let sleeping_total_minutes := sleeping_hours * 60 + (sleeping_fraction * 60).toNat
  let total_tv_time := tv_episodes * tv_episode_duration_minutes
  let total_movie_time := movie_count * movie_duration_minutes
  let total_spent_time := total_tv_time + sleeping_total_minutes + total_movie_time
  flight_duration_minutes - total_spent_time

theorem emily_remaining_flight_time :
  flight_time_remaining 10 3 25 4 1/2 2 105 = 45 :=
  by simp [flight_time_remaining]; sorry

end emily_remaining_flight_time_l446_446629


namespace profit_percentage_B_l446_446999

theorem profit_percentage_B (cost_price_A : ℝ) (sell_price_C : ℝ) 
  (profit_A_percent : ℝ) (profit_B_percent : ℝ) 
  (cost_price_A_eq : cost_price_A = 148) 
  (sell_price_C_eq : sell_price_C = 222) 
  (profit_A_percent_eq : profit_A_percent = 0.2) :
  profit_B_percent = 0.25 := 
by
  have cost_price_B := cost_price_A * (1 + profit_A_percent)
  have profit_B := sell_price_C - cost_price_B
  have profit_B_percent := (profit_B / cost_price_B) * 100 
  sorry

end profit_percentage_B_l446_446999


namespace find_three_digit_number_l446_446287

-- Define digits a, b, c where a is non-zero for the three-digit number
variables (a b c : ℕ)
-- Conditions for digits
variables (ha : a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (hb : b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (hc : c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

-- Define the three-digit number
def number := 100 * a + 10 * b + c

-- Define the sum of the digits
def digit_sum := a + b + c

-- Theorem to prove the characterization of the number
theorem find_three_digit_number (h : number a b c = 12 * digit_sum a b c) :
  number a b c = 108 :=
by {
  sorry
}

end find_three_digit_number_l446_446287


namespace fruit_seller_loss_percentage_l446_446224

theorem fruit_seller_loss_percentage :
  ∃ (C : ℝ), 
    (5 : ℝ) = C - (6.25 - C * (1 + 0.05)) → 
    (C = 6.25) → 
    (C - 5 = 1.25) → 
    (1.25 / 6.25 * 100 = 20) :=
by 
  sorry

end fruit_seller_loss_percentage_l446_446224


namespace intersection_point_of_diagonals_of_inscribed_quadrilateral_l446_446461

theorem intersection_point_of_diagonals_of_inscribed_quadrilateral {k : Type*} [EuclideanGeometry k]
    {A B C D E F G K : k} (h_circle : Circle k ) (h_tangency_E : TangentLine h_circle A B E)
    (h_tangency_F : TangentLine h_circle B C F) (h_tangency_G : TangentLine h_circle C D G)
    (h_tangency_K : TangentLine h_circle D A K) (h_inscribed : QuadrilateralInscribedInCircle A B C D h_circle) :
    ∃ M : k, Intersection (LineThroughPoints A C) (LineThroughPoints B D) M ∧
             Intersection (LineThroughPoints E G) (LineThroughPoints F K) M :=
sorry

end intersection_point_of_diagonals_of_inscribed_quadrilateral_l446_446461


namespace units_digit_diff_is_seven_l446_446848

noncomputable def units_digit_resulting_difference (a b c : ℕ) (h1 : a = c - 3) :=
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  let difference := original - reversed
  difference % 10

theorem units_digit_diff_is_seven (a b c : ℕ) (h1 : a = c - 3) :
  units_digit_resulting_difference a b c h1 = 7 :=
by sorry

end units_digit_diff_is_seven_l446_446848


namespace jerry_hawk_feathers_l446_446024

-- Define conditions
def initial_total_feathers (H : ℕ) : ℕ := H + 17 * H
def after_giving_feathers (H : ℕ) : ℕ := initial_total_feathers H - 10
def after_selling_half (H : ℕ) : ℕ := (after_giving_feathers H) / 2

-- Define the proof problem
theorem jerry_hawk_feathers : 
  ∃ (H : ℕ), after_selling_half H = 49 ∧ H = 6 :=
by
  existsi 6
  split
  sorry
  rfl

end jerry_hawk_feathers_l446_446024


namespace simplify_expression_l446_446823

theorem simplify_expression (a : ℤ) :
  ((36 * a^9)^4 * (63 * a^9)^4) = a^4 :=
sorry

end simplify_expression_l446_446823


namespace find_lengths_and_cosC_l446_446730

namespace TriangleProblem

variables {A B C : ℝ}
variables {a b c : ℝ}

-- Conditions
axiom cos_2C : cos (2 * C) = -1 / 4
axiom angle_C_range : 0 < C ∧ C < π / 2
axiom side_a : a = 2
axiom sin_A_eq_sin_C : 2 * sin A = sin C

-- Goal: Finding the correct answers
theorem find_lengths_and_cosC (cos_C : ℝ) (b : ℝ) (c : ℝ) :
  cos 2C = -1 / 4 ∧ 0 < C ∧ C < π / 2 ∧ a = 2 ∧ 2 * sin A = sin C →
  cos_C = sqrt 6 / 4 ∧ b = 2 * sqrt 6 ∧ c = 4 :=
sorry

end TriangleProblem

end find_lengths_and_cosC_l446_446730


namespace correct_statement_l446_446680

theorem correct_statement (a b : ℝ) (h_a : a ≥ 0) (h_b : b ≥ 0) : (a ≥ 0 ∧ b ≥ 0) :=
by
  exact ⟨h_a, h_b⟩

end correct_statement_l446_446680


namespace common_tangent_range_of_a_l446_446362

theorem common_tangent_range_of_a 
  (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ 
    (-a / x1^2 = 2 / x2) ∧ 
    (2 * a / x1 = 2 * (log x2) - 2)) 
  ↔ (-2 / Real.exp 1 ≤ a ∧ a < 0) := by
  sorry

end common_tangent_range_of_a_l446_446362


namespace solve_equation_l446_446824

theorem solve_equation (x : ℝ) : 
  16 * (x - 1) ^ 2 - 9 = 0 ↔ (x = 7 / 4 ∨ x = 1 / 4) := by
  sorry

end solve_equation_l446_446824


namespace possible_value_of_a_l446_446066

theorem possible_value_of_a (k : ℕ) (a b : ℝ) 
(h1 : ∑ i in (Finset.range 12), |a - (k + i)| = 358)
(h2 : ∑ i in (Finset.range 12), |b - (k + i)| = 212)
(h3 : a + b = 114.5) : 
a = 190 / 3 :=
by 
  sorry

end possible_value_of_a_l446_446066


namespace statistics_measuring_dispersion_l446_446172

-- Definition of standard deviation
def standard_deviation (X : List ℝ) : ℝ :=
  let mean := (X.sum) / (X.length)
  (X.map (λ x => (x - mean) ^ 2)).sum / X.length

-- Definition of range
def range (X : List ℝ) : ℝ :=
  X.maximum.get - X.minimum.get

-- Definition of median
noncomputable def median (X : List ℝ) : ℝ :=
  let sorted := List.sorted X
  if sorted.length % 2 == 1 then sorted.get (sorted.length / 2)
  else (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2

-- Definition of mean
def mean (X : List ℝ) : ℝ :=
  X.sum / X.length

-- The proof statement
theorem statistics_measuring_dispersion (X : List ℝ) :
  (standard_deviation X ≠ 0 ∧ range X ≠ 0) ∧
  (∀ x : ℝ, x ∈ X ↔ (median X = x ∨ mean X = x)) → true :=
  sorry

end statistics_measuring_dispersion_l446_446172


namespace cyclist_distance_l446_446223

variable (c : ℝ) -- speed of the cyclist in mph
variable (d : ℝ) -- distance covered by the cyclist in miles
variable (t : ℝ) -- time taken in hours
variable (s_car : ℝ) -- speed of the car in mph

-- Conditions
def conditions :=
  t = 8 ∧ -- time taken by both the cyclist and the car
  s_car = 6 ∧ -- speed of the car
  s_car = c + 5 ∧ -- speed of the car is 5 mph faster than the cyclist
  d = c * t -- distance covered by the cyclist

-- Theorem to prove 
theorem cyclist_distance
  (h : conditions) :
  d = 8 := by
  sorry

end cyclist_distance_l446_446223


namespace length_of_BD_l446_446731

-- Define the terms used in the problem
variable (A B C D : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable [LinearOrder B] 

-- Define the distances given in the problem
variable (dist_AC : B) (dist_BC : B) (dist_AB : B) (dist_CD : B)

-- Define the assumptions based on the problem conditions
axiom AC_eq_BC : dist_AC = 10
axiom B_extends_A_D : A ≤ B ∧ B ≤ D
axiom AB_eq_4 : dist_AB = 4
axiom CD_eq_12 : dist_CD = 12

-- Define the target length of BD (x = 4√3 - 2)
def BD : B := 4 * Real.sqrt 3 - 2

-- Prove the equivalence of the computed BD based on the conditions
theorem length_of_BD : dist_CD + dist_AB = dist_AC + dist_BC → BD = 4 * Real.sqrt 3 - 2 :=
 by
  sorry

end length_of_BD_l446_446731


namespace outlinedSquareDigit_l446_446132

-- We define the conditions for three-digit powers of 2 and 3
def isThreeDigitPowerOf (base : ℕ) (n : ℕ) : Prop :=
  let power := base ^ n
  power >= 100 ∧ power < 1000

-- Define the sets of three-digit powers of 2 and 3
def threeDigitPowersOf2 : List ℕ := [128, 256, 512]
def threeDigitPowersOf3 : List ℕ := [243, 729]

-- Define the condition that the digit in the outlined square should be common as a last digit in any power of 2 and 3 that's three-digit long
def commonLastDigitOfPowers (a b : List ℕ) : Option ℕ :=
  let aLastDigits := a.map (λ x => x % 10)
  let bLastDigits := b.map (λ x => x % 10)
  (aLastDigits.inter bLastDigits).head?

theorem outlinedSquareDigit : (commonLastDigitOfPowers threeDigitPowersOf2 threeDigitPowersOf3) = some 3 :=
by
  sorry

end outlinedSquareDigit_l446_446132


namespace sum_of_squares_of_perfect_squares_less_than_500_l446_446148

theorem sum_of_squares_of_perfect_squares_less_than_500 :
  (∑ n in {1, 16, 81, 256}, n) = 354 :=
by {
  sorry
}

end sum_of_squares_of_perfect_squares_less_than_500_l446_446148


namespace jenny_sold_192_packs_l446_446407

-- Define the conditions
def boxes_sold : ℝ := 24.0
def packs_per_box : ℝ := 8.0

-- The total number of packs sold
def total_packs_sold : ℝ := boxes_sold * packs_per_box

-- Proof statement that total packs sold equals 192.0
theorem jenny_sold_192_packs : total_packs_sold = 192.0 :=
by
  sorry

end jenny_sold_192_packs_l446_446407


namespace largest_x_value_l446_446927

theorem largest_x_value (x : ℝ) (h : (x / 3 + 1 / (3 * x)) = 1 / 2) : x ≤ 2 :=
  sorry

example : largest_x_value 2 (by norm_num [show (2 / 3 + 1 / (3 * 2)) = 1 / 2, by ring]) :=
  sorry

end largest_x_value_l446_446927


namespace pete_travel_time_l446_446603

-- Definitions for the given conditions
def map_distance := 5.0          -- in inches
def scale := 0.05555555555555555 -- in inches per mile
def speed := 60.0                -- in miles per hour
def real_distance := map_distance / scale

-- The theorem to state the proof problem
theorem pete_travel_time : 
  real_distance = 90 → -- Based on condition deduced from earlier
  real_distance / speed = 1.5 := 
by 
  intro h1
  rw[h1]
  norm_num
  sorry

end pete_travel_time_l446_446603


namespace total_chickens_and_ducks_l446_446446

-- Definitions based on conditions
def num_chickens : Nat := 45
def more_chickens_than_ducks : Nat := 8
def num_ducks : Nat := num_chickens - more_chickens_than_ducks

-- The proof statement
theorem total_chickens_and_ducks : num_chickens + num_ducks = 82 := by
  -- The actual proof is omitted, only the statement is required
  sorry

end total_chickens_and_ducks_l446_446446


namespace sum_first_10_terms_zero_l446_446047

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : ℕ → α) (d : α) :=
∀ n m, a (n + m) = a n + a m + (m * d)

variables {a : ℕ → α} {d : α}
variable (h1 : d ≠ 0)
variable (h2 : ((a 3)^2 + (a 4)^2 = (a 5)^2 + (a 6)^2))

theorem sum_first_10_terms_zero (h_arith : arithmetic_sequence a d):
  (∑ i in Finset.range 10, a i) = 0 :=
sorry

end sum_first_10_terms_zero_l446_446047


namespace women_population_percentage_l446_446379

theorem women_population_percentage (W M : ℕ) (h : M = 2 * W) : (W : ℚ) / (M : ℚ) = (50 : ℚ) / 100 :=
by
  -- Proof omitted
  sorry

end women_population_percentage_l446_446379


namespace symmetric_point_coordinates_l446_446839

/--
Let A be the point with coordinates (3, -2, 4).
Let M be the reference point with coordinates (0, 1, -3).
Let A' be the symmetric point of A with respect to M.

We aim to prove that the coordinates of A' are (-3, 4, -10), given that M is the midpoint of the line segment AA'.
-/
theorem symmetric_point_coordinates :
  let A := (3, -2, 4)
  let M := (0, 1, -3)
  (∃ (A' : ℝ × ℝ × ℝ), A' = (-3, 4, -10) ∧ (M = ((A.1 + A'.1) / 2, (A.2 + A'.2) / 2, (A.3 + A'.3) / 2))) :=
by {
  let A := (3, -2, 4),
  let M := (0, 1, -3),
  use (-3, 4, -10),
  split,
  refl,
  sorry
}

end symmetric_point_coordinates_l446_446839


namespace value_at_2pi_over_3_minimum_positive_period_monotonically_increasing_interval_l446_446689

section
  variable (x : ℝ)
  def f (x : ℝ) := (Real.sin x)^2 - (Real.cos x)^2 - 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x)

  theorem value_at_2pi_over_3 : f (2 * Real.pi / 3) = 2 := by sorry

  theorem minimum_positive_period : ∀ x, f (x + Real.pi) = f x := by sorry

  theorem monotonically_increasing_interval (k : ℤ) : 
    ∀ x : ℝ, (x ∈ set.Ico (-5 * Real.pi / 6 + k * Real.pi) (-Real.pi / 3 + k * Real.pi)) → 
             ∀ y : ℝ, (y ∈ set.Icc x (step4_limit x (k * Real.pi))) → f(x) ≤ f(y) := by sorry
end

end value_at_2pi_over_3_minimum_positive_period_monotonically_increasing_interval_l446_446689


namespace babblean_words_count_l446_446454

theorem babblean_words_count (alphabet_size word_max_length : ℕ)
  (h1 : alphabet_size = 7)
  (h2 : word_max_length = 4) :
  (alphabet_size ^ 1 + alphabet_size ^ 2 + alphabet_size ^ 3 + alphabet_size ^ 4) = 2800 :=
by
  have h_cases := (alphabet_size ^ 1) + (alphabet_size ^ 2) + (alphabet_size ^ 3) + (alphabet_size ^ 4)
  calc
    h_cases = 7 ^ 1 + 7 ^ 2 + 7 ^ 3 + 7 ^ 4 : by rw [h1]
    ... = 7 + 7 * 7 + 7 * 7 * 7 + 7 * 7 * 7 * 7 : by norm_num
    ... = 2800 : by norm_num
  sorry

end babblean_words_count_l446_446454


namespace field_width_is_250_l446_446889

noncomputable def total_area_km2 := 0.6
noncomputable def total_area_m2 : ℝ := total_area_km2 * 1_000_000
noncomputable def num_fields := 8
noncomputable def length := 300

noncomputable def area_per_field : ℝ := total_area_m2 / num_fields
noncomputable def width := area_per_field / length

theorem field_width_is_250 :
  width = 250 :=
by
  have h1 : total_area_m2 = total_area_km2 * 1_000_000 := rfl
  have h2 : total_area_m2 = 600_000 := by
    rw [h1]
    norm_num
  have h3 : area_per_field = total_area_m2 / num_fields := rfl
  have h4 : area_per_field = 75_000 := by
    rw [h3, h2]
    norm_num
  have h5 : width = area_per_field / length := rfl
  have h6 : width = 250 := by
    rw [h5, h4]
    norm_num
  exact h6

end field_width_is_250_l446_446889


namespace find_p_zero_l446_446430

noncomputable def p : ℝ → ℝ := sorry

lemma p_degree : degree p = 5 := sorry

lemma p_values (k : ℕ) (hk : k ∈ {1, 2, 3, 4, 5}) : p (2 ^ k - 1) = 1 / (2 ^ k - 1) := sorry

theorem find_p_zero : p 0 = 1 / 6174 :=
by
  sorry

end find_p_zero_l446_446430


namespace cube_weight_l446_446222

theorem cube_weight (l1 l2 V1 V2 k : ℝ) (h1: l2 = 2 * l1) (h2: V1 = l1^3) (h3: V2 = (2 * l1)^3) (h4: w2 = 48) (h5: V2 * k = w2) (h6: V1 * k = w1):
  w1 = 6 :=
by
  sorry

end cube_weight_l446_446222


namespace selling_price_of_article_l446_446227

theorem selling_price_of_article (CP : ℕ) (gain_percent : ℕ) (profit : ℕ) (SP : ℕ) 
  (h1 : CP = 10)
  (h2 : gain_percent = 50)
  (h3 : profit = (gain_percent * CP) / 100)
  (h4 : SP = CP + profit) : 
  SP = 15 := 
by
  rw [h1, h2] at h3 ⊢
  dsimp at h3
  linarith

end selling_price_of_article_l446_446227


namespace find_x_l446_446301

theorem find_x (x : ℝ) (h1 : x > 0) 
    (h2 : 1 / 2 * x * (3 * x) = 72) : x = 4 * real.sqrt 3 :=
by
  sorry

end find_x_l446_446301


namespace total_savings_l446_446795

def liam_oranges : ℕ := 40
def liam_price_per_two : ℝ := 2.50
def claire_oranges : ℕ := 30
def claire_price_per_one : ℝ := 1.20

theorem total_savings : (liam_oranges / 2 * liam_price_per_two) + (claire_oranges * claire_price_per_one) = 86 := by
  sorry

end total_savings_l446_446795


namespace find_x_triangle_area_l446_446303

theorem find_x_triangle_area (x : ℝ) (h1 : x > 0) (h2 : (1/2) * x * (3 * x) = 72) : x = 4 * real.sqrt 3 :=
begin
  sorry
end

end find_x_triangle_area_l446_446303


namespace point_in_fourth_quadrant_l446_446748

-- Define the point (2, -3)
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := 2, y := -3 }

-- Define what it means for a point to be in a specific quadrant
def inFirstQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y > 0

def inSecondQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y > 0

def inThirdQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

def inFourthQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y < 0

-- Define the theorem to prove that the point A lies in the fourth quadrant
theorem point_in_fourth_quadrant : inFourthQuadrant A :=
  sorry

end point_in_fourth_quadrant_l446_446748


namespace remainder_of_product_mod_7_l446_446882

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l446_446882


namespace janabel_widgets_sold_15_days_l446_446452

variables (n : ℕ) (a : ℕ)

def widgets_sold_on_day (n : ℕ) : ℕ :=
  2 + (n - 1) * 3

def total_widgets_sold (days : ℕ) : ℕ :=
  (days * (2 + widgets_sold_on_day days)) / 2 + if days = 15 then 1 else 0

theorem janabel_widgets_sold_15_days : total_widgets_sold 15 = 346 :=
by {
  unfold total_widgets_sold,
  unfold widgets_sold_on_day,
  norm_num,
  sorry
}

end janabel_widgets_sold_15_days_l446_446452


namespace increasing_sequence_lambda_range_l446_446693

theorem increasing_sequence_lambda_range (λ : ℝ) 
  (h : ∀ (n : ℕ), 1 ≤ n → (n+1)² + 2 * λ * (n+1) + 1 > n² + 2 * λ * n + 1) : 
  λ > -3 / 2 :=
sorry

end increasing_sequence_lambda_range_l446_446693


namespace find_rate_percent_l446_446549

theorem find_rate_percent
  (SI : ℝ) (P : ℝ) (T : ℝ) (hSI : SI = 250) (hP : P = 1500) (hT : T = 5) :
  ∃ R : ℝ, R ≈ 3.33 :=
by
  sorry

end find_rate_percent_l446_446549


namespace sum_of_valid_fourth_powers_is_354_l446_446151

-- Given conditions
def is_valid_fourth_power (n : ℕ) : Prop := n^4 < 500

-- The main theorem we want to prove
theorem sum_of_valid_fourth_powers_is_354 :
  let valid_n := { n // is_valid_fourth_power n }
  let fourth_powers := { n : ℕ | is_valid_fourth_power n }.to_finset.val
  fourth_powers.sum = 354 :=
by
  -- Proof omitted
  sorry

end sum_of_valid_fourth_powers_is_354_l446_446151


namespace smallest_S_value_l446_446529

noncomputable def smallest_possible_S (n : ℕ) : ℝ :=
1 - 1 / real.exp (real.log 2 / n)

theorem smallest_S_value (n : ℕ) (x : ℕ → ℝ) (hx : ∑ i in finset.range n, x i = 1) :
  let S := finset.sup (finset.range n) (λ i, x i / (1 + ∑ j in finset.range (i + 1), x j)) in
  S ≥ smallest_possible_S n :=
sorry

end smallest_S_value_l446_446529


namespace no_intersection_k_equal_pm1_l446_446674

theorem no_intersection_k_equal_pm1 (k : ℤ) :
  (¬ ∃ (x y : ℝ), x^2 + y^2 = k^2 ∧ x * y = k) → (k = 1 ∨ k = -1) :=
by sorry

end no_intersection_k_equal_pm1_l446_446674


namespace hike_distance_l446_446402

variable (total_hike : ℝ) (car_to_stream : ℝ) (stream_to_meadow : ℝ) (meadow_to_campsite : ℝ)

theorem hike_distance 
  (h_total : total_hike = 0.7)
  (h_car_to_stream : car_to_stream = 0.2)
  (h_stream_to_meadow : stream_to_meadow = 0.4)
  (h_total_dist_before_meadow : car_to_stream + stream_to_meadow = 0.6)
  (h_total_distance_condition : car_to_stream + stream_to_meadow + meadow_to_campsite = total_hike) 
  : meadow_to_campsite = 0.1 :=
by
  rw [h_total_distance_condition, h_total, h_car_to_stream, h_stream_to_meadow, h_total_dist_before_meadow]
  -- here would go the actual proof steps
  sorry

end hike_distance_l446_446402


namespace primes_sum_420_l446_446304

open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def Φ (q : ℕ) (x : ℕ) : ℕ := (Finset.range q).sum (λ i, x^(q-i-1))

theorem primes_sum_420 : 
  (Finset.filter (λ p, is_prime p ∧ 3 ≤ p ∧ p ≤ 100 ∧ (∃ (q : ℕ) (N : ℕ), 
    is_prime q ∧ q % 2 = 1 ∧ N > 0 ∧ (Nat.choose N (Φ q p) % p ≠ 0 ∧ 
    Nat.choose (2 * Φ q p) N % p ≠ 0)) 
  (Finset.range 101)).sum = 420 :=
by
  admit -- Proof to be written

end primes_sum_420_l446_446304


namespace num_ways_appearance_l446_446008

-- Definitions based on conditions:
def num_contestants : ℕ := 5
def num_females : ℕ := 3
def num_males : ℕ := 2

def females : Finset ℕ := {0, 1, 2} -- including female A as 0
def male_positions (l : List ℕ) (n : ℕ) : Bool :=
  ∀ (i : ℕ), (i < l.length ∧ l.get i = n) → 
    (i = 0 ∨ i = l.length - 1 ∨ l.get (i - 1) < 3 ∨ l.get (i + 1) < 3) -- masculines not consecutive.

def female_a_not_first (l : List ℕ) : Bool := l.head.lift false (λ x, x ≠ 0)

def valid_orders : List (List ℕ) := 
  (List.permutations [0, 1, 2, 3, 4]).filter (λ l, 
    male_positions l 3 ∧ male_positions l 4 ∧ female_a_not_first l)

-- Theorem based on the question:
theorem num_ways_appearance :
  valid_orders.length = 60 := sorry

end num_ways_appearance_l446_446008


namespace proof_correct_chemical_information_l446_446100

def chemical_formula_starch : String := "(C_{6}H_{10}O_{5})_{n}"
def structural_formula_glycine : String := "H_{2}N-CH_{2}-COOH"
def element_in_glass_ceramics_cement : String := "Si"
def elements_cause_red_tide : List String := ["N", "P"]

theorem proof_correct_chemical_information :
  chemical_formula_starch = "(C_{6}H_{10}O_{5})_{n}" ∧
  structural_formula_glycine = "H_{2}N-CH_{2}-COOH" ∧
  element_in_glass_ceramics_cement = "Si" ∧
  elements_cause_red_tide = ["N", "P"] :=
by
  sorry

end proof_correct_chemical_information_l446_446100


namespace product_remainder_mod_7_l446_446876

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l446_446876


namespace quilt_shading_fraction_l446_446120

/-- 
Statement:
Given a quilt block made from nine unit squares, where two unit squares are divided diagonally into triangles, 
and one unit square is divided into four smaller equal squares with one of the smaller squares shaded, 
the fraction of the quilt that is shaded is \( \frac{5}{36} \).
-/
theorem quilt_shading_fraction : 
  let total_area := 9 
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2 
  shaded_area / total_area = 5 / 36 :=
by
  -- Definitions based on conditions
  let total_area := 9
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2
  -- The proof statement (fraction of shaded area)
  have h : shaded_area / total_area = 5 / 36 := sorry
  exact h

end quilt_shading_fraction_l446_446120


namespace sufficient_condition_for_q_l446_446667

def p (a : ℝ) : Prop := a ≥ 0
def q (a : ℝ) : Prop := a^2 + a ≥ 0

theorem sufficient_condition_for_q (a : ℝ) : p a → q a := by 
  sorry

end sufficient_condition_for_q_l446_446667


namespace deck_initial_card_count_l446_446840

variable (r b : ℕ)

-- Conditions
def condition1 : Prop := r / (r + b) = 1 / 4
def condition2 : Prop := (r + 6) / (r + b + 6) = 1 / 3

-- Problem Statement
def initial_deck_card_count : (r b : ℕ) → Prop := 
    ∀ (r b : ℕ), condition1 r b → condition2 r b → (r + b = 48)

-- Proof placeholder
theorem deck_initial_card_count (r b : ℕ) : initial_deck_card_count r b := sorry

end deck_initial_card_count_l446_446840


namespace equilateral_triangle_side_length_l446_446329

theorem equilateral_triangle_side_length (x y z s : ℝ) 
  (h1 : s^2 = 4 + x^2)
  (h2 : s^2 = 9 + y^2)
  (h3 : s^2 = 12 + z^2) :
  s = sqrt 13 :=
by
  sorry

end equilateral_triangle_side_length_l446_446329


namespace polynomial_divisibility_l446_446055

noncomputable def divides (P Q : Polynomial ℤ) : Prop :=
  ∃ A : Polynomial ℤ, Q = A * P

theorem polynomial_divisibility (P Q : Polynomial ℤ) 
  (hP_nonconst : P.degree > 0) (hP_monic : P.leadingCoeff = 1)
  (hQ_nonconst : Q.degree > 0) (hQ_monic : Q.leadingCoeff = 1)
  (h_infinite_divisors : ∀ n : ℤ, ∃ k : ℤ, P.eval n \| Q.eval n) :
  divides P Q :=
sorry

end polynomial_divisibility_l446_446055


namespace two_boats_initial_distance_l446_446898

def initial_distance (speed1 speed2 time_before_collision distance_before_collision : ℝ) : ℝ :=
  let combined_speed := speed1 + speed2
  let distance_per_minute := combined_speed / 60
  distance_per_minute * time_before_collision + distance_before_collision

theorem two_boats_initial_distance :
  initial_distance 4 20 (1 / 60) 0.4 = 0.8 :=
sorry

end two_boats_initial_distance_l446_446898


namespace circle_radii_l446_446259

noncomputable def radius_B_C (r_B r_C : ℝ) : Prop :=
  (r_B = 1.5) ∧ (r_C = 3)

theorem circle_radii (r_B r_C d : ℝ) : 
  (r_C = 2 * r_B) → 
  (d = 6) → 
  let EA := 3 + r_B 
  let EB := 3 + r_C 
  let FH := r_B + r_C 
  (EA^2 + (d - r_B)^2 = d^2) → 
  radius_B_C r_B r_C :=
by
  simplify_eq
  sorry

end circle_radii_l446_446259


namespace total_number_of_students_l446_446137

/-- The total number of high school students in the school given sampling constraints. -/
theorem total_number_of_students (F1 F2 F3 : ℕ) (sample_size : ℕ) (consistency_ratio : ℕ) :
  F2 = 300 ∧ sample_size = 45 ∧ (F1 / F3) = 2 ∧ 
  (20 + 10 + (sample_size - 30)) = sample_size → F1 + F2 + F3 = 900 :=
by
  sorry

end total_number_of_students_l446_446137


namespace cyclic_quadrilateral_bisects_AD_l446_446556

theorem cyclic_quadrilateral_bisects_AD
  (A B C D P M : EuclideanGeometry.Point)
  (h_cyclic : EuclideanGeometry.cyclic_quadrilateral A B C D)
  (h_perpendicular_diagonals : ∠A P B = 90° ∧ ∠C P D = 90°)
  (h_intersection : EuclideanGeometry.intersects P A C D B)
  (h_perpendicular_to_BC : EuclideanGeometry.perpendicular P M (EuclideanGeometry.line_through B C))
  (h_M_on_AD : EuclideanGeometry.lies_on M (EuclideanGeometry.segment A D)) :
  EuclideanGeometry.segment AM = EuclideanGeometry.segment MD :=
sorry

end cyclic_quadrilateral_bisects_AD_l446_446556


namespace perpendicular_graphs_solve_a_l446_446497

theorem perpendicular_graphs_solve_a (a : ℝ) : 
  (∀ x y : ℝ, 2 * y + x + 3 = 0 → 3 * y + a * x + 2 = 0 → 
  ∀ m1 m2 : ℝ, (y = m1 * x + b1 → m1 = -1 / 2) →
  (y = m2 * x + b2 → m2 = -a / 3) →
  m1 * m2 = -1) → a = -6 :=
by
  sorry

end perpendicular_graphs_solve_a_l446_446497


namespace cost_per_hour_in_excess_l446_446487

theorem cost_per_hour_in_excess {x : ℝ} :
  (20.0 + 7*x) / 9 = 3.5833333333333335 → x = 1.75 :=
by
  assume h : (20.0 + 7*x) / 9 = 3.5833333333333335
  sorry

end cost_per_hour_in_excess_l446_446487


namespace tamia_total_slices_and_pieces_l446_446086

-- Define the conditions
def num_bell_peppers : ℕ := 5
def slices_per_pepper : ℕ := 20
def num_large_slices : ℕ := num_bell_peppers * slices_per_pepper
def num_half_slices : ℕ := num_large_slices / 2
def small_pieces_per_slice : ℕ := 3
def num_small_pieces : ℕ := num_half_slices * small_pieces_per_slice
def num_uncut_slices : ℕ := num_half_slices

-- Define the total number of pieces and slices
def total_pieces_and_slices : ℕ := num_uncut_slices + num_small_pieces

-- State the theorem and provide a placeholder for the proof
theorem tamia_total_slices_and_pieces : total_pieces_and_slices = 200 :=
by
  sorry

end tamia_total_slices_and_pieces_l446_446086


namespace Guido_costs_42840_l446_446799

def LightningMcQueenCost : ℝ := 140000
def MaterCost : ℝ := 0.1 * LightningMcQueenCost
def SallyCostBeforeModifications : ℝ := 3 * MaterCost
def SallyCostAfterModifications : ℝ := SallyCostBeforeModifications + 0.2 * SallyCostBeforeModifications
def GuidoCost : ℝ := SallyCostAfterModifications - 0.15 * SallyCostAfterModifications

theorem Guido_costs_42840 :
  GuidoCost = 42840 :=
sorry

end Guido_costs_42840_l446_446799


namespace carolyn_removes_sum_l446_446830

/-- Suppose n = 10 and Carolyn and Paul follow the updated rules of their number removal game:
- Carolyn always goes first and on each turn, she must remove the smallest number from the list that has
  at least one positive divisor other than itself remaining.
- Paul on his turn removes all positive divisors of the number that Carolyn just removed.
- The game ends when Carolyn cannot remove a number, following which Paul removes all remaining numbers.
- If Carolyn removes the integer 3 on her first turn, determine the sum of the numbers that Carolyn removes.
-/
theorem carolyn_removes_sum :
  let initial_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let carolyn_moves := [3, 6, 8, 9, 10]
  let sum_carolyn_removes := 3 + 6 + 8 + 9 + 10
  initial_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
  carolyn_moves = [3, 6, 8, 9, 10] ∧
  sum_carolyn_removes = 36 :=
by
  sorry

end carolyn_removes_sum_l446_446830


namespace find_parallel_lambda_l446_446703

theorem find_parallel_lambda 
  (a b : ℝ × ℝ)
  (h₁ : a = (1, 3))
  (h₂ : b = (2, 1))
  (h_parallel : ∃ k : ℝ, (1 + 2 * 2, 3 + 2 * 1) = k • (3 + λ * 2, 9 + λ * 1)) :
  λ = 6 :=
by
  -- The mathematical proof steps would go here
  sorry

end find_parallel_lambda_l446_446703


namespace complex_identity_l446_446780

theorem complex_identity 
  {n : ℕ} (h : n ≥ 2) (A B : Fin n → ℂ) :
  ∑ k : Fin n, (∏ j in Finset.univ.filter (λ j, j ≠ k), (A k + B j)) / (∏ j in Finset.univ.filter (λ j, j ≠ k), (A k - A j))
  = ∑ k : Fin n, (∏ j in Finset.univ.filter (λ j, j ≠ k), (B k + A j)) / (∏ j in Finset.univ.filter (λ j, j ≠ k), (B k - B j)) :=
by sorry

end complex_identity_l446_446780


namespace unique_7tuple_exists_l446_446634

theorem unique_7tuple_exists 
  (x : Fin 7 → ℝ) 
  (h : (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7) 
  : ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7 :=
sorry

end unique_7tuple_exists_l446_446634


namespace sum_fourth_powers_below_500_is_354_l446_446155

noncomputable def sum_fourth_powers_below_500 : ℕ :=
  ∑ n in Finset.Icc 1 4, n^4

theorem sum_fourth_powers_below_500_is_354 : sum_fourth_powers_below_500 = 354 :=
by
  sorry

end sum_fourth_powers_below_500_is_354_l446_446155


namespace factorize_expression_l446_446631

-- The problem is about factorizing the expression x^3y - xy
theorem factorize_expression (x y : ℝ) : x^3 * y - x * y = x * y * (x - 1) * (x + 1) := 
by sorry

end factorize_expression_l446_446631


namespace partitions_eq_l446_446582

-- Definitions of the required concepts: partitions of n and conditions
def is_partition (n : ℕ) (parts : List ℕ) : Prop :=
  parts.sum = n

def has_k_terms (parts : List ℕ) (k : ℕ) : Prop :=
  parts.length = k

def largest_element_is_k (parts : List ℕ) (k : ℕ) : Prop :=
  parts.maximum = some k

-- The proof statement
theorem partitions_eq {n k : ℕ} (h₀ : 0 < n) (h₁ : 0 < k) :
  (card {parts // is_partition n parts ∧ has_k_terms parts k}) = 
  (card {parts // is_partition n parts ∧ largest_element_is_k parts k}) :=
sorry

end partitions_eq_l446_446582


namespace num_values_of_n_for_prime_g_l446_446052

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ i in Nat.divisors n, i

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∈ Finset.range (n-1).filter (λ d, d > 1), n % m ≠ 0

def g (n : ℕ) : ℕ :=
  sum_of_divisors n

theorem num_values_of_n_for_prime_g :
  (Finset.card ((Finset.range 31).filter (λ n, is_prime (g n)))) = 10 :=
by
  sorry

end num_values_of_n_for_prime_g_l446_446052


namespace label_elements_of_B_l446_446779

open Set

variable {B : Type*}
variable [Fintype B]

noncomputable def n : ℕ := sorry

variable (A : Fin (2 * n + 1) → Finset B)

-- Conditions
def condition_1 : Prop :=
  ∀ i : Fin (2 * n + 1), (A i).card = 2 * n

def condition_2 : Prop :=
  ∀ i j : Fin (2 * n + 1), i ≠ j → (A i ∩ A j).card = 1

def condition_3 : Prop :=
  ∀ b ∈ ⋃ i, A i, ∃ i j : Fin (2 * n + 1), i < j ∧ b ∈ A i ∧ b ∈ A j

-- Mathematical equivalence: For n being even, a specific labeling exists
theorem label_elements_of_B :
  condition_1 A →
  condition_2 A →
  condition_3 A →
  ∃ label : B → Bool, ∀ i : Fin (2 * n + 1), (A i).filter (λ b, label b = ff).card = n ↔ (even n) :=
by
  intros h1 h2 h3
  sorry

end label_elements_of_B_l446_446779


namespace red_card_probability_l446_446964

theorem red_card_probability :
  let total_cards := 8
  let bb_cards := 4
  let br_cards := 2
  let rr_cards := 2
  let total_red_sides := 2 * br_cards + 2 * rr_cards
  let red_sides_from_rr := 2 * rr_cards
  in (total_red_sides > 0) → (red_sides_from_rr / total_red_sides) = (2 / 3) := by
  have total_cards := 8
  have bb_cards := 4
  have br_cards := 2
  have rr_cards := 2
  have total_red_sides := 2 * br_cards + 2 * rr_cards
  have red_sides_from_rr := 2 * rr_cards
  have red_sides_from_rr / total_red_sides = 4 / 6
  have (4 / 6) = (2 / 3)
  sorry

end red_card_probability_l446_446964


namespace total_cost_of_umbrellas_l446_446761

theorem total_cost_of_umbrellas : 
  ∀ (h_umbrellas c_umbrellas cost_per_umbrella : ℕ),
  h_umbrellas = 2 → 
  c_umbrellas = 1 → 
  cost_per_umbrella = 8 → 
  (h_umbrellas + c_umbrellas) * cost_per_umbrella = 24 :=
by 
  intros h_umbrellas c_umbrellas cost_per_umbrella h_eq c_eq cost_eq
  rw [h_eq, c_eq, cost_eq]
  sorry

end total_cost_of_umbrellas_l446_446761


namespace probability_nonagon_diagonal_intersect_l446_446955

theorem probability_nonagon_diagonal_intersect (n : ℕ) (h_n : n = 9) :
  let diagonals := (n.choose 2) - n,
      total_diagonals_pairs := (diagonals.choose 2),
      intersecting_pairs := (n.choose 4)
  in (intersecting_pairs : ℚ) / total_diagonals_pairs = 14 / 39 :=
by {
  sorry
}

end probability_nonagon_diagonal_intersect_l446_446955


namespace complement_union_eq_complement_l446_446788

open Set

variable (U : Set ℤ) 
variable (A : Set ℤ) 
variable (B : Set ℤ)

theorem complement_union_eq_complement : 
  U = {-2, -1, 0, 1, 2, 3} →
  A = {-1, 2} →
  B = {x | x^2 - 4*x + 3 = 0} →
  (U \ (A ∪ B)) = {-2, 0} :=
by
  intros hU hA hB
  -- sorry to skip the proof
  sorry

end complement_union_eq_complement_l446_446788


namespace bookshelf_arrangement_l446_446383

theorem bookshelf_arrangement:
  ∀ (M : ℕ) (H : ℕ), M = 4 → H = 6 →
  (∃ middle : ℕ → middle ∈ finset.range H → 6 * nat.factorial (M + H - 1) = 30240) :=
begin
  intros M H hM hH,
  use [6, sorry]
end

end bookshelf_arrangement_l446_446383


namespace probability_three_cards_l446_446894

theorem probability_three_cards (deck_size : ℕ) (kings : ℕ) (aces : ℕ) (tens : ℕ) :
  deck_size = 52 →
  kings = 4 →
  aces = 4 →
  tens = 4 →
  ((kings.toRat / deck_size.toRat) * 
   (aces.toRat / (deck_size - 1).toRat) * 
   (tens.toRat / (deck_size - 2).toRat)) = (8 / 16575) := 
by
  intro h1 h2 h3 h4
  sorry

end probability_three_cards_l446_446894


namespace emily_remainder_l446_446628

theorem emily_remainder (c d : ℤ) (h1 : c % 60 = 53) (h2 : d % 42 = 35) : (c + d) % 21 = 4 :=
by
  sorry

end emily_remainder_l446_446628


namespace polynomial_remainder_l446_446769

theorem polynomial_remainder (P : ℝ → ℝ) :
  P 49 = 61 → P 61 = 49 → ∃a b, ∀ x, P(x) = (x - 49) * (x - 61) * Q(x) + ax + b → P(x) % ((x - 49) * (x - 61)) = -x + 112 :=
by
  intros h1 h2
  use -1
  use 112
  sorry

end polynomial_remainder_l446_446769


namespace points_satisfy_l446_446346

theorem points_satisfy (x y : ℝ) : 
  (y^2 - y = x^2 - x) ↔ (y = x ∨ y = 1 - x) :=
by sorry

end points_satisfy_l446_446346


namespace wire_total_length_l446_446987

theorem wire_total_length (a b c total_length : ℕ) (h1 : a = 7) (h2 : b = 3) (h3 : c = 2) (h4 : c * 16 = 32) :
  total_length = (a + b + c) * (16 / c) :=
by
  have h5 : c = 2 := by rw [←nat.add_assoc, add_comm, h3]
  have h6 : total_length = (a + b + c) * 8 := sorry
  exact h6

end wire_total_length_l446_446987


namespace temperature_on_friday_l446_446482

def temperatures (M T W Th F : ℝ) : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 40 ∧
  M = 42

theorem temperature_on_friday (M T W Th F : ℝ) (h : temperatures M T W Th F) : 
  F = 10 :=
  by
    -- problem statement
    sorry

end temperature_on_friday_l446_446482


namespace initial_incorrect_average_l446_446481

theorem initial_incorrect_average :
  let avg_correct := 24
  let incorrect_insertion := 26
  let correct_insertion := 76
  let n := 10  
  let correct_sum := avg_correct * n
  let incorrect_sum := correct_sum - correct_insertion + incorrect_insertion   
  avg_correct * n - correct_insertion + incorrect_insertion = incorrect_sum →
  incorrect_sum / n = 19 :=
by 
  sorry

end initial_incorrect_average_l446_446481


namespace perpendicular_lines_slope_l446_446541

theorem perpendicular_lines_slope (a : ℝ) (h1 :  a * (a + 2) = -1) : a = -1 :=
by 
-- Perpendicularity condition given
sorry

end perpendicular_lines_slope_l446_446541


namespace pyramid_volume_l446_446230

theorem pyramid_volume (a b c : ℝ) (h_base : a = 9) (w_base : b = 12) 
  (edge_length : c = 15) : 
  (1 / 3) * (a * b) * (real.sqrt (c^2 - (real.sqrt (a^2 + b^2) / 2)^2)) = 468 := 
by
  sorry

end pyramid_volume_l446_446230


namespace decreasing_interval_log_sin_l446_446501

theorem decreasing_interval_log_sin (k : ℤ) : 
  ∃ I : Set ℝ, I = (Set.Ioo (2 * k * Real.pi + 5 * Real.pi / 4) (2 * k * Real.pi + 7 * Real.pi / 4)) ∧ 
  ∀ x : ℝ, x ∈ I → 
  (∀ y1 y2, y1 < y2 → log (2 : ℝ) (sin (Real.pi / 4 - y1)) > log (2 : ℝ) (sin (Real.pi / 4 - y2))) :=
by 
sorry

end decreasing_interval_log_sin_l446_446501


namespace painting_ways_l446_446568

theorem painting_ways (grid : matrix (fin 3) (fin 3) bool) : 
      (∀ i j, grid i j = tt → (∀ k < i, grid k j = tt) ∧ (∀ l < j, grid i l = tt)) → 
      ∃ n, n = 18 :=
sorry

end painting_ways_l446_446568


namespace distinct_naturals_and_power_of_prime_l446_446768

theorem distinct_naturals_and_power_of_prime (a b : ℕ) (p k : ℕ) (h1 : a ≠ b) (h2 : a^2 + b ∣ b^2 + a) (h3 : ∃ (p : ℕ) (k : ℕ), b^2 + a = p^k) : (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2) :=
sorry

end distinct_naturals_and_power_of_prime_l446_446768


namespace toy_sword_cost_l446_446036

theorem toy_sword_cost (L S : ℕ) (play_dough_cost total_cost : ℕ) :
    L = 250 →
    play_dough_cost = 35 →
    total_cost = 1940 →
    3 * L + 7 * S + 10 * play_dough_cost = total_cost →
    S = 120 :=
by
  intros hL h_play_dough_cost h_total_cost h_eq
  sorry

end toy_sword_cost_l446_446036


namespace carter_average_goals_l446_446257

theorem carter_average_goals (C : ℝ)
  (h1 : C + (1 / 2) * C + (C - 3) = 7) : C = 4 :=
by
  sorry

end carter_average_goals_l446_446257


namespace dispersion_measures_correct_l446_446190

-- Define a sample data set
variable {x : ℕ → ℝ}
variable {n : ℕ}

-- Definitions of the four statistics
def standard_deviation (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def median (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def range (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def mean (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry

-- Definition of measures_dispersion function
def measures_dispersion (stat : ℕ → ℝ → ℝ) (x : ℕ → ℝ) (n : ℕ) : Prop :=
  sorry -- Define what it means for a statistic to measure dispersion

-- Problem statement in Lean
theorem dispersion_measures_correct :
  measures_dispersion standard_deviation x n ∧
  measures_dispersion range x n ∧
  ¬measures_dispersion median x n ∧
  ¬measures_dispersion mean x n :=
by sorry

end dispersion_measures_correct_l446_446190


namespace total_cost_of_feeding_the_fish_l446_446392

theorem total_cost_of_feeding_the_fish
  (num_goldfish : ℕ) (num_koi : ℕ) (num_guppies : ℕ)
  (goldfish_food_rate : ℕ → ℝ) (koi_food_rate : ℕ → ℝ) (guppy_food_rate : ℕ → ℝ)
  (goldfish_special_ratio koi_special_ratio guppy_special_ratio : ℝ)
  (goldfish_special_cost koi_special_cost guppy_special_cost : ℝ)
  (regular_food_cost : ℝ) :
  num_goldfish = 50 → num_koi = 30 → num_guppies = 20 →
  goldfish_food_rate num_goldfish = 1.5 → koi_food_rate num_koi = 2.5 → guppy_food_rate num_guppies = 0.75 →
  goldfish_special_ratio = 0.25 → koi_special_ratio = 0.4 → guppy_special_ratio = 0.1 →
  goldfish_special_cost = 3 → koi_special_cost = 4 → guppy_special_cost = 4 →
  regular_food_cost = 2 →
  let goldfish_special_count := float.floor ((goldfish_special_ratio * num_goldfish).to_float) in
  let koi_special_count := float.floor ((koi_special_ratio * num_koi).to_float) in
  let guppy_special_count := float.floor ((guppy_special_ratio * num_guppies).to_float) in
  let goldfish_regular_count := num_goldfish - goldfish_special_count in
  let koi_regular_count := num_koi - koi_special_count in
  let guppy_regular_count := num_guppies - guppy_special_count in
  let goldfish_special_daily_cost := goldfish_special_count * goldfish_food_rate num_goldfish * goldfish_special_cost in
  let koi_special_daily_cost := koi_special_count * koi_food_rate num_koi * koi_special_cost in
  let guppy_special_daily_cost := guppy_special_count * guppy_food_rate num_guppies * guppy_special_cost in
  let goldfish_regular_daily_cost := goldfish_regular_count * goldfish_food_rate num_goldfish * regular_food_cost in
  let koi_regular_daily_cost := koi_regular_count * koi_food_rate num_koi * regular_food_cost in
  let guppy_regular_daily_cost := guppy_regular_count * guppy_food_rate num_guppies * regular_food_cost in
  let total_daily_cost := goldfish_special_daily_cost + koi_special_daily_cost + guppy_special_daily_cost + goldfish_regular_daily_cost + koi_regular_daily_cost + guppy_regular_daily_cost in
  let total_monthly_cost := total_daily_cost * 30 in
  total_monthly_cost = 12375 :=
by
  intros
  sorry -- proof not required

end total_cost_of_feeding_the_fish_l446_446392


namespace find_n_of_divisors_product_l446_446507

theorem find_n_of_divisors_product (n : ℕ) (h1 : 0 < n)
  (h2 : ∏ k in (finset.filter (λ d : ℕ, d ∣ n) (finset.range (n + 1))), d = 1024) :
  n = 16 :=
sorry

end find_n_of_divisors_product_l446_446507


namespace find_n_of_divisors_product_l446_446505

theorem find_n_of_divisors_product (n : ℕ) (h1 : 0 < n)
  (h2 : ∏ k in (finset.filter (λ d : ℕ, d ∣ n) (finset.range (n + 1))), d = 1024) :
  n = 16 :=
sorry

end find_n_of_divisors_product_l446_446505


namespace product_remainder_mod_7_l446_446862

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l446_446862


namespace increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446912

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446912


namespace f_2012_sum_l446_446677

noncomputable def f : ℤ → ℝ
axiom f_domain (x : ℤ) : f(x) = f(x-1) + f(x+1)
axiom f_neg1 : f (-1) = 6
axiom f_1 : f (1) = 7

theorem f_2012_sum : f 2012 + f (-2012) = -13 := sorry

end f_2012_sum_l446_446677


namespace robe_savings_before_l446_446074

noncomputable def repair_fee : ℕ := 10
noncomputable def corner_light_cost : ℕ := 2 * repair_fee
noncomputable def brake_disk_cost : ℕ := 3 * corner_light_cost
noncomputable def floor_mats_cost : ℕ := brake_disk_cost
noncomputable def steering_wheel_cost : ℕ := corner_light_cost / 2
noncomputable def seat_covers_cost : ℕ := 2 * floor_mats_cost

noncomputable def total_expenses : ℕ :=
  repair_fee +
  corner_light_cost +
  2 * brake_disk_cost +
  floor_mats_cost +
  steering_wheel_cost +
  seat_covers_cost

theorem robe_savings_before :
  total_expenses = 340 →
  ∀ (remaining_savings : ℕ = 480), total_savings = 820 :=
by
  intros h_total_expenses h_remaining_savings
  let total_savings := h_remaining_savings + total_expenses
  sorry

end robe_savings_before_l446_446074


namespace red_fraction_after_changes_l446_446732

theorem red_fraction_after_changes
  (initial_blue_fraction initial_red_fraction : ℚ)
  (h1 : initial_blue_fraction = 3 / 7)
  (h2 : initial_red_fraction = 4 / 7) :
  let new_red_fraction := 3 * initial_red_fraction
      new_blue_fraction := initial_blue_fraction / 2
      total_marbles := new_red_fraction + new_blue_fraction in
  new_red_fraction / total_marbles = 8 / 9 :=
by
  sorry

end red_fraction_after_changes_l446_446732


namespace coeff_x10_in_expansion_l446_446140

theorem coeff_x10_in_expansion : 
  ∀ (x : ℝ), (∑ k in finset.range (11 + 1), (nat.choose 11 k) * (x ^ (11 - k)) * (1 : ℝ)^k) = 11 * (x ^ 10) + ∑ k in (finset.range (11 + 1)).filter (λ i, i ≠ 1), (nat.choose 11 k) * (x ^ (11 - k)) * (1 : ℝ)^k :=
begin
  sorry
end

end coeff_x10_in_expansion_l446_446140


namespace probability_F_l446_446225

theorem probability_F {P : (Char → ℚ)} (hD : P 'D' = 2 / 5) (hE : P 'E' = 1 / 3) (hTotal : P 'D' + P 'E' + P 'F' = 1) :
  P 'F' = 4 / 15 := 
begin
  sorry
end

end probability_F_l446_446225


namespace product_of_divisors_eq_1024_l446_446521

theorem product_of_divisors_eq_1024 (n : ℕ) (h1 : 0 < n) (h2 : ∏ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446521


namespace centroid_M1_midpoint_MX_l446_446564

variables {V : Type*} [EuclideanSpace V]
variables {A B C X M A1 B1 C1 M1 : V}

-- Conditions
variable [centroid_M : centroid A B C M]
variable [is_parallel_A1X_AM : parallel A1 X A M]
variable [is_parallel_B1X_BM : parallel B1 X B M]
variable [is_parallel_C1X_CM : parallel C1 X C M]
variable [A1_on_line_BC : collinear A1 B C]
variable [B1_on_line_CA : collinear B1 C A]
variable [C1_on_line_AB : collinear C1 A B]
variable [X_arbitrary : point X]

theorem centroid_M1_midpoint_MX :
  centroid A1 B1 C1 M1 ∧ is_midpoint M X M1 :=
sorry

end centroid_M1_midpoint_MX_l446_446564


namespace exists_set_M_l446_446459

theorem exists_set_M (n : ℕ) (h : 2 ≤ n) : 
  ∃ M : Finset ℕ, M.card = n ∧ ∀ a b ∈ M, a ≠ b → (a - b).natAbs ∣ (a + b) :=
by
  sorry

end exists_set_M_l446_446459


namespace new_average_after_increase_l446_446563

theorem new_average_after_increase
  {s : Finset ℝ} 
  (h1 : s.card = 10)
  (h2 : s.sum / 10 = 6.2)
  (h3 : ∃ x ∈ s, s.erase x).sum + 8 = s.sum + 8 :=
  sorry

end new_average_after_increase_l446_446563


namespace exercise_l446_446774

variables (A B C D E : Type)
variables [inner_product_space ℝ A] [metric_space A] -- Assuming an inner product for 90 degree definition
variables [affine_space A]

noncomputable def right_triangle (A B C : A) : Prop :=
∠ BCA = π / 2

noncomputable def point_D (A C D B : A) : Prop :=
dist C D = dist C B ∧ (C - A) • (D - A) > 0 -- C is between A and D

noncomputable def perpendicular_through_D (D A B E C : A) : Prop :=
is_perp (D - B) (A - B) ∧ (D - C) • (E - C) = 0 -- DE perpendicular to AB

theorem exercise
  (h_right_triangle : right_triangle A B C)
  (h_point_D : point_D A C D B)
  (h_perpendicular : perpendicular_through_D D A B E C) :
  dist A C = dist C E := sorry

end exercise_l446_446774


namespace avg_hits_next_6_games_each_player_l446_446218

-- Definitions based on the conditions
def total_games : ℕ := 5
def total_team_players : ℕ := 11
def avg_hits_per_game_per_team : ℕ := 15
def best_player_total_hits : ℕ := 25
def additional_games : ℕ := 6

-- Proof statement
theorem avg_hits_next_6_games_each_player 
  (total_games : ℕ)
  (total_team_players : ℕ)
  (avg_hits_per_game_per_team : ℕ)
  (best_player_total_hits : ℕ)
  (additional_games : ℕ) :
  let best_player_avg_hits := best_player_total_hits / total_games in
  let rest_team_hits := avg_hits_per_game_per_team - best_player_avg_hits in
  let rest_player_avg_hits := rest_team_hits / (total_team_players - 1) in
  let avg_hits_next_6_games_each_player := rest_player_avg_hits * additional_games in
  avg_hits_next_6_games_each_player = 6 :=
by
  sorry

end avg_hits_next_6_games_each_player_l446_446218


namespace egg_processing_l446_446002

theorem egg_processing (E : ℕ) 
  (h1 : (24 / 25) * E + 12 = (99 / 100) * E) : 
  E = 400 :=
sorry

end egg_processing_l446_446002


namespace min_flowers_for_bouquets_l446_446000

open Classical

noncomputable def minimum_flowers (types : ℕ) (flowers_for_bouquet : ℕ) (bouquets : ℕ) : ℕ := 
  sorry

theorem min_flowers_for_bouquets : minimum_flowers 6 5 10 = 70 := 
  sorry

end min_flowers_for_bouquets_l446_446000


namespace find_x_l446_446300

theorem find_x (x : ℝ) (h1 : x > 0) 
    (h2 : 1 / 2 * x * (3 * x) = 72) : x = 4 * real.sqrt 3 :=
by
  sorry

end find_x_l446_446300


namespace minimum_area_AI1I2_l446_446770

-- Define the side lengths of the triangle
def AB : ℝ := 30
def BC : ℝ := 32
def AC : ℝ := 34

-- Define points and their properties
axiom point_X : ℝ
axiom X_in_BC : 0 < point_X ∧ point_X < BC  -- X lies in the interior of BC

-- Define incenters I1 and I2 of triangles ABX and ACX
noncomputable def I1 : Type := sorry
noncomputable def I2 : Type := sorry

-- Define the function that represents the area of triangle AI1I2
noncomputable def area_AI1I2 (X : ℝ) : ℝ := sorry

-- The minimum area of triangle AI1I2
theorem minimum_area_AI1I2 : ∃ X, X_in_BC → area_AI1I2 X = 126 :=
sorry

end minimum_area_AI1I2_l446_446770


namespace option_C_option_D_l446_446330

def star (a b : ℕ) : ℕ := a ^ b

theorem option_C (a b m : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : m > 0) : 
  (star a b) ^ m = star a (b * m) :=
by
  sorry

theorem option_D (a m n : ℕ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) : 
  (star a m) ^ n = star a^n (m^n) :=
by
  sorry

end option_C_option_D_l446_446330


namespace sum_of_subsets_l446_446395

open Finset

theorem sum_of_subsets (n : ℕ) (a : Fin n → ℕ) :
    let total_subset_sums := (powerset (univ : Finset (Fin n)).val).sum (λ s, (s.image a).sum)
     in total_subset_sums = 2^(n-1) * (range n).sum (λ i, a (⟨i, sorry⟩)) :=
by
  sorry

end sum_of_subsets_l446_446395


namespace numberOfWaysToChooseLeadership_is_correct_l446_446984

noncomputable def numberOfWaysToChooseLeadership (totalMembers : ℕ) : ℕ :=
  let choicesForGovernor := totalMembers
  let remainingAfterGovernor := totalMembers - 1

  let choicesForDeputies := Nat.choose remainingAfterGovernor 3
  let remainingAfterDeputies := remainingAfterGovernor - 3

  let choicesForLieutenants1 := Nat.choose remainingAfterDeputies 3
  let remainingAfterLieutenants1 := remainingAfterDeputies - 3

  let choicesForLieutenants2 := Nat.choose remainingAfterLieutenants1 3
  let remainingAfterLieutenants2 := remainingAfterLieutenants1 - 3

  let choicesForLieutenants3 := Nat.choose remainingAfterLieutenants2 3
  let remainingAfterLieutenants3 := remainingAfterLieutenants2 - 3

  let choicesForSubordinates : List ℕ := 
    (List.range 8).map (λ i => Nat.choose (remainingAfterLieutenants3 - 2*i) 2)

  choicesForGovernor 
  * choicesForDeputies 
  * choicesForLieutenants1 
  * choicesForLieutenants2 
  * choicesForLieutenants3 
  * List.prod choicesForSubordinates

theorem numberOfWaysToChooseLeadership_is_correct : 
  numberOfWaysToChooseLeadership 35 = 
    35 * Nat.choose 34 3 * Nat.choose 31 3 * Nat.choose 28 3 * Nat.choose 25 3 *
    Nat.choose 16 2 * Nat.choose 14 2 * Nat.choose 12 2 * Nat.choose 10 2 *
    Nat.choose 8 2 * Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2 :=
by
  sorry

end numberOfWaysToChooseLeadership_is_correct_l446_446984


namespace tamia_bell_pepper_pieces_l446_446088

def total_pieces (n k p : Nat) : Nat :=
  let slices := n * k
  let half_slices := slices / 2
  let smaller_pieces := half_slices * p
  let total := half_slices + smaller_pieces
  total

theorem tamia_bell_pepper_pieces :
  total_pieces 5 20 3 = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l446_446088


namespace prob_same_color_l446_446377

-- Define the given conditions
def total_pieces : ℕ := 15
def black_pieces : ℕ := 6
def white_pieces : ℕ := 9
def prob_two_black : ℚ := 1/7
def prob_two_white : ℚ := 12/35

-- Define the statement to be proved
theorem prob_same_color : prob_two_black + prob_two_white = 17 / 35 := by
  sorry

end prob_same_color_l446_446377


namespace sin_lt_a_solution_set_l446_446651

theorem sin_lt_a_solution_set (a : ℝ) (theta : ℝ) (n : ℤ) :
  -1 < a → a < 0 → theta = Real.arcsin a →
  {x : ℝ | ∃ n : ℤ, (2 * n - 1) * Real.pi - theta < x ∧ x < 2 * n * Real.pi + theta} = 
  { x | sin x < a } :=
by
  sorry

end sin_lt_a_solution_set_l446_446651


namespace equal_points_probability_not_always_increasing_l446_446909

theorem equal_points_probability_not_always_increasing 
  (p q : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ q) (h₂ : q ≤ 1) :
  ¬ ∀ p₀ p₁, 0 ≤ p₀ ∧ p₀ ≤ p₁ ∧ p₁ ≤ 1 → 
    let f := λ x : ℝ, (3 * x^2 - 2 * x + 1) / 4 in
    f p₀ ≤ f p₁ := by
    sorry

end equal_points_probability_not_always_increasing_l446_446909


namespace minimum_value_of_GP_l446_446005

theorem minimum_value_of_GP (a : ℕ → ℝ) (h : ∀ n, 0 < a n) (h_prod : a 2 * a 10 = 9) :
  a 5 + a 7 = 6 :=
by
  -- proof steps will be filled in here
  sorry

end minimum_value_of_GP_l446_446005


namespace solution_count_l446_446784

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 1 else 3 * x - 5

theorem solution_count : (∃! x : ℝ, f(f(x)) = 7) :=
sorry

end solution_count_l446_446784


namespace equal_points_probability_not_always_increasing_l446_446907

theorem equal_points_probability_not_always_increasing 
  (p q : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ q) (h₂ : q ≤ 1) :
  ¬ ∀ p₀ p₁, 0 ≤ p₀ ∧ p₀ ≤ p₁ ∧ p₁ ≤ 1 → 
    let f := λ x : ℝ, (3 * x^2 - 2 * x + 1) / 4 in
    f p₀ ≤ f p₁ := by
    sorry

end equal_points_probability_not_always_increasing_l446_446907


namespace composite_and_factored_l446_446312

theorem composite_and_factored (n : ℕ) (hn : n > 10) :
  let N := n^4 - 90 * n^2 - 91 * n - 90 in
  (N > 1) ∧ ∃ a b c : ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ N = a * b * c :=
by sorry

end composite_and_factored_l446_446312


namespace bus_interval_duration_l446_446306

-- Definition of the conditions
def total_minutes : ℕ := 60
def total_buses : ℕ := 11
def intervals : ℕ := total_buses - 1

-- Theorem stating the interval between each bus departure
theorem bus_interval_duration : total_minutes / intervals = 6 := 
by
  -- The proof is omitted. 
  sorry

end bus_interval_duration_l446_446306


namespace part1_part2_l446_446335

section problem1
variable {a : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x * (Real.log x)
def f_derivative (x : ℝ) (a : ℝ) : ℝ := 2 * x + a * Real.log x + a

theorem part1 (h : ∀ x > 1, f_derivative x a ≥ 0) : a ≥ -2 :=
sorry
end problem1

section problem2
theorem part2 (x : ℝ) (hx : 0 < x) : (x^2 + x * (Real.log x)) ≥ (x - Real.exp (-x)) :=
sorry
end problem2

end part1_part2_l446_446335


namespace total_length_of_board_l446_446962

theorem total_length_of_board (x y : ℝ) (h1 : y = 2 * x) (h2 : y = 46) : x + y = 69 :=
by
  sorry

end total_length_of_board_l446_446962


namespace remainder_of_product_mod_7_l446_446878

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l446_446878


namespace statistics_measuring_dispersion_l446_446175

-- Definition of standard deviation
def standard_deviation (X : List ℝ) : ℝ :=
  let mean := (X.sum) / (X.length)
  (X.map (λ x => (x - mean) ^ 2)).sum / X.length

-- Definition of range
def range (X : List ℝ) : ℝ :=
  X.maximum.get - X.minimum.get

-- Definition of median
noncomputable def median (X : List ℝ) : ℝ :=
  let sorted := List.sorted X
  if sorted.length % 2 == 1 then sorted.get (sorted.length / 2)
  else (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2

-- Definition of mean
def mean (X : List ℝ) : ℝ :=
  X.sum / X.length

-- The proof statement
theorem statistics_measuring_dispersion (X : List ℝ) :
  (standard_deviation X ≠ 0 ∧ range X ≠ 0) ∧
  (∀ x : ℝ, x ∈ X ↔ (median X = x ∨ mean X = x)) → true :=
  sorry

end statistics_measuring_dispersion_l446_446175


namespace ratio_sum_l446_446474

variable (x y z : ℝ)

-- Conditions
axiom geometric_sequence : 16 * y^2 = 15 * x * z
axiom arithmetic_sequence : 2 / y = 1 / x + 1 / z

-- Theorem to prove
theorem ratio_sum : x ≠ 0 → y ≠ 0 → z ≠ 0 → 
  (16 * y^2 = 15 * x * z) → (2 / y = 1 / x + 1 / z) → (x / z + z / x = 34 / 15) :=
by
  -- proof goes here
  sorry

end ratio_sum_l446_446474


namespace johns_annual_haircut_cost_l446_446026

-- Define constants given in the problem
def hair_growth_rate : ℝ := 1.5 -- inches per month
def initial_hair_length_before_cut : ℝ := 9 -- inches
def initial_hair_length_after_cut : ℝ := 6 -- inches
def haircut_cost : ℝ := 45 -- dollars
def tip_percentage : ℝ := 0.2 -- 20%

-- Calculate the number of haircuts per year and the total cost including tips
def monthly_hair_growth := initial_hair_length_before_cut - initial_hair_length_after_cut
def haircut_interval := monthly_hair_growth / hair_growth_rate
def number_of_haircuts_per_year := 12 / haircut_interval
def tip_amount_per_haircut := haircut_cost * tip_percentage
def total_cost_per_haircut := haircut_cost + tip_amount_per_haircut
def total_annual_cost := number_of_haircuts_per_year * total_cost_per_haircut

-- Prove the total annual cost to be $324
theorem johns_annual_haircut_cost : total_annual_cost = 324 := by
  sorry

end johns_annual_haircut_cost_l446_446026


namespace optimal_strategy_l446_446972

noncomputable def prob_A_correct : ℝ := 0.8
noncomputable def prob_B_correct : ℝ := 0.6
noncomputable def score_A_correct : ℝ := 20
noncomputable def score_B_correct : ℝ := 80

def X_distribution : (ℝ → ℝ) :=
λ x, if x = 0 then 1 - prob_A_correct
     else if x = 20 then prob_A_correct * (1 - prob_B_correct)
     else if x = 100 then prob_A_correct * prob_B_correct
     else 0

noncomputable def E_X : ℝ :=
(0 * (1 - prob_A_correct)) + (20 * (prob_A_correct * (1 - prob_B_correct))) + (100 * (prob_A_correct * prob_B_correct))

noncomputable def E_Y : ℝ :=
(0 * (1 - prob_B_correct)) + (80 * (prob_B_correct * (1 - prob_A_correct))) + (100 * (prob_B_correct * prob_A_correct))

theorem optimal_strategy : E_X = 54.4 ∧ E_Y = 57.6 → (57.6 > 54.4) :=
by {
  sorry 
}

end optimal_strategy_l446_446972


namespace egg_processing_plant_l446_446003

-- Definitions based on the conditions
def original_ratio (E : ℕ) : ℕ × ℕ := (24 * E / 25, E / 25)
def new_ratio (E : ℕ) : ℕ × ℕ := (99 * E / 100, E / 100)

-- The mathematical proof problem
theorem egg_processing_plant (E : ℕ) (h : new_ratio E = (original_ratio E).fst + 12, (original_ratio E).snd) : E = 400 := 
  sorry

end egg_processing_plant_l446_446003


namespace find_B_coordinates_l446_446345

-- Definition of vectors a and b
def vector_a : (ℝ × ℝ) := (-2, 3)

structure Vector :=
  (x : ℝ)
  (y : ℝ)

def is_parallel (v1 v2 : Vector) : Prop :=
  v1.x * v2.y = v1.y * v2.x

def B_on_x_axis (B : Vector) : Prop :=
  B.y = 0

def B_on_y_axis (B : Vector) : Prop :=
  B.x = 0

noncomputable def vector_bx (x : ℝ) : Vector := ⟨x - 1, -2⟩
noncomputable def vector_by (y : ℝ) : Vector := ⟨-1, y - 2⟩

theorem find_B_coordinates :
  (∃ (B : Vector), is_parallel (vector_bx B.x) {x := -2, y := 3} ∧ B_on_x_axis B ∧ B.x = 7 / 3) ∧
  (∃ (B : Vector), is_parallel (vector_by B.y) {x := -2, y := 3} ∧ B_on_y_axis B ∧ B.y = 7 / 2) :=
by
  sorry

end find_B_coordinates_l446_446345


namespace last_collision_time_l446_446644

variables (v1 v2 v3 v4 v5 l t : ℝ)
variable (collides_at : ℝ → ℝ → ℝ → ℝ → ℝ → Prop)

-- Conditions
def initial_conditions : Prop :=
  v1 = 0.5 ∧ v2 = 0.5 ∧ v3 = 0.3 ∧ v4 = 0.3 ∧ v5 = 0.3 ∧ l = 1

-- Question rephrased into proof problem with correct answer
theorem last_collision_time
  (h1 : initial_conditions v1 v2 v3 v4 v5 l)
  (h2 : collides_at v1 v2 v3 v4 v5 5) :
  t = 5 :=
sorry

end last_collision_time_l446_446644


namespace range_of_a_l446_446373

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a - x > 1 → x < 2 * a - 1)) ∧
  (∀ x : ℝ, (2 * x + 5 > 3 * a → x > (3 * a - 5) / 2)) ∧
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 6 →
    (x < 2 * a - 1 ∧ x > (3 * a - 5) / 2))) →
  7 / 3 ≤ a ∧ a ≤ 7 / 2 :=
by
  sorry

end range_of_a_l446_446373


namespace solve_quadratic_l446_446470

theorem solve_quadratic (x : ℚ) (h_pos : x > 0) (h_eq : 3 * x^2 + 8 * x - 35 = 0) : 
    x = 7/3 :=
by
    sorry

end solve_quadratic_l446_446470


namespace simplify_expression_l446_446783

theorem simplify_expression 
  (a b c d : ℝ) 
  (h_a : a ≠ 0) 
  (h_b : b ≠ 0) 
  (h_c : c ≠ 0) 
  (h_d : d ≠ 0) :
  let x := (b/c) + (c/b),
      y := (a/c) + (c/a),
      z := (a/b) + (b/a),
      w := (a/d) + (d/a)
  in (x^2 + y^2 + z^2 + w^2 - x*y*z*w) = 8 := 
sorry

end simplify_expression_l446_446783


namespace prime_exponent_50_factorial_5_l446_446017

def count_factors_in_factorial (n p : ℕ) : ℕ :=
  if p ≤ 1 then 0 else
    let rec count (k acc : ℕ) : ℕ :=
      if n < p^k then acc else count (k+1) (acc + n/(p^k))
    count 1 0

theorem prime_exponent_50_factorial_5 : count_factors_in_factorial 50 5 = 12 :=
  by
    sorry

end prime_exponent_50_factorial_5_l446_446017


namespace point_P_pos_relation_parallelogram_area_l446_446746

-- Definitions
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 2) + (y^2 / 3) = 1

def point_P (x y : ℝ) : Prop := x = 4 / 5 ∧ y = 6 / 5

def inside_ellipse (x y : ℝ) : Prop := (x^2 / 2) + (y^2 / 3) < 1

-- Proof Problem
theorem point_P_pos_relation :
  ∃ (x y : ℝ), point_P x y → inside_ellipse x y :=
by
  sorry

-- Definitions for the area calculation
def area_parallelogram : ℝ := 4 / 5 * sqrt 6

-- Proof Problem
theorem parallelogram_area : 
  ∃ (a : ℝ), 
    ( let x1 x2 := 4 / 5 in
      let x_product := - 4 / 5 in
      let mn := sqrt (1 + 1^2) * (x1 - x2) :=
        (x1 + x2)^2 - 4 * x_product =
        8 / 5 * sqrt 3 ∧
      let h := sqrt 2 / 2 in
      let area := mn * h in
      a = area_parallelogram ) :=
by
  sorry

end point_P_pos_relation_parallelogram_area_l446_446746


namespace sam_football_games_this_year_l446_446817

theorem sam_football_games_this_year (last_year_games total_games : ℕ)
  (h1 : last_year_games = 29)
  (h2 : total_games = 43) :
  let Sam_this_year := total_games - last_year_games in
  Sam_this_year = 14 :=
by
  rw [h1, h2]
  simp
  sorry

end sam_football_games_this_year_l446_446817


namespace ratio_swordfish_to_pufferfish_l446_446380

theorem ratio_swordfish_to_pufferfish (P S : ℕ) (n : ℕ) 
  (hP : P = 15)
  (hTotal : S + P = 90)
  (hRelation : S = n * P) : 
  (S : ℚ) / (P : ℚ) = 5 := 
by 
  sorry

end ratio_swordfish_to_pufferfish_l446_446380


namespace product_of_abscissas_l446_446331

noncomputable def f : ℝ → ℝ
| x => if x > 1 then Real.log x else -Real.log x

def deriv_f (x : ℝ) : ℝ :=
if x > 1 then 1 / x else -1 / x

theorem product_of_abscissas (x₁ x₂ : ℝ) (h₁ : 0 < x₁ ∧ x₁ < 1) (h₂ : x₂ > 1)
    (h_perp : deriv_f x₁ * deriv_f x₂ = -1) : x₁ * x₂ = 1 := 
by
    -- sorry to skip the proof
    sorry

end product_of_abscissas_l446_446331


namespace dispersion_measures_correct_l446_446188

-- Define a sample data set
variable {x : ℕ → ℝ}
variable {n : ℕ}

-- Definitions of the four statistics
def standard_deviation (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def median (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def range (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry
def mean (x : ℕ → ℝ) (n : ℕ) : ℝ := sorry

-- Definition of measures_dispersion function
def measures_dispersion (stat : ℕ → ℝ → ℝ) (x : ℕ → ℝ) (n : ℕ) : Prop :=
  sorry -- Define what it means for a statistic to measure dispersion

-- Problem statement in Lean
theorem dispersion_measures_correct :
  measures_dispersion standard_deviation x n ∧
  measures_dispersion range x n ∧
  ¬measures_dispersion median x n ∧
  ¬measures_dispersion mean x n :=
by sorry

end dispersion_measures_correct_l446_446188


namespace quadrilateral_smallest_angle_l446_446996

theorem quadrilateral_smallest_angle
  (a d : ℝ)
  (h1 : a + (a + 2 * d) = 160)
  (h2 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) :
  a = 60 :=
by
  sorry

end quadrilateral_smallest_angle_l446_446996


namespace michael_pets_kangaroos_l446_446443

theorem michael_pets_kangaroos :
  let total_pets := 24
  let fraction_dogs := 1 / 8
  let fraction_not_cows := 3 / 4
  let fraction_not_cats := 2 / 3
  let num_dogs := fraction_dogs * total_pets
  let num_cows := (1 - fraction_not_cows) * total_pets
  let num_cats := (1 - fraction_not_cats) * total_pets
  let num_kangaroos := total_pets - num_dogs - num_cows - num_cats
  num_kangaroos = 7 :=
by
  sorry

end michael_pets_kangaroos_l446_446443


namespace no_triples_exist_l446_446632

theorem no_triples_exist (m p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m > 0) :
  2^m * p^2 + 1 ≠ q^7 :=
sorry

end no_triples_exist_l446_446632


namespace grapes_filled_per_day_l446_446604

theorem grapes_filled_per_day (pickers : ℕ) (drums : ℕ) (days : ℕ) (h : pickers = 266) (h_d : drums = 90) (h_t : days = 5) : (drums / days) = 18 :=
by
  rw [h_d, h_t]
  simp
  sorry

end grapes_filled_per_day_l446_446604


namespace common_tangent_exists_range_of_a_l446_446368

noncomputable def a_range : Set ℝ := Set.Ico (-2/Real.exp 1) 0

theorem common_tangent_exists_range_of_a
  (x1 x2 : ℝ)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (a : ℝ)
  (h_tangent : (∀ (x > 0), DifferentiableAt ℝ (λ x, a / x) x) ∧ 
               (∀ (x > 0), DifferentiableAt ℝ (λ x, 2 * Real.log x) x) ∧ 
               ∃ (x1 x2 : ℝ), a/x1^2 = -2/x2 ∧ (a/x1 = Real.log x2 - 1)) :
  a ∈ a_range :=
sorry

end common_tangent_exists_range_of_a_l446_446368


namespace find_k_l446_446342

/- Definitions for vectors -/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/- Prove that if ka + b is perpendicular to a, then k = -1/5 -/
theorem find_k (k : ℝ) : 
  dot_product (k • (1, 2) + (-3, 2)) (1, 2) = 0 → 
  k = -1 / 5 := 
  sorry

end find_k_l446_446342


namespace mass_of_man_l446_446963

noncomputable def boat_length : ℝ := 4
noncomputable def boat_breadth : ℝ := 2
noncomputable def boat_sinking_height : ℝ := 0.01
noncomputable def water_density : ℝ := 1000

theorem mass_of_man :
  let volume := boat_length * boat_breadth * boat_sinking_height in
  let mass := water_density * volume in
  mass = 80 := by
    sorry

end mass_of_man_l446_446963


namespace prove_notebooks_l446_446127

open Nat

variables (total half notebooks_second_half_total total_notebooks x : ℕ)

def problem_statement :=
  total = 28 ∧
  half = total / 2 ∧
  notebooks_second_half_total = half * 3 ∧
  total_notebooks = 112 ∧
  (14 * x + notebooks_second_half_total = total_notebooks)
  
theorem prove_notebooks : problem_statement total half notebooks_second_half_total total_notebooks x → x = 5 :=
by 
  intro h
  obtain ⟨h_total, h_half, h_nsh_total, h_total_notebooks, h_eq⟩ := h
  have h1 : half = 14 := by rw [h_total, Nat.div_self (Nat.succ_pos 27)]
  have h2 : notebooks_second_half_total = 42 := by rw [h1, h_nsh_total, h1]
  have h3 : 14 * x + 42 = 112 := by rw [h_eq, h2]
  have h4 : 14 * x = 70 := by linarith
  have h5 : x = 5 := by linarith
  exact h5

end prove_notebooks_l446_446127


namespace imo1985_p6_l446_446778

theorem imo1985_p6 (k : ℕ) (n : Fin k → ℕ) 
  (h1 : 2 ≤ k) 
  (h2 : ∀ i : Fin k, i.succ < k → (n i.succ) ∣ (2 ^ (n i) - 1)) 
  (h3 : (n 0) ∣ (2 ^ (n (k - 1)) - 1)) : 
  ∀ i : Fin k, n i = 1 := 
  sorry

end imo1985_p6_l446_446778


namespace parallelogram_area_correct_l446_446579

def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem parallelogram_area_correct :
  parallelogram_area 15 5 = 75 :=
by
  sorry

end parallelogram_area_correct_l446_446579


namespace greatest_integer_sum_l446_446292

theorem greatest_integer_sum (a : ℕ) (n : ℕ := 1983) (m : ℕ := 2^(13 * a)) :
  (⌊ ∑ k in Finset.range (m + 1), (k + 1)^(1 / n - 1) ⌋) = 1983 := 
by
  sorry

end greatest_integer_sum_l446_446292


namespace remove_to_maximize_pairs_summing_to_12_l446_446545

theorem remove_to_maximize_pairs_summing_to_12 :
  ∃ x ∈ [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], 
  ∀ lst, lst = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12] → 
    max_pairs_summing_to_12 (lst.erase x) :=
begin
  sorry
end

noncomputable def max_pairs_summing_to_12 (lst : list ℤ) : ℕ :=
  list.length (list.filter (λ (pair : ℤ × ℤ), pair.fst + pair.snd = 12) (list.prod lst lst))

end remove_to_maximize_pairs_summing_to_12_l446_446545


namespace product_mod_7_l446_446866

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l446_446866


namespace shortest_path_avoid_unit_circle_l446_446391

def shortest_path_length : ℝ :=
  2 * real.sqrt 3 + real.pi / 3

theorem shortest_path_avoid_unit_circle :
  let path_length := shortest_path_length in
  path_length = 2 * real.sqrt 3 + real.pi / 3 :=
sorry

end shortest_path_avoid_unit_circle_l446_446391


namespace adam_laptop_cost_l446_446596

theorem adam_laptop_cost : 
  let first_laptop_cost := 500
  let second_laptop_cost := 3 * first_laptop_cost
  let discount_second_laptop := (15 / 100) * second_laptop_cost
  let second_laptop_cost_after_discount := second_laptop_cost - discount_second_laptop
  let external_hard_drive_cost := 80
  let mouse_cost := 20
  let accessories_cost := external_hard_drive_cost + mouse_cost
  let total_accessories_cost := 2 * accessories_cost
  let total_cost := first_laptop_cost + second_laptop_cost_after_discount + total_accessories_cost
in total_cost = 1975 :=
by
  sorry

end adam_laptop_cost_l446_446596


namespace compare_ln_log_exp_l446_446659

theorem compare_ln_log_exp (x y z : ℝ) (h1 : x = Real.log Real.pi) (h2 : y = Real.log (1 / 2) Real.pi) (h3 : z = Real.exp (-1/2)) : y < z ∧ z < x :=
by
  sorry

end compare_ln_log_exp_l446_446659


namespace simplify_expression_evaluate_expression_l446_446948

variable {α : ℝ}

-- Problem 1
theorem simplify_expression (α : ℝ) :
  (tan (π + α) * cos (2 * π + α) * sin (α - π / 2)) / (cos (-α - 3 * π) * sin (-3 * π - α)) = 1 :=
sorry

-- Problem 2
theorem evaluate_expression (α : ℝ) (h : tan α = 1 / 2) :
  2 * sin α ^ 2 - sin α * cos α + cos α ^ 2 = 4 / 5 :=
sorry

end simplify_expression_evaluate_expression_l446_446948


namespace quartic_polynomial_sum_l446_446585

theorem quartic_polynomial_sum :
  ∀ (q : ℤ → ℤ),
    q 1 = 4 →
    q 8 = 26 →
    q 12 = 14 →
    q 15 = 34 →
    q 19 = 44 →
    (q 1 + q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 +
     q 11 + q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18 + q 19 + q 20) = 252 :=
by
  intros
  sorry

end quartic_polynomial_sum_l446_446585


namespace num_ways_to_cover_board_l446_446743

def tile_ways : ℕ → ℕ
| 1 := 1
| 2 := 2
| (n + 1) := tile_ways n + tile_ways (n - 1)

theorem num_ways_to_cover_board : tile_ways 13 = 377 := 
by {
  -- You would normally write a proof here, but we're adding a placeholder.
  sorry
}

end num_ways_to_cover_board_l446_446743


namespace car_new_speed_l446_446406

theorem car_new_speed (original_speed : ℝ) (supercharge_percent : ℝ) (weight_cut_speed_increase : ℝ) :
  original_speed = 150 → supercharge_percent = 0.30 → weight_cut_speed_increase = 10 → 
  original_speed * (1 + supercharge_percent) + weight_cut_speed_increase = 205 :=
by
  intros h_orig h_supercharge h_weight
  rw [h_orig, h_supercharge]
  sorry

end car_new_speed_l446_446406


namespace oliver_learning_days_l446_446645

theorem oliver_learning_days :
  (∀ (days_per_alphabet : ℕ) (total_days : ℕ),
    total_days = 25 →
    (∃ (vowels_count : ℕ), vowels_count = 5) →
    (total_days / vowels_count = days_per_alphabet) →
    days_per_alphabet = 5) :=
by
  intros days_per_alphabet total_days htotal_days hvowel_count heq.
  cases hvowel_count with vowels_count hvowel_eq.
  rw [hvowel_eq, htotal_days] at heq.
  exact heq.symm

end oliver_learning_days_l446_446645


namespace max_stand_proof_l446_446249

noncomputable def table_max_stand (barons counts marquises : ℕ) : ℕ :=
  if barons = 25 ∧ counts = 20 ∧ marquises = 10 then 52 else 0

theorem max_stand_proof :
  ∃ barons counts marquises, barons = 25 ∧ counts = 20 ∧ marquises = 10 →
  table_max_stand barons counts marquises = 52 :=
by
  use [25, 20, 10]
  intro h
  simp [table_max_stand, h]
  sorry

end max_stand_proof_l446_446249


namespace product_remainder_mod_7_l446_446871

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l446_446871


namespace total_cakes_served_l446_446997

-- Defining the values for cakes served during lunch and dinner
def lunch_cakes : ℤ := 6
def dinner_cakes : ℤ := 9

-- Stating the theorem that the total number of cakes served today is 15
theorem total_cakes_served : lunch_cakes + dinner_cakes = 15 :=
by
  sorry

end total_cakes_served_l446_446997


namespace hyperbola_asymptotes_hyperbola_slope_line_l446_446266

-- Part 1 Asymptotes for b^2 = 2 and angle of inclination π/2
theorem hyperbola_asymptotes (b : ℝ) (b_pos : 0 < b) 
  (inclination : real.angle = real.pi / 2)
  (equilateral_triangle : Δ (F1 : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)) : 
  (b^2 = 2) → (∀ x : ℝ, x^2 - (y^2 / 2) = 1) → 
  (∀ x : ℝ, y = ±√2 * x) :=
sorry

-- Part 2 Slope of line l for b = √3
theorem hyperbola_slope_line (b : ℝ) (b_eq_sqrt_3 : b = √3)
  (exists_slope : ∃ k : ℝ, true)
  (midpoint_M : ℝ × ℝ)
  (dot_product_zero : ∀ (FM : ℝ × ℝ), FM ⋅ AB = 0) : 
  ∀ (k : ℝ), k = ±√(15) / 5 :=
sorry

end hyperbola_asymptotes_hyperbola_slope_line_l446_446266


namespace number_eq_1925_l446_446719

theorem number_eq_1925 (x : ℝ) (h : x / 7 - x / 11 = 100) : x = 1925 :=
sorry

end number_eq_1925_l446_446719


namespace simplify_and_evaluate_l446_446822

theorem simplify_and_evaluate (a : ℚ) (h : a = -1/6) : 
  2 * (a + 1) * (a - 1) - a * (2 * a - 3) = -5 / 2 := by
  rw [h]
  sorry

end simplify_and_evaluate_l446_446822


namespace XH_HP_ratio_l446_446399

-- Define the basic structure of triangle XYZ with specific lengths and angle.
structure Triangle :=
(X Y Z : Type)
(xz : ℝ)
(yz : ℝ)
(angle_z : ℝ)

noncomputable def orthocenter (T : Triangle) : Prop :=
  -- Definitions related to orthocenter can be placed here.
  sorry

-- Given conditions
def triangle_XYZ : Triangle :=
{ X := unit,
  Y := unit,
  Z := unit,
  xz := 5,
  yz := 4 * Real.sqrt 2,
  angle_z := 45 }
  
-- Prove the ratio XH:HP = 3
theorem XH_HP_ratio (T : Triangle) (h_ortho : orthocenter T) : 
  T = triangle_XYZ → (/* XH/HP */ sorry) = 3 :=
by
  -- The proof goes here
  sorry

end XH_HP_ratio_l446_446399


namespace slope_of_line_l446_446226

variable {p : ℝ} (hp : p > 0)

def parabola := { P : ℝ × ℝ | P.2^2 = 2 * p * P.1 }

def line (k : ℝ) := { Q : ℝ × ℝ | Q.2 = k * Q.1 + p / 2 }

noncomputable def area_of_triangle (O A B : ℝ × ℝ) : ℝ := 
  0.5 * |O.1 * A.2 + A.1 * B.2 + B.1 * O.2 - O.2 * A.1 - A.2 * B.1 - B.2 * O.1|

theorem slope_of_line (A B : ℝ × ℝ) (k : ℝ) 
    (hA_in_parabola : A ∈ parabola hp)
    (hB_in_parabola : B ∈ parabola hp)
    (hA_on_line : A ∈ line p k)
    (hB_on_line : B ∈ line p k)
    (hfoc : ∃ F : ℝ × ℝ, F = (p / 2, 0))
    (h_area : ∀ O F : ℝ × ℝ, area_of_triangle O A F = 4 * area_of_triangle O B F) :
    k = 4 / 3 ∨ k = -4 / 3 :=
sorry

end slope_of_line_l446_446226


namespace irrational_sqrt_seven_l446_446936

theorem irrational_sqrt_seven : irrational (real.sqrt 7) :=
sorry

end irrational_sqrt_seven_l446_446936


namespace angle_ACB_geq_sixty_l446_446838

variable {A B C B₁ A₁ K : Point} -- Define points

noncomputable def tangent_circle_properties (Δ : Triangle A B C) 
  (insc : Circle) (HA_cb : insc.TangentPoint A C = B₁) (HB_ca : insc.TangentPoint B C = A₁) : Prop :=
  insc.TangentPoint A C = B₁ ∧ insc.TangentPoint B C = A₁ -- Defining tangency conditions

noncomputable def point_on_side (A B B₁ A₁ : Point) (K : Point) (h₁ : K ∈ OpenSegment A B) : Prop :=
  dist A K = dist K B₁ ∧ dist B K = dist K A₁

theorem angle_ACB_geq_sixty (Δ : Triangle A B C) (insc : Circle) 
  (pb₁ : tangent_circle_properties Δ insc (insc.TangentPoint A C = B₁) (insc.TangentPoint B C = A₁))
  (hK : point_on_side A B B₁ A₁ K) :
  ∠ A C B ≥ 60° := by
  sorry

end angle_ACB_geq_sixty_l446_446838


namespace XiaoMing_strategy_l446_446971

noncomputable def prob_A_correct : ℝ := 0.8
noncomputable def prob_B_correct : ℝ := 0.6

def points_A_correct : ℝ := 20
def points_B_correct : ℝ := 80

def prob_XA_0 : ℝ := 1 - prob_A_correct
def prob_XA_20 : ℝ := prob_A_correct * (1 - prob_B_correct)
def prob_XA_100 : ℝ := prob_A_correct * prob_B_correct

def expected_XA : ℝ := 0 * prob_XA_0 + points_A_correct * prob_XA_20 + (points_A_correct + points_B_correct) * prob_XA_100

def prob_YB_0 : ℝ := 1 - prob_B_correct
def prob_YB_80 : ℝ := prob_B_correct * (1 - prob_A_correct)
def prob_YB_100 : ℝ := prob_B_correct * prob_A_correct

def expected_YB : ℝ := 0 * prob_YB_0 + points_B_correct * prob_YB_80 + (points_A_correct + points_B_correct) * prob_YB_100

def distribution_A_is_correct : Prop :=
  prob_XA_0 = 0.2 ∧ prob_XA_20 = 0.32 ∧ prob_XA_100 = 0.48

def choose_B_first : Prop :=
  expected_YB > expected_XA

theorem XiaoMing_strategy :
  distribution_A_is_correct ∧ choose_B_first :=
by
  sorry

end XiaoMing_strategy_l446_446971


namespace smallest_value_of_a_l446_446297

noncomputable def smallest_a (f : ℝ → ℝ) : ℝ := 
  if h : ∃ a, f a 
  then Classical.choose h 
  else 0

theorem smallest_value_of_a :
  (∃ a : ℝ, ((3 * a) ^ 2 - 2 * a^2 = 0.28) ∧ a = -0.2 ∨ a = 0.2) ∧ 
  (0.2 > -0.2) ∧ 
  ((-0.2) = smallest_a (λ a, (3 * a) ^ 2 - 2 * a^2 = 0.28)) :=
sorry

end smallest_value_of_a_l446_446297


namespace equilateral_triangle_one_of_OXY_OYZ_OZX_l446_446378

variables (A B C D E F O X Y Z : Point)
variable [s : TriangleABC_Scalene A B C]

-- D is the foot of the altitude through A
axiom altitude_A_D : Altitude A D

-- E is the intersection of AC with the bisector of ∠ABC
axiom intersection_bisector_AC_E : Intersection AC (AngleBisector ABC) E

-- F is a point on AB
axiom point_on_AB_F : OnLine F AB

-- O is the circumcenter of ΔABC
axiom circumcenter_O : Circumcenter O A B C

-- X = AD ∩ BE
axiom intersection_X : Intersection AD BE X

-- Y = BE ∩ CF
axiom intersection_Y : Intersection BE CF Y

-- Z = CF ∩ AD
axiom intersection_Z : Intersection CF AD Z

-- ΔXYZ is equilateral
axiom equilateral_XYZ : EquilateralTriangle X Y Z

-- Statement to prove: one of the triangles OXY, OYZ, OZX is equilateral
theorem equilateral_triangle_one_of_OXY_OYZ_OZX 
  (A B C D E F O X Y Z : Point)
  [s : TriangleABC_Scalene A B C]
  (altitude_A_D : Altitude A D)
  (intersection_bisector_AC_E : Intersection AC (AngleBisector ABC) E)
  (point_on_AB_F : OnLine F AB)
  (circumcenter_O : Circumcenter O A B C)
  (intersection_X : Intersection AD BE X)
  (intersection_Y : Intersection BE CF Y)
  (intersection_Z : Intersection CF AD Z)
  (equilateral_XYZ : EquilateralTriangle X Y Z) :
  EquilateralTriangle O X Y ∨ EquilateralTriangle O Y Z ∨ EquilateralTriangle O Z X :=
sorry

end equilateral_triangle_one_of_OXY_OYZ_OZX_l446_446378


namespace find_a9_l446_446663

theorem find_a9 (a : ℕ → ℕ) 
  (h_add : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
  (h_a2 : a 2 = 4) 
  : a 9 = 18 :=
sorry

end find_a9_l446_446663


namespace S_inter_T_is_finite_l446_446721

open Set

def S : Set ℝ := { y | ∃ x : ℝ, y = 3^x }
def T : Set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }

theorem S_inter_T_is_finite : S ∩ T = { y | ∃ x : ℝ, 3^x = x^2 - 1 } ∧ Finite (S ∩ T) :=
sorry

end S_inter_T_is_finite_l446_446721


namespace xiao_ying_should_pay_l446_446444

variable (x y z : ℝ)

def equation1 := 3 * x + 7 * y + z = 14
def equation2 := 4 * x + 10 * y + z = 16
def equation3 := 2 * (x + y + z) = 20

theorem xiao_ying_should_pay :
  equation1 x y z →
  equation2 x y z →
  equation3 x y z :=
by
  intros h1 h2
  sorry

end xiao_ying_should_pay_l446_446444


namespace minimum_pumps_needed_to_empty_well_l446_446533

-- Definitions based on the given conditions
def amount_inflow_per_minute (M : ℝ) : ℝ := M 
def rate_of_one_pump (A : ℝ) : ℝ := A
def time_for_4_pumps (T₄ : ℝ) : ℝ := 40
def time_for_5_pumps (T₅ : ℝ) : ℝ := 30
def time_for_target (T_target : ℝ) : ℝ := 24

-- Total water drawn by pumps
def total_water_drawn (n : ℕ) (A : ℝ) (T : ℝ) : ℝ := n * A * T 

-- Required number of pumps
def required_pumps (W : ℝ) (A : ℝ) (M : ℝ) (T_target : ℝ) : ℕ := 
  let W := 120 * M in (W + T_target * M) / (T_target * A)

-- Proof statement
theorem minimum_pumps_needed_to_empty_well (
  M A : ℝ
  (h1 : rate_of_one_pump A = amount_inflow_per_minute M)
) : required_pumps 120 M T_target = 6 :=
by 
  -- Based on the solution provided previously
  sorry

end minimum_pumps_needed_to_empty_well_l446_446533


namespace area_transformed_region_l446_446422

-- Define the transformation matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 1], ![4, 3]]

-- Define the area of region T
def area_T := 6

-- The statement we want to prove: the area of T' is 30.
theorem area_transformed_region :
  let det := matrix.det
  area_T * det = 30 :=
by
  sorry

end area_transformed_region_l446_446422


namespace profit_percentage_for_40pct_of_apples_l446_446234

theorem profit_percentage_for_40pct_of_apples (P : ℝ)
  (H1 : 280 * 0.4 * (1 + P / 100) + 280 * 0.6 * 1.30 = 280 * 1.26) :
  P = 20 :=
by
  have eq1 : 112 * (1 + P / 100) + 168 * 1.30 = 352.8 := sorry,
  have eq2 : 112 + 112 * P / 100 + 218.4 = 352.8 := sorry,
  have eq3 : 112 * P / 100 = 22.4 := sorry,
  have eq4 : P / 100 = 22.4 / 112 := sorry,
  have eq5 : P / 100 = 0.2 := sorry,
  have eq6 : P = 0.2 * 100 := sorry,
  show P = 20, from sorry

end profit_percentage_for_40pct_of_apples_l446_446234


namespace part_1_part_2_part_3_l446_446341

noncomputable def Line (k : ℝ) : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, k * x + 1) }
noncomputable def Circle : Set (ℝ × ℝ) := { p | (p.fst - 1)^2 + (p.snd + 1)^2 = 12 }

theorem part_1 (k : ℝ) : ∀ (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)), 
  l = Line k → C = Circle → 
  ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ l ∧ A ∈ C ∧ B ∈ l ∧ B ∈ C :=
begin
  intros l C hl hC,
  rw [hl, hC],
  sorry
end

theorem part_2 (k : ℝ) : ∃ P : ℝ × ℝ, ∀ (l : Set (ℝ × ℝ)), l = Line k → P ∈ l :=
begin
  use (0, 1),
  intros l hl,
  rw hl,
  use 0,
  simp,
end

theorem part_3 (k : ℝ) : ∀ (l : Set (ℝ × ℝ)), l = Line k → 
  ∃ length : ℝ, ∀ A B : ℕ × ℕ, A = (x - 1)^2 + (y + 1)^2 = 12 → length = 2 * sqrt 7 :=
begin
  intros l hl,
  sorry
end

end part_1_part_2_part_3_l446_446341


namespace sum_intersections_g_eq_2_l446_446102

-- Define the function g(x) piecewise using the given line segments
noncomputable def g (x : ℝ) : ℝ :=
  if h₁ : -4 ≤ x ∧ x ≤ -2 then 2 * x + 4
  else if h₂ : -2 ≤ x ∧ x ≤ 0 then -3 * x - 1
  else if h₃ : 0 ≤ x ∧ x ≤ 2 then 2 * x - 1
  else if h₄ : 2 ≤ x ∧ x ≤ 4 then x + 1
  else 0

-- State the Lean 4 theorem
theorem sum_intersections_g_eq_2 : (∑ x in {x | g x = 2}, x) = 0 :=
by {
  sorry
}

end sum_intersections_g_eq_2_l446_446102


namespace sin_symmetry_pi_l446_446495

theorem sin_symmetry_pi : ∀ x : ℝ, sin x = sin (π - x) := 
by
  sorry

end sin_symmetry_pi_l446_446495


namespace initial_boys_21_l446_446985

variable (p : ℕ) (boys : ℕ) (girls : ℕ)
variable (initial_boys_percentage : ℚ) (final_boys_percentage : ℚ)

def initial_conditions : Prop :=
  initial_boys_percentage = 0.35 ∧ final_boys_percentage = 0.4

def change_conditions : Prop :=
  boys - 3 = final_boys_percentage * (p : ℚ)

def find_initial_boys : Prop :=
  0.35 * (p : ℚ) = 21

theorem initial_boys_21 (h1 : initial_conditions) (h2 : change_conditions) : find_initial_boys := by
  sorry

end initial_boys_21_l446_446985


namespace increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446910

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446910


namespace prime_factorization_of_expression_l446_446442

theorem prime_factorization_of_expression :
  2 * 3 * 5 * 7 - 1 = 11 * 19 :=
sorry

end prime_factorization_of_expression_l446_446442


namespace dispersion_measures_l446_446170
-- Definitions for statistical measures (for clarity, too simplistic)
def standard_deviation (x : List ℝ) : ℝ := 
  let mean := (x.sum / x.length)
  Math.sqrt ((x.map (λ xi => (xi - mean)^2)).sum / (x.length - 1))

def median (x : List ℝ) : ℝ := 
  let sorted := x.qsort (≤)
  if h : sorted.length % 2 = 1 then (sorted.sorted.nth (sorted.length / 2))
  else ((sorted.nth (sorted.length / 2 - 1) + sorted.nth (sorted.length / 2)) / 2)

def range (x : List ℝ) : ℝ := x.maximum - x.minimum

def mean (x : List ℝ) : ℝ := x.sum / x.length

-- Statement to prove
theorem dispersion_measures (x : List ℝ) : 
  (standard_deviation x ∈ {standard_deviation x, range x}) ∧ 
  (range x ∈ {standard_deviation x, range x}) ∧
  ¬ (median x ∈ {standard_deviation x, range x})  ∧
  ¬ (mean x ∈ {standard_deviation x, range x}) := 
sorry

end dispersion_measures_l446_446170


namespace inverse_of_3A_squared_l446_446353

variable {A : Matrix (Fin 2) (Fin 2) ℚ}
variable hA_inv : A⁻¹ = Matrix.of ![![3, 4], ![-2, -2]]

theorem inverse_of_3A_squared :
  3 * (A ⬝ A)⁻¹ = Matrix.of ![![1 / 3, 4 / 3], ![-2 / 3, -4 / 3]] :=
by
  -- The proof would go here
  sorry

end inverse_of_3A_squared_l446_446353


namespace product_remainder_mod_7_l446_446861

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l446_446861


namespace find_a_l446_446044

noncomputable def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x * a = 1}
axiom A_is_B (a : ℝ) : A ∩ B a = B a → (a = 0) ∨ (a = 1/3) ∨ (a = 1/5)

-- statement to prove
theorem find_a (a : ℝ) (h : A ∩ B a = B a) : (a = 0) ∨ (a = 1/3) ∨ (a = 1/5) :=
by 
  apply A_is_B
  assumption

end find_a_l446_446044


namespace inverse_passes_through_point_l446_446371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log 2 ((x - a) / (x + 1))

theorem inverse_passes_through_point 
  (h : ∀ x, (f a x = (-2 : ℝ)) ↔ x = 3) : a = 2 :=
by 
  sorry

end inverse_passes_through_point_l446_446371


namespace total_number_of_people_l446_446729

def total_people_at_park(hikers bike_riders : Nat) : Nat :=
  hikers + bike_riders

theorem total_number_of_people 
  (bike_riders : Nat)
  (hikers : Nat)
  (hikers_eq_bikes_plus_178 : hikers = bike_riders + 178)
  (bikes_eq_249 : bike_riders = 249) :
  total_people_at_park hikers bike_riders = 676 :=
by
  sorry

end total_number_of_people_l446_446729


namespace initial_average_mark_l446_446837

theorem initial_average_mark (A : ℝ) 
  (h1 : 35 * A = 30 * 90 + 5 * 20) : 
  A = 80 := 
begin
  linarith,
end

end initial_average_mark_l446_446837


namespace repeated_division_l446_446926

theorem repeated_division (a b c : ℕ) (hab_rc : a = 102) (hbc_rc : b = 6) (hc_rc : c = 3) :
  let d := a / b,
      e := d / c,
      r := (d % c) in 
  (d = 17) ∧ (e = 5) ∧ (r = 2) := 
by {
  intros; 
  dsimp; 
  sorry,
}

end repeated_division_l446_446926


namespace union_complement_eq_l446_446343

open Set

variable {U : Set ℝ} {A B : Set ℝ}

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | (x + 2) * (x - 1) > 0}

-- Define set B
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 0}

-- Define the complement of B in U
def C_U_B : Set ℝ := compl B

theorem union_complement_eq :
  A ∪ C_U_B = {x : ℝ | x < -1 ∨ x ≥ 0} := by
    sorry

end union_complement_eq_l446_446343


namespace angle_BDA_given_triangle_ABC_l446_446376

theorem angle_BDA_given_triangle_ABC 
  (A B C H D : Type)
  (angle_A angle_B angle_C : ℝ)
  (is_altitude : BH > 0)
  (extend_BH_to_D : BD = BC)
  (angle_A_eq : angle_A = 80)
  (angle_B_eq : angle_B = 30)
  (angle_C_eq : angle_C = 70) :
  angle BDA = 70 :=
by
  sorry

end angle_BDA_given_triangle_ABC_l446_446376


namespace zero_point_in_interval_l446_446125

open Real

def f (x : ℝ) : ℝ := log x / log 2 - 1 / x

theorem zero_point_in_interval : ∃ c ∈ Ioo (1 : ℝ) 2, f c = 0 :=
begin
  -- Ensure necessary definitions upcoming from the problem
  sorry
end

end zero_point_in_interval_l446_446125


namespace P_investment_l446_446069

variable Q_investment : ℕ
variable P_profit_share : ℕ
variable Q_profit_share : ℕ

def investment_ratio := 5 / 1
def profit_share_ratio := 5 / (Q_profit_share + P_profit_share)

theorem P_investment (Q_investment = 15000) (profit_share_ratio = 5 / 6) : P_investment = 75000 :=
by
sorry

end P_investment_l446_446069


namespace sara_total_score_l446_446006

-- Definitions based on the conditions
def correct_points (correct_answers : Nat) : Int := correct_answers * 2
def incorrect_points (incorrect_answers : Nat) : Int := incorrect_answers * (-1)
def unanswered_points (unanswered_questions : Nat) : Int := unanswered_questions * 0

def total_score (correct_answers incorrect_answers unanswered_questions : Nat) : Int :=
  correct_points correct_answers + incorrect_points incorrect_answers + unanswered_points unanswered_questions

-- The main theorem stating the problem requirement
theorem sara_total_score :
  total_score 18 10 2 = 26 :=
by
  sorry

end sara_total_score_l446_446006


namespace product_of_five_primes_is_982982_l446_446584

theorem product_of_five_primes_is_982982 (A B C : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : prime 491)
  (h5 : ∃ p1 p2 p3 p4 p5 : ℕ, prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ prime p5 ∧ (p1 * p2 * p3 * p4 * p5 = 1001 * (100 * A + 10 * B + C))):
  1001 * (100 * A + 10 * B + C) = 982982 := sorry

end product_of_five_primes_is_982982_l446_446584


namespace sequence_arithmetic_l446_446662

-- Define the sequence and sum conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (p : ℝ)

-- We are given that the sum of the first n terms is Sn = n * p * a_n
axiom sum_condition (n : ℕ) (hpos : n > 0) : S n = n * p * a n

-- Also, given that a_1 ≠ a_2
axiom a1_ne_a2 : a 1 ≠ a 2

-- Define what we need to prove
theorem sequence_arithmetic (n : ℕ) (hn : n ≥ 2) :
  ∃ (a2 : ℝ), p = 1/2 ∧ a n = (n-1) * a2 :=
by
  sorry

end sequence_arithmetic_l446_446662


namespace sandy_shopping_l446_446466

variable (X : ℝ)

theorem sandy_shopping (h : 0.70 * X = 210) : X = 300 := by
  sorry

end sandy_shopping_l446_446466


namespace graph_not_pass_second_quadrant_l446_446426

theorem graph_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) :
  ¬ ∃ (x : ℝ), y = a^x + b ∧ x < 0 ∧ y > 0 :=
by
  sorry

end graph_not_pass_second_quadrant_l446_446426


namespace flagpole_break_height_l446_446576

theorem flagpole_break_height (height flagpole_height dangle_height : ℝ) 
  (h1 : flagpole_height = 18) 
  (h2 : dangle_height = 3) 
  (h_angle : ∃ θ : ℝ, θ = 30) :
  height = 7.5 :=
by
  -- Define the broken height given the hypotenuse calculation
  let hypotenuse := flagpole_height - dangle_height
  have h_hypotenuse : hypotenuse = 15, from calc
    hypotenuse = flagpole_height - dangle_height : rfl
              ... = 18 - 3 : by rw [h1, h2]
              ... = 15 : rfl,
  -- Calculate the part of the flagpole still standing using 30-60-90 triangle properties
  let x := (1/2) * hypotenuse
  have h_x : x = 7.5, from calc
    x = (1/2) * 15 : rfl
     .. = 7.5 : rfl,
  -- The height at which the flagpole broke is the original height minus the dangling part
  exact h_x.symm

end flagpole_break_height_l446_446576


namespace minValue_expression_l446_446354

noncomputable def minValue (x y : ℝ) : ℝ :=
  4 / x^2 + 4 / (x * y) + 1 / y^2

theorem minValue_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (x - 2 * y)^2 = (x * y)^3) :
  minValue x y = 4 * Real.sqrt 2 :=
sorry

end minValue_expression_l446_446354


namespace base_number_l446_446722

theorem base_number (base y : ℤ) (h1 : base ^ y = 3 ^ 14) (h2 : y = 7) : base = 9 := 
by
  sorry

end base_number_l446_446722


namespace jeans_price_increase_l446_446982

theorem jeans_price_increase
  (C R P : ℝ)
  (h1 : P = 1.15 * R)
  (h2 : P = 1.6100000000000001 * C) :
  R = 1.4 * C :=
by
  sorry

end jeans_price_increase_l446_446982


namespace exists_1998_distinct_natural_numbers_l446_446625

noncomputable def exists_1998_distinct_numbers : Prop :=
  ∃ (s : Finset ℕ), s.card = 1998 ∧
    (∀ {x y : ℕ}, x ∈ s → y ∈ s → x ≠ y → (x * y) % ((x - y) ^ 2) = 0)

theorem exists_1998_distinct_natural_numbers : exists_1998_distinct_numbers :=
by
  sorry

end exists_1998_distinct_natural_numbers_l446_446625


namespace incorrect_proposition_1_incorrect_proposition_3_l446_446440

section PlanesAndLines

variables (α β : Type) (a b : Type)

def A (α : Type) : Set Type := {p | is_perpendicular p α}
def B (β : Type) : Set Type := {p | is_perpendicular p β}
def M (a : Type) : Set Type := {l | is_perpendicular l a}
def N (b : Type) : Set Type := {l | is_perpendicular l b}

theorem incorrect_proposition_1 (h : A α ∩ B β ≠ ∅) : ¬ parallel α β := sorry
theorem incorrect_proposition_3 (h : skew_lines a b) : M a ∩ N b ≠ ∅ := sorry

end PlanesAndLines

end incorrect_proposition_1_incorrect_proposition_3_l446_446440


namespace range_of_a_with_common_tangent_l446_446364

noncomputable def has_common_tangent (a x₁ x₂ : ℝ) (pos_x1 : x₁ > 0) (pos_x2 : x₂ > 0) :=
  (-a / (x₁ ^ 2) = 2 / x₂) ∧ (2 * a / x₁ = 2 * (ln x₂) - 2)

theorem range_of_a_with_common_tangent : 
  ∃ (a : ℝ), ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → has_common_tangent a x₁ x₂ x₁ x₂ → (-2 / real.exp(1)) ≤ a ∧ a < 0 :=
sorry

end range_of_a_with_common_tangent_l446_446364


namespace find_k_l446_446669

def point (x y : ℝ) := (x, y)

def line (k m : ℝ) : ℝ → ℝ := fun x => k * x - m

def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def area_triangle (A B C : (ℝ × ℝ)) : ℝ := 
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

def S₁ (A B C : (ℝ × ℝ)) : ℝ := area_triangle A B C

def S₂ (O B C : (ℝ × ℝ)) : ℝ := area_triangle O B C

theorem find_k (k m : ℝ) (A O B C : (ℝ × ℝ)) (θ : ℝ) (h_circ : circle O.1 O.2) 
  (h_line : line k m = fun y => k * y - m) (h_A : A = (0, 1)) 
  (h_angle : θ = (60 : ℝ)) (h_S₁ := 2 * S₂ O B C) : k = ±√3 :=
by
  sorry

end find_k_l446_446669


namespace no_real_solutions_l446_446113

def operation (x y : ℝ) : ℝ := (x - y) / (x * y)

theorem no_real_solutions : ¬∃ a : ℝ, operation a (operation a 2) = 1 :=
by
  sorry

end no_real_solutions_l446_446113


namespace sum_of_squares_of_perfect_squares_less_than_500_l446_446149

theorem sum_of_squares_of_perfect_squares_less_than_500 :
  (∑ n in {1, 16, 81, 256}, n) = 354 :=
by {
  sorry
}

end sum_of_squares_of_perfect_squares_less_than_500_l446_446149


namespace problem_f_2017_9_l446_446713

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

noncomputable def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

noncomputable def f_iter : ℕ → ℕ → ℕ
| 0, n := n
| (m + 1), n := f (f_iter m n)

theorem problem_f_2017_9 :
  f_iter 2017 9 = 8 :=
sorry

end problem_f_2017_9_l446_446713


namespace increase_p_does_not_always_increase_equal_points_l446_446918

-- Define the function representing equal points probability
def equal_points_probability (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

-- The main theorem states that increasing the probability 'p' of a draw 
-- does not necessarily increase the probability of the teams acquiring equal points.
theorem increase_p_does_not_always_increase_equal_points :
  ∃ p₁ p₂ : ℝ, 0 ≤ p₁ ∧ p₁ < p₂ ∧ p₂ ≤ 1 ∧ equal_points_probability p₁ ≥ equal_points_probability p₂ :=
by
  sorry

end increase_p_does_not_always_increase_equal_points_l446_446918


namespace dot_product_focus_line_fixed_point_line_l446_446747

-- Definitions and parameters for the points and conditions
def parabola : set (ℝ × ℝ) := { p | ∃ y, p = (y^2 / 4, y) }

def focus : (ℝ × ℝ) := (1, 0)

def is_line_through_point (l : ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  ∃ m b, ∀ (x y : ℝ), l (x, y) ↔ (y = m * x + b) ∧ (m * p.1 + b = p.2)

def intersect_at_two_points (l : ℝ × ℝ → Prop) : Prop :=
  ∃ A B, A ≠ B ∧ parabola A ∧ parabola B ∧ l A ∧ l B

-- (1) Proof problem statement
theorem dot_product_focus_line (l : ℝ × ℝ → Prop)
  (hl : is_line_through_point l focus)
  (hi : intersect_at_two_points l)
  : ∃ A B, A ≠ B ∧ parabola A ∧ parabola B ∧ l A ∧ l B ∧
    (let (x₁, y₁) := A, (x₂, y₂) := B in x₁ * x₂ + y₁ * y₂ = -3) :=
sorry

-- (2) Proof problem statement
theorem fixed_point_line (l : ℝ × ℝ → Prop)
  (h_dot_product : ∃ A B, A ≠ B ∧ parabola A ∧ parabola B ∧ l A ∧ l B ∧
    (let (x₁, y₁) := A, (x₂, y₂) := B in x₁ * x₂ + y₁ * y₂ = -4))
  : is_line_through_point l (2, 0) :=
sorry

end dot_product_focus_line_fixed_point_line_l446_446747


namespace solve_system_of_equations_l446_446827
noncomputable theory

theorem solve_system_of_equations 
  (x y : ℝ)
  (h1 : 3 * x + y = 8)
  (h2 : 2 * x - y = 7) :
  x = 3 ∧ y = -1 :=
sorry

end solve_system_of_equations_l446_446827


namespace tamia_bell_pepper_pieces_l446_446092

theorem tamia_bell_pepper_pieces :
  let bell_peppers := 5
  let slices_per_pepper := 20
  let initial_slices := bell_peppers * slices_per_pepper
  let half_slices_cut := initial_slices / 2
  let small_pieces := half_slices_cut * 3
  let total_pieces := (initial_slices - half_slices_cut) + small_pieces
  total_pieces = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l446_446092


namespace fraction_of_red_knights_magical_l446_446108

variable (total_knights : ℕ) (red_knights : ℕ) (blue_knights : ℕ)
variable (magical_knights : ℕ) (magical_fraction_red : ℚ) (magical_fraction_blue : ℚ)
variable (red_fraction : ℚ) (magical_total_fraction : ℚ)

-- Given conditions
def conditions := 
  (total_knights = 64) ∧ 
  (red_knights = (3 / 8 * total_knights).to_nat) ∧ 
  (blue_knights = total_knights - red_knights) ∧ 
  (magical_knights = (1 / 8 * total_knights).to_nat) ∧ 
  (magical_fraction_red = 3 * magical_fraction_blue)

-- Proving that the fraction of red knights who are magical is 3/14
theorem fraction_of_red_knights_magical : conditions →
  magical_fraction_red = 3 / 14 := 
sorry

end fraction_of_red_knights_magical_l446_446108


namespace product_remainder_mod_7_l446_446858

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l446_446858


namespace sum_of_four_non_neg_nums_ge_50_l446_446122

theorem sum_of_four_non_neg_nums_ge_50 {n : ℕ} (a : Fin n → ℝ) (h1 : (∀ i, 0 ≤ a i)) 
  (h_sum_le_200 : ∑ i, a i ≤ 200) (h_sum_sq_ge_2500 : ∑ i, (a i)^2 ≥ 2500) :
  ∃ (i j k l : Fin n), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    a i + a j + a k + a l ≥ 50 := by
    sorry

end sum_of_four_non_neg_nums_ge_50_l446_446122


namespace melted_mixture_weight_l446_446944

theorem melted_mixture_weight (Z C : ℝ) (h_ratio : Z / C = 9 / 11) (h_zinc : Z = 28.8) : Z + C = 64 :=
by
  sorry

end melted_mixture_weight_l446_446944


namespace sum_max_at_19_l446_446690

def a (n : ℕ) : ℤ := -4 * n + 78
def S (n : ℕ) : ℤ := (finset.range (n + 1)).sum a

theorem sum_max_at_19 : ∃ n : ℕ, S n = S 19 :=
begin
  use 19,
  sorry
end

end sum_max_at_19_l446_446690


namespace area_ratio_of_inner_region_l446_446528

-- Defining the regular hexagon and the points
structure RegularHexagon (A B C D E F : Point) :=
  (side_length : ℝ)

-- Define trisected points and properties
structure TrisectedHexagon (A1 A2 ... A18 : Point) :=
  (is_regular: RegularHexagon)
  (trisected: ∀ i, i ∈ {1, 2, ..., 18} → True) -- Simplifying representation of the 18 points
  
-- Function to calculate the area ratio
def area_ratio (hex : TrisectedHexagon) : ℝ := 
  sorry

-- Main theorem statement
theorem area_ratio_of_inner_region (A B C D E F : Point) :
  (RegularHexagon A B C D E F) →
  (TrisectedHexagon A1 A2 ... A18) →
  area_ratio (TrisectedHexagon A1 A2 ... A18) = 9/13 :=
sorry

end area_ratio_of_inner_region_l446_446528


namespace transformed_function_correct_l446_446082

-- Definitions of the original function and transformations
def original_function (x : ℝ) : ℝ := sin (x - π / 3)

-- Transformation 1: Stretch x-coordinates
def stretch_x (f : ℝ → ℝ) (x : ℝ) : ℝ := f (1 / 2 * x)

-- Transformation 2: Shift graph to the left
def shift_left (f : ℝ → ℝ) (d : ℝ) (x : ℝ) : ℝ := f (x + d)

-- Apply transformations to the original function
def transformed_function (x : ℝ) : ℝ :=
  shift_left (stretch_x original_function) (π / 3) x

-- Simplified result
def expected_function (x : ℝ) : ℝ := sin (1 / 2 * x - π / 6)

-- Proof statement
theorem transformed_function_correct : ∀ x : ℝ, transformed_function x = expected_function x :=
by
  intros x
  sorry

end transformed_function_correct_l446_446082


namespace weighted_average_correct_l446_446267

-- Define the marks and credits for each subject
def marks_english := 90
def marks_mathematics := 92
def marks_physics := 85
def marks_chemistry := 87
def marks_biology := 85

def credits_english := 3
def credits_mathematics := 4
def credits_physics := 4
def credits_chemistry := 3
def credits_biology := 2

-- Define the weighted sum and total credits
def weighted_sum := marks_english * credits_english + marks_mathematics * credits_mathematics + marks_physics * credits_physics + marks_chemistry * credits_chemistry + marks_biology * credits_biology
def total_credits := credits_english + credits_mathematics + credits_physics + credits_chemistry + credits_biology

-- Prove that the weighted average is 88.0625
theorem weighted_average_correct : (weighted_sum.toFloat / total_credits.toFloat) = 88.0625 :=
by 
  sorry

end weighted_average_correct_l446_446267


namespace solve_for_x_l446_446077

theorem solve_for_x (x : ℝ)
  (h : 16^x * 16^x * 16^x * 16^x = 256^10) :
  x = 5 :=
sorry

end solve_for_x_l446_446077


namespace value_expression_l446_446720

theorem value_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by 
  sorry

end value_expression_l446_446720


namespace find_x_when_y_neg4_l446_446084

variable {x y : ℝ}
variable (k : ℝ)

-- Condition: x is inversely proportional to y
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop :=
  x * y = k

theorem find_x_when_y_neg4 (h : inversely_proportional 5 10 50) :
  inversely_proportional x (-4) 50 → x = -25 / 2 :=
by sorry

end find_x_when_y_neg4_l446_446084


namespace base_7_multiplication_addition_l446_446803

theorem base_7_multiplication_addition :
  (25 * 3 + 144) % 7^3 = 303 :=
by sorry

end base_7_multiplication_addition_l446_446803


namespace solve_for_x_l446_446471

theorem solve_for_x (x : ℝ) (h : 4 * x + 45 ≠ 0) :
  (8 * x^2 + 80 * x + 4) / (4 * x + 45) = 2 * x + 3 → x = -131 / 22 := 
by 
  sorry

end solve_for_x_l446_446471


namespace complex_inequality_iff_l446_446263

noncomputable theory

variables (v w : ℂ) (z : ℂ) (k : ℝ)

/- distinct non-zero complex numbers -/
def distinct_nonzero (v w : ℂ) := v ≠ w ∧ v ≠ 0 ∧ w ≠ 0

/- given conditions: |z| = 1 -/
def unit_modulus (z : ℂ) := abs z = 1

/- goal statement -/
theorem complex_inequality_iff (h1 : distinct_nonzero v w) (h2 : unit_modulus z) :
  (∀ z : ℂ, abs z = 1 → abs (z * w + conj w) ≤ abs (z * v + conj v)) ↔
  ∃ k ∈ Icc (-1 : ℝ) 1, w = k * v :=
sorry

end complex_inequality_iff_l446_446263


namespace probability_nonagon_diagonal_intersect_l446_446953

theorem probability_nonagon_diagonal_intersect (n : ℕ) (h_n : n = 9) :
  let diagonals := (n.choose 2) - n,
      total_diagonals_pairs := (diagonals.choose 2),
      intersecting_pairs := (n.choose 4)
  in (intersecting_pairs : ℚ) / total_diagonals_pairs = 14 / 39 :=
by {
  sorry
}

end probability_nonagon_diagonal_intersect_l446_446953


namespace gray_part_area_l446_446136

theorem gray_part_area (area_rect1 area_rect2 area_black area_white gray_part_area : ℕ)
  (h_rect1 : area_rect1 = 80)
  (h_rect2 : area_rect2 = 108)
  (h_black : area_black = 37)
  (h_white : area_white = area_rect1 - area_black)
  (h_white_correct : area_white = 43)
  : gray_part_area = area_rect2 - area_white :=
by
  sorry

end gray_part_area_l446_446136


namespace john_dog_expenses_l446_446410

theorem john_dog_expenses
  (vet_cost : Nat)
  (num_vet_appointments : Nat)
  (medication : Nat)
  (grooming_services : Nat)
  (pet_food : Nat)
  (insurance_cost : Nat)
  (insurance_coverage_vet : Nat)
  (insurance_coverage_med : Nat) :
  vet_cost = 400 → num_vet_appointments = 3 →
  medication = 250 → grooming_services = 120 →
  pet_food = 300 → insurance_cost = 100 →
  insurance_coverage_vet = 80 → insurance_coverage_med = 50 →
  let total_vet_cost := (num_vet_appointments * vet_cost)
  let first_vet_appointment_cost := vet_cost
  let subsequent_vet_cost := 2 * vet_cost * (100 - insurance_coverage_vet) / 100
  let total_med_cost := medication * (100 - insurance_coverage_med) / 100
  let total_cost := first_vet_appointment_cost + subsequent_vet_cost +
                    total_med_cost + grooming_services + pet_food + insurance_cost
  in total_cost = 1205 := 
begin
  intros,
  rw [total_vet_cost, first_vet_appointment_cost, subsequent_vet_cost,
      total_med_cost, total_cost],
  ring,
  sorry,
end

end john_dog_expenses_l446_446410


namespace m_mul_m_add_1_not_power_of_integer_l446_446071

theorem m_mul_m_add_1_not_power_of_integer (m n k : ℕ) : m * (m + 1) ≠ n^k :=
by
  sorry

end m_mul_m_add_1_not_power_of_integer_l446_446071


namespace area_QPOC_eq_17k_over_16_l446_446397

variables {A B C D P Q N M O: Type*}
variables [Trapezoid ABCD] (k : ℝ)
variables [Geometric_BBisector DP BC N AC P]
variables [Geometric_ABisector CQ AD M BD Q]
variables [Intersection DP CQ O]

theorem area_QPOC_eq_17k_over_16 (hABparallelCD: Parallel AB CD)
                                 (hAreaTrapezoid: Area(Trapezoid ABCD) = k):
  Area(Quadrilateral QPOC) = (17 * k) / 16 :=
sorry

end area_QPOC_eq_17k_over_16_l446_446397


namespace oranges_savings_l446_446791

-- Definitions for the conditions
def liam_oranges : Nat := 40
def liam_price_per_set : Real := 2.50
def oranges_per_set : Nat := 2

def claire_oranges : Nat := 30
def claire_price_per_orange : Real := 1.20

-- Statement of the problem to be proven
theorem oranges_savings : 
  liam_oranges / oranges_per_set * liam_price_per_set + 
  claire_oranges * claire_price_per_orange = 86 := 
by 
  sorry

end oranges_savings_l446_446791


namespace triangle_similarity_l446_446133

theorem triangle_similarity
  (GH FG IJ : ℝ)
  (hGH : GH = 30)
  (hFG : FG = 24)
  (hIJ : IJ = 20)
  (sim : Triangle G H F ≅ Triangle I J K)
  : ∃ JK : ℝ, JK = 25 := by
  sorry

end triangle_similarity_l446_446133


namespace range_of_x_l446_446337

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then x else Real.log (x + 1)

theorem range_of_x 
  (x : ℝ) 
  (h : f (2 - x^2) > f x) : 
  -2 < x ∧ x < 1 :=
by 
  sorry

end range_of_x_l446_446337


namespace pond_length_l446_446382

theorem pond_length (V W D L : ℝ) (hV : V = 1600) (hW : W = 10) (hD : D = 8) :
  L = 20 ↔ V = L * W * D :=
by
  sorry

end pond_length_l446_446382


namespace correct_propositions_l446_446658

-- Definitions of lines, planes, and relationships among them
variables {m n : Type} [linear_space m] [linear_space n]
variables {α β : Type} [plane_space α] [plane_space β]

-- Definitions of different types of relationships
def line_intersect_plane (m : Type) (α : Type) [linear_space m] [plane_space α] : Prop := sorry
def plane_perpendicular (α β : Type) [plane_space α] [plane_space β] : Prop := sorry
def plane_parallel (α β : Type) [plane_space α] [plane_space β] : Prop := sorry
def line_perpendicular (m n : Type) [linear_space m] [linear_space n] : Prop := sorry
def line_parallel (m n : Type) [linear_space m] [linear_space n] : Prop := sorry
def plane_contains_line (α : Type) (m : Type) [plane_space α] [linear_space m] : Prop := sorry

-- Proposition 1
def prop1 (m n : Type) [linear_space m] [linear_space n] (α β : Type) [plane_space α] [plane_space β] : Prop :=
  line_intersect_plane m α ∧ plane_contains_line α n ∧ line_perpendicular n m → plane_perpendicular α β

-- Proposition 2
def prop2 (m : Type) [linear_space m] (α β : Type) [plane_space α] [plane_space β] : Prop :=
  line_perpendicular m α ∧ line_perpendicular m β → plane_parallel α β

-- Proposition 3
def prop3 (m n : Type) [linear_space m] [linear_space n] (α β : Type) [plane_space α] [plane_space β] : Prop :=
  line_perpendicular m α ∧ line_perpendicular n β ∧ line_perpendicular m n → plane_perpendicular α β

-- Proposition 4
def prop4 (m n : Type) [linear_space m] [linear_space n] (α β : Type) [plane_space α] [plane_space β] : Prop :=
  line_parallel m α ∧ line_parallel n β ∧ line_parallel m n → plane_parallel α β

-- The statement we want to prove
theorem correct_propositions :
  prop1 m n α β ∧ 
  prop2 m α β ∧ 
  prop3 m n α β ∧ 
  ¬ prop4 m n α β :=
sorry

end correct_propositions_l446_446658


namespace descending_order_of_weights_l446_446109

variables (S B K A : ℝ)

theorem descending_order_of_weights
  (h1 : S > B)
  (h2 : A + B > S + K)
  (h3 : K + A = S + B) :
  A > S ∧ S > B ∧ B > K :=
by
  -- Sketch of proof for illustration:
  -- From h3: K + A = S + B, we can rearrange to get A = S + B - K
  -- Substitute A in h2: S + B - K + B > S + K
  -- Simplify: S + 2B - K > S + K
  -- Therefore: 2B > 2K, hence B > K
  -- From h1 and derived B > K, we also have A > S
  -- Summarize the order: A > S > B > K
  sorry

end descending_order_of_weights_l446_446109


namespace quadratic_root_difference_l446_446078

theorem quadratic_root_difference (p : ℝ) :
  let a : ℝ := 1
  let b : ℝ := -2 * p
  let c : ℝ := p^2 - 4
  ∃ (r s : ℝ), (r >= s) ∧ (a * r^2 + b * r + c = 0) ∧ (a * s^2 + b * s + c = 0) ∧ (r - s = 4) := 
begin
  sorry
end

end quadratic_root_difference_l446_446078


namespace round_robin_product_of_four_consecutive_points_l446_446650

theorem round_robin_product_of_four_consecutive_points :
  ∃ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d ∧
    (∀ i ∈ {a, b, c, d}, i > 0) ∧
    (a + 1 = b ∧ b + 1 = c ∧ c + 1 = d) ∧
    (6 * 2 ≤ a + b + c + d ∧ a + b + c + d ≤ 6 * 3) ∧
    (a * b * c * d = 120) := 
sorry

end round_robin_product_of_four_consecutive_points_l446_446650


namespace min_cos_angle_l446_446668

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (ha : a ≠ 0) (hb : b ≠ 0)
variables (hcond : (a • a) = (5 • a - 4 • b) • b)

theorem min_cos_angle :
  cos_angle a b ≥ (4 / 5) := by
  sorry

end min_cos_angle_l446_446668


namespace factor_is_two_l446_446590

theorem factor_is_two (n f : ℤ) (h1 : n = 121) (h2 : n * f - 140 = 102) : f = 2 :=
by
  sorry

end factor_is_two_l446_446590


namespace point_not_on_graph_l446_446938

-- Define the function given in the problem
def func (x : ℝ) : ℝ :=
  if x ≠ -2 then 2 * x / (x + 2) else 0  -- The function is defined for all x ≠ -2, we can define a dummy value of 0 for x = -2

-- Prove the point (x, y) = (-2, -1) is not on the graph
theorem point_not_on_graph : ¬ (func (-2) = -1) :=
by {
  -- Directly show that the function is undefined at x = -2
  sorry
}

end point_not_on_graph_l446_446938


namespace diameter_of_larger_circle_l446_446095

theorem diameter_of_larger_circle (R r D : ℝ) 
  (h1 : R^2 - r^2 = 25) 
  (h2 : D = 2 * R) : 
  D = Real.sqrt (100 + 4 * r^2) := 
by 
  sorry

end diameter_of_larger_circle_l446_446095


namespace tamia_bell_pepper_pieces_l446_446091

theorem tamia_bell_pepper_pieces :
  let bell_peppers := 5
  let slices_per_pepper := 20
  let initial_slices := bell_peppers * slices_per_pepper
  let half_slices_cut := initial_slices / 2
  let small_pieces := half_slices_cut * 3
  let total_pieces := (initial_slices - half_slices_cut) + small_pieces
  total_pieces = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l446_446091


namespace intersecting_diagonals_prob_nonagon_l446_446956

theorem intersecting_diagonals_prob_nonagon :
  let num_vertices := 9
  let num_diagonals := (nat.choose num_vertices 2) - num_vertices
  let pairs_of_diagonals := nat.choose num_diagonals 2
  let num_intersecting_diagonals := nat.choose num_vertices 4
  (num_intersecting_diagonals / pairs_of_diagonals) = 6 / 13 :=
by
  sorry

end intersecting_diagonals_prob_nonagon_l446_446956


namespace small_triangles_count_l446_446210

theorem small_triangles_count (n : ℕ) (h : n = 2012) : 
    let total_points := n + 3 in 
    let no_three_collinear := ∀ (p1 p2 p3 : ℕ), (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) → 
                              ¬ collinear_points p1 p2 p3
    in count_small_triangles total_points no_three_collinear = 2 * n + 1 :=
begin
  sorry
end

end small_triangles_count_l446_446210


namespace cody_lost_tickets_l446_446250

/--
Cody had 49.0 tickets initially, spent 25.0 tickets, and has 18 tickets left.
Prove that the number of tickets Cody lost is 6.0.
-/
theorem cody_lost_tickets :
  ∀ (initial_tickets spent_tickets remained_tickets lost_tickets : ℝ),
  initial_tickets = 49.0 →
  spent_tickets = 25.0 →
  remained_tickets = 18.0 →
  lost_tickets = (initial_tickets - spent_tickets) - remained_tickets →
  lost_tickets = 6.0 :=
by
  intros initial_tickets spent_tickets remained_tickets lost_tickets
  intros h_initial h_spent h_remaining h_lost
  simp only [h_initial, h_spent, h_remaining] at h_lost
  rw h_lost
  exact rfl

end cody_lost_tickets_l446_446250


namespace first_integer_is_11_l446_446213

-- We define the conditions of the problem:
def consecutive_odd_integers (x : ℤ) : Prop :=
  let x1 := x + 2
  let x2 := x + 4
  x + x1 + x2 = x1 + 17

-- The Lean 4 theorem to be proved:
theorem first_integer_is_11 : ∃ x : ℤ, consecutive_odd_integers x ∧ x = 11 :=
begin
  sorry
end

end first_integer_is_11_l446_446213


namespace smallest_term_Sn_l446_446666

-- Definitions based on conditions in the problem
variable {a : ℕ → ℤ} -- Define the arithmetic sequence

definition S (n : ℕ) : ℤ := (n * (a 1 + a n)) / 2 -- Define the sum of the first n terms S_n

-- Stating the theorem based on the condition and the question
theorem smallest_term_Sn (a : ℕ → ℤ) (h1 : S 17 < 0) (h2 : S 18 > 0) : 
  ∃ n, ∀ m < n, S m < 0 ∧ ∀ p ≥ n, S p ≥ 0 ∧ n = 8 :=
sorry

end smallest_term_Sn_l446_446666


namespace q_is_false_l446_446726

variable {p q : Prop}

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end q_is_false_l446_446726


namespace problem_statement_l446_446333

def mutually_exclusive (A B : Prop) : Prop := ¬ (A ∧ B)

def cond1 : Prop := mutually_exclusive (∃ (a : Type), a = 7) (∃ (a : Type), a = 8)

def cond2 (A B : Type) : Prop := ¬ mutually_exclusive (∃ (a : A), a = 7) (∃ (b : B), b = 8)

def cond3 (A B : Type) : Prop := mutually_exclusive (∀ (a : A) (b : B), true) (∀ (a : A) (b : B), false)

def cond4 (A B : Type) : Prop := ¬ mutually_exclusive (∃ (a : A) (b : B), true) (∃ (a : A) (b : B), true)

theorem problem_statement : (cond1) ∧ (¬ cond2 ℕ ℕ) ∧ (cond3 ℕ ℕ) ∧ (¬ cond4 ℕ ℕ) → 2 = 2 := 
by {
  intros,
  sorry
}

end problem_statement_l446_446333


namespace function_range_l446_446684

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sin x

theorem function_range : 
  ∀ x : ℝ, (0 < x ∧ x < Real.pi) → 1 ≤ f x ∧ f x ≤ 3 / 2 :=
by
  intro x
  sorry

end function_range_l446_446684


namespace who_is_correct_l446_446010

-- Definitions based on conditions
variables (Italians Swedes : ℕ)
variables (total_racers : ℕ)
variables (tall_blondes short_brunettes : ℕ)

-- Given conditions
axiom equal_number_of_racers : Italians = Swedes
axiom swedes_mostly_tall_blondes : ∀ S : ℕ, Swedes = S → 4 * S / 5 = tall_blondes
axiom swedes_one_fifth_short_brunettes : ∀ S : ℕ, Swedes = S → 1 * S / 5 = short_brunettes
axiom no_tall_brunettes_no_short_blondes : ∀ T : ℕ, total_racers = T → tall_blondes + short_brunettes = T
axiom blondes_to_brunettes_ratio : ∀ T : ℕ, total_racers = T → tall_blondes = 2 * T / 5 ∧ short_brunettes = 3 * T / 5

-- Problem statement
theorem who_is_correct (tall_blonde_appeared : ∀ T : ℕ, total_racers = T → T > 0 → ∃ S : ℕ, Swedes = S ∧ 4 * S / 5 = tall_blondes):
  (∃ S : ℕ, Swedes = S ∧ 4 * S / 5 = tall_blondes) → (tall_blonde_appeared total_racers) :=
  sorry

end who_is_correct_l446_446010


namespace value_of_h_l446_446352

theorem value_of_h (h : ℝ) : (∃ x : ℝ, x^3 + h * x - 14 = 0 ∧ x = 3) → h = -13/3 :=
by
  sorry

end value_of_h_l446_446352


namespace temperature_difference_l446_446555

-- Definitions based on the conditions
def refrigeration_compartment_temperature : ℤ := 5
def freezer_compartment_temperature : ℤ := -2

-- Mathematically equivalent proof problem statement
theorem temperature_difference : refrigeration_compartment_temperature - freezer_compartment_temperature = 7 := by
  sorry

end temperature_difference_l446_446555


namespace solution_to_part1_solution_to_part2_solution_to_part3_l446_446968

-- Conditions from the problem
variables (m n : ℕ) (cost_A cost_B : ℕ)
variable (x : ℕ) 
variables (profit margin : ℝ)
variables (a : ℝ)

-- Definitions based on conditions
def eq_1 : Prop := 15 * m + 20 * n = 430
def eq_2 : Prop := 10 * m + 8 * n = 212
def total_investment_condition : Prop := x <= 100 ∧ x <= 60
def total_investment : Prop := 10 * x + 14 * (100 - x) <= 1168
def donation_condition : Prop := 2 * a ≤ 1.8
def profit_condition : Prop := profit ≥ margin * (10 * 60 + 14 * 40)

-- Statements of the problem
theorem solution_to_part1 : eq_1 ∧ eq_2 → m = 10 ∧ n = 14 := by
  sorry

theorem solution_to_part2 : total_investment_condition ∧ total_investment → x ∈ {58, 59, 60} := by
  sorry

theorem solution_to_part3 : profit - 160 * a ≥ 233.6 → a ≤ 1.8 := by 
  sorry

end solution_to_part1_solution_to_part2_solution_to_part3_l446_446968


namespace find_number_of_girls_l446_446196

theorem find_number_of_girls (B G : ℕ) 
  (h1 : B + G = 604) 
  (h2 : 12 * B + 11 * G = 47 * 604 / 4) : 
  G = 151 :=
by
  sorry

end find_number_of_girls_l446_446196


namespace sum_binomial_equality_l446_446261

theorem sum_binomial_equality :
  (1 / 2 ^ 2024) * (∑ n in Finset.range 1012, (-3 : ℝ) ^ n * Nat.choose 2024 (2 * n)) = -1 / 2 :=
by
  sorry

end sum_binomial_equality_l446_446261


namespace increase_p_does_not_always_increase_equal_points_l446_446916

-- Define the function representing equal points probability
def equal_points_probability (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

-- The main theorem states that increasing the probability 'p' of a draw 
-- does not necessarily increase the probability of the teams acquiring equal points.
theorem increase_p_does_not_always_increase_equal_points :
  ∃ p₁ p₂ : ℝ, 0 ≤ p₁ ∧ p₁ < p₂ ∧ p₂ ≤ 1 ∧ equal_points_probability p₁ ≥ equal_points_probability p₂ :=
by
  sorry

end increase_p_does_not_always_increase_equal_points_l446_446916


namespace equal_points_probability_l446_446903

theorem equal_points_probability (p : ℝ) (prob_draw_increases : 0 ≤ p ∧ p ≤ 1) :
  (∀ q : ℝ, (0 ≤ q ∧ q < p) → (q^2 + (1 - q)^2 / 2 < p^2 + (1 - p)^2 / 2)) → False :=
begin
  sorry
end

end equal_points_probability_l446_446903


namespace area_difference_675pi_l446_446350

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  π * r^2

theorem area_difference_675pi (r1 r2 d2 : ℝ) (h_r1 : r1 = 30) (h_d2 : d2 = 30) (h_r2 : r2 = d2 / 2) :
  (area_of_circle r1 - area_of_circle r2) = 675 * π :=
by sorry

end area_difference_675pi_l446_446350


namespace sum_of_first_4_terms_l446_446736

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem sum_of_first_4_terms (a r : ℝ) 
  (h1 : a * (1 + r + r^2) = 13) (h2 : a * (1 + r + r^2 + r^3 + r^4) = 121) : 
  a * (1 + r + r^2 + r^3) = 27.857 :=
by
  sorry

end sum_of_first_4_terms_l446_446736


namespace min_empty_cells_l446_446742

-- The definition of the problem conditions
structure Board :=
  (size : ℕ) -- the board is size x size (here 8x8)
  (occupiedCells : ℕ) -- number of cells occupied by triangles
  (maxTriangles : ℕ → Prop) -- a property defining the maximum number of triangles that fit in the grid

-- Given this specific configuration for our board:
def chessBoard : Board :=
{ size := 8,
  occupiedCells := 64,
  maxTriangles := λ k, k ≤ 40 -- given the configuration described, maximum possible triangles without overlap is 40
}

-- Defining the minimum number of empty cells as a theorem
theorem min_empty_cells (b : Board) : b.size = 8 → b.occupiedCells = 64 → b.maxTriangles 40 → 
  (64 - 40 = 24) :=
by
  sorry

end min_empty_cells_l446_446742


namespace number_of_pencils_purchased_l446_446567

variable {total_pens : ℕ} (total_cost : ℝ) (avg_price_pencil avg_price_pen : ℝ)

theorem number_of_pencils_purchased 
  (h1 : total_pens = 30)
  (h2 : total_cost = 570)
  (h3 : avg_price_pencil = 2.00)
  (h4 : avg_price_pen = 14)
  : 
  ∃ P : ℕ, P = 75 :=
by
  sorry

end number_of_pencils_purchased_l446_446567


namespace oldest_child_age_l446_446098

theorem oldest_child_age 
  (x : ℕ)
  (h1 : (6 + 8 + 10 + x) / 4 = 9)
  (h2 : 6 + 8 + 10 = 24) :
  x = 12 := 
by 
  sorry

end oldest_child_age_l446_446098


namespace tino_jellybeans_l446_446895

theorem tino_jellybeans (Tino Lee Arnold Joshua : ℕ)
  (h1 : Tino = Lee + 24)
  (h2 : Arnold = Lee / 2)
  (h3 : Joshua = 3 * Arnold)
  (h4 : Arnold = 5) : Tino = 34 := by
sorry

end tino_jellybeans_l446_446895


namespace remainder_of_7_pow_145_mod_9_l446_446144

theorem remainder_of_7_pow_145_mod_9 : (7 ^ 145) % 9 = 7 := by
  sorry

end remainder_of_7_pow_145_mod_9_l446_446144


namespace three_digit_number_108_l446_446286

theorem three_digit_number_108 (a b c : ℕ) (ha : a ≠ 0) (h₀ : a < 10) (h₁ : b < 10) (h₂ : c < 10) (h₃: 100*a + 10*b + c = 12*(a + b + c)) : 
  100*a + 10*b + c = 108 := 
by 
  sorry

end three_digit_number_108_l446_446286


namespace vector_sub_bc_l446_446701

-- Define vector types
def Vector := (Int × Int)

-- Define the vectors AB and AC
def AB : Vector := (2, 3)
def AC : Vector := (4, 7)

-- Define vector subtraction
def vector_sub (v1 v2 : Vector) : Vector :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- State the theorem
theorem vector_sub_bc : vector_sub AC AB = (2, 4) :=
by
  -- Proof would go here
  sorry

end vector_sub_bc_l446_446701


namespace matrices_condition_even_n_l446_446620

theorem matrices_condition_even_n (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) 
  (hA : Invertible A) (hB : Invertible B) (h : A ⬝ B - B ⬝ A = B ⬝ B ⬝ A) : Even n := 
sorry

end matrices_condition_even_n_l446_446620


namespace knight_possible_moves_l446_446986

theorem knight_possible_moves (a b : ℤ) :
  ∃ seq1 seq2 : list (ℤ × ℤ), 
    (seq1.head = (0, 0) ∧ seq1.last = (1, a) ∧ seq1.transforms from 0 to b) ∧ 
    (seq2.head = (0, 0) ∧ seq2.last = (-1, a) ∧ seq2.transforms from 0 to b) :=
sorry

end knight_possible_moves_l446_446986


namespace triangle_perimeter_l446_446110

theorem triangle_perimeter (a b c : ℕ) (root1 root2 : ℕ) 
  (h_roots : ∀ (x : ℕ), (x = root1 ∨ x = root2) → (x^2 - 7 * x + 10) = 0) 
  (h_side_lengths : {a, b, c} ⊆ {root1, root2}) :
  a + b + c = 12 ∨ a + b + c = 6 ∨ a + b + c = 15 :=
sorry

end triangle_perimeter_l446_446110


namespace sum_of_valid_fourth_powers_is_354_l446_446150

-- Given conditions
def is_valid_fourth_power (n : ℕ) : Prop := n^4 < 500

-- The main theorem we want to prove
theorem sum_of_valid_fourth_powers_is_354 :
  let valid_n := { n // is_valid_fourth_power n }
  let fourth_powers := { n : ℕ | is_valid_fourth_power n }.to_finset.val
  fourth_powers.sum = 354 :=
by
  -- Proof omitted
  sorry

end sum_of_valid_fourth_powers_is_354_l446_446150


namespace annual_haircut_cost_l446_446029

-- Define conditions as Lean definitions
def hair_growth_rate_per_month : ℝ := 1.5
def initial_hair_length : ℝ := 9
def post_haircut_length : ℝ := 6
def haircut_cost : ℝ := 45
def tip_percentage : ℝ := 0.2

-- The question to be answered as a theorem
theorem annual_haircut_cost :
  (hair_growth_rate_per_month > 0) →
  (initial_hair_length > post_haircut_length) →
  let length_cut := initial_hair_length - post_haircut_length in
  let months_between_haircuts := length_cut / hair_growth_rate_per_month in
  let haircuts_per_year := 12 / months_between_haircuts in
  let tip_amount := haircut_cost * tip_percentage in
  let cost_per_haircut := haircut_cost + tip_amount in
  haircuts_per_year * cost_per_haircut = 324 := 
by 
  -- Skipping the proof
  sorry

end annual_haircut_cost_l446_446029


namespace number_of_divisors_l446_446781

theorem number_of_divisors
  (k : ℕ)
  (p : ℕ → ℕ)
  (α : ℕ → ℕ)
  (n : ℕ)
  (h : n = ∏ i in finset.range k, (p i)^(α i)) :
  (∀ d, d ∣ n → ∃ β : ℕ → ℕ, (∀ i < k, 0 ≤ β i ∧ β i ≤ α i) ∧ d = ∏ i in finset.range k, (p i)^(β i)) →
  (finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1))) = ∏ i in finset.range k, (α i + 1)) :=
sorry

end number_of_divisors_l446_446781


namespace x_sq_y_sq_value_l446_446374

theorem x_sq_y_sq_value (x y : ℝ) 
  (h1 : x + y = 25) 
  (h2 : x^2 + y^2 = 169) 
  (h3 : x^3 * y^3 + y^3 * x^3 = 243) :
  x^2 * y^2 = 51984 := 
by 
  -- Proof to be added
  sorry

end x_sq_y_sq_value_l446_446374


namespace tamia_bell_pepper_pieces_l446_446093

theorem tamia_bell_pepper_pieces :
  let bell_peppers := 5
  let slices_per_pepper := 20
  let initial_slices := bell_peppers * slices_per_pepper
  let half_slices_cut := initial_slices / 2
  let small_pieces := half_slices_cut * 3
  let total_pieces := (initial_slices - half_slices_cut) + small_pieces
  total_pieces = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l446_446093


namespace arithmetic_geometric_sequence_l446_446315

theorem arithmetic_geometric_sequence (d : ℤ) (a_1 a_2 a_5 : ℤ)
  (h1 : d ≠ 0)
  (h2 : a_2 = a_1 + d)
  (h3 : a_5 = a_1 + 4 * d)
  (h4 : a_2 ^ 2 = a_1 * a_5) :
  a_5 = 9 * a_1 := 
sorry

end arithmetic_geometric_sequence_l446_446315


namespace simplify_and_rationalize_l446_446469

theorem simplify_and_rationalize (x : ℝ) (hx : 0 < x) :
  (√5 / √7) * (√x / √12) * (√6 / √8) = √(1260 * x) / 168 :=
by simp only [Real.sqrt_div, Real.sqrt_mul, Real.sqrt_one, Real.sqrt_bit0, Real.sqrt_bit1, 
               Real.sqrt_inv, Real.sqrt_eq_rpow, Real.rpow_nat_cast, Real.rpow_mul];
   field_simp; norm_num

end simplify_and_rationalize_l446_446469


namespace circles_tangent_length_of_chord_l446_446612

open Real

-- Definitions based on given conditions
def r₁ := 4
def r₂ := 10
def r₃ := 14
def length_of_chord : ℝ := (8 * real.sqrt 390) / 7

-- Main theorem statement
theorem circles_tangent_length_of_chord (m n p : ℕ) 
  (h₁ : ∃ m n p, length_of_chord = (m * real.sqrt n) / p)
  (h₂ : m.gcd p = 1)
  (h₃ : ∀ k : ℕ, k^2 ∣ n → k=1) 
  : m + n + p = 405 :=
by 
  use 8, 390, 7
  sorry

end circles_tangent_length_of_chord_l446_446612


namespace convex_polygon_max_interior_angles_l446_446734

theorem convex_polygon_max_interior_angles (n : ℕ) (h1 : n ≥ 3) (h2 : n < 360) :
  ∃ x, x ≤ 4 ∧ ∀ k, k > 4 → False :=
by
  sorry

end convex_polygon_max_interior_angles_l446_446734


namespace impossible_to_form_rectangle_with_2_squares_impossible_to_form_rectangle_with_3_squares_impossible_to_form_rectangle_with_4_squares_impossible_to_form_rectangle_with_5_squares_impossible_to_form_rectangle_with_6_squares_impossible_to_form_rectangle_with_7_squares_impossible_to_form_rectangle_with_8_squares_l446_446194

-- Definitions
def pairwise_distinct (squares : List ℕ) : Prop :=
  squares.nodup

def non_overlapping (squares : List ℕ) : Prop :=
  -- An appropriate definition to capture the non-overlapping condition
  -- For the sake of the problem, assume the definition is satisfied.
  true

def forms_rectangle (squares : List ℕ) : Prop :=
  -- An appropriate definition to capture the rectangle formation condition
  -- For the sake of the problem, assume the definition is satisfied.
  false

-- Theorems

theorem impossible_to_form_rectangle_with_2_squares (squares : List ℕ):
  (squares.length = 2) → pairwise_distinct squares → non_overlapping squares → ¬ forms_rectangle squares :=
by sorry

theorem impossible_to_form_rectangle_with_3_squares (squares : List ℕ):
  (squares.length = 3) → pairwise_distinct squares → non_overlapping squares → ¬ forms_rectangle squares :=
by sorry

theorem impossible_to_form_rectangle_with_4_squares (squares : List ℕ):
  (squares.length = 4) → pairwise_distinct squares → non_overlapping squares → ¬ forms_rectangle squares :=
by sorry

theorem impossible_to_form_rectangle_with_5_squares (squares : List ℕ):
  (squares.length = 5) → pairwise_distinct squares → non_overlapping squares → ¬ forms_rectangle squares :=
by sorry

theorem impossible_to_form_rectangle_with_6_squares (squares : List ℕ):
  (squares.length = 6) → pairwise_distinct squares → non_overlapping squares → ¬ forms_rectangle squares :=
by sorry

theorem impossible_to_form_rectangle_with_7_squares (squares : List ℕ):
  (squares.length = 7) → pairwise_distinct squares → non_overlapping squares → ¬ forms_rectangle squares :=
by sorry

theorem impossible_to_form_rectangle_with_8_squares (squares : List ℕ):
  (squares.length = 8) → pairwise_distinct squares → non_overlapping squares → ¬ forms_rectangle squares :=
by sorry

end impossible_to_form_rectangle_with_2_squares_impossible_to_form_rectangle_with_3_squares_impossible_to_form_rectangle_with_4_squares_impossible_to_form_rectangle_with_5_squares_impossible_to_form_rectangle_with_6_squares_impossible_to_form_rectangle_with_7_squares_impossible_to_form_rectangle_with_8_squares_l446_446194


namespace box_triangle_area_138_l446_446586

def gcd (a b : ℕ) : ℕ := if h : b = 0 then a else gcd b (a % b)
def relatively_prime (a b : ℕ) : Prop := gcd a b = 1

theorem box_triangle_area_138 (m n : ℕ) 
  (h1 : relatively_prime m n)
  (h2 : ∃ h : ℚ, h = m / n ∧ 6.4 = h / 2)
  (width length : ℕ) 
  (h3 : width = 15)
  (h4 : length = 20)
  (h5 : ∃ area : ℚ, area = 40) : 
  m + n = 138 := 
by 
  sorry

end box_triangle_area_138_l446_446586


namespace complement_of_A_in_U_l446_446700

open Set

variable (U : Set ℕ) (A : Set ℕ)

def universal_set : U = {1, 2, 3, 4, 5, 6, 7} := sorry
def subset_A : A = {2, 4, 5} := sorry

theorem complement_of_A_in_U : U \ A = {1, 3, 6, 7} :=
by {
  have hU : U = {1, 2, 3, 4, 5, 6, 7} := sorry,
  have hA : A = {2, 4, 5} := sorry,
  rw [hU, hA],
  -- Proof steps would go here, but we skip to the conclusion:
  exact sorry
}

end complement_of_A_in_U_l446_446700


namespace g_property_l446_446777

theorem g_property (g : ℤ → ℤ) (h : ∀ m n : ℤ, g (m + n) + g (mn + 1) = g m * g n + 1) :
  let val := g 2 in
  (∃ n s : ℤ, n = 1 ∧ s = 1 ∧ n * s = 1) :=
by
  let val := g 2
  use [1, 1]
  split
  · rfl
  split
  · rfl
  exact rfl

end g_property_l446_446777


namespace max_stand_proof_l446_446248

noncomputable def table_max_stand (barons counts marquises : ℕ) : ℕ :=
  if barons = 25 ∧ counts = 20 ∧ marquises = 10 then 52 else 0

theorem max_stand_proof :
  ∃ barons counts marquises, barons = 25 ∧ counts = 20 ∧ marquises = 10 →
  table_max_stand barons counts marquises = 52 :=
by
  use [25, 20, 10]
  intro h
  simp [table_max_stand, h]
  sorry

end max_stand_proof_l446_446248


namespace trajectory_of_moving_circle_l446_446750

theorem trajectory_of_moving_circle (M : ℝ × ℝ) (A : ℝ × ℝ) (l : ℝ) :
  A = (-2, 0) → (l = 2) →
  (∃ x y, M = (x, y) ∧ dist M A = dist M (l, y)) →
  (y^2 = -8 * x) := 
begin
  intros hA hl hexist,
  sorry
end

end trajectory_of_moving_circle_l446_446750


namespace incorrect_statement_D_l446_446679

open Real

def ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1

def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

def foci (a b c : ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  a = 5 ∧ b = 4 ∧ c = 3 ∧ 
  F1 = (-c, 0) ∧ F2 = (c, 0)

theorem incorrect_statement_D :
  ∀ (A B F1 F2 : ℝ × ℝ),
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    symmetric_wrt_origin A B →
    foci 5 4 3 F1 F2 →
    ¬(A.1 = 0 ∧ distance A F1 * distance A F2 = 0) := 
by
  intros A B F1 F2 ha hb ht hf
  sorry

end incorrect_statement_D_l446_446679


namespace max_standing_people_l446_446247

def num_people := 55
def num_barons := 25
def num_counts := 20
def num_marquises := 10
def standing_condition (left middle right : String) := (left = right)

theorem max_standing_people (a : Fin num_people) (title : Fin num_people → String)
  (H1 : ∃! (i j k : Fin num_people), standing_condition (title i) (title j) (title k) ∧ i ≠ j ∧ j ≠ k ∧ k ≠ i):
  (∑ (i : Fin num_people), if standing_condition (title (i - 1)) (title i) (title (i + 1)) then 1 else 0) = 52 := 
sorry

end max_standing_people_l446_446247


namespace relative_prime_integers_l446_446717

theorem relative_prime_integers (a b : ℤ) (h : a ≠ b) : ∃∞ n : ℕ, Int.gcd (a + n) (b + n) = 1 :=
by
  sorry

end relative_prime_integers_l446_446717


namespace find_derivative_f_pi_div_two_l446_446656

def f (x : ℝ) : ℝ := 5 * Real.cos x

theorem find_derivative_f_pi_div_two : deriv f (Real.pi / 2) = -5 :=
by
  sorry

end find_derivative_f_pi_div_two_l446_446656


namespace find_x_when_gx_is_21_l446_446083

theorem find_x_when_gx_is_21 (g f : ℝ → ℝ) (h1 : ∀ x, g x = 3 * (f ⁻¹' {x}).to_real) 
  (h2 : ∀ x, f x = 48 / (x + 6)) : 
  ∃ x, g x = 21 → x = 48 / 13 :=
by
  sorry

end find_x_when_gx_is_21_l446_446083


namespace isosceles_triangles_perimeter_l446_446441

theorem isosceles_triangles_perimeter (c d : ℕ) 
  (h1 : ¬(7 = c ∧ 10 = d) ∧ ¬(7 = d ∧ 10 = c))
  (h2 : 2 * c + d = 24) :
  d = 2 :=
sorry

end isosceles_triangles_perimeter_l446_446441


namespace sugar_percentage_of_second_solution_l446_446457

theorem sugar_percentage_of_second_solution :
  ∀ (W : ℝ) (P : ℝ),
  (0.10 * W * (3 / 4) + P / 100 * (1 / 4) * W = 0.18 * W) → 
  (P = 42) :=
by
  intros W P h
  sorry

end sugar_percentage_of_second_solution_l446_446457


namespace measure_of_angle_EBC_l446_446389

-- Definitions based on the conditions
variables {A B D E C : Type}
variables [linear_order A] [linear_order B] [linear_order D] [linear_order E] [linear_order C]

-- Mathematical conditions
def point_on_segment (E : A) (A B : A) : Prop := E ≥ A ∧ E ≤ B
def is_isosceles (A B C : Type) (α β : ℝ) : Prop := A = B ∧ α = β
def three_times_angle (α β : ℝ) : Prop := α = 3 * β

-- Given conditions
variable (E_on_AB : point_on_segment E A B)
variable (AED_is_isosceles : is_isosceles A E D 60)
variable (BEC_is_isosceles : is_isosceles B E C 0) -- This is inferred from the problem solution
variable (angle_DEC_three_times_ADE : three_times_angle 180 60)
variable (angle_AED : ℝ := 60)

-- Prove the measure of angle ∠EBC
theorem measure_of_angle_EBC : angle_EBC = 0 :=
by
  sorry

end measure_of_angle_EBC_l446_446389


namespace product_of_divisors_eq_1024_l446_446523

theorem product_of_divisors_eq_1024 (n : ℕ) (h1 : 0 < n) (h2 : ∏ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d = 1024) : n = 16 :=
sorry

end product_of_divisors_eq_1024_l446_446523


namespace largest_five_digit_divisible_by_3_5_7_13_l446_446468

theorem largest_five_digit_divisible_by_3_5_7_13 :
  ∃ n : ℕ, (n < 100000) ∧ (n > 9999) ∧ (∀ i j k l m : ℕ, (i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ k ≠ l ∧ k ≠ m ∧ l ≠ m) →
  (set.mem (n.digits 10) {i, j, k, l, m})) ∧ (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 13 = 0) ∧ n = 94185 :=
by
  sorry

end largest_five_digit_divisible_by_3_5_7_13_l446_446468


namespace measure_angle_BAC_l446_446498

noncomputable def incircle {ABC : Type*} [triangle ABC] (X Y : Type*) [incircle_touches_sides ABC X Y] : Prop :=
  XY_bisects_AK X Y (midpoint_arc_ABC_K AK)

theorem measure_angle_BAC (ABC : Type*) [triangle ABC]
  (X Y : Type*) [incircle_touches_sides ABC X Y]
  (K : Type*) [midpoint_arc_ABC_K AK]
  (XY_bisects_AK X Y K) :
  angle_BAC ABC = 120 :=
sorry

end measure_angle_BAC_l446_446498


namespace probability_inequality_l446_446577

-- Define the interval and the inequality
def interval : Set ℝ := { x | -5 ≤ x ∧ x ≤ 4 }
def inequality (x : ℝ) : Prop := (3 / (x + 2)) > 1

-- Prove that the probability that the inequality holds for x in the interval is 1/3
theorem probability_inequality :
  (interval : measure_theory.measure_space ℝ).volume { x | x ∈ interval ∧ inequality x } /
  (interval : measure_theory.measure_space ℝ).volume interval = 1 / 3 :=
by
  sorry

end probability_inequality_l446_446577


namespace egg_processing_l446_446001

theorem egg_processing (E : ℕ) 
  (h1 : (24 / 25) * E + 12 = (99 / 100) * E) : 
  E = 400 :=
sorry

end egg_processing_l446_446001


namespace sum_not_divisible_by_5_l446_446462

theorem sum_not_divisible_by_5 (n : ℕ) : ¬ (∃ m, (∑ k in Finset.range (n + 1), (Nat.choose (2 * n + 1) (2 * k + 1)) * 2 * 3^k) = 5 * m) :=
by
  sorry

end sum_not_divisible_by_5_l446_446462


namespace product_remainder_mod_7_l446_446869

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l446_446869


namespace probability_diagonals_intersecting_in_nonagon_l446_446952

theorem probability_diagonals_intersecting_in_nonagon :
  let num_diagonals := (finset.card (finset.univ : finset (fin (nat.choose 9 2))) - 9) in
  let num_intersections := (nat.choose 9 4) in
  let total_diagonal_pairs := (nat.choose num_diagonals 2) in
  (num_intersections.to_rat / total_diagonal_pairs.to_rat) = (14/39 : ℚ) :=
by
  sorry

end probability_diagonals_intersecting_in_nonagon_l446_446952


namespace increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446914

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end increase_in_p_does_not_imply_increase_in_equal_points_probability_l446_446914


namespace package_equivalent_fullsize_l446_446216

/-- Problem statement: 
Given the following conditions:
- There are initially 1,500 regular-sized cakes and 750 half-sized cakes.
- An additional 350 regular-sized cakes and 200 half-sized cakes are baked.
- Half-sized cakes are 50% of the size of a regular-sized cake.
- Each package contains 5 regular-sized cakes, 8 half-sized cakes, and 3 miniature cakes.
- Miniature cakes are 25% of the size of a regular-sized cake.
Show that each package is equivalent to 9.75 regular-sized cakes.
--/
theorem package_equivalent_fullsize:
  let initial_regular := 1500 in
  let initial_half := 750 in
  let additional_regular := 350 in
  let additional_half := 200 in
  let half_to_regular := 0.5 in
  let mini_to_regular := 0.25 in
  let package_regular := 5 in
  let package_half := 8 in
  let package_mini := 3 in
  (package_regular + package_half * half_to_regular + package_mini * mini_to_regular) = 9.75 :=
by
  sorry

end package_equivalent_fullsize_l446_446216


namespace product_pqr_l446_446356

/-- Mathematical problem statement -/
theorem product_pqr (p q r : ℤ) (hp: p ≠ 0) (hq: q ≠ 0) (hr: r ≠ 0)
  (h1 : p + q + r = 36)
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 540 / (p * q * r) = 1) :
  p * q * r = 864 :=
sorry

end product_pqr_l446_446356


namespace umbrella_cost_l446_446764

theorem umbrella_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) (h1 : house_umbrellas = 2) (h2 : car_umbrellas = 1) (h3 : cost_per_umbrella = 8) : 
  (house_umbrellas + car_umbrellas) * cost_per_umbrella = 24 := 
by
  sorry

end umbrella_cost_l446_446764


namespace total_students_l446_446833

-- Define the problem statement in Lean 4
theorem total_students (n : ℕ) (h1 : n < 400)
  (h2 : n % 17 = 15) (h3 : n % 19 = 10) : n = 219 :=
sorry

end total_students_l446_446833


namespace problem_statement_l446_446355

open Real

noncomputable def sum_fraction (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1).filter (0 < ·), (k : ℝ) / (k ^ 2 + 1)

theorem problem_statement (n : ℕ) (hn : 0 < n) :
  -1 < sum_fraction n - ln n ∧ sum_fraction n - ln n ≤ 1 / 2 :=
by
  sorry

end problem_statement_l446_446355


namespace find_p_l446_446562

theorem find_p (m n p : ℝ) 
  (h1 : m = (n / 2) - (2 / 5)) 
  (h2 : m + p = ((n + 4) / 2) - (2 / 5)) :
  p = 2 :=
sorry

end find_p_l446_446562


namespace speed_of_second_train_correct_l446_446923

noncomputable def length_first_train : ℝ := 140 -- in meters
noncomputable def length_second_train : ℝ := 160 -- in meters
noncomputable def time_to_cross : ℝ := 10.799136069114471 -- in seconds
noncomputable def speed_first_train : ℝ := 60 -- in km/hr
noncomputable def speed_second_train : ℝ := 40 -- in km/hr

theorem speed_of_second_train_correct :
  (length_first_train + length_second_train)/time_to_cross - (speed_first_train * (5/18)) = speed_second_train * (5/18) :=
by
  sorry

end speed_of_second_train_correct_l446_446923
