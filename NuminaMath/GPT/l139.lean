import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.CharP.Basic
import Mathlib.Algebra.Group.BigOperators
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Order.Rearrangements
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trig
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combination
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.NumberTheory.PrimitiveRoots
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Simp
import Mathlib.Topology.Instances.Nnreal

namespace factorization_c_minus_d_l139_139970

theorem factorization_c_minus_d : 
  ∃ (c d : ℤ), (∀ (x : ℤ), (4 * x^2 - 17 * x - 15 = (4 * x + c) * (x + d))) ∧ (c - d = 8) :=
by
  sorry

end factorization_c_minus_d_l139_139970


namespace decreasing_interval_l139_139319

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 5 * x^2 + 3 * x - 2

noncomputable def f' (a x : ℝ) : ℝ := 3 * a * x^2 - 10 * x + 3

theorem decreasing_interval (a : ℝ) (h : f' a 3 = 0) : 
  a = 1 → ∀ x, f' a x ≤ 0 ↔ (1 / 3 : ℝ) ≤ x ∧ x ≤ 3 :=
begin
  sorry
end

end decreasing_interval_l139_139319


namespace DP_eq_DR_l139_139438

-- Define points A, B, C on a line such that AB < BC
variables {A B C D E P Q R : Point}
variables [Line A B C] [Square ABDE]
variables [Circle_with_diameter AC]
variables [Intersection_points P Q (Circle_with_diameter AC) (Line D E)]
variable [Between P D E]
variable [Intersection_point R (Line AQ) (Line BD)]

-- Prove that DP = DR
theorem DP_eq_DR :
  DP = DR :=
sorry

end DP_eq_DR_l139_139438


namespace find_g5_l139_139070

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l139_139070


namespace solve_varphi_l139_139316

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

noncomputable def f (x : ℝ) : ℝ := real.sin (2 * x + real.pi / 6)

theorem solve_varphi (φ : ℝ) (h₀ : 0 < φ) (h₁ : φ < real.pi / 2) 
  (h2 : is_even_function (λ x, f (x - φ))) : φ = real.pi / 3 :=
sorry

end solve_varphi_l139_139316


namespace minExpression_l139_139686

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then -1 else 1

theorem minExpression (a b : ℝ) (h : a ≠ b) :
  (a + b + (a - b) * f (a - b)) / 2 = min a b :=
by
  cases lt_or_gt_of_ne h with
  | inl h1 =>
    have h2 : a - b < 0 := sub_lt_zero.mpr h1
    have hf : f (a - b) = 1 := by simp [f, h2]
    rw [hf, (sub_mul (a + b) (a - b) 1), sub_add_eq_add_sub, add_sub_cancel]
    exact (min_eq_left_of_lt h1).symm
  | inr h1 =>
    have h2 : a - b > 0 := sub_pos.mpr h1
    have hf : f (a - b) = -1 := by simp [f, h2]
    rw [hf, (sub_mul (a + b) (a - b) (-1)), sub_neg_eq_add, add_add_sub_cancel]
    exact (min_eq_right_of_lt h1).symm
  sorry

end minExpression_l139_139686


namespace shortest_distance_to_parabola_l139_139264

noncomputable def shortest_distance (p : ℝ × ℝ) (parabola : ℝ → ℝ) : ℝ :=
  let y_coord := 4
  let x_coord := y_coord^2 / 4
  let dist := (p.1 - x_coord)^2 + (p.2 - y_coord)^2
  dist.sqrt

theorem shortest_distance_to_parabola : shortest_distance (4, 8) (λ y, y^2 / 4) = 4 :=
sorry

end shortest_distance_to_parabola_l139_139264


namespace power_modulo_l139_139134

theorem power_modulo (a b c n : ℕ) (h1 : a = 17) (h2 : b = 1999) (h3 : c = 29) (h4 : n = a^b % c) : 
  n = 17 := 
by
  -- Note: Additional assumptions and intermediate calculations could be provided as needed
  sorry

end power_modulo_l139_139134


namespace eccentricity_of_ellipse_is_sqrt3_minus_1_l139_139579

noncomputable def eccentricity_of_ellipse (c : ℝ) : ℝ := 
  let a := (c * (Real.sqrt 3 + 1)) / 2
  c / a

theorem eccentricity_of_ellipse_is_sqrt3_minus_1
  (c a e : ℝ)
  (h1 : e = c / a)
  (h2 : a = (c * (Real.sqrt 3 + 1)) / 2) :
  e = Real.sqrt 3 - 1 :=
by
  rw [h2, h1]
  sorry

end eccentricity_of_ellipse_is_sqrt3_minus_1_l139_139579


namespace circle_radius_l139_139683

theorem circle_radius (x y : ℝ) : x^2 + y^2 - 2 * x + 4 * y + 2 = 0 → ∃ r : ℝ, r = √3 :=
by
  sorry

end circle_radius_l139_139683


namespace find_g5_l139_139066

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l139_139066


namespace find_x_value_l139_139986

theorem find_x_value (x : ℝ) (hx : x ≠ 0) : 
    (1/x) + (3/x) / (6/x) = 1 → x = 2 := 
by 
    intro h
    sorry

end find_x_value_l139_139986


namespace autumn_grain_purchase_exceeds_1_8_billion_tons_l139_139019

variable (x : ℝ)

theorem autumn_grain_purchase_exceeds_1_8_billion_tons 
  (h : x > 0.18) : 
  x > 1.8 := 
by 
  sorry

end autumn_grain_purchase_exceeds_1_8_billion_tons_l139_139019


namespace value_of_2Tn_add_9_l139_139657

def arithmetic_seq (a_n : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d

def geometric_mean (a1 a2 a3 : ℕ) : Prop :=
  a3 * a3 = a1 * a2

def geometric_seq (a_n : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n * q

def sum_geometric_seq (k_n : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Nat.sum (fun i => k_n i) (Finset.range n))

theorem value_of_2Tn_add_9
  (a_n : ℕ → ℕ) (d : ℕ) (a2 a3 a4 a5 : ℕ)
  (k_n : ℕ → ℕ) (T_n : ℕ → ℕ)
  (n : ℕ) :
  arithmetic_seq a_n d →
  a3 = (a_n 3) →
  geometric_mean a2 a5 a3 →
  geometric_seq (λ n, a_n (2 + 2 * n)) 3 →
  k_n = λ n, 3^(n + 1) + 1 →
  T_n n = (3^(n + 2) - 9 + 2 * n) / 2 →
  2 * T_n n + 9 = 3^(n + 2) + 2 * n :=
by
  intros
  sorry

end value_of_2Tn_add_9_l139_139657


namespace mrs_hilt_walks_240_feet_l139_139015

-- Define the distances and trips as given conditions
def distance_to_fountain : ℕ := 30
def trips_to_fountain : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain
def total_distance_walked (round_trip_distance trips_to_fountain : ℕ) : ℕ :=
  round_trip_distance * trips_to_fountain

-- State the theorem
theorem mrs_hilt_walks_240_feet :
  total_distance_walked round_trip_distance trips_to_fountain = 240 :=
by
  sorry

end mrs_hilt_walks_240_feet_l139_139015


namespace find_g5_l139_139083

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l139_139083


namespace semicircle_area_ratio_l139_139371

noncomputable def ratio_of_area (a b c : ℝ) (h : a/b = 2/3 ∧ b/c = 3/4) : ℝ :=
  let x := c / 4 in
  let S := (3 * x^2 * real.sqrt 15) / 4 in
  let r := (3 * x * real.sqrt 15) / 10 in
  let S1 := (real.pi * (r^2)) / 2 in
  S1 / S

theorem semicircle_area_ratio (a b c : ℝ) (h : a/b = 2/3 ∧ b/c = 3/4) :
  ratio_of_area a b c h = 9 * real.pi / (10 * real.sqrt 15) :=
sorry

end semicircle_area_ratio_l139_139371


namespace g_five_eq_248_l139_139052

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l139_139052


namespace cheaper_fluid_cost_is_20_l139_139800

variable (x : ℕ) -- Denote the cost per drum of the cheaper fluid as x

-- Given conditions:
variable (total_drums : ℕ) (cheaper_drums : ℕ) (expensive_cost : ℕ) (total_cost : ℕ)
variable (remaining_drums : ℕ) (total_expensive_cost : ℕ)

axiom total_drums_eq : total_drums = 7
axiom cheaper_drums_eq : cheaper_drums = 5
axiom expensive_cost_eq : expensive_cost = 30
axiom total_cost_eq : total_cost = 160
axiom remaining_drums_eq : remaining_drums = total_drums - cheaper_drums
axiom total_expensive_cost_eq : total_expensive_cost = remaining_drums * expensive_cost

-- The equation for the total cost:
axiom total_cost_eq2 : total_cost = cheaper_drums * x + total_expensive_cost

-- The goal: Prove that the cheaper fluid cost per drum is $20
theorem cheaper_fluid_cost_is_20 : x = 20 :=
by
  sorry

end cheaper_fluid_cost_is_20_l139_139800


namespace walts_running_speed_l139_139426

variable (dLw : ℝ)  -- Distance between Lionel's house and Walt's house
variable (sL : ℝ)   -- Speed of Lionel
variable (dL : ℝ)   -- Distance Lionel walked
variable (tD : ℝ)   -- Time difference when Walt starts running

-- Given conditions as definitions
def DistanceLionelWalked := dL = 15
def SpeedOfLionel := sL = 2
def DistanceBetweenHouses := dLw = 48
def TimeDifference := tD = 2
def DistanceWaltRan (v : ℝ) := (dLw - dL) = v * (dL / sL - tD)

-- Proof problem
theorem walts_running_speed (v : ℝ) (h1 : DistanceLionelWalked) (h2 : SpeedOfLionel) (h3 : DistanceBetweenHouses) (h4 : TimeDifference) : 
  DistanceWaltRan v ∧ v = 6 := 
by
  sorry

end walts_running_speed_l139_139426


namespace smallest_is_57_l139_139995

noncomputable def smallest_of_four_numbers (a b c d : ℕ) : ℕ :=
  if h1 : a + b + c = 234 ∧ a + b + d = 251 ∧ a + c + d = 284 ∧ b + c + d = 299
  then Nat.min (Nat.min a b) (Nat.min c d)
  else 0

theorem smallest_is_57 (a b c d : ℕ) (h1 : a + b + c = 234) (h2 : a + b + d = 251)
  (h3 : a + c + d = 284) (h4 : b + c + d = 299) :
  smallest_of_four_numbers a b c d = 57 :=
sorry

end smallest_is_57_l139_139995


namespace range_of_fx_in_interval_range_of_b_l139_139684

noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x, -2 * x^2 + a * x + b 

theorem range_of_fx_in_interval (a b : ℝ) (h1 : f a b 2 = -3) (h2 : a = 4) :
  ∃ y_min y_max, y_min = -19 ∧ y_max = -1 :=
by {
  sorry
}

theorem range_of_b (a b : ℝ) (h1 : f a b 2 = -3) (h3 : ∀ x, x ≥ 1 → f a b (x + 1) ≤ f a b x) :
  b ≥ -3 :=
by {
  sorry
}

end range_of_fx_in_interval_range_of_b_l139_139684


namespace problem_statement_l139_139275

variables {Line Plane : Type}

-- Defining the perpendicular relationship between a line and a plane
def perp (a : Line) (α : Plane) : Prop := sorry

-- Defining the parallel relationship between two planes
def para (α β : Plane) : Prop := sorry

-- The main statement to prove
theorem problem_statement (a : Line) (α β : Plane) (h1 : perp a α) (h2 : perp a β) : para α β := 
sorry

end problem_statement_l139_139275


namespace unit_vector_in_xz_plane_l139_139267

open Real

theorem unit_vector_in_xz_plane 
    (v : ℝ × ℝ × ℝ) 
    (h1 : v.2 = 0) 
    (h2 : (v.1^2 + v.3^2 = 1)) 
    (h3 : (v.1 + v.3)/sqrt(3) = sqrt(3)/2) 
    (h4 : v.3 / sqrt(2) = 1 / sqrt(2)) :
    v = (1/2, 0, 1) :=
by 
  sorry

end unit_vector_in_xz_plane_l139_139267


namespace cover_at_least_quarter_area_l139_139512

open Finset

noncomputable def radius (s : Sphere) : ℝ := sorry -- assume we have a way to obtain the radius of a sphere

axiom Sphere : Type
axiom disjoint (s₁ s₂: Sphere) : Prop

variables {unit_sphere : Sphere}
          {spheres : Fin 4 → Sphere}
          (disj : ∀ i j, i ≠ j → disjoint (spheres i) (spheres j))
          (sum_radii : ∑ i, radius (spheres i) = 2)

theorem cover_at_least_quarter_area :
  (∑ i, (π * (radius (spheres i)) ^ 2)) ≥ π :=
sorry

end cover_at_least_quarter_area_l139_139512


namespace find_g5_l139_139054

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l139_139054


namespace even_three_digit_numbers_sum_tens_units_eq_nine_l139_139710

theorem even_three_digit_numbers_sum_tens_units_eq_nine :
  finset.card {n | ∃ (h t u : ℕ), 100 ≤ n ∧ n < 1000 ∧ (n = 100 * h + 10 * t + u) ∧ (u % 2 = 0) ∧ ((t + u = 9))}
  = 36 :=
by
  sorry

end even_three_digit_numbers_sum_tens_units_eq_nine_l139_139710


namespace equivalent_contrapositive_l139_139847

-- Given definitions
variables {Person : Type} (possess : Person → Prop) (happy : Person → Prop)

-- The original statement: "If someone is happy, then they possess it."
def original_statement : Prop := ∀ p : Person, happy p → possess p

-- The contrapositive: "If someone does not possess it, then they are not happy."
def contrapositive_statement : Prop := ∀ p : Person, ¬ possess p → ¬ happy p

-- The theorem to prove logical equivalence
theorem equivalent_contrapositive : original_statement possess happy ↔ contrapositive_statement possess happy := 
by sorry

end equivalent_contrapositive_l139_139847


namespace sum_of_coefficients_l139_139228

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9 * x^3 - 6) + 4 * (7 * x^6 - 2 * x^3 + 8)

-- Statement to prove that the sum of the coefficients of P(x) is 62
theorem sum_of_coefficients : P 1 = 62 := sorry

end sum_of_coefficients_l139_139228


namespace find_g5_l139_139081

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l139_139081


namespace sum_of_binomial_coeff_11_l139_139998

theorem sum_of_binomial_coeff_11 :
  let a := (11.choose 0)
  let a1 := (11.choose 1)
  let a2 := (11.choose 2)
  let a3 := (11.choose 3)
  let a4 := (11.choose 4)
  let a5 := (11.choose 5)
  (a + a1 + a2 + a3 + a4 + a5) = 2^10 :=
by
  sorry

end sum_of_binomial_coeff_11_l139_139998


namespace train_actual_speed_l139_139921
-- Import necessary libraries

-- Define the given conditions and question
def departs_time := 6
def planned_speed := 100
def scheduled_arrival_time := 18
def actual_arrival_time := 16
def distance (t₁ t₂ : ℕ) (s : ℕ) : ℕ := s * (t₂ - t₁)
def actual_speed (d t₁ t₂ : ℕ) : ℕ := d / (t₂ - t₁)

-- Proof problem statement
theorem train_actual_speed:
  actual_speed (distance departs_time scheduled_arrival_time planned_speed) departs_time actual_arrival_time = 120 := by sorry

end train_actual_speed_l139_139921


namespace max_marks_is_667_l139_139189

-- Definitions based on the problem's conditions
def pass_threshold (M : ℝ) : ℝ := 0.45 * M
def student_score : ℝ := 225
def failed_by : ℝ := 75
def passing_marks := student_score + failed_by

-- The actual theorem stating that if the conditions are met, then the maximum marks M is 667
theorem max_marks_is_667 : ∃ M : ℝ, pass_threshold M = passing_marks ∧ M = 667 :=
by
  sorry -- Proof is omitted as per the instructions

end max_marks_is_667_l139_139189


namespace divide_square_into_equal_parts_l139_139607

-- Given a square with four shaded smaller squares inside
structure SquareWithShaded (n : ℕ) :=
  (squares : Fin n → Fin n → Prop) -- this models the presence of shaded squares
  (shaded : (Fin 2) → (Fin 2) → Prop)

-- To prove: we can divide the square into four equal parts with each containing one shaded square
theorem divide_square_into_equal_parts :
  ∀ (sq : SquareWithShaded 4),
  ∃ (parts : Fin 2 → Fin 2 → Prop),
  (∀ i j, parts i j ↔ 
    ((i = 0 ∧ j = 0) ∨ (i = 1 ∧ j = 0) ∨ 
    (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 1)) ∧
    (∃! k l, sq.shaded k l ∧ parts i j)) :=
sorry

end divide_square_into_equal_parts_l139_139607


namespace triangle_side_ratio_l139_139656

theorem triangle_side_ratio (a b c : ℝ) (h1 : a + b ≤ 2 * c) (h2 : b + c ≤ 3 * a) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  2 / 3 < c / a ∧ c / a < 2 :=
by
  sorry

end triangle_side_ratio_l139_139656


namespace jared_sent_in_november_l139_139759

noncomputable def text_messages (n : ℕ) : ℕ :=
  match n with
  | 0 => 1  -- November
  | 1 => 2  -- December
  | 2 => 4  -- January
  | 3 => 8  -- February
  | 4 => 16 -- March
  | _ => 0

theorem jared_sent_in_november : text_messages 0 = 1 :=
sorry

end jared_sent_in_november_l139_139759


namespace number_of_three_digit_numbers_l139_139118

-- Question and conditions used in the definitions
def digits : List ℕ := [0, 1, 2, 3, 4]
def valid_hundreds_digits : List ℕ := digits.erase 0

-- The Lean theorem statement to prove that the number of distinct three-digit numbers is 48
theorem number_of_three_digit_numbers : 
  ∃ count : ℕ, count = 4 * 4 * 3 ∧ count = 48 :=
by
  use 48
  split
  · calc 
      4 * 4 * 3 = 16 * 3 : by rfl
      ... = 48 : by rfl
  · rfl

end number_of_three_digit_numbers_l139_139118


namespace sin_17pi_over_6_l139_139637

theorem sin_17pi_over_6 : Real.sin (17 * Real.pi / 6) = 1 / 2 :=
by
  sorry

end sin_17pi_over_6_l139_139637


namespace perimeter_of_triangle_l139_139784

noncomputable def ellipse (x y a : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / 3) = 1

theorem perimeter_of_triangle {a : ℝ} 
  (h_a : a > real.sqrt 3)
  (h_point : ellipse 1 (3/2) a) : 
  ∃ p : ℝ, p = 6 :=
sorry

end perimeter_of_triangle_l139_139784


namespace minimal_unit_circles_to_cover_triangle_l139_139132

noncomputable
def triangle_side_a : ℝ := 2
noncomputable
def triangle_side_b : ℝ := 3
noncomputable
def triangle_side_c : ℝ := 4
noncomputable
def unit_circle_radius : ℝ := 1

theorem minimal_unit_circles_to_cover_triangle : 
  ∃ (n : ℕ), 
  n = 3 
  ∧ triangle_coverable_with_unit_circles 
      triangle_side_a 
      triangle_side_b 
      triangle_side_c 
      unit_circle_radius 
      n := 
by
  sorry

end minimal_unit_circles_to_cover_triangle_l139_139132


namespace solution_exists_l139_139958

noncomputable def verifySolution (x y z : ℝ) : Prop := 
  x^2 - y = (z - 1)^2 ∧
  y^2 - z = (x - 1)^2 ∧
  z^2 - x = (y- 1)^2 

theorem solution_exists (x y z : ℝ) (h : verifySolution x y z) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x, y, z) = (-2.93122, 2.21124, 0.71998) ∨ 
  (x, y, z) = (2.21124, 0.71998, -2.93122) ∨ 
  (x, y, z) = (0.71998, -2.93122, 2.21124) :=
sorry

end solution_exists_l139_139958


namespace smallest_positive_integer_n_l139_139877

def contains_digit_9 (n : ℕ) : Prop := 
  ∃ m : ℕ, (10^m) ∣ n ∧ (n / 10^m) % 10 = 9

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (∀ k : ℕ, k > 0 ∧ k < n → 
  (∃ a b : ℕ, k = 2^a * 5^b * 3) ∧ contains_digit_9 k ∧ (k % 3 = 0))
  → n = 90 :=
sorry

end smallest_positive_integer_n_l139_139877


namespace sum_of_digits_of_n_l139_139861

theorem sum_of_digits_of_n (n : ℕ) (hn : (n + 1)! + (n + 2)! = n! * 675) : Nat.digits 10 n = [2, 4] →
  n = 24 :=
begin
  sorry
end

end sum_of_digits_of_n_l139_139861


namespace find_vector_b_l139_139700

def vec_a : ℝ × ℝ := (Real.sqrt 3, 1)

def is_unit_vector (b : ℝ × ℝ) : Prop := Real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 1

def dot_product_eq (b : ℝ × ℝ) : Prop := (vec_a.1 * b.1 + vec_a.2 * b.2) = Real.sqrt 3

def not_parallel_to_x (b : ℝ × ℝ) : Prop := b.2 ≠ 0

theorem find_vector_b (x y : ℝ) :
  is_unit_vector (x, y) ∧ dot_product_eq (x, y) ∧ not_parallel_to_x (x, y) →
  (x = 1/2 ∧ y = Real.sqrt 3 / 2) :=
sorry

end find_vector_b_l139_139700


namespace DE_le_p_over_8_l139_139573

variable (ABC : Triangle) (D E : Point)
variable (p : ℝ) -- perimeter of the triangle ABC
variable [ LieOn ABC (Side AB) D]
variable [ LieOn ABC (Side AC) E]
variable [Parallel DE BC : DE. parallel BC]
variable [TouchInCircle DE] -- DE touches the incircle

theorem DE_le_p_over_8 :
  ∀ DE p, touches_incircle(DE, ABC) ∧ perimeter(ABC) = p → DE.length ≤ p / 8 :=
by
  sorry

end DE_le_p_over_8_l139_139573


namespace find_q_l139_139638

theorem find_q (q : ℚ) : (8^4 = (4^3 / 2) * 2^(15 * q)) → q = 7 / 15 := by
  sorry

end find_q_l139_139638


namespace horizontal_asymptote_l139_139515

theorem horizontal_asymptote (f : ℝ → ℝ) (h : ∀ x, f x = (8 * x^3 - 7 * x + 6) / (4 * x^3 + 3 * x^2 - 2)) :
  (tendsto f at_top (𝓝 2)) :=
by
  sorry

end horizontal_asymptote_l139_139515


namespace axis_of_symmetry_parabola_l139_139046

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), ∃ y : ℝ, y = (x - 5)^2 → x = 5 := 
by 
  sorry

end axis_of_symmetry_parabola_l139_139046


namespace parallelogram_area_eq_triangle_l139_139212

theorem parallelogram_area_eq_triangle (A B C E F G : Type) (D : ℝ) 
  (h_midpoint : ∃ (E : Type), ∀ (B C : Type), midpoint B C E) 
  (h_angle : ∃ (F : Type), ∀ (E : Type), angle DEF D) 
  (h_parallel1 : ∃ (G : Type), line_parallel_to CG EF C) 
  (h_parallel2 : ∃ (G : Type), line_parallel_to AG BC A)
  (h_intersect : ∃ (G : Type), intersection_point CG AG G)
  (h_parallelogram : parallelogram E C F G) :
  area ECFG = area ABC :=
sorry

end parallelogram_area_eq_triangle_l139_139212


namespace check_correct_l139_139550

-- Given the conditions
variable (x y : ℕ) (H1 : 10 ≤ x ∧ x ≤ 81) (H2 : y = x + 18)

-- Rewrite the problem and correct answer for verification in Lean
theorem check_correct (Hx : 10 ≤ x ∧ x ≤ 81) (Hy : y = x + 18) : 
  y = 2 * x ↔ x = 18 := 
by
  sorry

end check_correct_l139_139550


namespace trig_solutions_l139_139883

theorem trig_solutions (t : ℝ) :
  (sin(2 * t))^3 + (cos(2 * t))^3 + (1 / 2) * sin(4 * t) = 1 →
  ∃ k n : ℤ, t = k * π ∨ t = (π / 4) * (4 * n + 1) :=
by
  sorry

end trig_solutions_l139_139883


namespace find_g5_l139_139059

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l139_139059


namespace range_of_f_l139_139598

noncomputable def f (x : ℝ) : ℝ :=
if x = -5 then 0 else 3 * (x - 2)

theorem range_of_f :
  (Set.range (λ x : ℝ, if x = -5 then 0 else 3 * (x - 2))) = {y : ℝ | y ≠ -21} :=
by
  sorry

end range_of_f_l139_139598


namespace largest_possible_N_l139_139002

theorem largest_possible_N (N : ℕ) :
  let divisors := Nat.divisors N
  in (1 ∈ divisors) ∧ (N ∈ divisors) ∧ (divisors.length ≥ 3) ∧ (divisors[divisors.length - 3] = 21 * divisors[1]) → N = 441 := 
by
  sorry

end largest_possible_N_l139_139002


namespace find_least_x_l139_139516

theorem find_least_x:
  ∃ (x : ℕ), (least_positive_x x) ∧ (3 * x + 41 ≡ 0 [MOD 53]) :=
begin
  -- Definitions
  def least_positive_x (x : ℕ) : Prop := 
    ∃ (k : ℤ), x > 0 ∧ 3 * x + 41 = 53 * k,
  
  -- Solution steps and proof (skipped with sorry)
  sorry
end

end find_least_x_l139_139516


namespace largest_angle_of_consecutive_integers_in_hexagon_l139_139472

theorem largest_angle_of_consecutive_integers_in_hexagon : 
  ∀ (a : ℕ), 
    (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) = 720 → 
    a + 3 = 122.5 :=
by sorry

end largest_angle_of_consecutive_integers_in_hexagon_l139_139472


namespace fine_per_day_of_absence_l139_139174

theorem fine_per_day_of_absence :
  ∃ x: ℝ, ∀ (total_days work_wage total_received_days absent_days: ℝ),
  total_days = 30 →
  work_wage = 10 →
  total_received_days = 216 →
  absent_days = 7 →
  (total_days - absent_days) * work_wage - (absent_days * x) = total_received_days :=
sorry

end fine_per_day_of_absence_l139_139174


namespace min_selection_to_ensure_multiple_10_l139_139644

theorem min_selection_to_ensure_multiple_10 (S : set ℕ) (hS : S = {x : ℕ | x ≥ 1 ∧ x ≤ 2020}) :
  ∃ T ⊆ S, ∃ m ≥ 1837, ∃ x y ∈ T, x ≠ y ∧ (x = 10 * y ∨ y = 10 * x) :=
by
  sorry

end min_selection_to_ensure_multiple_10_l139_139644


namespace sphere_volume_in_cone_l139_139915

theorem sphere_volume_in_cone :
  ∀ (d : ℝ),
  d = 24 → 
  let r := d / 4 in 
  let V := (4 / 3) * Real.pi * r^3 in
  V = 288 * Real.pi :=
by
  intros d h₁ r h₂ V h₃
  sorry

end sphere_volume_in_cone_l139_139915


namespace arithmetic_sequence_solution_l139_139850

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  (a 4 = 9) ∧ (a 3 + a 7 = 22)

-- Given conditions
theorem arithmetic_sequence_solution (a : ℕ → ℕ) (b : ℕ → ℝ) (S T : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a n = 2 * n + 1) →
  (S n = n^2 + 2 * n) →
  (b n = 1 / (a n * a (n + 1))) →
  (T n = ∑ i in Finset.range n, b i) →
  T n = n / (3 * (2 * n + 3)) :=
by
  intros h_a h_an h_Sn h_bn h_Tn
  sorry

end arithmetic_sequence_solution_l139_139850


namespace fraction_of_y_l139_139717

theorem fraction_of_y (w x y : ℝ) (h1 : wx = y) 
  (h2 : (w + x) / 2 = 0.5) : 
  (2 / w + 2 / x = 2 / y) := 
by
  sorry

end fraction_of_y_l139_139717


namespace bono_jelly_beans_l139_139576

variable (t A B C : ℤ)

theorem bono_jelly_beans (h₁ : A + B = 6 * t + 3) 
                         (h₂ : A + C = 4 * t + 5) 
                         (h₃ : B + C = 6 * t) : 
                         B = 4 * t - 1 := by
  sorry

end bono_jelly_beans_l139_139576


namespace total_people_is_120_l139_139044

def num_children : ℕ := 80

def num_adults (num_children : ℕ) : ℕ := num_children / 2

def total_people (num_children num_adults : ℕ) : ℕ := num_children + num_adults

theorem total_people_is_120 : total_people num_children (num_adults num_children) = 120 := by
  sorry

end total_people_is_120_l139_139044


namespace largest_possible_N_l139_139001

theorem largest_possible_N (N : ℕ) :
  let divisors := Nat.divisors N
  in (1 ∈ divisors) ∧ (N ∈ divisors) ∧ (divisors.length ≥ 3) ∧ (divisors[divisors.length - 3] = 21 * divisors[1]) → N = 441 := 
by
  sorry

end largest_possible_N_l139_139001


namespace range_of_a_l139_139671

theorem range_of_a (p q : Set ℝ) (a : ℝ) (h1 : ∀ x, 2 * x^2 - 3 * x + 1 ≤ 0 → x ∈ p) 
                             (h2 : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a ≤ 0 → x ∈ q)
                             (h3 : ∀ x, p x → q x ∧ ∃ x, ¬p x ∧ q x) : 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l139_139671


namespace marbles_given_by_Joan_l139_139031

def initial_yellow_marbles : ℝ := 86.0
def final_yellow_marbles : ℝ := 111.0

theorem marbles_given_by_Joan :
  final_yellow_marbles - initial_yellow_marbles = 25 := by
  sorry

end marbles_given_by_Joan_l139_139031


namespace sum_gcf_lcm_add_10_eq_38_l139_139136

theorem sum_gcf_lcm_add_10_eq_38 : 
  let gcf := Nat.gcd 8 12 in
  let lcm := Nat.lcm 8 12 in
  gcf + lcm + 10 = 38 :=
by
  let gcf := Nat.gcd 8 12
  let lcm := Nat.lcm 8 12
  have h1 : gcf = 4 := by simp [Nat.gcd]
  have h2 : lcm = 24 := by simp [Nat.lcm]
  have h3 := h1.symm ▸ h2.symm ▸ rfl
  show gcf + lcm + 10 = 38 from h3.symm ▸ rfl
  sorry

end sum_gcf_lcm_add_10_eq_38_l139_139136


namespace solve_log_equation_l139_139035

open Real

theorem solve_log_equation (x : ℝ) (h : (4 / (sqrt (log 3 (81 * x)) + sqrt (log 3 x)) + sqrt (log 3 x) = 3)) : x = 243 :=
sorry

end solve_log_equation_l139_139035


namespace problem_solution_l139_139270

theorem problem_solution (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (h : x - y = x / y) : 
  (1 / x - 1 / y = -1 / y^2) := 
by sorry

end problem_solution_l139_139270


namespace seashells_total_l139_139790

theorem seashells_total (x y z T : ℕ) (m k : ℝ) 
  (h₁ : x = 2) 
  (h₂ : y = 5) 
  (h₃ : z = 9) 
  (h₄ : x + y = T) 
  (h₅ : m * x + k * y = z) : 
  T = 7 :=
by
  -- This is where the proof would go.
  sorry

end seashells_total_l139_139790


namespace pipeline_problem_l139_139158

theorem pipeline_problem 
  (length_pipeline : ℕ) 
  (extra_meters : ℕ) 
  (days_saved : ℕ) 
  (x : ℕ)
  (h1 : length_pipeline = 4000) 
  (h2 : extra_meters = 10) 
  (h3 : days_saved = 20) 
  (h4 : (4000:ℕ) / (x - extra_meters) - (4000:ℕ) / x = days_saved) :
  x = 4000 / ((4000 / (x - extra_meters) + 20)) + extra_meters :=
by
  -- The proof goes here
  sorry

end pipeline_problem_l139_139158


namespace ratio_of_areas_of_concentric_circles_l139_139112

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ)
  (h : (30 / 360) * C1 = (24 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 16 / 25 := by
  sorry

end ratio_of_areas_of_concentric_circles_l139_139112


namespace balanced_path_divides_square_l139_139179

def balanced_path (n : ℕ) : Prop :=
  ∃ (path : (ℕ × ℕ) → (ℕ × ℕ)),
  (path (0, 0) = (0, 0)) ∧
  (path (2 * n, 2 * n) = (n, n)) ∧
  (∀ (i : ℕ), 0 ≤ i ∧ i < 2 * n → 
    let (x_i, y_i) := path (i, 2 * n - i) in 
    (x_i = i + 1 ∧ y_i = y_i) ∨ 
    (x_i = x_i ∧ y_i = i + 1)) ∧
  ((finset.univ.sum (λ i, (path (i, 2 * n - i)).fst)) =
   (finset.univ.sum (λ i, (path (i, 2 * n - i)).snd)))

theorem balanced_path_divides_square (n : ℕ) (h : balanced_path n) :
  let square_area := n * n in
  let path_area := ∑ i in finset.range (2 * n), 
                   let (x_i, y_i) := h.1 (i, 2 * n - i) in 
                   (y_i - y_i.min (i - 1, 2 * n - (i - 1))) in
  path_area = square_area / 2 :=
sorry

end balanced_path_divides_square_l139_139179


namespace find_side_e_l139_139393

theorem find_side_e
  (D E : Real)
  (d f : ℝ)
  (h1 : E = 4 * D)
  (h2 : d = 36)
  (h3 : f = 60) :
  ∃ e : ℝ, e = 36 * sin (5 * D) / Real.sqrt (sin D ^ 2) :=
by
  sorry

end find_side_e_l139_139393


namespace find_least_x_l139_139517

theorem find_least_x:
  ∃ (x : ℕ), (least_positive_x x) ∧ (3 * x + 41 ≡ 0 [MOD 53]) :=
begin
  -- Definitions
  def least_positive_x (x : ℕ) : Prop := 
    ∃ (k : ℤ), x > 0 ∧ 3 * x + 41 = 53 * k,
  
  -- Solution steps and proof (skipped with sorry)
  sorry
end

end find_least_x_l139_139517


namespace base10_to_base8_conversion_l139_139601

theorem base10_to_base8_conversion (n : ℕ) (h₁ : n = 512) : nat.to_digits 8 n = [1, 0, 0, 0] :=
by {
  rw h₁,
  simp,
  sorry
}

end base10_to_base8_conversion_l139_139601


namespace find_first_spill_l139_139812

def bottle_capacity : ℕ := 20
def refill_count : ℕ := 3
def days : ℕ := 7
def total_water_drunk : ℕ := 407
def second_spill : ℕ := 8

theorem find_first_spill :
  let total_without_spill := bottle_capacity * refill_count * days
  let total_spilled := total_without_spill - total_water_drunk
  let first_spill := total_spilled - second_spill
  first_spill = 5 :=
by
  -- Proof goes here.
  sorry

end find_first_spill_l139_139812


namespace graph_paper_fold_l139_139181

theorem graph_paper_fold (m n : ℝ) :
  (∃ fold_line : ℝ → ℝ, 
    ((∀ (x : ℝ), fold_line x = (5 / 3) * x - 1) ∧
     (fold_line 2.5 = 1.5) ∧ 
     ((8, 4), (m, n)) ∈ set_of_points_folded_by (fold_line))) →
  m + n = 9.75 :=
sorry

end graph_paper_fold_l139_139181


namespace butterfly_cocoon_time_l139_139398

theorem butterfly_cocoon_time :
  ∀ (L C : ℕ), L = 3 * C ∧ L + C = 120 → C = 30 := by
  intros L C h
  cases' h with h1 h2
  have h3 : 3 * C + C = 120 := by rw [h1, add_comm] at h2
  have h4 : 4 * C = 120 := by rw add_mul 3 1 C at h3
  rw mul_comm at h4
  exact Nat.eq_of_mul_eq_mul_left (ne_of_gt (show 0 < 4 from by norm_num)) h4

end butterfly_cocoon_time_l139_139398


namespace tyre_punctures_deflation_time_l139_139885

theorem tyre_punctures_deflation_time :
  (1 / (1 / 9 + 1 / 6)) = 3.6 :=
by
  sorry

end tyre_punctures_deflation_time_l139_139885


namespace projectile_height_time_l139_139453

theorem projectile_height_time :
  ∃ t, t ≥ 0 ∧ -16 * t^2 + 80 * t = 72 ↔ t = 1 := 
by sorry

end projectile_height_time_l139_139453


namespace max_profit_is_4sqrt6_add_21_l139_139159

noncomputable def profit (x : ℝ) : ℝ :=
  let y1 : ℝ := -2 * (3 - x)^2 + 14 * (3 - x)
  let y2 : ℝ := - (1 / 3) * x^3 + 2 * x^2 + 5 * x
  let F : ℝ := y1 + y2 - 3
  F

theorem max_profit_is_4sqrt6_add_21 : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ profit x = 4 * Real.sqrt 6 + 21 :=
sorry

end max_profit_is_4sqrt6_add_21_l139_139159


namespace radius_of_circumscribed_circle_l139_139186

-- Definitions based on conditions
def sector (radius : ℝ) (central_angle : ℝ) : Prop :=
  central_angle = 120 ∧ radius = 10

-- Statement of the theorem we want to prove
theorem radius_of_circumscribed_circle (r R : ℝ) (h : sector r 120) : R = 20 := 
by
  sorry

end radius_of_circumscribed_circle_l139_139186


namespace mikey_cereal_l139_139431

/--
Mikey likes his honey cluster of oats cereal. In his first bowl of cereal, 
each spoonful contains 4 clusters of oats and he gets 25 spoonfuls. 
However, with each subsequent bowl, the number of oat clusters per spoonful increases by 1 
and the number of spoonfuls in each bowl decreases by 2. 
If each box of cereal contains 500 clusters of oats, 
prove that Mikey can make 4 bowlfuls of cereal from each box given these changing conditions.
-/
theorem mikey_cereal :
  let clusters_of_oats (n : ℕ) := (3 + n) * (27 - 2 * n)
  ∑ n in range 4, clusters_of_oats n ≤ 500 ∧ (∑ n in range 5, clusters_of_oats n > 500) := 
sorry

end mikey_cereal_l139_139431


namespace vector_addition_subtraction_l139_139971

open Matrix

def v1 : Matrix (Fin 2) (Fin 1) ℤ := ![![5], ![-6]]
def v2 : Matrix (Fin 2) (Fin 1) ℤ := ![[-2], ![13]]
def v3 : Matrix (Fin 2) (Fin 1) ℤ := ![![1], ![-2]]
def v_result : Matrix (Fin 2) (Fin 1) ℤ := ![[0], ![13]]

theorem vector_addition_subtraction : 
  v1 + v2 - (3 • v3) = v_result := by
  sorry

end vector_addition_subtraction_l139_139971


namespace tan_B_l139_139390

-- Definitions based on given conditions:
structure Triangle :=
  (AC AB : ℝ)
  (C_is_right_angle : true)

noncomputable def AC : ℝ := 4
noncomputable def AB : ℝ := Real.sqrt 41
noncomputable def BC : ℝ := Real.sqrt (AB^2 - AC^2)

-- The statement to prove:
theorem tan_B (ABC : Triangle) (h : ABC.C_is_right_angle ∧ ABC.AC = 4 ∧ ABC.AB = Real.sqrt 41) :
  Real.tan (Real.arctan (ABC.AC / BC)) = 4 / 5 :=
by {
  have h_BC : ABC.BC = 5, by {
    sorry
  },
  sorry
}

end tan_B_l139_139390


namespace max_students_exam_l139_139198

/--
An exam contains 4 multiple-choice questions, each with three options (A, B, C). Several students take the exam.
For any group of 3 students, there is at least one question where their answers are all different.
Each student answers all questions. Prove that the maximum number of students who can take the exam is 9.
-/
theorem max_students_exam (n : ℕ) (A B C : ℕ → ℕ → ℕ) (q : ℕ) :
  (∀ (s1 s2 s3 : ℕ), ∃ (q : ℕ), (1 ≤ q ∧ q ≤ 4) ∧ (A s1 q ≠ A s2 q ∧ A s1 q ≠ A s3 q ∧ A s2 q ≠ A s3 q)) →
  q = 4 ∧ (∀ s, 1 ≤ s → s ≤ n) → n ≤ 9 :=
by
  sorry

end max_students_exam_l139_139198


namespace butterfly_cocoon_l139_139397

theorem butterfly_cocoon (c l : ℕ) (h1 : l + c = 120) (h2 : l = 3 * c) : c = 30 :=
by
  sorry

end butterfly_cocoon_l139_139397


namespace modulus_of_z_equals_two_l139_139423

namespace ComplexProblem

open Complex

-- Definition and conditions of the problem
def satisfies_condition (z : ℂ) : Prop :=
  (z + I) * (1 + I) = 1 - I

-- Statement that needs to be proven
theorem modulus_of_z_equals_two (z : ℂ) (h : satisfies_condition z) : abs z = 2 :=
sorry

end ComplexProblem

end modulus_of_z_equals_two_l139_139423


namespace largest_possible_N_l139_139000

theorem largest_possible_N (N : ℕ) :
  let divisors := Nat.divisors N
  in (1 ∈ divisors) ∧ (N ∈ divisors) ∧ (divisors.length ≥ 3) ∧ (divisors[divisors.length - 3] = 21 * divisors[1]) → N = 441 := 
by
  sorry

end largest_possible_N_l139_139000


namespace mod_45_remainder_of_14_to_100_l139_139262

theorem mod_45_remainder_of_14_to_100 (gcd_14_45 : Nat.gcd 14 45 = 1)
    (phi_45 : Nat.totient 45 = 24)
    (euler_theorem : (14 ^ 24) % 45 = 1) :
    (14 ^ 100) % 45 = 31 := 
by 
  sorry

end mod_45_remainder_of_14_to_100_l139_139262


namespace hexagon_diagonal_lengths_l139_139833

theorem hexagon_diagonal_lengths (A B C D E F : Type) 
  (dist : A → A → ℝ)
  (hexagon : convex_hexagon A B C D E F)
  (h_AB : dist A B < 1)
  (h_BC : dist B C < 1)
  (h_CD : dist C D < 1)
  (h_DE : dist D E < 1)
  (h_EF : dist E F < 1)
  (h_FA : dist F A < 1) :
  ¬ (dist A D ≥ 2 ∧ dist B E ≥ 2 ∧ dist C F ≥ 2) := 
sorry

end hexagon_diagonal_lengths_l139_139833


namespace last_digit_of_3_pow_2004_l139_139018

theorem last_digit_of_3_pow_2004 : (3 ^ 2004) % 10 = 1 := by
  sorry

end last_digit_of_3_pow_2004_l139_139018


namespace max_ships_1x4_on_10x10_board_l139_139155

theorem max_ships_1x4_on_10x10_board :
  ∀ (board : fin 10 × fin 10 → bool) (non_overlap : ∀ (i j : fin 10) (dir : bool), board (i, j) = false → ( ∀ (k : fin 4), board (i + k.val if dir else i, j if dir else j + k.val) = false))
    (non_adjacent : ∀ (i j : fin 10) (dir : bool), board (i, j) = false → ( ∀ (k : fin 5), board (i + k.val - 1 if dir else i, j if dir else j + k.val - 1) = true → false)) ,
  ∃ n ≤ 24, ∀ s : fin n, ∃ i j : fin 10, dir : bool,
    (∀ (k : fin 4), board (i + k.val if dir else i, j if dir else j + k.val) = true) :=
  sorry

end max_ships_1x4_on_10x10_board_l139_139155


namespace PL_parallel_CD_l139_139394

open EuclideanGeometry

noncomputable def midpoint (A B : Point): Point := sorry

theorem PL_parallel_CD (A B C D E G P L : Point)
    (hD : D = midpoint A B)
    (hE : E = midpoint A C)
    (hG : is_intersection_point B E C D G)
    (hP : ∃ k, is_circumcircle A B E k ∧ is_circumcircle A C D k ∧ P ∈ k ∧ P ≠ A)
    (hL : ∃ c, is_circumcircle A C D c ∧ L ∈ c ∧ A ∈ line_through P L):
    parallel (line_through P L) (line_through C D) :=
begin
  sorry
end

end PL_parallel_CD_l139_139394


namespace largest_angle_of_consecutive_integers_hexagon_l139_139465

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l139_139465


namespace magnitude_of_combined_vector_l139_139716

variables {a b c : EuclideanSpace ℝ (Fin 3)}
variables (h₁ : ‖a‖ = 1) (h₂ : ‖b‖ = 1) (h₃ : ‖c‖ = 1)
variables (h₄ : ⟪a, b⟫ = 0) (h₅ : ⟪a, c⟫ = 0) (h₆ : ⟪b, c⟫ = 0)

theorem magnitude_of_combined_vector :
  ‖2 • a + 2 • b - 3 • c‖ = Real.sqrt 17 :=
by 
  sorry

end magnitude_of_combined_vector_l139_139716


namespace proof_problem_l139_139042

noncomputable def g (x : ℝ) : ℝ :=
  if x = 1 then 4
  else if x = 2 then 6
  else if x = 3 then 9
  else if x = 4 then 10
  else if x = 5 then 12
  else sorry

theorem proof_problem :
  (exists f : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)) →
  g(g(2)) + g(g⁻¹ 12) + g⁻¹(g⁻¹ 10) = 25 :=
by
  -- Define the inverse function g⁻¹
  let g_inv : ℝ → ℝ :=
    λ y, if y = 4 then 1
         else if y = 6 then 2
         else if y = 9 then 3
         else if y = 10 then 4
         else if y = 12 then 5
         else sorry,

  -- Use the inversibility properties in the statement
  assume h : (exists f : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)),
  cases h with f hf,
  have hg_inv : g⁻¹ (y: ℝ) = g_inv y :=
    funext (λ y, (hf.left y).symm ▸ (hf.right (g_inv y))),
  
  -- Compute the required expression
  calc
    g(g(2)) + g(g⁻¹ 12) + g⁻¹(g⁻¹ 10)
        = g 6 + g 5 + g⁻¹(g 4) : by 
          -- Compute step-by-step based on provided conditions and inverted values
          rw [g, if_pos rfl, hb, g, if_pos rfl, hg_inv]
        = 12 + 12 + 1 : sorry

end proof_problem_l139_139042


namespace functions_equivalent_l139_139578

def f (x : ℝ) := x
def g (x : ℝ) := x^2 / x
def h (x : ℝ) := x - 1
def i (x : ℝ) := (x - 1)^2
def j (x : ℝ) := x
def k (x : ℝ) := (x^3)^(1/3)
def l (x : ℝ) := abs x
def m (x : ℝ) := (sqrt x)^2

theorem functions_equivalent :
  (∀ x : ℝ, f x = k x) ∧ (∀ x : ℝ, f x = k x) :=
by
  sorry

end functions_equivalent_l139_139578


namespace product_eq_one_l139_139536

-- Definitions of the given conditions
def condition_1 (a b : ℝ) : Prop :=
  (∑ n in Finset.range 2016, (a^3 + n) / (b^3 + n)) = 2016

def b_positive (b : ℝ) : Prop :=
  b > 0

-- The main theorem stating the proof problem
theorem product_eq_one (a b : ℝ) (h1 : condition_1 a b) (h2 : b_positive b) : 
  (∏ n in Finset.range 2016, (a^3 + n) / (b^3 + n)) = 1 :=
by
  sorry

end product_eq_one_l139_139536


namespace rotated_point_coordinates_l139_139379

def rotate_point (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ)

theorem rotated_point_coordinates :
  rotate_point (135 * Real.pi / 180) (-1, 1) = (0, -Real.sqrt 2) :=
by
  sorry

end rotated_point_coordinates_l139_139379


namespace prod_X_ge_prod_Y_l139_139292

-- Define the conditions
variables {m n : ℕ}
variables (a : ℕ → ℕ → ℝ)
variables (a_non_neg : ∀ i j, 0 ≤ a i j)
variables (row_non_increasing : ∀ i, ∀ j k, (j < k) → (a i j ≥ a i k))
variables (col_non_increasing : ∀ j, ∀ i k, (i < k) → (a i j ≥ a k j))

-- Definitions based on the conditions
def X (i j : ℕ) : ℝ :=
  ∑ k in Finset.range (i + 1), a k j + ∑ l in Finset.range (j + 1), a i l

def Y (i j : ℕ) : ℝ :=
  ∑ k in Finset.range (m - i), a (m - k) j + ∑ l in Finset.range (n - j), a i (j + l)

-- Main theorem statement
theorem prod_X_ge_prod_Y : 
  (∏ i in Finset.range m, ∏ j in Finset.range n, X a i j) ≥ (∏ i in Finset.range m, ∏ j in Finset.range n, Y a i j) :=
sorry

end prod_X_ge_prod_Y_l139_139292


namespace bipartite_partition_l139_139027

-- Defining a graph structure, bipartiteness, and the theorem.
structure Graph (V : Type) :=
  (E : V → V → Prop)

def bipartite {V : Type} (G : Graph V) : Prop :=
  ∃ (S T : set V), (∀ v ∈ S, ∀ w ∈ S, ¬ G.E v w) ∧ (∀ v ∈ T, ∀ w ∈ T, ¬ G.E v w) ∧
    (∀ v ∈ S, ∀ w ∈ T, G.E v w ∨ G.E w v)

theorem bipartite_partition {V : Type} (G : Graph V) (h : bipartite G) :
  ∃ S T : set V, (∀ v ∈ S, ∀ w ∈ S, ¬ G.E v w) ∧ (∀ v ∈ T, ∀ w ∈ T, ¬ G.E v w) ∧
    (∀ v ∈ S, ∀ w ∈ T, G.E v w ∨ G.E w v) :=
by
  sorry

end bipartite_partition_l139_139027


namespace root_interval_l139_139279

noncomputable def f (x : ℝ) := 3^x + 3*x - 8

theorem root_interval :
  (∀ x, f x = 3^x + 3*x - 8) →
  f 1 < 0 →
  f 2 > 0 →
  f 1.5 > 0 →
  f 1.25 < 0 →
  ∃ a b, (a = 1.25 ∧ b = 1.5 ∧ ∀ x, (a < x ∧ x < b) → f x = 0 := sorry

end root_interval_l139_139279


namespace butterfly_cocoon_l139_139396

theorem butterfly_cocoon (c l : ℕ) (h1 : l + c = 120) (h2 : l = 3 * c) : c = 30 :=
by
  sorry

end butterfly_cocoon_l139_139396


namespace jogger_ahead_engine_l139_139166

-- Define the given constants for speed and length
def jogger_speed : ℝ := 2.5 -- in m/s
def train_speed : ℝ := 12.5 -- in m/s
def train_length : ℝ := 120 -- in meters
def passing_time : ℝ := 40 -- in seconds

-- Define the target distance
def jogger_ahead : ℝ := 280 -- in meters

-- Lean 4 statement to prove the jogger is 280 meters ahead of the train's engine
theorem jogger_ahead_engine :
  passing_time * (train_speed - jogger_speed) - train_length = jogger_ahead :=
by
  sorry

end jogger_ahead_engine_l139_139166


namespace g_five_eq_248_l139_139049

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l139_139049


namespace find_n_l139_139260

theorem find_n :
  (n : ℕ) →
  (\arctan (1/2) + \arctan (1/3) + \arctan (1/7) + \arctan (1/n) = Real.pi / 2) →
  n = 4 :=
by
  sorry

end find_n_l139_139260


namespace find_plaintext_from_ciphertext_l139_139867

theorem find_plaintext_from_ciphertext : 
  ∃ x : ℕ, ∀ a : ℝ, (a^3 - 2 = 6) → (1022 = a^x - 2) → x = 10 :=
by
  use 10
  intros a ha hc
  -- Proof omitted
  sorry

end find_plaintext_from_ciphertext_l139_139867


namespace f_odd_function_l139_139672

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (a b : ℝ) : f (a + b) = f a + f b

theorem f_odd_function : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  sorry

end f_odd_function_l139_139672


namespace change_calculation_l139_139537

def cost_of_apple : ℝ := 0.75
def amount_paid : ℝ := 5.00

theorem change_calculation : (amount_paid - cost_of_apple) = 4.25 := by
  sorry

end change_calculation_l139_139537


namespace poly_int_if_int_coeffs_l139_139029

theorem poly_int_if_int_coeffs (a b c : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, f(x) = k) ↔ (2 * a ∈ ℤ ∧ (a + b) ∈ ℤ ∧ c ∈ ℤ) :=
by
  -- Defining the quadratic polynomial
  let f := λ x : ℤ, a * x^2 + b * x + c

  -- Providing the theorem statement
  sorry

end poly_int_if_int_coeffs_l139_139029


namespace basketball_team_starters_l139_139164

open Nat

def totalPlayers : Nat := 18
def triplets : List String := ["Bob", "Bill", "Brenda"]
def numStarters : Nat := 7
def numTripletsInStarters : Nat := 2
def remainingPlayers : Nat := totalPlayers - triplets.length
def remainingStartersNeeded : Nat := numStarters - numTripletsInStarters

theorem basketball_team_starters :
  choose(triplets.length, numTripletsInStarters) * choose(remainingPlayers, remainingStartersNeeded) = 9009 := by
  -- Sorry is used to skip the proof
  sorry

end basketball_team_starters_l139_139164


namespace evaluate_trigonometric_expression_l139_139648

noncomputable def trigonometric_identity (x : ℝ) : Prop :=
  (cos (π / 4 + x) = 3 / 5) ∧ 
  (17 / 12 * π < x ∧ x < 7 / 4 * π)

theorem evaluate_trigonometric_expression (x : ℝ) 
  (h : trigonometric_identity x) : 
  (sin (2 * x) + 2 * sin x ^ 2) / (1 - tan x) = -28 / 75 :=
by
  sorry

end evaluate_trigonometric_expression_l139_139648


namespace closest_integer_to_cubert_of_sum_l139_139121

theorem closest_integer_to_cubert_of_sum : 
  let a := 5
  let b := 7
  let a_cubed := a^3
  let b_cubed := b^3
  let sum_cubed := a_cubed + b_cubed
  7^3 < sum_cubed ∧ sum_cubed < 8^3 →
  Int.abs (Int.floor (Real.cbrt (sum_cubed)) - 8) < 
  Int.abs (Int.floor (Real.cbrt (sum_cubed)) - 7) :=
by
  sorry

end closest_integer_to_cubert_of_sum_l139_139121


namespace equal_intercepts_l139_139463

theorem equal_intercepts (a : ℝ) (h : ∃ (x y : ℝ), (x = (2 + a) / a ∧ y = 2 + a ∧ x = y)) :
  a = -2 ∨ a = 1 :=
by sorry

end equal_intercepts_l139_139463


namespace complement_P_correct_l139_139768

def is_solution (x : ℝ) : Prop := |x + 3| + |x + 6| = 3

def P : Set ℝ := {x | is_solution x}

def C_R (P : Set ℝ) : Set ℝ := {x | x ∉ P}

theorem complement_P_correct : C_R P = {x | x < -6 ∨ x > -3} :=
by
  sorry

end complement_P_correct_l139_139768


namespace green_pill_cost_is_20_67_l139_139195

variable (green_pill_cost pink_pill_cost : ℝ)

def condition1 : Prop := pink_pill_cost = green_pill_cost - 2
def condition2 : Prop := 10 * (2 * green_pill_cost + pink_pill_cost) = 600

theorem green_pill_cost_is_20_67 (h₀ : condition1) (h₁ : condition2) : green_pill_cost = 20.67 :=
by sorry

end green_pill_cost_is_20_67_l139_139195


namespace sum_of_squares_and_product_l139_139494

open Real

theorem sum_of_squares_and_product (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
    x + y = Real.sqrt 565 := by
  sorry

end sum_of_squares_and_product_l139_139494


namespace fish_to_rice_l139_139380

-- Defining the variables
variable (f l r : ℝ)

-- Conditions from the problem
def condition1 := 5 * f = 3 * l
def condition2 := l = 7 * r

-- Theorem to prove
theorem fish_to_rice (h1 : condition1 f l r) (h2 : condition2 l r) : f = (21 / 5) * r := by
  sorry

end fish_to_rice_l139_139380


namespace last_digit_base4_of_389_l139_139605

theorem last_digit_base4_of_389 : (389 % 4 = 1) :=
by sorry

end last_digit_base4_of_389_l139_139605


namespace sqrt_meaningful_l139_139522

theorem sqrt_meaningful (x : ℝ) : (∃ (y : ℝ), y = sqrt (2 * x - 1)) ↔ x ≥ 1 / 2 :=
by
  sorry

end sqrt_meaningful_l139_139522


namespace leading_coefficient_poly_l139_139254

def poly : Polynomial ℤ := -2 * (Polynomial.monomial 5 1 - Polynomial.monomial 4 1 + Polynomial.monomial 3 2) +
                            6 * (Polynomial.monomial 5 1 + Polynomial.monomial 2 1 - Polynomial.monomial 0 1) -
                            5 * (Polynomial.monomial 5 3 + Polynomial.monomial 3 1 + Polynomial.monomial 0 4)

theorem leading_coefficient_poly : Polynomial.leadingCoeff poly = -11 := by
  sorry

end leading_coefficient_poly_l139_139254


namespace find_g5_l139_139067

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l139_139067


namespace most_difficult_to_react_with_H2_l139_139996

def substance : Type := Sort 0

inductive Substances : substance
| Fluorine : Substances
| Nitrogen : Substances
| Chlorine : Substances
| Oxygen : Substances

def bond_energy : Substances → ℝ 
| Substances.Fluorine := -- specify fluorine bond energy
| Substances.Nitrogen := -- specify nitrogen bond energy (highest triple bond energy)
| Substances.Chlorine := -- specify chlorine bond energy 
| Substances.Oxygen := -- specify oxygen bond energy

def nitrogen_triple_bond (s: Substances) : Prop :=
  s = Substances.Nitrogen → bond_energy Substances.Nitrogen > bond_energy Substances.Fluorine ∧ 
  bond_energy Substances.Nitrogen > bond_energy Substances.Chlorine ∧ 
  bond_energy Substances.Nitrogen > bond_energy Substances.Oxygen

theorem most_difficult_to_react_with_H2 : 
  ∀ (s: Substances), nitrogen_triple_bond s → s = Substances.Nitrogen :=
by
  intros
  sorry

end most_difficult_to_react_with_H2_l139_139996


namespace problem_part_one_problem_part_two_l139_139321

def f (x m : ℝ) := cos x + m * (x + (Real.pi / 2)) * sin x

theorem problem_part_one (m : ℝ) (hm : m ≤ 1) :
  ∃! x : ℝ, x ∈ Ioo (-Real.pi) 0 ∧ f x m = 0 :=
sorry

theorem problem_part_two (m : ℝ) :
  (∃ t > 0, ∀ x ∈ Ioo (-Real.pi / 2 - t) (-Real.pi / 2), |f x m| < -2 * x - Real.pi) →
  m = -1 :=
sorry

end problem_part_one_problem_part_two_l139_139321


namespace count_valid_labelings_l139_139965

-- Definitions for the conditions
structure CubeLabeling :=
  (edges : Fin 12 → Bool)
  (faces : Fin 6 → List (Fin 12))
  (sum_labels : Fin 6 → Nat)
  (adjacent_faces : Fin 6 → Fin 6 → Bool)

-- Condition that the sum of the labels on each face is either 1 or 3.
def valid_face_sum (L : CubeLabeling) : Prop :=
  ∀ f : Fin 6, L.sum_labels f = 1 ∨ L.sum_labels f = 3

-- Condition that no two adjacent faces can have more than one edge labeled 1 in common.
def valid_adjacent_faces (L : CubeLabeling) : Prop :=
  ∀ (f1 f2 : Fin 6), L.adjacent_faces f1 f2 → 
    (∑ edge in (L.faces f1).inter (L.faces f2), L.edges edge) ≤ 1

-- Main problem: Proving the number of valid labelings is 12.
theorem count_valid_labelings : (∃ L : CubeLabeling, valid_face_sum L ∧ valid_adjacent_faces L) ↔ (12) := 
sorry

end count_valid_labelings_l139_139965


namespace inverse_proportion_points_l139_139662

theorem inverse_proportion_points (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : x2 > 0)
  (h3 : y1 = -8 / x1)
  (h4 : y2 = -8 / x2) :
  y2 < 0 ∧ 0 < y1 :=
by
  sorry

end inverse_proportion_points_l139_139662


namespace largest_angle_of_consecutive_integers_in_hexagon_l139_139474

theorem largest_angle_of_consecutive_integers_in_hexagon : 
  ∀ (a : ℕ), 
    (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) = 720 → 
    a + 3 = 122.5 :=
by sorry

end largest_angle_of_consecutive_integers_in_hexagon_l139_139474


namespace triangle_CD_length_l139_139762

noncomputable def triangle_AB_values : ℝ := 4024
noncomputable def triangle_AC_values : ℝ := 4024
noncomputable def triangle_BC_values : ℝ := 2012
noncomputable def CD_value : ℝ := 504.5

theorem triangle_CD_length 
  (AB AC : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (h1 : AB = triangle_AB_values)
  (h2 : AC = triangle_AC_values)
  (h3 : BC = triangle_BC_values) :
  CD = CD_value := by
  sorry

end triangle_CD_length_l139_139762


namespace hyperbola_equation_l139_139326

noncomputable def hyperbola_foci_equation (x y : ℝ) : Prop :=
  let F1 := (-real.sqrt 5, 0)
  let F2 := (real.sqrt 5, 0)
  let PF1 := sqrt ((x + real.sqrt 5)^2 + y^2)
  let PF2 := sqrt ((x - real.sqrt 5)^2 + y^2)
  (x^2 / 4) − y^2 = 1

theorem hyperbola_equation :
  (x y : ℝ) (F1 : (ℝ × ℝ)) (F2 : (ℝ × ℝ))
  (PF1 PF2 : ℝ)
  (F1 = (-real.sqrt 5, 0)) (F2 = (real.sqrt 5, 0))
  (PF1 := sqrt ((x + real.sqrt 5)^2 + y^2))
  (PF2 := sqrt ((x - real.sqrt 5)^2 + y^2))
  (PF1 * PF2 = 2)
  (inner (x + real.sqrt 5, y) (x - real.sqrt 5, y) = 0)
  : (x^2 / 4) - y^2 = 1 := by
  sorry

end hyperbola_equation_l139_139326


namespace combined_ages_100_in_2024_l139_139509

noncomputable def Ulysses_age : ℕ := 12
noncomputable def Kim_age : ℕ := 14
noncomputable def Mei_age : ℕ := 15
noncomputable def Tanika_age : ℕ := 15
noncomputable def Year_current : ℕ := 2023

theorem combined_ages_100_in_2024 :
  let total_age := Ulysses_age + Kim_age + Mei_age + Tanika_age,
      annual_increase := 4 in
  total_age = 56 ∧
  (total_age + annual_increase * 11 = 100) ∧
  (Year_current + 11 = 2024) := by
  sorry

end combined_ages_100_in_2024_l139_139509


namespace no_transform_possible_l139_139729

open Matrix

def initial_table : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 2, 3],
    ![4, 5, 6],
    ![7, 8, 9]
  ]

def target_table : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 4, 7],
    ![2, 5, 8],
    ![3, 6, 9]
  ]

theorem no_transform_possible : ¬∃ (f : Matrix (Fin 3) (Fin 3) ℕ → Matrix (Fin 3) (Fin 3) ℕ), 
  (∀ A, f (swap_rows A) = swap_rows (f A)) → 
  (∀ A, f (swap_columns A) = swap_columns (f A)) → 
  f initial_table = target_table := 
  sorry

end no_transform_possible_l139_139729


namespace largest_possible_value_of_N_l139_139008

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l139_139008


namespace hyperbola_equation_dot_product_zero_l139_139282

noncomputable def hyperbola_center_origin (e : ℝ) (M : ℝ × ℝ) : Prop :=
  e = real.sqrt 2 ∧ M = (4, -real.sqrt 10) ∧ ∃ λ : ℝ, λ ≠ 0 ∧ ∀ x y, x^2 - y^2 = λ

noncomputable def point_on_hyperbola (N : ℝ × ℝ) (λ : ℝ) : Prop :=
  N = (3, m) ∧ (3^2 - m^2 = λ)

theorem hyperbola_equation : hyperbola_center_origin (real.sqrt 2) (4, -real.sqrt 10) → ∃ λ, λ = 6 :=
by
  sorry
  
theorem dot_product_zero (m : ℝ) (λ : ℝ) (F1 F2 : ℝ × ℝ) (N : ℝ × ℝ) :
  point_on_hyperbola (3, m) λ →
  F1 = (-2 * real.sqrt 3, 0) →
  F2 = (2 * real.sqrt 3, 0) →
  ∃ (N : ℝ × ℝ), N = (3, m) ∧ ((-2 * real.sqrt 3 - 3, -m) • (2 * real.sqrt 3 - 3, -m)) = 0 :=
by
  sorry

end hyperbola_equation_dot_product_zero_l139_139282


namespace shift_cosine_correct_l139_139868

noncomputable def shift_cosine (A : ℝ) : Prop :=
  ∀ x : ℝ, cos (2 * (x - A) + (π / 4)) = cos (2 * x)

theorem shift_cosine_correct : shift_cosine (π / 8) :=
by
  sorry

end shift_cosine_correct_l139_139868


namespace angle_BDC_is_correct_l139_139735

noncomputable def angle_BDC (A B C D : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  (angle_CAD : ℝ) (angle_DBC : ℝ) (angle_BAD : ℝ) (angle_ABC : ℝ) : ℝ :=
let x := 10, y := 20, z := 40, w := 50 in if (angle_CAD = x ∧ angle_DBC = y ∧ angle_BAD = z ∧ angle_ABC = w) then 40 else 0

theorem angle_BDC_is_correct (A B C D : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] :
  forall (angle_CAD angle_DBC angle_BAD angle_ABC : ℝ),
    angle_CAD = 10 → angle_DBC = 20 → angle_BAD = 40 → angle_ABC = 50 → angle_BDC A B C D angle_CAD angle_DBC angle_BAD angle_ABC = 40 :=
by
  intros
  rw [angle_BDC]
  split_ifs with h
  · exact rfl
  · contradiction
  sorry

end angle_BDC_is_correct_l139_139735


namespace mod_squares_eq_one_l139_139649

theorem mod_squares_eq_one
  (n : ℕ)
  (h : n = 5)
  (a : ℤ)
  (ha : ∃ b : ℕ, ↑b = a ∧ b * b ≡ 1 [MOD 5]) :
  (a * a) % n = 1 :=
by
  sorry

end mod_squares_eq_one_l139_139649


namespace hostel_food_duration_l139_139907

noncomputable def food_last_days (total_food_units daily_consumption_new: ℝ) : ℝ :=
  total_food_units / daily_consumption_new

theorem hostel_food_duration:
  let x : ℝ := 1 -- assuming x is a positive real number
  let men_initial := 100
  let women_initial := 100
  let children_initial := 50
  let total_days := 40
  let consumption_man := 3 * x
  let consumption_woman := 2 * x
  let consumption_child := 1 * x
  let food_sufficient_for := 250
  let total_food_units := 550 * x * 40
  let men_leave := 30
  let women_leave := 20
  let children_leave := 10
  let men_new := men_initial - men_leave
  let women_new := women_initial - women_leave
  let children_new := children_initial - children_leave
  let daily_consumption_new := 210 * x + 160 * x + 40 * x 
  (food_last_days total_food_units daily_consumption_new) = 22000 / 410 := 
by
  sorry

end hostel_food_duration_l139_139907


namespace combined_average_score_girls_l139_139935

open BigOperators

variable (A a B b C c : ℕ) -- number of boys and girls at each school
variable (x : ℕ) -- common value for number of boys and girls

axiom Adams_HS : 74 * (A : ℤ) + 81 * (a : ℤ) = 77 * (A + a)
axiom Baker_HS : 83 * (B : ℤ) + 92 * (b : ℤ) = 86 * (B + b)
axiom Carter_HS : 78 * (C : ℤ) + 85 * (c : ℤ) = 80 * (C + c)

theorem combined_average_score_girls :
  (A = a ∧ B = b ∧ C = c) →
  (A = B ∧ B = C) →
  (81 * (A : ℤ) + 92 * (B : ℤ) + 85 * (C : ℤ)) / (A + B + C) = 86 := 
by
  intro h1 h2
  sorry

end combined_average_score_girls_l139_139935


namespace mady_balls_2010th_step_l139_139427

theorem mady_balls_2010th_step :
  let base_5_digits (n : Nat) : List Nat := (Nat.digits 5 n)
  (base_5_digits 2010).sum = 6 := by
  sorry

end mady_balls_2010th_step_l139_139427


namespace eat_five_pastries_in_46_875_minutes_l139_139013

def missQuickRate := 1 / 15 -- Miss Quick's eating rate in pastries per minute
def missSlowRate := 1 / 25 -- Miss Slow's eating rate in pastries per minute
def totalPastries := 5 -- The total number of pastries they need to eat
def combinedRate := missQuickRate + missSlowRate -- Combined eating rate
def totalTime := totalPastries / combinedRate -- Total time for eating the pastries

theorem eat_five_pastries_in_46_875_minutes : totalTime = 46.875 := by
  -- Here is where the proof would go. We skip it by using 'sorry'.
  sorry

end eat_five_pastries_in_46_875_minutes_l139_139013


namespace series_sum_eq_neg_one_l139_139947

   noncomputable def sum_series : ℝ :=
     ∑' k : ℕ, if k = 0 then 0 else (12 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

   theorem series_sum_eq_neg_one : sum_series = -1 :=
   sorry
   
end series_sum_eq_neg_one_l139_139947


namespace total_sugar_weight_l139_139113

theorem total_sugar_weight (x y : ℝ) (h1 : y - x = 8) (h2 : x - 1 = 0.6 * (y + 1)) : x + y = 40 := by
  sorry

end total_sugar_weight_l139_139113


namespace largest_possible_value_of_N_l139_139007

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l139_139007


namespace sqrt_meaningful_real_l139_139358

theorem sqrt_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_meaningful_real_l139_139358


namespace no_real_roots_x_squared_minus_x_plus_nine_l139_139093

theorem no_real_roots_x_squared_minus_x_plus_nine :
  ∀ x : ℝ, ¬ (x^2 - x + 9 = 0) :=
by 
  intro x 
  sorry

end no_real_roots_x_squared_minus_x_plus_nine_l139_139093


namespace candy_bar_cost_l139_139588

noncomputable def cost_per_candy_bar (cost_soft_drink : ℕ) (num_soft_drinks : ℕ) (total_spent : ℕ) (num_candy_bars : ℕ) : ℕ :=
  (total_spent - num_soft_drinks * cost_soft_drink) / num_candy_bars

theorem candy_bar_cost :
  ∀ (cost_soft_drink num_soft_drinks total_spent num_candy_bars cost_candy_bar : ℕ),
  cost_soft_drink = 4 →
  num_soft_drinks = 2 →
  total_spent = 28 →
  num_candy_bars = 5 →
  cost_stat = 4 →
  cost_per_candy_bar cost_soft_drink num_soft_drinks total_spent num_candy_bars = cost_stat
by {
  intros,
  sorry
}

end candy_bar_cost_l139_139588


namespace inverse_function_of_exp_x_plus_1_l139_139086

theorem inverse_function_of_exp_x_plus_1 :
  ∀ (x : ℝ), (y = e^(x + 1)) ↔ (x = -1 + ln(y)) (x > 0) := by 
sorry

end inverse_function_of_exp_x_plus_1_l139_139086


namespace find_special_N_l139_139974

def digit (d : ℕ) : Prop := d < 10

-- We define the condition for the number N
noncomputable def N (a : list ℕ) : ℕ := 
  a.foldr (λ x acc, x + acc * 10) 0

-- We ensure all elements of the list are digits
def is_digit_list (a : list ℕ) : Prop :=
  ∀ x ∈ a, digit x

-- At most one of the digits in the list is zero
def at_most_one_zero (a : list ℕ) : Prop :=
  (a.filter (λ x, x = 0)).length ≤ 1

-- Reversing the number
noncomputable def reverse_N (a : list ℕ) : ℕ :=
  N a.reverse

-- Problem conditions
def problem_conditions (a : list ℕ) : Prop :=
  is_digit_list a ∧ at_most_one_zero a

-- Prove that N is of the expected form given the conditions
theorem find_special_N (a : list ℕ) (h : problem_conditions a) :
  9 * N a = reverse_N a → 
  N a = 0 ∨ ∃ k : ℕ, N a = list.repeat 9 k ++ [1, 0] := sorry

end find_special_N_l139_139974


namespace marie_keeps_lollipops_l139_139429

def total_lollipops (raspberry mint blueberry coconut : ℕ) : ℕ :=
  raspberry + mint + blueberry + coconut

def lollipops_per_friend (total friends : ℕ) : ℕ :=
  total / friends

def lollipops_kept (total friends : ℕ) : ℕ :=
  total % friends

theorem marie_keeps_lollipops :
  lollipops_kept (total_lollipops 75 132 9 315) 13 = 11 :=
by
  sorry

end marie_keeps_lollipops_l139_139429


namespace sugar_water_inequality_one_sugar_water_inequality_two_l139_139799

variable (a b m : ℝ)

-- Condition constraints
variable (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m)

-- Sugar Water Experiment One Inequality
theorem sugar_water_inequality_one : a / b > a / (b + m) := 
by
  sorry

-- Sugar Water Experiment Two Inequality
theorem sugar_water_inequality_two : a / b < (a + m) / b := 
by
  sorry

end sugar_water_inequality_one_sugar_water_inequality_two_l139_139799


namespace smallest_integer_satisfying_mod_conditions_l139_139962

theorem smallest_integer_satisfying_mod_conditions :
  ∃ n : ℕ, n > 0 ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 4) ∧ 
  (n % 7 = 6) ∧ 
  (n % 11 = 10) ∧ 
  n = 1154 := 
sorry

end smallest_integer_satisfying_mod_conditions_l139_139962


namespace find_c_l139_139248

-- Define c and the floor function
def c : ℝ := 13.1

theorem find_c (h : c + ⌊c⌋ = 25.6) : c = 13.1 :=
sorry

end find_c_l139_139248


namespace problem_statement_l139_139943

noncomputable def calc : ℝ :=
  (-4)^2013 * (-0.25)^2014

theorem problem_statement : calc = -0.25 := 
by 
  sorry

end problem_statement_l139_139943


namespace irrational_roots_of_quadratic_l139_139243

theorem irrational_roots_of_quadratic {
  k : ℝ
} (h1 : ∃ k, k^2 = 16 / 3) :
  let Δ := 25 * k^2 - 12 * k^2
  in (Δ > 0 ∧ ∀ n : ℤ, n * n ≠ Δ) :=
by {
  sorry
}

end irrational_roots_of_quadratic_l139_139243


namespace percentage_of_men_tenured_approx_eq_l139_139586

variable (total_profs : ℕ)
variable (women_percentage : ℝ)
variable (tenured_percentage : ℝ)
variable (women_or_tenured_percentage : ℝ)

variable [fact (total_profs = 100)]
variable [fact (women_percentage = 0.69)]
variable [fact (tenured_percentage = 0.70)]
variable [fact (women_or_tenured_percentage = 0.90)]

theorem percentage_of_men_tenured_approx_eq :
  let women_profs := women_percentage * total_profs in
  let tenured_profs := tenured_percentage * total_profs in
  let women_or_tenured_profs := women_or_tenured_percentage * total_profs in
  let men_profs := total_profs - women_profs in
  let neither_women_nor_tenured_profs := total_profs - women_or_tenured_profs in
  let tenured_men_profs := men_profs - neither_women_nor_tenured_profs in
  (tenured_men_profs / men_profs) * 100 ≈ 67.74 := 
by sorry

end percentage_of_men_tenured_approx_eq_l139_139586


namespace find_prime_p_l139_139972

theorem find_prime_p (p : ℕ) (hp : Nat.Prime p) (hp_plus_10 : Nat.Prime (p + 10)) (hp_plus_14 : Nat.Prime (p + 14)) : p = 3 := 
sorry

end find_prime_p_l139_139972


namespace max_A_value_l139_139284

-- Define the conditions
variable (n : ℕ) (x : ℕ → Bool)
variable (h_odd : n % 2 = 1)

-- Define the counting function A
def count_triplets : ℕ :=
  ∑ i in Finset.range n, ∑ j in Finset.range i, ∑ k in Finset.range j,
  if x k ≠ x j ∧ (x i, x j, x k) = (false, true, false) ∨ (x i, x j, x k) = (true, false, true) then 1 else 0

-- Maximum value of A for odd n
theorem max_A_value : count_triplets n x = n * (n^2 - 1) / 24 :=
sorry

end max_A_value_l139_139284


namespace inequality_solution_l139_139037

open Set

theorem inequality_solution (x : ℝ) : 
  (x ∈ (Ioo (-∞) 2 ∪ Ioo 2 6)) ↔ (x ≠ 2 ∧ (x - 6) / (x - 2)^2 < 0) :=
sorry

end inequality_solution_l139_139037


namespace TeamWinningPercentage_l139_139232

theorem TeamWinningPercentage 
  (games_played: ℕ) 
  (first_100_games_won_percent: ℝ)
  (remaining_games_won_percent: ℝ)
  (total_games_played: ℕ)
  (total_games_played_eq: games_played = 175)
  (first_100_games_won_percent_eq: first_100_games_won_percent = 85 / 100)
  (remaining_games_won_percent_eq: remaining_games_won_percent = 50 / 100):
  (100 * ((85 / 100 * 100) + (50 / 100 * (175 - 100))) / 175) ≈ 69.71 := 
sorry

end TeamWinningPercentage_l139_139232


namespace complex_polynomial_roots_mod_one_l139_139409

theorem complex_polynomial_roots_mod_one 
  (a b c : ℂ)
  (h : ∀ r : ℂ, r ∈ (Finset.roots (by exact (X^3 + C a * X^2 + C b * X + C c))) → |r| = 1) :
  ∀ r' : ℂ, r' ∈ (Finset.roots (by exact (X^3 + C |a| * X^2 + C |b| * X + C |c|))) → |r'| = 1 :=
sorry

end complex_polynomial_roots_mod_one_l139_139409


namespace colonization_combinations_l139_139713

/-- Number of different combinations of planets that can be colonized using 15 units. --/
theorem colonization_combinations (units : ℕ) (e_earth : ℕ) (m_mars : ℕ) : 
  units = 15 ∧ e_earth = 5 ∧ m_mars = 8 → 
  ∑ (a : ℕ) in {0, 1, 2, 3, 4, 5}, ∑ (b : ℕ) in {0, 1, 2, 3, 4, 5, 6, 7, 8}, 
  if 2 * a + b = 15 then (Nat.choose e_earth a * Nat.choose m_mars b) else 0 = 96 :=
begin
  sorry
end

end colonization_combinations_l139_139713


namespace sum_of_star_tips_l139_139449

/-- Given ten points that are evenly spaced on a circle and connected to form a 10-pointed star,
prove that the sum of the angle measurements of the ten tips of the star is 720 degrees. -/
theorem sum_of_star_tips (n : ℕ) (h : n = 10) :
  (10 * 72 = 720) :=
by
  sorry

end sum_of_star_tips_l139_139449


namespace parallel_KP_MC_l139_139206

-- Define the geometric entities and their properties
variables 
  {k1 k2 : Type} [Circle k1] [Circle k2]
  {A B O K L M P C : Type} [Point A] [Point B] [Point O] [Point K]
  [Point L] [Point M] [Point P] [Point C]
  {p : Line} [Collinear p K O] [Collinear p L M]
  [Between K L O] [Intersection k1 k2 A B] [Center k2 O]
  [Projection P L (Line.mk A B)] [Midpoint C A B]

-- Define the parallelism proof goal
theorem parallel_KP_MC :
  Parallel (Line.mk K P) (Median.mk M (Triangle.mk A B M)) :=
sorry

end parallel_KP_MC_l139_139206


namespace grazing_time_for_36_cows_l139_139860

-- Defining the problem conditions and the question in Lean 4
theorem grazing_time_for_36_cows :
  ∀ (g r b : ℕ), 
    (24 * 6 * b = g + 6 * r) →
    (21 * 8 * b = g + 8 * r) →
    36 * 3 * b = g + 3 * r :=
by
  intros
  sorry

end grazing_time_for_36_cows_l139_139860


namespace no_non_degenerate_triangle_l139_139866

theorem no_non_degenerate_triangle 
  (a b c : ℕ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ c) 
  (h3 : a ≠ c) 
  (h4 : Nat.gcd a (Nat.gcd b c) = 1) 
  (h5 : a ∣ (b - c) * (b - c)) 
  (h6 : b ∣ (a - c) * (a - c)) 
  (h7 : c ∣ (a - b) * (a - b)) : 
  ¬ (a < b + c ∧ b < a + c ∧ c < a + b) := 
sorry

end no_non_degenerate_triangle_l139_139866


namespace perimeter_difference_is_zero_l139_139111

-- Definitions for the first figure
def first_figure_perimeter : ℕ :=
  let horizontal := 6 + 6  -- Bottom and top sides
  let vertical := 1 + 1 + 2 + 2  -- Vertical sides and extended middle segment
  horizontal + vertical  -- Perimeter calculation

-- Definitions for the second figure
def second_figure_perimeter : ℕ :=
  let horizontal := 7 + 7  -- Top and bottom sides
  let vertical := 2 + 2  -- Sides
  horizontal + vertical  -- Perimeter calculation

-- Positive difference between the perimeters of the two figures
def positive_difference : ℕ :=
  abs (first_figure_perimeter - second_figure_perimeter)

-- Theorem statement: the positive difference is 0
theorem perimeter_difference_is_zero : positive_difference = 0 := by
  sorry

end perimeter_difference_is_zero_l139_139111


namespace mass_percentage_of_h_in_chromic_acid_l139_139255

noncomputable def molar_mass_h2cro4 : ℝ :=
  (2 * 1.01) + (1 * 51.996) + (4 * 16)

noncomputable def total_mass_hydrogen : ℝ :=
  2 * 1.01

noncomputable def mass_percentage_h : ℝ :=
  (total_mass_hydrogen / molar_mass_h2cro4) * 100

theorem mass_percentage_of_h_in_chromic_acid :
  mass_percentage_h ≈ 1.712 :=
sorry

end mass_percentage_of_h_in_chromic_acid_l139_139255


namespace mode_and_median_of_scores_l139_139569

noncomputable def scores : List ℕ := [91, 95, 89, 93, 88, 94, 95]

def mode (l : List ℕ) : ℕ :=
  List.mode ℕ l

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.getD (l.length / 2) 0

theorem mode_and_median_of_scores :
  mode scores = 95 ∧ median scores = 93 := by
  sorry

end mode_and_median_of_scores_l139_139569


namespace product_xyz_equals_1080_l139_139415

noncomputable def xyz_product (x y z : ℝ) : ℝ :=
  if (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x * (y + z) = 198) ∧ (y * (z + x) = 216) ∧ (z * (x + y) = 234)
  then x * y * z
  else 0 

theorem product_xyz_equals_1080 {x y z : ℝ} :
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x * (y + z) = 198) ∧ (y * (z + x) = 216) ∧ (z * (x + y) = 234) →
  xyz_product x y z = 1080 :=
by
  intros h
  -- Proof skipped
  sorry

end product_xyz_equals_1080_l139_139415


namespace part1_part2_l139_139666

variable (a b : ℝ) (x : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) : (a^2 / b) + (b^2 / a) ≥ a + b :=
sorry

theorem part2 (h3 : 0 < x) (h4 : x < 1) : 
(∀ y : ℝ, y = ((1 - x)^2 / x) + (x^2 / (1 - x)) → y ≥ 1) ∧ ((1 - x) = x → y = 1) :=
sorry

end part1_part2_l139_139666


namespace expanded_figure_perimeter_l139_139817

def side_length : ℕ := 2
def bottom_row_squares : ℕ := 3
def total_squares : ℕ := 4

def perimeter (side_length : ℕ) (bottom_row_squares : ℕ) (total_squares: ℕ) : ℕ :=
  2 * side_length * (bottom_row_squares + 1)

theorem expanded_figure_perimeter : perimeter side_length bottom_row_squares total_squares = 20 :=
by
  sorry

end expanded_figure_perimeter_l139_139817


namespace largest_angle_in_consecutive_integer_hexagon_l139_139476

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l139_139476


namespace equivalent_sums_l139_139236

def sum_sin_sec_series (f : ℝ → ℝ → ℝ) (g : ℝ → ℝ → ℝ) (h : ℝ → ℝ → ℝ) : ℝ :=
  ∑ x in finset.Ico 3 (45 + 1), 2 * real.sin x * real.sin 2 * (1 + (real.sec (x - 2) * real.sec (x + 2)))

def sum_phi_psi_series (Φ Ψ : ℝ → ℝ) (θ : fin 4 → ℝ) : ℝ :=
  ∑ n in finset.range 4, ((-1) ^ (n + 1)) * (Φ (θ n) / Ψ (θ n))

theorem equivalent_sums :
  ∃ (θ : fin 4 → ℝ), 
    θ 0 = 1 ∧ θ 1 = 2 ∧ θ 2 = 45 ∧ θ 3 = 47 ∧
    sum_sin_sec_series (λ x y, real.cos (x - y) - real.cos (x + y))
                      (λ x y, real.sin (x - y))
                      (λ x y, real.sin (x + y)) 
    = sum_phi_psi_series (λ θ, real.sin θ * real.sin 2) 
                         (λ θ, real.cos θ * real.cos 2) 
                         θ
    ∧ (θ 0 + θ 1 + θ 2 + θ 3 = 95) :=
begin
  sorry
end

end equivalent_sums_l139_139236


namespace assignment_correct_l139_139639

noncomputable def task_assignment_count : ℕ :=
  (nat.choose 3 1 * nat.choose 4 2 * nat.perm 3 3) + (nat.choose 3 2 * nat.perm 3 3)

theorem assignment_correct:
  (let A := "A"; B := "B"; C := "C"; D := "D"; E := "E";
       tasks := ["Translation", "TourGuide", "Etiquette", "Driver"];
       count := task_assignment_count in
   count = (nat.choose 3 1 * nat.choose 4 2 * nat.perm 3 3) + (nat.choose 3 2 * nat.perm 3 3)) :=
by
  -- Proof would go here
  sorry

end assignment_correct_l139_139639


namespace min_value_expression_l139_139258

theorem min_value_expression : 
  ∀ (x y : ℝ), (3 * x * x + 4 * x * y + 4 * y * y - 12 * x - 8 * y ≥ -28) ∧ 
  (3 * ((8:ℝ)/3) * ((8:ℝ)/3) + 4 * ((8:ℝ)/3) * -1 + 4 * -1 * -1 - 12 * ((8:ℝ)/3) - 8 * -1 = -28) := 
by sorry

end min_value_expression_l139_139258


namespace b_and_c_work_days_l139_139141

theorem b_and_c_work_days
  (A B C : ℝ)
  (h1 : A + B = 1 / 8)
  (h2 : A + C = 1 / 8)
  (h3 : A + B + C = 1 / 6) :
  B + C = 1 / 24 :=
sorry

end b_and_c_work_days_l139_139141


namespace find_g5_l139_139071

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l139_139071


namespace range_of_k_l139_139675

noncomputable def op (a b : ℝ) : ℝ := real.sqrt (a * b) + a + b

theorem range_of_k (k : ℝ) (h_pos : 0 < k) : op 1 k < 3 ↔ k > 2 + real.sqrt 3 :=
sorry

end range_of_k_l139_139675


namespace primes_of_form_3k_plus_2_infinite_l139_139810

theorem primes_of_form_3k_plus_2_infinite :
  ∀ n : ℕ, ∃ p ≥ n, nat.prime p ∧ ∃ k : ℕ, p = 3 * k + 2 :=
sorry

end primes_of_form_3k_plus_2_infinite_l139_139810


namespace range_of_slope_angle_l139_139769

def curve (x : ℝ) : ℝ := -x^3 + x + 2 / 3

theorem range_of_slope_angle :
  (∀ P : ℝ, curve P = -P^3 + P + 2 / 3 →
   let slope := -3 * P^2 + 1;
   ∃ α, slope = Math.atan (Real.toReal slope) →
   0 ≤ α ∧ α ≤ Math.pi / 4 ∨ Math.pi / 2 < α ∧ α < Math.pi) :=
by
  sorry

end range_of_slope_angle_l139_139769


namespace relationship_abc_l139_139306

noncomputable def f (x : ℝ) : ℝ := sorry

def F (x : ℝ) : ℝ := x * f x

def a := 2 * f 2
def b := Real.log 2 * f (Real.log 2)
def c := Real.log 8 / Real.log 2 * f (Real.log 8 / Real.log 2)

theorem relationship_abc
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_prime : ∀ x ≤ 0, F' x < 0) :
  b > a ∧ a > c :=
sorry

end relationship_abc_l139_139306


namespace find_g5_l139_139058

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l139_139058


namespace volume_ratio_of_inscribed_cube_l139_139084

theorem volume_ratio_of_inscribed_cube (x : ℝ) (h : x > 0) :
  let a := (2 / 3) * x,
      V_pyramid := (1 / 3) * (x ^ 2) * (2 * x),
      V_cube := a ^ 3
  in V_cube / V_pyramid = 4 / 9 :=
by
  sorry

end volume_ratio_of_inscribed_cube_l139_139084


namespace smallest_sum_divisible_by_5_l139_139635

-- Definition of a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of four consecutive primes greater than 5
def four_consecutive_primes_greater_than_five (a b c d : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ a > 5 ∧ b > 5 ∧ c > 5 ∧ d > 5 ∧ 
  b = a + 4 ∧ c = b + 6 ∧ d = c + 2

-- The statement to prove
theorem smallest_sum_divisible_by_5 :
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) % 5 = 0 ∧
   ∀ x y z w : ℕ, four_consecutive_primes_greater_than_five x y z w → (x + y + z + w) % 5 = 0 → a + b + c + d ≤ x + y + z + w) →
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) = 60) :=
by
  sorry

end smallest_sum_divisible_by_5_l139_139635


namespace magician_earnings_at_least_l139_139172

def magician_starting_decks := 15
def magician_remaining_decks := 3
def decks_sold := magician_starting_decks - magician_remaining_decks
def standard_price_per_deck := 3
def discount := 1
def discounted_price_per_deck := standard_price_per_deck - discount
def min_earnings := decks_sold * discounted_price_per_deck

theorem magician_earnings_at_least :
  min_earnings ≥ 24 :=
by sorry

end magician_earnings_at_least_l139_139172


namespace incorrect_proper_subset_l139_139646

variable (M N : Set)

theorem incorrect_proper_subset (hM : M ⊆ N) : ¬(M ⊂ N) := by
  sorry

end incorrect_proper_subset_l139_139646


namespace find_a_l139_139387

theorem find_a (a : ℝ) (A B : ℝ × ℝ × ℝ) (hA : A = (-1, 1, -a)) (hB : B = (-a, 3, -1)) (hAB : dist A B = 2) : a = -1 := by
  sorry

end find_a_l139_139387


namespace minor_axis_length_l139_139680

theorem minor_axis_length (m : ℝ) (h1 : 10 - m > 0) (h2 : m - 2 > 10 - m > 0) (focal_dist : 2 * 2 = 4) :
  2 * Real.sqrt (10 - m) = 2 * Real.sqrt 2 :=
by
  have h3 : m = 2 * 8 / 2 :=
    sorry
  rw [h3] at h1
  sorry

end minor_axis_length_l139_139680


namespace population_increase_l139_139997

theorem population_increase :
  (∀ p : ℝ, ∀ t : ℝ, t = 0 → t = 2 → p_2 = p * 1.10) →
  (∀ p_2 : ℝ, ∀ t : ℝ, t = 2 → t = 5 → p_5 = p_2 * 1.20) →
  p_5 = p * 1.32 :=
by
  intros p t h1 h2
  sorry

end population_increase_l139_139997


namespace mr_hernandez_tax_l139_139432

theorem mr_hernandez_tax :
  let taxable_income := 42500
  let resident_months := 9
  let standard_deduction := if resident_months > 6 then 5000 else 0
  let adjusted_income := taxable_income - standard_deduction
  let tax_bracket_1 := min adjusted_income 10000 * 0.01
  let tax_bracket_2 := min (max (adjusted_income - 10000) 0) 20000 * 0.03
  let tax_bracket_3 := min (max (adjusted_income - 30000) 0) 30000 * 0.05
  let total_tax_before_credit := tax_bracket_1 + tax_bracket_2 + tax_bracket_3
  let tax_credit := if resident_months < 10 then 500 else 0
  total_tax_before_credit - tax_credit = 575 := 
by
  sorry
  
end mr_hernandez_tax_l139_139432


namespace g_5_is_248_l139_139063

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l139_139063


namespace find_n_l139_139324

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n + 4

-- Define the condition a_n = 13
def condition (n : ℕ) : Prop := a n = 13

-- Prove that under this condition, n = 3
theorem find_n (n : ℕ) (h : condition n) : n = 3 :=
by {
  sorry
}

end find_n_l139_139324


namespace minimum_value_l139_139224

noncomputable def minSum (a b c : ℝ) : ℝ :=
  a / (3 * b) + b / (6 * c) + c / (9 * a)

theorem minimum_value (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  ∃ x : ℝ, (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → minSum a b c ≥ x) ∧ x = 3 / Real.cbrt 162 :=
sorry

end minimum_value_l139_139224


namespace total_tshirts_bought_l139_139217

-- Given conditions
def white_packs : ℕ := 3
def white_tshirts_per_pack : ℕ := 6
def blue_packs : ℕ := 2
def blue_tshirts_per_pack : ℕ := 4

-- Theorem statement: Total number of T-shirts Dave bought
theorem total_tshirts_bought : white_packs * white_tshirts_per_pack + blue_packs * blue_tshirts_per_pack = 26 := by
  sorry

end total_tshirts_bought_l139_139217


namespace variable_value_l139_139718

theorem variable_value (w x v : ℝ) (h1 : 5 / w + 5 / x = 5 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) : v = 0.25 :=
by
  sorry

end variable_value_l139_139718


namespace average_infected_per_round_is_nine_l139_139581

theorem average_infected_per_round_is_nine (x : ℝ) :
  1 + x + x * (1 + x) = 100 → x = 9 :=
by {
  sorry
}

end average_infected_per_round_is_nine_l139_139581


namespace height_of_oil_truck_tank_l139_139162

/-- 
Given that a stationary oil tank is a right circular cylinder 
with a radius of 100 feet and its oil level dropped by 0.025 feet,
proving that if this oil is transferred to a right circular 
cylindrical oil truck's tank with a radius of 5 feet, then the 
height of the oil in the truck's tank will be 10 feet. 
-/
theorem height_of_oil_truck_tank
    (radius_stationary : ℝ) (height_drop_stationary : ℝ) (radius_truck : ℝ) 
    (height_truck : ℝ) (π : ℝ)
    (h1 : radius_stationary = 100)
    (h2 : height_drop_stationary = 0.025)
    (h3 : radius_truck = 5)
    (pi_approx : π = 3.14159265) :
    height_truck = 10 :=
by
    sorry

end height_of_oil_truck_tank_l139_139162


namespace find_m_for_parallel_l139_139703

/-- Define vectors a and b -/
def vector_a : ℝ × ℝ := (1, -3)
def vector_b (m : ℝ) : ℝ × ℝ := (m, 6)

/-- Define the condition for parallel vectors -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The main theorem stating m = -2 when vectors are parallel -/
theorem find_m_for_parallel :
  ∃ m : ℝ, are_parallel vector_a (vector_b m) ∧ m = -2 :=
begin
  -- proof to be provided, but this is the statement
  sorry
end

end find_m_for_parallel_l139_139703


namespace total_views_l139_139933

def first_day_views : ℕ := 4000
def views_after_4_days : ℕ := 40000 + first_day_views
def views_after_6_days : ℕ := views_after_4_days + 50000

theorem total_views : views_after_6_days = 94000 := by
  have h1 : first_day_views = 4000 := rfl
  have h2 : views_after_4_days = 40000 + first_day_views := rfl
  have h3 : views_after_6_days = views_after_4_days + 50000 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end total_views_l139_139933


namespace find_min_and_max_diff_l139_139874

theorem find_min_and_max_diff :
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧
    let x = 1000 * a + 100 * b + 10 * c + d in
    let y = 1000 * d + 100 * c + 10 * b + a in
    x > y ∧ (x - y = 279 ∨ x - y = 8712)) :=
sorry

end find_min_and_max_diff_l139_139874


namespace probability_of_green_ball_is_2_over_5_l139_139213

noncomputable def container_probabilities : ℚ :=
  let prob_A_selected : ℚ := 1/2
  let prob_B_selected : ℚ := 1/2
  let prob_green_in_A : ℚ := 5/10
  let prob_green_in_B : ℚ := 3/10

  prob_A_selected * prob_green_in_A + prob_B_selected * prob_green_in_B

theorem probability_of_green_ball_is_2_over_5 :
  container_probabilities = 2 / 5 := by
  sorry

end probability_of_green_ball_is_2_over_5_l139_139213


namespace ellipse_foci_coordinates_l139_139977

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) → ∃ c : ℝ, (c = 4) ∧ (x = c ∨ x = -c) ∧ (y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l139_139977


namespace price_returns_to_initial_l139_139734

theorem price_returns_to_initial {P₀ P₁ P₂ P₃ P₄ : ℝ} (y : ℝ) (h₁ : P₀ = 100)
  (h₂ : P₁ = P₀ * 1.30) (h₃ : P₂ = P₁ * 0.70) (h₄ : P₃ = P₂ * 1.40) 
  (h₅ : P₄ = P₃ * (1 - y / 100)) : P₄ = P₀ → y = 22 :=
by
  sorry

end price_returns_to_initial_l139_139734


namespace find_g5_l139_139074

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l139_139074


namespace range_of_function_l139_139842

noncomputable def f (x : ℝ) : ℝ := (1 - Real.log10 x) / (1 + Real.log10 x)

theorem range_of_function : 
  ∀ x, (x ≥ 1) → (f x ∈ Set.Ioc (-1) 1) :=
begin
  sorry
end

end range_of_function_l139_139842


namespace sum_coeffs_odd_exp_l139_139783

noncomputable def polynomial_expansion : Polynomial ℤ := Polynomial.X * 3 - 1

theorem sum_coeffs_odd_exp (a0 a1 a2 a3 a4 a5 a6 a7 : ℤ) :
  (polynomial_expansion ^ 7).coeff 0 = a0 →
  (polynomial_expansion ^ 7).coeff 1 = a1 →
  (polynomial_expansion ^ 7).coeff 2 = a2 →
  (polynomial_expansion ^ 7).coeff 3 = a3 →
  (polynomial_expansion ^ 7).coeff 4 = a4 →
  (polynomial_expansion ^ 7).coeff 5 = a5 →
  (polynomial_expansion ^ 7).coeff 6 = a6 →
  (polynomial_expansion ^ 7).coeff 7 = a7 →
  a1 + a3 + a5 + a7 = 8256 := by
  sorry

end sum_coeffs_odd_exp_l139_139783


namespace parallelogram_in_triangle_l139_139912

theorem parallelogram_in_triangle
  (A B C : ℝ)
  (hA : A = 9)
  (hB : B = 15)
  (DE : ℝ)
  (hDE : DE = 6)
  (h_parallelogram : is_parallelogram_in_triangle A B C DE)
  : parallelogram_other_side DE = 4 * Real.sqrt 2 ∧ triangle_base A B C = 18 := 
sorry


end parallelogram_in_triangle_l139_139912


namespace range_of_8x_plus_y_l139_139651

theorem range_of_8x_plus_y (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_condition : 1 / x + 2 / y = 2) : 8 * x + y ≥ 9 :=
by
  sorry

end range_of_8x_plus_y_l139_139651


namespace xiangshan_port_investment_scientific_notation_l139_139823

-- Definition of scientific notation
def in_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

-- Theorem stating the equivalence of the investment in scientific notation
theorem xiangshan_port_investment_scientific_notation :
  in_scientific_notation 7.7 9 7.7e9 :=
by {
  sorry
}

end xiangshan_port_investment_scientific_notation_l139_139823


namespace exponential_equation_solution_count_l139_139641

theorem exponential_equation_solution_count (m n : ℝ) :
  (∃ x : ℝ, m * 4^x + n * 9^x = 6^x) ↔ 
    (n = 0 ∧ m > 0) ∨ 
    (m = 0 ∧ n > 0) ∨ 
    (mn < 0 ∨ (0 < mn ∧ mn < 1/4 ∧ n > 0 → ∃ x1 x2 : ℝ, 
    x1 ≠ x2 ∧ m * 4^x1 + n * 9^x1 = 6^x1 ∧ m * 4^x2 + n * 9^x2 = 6^x2)) ∨ 
    (mn = 1/4 ∧ n > 0 ∧ ∃ x : ℝ, m * 4^x + n * 9^x = 6^x).
    
sorry

end exponential_equation_solution_count_l139_139641


namespace probability_xi_leq_6_l139_139362

noncomputable def redBalls := 4
noncomputable def blackBalls := 3
noncomputable def score (red black : ℕ) : ℕ := red * 1 + black * 3
noncomputable def xi : ℕ → ℕ → ℕ := λ r b, ((r + b = 4) → score r b)
noncomputable def P_xi_leq_6 := (13 : ℕ) / (35 : ℕ)

theorem probability_xi_leq_6 : 
  (∃ red black, score red black ≤ 6 → ((red = 4 ∧ black = 0) ∨ (red = 3 ∧ black = 1)) →
  P_xi_leq_6 = 13 / 35) :=
sorry

end probability_xi_leq_6_l139_139362


namespace calculate_expression_l139_139589

theorem calculate_expression :
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5 / 4 :=
by
  sorry

end calculate_expression_l139_139589


namespace closest_point_on_line_l139_139259

-- Definitions for clarity
def line (a b : ℝ) : ℝ → ℝ := fun x => a * x + b
def is_closest_point (P : ℝ × ℝ) (Q : ℝ × ℝ) (L : ℝ → ℝ) : Prop := 
  ∀ R : ℝ × ℝ, (R.2 = L R.1) → dist P Q ≤ dist R Q

theorem closest_point_on_line (a b : ℝ) (Q : ℝ × ℝ) (P : ℝ × ℝ) :
  (P.1 = 18 / 5 ∧ P.2 = 3) →
  (Q.1 = 2 ∧ Q.2 = 5) →
  (∀ x, a = 2 ∧ b = -3) →
  is_closest_point P Q (line 2 (-3)) :=
by
  -- Proof is omitted
  sorry

end closest_point_on_line_l139_139259


namespace partition_pos_integers_100_subsets_l139_139030

theorem partition_pos_integers_100_subsets :
  ∃ (P : (ℕ+ → Fin 100)), ∀ a b c : ℕ+, (a + 99 * b = c) → P a = P c ∨ P a = P b ∨ P b = P c :=
sorry

end partition_pos_integers_100_subsets_l139_139030


namespace right_triangle_property_l139_139356

-- Variables representing the lengths of the sides and the height of the right triangle
variables (a b c h : ℝ)

-- Hypotheses from the conditions
-- 1. a and b are the lengths of the legs of the right triangle
-- 2. c is the length of the hypotenuse
-- 3. h is the height to the hypotenuse
-- Given equation: 1/2 * a * b = 1/2 * c * h
def given_equation (a b c h : ℝ) : Prop := (1 / 2) * a * b = (1 / 2) * c * h

-- The theorem to prove
theorem right_triangle_property (a b c h : ℝ) (h_eq : given_equation a b c h) : (1 / a^2 + 1 / b^2) = 1 / h^2 :=
sorry

end right_triangle_property_l139_139356


namespace problem_ACD_l139_139661

-- Definitions
def point := (ℝ × ℝ)

def M : point := (1, 2)

def hyperbola (P : point) : Prop := (P.1 ^ 2) / 9 - (P.2 ^ 2) / 16 = 1

def focus_right : point := (5, 0)

def circle (N : point) : Prop := (N.1 + 5) ^ 2 + N.2 ^ 2 = 1

-- Proving the statements
theorem problem_ACD :
  (∀ P : point, hyperbola P → dist P focus_right ≥ 8) ∧
  (∀ P : point, hyperbola P → ∀ N : point, circle N → abs (dist P M - dist N M) = 5 - 2 * real.sqrt 5) ∧
  (∀ t : ℝ, ∀ E : point, E = (-3, t) → abs t = 3 / 2 → dist E (focus_right) = 1 + abs t)
  :=
by
  sorry

end problem_ACD_l139_139661


namespace vector_x_value_l139_139788

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_x_value (x : ℝ) : (perpendicular (a x) b) → x = -2 / 3 := by
  intro h
  sorry

end vector_x_value_l139_139788


namespace minimum_handshakes_l139_139368

theorem minimum_handshakes (n : ℕ) (k : ℕ) (hn : n = 30) (hk : k = 3) :
  (n * k) / 2 = 45 :=
by
  rw [hn, hk]
  exact Nat.div_eq_of_eq_mul Nat.gcd_zero_right rfl
  sorry

end minimum_handshakes_l139_139368


namespace minimum_value_sum_l139_139618

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  (a / (3 * b) + b / (5 * c) + c / (7 * a)) ≥ (3 / Real.cbrt(105)) :=
by
  sorry

end minimum_value_sum_l139_139618


namespace calculate_angle_AMO_l139_139808

-- Definitions based on conditions
variables {A B C M O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] 
variables (radius : ℝ) (center : O) (tangent_point : B) (tangent_line : M) (angle_AMC : real.angle)
variable (is_geometric_progression : ℝ → ℝ → ℝ → Prop)
variables (on_circle : A → Prop) (lie_on_circle : A ∈ circle center radius) (B_lies_on : B ∈ circle center radius) 
variables (C_lies_on : C ∈ circle center radius) (on_tangent : M ∈ tangent_line) (tangent_at_B : circle center radius ∩ tangent_line = {B})
variables (geometric_prog : is_geometric_progression (AM : ℝ) (BM : ℝ) (CM : ℝ))

-- Lean statement
theorem calculate_angle_AMO 
  (lie_on_circle : ∀ x ∈ {A, B, C}, x ∈ circle center radius)
  (on_tangent : M ∈ tangent_line)
  (angle_AMC : angle_AMC = 42)
  (tangent_at_B : B ∈ tangent_point)
  (tangent_line : ∀ x ∈ tangent_line, x ∉ circle center radius ∪ tangent_point)
  (geometric_prog : is_geometric_progression (AM : ℝ) (BM : ℝ) (CM : ℝ)) :
  ∃ angle_AMO, angle_AMO = 52 := sorry

end calculate_angle_AMO_l139_139808


namespace proof_equivalent_triples_l139_139247

noncomputable def valid_triples := 
  { (a, b, c) : ℝ × ℝ × ℝ |
    a * b + b * c + c * a = 1 ∧
    a^2 * b + c = b^2 * c + a ∧
    a^2 * b + c = c^2 * a + b }

noncomputable def desired_solutions := 
  { (a, b, c) |
    (a = 0 ∧ b = 1 ∧ c = 1) ∨
    (a = 0 ∧ b = 1 ∧ c = -1) ∨
    (a = 0 ∧ b = -1 ∧ c = 1) ∨
    (a = 0 ∧ b = -1 ∧ c = -1) ∨

    (a = 1 ∧ b = 1 ∧ c = 0) ∨
    (a = 1 ∧ b = -1 ∧ c = 0) ∨
    (a = -1 ∧ b = 1 ∧ c = 0) ∨
    (a = -1 ∧ b = -1 ∧ c = 0) ∨

    (a = 1 ∧ b = 0 ∧ c = 1) ∨
    (a = 1 ∧ b = 0 ∧ c = -1) ∨
    (a = -1 ∧ b = 0 ∧ c = 1) ∨
    (a = -1 ∧ b = 0 ∧ c = -1) ∨

    ((a = (Real.sqrt 3) / 3 ∧ b = (Real.sqrt 3) / 3 ∧ 
      c = (Real.sqrt 3) / 3) ∨
     (a = -(Real.sqrt 3) / 3 ∧ b = -(Real.sqrt 3) / 3 ∧ 
      c = -(Real.sqrt 3) / 3)) }

theorem proof_equivalent_triples :
  valid_triples = desired_solutions :=
sorry

end proof_equivalent_triples_l139_139247


namespace length_XY_is_correct_l139_139743

noncomputable def XY_length
  (FGH_sim_XYZ : ∀ (a b c d e f : ℝ), a / b = c / d → e / f = a / b)
  (GH : ℝ) (YZ : ℝ) (FX : ℝ) (XG : ℝ) : ℝ :=
  let FG := FX + XG in
  let ratio := YZ / GH in
  (YZ * FG) / GH

theorem length_XY_is_correct
  (GH := 18 : ℝ) (YZ := 9 : ℝ) (FX := 15 : ℝ) (XG := 8 : ℝ) :
  XY_length (fun _ _ _ _ _ _ _ => by sorry) GH YZ FX XG = 11.5 :=
sorry

end length_XY_is_correct_l139_139743


namespace find_area_of_triangle_l139_139391

noncomputable def area_of_triangle (b B C : ℝ) : ℝ := (1 / 2) * b * (b * (Real.sin C / Real.sin B)) * Real.sin (π - B - C)

theorem find_area_of_triangle (b : ℝ) (B C : ℝ) (hb : b = 2) (hB : B = π / 6) (hC : C = π / 4) :
  area_of_triangle b B C = sqrt 3 + 1 :=
by {
  rw [hb, hB, hC],
  simp,
  sorry
}

end find_area_of_triangle_l139_139391


namespace problem_a_4_l139_139613

def a_sequence : ℕ → ℝ
| 1       := 0.2
| 2       := (0.21)^0.2
| (k + 1) := if (k + 1) % 2 = 1 then (0.2 + 0.005 * (k + 1))^(a_sequence k)
             else (0.21 * (a_sequence k))^(a_sequence k)

theorem problem_a_4 :
  a_sequence 4 = ((0.21 * (0.215)^(0.21)^(0.2))^(0.215)^(0.21)^(0.2)) := sorry

end problem_a_4_l139_139613


namespace number_of_possible_values_for_n_l139_139560

-- We define the set and the necessary conditions for the problem
def initial_set := {2, 5, 8, 11}
def sum_initial_set := 2 + 5 + 8 + 11  -- which equals 26
def mean_initial_set_with_n (n : ℤ) := (sum_initial_set + n) / 5
def possible_medians (n : ℤ) := ({5, 8} : set ℤ) ∪ {n}

-- We state the mathematically equivalent proof problem
theorem number_of_possible_values_for_n : 
  let ns := { n : ℤ | ∃ m ∈ possible_medians n, mean_initial_set_with_n n = 2 * m }
  in fintype.card ns = 1 :=
by
  sorry

end number_of_possible_values_for_n_l139_139560


namespace mass_percentage_of_O_in_BaO_l139_139256

theorem mass_percentage_of_O_in_BaO :
  let mass_Ba := 137.33
  let mass_O := 16.00
  let mass_BaO := mass_Ba + mass_O
  let percentage_O := (mass_O / mass_BaO) * 100
  percentage_O ≈ 10.43 := 
by
  let mass_Ba := 137.33 : Real
  let mass_O := 16.00 : Real
  let mass_BaO := mass_Ba + mass_O
  let percentage_O := (mass_O / mass_BaO) * 100
  show percentage_O ≈ 10.43
  sorry

end mass_percentage_of_O_in_BaO_l139_139256


namespace coeff_x3_expansion_l139_139742

noncomputable def binom : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 m := 0
| n m := binom (n - 1) (m - 1) + binom (n - 1) m

def expansion_term (r : ℕ) : ℕ := binom 3 r * (-2)^r

theorem coeff_x3_expansion : 
  let term1 := expansion_term 1 
  let term2 := expansion_term 0 
  term1 + term2 = -5 :=
by
  let term1 := expansion_term 1 
  let term2 := expansion_term 0 
  have : term1 = -6 := by calc 
    term1 = binom 3 1 * (-2)^1 := rfl
        ... = 3 * (-2) := dec_trivial
        ... = -6 := rfl
  have : term2 = 1 := by calc 
    term2 = binom 3 0 * (-2)^0 := rfl
        ... = 1 * 1 := dec_trivial
        ... = 1 := rfl
  calc 
    term1 + term2 = -6 + 1 := rfl
           ...  = -5 := rfl

end coeff_x3_expansion_l139_139742


namespace log_bounds_l139_139720

-- Definitions and assumptions
def tenCubed : Nat := 1000
def tenFourth : Nat := 10000
def twoNine : Nat := 512
def twoFourteen : Nat := 16384

-- Statement that encapsulates the proof problem
theorem log_bounds (h1 : 10^3 = tenCubed) 
                   (h2 : 10^4 = tenFourth) 
                   (h3 : 2^9 = twoNine) 
                   (h4 : 2^14 = twoFourteen) : 
  (2 / 7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (1 / 3 : ℝ) :=
sorry

end log_bounds_l139_139720


namespace distances_on_circumcircle_l139_139540

theorem distances_on_circumcircle 
  {A B C P : Point}
  (h1 h2 h3 : ℝ)
  (a b c : ℝ)
  (h_circumcircle : P ∈ circumcircle(A, B, C))
  (h_opposite: P and A are on opposite sides of BC)
  (h_distances: distance(P, BC) = h1 ∧ distance(P, CA) = h2 ∧ distance(P, AB) = h3) :
  a / h1 = b / h2 + c / h3 := 
sorry

end distances_on_circumcircle_l139_139540


namespace rationality_problem_l139_139528

-- Defining the numbers 
def n1 := Real.sqrt (Real.exp 2)
def n2 := Real.cbrt 0.216
def n3 := Real.root 4 0.0625
def n4 := Real.cbrt (-8) * Real.sqrt (0.25⁻¹)

-- Prove that these numbers satisfy the given conditions
theorem rationality_problem : 
  ¬ Rational.is_rational n1 ∧ -- \(\sqrt{e^2}\)
  Rational.is_rational n2 ∧ -- \(\sqrt[3]{0.216}\)
  Rational.is_rational n3 ∧ -- \(\sqrt[4]{0.0625}\)
  Rational.is_rational n4 := -- \(\sqrt[3]{-8} \cdot \sqrt{(0.25)^{-1}}\)
by
  sorry

end rationality_problem_l139_139528


namespace triangle_side_values_l139_139835

theorem triangle_side_values {m : ℕ} (h1 : real.log10 15 + real.log10 90 > real.log10 m)
                             (h2 : real.log10 15 + real.log10 m > real.log10 90)
                             (h3 : real.log10 90 + real.log10 m > real.log10 15)
                             (h4 : m > 0) :
    (7 ≤ m ∧ m ≤ 1349) → nat.card (set.Icc 7 1349) = 1343 :=
by {
  intros _,
  exact ((1349 - 7) + 1)
}

end triangle_side_values_l139_139835


namespace base_AD_correct_l139_139750

-- Definitions for the trapezoid and its properties
structure Trapezoid :=
  (A B C D : Type)
  (BC AB CD : ℝ)
  (angle_diagonals : ℝ)
  (BC_eq : BC = 3)
  (AB_eq : AB = 3)
  (CD_eq : CD = 3)
  (angle_diagonals_eq : angle_diagonals = 60)

-- Function to compute the base AD given the conditions of the trapezoid
def base_AD (T : Trapezoid) : ℝ :=
  if T.BC_eq = 3 ∧ T.AB_eq = 3 ∧ T.CD_eq = 3 ∧ T.angle_diagonals_eq = 60 then
    6
  else
    sorry

-- Theorem statement to prove the base AD
theorem base_AD_correct (T : Trapezoid) : base_AD T = 6 := by
  -- Proof omitted
  sorry

end base_AD_correct_l139_139750


namespace budget_remaining_l139_139572

noncomputable theory
open_locale big_operators

def conversion_rate : ℝ := 1.2
def last_year_budget_euros : ℝ := 6
def this_year_allocation : ℝ := 50
def additional_grant : ℝ := 20
def gift_card : ℝ := 10

def initial_price_textbooks : ℝ := 45
def discount_textbooks : ℝ := 0.15
def tax_textbooks : ℝ := 0.08

def initial_price_notebooks : ℝ := 18
def discount_notebooks : ℝ := 0.10
def tax_notebooks : ℝ := 0.05

def initial_price_pens : ℝ := 27
def discount_pens : ℝ := 0.05
def tax_pens : ℝ := 0.06

def initial_price_art_supplies : ℝ := 35
def discount_art_supplies : ℝ := 0
def tax_art_supplies : ℝ := 0.07

def initial_price_folders : ℝ := 15
def voucher_folders : ℝ := 5
def tax_folders : ℝ := 0.04

def convert_budget (e: ℝ) (r: ℝ): ℝ := e * r

def calculate_discounted_price (price: ℝ) (discount: ℝ) : ℝ :=
  price - price * discount

def calculate_tax_price (price: ℝ) (tax: ℝ) : ℝ :=
  price + price * tax

def compute_final_price (price: ℝ) (discount: ℝ) (tax: ℝ) : ℝ :=
  calculate_tax_price (calculate_discounted_price(price, discount), tax)

def total_budget : ℝ :=
  convert_budget last_year_budget_euros conversion_rate +
  this_year_allocation +
  additional_grant +
  gift_card

def cost_textbooks : ℝ :=
  compute_final_price(initial_price_textbooks, discount_textbooks, tax_textbooks)

def cost_notebooks : ℝ :=
  compute_final_price(initial_price_notebooks, discount_notebooks, tax_notebooks)

def cost_pens : ℝ :=
  compute_final_price(initial_price_pens, discount_pens, tax_pens)

def cost_art_supplies : ℝ :=
  compute_final_price(initial_price_art_supplies, discount_art_supplies, tax_art_supplies)

def cost_folders : ℝ :=
  compute_final_price(initial_price_folders - voucher_folders, 0, tax_folders)

def total_cost : ℝ :=
  cost_textbooks +
  cost_notebooks +
  cost_pens +
  cost_art_supplies +
  cost_folders

theorem budget_remaining :
  total_budget - (total_cost - gift_card) = -36.16 :=
sorry

end budget_remaining_l139_139572


namespace normal_vector_of_plane_l139_139291

theorem normal_vector_of_plane (
  A B C: ℝ × ℝ × ℝ, -- Points in 3D space
  n: ℝ × ℝ × ℝ      -- Normal vector candidate
) 
  (hA : A = (0, 0, 0))
  (hB : B = (0, 0, 1))
  (hC : C = (1, 1, 0))
: (n = (1, -1, 0)) ∨ (n = (-1, 1, 0)) := 
sorry

end normal_vector_of_plane_l139_139291


namespace find_ABCDE_l139_139534

theorem find_ABCDE : 
  ∃ (A B C D E : ℕ), 
    (∀ x y : ℕ, x ≠ y → x ∉ {A, B, C, D, E} → y ∉ {A, B, C, D, E}) ∧
    (ABCDE = 10000 * A + 1000 * B + 100 * C + 10 * D + E) ∧ 
    (BCDE = 1000 * B + 100 * C + 10 * D + E) ∧
    (CDE = 100 * C + 10 * D + E) ∧
    (DE = 10 * D + E) ∧ 
    (ABCDE + BCDE + CDE + DE + E = 11111 * A) ∧ 
    (ABCDE = 52487) := 
by
  sorry

end find_ABCDE_l139_139534


namespace xiaoming_money_l139_139501

open Real

noncomputable def verify_money_left (M P_L : ℝ) : Prop := M = 12 * P_L

noncomputable def verify_money_right (M P_R : ℝ) : Prop := M = 14 * P_R

noncomputable def price_relationship (P_L P_R : ℝ) : Prop := P_R = P_L - 1

theorem xiaoming_money (M P_L P_R : ℝ) 
  (h1 : verify_money_left M P_L) 
  (h2 : verify_money_right M P_R) 
  (h3 : price_relationship P_L P_R) : 
  M = 84 := 
  by
  sorry

end xiaoming_money_l139_139501


namespace convex_hexagon_largest_angle_l139_139483

theorem convex_hexagon_largest_angle 
  (x : ℝ)                                 -- Denote the measure of the third smallest angle as x.
  (angles : Fin 6 → ℝ)                     -- Define the angles as a function from Fin 6 to ℝ.
  (h1 : ∀ i : Fin 6, angles i = x + (i : ℝ) - 3)  -- The six angles in increasing order.
  (h2 : 0 < x - 3 ∧ x - 3 < 180)           -- Convex condition: each angle is between 0 and 180.
  (h3 : angles ⟨0⟩ + angles ⟨1⟩ + angles ⟨2⟩ + angles ⟨3⟩ + angles ⟨4⟩ + angles ⟨5⟩ = 720) -- Sum of interior angles of a hexagon.
  : (∃ a, a = angles ⟨5⟩ ∧ a = 122.5) :=   -- Prove the largest angle in this arrangement is 122.5.
sorry

end convex_hexagon_largest_angle_l139_139483


namespace GuntherFreeTime_l139_139335

def GuntherCleaning : Nat := 45 + 60 + 30 + 15

def TotalFreeTime : Nat := 180

theorem GuntherFreeTime : TotalFreeTime - GuntherCleaning = 30 := by
  sorry

end GuntherFreeTime_l139_139335


namespace A_made_profit_of_1_yuan_l139_139182

noncomputable def profit_A_made : ℝ :=
  let initial_price : ℝ := 1000
  let profit_A_to_B : ℝ := initial_price * 0.1
  let price_B_to_A : ℝ := (initial_price + profit_A_to_B) * 0.9
  let final_sale_price : ℝ := price_B_to_A * 0.9
  in profit_A_to_B - (price_B_to_A - final_sale_price)

theorem A_made_profit_of_1_yuan :
  profit_A_made = 1 := by
    sorry

end A_made_profit_of_1_yuan_l139_139182


namespace square_perimeter_l139_139825

theorem square_perimeter :
  let rectangle_length := 50
  let rectangle_width := 10
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  let square_perimeter := 4 * square_side
  square_perimeter = 200 := by
  let rectangle_length := 50
  let rectangle_width := 10
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  let square_perimeter := 4 * square_side
  have h1: square_perimeter = 4 * (Real.sqrt (5 * (rectangle_length * rectangle_width)))
  sorry

end square_perimeter_l139_139825


namespace interviewee_count_l139_139857

theorem interviewee_count (p : ℚ) (n : ℕ) (h : p = 1 / 70) :
  (2.choose 2 * (n - 2).choose 1) / n.choose 3 = p → n = 21 :=
by
  intro prob_eq
  have h_prob : (2.choose 2 * (n - 2).choose 1) / n.choose 3 = 1 / 70 := by rw [prob_eq, h]
  sorry

end interviewee_count_l139_139857


namespace min_selling_price_is_400_l139_139208

-- Definitions for the problem conditions
def total_products := 20
def average_price := 1200
def less_than_1000_count := 10
def price_of_most_expensive := 11000
def total_retail_price := total_products * average_price

-- The theorem to state the problem condition and the expected result
theorem min_selling_price_is_400 (x : ℕ) :
  -- Condition 1: Total retail price
  total_retail_price =
  -- 10 products sell for x dollars
  (10 * x) +
  -- 9 products sell for 1000 dollars
  (9 * 1000) +
  -- 1 product sells for the maximum price 11000
  price_of_most_expensive → 
  -- Conclusion: The minimum price x is 400
  x = 400 :=
by
  sorry

end min_selling_price_is_400_l139_139208


namespace probability_two_girls_from_twelve_l139_139901

theorem probability_two_girls_from_twelve : 
  let total_members := 12
  let boys := 4
  let girls := 8
  let choose_two_total := Nat.choose total_members 2
  let choose_two_girls := Nat.choose girls 2
  let probability := (choose_two_girls : ℚ) / (choose_two_total : ℚ)
  probability = (14 / 33) := by
  -- Proof goes here
  sorry

end probability_two_girls_from_twelve_l139_139901


namespace largest_angle_of_consecutive_integers_hexagon_l139_139468

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l139_139468


namespace find_y_ratio_l139_139819

variable {R : Type} [LinearOrderedField R]
variables (x y : R → R) (x1 x2 y1 y2 : R)

-- Condition: x is inversely proportional to y, so xy is constant.
def inversely_proportional (x y : R → R) : Prop := ∀ (a b : R), x a * y a = x b * y b

-- Condition: ∀ nonzero x values, we have these specific ratios
variable (h_inv_prop : inversely_proportional x y)
variable (h_ratio_x : x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 / x2 = 4 / 5)
variable (h_nonzero_y : y1 ≠ 0 ∧ y2 ≠ 0)

-- Claim to prove
theorem find_y_ratio : (y1 / y2) = 5 / 4 :=
by
  sorry

end find_y_ratio_l139_139819


namespace sequence_sum_l139_139592

theorem sequence_sum :
  let S := (list.iota 101)
  in (list.sum (S.map (λ i, if i % 2 = 0 then 2020 - 10 * i else 2000 - 10 * (i - 1)))) = 1010 :=
by
  sorry

end sequence_sum_l139_139592


namespace largest_angle_of_consecutive_integers_hexagon_l139_139469

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l139_139469


namespace wire_length_from_sphere_volume_l139_139143

theorem wire_length_from_sphere_volume
  (r_sphere : ℝ) (r_cylinder : ℝ) (h : ℝ)
  (h_sphere : r_sphere = 12)
  (h_cylinder : r_cylinder = 4)
  (volume_conservation : (4/3 * Real.pi * r_sphere^3) = (Real.pi * r_cylinder^2 * h)) :
  h = 144 :=
by {
  sorry
}

end wire_length_from_sphere_volume_l139_139143


namespace g_5_is_248_l139_139061

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l139_139061


namespace g_5_is_248_l139_139062

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l139_139062


namespace marcus_baseball_cards_l139_139793

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the number of additional cards Marcus has compared to Carter
def additional_cards : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + additional_cards

-- The proof statement asserting Marcus' total number of baseball cards
theorem marcus_baseball_cards : marcus_cards = 210 :=
by {
  -- This is where the proof steps would go, but we are skipping with sorry
  sorry
}

end marcus_baseball_cards_l139_139793


namespace max_value_of_sum_of_cubes_l139_139773

theorem max_value_of_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 := by
  sorry

end max_value_of_sum_of_cubes_l139_139773


namespace log5_930_nearest_integer_l139_139235

theorem log5_930_nearest_integer :
  ∀ (a : ℤ), (log (5 : ℝ) (625 : ℝ) = 4) ∧ (5^4 = 625) ∧ (log (5 : ℝ) (3125 : ℝ) = 5) ∧ (5^5 = 3125) → (a = 4) → (log (5 : ℝ) (930 : ℝ) = a) → a = 4 :=
by
  intros a h_conditions h_assumption h_computation,
  sorry

end log5_930_nearest_integer_l139_139235


namespace min_perimeter_section_l139_139748

variable {a α β : ℝ}

/-
Given:
- \(ABCD\) is a tetrahedron.
- \(AB \perp CD\).
- \(AB = a\).
- The angle between \(AB\) and the plane \(BCD\) is \(\alpha\).
- The dihedral angle \(A-CD-B\) is \(\beta\).

Prove:
The minimum perimeter of the section of the tetrahedron passing through the edge \(AB\) is \( \frac{a}{\sin \beta} \left[\sin \alpha + \sin \beta + \sin (\alpha + \beta)\right] \).
-/

theorem min_perimeter_section 
  (h1 : (ABCD : Type))
  (h2 : AB ⟂ CD)
  (h3 : AB = a)
  (h4 : angle_between AB (plane BCD) = α)
  (h5 : dihedral_angle A CD B = β) :
  let P := \frac{a}{\sin β} \left[\sin \alpha + \sin β + \sin (\alpha + β)\right] in
  ∃ (perimeter : ℝ), perimeter = P :=
sorry

end min_perimeter_section_l139_139748


namespace part_I_part_II_l139_139691

def f (x b : ℝ) := -x^3 + x^2 + b
def g (x a : ℝ) := a * log x

theorem part_I (b : ℝ) :
  (∀ x ∈ set.Ico (-1/2 : ℝ) 1, ∀ y ∈ set.Ico (-1/2 : ℝ) 1, 
    f x b ≤ f y b) → f (2/3 : ℝ) b = 3/8 → b = 0 := sorry

theorem part_II (a : ℝ) :
  (∀ x ∈ set.Icc (1 : ℝ) real.exp 1, g x a ≥ -x^2 + (a+2)*x) → a ≤ -1 := sorry

end part_I_part_II_l139_139691


namespace marcus_has_210_cards_l139_139795

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the increment of baseball cards Marcus has over Carter
def increment : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + increment

-- Prove that Marcus has 210 baseball cards
theorem marcus_has_210_cards : marcus_cards = 210 :=
by simp [marcus_cards, carter_cards, increment]

end marcus_has_210_cards_l139_139795


namespace sum_of_swapped_digits_l139_139489

theorem sum_of_swapped_digits :
  let n := 123456789
  let m := 987654321
  let prod := n * 8
  (prod = m : Prop) ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (m = swap_digits prod d1 d2))
  → d1 + d2 = 3 :=
by
  sorry

end sum_of_swapped_digits_l139_139489


namespace mahdi_plays_tennis_on_monday_l139_139428

def day := ℕ -- Representing each day by a number 0 (Sunday) through 6 (Saturday)

def plays_sport (d : day) : Prop := true

def runs (d : day) : Prop := true
def basketball (d : day) : Prop := (d = 3)  -- Thursday is day 3
def golf (d : day) : Prop := (d = 1)  -- Two days before Thursday is Tuesday, day 1
def swims (d : day) : Prop := true
def tennis (d : day) : Prop := true

-- Conditions
axiom one_sport_each_day (d : day) : plays_sport d
axiom runs_three_days (days : list day) : (∀ d ∈ days, runs d) ∧ (days.length = 3)
axiom non_consecutive_run_days (d1 d2 : day) : runs d1 → runs d2 → (d1 ≠ d2 + 1) ∧ (d1 ≠ d2 - 1)
axiom basketball_thursday : basketball 3
axiom golf_tuesday : golf 1
axiom no_tennis_after_swimming (d : day) : swims d → tennis (d + 1) → false
axiom no_tennis_before_running (d : day) : runs d → tennis (d - 1) → false

-- Prove that Mahdi plays tennis on Monday
theorem mahdi_plays_tennis_on_monday : tennis 0 := by
  sorry

end mahdi_plays_tennis_on_monday_l139_139428


namespace sum_of_cubes_eq_square_of_sum_l139_139017

theorem sum_of_cubes_eq_square_of_sum (n : ℕ) :
  (∑ i in Finset.range (n + 1), i ^ 3) = (∑ i in Finset.range (n + 1), i) ^ 2 :=
by sorry

end sum_of_cubes_eq_square_of_sum_l139_139017


namespace root_in_interval_implies_k_l139_139724

open Classical

noncomputable def f (x : ℝ) : ℝ := log 10 x + x - 3

theorem root_in_interval_implies_k (k : ℤ) (root_in_interval : ∃ x : ℝ, f x = 0 ∧ k < x ∧ x < k + 1) : k = 2 :=
  sorry

end root_in_interval_implies_k_l139_139724


namespace mary_blue_marbles_l139_139611

theorem mary_blue_marbles (dan_blue_marbles mary_blue_marbles : ℕ)
  (h1 : dan_blue_marbles = 5)
  (h2 : mary_blue_marbles = 2 * dan_blue_marbles) : mary_blue_marbles = 10 := 
by
  sorry

end mary_blue_marbles_l139_139611


namespace fraction_area_below_diagonal_is_one_l139_139807

noncomputable def fraction_below_diagonal (s : ℝ) : ℝ := 1

theorem fraction_area_below_diagonal_is_one (s : ℝ) :
  let long_side := 2 * s
  let P := (2 * s / 3, 0)
  let Q := (s, s / 2)
  -- Total area of the rectangle
  let total_area := s * 2 * s -- 2s^2
  -- Total area below the diagonal
  let area_below_diagonal := 2 * s * s  -- 2s^2
  -- Fraction of the area below diagonal
  fraction_below_diagonal s = area_below_diagonal / total_area := 
by 
  sorry

end fraction_area_below_diagonal_is_one_l139_139807


namespace hawkeye_charged_4_times_l139_139338

variables (C B L S : ℝ) (N : ℕ)
def hawkeye_charging_problem : Prop :=
  C = 3.5 ∧ B = 20 ∧ L = 6 ∧ S = B - L ∧ N = (S / C) → N = 4 

theorem hawkeye_charged_4_times : hawkeye_charging_problem C B L S N :=
by {
  repeat { sorry }
}

end hawkeye_charged_4_times_l139_139338


namespace selection_methods_l139_139500

-- Define the number of students and lectures.
def numberOfStudents : Nat := 6
def numberOfLectures : Nat := 5

-- Define the problem as proving the number of selection methods equals 5^6.
theorem selection_methods : (numberOfLectures ^ numberOfStudents) = 15625 := by
  -- Include the proper mathematical equivalence statement
  sorry

end selection_methods_l139_139500


namespace h_in_terms_of_L_and_m_min_h_when_L_fixed_l139_139701

variables {ℝ : Type} [linear_ordered_field ℝ]

def parabola_y (x : ℝ) := x^2

def midpoint_y (p q : ℝ) := (p^2 + q^2) / 2

def length_segment (p q : ℝ) := real.sqrt ((q - p)^2 * (1 + (p + q)^2))

def slope_segment (p q : ℝ) := p + q

theorem h_in_terms_of_L_and_m (L m : ℝ) (h : ℝ) :
  h = 1/4 * (L^2 / (1 + m^2) + m^2) :=
sorry

theorem min_h_when_L_fixed (L h : ℝ) :
  (L < 1 → h = L^2 / 4) ∧ (L ≥ 1 → h = (2 * L - 1) / 4) :=
sorry

end h_in_terms_of_L_and_m_min_h_when_L_fixed_l139_139701


namespace product_of_solutions_l139_139261

theorem product_of_solutions :
  (∀ y : ℝ, (|y| = 2 * (|y| - 1)) → y = 2 ∨ y = -2) →
  (∀ y1 y2 : ℝ, (y1 = 2 ∧ y2 = -2) → y1 * y2 = -4) :=
by
  intro h
  have h1 := h 2
  have h2 := h (-2)
  sorry

end product_of_solutions_l139_139261


namespace count_integers_satisfying_Q_le_0_l139_139954

def Q (x : ℤ) : ℤ :=
  (x - 1^2) * (x - 2^2) * (x - 3^2) * (x - 4^2) * (x - 5^2) *
  (x - 6^2) * (x - 7^2) * (x - 8^2) * (x - 9^2) * (x - 10^2) *
  (x - 11^2) * (x - 12^2) * (x - 13^2) * (x - 14^2) * (x - 15^2) *
  (x - 16^2) * (x - 17^2) * (x - 18^2) * (x - 19^2) * (x - 20^2) *
  (x - 21^2) * (x - 22^2) * (x - 23^2) * (x - 24^2) * (x - 25^2) *
  (x - 26^2) * (x - 27^2) * (x - 28^2) * (x - 29^2) * (x - 30^2) *
  (x - 31^2) * (x - 32^2) * (x - 33^2) * (x - 34^2) * (x - 35^2) *
  (x - 36^2) * (x - 37^2) * (x - 38^2) * (x - 39^2) * (x - 40^2) *
  (x - 41^2) * (x - 42^2) * (x - 43^2) * (x - 44^2) * (x - 45^2) *
  (x - 46^2) * (x - 47^2) * (x - 48^2) * (x - 49^2) * (x - 50^2)

theorem count_integers_satisfying_Q_le_0 : 
  { m : ℤ | Q m ≤ 0 }.finite.to_finset.card = 1300 := 
sorry

end count_integers_satisfying_Q_le_0_l139_139954


namespace eccentricity_is_correct_l139_139297

section hyperbola

variables (a b c : ℝ) (P : ℝ × ℝ) (ecc : ℝ)
variables (a_pos : 0 < a) (b_pos : 0 < b)
variables (hyperbola_eq : P.1 = -3 * (a^2 / c) ∧ P.2 = -2 * b)
variables (on_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)

def hyperbola_eccentricity (a b c : ℝ) : ℝ := c / a

theorem eccentricity_is_correct :
  let e := hyperbola_eccentricity a b c in
  (9 * a^2 / c^2 = 5) → e = (3 * Real.sqrt 5 / 5) :=
by {
  sorry
}

end hyperbola

end eccentricity_is_correct_l139_139297


namespace pentagon_coloring_l139_139989

theorem pentagon_coloring (convex : Prop) (unequal_sides : Prop)
  (colors : Prop) (adjacent_diff_color : Prop) :
  ∃ n : ℕ, n = 30 := by
  -- Definitions for conditions (in practical terms, these might need to be more elaborate)
  let convex := true           -- Simplified representation
  let unequal_sides := true    -- Simplified representation
  let colors := true           -- Simplified representation
  let adjacent_diff_color := true -- Simplified representation
  
  -- Proof that the number of coloring methods is 30
  existsi 30
  sorry

end pentagon_coloring_l139_139989


namespace petya_catch_bus_l139_139024

theorem petya_catch_bus 
    (v_p v_b d : ℝ) 
    (h1 : v_b = 5 * v_p)
    (h2 : ∀ t : ℝ, 5 * v_p * t ≤ 0.6) 
    : d = 0.12 := 
sorry

end petya_catch_bus_l139_139024


namespace sum_of_reciprocal_f_l139_139779

def f (n : ℕ) : ℤ :=
  let m := (Real.cbrt n).toInt
  if (Real.cbrt n) - m < 0.5 then m else m + 1

theorem sum_of_reciprocal_f :
  (∑ k in Finset.range 2023, (1 : ℝ) / (f (k + 1))) = 251.385 := by
  sorry

end sum_of_reciprocal_f_l139_139779


namespace triangle_not_acute_third_side_length_l139_139191

theorem triangle_not_acute_third_side_length (a b : ℕ) (h₁ : a = 20) (h₂ : b = 19) :
  (∃ s : ℕ, 2 ≤ s ∧ s ≤ 6 ∨ 28 ≤ s ∧ s ≤ 38) → (nat.card {s // (2 ≤ s ∧ s ≤ 6) ∨ (28 ≤ s ∧ s ≤ 38)} = 16) :=
by
  sorry

end triangle_not_acute_third_side_length_l139_139191


namespace max_segment_length_l139_139600

theorem max_segment_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
∃ P Q R S : Type,
  ∃ (QR : P → Q → ℝ) (PS : P → S → ℝ) (PR : P → R → ℝ) 
    (PQ : P → Q → ℝ),
    -- Triangle PQR
    QR P Q = a ∧ PR P R = b ∧
    -- Angle bisector intersection at S
    PS P S ∧
    -- Requirement with angle bisector theorem and trigonometric identities
    ∀ (LM : P → R → ℝ) (P-Q-S conditions), 
      max_length (LM P _ P QR) = (2 * a * b) / real.sqrt (a^2 + b^2):= by
  sorry

end max_segment_length_l139_139600


namespace binomial_coeff_difference_l139_139830

theorem binomial_coeff_difference (x : ℝ) :
  (1 - x) ^ 10 = ∑ r in finset.range (11), (-1)^r * (nat.choose 10 r) * x^r →
  (∑ r in finset.range (11), (-1)^r * (nat.choose 10 r) * (if r = 1 then x else 0)) - 
  (∑ r in finset.range (11), (-1)^r * (nat.choose 10 r) * (if r = 9 then x else 0)) = 0 :=
by 
  sorry

end binomial_coeff_difference_l139_139830


namespace carols_total_peanuts_l139_139593

-- Define the initial number of peanuts Carol has
def initial_peanuts : ℕ := 2

-- Define the number of peanuts given by Carol's father
def peanuts_given : ℕ := 5

-- Define the total number of peanuts Carol has
def total_peanuts : ℕ := initial_peanuts + peanuts_given

-- The statement we need to prove
theorem carols_total_peanuts : total_peanuts = 7 := by
  sorry

end carols_total_peanuts_l139_139593


namespace inner_circle_radius_l139_139107

theorem inner_circle_radius (r : ℝ) (r_outer : ℝ) (A1 A2 : ℝ)
  (h1 : r_outer = 8)
  (h2 : A1 = π * (r_outer ^ 2 - r ^ 2))
  (h3 : r_outer + 0.25 * r_outer = 10)
  (h4 : r - 0.5 * r = 0.5 * r)
  (h5 : A2 = π * (10 ^ 2 - (0.5 * r) ^ 2))
  (h6 : A2 = 3.25 * A1) :
  r = 6 :=
begin
  sorry
end

end inner_circle_radius_l139_139107


namespace average_visitors_other_days_l139_139167

-- Define the given conditions
variables (avg_sundays avg_other_days avg_monthly : ℕ)
variable (days_in_month : ℕ) 
variable (sundays other_days : ℕ)

-- Specific given values
variables (h1 : avg_sundays = 140)
variables (h2 : avg_monthly = 90)
variables (h3 : days_in_month = 30)
variables (h4 : sundays = 5)
variables (h5 : other_days = 25)

-- Proof goal
theorem average_visitors_other_days :
  (5 * avg_sundays + 25 * avg_other_days = 30 * avg_monthly) → 
  avg_other_days = 80 :=
by
  intros h
  rw [h1, h2, h3, h4, h5] at h
  -- other steps are not needed, so we add sorry
  sorry

end average_visitors_other_days_l139_139167


namespace total_views_correct_l139_139927

-- Definitions based on the given conditions
def initial_views : ℕ := 4000
def views_increase := 10 * initial_views
def additional_views := 50000
def total_views_after_6_days := initial_views + views_increase + additional_views

-- The theorem we are going to state
theorem total_views_correct :
  total_views_after_6_days = 94000 :=
sorry

end total_views_correct_l139_139927


namespace fractions_sum_equals_one_l139_139410

variable {a b c x y z : ℝ}

variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 29 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

theorem fractions_sum_equals_one (a b c x y z : ℝ) 
  (h1 : 17 * x + b * y + c * z = 0)
  (h2 : a * x + 29 * y + c * z = 0)
  (h3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 := by 
  sorry

end fractions_sum_equals_one_l139_139410


namespace train_length_l139_139920

-- Definitions of the conditions as Lean terms/functions
def V (L : ℕ) := (L + 170) / 15
def U (L : ℕ) := (L + 250) / 20

-- The theorem to prove that the length of the train is 70 meters.
theorem train_length : ∃ L : ℕ, (V L = U L) → L = 70 := by
  sorry

end train_length_l139_139920


namespace find_r_plus_s_l139_139770

def parabola (x : ℝ) : ℝ := 2 * x^2

def line_through_Q (m x : ℝ) : ℝ := m * (x - 10) - 6

theorem find_r_plus_s :
  let Q := (10, -6)
  let m := x in
  let intersection_eq := 2 * x^2 + 6 = m * (x - 10) in
  let quadratic_eq := 2 * x^2 - m * x + (10 * m + 6) in
  ∀ r s: ℝ, (r < m ∧ m < s) → m^2 - 80 * m - 48 < 0 → r + s = 80 := 
sorry

end find_r_plus_s_l139_139770


namespace quadratic_inequality_solution_range_l139_139694

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end quadratic_inequality_solution_range_l139_139694


namespace proof_problem_l139_139786

-- Define sets A and B according to the given conditions
def A : Set ℝ := { x | x ≥ -1 }
def B : Set ℝ := { x | x > 2 }
def complement_B : Set ℝ := { x | ¬ (x > 2) }  -- Complement of B

-- Remaining intersection expression
def intersect_expr : Set ℝ := { x | x ≥ -1 ∧ x ≤ 2 }

-- Statement to prove
theorem proof_problem : (A ∩ complement_B) = intersect_expr :=
sorry

end proof_problem_l139_139786


namespace total_area_inf_series_l139_139039

noncomputable def S (n : ℕ) : ℝ := 1 / 2 ^ (n - 1)
noncomputable def T (n : ℕ) : ℝ := 1 / 4 * (1 / 2 ^ (n - 1))

theorem total_area_inf_series : (∑ n in (Finset.range ∞), S n + T n) = 5 / 2 :=
by
  sorry

end total_area_inf_series_l139_139039


namespace convex_hexagon_largest_angle_l139_139481

theorem convex_hexagon_largest_angle 
  (x : ℝ)                                 -- Denote the measure of the third smallest angle as x.
  (angles : Fin 6 → ℝ)                     -- Define the angles as a function from Fin 6 to ℝ.
  (h1 : ∀ i : Fin 6, angles i = x + (i : ℝ) - 3)  -- The six angles in increasing order.
  (h2 : 0 < x - 3 ∧ x - 3 < 180)           -- Convex condition: each angle is between 0 and 180.
  (h3 : angles ⟨0⟩ + angles ⟨1⟩ + angles ⟨2⟩ + angles ⟨3⟩ + angles ⟨4⟩ + angles ⟨5⟩ = 720) -- Sum of interior angles of a hexagon.
  : (∃ a, a = angles ⟨5⟩ ∧ a = 122.5) :=   -- Prove the largest angle in this arrangement is 122.5.
sorry

end convex_hexagon_largest_angle_l139_139481


namespace range_of_a_l139_139325

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, ((x - 2) ^ 2 + (y - 1) ^ 2 ≤ 1) → (2 * |x - 1| + |y - 1| ≤ a)) ↔ a ≥ 2 + Real.sqrt 5 :=
by 
  sorry

end range_of_a_l139_139325


namespace anthony_pets_ratio_l139_139938

variable (C D : ℕ)

theorem anthony_pets_ratio
  (h1 : C + D = 12)
  (h2 : (C / 2 : ℕ) + (D + 7) + (C + D) = 27) :
  C / (C + D) = 2 / 3 :=
by
  sorry

end anthony_pets_ratio_l139_139938


namespace angle_correct_l139_139328

open Real

noncomputable def angle_between_vectors (θ : ℝ) (h : θ ∈ Ioo (π / 2) π) : ℝ :=
3 * π / 2 - θ

theorem angle_correct (θ : ℝ) (hθ : θ ∈ Ioo (π / 2) π) :
  let a := (2 * cos θ, 2 * sin θ)
  let b := (0, -2)
  angle_between_vectors θ hθ = 3 * π / 2 - θ :=
by
  let a := (2 * cos θ, 2 * sin θ)
  let b := (0, -2)
  sorry

end angle_correct_l139_139328


namespace log_base32_four_l139_139627

theorem log_base32_four : ∀ (b x : ℝ), b = 32 → x = 4 → log b x = (2 / 5) := by
  sorry

end log_base32_four_l139_139627


namespace seating_arrangement_l139_139739

theorem seating_arrangement :
  ∃ (n : ℕ), n = 2 * (6 * 5 * 4 * 3) ∧ n = 720 :=
by
  use 720
  split
  { calc
      720 = 2 * (6 * 5 * 4 * 3) : by norm_num
  }
  { refl }

end seating_arrangement_l139_139739


namespace range_f_1_l139_139312

noncomputable def f (x : ℝ) (a : ℝ) := 2 * x ^ 2 - a * x + 5

theorem range_f_1 (a : ℝ) (h : a ≤ 4) : 7 - a ∈ set.Ici (3) :=
begin
  simp only [set.mem_Ici],
  have ha_ge_neg4 : -a ≥ -4 := by linarith,
  linarith,
end

end range_f_1_l139_139312


namespace bells_ring_together_l139_139168

open Nat

theorem bells_ring_together :
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  next_ring_time / total_minutes_in_an_hour = 6 :=
by
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  have h_next_ring_time : next_ring_time = 360 := by
    sorry
  have h_hours : next_ring_time / total_minutes_in_an_hour = 6 := by
    sorry
  exact h_hours

end bells_ring_together_l139_139168


namespace quadractic_roots_eq_l139_139462

theorem quadractic_roots_eq:
  ∀ (p q : ℝ) (A B M R T : Point),
  length (A - B) = p ∧
  midpoint A B M ∧
  perpendicular_from_midpoint A B M R ∧
  length (M - R) = q ∧
  arc_from_point R (p / 2) T ∧
  intersection AB T ∧
  is_root (x^2 - px + q^2) (length (A - T)) ∧
  is_root (x^2 - px + q^2) (length (T - B)) :=
by
  intro p q A B M R T
  assume length (A - B) = p
  assume midpoint A B M
  assume perpendicular_from_midpoint A B M R
  assume length (M - R) = q
  assume arc_from_point R (p / 2) T
  assume intersection AB T
  sorry

end quadractic_roots_eq_l139_139462


namespace equivalence_mod_equivalence_divisible_l139_139354

theorem equivalence_mod (a b c : ℤ) :
  (∃ k : ℤ, a - b = k * c) ↔ (a % c = b % c) := by
  sorry

theorem equivalence_divisible (a b c : ℤ) :
  (a % c = b % c) ↔ (∃ k : ℤ, a - b = k * c) := by
  sorry

end equivalence_mod_equivalence_divisible_l139_139354


namespace determine_chipped_marbles_l139_139033

def seven_bags := [15, 20, 22, 31, 33, 37, 40]

def total_marbles (bags : List ℕ) : ℕ :=
  bags.sum

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem determine_chipped_marbles :
  let total := total_marbles seven_bags in
  ∃ chipped_marble (chip_free_bags : List ℕ),
  chipped_marble ∈ seven_bags ∧
  chip_free_bags = seven_bags.erase chipped_marble ∧
  (∃ jane (jane_bags george_bags : List ℕ),
   jane_bags.length = 4 ∧
   george_bags.length = 2 ∧
   jane_bags ++ george_bags = chip_free_bags ∧
   total_marbles jane_bags = 1.5 * total_marbles george_bags ∧
   is_divisible_by_5 (total_marbles jane_bags + total_marbles george_bags)) ∧
  chipped_marble = 33 :=
by sorry

end determine_chipped_marbles_l139_139033


namespace largest_angle_of_consecutive_integers_in_hexagon_l139_139470

theorem largest_angle_of_consecutive_integers_in_hexagon : 
  ∀ (a : ℕ), 
    (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) = 720 → 
    a + 3 = 122.5 :=
by sorry

end largest_angle_of_consecutive_integers_in_hexagon_l139_139470


namespace geometric_sequence_log_sum_l139_139281

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (c : ℝ) (h1 : ∀ n, S n = 2^n - c)
  (h2 : ∏ i in finset.range 5, real.log (a i) / real.log 2 = 10) : 
  ∀ n, n = 5 :=
by
  sorry

end geometric_sequence_log_sum_l139_139281


namespace sum_of_first_18_terms_l139_139098

noncomputable def sum_of_first_n_terms (a d : ℕ → ℕ) (n : ℕ) : ℕ := n * (2 * (a 1) + (n - 1) * d 1) / 2

variable (a d : ℕ → ℕ)
variable h1 : sum_of_first_n_terms a d 6 = 30
variable h2 : sum_of_first_n_terms a d 12 = 100

theorem sum_of_first_18_terms : 
  sum_of_first_n_terms a d 18 = 210 :=
sorry

end sum_of_first_18_terms_l139_139098


namespace maximize_f_should_inspect_l139_139160

-- Definition of f(p) and its properties
def f (p : ℝ) : ℝ := (nat.choose 20 2) * p^2 * (1 - p)^18

-- Definition of derivative f'
def f' (p : ℝ) : ℝ := (nat.choose 20 2) * (2 * p * (1 - p)^18 - 18 * p^2 * (1 - p)^17)

-- Condition on p within (0, 1)
lemma p_in_range (p : ℝ) : 0 < p ∧ p < 1 := sorry

-- Proof that f is maximized at p = 0.1 and not elsewhere within (0, 1)
theorem maximize_f : ∀ p : ℝ, p_in_range p → p ≠ 0.1 → f' p < 0 ∨ f' p > 0 :=
sorry

-- Assign p_0 the value 0.1 obtained from (1)
def p_0 : ℝ := 0.1

-- Definition of EX given p = p_0
def EX : ℝ := 40 + 25 * (180 * p_0)

-- Cost-effectiveness decision based on given expectations
theorem should_inspect (EX : ℝ) : EX > 400 :=
by
  have EX_val : EX = 490 := by
    rw [EX, @p_0, mul_assoc, ← mul_assoc 25 180 0.1, mul_comm 0.1, mul_assoc 180 25 0.1, mul_assoc, mul_comm 20 10]
    exact rfl
  linarith


end maximize_f_should_inspect_l139_139160


namespace problem1_problem2_l139_139690

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + (1 / 6) * x^3 + a * x^2 - x + 1

-- Problem 1: g(x) is the derivative of f(x)
def g (x : ℝ) (a : ℝ) : ℝ := f x a

-- Problem 1: g'(x) = derivative of g(x)
def g' (x : ℝ) (a : ℝ) : ℝ := (1 / x) + x + 2 * a

theorem problem1 (a : ℝ) : (∀ x > 0, g' x a ≥ 0) ↔ -1 ≤ a :=
sorry

-- Problem 2: When a = -1/8
def h (x : ℝ) : ℝ := x * Real.log x - (1 / 8) * x^2 - x + 1

-- Problem 2: Function h(x) - 1/6 x^3
def h_extreme (x : ℝ) : ℝ := h x

theorem problem2 (x1 x2 : ℝ) (a : ℝ) : 
  a = -1 / 8 → x1 < x2 → h_extreme (a := a) x1 = 0 → h_extreme (a := a) x2 = 0 → 
  (x1 * x2^2 > Real.exp 3) :=
sorry

end problem1_problem2_l139_139690


namespace arithmetic_sequence_l139_139287

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h_n : n > 0) 
  (h_Sn : S (2 * n) - S (2 * n - 1) + a 2 = 424) : 
  a (n + 1) = 212 :=
sorry

end arithmetic_sequence_l139_139287


namespace dot_product_a_b_solve_lambda_l139_139704

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def dot_product (v₁ v₂ : Vector2D) : ℝ :=
  v₁.x * v₂.x + v₁.y * v₂.y

def a : Vector2D := ⟨1, 2⟩
def b : Vector2D := ⟨2, -2⟩

theorem dot_product_a_b : dot_product a b = -2 :=
by
  sorry

theorem solve_lambda (λ : ℝ) : dot_product a ⟨a.x + λ * b.x, a.y + λ * b.y⟩ = 0 → λ = 5 / 2 :=
by
  sorry

end dot_product_a_b_solve_lambda_l139_139704


namespace find_m_range_l139_139660

noncomputable def p (m : ℝ) :=
  (m^2 - 16 > 0) ∧ (-m < 0) ∧ (4 > 0)

noncomputable def q (m : ℝ) :=
  ((4*(m-2))^2 - 4*4 < 0)

theorem find_m_range (m : ℝ) :
  ((p(m) ∨ q(m)) ∧ ¬(p(m) ∧ q(m))) ↔ (1 < m ∧ m < 3) ∨ (4 < m) :=
  sorry

end find_m_range_l139_139660


namespace area_between_hexagon_and_square_l139_139568

noncomputable def circleRadius : ℝ := 6

noncomputable def centralAngleSquare : ℝ := Real.pi / 2

noncomputable def centralAngleHexagon : ℝ := Real.pi / 3

noncomputable def areaSegment (r α : ℝ) : ℝ :=
  0.5 * r^2 * (α - Real.sin α)

noncomputable def areaBetweenArcs : ℝ :=
  let r := circleRadius
  let T_AB := areaSegment r centralAngleSquare
  let T_CD := areaSegment r centralAngleHexagon
  2 * (T_AB - T_CD)

theorem area_between_hexagon_and_square :
  abs (areaBetweenArcs - 14.03) < 0.01 :=
by
  sorry

end area_between_hexagon_and_square_l139_139568


namespace wholesale_price_of_milk_l139_139557

theorem wholesale_price_of_milk (W : ℝ) 
  (h1 : ∀ p : ℝ, p = 1.25 * W) 
  (h2 : ∀ q : ℝ, q = 0.95 * (1.25 * W)) 
  (h3 : q = 4.75) :
  W = 4 :=
by
  sorry

end wholesale_price_of_milk_l139_139557


namespace jenny_games_ratio_l139_139403

theorem jenny_games_ratio :
  ∀ (x : ℕ),
  (∀ (games_played_with_Mark games_won_with_Mark games_won_with_Jill total_wins : ℕ),
    games_played_with_Mark = 10 →
    games_won_with_Mark = 9 →
    total_wins = 14 →
    games_won_with_Jill = total_wins - games_won_with_Mark →
    games_won_with_Jill = 0.25 * x →
    ratio : ℕ :=
    ratio = x / games_played_with_Mark)
  → ∃ x, ratio = 2 :=
begin
  intros x h,
  existsi 20,
  have h1 : (0.25 : ℝ) * 20 = 5, by norm_num,
  have h2 : 20 / 10 = 2, by norm_num,
  rw [h1, h2],
  exact h x 10 9 14 5 rfl rfl rfl rfl h1,
end

end jenny_games_ratio_l139_139403


namespace arithmetic_sequence_sum_l139_139294

noncomputable def S : ℕ → ℤ := sorry  -- since we are defining the sum function

theorem arithmetic_sequence_sum :
  (S 2017) = 2017 :=
begin
  -- Conditions
  assume S_n_arithmetic_sequence : ∀ n, S n = A * n^2 + B * n,
  assume a1_is_neg2016 : ∀ a1, a1 = -2016,
  assume condition : (S 2014 / 2014) - (S 2008 / 2008) = 6,

  -- Proof would go here, but is replaced by sorry for the statement
  sorry
end

end arithmetic_sequence_sum_l139_139294


namespace new_pencil_length_is_nine_l139_139755

-- Declare the lengths of the pencils
def length_first_pencil : ℕ := 12
def length_second_pencil : ℕ := 12
def length_third_pencil : ℕ := 8

-- Define the fractions used from the second and third pencils
def used_length_second_pencil : ℚ := 1 / 2 * length_second_pencil
def used_length_third_pencil : ℚ := 1 / 3 * length_third_pencil

-- Define the total length of the new pencil
def new_pencil_length : ℕ := (used_length_second_pencil + used_length_third_pencil).round

-- Prove that the length of the new pencil is 9 cubes
theorem new_pencil_length_is_nine : new_pencil_length = 9 :=
by
  sorry

end new_pencil_length_is_nine_l139_139755


namespace giftWrapperPerDay_l139_139234

variable (giftWrapperPerBox : ℕ)
variable (boxesPer3Days : ℕ)

def giftWrapperUsedIn3Days := giftWrapperPerBox * boxesPer3Days

theorem giftWrapperPerDay (h_giftWrapperPerBox : giftWrapperPerBox = 18)
  (h_boxesPer3Days : boxesPer3Days = 15) : giftWrapperUsedIn3Days giftWrapperPerBox boxesPer3Days / 3 = 90 :=
by
  sorry

end giftWrapperPerDay_l139_139234


namespace dietary_preferences_count_correct_l139_139437

noncomputable def students_pref_liked (total_liked : Nat) := 
  let vegans := 0.25 * total_liked 
  let vegetarians := 0.20 * total_liked 
  let gluten_free := 0.15 * total_liked 
  let no_pref := 0.40 * total_liked 
  (vegans, vegetarians, gluten_free, no_pref)

noncomputable def students_pref_disliked (total_disliked : Nat) := 
  let lactose_intolerant := 0.35 * total_disliked 
  let low_sodium := 0.30 * total_disliked 
  let pescatarians := 0.10 * total_disliked 
  let meat_lovers := 0.25 * total_disliked 
  (lactose_intolerant, low_sodium, pescatarians, meat_lovers)

theorem dietary_preferences_count_correct : 
  let liked_prefs := students_pref_liked 300 in 
  let disliked_prefs := students_pref_disliked 200 in 
  liked_prefs = (75, 60, 45, 120) ∧ disliked_prefs = (70, 60, 20, 50) := by
  sorry

end dietary_preferences_count_correct_l139_139437


namespace quadrilateral_intersection_parallelogram_l139_139564

theorem quadrilateral_intersection_parallelogram 
  (P: Plane) (Q: Quadrilateral)
  (h1: ∃ pts: Finset Point, pts.card = 4 ∧ ∀ p ∈ pts, p ∈ P ∧ p ∈ Q)
  (h2: ∀ d1 d2: Line, is_diagonal d1 Q ∧ is_diagonal d2 Q → parallel d1 P ∧ parallel d2 P) : 
  ∃ R: Quadrilateral, is_parallelogram R ∧ ∀ pt ∈ R.points, pt ∈ P ∧ pt ∈ Q := 
sorry

end quadrilateral_intersection_parallelogram_l139_139564


namespace winning_candidate_percentage_l139_139863

theorem winning_candidate_percentage :
  let votes1 := 5136
  let votes2 := 7636
  let votes3 := 11628
  let total_votes := votes1 + votes2 + votes3
  let winning_votes := 11628
  let percentage := (winning_votes / total_votes.toFloat) * 100
  percentage ≈ 47.66 := 
by
  sorry

end winning_candidate_percentage_l139_139863


namespace exists_integer_cube_ends_with_2007_ones_l139_139230

theorem exists_integer_cube_ends_with_2007_ones :
  ∃ x : ℕ, x^3 % 10^2007 = 10^2007 - 1 :=
sorry

end exists_integer_cube_ends_with_2007_ones_l139_139230


namespace three_2x2_squares_exceed_100_l139_139485

open BigOperators

noncomputable def sum_of_1_to_64 : ℕ :=
  (64 * (64 + 1)) / 2

theorem three_2x2_squares_exceed_100 :
  ∀ (s : Fin 16 → ℕ),
    (∑ i, s i = sum_of_1_to_64) →
    (∀ i j, i ≠ j → s i = s j ∨ s i > s j ∨ s i < s j) →
    (∃ i₁ i₂ i₃, i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₁ ≠ i₃ ∧ s i₁ > 100 ∧ s i₂ > 100 ∧ s i₃ > 100) := sorry

end three_2x2_squares_exceed_100_l139_139485


namespace smallest_odd_prime_factor_of_2021_pow_10_plus_1_l139_139227

theorem smallest_odd_prime_factor_of_2021_pow_10_plus_1 : 
  ∃ p : ℕ, nat.prime p ∧ p = 61 ∧ p ∣ (2021^10 + 1) ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ (2021^10 + 1) → 61 ≤ q := 
sorry

end smallest_odd_prime_factor_of_2021_pow_10_plus_1_l139_139227


namespace find_t_l139_139045

theorem find_t (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
    (havg : (k + m + r + s + t) / 5 = 18)
    (hmed : r = 23) 
    (hpos_k : 0 < k)
    (hpos_m : 0 < m)
    (hpos_r : 0 < r)
    (hpos_s : 0 < s)
    (hpos_t : 0 < t) :
  t = 40 := sorry

end find_t_l139_139045


namespace covered_squares_in_checkerboard_l139_139551

def distance (x y: ℝ) : ℝ := real.sqrt (x^2 + y^2)
def covers_square (D i j: ℝ) : Prop := distance (i * D - 4 * D) (j * D - 4 * D) ≤ 4 * D
def quadrant_covered_squares (D: ℝ) : ℕ :=
  finset.card {arr ∈ (finset.iota 4).product (finset.iota 4) | covers_square D arr.1 arr.2}

theorem covered_squares_in_checkerboard (D: ℝ) : 4 * quadrant_covered_squares D = 32 :=
  sorry

end covered_squares_in_checkerboard_l139_139551


namespace count_even_three_digit_numbers_with_sum_9_l139_139707

theorem count_even_three_digit_numbers_with_sum_9 : 
  (∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 2 = 0 ∧ (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in d2 + d3 = 9) ∧
  (∀ k, 100 ≤ k ∧ k < 1000 ∧ k % 2 = 0 ∧ (let d1 := k / 100, d2 := (k / 10) % 10, d3 := k % 10 in d2 + d3 = 9) ↔ n = k)) → 
  (Set.count {n | 100 ≤ n ∧ n < 1000 ∧ n % 2 = 0 ∧ (let d2 := (n / 10) % 10, d3 := n % 10 in d2 + d3 = 9)} = 45) :=
by
  sorry

end count_even_three_digit_numbers_with_sum_9_l139_139707


namespace Doris_spent_6_l139_139109

variable (D : ℝ)

theorem Doris_spent_6 (h0 : 24 - (D + D / 2) = 15) : D = 6 :=
by
  sorry

end Doris_spent_6_l139_139109


namespace probability_at_least_one_tree_survives_l139_139923

noncomputable def prob_at_least_one_survives (survival_rate_A survival_rate_B : ℚ) (n_A n_B : ℕ) : ℚ :=
  1 - ((1 - survival_rate_A)^(n_A) * (1 - survival_rate_B)^(n_B))

theorem probability_at_least_one_tree_survives :
  prob_at_least_one_survives (5/6) (4/5) 2 2 = 899 / 900 :=
by
  sorry

end probability_at_least_one_tree_survives_l139_139923


namespace total_ingredients_l139_139845

structure Recipe (butter flour sugar total: ℕ)

theorem total_ingredients
  (h_ratio: Recipe 2 5 3 1)
  (h_sugar: 9 = 3 * h_ratio.sugar):
  h_ratio.butter * 3 + h_ratio.flour * 3 + h_ratio.sugar * 3 = 30 :=
by
  sorry

end total_ingredients_l139_139845


namespace geo_seq_property_l139_139300

theorem geo_seq_property (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n+1) = r * a n)
  (h4_8 : a 4 + a 8 = -3) : a 6 * (a 2 + 2 * a 6 + a 10) = 9 := 
sorry

end geo_seq_property_l139_139300


namespace no_common_solution_l139_139229

theorem no_common_solution :
  ¬(∃ y : ℚ, (6 * y^2 + 11 * y - 1 = 0) ∧ (18 * y^2 + y - 1 = 0)) :=
by
  sorry

end no_common_solution_l139_139229


namespace vector_problem_l139_139659

variables {a b c : Nat → ℝ}
variables {x y : ℝ}

def vectorLength (v : Nat → ℝ) : ℝ :=
  real.sqrt (∑ i in finset.range 2, v i ^ 2)

def dotProduct (u v : Nat → ℝ) : ℝ :=
  ∑ i in finset.range 2, u i * v i

noncomputable def maximum_value (a b : Nat → ℝ) : ℝ := 
  -- Assume values
sorry

noncomputable def minimum_value (a b : Nat → ℝ) : ℝ := 
  -- Assume values
sorry

theorem vector_problem
  (a b c : Nat → ℝ)
  (h1 : vectorLength (λ i, a i - b i) = 4)
  (h2 : vectorLength b = 4)
  (h3 : dotProduct (λ i, a i - c i) (λ i, b i - c i) = 0)
  (h4 : maximum_value a b = 2 + 2 * real.sqrt 3)
  (h5 : minimum_value a b = 2 * real.sqrt 3 - 2) :
  maximum_value a b - minimum_value a b = 4 :=
sorry

end vector_problem_l139_139659


namespace second_smallest_relative_prime_210_l139_139263

theorem second_smallest_relative_prime_210 (x : ℕ) (h1 : x > 1) (h2 : Nat.gcd x 210 = 1) : x = 13 :=
sorry

end second_smallest_relative_prime_210_l139_139263


namespace geometric_sequence_property_converse_geometric_sequence_inverse_geometric_sequence_false_contrapositive_geometric_sequence_count_correct_statements_l139_139895

variables {a b c : ℝ}

def is_geometric_sequence (a b c : ℝ) : Prop := (b / a) = (c / b)

theorem geometric_sequence_property : 
  is_geometric_sequence a b c → (b * b = a * c) := 
by
  assume h : is_geometric_sequence a b c
  sorry -- proof omitted

theorem converse_geometric_sequence : 
  (b * b = a * c) → is_geometric_sequence a b c := 
by
  assume h : b * b = a * c
  sorry -- proof omitted

theorem inverse_geometric_sequence_false : 
  ¬is_geometric_sequence a b c → ¬(b * b = a * c) := 
by
  assume h : ¬is_geometric_sequence a b c
  sorry -- proof omitted

theorem contrapositive_geometric_sequence : 
  ¬(b * b = a * c) → ¬is_geometric_sequence a b c := 
by
  assume h : ¬(b * b = a * c)
  sorry -- proof omitted

theorem count_correct_statements : 
  (geometric_sequence_property a b c) ∧
  (converse_geometric_sequence b a c) ∧
  ¬(inverse_geometric_sequence_false a b c) ∧
  (contrapositive_geometric_sequence a b c) → 
  3 = 3 :=
by
  assume h : (geometric_sequence_property a b c) ∧
  (converse_geometric_sequence b a c) ∧
  ¬(inverse_geometric_sequence_false a b c) ∧
  (contrapositive_geometric_sequence a b c)
  trivial

end geometric_sequence_property_converse_geometric_sequence_inverse_geometric_sequence_false_contrapositive_geometric_sequence_count_correct_statements_l139_139895


namespace dice_probability_l139_139506

open Set

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def red_die_outcomes : Finset ℕ := (Finset.range 13).filter is_even
def green_die_outcomes : Finset ℕ := (Finset.range 13).filter is_perfect_square

def total_outcomes := 12 * 12
def successful_outcomes := red_die_outcomes.card * green_die_outcomes.card

noncomputable def probability_event := (successful_outcomes : ℚ) / total_outcomes

theorem dice_probability : probability_event = 1 / 8 := 
sorry

end dice_probability_l139_139506


namespace sphere_radius_l139_139854

theorem sphere_radius {r1 r2 : ℝ} (w1 w2 : ℝ) (S : ℝ → ℝ) 
  (h1 : S r1 = 4 * Real.pi * r1^2)
  (h2 : S r2 = 4 * Real.pi * r2^2)
  (w_s1 : w1 = 8)
  (w_s2 : w2 = 32)
  (r2_val : r2 = 0.3)
  (prop : ∀ r, w_s2 = w1 * S r2 / S r1 → w2 = w1 * S r2 / S r1 ) :
  r1 = 0.15 :=
by sorry

end sphere_radius_l139_139854


namespace danny_initial_amount_l139_139542

theorem danny_initial_amount
  (total_amount : ℝ)
  (share_ratios : ℝ × ℝ × ℝ × ℝ)
  (removed_amounts : ℝ × ℝ × ℝ × ℝ)
  (H_total : total_amount = 2210)
  (H_ratios : share_ratios = (11, 18, 24, 32))
  (H_removed : removed_amounts = (30, 50, 40, 80)) :
  ∃ (D_initial : ℝ), D_initial = 916.80 :=
by
  let k := 2010 / 85
  let A := 11 * k + 30
  let B := 18 * k + 50
  let C := 24 * k + 40
  let D := 32 * k + 80
  have H1 : A + B + C + D = 2210 := by
    -- Proof omitted
    sorry
  have H2 : k = 23.65 := by
    -- Proof omitted
    sorry
  let D_initial := D + 80
  have H3 : D_initial = 916.80 := by
    -- Proof omitted
    sorry
  use D_initial
  exact H3

end danny_initial_amount_l139_139542


namespace collinear_circumcenters_l139_139583

theorem collinear_circumcenters
  (A B C M N O1 O2 : Point)
  (hABC_acute : Triangle acute A B C)
  (hAB_AC : AB A B > AC A C)
  (hM_N_on_BC : M ≠ N ∧ (linesegment B C).has_point M ∧ (linesegment B C).has_point N)
  (h_bam_can : ∠ B A M = ∠ C A N)
  (hO1_circumcenter_ABC : circumcenter A B C = O1)
  (hO2_circumcenter_AMN : circumcenter A M N = O2) :
  collinear {O1, O2, A} :=
sorry

end collinear_circumcenters_l139_139583


namespace classrooms_student_rabbit_difference_l139_139966

-- Definitions from conditions
def students_per_classroom : Nat := 20
def rabbits_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Theorem statement
theorem classrooms_student_rabbit_difference :
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 102 := by
  sorry

end classrooms_student_rabbit_difference_l139_139966


namespace total_number_of_coins_l139_139104

theorem total_number_of_coins (n_pockets n_coins_per_pocket : ℕ) (h1 : n_pockets = 4) (h2 : n_coins_per_pocket = 16) :
  n_pockets * n_coins_per_pocket = 64 :=
by
  rw [h1, h2]
  norm_num
  sorry 

end total_number_of_coins_l139_139104


namespace find_g5_l139_139078

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l139_139078


namespace largest_possible_N_l139_139003

theorem largest_possible_N (N : ℕ) :
  let divisors := Nat.divisors N
  in (1 ∈ divisors) ∧ (N ∈ divisors) ∧ (divisors.length ≥ 3) ∧ (divisors[divisors.length - 3] = 21 * divisors[1]) → N = 441 := 
by
  sorry

end largest_possible_N_l139_139003


namespace blocks_needed_l139_139547

noncomputable def block_volume : ℝ := 6 * 2 * 1
noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

theorem blocks_needed (h_cylinder : ℝ) (d_cylinder : ℝ) (V_block : ℝ) (r_cylinder := d_cylinder / 2) :
  h_cylinder = 10 → d_cylinder = 5 → V_block = 12 →
  let V_cylinder := cylinder_volume r_cylinder h_cylinder in
  let num_blocks := V_cylinder / V_block in
  ⌈num_blocks⌉ = 17 :=
by
  intros
  sorry

end blocks_needed_l139_139547


namespace union_M_N_eq_U_l139_139647

def U : Set Nat := {2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5, 7}
def N : Set Nat := {2, 4, 5, 6}

theorem union_M_N_eq_U : M ∪ N = U := 
by {
  -- Proof would go here
  sorry
}

end union_M_N_eq_U_l139_139647


namespace spade_to_heart_l139_139214

-- Definition for spade and heart can be abstract geometric shapes
structure Spade := (arcs_top: ℕ) (stem_bottom: ℕ)
structure Heart := (arcs_top: ℕ) (pointed_bottom: ℕ)

-- Condition: the spade symbol must be cut into three parts
def cut_spade (s: Spade) : List (ℕ × ℕ) :=
  [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)]

-- Define a function to verify if the rearranged parts form a heart
def can_form_heart (pieces: List (ℕ × ℕ)) : Prop :=
  pieces = [(1, 0), (0, 1), (0, 1)]

-- The theorem that the spade parts can form a heart
theorem spade_to_heart (s: Spade) (h: Heart):
  (cut_spade s) = [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)] →
  can_form_heart [(s.arcs_top, 0), (s.stem_bottom, 0), (s.stem_bottom, 0)] := 
by
  sorry


end spade_to_heart_l139_139214


namespace least_common_denominator_l139_139590

-- We first need to define the function to compute the LCM of a list of natural numbers.
def lcm_list (ns : List ℕ) : ℕ :=
ns.foldr Nat.lcm 1

theorem least_common_denominator : 
  lcm_list [3, 4, 5, 8, 9, 11] = 3960 := 
by
  -- Here's where the proof would go
  sorry

end least_common_denominator_l139_139590


namespace minimum_value_sum_alpha_beta_l139_139310

def f (x : ℝ) := (x - 2) * Real.exp x

-- Minimum value h(t) on [t, t+2]
def h (t : ℝ) : ℝ :=
  if t ≤ -1 then
    t * Real.exp (t + 2)
  else if t ≤ 1 then
    -Real.exp 1
  else
    (t - 2) * Real.exp t

theorem minimum_value (t : ℝ) : 
  ∀ x ∈ Set.Icc t (t + 2), f(t) ≤ f(x) :=
by sorry 

theorem sum_alpha_beta {α β : ℝ} (h_diff : α ≠ β) (h_eq : f α = f β) : α + β < 2 :=
by sorry

end minimum_value_sum_alpha_beta_l139_139310


namespace bisector_intersections_vary_l139_139737

noncomputable def nature_of_intersections (O B C A D : Point) (circle : Circle) :=
  ¬(BC.is_diameter) ∧ (angle BAC = 60) ∧ (CD.is_perpendicular BC) ∧ (C.moves_along arc BAC)
  → (intersection_points_vary)

theorem bisector_intersections_vary (O B C A D : Point) (circle : Circle) :
  (¬(BC.is_diameter) ∧ (angle BAC = 60) ∧ (CD.is_perpendicular BC) ∧ (C.moves_along arc BAC)) →
  (intersection_points_vary) :=
by
  sorry

end bisector_intersections_vary_l139_139737


namespace no_solutions_abs_sin_eq_two_l139_139640

theorem no_solutions_abs_sin_eq_two : ∀ (q : ℝ), ¬ (| || sin (2 * |q - 5|) - 10 | - 5 | = 2) :=
by
  intros q
  have h1 : | x | ≥ 0 for all x := abs_nonneg x
  have h2 : | x | = 2 implies x = 2 or x = -2 := abs_eq h1
  have h3 : sin (x) in [-1, 1] for all x := sin_range x
  sorry

end no_solutions_abs_sin_eq_two_l139_139640


namespace large_triangle_area_l139_139859

/-
Conditions:
1. A central equilateral triangle of side length 2.
2. Extensions forming three additional equilateral triangles (each with side length 2).

Prove:
The area of the large triangle formed by the centers of the three outer equilateral triangles is 4 * sqrt(3).
-/

theorem large_triangle_area 
  (central_length : ℝ) 
  (outer_length : ℝ) 
  (h1 : central_length = 2) 
  (h2 : outer_length = 2) :
  ∃ (area : ℝ), area = 4 * real.sqrt 3 :=
by
  sorry

end large_triangle_area_l139_139859


namespace no_matrix_triples_second_column_l139_139981

open Matrix

def matrix_N_triples_second_column (N : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  ∀ (a b c d : ℚ),
    mul N (Matrix.of ![![a, b], ![c, d]]) = (Matrix.of ![![a, 3*b], ![c, 3*d]])

theorem no_matrix_triples_second_column :
  ∀ N : Matrix (Fin 2) (Fin 2) ℚ, ¬ matrix_N_triples_second_column N :=
begin
  intro N,
  assume h,
  sorry
end

end no_matrix_triples_second_column_l139_139981


namespace max_final_product_l139_139917

theorem max_final_product (a : ℕ → ℕ) (h_sorted : ∀ i j, i < j → a i ≥ a j) :
  let b := λ i, a (2002 - i)
  (finset.range 1001).prod (λ i, (2 * a i + b i)) =
  (2 * a 0 + a 2001) * (2 * a 1 + a 2000) * ... * (2 * a 1000 + a 1001) :=
sorry

end max_final_product_l139_139917


namespace find_pqr_sum_l139_139373

-- Definitions for the geometric elements involved
structure Triangle := (P Q R : Point)
structure Point := (x y : ℝ)

def radius : ℝ := 3
def PS : ℝ := 15
def PT : ℝ := 9
def Angle60 := Real.pi / 3
def Angle120 := 2 * Real.pi / 3

-- Problem statement in Lean
theorem find_pqr_sum (P Q R S T U V : Point)
  (triangle_PQR : Triangle P Q R) 
  (circle_inscribed : ∀ A : Point, dist A P = radius ∨ dist A Q = radius ∨ dist A R = radius)
  (extend_PQ : S.x = Q.x + PS ∧ S.y = Q.y)
  (extend_PR : T.x = R.x + PT ∧ T.y = R.y)
  (parallel_PT_l1 : ∀ x : ℝ, S.y - T.y ≠ 0 → (U.y = S.y + (T.y - S.y) / (T.x - S.x) * (x - S.x)))
  (parallel_PS_l2 : ∀ x : ℝ, T.y - S.y ≠ 0 → (U.y = T.y + (S.y - T.y) / (S.x - T.x) * (x - T.x)))
  (collinear : collinear P U V)
  (distinct_pv : P ≠ V) :
  let p := 27
      q := 3
      r := 49
  in p + q + r = 79 := 
by 
  sorry -- Proof goes here

end find_pqr_sum_l139_139373


namespace minimum_k_points_l139_139802

theorem minimum_k_points (A B : ℝ × ℝ) (k : ℕ)
  (P : ℕ → ℝ × ℝ)
  (hP : ∀ i j : ℕ, i ≠ j → i < k → j < k →
    abs (Real.sin (angle A (P i) B) - Real.sin (angle A (P j) B)) ≤ 1 / 1992) :
  k ≥ 1993 :=
sorry

end minimum_k_points_l139_139802


namespace find_m_l139_139357

theorem find_m (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 2 → 1/2 * x^2 - 2 * x + m * x < 0) → m = 1 := 
by 
  assume h : ∀ x : ℝ, 0 < x ∧ x < 2 → 1/2 * x^2 - 2 * x + m * x < 0,
  sorry

end find_m_l139_139357


namespace find_constants_l139_139249

theorem find_constants : ∃ (A B C : ℝ), ∀ x : ℝ,
  x ≠ 0 ∧ x^2 ≠ 1 →
  (x^2 - 5 * x + 6) / (x^3 - x) = A / x + (B * x + C) / (x^2 - 1) :=
by
  use (-6 : ℝ)
  use (7 : ℝ)
  use (-5 : ℝ)
  intro x hx
  -- hx will be a conjunction so we extract both parts with hyphens
  cases hx with hx1 hx2
  have h : x^3 - x ≠ 0 := by
    intro h0
    rw [sub_eq_zero, ←mul_eq_zero, ←mul_eq_zero] at h0
    cases h0 with h0 h0
    exact hx2 h0
    rw [sub_eq_zero, ←pow_two_eq_pow_two_iff] at h0
    cases h0 with h0 h0
    exact hx1 h0.symm
    exact hx1 h0
  field_simp [h]
  ring

end find_constants_l139_139249


namespace length_of_PS_l139_139374

theorem length_of_PS (PQ QR RS : ℝ) (angleQ_right angleR_right : Bool) (Q_eq_R_right : angleQ_right = True ∧ angleR_right = True) 
  (hPQ : PQ = 6) (hQR : QR = 9) (hRS : RS = 12) : sqrt (QR ^ 2 + (RS - QR) ^ 2) = 3 * sqrt 10 := 
by 
  sorry

end length_of_PS_l139_139374


namespace collinear_external_bisectors_l139_139535

theorem collinear_external_bisectors 
    (ABC : Triangle)
    (A1 B1 C1 : Point)
    (hA : IsExternalAngleBisector ABC.A ABC.B ABC.C A1)
    (hB : IsExternalAngleBisector ABC.B ABC.C ABC.A B1)
    (hC : IsExternalAngleBisector ABC.C ABC.A ABC.B C1)
    (hA1_on_BC : LiesOn A1 (LineThrough ABC.B ABC.C))
    (hB1_on_CA : LiesOn B1 (LineThrough ABC.C ABC.A))
    (hC1_on_AB : LiesOn C1 (LineThrough ABC.A ABC.B)) :
    Collinear A1 B1 C1 := 
sorry

end collinear_external_bisectors_l139_139535


namespace value_of_f_at_cos_100_l139_139278

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * (Real.sin x)^3 + b * Real.cbrt x * (Real.cos x)^3 + 4

theorem value_of_f_at_cos_100 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : f a b (Real.sin (10 * Real.pi / 180)) = 5) :
  f a b (Real.cos (100 * Real.pi / 180)) = 3 := 
sorry

end value_of_f_at_cos_100_l139_139278


namespace radius_of_base_is_six_l139_139140

-- Define the sector radius and central angle
def sector_radius : ℝ := 9
def central_angle_degrees : ℝ := 240

-- Convert the central angle from degrees to radians
def central_angle_radians : ℝ := (240 / 360) * 2 * Real.pi

-- Define the arc length of the sector
def arc_length_of_sector : ℝ := central_angle_radians * sector_radius

-- The circumference of the base of the cone
def circumference_of_base (r : ℝ) : ℝ := 2 * Real.pi * r

-- The proof problem statement: Prove that the radius of the base is 6 cm
theorem radius_of_base_is_six : ∃ r : ℝ, circumference_of_base r = arc_length_of_sector ∧ r = 6 := by
  sorry

end radius_of_base_is_six_l139_139140


namespace clea_ride_escalator_time_l139_139756

def clea_time_not_walking (x k y : ℝ) : Prop :=
  60 * x = y ∧ 24 * (x + k) = y ∧ 1.5 * x = k ∧ 40 = y / k

theorem clea_ride_escalator_time :
  ∀ (x y k : ℝ), 60 * x = y → 24 * (x + k) = y → (1.5 * x = k) → 40 = y / k :=
by
  intros x y k H1 H2 H3
  sorry

end clea_ride_escalator_time_l139_139756


namespace find_A_l139_139544

theorem find_A :
  ∃ A B : ℕ, A < 10 ∧ B < 10 ∧ 5 * 100 + A * 10 + 8 - (B * 100 + 1 * 10 + 4) = 364 ∧ A = 7 :=
by
  sorry

end find_A_l139_139544


namespace num_solutions_eq_two_l139_139632

-- Predicate to check if a value of x is a solution to the given equations
def is_solution (x : ℤ) : Prop :=
  (x^2 - x = 5x - 5 ∨ x^2 - x = 16 - (5x - 5)) ∧ (x^2 - x ≤ 16) ∧ (5x - 5 ≤ 16)

-- Counting the number of integer solutions that satisfy the given conditions
def count_solutions : ℕ :=
  (finset.filter is_solution (finset.range 17)).card

-- Statement of the problem
theorem num_solutions_eq_two : count_solutions = 2 :=
by
  sorry

end num_solutions_eq_two_l139_139632


namespace find_x_l139_139171

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem find_x (x : ℝ) (hx : x > 0) :
  distance (1, 3) (x, -4) = 15 → x = 1 + Real.sqrt 176 :=
by
  sorry

end find_x_l139_139171


namespace range_of_k_l139_139305

def f (k x : ℝ) : ℝ :=
  if x ≥ 2 then (k - 1) * x^2 - 3 * (k - 1) * x + (13 * k - 9) / 4
  else (1 / 2) ^ x - 1

theorem range_of_k (k : ℝ) :
  (∀ n : ℕ+, f k (n + 1) < f k n) → k < -1 / 5 :=
by
  sorry

end range_of_k_l139_139305


namespace race_time_A_l139_139370

noncomputable def time_for_A_to_cover_distance (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) : ℝ :=
  let speed_of_B := distance / time_of_B
  let time_for_B_to_cover_remaining := remaining_distance_for_B / speed_of_B
  time_for_B_to_cover_remaining

theorem race_time_A (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) :
  distance = 100 ∧ time_of_B = 25 ∧ remaining_distance_for_B = distance - 20 →
  time_for_A_to_cover_distance distance time_of_B remaining_distance_for_B = 20 :=
by
  intros h
  rcases h with ⟨h_distance, h_time_of_B, h_remaining_distance_for_B⟩
  rw [h_distance, h_time_of_B, h_remaining_distance_for_B]
  sorry

end race_time_A_l139_139370


namespace max_f_on_interval_l139_139982

def f (x : ℝ) : ℝ := (6 * x) / (1 + x^2)

theorem max_f_on_interval : 
  ∃ x ∈ set.Icc (0 : ℝ) (3 : ℝ), ∀ y ∈ set.Icc (0 : ℝ) (3 : ℝ), f y ≤ f x := 
by
  sorry

end max_f_on_interval_l139_139982


namespace nina_math_homework_l139_139016

theorem nina_math_homework (x : ℕ) :
  let ruby_math := 6
  let ruby_read := 2
  let nina_math := x * ruby_math
  let nina_read := 8 * ruby_read
  let nina_total := nina_math + nina_read
  nina_total = 48 → x = 5 :=
by
  intros
  sorry

end nina_math_homework_l139_139016


namespace find_blue_sea_glass_pieces_l139_139940

-- Define all required conditions and the proof problem.
theorem find_blue_sea_glass_pieces (B : ℕ) : 
  let BlancheRed := 3
  let RoseRed := 9
  let DorothyRed := 2 * (BlancheRed + RoseRed)
  let DorothyBlue := 3 * B
  let DorothyTotal := 57
  DorothyTotal = DorothyRed + DorothyBlue → B = 11 :=
by {
  sorry
}

end find_blue_sea_glass_pieces_l139_139940


namespace artworks_per_student_in_first_half_l139_139791

theorem artworks_per_student_in_first_half (x : ℕ) (h1 : 10 = 10) (h2 : 20 = 20) (h3 : 5 * x + 5 * 4 = 35) : x = 3 := by
  sorry

end artworks_per_student_in_first_half_l139_139791


namespace ratio_upstream_downstream_l139_139173

def speed_man_still_water := 3  -- Speed of the man in still water (V_m)
def speed_stream := 1            -- Speed of the stream (V_s)
def distance := Sorry          -- Distance doesn't have a fixed value here, it is assumed as D

def upstream_speed := speed_man_still_water - speed_stream  -- V_u = V_m - V_s
def downstream_speed := speed_man_still_water + speed_stream  -- V_d = V_m + V_s

def time_upstream := distance / upstream_speed  -- T_u = D / V_u
def time_downstream := distance / downstream_speed  -- T_d = D / V_d

theorem ratio_upstream_downstream : (time_upstream / time_downstream) = 2 :=
by sorry

end ratio_upstream_downstream_l139_139173


namespace max_area_of_triangle_l139_139752

theorem max_area_of_triangle 
  (A B C : ℝ) (a b c : ℝ) (h_a : a = 2) (h_sin : sin B = sqrt 3 * sin C) : 
  ∃ S, S = sqrt 3 ∧ 
  ∀ x y z : ℝ, (x = a ∧ y = b ∧ z = c) → 
  (1 / 2) * x * y * sin C ≤ S :=
sorry

end max_area_of_triangle_l139_139752


namespace circle_properties_l139_139677

noncomputable def circle_radius {x y r : ℝ} : Prop :=
  x^2 + y^2 = r^2 ∧ r > 0 ∧ (1^2 + 2^2) = r^2

noncomputable def line_exists {x y r : ℝ} (A B M : ℝ × ℝ) : Prop := 
  x^2 + y^2 = r^2 ∧ r = 2 ∧
  (∃ l : ℝ → ℝ, ∀ x, l (-1) = 1 ∧ (∃ A B : ℝ × ℝ, A ≠ B ∧ 
  l (-1, 1) = (x - y + 2 = 0) ∧
  (l ((fst A), 1) = 2 ∧ l (fst B, 1) = 2) ∧
  OM = OA + OB))

theorem circle_properties :
  ∃ r : ℝ, circle_radius ∧ r = 2 ∧ (∀ l, line_exists l → (l (-1) = 1) ∧ 
  ((l ((-1, 1)) = (x - y + 2 = 0)))) :=
begin
  sorry
end

end circle_properties_l139_139677


namespace area_of_triangle_l139_139889

variable (α β γ l m n : ℝ)
variable (hα : α > 0) (hβ : β > 0) (hγ : γ > 0)
variable (hα_plus_β_plus_γ : α + β + γ = π)

theorem area_of_triangle
  (P : Type)
  (h_acute : α + β + γ = π)
  (h_perpend_AB : ⊥ P AB = l)
  (h_perpend_BC : ⊥ P BC = m)
  (h_perpend_CA : ⊥ P CA = n) :
  area ABC = (l * sin γ + m * sin α + n * sin β)^2 / (2 * sin α * sin β * sin γ) :=
sorry

end area_of_triangle_l139_139889


namespace B_is_pi_over_3_range_a_plus_b_over_c_correct_l139_139771

noncomputable def B_value (a b c A C : ℝ) (h₁ : 0 < A) (h₂ : A < π) (h₃ : 0 < C) (h₄ : C < π) 
  (h_cond : sqrt 3 * a * cos ((A + C)/2) = b * sin A) : ℝ :=
π / 3

theorem B_is_pi_over_3 (a b c A C : ℝ) (h₁ : 0 < A) (h₂ : A < π) (h₃ : 0 < C) (h₄ : C < π) 
  (h_cond : sqrt 3 * a * cos ((A + C)/2) = b * sin A) : B_value a b c A C h₁ h₂ h₃ h₄ h_cond = π / 3 := 
sorry

noncomputable def range_a_plus_b_over_c (a b c A C : ℝ) 
  (h₁ : 0 < A) (h₂ : A < π / 2) (h₃ : 0 < C) (h₄ : C < π / 2)
  (h_cond : sqrt 3 * a * cos ((A + C)/2) = b * sin A) : Set ℝ :=
{ x | (1 + sqrt 3)/2 < x ∧ x < 2 + sqrt 3 }

theorem range_a_plus_b_over_c_correct (a b c A C : ℝ) 
  (h₁ : 0 < A) (h₂ : A < π / 2) (h₃ : 0 < C) (h₄ : C < π / 2)
  (h_cond : sqrt 3 * a * cos ((A + C)/2) = b * sin A) : 
  ∀ x, x ∈ range_a_plus_b_over_c a b c A C h₁ h₂ h₃ h₄ h_cond ↔ (1 + sqrt 3)/2 < x ∧ x < 2 + sqrt 3 :=
sorry

end B_is_pi_over_3_range_a_plus_b_over_c_correct_l139_139771


namespace video_views_l139_139929

theorem video_views (initial_views : ℕ) (increase_factor : ℕ) (additional_views : ℕ) :
  initial_views = 4000 →
  increase_factor = 10 →
  additional_views = 50000 →
  let views_after_4_days := initial_views + increase_factor * initial_views in
  let total_views := views_after_4_days + additional_views in
  total_views = 94000 :=
by
  intros h_initial_views h_increase_factor h_additional_views
  have views_after_4_days_def : views_after_4_days = initial_views + increase_factor * initial_views
  have total_views_def : total_views = views_after_4_days + additional_views
  rw [h_initial_views, h_increase_factor, h_additional_views]
  rw [views_after_4_days_def, total_views_def]
  sorry

end video_views_l139_139929


namespace white_wins_with_perfect_play_l139_139529

theorem white_wins_with_perfect_play :
  ∀ (board : ℕ × ℕ) 
    (king_pos : (ℕ × ℕ) × (ℕ × ℕ)), 
    king_pos = ((1, 1), (8, 8)) → 
    (∀ (moves : ℕ → ((ℕ × ℕ) × (ℕ × ℕ))),
      (∀ n, ∃ m, moves n = (m, m) ∨ moves n = (m, m) ∧ abs (fst m - fst (snd m)) ≤ 1 ∧ abs (snd m - snd (snd m)) ≤ 1) → 
      (∃ n, (fst (moves n).fst = 8 ∨ snd (moves n).fst = 8) ∧ 
             (fst (moves n).snd = 1 ∨ snd (moves n).snd = 1))) → 
  (∃ n, (fst ((1, 1)) = 8 ∨ snd ((1, 1)) = 8)) :=
begin
  sorry
end

end white_wins_with_perfect_play_l139_139529


namespace angle_bisector_iff_eq_sides_l139_139763

-- Define the triangle ABC and its points
variables {A B C O E F : Type*}
variables [InTriangle ABC] [OnCircle K (triangle ABC)]
variables {E_on_AO : OnLine E A O}
variables {F_on_BO : OnLine F B O}
variables {CE_eq_CF : CE = CF}
variables {acute_triangle : AcuteTriangle ABC}
variables {circumscribed_circle : CircumscribedCircle K ABC}

-- Required goal
theorem angle_bisector_iff_eq_sides 
  (h_acute: acute_triangle)
  (h_circumscribed: circumscribed_circle)
  (h_point_O: InTriangle O ABC)
  (h_CE_eq_CF: CE_eq_CF)
  (h_E_on_AO: E_on_AO)
  (h_F_on_BO: F_on_BO) :
  (OnAngleBisector O ∠ ACB ↔ AC = BC) :=
sorry

end angle_bisector_iff_eq_sides_l139_139763


namespace integer_part_of_A_l139_139416

theorem integer_part_of_A :
  let num := 21 * 62 + 22 * 63 + 23 * 64 + 24 * 65 + 25 * 66
  let denom := 21 * 61 + 22 * 62 + 23 * 63 + 24 * 64 + 25 * 65
  let A := (num / denom : ℚ) * 199
  ⌊A⌋ = 202 :=
by
  let num := 21 * 62 + 22 * 63 + 23 * 64 + 24 * 65 + 25 * 66
  let denom := 21 * 61 + 22 * 62 + 23 * 63 + 24 * 64 + 25 * 65
  let A := (num / denom : ℚ) * 199
  have h : 202 < A := sorry
  have h' : A < 203 := sorry
  exact nat.floor_eq_iff.mpr ⟨h, h'⟩

end integer_part_of_A_l139_139416


namespace intersects_x_axis_vertex_coordinates_l139_139323

-- Definition of the quadratic function and conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - a * x - 2 * a^2

-- Condition: a ≠ 0
axiom a_nonzero (a : ℝ) : a ≠ 0

-- Statement for the first part of the problem
theorem intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x₁ x₂ : ℝ, quadratic_function a x₁ = 0 ∧ quadratic_function a x₂ = 0 ∧ x₁ * x₂ < 0 :=
by 
  sorry

-- Statement for the second part of the problem
theorem vertex_coordinates (a : ℝ) (h : a ≠ 0) (hy_intercept : quadratic_function a 0 = -2) :
  ∃ x_vertex : ℝ, quadratic_function a x_vertex = (if a = 1 then (1/2)^2 - 9/4 else (1/2)^2 - 9/4) :=
by 
  sorry


end intersects_x_axis_vertex_coordinates_l139_139323


namespace collinear_intersection_points_of_hexagon_inscribed_in_conic_l139_139767

theorem collinear_intersection_points_of_hexagon_inscribed_in_conic
  (A B C D E F M N P : Point)
  (conic : ConicSection)
  (h_inscribed : InscribedInHexagon A B C D E F conic)
  (h_intersect_AB_DE : ∃ M, intersection_point A B D E = M)
  (h_intersect_BC_EF : ∃ N, intersection_point B C E F = N)
  (h_intersect_CD_FA : ∃ P, intersection_point C D F A = P) :
  Collinear M N P := 
sorry

end collinear_intersection_points_of_hexagon_inscribed_in_conic_l139_139767


namespace value_of_expression_l139_139950

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + real.sqrt 3)

theorem value_of_expression :
  real.sqrt 3 * (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6) = 6 :=
by
  sorry

end value_of_expression_l139_139950


namespace problem_statement_l139_139197

noncomputable def count_propositions_and_true_statements 
  (statements : List String)
  (is_proposition : String → Bool)
  (is_true_proposition : String → Bool) 
  : Nat × Nat :=
  let props := statements.filter is_proposition
  let true_props := props.filter is_true_proposition
  (props.length, true_props.length)

theorem problem_statement : 
  (count_propositions_and_true_statements 
     ["Isn't an equilateral triangle an isosceles triangle?",
      "Are two lines perpendicular to the same line necessarily parallel?",
      "A number is either positive or negative",
      "What a beautiful coastal city Zhuhai is!",
      "If x + y is a rational number, then x and y are also rational numbers",
      "Construct △ABC ∼ △A₁B₁C₁"]
     (fun s => 
        s = "A number is either positive or negative" ∨ 
        s = "If x + y is a rational number, then x and y are also rational numbers")
     (fun s => false))
  = (2, 0) :=
by
  sorry

end problem_statement_l139_139197


namespace remainder_3_pow_88_add_5_mod_7_l139_139520

-- Define the cyclic property of powers of 3 modulo 7
lemma power_cycle_mod (n : ℕ) : (3^n % 7) = [3, 2, 6, 4, 5, 1].nth! (n % 6) :=
by sorry

-- The statement of the problem
theorem remainder_3_pow_88_add_5_mod_7 : (3^88 + 5) % 7 = 2 :=
by
  -- Use the cyclic property and the fact that 3^88 % 7 is the same as 3^(88 % 6) % 7
  have h : 3^88 % 7 = 3^4 % 7,
  { exact power_cycle_mod 88, },
  rw h,
  -- Calculate 3^4 % 7 and finish the proof
  norm_num,
  -- Final computation to check the remainder
  sorry

end remainder_3_pow_88_add_5_mod_7_l139_139520


namespace find_x_for_power_function_l139_139894

theorem find_x_for_power_function :
  (∃ a : ℝ, (2 : ℝ) ^ a = 8) →
  (∃ x : ℝ, (x : ℝ) ^ 3 = 27) :=
by
  intros h
  rcases h with ⟨a, ha⟩
  have a_eq : a = 3 := sorry  -- Solve 2^a = 8 to find a = 3
  refine ⟨3, _⟩
  norm_num
  sorry -- Proof simplification and verification

end find_x_for_power_function_l139_139894


namespace convex_hexagon_largest_angle_l139_139482

theorem convex_hexagon_largest_angle 
  (x : ℝ)                                 -- Denote the measure of the third smallest angle as x.
  (angles : Fin 6 → ℝ)                     -- Define the angles as a function from Fin 6 to ℝ.
  (h1 : ∀ i : Fin 6, angles i = x + (i : ℝ) - 3)  -- The six angles in increasing order.
  (h2 : 0 < x - 3 ∧ x - 3 < 180)           -- Convex condition: each angle is between 0 and 180.
  (h3 : angles ⟨0⟩ + angles ⟨1⟩ + angles ⟨2⟩ + angles ⟨3⟩ + angles ⟨4⟩ + angles ⟨5⟩ = 720) -- Sum of interior angles of a hexagon.
  : (∃ a, a = angles ⟨5⟩ ∧ a = 122.5) :=   -- Prove the largest angle in this arrangement is 122.5.
sorry

end convex_hexagon_largest_angle_l139_139482


namespace c_linear_comb_l139_139726

structure Vector2D where
  x : ℝ
  y : ℝ

def a : Vector2D := ⟨1, 1⟩
def b : Vector2D := ⟨1, -1⟩
def c : Vector2D := ⟨-1, -2⟩

theorem c_linear_comb:
  c = let λ : ℝ := -3/2
      let μ : ℝ := 1/2
      ⟨λ * a.x + μ * b.x, λ * a.y + μ * b.y⟩ := by
  sorry

end c_linear_comb_l139_139726


namespace avg_salary_increase_l139_139452

theorem avg_salary_increase (A1 : ℝ) (M : ℝ) (n : ℕ) (N : ℕ) 
  (h1 : n = 20) (h2 : A1 = 1500) (h3 : M = 4650) (h4 : N = n + 1) :
  (20 * A1 + M) / N - A1 = 150 :=
by
  -- proof goes here
  sorry

end avg_salary_increase_l139_139452


namespace express_y_in_terms_of_x_l139_139969

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 1) : y = 1 - 5 * x :=
by
  sorry

end express_y_in_terms_of_x_l139_139969


namespace total_views_l139_139932

def first_day_views : ℕ := 4000
def views_after_4_days : ℕ := 40000 + first_day_views
def views_after_6_days : ℕ := views_after_4_days + 50000

theorem total_views : views_after_6_days = 94000 := by
  have h1 : first_day_views = 4000 := rfl
  have h2 : views_after_4_days = 40000 + first_day_views := rfl
  have h3 : views_after_6_days = views_after_4_days + 50000 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end total_views_l139_139932


namespace projections_on_circle_l139_139487

-- Let A, B, C, D be the vertices of the triangular pyramid ABCD
variables {A B C D K L M N : Type}
variables (alpha : Type) (A B C D : alpha) (K L M N : ( α → α ) ) 

-- Assume the plane α intersects the edges AB, BC, CD, and DA at points K, L, M, and N respectively
axiom plane_alpha_intersections : 
  ( ∃ ( alpha : Type ) , ∀ ( A B C D : alpha ) ( K L M N : alpha ) , (intersects alpha A B C D K L M N))

-- Assume the dihedral angles ∠(KLA, KLM), ∠(LMB, LMN), ∠(MNC, MNK), and ∠(NKD, NKL) are equal
axiom dihedral_angles_equal : 
  ( ∀ ( K L A K L M L M B L M N M N C M N K N K D N K L : Type) , (equal_angles K L A K L M L M B L M N M N C M N K N K D N K L))

-- Let the projections of the vertices A, B, C, and D onto the plane α be A', B', C', and D' respectively
def projections (alpha : Type) (A B C D : alpha) : Type := 
  (project_alpha alpha A B C D)

-- Theorem: Prove that the projections of the vertices A, B, C, and D onto the plane α lie on a single circle
theorem projections_on_circle (alpha : Type) (A B C D : alpha) : 
  (lie_on_single_circle alpha A B C D K L M N) :=
sorry

end projections_on_circle_l139_139487


namespace math_problem_l139_139347

theorem math_problem (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) := 
by
  sorry

end math_problem_l139_139347


namespace distance_to_other_focus_l139_139355

variable (P F1 F2 : Type)
variable (d : P → F1 → ℝ) -- defining the distance function between point P and focus F1
variable (d' : P → F2 → ℝ) -- defining the distance function between point P and focus F2

def ellipse_property (point : P) (focus1 : F1) (focus2 : F2) : Prop :=
  d point focus1 + d' point focus2 = 20 -- condition of the ellipse

def given_distance (point : P) (focus1 : F1) : Prop :=
  d point focus1 = 6 -- given distance

def desired_distance (point : P) (focus2 : F2) : Prop :=
  d' point focus2 = 14 -- desired distance

theorem distance_to_other_focus (point : P) (focus1 : F1) (focus2 : F2)
  (h1 : ellipse_property point focus1 focus2)
  (h2 : given_distance point focus1) :
  desired_distance point focus2 :=
begin
  sorry
end

end distance_to_other_focus_l139_139355


namespace dolls_collection_l139_139626

-- Definitions and conditions
def initial_dolls_increase := 5
def initial_increase_percentage := 0.20
def updated_increase_percentage := 0.10

-- The theorem we want to prove
theorem dolls_collection (X Z : ℕ) 
  (h1 : initial_dolls_increase = 5)
  (h2 : initial_increase_percentage * X = initial_dolls_increase)
  (h3 : updated_increase_percentage * (X + initial_dolls_increase) + (X + initial_dolls_increase) = Z) :
  X = 25 ∧ Z = 33 := 
by
  sorry

end dolls_collection_l139_139626


namespace minimum_sum_am_gm_l139_139225

theorem minimum_sum_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ (1 / 2) :=
sorry

end minimum_sum_am_gm_l139_139225


namespace minimize_distance_at_P_i_l139_139814

-- Definitions and conditions
def evenly_spaced_points (n : ℕ) (r : ℝ) : list ℝ :=
list.map (λ k, 2 * π * k / n) (list.range n)

def direct_circular_distance (r θ : ℝ) : ℝ :=
r * θ

-- Lean 4 statement
theorem minimize_distance_at_P_i
  (n : ℕ) (r : ℝ) (P : ℝ) (h1 : n = 7) (h2 : (0 < P) ∧ (P < 2 * π)) :
  let points := evenly_spaced_points n r in
  ∃ i, P = points.nth_le i sorry :=
begin
  sorry
end

end minimize_distance_at_P_i_l139_139814


namespace relationship_among_a_b_c_l139_139298

noncomputable def f : ℝ → ℝ := sorry

lemma even_function (x : ℝ) : f x = f (-x) := sorry

lemma decreasing_on_negative (x y : ℝ) (h₁ : x < y) (h₂ : y ≤ 0) : f x > f y := sorry

def a : ℝ := f (2 ^ 0.3)
def b : ℝ := f (Real.logb (1/2) 4)
def c : ℝ := f (Real.logb 2 5)

theorem relationship_among_a_b_c : c > b ∧ b > a :=
by
  sorry

end relationship_among_a_b_c_l139_139298


namespace smallest_value_of_Q_l139_139951

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

noncomputable def A := Q (-1)
noncomputable def B := Q (0)
noncomputable def C := (2 : ℝ)^2
def D := 1 - 2 + 3 - 4 + 5
def E := 2 -- assuming all zeros are real

theorem smallest_value_of_Q :
  min (min (min (min A B) C) D) E = 2 :=
by sorry

end smallest_value_of_Q_l139_139951


namespace perp_tangent_slope_at_pi_over_2_l139_139679

def f (x : ℝ) : ℝ := x * sin x + 1

theorem perp_tangent_slope_at_pi_over_2 {a : ℝ} (h : f' (π / 2) = 2 / a) : a = 2 :=
by sorry

end perp_tangent_slope_at_pi_over_2_l139_139679


namespace outlet_two_rate_l139_139918

/-- Definitions and conditions for the problem -/
def tank_volume_feet : ℝ := 20
def inlet_rate_cubic_inches_per_min : ℝ := 5
def outlet_one_rate_cubic_inches_per_min : ℝ := 9
def empty_time_minutes : ℝ := 2880
def cubic_feet_to_cubic_inches : ℝ := 1728
def tank_volume_cubic_inches := tank_volume_feet * cubic_feet_to_cubic_inches

/-- Statement to prove the rate of the other outlet pipe -/
theorem outlet_two_rate (x : ℝ) :
  tank_volume_cubic_inches / empty_time_minutes = outlet_one_rate_cubic_inches_per_min + x - inlet_rate_cubic_inches_per_min → 
  x = 8 :=
by
  sorry

end outlet_two_rate_l139_139918


namespace equivalent_function_transformation_l139_139869

-- Define the original function
def original_function (x : ℝ) : ℝ := sin (2 * x)

-- Define the translated function
def translated_function (x : ℝ) : ℝ := sin (2 * (x - π))

-- Define the enlarged ordinate function
def transformed_function (x : ℝ) : ℝ := 2 * translated_function x

-- Target function after transformations
def target_function (x : ℝ) : ℝ := 2 * sin (2 * x)

-- The theorem stating that the two functions are equivalent
theorem equivalent_function_transformation : ∀ x : ℝ, transformed_function x = target_function x :=
by
  -- Proof goes here (not required to write out the proof)
  sorry


end equivalent_function_transformation_l139_139869


namespace sec_pi_over_18_minus_3_sin_pi_over_9_l139_139209

open Real

-- Definition of the main trigonometric problem conditions
def sec (x : ℝ) : ℝ := 1 / cos x
def sin_dbl_angle (x : ℝ) : ℝ := 2 * sin (x / 2) * cos (x / 2)

lemma trig_identity (x : ℝ) : sin (2 * x) = 2 * sin x * cos x := by sorry

theorem sec_pi_over_18_minus_3_sin_pi_over_9 :
  sec (π / 18) - 3 * sin (π / 9) = (1 - 3 * real.sqrt 2) / 4 :=
by
  have h1 : sec (π / 18) = 1 / cos (π / 18) := by sorry
  have h2 : sin (π / 9) = 2 * sin (π / 18) * cos (π / 18) := by sorry
  have h3 : sin (2 * (π / 18)) = 2 * sin (π / 18) * cos (π / 18) := trig_identity (π / 18)
  sorry

end sec_pi_over_18_minus_3_sin_pi_over_9_l139_139209


namespace sunmi_above_average_in_two_subjects_l139_139040

theorem sunmi_above_average_in_two_subjects (Korean Math Science English : ℝ) (total_subjects : ℝ) 
  (hKorean : Korean = 75) (hMath : Math = 85) (hScience : Science = 90) (hEnglish : English = 65) 
  (hTotal_subjects : total_subjects = 4) :
  let total_score := Korean + Math + Science + English,
      average_score := total_score / total_subjects in
  (if Math > average_score then 1 else 0) + (if Science > average_score then 1 else 0) + 
  (if Korean > average_score then 1 else 0) + (if English > average_score then 1 else 0) = 2 :=
by sorry

end sunmi_above_average_in_two_subjects_l139_139040


namespace second_player_wins_l139_139102

open Nat

noncomputable def game_piles := List Nat -- a list representing piles with one nut each.

def relatively_prime (a b : Nat) : Prop :=
  gcd a b = 1

def valid_move (state : game_piles) (p1 p2 : Nat) : Prop :=
  p1 ∈ state ∧ p2 ∈ state ∧ relatively_prime p1 p2

def new_state (state : game_piles) (p1 p2 : Nat) : game_piles :=
  (state.erase p1).erase p2 ++ [(p1 + p2)] -- combine piles

theorem second_player_wins (N : Nat) (hN : N > 2) :
  ∃ strategy : (game_piles → game_piles → Prop), ∀ init_state, game_piles.length init_state = N → 
    strategy init_state (new_state init_state + 1 1) →
    (strategy ∷ second_player_wins N (N - 1) init_state ∧ valid_move init_state 
    sorry

end second_player_wins_l139_139102


namespace quadrilateral_PQRS_has_all_acute_angles_l139_139157

-- Conditions
variables {A B C D P Q R S : Type*}
variable [convex_quadrilateral : convex_quadrilateral A B C D]
variable [incircle_touches : incircle_touches A B C D P Q R S]
variable [angle_bisectors : ∀ t, t ∈ {A, B, C, D} → is_angle_bisector t P Q R S]

-- Statement to prove
theorem quadrilateral_PQRS_has_all_acute_angles 
  [isosceles_triangles : ∀ t ∈ {APQ, BQR, CRS, DSP}, is_isosceles t] : 
  ∀ θ, θ ∈ angles_quadrilateral (P, Q, R, S) → θ < 90 := 
sorry

end quadrilateral_PQRS_has_all_acute_angles_l139_139157


namespace tetrahedron_property_l139_139422

variable (V : ℝ)
variable (S₁ S₂ S₃ S₄ : ℝ)
variable (H₁ H₂ H₃ H₄ : ℝ)
variable (k : ℝ)
variable h_V : S₁ * H₁ / 3 + S₂ * H₂ / 3 + S₃ * H₃ / 3 + S₄ * H₄ / 3 = V
variable h_k : S₁ / 1 = k ∧ S₂ / 2 = k ∧ S₃ / 3 = k ∧ S₄ / 4 = k

theorem tetrahedron_property :
  H₁ + 2 * H₂ + 3 * H₃ + 4 * H₄ = 3 * V / k :=
sorry

end tetrahedron_property_l139_139422


namespace hexagon_classroom_students_l139_139553

-- Define the number of sleeping students
def num_sleeping_students (students_detected : Nat → Nat) :=
  students_detected 2 + students_detected 3 + students_detected 6

-- Define the condition that the sum of snore-o-meter readings is 7
def snore_o_meter_sum (students_detected : Nat → Nat) :=
  2 * students_detected 2 + 3 * students_detected 3 + 6 * students_detected 6 = 7

-- Proof that the number of sleeping students is 3 given the conditions
theorem hexagon_classroom_students : 
  ∀ (students_detected : Nat → Nat), snore_o_meter_sum students_detected → num_sleeping_students students_detected = 3 :=
by
  intro students_detected h
  sorry

end hexagon_classroom_students_l139_139553


namespace angle_between_lines_l139_139824

theorem angle_between_lines : 
  let line1 := { x : ℝ | x - y + 5 = 0 } in
  let line2 := { x : ℝ | x = 3 } in
  abs (atan (1) - π/2) = π/4 :=
by
  sorry

end angle_between_lines_l139_139824


namespace points_within_rhombus_l139_139418

open EuclideanGeometry

noncomputable def circle_center (A1 A2 B1 B2 : Point) : Point := sorry
noncomputable def perpendicular_bisector (A P O: Point): Line := sorry
noncomputable def rhombus_region (l1 l2 l3 l4: Line): Set Point := sorry

theorem points_within_rhombus (A1 A2 B1 B2 P O: Point) :
  (dist A1 P > dist O P) ∧ (dist A2 P > dist O P) ∧ 
  (dist B1 P > dist O P) ∧ (dist B2 P > dist O P) →
  P ∈ rhombus_region (perpendicular_bisector A1 O O) 
                              (perpendicular_bisector A2 O O)
                              (perpendicular_bisector B1 O O)
                              (perpendicular_bisector B2 O O) :=
sorry

end points_within_rhombus_l139_139418


namespace player_1_guarantees_win_l139_139872

theorem player_1_guarantees_win :
  ∃ strategy : (3 × 100) → bool, 
  ∀ play : (3 × 100) → (1 × 2) → bool, 
  strategy (play (3 × 100) (1 × 2)) → 
  (∃ p1_wins : player → bool → Prop, p1_wins player1 true) :=
sorry

end player_1_guarantees_win_l139_139872


namespace number_of_gcd_values_l139_139879

theorem number_of_gcd_values (gcd lcm : ℕ → ℕ → ℕ) : 
  ∃ (a b : ℕ), gcd a b * lcm a b = 360 ∧ 
  {g : ℕ | ∃ (a b : ℕ), gcd a b = g ∧ gcd a b * lcm a b = 360}.size = 12 := 
by 
  sorry

end number_of_gcd_values_l139_139879


namespace range_of_m_l139_139085

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x - (m^2 - 2 * m + 4) * y - 6 > 0) ↔ (x, y) ≠ (-1, -1)) →
  -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l139_139085


namespace find_g5_l139_139073

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l139_139073


namespace area_of_triangle_QST_l139_139460

theorem area_of_triangle_QST
  (h_isosceles_right: ∃(P Q R: ℝ×ℝ), (P.1 = Q.1) ∧ (Q.2 = R.2) ∧ (Q.1 ≠ R.1) ∧ (P.1 ≠ R.1)
       ∧ ((triangleArea P Q R) = 18)
       ∧ (angle P Q R = π/2))
  (S T: ℝ×ℝ)
  (h_bisect: ∃(ray1 ray2: Ray), bisects ray1 P QR ∧ bisects ray2 Q PR 
       ∧ intersects ray1 ray2 (lineThru P R) S
       ∧ intersects ray2 ray1 (lineThru R P) T): 
  ∃(Q S T: ℝ×ℝ), (triangleArea Q S T) = 6 * (real.sqrt 2) := 
λ h_isosceles_right h_bisect,
  sorry

end area_of_triangle_QST_l139_139460


namespace h_even_l139_139777

-- Assume we have an odd function g
variable (g : ℝ → ℝ)
variable (hg_odd : ∀ (y : ℝ), g (-y) = - g y)

-- Define the function h based on g
def h (x : ℝ) : ℝ := |g (x^4)|

-- Prove that h is even
theorem h_even : ∀ (x : ℝ), h g x = h g (-x) :=
by
  intro x
  unfold h
  rw [pow_four]
  exact abs_eq_abs

-- Auxiliary definition for the fourth power
def pow_four (x : ℝ) : ∀ y, y = (x^4) := sorry

end h_even_l139_139777


namespace subcommittee_count_l139_139488

theorem subcommittee_count :
  let total_members := 12
  let teachers := 5
  let total_subcommittees := (Nat.choose total_members 4)
  let subcommittees_with_zero_teachers := (Nat.choose 7 4)
  let subcommittees_with_one_teacher := (Nat.choose teachers 1) * (Nat.choose 7 3)
  let subcommittees_with_fewer_than_two_teachers := subcommittees_with_zero_teachers + subcommittees_with_one_teacher
  let subcommittees_with_at_least_two_teachers := total_subcommittees - subcommittees_with_fewer_than_two_teachers
  subcommittees_with_at_least_two_teachers = 285 := by
  sorry

end subcommittee_count_l139_139488


namespace mary_oranges_l139_139798

theorem mary_oranges (total_oranges jason_oranges : ℕ) (h1 : total_oranges = 55) (h2 : jason_oranges = 41) : 
  total_oranges - jason_oranges = 14 :=
by {
  rw [h1, h2],
  norm_num,
}

end mary_oranges_l139_139798


namespace average_position_l139_139582

open BigOperators

def fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]

noncomputable def average : ℚ := (fractions.sum / fractions.length)

theorem average_position : ∃ n, list.sort (≤) (average :: fractions) = fractions.take n ++ [average] ++ fractions.drop n ∧ n = 4 := by
  sorry

end average_position_l139_139582


namespace ratio_of_sides_parallelogram_l139_139829

theorem ratio_of_sides_parallelogram (A B C D E F : Point)
    (h_parallelogram_ABCD : is_parallelogram A B C D)
    (h_diagonals_intersect_E : intersect_diagonals A B C D E)
    (h_bisectors_F : is_bisector_angle D A E F ∧ is_bisector_angle E B C F)
    (h_parallelogram_ECFD : is_parallelogram E C F D) 
  : ratio (distance B A) (distance D A) = Real.sqrt 3 := 
sorry

end ratio_of_sides_parallelogram_l139_139829


namespace part1_part2_l139_139329

noncomputable def a (k : ℝ) : ℝ × ℝ := (k - 1, 2)
def b : ℝ × ℝ := (2, -3)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (u : ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2^2)

-- Part (1)
theorem part1 (k : ℝ) (h : k = 1) : 
  real.cos (dot_product (a k) b / (magnitude (a k) * magnitude b)) = - (3 * real.sqrt 13) / 13 :=
by 
  have h1 : \overrightarrow{a} = (0, 2) := sorry
  sorry

-- Part (2)
theorem part2 (k : ℝ) (h : dot_product (2 * a k + b) (2 * a k - k * b) = 0) : k = 4 :=
by
  sorry

end part1_part2_l139_139329


namespace lean_proof_l139_139594

open Complex

noncomputable def problem_statement : Prop :=
  (1 / 2 ^ 2010 * ∑ n in Finset.range (1005 + 1), (-4) ^ n * Nat.choose 2010 (2 * n) 
   = -(5:ℝ) ^ 1005 * Real.cos (40 * Real.pi / 180) / 2 ^ 2010)

theorem lean_proof : problem_statement :=
  sorry

end lean_proof_l139_139594


namespace lives_per_player_l139_139502

theorem lives_per_player (initial_players joined_players total_lives : ℕ) (h1 : initial_players = 7) (h2 : joined_players = 2) (h3 : total_lives = 63) :
  let total_players := initial_players + joined_players in
  total_lives / total_players = 7 :=
by
  sorry

end lives_per_player_l139_139502


namespace no_functions_are_same_l139_139963

def f1 : ℝ → ℝ := λ x, if x = -3 then 0 else (x - 5)
def g1 : ℝ → ℝ := λ x, x - 5

def f2 : ℝ → ℝ := λ x, if x > 1 ∨ x < -1 then real.sqrt (x + 1) * real.sqrt (x - 1) else 0
def g2 : ℝ → ℝ := λ x, if x ≥ 1 ∨ x ≤ -1 then real.sqrt ((x + 1) * (x - 1)) else 0

def f3 : ℝ → ℝ := λ x, x
def g3 : ℝ → ℝ := λ x, real.sqrt (x ^ 2)

def f4 : ℝ → ℝ := λ x, x
def g4 : ℝ → ℝ := λ x, 3 * (x ^ 3)

def f5 : ℝ → ℝ := λ x, if 2 * x - 5 ≥ 0 then (real.sqrt (2 * x - 5)) ^ 2 else 0
def g5 : ℝ → ℝ := λ x, 2 * x - 5

theorem no_functions_are_same :
  ¬ (∀ (x : ℝ), f1 x = g1 x) ∧ 
  ¬ (∀ (x : ℝ), f2 x = g2 x) ∧ 
  ¬ (∀ (x : ℝ), f3 x = g3 x) ∧ 
  ¬ (∀ (x : ℝ), f4 x = g4 x) ∧ 
  ¬ (∀ (x : ℝ), f5 x = g5 x) :=
by sorry

end no_functions_are_same_l139_139963


namespace emily_initial_cards_l139_139624

theorem emily_initial_cards (x : ℤ) (h1 : x + 7 = 70) : x = 63 :=
by
  sorry

end emily_initial_cards_l139_139624


namespace correct_factorization_l139_139138

-- Define the polynomial expressions
def polyA (x : ℝ) := x^3 - x
def factorA1 (x : ℝ) := x * (x^2 - 1)
def factorA2 (x : ℝ) := x * (x + 1) * (x - 1)

def polyB (a : ℝ) := 4 * a^2 - 4 * a + 1
def factorB (a : ℝ) := 4 * a * (a - 1) + 1

def polyC (x y : ℝ) := x^2 + y^2
def factorC (x y : ℝ) := (x + y)^2

def polyD (x : ℝ) := -3 * x + 6 * x^2 - 3 * x^3
def factorD (x : ℝ) := -3 * x * (x - 1)^2

-- Statement of the correctness of factorization D
theorem correct_factorization : ∀ (x : ℝ), polyD x = factorD x :=
by
  intro x
  sorry

end correct_factorization_l139_139138


namespace find_g5_l139_139082

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l139_139082


namespace find_largest_number_l139_139650

theorem find_largest_number
  (a b c d e : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h₁ : a + b = 32)
  (h₂ : a + c = 36)
  (h₃ : b + c = 37)
  (h₄ : c + e = 48)
  (h₅ : d + e = 51) :
  (max a (max b (max c (max d e)))) = 27.5 :=
sorry

end find_largest_number_l139_139650


namespace probability_sum_eq_erika_age_l139_139625

def conditions : Prop :=
  let fair_coin := {15, 20}
  ∧ let die_faces := {1, 2, 3, 4, 5, 6}
  ∧ let erika_age := 16 in
  true -- conditions are contextual and inherently true in this context

theorem probability_sum_eq_erika_age : 
  ∀ (coin_flip : ℕ) (die_roll : ℕ), 
  coin_flip ∈ {15, 20} → 
  die_roll ∈ {1, 2, 3, 4, 5, 6} → 
  (coin_flip + die_roll = 16) → 
  (1 / 2) * (1 / 6) = 1 / 12 := 
by
  intros coin_flip die_roll coin_flip_in coin_flip_in die_roll_in sum_eq
  sorry -- proof goes here

end probability_sum_eq_erika_age_l139_139625


namespace number_of_true_propositions_l139_139785

theorem number_of_true_propositions (α β : ℝ) (h : α = β → cos α = cos β) :
  num_true_props α β h = 2 := sorry

def prop_p_true (α β : ℝ) (h : α = β → cos α = cos β) : Prop :=
  α = β → cos α = cos β

def contrapositive_p_true (α β : ℝ) (h : α = β → cos α = cos β) : Prop :=
  cos α ≠ cos β → α ≠ β

def converse_p_false (α β : ℝ) (h : α = β → cos α = cos β) : Prop :=
  ¬ (cos α = cos β → α = β)

def inverse_p_false (α β : ℝ) (h : α = β → cos α = cos β) : Prop :=
  ¬ (α ≠ β → cos α ≠ cos β)

def num_true_props (α β : ℝ) (h : α = β → cos α = cos β) : ℕ :=
  (if prop_p_true α β h then 1 else 0) +
  (if contrapositive_p_true α β h then 1 else 0) +
  (if converse_p_false α β h then 0 else 1) + -- false count as 0
  (if inverse_p_false α β h then 0 else 1)    -- false count as 0

end number_of_true_propositions_l139_139785


namespace least_positive_integer_l139_139126

theorem least_positive_integer (n : ℕ) :
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 5 ∧
  n % 7 = 2 ↔
  n = 83 :=
by
  sorry

end least_positive_integer_l139_139126


namespace find_a1_l139_139669

theorem find_a1 
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d = 1)
  (h2 : ∀ (n : ℕ), a n = a 1 + (n - 1) * d)
  (h3 : (a 5) ^ 2 = (a 3 * a 11)) : a 1 = -1 :=
by
  let a1 := a 1
  have eq_a5: a 5 = a1 + 4, from (by rw [h2 5, h1, mul_one, show 5 - 1 = 4 by norm_num])
  have eq_a3: a 3 = a1 + 2, from (by rw [h2 3, h1, mul_one, show 3 - 1 = 2 by norm_num])
  have eq_a11: a 11 = a1 + 10, from (by rw [h2 11, h1, mul_one, show 11 - 1 = 10 by norm_num])
  sorry

end find_a1_l139_139669


namespace polar_intersection_identity_l139_139377

noncomputable def parametric_eq_c1 (α : ℝ) : ℝ × ℝ :=
  (2 + Real.cos α, 2 + Real.sin α)

def line_eq_c2 (x : ℝ) : ℝ :=
  Real.sqrt 3 * x

theorem polar_intersection_identity :
  let C1 := λ α, (2 + Real.cos α, 2 + Real.sin α)
  let C2 := λ x, Real.sqrt 3 * x
  ∃ A B (ρ₁ ρ₂ : ℝ), 
    line_eq_c2 (2 + Real.cos A) = 2 + Real.sin A ∧
    line_eq_c2 (2 + Real.cos B) = 2 + Real.sin B ∧
    1 / (Real.sqrt ((2 + ρ₁ * Real.cos (π / 3) - 2)^2 + (2 + ρ₁ * Real.sin (π / 3) - 2)^2)) +
    1 / (Real.sqrt ((2 + ρ₂ * Real.cos (π / 3) - 2)^2 + (2 + ρ₂ * Real.sin (π / 3) - 2)^2)) =
    (2 * Real.sqrt 3 + 2) / 7 := by
  sorry

end polar_intersection_identity_l139_139377


namespace equivalent_tangent_sums_l139_139026

theorem equivalent_tangent_sums (n : ℕ) (A B : ℝ) 
  (α β : ℕ → ℝ) (i : ℕ)
  (h1 : 0 < i) (h2 : i ≤ n)
  (h3 : ∀ k, α (k + n) = -(π - α k)) 
  (h4 : ∀ k, β (k + n) = -(π - β k))
  : ∑ k in finset.range (n - i + 1), (tan (α (i + k)))^2 = 
    ∑ k in finset.range (n - i + 1), (tan (β (i + k)))^2 := 
begin
  sorry
end

end equivalent_tangent_sums_l139_139026


namespace group_membership_l139_139838

theorem group_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 11 = 6) (h3 : 100 ≤ n ∧ n ≤ 200) :
  n = 116 ∨ n = 193 :=
sorry

end group_membership_l139_139838


namespace geometric_sequence_common_ratio_l139_139456

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (a_mono : ∀ n, a n < a (n+1))
    (a2a5_eq_6 : a 2 * a 5 = 6)
    (a3a4_eq_5 : a 3 + a 4 = 5) 
    (q : ℝ) (hq : ∀ n, a n = a 1 * q ^ (n - 1)) :
    q = 3 / 2 :=
by
    sorry

end geometric_sequence_common_ratio_l139_139456


namespace fewer_hours_worked_l139_139215

noncomputable def total_earnings_summer := 6000
noncomputable def total_weeks_summer := 10
noncomputable def hours_per_week_summer := 50
noncomputable def total_earnings_school_year := 8000
noncomputable def total_weeks_school_year := 40

noncomputable def hourly_wage := total_earnings_summer / (hours_per_week_summer * total_weeks_summer)
noncomputable def total_hours_school_year := total_earnings_school_year / hourly_wage
noncomputable def hours_per_week_school_year := total_hours_school_year / total_weeks_school_year
noncomputable def fewer_hours_per_week := hours_per_week_summer - hours_per_week_school_year

theorem fewer_hours_worked :
  fewer_hours_per_week = hours_per_week_summer - (total_earnings_school_year / hourly_wage / total_weeks_school_year) := by
  sorry

end fewer_hours_worked_l139_139215


namespace area_APQD_l139_139924

-- Define the geometric constructions and conditions used in the problem
variables (A B C D E P Q : Type)
variables [RegularPentagon ABCDE] (star_area : star_area ABCDE)

-- Given conditions
def star_area (ABCDE : Type) := 1
def meet_at (A B P : Type) := 
def meet_at (B D Q : Type) := 

-- Statement
theorem area_APQD (ABCDE : Type) (star_area : star_area ABCDE) (P Q : Type) (meet_at_AC_BE : meet_at A C P) (meet_at_BD_CE : meet_at B D Q):
  (area APQD = 1 / 2) :=
sorry

end area_APQD_l139_139924


namespace slope_angle_FA_is_correct_l139_139693

noncomputable def parabola : Type := {p : ℝ × ℝ // p.2^2 = 3 * p.1}

def focus : ℝ × ℝ := (3/4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def slope_angle (p1 p2 : ℝ × ℝ) : ℝ := real.atan (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_angle_FA_is_correct (A : parabola) (hFA : distance focus A.val = 3) :
  slope_angle focus A.val = real.pi / 3 ∨ slope_angle focus A.val = 2 * real.pi / 3 :=
sorry

end slope_angle_FA_is_correct_l139_139693


namespace edges_can_be_colored_l139_139556

-- Definitions of the context
def unit_cube (n : ℕ) := {x : ℕ × ℕ × ℕ // x.1 < n ∧ x.2 < n ∧ x.2.snd < n}

-- Boundary definitions for neighboring cubes
def is_neighbor (a b : ℕ × ℕ × ℕ) : Prop :=
  (abs (a.1 - b.1) + abs (a.2 - b.2) + abs (a.2.snd - b.2.snd)) = 1

-- Definition of the broken line intersecting faces
def broken_line (n : ℕ) := list (ℕ × ℕ × ℕ)
def is_non_self_intersecting (line : broken_line n) : Prop :=
  ∀ (i j : ℕ), i ≠ j → line.nth i ≠ line.nth j

def is_closed (line : broken_line n) : Prop :=
  line.head = line.last

-- Coloring rule
inductive color | white | black

def edge_color (n : ℕ) (line : broken_line n) (edge : ℕ × ℕ × ℕ) : color :=
  if (count_intersections_with_edge line edge) % 2 = 0 then color.white else color.black

-- Definition of intersection count
def count_intersections_with_edge (line : broken_line n) (edge : ℕ × ℕ × ℕ) : ℕ :=
  sorry -- to be defined properly based on the problem context

-- Property to check marked faces
def marked_face_has_correct_colors (line : broken_line n) (edge : ℕ × ℕ × ℕ) : Prop :=
  sorry -- to ensure marked face has odd number of each color edges

-- Property to check unmarked faces
def unmarked_face_has_correct_colors (line : broken_line n) (edge : ℕ × ℕ × ℕ) : Prop :=
  sorry -- to ensure unmarked face has even number of each color edges

-- The main theorem
theorem edges_can_be_colored :
  ∀ (n : ℕ) (line : broken_line n),
    is_non_self_intersecting line →
    is_closed line →
    (∀ face, face ∈ marked_faces line → marked_face_has_correct_colors line face) ∧
    (∀ face, face ∈ unmarked_faces line → unmarked_face_has_correct_colors line face) :=
sorry

end edges_can_be_colored_l139_139556


namespace math_problem_l139_139413

noncomputable def a : ℂ := sorry

theorem math_problem :
  (a^2 - a + 1 = 0) →
  (a^10 + a^20 + a^30 ≠ -1) ∧
  (a^10 + a^20 + a^30 ≠ 0) ∧
  (a^10 + a^20 + a^30 ≠ 1) :=
by
  intro h
  have ha := by sorry
  have h₁ := by sorry
  have h₂ := by sorry
  have h₃ := by sorry
  exact ⟨h₁, h₂, h₃⟩

end math_problem_l139_139413


namespace speed_of_first_half_of_journey_l139_139192

theorem speed_of_first_half_of_journey:
  ∀ (total_distance time_taken distance_half second_half_speed: ℕ),
    total_distance = 672 →
    time_taken = 30 →
    distance_half = 336 →
    second_half_speed = 24 →
    let first_half_time := time_taken - (distance_half / second_half_speed) in
    let first_half_speed := distance_half / first_half_time in
    first_half_speed = 21 :=
by
  intros total_distance time_taken distance_half second_half_speed
  intros h_total_distance h_time_taken h_distance_half h_second_half_speed
  let first_half_time := time_taken - (distance_half / second_half_speed)
  let first_half_speed := distance_half / first_half_time
  have : first_half_speed = 21 := sorry
  exact this

end speed_of_first_half_of_journey_l139_139192


namespace find_ball_contact_height_l139_139253

theorem find_ball_contact_height :
  ∀ (h : ℕ), ((h ^ 2) + (7 ^ 2) = (53 ^ 2)) → h = 2 :=
by
  intro h
  assume (h_equation : (h ^ 2) + (7 ^ 2) = (53 ^ 2))
  have h_squared : h ^ 2 = (53 ^ 2) - (7 ^ 2), from sorry
  have h_squared_value : h ^ 2 = 2809 - 49, from sorry
  have h_squared_value : h ^ 2 = 2760, from sorry
  have h_value : h = Nat.sqrt 2760, from sorry
  have h_value : h = 2, from sorry
  exact h_value

end find_ball_contact_height_l139_139253


namespace min_value_when_a_eq_1_monotonicity_of_f_range_of_a_for_perpendicular_tangents_l139_139307

noncomputable def f (x : ℝ) (a : ℝ) := real.exp x - a * x - a
noncomputable def h (x : ℝ) (a : ℝ) := -f x a - (a + 1) * x + 2 * a
noncomputable def g (x : ℝ) (a : ℝ) := (x - 1) * a + 2 * real.cos x

theorem min_value_when_a_eq_1 :
  f 0 1 = 0 :=
sorry

theorem monotonicity_of_f (a : ℝ) :
  (∀ x, a ≤ 0 → real.exp x - a > 0 ∧ f x a = f 0 a) ∧ -- f is increasing when a ≤ 0
  (∀ x, a > 0 → 
    (x < real.log a → real.exp x - a < 0) ∧
    (x = real.log a → real.exp x - a = 0) ∧
    (x > real.log a → real.exp x - a > 0)) := 
sorry

theorem range_of_a_for_perpendicular_tangents :
  -1 ≤ a ∧ a ≤ 2 :=
sorry

end min_value_when_a_eq_1_monotonicity_of_f_range_of_a_for_perpendicular_tangents_l139_139307


namespace cube_root_closest_integer_l139_139123

noncomputable def closest_integer_to_cubicroot (n : ℕ) : ℕ :=
  if (Int.natAbs(Int.ofNat n - Int.ofNat (n^(1/3)))) >=
     (Int.natAbs(Int.ofNat n - Int.ofNat (n+1)^(1/3))) then n
  else n + 1

theorem cube_root_closest_integer :
  closest_integer_to_cubicroot (5^3 + 7^3) = 8 :=
by
  sorry

end cube_root_closest_integer_l139_139123


namespace domain_sqrt_3x_plus_2_domain_sqrt_x_plus_3_plus_1_over_x_plus_2_l139_139978

-- Define the predicates for the conditions

def condition1 (x : ℝ) := 3 * x + 2 ≥ 0
def condition2 (x : ℝ) := x + 3 ≥ 0
def condition3 (x : ℝ) := x + 2 ≠ 0

-- Prove the domain for the function f(x) = sqrt(3x + 2)
theorem domain_sqrt_3x_plus_2 :
  (∀ x, condition1 x ↔ (x ∈ set.Ici (-2/3))) :=
by sorry

-- Prove the domain for the function f(x) = sqrt(x + 3) + 1/(x + 2)
theorem domain_sqrt_x_plus_3_plus_1_over_x_plus_2 :
  (∀ x, (condition2 x ∧ condition3 x) ↔ (x ∈ (set.Ico (-3) (-2) ∪ set.Ioi (-2)))) :=
by sorry

end domain_sqrt_3x_plus_2_domain_sqrt_x_plus_3_plus_1_over_x_plus_2_l139_139978


namespace number_of_three_digit_numbers_with_5_and_7_l139_139712

def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def containsDigit (n : ℕ) (d : ℕ) : Prop := d ∈ (n.digits 10) 
def hasAtLeastOne5andOne7 (n : ℕ) : Prop := containsDigit n 5 ∧ containsDigit n 7
def totalThreeDigitNumbersWith5and7 : ℕ := 50

theorem number_of_three_digit_numbers_with_5_and_7 :
  ∃ n : ℕ, isThreeDigitNumber n ∧ hasAtLeastOne5andOne7 n → n = 50 := sorry

end number_of_three_digit_numbers_with_5_and_7_l139_139712


namespace distinct_collections_l139_139538

def vowels := ['A', 'A', 'A', 'I', 'E']
def consonants := ['T', 'T', 'T', 'M', 'M', 'C', 'N']

def indistinguishable (ch : Char) : (Char → Bool) :=
  λ c, match c, ch with
  | 'A', 'A' => true
  | 'T', 'T' => true
  | 'M', 'M' => true
  | _, _ => false

def count_vowel_combinations : ℕ :=
  -- 4 combinations: (0A 2), (1A 2), (2A 1), (3A 0)
  1 + 2 + 1
  -- Where 1 + 2 + 1 represents the ways to choose vowels

def count_consonant_combinations : ℕ :=
  -- 3 combinations: (2T 2M), (3T 1M), (3T 0M)
  1 + 2
  -- Where 1 + 2 represents the ways to choose consonants

theorem distinct_collections : count_vowel_combinations * count_consonant_combinations = 12 := 
  by sorry

end distinct_collections_l139_139538


namespace math_problem_l139_139124

theorem math_problem 
    : 12 * ((1/3) + (1/4) + (1/6))⁻¹ = 16 :=
by
  sorry

end math_problem_l139_139124


namespace car_speed_conversion_l139_139095

theorem car_speed_conversion :
  let speed_mps := 10 -- speed of the car in meters per second
  let conversion_factor := 3.6 -- conversion factor from m/s to km/h
  let speed_kmph := speed_mps * conversion_factor -- speed of the car in kilometers per hour
  speed_kmph = 36 := 
by
  sorry

end car_speed_conversion_l139_139095


namespace additional_money_needed_l139_139945

noncomputable def carlSavingsAtEndOfWeek (savings: ℤ) (weeks: ℤ) : ℤ :=
  match weeks with
  | 1 => savings + 25
  | 2 => savings + 25
  | 3 => savings + 25
  | 4 => savings + 25 + 50 - 15
  | 5 => let intermediate := savings + 20 
         intermediate + int.ofNat ((2 * intermediate.natAbs) / 100)
  | 6 => savings + 34 - 12
  | 7 => 235
  | _ => 0

def costOfCoatInUSD (original_cost_eur: ℚ) (discount_rate: ℚ) (tax_rate: ℚ) (exchange_rate: ℚ) : ℚ :=
  let discounted_price := original_cost_eur * (1 - discount_rate)
  let total_cost_eur := discounted_price * (1 + tax_rate)
  total_cost_eur * (1 / exchange_rate)

theorem additional_money_needed :
  let final_savings := carlSavingsAtEndOfWeek 0 7
  let coat_cost_usd := costOfCoatInUSD 220 (10 / 100) (7 / 100) (85 / 100)
  final_savings + 14.25 = coat_cost_usd := sorry

end additional_money_needed_l139_139945


namespace pages_with_same_units_digit_count_l139_139175

/-
  A notebook has 75 pages numbered 1 to 75. If the pages are renumbered in reverse order, from 75 to 1,
  determine how many pages have the new page number and the old page number sharing the same units digit.
-/

theorem pages_with_same_units_digit_count : 
  let reverse_number (n : ℕ) := 76 - n in
  (card {x | (1 ≤ x ∧ x ≤ 75) ∧ (x % 10 = (reverse_number x) % 10)}) = 15 :=
by
  sorry


end pages_with_same_units_digit_count_l139_139175


namespace sqrt_x_minus_1_meaningful_l139_139350

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 1)) → x ≥ 1 := by
  intros h
  cases h with y hy
  rw hy
  have := real.sqrt_nonneg (x - 1)
  sorry

end sqrt_x_minus_1_meaningful_l139_139350


namespace trigonometric_product_eq_l139_139530

open Real

theorem trigonometric_product_eq :
  3.420 * (sin (10 * pi / 180)) * (sin (20 * pi / 180)) * (sin (30 * pi / 180)) *
  (sin (40 * pi / 180)) * (sin (50 * pi / 180)) * (sin (60 * pi / 180)) *
  (sin (70 * pi / 180)) * (sin (80 * pi / 180)) = 3 / 256 := 
sorry

end trigonometric_product_eq_l139_139530


namespace common_divisors_count_l139_139959

-- Define the prime factorizations
def factorization_75 : List ℕ := [3, 5, 5]
def factorization_90 : List ℕ := [2, 3, 3, 5]

-- Define the set of divisors function
def divisors (n : ℕ) : Set ℤ :=
  {d : ℤ | d ≠ 0 ∧ (d ∣ n)}

-- Define the numbers 75 and 90
def n75 : ℕ := 75
def n90 : ℕ := 90

-- Statement of the problem
theorem common_divisors_count :
  (divisors n75 ∩ divisors n90).size = 8 := sorry

end common_divisors_count_l139_139959


namespace minimize_sum_of_product_eq_factorial_l139_139455

theorem minimize_sum_of_product_eq_factorial (p q r s : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (h : p * q * r * s = nat.factorial 12) : 
  p + q + r + s ≥ 736 :=
sorry

end minimize_sum_of_product_eq_factorial_l139_139455


namespace total_pieces_of_clothing_l139_139760

-- Define Kaleb's conditions
def pieces_in_one_load : ℕ := 19
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- The total pieces of clothing Kaleb has
theorem total_pieces_of_clothing : pieces_in_one_load + num_equal_loads * pieces_per_load = 39 :=
by
  sorry

end total_pieces_of_clothing_l139_139760


namespace place_mat_length_l139_139552

/--
A circular table has a radius of 5 units. Eight rectangular place mats are placed on the table. Each place mat has a width of 2 units and a length y.
They are positioned such that each mat has two corners on the edge of the table, and these two corners are endpoints of the same side of length y.
Moreover, the mats are arranged such that their inner corners each touch an inner corner of two adjacent mats forming an octagon.
Determine y.
-/
theorem place_mat_length
  (y : ℝ)
  (r : ℝ := 5)
  (width : ℝ := 2)
  (s := 5 * Real.sqrt (2 - Real.sqrt 2)) :
  r^2 = 1^2 + (y + s / 2 - 1)^2 →
  y = Real.sqrt (24 - 5 * Real.sqrt (2 - Real.sqrt 2)) - (5 * Real.sqrt (2 - Real.sqrt 2)) / 2 + 1 :=
sorry

end place_mat_length_l139_139552


namespace total_views_correct_l139_139926

-- Definitions based on the given conditions
def initial_views : ℕ := 4000
def views_increase := 10 * initial_views
def additional_views := 50000
def total_views_after_6_days := initial_views + views_increase + additional_views

-- The theorem we are going to state
theorem total_views_correct :
  total_views_after_6_days = 94000 :=
sorry

end total_views_correct_l139_139926


namespace largest_angle_in_consecutive_integer_hexagon_l139_139475

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l139_139475


namespace explicit_and_implicit_definitions_l139_139442

-- Definitions for explicit and implicit functions
def explicit_function (f : ℝ → ℝ) (x : ℝ) : ℝ := f(x)

def implicit_function (F : ℝ → ℝ → ℝ) (x y : ℝ) : Prop := F(x, y) = 0

-- Example of implicit function 2x - 3y - 1 = 0
def example_implicit1 (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

-- Example of implicit function pv = c
def example_implicit2 (p v c : ℝ) : Prop := p * v = c

-- Show that solving pv = c for p yields p = c / v
def solve_implicit2 (v c : ℝ) (h : v ≠ 0) : ℝ := c / v

theorem explicit_and_implicit_definitions :
    (∀ (f : ℝ → ℝ) (x : ℝ), explicit_function f x = f(x))
    ∧ (∀ (F : ℝ → ℝ → ℝ) (x y : ℝ), implicit_function F x y ↔ F(x, y) = 0)
    ∧ (∃ (x y : ℝ), example_implicit1 x y)
    ∧ (∃ (p v c : ℝ), example_implicit2 p v c)
    ∧ (∀ (v c : ℝ) (h : v ≠ 0), solve_implicit2 v c h = c / v) :=
by {
  split,
  {
    -- Definition of explicit function
    intros f x,
    exact rfl,
  },
  split,
  {
    -- Definition of implicit function
    intros F x y,
    exact iff.rfl,
  },
  split,
  {
    -- Provide example of implicit function: 2x - 3y - 1 = 0
    use [1, 1],
    exact rfl,
  },
  split,
  {
    -- Provide example of implicit function: pv = c
    use [1, 1, 1],
    exact rfl,
  },
  {
    -- Show that solving pv = c for p yields p = c / v
    intros v c h,
    exact rfl,
  },
}

-- Proof is currently skipped as specified
#check explicit_and_implicit_definitions -- Ensure the theorem is well-formed

end explicit_and_implicit_definitions_l139_139442


namespace min_unit_circles_to_cover_triangle_l139_139130

/-- Prove that the minimum number of unit circles required to completely cover a
    triangle with sides 2, 3, and 4 is 3. -/
theorem min_unit_circles_to_cover_triangle : ∀ (a b c : ℝ), 
  a = 2 → b = 3 → c = 4 → 
  ∃ n : ℕ, n = 3 ∧ 
    ∀ (cover : ℕ → set (ℝ × ℝ)), 
    (∀ i, ∃ (x y : ℝ), cover i = { p | dist p (x, y) < 1 }) →
    ∃ (covers_triangle : set (ℝ × ℝ) → Prop), 
    covers_triangle (⋃ i in finset.range n, cover i) ∧
    (∀ m < n, ¬ covers_triangle (⋃ i in finset.range m, cover i)) :=
begin
  sorry
end

end min_unit_circles_to_cover_triangle_l139_139130


namespace cosine_phase_shift_l139_139504

theorem cosine_phase_shift :
  ∀ (x : ℝ), cos (2 * x - π / 4) = cos (2 * (x - π / 8)) :=
by sorry

end cosine_phase_shift_l139_139504


namespace triangle_area_intersection_l1_l2_l139_139746

noncomputable def line_l1_polar (θ : ℝ) (ρ : ℝ) : Prop :=
  ρ * cos θ + 2 = 0

noncomputable def curve_C_polar (θ : ℝ) (ρ : ℝ) : Prop :=
  ρ = 4 * sin θ

theorem triangle_area (r : ℝ) (C_x C_y : ℝ) (M_x M_y N_x N_y : ℝ) : 
  (r = 2) → 
  (C_x = 0 ∧ C_y = 2) → 
  (M_x = 2 * cos (π / 4) ∧ M_y = 2 + 2 * sin (π / 4)) →
  (N_x = 2 * cos (π / 4) ∧ N_y = 2 + 2 * sin (π / 4)) →
  ∃ s : real, s = 2 :=
sorry

theorem intersection_l1_l2 (θ : ℝ) (ρ : ℝ) :
  (ρ * cos θ + 2 = 0) ∧ (θ = π / 4) → 
  (ρ, θ) = (-2 * sqrt 2, π / 4) :=
sorry

end triangle_area_intersection_l1_l2_l139_139746


namespace lucky_consecutive_pairs_l139_139176

noncomputable def sum_of_squares_of_digits (n : ℕ) : ℕ :=
-- This recursively computes the sum of the squares of the digits of the number n.
nat.rec_on n 0 (fun d r => ((d % 10) * (d % 10)) + sum_of_squares_of_digits (d / 10))

def is_lucky (n : ℕ) : Prop :=
∃ k : ℕ, nat.repeat sum_of_squares_of_digits k n = 1

theorem lucky_consecutive_pairs : ∀ (n : ℕ), is_lucky (211 * 10^(n+1) + 1) ∧ is_lucky (211 * 10^(n+1) + 2) := 
by
  sorry

end lucky_consecutive_pairs_l139_139176


namespace congruent_CDE1_CDE2_l139_139809

-- Definition that points A, B, C, D are collinear
def collinear (A B C D : Point) : Prop :=
  (A.coord.y = B.coord.y) ∧ (B.coord.y = C.coord.y) ∧ (C.coord.y = D.coord.y)

-- Definition that two triangles are congruent
def triangles_congruent (T1 T2 : Triangle) : Prop :=
  (T1.side1 = T2.side1) ∧ (T1.side2 = T2.side2) ∧ (T1.side3 = T2.side3) ∧
  (T1.angle1 = T2.angle1) ∧ (T1.angle2 = T2.angle2) ∧ (T1.angle3 = T2.angle3)

-- Points A, B, C, D and points E1, E2
variables {A B C D E1 E2 : Point}

-- Triangles ABE1 and ABE2
def triangle_ABE1 : Triangle := Triangle.mk A B E1
def triangle_ABE2 : Triangle := Triangle.mk A B E2

-- Triangles CDE1 and CDE2
def triangle_CDE1 : Triangle := Triangle.mk C D E1
def triangle_CDE2 : Triangle := Triangle.mk C D E2

-- Hypothesis: Points A, B, C, D are collinear and Triangles ABE1 and ABE2 are congruent
theorem congruent_CDE1_CDE2 (h_collinear : collinear A B C D)
  (h_congruent_ABE : triangles_congruent triangle_ABE1 triangle_ABE2) :
  triangles_congruent triangle_CDE1 triangle_CDE2 := by
  sorry

end congruent_CDE1_CDE2_l139_139809


namespace factorial_equation_solution_l139_139976

theorem factorial_equation_solution (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → a = 3 ∧ b = 3 ∧ c = 4 := by
  sorry

end factorial_equation_solution_l139_139976


namespace greatest_common_divisor_digital_reductions_l139_139956

/-- Given a two-digit number defined as 10A + B, 
where A is the tens digit and B is the units digit, 
the digital reduction is defined as (10A + B) - A - B, 
which simplifies to 9A. 
The task is to prove that the greatest common divisor 
of the digital reductions of all two-digit numbers is 9. -/
theorem greatest_common_divisor_digital_reductions :
  let digital_reduction (n : ℕ) (A : ℕ) (B : ℕ) := (10 * A + B) - A - B in
  let reductions := {9 * A | A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}} in
  ∀ A B, 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 → 
  A ∈ (finset.range 10 \ {0}) →
  B ∈ finset.range 10 →
  finset.gcd reductions = 9 :=
by
  sorry

end greatest_common_divisor_digital_reductions_l139_139956


namespace sum_abcd_eq_neg_46_div_3_l139_139412

theorem sum_abcd_eq_neg_46_div_3
  (a b c d : ℝ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 15) :
  a + b + c + d = -46 / 3 := 
by sorry

end sum_abcd_eq_neg_46_div_3_l139_139412


namespace compound_proposition_truth_l139_139622

-- Propositions
def p := ∃ α β : ℝ, sin (α + β) = sin α + sin β
def q (a : ℝ) := log a 2 + log 2 a ≥ 2

-- Conditions and proving the compound proposition
theorem compound_proposition_truth (h_p : p) (h_q : ∀ a : ℝ, a > 0 ∧ a ≠ 1 → ¬ q a) : (p ∨ ∀ a : ℝ, a > 0 ∧ a ≠ 1 → ¬ q a) :=
by {
  exact or.inl h_p
}

end compound_proposition_truth_l139_139622


namespace maximum_y_coordinate_l139_139692

variable (x y b : ℝ)

def hyperbola (x y b : ℝ) : Prop := (x^2) / 4 - (y^2) / b = 1

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def op_condition (x y b : ℝ) : Prop := (x^2 + y^2) = 4 + b

noncomputable def eccentricity (b : ℝ) : ℝ := (Real.sqrt (4 + b)) / 2

theorem maximum_y_coordinate (hb : b > 0) 
                            (h_ec : 1 < eccentricity b ∧ eccentricity b ≤ 2) 
                            (h_hyp : hyperbola x y b) 
                            (h_first : first_quadrant x y) 
                            (h_op : op_condition x y b) 
                            : y ≤ 3 :=
sorry

end maximum_y_coordinate_l139_139692


namespace even_three_digit_numbers_sum_tens_units_eq_nine_l139_139709

theorem even_three_digit_numbers_sum_tens_units_eq_nine :
  finset.card {n | ∃ (h t u : ℕ), 100 ≤ n ∧ n < 1000 ∧ (n = 100 * h + 10 * t + u) ∧ (u % 2 = 0) ∧ ((t + u = 9))}
  = 36 :=
by
  sorry

end even_three_digit_numbers_sum_tens_units_eq_nine_l139_139709


namespace percentage_of_500_is_125_l139_139984

theorem percentage_of_500_is_125 (part whole : ℕ) (h_part : part = 125) (h_whole : whole = 500) :
  (part / (whole : ℝ)) * 100 = 25 :=
by {
  rw [h_part, h_whole],
  norm_cast,
  simp,
  norm_num,
}

end percentage_of_500_is_125_l139_139984


namespace ratio_p_q_l139_139233

-- Definitions of probabilities p and q based on combinatorial choices and probabilities described.
noncomputable def p : ℚ :=
  (Nat.choose 6 1) * (Nat.choose 5 2) * (Nat.choose 24 2) * (Nat.choose 22 4) * (Nat.choose 18 4) * (Nat.choose 14 5) * (Nat.choose 9 5) * (Nat.choose 4 5) / (6 ^ 24)

noncomputable def q : ℚ :=
  (Nat.choose 6 2) * (Nat.choose 24 3) * (Nat.choose 21 3) * (Nat.choose 18 4) * (Nat.choose 14 4) * (Nat.choose 10 4) * (Nat.choose 6 4) / (6 ^ 24)

-- Lean statement to prove p / q = 6
theorem ratio_p_q : p / q = 6 := by
  sorry

end ratio_p_q_l139_139233


namespace gcd_polynomial_l139_139296

theorem gcd_polynomial (b : ℤ) (hb : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^3 + b^2 + 6 * b + 95) b = 95 :=
by
  sorry

end gcd_polynomial_l139_139296


namespace largest_angle_in_consecutive_integer_hexagon_l139_139479

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l139_139479


namespace solve_system_l139_139345

variable {x y z : ℝ}

theorem solve_system :
  (y + z = 16 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 13 - 4 * z) →
  2 * x + 2 * y + 2 * z = 11 / 3 :=
by
  intros h1 h2 h3
  -- proof skips, to be completed
  sorry

end solve_system_l139_139345


namespace bottles_remaining_l139_139813

theorem bottles_remaining 
  (initial_bottles : ℕ) 
  (first_break_bottles : ℕ) 
  (second_break_bottles : ℕ) 
  (third_break_bottles : ℕ) 
  (final_bottles_remaining : ℕ) :
  initial_bottles = 60 →
  first_break_bottles = 22 →
  second_break_bottles = 18 →
  third_break_bottles = 12 →
  final_bottles_remaining = initial_bottles - (first_break_bottles + second_break_bottles + third_break_bottles) →
  final_bottles_remaining = 8 :=
by
  intros h_init h_first h_second h_third h_eq
  rw [h_init, h_first, h_second, h_third] at h_eq
  exact h_eq

-- Proof is skipped with "sorry"
sorry

end bottles_remaining_l139_139813


namespace log_base_2_of_a_l139_139277

theorem log_base_2_of_a (a : ℝ) (h1 : a^(1/2) = 4) (h2 : 0 < a) : log 2 a = 4 :=
sorry

end log_base_2_of_a_l139_139277


namespace expected_participants_2003_l139_139384

theorem expected_participants_2003 :
  let participants : ℕ → ℝ := λ n, 1000 * 1.6 ^ n
  participants 3 = 4096 := by
  sorry

end expected_participants_2003_l139_139384


namespace exists_convex_inscribed_1992_gon_l139_139034

theorem exists_convex_inscribed_1992_gon :
  ∃ (polygon : List ℝ), 
    polygon.length = 1992 ∧ 
    (∀ (side_length : ℝ), side_length ∈ polygon → (1 ≤ side_length ∧ side_length ≤ 1992)) ∧ 
    Multiset.Nodup (polygon.toMultiset) ∧ 
    is_convex polygon ∧ 
    has_inscribed_circle polygon :=
sorry

end exists_convex_inscribed_1992_gon_l139_139034


namespace distance_between_starting_points_l139_139508

theorem distance_between_starting_points :
  let speed1 := 70
  let speed2 := 80
  let start_time := 10 -- in hours (10 am)
  let meet_time := 14 -- in hours (2 pm)
  let travel_time := meet_time - start_time
  let distance1 := speed1 * travel_time
  let distance2 := speed2 * travel_time
  distance1 + distance2 = 600 :=
by
  sorry

end distance_between_starting_points_l139_139508


namespace solve_eq1_solve_eq2_l139_139816

-- Problem 1: Prove that the solutions to x^2 - 6x + 3 = 0 are x = 3 + sqrt 6 or x = 3 - sqrt 6
theorem solve_eq1 (x : ℝ) : x^2 - 6 * x + 3 = 0 ↔ x = 3 + real.sqrt 6 ∨ x = 3 - real.sqrt 6 :=
by sorry

-- Problem 2: Prove that the solutions to 2x(x-1) = 3 - 3x are x = 1 or x = -1.5
theorem solve_eq2 (x : ℝ) : 2 * x * (x - 1) = 3 - 3 * x ↔ x = 1 ∨ x = -3 / 2 :=
by sorry

end solve_eq1_solve_eq2_l139_139816


namespace Larry_spends_108_minutes_l139_139406

-- Define conditions
def half_hour_twice_daily := 30 * 2
def fifth_of_an_hour_daily := 60 / 5
def quarter_hour_twice_daily := 15 * 2
def tenth_of_an_hour_daily := 60 / 10

-- Define total times spent on each pet
def total_time_dog := half_hour_twice_daily + fifth_of_an_hour_daily
def total_time_cat := quarter_hour_twice_daily + tenth_of_an_hour_daily

-- Define the total time spent on pets
def total_time_pets := total_time_dog + total_time_cat

-- Lean theorem statement
theorem Larry_spends_108_minutes : total_time_pets = 108 := 
  by 
    sorry

end Larry_spends_108_minutes_l139_139406


namespace quadrilateral_with_three_right_angles_is_rectangle_l139_139526

-- Define a quadrilateral with angles
structure Quadrilateral :=
  (a1 a2 a3 a4 : ℝ)
  (sum_angles : a1 + a2 + a3 + a4 = 360)

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop :=
  angle = 90

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  is_right_angle q.a1 ∧ is_right_angle q.a2 ∧ is_right_angle q.a3 ∧ is_right_angle q.a4

-- The main theorem: if a quadrilateral has three right angles, it is a rectangle
theorem quadrilateral_with_three_right_angles_is_rectangle 
  (q : Quadrilateral) 
  (h1 : is_right_angle q.a1) 
  (h2 : is_right_angle q.a2) 
  (h3 : is_right_angle q.a3) 
  : is_rectangle q :=
sorry

end quadrilateral_with_three_right_angles_is_rectangle_l139_139526


namespace prime_lonely_infinite_non_lonely_l139_139566

-- Define the sum of the reciprocals of the divisors of n
def T (n : ℕ) : ℚ := ∑ d in (Finset.filter (λ d, d ≠ 0 ∧ n % d = 0) (Finset.range (n+1))), (1 : ℚ) / d

-- Define lonely numbers
def lonely (n : ℕ) : Prop :=
  ∀ m : ℕ, n ≠ m → T n ≠ T m

-- Part (a) Prime numbers are lonely
theorem prime_lonely (p : ℕ) (hp : Nat.Prime p) : lonely p :=
sorry

-- Part (b) Infinitely many non-lonely numbers
theorem infinite_non_lonely : ∃ f : ℕ → ℕ, (function.surjective f) ∧ (∀ k, ¬lonely (6 * f k)) :=
sorry

end prime_lonely_infinite_non_lonely_l139_139566


namespace dave_bought_26_tshirts_l139_139220

def total_tshirts :=
  let white_tshirts := 3 * 6
  let blue_tshirts := 2 * 4
  white_tshirts + blue_tshirts

theorem dave_bought_26_tshirts : total_tshirts = 26 :=
by
  unfold total_tshirts
  have white_tshirts : 3 * 6 = 18 := by norm_num
  have blue_tshirts : 2 * 4 = 8 := by norm_num
  rw [white_tshirts, blue_tshirts]
  norm_num

end dave_bought_26_tshirts_l139_139220


namespace flowers_per_set_l139_139400

variable (totalFlowers : ℕ) (numSets : ℕ)

theorem flowers_per_set (h1 : totalFlowers = 270) (h2 : numSets = 3) : totalFlowers / numSets = 90 :=
by
  sorry

end flowers_per_set_l139_139400


namespace marie_divided_by_alex_l139_139797

theorem marie_divided_by_alex :
  let maries_sum := (finset.range 300).sum (λ n, 2 * (n + 1))
  let alexs_sum := (finset.range 300).sum (λ n, n + 1)
  (maries_sum : ℚ) / alexs_sum = 2 :=
by
  -- Definitions based on the problem
  let maries_sum := (finset.range 300).sum (λ n, 2 * (n + 1))
  let alexs_sum := (finset.range 300).sum (λ n, n + 1)
  -- sorry added to indicate the proof is not complete
  sorry

end marie_divided_by_alex_l139_139797


namespace tangents_are_concurrent_at_fixed_point_l139_139440

theorem tangents_are_concurrent_at_fixed_point (p q : ℝ → ℝ) (k : ℝ) :
  (∀ x, p(x) ≠ 0 ∧ q(x) ≠ 0) →
  (∀ y x, (∂y/∂x + p(x) * y = q(x))) →
  ∀ f : ℝ → ℝ, (∀ x, (∂f/∂x = q(x) - p(x) * f(x))) →
  let fixed_point := (k + 1 / p(k), q(k) / p(k)) in
  ∀ x₀ x₁, x₀ ≠ x₁ → f(x₀) = y₀ → f(x₁) = y₁ →
  let tangent₀ := (∂y/∂x at x = x₀) in
  let tangent₁ := (∂y/∂x at x = x₁) in
  line_through (x₀, y₀) tangent₀ fixed_point ∧ line_through (x₁, y₁) tangent₁ fixed_point :=
begin
  sorry
end

end tangents_are_concurrent_at_fixed_point_l139_139440


namespace sum_of_products_l139_139890

def is_positive (x : ℝ) := 0 < x

theorem sum_of_products 
  (x y z : ℝ) 
  (hx : is_positive x)
  (hy : is_positive y)
  (hz : is_positive z)
  (h1 : x^2 + x * y + y^2 = 27)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 52) :
  x * y + y * z + z * x = 30 :=
  sorry

end sum_of_products_l139_139890


namespace main_l139_139148

variable {α : Type} [LinearOrderedField α]

-- Definitions of points and the acute triangle
variables (A B C H P Q O : α × α)
variables (AO AH : α)

noncomputable def collinear (p1 p2 p3 : α × α) : Prop :=
(∃ k : α, k ≠ 0 ∧ p1.1 = k * p2.1 ∧ p1.2 = k * p2.2) ∨
(∃ k : α, k ≠ 0 ∧ p2.1 = k * p3.1 ∧ p2.2 = k * p3.2) ∨ 
(∃ k : α, k ≠ 0 ∧ p3.1 = k * p1.1 ∧ p3.2 = k * p1.2)

def problem_statement : Prop :=
  (A.1 ≠ B.1 ∨ A.2 ≠ B.2) ∧ (B.1 ≠ C.1 ∨ B.2 ≠ C.2) ∧ (C.1 ≠ A.1 ∨ C.2 ≠ A.2)
  ∧ AH^2 = 2 * AO^2
  ∧ collinear O P Q

theorem main : problem_statement A B C H P Q O AH AO := sorry

end main_l139_139148


namespace well_diameter_l139_139251

theorem well_diameter 
  (h : ℝ) 
  (P : ℝ) 
  (C : ℝ) 
  (V : ℝ) 
  (r : ℝ) 
  (d : ℝ) 
  (π : ℝ) 
  (h_eq : h = 14)
  (P_eq : P = 15)
  (C_eq : C = 1484.40)
  (V_eq : V = C / P)
  (volume_eq : V = π * r^2 * h)
  (radius_eq : r^2 = V / (π * h))
  (diameter_eq : d = 2 * r) : 
  d = 3 :=
by
  sorry

end well_diameter_l139_139251


namespace alice_distance_correct_l139_139177

noncomputable def alice_distance_from_start : ℝ :=
  let side_length := 2 in
  let distance_walked := 5 in
  let hexagon_vertices := [
    (0, 0),
    (2, 0),
    (3, Real.sqrt 3),
    (2, 2 * Real.sqrt 3),
    (0, 2 * Real.sqrt 3),
    (-1, Real.sqrt 3)
  ];
  let coord_c := (3, Real.sqrt 3) in
  let unit_dir_vector := (-1/2 : ℝ, Real.sqrt 3 / 2 : ℝ) in
  let final_position := (coord_c.1 + unit_dir_vector.1, coord_c.2 + unit_dir_vector.2) in
  let dist_sq := final_position.1^2 + final_position.2^2 in
  Real.sqrt dist_sq

theorem alice_distance_correct :
  alice_distance_from_start = Real.sqrt 13 := sorry

end alice_distance_correct_l139_139177


namespace avg_price_of_pen_l139_139897

theorem avg_price_of_pen 
  (total_pens : ℕ) (total_pencils : ℕ) (total_cost : ℕ) 
  (avg_price_pencil : ℕ) (total_pens_cost : ℕ) (total_pencils_cost : ℕ)
  (total_cost_eq : total_cost = total_pens_cost + total_pencils_cost)
  (total_pencils_cost_eq : total_pencils_cost = total_pencils * avg_price_pencil)
  (pencils_count : total_pencils = 75) (pens_count : total_pens = 30) 
  (avg_price_pencil_eq : avg_price_pencil = 2)
  (total_cost_eq' : total_cost = 450) :
  total_pens_cost / total_pens = 10 :=
by
  sorry

end avg_price_of_pen_l139_139897


namespace impossible_pawn_placement_l139_139944

theorem impossible_pawn_placement :
  ¬(∃ a b c : ℕ, a + b + c = 50 ∧ 
  ∀ (x y z : ℕ), 2 * a ≤ x ∧ x ≤ 2 * b ∧ 2 * b ≤ y ∧ y ≤ 2 * c ∧ 2 * c ≤ z ∧ z ≤ 2 * a) := sorry

end impossible_pawn_placement_l139_139944


namespace min_value_of_fraction_sum_l139_139315

open Real

theorem min_value_of_fraction_sum (x1 x2 : ℝ) (h1 : x1 ≠ x2) 
  (h2 : (log x1 / log 2)^2 - 4 * (log x1 / log 2) + 3 = (log x2 / log 2)^2 - 4 * (log x2 / log 2) + 3) : 
  (13 / x1) + (16 / x2) ≥ 2 * sqrt 13 :=
by 
  sorry

end min_value_of_fraction_sum_l139_139315


namespace sufficient_but_not_necessary_condition_l139_139276

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a < b) : 
  ((a - b) * a^2 < 0) ↔ (a < b) :=
sorry

end sufficient_but_not_necessary_condition_l139_139276


namespace remainder_division_seventeen_l139_139523

theorem remainder_division_seventeen (A : ℕ) :
  let dividend := 17
  let divisor := 5
  let quotient := 3
  in dividend - (divisor * quotient) = A → A = 2 :=
by
  let dividend := 17
  let divisor := 5
  let quotient := 3
  assume h: dividend - (divisor * quotient) = A
  have h1: 17 - (5 * 3) = A, from h
  have h2: 17 - 15 = A, from h1
  have h3: 2 = A, from h2
  show A = 2, from eq.symm h3

end remainder_division_seventeen_l139_139523


namespace measure_of_angle_x_l139_139983

theorem measure_of_angle_x :
  ∀ (angle_ABC angle_BDE angle_DBE angle_ABD x : ℝ),
    angle_ABC = 132 ∧
    angle_BDE = 31 ∧
    angle_DBE = 30 ∧
    angle_ABD = 180 - 132 →
    x = 180 - (angle_BDE + angle_DBE) →
    x = 119 :=
by
  intros angle_ABC angle_BDE angle_DBE angle_ABD x h h2
  sorry

end measure_of_angle_x_l139_139983


namespace marcus_has_210_cards_l139_139796

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the increment of baseball cards Marcus has over Carter
def increment : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + increment

-- Prove that Marcus has 210 baseball cards
theorem marcus_has_210_cards : marcus_cards = 210 :=
by simp [marcus_cards, carter_cards, increment]

end marcus_has_210_cards_l139_139796


namespace find_g5_l139_139056

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l139_139056


namespace combination_count_l139_139643

open Nat

def valid_combinations (cards : List Nat) : Bool :=
  ∀ i j, {i, j} ⊆ Finset.range cards.length → |cards[i] - cards[j]| ≥ 2

theorem combination_count :
  ∃ card_selections : Finset (Finset Fin 7), card_selections.card = 10 ∧
  ∀ cards ∈ card_selections, valid_combinations cards.to_list :=
begin
  sorry,
end

end combination_count_l139_139643


namespace largest_possible_N_l139_139004

theorem largest_possible_N (N : ℕ) :
  let divisors := Nat.divisors N
  in (1 ∈ divisors) ∧ (N ∈ divisors) ∧ (divisors.length ≥ 3) ∧ (divisors[divisors.length - 3] = 21 * divisors[1]) → N = 441 := 
by
  sorry

end largest_possible_N_l139_139004


namespace speed_of_faster_train_l139_139115

-- Define the variables and conditions
def length_of_train : ℕ := 25
def speed_slower_train : ℚ := 36   -- speed in km/hr
def time_to_pass : ℕ := 18         -- time in seconds

-- Define the theorem to prove
theorem speed_of_faster_train : 
  ∃ (V : ℚ), V = 46 ∧ V * (5 / 18) - speed_slower_train * (5 / 18) = (length_of_train + length_of_train) / time_to_pass ∧ 
  speed_slower_train = 36 := 
by 
  use 46
  split
  . exact rfl
  split
  . calc 
    46 * (5 / 18) - 36 * (5 / 18) = (46 - 36) * (5 / 18) : by ring
    ... = 10 * (5 / 18) : by norm_num1
    ... = 50 / 18 : by norm_cast
  . exact rfl

end speed_of_faster_train_l139_139115


namespace max_pairwise_obtuse_rays_l139_139129

theorem max_pairwise_obtuse_rays (S : Point) : ∃! (n : ℕ), ∀ (rays : Fin n → Ray S), (∀ i j, i ≠ j → obtuse_angle (rays i) (rays j)) → n = 4 :=
by sorry

end max_pairwise_obtuse_rays_l139_139129


namespace range_of_m_if_monotonic_compare_f_with_xcubed_prove_inequality_l139_139308

-- Definition of the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * Real.log (x + 1)

-- (1) If f(x) is monotonic on its domain, find the range of m
theorem range_of_m_if_monotonic (x : ℝ) (h : -1 < x) (m : ℝ) (hm : ∀ x, -1 < x → 2 * x + m / (x + 1) ≥ 0 ∨ 2 * x + m / (x + 1) ≤ 0) : m ≥ 1 / 2 :=
sorry

-- (2) If m = -1, compare f(x) with x^3 for x ∈ (0, ∞)
theorem compare_f_with_xcubed (x : ℝ) (hx : 0 < x) : (f x (-1)) < x^3 :=
sorry

-- (3) Prove the inequality for any positive integer n
theorem prove_inequality (n : ℕ) (hn : 0 < n) : (∑ i in Finset.range n, Real.exp ((1-(i:ℝ))*(i:ℝ)^2)) < n * (n + 3) / 2 :=
sorry

end range_of_m_if_monotonic_compare_f_with_xcubed_prove_inequality_l139_139308


namespace inequality_l139_139439

theorem inequality {n m : ℕ} (hn : n ≥ m) (hn_pos : n > 0) (hm_pos : m > 0) :
  (n + 1) ^ m * n ^ m ≥ (n + m)! / (n - m)! ∧ (n + m)! / (n - m)! ≥ 2 ^ m * m! :=
by
  sorry

end inequality_l139_139439


namespace probability_x_y_le_5_l139_139565

theorem probability_x_y_le_5 :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (4 : ℝ, 0 : ℝ)
  let C := (4 : ℝ, 8 : ℝ)
  let D := (0 : ℝ, 8 : ℝ)
  let area_rectangle := (B.1 - A.1) * (D.2 - A.2)
  let x_y_le_5 := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 8 ∧ p.1 + p.2 ≤ 5}
  let vertices := [(0 : ℝ, 0 : ℝ), (4 : ℝ, 1 : ℝ), (0 : ℝ, 5 : ℝ)]
  let area_triangle := 1 / 2 * (vertices[1].1 - vertices[0].1) * (vertices[2].2 - vertices[0].2)
  let probability := area_triangle / area_rectangle
  probability = 5 / 16 := 
by
  -- Omitted the proof details
  sorry

end probability_x_y_le_5_l139_139565


namespace find_extremal_path_length_l139_139200

noncomputable def extremal_point_on_circle 
  (A B : ℝ×ℝ) (O : ℝ×ℝ) (r : ℝ) : (ℝ×ℝ) :=
sorry

theorem find_extremal_path_length 
  (A B O : ℝ×ℝ) (r : ℝ) 
  (hA : (A.1 - O.1)^2 + (A.2 - O.2)^2 < r^2) 
  (hB : (B.1 - O.1)^2 + (B.2 - O.2)^2 < r^2) :
  ∃ P: ℝ×ℝ, (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 ∧
  (∀ Q: ℝ×ℝ, (Q.1 - O.1)^2 + (Q.2 - O.2)^2 = r^2 → AP + PB ≤ AQ + QB) ∨
  (∀ Q: ℝ×ℝ, (Q.1 - O.1)^2 + (Q.2 - O.2)^2 = r^2 → AP + PB ≥ AQ + QB) :=
sorry

end find_extremal_path_length_l139_139200


namespace distance_between_foci_hyperbola_l139_139451

def hyperbola_asymptotes_center (asymptote1 asymptote2 : ℝ → Prop) : ℝ × ℝ :=
  let ⟨x, y, _⟩ := exists_intersect_point asymptote1 asymptote2 in (x, y)

def passes_through_point (P : ℝ × ℝ) (x y : ℝ) : Prop :=
  P.fst = x ∧ P.snd = y

theorem distance_between_foci_hyperbola:
  let asymptote1 : ℝ → Prop := λ y, y = x + 3
  let asymptote2 : ℝ → Prop := λ y, y = -x + 5
  let P : ℝ × ℝ := (1, 5)

  (passes_through_point P (1, 5)) ∧ 
  (hyperbola_asymptotes_center asymptote1 asymptote2 = (1, 4)) →
  ∃ d : ℝ, d = 2 * real.sqrt 2 :=
by sorry

end distance_between_foci_hyperbola_l139_139451


namespace nonagon_interior_angle_140_l139_139914

theorem nonagon_interior_angle_140 (interior_angle : ℕ) (h1 : interior_angle = 140) (h2 : ∀ (x : ℕ), x = 180 - interior_angle)
  (h3 : (∑ e in (Ico 1 (interior_angle)), (180 - e)) = 360) : (∑ e in (Ico 1 (interior_angle)),  (360 / (180 - e)).card = 9) :=
sorry

end nonagon_interior_angle_140_l139_139914


namespace percentage_increase_l139_139846

-- Conditions
variables (S_final S_initial : ℝ) (P : ℝ)
def conditions := (S_final = 3135) ∧ (S_initial = 3000) ∧
  (S_final = (S_initial + (P/100) * S_initial) - 0.05 * (S_initial + (P/100) * S_initial))

-- Statement of the problem
theorem percentage_increase (S_final S_initial : ℝ) 
  (cond : conditions S_final S_initial P) : P = 10 := by
  sorry

end percentage_increase_l139_139846


namespace Lesha_can_leave_two_columns_l139_139100

theorem Lesha_can_leave_two_columns :
  ∀ (table : ℕ → ℕ → ℕ), (∀ i j, table i j ≤ 2) → 
  (∃ C1 C2, C1 ≠ C2 ∧ (∀ i, table i C1 ≠ table i C2)) → 
  (∃ C1 C2, C1 ≠ C2 ∧ (∀ i, (∀ j, table i j ≠ table i C1 → table i C2 ≠ j) ∧ (∀ j, table i j ≠ table i C2 → table i C1 ≠ j))) →
  true :=
begin
  sorry
end

end Lesha_can_leave_two_columns_l139_139100


namespace order_of_numbers_l139_139503

-- Definitions of the three numbers as per the conditions
def a : ℝ := 6 ^ 0.7
def b : ℝ := 0.7 ^ 6
def c : ℝ := Real.log 6 / Real.log 0.7 -- log_base_0.7(6)

-- The theorem that states the order of the numbers
theorem order_of_numbers : c < b ∧ b < a := by
  sorry

end order_of_numbers_l139_139503


namespace students_and_confucius_same_arrival_time_l139_139891

noncomputable def speed_of_students_walking (x : ℝ) : ℝ := x

noncomputable def speed_of_bullock_cart (x : ℝ) : ℝ := 1.5 * x

noncomputable def time_for_students_to_school (x : ℝ) : ℝ := 30 / x

noncomputable def time_for_confucius_to_school (x : ℝ) : ℝ := 30 / (1.5 * x) + 1

theorem students_and_confucius_same_arrival_time (x : ℝ) (h1 : 0 < x) :
  30 / x = 30 / (1.5 * x) + 1 :=
by
  sorry

end students_and_confucius_same_arrival_time_l139_139891


namespace solution_set_inequality_l139_139094

theorem solution_set_inequality : 
  {x : ℝ | abs ((x - 3) / x) > ((x - 3) / x)} = {x : ℝ | 0 < x ∧ x < 3} :=
sorry

end solution_set_inequality_l139_139094


namespace least_x_multiple_of_53_l139_139519

theorem least_x_multiple_of_53 : ∃ (x : ℕ), (x > 0) ∧ (3 * x + 41) % 53 = 0 ∧ x = 4 :=
by
  have : ∃ (x : ℕ), (x > 0) ∧ (3 * x + 41) % 53 = 0, from sorry
  exact ⟨4, sorry, sorry⟩

end least_x_multiple_of_53_l139_139519


namespace gunther_free_time_remaining_l139_139334

-- Define the conditions
def vacuum_time : ℕ := 45
def dust_time : ℕ := 60
def mop_time : ℕ := 30
def brushing_time_per_cat : ℕ := 5
def number_of_cats : ℕ := 3
def free_time_hours : ℕ := 3

-- Convert the conditions into a proof problem
theorem gunther_free_time_remaining : 
  let total_cleaning_time := vacuum_time + dust_time + mop_time + brushing_time_per_cat * number_of_cats
  let free_time_minutes := free_time_hours * 60
  in free_time_minutes - total_cleaning_time = 30 :=
by
  sorry

end gunther_free_time_remaining_l139_139334


namespace problem1_l139_139893

theorem problem1 (a : ℝ) 
    (circle_eqn : ∀ (x y : ℝ), x^2 + y^2 - 2*a*x + a = 0)
    (line_eqn : ∀ (x y : ℝ), a*x + y + 1 = 0)
    (chord_length : ∀ (x y : ℝ), (ax + y + 1 = 0) ∧ (x^2 + y^2 - 2*a*x + a = 0)  -> ((x - x')^2 + (y - y')^2 = 4)) : 
    a = -2 := sorry

end problem1_l139_139893


namespace find_y_from_equation_l139_139987

theorem find_y_from_equation :
  ∀ y : ℕ, (12 ^ 3 * 6 ^ 4) / y = 5184 → y = 432 :=
by
  sorry

end find_y_from_equation_l139_139987


namespace largest_possible_value_of_N_l139_139011

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l139_139011


namespace coefficients_integers_or_even_l139_139447

-- Definitions and conditions
variables {R : Type*} [Ring R]
variables (a1 b1 c1 a2 b2 c2 : R)
variables (x y : R)

-- Problem statement
theorem coefficients_integers_or_even (h : ∀ x y : ℤ, (a1 * x + b1 * y + c1).Even ∨ (a2 * x + b2 * y + c2).Even) :
  (a1 ∈ ℤ ∧ b1 ∈ ℤ ∧ c1 ∈ ℤ) ∨ (a2 ∈ ℤ ∧ b2 ∈ ℤ ∧ c2 ∈ ℤ) :=
sorry

end coefficients_integers_or_even_l139_139447


namespace necessary_but_not_sufficient_l139_139827

theorem necessary_but_not_sufficient (a b : ℕ) : 
  (a ≠ 1 ∨ b ≠ 2) → ¬ (a + b = 3) → ¬(a = 1 ∧ b = 2) ∧ ((a = 1 ∧ b = 2) → (a + b = 3)) := sorry

end necessary_but_not_sufficient_l139_139827


namespace binomial_coefficient_x2_l139_139715

theorem binomial_coefficient_x2 
  (n : ℝ) 
  (h_integral : ∫ x in 0..n, |x - 5| = 25) 
  : ∃ k : ℕ, (2*x - 1)^n = k * x^2 + terms 
  ∧ k = 180 := 
  sorry

end binomial_coefficient_x2_l139_139715


namespace evaluate_expression_l139_139496

theorem evaluate_expression : (4 * 4 + 4) / (2 * 2 - 2) = 10 := by
  sorry

end evaluate_expression_l139_139496


namespace ellipse_equation_correct_l139_139580

theorem ellipse_equation_correct :
  ∃ (a b h k : ℝ), 
    h = 4 ∧ 
    k = 0 ∧ 
    a = 10 + 2 * Real.sqrt 10 ∧ 
    b = Real.sqrt (101 + 20 * Real.sqrt 10) ∧ 
    (∀ x y : ℝ, (x, y) = (9, 6) → 
    ((x - h)^2 / a^2 + y^2 / b^2 = 1)) ∧
    (dist (4 - 3, 0) (4 + 3, 0) = 6) := 
sorry

end ellipse_equation_correct_l139_139580


namespace evaluate_sum_l139_139239

-- Definition of the given problem conditions
def Phi (θ : ℝ) : ℝ := sin θ ^ 2
def Psi (θ : ℝ) : ℝ := cos θ
def sequence : List ℝ := [1, 2, 45, 47]

noncomputable def sum_expr : ℝ :=
  ∑ i in List.finRange 4, (-1) ^ (i + 1) * (Phi (sequence.nthLe i (by simp)) / Psi (sequence.nthLe i (by simp)))

theorem evaluate_sum : sum_expr = 95 := 
  sorry

end evaluate_sum_l139_139239


namespace sequence_sum_l139_139655

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = 2 ∧ a 2 = 2 ∧
  (∀ n : ℕ, n > 0 → a (n + 2) = (1 + Real.cos (n * Real.pi)) * (a n - 1) + 2) →
  (∀ n : ℕ, S (2 * n) = ∑ k in Finset.range (2 * n + 1), a k) →
  (∀ n : ℕ, S (2 * n) = 2 ^ (n + 1) + 2 * n - 2) :=
sorry

end sequence_sum_l139_139655


namespace concurrency_of_lines_l139_139417

variable {A B C D E F A1 B1 C1 : Type _}

-- Assume we have a triangle and relevant points as described
variables [triangle ABC]
variables [circumcircle Γ]
variables (incircle : touches_at Γ ⟨A, B, C⟩ ⟨D, E, F⟩)
variables (tangent_circle_A1 : tangent_to_segment_and_arc Γ ⟨BC, D⟩ ⟨A1, BC⟩)
variables (tangent_circle_B1 : tangent_to_segment_and_arc Γ ⟨CA, E⟩ ⟨B1, CA⟩)
variables (tangent_circle_C1 : tangent_to_segment_and_arc Γ ⟨AB, F⟩ ⟨C1, AB⟩)

theorem concurrency_of_lines :
  concurrent (line_through_points A1 D) (line_through_points B1 E) (line_through_points C1 F) := 
sorry

end concurrency_of_lines_l139_139417


namespace triangle_angles_4_4_2sqrt2_l139_139922

noncomputable def cosine_inverse (x : ℝ) : ℝ := Real.arccos x

noncomputable def triangle_angles (a b c : ℝ) : (ℝ × ℝ × ℝ) :=
let θ := cosine_inverse ((a*a + b*b - c*c) / (2 * a * b)) in
(θ, (180 - θ) / 2, (180 - θ) / 2)

theorem triangle_angles_4_4_2sqrt2 :
  triangle_angles 4 4 (2 * Real.sqrt 2) = (41.41, 69.295, 69.295) :=
by {
  -- The actual proof goes here, but it's skipped as per instructions.
  sorry
}

end triangle_angles_4_4_2sqrt2_l139_139922


namespace find_g5_l139_139068

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l139_139068


namespace room_width_is_4_75_l139_139461

-- Definitions based on the problem conditions
def length (room : Type) : ℝ := 7
def cost_per_sq_meter : ℝ := 900
def total_cost : ℝ := 29925

-- The area of the floor can be derived from the cost
def area : ℝ := total_cost / cost_per_sq_meter

-- The width can be derived from the length and the area
def width (room : Type) : ℝ := area / length room

-- The theorem that represents the proof problem
theorem room_width_is_4_75 (room : Type) : width room = 4.75 := 
by
  sorry

end room_width_is_4_75_l139_139461


namespace total_profit_l139_139165

variable (InvestmentA InvestmentB InvestmentTimeA InvestmentTimeB ShareA : ℝ)
variable (hA : InvestmentA = 150)
variable (hB : InvestmentB = 200)
variable (hTimeA : InvestmentTimeA = 12)
variable (hTimeB : InvestmentTimeB = 6)
variable (hShareA : ShareA = 60)

theorem total_profit (TotalProfit : ℝ) :
  (ShareA / 3) * 5 = TotalProfit := 
by
  sorry

end total_profit_l139_139165


namespace problem_solution_l139_139092

theorem problem_solution (x y : ℝ) 
  (h : (⟨3, x, -8⟩ : ℝ × ℝ × ℝ) × (⟨4, 9, y⟩ : ℝ × ℝ × ℝ) = ⟨0, 0, 0⟩) : (x = 27 / 4) ∧ (y = -32 / 3) :=
by 
  sorry

end problem_solution_l139_139092


namespace mary_blue_marbles_l139_139610

theorem mary_blue_marbles (dan_blue_marbles mary_blue_marbles : ℕ)
  (h1 : dan_blue_marbles = 5)
  (h2 : mary_blue_marbles = 2 * dan_blue_marbles) : mary_blue_marbles = 10 := 
by
  sorry

end mary_blue_marbles_l139_139610


namespace combined_selling_price_correct_l139_139563

def ArticleA_Cost : ℝ := 500
def ArticleA_Profit_Percent : ℝ := 0.45
def ArticleB_Cost : ℝ := 300
def ArticleB_Profit_Percent : ℝ := 0.30
def ArticleC_Cost : ℝ := 1000
def ArticleC_Profit_Percent : ℝ := 0.20
def Sales_Tax_Percent : ℝ := 0.12

def CombinedSellingPrice (A_cost A_profit_percent B_cost B_profit_percent C_cost C_profit_percent tax_percent : ℝ) : ℝ :=
  let A_selling_price := A_cost * (1 + A_profit_percent)
  let A_final_price := A_selling_price * (1 + tax_percent)
  let B_selling_price := B_cost * (1 + B_profit_percent)
  let B_final_price := B_selling_price * (1 + tax_percent)
  let C_selling_price := C_cost * (1 + C_profit_percent)
  let C_final_price := C_selling_price * (1 + tax_percent)
  A_final_price + B_final_price + C_final_price

theorem combined_selling_price_correct :
  CombinedSellingPrice ArticleA_Cost ArticleA_Profit_Percent ArticleB_Cost ArticleB_Profit_Percent ArticleC_Cost ArticleC_Profit_Percent Sales_Tax_Percent = 2592.8 := by
  sorry

end combined_selling_price_correct_l139_139563


namespace sequence_general_term_l139_139097

noncomputable def S : ℕ → ℕ
| n := n^2 + 3 * n + 1

noncomputable def a : ℕ → ℕ
| 1     := 5
| (n+2) := 2 * (n+2) + 2

theorem sequence_general_term (n : ℕ) : 
  a n = (if n = 1 then 5 else 2 * n + 2) :=
by
  sorry

end sequence_general_term_l139_139097


namespace second_term_of_polynomial_l139_139803

theorem second_term_of_polynomial :
  ∀ x : ℝ, (x^2 - x^3).second_term = -x^3 := sorry

end second_term_of_polynomial_l139_139803


namespace find_particular_number_l139_139908

theorem find_particular_number (A B : ℤ) (x : ℤ) (hA : A = 14) (hB : B = 24)
  (h : (((A + x) * A - B) / B = 13)) : x = 10 :=
by {
  -- You can add an appropriate lemma or proof here if necessary
  sorry
}

end find_particular_number_l139_139908


namespace correct_statements_count_l139_139088

theorem correct_statements_count :
  (∀ x > 0, x > Real.sin x) ∧
  (¬ (∀ x > 0, x - Real.log x > 0) ↔ (∃ x > 0, x - Real.log x ≤ 0)) ∧
  ¬ (∀ p q : Prop, (p ∨ q) → (p ∧ q)) →
  2 = 2 :=
by sorry

end correct_statements_count_l139_139088


namespace range_of_f_1_over_f_2_l139_139163

theorem range_of_f_1_over_f_2 {f : ℝ → ℝ} (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  1 / 8 < f 1 / f 2 ∧ f 1 / f 2 < 1 / 4 :=
by sorry

end range_of_f_1_over_f_2_l139_139163


namespace part_a_impossible_part_b_possible_l139_139023

-- Part (a)
theorem part_a_impossible : ∀ (digits : List ℕ),
  digits = [3, 2, 4, 5, 6, 1] →
  (∀ (new_digits : List ℕ), false ∨ ¬ (new_digits = [4, 3, 4, 4, 3, 4])) :=
begin
  sorry
end

-- Part (b)
theorem part_b_possible : ∃ (digits : List ℕ),
  digits = [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
  digits.foldl (λ acc x, acc * 10 + x) 0 > 800000000 :=
begin
  sorry
end

end part_a_impossible_part_b_possible_l139_139023


namespace sum_first_100_even_l139_139266

theorem sum_first_100_even :
  (∑ k in Finset.range 100, 2 * (k + 1)) = 10100 :=
by
  sorry

end sum_first_100_even_l139_139266


namespace smallest_b_no_inverse_mod_72_80_l139_139876

theorem smallest_b_no_inverse_mod_72_80 : 
  ∃ (b : ℕ), b > 0 ∧ (∀ n, b * n % 72 ≠ 1) ∧ (∀ n, b * n % 80 ≠ 1) ∧ b = 4 :=
begin
  sorry
end

end smallest_b_no_inverse_mod_72_80_l139_139876


namespace incorrect_statements_l139_139527

open_locale classical

-- Definition: Vectors are equal if they have the same magnitude and direction
def vectors_equal (a b : ℝ × ℝ) : Prop :=
  a = b

-- Definition: "Greater than" and "Less than" do not apply to vectors.
def vector_comparable (a b : ℝ × ℝ) : Prop :=
  false

-- Definition: Vectors are parallel if they lie on parallel or the same lines.
def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2)

-- Definition: Vectors are collinear if they lie on the same line.
def vectors_collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Problem statement: Prove that A, B, and C are incorrect given the above definitions.
theorem incorrect_statements :
  ¬ vectors_equal (1, 2) (2, 1) ∧
  ¬ vector_comparable (1, 2) (2, 1) ∧
  ¬ (∀ a b : ℝ × ℝ, vectors_parallel a b → a ≠ (0,0) → b ≠ (0,0) → a ≠ b) :=
by {
  sorry
}

end incorrect_statements_l139_139527


namespace all_numbers_zero_l139_139596

theorem all_numbers_zero (n : ℕ) (h1 : n ≥ 3) 
  (points : fin n → ℝ) 
  (h2 : ¬ collinear (fin n) points)
  (h3 : ∀ (D : set (fin n)), (∃ p1 p2, p1 ≠ p2 ∧ p1 ∈ D ∧ p2 ∈ D) → ∑ p in D, points p = 0) :
  ∀ p, points p = 0 :=
by
  sorry

end all_numbers_zero_l139_139596


namespace calculation_l139_139942

theorem calculation : 2005^2 - 2003 * 2007 = 4 :=
by
  have h1 : 2003 = 2005 - 2 := by rfl
  have h2 : 2007 = 2005 + 2 := by rfl
  sorry

end calculation_l139_139942


namespace find_x_l139_139194

theorem find_x (A V R S x : ℝ) 
  (h1 : A + x = V - x)
  (h2 : V + 2 * x = A - 2 * x + 30)
  (h3 : (A + R / 2) + (V + R / 2) = 120)
  (h4 : S - 0.25 * S + 10 = 2 * (R / 2)) :
  x = 5 :=
  sorry

end find_x_l139_139194


namespace GuntherFreeTime_l139_139336

def GuntherCleaning : Nat := 45 + 60 + 30 + 15

def TotalFreeTime : Nat := 180

theorem GuntherFreeTime : TotalFreeTime - GuntherCleaning = 30 := by
  sorry

end GuntherFreeTime_l139_139336


namespace find_g5_l139_139079

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l139_139079


namespace integral_solution_l139_139941

noncomputable def integral_problem : Prop :=
  ∫ x in 6..9, sqrt((9 - 2 * x) / (2 * x - 21)) = Real.pi

theorem integral_solution : integral_problem :=
sorry

end integral_solution_l139_139941


namespace cos_4050_eq_zero_l139_139946

theorem cos_4050_eq_zero : cos (4050 * real.pi / 180) = 0 :=
by
  -- The detailed steps of the proof will go here
  sorry

end cos_4050_eq_zero_l139_139946


namespace find_k_find_m_l139_139318

variable {a k m x : ℝ}
variable (a_pos : a > 0) (a_neq_one : a ≠ 1)

-- First part: Prove that k = 0 for f being an odd function
theorem find_k (f : ℝ → ℝ) (odd_f : ∀ x, f(-x) = -f(x)) :
  (k = 0) := 
sorry

noncomputable def f (x : ℝ) : ℝ := a^x - (k + 1) * a^(-x)

-- Second part: Find the value of m given conditions
def g (x : ℝ) : ℝ := a^(2 * x) + a^(-2 * x) - 2 * m * f x

theorem find_m (k_eq_zero : k = 0) (f_one : f 1 = 3 / 2)
  (min_val_g : ∀ x ≥ 0, g x ≥ -6) :
  (m = 2 * Real.sqrt 2) := 
sorry

end find_k_find_m_l139_139318


namespace range_g_l139_139960

noncomputable def g (A : ℝ) : ℝ :=
  (sin A * (4 * cos A ^ 2 + cos A ^ 4 + 4 * sin A ^ 2 + 2 * sin A ^ 2 * cos A ^ 2)) / 
  (tan A * (sec A - 2 * sin A * tan A))

theorem range_g :
  (∀ A : ℝ, (∃ n : ℤ, A = n * π / 2) → false) →
  ∀ y : ℝ, (y = g(A) → A \in (4, 5) :=
sorry

end range_g_l139_139960


namespace problem1_l139_139150

theorem problem1 :
  (2021 - Real.pi)^0 + (Real.sqrt 3 - 1) - 2 + (2 * Real.sqrt 3) = 3 * Real.sqrt 3 - 2 :=
by
  sorry

end problem1_l139_139150


namespace student_A_selection_probability_l139_139570

def probability_student_A_selected (total_students : ℕ) (students_removed : ℕ) (representatives : ℕ) : ℚ :=
  representatives / (total_students : ℚ)

theorem student_A_selection_probability :
  probability_student_A_selected 752 2 5 = 5 / 752 :=
by
  sorry

end student_A_selection_probability_l139_139570


namespace jordan_probability_l139_139458

-- Definitions based on conditions.
def total_students := 28
def enrolled_in_french := 20
def enrolled_in_spanish := 23
def enrolled_in_both := 17

-- Calculate students enrolled only in one language.
def only_french := enrolled_in_french - enrolled_in_both
def only_spanish := enrolled_in_spanish - enrolled_in_both

-- Calculation of combinations.
def total_combinations := Nat.choose total_students 2
def only_french_combinations := Nat.choose only_french 2
def only_spanish_combinations := Nat.choose only_spanish 2

-- Probability calculations.
def prob_both_one_language := (only_french_combinations + only_spanish_combinations) / total_combinations

def prob_both_languages : ℚ := 1 - prob_both_one_language

theorem jordan_probability :
  prob_both_languages = (20 : ℚ) / 21 := by
  sorry

end jordan_probability_l139_139458


namespace unique_Q_determined_l139_139271

-- Define the arithmetic sequence and related sums
variables (b e : ℝ) (n k : ℕ)

-- Define the sequence and sums
def arith_seq (b n e : ℝ) := b + (n - 1) * e

def P (b e n : ℝ) := (n / 2) * (2 * b + (n - 1) * e)

def Q (b e n k : ℝ) := ∑ k in finset.range (n + 1), P b e k

-- Theorem statement
theorem unique_Q_determined (P2023 unique_b_e_value : ℝ) : P b e 2023 = P2023 → (b + 1011 * e = unique_b_e_value) → (Q b e 3034 = (P2023 / 2023) * 3034) :=
by
  -- Proof to be filled here
  sorry

end unique_Q_determined_l139_139271


namespace longest_side_of_triangle_l139_139836

theorem longest_side_of_triangle :
  ∃ y : ℚ, 6 + (y + 3) + (3 * y - 2) = 40 ∧ max (6 : ℚ) (max (y + 3) (3 * y - 2)) = 91 / 4 :=
by
  sorry

end longest_side_of_triangle_l139_139836


namespace max_n_51_l139_139764

-- define the problem conditions
def n_max (n : ℕ) : Prop :=
  (n ≤ 100) ∧ 
  (∀ (table : ℕ → ℕ → ℝ), -- for any 100x100 table
    (∃ (queries : list (ℕ × ℕ × ℕ × ℕ)), -- there exists a list of queries
      (∀ query ∈ queries, 
         let (r1, c1, r2, c2) := query in
         r1 ≤ 100 ∧ c1 ≤ 100 ∧ r2 ≤ 100 ∧ c2 ≤ 100) ∧ -- queries are within bounds
      (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 → -- for every cell in the table
        (r1 ≤ i ∧ i ≤ r2 ∧ c1 ≤ j ∧ j ≤ c2 ∧ r2 - r1 = n ∧ c2 - c1 = n → -- part of n x n square
         table i j = sum (table k l for k in r1..r2, l in c1..c2)) ∨ -- sum of n x n square query
        (r1 ≤ i ∧ i ≤ r2 ∧ c1 ≤ j ∧ j ≤ c2 ∧ (r2 - r1 = n-1 ∨ c2 - c1 = n-1) → -- part of 1 x (n-1) rectangle
         table i j = sum (table k l for k in r1..r2, l in c1..c2))))) -- sum of 1 x (n-1) rectangle query

-- define the proof problem
theorem max_n_51 : ∃ n, n_max n ∧ n = 51 := by
  existsi 51
  split
  sorry -- proof would go here

end max_n_51_l139_139764


namespace problem_geometric_and_arithmetic_l139_139411

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∃ (a1 : ℝ), (∀ n : ℕ, a n = a1 * q ^ n)

def Sn (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n + 1), a i

def arithmetic_sequence (x y z : ℝ) := 2 * y = x + z

theorem problem_geometric_and_arithmetic (a : ℕ → ℝ) (q : ℝ) (h_q : q > 1) 
  (h_geo : geometric_sequence a q) (h_S3 : Sn a 2 = 7)
  (h_arith : arithmetic_sequence (a 0 + 3) (3 * a 1) (a 2 + 4)) :
  (∀ n : ℕ, a n = 2^(n-1)) 
  ∧ (∀ b : ℕ → ℝ, b = λ n, real.log (a (3*n + 1)) → 
      ∀ T_n : ℕ → ℝ, T_n = λ n, ∑ i in finset.range (n + 1), b i → 
        ∀ n, T_n n = (3 * n * (n + 1) / 2) * real.log 2) :=
begin
  sorry
end

end problem_geometric_and_arithmetic_l139_139411


namespace perpendicular_vectors_t_value_l139_139706

open Real

theorem perpendicular_vectors_t_value :
  ∀ (t : ℝ), let a := (1 : ℝ, 1 : ℝ) in let b := (2 : ℝ, t) in
  (a.1 * b.1 + a.2 * b.2 = 0) → t = -2 :=
by
  intro t
  let a := (1 : ℝ, 1 : ℝ)
  let b := (2 : ℝ, t)
  intro h
  sorry

end perpendicular_vectors_t_value_l139_139706


namespace find_f_neg_two_l139_139722

-- Define the polynomial functions and their discriminants
def discriminant_f (a b c : ℝ) : ℝ := (2 * b)^2 - 4 * a * c
def discriminant_g (a b c : ℝ) : ℝ := (2 * (b + 2))^2 - 4 * (a + 1) * (c + 4)

-- Assert the condition for the discriminants' difference and define f(-2)
theorem find_f_neg_two (a b c : ℝ)
  (h : discriminant_f a b c - discriminant_g a b c = 24) :
  let f := λ x : ℝ, a * x^2 + 2 * b * x + c
  f (-2) = 6 := 
sorry

end find_f_neg_two_l139_139722


namespace seven_pow_n_plus_one_not_div_by_48_seven_pow_n_minus_one_div_by_48_iff_even_l139_139201

-- Problem statement for 7^n + 1
theorem seven_pow_n_plus_one_not_div_by_48 (n : ℕ) (hn : n > 0) : ¬ (48 ∣ (7^n + 1)) := 
sorry

-- Problem statement for 7^n - 1 (with the necessary condition that it holds if and only if n is even)
theorem seven_pow_n_minus_one_div_by_48_iff_even (n : ℕ) (hn : n > 0) : (48 ∣ (7^n - 1)) ↔ even n := 
sorry

end seven_pow_n_plus_one_not_div_by_48_seven_pow_n_minus_one_div_by_48_iff_even_l139_139201


namespace find_a_range_l139_139696

variable (a x : ℝ)

def p : Prop := 4 - x ≤ 6
def q : Prop := x > a - 1
def p_sufficient_not_necessary_for_q : Prop := p → q ∧ ¬(q → p)

theorem find_a_range (h : p_sufficient_not_necessary_for_q x a) : a < -1 :=
by
  sorry

end find_a_range_l139_139696


namespace area_of_rectangle_l139_139865

-- Define the properties and conditions of the problem
variables (P Q R A B C D : Point)
variable (diameter : ℝ)

-- Conditions from part a.
def is_congruent_circles : Prop := diameter = 6
def circle_condition : Prop := distance P Q = diameter / 2 ∧ distance Q R = diameter / 2 ∧ distance P R = diameter

-- Define the rectangle properties
noncomputable def height_AD : ℝ := 6
noncomputable def width_AB : ℝ := 18
noncomputable def area_rect : ℝ := height_AD * width_AB

-- Main statement to be proven
theorem area_of_rectangle
  (h_congruent : is_congruent_circles)
  (h_circle_condition : circle_condition P Q R diameter)
  (h_AD_condition : height_AD = 6)
  (h_AB_condition : width_AB = 18)
  : area_rect = 108 :=
by sorry

end area_of_rectangle_l139_139865


namespace volume_comparison_l139_139108

-- Define the height and diameter of the first box.
variables (h1 d1 : ℝ)

-- Define the height and diameter of the second box as given in conditions.
def h2 := 2 * h1
def d2 := d1 / 2

-- Define the radii of both boxes.
def r1 := d1 / 2
def r2 := d2 / 2

-- Define the volumes of both boxes using the formula for the volume of a cylinder.
def V1 : ℝ := π * (r1 ^ 2) * h1
def V2 : ℝ := π * (r2 ^ 2) * h2

-- Prove that the volume of the first box is twice the volume of the second box.
theorem volume_comparison (h1 d1 : ℝ) : V1 h1 d1 = 2 * (V2 h1 d1) :=
by
  sorry

end volume_comparison_l139_139108


namespace moment_goal_equality_l139_139864

theorem moment_goal_equality (total_goals_russia total_goals_tunisia : ℕ) (T : total_goals_russia = 9) (T2 : total_goals_tunisia = 5) :
  ∃ n, n ≤ 9 ∧ (9 - n) = total_goals_tunisia :=
by
  sorry

end moment_goal_equality_l139_139864


namespace car_mileage_increase_l139_139142

theorem car_mileage_increase 
  (mpg_before : ℝ)
  (gallons : ℝ)
  (mod_ratio : ℝ)
  (mpg_after : ℝ)
  (miles_before : ℝ)
  (miles_after : ℝ)
  (extra_miles : ℝ) :
  mpg_before = 27 → 
  gallons = 14 → 
  mod_ratio = 0.75 →
  mpg_after = mpg_before / mod_ratio → 
  miles_before = mpg_before * gallons → 
  miles_after = mpg_after * gallons → 
  extra_miles = miles_after - miles_before →
  extra_miles = 94.5 :=
by
  intros
  rw [‹mpg_before = 27›, ‹gallons = 14›, ‹mod_ratio = 0.75›, ‹mpg_after = mpg_before / mod_ratio›,
      ‹miles_before = mpg_before * gallons›, ‹miles_after = mpg_after * gallons›, ‹extra_miles = miles_after - miles_before›]
  sorry

end car_mileage_increase_l139_139142


namespace prob_score_3_points_l139_139858

-- Definitions for the probabilities
def probability_hit_A := 3/4
def score_hit_A := 1
def score_miss_A := -1

def probability_hit_B := 2/3
def score_hit_B := 2
def score_miss_B := 0

-- Conditional probabilities and their calculations
noncomputable def prob_scenario_1 : ℚ := 
  probability_hit_A * 2 * probability_hit_B * (1 - probability_hit_B)

noncomputable def prob_scenario_2 : ℚ := 
  (1 - probability_hit_A) * probability_hit_B^2

noncomputable def total_prob : ℚ := 
  prob_scenario_1 + prob_scenario_2

-- The final proof statement
theorem prob_score_3_points : total_prob = 4/9 := sorry

end prob_score_3_points_l139_139858


namespace sqrt_x_minus_1_meaningful_l139_139351

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 1)) → x ≥ 1 := by
  intros h
  cases h with y hy
  rw hy
  have := real.sqrt_nonneg (x - 1)
  sorry

end sqrt_x_minus_1_meaningful_l139_139351


namespace number_of_triangles_l139_139949

-- Define a rectangle with vertices A, B, C, and D, and midpoints M, N, P, Q of each side.
structure Rectangle :=
  (A B C D M N P Q : Type)
  (midpoint_AB : LineSegment A B -> M)
  (midpoint_BC : LineSegment B C -> N)
  (midpoint_CD : LineSegment C D -> P)
  (midpoint_DA : LineSegment D A -> Q)
  (diagonal_AC : Line A C)
  (diagonal_BD : Line B D)
  (segment_MC : LineSegment M C)
  (segment_ND : LineSegment N D)
  (segment_PA : LineSegment P A)
  (segment_QB : LineSegment Q B)

-- Theorem: The total number of triangles formed by the above configuration is 20.
theorem number_of_triangles (R : Rectangle) : 
  ∃ n : ℕ, n = 20 :=
by
  sorry

end number_of_triangles_l139_139949


namespace min_value_of_abcd_l139_139652

def is_valid_number (abcd n : ℕ) : Prop :=
  abcd + (div abcd 100) * (mod abcd 100) = 1111 * n

theorem min_value_of_abcd : ∃ (abcd : ℕ), abcd = 1729 ∧ ∃ (n : ℕ), is_valid_number abcd n :=
by {
  sorry
}

end min_value_of_abcd_l139_139652


namespace least_possible_value_l139_139888

theorem least_possible_value (x y z : ℕ) (hx : 2 * x = 5 * y) (hy : 5 * y = 8 * z) (hz : 8 * z = 2 * x) (hnz_x: x > 0) (hnz_y: y > 0) (hnz_z: z > 0) :
  x + y + z = 33 :=
sorry

end least_possible_value_l139_139888


namespace max_distance_to_line_l139_139464

theorem max_distance_to_line (P A : ℝ × ℝ) (l : ℝ → ℝ) (k : ℝ) :
  P = (-1, 3) → A = (2, 0) → l = (λ x, k * (x - 2)) →
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 :=
by
  intros hP hA hl
  use 3 * Real.sqrt 2
  sorry

end max_distance_to_line_l139_139464


namespace three_circles_on_common_sphere_l139_139231

theorem three_circles_on_common_sphere 
  (circle1 circle2 circle3 : ℝ)
  (r1 r2 r3 : ℝ)
  (h1 : r1 ≠ r2) 
  (h2 : r2 ≠ r3) 
  (h3 : r1 ≠ r3) 
  (center1 center2 center3: ℝ)
  (h4 : center1 < center2)
  (h5 : center2 < center3) 
  (h6 : center3 > center1) 
  (h7 : ∀ (x : ℝ), x ≠ center1 → x ≠ center2 → x ≠ center3) 
  (h8 : ∀ (x : ℝ), x ≠ r1 → x ≠ r2 → x ≠ r3) :
  ∃ (fold1 fold2 : straight_line),
  circles_aligned_on_common_sphere circle1 circle2 circle3 fold1 fold2 := 
begin
  sorry
end

end three_circles_on_common_sphere_l139_139231


namespace g_5_is_248_l139_139065

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l139_139065


namespace a_1000_value_l139_139363

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), 
    (a 1 = 2010) ∧
    (a 2 = 2011) ∧
    (∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n + 3) ∧
    (a 1000 = 2676) :=
by {
  -- sorry is used to skip the proof
  sorry 
}

end a_1000_value_l139_139363


namespace g_five_eq_248_l139_139050

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l139_139050


namespace algebra_or_drafting_not_both_l139_139450

theorem algebra_or_drafting_not_both {A D : Finset ℕ} (h1 : (A ∩ D).card = 10) (h2 : A.card = 24) (h3 : D.card - (A ∩ D).card = 11) : (A ∪ D).card - (A ∩ D).card = 25 := by
  sorry

end algebra_or_drafting_not_both_l139_139450


namespace compute_box_length_l139_139898

noncomputable def box_length (total_volume : ℝ) (cost_per_box : ℝ) (total_monthly_cost : ℝ) : ℝ :=
  let number_of_boxes := total_monthly_cost / cost_per_box
  let volume_per_box := total_volume / number_of_boxes
  real.cbrt volume_per_box

theorem compute_box_length :
  box_length 1080000 0.8 480 = 12.2 :=
by
  sorry

end compute_box_length_l139_139898


namespace log_equation_solution_l139_139973

theorem log_equation_solution (x : ℝ) (hx : log x 64 = log 3 27) : x = 4 := 
by
  sorry

end log_equation_solution_l139_139973


namespace second_lowest_mark_possibilities_l139_139119

theorem second_lowest_mark_possibilities :
  ∃ (marks : list ℤ), 
    marks.length = 6 ∧
    marks.sum / 6 = 74 ∧
    (∀ x ∈ marks, x = 76 → marks.count x = 2) ∧
    (∀ x ∈ marks, x ≠ 76 → marks.count x ≤ 1) ∧
    marks.sorted.nth 2 = 76 ∧
    marks.sorted.nth 3 = 76 ∧
    marks.head = 50 ∧
    marks.last = 94 ∧
    ∀ m ∈ marks, m ∈ (55..71) →
    marks.sum = 444 ∧
    m + (148 - m) = 148 →
    ∃ (M : ℕ), M = 17 :=
by
  sorry

end second_lowest_mark_possibilities_l139_139119


namespace estimate_population_mean_l139_139183

noncomputable def sample_data : List (ℤ × ℕ) := [(-2, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 1)]

def n : ℕ := 10

def sample_mean (data : List (ℤ × ℕ)) : ℚ :=
  let total := data.foldr (λ (a : ℤ × ℕ) (acc : ℚ), acc + (a.fst * a.snd : ℚ)) 0
  total / n

def sample_variance (data : List (ℤ × ℕ)) (mean : ℚ) : ℚ :=
  let sigma := data.foldr (λ (a : ℤ × ℕ) (acc : ℚ), acc + (a.snd : ℚ) * ((a.fst : ℚ) - mean)^2) 0
  sigma / (n - 1)

def t_value : ℚ := 2.262

def confidence_interval (mean : ℚ) (s : ℚ) : (ℚ × ℚ) :=
  let margin_of_error := t_value * (s / Real.sqrt n)
  (mean - margin_of_error, mean + margin_of_error)

theorem estimate_population_mean :
  let mean := sample_mean sample_data
  let variance := sample_variance sample_data mean
  let std_dev := Real.sqrt variance
  let (lower_bound, upper_bound) := confidence_interval mean std_dev
  0.363 < lower_bound ∧ upper_bound < 3.837 :=
by
  sorry

end estimate_population_mean_l139_139183


namespace ball_distribution_ways_l139_139405

theorem ball_distribution_ways :
  let R := 5
  let W := 3
  let G := 2
  let total_balls := 10
  let balls_in_first_box := 4
  ∃ (distributions : ℕ), distributions = (Nat.choose total_balls balls_in_first_box) ∧ distributions = 210 :=
by
  sorry

end ball_distribution_ways_l139_139405


namespace necessary_but_not_sufficient_l139_139367

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^n

def is_increasing_sequence (s : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, s n < s (n + 1)

def sum_first_n_terms (a : ℕ → ℝ) (q : ℝ) : ℕ → ℝ 
| 0 => a 0
| (n+1) => (sum_first_n_terms a q n) + (a 0 * q ^ n)

theorem necessary_but_not_sufficient (a : ℕ → ℝ) (q : ℝ) (h_geometric: is_geometric_sequence a q) :
  (q > 0) ∧ is_increasing_sequence (sum_first_n_terms a q) ↔ (q > 0)
:= sorry

end necessary_but_not_sufficient_l139_139367


namespace sqrt_meaningful_range_l139_139353

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x :=
by
sorry

end sqrt_meaningful_range_l139_139353


namespace tangent_line_at_neg_pi_l139_139454

noncomputable def f (x : ℝ) : ℝ := sin x / x

noncomputable def derivative_at (x : ℝ) : ℝ := (x * cos x - sin x) / x^2

theorem tangent_line_at_neg_pi : 
  derivative_at (-π) = 1 / π ∧ (∀ x y, y = (1 / π) * (x + π) → x - π * y + π = 0) :=
by
  have h₁ : derivative_at (-π) = 1 / π := sorry
  have h₂ : ∀ x y, y = (1 / π) * (x + π) → x - π * y + π = 0 := sorry
  exact ⟨h₁, h₂⟩

end tangent_line_at_neg_pi_l139_139454


namespace range_a_f_x_neg_l139_139687

noncomputable def f (a x : ℝ) : ℝ := x^2 + (2 * a - 1) * x - 3

theorem range_a_f_x_neg (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ f a x < 0) → a < 3 / 2 := sorry

end range_a_f_x_neg_l139_139687


namespace ellipse_equation_from_foci_arithmetic_mean_condition_l139_139304

theorem ellipse_equation_from_foci_arithmetic_mean_condition :
  let F1 := (-1, 0)
  let F2 := (1, 0)
  let P : ℝ × ℝ
  let f1_f2 := dist F1 F2
  let pf1 := dist P F1
  let pf2 := dist P F2
  f1_f2 = 2 ∧ P ∈ set_of (λ Q, (dist Q F1 + dist Q F2 = 4)) →
  ∃ x y : ℝ, (P = (x, y) ∧ (x^2 / 4 + y^2 / 3 = 1)) :=
by {
  sorry
}

end ellipse_equation_from_foci_arithmetic_mean_condition_l139_139304


namespace simplify_expression_l139_139445

theorem simplify_expression :
  (0.7264 * 0.4329 * 0.5478) + (0.1235 * 0.3412 * 0.6214) - ((0.1289 * 0.5634 * 0.3921) / (0.3785 * 0.4979 * 0.2884)) - (0.2956 * 0.3412 * 0.6573) = -0.3902 :=
by
  sorry

end simplify_expression_l139_139445


namespace find_x_l139_139331

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (4, x)

theorem find_x (x : ℝ) (h: ∠(vector_a x) (vector_b x) = π) : x = -2 :=
sorry

end find_x_l139_139331


namespace triangle_BC_l139_139361

theorem triangle_BC 
  (A B C : Type) 
  [InnerProductSpace ℝ A] 
  [C : CompleteSpace A]
  [NormedAddTorsor (Module ℝ) A]
  (angle_A : ∠ A = 90)
  (tan_B : Real.tan B = 5 / 12)
  (AB_len : ∥B - A∥ = 25) :
  ∥C - B∥ = 300 / 13 := 
sorry

end triangle_BC_l139_139361


namespace min_rectangles_to_cover_minimum_number_of_rectangles_required_l139_139511

-- Definitions based on the conditions
def corners_type1 : Nat := 12
def corners_type2 : Nat := 12

theorem min_rectangles_to_cover (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) : Nat :=
12

theorem minimum_number_of_rectangles_required (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) :
  min_rectangles_to_cover type1_corners type2_corners h1 h2 = 12 := by
  sorry

end min_rectangles_to_cover_minimum_number_of_rectangles_required_l139_139511


namespace problem_I_problem_II_l139_139320

-- Problem (I)
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) : 
  (f (x + 8) ≥ 10 - f x) ↔ (x ≤ -10 ∨ x ≥ 0) :=
sorry

-- Problem (II)
theorem problem_II (x y : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) 
(h_abs_x : |x| > 1) (h_abs_y : |y| < 1) :
  f y < |x| * f (y / x^2) :=
sorry

end problem_I_problem_II_l139_139320


namespace prove_Vens_are_Yins_l139_139738

variables (Zarb Yin Xip Won Ven : Type)
variable (A : Set Zarb)
variable (B : Set Alpha)

-- Conditions
variable (Zarbs_are_Yins : (A ⊆ B))
variable (Xips_are_Yins : (Xip ⊆ B))
variable (Wons_are_Xips : (Won ⊆ Xip))
variable (Vens_are_Zarbs : (C ⊆ A))

-- Goal
theorem prove_Vens_are_Yins : (Ven ⊆ B) :=
begin
  sorry
end

end prove_Vens_are_Yins_l139_139738


namespace all_digits_same_l139_139615

theorem all_digits_same (n : ℕ) (h : n > 0) (d : ℕ)
  (hd: (d < 10) ∧ (∀ k < (nat.log 10 (6^n + 1)) + 1, ((6^n + 1) / 10^k) % 10 = d)) :
  n = 1 ∨ n = 5 :=
sorry

end all_digits_same_l139_139615


namespace passengers_got_off_l139_139105

theorem passengers_got_off :
  ∀ (initial_boarded new_boarded final_left got_off : ℕ),
    initial_boarded = 28 →
    new_boarded = 7 →
    final_left = 26 →
    got_off = initial_boarded + new_boarded - final_left →
    got_off = 9 :=
by
  intros initial_boarded new_boarded final_left got_off h_initial h_new h_final h_got_off
  rw [h_initial, h_new, h_final] at h_got_off
  exact h_got_off

end passengers_got_off_l139_139105


namespace MaryHasBlueMarbles_l139_139609

-- Define the number of blue marbles Dan has
def DanMarbles : Nat := 5

-- Define the relationship of Mary's marbles to Dan's marbles
def MaryMarbles : Nat := 2 * DanMarbles

-- State the theorem that we need to prove
theorem MaryHasBlueMarbles : MaryMarbles = 10 :=
by
  sorry

end MaryHasBlueMarbles_l139_139609


namespace phil_wins_when_n_12_ellie_wins_when_n_2012_l139_139269

def ellie_wins (n : ℕ) : Prop :=
  ∃ M : matrix (fin n) (fin n) ℕ,
  ∀ (moves : ℕ) (config : matrix (fin n) (fin n) ℕ → matrix (fin n) (fin n) ℕ),
  ( ∀ i j, config moves M i j < n) → 
  (∀ i j, config moves M i j = 0)

theorem phil_wins_when_n_12 : ∀ M : matrix (fin 12) (fin 12) ℕ,
  (∑ i j, M i j) % 3 ≠ 0 → ∀ moves (config : matrix (fin 12) (fin 12) ℕ → matrix (fin 12) (fin 12) ℕ),
  ( ∀ i j, config moves M i j < 12) → 
  ¬ ( ∀ i j, config moves M i j = 0 ) := by
  sorry

theorem ellie_wins_when_n_2012 : ellie_wins 2012 := by 
  sorry

end phil_wins_when_n_12_ellie_wins_when_n_2012_l139_139269


namespace min_value_of_quadratic_function_l139_139990

-- Given the quadratic function y = x^2 + 4x - 5
def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 4*x - 5

-- Statement of the proof in Lean 4
theorem min_value_of_quadratic_function :
  ∃ (x_min y_min : ℝ), y_min = quadratic_function x_min ∧
  ∀ x : ℝ, quadratic_function x ≥ y_min ∧ x_min = -2 ∧ y_min = -9 :=
by
  sorry

end min_value_of_quadratic_function_l139_139990


namespace equidistant_points_quadratic_equidistant_points_area_equidistant_points_cubic_l139_139614

-- Part 1: 
theorem equidistant_points_quadratic (x y : ℝ) :
  y = x^2 + 2 * x ↔ y = x ∨ y = -x := 
sorry

-- Part 2: 
theorem equidistant_points_area (A B C : ℝ × ℝ) (b : ℝ) :
  (A = (-2, 2)) ∧ (B = (2, -2)) → 
  C = (b/2, b/2) → 
  2 * Real.sqrt 3 = abs(2 - (-2)) * abs(b/2) / 2 :=
sorry

-- Part 3: 
theorem equidistant_points_cubic (m : ℝ) :
  ∃ (x : ℝ), (x^2 + (1+m) * x + 2 * m + 2 = 0) ∧ 
  m < (-7-4*Real.sqrt 2) ∨ m > (-7+4*Real.sqrt 2) ∨ 
  ∃ (x : ℝ), (x^2 + (3+m) * x + 2 * m + 2 = 0) ∧ 
  m < (-9) ∨ m > (-1) :=
sorry

end equidistant_points_quadratic_equidistant_points_area_equidistant_points_cubic_l139_139614


namespace problem_part_I_problem_part_II_l139_139731

-- Definitions of the given conditions
variables {a b c A B C : ℝ}

-- Conditions as definitions in Lean
def triangle_conditions :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi ∧
  b = 2 ∧ 
  cos B = 1/4 ∧
  (cos A - 2 * cos C) / cos B = (2 * c - a) / b

-- Define the proof problem as two theorems to prove
theorem problem_part_I (h : triangle_conditions) :
  (Real.sin C / Real.sin A) = 2 :=
sorry

theorem problem_part_II (h : triangle_conditions) :
  (1 / 2) * a * c * Real.sin B = Real.sqrt 15 / 4 :=
sorry

end problem_part_I_problem_part_II_l139_139731


namespace total_birds_and_storks_l139_139896

theorem total_birds_and_storks
  (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ)
  (hb : initial_birds = 3) (hs : initial_storks = 4) (has : additional_storks = 6) :
  initial_birds + (initial_storks + additional_storks) = 13 :=
by
  sorry

end total_birds_and_storks_l139_139896


namespace number_of_ways_to_choose_three_socks_l139_139268

theorem number_of_ways_to_choose_three_socks (n : ℕ) (k : ℕ) (hn : n = 5) (hk : k = 3) : (nat.choose n k) = 10 :=
by
  rw [hn, hk]
  simp
  exact nat.choose_eq_factorial_div_factorial (by decide) (by decide)

end number_of_ways_to_choose_three_socks_l139_139268


namespace decimal_to_base8_conversion_l139_139604

theorem decimal_to_base8_conversion : (512 : ℕ) = 8^3 :=
by
  sorry

end decimal_to_base8_conversion_l139_139604


namespace minimum_value_graph_transformation_l139_139313

open Real

def f (x : ℝ) : ℝ := 3 * sin (1/2 * x + π/4) - 1

theorem minimum_value (x : ℝ) :
  (∀ k : ℤ, x = 4*k*π - 3*π/2 → f x = -4) ∧
  (∃ y : ℝ, f y = -4) :=
by
  sorry

theorem graph_transformation :
  ∀ y : ℝ, (∃ x : ℝ, y = sin x) →
  (∃ a b c d : ℝ, y = 3 * sin (1/2 * (x + π/4)) - 1) :=
by
  sorry

end minimum_value_graph_transformation_l139_139313


namespace work_completed_in_initial_days_l139_139757

theorem work_completed_in_initial_days (x : ℕ) : 
  (100 * x = 50 * 40) → x = 20 :=
by
  sorry

end work_completed_in_initial_days_l139_139757


namespace smallest_prime_x_l139_139106

-- Define prime number checker
def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem conditions and proof goal
theorem smallest_prime_x 
  (x y z : ℕ) 
  (hx : is_prime x)
  (hy : is_prime y)
  (hz : is_prime z)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hyz : y ≠ z)
  (hd : ∀ d : ℕ, d ∣ (x * x * y * z) ↔ (d = 1 ∨ d = x ∨ d = x * x ∨ d = y ∨ d = x * y ∨ d = x * x * y ∨ d = z ∨ d = x * z ∨ d = x * x * z ∨ d = y * z ∨ d = x * y * z ∨ d = x * x * y * z)) 
  : x = 2 := 
sorry

end smallest_prime_x_l139_139106


namespace restaurant_dinners_sold_on_Monday_l139_139435

theorem restaurant_dinners_sold_on_Monday (M : ℕ) 
  (h1 : ∀ tues_dinners, tues_dinners = M + 40) 
  (h2 : ∀ wed_dinners, wed_dinners = (M + 40) / 2)
  (h3 : ∀ thurs_dinners, thurs_dinners = ((M + 40) / 2) + 3)
  (h4 : M + (M + 40) + ((M + 40) / 2) + (((M + 40) / 2) + 3) = 203) : 
  M = 40 := 
sorry

end restaurant_dinners_sold_on_Monday_l139_139435


namespace sphere_radius_proportional_l139_139852

theorem sphere_radius_proportional
  (k : ℝ)
  (r1 r2 : ℝ)
  (W1 W2 : ℝ)
  (h_weight_area : ∀ (r : ℝ), W1 = k * (4 * π * r^2))
  (h_given1: W2 = 32)
  (h_given2: r2 = 0.3)
  (h_given3: W1 = 8):
  r1 = 0.15 := 
by
  sorry

end sphere_radius_proportional_l139_139852


namespace abs_neg_five_l139_139851

-- Definition of absolute value function
def abs (x : Int) : Int :=
  if x < 0 then -x else x

-- Main theorem to prove
theorem abs_neg_five : abs (-5) = 5 :=
  by
  -- Proof would go here, skipped with 'sorry'
  sorry

end abs_neg_five_l139_139851


namespace g_five_eq_248_l139_139051

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l139_139051


namespace largest_angle_of_consecutive_integers_hexagon_l139_139466

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l139_139466


namespace calculate_B_l139_139360
open Real

theorem calculate_B 
  (A B : ℝ) 
  (a b : ℝ) 
  (hA : A = π / 6) 
  (ha : a = 1) 
  (hb : b = sqrt 3) 
  (h_sin_relation : sin B = (b * sin A) / a) : 
  (B = π / 3 ∨ B = 2 * π / 3) :=
sorry

end calculate_B_l139_139360


namespace number_of_factors_n_l139_139414

def n : ℕ := 2^4 * 3^5 * 4^6 * 6^7

theorem number_of_factors_n :
  let num_factors (m : ℕ) := (divisors m).card
  num_factors n = 312 :=
by sorry

end number_of_factors_n_l139_139414


namespace sum_possible_k_is_neg_quarter_l139_139597

noncomputable def sum_of_possible_k (k : ℝ) : Prop :=
  let A := (0, 0)
  let B := (2, 2)
  let C := (8 * k, 0)
  (∃ k : ℝ, k > 0 ∧ y = k * x ∧ (y = k * x) divides the triangle formed by A, B, C into equal areas) ∧ 
  (let quad_eq := 8 * k^2 + 2 * k - 2 in
   k_roots := [(-2 + sqrt (4 + 64)) / 16, (-2 - sqrt (4 + 64)) / 16]
   sum k_roots = -1/4)

theorem sum_possible_k_is_neg_quarter (k : ℝ) : sum_of_possible_k k := 
begin
  sorry
end

end sum_possible_k_is_neg_quarter_l139_139597


namespace combined_transform_is_correct_l139_139257

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def reflection_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, -1]

def combined_transform (dilation_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  dilation_matrix dilation_factor * reflection_x_matrix

theorem combined_transform_is_correct :
  combined_transform 5 = !![5, 0; 0, -5] :=
by
  sorry

end combined_transform_is_correct_l139_139257


namespace relationship_between_alpha2_beta2_find_b_l139_139681
noncomputable theory

def roots_of_quadratic (b : ℝ) (α β : ℝ) : Prop :=
  α + β = -b ∧ α * β = 1

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem relationship_between_alpha2_beta2 (b : ℝ) (α β : ℝ) 
  (h1 : roots_of_quadratic b α β) : α^2 + β^2 ≥ 2 := sorry

theorem find_b (b : ℝ) (α β : ℝ) 
  (h1 : roots_of_quadratic b α β)
  (h2 : α > β)
  (h3 : is_isosceles_triangle (α^2 + β^2) (3*α - 3*β) (α*β)) : b = sqrt 5 ∨ b = -sqrt 5 ∨ b = sqrt 8 ∨ b = -sqrt 8 := sorry

end relationship_between_alpha2_beta2_find_b_l139_139681


namespace statement_B_false_l139_139955

def f (x : ℝ) : ℝ := 3 * x

def diamondsuit (x y : ℝ) : ℝ := abs (f x - f y)

theorem statement_B_false (x y : ℝ) : 3 * diamondsuit x y ≠ diamondsuit (3 * x) (3 * y) :=
by
  sorry

end statement_B_false_l139_139955


namespace arithmetic_sequence_sum_l139_139101

noncomputable def sum_first_ten_terms (a d : ℕ) : ℕ :=
  (10 / 2) * (2 * a + (10 - 1) * d)

theorem arithmetic_sequence_sum 
  (a d : ℕ) 
  (h1 : a + 2 * d = 8) 
  (h2 : a + 5 * d = 14) :
  sum_first_ten_terms a d = 130 :=
by
  sorry

end arithmetic_sequence_sum_l139_139101


namespace values_of_a_and_b_to_satisfy_condition_l139_139776

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a*x + b*cos x

theorem values_of_a_and_b_to_satisfy_condition :
  {a b : ℝ | ∃ S : set ℝ, S = {x | f x a b = 0} ∧ S.nonempty ∧ S = {x | f (f x a b) a b = 0}} 
  = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 < 4 ∧ p.2 = 0} :=
by
  sorry

end values_of_a_and_b_to_satisfy_condition_l139_139776


namespace necessary_but_not_sufficient_l139_139366

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^n

def is_increasing_sequence (s : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, s n < s (n + 1)

def sum_first_n_terms (a : ℕ → ℝ) (q : ℝ) : ℕ → ℝ 
| 0 => a 0
| (n+1) => (sum_first_n_terms a q n) + (a 0 * q ^ n)

theorem necessary_but_not_sufficient (a : ℕ → ℝ) (q : ℝ) (h_geometric: is_geometric_sequence a q) :
  (q > 0) ∧ is_increasing_sequence (sum_first_n_terms a q) ↔ (q > 0)
:= sorry

end necessary_but_not_sufficient_l139_139366


namespace solution_set_of_sine_inequality_l139_139490

theorem solution_set_of_sine_inequality :
  {x : ℝ // - (Real.sqrt 2) / 2 < Real.sin x ∧ Real.sin x ≤ 1 / 2} =
  ⋃ k : ℤ, (set.Ioo (-π / 4 + 2 * ↑k * π) (π / 6 + 2 * ↑k * π) ∪ 
            set.Ico (5 * π / 6 + 2 * ↑k * π) (5 * π / 4 + 2 * ↑k * π)) :=
by
  sorry

end solution_set_of_sine_inequality_l139_139490


namespace angles_equal_l139_139303

-- Definitions for geometric properties and the problem setup

variables {α : Type*} [linear_ordered_field α] [floor_ring α]
variables {circle : Type*} (P Q R S T : circle)
variables {M : point} -- Midpoint of arc BC

-- Note: Adjust or define the appropriate geometric constructs
-- and circle properties as needed

-- Definition: Midpoint of Arc
def is_midpoint_of_arc (M B C : point) : Prop :=
  -- Details that explain that M is midpoint of arc BC
  sorry

-- Definition: Concyclic Points
def concyclic (A B C M : point) : Prop :=
  -- Details that explain the points A, B, C, M lie on the same circle
  sorry

-- The theorem to prove
theorem angles_equal (A B C M : point) (h1 : concyclic A B C M) (h2 : is_midpoint_of_arc M B C) :
  ∠ BAM = ∠ MAC :=
by
  -- Proof placeholder
  sorry

end angles_equal_l139_139303


namespace number_of_valid_seating_arrangements_l139_139110

def valid_permutations (p : Perm (Fin 6)) : Prop :=
  ¬((p 0 = 1 ∧ p 1 = 2) ∨ (p 1 = 1 ∧ p 0 = 2)) ∧
  ¬((p 2 = 3 ∧ p 3 = 4) ∨ (p 3 = 3 ∧ p 2 = 4)) ∧
  ¬((p 4 = 5 ∧ p 5 = 6) ∨ (p 5 = 5 ∧ p 4 = 6))

theorem number_of_valid_seating_arrangements : 
  let p_set := {p : Perm (Fin 6) // valid_permutations p ∧ p 0 = 0} 
  in Fintype.card p_set = 16 := by
sorry

end number_of_valid_seating_arrangements_l139_139110


namespace nba_game_impossibility_l139_139499

theorem nba_game_impossibility (teams : ℕ) (games_per_team : ℕ) (inter_conference_ratio : ℕ) : 
  (teams = 30 ∧ games_per_team = 82 ∧ inter_conference_ratio = 2) →
  ¬ (∃ (k : ℕ) (x y z : ℕ),
        (teams / 2 = k) ∧ 
        (82 * k = 2 * x + z) ∧
        (x + y + z = 30 * 82 / 2) ∧
        (z = (x + y + z) / inter_conference_ratio)) :=
by {
  intros h,
  sorry
}

end nba_game_impossibility_l139_139499


namespace percentage_not_hawk_paddyfield_kingfisher_l139_139531

def goshawk_park_problem (B : ℕ) : ℕ :=
  let hawks := 0.30 * B
  let non_hawks := 0.70 * B
  let paddyfield_warblers := 0.40 * non_hawks
  let kingfishers := 0.25 * paddyfield_warblers
  let percentage_non_hawks_paddyfield_kingfishers := 0.30 + 0.28 + 0.07 in
  100 - (percentage_non_hawks_paddyfield_kingfishers * 100)

theorem percentage_not_hawk_paddyfield_kingfisher (B : ℕ) :
  goshawk_park_problem B = 35 := 
by
  sorry

end percentage_not_hawk_paddyfield_kingfisher_l139_139531


namespace fruits_left_l139_139758

theorem fruits_left (plums guavas apples given : ℕ) (h1 : plums = 16) (h2 : guavas = 18) (h3 : apples = 21) (h4 : given = 40) : 
  (plums + guavas + apples - given = 15) :=
by
  sorry

end fruits_left_l139_139758


namespace team_B_wins_first_game_prob_l139_139821

noncomputable def probability_B_wins_first_game (series_length : ℕ) (wins_for_A : ℕ) (wins_for_B : ℕ) 
  (unique_third_game_win_B : Bool) (finally_A_wins_series : Bool) : ℚ :=
if unique_third_game_win_B && finally_A_wins_series then
  2 / 3
else
  0

theorem team_B_wins_first_game_prob (series_length : ℕ) (wins_for_A : ℕ) (wins_for_B : ℕ) :
  (wins_for_A = 4) → (wins_for_B < 4) → (series_length = wins_for_A + wins_for_B) → 
  (∀ (g : ℕ), g ≠ 3 → (wins_for_B = g) ↔ team_B_wins g) → 
  (probability_B_wins_first_game series_length wins_for_A wins_for_B true true = 2 / 3) :=
by
  intros hA hB hSL hWins
  sorry

end team_B_wins_first_game_prob_l139_139821


namespace convex_hexagon_largest_angle_l139_139484

theorem convex_hexagon_largest_angle 
  (x : ℝ)                                 -- Denote the measure of the third smallest angle as x.
  (angles : Fin 6 → ℝ)                     -- Define the angles as a function from Fin 6 to ℝ.
  (h1 : ∀ i : Fin 6, angles i = x + (i : ℝ) - 3)  -- The six angles in increasing order.
  (h2 : 0 < x - 3 ∧ x - 3 < 180)           -- Convex condition: each angle is between 0 and 180.
  (h3 : angles ⟨0⟩ + angles ⟨1⟩ + angles ⟨2⟩ + angles ⟨3⟩ + angles ⟨4⟩ + angles ⟨5⟩ = 720) -- Sum of interior angles of a hexagon.
  : (∃ a, a = angles ⟨5⟩ ∧ a = 122.5) :=   -- Prove the largest angle in this arrangement is 122.5.
sorry

end convex_hexagon_largest_angle_l139_139484


namespace cost_of_one_unit_each_l139_139900

variables (x y z : ℝ)

theorem cost_of_one_unit_each
  (h1 : 2 * x + 3 * y + z = 130)
  (h2 : 3 * x + 5 * y + z = 205) :
  x + y + z = 55 :=
by
  sorry

end cost_of_one_unit_each_l139_139900


namespace find_parabola_line_passes_through_fixed_point_l139_139664

-- Definition of the parabola and point P along with the conditions
variables {p x y d₁ d₂ : ℝ} (hk : p > 0)
def parabola (x y : ℝ) : Prop := y^2 = 2 * p * x
def P_on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2
def distance_from_line (P : ℝ × ℝ) : ℝ := abs (P.1 - P.2 + 4) / real.sqrt 2
def distance_from_directrix (P : ℝ × ℝ) : ℝ := abs (P.1 + p / 2)
def min_value_d1_d2 (P : ℝ × ℝ) : Prop := distance_from_line P + distance_from_directrix P = 3 * real.sqrt 2

-- Theorem 1: Finding the equation of the parabola
theorem find_parabola : ∃ (p : ℝ), y^2 = 2 * p * x ∧ p = 4 := sorry

-- Definition of the lines l₁ and l₂ and intersection points A, B, C, D
variables {k₁ k₂ : ℝ} (hk₁k₂ : k₁ * k₂ = -3 / 2)
def line_l₁ (x y : ℝ) : Prop := y = k₁ * (x - 1)
def line_l₂ (x y : ℝ) : Prop := y = k₂ * (x - 1)
def intersects_parabola (k : ℝ) (x y : ℝ) : Prop := y^2 = 8 * x ∧ y = k * (x - 1)

-- Proving the line always passes through a fixed point
theorem line_passes_through_fixed_point (k x y : ℝ) :
  ∃ k x y, (k₁ * k₂ = -3 / 2) → (kx - y - k * k₁ - k * k₂ = 0) → (x = 0 → y = 3 / 2) := sorry

end find_parabola_line_passes_through_fixed_point_l139_139664


namespace orthocenters_concyclic_l139_139778

theorem orthocenters_concyclic 
  (A₁ A₂ A₃ A₄ : Point)
  (O : Circle)
  (H₁ H₂ H₃ H₄ : Point)
  (h_cyclic : InscribedIn A₁ A₂ A₃ A₄ O)
  (hH₁ : Orthocenter H₁ A₂ A₃ A₄)
  (hH₂ : Orthocenter H₂ A₃ A₄ A₁)
  (hH₃ : Orthocenter H₃ A₄ A₁ A₂)
  (hH₄ : Orthocenter H₄ A₁ A₂ A₃) :
  ∃ (circle : Circle), Concyclic H₁ H₂ H₃ H₄ circle ∧ center circle = A₁ + A₂ + A₃ + A₄ :=
by
  sorry

end orthocenters_concyclic_l139_139778


namespace option_C_incorrect_l139_139222

variable (a b : ℝ)

theorem option_C_incorrect : ((-a^3)^2 * (-b^2)^3) ≠ (a^6 * b^6) :=
by {
  sorry
}

end option_C_incorrect_l139_139222


namespace gasoline_needed_l139_139548

theorem gasoline_needed (D : ℕ) 
    (fuel_efficiency : ℕ) 
    (fuel_efficiency_proof : fuel_efficiency = 20)
    (gallons_for_130km : ℕ) 
    (gallons_for_130km_proof : gallons_for_130km = 130 / 20) :
    (D : ℕ) / fuel_efficiency = (D : ℕ) / 20 :=
by
  -- The proof is omitted as per the instruction
  sorry

end gasoline_needed_l139_139548


namespace mod_inverse_5_221_l139_139245

theorem mod_inverse_5_221 : ∃ x : ℤ, 0 ≤ x ∧ x < 221 ∧ (5 * x) % 221 = 1 % 221 :=
by
  use 177
  sorry

end mod_inverse_5_221_l139_139245


namespace find_g5_l139_139077

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l139_139077


namespace find_c_l139_139884

-- Given conditions
variables {a b c d e : ℕ} (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e)
variables (h6 : a + b = e - 1) (h7 : a * b = d + 1)

-- Required to prove
theorem find_c : c = 4 := by
  sorry

end find_c_l139_139884


namespace count_even_three_digit_numbers_with_sum_9_l139_139708

theorem count_even_three_digit_numbers_with_sum_9 : 
  (∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 2 = 0 ∧ (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in d2 + d3 = 9) ∧
  (∀ k, 100 ≤ k ∧ k < 1000 ∧ k % 2 = 0 ∧ (let d1 := k / 100, d2 := (k / 10) % 10, d3 := k % 10 in d2 + d3 = 9) ↔ n = k)) → 
  (Set.count {n | 100 ≤ n ∧ n < 1000 ∧ n % 2 = 0 ∧ (let d2 := (n / 10) % 10, d3 := n % 10 in d2 + d3 = 9)} = 45) :=
by
  sorry

end count_even_three_digit_numbers_with_sum_9_l139_139708


namespace value_of_y_l139_139089

noncomputable def line_equation (x : ℝ) : ℝ := 3 * x + 5

theorem value_of_y (x : ℝ) : (line_equation x = 3 * x + 5) → line_equation 50 = 155 :=
by
  intro h
  calc 
    line_equation 50 = 3 * 50 + 5 : h
                ... = 150 + 5 : by rw mul_add
                ... = 155 : by rw add_comm

end value_of_y_l139_139089


namespace solution_to_equation_l139_139146

theorem solution_to_equation (p : ℝ) (h : 0 ≤ p ∧ p ≤ 4 / 3) :
  let x := (4 - p) / Real.sqrt (8 * (2 - p))
  in sqrt (x^2 - p) + 2 * sqrt (x^2 - 1) = x :=
by
  sorry

end solution_to_equation_l139_139146


namespace minimum_possible_value_of_sum_l139_139617

noncomputable def minimum_value_of_sum (a b c : ℝ) : ℝ :=
  (if a > 0 ∧ b > 0 ∧ c > 0 then 3 * (1 / (105 : ℝ) ^ (1 / 3 : ℝ)) else 0)

theorem minimum_possible_value_of_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a / (3 * b) + b / (5 * c) + c / (7 * a)) ≥ minimum_value_of_sum a b c :=
begin
  sorry -- Proof is omitted as per instructions
end

end minimum_possible_value_of_sum_l139_139617


namespace congruent_polygons_in_grid_l139_139280

theorem congruent_polygons_in_grid :
  ∀ (n m : ℕ) (segments : ℕ) (polygons : ℕ), 
    n = 10 →
    m = 10 → 
    segments = 80 → 
    polygons = 20 → 
    ∀ i j : ℕ, (i < polygons) → (j < polygons) → 
    congruent_polygons n m segments polygons i j := 
by 
  intros n m segments polygons h_n h_m h_segments h_polygons i j h_i h_j
  sorry

end congruent_polygons_in_grid_l139_139280


namespace hyperbola_asymptotes_l139_139514

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - 4 * y^2 = 1) → (x = 2 * y ∨ x = -2 * y) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l139_139514


namespace projects_contracting_total_ways_l139_139207

theorem projects_contracting_total_ways : 
  let C (n k : ℕ) := (nat.choose n k)
  (C 8 3 * C 5 1 * C 4 2 * C 2 2 = 1680) := 
by {
  let C := λ n k, nat.choose n k,
  sorry
}

end projects_contracting_total_ways_l139_139207


namespace black_beans_count_l139_139545

theorem black_beans_count (B G O : ℕ) (h₁ : G = B + 2) (h₂ : O = G - 1) (h₃ : B + G + O = 27) : B = 8 := by
  sorry

end black_beans_count_l139_139545


namespace volume_conversion_l139_139567

theorem volume_conversion (v_feet : ℕ) (h : v_feet = 250) : (v_feet / 27 : ℚ) = 250 / 27 := by
  sorry

end volume_conversion_l139_139567


namespace median_BC_equation_altitude_BC_equation_l139_139375

noncomputable def coordsA : (ℝ × ℝ) := (2, 3)
noncomputable def coordsB : (ℝ × ℝ) := (1, -3)
noncomputable def coordsC : (ℝ × ℝ) := (-3, -1)

-- Define point and line structures
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define vertices of the triangle
def A : Point := ⟨2, 3⟩
def B : Point := ⟨1, -3⟩
def C : Point := ⟨-3, -1⟩

-- Statement: (I) The equation of the median drawn to side BC
theorem median_BC_equation :
  ∃ a b c : ℝ, (a = 5) ∧ (b = -3) ∧ (c = -1) ∧ (∀ P : Point, P.x * a + P.y * b + c = 0 ↔ LineThrough P A ∧ LineThrough P (midpoint B C))
:= sorry

-- Statement: (II) The equation of the altitude drawn to side BC
theorem altitude_BC_equation :
  ∃ a b c : ℝ, (a = 2) ∧ (b = -1) ∧ (c = -1) ∧ (∀ P : Point, P.x * a + P.y * b + c = 0 ↔ LineThrough P A ∧ PerpendicularLine P B C)
:= sorry

-- Definitions to support points on the line and other geometric concepts
def LineThrough (P Q : Point) : Prop :=
  -- Implementation of line-through condition
  sorry

def midpoint (P Q : Point) : Point :=
  -- Implementation for calculating the midpoint
  sorry

def PerpendicularLine (P : Point) (Q R : Point) : Prop :=
  -- Implementation for checking perpendicularity of the line through P with line QR
  sorry

end median_BC_equation_altitude_BC_equation_l139_139375


namespace sum_of_m_and_n_l139_139341

theorem sum_of_m_and_n (m n : ℚ) (h : (m - 3) * (Real.sqrt 5) + 2 - n = 0) : m + n = 5 :=
sorry

end sum_of_m_and_n_l139_139341


namespace find_positive_integers_l139_139975

theorem find_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^2 - Nat.factorial y = 2019 ↔ x = 45 ∧ y = 3 :=
by
  sorry

end find_positive_integers_l139_139975


namespace initial_days_to_complete_work_l139_139905

theorem initial_days_to_complete_work (original_men absent_men remaining_days : ℕ) :
  original_men = 15 →
  absent_men = 5 →
  remaining_days = 60 →
  ∃ D : ℕ, original_men * D = (original_men - absent_men) * remaining_days ∧ D = 40 :=
begin
  intros h_orig h_absent h_days,
  use 40,
  split,
  {
    rw [h_orig, h_absent, h_days],
    norm_num,
  },
  {
    refl,
  }
end

end initial_days_to_complete_work_l139_139905


namespace arithmetic_sequence_problem_l139_139993

-- Definitions of arithmetic sequences
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition of arithmetic sequences and their sum ratio
def condition (a b : ℕ → ℚ) : Prop :=
  is_arithmetic_sequence a ∧
  is_arithmetic_sequence b ∧
  ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, a (i + 1)) / (∑ i in Finset.range n, b (i + 1)) = (7 * n + 2) / (n + 3)

-- Prove that a_5 / b_5 = 65/12
theorem arithmetic_sequence_problem (a b : ℕ → ℚ)
  (h : condition a b) :
  a 5 / b 5 = 65 / 12 :=
sorry

end arithmetic_sequence_problem_l139_139993


namespace soccer_lineup_count_l139_139022

theorem soccer_lineup_count :
  let total_players : ℕ := 16
  let total_starters : ℕ := 7
  let m_j_players : ℕ := 2 -- Michael and John
  let other_players := total_players - m_j_players
  let total_ways : ℕ :=
    2 * Nat.choose other_players (total_starters - 1) + Nat.choose other_players (total_starters - 2)
  total_ways = 8008
:= sorry

end soccer_lineup_count_l139_139022


namespace triangle_geometric_sequence_sine_rule_l139_139359

noncomputable def sin60 : Real := Real.sqrt 3 / 2

theorem triangle_geometric_sequence_sine_rule 
  {a b c : Real} 
  {A B C : Real} 
  (h1 : a / b = b / c) 
  (h2 : A = 60 * Real.pi / 180) :
  b * Real.sin B / c = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_geometric_sequence_sine_rule_l139_139359


namespace system_of_equations_has_four_real_solutions_l139_139339

theorem system_of_equations_has_four_real_solutions :
  ∃ (S : set (ℝ × ℝ)), 
    (∀ (x y : ℝ), ((x^2 + y = 5) ∧ (x + y^2 = 3)) ↔ ((x, y) ∈ S)) ∧
    S.finite ∧ 
    S.card = 4 :=
by
  sorry

end system_of_equations_has_four_real_solutions_l139_139339


namespace octal_to_decimal_l139_139185

theorem octal_to_decimal (a b c : ℕ) (h : a * 8^2 + b * 8^1 + c * 8^0 = 468) : 
  let n := 724 in 
  a = 7 ∧ b = 2 ∧ c = 4 ∧ n = 724 ∧ a * 8^2 + b * 8^1 + c * 8^0 = 468 := by
  sorry

end octal_to_decimal_l139_139185


namespace probability_multiple_of_3_or_4_l139_139841

theorem probability_multiple_of_3_or_4 : ((15 : ℚ) / 30) = (1 / 2) := by
  sorry

end probability_multiple_of_3_or_4_l139_139841


namespace net_total_expense_l139_139188

def cases_april_soda_A := 100
def cases_april_soda_B := 50
def price_per_bottle_soda_A := 1.5
def price_per_bottle_soda_B := 2.0
def discount_april_soda_A := 0.15

def cases_may_soda_A := 80
def cases_may_soda_B := 40
def sales_tax_may := 0.10

def cases_june_soda_A := 120
def cases_june_soda_B := 60
def special_discount_june := 0.10

def bottles_per_case_soda_A := 24
def bottles_per_case_soda_B := 30

theorem net_total_expense : 
  let
    price_april_soda_A := (cases_april_soda_A * bottles_per_case_soda_A * price_per_bottle_soda_A) * (1 - discount_april_soda_A)
    price_april_soda_B := cases_april_soda_B * bottles_per_case_soda_B * price_per_bottle_soda_B
    total_april := price_april_soda_A + price_april_soda_B
    
    price_may_soda_A := cases_may_soda_A * bottles_per_case_soda_A * price_per_bottle_soda_A
    price_may_soda_B := cases_may_soda_B * bottles_per_case_soda_B * price_per_bottle_soda_B
    total_may_without_tax := price_may_soda_A + price_may_soda_B
    total_may := total_may_without_tax * (1 + sales_tax_may)
    
    price_june_soda_A := cases_june_soda_A * bottles_per_case_soda_A * price_per_bottle_soda_A
    price_june_soda_B := cases_june_soda_B * bottles_per_case_soda_B * price_per_bottle_soda_B
    discount_june_soda_A := (bottles_per_case_soda_A * price_per_bottle_soda_A) * special_discount_june
    total_june := (price_june_soda_A - (cases_june_soda_B * discount_june_soda_A)) + price_june_soda_B
    
    net_total := total_april + total_may + total_june
  in
  net_total = 19572 :=
by
  sorry

end net_total_expense_l139_139188


namespace train_speed_excluding_stoppages_l139_139240

theorem train_speed_excluding_stoppages (S : ℝ) (speed_with_stoppages : ℝ = 40) (stoppage_time_per_hour : ℝ = 10 / 60) : S = 48 :=
by
  -- Given conversion rates and the problem's conditions
  let running_time_per_hour : ℝ := 1 - stoppage_time_per_hour -- Remaining running time per hour
  have eq1 : running_time_per_hour = 50 / 60 := by norm_num
  have eq2 : running_time_per_hour = 5 / 6 := by norm_num
  rw [eq2] at eq1
  -- Express the relationship as given
  have proportion := (5 / 6) * S = speed_with_stoppages
  -- Solve for S
  have solve_for_S : S = 48 := by norm_num
  exact solve_for_S

end train_speed_excluding_stoppages_l139_139240


namespace inequality_proof_l139_139096

theorem inequality_proof (n : ℕ) (h : n ≥ 2) :
  let S_n := 3*n - 2*n^2,
      a_1 := 1,
      a_n := -4*n + 5 in
  n * a_1 > S_n ∧ S_n > n * a_n :=
by
  let S_n := 3 * n - 2 * n ^ 2
  let a_1 := 1
  let a_n := -4 * n + 5
  exact sorry

end inequality_proof_l139_139096


namespace area_original_l139_139117

noncomputable def area_original_figure (a b : ℝ) : ℝ :=
  a * b * Real.sin (120 * Real.pi / 180)

theorem area_original {S_intuitive : ℝ} (h1 : 0 < S_intuitive)
    (h2 : S_intuitive = (1 / 2) * 4 * 4 * Math.sin (real.pi / 3))
    (h3 : 2 * real.sqrt 2 * S_intuitive = 2 * real.sqrt 2 * S_intuitive) :
    (area_original_figure 2 (4 * √3) = 8 * √6) :=
begin
  sorry,
end

end area_original_l139_139117


namespace line_not_intersect_graph_tan_l139_139910

theorem line_not_intersect_graph_tan :
  ∀ k : ℤ, x ≠ (k * (π / 2) + π / 8) → ¬(∃ y : ℝ, y = tan (2 * x + π / 4)) :=
by {
  sorry -- the proof is omitted.
}

end line_not_intersect_graph_tan_l139_139910


namespace fido_yard_area_fraction_l139_139628

theorem fido_yard_area_fraction (r : ℝ) (h : r > 0) :
  let square_area := (2 * r)^2
  let reachable_area := π * r^2
  let fraction := reachable_area / square_area
  ∃ a b : ℕ, (fraction = (Real.sqrt a) / b * π) ∧ (a * b = 4) := by
  sorry

end fido_yard_area_fraction_l139_139628


namespace pentagon_diagonal_probability_l139_139766

theorem pentagon_diagonal_probability :
  let S := set (fin 10) in
  (∃ s1 s2 ∈ S, s1 ≠ s2 ∧ 
   let same_length :=
     (s1 < 5 ∧ s2 < 5) ∨ (s1 ≥ 5 ∧ s2 ≥ 5) in
   (∑ x in S, ∑ y in (S.erase x), if same_length then 1 else 0) = 4 / 9 * (|S| * (|S| - 1))) := sorry

end pentagon_diagonal_probability_l139_139766


namespace least_positive_integer_l139_139125

theorem least_positive_integer (n : ℕ) (h1 : n > 1)
  (h2 : n % 3 = 2) (h3 : n % 4 = 2) (h4 : n % 5 = 2) (h5 : n % 11 = 2) :
  n = 662 :=
sorry

end least_positive_integer_l139_139125


namespace length_BI_in_triangle_l139_139388

theorem length_BI_in_triangle (A B C I D E F : Type)
  (AB AC BC : ℝ) (hAB : AB = 10) (hAC : AC = 12) (hBC : BC = 8)
  (incircle_touches : incenter_touches_sides AB AC BC I D E F) : 
  distance B I = 4 :=
sorry

end length_BI_in_triangle_l139_139388


namespace area_of_triangle_ABC_l139_139751

-- Define the points A, B, and C
def A : ℝ × ℝ := (7, 8)
def B : ℝ × ℝ := (10, 4)
def C : ℝ × ℝ := (2, -4)

-- Define the formula for the area of a triangle given three points in the plane using determinant method.
def area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2)))

-- Statement of the problem
theorem area_of_triangle_ABC : area A B C = 28 := by
  simp [area, A, B, C]
  ring_nf
  norm_num
  sorry

end area_of_triangle_ABC_l139_139751


namespace formulate_numbers_with_twos_l139_139116

theorem formulate_numbers_with_twos :
  (∃ a1 a2 a3 a4 a5 : ℕ,
    a1 = ((2 / 2) * ((2 / 2) ^ 2)) ∧
    a2 = ((2 / 2) + ((2 / 2) ^ 2)) ∧
    a3 = (2 + ((2 / 2) * (2 / 2))) ∧
    a4 = ((2 + 2) * ((2 / 2) ^ 2)) ∧
    a5 = (2 + 2 + ((2 / 2) ^ 2)) ∧
    a1 = 1 ∧
    a2 = 2 ∧
    a3 = 3 ∧
    a4 = 4 ∧
    a5 = 5) :=
begin
  sorry
end

end formulate_numbers_with_twos_l139_139116


namespace total_tshirts_bought_l139_139218

-- Given conditions
def white_packs : ℕ := 3
def white_tshirts_per_pack : ℕ := 6
def blue_packs : ℕ := 2
def blue_tshirts_per_pack : ℕ := 4

-- Theorem statement: Total number of T-shirts Dave bought
theorem total_tshirts_bought : white_packs * white_tshirts_per_pack + blue_packs * blue_tshirts_per_pack = 26 := by
  sorry

end total_tshirts_bought_l139_139218


namespace winnie_balloons_l139_139881

-- Definition of the problem's states and conditions
def total_balloons : ℕ := 22 + 44 + 78 + 90 -- Total number of balloons
def number_of_friends : ℕ := 10 -- Number of friends

-- The Lean 4 statement of the proof problem
theorem winnie_balloons : 234 % number_of_friends = 4 := by
  -- Calculation step
  have h : total_balloons = 234 := by rfl
  -- Now we can state the final theorem directly
  rw [h]
  exact Nat.mod_eq_of_lt (by norm_num) -- Check that the modulus operation yields 4

end winnie_balloons_l139_139881


namespace largest_angle_in_consecutive_integer_hexagon_l139_139477

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l139_139477


namespace determine_a_square_binomial_l139_139623

theorem determine_a_square_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, (4 * x^2 + 16 * x + a) = (2 * x + b)^2) → a = 16 := 
by
  sorry

end determine_a_square_binomial_l139_139623


namespace triangle_solutions_l139_139730

noncomputable def number_of_solutions (a b A : ℝ) : ℕ :=
if (b * b) * (3 / 4) - 2 * a * b * (1 / 2) + a * a > 0 then 2 else
if (b * b) * (3 / 4) - 2 * a * b * (1 / 2) + a * a = 0 then 1 else 0

theorem triangle_solutions :
  number_of_solutions 18 24 (real.pi / 6) = 2 :=
sorry

end triangle_solutions_l139_139730


namespace monotonically_increasing_interval_l139_139317

theorem monotonically_increasing_interval (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : ∃ x, ∀ y, f y ≤ f x) :
  {x | x^2 - 2*x > 0} = {x : ℝ | x < 0 ∨ x > 2} :=
by
  sorry

end monotonically_increasing_interval_l139_139317


namespace always_exists_triangle_l139_139670

variable (a1 a2 a3 a4 d : ℕ)

def arithmetic_sequence (a1 a2 a3 a4 d : ℕ) :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def positive_terms (a1 a2 a3 a4 : ℕ) :=
  a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0

theorem always_exists_triangle (a1 a2 a3 a4 d : ℕ)
  (h1 : arithmetic_sequence a1 a2 a3 a4 d)
  (h2 : d > 0)
  (h3 : positive_terms a1 a2 a3 a4) :
  a2 + a3 > a4 ∧ a2 + a4 > a3 ∧ a3 + a4 > a2 :=
sorry

end always_exists_triangle_l139_139670


namespace sum_five_smallest_primes_l139_139135

theorem sum_five_smallest_primes : (2 + 3 + 5 + 7 + 11) = 28 := by
  -- We state the sum of the known five smallest prime numbers.
  sorry

end sum_five_smallest_primes_l139_139135


namespace area_ratio_l139_139952

-- Definitions
def S3 : set (ℝ × ℝ) := {p : ℝ × ℝ | log 10 (3 + p.1^2 + p.2^2) ≤ log 10 (90 * p.1 + 90 * p.2)}
def S4 : set (ℝ × ℝ) := {p : ℝ × ℝ | log 10 (4 + p.1^2 + p.2^2) ≤ log 10 (9 * (p.1 + p.2) + 100)}

-- Proof Problem Statement
theorem area_ratio (hS3 : S3 = {p | (p.1 - 45)^2 + (p.2 - 45)^2 ≤ 4050})
  (hS4 : S4 = {p | (p.1 - 9/2)^2 + (p.2 - 9/2)^2 ≤ 113}) :
  let area_S3 := π * 4050,
      area_S4 := π * 113
  in ratio := area_S4 / area_S3 = 113 / 4050 :=
sorry

end area_ratio_l139_139952


namespace problem_1_problem_2_problem_3_l139_139689

def f (x m : ℝ) : ℝ := x^2 - 2 * m * x + 10

theorem problem_1 {m : ℝ} (h1 : m > 1) (h2 : f m m = 1) :
  ∃ m, m = 3 ∧ ∀ x, f x 3 = x^2 - 6 * x + 10 :=
by sorry

theorem problem_2 {m : ℝ} (h1 : m ≥ 2) (h2 : ∀ x1 x2 ∈ set.Icc 1 (m + 1), abs (f x1 m - f x2 m) ≤ 9) :
  m ∈ set.Icc 2 4 :=
by sorry

theorem problem_3 {m : ℝ} (h1 : ∃ x ∈ set.Icc 3 5, f x m = 0) :
  m ∈ set.Icc (Real.sqrt 10) (7 / 2) :=
by sorry

end problem_1_problem_2_problem_3_l139_139689


namespace esperanzas_rent_l139_139021

def gross_monthly_salary := 4840
def rent_amount : ℝ := 600
def amount_on_food (R : ℝ) := (3/5) * R
def mortgage_bill (R : ℝ) := 3 * amount_on_food R
def savings := 2000
def taxes := (2/5) * savings
def total_expenses_and_savings (R : ℝ) := R + amount_on_food R + mortgage_bill R + savings + taxes

theorem esperanzas_rent : ∃ R : ℝ, total_expenses_and_savings R = gross_monthly_salary ∧ R = rent_amount :=
by {
  exists 600,
  calc
    total_expenses_and_savings 600 = 600 + (3/5) * 600 + 3 * (3/5) * 600 + 2000 + 800 : by rfl
    ... = 600 + 360 + 1080 + 2000 + 800 : by norm_num
    ... = 4840 : by norm_num,
  split; norm_num,
  sorry
}

end esperanzas_rent_l139_139021


namespace sum_of_squares_and_product_pos_ints_l139_139492

variable (x y : ℕ)

theorem sum_of_squares_and_product_pos_ints :
  x^2 + y^2 = 289 ∧ x * y = 120 → x + y = 23 :=
by
  intro h
  sorry

end sum_of_squares_and_product_pos_ints_l139_139492


namespace seating_arrangements_count_l139_139899

-- Assuming no two spectators are adjacent and each can have at 
-- most two empty seats on either side, prove there are 42 ways.
theorem seating_arrangements_count :
  ∃(arrangements : ℕ), 
    (let seats := 9 in
     let spectators := 3 in
     let min_gaps := 1 in
     let max_gaps := 2 in
     arrangements = 42) :=
sorry

end seating_arrangements_count_l139_139899


namespace sequence_count_is_840_l139_139443

def count_sequences_with_subpatterns :
  (Σ (seq : list bool),
    (count_subsequence seq [tt, tt] = 2) ∧
    (count_subsequence seq [ff, ff] = 6) ∧
    (count_subsequence seq [tt, ff] = 4) ∧
    (count_subsequence seq [ff, tt] = 4))
  → ℕ :=
  sorry

theorem sequence_count_is_840 :
  (count_sequences_with_subpatterns (list.replicate 16 bool.tt)) = 840 :=
  sorry

end sequence_count_is_840_l139_139443


namespace quadratic_distinct_real_roots_l139_139723

theorem quadratic_distinct_real_roots (k : ℝ) :
  ((k - 1) * x^2 + 6 * x + 3 = 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k - 1) * x1^2 + 6 * x1 + 3 = 0) ∧ ((k - 1) * x2^2 + 6 * x2 + 3 = 0)) ↔ (k < 4 ∧ k ≠ 1) :=
by {
  sorry
}

end quadratic_distinct_real_roots_l139_139723


namespace largest_angle_of_consecutive_integers_in_hexagon_l139_139471

theorem largest_angle_of_consecutive_integers_in_hexagon : 
  ∀ (a : ℕ), 
    (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) = 720 → 
    a + 3 = 122.5 :=
by sorry

end largest_angle_of_consecutive_integers_in_hexagon_l139_139471


namespace expected_value_winnings_l139_139904

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def winnings (n : ℕ) : ℕ :=
  if is_prime n then 3 else 0

/-- The expected value of winnings when rolling a fair 8-sided die. -/
theorem expected_value_winnings : 
  (∑ i in Finset.range 8, (if is_prime (i + 1) then 3 else 0) / 8 : ℚ) = 1.5 :=
by
  sorry

end expected_value_winnings_l139_139904


namespace angle_CBD_measure_l139_139381

theorem angle_CBD_measure
  (ABC_line : StraightLine ABC)
  (angle_BAD : ∠BAD = 35)
  (angle_BDA : ∠BDA = 45)
  (angle_ABC : ∠ABC = 58) :
  ∠CBD = 42 := 
sorry

end angle_CBD_measure_l139_139381


namespace radius_product_l139_139290

-- Definitions of circles
def C1 (x y r1 : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = r1^2
def C2 (x y r2 : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = r2^2

-- Tangency condition
def are_tangent (r1 r2 : ℝ) : Prop := (r1 + r2) / sqrt (50) = 3 * sqrt (2)

-- Slope condition for the common tangent
def has_slope (slope : ℝ) : Prop := slope = 7

theorem radius_product (r1 r2 : ℝ) (h1 : r1 > 0) (h2 : r2 > 0) 
  (c1 : C1 2 2 r1) (c2 : C2 (-1) (-1) r2) (tangent : are_tangent r1 r2) 
  (slope_cond : has_slope 7) : 
  r1 * r2 = 72 / 25 :=
sorry

end radius_product_l139_139290


namespace not_sixth_power_of_integer_l139_139441

theorem not_sixth_power_of_integer (n : ℕ) : ¬ ∃ k : ℤ, 6 * n^3 + 3 = k^6 :=
by
  sorry

end not_sixth_power_of_integer_l139_139441


namespace polar_equation_of_circle_chord_length_range_l139_139744

theorem polar_equation_of_circle :
  let o := (Real.sqrt 2, Real.pi / 4)
  let r := Real.sqrt 3
  ∀ (ρ θ : ℝ), ((ρ - Real.sqrt 2 * Real.cos θ - Real.sqrt 2 * Real.sin θ)^2 = r^2) ↔ 
  (ρ^2 - 2 * ρ * (Real.cos θ + Real.sin θ) - 1 = 0) 
:= sorry

theorem chord_length_range (α : ℝ) :
    0 ≤ α ∧ α < Real.pi / 4 →
    let t := λ (t : ℝ), ∃ (x y : ℝ), x = 2 + t * Real.cos α ∧ y = 2 + t * Real.sin α ∧ (x - 1)^2 + (y - 1)^2 = 3 
    ∀ (t1 t2 : ℝ), t t1 ∧ t t2 → (t1 + t2 = -2 * (Real.cos α + Real.sin α)) ∧ (t1 * t2 = -1) → 
    let AB := 2 * Real.sqrt (2 + Real.sin (2 * α))
    2 * Real.sqrt 2 ≤ AB ∧ AB < 2 * Real.sqrt 3 
:= sorry

end polar_equation_of_circle_chord_length_range_l139_139744


namespace cos_angle_l139_139327

open Real

variable (a b : ℝ × ℝ) (theta : ℝ)

-- Vectors given
def vector_a := (1, 1)
def vector_b := (2, 0)

-- Dot product of a and b
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Magnitude of vector
def magnitude (u : ℝ × ℝ) : ℝ :=
  sqrt (u.1^2 + u.2^2)

-- The cosine of the angle between vectors a and b
def cos_theta (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude a * magnitude b)

theorem cos_angle:
  cos_theta vector_a vector_b = 1 / sqrt 2 := sorry

end cos_angle_l139_139327


namespace specific_time_l139_139849

theorem specific_time :
  (∀ (s : ℕ), 0 ≤ s ∧ s ≤ 7 → (∃ (t : ℕ), (t ^ 2 + 2 * t) - (3 ^ 2 + 2 * 3) = 20 ∧ t = 5)) :=
  by sorry

end specific_time_l139_139849


namespace basketball_team_selection_l139_139906

theorem basketball_team_selection :
  ∑ k in { k | k ≤ 5 }, (nat.choose 16 k) - 2 * (nat.choose 14 3) + (nat.choose 12 1) = 3652 := 
sorry

end basketball_team_selection_l139_139906


namespace cost_reduction_l139_139486

variable (a : ℝ) -- original cost
variable (p : ℝ) -- percentage reduction (in decimal form)
variable (m : ℕ) -- number of years

def cost_after_years (a p : ℝ) (m : ℕ) : ℝ :=
  a * (1 - p) ^ m

theorem cost_reduction (a p : ℝ) (m : ℕ) :
  m > 0 → cost_after_years a p m = a * (1 - p) ^ m :=
sorry

end cost_reduction_l139_139486


namespace f_g_minus_g_f_l139_139775

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem f_g_minus_g_f : f(g(2)) - g(f(2)) = 1 := by
  sorry

end f_g_minus_g_f_l139_139775


namespace combinatorial_identity_solution_l139_139999

theorem combinatorial_identity_solution :
  (∃ n : ℕ, C (20 : ℕ) (2 * n + 6) = C (20 : ℕ) (n + 2)) →
  (∃ (a : ℕ → ℤ), (2 - X)^4 = a 0 + a 1 * X + a 2 * X^2 + a 3 * X^3 + a 4 * X^4) →
  Σ (a : (ℕ → ℤ)), (a 0 - a 1 + a 2 - a 3 + a 4) = 81 := 
by 
  sorry

end combinatorial_identity_solution_l139_139999


namespace distinct_sums_distinct_sums_false_l139_139149

theorem distinct_sums (k : ℕ) (n : ℕ) (h_k : k ≥ 3) (h_n : n > Nat.choose k 3) 
  (a b c : Fin n → ℝ) (h_distinct : Function.Injective (Prod.fst ∘ Fin.sigma (λ _ : Fin n, Prod.mk (a _) (b _)) : Fin n → ℝ × ℝ)) :
  ∃ (s : Finset ℝ), s.card ≥ k + 1 ∧ ∀ i, s = {a i + b i, a i + c i, b i + c i} :=
sorry

theorem distinct_sums_false (k : ℕ) (n : ℕ) (h_k : k ≥ 3) (h_n : n = Nat.choose k 3)
  (a b c : Fin n → ℝ) :
  ∀ m : ℕ, ¬ (m > k → ∀ i, i < m → ¬ Function.Injective (λ i, a i + b i + c i)) :=
sorry

end distinct_sums_distinct_sums_false_l139_139149


namespace proof_ϕ_eq_f_on_real_l139_139719

variable (X : Type) [RandomVariable X] (ϕ : ℝ → ℂ) (f : ℝ → ℂ)
variable (c : ℕ → ℂ)

noncomputable def is_entire (f : ℝ → ℂ) :=
  ∃ (c : ℕ → ℂ), ∀ t : ℝ,  f t = ∑ n in Finset.range (n + 1), c n * t^n

axiom char_fn_coincides_with_entire_fn_neighborhood_zero
  (h1 : ∀ t : ℝ, ϕ t = ∑ n in Finset.range (n + 1), (i^n * 𝔼[X^n]) / n! * t^n)
  (h2 : ∀ t : ℝ,  f t = ∑ n in Finset.range (n + 1), c n * t^n)
  (h3 : ∀ t : ℝ, ϕ t = f t) : ϕ = f

theorem proof_ϕ_eq_f_on_real
  (h1 : ∀ t : ℝ, ϕ t = ∑ n in Finset.range (n + 1), (i^n * 𝔼[X^n]) / n! * t^n)
  (h2 : is_entire f)
  (h3 : ∀ t : ℝ, ϕ t = f t)
  : ∀ t : ℝ, ϕ t = f t :=
begin
  exact char_fn_coincides_with_entire_fn_neighborhood_zero h1 h2 h3,
end

end proof_ϕ_eq_f_on_real_l139_139719


namespace max_sum_sqrt_l139_139782

open Real

theorem max_sum_sqrt :
  ∀ (x1 x2 x3 x4 : ℝ), 
    (0 ≤ x1) → (0 ≤ x2) → (0 ≤ x3) → (0 ≤ x4) →
    (x1 + x2 + x3 + x4 = 1) →
    (∑ (i j : Fin₄) in {(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)}, 
     let xi := Fin₄.val i
     let xj := Fin₄.val j 
     (xi + xj) * sqrt(xi * xj) ≤ 3 / 4  :=
sorry

end max_sum_sqrt_l139_139782


namespace angle_BAC_in_trapezoid_is_15_l139_139749

theorem angle_BAC_in_trapezoid_is_15
  (ABCD : Type)
  [trapezoid ABCD]
  (base_AD : base ABCD AD)
  (diagonals_bisect_angles : ∀ (B C : ABCD), is_bisector B ∧ is_bisector C)
  (angle_C_eq : ∠ C = 110)
  : ∠ (BAC : Triangle) = 15 := sorry

end angle_BAC_in_trapezoid_is_15_l139_139749


namespace sum_200_equals_100_l139_139286

-- Define an arithmetic sequence and its sum function
variables {a : ℕ → ℝ}

def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_arithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 0 + a (n - 1)) / 2

-- Given conditions
variables (a_is_arithmetic : is_arithmetic a)
variables (a1 a200 : ℝ)
variables (h_collinear : a1 + a200 = 1)

-- Question to prove
theorem sum_200_equals_100 : sum_arithmetic a 200 = 100 :=
sorry

end sum_200_equals_100_l139_139286


namespace larger_angle_measure_l139_139114

theorem larger_angle_measure (x : ℝ) (h : 4 * x + 5 * x = 180) : 5 * x = 100 :=
by
  sorry

end larger_angle_measure_l139_139114


namespace triangle_point_collinear_l139_139939

open Euclidean

/-- Given a right-angled triangle ABC where ∠ACB = 90°, CH is the altitude to hypotenuse AB. Circle with center A and radius AC intersects any secant line through B at points D and E, with D between B and the intersection of the secant with CH. Given ∠ABG = ∠ABD where G is on the circle and on the opposite side of AB from D, then show E, H, and G are collinear. --/
theorem triangle_point_collinear {A B C H D E G : Point} :
  let circle : Circle := Circle.mk A (dist A C)
  let CH := altitude C A B
  right_triangle A C B (90: ℝ) ∧
  on_circle circle D ∧
  on_circle circle E ∧
  line_through B H ∧
  is_betw D B H ∧
  angle A B G = angle A B D ∧
  G ≠ D ∧
  dist G A = dist D A ∧
  G.x ≠ D.x ∧
  ∀ {F : Point}, on_secant circle B F → F = intersection (CH) (line_through B H)
  → collinear ({E, H, G} : Set Point) :=
by
  sorry

end triangle_point_collinear_l139_139939


namespace min_unit_circles_to_cover_triangle_l139_139131

/-- Prove that the minimum number of unit circles required to completely cover a
    triangle with sides 2, 3, and 4 is 3. -/
theorem min_unit_circles_to_cover_triangle : ∀ (a b c : ℝ), 
  a = 2 → b = 3 → c = 4 → 
  ∃ n : ℕ, n = 3 ∧ 
    ∀ (cover : ℕ → set (ℝ × ℝ)), 
    (∀ i, ∃ (x y : ℝ), cover i = { p | dist p (x, y) < 1 }) →
    ∃ (covers_triangle : set (ℝ × ℝ) → Prop), 
    covers_triangle (⋃ i in finset.range n, cover i) ∧
    (∀ m < n, ¬ covers_triangle (⋃ i in finset.range m, cover i)) :=
begin
  sorry
end

end min_unit_circles_to_cover_triangle_l139_139131


namespace prime_looking_count_less_than_500_l139_139205

noncomputable def is_prime_looking (n : ℕ) : Prop :=
  (¬ (even n)) ∧ (¬ (n % 3 = 0)) ∧ (¬ (n % 5 = 0)) ∧ (¬ (n % 7 = 0)) ∧ (¬ (Nat.Prime n)) ∧ (1 < n)

noncomputable def num_primes_less_than_500 : ℕ := 95

def num_prime_looking_numbers_less_than (n : ℕ) : ℕ :=
  (Nat.filter (fun k => is_prime_looking k) (List.range n)).length

theorem prime_looking_count_less_than_500 :
  num_prime_looking_numbers_less_than 500 = 60 :=
sorrozen

end prime_looking_count_less_than_500_l139_139205


namespace integer_n_satisfies_congruence_l139_139513

theorem integer_n_satisfies_congruence :
  ∃ n : ℤ, 0 ≤ n ∧ n < 203 ∧ 150 * n ≡ 95 [MOD 203] ∧ n = 144 :=
by {
  -- The conditions are combined into a single proof statement.
  -- Proof would go here, but we're skipping it.
  sorry
}

end integer_n_satisfies_congruence_l139_139513


namespace sick_cows_variance_l139_139562

noncomputable def ξ : ℕ → ℝ := binomial 10 0.02

theorem sick_cows_variance :
  variance (ξ 10) = 0.196 :=
by
  sorry

end sick_cows_variance_l139_139562


namespace coin_probability_l139_139448

theorem coin_probability :
  let PA := 3/4
  let PB := 1/2
  let PC := 1/4
  (PA * PB * (1 - PC)) = 9/32 :=
by
  sorry

end coin_probability_l139_139448


namespace minimal_unit_circles_to_cover_triangle_l139_139133

noncomputable
def triangle_side_a : ℝ := 2
noncomputable
def triangle_side_b : ℝ := 3
noncomputable
def triangle_side_c : ℝ := 4
noncomputable
def unit_circle_radius : ℝ := 1

theorem minimal_unit_circles_to_cover_triangle : 
  ∃ (n : ℕ), 
  n = 3 
  ∧ triangle_coverable_with_unit_circles 
      triangle_side_a 
      triangle_side_b 
      triangle_side_c 
      unit_circle_radius 
      n := 
by
  sorry

end minimal_unit_circles_to_cover_triangle_l139_139133


namespace dave_bought_26_tshirts_l139_139219

def total_tshirts :=
  let white_tshirts := 3 * 6
  let blue_tshirts := 2 * 4
  white_tshirts + blue_tshirts

theorem dave_bought_26_tshirts : total_tshirts = 26 :=
by
  unfold total_tshirts
  have white_tshirts : 3 * 6 = 18 := by norm_num
  have blue_tshirts : 2 * 4 = 8 := by norm_num
  rw [white_tshirts, blue_tshirts]
  norm_num

end dave_bought_26_tshirts_l139_139219


namespace find_lambda_condition_l139_139705

-- Define the vectors
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (1, 0)

-- Define the specific scalar we need to prove
def λ_correct : ℝ := 3

-- Prove the orthogonality condition holds
theorem find_lambda_condition : 
  ((a.1 - b.1, a.2 - b.2) • (2 * a.1 + λ_correct * b.1, 2 * a.2 + λ_correct * b.2)) = 0 :=
by
  sorry

end find_lambda_condition_l139_139705


namespace MaryHasBlueMarbles_l139_139608

-- Define the number of blue marbles Dan has
def DanMarbles : Nat := 5

-- Define the relationship of Mary's marbles to Dan's marbles
def MaryMarbles : Nat := 2 * DanMarbles

-- State the theorem that we need to prove
theorem MaryHasBlueMarbles : MaryMarbles = 10 :=
by
  sorry

end MaryHasBlueMarbles_l139_139608


namespace find_g5_l139_139076

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l139_139076


namespace maximize_profit_l139_139554

noncomputable def profit (x : ℕ) : ℝ := - (x : ℝ)^2 + 18 * x - 25

theorem maximize_profit :
  (∃ x : ℕ, x ∈ {1, 2, 3, ..., 1000} ∧ profit x = 8) ∧
  (∀ x : ℕ, x ∈ {1, 2, 3, ..., 1000} → profit 5 ≥ profit x) :=
by
  sorry

end maximize_profit_l139_139554


namespace decimal_to_binary_23_probability_A_selected_range_of_z_symmetric_point_variance_after_addition_l139_139892

-- Problem 1: Conversion of 23 to binary
theorem decimal_to_binary_23 : decimal_to_binary 23 = "10111" := sorry

-- Problem 2: Probability of A being selected from A, B, C, D
theorem probability_A_selected : 
  (choose 3 (finset.univ : finset (fin 4))).card ≠ 0 → (choose 3 (finset.univ : finset (fin 4))).card / (choose 3 (finset.singleton 0 : finset (fin 3))).card = 3 / 4 := sorry

-- Problem 3: Range of z = y / x given inequalities
theorem range_of_z (x y : ℝ) (hx : x - y - 2 ≤ 0) (hy : x + 2 * y - 5 ≥ 0) (hz : y - 2 ≤ 0) : (⅓ ≤ y / x ∧ y / x ≤ 2) := sorry

-- Problem 4: Symmetric point with respect to the line l: x + y - 1 = 0
theorem symmetric_point : symmetric_point (0, 2) (λ (x y : ℝ), x + y - 1 = 0) = (-1, 1) := sorry

-- Problem 5: Variance after adding data point
theorem variance_after_addition (data : list ℝ) (havg : list.mean data = 5) (hvar : list.variance data = 3) (hcount : data.length = 8) : 
  list.variance (data ++ [5]) = 8 / 3 := sorry

end decimal_to_binary_23_probability_A_selected_range_of_z_symmetric_point_variance_after_addition_l139_139892


namespace equivalent_sums_l139_139237

def sum_sin_sec_series (f : ℝ → ℝ → ℝ) (g : ℝ → ℝ → ℝ) (h : ℝ → ℝ → ℝ) : ℝ :=
  ∑ x in finset.Ico 3 (45 + 1), 2 * real.sin x * real.sin 2 * (1 + (real.sec (x - 2) * real.sec (x + 2)))

def sum_phi_psi_series (Φ Ψ : ℝ → ℝ) (θ : fin 4 → ℝ) : ℝ :=
  ∑ n in finset.range 4, ((-1) ^ (n + 1)) * (Φ (θ n) / Ψ (θ n))

theorem equivalent_sums :
  ∃ (θ : fin 4 → ℝ), 
    θ 0 = 1 ∧ θ 1 = 2 ∧ θ 2 = 45 ∧ θ 3 = 47 ∧
    sum_sin_sec_series (λ x y, real.cos (x - y) - real.cos (x + y))
                      (λ x y, real.sin (x - y))
                      (λ x y, real.sin (x + y)) 
    = sum_phi_psi_series (λ θ, real.sin θ * real.sin 2) 
                         (λ θ, real.cos θ * real.cos 2) 
                         θ
    ∧ (θ 0 + θ 1 + θ 2 + θ 3 = 95) :=
begin
  sorry
end

end equivalent_sums_l139_139237


namespace net_increase_is_96078_l139_139407

noncomputable def net_population_increase : ℕ :=
  let b := 90171
  let i := 16320
  let e := 8212
  let P := 2876543
  let d_r := 0.0008
  let d := Int.floor (d_r * P)
  in b + i - e - d

theorem net_increase_is_96078 : net_population_increase = 96078 := by
  sorry

end net_increase_is_96078_l139_139407


namespace distinct_arrangements_of_PHONE_l139_139711

-- Condition: The word PHONE consists of 5 distinct letters
def distinctLetters := 5

-- Theorem: The number of distinct arrangements of the letters in the word PHONE
theorem distinct_arrangements_of_PHONE : Nat.factorial distinctLetters = 120 := sorry

end distinct_arrangements_of_PHONE_l139_139711


namespace product_of_roots_is_six_l139_139633

noncomputable theory

-- Define the quadratic equation
def quadratic_eq : Polynomial ℝ := Polynomial.C 6 + Polynomial.X * -5 + Polynomial.X^2

-- State the main theorem
theorem product_of_roots_is_six : (∏ x in (quadratic_eq.roots.toFinset : Finset ℝ), x) = 6 := by
  sorry

end product_of_roots_is_six_l139_139633


namespace bus_speed_express_mode_l139_139834

theorem bus_speed_express_mode (L : ℝ) (t_red : ℝ) (speed_increase : ℝ) (x : ℝ) (normal_speed : ℝ) :
  L = 16 ∧ t_red = 1 / 15 ∧ speed_increase = 8 ∧ normal_speed = x - 8 ∧ 
  (16 / normal_speed - 16 / x = 1 / 15) → x = 48 :=
by
  sorry

end bus_speed_express_mode_l139_139834


namespace intersection_a_minus_one_range_of_a_l139_139699

open Set Real

def U := univ
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 + 5 * a * x + 6 * a^2 ≤ 0}

theorem intersection_a_minus_one :
  B (-1) ∩ A = {x | 2 ≤ x ∧ x ≤ 3} ∧ B (-1) ∩ compl A = ∅ := by sorry

theorem range_of_a (a : ℝ) (h1: A ∪ B a = A) (h2: a < 0) :
  -1 / 2 ≥ a ∧ a > -4 / 3 := by sorry

end intersection_a_minus_one_range_of_a_l139_139699


namespace shobha_current_age_l139_139887

variable (S B : ℕ)
variable (h_ratio : 4 * B = 3 * S)
variable (h_future_age : S + 6 = 26)

theorem shobha_current_age : B = 15 :=
by
  sorry

end shobha_current_age_l139_139887


namespace number_to_remove_l139_139184

theorem number_to_remove (total_students sample_size : ℕ) : total_students = 254 → sample_size = 42 → (254 % 42) = 2 :=
begin
  intros h1 h2,
  rw [h1, h2],
  exact nat.mod_eq_of_lt (by norm_num : 254 < 42 * 7)
end

end number_to_remove_l139_139184


namespace distance_AB_eq_13_l139_139378

-- Define the coordinates of point A
def A : ℝ × ℝ × ℝ := (1, 3, 5)
-- Define the coordinates of point B
def B : ℝ × ℝ × ℝ := (-3, 6, -7)

-- Define the square of the distance between two points in 3D space
def dist_sq (P1 P2 : ℝ × ℝ × ℝ) : ℝ :=
  (P2.1 - P1.1)^2 + (P2.2 - P1.2)^2 + (P2.3 - P1.3)^2

-- The statement to be proven
theorem distance_AB_eq_13 : Real.sqrt (dist_sq A B) = 13 := sorry

end distance_AB_eq_13_l139_139378


namespace statement_B_statement_C_l139_139139

-- Define the conditions for each statement
def congruent_triangles (T1 T2 : Type) [triangle T1] [triangle T2] : Prop :=
  -- Assume T1 and T2 are congruent
  ∃ (T1 ≡ T2), true

def equal_areas (T1 T2 : Type) [triangle T1] [triangle T2] (area1 area2 : ℝ) : Prop :=
  -- Assume T1 and T2 have equal areas
  area1 = area2

def sufficient_not_necessary (A B : Prop) : Prop :=
  -- Prove A is sufficient but not necessary for B
  (A → B) ∧ ¬(B → A)

def necessary_not_sufficient (P Q : Prop) : Prop :=
  -- Prove P is necessary but not sufficient for Q
  (Q → P) ∧ ¬(P → Q)

variables {T1 T2 : Type} [triangle T1] [triangle T2] (a1 a2 : ℝ) (m : ℝ)

-- Statement B
theorem statement_B : sufficient_not_necessary (congruent_triangles T1 T2) (equal_areas T1 T2 a1 a2) := 
  sorry

-- Statement C
theorem statement_C : necessary_not_sufficient (m ≤ 1) (∃ x : ℝ, ∃ y : ℝ, x ≠ y ∧ (m * x^2 + 2*x + 1 = 0) ∧ (m * y^2 + 2*y + 1 = 0)) := 
  sorry

end statement_B_statement_C_l139_139139


namespace sum_digits_in_possibilities_l139_139383

noncomputable def sum_of_digits (a b c d : ℕ) : ℕ :=
  a + b + c + d

theorem sum_digits_in_possibilities :
  ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (sum_of_digits a b c d = 10 ∨ sum_of_digits a b c d = 18 ∨ sum_of_digits a b c d = 19) := sorry

end sum_digits_in_possibilities_l139_139383


namespace tech_gadget_cost_inr_l139_139436

def conversion_ratio (a b : ℝ) : Prop := a = b

theorem tech_gadget_cost_inr :
  (forall a b c : ℝ, conversion_ratio (a / b) c) →
  (forall a b c d : ℝ, conversion_ratio (a / b) c → conversion_ratio (a / d) c) →
  ∀ (n_usd : ℝ) (n_inr : ℝ) (cost_n : ℝ), 
    n_usd = 8 →
    n_inr = 5 →
    cost_n = 160 →
    cost_n / n_usd * n_inr = 100 :=
by
  sorry

end tech_gadget_cost_inr_l139_139436


namespace compute_g_100_l139_139612

-- Define the function g(x) according to the conditions in Lean
def g : ℕ → ℕ 
| x := if (∃ (n : ℤ), (x : ℤ) = 3^n) then 3 * (Int.log x 3) else 1 + g (x + 1)

-- The main statement to prove
theorem compute_g_100 : g 100 = 158 :=
by sorry

end compute_g_100_l139_139612


namespace gunther_free_time_remaining_l139_139333

-- Define the conditions
def vacuum_time : ℕ := 45
def dust_time : ℕ := 60
def mop_time : ℕ := 30
def brushing_time_per_cat : ℕ := 5
def number_of_cats : ℕ := 3
def free_time_hours : ℕ := 3

-- Convert the conditions into a proof problem
theorem gunther_free_time_remaining : 
  let total_cleaning_time := vacuum_time + dust_time + mop_time + brushing_time_per_cat * number_of_cats
  let free_time_minutes := free_time_hours * 60
  in free_time_minutes - total_cleaning_time = 30 :=
by
  sorry

end gunther_free_time_remaining_l139_139333


namespace list_price_is_40_l139_139934

theorem list_price_is_40 (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to indicate we're skipping the proof.
  sorry

end list_price_is_40_l139_139934


namespace problem_statement_l139_139745

-- Definitions and transformations
def polarToRectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def parametricLineOP (t : ℝ) : ℝ × ℝ :=
  (3 + (sqrt 3 / 2) * t, sqrt 3 + (1 / 2) * t)

def curveC (x y : ℝ) : Prop :=
  x^2 - y^2 = 9

-- Lean statement for the proof problem
theorem problem_statement : 
  let P := polarToRectangular (2 * sqrt 3) (π / 6) in
  let parametric := parametricLineOP in
  let C := curveC in
  ∃ (A B : ℝ × ℝ), 
    (parametric A.1, parametric A.2) ∈ C ∧
    (parametric B.1, parametric B.2) ∈ C ∧ 
    (∃ t1 t2 : ℝ, 
       t1 + t2 = -4 * sqrt 3 ∧ 
       t1 * t2 = -6 ∧ 
       (1 / |t1| + 1 / |t2| = sqrt 2)) := 
sorry

end problem_statement_l139_139745


namespace congruent_if_and_only_if_nagel_point_l139_139408

-- Define the necessary structures and the problem statement

structure Triangle :=
(α β γ : ℝ)
(ABC : ℝ)

def Circumcircle : Triangle → ℝ := sorry

def isTangent :: ℝ → ℝ → ℝ → ℝ := sorry

def NagelPoint : Triangle → ℝ := sorry

noncomputable def areCongruent (T1 T2 : ℝ) : Prop := sorry

theorem congruent_if_and_only_if_nagel_point
(T : Triangle)
(A' : ℝ)
(hA' : A' ∈ T.γ) 
(TA'B : ℝ)
(TA'C : ℝ)
(hTA'B_tangent : isTangent T.α A' T.β TA'B)
(hTA'C_tangent : isTangent T.α A' T.γ TA'C)
(Gamma := Circumcircle T) :
  (areCongruent TA'B TA'C) ↔
  (A' ∈ NagelPoint T) := sorry

end congruent_if_and_only_if_nagel_point_l139_139408


namespace binom_15_13_eq_105_l139_139204

theorem binom_15_13_eq_105 : nat.choose 15 13 = 105 :=
by
sorry

end binom_15_13_eq_105_l139_139204


namespace double_root_relationship_l139_139272

theorem double_root_relationship (a b c : ℝ) (α : ℝ) (h : a ≠ 0) 
    (root_condition : ∃ β, α + β = -b / a ∧ α * β = c / a ∧ β = 2 * α) : 
    2 * b^2 = 9 * a * c :=
begin
  sorry
end

end double_root_relationship_l139_139272


namespace value_of_x2_plus_inv_x2_l139_139676

theorem value_of_x2_plus_inv_x2 (x : ℝ) (hx : x ≠ 0) (h : x^4 + 1 / x^4 = 47) : x^2 + 1 / x^2 = 7 :=
sorry

end value_of_x2_plus_inv_x2_l139_139676


namespace sin_pi_plus_alpha_l139_139274

theorem sin_pi_plus_alpha (α : ℝ) (h1 : sin (π / 2 + α) = 3 / 5) (h2 : 0 < α ∧ α < π / 2) :
  sin (π + α) = -4 / 5 :=
sorry

end sin_pi_plus_alpha_l139_139274


namespace find_lambda_l139_139789

noncomputable def vector_a (λ : ℝ) : ℝ × ℝ × ℝ := (1, λ, 2)
def vector_b : ℝ × ℝ × ℝ := (2, -1, 2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

def cos_angle (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

noncomputable def satisfy_condition (λ : ℝ) : Prop :=
  cos_angle (vector_a λ) vector_b ≥ 8 / 9

theorem find_lambda
  (λ : ℝ) :
  satisfy_condition λ →
  λ = -2 ∨ λ = 2 / 55 :=
sorry

end find_lambda_l139_139789


namespace pythagorean_triplet_l139_139434

theorem pythagorean_triplet (a b c : ℕ) (h : a² + b² = c²) (ha : a = 11) : c = 61 :=
by
  sorry

end pythagorean_triplet_l139_139434


namespace part_I_solution_part_II_solution_l139_139444

-- Definition for the given function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- Part (I): Conditions and goal
theorem part_I_solution (x : ℝ) : (f x (-1) ≤ 2) → (-1/2 ≤ x ∧ x ≤ 1/2) :=
    sorry

-- Part (II): Conditions and goal
theorem part_II_solution (a : ℝ) : 
  (\forall (x : ℝ), (1/2 ≤ x ∧ x ≤ 1) → (f x a ≤ |2 * x + 1|)) → (0 ≤ a ∧ a ≤ 3) :=
    sorry

end part_I_solution_part_II_solution_l139_139444


namespace g_5_is_248_l139_139060

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l139_139060


namespace expressions_equal_iff_sum_zero_l139_139953

theorem expressions_equal_iff_sum_zero (p q r : ℝ) : (p + qr = (p + q) * (p + r)) ↔ (p + q + r = 0) :=
sorry

end expressions_equal_iff_sum_zero_l139_139953


namespace total_views_l139_139931

def first_day_views : ℕ := 4000
def views_after_4_days : ℕ := 40000 + first_day_views
def views_after_6_days : ℕ := views_after_4_days + 50000

theorem total_views : views_after_6_days = 94000 := by
  have h1 : first_day_views = 4000 := rfl
  have h2 : views_after_4_days = 40000 + first_day_views := rfl
  have h3 : views_after_6_days = views_after_4_days + 50000 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end total_views_l139_139931


namespace ratio_of_speeds_correct_l139_139539

noncomputable def ratio_speeds_proof_problem : Prop :=
  ∃ (v_A v_B : ℝ),
    (∀ t : ℝ, 0 ≤ t ∧ t = 3 → 3 * v_A = abs (-800 + 3 * v_B)) ∧
    (∀ t : ℝ, 0 ≤ t ∧ t = 15 → 15 * v_A = abs (-800 + 15 * v_B)) ∧
    (3 * 15 * v_A / (15 * v_B) = 3 / 4)

theorem ratio_of_speeds_correct : ratio_speeds_proof_problem :=
sorry

end ratio_of_speeds_correct_l139_139539


namespace tan_three_degrees_sum_l139_139091

noncomputable def p : ℕ := 10
noncomputable def q : ℕ := 5
noncomputable def r : ℕ := 2
noncomputable def s : ℕ := 3

theorem tan_three_degrees_sum :
  (∃ (p q r s : ℕ), 
    p ≥ q ∧ q ≥ r ∧ r ≥ s ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ 
    Real.tan (Real.pi / 60) = Real.sqrt p - Real.sqrt q + Real.sqrt r - s ∧ 
    p + q + r + s = 20) := 
  by
    use p, q, r, s
    have hpqrs : p ≥ q ∧ q ≥ r ∧ r ≥ s ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 := by
      repeat {simp [p, q, r, s]}
    exact ⟨hpqrs.1, hpqrs.2.1, hpqrs.2.2.1, hpqrs.2.2.2.1, hpqrs.2.2.2.2.1, hpqrs.2.2.2.2.2.1, hpqrs.2.2.2.2.2.2, sorry, rfl⟩

end tan_three_degrees_sum_l139_139091


namespace find_g5_l139_139057

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l139_139057


namespace floral_shop_bouquets_l139_139161

theorem floral_shop_bouquets (T : ℕ) 
  (h1 : 12 + T + T / 3 = 60) 
  (hT : T = 36) : T / 12 = 3 :=
by
  -- Proof steps go here
  sorry

end floral_shop_bouquets_l139_139161


namespace second_particle_catches_first_l139_139805

open Real

-- Define the distance functions for both particles
def distance_first (t : ℝ) : ℝ := 34 + 5 * t
def distance_second (t : ℝ) : ℝ := 0.25 * t^2 + 2.75 * t

-- The proof statement
theorem second_particle_catches_first : ∃ t : ℝ, distance_second t = distance_first t ∧ t = 17 :=
by
  have : distance_first 17 = 34 + 5 * 17 := by sorry
  have : distance_second 17 = 0.25 * 17^2 + 2.75 * 17 := by sorry
  sorry

end second_particle_catches_first_l139_139805


namespace decimal_to_base8_conversion_l139_139603

theorem decimal_to_base8_conversion : (512 : ℕ) = 8^3 :=
by
  sorry

end decimal_to_base8_conversion_l139_139603


namespace multiple_of_four_l139_139421

open BigOperators

theorem multiple_of_four (n : ℕ) (x y z : Fin n → ℤ)
  (hx : ∀ i, x i = 1 ∨ x i = -1)
  (hy : ∀ i, y i = 1 ∨ y i = -1)
  (hz : ∀ i, z i = 1 ∨ z i = -1)
  (hxy : ∑ i, x i * y i = 0)
  (hxz : ∑ i, x i * z i = 0)
  (hyz : ∑ i, y i * z i = 0) :
  (n % 4 = 0) :=
sorry

end multiple_of_four_l139_139421


namespace number_of_sampled_medium_stores_is_five_l139_139733

-- Definitions based on the conditions
def total_stores : ℕ := 300
def large_stores : ℕ := 30
def medium_stores : ℕ := 75
def small_stores : ℕ := 195
def sample_size : ℕ := 20

-- Proportion calculation function
def medium_store_proportion := (medium_stores : ℚ) / (total_stores : ℚ)

-- Sampled medium stores calculation
def sampled_medium_stores := medium_store_proportion * (sample_size : ℚ)

-- Theorem stating the number of medium stores drawn using stratified sampling
theorem number_of_sampled_medium_stores_is_five :
  sampled_medium_stores = 5 := 
by 
  sorry

end number_of_sampled_medium_stores_is_five_l139_139733


namespace time_to_complete_trip_l139_139041

-- Definitions for the problem
def radius_of_planet : ℝ := 2000 -- in miles
def speed_of_jet : ℝ := 600 -- in miles per hour
def pi : ℝ := Real.pi -- π

-- The mathematically equivalent proof problem statement

theorem time_to_complete_trip : 
  let circumference := 2 * pi * radius_of_planet,
      time := circumference / speed_of_jet
  in abs (time - 21) < 1 :=
by
  let circumference := 2 * pi * radius_of_planet
  let time := circumference / speed_of_jet
  have h1: abs (time - 21) < 1 := sorry
  exact h1

end time_to_complete_trip_l139_139041


namespace bees_each_day_pattern_l139_139433

theorem bees_each_day_pattern :
  let monday := 144
  let tuesday := 3 * monday
  let wednesday := 0.5 * tuesday
  let thursday := 2 * wednesday
  let friday := 1.5 * thursday
  let saturday := 0.75 * friday
  let sunday := 4 * saturday
  in sunday = 1944 :=
by
  sorry

end bees_each_day_pattern_l139_139433


namespace g_5_is_248_l139_139064

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l139_139064


namespace average_age_group_l139_139533

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = n * 14) (h2 : T + 32 = (n + 1) * 15) : n = 17 :=
by
  sorry

end average_age_group_l139_139533


namespace least_x_multiple_of_53_l139_139518

theorem least_x_multiple_of_53 : ∃ (x : ℕ), (x > 0) ∧ (3 * x + 41) % 53 = 0 ∧ x = 4 :=
by
  have : ∃ (x : ℕ), (x > 0) ∧ (3 * x + 41) % 53 = 0, from sorry
  exact ⟨4, sorry, sorry⟩

end least_x_multiple_of_53_l139_139518


namespace light_travel_distance_l139_139546

noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem light_travel_distance : 
  let P := (1 : ℝ, 1 : ℝ, 1 : ℝ)
  let M := (1 : ℝ, 1 : ℝ, -1 : ℝ)
  let Q := (3 : ℝ, 3 : ℝ, 6 : ℝ)
  distance Q.1 Q.2 Q.3 M.1 M.2 M.3 = Real.sqrt 57 :=
by
  sorry

end light_travel_distance_l139_139546


namespace max_tan_B_triangle_l139_139392

open Real
open EuclideanGeometry

theorem max_tan_B_triangle (A B C : Point) (hAB : dist A B = 25) (hBC : dist B C = 20) (hRightAngle : angle C A B = π / 2) :  
  tan (angle A B C) = 3 / 4 :=
by
  sorry

end max_tan_B_triangle_l139_139392


namespace math_problem_l139_139697

noncomputable def a_seq : ℕ+ → ℚ
| ⟨1, _⟩ := 1
| ⟨n+1, h⟩ := 1 - (1 / (4 * a_seq ⟨n, h⟩))

noncomputable def b_seq (n : ℕ+) : ℚ := (2 / (2 * a_seq n - 1))

noncomputable def c_seq (n : ℕ) : ℚ := (4 * a_seq ⟨n + 1, by linarith⟩) / (n + 1)

noncomputable def T_seq (n : ℕ) : ℚ :=
∑ i in Finset.range n, c_seq i * c_seq (i + 2)

theorem math_problem:
  (∀ n : ℕ+, b_seq n = 2 * n) ∧
  (∀ n : ℕ+, a_seq n = ((n:ℚ) + 1) / (2 * n)) ∧
  (∃ m : ℕ+, ∀ n : ℕ+, T_seq n < 1 / (c_seq m * c_seq (m + 1)) ∧ m = 3) :=
sorry

end math_problem_l139_139697


namespace pheromone_effect_on_population_l139_139510

-- Definitions of conditions
def disrupt_sex_ratio (uses_pheromones : Bool) : Bool :=
  uses_pheromones = true

def decrease_birth_rate (disrupt_sex_ratio : Bool) : Bool :=
  disrupt_sex_ratio = true

def decrease_population_density (decrease_birth_rate : Bool) : Bool :=
  decrease_birth_rate = true

-- Problem Statement for Lean 4
theorem pheromone_effect_on_population (uses_pheromones : Bool) :
  disrupt_sex_ratio uses_pheromones = true →
  decrease_birth_rate (disrupt_sex_ratio uses_pheromones) = true →
  decrease_population_density (decrease_birth_rate (disrupt_sex_ratio uses_pheromones)) = true :=
sorry

end pheromone_effect_on_population_l139_139510


namespace mod_equiv_solution_l139_139344

theorem mod_equiv_solution (a b : ℤ) (n : ℤ) 
  (h₁ : a ≡ 22 [ZMOD 50])
  (h₂ : b ≡ 78 [ZMOD 50])
  (h₃ : 150 ≤ n ∧ n ≤ 201)
  (h₄ : n = 194) :
  a - b ≡ n [ZMOD 50] :=
by
  sorry

end mod_equiv_solution_l139_139344


namespace thief_speed_l139_139919

theorem thief_speed (v : ℝ) :
  (let distance_thief_start := v * 1 in      -- distance the thief has covered in 1 hour
   let distance_police := 40 * 4 in           -- distance the police has covered in 4 hours
   let distance_thief := v * 4 in             -- distance the thief has covered in 4 hours
   distance_thief_start + distance_thief = distance_police) 
   → v = 32 :=
by 
  intros h 
  sorry

end thief_speed_l139_139919


namespace vertical_angles_are_equal_l139_139525

theorem vertical_angles_are_equal (l₁ l₂ : Line) (P Q R S : Point) 
    (h₁ : Intersect l₁ l₂ P)
    (h₂ : Intersect l₁ l₂ R)
    (h₃ : ∠PQR)
    (h₄ : ∠PQS) 
    (h₅: vertical_angle P Q) :
  ∠PQR = ∠PQS :=
sorry

end vertical_angles_are_equal_l139_139525


namespace marble_probability_l139_139862

theorem marble_probability
  (total_marbles : ℕ := 84)
  (P_G : ℚ := 2 / 7) -- probability of picking a green marble
  (P_R_or_B : ℝ := 0.4642857142857143) -- probability of picking a red or blue marble
  (P_W : ℝ := 1 - (P_G.toReal + P_R_or_B)) -- probability of picking a white marble
  : P_W = 0.25 := sorry

end marble_probability_l139_139862


namespace find_norm_n_l139_139673

variables {R : Type*} [NormedField R] [InnerProductSpace R (EuclideanSpace R ℝ)]
variables (m n : EuclideanSpace R ℝ)

def m_dot_n_eq_zero : Prop := inner m n = 0
def m_norm_eq_5 : Prop := ‖m‖ = 5
def m_sub_2n_eq : Prop := m - 2 • n = ![(11 : R), (-2 : R)]

theorem find_norm_n 
  (hmn : m_dot_n_eq_zero m n) 
  (hm5 : m_norm_eq_5 m) 
  (hm2n : m_sub_2n_eq m n) : 
    ‖n‖ = 5 :=
sorry

end find_norm_n_l139_139673


namespace sum_of_coefficients_of_expansion_l139_139645

theorem sum_of_coefficients_of_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 1)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + a_2 + a_3 + a_4 + a_5 = 2 :=
by
  intro h
  have h0 := h 0
  have h1 := h 1
  sorry

end sum_of_coefficients_of_expansion_l139_139645


namespace range_of_m_l139_139309

def f (x m : ℝ) : ℝ :=
  if x < m then x^2 + 4*x - 3 else 4

def g (x m : ℝ) : ℝ :=
  f x m - 2*x

theorem range_of_m {m : ℝ} :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ g x1 m = 0 ∧ g x2 m = 0 ∧ g x3 m = 0) →
  1 < m ∧ m ≤ 2 := by
  sorry

end range_of_m_l139_139309


namespace blocks_to_gallery_proof_l139_139203

variable (blocks_to_store blocks_to_gallery final_blocks initial_blocks total_blocks : ℕ)

theorem blocks_to_gallery_proof 
  (h1 : blocks_to_store = 11) 
  (h2 : final_blocks = 8) 
  (h3 : initial_blocks = 5) 
  (h4 : total_blocks = initial_blocks + 20) 
  (h5 : total_blocks = blocks_to_store + blocks_to_gallery + final_blocks) :
  blocks_to_gallery = 6 :=
by
  have h6 : total_blocks = 25 := by rw [h4, Nat.add_comm, Nat.add_sub_assoc []; exact le_of_lt (Nat.lt_add_of_pos_right _.zero_lt)]
  have h7 : blocks_to_gallery = 25 - blocks_to_store - final_blocks := 
    by rw [h6, h5, Nat.sub_sub_sub_cancel_right blocks_to_store final_blocks (by exact le_of_lt (Nat.lt_add_of_pos_right (by simp) blocks_to_store.fin_pos))]
  rw [blocks_to_store, final_blocks] at h7
  exact h7

end blocks_to_gallery_proof_l139_139203


namespace find_g5_l139_139080

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l139_139080


namespace flowchart_output_value_l139_139811

theorem flowchart_output_value :
  ∃ n, ∀ m, m < n → (s m ≥ 120) ∧ (s n < 120) :=
sorry

def s : Nat → Nat
| 0 => 2010
| 1 => 1002
| 2 => 498
| 3 => 246
| 4 => 120
| 5 => 57
| _ => sorry

example : ∃ n, n = 5 ∧ (s n < 120) :=
exists.intro 5 (and.intro rfl (by norm_num))

end flowchart_output_value_l139_139811


namespace slope_of_line_l139_139302

variable (k : ℝ)

-- Condition 1: Equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 = 0

-- Condition 2: Equation of the line l
def line_eq (x y : ℝ) : Prop :=
  k*x - y + 2 - 2*k = 0

-- Condition 3: Line l intersects circle C at points A and B
def intersects (l_circle l_line : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), l_circle x1 y1 ∧ l_line x1 y1 ∧ l_circle x2 y2 ∧ l_line x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2)

-- Condition 4: The area of triangle ΔABC is at its maximum
def max_triangle_area (d : ℝ) : Prop :=
  d = (Real.sqrt 2) / 2 -- implies d is the half distance.

-- Prove k = 1 or k = 7 given the above conditions
theorem slope_of_line (d : ℝ) (l_circle l_line : ℝ → ℝ → Prop) :
  max_triangle_area d ∧ intersects l_circle l_line → k = 1 ∨ k = 7 :=
  by obtain ⟨k, d⟩; sorry

end slope_of_line_l139_139302


namespace polygon_sides_l139_139725

theorem polygon_sides (n : ℕ) 
  (h1 : sum_interior_angles = 180 * (n - 2))
  (h2 : sum_exterior_angles = 360)
  (h3 : sum_interior_angles = 3 * sum_exterior_angles) : 
  n = 8 :=
by
  sorry

end polygon_sides_l139_139725


namespace bags_of_cookies_l139_139728

theorem bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 33) (h2 : cookies_per_bag = 11) : total_cookies / cookies_per_bag = 3 :=
by
  sorry

end bags_of_cookies_l139_139728


namespace area_triangle_PQR_l139_139211

-- Define the points as provided in the conditions
def P := (0, 0 : ℝ × ℝ)
def Q := (2, 0 : ℝ × ℝ)
def R := (2, 2 : ℝ × ℝ)

-- Define a function to calculate the area of the triangle given the three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

-- State the theorem to prove the area of the triangle PQR
theorem area_triangle_PQR : triangle_area P Q R = 2 :=
by
  sorry

end area_triangle_PQR_l139_139211


namespace rhombus_perimeter_l139_139047

variable (d1 d2 : ℝ)
variable (h1 : d1 = 8) (h2 : d2 = 15)

theorem rhombus_perimeter (h3 : 4 * (sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 34)) : 
  4 * (sqrt ((8 / 2)^2 + (15 / 2)^2)) = 34 :=
by 
  rw [h1, h2]
  exact h3

end rhombus_perimeter_l139_139047


namespace exist_plane_parallelogram_l139_139340

-- Define A, B, C, D as distinct points in space
variables (A B C D : ℝ^3)

-- Definition of orthogonal projection
def orthogonal_projection (P : affine_subspace ℝ (ℝ^3)) (x : ℝ^3) : ℝ^3 :=
  sorry -- Placeholder for the actual orthogonal projection function

-- Definition of a parallelogram in terms of projections
def is_parallelogram (P : affine_subspace ℝ (ℝ^3)) : Prop :=
  let A' := orthogonal_projection P A in
  let B' := orthogonal_projection P B in
  let C' := orthogonal_projection P C in
  let D' := orthogonal_projection P D in
  (A' - B' = C' - D') ∧ (A' - D' = B' - C')

-- Main theorem statement
theorem exist_plane_parallelogram (A B C D : ℝ^3) (h : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  ∃ (P : affine_subspace ℝ (ℝ^3)), is_parallelogram A B C D P :=
sorry

end exist_plane_parallelogram_l139_139340


namespace problem_1_problem_2_l139_139685

noncomputable def f (a x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 8
noncomputable def g (a x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 - 12 * a^2 * x + 3 * a^3

theorem problem_1 (a : ℝ) : (∀ x ∈ set.Icc (1 : ℝ) 2, f a x < 0) → 10 < a :=
sorry

theorem problem_2 : ¬ ∃ a : ℤ, ∃ x ∈ set.Ioo (0 : ℝ) 1, is_local_min (g a) x :=
sorry

end problem_1_problem_2_l139_139685


namespace largest_angle_of_consecutive_integers_hexagon_l139_139467

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l139_139467


namespace investment_c_is_correct_l139_139886

-- Define the investments of a and b
def investment_a : ℕ := 45000
def investment_b : ℕ := 63000
def profit_c : ℕ := 24000
def total_profit : ℕ := 60000

-- Define the equation to find the investment of c
def proportional_share (x y total : ℕ) : Prop :=
  2 * (x + y + total) = 5 * total

-- The theorem to prove c's investment given the conditions
theorem investment_c_is_correct (c : ℕ) (h_proportional: proportional_share investment_a investment_b c) :
  c = 72000 :=
by
  sorry

end investment_c_is_correct_l139_139886


namespace min_students_with_brown_eyes_and_lunch_box_l139_139870

theorem min_students_with_brown_eyes_and_lunch_box:
  ∀ (B L T : ℕ), B = 12 ∧ L = 20 ∧ T = 30 → 
  (∃ (N : ℕ), N = B - (T - L) ∧ N = 2) := by
  intros B L T h
  rcases h with ⟨hB, hL, hT⟩
  use (B - (T - L))
  split
  case left {
    rw [hB, hL, hT]
  }
  case right {
    sorry -- steps to show (B - (T - L)) = 2
  }

end min_students_with_brown_eyes_and_lunch_box_l139_139870


namespace power_function_half_l139_139322

-- Define the power function as f(x) = x ^ a
def power_function (a x : ℝ) : ℝ := x ^ a

-- Given the condition that the power function passes through the point (2, sqrt(2))
def condition (a : ℝ) := power_function a 2 = Real.sqrt 2

-- Prove that the power function must be of the form y = x^1/2
theorem power_function_half :
  ∃ a : ℝ, condition a ∧ ∀ x : ℝ, power_function a x = x ^ (1 / 2) :=
by
  sorry

end power_function_half_l139_139322


namespace Linda_sold_7_tees_l139_139425

variables (T : ℕ)
variables (jeans_price tees_price total_money_from_jeans total_money total_money_from_tees : ℕ)
variables (jeans_sold : ℕ)

def tees_sold :=
  jeans_price = 11 ∧ tees_price = 8 ∧ jeans_sold = 4 ∧
  total_money = 100 ∧ total_money_from_jeans = jeans_sold * jeans_price ∧
  total_money_from_tees = total_money - total_money_from_jeans ∧
  T = total_money_from_tees / tees_price
  
theorem Linda_sold_7_tees (h : tees_sold T jeans_price tees_price total_money_from_jeans total_money total_money_from_tees jeans_sold) : T = 7 :=
by
  sorry

end Linda_sold_7_tees_l139_139425


namespace find_pairs_satisfying_eq_l139_139221

-- Definitions for Euler's totient function and the target equation
def euler_totient (n : ℕ) : ℕ := n * (nat.proper_divisors n).filter (nat.coprime n).card

def target_eq (a b : ℕ) : Prop :=
  14 * (euler_totient a)^2 - euler_totient (a * b) + 22 * (euler_totient b)^2 = a^2 + b^2

-- The main theorem statement
theorem find_pairs_satisfying_eq :
  ∀ a b : ℕ,
  target_eq a b →
  ∃ x y : ℕ, a = 30 * 2^x * 3^y ∧ b = 6 * 2^x * 3^y :=
by sorry

end find_pairs_satisfying_eq_l139_139221


namespace ratio_of_milk_to_water_l139_139369

theorem ratio_of_milk_to_water (initial_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) (added_water : ℕ) :
  initial_volume = 45 → milk_ratio = 4 → water_ratio = 1 → added_water = 18 → 
  (milk_ratio * 45 / (milk_ratio + water_ratio)) : (water_ratio * 45 / (milk_ratio + water_ratio) + added_water) = 4 : 3 :=
by 
  sorry

end ratio_of_milk_to_water_l139_139369


namespace jenna_eel_length_l139_139402

theorem jenna_eel_length (J B L : ℝ)
  (h1 : J = (2 / 5) * B)
  (h2 : J = (3 / 7) * L)
  (h3 : J + B + L = 124) : 
  J = 21 := 
sorry

end jenna_eel_length_l139_139402


namespace b_seq_arithmetic_l139_139698

noncomputable def a_seq : ℕ → ℚ
| 1       := 2
| (n + 1) := 2 - 1 / a_seq n

def b_seq (n : ℕ) : ℚ := 1 / (a_seq (n + 1) - 1)

theorem b_seq_arithmetic (n : ℕ) : b_seq (n + 1) - b_seq n = 1 :=
by
  sorry

end b_seq_arithmetic_l139_139698


namespace monotone_increasing_interval_f_l139_139839

noncomputable def f (x : ℝ) : ℝ := Real.logBase 0.5 (x^2 - 4)

theorem monotone_increasing_interval_f :
  (∀ x : ℝ, x ∈ Iio (-2) → ∀ y : ℝ, y ∈ Iio (-2) → y ≥ x → f y ≥ f x) :=
by
  sorry

end monotone_increasing_interval_f_l139_139839


namespace vector_relationship_l139_139653

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
          (A A1 B D E : V) (x y z : ℝ)

-- Given Conditions
def inside_top_face_A1B1C1D1 (E : V) : Prop :=
  ∃ (y z : ℝ), (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧
  E = A1 + y • (B - A) + z • (D - A)

-- Prove the desired relationship
theorem vector_relationship (h : E = x • (A1 - A) + y • (B - A) + z • (D - A))
  (hE : inside_top_face_A1B1C1D1 A A1 B D E) : 
  x = 1 ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) :=
sorry

end vector_relationship_l139_139653


namespace video_views_l139_139928

theorem video_views (initial_views : ℕ) (increase_factor : ℕ) (additional_views : ℕ) :
  initial_views = 4000 →
  increase_factor = 10 →
  additional_views = 50000 →
  let views_after_4_days := initial_views + increase_factor * initial_views in
  let total_views := views_after_4_days + additional_views in
  total_views = 94000 :=
by
  intros h_initial_views h_increase_factor h_additional_views
  have views_after_4_days_def : views_after_4_days = initial_views + increase_factor * initial_views
  have total_views_def : total_views = views_after_4_days + additional_views
  rw [h_initial_views, h_increase_factor, h_additional_views]
  rw [views_after_4_days_def, total_views_def]
  sorry

end video_views_l139_139928


namespace lcm_36_90_eq_180_l139_139979

theorem lcm_36_90_eq_180 : Nat.lcm 36 90 = 180 := 
by 
  sorry

end lcm_36_90_eq_180_l139_139979


namespace largest_possible_value_of_N_l139_139012

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l139_139012


namespace sum_of_squares_five_consecutive_not_perfect_square_l139_139099

theorem sum_of_squares_five_consecutive_not_perfect_square 
  (x : ℤ) : ¬ ∃ k : ℤ, (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 = k^2 :=
by 
  sorry

end sum_of_squares_five_consecutive_not_perfect_square_l139_139099


namespace largest_possible_value_of_N_l139_139006

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l139_139006


namespace monochromatic_triangle_min_6_monochromatic_triangle_min_2k_l139_139801

-- Define the problem conditions
variable (n : ℕ)

-- Define what constitutes a monochromatic triangle
def is_monochromatic_triangle (T : set (ℕ × ℕ)) (color : (ℕ × ℕ) → bool) : Prop :=
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a, b) ∈ T ∧ (b, c) ∈ T ∧ (a, c) ∈ T ∧
  (color (a, b) = color (b, c) ∧ color (b, c) = color (a, c))

-- Define the statement for n = 6
theorem monochromatic_triangle_min_6 :
  ∃ (S : ℕ), S = 1 ∧ ∀ (color : (ℕ × ℕ) → bool), min {S | ∀ T, T ⊆ (set.univ : set (ℕ × ℕ)) ∧
  is_monochromatic_triangle T color → true} = 1 :=
by sorry

-- Define the statement for n = 2k (k ≥ 4)
theorem monochromatic_triangle_min_2k (k : ℕ) (hk : k ≥ 4) :
  ∃ (S : ℕ), S = k * (k - 1) * (k - 2) / 3 ∧ ∀ (color : (ℕ × ℕ) → bool), min {S | ∀ T, T ⊆ (set.univ : set (ℕ × ℕ)) ∧
  is_monochromatic_triangle T color → true} = k * (k - 1) * (k - 2) / 3 :=
by sorry

end monochromatic_triangle_min_6_monochromatic_triangle_min_2k_l139_139801


namespace largest_angle_of_consecutive_integers_in_hexagon_l139_139473

theorem largest_angle_of_consecutive_integers_in_hexagon : 
  ∀ (a : ℕ), 
    (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) = 720 → 
    a + 3 = 122.5 :=
by sorry

end largest_angle_of_consecutive_integers_in_hexagon_l139_139473


namespace video_views_l139_139930

theorem video_views (initial_views : ℕ) (increase_factor : ℕ) (additional_views : ℕ) :
  initial_views = 4000 →
  increase_factor = 10 →
  additional_views = 50000 →
  let views_after_4_days := initial_views + increase_factor * initial_views in
  let total_views := views_after_4_days + additional_views in
  total_views = 94000 :=
by
  intros h_initial_views h_increase_factor h_additional_views
  have views_after_4_days_def : views_after_4_days = initial_views + increase_factor * initial_views
  have total_views_def : total_views = views_after_4_days + additional_views
  rw [h_initial_views, h_increase_factor, h_additional_views]
  rw [views_after_4_days_def, total_views_def]
  sorry

end video_views_l139_139930


namespace non_swimmers_play_soccer_80_percent_l139_139585

section WestvilleSummerRetreat

variables 
  (N : ℕ) -- total number of children
  (h_soccer : 0.7 * N) -- 70% of children play soccer
  (h_swim : 0.5 * N) -- 50% of children swim
  (h_soccer_swim : 0.3 * (0.7 * N)) -- 30% of soccer players also swim
  (h_basketball : 0.2 * N) -- 20% of children participate in basketball
  (h_basketball_soccer_not_swim : 0.25 * (0.2 * N)) -- 25% basketball players play soccer but do not swim

theorem non_swimmers_play_soccer_80_percent :
  let non_swimming_soccer_players := 0.7 * N - 0.3 * 0.7 * N - 0.25 * 0.2 * N in
  let non_swimmers := N - 0.5 * N + 0.25 * 0.2 * N in
  (non_swimming_soccer_players / non_swimmers) * 100 = 80 :=
by 
  have h1 : non_swimming_soccer_players = 0.44 * N := sorry,
  have h2 : non_swimmers = 0.55 * N := sorry,
  have h3 : (0.44 * N) / (0.55 * N) * 100 = 80 := by 
    rw [div_mul_eq_mul_div, mul_comm, mul_div_right_comm]
    exact rfl,
  exact h3

end WestvilleSummerRetreat

end non_swimmers_play_soccer_80_percent_l139_139585


namespace min_value_of_squares_l139_139663

theorem min_value_of_squares (x y z : ℝ) (h : x + y + z = 1) : x^2 + y^2 + z^2 ≥ 1 / 3 := sorry

end min_value_of_squares_l139_139663


namespace monotonic_increasing_a_eq_1_max_a_for_min_value_l139_139311

-- Part 1: Prove the function is monotonically increasing for a = 1
theorem monotonic_increasing_a_eq_1 : 
  ∀ x : ℝ,  ∃ (f : ℝ → ℝ), (f(x) = (x - 2) * exp (x - 1) - 1/2 * x^2 + x)
  ∧ (∀ x y : ℝ, x < y → f x ≤ f y) := 
sorry

-- Part 2: Prove maximum value of a such that min value of function in (0, +∞) is -1/2
theorem max_a_for_min_value : ∃ a : ℝ, 
  (∀ x : ℝ, x > 0 → (x - a - 1) * exp (x - 1) - 1/2 * x^2 + a * x ≥ -1/2)
  ∧ (∀ b : ℝ, (∀ x : ℝ, x > 0 → (x - b - 1) * exp (x - 1) - 1/2 * x^2 + b * x ≥ -1/2) → b ≤ (exp 1 / 2 - 1)) :=
sorry

end monotonic_increasing_a_eq_1_max_a_for_min_value_l139_139311


namespace center_number_is_4_l139_139577

-- Define the numbers and the 3x3 grid
inductive Square
| center | top_middle | left_middle | right_middle | bottom_middle

-- Define the properties of the problem
def isConsecutiveAdjacent (a b : ℕ) : Prop := 
  (a + 1 = b ∨ a = b + 1)

-- The condition to check the sum of edge squares
def sum_edge_squares (grid : Square → ℕ) : Prop := 
  grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28

-- The condition that the center square number is even
def even_center (grid : Square → ℕ) : Prop := 
  grid Square.center % 2 = 0

-- The main theorem statement
theorem center_number_is_4 (grid : Square → ℕ) :
  (∀ i j : Square, i ≠ j → isConsecutiveAdjacent (grid i) (grid j)) → 
  (grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28) →
  (grid Square.center % 2 = 0) →
  grid Square.center = 4 :=
by sorry

end center_number_is_4_l139_139577


namespace compute_dist_and_check_ratio_l139_139702

-- Define the given points
def point1 : ℝ × ℝ := (4, 3)
def point2 : ℝ × ℝ := (2, -3)

-- Calculate the horizontal and vertical distances
def horizontal_dist : ℝ := point1.1 - point2.1
def vertical_dist : ℝ := point1.2 - point2.2

-- Calculate the direct distance using the distance formula
def direct_dist : ℝ := Real.sqrt (horizontal_dist ^ 2 + vertical_dist ^ 2)

-- Define the ratio of the horizontal distance to the direct distance
def ratio : ℝ := horizontal_dist / direct_dist

-- Lean 4 statement to prove the direct distance and the ratio
theorem compute_dist_and_check_ratio :
  direct_dist = 2 * Real.sqrt 10 ∧ ratio = 1 / Real.sqrt 10 ∧ ¬(∃ n : ℤ, ratio = n) :=
by {
  sorry
}

end compute_dist_and_check_ratio_l139_139702


namespace math_problem_l139_139446

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 1000
  else -1000 / n * (Finset.range n).sum (λ k, sequence k)

theorem math_problem :
  (1 / 2^2) * sequence 0 +
  (Finset.finRange 1001).sum (λ n, 2^(n-1) * sequence n) = 250 :=
sorry

end math_problem_l139_139446


namespace min_S_n_at_24_l139_139991

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n : ℤ) * (2 * n - 48)

theorem min_S_n_at_24 : (∀ n : ℕ, n > 0 → S_n n ≥ S_n 24) ∧ S_n 24 < S_n 25 :=
by 
  sorry

end min_S_n_at_24_l139_139991


namespace cover_half_board_l139_139937

variable (n : ℕ) (k l : ℕ)
noncomputable def is_subgrid (k l : ℕ) := k + l ≥ n

theorem cover_half_board (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ (k_i l_i : ℕ), is_subgrid n k_i l_i ∧ k_i + l_i ≥ n) :
  ∑ i in finset.range n, (k + l) ≥ n^2 →
  2 * (∑ i in finset.range n, (k + l) - n) ≥ n^2 :=
sorry

end cover_half_board_l139_139937


namespace polar_equation_of_circle_c_range_of_op_oq_l139_139376

noncomputable def circle_param_eq (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

noncomputable def line_kl_eq (θ : ℝ) : ℝ :=
  3 * Real.sqrt 3 / (Real.sin θ + Real.sqrt 3 * Real.cos θ)

theorem polar_equation_of_circle_c :
  ∀ θ : ℝ, ∃ ρ : ℝ, ρ = 2 * Real.cos θ :=
by sorry

theorem range_of_op_oq (θ₁ : ℝ) (hθ : 0 < θ₁ ∧ θ₁ < Real.pi / 2) :
  0 < (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) ∧
  (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) < 6 :=
by sorry

end polar_equation_of_circle_c_range_of_op_oq_l139_139376


namespace lucas_min_deliveries_l139_139792

theorem lucas_min_deliveries (cost_of_scooter earnings_per_delivery fuel_cost_per_delivery parking_fee_per_delivery : ℕ)
  (cost_eq : cost_of_scooter = 3000)
  (earnings_eq : earnings_per_delivery = 12)
  (fuel_cost_eq : fuel_cost_per_delivery = 4)
  (parking_fee_eq : parking_fee_per_delivery = 1) :
  ∃ d : ℕ, 7 * d ≥ cost_of_scooter ∧ d = 429 := by
  sorry

end lucas_min_deliveries_l139_139792


namespace montana_more_than_ohio_l139_139424

-- Define the total number of combinations for Ohio and Montana
def ohio_combinations : ℕ := 26^4 * 10^3
def montana_combinations : ℕ := 26^5 * 10^2

-- The total number of combinations from both states
def ohio_total : ℕ := ohio_combinations
def montana_total : ℕ := montana_combinations

-- Prove the difference
theorem montana_more_than_ohio : montana_total - ohio_total = 731161600 := by
  sorry

end montana_more_than_ohio_l139_139424


namespace minimum_value_l139_139223

noncomputable def minSum (a b c : ℝ) : ℝ :=
  a / (3 * b) + b / (6 * c) + c / (9 * a)

theorem minimum_value (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  ∃ x : ℝ, (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → minSum a b c ≥ x) ∧ x = 3 / Real.cbrt 162 :=
sorry

end minimum_value_l139_139223


namespace tangent_line_at_origin_l139_139772

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x

theorem tangent_line_at_origin (h_even : ∀ x, (3 * x^2 + 2 * a * x + (a - 3)) = 3 * (-x)^2 + 2 * a * (-x) + (a - 3)) :
  (∃ m b : ℝ, ∀ x, m = 0 ∧ b = -3) :=
sorry

end tangent_line_at_origin_l139_139772


namespace distinct_values_count_less_than_n_l139_139153

variables {α : Type*} (S : Finset α) (f : Finset α → ℕ)

-- Property 1: f(A) = f(S - A)
def symmetric (A : Finset α) : Prop :=
  f(A) = f(S \ A)

-- Property 2: max(f(A), f(B)) ≥ f(A ∪ B)
def subadditive (A B : Finset α) : Prop :=
  max f(A) f(B) ≥ f(A ∪ B)

theorem distinct_values_count_less_than_n
  (n : ℕ)
  (hs : S.card = n)
  (h1 : ∀ A : Finset α, symmetric S f A)
  (h2 : ∀ A B : Finset α, subadditive S f A B)
  : (∃ (k : ℕ), k < n ∧ (∀ x : ℕ, x ∈ f '' S.powerset → x < k)) :=
sorry

end distinct_values_count_less_than_n_l139_139153


namespace ratio_surface_area_cone_l139_139902

theorem ratio_surface_area_cone (side_length : ℝ) (angle : ℝ) (h1 : side_length = 4) (h2 : angle = 30) :
  let r := side_length / 2,
      l := r * √3,
      S1 := π * r * l,
      S2 := π * r^2 + π * r * l in
  S2 / S1 = 3 / 2 :=
by {
  -- The problem statement in Lean as required.
  -- Proof would proceed by asserting the correctness of given operations.
  sorry
}

end ratio_surface_area_cone_l139_139902


namespace problem_l139_139342

def S (n : ℕ) : ℤ :=
  (List.range n).sum (λ k => if k % 2 = 0 then (k + 1 : ℕ) else -((k + 1) : ℕ))

theorem problem (h1 : S 17 = 9)
               (h2 : S 33 = 17)
               (h3 : S 50 = -25) :
  S 17 + S 33 + S 50 = 1 :=
by
  sorry

end problem_l139_139342


namespace triangle_identity_l139_139025

theorem triangle_identity (A B C D : Point)
  (hD_on_BC : D ∈ Line.segment B C)
  (AB AC AD BC BD DC : ℝ)
  (hAB : dist A B = AB)
  (hAC : dist A C = AC)
  (hAD : dist A D = AD)
  (hBC : dist B C = BC)
  (hBD : dist B D = BD)
  (hDC : dist D C = DC) :
  AB^2 * DC + AC^2 * BD - AD^2 * BC = BC * DC * BD :=
sorry

end triangle_identity_l139_139025


namespace min_gloves_proof_l139_139832

-- Let n represent the number of participants
def n : Nat := 63

-- Let g represent the number of gloves per participant
def g : Nat := 2

-- The minimum number of gloves required
def min_gloves : Nat := n * g

theorem min_gloves_proof : min_gloves = 126 :=
by 
  -- Placeholder for the proof
  sorry

end min_gloves_proof_l139_139832


namespace standard_deviation_bound_l139_139491

theorem standard_deviation_bound (mu sigma : ℝ) (h_mu : mu = 51) (h_ineq : mu - 3 * sigma > 44) : sigma < 7 / 3 :=
by
  sorry

end standard_deviation_bound_l139_139491


namespace sergeant_can_balance_recruits_l139_139558

theorem sergeant_can_balance_recruits (orientation : List ℕ) :
  ∃ k, let (m_left, n_right) := orientation.splitAt k in
       m_left.filter (· = 0).length = n_right.filter (· = 1).length :=
  sorry

end sergeant_can_balance_recruits_l139_139558


namespace probability_MAME_on_top_l139_139873

theorem probability_MAME_on_top : 
  let num_sections := 8
  let favorable_outcome := 1
  (favorable_outcome : ℝ) / (num_sections : ℝ) = 1 / 8 :=
by 
  sorry

end probability_MAME_on_top_l139_139873


namespace find_triples_l139_139630

theorem find_triples (x y z : ℕ) :
  (1 / x + 2 / y - 3 / z = 1) ↔ 
  ((x = 2 ∧ y = 1 ∧ z = 2) ∨
   (x = 2 ∧ y = 3 ∧ z = 18) ∨
   ∃ (n : ℕ), n ≥ 1 ∧ x = 1 ∧ y = 2 * n ∧ z = 3 * n ∨
   ∃ (k : ℕ), k ≥ 1 ∧ x = k ∧ y = 2 ∧ z = 3 * k) := sorry

end find_triples_l139_139630


namespace total_views_correct_l139_139925

-- Definitions based on the given conditions
def initial_views : ℕ := 4000
def views_increase := 10 * initial_views
def additional_views := 50000
def total_views_after_6_days := initial_views + views_increase + additional_views

-- The theorem we are going to state
theorem total_views_correct :
  total_views_after_6_days = 94000 :=
sorry

end total_views_correct_l139_139925


namespace sum_of_roots_l139_139985

theorem sum_of_roots :
  let eq1 := 3 * x^3 + 2 * x^2 - 9 * x + 15
  let eq2 := 4 * x^3 - 16 * x^2 + x + 4
  ∑ (root ∈ roots eq1) + ∑ (root ∈ roots eq2) = 10 / 3 :=
by
  sorry

end sum_of_roots_l139_139985


namespace power_function_form_explicit_form_of_function_l139_139831

noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x, x^α

theorem power_function_form (α : ℝ) (h : power_function α 2 = 1/8) : α = -3 :=
by {
  sorry
}

theorem explicit_form_of_function : power_function (-3) = λ x, x ^ -3 :=
by {
  sorry
}

end power_function_form_explicit_form_of_function_l139_139831


namespace distance_center_to_plane_l139_139301

theorem distance_center_to_plane (R r d : ℝ) (surface_area_sphere : 4 * π * R^2 = 8 * π)
  (A B C : ℝ → ℝ) (dist_AB : A B = sqrt 2) (dist_AC : A C = sqrt 2) (dist_BC : B C = 2) :
  d = 1 :=
by
  sorry

end distance_center_to_plane_l139_139301


namespace person_c_start_time_l139_139584

noncomputable def trisect_points (A B : ℝ) : (ℝ × ℝ) := (A + (B - A)/3, A + 2*(B - A)/3)

def personA_velocity (AB : ℝ) : ℝ := (2/3) * AB / 24
def personB_velocity (AB : ℝ) : ℝ := (1/3) * AB / 12
def personC_velocity (AB : ℝ) : ℝ := (1/3) * AB / 8

theorem person_c_start_time (A B : ℝ) (t_A_start : ℝ) (t_B_start : ℝ) (t_meeting_C : ℝ) (t_meeting_AC : ℝ):
  let t_C_start := t_meeting_AC - 8 in
  let C := trisect_points A B in
  let D := trisect_points A B in
  t_A_start = 0 → t_B_start = 12 → t_meeting_C = 24 →
  ∀ (t_C_start : ℝ), t_meeting_AC = 30 → personC_velocity (B - A) * 8 = D.1 - B →
  t_C_start = 22 :=
  by {
    intros,
    sorry
  }

end person_c_start_time_l139_139584


namespace inequality_proof_l139_139781

noncomputable theory

open Nat

-- The statement of the problem
theorem inequality_proof (n k : ℕ) (hn : n > 0) (hk : k > 0) (h : n > k) :
  (1 / (n+1 : ℝ)) * (n^n / (k^k * (n-k)^(n-k) : ℝ)) < 
    (nat.factorial n : ℝ) / ((nat.factorial k) * (nat.factorial (n-k)) : ℝ) ∧
  (nat.factorial n : ℝ) / ((nat.factorial k) * (nat.factorial (n-k)) : ℝ) <
    n^n / (k^k * (n-k)^(n-k) : ℝ) :=
sorry

end inequality_proof_l139_139781


namespace marcus_baseball_cards_l139_139794

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the number of additional cards Marcus has compared to Carter
def additional_cards : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + additional_cards

-- The proof statement asserting Marcus' total number of baseball cards
theorem marcus_baseball_cards : marcus_cards = 210 :=
by {
  -- This is where the proof steps would go, but we are skipping with sorry
  sorry
}

end marcus_baseball_cards_l139_139794


namespace initial_stock_of_books_l139_139571

theorem initial_stock_of_books
  (acquired_books : ℝ)
  (shelves : ℕ)
  (books_per_shelf : ℝ)
  (total_books_on_shelves : ℝ)
  (initial_stock : ℝ) : 
  acquired_books = 20.0 →
  shelves = 15 →
  books_per_shelf = 4.0 →
  total_books_on_shelves = shelves * books_per_shelf →
  initial_stock = total_books_on_shelves - acquired_books →
  initial_stock = 40.0 :=
by
  intros h_acquired h_shelves h_books_per_shelf h_total_books h_initial_stock
  rw [h_acquired, h_shelves, h_books_per_shelf]
  sorry

end initial_stock_of_books_l139_139571


namespace sum_first_50_digits_of_one_div_10101_l139_139878

theorem sum_first_50_digits_of_one_div_10101 : 
  let fractional_part := Nat.digits 10 (1 % 10101) in
  let relevant_digits := List.take 50 fractional_part in
  List.sum relevant_digits = 180 :=
sorry

end sum_first_50_digits_of_one_div_10101_l139_139878


namespace find_g5_l139_139055

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l139_139055


namespace possible_values_of_a_l139_139688

theorem possible_values_of_a (a : ℝ) :
  (∀ x ∈ set.Icc a (a + 6), (x^2 + 2 * x + 1) ≥ 9) ∧ 
  (∃ x ∈ set.Icc a (a + 6), (x^2 + 2 * x + 1) = 9) ↔ 
  a = 2 ∨ a = -10 := 
by
  sorry

end possible_values_of_a_l139_139688


namespace minimum_sum_am_gm_l139_139226

theorem minimum_sum_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ (1 / 2) :=
sorry

end minimum_sum_am_gm_l139_139226


namespace average_speed_for_entire_trip_l139_139337

-- Definitions as per conditions
def total_distance : ℝ := 60
def first_half_distance : ℝ := total_distance / 2
def first_half_speed : ℝ := 24
def second_half_speed : ℝ := first_half_speed + 16
def second_half_distance : ℝ := first_half_distance

-- Calculate times for each half of the journey
def time_first_half := first_half_distance / first_half_speed
def time_second_half := second_half_distance / second_half_speed
def total_time := time_first_half + time_second_half

-- Total distance already defined above

-- Statement to prove
theorem average_speed_for_entire_trip :
  (total_distance / total_time) = 30 := by
  sorry

end average_speed_for_entire_trip_l139_139337


namespace length_CD_l139_139505

structure Trapezoid :=
(ABCD : Type)
(AD BC : ℝ)
(AC: ℝ := 2)
(CAB BDC : ℝ)
(AD_parallel_BC : AD = BC)
(Angle_CAB : CAB = real.pi / 6)
(Angle_BDC : BDC = real.pi / 3)
(Ratio_BC_AD : BC / AD = 3 / 2)

theorem length_CD (t : Trapezoid) : 
  CD t.ABCD  = 3 :=
by
  sorry

end length_CD_l139_139505


namespace ratio_friend_to_remaining_l139_139575

-- Define the initial conditions and variables
def total_cranes := 1000
def alice_folded := total_cranes / 2
def remaining_cranes := total_cranes - alice_folded
def alice_still_needs := 400
def friends_folded := remaining_cranes - alice_still_needs
def ratio := friends_folded : remaining_cranes

-- State the theorem to prove
theorem ratio_friend_to_remaining : ratio = 1 / 5 := by
  sorry

end ratio_friend_to_remaining_l139_139575


namespace g_five_eq_248_l139_139048

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l139_139048


namespace red_peaches_in_basket_l139_139103

theorem red_peaches_in_basket (total_peaches green_peaches : ℕ) (h1 : total_peaches = 10) (h2 : green_peaches = 6) :
  total_peaches - green_peaches = 4 :=
by
  rw [h1, h2]
  exact Nat.sub_self


end red_peaches_in_basket_l139_139103


namespace minimum_n_value_l139_139273

def satisfies_terms_condition (n : ℕ) : Prop :=
  (n + 1) * (n + 1) ≥ 2021

theorem minimum_n_value :
  ∃ n : ℕ, n > 0 ∧ satisfies_terms_condition n ∧ ∀ m : ℕ, m > 0 ∧ satisfies_terms_condition m → n ≤ m := by
  sorry

end minimum_n_value_l139_139273


namespace total_population_estimate_l139_139747

theorem total_population_estimate :
  let n := 30
  let avg_min := 5200
  let avg_max := 5700
  let avg := (avg_min + avg_max) / 2
  n * avg = 163500 :=
by
  let n := 30
  let avg_min := 5200
  let avg_max := 5700
  let avg := (avg_min + avg_max) / 2
  have total := n * avg
  show total = 163500
  sorry

end total_population_estimate_l139_139747


namespace parallel_vectors_eq_l139_139330

theorem parallel_vectors_eq (x : ℝ) :
  let a := (x, 1)
  let b := (2, 4)
  (a.1 / b.1 = a.2 / b.2) → x = 1 / 2 :=
by
  intros h
  sorry

end parallel_vectors_eq_l139_139330


namespace be_length_l139_139389

theorem be_length {A B C D E : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : A) (BC : B) (CA : C)
  (AB_eq : dist A B = 14) (BC_eq : dist B C = 17) (CA_eq : dist C A = 15)
  (D : B) (CD_eq : dist C D = 7)
  (E : B) (BAE_eq_CAD : ∠ A B E = ∠ C A D) :
  dist B E = 11662 / 1811 :=
begin
  sorry
end

end be_length_l139_139389


namespace sequence_general_formula_l139_139992

theorem sequence_general_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2 - 2 * n + 2):
  (a 1 = 1) ∧ (∀ n, 1 < n → a n = S n - S (n - 1)) → 
  (∀ n, a n = if n = 1 then 1 else 2 * n - 3) :=
by
  intro h
  sorry

end sequence_general_formula_l139_139992


namespace solve_x_l139_139631

theorem solve_x : ∃ x : ℝ, (2^(3*x + 2)) * (4^(2*x + 1)) = 8^(3*x + 4) ∧ x = -4 := 
by
  sorry

end solve_x_l139_139631


namespace limit_sqrt3_l139_139980

theorem limit_sqrt3 :
  (∀ n : ℕ, limit_n_numerator_denominator_expr n = 
  (∑ k in finset.range(n+1), nat.choose (2*n) (2*k) * 3^k : ℝ) / 
  (∑ k in finset.range(n), nat.choose (2*n) (2*k+1) * 3^k : ℝ)) → 
  (∀ n : ℕ, limit_n_numerator_denominator_expr n) = sqrt 3 := 
begin
  sorry
end

end limit_sqrt3_l139_139980


namespace find_a_range_l139_139818

-- Define the function f
def f (x : ℝ) (a : ℝ) := x^3 - 3*x - a

-- Define the function g
def g (x : ℝ) (a : ℝ) := x^2 - a * Real.log x

-- Define p, q and the range of a
def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ set.Icc (-1/2 : ℝ) (Real.sqrt 3) ∧ f x a = 0

def q (a : ℝ) : Prop := 
  0 < a ∧ ∀ x : ℝ, x ∈ set.Ioo 0 (a / 2) → deriv (λ x, g x a) x < 0

theorem find_a_range (a : ℝ) :
  (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a) ↔ a ∈ set.Icc (-2 : ℝ) 0 ∪ set.Ioc (11 / 8) 2 :=
sorry

end find_a_range_l139_139818


namespace find_P_l139_139948

-- Definitions of midpoints
def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

-- Conditions from the problem
def midpoint_PQ : ℝ × ℝ × ℝ := (3, 2, 5)
def midpoint_PR : ℝ × ℝ × ℝ := (0, -1, 4)
def midpoint_QR : ℝ × ℝ × ℝ := (1, 3, 6)

-- Theorem to find the coordinates of P
theorem find_P (Q R : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ) :
    midpoint P Q = midpoint_PQ →
    midpoint P R = midpoint_PR →
    midpoint Q R = midpoint_QR →
    P = (-2, 0, 5) :=
by
  intros hPQ hPR hQR
  sorry

end find_P_l139_139948


namespace bisection_ratio_l139_139170

noncomputable def convexPolygon := sorry -- Define the convex polygon (details abstracted)

def line_l (P : convexPolygon) : set (ℝ × ℝ) := sorry -- Line l that bisects the area of the convex polygon

def projection (P : convexPolygon) (l : set (ℝ × ℝ)) : segment := sorry 
-- Projection of the polygon onto the line perpendicular to l, yielding segment AC

def perpendicularLine (l : set (ℝ × ℝ)) : set (ℝ × ℝ) := sorry
-- Line perpendicular to l

theorem bisection_ratio (P : convexPolygon) (l : set (ℝ × ℝ))
  (h_bisect : bisects P l) : 
  let proj_AC := projection P (perpendicularLine l) in
  proj_AC.ratio ≤ 1 + Real.sqrt 2 := 
sorry

end bisection_ratio_l139_139170


namespace cyclic_quadrilateral_lines_intersect_on_circle_l139_139780

variables (g : Line) (k1 k2 k3 : Circle)
variables (A B C D : Point)
variables (M1 M2 M3 : Point) -- Centers of k1, k2, k3 respectively

-- Given conditions
def conditions : Prop :=
  (k1 ∣∣ g ∧ k2 ∣∣ g ∧ k3 ∣∣ k1 ∧ k3 ∣∣ k2 ∧ k1.affine_hull ∣ g ∧ k2.affine_hull ∣ g ∧ k3.affine_hull ∈ ∥ k1.affine_hull ∧ k2.affine_hull)

-- Statement (a)
theorem cyclic_quadrilateral (h : conditions g k1 k2 k3 A B C D) :
  is_cyclic_quadrilateral A B C D :=
sorry

-- Statement (b)
theorem lines_intersect_on_circle (h : conditions g k1 k2 k3 A B C D) :
  intersects_on_circle (line_through B C) (line_through A D) k3 :=
sorry

end cyclic_quadrilateral_lines_intersect_on_circle_l139_139780


namespace katie_clock_l139_139761

theorem katie_clock (t_clock t_actual : ℕ) :
  t_clock = 540 →
  t_actual = (540 * 60) / 37 →
  8 * 60 + 875 = 22 * 60 + 36 :=
by
  intros h1 h2
  have h3 : 875 = (540 * 60 / 37) := sorry
  have h4 : 8 * 60 + 875 = 480 + 875 := sorry
  have h5 : 480 + 875 = 22 * 60 + 36 := sorry
  exact h5

end katie_clock_l139_139761


namespace solve_cubic_equation_l139_139629

theorem solve_cubic_equation (x : ℝ) :
  (∛(17*x - 2) + ∛(11*x + 2) = 2 * ∛(9*x)) ↔
  (x = 0 ∨ x = (2 + Real.sqrt 35) / 31 ∨ x = (2 - Real.sqrt 35) / 31) :=
  sorry

end solve_cubic_equation_l139_139629


namespace length_side_AB_is_4_l139_139147

-- Defining a triangle ABC with area 6
variables {A B C K L Q : Type*}
variables {side_AB : Float} {ratio_K : Float} {ratio_L : Float} {dist_Q : Float}
variables (area_ABC : ℝ := 6) (ratio_AK_BK : ℝ := 2 / 3) (ratio_AL_LC : ℝ := 5 / 3)
variables (dist_Q_to_AB : ℝ := 1.5)

theorem length_side_AB_is_4 : 
  side_AB = 4 → 
  (area_ABC = 6 ∧ ratio_AK_BK = 2 / 3 ∧ ratio_AL_LC = 5 / 3 ∧ dist_Q_to_AB = 1.5) :=
by
  sorry

end length_side_AB_is_4_l139_139147


namespace goods_train_length_l139_139559

--Definitions for the given problem
def man_train_speed_kmph : ℝ := 36
def goods_train_speed_kmph : ℝ := 50.4
def passing_time_seconds : ℝ := 10

-- Conversions and calculations
def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)
def relative_speed_kmph : ℝ := man_train_speed_kmph + goods_train_speed_kmph
def relative_speed_mps : ℝ := kmph_to_mps relative_speed_kmph

-- Theorem statement to prove the length of the goods train
theorem goods_train_length : relative_speed_mps * passing_time_seconds = 1200 := by
  sorry

end goods_train_length_l139_139559


namespace product_of_roots_l139_139634

theorem product_of_roots (p1 : Polynomial ℝ) (p2 : Polynomial ℝ) 
  (h1 : p1 = 3 * X^3 + 2 * X^2 - 9 * X + 25) 
  (h2 : p2 = 7 * X^4 - 28 * X^3 + 60 * X^2 + 3 * X - 15) : 
  Polynomial.roots (p1 * p2).prod = -125 / 7 :=
by
  sorry

end product_of_roots_l139_139634


namespace largest_angle_in_consecutive_integer_hexagon_l139_139478

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l139_139478


namespace isosceles_triangle_of_equal_bisectors_l139_139806

variable {A B C O M N : Type*}
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O] [Inhabited M] [Inhabited N]

-- Assume we have a triangle and bisectors satisfying the given conditions
variable (triangle_ABC : Triangle A B C)
variable (angle_bisectors_AM_CN : AngleBisectors triangle_ABC AM CN)
variable (intersection_point_of_bisectors : Intersection O AM CN)
variable (equality_of_segments_OM_ON : OM = ON)

-- To prove that the triangle is isosceles
theorem isosceles_triangle_of_equal_bisectors :
  IsoscelesTriangle triangle_ABC :=
sorry

end isosceles_triangle_of_equal_bisectors_l139_139806


namespace garden_area_not_occupied_l139_139967

theorem garden_area_not_occupied (length width : ℝ) (π : ℝ)
  (h_length : length = 30) (h_width : width = 24)
  (h_pi : π = real.pi) :
  let area_rectangle := length * width,
      radius := width / 4, 
      single_bed_area := π * radius^2,
      total_beds_area := 4 * single_bed_area,
      area_not_occupied := area_rectangle - total_beds_area
  in area_not_occupied = 720 - 144 * π :=
by 
  intros
  rw [h_length, h_width, h_pi]
  simp [area_rectangle, radius, single_bed_area, total_beds_area, area_not_occupied]
  norm_num

end garden_area_not_occupied_l139_139967


namespace complex_values_l139_139678

open Complex

theorem complex_values (z : ℂ) (h : z ^ 3 + z = 2 * (abs z) ^ 2) :
  z = 0 ∨ z = 1 ∨ z = -1 + 2 * Complex.I ∨ z = -1 - 2 * Complex.I :=
by sorry

end complex_values_l139_139678


namespace exists_functions_f_g_l139_139668

theorem exists_functions_f_g :
  ∃ (f g : ℝ → ℝ), 
    (∀ x : ℝ, ¬(f (-x) = f x ∨ f (-x) = -f x)) ∧ 
    (∀ x : ℝ, ¬(g (-x) = g x ∨ g (-x) = -g x)) ∧
    (∀ x : ℝ, (f x * g x) = (f (-x) * g (-x))) ∧
    f = (λ x, x + 1) ∧ g = (λ x, x - 1) := 
by
  use (λ x, x + 1), (λ x, x - 1)
  sorry

end exists_functions_f_g_l139_139668


namespace correct_mark_l139_139826

theorem correct_mark 
  (avg_wrong : ℝ := 60)
  (wrong_mark : ℝ := 90)
  (num_students : ℕ := 30)
  (avg_correct : ℝ := 57.5) :
  (wrong_mark - (avg_wrong * num_students - avg_correct * num_students)) = 15 :=
by
  sorry

end correct_mark_l139_139826


namespace problem_statement_a_problem_statement_c_l139_139343

def poly (x : ℝ) (n : ℕ) : ℝ :=
  (1 - 2*x)^(2*n)

lemma expansion_correct (a : ℕ → ℝ) (x : ℝ) (n : ℕ) :
  poly x n = ∑ i in finset.range (2*n + 1), a i * x^i :=
  sorry

theorem problem_statement_a (a : ℕ → ℝ) (n : ℕ) (x : ℝ) :
  expansion_correct a x n → a 0 = 1 :=
  sorry

theorem problem_statement_c (n : ℕ) :
  ∑ i in finset.range (n + 1), (nat.choose n i)^2 = nat.choose (2 * n) n :=
  sorry

end problem_statement_a_problem_statement_c_l139_139343


namespace expected_winnings_is_correct_l139_139180

noncomputable def peculiar_die_expected_winnings : ℝ :=
  (1/4) * 2 + (1/2) * 5 + (1/4) * (-10)

theorem expected_winnings_is_correct :
  peculiar_die_expected_winnings = 0.5 := by
  sorry

end expected_winnings_is_correct_l139_139180


namespace divide_loot_l139_139532

theorem divide_loot (n : ℕ) (valuations : fin n → set ℝ) (valuation_function : fin n → set ℝ → ℝ → Prop) : 
  (∀ i : fin n, ∃ share : set ℝ, valuation_function i share (1 / n)) :=
sorry

end divide_loot_l139_139532


namespace center_of_mass_combined_is_weighted_sum_l139_139028

-- Define the mass summations
variables {n m : ℕ}
variables {a_i : fin n → ℝ} {b_j : fin m → ℝ}
variables (a := finset.univ.sum a_i) (b := finset.univ.sum b_j)

-- Define the points
variables {X_i : fin n → point} {Y_j : fin m → point}

-- Define the center of mass of X and Y
def center_mass_X (a_i : fin n → ℝ) (X_i : fin n → point) (a : ℝ) : point :=
  (1 / a) • (finset.univ.sum (λ i, a_i i • X_i i))

def center_mass_Y (b_j : fin m → ℝ) (Y_j : fin m → point) (b : ℝ) : point :=
  (1 / b) • (finset.univ.sum (λ j, b_j j • Y_j j))

-- Define the combined center of mass
def combined_center_mass (a_i : fin n → ℝ) (X_i : fin n → point) (b_j : fin m → ℝ) (Y_j : fin m → point) (a b : ℝ) : point :=
  (1 / (a + b)) • (finset.univ.sum (λ i, a_i i • X_i i) + finset.univ.sum (λ j, b_j j • Y_j j))

-- The target theorem
theorem center_of_mass_combined_is_weighted_sum :
  combined_center_mass a_i X_i b_j Y_j a b
  = (1 / (a + b)) • (a • center_mass_X a_i X_i a + b • center_mass_Y b_j Y_j b) :=
sorry

end center_of_mass_combined_is_weighted_sum_l139_139028


namespace positive_root_exists_iff_p_range_l139_139957

theorem positive_root_exists_iff_p_range (p : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x^4 + 4 * p * x^3 + x^2 + 4 * p * x + 4 = 0) ↔ 
  p ∈ Set.Iio (-Real.sqrt 2 / 2) ∪ Set.Ioi (Real.sqrt 2 / 2) :=
by
  sorry

end positive_root_exists_iff_p_range_l139_139957


namespace equal_chords_l139_139169

-- Definitions
structure Circle (P : Type u) [MetricSpace P] :=
(center : P)
(radius : ℝ)
(proof_pos : 0 < radius)

variables {P : Type u} [MetricSpace P]

-- Given a circle k with center O and radius r
variable (k : Circle P)
variable [inst : inhabited P]
include inst

-- Given points A and B on the circle k
variables (A B : P)
variable (on_circle_A : Metric.dist A k.center = k.radius)
variable (on_circle_B : Metric.dist B k.center = k.radius)

-- Given a line e which intersects k at points A and B
structure Line (P : Type u) [MetricSpace P] :=
(point : P → Prop)

variable (e : Line P)
variable (intersect_A : e.point A)
variable (intersect_B : e.point B)

-- Given an arbitrary line f passing through point A
variables (f : Line P)
variable (pass_through_A : f.point A)

-- Given f intersects k at another point C1
variable (C1 : P)
variable (on_circle_C1 : Metric.dist C1 k.center = k.radius)
variable (intersect_f_C1 : f.point C1)

-- Given the reflection of f over e intersects k at another point C2
variable (C2 : P)
variable (on_circle_C2 : Metric.dist C2 k.center = k.radius)
variable (reflection_intersect_C2 : true) -- Abstracting the reflection operation

-- Theorem statement to prove BC1 = BC2 under the given conditions
theorem equal_chords : Metric.dist B C1 = Metric.dist B C2 :=
sorry

end equal_chords_l139_139169


namespace geo_seq_sum_monotone_l139_139365

theorem geo_seq_sum_monotone (q a1 : ℝ) (n : ℕ) (S : ℕ → ℝ) :
  (∀ n, S (n + 1) > S n) ↔ (a1 > 0 ∧ q > 0) :=
sorry -- Proof of the theorem (omitted)

end geo_seq_sum_monotone_l139_139365


namespace domain_of_f_odd_function_a_one_monotonically_decreasing_l139_139314

namespace MathProof

def f (x a : ℝ) : ℝ := (2^x + a) / (2^x - 1)

-- Domain proof statement
theorem domain_of_f : {x : ℝ | f x 1} = {x : ℝ | x ≠ 0} := sorry

-- Odd function implies a = 1 proof statement
theorem odd_function_a_one (h : ∀ x, f (-x) = -f x) : a = 1 := sorry

-- Monotonically decreasing proof statement
theorem monotonically_decreasing (h : ∀ x, f (-x) = -f x) : 
  ∀ x1 x2, 0 < x1 ∧ x1 < x2 → f x1 1 > f x2 1 := sorry

end MathProof

end domain_of_f_odd_function_a_one_monotonically_decreasing_l139_139314


namespace ratio_of_cube_dimensions_l139_139555

theorem ratio_of_cube_dimensions (V_original V_larger : ℝ) (hV_org : V_original = 64) (hV_lrg : V_larger = 512) :
  (∃ r : ℝ, r^3 = V_larger / V_original) ∧ r = 2 := 
sorry

end ratio_of_cube_dimensions_l139_139555


namespace base10_to_base8_conversion_l139_139602

theorem base10_to_base8_conversion (n : ℕ) (h₁ : n = 512) : nat.to_digits 8 n = [1, 0, 0, 0] :=
by {
  rw h₁,
  simp,
  sorry
}

end base10_to_base8_conversion_l139_139602


namespace describe_geometric_figure_as_hyperbola_l139_139420

noncomputable def geometric_figure (m n : ℝ) (z : ℂ) : Prop :=
  ∀ (i : ℂ) (ni mi : ℂ), 
    i * i = -1 ∧ 
    ni = n * i ∧ 
    mi = m * i ∧ 
    |z + ni| + |z - mi| = n ∧ 
    |z + ni| - |z - mi| = -m →
    -- Prove that the equations describe a hyperbola
    ∃ (a b : ℝ),  (a ≠ 0 ∧ b ≠ 0) ∧ (a * x^2 - b * y^2 = 1)

-- Statement:
theorem describe_geometric_figure_as_hyperbola (m n : ℝ) (z : ℂ) :
  geometric_figure m n z := sorry

end describe_geometric_figure_as_hyperbola_l139_139420


namespace part_a_l139_139541

theorem part_a (p : ℕ → ℕ → ℝ) (m : ℕ) (hm : m ≥ 1) : p m 0 = (3 / 4) * p (m-1) 0 + (1 / 2) * p (m-1) 2 + (1 / 8) * p (m-1) 4 :=
by
  sorry

end part_a_l139_139541


namespace general_term_a_sum_reciprocals_T_l139_139654

variable {n : ℕ} (a S : ℕ → ℕ) (b T : ℕ → ℕ)

-- Condition for sequence {a_n}
axiom a1 : a 1 = 2
axiom a_rec : ∀ n : ℕ, 1 ≤ n → a (n + 1) = S n + 2
axiom S_def : ∀ n : ℕ, S n = ∑ i in Finset.range n, a (i + 1)

-- Question 1: General term for {a_n}
theorem general_term_a : ∀ n : ℕ, 1 ≤ n → a n = 2 ^ n := 
sorry

-- Conditions for the arithmetic sequence {b_n}
axiom b2 : b 2 = 2
axiom b4 : b 4 = 4
axiom a1_eq_b2 : a 1 = b 2
axiom a2_eq_b4 : a 2 = b 4
axiom T_def : ∀ n : ℕ, T n = (n * (n + 1)) / 2

-- Question 2: Sum of reciprocals of T_n
theorem sum_reciprocals_T 
    : ∀ n : ℕ, 1 ≤ n → (∑ i in Finset.range n, (1 : ℚ) / T (i + 1)) = (2 * n) / (n + 1) :=
sorry

end general_term_a_sum_reciprocals_T_l139_139654


namespace last_row_number_l139_139828

/-
Given:
1. Each row forms an arithmetic sequence.
2. The common differences of the rows are:
   - 1st row: common difference = 1
   - 2nd row: common difference = 2
   - 3rd row: common difference = 4
   - ...
   - 2015th row: common difference = 2^2014
3. The nth row starts with \( (n+1) \times 2^{n-2} \).

Prove:
The number in the last row (2016th row) is \( 2017 \times 2^{2014} \).
-/
theorem last_row_number
  (common_diff : ℕ → ℕ)
  (h1 : common_diff 1 = 1)
  (h2 : common_diff 2 = 2)
  (h3 : common_diff 3 = 4)
  (h_general : ∀ n, common_diff n = 2^(n-1))
  (first_number_in_row : ℕ → ℕ)
  (first_number_in_row_def : ∀ n, first_number_in_row n = (n + 1) * 2^(n - 2)) :
  first_number_in_row 2016 = 2017 * 2^2014 := by
    sorry

end last_row_number_l139_139828


namespace lunch_break_duration_l139_139032

def rate_sandra : ℝ := 0 -- Sandra's painting rate in houses per hour
def rate_helpers : ℝ := 0 -- Combined rate of the three helpers in houses per hour
def lunch_break : ℝ := 0 -- Lunch break duration in hours

axiom monday_condition : (8 - lunch_break) * (rate_sandra + rate_helpers) = 0.6
axiom tuesday_condition : (6 - lunch_break) * rate_helpers = 0.3
axiom wednesday_condition : (2 - lunch_break) * rate_sandra = 0.1

theorem lunch_break_duration : lunch_break = 0.5 :=
by {
  sorry
}

end lunch_break_duration_l139_139032


namespace sqrt_fifth_root_div_fraction_l139_139241

theorem sqrt_fifth_root_div_fraction : 
  (∃ (x y : ℚ), x = 16.2 ∧ y = \(16.2) / 5 ∧ \sqrt[5]{9 / x} = \sqrt[5]{5/9}) :=
sorry

end sqrt_fifth_root_div_fraction_l139_139241


namespace distance_C_to_D_l139_139386

noncomputable def side_length_smaller_square (perimeter : ℝ) : ℝ := perimeter / 4
noncomputable def side_length_larger_square (area : ℝ) : ℝ := Real.sqrt area

theorem distance_C_to_D 
  (perimeter_smaller : ℝ) (area_larger : ℝ) (h1 : perimeter_smaller = 8) (h2 : area_larger = 36) :
  let s_smaller := side_length_smaller_square perimeter_smaller
  let s_larger := side_length_larger_square area_larger 
  let leg1 := s_larger 
  let leg2 := s_larger - 2 * s_smaller 
  Real.sqrt (leg1 ^ 2 + leg2 ^ 2) = 2 * Real.sqrt 10 :=
by
  sorry

end distance_C_to_D_l139_139386


namespace ff_eq_f_sol_l139_139419

theorem ff_eq_f_sol (x : ℝ) : 
  let f := λ x : ℝ, x^3 - 3 * x^2 in
  f (f x) = f x → x = 0 ∨ x = 3 := 
by
  intro h
  have : f(f(x)) = f(x),
  sorry

end ff_eq_f_sol_l139_139419


namespace min_value_of_y_l139_139620

theorem min_value_of_y (x : ℝ) : ∃ x0 : ℝ, (∀ x : ℝ, 4 * x^2 + 8 * x + 16 ≥ 12) ∧ (4 * x0^2 + 8 * x0 + 16 = 12) :=
sorry

end min_value_of_y_l139_139620


namespace largest_possible_value_of_N_l139_139010

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l139_139010


namespace num_triangles_with_perimeter_20_l139_139642

theorem num_triangles_with_perimeter_20 : 
  ∃ (triangles : List (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → a + b + c = 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
    triangles.length = 8 :=
sorry

end num_triangles_with_perimeter_20_l139_139642


namespace angle_ratio_l139_139382

-- Definitions as per the conditions
def bisects (x y z : ℝ) : Prop := x = y / 2
def trisects (x y z : ℝ) : Prop := y = x / 3

theorem angle_ratio (ABC PBQ BM x : ℝ) (h1 : bisects PBQ ABC PQ)
                                    (h2 : trisects PBQ BM M) :
  PBQ = 2 * x →
  PBQ = ABC / 2 →
  MBQ = x →
  ABQ = 4 * x →
  MBQ / ABQ = 1 / 4 :=
by
  intros
  sorry

end angle_ratio_l139_139382


namespace find_coordinates_of_P_l139_139332

-- Define the problem conditions
def P (m : ℤ) := (2 * m + 4, m - 1)
def A := (2, -4)
def line_l (y : ℤ) := y = -4
def P_on_line_l (m : ℤ) := line_l (m - 1)

theorem find_coordinates_of_P (m : ℤ) (h : P_on_line_l m) : P m = (-2, -4) := 
  by sorry

end find_coordinates_of_P_l139_139332


namespace interest_earned_is_91_dollars_l139_139822

-- Define the initial conditions
def P : ℝ := 2000
def r : ℝ := 0.015
def n : ℕ := 3

-- Define the compounded amount function
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Prove the interest earned after 3 years is 91 dollars
theorem interest_earned_is_91_dollars : 
  (compound_interest P r n) - P = 91 :=
by
  sorry

end interest_earned_is_91_dollars_l139_139822


namespace Isabelle_sec_x_sum_one_l139_139404

theorem Isabelle_sec_x_sum_one (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) 
    (isabelle : (cos x)⁻¹ = 1) : 
  (cos x)⁻¹ = 1 :=
by
  sorry

end Isabelle_sec_x_sum_one_l139_139404


namespace g_five_eq_248_l139_139053

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l139_139053


namespace find_r_l139_139346

-- Define the basic conditions based on the given problem.
def pr (r : ℕ) := 360 / 6
def p := pr 4 / 4
def cr (c r : ℕ) := 6 * c * r

-- Prove that r = 4 given the conditions.
theorem find_r (r : ℕ) : r = 4 :=
by
  sorry

end find_r_l139_139346


namespace exists_finite_field_p_squared_l139_139145

-- Define the predicate that checks if a number p is of the form 12k + {2,3,5,7,8,11}
def special_prime (p : ℕ) : Prop :=
  ∃ k : ℕ, p = 12 * k + 2 ∨ p = 12 * k + 3 ∨ p = 12 * k + 5 ∨ p = 12 * k + 7 ∨ p = 12 * k + 8 ∨ p = 12 * k + 11

-- State the theorem that for such a prime p, there exists a finite field with p^2 elements
theorem exists_finite_field_p_squared (p : ℕ) [hp : Fact (Nat.Prime p)] (h : special_prime p) : 
  ∃ (F : Type) [field F], fintype.card F = p^2 := sorry

end exists_finite_field_p_squared_l139_139145


namespace problem_proof_l139_139549

variables {n : ℕ} (x y : Fin n → ℝ) (x̄ ȳ : ℝ)

def sum_sq_diff (m : ℕ) (f : Fin m → ℝ) (mean : ℝ) :=
  ∑ i in Finset.range m, (f i - mean) ^ 2

/- Given conditions -/
def condition1 : Prop := sum_sq_diff 20 x x̄ = 80
def condition2 : Prop := sum_sq_diff 20 y ȳ = 9000
def condition3 : Prop := 
  ∑ i in Finset.range 20, (x i - x̄) * (y i - ȳ) = 800

/- Correlation coefficient definition -/
def correlation_coefficient : ℝ := 
  ∑ i in Finset.range 20, (x i - x̄) * (y i - ȳ) / 
  Real.sqrt (sum_sq_diff 20 x x̄ * sum_sq_diff 20 y ȳ))

noncomputable def r := correlation_coefficient x y x̄ ȳ

/- Rabbit catching problem -/
def prob_A := 3 / 10
def prob_B1 := 1 / 3
def prob_B2 := 1 / 4
def prob_B3 := 1 / 6

def expected_rabbits_A := 3 * prob_A
def expected_rabbits_B := prob_B1 + prob_B2 + prob_B3

def profit_per_rabbit := 40
def cost_per_person := 20
def cost_per_family := 60  -- since each family has 3 members

def expected_profit_A := profit_per_rabbit * expected_rabbits_A - cost_per_family
def expected_profit_B := profit_per_rabbit * expected_rabbits_B - cost_per_family

theorem problem_proof 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) : 
  r ≈ 0.94 ∧ 
  expected_profit_B < expected_profit_A := by 
  sorry

end problem_proof_l139_139549


namespace remainder_theorem_example_l139_139961

def polynomial (x : ℝ) : ℝ := x^15 + 3

theorem remainder_theorem_example :
  polynomial (-2) = -32765 :=
by
  -- Substitute x = -2 in the polynomial and show the remainder is -32765
  sorry

end remainder_theorem_example_l139_139961


namespace C_share_correct_l139_139193

noncomputable def C_share_of_profit (B_investment : ℝ) (total_profit : ℝ) : ℝ :=
  let A_investment := 3 * B_investment
  let C_investment := (3 / 2) * A_investment
  let total_investment := A_investment + B_investment + C_investment
  let C_share := (C_investment / total_investment) * total_profit
  C_share

theorem C_share_correct :
  C_share_of_profit(final_investment, 55000) ≈ 29117.65 := 
by 
  sorry

end C_share_correct_l139_139193


namespace arithmetic_geometric_mean_inequality_l139_139815

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end arithmetic_geometric_mean_inequality_l139_139815


namespace bisector_angle_DAB_l139_139595

open EuclideanGeometry

variables (A B C D E : Point)
variable (l : Line)

-- Conditions
def conditions : Prop :=
  let parallelogram := (A ≠ B ∧ A ≠ D ∧ A ≠ C ∧ B ≠ D ∧ B ≠ C ∧ D ≠ C) ∧
    parallelogram A B C D
  let circle := lies_on_circle B C E D
  let A_on_l := A ∈ l
  let intersects_F := l.intersects_interior_segment D C F
  let intersects_G := l.intersects_line B C G
  let equal_segments := dist E F = dist E G ∧ dist E F = dist E C
  parallelogram ∧ circle ∧ A_on_l ∧ intersects_F ∧ intersects_G ∧ equal_segments

-- Conclusion
theorem bisector_angle_DAB : conditions A B C D E l → is_angle_bisector l (∠ DAB) :=
sorry

end bisector_angle_DAB_l139_139595


namespace ellipse_equation_l139_139288

theorem ellipse_equation (a b : ℝ) (A B : ℝ × ℝ) (x1 y1 x2 y2 : ℝ)
  (h1 : a > b > 0)
  (h2 : A = (3, 0))
  (h3 : x1 = 2 ∧ y1 = -1 ∧ x2 = 0 ∧ y2 = 1)
  (h4 : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = -1)
  (h5 : (y1 - y2) / (x1 - x2) = 1/2)
  (h6 : sqrt (a^2 - b^2) = 3):
  E = (λ x y, x^2 / 18 + y^2 / 9 = 1) :=
by
  sorry

end ellipse_equation_l139_139288


namespace whiteTigerNumberCount_l139_139497

def isWhiteTiger (n : ℕ) : Prop :=
  n % 6 = 0 ∧ (n.digits 10).sum = 6

theorem whiteTigerNumberCount : 
  (Finset.range 2023).filter isWhiteTiger).card = 30 := 
by 
  sorry

end whiteTigerNumberCount_l139_139497


namespace log_sum_sin_eq_neg_3_l139_139151

open Real

theorem log_sum_sin_eq_neg_3 :
  log 2 (sin (π / 12)) + log 2 (sin (π / 6)) + log 2 (sin (5 * π / 12)) = -3 := 
  sorry

end log_sum_sin_eq_neg_3_l139_139151


namespace right_triangle_sides_l139_139964

theorem right_triangle_sides (r R : ℝ) (a b c : ℝ) 
    (r_eq : r = 8)
    (R_eq : R = 41)
    (right_angle : a^2 + b^2 = c^2)
    (inradius : 2*r = a + b - c)
    (circumradius : 2*R = c) :
    (a = 18 ∧ b = 80 ∧ c = 82) ∨ (a = 80 ∧ b = 18 ∧ c = 82) :=
by
  sorry

end right_triangle_sides_l139_139964


namespace butterfly_cocoon_time_l139_139399

theorem butterfly_cocoon_time :
  ∀ (L C : ℕ), L = 3 * C ∧ L + C = 120 → C = 30 := by
  intros L C h
  cases' h with h1 h2
  have h3 : 3 * C + C = 120 := by rw [h1, add_comm] at h2
  have h4 : 4 * C = 120 := by rw add_mul 3 1 C at h3
  rw mul_comm at h4
  exact Nat.eq_of_mul_eq_mul_left (ne_of_gt (show 0 < 4 from by norm_num)) h4

end butterfly_cocoon_time_l139_139399


namespace arithmetic_sequence_general_formula_sum_of_inverse_sn_l139_139665

noncomputable theory
open_locale big_operators

-- Define the arithmetic sequence and sum conditions
def Sn (n : ℕ) : ℕ := (a 1 + a n) * n / 2

-- First problem: find the general formula for a_n given conditions
theorem arithmetic_sequence_general_formula (S5_eq_35 : Sn 5 = 35)
  (geo_cond : ∀ {a : ℕ}, a 1 * a 13= (a 4)^2 
  (non_zero_common_diff : ∀ {d : ℕ}, d ≠ 0) :
  ∀ n : ℕ, a n = 2*n + 3 := 
sorry

-- Second problem: find the sum of the first n terms of the sequence {1/S_n}
theorem sum_of_inverse_sn (n : ℕ) (S5_eq_35 : Sn 5 = 35)
  (geo_cond : ∀ {a : ℕ}, a 1 * a 13 = (a 4)^2)
  (non_zero_common_diff : ∀ {d : ℕ}, d ≠ 0) :
  ∑ i in finset.range (n + 1), 1 / Sn i = 3/4 - (2*n + 3) / (4 * (n + 1) * (n + 2)) := 
sorry

end arithmetic_sequence_general_formula_sum_of_inverse_sn_l139_139665


namespace cell_D4_value_l139_139242

theorem cell_D4_value :
  ∃ (A B C D E : Fin 5 → ℕ), 
    (∀ i, A i ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
         B i ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
         C i ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
         D i ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
         E i ∈ ({1, 2, 3, 4, 5} : Set ℕ)) ∧ 
    (∀ i, ∑ j, A i = 15 ∧ ∑ j, B i = 15 ∧ ∑ j, C i = 15 ∧ ∑ j, D i = 15 ∧ ∑ j, E i = 15) ∧
    (∀ j, ∑ i, A i j + B i j + C i j + D i j + E i j = 15) ∧
    (A 3 + B 3 + C 3 + D 3 + E 3 = 9) ∧
    (C 0 + C 2 + C 4 = 7) ∧
    (A 1 + C 1 + E 1 = 8) ∧
    (∑ j, B j < ∑ j, D j) ∧
    D 3 = 5 :=
begin 
  sorry 
end

end cell_D4_value_l139_139242


namespace smallest_number_bob_l139_139574

-- Define the conditions given in the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_factors (x : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ x }

-- The problem statement
theorem smallest_number_bob (b : ℕ) (h1 : prime_factors 30 = prime_factors b) : b = 30 :=
by
  sorry

end smallest_number_bob_l139_139574


namespace regular_octagon_product_l139_139199

noncomputable def product_of_regular_octagon (Q : ℕ → ℂ) (is_regular_octagon : ∀ k, Q k = (3 + exp (2 * π * I * (k : ℂ) / 8))) : ℂ := 
  (Q 1) * (Q 2) * (Q 3) * (Q 4) * (Q 5) * (Q 6) * (Q 7) * (Q 8)

theorem regular_octagon_product (Q : ℕ → ℂ) 
  (is_regular_octagon : ∀ k, Q k = (3 + exp (2 * π * I * (k : ℂ) / 8))) :
  product_of_regular_octagon Q is_regular_octagon = 6559 := 
sorry

end regular_octagon_product_l139_139199


namespace line_through_orthocenter_line_through_C_iff_ratio_l139_139804

open EuclideanGeometry

variables {A B C A_1 B_1 : Point}
variables (ABC : Triangle A B C)
variables (l : Line)
variables (H : Point)
variables (H_a H_b : Line)

-- Conditions
def on_side_BC (A_1 : Point) : Prop := A_1 ∈ (line_segment B C)
def on_side_AC (B_1 : Point) : Prop := B_1 ∈ (line_segment A C)
def passes_through_circles_common_points (l : Line) : Prop :=
  ∃ (circleAA1 circleBB1 : Circle), 
    circleAA1 = circle_diameter A A_1 ∧ 
    circleBB1 = circle_diameter B B_1 ∧ 
    ∀ (P : Point), P ∈ l ↔ P ∈ circleAA1 ∧ P ∈ circleBB1

def orthocenter (ABC : Triangle A B C) : Point := sorry

-- Part a) Theorem
theorem line_through_orthocenter 
(hBC : on_side_BC A_1) 
(hAC : on_side_AC B_1)
(h_l_passes : passes_through_circles_common_points l)
(hH : orthocenter ABC = H) :
  l ∈ H := sorry

-- Part b) Theorem
theorem line_through_C_iff_ratio 
(hBC : on_side_BC A_1) 
(hAC : on_side_AC B_1)
(h_l_passes : passes_through_circles_common_points l) :
  (l ∈ C) ↔ (AB_1 / AC = BA_1 / BC) := sorry

end line_through_orthocenter_line_through_C_iff_ratio_l139_139804


namespace problem_part1_problem_part2_problem_part3_l139_139682

-- Given the equation (x-1)(x^2-3x+m) = 0, where m is a real number
-- Part (1)
theorem problem_part1 (m : ℝ) (h_m : m = 4) : 
  (x : ℝ) ((x - 1) * (x^2 - 3*x + m) = 0) ↔ x = 1 :=
by sorry

-- Part (2)
theorem problem_part2 (m : ℝ) : 
  (∃ x1 x2 x3 : ℝ, (x1 - 1) * (x2*x3 - 3x*x + m) = 0 ∧ (x1 = x2 ∨ x2 = x3 ∨ x1 = x3)) ↔ (m = 2 ∨ m = 9/4) :=
by sorry

-- Part (3)
theorem problem_part3 (m : ℝ) : 
  (∃ x1 x2 x3 : ℝ, (x1 - 1) * (x2*x3 - 3x*x + m) = 0 ∧ (x1 + x2 + x3 ≥ x1 + x2 + x3) ∧ (|x2 - x3| < 1 ∧ m > 0 ∧ 9 - 4 * m ≥ 0)) ↔ (2 < m ∧ m ≤ 9/4) :=
by sorry

end problem_part1_problem_part2_problem_part3_l139_139682


namespace correct_sampling_methods_l139_139916

/-- 
Given:
1. A group of 500 senior year students with the following blood type distribution: 200 with blood type O,
125 with blood type A, 125 with blood type B, and 50 with blood type AB.
2. A task to select a sample of 20 students to study the relationship between blood type and color blindness.
3. A high school soccer team consisting of 11 players, and the need to draw 2 players to investigate their study load.
4. Sampling methods: I. Random sampling, II. Systematic sampling, III. Stratified sampling.

Prove:
The correct sampling methods are: Stratified sampling (III) for the blood type-color blindness study and
Random sampling (I) for the soccer team study.
-/ 

theorem correct_sampling_methods (students : Finset ℕ) (blood_type_O blood_type_A blood_type_B blood_type_AB : ℕ)
  (sample_size_students soccer_team_size draw_size_soccer_team : ℕ)
  (sampling_methods : Finset ℕ) : 
  (students.card = 500) →
  (blood_type_O = 200) →
  (blood_type_A = 125) →
  (blood_type_B = 125) →
  (blood_type_AB = 50) →
  (sample_size_students = 20) →
  (soccer_team_size = 11) →
  (draw_size_soccer_team = 2) →
  (sampling_methods = {1, 2, 3}) →
  (s = (3, 1)) :=
by
  sorry

end correct_sampling_methods_l139_139916


namespace find_g5_l139_139069

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l139_139069


namespace percentage_of_additional_money_is_10_l139_139882

-- Define the conditions
def months := 11
def payment_per_month := 15
def total_borrowed := 150

-- Define the function to calculate the total amount paid
def total_paid (months payment_per_month : ℕ) : ℕ :=
  months * payment_per_month

-- Define the function to calculate the additional amount paid
def additional_paid (total_paid total_borrowed : ℕ) : ℕ :=
  total_paid - total_borrowed

-- Define the function to calculate the percentage of the additional amount
def percentage_additional (additional_paid total_borrowed : ℕ) : ℕ :=
  (additional_paid * 100) / total_borrowed

-- State the theorem to prove the percentage of the additional money is 10%
theorem percentage_of_additional_money_is_10 :
  percentage_additional (additional_paid (total_paid months payment_per_month) total_borrowed) total_borrowed = 10 :=
by
  sorry

end percentage_of_additional_money_is_10_l139_139882


namespace expression_equivalence_l139_139524

theorem expression_equivalence:
  let a := 10006 - 8008
  let b := 10000 - 8002
  a = b :=
by {
  sorry
}

end expression_equivalence_l139_139524


namespace problem_part1_problem_part2_l139_139289

noncomputable def circle : set (ℝ × ℝ) := { p | p.1 ^ 2 + p.2 ^ 2 = 4 }

noncomputable def line (l : ℝ → ℝ → Prop) : set (ℝ × ℝ) :=
  { p | ∃ x y, l x y ∧ p = (x, y) }

noncomputable def line_l1 (x y : ℝ) : Prop :=
  sqrt 3 * x + y = 2 * sqrt 3

theorem problem_part1 :
  ∃ A B, A ∈ (circle ∩ line line_l1) ∧ B ∈ (circle ∩ line line_l1) ∧ (abs (A.1 - B.1) ^ 2 + abs (A.2 - B.2) ^ 2) = 4 := 
sorry

theorem problem_part2 :
  ∀ (P : ℝ × ℝ), P ∈ circle ∧ P.1 ≠ 1 ∧ P.1 ≠ -1 →
  ∃ m n, (let P1 := (-P.1, -P.2),
              P2 := (P.1, -P.2),
              A := (1, sqrt 3),
              m := (sqrt 3 * P.1 - P.2) / (1 + P.1),
              n := (-sqrt 3 * P.1 - P.2) / (1 - P.1) in 
          m * n = 4) :=
sorry

end problem_part1_problem_part2_l139_139289


namespace convex_hexagon_largest_angle_l139_139480

theorem convex_hexagon_largest_angle 
  (x : ℝ)                                 -- Denote the measure of the third smallest angle as x.
  (angles : Fin 6 → ℝ)                     -- Define the angles as a function from Fin 6 to ℝ.
  (h1 : ∀ i : Fin 6, angles i = x + (i : ℝ) - 3)  -- The six angles in increasing order.
  (h2 : 0 < x - 3 ∧ x - 3 < 180)           -- Convex condition: each angle is between 0 and 180.
  (h3 : angles ⟨0⟩ + angles ⟨1⟩ + angles ⟨2⟩ + angles ⟨3⟩ + angles ⟨4⟩ + angles ⟨5⟩ = 720) -- Sum of interior angles of a hexagon.
  : (∃ a, a = angles ⟨5⟩ ∧ a = 122.5) :=   -- Prove the largest angle in this arrangement is 122.5.
sorry

end convex_hexagon_largest_angle_l139_139480


namespace wrong_observation_value_l139_139837

theorem wrong_observation_value
  (mean_initial : ℝ)
  (n : ℕ)
  (sum_initial : ℝ)
  (mean_corrected : ℝ)
  (sum_corrected : ℝ)
  (wrong_value : ℝ)
  (correct_value : ℝ)
  (correction : ℝ)
  (initial_observation : ℝ) :

  mean_initial = 40 →
  n = 50 →
  sum_initial = n * mean_initial →
  mean_corrected = 40.66 →
  sum_corrected = n * mean_corrected →
  initial_observation = 45 →
  correction = sum_corrected - sum_initial →
  correct_value = initial_observation - correction →
  
  correct_value = 12 :=
begin
  sorry
end

end wrong_observation_value_l139_139837


namespace find_x_l139_139754

def integers_x_y (x y : ℤ) : Prop :=
  x > y ∧ y > 0 ∧ x + y + x * y = 110

theorem find_x (x y : ℤ) (h : integers_x_y x y) : x = 36 := sorry

end find_x_l139_139754


namespace dice_sum_lt_10_probability_l139_139507

open Probability

-- Define the event space for rolling two six-sided dice
noncomputable def event_space := {a : ℕ × ℕ | 1 ≤ a.1 ∧ a.1 ≤ 6 ∧ 1 ≤ a.2 ∧ a.2 ≤ 6}

-- Define the event where the sum of numbers showing on two dice is less than 10
def sum_lt_10 (a : ℕ × ℕ) : Prop := (a.1 + a.2 < 10)

-- Calculate the probability that the sum of two fair six-sided dice is less than 10
theorem dice_sum_lt_10_probability : ∃ p : ℚ, p = 5/6 ∧
  (Pr {a ∈ event_space | sum_lt_10 a} = p) :=
begin
  sorry
end

end dice_sum_lt_10_probability_l139_139507


namespace intersection_distance_is_1_min_distance_is_correct_l139_139820

noncomputable def intersection_distance : ℝ :=
  let A := (1 : ℝ, 0 : ℝ)
  let B := (1 / 2, - sqrt 3 / 2)
  sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem intersection_distance_is_1 : intersection_distance = 1 := sorry

noncomputable def min_distance_to_l : ℝ :=
  let P (θ: ℝ) := (1 / 2 * cos θ, sqrt 3 / 2 * sin θ)
  let distance (P: ℝ × ℝ) := 
    abs (sqrt 3 / 2 * cos P.1 - sqrt 3 / 2 * sin P.2 - sqrt 3) / sqrt (3 + 1)
  let min_dist := min (λ θ => distance (P θ))
  min_dist

theorem min_distance_is_correct :
  min_distance_to_l = sqrt 6 / 4 * (sqrt 2 - 1) := sorry

end intersection_distance_is_1_min_distance_is_correct_l139_139820


namespace log_sum_geometric_sequence_l139_139736

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def sequence_property (a : ℕ → ℝ) (n m : ℕ) : ℝ :=
a n * a m

theorem log_sum_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (seq : geometric_sequence a) (h : sequence_property a 4 7 = 9) :
  ∑ i in finset.range 10, real.log 3 (a (i + 1)) = 10 :=
sorry

end log_sum_geometric_sequence_l139_139736


namespace total_triangles_in_grid_l139_139210

-- Conditions
def bottom_row_triangles : Nat := 3
def next_row_triangles : Nat := 2
def top_row_triangles : Nat := 1
def additional_triangle : Nat := 1

def small_triangles := bottom_row_triangles + next_row_triangles + top_row_triangles + additional_triangle

-- Combining the triangles into larger triangles
def larger_triangles := 1 -- Formed by combining 4 small triangles
def largest_triangle := 1 -- Formed by combining all 7 small triangles

-- Math proof problem
theorem total_triangles_in_grid : small_triangles + larger_triangles + largest_triangle = 9 :=
by
  sorry

end total_triangles_in_grid_l139_139210


namespace function_range_l139_139843

theorem function_range (y : ℝ) (x : ℝ) (h : x ≥ 0) :
  y = (sqrt x - 1) / (sqrt x + 1) → y ∈ Ico (-1) 1 :=
sorry

end function_range_l139_139843


namespace minimum_hat_changes_l139_139372

-- Conditions
variable (D : Type) [Fintype D] [DecidableEq D] (dw : D → Prop)
variable (red_hats : {d : D // dw d})
variable (blue_hats : {d : D // ¬ dw d})
variable (truth : ∀ (d : D), dw d → true) 
variable (lie : ∀ (d : D), ¬ dw d → (true ∨ true)) -- Truth statement irrelevant for blue hats
variable (change_hat : D → Prop)
variable (claim : (d1 d2 : D), d1 ≠ d2 → claim d1 d2)

-- Question: Minimum number of times the dwarfs changed the colors of their hats
theorem minimum_hat_changes (cs : ∀ (d1 d2 : D), d1 ≠ d2 → claim d1 d2 = (¬ dw d1 ∧ ¬ dw d2)) : 
  ∑ (d : D), change_hat d = 2009 := 
sorry

end minimum_hat_changes_l139_139372


namespace sum_b_inv_eq_l139_139667

noncomputable def a (n : ℕ) : ℝ := 2 ^ (n - 2)

noncomputable def b (n : ℕ) : ℝ :=
  (real.log2 (a (2 * n + 1))) * (real.log2 (a (2 * n + 3)))

noncomputable def b_inv (n : ℕ) : ℝ := 1 / b n

noncomputable def sum_b_inv (n : ℕ) : ℝ :=
  (∑ i in finset.range n, b_inv i)

theorem sum_b_inv_eq (n : ℕ) : sum_b_inv n = n / (2 * n + 1) :=
by
  sorry

end sum_b_inv_eq_l139_139667


namespace f_identically_zero_l139_139299

open Real

-- Define the function f and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom func_eqn (a b : ℝ) : f (a * b) = a * f b + b * f a 
axiom func_bounded (x : ℝ) : |f x| ≤ 1

-- Goal: Prove that f is identically zero
theorem f_identically_zero : ∀ x : ℝ, f x = 0 := 
by
  sorry

end f_identically_zero_l139_139299


namespace largest_possible_N_l139_139005

theorem largest_possible_N (N : ℕ) :
  let divisors := Nat.divisors N
  in (1 ∈ divisors) ∧ (N ∈ divisors) ∧ (divisors.length ≥ 3) ∧ (divisors[divisors.length - 3] = 21 * divisors[1]) → N = 441 := 
by
  sorry

end largest_possible_N_l139_139005


namespace time_correct_l139_139395

theorem time_correct {t : ℝ} (h : 0 < t ∧ t < 60) :
  |6 * (t + 5) - (90 + 0.5 * (t - 4))| = 180 → t = 43 := by
  sorry

end time_correct_l139_139395


namespace C1_eq_cartesian_C2_eq_cartesian_min_PQ_l139_139741

noncomputable def C1_parametric (α : ℝ) : ℝ × ℝ := (sqrt 3 * Real.cos α, Real.sin α)

def C2_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + π/4) = 2 * sqrt 2

def C1_cartesian (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

def C2_cartesian (x y : ℝ) : Prop := x + y - 4 = 0

theorem C1_eq_cartesian :
    ∀ α : ℝ, C1_cartesian (sqrt 3 * Real.cos α) (Real.sin α) :=
  by
    intro α
    sorry

theorem C2_eq_cartesian :
    ∀ ρ θ : ℝ, C2_polar ρ θ → C2_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ) :=
  by
    intro ρ θ h
    sorry

theorem min_PQ :
    ∃ P Q : ℝ × ℝ, 
      (C1_cartesian P.1 P.2) ∧ 
      (C2_cartesian Q.1 Q.2) ∧ 
      |P.1 - Q.1| + |P.2 - Q.2| = sqrt 2 ∧ 
      P = (3/2, 1/2) :=
  by 
    sorry

end C1_eq_cartesian_C2_eq_cartesian_min_PQ_l139_139741


namespace least_value_is_one_l139_139127

noncomputable def least_possible_value (x y : ℝ) : ℝ := (x^2 * y - 1)^2 + (x^2 + y)^2

theorem least_value_is_one : ∀ x y : ℝ, (least_possible_value x y) ≥ 1 :=
by
  sorry

end least_value_is_one_l139_139127


namespace circle_area_ratio_l139_139871

theorem circle_area_ratio {AB CD BC DE : ℝ} (h1 : tangent cir1 AB) (h2 : tangent cir1 CD) (h3 : tangent cir1 BC)
  (h4 : tangent cir2 AB) (h5 : tangent cir2 CD) (h6 : tangent cir2 DE) :
  ratio_area cir2 cir1 = 4 :=
sorry

end circle_area_ratio_l139_139871


namespace net_population_change_l139_139498

theorem net_population_change (P : ℝ) : 
  let P1 := P * (6/5)
  let P2 := P1 * (7/10)
  let P3 := P2 * (6/5)
  let P4 := P3 * (7/10)
  (P4 / P - 1) * 100 = -29 := 
by
  sorry

end net_population_change_l139_139498


namespace find_percentage_l139_139154

def percent_of_num (p : ℝ) (n : ℝ) : ℝ := p * n

def given_number := 5600

def given_condition1 : percent_of_num 0.15 (percent_of_num 0.30 (percent_of_num P given_number)) = 126

theorem find_percentage (P : ℝ) : P = 0.5 :=
by
  -- Proof skipped
  sorry

end find_percentage_l139_139154


namespace binom_series_sum_l139_139968

theorem binom_series_sum :
  (finset.range 26).sum (λ k, (-1)^k * nat.choose 50 (2 * k)) = 0 :=
by
  sorry

end binom_series_sum_l139_139968


namespace max_value_dot_product_l139_139295

variable {V : Type*} [InnerProductSpace ℝ V]

-- Definitions for conditions: unit vectors and specific angle between vectors
def is_unit_vector (v : V) : Prop :=
  ∥v∥ = 1

def angle_eq (a b : V) (theta : ℝ) : Prop :=
  real_inner a b = ∥a∥ * ∥b∥ * real.cos theta

-- The main theorem statement
theorem max_value_dot_product (a b c : V) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b)
  (hc : is_unit_vector c) 
  (hab_angle : angle_eq a b (real.pi / 3)) :
  (a + b + c) ⬝ c ≤ real.sqrt 3 + 1 :=
  sorry

end max_value_dot_product_l139_139295


namespace main_proof_l139_139658

def ellipse_equation (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def point_P := (1 : ℝ, -√3 / 2)

def eccentricity := √3 / 2

def vertices_condition (a b : ℝ) : Prop := a > b ∧ b > 0

noncomputable def main (a b : ℝ) (F A B : ℝ × ℝ) (line_l : ℝ → ℝ) : Prop :=
    (vertices_condition a b ∧
    ellipse_equation (fst point_P) (snd point_P) a b ∧
    (c : ℝ^2) ∈ ellipse_equation c.fst c.snd a b :=
        a = 2 ∧ b = 1 ∧
        line_l (fst F) = snd F ∧
        (∀ (S1 S2 : ℝ), (A ≠ B) ∧
        ∃ l, (l = line_l ∧
        (C D : ℝ × ℝ) ∈ ellipse_equation (fst C) (snd C) 2 1 =
            S1 - S2 ∈ [-√3, √3]) ∨
        (S1 - S2 = 0)))

theorem main_proof : ∃ a b F A B line_l, main a b F A B line_l :=
begin
  sorry,
end

end main_proof_l139_139658


namespace range_of_a_l139_139695

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, a * x ^ 2 + 2 * a * x + 1 ≤ 0) →
  0 ≤ a ∧ a < 1 :=
by
  -- sorry to skip the proof
  sorry

end range_of_a_l139_139695


namespace length_of_segment_AB_l139_139599

theorem length_of_segment_AB :
  ∀ A B : ℝ × ℝ,
  (∃ x y : ℝ, y^2 = 8 * x ∧ y = (y - 0) / (4 - 2) * (x - 2))
  ∧ (A.1 + B.1) / 2 = 4
  → dist A B = 12 := 
by
  sorry

end length_of_segment_AB_l139_139599


namespace jefferson_high_school_ninth_graders_l139_139840

theorem jefferson_high_school_ninth_graders (total_students science_students arts_students students_taking_both : ℕ):
  total_students = 120 →
  science_students = 85 →
  arts_students = 65 →
  students_taking_both = 150 - 120 →
  science_students - students_taking_both = 55 :=
by
  sorry

end jefferson_high_school_ninth_graders_l139_139840


namespace shortest_distance_to_parabola_l139_139265

noncomputable def shortest_distance (p : ℝ × ℝ) (parabola : ℝ → ℝ) : ℝ :=
  let y_coord := 4
  let x_coord := y_coord^2 / 4
  let dist := (p.1 - x_coord)^2 + (p.2 - y_coord)^2
  dist.sqrt

theorem shortest_distance_to_parabola : shortest_distance (4, 8) (λ y, y^2 / 4) = 4 :=
sorry

end shortest_distance_to_parabola_l139_139265


namespace small_bottles_initial_count_l139_139187

theorem small_bottles_initial_count (big_bottles init_small remaining_bottles : ℕ)
  (H1 : big_bottles = 14000)
  (H_remaining_small : remaining_bottles = 15580 - 0.231 * big_bottles)
  (H_sold_small : 0.20 * init_small = 0.80 * init_small - H_remaining_small)
  : init_small = 6000 := by
sorrry

end small_bottles_initial_count_l139_139187


namespace problem1_problem2_l139_139152

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |2 * x + a| - |2 * x + 3|
def g (x : ℝ) : ℝ := |x - 1| - 3

-- Statement 1: Solve the inequality |g(x)| < 2
theorem problem1 (x : ℝ) : |g(x)| < 2 ↔ -4 < x ∧ x < 6 :=
  sorry

-- Statement 2: Find the range of values for a such that ∀ x1 ∈ ℝ, ∃ x2 ∈ ℝ, f(x1) = g(x2)
theorem problem2 (a : ℝ) : (∀ x1 : ℝ, ∃ x2 : ℝ, f(x1, a) = g(x2)) ↔ (0 ≤ a ∧ a ≤ 6) :=
  sorry

end problem1_problem2_l139_139152


namespace sum_of_solutions_l139_139621

theorem sum_of_solutions (a b c : ℝ) (h_equation : 63 - 21*x - x^2 = 0) :
    ∀ (x : ℝ),
      let sum_of_solutions := -b / a
      in sum_of_solutions = -21 :=
by
  have h_standard_form : -x^2 - 21*x + 63 = 0, by sorry
  have h_multiplied : x^2 + 21*x - 63 = 0, by sorry
  have h_a : a = 1, by sorry
  have h_b : b = 21, by sorry
  have h_c : c = -63, by sorry
  exact sorry

end sum_of_solutions_l139_139621


namespace evaluate_sum_l139_139238

-- Definition of the given problem conditions
def Phi (θ : ℝ) : ℝ := sin θ ^ 2
def Psi (θ : ℝ) : ℝ := cos θ
def sequence : List ℝ := [1, 2, 45, 47]

noncomputable def sum_expr : ℝ :=
  ∑ i in List.finRange 4, (-1) ^ (i + 1) * (Phi (sequence.nthLe i (by simp)) / Psi (sequence.nthLe i (by simp)))

theorem evaluate_sum : sum_expr = 95 := 
  sorry

end evaluate_sum_l139_139238


namespace slope_tangent_at_point_l139_139848

-- Define the function y = x * exp(x - 1)
def f (x : ℝ) : ℝ := x * Real.exp (x - 1)

-- State the theorem about the derivative at x = 1
theorem slope_tangent_at_point : HasDerivAt f 2 1 := sorry

end slope_tangent_at_point_l139_139848


namespace Ms_Smiths_Class_Books_Distribution_l139_139732

theorem Ms_Smiths_Class_Books_Distribution :
  ∃ (x : ℕ), (20 * 2 * x + 15 * x + 5 * x = 840) ∧ (20 * 2 * x = 560) ∧ (15 * x = 210) ∧ (5 * x = 70) :=
by
  let x := 14
  have h1 : 20 * 2 * x + 15 * x + 5 * x = 840 := by sorry
  have h2 : 20 * 2 * x = 560 := by sorry
  have h3 : 15 * x = 210 := by sorry
  have h4 : 5 * x = 70 := by sorry
  exact ⟨x, h1, h2, h3, h4⟩

end Ms_Smiths_Class_Books_Distribution_l139_139732


namespace segments_intersect_on_bisector_l139_139753

variables (A B C A₁ B₁ B₂ C₂ X : Type)
variables [Triangle A B C] -- Define a triangle ABC
variables (M_A1 : is_midpoint A₁ B C) -- A₁ is midpoint of BC
variables (M_B1 : is_midpoint B₁ A C) -- B₁ is midpoint of AC
variables (T_B2 : is_tangency_point B₂ A C incircle) -- B₂ is tangency point on AC
variables (T_C2 : is_tangency_point C₂ A B incircle) -- C₂ is tangency point on AB

theorem segments_intersect_on_bisector 
  (HABgtBC : length A B > length B C) -- Condition: AB > BC
  : is_angle_bisector X B := -- Goal: X lies on angle bisector of B
sorry

end segments_intersect_on_bisector_l139_139753


namespace one_fifty_percent_of_eighty_l139_139521

theorem one_fifty_percent_of_eighty : (150 / 100) * 80 = 120 :=
  by sorry

end one_fifty_percent_of_eighty_l139_139521


namespace total_money_divided_l139_139156

theorem total_money_divided (A B C : ℝ) (hA : A = 280) (h1 : A = (2 / 3) * (B + C)) (h2 : B = (2 / 3) * (A + C)) :
  A + B + C = 700 := by
  sorry

end total_money_divided_l139_139156


namespace minimum_possible_value_of_sum_l139_139616

noncomputable def minimum_value_of_sum (a b c : ℝ) : ℝ :=
  (if a > 0 ∧ b > 0 ∧ c > 0 then 3 * (1 / (105 : ℝ) ^ (1 / 3 : ℝ)) else 0)

theorem minimum_possible_value_of_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a / (3 * b) + b / (5 * c) + c / (7 * a)) ≥ minimum_value_of_sum a b c :=
begin
  sorry -- Proof is omitted as per instructions
end

end minimum_possible_value_of_sum_l139_139616


namespace most_likely_outcomes_l139_139988
open ProbabilityTheory

noncomputable def probability_boys (n : ℕ) : ℚ := (1 / 2) ^ n

noncomputable def probability_outcome_C : ℚ := (nat.choose 5 2) * (probability_boys 5)

noncomputable def probability_outcome_D : ℚ := 2 * (nat.choose 5 1) * (probability_boys 5)

theorem most_likely_outcomes :
  ((probability_outcome_C = 5 / 16) ∧ (probability_outcome_D = 5 / 16)) ↔ 
  (5 children are equally likely to be boys or girls → gender independence → 
  {(2 boys, 3 girls), (4 of one gender, 1 of the other)} are the most likely outcomes) :=
by
  sorry

end most_likely_outcomes_l139_139988


namespace f_g_of_4_l139_139774

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - x - 4

theorem f_g_of_4 : f (g 4) = 23 * Real.sqrt 10 / 5 := by
  sorry

end f_g_of_4_l139_139774


namespace meaningful_expression_iff_l139_139844

noncomputable theory
open Classical -- For dealing with classical logic (if necessary)

-- Define the conditions as Lean terms
def meaningful_expression (x : ℝ) : Prop := (x + 1 ≥ 0) ∧ (x ≠ 0)

-- Theorem stating the equivalent condition
theorem meaningful_expression_iff {x : ℝ} :
  meaningful_expression x ↔ (x ≥ -1 ∧ x ≠ 0) :=
begin
  sorry, -- Proof is left out
end

end meaningful_expression_iff_l139_139844


namespace sphere_radius_proportional_l139_139853

theorem sphere_radius_proportional
  (k : ℝ)
  (r1 r2 : ℝ)
  (W1 W2 : ℝ)
  (h_weight_area : ∀ (r : ℝ), W1 = k * (4 * π * r^2))
  (h_given1: W2 = 32)
  (h_given2: r2 = 0.3)
  (h_given3: W1 = 8):
  r1 = 0.15 := 
by
  sorry

end sphere_radius_proportional_l139_139853


namespace min_tiles_for_L_shape_l139_139348

theorem min_tiles_for_L_shape : ∀ a : ℕ, (∀ t : ℕ, t ∈ {1, 1, 1} → t = t) → (3 * a) ≥ 4 := sorry

end min_tiles_for_L_shape_l139_139348


namespace expected_value_of_win_is_2_5_l139_139903

noncomputable def expected_value_of_win : ℚ := 
  (1/6) * (6 - 1) + (1/6) * (6 - 2) + (1/6) * (6 - 3) + 
  (1/6) * (6 - 4) + (1/6) * (6 - 5) + (1/6) * (6 - 6)

theorem expected_value_of_win_is_2_5 : expected_value_of_win = 5 / 2 := 
by
  -- Proof steps will go here
  sorry

end expected_value_of_win_is_2_5_l139_139903


namespace parts_rate_relation_l139_139430

theorem parts_rate_relation
  (x : ℝ)
  (total_parts_per_hour : ℝ)
  (master_parts : ℝ)
  (apprentice_parts : ℝ)
  (h_total : total_parts_per_hour = 40)
  (h_master : master_parts = 300)
  (h_apprentice : apprentice_parts = 100)
  (h : total_parts_per_hour = x + (40 - x)) :
  (master_parts / x) = (apprentice_parts / (40 - x)) := 
by
  sorry

end parts_rate_relation_l139_139430


namespace largest_multiple_60_div_15_l139_139459

theorem largest_multiple_60_div_15 :
  (∃ n : ℕ, (∀ d ∈ (nat.digits 10 n), d = 7 ∨ d = 0) ∧ n % 60 = 0 ∧ n / 15 = 518) :=
sorry

end largest_multiple_60_div_15_l139_139459


namespace ratio_vitamins_supplements_percentage_vitamins_supplements_difference_percentage_l139_139014

theorem ratio_vitamins_supplements (vitamins supplements : ℕ) (hvitamins : vitamins = 472) (hsupplements : supplements = 288) :
  let gcdvs := Nat.gcd vitamins supplements;
  let simplest_vitamins := vitamins / gcdvs;
  let simplest_supplements := supplements / gcdvs;
  (simplest_vitamins, simplest_supplements) = (59, 36) :=
by
  sorry

theorem percentage_vitamins_supplements (vitamins supplements : ℕ) (hvitamins : vitamins = 472) (hsupplements : supplements = 288) :
  let total := vitamins + supplements;
  let percentage_vitamins := (vitamins * 100) / total;
  let percentage_supplements := (supplements * 100) / total;
  percentage_vitamins ≈ 62.11 ∧ percentage_supplements ≈ 37.89 :=
by
  sorry

theorem difference_percentage (vitamins supplements : ℕ) (hvitamins : vitamins = 472) (hsupplements : supplements = 288) :
  let total := vitamins + supplements;
  let percentage_vitamins := (vitamins * 100) / total;
  let percentage_supplements := (supplements * 100) / total;
  let difference := percentage_vitamins - percentage_supplements;
  difference ≈ 24.22 :=
by
  sorry

end ratio_vitamins_supplements_percentage_vitamins_supplements_difference_percentage_l139_139014


namespace nine_linked_rings_solution_l139_139087

noncomputable def a : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := if (n+1).even then 2 * a (n + 1) - 1 else 2 * a (n + 1) + 2

theorem nine_linked_rings_solution :
  a 4 = 7 :=
by {
  sorry,
}

end nine_linked_rings_solution_l139_139087


namespace circle_center_l139_139250

theorem circle_center {x y : ℝ} : x^2 - 8x + y^2 + 4y = -4 → ∃ c : ℝ × ℝ, c = (4, -2) :=
by
  intro h
  sorry

end circle_center_l139_139250


namespace six_digit_special_number_count_l139_139606

open Finset

noncomputable def count_special_numbers : ℕ :=
  288

theorem six_digit_special_number_count :
  ∀ {n : ℕ}, n = 288 ↔ 
  ∃ (d : Finset ℕ), d = {1, 2, 3, 4, 5, 6} ∧
  ∀ s ∈ d, ∀ k ∈ d, s ≠ k →
  ∃ l : list ℕ, (l.length = 6) ∧
  (∀ x ∈ l, x ∈ d) ∧
  (∀ i ∈ [0, 5], l[i] ≠ 1 ∧ ∃ j ∈ [1,4], (l[0] ≠ 1 ∧ l[5] ≠ 1)) ∧
  let evens := {2, 4, 6} in
  ∃ adj : (ℕ → ℕ → Prop), (∀ e₁ e₂ ∈ evens, adj e₁ e₂) ∧ 
  ∃ h : 2 ≤ card (filter adj (l.zip_with adj l)),
  n = count_special_numbers
:= sorry

end six_digit_special_number_count_l139_139606


namespace selling_price_is_120_l139_139911

-- Declaring all necessary definitions and conditions
def cost_price : ℝ := 100
def gain_percent : ℝ := 20
def profit : ℝ := (gain_percent / 100) * cost_price
def selling_price : ℝ := cost_price + profit

-- Statement of the theorem
theorem selling_price_is_120 : selling_price = 120 :=
by
  -- Skipping Proof
  sorry

end selling_price_is_120_l139_139911


namespace subtraction_example_l139_139591

theorem subtraction_example : 34.256 - 12.932 - 1.324 = 20.000 := 
by
  sorry

end subtraction_example_l139_139591


namespace find_nickels_l139_139216

def quarters : ℕ := 76
def dimes : ℕ := 85
def pennies : ℕ := 150
def quarter_value : ℝ := 0.25
def dime_value : ℝ := 0.10
def penny_value : ℝ := 0.01
def fee_rate : ℝ := 0.10
def received_amount : ℝ := 27

theorem find_nickels :
  let total_amount_before_fee := received_amount / (1 - fee_rate)
  let total_value_quarters := quarters * quarter_value
  let total_value_dimes := dimes * dime_value
  let total_value_pennies := pennies * penny_value
  let total_value_quarters_dimes_pennies := total_value_quarters + total_value_dimes + total_value_pennies
  let value_nickels := total_amount_before_fee - total_value_quarters_dimes_pennies
  (value_nickels / 0.05).to_nat = 20 :=
by
  sorry

end find_nickels_l139_139216


namespace one_girl_made_a_mistake_l139_139994

variables (c_M c_K c_L c_O : ℤ)

theorem one_girl_made_a_mistake (h₁ : c_M + c_K = c_L + c_O + 12) (h₂ : c_K + c_L = c_M + c_O - 7) :
  false := by
  -- Proof intentionally missing
  sorry

end one_girl_made_a_mistake_l139_139994


namespace solve_series_l139_139244

theorem solve_series (x : ℝ) (hx : 3 + Sum (λ n, (4*n + 3)*(x^n)) = 60) : x = 57 / 61 :=
sorry

end solve_series_l139_139244


namespace sign_pyramid_top_plus_l139_139740

-- Definition of the problem: number of ways to fill the bottom cells
def signPyramid_valid_assignments_count : ℕ := 12

-- Statement: There are exactly 12 ways to fill the five cells in the bottom row to result in a "+" at the top.
theorem sign_pyramid_top_plus :
  ∃ (f : vector (ℤ) 5 → ℤ), 
  (∀ a b c d e : ℤ, a = 1 ∨ a = -1 → b = 1 ∨ b = -1 → c = 1 ∨ c = -1 → d = 1 ∨ d = -1 → e = 1 ∨ e = -1 →
  f ⟨[a, b, c, d, e], @vector.length_eq 5 5 rfl⟩ = 1) ∧
  (finset.univ.filter (λ v : vector ℤ 5, (∀ i, v.nth i = 1 ∨ v.nth i = -1) ∧ 
                       f v = 1)).card = signPyramid_valid_assignments_count := by sorry

end sign_pyramid_top_plus_l139_139740


namespace triangle_minimum_parts_l139_139285

theorem triangle_minimum_parts (ABC : Triangle) : 
  ∃ n, (∀ parts, parts = divide_triangle ABC → flip_and_reassemble parts = ABC → n ≤ 3) := sorry

end triangle_minimum_parts_l139_139285


namespace must_be_nonzero_l139_139457

noncomputable def Q (a b c d : ℝ) : ℝ → ℝ :=
  λ x => x^5 + a * x^4 + b * x^3 + c * x^2 + d * x

theorem must_be_nonzero (a b c d : ℝ)
  (h_roots : ∃ p q r s : ℝ, (∀ y : ℝ, Q a b c d y = 0 → y = 0 ∨ y = -1 ∨ y = p ∨ y = q ∨ y = r ∨ y = s) ∧ p ≠ 0 ∧ p ≠ -1 ∧ q ≠ 0 ∧ q ≠ -1 ∧ r ≠ 0 ∧ r ≠ -1 ∧ s ≠ 0 ∧ s ≠ -1)
  (h_distinct : (∀ x₁ x₂ : ℝ, Q a b c d x₁ = 0 ∧ Q a b c d x₂ = 0 → x₁ ≠ x₂ ∨ x₁ = x₂) → False)
  (h_f_zero : Q a b c d 0 = 0) :
  d ≠ 0 := by
  sorry

end must_be_nonzero_l139_139457


namespace sqrt_meaningful_range_l139_139352

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x :=
by
sorry

end sqrt_meaningful_range_l139_139352


namespace correctness_of_option_C_l139_139880

-- Define the conditions as hypotheses
variable (x y : ℝ)

def condA : Prop := ∀ x: ℝ, x^3 * x^5 = x^15
def condB : Prop := ∀ x y: ℝ, 2 * x + 3 * y = 5 * x * y
def condC : Prop := ∀ x y: ℝ, 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y
def condD : Prop := ∀ x: ℝ, (x - 2)^2 = x^2 - 4

-- State the proof problem is correct
theorem correctness_of_option_C (x y : ℝ) : 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y := by
  sorry

end correctness_of_option_C_l139_139880


namespace quadratic_to_vertex_form_l139_139727

theorem quadratic_to_vertex_form :
  ∀ (x a h k : ℝ), (x^2 - 7*x = a*(x - h)^2 + k) → k = -49 / 4 :=
by
  intros x a h k
  sorry

end quadratic_to_vertex_form_l139_139727


namespace parallelogram_circle_intersections_l139_139283

-- Definitions of points and pertinent lengths of segments
variables {A B C D M K N : Point}
variables {AB AC AD AM AK AN : ℝ}
variables {parallelogram : parallelogram ABCD}
variables {circle : circle_contains A [B,C,D]}

-- Key points on segments
variables (circle_int_AB : meets_inner circle (segment A B) M)
variables (circle_int_AC : meets_inner circle (segment A C) K)
variables (circle_int_AD : meets_inner circle (segment A D) N)

-- The statement to be proven
theorem parallelogram_circle_intersections :
  |AB| * |AM| + |AD| * |AN| = |AK| * |AC| :=
sorry

end parallelogram_circle_intersections_l139_139283


namespace func_eq_l139_139246

-- Define the conditions
def f (x : ℚ) : ℚ := sorry

-- Define the conjecture to be proven
theorem func_eq (f : ℚ → ℚ) :
  (∀ x y : ℚ, f(x - f(y)) = f(x) * f(y)) →
  (∀ x : ℚ, f(x) = 0 ∨ f(x) = 1) :=
sorry

end func_eq_l139_139246


namespace eccentricity_of_hyperbola_l139_139721

theorem eccentricity_of_hyperbola
  (a b : ℝ)
  (hyp : a ≠ 0 ∧ b ≠ 0)
  (p : (ℝ × ℝ) := (3, -4))
  (asymptote_cond : ∃ (k : ℝ), p.2 = k * p.1 ∧ k = - b / a ∨ k = b / a)
  (b_val : b = 4 / 3 * a) :
  let c := real.sqrt (a^2 + b^2)
  in let e := c / a
  in e = 5 / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l139_139721


namespace minimum_value_sum_l139_139619

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  (a / (3 * b) + b / (5 * c) + c / (7 * a)) ≥ (3 / Real.cbrt(105)) :=
by
  sorry

end minimum_value_sum_l139_139619


namespace probability_all_four_genuine_given_weights_equal_l139_139043

def num_genuine_coins := 12
def num_counterfeit_coins := 3
def total_coins := num_genuine_coins + num_counterfeit_coins

-- Define the events
def event_A := "all four selected coins are genuine"
def event_B := "the combined weight of the first pair equals the combined weight of the second pair"

-- State the problem
theorem probability_all_four_genuine_given_weights_equal :
  (P (event_A | event_B)) = 15 / 19 :=
sorry

end probability_all_four_genuine_given_weights_equal_l139_139043


namespace geo_seq_sum_monotone_l139_139364

theorem geo_seq_sum_monotone (q a1 : ℝ) (n : ℕ) (S : ℕ → ℝ) :
  (∀ n, S (n + 1) > S n) ↔ (a1 > 0 ∧ q > 0) :=
sorry -- Proof of the theorem (omitted)

end geo_seq_sum_monotone_l139_139364


namespace largest_possible_value_of_N_l139_139009

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l139_139009


namespace area_difference_l139_139909

theorem area_difference (x : ℝ) (h : 3x < 2x + 10 ∧ x < x + 3) : 
  (2 * x + 10) * (x + 3) - 3 * x ^ 2 = - x ^ 2 + 16 * x + 30 :=
by
  sorry

end area_difference_l139_139909


namespace correct_square_placement_l139_139385

def square (S : Type) := 1 ∈ S → 9 ∈ S → (∀ (i : ℕ), i ∈ {2, 3, 4, 5, 6, 7, 8} → i + 1 ∈ S)

def decide_square_placement (A B C D E F G : ℕ) : Prop :=
 (A = 6) ∧ (B = 2) ∧ (C = 4) ∧ (D = 5) ∧ (E = 3) ∧ (F = 8) ∧ (G = 7)

theorem correct_square_placement (placement : square ℕ):
  decide_square_placement 6 2 4 5 3 8 7 :=
by
  sorry

end correct_square_placement_l139_139385


namespace ratio_of_horses_to_cows_l139_139020

/-- Let H and C be the initial number of horses and cows respectively.
Given that:
1. (H - 15) / (C + 15) = 7 / 3,
2. H - 15 = C + 75,
prove that the initial ratio of horses to cows is 4:1. -/
theorem ratio_of_horses_to_cows (H C : ℕ) 
  (h1 : (H - 15 : ℚ) / (C + 15 : ℚ) = 7 / 3)
  (h2 : H - 15 = C + 75) :
  H / C = 4 :=
by
  sorry

end ratio_of_horses_to_cows_l139_139020


namespace find_distance_between_points_l139_139202

noncomputable def distance_between_points (A B C : Type) [metric_space A] 
  [metric_space B] [metric_space C] (ab_distance bc_distance ca_distance speed1 speed2 : ℝ)
  (travel_time : ℝ) (equidistance : ab_distance = bc_distance ∧ bc_distance = ca_distance) : 
  Prop :=
  let S := ab_distance in
  let min_time := travel_time in
  S = 26

theorem find_distance_between_points :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
  (ab_distance bc_distance ca_distance speed1 speed2 : ℝ)
  (travel_time : ℝ) (equidistance : ab_distance = bc_distance ∧ bc_distance = ca_distance),
  speed1 = 15 ∧ speed2 = 5 ∧ travel_time = 1.4 ∧ 
  distance_between_points A B C ab_distance bc_distance ca_distance speed1 speed2 travel_time equidistance :=
  by 
  intro _ _ _ _ _ _ _ _ _ _ _ _ 
  sorry

end find_distance_between_points_l139_139202


namespace ratio_of_shares_l139_139913

theorem ratio_of_shares 
  (total : ℕ) (B_share : ℕ) (A_share C_share : ℕ)
  (h_total : total = 5400) 
  (h_B : B_share = 1800) 
  (h_A_C : A_share = C_share) 
  (h_remaining : total - B_share = A_share + C_share) 
  (h_valid : A_share = 1800) : 
  (A_share : B_share : C_share) = (1 : 1 : 1) :=
by 
  sorry

end ratio_of_shares_l139_139913


namespace trapezoid_area_l139_139875

noncomputable def area_of_trapezoid : ℝ :=
  let x1 := 5 in
  let x2 := 2.5 in
  let y1 := 10 in
  let y2 := 5 in
  let b1 := x2 in
  let b2 := x1 in
  let h := y1 - y2 in
  (b1 + b2) / 2 * h

theorem trapezoid_area :
  area_of_trapezoid = 18.8 :=
by sorry

end trapezoid_area_l139_139875


namespace minimum_medals_l139_139190

def competitor := ℕ

structure Tournament :=
  (num_competitors : ℕ)
  (skill_levels : fin num_competitors → ℕ)
  (plays_twice : ∀ c : fin num_competitors, ∃ a b : fin num_competitors, a ≠ b ∧ c ≠ a ∧ c ≠ b)

def wins (L1 L2 : ℕ) : Prop :=
  L1 > L2

def receives_medal (c : fin 100) (t : Tournament) : Prop :=
  ∃ a b : fin 100, t.plays_twice c ∧ wins (t.skill_levels c) (t.skill_levels a) ∧ wins (t.skill_levels c) (t.skill_levels b)

theorem minimum_medals (t : Tournament) : (∃ c : fin 100, receives_medal c t) ∧ 
  (∀ d : fin 100, receives_medal d t → d = ⟨99, by decide⟩) →
  ∃ m : ℕ, m = 1 :=
by
  sorry

end minimum_medals_l139_139190


namespace sum_of_squares_and_product_pos_ints_l139_139493

variable (x y : ℕ)

theorem sum_of_squares_and_product_pos_ints :
  x^2 + y^2 = 289 ∧ x * y = 120 → x + y = 23 :=
by
  intro h
  sorry

end sum_of_squares_and_product_pos_ints_l139_139493


namespace find_x0_l139_139787

noncomputable def integral_eq (a b : ℝ) (h : a ≠ 0) (x0 : ℝ) (hx0: 0 < x0) : Prop :=
  let f := λ x, a * x^2 + b
  ∫ 0 to 2, f x = 2 * f x0

theorem find_x0 (a b x0 : ℝ) (h : a ≠ 0) (hx0: 0 < x0) (hint: integral_eq a b h x0 hx0) :
  x0 = 2 * Real.sqrt(3) / 3 :=
sorry

end find_x0_l139_139787


namespace original_volume_l139_139144

-- Considering the conditions from part (a)
variables (V : ℝ)
def afterFirstHour (V : ℝ) : ℝ := V * (1 / 4)
def afterSecondHour (V : ℝ) : ℝ := (afterFirstHour V) * (1 / 4)

-- The proof problem in Lean 4 statement
theorem original_volume (h1 : afterSecondHour V = 0.75) : V = 12 :=
by
  sorry

end original_volume_l139_139144


namespace half_angle_quadrant_l139_139714

-- Define what it means for an angle to be in the first quadrant
def in_first_quadrant (α : ℝ) := ∃ k : ℤ, 2 * k * π < α ∧ α < 2 * k * π + π / 2

-- Define what it means for an angle to be in the first or third quadrant
def in_first_or_third_quadrant (β : ℝ) := ∃ k : ℤ, k * π < β ∧ β < k * π + π / 4

-- The main statement to prove
theorem half_angle_quadrant (α : ℝ) (h : in_first_quadrant α) : in_first_or_third_quadrant (α / 2) :=
sorry

end half_angle_quadrant_l139_139714


namespace sum_of_squares_and_product_l139_139495

open Real

theorem sum_of_squares_and_product (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
    x + y = Real.sqrt 565 := by
  sorry

end sum_of_squares_and_product_l139_139495


namespace max_elevation_l139_139178

def s (t : ℝ) : ℝ := 100 * t - 5 * t^2

theorem max_elevation : ∃ t : ℝ, s t = 500 :=
by {
  use 10,
  unfold s,
  ring,
  norm_num,
}
-- Sorry is added to denote the places where detailed proofs are necessary.

end max_elevation_l139_139178


namespace propositions_true_l139_139936

theorem propositions_true :
  (∀ (x : ℝ), ¬ (∃ (x : ℝ), x^2 + 1 > 3*x) → (x^2 + 1 ≤ 3*x)) ∧
  (∀ (m : ℝ), (-2 = m) → ((m + 2) * (m - 2) + m * (m + 2) = 0) ∧ (-2 = m ∨ 1 = m)) ∧
  (∀ (D E F x1 x2 y1 y2 : ℝ), (D^2 + E^2 - 4*F > 0) →
    x2 * x2 = F → (x2 = x1) →
    y2 * y2 = F → (y2 = y1) →
    (x1 * x2 - y1 * y2 = 0)) ∧
  (∀ (m x: ℝ), |x+1| + |x-3| ≥ m → m ≤ 4) :=
by
  split;
  -- Proofs go here
  sorry

end propositions_true_l139_139936


namespace loaned_books_count_l139_139561

def books_at_start := 75
def books_at_end := 66
def return_rate := 0.80
def not_returned_rate := 0.20
def discrepancy := books_at_start - books_at_end

theorem loaned_books_count (x : ℝ) : 
  books_at_start = 75 →
  books_at_end = 66 →
  return_rate = 0.80 →
  discrepancy = books_at_start - books_at_end →
  not_returned_rate * x = discrepancy →
  x = 45 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end loaned_books_count_l139_139561


namespace total_games_played_14_teams_l139_139856

def total_games (teams : ℕ) : ℕ :=
  teams * (teams - 1) / 2

theorem total_games_played_14_teams : total_games 14 = 91 :=
by
  calc
    total_games 14 = 14 * (14 - 1) / 2 : rfl
    ... = 14 * 13 / 2 : rfl
    ... = 91 : by norm_num

end total_games_played_14_teams_l139_139856


namespace line_intersects_ellipse_l139_139090

theorem line_intersects_ellipse (k : ℝ) :
  let y := k * (1 - 1) + 1 in
  (1 / 9) + (1 / 4) < 1 → 
  ∃ x y, y = k * x - k + 1 ∧ (x^2 / 9) + (y^2 / 4) = 1 :=
by
  intros
  sorry

end line_intersects_ellipse_l139_139090


namespace function_property_l139_139765

theorem function_property (p : ℕ) (hp : p > 2) (f : ℤ → (Fin p)) :
  (∀ n : ℤ, p ∣ (f (f n) - f (n + 1) + 1)) →
  (∀ n : ℤ, f (n + p) = f n) →
  (∀ n : ℤ, f n = ⟨n % p, sorry⟩) := sorry

end function_property_l139_139765


namespace min_time_shoe_horses_l139_139543

variable (blacksmiths horses hooves_per_horse minutes_per_hoof : ℕ)
variable (total_time : ℕ)

theorem min_time_shoe_horses (h_blacksmiths : blacksmiths = 48) 
                            (h_horses : horses = 60)
                            (h_hooves_per_horse : hooves_per_horse = 4)
                            (h_minutes_per_hoof : minutes_per_hoof = 5)
                            (h_total_time : total_time = (horses * hooves_per_horse * minutes_per_hoof) / blacksmiths) :
                            total_time = 25 := 
by
  sorry

end min_time_shoe_horses_l139_139543


namespace number_of_kids_l139_139587

theorem number_of_kids (A K : ℕ) (h1 : A + K = 13) (h2 : 7 * A = 28) : K = 9 :=
by
  sorry

end number_of_kids_l139_139587


namespace closest_integer_to_cubert_of_sum_l139_139120

theorem closest_integer_to_cubert_of_sum : 
  let a := 5
  let b := 7
  let a_cubed := a^3
  let b_cubed := b^3
  let sum_cubed := a_cubed + b_cubed
  7^3 < sum_cubed ∧ sum_cubed < 8^3 →
  Int.abs (Int.floor (Real.cbrt (sum_cubed)) - 8) < 
  Int.abs (Int.floor (Real.cbrt (sum_cubed)) - 7) :=
by
  sorry

end closest_integer_to_cubert_of_sum_l139_139120


namespace incorrect_derivatives_l139_139137

theorem incorrect_derivatives :
  (¬ ((derivative (fun _ => cos (π / 3))) (0) = -sin (π / 3))) ∧
  (¬ ((derivative (fun x => exp (2 * x))) x = exp (2 * x))) := by
  sorry

end incorrect_derivatives_l139_139137


namespace smallest_m_l139_139674

-- Define the arithmetic sequence {a_n} with given conditions
def arithmetic_sequence (a : ℕ → ℝ) :=
  a 5 = 14 ∧ a 7 = 20 ∧ ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n 

-- Define the sum of the sequence {b_n} with given conditions
def b_sum (S : ℕ → ℝ) :=
  S 1 = 2 / 3 / 3 ∧ (∀ n ≥ 2, 3 * S n = S (n - 1) + 2)

-- Define the sequence {b_n}
def geometric_sequence (b : ℕ → ℝ) :=
  b 1 = 2 / 3 ∧ ∀ n : ℕ, b (n + 1) = b n / 3

-- Define the sequence {c_n}
def c_sequence (c : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n : ℕ, c n = a n * b n

-- Define the sum of the first n terms of {c_n}
def T_sum (T : ℕ → ℝ) (c : ℕ → ℝ) :=
  ∀ n : ℕ, T n = ∑ k in finset.range n, c (k + 1)

-- Main theorem statement
theorem smallest_m (a: ℕ → ℝ) (b: ℕ → ℝ) (c: ℕ → ℝ) (S: ℕ → ℝ) (T: ℕ → ℝ) : 
  arithmetic_sequence a →
  b_sum S →
  geometric_sequence b →
  c_sequence c a b →
  T_sum T c →
  ∀ n : ℕ, n > 0 → T n < 7 / 2 :=
begin
  intros,
  sorry
end

end smallest_m_l139_139674


namespace sphere_radius_l139_139855

theorem sphere_radius {r1 r2 : ℝ} (w1 w2 : ℝ) (S : ℝ → ℝ) 
  (h1 : S r1 = 4 * Real.pi * r1^2)
  (h2 : S r2 = 4 * Real.pi * r2^2)
  (w_s1 : w1 = 8)
  (w_s2 : w2 = 32)
  (r2_val : r2 = 0.3)
  (prop : ∀ r, w_s2 = w1 * S r2 / S r1 → w2 = w1 * S r2 / S r1 ) :
  r1 = 0.15 :=
by sorry

end sphere_radius_l139_139855


namespace cube_root_closest_integer_l139_139122

noncomputable def closest_integer_to_cubicroot (n : ℕ) : ℕ :=
  if (Int.natAbs(Int.ofNat n - Int.ofNat (n^(1/3)))) >=
     (Int.natAbs(Int.ofNat n - Int.ofNat (n+1)^(1/3))) then n
  else n + 1

theorem cube_root_closest_integer :
  closest_integer_to_cubicroot (5^3 + 7^3) = 8 :=
by
  sorry

end cube_root_closest_integer_l139_139122


namespace pairs_of_socks_calculation_l139_139401

variable (num_pairs_socks : ℤ)
variable (cost_per_pair : ℤ := 950) -- in cents
variable (cost_shoes : ℤ := 9200) -- in cents
variable (money_jack_has : ℤ := 4000) -- in cents
variable (money_needed : ℤ := 7100) -- in cents
variable (total_money_needed : ℤ := money_jack_has + money_needed)

theorem pairs_of_socks_calculation (x : ℤ) (h : cost_per_pair * x + cost_shoes = total_money_needed) : x = 2 :=
by
  sorry

end pairs_of_socks_calculation_l139_139401


namespace sum_of_integers_k_l139_139636

theorem sum_of_integers_k (k : ℕ) (cond : Nat.choose 15 3 + Nat.choose 15 4 = Nat.choose 16 k) :
  k = 4 ∨ k = 12 → k + 16 - k = 16 :=
by
  intro h
  cases h
  .—
    rw[h]
  .—
    rw[h]
  rw[Nat.add_sub_cancel]
  rw[Nat.add_sub_cancel]
  sorry

#check sum_of_integers_k

end sum_of_integers_k_l139_139636


namespace volume_pyramid_l139_139038

theorem volume_pyramid (A B C D P : ℝ × ℝ × ℝ)
  (h_AB : A = (0, 0, 0))
  (h_B : B = (6, 0, 0))
  (h_C : C = (6, 6, 0))
  (h_D : D = (0, 6, 0))
  (h_P : P = (3, 0, 3 * Real.sqrt 3))
  (h_eq_side : A.dist B = 6 ∧ B.dist P = 6 ∧ P.dist A = 6)
  : volume_pyramid' A B C D P = 36 * Real.sqrt 3 := sorry

end volume_pyramid_l139_139038


namespace find_two_digit_number_l139_139349

theorem find_two_digit_number (x y a b : ℕ) :
  10 * x + y + 46 = 10 * a + b →
  a * b = 6 →
  a + b = 14 →
  (x = 7 ∧ y = 7) ∨ (x = 8 ∧ y = 6) :=
by {
  sorry
}

end find_two_digit_number_l139_139349


namespace Ali_is_8_l139_139196

open Nat

-- Definitions of the variables based on the conditions
def YusafAge (UmarAge : ℕ) : ℕ := UmarAge / 2
def AliAge (YusafAge : ℕ) : ℕ := YusafAge + 3

-- The specific given conditions
def UmarAge : ℕ := 10
def Yusaf : ℕ := YusafAge UmarAge
def Ali : ℕ := AliAge Yusaf

-- The theorem to be proved
theorem Ali_is_8 : Ali = 8 :=
by
  sorry

end Ali_is_8_l139_139196


namespace find_third_number_divisible_by_7_l139_139252

theorem find_third_number_divisible_by_7 {n : ℕ} 
  (h1 : Int.gcd 35 91 = 7) 
  (h2 : Int.gcd 35 91 n = Int.gcd 7 n) 
  : ∃ k : ℕ, n = 7 * k := 
sorry

end find_third_number_divisible_by_7_l139_139252


namespace equations_solutions_l139_139036

-- Definition and statement for Equation 1
noncomputable def equation1_solution1 : ℝ :=
  (-3 + Real.sqrt 17) / 4

noncomputable def equation1_solution2 : ℝ :=
  (-3 - Real.sqrt 17) / 4

-- Definition and statement for Equation 2
def equation2_solution : ℝ :=
  -6

-- Theorem proving the solutions to the given equations
theorem equations_solutions :
  (∃ x : ℝ, 2 * x^2 + 3 * x = 1 ∧ (x = equation1_solution1 ∨ x = equation1_solution2)) ∧
  (∃ x : ℝ, 3 / (x - 2) = 5 / (2 - x) - 1 ∧ x = equation2_solution) :=
by
  sorry

end equations_solutions_l139_139036


namespace find_g5_l139_139075

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l139_139075


namespace distinct_values_of_z_l139_139293

theorem distinct_values_of_z :
  ∃ zs : Finset ℤ, (∀ x y z : ℤ, 100 ≤ x ∧ x ≤ 999 ∧ 
                      100 ≤ y ∧ y ≤ 999 ∧ 
                      x = 100 * (x / 100) + 10 * ((x / 10) % 10) + (x % 10) ∧ 
                      y = 100 * (x % 10) + 10 * ((x / 10) % 10) + (x / 100) ∧ 
                      z = |x - y| → z ∈ zs) ∧ 
                   zs.card = 9 := sorry

end distinct_values_of_z_l139_139293


namespace mass_percentage_Ca_in_CaOH2_is_approx_54_09_l139_139128

noncomputable def mass_percentage_Ca_in_CaOH2 : ℝ :=
  let molar_mass_Ca := 40.08
  let molar_mass_O := 16.00
  let molar_mass_H := 1.01
  let molar_mass_OH := molar_mass_O + molar_mass_H
  let molar_mass_CaOH2 := molar_mass_Ca + 2 * molar_mass_OH
  (molar_mass_Ca / molar_mass_CaOH2) * 100

theorem mass_percentage_Ca_in_CaOH2_is_approx_54_09 :
  mass_percentage_Ca_in_CaOH2 ≈ 54.09 :=
sorry

end mass_percentage_Ca_in_CaOH2_is_approx_54_09_l139_139128


namespace find_g5_l139_139072

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l139_139072
