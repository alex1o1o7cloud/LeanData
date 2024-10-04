import Mathlib
import Mathlib--
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Defs--
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.IntervalIntegral
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Probability.Basic
import Mathlib.Tactic

namespace binary_representation_of_38_l811_811950

theorem binary_representation_of_38 : ∃ binary : ℕ, binary = 0b100110 ∧ binary = 38 :=
by
  sorry

end binary_representation_of_38_l811_811950


namespace find_z_l811_811590

def is_solution (z : Complex) : Prop :=
  Complex.abs (z + 1) = Real.sqrt 10 ∧
  let a := z.re
  let b := z.im
  let c := 3 * z.conj
  RealPart (z - c) = -ImagPart (z - c)

theorem find_z (z : Complex) : 
  is_solution z ↔
  (z = Complex.mk 2 1 ∨ z = Complex.mk (-18 / 5) (-9 / 5)) :=
sorry

end find_z_l811_811590


namespace average_side_length_of_squares_l811_811275

theorem average_side_length_of_squares (a b c : ℕ) (ha : a = 25) (hb : b = 64) (hc : c = 225) :
  (\(\frac{\sqrt{a} + \sqrt{b} + \sqrt{c}}{3}\)) = \(\frac{28}{3}\) :=
by
  sorry

end average_side_length_of_squares_l811_811275


namespace proof_S10_eq_90_l811_811011

open Lean
open Nat

noncomputable theory

-- Define the arithmetic sequence with common difference -2
def a (n : ℕ) : ℤ :=
  let a1 := 18 in
  a1 + (n - 1) * (-2)

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) : ℤ :=
  n * (2 * a 1 + (n - 1) * (-2)) / 2

theorem proof_S10_eq_90 :
  S 10 = 90 :=
  by
  -- This is where the proof steps would go
  sorry

end proof_S10_eq_90_l811_811011


namespace value_of_a_plus_c_l811_811681

-- Define the polynomials
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- Define the condition for the vertex of polynomial f being a root of g
def vertex_of_f_is_root_of_g (a b c d : ℝ) : Prop :=
  g c d (-a / 2) = 0

-- Define the condition for the vertex of polynomial g being a root of f
def vertex_of_g_is_root_of_f (a b c d : ℝ) : Prop :=
  f a b (-c / 2) = 0

-- Define the condition that both polynomials have the same minimum value
def same_minimum_value (a b c d : ℝ) : Prop :=
  f a b (-a / 2) = g c d (-c / 2)

-- Define the condition that the polynomials intersect at (100, -100)
def polynomials_intersect (a b c d : ℝ) : Prop :=
  f a b 100 = -100 ∧ g c d 100 = -100

-- Lean theorem statement for the problem
theorem value_of_a_plus_c (a b c d : ℝ) 
  (h1 : vertex_of_f_is_root_of_g a b c d)
  (h2 : vertex_of_g_is_root_of_f a b c d)
  (h3 : same_minimum_value a b c d)
  (h4 : polynomials_intersect a b c d) :
  a + c = -400 := 
sorry

end value_of_a_plus_c_l811_811681


namespace equation_of_line_AC_l811_811044

-- Definitions of points and lines
structure Point :=
  (x : ℝ)
  (y : ℝ)

def line_equation (A B C : ℝ) (P : Point) : Prop :=
  A * P.x + B * P.y + C = 0

-- Given points and lines
def B : Point := ⟨-2, 0⟩
def altitude_on_AB (P : Point) : Prop := line_equation 1 3 (-26) P

-- Required equation of line AB
def line_AB (P : Point) : Prop := line_equation 3 (-1) 6 P

-- Angle bisector given in the condition
def angle_bisector (P : Point) : Prop := line_equation 1 1 (-2) P

-- Derived Point A
def A : Point := ⟨-1, 3⟩

-- Symmetric point B' with respect to the angle bisector
def B' : Point := ⟨2, 4⟩

-- Required equation of line AC
def line_AC (P : Point) : Prop := line_equation 1 (-3) 10 P

-- The proof statement
theorem equation_of_line_AC :
  ∀ P : Point, (line_AB B ∧ angle_bisector A ∧ P = A → P = B' → line_AC P) :=
by
  intros P h h1 h2
  sorry

end equation_of_line_AC_l811_811044


namespace part_a_part_d_l811_811496

noncomputable def sqrt_2 := Real.sqrt 2
noncomputable def sqrt_3 := Real.sqrt 3

axiom sin_15 : Real.sin (15 * Real.pi / 180) = _ -- Provide the correct value
axiom cos_15 : Real.cos (15 * Real.pi / 180) = _ -- Provide the correct value
axiom tan_15 : Real.tan (15 * Real.pi / 180) = _ -- Provide the correct value

theorem part_a : sqrt_2 * sin_15 + sqrt_2 * cos_15 = sqrt_3 :=
by
  sorry

theorem part_d : (1 + tan_15) / (1 - tan_15) = sqrt_3 :=
by
  sorry

end part_a_part_d_l811_811496


namespace sum_gcf_lcm_l811_811358

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l811_811358


namespace prism_slice_surface_area_l811_811466

-- Define the problem conditions as Lean statements
def PQRSTUV (PQ PR heightX QY ZV : ℝ) : Prop :=
  PQ = 15 ∧ PR = 20  ∧ heightX = 20 ∧
  QY = 1 / 4 * 20 ∧ ZV = 1 / 3 * 20

-- Define the known points in space based on prism slicing
def slicing_points (PQ PR QY ZV : ℝ) (X Y Z : ℝ × ℝ) : Prop :=
  X = (3 / 5 * PQ, 0) ∧ Y = (0, 1 / 4 * PR) ∧ Z = (0, 0)

-- Prove the area of sliced solid (RXYZ) part
theorem prism_slice_surface_area : 
  ∀ (PQ PR heightX QY ZV : ℝ) 
      (X Y Z : ℝ × ℝ), 
  PQRSTUV PQ PR heightX QY ZV ∧ slicing_points PQ PR QY ZV X Y Z
  → surface_area_solid (RXYZ_side_faces PQ PR heightX QY ZV X Y Z) =
     120.39 :=
by
  sorry

end prism_slice_surface_area_l811_811466


namespace percentage_of_part_l811_811833

theorem percentage_of_part (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 50) : (Part / Whole) * 100 = 240 := 
by
  sorry

end percentage_of_part_l811_811833


namespace construct_isosceles_triangle_with_centroid_l811_811853

structure Point (α : Type) :=
(x : α)
(y : α)

structure Triangle (α : Type) :=
(A : Point α)
(B : Point α)
(C : Point α)

def centroid {α : Type} [Add α] [Div α] (T : Triangle α) : Point α :=
  { x := (T.A.x + T.B.x + T.C.x) / 3,
    y := (T.A.y + T.B.y + T.C.y) / 3 }

theorem construct_isosceles_triangle_with_centroid 
  (α : Type) [LinearOrderedField α]
  (s1 s2 : α)
  (Q : Point α)
  (P1 P2 : Point α) :
  (∃ (T : Triangle α), centroid T = Q 
    ∧ (T.A = P2) 
    ∧ (T.B = P1) 
    ∧ (T.A.x - T.B.x) * (T.A.x - T.B.x) + (T.A.y - T.B.y) * (T.A.y - T.B.y) 
      = (T.A.x - T.C.x) * (T.A.x - T.C.x) + (T.A.y - T.C.y) * (T.A.y - T.C.y)) :=
sorry

end construct_isosceles_triangle_with_centroid_l811_811853


namespace correct_option_b_l811_811405

theorem correct_option_b (a : ℝ) : (-2 * a ^ 4) ^ 3 = -8 * a ^ 12 :=
sorry

end correct_option_b_l811_811405


namespace correct_calculation_l811_811805

theorem correct_calculation :
  let a := (\sqrt 3)^2,
    b := sqrt ((-3)^2),
    c := -sqrt (3^2),
    d := (-sqrt 3)^2
  in a = 3 ∧ b ≠ -3 ∧ c ≠ 3 ∧ d ≠ -3 :=
by
  sorry

end correct_calculation_l811_811805


namespace find_max_sum_pair_l811_811531

theorem find_max_sum_pair :
  ∃ a b : ℕ, 2 * a * b + 3 * b = b^2 + 6 * a + 6 ∧ (∀ a' b' : ℕ, 2 * a' * b' + 3 * b' = b'^2 + 6 * a' + 6 → a + b ≥ a' + b') ∧ a = 5 ∧ b = 9 :=
by {
  sorry
}

end find_max_sum_pair_l811_811531


namespace range_of_a_l811_811051

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (1 ≤ x) → ∀ y : ℝ, (1 ≤ y) → (x ≤ y) → (Real.exp (abs (x - a)) ≤ Real.exp (abs (y - a)))) : a ≤ 1 :=
sorry

end range_of_a_l811_811051


namespace distinct_prime_factors_90_l811_811135

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811135


namespace anton_thought_number_l811_811883

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l811_811883


namespace arlekin_first_and_win_l811_811495

/- Define the initial state and rules of the game -/
def initial_matches : ℕ := 2016
def final_matches : ℕ := 2

/- Define the moves that each player can make -/
def Arlekin_moves : list ℕ := [5, 26]
def Pierrot_moves : list ℕ := [9, 23]

/- Define the winner and who made the first move based on the match sequence -/
theorem arlekin_first_and_win (h1 : initial_matches % 7 = 0)
                              (h2 : final_matches = 2)
                              (h3 : ∀ n, n % 7 = 0 → (n + 5) % 7 = 5 ∨ (n + 26) % 7 = 5)
                              (h4 : ∀ n, n % 7 = 0 → (n + 9) % 7 = 2 ∨ (n + 23) % 7 = 2) :
  (/* Arlekin made the first move and won */ sorry)

end arlekin_first_and_win_l811_811495


namespace max_items_sum_l811_811523

theorem max_items_sum (m n : ℕ) (h : 5 * m + 17 * n = 203) : m + n ≤ 31 :=
sorry

end max_items_sum_l811_811523


namespace rectangle_unique_property_l811_811775

-- Define types and properties
structure Parallelogram (α : Type) :=
  (has_parallel_sides : Bool) -- placeholder for parallelogram properties

structure Rectangle (α : Type) extends Parallelogram α :=
  (all_angles_right : Prop)

structure Rhombus (α : Type) extends Parallelogram α :=
  (equal_sides : Prop)
  (all_angles_right : Prop → False) -- rhombuses do not necessarily have right angles

-- Theorem stating that the unique property of rectangles but not rhombuses is having all four angles as right angles
theorem rectangle_unique_property {α : Type} :
  ∀ (r : Rectangle α) (rh : Rhombus α), r.all_angles_right ∧ ¬rh.all_angles_right :=
by
  sorry

end rectangle_unique_property_l811_811775


namespace relationship_between_a_b_c_l811_811561

def a : ℝ := log 9 / log 3
def b : ℝ := 2 ^ 0.7
def c : ℝ := (1 / 2) ^ (-2 / 3)

theorem relationship_between_a_b_c : a > b ∧ b > c := 
by
  sorry

end relationship_between_a_b_c_l811_811561


namespace ratio_of_first_term_to_common_difference_l811_811955

theorem ratio_of_first_term_to_common_difference 
  (a d : ℤ) 
  (h : 15 * a + 105 * d = 3 * (10 * a + 45 * d)) :
  a = -2 * d :=
by 
  sorry

end ratio_of_first_term_to_common_difference_l811_811955


namespace count_arithmetic_sequence_l811_811167

theorem count_arithmetic_sequence: 
  ∃ n : ℕ, (2 + (n - 1) * 3 = 2014) ∧ n = 671 := 
sorry

end count_arithmetic_sequence_l811_811167


namespace proof_problem1_proof_problem2_proof_problem3_proof_problem4_l811_811834

noncomputable def problem1 : Prop := 
  2500 * (1/10000) = 0.25

noncomputable def problem2 : Prop := 
  20 * (1/100) = 0.2

noncomputable def problem3 : Prop := 
  45 * (1/60) = 3/4

noncomputable def problem4 : Prop := 
  1250 * (1/10000) = 0.125

theorem proof_problem1 : problem1 := by
  sorry

theorem proof_problem2 : problem2 := by
  sorry

theorem proof_problem3 : problem3 := by
  sorry

theorem proof_problem4 : problem4 := by
  sorry

end proof_problem1_proof_problem2_proof_problem3_proof_problem4_l811_811834


namespace number_of_colorings_l811_811170

-- Define the set of colors
inductive Color
| red | green | blue | yellow

open Color

-- Define the proper divisors function
def proper_divisors (n : ℕ) : List ℕ :=
  match n with
  | 2 => []
  | 3 => []
  | 4 => [2]
  | 5 => []
  | 6 => [2, 3]
  | 7 => []
  | 8 => [2, 4]
  | 9 => [3]
  | 10 => [2, 5]
  | 11 => []
  | _ => []

-- The statement of the problem
theorem number_of_colorings : 
  (∑ (perm : Fin 4 → Color) in Finset.univ, if ∀ n ∈ (Finset.range 10).filter (λ x => 2 ≤ x + 2) then ∀ (d : ℕ), d ∈ proper_divisors (n + 2) → perm ⟨d, sorry⟩ ≠ perm ⟨n + 2, sorry⟩ else 0) = 6144 := sorry

end number_of_colorings_l811_811170


namespace lambda_value_l811_811010

theorem lambda_value (λ : ℝ) 
  (h1 : ∀ (a b : Point ℝ), a * b = 0)
  (h2 : (2, λ) * (-4, 10) = 0) : λ = 4 / 5 :=
by
  sorry

end lambda_value_l811_811010


namespace log_fraction_pow_l811_811967

theorem log_fraction_pow : log (1 / 4) 16 = -2 :=
by {
    sorry
}

end log_fraction_pow_l811_811967


namespace geometric_sequence_a2_a4_sum_l811_811018

theorem geometric_sequence_a2_a4_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), (∀ n, a n = a 1 * q ^ (n - 1)) ∧
    (a 2 * a 4 = 9) ∧
    (9 * (a 1 * (1 - q^4) / (1 - q)) = 10 * (a 1 * (1 - q^2) / (1 - q))) ∧
    (a 2 + a 4 = 10) :=
by
  sorry

end geometric_sequence_a2_a4_sum_l811_811018


namespace number_of_distinct_prime_factors_of_90_l811_811082

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811082


namespace quadratic_roots_range_l811_811182

theorem quadratic_roots_range (k : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (k * x₁^2 - 4 * x₁ + 1 = 0) ∧ (k * x₂^2 - 4 * x₂ + 1 = 0)) 
  ↔ (k < 4 ∧ k ≠ 0) := 
by
  sorry

end quadratic_roots_range_l811_811182


namespace minimum_f_value_minimum_fraction_value_l811_811682

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem minimum_f_value : ∃ x : ℝ, f x = 2 :=
by
  -- proof skipped, please insert proof here
  sorry

theorem minimum_fraction_value (a b : ℝ) (h : a^2 + b^2 = 2) : 
  (1 / (a^2 + 1)) + (4 / (b^2 + 1)) = 9 / 4 :=
by
  -- proof skipped, please insert proof here
  sorry

end minimum_f_value_minimum_fraction_value_l811_811682


namespace expected_number_of_different_faces_l811_811451

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l811_811451


namespace part1_lambda_part1_an_part2_sn_l811_811021

noncomputable def a : ℕ → ℤ
| 1 := 1
| n + 1 := a n + (2 * (n + 1) - 1)
| _ := 0 -- for undefined cases

def b (n : ℕ) : ℤ := (-1) ^ n * (a n + n)

def S (n : ℕ) : ℤ := (List.range n).sumBy b

theorem part1_lambda : λ = 2 := sorry

theorem part1_an (n : ℕ) : a n = n ^ 2 := sorry

theorem part2_sn (n : ℕ) : S (2 * n) = 2 * n ^ 2 + 2 * n := sorry

end part1_lambda_part1_an_part2_sn_l811_811021


namespace distinct_prime_factors_count_l811_811124

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811124


namespace matt_needs_38_plates_l811_811699

def plates_needed (days_with_only_matt_and_son days_with_parents plates_per_day plates_per_person_with_parents : ℕ) : ℕ :=
  (days_with_only_matt_and_son * plates_per_day) + (days_with_parents * 4 * plates_per_person_with_parents)

theorem matt_needs_38_plates :
  plates_needed 3 4 2 2 = 38 :=
by
  sorry

end matt_needs_38_plates_l811_811699


namespace angle_B_magnitude_value_of_b_l811_811187
open Real

theorem angle_B_magnitude (B : ℝ) (h : 2 * sin B - 2 * sin B ^ 2 - cos (2 * B) = sqrt 3 - 1) :
  B = π / 3 ∨ B = 2 * π / 3 := sorry

theorem value_of_b (a B S : ℝ) (hB : B = π / 3) (ha : a = 6) (hS : S = 6 * sqrt 3) :
  let c := 4
  let b := 2 * sqrt 7
  let half_angle_B := 1 / 2 * a * c * sin B
  half_angle_B = S :=
by
  sorry

end angle_B_magnitude_value_of_b_l811_811187


namespace monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l811_811945

noncomputable def f (x : ℝ) := Real.exp x - (1 / 2) * x^2 - x - 1
noncomputable def f' (x : ℝ) := Real.exp x - x - 1
noncomputable def f'' (x : ℝ) := Real.exp x - 1
noncomputable def g (x : ℝ) := -f (-x)

-- Proof of (I)
theorem monotonic_intervals_and_extreme_values_of_f' :
  f' 0 = 0 ∧ (∀ x < 0, f'' x < 0 ∧ f' x > f' 0) ∧ (∀ x > 0, f'' x > 0 ∧ f' x > f' 0) := 
sorry

-- Proof of (II)
theorem f_g_inequality (x : ℝ) (hx : x > 0) : f x > g x :=
sorry

-- Proof of (III)
theorem sum_of_x1_x2 (x1 x2 : ℝ) (h : f x1 + f x2 = 0) (hne : x1 ≠ x2) : x1 + x2 < 0 := 
sorry

end monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l811_811945


namespace pancake_problem_l811_811783

theorem pancake_problem :
  let mom_rate := (100 : ℚ) / 30
  let anya_rate := (100 : ℚ) / 40
  let andrey_rate := (100 : ℚ) / 60
  let combined_baking_rate := mom_rate + anya_rate
  let net_rate := combined_baking_rate - andrey_rate
  let target_pancakes := 100
  let time := target_pancakes / net_rate
  time = 24 := by
sorry

end pancake_problem_l811_811783


namespace distinct_prime_factors_90_l811_811136

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811136


namespace point_N_coordinates_max_area_MNP_perpendicular_line_slope_max_tan_angle_MNP_l811_811588

noncomputable def point (x y : ℝ) := (x, y)

variables (M N : point) (l : ℝ → ℝ) (C : ℝ → ℝ → Prop) (P : point)

-- Conditions
axiom line_passes_through_M : M = point 1 2
axiom line_tangent_to_circle : ∀ (x y : ℝ), C x y → (x - 2)^2 + y^2 = 5
axiom line_intersects_x_axis : ∃ x : ℝ, (l x = 0) ∧ N = point x 0
axiom P_on_circle_C : ∀ (x y : ℝ), C x y → P = point x y

-- Statements to Prove
theorem point_N_coordinates : N = point (-3) 0 := sorry

theorem max_area_MNP : ∃ max_area : ℝ, max_area = 10 := sorry

theorem perpendicular_line_slope (a : ℝ) : l = (λ x, (1/2) * x - 3/2) →
  (∀ (k a : ℝ), a * k = -1 → k = 1/2 → a = -2) := sorry

theorem max_tan_angle_MNP : ∃ max_tan : ℝ, max_tan = 4 / 3 := sorry

end point_N_coordinates_max_area_MNP_perpendicular_line_slope_max_tan_angle_MNP_l811_811588


namespace anton_thought_of_729_l811_811871

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l811_811871


namespace smallest_rel_prime_90_correct_l811_811997

def smallest_rel_prime_90 : ℕ :=
  if h : ∃ x : ℕ, x > 1 ∧ x < 15 ∧ Nat.gcd x 90 = 1 then
    Nat.find h
  else
    0

theorem smallest_rel_prime_90_correct : smallest_rel_prime_90 = 7 := by
  have exists_x : ∃ x, x > 1 ∧ x < 15 ∧ Nat.gcd x 90 = 1 :=
    ⟨7, by norm_num, by norm_num, by norm_num⟩
  exact Nat.find_spec exists_x
sorry

end smallest_rel_prime_90_correct_l811_811997


namespace uncle_taller_than_james_l811_811654

def james_initial_height (uncle_height : ℕ) : ℕ := (2 * uncle_height) / 3

def james_final_height (initial_height : ℕ) (growth_spurt : ℕ) : ℕ := initial_height + growth_spurt

theorem uncle_taller_than_james (uncle_height : ℕ) (growth_spurt : ℕ) :
  uncle_height = 72 →
  growth_spurt = 10 →
  uncle_height - (james_final_height (james_initial_height uncle_height) growth_spurt) = 14 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end uncle_taller_than_james_l811_811654


namespace jia_card_count_jia_card_sum_l811_811215

-- Definitions
def isJiaCard (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 999 ∧ (∀ d in toDigits n, d ≤ 5)

def cardCountJia : ℕ :=
  (finset.filter isJiaCard (finset.range 1000)).card

def sumJia : ℕ :=
  (finset.filter isJiaCard (finset.range 1000)).sum id

-- Theorems to prove
theorem jia_card_count : cardCountJia = 215 :=
  sorry

theorem jia_card_sum : sumJia = 59940 :=
  sorry

end jia_card_count_jia_card_sum_l811_811215


namespace find_equation_AC_l811_811042

noncomputable def triangleABC (A B C : (ℝ × ℝ)) : Prop :=
  B = (-2, 0) ∧ 
  ∃ (lineAB : ℝ × ℝ → ℝ), ∀ P, lineAB P = 3 * P.1 - P.2 + 6 

noncomputable def conditions (A B : (ℝ × ℝ)) : Prop :=
  (3 * B.1 - B.2 + 6 = 0) ∧ 
  (B.1 + 3 * B.2 - 26 = 0) ∧
  (A.1 + A.2 - 2 = 0)

noncomputable def equationAC (A C : (ℝ × ℝ)) : Prop :=
  (C.1 - 3 * C.2 + 10 = 0)

theorem find_equation_AC (A B C : (ℝ × ℝ)) (h₁ : triangleABC A B C) (h₂ : conditions A B) : 
  equationAC A C :=
sorry

end find_equation_AC_l811_811042


namespace satisfies_equation_l811_811760

theorem satisfies_equation (x : ℝ) :
  (x = (Real.log 8) / (Real.log 3) ∨ x = (Real.log 5) / (Real.log 3)) ↔
  3^(2 * x) - 13 * 3^x + 40 = 0 :=
by
  sorry

end satisfies_equation_l811_811760


namespace distinct_prime_factors_of_90_l811_811104

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811104


namespace apple_distribution_l811_811476

theorem apple_distribution :
  let apples : ℕ := 30 in
  let alice_min : ℕ := 3 in
  let becky_min : ℕ := 3 in
  let chris_min : ℕ := 3 in
  ∃ (a b c : ℕ), a + b + c = apples ∧ a ≥ alice_min ∧ b ≥ becky_min ∧ c ≥ chris_min ∧ 
  number_of_ways_to_distribute_apples a b c = 253 :=
sorry

end apple_distribution_l811_811476


namespace percentage_less_than_a_plus_d_l811_811813

-- Define the mean, standard deviation, and given conditions
variables (a d : ℝ)
axiom symmetric_distribution : ∀ x, x = 2 * a - x 

-- Main theorem
theorem percentage_less_than_a_plus_d :
  (∃ (P_less_than : ℝ → ℝ), P_less_than (a + d) = 0.84) :=
sorry

end percentage_less_than_a_plus_d_l811_811813


namespace magnitude_result_l811_811035

variables (a b : ℝ^3)
variable (θ : ℝ)

def angle_between (a b : ℝ^3) : ℝ := 
if ha : a ≠ 0 ∧ b ≠ 0 then 
  real.acos ((a • b) / (|a| * |b|))
else 0

theorem magnitude_result :
  |a| = 1 ∧ |b| = 2 ∧ angle_between a b = real.pi * 2 / 3 → |2 • a + b| = 2 :=
sorry

end magnitude_result_l811_811035


namespace union_eq_M_l811_811225

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def S : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem union_eq_M : M ∪ S = M := by
  /- this part is for skipping the proof -/
  sorry

end union_eq_M_l811_811225


namespace fraction_solved_l811_811801

theorem fraction_solved (N f : ℝ) (h1 : N * f^2 = 6^3) (h2 : N * f^2 = 7776) : f = 1 / 6 :=
by sorry

end fraction_solved_l811_811801


namespace distinct_prime_factors_count_l811_811119

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811119


namespace eggs_sold_l811_811611

/-- Define the notion of trays and eggs in this context -/
def trays_of_eggs : ℤ := 30

/-- Define the initial collection of trays by Haman -/
def initial_trays : ℤ := 10

/-- Define the number of trays dropped by Haman -/
def dropped_trays : ℤ := 2

/-- Define the additional trays that Haman's father told him to collect -/
def additional_trays : ℤ := 7

/-- Define the total eggs sold -/
def total_eggs_sold : ℤ :=
  (initial_trays - dropped_trays) * trays_of_eggs + additional_trays * trays_of_eggs

-- Theorem to prove the total eggs sold
theorem eggs_sold : total_eggs_sold = 450 :=
by 
  -- Insert proof here
  sorry

end eggs_sold_l811_811611


namespace cooling_time_for_boiling_water_l811_811705

/-- Newton's temperature cooling model -/
def cooling_time (θ0 θ1 θ k : ℝ) : ℝ :=
  - (1 / k) * real.log ((θ - θ0) / (θ1 - θ0))

theorem cooling_time_for_boiling_water :
  let θ0 := 20
  let θ1 := 100
  let θ := 40
  let k := 0.2
  ln 2 ≈ 0.7 →
  cooling_time θ0 θ1 θ k ≈ 7 := by
  intros h_approx
  rw [cooling_time, h_approx]
  exact
  sorry

end cooling_time_for_boiling_water_l811_811705


namespace quadratic_function_property_l811_811238

theorem quadratic_function_property
    (a b c : ℝ)
    (f : ℝ → ℝ)
    (h_f_def : ∀ x, f x = a * x^2 + b * x + c)
    (h_vertex : f (-2) = a^2)
    (h_point : f (-1) = 6)
    (h_vertex_condition : -b / (2 * a) = -2)
    (h_a_neg : a < 0) :
    (a + c) / b = 1 / 2 :=
by
  sorry

end quadratic_function_property_l811_811238


namespace z_in_third_quadrant_l811_811581

-- Definition of the complex number z given the imaginary unit i
def complex_z : ℂ := (-5 * complex.I) / (2 + 3 * complex.I)

-- Statement to prove that z lies in the third quadrant
theorem z_in_third_quadrant :
  let a := complex.re complex_z
  let b := complex.im complex_z
  a < 0 ∧ b < 0 :=
sorry

end z_in_third_quadrant_l811_811581


namespace asymptotic_lines_hyperbola_l811_811747

theorem asymptotic_lines_hyperbola (x y : ℝ) :
  (\frac{x ^ 2}{9} - y ^ 2 = 1) → (y = (1 / 3) * x ∨ y = -(1 / 3) * x) :=
by
  sorry

end asymptotic_lines_hyperbola_l811_811747


namespace binomial_expansion_sum_l811_811037

theorem binomial_expansion_sum (n a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℕ)
  (h1 : ∀ k : ℕ, (k = 5) → binom_expansion_coeff_largest (1 - 2 * x) ^ n)
  (h2 : (1 - 2 * x) ^ n = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8) :
  (|a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8|) = 3^8 - 1 :=
by
  sorry

end binomial_expansion_sum_l811_811037


namespace buns_distribution_not_equal_for_all_cases_l811_811294

theorem buns_distribution_not_equal_for_all_cases :
  ∀ (initial_buns : Fin 30 → ℕ),
  (∃ (p : ℕ → Fin 30 → Fin 30), 
    (∀ t, 
      (∀ i, 
        (initial_buns (p t i) = initial_buns i ∨ 
         initial_buns (p t i) = initial_buns i + 2 ∨ 
         initial_buns (p t i) = initial_buns i - 2))) → 
    ¬ ∀ n : Fin 30, initial_buns n = 2) := 
sorry

end buns_distribution_not_equal_for_all_cases_l811_811294


namespace f_of_f_of_neg_three_l811_811052

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2
  else if x = 0 then Real.pi
  else 0

theorem f_of_f_of_neg_three : f (f (-3)) = Real.pi := by
  sorry

end f_of_f_of_neg_three_l811_811052


namespace distinct_prime_factors_of_90_l811_811085

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811085


namespace hillary_minutes_read_on_saturday_l811_811070

theorem hillary_minutes_read_on_saturday :
  let total_minutes := 60
  let friday_minutes := 16
  let sunday_minutes := 16
  total_minutes - (friday_minutes + sunday_minutes) = 28 := by
sorry

end hillary_minutes_read_on_saturday_l811_811070


namespace Anton_thought_number_is_729_l811_811910

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l811_811910


namespace expected_faces_rolled_six_times_l811_811442

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l811_811442


namespace sum_of_gcd_and_lcm_is_28_l811_811386

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l811_811386


namespace distinct_prime_factors_count_l811_811115

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811115


namespace distinct_prime_factors_90_l811_811137

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811137


namespace parallelogram_area_l811_811535

noncomputable def side_length_b : ℝ := 24
noncomputable def side_length_a : ℝ := 18
noncomputable def included_angle_deg : ℝ := 130

-- Convert degrees to radians for sine function
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * (Real.pi / 180)
noncomputable def included_angle_rad := deg_to_rad included_angle_deg

-- Define the height of the parallelogram
noncomputable def height := side_length_b * Real.sin included_angle_rad

-- Define the area of the parallelogram
noncomputable def area := side_length_b * height

theorem parallelogram_area :
  area ≈ 441.24 :=
by
  -- Proof will go here
  sorry

end parallelogram_area_l811_811535


namespace n_squared_divisible_by_144_l811_811178

theorem n_squared_divisible_by_144 (n : ℕ) (h1 : 0 < n) (h2 : ∃ t : ℕ, t = 12 ∧ ∀ d : ℕ, d ∣ n → d ≤ t) : 144 ∣ n^2 :=
sorry

end n_squared_divisible_by_144_l811_811178


namespace problem_I_problem_II_problem_III_l811_811573

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 3

theorem problem_I (a b : ℝ) (h_a : a = 0) :
  (b ≥ 0 → ∀ x : ℝ, 3 * x^2 + b ≥ 0) ∧
  (b < 0 → 
    ∀ x : ℝ, (x < -Real.sqrt (-b / 3) ∨ x > Real.sqrt (-b / 3)) → 
      3 * x^2 + b > 0) := sorry

theorem problem_II (b : ℝ) :
  ∃ x0 : ℝ, f x0 0 b = x0 ∧ (3 * x0^2 + b = 0) ↔ b = -3 := sorry

theorem problem_III :
  ∀ a b : ℝ, ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧
    (3 * x1^2 + 2 * a * x1 + b = 0) ∧
    (3 * x2^2 + 2 * a * x2 + b = 0) ∧
    (f x1 a b = x1) ∧
    (f x2 a b = x2)) := sorry

end problem_I_problem_II_problem_III_l811_811573


namespace dana_total_earnings_l811_811952

-- Define the constants for Dana's hourly rate and hours worked each day
def hourly_rate : ℝ := 13
def friday_hours : ℝ := 9
def saturday_hours : ℝ := 10
def sunday_hours : ℝ := 3

-- Define the total earnings calculation function
def total_earnings (rate : ℝ) (hours1 hours2 hours3 : ℝ) : ℝ :=
  rate * hours1 + rate * hours2 + rate * hours3

-- The main statement
theorem dana_total_earnings : total_earnings hourly_rate friday_hours saturday_hours sunday_hours = 286 := by
  sorry

end dana_total_earnings_l811_811952


namespace third_player_games_l811_811310

theorem third_player_games (p1 p2 p3 : ℕ) (h1 : p1 = 21) (h2 : p2 = 10)
  (total_games : p1 = p2 + p3) : p3 = 11 :=
by
  sorry

end third_player_games_l811_811310


namespace g_of_minus_three_l811_811057

def g (x : ℝ) : ℝ := (3*x - 2) / (x + 2)

theorem g_of_minus_three : g (-3) = 11 := by
  sorry

end g_of_minus_three_l811_811057


namespace area_ratio_of_ABE_to_ABCD_l811_811640

theorem area_ratio_of_ABE_to_ABCD
  (A B C D E : Type)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  [metric_space E]
  (AC BC : ℝ) 
  (BD : ℝ)
  (h1 : AC = 2 * BD)
  (midpoint : E = midpoint A C)
  (angle_bisector : bisects BE ( ∠ABC ))
  : area ABE / area ABCD = 1 / 4 :=
sorry

end area_ratio_of_ABE_to_ABCD_l811_811640


namespace transform_to_identical_numbers_l811_811002

theorem transform_to_identical_numbers (n : ℕ) (h : n ≥ 3) : 
  (∃ x : ℕ, ∀ i ∈ (finset.range (n + 1)).filter(λ i, i ≠ 0), i = x) ↔ (∃ k : ℕ, n ≠ 4 * k + 2) :=
by
  sorry

end transform_to_identical_numbers_l811_811002


namespace gcd_lcm_sum_8_12_l811_811394

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l811_811394


namespace height_difference_l811_811652

-- Define the initial height of James's uncle
def uncle_height : ℝ := 72

-- Define the initial height ratio of James compared to his uncle
def james_initial_height_ratio : ℝ := 2 / 3

-- Define the height gained by James from his growth spurt
def james_growth_spurt : ℝ := 10

-- Define the initial height of James before the growth spurt
def james_initial_height : ℝ := uncle_height * james_initial_height_ratio

-- Define the new height of James after the growth spurt
def james_new_height : ℝ := james_initial_height + james_growth_spurt

-- Theorem: The difference in height between James's uncle and James after the growth spurt is 14 inches
theorem height_difference : uncle_height - james_new_height = 14 := sorry

end height_difference_l811_811652


namespace f_range_sum_l811_811946

noncomputable def f (x : ℝ) : ℝ :=
  x * Real.sin (abs x) + Real.log ((2019 - x) / (2019 + x))

theorem f_range_sum (m n : ℝ) (h1 : ∀ x ∈ Icc (-2018 : ℝ) 2018, f (-x) = -f x)
  (h2 : ∃ m n, (∀ y, f y ∈ Set.Ioo m n) ∧ (m + n = 0)) :
  f (m + n) = 0 :=
by
  rcases h2 with ⟨m, n, h3, h4⟩
  rw [h4]
  exact f 0

end f_range_sum_l811_811946


namespace problem_1_problem_2a_problem_2b_problem_3_l811_811047

noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 / x

theorem problem_1 (b c : ℝ) (h1 : f 1 = 4) (h2 : f 2 = 5) : b = 2 ∧ c = 0 := by
  sorry

theorem problem_2a : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 → f x1 > f x2 := by
  sorry

theorem problem_2b : ∀ x1 x2 : ℝ, 1 < x1 ∧ x1 < x2 → f x1 < f x2 := by
  sorry

theorem problem_3 (m : ℝ) (h : ∃ x ∈ set.Icc (1/2 : ℝ) 3, 
  (1/2) * f x + 4 * m < (1/2) * f (-x) + m^2 + 4) : m ∈ set.Ioo (-∞) 0 ∪ set.Ioo 4 ∞ := by
  sorry

end problem_1_problem_2a_problem_2b_problem_3_l811_811047


namespace polynomial_remainder_l811_811673

theorem polynomial_remainder (P : ℝ → ℝ) (h1 : P 19 = 16) (h2 : P 15 = 8) : 
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 15) * (x - 19) * Q x + 2 * x - 22 :=
by
  sorry

end polynomial_remainder_l811_811673


namespace smallest_positive_period_monotonically_increasing_intervals_max_min_values_in_interval_l811_811593

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sqrt 3 * sin x * cos x

theorem smallest_positive_period (x : ℝ) :
  ∀ x, f (x + π) = f x := 
sorry

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 3 → f x ≤ f (x + δ) :=
sorry

theorem max_min_values_in_interval :
  ∀ x, -π / 6 ≤ x ∧ x ≤ π / 3 → (0 ≤ f x ∧ f x ≤ 3 / 2) :=
sorry

end smallest_positive_period_monotonically_increasing_intervals_max_min_values_in_interval_l811_811593


namespace max_det_of_3x3_matrix_with_distinct_elements_l811_811223

theorem max_det_of_3x3_matrix_with_distinct_elements (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
                                                      (h_sum_pos : a + b + c > 0) :
  let M := {A : Matrix (Fin 3) (Fin 3) ℝ // (∀ i, ∃ j₁ j₂ j₃, A i j₁ = a ∧ A i j₂ = b ∧ A i j₃ = c) ∧
                                          (∀ j, ∃ i₁ i₂ i₃, A i₁ j = a ∧ A i₂ j = b ∧ A i₃ j = c)} in
  (∃ max_val, ∃ n, (∀ A ∈ M, det A ≤ max_val) ∧ (∃ A ∈ M, det A = max_val) ∧ n = 6) :=
sorry

end max_det_of_3x3_matrix_with_distinct_elements_l811_811223


namespace find_share_of_A_l811_811724

variable (A B C : ℝ)
variable (h1 : A = (2/3) * B)
variable (h2 : B = (1/4) * C)
variable (h3 : A + B + C = 510)

theorem find_share_of_A : A = 60 :=
by
  sorry

end find_share_of_A_l811_811724


namespace max_souls_l811_811502

theorem max_souls : 
  ∀ (distribute_nuts : (ℕ × ℕ)),
  distribute_nuts.1 + distribute_nuts.2 = 222 →
  ∀ (N : ℕ), 1 ≤ N ∧ N ≤ 222 →
  ∃ (move_nuts : ℕ), move_nuts ≤ 37 ∧
  (∃(box1 box2 : ℕ), box1 + box2 = N ∨ (∃(third_box : ℕ), (box1 + box2 + third_box) = N)) :=
begin
  sorry
end

end max_souls_l811_811502


namespace total_clothes_donated_l811_811473

theorem total_clothes_donated
  (pants : ℕ) (jumpers : ℕ) (pajama_sets : ℕ) (tshirts : ℕ)
  (friends : ℕ)
  (adam_donation : ℕ)
  (half_adam_donated : ℕ)
  (friends_donation : ℕ)
  (total_donation : ℕ)
  (h1 : pants = 4) 
  (h2 : jumpers = 4) 
  (h3 : pajama_sets = 4 * 2) 
  (h4 : tshirts = 20) 
  (h5 : friends = 3)
  (h6 : adam_donation = pants + jumpers + pajama_sets + tshirts) 
  (h7 : half_adam_donated = adam_donation / 2) 
  (h8 : friends_donation = friends * adam_donation) 
  (h9 : total_donation = friends_donation + half_adam_donated) :
  total_donation = 126 :=
by
  sorry

end total_clothes_donated_l811_811473


namespace distance_to_Rock_Mist_Mountains_l811_811854

theorem distance_to_Rock_Mist_Mountains (d_Sky_Falls : ℕ) (multiplier : ℕ) (d_Rock_Mist : ℕ) :
  d_Sky_Falls = 8 → multiplier = 50 → d_Rock_Mist = d_Sky_Falls * multiplier → d_Rock_Mist = 400 :=
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end distance_to_Rock_Mist_Mountains_l811_811854


namespace infinite_product_eq_four_four_thirds_l811_811509

theorem infinite_product_eq_four_four_thirds :
  ∏' n : ℕ, (4^(n+1)^(1/(2^(n+1)))) = 4^(4/3) :=
sorry

end infinite_product_eq_four_four_thirds_l811_811509


namespace freddy_total_call_cost_l811_811005

def lm : ℕ := 45
def im : ℕ := 31
def lc : ℝ := 0.05
def ic : ℝ := 0.25

theorem freddy_total_call_cost : lm * lc + im * ic = 10.00 := by
  sorry

end freddy_total_call_cost_l811_811005


namespace gcd_lcm_sum_8_12_l811_811348

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l811_811348


namespace convert_38_to_binary_l811_811948

theorem convert_38_to_binary :
  let decimal_to_binary (n : ℕ) : list ℕ :=
    if n = 0 then []
    else (n % 2) :: decimal_to_binary (n / 2)
  decimal_to_binary 38.reverse = [1, 0, 0, 1, 1, 0] :=
by
  sorry

end convert_38_to_binary_l811_811948


namespace solution_l811_811172

theorem solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 := 
by 
  -- Insert proof here
  sorry

end solution_l811_811172


namespace classification_of_numbers_l811_811529

noncomputable def negative_numbers := {-2/3, -π, -|3.14|, -0.1515}
noncomputable def non_negative_integers := {4, 0}
noncomputable def fractions := { -2/3, 22/9, 3, 2/5, -3.14, -0.1515 }
noncomputable def irrational_numbers := { -π, 0.050050005, π/3}

theorem classification_of_numbers :
  { -4/3, -π, -|3.14|, -0.1515 } = negative_numbers ∧ 
  { ({-2})^2, 0 } = non_negative_integers ∧ 
  { -4/3, 22/9, 3, 2/5, -|3.14|, -0.1515 } = fractions ∧ 
  { -π, 0.050050005, π/3 } = irrational_numbers :=
by
  split; sorry

end classification_of_numbers_l811_811529


namespace number_of_distinct_prime_factors_of_90_l811_811077

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811077


namespace skirt_more_than_pants_l811_811850

def amount_cut_off_skirt : ℝ := 0.75
def amount_cut_off_pants : ℝ := 0.5

theorem skirt_more_than_pants : 
  amount_cut_off_skirt - amount_cut_off_pants = 0.25 := 
by
  sorry

end skirt_more_than_pants_l811_811850


namespace number_of_distinct_prime_factors_90_l811_811152

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811152


namespace abs_inequality_solution_l811_811289

theorem abs_inequality_solution {x : ℝ} :
  |x - 3| < 5 ↔ -2 < x ∧ x < 8 :=
begin
  sorry
end

end abs_inequality_solution_l811_811289


namespace dora_sequence_2017th_l811_811482

/-- The 2017th number in the ascending list of positive integers divisible by 2, 3, or 4 is 3026. -/
theorem dora_sequence_2017th :
  let dora_sequence := {n ∈ Nat | (n % 2 = 0) ∨ (n % 3 = 0) ∨ (n % 4 = 0)} in
  sorted_list_nth dora_sequence 2017 = 3026 := 
sorry

end dora_sequence_2017th_l811_811482


namespace lines_skew_l811_811533

theorem lines_skew (b : ℝ) (t u : ℝ)
  (h1 : 3 + 2 * t = 4 + 3 * u)
  (h2 : 2 + 3 * t = 1 + 4 * u)
  (h3 : b + 4 * t = 2 * u)
  (hneq : b ≠ 18) : ¬∃ t u : ℝ, (3 + 2 * t = 4 + 3 * u) ∧ (2 + 3 * t = 1 + 4 * u) ∧ (b + 4 * t = 2 * u) :=
begin
  sorry
end

end lines_skew_l811_811533


namespace num_distinct_prime_factors_90_l811_811131

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811131


namespace sum_gcf_lcm_eq_28_l811_811369

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l811_811369


namespace ice_cream_flavors_l811_811171

theorem ice_cream_flavors : (Nat.choose 8 3) = 56 := 
by {
    sorry
}

end ice_cream_flavors_l811_811171


namespace proposition_1_proposition_2_proposition_3_proposition_4_combine_propositions_l811_811478

theorem proposition_1 (A B : ℝ) (h1 : sin A > sin B) : A > B := 
sorry

theorem proposition_2 {a b : ℝ} (h2 : a > 0) (h3 : b > 0) (h4 : a + b = 4) :
  sqrt (a + 3) + sqrt (b + 2) ≤ 3 * sqrt 2 := 
sorry

theorem proposition_3 (f : ℕ → ℝ) (h5 : ∀ n, f n = linear n) : 
  ∀ n, a_n = f n → is_arithmetic_sequence a_n :=
sorry

theorem proposition_4 {q : ℝ} (n : ℕ) (h6 : q ≠ 1) : 
  ∑_{i=0}^{n-1} q^i = (q * (1 - q^n)) / (1 - q) :=
sorry

theorem combine_propositions :
  (proposition_1 true) ∧ (proposition_2 true) ∧ (proposition_3 true) ∧ (proposition_4 false) := 
by
  split
  · apply proposition_1
  · apply proposition_2
  · apply proposition_3
  · intro h; contradiction

end proposition_1_proposition_2_proposition_3_proposition_4_combine_propositions_l811_811478


namespace find_x_l811_811984

-- Define the series with a function
def series (x : ℝ) : ℝ := ∑ n in (range N), (2 + 5 * n) * (x ^ n)

-- State the main theorem
theorem find_x (x : ℝ) (h1 : series x = 100) (h2 : -1 < x ∧ x < 1) : 
  x = 2 / 25 :=
sorry

end find_x_l811_811984


namespace mary_sold_at_least_twelve_boxes_l811_811245

theorem mary_sold_at_least_twelve_boxes (cases boxes_per_case : ℕ) (extra : ℕ) (h_cases : cases = 2) (h_boxes_per_case : boxes_per_case = 6) :
  let total_boxes := cases * boxes_per_case + extra in
  total_boxes ≥ 12 :=
by
  sorry

end mary_sold_at_least_twelve_boxes_l811_811245


namespace vincent_earnings_l811_811334

theorem vincent_earnings 
  (price_fantasy_book : ℕ)
  (num_fantasy_books_per_day : ℕ)
  (num_lit_books_per_day : ℕ)
  (num_days : ℕ)
  (h1 : price_fantasy_book = 4)
  (h2 : num_fantasy_books_per_day = 5)
  (h3 : num_lit_books_per_day = 8)
  (h4 : num_days = 5) :
  let price_lit_book := price_fantasy_book / 2
      daily_earnings_fantasy := price_fantasy_book * num_fantasy_books_per_day
      daily_earnings_lit := price_lit_book * num_lit_books_per_day
      total_daily_earnings := daily_earnings_fantasy + daily_earnings_lit
      total_earnings := total_daily_earnings * num_days
  in total_earnings = 180 := 
  by 
  {
    sorry
  }

end vincent_earnings_l811_811334


namespace max_souls_guarantee_l811_811503

theorem max_souls_guarantee :
  ∀ (n1 n2 : ℕ), n1 + n2 = 222 →
  ∃ (N : ℕ), 1 ≤ N ∧ N ≤ 222 ∧ 
  ∀ (moved_nuts : ℕ), moved_nuts = if N = n1 ∨ N = n2 ∨ N = n1 + n2 then 0 else (max n1 n2 - min n1 n2) →
  moved_nuts ≤ 37 :=
begin
  sorry
end

end max_souls_guarantee_l811_811503


namespace number_of_elements_in_A_l811_811241

noncomputable def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def isValidTriple (x y z : ℕ) : Prop :=
  x ∈ M ∧ y ∈ M ∧ z ∈ M ∧ 9 ∣ (x ^ 3 + y ^ 3 + z ^ 3)

def A : Set (ℕ × ℕ × ℕ) :=
  { t | isValidTriple t.1 t.2 t.3 }

theorem number_of_elements_in_A : ∃ n, n = 243 ∧ n = Set.card A := sorry

end number_of_elements_in_A_l811_811241


namespace max_non_overlapping_fences_l811_811636

theorem max_non_overlapping_fences (houses : ℕ) (fences : ℕ) 
  (h1 : houses = 100)
  (h2 : ∀ f1 f2, f1 ≠ f2 → (f1 ∩ f2 = ∅))
  (h3 : ∀ f, ∃ h, h ∈ f) :
  fences = 199 :=
by
  intro houses fences h1 h2 h3
  sorry

end max_non_overlapping_fences_l811_811636


namespace anton_thought_number_l811_811879

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l811_811879


namespace curve_focus_x_axis_l811_811281

theorem curve_focus_x_axis : 
    (x^2 - y^2 = 1)
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (a*x^2 + b*y^2 = 1 → False)
    )
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (b*y^2 - a*x^2 = 1 → False)
    )
    ∨ (∃ c : ℝ, c ≠ 0 ∧ 
        (y = c*x^2 → False)
    ) :=
sorry

end curve_focus_x_axis_l811_811281


namespace polynomial_solution_bound_l811_811663

theorem polynomial_solution_bound (S : Finset ℤ) :
  ∃ c : ℕ, ∀ (f : ℤ[X]), f.degree > 0 → (∃ k : ℕ, k ∈ S.toList ∧ f.eval k ∈ S) → ∃n ≤ c, ∀k, (k ∈ S.toList ∧ f.eval k ∈ S) := 
sorry

end polynomial_solution_bound_l811_811663


namespace sum_of_gcd_and_lcm_is_28_l811_811391

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l811_811391


namespace number_of_women_attended_l811_811484

theorem number_of_women_attended
  (m : ℕ) (w : ℕ)
  (men_dance_women : m = 15)
  (women_dance_men : ∀ i : ℕ, i < 15 → i * 4 = 60)
  (women_condition : w * 3 = 60) :
  w = 20 :=
sorry

end number_of_women_attended_l811_811484


namespace number_of_distinct_prime_factors_90_l811_811153

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811153


namespace average_cost_of_5_pillows_l811_811722

theorem average_cost_of_5_pillows:
  let cost_4_pillows := 4 * 5 in
  let cost_5th_pillow := 10 in
  let total_cost := cost_4_pillows + cost_5th_pillow in
  let num_pillows := 5 in
  (total_cost / num_pillows = 6) :=
by
  let cost_4_pillows := 4 * 5
  let cost_5th_pillow := 10
  let total_cost := cost_4_pillows + cost_5th_pillow
  let num_pillows := 5
  show total_cost / num_pillows = 6
  sorry

end average_cost_of_5_pillows_l811_811722


namespace minimum_omega_value_l811_811318

theorem minimum_omega_value (ω : ℝ) (k : ℤ) 
  (h1 : ω > 0)
  (h2 : ∀ x, g x = Real.sin (ω * x - (ω * Real.pi) / 12))
  (h3 : 3 * (ω) - ω / 12 = (4 * k + 2) * Real.pi + Real.pi / 2):
  ω = 2 :=
by
  sorry

end minimum_omega_value_l811_811318


namespace elsa_final_marbles_l811_811962

def initial_marbles : ℕ := 40
def marbles_lost_at_breakfast : ℕ := 3
def marbles_given_to_susie : ℕ := 5
def marbles_bought_by_mom : ℕ := 12
def twice_marbles_given_back : ℕ := 2 * marbles_given_to_susie

theorem elsa_final_marbles :
    initial_marbles
    - marbles_lost_at_breakfast
    - marbles_given_to_susie
    + marbles_bought_by_mom
    + twice_marbles_given_back = 54 := 
by
    sorry

end elsa_final_marbles_l811_811962


namespace anton_thought_number_l811_811903

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l811_811903


namespace volume_ratio_l811_811960

variables (A B : ℝ)

-- Define the initial conditions
def initial_cond_A := (3 / 4) * A
def initial_cond_B := (1 / 4) * B
def final_cond_B := (7 / 8) * B

-- State the main problem
theorem volume_ratio (h : initial_cond_A + initial_cond_B = final_cond_B) : A / B = 5 / 6 :=
sorry

end volume_ratio_l811_811960


namespace original_price_of_painting_l811_811861

theorem original_price_of_painting (purchase_price : ℝ) (fraction : ℝ) (original_price : ℝ) :
  purchase_price = 200 → fraction = 1/4 → purchase_price = original_price * fraction → original_price = 800 :=
by
  intros h1 h2 h3
  -- proof steps here
  sorry

end original_price_of_painting_l811_811861


namespace books_not_sold_l811_811838

theorem books_not_sold (X : ℕ) (H1 : (2/3 : ℝ) * X * 4 = 288) : (1 / 3 : ℝ) * X = 36 :=
by
  -- Proof goes here
  sorry

end books_not_sold_l811_811838


namespace distinct_prime_factors_count_l811_811116

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811116


namespace area_of_triangle_QCA_l811_811516

theorem area_of_triangle_QCA (p : ℝ) : 
  let Q := (0 : ℝ, 12 : ℝ)
  let A := (2 : ℝ, 12 : ℝ)
  let C := (0 : ℝ, p)
  (1 / 2) * (dist (Q.1, Q.2) (A.1, A.2)) * (dist (Q.1, Q.2) (C.1, C.2)) = 12 - p := 
by {
  sorry
}

end area_of_triangle_QCA_l811_811516


namespace stone_breadth_5_l811_811453

theorem stone_breadth_5 (hall_length_m hall_breadth_m stone_length_dm num_stones b₁ b₂ : ℝ) 
  (h1 : hall_length_m = 36) 
  (h2 : hall_breadth_m = 15) 
  (h3 : stone_length_dm = 3) 
  (h4 : num_stones = 3600)
  (h5 : hall_length_m * 10 * hall_breadth_m * 10 = 54000)
  (h6 : stone_length_dm * b₁ * num_stones = hall_length_m * 10 * hall_breadth_m * 10) :
  b₂ = 5 := 
  sorry

end stone_breadth_5_l811_811453


namespace lamps_remaining_on_l811_811293

theorem lamps_remaining_on (n : ℕ) (hn : n = 2015) : 
  (∑ k in finset.range n, 
    if k % 2 = 0 ∨ k % 3 = 0 ∨ k % 5 = 0 then 0 else 1) = 1006 :=
by
  sorry

end lamps_remaining_on_l811_811293


namespace number_of_distinct_prime_factors_of_90_l811_811106

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811106


namespace anton_thought_number_l811_811875

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l811_811875


namespace lyra_remaining_budget_l811_811696

variable (budget : ℕ)
variable (cost_fried_chicken : ℕ)
variable (cost_beef_per_pound : ℕ)
variable (pounds_beef : ℕ)

def remaining_budget (budget : ℕ) (cost_fried_chicken : ℕ) (cost_beef_per_pound : ℕ) (pounds_beef : ℕ) : ℕ :=
  budget - (cost_fried_chicken + cost_beef_per_pound * pounds_beef)

theorem lyra_remaining_budget : 
  remaining_budget 80 12 3 5 = 53 := 
by 
  -- substitute the variable values
  let budget_val : ℕ := 80
  let cost_fried_chicken_val : ℕ := 12
  let cost_beef_per_pound_val : ℕ := 3
  let pounds_beef_val : ℕ := 5
  
  -- calculate the cost of the beef
  let cost_beef : ℕ := cost_beef_per_pound_val * pounds_beef_val
  have h_cost_beef : cost_beef = 15 := by rfl
  
  -- calculate the total cost of the food
  let total_cost_food : ℕ := cost_fried_chicken_val + cost_beef
  have h_total_cost_food : total_cost_food = 27 := by rfl
  
  -- calculate the remaining budget
  show budget_val - total_cost_food = 53 from
    calc
      budget_val - total_cost_food = 80 - 27 : by rfl
                               ... = 53 : by rfl

end lyra_remaining_budget_l811_811696


namespace hyperbola_line_intersection_unique_l811_811601

theorem hyperbola_line_intersection_unique :
  ∀ (x y : ℝ), (x^2 / 9 - y^2 = 1) ∧ (y = 1/3 * (x + 1)) → ∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y :=
by
  sorry

end hyperbola_line_intersection_unique_l811_811601


namespace total_clothing_donated_l811_811474

-- Definition of the initial donation by Adam
def adam_initial_donation : Nat := 4 + 4 + 4*2 + 20 -- 4 pairs of pants, 4 jumpers, 4 pajama sets (8 items), 20 t-shirts

-- Adam's friends' total donation
def friends_donation : Nat := 3 * adam_initial_donation

-- Adam's donation after keeping half
def adam_final_donation : Nat := adam_initial_donation / 2

-- Total donation being the sum of Adam's and friends' donations
def total_donation : Nat := adam_final_donation + friends_donation

-- The statement to prove
theorem total_clothing_donated : total_donation = 126 := by
  -- This is skipped as per instructions
  sorry

end total_clothing_donated_l811_811474


namespace freddy_total_call_cost_l811_811004

def lm : ℕ := 45
def im : ℕ := 31
def lc : ℝ := 0.05
def ic : ℝ := 0.25

theorem freddy_total_call_cost : lm * lc + im * ic = 10.00 := by
  sorry

end freddy_total_call_cost_l811_811004


namespace trajectory_of_P_is_straight_line_l811_811606

open Real EuclideanGeometry

noncomputable def F1 : Point := (-4, 0)
noncomputable def F2 : Point := (4, 0)

def is_trajectory_straight_line (P : Point) : Prop :=
  (dist P F1 + dist P F2 = 8)

theorem trajectory_of_P_is_straight_line :
  ∀ P : Point, is_trajectory_straight_line P → 
    ∀ Q R : Point, (dist Q F1 + dist Q F2 = 8) ∧ (dist R F1 + dist R F2 = 8) → 
    collinear {P, Q, R} :=
sorry

end trajectory_of_P_is_straight_line_l811_811606


namespace anton_thought_number_l811_811896

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l811_811896


namespace least_number_of_marbles_l811_811847

theorem least_number_of_marbles (n : ℕ) : 
  (∀ k ∈ [3, 4, 6, 8, 9], k ∣ n) → n = 72 :=
by
  intro h
  have h_lcm : LCM [3, 4, 6, 8, 9] = 72 := 
    by sorry
  exact nat.eq_of_dvd_of_lt (h 3 (by simp)) 
    (lt_of_lt_of_le (h_lcm.symm ▸ lcm_le mul_le_mul_left)
    (mul_le_mul_div [3, 4, 6, 8, 9]))

-- LCM definition to enforce the least common multiple
-- requirement within Lean's framework.
def LCM (list : list ℕ) : ℕ := list.foldr nat.lcm 1

end least_number_of_marbles_l811_811847


namespace solve_for_y_l811_811733

theorem solve_for_y : (∃ y : ℤ, 3^(y + 3) = (3^4)^y) ↔ y = 1 :=
by
  sorry

end solve_for_y_l811_811733


namespace trapezium_other_side_length_l811_811536

theorem trapezium_other_side_length :
  ∃ (x : ℝ), 1/2 * (18 + x) * 17 = 323 ∧ x = 20 :=
by
  sorry

end trapezium_other_side_length_l811_811536


namespace shoot_down_probability_l811_811193

-- Define the probabilities
def P_hit_nose := 0.2
def P_hit_middle := 0.4
def P_hit_tail := 0.1
def P_miss := 0.3

-- Define the condition: probability of shooting down the plane with at most 2 shots
def condition := (P_hit_tail + (P_hit_nose * P_hit_nose) + (P_miss * P_hit_tail))

-- Proving the probability matches the required value
theorem shoot_down_probability : condition = 0.23 :=
by
  sorry

end shoot_down_probability_l811_811193


namespace sum_of_cubes_of_consecutive_integers_div_by_9_l811_811253

theorem sum_of_cubes_of_consecutive_integers_div_by_9 (x : ℤ) : 
  let a := (x - 1) ^ 3
  let b := x ^ 3
  let c := (x + 1) ^ 3
  (a + b + c) % 9 = 0 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_div_by_9_l811_811253


namespace magnitude_of_vector_addition_l811_811269

open Real

-- Define the angle in radians for 60 degrees
def theta : ℝ := π / 3

-- Define vectors a and b
def vec_a : ℝ × ℝ := (2, 0)
def vec_b : ℝ × ℝ -- b is of unit length but its direction is unspecified yet
:= (x, y) -- x and y should be defined such that the magnitude is 1

axiom magnitude_b : sqrt (x^2 + y^2) = 1

-- Condition: angle theta between vectors a and b is 60 degrees, hence cos(theta) = 1/2
def cos_theta : ℝ := 1 / 2

-- Define the magnitude of vector sum
def magnitude_vec_sum : ℝ :=
  sqrt ((fst vec_a)^2 + 4 * (fst vec_b)^2 + 4 * (snd vec_b)^2 + 
        2 * (fst vec_a) * 2 * (fst vec_b) * cos_theta)

-- The problem statement to be proven
theorem magnitude_of_vector_addition : magnitude_vec_sum = 2 * sqrt 3 := sorry

end magnitude_of_vector_addition_l811_811269


namespace anton_thought_number_l811_811927

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l811_811927


namespace original_mixture_volume_l811_811427

theorem original_mixture_volume :
  ∃ (x : ℝ), (0.25 * x / (x + 3) = 0.20833333333333334) ∧ x = 15 :=
by {
  use 15,
  split,
  {
    norm_num,
  },
  {
    refl,
  }
}

end original_mixture_volume_l811_811427


namespace find_m_n_l811_811177

axiom original_equation (x m n : ℝ) : x^2 + m * x - 15 = (x + 5) * (x + n)

theorem find_m_n (x : ℝ) : ∃ (m n : ℝ), 
  x^2 + m * x - 15 = (x + 5) * (x + n) ∧ m = 2 ∧ n = -3 :=
by {
  let m := 2,
  let n := -3,
  use [m, n],
  split,
  { exact original_equation x m n, },
  split,
  { exact rfl, },
  { exact rfl, },
}

end find_m_n_l811_811177


namespace product_of_roots_is_one_l811_811226

noncomputable def Q : Polynomial ℚ := by
  have h1 : (X^3 : Polynomial ℚ) - 5 = 0 := sorry
  have h2 : 5 * (X^3 : Polynomial ℚ) + 15 * (X^2 : Polynomial ℚ) + 15 * (X : Polynomial ℚ) - 59 = 0 := sorry
  have h3 : Polynomial.degree ?Q ≤ 3 := sorry
  exists Q
  exact ⟨h1, h2, h3⟩

theorem product_of_roots_is_one {Q : Polynomial ℚ} (hQ : Q = minimal_poly_with_rational_coeff (sqrt5 - sqrt25)) :
  (∏ root in Q.roots, root) = 1 := by
  sorry

end product_of_roots_is_one_l811_811226


namespace find_max_term_of_sequence_l811_811063

theorem find_max_term_of_sequence :
  ∃ m : ℕ, (m = 8) ∧ ∀ n : ℕ, (0 < n → n ≠ m → a_n = (n - 7) / (n - 5 * Real.sqrt 2)) :=
by
  sorry

end find_max_term_of_sequence_l811_811063


namespace part_a_part_b_l811_811418

noncomputable def arithmetic_progression_a (a₁: ℕ) (r: ℕ) : ℕ :=
  a₁ + 3 * r

theorem part_a (a₁: ℕ) (r: ℕ) (h_a₁ : a₁ = 2) (h_r : r = 3) : arithmetic_progression_a a₁ r = 11 := 
by 
  sorry

noncomputable def arithmetic_progression_formula (d: ℕ) (r: ℕ) (n: ℕ) : ℕ :=
  d + (n - 1) * r

theorem part_b (a3: ℕ) (a6: ℕ) (a9: ℕ) (a4_plus_a7_plus_a10: ℕ) (a_sum: ℕ) (h_a3 : a3 = 3) (h_a6 : a6 = 6) (h_a9 : a9 = 9) 
  (h_a4a7a10 : a4_plus_a7_plus_a10 = 207) (h_asum : a_sum = 553) 
  (h_eqn1: 3 * a3 + a6 * 2 = 207) (h_eqn2: a_sum = 553): 
  arithmetic_progression_formula 9 10 11 = 109 := 
by 
  sorry

end part_a_part_b_l811_811418


namespace locus_of_intersection_point_P_l811_811566

def locus_of_P (a b : ℝ) (h : 0 < a ∧ a < b) : set (ℝ × ℝ) :=
  {p | p.1 = (a + b) / 2}

theorem locus_of_intersection_point_P (a b : ℝ) (h : 0 < a ∧ a < b) :
  ∃ (P : ℝ × ℝ), P ∈ locus_of_P a b h :=
sorry

end locus_of_intersection_point_P_l811_811566


namespace distinct_prime_factors_of_90_l811_811102

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811102


namespace sum_gcf_lcm_l811_811355

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l811_811355


namespace symmetric_points_sum_l811_811576

theorem symmetric_points_sum (a b : ℝ) (h1 : B = (-A)) (h2 : A = (1, a)) (h3 : B = (b, 2)) : a + b = -3 := by
  sorry

end symmetric_points_sum_l811_811576


namespace timPaid_l811_811312

def timPayment (p : Real) (d : Real) : Real :=
  p - p * d

theorem timPaid (p : Real) (d : Real) (a : Real) : 
  p = 1200 ∧ d = 0.15 → a = 1020 :=
by
  intro h
  cases h with hp hd
  rw [hp, hd]
  have hdiscount : 1200 * 0.15 = 180 := by norm_num
  have hpayment : 1200 - 180 = 1020 := by norm_num
  rw [hdiscount, hpayment]
  sorry

end timPaid_l811_811312


namespace least_distance_midpoints_l811_811510

open Real

structure RegularTetrahedron (A B C D : Point) :=
  (edge_length : ℝ)
  (is_unit_length : edge_length = 1)
  (is_regular : ∀ (X Y : Point), (X ≠ Y) → (dist X Y = edge_length))

def least_distance_point_point (A B C D P Q : Point) [RegularTetrahedron A B C D] 
  (P_on_AB : P ∈ line_segment A B) (Q_on_BC : Q ∈ line_segment B C) : ℝ :=
  sorry

theorem least_distance_midpoints (A B C D P Q : Point) [RegularTetrahedron A B C D]
  (P_on_AB : P ∈ line_segment A B) (Q_on_BC : Q ∈ line_segment B C) 
  (P_is_midpoint : P = midpoint A B) (Q_is_midpoint : Q = midpoint B C) :
  least_distance_point_point A B C D P Q = 1/2 :=
sorry

end least_distance_midpoints_l811_811510


namespace area_enclosed_by_curves_l811_811987

theorem area_enclosed_by_curves :
  let f := λ x : ℝ, x^2 - 1,
      g := λ x : ℝ, 2 - 2x^2  in
  ∫ x in 0..1, (g x - f x) = 2 * 3 - 2 * ∫ x in 0..1, x^2 :=
by sorry

end area_enclosed_by_curves_l811_811987


namespace anton_thought_number_l811_811880

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l811_811880


namespace gcd_lcm_sum_8_12_l811_811345

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l811_811345


namespace prove_perpendicular_BF_KL_l811_811020

-- Definitions of the geometric elements and conditions
variables {A B C D K L F : Type*}

-- Assuming A, B, C, D are points forming a rectangle, and defining K, L and F as specified
axiom rectangle (A B C D : Type*) : Prop
axiom perpendicular (x y : Type*) : Prop
axiom line_intersects (x y z : Type*) : Prop
axiom side_intersects (x y : Type*) (z : Type*) : Prop
axiom extends (x y: Type*) : Prop
axiom intersection_point (x y : Type*) : Type*

variables (ABCD : rectangle A B C D)
variables (BK_perpendicular_AD : perpendicular B K)
variables (BL_perpendicular_CD_ext : perpendicular B L)
variables (K_is_intersection_AD : side_intersects K A D)
variables (L_is_intersection_CD_ext : side_intersects L D (extends C D))
variables (F_is_intersection_KL_AC : line_intersects F (intersection_point K L))

-- Proof Statement
theorem prove_perpendicular_BF_KL : perpendicular (intersection_point B F) (intersection_point K L) :=
sorry

end prove_perpendicular_BF_KL_l811_811020


namespace tetrahedral_section_volume_l811_811465

noncomputable def volume_of_tetrahedral_section (t : ℝ) (AE BF : ℝ) (parallel_AB_EF : Prop) := 
  let S := 6 * real.sqrt 3 in
  (real.sqrt 2 * S^3) / 12

theorem tetrahedral_section_volume (t AE BF : ℝ) (parallel_AB_EF : Prop) (t_val : t = 3 * real.sqrt 3) 
  (AE_val : AE = t) (BF_val : BF = 2 * t) : 
  volume_of_tetrahedral_section t AE BF parallel_AB_EF = 54 * real.sqrt 6 :=
by 
  -- Note that the details of the proof are skipped, represented by sorry.
  sorry

end tetrahedral_section_volume_l811_811465


namespace number_of_distinct_prime_factors_of_90_l811_811110

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811110


namespace gcd_lcm_sum_8_12_l811_811351

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l811_811351


namespace sum_of_gcd_and_lcm_is_28_l811_811387

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l811_811387


namespace number_of_distinct_prime_factors_of_90_l811_811107

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811107


namespace intersection_point_l811_811992

noncomputable def point := (x : ℚ, y : ℚ)

def line1 (p : point) : Prop := 3 * p.2 = -2 * p.1 + 6
def line2 (p : point) : Prop := -2 * p.2 = 7 * p.1 - 3

theorem intersection_point : ∃ p : point, line1 p ∧ line2 p ∧ p = (-3/17, 36/17) := sorry

end intersection_point_l811_811992


namespace distinct_prime_factors_90_l811_811139

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811139


namespace find_equation_AC_l811_811043

noncomputable def triangleABC (A B C : (ℝ × ℝ)) : Prop :=
  B = (-2, 0) ∧ 
  ∃ (lineAB : ℝ × ℝ → ℝ), ∀ P, lineAB P = 3 * P.1 - P.2 + 6 

noncomputable def conditions (A B : (ℝ × ℝ)) : Prop :=
  (3 * B.1 - B.2 + 6 = 0) ∧ 
  (B.1 + 3 * B.2 - 26 = 0) ∧
  (A.1 + A.2 - 2 = 0)

noncomputable def equationAC (A C : (ℝ × ℝ)) : Prop :=
  (C.1 - 3 * C.2 + 10 = 0)

theorem find_equation_AC (A B C : (ℝ × ℝ)) (h₁ : triangleABC A B C) (h₂ : conditions A B) : 
  equationAC A C :=
sorry

end find_equation_AC_l811_811043


namespace max_cigarettes_with_staggered_packing_l811_811839

theorem max_cigarettes_with_staggered_packing :
  ∃ n : ℕ, n > 160 ∧ n = 176 :=
by
  let diameter := 2
  let rows_initial := 8
  let cols_initial := 20
  let total_initial := rows_initial * cols_initial
  have h1 : total_initial = 160 := by norm_num
  let alternative_packing_capacity := 176
  have h2 : alternative_packing_capacity > total_initial := by norm_num
  use alternative_packing_capacity
  exact ⟨h2, rfl⟩

end max_cigarettes_with_staggered_packing_l811_811839


namespace num_distinct_prime_factors_90_l811_811133

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811133


namespace ellipse_length_AB_ellipse_slope_one_l811_811029

-- Definitions:
def ellipse (x y : ℝ) (b : ℝ) : Prop := x^2 + (y^2 / b^2) = 1
def arithmetic_prog (a b c : ℝ) : Prop := 2 * b = a + c

-- Problem 1
theorem ellipse_length_AB (AF2 AB BF2 : ℝ) (h1 : AF2 + AB + BF2 = 4) (h2 : arithmetic_prog AF2 AB BF2) : AB = 4 / 3 :=
by
  sorry

-- Problem 2
theorem ellipse_slope_one (b : ℝ) (h1 : 0 < b) (h2 : b < 1) (h3 : ∀ x y, ellipse x y b → y = x + sqrt (1 - b^2)) : b = sqrt 2 / 2 :=
by
  -- Assuming |AB| = 4/3 and substituting in calculations
  have hAB: ∀ AF2 AB BF2 : ℝ, AF2 + AB + BF2 = 4 → arithmetic_prog AF2 AB BF2 → AB = 4 / 3 := 
    ellipse_length_AB
  sorry

end ellipse_length_AB_ellipse_slope_one_l811_811029


namespace arithmetic_square_root_of_16_is_4_l811_811746

theorem arithmetic_square_root_of_16_is_4 : ∃ x : ℤ, x * x = 16 ∧ x = 4 := 
sorry

end arithmetic_square_root_of_16_is_4_l811_811746


namespace minimum_value_l811_811995

noncomputable def min_y := 
  min (λ x : ℝ, sin (x + π / 4) + sin (x + π / 3) + cos (x + π / 4)) 0 (π / 12)

theorem minimum_value :
  min_y = -- calculated value
sorry

end minimum_value_l811_811995


namespace find_range_of_a_l811_811595

noncomputable def f (x : ℝ) : ℝ := (1 / Real.exp x) - (Real.exp x) + 2 * x - (1 / 3) * x ^ 3

theorem find_range_of_a (a : ℝ) (h : f (3 * a ^ 2) + f (2 * a - 1) ≥ 0) : a ∈ Set.Icc (-1 : ℝ) (1 / 3) :=
sorry

end find_range_of_a_l811_811595


namespace find_a_l811_811694

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ := a * (2^n - 1)
def a_4 (S4 S3 : ℕ) : ℕ := S4 - S3
axiom given_sum_relation : ∀ a n, sequence_sum a n

theorem find_a : (S4 S3 : ℕ) → (h : a_4 S4 S3 = 24) → (sequence_sum a 4 - sequence_sum a 3 = 24) → (8 * a = 24) → a = 3 :=
by
  intros,
  sorry

end find_a_l811_811694


namespace problem_statement_l811_811233

open Classical

variable (n : ℕ) (a b : Fin n → ℝ)
variable (h₁ : 0 < n)
variable (h₂ : ∀ i, 0 < a i)
variable (h₃ : ∀ i, 0 < b i)
variable (hSumA : (∑ i, a i) = 1)
variable (hSumB : (∑ i, b i) = 1)

theorem problem_statement :
  (∑ i, abs (a i - b i)) ≤ (2 - min (λ i, a i / b i) (finset.univ) - min (λ i, b i / a i) (finset.univ))
  := sorry

end problem_statement_l811_811233


namespace employees_females_l811_811633

theorem employees_females
  (total_employees : ℕ)
  (adv_deg_employees : ℕ)
  (coll_deg_employees : ℕ)
  (males_coll_deg : ℕ)
  (females_adv_deg : ℕ)
  (females_coll_deg : ℕ)
  (h1 : total_employees = 180)
  (h2 : adv_deg_employees = 90)
  (h3 : coll_deg_employees = 180 - 90)
  (h4 : males_coll_deg = 35)
  (h5 : females_adv_deg = 55)
  (h6 : females_coll_deg = 90 - 35) :
  females_coll_deg + females_adv_deg = 110 :=
by
  sorry

end employees_females_l811_811633


namespace paths_expression_value_l811_811483

theorem paths_expression_value (P Q : ℕ) (hP : P = 130) (hQ : Q = 65) : 
  P - 2 * Q + 2014 = 2014 :=
by
  rw [hP, hQ]
  simp

end paths_expression_value_l811_811483


namespace number_of_ways_to_assign_friends_to_teams_l811_811169

theorem number_of_ways_to_assign_friends_to_teams (n m : ℕ) (h_n : n = 7) (h_m : m = 4) : m ^ n = 16384 :=
by
  rw [h_n, h_m]
  exact pow_succ' 4 6

end number_of_ways_to_assign_friends_to_teams_l811_811169


namespace complex_norm_solution_l811_811739

noncomputable def complex_norm (z : Complex) : Real :=
  Complex.abs z

theorem complex_norm_solution (w z : Complex) 
  (wz_condition : w * z = 24 - 10 * Complex.I)
  (w_norm_condition : complex_norm w = Real.sqrt 29) :
  complex_norm z = (26 * Real.sqrt 29) / 29 :=
by
  sorry

end complex_norm_solution_l811_811739


namespace second_mission_duration_l811_811218

theorem second_mission_duration 
  (planned_first_mission_days : ℕ) 
  (duration_percentage_increase : ℕ) 
  (total_mission_days : ℕ) :
  planned_first_mission_days = 5 →
  duration_percentage_increase = 60 →
  total_mission_days = 11 →
  let first_mission_days := planned_first_mission_days + (planned_first_mission_days * duration_percentage_increase / 100) in
  let second_mission_days := total_mission_days - first_mission_days in
  second_mission_days = 3 :=
by
  intros hplanned hduration htotal
  let first_mission_days := planned_first_mission_days + (planned_first_mission_days * duration_percentage_increase / 100)
  let second_mission_days := total_mission_days - first_mission_days
  sorry

end second_mission_duration_l811_811218


namespace distinct_prime_factors_90_l811_811160

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811160


namespace roots_real_polynomial_ab_le_zero_l811_811250

theorem roots_real_polynomial_ab_le_zero (a b c : ℝ) (h : ∀ x : ℝ, x^4 + a*x^3 + b*x + c = 0 → x ∈ ℝ) : a * b ≤ 0 := 
  sorry

end roots_real_polynomial_ab_le_zero_l811_811250


namespace arithmetic_sequence_sum_l811_811022

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a →
  (a 4 + a 8 = 8) →
  (∑ i in Finset.range 11, a i) = 44 := 
by
  sorry

end arithmetic_sequence_sum_l811_811022


namespace solve_diff_eq_l811_811991

noncomputable def general_solution (y : ℝ → ℝ) : Prop :=
  ∃ (C1 C2 C3 : ℝ), ∀ x : ℝ, y x = Math.cos x - Math.sin x + (C1 * x^2) / 2 + C2 * x + C3

theorem solve_diff_eq (y : ℝ → ℝ) :
  (∀ x : ℝ, deriv (deriv (deriv y)) x = Math.sin x + Math.cos x) →
  general_solution y :=
by
  intros h
  -- Proof would be provided here, currently skipped
  sorry

end solve_diff_eq_l811_811991


namespace siblings_gmat_scores_l811_811311

-- Define the problem conditions
variables (x y z : ℝ)

theorem siblings_gmat_scores (h1 : x - y = 1/3) (h2 : z = (x + y) / 2) : 
  y = x - 1/3 ∧ z = x - 1/6 :=
by
  sorry

end siblings_gmat_scores_l811_811311


namespace sum_of_GCF_and_LCM_l811_811367

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l811_811367


namespace line_equation_through_point_and_angle_l811_811282

noncomputable def slope_of_inclination (angle : ℝ) : ℝ :=
Real.tan angle

def line_equation_through_point_slope (p : ℝ × ℝ) (m : ℝ) : (ℝ × ℝ) → Prop :=
λ (x y : ℝ), y - p.2 = m * (x - p.1)

theorem line_equation_through_point_and_angle {x y : ℝ} :
  let p := (-2 : ℝ, 1 : ℝ)
  let angle := Real.pi / 3 -- 60 degrees in radians
  let m := slope_of_inclination angle
  in line_equation_through_point_slope p m (x + 2) (y - 1) :=
by
  let p := (-2 : ℝ, 1 : ℝ)
  let angle := Real.pi / 3
  let m := slope_of_inclination angle
  have h_slope := by rw [slope_of_inclination, Real.tan_pi_div_three]; exact (m = Real.sqrt 3)
  have h_line_eq := by rw [line_equation_through_point_slope]; exact (y - 1 = Real.sqrt 3 * (x + 2))
  exact sorry

end line_equation_through_point_and_angle_l811_811282


namespace log_fraction_pow_l811_811966

theorem log_fraction_pow : log (1 / 4) 16 = -2 :=
by {
    sorry
}

end log_fraction_pow_l811_811966


namespace anton_thought_number_l811_811873

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l811_811873


namespace inequality_always_true_l811_811032

theorem inequality_always_true 
  (a b : ℝ) 
  (h1 : ab > 0) : 
  (b / a) + (a / b) ≥ 2 := 
by sorry

end inequality_always_true_l811_811032


namespace length_of_rectangular_sheet_l811_811323

/-- The length of each rectangular sheet is 10 cm given that:
    1. Two identical rectangular sheets each have an area of 48 square centimeters,
    2. The covered area when overlapping the sheets is 72 square centimeters,
    3. The diagonal BD of the overlapping quadrilateral ABCD is 6 centimeters. -/
theorem length_of_rectangular_sheet :
  ∀ (length width : ℝ),
    width * length = 48 ∧
    2 * 48 - 72 = width * 6 ∧
    width * 6 = 24 →
    length = 10 :=
sorry

end length_of_rectangular_sheet_l811_811323


namespace expected_number_of_different_faces_l811_811439

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l811_811439


namespace sum_gcf_lcm_l811_811357

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l811_811357


namespace interest_rate_per_annum_is_four_l811_811458

-- Definitions
def P : ℕ := 300
def t : ℕ := 8
def I : ℤ := P - 204

-- Interest formula
def simple_interest (P : ℕ) (r : ℕ) (t : ℕ) : ℤ := P * r * t / 100

-- Statement to prove
theorem interest_rate_per_annum_is_four :
  ∃ r : ℕ, I = simple_interest P r t ∧ r = 4 :=
by sorry

end interest_rate_per_annum_is_four_l811_811458


namespace log_base_fraction_eq_l811_811980

theorem log_base_fraction_eq (x : ℝ) : (1 / 4) ^ x = 16 → x = -2 :=
by
  sorry

end log_base_fraction_eq_l811_811980


namespace expected_number_of_different_faces_l811_811433

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l811_811433


namespace number_of_distinct_prime_factors_of_90_l811_811108

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811108


namespace tiles_per_row_l811_811270

def area_square (s : ℝ) : ℝ := s * s
def area_tile (l : ℝ) : ℝ := l * l
def inch_to_feet (x : ℝ) : ℝ := x / 12
def feet_to_inch (x : ℝ) : ℝ := x * 12

theorem tiles_per_row (room_area : ℝ) (tile_length_inch : ℝ) :
  room_area = 144 ∧ tile_length_inch = 8 →
  let room_side_feet := Real.sqrt room_area in
  let room_side_inch := feet_to_inch room_side_feet in
  room_side_inch / tile_length_inch = 18 := 
by {
  sorry
}

end tiles_per_row_l811_811270


namespace R_and_D_expense_corresponding_to_productivity_increase_l811_811936

/-- Given values for R&D expenses and increase in average labor productivity -/
def R_and_D_t : ℝ := 2640.92
def Delta_APL_t_plus_2 : ℝ := 0.81

/-- Statement to be proved: the R&D expense in million rubles corresponding 
    to an increase in average labor productivity by 1 million rubles per person -/
theorem R_and_D_expense_corresponding_to_productivity_increase : 
  R_and_D_t / Delta_APL_t_plus_2 = 3260 := 
by
  sorry

end R_and_D_expense_corresponding_to_productivity_increase_l811_811936


namespace distinct_real_roots_iff_l811_811624

theorem distinct_real_roots_iff (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x, x^2 + 3 * x - a = 0 → (x = x₁ ∨ x = x₂))) ↔ a > - (9 : ℝ) / 4 :=
sorry

end distinct_real_roots_iff_l811_811624


namespace Tim_paid_amount_l811_811314

theorem Tim_paid_amount (original_price : ℝ) (discount_percentage : ℝ) (discounted_price : ℝ) 
    (h1 : original_price = 1200) (h2 : discount_percentage = 0.15) 
    (discount_amount : ℝ) (h3 : discount_amount = original_price * discount_percentage) 
    (h4 : discounted_price = original_price - discount_amount) : discounted_price = 1020 := 
    by {
        sorry
    }

end Tim_paid_amount_l811_811314


namespace anton_thought_of_729_l811_811868

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l811_811868


namespace splitting_cost_of_new_apartment_l811_811217

def john_old_apartment_cost_per_month : ℕ := 1200
def apartment_price_increase_percentage : ℕ := 40
def john_and_siblings_count : ℕ := 3

theorem splitting_cost_of_new_apartment:
  let new_apartment_cost := john_old_apartment_cost_per_month * (1 + apartment_price_increase_percentage / 100) in
  john_and_siblings_count = 3 :=
by
  sorry

end splitting_cost_of_new_apartment_l811_811217


namespace greatest_y_value_l811_811266

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end greatest_y_value_l811_811266


namespace distinct_prime_factors_of_90_l811_811087

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811087


namespace maximum_omega_l811_811594

noncomputable def f (ω varphi : ℝ) (x : ℝ) : ℝ :=
  sin (ω * x + varphi)

theorem maximum_omega {ω : ℝ} (hω : ω > 0) {varphi : ℝ}
  (hvarphi : |varphi| ≤ π / 2)
  (hzero : f ω varphi (-π / 4) = 0)
  (hsymmetry : ∀ x, f ω varphi (π / 4 - x) = f ω varphi (π / 4 + x))
  (hmonotonic : monotone_on (f ω varphi) (Set.Ioo (π / 18) (5 * π / 36))) :
  ω = 9 :=
  sorry

end maximum_omega_l811_811594


namespace intersection_with_y_axis_l811_811756

theorem intersection_with_y_axis (x y : ℝ) (h : y = x + 3) (hx : x = 0) : (x, y) = (0, 3) := 
by 
  subst hx 
  rw [h]
  rfl
-- sorry to skip the proof

end intersection_with_y_axis_l811_811756


namespace uncle_taller_than_james_l811_811655

def james_initial_height (uncle_height : ℕ) : ℕ := (2 * uncle_height) / 3

def james_final_height (initial_height : ℕ) (growth_spurt : ℕ) : ℕ := initial_height + growth_spurt

theorem uncle_taller_than_james (uncle_height : ℕ) (growth_spurt : ℕ) :
  uncle_height = 72 →
  growth_spurt = 10 →
  uncle_height - (james_final_height (james_initial_height uncle_height) growth_spurt) = 14 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end uncle_taller_than_james_l811_811655


namespace total_eggs_sold_l811_811608

def initial_trays : Nat := 10
def dropped_trays : Nat := 2
def added_trays : Nat := 7
def eggs_per_tray : Nat := 36

theorem total_eggs_sold : initial_trays - dropped_trays + added_trays * eggs_per_tray = 540 := by
  sorry

end total_eggs_sold_l811_811608


namespace divisible_323_even_n_l811_811556

theorem divisible_323_even_n (n : ℤ) (h_even : 2 ∣ n) : 323 ∣ (20^n + 16^n - 3^n - 1) := 
by {
  sorry
}

end divisible_323_even_n_l811_811556


namespace tourist_groups_meet_l811_811785

theorem tourist_groups_meet (x y : ℝ) (h1 : 4.5 * x + 2.5 * y = 30) (h2 : 3 * x + 5 * y = 30) : 
  x = 5 ∧ y = 3 := 
sorry

end tourist_groups_meet_l811_811785


namespace factorize_expression_l811_811983

theorem factorize_expression (x : ℝ) : x^3 - 2 * x^2 + x = x * (x - 1)^2 :=
by sorry

end factorize_expression_l811_811983


namespace math_problem_l811_811664

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => if a n < 2 * n then a n + 1 else a n

theorem math_problem (n : ℕ) (hn : n > 0) (ha_inc : ∀ m, m > 0 → a m < a (m + 1)) 
  (ha_rec : ∀ m, m > 0 → a (m + 1) ≤ 2 * m) : 
  ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ n = a p - a q := sorry

end math_problem_l811_811664


namespace women_attended_gathering_l811_811492

theorem women_attended_gathering :
  ∀ (m : ℕ) (w_per_man : ℕ) (m_per_woman : ℕ),
  m = 15 ∧ w_per_man = 4 ∧ m_per_woman = 3 →
  ∃ (w : ℕ), w = 20 :=
by
  intros m w_per_man m_per_woman h,
  cases h with hm hw_wom,
  cases hw_wom with hwm hmw,
  sorry

end women_attended_gathering_l811_811492


namespace find_BK_BM_l811_811721

-- Defining necessary points and conditions
variables {A B C D A1 C1 M K P : Type} 
constants (DC DA m n a c x : ℝ)
constants (BK BM : ℝ) 
constants (isInscribed : ∀(T: Type), T) -- Dummy function to denote the quadrilateral is inscribed in a circle
constants (intersectBD : (A1 × M) ∩ (C1 × K) = P) -- Intersection at diagonal BD

-- Defining the equalities and properties
axiom DC_eq_m : DC = m
axiom DA_eq_n : DA = n

-- Main statement to prove
theorem find_BK_BM :
    BK = (a * c * (n - m)) / (c * n - a * m) ∧ BM = (a * c * (n - m)) / (c * n - a * m) :=
sorry

end find_BK_BM_l811_811721


namespace R_and_D_expense_corresponding_to_productivity_increase_l811_811937

/-- Given values for R&D expenses and increase in average labor productivity -/
def R_and_D_t : ℝ := 2640.92
def Delta_APL_t_plus_2 : ℝ := 0.81

/-- Statement to be proved: the R&D expense in million rubles corresponding 
    to an increase in average labor productivity by 1 million rubles per person -/
theorem R_and_D_expense_corresponding_to_productivity_increase : 
  R_and_D_t / Delta_APL_t_plus_2 = 3260 := 
by
  sorry

end R_and_D_expense_corresponding_to_productivity_increase_l811_811937


namespace sum_gcd_lcm_eight_twelve_l811_811382

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l811_811382


namespace negation_of_p_correct_l811_811061

def p := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p_correct :
  (¬ p) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end negation_of_p_correct_l811_811061


namespace sqrt_of_factorial_div_l811_811999

theorem sqrt_of_factorial_div : 
  √(9! / 126) = 8 * √5 :=
by
  sorry

end sqrt_of_factorial_div_l811_811999


namespace vector_subtraction_l811_811009

variable (x : ℝ)

def a : ℝ × ℝ := (2, -x)
def b : ℝ × ℝ := (-1, 3)

theorem vector_subtraction (h : (2 * -1) + (-x * 3) = 4) : 
  a - 2 * b = (4, -4) := sorry

end vector_subtraction_l811_811009


namespace probability_open_lock_l811_811039

/-- Given 5 keys and only 2 can open the lock, the probability of opening the lock by selecting one key randomly is 0.4. -/
theorem probability_open_lock (k : Finset ℕ) (h₁ : k.card = 5) (s : Finset ℕ) (h₂ : s.card = 2 ∧ s ⊆ k) :
  ∃ p : ℚ, p = 0.4 :=
by
  sorry

end probability_open_lock_l811_811039


namespace classical_cd_even_l811_811713

theorem classical_cd_even (C : ℕ) : 
  14 + C + 8 = 22 + C →
  (∀ k : ℕ, k ≥ 1 → k ≤ 2 → 2 ∣ k) →
  (∃ C : ℕ, 2 ∣ C) :=
by
  intros ht hdiv
  have hC := eq.trans (by ring) ht
  sorry

end classical_cd_even_l811_811713


namespace negation_of_prop_p_l811_811026

theorem negation_of_prop_p : (¬ (∀ x : ℝ, x > 0 → 2^x > 1)) ↔ (∃ x : ℝ, x > 0 ∧ 2^x ≤ 1) :=
by
  sorry

end negation_of_prop_p_l811_811026


namespace log_fraction_pow_l811_811965

theorem log_fraction_pow : log (1 / 4) 16 = -2 :=
by {
    sorry
}

end log_fraction_pow_l811_811965


namespace dot_product_result_l811_811068

open Real

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, 2)

def scale_vec (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add_vec (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_result :
  dot_product (add_vec (scale_vec 2 a) b) a = 6 :=
by
  sorry

end dot_product_result_l811_811068


namespace geom_arith_seq_sum_zero_l811_811678

theorem geom_arith_seq_sum_zero (a b : ℝ) 
  (S : ℕ → ℝ) (T : ℕ → ℝ)
  (h₁ : ∀ n, S n = 2^n + a)
  (h₂ : ∀ n, T n = n^2 - 2n + b)
  (hS0 : S 0 = 0)
  (hT0 : T 0 = 0) : 
  a + b = 0 :=
by sorry

end geom_arith_seq_sum_zero_l811_811678


namespace intersection_with_y_axis_is_03_l811_811754

-- Define the line equation
def line (x : ℝ) : ℝ := x + 3

-- The intersection point with the y-axis, i.e., where x = 0
def y_axis_intersection : Prod ℝ ℝ := (0, line 0)

-- Prove that the intersection point is (0, 3)
theorem intersection_with_y_axis_is_03 : y_axis_intersection = (0, 3) :=
by
  simp [y_axis_intersection, line]
  sorry

end intersection_with_y_axis_is_03_l811_811754


namespace distinct_prime_factors_of_90_l811_811097

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811097


namespace no_unique_solution_for_fractions_l811_811789

theorem no_unique_solution_for_fractions 
  (initial_intensity : ℝ) (final_intensity : ℝ) 
  (fraction_step1 : ℝ) (fraction_step2 : ℝ) 
  (fraction_step3 : ℝ) (fraction_step4 : ℝ) 
  (fraction_step5 : ℝ) 
  (intensity1 : ℝ) (intensity2 : ℝ) 
  (intensity3 : ℝ) (intensity4 : ℝ) 
  (intensity5 : ℝ) : 
  initial_intensity = 0.6 → 
  intensity1 = 0.25 → 
  intensity2 = 0.45 → 
  intensity3 = 0.3  → 
  intensity4 = 0.55 → 
  intensity5 = 0.2  → 
  final_intensity = 0.4 →
  ¬(∃ steps, 
    steps = [fraction_step1, fraction_step2, fraction_step3, fraction_step4, fraction_step5] ∧ 
    (0.5 * 0.6 + 0.5 * 0.25 = 0.425 ∨ -- Example intermediate step calculation
     -- Other possible intermediate steps
    ) ∧ 
    -- Final mixture intensity without exact constraints
    true
  ) :=
by {
  intros h_init h_int1 h_int2 h_int3 h_int4 h_int5 h_final,
  sorry
}

end no_unique_solution_for_fractions_l811_811789


namespace greg_has_14_more_marbles_than_adam_l811_811852

theorem greg_has_14_more_marbles_than_adam (adam_marbles greg_marbles : ℕ) 
    (h_adam : adam_marbles = 29) (h_greg : greg_marbles = 43) : 
    greg_marbles - adam_marbles = 14 :=
by
  rw [h_adam, h_greg]
  exact Nat.sub_self 29 43 sorry

end greg_has_14_more_marbles_than_adam_l811_811852


namespace solve_problem_l811_811038

open Real

-- Definitions of lines and conditions
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 3 = 0
def line2 (x y m : ℝ) : Prop := 6 * x + m * y + 14 = 0
def lines_parallel (m : ℝ) : Prop := ∃ k : ℝ, 6 = k * 3 ∧ m = k * 4

theorem solve_problem (m : ℝ) : (lines_parallel m) → m = 8 ∧ ∀ (x y : ℝ), 
(line1 x y → line2 x y 8 → distance (x, y) (0, (-3-7) / sqrt(3^2 + 4^2)) = 2) :=
by
  sorry

end solve_problem_l811_811038


namespace find_postal_codes_l811_811299

-- Define the postal codes A, B, C, and D
def A := [3,2,0,6,5,1]
def B := [1,0,5,2,6,3]
def C := [6,1,2,3,0,5]
def D := [3,1,6,2,5,0]

-- Define a function to count matching digits given positions
def matching_digits (M N : list ℕ) (A : list ℕ) : ℕ :=
  (list.zip_with (λ m a, if m = a then 1 else 0) M A).sum

-- Prove the postal codes M and N
theorem find_postal_codes :
  ∃ (M N : list ℕ), M = [6,1,0,2,5,3] ∨ M = [3,1,0,2,6,5] ∧
    matching_digits M N A = 2 ∧
    matching_digits M N B = 2 ∧
    matching_digits M N C = 2 ∧
    matching_digits M N D = 3 :=
by
  admit  -- Proof omitted

end find_postal_codes_l811_811299


namespace prime_pairs_distinct_not_divisors_l811_811668

theorem prime_pairs_distinct_not_divisors (n : ℕ) (h : n > 0)
  (pairing : ∃ (pairs : list (ℕ × ℕ)), (∀ (d : ℕ), d ∣ n → d > 0 → 
    ∃ (pair : ℕ × ℕ), pair ∈ pairs ∧ (d = pair.fst ∨ d = pair.snd)) ∧
    (∀ (pair : ℕ × ℕ), pair ∈ pairs → nat.prime (pair.fst + pair.snd))) :
  (∀ (pair1 pair2 : ℕ × ℕ), pair1 ∈ pairing.some → pair2 ∈ pairing.some → pair1 ≠ pair2 → 
    pair1.fst + pair1.snd ≠ pair2.fst + pair2.snd) ∧
  (∀ (pair : ℕ × ℕ), pair ∈ pairing.some → ¬(pair.fst + pair.snd ∣ n)) :=
by sorry

end prime_pairs_distinct_not_divisors_l811_811668


namespace sum_gcd_lcm_eight_twelve_l811_811377

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l811_811377


namespace area_of_triangle_PQR_l811_811511

-- Define the square pyramid with its dimensions
structure Pyramid :=
  (side_length : ℝ)
  (altitude : ℝ)

-- Define the points on the edges of the pyramid
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the specific pyramid with side length and altitude
def pyramid : Pyramid :=
  { side_length := 4, altitude := 8 }

-- Define points P, Q, and R based on the given conditions
def point_P : Point :=
  { x := 0, y := 0, z := (4 * Real.sqrt 5) / 4 }

def point_Q : Point :=
  { x := 0, y := 4, z := (4 * Real.sqrt 5) / 4 }

def point_R : Point :=
  { x := 4, y := 0, z := (3 * 4 * Real.sqrt 5) / 4 }

-- Theorem stating the problem
theorem area_of_triangle_PQR : 
  let area_PQR := 5 in 
  area_PQR = 5 := 
sorry

end area_of_triangle_PQR_l811_811511


namespace distinct_prime_factors_count_l811_811121

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811121


namespace smallest_positive_period_l811_811543

theorem smallest_positive_period :
  ∀ x, ∃ T > 0, ∀ x, y = (sin x + cos x) * (sin x - cos x) → y(x + T) = y x ∧ ∀ T' > 0, T' < T → ∃ x, y(x + T') ≠ y x := 
begin
  sorry
end

end smallest_positive_period_l811_811543


namespace complex_sum_inequality_l811_811224

theorem complex_sum_inequality 
  (n : ℕ) 
  (z w : Fin n → ℂ) 
  (h : ∀ (ε : Fin n → {x // x = -1 ∨ x = 1}), 
    abs (∑ i, ε i * z i) ≤ abs (∑ i, ε i * w i)) : 
  (∑ i, abs (z i) ^ 2) ≤ (∑ i, abs (w i) ^ 2) := 
sorry

end complex_sum_inequality_l811_811224


namespace anton_thought_number_is_729_l811_811889

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l811_811889


namespace simplification_and_sum_l811_811766

theorem simplification_and_sum :
  (∃ A B C D : ℤ,
    (∀ x : ℝ, x ≠ D → (x^3 + 9 * x^2 + 26 * x + 24) / (x + 3) = A * x^2 + B * x + C) ∧
    (D = -3) ∧
    (A + B + C + D = 12)) :=
begin
  use [1, 6, 8, -3],
  split,
  { intros x h,
    have h₁ : x + 3 ≠ 0 := ne_of_ne_neg_of_eq h,
    rw [←mul_div_cancel' (x^2 + 6 * x + 8) h₁, ←div_eq_mul_one_div],
    simp [mul_comm, div_one],
  },
  split,
  exact rfl,
  exact rfl,
end

end simplification_and_sum_l811_811766


namespace quadrilateral_rhombus_l811_811462

variables {A B C D O : Type*}
variables [metric_space A B C D O] [nonempty_space O]

def circumscribed_around_circle (ABCD : quadrilateral A B C D) (O : point O) := 
  circle_is_inscribed (circle O) ABCD

def diagonals_intersect_at_center (ABCD : quadrilateral A B C D) (O : point O) :=
  intersection_of_diagonals ABCD O

theorem quadrilateral_rhombus
  (ABCD : quadrilateral A B C D)
  (O : point O)
  (circum_circle : circumscribed_around_circle ABCD O)
  (diag_intersect : diagonals_intersect_at_center ABCD O) :
  is_rhombus ABCD :=
sorry

end quadrilateral_rhombus_l811_811462


namespace correct_operation_l811_811407

-- Define the problem as a theorem in Lean
theorem correct_operation (a : ℝ) : 3 * a^2 * 2 * a^2 = 6 * a^4 := 
by {
    Calc
    sorry
}

end correct_operation_l811_811407


namespace Anton_thought_of_729_l811_811913

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l811_811913


namespace uncle_fyodor_possible_l811_811551

theorem uncle_fyodor_possible (x y : ℕ) :
  (x * y % 10 = 8) → ((x + y) % 10 = 9) ∧ ((x * y + 12) % 10 = 0) :=
by
  intros h1
  split
  -- Prove (x + y) % 10 = 9 based on given condition x * y ≡ 8 (mod 10)
  {
    have : (x * y % 10 + x + y + 1) % 10 = x * y % 10,
    { rw h1 },
    have : (8 + x + y + 1) % 10 = 8,
    { rw h1 },
    have : (x + y + 1) % 10 = 0,
    { -- Simplify (8 + x + y + 1) % 10 = 8 to (x + y + 1) % 10 = 0
      exact (Nat.add_mod_cancel_left x y 8 1).mp this.symm },
    exact Nat.add_mod_cancel_left x y 1 10 this,
  },
  { 
    exact h1 
  } 

  sorry -- Moreover prove (x * y + 12) % 10 = 0


end uncle_fyodor_possible_l811_811551


namespace people_eating_both_l811_811194

theorem people_eating_both (only_veg : ℕ) (total_veg : ℕ) : 
  (∃ both : ℕ, only_veg + both = total_veg) → (∃ both : ℕ, both = 8) :=
by
  -- Given conditions in the problem
  assume h1 : 13 = only_veg,
  assume h2 : 21 = total_veg,
  -- Translate given condition ∃ both : ℕ, only_veg + both = total_veg to Lean
  assume h3 : ∃ both : ℕ, only_veg + both = total_veg,
  -- Now we use the condition h1 and h2 to find the number of people who eat both
  cases h3 with both hboth,
  use 8,
  suffices h4 : 13 + 8 = 21,
  exact h4,
  exact Eq.trans (Eq.symm h1) (Eq.trans hboth h2)

# The statement is "If there exists a 'both' such that only_veg + both = total_veg, then 'both' must be 8, given that 13 = only_veg and 21 = total_veg."

end people_eating_both_l811_811194


namespace women_attended_gathering_l811_811491

theorem women_attended_gathering :
  ∀ (m : ℕ) (w_per_man : ℕ) (m_per_woman : ℕ),
  m = 15 ∧ w_per_man = 4 ∧ m_per_woman = 3 →
  ∃ (w : ℕ), w = 20 :=
by
  intros m w_per_man m_per_woman h,
  cases h with hm hw_wom,
  cases hw_wom with hwm hmw,
  sorry

end women_attended_gathering_l811_811491


namespace gcd_lcm_sum_8_12_l811_811393

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l811_811393


namespace Brian_Frodo_ratio_l811_811932

-- Definitions from the conditions
def Lily_tennis_balls : Int := 3
def Frodo_tennis_balls : Int := Lily_tennis_balls + 8
def Brian_tennis_balls : Int := 22

-- The proof statement
theorem Brian_Frodo_ratio :
  Brian_tennis_balls / Frodo_tennis_balls = 2 := by
  sorry

end Brian_Frodo_ratio_l811_811932


namespace sum_gcd_lcm_eight_twelve_l811_811383

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l811_811383


namespace value_of_expression_l811_811287

theorem value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 10) : (12 * y - 4)^2 = 80 := 
by 
  sorry

end value_of_expression_l811_811287


namespace a_finishes_race_in_t_seconds_l811_811628

theorem a_finishes_race_in_t_seconds 
  (time_B : ℝ := 45)
  (dist_B : ℝ := 100)
  (dist_A_wins_by : ℝ := 20)
  (total_dist : ℝ := 100)
  : ∃ t : ℝ, t = 36 := 
  sorry

end a_finishes_race_in_t_seconds_l811_811628


namespace sum_of_gcd_and_lcm_is_28_l811_811389

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l811_811389


namespace usual_time_to_cover_distance_l811_811820

theorem usual_time_to_cover_distance (S T : ℝ) (h : T > 0) : (0.5 * S) / S = (T + 24) / T → T = 24 :=
by
  intro h_eq
  have h1 : 0.5 = (T + 24) / T, by rw [div_mul_cancel _ (ne_of_gt h)] at h_eq; assumption
  have h2 : 2 * T = T + 24, by linarith
  linarith

end usual_time_to_cover_distance_l811_811820


namespace trigonometry_problem_l811_811184

theorem trigonometry_problem (α : ℝ) (b : ℝ) : 
  (cos α = -3 / 5 ∧ ∃ y, y = b ∧ y^2 = (4 : ℝ)^2) → 
  b = 4 ∨ b = -4 ∧ (sin α = 4 / 5 ∨ sin α = -4 / 5) := 
sorry

end trigonometry_problem_l811_811184


namespace max_value_trig_expression_l811_811227

theorem max_value_trig_expression (a b c : ℝ) (h : ∃ θ : ℝ, cos θ ≠ 0) :
  ∃ θ : ℝ, a * cos θ + b * sin θ + c * tan θ = sqrt (a^2 + b^2) + c * (b / a) := 
by
  sorry -- Proof to be filled in

end max_value_trig_expression_l811_811227


namespace num_distinct_prime_factors_90_l811_811128

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811128


namespace problem_statement_l811_811596

-- Define the function f and its derivative
def f (m x : ℝ) := (1/2) * m * x^2 + (Real.log x) - 2 * x
def f' (m x : ℝ) := m * x + 1 / x - 2

-- Define the condition that f is increasing
def is_increasing (m : ℝ) := ∀ x : ℝ, x > 0 → f' m x ≥ 0

-- The theorem statement
theorem problem_statement (m : ℝ) : is_increasing m → m ≥ 1 :=
sorry

end problem_statement_l811_811596


namespace raccoon_carrots_hid_l811_811214

theorem raccoon_carrots_hid 
  (r : ℕ)
  (b : ℕ)
  (h1 : 5 * r = 8 * b)
  (h2 : b = r - 3) 
  : 5 * r = 40 :=
by
  sorry

end raccoon_carrots_hid_l811_811214


namespace Poncelets_theorem_l811_811326

-- Define the circles α and β
variables (α β : Circle)

-- Define the existence of one n-gon inscribed in α and circumscribed around β
def exists_ngon_inscribed (n : ℕ) (α β : Circle) : Prop :=
  ∃ (A : Fin n → Point), 
    (∀ i, A i ∈ α) ∧ 
    (∀ i, tangent (segment (A i) (A ((i + 1) % n))) β)

-- Define the Poncelet's theorem statement in Lean 4
theorem Poncelets_theorem (n : ℕ) (h : exists_ngon_inscribed n α β) : 
  ∃ (T : Point) (A' : Fin n → Point), 
    A' 0 = T ∧ 
    (∀ i, A' i ∈ α) ∧ 
    (∀ i, tangent (segment (A' i) (A' ((i + 1) % n))) β) :=
begin
  sorry
end

end Poncelets_theorem_l811_811326


namespace average_of_xyz_l811_811614

variable (x y z : ℝ)

theorem average_of_xyz (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 := by
  sorry

end average_of_xyz_l811_811614


namespace intersection_with_y_axis_is_03_l811_811755

-- Define the line equation
def line (x : ℝ) : ℝ := x + 3

-- The intersection point with the y-axis, i.e., where x = 0
def y_axis_intersection : Prod ℝ ℝ := (0, line 0)

-- Prove that the intersection point is (0, 3)
theorem intersection_with_y_axis_is_03 : y_axis_intersection = (0, 3) :=
by
  simp [y_axis_intersection, line]
  sorry

end intersection_with_y_axis_is_03_l811_811755


namespace greatest_difference_in_baskets_l811_811308

theorem greatest_difference_in_baskets :
  let A_red := 4
  let A_yellow := 2
  let B_green := 6
  let B_yellow := 1
  let C_white := 3
  let C_yellow := 9
  max (abs (A_red - A_yellow)) (max (abs (B_green - B_yellow)) (abs (C_white - C_yellow))) = 6 :=
by
  sorry

end greatest_difference_in_baskets_l811_811308


namespace quadratic_completion_l811_811735

noncomputable def sum_of_r_s (r s : ℝ) : ℝ := r + s

theorem quadratic_completion (x r s : ℝ) (h : 16 * x^2 - 64 * x - 144 = 0) :
  ((x + r)^2 = s) → sum_of_r_s r s = -7 :=
by
  sorry

end quadratic_completion_l811_811735


namespace anton_thought_number_l811_811922

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l811_811922


namespace distinct_prime_factors_of_90_l811_811101

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811101


namespace cos_beta_value_l811_811030

theorem cos_beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : tan α = 2) (h4 : sin (α + β) = (√2) / 2) :
  cos β = (√10) / 10 :=
sorry

end cos_beta_value_l811_811030


namespace anton_thought_number_l811_811882

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l811_811882


namespace areaABDE_inRegularHexagon_l811_811222

-- Definition of a regular hexagon with side length 2
def isRegularHexagon (A B C D E F : ℝ × ℝ) :=
  let sideLength := 2
  (dist A B = sideLength) ∧
  (dist B C = sideLength) ∧
  (dist C D = sideLength) ∧
  (dist D E = sideLength) ∧
  (dist E F = sideLength) ∧
  (dist F A = sideLength)

-- Coordinates of the vertices for reference
def vertexA : ℝ × ℝ := (2, 0)
def vertexB : ℝ × ℝ := (1, Real.sqrt 3)
def vertexD : ℝ × ℝ := (-2, 0)
def vertexE : ℝ × ℝ := (-1, -Real.sqrt 3)

-- Prove area of quadrilateral ABDE
theorem areaABDE_inRegularHexagon : 
  isRegularHexagon vertexA vertexB (1, -Real.sqrt 3) vertexD vertexE vertexA →
  (dist vertexA vertexE) = 4 -> -- This helps to assert vertices are in correct parallelogram configuration
  (dist vertexB vertexD) = 4 ->
  (dist vertexA vertexB) = 2 -> 
  ∃ base height : ℝ, 
  base = 2 ∧ 
  height = 2 * Real.sqrt 3 ∧  
  1/2 * base * height = 4 * Real.sqrt 3 :=
begin
  sorry
end

end areaABDE_inRegularHexagon_l811_811222


namespace area_ABC_is_9n_l811_811204

variables {ABC : Type*} [triangle ABC]
variables {A B C M N O P Q : ABC}
variable (n : ℝ)

-- Defining the conditions
axiom is_isosceles : AB = AC
axiom M_is_midpoint_BC : midpoint M B C
axiom N_is_midpoint_AB : midpoint N A B
axiom P_is_midpoint_BC : midpoint P B C
axiom AM_intersects_CN_at_O : ∃ O, A ≠ O ∧ N ≠ O ∧ B ≠ O ∧ C ≠ O ∧ M ≠ O
axiom area_OMQ_is_n : area ⟨O, M, Q⟩ = n

-- The theorem we need to prove
theorem area_ABC_is_9n :
  area ⟨A, B, C⟩ = 9 * n :=
sorry

end area_ABC_is_9n_l811_811204


namespace maximize_profit_l811_811425

noncomputable def revenue (p : ℝ) : ℝ := p * (150 - 6 * p)

def fixed_cost : ℝ := 200

noncomputable def profit (p : ℝ) : ℝ := revenue(p) - fixed_cost

theorem maximize_profit : ∃ (p : ℝ), p = 12.5 ∧ ∀ p' ≤ 30, profit(p') ≤ profit(12.5) := by
  sorry

end maximize_profit_l811_811425


namespace sandwich_cost_l811_811341

-- Defining the cost of each sandwich and the known conditions
variable (S : ℕ) -- Cost of each sandwich in dollars

-- Conditions as hypotheses
def buys_three_sandwiches (S : ℕ) : ℕ := 3 * S
def buys_two_drinks (drink_cost : ℕ) : ℕ := 2 * drink_cost
def total_cost (sandwich_cost drink_cost total_amount : ℕ) : Prop := buys_three_sandwiches sandwich_cost + buys_two_drinks drink_cost = total_amount

-- Given conditions in the problem
def given_conditions : Prop :=
  (buys_two_drinks 4 = 8) ∧ -- Each drink costs $4
  (total_cost S 4 26)       -- Total spending is $26

-- Theorem to prove the cost of each sandwich
theorem sandwich_cost : given_conditions S → S = 6 :=
by sorry

end sandwich_cost_l811_811341


namespace a2016_plus_a3_l811_811603

-- Definitions associated with the conditions
def sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  a 1 = 1 ∧
  (∀ n, a (n + 2) = 1 / (a n + 1)) ∧
  a 6 = a 2

-- The main theorem to be proven
theorem a2016_plus_a3 (a : ℕ → ℝ) (h : sequence a) : a 2016 + a 3 = (Real.sqrt 5) / 2 :=
sorry

end a2016_plus_a3_l811_811603


namespace dot_product_problem_l811_811173

variables {a b : ℝ^3}
hypothesis ha : ∥a∥ = 3
hypothesis hb : ∥b∥ = 6

theorem dot_product_problem : (a + b) • (a - b) = -27 := by
  sorry

end dot_product_problem_l811_811173


namespace unique_line_through_point_with_intercept_condition_l811_811771

def point : ℝ × ℝ := (1, 4)

def line_eq (m b : ℝ) : ℝ → ℝ := λ x, m * x + b

def intercept_condition (m b : ℝ) : Prop := abs m = abs b

def passes_through_point (m b : ℝ) : Prop := line_eq m b (point.fst) = point.snd

theorem unique_line_through_point_with_intercept_condition :
  ∃! m b : ℝ, intercept_condition m b ∧ passes_through_point m b :=
sorry

end unique_line_through_point_with_intercept_condition_l811_811771


namespace sum_of_cubes_of_consecutive_integers_div_by_9_l811_811254

theorem sum_of_cubes_of_consecutive_integers_div_by_9 (x : ℤ) : 
  let a := (x - 1) ^ 3
  let b := x ^ 3
  let c := (x + 1) ^ 3
  (a + b + c) % 9 = 0 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_div_by_9_l811_811254


namespace min_cost_at_100_l811_811431

noncomputable def cost_function (v : ℝ) : ℝ :=
if (0 < v ∧ v ≤ 50) then (123000 / v + 690)
else if (v > 50) then (3 * v^2 / 50 + 120000 / v + 600)
else 0

theorem min_cost_at_100 : ∃ v : ℝ, v = 100 ∧ cost_function v = 2400 :=
by
  -- We are not proving but stating the theorem here
  sorry

end min_cost_at_100_l811_811431


namespace perimeter_of_triangle_ABC_l811_811643

-- Given the conditions of the problem
def angle_ABC_equals_angle_ACB (ABC ACB : Prop) := ABC = ACB
def side_BC_equals_eight  (BC : ℕ) := BC = 8
def side_AB_equals_ten  (AB : ℕ) := AB = 10
def is_isosceles (ABC_ACB_eq : Prop) (AB AC : ℕ) := ABC_ACB_eq → AB = AC

-- The theorem about the perimeter
theorem perimeter_of_triangle_ABC
  (ABC ACB : Prop)
  (BC AB AC perimeter : ℕ)
  (h1 : angle_ABC_equals_angle_ACB ABC ACB)
  (h2 : side_BC_equals_eight BC)
  (h3 : side_AB_equals_ten AB)
  (h4 : is_isosceles h1 AB AC) :
  perimeter = AB + BC + AC :=
begin
  -- Dummy proof placeholder
  sorry,
end

end perimeter_of_triangle_ABC_l811_811643


namespace num_distinct_prime_factors_90_l811_811132

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811132


namespace function_discontinuities_must_not_meet_condition_l811_811232

variables {a b : ℝ} (f : ℝ → ℝ)

theorem function_discontinuities_must_not_meet_condition (h : ∀ x ∈ Ioo a b, ∃ ε > 0, (∀ y ∈ Ioo (x - ε) (x + ε), f y = f x)) :
  ¬ (∃ N : ℕ, { x ∈ Ioo a b | ¬ continuous_at f x }.finite) :=
sorry

end function_discontinuities_must_not_meet_condition_l811_811232


namespace hexagon_area_correct_l811_811671

noncomputable theory

def area_of_hexagon :=
  let AB := 13
  let BC := 7
  let CD := 23
  let DA := 9
  let trapezoid_altitude := 6  -- Approximation of 2x from solution
  let area_trapezoid := (AB + CD) * trapezoid_altitude / 2
  let area_triangle_ADP := DA * trapezoid_altitude / 2
  let area_triangle_BCQ := BC * trapezoid_altitude / 2
  area_trapezoid - area_triangle_ADP - area_triangle_BCQ

theorem hexagon_area_correct : area_of_hexagon = 85.68 := by
  sorry

end hexagon_area_correct_l811_811671


namespace average_rate_second_drive_l811_811862

theorem average_rate_second_drive 
 (distance : ℕ) (total_time : ℕ) (d1 d2 d3 : ℕ)
 (t1 t2 t3 : ℕ) (r1 r2 r3 : ℕ)
 (h_distance : d1 = d2 ∧ d2 = d3 ∧ d1 + d2 + d3 = distance)
 (h_total_time : t1 + t2 + t3 = total_time)
 (h_drive_1 : r1 = 4 ∧ t1 = d1 / r1)
 (h_drive_2 : r3 = 6 ∧ t3 = d3 / r3)
 (h_distance_total : distance = 180)
 (h_total_time_val : total_time = 37)
  : r2 = 5 := 
by sorry

end average_rate_second_drive_l811_811862


namespace is_integer_bibinomial_coefficient_l811_811552

def double_factorial : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 1) := if (n + 1) % 2 = 0 then (n + 1) * double_factorial (n - 1) else (n + 1) * double_factorial (n - 1)

def bibinomial_coefficient (n k : ℕ) : ℕ :=
  double_factorial n / (double_factorial k * double_factorial (n - k))

theorem is_integer_bibinomial_coefficient (n k : ℕ) (h : n ≥ k) :
  ∃ (c : ℤ), bibinomial_coefficient n k = c ↔ 
    k = 0 ∨ k = n ∨ (n % 2 = 0 ∧ k % 2 = 0) ∨ (n = 2 ∧ k = 1) :=
by
  sorry

end is_integer_bibinomial_coefficient_l811_811552


namespace max_distance_ellipse_to_line_l811_811571

open Real

theorem max_distance_ellipse_to_line :
  ∃ (M : ℝ × ℝ),
    (M.1 ^ 2 / 12 + M.2 ^ 2 / 4 = 1) ∧
    (∃ (d : ℝ), 
      (d = abs (2 * sqrt 3 * cos (3 * π / 2 - π / 3) + 2 * sin (3 * π / 2 - π / 3) - 4) / sqrt 2) ∧ 
      d = 4 * sqrt 2 ∧ M = (-3, -1)) :=
sorry

end max_distance_ellipse_to_line_l811_811571


namespace no_such_n_exists_l811_811278

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, (∀ d : ℕ, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ↔
    (∃ c : ℕ, ∀ d' : ℕ, d' ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
    digit_count d' (concatenated_natural_numbers_up_to n) = c)) := 
sorry

def concatenated_natural_numbers_up_to (n : ℕ) : string :=
  (List.range' 1 n).map (λ x => x.toString).foldl (++) ""

def digit_count (d : ℕ) (s : string) : ℕ :=
  s.data.count (λ ch => ch.toString.toNat = some d)

end no_such_n_exists_l811_811278


namespace compute_b_l811_811027

-- Define the predicate stating that a given number is a root of the polynomial
def is_root (p : Polynomial ℚ) (r : ℚ) : Prop :=
  p.eval r = 0

-- The theorem stating that if 2 + sqrt(3) is a root of the given polynomial,
-- then the coefficient b is -39
theorem compute_b (a b : ℚ) (h : is_root (Polynomial.mk [10, b, a, 1]) (2 + real.sqrt 3)) :
  b = -39 :=
sorry

end compute_b_l811_811027


namespace find_interval_l811_811545

theorem find_interval (n : ℕ) (h1 : n < 3000)
  (h2 : ∃ ab : ℕ, ab < 100 ∧ 100*k + ab = n for some k)
  (h3 : ∃ uvw : ℕ, uvw < 1000 ∧ 1000*m + uvw = n + 8 for some m):
  601 ≤ n ∧ n ≤ 1200 :=
sorry

end find_interval_l811_811545


namespace expected_faces_rolled_six_times_l811_811444

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l811_811444


namespace sum_gcd_lcm_eight_twelve_l811_811381

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l811_811381


namespace distance_between_parallel_lines_l811_811786

theorem distance_between_parallel_lines 
  (circle : Type)
  (O : circle)
  (A B C D E F P : circle)
  (AB CD EF : ℝ)
  (d : ℝ)
  (h1 : AB = 26)
  (h2 : CD = 38)
  (h3 : EF = 34)
  (h4 : IsMidpoint P C D)
  (h5 : IsPerpendicular (line O P) (line A B))
  (h6 : IsPerpendicular (line O P) (line C D))
  : d = 14 := 
sorry

end distance_between_parallel_lines_l811_811786


namespace log_base_fraction_eq_l811_811979

theorem log_base_fraction_eq (x : ℝ) : (1 / 4) ^ x = 16 → x = -2 :=
by
  sorry

end log_base_fraction_eq_l811_811979


namespace convex_dodecagon_diagonals_l811_811074

theorem convex_dodecagon_diagonals :
  let n := 12 in
  let D := (n * (n - 3)) / 2 in
  D = 54 := 
by
  let n := 12
  let D := (n * (n - 3)) / 2
  show D = 54 from sorry

end convex_dodecagon_diagonals_l811_811074


namespace log_base_frac_l811_811976

theorem log_base_frac (y : ℝ) : (log (1/4) 16) = -2 := by
  sorry

end log_base_frac_l811_811976


namespace tangency_of_circumcircle_l811_811676

open Classical

variables (A B C : Type) [EuclideanSpace A B C] 
variables (Γ : Circle) [CircumcircleABC : Circumcircle Γ A B C]
variables (AC BC : LineSegment) [lt_AC_BC : AC.length < BC.length]
variables (M : Point) [MidpointAB : Midpoint M A B]
variables (CC' : LineSegment) [DiameterCC' : Diameter Γ CC']
variables (CM : Line) 
variables (K L : Point) [IntersectCM : Intersects CM CC' K L]
variables (l1 l2 : Line) [PerpendicularK_AC'_L : Perpendicular (Point K) (Line AC') l1]
variables [PerpendicularL_BC'_L : Perpendicular (Point L) (Line BC') l2]

theorem tangency_of_circumcircle :
  TangentCircumcircle (CircumcircleTriangle l1 l2 AB) Γ :=
sorry

end tangency_of_circumcircle_l811_811676


namespace no_call_days_in_2017_l811_811246

theorem no_call_days_in_2017 :
  let total_days := 365,
      call_interval_1 := 2,
      call_interval_2 := 3,
      call_interval_3 := 6,
      calls_by_nephew_1 := total_days / call_interval_1,
      calls_by_nephew_2 := total_days / call_interval_2,
      calls_by_nephew_3 := total_days / call_interval_3,
      overlap_1_2 := Nat.lcm call_interval_1 call_interval_2,
      overlap_total := call_interval_3,
      calls_on_overlap_1_2 := total_days / overlap_1_2,
      calls_on_overlap_total := total_days / overlap_total,
      days_with_calls :=
        (calls_by_nephew_1 + calls_by_nephew_2 + calls_by_nephew_3) -
        (calls_on_overlap_1_2 + calls_on_overlap_total) + calls_on_overlap_total in
  total_days - days_with_calls = 122 :=
by
  sorry

end no_call_days_in_2017_l811_811246


namespace angle_co_terminal_with_neg_525_l811_811986

theorem angle_co_terminal_with_neg_525 :
  ∃ k : ℤ, -525° + k * 360° = 195° :=
sorry

end angle_co_terminal_with_neg_525_l811_811986


namespace min_value_of_expression_l811_811585

theorem min_value_of_expression :
  ∃ (a b : ℝ), (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ x^2 + a * x + b - 3 = 0) ∧ a^2 + (b - 4)^2 = 2 :=
sorry

end min_value_of_expression_l811_811585


namespace hyperbola_focal_length_l811_811059

def is_hyperbola (x y a : ℝ) : Prop := (x^2) / (a^2) - (y^2) = 1
def is_perpendicular_asymptote (slope_asymptote slope_line : ℝ) : Prop := slope_asymptote * slope_line = -1

theorem hyperbola_focal_length {a : ℝ} (h1 : is_hyperbola x y a)
  (h2 : is_perpendicular_asymptote (1 / a) (-1)) : 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
sorry

end hyperbola_focal_length_l811_811059


namespace problem1_problem2_l811_811265

noncomputable def f (x : ℝ) := abs (x - 1) + abs (x + 1)

theorem problem1 (x : ℝ) : f x ≤ x + 2 → 0 ≤ x ∧ x ≤ 2 := by
  sorry

theorem problem2 (x : ℝ) (hx : ∀ (a : ℝ), a ≠ 0 → f x ≥ (abs (a + 1) - abs (2a - 1)) / abs a) :
  x ≤ -3 / 2 ∨ x ≥ 3 / 2 := by
  sorry

end problem1_problem2_l811_811265


namespace min_tan_diff_l811_811591

theorem min_tan_diff (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (e : ℝ) (h4 : e = (Real.sqrt 3) / 2) 
    (P : ℝ × ℝ) (hP : (P ≠ (-a, 0)) ∧ (P ≠ (a, 0)) ∧ (P.fst^2 / a^2 + P.snd^2 / b^2 = 1))
    (α β : ℝ) (hα : Real.tan α = P.snd / (P.fst + a)) (hβ : Real.tan β = P.snd / (P.fst - a)) :
    Real.abs (Real.tan α - Real.tan β) ≥ 1 :=
begin
  -- proof will go here eventually
  sorry
end

end min_tan_diff_l811_811591


namespace number_of_diagonals_dodecagon_l811_811072

theorem number_of_diagonals_dodecagon : 
  let n := 12 
  in (n * (n - 3)) / 2 = 54 := 
by
  let n := 12
  have h1 : n * (n - 3) = 108 := by sorry
  have h2 : 108 / 2 = 54 := by sorry
  exact Eq.trans (Eq.trans (Eq.trans rfl h1) (Eq.symm h2)) rfl

end number_of_diagonals_dodecagon_l811_811072


namespace cookies_in_fridge_l811_811325

theorem cookies_in_fridge (total_baked : ℕ) (cookies_Tim : ℕ) (cookies_Mike : ℕ) (cookies_Sarah : ℕ) (cookies_Anna : ℕ)
  (h_total_baked : total_baked = 1024)
  (h_cookies_Tim : cookies_Tim = 48)
  (h_cookies_Mike : cookies_Mike = 58)
  (h_cookies_Sarah : cookies_Sarah = 78)
  (h_cookies_Anna : cookies_Anna = (2 * (cookies_Tim + cookies_Mike)) - (cookies_Sarah / 2)) :
  total_baked - (cookies_Tim + cookies_Mike + cookies_Sarah + cookies_Anna) = 667 := by
sorry

end cookies_in_fridge_l811_811325


namespace sum_of_distances_l811_811823

open Real

-- Define the conditions
variables (M AC ABM CBM : Point)
variables (beta1 beta2 : ℝ)
variables (a R m n b : ℝ)
variables (h1 : M ∈ segment AC)
variables (h2 : angle ABM = beta1)
variables (h3 : angle CBM = beta2)
variables (h4 : distance AB = a)
variables (h5 : R ≠ 0)
variables (h6 : distance AM = m)
variables (h7 : distance CM = n)
variables (h8 : distance BM = b)
variables (h9 : m = 2 * R * sin beta1)
variables (h10: n = 2 * R * sin beta2)
variables (h11: beta2 = 60 * π / 180 - beta1)

-- Prove the statement
theorem sum_of_distances (h1 : M ∈ segment AC)
                         (h2 : angle ABM = beta1)
                         (h3 : angle CBM = beta2)
                         (h4 : distance AB = a)
                         (h5 : R ≠ 0)
                         (h6 : distance AM = m)
                         (h7 : distance CM = n)
                         (h8 : distance BM = b)
                         (h9 : m = 2 * R * sin beta1)
                         (h10: n = 2 * R * sin beta2)
                         (h11: beta2 = 60 * π / 180 - beta1) : 
                         m + n = b := 
by sorry

end sum_of_distances_l811_811823


namespace sum_of_solutions_abs_quadratic_l811_811799

theorem sum_of_solutions_abs_quadratic :
  let cond := (fun x : ℝ => abs (x^2 - 14*x + 44) = 4),
      sum_solutions := (fun f => ∑ x in f.to_finset, x)
  in
  sum_solutions (cond.to_fun {x : ℝ | cond x}) = 28 :=
by sorry

end sum_of_solutions_abs_quadratic_l811_811799


namespace marbles_end_of_day_l811_811963

theorem marbles_end_of_day :
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54 :=
by
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  show initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54
  sorry

end marbles_end_of_day_l811_811963


namespace sqrt_of_factorial_div_l811_811998

theorem sqrt_of_factorial_div : 
  √(9! / 126) = 8 * √5 :=
by
  sorry

end sqrt_of_factorial_div_l811_811998


namespace distance_light_300_years_eq_l811_811759

-- Define the constant distance light travels in one year
def distance_light_year : ℕ := 9460800000000

-- Define the time period in years
def time_period : ℕ := 300

-- Define the expected distance light travels in 300 years in scientific notation
def expected_distance : ℝ := 28382 * 10^13

-- The theorem to prove
theorem distance_light_300_years_eq :
  (distance_light_year * time_period) = 2838200000000000 :=
by
  sorry

end distance_light_300_years_eq_l811_811759


namespace perpendicular_bisector_eqn_parallel_line_eqn_l811_811066

-- Define the points A and B
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Define the equations for the perpendicular bisector and the parallel line
def equation_of_perpendicular_bisector : ℝ → ℝ → Prop := λ x y,
  3 * x - 4 * y - 23 = 0

def equation_of_parallel_line : ℝ → ℝ → Prop := λ x y,
  4 * x + 3 * y + 1 = 0

-- The goal is to prove that the given equations match the expected ones.
theorem perpendicular_bisector_eqn :
  ∃ x y, equation_of_perpendicular_bisector x y :=
sorry

theorem parallel_line_eqn :
  ∃ x y, equation_of_parallel_line x y :=
sorry

end perpendicular_bisector_eqn_parallel_line_eqn_l811_811066


namespace Anton_thought_number_is_729_l811_811909

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l811_811909


namespace Anton_thought_number_is_729_l811_811904

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l811_811904


namespace number_of_women_attended_l811_811486

theorem number_of_women_attended
  (m : ℕ) (w : ℕ)
  (men_dance_women : m = 15)
  (women_dance_men : ∀ i : ℕ, i < 15 → i * 4 = 60)
  (women_condition : w * 3 = 60) :
  w = 20 :=
sorry

end number_of_women_attended_l811_811486


namespace binomial_divisible_by_prime_l811_811549

theorem binomial_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  (Nat.choose n p) - (n / p) % p = 0 := 
sorry

end binomial_divisible_by_prime_l811_811549


namespace min_value_of_z_l811_811237

noncomputable def minimum_value_of_norm (z : ℂ) : ℝ :=
  if h : |z - 2 * Complex.i| + |z - 5| = 7 then |z|
  else 0

theorem min_value_of_z {z : ℂ} (h : |z - 2 * Complex.i| + |z - 5| = 7) :
  minimum_value_of_norm z = (20 * Real.sqrt 29) / 29 :=
sorry

end min_value_of_z_l811_811237


namespace expected_faces_rolled_six_times_l811_811443

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l811_811443


namespace probability_quadratic_expression_negative_l811_811179

-- Definition of the quadratic expression
def quadratic_expr (p : ℕ) : ℤ :=
  p^2 - 13 * p + 40

-- Definition of the condition that p is a positive integer between 1 and 10 inclusive
def valid_p (p : ℕ) : Prop :=
  1 ≤ p ∧ p ≤ 10

-- The theorem stating the probability that s < 0 is 1/5
theorem probability_quadratic_expression_negative :
  (∑ p in (Finset.range 10).filter valid_p, if quadratic_expr p < 0 then 1 else 0) / 10 = 1 / 5 :=
by
    sorry

end probability_quadratic_expression_negative_l811_811179


namespace anton_thought_number_is_729_l811_811893

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l811_811893


namespace number_of_distinct_prime_factors_of_90_l811_811079

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811079


namespace find_certain_number_l811_811185

theorem find_certain_number (mystery_number certain_number : ℕ) (h1 : mystery_number = 47) 
(h2 : mystery_number + certain_number = 92) : certain_number = 45 :=
by
  sorry

end find_certain_number_l811_811185


namespace find_value_of_a_l811_811647

-- Defining the conditions
def curve1 (ρ θ : ℝ) : Prop := ρ * (sqrt 2 * real.cos θ + real.sin θ) = 1
def curve2 (ρ a : ℝ) (ha : a > 0) : Prop := ρ = a

-- Lean statement for the proof problem
theorem find_value_of_a (a : ℝ) (ha : a > 0) :
  (∃ θ : ℝ, curve1 a θ ∧ curve2 a a ha ∧ (a * real.sin θ = 0)) → a = sqrt 2 / 2 :=
by
  sorry

end find_value_of_a_l811_811647


namespace sum_of_cubes_consecutive_integers_divisible_by_9_l811_811251

theorem sum_of_cubes_consecutive_integers_divisible_by_9 (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 :=
sorry

end sum_of_cubes_consecutive_integers_divisible_by_9_l811_811251


namespace sum_of_gcd_and_lcm_is_28_l811_811388

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l811_811388


namespace polynomial_center_of_symmetry_l811_811774

noncomputable def polynomial_has_symmetry_center (P : Polynomial ℝ) :=
  (∀ m n : ℤ, P.eval (m : ℝ) + P.eval (n : ℝ) = 0 → m = n) → (∃ a : ℝ, ∀ x : ℝ, P.eval (a - x) = -P.eval (a + x))

-- Statement of the theorem
theorem polynomial_center_of_symmetry {P : Polynomial ℝ}
  (h1 : ∀ m n : ℤ, P.eval (m : ℝ) + P.eval (n : ℝ) = 0 → m = n)
  (h2 : ∀ᶠ n : ℤ in filter.at_top, ∃ m : ℤ, P.eval (m : ℝ) + P.eval (n : ℝ) = 0) :
  ∃ a : ℝ, ∀ x : ℝ, P.eval (a - x) = -P.eval (a + x) := sorry

end polynomial_center_of_symmetry_l811_811774


namespace gcd_lcm_sum_8_12_l811_811349

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l811_811349


namespace max_omega_value_l811_811049

theorem max_omega_value :
  ∀ (f : ℝ → ℝ) (ω : ℝ) (φ : ℝ),
    (ω > 0) →
    (|φ| ≤ π / 2) →
    (f = λ x, sin (ω * x + φ)) →
    (f (-π / 4) = 0) →
    (f (π / 4) = f (-π / 4)) →
    (∀ x ∈ Ioo (π / 18) (5 * π / 36), f x) →
    ω ≤ 9 := 
sorry

end max_omega_value_l811_811049


namespace anton_thought_number_l811_811885

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l811_811885


namespace sebastian_missed_days_l811_811259

noncomputable def daily_salary : ℝ := 208.60

noncomputable def deducted_amount : ℝ := 298

noncomputable def missed_workdays : ℝ := deducted_amount / daily_salary

theorem sebastian_missed_days 
    (weekly_salary : ℝ := 1043) 
    (deducted_salary : ℝ := 745) 
    (days_per_week : ℕ := 5) :
    missed_workdays ≈ 2 :=
by
  sorry

end sebastian_missed_days_l811_811259


namespace find_a4_l811_811675

variable {a_n : ℕ → ℕ}
variable {S : ℕ → ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def sum_first_n_terms (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_a4 (h : S 7 = 35) (hs : sum_first_n_terms S a_n) (ha : is_arithmetic_sequence a_n) : a_n 4 = 5 := 
  by sorry

end find_a4_l811_811675


namespace sam_money_left_l811_811257

/- Definitions -/

def initial_dimes : ℕ := 38
def initial_quarters : ℕ := 12
def initial_nickels : ℕ := 25
def initial_pennies : ℕ := 30

def price_per_candy_bar_dimes : ℕ := 4
def price_per_candy_bar_nickels : ℕ := 2
def candy_bars_bought : ℕ := 5

def price_per_lollipop_nickels : ℕ := 6
def price_per_lollipop_pennies : ℕ := 10
def lollipops_bought : ℕ := 2

def price_per_bag_of_chips_quarters : ℕ := 1
def price_per_bag_of_chips_dimes : ℕ := 3
def price_per_bag_of_chips_pennies : ℕ := 5
def bags_of_chips_bought : ℕ := 3

/- Proof problem statement -/

theorem sam_money_left : 
  (initial_dimes * 10 + initial_quarters * 25 + initial_nickels * 5 + initial_pennies * 1) - 
  (
    candy_bars_bought * (price_per_candy_bar_dimes * 10 + price_per_candy_bar_nickels * 5) + 
    lollipops_bought * (price_per_lollipop_nickels * 5 + price_per_lollipop_pennies * 1) +
    bags_of_chips_bought * (price_per_bag_of_chips_quarters * 25 + price_per_bag_of_chips_dimes * 10 + price_per_bag_of_chips_pennies * 1)
  ) = 325 := 
sorry

end sam_money_left_l811_811257


namespace factorization_of_x6_minus_81_l811_811507

theorem factorization_of_x6_minus_81 :
  (x : ℝ) → (x^6 - 81 = (x^3 + 9) * (x - 3) * (x^2 + 3x + 9)) :=
by
  intro x
  sorry

end factorization_of_x6_minus_81_l811_811507


namespace anton_thought_of_729_l811_811870

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l811_811870


namespace freddy_spent_10_dollars_l811_811006

theorem freddy_spent_10_dollars 
  (talk_time_dad : ℕ) (talk_time_brother : ℕ) 
  (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ)
  (conversion_cents_to_dollar : ℕ)
  (h1 : talk_time_dad = 45)
  (h2 : talk_time_brother = 31)
  (h3 : local_cost_per_minute = 5)
  (h4 : international_cost_per_minute = 25)
  (h5 : conversion_cents_to_dollar = 100):
  (local_cost_per_minute * talk_time_dad + international_cost_per_minute * talk_time_brother) / conversion_cents_to_dollar = 10 :=
by
  sorry

end freddy_spent_10_dollars_l811_811006


namespace ratio_of_toys_l811_811262

theorem ratio_of_toys (total_toys : ℕ) (total_friends : ℕ)
  (h_total_toys : total_toys = 118)
  (h_total_friends : total_friends = 4) :
  (29 : ℚ) / (118 : ℚ) = (1 : ℚ) / (4 : ℚ) :=
by {
  -- Condition on total toys and total friends are not really mathematical constants that affect adult proof,
  -- but we keep them for clarity and context of the problem. 
  rw [h_total_toys, h_total_friends],
  exact (by norm_num : (29 : ℚ) / (118 : ℚ) = (1 : ℚ) / (4 : ℚ)),
  sorry
}

end ratio_of_toys_l811_811262


namespace right_triangle_construction_l811_811933

-- Definitions based on the conditions
variables {c m : ℝ} -- c is the given leg (cathetus) and m is the given median

-- The mathematically equivalent proof problem rewritten in Lean 4 statement
theorem right_triangle_construction (h1 : c > 0) (h2 : m > 0) :
  ∃ (A B C : ℝ × ℝ), 
    dist A C = c ∧ 
    ∃ M, midpoint ℝ A B M ∧ dist C M = m ∧ 
    ∃ r, (A.1 - B.1)^2 + (A.2 - B.2)^2 = r^2 ∧ 
    (A.1 - M.1)^2 + (A.2 - M.2)^2 = (M.1 - B.1)^2 + (M.2 - B.2)^2 ∧ (C.1 - M.1)* (A.1 - M.1) + (C.2 - M.2) * (A.2 - M.2) = 0 :=
sorry

end right_triangle_construction_l811_811933


namespace distinct_prime_factors_count_l811_811123

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811123


namespace triangle_angle_A_eq_pi_div_3_l811_811186

noncomputable def geometric_seq (a b c : ℝ) : Prop :=
  a * c = b * b

noncomputable def arithmetic_seq (a b c : ℝ) (α β γ : ℝ) : Prop :=
  a + c = 2 * b

theorem triangle_angle_A_eq_pi_div_3 
  (A B C : ℝ) 
  (AC CB BA : ℝ) 
  (h_geom : geometric_seq AC CB BA) 
  (BA_dot_BC AB_dot_AC CA_dot_CB : ℝ)
  (h_arith : arithmetic_seq BA_dot_BC AB_dot_AC CA_dot_CB) 
  (dot_product_eq : BA_dot_BC + AB_dot_AC = CA_dot_CB)
  (cos_B cos_C : ℝ)
  (BC : ℝ)
  (BC_eq : AB_dot_AC = 2 * BC^2)
  (cos_A_eq : cos_B + cos_C = 2 * cos (A)) :
  cos (A) = 1 / 2 → A = Real.pi / 3 :=
by
  sorry

end triangle_angle_A_eq_pi_div_3_l811_811186


namespace anton_thought_number_l811_811884

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l811_811884


namespace log_one_fourth_sixteen_l811_811971

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 :=
by
  sorry

end log_one_fourth_sixteen_l811_811971


namespace anton_thought_number_l811_811920

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l811_811920


namespace smallest_YZ_minus_XZ_l811_811320

theorem smallest_YZ_minus_XZ 
  (XZ YZ XY : ℕ)
  (h_sum : XZ + YZ + XY = 3001)
  (h_order : XZ < YZ ∧ YZ ≤ XY)
  (h_triangle_ineq1 : XZ + YZ > XY)
  (h_triangle_ineq2 : XZ + XY > YZ)
  (h_triangle_ineq3 : YZ + XY > XZ) :
  ∃ XZ YZ XY : ℕ, YZ - XZ = 1 := sorry

end smallest_YZ_minus_XZ_l811_811320


namespace number_of_distinct_prime_factors_of_90_l811_811105

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811105


namespace polynomial_factorization_l811_811288

theorem polynomial_factorization (a b : ℤ) (h : (x^2 + x - 6) = (x + a) * (x + b)) :
  (a + b)^2023 = 1 :=
sorry

end polynomial_factorization_l811_811288


namespace anton_thought_number_is_729_l811_811892

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l811_811892


namespace vincent_earnings_after_5_days_l811_811331

def fantasy_book_price : ℕ := 4
def daily_fantasy_books_sold : ℕ := 5
def literature_book_price : ℕ := fantasy_book_price / 2
def daily_literature_books_sold : ℕ := 8
def days : ℕ := 5

def daily_earnings : ℕ :=
  (fantasy_book_price * daily_fantasy_books_sold) +
  (literature_book_price * daily_literature_books_sold)

def total_earnings (d : ℕ) : ℕ :=
  daily_earnings * d

theorem vincent_earnings_after_5_days : total_earnings days = 180 := by
  sorry

end vincent_earnings_after_5_days_l811_811331


namespace max_value_of_sum_l811_811228

theorem max_value_of_sum (x y z : ℝ) (h : x^2 + 4 * y^2 + 9 * z^2 = 3) : x + 2 * y + 3 * z ≤ 3 :=
sorry

end max_value_of_sum_l811_811228


namespace magnitude_of_c_l811_811607
-- import the necessary libraries

-- Define the conditions
variables {ℝ : Type*} -- Ensure vector components are over real numbers

-- Given vectors
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (-2, 4)
def c (x y : ℝ) : ℝ × ℝ := (x, y)

-- Conditions
def parallel_condition (x y : ℝ) : Prop :=
  x = -y

def perpendicular_condition (x y : ℝ) : Prop :=
  let bc := (b.1 + x, b.2 + y)
  (a.1 * bc.1 + a.2 * bc.2 = 0)

def c_magnitude (x y : ℝ) : ℝ :=
  Real.sqrt (x * x + y * y)

-- The proof statement
theorem magnitude_of_c 
  (x y : ℝ) 
  (h1 : parallel_condition x y)
  (h2 : perpendicular_condition x y) :
  c_magnitude x y = 3 * Real.sqrt 2 :=
sorry

end magnitude_of_c_l811_811607


namespace properties_true_l811_811761

def f (x : ℝ) : ℝ := sin x - sqrt 3 * cos x

theorem properties_true :
  (¬(f (π / 6) = 1 ∨ f (π / 6) = -1)) ∧
  (∀ x, f x ≤ 2) ∧
  (∀ x, f x - (f x - 2) = 2) ∧
  (∀ x, (-π / 6) ≤ x ∧ x ≤ (5 * π / 6) → ∃ a b, a < b → f a < f b) ∧
  (f (4 * π / 3) = 0) :=
by
  sorry

end properties_true_l811_811761


namespace Anton_thought_of_729_l811_811917

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l811_811917


namespace count_integer_values_l811_811615

theorem count_integer_values (x : ℤ) (h : ⌈real.sqrt x⌉ = 18) : 
  ∃ (n : ℕ), n = 35 :=
by 
  sorry

end count_integer_values_l811_811615


namespace ranking_arrangements_l811_811637

/-- In a ranking competition, five students A, B, C, D, and E are ranked from first to fifth place.
   Given that:
   1. Student A did not get the first place.
   2. Student B got the third place.
   Prove that the number of possible ranking arrangements is 18. --/
theorem ranking_arrangements {A B C D E : Type} :
  let possible_arrangements := 4! - 3! in  -- The number of possible ranks
  possible_arrangements = 18 :=
by
  sorry

end ranking_arrangements_l811_811637


namespace arithmetic_sequence_x_value_l811_811513

theorem arithmetic_sequence_x_value :
  (∀ n : ℕ, ∃ (a d : ℤ), (n = 0 → a = 1) ∧ (n = 1 → a = 2 * (-11) - 3) ∧ (n = 2 → a = 5 * (-11) + 4) ∧ (∀ m : ℕ, a + m * d = 1 + n * (d + m * d)))
  → (-11: ℤ) :=
sorry

end arithmetic_sequence_x_value_l811_811513


namespace anton_thought_number_l811_811923

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l811_811923


namespace a_n_formula_smallest_n_l811_811777

-- Define sequence a_n based on given condition
def a (n : ℕ) : ℕ := 
  if n = 1 then 4 
  else 2 * n * (n + 1)

-- Define sequence b_n as the reciprocal of a_n
def b (n : ℕ) : ℚ := 1 / (a n : ℚ)

-- Define the sum of first n terms of the sequence b_n
def S (n : ℕ) : ℚ := 
  (Finset.range (n + 1)).sum (λ k, b k)

-- Prove the general term formula for a_n
theorem a_n_formula (n : ℕ) (h : n ≠ 0) : a n = 2 * n * (n + 1) := 
  sorry

-- Prove the smallest positive integer n such that S_n > 9 / 20 is 10
theorem smallest_n (n : ℕ) (h : S n > 9 / 20) : n = 10 := 
  sorry

end a_n_formula_smallest_n_l811_811777


namespace cannot_place_1_to_21_in_diagram_l811_811939

theorem cannot_place_1_to_21_in_diagram :
  ¬ ∃ (a b c d e f : ℕ) (numbers : Fin 21 → ℕ),
    (∀ i, numbers i ∈ Finset.range 21.succ) ∧
    (numbers 0 = a) ∧
    (numbers 1 = b) ∧
    (numbers 2 = c) ∧
    (numbers 3 = d) ∧
    (numbers 4 = e) ∧
    (numbers 5 = f) ∧
    -- Second row
    (numbers 6 = |a - b|) ∧
    (numbers 7 = |b - c|) ∧
    (numbers 8 = |c - d|) ∧
    (numbers 9 = |d - e|) ∧
    (numbers 10 = |e - f|) ∧
    -- Third row
    (numbers 11 = |numbers 6 - numbers 7|) ∧
    (numbers 12 = |numbers 7 - numbers 8|) ∧
    (numbers 13 = |numbers 8 - numbers 9|) ∧
    (numbers 14 = |numbers 9 - numbers 10|) ∧
    -- Fourth row
    (numbers 15 = |numbers 11 - numbers 12|) ∧
    (numbers 16 = |numbers 12 - numbers 13|) ∧
    (numbers 17 = |numbers 13 - numbers 14|) ∧
    -- Fifth row
    (numbers 18 = |numbers 15 - numbers 16|) ∧
    (numbers 19 = |numbers 16 - numbers 17|) ∧
    -- Sixth row
    (numbers 20 = |numbers 18 - numbers 19|).
Proof := sorry

end cannot_place_1_to_21_in_diagram_l811_811939


namespace carl_additional_gift_bags_l811_811940

theorem carl_additional_gift_bags (definite_visitors additional_visitors extravagant_bags average_bags total_bags_needed : ℕ) :
  definite_visitors = 50 →
  additional_visitors = 40 →
  extravagant_bags = 10 →
  average_bags = 20 →
  total_bags_needed = 90 →
  (total_bags_needed - (extravagant_bags + average_bags)) = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end carl_additional_gift_bags_l811_811940


namespace kevin_spent_on_drinks_l811_811725

-- Definitions for the conditions
def total_budget_kevin : ℝ := 20
def food_expense_kevin : ℝ := 4

-- Proving the amount Kevin spent on drinks is $16
theorem kevin_spent_on_drinks : 
  let drinks_expense_kevin := total_budget_kevin - food_expense_kevin in
  drinks_expense_kevin = 16 :=
by
  sorry

end kevin_spent_on_drinks_l811_811725


namespace area_under_curve_l811_811419

open Real IntervalIntegral

-- Defining the given function
def f (x : ℝ) := cos x * (sin x) ^ 2

-- The interval boundaries
def a : ℝ := 0
def b : ℝ := π / 2

-- Stating the theorem
theorem area_under_curve : ∫ x in a..b, f x = 1 / 3 :=
sorry

end area_under_curve_l811_811419


namespace number_of_distinct_prime_factors_of_90_l811_811084

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811084


namespace least_area_of_triangles_l811_811660

-- Define the points A, B, C, D of the unit square
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (0, 1)

-- Define the function s(M, N) as the least area of the triangles having their vertices in the set {A, B, C, D, M, N}
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

noncomputable def s (M N : ℝ × ℝ) : ℝ :=
  min (min (min (min (min (triangle_area A B M) (triangle_area A B N)) (triangle_area A C M)) (triangle_area A C N)) (min (triangle_area A D M) (triangle_area A D N)))
    (min (min (min (triangle_area B C M) (triangle_area B C N)) (triangle_area B D M)) (min (triangle_area B D N) (min (triangle_area C D M) (triangle_area C D N))))

-- Define the statement to prove
theorem least_area_of_triangles (M N : ℝ × ℝ)
  (hM : M.1 > 0 ∧ M.1 < 1 ∧ M.2 > 0 ∧ M.2 < 1)
  (hN : N.1 > 0 ∧ N.1 < 1 ∧ N.2 > 0 ∧ N.2 < 1)
  (hMN : (M ≠ A ∨ N ≠ A) ∧ (M ≠ B ∨ N ≠ B) ∧ (M ≠ C ∨ N ≠ C) ∧ (M ≠ D ∨ N ≠ D))
  : s M N ≤ 1 / 8 := 
sorry

end least_area_of_triangles_l811_811660


namespace sandwich_cost_l811_811338

theorem sandwich_cost (c : ℕ) 
  (sandwiches : ℕ := 3)
  (drinks : ℕ := 2)
  (cost_per_drink : ℕ := 4)
  (total_spent : ℕ := 26)
  (drink_cost : ℕ := drinks * cost_per_drink)
  (sandwich_spent : ℕ := total_spent - drink_cost) :
  (∀ s, sandwich_spent = s * sandwiches → s = 6) :=
by
  intros s hs
  have hsandwich_count : sandwiches = 3 := by rfl
  have hdrinks : drinks = 2 := by rfl
  have hcost_per_drink : cost_per_drink = 4 := by rfl
  have htotal_spent : total_spent = 26 := by rfl
  have hdrink_cost : drink_cost = 8 := by
    calc 
      drinks * cost_per_drink 
      = 2 * 4 : by rw [hdrinks, hcost_per_drink]
      = 8 : by norm_num
  have hsandwich_spent : sandwich_spent = 18 := by
    calc
      total_spent - drink_cost 
      = 26 - 8 : by rw [htotal_spent, hdrink_cost]
      = 18 : by norm_num
  rw hsandwich_count at hs
  rw hsandwich_spent at hs
  linarith

end sandwich_cost_l811_811338


namespace convex_dodecagon_diagonals_l811_811073

theorem convex_dodecagon_diagonals :
  let n := 12 in
  let D := (n * (n - 3)) / 2 in
  D = 54 := 
by
  let n := 12
  let D := (n * (n - 3)) / 2
  show D = 54 from sorry

end convex_dodecagon_diagonals_l811_811073


namespace problem_1_problem_2_l811_811830

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x-2) / (x+2) * exp x

-- Problem (1): Prove that when x > 0, (x-2)e^x + x + 2 > 0
theorem problem_1 (x : ℝ) (hx : x > 0) : (x - 2) * exp x + x + 2 > 0 :=
sorry

-- Define the function g(x)
def g (x a : ℝ) : ℝ := (exp x - a * x - a) / (x * x)

-- Define the function h(a) as per given definition
def h (a t : ℝ) : ℝ := (exp t) / (t + 2)

-- Problem (2): Prove that when a ∈ [0, 1), g(x) has a minimum value and find the range of h(a)
theorem problem_2 (a : ℝ) (ha : 0 ≤ a) (ha1 : a < 1) :
  ∃ t, ∀ x, g t a ≤ g x a ∧ (h a t ∈ Set.Icc (1/2) (exp 2 / 4)) :=
sorry

end problem_1_problem_2_l811_811830


namespace son_age_l811_811455

theorem son_age (M S : ℕ) (h1: M = S + 26) (h2: M + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end son_age_l811_811455


namespace g_at_3_l811_811764

-- Definition of the function and its property
def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, x ≠ 0 → g(x) - 3 * g(1/x) = 3^x

-- Goal: Prove that g(3) = 16.7
theorem g_at_3 : g 3 = 16.7 :=
by
  sorry

end g_at_3_l811_811764


namespace eccentricity_of_ellipse_l811_811034

theorem eccentricity_of_ellipse (m n : ℝ) 
  (h_arithmetic : 2 * n = m + (m + n))
  (h_geometric : n^2 = m * (m * n)) :
  let a := real.sqrt n
  let b := real.sqrt m
  let c := real.sqrt (a^2 - b^2)
  (c / a = real.sqrt 2 / 2) :=
by {
  let a := real.sqrt n,
  let b := real.sqrt m,
  let c := real.sqrt (a^2 - b^2),
  sorry
}

end eccentricity_of_ellipse_l811_811034


namespace bhupathi_amount_l811_811411

variable (A B : ℝ)

theorem bhupathi_amount :
  (A + B = 1210 ∧ (4 / 15) * A = (2 / 5) * B) → B = 484 :=
by
  sorry

end bhupathi_amount_l811_811411


namespace log_base_fraction_eq_l811_811977

theorem log_base_fraction_eq (x : ℝ) : (1 / 4) ^ x = 16 → x = -2 :=
by
  sorry

end log_base_fraction_eq_l811_811977


namespace distinct_prime_factors_of_90_l811_811095

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811095


namespace det_min_element_matrix_is_one_l811_811824

def min_element_matrix (n : ℕ) : matrix (fin n) (fin n) ℝ :=
  λ i j, min ((i : ℕ) + 1) ((j : ℕ) + 1)

theorem det_min_element_matrix_is_one (n : ℕ) :
  matrix.det (min_element_matrix n) = 1 :=
sorry

end det_min_element_matrix_is_one_l811_811824


namespace hunter_rats_l811_811221

-- Defining the conditions
variable (H : ℕ) (E : ℕ := H + 30) (K : ℕ := 3 * (H + E)) 
  
-- Defining the total number of rats condition
def total_rats : Prop := H + E + K = 200

-- Defining the goal: Prove Hunter has 10 rats
theorem hunter_rats (h : total_rats H) : H = 10 := by
  sorry

end hunter_rats_l811_811221


namespace sum_of_GCF_and_LCM_l811_811364

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l811_811364


namespace QED_mul_eq_neg_25I_l811_811674

namespace ComplexMultiplication

open Complex

def Q : ℂ := 3 + 4 * Complex.I
def E : ℂ := -Complex.I
def D : ℂ := 3 - 4 * Complex.I

theorem QED_mul_eq_neg_25I : Q * E * D = -25 * Complex.I :=
by
  sorry

end ComplexMultiplication

end QED_mul_eq_neg_25I_l811_811674


namespace slope_of_line_l811_811602

theorem slope_of_line (a : ℝ) (h : a = (Real.tan (Real.pi / 3))) : a = Real.sqrt 3 := by
sorry

end slope_of_line_l811_811602


namespace option_A_option_B_option_C_option_D_l811_811056

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (x - 1) * Real.cos (Real.pi / 2 - x)

theorem option_A : ∀ x ∈ set.Icc (-Real.pi / 2) (0 : ℝ), 
  (differentiable ℝ f) → 0 < (fderiv ℝ f x).toFun 1 :=
sorry

theorem option_B : ∃ a b : ℝ, a < b ∧ a ∈ set.Icc (-Real.pi) (0 : ℝ) ∧ b ∈ set.Icc (-Real.pi) (0) ∧ f a = 0 ∧ f b = 0 :=
sorry

theorem option_C : ∀ k : ℤ, k = -3 → ∀ x ∈ set.Icc (-Real.pi : ℝ) 0, f x - 2 * k ≥ 0 :=
sorry

theorem option_D : ∀ x ∈ set.Ioo 0 (1 : ℝ), f x < Real.exp 1 - Real.log x :=
sorry

end option_A_option_B_option_C_option_D_l811_811056


namespace num_distinct_prime_factors_90_l811_811125

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811125


namespace december_revenue_times_average_l811_811816

def revenue_in_december_is_multiple_of_average_revenue (R_N R_J R_D : ℝ) : Prop :=
  R_N = (3/5) * R_D ∧    -- Condition: November's revenue is 3/5 of December's revenue
  R_J = (1/3) * R_N ∧    -- Condition: January's revenue is 1/3 of November's revenue
  R_D = 2.5 * ((R_N + R_J) / 2)   -- Question: December's revenue is 2.5 times the average of November's and January's revenue

theorem december_revenue_times_average (R_N R_J R_D : ℝ) :
  revenue_in_december_is_multiple_of_average_revenue R_N R_J R_D :=
by
  -- adding sorry to skip the proof
  sorry

end december_revenue_times_average_l811_811816


namespace sum_of_GCF_and_LCM_l811_811360

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l811_811360


namespace number_of_distinct_prime_factors_of_90_l811_811075

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811075


namespace regression_incorrect_stmt_C_l811_811803

variables {n : ℕ}
variables (x y : Fin n → ℝ) (x_bar y_bar : ℝ)
variables (b a : ℝ)
noncomputable def R_squared := 1 - (∑ i, (y i - (b * x i + a))^2) / (∑ i, (y i - y_bar)^2)

theorem regression_incorrect_stmt_C 
  (H1 : ∀ a b, ∃ (i : Fin n), (b * (x i) + a = y_bar))
  (H2 : ∀ a b, ∃ (i : Fin n), ∑ i, (y i - (b * x i + a))^2 < ∑ i, (y i - y_bar)^2)
  (H3 : R_squared x y y_bar b a < 1) :
  (R_squared x y y_bar b a).toReal < 1 := by 
  sorry

end regression_incorrect_stmt_C_l811_811803


namespace find_a_l811_811041

theorem find_a 
  (a : ℝ) 
  (h1 : ∃ (α : ℝ), ∃ (x y : ℝ), x = 3 * a ∧ y = 4 ∧ cos α = -3/5 
    ∧ x^2 + y^2 = (sqrt ((3 * a)^2 + 4^2))^2) 
  : a = -1 :=
sorry

end find_a_l811_811041


namespace exists_distinct_pure_powers_l811_811547

-- Definitions and conditions
def is_pure_kth_power (k m : ℕ) : Prop := ∃ t : ℕ, m = t ^ k

-- The main theorem statement
theorem exists_distinct_pure_powers (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n → ℕ),
    (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧ 
    is_pure_kth_power 2009 (Finset.univ.sum a) ∧ 
    is_pure_kth_power 2010 (Finset.univ.prod a) :=
sorry

end exists_distinct_pure_powers_l811_811547


namespace bottles_per_crate_l811_811219

theorem bottles_per_crate (num_bottles total_bottles bottles_not_placed num_crates : ℕ) 
    (h1 : total_bottles = 130)
    (h2 : bottles_not_placed = 10)
    (h3 : num_crates = 10) 
    (h4 : num_bottles = total_bottles - bottles_not_placed) :
    (num_bottles / num_crates) = 12 := 
by 
    sorry

end bottles_per_crate_l811_811219


namespace peanuts_in_box_after_addition_l811_811415

theorem peanuts_in_box_after_addition : 4 + 12 = 16 := by
  sorry

end peanuts_in_box_after_addition_l811_811415


namespace anton_thought_number_l811_811887

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l811_811887


namespace max_teams_l811_811190

theorem max_teams (n : ℕ) (cond1 : ∀ t, card t = 3) (cond2 : ∀ t1 t2 (ht1 : t1 ≠ t2), 
  ∀ p1 ∈ t1, ∀ p2 ∈ t2, p1 ≠ p2) (cond3 : 9 * n * (n - 1) / 2 ≤ 200) : 
  n ≤ 7 := 
sorry

end max_teams_l811_811190


namespace volume_ratio_of_tetrahedrons_l811_811198

noncomputable def tetrahedron_volume_ratio : ℝ :=
let edge_length := 3 in
let smaller_tetrahedron_edge_length := edge_length / 3 in
(smaller_tetrahedron_edge_length^3) / (edge_length^3)

theorem volume_ratio_of_tetrahedrons : tetrahedron_volume_ratio = (1 / 27) :=
sorry

end volume_ratio_of_tetrahedrons_l811_811198


namespace no_adjacent_performers_probability_l811_811261

-- A definition to model the probability of non-adjacent performers in a circle of 6 people.
def probability_no_adjacent_performers : ℚ :=
  -- Given conditions: fair coin tosses by six people, modeling permutations
  -- and specific valid configurations derived from the problem.
  9 / 32

-- Proving the final probability calculation is correct
theorem no_adjacent_performers_probability :
  probability_no_adjacent_performers = 9 / 32 :=
by
  -- Using sorry to indicate the proof needs to be filled in, acknowledging the correct answer.
  sorry

end no_adjacent_performers_probability_l811_811261


namespace number_of_distinct_prime_factors_of_90_l811_811114

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811114


namespace measure_angle_BAC_is_45_l811_811709

-- This notation uses the standard R^2 space over reals to place points A, B, and C.
structure Point := (x : ℝ) (y: ℝ)

-- Define vertices of the triangle 
variables (A B C : Point)
-- Define the equality of the legs AB and BC
axiom equal_legs_AB_BC : dist A B = dist B C
-- Define right angle at vertex B
axiom right_angle_B : ∠ A B C = 90

-- Define the goal, i.e., the measure of the angle BAC is 45 degrees.
theorem measure_angle_BAC_is_45 : ∠ A B C = 90 → dist A B = dist B C → ∠ B A C = 45 :=
  by sorry

end measure_angle_BAC_is_45_l811_811709


namespace sum_of_GCF_and_LCM_l811_811361

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l811_811361


namespace sum_gcf_lcm_l811_811352

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l811_811352


namespace quadratic_has_real_roots_l811_811807

theorem quadratic_has_real_roots (k : ℝ) (h : k > 0) : ∃ x : ℝ, x^2 + 2 * x - k = 0 :=
by
  sorry

end quadratic_has_real_roots_l811_811807


namespace largest_inscribed_circle_radius_l811_811477

theorem largest_inscribed_circle_radius (k : ℝ) (h_perimeter : 0 < k) :
  ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2) :=
by
  have h_r : ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2)
  exact ⟨(k / 2) * (3 - 2 * Real.sqrt 2), rfl⟩
  exact h_r

end largest_inscribed_circle_radius_l811_811477


namespace anton_thought_of_729_l811_811869

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l811_811869


namespace tangent_lines_through_M_l811_811990

-- Define the point M
def M := (3 : ℝ, 1 : ℝ)

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 4

-- Define the equations of the lines to be tested
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 13 = 0
def line2 (x y : ℝ) : Prop := x = 3

-- Define the center of the circle and its radius
def center := (1 : ℝ, 0 : ℝ)
def radius := 2 : ℝ

-- Define the distance from a point to a line
def distance_point_line (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ :=
  -- The distance from point (px, py) to line ax + by + c = 0 is |a * px + b * py + c| / sqrt (a^2 + b^2)
  let (px, py) := p in
  let f {a b c : ℝ} (p : ℝ × ℝ) : ℝ := abs (a * px + b * py + c) / real.sqrt (a ^ 2 + b ^ 2) in
  if l = line1 then f (3, 4, -13)
  else if l = line2 then f (1, 0, -3)
  else 0

-- The Lean statement to prove
theorem tangent_lines_through_M : 
  (line1 M.1 M.2 ∧ circle (M.1 + 3/4) (M.2 - 3/4) ∧ distance_point_line center line1 = radius) ∨ 
  (line2 M.1 M.2 ∧ circle 3 1 ∧ distance_point_line center line2 = radius) :=
by
  sorry

end tangent_lines_through_M_l811_811990


namespace count_arithmetic_sequence_l811_811168

theorem count_arithmetic_sequence: 
  ∃ n : ℕ, (2 + (n - 1) * 3 = 2014) ∧ n = 671 := 
sorry

end count_arithmetic_sequence_l811_811168


namespace pin_sheets_l811_811708

universe u
variable {α : Type u}

-- Definitions of the conditions
structure Table where
  length : ℕ
  width : ℕ

structure Sheet where
  side_length : ℕ

-- The main theorem statement
theorem pin_sheets (T : Table) (sheets : list Sheet) :
  ∀ sheet ∈ sheets, ∃ pin : fin (T.length * T.width), true := 
by
  sorry

end pin_sheets_l811_811708


namespace partition_nat_three_sets_partition_nat_four_sets_not_partition_nat_three_sets_for_235_l811_811825

-- First part
theorem partition_nat_three_sets :
  ∃ (A B C : set ℕ), (∀ (n : ℕ), (n ∈ A ∨ n ∈ B ∨ n ∈ C)) ∧ (∀ m n, |m-n| = 2 ∨ |m-n| = 5 → 
  ¬ (m ∈ A ∧ n ∈ A) ∧ ¬ (m ∈ B ∧ n ∈ B) ∧ ¬ (m ∈ C ∧ n ∈ C)) :=
sorry

-- Second part
theorem partition_nat_four_sets :
  ∃ (A B C D : set ℕ), (∀ (n : ℕ), (n ∈ A ∨ n ∈ B ∨ n ∈ C ∨ n ∈ D)) ∧ (∀ m n, |m-n| = 2 ∨ |m-n| = 3 ∨ |m-n| = 5 → 
  ¬ (m ∈ A ∧ n ∈ A) ∧ ¬ (m ∈ B ∧ n ∈ B) ∧ ¬ (m ∈ C ∧ n ∈ C) ∧ ¬ (m ∈ D ∧ n ∈ D)) :=
sorry

-- Third part
theorem not_partition_nat_three_sets_for_235 :
  ¬ ∃ (A B C : set ℕ), (∀ (n : ℕ), (n ∈ A ∨ n ∈ B ∨ n ∈ C)) ∧ (∀ m n, |m-n| = 2 ∨ |m-n| = 3 ∨ |m-n| = 5 → 
  ¬ (m ∈ A ∧ n ∈ A) ∧ ¬ (m ∈ B ∧ n ∈ B) ∧ ¬ (m ∈ C ∧ n ∈ C)) :=
sorry

end partition_nat_three_sets_partition_nat_four_sets_not_partition_nat_three_sets_for_235_l811_811825


namespace balls_balance_l811_811327

theorem balls_balance (G Y W B : ℕ) (h1 : G = 2 * B) (h2 : Y = 5 * B / 2) (h3 : W = 3 * B / 2) :
  5 * G + 3 * Y + 3 * W = 22 * B :=
by
  sorry

end balls_balance_l811_811327


namespace number_of_distinct_prime_factors_90_l811_811147

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811147


namespace damage_in_usd_correct_l811_811849

def exchange_rate := (125 : ℚ) / 100
def damage_CAD := 45000000
def damage_USD := damage_CAD / exchange_rate

theorem damage_in_usd_correct (CAD_to_USD : exchange_rate = (125 : ℚ) / 100) (damage_in_cad : damage_CAD = 45000000) : 
  damage_USD = 36000000 :=
by
  sorry

end damage_in_usd_correct_l811_811849


namespace circumcenter_lies_on_AB_l811_811574

theorem circumcenter_lies_on_AB
  (A B C H E F : Type) 
  [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty H] [Nonempty E] [Nonempty F] 
  (ABC : A ∧ B ∧ C)
  (acute_angled_triangle : ∀ (α : Real), α ≤ 90)
  (AB_gt_AC : (AB : Real) > (AC : Real))
  (orthocenter_of_triangle : orthocenter H A B C)
  (reflection_of_C : reflect C AH = E)
  (intersection_F : line EH ∩ line AC = F) :
  circumcenter (triangle A E F) ∈ line AB := 
sorry

end circumcenter_lies_on_AB_l811_811574


namespace term_1005_l811_811001

variable (a : ℕ → ℕ)

-- Condition: For each positive integer n, the mean of the first n terms of a sequence is n + 1
def mean_condition := ∀ (n : ℕ), 0 < n → (∑ k in Finset.range (n + 1), a k) / n = n + 1

-- The goal is to prove that the 1005th term of the sequence is 2010
theorem term_1005 (h : mean_condition a) : a 1005 = 2010 := 
  sorry

end term_1005_l811_811001


namespace abs_sum_lt_ineq_l811_811176

theorem abs_sum_lt_ineq (x : ℝ) (a : ℝ) (h₀ : 0 < a) : 
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ (1 < a) :=
by
  sorry

end abs_sum_lt_ineq_l811_811176


namespace gcd_lcm_sum_8_12_l811_811399

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l811_811399


namespace anton_thought_number_l811_811900

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l811_811900


namespace anton_thought_number_is_729_l811_811890

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l811_811890


namespace exists_n_for_all_k_l811_811719

theorem exists_n_for_all_k (k : ℕ) : ∃ n : ℕ, 5^k ∣ (n^2 + 1) :=
sorry

end exists_n_for_all_k_l811_811719


namespace anton_thought_number_l811_811902

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l811_811902


namespace term_free_of_x_l811_811802

namespace PolynomialExpansion

theorem term_free_of_x (m n k : ℕ) (h : (x : ℝ)^(m * k - (m + n) * r) = 1) :
  (m * k) % (m + n) = 0 :=
by
  sorry

end PolynomialExpansion

end term_free_of_x_l811_811802


namespace sum_of_GCF_and_LCM_l811_811362

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l811_811362


namespace Tim_paid_amount_l811_811315

theorem Tim_paid_amount (original_price : ℝ) (discount_percentage : ℝ) (discounted_price : ℝ) 
    (h1 : original_price = 1200) (h2 : discount_percentage = 0.15) 
    (discount_amount : ℝ) (h3 : discount_amount = original_price * discount_percentage) 
    (h4 : discounted_price = original_price - discount_amount) : discounted_price = 1020 := 
    by {
        sorry
    }

end Tim_paid_amount_l811_811315


namespace quadrilateral_is_rhombus_l811_811459

-- Structure for a quadrilateral
structure Quadrilateral (V : Type) :=
(A B C D : V)

-- Define the conditions for the quadrilateral in the problem
structure CircumscribedQuadrilateral (V : Type) [MetricSpace V] :=
(q : Quadrilateral V)
(circle : Set V)
(center : V)
(diag_intersect_center : (q.A - q.C : V) = center ∧ (q.B - q.D : V) = center)
(circumscribed : ∀ p ∈ [q.A, q.B, q.C, q.D], metric.dist p center = metric.dist center circle)

-- Theorem statement that the quadrilateral is a rhombus
theorem quadrilateral_is_rhombus {V : Type} [MetricSpace V] (quad : CircumscribedQuadrilateral V) :
  ∀ e ∈ [quad.q.A, quad.q.B, quad.q.C, quad.q.D], metric.dist quad.q.A quad.q.B = metric.dist quad.q.A quad.q.D ∧
                                                   metric.dist quad.q.B quad.q.C = metric.dist quad.q.C quad.q.D ∧
                                                   metric.dist quad.q.A quad.q.D = metric.dist quad.q.C quad.q.D :=
sorry

end quadrilateral_is_rhombus_l811_811459


namespace sum_of_GCF_and_LCM_l811_811365

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l811_811365


namespace exists_100_integers_with_distinct_pairwise_sums_l811_811213

-- Define number of integers and the constraint limit
def num_integers : ℕ := 100
def max_value : ℕ := 25000

-- Define the predicate for all pairwise sums being different
def pairwise_different_sums (as : Fin num_integers → ℕ) : Prop :=
  ∀ i j k l : Fin num_integers, i ≠ j ∧ k ≠ l → as i + as j ≠ as k + as l

-- Main theorem statement
theorem exists_100_integers_with_distinct_pairwise_sums :
  ∃ as : Fin num_integers → ℕ, (∀ i : Fin num_integers, as i > 0 ∧ as i ≤ max_value) ∧ pairwise_different_sums as :=
sorry

end exists_100_integers_with_distinct_pairwise_sums_l811_811213


namespace modulus_of_z_l811_811033

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := -1 + 3 * i

theorem modulus_of_z : complex.abs z = real.sqrt 10 :=
by
  sorry

end modulus_of_z_l811_811033


namespace delivery_truck_speed_l811_811471

theorem delivery_truck_speed :
  ∀ d t₁ t₂: ℝ,
    (t₁ = 15 / 60) ∧ (t₂ = -15 / 60) ∧ 
    (t₁ = d / 20 - 1 / 4) ∧ (t₂ = d / 60 + 1 / 4) →
    (d = 15) →
    (t = 1 / 2) →
    ( ∃ v: ℝ, t = d / v ∧ v = 30 ) :=
by sorry

end delivery_truck_speed_l811_811471


namespace Anton_thought_number_is_729_l811_811911

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l811_811911


namespace Anton_thought_of_729_l811_811918

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l811_811918


namespace number_of_valid_pairs_l811_811553

def valid_pairs_count (n : ℕ) : Prop :=
  n = 3

theorem number_of_valid_pairs : ∃ n, valid_pairs_count n :=
by {
  use 3,
  sorry
}

end number_of_valid_pairs_l811_811553


namespace positive_integers_not_in_E_are_perfect_squares_l811_811515

open Set

def E : Set ℕ := {m | ∃ n : ℕ, m = Int.floor (n + Real.sqrt n + 0.5)}

theorem positive_integers_not_in_E_are_perfect_squares (m : ℕ) (h_pos : 0 < m) :
  m ∉ E ↔ ∃ t : ℕ, m = t^2 := 
by
    sorry

end positive_integers_not_in_E_are_perfect_squares_l811_811515


namespace greatest_difference_l811_811305

-- Definitions: Number of marbles in each basket
def basketA_red : Nat := 4
def basketA_yellow : Nat := 2
def basketB_green : Nat := 6
def basketB_yellow : Nat := 1
def basketC_white : Nat := 3
def basketC_yellow : Nat := 9

-- Define the differences
def diff_basketA : Nat := basketA_red - basketA_yellow
def diff_basketB : Nat := basketB_green - basketB_yellow
def diff_basketC : Nat := basketC_yellow - basketC_white

-- The goal is to prove that 6 is the greatest difference
theorem greatest_difference : max (max diff_basketA diff_basketB) diff_basketC = 6 :=
by 
  -- The proof is not provided
  sorry

end greatest_difference_l811_811305


namespace sum_of_digits_repeated_l811_811688

def sum_of_digits (n : ℕ) : ℕ :=
  nat.digits 10 n |>.sum

theorem sum_of_digits_repeated (n : ℕ) :
  let f := sum_of_digits in
  f (f (f (4444 ^ 4444))) = 7 :=
by
  let f := sum_of_digits
  have h : 4444 % 9 = 7 := by sorry
  have h2 : (4444 ^ 4444) % 9 = 7 := by sorry
  exact sorry

end sum_of_digits_repeated_l811_811688


namespace anton_thought_of_729_l811_811866

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l811_811866


namespace sum_gcd_lcm_eight_twelve_l811_811379

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l811_811379


namespace other_asymptote_l811_811711

theorem other_asymptote (a b : ℝ) :
  (∀ x y : ℝ, y = 2 * x → y - b = a * (x - (-4))) ∧
  (∀ c d : ℝ, c = -4) →
  ∃ m b' : ℝ, m = -1/2 ∧ b' = -10 ∧ ∀ x y : ℝ, y = m * x + b' :=
by
  sorry

end other_asymptote_l811_811711


namespace fraction_sum_l811_811680

-- Define the given conditions in Lean 4
variables {a b c x y z : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z)
variables (h7 : a^2 + b^2 + c^2 = 10)
variables (h8 : x^2 + y^2 + z^2 = 40)
variables (h9 : a * x + b * y + c * z = 20)

-- Goal: Prove the given equality
theorem fraction_sum (a b c x y z : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < x) (h5 : 0 < y)
  (h6 : 0 < z) (h7 : a^2 + b^2 + c^2 = 10) (h8 : x^2 + y^2 + z^2 = 40) (h9 : a * x + b * y + c * z = 20) :
  (a + b + c) / (x + y + z) = 1 / 2 :=
begin
  sorry
end

end fraction_sum_l811_811680


namespace investment_plans_correct_l811_811430

noncomputable def num_investment_plans : ℕ := 
  (Nat.choose 4 2 * 3.factorial) + 4.factorial / ((4 - 3).factorial)

theorem investment_plans_correct : num_investment_plans = 60 :=
by
  sorry

end investment_plans_correct_l811_811430


namespace nat_divides_2_pow_n_minus_1_l811_811530

theorem nat_divides_2_pow_n_minus_1 (n : ℕ) (hn : 0 < n) : n ∣ 2^n - 1 ↔ n = 1 :=
  sorry

end nat_divides_2_pow_n_minus_1_l811_811530


namespace nordica_population_increase_l811_811208

/-- In the state of Nordica, it is estimated there is a baby born every 6 hours,
there is a death every 2 days, additionally, there are 10 immigrants arriving every week.
To the nearest hundred, how many people are added to the population of Nordica each year? -/
theorem nordica_population_increase : 
  let births_per_day : ℝ := 24 / 6
  let deaths_per_day : ℝ := 1 / 2
  let immigrants_per_day : ℝ := 10 / 7
  let net_daily_increase : ℝ := births_per_day - deaths_per_day + immigrants_per_day
  let yearly_increase : ℝ := net_daily_increase * 365
  round(yearly_increase / 100) * 100 = 1800 := 
by 
  sorry

end nordica_population_increase_l811_811208


namespace alex_age_proof_l811_811855

theorem alex_age_proof : 
  ∃ X : ℝ, 
    let A := 16.9996700066 in
    let F := 2 * A + 5 in
    X = 6.4998350033 ∧ (A - X = 1 / 3 * (F - X)) :=
by
  let A := 16.9996700066
  let F := 2 * A + 5
  have h1 : F = 38.9993400132 := by norm_num
  have h2 : A - 6.4998350033 = 1 / 3 * (F - 6.4998350033) := by norm_num
  exact ⟨6.4998350033, h1, h2⟩

end alex_age_proof_l811_811855


namespace find_radius_of_circle_l811_811428

def isosceles_triangle_radius (base : ℝ) (height : ℝ) (radius : ℝ) : Prop :=
  let b := base / 2
  let h := height
  let r := radius
  ∃ (A B C : ℝ × ℝ) (O : ℝ × ℝ), 
    -- Define A, B, C as the vertices of the triangle
    A = (0, height)  ∧ 
    B = (-b, 0)  ∧ 
    C = (b, 0)  ∧ 
    -- Define O as the center of the circle with radius
    O = (0, height - radius) ∧ 
    -- Ensure the circle is tangent to the legs of the triangle
    real.dist O B = radius ∧ 
    real.dist O C = radius

theorem find_radius_of_circle (base height : ℝ) : 
  base = 8 →
  height = 3 →
  ∃ radius : ℝ, isosceles_triangle_radius base height radius ∧ radius = 20 / 3 := 
by
  intros
  sorry

end find_radius_of_circle_l811_811428


namespace sum_gcd_lcm_eight_twelve_l811_811380

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l811_811380


namespace count_valid_combinations_l811_811401

-- Define a function to verify combinations
def valid_combination (nums : list ℕ) : Prop :=
  nums.sum = 10 ∧ nums.length = 4 ∧ ∀ n ∈ nums, n = 1 ∨ n = 2 ∨ n = 3

-- Define the main theorem
theorem count_valid_combinations : 
  {nums : list ℕ | valid_combination nums}.to_finset.card = 10 :=
sorry

end count_valid_combinations_l811_811401


namespace greatest_difference_l811_811304

-- Definitions: Number of marbles in each basket
def basketA_red : Nat := 4
def basketA_yellow : Nat := 2
def basketB_green : Nat := 6
def basketB_yellow : Nat := 1
def basketC_white : Nat := 3
def basketC_yellow : Nat := 9

-- Define the differences
def diff_basketA : Nat := basketA_red - basketA_yellow
def diff_basketB : Nat := basketB_green - basketB_yellow
def diff_basketC : Nat := basketC_yellow - basketC_white

-- The goal is to prove that 6 is the greatest difference
theorem greatest_difference : max (max diff_basketA diff_basketB) diff_basketC = 6 :=
by 
  -- The proof is not provided
  sorry

end greatest_difference_l811_811304


namespace number_of_valid_four_digit_numbers_l811_811558

def digits : Set ℕ := {0, 1, 2, 3, 4, 5}

def is_four_digit_without_repeating (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (n.digits : List ℕ).nodup

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem number_of_valid_four_digit_numbers :
  (#{n : ℕ | ∃ d ⊆ digits, d.card = 4 ∧ is_four_digit_without_repeating n ∧ is_divisible_by_3 n}).card = 96 :=
sorry

end number_of_valid_four_digit_numbers_l811_811558


namespace quadratic_properties_l811_811723

theorem quadratic_properties (a b c : ℝ) (h : a ≠ 0) (f : ℝ → ℝ) (hf : ∀ x, f x = a * x^2 + b * x + c) :
  let Δ := b^2 - 4 * a * c in
  (a > 0) ∧ (f 0 = c) ∧ (Δ < 0) ∧ (∀ x, x < -b / (2 * a) → f' x < 0) ↔ B := by
  sorry
    where B := "intersects the y-axis below the x-axis (i.e. f(0) < 0)" is incorrect

end quadratic_properties_l811_811723


namespace sum_gcf_lcm_eq_28_l811_811370

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l811_811370


namespace passengers_taken_at_first_station_l811_811470

def passengers_initial := 270
def passengers_third_station := 242

def drop_at_first_station (initial : ℕ) := initial - initial/3
def take_at_first_station (n : ℕ) := n
def drop_at_second_station (result_first : ℕ) := result_first/2
def take_at_second_station (result_second : ℕ) := 12

theorem passengers_taken_at_first_station (x : ℕ) :
  passengers_third_station = let after_first_drop := drop_at_first_station passengers_initial;
                                 after_first_take := after_first_drop + take_at_first_station x;
                                 after_second_drop := after_first_take - drop_at_second_station after_first_take;
                             in after_second_drop + take_at_second_station after_second_drop
:= sorry

end passengers_taken_at_first_station_l811_811470


namespace distinct_prime_factors_90_l811_811143

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811143


namespace pyramid_plane_intersection_area_l811_811468

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Pyramid where
  A B C D E : Point
  edge_length : ℝ

def midpoint (p1 p2 : Point) : Point := {
  x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2,
  z := (p1.z + p2.z) / 2,
}

def plane_eq (p1 p2 p3 : Point) : Point → Prop := sorry

noncomputable def pyramid_intersection_area (p : Pyramid) : ℝ :=
  let A := p.A
  let E := p.E
  let AE_mid := midpoint A E
  let AB_mid := midpoint A p.B
  let CD_mid := midpoint p.C p.D
  let area := 4.5 * Real.sqrt 3
  area

theorem pyramid_plane_intersection_area (p : Pyramid) (h : p.edge_length = 6) :
    pyramid_intersection_area p = 4.5 * Real.sqrt 3 :=
  sorry

end pyramid_plane_intersection_area_l811_811468


namespace evaluate_expression_at_b_eq_3_l811_811800

theorem evaluate_expression_at_b_eq_3 : 
  (∀ (b : ℝ), b = 3 → (3*b^(-2) + (b^(-2))/3) / b^2 = 10/243) :=
by
  intro b
  intro hb
  rw [hb]
  sorry
 
end evaluate_expression_at_b_eq_3_l811_811800


namespace tom_age_ratio_l811_811784

-- Definitions of the variables
variables (T : ℕ) (N : ℕ)

-- Conditions given in the problem
def condition1 : Prop := T = 2 * (T / 2)
def condition2 : Prop := (T - 3) = 3 * (T / 2 - 12)

-- The ratio theorem to prove
theorem tom_age_ratio (h1 : condition1 T) (h2 : condition2 T) : T / N = 22 :=
by
  sorry

end tom_age_ratio_l811_811784


namespace school_sample_proof_l811_811317

open Probability

noncomputable def classes_in_schools : ℕ → ℕ
| 0 => 12  -- School A
| 1 => 6   -- School B
| 2 => 18  -- School C
| _ => 0

def total_classes : ℕ := classes_in_schools 0 + classes_in_schools 1 + classes_in_schools 2

def sampled_classes : ℕ := 6

def sample_proportion (school_index : ℕ) : ℚ :=
  (classes_in_schools school_index : ℚ) / (total_classes)

noncomputable def number_of_sampled_classes (school_index : ℕ) : ℕ :=
  (sampled_classes * sample_proportion school_index).to_nat

def random_selection : List ℕ := [number_of_sampled_classes 0, number_of_sampled_classes 1, number_of_sampled_classes 2]

def all_possible_outcomes : List (ℕ × ℕ) :=
  [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2), 
   (2,3), (3,1), (3,2), (2,3), (3,3), (1,3), (1,2)]

def event_D : List (ℕ × ℕ) :=
  [(0,0), (0,1), (0,2), (0,3), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2)]

noncomputable def probability_D : ℚ :=
  event_D.length / all_possible_outcomes.length

theorem school_sample_proof :
  number_of_sampled_classes 0 = 2 ∧ number_of_sampled_classes 1 = 1 ∧ number_of_sampled_classes 2 = 3 ∧
  probability_D = 3 / 5 :=
by
  sorry

end school_sample_proof_l811_811317


namespace largest_digit_M_l811_811795

-- Define the conditions as Lean types
def digit_sum_divisible_by_3 (M : ℕ) := (4 + 5 + 6 + 7 + M) % 3 = 0
def even_digit (M : ℕ) := M % 2 = 0

-- Define the problem statement in Lean
theorem largest_digit_M (M : ℕ) (h : even_digit M ∧ digit_sum_divisible_by_3 M) : M ≤ 8 ∧ (∀ N : ℕ, even_digit N ∧ digit_sum_divisible_by_3 N → N ≤ M) :=
sorry

end largest_digit_M_l811_811795


namespace sum_of_distinct_three_digit_numbers_using_1_3_5_l811_811959

theorem sum_of_distinct_three_digit_numbers_using_1_3_5 :
  let numbers := [135, 153, 315, 351, 513, 531]
  ∑ n in numbers, n = 1998 :=
by
  sorry

end sum_of_distinct_three_digit_numbers_using_1_3_5_l811_811959


namespace positive_integer_divisors_count_l811_811540

theorem positive_integer_divisors_count :
  ∃ k : ℕ, k = 16 ∧ ∀ n : ℕ, 0 < n → (n - 1) ∣ (n + 2 * n^2 + 3 * n^3 + ∀ i in Ico 4 2006, (i : ℕ) * n^i) := sorry

end positive_integer_divisors_count_l811_811540


namespace problem_statement_l811_811599

-- Define the main functions and properties
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

-- Statement of the problem
theorem problem_statement (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x ↔ a ≤ -2) ∧
  (∀ x : ℝ, x ∈ set.Icc 0 2 → h a x ≤ if a ≥ -3 then a + 3 else 0) :=
by
  -- Proof goes here
  sorry

end problem_statement_l811_811599


namespace anton_thought_number_l811_811881

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l811_811881


namespace abs_eq_sum_solutions_l811_811832

theorem abs_eq_sum_solutions (x : ℝ) : (|3*x - 2| + |3*x + 1| = 3) ↔ 
  (x = -1 / 3 ∨ (-1 / 3 < x ∧ x <= 2 / 3)) :=
by
  sorry

end abs_eq_sum_solutions_l811_811832


namespace geometric_sequence_product_max_l811_811017

theorem geometric_sequence_product_max (a : ℕ → ℝ) (q : ℝ) 
  (h_seq_geometric : ∀ n, a (n + 1) = a n * q)
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 2 + a 4 = 5) :
  ∃ n, n = 3 ∨ n = 4 ∧ a 1 * a 2 * ⋯ * a n = 64 := 
by
  sorry

end geometric_sequence_product_max_l811_811017


namespace roots_of_polynomial_l811_811541

-- Define the polynomial
def polynomial : Polynomial ℝ :=
  8 * Polynomial.X ^ 4 + 26 * Polynomial.X ^ 3 - 65 * Polynomial.X ^ 2 + 24 * Polynomial.X

-- Define the theorem to prove the roots of the polynomial
theorem roots_of_polynomial : Polynomial.roots polynomial = {0, 1/2, 3/2, -4} :=
  sorry

end roots_of_polynomial_l811_811541


namespace expected_number_of_different_faces_l811_811441

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l811_811441


namespace height_difference_l811_811651

-- Define the initial height of James's uncle
def uncle_height : ℝ := 72

-- Define the initial height ratio of James compared to his uncle
def james_initial_height_ratio : ℝ := 2 / 3

-- Define the height gained by James from his growth spurt
def james_growth_spurt : ℝ := 10

-- Define the initial height of James before the growth spurt
def james_initial_height : ℝ := uncle_height * james_initial_height_ratio

-- Define the new height of James after the growth spurt
def james_new_height : ℝ := james_initial_height + james_growth_spurt

-- Theorem: The difference in height between James's uncle and James after the growth spurt is 14 inches
theorem height_difference : uncle_height - james_new_height = 14 := sorry

end height_difference_l811_811651


namespace negation_of_exists_ellipse_with_eccentricity_lt_1_l811_811249

namespace EccentricityProof

-- Define the proposition "There exists an ellipse with eccentricity e < 1"
def exists_ellipse_with_eccentricity_lt_1 : Prop :=
  ∃ e : ℝ, e < 1 ∧ is_ellipse e

-- Define what it means to be an ellipse
def is_ellipse (e : ℝ) : Prop := -- placeholder definition for ellipses
  sorry

-- Prove the negation of the proposition
theorem negation_of_exists_ellipse_with_eccentricity_lt_1 :
  ¬exists_ellipse_with_eccentricity_lt_1 ↔ ∀ (e : ℝ), is_ellipse e → e ≥ 1 :=
by
  sorry

end EccentricityProof

end negation_of_exists_ellipse_with_eccentricity_lt_1_l811_811249


namespace nikita_claim_is_incorrect_l811_811811

theorem nikita_claim_is_incorrect (x y : ℤ) (hx : 9*x + 4*y - (4*x + 9*y) = 49) : false :=
by
  have h1 : 5 * (x - y) = 49 := by linarith
  have h2 : 49 % 5 ≠ 0 := by norm_num
  have h3 : 5 ∣ 49 := by
    rw ←h1
    exact dvd_rfl
  contradiction

end nikita_claim_is_incorrect_l811_811811


namespace sum_of_gcd_and_lcm_is_28_l811_811385

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l811_811385


namespace distinct_prime_factors_90_l811_811138

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811138


namespace transformed_triangle_area_l811_811742

variables {x₁ x₂ x₃ : ℝ}
variable {f : ℝ → ℝ}
variable (area_original : ℝ)

-- conditions
def points_form_triangle (x₁ x₂ x₃ : ℝ) (f : ℝ → ℝ) := 
  set.pairwise {x₁, x₂, x₃} (λ x y, x ≠ y)} ∧
  collinear { (x₁, f x₁), (x₂, f x₂), (x₃, f x₃) }

-- This definition assumes an area function for a set of points
def triangle_area (pts : set (ℝ × ℝ)) := sorry

-- Given conditions
axiom H1 : points_form_triangle x₁ x₂ x₃ f
axiom H2 : triangle_area {(x₁, f x₁), (x₂, f x₂), (x₃, f x₃)} = 32

-- The problem to prove
theorem transformed_triangle_area :
  triangle_area {(x₁ / 2, 2 * f (x₁)), (x₂ / 2, 2 * f (x₂)), (x₃ / 2, 2 * f (x₃))} = 32 :=
sorry

end transformed_triangle_area_l811_811742


namespace num_distinct_prime_factors_90_l811_811127

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811127


namespace expected_number_of_different_faces_l811_811438

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l811_811438


namespace cube_root_solutions_l811_811234

theorem cube_root_solutions (p : ℕ) (hp : p > 3) :
    (∃ (k : ℤ) (h1 : k^2 ≡ -3 [ZMOD p]), ∀ x, x^3 ≡ 1 [ZMOD p] → 
        (x = 1 ∨ (x^2 + x + 1 ≡ 0 [ZMOD p])) )
    ∨ 
    (∀ x, x^3 ≡ 1 [ZMOD p] → x = 1) := 
sorry

end cube_root_solutions_l811_811234


namespace correct_statements_l811_811809

-- Let S1, S2, S3, S4 represent the given statements.
def S1 : Prop := "The larger the R^2, the smaller the residual sum of squares, indicating a better fitting effect."
def S2 : Prop := "Inferring the properties of a sphere from those of a circle is an example of analogical reasoning."
def S3 : Prop := "It is impossible to compare the magnitude of any two complex numbers."
def S4 : Prop := "Flowcharts can have multiple endpoints."

-- Assertion that statements ① and ② are correct, ③ and ④ are incorrect.
theorem correct_statements :
  (S1 = true) ∧ (S2 = true) ∧ (S3 = false) ∧ (S4 = false) :=
sorry

end correct_statements_l811_811809


namespace number_of_distinct_prime_factors_of_90_l811_811076

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811076


namespace intersection_point_of_line_and_plane_l811_811538

noncomputable def line (t : ℝ) : ℝ × ℝ × ℝ := (2 + t, 3 + t, 2 * t + 4)

def plane (p : ℝ × ℝ × ℝ) : Prop := 2 * p.1 + p.2 + p.3 = 0

theorem intersection_point_of_line_and_plane :
  ∃ t : ℝ, plane (line t) ∧ line t = (-0.2, 0.8, -0.4) :=
by
  exists -2.2
  unfold line plane
  simp
  sorry  -- Needs proof but the statement satisfies the requirements.

end intersection_point_of_line_and_plane_l811_811538


namespace second_player_win_strategy_l811_811787

theorem second_player_win_strategy {n : ℕ} : 
  (∃ k : ℕ, n = 4 + 3*k) ↔ 
  (∀ n, ∀ colored_sides : set (fin n), 
    ∀ first_player_colorable_sides : set (fin n) → fin n → Prop, 
    ∀ second_player_colorable_sides : set (fin n) → fin n → Prop,
    (∀ s ∈ first_player_colorable_sides, (finset.card (colored_sides ∩ s.faces) = 0 ∨ finset.card (colored_sides ∩ s.faces) = 2)) →
    (∀ t ∈ second_player_colorable_sides, finset.card (colored_sides ∩ t.faces) = 1) →
    (∀ p1 ∈ (first_player_colorable_sides ∩ second_player_colorable_sides), false) →
    (∀ uncolored_sides : set (fin n), 
      ∀ s ∈ uncolored_sides, ∃ first_move ∈ uncolored_sides, 
      (∀ second_move ∈ uncolored_sides, 
       wins_second_player n (colored_sides ∪ {first_move} ∪ {second_move})))) :=
begin
  sorry
end

end second_player_win_strategy_l811_811787


namespace expected_faces_rolled_six_times_l811_811445

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l811_811445


namespace uncle_taller_by_14_l811_811658

namespace height_problem

def uncle_height : ℝ := 72
def james_height_before_spurt : ℝ := (2 / 3) * uncle_height
def growth_spurt : ℝ := 10
def james_height_after_spurt : ℝ := james_height_before_spurt + growth_spurt

theorem uncle_taller_by_14 : uncle_height - james_height_after_spurt = 14 := by
  sorry

end height_problem

end uncle_taller_by_14_l811_811658


namespace vincent_earnings_l811_811337

-- Definitions based on the problem conditions
def fantasy_book_cost : ℕ := 4
def literature_book_cost : ℕ := fantasy_book_cost / 2
def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def duration : ℕ := 5

-- Calculation functions
def daily_earnings_from_fantasy_books : ℕ := fantasy_books_sold_per_day * fantasy_book_cost
def daily_earnings_from_literature_books : ℕ := literature_books_sold_per_day * literature_book_cost
def total_daily_earnings : ℕ := daily_earnings_from_fantasy_books + daily_earnings_from_literature_books
def total_earnings_after_five_days : ℕ := total_daily_earnings * duration

-- Statement to prove
theorem vincent_earnings : total_earnings_after_five_days = 180 := 
by
  calc total_daily_earnings * duration = 180 : sorry

end vincent_earnings_l811_811337


namespace trapezoid_ratio_l811_811929

theorem trapezoid_ratio (A B C D E : Point) (F : Line) 
  (h1 : IsoscelesTrapezoid A B C D)
  (h2 : Parallel A B C D)
  (h3 : Length A B = 2 * Length C D)
  (h4 : Angle A = 60°)
  (h5 : PointOn E (Base A B))
  (h6 : Distance F E = Distance F B)
  (h7 : Distance F B = Distance A C)
  (h8 : Distance F A = Distance A B) :
  Ratio (A E) (E B) = 1 / 3 :=
sorry

end trapezoid_ratio_l811_811929


namespace fraction_equals_2025_l811_811519

-- Define the numerator as a sum
def numerator : ℚ :=
  (Finset.range 2024).sum (λ k, (2025 - (k + 1)) / (k + 1))

-- Define the denominator as a harmonic sum
def denominator : ℚ :=
  Finset.range 2024 |>.image (λ k, k + 2) |>.sum (λ k, 1 / k)

-- State the theorem that the quotient is 2025
theorem fraction_equals_2025 : numerator / denominator = 2025 :=
  sorry

end fraction_equals_2025_l811_811519


namespace women_attended_l811_811487

theorem women_attended :
  (15 * 4) / 3 = 20 :=
by
  sorry

end women_attended_l811_811487


namespace transformed_triangle_area_l811_811743

variables {x₁ x₂ x₃ : ℝ}
variable {f : ℝ → ℝ}
variable (area_original : ℝ)

-- conditions
def points_form_triangle (x₁ x₂ x₃ : ℝ) (f : ℝ → ℝ) := 
  set.pairwise {x₁, x₂, x₃} (λ x y, x ≠ y)} ∧
  collinear { (x₁, f x₁), (x₂, f x₂), (x₃, f x₃) }

-- This definition assumes an area function for a set of points
def triangle_area (pts : set (ℝ × ℝ)) := sorry

-- Given conditions
axiom H1 : points_form_triangle x₁ x₂ x₃ f
axiom H2 : triangle_area {(x₁, f x₁), (x₂, f x₂), (x₃, f x₃)} = 32

-- The problem to prove
theorem transformed_triangle_area :
  triangle_area {(x₁ / 2, 2 * f (x₁)), (x₂ / 2, 2 * f (x₂)), (x₃ / 2, 2 * f (x₃))} = 32 :=
sorry

end transformed_triangle_area_l811_811743


namespace largest_digit_M_divisible_by_six_l811_811792

theorem largest_digit_M_divisible_by_six :
  (∃ M : ℕ, M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ ∀ m : ℕ, m ≤ M → (45670 + m) % 6 ≠ 0) :=
sorry

end largest_digit_M_divisible_by_six_l811_811792


namespace polyhedron_volume_correct_l811_811300

noncomputable def volume_of_polyhedron : ℝ :=
  let side_length : ℝ := 12 in
  let cube_volume := side_length ^ 3 in
  (1 / 2) * cube_volume

theorem polyhedron_volume_correct :
  volume_of_polyhedron = 864 := by
  sorry

end polyhedron_volume_correct_l811_811300


namespace total_red_yellow_black_l811_811008

/-- Calculate the total number of red, yellow, and black shirts Gavin has,
given that he has 420 shirts in total, 85 of them are blue, and 157 are
green. -/
theorem total_red_yellow_black (total_shirts : ℕ) (blue_shirts : ℕ) (green_shirts : ℕ) :
  total_shirts = 420 → blue_shirts = 85 → green_shirts = 157 → 
  (total_shirts - (blue_shirts + green_shirts) = 178) :=
by
  intros h1 h2 h3
  sorry

end total_red_yellow_black_l811_811008


namespace smallest_perimeter_circle_circle_with_center_on_line_l811_811837

-- Definitions for conditions:
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (-1, 4)
def line : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), 2 * p.1 - p.2 - 4 = 0

-- Problem statements:
theorem smallest_perimeter_circle :
  ∃ (a b r : ℝ), (a = 0 ∧ b = 1 ∧ r = sqrt 10) ∧
  ∀ (p : ℝ × ℝ), (p = A ∨ p = B) → (p.1 - a)^2 + (p.2 - b)^2 = r^2 :=
sorry

theorem circle_with_center_on_line :
  ∃ (a b r : ℝ), (a = 3 ∧ b = 2 ∧ r = 2 * sqrt 5) ∧ 
  ∀ (p : ℝ × ℝ), (p = A ∨ p = B) → (p.1 - a)^2 + (p.2 - b)^2 = r^2 :=
sorry

end smallest_perimeter_circle_circle_with_center_on_line_l811_811837


namespace geometry_inequality_l811_811229

variable {V : Type} [inner_product_space ℝ V]

theorem geometry_inequality (A B C P : V)
  (h : ¬ collinear ({A, B, C} : set V)) :
  (dist P A / dist B C) + 
  (dist P B / dist C A) + 
  (dist P C / dist A B) ≥ sqrt 3 := 
sorry

end geometry_inequality_l811_811229


namespace inflection_point_l811_811212

noncomputable def f : ℝ → ℝ := λ x => x^3 - 3 * x^2 + 5

theorem inflection_point :
  ∃ x y, x = 1 ∧ y = 3 ∧ (∀ z < x, deriv^2 f z < 0) ∧ (∀ z > x, deriv^2 f z > 0)
  ∧ f 1 = 3 := 
sorry

end inflection_point_l811_811212


namespace num_distinct_prime_factors_90_l811_811126

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811126


namespace recursive_relation_a_recursive_relation_b_recursive_relation_c_recursive_relation_d_recursive_relation_e_l811_811058

def a (n : ℕ) : ℕ := n
def b (n : ℕ) : ℕ := n^2
def c (n : ℕ) : ℕ := n^3
def d (n : ℕ) : ℕ := n^4
def e (n : ℕ) : ℕ := n^5

theorem recursive_relation_a (n : ℕ) : a (n+2) = 2 * a (n+1) - a n :=
by sorry

theorem recursive_relation_b (n : ℕ) : b (n+3) = 3 * b (n+2) - 3 * b (n+1) + b n :=
by sorry

theorem recursive_relation_c (n : ℕ) : c (n+4) = 4 * c (n+3) - 6 * c (n+2) + 4 * c (n+1) - c n :=
by sorry

theorem recursive_relation_d (n : ℕ) : d (n+5) = 5 * d (n+4) - 10 * d (n+3) + 10 * d (n+2) - 5 * d (n+1) + d n :=
by sorry

theorem recursive_relation_e (n : ℕ) : 
  e (n+6) = 6 * e (n+5) - 15 * e (n+4) + 20 * e (n+3) - 15 * e (n+2) + 6 * e (n+1) - e n :=
by sorry

end recursive_relation_a_recursive_relation_b_recursive_relation_c_recursive_relation_d_recursive_relation_e_l811_811058


namespace sum_of_cubes_l811_811267

theorem sum_of_cubes (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x * y + y * z + z * x = -3) (h3 : x * y * z = 2) : 
  x^3 + y^3 + z^3 = 32 := 
sorry

end sum_of_cubes_l811_811267


namespace greatest_difference_in_baskets_l811_811309

theorem greatest_difference_in_baskets :
  let A_red := 4
  let A_yellow := 2
  let B_green := 6
  let B_yellow := 1
  let C_white := 3
  let C_yellow := 9
  max (abs (A_red - A_yellow)) (max (abs (B_green - B_yellow)) (abs (C_white - C_yellow))) = 6 :=
by
  sorry

end greatest_difference_in_baskets_l811_811309


namespace number_of_distinct_prime_factors_90_l811_811145

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811145


namespace max_area_of_triangle_PAB_l811_811015

noncomputable def point := ℝ × ℝ

def some_line (p: point) (q: point) : Prop := ∃ k b, ∀ x y, y = k * x + b ↔ (x, y) = p ∨ (x, y) = q

def circle_eq_center_and_radius (C: point) (r: ℝ) (x y: ℝ) : Prop :=
(x - C.1)^2 + (y - C.2)^2 = r^2

def distance (p q: point) : ℝ := real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

theorem max_area_of_triangle_PAB:
∀ (A B P: point),
A = (-1, 0) →
B = (3, 4) →
(∃ C, (C.1 + 3 * C.2 - 15 = 0) ∧ circle_eq_center_and_radius C (2 * real.sqrt 10) P.1 P.2) →
((x - (-3))^2 + (y - 6)^2 = 40) →
(∀ P ∈ circle, let area := 1 / 2 * distance A B * (distance A P + 2 * real.sqrt 10) in
area ≤ 16 + 8 * real.sqrt 5) :=
sorry

end max_area_of_triangle_PAB_l811_811015


namespace theta_eq_pi_div_4_l811_811181
noncomputable def y_shifted (θ : ℝ) : ℝ → ℝ :=
  λ x, sin (2 * (x + θ))

theorem theta_eq_pi_div_4 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) (h_sym : ∀ x, y_shifted θ x = y_shifted θ (-x)) : θ = π / 4 :=
sorry

end theta_eq_pi_div_4_l811_811181


namespace vincent_earnings_after_5_days_l811_811329

def fantasy_book_price : ℕ := 4
def daily_fantasy_books_sold : ℕ := 5
def literature_book_price : ℕ := fantasy_book_price / 2
def daily_literature_books_sold : ℕ := 8
def days : ℕ := 5

def daily_earnings : ℕ :=
  (fantasy_book_price * daily_fantasy_books_sold) +
  (literature_book_price * daily_literature_books_sold)

def total_earnings (d : ℕ) : ℕ :=
  daily_earnings * d

theorem vincent_earnings_after_5_days : total_earnings days = 180 := by
  sorry

end vincent_earnings_after_5_days_l811_811329


namespace term_1005_l811_811000

variable (a : ℕ → ℕ)

-- Condition: For each positive integer n, the mean of the first n terms of a sequence is n + 1
def mean_condition := ∀ (n : ℕ), 0 < n → (∑ k in Finset.range (n + 1), a k) / n = n + 1

-- The goal is to prove that the 1005th term of the sequence is 2010
theorem term_1005 (h : mean_condition a) : a 1005 = 2010 := 
  sorry

end term_1005_l811_811000


namespace expected_number_of_different_faces_l811_811449

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l811_811449


namespace distinct_prime_factors_of_90_l811_811103

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811103


namespace number_of_distinct_prime_factors_of_90_l811_811078

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811078


namespace arithmetic_sequence_length_correct_l811_811165

noncomputable def arithmetic_sequence_length (a d last_term : ℕ) : ℕ :=
  ((last_term - a) / d) + 1

theorem arithmetic_sequence_length_correct :
  arithmetic_sequence_length 2 3 2014 = 671 :=
by
  sorry

end arithmetic_sequence_length_correct_l811_811165


namespace seryozha_missing_number_l811_811727

theorem seryozha_missing_number :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}.sum in
  ∃ (x : ℕ) (H : x ∈ {1, 2, 3, 4, 5, 6, 7, 8}),
      (S - x) % 5 = 0 ∧ (S - x) / 5 = 6 ∧ x = 6 :=
by
  let S := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8
  have : S = 36, 
  { sorry },
  use [6]
  split
  { exact sorry },
  split
  { sorry },
  { sorry }

end seryozha_missing_number_l811_811727


namespace sum_of_gcd_and_lcm_is_28_l811_811390

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l811_811390


namespace minimum_value_of_quadratic_function_l811_811062

variable (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

theorem minimum_value_of_quadratic_function : 
  (∃ x : ℝ, x = p) ∧ (∀ x : ℝ, (x^2 - 2 * p * x + 4 * q) ≥ (p^2 - 2 * p * p + 4 * q)) :=
sorry

end minimum_value_of_quadratic_function_l811_811062


namespace max_largest_number_l811_811248

theorem max_largest_number (n : ℕ) (a : ℕ → ℕ)
  (h1 : n = 10)
  (h2 : (∑ i in Finset.range n, a i) = 200)
  (h3 : ∀ i : ℕ, i < n → (a i + a ((i + 1) % n) + a ((i + 2) % n) ≥ 58)) :
  ∃ i : ℕ, i < n ∧ a i = 26 :=
by
  sorry

end max_largest_number_l811_811248


namespace bouquet_combinations_l811_811429

theorem bouquet_combinations 
  (budget : ℕ)
  (cost_rose : ℕ)
  (cost_carnation : ℕ)
  (min_roses : ℕ)
  (r : ℕ)
  (c : ℕ) :
  budget = 60 ∧ cost_rose = 4 ∧ cost_carnation = 2 ∧ min_roses = 5 →
  4 * r + 2 * c = 60 ∧ r ≥ 5 →
  ∃ n : ℕ, n = 11 ∧ (finset.range (16)).filter (λ r, 4 * r + 2 * (30 - 2 * r)) = finset.range n.succ :=
sorry

end bouquet_combinations_l811_811429


namespace area_inequality_l811_811662

variables {A B C G B1 C1 : Point}
variables (ABC BGC1 CGB1 : Triangle)

-- Statements about the geometric properties:
def is_centroid (G : Point) (A B C : Point) := sorry  -- Definition of centroid
def not_separated_by (A G : Point) (d : Line) := sorry  -- Definition of not separated by

-- Main statement of the problem:
theorem area_inequality (hG : is_centroid G A B C) (hd : d.intersects AB B1) (hd2 : d.intersects AC C1) (hne : not_separated_by A G d) :
  area (BB1GC1) + area (CC1GB1) ≥ (4 / 9) * area (ABC) :=
sorry

end area_inequality_l811_811662


namespace probability_independent_events_intersection_l811_811817

theorem probability_independent_events_intersection (PA PB : ℚ) (hPA : PA = 4 / 7) (hPB : PB = 2 / 5) (independent : independent_events A B) :
  P (A ∩ B) = 8 / 35 :=
by 
  -- skipping proof
  sorry

end probability_independent_events_intersection_l811_811817


namespace log_squared_sum_eq_one_l811_811499

open Real

theorem log_squared_sum_eq_one :
  (log 2)^2 * log 250 + (log 5)^2 * log 40 = 1 := by
  sorry

end log_squared_sum_eq_one_l811_811499


namespace simplify_fraction_l811_811829

theorem simplify_fraction : 
  (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end simplify_fraction_l811_811829


namespace main_theorem_l811_811642

def line_l_param (t : ℝ) : ℝ × ℝ := (2 + 1/2 * t, sqrt 3 / 2 * t)

def curve_C_polar (ρ θ : ℝ) : Prop := ρ * sin θ ^ 2 - 4 * cos θ = 0

def line_l_standard (x y : ℝ) : Prop := y = sqrt 3 * (x - 2)

def curve_C_cartesian (x y : ℝ) : Prop := y ^ 2 = 4 * x

def M : ℝ × ℝ := (2, 0)

theorem main_theorem :
  (∀ t, line_l_param t = (2 + 1/2 * t, sqrt 3 / 2 * t)) ∧
  (∀ ρ θ, curve_C_polar ρ θ → curve_C_cartesian ρ θ) ∧
  (∀ (t₁ t₂ : ℝ),
    (t₁ + t₂ = 8 / 3 ∧ t₁ * t₂ = -32 / 3) →
    |1 / ((λ (t : ℝ × ℝ), real.sqrt ((t.1 - M.1) ^ 2 + (t.2 - M.2) ^ 2)) (line_l_param t₁)) -
     1 / ((λ (t : ℝ × ℝ), real.sqrt ((t.1 - M.1) ^ 2 + (t.2 - M.2) ^ 2)) (line_l_param t₂))| =
    1 / 4) :=
sorry

end main_theorem_l811_811642


namespace distribute_awards_l811_811728

theorem distribute_awards (awards : Finset ℕ) (students : Finset ℕ) 
  (h_awards : awards.card = 7) (h_students : students.card = 4)
  (h_condition : ∀ s ∈ students, ∃ a ∈ awards, a ∈ s):
  920 :=
by 
  sorry

end distribute_awards_l811_811728


namespace vincent_earnings_l811_811332

theorem vincent_earnings 
  (price_fantasy_book : ℕ)
  (num_fantasy_books_per_day : ℕ)
  (num_lit_books_per_day : ℕ)
  (num_days : ℕ)
  (h1 : price_fantasy_book = 4)
  (h2 : num_fantasy_books_per_day = 5)
  (h3 : num_lit_books_per_day = 8)
  (h4 : num_days = 5) :
  let price_lit_book := price_fantasy_book / 2
      daily_earnings_fantasy := price_fantasy_book * num_fantasy_books_per_day
      daily_earnings_lit := price_lit_book * num_lit_books_per_day
      total_daily_earnings := daily_earnings_fantasy + daily_earnings_lit
      total_earnings := total_daily_earnings * num_days
  in total_earnings = 180 := 
  by 
  {
    sorry
  }

end vincent_earnings_l811_811332


namespace largest_possible_integer_l811_811841

def list_of_integers_satisfying_conditions (l : List ℕ) : Prop :=
  l.length = 5 ∧
  7 ∈ l ∧ 
  (l.count 7 > 1) ∧
  List.medianOfOdd l = 10 ∧
  (l.sum : ℚ) / l.length = 12

theorem largest_possible_integer (l : List ℕ) (h : list_of_integers_satisfying_conditions l) : 
  ∃ x : ℕ, x ∈ l ∧ ∀ y ∈ l, y ≤ x ∧ x = 25 :=
by
  sorry

end largest_possible_integer_l811_811841


namespace sandwich_cost_l811_811340

-- Defining the cost of each sandwich and the known conditions
variable (S : ℕ) -- Cost of each sandwich in dollars

-- Conditions as hypotheses
def buys_three_sandwiches (S : ℕ) : ℕ := 3 * S
def buys_two_drinks (drink_cost : ℕ) : ℕ := 2 * drink_cost
def total_cost (sandwich_cost drink_cost total_amount : ℕ) : Prop := buys_three_sandwiches sandwich_cost + buys_two_drinks drink_cost = total_amount

-- Given conditions in the problem
def given_conditions : Prop :=
  (buys_two_drinks 4 = 8) ∧ -- Each drink costs $4
  (total_cost S 4 26)       -- Total spending is $26

-- Theorem to prove the cost of each sandwich
theorem sandwich_cost : given_conditions S → S = 6 :=
by sorry

end sandwich_cost_l811_811340


namespace area_ratio_of_squares_l811_811773

open Real

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 4 * 4 * b) : (a^2) / (b^2) = 16 := 
by
  sorry

end area_ratio_of_squares_l811_811773


namespace range_of_a_l811_811563

theorem range_of_a (i : ℂ) (a : ℝ) (z : ℂ) (M_re M_im : ℝ) :
  i = complex.I →
  z = (1 - 2 * complex.I) * (a + complex.I) →
  M_re = a + 2 →
  M_im = 1 - 2 * a →
  (M_re > 0) ∧ (M_im < 0) → 
  a > 1 / 2 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end range_of_a_l811_811563


namespace relationship_among_a_b_c_l811_811283

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (x : ℝ)

-- Assumptions
def symmetric_about_1 : Prop := ∀ x : ℝ, f(x) = f(2 - x)
def derivative_inequality : Prop := ∀ x : ℝ, x ≠ 1 → (x - 1) * f'(x) < 0

-- Specific function values
def a : ℝ := f 0.5
def b : ℝ := f (4/3)
def c : ℝ := f 3

-- Theorem statement
theorem relationship_among_a_b_c
  (h_symm : symmetric_about_1 f)
  (h_deriv_ineq : derivative_inequality f f') :
  b > a ∧ a > c := sorry

end relationship_among_a_b_c_l811_811283


namespace quadrilateral_is_rhombus_l811_811460

-- Structure for a quadrilateral
structure Quadrilateral (V : Type) :=
(A B C D : V)

-- Define the conditions for the quadrilateral in the problem
structure CircumscribedQuadrilateral (V : Type) [MetricSpace V] :=
(q : Quadrilateral V)
(circle : Set V)
(center : V)
(diag_intersect_center : (q.A - q.C : V) = center ∧ (q.B - q.D : V) = center)
(circumscribed : ∀ p ∈ [q.A, q.B, q.C, q.D], metric.dist p center = metric.dist center circle)

-- Theorem statement that the quadrilateral is a rhombus
theorem quadrilateral_is_rhombus {V : Type} [MetricSpace V] (quad : CircumscribedQuadrilateral V) :
  ∀ e ∈ [quad.q.A, quad.q.B, quad.q.C, quad.q.D], metric.dist quad.q.A quad.q.B = metric.dist quad.q.A quad.q.D ∧
                                                   metric.dist quad.q.B quad.q.C = metric.dist quad.q.C quad.q.D ∧
                                                   metric.dist quad.q.A quad.q.D = metric.dist quad.q.C quad.q.D :=
sorry

end quadrilateral_is_rhombus_l811_811460


namespace anton_thought_number_l811_811874

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l811_811874


namespace expected_number_of_different_faces_l811_811434

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l811_811434


namespace limit_x_eq_3_l811_811729

open Filter
open Topology

def x (t : ℝ) : ℝ := (6 * t^3 - 9 * t + 1) / (2 * t^3 - 3 * t)

theorem limit_x_eq_3 : tendsto (λ t => x t) at_top (nhds 3) :=
by
  sorry

end limit_x_eq_3_l811_811729


namespace subset_condition_for_a_l811_811670

theorem subset_condition_for_a (a : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 ≤ 5 / 4 → (|x - 1| + 2 * |y - 2| ≤ a)) → a ≥ 5 / 2 :=
by
  intro H
  sorry

end subset_condition_for_a_l811_811670


namespace vincent_earnings_l811_811335

-- Definitions based on the problem conditions
def fantasy_book_cost : ℕ := 4
def literature_book_cost : ℕ := fantasy_book_cost / 2
def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def duration : ℕ := 5

-- Calculation functions
def daily_earnings_from_fantasy_books : ℕ := fantasy_books_sold_per_day * fantasy_book_cost
def daily_earnings_from_literature_books : ℕ := literature_books_sold_per_day * literature_book_cost
def total_daily_earnings : ℕ := daily_earnings_from_fantasy_books + daily_earnings_from_literature_books
def total_earnings_after_five_days : ℕ := total_daily_earnings * duration

-- Statement to prove
theorem vincent_earnings : total_earnings_after_five_days = 180 := 
by
  calc total_daily_earnings * duration = 180 : sorry

end vincent_earnings_l811_811335


namespace car_actual_speed_is_40_l811_811413

variable (v : ℝ) -- actual speed (we will prove it is 40 km/h)

-- Conditions
variable (hyp_speed : ℝ := v + 20) -- hypothetical speed
variable (distance : ℝ := 60) -- distance traveled
variable (time_difference : ℝ := 0.5) -- time difference in hours

-- Define the equation derived from the given conditions:
def speed_equation : Prop :=
  (distance / v) - (distance / hyp_speed) = time_difference

-- The theorem to prove:
theorem car_actual_speed_is_40 : speed_equation v → v = 40 :=
by
  sorry

end car_actual_speed_is_40_l811_811413


namespace last_letter_150th_permutation_l811_811695

theorem last_letter_150th_permutation : (perms : List String) (H : perms = "AHSMEI".toList.permutations.map (λ l, l.asString)) (sorted_perms : List String) (H2 : sorted_perms = perms.qsort (λ x y, x < y)) : 
  (sorted_perms.nth 149).getLast ' ' = 'M' := 
by
  sorry

end last_letter_150th_permutation_l811_811695


namespace possible_to_make_sum_1986_l811_811195

-- Define the 400-digit number as a list of digits (for simplicity).
def four_hundred_digit_number : List ℕ := [8, 6, 1, 9] ++ List.replicate 98 [8, 6, 1, 9].sum

-- The main theorem statement to prove that we can remove some digits so that the sum of the remaining digits equals 1986.
theorem possible_to_make_sum_1986 : 
  ∃ (digits_to_remove : List ℕ),
  (digits_to_remove.all (· ∈ four_hundred_digit_number))
  ∧ (sum (four_hundred_digit_number.diff digits_to_remove) = 1986) := 
sorry

end possible_to_make_sum_1986_l811_811195


namespace sum_of_coefficients_l811_811689

/-- Let f and g be two distinct real polynomials defined as
      f(x) = x^2 + a * x + b
      g(x) = x^2 + c * x + d
    such that:
    1. The x-coordinate of the vertex of f is a root of g.
    2. The x-coordinate of the vertex of g is a root of f.
    3. Both f and g have the same minimum value.
    4. The graphs of the two polynomials intersect at the point (2012, -2012).
    Then, the sum of the coefficients a and c is -8048.
-/
theorem sum_of_coefficients (a b c d : ℝ)
    (hf_vertex_root : (g (-a / 2)) = 0)
    (hg_vertex_root : (f (-c / 2)) = 0)
    (h_min_equal : b - (a^2 / 4) = d - (c^2 / 4))
    (h_intersect : f 2012 = -2012 ∧ g 2012 = -2012) :
    a + c = -8048 := by
  sorry

end sum_of_coefficients_l811_811689


namespace expected_number_of_different_faces_l811_811440

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l811_811440


namespace verify_options_l811_811767

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + 1

theorem verify_options :
  let A := ∀ x, g x ≤ 2,
      B := ∀ x ∈ Icc (-5 * Real.pi / 12) (Real.pi / 12), 
           deriv g x ≥ 0,
      C := g (5 * Real.pi / 6) = 1 ∧ (∀ x, g (5 * Real.pi / 6 - x) = g (5 * Real.pi / 6 + x)),
      D := g (-Real.pi / 6) = 1 ∧ (∀ x, g (-Real.pi / 6 - x) = g (-Real.pi / 6 + x))
  in A ∧ B ∧ C ∧ ¬D :=
by
  sorry

end verify_options_l811_811767


namespace card_game_termination_l811_811324

open Function Relation 

-- Defining the card game environment

universe u

structure Card : Type

-- We assume the card comparison and beating relationship
structure GameState where
  P1_cards : List Card
  P2_cards : List Card

-- Beat relation: a relation on cards determining which card beats which.
def beats_relation (a b : Card) : Prop := sorry -- Assume this relation is given.

noncomputable def card_game (n : ℕ) (initial_state : GameState) : Prop :=
  ∃ final_state : GameState, (final_state.P1_cards.length = 0 ∨ final_state.P2_cards.length = 0) ∧ 
  reachable initial_state final_state

-- reachable definition: determines if final_state can be reached from initial_state
def reachable : GameState → GameState → Prop := sorry -- A proper inductive definition based on the game moves

-- Theorem that one player will end up with all cards (the other having none)
theorem card_game_termination (n : ℕ) (initial_state : GameState) (Hn : initial_state.P1_cards.length + initial_state.P2_cards.length = n) (Hbeats : ∀ a b : Card, beats_relation a b ∨ beats_relation b a) : card_game n initial_state :=
sorry

end card_game_termination_l811_811324


namespace maximum_value_integral_l811_811994

open Real

-- Define the conditions and problem.
def is_continuously_differentiable (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  continuous_on f (Icc a b) ∧ differentiable_on ℝ f (Ioo a b)

def problem (f : ℝ → ℝ) : Prop :=
  is_continuously_differentiable f 0 1 ∧
  f 0 = 0 ∧
  ∫ x in 0..1, (f' x) ^ 2 ≤ 1

theorem maximum_value_integral 
  (f : ℝ → ℝ) (h : problem f) :
  ∫ x in 0..1, (f' x) ^ 2 * |f x| * (1 / sqrt x) ≤ 2 / 3 :=
sorry

end maximum_value_integral_l811_811994


namespace range_of_expression_l811_811013

-- Helper function to define the expression
def expression (a b : ℝ) : ℝ := 1/a + a/b

-- The main theorem stating the proof problem
theorem range_of_expression (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  ∀ x, (∃ (a b : ℝ), a + 2 * b = 1 ∧ a > 0 ∧ b > 0 ∧ expression a b = x) ↔ x ∈ set.Ici (1 + 2 * Real.sqrt 2) :=
by {
  sorry,
}

end range_of_expression_l811_811013


namespace distinct_prime_factors_90_l811_811144

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811144


namespace min_vector_b_length_l811_811065

variables {R : Type*} [linear_ordered_field R]

noncomputable def vector_length (v : ℝ × ℝ) : R := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def dot_product (v w : ℝ × ℝ) : R := v.1 * w.1 + v.2 * w.2

theorem min_vector_b_length (a b : ℝ × ℝ)
  (h1 : vector_length a = 1)
  (h2 : dot_product a b = real.sqrt 3)
  (h3 : ∀ t : ℝ, vector_length ( (b.1 - t * a.1, b.2 - t * a.2) ) ≥ 2) :
  vector_length b ≥ real.sqrt 7 :=
begin
  sorry,
end

end min_vector_b_length_l811_811065


namespace max_value_f_l811_811053

noncomputable def f (x : ℝ) : ℝ := - (2 * f' 1 / 3) * (Real.sqrt x) - x^2

theorem max_value_f :
  ∃ a : ℝ, (∀ x : ℝ, f a ≥ f x) ∧ a = 34 / 4 :=
sorry

end max_value_f_l811_811053


namespace lindy_total_distance_l811_811417

-- Definitions and Conditions
def distance_between : ℝ := 240
def jack_speed : ℝ := 3
def christina_speed : ℝ := 3
def lindy_speed : ℝ := 10

-- Prove the total distance Lindy travels is 400 feet
theorem lindy_total_distance :
  let time_to_meet : ℝ := distance_between / (jack_speed + christina_speed),
      lindy_distance_traveled : ℝ := lindy_speed * time_to_meet
  in lindy_distance_traveled = 400 :=
by
  let time_to_meet := distance_between / (jack_speed + christina_speed)
  let lindy_distance_traveled := lindy_speed * time_to_meet
  have h_time_to_meet : time_to_meet = 40 := by
    unfold time_to_meet
    sorry
  have h_lindy_distance_traveled : lindy_distance_traveled = lindy_speed * time_to_meet := by
    unfold lindy_distance_traveled
    sorry
  have h_final : lindy_distance_traveled = 400 := by
    rw [h_lindy_distance_traveled, h_time_to_meet]
    sorry
  exact h_final

end lindy_total_distance_l811_811417


namespace correct_function_l811_811404

-- Definitions of the functions
def f1 (x : ℝ) : ℝ := x^2
def f2 (x : ℝ) : ℝ := x
def f3 (x : ℝ) : ℝ := 1 / x
def f4 (x : ℝ) : ℝ := -x^2 + 1

-- Properties predicates for easy re-use
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_monotonically_decreasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 ≥ f x2

-- Theorem statement
theorem correct_function :
  is_even f4 ∧ is_monotonically_decreasing_on_pos f4 ∧
  ¬ (is_even f1 ∧ is_monotonically_decreasing_on_pos f1) ∧
  ¬ (is_even f2 ∧ is_monotonically_decreasing_on_pos f2) ∧
  ¬ (is_even f3 ∧ is_monotonically_decreasing_on_pos f3) :=
by
  sorry

end correct_function_l811_811404


namespace geom_sequence_product_l811_811646

theorem geom_sequence_product (q a1 : ℝ) (h1 : a1 * (a1 * q) * (a1 * q^2) = 3) (h2 : (a1 * q^9) * (a1 * q^10) * (a1 * q^11) = 24) :
  (a1 * q^12) * (a1 * q^13) * (a1 * q^14) = 48 :=
by
  sorry

end geom_sequence_product_l811_811646


namespace coconut_grove_nut_yield_l811_811632

/--
In a coconut grove, the trees produce nuts based on some given conditions. Prove that the number of nuts produced by (x + 4) trees per year is 720 when x is 8. The conditions are:

1. (x + 4) trees yield a certain number of nuts per year.
2. x trees yield 120 nuts per year.
3. (x - 4) trees yield 180 nuts per year.
4. The average yield per year per tree is 100.
5. x is 8.
-/

theorem coconut_grove_nut_yield (x : ℕ) (y z w: ℕ) (h₁ : x = 8) (h₂ : y = 120) (h₃ : z = 180) (h₄ : w = 100) :
  ((x + 4) * w) - (x * y + (x - 4) * z) = 720 := 
by
  sorry

end coconut_grove_nut_yield_l811_811632


namespace tara_ice_cream_cartons_l811_811744

-- Definitions of the conditions
def number_of_yoghurt_cartons : ℕ := 4
def cost_of_ice_cream_carton : ℕ := 7
def cost_of_yoghurt_carton : ℕ := 1
def additional_spent_on_ice_cream : ℕ := 129

-- Theorem stating the question and its proof goal to be proved later
theorem tara_ice_cream_cartons (number_of_ice_cream_cartons : ℕ) 
  (h1 : cost_of_ice_cream_carton * number_of_ice_cream_cartons = (cost_of_yoghurt_carton * number_of_yoghurt_cartons) + additional_spent_on_ice_cream) :
  number_of_ice_cream_cartons = 19 :=
begin
  sorry
end

end tara_ice_cream_cartons_l811_811744


namespace derivative_of_f_l811_811279

variable {a x : ℝ}

-- Define the function \(f(x) = ( \frac{1}{a} )^x\)
def f (x : ℝ) := (1 / a) ^ x

-- State the conditions \(a > 0\) and \(a \neq 1\)
variables (ha : 0 < a) (ha_ne_one : a ≠ 1)

-- State that the derivative of the function equals the correct answer
theorem derivative_of_f :
  ∀ x : ℝ, deriv (λ x, (1 / a) ^ x) x = -a^(-x) * log a :=
by
  intro x
  sorry

end derivative_of_f_l811_811279


namespace a_finishes_race_in_t_seconds_l811_811627

theorem a_finishes_race_in_t_seconds 
  (time_B : ℝ := 45)
  (dist_B : ℝ := 100)
  (dist_A_wins_by : ℝ := 20)
  (total_dist : ℝ := 100)
  : ∃ t : ℝ, t = 36 := 
  sorry

end a_finishes_race_in_t_seconds_l811_811627


namespace rope_knot_length_reduction_l811_811328

theorem rope_knot_length_reduction :
  ∀ (length_of_rope : ℝ) (pieces : ℝ) (tied_pieces : ℝ) (final_length : ℝ),
    length_of_rope = 72 → pieces = 12 → tied_pieces = 3 → final_length = 15 →
    let original_piece_length := length_of_rope / pieces in
    let combined_length := tied_pieces * original_piece_length in
    let length_lost := combined_length - final_length in
    length_lost / (tied_pieces - 1) = 1.5 :=
by intros; simp; sorry

end rope_knot_length_reduction_l811_811328


namespace find_a_c_pair_l811_811776

-- Given conditions in the problem
variable (a c : ℝ)

-- First condition: The quadratic equation has exactly one solution
def quadratic_eq_has_one_solution : Prop :=
  let discriminant := (30:ℝ)^2 - 4 * a * c
  discriminant = 0

-- Second condition: Sum of a and c
def sum_eq_41 : Prop := a + c = 41

-- Third condition: a is less than c
def a_lt_c : Prop := a < c

-- State the proof problem
theorem find_a_c_pair (a c : ℝ) (h1 : quadratic_eq_has_one_solution a c) (h2 : sum_eq_41 a c) (h3 : a_lt_c a c) : (a, c) = (6.525, 34.475) :=
sorry

end find_a_c_pair_l811_811776


namespace gcd_lcm_sum_8_12_l811_811347

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l811_811347


namespace constant_term_of_expansion_l811_811791

theorem constant_term_of_expansion :
  let expr := (3 * x + 2 / (5 * x)) ^ 8 in
  -- condition using binomial theorem, simplify and find constant term
  expr = 5670000 :=
by sorry

end constant_term_of_expansion_l811_811791


namespace find_x_l811_811480

structure Triangle :=
(base : ℝ)
(side : ℝ)

structure IsoscelesTriangle (t : Triangle): Prop :=
(isosceles : t.side = 18)
(base_length : t.base = 12)

structure Trapezoid :=
(perimeter : ℝ)

def lengthsBD_BE (t : Triangle) (x : ℝ) : ℝ :=
2 * t.side - 2 * x + t.base

def validPerimeter (tr : IsoscelesTriangle) (x : ℝ) : Prop :=
Trapezoid.perimeter (Trapezoid.mk (lengthsBD_BE tr.toTriangle x)) = 40

theorem find_x :
  ∀ (t : Triangle) (trI : IsoscelesTriangle t),
  ∃ x : ℝ, validPerimeter trI x ∧ x = 6 := by
  sorry

end find_x_l811_811480


namespace ned_total_mows_l811_811704

def ned_mowed_front (spring summer fall : Nat) : Nat :=
  spring + summer + fall

def ned_mowed_backyard (spring summer fall : Nat) : Nat :=
  spring + summer + fall

theorem ned_total_mows :
  let front_spring := 6
  let front_summer := 5
  let front_fall := 4
  let backyard_spring := 5
  let backyard_summer := 7
  let backyard_fall := 3
  ned_mowed_front front_spring front_summer front_fall +
  ned_mowed_backyard backyard_spring backyard_summer backyard_fall = 30 := by
  sorry

end ned_total_mows_l811_811704


namespace distinct_prime_factors_90_l811_811158

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811158


namespace movie_production_cost_l811_811842

-- Definitions based on the conditions
def opening_revenue : ℝ := 120 -- in million dollars
def total_revenue : ℝ := 3.5 * opening_revenue -- movie made during its entire run
def kept_revenue : ℝ := 0.60 * total_revenue -- production company keeps 60% of total revenue
def profit : ℝ := 192 -- in million dollars

-- Theorem stating the cost to produce the movie
theorem movie_production_cost : 
  (kept_revenue - 60) = profit :=
by
  sorry

end movie_production_cost_l811_811842


namespace hyperbola_equation_exists_l811_811422

theorem hyperbola_equation_exists :
  ∃ λ : ℝ, (λ = -3) ∧ (∀ x y : ℝ, (y = 4 ∧ x = 3 → (x^2 / 9) - (y^2 / 4) = λ) →
    (λ = (y^2 / 12) - (x^2 / 27))) := 
sorry

end hyperbola_equation_exists_l811_811422


namespace hyperbola_min_value_l811_811280

-- Definitions based on the problem conditions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

def eccentricity (a b : ℝ) : Prop := (b^2 - a^2)^0.5 / a = 2

-- Main theorem to be proven
theorem hyperbola_min_value (a b : ℝ) (h1 : 1 ≤ a) (h2 : 1 ≤ b) 
  (h3 : hyperbola a b 0 1) (h4 : eccentricity a b) :
  (b^2 + 1) / (real.sqrt 3 * a) = 4 * real.sqrt 3 / 3 :=
sorry

end hyperbola_min_value_l811_811280


namespace number_of_distinct_prime_factors_of_90_l811_811112

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811112


namespace prime_between_30_and_40_with_remainder_1_l811_811286

theorem prime_between_30_and_40_with_remainder_1 (n : ℕ) : 
  n.Prime → 
  30 < n → n < 40 → 
  n % 6 = 1 → 
  n = 37 := 
sorry

end prime_between_30_and_40_with_remainder_1_l811_811286


namespace anton_thought_number_l811_811924

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l811_811924


namespace log_base_fraction_eq_l811_811978

theorem log_base_fraction_eq (x : ℝ) : (1 / 4) ^ x = 16 → x = -2 :=
by
  sorry

end log_base_fraction_eq_l811_811978


namespace simplify_fraction_l811_811260

theorem simplify_fraction (n : Nat) : (2^(n+4) - 3 * 2^n) / (2 * 2^(n+3)) = 13 / 16 :=
by
  sorry

end simplify_fraction_l811_811260


namespace smaller_angle_at_8_oclock_l811_811517

theorem smaller_angle_at_8_oclock : 
  let hour_angle := 30 in
  let angle_at_8 := 8 * hour_angle in
  let full_circle := 360 in
  let smaller_angle := full_circle - angle_at_8 in
  smaller_angle = 120 :=
by
  sorry

end smaller_angle_at_8_oclock_l811_811517


namespace trays_from_first_table_l811_811255

-- Definitions based on conditions
def trays_per_trip : ℕ := 4
def trips : ℕ := 3
def trays_from_second_table : ℕ := 2

-- Theorem statement to prove the number of trays picked up from the first table
theorem trays_from_first_table : trays_per_trip * trips - trays_from_second_table = 10 := by
  sorry

end trays_from_first_table_l811_811255


namespace distinct_prime_factors_of_90_l811_811089

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811089


namespace kanul_raw_material_expense_l811_811220

theorem kanul_raw_material_expense
  (total_amount : ℝ)
  (machinery_cost : ℝ)
  (raw_materials_cost : ℝ)
  (cash_fraction : ℝ)
  (h_total_amount : total_amount = 137500)
  (h_machinery_cost : machinery_cost = 30000)
  (h_cash_fraction: cash_fraction = 0.20)
  (h_eq : total_amount = raw_materials_cost + machinery_cost + cash_fraction * total_amount) :
  raw_materials_cost = 80000 :=
by
  rw [h_total_amount, h_machinery_cost, h_cash_fraction] at h_eq
  sorry

end kanul_raw_material_expense_l811_811220


namespace necessary_but_not_sufficient_l811_811956

-- Define the geometric mean condition between 2 and 8
def is_geometric_mean (m : ℝ) := m = 4 ∨ m = -4

-- Prove that m = 4 is a necessary but not sufficient condition for is_geometric_mean
theorem necessary_but_not_sufficient (m : ℝ) :
  (is_geometric_mean m) ↔ (m = 4) :=
sorry

end necessary_but_not_sufficient_l811_811956


namespace greatest_difference_l811_811302

def difference_marbles : Nat :=
  let A_diff := 4 - 2
  let B_diff := 6 - 1
  let C_diff := 9 - 3
  max (max A_diff B_diff) C_diff

theorem greatest_difference :
  difference_marbles = 6 :=
by
  sorry

end greatest_difference_l811_811302


namespace largest_digit_M_l811_811794

-- Define the conditions as Lean types
def digit_sum_divisible_by_3 (M : ℕ) := (4 + 5 + 6 + 7 + M) % 3 = 0
def even_digit (M : ℕ) := M % 2 = 0

-- Define the problem statement in Lean
theorem largest_digit_M (M : ℕ) (h : even_digit M ∧ digit_sum_divisible_by_3 M) : M ≤ 8 ∧ (∀ N : ℕ, even_digit N ∧ digit_sum_divisible_by_3 N → N ≤ M) :=
sorry

end largest_digit_M_l811_811794


namespace lending_rate_l811_811457

noncomputable def principal: ℝ := 5000
noncomputable def rate_borrowed: ℝ := 4
noncomputable def time_years: ℝ := 2
noncomputable def gain_per_year: ℝ := 100

theorem lending_rate :
  ∃ (rate_lent: ℝ), 
  (principal * rate_lent * time_years / 100) - (principal * rate_borrowed * time_years / 100) / time_years = gain_per_year ∧
  rate_lent = 6 :=
by
  sorry

end lending_rate_l811_811457


namespace distinct_prime_factors_90_l811_811161

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811161


namespace log_one_fourth_sixteen_l811_811969

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 :=
by
  sorry

end log_one_fourth_sixteen_l811_811969


namespace anton_thought_number_l811_811876

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l811_811876


namespace floor_r_is_3_l811_811692

noncomputable def f (x : ℝ) : ℝ := sin x + 2 * cos x + 2 * tan x

theorem floor_r_is_3 : ∃ r > 0, f r = 0 ∧ ⌊r⌋ = 3 :=
by sorry

end floor_r_is_3_l811_811692


namespace complex_expression_difference_l811_811175

noncomputable def complex_expression (c : ℂ) := c^24 

theorem complex_expression_difference :
  (i : ℂ) (hi : i^2 = -1) : 
  complex_expression (2 + i) - complex_expression (2 - i) = -5^12 * 0.544 * i :=
sorry

end complex_expression_difference_l811_811175


namespace number_of_distinct_prime_factors_90_l811_811148

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811148


namespace gcd_lcm_sum_8_12_l811_811397

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l811_811397


namespace area_of_transformed_triangle_l811_811741

-- Definitions
variable {F : Type*}
variables {f : F → F} {x1 x2 x3 : F}

-- Given conditions and proof statement
theorem area_of_transformed_triangle (domain : set F) 
  (h_domain : domain = {x1, x2, x3}) 
  (triangle_area : ∀ (x1 x2 x3 : F), 
    (area_of_triangle (f x1) (f x2) (f x3)) = 32) :
  let g (x : F) := 2 * f (2 * x) in
  area_of_triangle (g (x1/2)) (g (x2/2)) (g (x3/2)) = 32 := 
by
  sorry

end area_of_transformed_triangle_l811_811741


namespace income_of_A_l811_811748

theorem income_of_A (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050) 
  (h2 : (B + C) / 2 = 5250) 
  (h3 : (A + C) / 2 = 4200) : 
  A = 3000 :=
by
  sorry

end income_of_A_l811_811748


namespace correct_calculation_l811_811402

variable (a b : ℝ)

theorem correct_calculation :
  -(a - b) = -a + b := by
  sorry

end correct_calculation_l811_811402


namespace perpendicular_bisector_AD_passes_through_C_l811_811710

variable {A B C D J N M : Point}

-- Axiom/Properties for the geometric setups
axiom triangle_ABC : Triangle A B C
axiom D_on_BC : D ∈ Segment B C
axiom J_incenter : Incenter J (Triangle A B C)
axiom perpendicular_bisector_AD_passes_through_J : PerpendicularBisector (Segment A D) J

theorem perpendicular_bisector_AD_passes_through_C :
  PerpendicularBisector (Segment A D) C :=
by
  sorry

end perpendicular_bisector_AD_passes_through_C_l811_811710


namespace fraction_equiv_ratio_equiv_percentage_equiv_l811_811827

-- Define the problem's components and conditions.
def frac_1 : ℚ := 3 / 5
def frac_2 (a b : ℚ) : Prop := 3 / 5 = a / b
def ratio_1 (a b : ℚ) : Prop := 10 / a = b / 100
def percentage_1 (a b : ℚ) : Prop := (a / b) * 100 = 60

-- Problem statement 1: Fraction equality
theorem fraction_equiv : frac_2 12 20 := 
by sorry

-- Problem statement 2: Ratio equality
theorem ratio_equiv : ratio_1 (50 / 3) 60 := 
by sorry

-- Problem statement 3: Percentage equality
theorem percentage_equiv : percentage_1 60 100 := 
by sorry

end fraction_equiv_ratio_equiv_percentage_equiv_l811_811827


namespace correct_option_b_l811_811406

theorem correct_option_b (a : ℝ) : (-2 * a ^ 4) ^ 3 = -8 * a ^ 12 :=
sorry

end correct_option_b_l811_811406


namespace total_clothing_donated_l811_811475

-- Definition of the initial donation by Adam
def adam_initial_donation : Nat := 4 + 4 + 4*2 + 20 -- 4 pairs of pants, 4 jumpers, 4 pajama sets (8 items), 20 t-shirts

-- Adam's friends' total donation
def friends_donation : Nat := 3 * adam_initial_donation

-- Adam's donation after keeping half
def adam_final_donation : Nat := adam_initial_donation / 2

-- Total donation being the sum of Adam's and friends' donations
def total_donation : Nat := adam_final_donation + friends_donation

-- The statement to prove
theorem total_clothing_donated : total_donation = 126 := by
  -- This is skipped as per instructions
  sorry

end total_clothing_donated_l811_811475


namespace pages_per_day_l811_811003

-- Define the given conditions
def total_pages : ℕ := 957
def total_days : ℕ := 47

-- State the theorem based on the conditions and the required proof
theorem pages_per_day (p : ℕ) (d : ℕ) (h1 : p = total_pages) (h2 : d = total_days) :
  p / d = 20 := by
  sorry

end pages_per_day_l811_811003


namespace triangle_inequality_1_not_necessarily_triangle_inequality_l811_811626

theorem triangle_inequality_1 (p q r : ℝ) (hp : 0 < p) (hpq : p ≤ q) (hqr : q ≤ r) (h : p + q > r) : 
  (sqrt p + sqrt q > sqrt r) := sorry

theorem not_necessarily_triangle_inequality (p q r : ℝ) (hp : 0 < p) (hpq : p ≤ q) (hqr : q ≤ r) (h : p + q > r) :
  ¬(p^2 + q^2 > r^2) := sorry

end triangle_inequality_1_not_necessarily_triangle_inequality_l811_811626


namespace smallest_positive_period_intervals_of_monotonicity_symmetry_axes_centers_l811_811055

/-- Given the function f(x) = sin x * cos x - sqrt(3) * cos^2 x + (1 / 2) * sqrt(3) -/
def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * cos x ^ 2 + (1 / 2) * sqrt 3

-- Prove that the smallest positive period is π
theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x :=
sorry

-- Prove the intervals of monotonicity
theorem intervals_of_monotonicity {k : ℤ} :
  (∀ x : ℝ, (k * π - π / 12 <= x ∧ x <= k * π + 5 * π / 12) → f' x ≥ 0) ∧
  (∀ x : ℝ, (k * π + 5 * π / 12 <= x ∧ x <= k * π + 11 * π / 12) → f' x ≤ 0) :=
sorry

-- Prove the axes of symmetry and centers of symmetry
theorem symmetry_axes_centers {k : ℤ} :
  (∃ x : ℝ, f (x - (5 * π / 12 + k * π / 2)) = f (x + (5 * π / 12 + k * π / 2))) ∧
  (∃ x : ℝ, f (x - (k * π / 2 + π / 6)) = f (x + (k * π / 2 + π / 6))) :=
sorry

end smallest_positive_period_intervals_of_monotonicity_symmetry_axes_centers_l811_811055


namespace tangents_equal_l811_811730

theorem tangents_equal (α β γ : ℝ) (h1 : Real.sin α + Real.sin β + Real.sin γ = 0) (h2 : Real.cos α + Real.cos β + Real.cos γ = 0) :
  Real.tan (3 * α) = Real.tan (3 * β) ∧ Real.tan (3 * β) = Real.tan (3 * γ) := 
sorry

end tangents_equal_l811_811730


namespace probability_300_feet_or_less_l811_811860

noncomputable def calculate_probability : ℚ :=
  let gates := 16
  let distance := 75
  let max_distance := 300
  let initial_choices := gates
  let final_choices := gates - 1 -- because the final choice cannot be the same as the initial one
  let total_choices := initial_choices * final_choices
  let valid_choices :=
    (2 * 4 + 2 * 5 + 2 * 6 + 2 * 7 + 8 * 8) -- the total valid assignments as calculated in the solution
  (valid_choices : ℚ) / total_choices

theorem probability_300_feet_or_less : calculate_probability = 9 / 20 := 
by 
  sorry

end probability_300_feet_or_less_l811_811860


namespace cells_after_n_hours_l811_811745

variable (a : ℕ) (a_n : ℕ → ℕ)

-- Initial number of cells
def initial_cells : ℕ := 2

-- Rule for cell division and death
def cell_division_rule (n : ℕ) : ℤ :=
  by sorry

-- Transformation and sequence
def transformed_sequence (n : ℕ) : Prop :=
  a_n = λ n, 2 ^ n + 1

-- Prove the final result
theorem cells_after_n_hours (n : ℕ) : a_n = 2 ^ n + 1 :=
  by sorry

end cells_after_n_hours_l811_811745


namespace lunch_break_duration_l811_811856

def alice_rate (L : ℝ) : ℝ := 0.3 / (9 - L)
def assistants_rate (L : ℝ) : ℝ := 0.3 / (6 - L)
def total_work_day1 (a t L : ℝ) : ℝ := (8 - L) * (a + t)

theorem lunch_break_duration : 
  ∃ (L : ℝ), 
    let a := alice_rate L in
    let t := assistants_rate L in
    total_work_day1 a t L = 0.4 ∧
    (8 - L) * (a + t) = 0.4 ∧
    (6 - L) * t = 0.3 ∧
    (9 - L) * a = 0.3 ∧
    L = 45 / 60 := by
    sorry

end lunch_break_duration_l811_811856


namespace angle_between_vectors_l811_811564

variables (a b : ℝ)
variables (theta : ℝ)

axiom a_norm : ∥a∥ = 2
axiom b_norm : ∥b∥ = 2
axiom dot_product_condition : (a + 2 * b) • (a - b) = -2

theorem angle_between_vectors :
  θ = (arccos (1 / 2)) → θ = π / 3 :=
sorry

end angle_between_vectors_l811_811564


namespace max_value_of_a_l811_811284

def f (x : ℝ) : ℝ := Real.cos (2 * x)

def g (x : ℝ) : ℝ := Real.cos (2 * (x + (Real.pi / 12)))

theorem max_value_of_a :
  (∀ x ∈ set.Icc (0 : ℝ) (5 * Real.pi / 12), deriv g x < 0) ↔ true := sorry

end max_value_of_a_l811_811284


namespace range_of_t_l811_811584

theorem range_of_t (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  sqrt 3 ≤ (a + b + c) / sqrt (a * b + b * c + c * a) ∧ (a + b + c) / sqrt (a * b + b * c + c * a) < 2 :=
sorry

end range_of_t_l811_811584


namespace problem_statement_l811_811569

variable {f : ℝ → ℝ}

theorem problem_statement (h : ∀ x : ℝ, (x - 1) * (f' x) < 0) :
  f 0 + f 2 < 2 * f 1 :=
sorry

end problem_statement_l811_811569


namespace abs_difference_21st_term_l811_811322

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 - 14 * (n - 1)

theorem abs_difference_21st_term :
  |sequence_C 21 - sequence_D 21| = 520 := by
  sorry

end abs_difference_21st_term_l811_811322


namespace expected_number_of_different_faces_l811_811437

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l811_811437


namespace arithmetic_sequence_length_correct_l811_811166

noncomputable def arithmetic_sequence_length (a d last_term : ℕ) : ℕ :=
  ((last_term - a) / d) + 1

theorem arithmetic_sequence_length_correct :
  arithmetic_sequence_length 2 3 2014 = 671 :=
by
  sorry

end arithmetic_sequence_length_correct_l811_811166


namespace distinct_prime_factors_90_l811_811163

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811163


namespace x_square_minus_5x_is_necessary_not_sufficient_l811_811583

theorem x_square_minus_5x_is_necessary_not_sufficient (x : ℝ) :
  (x^2 - 5 * x < 0) → (|x - 1| < 1) → (x^2 - 5 * x < 0 ∧ ∃ y : ℝ, (0 < y ∧ y < 2) → x = y) :=
by
  sorry

end x_square_minus_5x_is_necessary_not_sufficient_l811_811583


namespace perpendicular_m_n_l811_811677

variables (α β : Plane) (m n : Line)

-- Let's assume definitions of Line and Plane along with relevant properties.
axiom diff_planes : α ≠ β
axiom diff_lines : m ≠ n
axiom perp_m_beta : perpendicular m β
axiom para_n_beta : parallel n β

-- State to prove that m is perpendicular to n.
theorem perpendicular_m_n : perpendicular m n :=
by sorry

end perpendicular_m_n_l811_811677


namespace perpendicular_MP_MQ_l811_811023

variable (k m : ℝ)

def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1

def line (x y : ℝ) := y = k*x + m

def fixed_point_exists (k m : ℝ) : Prop :=
  let P := (-(4 * k) / m, 3 / m)
  let Q := (4, 4 * k + m)
  ∃ (M : ℝ), (M = 1 ∧ ((P.1 - M) * (Q.1 - M) + P.2 * Q.2 = 0))

theorem perpendicular_MP_MQ : fixed_point_exists k m := sorry

end perpendicular_MP_MQ_l811_811023


namespace find_GQ_l811_811209

noncomputable def XYZTriangle : Type :=
{ XYZ : Type,
  XY : ℝ,
  XZ : ℝ,
  YZ : ℝ,
  medians_intersect_at_centroid : centroid_exists XYZ XY XZ YZ,
  Q_is_foot_of_altitude : foot_of_altitude G YZ = Q  }

theorem find_GQ (XYZ : Type) (XY XZ YZ : ℝ) (G : XYZ) (Q : XYZ) [XYZTriangle] :
  XY = 15 →
  XZ = 17 →
  YZ = 24 →
  foot_of_altitude G YZ = Q →
  centroid_exists XYZ XY XZ YZ G →
  calculate_GQ Q = 3.5 :=
by
  sorry

end find_GQ_l811_811209


namespace expected_number_of_different_faces_l811_811436

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l811_811436


namespace largest_digit_M_divisible_by_six_l811_811793

theorem largest_digit_M_divisible_by_six :
  (∃ M : ℕ, M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ ∀ m : ℕ, m ≤ M → (45670 + m) % 6 ≠ 0) :=
sorry

end largest_digit_M_divisible_by_six_l811_811793


namespace height_difference_l811_811650

-- Define the initial height of James's uncle
def uncle_height : ℝ := 72

-- Define the initial height ratio of James compared to his uncle
def james_initial_height_ratio : ℝ := 2 / 3

-- Define the height gained by James from his growth spurt
def james_growth_spurt : ℝ := 10

-- Define the initial height of James before the growth spurt
def james_initial_height : ℝ := uncle_height * james_initial_height_ratio

-- Define the new height of James after the growth spurt
def james_new_height : ℝ := james_initial_height + james_growth_spurt

-- Theorem: The difference in height between James's uncle and James after the growth spurt is 14 inches
theorem height_difference : uncle_height - james_new_height = 14 := sorry

end height_difference_l811_811650


namespace distinct_prime_factors_90_l811_811155

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811155


namespace exists_positive_integer_n_l811_811521

theorem exists_positive_integer_n (m : ℕ) (x_m : ℕ) (k : ℕ) : 
  (∃ n : ℕ, ∃ F : finset (fin (3^(k+1))) → Prop, 
    ∀ S : finset (fin (3^(k+1))), S.card = 2000 * n → 
    ∀ T : finset (fin (3^(k+1))), T.card = 3 → 
    ¬T ⊆ S → is_equilateral_triangle T) := 
begin
  sorry
end

end exists_positive_integer_n_l811_811521


namespace probability_lamps_l811_811256

noncomputable def num_ways_to_arrange_lamps : ℕ :=
  nat.choose 8 4

noncomputable def num_ways_to_arrange_remaining_lamps : ℕ :=
  nat.choose 6 3

noncomputable def num_ways_to_turn_on_lamps : ℕ :=
  nat.choose 7 3

def favorable_arrangements : ℕ :=
  num_ways_to_arrange_remaining_lamps * num_ways_to_turn_on_lamps

def total_arrangements : ℕ :=
  num_ways_to_arrange_lamps * num_ways_to_arrange_lamps

theorem probability_lamps : 
  (favorable_arrangements : ℝ) / (total_arrangements : ℝ) = 1 / 7 :=
by
  sorry

end probability_lamps_l811_811256


namespace num_distinct_prime_factors_90_l811_811134

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811134


namespace transformation_matrix_exists_l811_811534

def mul_matrix (M N : Matrix (Fin 3) (Fin 3) ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  M.mul N

theorem transformation_matrix_exists (N : Matrix (Fin 3) (Fin 3) ℝ) :
  let a := N 0 0
  let b := N 0 1
  let c := N 0 2
  let d := N 1 0
  let e := N 1 1
  let f := N 1 2
  let g := N 2 0
  let h := N 2 1
  let i := N 2 2
  mul_matrix (λ i j, match (i, j) with 
              | (0, 0) => 0 | (0, 1) => 0 | (0, 2) => 1
              | (1, 0) => 0 | (1, 1) => 3 | (1, 2) => 0
              | (2, 0) => 1 | (2, 1) => 0 | (2, 2) => 0) N =
  λ i j, match (i, j) with 
              | (0, 0) => g | (0, 1) => h | (0, 2) => i
              | (1, 0) => 3*d | (1, 1) => 3*e | (1, 2) => 3*f
              | (2, 0) => a | (2, 1) => b | (2, 2) => c := 
sorry

end transformation_matrix_exists_l811_811534


namespace expected_number_of_different_faces_l811_811447

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l811_811447


namespace sum_gcf_lcm_eq_28_l811_811368

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l811_811368


namespace anton_thought_number_l811_811921

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l811_811921


namespace total_cards_square_l811_811778

theorem total_cards_square (s : ℕ) (h_perim : 4 * s - 4 = 240) : s * s = 3721 := by
  sorry

end total_cards_square_l811_811778


namespace rug_floor_coverage_l811_811463

/-- A rectangular rug with side lengths of 2 feet and 7 feet is placed on an irregularly-shaped floor composed of a square with an area of 36 square feet and a right triangle adjacent to one of the square's sides, with leg lengths of 6 feet and 4 feet. If the surface of the rug does not extend beyond the area of the floor, then the fraction of the area of the floor that is not covered by the rug is 17/24. -/
theorem rug_floor_coverage : (48 - 14) / 48 = 17 / 24 :=
by
  -- proof goes here
  sorry

end rug_floor_coverage_l811_811463


namespace positive_slope_of_hyperbola_asymptote_l811_811512

def hyperbola_asymptote_slope : ℝ :=
  let A := (2, -3)
  let B := (8, -3)
  let AB := real.sqrt (6^2)
  let c := AB / 2
  let a := 4 / 2
  let b := real.sqrt (c^2 - a^2)
  b / a

theorem positive_slope_of_hyperbola_asymptote :
  ∃ slope : ℝ, slope = hyperbola_asymptote_slope ∧ slope > 0 :=
begin
  sorry
end

end positive_slope_of_hyperbola_asymptote_l811_811512


namespace problem_statement_l811_811568

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x ≥ 2) (h₂ : x + 4 / x ^ 2 ≥ 3) (h₃ : x + 27 / x ^ 3 ≥ 4) :
  ∀ a : ℝ, (x + a / x ^ 4 ≥ 5) → a = 4 ^ 4 := 
by 
  sorry

end problem_statement_l811_811568


namespace anton_thought_of_729_l811_811867

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l811_811867


namespace sum_gcf_lcm_eq_28_l811_811375

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l811_811375


namespace cone_volume_given_sphere_l811_811467

noncomputable def volume_of_cone (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H

theorem cone_volume_given_sphere (r V_sphere V_cone : ℝ) (h1 : V_sphere = (4 / 3) * π * r^3)
  (h2 : V_sphere = 32 * π / 3) : V_cone = (24 * π) := by
  -- definitions and conditions
  let a := 4 * real.sqrt 3
  let R := a / 2

  -- the height of the cone (calculated based on given sphere volume and cone geometry)
  let r := 2
  let H := 6

  -- calculate the volume of the cone
  have V_cone := volume_of_cone R H
  calc
    V_cone = 24 * π : by sorry

end cone_volume_given_sphere_l811_811467


namespace elsa_final_marbles_l811_811961

def initial_marbles : ℕ := 40
def marbles_lost_at_breakfast : ℕ := 3
def marbles_given_to_susie : ℕ := 5
def marbles_bought_by_mom : ℕ := 12
def twice_marbles_given_back : ℕ := 2 * marbles_given_to_susie

theorem elsa_final_marbles :
    initial_marbles
    - marbles_lost_at_breakfast
    - marbles_given_to_susie
    + marbles_bought_by_mom
    + twice_marbles_given_back = 54 := 
by
    sorry

end elsa_final_marbles_l811_811961


namespace distinct_prime_factors_of_90_l811_811096

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811096


namespace synthetic_analytic_incorrect_statement_l811_811408

theorem synthetic_analytic_incorrect_statement
  (basic_methods : ∀ (P Q : Prop), (P → Q) ∨ (Q → P))
  (synthetic_forward : ∀ (P Q : Prop), (P → Q))
  (analytic_backward : ∀ (P Q : Prop), (Q → P)) :
  ¬ (∀ (P Q : Prop), (P → Q) ∧ (Q → P)) :=
by
  sorry

end synthetic_analytic_incorrect_statement_l811_811408


namespace intersection_with_y_axis_l811_811758

theorem intersection_with_y_axis (x y : ℝ) (h : y = x + 3) (hx : x = 0) : (x, y) = (0, 3) := 
by 
  subst hx 
  rw [h]
  rfl
-- sorry to skip the proof

end intersection_with_y_axis_l811_811758


namespace sum_gcf_lcm_l811_811354

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l811_811354


namespace max_parade_members_l811_811297

theorem max_parade_members (n : ℤ) (hn_mod : 15 * n ≡ 3 [MOD 24]) (hn_lt : 15 * n < 1200) : 15 * n = 1155 :=
by
  -- rest of the proof would go here
  sorry

end max_parade_members_l811_811297


namespace transformed_stddev_l811_811625

noncomputable def variance (l : list ℝ) : ℝ :=
  let mean := l.sum / l.length in
  (l.map (λ x, (x - mean) ^ 2)).sum / l.length

def stddev (l : list ℝ) : ℝ :=
  real.sqrt (variance l)

theorem transformed_stddev (l : list ℝ) (h_length : l.length = 10) (h_stddev : stddev l = 8) :
  stddev (l.map (λ x, 2 * x - 1)) = 16 := 
sorry

end transformed_stddev_l811_811625


namespace unique_solution_l811_811532

open Nat

-- Define the equation condition as a predicate.
def equation_condition (p : List ℕ) : Prop :=
  p.prod = 10 * p.sum

-- Define the prime condition as a predicate.
def primes (p : List ℕ) : Prop :=
  ∀ n ∈ p, Prime n

-- The final statement asserts that the only solution is (2, 3, 5, 5).
theorem unique_solution :
  ∀ p : List ℕ, equation_condition p ∧ primes p ↔ p = [2, 3, 5, 5] := 
by
  intros
  sorry

end unique_solution_l811_811532


namespace total_clothes_donated_l811_811472

theorem total_clothes_donated
  (pants : ℕ) (jumpers : ℕ) (pajama_sets : ℕ) (tshirts : ℕ)
  (friends : ℕ)
  (adam_donation : ℕ)
  (half_adam_donated : ℕ)
  (friends_donation : ℕ)
  (total_donation : ℕ)
  (h1 : pants = 4) 
  (h2 : jumpers = 4) 
  (h3 : pajama_sets = 4 * 2) 
  (h4 : tshirts = 20) 
  (h5 : friends = 3)
  (h6 : adam_donation = pants + jumpers + pajama_sets + tshirts) 
  (h7 : half_adam_donated = adam_donation / 2) 
  (h8 : friends_donation = friends * adam_donation) 
  (h9 : total_donation = friends_donation + half_adam_donated) :
  total_donation = 126 :=
by
  sorry

end total_clothes_donated_l811_811472


namespace area_enclosed_by_curves_l811_811271

theorem area_enclosed_by_curves : 
  (∫ x in 0..1, (Real.sqrt x - x^2)) = 1 / 3 := 
sorry

end area_enclosed_by_curves_l811_811271


namespace gcd_lcm_sum_8_12_l811_811395

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l811_811395


namespace sandwich_cost_l811_811339

theorem sandwich_cost (c : ℕ) 
  (sandwiches : ℕ := 3)
  (drinks : ℕ := 2)
  (cost_per_drink : ℕ := 4)
  (total_spent : ℕ := 26)
  (drink_cost : ℕ := drinks * cost_per_drink)
  (sandwich_spent : ℕ := total_spent - drink_cost) :
  (∀ s, sandwich_spent = s * sandwiches → s = 6) :=
by
  intros s hs
  have hsandwich_count : sandwiches = 3 := by rfl
  have hdrinks : drinks = 2 := by rfl
  have hcost_per_drink : cost_per_drink = 4 := by rfl
  have htotal_spent : total_spent = 26 := by rfl
  have hdrink_cost : drink_cost = 8 := by
    calc 
      drinks * cost_per_drink 
      = 2 * 4 : by rw [hdrinks, hcost_per_drink]
      = 8 : by norm_num
  have hsandwich_spent : sandwich_spent = 18 := by
    calc
      total_spent - drink_cost 
      = 26 - 8 : by rw [htotal_spent, hdrink_cost]
      = 18 : by norm_num
  rw hsandwich_count at hs
  rw hsandwich_spent at hs
  linarith

end sandwich_cost_l811_811339


namespace number_of_distinct_prime_factors_of_90_l811_811113

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811113


namespace distinct_prime_factors_count_l811_811117

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811117


namespace greatest_difference_l811_811306

-- Definitions: Number of marbles in each basket
def basketA_red : Nat := 4
def basketA_yellow : Nat := 2
def basketB_green : Nat := 6
def basketB_yellow : Nat := 1
def basketC_white : Nat := 3
def basketC_yellow : Nat := 9

-- Define the differences
def diff_basketA : Nat := basketA_red - basketA_yellow
def diff_basketB : Nat := basketB_green - basketB_yellow
def diff_basketC : Nat := basketC_yellow - basketC_white

-- The goal is to prove that 6 is the greatest difference
theorem greatest_difference : max (max diff_basketA diff_basketB) diff_basketC = 6 :=
by 
  -- The proof is not provided
  sorry

end greatest_difference_l811_811306


namespace system_of_equations_solutions_l811_811263

theorem system_of_equations_solutions (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 1000) 
(h4 : x ^ real.log y + y ^ real.log (real.sqrt x) = 110) : 
(x = 10 ∧ y = 100) ∨ (x = 100 ∧ y = 10) :=
sorry

end system_of_equations_solutions_l811_811263


namespace maximum_vertex_product_sum_l811_811857

open BigOperators

theorem maximum_vertex_product_sum :
  ∃ (a b c d e f : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                            b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                            c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                            d ≠ e ∧ d ≠ f ∧
                            e ≠ f ∧
                            {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6}) ∧
  (ace + acf + ade + adf + bce + bcf + bde + bdf = 343) :=
  sorry

end maximum_vertex_product_sum_l811_811857


namespace size_of_first_file_l811_811514

theorem size_of_first_file (internet_speed_mbps : ℝ) (time_hours : ℝ) (file2_mbps : ℝ) (file3_mbps : ℝ) (total_downloaded_mbps : ℝ) :
  internet_speed_mbps = 2 →
  time_hours = 2 →
  file2_mbps = 90 →
  file3_mbps = 70 →
  total_downloaded_mbps = internet_speed_mbps * 60 * time_hours →
  total_downloaded_mbps - (file2_mbps + file3_mbps) = 80 :=
by
  intros
  sorry

end size_of_first_file_l811_811514


namespace binary_representation_of_38_l811_811949

theorem binary_representation_of_38 : ∃ binary : ℕ, binary = 0b100110 ∧ binary = 38 :=
by
  sorry

end binary_representation_of_38_l811_811949


namespace dad_borrowed_nickels_l811_811726

-- Definitions for the initial and remaining nickels
def initial_nickels : ℕ := 31
def remaining_nickels : ℕ := 11

-- Statement of the problem in Lean
theorem dad_borrowed_nickels : initial_nickels - remaining_nickels = 20 := by
  -- Proof goes here
  sorry

end dad_borrowed_nickels_l811_811726


namespace crabapple_recipients_sequence_count_l811_811702

/-- Mrs. Crabapple teaches a class of 15 students and her advanced literature class meets three times a week.
    She picks a new student each period to receive a crabapple, ensuring no student receives more than one
    crabapple in a week. Prove that the number of different sequences of crabapple recipients is 2730. -/
theorem crabapple_recipients_sequence_count :
  ∃ sequence_count : ℕ, sequence_count = 15 * 14 * 13 ∧ sequence_count = 2730 :=
by
  sorry

end crabapple_recipients_sequence_count_l811_811702


namespace race_completion_time_l811_811630

variable (t : ℕ)
variable (vA vB : ℕ)
variable (tB : ℕ := 45)
variable (d : ℕ := 100)
variable (diff : ℕ := 20)
variable h1 : vA * t = d
variable h2 : vB * t = d - diff
variable h3 : vB = d / tB

theorem race_completion_time (h : vB = d / tB): t = 36 :=
by sorry

end race_completion_time_l811_811630


namespace problem_6_l811_811420

-- Define the prime checker function
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m.divides n → (m = 1 ∨ m = n)

noncomputable def losing_positions (limit : ℕ) : List ℕ :=
  let rec helper (n : ℕ) (acc : List ℕ) :=
    if n > limit then acc else
    if (∀ p, is_prime p ∧ (n - p) ≥ 0 → ¬(n - p) ∈ acc)
    then helper (n + 1) (n :: acc)
    else helper (n + 1) acc
  in helper 1 []

-- Statement of the problem
theorem problem_6 :
  let winning_sum :=
    (List.range (31)).filter (λ z => ¬(z ∈ losing_positions 30))
                           .sum
  in winning_sum = 45 :=
sorry

end problem_6_l811_811420


namespace max_ratio_of_odd_integers_is_nine_l811_811235

-- Define odd positive integers x and y whose mean is 55
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_positive (n : ℕ) : Prop := 0 < n
def mean_is_55 (x y : ℕ) : Prop := (x + y) / 2 = 55

-- The problem statement
theorem max_ratio_of_odd_integers_is_nine (x y : ℕ) 
  (hx : is_positive x) (hy : is_positive y)
  (ox : is_odd x) (oy : is_odd y)
  (mean : mean_is_55 x y) : 
  ∀ r, r = (x / y : ℚ) → r ≤ 9 :=
by
  sorry

end max_ratio_of_odd_integers_is_nine_l811_811235


namespace jimmy_paid_total_l811_811659

-- Data for the problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100
def park_pizzas : ℕ := 3
def building_distance : ℕ := 2000
def building_pizzas : ℕ := 2
def house_distance : ℕ := 800
def house_pizzas : ℕ := 4
def community_center_distance : ℕ := 1500
def community_center_pizzas : ℕ := 5
def office_distance : ℕ := 300
def office_pizzas : ℕ := 1
def bus_stop_distance : ℕ := 1200
def bus_stop_pizzas : ℕ := 3

def cost (distance pizzas : ℕ) : ℕ := 
  let base_cost := pizzas * pizza_cost
  if distance > 1000 then base_cost + delivery_charge else base_cost

def total_cost : ℕ :=
  cost park_distance park_pizzas +
  cost building_distance building_pizzas +
  cost house_distance house_pizzas +
  cost community_center_distance community_center_pizzas +
  cost office_distance office_pizzas +
  cost bus_stop_distance bus_stop_pizzas

theorem jimmy_paid_total : total_cost = 222 :=
  by
    -- Proof omitted
    sorry

end jimmy_paid_total_l811_811659


namespace women_attended_l811_811489

theorem women_attended :
  (15 * 4) / 3 = 20 :=
by
  sorry

end women_attended_l811_811489


namespace can_cover_101x101_with_102_cells_100_times_l811_811631

theorem can_cover_101x101_with_102_cells_100_times :
  ∃ f : Fin 100 → Fin 101 → Fin 101 → Bool,
  (∀ i j : Fin 101, (i ≠ 100 ∨ j ≠ 100) → ∃ t : Fin 100, 
    f t i j = true) :=
sorry

end can_cover_101x101_with_102_cells_100_times_l811_811631


namespace transformed_polynomial_l811_811685

noncomputable def P : Polynomial ℝ := Polynomial.C 9 + Polynomial.X ^ 3 - 4 * Polynomial.X ^ 2 

noncomputable def Q : Polynomial ℝ := Polynomial.C 243 + Polynomial.X ^ 3 - 12 * Polynomial.X ^ 2 

theorem transformed_polynomial :
  ∀ (r : ℝ), Polynomial.aeval r P = 0 → Polynomial.aeval (3 * r) Q = 0 := 
by
  sorry

end transformed_polynomial_l811_811685


namespace imaginary_part_of_z_l811_811582

def imaginary_unit := Complex.I
def complex_number (i : Complex) : Complex :=
  (1 - i)^2 / (1 + i)

theorem imaginary_part_of_z (i : Complex) (h : i = imaginary_unit) :
  Complex.im (complex_number i) = -1 :=
sorry

end imaginary_part_of_z_l811_811582


namespace ella_days_11_years_old_l811_811526

theorem ella_days_11_years_old (x y z : ℕ) (h1 : 40 * x + 44 * y + 48 * (180 - x - y) = 7920) (h2 : x + y + z = 180) (h3 : 2 * x + y = 180) : y = 60 :=
by {
  -- proof can be derived from the given conditions
  sorry
}

end ella_days_11_years_old_l811_811526


namespace problem_statement_l811_811597

noncomputable def f (x φ : ℝ) :=
  (1/2) * (Real.sin (2 * x)) * (Real.sin φ) +
  (Real.cos x) ^ 2 * (Real.cos φ) +
  (1/2) * (Real.sin ((3 * Real.pi / 2) - φ))

theorem problem_statement (φ x0 : ℝ) (hφ : 0 < φ ∧ φ < Real.pi) (hx0 : Real.sin x0 = 3/5 ∧ (Real.pi / 2) < x0 ∧ x0 < Real.pi) :
  (f (Real.pi / 6) φ = 1 / 2) →
  (∀ x ∈ ℝ, f x φ = (1/2) * Real.cos(2 * x - φ)) →
  (∀ x ∈ ℝ, (x ∈ set.Icc (Real.pi / 6) (2 * Real.pi / 3) → (Real.deriv (f x φ)) < 0))
  ∧ (f x0 φ = (7 - 24 * Real.sqrt 3) / 100) :=
by
  sorry

end problem_statement_l811_811597


namespace sum_of_squares_induction_l811_811788

theorem sum_of_squares_induction (n : ℕ) :
  (∀ n, n > 0 → (∑ i in Finset.range n, i^2 / ((2*i-1)*(2*i+1))) = n*(n+1) / (2*(2*n+1))) :=
begin
  sorry
end

end sum_of_squares_induction_l811_811788


namespace max_cities_both_forests_l811_811291

theorem max_cities_both_forests (n : ℕ) :
  (∀ G : SimpleGraph (Fin n), (G.IsForest ∧ G.complement.IsForest) → n ≤ 4) :=
sorry

end max_cities_both_forests_l811_811291


namespace distinct_prime_factors_90_l811_811159

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811159


namespace LeanProof_l811_811206

noncomputable def ProblemStatement : Prop :=
  let AB_parallel_YZ := True -- given condition that AB is parallel to YZ
  let AZ := 36 
  let BQ := 15
  let QY := 20
  let similarity_ratio := BQ / QY = 3 / 4
  ∃ QZ : ℝ, AZ = (3 / 4) * QZ + QZ ∧ QZ = 144 / 7

theorem LeanProof : ProblemStatement :=
sorry

end LeanProof_l811_811206


namespace books_left_unchanged_l811_811715

theorem books_left_unchanged (initial_books : ℕ) (initial_pens : ℕ) (pens_sold : ℕ) (pens_left : ℕ) :
  initial_books = 51 → initial_pens = 106 → pens_sold = 92 → pens_left = 14 → initial_books = 51 := 
by
  intros h_books h_pens h_sold h_left
  exact h_books

end books_left_unchanged_l811_811715


namespace jellybeans_in_jar_l811_811298

theorem jellybeans_in_jar (num_kids_normal : ℕ) (num_absent : ℕ) (num_jellybeans_each : ℕ) (num_leftover : ℕ) 
  (h1 : num_kids_normal = 24) (h2 : num_absent = 2) (h3 : num_jellybeans_each = 3) (h4 : num_leftover = 34) : 
  (num_kids_normal - num_absent) * num_jellybeans_each + num_leftover = 100 :=
by sorry

end jellybeans_in_jar_l811_811298


namespace alternating_series_sum_l811_811421

theorem alternating_series_sum : ∑ n in finset.range 10, (-1)^(n+1) * (2*n + 1) = -10 := 
  by sorry

end alternating_series_sum_l811_811421


namespace incorrect_student_l811_811557

theorem incorrect_student (b c : ℝ) 
  (hA : ∃ k, k = 1 ∧ ∃ v, v = x^2 + bx + c ∧ ∀ x, v ≥ k)
  (hC : ∃ x, x = 1 ∧ y = 3)
  (hD : y = 4 ↔ x = 2) :
  ¬(x = -1 ∧ y = 0) :=
by sorry

end incorrect_student_l811_811557


namespace anton_thought_number_is_729_l811_811895

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l811_811895


namespace necessary_but_not_sufficient_condition_l811_811554
-- Import the Mathlib library for all necessary mathematics

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (|x|) = f (|-x|)

-- Proof statement (with condition)
theorem necessary_but_not_sufficient_condition (f : ℝ → ℝ) :
  is_symmetric_about_y_axis f ↔ ∀ x, is_symmetric_about_y_axis (λ x, f (|x|)) :=
sorry

end necessary_but_not_sufficient_condition_l811_811554


namespace david_practices_athletics_l811_811244

-- Define the types for Person and Sport
inductive Person where
| Maria | Tania | Juan | David
deriving DecidableEq

inductive Sport where
| Swimming | Volleyball | Gymnastics | Athletics
deriving DecidableEq

-- Define the seating arrangement as a square table
structure Table where
  seat : Person → Option Sport
  symmetry : Person → Person

-- Conditions as functions of the seating arrangement
def left_of (t : Table) (p1 p2 : Person) : Prop :=
  t.symmetry p1 = p2

def opposite (t : Table) (p1 p2 : Person) : Prop :=
  t.symmetry (t.symmetry p1) = p2

def side_by_side (t : Table) (p1 p2 : Person) : Prop :=
  t.symmetry p1 ≠ p2 ∧ 
  (t.symmetry p1 = p2 ∨ t.symmetry (t.symmetry p1) = p2)

def next_to_woman (t : Table) (p : Person) : Prop :=  
  (t.seat p1 ∈ [Sport.Volleyball] → 
  (t.symmetry (t.symmetry p1) = Person.Maria ∨ 
   t.symmetry (t.symmetry p1) = Person.Tania))

-- Define the conditions
def conditions (t : Table) : Prop :=
  left_of t Sport.Swimming Person.Maria ∧
  opposite t Sport.Gymnastics Person.Juan ∧
  side_by_side t Person.Tania Person.David ∧
  ∃ (p : Person), t.seat p = Sport.Volleyball ∧
                   ((t.symmetry (t.symmetry p) = Person.Maria) ∨ 
                    (t.symmetry (t.symmetry p) = Person.Tania))

-- The theorem to prove
theorem david_practices_athletics (t : Table) (H : conditions t) :
  t.seat Person.David = Sport.Athletics :=
by
  sorry

end david_practices_athletics_l811_811244


namespace path_segment_length_property_l811_811230

noncomputable def square := { points : Finset (ℝ × ℝ) // ∀ (x y : ℝ × ℝ), (x ∈ points → y ∈ points → (abs ((x.1 - y.1)^2 + (x.2 - y.2)^2) ≤ 100*100)) }

noncomputable def path_in_square (n : ℕ) := 
  { path : Fin n → (ℝ × ℝ) // (∀ (i j : Fin n), i ≠ j → path i ≠ path j) ∧ (∀ (i : Fin n), path i ∈ square) }

noncomputable def boundary_distance_constraint (S : square) (L : path_in_square n) := 
  ∀ (P : (ℝ × ℝ)), (P ∈ S) → (∃ (Q : (ℝ × ℝ)), (Q ∈ L) ∧ abs ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ (1/2)^2)

theorem path_segment_length_property 
  (S : square) 
  (L : path_in_square n) 
  (P_prop : boundary_distance_constraint S L) : 
  ∃ (X Y : (ℝ × ℝ)), (X ∈ L) ∧ (Y ∈ L) ∧ abs ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) ≤ 1 
  ∧ ∑ i in fin_range X ... Y, path_length (L i) ≥ 198 := 
by
  sorry

end path_segment_length_property_l811_811230


namespace women_attended_l811_811488

theorem women_attended :
  (15 * 4) / 3 = 20 :=
by
  sorry

end women_attended_l811_811488


namespace uncle_taller_than_james_l811_811653

def james_initial_height (uncle_height : ℕ) : ℕ := (2 * uncle_height) / 3

def james_final_height (initial_height : ℕ) (growth_spurt : ℕ) : ℕ := initial_height + growth_spurt

theorem uncle_taller_than_james (uncle_height : ℕ) (growth_spurt : ℕ) :
  uncle_height = 72 →
  growth_spurt = 10 →
  uncle_height - (james_final_height (james_initial_height uncle_height) growth_spurt) = 14 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end uncle_taller_than_james_l811_811653


namespace log_fraction_value_l811_811958

theorem log_fraction_value :
  let log := (λ (a b : ℝ), Real.log a / Real.log b) in
  log 9 8 / log 3 2 = 2 / 3 :=
by
  sorry

end log_fraction_value_l811_811958


namespace missing_fraction_is_11_div_10_l811_811779

theorem missing_fraction_is_11_div_10 : 
  let f1 : ℚ := 1/3
      f2 : ℚ := 1/2
      f3 : ℚ := -5/6
      f4 : ℚ := 1/4
      f5 : ℚ := -9/20
      f6 : ℚ := -9/20
      given_sum : ℚ := f1 + f2 + f3 + f4 + f5 + f6
  in given_sum = 9/20 → 
  let remaining_sum : ℚ := 0.45 - given_sum
  in remaining_sum = 11/10 := 
by {
  sorry
}

end missing_fraction_is_11_div_10_l811_811779


namespace find_angle_A_l811_811031

theorem find_angle_A
  (a b : ℝ) (A B C : ℝ)
  (ha : a = 2)
  (hb : b = real.sqrt 6)
  (hAC : A + C = 2 * B)
  (h_triangle : A + B + C = real.pi) :
  A = real.pi / 4 :=
by
  -- We need to prove A = π/4
  sorry

end find_angle_A_l811_811031


namespace inscribed_circle_radius_l811_811690

theorem inscribed_circle_radius (a b c r : ℝ) (h : a^2 + b^2 = c^2) (h' : r = (a + b - c) / 2) : r = (a + b - c) / 2 :=
by
  sorry

end inscribed_circle_radius_l811_811690


namespace number_of_distinct_prime_factors_90_l811_811149

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811149


namespace distinct_prime_factors_90_l811_811140

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811140


namespace distinct_prime_factors_count_l811_811122

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811122


namespace at_least_one_composite_l811_811738

theorem at_least_one_composite (a b c k : ℕ) (h1 : a ≥ 3) (h2 : b ≥ 3) (h3 : c ≥ 3) (h4 : a * b * c = k^2 + 1) : 
  ∃ x ∈ {a - 1, b - 1, c - 1}, ¬Nat.Prime x :=
by
  sorry

end at_least_one_composite_l811_811738


namespace error_in_square_area_l811_811412

theorem error_in_square_area (s : ℝ) (hs : s > 0) :
  let measured_side := 1.10 * s,
      actual_area := s^2,
      calculated_area := (1.10 * s)^2,
      error_area := calculated_area - actual_area,
      percentage_error := (error_area / actual_area) * 100
  in percentage_error = 21 :=
by
  sorry

end error_in_square_area_l811_811412


namespace isosceles_right_triangle_l811_811025

noncomputable theory

open real

theorem isosceles_right_triangle 
  (a : ℝ) (AP PB CP : ℝ) (h1 : AP = (2 / 3) * a * sqrt 2) (h2 : PB = (1 / 3) * a * sqrt 2) (h3 : CP = a) :
  let s := AP^2 + PB^2 in s < 2 * CP^2 :=
by
  sorry

end isosceles_right_triangle_l811_811025


namespace part1_part2_l811_811498

-- Part 1
theorem part1 (a : ℝ) (h : a ^ (1 / 2) + a ^ (-1 / 2) = 3) : a + a ^ (-1) = 7 := 
  sorry

-- Part 2
theorem part2 : 2*(Real.log (sqrt 2))^2 + Real.log (sqrt 2) * Real.log 5 
+ sqrt ((Real.log (sqrt 2))^2 - 2*Real.log (sqrt 2) + 1) = 1 :=
  sorry

end part1_part2_l811_811498


namespace part1_part2_l811_811046

theorem part1 : 
  (∃ x : ℝ, |x - 1| - |x - 2| ≥ t) → (∀ t : ℝ, t ∈ T ↔ t ≤ 1) := 
by
  sorry

theorem part2 (m n : ℝ) (hm : m > 1) (hn : n > 1) :
  (∀ t ∈ T, (log 3 m) * (log 3 n) ≥ t) → m^2 + n^2 ≥ 18 := 
by
  sorry

end part1_part2_l811_811046


namespace distinct_prime_factors_90_l811_811156

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811156


namespace distinct_prime_factors_90_l811_811142

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811142


namespace find_monotonically_decreasing_function_l811_811806

open Real

def is_monotonically_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

def f (x : ℝ) : ℝ := log 2 x
def g (x : ℝ) : ℝ := 2^(-x)
def h (x : ℝ) : ℝ := sqrt (x + 1)
def k (x : ℝ) : ℝ := x^3

theorem find_monotonically_decreasing_function :
  ∃! f, (f = g ∧ is_monotonically_decreasing_on f {x : ℝ | 0 < x})
:= 
  sorry

end find_monotonically_decreasing_function_l811_811806


namespace prob_two_red_scheme1_scheme2_more_advantageous_l811_811835

-- Define the conditions for the problem
def bagContains (redBalls yellowBalls : ℕ) : Prop := redBalls = 2 ∧ yellowBalls = 3
def rewardAmount (redBallsDrawn : ℕ) : ℕ :=
  if redBallsDrawn = 0 then 5
  else if redBallsDrawn = 1 then 10
  else if redBallsDrawn = 2 then 20
  else 0
def scheme1 (draws: List (List Prop)) : Prop := True -- Placeholder definition
def scheme2 (draws: List (List Prop)) : Prop := True -- Placeholder definition

-- Part (1)
theorem prob_two_red_scheme1 : 
  bagContains 2 3 →
  scheme1 [(List.replicate 2 true ++ List.replicate 3 false)] →
  prob (draw two red balls) = 1 / 10 :=
sorry

-- Part (2)
theorem scheme2_more_advantageous :
  bagContains 2 3 →
  (∀ n, rewardAmount n ∈ {5, 10, 20}) →
  scheme1 [(List.replicate 2 true ++ List.replicate 3 false)] →
  scheme2 [(List.replicate 2 true ++ List.replicate 3 false)] →
  averageEarnings scheme2 > averageEarnings scheme1 :=
sorry

end prob_two_red_scheme1_scheme2_more_advantageous_l811_811835


namespace subscriptions_to_grandfather_l811_811243

/-- 
Maggie earns $5.00 for every magazine subscription sold. 
She sold 4 subscriptions to her parents, 2 to the next-door neighbor, 
and twice that amount to another neighbor. Maggie earned $55 in total. 
Prove that the number of subscriptions Maggie sold to her grandfather is 1.
-/
theorem subscriptions_to_grandfather (G : ℕ) 
  (h1 : 5 * (4 + G + 2 + 4) = 55) : 
  G = 1 :=
by {
  sorry
}

end subscriptions_to_grandfather_l811_811243


namespace problem_equiv_l811_811231

noncomputable theory
open Real

-- Definition of the problem in Lean
theorem problem_equiv :
  ∀ (a b c p q r : ℝ),
    17 * p + b * q + c * r = 0 →
    a * p + 29 * q + c * r = 0 →
    a * p + b * q + 56 * r = 0 →
    a ≠ 17 →
    p ≠ 0 →
    (a / (a - 17) + b / (b - 29) + c / (c - 56) = 1) :=
by sorry

end problem_equiv_l811_811231


namespace visible_surface_area_correct_l811_811944

-- Define the side lengths from the given volumes
def side_lengths := [1, 3, 4, 5, 6, 7, 8]

-- Define the function to calculate the visible surface area
def visible_surface_area : ℕ :=
  let top_cube := 6 * (side_lengths.head!)^2
  let bottom_cube := 6 * (side_lengths.getLast!.getOrElse 0)^2
  let middle_cubes := side_lengths.zipWith (λ n acc => if n = 1 then acc else if n = 8 then acc else acc + 2 * n^2) side_lengths.tail!.init
  top_cube + bottom_cube + middle_cubes

theorem visible_surface_area_correct : 
  visible_surface_area = 660 :=
by
  sorry

end visible_surface_area_correct_l811_811944


namespace find_value_of_m_l811_811560

noncomputable def m : ℤ := -2

theorem find_value_of_m (m : ℤ) :
  (m-2) ≠ 0 ∧ (m^2 - 3 = 1) → m = -2 :=
by
  intros h
  sorry

end find_value_of_m_l811_811560


namespace min_unitary_polynomial_value_l811_811684

noncomputable def unitary_polynomial_min_value (n : ℕ) (P : Polynomial ℝ) : ℝ := 
  if Polynomial.degree P = n ∧ Polynomial.leadingCoeff P = 1 then 
    max (List.map (λ k, |P.eval k|) (List.range (n+1)))
  else 
    0

theorem min_unitary_polynomial_value (n : ℕ) (P : Polynomial ℝ) (h : Polynomial.degree P = n ∧ Polynomial.leadingCoeff P = 1) : 
  unitary_polynomial_min_value n P ≥ n.factorial / 2 ^ n := 
sorry

end min_unitary_polynomial_value_l811_811684


namespace function_one_solution_in_interval_l811_811180

theorem function_one_solution_in_interval (a : ℝ) :
  (∃ x ∈ Ioo 0 1, (λ x : ℝ, a * x + 1) x = 0) → a < -1 :=
by
  sorry

end function_one_solution_in_interval_l811_811180


namespace length_width_percentage_change_l811_811770

variables (L W : ℝ) (x : ℝ)
noncomputable def area_change_percent : ℝ :=
  (L * (1 + x / 100) * W * (1 - x / 100) - L * W) / (L * W) * 100

theorem length_width_percentage_change (h : area_change_percent L W x = 4) :
  x = 20 :=
by
  sorry

end length_width_percentage_change_l811_811770


namespace intersection_with_y_axis_is_03_l811_811753

-- Define the line equation
def line (x : ℝ) : ℝ := x + 3

-- The intersection point with the y-axis, i.e., where x = 0
def y_axis_intersection : Prod ℝ ℝ := (0, line 0)

-- Prove that the intersection point is (0, 3)
theorem intersection_with_y_axis_is_03 : y_axis_intersection = (0, 3) :=
by
  simp [y_axis_intersection, line]
  sorry

end intersection_with_y_axis_is_03_l811_811753


namespace cells_within_circle_l811_811707

noncomputable def circle_area (r : ℝ) : ℝ := π * r ^ 2

noncomputable def cell_side_length := 1

theorem cells_within_circle (r : ℝ) (ε : ℝ) (h : 0 < ε ∧ ε < 1) :
  let circle_area_r := circle_area r
  let inner_circle_radius := r - cell_side_length
  let inner_circle_area := circle_area inner_circle_radius
  inner_circle_area / circle_area_r ≥ 1 - ε :=
by
  -- Placeholder, to be replaced with actual proof.
  sorry

example : cells_within_circle 1000 0.01 (by norm_num) := by 
  sorry

end cells_within_circle_l811_811707


namespace tetrahedron_volume_calculation_l811_811851

structure Tetrahedron (A B C D : Type) :=
  (angle_ABC_BCD : ℝ) -- Angle between plane ABC and BCD
  (area_ABC : ℝ)     -- Area of triangle ABC
  (area_BCD : ℝ)     -- Area of triangle BCD
  (BC : ℝ)           -- Length of side BC

noncomputable def volume_of_tetrahedron (T : Tetrahedron ℝ ℝ ℝ ℝ) : ℝ :=
  if hD : T.BC ≠ 0 then
    let hD := 2 * T.area_BCD / T.BC in
    let H := hD * Real.sin (T.angle_ABC_BCD / 2) in
    (1/3) * T.area_ABC * H
  else 0

theorem tetrahedron_volume_calculation :
  ∀ (T : Tetrahedron ℝ ℝ ℝ ℝ),
  T.angle_ABC_BCD = Real.pi / 6 →
  T.area_ABC = 120 →
  T.area_BCD = 80 →
  T.BC = 10 →
  volume_of_tetrahedron T = 320 :=
by
  intros T h1 h2 h3 h4
  sorry

end tetrahedron_volume_calculation_l811_811851


namespace identify_fraction_l811_811479

variable {a b : ℚ}

def is_fraction (x : ℚ) (y : ℚ) := ∃ (n : ℚ), x = n / y

theorem identify_fraction :
  is_fraction 2 a ∧ ¬ is_fraction (2 * a) 3 ∧ ¬ is_fraction (-b) 2 ∧ ¬ is_fraction (3 * a + 1) 2 :=
by
  sorry

end identify_fraction_l811_811479


namespace eggs_sold_l811_811610

/-- Define the notion of trays and eggs in this context -/
def trays_of_eggs : ℤ := 30

/-- Define the initial collection of trays by Haman -/
def initial_trays : ℤ := 10

/-- Define the number of trays dropped by Haman -/
def dropped_trays : ℤ := 2

/-- Define the additional trays that Haman's father told him to collect -/
def additional_trays : ℤ := 7

/-- Define the total eggs sold -/
def total_eggs_sold : ℤ :=
  (initial_trays - dropped_trays) * trays_of_eggs + additional_trays * trays_of_eggs

-- Theorem to prove the total eggs sold
theorem eggs_sold : total_eggs_sold = 450 :=
by 
  -- Insert proof here
  sorry

end eggs_sold_l811_811610


namespace hyperbola_asymptote_l811_811600

theorem hyperbola_asymptote (m : ℝ) (h_pos : m > 0) :
  (∃ (x y : ℝ), (x^2 / m^2) - y^2 = 1 ∧ (x + sqrt 3 * y = 0)) → m = sqrt 3 :=
by
  sorry

end hyperbola_asymptote_l811_811600


namespace trajectory_of_z_l811_811016

noncomputable def point_trajectory_is_straight_line (z : ℂ) : Prop :=
  |z - (3 - 4 * complex.I)| = |z + (3 - 4 * complex.I)|

theorem trajectory_of_z (z : ℂ) (h : point_trajectory_is_straight_line z) :
  ∃ a b c : ℝ, ∀ z : ℂ, abs (z - (3 - 4*complex.I)) = abs (z + (3 - 4*complex.I)) → a * z.re + b * z.im + c = 0 :=
sorry

end trajectory_of_z_l811_811016


namespace joan_seashells_l811_811216

theorem joan_seashells (given_to_mike : ℕ) (seashells_left : ℕ) (total_seashells : ℕ) : 
  given_to_mike = 63 → seashells_left = 16 → total_seashells = given_to_mike + seashells_left → total_seashells = 79 :=
by {
  intros h_given h_left h_total,
  rw [h_given, h_left] at h_total,
  exact h_total,
}

end joan_seashells_l811_811216


namespace distance_between_points_l811_811988

theorem distance_between_points : ∀ (x1 y1 x2 y2 : ℤ), 
  (x1 = 2) → (y1 = 5) → (x2 = 7) → (y2 = -1) → 
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = real.sqrt 61 :=
by
  intros x1 y1 x2 y2 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    real.sqrt ((7 - 2) ^ 2 + (-1 - 5) ^ 2)
      = real.sqrt (5 ^ 2 + (-6) ^ 2) : by norm_num
      ... = real.sqrt (25 + 36) : by norm_num
      ... = real.sqrt 61 : by norm_num

end distance_between_points_l811_811988


namespace find_negative_number_l811_811858

theorem find_negative_number : ∃ x ∈ {abs (-2023), 2023⁻¹, -(-2023), -abs (-2023)}, x < 0 := by
  sorry

end find_negative_number_l811_811858


namespace lcm_of_8_9_5_10_l811_811796

theorem lcm_of_8_9_5_10 : Nat.lcm (Nat.lcm 8 9) (Nat.lcm 5 10) = 360 := by
  sorry

end lcm_of_8_9_5_10_l811_811796


namespace find_f3_l811_811562

noncomputable def f : ℤ → ℤ
| x := if x < 6 then f (x + 4) else x - 5

theorem find_f3 : f 3 = 2 :=
by sorry

end find_f3_l811_811562


namespace distinct_prime_factors_of_90_l811_811100

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811100


namespace trajectory_of_Q_l811_811060

variable (x y m n : ℝ)

def line_l (x y : ℝ) : Prop := 2 * x + 4 * y + 3 = 0

def point_P_on_line_l (x y m n : ℝ) : Prop := line_l m n

def origin (O : (ℝ × ℝ)) := O = (0, 0)

def Q_condition (O Q P : (ℝ × ℝ)) : Prop := 2 • O + 2 • Q = Q + P

theorem trajectory_of_Q (x y m n : ℝ) (O : (ℝ × ℝ)) (P Q : (ℝ × ℝ)) :
  point_P_on_line_l x y m n → origin O → Q_condition O Q P → 
  2 * x + 4 * y + 1 = 0 := 
sorry

end trajectory_of_Q_l811_811060


namespace proof_problem_l811_811648

-- Definition of polar coordinate equation for curve C₁
def polar_eq_C1 (ρ θ : ℝ) : Prop := ρ = 24 / (4 * cos θ + 3 * sin θ)

-- Parametric equations for curve C₂
def param_eq_C2 (x y θ : ℝ) : Prop := x = cos θ ∧ y = sin θ

-- Expansion transformation
def expansion_transformation (x' y' x y : ℝ) : Prop :=
  x' = 2 * sqrt 2 * x ∧ y' = 2 * y

-- Prove the rectangular coordinate equation of curve C₁ 
def rectangular_eq_C1 (x y : ℝ) : Prop := 4 * x + 3 * y = 24

-- Prove the ordinary equation of curve C₂
def ordinary_eq_C2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Minimum distance calculation
def min_dist (α φ : ℝ) : ℝ := (24 - 2 * sqrt 41) / 5

theorem proof_problem 
  (ρ θ x y x' y' α φ: ℝ)
  (h1: polar_eq_C1 ρ θ)
  (h2: param_eq_C2 x y θ)
  (h3: expansion_transformation x' y' x y):
  rectangular_eq_C1 x y ∧ ordinary_eq_C2 x y ∧ ∃ α φ, min_dist α φ = (24 - 2 * sqrt 41) / 5 :=
sorry

end proof_problem_l811_811648


namespace fraction_blue_balls_l811_811295

theorem fraction_blue_balls (total_balls : ℕ) (red_fraction : ℚ) (other_balls : ℕ) (remaining_blue_fraction : ℚ) 
  (h1 : total_balls = 360) 
  (h2 : red_fraction = 1/4) 
  (h3 : other_balls = 216) 
  (h4 : remaining_blue_fraction = 1/5) :
  (total_balls - (total_balls / 4) - other_balls) = total_balls * (5 * red_fraction / 270) := 
by
  sorry

end fraction_blue_balls_l811_811295


namespace greatest_difference_l811_811303

def difference_marbles : Nat :=
  let A_diff := 4 - 2
  let B_diff := 6 - 1
  let C_diff := 9 - 3
  max (max A_diff B_diff) C_diff

theorem greatest_difference :
  difference_marbles = 6 :=
by
  sorry

end greatest_difference_l811_811303


namespace lydia_candy_problem_l811_811242

theorem lydia_candy_problem :
  ∃ m: ℕ, (∀ k: ℕ, (k * 24 = Nat.lcm (Nat.lcm 16 18) 20) → k ≥ m) ∧ 24 * m = Nat.lcm (Nat.lcm 16 18) 20 ∧ m = 30 :=
by
  sorry

end lydia_candy_problem_l811_811242


namespace parallelogram_angle_x_l811_811207

theorem parallelogram_angle_x (A B C D : Type) [Parallelogram A B C D] (angle_A : ℝ) (angle_B : ℝ) 
  (angle_A_val : angle_A = 80) (angle_B_val : angle_B = 150) (angle_sum_triangle : ∀ (a b c : ℝ), a + b + c = 180) :
  x = 70 :=
by
  -- We assume the necessary angle conditions for a parallelogram and the sum of angles in a triangle 
  sorry

end parallelogram_angle_x_l811_811207


namespace greatest_difference_l811_811301

def difference_marbles : Nat :=
  let A_diff := 4 - 2
  let B_diff := 6 - 1
  let C_diff := 9 - 3
  max (max A_diff B_diff) C_diff

theorem greatest_difference :
  difference_marbles = 6 :=
by
  sorry

end greatest_difference_l811_811301


namespace sum_gcf_lcm_l811_811359

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l811_811359


namespace sum_of_first_four_squares_geometric_sequence_l811_811645

theorem sum_of_first_four_squares_geometric_sequence (a1 q : ℕ) (h1 : a1 = 1) (hq : q = 2) :
  let a := λ n : ℕ, a1 * q^n in
  let sum_of_squares := (a 0)^2 + (a 1)^2 + (a 2)^2 + (a 3)^2 in
  sum_of_squares = 85 :=
by
  sorry

end sum_of_first_four_squares_geometric_sequence_l811_811645


namespace sequence_bounds_l811_811240

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 0       := 1 / 2
| (k + 1) := sequence k + (1 / n) * (sequence k)^2

theorem sequence_bounds (n : ℕ) : 1 - (1 / n) < sequence n n ∧ sequence n n < 1 := by
  sorry

end sequence_bounds_l811_811240


namespace eval_power_expression_l811_811981

theorem eval_power_expression : (3^3)^2 / 3^2 = 81 := by
  sorry -- Proof omitted as instructed

end eval_power_expression_l811_811981


namespace restaurant_budget_l811_811844

theorem restaurant_budget (B : ℝ) (h₁ : B > 0) (h₂ : \text{Rent} = (1 / 4) * B) (h₃ : \text{FoodAndBeverages} = 0.1875 * B) : 
  \text{(FoodAndBeverages / (B - Rent))} = 0.25 :=
by
  sorry

end restaurant_budget_l811_811844


namespace min_value_x_plus_2y_l811_811578

theorem min_value_x_plus_2y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2 * y - x * y = 0) : x + 2 * y = 8 := 
by
  sorry

end min_value_x_plus_2y_l811_811578


namespace merchant_profit_percentage_l811_811456

def cost_price : ℝ := 100
def markup_percentage : ℝ := 0.40
def discount_percentage : ℝ := 0.15

theorem merchant_profit_percentage :
  let CP := cost_price,
      MP := CP + CP * markup_percentage,
      Discount := MP * discount_percentage,
      SP := MP - Discount in
  (SP - CP) / CP * 100 = 19 :=
by
  sorry

end merchant_profit_percentage_l811_811456


namespace expected_number_of_different_faces_l811_811432

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l811_811432


namespace max_value_of_ci_l811_811687

noncomputable def max_ci (i : ℕ) : ℕ :=
  if i = 1 then 2 else 4 * 3^(i - 2)

theorem max_value_of_ci (c : ℕ → ℕ) (h : ∀ (m n : ℕ), 1 ≤ m ∧ m ≤ ∑ i in finset.range n, c i → 
  ∃ a : fin n → ℕ, m = ∑ i in finset.range n, c i / nat.succ i) 
  : ∀ i, c i ≤ max_ci i :=
sorry

end max_value_of_ci_l811_811687


namespace original_values_l811_811790

noncomputable def vertex_values : Type :=
  { A B C D E F G H : ℝ // 
    (A = (D + E + B) / 3) ∧
    (B = (A + F + C) / 3) ∧
    (C = (D + G + B) / 3) ∧
    (D = (A + C + H) / 3) ∧
    (E = (A + H + F) / 3) ∧
    (F = (E + G + B) / 3) ∧
    (G = (H + F + C) / 3) ∧
    (H = (D + G + E) / 3) }

theorem original_values :
  ∃ (vals : vertex_values),
    vals.A = 0 ∧ vals.B = 12 ∧ vals.C = 6 ∧ vals.D = 3 ∧ 
    vals.E = 3 ∧ vals.F = 3 ∧ vals.G = 3 ∧ vals.H = 6 :=
by
  -- Proof that such vals satisfying the conditions exist
  -- matching the values found in the solution.
  let vals : vertex_values := 
    ⟨0, 12, 6, 3, 3, 3, 3, 6, 
      by norm_num [←(3:ℝ)];
      by norm_num [←(3:ℝ)];
      by norm_num [←(3:ℝ)];
      by norm_num [←(3:ℝ)];
      by norm_num [←(3:ℝ)];
      by norm_num [←(3:ℝ)];
      by norm_num [←(3:ℝ)];
      by norm_num [←(3:ℝ)]⟩
  exact ⟨vals, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

end original_values_l811_811790


namespace max_dot_product_is_4_l811_811863

noncomputable def ellipse_max_dot_product : ℝ :=
  let a := 2 in
  let e := 1 / 2 in
  let b := Real.sqrt (4 - a^2 * (1 - e^2)) in
  let P (x_0 y_0 : ℝ) := ∃ x_0 y_0, x_0^2 / a^2 + y_0^2 / b^2 = 1 in
  let F := (-1 : ℝ, 0 : ℝ) in
  let A := (2 : ℝ, 0 : ℝ) in
  let dot_product := λ x_0 y_0, let PF := (-1 - x_0, -y_0) in
                            let PA := (2 - x_0, -y_0) in
                            (PF.1 * PA.1 + PF.2 * PA.2) in
  ⨆ (x_0 y_0 : ℝ) (hP : P x_0 y_0), dot_product x_0 y_0

theorem max_dot_product_is_4 : ellipse_max_dot_product = 4 := sorry

end max_dot_product_is_4_l811_811863


namespace problem1_problem2_l811_811500

-- Problem 1
theorem problem1 : (Real.sqrt 8 - Real.sqrt 27 - (4 * Real.sqrt (1 / 2) + Real.sqrt 12)) = -5 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem2 : ((Real.sqrt 6 + Real.sqrt 12) * (2 * Real.sqrt 3 - Real.sqrt 6) - 3 * Real.sqrt 32 / (Real.sqrt 2 / 2)) = -18 := by
  sorry

end problem1_problem2_l811_811500


namespace triangle_GKV_similar_triangle_GHS_l811_811941

-- Define the setup
variables {Circle : Type*} [MetricSpace Circle]

structure Point (Circle : Type*) :=
(x y : ℝ)

structure Chord (Circle : Type*) :=
(P Q : Point Circle)

structure Line (Circle : Type*) :=
(start end : Point Circle)

noncomputable def midpoint (P Q : Point Circle) : Point Circle := 
{ x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Assume we have a circle, chords GH and PQ intersect at K, GH is a perpendicular bisector of PQ.
variables (c : Circle) (G H P Q K V S : Point c)
variable (GH : Chord c)
variable (PQ : Chord c)

axiom chord_GH_perpendicular_bisector_PQ : GH = ⟨G, H⟩ ∧ PQ = ⟨P, Q⟩ ∧ midpoint P Q = K ∧ (∃ M, M = midpoint G H ∧ dist G M = dist H M)
axiom point_V_on_PQ : V.x > P.x ∧ V.x < K.x ∧ V.y = P.y
axiom extend_GV_to_S_on_circle : ∃ S : Point c, S ≠ V ∧ Line.mk G V = Line.mk G S

-- Goal
theorem triangle_GKV_similar_triangle_GHS :
  triangle G K V ∼ triangle G H S :=
begin
  -- Proof goes here
  sorry
end

end triangle_GKV_similar_triangle_GHS_l811_811941


namespace tangent_planes_through_line_l811_811570

open EuclideanGeometry

-- Definitions and assumptions
def exists_tangent_planes_to_cone (π : Plane) (O : Point) (r : Real) (S : Point) (g : Line) : Prop :=
  ∃ P T1 T2 : Point, 
  P ∈ g ∧ P ∈ π ∧ 
  TangentLine P O r T1 ∧ TangentLine P O r T2 ∧ 
  ∃ plane1 plane2 : Plane, 
  (TangentPlane S g T1 plane1) ∧ (TangentPlane S g T2 plane2)

axiom TangentLine (P O : Point) (r : Real) (T : Point) : Prop
axiom TangentPlane (S : Point) (g : Line) (T : Point) (π : Plane) : Prop

theorem tangent_planes_through_line (π : Plane) (O : Point) (r : Real) (S : Point) (g : Line) :
  exists_tangent_planes_to_cone π O r S g :=
sorry

end tangent_planes_through_line_l811_811570


namespace can_Petya_lose_l811_811716

open Function

structure Board (n : ℕ) where
  cells : ℕ × ℕ → bool

def is_boundary (n i j : ℕ) : Prop :=
  i = 0 ∨ i = n - 1 ∨ j = 0 ∨ j = n - 1

def adjacent (i j m n : ℕ) : Prop :=
  (abs (i - m) = 1 ∧ j = n) ∨ (abs (j - n) = 1 ∧ i = m)

def symmetrical (n i j : ℕ) : Prop :=
  abs (i - (n/2)) = abs (j - (n/2))

inductive Player
| Petya
| Vasya
| Tolya

noncomputable def next_player : Player → Player
| Player.Petya := Player.Vasya
| Player.Vasya := Player.Tolya
| Player.Tolya := Player.Petya

def valid_move (b : Board 100) (i j : ℕ) : Prop :=
  is_boundary 100 i j ∧
  ¬ (∃ m n, (adjacent i j m n ∨ symmetrical 100 i j) ∧ b.cells (m, n))

theorem can_Petya_lose (b : Board 100) :
  (∃ strategy_for_Vasya_and_Tolya : (Board 100 → Player → (ℕ × ℕ)) →
      ∀ moves : (ℕ × ℕ), valid_move b (moves.fst) (moves.snd),
      ¬ valid_move b (strategy_for_Vasya_and_Tolya b Player.Petya).fst
                     (strategy_for_Vasya_and_Tolya b Player.Petya).snd) :=
sorry

end can_Petya_lose_l811_811716


namespace sum_of_coefficients_l811_811737

noncomputable def problem_expr (d : ℝ) := (16 * d + 15 + 18 * d^2 + 3 * d^3) + (4 * d + 2 + d^2 + 2 * d^3)
noncomputable def simplified_expr (d : ℝ) := 5 * d^3 + 19 * d^2 + 20 * d + 17

theorem sum_of_coefficients (d : ℝ) (h : d ≠ 0) : 
  problem_expr d = simplified_expr d ∧ (5 + 19 + 20 + 17 = 61) := 
by
  sorry

end sum_of_coefficients_l811_811737


namespace part1_part2_l811_811720

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

-- Part 1
theorem part1 (n : ℕ) :
  (∑ i in finset.range (n+1), (binomial_coefficient n i)^2) = binomial_coefficient (2*n) n := sorry

-- Part 2
theorem part2 (n : ℕ) :
  (∑ k in finset.range (n+1), (binomial_coefficient (2*n) (2*k + 1))^2) =
  (1 / 2) * (binomial_coefficient (4*n) (2*n) + (-1 : ℤ)^(n-1) * binomial_coefficient (2*n) n) := sorry

end part1_part2_l811_811720


namespace gas_volume_at_10_degrees_l811_811550

theorem gas_volume_at_10_degrees (V T : ℕ) 
  (h1 : ∀ t, V (t + 5) = V t + 6)
  (h2 : V 30 = 100) : V 10 = 76 := 
by {
  sorry
}

end gas_volume_at_10_degrees_l811_811550


namespace probability_of_selecting_3_co_captains_is_correct_l811_811296

def teams : List ℕ := [4, 6, 7, 9]

def probability_of_selecting_3_co_captains (n : ℕ) : ℚ :=
  if n = 4 then 1/4
  else if n = 6 then 1/20
  else if n = 7 then 1/35
  else if n = 9 then 1/84
  else 0

def total_probability : ℚ :=
  (1/4) * (probability_of_selecting_3_co_captains 4 +
            probability_of_selecting_3_co_captains 6 +
            probability_of_selecting_3_co_captains 7 +
            probability_of_selecting_3_co_captains 9)

theorem probability_of_selecting_3_co_captains_is_correct :
  total_probability = 143 / 1680 :=
by
  -- The proof will be inserted here
  sorry

end probability_of_selecting_3_co_captains_is_correct_l811_811296


namespace max_teams_l811_811191

theorem max_teams (n : ℕ) (cond1 : ∀ t, card t = 3) (cond2 : ∀ t1 t2 (ht1 : t1 ≠ t2), 
  ∀ p1 ∈ t1, ∀ p2 ∈ t2, p1 ≠ p2) (cond3 : 9 * n * (n - 1) / 2 ≤ 200) : 
  n ≤ 7 := 
sorry

end max_teams_l811_811191


namespace average_speed_of_car_l811_811426

theorem average_speed_of_car : 
  let distance1 := 30
  let speed1 := 60
  let distance2 := 35
  let speed2 := 70
  let distance3 := 36
  let speed3 := 80
  let distance4 := 20
  let speed4 := 55
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  average_speed = 66.70 := sorry

end average_speed_of_car_l811_811426


namespace intersection_point_with_y_axis_l811_811752

theorem intersection_point_with_y_axis : 
  ∃ y, (0, y) = (0, 3) ∧ (y = 0 + 3) :=
by
  sorry

end intersection_point_with_y_axis_l811_811752


namespace sum_reciprocal_a_l811_811064

noncomputable def a : (ℕ → ℝ)
| 0     := 3
| (n+1) := 3 - ((18 : ℝ) / (6 + a n))

theorem sum_reciprocal_a (n : ℕ) :
  (∑ i in Finset.range (n + 1), 1 / a i) = (1 / 3) * (2 ^ (n + 2) - n - 3) :=
by
  sorry

end sum_reciprocal_a_l811_811064


namespace smallest_n_for_irreducible_fractions_l811_811542

open Nat

theorem smallest_n_for_irreducible_fractions :
  (∃ n : ℕ, (∀ k : ℕ, 19 ≤ k ∧ k ≤ 91 → gcd k (n + 2) = 1) ∧
   (∀ m : ℕ, (∀ k : ℕ, 19 ≤ k ∧ k ≤ 91 → gcd k (m + 2) = 1) → n ≤ m)) ∧
  someN = 95 :=
by
  let candidates := {n | (∀ k : ℕ, 19 ≤ k ∧ k ≤ 91 → gcd k (n + 2) = 1)}
  have smallest := well_founded.has_min (measure_wf id) candidates
  exact ⟨smallest, sorry⟩

end smallest_n_for_irreducible_fractions_l811_811542


namespace ratio_of_areas_l811_811639

-- Definition of a regular hexagon and its properties
structure RegularHexagon (α : Type) :=
  (A B C D E F : α)
  (is_regular : true) -- Placeholder for regularity condition

-- Midpoint definition (more formally would require coordinates, skipped for brevity)
def midpoint {α : Type} (x y : α) : α := sorry

-- Area definition for the quadrilateral in the hexagon context
def area {α : Type} (x y z w : α) : ℝ := sorry

-- Hexagon instance
variables (H : RegularHexagon ℕ)
variables (M N : ℕ)
variables (is_midpoint_M : M = midpoint H.A H.B)
variables (is_midpoint_N : N = midpoint H.D H.E)

-- Proposition about the area ratios
theorem ratio_of_areas : area H.A M N H.F = area H.F M N H.C :=
by
  sorry

end ratio_of_areas_l811_811639


namespace terminal_side_of_1000_degrees_l811_811290

def full_rotation := 360

def angle := 1000

def quadrant (θ : ℕ) : ℕ :=
  match θ % full_rotation with
  | n if 0 < n ∧ n <= 90  => 1
  | n if 90 < n ∧ n <= 180 => 2
  | n if 180 < n ∧ n <= 270 => 3
  | _ => 4

theorem terminal_side_of_1000_degrees :
  quadrant angle = 4 :=
by
  sorry

end terminal_side_of_1000_degrees_l811_811290


namespace total_cost_of_flower_pots_l811_811697

theorem total_cost_of_flower_pots
  (cost_largest_pot : ℝ)
  (increment_between_pots : ℝ)
  (cost_largest_is_correct : cost_largest_pot = 1.925)
  (increment_between_is_correct : increment_between_pots = 0.25)
  : 
  ∃ (cost_smallest_pot : ℝ),
  let x := cost_smallest_pot
  in
  x = cost_largest_pot - 5 * increment_between_pots ∧ 
  x + (x + increment_between_pots) + (x + 2 * increment_between_pots) + (x + 3 * increment_between_pots) + (x + 4 * increment_between_pots) + (x + 5 * increment_between_pots) = 7.80 := 
begin
  use (1.925 - 5 * 0.25), -- x = 0.675
  split,
  { simp [cost_largest_pot, increment_between_pots, cost_largest_is_correct, increment_between_is_correct] },
  { 
    have h : 0.675 + (0.675 + 0.25) + (0.675 + 0.50) + (0.675 + 0.75) + (0.675 + 1.00) + (0.675 + 1.25) = 7.80,
    { norm_num },
    exact h
  },
end


end total_cost_of_flower_pots_l811_811697


namespace series_sum_l811_811938

theorem series_sum :
  (∑ k in Finset.range 1008, (4 * k + 1) * (4 * k + 2) - (4 * k + 3) * (4 * k + 4)) = -1014204008 :=
by
  sorry

end series_sum_l811_811938


namespace light_flashes_l811_811840

theorem light_flashes (t1 t2 t3 : ℕ) (r1 r2 r3 : ℕ) (n1 n2 n3 : ℕ) 
  (h1 : t1 = 5 * 60) (h2 : r1 = 3) (h3 : n1 = t1 / r1)
  (h4 : t2 = 15 * 60) (h5 : r2 = 5) (h6 : n2 = t2 / r2)
  (h7 : t3 = 40.5 * 60) (h8 : r3 = 7) (h9 : n3 = t3 / r3) : 
  n1 + n2 + n3 = 627 := 
by 
  -- proof goes here 
  sorry

end light_flashes_l811_811840


namespace Anton_thought_of_729_l811_811912

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l811_811912


namespace quiz_score_difference_is_4_l811_811634

noncomputable def mean_of_scores : ℝ :=
  0.20 * 60 + 0.25 * 75 + 0.25 * 85 + 0.15 * 95 + 0.15 * 100

def median_of_scores : ℝ := 85

def difference_between_mean_and_median : ℝ := median_of_scores - mean_of_scores

theorem quiz_score_difference_is_4 :
  difference_between_mean_and_median = 4 := 
sorry

end quiz_score_difference_is_4_l811_811634


namespace divisibility_criterion_l811_811985

theorem divisibility_criterion (n : ℕ) : 
  (20^n - 13^n - 7^n) % 309 = 0 ↔ 
  ∃ k : ℕ, n = 1 + 6 * k ∨ n = 5 + 6 * k := 
  sorry

end divisibility_criterion_l811_811985


namespace problem_statement_l811_811423

-- Define the number of balls and the counts of red and black balls
def total_balls := 10
def red_balls := 4
def black_balls := 6

-- Define binomial probability parameters for experiment 1
def n1 := 3
def p1 := (red_balls : ℚ) / total_balls

-- Define expectation and variance for experiment 1
def E_X1 : ℚ := n1 * p1
def D_X1 : ℚ := n1 * p1 * (1 - p1)

-- Define probabilities for different counts of red balls in experiment 2
def P_X2_0 : ℚ := (choose red_balls 0 * choose black_balls 3) / choose total_balls 3
def P_X2_1 : ℚ := (choose red_balls 1 * choose black_balls 2) / choose total_balls 3
def P_X2_2 : ℚ := (choose red_balls 2 * choose black_balls 1) / choose total_balls 3
def P_X2_3 : ℚ := (choose red_balls 3 * choose black_balls 0) / choose total_balls 3

-- Define expectation for experiment 2
def E_X2 : ℚ := 0 * P_X2_0 + 1 * P_X2_1 + 2 * P_X2_2 + 3 * P_X2_3

-- Define variance for experiment 2
def D_X2 : ℚ := (0 - E_X2) ^ 2 * P_X2_0 + (1 - E_X2) ^ 2 * P_X2_1 + (2 - E_X2) ^ 2 * P_X2_2 + (3 - E_X2) ^ 2 * P_X2_3

-- Proof problem statement
theorem problem_statement : E_X1 = E_X2 ∧ D_X1 > D_X2 := by
  sorry

end problem_statement_l811_811423


namespace scientific_notation_2400000_l811_811772

def scientific_notation (x : ℝ) : Prop :=
  ∃ a n : ℝ, (1 ≤ |a| ∧ |a| < 10) ∧ x = a * 10^n

theorem scientific_notation_2400000 : scientific_notation 2400000 :=
by
  use 2.4
  use 6
  split
  . split
    . norm_num
    . norm_num
  . norm_num

end scientific_notation_2400000_l811_811772


namespace marble_probability_l811_811343

def total_marbles : Nat := 4 + 3 + 8
def favorable_outcomes : Nat := 4 + 3
def probability : ℚ := favorable_outcomes / total_marbles

theorem marble_probability : probability.toReal ≈ 0.4667 := sorry

end marble_probability_l811_811343


namespace boys_in_fifth_grade_l811_811644

theorem boys_in_fifth_grade (T S : ℕ) (percent_boys_soccer : ℝ) (girls_not_playing_soccer : ℕ) 
    (hT : T = 420) (hS : S = 250) (h_percent : percent_boys_soccer = 0.86) 
    (h_girls_not_playing_soccer : girls_not_playing_soccer = 65) : 
    ∃ B : ℕ, B = 320 :=
by
  -- We don't need to provide the proof details here
  sorry

end boys_in_fifth_grade_l811_811644


namespace hexagon_colorings_count_l811_811524

-- Define parameters
constant Color : Type
constant hexagon_vertices : Fin 6 → Type

-- There are 7 possible colors
constant seven_colors : Fin 7 → Color

-- Define the function assigning colors to vertices such that no two vertices connected 
-- by a diagonal (i.e., (A, D), (B, E), (C, F)) have the same color.
noncomputable def color_hexagon (c : hexagon_vertices → Color) :=
  ∀ i j, (i ≠ j) → (abs (i.1 - j.1) ∈ [3]) → c i ≠ c j

-- Define the problem statement
theorem hexagon_colorings_count : ∃ C : ℕ, C = 27216 ∧
  ∃ c : hexagon_vertices → Color, color_hexagon c :=
begin
  sorry
end

end hexagon_colorings_count_l811_811524


namespace anton_thought_of_729_l811_811864

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l811_811864


namespace sum_of_cubes_consecutive_integers_divisible_by_9_l811_811252

theorem sum_of_cubes_consecutive_integers_divisible_by_9 (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 :=
sorry

end sum_of_cubes_consecutive_integers_divisible_by_9_l811_811252


namespace distinct_prime_factors_count_l811_811118

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811118


namespace Anton_thought_number_is_729_l811_811908

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l811_811908


namespace intersection_with_y_axis_l811_811757

theorem intersection_with_y_axis (x y : ℝ) (h : y = x + 3) (hx : x = 0) : (x, y) = (0, 3) := 
by 
  subst hx 
  rw [h]
  rfl
-- sorry to skip the proof

end intersection_with_y_axis_l811_811757


namespace revenue_comparison_l811_811622

theorem revenue_comparison 
  (D N J F : ℚ) 
  (hN : N = (2 / 5) * D) 
  (hJ : J = (2 / 25) * D) 
  (hF : F = (3 / 4) * D) : 
  D / ((N + J + F) / 3) = 100 / 41 := 
by 
  sorry

end revenue_comparison_l811_811622


namespace anton_thought_number_l811_811877

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l811_811877


namespace find_ellipse_equation_max_area_triangle_OAB_l811_811024

-- Define the conditions and given data
def ellipse_eq (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1

def eccentricity (c a : ℝ) : Prop :=
  c / a = 1 / 2

def focus_distance (c : ℝ) : Prop :=
  3 * c / (real.sqrt (3^2 + 4^2)) = 3 / 5

-- Part I: Equation of the ellipse
theorem find_ellipse_equation (a b c : ℝ) (h_ellipse : ellipse_eq a b)
    (h_ecc : eccentricity c a) (h_dist : focus_distance c) :
  a = 2 ∧ b = real.sqrt 3 := by sorry

-- Part II: Maximum area of triangle OAB
theorem max_area_triangle_OAB (k m : ℝ) (h_line : k ≠ 0) (h_ellipse_eq : a = 2 ∧ b = real.sqrt 3)
    (h_midpoint : ∀ x1 y1 x2 y2 : ℝ,
      y1 = k * x1 + m ∧ y2 = k * x2 + m ∧ ((3 * (-4 * k * m) / (4 * k^2 + 3)) + (4 * (3 * m) / (4 * k^2 + 3))) = 0) :
  (∃ S : ℝ, S = real.sqrt 3) := by sorry

end find_ellipse_equation_max_area_triangle_OAB_l811_811024


namespace Anton_thought_number_is_729_l811_811905

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l811_811905


namespace g_n_minus_g_n_eq_zero_l811_811665

def f : ℕ → ℕ
| 0       := 0
| (2*n+1) := 2 * f n
| (n+1+1) := 2 * f (n/2) + 1

def g (n : ℕ) : ℕ := f (f n)

theorem g_n_minus_g_n_eq_zero (n : ℕ) : g (n - g n) = 0 :=
by
  sorry

end g_n_minus_g_n_eq_zero_l811_811665


namespace cone_lateral_to_total_surface_area_ratio_l811_811623

theorem cone_lateral_to_total_surface_area_ratio
  (r : ℝ) 
  (h : r > 0)
  (E : ∀ (C : Cone), C.axial_section = EquilateralTriangle ∧ C.base_radius = r → C.lateral_area = 2 * π * r^2 ∧ C.total_area = 3 * π * r^2):
  2 / 3 :=
by
  sorry

end cone_lateral_to_total_surface_area_ratio_l811_811623


namespace john_mary_game_l811_811666

theorem john_mary_game (n : ℕ) (h : n ≥ 3) :
  ∃ S : ℕ, S = n * (n + 1) :=
by
  sorry

end john_mary_game_l811_811666


namespace number_of_diagonals_dodecagon_l811_811071

theorem number_of_diagonals_dodecagon : 
  let n := 12 
  in (n * (n - 3)) / 2 = 54 := 
by
  let n := 12
  have h1 : n * (n - 3) = 108 := by sorry
  have h2 : 108 / 2 = 54 := by sorry
  exact Eq.trans (Eq.trans (Eq.trans rfl h1) (Eq.symm h2)) rfl

end number_of_diagonals_dodecagon_l811_811071


namespace anton_thought_number_l811_811878

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l811_811878


namespace distinct_prime_factors_of_90_l811_811093

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811093


namespace arithmetic_sequence_problem_l811_811679

variable {a : ℕ → ℤ} (d : ℤ)

-- Arithmetic sequence assumption
axiom common_difference : ∀ n : ℕ, a (n + 1) - a n = d

-- Condition
axiom sum_condition : ∑ i in finset.range 33, a (1 + 3 * i) = 50

-- Value we want to prove
theorem arithmetic_sequence_problem : a 3 + a 6 + ⋯ + a 99 = -82 :=
by
    -- proof goes here
sorry

end arithmetic_sequence_problem_l811_811679


namespace square_area_when_a_eq_b_eq_c_l811_811211

theorem square_area_when_a_eq_b_eq_c {a b c : ℝ} (h : a = b ∧ b = c) :
  ∃ x : ℝ, (x = a * Real.sqrt 2) ∧ (x ^ 2 = 2 * a ^ 2) :=
by
  sorry

end square_area_when_a_eq_b_eq_c_l811_811211


namespace phi_values_l811_811285

noncomputable def symmetric_cosine (f : ℝ → ℝ) (φ : ℝ) : Prop :=
  ∀ x, f (-x) = f (x)

noncomputable def cos_func (φ : ℝ) : ℝ → ℝ :=
  λ x, Real.cos (3 * x + φ)

theorem phi_values (φ : ℝ) (k : ℤ) : 
  (symmetric_cosine (cos_func φ) φ) → φ = k * Real.pi + Real.pi / 2 :=
by
  intros h
  sorry

end phi_values_l811_811285


namespace unique_fixed_point_l811_811579

universe u

-- Definitions for points, lines, and the set of all lines in a plane
def Point := Type u
def Line := Type u
def S (Plane : Type u) := Plane 

-- Function f which assigns a point to each line
def f {Plane : Type u} (L : Line → Plane) : (∀ l : Line, (L l) → Point) := sorry

-- Main theorem statement asserting the existence of a unique point P
theorem unique_fixed_point {Plane : Type u} (L: Line → Plane)
  (h₁ : ∀ l : Line, (f L l) = L l)
  (h₂ : ∀ (X : Point) (l₁ l₂ l₃ : Line), l₁ ≠ l₂ → l₂ ≠ l₃ → l₁ ≠ l₃ → 
    (X = f L l₁) → (X = f L l₂) → (X = f L l₃) → (f L l₁, f L l₂, f L l₃, X).collinear) :
  ∃! P : Point, ∀ l : Line, L l = P → f L l = P := sorry

end unique_fixed_point_l811_811579


namespace max_teams_participation_l811_811189

theorem max_teams_participation (n : ℕ) (H : 9 * n * (n - 1) / 2 ≤ 200) : n ≤ 7 := by
  -- Proof to be filled in
  sorry

end max_teams_participation_l811_811189


namespace intersection_result_l811_811605

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 5 }
def M_compl : Set ℝ := { x | x < 1 }

theorem intersection_result : N ∩ M_compl = { x | 0 ≤ x ∧ x < 1 } :=
by sorry

end intersection_result_l811_811605


namespace find_parallel_line_eq_l811_811989

-- Defining the conditions
def parallel_line (x y : ℝ) (c : ℝ) : Prop := 3 * x - 4 * y + c = 0
def distance (c1 c2 : ℝ) : ℝ := abs (c2 - c1) / real.sqrt (3^2 + (-4)^2)

-- Proving the final result
theorem find_parallel_line_eq (c : ℝ) : 
  (distance 1 c = 3) ↔ (c = 16 ∨ c = -14) :=
sorry

end find_parallel_line_eq_l811_811989


namespace min_pairs_of_acute_angle_segments_l811_811522

-- Problem: Given a large equilateral triangle side length n, divided into n^2 smaller triangles by parallel lines
-- through every vertex exactly once, prove there are at least n pairs of adjacent segments forming an acute angle.
theorem min_pairs_of_acute_angle_segments {n : ℕ} (h : n ≥ 1) : 
  ∃ M, M ≥ n ∧ (∀ l, broken_line_passes_through_vertices l) → has_min_acute_pairs l M :=
sorry

end min_pairs_of_acute_angle_segments_l811_811522


namespace parabola_focus_l811_811276

theorem parabola_focus (x : ℝ) : ∃ f : ℝ × ℝ, f = (0, 1 / 4) ∧ ∀ y : ℝ, y = x^2 → f = (0, 1 / 4) :=
by
  sorry

end parabola_focus_l811_811276


namespace sum_first_9_terms_eq_0_l811_811268

variable {a : ℕ → ℝ}

-- Definitions based on given conditions
def a6 : ℝ := a 6
def sum_a3_a8 : ℝ := a 3 + a 8

-- Given conditions
axiom a6_is_5 : a6 = 5
axiom sum_a3_a8_is_5 : sum_a3_a8 = 5

-- Prove that the sum of the first 9 terms of the arithmetic sequence is 0
theorem sum_first_9_terms_eq_0 : (∑ i in Finset.range 9, a i) = 0 :=
by
  sorry

end sum_first_9_terms_eq_0_l811_811268


namespace number_of_distinct_prime_factors_of_90_l811_811111

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811111


namespace anton_thought_number_l811_811899

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l811_811899


namespace zoo_initial_animals_l811_811848

theorem zoo_initial_animals (X : ℕ) :
  X - 6 + 1 + 3 + 8 + 16 = 90 → X = 68 :=
by
  intro h
  sorry

end zoo_initial_animals_l811_811848


namespace lean_proof_l811_811592

variables {f g : ℝ → ℝ}
variables (w α : ℝ)

noncomputable def f_def := λ x, 2 * sqrt 3 * sin (w * x + π / 6) * cos (w * x)

noncomputable def g_def := λ x, 2 * sqrt 3 * sin (w * (x - π / 6) + π / 6) * cos (w * (x - π / 6))

lemma problem_conditions :
  (0 < w ∧ w < 2) ∧
  (f_def w (5 * π / 12) = sqrt 3 / 2) ∧
  (g_def w (α / 2) = 5 * sqrt 3 / 6) :=
sorry

theorem lean_proof :
  (∃ w, (0 < w ∧ w < 2) ∧
    w = 1 ∧ period (f_def w) = π ∧
    ∃ α, g_def w (α/2) = 5 * sqrt 3 / 6 ∧
    cos (2 * α - π / 3) = 7 / 9) :=
sorry

end lean_proof_l811_811592


namespace PQ_perp_BI_l811_811749

open Set Classical

noncomputable theory

variables {α : Type*} [EuclideanSpace α]

-- Definitions of points P, Q, I, and B
variable {A B C I P Q : α}

-- Conditions
axiom bisector_CA_Circle_intersects (A B C P Q I : α) :
  -- Bisectors intersect circumcircle at points P and Q respectively
  Bisector (Angle CAB) (Circumcircle A B C) P ∧
  Bisector (Angle BCA) (Circumcircle A B C) Q ∧
  -- Bisectors intersect each other at I
  Bisector (Angle CAB) I ∧
  Bisector (Angle BCA) I

-- To prove
theorem PQ_perp_BI (A B C I P Q : α) :
  Bisector (Angle CAB) (Circumcircle A B C) P →
  Bisector (Angle BCA) (Circumcircle A B C) Q →
  Bisector (Angle CAB) I →
  Bisector (Angle BCA) I →
  Perpendicular PQ BI :=
  sorry

end PQ_perp_BI_l811_811749


namespace distinct_prime_factors_of_90_l811_811091

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811091


namespace blue_flowers_percentage_l811_811199

theorem blue_flowers_percentage :
  let total_flowers := 96
  let green_flowers := 9
  let red_flowers := 3 * green_flowers
  let yellow_flowers := 12
  let accounted_flowers := green_flowers + red_flowers + yellow_flowers
  let blue_flowers := total_flowers - accounted_flowers
  (blue_flowers / total_flowers : ℝ) * 100 = 50 :=
by
  sorry

end blue_flowers_percentage_l811_811199


namespace least_number_remainder_1_l811_811814

theorem least_number_remainder_1 (x : ℕ) :
  (x % 35 = 1 ∧ x % 11 = 1) -> x = 386 :=
begin
  intro h,
  sorry
end

end least_number_remainder_1_l811_811814


namespace log_fraction_pow_l811_811968

theorem log_fraction_pow : log (1 / 4) 16 = -2 :=
by {
    sorry
}

end log_fraction_pow_l811_811968


namespace women_attended_gathering_l811_811490

theorem women_attended_gathering :
  ∀ (m : ℕ) (w_per_man : ℕ) (m_per_woman : ℕ),
  m = 15 ∧ w_per_man = 4 ∧ m_per_woman = 3 →
  ∃ (w : ℕ), w = 20 :=
by
  intros m w_per_man m_per_woman h,
  cases h with hm hw_wom,
  cases hw_wom with hwm hmw,
  sorry

end women_attended_gathering_l811_811490


namespace anton_thought_of_729_l811_811865

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l811_811865


namespace average_side_length_of_squares_l811_811274

theorem average_side_length_of_squares (a b c : ℕ) (ha : a = 25) (hb : b = 64) (hc : c = 225) :
  (\(\frac{\sqrt{a} + \sqrt{b} + \sqrt{c}}{3}\)) = \(\frac{28}{3}\) :=
by
  sorry

end average_side_length_of_squares_l811_811274


namespace timPaid_l811_811313

def timPayment (p : Real) (d : Real) : Real :=
  p - p * d

theorem timPaid (p : Real) (d : Real) (a : Real) : 
  p = 1200 ∧ d = 0.15 → a = 1020 :=
by
  intro h
  cases h with hp hd
  rw [hp, hd]
  have hdiscount : 1200 * 0.15 = 180 := by norm_num
  have hpayment : 1200 - 180 = 1020 := by norm_num
  rw [hdiscount, hpayment]
  sorry

end timPaid_l811_811313


namespace g_value_l811_811762

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value (h : ∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x) :
  g 3 = -(27 + 3 * (3:ℝ)^(1/3)) / 8 :=
sorry

end g_value_l811_811762


namespace ratio_of_areas_l811_811717

theorem ratio_of_areas (ABC : Type*) [linear_ordered_field ABC] (s : ABC) (A B C D : ABC)
  (h_equilateral : ∀ {X Y Z : ABC}, X = A → Y = B → Z = C → Triangle X Y Z)
  (h120 : ∃ (X Y: ABC), X = D → distance X Y = s)
  (h60 : ∠DBA = 60)
  (h_on_AC : D ∈ segment A C):
  (area A D B / area C D B) = (sqrt 3 / 3) :=
by
  sorry

end ratio_of_areas_l811_811717


namespace increase_productivity_RnD_l811_811934

theorem increase_productivity_RnD :
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  RnD_t / ΔAPL_t2 = 3260 :=
by
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  have h : RnD_t / ΔAPL_t2 = 3260 := sorry
  exact h

end increase_productivity_RnD_l811_811934


namespace distinct_prime_factors_90_l811_811157

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811157


namespace distinct_prime_factors_of_90_l811_811088

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811088


namespace anton_thought_number_l811_811925

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l811_811925


namespace seq_a_formula_T_n_formula_max_n_not_exist_l811_811239

def seq_a (n : ℕ) : ℕ := by
  if n = 1 then exact 2
  else if n = 2 then exact 8
  else sorry

def S (n : ℕ) : ℕ := by
  if n = 0 then exact 0
  else (nat.sum $ λ i, seq_a₀ i) n

def seq_log2_a (n : ℕ) : ℕ :=
  log 2 (seq_a₀ n)

def T (n : ℕ) : ℕ :=
  nat.sum $ λ i, seq_log2_a₀ i

theorem seq_a_formula (n : ℕ) (hn : n > 0) : 
  seq_a₀ n = 2^(2*n - 1) :=
sorry

theorem T_n_formula (n : ℕ) (hn : n > 0) :
  T₀ n = n^2 :=
sorry

theorem max_n_not_exist (n : ℕ) (hn : n ≥ 2) :
  ¬ ∃ n, (∀ k, 2 ≤ k → k ≤ n → (1 - (1 / (T₀ k))) > (2013 / 2014)) :=
sorry

end seq_a_formula_T_n_formula_max_n_not_exist_l811_811239


namespace area_of_octagon_in_rectangle_l811_811028

theorem area_of_octagon_in_rectangle (BDEF : Set Point) (h_rect : IsRectangle BDEF)
  (A B C : Point) (h_AB : distance A B = 1) (h_BC : distance B C = 2):
  let AC := distance A C in
  let AC_sq := AC ^ 2 in
  let AB_sq := distance A B ^ 2 in
  let BC_sq := distance B C ^ 2 in
  (AC_sq = AB_sq + BC_sq) ->
  let side_octagon := Real.sqrt 5 in
  let area_tri := 1 * 2 / 2 in
  let total_area_tri := 4 * area_tri in
  let side_length_BDEF := 2 + Real.sqrt 5 in
  let area_rect := (2 + Real.sqrt 5) * (4 + 2 * Real.sqrt 5) in
  let area_octagon := area_rect - total_area_tri in
  area_octagon = 14 + 8 * Real.sqrt 5 :=
by sorry

end area_of_octagon_in_rectangle_l811_811028


namespace codes_available_for_reckha_l811_811703

-- Define a function that verifies if two three-digit codes differ in one digit
def differs_in_one_digit (a b : ℕ) : Prop :=
  let digits (x : ℕ) := [div x 100, (div x 10) % 10, x % 10]
  (list.zip_with ne (digits a) (digits b)).count (true) = 1

-- Define the total number of three-digit codes
def total_codes : ℕ := 1000

-- Define my code
def my_code : ℕ := 317

-- Define the set of codes differing in exactly one digit from my code
def codes_differs_in_one : set ℕ := {b | differs_in_one_digit my_code b}

-- Define the available codes for Reckha
def available_codes_for_reckha : ℕ :=
  total_codes - codes_differs_in_one.to_finset.card - 1

-- The theorem we need to prove
theorem codes_available_for_reckha : available_codes_for_reckha = 972 := by
  sorry

end codes_available_for_reckha_l811_811703


namespace flat_ground_distance_l811_811469

theorem flat_ground_distance (x : ℝ) (t_total : ℝ)
  (h_speeds : ∀ d s, d / s < t_total)
  (h_total_distance : ∀ d1 d2 d3, d1 + d2 + d3 = 2 * 9)
  (h_times_ratio : t_total = (9 - x) / 4 + (9 - x) / 6 + 2 * x / 5)
  (t_total_value : t_total = 221/60) :
  x = 4 :=
begin
  have h1 : (9 - x) / 4 + (9 - x) / 6 = (5 * (9 - x)) / 12,
  {
    sorry,
  },
  have h2 : 2 * x / 5 = x * 2 / 5,
  {
    sorry,
  },
  rw [h1, h2] at h_times_ratio,
  sorry
end

end flat_ground_distance_l811_811469


namespace equation_of_line_AC_l811_811045

-- Definitions of points and lines
structure Point :=
  (x : ℝ)
  (y : ℝ)

def line_equation (A B C : ℝ) (P : Point) : Prop :=
  A * P.x + B * P.y + C = 0

-- Given points and lines
def B : Point := ⟨-2, 0⟩
def altitude_on_AB (P : Point) : Prop := line_equation 1 3 (-26) P

-- Required equation of line AB
def line_AB (P : Point) : Prop := line_equation 3 (-1) 6 P

-- Angle bisector given in the condition
def angle_bisector (P : Point) : Prop := line_equation 1 1 (-2) P

-- Derived Point A
def A : Point := ⟨-1, 3⟩

-- Symmetric point B' with respect to the angle bisector
def B' : Point := ⟨2, 4⟩

-- Required equation of line AC
def line_AC (P : Point) : Prop := line_equation 1 (-3) 10 P

-- The proof statement
theorem equation_of_line_AC :
  ∀ P : Point, (line_AB B ∧ angle_bisector A ∧ P = A → P = B' → line_AC P) :=
by
  intros P h h1 h2
  sorry

end equation_of_line_AC_l811_811045


namespace binom_factorial_binom_expansion_sum_binomial_even_l811_811548

noncomputable def binom : ℕ → ℕ → ℕ
| n 0       := 1
| n k@(k' + 1) := if n = k then 1 else binom n k' + binom (n - 1) k

theorem binom_factorial (n k : ℕ) (h : k ≤ n) :
  binom n k = n.factorial / (k.factorial * (n - k).factorial) :=
sorry

theorem binom_expansion (a b : ℝ) (n : ℕ) :
  (a + b)^n = ∑ k in finset.range (n+1), binom n k * a ^ k * b ^ (n - k) :=
sorry

theorem sum_binomial_even (n : ℕ) :
  ∑ k in finset.range (n + 1), binom (2 * n) (2 * k) = 2 ^ (2 * n - 1) :=
sorry

end binom_factorial_binom_expansion_sum_binomial_even_l811_811548


namespace twenty_fifth_number_in_base_five_l811_811638

theorem twenty_fifth_number_in_base_five : 
  let n := 25
  ∃ b, b = 5 ∧ nat.change_base n b = 100 :=
by sorry

end twenty_fifth_number_in_base_five_l811_811638


namespace c_n_smallest_l811_811572

theorem c_n_smallest (n : ℕ) (h1 : n ≥ 2) :
  let c := (n - 1 : ℝ) / n in
  ∀ (a : Fin n → ℝ), 
  (∀ i, 0 < a i) →
  (∑ i, a i) / n.toReal - Real.exp ((∑ i, (Real.log (a i) / n.toReal)) : ℝ) ≤ 
  c * (Finset.univ.sup (λ ⟨i, hi⟩, Finset.univ.sup (λ ⟨j, hj⟩, (Real.sqrt (a i) - Real.sqrt (a j))^2))) :=
sorry

end c_n_smallest_l811_811572


namespace sum_gcf_lcm_l811_811353

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l811_811353


namespace problem_l811_811620

theorem problem (x y : ℝ) (h : |x - 3| + (y + 4)^2 = 0) : (x + y)^2023 = -1 := by
  have : |x - 3| = 0 := by sorry -- Placeholder for the necessary proof
  have : (y + 4)^2 = 0 := by sorry -- Placeholder for the necessary proof
  have hx : x = 3 := by sorry -- Placeholder for the necessary proof
  have hy : y = -4 := by sorry -- Placeholder for the necessary proof
  calc (x + y)^2023 = (3 + (-4))^2023 : by rw [hx, hy]
                ... = (-1)^2023      : by ring
                ... = -1            : by norm_num
  sorry

end problem_l811_811620


namespace expected_faces_rolled_six_times_l811_811446

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l811_811446


namespace distinct_prime_factors_of_90_l811_811090

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811090


namespace fixed_point_pq_l811_811546

theorem fixed_point_pq {A B C : Point} (X : Point) (hX : lies_on X (line B C)) (hC_between : between B C X)
  (P Q : Point) (hP_Q : P ≠ Q)
  (h_incircles : incircle ABX ∩ incircle ACX = {P, Q}) :
  ∃ (F : Point), ∀ (X : Point) (hX : lies_on X (line B C)) (hC_between : between B C X), 
  line_through P Q = line_through F :=
sorry

end fixed_point_pq_l811_811546


namespace simplify_expr_l811_811804

noncomputable theory

def expr := λ (x : ℝ), sqrt (1 + ( (x^6 - 1) / (2 * x^3) ) ^ 2)

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) : 
  expr x = (x^3 / 2) + (1 / (2 * x^3)) :=
sorry

end simplify_expr_l811_811804


namespace incorrect_expression_l811_811403

theorem incorrect_expression :
  ¬((|(-5 : ℤ)|)^2 = 5) :=
by
sorry

end incorrect_expression_l811_811403


namespace number_of_people_l811_811812

theorem number_of_people (total_cookies : ℕ) (cookies_per_person : ℝ) (h1 : total_cookies = 144) (h2 : cookies_per_person = 24.0) : total_cookies / cookies_per_person = 6 := 
by 
  -- Placeholder for actual proof.
  sorry

end number_of_people_l811_811812


namespace num_distinct_prime_factors_90_l811_811129

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811129


namespace find_sum_of_ab_l811_811048

def f (x : ℝ) : ℝ := 3^x + x - 5

theorem find_sum_of_ab (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : b - a = 1) (h4 : ∃ x, f x = 0 ∧ a ≤ x ∧ x ≤ b) : a + b = 3 :=
sorry

end find_sum_of_ab_l811_811048


namespace convert_38_to_binary_l811_811947

theorem convert_38_to_binary :
  let decimal_to_binary (n : ℕ) : list ℕ :=
    if n = 0 then []
    else (n % 2) :: decimal_to_binary (n / 2)
  decimal_to_binary 38.reverse = [1, 0, 0, 1, 1, 0] :=
by
  sorry

end convert_38_to_binary_l811_811947


namespace expected_number_of_different_faces_l811_811450

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l811_811450


namespace log_xy_value_l811_811621

noncomputable def log_system (x y : ℝ) : Prop :=
  (Real.log (x * y^2) = 2) ∧ (Real.log (x^3 * y) = 3)

theorem log_xy_value (x y : ℝ) (h : log_system x y) : Real.log (x * y) = 7 / 5 :=
by
  cases h with h1 h2
  sorry

end log_xy_value_l811_811621


namespace number_of_distinct_prime_factors_90_l811_811146

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811146


namespace max_MN_to_AB_ratio_l811_811019

theorem max_MN_to_AB_ratio (p : ℝ) (hp : 0 < p) (A B M N : ℝ) 
  (hA : ∀ x, ∃ y, y^2 = 2 * p * x ∧ y = A)
  (hB : ∀ x, ∃ y, y^2 = 2 * p * x ∧ y = B)
  (hM : M = (A + B) / 2)
  (hN : N = ⟨0, sqrt (2 * p * M)⟩)
  (angle_cond : angle (A - F) (B - F) = π / 3) :
  ∃ r1 r2 : ℝ, 
  r1 = dist A F ∧ r2 = dist B F ∧ |MN| = (r1 + r2) / 2 ∧ 
  (|AB|^2 = r1^2 + r2^2 - 2 * r1 * r2 * cos(π / 3)) ∧
  (forall max_MN_to_AB_ratio, max_MN_to_AB_ratio = 1 := sorry

end max_MN_to_AB_ratio_l811_811019


namespace Anton_thought_of_729_l811_811916

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l811_811916


namespace cyclic_inequality_l811_811565

theorem cyclic_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := 
by
  sorry

end cyclic_inequality_l811_811565


namespace triangle_properties_correct_l811_811836

noncomputable def triangle_properties :
  ∀ (r_ω r_Ω : ℝ) (OM_radius area_ratio : ℝ)
  (P K M C F T O : Type) [Incircle ω P K M C F T O r_ω]
  [Circumcircle Ω P K M r_Ω T]
  [OM OM_radius]
  [AreasRatio area_ratio P K M C F T],
  OM_radius = 5 * Real.sqrt 13 ∧
  MA_length = (20 * Real.sqrt 13) / 3 ∧
  Area_CFM = 204 :=
by sorry -- proof to be filled in later

# Variables and Definitions
constant P K M C F T O : Type
constant r_ω r_Ω : ℝ
constant OM_radius area_ratio : ℝ

axioms
(Incircle ω P K M C F T O r_ω : Type)
(Circumcircle Ω P K M r_Ω T : Type)
(OM_radius : OM)
(AreasRatio area_ratio P K M C F T : Type)

# Proof
theorem triangle_properties_correct :
  OM_radius = 5 * Real.sqrt 13 ∧
  MA_length = (20 * Real.sqrt 13) / 3 ∧
  Area_CFM = 204 :=
by sorry

end triangle_properties_correct_l811_811836


namespace prob_max_at_3_or_4_l811_811616

noncomputable def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
nat.choose n k * p^k * (1 - p)^(n - k)

def max_prob_value_k {n : ℕ} {p : ℝ} [fact (0 < p) ] [fact (p < 1)] : ℕ :=
if n = 0 then 0 else if p = 1 then n else
  (finset.range (n + 1)).max' (finset.nonempty_range_succ n) (λ k, binomial_pmf n p k)

theorem prob_max_at_3_or_4 :
  max_prob_value_k 15 (1/4) = 3 ∨ max_prob_value_k 15 (1/4) = 4 :=
sorry

end prob_max_at_3_or_4_l811_811616


namespace log_base_frac_l811_811973

theorem log_base_frac (y : ℝ) : (log (1/4) 16) = -2 := by
  sorry

end log_base_frac_l811_811973


namespace a_2n_is_perfect_square_l811_811567

-- Define a_n based on the conditions.
def a_n : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 1
| 3     := 2
| 4     := 4
| (n+5) := a_n (n+4) + a_n (n+2) + a_n n

-- Representation of a_n for a specific n being a perfect square.
theorem a_2n_is_perfect_square (n : ℕ) : ∃ m : ℕ, a_n (2 * n) = m * m :=
sorry

end a_2n_is_perfect_square_l811_811567


namespace closest_perfect_square_l811_811400

theorem closest_perfect_square (n : ℕ) (h : n = 550) : ∃ k : ℕ, k^2 = 529 ∧ 
  (∀ m : ℕ, m^2 ≤ n → (n - m^2 ≤ n - k^2) ∧ (m^2 - n ≤ k^2 - n)) :=
begin
  use 23,
  split,
  { norm_num },
  { intros m hm,
    split,
    { rw h at hm,
      cases lt_or_eq_of_le hm,
      { norm_num,
        exact abs_nonneg (550 - k^2) },
      { norm_num,
        sorry } },
    { norm_num,
      sorry } }
end

end closest_perfect_square_l811_811400


namespace sum_greater_than_16_l811_811210

theorem sum_greater_than_16 {n : ℕ} (h : n ≥ 2) 
  (initial_numbers: Finset ℕ)
  (H_initial_numbers: initial_numbers = Finset.range (2 * (n + 1)) \ Finset.range 2)
  (invariant_sum_of_reciprocals : 
    ∀ S : Finset ℕ,
      (∀ a b c ∈ S, S = insert (S.erase a).erase b (S.erase c) ∪ { (a * b * c) / (a * b + b * c + c * a)})) :
  ∀ (a b : ℕ) (h_a_in : a ∈ initial_numbers) (h_b_in : b ∈ initial_numbers)
  (h_size : initial_numbers.card = 2),
  a + b > 16 :=
by 
  sorry

end sum_greater_than_16_l811_811210


namespace anton_thought_number_l811_811872

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l811_811872


namespace arithmetic_sequence_proof_arithmetic_sequence_formula_exists_m_M_l811_811050

theorem arithmetic_sequence_proof :
  (∃ (a : ℕ → ℕ) (d : ℕ),
    (∀ n, a (n + 1) = a n + d) ∧
    (∀ (n : ℕ), f (n + 1) = n^2) ∧
    (∀ (n : ℕ), (∃ g : ℚ → ℚ, g (1/2) < 14/9 ∧ 1/2 ≤ g (1/2)))) :=
  sorry

def f (x : ℚ) : ℕ → ℚ :=
  λ n, (finset.range (n + 1)).sum (λ i, (nat.rec_on i 0 (λ _ ih, ih + i)))

theorem arithmetic_sequence_formula (a : ℕ → ℕ) (h : ∀ n, a n = 2 * n - 1) :
  ∀ n, a n = 2 * n - 1 :=
  sorry

def g (x : ℚ) : ℕ → ℚ :=
  λ n, 1/2 * (f n - f (-n))
  
theorem exists_m_M (f : ℚ → ℕ → ℚ) (g : ℚ → ℕ → ℚ) :
  (∃ (m M : ℕ), ∀ n, m < g 1/2 n ∧ g 1/2 n < M) ∧
  (inf (range (λ n, g n (1/2))) = 7/9) ∧
  (sup (range (λ n, g n (1/2))) = 1) :=
  sorry

end arithmetic_sequence_proof_arithmetic_sequence_formula_exists_m_M_l811_811050


namespace dot_product_constant_l811_811575

-- Define the variables and constants
variables {a b c : ℝ} (x y k : ℝ)
noncomputable def focal_distance (a b : ℝ) := 2 * real.sqrt (a^2 - b^2) 

-- Define the point on the ellipse
def point_on_ellipse (x y a b : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)

-- Define the ellipse
def ellipse := ∃ (a b : ℝ), a > b ∧ b > 0 ∧ focal_distance a b = 2 * real.sqrt 3 ∧ 
  point_on_ellipse (real.sqrt 2) (-real.sqrt 2 / 2) a b

-- Define the curve E 
def curve_E (x y : ℝ) := x^2 + y^2 = 1

-- Define the points A and B on curve E, when line l passes through origin
def points_on_curve_E := ∀ (k : ℝ), ∃ (A B : ℝ × ℝ), 
  A = (1 / real.sqrt (k^2 + 1), k / real.sqrt (k^2 + 1)) ∧
  B = (-1 / real.sqrt (k^2 + 1), -k / real.sqrt (k^2 + 1))

-- Define the point D
def point_D := (-2, 0 : ℝ × ℝ)

-- Define the vectors DA and DB
def vector_DA (D A : ℝ × ℝ) := (A.1 - D.1, A.2 - D.2)
def vector_DB (D B : ℝ × ℝ) := (B.1 - D.1, B.2 - D.2)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

-- Main statement to prove
theorem dot_product_constant : 
  ellipse → curve_E x y → points_on_curve_E → (dot_product (vector_DA point_D A) (vector_DB point_D B)) = 3 :=
begin
  sorry
end

end dot_product_constant_l811_811575


namespace calc_log_sum_l811_811780

theorem calc_log_sum :
  10 * 61 * (∑ i in Finset.range (99) + 2, 1 / Real.log (Nat.factorial 100) / Real.log i) = 610 :=
by
  sorry

end calc_log_sum_l811_811780


namespace part_a_part_b_part_c_l811_811821

section PartA
  variables {n : ℕ} {R : ℝ}
  noncomputable def sum_of_squares_on_circle (M : Point) (vertices : Fin n → Point) 
  (circum_radius : ℝ) (circum_circle : M ∈ Circle circum_radius) : ℝ :=
  ∑ k : Fin n, (dist M (vertices k)) ^ 2

  theorem part_a (M : Point) (vertices : Fin n → Point) (circum_radius : ℝ) 
  (circum_circle : M ∈ Circle circum_radius) : sum_of_squares_on_circle M vertices circum_radius = 2 * n * circum_radius ^ 2 :=
  sorry
end PartA

section PartB
  variables {n : ℕ} {R l : ℝ}
  noncomputable def sum_of_squares_in_plane (M : Point) (vertices : Fin n → Point) 
  (o : Point) (circum_radius : ℝ) (dist_center : l = dist M o) : ℝ :=
  ∑ k : Fin n, (dist M (vertices k)) ^ 2

  theorem part_b (M : Point) (vertices : Fin n → Point) (o : Point) (circum_radius l : ℝ) 
  (dist_center : l = dist M o) : sum_of_squares_in_plane M vertices o circum_radius dist_center = n * (circum_radius ^ 2 + l ^ 2) :=
  sorry
end PartB

section PartC
  variables {n : ℕ} {R l : ℝ}
  noncomputable def sum_of_squares_arbitrary_point (M : Point) (vertices : Fin n → Point) 
  (o : Point) (circum_radius : ℝ) (dist_center : l = dist M o) : ℝ :=
  ∑ k : Fin n, (dist M (vertices k)) ^ 2

  theorem part_c (M : Point) (vertices : Fin n → Point) (o : Point) (circum_radius l : ℝ) 
  (dist_center : l = dist M o) : sum_of_squares_arbitrary_point M vertices o circum_radius dist_center = n * (circum_radius ^ 2 + l ^ 2) :=
  sorry
end PartC

end part_a_part_b_part_c_l811_811821


namespace annual_pension_l811_811845

theorem annual_pension (c d r s x k : ℝ) (hc : c ≠ 0) (hd : d ≠ c)
  (h1 : k * (x + c) ^ (3 / 2) = k * x ^ (3 / 2) + r)
  (h2 : k * (x + d) ^ (3 / 2) = k * x ^ (3 / 2) + s) :
  k * x ^ (3 / 2) = 4 * r^2 / (9 * c^2) :=
by
  sorry

end annual_pension_l811_811845


namespace rational_solutions_are_integers_l811_811036

-- Given two integers a and b, and two equations with rational solutions
variables (a b : ℤ)

-- The first equation is y - 2x = a
def eq1 (y x : ℚ) : Prop := y - 2 * x = a

-- The second equation is y^2 - xy + x^2 = b
def eq2 (y x : ℚ) : Prop := y^2 - x * y + x^2 = b

-- We want to prove that if y and x are rational solutions, they must be integers
theorem rational_solutions_are_integers (y x : ℚ) (h1 : eq1 a y x) (h2 : eq2 b y x) : 
    ∃ (y_int x_int : ℤ), y = y_int ∧ x = x_int :=
sorry

end rational_solutions_are_integers_l811_811036


namespace tennis_tournament_problem_l811_811635

theorem tennis_tournament_problem 
  (n : ℕ)
  (h1 : ∃ women ∈ (finset.range n), women)
  (h2 : ∃ men ∈ (finset.Icc (2 * n + 1) (finset.range (2 * n + 1))), men)
  (h3 : ∀ woman ∈ (finset.range n), ∀ players ∈ (finset.range (3 * n + 1)), woman ≠ players → match woman players = 1)
  (h4 : each woman plays exactly 2 extra matches with any two men)
  (h5 : wins_ratio : (total_matches_won_by_women / total_matches_won_by_men) = 3 / 2)
  : n = 2 := sorry

end tennis_tournament_problem_l811_811635


namespace distinct_prime_factors_90_l811_811164

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811164


namespace gender_association_with_facial_recognition_female_facial_recognition_distribution_and_expectation_l811_811506
open Probability

def contingency_table := {
  total_customers : ℕ := 100,
  facial_recognition : ℕ := 70,
  non_facial_recognition : ℕ := 30,
  male_facial : ℕ := 45,
  female_facial : ℕ := 25,
  male_non_facial : ℕ := 10,
  female_non_facial : ℕ := 20,
}

theorem gender_association_with_facial_recognition (α : ℝ) (chi_squared : ℝ) :
  α = 0.01 →
  chi_squared = (100 * (45 * 20 - 25 * 10)^2 / (55 * 45 * 70 * 30)) →
  chi_squared > 6.635 →
  true := sorry

theorem female_facial_recognition_distribution_and_expectation :
  let prob_0 := (1/126 : ℝ),
      prob_1 := (10/63 : ℝ),
      prob_2 := (10/21 : ℝ),
      prob_3 := (20/63 : ℝ),
      prob_4 := (5/126 : ℝ),
      expectation := 0 * prob_0 + 1 * prob_1 + 2 * prob_2 + 3 * prob_3 + 4 * prob_4
  in 
  prob_0 = 1/126 ∧
  prob_1 = 10/63 ∧
  prob_2 = 10/21 ∧
  prob_3 = 20/63 ∧
  prob_4 = 5/126 ∧
  expectation = 20/9 :=
  sorry

end gender_association_with_facial_recognition_female_facial_recognition_distribution_and_expectation_l811_811506


namespace EQ_value_l811_811319

theorem EQ_value 
  (EF FG GH HE : ℝ)
  (hEF : EF = 137)
  (hFG : FG = 75)
  (hGH : GH = 28)
  (hHE : HE = 105)
  (parallel_EF_GH : EF ∥ GH)
  (tangent_circle : ∃ Q : point, Q ∈ EF ∧ circle Q touches FG ∧ circle Q touches HE)
  : ∃ EQ : ℝ, EQ = 15 :=
by
  -- Proof goes here
  sorry

end EQ_value_l811_811319


namespace sum_gcf_lcm_eq_28_l811_811373

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l811_811373


namespace log_one_fourth_sixteen_l811_811970

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 :=
by
  sorry

end log_one_fourth_sixteen_l811_811970


namespace bananas_to_oranges_l811_811264

-- Definitions related to the conditions in the problem
def bananas : Type := ℕ
def oranges : Type := ℕ

def bananaValueInOranges (b : bananas) : oranges := (2 * b) / 3

-- Given that (3/4) of 12 bananas are worth 6 oranges
def condition : Prop := bananaValueInOranges (3 * 12 / 4) = 6

-- Goal: Prove that (1/4) of 12 bananas are worth 2 oranges
theorem bananas_to_oranges (h : condition) : bananaValueInOranges (12 / 4) = 2 := 
sorry

end bananas_to_oranges_l811_811264


namespace collinear_P_F_E_l811_811205

-- Definitions for the problem setup
variables (A B C D P Q E F O : Type)
variables [InCircle A B C D O] -- Quadrilateral ABCD inscribed in circle O

variables [IntersectsExt AB DC P] -- Extensions of AB and DC intersect at point P
variables [IntersectsExt AD BC Q] -- Extensions of AD and BC intersect at point Q
variables [TangentsFrom Q O E] -- Tangent from Q to circle O touches at E
variables [TangentsFrom Q O F] -- Tangent from Q to circle O touches at F

-- Goal statement
theorem collinear_P_F_E : Collinear P F E := sorry

end collinear_P_F_E_l811_811205


namespace log_one_fourth_sixteen_l811_811972

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 :=
by
  sorry

end log_one_fourth_sixteen_l811_811972


namespace solution_correct_l811_811667

noncomputable def solution : List (ℝ × ℕ) :=
  [(0, 4), (Real.arcsin (3 / 8) / 2, 7)]

theorem solution_correct {n : ℕ} (x : ℝ) (h1 : 4 ≤ n ∧ n ≤ 7) :
  (sin x + cos x = sqrt n / 2 ∧ tan (x / 2) = (sqrt n - 2) / 3 ∧ 0 ≤ x ∧ x < π / 4) ↔
  (x, n) ∈ solution :=
  sorry

end solution_correct_l811_811667


namespace gcd_lcm_sum_8_12_l811_811350

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l811_811350


namespace max_f_on_interval_l811_811954

noncomputable def circledast (a b : ℝ) : ℝ :=
  if a >= b then a else b^2

noncomputable def f (x : ℝ) : ℝ :=
  (circledast 1 x) * x - circledast 2 x

theorem max_f_on_interval : ∃ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x = 6 :=
by 
  let a := -2
  let b := 2
  sorry

end max_f_on_interval_l811_811954


namespace muffin_to_banana_ratio_l811_811525

-- Definitions of costs
def elaine_cost (m b : ℝ) : ℝ := 5 * m + 4 * b
def derek_cost (m b : ℝ) : ℝ := 3 * m + 18 * b

-- The problem statement
theorem muffin_to_banana_ratio (m b : ℝ) (h : derek_cost m b = 3 * elaine_cost m b) : m / b = 2 :=
by
  sorry

end muffin_to_banana_ratio_l811_811525


namespace find_sides_ADC_l811_811201

variables {a : ℝ}

-- Conditions
def is_isosceles_right_triangle (A B C : Type) (BC BA : ℝ) : Prop :=
BC = a

def BD_equals_BC_extension (A B C D : Type) (BD BC AB : ℝ) : Prop :=
BD = BC

-- Math problem statement
theorem find_sides_ADC {A B C D : Type} (h1 : is_isosceles_right_triangle A B C a)
  (h2 : BD_equals_BC_extension A B C D a) :
  AD = a * (Real.sqrt 2 + 1) ∧ CD = a * Real.sqrt (2 + Real.sqrt 2) ∧ AC = a :=
by
  sorry

end find_sides_ADC_l811_811201


namespace max_souls_guarantee_l811_811504

theorem max_souls_guarantee :
  ∀ (n1 n2 : ℕ), n1 + n2 = 222 →
  ∃ (N : ℕ), 1 ≤ N ∧ N ≤ 222 ∧ 
  ∀ (moved_nuts : ℕ), moved_nuts = if N = n1 ∨ N = n2 ∨ N = n1 + n2 then 0 else (max n1 n2 - min n1 n2) →
  moved_nuts ≤ 37 :=
begin
  sorry
end

end max_souls_guarantee_l811_811504


namespace anton_thought_number_l811_811897

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l811_811897


namespace anton_thought_number_is_729_l811_811891

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l811_811891


namespace inequality_holds_l811_811618

noncomputable def f (x : ℝ) : ℝ := 4 * x - 1

theorem inequality_holds (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) :
  (∀ x : ℝ, |x - 2 * b| < b → (x + a)^2 + |(f x) - 3 * b| < a^2) ↔ (a ≤ 4 * b) :=
by
  sorry

end inequality_holds_l811_811618


namespace speech_competition_speaking_orders_l811_811846

theorem speech_competition_speaking_orders:
  let students : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8} in
  let A := 1 in
  let B := 2 in
  ∃ orders : Finset (List ℕ),
    (∀ order ∈ orders, 
      A ∈ order ∨ B ∈ order) ∧ 
    (∀ order ∈ orders, 
      ¬ (A ∈ order ∧ B ∈ order ∧ ∃ (i : ℕ), 0 < i ∧ order.get? i = some A ∧ order.get? (i - 1) = some B)) ∧
    orders.card = 1140 :=
sorry

end speech_competition_speaking_orders_l811_811846


namespace freddy_spent_10_dollars_l811_811007

theorem freddy_spent_10_dollars 
  (talk_time_dad : ℕ) (talk_time_brother : ℕ) 
  (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ)
  (conversion_cents_to_dollar : ℕ)
  (h1 : talk_time_dad = 45)
  (h2 : talk_time_brother = 31)
  (h3 : local_cost_per_minute = 5)
  (h4 : international_cost_per_minute = 25)
  (h5 : conversion_cents_to_dollar = 100):
  (local_cost_per_minute * talk_time_dad + international_cost_per_minute * talk_time_brother) / conversion_cents_to_dollar = 10 :=
by
  sorry

end freddy_spent_10_dollars_l811_811007


namespace Anton_thought_number_is_729_l811_811906

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l811_811906


namespace geometric_sequence_a3_half_l811_811586

noncomputable def geometric_sequence (q : ℝ) (a_1 : ℝ) : ℕ → ℝ
| 0       => a_1
| (n + 1) => geometric_sequence n * q

theorem geometric_sequence_a3_half
  (q : ℝ) (h_q_pos : q > 0) (a_1 : ℝ) (h_a1_eq2 : a_1 = 2)
  (h_condition : 4 * geometric_sequence q a_1 1 * geometric_sequence q a_1 7 = (geometric_sequence q a_1 3)^2) :
  geometric_sequence q a_1 2 = 1 / 2 :=
by
  sorry

end geometric_sequence_a3_half_l811_811586


namespace solve_for_a_l811_811641

variable (t θ a : ℝ)

def line_passing_through_vertex (a : ℝ) : Prop :=
  let l := (x = t ∧ y = t - a)
  let C := (x = 3 * Real.cos θ ∧ y = 2 * Real.sin θ)
  ∀ t θ, (∃ x y, l ∧ C ∧ x = 3 ∧ y = 0)

theorem solve_for_a : line_passing_through_vertex a → a = 3 := sorry

end solve_for_a_l811_811641


namespace max_m_value_l811_811580

theorem max_m_value (a b : ℝ) (m : ℝ):
  let f := λ x : ℝ, Real.log (x + 1) - 2
  let g := λ x : ℝ, Real.exp x + b * x^2 + a
  let y := λ x : ℝ, a * x + b - Real.log 2
  let f_tangent := ∀ (x : ℝ), x = -1/2 → f x = y x
  let g_ineq := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → m ≤ g x ∧ g x ≤ m^2 - 2
  (a = 2) → (b = -1) → 
  (f_tangent = True) → 
  (g_ineq = True) →
  m ≤ e + 1 :=
by
  sorry

end max_m_value_l811_811580


namespace distinct_prime_factors_of_90_l811_811092

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811092


namespace Anton_thought_of_729_l811_811919

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l811_811919


namespace sequence_properties_l811_811649

theorem sequence_properties :
  (∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → a n = 4 * n - 2) ∧
  (∀ n : ℕ, a 1 = 2 ∧ a 17 = 66) ∧
  (a 20 = 78) ∧
  (∃ n : ℕ, a n = 398) :=
by {
  sorry
}

end sequence_properties_l811_811649


namespace gilda_marbles_l811_811559

theorem gilda_marbles (x : ℝ) (hx : x > 0) : 
  let marbles_after_pedro := 0.70 * x in
  let marbles_after_ebony := 0.595 * x in
  let marbles_after_jimmy := 0.476 * x in
  let marbles_after_marco := 0.4284 * x in
  (marbles_after_marco / x) * 100 = 42.84 := 
by 
  sorry

end gilda_marbles_l811_811559


namespace exists_integer_root_for_quadratic_poly_l811_811555

theorem exists_integer_root_for_quadratic_poly : 
  ∃ (n : ℕ) (a : fin n → ℤ), 
    n = 6 ∧ 
    (∃ r : ℤ, 
      r^2 - 2 * (∑ i, a i)^2 * r + (∑ i, a i ^ 4 + 1) = 0) := 
begin
  use 6,
  -- We'll construct the specific a[i]s used in the solution
  let a : fin 6 → ℤ := λ i, if i.val < 4 then 1 else -1,
  existsi a,
  split,
  { refl, },
  { use 1, -- One of the roots as found in the solution
    sorry,  -- Replace with the necessary proof showing the polynomial has 1 as a root
  }
end

end exists_integer_root_for_quadratic_poly_l811_811555


namespace average_side_length_of_squares_l811_811272

theorem average_side_length_of_squares 
  (A1 : ℝ) (A2 : ℝ) (A3 : ℝ) 
  (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 225) :
  (real.sqrt A1 + real.sqrt A2 + real.sqrt A3) / 3 = 28 / 3 :=
by 
  sorry

end average_side_length_of_squares_l811_811272


namespace log_base_frac_l811_811975

theorem log_base_frac (y : ℝ) : (log (1/4) 16) = -2 := by
  sorry

end log_base_frac_l811_811975


namespace number_of_distinct_prime_factors_of_90_l811_811083

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811083


namespace polynomial_roots_l811_811518

theorem polynomial_roots : ∀ x : ℝ, (x^3 - 4*x^2 - x + 4) * (x - 3) * (x + 2) = 0 ↔ 
  (x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 3 ∨ x = 4) :=
by 
  sorry

end polynomial_roots_l811_811518


namespace sum_of_incircle_areas_l811_811236

variables {a b c : ℝ} (ABC : Triangle ℝ) (s K r : ℝ)
  (hs : s = (a + b + c) / 2)
  (hK : K = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (hr : r = K / s)

theorem sum_of_incircle_areas :
  let larger_circle_area := π * r^2
  let smaller_circle_area := π * (r / 2)^2
  larger_circle_area + 3 * smaller_circle_area = 7 * π * r^2 / 4 :=
sorry

end sum_of_incircle_areas_l811_811236


namespace largest_inscribed_triangle_area_l811_811505

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 8) :
  ∃ (A : ℝ), A = 64 ∧ A = (1 / 2) * (2 * r) * r :=
by {
  use 64,
  split,
  { refl },
  { rw [h, mul_assoc, ←mul_assoc 2 8 8, mul_comm 2 8, mul_assoc],
    calc
      (1 / 2) * (2 * 8) * 8 = (1 / 2) * 16 * 8 : by ring
      ... = 8 * 8 : by ring
      ... = 64 : by norm_num
  }
}

end largest_inscribed_triangle_area_l811_811505


namespace valid_parameterizations_l811_811943

-- Definitions and conditions
def line_eq (x y : ℝ) : Prop := y = 2 * x - 5

-- Parameterizations
def param_A (t : ℝ) : ℝ × ℝ := (3 + t * -2, 1 + t * -4)
def param_B (t : ℝ) : ℝ × ℝ := (5 + t * 3, 5 + t * 6)
def param_C (t : ℝ) : ℝ × ℝ := (0 + t * 1, -5 + t * 2)
def param_D (t : ℝ) : ℝ × ℝ := (-1 + t * 2, -7 + t * 5)
def param_E (t : ℝ) : ℝ × ℝ := (2 + t * (1/2), -1 + t * 1)

-- Lean theorem statement
theorem valid_parameterizations :
  (∀ t, line_eq (param_A t).fst (param_A t).snd) ∧
  (∀ t, line_eq (param_B t).fst (param_B t).snd) ∧
  (∀ t, line_eq (param_C t).fst (param_C t).snd) ∧
  ¬ (∀ t, line_eq (param_D t).fst (param_D t).snd) ∧
  (∀ t, line_eq (param_E t).fst (param_E t).snd) :=
by
  repeat {apply And.intro}
  sorry

end valid_parameterizations_l811_811943


namespace length_of_bridge_l811_811769

/-- Length of the bridge problem -/
theorem length_of_bridge (train_length : ℕ) (train_speed : ℕ) (cross_time : ℕ) (speed_conversion_factor : ℚ) (hour_to_second : ℚ) 
  (train_length_eq : train_length = 110) (train_speed_eq : train_speed = 45) (cross_time_eq : cross_time = 30)
  (speed_conversion_factor_eq : speed_conversion_factor = 1000) (hour_to_second_eq : hour_to_second = 3600) : 
  let speed_m_per_s := (train_speed * speed_conversion_factor) / hour_to_second in
  let total_distance := speed_m_per_s * cross_time in
  let bridge_length := total_distance - train_length in
  bridge_length = 265 :=
by
  sorry

end length_of_bridge_l811_811769


namespace books_borrowed_second_day_l811_811454

theorem books_borrowed_second_day (initial_books : ℕ) (first_day_people : ℕ) (books_per_person : ℕ) 
  (remaining_books_after_second_day : ℕ) : 
  initial_books = 100 ∧ first_day_people = 5 ∧ books_per_person = 2 ∧ remaining_books_after_second_day = 70 → 
  (initial_books - (first_day_people * books_per_person) - remaining_books_after_second_day = 20) :=
by
  intros
  simp_all
  sorry

end books_borrowed_second_day_l811_811454


namespace intersection_eq_l811_811604

def A : Set ℝ := { x | 2^x > 1 }
def B : Set ℝ := { x | abs x < 3 }

theorem intersection_eq : A ∩ B = { x | 0 < x ∧ x < 3 } :=
by 
  sorry

end intersection_eq_l811_811604


namespace yeast_operations_correct_l811_811316

def condition_1 := "Put an appropriate amount of dry yeast into a conical flask containing a certain concentration of glucose solution and culture under suitable conditions"
def condition_2 := "After standing for a while, use a pipette to extract the culture medium from the conical flask"
def condition_3 := "Drop a drop of culture medium in the center of the hemocytometer, and cover it with a cover slip"
def condition_4 := "Use filter paper to absorb the excess culture medium at the edge of the hemocytometer"
def condition_5 := "Place the counting chamber in the center of the stage, wait for the yeast to settle to the bottom of the counting chamber, and observe and count under the microscope"

def correct_operations := {condition_1, condition_4, condition_5}

theorem yeast_operations_correct : correct_operations = {condition_1, condition_4, condition_5} := by
  sorry

end yeast_operations_correct_l811_811316


namespace g_increasing_l811_811683

def g (x : ℝ) : ℝ := 1 - 2 / (2^x + 1)

theorem g_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂ :=
by
  assume x₁ x₂ h,
  sorry

end g_increasing_l811_811683


namespace irrational_sqrt3_among_options_l811_811859

theorem irrational_sqrt3_among_options :
  let a := 2
  let b := -1/2
  let c := 3.14
  let d := Real.sqrt 3
  IsIrrational d ∧ ¬IsIrrational a ∧ ¬IsIrrational b ∧ ¬IsIrrational c :=
by
  sorry

end irrational_sqrt3_among_options_l811_811859


namespace number_of_distinct_prime_factors_90_l811_811150

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811150


namespace race_completion_time_l811_811629

variable (t : ℕ)
variable (vA vB : ℕ)
variable (tB : ℕ := 45)
variable (d : ℕ := 100)
variable (diff : ℕ := 20)
variable h1 : vA * t = d
variable h2 : vB * t = d - diff
variable h3 : vB = d / tB

theorem race_completion_time (h : vB = d / tB): t = 36 :=
by sorry

end race_completion_time_l811_811629


namespace infinitely_many_n_gt_sqrt_two_l811_811686

/-- A sequence of positive integers indexed by natural numbers. -/
def a (n : ℕ) : ℕ := sorry

/-- Main theorem stating there are infinitely many n such that 1 + a_n > a_{n-1} * root n of 2. -/
theorem infinitely_many_n_gt_sqrt_two :
  ∀ (a : ℕ → ℕ), (∀ n, a n > 0) → ∃ᶠ n in at_top, 1 + a n > a (n - 1) * (2 : ℝ) ^ (1 / n : ℝ) :=
by {
  sorry
}

end infinitely_many_n_gt_sqrt_two_l811_811686


namespace largest_number_is_B_l811_811808

-- Define the numbers as constants
def A : ℝ := 0.989
def B : ℝ := 0.998
def C : ℝ := 0.981
def D : ℝ := 0.899
def E : ℝ := 0.9801

-- State the theorem that B is the largest number
theorem largest_number_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  -- By comparison
  sorry

end largest_number_is_B_l811_811808


namespace intersection_point_with_y_axis_l811_811751

theorem intersection_point_with_y_axis : 
  ∃ y, (0, y) = (0, 3) ∧ (y = 0 + 3) :=
by
  sorry

end intersection_point_with_y_axis_l811_811751


namespace plum_balances_pear_l811_811613

variable (A G S : ℕ)

-- Definitions as per the problem conditions
axiom condition1 : 3 * A + G = 10 * S
axiom condition2 : A + 6 * S = G

-- The goal is to prove the following statement
theorem plum_balances_pear : G = 7 * S :=
by
  -- Skipping the proof as only statement is needed
  sorry

end plum_balances_pear_l811_811613


namespace four_p_minus_three_is_perfect_square_l811_811669

theorem four_p_minus_three_is_perfect_square 
  {n p : ℕ} (hn : 1 < n) (hp : 1 < p) (hp_prime : Prime p) 
  (h1 : n ∣ (p - 1)) (h2 : p ∣ (n^3 - 1)) :
  ∃ k : ℕ, 4 * p - 3 = k ^ 2 := 
by 
  sorry

end four_p_minus_three_is_perfect_square_l811_811669


namespace uncle_taller_by_14_l811_811656

namespace height_problem

def uncle_height : ℝ := 72
def james_height_before_spurt : ℝ := (2 / 3) * uncle_height
def growth_spurt : ℝ := 10
def james_height_after_spurt : ℝ := james_height_before_spurt + growth_spurt

theorem uncle_taller_by_14 : uncle_height - james_height_after_spurt = 14 := by
  sorry

end height_problem

end uncle_taller_by_14_l811_811656


namespace cars_served_from_4pm_to_6pm_l811_811931

theorem cars_served_from_4pm_to_6pm : 
  let cars_per_15_min_peak := 12
  let cars_per_15_min_offpeak := 8 
  let blocks_in_an_hour := 4 
  let total_peak_hour := cars_per_15_min_peak * blocks_in_an_hour 
  let total_offpeak_hour := cars_per_15_min_offpeak * blocks_in_an_hour 
  total_peak_hour + total_offpeak_hour = 80 := 
by 
  sorry 

end cars_served_from_4pm_to_6pm_l811_811931


namespace sum_gcd_lcm_eight_twelve_l811_811378

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l811_811378


namespace sum_gcf_lcm_eq_28_l811_811372

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l811_811372


namespace find_circle_values_l811_811203

-- Define the conditions
variable (A B C D : ℕ)
variable (S : ℕ)
variable (h1 : 1 + B + 5 = S)
variable (h2 : 3 + 4 + D = S)
variable (h3 : 2 + A + 4 = S)
variable (perm : {A, B, C, D} = {6, 7, 8, 9})

-- The proposition we need to prove
theorem find_circle_values : (A = 6) ∧ (B = 8) ∧ (C = 7) ∧ (D = 9) :=
sorry

end find_circle_values_l811_811203


namespace partition_555_weights_l811_811410

theorem partition_555_weights :
  ∃ A B C : Finset ℕ, 
  (∀ x ∈ A, x ∈ Finset.range (555 + 1)) ∧ 
  (∀ y ∈ B, y ∈ Finset.range (555 + 1)) ∧ 
  (∀ z ∈ C, z ∈ Finset.range (555 + 1)) ∧ 
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ 
  A ∪ B ∪ C = Finset.range (555 + 1) ∧ 
  A.sum id = 51430 ∧ B.sum id = 51430 ∧ C.sum id = 51430 := sorry

end partition_555_weights_l811_811410


namespace three_digit_even_numbers_l811_811815

theorem three_digit_even_numbers (h1: true) (h2: true) (h3: true) (h4: true) (h5: true) : 
  (card { n : ℕ | 100 ≤ n ∧ n < 1000 ∧ (∃ a b c, n = a * 100 + b * 10 + c ∧ c % 2 = 0 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) }) = 360 :=
by {
  sorry
}

end three_digit_even_numbers_l811_811815


namespace uncle_taller_by_14_l811_811657

namespace height_problem

def uncle_height : ℝ := 72
def james_height_before_spurt : ℝ := (2 / 3) * uncle_height
def growth_spurt : ℝ := 10
def james_height_after_spurt : ℝ := james_height_before_spurt + growth_spurt

theorem uncle_taller_by_14 : uncle_height - james_height_after_spurt = 14 := by
  sorry

end height_problem

end uncle_taller_by_14_l811_811657


namespace sum_gcf_lcm_eq_28_l811_811374

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l811_811374


namespace last_operation_ends_at_eleven_am_l811_811196

-- Definitions based on conditions
def operation_duration : ℕ := 45 -- duration of each operation in minutes
def start_time : ℕ := 8 * 60 -- start time of the first operation in minutes since midnight
def interval : ℕ := 15 -- interval between operations in minutes
def total_operations : ℕ := 10 -- total number of operations

-- Compute the start time of the last operation (10th operation)
def start_time_last_operation : ℕ := start_time + interval * (total_operations - 1)

-- Compute the end time of the last operation
def end_time_last_operation : ℕ := start_time_last_operation + operation_duration

-- End time of the last operation expected to be 11:00 a.m. in minutes since midnight
def expected_end_time : ℕ := 11 * 60 

theorem last_operation_ends_at_eleven_am : 
  end_time_last_operation = expected_end_time := by
  sorry

end last_operation_ends_at_eleven_am_l811_811196


namespace sophie_walking_distance_l811_811736

-- Define the variables and constants mentioned in the problem statement.
variables (d : ℝ) (t_total : ℝ) (v_scoot v_walk : ℝ)
hypothesis (total_time : t_total = 1.8)
hypothesis (speed_scoot : v_scoot = 20)
hypothesis (speed_walk : v_walk = 4)

-- Define the scooting and walking distances as functions of the total distance.
def scooting_distance (d : ℝ) : ℝ := (2 / 3) * d
def walking_distance (d : ℝ) : ℝ := (1 / 3) * d

-- Define the time taken for scooting and walking.
def time_scooting (d : ℝ) (v_scoot : ℝ) : ℝ := (scooting_distance d) / v_scoot
def time_walking (d : ℝ) (v_walk : ℝ) : ℝ := (walking_distance d) / v_walk

noncomputable def total_travel_time (d : ℝ) (v_scoot : ℝ) (v_walk : ℝ) : ℝ := 
  (time_scooting d v_scoot) + (time_walking d v_walk)

-- Define the Lean theorem statement asserting the problem
theorem sophie_walking_distance : 
  ∃ d, (total_time := 1.8) → (speed_scoot := 20) → (speed_walk := 4) → 
  (total_travel_time d speed_scoot speed_walk = t_total) → 
  (walking_distance d).round = 5.1 :=
sorry

end sophie_walking_distance_l811_811736


namespace area_of_triangle_l811_811342

theorem area_of_triangle : 
  let line1 := (λ x : ℝ, 5)
  let line2 := (λ x : ℝ, 3 + x)
  let line3 := (λ x : ℝ, 1 - x)
  let p1 := (2, 5)
  let p2 := (-4, 5)
  let p3 := (-1, 2)
in 
  (abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p1.2) - (p2.1 * p1.2 + p3.1 * p2.2 + p1.1 * p3.2))) / 2 = 9 := sorry

end area_of_triangle_l811_811342


namespace largest_possible_value_of_n_l811_811928

theorem largest_possible_value_of_n :
  ∃ (n : ℕ), 
  ∃ (a b : ℕ → ℕ), 
  (∀ k, a k = 1 + (k-1) * (a 2 - 1)) ∧ 
  (∀ k, b k = 1 + (k-1) * (b 2 - 1)) ∧ 
  a 1 = b 1 = 1 ∧ 
  1 < a 2 ∧ 
  a 2 ≤ b 2 ∧ 
  a n * b n = 1540 ∧
  (∀ m, (a m * b m = 1540 → m ≤ n)) ∧
  n = 512 :=
sorry

end largest_possible_value_of_n_l811_811928


namespace schoolchildren_chocolate_l811_811258

theorem schoolchildren_chocolate (m d : ℕ) 
  (h1 : 7 * d + 2 * m > 36)
  (h2 : 8 * d + 4 * m < 48) :
  m = 1 ∧ d = 5 :=
by
  sorry

end schoolchildren_chocolate_l811_811258


namespace terminal_side_symmetric_l811_811183

noncomputable theory
open Real
open Int

def angle_symmetric_about_line (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + 2 * k * π

theorem terminal_side_symmetric (α : ℝ)
  (h1 : angle_symmetric_about_line α (π / 3))
  (h2 : α ∈ set.Ioo (-4 * π) (-2 * π)) :
  α = - (11 * π) / 3 ∨ α = - (5 * π) / 3 :=
begin
  sorry
end

end terminal_side_symmetric_l811_811183


namespace students_going_on_field_trip_l811_811069

-- Define conditions
def van_capacity : Nat := 7
def number_of_vans : Nat := 6
def number_of_adults : Nat := 9

-- Define the total capacity
def total_people_capacity : Nat := number_of_vans * van_capacity

-- Define the number of students
def number_of_students : Nat := total_people_capacity - number_of_adults

-- Prove the number of students is 33
theorem students_going_on_field_trip : number_of_students = 33 := by
  sorry

end students_going_on_field_trip_l811_811069


namespace binomial_max_coefficient_term_binomial_coefficients_sum_binomial_sum_odd_coefficients_l811_811982

noncomputable def binomial_term_max_coefficient (n k : ℕ) : ℕ := Nat.choose n k * 2^k

theorem binomial_max_coefficient_term :
  binomial_term_max_coefficient 8 4 = 1120 := by
  sorry

theorem binomial_coefficients_sum :
  (Finset.range (8 + 1)).sum (λ k, Nat.choose 8 k) = 256 := by
  sorry

noncomputable def binomial_coefficients_odd_powers_sum : ℤ :=
  ((1 : ℤ) + (-1 : ℤ))^8 + 1) / 2

theorem binomial_sum_odd_coefficients :
  binomial_coefficients_odd_powers_sum = 3281 := by
  sorry

end binomial_max_coefficient_term_binomial_coefficients_sum_binomial_sum_odd_coefficients_l811_811982


namespace num_distinct_prime_factors_90_l811_811130

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l811_811130


namespace intersection_point_of_lines_l811_811277

theorem intersection_point_of_lines :
  (∃ x y : ℝ, y = x ∧ y = -x + 2 ∧ (x = 1 ∧ y = 1)) :=
sorry

end intersection_point_of_lines_l811_811277


namespace quadrilateral_rhombus_l811_811461

variables {A B C D O : Type*}
variables [metric_space A B C D O] [nonempty_space O]

def circumscribed_around_circle (ABCD : quadrilateral A B C D) (O : point O) := 
  circle_is_inscribed (circle O) ABCD

def diagonals_intersect_at_center (ABCD : quadrilateral A B C D) (O : point O) :=
  intersection_of_diagonals ABCD O

theorem quadrilateral_rhombus
  (ABCD : quadrilateral A B C D)
  (O : point O)
  (circum_circle : circumscribed_around_circle ABCD O)
  (diag_intersect : diagonals_intersect_at_center ABCD O) :
  is_rhombus ABCD :=
sorry

end quadrilateral_rhombus_l811_811461


namespace incorrect_statement_B_l811_811810

-- Conditions from the problem
variable (Quad : Type) [quadrilateral Quad]
variable (Paral : Type) [parallelogram Paral]
variable (Rect : Type) [rectangle Rect]
variable (Rhom : Type) [rhombus Rhom]
variable (Quad_diag_bisect : Quad → Prop)  -- Quadrilateral with diagonals that bisect each other
variable (Quad_diag_equal : Quad → Prop)   -- Quadrilateral with equal diagonals
variable (Rect_diag_perp : Rect → Prop)    -- Rectangle with perpendicular diagonals
variable (Rhom_diag_equal : Rhom → Prop)   -- Rhombus with equal diagonals

-- Prove that statement B is incorrect
theorem incorrect_statement_B (h1 : ∀ Q : Quad, Quad_diag_bisect Q → Paral)
                               (h2 : ∀ Q : Quad, Quad_diag_equal Q → Rect)
                               (h3 : ∀ Q : Rect, Rect_diag_perp Q → ∃ S : Rect, S = square)
                               (h4 : ∀ Q : Rhom, Rhom_diag_equal Q → ∃ S : Rhom, S = square) :
  ¬ (∀ Q : Quad, Quad_diag_equal Q → Rect) :=
by
  sorry

end incorrect_statement_B_l811_811810


namespace tangent_at_1_eq_tangent_through_point_A_l811_811598

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + x + 2

theorem tangent_at_1_eq :
  ∃ (a b c : ℝ), (a, b, c) = (2, -1, 1) ∧ ∀ x : ℝ, f 1 + (fderiv ℝ f 1 x) * (x - 1) = a * x + b * f x + c :=
begin
  sorry,
end

theorem tangent_through_point_A :
  ∃ (a b : ℝ), (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∧ ∀ x : ℝ, 3 + ((3 * a^2 - 2 * a + 1) * (x - a)) = x * f x + b :=
begin
  sorry,
end

end tangent_at_1_eq_tangent_through_point_A_l811_811598


namespace average_side_length_of_squares_l811_811273

theorem average_side_length_of_squares 
  (A1 : ℝ) (A2 : ℝ) (A3 : ℝ) 
  (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 225) :
  (real.sqrt A1 + real.sqrt A2 + real.sqrt A3) / 3 = 28 / 3 :=
by 
  sorry

end average_side_length_of_squares_l811_811273


namespace find_probability_xi_equals_1_l811_811617

variables {n : ℕ} {p : ℚ} {ξ : ℕ → ℚ}

-- Define that ξ follows the binomial distribution B(n, p)
axiom binomial_distribution (ξ : ℕ → ℚ) (n : ℕ) (p : ℚ) : Prop

-- Given conditions
axiom exp_xi_is_6 : E (λ k, k * ξ k) = 6
axiom var_xi_is_3 : Var (λ k, k * ξ k) = 3

-- Theorem to prove
theorem find_probability_xi_equals_1 : binomial_distribution ξ n p → E (λ k, k * ξ k) = 6 → Var (λ k, k * ξ k) = 3 → P (ξ 1) = 3 / 2^10 :=
by
  sorry

end find_probability_xi_equals_1_l811_811617


namespace find_x_for_infinite_power_tower_equals_four_l811_811528

noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
if hx : x > 0 then sorry else 0  -- the infinite power tower definition is non-trivial and thus skipped

theorem find_x_for_infinite_power_tower_equals_four (x : ℝ) (hx : x > 0) :
  infinite_power_tower x = 4 → x = real.sqrt 2 :=
begin
  -- Proof skipped
  sorry
end

end find_x_for_infinite_power_tower_equals_four_l811_811528


namespace anton_thought_number_l811_811926

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l811_811926


namespace limit_n_g_l811_811661

noncomputable def g (n : ℕ) := Real.log (n + 1) - Real.log n

def A (n : ℕ) := (n + 1 / 2) * g n - 1
def B (n : ℕ) := 1 / (2 * n) + Real.log n - (n + 1) * g n + 1

theorem limit_n_g (h : ℕ → ℝ) (H : ∀ n, h n = g n) : 
  ∃ (L : ℝ), Filter.Tendsto (λ n, n * (1 - n * h n)) Filter.at_top (𝓝 L) := 
begin
  use 0,
  apply Filter.Tendsto.const_mul,
  { exact Filter.Tendsto.const_sub _ (Filter.Tendsto.mul (Filter.Tendsto_id') _),
    exact Filter.Tendsto.inv_at_top_zero'.comp (Filter.Tendsto.const_add n) },
  simp,
  sorry
end

end limit_n_g_l811_811661


namespace soccer_club_girls_l811_811464

theorem soccer_club_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : (1 / 3 : ℚ) * G + B = 18) : 
  G = 18 := 
  by sorry

end soccer_club_girls_l811_811464


namespace sum_gcf_lcm_l811_811356

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l811_811356


namespace complex_power_difference_l811_811619

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i)^10 - (1 - i)^10 = 64 * i := 
by sorry

end complex_power_difference_l811_811619


namespace sin_theta_correct_l811_811828

-- Define points and distances
noncomputable def point_A := (0, 0 : ℝ × ℝ)
noncomputable def point_B := (0, 4 : ℝ × ℝ)
noncomputable def point_C := (4, 4 : ℝ × ℝ)
noncomputable def point_D := (4, 0 : ℝ × ℝ)
noncomputable def point_M := (2, 4 : ℝ × ℝ)
noncomputable def point_N := (4, 2 : ℝ × ℝ)

-- Define distances
noncomputable def distance_AM := Real.sqrt ((2 - 0) ^ 2 + (4 - 0) ^ 2)
noncomputable def distance_AN := Real.sqrt ((4 - 0) ^ 2 + (2 - 0) ^ 2)
noncomputable def distance_MN := Real.sqrt ((2 - 4) ^ 2 + (4 - 2) ^ 2)

-- Calculate cos θ
noncomputable def cos_theta := 
  (distance_AM ^ 2 + distance_AN ^ 2 - distance_MN ^ 2) / (2 * distance_AM * distance_AN)

-- Calculate sin θ
noncomputable def sin_theta :=
  Real.sqrt (1 - cos_theta ^ 2)

theorem sin_theta_correct : sin_theta = 3 / 5 :=
by
  sorry

end sin_theta_correct_l811_811828


namespace gcd_lcm_sum_8_12_l811_811346

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l811_811346


namespace students_total_l811_811493

theorem students_total (position_eunjung : ℕ) (following_students : ℕ) (h1 : position_eunjung = 6) (h2 : following_students = 7) : 
  position_eunjung + following_students = 13 :=
by
  sorry

end students_total_l811_811493


namespace total_space_compacted_l811_811247

-- Definitions according to the conditions
def num_cans : ℕ := 60
def space_per_can_before : ℝ := 30
def compaction_rate : ℝ := 0.20

-- Theorem statement
theorem total_space_compacted : num_cans * (space_per_can_before * compaction_rate) = 360 := by
  sorry

end total_space_compacted_l811_811247


namespace jaguars_total_games_l811_811494

-- Defining constants for initial conditions
def initial_win_rate : ℚ := 0.55
def additional_wins : ℕ := 8
def additional_losses : ℕ := 2
def final_win_rate : ℚ := 0.6

-- Defining the main problem statement
theorem jaguars_total_games : 
  ∃ y x : ℕ, (x = initial_win_rate * y) ∧ (x + additional_wins = final_win_rate * (y + (additional_wins + additional_losses))) ∧ (y + (additional_wins + additional_losses) = 50) :=
sorry

end jaguars_total_games_l811_811494


namespace interval_of_increase_for_f_l811_811993

-- Define the function g(x)
def g (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Define the condition for the function g(x)
def condition (x : ℝ) : Prop := g x > 0

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log (g x) / Real.log(0.5)

-- Define the statement proving the interval of increase of the function f
theorem interval_of_increase_for_f :
  ∀ x : ℝ, condition x → f' x > 0 ↔ x ∈ Ioo (-1) 1 :=
sorry

end interval_of_increase_for_f_l811_811993


namespace sum_of_GCF_and_LCM_l811_811363

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l811_811363


namespace parallelogram_side_length_l811_811843

theorem parallelogram_side_length (s : ℝ) (h : s > 0)
  (adj_sides : ℝ) (angle_deg : ℝ) (area : ℝ) :
  adj_sides = 3 * s →
  angle_deg = 30 →
  area = 27 * Real.sqrt 3 →
  s = 3 :=
by
  intros adj_sides_eq angle_eq area_eq
  have adj_sides_eq : 3 * s = 3 * s, by exact adj_sides_eq
  have area_eq : 3 * s^2 * Real.sqrt 3 = 27 * Real.sqrt 3, by rw [←area_eq]
  sorry

end parallelogram_side_length_l811_811843


namespace expected_number_of_different_faces_l811_811448

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l811_811448


namespace probability_multiple_of_3_l811_811819

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

def total_tickets : ℕ := 20
def multiples_of_3_count : ℕ := (Finset.range total_tickets).filter is_multiple_of_3).card

theorem probability_multiple_of_3 : 
  (multiples_of_3_count.toFloat / total_tickets.toFloat) = 0.3 := by
  sorry

end probability_multiple_of_3_l811_811819


namespace Anton_thought_of_729_l811_811914

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l811_811914


namespace short_stack_pancakes_l811_811930

theorem short_stack_pancakes (x : ℕ) :
  (∃ x, 9 * x + 6 * 5 = 57) → x = 3 :=
by
  intro h,
  obtain ⟨y, hy⟩ := h,
  have h_eq : 9 * y + 30 = 57 := hy,
  have h_solve : 9 * y = 27 := by linarith,
  have y_eq : y = 3 := by linarith,
  rw y_eq,
  exact rfl

end short_stack_pancakes_l811_811930


namespace area_of_transformed_triangle_l811_811740

-- Definitions
variable {F : Type*}
variables {f : F → F} {x1 x2 x3 : F}

-- Given conditions and proof statement
theorem area_of_transformed_triangle (domain : set F) 
  (h_domain : domain = {x1, x2, x3}) 
  (triangle_area : ∀ (x1 x2 x3 : F), 
    (area_of_triangle (f x1) (f x2) (f x3)) = 32) :
  let g (x : F) := 2 * f (2 * x) in
  area_of_triangle (g (x1/2)) (g (x2/2)) (g (x3/2)) = 32 := 
by
  sorry

end area_of_transformed_triangle_l811_811740


namespace max_teams_participation_l811_811188

theorem max_teams_participation (n : ℕ) (H : 9 * n * (n - 1) / 2 ≤ 200) : n ≤ 7 := by
  -- Proof to be filled in
  sorry

end max_teams_participation_l811_811188


namespace greatest_difference_in_baskets_l811_811307

theorem greatest_difference_in_baskets :
  let A_red := 4
  let A_yellow := 2
  let B_green := 6
  let B_yellow := 1
  let C_white := 3
  let C_yellow := 9
  max (abs (A_red - A_yellow)) (max (abs (B_green - B_yellow)) (abs (C_white - C_yellow))) = 6 :=
by
  sorry

end greatest_difference_in_baskets_l811_811307


namespace sum_of_gcd_and_lcm_is_28_l811_811384

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l811_811384


namespace max_consecutive_sum_l811_811822

noncomputable def sequence : List ℕ := [1, 3, 9, 27, 81, 243, 729]

theorem max_consecutive_sum : ∀ (n : ℕ), n ≤ 1093 → 
  ∃ S : Set ℤ, 
    (∀ (x ∈ S), x ∈ (sequence.map (λ x, {-(↑x : ℤ), (↑x : ℤ)})).to_finset) ∧ 
    S.sum ∈ (Finset.Icc 1 1093 : Finset ℤ) :=
begin
  sorry
end

end max_consecutive_sum_l811_811822


namespace anton_thought_number_is_729_l811_811894

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l811_811894


namespace distinct_prime_factors_90_l811_811162

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l811_811162


namespace problem_p2_l811_811818

open Nat

theorem problem_p2 :
  ∃ (count : ℕ), count = 1000 ∧
  count = (card {n ∈ fin 1000001 | (sqrt n - floor (sqrt n : ℝ)) < 1 / 2013}) :=
sorry

end problem_p2_l811_811818


namespace gcd_lcm_sum_8_12_l811_811398

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l811_811398


namespace number_of_distinct_prime_factors_of_90_l811_811080

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811080


namespace fraction_not_read_l811_811424

theorem fraction_not_read (r : ℚ) (h : r > 1/2) : (1 - r) = 2/5 → false :=
by
  intro h_not_read
  have h_ge : 1 - r ≥ 0 := sub_nonneg.mpr h
  linarith

end fraction_not_read_l811_811424


namespace P_inter_M_l811_811831

def set_P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def set_M : Set ℝ := {x | x^2 ≤ 9}

theorem P_inter_M :
  set_P ∩ set_M = {x | 0 ≤ x ∧ x < 3} := sorry

end P_inter_M_l811_811831


namespace percentage_boy_scouts_l811_811452

theorem percentage_boy_scouts (S B G : ℝ) (h1 : B + G = S)
  (h2 : 0.60 * S = 0.50 * B + 0.6818 * G) : (B / S) * 100 = 45 := by
  sorry

end percentage_boy_scouts_l811_811452


namespace anton_thought_number_l811_811901

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l811_811901


namespace distinct_prime_factors_90_l811_811141

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l811_811141


namespace g_value_l811_811763

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value (h : ∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x) :
  g 3 = -(27 + 3 * (3:ℝ)^(1/3)) / 8 :=
sorry

end g_value_l811_811763


namespace distinct_prime_factors_of_90_l811_811094

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811094


namespace total_eggs_sold_l811_811609

def initial_trays : Nat := 10
def dropped_trays : Nat := 2
def added_trays : Nat := 7
def eggs_per_tray : Nat := 36

theorem total_eggs_sold : initial_trays - dropped_trays + added_trays * eggs_per_tray = 540 := by
  sorry

end total_eggs_sold_l811_811609


namespace steiner_ellipse_equation_l811_811537

theorem steiner_ellipse_equation
  (α β γ : ℝ) 
  (h : α + β + γ = 1) :
  β * γ + α * γ + α * β = 0 := 
sorry

end steiner_ellipse_equation_l811_811537


namespace solve_for_y_l811_811734

theorem solve_for_y : (∃ y : ℤ, 3^(y + 3) = (3^4)^y) ↔ y = 1 :=
by
  sorry

end solve_for_y_l811_811734


namespace log_base_frac_l811_811974

theorem log_base_frac (y : ℝ) : (log (1/4) 16) = -2 := by
  sorry

end log_base_frac_l811_811974


namespace misha_dollars_l811_811701

theorem misha_dollars (e t i : ℕ) (h1 : e = 13) (h2 : t = 47) : i = t - e → i = 34 :=
by
  intros h3
  rw [h1, h2, h3]
  exact rfl

end misha_dollars_l811_811701


namespace trigonometric_expression_l811_811040

-- Define the coordinates of P and derived values
variables (x y r : ℝ)
variables (P : P × ℝ × ℝ)

-- Point P is given
def point_P := P = (-4, 3)

-- Distance r from the origin to point P
def radius := r = Real.sqrt (x^2 + y^2) 

-- Given values of sin, cos terms derived from P
def sin_theta := sin θ = y / r
def cos_theta := cos θ = x / r
def tan_theta := tan θ = y / x

-- Prove the trigonometric equation
theorem trigonometric_expression :
  (cos (θ - (π / 2)) / sin ((π / 2) + θ)) * sin (θ + π) * cos (2π - θ) = - (3 / 5)^2 := 
by
  sorry

end trigonometric_expression_l811_811040


namespace correct_statements_l811_811409

-- Statement B
def statementB : Prop := 
∀ x : ℝ, x < 1/2 → (∃ y : ℝ, y = 2 * x + 1 / (2 * x - 1) ∧ y = -1)

-- Statement D
def statementD : Prop :=
∃ y : ℝ, (∀ x : ℝ, y = 1 / (Real.sin x) ^ 2 + 4 / (Real.cos x) ^ 2) ∧ y = 9

-- Combined proof problem
theorem correct_statements : statementB ∧ statementD :=
sorry

end correct_statements_l811_811409


namespace real_part_of_z_l811_811691

theorem real_part_of_z (z : ℂ) (h1 : abs (z - 1) = 2) (h2 : abs (z ^ 2 - 1) = 6) : z.re = 5 / 4 :=
sorry

end real_part_of_z_l811_811691


namespace gcd_lcm_sum_8_12_l811_811396

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l811_811396


namespace sum_of_geometric_sequence_l811_811589

theorem sum_of_geometric_sequence :
  ∃ (a_n : ℕ → ℝ), 
    (∀ n, a_n = a_1 * (q ^ n)) ∧
    (a_1 + a_n 4 = 9) ∧
    (a_n 2 * a_n 3 = 8) ∧
    let S_2018 := ∑ i in range 2018, a_n i in
    S_2018 = 2^2018 - 1 :=
begin
  sorry
end

end sum_of_geometric_sequence_l811_811589


namespace correct_product_l811_811202

-- Definitions for conditions
def reversed_product (a b : ℕ) : Prop :=
  let reversed_a := (a % 10) * 10 + (a / 10)
  reversed_a * b = 204

theorem correct_product (a b : ℕ) (h : reversed_product a b) : a * b = 357 := 
by
  sorry

end correct_product_l811_811202


namespace Anton_thought_of_729_l811_811915

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l811_811915


namespace triangle_transform_orthogonal_and_equal_sides_l811_811321

theorem triangle_transform_orthogonal_and_equal_sides
  (A B C P Q R : Point)
  (hABC : triangle A B C)
  (hBPC : triangle B P C)
  (hCQA : triangle C Q A)
  (hARB : triangle A R B)
  (hPBC : ∠ B P C = 45)
  (hCAQ : ∠ C Q A = 45)
  (hBCP : ∠ B C P = 30)
  (hQCA : ∠ Q C A = 30)
  (hABR : ∠ A B R = 15)
  (hBAR : ∠ B A R = 15) :
  ∠ Q R P = 90 ∧ dist Q R = dist R P := sorry

end triangle_transform_orthogonal_and_equal_sides_l811_811321


namespace number_of_women_attended_l811_811485

theorem number_of_women_attended
  (m : ℕ) (w : ℕ)
  (men_dance_women : m = 15)
  (women_dance_men : ∀ i : ℕ, i < 15 → i * 4 = 60)
  (women_condition : w * 3 = 60) :
  w = 20 :=
sorry

end number_of_women_attended_l811_811485


namespace simplify_and_evaluate_l811_811732

variable (a : ℝ)
variable (b : ℝ)

theorem simplify_and_evaluate (h : b = -1/3) : (a + b)^2 - a * (2 * b + a) = 1/9 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l811_811732


namespace sum_gcf_lcm_eq_28_l811_811371

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l811_811371


namespace sum_AB_AC_l811_811942

open Real

noncomputable def circle_center : point := ⟨0, 0⟩

constants (ω : set point) (O A B C : point)
constants (radius ω_center_distance tangent_distance BC_distance : ℝ)
axioms
  (h1 : ω = { p : point | (dist p circle_center) = radius })
  (h2 : dist O circle_center = 0)
  (h3 : radius = 6)
  (h4 : OA = 15)
  (h5 : is_tangent_to_circle A ω)
  (h6 : dist B C = 9)

theorem sum_AB_AC :
  dist O A = 15 ∧
  tangent_distance B O = 6 ∧
  tangent_distance C O = 6 ∧
  tangent_distance A B + 
  tangent_distance A C + 
  dist B C = 9
  → dist B A + dist C A = 6 * sqrt 21 + 9 :=
begin
  sorry
end

end sum_AB_AC_l811_811942


namespace anton_thought_number_l811_811898

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l811_811898


namespace dino_remaining_money_l811_811520

-- Definitions of the conditions
def hours_gig_1 : ℕ := 20
def hourly_rate_gig_1 : ℕ := 10

def hours_gig_2 : ℕ := 30
def hourly_rate_gig_2 : ℕ := 20

def hours_gig_3 : ℕ := 5
def hourly_rate_gig_3 : ℕ := 40

def expenses : ℕ := 500

-- The theorem to be proved: Dino's remaining money at the end of the month
theorem dino_remaining_money : 
  (hours_gig_1 * hourly_rate_gig_1 + hours_gig_2 * hourly_rate_gig_2 + hours_gig_3 * hourly_rate_gig_3) - expenses = 500 := by
  sorry

end dino_remaining_money_l811_811520


namespace Anton_thought_number_is_729_l811_811907

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l811_811907


namespace distinct_prime_factors_of_90_l811_811086

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811086


namespace number_of_distinct_prime_factors_of_90_l811_811109

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l811_811109


namespace expected_number_of_different_faces_l811_811435

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l811_811435


namespace gcd_lcm_sum_8_12_l811_811392

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l811_811392


namespace fraction_decomposition_l811_811953

theorem fraction_decomposition :
  ∃ (x y z a b c : ℕ),
    gcd a b = 1 ∧ gcd b c = 1 ∧ gcd a c = 1 ∧
    gcd 385 a = 1 ∧ gcd 385 b = 1 ∧ gcd 385 c = 1 ∧
    (↑x / a + ↑y / b + ↑z / c = 674 / 385) ∧
    (x + y + z = sum_digits a + sum_digits b + sum_digits c) :=
sorry

-- Define the function to calculate the sum of the digits of a number
def sum_digits : ℕ → ℕ
| 0 := 0
| n := n % 10 + sum_digits (n / 10)

end fraction_decomposition_l811_811953


namespace bananas_on_sunday_l811_811714

-- Define the number of bananas eaten each day
theorem bananas_on_sunday (a : ℕ) (bananas_eaten : ℕ → ℕ) :
  (bananas_eaten 0 = a) ∧
  (∀ n, bananas_eaten (n + 1) = bananas_eaten n + 4) ∧ 
  (∑ n in Finset.range 7, bananas_eaten n = 161) →
  bananas_eaten 6 = 35 :=
by sorry

end bananas_on_sunday_l811_811714


namespace distinct_prime_factors_of_90_l811_811098

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811098


namespace distinct_prime_factors_of_90_l811_811099

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l811_811099


namespace percentage_trucks_returned_l811_811712

theorem percentage_trucks_returned (total_trucks rented_trucks returned_trucks : ℕ)
  (h1 : total_trucks = 24)
  (h2 : rented_trucks = total_trucks)
  (h3 : returned_trucks ≥ 12)
  (h4 : returned_trucks ≤ total_trucks) :
  (returned_trucks / rented_trucks) * 100 = 50 :=
by sorry

end percentage_trucks_returned_l811_811712


namespace determine_k_value_l811_811957

theorem determine_k_value (x y z k : ℝ) 
  (h1 : 5 / (x + y) = k / (x - z))
  (h2 : k / (x - z) = 9 / (z + y)) :
  k = 14 :=
sorry

end determine_k_value_l811_811957


namespace angela_insects_l811_811481

theorem angela_insects (A J D : ℕ) (h1 : A = J / 2) (h2 : J = 5 * D) (h3 : D = 30) : A = 75 :=
by
  sorry

end angela_insects_l811_811481


namespace profit_percent_is_25_percent_l811_811414

def cost_price (SP : ℝ) : ℝ := 0.80 * SP
def profit (SP : ℝ) (CP : ℝ) : ℝ := SP - CP
def profit_percent (SP CP : ℝ) : ℝ := (profit SP CP / CP) * 100

theorem profit_percent_is_25_percent (SP : ℝ) (hCP : cost_price SP = 0.80 * SP) :
  profit_percent SP (cost_price SP) = 25 := by
  sorry

end profit_percent_is_25_percent_l811_811414


namespace vincent_earnings_l811_811333

theorem vincent_earnings 
  (price_fantasy_book : ℕ)
  (num_fantasy_books_per_day : ℕ)
  (num_lit_books_per_day : ℕ)
  (num_days : ℕ)
  (h1 : price_fantasy_book = 4)
  (h2 : num_fantasy_books_per_day = 5)
  (h3 : num_lit_books_per_day = 8)
  (h4 : num_days = 5) :
  let price_lit_book := price_fantasy_book / 2
      daily_earnings_fantasy := price_fantasy_book * num_fantasy_books_per_day
      daily_earnings_lit := price_lit_book * num_lit_books_per_day
      total_daily_earnings := daily_earnings_fantasy + daily_earnings_lit
      total_earnings := total_daily_earnings * num_days
  in total_earnings = 180 := 
  by 
  {
    sorry
  }

end vincent_earnings_l811_811333


namespace trajectory_of_P_ellipse_minimum_distance_of_Q_l811_811577

section

variables (P Q M N : EuclideanSpace ℝ (Fin 2))
variables (l : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))) 
variables (C : Set (EuclideanSpace ℝ (Fin 2)))

def M := EuclideanSpace.cons 4.0 (EuclideanSpace.cons 0.0 EuclideanSpace.nil)
def N := EuclideanSpace.cons 1.0 (EuclideanSpace.cons 0.0 EuclideanSpace.nil)

def P (x y : ℝ) : EuclideanSpace ℝ (Fin 2) :=
  EuclideanSpace.cons x (EuclideanSpace.cons y EuclideanSpace.nil)

def l : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2)) :=
  { carrier := {p | 1*p 0 + 2*p 1 - 12 = 0},
    direction := submodule.span ℝ {EuclideanSpace.cons 1.0 (EuclideanSpace.cons 2.0 EuclideanSpace.nil)} }

noncomputable def C : Set (EuclideanSpace ℝ (Fin 2)) :=
  {P | let (x, y) := (P 0, P 1) in (x^2 / 4 + y^2 / 3) = 1}

theorem trajectory_of_P_ellipse :
  ∀ P, (EuclideanSpace.inner (M - N) (M - P) = 6 * EuclideanSpace.norm (N - P)) → P ∈ C := 
sorry

theorem minimum_distance_of_Q :
  let Q := P 1 (3 / 2) in
  ∀ P ∈ C, dist Q l = sqrt ((8 * sqrt 5) / 5) :=
sorry

end

end trajectory_of_P_ellipse_minimum_distance_of_Q_l811_811577


namespace increase_productivity_RnD_l811_811935

theorem increase_productivity_RnD :
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  RnD_t / ΔAPL_t2 = 3260 :=
by
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  have h : RnD_t / ΔAPL_t2 = 3260 := sorry
  exact h

end increase_productivity_RnD_l811_811935


namespace minimum_value_of_f_l811_811539

-- Define the function
def f (x : ℝ) : ℝ := (1 + x^2) / (1 + x)

-- Define the condition
def is_nonneg (x : ℝ) : Prop := x ≥ 0

-- Define the minimum value found
def min_value : ℝ := -2 + 2 * Real.sqrt 2

-- State the theorem that proves the minimum value
theorem minimum_value_of_f : ∃ x : ℝ, is_nonneg x ∧ f x = min_value := 
by
  sorry

end minimum_value_of_f_l811_811539


namespace horner_evaluation_at_two_l811_811497

/-- Define the polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 * x^6 + 3 * x^5 + 5 * x^3 + 6 * x^2 + 7 * x + 8

/-- States that the value of f(2) using Horner's Rule equals 14. -/
theorem horner_evaluation_at_two : f 2 = 14 :=
sorry

end horner_evaluation_at_two_l811_811497


namespace floor_sqrt_72_l811_811527

theorem floor_sqrt_72 : ⌊Real.sqrt 72⌋ = 8 :=
by
  -- Proof required here
  sorry

end floor_sqrt_72_l811_811527


namespace anton_thought_number_is_729_l811_811888

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l811_811888


namespace minimum_area_l811_811197

-- Define the points A, B, and C lie on the hyperbola xy = 1
def lies_on_hyperbola (a b c : ℝ) : Prop :=
  a * (1/a) = 1 ∧ b * (1/b) = 1 ∧ c * (1/c) = 1

-- Define isosceles right triangle condition
def isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let (a1, a2) := A in
  let (b1, b2) := B in
  let (c1, c2) := C in
  (b1 - a1) * (c1 - a1) + (b2 - a2) * (c2 - a2) = 0 ∧
  ((b1 - a1)^2 + (b2 - a2)^2) = ((c1 - a1)^2 + (c2 - a2)^2)

-- Define the proof statement
theorem minimum_area (a b c : ℝ) 
  (h_hyperbola: lies_on_hyperbola a b c)
  (h_triangle: isosceles_right_triangle (a, 1/a) (b, 1/b) (c, 1/c)) :
  ∃ S, S = 3 * real.sqrt 3 := 
sorry

end minimum_area_l811_811197


namespace max_souls_l811_811501

theorem max_souls : 
  ∀ (distribute_nuts : (ℕ × ℕ)),
  distribute_nuts.1 + distribute_nuts.2 = 222 →
  ∀ (N : ℕ), 1 ≤ N ∧ N ≤ 222 →
  ∃ (move_nuts : ℕ), move_nuts ≤ 37 ∧
  (∃(box1 box2 : ℕ), box1 + box2 = N ∨ (∃(third_box : ℕ), (box1 + box2 + third_box) = N)) :=
begin
  sorry
end

end max_souls_l811_811501


namespace goldfish_ratio_l811_811612

theorem goldfish_ratio (initial_betta : ℕ) (initial_goldfish : ℕ) (betta_ratio : ℚ)
  (left_fish_after_gift : ℕ) (num_betta_after_bexley : ℕ) (num_fish_before_gift: ℕ)
  (num_goldfish_after_bexley : ℕ) (goldfish_brought_by_bexley : ℕ) (ratio : ℕ × ℕ) :
  initial_betta = 10 →
  initial_goldfish = 15 →
  betta_ratio = 2/5 →
  left_fish_after_gift = 17 →
  num_betta_after_bexley = initial_betta + (betta_ratio.num * initial_betta / betta_ratio.denom : ℕ) →
  num_fish_before_gift = left_fish_after_gift * 2 →
  num_goldfish_after_bexley = num_fish_before_gift - num_betta_after_bexley →
  goldfish_brought_by_bexley = num_goldfish_after_bexley - initial_goldfish →
  ratio = (goldfish_brought_by_bexley, initial_goldfish) →
  ratio.snd = initial_goldfish →
  (ratio.fst / Nat.gcd ratio.fst ratio.snd : ℕ) = 1 →
  (ratio.snd / Nat.gcd ratio.fst ratio.snd : ℕ) = 3 :=
by
{ intros,
  sorry
}

end goldfish_ratio_l811_811612


namespace find_f0_f1_l811_811693

-- Define the function f and the conditions given
def f : ℤ → ℤ := sorry

axiom condition1 (x : ℤ) : f(x + 5) - f(x) = 10 * x + 25
axiom condition2 (x : ℤ) : f(x^2 - 1) = (f(x) - x)^2 + x^2 - 3*x + 2

-- State the theorem to prove (f(0), f(1)) = (-1, 1)
theorem find_f0_f1 : (f 0, f 1) = (-1, 1) := by
  sorry

end find_f0_f1_l811_811693


namespace reflection_calculation_l811_811996

open Real

def point := ℝ × ℝ

def vec1 : point := (3, -2)
def vec2 : point := (-1, -4)

def dot_product (u v : point) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def projection (v u : point) : point :=
  let scalar := (dot_product u v) / (dot_product v v)
  (scalar * v.1, scalar * v.2)

def reflection (p reflect_axis : point) : point :=
  let proj := projection reflect_axis p
  (2 * proj.1 - p.1, 2 * proj.2 - p.2)

theorem reflection_calculation : 
  reflection vec1 vec2 = (-61/17, -6/17) :=
sorry

end reflection_calculation_l811_811996


namespace anton_thought_number_l811_811886

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l811_811886


namespace min_elements_in_set_l811_811012

open Set

theorem min_elements_in_set (a : ℝ) (ha : a > 1) : 
  let M := { x : ℝ | ((a-1)*x - a^2 + 4*a - 6) * (x + 3) < 0 } in
  let N := M ∩ (Set.univ : Set ℤ) in
  N = {-2, -1, 0, 1} := 
by
  sorry

end min_elements_in_set_l811_811012


namespace painting_time_l811_811174

variable (a d e : ℕ)

theorem painting_time (h : a * e * d = a * d * e) : (d * x = a^2 * e) := 
by
   sorry

end painting_time_l811_811174


namespace remainder_division_l811_811797

noncomputable def polynomial_remainder : ℕ → Polynomial ℚ → Polynomial ℚ → Polynomial ℚ
| 0, p, q := p
| (n + 1), p, q := 
  let r := p.evaluateAt (degree q)
  in polynomial_remainder n (p - r * q) q

theorem remainder_division :
  let f := 3 * X^4 + 14 * X^3 - 35 * X^2 - 80 * X + 56
  let g := X^2 + 8 * X - 6
  let r := polynomial_remainder (degree f).natAbs f g
  r = 364 * X - 322 
by
  sorry

end remainder_division_l811_811797


namespace g_at_3_l811_811765

-- Definition of the function and its property
def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, x ≠ 0 → g(x) - 3 * g(1/x) = 3^x

-- Goal: Prove that g(3) = 16.7
theorem g_at_3 : g 3 = 16.7 :=
by
  sorry

end g_at_3_l811_811765


namespace cistern_filling_time_l811_811782

theorem cistern_filling_time 
  (timeA timeB timeC timeD timeE : ℝ)
  (hA : timeA = 10) (hB : timeB = 12) (hC : timeC = 20)
  (hD : timeD = 15) (hE : timeE = 30) :
  let rateA := 1 / timeA,
      rateB := 1 / timeB,
      rateC := 1 / timeC,
      rateD := -1 / timeD,
      rateE := -1 / timeE,
      overallRate := rateA + rateB + rateC + rateD + rateE,
      fillTime := 1 / overallRate
  in fillTime = 7.5 :=
by
  sorry

end cistern_filling_time_l811_811782


namespace minimum_area_OPQ_l811_811718

theorem minimum_area_OPQ : 
  let P := (x, y) in
  let O := (0, 0) in
  let Q := (2, 2) in
  let circle_eq := (x + 3)^2 + (y - 1)^2 = 2 in
   -- Assuming P is on the circle
  P ∈ {p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 1)^2 = 2} →
  let area_△OPQ := 0.5 * abs ((O.1 * (P.2 - Q.2) + P.1 * (Q.2 - O.2) + Q.1 * (O.2 - P.2))) in
  ∃ (P : ℝ × ℝ), 
   (P ∈ {p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 1)^2 = 2} ∧
   (area_△OPQ = 2)) :=
begin
  sorry
end

end minimum_area_OPQ_l811_811718


namespace marbles_end_of_day_l811_811964

theorem marbles_end_of_day :
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54 :=
by
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  show initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54
  sorry

end marbles_end_of_day_l811_811964


namespace time_D_to_complete_job_alone_l811_811951

variable (D : ℝ)

-- Condition: Annie's work rate
def Annie_work_rate := 1 / 10

-- Condition: Dan's work rate
def Dan_work_rate := 1 / D

-- Condition: Work completed by Dan in 6 hours
def Dan_work_completed := 6 * Dan_work_rate

-- Condition: Remaining job after Dan's work
def Remaining_job := 1 - Dan_work_completed

-- Condition: Work completed by Annie in 6 hours
def Annie_work_completed := 6 * Annie_work_rate

-- The theorem to prove
theorem time_D_to_complete_job_alone :
  1 - 6 * (1 / D) = 6 * (1 / 10) → D = 15 :=
by
  intro h
  sorry

end time_D_to_complete_job_alone_l811_811951


namespace Nicky_Cristina_race_catchup_time_l811_811706

noncomputable def cristina_speed : ℕ := 5
noncomputable def nicky_speed : ℕ := 3
noncomputable def nicky_head_start : ℕ := 48

theorem Nicky_Cristina_race_catchup_time : ∃ t : ℕ, cristina_speed * t = nicky_head_start + nicky_speed * t ∧ t = 24 :=
by
  use 24
  simp [cristina_speed, nicky_speed, nicky_head_start]
  sorry

end Nicky_Cristina_race_catchup_time_l811_811706


namespace students_basketball_or_cricket_or_both_l811_811416

theorem students_basketball_or_cricket_or_both (B C BC : ℕ) (h_B : B = 10) (h_C : C = 8) (h_BC : BC = 4) :
  B + C - BC = 14 :=
by
  rw [h_B, h_C, h_BC]
  exact rfl

end students_basketball_or_cricket_or_both_l811_811416


namespace sum_gcd_lcm_eight_twelve_l811_811376

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l811_811376


namespace number_of_distinct_prime_factors_90_l811_811154

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811154


namespace problem_statement_l811_811672

-- Define points and vector operations
structure Point := (x : ℝ) (y : ℝ)

def parabola (p : Point) : Prop := p.x^2 = 4 * p.y
def distance (p1 p2 : Point) : ℝ := real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
def focus : Point := ⟨0, 1⟩

noncomputable def vector_sum_zero (a b c : Point) : Prop := 
  (⟨a.x - focus.x, a.y - focus.y⟩ + ⟨b.x - focus.x, b.y - focus.y⟩ + ⟨c.x - focus.x, c.y - focus.y⟩) = ⟨0, 0⟩

noncomputable def length_sum (a b c : Point) : ℝ := 
  distance focus a + distance focus b + distance focus c

theorem problem_statement (A B C : Point) 
  (hA : parabola A) (hB : parabola B) (hC : parabola C)
  (hVec : vector_sum_zero A B C) : 
  length_sum A B C = 6 := 
sorry

end problem_statement_l811_811672


namespace sum_of_GCF_and_LCM_l811_811366

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l811_811366


namespace max_magnitude_l811_811067

open Real

variables (θ : ℝ)

def vector_a : ℝ × ℝ := (cos θ, sin θ)
def vector_b : ℝ × ℝ := (0, -1)
def vector_diff : ℝ × ℝ := (vector_a θ).fst - (vector_b θ).fst, (vector_a θ).snd - (vector_b θ).snd
def magnitude := sqrt ((vector_diff θ).fst ^ 2 + (vector_diff θ).snd ^ 2)

theorem max_magnitude (hθ : 0 ≤ θ ∧ θ ≤ 2 * π) : magnitude θ ≤ 2 :=
sorry

end max_magnitude_l811_811067


namespace distinct_prime_factors_count_l811_811120

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l811_811120


namespace parallelogram_smaller_angle_proof_l811_811768

noncomputable def smaller_angle (x : ℝ) : Prop :=
  let larger_angle := x + 120
  let angle_sum := x + larger_angle + x + larger_angle = 360
  angle_sum

theorem parallelogram_smaller_angle_proof (x : ℝ) (h1 : smaller_angle x) : x = 30 := by
  sorry

end parallelogram_smaller_angle_proof_l811_811768


namespace intersection_point_with_y_axis_l811_811750

theorem intersection_point_with_y_axis : 
  ∃ y, (0, y) = (0, 3) ∧ (y = 0 + 3) :=
by
  sorry

end intersection_point_with_y_axis_l811_811750


namespace probability_of_drawing_desired_cups_l811_811192

theorem probability_of_drawing_desired_cups :
  let total_cups := 8
  let white_cups := 3
  let red_cups := 3
  let black_cups := 2
  let drawn_cups := 5
  ∃ (p : ℚ), p ≈ 0.64 :=
  sorry

end probability_of_drawing_desired_cups_l811_811192


namespace area_of_triangle_ABC_proof_l811_811587

noncomputable def area_of_triangle_oblique_transformed 
  (A'B' B'C' : ℝ)
  (angle_A'B'C' : ℝ)
  (transformation_ratio_area : ℝ) : ℝ :=
  let area_A'B'C' := (1/2) * A'B' * B'C' * real.sin angle_A'B'C'
  in transformation_ratio_area * area_A'B'C'

theorem area_of_triangle_ABC_proof :
  let A'B' := 4
  let B'C' := 3
  let angle_A'B'C' := real.pi / 3 -- 60 degrees in radians
  let transformation_ratio_area := 2 * real.sqrt 2
  area_of_triangle_oblique_transformed A'B' B'C' angle_A'B'C' transformation_ratio_area = 6 * real.sqrt 6 :=
  sorry

end area_of_triangle_ABC_proof_l811_811587


namespace missing_number_solution_l811_811731

theorem missing_number_solution (a b : ℝ) (h1 : a = (3/4) * 60) (h2 : b = (8/5) * 60) :
  ∃ (x : ℝ), a - b + x = 12 ∧ x = 63 :=
by
  use 63
  split
  sorry
  refl

end missing_number_solution_l811_811731


namespace rachel_makes_money_l811_811826

theorem rachel_makes_money (cost_per_bar total_bars remaining_bars : ℕ) (h_cost : cost_per_bar = 2) (h_total : total_bars = 13) (h_remaining : remaining_bars = 4) :
  cost_per_bar * (total_bars - remaining_bars) = 18 :=
by 
  sorry

end rachel_makes_money_l811_811826


namespace sum_f_eq_75_l811_811544

noncomputable def a_n (n : ℕ) : ℕ :=
  10^(n + 3) - 256

noncomputable def f (n : ℕ) : ℕ :=
  Nat.findGreatest (fun m => 2^m ∣ a_n n) (a_n n)

theorem sum_f_eq_75 : (List.range 10).sum (fun n => f (n + 1)) = 75 :=
  sorry

end sum_f_eq_75_l811_811544


namespace part1_simplify_and_period_part2_max_min_values_l811_811054

def f (x : ℝ) : ℝ := 2 * sin x * cos x - real.sqrt 3 * cos (2 * x) + 1

theorem part1_simplify_and_period :
  ∃ T > 0, f x = 2 * sin (2 * x - π / 3) + 1 ∧ ∀ x, f (x + T) = f x :=
sorry

theorem part2_max_min_values :
  ∃ max_val min_val, ∀ x ∈ set.Icc (π / 4) (π / 2), 
  (max_val = 3) ∧ (min_val = 2) ∧ min_val ≤ f x ∧ f x ≤ max_val :=
sorry

end part1_simplify_and_period_part2_max_min_values_l811_811054


namespace vincent_earnings_after_5_days_l811_811330

def fantasy_book_price : ℕ := 4
def daily_fantasy_books_sold : ℕ := 5
def literature_book_price : ℕ := fantasy_book_price / 2
def daily_literature_books_sold : ℕ := 8
def days : ℕ := 5

def daily_earnings : ℕ :=
  (fantasy_book_price * daily_fantasy_books_sold) +
  (literature_book_price * daily_literature_books_sold)

def total_earnings (d : ℕ) : ℕ :=
  daily_earnings * d

theorem vincent_earnings_after_5_days : total_earnings days = 180 := by
  sorry

end vincent_earnings_after_5_days_l811_811330


namespace gcd_lcm_sum_8_12_l811_811344

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l811_811344


namespace number_of_distinct_prime_factors_of_90_l811_811081

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l811_811081


namespace house_spirit_1000_enters_on_4th_floor_l811_811200

-- Define the conditions for the problem
def num_floors : ℕ := 7
def spirits_per_cycle : ℕ := 12
def total_spirits_needed : ℕ := 1000

-- Define the number of complete cycles the elevator makes
def complete_cycles : ℕ := total_spirits_needed / spirits_per_cycle
def spirits_after_complete_cycles : ℕ := complete_cycles * spirits_per_cycle
def remaining_spirits : ℕ := total_spirits_needed - spirits_after_complete_cycles

-- Statement to prove the target floor for the 1000th spirit
theorem house_spirit_1000_enters_on_4th_floor :
  remaining_spirits + 1 = 4 :=
by
  unfold num_floors spirits_per_cycle total_spirits_needed complete_cycles spirits_after_complete_cycles remaining_spirits
  -- Here we would include the proof steps, but they are not required in this task
  sorry

end house_spirit_1000_enters_on_4th_floor_l811_811200


namespace number_of_distinct_prime_factors_90_l811_811151

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l811_811151


namespace power_mean_inequality_l811_811014

variables {a b c : ℝ}
variables {n p q r : ℕ}

theorem power_mean_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 0 < n)
  (hpqr_nonneg : 0 ≤ p ∧ 0 ≤ q ∧ 0 ≤ r)
  (sum_pqr : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p :=
sorry

end power_mean_inequality_l811_811014


namespace log_mult_l811_811508

theorem log_mult : 
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by 
  sorry

end log_mult_l811_811508


namespace matt_needs_38_plates_l811_811698

def plates_needed (days_with_only_matt_and_son days_with_parents plates_per_day plates_per_person_with_parents : ℕ) : ℕ :=
  (days_with_only_matt_and_son * plates_per_day) + (days_with_parents * 4 * plates_per_person_with_parents)

theorem matt_needs_38_plates :
  plates_needed 3 4 2 2 = 38 :=
by
  sorry

end matt_needs_38_plates_l811_811698


namespace maya_daily_saving_l811_811700

def daily_saving (d : ℕ) : ℕ := d / 30

theorem maya_daily_saving :
  let total_savings := 3 * 10 in
  daily_saving total_savings = 1 :=
by 
  sorry

end maya_daily_saving_l811_811700


namespace smallest_positive_multiple_l811_811798

theorem smallest_positive_multiple (a : ℕ) (k : ℕ) (h : 17 * a ≡ 7 [MOD 101]) : 
  ∃ k, k = 17 * 42 := 
sorry

end smallest_positive_multiple_l811_811798


namespace minimum_odd_integers_l811_811781

theorem minimum_odd_integers (a b c d e f : ℤ)
  (h1 : a + b + c = 36)
  (h2 : a + b + c + d + e = 59)
  (h3 : a + b + c + d + e + f = 78) :
  ∃ n, (n ≥ 2) ∧ (∀m, m < 2 → ¬∀ i : ℕ, i < 6 → (([a, b, c, d, e, f].nth i).get_or_else 0 % 2 = 1)) :=
begin
  sorry
end

end minimum_odd_integers_l811_811781


namespace books_already_read_l811_811292

def total_books : ℕ := 20
def unread_books : ℕ := 5

theorem books_already_read : (total_books - unread_books = 15) :=
by
 -- Proof goes here
 sorry

end books_already_read_l811_811292


namespace vincent_earnings_l811_811336

-- Definitions based on the problem conditions
def fantasy_book_cost : ℕ := 4
def literature_book_cost : ℕ := fantasy_book_cost / 2
def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def duration : ℕ := 5

-- Calculation functions
def daily_earnings_from_fantasy_books : ℕ := fantasy_books_sold_per_day * fantasy_book_cost
def daily_earnings_from_literature_books : ℕ := literature_books_sold_per_day * literature_book_cost
def total_daily_earnings : ℕ := daily_earnings_from_fantasy_books + daily_earnings_from_literature_books
def total_earnings_after_five_days : ℕ := total_daily_earnings * duration

-- Statement to prove
theorem vincent_earnings : total_earnings_after_five_days = 180 := 
by
  calc total_daily_earnings * duration = 180 : sorry

end vincent_earnings_l811_811336
