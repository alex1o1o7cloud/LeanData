import Mathlib
import Mathlib.Algebra.Field.Power
import Mathlib.Algebra.Fraction
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Modulo
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Polynomial.GroupRing
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Trig
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Logarithm
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Modeq
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.GroupTheory.Representation.Basic
import Mathlib.GroupTheory.Subgroup.Center
import Mathlib.LinearAlgebra.Projection
import Mathlib.Probability.Basic
import Mathlib.Probability.Probability
import Mathlib.Tactic
import Probability.ConditionalProbability
import Real

namespace solve_system_of_equations_l205_205928

theorem solve_system_of_equations : 
  {p : ℚ × ℚ | let (x, y) := p in 2 * x^2 - 7 * x * y - 4 * y^2 + 9 * x - 18 * y + 10 = 0 ∧ x^2 + 2 * y^2 = 6}
  = {⟨2, 1⟩, ⟨-2, -1⟩, ⟨-22/9, -1/9⟩} :=
by
  sorry

end solve_system_of_equations_l205_205928


namespace problem_fractional_imaginary_l205_205275

theorem problem_fractional_imaginary (i : ℂ) (h1 : i^2 = -1) :
  let z := (1 - i) / Real.sqrt 2
  in z^44 = -1 :=
by
  -- Introduce variables and hypotheses
  let z := (1 - i) / Real.sqrt 2
  
  -- Sorry is used to indicate the proof is omitted
  sorry

end problem_fractional_imaginary_l205_205275


namespace units_digit_47_pow_47_l205_205608

theorem units_digit_47_pow_47 :
  let cycle := [7, 9, 3, 1] in
  cycle.nth (47 % 4) = some 3 :=
by
  let cycle := [7, 9, 3, 1]
  have h : 47 % 4 = 3 := by norm_num
  rw h
  simp
  exact trivial

end units_digit_47_pow_47_l205_205608


namespace min_m_plus_n_l205_205496

def m_and_n_min_sum : ℕ → ℕ → Prop := 
λ m n, (0 < m ∧ 0 < n ∧ 50 * m = n^3 ∧ (∀ m' n', 0 < m' → 0 < n' → 50 * m' = n'^3 → m' + n' ≥ m + n))

theorem min_m_plus_n (m n : ℕ) : m_and_n_min_sum m n → m + n = 30 := 
by
  sorry

end min_m_plus_n_l205_205496


namespace tangency_proof_l205_205888

variables {α : Type*} [EuclideanSpace α]
variables (A B C I D E F P X Y : α)
variables (ω_B ω_C : Circle α)
variable (B' C' : Point α)

variables (tangent_to_BC_ω_B : tangent_point_on_line_segment(ω_B, E, B, C))
variables (tangent_to_BC_ω_C : tangent_point_on_line_segment(ω_C, F, B, C))

def incircle_of_ABC (I_center : I : α)

-- Define the intersection points
def intersection_AD_P (P_center : P = line_intersection(A, D, ω_B, ω_C))
def intersection_BI_CP (X_intersection : X = line_intersection(B, I, C, P))
def intersection_CI_BP (Y_intersection : Y = line_intersection(C, I, B, P))

-- Tangency Proof problem
theorem tangency_proof :
  (tangent_point_on_circle intersection_BC_EX : intersection_points(EX, incircle_of_ABC)) ∧
  (tangent_point_on_circle intersection_FY_Y : intersection_points(FY, incircle_of_ABC)) :=
sorry

end tangency_proof_l205_205888


namespace distance_between_parallel_lines_l205_205040

theorem distance_between_parallel_lines :
  let l1 : ℝ × ℝ × ℝ := (1, -2, 1)
  let l2 : ℝ × ℝ × ℝ := (1, -2, -4)
  ∀ (a b c1 c2 : ℝ),
    (a, b, c1) = l1 →
    (a, b, c2) = l2 →
    (a ≠ 0 ∨ b ≠ 0) →
    real.sqrt ((c1 - c2) * (c1 - c2)) / real.sqrt (a * a + b * b) = real.sqrt 5 :=
begin
  intros a b c1 c2,
  intros h1 h2 ha,
  have h : (c1 - c2) = 5,
  {
    rw [h1, h2],
    simp,
  },
  rw h,
  simp,
  sorry
end

end distance_between_parallel_lines_l205_205040


namespace equal_center_intersection_l205_205485

/-- Given a finite group G, and subgroups H1 and H2, if for any representation of G on
a finite-dimensional complex vector space V, the dimensions of the H1-invariant
and H2-invariant subspaces are equal, then Z(G) ∩ H1 = Z(G) ∩ H2, where Z(G) is the center of G. -/
theorem equal_center_intersection {G : Type*} [Group G] [Fintype G] [FiniteGroup G]
  (H1 H2 : Subgroup G)
  (h : ∀ (V : Type*) [AddCommGroup V] [Module ℂ V] [FiniteDimensional ℂ V],
    ∀ (ρ : Representation ℂ G V),
      FiniteDimensional.finrank ℂ (ρ.fixedPoints H1) =
      FiniteDimensional.finrank ℂ (ρ.fixedPoints H2)) :
  (Subgroup.center G) ⊓ H1 = (Subgroup.center G) ⊓ H2 :=
by sorry

end equal_center_intersection_l205_205485


namespace impossible_to_place_numbers_l205_205465

noncomputable def problem_statement : Prop :=
∀ (table : list (list ℕ)),
  (∀ row ∈ table, row.length = 7) ∧   -- Each row has 7 columns
  table.length = 6 ∧                  -- The table has 6 rows
  (∀ i j, 1 ≤ i ∧ i < 6 → 1 ≤ j ∧ j ≤ 7 → table.nth_le i j + table.nth_le (i+1) j % 2 = 0) → -- Each 1x2 vertical rectangle sum is even
  false                               -- Conclusion: It's impossible (contradiction)

theorem impossible_to_place_numbers : problem_statement :=
sorry

end impossible_to_place_numbers_l205_205465


namespace units_digit_47_power_47_l205_205613

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l205_205613


namespace solve_for_diamond_l205_205429

theorem solve_for_diamond (d : ℕ) (h : d * 5 + 3 = d * 6 + 2) : d = 1 :=
by
  sorry

end solve_for_diamond_l205_205429


namespace red_candies_count_l205_205063

def total_candies : ℕ := 3409
def blue_candies : ℕ := 3264

theorem red_candies_count : total_candies - blue_candies = 145 := by
  sorry

end red_candies_count_l205_205063


namespace sum_reciprocal_arith_seq_l205_205757

def arithmetic_seq (a : ℕ → ℕ) := ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℕ) (n : ℕ) := ∑ i in range n, a (i + 1)

theorem sum_reciprocal_arith_seq (a : ℕ → ℕ) (n : ℕ)
  (h_seq : arithmetic_seq a)
  (h_a3 : a 3 = 3)
  (h_S4 : Sn a 4 = 10) :
  \sum k in range n, (1 / Sn a k.succ) = 2n / (n+1) :=
sorry

end sum_reciprocal_arith_seq_l205_205757


namespace inequality_proof_l205_205007

theorem inequality_proof 
  (a b c d e f : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hd : 0 < d) 
  (he : 0 < e) 
  (hf : 0 < f) : 
  real.sqrt 3 (abc / (a + b + d)) + real.sqrt 3 (def / (c + e + f)) < 
  real.sqrt 3 ((a + b + d) * (c + e + f)) :=
sorry

end inequality_proof_l205_205007


namespace meteor_point_movement_to_line_l205_205222

theorem meteor_point_movement_to_line (M : Type) [point M] [line M]
  (meteor : M → Prop) (trail : M → Prop):
  (∀ p : M, meteor p → point p) → 
  (∀ p1 p2 : M, meteor p1 ∧ meteor p2 → line (p1, p2) → trail (p1, p2)) → 
  ∃ L : M → M → Prop, ∀ p1 p2 : M, meteor p1 ∧ meteor p2 → L p1 p2 :=
begin
  sorry
end

end meteor_point_movement_to_line_l205_205222


namespace solve_functional_equation_l205_205308

-- Definition of the conditions for the problem
def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y - 1) + f(x) * f(y) = 2 * x * y - 1

theorem solve_functional_equation :
  ∀ f : ℝ → ℝ, satisfies_functional_equation f → (f = (λ x, x) ∨ f = (λ x, -x^2)) :=
begin
  sorry
end

end solve_functional_equation_l205_205308


namespace equation_of_hyperbola_P_lies_on_fixed_line_l205_205363

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- The conditions provided in the problem
def focus := (-2 * Real.sqrt 5, 0)
def center := (0, 0)
def eccentricity : ℝ := Real.sqrt 5
def vertex1 := (-2, 0)
def vertex2 := (2, 0)
def passing_point := (-4, 0)

-- Finding the equation of the hyperbola
theorem equation_of_hyperbola : 
  ∀ (a b c : ℝ), 
  c = 2 * Real.sqrt 5 ∧ 
  eccentricity = Real.sqrt 5 ∧ 
  a = 2 ∧
  b = 4 → 
  hyperbola 2 4 :=
by
  sorry

-- Proving P lies on a fixed line
theorem P_lies_on_fixed_line :
  ∀ (m : ℝ), 
  let line : ℝ → ℝ := λ y, m * y - 4 in
  let intersection_M := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let intersection_N := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let P := (*) -- Intersection point of lines MA1 and NA2 -- (*) 
  ∃ (P : ℝ × ℝ), P.fst = -1 :=
by
  sorry

end equation_of_hyperbola_P_lies_on_fixed_line_l205_205363


namespace tan_half_angle_ratio_l205_205516

noncomputable def hyperbola {a b x y : ℝ} (P : ℝ × ℝ)
(f1 f2 α β : ℝ) :=
x^2 / a^2 - y^2 / b^2 = 1 ∧
f1 = (-sqrt (a^2 + b^2), 0) ∧
f2 = (sqrt (a^2 + b^2), 0) ∧
∃ (α β : ℝ), α = angle P f1 f2 ∧ β = angle P f2 f1

theorem tan_half_angle_ratio {a b c : ℝ}
(P : ℝ × ℝ)
(f1 f2 : ℝ × ℝ)
(α β : ℝ)
(hyperb : hyperbola P f1 f2 α β)
(h_c : c = sqrt (a^2 + b^2)) :
( (Real.tan (α / 2)) / (Real.tan (β / 2)) = (c - a) / (c + a) ) :=
sorry

end tan_half_angle_ratio_l205_205516


namespace angle_FAC_eq_angle_EDB_l205_205001

theorem angle_FAC_eq_angle_EDB
  (A B C D E F : Type)
  [convex_quadrilateral A B C D]
  (h_EF_on_BC : E ∈ line_segment B C ∧ F ∈ line_segment B C)
  (h_E_closer_to_B : dist B E < dist B F)
  (h_angle_BAE_eq_angle_CDF : ∠BAE = ∠CDF)
  (h_angle_EAF_eq_angle_FDE : ∠EAF = ∠FDE) :
  ∠FAC = ∠EDB := 
sorry

end angle_FAC_eq_angle_EDB_l205_205001


namespace equation_of_hyperbola_point_P_on_fixed_line_l205_205375

-- Definition and conditions
def hyperbola_center_origin := (0, 0)
def hyperbola_left_focus := (-2 * Real.sqrt 5, 0)
def hyperbola_eccentricity := Real.sqrt 5
def point_on_line_through_origin := (-4, 0)
def point_M_in_second_quadrant (M : ℝ × ℝ) := M.1 < 0 ∧ M.2 > 0
def vertices_A1_A2 := ((-2, 0), (2, 0))

theorem equation_of_hyperbola (c a b : ℝ) (h_c : c = 2 * Real.sqrt 5)
  (h_e : hyperbola_eccentricity = Real.sqrt 5) (h_a : a = 2) (h_b : b = 4) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 16) = 1) := 
by
  intros x y h_x h_y
  have h : c^2 = a^2 + b^2 := sorry
  have h2 : hyperbola_eccentricity = c / a := sorry
  exact sorry

theorem point_P_on_fixed_line (M N P : ℝ × ℝ) (A1 A2 : ℝ × ℝ)
  (h_vertices : vertices_A1_A2 = (A1, A2))
  (h_M_in_2nd : point_M_in_second_quadrant M) 
  (h_P_intersects : ∀ t : ℝ, P = (t, (A1.2 * A2.1 - A1.1 * A2.2) / (A1.1 - A2.1))) :
  (P.1 = -1) := sorry

end equation_of_hyperbola_point_P_on_fixed_line_l205_205375


namespace units_digit_of_sum_of_squares_of_first_1505_odds_l205_205710

def odd_sequence (n : ℕ) : ℕ := 2 * n + 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_squares_units_digit (n : ℕ) : ℕ :=
  let squares := List.map (λ i, odd_sequence i ^ 2) (List.range n)
  units_digit (List.sum (List.map units_digit squares))

theorem units_digit_of_sum_of_squares_of_first_1505_odds :
  sum_of_squares_units_digit 1505 = 5 :=
sorry

end units_digit_of_sum_of_squares_of_first_1505_odds_l205_205710


namespace problem_solution_l205_205740

theorem problem_solution (x : ℝ) : 
  (x < -2 ∨ (-2 < x ∧ x ≤ 0) ∨ (0 < x ∧ x < 2) ∨ (2 ≤ x ∧ x < (15 - Real.sqrt 257) / 8) ∨ ((15 + Real.sqrt 257) / 8 < x)) ↔ 
  (x^2 - 1) / (x + 2) ≥ 3 / (x - 2) + 7 / 4 := sorry

end problem_solution_l205_205740


namespace smallest_pos_int_div_by_four_primes_l205_205113

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205113


namespace quadratic_polynomial_sums_to_x_squared_l205_205909

noncomputable def P (x : ℝ) : ℝ := (1/11) * x^2 - (10/11) * x + (15/11)

theorem quadratic_polynomial_sums_to_x_squared : 
  ∀ x : ℝ, (∑ k in Finset.range 11, P (x + k)) = x^2 := 
by
  sorry

end quadratic_polynomial_sums_to_x_squared_l205_205909


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205073

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205073


namespace smallest_positive_period_of_f_range_of_f_in_interval_l205_205792

variables (x : ℝ)

-- Define the given function
def f (x : ℝ) : ℝ := (sin x)^2 + 2 * sqrt 3 * (sin x) * (cos x) + 3 * (cos x)^2

-- Problem 1: Period Proof
theorem smallest_positive_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, ((∀ x, f (x + T') = f x) → T' ≥ T)) := sorry

-- Problem 2: Range Proof
theorem range_of_f_in_interval : set.range (λ x, f x) (set.Icc (-π/6) (π/3)) = set.Icc 1 4 := sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l205_205792


namespace cos_C_in_triangle_l205_205459

theorem cos_C_in_triangle
  (A B C : ℝ)
  (sin_A : Real.sin A = 4 / 5)
  (cos_B : Real.cos B = 3 / 5) :
  Real.cos C = 7 / 25 :=
sorry

end cos_C_in_triangle_l205_205459


namespace probability_of_divisible_by_9_and_sum_is_913_l205_205876

open Nat

def S := {n : ℕ | ∃ j k : ℕ, j < k ∧ k < 40 ∧ n = 2^j + 2^k}

def number_of_elements_divisible_by_9_in_S : ℕ := 133

def total_number_of_elements_in_S : ℕ := 780

def fraction_in_simplest_form : (ℕ × ℕ) := (133, 780)

theorem probability_of_divisible_by_9_and_sum_is_913 :
  let p := fraction_in_simplest_form.fst
  let q := fraction_in_simplest_form.snd in
  p + q = 913 :=
by
  sorry

end probability_of_divisible_by_9_and_sum_is_913_l205_205876


namespace find_hyperbola_fixed_line_through_P_l205_205395

-- Define the conditions of the problem
def center_origin (C : Type) : Prop := 
  C.center = (0, 0)

def left_focus_at (C : Type) : Prop :=
  C.left_focus = (-2 * Real.sqrt 5, 0)

def eccentricity_sqrt_5 (C : Type) : Prop :=
  C.eccentricity = Real.sqrt 5

-- Define the specific hyperbola equation based on the conditions
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 16) = 1

-- State the results to prove
theorem find_hyperbola (C : Type) :
  center_origin C →
  left_focus_at C →
  eccentricity_sqrt_5 C →
  ∀ x y : ℝ, hyperbola_equation x y :=
sorry

-- Define the fixed line for point P
def fixed_line (P : Type) : Prop :=
  P.x = -1

-- State the results to prove for point P
theorem fixed_line_through_P (M N A1 A2 P : Type) (x y : ℝ) :
  (M.line_passing_through (Point.mk (-4, 0))).intersects_left_branch (N : Point) →
  M.in_second_quadrant →
  (A1.coords = (-2, 0)) ∧ (A2.coords = (2, 0)) →
  Point.mk x y = P →
  fixed_line P :=
sorry

end find_hyperbola_fixed_line_through_P_l205_205395


namespace horizontal_asymptote_crossing_l205_205761

def g (x : ℝ) : ℝ := (3 * x^2 - 6 * x - 9) / (x^2 - 5 * x + 6)

theorem horizontal_asymptote_crossing : ∃ x : ℝ, g x = 3 ∧ x = 3 :=
by
  use 3
  split
  sorry
  rfl

end horizontal_asymptote_crossing_l205_205761


namespace equation_of_hyperbola_P_lies_on_fixed_line_l205_205357

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- The conditions provided in the problem
def focus := (-2 * Real.sqrt 5, 0)
def center := (0, 0)
def eccentricity : ℝ := Real.sqrt 5
def vertex1 := (-2, 0)
def vertex2 := (2, 0)
def passing_point := (-4, 0)

-- Finding the equation of the hyperbola
theorem equation_of_hyperbola : 
  ∀ (a b c : ℝ), 
  c = 2 * Real.sqrt 5 ∧ 
  eccentricity = Real.sqrt 5 ∧ 
  a = 2 ∧
  b = 4 → 
  hyperbola 2 4 :=
by
  sorry

-- Proving P lies on a fixed line
theorem P_lies_on_fixed_line :
  ∀ (m : ℝ), 
  let line : ℝ → ℝ := λ y, m * y - 4 in
  let intersection_M := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let intersection_N := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let P := (*) -- Intersection point of lines MA1 and NA2 -- (*) 
  ∃ (P : ℝ × ℝ), P.fst = -1 :=
by
  sorry

end equation_of_hyperbola_P_lies_on_fixed_line_l205_205357


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205154

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205154


namespace find_angle_BPC_l205_205841

-- Define the problem setup
def rectangle (A B C D E P : Type) (AB BC BE : ℝ) (h_AB : AB = 6) (h_BC : BC = 3) (h_BE : BE = 4) : Prop :=
triangle_is_right A B E ∧ line_segments_intersect BE AC P ∧ AB = 6 ∧ BC = 3 ∧ BE = 4

-- Define the main theorem statement
theorem find_angle_BPC {A B C D E P : Type} (AB BC BE : ℝ) (h_AB : AB = 6) (h_BC : BC = 3) (h_BE : BE = 4)
  (h_rectangle : rectangle A B C D E P AB BC BE h_AB h_BC h_BE) : 
  angle_measures_BPC := 90 :=
begin
  -- Placeholder for the proof
  sorry
end

end find_angle_BPC_l205_205841


namespace stratified_sampling_l205_205233

theorem stratified_sampling
  (total_students : ℕ) (freshmen : ℕ) (sophomores : ℕ) (juniors : ℕ) (sample_size : ℕ)
  (h1 : total_students = 2000)
  (h2 : freshmen = 800)
  (h3 : sophomores = 600)
  (h4 : juniors = 600)
  (h5 : sample_size = 50) :
  let probability := (sample_size : ℚ) / (total_students : ℚ),
      sampled_freshmen := (freshmen : ℚ) * probability,
      sampled_sophomores := (sophomores : ℚ) * probability,
      sampled_juniors := (juniors : ℚ) * probability
  in sampled_freshmen = 20 ∧ sampled_sophomores = 15 ∧ sampled_juniors = 15 :=
by
  sorry

end stratified_sampling_l205_205233


namespace frustum_volume_ratio_l205_205953

theorem frustum_volume_ratio (A1 A2 V1 V2: ℝ) (h1 h2: ℝ) 
  (hA_ratio: A1 / A2 = 1 / 9) 
  (h1_eq: h1 = sqrt (A1 / A2)) 
  (h2_eq: h2 = sqrt (A2 / A1)) :
  (V1 / V2 = 7 / 19) :=
sorry

end frustum_volume_ratio_l205_205953


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205131

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205131


namespace largest_integer_x_l205_205598

theorem largest_integer_x (x : ℤ) (h : 3 - 5 * x > 22) : x ≤ -4 :=
by
  sorry

end largest_integer_x_l205_205598


namespace only_g_increases_as_x_increases_l205_205263

open Real

noncomputable def f (x : ℝ) : ℝ := -2 * x
noncomputable def g (x : ℝ) : ℝ := 3 * x - 1
noncomputable def h (x : ℝ) : ℝ := 1 / x
noncomputable def k (x : ℝ) : ℝ := x ^ 2

theorem only_g_increases_as_x_increases :
  (∀ x, deriv g x > 0) ∧
  ¬(∀ x, deriv f x > 0) ∧
  ¬(∀ x ∈ Ioi 0, deriv h x > 0) ∧
  ¬(∀ x, deriv k x > 0) :=
by
  sorry

end only_g_increases_as_x_increases_l205_205263


namespace identity_element_exists_identity_element_self_commutativity_associativity_l205_205014

noncomputable def star_op (a b : ℤ) : ℤ := a + b + a * b

theorem identity_element_exists : ∃ E : ℤ, ∀ a : ℤ, star_op a E = a :=
by sorry

theorem identity_element_self (E : ℤ) (h1 : ∀ a : ℤ, star_op a E = a) : star_op E E = E :=
by sorry

theorem commutativity (a b : ℤ) : star_op a b = star_op b a :=
by sorry

theorem associativity (a b c : ℤ) : star_op (star_op a b) c = star_op a (star_op b c) :=
by sorry

end identity_element_exists_identity_element_self_commutativity_associativity_l205_205014


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205143

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205143


namespace length_DE_l205_205549

-- Definitions for conditions in the problem
variables (AB : ℝ) (DE : ℝ) (areaABC : ℝ)

-- Condition: AB is 15 cm
axiom length_AB : AB = 15

-- Condition: The area of triangle projected below the base is 25% of the area of triangle ABC
axiom area_ratio_condition : (1 / 4) * areaABC = (1 / 2)^2 * areaABC

-- The problem statement translated to Lean proof
theorem length_DE : DE = 7.5 :=
by
  -- Definitions and conditions
  have h1 : AB = 15 := length_AB
  have h2 : (1 / 2)^2 = 1 / 4 := by ring
  calc
    DE = (0.5) * AB :  sorry  -- proportional relationship since triangles are similar
    ... = 0.5 * 15   :  by rw [h1]
    ... = 7.5       :  by norm_num

end length_DE_l205_205549


namespace arithmetic_sequence_fourth_term_l205_205847

variables {b d : ℝ}
def third_term := b
def common_difference := d
def fourth_term := b + d
def fifth_term := b + 2 * d

theorem arithmetic_sequence_fourth_term :
  (third_term + fifth_term = 10) → (fourth_term = 5) :=
by intros h ; sorry

end arithmetic_sequence_fourth_term_l205_205847


namespace area_of_square_field_l205_205028

theorem area_of_square_field (s : ℕ) (A : ℕ) (cost_per_meter : ℕ) 
  (total_cost : ℕ) (gate_width : ℕ) (num_gates : ℕ) 
  (h1 : cost_per_meter = 1)
  (h2 : total_cost = 666)
  (h3 : gate_width = 1)
  (h4 : num_gates = 2)
  (h5 : (4 * s - num_gates * gate_width) * cost_per_meter = total_cost) :
  A = s * s → A = 27889 :=
by
  sorry

end area_of_square_field_l205_205028


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205159

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205159


namespace max_distance_to_line_l205_205411

-- Definitions based on the given problem conditions
def curveC (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1
def lineL (x y : ℝ) : Prop := x + y - 4 = 0

-- The problem statement in Lean 4
theorem max_distance_to_line :
  ∀ θ : ℝ, ∃ d : ℝ, d = 3 * Real.sqrt 2 ∧
    ∀ P : ℝ × ℝ, curveC P.1 P.2 → 
      let dist := |P.1 + P.2 - 4| / Real.sqrt 2 in
        dist ≤ d :=
sorry

end max_distance_to_line_l205_205411


namespace radius_of_shorter_can_l205_205067

-- Define variables and assumptions
variables (h r : ℝ) (V : ℝ) (π : ℝ := real.pi)

-- Define the conditions
def cans_have_same_volume (V : ℝ) (h r : ℝ) : Prop :=
  (π * (16 ^ 2) * (4 * h)) = (π * (r ^ 2) * h)

def height_ratio_four (h : ℝ) : Prop :=
  h = 4 * h

def radius_of_taller_can : Prop :=
  true -- this condition is well-defined by variables

-- The problem statement: Prove that radius of the shorter can is 32 units.
theorem radius_of_shorter_can (h : ℝ) (r: ℝ) (h : ℝ) (H1 : cans_have_same_volume h r) (H2 : height_ratio_four h) (H3 : radius_of_taller_can) : 
  r = 32 :=
by
  sorry

end radius_of_shorter_can_l205_205067


namespace locus_of_centers_of_inscribed_parallelograms_is_open_line_l205_205660

-- Define a quadrilateral with vertices A, B, C, and D
variables {A B C D : Point}
-- Define midpoints P and Q of the diagonals AC and BD, respectively
variables {P : Point} (hP : midpoint A C = P)
variables {Q : Point} (hQ : midpoint B D = Q)
-- Define a series of parallelograms MNKL inscribed in quadrilateral ABCD
variables {M N K L : Point}
-- The family of parallelograms MNKL with sides parallel to the diagonals of the quadrilateral
variables (h1 : parallel (line M N) (line A C)) 
          (h2 : parallel (line K L) (line B D))
          (h3 : parallel (line M L) (line B D)) 
          (h4 : parallel (line N K) (line A C))

-- The proof statement
theorem locus_of_centers_of_inscribed_parallelograms_is_open_line (center : Point) :
  center ∈ line_segment P Q :=
sorry

end locus_of_centers_of_inscribed_parallelograms_is_open_line_l205_205660


namespace planting_plan_l205_205305

theorem planting_plan (a : ℝ) (ha : 0 ≤ a) (ha_le : a ≤ 100) :
  4.9 * a + 0.5 * (100 - a) * 10 ≥ 496 ↔ a ≤ 40 :=
begin
  sorry
end

end planting_plan_l205_205305


namespace last_digit_infinite_occurrences_l205_205857

theorem last_digit_infinite_occurrences :
  ∃ᶠ (n : ℕ) in at_top, (n^n % 10) ∈ {1, 6, 7, 9} :=
begin
  sorry
end

end last_digit_infinite_occurrences_l205_205857


namespace missing_number_in_sequence_l205_205061

theorem missing_number_in_sequence : 
  ∃ n : ℕ, (1 :: 2 :: 4 :: n :: 16 :: 32 :: list.nil) = [1, 2, 4, 8, 16, 32] :=
by
  use 8
  sorry

end missing_number_in_sequence_l205_205061


namespace heads_at_least_twice_in_5_tosses_l205_205642

noncomputable def probability_at_least_two_heads (n : ℕ) (p : ℚ) : ℚ :=
1 - (n : ℚ) * p^(n : ℕ)

theorem heads_at_least_twice_in_5_tosses :
  probability_at_least_two_heads 5 (1/2) = 13/16 :=
by
  sorry

end heads_at_least_twice_in_5_tosses_l205_205642


namespace amount_dog_ate_cost_l205_205481

-- Define the costs of each ingredient
def cost_flour : Real := 4
def cost_sugar : Real := 2
def cost_butter : Real := 2.5
def cost_eggs : Real := 0.5

-- Define the number of slices
def number_of_slices := 6

-- Define the number of slices eaten by Laura's mother
def slices_eaten_by_mother := 2

-- Calculate the total cost of the ingredients
def total_cost := cost_flour + cost_sugar + cost_butter + cost_eggs

-- Calculate the cost per slice
def cost_per_slice := total_cost / number_of_slices

-- Calculate the number of slices eaten by Kevin
def slices_eaten_by_kevin := number_of_slices - slices_eaten_by_mother

-- Define the total cost of slices eaten by Kevin
def cost_eaten_by_kevin := slices_eaten_by_kevin * cost_per_slice

-- The main statement to prove
theorem amount_dog_ate_cost :
  cost_eaten_by_kevin = 6 := by
    sorry

end amount_dog_ate_cost_l205_205481


namespace vector_magnitude_proof_l205_205420

noncomputable def vector_a : Type := ℝ
noncomputable def vector_b : Type := ℝ

def max_norm (a b : vector_a × vector_b) : ℝ := 
  let ⟨a1, a2⟩ := a in
  let ⟨b1, b2⟩ := b in
  sqrt ((a1 + 5 * b1) ^ 2 + (a2 + 5 * b2) ^ 2)

theorem vector_magnitude_proof
  (a b : vector_a × vector_b)
  (h₁ : (2 * a - 3 * b).norm = 2)
  (h₂ : (3 * a + 2 * b).norm = 1) :
  (max_norm a b ≤ 4) ∧ (b.norm / a.norm = 8) := sorry

end vector_magnitude_proof_l205_205420


namespace find_valid_n_l205_205048

def append_fives (n : ℕ) : ℕ := 1200 * 6^(10*n+2) + (5 * (6^0 + 6^1 + ... + 6^(10*n+1)))

def has_two_distinct_prime_factors (x : ℕ) : Prop :=
  let factors := factorization x
  factors.keys.to_finset.card = 2

theorem find_valid_n (n : ℕ) (x : ℕ) :
  x = append_fives n →
  has_two_distinct_prime_factors x :=
sorry

end find_valid_n_l205_205048


namespace dog_ate_cost_6_l205_205473

noncomputable def totalCost : ℝ := 4 + 2 + 0.5 + 2.5
noncomputable def costPerSlice : ℝ := totalCost / 6
noncomputable def slicesEatenByDog : ℕ := 6 - 2
noncomputable def costEatenByDog : ℝ := slicesEatenByDog * costPerSlice

theorem dog_ate_cost_6 : costEatenByDog = 6 := by
  sorry

end dog_ate_cost_6_l205_205473


namespace Rob_total_money_l205_205010

theorem Rob_total_money 
  (quarters: ℕ) 
  (dimes: ℕ) 
  (nickels: ℕ) 
  (pennies: ℕ) 
  (v_quarters: quarters = 7) 
  (v_dimes: dimes = 3) 
  (v_nickels: nickels = 5) 
  (v_pennies: pennies = 12) :
  0.25 * quarters + 0.10 * dimes + 0.05 * nickels + 0.01 * pennies = 2.42 :=
by
  -- The proof will go here
  sorry

end Rob_total_money_l205_205010


namespace median_to_BC_eq_Line_AD_l205_205809

structure Point := (x : ℝ) (y : ℝ)

def A := Point.mk (-1) 2
def B := Point.mk 3 (-1)
def C := Point.mk (-1) (-3)

def midpoint (P Q : Point) : Point :=
  Point.mk ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

def line_equation (P : Point) (m : ℝ) (x : ℝ) : ℝ :=
  m * (x - P.x) + P.y

theorem median_to_BC_eq_Line_AD : 
  ∃ (m : ℝ) (b : ℝ), ∀ x : ℝ, (midpoint B C).x = 1 → (midpoint B C).y = -2 → slope A (midpoint B C) = -2 → b = 0 → 
  (line_equation A (-2) x) = (-2 * x) :=
by
  sorry

end median_to_BC_eq_Line_AD_l205_205809


namespace find_standard_equation_validate_ratio_constant_l205_205400

-- Definitions stemming from the conditions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def passes_through (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 1 ∧ y = (2 * real.sqrt 3) / 3 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def valid_circle_line_intersection (b : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = 2 ∧ x + y + b = 0 → 
  real.sqrt ((x - (-y))^2 + (y - (-x))^2) = 2

-- Problems rephrased as proofs
theorem find_standard_equation (a b : ℝ) :
  is_ellipse a b 1 ((2 * real.sqrt 3) / 3) →
  valid_circle_line_intersection b →
  ∃ (a b : ℝ), (x y : ℝ) (is_ellipse a b x y → a = real.sqrt 3 ∧ b = real.sqrt 2) := sorry

theorem validate_ratio_constant (a b : ℝ) :
  is_ellipse a b 1 ((2 * real.sqrt 3) / 3) →
  valid_circle_line_intersection b →
  ∀ Q : ℝ × ℝ, Q ≠ (0, 0) →
  ∃ (M N OQ : ℝ × ℝ), (OQ.1^2 + OQ.2^2 ≠ 0) →
  (let OQ := λ (x, y : ℝ), real.sqrt (x^2 + y^2);
   ∃ ratio : ℝ, ratio = 2 * real.sqrt 3 / 3) := sorry

end find_standard_equation_validate_ratio_constant_l205_205400


namespace mass_of_fat_max_mass_of_carbohydrates_l205_205658

-- Definitions based on conditions
def total_mass : ℤ := 500
def fat_percentage : ℚ := 5 / 100
def protein_to_mineral_ratio : ℤ := 4

-- Lean 4 statement for Part 1: mass of fat
theorem mass_of_fat : (total_mass : ℚ) * fat_percentage = 25 := sorry

-- Definitions to utilize in Part 2
def max_percentage_protein_carbs : ℚ := 85 / 100
def mass_protein (x : ℚ) : ℚ := protein_to_mineral_ratio * x

-- Lean 4 statement for Part 2: maximum mass of carbohydrates
theorem max_mass_of_carbohydrates (x : ℚ) :
  x ≥ 50 → (total_mass - 25 - x - mass_protein x) ≤ 225 := sorry

end mass_of_fat_max_mass_of_carbohydrates_l205_205658


namespace weights_of_first_two_cats_l205_205468

noncomputable def cats_weight_proof (W : ℝ) : Prop :=
  (∀ (w1 w2 : ℝ), w1 = W ∧ w2 = W ∧ (w1 + w2 + 14.7 + 9.3) / 4 = 12) → (W = 12)

theorem weights_of_first_two_cats (W : ℝ) :
  cats_weight_proof W :=
by
  sorry

end weights_of_first_two_cats_l205_205468


namespace tan_of_pi_over_two_minus_alpha_l205_205775

theorem tan_of_pi_over_two_minus_alpha (α : ℝ) (h : Real.sin (Real.pi + α) = -1/3) : 
  Real.tan (Real.pi / 2 - α) = 2 * Real.sqrt 2 ∨ Real.tan (Real.pi / 2 - α) = -2 * Real.sqrt 2 := 
sorry

end tan_of_pi_over_two_minus_alpha_l205_205775


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205157

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205157


namespace not_exists_natural_number_divisible_by_5_l205_205733

def S (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), Nat.choose (2 * n + 1) (2 * k + 1) * 2 ^ (3 * k)

theorem not_exists_natural_number_divisible_by_5 :
  ¬ ∃ n : ℕ, S n % 5 = 0 :=
by
  sorry

end not_exists_natural_number_divisible_by_5_l205_205733


namespace length_DE_l205_205548

-- Definitions for conditions in the problem
variables (AB : ℝ) (DE : ℝ) (areaABC : ℝ)

-- Condition: AB is 15 cm
axiom length_AB : AB = 15

-- Condition: The area of triangle projected below the base is 25% of the area of triangle ABC
axiom area_ratio_condition : (1 / 4) * areaABC = (1 / 2)^2 * areaABC

-- The problem statement translated to Lean proof
theorem length_DE : DE = 7.5 :=
by
  -- Definitions and conditions
  have h1 : AB = 15 := length_AB
  have h2 : (1 / 2)^2 = 1 / 4 := by ring
  calc
    DE = (0.5) * AB :  sorry  -- proportional relationship since triangles are similar
    ... = 0.5 * 15   :  by rw [h1]
    ... = 7.5       :  by norm_num

end length_DE_l205_205548


namespace sequence_property_l205_205856

noncomputable def sequence (n : ℕ) : ℚ :=
if n % 4 = 0 then 3
else if n % 4 = 1 then 1 / 2
else if n % 4 = 2 then 1 / 3
else 2

theorem sequence_property :
  (sequence 2016) + (sequence 2017) = 7 / 2 := by
  sorry

end sequence_property_l205_205856


namespace wyatt_envelopes_fewer_l205_205191

-- Define assets for envelopes
variables (blue_envelopes yellow_envelopes : ℕ)

-- Conditions from the problem
def wyatt_conditions :=
  blue_envelopes = 10 ∧ yellow_envelopes < blue_envelopes ∧ blue_envelopes + yellow_envelopes = 16

-- Theorem: How many fewer yellow envelopes Wyatt has compared to blue envelopes?
theorem wyatt_envelopes_fewer (hb : blue_envelopes = 10) (ht : blue_envelopes + yellow_envelopes = 16) : 
  blue_envelopes - yellow_envelopes = 4 := 
by sorry

end wyatt_envelopes_fewer_l205_205191


namespace vector_transitivity_l205_205983

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem vector_transitivity (h1 : a = b) (h2 : b = c) : a = c :=
by {
  sorry
}

end vector_transitivity_l205_205983


namespace avg_salary_feb_mar_apr_may_l205_205939

def avg_salary_4_months : ℝ := 8000
def salary_jan : ℝ := 3700
def salary_may : ℝ := 6500
def total_salary_4_months := 4 * avg_salary_4_months
def total_salary_feb_mar_apr := total_salary_4_months - salary_jan
def total_salary_feb_mar_apr_may := total_salary_feb_mar_apr + salary_may

theorem avg_salary_feb_mar_apr_may : total_salary_feb_mar_apr_may / 4 = 8700 := by
  sorry

end avg_salary_feb_mar_apr_may_l205_205939


namespace arithmetic_mean_integers_l205_205707

theorem arithmetic_mean_integers (m n : ℤ) (h : m < n) : 
  (∑ i in finset.Icc m n, i : ℚ) / (n - m + 1) = (m + n) / 2 :=
sorry

end arithmetic_mean_integers_l205_205707


namespace find_abcd_l205_205573

theorem find_abcd {abcd abcde M : ℕ} :
  (M > 0) ∧ (∃ e, M % 100000 = e ∧ M^2 % 100000 = e) ∧ (M // 10000 > 0) ∧ (M // 10000 < 10) →
  (abcd = M // 10) →
  abcd = 9687 :=
by
  sorry

end find_abcd_l205_205573


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205146

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205146


namespace men_joined_l205_205557

-- Definitions for initial conditions
def initial_men : ℕ := 10
def initial_days : ℕ := 50
def extended_days : ℕ := 25

-- Theorem stating the number of men who joined the camp
theorem men_joined (x : ℕ) 
    (initial_food : initial_men * initial_days = (initial_men + x) * extended_days) : 
    x = 10 := 
sorry

end men_joined_l205_205557


namespace polygons_intersect_inside_circle_l205_205009

noncomputable def numberOfIntersections : ℕ :=
  let intersections_pair (n m : ℕ) : ℕ := 2 * (n - 2) * (m - 2) in
  intersections_pair 6 7 + intersections_pair 7 8 + intersections_pair 8 9

theorem polygons_intersect_inside_circle : numberOfIntersections = 184 := by
  sorry

end polygons_intersect_inside_circle_l205_205009


namespace remaining_fraction_repr_l205_205718

theorem remaining_fraction_repr {p : ℕ} (hp : Nat.Prime p) (hp_gt : p > 5) 
    (removedDigits : List ℕ) (hlen : removedDigits.length = 2012) :
    ∃ (a b : ℕ), Nat.coprime a b ∧ b % p = 0 :=
sorry

end remaining_fraction_repr_l205_205718


namespace cooperative_game_solution_l205_205069

theorem cooperative_game_solution (n : ℕ) (h1 : n < 60) (h2 : ∃ p q : ℕ, p ≠ q ∧ n % p = 0 ∧ n % q = 0) 
  (h3 : ∀ k : ℕ, (n % 10 ≠ k % 10 ∨ (∃ d : ℕ, d ∣ n ∧ d ∣ k)) → n ≠ k) 
  (h4 : ∀ d : ℕ, (nat.divisors n = nat.divisors d) → n = d) : n = 10 := 
sorry

end cooperative_game_solution_l205_205069


namespace final_weight_of_box_l205_205867

theorem final_weight_of_box (w1 w2 w3 w4 : ℝ) (h1 : w1 = 2) (h2 : w2 = 3 * w1) (h3 : w3 = w2 + 2) (h4 : w4 = 2 * w3) : w4 = 16 :=
by
  sorry

end final_weight_of_box_l205_205867


namespace total_presents_l205_205949

-- Definitions based on the problem conditions
def numChristmasPresents : ℕ := 60
def numBirthdayPresents : ℕ := numChristmasPresents / 2

-- Theorem statement
theorem total_presents : numChristmasPresents + numBirthdayPresents = 90 :=
by
  -- Proof is omitted
  sorry

end total_presents_l205_205949


namespace smallest_w_value_l205_205988

theorem smallest_w_value (w : ℕ) (hw : w > 0) :
  (∀ k : ℕ, (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (10^2 ∣ 936 * w)) ↔ w = 900 := 
sorry

end smallest_w_value_l205_205988


namespace not_floor_neg_eq_neg_floor_not_floor_double_x_eq_double_floor_x_not_floor_sum_le_sum_floor_l205_205998

noncomputable def floor_le_sub_floor {x y : ℝ} : ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ :=
by
  sorry

theorem not_floor_neg_eq_neg_floor {x : ℝ} : ¬(⌊ -x ⌋ = -⌊x⌋) :=
by
  have h1 : ⌊ -1.8 ⌋ ≠ -⌊ 1.8 ⌋ := by sorry
  show ¬(⌊ -x ⌋ = -⌊x⌋), from h1

theorem not_floor_double_x_eq_double_floor_x {x : ℝ} : ¬(⌊ 2x ⌋ = 2⌊x⌋) :=
by
  have h2 : ⌊ -2.8 ⌋ ≠ -4 := by sorry
  show ¬(⌊ 2x ⌋ = 2⌊x⌋), from h2

theorem not_floor_sum_le_sum_floor {x y : ℝ} : ¬(⌊ x + y ⌋ ≤ ⌊ x ⌋ + ⌊ y ⌋) :=
by
  have h3 : ⌊ 3.6 ⌋ ≤ 2 := by sorry
  show ¬(⌊ x + y ⌋ ≤ ⌊ x ⌋ + ⌊ y ⌋), from h3

end not_floor_neg_eq_neg_floor_not_floor_double_x_eq_double_floor_x_not_floor_sum_le_sum_floor_l205_205998


namespace dark_square_excess_in_grid_l205_205716

theorem dark_square_excess_in_grid :
  ∀ (m n : ℕ), m = 5 → n = 9 → 
  let dark_count := ((m + 1) / 2) * ((n + 1) / 2) + (m / 2) * (n / 2 + 1),
      light_count := ((m + 1) / 2) * (n / 2) + (m / 2) * ((n + 1) / 2)
  in dark_count - light_count = 5 :=
by sorry

end dark_square_excess_in_grid_l205_205716


namespace max_squares_on_grid_l205_205594

theorem max_squares_on_grid : 
  ∀ (points : Finset (ℤ × ℤ)), 
  points.card = 12 ∧ (∀ p ∈ points, p.1 ∈ {0, 1, 2} ∧ p.2 ∈ {0, 1, 2, 3}) →
  (∃ (squares : Finset (Finset (ℤ × ℤ))),
    ∀ (square ∈ squares), square.card = 4 ∧
    (∃ (a b : ℤ × ℤ), 
      (a ∈ square ∧ b ∈ square ∧ a ≠ b ∧ 
      (a.1 = b.1 ∨ a.2 = b.2 ∨ abs (a.1 - b.1) = abs (a.2 - b.2))) ∧
      ( ∀ (x y ∈ square), x.1 - y.1 = 0 ∨ x.2 - y.2 = 0 ∨ abs (x.1 - y.1) = abs (x.2 - y.2))) ∧
  squares.card = 11) :=
sorry

end max_squares_on_grid_l205_205594


namespace leak_drains_tank_in_6_hours_l205_205242

theorem leak_drains_tank_in_6_hours :
  (∀ (P L : ℝ),
    P = 1 / 2 →                     -- Condition 1: Pump fills the tank in 2 hours
    (P - L = 1 / 3) →               -- Condition 2: Combined rate with leak
    L > 0 →                         -- Condition 3: Leak rate is positive
    L = 1 / 6 ∨ (L = 0.167 ∧ (L * 6 ≈ 1))    -- Question: Proving that the leak takes approximately 6 hours to drain the tank
  ) :=
by
  intros P L hP hCombined hLpos
  sorry

end leak_drains_tank_in_6_hours_l205_205242


namespace total_guppies_l205_205260

noncomputable def initial_guppies : Nat := 7
noncomputable def baby_guppies_first_set : Nat := 3 * 12
noncomputable def baby_guppies_additional : Nat := 9

theorem total_guppies : initial_guppies + baby_guppies_first_set + baby_guppies_additional = 52 :=
by
  sorry

end total_guppies_l205_205260


namespace solve_equation_l205_205741

theorem solve_equation : 
  ∀ x : ℝ, 
    (\frac{7}{real.sqrt (x - 8) - 10} + 
     \frac{2}{real.sqrt (x - 8) - 4} + 
     \frac{9}{real.sqrt (x - 8) + 4} + 
     \frac{14}{real.sqrt (x - 8) + 10} = 0) 
  ↔ x = 55 := 
by sorry

end solve_equation_l205_205741


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205084

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205084


namespace greatest_integer_less_than_AD_l205_205850

noncomputable def findGreatestIntegerLessThanAD (A B C D E : Point)
  (AB_length : Real) 
  (AB_eq : AB_length = 120)
  (isMidpoint : isMidpoint E A D)
  (angle_condition : angle BE AC = 45) : ℤ :=
by
  sorry
  
theorem greatest_integer_less_than_AD :
  ∀ (A B C D E: Point)
  (AB_length : Real) 
  (AB_eq : AB_length = 120)
  (isMidpoint : isMidpoint E A D)
  (angle_condition : angle BE AC = 45),
  findGreatestIntegerLessThanAD A B C D E AB_length AB_eq isMidpoint angle_condition = 169 := 
by 
  sorry

end greatest_integer_less_than_AD_l205_205850


namespace inequality_problem_l205_205345

open Real

theorem inequality_problem {a b c d : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_ac : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3 := 
by 
  sorry

end inequality_problem_l205_205345


namespace magnitude_vector_diff_eq_l205_205418

open Real

def vector := ℝ × ℝ

def magnitude (v : vector) : ℝ := sqrt (v.1^2 + v.2^2)

def parallel (v₁ v₂ : vector) : Prop :=
  ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

def vector_diff (v₁ v₂ : vector) : vector := 
  (v₁.1 - v₂.1, v₁.2 - v₂.2)

noncomputable def vector_a : vector := (2, -1)

noncomputable def vector_b (x : ℝ) : vector := (6, x)

theorem magnitude_vector_diff_eq : 
  ∀ x : ℝ, parallel vector_a (vector_b x) → magnitude (vector_diff vector_a (vector_b x)) = 2 * sqrt 5 :=
by
  sorry

end magnitude_vector_diff_eq_l205_205418


namespace packages_per_box_l205_205471

theorem packages_per_box (P : ℕ) 
  (h1 : 100 * 25 = 2500) 
  (h2 : 2 * P * 250 = 2500) : 
  P = 5 := 
sorry

end packages_per_box_l205_205471


namespace negation_of_p_l205_205517

theorem negation_of_p : (¬ ∀ x : ℝ, exp x ≥ 1) ↔ ∃ x : ℝ, exp x < 1 := by
  sorry

end negation_of_p_l205_205517


namespace dense_position_system_correct_angles_l205_205553

def dense_position_system_angle_valid (α : ℝ) : Prop :=
  (sin α - cos α) ^ 2 = 2 * sin α * cos α

def angle_values_in_dense_position_system (α : ℝ) : Prop :=
  ∃ k : ℤ, α = (250 / 6000) * (2 * Real.pi) + k * Real.pi ∨ 
            α = (1250 / 6000) * (2 * Real.pi) + k * Real.pi ∨ 
            α = (3250 / 6000) * (2 * Real.pi) + k * Real.pi

theorem dense_position_system_correct_angles :
  ∀ α : ℝ, dense_position_system_angle_valid α → angle_values_in_dense_position_system α := 
by 
  sorry

end dense_position_system_correct_angles_l205_205553


namespace projection_onto_v_l205_205577

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

def vec1 : V := ⟨2, -1, 5⟩
def proj_v_on_vec1 : V := ⟨2, -1, 1⟩
def vec2 : V := ⟨5, 0, -3⟩
def proj_v_on_vec2 : V := ⟨14 / 6, -7 / 6, 7 / 6⟩

theorem projection_onto_v:
  (∃ (v : V), proj_of vec1 v = proj_v_on_vec1) → 
  proj_of vec2 v = proj_v_on_vec2 :=
by
  sorry

end projection_onto_v_l205_205577


namespace fraction_of_180_l205_205071

theorem fraction_of_180 : (1 / 2) * (1 / 3) * (1 / 6) * 180 = 5 := by
  sorry

end fraction_of_180_l205_205071


namespace price_of_10_pound_bag_l205_205663

variables (P : ℝ) -- price of the 10-pound bag
def cost (n5 n10 n25 : ℕ) := n5 * 13.85 + n10 * P + n25 * 32.25

theorem price_of_10_pound_bag (h : ∃ (n5 n10 n25 : ℕ), n5 * 5 + n10 * 10 + n25 * 25 ≥ 65
  ∧ n5 * 5 + n10 * 10 + n25 * 25 ≤ 80 
  ∧ cost P n5 n10 n25 = 98.77) : 
  P = 20.42 :=
by
  -- Proof skipped
  sorry

end price_of_10_pound_bag_l205_205663


namespace student_competition_distribution_l205_205764

theorem student_competition_distribution :
  ∃ f : Fin 4 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ x : Fin 4, f x = i ∧ ∃ y : Fin 4, f y = j) ∧ 
  (Finset.univ.image f).card = 3 := 
sorry

end student_competition_distribution_l205_205764


namespace least_pos_int_div_by_four_distinct_primes_l205_205096

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205096


namespace Jessie_final_weight_l205_205469

variable (initial_weight : ℝ) (loss_first_week : ℝ) (loss_rate_second_week : ℝ)
variable (loss_second_week : ℝ) (total_loss : ℝ) (final_weight : ℝ)

def Jessie_weight_loss_problem : Prop :=
  initial_weight = 92 ∧
  loss_first_week = 5 ∧
  loss_rate_second_week = 1.3 ∧
  loss_second_week = loss_rate_second_week * loss_first_week ∧
  total_loss = loss_first_week + loss_second_week ∧
  final_weight = initial_weight - total_loss ∧
  final_weight = 80.5

theorem Jessie_final_weight : Jessie_weight_loss_problem initial_weight loss_first_week loss_rate_second_week loss_second_week total_loss final_weight :=
by
  sorry

end Jessie_final_weight_l205_205469


namespace dog_ate_cost_is_six_l205_205476

-- Definitions for the costs
def flour_cost : ℝ := 4
def sugar_cost : ℝ := 2
def butter_cost : ℝ := 2.5
def eggs_cost : ℝ := 0.5

-- Total cost calculation
def total_cost := flour_cost + sugar_cost + butter_cost + eggs_cost

-- Initial slices and remaining slices
def initial_slices : ℕ := 6
def eaten_slices : ℕ := 2
def remaining_slices := initial_slices - eaten_slices

-- The cost calculation of the amount the dog ate
def dog_ate_cost := (remaining_slices / initial_slices) * total_cost

-- Proof statement
theorem dog_ate_cost_is_six : dog_ate_cost = 6 :=
by
  sorry

end dog_ate_cost_is_six_l205_205476


namespace simplify_trigonometric_sum_l205_205528

noncomputable def trigonometric_sum_simplification (n : ℕ) (α θ : ℝ) : ℝ :=
  ∑ k in finset.range(n+1), (-1)^k * (nat.choose n k) * real.sin (α + k * θ)

theorem simplify_trigonometric_sum (n : ℕ) (α θ : ℝ) :
  trigonometric_sum_simplification n α θ = 
    2^n * (real.sin (θ / 2))^n * real.sin ((3 * n * real.pi / 2) + (n * θ / 2) + α) :=
sorry

end simplify_trigonometric_sum_l205_205528


namespace isosceles_triangle_area_l205_205742

theorem isosceles_triangle_area (a b h : ℝ) (h_eq : h = a / (2 * Real.sqrt 3)) :
  (1 / 2 * a * h) = (a^2 * Real.sqrt 3) / 12 :=
by
  -- Define the necessary parameters and conditions
  let area := (1 / 2) * a * h
  have h := h_eq
  -- Substitute and prove the calculated area
  sorry

end isosceles_triangle_area_l205_205742


namespace find_equation_of_line_l205_205335

def line := ℝ × ℝ × ℝ -- ax + by + c = 0 represented as (a, b, c)

def point := ℝ × ℝ

def passes_through (P : point) (l : line) : Prop :=
  let (x, y) := P
  let (a, b, c) := l
  a * x + b * y + c = 0

def distance (A : point) (l : line) : ℝ :=
  let (x, y) := A
  let (a, b, c) := l
  abs (a * x + b * y + c) / sqrt (a*a + b*b)

def equidistant (A B : point) (l : line) : Prop :=
  distance A l = distance B l

theorem find_equation_of_line (P A B : point) (l : line) :
    P = (3, 4) → A = (-2, 2) → B = (4, -2) →
    passes_through P l ∧ equidistant A B l →
    l = (2, -1, -2) ∨ l = (2, 3, -18) :=
by
  intros
  sorry

end find_equation_of_line_l205_205335


namespace matrix_transformation_and_eigen_l205_205408

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 0], ![0, 4]]

-- Define points M and N
def M : Fin 2 → ℝ := ![-1, -1]
def N : Fin 2 → ℝ := ![1, 1]

-- Transformation of M and N by matrix A
def M' : Fin 2 → ℝ := A.mulVec M
def N' : Fin 2 → ℝ := A.mulVec N

-- Length of line segment M'N'
def length_MN_prime : ℝ :=
  Real.sqrt ((M' 0 - N' 0) ^ 2 + (M' 1 - N' 1) ^ 2)

-- Statement of the proof problem
theorem matrix_transformation_and_eigen (A : Matrix (Fin 2) (Fin 2) ℝ)
  (M N : Fin 2 → ℝ) :
  A = ![![3, 0], ![0, 4]] → M = ![-1, -1] → N = ![1, 1] →
  (Real.sqrt ((A.mulVec M 0 - A.mulVec N 0) ^ 2 + (A.mulVec M 1 - A.mulVec N 1) ^ 2) = 10) ∧ 
  (∀ λ : ℝ, (A - λ • (1 : Fin 2 → Fin 2 → ℝ)).det = 0 ↔ (λ = 3 ∨ λ = 4)) ∧
  (A.mulVec ![1, 0] = 3 • ![1, 0]) ∧ (A.mulVec ![0, 1] = 4 • ![0, 1]) :=
by 
  intros hA hM hN
  sorry

end matrix_transformation_and_eigen_l205_205408


namespace least_positive_integer_divisible_by_four_primes_l205_205117

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205117


namespace x_2013_eq_l205_205433

def f (x : ℝ) : ℝ := 4 * x / (x + 4)

def x₁ : ℝ := 1

noncomputable def x (n : ℕ) : ℝ :=
  if n = 1 then x₁ else f (x (n - 1))

theorem x_2013_eq : x 2013 = 1 / 504 :=
  sorry

end x_2013_eq_l205_205433


namespace drainage_volume_function_max_waste_density_volume_l205_205996

-- Define the conditions as given in the problem.
def V (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 0.2 then 90
  else if 0.2 < x ∧ x ≤ 2 then -50 * x + 100
  else 0

def f (x : ℝ) : ℝ := x * V(x)

-- Prove the required statements
theorem drainage_volume_function :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → V(x) = if 0 ≤ x ∧ x ≤ 0.2 then 90 else -50 * x + 100 :=
by
  intros x hx
  unfold V
  split_ifs with h1 h2
  . refl
  . refl
  sorry -- proof omitted for brevity

theorem max_waste_density_volume : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ f(x) = 50 :=
by
  use 1
  unfold f V
  split
  . linarith
  split
  . linarith
  . simp
  . ring
  . exact (by norm_num : 50 = 50)
  sorry -- proof omitted for brevity

#check drainage_volume_function
#check max_waste_density_volume

end drainage_volume_function_max_waste_density_volume_l205_205996


namespace class_2_3_tree_count_total_tree_count_l205_205284

-- Definitions based on the given conditions
def class_2_5_trees := 142
def class_2_3_trees := class_2_5_trees - 18

-- Statements to be proved
theorem class_2_3_tree_count :
  class_2_3_trees = 124 :=
sorry

theorem total_tree_count :
  class_2_5_trees + class_2_3_trees = 266 :=
sorry

end class_2_3_tree_count_total_tree_count_l205_205284


namespace correct_statements_l205_205268

theorem correct_statements :
  let f1 := λ x : ℝ, |x| / x,
      g := λ x : ℝ, if x ≥ 0 then 1 else -1,
      f2 := λ x : ℝ, x^2 + 2 + 1 / (x^2 + 2),
      f3 := λ x : ℝ, |x - 1| - |x|
  in (∀ x : ℝ, f1 x = g x) ∧
     (∀ y : ℝ, y = 1 ↔ ∃ x : ℝ, f2 x = y) ∧
     f2 = 2 ∧
     f3 (f3 (1 / 2)) = 1 :=
begin
  sorry
end

end correct_statements_l205_205268


namespace contrapositive_example_l205_205941

theorem contrapositive_example (x : ℝ) :
  (x ^ 2 < 1 → -1 < x ∧ x < 1) ↔ (x ≥ 1 ∨ x ≤ -1 → x ^ 2 ≥ 1) :=
sorry

end contrapositive_example_l205_205941


namespace sin_of_tan_l205_205440

theorem sin_of_tan (k : ℝ) (h₁ : k > 0) (h₂ : ∠ MNP = 90°) (h₃ : tan P = 4/3) : sin P = 4/5 :=
sorry

end sin_of_tan_l205_205440


namespace units_digit_47_power_47_l205_205612

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l205_205612


namespace find_m_if_polynomial_is_perfect_square_l205_205436

theorem find_m_if_polynomial_is_perfect_square (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = x^2 + m * x + 4) → (m = 4 ∨ m = -4) :=
sorry

end find_m_if_polynomial_is_perfect_square_l205_205436


namespace perpendicular_bisector_of_AB_parallel_tangent_lines_to_circle_l205_205414

open Real

-- Define the given points A and B
def A : P := (8, -6)
def B : P := (2, 2)

-- Define the circle equation x^2 + y^2 = 16
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3 * x - 4 * y - 23 = 0

-- Define the parallel and tangent lines' equations
def parallel_tangent1 (x y : ℝ) : Prop := 4 * x + 3 * y + 20 = 0
def parallel_tangent2 (x y : ℝ) : Prop := 4 * x + 3 * y - 20 = 0

theorem perpendicular_bisector_of_AB : ∀ x y : ℝ, perp_bisector x y ↔ True :=
by sorry

theorem parallel_tangent_lines_to_circle : 
  ∀ (x y : ℝ), (parallel_tangent1 x y ↔ True) ∨ (parallel_tangent2 x y ↔ True) :=
by sorry

end perpendicular_bisector_of_AB_parallel_tangent_lines_to_circle_l205_205414


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205078

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205078


namespace equation_of_hyperbola_P_lies_on_fixed_line_l205_205358

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- The conditions provided in the problem
def focus := (-2 * Real.sqrt 5, 0)
def center := (0, 0)
def eccentricity : ℝ := Real.sqrt 5
def vertex1 := (-2, 0)
def vertex2 := (2, 0)
def passing_point := (-4, 0)

-- Finding the equation of the hyperbola
theorem equation_of_hyperbola : 
  ∀ (a b c : ℝ), 
  c = 2 * Real.sqrt 5 ∧ 
  eccentricity = Real.sqrt 5 ∧ 
  a = 2 ∧
  b = 4 → 
  hyperbola 2 4 :=
by
  sorry

-- Proving P lies on a fixed line
theorem P_lies_on_fixed_line :
  ∀ (m : ℝ), 
  let line : ℝ → ℝ := λ y, m * y - 4 in
  let intersection_M := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let intersection_N := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let P := (*) -- Intersection point of lines MA1 and NA2 -- (*) 
  ∃ (P : ℝ × ℝ), P.fst = -1 :=
by
  sorry

end equation_of_hyperbola_P_lies_on_fixed_line_l205_205358


namespace least_pos_int_div_by_four_distinct_primes_l205_205164

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205164


namespace least_pos_int_div_by_four_distinct_primes_l205_205168

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205168


namespace sum_of_squared_distances_l205_205564

variables (A B P : ℝ × ℝ)
variables (m : ℝ)

-- Define the fixed points A and B
def point_A := (0 : ℝ, 0 : ℝ)
def point_B := (1 : ℝ, -3 : ℝ)

-- Define the line equations
def line1 (m : ℝ) (P : ℝ × ℝ) := P.1 - m * P.2 = 0
def line2 (m : ℝ) (P : ℝ × ℝ) := m * P.1 + P.2 - m + 3 = 0

-- Prove the sum of squared distances |PA|^2 + |PB|^2
theorem sum_of_squared_distances (h1 : line1 m P) (h2 : line2 m P) (h_perpendicular : 1 + m^2 = 0) :
  (P.1 - point_A.1)^2 + (P.2 - point_A.2)^2 + (P.1 - point_B.1)^2 + (P.2 - point_B.2)^2 = 10 :=
sorry

end sum_of_squared_distances_l205_205564


namespace fixed_point_line_parabola_l205_205452

noncomputable theory
open_locale classical

variables {A B : Type} [real_field A] [A : real_field B]

theorem fixed_point_line_parabola
    (l : B → B) -- line equation in parametric form
    (h1 : ∃ (x y : B), (y = l x ∧ y^2 = 4 * x) ∧ (y = l x ∧ y^2 = 4 * x))
    (h2 : let O := (0 : B, 0 : B),
              A := (x1 : B, l x1),
              B := (x2 : B, l x2)
          in (x1 * x2 + (l x1) * (l x2)) = -4) :
    ∃ (p : B × B), p = (2, 0) :=
by {
  sorry
}

end fixed_point_line_parabola_l205_205452


namespace trig_identity_problem_l205_205326

theorem trig_identity_problem {α : ℝ} (h : Real.tan α = 3) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end trig_identity_problem_l205_205326


namespace problem_bounds_l205_205512

theorem problem_bounds :
  ∀ (A_0 B_0 C_0 A_1 B_1 C_1 A_2 B_2 C_2 A_3 B_3 C_3 : Point),
    (A_0B_0 + B_0C_0 + C_0A_0 = 1) →
    (A_1B_1 = A_0B_0) →
    (B_1C_1 = B_0C_0) →
    (A_2 = A_1 ∧ B_2 = B_1 ∧ C_2 = C_1 ∨
     A_2 = A_1 ∧ B_2 = C_1 ∧ C_2 = B_1 ∨
     A_2 = B_1 ∧ B_2 = A_1 ∧ C_2 = C_1 ∨
     A_2 = B_1 ∧ B_2 = C_1 ∧ C_2 = A_1 ∨
     A_2 = C_1 ∧ B_2 = A_1 ∧ C_2 = B_1 ∨
     A_2 = C_1 ∧ B_2 = B_1 ∧ C_2 = A_1) →
    (A_3B_3 = A_2B_2) →
    (B_3C_3 = B_2C_2) →
    (A_3B_3 + B_3C_3 + C_3A_3) ≥ 1 / 3 ∧ 
    (A_3B_3 + B_3C_3 + C_3A_3) ≤ 3 :=
by
  -- Proof goes here
  sorry

end problem_bounds_l205_205512


namespace equation_of_hyperbola_point_P_fixed_line_l205_205366

-- Definitions based on conditions
def center_origin := (0, 0)
def left_focus := (-2 * real.sqrt 5, 0)
def eccentricity := real.sqrt 5
def left_vertex := (-2, 0)
def right_vertex := (2, 0)
def point_passed_through_line := (-4, 0)

-- The equation of the hyperbola that satisfies the given conditions.
theorem equation_of_hyperbola :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ 
    (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ 
      (2 * real.sqrt 5)^2 = a^2 + b^2) :=
by
  use [2, 4]
  split
  exact rfl
  split
  exact rfl
  sorry

-- Proving P lies on the fixed line x = -1 for given lines and vertices conditions
theorem point_P_fixed_line :
  ∀ m : ℝ, ∀ y1 y2 x1 x2 : ℝ, 
    x1 = m * y1 - 4 ∧ x2 = m * y2 - 4 ∧ 
    (y1 + y2 = (32 * m) / ((4 * m^2) - 1)) ∧
    (y1 * y2 = 48 / ((4 * m^2) - 1)) ∧ 
    (x1 + 2) ≠ 0 ∧ 
    (x2 - 2) ≠ 0 → 
    (P : ℝ × ℝ) ((P.1, P.2) = (-1, P.2) ∧ P.2 = (y1 * (x + 2) / x1 ≠ -2 ∧ y2 * (x - 2) / x2 ≠ 2)) := 
by
  intros
  sorry

end equation_of_hyperbola_point_P_fixed_line_l205_205366


namespace brad_siblings_product_l205_205443

theorem brad_siblings_product (S B : ℕ) (hS : S = 5) (hB : B = 7) : S * B = 35 :=
by
  have : S = 5 := hS
  have : B = 7 := hB
  sorry

end brad_siblings_product_l205_205443


namespace star_m_l205_205490

def star (x : ℕ) : ℕ := x.digits.sum
def S : Finset ℕ := { n | star n = 16 ∧ n < 10^8 }.toFinset
def m : ℕ := S.card

theorem star_m : star m = 21 := by
  sorry

end star_m_l205_205490


namespace debby_initial_candy_l205_205721

theorem debby_initial_candy :
  ∃ (C : ℕ), (C - 9 = 3) ∧ (C = 12) :=
by
  have h1 : ∀ C : ℕ, C - 9 = 3 → C = 12 := sorry
  exact ⟨12, by { split, norm_num, sorry }⟩

end debby_initial_candy_l205_205721


namespace least_pos_int_div_by_four_distinct_primes_l205_205161

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205161


namespace num_of_permutations_2021_l205_205749

theorem num_of_permutations_2021 : 
  {l : List ℕ // l.length = 4 ∧ l.nodup ∧ l.permutations.count (λ p, p.head ≠ 0) = 9} :=
sorry

end num_of_permutations_2021_l205_205749


namespace impossibility_of_even_sum_1x2_vertical_rectangles_l205_205466

theorem impossibility_of_even_sum_1x2_vertical_rectangles :
  ¬ ∃ (f : Fin 6 → Fin 7 → ℕ), (∀ i j, 1 ≤ f i j ∧ f i j ≤ 42) ∧
  (∃ g : set (Fin 6 × Fin 7) × (Fin 6 × Fin 7), ∀ (i : Fin 5) (j : Fin 7),
  f i j + f i.succ j % 2 = 0) := sorry

end impossibility_of_even_sum_1x2_vertical_rectangles_l205_205466


namespace length_DE_l205_205547

theorem length_DE (AB : ℝ) (h_base : AB = 15) (DE_parallel : ∀ x y z : Triangle, True) (area_ratio : ℝ) (h_area_ratio : area_ratio = 0.25) : 
  ∃ DE : ℝ, DE = 7.5 :=
by
  sorry

end length_DE_l205_205547


namespace correct_statements_B_and_D_l205_205269

noncomputable def f1 (x : ℝ) : ℝ := |x| / x

def g (x : ℝ) : ℝ :=
if x ≥ 0 then 1 else -1

def statementB (f : ℝ → ℝ) : Prop :=
(∃ y, f 1 = y) → ∀ y, (f 1 = y →  ∃! x, x = 1)

noncomputable def f2 (x : ℝ) : ℝ := x^2 + 2 + 1 / (x^2 + 2)

def statementC : Prop :=
¬ ∃ (x : ℝ), f2 x = 2

noncomputable def f3 (x : ℝ) : ℝ := |x - 1| - |x|

def statementD : Prop :=
f3 (f3 (1 / 2)) = 1

theorem correct_statements_B_and_D :
  statementB f ∧ statementD :=
begin
  split,
  -- statement B: Assuming the function is arbitrary,
  -- proof: see problem analysis, skipping with sorry.
  sorry, -- Leaving out the proof for brevity.
  -- statement D: proof: see problem analysis, skipping with sorry.
  sorry  -- Leaving out the proof for brevity.
end

end correct_statements_B_and_D_l205_205269


namespace optimal_solution_for_z_is_1_1_l205_205884

def x := 1
def y := 1
def z (x y : ℝ) := 2 * x + y

theorem optimal_solution_for_z_is_1_1 :
  ∀ (x y : ℝ), z x y ≥ z 1 1 := 
by
  simp [z]
  sorry

end optimal_solution_for_z_is_1_1_l205_205884


namespace alberto_winning_strategy_l205_205693

theorem alberto_winning_strategy (n1 : ℕ) (even_piles : List ℕ) :
  (n1 % 2 = 1 ∨ (even_piles.foldl (λ acc k, if k % 2 = 0 then acc + 1 else acc) 0) % 2 = 1) ↔ 
  ∃ strategy : string, (strategy = "winning" ∧ strategy = "alberto") :=
sorry

end alberto_winning_strategy_l205_205693


namespace rhombus_perimeter_l205_205038

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 40 :=
by
  sorry

end rhombus_perimeter_l205_205038


namespace area_of_DOE_l205_205810

-- Conditions
variables (A B C D E O : Type)
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited O]

-- Definitions related to areas and ratios
variables (area_ABC : ℝ)
variables (area_DOE : ℝ)
variables (r1 : ℝ := 1 / 4) -- ratio for DO:OB
variables (r2 : ℝ := 4 / 9) -- ratio for EO:OA

-- Given conditions
axiom area_ABC_is_one : area_ABC = 1
axiom DO_OB_ratio : r1 = 1 / 4
axiom EO_OA_ratio : r2 = 4 / 9

-- The resulting area of triangle DOE
axiom area_DOE_is : area_DOE = 1 / 39

-- Proof statement
theorem area_of_DOE (habc : area_ABC = 1)
  (hDO : r1 = 1 / 4)
  (hEO : r2 = 4 / 9) :
  area_DOE = 11 / 135 :=
sorry

end area_of_DOE_l205_205810


namespace f_pi_six_eq_sqrt3_l205_205797

-- Define the initial function f(x)
def f (x φ : ℝ) : ℝ := 2 * sin (x + φ)

-- Define g(x) as the shifted function
def g (x : ℝ) (φ : ℝ) : ℝ := 2 * sin (x + (π / 3) + φ)

-- State and prove the equivalent proof problem
theorem f_pi_six_eq_sqrt3 (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π / 2) (h3 : ∀ x, g x φ = g (-x) φ) : 
  f (π / 6) φ = √3 :=
by
  -- Use Derived conditions
  have hφ : φ = π / 6 := 
    by -- Proof of this derived condition goes here (skipping for now)
      exact sorry
  rw [hφ]
  -- Prove the final expression
  show 2 * sin (π / 3) = √3,
    by -- Proof of this simplification goes here (skipping for now)
      exact sorry

end f_pi_six_eq_sqrt3_l205_205797


namespace dog_ate_cost_is_six_l205_205477

-- Definitions for the costs
def flour_cost : ℝ := 4
def sugar_cost : ℝ := 2
def butter_cost : ℝ := 2.5
def eggs_cost : ℝ := 0.5

-- Total cost calculation
def total_cost := flour_cost + sugar_cost + butter_cost + eggs_cost

-- Initial slices and remaining slices
def initial_slices : ℕ := 6
def eaten_slices : ℕ := 2
def remaining_slices := initial_slices - eaten_slices

-- The cost calculation of the amount the dog ate
def dog_ate_cost := (remaining_slices / initial_slices) * total_cost

-- Proof statement
theorem dog_ate_cost_is_six : dog_ate_cost = 6 :=
by
  sorry

end dog_ate_cost_is_six_l205_205477


namespace sum_of_segments_is_198_sqrt_41_l205_205911

-- Condition definitions
def side_AB := 5
def side_CB := 4
def num_segments := 200
def AC := Real.sqrt (side_AB^2 + side_CB^2)

-- Main theorem statement
theorem sum_of_segments_is_198_sqrt_41 (k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ num_segments - 1) : 
  2 * (Real.sqrt 41) * (Finset.sum (Finset.range num_segments) (λ k, (num_segments - k) / num_segments : ℕ → ℝ)) - AC = 198 * Real.sqrt 41 := by
  sorry

end sum_of_segments_is_198_sqrt_41_l205_205911


namespace olga_total_fish_l205_205509

theorem olga_total_fish :
  ∃ (F : ℕ),
    let yellow := 12,
        blue := yellow / 2,
        green := 2 * yellow,
        purple := blue / 2,
        pink := green / 3,
        orange := 0.20 * F,
        grey := 0.10 * F,
        known_fish := yellow + blue + green + purple + pink,
        percentage_fish := (orange + grey) in
    F = 76 := by
sorry

end olga_total_fish_l205_205509


namespace can_combine_with_sqrt3_l205_205265

theorem can_combine_with_sqrt3 
  (A : ℝ) (hA : A = real.sqrt 30)
  (B : ℝ) (hB : B = real.sqrt (1 / 2))
  (C : ℝ) (hC : C = real.sqrt 8)
  (D : ℝ) (hD : D = real.sqrt 27) :
  (∃ k : ℝ, D = k * real.sqrt 3) ∧ ¬ (∃ k : ℝ, A = k * real.sqrt 3) ∧ ¬ (∃ k : ℝ, B = k * real.sqrt 3) ∧ ¬ (∃ k : ℝ, C = k * real.sqrt 3) :=
by sorry

end can_combine_with_sqrt3_l205_205265


namespace smallest_non_unit_digit_multiple_of_five_l205_205974

theorem smallest_non_unit_digit_multiple_of_five :
  ∀ (d : ℕ), ((d = 0) ∨ (d = 5)) → (d ≠ 1 ∧ d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 6 ∧ d ≠ 7 ∧ d ≠ 8 ∧ d ≠ 9) :=
by {
  sorry
}

end smallest_non_unit_digit_multiple_of_five_l205_205974


namespace find_m_of_equation_with_imaginary_roots_l205_205781

theorem find_m_of_equation_with_imaginary_roots (m : ℝ) :
  (∀ (α β : ℂ), 3 * α^2 - 6 * (m - 1) * α + (m^2 + 1) = 0 ∧
                3 * β^2 - 6 * (m - 1) * β + (m^2 + 1) = 0 ∧
                Im(α) ≠ 0 ∧ Im(β) ≠ 0 ∧
                α + β = 2 * (α.re:ℂ) ∧ abs α + abs β = 2) →
  m = real.sqrt 2 :=
by
  sorry

end find_m_of_equation_with_imaginary_roots_l205_205781


namespace length_exceeds_breadth_by_50_l205_205562

theorem length_exceeds_breadth_by_50 :
  ∀ (b : ℝ) (cost_per_meter total_cost l : ℝ),
    cost_per_meter = 26.50 →
    total_cost = 5300 →
    l = 75 →
    total_cost = (2 * l + 2 * b) * cost_per_meter →
    l - b = 50 :=
begin
  -- proof goes here
  sorry
end

end length_exceeds_breadth_by_50_l205_205562


namespace find_ab_solutions_l205_205310

theorem find_ab_solutions (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h1 : (a + 1) ∣ (a ^ 3 * b - 1))
  (h2 : (b - 1) ∣ (b ^ 3 * a + 1)) : 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) :=
sorry

end find_ab_solutions_l205_205310


namespace part_a_l205_205652

theorem part_a (S : Finset (Fin 6)) (L : Finset (Finset (Fin 6))) 
  (hL : ∀ l ∈ L, l.card = 3) (hS : S.card = 6) (h1 : ∀ l ∈ L, l ⊆ S)
  (h2 : ∀ t : Finset (Fin 6), t.card = 3 → ∃ l ∈ L, t ⊆ l) :
  ∃ f : Fin 6 → Fin 6, 
    function.bijective f ∧ (∀ l ∈ L, ∃ l' ∈ L, ∀ x ∈ l, f x ∈ l') :=
by
  sorry

end part_a_l205_205652


namespace price_reduction_for_1920_profit_maximum_profit_calculation_l205_205669

-- Definitions based on given conditions
def cost_price : ℝ := 12
def base_price : ℝ := 20
def base_quantity_sold : ℝ := 240
def increment_per_dollar : ℝ := 40

-- Profit function
def profit (x : ℝ) : ℝ := (base_price - cost_price - x) * (base_quantity_sold + increment_per_dollar * x)

-- Prove price reduction for $1920 profit per day
theorem price_reduction_for_1920_profit : ∃ x : ℝ, profit x = 1920 ∧ x = 8 := by
  sorry

-- Prove maximum profit calculation
theorem maximum_profit_calculation : ∃ x y : ℝ, x = 4 ∧ y = 2560 ∧ ∀ z, profit z ≤ y := by
  sorry

end price_reduction_for_1920_profit_maximum_profit_calculation_l205_205669


namespace trigonometric_expression_value_l205_205341

variable {α : ℝ}

-- Definition: α is an angle in the second quadrant
def in_second_quadrant (α : ℝ) : Prop := π/2 < α ∧ α < π

-- The trigonometric identity sin^2 α + cos^2 α = 1
def trig_identity (α : ℝ) : Prop := sin α ^ 2 + cos α ^ 2 = 1

-- The assertion that we need to prove
theorem trigonometric_expression_value (h1 : in_second_quadrant α) (h2 : trig_identity α) :
  (2 * sin α) / (sqrt (1 - cos α ^ 2)) + (sqrt (1 - sin α ^ 2)) / cos α = 1 :=
sorry

end trigonometric_expression_value_l205_205341


namespace tilly_star_ratio_l205_205064

theorem tilly_star_ratio :
  ∀ (x : ℕ), 120 + 120 * x = 840 → (120 * x) / 120 = 6 :=
by
  intros x h
  rw [←nat.add_sub_assoc, nat.add_sub_cancel] at h
  sorry

end tilly_star_ratio_l205_205064


namespace train_passes_platform_in_5t_l205_205254

variables (l t : ℝ)
-- Condition: A train of length l passes a pole in t seconds.
-- Question: What is the multiple of t seconds it takes for the same train traveling at the same velocity to pass a platform of length 4l?
theorem train_passes_platform_in_5t (l t : ℝ) : 
  let v := l / t in -- train's velocity
  ∃ t' : ℝ, t' = 5 * t :=
by
  -- Train covering its own length and platform length
  let distance := l + 4 * l
  have h_distance : distance = 5 * l := by norm_num
  have h_time : distance / v = 5 * t := by
    rw [←h_distance, div_eq_mul_one_div, mul_div_cancel' _ (ne_of_gt (zero_lt l))]
    simp
  exact ⟨5 * t, h_time⟩

end train_passes_platform_in_5t_l205_205254


namespace points_equidistant_from_line_AB_l205_205282

open EuclideanGeometry

theorem points_equidistant_from_line_AB 
  (ω₁ ω₂ : Circle)
  (A B K₁ K₂ L₁ L₂ : Point)
  (h1 : Intersect ω₁ ω₂ A B)
  (h2 : OnCircle K₁ ω₁)
  (h3 : TangentAt K₁ A ω₂)
  (h4 : OnCircle K₂ ω₂)
  (h5 : TangentAt K₂ A ω₁)
  (circumcircle_KBK : Circle)
  (h6 : CircumscribedCircle K₁ B K₂ circumcircle_KBK)
  (h7 : IntersectLineCircle (LineThrough A K₁) circumcircle_KBK L₁₍L₁₎₎)
  (h8 : IntersectLineCircle (LineThrough A K₂) circumcircle_KBK L₁₍L₂₎₎) : 
  EquidistantFromLine L₁ L₂ (LineThrough A B) :=
sorry

end points_equidistant_from_line_AB_l205_205282


namespace max_tables_chairs_l205_205212

def budget := 2000
def cost_table := 50
def cost_chair := 20
def tables := 25
def chairs := 37

theorem max_tables_chairs (tables chairs : ℕ) (budget cost_table cost_chair : ℕ)
    (H1 : budget = 2000) 
    (H2 : cost_table = 50) 
    (H3 : cost_chair = 20)
    (H4 : tables = 25) 
    (H5 : chairs = 37) : 
    (tables * cost_table + chairs * cost_chair ≤ budget) ∧ 
    (chairs ≥ tables) ∧ 
    (chairs ≤ 1.5 * tables) ∧ 
    (∀ t c, t * cost_table + c * cost_chair ≤ budget → 
             c ≥ t → 
             c ≤ 3 / 2 * t → 
             t + c ≤ tables + chairs) := by
  sorry

end max_tables_chairs_l205_205212


namespace sphere_intersection_radius_l205_205249

theorem sphere_intersection_radius (r : ℝ) :
  (let center := (3 : ℝ, 5, -8),
       xy_center := (3, 5, 0),
       yz_center := (0, 5, -8),
       xy_radius := 2,
       sphere_radius := real.sqrt (2^2 + (xy_center.3 - center.3)^2) in
   (real.sqrt (r^2 + (yz_center.1 - center.1)^2) = sphere_radius)) :=
by
  let center := (3 : ℝ, 5, -8),
      xy_center := (3, 5, 0),
      yz_center := (0, 5, -8),
      xy_radius := 2,
      sphere_radius := real.sqrt (2^2 + (xy_center.3 - center.3)^2)
  use (real.sqrt 59)
  sorry

end sphere_intersection_radius_l205_205249


namespace train_speed_l205_205685

theorem train_speed (train_length bridge_length : ℕ) (cross_time : ℕ) 
  (h_train_length : train_length = 170)
  (h_bridge_length : bridge_length = 205)
  (h_cross_time : cross_time = 30) :
  let total_distance := train_length + bridge_length in
  let speed_mps := total_distance / cross_time in
  let speed_kmph := speed_mps * 36 / 10 in
  speed_kmph = 45 :=
by 
  have h1 : total_distance = 375 := by 
    rw [h_train_length, h_bridge_length]
    rfl
  have h_speed_mps : speed_mps = 12.5 :=
    show speed_mps = 375 / 30 from by 
      rw [h1, h_cross_time]
      norm_num
  have h_speed_kmph : speed_kmph = 12.5 * 3.6 := by
    rw [h_speed_mps]
    norm_num
  exact h_speed_kmph

end train_speed_l205_205685


namespace max_value_fraction_l205_205404

-- Definitions based on the given problem conditions
def f (a b c x : ℝ) := Real.sqrt (a * x^2 + b * x + c)

-- The theorem statement
theorem max_value_fraction (a b c : ℝ) (h1 : a < b) (h2 : 0 < a) (h3 : b^2 - 4 * a * c ≤ 0) (h4 : c ≥ b^2 / (4 * a)) :
  ∃ x, x = (b - a) / (a + 2 * b + 4 * c) ∧ x ≤ 1 / 8 :=
sorry

end max_value_fraction_l205_205404


namespace find_y_l205_205826

-- Define the known values and the proportion relation
variable (x y : ℝ)
variable (h1 : 0.75 / x = y / 7)
variable (h2 : x = 1.05)

theorem find_y : y = 5 :=
by
sorry

end find_y_l205_205826


namespace part_I_part_II_l205_205409

-- Definitions based on conditions
def parabola_def (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def is_point_on_parabola (p : ℝ) (x y : ℝ) : Prop := parabola_def p x y

-- The given conditions
variables {p : ℝ} {m : ℝ}
axiom positive_p : p > 0
axiom point_on_parabola : is_point_on_parabola p 1 m
axiom distance_to_focus : sqrt ((1 - (p/2))^2 + m^2) = 2

-- Proof Statements
theorem part_I : parabola_def 2 = (λ x y, y^2 = 4 * x) ∧ (1, 0) = ⟨1, 0⟩ :=
sorry

theorem part_II : ∃ k : ℝ, k ≠ 0 ∧ ((λ x y, y = k * (x - 1)) = y = x - 1 ∨ y = -x + 1) :=
sorry

end part_I_part_II_l205_205409


namespace eccentricity_of_hyperbola_l205_205336

variables {a b : ℝ} (h_positive_a : a > 0) (h_positive_b : b > 0)

def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def line_slope := (y2 y1 x2 x1 : ℝ) → (y2 - y1) / (x2 - x1) = 3

def midpoint_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 + x2 = 12) ∧ (y1 + y2 = 4)

theorem eccentricity_of_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x1 y1 x2 y2 : ℝ), hyperbola_equation x1 y1 ∧ hyperbola_equation x2 y2 ∧ midpoint_condition x1 y1 x2 y2 ∧ line_slope y1 y2 x1 x2 → 
  (sqrt (1 + b^2 / a^2) = sqrt 2) :=
sorry

end eccentricity_of_hyperbola_l205_205336


namespace license_plates_at_least_two_same_l205_205296

def num_letters : ℕ := 26
def num_digits : ℕ := 10

def total_license_plates : ℕ := num_letters ^ 2 * num_digits ^ 2

def distinct_license_plates : ℕ :=
  num_letters * (num_letters - 1) * num_digits * (num_digits - 1)

def license_plates_with_repeats : ℕ :=
  total_license_plates - distinct_license_plates

theorem license_plates_at_least_two_same :
  license_plates_with_repeats = 9100 := by
  rw [total_license_plates, distinct_license_plates]
  sorry

end license_plates_at_least_two_same_l205_205296


namespace concurrency_AM_EF_ND_l205_205004

theorem concurrency_AM_EF_ND :
  ∃ (A B C D E F M N : Type)
  [Collinear A B C D]
  [Circle (center := midpoint A C) (radius := dist A (midpoint A C))]
  (k1 : Set A) (k2 : Set B)
  [Inter (circle k1) (circle k2) E F]
  [Tangent (k1) (k2) M N],
  Concurrent (Line A M) (Line E F) (Line N D) :=
sorry

end concurrency_AM_EF_ND_l205_205004


namespace range_of_m_l205_205772

open Set

variable {α : Type}

noncomputable def A (m : ℝ) : Set ℝ := {x | m+1 ≤ x ∧ x ≤ 2*m-1}
noncomputable def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

theorem range_of_m (m : ℝ) (hA : A m ⊆ B) (hA_nonempty : A m ≠ ∅) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end range_of_m_l205_205772


namespace shelby_gold_stars_l205_205525

theorem shelby_gold_stars (stars_yesterday stars_today : ℕ) (h_yesterday : stars_yesterday = 4)
    (h_today : stars_today = 3) : stars_yesterday + stars_today = 7 := by
  -- Define the conditions given in the problem.
  rw [h_yesterday, h_today]
  -- Add the two amounts (4 + 3).
  -- The expected result should be 7.
  exact rfl

end shelby_gold_stars_l205_205525


namespace simplify_sqrt_expr_l205_205923

/-- Simplify the given radical expression and prove its equivalence to the expected result. -/
theorem simplify_sqrt_expr :
  (Real.sqrt (5 * 3) * Real.sqrt ((3 ^ 4) * (5 ^ 2)) = 225 * Real.sqrt 15) := 
by
  sorry

end simplify_sqrt_expr_l205_205923


namespace plugs_added_l205_205817

theorem plugs_added : 
  ∀ (pairs_mittens pairs_plugs pairs_plugs_now : ℕ),
  pairs_mittens = 150 →
  pairs_plugs = pairs_mittens - 20 →
  pairs_plugs_now = 400 →
  ((pairs_plugs_now - (pairs_plugs * 2)) / 2) = 70 := 
by {
  intros pairs_mittens pairs_plugs pairs_plugs_now h1 h2 h3,
  -- rest of the proof here
  sorry
}

end plugs_added_l205_205817


namespace amount_A_l205_205012

theorem amount_A (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : A = 62 := by
  sorry

end amount_A_l205_205012


namespace total_sample_needed_l205_205226

-- Given constants
def elementary_students : ℕ := 270
def junior_high_students : ℕ := 360
def senior_high_students : ℕ := 300
def junior_high_sample : ℕ := 12

-- Calculate the total number of students in the school
def total_students : ℕ := elementary_students + junior_high_students + senior_high_students

-- Define the sampling ratio based on junior high section
def sampling_ratio : ℚ := junior_high_sample / junior_high_students

-- Apply the sampling ratio to the total number of students to get the total sample size
def total_sample : ℚ := sampling_ratio * total_students

-- Prove that the total number of students that need to be sampled is 31
theorem total_sample_needed : total_sample = 31 := sorry

end total_sample_needed_l205_205226


namespace smallest_solution_l205_205209

noncomputable def expr (x : ℝ) : ℝ := 
  (-Real.log (105 + 2 * x * Real.sqrt (x + 19)) / Real.log 2)^3 + 
  Real.abs ((Real.log (105 + 2 * x * Real.sqrt (x + 19)) / Real.log 2) - 4 * (Real.log (x^2 + x + 3) / Real.log 2)) / 
  (9 * (Real.log (76 + 2 * x * Real.sqrt (x + 19)) / Real.log 5) - 4 * (Real.log (105 + 2 * x * Real.sqrt (x + 19)) / Real.log 2))

theorem smallest_solution :
  ∀ x : ℝ, expr x ≥ 0 → x ≥ (⟨ (-21 + Real.sqrt 33) / 2 , sorry⟩) := 
sorry

end smallest_solution_l205_205209


namespace unique_four_digit_numbers_from_2021_l205_205752

theorem unique_four_digit_numbers_from_2021 : 
  let digits := [2, 0, 2, 1] in 
  let unique_permutations := { p | multiset.erase_dup (multiset.of_list (list.permutations digits)).to_finset.count = 9 } in
  unique_permutations.to_list.count = 9 :=
sorry

end unique_four_digit_numbers_from_2021_l205_205752


namespace tiling_problem_solution_l205_205494

def C_n (n : ℕ) : Set (ℕ × ℕ) := 
  {(i, j) | i < n ∧ j < n ∧ i ≠ j}

def L_tiles : Set (Set (ℕ × ℕ)) :=
  { {(0, 0), (0, 1), (1, 0)}, {(0, 0), (0, -1), (-1, 0)}, {(0, 0), (0, 1), (-1, 0)}, {(0, 0), (0, -1), (1, 0)} }

def can_tile_with_L (n : ℕ) : Prop :=
  ∃ (tiles : Set (Set (ℕ × ℕ))), 
    (∀ t ∈ tiles, t ∈ L_tiles) ∧
    (⋃ t ∈ tiles, t) = C_n n

theorem tiling_problem_solution :
  ∀ (n : ℕ), n ≥ 2 → 
    (can_tile_with_L n ↔ (n = 3 ∨ ∃ (k : ℕ) (i ∈ {1, 3, 4, 6}), n = 6*k + i)) := 
sorry

end tiling_problem_solution_l205_205494


namespace probability_product_divisible_by_3_greater_than_15_l205_205015

-- Define the balls
def balls : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define a function to check if a number is divisible by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define a function to get all possible outcomes of two draws
def possible_outcomes (lst : List ℕ) : List (ℕ × ℕ) :=
  List.bind lst (λ x => List.map (λ y => (x, y)) lst)

-- Filter the outcomes where the product is divisible by 3 and greater than 15
def valid_outcomes (lst : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  lst |>.filter (λ ⟨x, y⟩ => divisible_by_3 (x * y) ∧ x * y > 15)

-- Total number of possible outcomes
def total_outcomes : ℕ := List.length (possible_outcomes balls)

-- Number of valid outcomes
def favorable_outcomes : ℕ := List.length (valid_outcomes (possible_outcomes balls))

-- Probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Theorem to prove the probability
theorem probability_product_divisible_by_3_greater_than_15 :
  probability = 5 / 36 :=
by
  -- Insert proof here
  sorry

end probability_product_divisible_by_3_greater_than_15_l205_205015


namespace angle_between_vectors_l205_205833

variables {a b : EuclideanSpace ℝ (Fin 2)}

def magnitude (v : EuclideanSpace ℝ (Fin 2)) : ℝ := Real.sqrt (v.dot_product v)

def dot_product (u v : EuclideanSpace ℝ (Fin 2)) : ℝ := u.1 * v.1 + u.2 * v.2

noncomputable def angle_between (u v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  Real.arccos ((u.dot_product v) / (magnitude u * magnitude v))

theorem angle_between_vectors :
  magnitude a = 2 →
  magnitude b = 4 →
  dot_product a b = 4 →
  angle_between a b = π / 3 :=
by
  intros h1 h2 h3
  sorry

end angle_between_vectors_l205_205833


namespace induction_inequality_l205_205519

-- Function definitions and lemmas
def summation (n : ℕ) : ℝ :=
(1 + ∑ i in finset.range (n - 1), (1 / (i + 2)^2))

noncomputable def RHS (n : ℕ) : ℝ := 2 - 1 / n

-- The main theorem statement
theorem induction_inequality (n : ℕ) (h : 2 ≤ n) : summation n < RHS n := by
  sorry

end induction_inequality_l205_205519


namespace hyperbola_and_fixed_line_proof_l205_205385

-- Definitions based on conditions
def center_at_origin (C : Type) [Hyperbola C] : Prop :=
  C.center = (0, 0)

def left_focus (C : Type) [Hyperbola C] : Prop :=
  C.focus1 = (-2 * Real.sqrt 5, 0)

def eccentricity (C : Type) [Hyperbola C] : Prop :=
  C.eccentricity = Real.sqrt 5

def vertices (C : Type) [Hyperbola C] : Prop :=
  C.vertex1 = (-2, 0) ∧ C.vertex2 = (2, 0)

def passing_line_intersections (C : Type) (M N : Type) [Hyperbola C] : Prop :=
  ∃ l : Line, l ⨯ (-4, 0) ∧ l.intersects_hyperbola_left_branch C = {M, N}

def intersection (M N A1 A2 P : Type) [Hyperbola C] : Prop :=
  ∃ A1 A2, A1 = (-2, 0) ∧ A2 = (2, 0) ∧ (MA1.M ∩ NA2.N = P)

-- Proof problem statement
theorem hyperbola_and_fixed_line_proof (C : Type) [Hyperbola C] (M N P : Type)
  (h1 : center_at_origin C) (h2 : left_focus C) (h3 : eccentricity C)
  (h4 : passing_line_intersections C M N) (h5 : vertices C) (h6 : intersection M N C.vertex1 C.vertex2 P) : 
  (C.equation = (∀ x y, C.vertical_transverse_axis x y (x^2 / 4 - y^2 / 16)) ∧ 
   ∀ (x : ℝ), P.x = -1) :=
by sorry

end hyperbola_and_fixed_line_proof_l205_205385


namespace find_k_m_n_l205_205315

theorem find_k_m_n
  (k m n : ℕ)
  (h_rel_prime : Nat.coprime m n)
  (h1 : (1 + 2 * Real.sin t) * (1 + 2 * Real.cos t) = 9 / 4)
  (h2 : (1 - 2 * Real.sin t) * (1 - 2 * Real.cos t) = m / n - Real.sqrt k)
  (h_pos_m : 0 < m)
  (h_pos_n : 0 < n)
  (h_pos_k : 0 < k) :
  k = 11 ∧ m = 27 ∧ n = 4 :=
sorry

end find_k_m_n_l205_205315


namespace length_DE_l205_205551

-- Definitions for conditions in the problem
variables (AB : ℝ) (DE : ℝ) (areaABC : ℝ)

-- Condition: AB is 15 cm
axiom length_AB : AB = 15

-- Condition: The area of triangle projected below the base is 25% of the area of triangle ABC
axiom area_ratio_condition : (1 / 4) * areaABC = (1 / 2)^2 * areaABC

-- The problem statement translated to Lean proof
theorem length_DE : DE = 7.5 :=
by
  -- Definitions and conditions
  have h1 : AB = 15 := length_AB
  have h2 : (1 / 2)^2 = 1 / 4 := by ring
  calc
    DE = (0.5) * AB :  sorry  -- proportional relationship since triangles are similar
    ... = 0.5 * 15   :  by rw [h1]
    ... = 7.5       :  by norm_num

end length_DE_l205_205551


namespace remaining_money_left_l205_205668

theorem remaining_money_left (salary : ℝ) (food_factor rent_factor clothes_factor remaining_amount : ℝ) :
  salary = 8123.08 →
  food_factor = 1/3 →
  rent_factor = 1/4 →
  clothes_factor = 1/5 →
  remaining_amount = 1759.00 →
  remaining_amount ≈ salary - (food_factor * salary + rent_factor * salary + clothes_factor * salary) :=
by sorry

end remaining_money_left_l205_205668


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205093

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205093


namespace value_of_expression_l205_205624

theorem value_of_expression :
  (43 + 15)^2 - (43^2 + 15^2) = 2 * 43 * 15 :=
by
  sorry

end value_of_expression_l205_205624


namespace angle_between_height_and_generatrix_l205_205068

-- Define the conditions as assumptions
variables (D E F : Point) (α : Plane)
variables (DE DF : Line) (φ β : Real)
variable (γ : Real)

-- Assume necessary conditions
-----------------------------------------------------------------
-- 1. Identical cones with common vertex D positioned on opposite sides of plane α
-- 2. Cones touch the plane along generatrices DE and DF respectively
-- 3. Angle EDF = φ
-- 4. Angle between the line of intersection of base planes of cones and the plane α is β
-----------------------------------------------------------------

-- The theorem statement itself
theorem angle_between_height_and_generatrix :
  ∀ (D E F : Point) (α : Plane) (DE DF : Line), 
  ∠(DE, DF) = φ → 
  angle_between_lines_in_planes_planes_intersection_with_plane α = β → 
  (γ = arctan (sin (φ / 2) / cos (π / 180 * β / 9))) :=
sorry

end angle_between_height_and_generatrix_l205_205068


namespace horner_multiplications_additions_l205_205280

-- Define the polynomial
def f (x : ℤ) : ℤ := x^7 + 2 * x^5 + 3 * x^4 + 4 * x^3 + 5 * x^2 + 6 * x + 7

-- Define the number of multiplications and additions required by Horner's method
def horner_method_mults (n : ℕ) : ℕ := n
def horner_method_adds (n : ℕ) : ℕ := n - 1

-- Define the value of x
def x : ℤ := 3

-- Define the degree of the polynomial
def degree_of_polynomial : ℕ := 7

-- Define the statements for the proof
theorem horner_multiplications_additions :
  horner_method_mults degree_of_polynomial = 7 ∧
  horner_method_adds degree_of_polynomial = 6 :=
by
  sorry

end horner_multiplications_additions_l205_205280


namespace missing_number_eq_6_l205_205214

theorem missing_number_eq_6 (x : ℕ) : (x! - 4!) / 5! = 58 / 10 → x = 6 := by
  have four_factorial : 4! = 24 := by norm_num
  have five_factorial : 5! = 120 := by norm_num
  rw [four_factorial, five_factorial]
  sorry

end missing_number_eq_6_l205_205214


namespace min_value_of_f_l205_205187

noncomputable def f (x : ℝ) : ℝ := x^2 + 11 * x - 5

theorem min_value_of_f : (∀ x : ℝ, f(x) ≥ f(-11 / 2)) :=
by 
  sorry

end min_value_of_f_l205_205187


namespace slope_angle_of_line_l205_205055

theorem slope_angle_of_line (θ : ℝ) (h1: θ ∈ Set.Ico 0 180)
  (h2: tan (Real.radians θ) = (Real.sqrt 3) / 3) : θ = 30 :=
  sorry

end slope_angle_of_line_l205_205055


namespace candies_remaining_after_carlos_ate_l205_205634

theorem candies_remaining_after_carlos_ate(
  red : ℕ := 60,
  yellow : ℕ := 3 * red - 30,
  blue : ℕ := (2 * yellow) / 4,
  green : ℕ := 40,
  purple : ℕ := green / 3,
  silver : ℕ := 15,
  gold : ℕ := silver / 2,
  total_before_eaten : ℕ := red + yellow + blue + green + purple + silver + gold,
  yellow_eaten : ℕ := yellow,
  green_eaten : ℕ := 3 * green / 4,
  blue_eaten : ℕ := blue / 3,
  total_eaten : ℕ := yellow_eaten + green_eaten + blue_eaten
) : total_before_eaten - total_eaten = 155 := by
  sorry

end candies_remaining_after_carlos_ate_l205_205634


namespace average_sale_six_months_l205_205232

theorem average_sale_six_months :
  let sale1 := 2500
  let sale2 := 6500
  let sale3 := 9855
  let sale4 := 7230
  let sale5 := 7000
  let sale6 := 11915
  let total_sales := sale1 + sale2 + sale3 + sale4 + sale5 + sale6
  let num_months := 6
  (total_sales / num_months) = 7500 :=
by
  sorry

end average_sale_six_months_l205_205232


namespace projection_matrix_l205_205487

variables (x y z : ℝ)
def v : Matrix (Fin 3) (Fin 1) ℝ := ![![x], ![y], ![z]]
def n : Matrix (Fin 3) (Fin 1) ℝ := ![![2], ![-1], ![3]]
def Q : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![10 / 14, 2 / 14, -6 / 14],
  ![2 / 14, 13 / 14, 3 / 14],
  ![-6 / 14, 3 / 14, 5 / 14]
]

theorem projection_matrix (x y z : ℝ) :
  let v := (λ (x y z : ℝ), ![![x], ![y], ![z]]) x y z
  let n := ![![2], ![-1], ![3]]
  let Q := ![
    ![10 / 14, 2 / 14, -6 / 14],
    ![2 / 14, 13 / 14, 3 / 14],
    ![-6 / 14, 3 / 14, 5 / 14]
  ]
  Q.mul v = v - (Matrix.dotProduct v n / Matrix.dotProduct n n) * n := 
sorry

end projection_matrix_l205_205487


namespace routes_A_to_B_l205_205283

-- Define the cities
inductive City
| A | B | C | D | E
deriving DecidableEq, Inhabited

open City

-- Define the roads as pairs of cities
def roads : List (City × City) := [
  (A, B), (A, D), (A, E),
  (B, C), (B, D), (C, D),
  (D, E)
]

-- The main proof statement regarding the number of routes from A to B
theorem routes_A_to_B : 
  let use_each_road_once (path : List (City × City)) := 
      roads = path ∧ path.length = List.length roads in
  (∃ path : List (City × City), use_each_road_once path) → 
  (∀ path : List (City × City), use_each_road_once path → path.head = (A, _)) → 
  (∀ path : List (City × City), use_each_road_once path → path.last = (_, B)) → 
  (count_paths roads = 16) :=
sorry

end routes_A_to_B_l205_205283


namespace rate_of_current_l205_205957

theorem rate_of_current (c : ℝ) (v_boat : ℝ) (d : ℝ) (t : ℝ) 
  (h1 : v_boat = 20) (h2 : d = 5.2) (h3 : t = 0.2) :
  20 + c = d / t → c = 6 := 
by
  intro h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact eq_of_sub_eq_zero (sub_eq_zero.mpr h4)

sorry

end rate_of_current_l205_205957


namespace max_planes_from_points_l205_205972

theorem max_planes_from_points (points : ℕ) (h1 : points = 15)
  (h2 : ∀ p q r : ℕ, p ≠ q → p ≠ r → q ≠ r)
  (h3 : ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (nat.choose 15 3 = 455) :=
sorry

end max_planes_from_points_l205_205972


namespace smallest_positive_period_of_f_increasing_interval_of_f_l205_205314

noncomputable def f (x : ℝ) : ℝ := sin x - cos x - 1

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
  by
  use 2 * π
  sorry

theorem increasing_interval_of_f : ∀ k : ℤ,
  ∀ x : ℝ, 2 * k * π - π / 4 ≤ x ∧ x ≤ 2 * k * π + 3 * π / 4 → ∀ y : ℝ, x ≤ y → f x ≤ f y :=
  by
  sorry

end smallest_positive_period_of_f_increasing_interval_of_f_l205_205314


namespace fraction_numerator_exceeds_denominator_l205_205050

theorem fraction_numerator_exceeds_denominator (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 3) :
  4 * x + 5 > 10 - 3 * x ↔ (5 / 7) < x ∧ x ≤ 3 :=
by 
  sorry

end fraction_numerator_exceeds_denominator_l205_205050


namespace mailman_magazines_l205_205667

theorem mailman_magazines (total_mail junk_mail : ℕ) (h_total : total_mail = 11) (h_junk : junk_mail = 6) :
  total_mail - junk_mail = 5 :=
by {
  rw [h_total, h_junk],
  norm_num,
  exact dec_trivial,
}

end mailman_magazines_l205_205667


namespace racers_meet_at_start_again_l205_205026

-- We define the conditions as given
def RacingMagic_time := 60
def ChargingBull_time := 60 * 60 / 40 -- 90 seconds
def SwiftShadow_time := 80
def SpeedyStorm_time := 100

-- Prove the LCM of their lap times is 3600 seconds,
-- which is equivalent to 60 minutes.
theorem racers_meet_at_start_again :
  Nat.lcm (Nat.lcm (Nat.lcm RacingMagic_time ChargingBull_time) SwiftShadow_time) SpeedyStorm_time = 3600 ∧
  3600 / 60 = 60 := by
  sorry

end racers_meet_at_start_again_l205_205026


namespace equilibrium_mass_l205_205958

variable (l m2 S g : ℝ) (m1 : ℝ)

-- Given conditions
def length_of_rod : ℝ := 0.5 -- length l in meters
def mass_of_rod : ℝ := 2 -- mass m2 in kg
def distance_S : ℝ := 0.1 -- distance S in meters
def gravity : ℝ := 9.8 -- gravitational acceleration in m/s^2

-- Equivalence statement
theorem equilibrium_mass (h1 : l = length_of_rod)
                         (h2 : m2 = mass_of_rod)
                         (h3 : S = distance_S)
                         (h4 : g = gravity) :
  m1 = 10 := sorry

end equilibrium_mass_l205_205958


namespace hyperbola_center_origin_point_P_on_fixed_line_l205_205354

variables {x y : ℝ}

def hyperbola (h : (x^2 / 4) - (y^2 / 16) = 1) : Prop := true

theorem hyperbola_center_origin 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (left_focus : ∃ x y : ℝ, x = -2 * Real.sqrt 5 ∧ y = 0)
  (eccentricity_root5 : ∃ e : ℝ, e = Real.sqrt 5) :
  hyperbola ((x^2 / 4) - (y^2 / 16) = 1) := 
sorry

theorem point_P_on_fixed_line
  (line_through_point : ∃ x y : ℝ, x = -4 ∧ y = 0)
  (M_second_quadrant : ∃ Mx My : ℝ, Mx < 0 ∧ My > 0)
  (A1_vertex : ∃ x y : ℝ, x = -2 ∧ y = 0)
  (A2_vertex : ∃ x y : ℝ, x = 2 ∧ y = 0)
  (P_intersection : ∃ Px Py : ℝ, Px = -1) :
  Px = -1 :=
sorry

end hyperbola_center_origin_point_P_on_fixed_line_l205_205354


namespace kangaroo_inaccessible_points_l205_205665

def kangaroo_jump (x y : ℕ) : Prop :=
  ∀ (x' y' : ℕ), (x' = x + 1 ∧ y' = y - 1 ∨ x' = x - 5 ∧ y' = y + 7) →
    x' ≥ 0 ∧ y' ≥ 0 → x' + y' ≤ 4

theorem kangaroo_inaccessible_points :
  {p : ℕ × ℕ | let (x, y) := p in x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4} =
  {p : ℕ × ℕ | let (x, y) := p in kangaroo_jump x y} :=
sorry

end kangaroo_inaccessible_points_l205_205665


namespace hyeoncheol_initial_money_l205_205425

theorem hyeoncheol_initial_money
  (X : ℕ)
  (h1 : X / 2 / 2 = 1250) :
  X = 5000 :=
sorry

end hyeoncheol_initial_money_l205_205425


namespace laura_weekly_miles_l205_205482

theorem laura_weekly_miles (house_to_school_roundtrip supermarket_distance : ℕ)
  (school_trips_per_week supermarket_trips_per_week : ℕ)
  (house_to_school_roundtrip = 20)
  (supermarket_distance = 30)
  (school_trips_per_week = 5)
  (supermarket_trips_per_week = 2) : 
  (house_to_school_roundtrip * school_trips_per_week) + (supermarket_distance * supermarket_trips_per_week) = 180 := 
by
  sorry

end laura_weekly_miles_l205_205482


namespace simplify_trig_expression_l205_205926

theorem simplify_trig_expression (α : ℝ) : 
  let y := 2 * sin (2 * α) / (1 + cos (2 * α))
  in y = 2 * tan α :=
by
  let y := 2 * sin (2 * α) / (1 + cos (2 * α))
  show y = 2 * tan α,
  sorry

end simplify_trig_expression_l205_205926


namespace boundary_length_is_25_point_7_l205_205679

-- Define the side length derived from the given area.
noncomputable def sideLength (area : ℝ) : ℝ :=
  Real.sqrt area

-- Define the length of each segment when the square's side is divided into four equal parts.
noncomputable def segmentLength (side : ℝ) : ℝ :=
  side / 4

-- Define the total boundary length, which includes the circumference of the quarter-circle arcs and the straight segments.
noncomputable def totalBoundaryLength (area : ℝ) : ℝ :=
  let side := sideLength area
  let segment := segmentLength side
  let arcsLength := 2 * Real.pi * segment  -- the full circle's circumference
  let straightLength := 4 * segment
  arcsLength + straightLength

-- State the theorem that the total boundary length is approximately 25.7 units.
theorem boundary_length_is_25_point_7 :
  totalBoundaryLength 100 = 5 * Real.pi + 10 :=
by sorry

end boundary_length_is_25_point_7_l205_205679


namespace isosceles_obtuse_triangle_smallest_angle_l205_205697

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β : ℝ), 0 < α ∧ α = 1.5 * 90 ∧ α + 2 * β = 180 ∧ β = 22.5 := by
  sorry

end isosceles_obtuse_triangle_smallest_angle_l205_205697


namespace part_a_part_b_l205_205032

-- Define what it means for a coloring to be valid.
def valid_coloring (n : ℕ) (colors : Fin n → Fin 3) : Prop :=
  ∀ (i : Fin n),
  ∃ j k : Fin n, 
  ((i + 1) % n = j ∧ (i + 2) % n = k ∧ colors i ≠ colors j ∧ colors i ≠ colors k ∧ colors j ≠ colors k)

-- Part (a)
theorem part_a (n : ℕ) (hn : 3 ∣ n) : ∃ (colors : Fin n → Fin 3), valid_coloring n colors :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) : (∃ (colors : Fin n → Fin 3), valid_coloring n colors) → 3 ∣ n :=
by sorry

end part_a_part_b_l205_205032


namespace smallest_n_for_unique_sum_l205_205722

-- Define what it means for a positive real number to be unique
def is_unique (x : ℝ) : Prop :=
  ∃ (digits : List ℕ), (∀ d ∈ digits, d = 0 ∨ d = 5) ∧ x = List.foldr (λ (d : ℕ) (acc : ℝ), acc / 10 + d) 0 digits / (10 ^ digits.length)

-- The desired theorem statement
theorem smallest_n_for_unique_sum (n : ℕ) : (∀ x : ℝ, is_unique x → ∃ (l : List ℝ), (∀ (y : ℝ) ∈ l, is_unique y) ∧ List.sum l = 1 ∧ l.length = n) ↔ n = 5 := 
sorry

end smallest_n_for_unique_sum_l205_205722


namespace simplify_sqrt_expression_l205_205924

theorem simplify_sqrt_expression :
  sqrt (5 * 3) * sqrt (3^4 * 5^2) = 15 * sqrt 15 :=
by sorry

end simplify_sqrt_expression_l205_205924


namespace find_n_l205_205047

def append_fives (n : ℕ) : String :=
  "1200" ++ String.replicate (10 * n + 2) '5'

def base_6_to_nat (s : String) : ℕ :=
  s.foldl (λ acc c, acc * 6 + (c.to_nat - '0'.to_nat)) 0

def x (n : ℕ) : ℕ :=
  base_6_to_nat (append_fives n)

def has_two_prime_factors (n : ℕ) : Prop :=
  (nat.prime_factors n).to_finset.card = 2

theorem find_n : ∀ n : ℕ, has_two_prime_factors (x n) ↔ n = 0 :=
by
  sorry

end find_n_l205_205047


namespace constant_term_in_expansion_l205_205342

theorem constant_term_in_expansion (a : ℝ) (h_pos : a > 0) (h_sum : (1:ℝ)^2 * (a - 1)^6 = 1) : 
  let expansion := (2 * Real.exp (6:ℝ) - Real.exp (-6))^6,
      constant_term := 60
  in constant_term = 60 :=
by 
  sorry

end constant_term_in_expansion_l205_205342


namespace smallest_pos_int_div_by_four_primes_l205_205106

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205106


namespace arthur_reading_pages_l205_205700

theorem arthur_reading_pages :
  let total_goal : ℕ := 800
  let pages_read_from_500_book : ℕ := 500 * 80 / 100 -- 80% of 500 pages
  let pages_read_from_1000_book : ℕ := 1000 / 5 -- 1/5 of 1000 pages
  let total_pages_read : ℕ := pages_read_from_500_book + pages_read_from_1000_book
  let remaining_pages : ℕ := total_goal - total_pages_read
  remaining_pages = 200 :=
by
  -- placeholder for actual proof
  sorry

end arthur_reading_pages_l205_205700


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205077

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205077


namespace hyperbola_and_fixed_line_proof_l205_205380

-- Definitions based on conditions
def center_at_origin (C : Type) [Hyperbola C] : Prop :=
  C.center = (0, 0)

def left_focus (C : Type) [Hyperbola C] : Prop :=
  C.focus1 = (-2 * Real.sqrt 5, 0)

def eccentricity (C : Type) [Hyperbola C] : Prop :=
  C.eccentricity = Real.sqrt 5

def vertices (C : Type) [Hyperbola C] : Prop :=
  C.vertex1 = (-2, 0) ∧ C.vertex2 = (2, 0)

def passing_line_intersections (C : Type) (M N : Type) [Hyperbola C] : Prop :=
  ∃ l : Line, l ⨯ (-4, 0) ∧ l.intersects_hyperbola_left_branch C = {M, N}

def intersection (M N A1 A2 P : Type) [Hyperbola C] : Prop :=
  ∃ A1 A2, A1 = (-2, 0) ∧ A2 = (2, 0) ∧ (MA1.M ∩ NA2.N = P)

-- Proof problem statement
theorem hyperbola_and_fixed_line_proof (C : Type) [Hyperbola C] (M N P : Type)
  (h1 : center_at_origin C) (h2 : left_focus C) (h3 : eccentricity C)
  (h4 : passing_line_intersections C M N) (h5 : vertices C) (h6 : intersection M N C.vertex1 C.vertex2 P) : 
  (C.equation = (∀ x y, C.vertical_transverse_axis x y (x^2 / 4 - y^2 / 16)) ∧ 
   ∀ (x : ℝ), P.x = -1) :=
by sorry

end hyperbola_and_fixed_line_proof_l205_205380


namespace power_function_value_l205_205804

variable (f : ℝ → ℝ)

-- Condition: power function passes through (1/2, 8)
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f (1 / 2) = 8

-- Question: What is f(2)?
theorem power_function_value (h : passes_through_point f) : f 2 = 1 / 8 := 
by
  sorry

end power_function_value_l205_205804


namespace complete_the_square_factorization_l205_205211

-- Completing the square problem
theorem complete_the_square (x p : ℝ) : x^2 + 2 * p * x + 1 = (x + p)^2 + (1 - p^2) :=
sorry

-- Factorizing the given expression
theorem factorization (a b : ℝ) : a^2 - b^2 + 4 * a + 2 * b + 3 = (a + b + 1) * (a - b + 3) :=
sorry

end complete_the_square_factorization_l205_205211


namespace probability_zero_for_specific_roots_l205_205021

theorem probability_zero_for_specific_roots (z x : ℂ) (h : z ≠ x ∧ z^(2017) = 1 ∧ x^(2017) = 1) :
  ∃ k, k ∈ finset.range(2017) ∧ z = complex.exp(2 * real.pi * complex.I * k / 2017) ∧
  x = complex.exp(2 * real.pi * complex.I * k / 2017) ∧
  sqrt(2 + sqrt(5)) ≤ complex.abs(z + x) -> false :=
sorry

end probability_zero_for_specific_roots_l205_205021


namespace hyperbola_center_origin_point_P_on_fixed_line_l205_205353

variables {x y : ℝ}

def hyperbola (h : (x^2 / 4) - (y^2 / 16) = 1) : Prop := true

theorem hyperbola_center_origin 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (left_focus : ∃ x y : ℝ, x = -2 * Real.sqrt 5 ∧ y = 0)
  (eccentricity_root5 : ∃ e : ℝ, e = Real.sqrt 5) :
  hyperbola ((x^2 / 4) - (y^2 / 16) = 1) := 
sorry

theorem point_P_on_fixed_line
  (line_through_point : ∃ x y : ℝ, x = -4 ∧ y = 0)
  (M_second_quadrant : ∃ Mx My : ℝ, Mx < 0 ∧ My > 0)
  (A1_vertex : ∃ x y : ℝ, x = -2 ∧ y = 0)
  (A2_vertex : ∃ x y : ℝ, x = 2 ∧ y = 0)
  (P_intersection : ∃ Px Py : ℝ, Px = -1) :
  Px = -1 :=
sorry

end hyperbola_center_origin_point_P_on_fixed_line_l205_205353


namespace count_solutions_of_equation_l205_205424

theorem count_solutions_of_equation :
  let valid_x := {x ∈ Finset.range 151 | ∀ k: ℕ, x ≠ k^3} in
  valid_x.card = 145 :=
by
  sorry

end count_solutions_of_equation_l205_205424


namespace least_pos_int_div_by_four_distinct_primes_l205_205166

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205166


namespace length_of_string_C_l205_205044

theorem length_of_string_C (A B C : ℕ) (h1 : A = 6 * C) (h2 : A = 5 * B) (h3 : B = 12) : C = 10 :=
sorry

end length_of_string_C_l205_205044


namespace derivative_of_y_is_correct_l205_205554

noncomputable def y (x : ℝ) := x^2 * Real.sin x

theorem derivative_of_y_is_correct : (deriv y x = 2 * x * Real.sin x + x^2 * Real.cos x) :=
by
  sorry

end derivative_of_y_is_correct_l205_205554


namespace two_digit_swap_diff_divisible_by_9_l205_205189

theorem two_digit_swap_diff_divisible_by_9 (a b : ℕ) (ha : a < 10) (hb : b < 10) (h_neq : a ≠ b) :
  ∃ k : ℕ, 9 * k = |(10 * a + b) - (10 * b + a)| :=
by
  sorry

end two_digit_swap_diff_divisible_by_9_l205_205189


namespace smallest_abs_value_of_36_pow_k_minus_5_pow_l_l205_205694

theorem smallest_abs_value_of_36_pow_k_minus_5_pow_l :
  ∃ k l : ℕ, 1 ≤ k ∧ 1 ≤ l ∧ abs (36^k - 5^l) = 11 :=
sorry

end smallest_abs_value_of_36_pow_k_minus_5_pow_l_l205_205694


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205075

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205075


namespace perimeter_square_combined_perimeter_not_determined_l205_205680

theorem perimeter_square (side_square : ℝ) (h : side_square = 7) : 
  4 * side_square = 28 :=
by
  rw [h]
  norm_num

theorem combined_perimeter_not_determined (base_triangle height_triangle : ℝ) 
  (h_base : base_triangle = 5) (h_height : height_triangle = 6) : 
  ∃ side1 side2 : ℝ, 
  side1 ≠ 0 ∧ side2 ≠ 0 ∧ side1 ≠ side2 ∧ ¬(∀ (triangle_perimeter : ℝ), 
  triangle_perimeter = base_triangle + side1 + side2) :=
by
  intros side1 side2 
  use side1
  use side2
  split
  { sorry }
  split
  { sorry }
  split
  { sorry }
  { sorry }

end perimeter_square_combined_perimeter_not_determined_l205_205680


namespace quadratic_discriminant_l205_205628

theorem quadratic_discriminant : 
  ∀ (a b c : ℤ), a = 1 → b = -4 → c = -11 →
  let Δ := b^2 - 4 * a * c in Δ = 60 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  let Δ := b^2 - 4 * a * c
  have hΔ : Δ = (-4)^2 - 4 * 1 * (-11) := rfl
  rw [hΔ]
  norm_num
  sorry

end quadratic_discriminant_l205_205628


namespace units_digit_47_pow_47_l205_205618

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l205_205618


namespace line_eq_l205_205635

variables {x x1 x2 y y1 y2 : ℝ}

theorem line_eq (h : x2 ≠ x1 ∧ y2 ≠ y1) : 
  (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1) :=
sorry

end line_eq_l205_205635


namespace workers_number_l205_205641

theorem workers_number (W A : ℕ) (h1 : W * 25 = A) (h2 : (W + 10) * 15 = A) : W = 15 :=
by
  sorry

end workers_number_l205_205641


namespace no_tetrahedron_with_all_edges_obtuse_l205_205907

theorem no_tetrahedron_with_all_edges_obtuse :
  ¬∃ (T : set (fin 4)) (edges : finset (fin 4 × fin 4)),
  (∀ {a b c : fin 4},
    {a, b, c} ∈ T → ({a, b} ∈ edges ∧ {b, c} ∈ edges ∧ {c, a} ∈ edges)) ∧
  (∀ {a b : fin 4},
    {a, b} ∈ edges → ∃ {c d : fin 4},
    ({a, b, c} ∈ T ∧ {a, b, d} ∈ T ∧ obtuse_angle a b c ∧ obtuse_angle a b d)) :=
sorry

end no_tetrahedron_with_all_edges_obtuse_l205_205907


namespace total_distance_travelled_by_Sravan_l205_205020

-- Define the conditions
variables (D : ℝ) -- Total distance travelled by Sravan in km
variables (T : ℝ) -- Total travelling time in hours
variables (V1 V2 : ℝ) -- Speeds in km/h

-- Assume conditions given in the problem
def conditions := 
  V1 = 45 ∧ 
  V2 = 30 ∧ 
  T = 15 ∧ 
  ((D / 2) / V1 + (D / 2) / V2 = T)

-- Theorem statement
theorem total_distance_travelled_by_Sravan :
  conditions D T V1 V2 → D = 270 :=
begin
  sorry
end

end total_distance_travelled_by_Sravan_l205_205020


namespace equation_of_hyperbola_point_P_fixed_line_l205_205364

-- Definitions based on conditions
def center_origin := (0, 0)
def left_focus := (-2 * real.sqrt 5, 0)
def eccentricity := real.sqrt 5
def left_vertex := (-2, 0)
def right_vertex := (2, 0)
def point_passed_through_line := (-4, 0)

-- The equation of the hyperbola that satisfies the given conditions.
theorem equation_of_hyperbola :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ 
    (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ 
      (2 * real.sqrt 5)^2 = a^2 + b^2) :=
by
  use [2, 4]
  split
  exact rfl
  split
  exact rfl
  sorry

-- Proving P lies on the fixed line x = -1 for given lines and vertices conditions
theorem point_P_fixed_line :
  ∀ m : ℝ, ∀ y1 y2 x1 x2 : ℝ, 
    x1 = m * y1 - 4 ∧ x2 = m * y2 - 4 ∧ 
    (y1 + y2 = (32 * m) / ((4 * m^2) - 1)) ∧
    (y1 * y2 = 48 / ((4 * m^2) - 1)) ∧ 
    (x1 + 2) ≠ 0 ∧ 
    (x2 - 2) ≠ 0 → 
    (P : ℝ × ℝ) ((P.1, P.2) = (-1, P.2) ∧ P.2 = (y1 * (x + 2) / x1 ≠ -2 ∧ y2 * (x - 2) / x2 ≠ 2)) := 
by
  intros
  sorry

end equation_of_hyperbola_point_P_fixed_line_l205_205364


namespace max_largest_element_l205_205235

theorem max_largest_element (L : list ℕ) (hlen : L.length = 6) (hmed : list.median L = 4) (hmean : list.sum L = 60) :
  list.maximum L ≤ 49 :=
by
  sorry

end max_largest_element_l205_205235


namespace equation_of_hyperbola_point_P_on_fixed_line_l205_205372

-- Definition and conditions
def hyperbola_center_origin := (0, 0)
def hyperbola_left_focus := (-2 * Real.sqrt 5, 0)
def hyperbola_eccentricity := Real.sqrt 5
def point_on_line_through_origin := (-4, 0)
def point_M_in_second_quadrant (M : ℝ × ℝ) := M.1 < 0 ∧ M.2 > 0
def vertices_A1_A2 := ((-2, 0), (2, 0))

theorem equation_of_hyperbola (c a b : ℝ) (h_c : c = 2 * Real.sqrt 5)
  (h_e : hyperbola_eccentricity = Real.sqrt 5) (h_a : a = 2) (h_b : b = 4) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 16) = 1) := 
by
  intros x y h_x h_y
  have h : c^2 = a^2 + b^2 := sorry
  have h2 : hyperbola_eccentricity = c / a := sorry
  exact sorry

theorem point_P_on_fixed_line (M N P : ℝ × ℝ) (A1 A2 : ℝ × ℝ)
  (h_vertices : vertices_A1_A2 = (A1, A2))
  (h_M_in_2nd : point_M_in_second_quadrant M) 
  (h_P_intersects : ∀ t : ℝ, P = (t, (A1.2 * A2.1 - A1.1 * A2.2) / (A1.1 - A2.1))) :
  (P.1 = -1) := sorry

end equation_of_hyperbola_point_P_on_fixed_line_l205_205372


namespace least_positive_integer_divisible_by_four_primes_l205_205123

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205123


namespace find_hyperbola_fixed_line_through_P_l205_205394

-- Define the conditions of the problem
def center_origin (C : Type) : Prop := 
  C.center = (0, 0)

def left_focus_at (C : Type) : Prop :=
  C.left_focus = (-2 * Real.sqrt 5, 0)

def eccentricity_sqrt_5 (C : Type) : Prop :=
  C.eccentricity = Real.sqrt 5

-- Define the specific hyperbola equation based on the conditions
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 16) = 1

-- State the results to prove
theorem find_hyperbola (C : Type) :
  center_origin C →
  left_focus_at C →
  eccentricity_sqrt_5 C →
  ∀ x y : ℝ, hyperbola_equation x y :=
sorry

-- Define the fixed line for point P
def fixed_line (P : Type) : Prop :=
  P.x = -1

-- State the results to prove for point P
theorem fixed_line_through_P (M N A1 A2 P : Type) (x y : ℝ) :
  (M.line_passing_through (Point.mk (-4, 0))).intersects_left_branch (N : Point) →
  M.in_second_quadrant →
  (A1.coords = (-2, 0)) ∧ (A2.coords = (2, 0)) →
  Point.mk x y = P →
  fixed_line P :=
sorry

end find_hyperbola_fixed_line_through_P_l205_205394


namespace total_amount_invested_l205_205439

theorem total_amount_invested (x y total : ℝ) (h1 : 0.10 * x - 0.08 * y = 83) (h2 : y = 650) : total = 2000 :=
sorry

end total_amount_invested_l205_205439


namespace average_weight_is_57_l205_205650

noncomputable def average_weight_of_remaining_students : ℝ :=
  let weight_of_leaving_student := 45
  let weight_increase := 0.2
  let total_students_before := 60
  let total_students_after := 59
  let equation := total_students_before * (average_weight_of_remaining_students - weight_increase) = total_students_after * average_weight_of_remaining_students + weight_of_leaving_student
  solve_for average_weight_of_remaining_students

theorem average_weight_is_57 :
  average_weight_of_remaining_students = 57 :=
by
  sorry

end average_weight_is_57_l205_205650


namespace horizontal_asymptote_crossing_l205_205760

def g (x : ℝ) : ℝ := (3 * x^2 - 6 * x - 9) / (x^2 - 5 * x + 6)

theorem horizontal_asymptote_crossing : ∃ x : ℝ, g x = 3 ∧ x = 3 :=
by
  use 3
  split
  sorry
  rfl

end horizontal_asymptote_crossing_l205_205760


namespace farmer_price_per_dozen_l205_205662

noncomputable def price_per_dozen 
(farmer_chickens : ℕ) 
(eggs_per_chicken : ℕ) 
(total_money_made : ℕ) 
(total_weeks : ℕ) 
(eggs_per_dozen : ℕ) 
: ℕ :=
total_money_made / (total_weeks * (farmer_chickens * eggs_per_chicken) / eggs_per_dozen)

theorem farmer_price_per_dozen 
  (farmer_chickens : ℕ) 
  (eggs_per_chicken : ℕ) 
  (total_money_made : ℕ) 
  (total_weeks : ℕ) 
  (eggs_per_dozen : ℕ) 
  (h_chickens : farmer_chickens = 46) 
  (h_eggs_per_chicken : eggs_per_chicken = 6) 
  (h_money : total_money_made = 552) 
  (h_weeks : total_weeks = 8) 
  (h_dozen : eggs_per_dozen = 12) 
: price_per_dozen farmer_chickens eggs_per_chicken total_money_made total_weeks eggs_per_dozen = 3 := 
by 
  rw [h_chickens, h_eggs_per_chicken, h_money, h_weeks, h_dozen]
  have : (552 : ℕ) / (8 * (46 * 6) / 12) = 3 := by norm_num
  exact this

end farmer_price_per_dozen_l205_205662


namespace least_pos_int_div_by_four_distinct_primes_l205_205162

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205162


namespace area_ratio_l205_205689

theorem area_ratio (ABC : Triangle) (X : Point)
  (hX : on_segment X ABC.A ABC.B ∧ (AX / AB) = 1/4)
  (A' B'' : Point) :
  -- Define centroid G in the context
  G = centroid ABC.A ABC.B ABC.C →
  -- Define the conditions for A' and B'' intersecting medians
  A' = intersection (median G ABC.C) (line_segment X ABC.C) →
  B'' = intersection (median G ABC.C) (line_segment X ABC.A) →
  -- The relationship between the areas
  area_ratio (triangle A'' B'' C'') (triangle A' B' C') = 25/49 :=
sorry

end area_ratio_l205_205689


namespace count_4digit_numbers_less_than_2013_l205_205423

theorem count_4digit_numbers_less_than_2013 : 
  let digits := [2, 0, 1, 3] in 
  (∑ (d1 d2 d3 d4 : ℕ) in (finset.univ.filter (λ x, x ∈ digits)), 
  if 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 2013 then 1 else 0) = 77 := sorry

end count_4digit_numbers_less_than_2013_l205_205423


namespace probability_A_mc_and_B_tf_probability_at_least_one_mc_l205_205515

section ProbabilityQuiz

variable (total_questions : ℕ) (mc_questions : ℕ) (tf_questions : ℕ)

def prob_A_mc_and_B_tf (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  (mc_questions * tf_questions : ℚ) / (total_questions * (total_questions - 1))

def prob_at_least_one_mc (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  1 - ((tf_questions * (tf_questions - 1) : ℚ) / (total_questions * (total_questions - 1)))

theorem probability_A_mc_and_B_tf :
  prob_A_mc_and_B_tf 10 6 4 = 4 / 15 := by
  sorry

theorem probability_at_least_one_mc :
  prob_at_least_one_mc 10 6 4 = 13 / 15 := by
  sorry

end ProbabilityQuiz

end probability_A_mc_and_B_tf_probability_at_least_one_mc_l205_205515


namespace find_angle_x_l205_205855

theorem find_angle_x (AB DC : Set Point)
  (ACF : Set Point)
  (A B C D F : Point)
  (H1 : parallel AB DC)
  (H2 : ∃ P: Point, P ∈ ACF ∧ (A = P ∨ C = P ∨ F = P))
  (H3 : angle ACB = 100)
  (H4 : angle ABC = 70)
  (H5 : angle ADC = 130) : 
  angle DAC = 20 :=
by
  sorry

end find_angle_x_l205_205855


namespace trapezoid_area_l205_205037

-- Define that ABCD is a symmetric trapezoid and the areas of the triangles
variables (ABCD : Type) [trapezoid ABCD]
variables (O : Point) [intersection_of_diagonals ABCD O]
variables (area_AOB : ℝ) (area_COD : ℝ)
axiom AOB_area : area_AOB = 52
axiom COD_area : area_COD = 117

-- Define the target theorem to prove the area of the trapezoid
theorem trapezoid_area : ∃ (area_ABCD : ℝ), area_ABCD = 325 :=
by
  sorry

end trapezoid_area_l205_205037


namespace find_min_x_l205_205427

theorem find_min_x (x y : ℕ) (h1 : 0.6 = y / (468 + x)) (h2 : x > 0) (h3 : y > 0) : x = 2 :=
by
  sorry

end find_min_x_l205_205427


namespace sum_xyz_is_sqrt_13_l205_205777

variable (x y z : ℝ)

-- The conditions
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z

axiom eq1 : x^2 + y^2 + x * y = 3
axiom eq2 : y^2 + z^2 + y * z = 4
axiom eq3 : z^2 + x^2 + z * x = 7 

-- The theorem statement: Prove that x + y + z = sqrt(13)
theorem sum_xyz_is_sqrt_13 : x + y + z = Real.sqrt 13 :=
by
  sorry

end sum_xyz_is_sqrt_13_l205_205777


namespace impossibility_of_even_sum_1x2_vertical_rectangles_l205_205467

theorem impossibility_of_even_sum_1x2_vertical_rectangles :
  ¬ ∃ (f : Fin 6 → Fin 7 → ℕ), (∀ i j, 1 ≤ f i j ∧ f i j ≤ 42) ∧
  (∃ g : set (Fin 6 × Fin 7) × (Fin 6 × Fin 7), ∀ (i : Fin 5) (j : Fin 7),
  f i j + f i.succ j % 2 = 0) := sorry

end impossibility_of_even_sum_1x2_vertical_rectangles_l205_205467


namespace dog_ate_cost_6_l205_205474

noncomputable def totalCost : ℝ := 4 + 2 + 0.5 + 2.5
noncomputable def costPerSlice : ℝ := totalCost / 6
noncomputable def slicesEatenByDog : ℕ := 6 - 2
noncomputable def costEatenByDog : ℝ := slicesEatenByDog * costPerSlice

theorem dog_ate_cost_6 : costEatenByDog = 6 := by
  sorry

end dog_ate_cost_6_l205_205474


namespace average_marks_all_students_l205_205199

theorem average_marks_all_students (A1 : nat) (M1 : nat) (A2 : nat) (M2 : nat) 
  (H1 : A1 = 28) (H2 : M1 = 40) (H3 : A2 = 50) (H4 : M2 = 60) : 
  (A1 * M1 + A2 * M2) / (A1 + A2) = 428 / 78 :=
by
  -- This part would include the actual proof steps.
  sorry

end average_marks_all_students_l205_205199


namespace chromatic_number_one_is_isolated_l205_205597

-- Define what it means for a graph to have a chromatic number of 1
def chromatic_number(G : Type) [Graph G] : ℕ :=
  ∃ c : G → ℕ, (∀ v w : G, G.adj v w → c v ≠ c w) ∧ ( ∀ k < 1, ∃ v : G, c v = k)

-- Define what it means for a graph to be a collection of isolated vertices
def isolated_vertices(G : Type) [Graph G] : Prop :=
  ∀ v w : G, ¬ G.adj v w

-- The theorem stating that a graph with chromatic number 1 is a collection of isolated vertices
theorem chromatic_number_one_is_isolated (G : Type) [Graph G] : 
  chromatic_number G = 1 → isolated_vertices G :=
begin
  sorry
end

end chromatic_number_one_is_isolated_l205_205597


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205172

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205172


namespace length_DE_l205_205546

theorem length_DE (AB : ℝ) (h_base : AB = 15) (DE_parallel : ∀ x y z : Triangle, True) (area_ratio : ℝ) (h_area_ratio : area_ratio = 0.25) : 
  ∃ DE : ℝ, DE = 7.5 :=
by
  sorry

end length_DE_l205_205546


namespace smallest_solution_l205_205206

noncomputable def f (x : ℝ) : ℝ :=
  (-Real.logBase 2 (105 + 2 * x * Real.sqrt (x + 19)) ^ 3 + 
   |Real.logBase 2 ((105 + 2 * x * Real.sqrt (x + 19)) / (x ^ 2 + x + 3) ^ 4)|) /
  (9 * Real.logBase 5 (76 + 2 * x * Real.sqrt (x + 19)) - 4 * Real.logBase 2 (105 + 2 * x * Real.sqrt (x + 19)))

theorem smallest_solution :
  f ((-21 + Real.sqrt 33) / 2) ≥ 0 :=
sorry

end smallest_solution_l205_205206


namespace least_positive_integer_divisible_by_four_primes_l205_205119

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205119


namespace tangent_length_problem_l205_205286

open Real

def point (x y : ℝ) := (x, y)
def distance (p1 p2 : point) := sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A : point := point 3 4
def B : point := point 7 9
def C : point := point 5 12
def P : point := point 1 1

noncomputable def circumcenter_x : ℝ :=
  ((25 + 16)*(9 - 12) + (49 + 81)*(12 - 4) + (25 + 144)*(4 - 9)) / 
  (2 * (3 * (9 - 12) + 4 * (7 - 5) + 7 * 12 - 9 * 5))

noncomputable def circumcenter_y : ℝ :=
  ((25 + 16)*(5 - 7) + (49 + 81)*(3 - 5) + (25 + 144)*(7 - 3)) / 
  (2 * (3 * (9 - 12) + 4 * (7 - 5) + 7 * 12 - 9 * 5))

def circumcenter : point := point circumcenter_x circumcenter_y

noncomputable def circumradius : ℝ :=
  sqrt ((3 - circumcenter_x)^2 + (4 - circumcenter_y)^2)

noncomputable def PA := distance P A
noncomputable def PB := distance P B

noncomputable def tangent_length : ℝ := sqrt (PA * PB)

theorem tangent_length_problem : 
  tangent_length = sqrt (distance P A * distance P B) := sorry

end tangent_length_problem_l205_205286


namespace monotonicity_of_f_tangent_condition_minimum_value_of_g_l205_205402

section Problem

variables {a : ℝ} {x : ℝ} (e : ℝ) [NeZero a]

noncomputable def f (x : ℝ) : ℝ := a * (x - 1) / x^2

-- 1. Monotonicity of f
theorem monotonicity_of_f (hx : x ≠ 0) (ha : a ≠ 0) : 
  ((f' x > 0) ↔ (0 < x ∧ x < 2)) ∧ ((f' x < 0) ↔ ((x < 0) ∨ (x > 2))) ∨
  ((f' x < 0) ↔ (0 < x ∧ x < 2)) ∧ ((f' x > 0) ↔ ((x < 0) ∨ (x > 2))) :=
sorry

-- 2. Tangent Condition
theorem tangent_condition (hx : x = 1 ∨ x = sqrt a ∨ x = -sqrt a) 
  (line_tangent : x - f x - 1 = 0) : a = 1 :=
sorry

-- 3. Minimum value of g(x)
noncomputable def g (x : ℝ) : ℝ := x * log x - x^2 * f x

theorem minimum_value_of_g (h_interval : 1 ≤ x ∧ x ≤ e) 
  (ha : 0 < a ∧ a ≤ 1 ∨ 1 < a ∧ a < 2 ∨ a ≥ 2) :
  (g 1 = 0 ∧ 0 < a ∧ a ≤ 1) ∨ 
  (g (exp (a - 1)) = a - exp (a - 1) ∧ 1 < a ∧ a < 2) ∨ 
  (g e = e * (1 - a) + a ∧ a ≥ 2) :=
sorry

end Problem

end monotonicity_of_f_tangent_condition_minimum_value_of_g_l205_205402


namespace max_distance_from_point_on_sphere_to_triangle_l205_205337

theorem max_distance_from_point_on_sphere_to_triangle
  (V_sphere : ℝ := 4 * real.sqrt 3 * real.pi)
  (side_length_ABC : ℝ := 2 * real.sqrt 2) :
  let radius_sphere := real.sqrt 3 in
  let radius_circumcircle_ABC := 2 * real.sqrt 6 / 3 in
  let distance_center_to_plane := real.sqrt (radius_sphere ^ 2 - radius_circumcircle_ABC ^ 2) in
  let max_distance := radius_sphere + distance_center_to_plane in
  max_distance = 4 * real.sqrt 3 / 3 :=
by
  sorry

end max_distance_from_point_on_sphere_to_triangle_l205_205337


namespace train_speed_l205_205683

-- Train length (in meters)
def train_length : ℕ := 170

-- Bridge length (in meters)
def bridge_length : ℕ := 205

-- Time to cross bridge (in seconds)
def crossing_time : ℕ := 30

-- Function to calculate total distance
def total_distance (train : ℕ) (bridge : ℕ) : ℕ :=
  train + bridge

-- Function to calculate speed in m/s
def speed_m_s (distance : ℕ) (time : ℕ) : ℝ :=
  distance.to_real / time.to_real

-- Conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Function to convert speed to km/hr
def speed_km_hr (speed : ℝ) : ℝ :=
  speed * conversion_factor

-- Main theorem to prove
theorem train_speed : speed_km_hr (speed_m_s (total_distance train_length bridge_length) crossing_time) = 45 :=
by
  sorry

end train_speed_l205_205683


namespace square_measurement_error_l205_205272

theorem square_measurement_error (S S' : ℝ) (error_percentage : ℝ)
  (area_error_percentage : ℝ) (h1 : area_error_percentage = 2.01) :
  error_percentage = 1 :=
by
  sorry

end square_measurement_error_l205_205272


namespace part1_part2_l205_205796

def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 6)

theorem part1 (k : ℝ) : (∀ x, f x > k -> (x < -3 ∨ x > -2)) ↔ k = -2 / 5 :=
sorry

theorem part2 (t : ℝ) : (∀ x > 0, f x ≤ t) ↔ t ≥ sqrt(6) / 6 :=
sorry

end part1_part2_l205_205796


namespace number_of_correct_statements_l205_205558

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

lemma min_max_property 
  {f : ℝ → ℝ} 
  (odd_f : odd_function f) 
  (min_f_pos : ∃ x, 0 < x ∧ ∀ y, 0 < y → f (y) ≥ f (x) ∧ f (x) = 1) :
  ∃ z, z < 0 ∧ ∀ y, y < 0 → f (y) ≤ f (z) ∧ f (z) = -1 := 
sorry

lemma increasing_decreasing_property
  {f : ℝ → ℝ}
  (odd_f : odd_function f)
  (increasing_pos : ∀ x y, 0 < x → 0 < y → x < y → f (x) < f (y)) :
  ∀ a b, a < 0 → b < 0 → a < b → f (a) < f (b) →
  false := 
sorry

lemma specific_values_on_intervals
  {f : ℝ → ℝ}
  (odd_f : odd_function f)
  (fx_pos : ∀ x, 0 < x → f(x) = x^2 + 2*x) :
  ∀ x, x < 0 → f(x) = -x^2 - 2*x :=
sorry

theorem number_of_correct_statements (f : ℝ → ℝ) 
  (odd_f : odd_function f)
  (h1 : f 0 = 0)
  (h2 : ∃ x, 0 < x ∧ ∀ y, 0 < y → f (y) ≥ f (x) ∧ f (x) = 1 → ∃ z, z < 0 ∧ ∀ y, y < 0 → f (y) ≤ f (z) ∧ f (z) = -1)
  (h3 : ∀ x y, 0 < x → 0 < y → x < y → f (x) < f (y) → ∀ a b, a < 0 → b < 0 → a < b → f (a) < f (b)) 
  (h4 : ∀ x, 0 < x → f(x) = x^2 + 2*x ↔ ∀ x, x < 0 → f(x) = -x^2 - 2*x) :
  ∃ p, (p = 2) :=
by {
  -- Based on the given conditions and the odd function properties,
  -- through verification, there must be exactly 2 correct statements.
  sorry
}

end number_of_correct_statements_l205_205558


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205147

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205147


namespace knights_in_labyrinth_l205_205898

theorem knights_in_labyrinth (n : ℕ) (L : Type) [Labyrinth L] : k(L) = n + 1 :=
by sorry

end knights_in_labyrinth_l205_205898


namespace ratio_F3_over_V2_for_square_rotation_l205_205253

theorem ratio_F3_over_V2_for_square_rotation (a : ℝ) (π : ℝ) :
  let F_cylinder := 6 * π * a^2,
      V_cylinder := 2 * π * a^3,
      F_cone := 4 * π * a^2 * real.sqrt 2,
      V_cone := (4 * a^3 * π * real.sqrt 2) / 3
  in (let ratio_cylinder := (F_cylinder^3) / (V_cylinder^2),
          ratio_cone := (F_cone^3) / (V_cone^2)
      in ratio_cylinder = 54 * π ∨ ratio_cone = 36 * real.sqrt 2 * π) :=
sorry

end ratio_F3_over_V2_for_square_rotation_l205_205253


namespace speed_conversion_eq_l205_205295

theorem speed_conversion_eq:
  let speed_m_per_s := (13 : ℚ) / 36 in
  let conversion_factor := 3.6 in
  speed_m_per_s * conversion_factor = 1.3 :=
by
  sorry

end speed_conversion_eq_l205_205295


namespace units_digit_47_pow_47_l205_205610

theorem units_digit_47_pow_47 :
  let cycle := [7, 9, 3, 1] in
  cycle.nth (47 % 4) = some 3 :=
by
  let cycle := [7, 9, 3, 1]
  have h : 47 % 4 = 3 := by norm_num
  rw h
  simp
  exact trivial

end units_digit_47_pow_47_l205_205610


namespace remainder_degrees_division_l205_205627

theorem remainder_degrees_division (f : Polynomial ℤ) :
  ∃ r : Polynomial ℤ, ∃ q : Polynomial ℤ, f = q * (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X^2 + Polynomial.C 3 * Polynomial.X - Polynomial.C 9) + r ∧ 
    degree r < 3 := sorry

end remainder_degrees_division_l205_205627


namespace least_pos_int_div_by_four_distinct_primes_l205_205102

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205102


namespace real_operations_closed_add_real_operations_closed_sub_real_operations_closed_mul_real_operations_closed_div_l205_205633

variable {x y : ℝ}

-- Define the proof problem proving 4 arithmetic properties hold for real numbers.
theorem real_operations_closed_add : x + y ∈ ℝ := sorry

theorem real_operations_closed_sub : x - y ∈ ℝ := sorry

theorem real_operations_closed_mul : x * y ∈ ℝ := sorry

theorem real_operations_closed_div (hy : y ≠ 0) : x / y ∈ ℝ := sorry

end real_operations_closed_add_real_operations_closed_sub_real_operations_closed_mul_real_operations_closed_div_l205_205633


namespace mod_inverse_28_mod_29_l205_205747

-- Define the question and conditions as Lean definitions
def mod_inverse (a m x : ℕ) : Prop := (a * x) % m = 1

theorem mod_inverse_28_mod_29 :
  ∃ (x : ℕ), x ≤ 28 ∧ mod_inverse 28 29 x :=
begin
  existsi 28,
  split,
  { norm_num }, -- checks that 28 ≤ 28
  { unfold mod_inverse, norm_num }, -- checks that 28 * 28 % 29 = 1
end

end mod_inverse_28_mod_29_l205_205747


namespace sakshi_days_l205_205913

-- Define efficiency factor and days taken by Tanya
def efficiency_factor : ℝ := 1.25
def days_taken_by_tanya : ℝ := 16

-- Main theorem to prove Sakshi's days taken
theorem sakshi_days :
  let days_taken_by_sakshi := efficiency_factor * days_taken_by_tanya
  in days_taken_by_sakshi = 20 := 
by 
  -- Calculations that would be done within the proof
  sorry

end sakshi_days_l205_205913


namespace number_of_even_red_faces_cubes_l205_205258

def painted_cubes_even_faces : Prop :=
  let block_length := 4
  let block_width := 4
  let block_height := 1
  let edge_cubes_count := 8  -- The count of edge cubes excluding corners
  edge_cubes_count = 8

theorem number_of_even_red_faces_cubes : painted_cubes_even_faces := by
  sorry

end number_of_even_red_faces_cubes_l205_205258


namespace solve_triangle_l205_205462

noncomputable def measure_of_angle_A (A : ℝ) : Prop :=
  sin A + sqrt 3 * cos A = 0 ∧ A = 2 * Real.pi / 3

noncomputable def length_of_side_c (a b A c: ℝ) : Prop :=
  a = 2 * sqrt 7 ∧ b = 2 ∧ A = 2 * Real.pi / 3 ∧ c = 4

noncomputable def area_of_triangle_ABC (b c A S : ℝ) : Prop :=
  b = 2 ∧ c = 4 ∧ A = 2 * Real.pi / 3 ∧ S = 2 * sqrt 3

theorem solve_triangle (A a b c S : ℝ) 
  (h1 : sin A + sqrt 3 * cos A = 0)
  (h2 : a = 2 * sqrt 7)
  (h3 : b = 2) 
  : measure_of_angle_A A ∧ length_of_side_c a b A c ∧ area_of_triangle_ABC b c A S :=
by
  split
  · unfold measure_of_angle_A; simp; exact ⟨h1, by sorry⟩
  · split
    · unfold length_of_side_c; simp; exact ⟨h2, h3, by sorry, by sorry⟩
    · unfold area_of_triangle_ABC; simp; exact ⟨h3, by sorry, by sorry, by sorry⟩

end solve_triangle_l205_205462


namespace sum_of_valid_n_values_l205_205319

-- Define the function f(n) as described.
def f (n : ℕ) : ℕ :=
if n % 3 = 0 then n * (n / 2) + n / 3
else n * (n / 2)

-- Define the problem and the required proof.
theorem sum_of_valid_n_values : 
  (∑ n in (finset.range 158).filter (λ n, n ≥ 3 ∧ f(n + 1) = f(n) + 78), n) = 245 :=
begin
  sorry
end

end sum_of_valid_n_values_l205_205319


namespace polynomial_divisibility_l205_205725

theorem polynomial_divisibility (A B : ℝ) 
    (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^(205 : ℕ) + A * x + B = 0) : 
    A + B = -1 :=
by
  sorry

end polynomial_divisibility_l205_205725


namespace final_price_is_correct_l205_205571

/-- 
  The original price of a suit is $200.
-/
def original_price : ℝ := 200

/-- 
  The price increased by 25%, therefore the increase is 25% of the original price.
-/
def increase : ℝ := 0.25 * original_price

/-- 
  The new price after the price increase.
-/
def increased_price : ℝ := original_price + increase

/-- 
  After the increase, a 25% off coupon is applied.
-/
def discount : ℝ := 0.25 * increased_price

/-- 
  The final price consumers pay for the suit.
-/
def final_price : ℝ := increased_price - discount

/-- 
  Prove that the consumers paid $187.50 for the suit.
-/
theorem final_price_is_correct : final_price = 187.50 :=
by sorry

end final_price_is_correct_l205_205571


namespace perpendicular_vectors_x_l205_205814

theorem perpendicular_vectors_x 
  (x : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (x, -2))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : 
  x = 4 := 
  by 
  sorry

end perpendicular_vectors_x_l205_205814


namespace blue_to_red_marble_ratio_l205_205963

-- Define the given conditions and the result.
theorem blue_to_red_marble_ratio (total_marble yellow_marble : ℕ) 
  (h1 : total_marble = 19)
  (h2 : yellow_marble = 5)
  (red_marble : ℕ)
  (h3 : red_marble = yellow_marble + 3) : 
  ∃ blue_marble : ℕ, (blue_marble = total_marble - (yellow_marble + red_marble)) 
  ∧ (blue_marble / (gcd blue_marble red_marble)) = 3 
  ∧ (red_marble / (gcd blue_marble red_marble)) = 4 :=
by {
  --existence of blue_marble and the ratio
  sorry
}

end blue_to_red_marble_ratio_l205_205963


namespace valid_parameterizations_l205_205292

noncomputable def line_equation (x y : ℝ) : Prop := y = (5/3) * x + 1

def parametrize_A (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (3 + t * 3, 6 + t * 5) ∧ line_equation x y

def parametrize_D (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (-1 + t * 3, -2/3 + t * 5) ∧ line_equation x y

theorem valid_parameterizations : parametrize_A t ∧ parametrize_D t :=
by
  -- Proof steps are skipped
  sorry

end valid_parameterizations_l205_205292


namespace least_positive_integer_divisible_by_four_primes_l205_205120

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205120


namespace number_of_ties_l205_205827

/-- Define real numbers representing the costs of shirts, trousers, and ties. --/
variables (S T Y : ℝ)

/-- Represent the equations corresponding to the problem's conditions. --/
def equation1 := 6 * S + 4 * T + n * Y = 80
def equation2 := 4 * S + 2 * T + 2 * Y = 140
def equation3 := 5 * S + 3 * T + 2 * Y = 110

/-- Define the main problem as a theorem to be proved in Lean. --/
theorem number_of_ties (n : ℝ) (S T Y : ℝ) 
  (h1 : 6 * S + 4 * T + n * Y = 80)
  (h2 : 4 * S + 2 * T + 2 * Y = 140)
  (h3 : 5 * S + 3 * T + 2 * Y = 110) : Prop :=
/- To be filled by the learner in terms of Lean proof steps. -/
sorry

end number_of_ties_l205_205827


namespace find_x_l205_205600

def digit_sum (n : ℕ) : ℕ := 
  n.digits 10 |> List.sum

def k := (10^45 - 999999999999999999999999999999999999999999994 : ℕ)

theorem find_x :
  digit_sum k = 397 := 
sorry

end find_x_l205_205600


namespace equation_of_hyperbola_point_P_on_fixed_line_l205_205377

-- Definition and conditions
def hyperbola_center_origin := (0, 0)
def hyperbola_left_focus := (-2 * Real.sqrt 5, 0)
def hyperbola_eccentricity := Real.sqrt 5
def point_on_line_through_origin := (-4, 0)
def point_M_in_second_quadrant (M : ℝ × ℝ) := M.1 < 0 ∧ M.2 > 0
def vertices_A1_A2 := ((-2, 0), (2, 0))

theorem equation_of_hyperbola (c a b : ℝ) (h_c : c = 2 * Real.sqrt 5)
  (h_e : hyperbola_eccentricity = Real.sqrt 5) (h_a : a = 2) (h_b : b = 4) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 16) = 1) := 
by
  intros x y h_x h_y
  have h : c^2 = a^2 + b^2 := sorry
  have h2 : hyperbola_eccentricity = c / a := sorry
  exact sorry

theorem point_P_on_fixed_line (M N P : ℝ × ℝ) (A1 A2 : ℝ × ℝ)
  (h_vertices : vertices_A1_A2 = (A1, A2))
  (h_M_in_2nd : point_M_in_second_quadrant M) 
  (h_P_intersects : ∀ t : ℝ, P = (t, (A1.2 * A2.1 - A1.1 * A2.2) / (A1.1 - A2.1))) :
  (P.1 = -1) := sorry

end equation_of_hyperbola_point_P_on_fixed_line_l205_205377


namespace marissa_coins_difference_l205_205502

theorem marissa_coins_difference :
  ∃ (p : ℕ), 3 ≤ p ∧ p ≤ 2033 ∧ (10175 - 4 * 3) - (10175 - 4 * 2033) = 8120 :=
by
  use 3
  use 2033
  sorry

end marissa_coins_difference_l205_205502


namespace number_of_planes_through_point_parallel_to_skew_lines_l205_205813

noncomputable def skew_lines60 (a b : ℝ → (ℝ × ℝ × ℝ)) (P : ℝ × ℝ × ℝ) : Prop :=
  let angle_ab := 60
  ∧ (¬∃ t, P = a t ∨ P = b t)
  ∧ PlaneParallelToBoth : (Plane P a b)
  ∧ LineMakingAngle := (Line P a b angle_ab)

theorem number_of_planes_through_point_parallel_to_skew_lines (a b : ℝ → ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ) :
  skew_lines60 a b P →
  (∃! π, π = Plane P a b) ∧ 
  ∃! l, l = Line P a b 60 :=
sorry

end number_of_planes_through_point_parallel_to_skew_lines_l205_205813


namespace main_theorem_l205_205717

theorem main_theorem :
  ∃ k : ℂ, k = 4 / 3 ∧ ∀ x : ℂ, (x ≠ 0) →
    (x / (x + 1) + x / (x + 3) = k * x → (degree (polynomial.of_complex (kx^2 + (4k-2)x + (3k-4))) = 2)) :=
by
  sorry

end main_theorem_l205_205717


namespace larger_number_is_70380_l205_205991

theorem larger_number_is_70380 (A B : ℕ) 
    (hcf : Nat.gcd A B = 20) 
    (lcm : Nat.lcm A B = 20 * 9 * 17 * 23) :
    max A B = 70380 :=
  sorry

end larger_number_is_70380_l205_205991


namespace exists_positive_ints_seq_l205_205204

def num_divisors (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else (Finset.ask (Finset.range n).filter (λ d, d > 0 ∧ n % d = 0)).card

theorem exists_positive_ints_seq :
  ∃ (a : Fin 101 → ℕ), (∀ k : Fin 100, 0 < a k ∧ num_divisors (Finset.range (k.val + 1).sum (λ i, a ⟨i, sorry⟩)) = a k) :=
sorry

end exists_positive_ints_seq_l205_205204


namespace tom_dimes_now_l205_205592

-- Define the initial number of dimes and the number of dimes given by dad
def initial_dimes : ℕ := 15
def dimes_given_by_dad : ℕ := 33

-- Define the final count of dimes Tom has now
def final_dimes (initial_dimes dimes_given_by_dad : ℕ) : ℕ :=
  initial_dimes + dimes_given_by_dad

-- The main theorem to prove "how many dimes Tom has now"
theorem tom_dimes_now : initial_dimes + dimes_given_by_dad = 48 :=
by
  -- The proof can be skipped using sorry
  sorry

end tom_dimes_now_l205_205592


namespace intersection_eq_zero_set_l205_205892

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | x^2 ≤ 0}

theorem intersection_eq_zero_set : M ∩ N = {0} := by
  sorry

end intersection_eq_zero_set_l205_205892


namespace polyhedra_common_interior_point_l205_205333

-- Definitions based on conditions stated in Lean 4
structure Polyhedron (V : Type) [OrderedSemiring V] :=
  (vertices : Finset (V × V × V)) -- Assuming 3D space with vertices

-- Define a convex polyhedron with a set of vertices
variables {V : Type} [OrderedSemiring V]

-- Function to translate polyhedra
def translate (P : Polyhedron V) (A B : V × V × V) : Polyhedron V :=
  { vertices := P.vertices.map (λ v, (v.1 + (B.1 - A.1), v.2 + (B.2 - A.2), v.3 + (B.3 - A.3))) }

-- Conditions
variable (P1 : Polyhedron V)
variable (A : Fin 9 → V × V × V)  -- Representing 9 vertices A1, A2, ..., A9 

def Pi (i : Fin 9) : Polyhedron V :=
  if i = 0 then P1 else translate P1 (A 0) (A i)

-- Proof statement
theorem polyhedra_common_interior_point : 
  ∃ (i j : Fin 9), i ≠ j ∧ ∃ (x : V × V × V), x ∈ (interior (Pi i).vertices ∩ interior (Pi j).vertices) :=
sorry

end polyhedra_common_interior_point_l205_205333


namespace median_of_data_set_l205_205000

open List

-- Define the dataset and conditions
def data_set : List ℝ := [10, 10, x, 8]

-- Mode of the data set is 10
def mode_condition (l : List ℝ) : Prop := mode l = 10

-- Mean of the data set is 10
def mean_condition (l : List ℝ) : Prop := (sum l) / (length l) = 10

-- Given the conditions, prove the median is 10 when x is 12
theorem median_of_data_set (x : ℝ) (hmode : mode_condition data_set) (hmean : mean_condition data_set) (hx : x = 12) : 
  median ([8, 10, 10, x].sort (≤)) = 10 :=
by
  sorry

end median_of_data_set_l205_205000


namespace least_pos_int_div_by_four_distinct_primes_l205_205171

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205171


namespace mark_sold_8_boxes_less_l205_205503

theorem mark_sold_8_boxes_less (T M A x : ℕ) (hT : T = 9) 
    (hM : M = T - x) (hA : A = T - 2) 
    (hM_ge_1 : 1 ≤ M) (hA_ge_1 : 1 ≤ A) 
    (h_sum_lt_T : M + A < T) : x = 8 := 
by
  sorry

end mark_sold_8_boxes_less_l205_205503


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205081

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205081


namespace f_three_l205_205559

def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then log 3 (3 - x) else f (x - 1)

theorem f_three : f 3 = 1 := by
  sorry

end f_three_l205_205559


namespace mod_inverse_28_mod_29_l205_205746

theorem mod_inverse_28_mod_29 : ∃ a : ℕ, 28 * a % 29 = 1 ∧ 0 ≤ a ∧ a ≤ 28 ∧ a = 28 :=
by {
  use 28,
  simp,
  sorry
}

end mod_inverse_28_mod_29_l205_205746


namespace sum_b_seq_l205_205768

variable (a : ℝ) (a_seq : ℕ → ℝ) (S_seq : ℕ → ℝ) (b_seq : ℕ → ℝ)
-- Conditions
axiom seq_nonneg : ∀ n, a_seq n ≥ 0
axiom first_term : a_seq 1 = a
axiom S_recurrence : ∀ n ≥ 2, S_seq n = (Real.sqrt (S_seq (n - 1)) + Real.sqrt a)^2
axiom b_definition : ∀ n, b_seq n = (a_seq (n + 1) / a_seq n) + (a_seq n / a_seq (n + 1))

-- Theorem to prove
theorem sum_b_seq (n : ℕ) : 
  (∑ i in Finset.range n, b_seq (i + 1)) = (4 * n^2 + 6 * n) / (2 * n + 1) := sorry

end sum_b_seq_l205_205768


namespace stratified_sampling_example_l205_205066

theorem stratified_sampling_example
  (students_ratio : ℕ → ℕ) -- function to get the number of students in each grade, indexed by natural numbers
  (ratio_cond : students_ratio 0 = 4 ∧ students_ratio 1 = 3 ∧ students_ratio 2 = 2) -- the ratio 4:3:2
  (third_grade_sample : ℕ) -- number of students in the third grade in the sample
  (third_grade_sample_eq : third_grade_sample = 10) -- 10 students from the third grade
  (total_sample_size : ℕ) -- the sample size n
 :
  total_sample_size = 45 := 
sorry

end stratified_sampling_example_l205_205066


namespace jordan_weight_after_13_weeks_l205_205862

/-
Problem:
Jordan's starting weight = 250 pounds
For 1 ≤ i ≤ 4, Jordan loses 3 pounds every week
In week 5, Jordan loses 5 pounds
For 6 ≤ i ≤ 12, Jordan loses 2 pounds every week
In week 13, he gains back 2 pounds.
Prove Jordan's weight after week 13 = 221 pounds
-/

theorem jordan_weight_after_13_weeks:
  let start_weight := 250
  let weeks_1_to_4_loss := 3 * 4
  let week_5_loss := 5
  let weeks_6_to_12_loss := 2 * 7
  let week_13_gain := 2
  let total_loss := weeks_1_to_4_loss + week_5_loss + weeks_6_to_12_loss - week_13_gain
  let final_weight := start_weight - total_loss
  final_weight = 221 :=
by
  let start_weight := 250
  let weeks_1_to_4_loss := 3 * 4
  let week_5_loss := 5
  let weeks_6_to_12_loss := 2 * 7
  let week_13_gain := 2
  let total_loss := weeks_1_to_4_loss + week_5_loss + weeks_6_to_12_loss - week_13_gain
  let final_weight := start_weight - total_loss
  show final_weight = 221 from
  sorry

end jordan_weight_after_13_weeks_l205_205862


namespace decrease_percent_is_25_25_l205_205649

-- Defining the original tax and consumption
variables {T C : ℝ} (hT : T > 0) (hC : C > 0)

-- Defining conditions
-- New tax rate after diminishing by 35%
def new_tax_rate := 0.65 * T

-- New consumption rate after increasing by 15%
def new_consumption := 1.15 * C

-- Original revenue and new revenue
def original_revenue := T * C
def new_revenue := new_tax_rate * new_consumption

-- Decrease percent in revenue
def decrease_percent := ((original_revenue - new_revenue) / original_revenue) * 100

/-- The decrease percent in the revenue derived from the commodity is 25.25% -/
theorem decrease_percent_is_25_25 : decrease_percent = 25.25 := 
by
  sorry

end decrease_percent_is_25_25_l205_205649


namespace isosceles_triangle_probability_l205_205289

noncomputable def probability_point_closer_to_vertex (DEF : Triangle ℝ) (E D F : Point ℝ)
  (h1 : DE = 5) (h2 : EF = 5) (h3 : DF = 6) : ℝ :=
  -- Prove the probability that a point inside DEF is closer to E than to D or F is 5/8
  sorry

theorem isosceles_triangle_probability (DEF : Triangle ℝ) (E D F : Point ℝ)
  (h1 : DE = 5) (h2 : EF = 5) (h3 : DF = 6) :
  probability_point_closer_to_vertex DEF E D F h1 h2 h3 = 5 / 8 :=
  sorry

end isosceles_triangle_probability_l205_205289


namespace three_nabla_four_l205_205431

def operation_nabla (x y : ℝ) : ℝ :=
  (x + y) / (1 + (x * y) ^ 2)

theorem three_nabla_four : operation_nabla 3 4 = 7 / 145 := by
  sorry

end three_nabla_four_l205_205431


namespace remainder_of_product_l205_205973

theorem remainder_of_product (a b n : ℕ) (ha : a = 97) (hb : b = 103) (hn : n = 9) : ((a * b) % n) = 1 :=
by
  rw [ha, hb, hn]
  have h : (97 * 103) % 9 = 9991 % 9 := by sorry
  exact h

end remainder_of_product_l205_205973


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205175

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205175


namespace binom_26_6_l205_205774

open Nat

theorem binom_26_6 :
  nat.choose 23 5 = 33649 →
  nat.choose 23 6 = 33649 →
  nat.choose 25 5 = 53130 →
  nat.choose 26 6 = 163032 := by
  sorry

end binom_26_6_l205_205774


namespace sophie_total_spending_l205_205533

-- Definitions based on conditions
def num_cupcakes : ℕ := 5
def price_per_cupcake : ℝ := 2
def num_doughnuts : ℕ := 6
def price_per_doughnut : ℝ := 1
def num_slices_apple_pie : ℕ := 4
def price_per_slice_apple_pie : ℝ := 2
def num_cookies : ℕ := 15
def price_per_cookie : ℝ := 0.60

-- Total cost calculation
def total_cost : ℝ :=
  num_cupcakes * price_per_cupcake +
  num_doughnuts * price_per_doughnut +
  num_slices_apple_pie * price_per_slice_apple_pie +
  num_cookies * price_per_cookie

-- Theorem stating the total cost is 33
theorem sophie_total_spending : total_cost = 33 := by
  sorry

end sophie_total_spending_l205_205533


namespace hexagon_probability_l205_205770

-- Define the set of complex numbers representing the vertices of a centrally symmetric hexagon
def V : Set ℂ := {x | x = (0 : ℂ) + (sqrt 2) * Complex.i ∨
                       x = - (sqrt 2) * Complex.i ∨
                       x = (1 + Complex.i) / sqrt 8 ∨
                       x = -(1 + Complex.i) / sqrt 8 ∨
                       x = (1 - Complex.i) / sqrt 8 ∨
                       x = -(1 - Complex.i) / sqrt 8}

-- Assertion about the probability of product being -1
noncomputable def probability_P_eq_negative_one : ℚ :=
  (2^2 * 5 * 11) / 3^10

theorem hexagon_probability :
  let z : Fin 12 → ℂ := λ j, (Classical.choose (Set.nonempty_iff_exists.2 (Set.toFinset V).nonempty))
  let P : ℂ := ∏ j in (Finset.finRange 12), z j
  (Set.Prod.finRange_mem_range 12 (Set.toFinset V).nonempty)
  → (P = -1 → probability_P_eq_negative_one = (2^2 * 5 * 11) / 3^10) :=
by sorry

end hexagon_probability_l205_205770


namespace smallest_pos_int_div_by_four_primes_l205_205107

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205107


namespace sydney_cannot_reach_2021_2022_l205_205933

theorem sydney_cannot_reach_2021_2022 :
  ∀ (x y : ℤ), (x = 2021 ∧ y = 2022) → ¬∃ (steps : List (ℤ × ℤ)), 
    List.Head steps = (0, 0) ∧ List.Last steps (0, 0) = (2021, 2022) ∧
    (∀ (step : (ℤ × ℤ)) (m : (ℤ × ℤ)), step ∈ steps → m ∈ steps →
    (∃ p q : ℤ, (p, q) = ((step.1 + m.1) / 2, (step.2 + m.2) / 2) →
    ↑p ∈ ℤ ∧ ↑q ∈ ℤ)) := 
by
  intros x y hxy
  cases hxy with hxy1 hxy2
  have hx_even : Even 0 := ⟨0, rfl⟩
  have hy_even : Even 0 := ⟨0, rfl⟩
  have hx_odd : Odd 2021 := ⟨1010, rfl⟩
  have hy_even2 : Even 2022 := ⟨1011, rfl⟩
  sorry

end sydney_cannot_reach_2021_2022_l205_205933


namespace is_fake_coin_l205_205262

variable (n k : ℕ)

-- Conditions
axiom h₁ : 2k ≤ 2n+1

-- Definition of the problem
theorem is_fake_coin (selected_fake : bool) (weight_diff_odd : bool)
  (h₂ : ∃ a k₁ k₂: ℕ, (k₁ + k₂ = 2k) ∧ 
                      ((weight_diff_odd ∧ ((k₁ - k₂) % 2 = 1))
                      ∨ (¬weight_diff_odd ∧ ((k₁ - k₂) % 2 = 0)))) :
  selected_fake ↔ weight_diff_odd := 
by
  -- Proof would go here
  sorry

end is_fake_coin_l205_205262


namespace inequality_holds_l205_205520

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := sorry

end inequality_holds_l205_205520


namespace part1_part2_l205_205785

variable {V : Type} [AddCommGroup V] [Module ℝ V]

variables (a b : V) (not_collinear : ¬(∃ k : ℝ, a = k • b))
variables (OA OB OC : V)
variables (m n : ℝ)

-- Definitions of the vectors
def OA_def := (2 : ℝ) • a - b
def OB_def := a + (2 : ℝ) • b
def OC_def := m • a + n • b

-- Problem statement 1
theorem part1 (h1 : 2 • OA_def - OB_def = OC_def) : m = 3 ∧ n = -4 :=
sorry

-- Problem statement 2
theorem part2 (collinear : ∃ λ : ℝ, OC_def - OA_def = λ • (OB_def - OA_def)) :
  ∃ m n : ℝ, m = 3 ∧ n = -4 ∧ ∃ mn_max : ℝ, mn_max = (25 : ℝ) / 12 :=
sorry

end part1_part2_l205_205785


namespace polynomial_value_l205_205437

variables {R : Type*} [CommRing R] {x : R}

theorem polynomial_value (h : 2 * x^2 - x = 1) : 
  4 * x^4 - 4 * x^3 + 3 * x^2 - x - 1 = 1 :=
by 
  sorry

end polynomial_value_l205_205437


namespace find_p_q_r_l205_205883

-- Declare the constants p, q, and r
variables {p q r : ℝ}

-- State the conditions
def inequality_holds : Prop :=
  ∀ x : ℝ, (x < -1 ∨ (28 ≤ x ∧ x ≤ 32)) ↔ ((x - p) * (x - q) / (x - r) ≤ 0)

def p_less_q : Prop := p < q

-- The goal is to prove p + 2q + 3r = 89 given the conditions
theorem find_p_q_r (h : inequality_holds) (h_pq : p_less_q) : p + 2*q + 3*r = 89 :=
by
  sorry

end find_p_q_r_l205_205883


namespace probability_X_12_14_l205_205943

noncomputable def normal_probability (μ σ α β : ℝ) : ℝ :=
  let Zα := (α - μ) / σ
  let Zβ := (β - μ) / σ
  in Mathlib.Probability.CDF.normCDF Zβ - Mathlib.Probability.CDF.normCDF Zα

theorem probability_X_12_14 (X : ℝ → ℝ) (μ σ : ℝ) (hμ : μ = 10) (hσ : σ = 2) :
  normal_probability μ σ 12 14 = 0.1359 :=
by
  rw [hμ, hσ]
  sorry

end probability_X_12_14_l205_205943


namespace max_value_expr_l205_205317

theorem max_value_expr (y : ℝ) : 
  ∃ y, 0 < y → y = real.cbrt 3 → 
  (∀ x, (1 ≤ x^6 / (x^12 + 3*x^9 - 9*x^6 + 27*x^3 + 81)) ≤ 1/27) :=
begin
  sorry
end

end max_value_expr_l205_205317


namespace tank_capacity_l205_205195

-- Let's define the conditions
constant C : ℕ                                         -- The capacity of the tank in litres
constant leak_empty_rate : ℚ := C / 9                  -- Leak empties the tank in 9 hours
constant inlet_fill_rate : ℚ := 6 * 60                 -- Inlet fills at 6 litres per minute, i.e., 360 litres per hour
constant net_empty_rate : ℚ := C / 12                  -- Tank is empty in 12 hours with both leak and inlet

-- Define the final proof goal
theorem tank_capacity :
  360 - (C / 9) = C / 12 -> 
  C = 1851 := 
sorry

end tank_capacity_l205_205195


namespace length_of_DE_l205_205541

variable (A B C X Y Z D E : Type)
variable [LinearOrderedField ℝ]

def base_length_ABC : ℝ := 15
def triangle_area_ratio : ℝ := 0.25

theorem length_of_DE (h1 : DE // BC ∥ BC) 
                    (h2 : triangle_area_ratio * (base_length_ABC ^ 2) = DE ^ 2)
                    : DE = 7.5 :=
sorry

end length_of_DE_l205_205541


namespace sum_of_zero_points_l205_205405

noncomputable def e : ℝ := 2.71828

def f (x : ℝ) : ℝ :=
  if (0 ≤ x ∧ x < 1) then
    (2 * x) / (x + 1)
  else
    abs (x - 3) - 1

def g (x : ℝ) : ℝ := f x - 1 / e

theorem sum_of_zero_points :
  (let x1 := 1 / (2 * e - 1)
   let x2 := 4 + 1 / e
   let x3 := 2 - 1 / e
   in x1 + x2 + x3) = (1 / (2 * e - 1) + 6) :=
by
  sorry

end sum_of_zero_points_l205_205405


namespace max_value_of_a_l205_205438

open Real

theorem max_value_of_a (a : ℝ) :
  (∃ (l : linear_map ℝ ℝ ℝ), 
    ∃ (n m : ℝ), n > 0 ∧ m > 0 ∧ 
    (∀ x, l x = 2 * n * x - n^2) ∧ 
    (∀ x, l x = (a / m) * x + a * (log m - 1))
  ) → a ≤ 2 * exp 1 :=
begin
  sorry
end

end max_value_of_a_l205_205438


namespace fifteen_sided_figure_area_l205_205556

-- Define the vertices of the fifteen-sided figure
def vertices : List (ℝ × ℝ) :=
  [(1, 2), (2, 2), (2, 3), (3, 4), (4, 4), (5, 5), (6, 5), (7, 4),
   (6, 3), (6, 2), (5, 1), (4, 1), (3, 1), (2, 1), (1, 2)]

-- Define the area function using the Shoelace formula
def shoelace_area (vertices : List (ℝ × ℝ)) : ℝ :=
  let xy_pairs := vertices.zip (vertices.rotate 1)
  (xy_pairs.map (λ ((x1, y1), (x2, y2)), x1 * y2 - y1 * x2).sum) / 2

-- Define the theorem to prove the area of the figure as 15 cm²
theorem fifteen_sided_figure_area : shoelace_area vertices = 15 := by
  sorry

end fifteen_sided_figure_area_l205_205556


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205087

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205087


namespace sandwich_combinations_l205_205985

/-- Zoe has 12 different kinds of lunch meat and 8 different kinds of cheese.
    She wants to make a sandwich with one kind of meat and two kinds of cheese.
    The order of cheese selection does not matter. Prove that the number of 
    different combinations of sandwiches Zoe can make is 336. -/
theorem sandwich_combinations (meat_count cheese_count : ℕ) (h_meat : meat_count = 12) (h_cheese : cheese_count = 8) :
  let cheese_combinations := (finset.range cheese_count).card choose 2 in
  let total_combinations := meat_count * cheese_combinations in
  total_combinations = 336 :=
by
  sorry

end sandwich_combinations_l205_205985


namespace find_p_plus_q_plus_r_plus_s_l205_205887

theorem find_p_plus_q_plus_r_plus_s :
  ∃ (p q r s : ℕ), 
  let A B C : ℝ := undefined in
  let cos_A := Math.cos A
  let cos_B := Math.cos B
  let cos_C := Math.cos C
  let sin_A := Math.sin A
  let sin_B := Math.sin B
  let sin_C := Math.sin C
  angle_B_obtuse : B > π / 2 ∧ B < π ∧
  cos2_eq_frac1 : cos_A ^ 2 + cos_B ^ 2 + 2 * sin_A * sin_B * cos_C = 17 / 9 ∧
  cos2_eq_frac2 : cos_B ^ 2 + cos_C ^ 2 + 2 * sin_B * sin_C * cos_A = 15 / 8 ∧
  relatively_prime : (p + q).gcd s = 1 ∧
  not_divisible : ∀ prime, Nat.sqrt prime ∣ r → false
  in
  \cos^2 C + \math.cos^2 A + 2 \math.sin C \math.sin A \math.cos B = \frac{p - q \sqrt{r}}{s} ∧
  p + q + r + s = 182 := sorry

end find_p_plus_q_plus_r_plus_s_l205_205887


namespace percentage_saved_is_20_percent_l205_205236

noncomputable def income_in_first_year := ℝ
noncomputable def saving_portion_in_first_year (I S : ℝ) := S

noncomputable def income_in_second_year (I : ℝ) := 1.20 * I
noncomputable def saving_in_second_year (S : ℝ) := 2 * S

noncomputable def expenditure_in_first_year (I S : ℝ) := I - S
noncomputable def expenditure_in_second_year (I S : ℝ) := (1.20 * I) - (2 * S)

theorem percentage_saved_is_20_percent
    (I S : ℝ)
    (h1 : (I - S) + ((1.20 * I) - (2 * S)) = 2 * (I - S)) :
    (S / I) * 100 = 20 :=
begin
    sorry
end

end percentage_saved_is_20_percent_l205_205236


namespace total_profit_amount_l205_205241

-- Definitions representing the conditions:
def ratio_condition (P_X P_Y : ℝ) : Prop :=
  P_X / P_Y = (1 / 2) / (1 / 3)

def difference_condition (P_X P_Y : ℝ) : Prop :=
  P_X - P_Y = 160

-- The proof problem statement:
theorem total_profit_amount (P_X P_Y : ℝ) (h1 : ratio_condition P_X P_Y) (h2 : difference_condition P_X P_Y) :
  P_X + P_Y = 800 := by
  sorry

end total_profit_amount_l205_205241


namespace problem_statement_l205_205343

open Complex

theorem problem_statement (z : ℂ) (h : z + z⁻¹ = 2 * Complex.cos (5 * Real.pi / 180)) :
  z ^ 2010 + z ^ (-2010) = 0 :=
by
  sorry

end problem_statement_l205_205343


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205083

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205083


namespace total_guppies_l205_205261

noncomputable def initial_guppies : Nat := 7
noncomputable def baby_guppies_first_set : Nat := 3 * 12
noncomputable def baby_guppies_additional : Nat := 9

theorem total_guppies : initial_guppies + baby_guppies_first_set + baby_guppies_additional = 52 :=
by
  sorry

end total_guppies_l205_205261


namespace least_positive_integer_divisible_by_four_primes_l205_205125

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205125


namespace relay_race_length_correct_l205_205586

def relay_race_length (num_members distance_per_member : ℕ) : ℕ := num_members * distance_per_member

theorem relay_race_length_correct :
  relay_race_length 5 30 = 150 :=
by
  -- The proof would go here
  sorry

end relay_race_length_correct_l205_205586


namespace find_initial_amount_l205_205701

theorem find_initial_amount
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1050)
  (hR : R = 8)
  (hT : T = 5) :
  P = 750 :=
by
  have hSI : P * R * T / 100 = 1050 - P := sorry
  have hFormulaSimplified : P * 0.4 = 1050 - P := sorry
  have hFinal : P * 1.4 = 1050 := sorry
  exact sorry

end find_initial_amount_l205_205701


namespace gain_percentage_l205_205678

theorem gain_percentage (SP1 SP2 CP: ℝ) (h1 : SP1 = 102) (h2 : SP2 = 144) (h3 : SP1 = CP - 0.15 * CP) :
  ((SP2 - CP) / CP) * 100 = 20 := by
sorry

end gain_percentage_l205_205678


namespace log_value_0_or_1_l205_205776

-- Definitions of our sequence and conditions
def geometric_progression (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = a 1 * (a 2 / a 1) ^ n

def first_term_one (a : ℕ → ℝ) : Prop :=
a 1 = 1

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = (a 1) * (1 - (a 2 / a 1) ^ n) / (1 - a 2 / a 1)

def sum_condition (S : ℕ → ℝ) : Prop :=
S 4 = 5 * S 2

def log_condition (a : ℕ → ℝ) : ℝ :=
real.logb 4 (a 3)

-- Main theorem statement
theorem log_value_0_or_1 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : geometric_progression a)
  (h2 : first_term_one a)
  (h3 : sum_first_n_terms S a)
  (h4 : sum_condition S) :
  log_condition a = 0 ∨ log_condition a = 1 :=
sorry

end log_value_0_or_1_l205_205776


namespace hyperbola_and_fixed_line_proof_l205_205386

-- Definitions based on conditions
def center_at_origin (C : Type) [Hyperbola C] : Prop :=
  C.center = (0, 0)

def left_focus (C : Type) [Hyperbola C] : Prop :=
  C.focus1 = (-2 * Real.sqrt 5, 0)

def eccentricity (C : Type) [Hyperbola C] : Prop :=
  C.eccentricity = Real.sqrt 5

def vertices (C : Type) [Hyperbola C] : Prop :=
  C.vertex1 = (-2, 0) ∧ C.vertex2 = (2, 0)

def passing_line_intersections (C : Type) (M N : Type) [Hyperbola C] : Prop :=
  ∃ l : Line, l ⨯ (-4, 0) ∧ l.intersects_hyperbola_left_branch C = {M, N}

def intersection (M N A1 A2 P : Type) [Hyperbola C] : Prop :=
  ∃ A1 A2, A1 = (-2, 0) ∧ A2 = (2, 0) ∧ (MA1.M ∩ NA2.N = P)

-- Proof problem statement
theorem hyperbola_and_fixed_line_proof (C : Type) [Hyperbola C] (M N P : Type)
  (h1 : center_at_origin C) (h2 : left_focus C) (h3 : eccentricity C)
  (h4 : passing_line_intersections C M N) (h5 : vertices C) (h6 : intersection M N C.vertex1 C.vertex2 P) : 
  (C.equation = (∀ x y, C.vertical_transverse_axis x y (x^2 / 4 - y^2 / 16)) ∧ 
   ∀ (x : ℝ), P.x = -1) :=
by sorry

end hyperbola_and_fixed_line_proof_l205_205386


namespace intersection_l205_205412

namespace Proof

def A := {x : ℝ | 0 ≤ x ∧ x ≤ 6}
def B := {x : ℝ | 3 * x^2 + x - 8 ≤ 0}

theorem intersection (x : ℝ) : x ∈ A ∩ B ↔ 0 ≤ x ∧ x ≤ (4:ℝ)/3 := 
by 
  sorry  -- proof placeholder

end Proof

end intersection_l205_205412


namespace equation_of_hyperbola_point_P_on_fixed_line_l205_205374

-- Definition and conditions
def hyperbola_center_origin := (0, 0)
def hyperbola_left_focus := (-2 * Real.sqrt 5, 0)
def hyperbola_eccentricity := Real.sqrt 5
def point_on_line_through_origin := (-4, 0)
def point_M_in_second_quadrant (M : ℝ × ℝ) := M.1 < 0 ∧ M.2 > 0
def vertices_A1_A2 := ((-2, 0), (2, 0))

theorem equation_of_hyperbola (c a b : ℝ) (h_c : c = 2 * Real.sqrt 5)
  (h_e : hyperbola_eccentricity = Real.sqrt 5) (h_a : a = 2) (h_b : b = 4) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 16) = 1) := 
by
  intros x y h_x h_y
  have h : c^2 = a^2 + b^2 := sorry
  have h2 : hyperbola_eccentricity = c / a := sorry
  exact sorry

theorem point_P_on_fixed_line (M N P : ℝ × ℝ) (A1 A2 : ℝ × ℝ)
  (h_vertices : vertices_A1_A2 = (A1, A2))
  (h_M_in_2nd : point_M_in_second_quadrant M) 
  (h_P_intersects : ∀ t : ℝ, P = (t, (A1.2 * A2.1 - A1.1 * A2.2) / (A1.1 - A2.1))) :
  (P.1 = -1) := sorry

end equation_of_hyperbola_point_P_on_fixed_line_l205_205374


namespace value_of_expression_when_x_eq_4_l205_205184

theorem value_of_expression_when_x_eq_4 : (3 * 4 + 4)^2 = 256 := by
  sorry

end value_of_expression_when_x_eq_4_l205_205184


namespace solution_set_of_abs_inequality_l205_205956

theorem solution_set_of_abs_inequality (x : ℝ) : 
  (x < 5 ↔ |x - 8| - |x - 4| > 2) :=
sorry

end solution_set_of_abs_inequality_l205_205956


namespace sum_coeffs_no_x_l205_205584

theorem sum_coeffs_no_x (x y : ℝ) : 
  (∑ t in ((finset.range (5+1)).image (λ k, (binomial 5 k) * (-(5:ℝ))^k)).erase 0, t) = -1024 := 
by 
  sorry

end sum_coeffs_no_x_l205_205584


namespace cards_probability_ratio_l205_205017

theorem cards_probability_ratio :
  let total_cards := 60
  let numbers := 12
  let cards_per_number := total_cards / numbers
  let ways_to_draw_five_cards := Nat.choose total_cards 5
  -- p: probability of drawing five cards all with the same number
  let p := 12 / ways_to_draw_five_cards.toRat
  -- q: probability of drawing four cards with one number and one card with a different number
  let combinations_of_a_and_b := 12 * 11
  let ways_to_choose_cards := 5 * 5
  let ways_to_draw_4_and_1_cards := combinations_of_a_and_b * ways_to_choose_cards
  let q := ways_to_draw_4_and_1_cards / ways_to_draw_five_cards.toRat
  -- final ratio
  q / p = 275 := by
  sorry

end cards_probability_ratio_l205_205017


namespace arithmetic_progression_num_values_l205_205059

theorem arithmetic_progression_num_values (S : ℕ) (d : ℕ) (S_pos : S = 180) (d_pos : d = 3) :
  ∃ (n : ℕ), (∑ i in finset.range n, (a + (i - 1) * d) = S) ∧ n > 1 → (∃ k, a = k) :=
by
  sorry

end arithmetic_progression_num_values_l205_205059


namespace count_ways_to_make_9_cents_l205_205964

theorem count_ways_to_make_9_cents : 
  ∃ (x y z : ℕ), (5 * x + 2 * y + z = 9) ∧ (x ≤ 1) ∧ (y ≤ 4) ∧ (z ≤ 8) → 
  (finset.univ.filter (λ triplet : ℕ × ℕ × ℕ, 
    let (x, y, z) := triplet in (5 * x + 2 * y + z = 9 ∧ x ≤ 1 ∧ y ≤ 4 ∧ z ≤ 8))).card = 7 :=
by
  sorry

end count_ways_to_make_9_cents_l205_205964


namespace domain_sqrt_2_sin_minus_1_domain_sqrt_tan_minus_sqrt3_l205_205312

-- Prove the domain for y = sqrt(2 * sin x - 1)
theorem domain_sqrt_2_sin_minus_1 (x : ℝ) (k : ℤ) :
  2 * sin x - 1 ≥ 0 ↔ (∃ k : ℤ, x ∈ set.Icc (π / 6 + 2 * k * π) (5 * π / 6 + 2 * k * π)) := 
sorry

-- Prove the domain for y = sqrt(tan x - sqrt 3)
theorem domain_sqrt_tan_minus_sqrt3 (x : ℝ) (k : ℤ) :
  tan x - sqrt 3 ≥ 0 ↔ (∃ k : ℤ, x ∈ set.Ico (π / 3 + k * π) (π / 2 + k * π)) :=
sorry

end domain_sqrt_2_sin_minus_1_domain_sqrt_tan_minus_sqrt3_l205_205312


namespace ryan_bus_meet_exactly_once_l205_205013

-- Define respective speeds of Ryan and the bus
def ryan_speed : ℕ := 6 
def bus_speed : ℕ := 15 

-- Define bench placement and stop times
def bench_distance : ℕ := 300 
def regular_stop_time : ℕ := 45 
def extra_stop_time : ℕ := 90 

-- Initial positions
def ryan_initial_position : ℕ := 0
def bus_initial_position : ℕ := 300

-- Distance function D(t)
noncomputable def distance_at_time (t : ℕ) : ℤ :=
  let bus_travel_time : ℕ := 15  -- time for bus to travel 225 feet
  let bus_stop_time : ℕ := 45  -- time for bus to stop during regular stops
  let extended_stop_time : ℕ := 90  -- time for bus to stop during 3rd bench stops
  sorry -- calculation of distance function

-- Problem to prove: Ryan and the bus meet exactly once
theorem ryan_bus_meet_exactly_once : ∃ t₁ t₂ : ℕ, t₁ ≠ t₂ ∧ distance_at_time t₁ = 0 ∧ distance_at_time t₂ ≠ 0 := 
  sorry

end ryan_bus_meet_exactly_once_l205_205013


namespace find_lambda_l205_205416

-- Define what it means for vector to be perpendicular to another
def is_perpendicular {α : Type*} [inner_product_space ℝ α] (x y : α) : Prop :=
  inner x y = 0

-- Define the problem conditions
variables {V : Type*} [inner_product_space ℝ V]
variables {e1 e2 : V}
variables (h_unit_e1 : ∥e1∥ = 1) (h_unit_e2 : ∥e2∥ = 1)
variables (h_angle : real.angle.e1 e2 = real.angle π/3)  -- 60 degrees in radians
variables (λ : ℝ)

-- The proof goal
theorem find_lambda (h_perpendicular : is_perpendicular e1 (λ • e2 - e1)) : λ = 2 :=
by {
    -- Detailed proof would go here, but it's omitted for this example
    sorry
}

end find_lambda_l205_205416


namespace contractor_absent_days_l205_205986

theorem contractor_absent_days (x y : ℕ) (h1 : x + y = 30) (h2 : 25 * x - 7.5 * y = 425) :
  y = 10 :=
by
  sorry

end contractor_absent_days_l205_205986


namespace max_a_l205_205330

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem max_a (a : ℝ) :
  (∀ x ∈ set.Icc (1 / 2 : ℝ) (2 : ℝ), f x ≥ a) ↔ a ≤ 0 :=
by
  sorry

end max_a_l205_205330


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205138

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205138


namespace cos_angle_between_AB_AC_is_zero_l205_205743

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_between (A B : Point3D) : ℝ × ℝ × ℝ :=
  (B.x - A.x, B.y - A.y, B.z - A.z)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def cos_angle_between (A B C : Point3D) : ℝ :=
  let AB := vector_between A B
  let AC := vector_between A C
  dot_product AB AC / (magnitude AB * magnitude AC)

theorem cos_angle_between_AB_AC_is_zero :
  cos_angle_between ⟨2, 1, -1⟩ ⟨6, -1, -4⟩ ⟨4, 2, 1⟩ = 0 :=
by
  sorry

end cos_angle_between_AB_AC_is_zero_l205_205743


namespace problem_negation_l205_205518

noncomputable theory

open Classical

theorem problem_negation (x y : ℝ) :
  (∃ x y : ℝ, (x * y ≠ 0) ∧ (x ≠ 0)) ↔ ¬ (∀ x y : ℝ, (x * y = 0) → (x = 0)) := by
  sorry

end problem_negation_l205_205518


namespace find_x_in_triangle_XYZ_l205_205859

theorem find_x_in_triangle_XYZ (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ) (hx : y = 7) (hz : z = 6) (hcos : cos_Y_minus_Z = 47 / 64) : 
    ∃ x : ℝ, x = Real.sqrt 63.75 :=
by
  -- The proof will go here, but it is skipped for now.
  sorry

end find_x_in_triangle_XYZ_l205_205859


namespace limit_floor_function_l205_205205

theorem limit_floor_function (x : ℝ) : 
  ∃ l : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, abs x < δ → abs (x * (⟦1/x⟧) - l) < ε) ↔ l = 1 := 
by { sorry }

end limit_floor_function_l205_205205


namespace four_digit_even_numbers_count_balls_in_boxes_count_l205_205997

-- Prove the number of four-digit even numbers that can be formed using the digits 1, 2, 3, 4, 5 without repetition is 48
theorem four_digit_even_numbers_count (digits : finset ℕ) (even_digits : finset ℕ) :
  digits = {1, 2, 3, 4, 5} →
  even_digits = {2, 4} →
  (even_digits.card * (digits.erase even_digits.min' (by linarith)).card * (digits.erase even_digits.min' (by linarith)).card * 
    (digits.erase even_digits.min' (by linarith)).erase (digits.erase even_digits.min' (by linarith)).min' (by linarith).card = 48 :=
by sorry

-- Prove the number of ways to place 4 different balls into 4 boxes (1, 2, 3, 4) such that exactly one box is empty is 144
theorem balls_in_boxes_count (balls : finset ℕ) (boxes : finset ℕ) :
  balls = {1, 2, 3, 4} →
  boxes = {1, 2, 3, 4} →
  (balls.card.choose 2 * (boxes.card - 1).fact = 144 :=
by sorry

end four_digit_even_numbers_count_balls_in_boxes_count_l205_205997


namespace count_k_digit_numbers_l205_205526

theorem count_k_digit_numbers (k : ℕ) :
  ∃ (count : ℕ), count = 3 * (k - 1) ∧
  count = ∑ n in range (10^(k-1) * 9), 
    (n % 5 = 0 ∧
     (∀ d in (digits 10 n), odd d) ∧
     (∀ d in (digits 10 (n / 5)), odd d)) :=
sorry

end count_k_digit_numbers_l205_205526


namespace sakshi_work_days_l205_205917

theorem sakshi_work_days (tanya_days : ℕ) (efficiency_increase : ℕ) (efficiency_fraction : ℚ) : 
  efficiency_increase = 25 → tanya_days = 16 → efficiency_fraction = 1.25 → 
  let sakshi_days := tanya_days * efficiency_fraction in 
  sakshi_days = 20 := by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sakshi_work_days_l205_205917


namespace last_number_is_two_l205_205030

theorem last_number_is_two (A B C D : ℝ)
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) :
  D = 2 :=
sorry

end last_number_is_two_l205_205030


namespace compute_fraction_l205_205287

theorem compute_fraction :
  ( (11^4 + 400) * (25^4 + 400) * (37^4 + 400) * (49^4 + 400) * (61^4 + 400) ) /
  ( (5^4 + 400) * (17^4 + 400) * (29^4 + 400) * (41^4 + 400) * (53^4 + 400) ) = 799 := 
by
  sorry

end compute_fraction_l205_205287


namespace translate_A_coordinates_l205_205851

-- Definitions
def A_initial : ℝ × ℝ := (-3, 2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 - d)

-- Final coordinates after transformation
def A' : ℝ × ℝ :=
  let A_translated := translate_right A_initial 4
  translate_down A_translated 3

-- Proof statement
theorem translate_A_coordinates :
  A' = (1, -1) :=
by
  simp [A', translate_right, translate_down, A_initial]
  sorry

end translate_A_coordinates_l205_205851


namespace square_areas_l205_205852

theorem square_areas (z : ℂ) 
  (h1 : ¬ (2 : ℂ) * z^2 = z)
  (h2 : ¬ (3 : ℂ) * z^3 = z)
  (sz : (3 * z^3 - z) = (I * (2 * z^2 - z)) ∨ (3 * z^3 - z) = (-I * (2 * z^2 - z))) :
  ∃ (areas : Finset ℝ), areas = {85, 4500} :=
by {
  sorry
}

end square_areas_l205_205852


namespace problem_statement_l205_205484

noncomputable def AP : ℝ := sorry
noncomputable def BP : ℝ := sorry
noncomputable def CP : ℝ := sorry
noncomputable def DP : ℝ := sorry

theorem problem_statement (AP BP CP DP : ℝ) (condition1 : true) (condition2 : true) :
  AP + BP ≠ 0 → CP + DP ≠ 0 → (AP + BP) / (CP + DP) = sqrt 2 - 1 :=
by
  intros h1 h2
  sorry

end problem_statement_l205_205484


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205088

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205088


namespace meteor_point_movement_to_line_l205_205221

theorem meteor_point_movement_to_line (M : Type) [point M] [line M]
  (meteor : M → Prop) (trail : M → Prop):
  (∀ p : M, meteor p → point p) → 
  (∀ p1 p2 : M, meteor p1 ∧ meteor p2 → line (p1, p2) → trail (p1, p2)) → 
  ∃ L : M → M → Prop, ∀ p1 p2 : M, meteor p1 ∧ meteor p2 → L p1 p2 :=
begin
  sorry
end

end meteor_point_movement_to_line_l205_205221


namespace no_three_consecutive_geometric_l205_205906

open Nat

def a (n : ℕ) : ℤ := 3^n - 2^n

theorem no_three_consecutive_geometric :
  ∀ (k : ℕ), ¬ (∃ n m : ℕ, m = n + 1 ∧ k = m + 1 ∧ (a n) * (a k) = (a m)^2) :=
by
  sorry

end no_three_consecutive_geometric_l205_205906


namespace brothers_travel_distance_l205_205570

theorem brothers_travel_distance
  (x : ℝ)
  (hb_x : (120 : ℝ) / (x : ℝ) - 4 = (120 : ℝ) / (x + 40))
  (total_time : 2 = 2) :
  x = 20 ∧ (x + 40) = 60 :=
by
  -- we need to prove the distances
  sorry

end brothers_travel_distance_l205_205570


namespace correct_statements_l205_205267

theorem correct_statements :
  let f1 := λ x : ℝ, |x| / x,
      g := λ x : ℝ, if x ≥ 0 then 1 else -1,
      f2 := λ x : ℝ, x^2 + 2 + 1 / (x^2 + 2),
      f3 := λ x : ℝ, |x - 1| - |x|
  in (∀ x : ℝ, f1 x = g x) ∧
     (∀ y : ℝ, y = 1 ↔ ∃ x : ℝ, f2 x = y) ∧
     f2 = 2 ∧
     f3 (f3 (1 / 2)) = 1 :=
begin
  sorry
end

end correct_statements_l205_205267


namespace similar_triangles_l205_205947

theorem similar_triangles (A B C D O : Type) [EuclideanGeometry O] 
    (is_trapezoid : trapezoid A B C D)
    (O_intersect : meets O B O D ∧ parallel A B C D) : 
    similar (triangle O B C) (triangle O A D) :=
sorry

end similar_triangles_l205_205947


namespace tan_two_x_is_odd_l205_205655

noncomputable def tan_two_x (x : ℝ) : ℝ := Real.tan (2 * x)

theorem tan_two_x_is_odd :
  ∀ x : ℝ,
  (∀ k : ℤ, x ≠ (k * Real.pi / 2) + (Real.pi / 4)) →
  tan_two_x (-x) = -tan_two_x x :=
by
  sorry

end tan_two_x_is_odd_l205_205655


namespace diet_start_months_ago_l205_205470

-- Define the problem conditions as variables
variable (initial_weight current_weight future_weight: ℝ)
variable (months_future: ℝ)
variable (monthly_rate: ℝ)

-- Given conditions as hypotheses
def initial_condition := initial_weight = 222
def current_condition := current_weight = 198
def future_condition := future_weight = 170
def consistent_rate_condition := future_weight = current_weight - months_future * monthly_rate

-- The theorem we need to prove
theorem diet_start_months_ago
  (h_init: initial_condition)
  (h_curr: current_condition)
  (h_future: future_condition)
  (h_rate: consistent_rate_condition)
  (monthly_rate_value: monthly_rate = (198 - 170) / 3.5): 
  (222 - 198) / monthly_rate_value = 3 :=
by 
  sorry

end diet_start_months_ago_l205_205470


namespace distance_is_correct_l205_205644

noncomputable def distance_to_place (rowing_speed current_speed total_time : ℝ) : ℝ :=
let downstream_speed := rowing_speed + current_speed,
    upstream_speed := rowing_speed - current_speed,
    T1 := total_time / (1 + downstream_speed / upstream_speed) in
downstream_speed * T1

theorem distance_is_correct :
  distance_to_place 8 2 2 = 7.5 :=
by 
  let downstream_speed := 8 + 2
  let upstream_speed := 8 - 2
  let T1 := 2 / (1 + downstream_speed / upstream_speed)
  let D := downstream_speed * T1
  have h1 : downstream_speed = 10 := rfl
  have h2 : upstream_speed = 6 := rfl
  have h3 : T1 = 2 / (1 + 10 / 6) := rfl
  have h4 : T1 = 6 / 16 := by 
    sorry
  have h5 : D = 7.5 := by 
    sorry
  show D = 7.5 from h5

end distance_is_correct_l205_205644


namespace lines_parallel_l205_205455

noncomputable def line_plane (a b c : Prop) : Prop :=
  in_same_plane a b c ∧ (perpendicular a b) ∧ (perpendicular a c)

theorem lines_parallel (a b c : Prop) :
  line_plane a b c → parallel b c :=
by
  intro h
  cases h with _ hp
  cases hp with hab hac
  sorry

end lines_parallel_l205_205455


namespace units_digit_47_power_47_l205_205615

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l205_205615


namespace production_value_n_l205_205442

theorem production_value_n :
  -- Definitions based on conditions:
  (∀ a b : ℝ,
    (120 * a + 120 * b) / 60 = 6 ∧
    (100 * a + 100 * b) / 30 = 30) →
  (∃ n : ℝ, 80 * 3 * (a + b) = 480 * a + n * b) →
  n = 120 :=
by
  sorry

end production_value_n_l205_205442


namespace sum_of_other_six_interior_angles_l205_205432

-- Define a variable for the total sum of interior angles of a heptagon
def sum_interior_angles_heptagon : ℝ := 900

-- Define a variable for the given interior angle
def given_angle : ℝ := 100

-- Statement that the sum of the other six interior angles is 800
theorem sum_of_other_six_interior_angles :
  sum_interior_angles_heptagon - given_angle = 800 := 
begin
  sorry
end

end sum_of_other_six_interior_angles_l205_205432


namespace triangle_intersection_area_l205_205595

theorem triangle_intersection_area
  (X Y E F : Type)
  (XY YE FX EX FY : ℝ)
  (h₁ : XY = 12)
  (h₂ : YE = 15)
  (h₃ : FX = 15)
  (h₄ : EX = 20)
  (h₅ : FY = 20)
  (congruent_triangles : ∃ (T : Type), T = ∆ XYE ∧ T = ∆ XYF) :
  let p := 144
  let q := 7
  p + q = 151 :=
by
  sorry

end triangle_intersection_area_l205_205595


namespace least_pos_int_div_by_four_distinct_primes_l205_205099

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205099


namespace proportion_of_solution_x_in_mixture_l205_205648

-- Definitions for the conditions in given problem
def solution_x_contains_perc_a : ℚ := 0.20
def solution_y_contains_perc_a : ℚ := 0.30
def solution_z_contains_perc_a : ℚ := 0.40

def solution_y_to_z_ratio : ℚ := 3 / 2
def final_mixture_perc_a : ℚ := 0.25

-- Proving the proportion of solution x in the mixture equals 9/14
theorem proportion_of_solution_x_in_mixture
  (x y z : ℚ) (k : ℚ) (hx : x = 9 * k) (hy : y = 3 * k) (hz : z = 2 * k) :
  solution_x_contains_perc_a * x + solution_y_contains_perc_a * y + solution_z_contains_perc_a * z
  = final_mixture_perc_a * (x + y + z) →
  x / (x + y + z) = 9 / 14 :=
by
  intros h
  -- leaving the proof as a placeholder
  sorry

end proportion_of_solution_x_in_mixture_l205_205648


namespace equation_b_not_symmetric_about_x_axis_l205_205980

def equationA (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equationB (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equationC (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1
def equationD (x y : ℝ) : Prop := x + y^2 = -1

def symmetric_about_x_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, f x y ↔ f x (-y)

theorem equation_b_not_symmetric_about_x_axis : 
  ¬ symmetric_about_x_axis (equationB) :=
sorry

end equation_b_not_symmetric_about_x_axis_l205_205980


namespace smallest_solution_l205_205207

noncomputable def f (x : ℝ) : ℝ :=
  (-Real.logBase 2 (105 + 2 * x * Real.sqrt (x + 19)) ^ 3 + 
   |Real.logBase 2 ((105 + 2 * x * Real.sqrt (x + 19)) / (x ^ 2 + x + 3) ^ 4)|) /
  (9 * Real.logBase 5 (76 + 2 * x * Real.sqrt (x + 19)) - 4 * Real.logBase 2 (105 + 2 * x * Real.sqrt (x + 19)))

theorem smallest_solution :
  f ((-21 + Real.sqrt 33) / 2) ≥ 0 :=
sorry

end smallest_solution_l205_205207


namespace Distance_from_A_to_Plane_BCM_l205_205837

open Real

-- Defining points and conditions
structure Triangle :=
(A B C : Point)

def Triangle.AB_dist (T : Triangle) : ℝ := dist T.A T.B
def Point.distance_to_plane (P : Point) (u v w : ℝ) := sqrt (u * u + v * v + w * w)

-- Setting up the given problem
theorem Distance_from_A_to_Plane_BCM (A B C M: Point) 
(h1 : angle A C B = π / 2) 
(h2 : angle A B C = π / 6) 
(h3 : dist A C = 1) 
(h4 : M = midpoint A B) 
(h5 : dist A B = sqrt 2) : 
Point.distance_to_plane A B C M = sqrt(6) / 3 := 
sorry

-- Utility functions would include definitions for distance, midpoint, angle calculation, etc.

end Distance_from_A_to_Plane_BCM_l205_205837


namespace passes_after_6_l205_205965

-- Define the sequence a_n where a_n represents the number of ways the ball is in A's hands after n passes
def passes : ℕ → ℕ
| 0       => 1       -- Initially, the ball is in A's hands (1 way)
| (n + 1) => 2^n - passes n

-- Theorem to prove the number of different passing methods after 6 passes
theorem passes_after_6 : passes 6 = 22 := by
  sorry

end passes_after_6_l205_205965


namespace smallest_pos_int_div_by_four_primes_l205_205116

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205116


namespace problem1_problem2_l205_205708

noncomputable def exp1 : ℝ :=
  (1 / 2) * log 25 / log 10 + log 2 / log 10 - log (sqrt 0.1) / log 10 - (log 9 / log 2) * (log 2 / log 3)

noncomputable def exp2 : ℝ :=
  real.exp (log 64 / 3) - real.exp 0 + real.exp (-log 16 / 2) + log 20 / log 10 + (log 25 / log 100)

theorem problem1 : exp1 = -1 / 2 := by
  sorry

theorem problem2 : exp2 = 9 := by
  sorry

end problem1_problem2_l205_205708


namespace least_number_to_add_l205_205203

theorem least_number_to_add (k : ℕ) (h : 1019 % 25 = 19) : (1019 + k) % 25 = 0 ↔ k = 6 :=
by
  sorry

end least_number_to_add_l205_205203


namespace range_of_a2_minus_b3_l205_205643

def is_prime (n : ℕ) : Prop := Nat.Prime n

def a_in_range (a : ℕ) : Prop := a > 49 ∧ a < 61 ∧ is_prime a
def b_in_range (b : ℕ) : Prop := b > 59 ∧ b < 71 ∧ is_prime b

theorem range_of_a2_minus_b3 :
  ∃ (lower_bound upper_bound : ℤ),
  lower_bound = -297954 ∧ upper_bound = -223500 ∧
  ∀ (a b : ℕ),
  a_in_range a → b_in_range b →
  a^2 - b^3 ∈ Set.Icc lower_bound upper_bound :=
by
  use -297954
  use -223500
  split
  · rfl
  split
  · rfl
  intros a b ha hb
  have ha_prime : is_prime a := ha.2.2
  have hb_prime : is_prime b := hb.2.2
  sorry

end range_of_a2_minus_b3_l205_205643


namespace tim_total_spent_l205_205590

variables 
  (price_sandwich : ℚ := 10.50) 
  (discount_sandwich : ℚ := 0.15) 
  (price_side_salad : ℚ := 5.25) 
  (tax_side_salad : ℚ := 0.07) 
  (price_soda : ℚ := 1.75) 
  (tax_soda : ℚ := 0.05) 
  (tip : ℚ := 0.20)

def total_spent 
  (price_sandwich discount_sandwich price_side_salad tax_side_salad price_soda tax_soda tip : ℚ) 
  : ℚ := 
  let discounted_price_sandwich := price_sandwich * (1 - discount_sandwich) in
  let total_side_salad := price_side_salad * (1 + tax_side_salad) in
  let total_soda := price_soda * (1 + tax_soda) in
  let subtotal := discounted_price_sandwich + total_side_salad + total_soda in
  let tip_amount := subtotal * tip in
  subtotal + tip_amount

theorem tim_total_spent 
  : total_spent price_sandwich discount_sandwich price_side_salad tax_side_salad price_soda tax_soda tip = 19.66 := 
by
  sorry

end tim_total_spent_l205_205590


namespace least_pos_int_div_by_four_distinct_primes_l205_205167

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205167


namespace counters_arrangement_impossible_l205_205677

theorem counters_arrangement_impossible (n : ℕ) :
∃ (a : ℕ → ℕ), (∀ x ∈ finset.range (n+1), a x = n) →
¬ (∃ (arrangement : list ℕ),
  (∀ x ∈ finset.range n, (∀ (i j : ℕ),
    i < j → i < arrangement.index_of x → arrangement.index_of x < j → j + 1 = arrangement.index_of x + x)) :=
begin
  sorry
end

end counters_arrangement_impossible_l205_205677


namespace ratio_of_areas_l205_205666

noncomputable def large_square_side : ℝ := 4
noncomputable def large_square_area : ℝ := large_square_side ^ 2
noncomputable def inscribed_square_side : ℝ := 1  -- As it fits in the definition from the problem description
noncomputable def inscribed_square_area : ℝ := inscribed_square_side ^ 2

theorem ratio_of_areas :
  (inscribed_square_area / large_square_area) = 1 / 16 :=
by
  sorry

end ratio_of_areas_l205_205666


namespace mode_correct_l205_205654

noncomputable def visual_acuity_data : list (ℝ × ℕ) :=
  [(4.3, 2), (4.4, 3), (4.5, 6), (4.6, 9), (4.7, 12), (4.8, 8), (4.9, 5), (5.0, 3)]

def mode_of_visual_acuity (data : list (ℝ × ℕ)) : ℝ :=
  (data.max_by (λ x => x.snd)).fst

theorem mode_correct :
  mode_of_visual_acuity visual_acuity_data = 4.7 :=
sorry

end mode_correct_l205_205654


namespace proof_f_l205_205434

noncomputable def f (f_prime_one : ℝ) (x : ℝ) : ℝ := f_prime_one * x^3 - 2 * x^2 + 3

theorem proof_f'_2 (f_prime_one : ℝ) (h : f_prime_one = 2) : 
  let f' (x : ℝ) := 3 * f_prime_one * x^2 - 4 * x
  in f' 2 = 16 :=
by
  have f_prime_def : f' 2 = 3 * f_prime_one * 2^2 - 4 * 2 := by sorry
  rw [h] at f_prime_def
  have : f' 2 = 24 - 8 := by sorry
  exact this.symm

end proof_f_l205_205434


namespace count_triangles_l205_205514

theorem count_triangles :
  ∀ (a b c : ℕ), 
  (a = 10) → (b = 11) → (c = 12) →
  let total_points := a + b + c in
  let total_combinations := Nat.choose total_points 3 in
  let invalid_combinations := Nat.choose a 3 + Nat.choose b 3 + Nat.choose c 3 in
  total_combinations - invalid_combinations = 4951 :=
by
  intros a b c ha hb hc
  simp [ha, hb, hc]
  let total_points := 10 + 11 + 12
  let total_combinations := Nat.choose total_points 3
  let invalid_combinations := Nat.choose 10 3 + Nat.choose 11 3 + Nat.choose 12 3
  calc
    total_combinations - invalid_combinations = 5456 - 505 : by simp [total_combinations, invalid_combinations]
    ... = 4951 : by simp

end count_triangles_l205_205514


namespace hyperbola_and_fixed_line_proof_l205_205384

-- Definitions based on conditions
def center_at_origin (C : Type) [Hyperbola C] : Prop :=
  C.center = (0, 0)

def left_focus (C : Type) [Hyperbola C] : Prop :=
  C.focus1 = (-2 * Real.sqrt 5, 0)

def eccentricity (C : Type) [Hyperbola C] : Prop :=
  C.eccentricity = Real.sqrt 5

def vertices (C : Type) [Hyperbola C] : Prop :=
  C.vertex1 = (-2, 0) ∧ C.vertex2 = (2, 0)

def passing_line_intersections (C : Type) (M N : Type) [Hyperbola C] : Prop :=
  ∃ l : Line, l ⨯ (-4, 0) ∧ l.intersects_hyperbola_left_branch C = {M, N}

def intersection (M N A1 A2 P : Type) [Hyperbola C] : Prop :=
  ∃ A1 A2, A1 = (-2, 0) ∧ A2 = (2, 0) ∧ (MA1.M ∩ NA2.N = P)

-- Proof problem statement
theorem hyperbola_and_fixed_line_proof (C : Type) [Hyperbola C] (M N P : Type)
  (h1 : center_at_origin C) (h2 : left_focus C) (h3 : eccentricity C)
  (h4 : passing_line_intersections C M N) (h5 : vertices C) (h6 : intersection M N C.vertex1 C.vertex2 P) : 
  (C.equation = (∀ x y, C.vertical_transverse_axis x y (x^2 / 4 - y^2 / 16)) ∧ 
   ∀ (x : ℝ), P.x = -1) :=
by sorry

end hyperbola_and_fixed_line_proof_l205_205384


namespace ellipse_properties_l205_205771

noncomputable def ellipse_eq (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem ellipse_properties
  (a b e : ℝ)
  (h_a_pos : a > 0)
  (h_a_gt_b : a > b)
  (h_b_pos : b > 0)
  (h_ecc : e = real.sqrt (1 - (b^2 / a^2)))
  (h_pass1 : ellipse_eq a b 2 0)
  (h_pass2 : ∃ e, ellipse_eq a b 1 e):
  (a = 2 ∧ b = 1 ∧ ellipse_eq 2 1 x y) ∧
  (∀ (m : ℝ) (A B C D P Q : ℝ × ℝ),
    A ∈ elm_intersects m (1, 0) ∧
    B ∈ elm_intersects m (1, 0) ∧
    C ∈ elm_intersects (1 / m) (1, 0) ∧
    D ∈ elm_intersects (1 / m) (1, 0) ∧
    P = midpoint A B ∧
    Q = midpoint C D →
    (area_ratio (0, 0) P Q (1, 0) = 4)) :=
begin
  sorry,
end

end ellipse_properties_l205_205771


namespace prime_numbers_eq_l205_205724

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_eq 
  (p q r : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (h : p * (p - 7) + q * (q - 7) = r * (r - 7)) :
  (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 5 ∧ r = 7) ∨
  (p = 7 ∧ q = 5 ∧ r = 5) ∨ (p = 5 ∧ q = 7 ∧ r = 5) ∨
  (p = 5 ∧ q = 7 ∧ r = 2) ∨ (p = 7 ∧ q = 5 ∧ r = 2) ∨
  (p = 7 ∧ q = 3 ∧ r = 3) ∨ (p = 3 ∧ q = 7 ∧ r = 3) ∨
  (∃ (a : ℕ), is_prime a ∧ p = a ∧ q = 7 ∧ r = a) ∨
  (∃ (a : ℕ), is_prime a ∧ p = 7 ∧ q = a ∧ r = a) :=
sorry

end prime_numbers_eq_l205_205724


namespace max_value_of_g_on_interval_l205_205798

theorem max_value_of_g_on_interval : 
  ∀ (x : ℝ), 
  (0 ≤ x ∧ x ≤ 1) → 
  g x ≤ 0 := sorry

def g (x : ℝ) := x * (x^2 - 1)

end max_value_of_g_on_interval_l205_205798


namespace smallest_solution_l205_205208

noncomputable def expr (x : ℝ) : ℝ := 
  (-Real.log (105 + 2 * x * Real.sqrt (x + 19)) / Real.log 2)^3 + 
  Real.abs ((Real.log (105 + 2 * x * Real.sqrt (x + 19)) / Real.log 2) - 4 * (Real.log (x^2 + x + 3) / Real.log 2)) / 
  (9 * (Real.log (76 + 2 * x * Real.sqrt (x + 19)) / Real.log 5) - 4 * (Real.log (105 + 2 * x * Real.sqrt (x + 19)) / Real.log 2))

theorem smallest_solution :
  ∀ x : ℝ, expr x ≥ 0 → x ≥ (⟨ (-21 + Real.sqrt 33) / 2 , sorry⟩) := 
sorry

end smallest_solution_l205_205208


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205174

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205174


namespace triangle_area_l205_205454

theorem triangle_area (sqr_area : ℝ) (small_sq_side : ℝ) (base_length : ℝ) (altitude_length : ℝ) :
  sqr_area = 36 ∧ small_sq_side = 2 ∧ base_length = 2 ∧ altitude_length = 7 →
  (1/2 * base_length * altitude_length) = 7 :=
by
  intros h
  cases h with h_sqr_area h_remaining
  cases h_remaining with h_small_sq1 h_remaining2
  cases h_remaining2 with h_base_length h_altitude_length
  sorry

end triangle_area_l205_205454


namespace number_of_adults_swimming_per_day_l205_205863

-- Define the conditions as variables and constants
constant cost_per_kid : ℕ := 3
constant cost_per_adult : ℕ := 6  -- twice the amount for kids
constant num_kids_per_day : ℕ := 8
constant weekly_income : ℕ := 588

-- Prove the number of adults swimming per day
theorem number_of_adults_swimming_per_day (A : ℕ) : 
  (7 * (8 * cost_per_kid + A * cost_per_adult) = weekly_income) → 
  A = 10 :=
by
  sorry

end number_of_adults_swimming_per_day_l205_205863


namespace possible_omega_values_l205_205801

theorem possible_omega_values (ω : ℕ) (hω : ω > 0 ∧ ω ≤ 2023) :
  (∀ x : ℝ, sin (ω * x) + sin (2 * x) < 2) → 1770.

end possible_omega_values_l205_205801


namespace find_percentage_l205_205215

noncomputable def percentage_solve (x : ℝ) : Prop :=
  0.15 * 40 = (x / 100) * 16 + 2

theorem find_percentage (x : ℝ) (h : percentage_solve x) : x = 25 :=
by
  sorry

end find_percentage_l205_205215


namespace next_special_year_after_2009_l205_205962

def is_special_year (n : ℕ) : Prop :=
  ∃ d1 d2 d3 d4 : ℕ,
    (2000 ≤ n) ∧ (n < 10000) ∧
    (d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n) ∧
    (d1 ≠ 0) ∧
    ∀ (p q r s : ℕ),
    (p * 1000 + q * 100 + r * 10 + s < n) →
    (p ≠ d1 ∨ q ≠ d2 ∨ r ≠ d3 ∨ s ≠ d4)

theorem next_special_year_after_2009 : ∃ y : ℕ, is_special_year y ∧ y > 2009 ∧ y = 2022 :=
  sorry

end next_special_year_after_2009_l205_205962


namespace distance_from_P_to_AD_l205_205929

-- Definitions of points and circles
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 0, y := 4}
def D : Point := {x := 0, y := 0}
def C : Point := {x := 4, y := 0}
def M : Point := {x := 2, y := 0}
def radiusM : ℝ := 2
def radiusA : ℝ := 4

-- Definition of the circles
def circleM (P : Point) : Prop := (P.x - M.x)^2 + P.y^2 = radiusM^2
def circleA (P : Point) : Prop := P.x^2 + (P.y - A.y)^2 = radiusA^2

-- Definition of intersection point \(P\) of the two circles
def is_intersection (P : Point) : Prop := circleM P ∧ circleA P

-- Distance from point \(P\) to line \(\overline{AD}\) computed as the x-coordinate
def distance_to_line_AD (P : Point) : ℝ := P.x

-- The theorem to prove
theorem distance_from_P_to_AD :
  ∃ P : Point, is_intersection P ∧ distance_to_line_AD P = 16/5 :=
by {
  -- Use "sorry" as the proof placeholder
  sorry
}

end distance_from_P_to_AD_l205_205929


namespace max_ab_l205_205778

-- Definitions based on conditions
def circleC1 (a : ℝ) : set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 + 2)^2 = 4}
def circleC2 (b : ℝ) : set (ℝ × ℝ) := {p | (p.1 + b)^2 + (p.2 + 2)^2 = 1}

theorem max_ab {a b : ℝ} 
  (h₁ : (∀ p, p ∈ circleC1 a → p.1 ≠ p.1))
  (h₂ : (∀ p, p ∈ circleC2 b → p.1 ≠ p.1))
  (h_tangent : abs (a + b) = 3) : 
  ab <= (3/2) * (3/2) :=
by sorry

end max_ab_l205_205778


namespace problem_1_false_problem_2_false_problem_3_false_problem_4_false_l205_205298

-- Statement for problem (1)
theorem problem_1_false (α β : ℝ) : ¬(sin α ^ 2 + cos β ^ 2 = 1) := by
  sorry

-- Statement for problem (2)
theorem problem_2_false (α : ℝ) : ¬(∀ α, tan α = sin α / cos α) := by
  sorry

-- Statement for problem (3)
theorem problem_3_false (α : ℝ) : ¬(sin (Real.pi + α) = -sin α → α < Real.pi / 2) := by
  sorry

-- Statement for problem (4)
theorem problem_4_false (n : ℤ) : ¬(cos (n * Real.pi - θ) = 1 / 3 → cos θ = 1 / 3) := by
  sorry

end problem_1_false_problem_2_false_problem_3_false_problem_4_false_l205_205298


namespace final_weight_of_box_l205_205866

theorem final_weight_of_box (w1 w2 w3 w4 : ℝ) (h1 : w1 = 2) (h2 : w2 = 3 * w1) (h3 : w3 = w2 + 2) (h4 : w4 = 2 * w3) : w4 = 16 :=
by
  sorry

end final_weight_of_box_l205_205866


namespace alice_ride_top_speed_l205_205705

-- Define the conditions
variables (x y : Real) -- x is the hours at 25 mph, y is the hours at 15 mph.
def distance_eq : Prop := 25 * x + 15 * y + 10 * (9 - x - y) = 162
def time_eq : Prop := x + y ≤ 9

-- Define the final answer
def final_answer : Prop := x = 2.7

-- The statement to prove
theorem alice_ride_top_speed : distance_eq x y ∧ time_eq x y → final_answer x := sorry

end alice_ride_top_speed_l205_205705


namespace minimum_guests_needed_l205_205946

theorem minimum_guests_needed (total_food : ℕ) (max_food_per_guest : ℕ) (guests_needed : ℕ) : 
  total_food = 323 → max_food_per_guest = 2 → guests_needed = Nat.ceil (323 / 2) → guests_needed = 162 :=
by
  intros
  sorry

end minimum_guests_needed_l205_205946


namespace hose_filling_time_l205_205682

theorem hose_filling_time :
  ∀ (P A B C : ℝ), 
  (P / 3 = A + B) →
  (P / 5 = A + C) →
  (P / 4 = B + C) →
  (P / (A + B + C) = 2.55) :=
by
  intros P A B C hAB hAC hBC
  sorry

end hose_filling_time_l205_205682


namespace inequality_proof_l205_205954

theorem inequality_proof (x y z : ℝ) (h : x^2 + y^2 + z^2 = 3) :
  x^3 - (y^2 + yz + z^2)x + yz(y + z) ≤ 3 * Real.sqrt 3 ∧ 
  (x^3 - (y^2 + yz + z^2)x + yz(y + z) = 3 * Real.sqrt 3 → (x, y, z) = (Real.sqrt 3, 0, 0)) :=
by
  sorry

end inequality_proof_l205_205954


namespace length_DE_l205_205544

theorem length_DE (AB : ℝ) (h_base : AB = 15) (DE_parallel : ∀ x y z : Triangle, True) (area_ratio : ℝ) (h_area_ratio : area_ratio = 0.25) : 
  ∃ DE : ℝ, DE = 7.5 :=
by
  sorry

end length_DE_l205_205544


namespace borrowed_amount_correct_l205_205637

variables (monthly_payment : ℕ) (months : ℕ) (total_payment : ℕ) (borrowed_amount : ℕ)

def total_payment_calculation (monthly_payment : ℕ) (months : ℕ) : ℕ :=
  monthly_payment * months

theorem borrowed_amount_correct :
  monthly_payment = 15 →
  months = 11 →
  total_payment = total_payment_calculation monthly_payment months →
  total_payment = 110 * borrowed_amount / 100 →
  borrowed_amount = 150 :=
by
  intros h1 h2 h3 h4
  sorry

end borrowed_amount_correct_l205_205637


namespace counterfeit_coin_conundrum_l205_205332

theorem counterfeit_coin_conundrum (a : ℕ) (w l : ℕ) (h1 : w > a) (h2 : l < a)
  (h3 : 2 * a = 2 * a) (genuine : ℕ -> Prop) (counterfeit : ℕ -> Prop) :
  ∃ weighings : list (ℕ × ℕ), weighings.length = 3 ∧
  ((w + l < 2 * a) ∨ (w + l > 2 * a) ∨ (w + l = 2 * a)) :=
by
  sorry

end counterfeit_coin_conundrum_l205_205332


namespace least_number_to_add_l205_205992

theorem least_number_to_add (k n : ℕ) (h : k = 1015) (m : n = 25) : 
  ∃ x : ℕ, (k + x) % n = 0 ∧ x = 10 := by
  sorry

end least_number_to_add_l205_205992


namespace max_dist_to_plane_from_Q_l205_205846

noncomputable def point := (ℝ × ℝ × ℝ)

def PA : ℝ := 1
def PB : ℝ := 2
def PC : ℝ := 2

def A : point := (1, 0, 0)
def B : point := (0, 2, 0)
def C : point := (0, 0, 2)
def P : point := (0, 0, 0)

def dist_to_plane (p : point) : ℝ :=
  let (x, y, z) := p in
  abs (x + 2 * y + 2 * z - 1) / real.sqrt (1^2 + 2^2 + 2^2)

def max_dist : ℝ := (3 / 2) + (real.sqrt 6 / 6)

theorem max_dist_to_plane_from_Q (Q : point) (hQ : (Q.1 - 1)^2 + (Q.2 - 2)^2 + (Q.3 - 2)^2 = (real.sqrt 6 / 2)^2) :
  dist_to_plane Q <= max_dist :=
sorry

end max_dist_to_plane_from_Q_l205_205846


namespace garden_area_increase_l205_205244

-- Define the initial conditions
def initial_length : ℝ := 60
def initial_width : ℝ := 12
def fence_length : ℝ := 144

-- Define the calculations
def initial_area : ℝ := initial_length * initial_width
def square_side : ℝ := fence_length / 4
def square_area : ℝ := square_side * square_side
def area_increase : ℝ := square_area - initial_area

-- The theorem stating the problem to be proved
theorem garden_area_increase :
  area_increase = 576 := 
by simp [initial_length, initial_width, fence_length, initial_area, square_side, square_area, area_increase]; sorry

end garden_area_increase_l205_205244


namespace determine_height_of_balloon_l205_205316

variables {A B C D E O H : Type}

noncomputable def height_of_balloon (c d h : ℤ) :=
h^2 + c^2 = 140^2 ∧ h^2 + d^2 = 120^2 ∧ h^2 + (c^2 + d^2) = 130^2 ∧ c^2 + d^2 = 100^2

theorem determine_height_of_balloon (c d h : ℤ) (h1 : h^2 + c^2 = 140^2)
  (h2 : h^2 + d^2 = 120^2) (h3 : h^2 + (c^2 + d^2) = 130^2) (h4 : c^2 + d^2 = 100^2) :
  h = 30 * real.sqrt 7.67 :=
by
  sorry

end determine_height_of_balloon_l205_205316


namespace smallest_sum_of_five_distinct_primes_is_43_l205_205754

open Nat

-- Definition of the sum of five different prime numbers
def sum_of_five_primes (a b c d e : ℕ) : ℕ :=
  a + b + c + d + e

-- Predicate that checks if all numbers are distinct primes
def five_distinct_primes (a b c d e : ℕ) : Prop :=
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧ Prime e ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e

-- The main theorem statement
theorem smallest_sum_of_five_distinct_primes_is_43 :
  ∃ a b c d e : ℕ, five_distinct_primes a b c d e ∧ Prime (sum_of_five_primes a b c d e) ∧ sum_of_five_primes a b c d e = 43 :=
by
  sorry

end smallest_sum_of_five_distinct_primes_is_43_l205_205754


namespace equation_of_hyperbola_P_lies_on_fixed_line_l205_205362

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- The conditions provided in the problem
def focus := (-2 * Real.sqrt 5, 0)
def center := (0, 0)
def eccentricity : ℝ := Real.sqrt 5
def vertex1 := (-2, 0)
def vertex2 := (2, 0)
def passing_point := (-4, 0)

-- Finding the equation of the hyperbola
theorem equation_of_hyperbola : 
  ∀ (a b c : ℝ), 
  c = 2 * Real.sqrt 5 ∧ 
  eccentricity = Real.sqrt 5 ∧ 
  a = 2 ∧
  b = 4 → 
  hyperbola 2 4 :=
by
  sorry

-- Proving P lies on a fixed line
theorem P_lies_on_fixed_line :
  ∀ (m : ℝ), 
  let line : ℝ → ℝ := λ y, m * y - 4 in
  let intersection_M := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let intersection_N := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let P := (*) -- Intersection point of lines MA1 and NA2 -- (*) 
  ∃ (P : ℝ × ℝ), P.fst = -1 :=
by
  sorry

end equation_of_hyperbola_P_lies_on_fixed_line_l205_205362


namespace exists_natural_multiple_of_2015_with_digit_sum_2015_l205_205731

-- Definition of sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Proposition that we need to prove
theorem exists_natural_multiple_of_2015_with_digit_sum_2015 :
  ∃ n : ℕ, (2015 ∣ n) ∧ sum_of_digits n = 2015 :=
sorry

end exists_natural_multiple_of_2015_with_digit_sum_2015_l205_205731


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205074

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205074


namespace cotangent_relation_l205_205486

variable {A B C E : Type*} [EuclideanGeometry A B C E]
variables (α₁ α₂ δ : ℝ)

/-- Let E be the midpoint of side BC of the triangle ABC.
    Given the angles BAE = α₁, EAC = α₂, and AEB = δ,
    then the relationship cot α₂ - cot α₁ = 2 cot δ holds. -/

theorem cotangent_relation (h1 : A = midpoint B C)
                           (h2 : ∠ BAE = α₁)
                           (h3 : ∠ EAC = α₂)
                           (h4 : ∠ AEB = δ) :
  Real.cot α₂ - Real.cot α₁ = 2 * Real.cot δ :=
sorry

end cotangent_relation_l205_205486


namespace sam_total_coins_l205_205919

theorem sam_total_coins (nickel_count : ℕ) (dime_count : ℕ) (total_value_cents : ℤ) (nickel_value : ℤ) (dime_value : ℤ)
  (h₁ : nickel_count = 12)
  (h₂ : total_value_cents = 240)
  (h₃ : nickel_value = 5)
  (h₄ : dime_value = 10)
  (h₅ : nickel_count * nickel_value + dime_count * dime_value = total_value_cents) :
  nickel_count + dime_count = 30 := 
  sorry

end sam_total_coins_l205_205919


namespace neither_sufficient_nor_necessary_for_parallel_l205_205803

-- Definitions for the lines
def line1 (a : ℝ) := λ x y : ℝ, a * x + y + a = 0
def line2 (a : ℝ) := λ x y : ℝ, (a - 6) * x + (a - 4) * y - 4 = 0

-- Proof statement
theorem neither_sufficient_nor_necessary_for_parallel (a : ℝ) :
  ¬ ( (a = 2 ∧ line1 2 ∥ line2 2) ∨ (line1 a ∥ line2 a → a = 2)) :=
by
  sorry

end neither_sufficient_nor_necessary_for_parallel_l205_205803


namespace probability_two_chinese_knights_attack_l205_205510

-- Define the chess board and knight attack rules
def china_knight (i j : Nat) (board : Matrix Nat Nat Bool) : Bool :=
  -- Implement the attack condition for a Chinese knight
  sorry

-- Number of favourable outcomes
noncomputable def favorable_outcomes : ℚ := (79 : ℚ) / 256
  
-- Probabilistic model of the board
def board_model : ProbabilityModel (Matrix Nat Nat Bool) := sorry

-- Theorem asserting the correct probability under specified conditions
theorem probability_two_chinese_knights_attack :
  Probability board_model (λ board, ∃ i j, china_knight i j board) = favorable_outcomes :=
sorry

end probability_two_chinese_knights_attack_l205_205510


namespace min_value_problem_l205_205786

theorem min_value_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 4) :
  (x + 1) * (2 * y + 1) / (x * y) ≥ 9 / 2 :=
by
  sorry

end min_value_problem_l205_205786


namespace union_condition_intersection_condition_l205_205499

def setA : Set ℝ := {x | x^2 - 5 * x + 6 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ 3}

theorem union_condition (a : ℝ) : setA ∪ setB a = setB a ↔ a < 2 := sorry

theorem intersection_condition (a : ℝ) : setA ∩ setB a = setB a ↔ a ≥ 2 := sorry

end union_condition_intersection_condition_l205_205499


namespace amount_dog_ate_cost_l205_205480

-- Define the costs of each ingredient
def cost_flour : Real := 4
def cost_sugar : Real := 2
def cost_butter : Real := 2.5
def cost_eggs : Real := 0.5

-- Define the number of slices
def number_of_slices := 6

-- Define the number of slices eaten by Laura's mother
def slices_eaten_by_mother := 2

-- Calculate the total cost of the ingredients
def total_cost := cost_flour + cost_sugar + cost_butter + cost_eggs

-- Calculate the cost per slice
def cost_per_slice := total_cost / number_of_slices

-- Calculate the number of slices eaten by Kevin
def slices_eaten_by_kevin := number_of_slices - slices_eaten_by_mother

-- Define the total cost of slices eaten by Kevin
def cost_eaten_by_kevin := slices_eaten_by_kevin * cost_per_slice

-- The main statement to prove
theorem amount_dog_ate_cost :
  cost_eaten_by_kevin = 6 := by
    sorry

end amount_dog_ate_cost_l205_205480


namespace circumcircle_AOB_l205_205950

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)
noncomputable def circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  { P | (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2 }

theorem circumcircle_AOB
  (P : ℝ × ℝ) (P_eq : P = (4, 2))
  (circ : set (ℝ × ℝ)) (circ_eq : circ = circle (0, 0) (sqrt 2))
  (A B : ℝ × ℝ) (tangents : ∀ Q, Q ∈ circ → ∀ T, T = A ∨ T = B → Q = P)
  (O : ℝ × ℝ) (O_eq : O = (0, 0)) :
  circle (2, 1) (sqrt 5) = { T : ℝ × ℝ | (T.1 - 2)^2 + (T.2 - 1)^2 = 5 } :=
sorry

end circumcircle_AOB_l205_205950


namespace average_of_five_digit_palindromes_l205_205072

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def five_digit_palindromes := 
  {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧ is_palindrome n}

theorem average_of_five_digit_palindromes : 
  (finset.univ.filter (λ n : ℕ, n ∈ five_digit_palindromes)).sum id 
  / (finset.univ.filter (λ n : ℕ, n ∈ five_digit_palindromes)).card = 55000 := 
sorry

end average_of_five_digit_palindromes_l205_205072


namespace range_of_sqrt3x_plus_y_l205_205302

noncomputable def circle_polar_to_cartesian (rho theta : ℝ) : Prop :=
  rho = 4 * sin (theta - π / 6)

def line_parametric (t : ℝ) : ℝ × ℝ :=
  (-1 - sqrt 3 / 2 * t, sqrt 3 + 1 / 2 * t)

def cartesian_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 2 * sqrt 3 * y = 0

theorem range_of_sqrt3x_plus_y (t : ℝ) :
  let (x, y) := line_parametric t in
  cartesian_circle x y →
  -2 ≤ sqrt 3 * x + y ∧ sqrt 3 * x + y ≤ 2 :=
begin
  sorry
end

end range_of_sqrt3x_plus_y_l205_205302


namespace least_n_factorial_divisible_by_1029_l205_205198

theorem least_n_factorial_divisible_by_1029 :
  ∃ n : ℕ, (∀ k : ℕ, (prod x in finset.range k, x) % 1029 = 0 ↔ 21 ≤ k) :=
sorry

end least_n_factorial_divisible_by_1029_l205_205198


namespace hyperbola_center_origin_point_P_on_fixed_line_l205_205350

variables {x y : ℝ}

def hyperbola (h : (x^2 / 4) - (y^2 / 16) = 1) : Prop := true

theorem hyperbola_center_origin 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (left_focus : ∃ x y : ℝ, x = -2 * Real.sqrt 5 ∧ y = 0)
  (eccentricity_root5 : ∃ e : ℝ, e = Real.sqrt 5) :
  hyperbola ((x^2 / 4) - (y^2 / 16) = 1) := 
sorry

theorem point_P_on_fixed_line
  (line_through_point : ∃ x y : ℝ, x = -4 ∧ y = 0)
  (M_second_quadrant : ∃ Mx My : ℝ, Mx < 0 ∧ My > 0)
  (A1_vertex : ∃ x y : ℝ, x = -2 ∧ y = 0)
  (A2_vertex : ∃ x y : ℝ, x = 2 ∧ y = 0)
  (P_intersection : ∃ Px Py : ℝ, Px = -1) :
  Px = -1 :=
sorry

end hyperbola_center_origin_point_P_on_fixed_line_l205_205350


namespace tomatoes_grew_in_absence_l205_205445

def initial_tomatoes : ℕ := 36
def multiplier : ℕ := 100
def total_tomatoes_after_vacation : ℕ := initial_tomatoes * multiplier

theorem tomatoes_grew_in_absence : 
  total_tomatoes_after_vacation - initial_tomatoes = 3564 :=
by
  -- skipped proof with 'sorry'
  sorry

end tomatoes_grew_in_absence_l205_205445


namespace find_profit_percentage_l205_205695

variable (CP : ℝ) (SP : ℝ) (MP : ℝ)

def profit_percentage (P CP : ℝ) : ℝ := (P / CP) * 100

theorem find_profit_percentage
  (h1 : CP = 200)
  (h2 : SP = 266.67)
  (h3 : SP = 0.90 * MP)
  : profit_percentage (SP - CP) CP = 33.34 := 
by
  sorry

end find_profit_percentage_l205_205695


namespace find_k_l205_205738

-- Define the problem's conditions and constants
variables (S x y : ℝ)

-- Define the main theorem to prove k = 8 given the conditions
theorem find_k (h1 : 0.75 * x + ((S - 0.75 * x) * x) / (x + y) - (S * x) / (x + y) = 18) :
  (x * y / 3) / (x + y) = 8 := by 
  sorry

end find_k_l205_205738


namespace find_b_d_l205_205054

theorem find_b_d (b d : ℕ) (h1 : b + d = 41) (h2 : b < d) : 
  (∃! x, b * x * x + 24 * x + d = 0) → (b = 9 ∧ d = 32) :=
by 
  sorry

end find_b_d_l205_205054


namespace fixed_point_of_HN_l205_205780

noncomputable def ellipse : Type :=
{ x y : ℝ // (x^2) / 3 + (y^2) / 4 = 1 }

def A : ellipse := ⟨0, -2, by norm_num [div_eq_inv_mul, pow_two, mul_one_div]⟩

def B : ellipse := ⟨3 / 2, -1, by norm_num [div_eq_inv_mul, pow_two, add_div, mul_add, mul_div_cancel']⟩

def P : ℝ × ℝ := (1, -2)

def intersects_ellipse (L : ℝ × ℝ → Prop) : Prop :=
∀ (M N : ellipse), L (M.val) → L (N.val) → ∃! (M_val N_val : ℝ × ℝ), M_val = M.val ∧ N_val = N.val

def equation_of_line (p1 p2 : ℝ × ℝ) : ℝ × ℝ → Prop :=
fun x => ∃ (k b : ℝ), x.2 = k * x.1 + b ∧ k = (p2.2 - p1.2) / (p2.1 - p1.1) ∧ b = p1.2 - k * p1.1

def parallel_to_x_axis (x : ℝ × ℝ) : Prop := ∃ (b : ℝ), x.2 = b

def midpoint (x y : ℝ × ℝ) : ℝ × ℝ := ((x.1 + y.1) / 2, (x.2 + y.2) / 2)

theorem fixed_point_of_HN :
  let E := λ (x y : ℝ), (x^2) / 3 + (y^2) / 4 = 1 in
  let A := (0, -2) in 
  let B := (3 / 2, -1) in
  let P := (1, -2) in
  let line_through_P := λ (x : ℝ × ℝ), x.1 = 1 ∨ ∃ k : ℝ, x.2 = k * x.1 + k + 2 in
  ∀ (L : ℝ × ℝ → Prop),
  (intersects_ellipse line_through_P) →
  (∀ (M N : ℝ × ℝ), L M → L N → L (midpoint M N)) →
  (∀ (M T : ℝ × ℝ), L T → parallel_to_x_axis M → midpoint M T = midpoint T H) →
  ∃ (K : ℝ × ℝ), (K = (0, -2)) ∧ L H := sorry

end fixed_point_of_HN_l205_205780


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205129

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205129


namespace hyperbola_center_origin_point_P_on_fixed_line_l205_205351

variables {x y : ℝ}

def hyperbola (h : (x^2 / 4) - (y^2 / 16) = 1) : Prop := true

theorem hyperbola_center_origin 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (left_focus : ∃ x y : ℝ, x = -2 * Real.sqrt 5 ∧ y = 0)
  (eccentricity_root5 : ∃ e : ℝ, e = Real.sqrt 5) :
  hyperbola ((x^2 / 4) - (y^2 / 16) = 1) := 
sorry

theorem point_P_on_fixed_line
  (line_through_point : ∃ x y : ℝ, x = -4 ∧ y = 0)
  (M_second_quadrant : ∃ Mx My : ℝ, Mx < 0 ∧ My > 0)
  (A1_vertex : ∃ x y : ℝ, x = -2 ∧ y = 0)
  (A2_vertex : ∃ x y : ℝ, x = 2 ∧ y = 0)
  (P_intersection : ∃ Px Py : ℝ, Px = -1) :
  Px = -1 :=
sorry

end hyperbola_center_origin_point_P_on_fixed_line_l205_205351


namespace number_of_people_in_group_is_21_l205_205008

-- Definitions based directly on the conditions
def pins_contribution_per_day := 10
def pins_deleted_per_week_per_person := 5
def group_initial_pins := 1000
def final_pins_after_month := 6600
def weeks_in_a_month := 4

-- To be proved: number of people in the group is 21
theorem number_of_people_in_group_is_21 (P : ℕ)
  (h1 : final_pins_after_month - group_initial_pins = 5600)
  (h2 : weeks_in_a_month * (pins_contribution_per_day * 7 - pins_deleted_per_week_per_person) = 260)
  (h3 : 5600 / 260 = 21) :
  P = 21 := 
sorry

end number_of_people_in_group_is_21_l205_205008


namespace smallest_pos_int_div_by_four_primes_l205_205109

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205109


namespace equation_of_hyperbola_point_P_fixed_line_l205_205368

-- Definitions based on conditions
def center_origin := (0, 0)
def left_focus := (-2 * real.sqrt 5, 0)
def eccentricity := real.sqrt 5
def left_vertex := (-2, 0)
def right_vertex := (2, 0)
def point_passed_through_line := (-4, 0)

-- The equation of the hyperbola that satisfies the given conditions.
theorem equation_of_hyperbola :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ 
    (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ 
      (2 * real.sqrt 5)^2 = a^2 + b^2) :=
by
  use [2, 4]
  split
  exact rfl
  split
  exact rfl
  sorry

-- Proving P lies on the fixed line x = -1 for given lines and vertices conditions
theorem point_P_fixed_line :
  ∀ m : ℝ, ∀ y1 y2 x1 x2 : ℝ, 
    x1 = m * y1 - 4 ∧ x2 = m * y2 - 4 ∧ 
    (y1 + y2 = (32 * m) / ((4 * m^2) - 1)) ∧
    (y1 * y2 = 48 / ((4 * m^2) - 1)) ∧ 
    (x1 + 2) ≠ 0 ∧ 
    (x2 - 2) ≠ 0 → 
    (P : ℝ × ℝ) ((P.1, P.2) = (-1, P.2) ∧ P.2 = (y1 * (x + 2) / x1 ≠ -2 ∧ y2 * (x - 2) / x2 ≠ 2)) := 
by
  intros
  sorry

end equation_of_hyperbola_point_P_fixed_line_l205_205368


namespace percentage_markup_l205_205572

theorem percentage_markup (selling_price cost_price : ℚ)
  (h_selling_price : selling_price = 8325)
  (h_cost_price : cost_price = 7239.13) :
  ((selling_price - cost_price) / cost_price) * 100 = 15 := 
sorry

end percentage_markup_l205_205572


namespace sum_areas_constructed_parallelograms_equal_area_on_AB_CD_l205_205513

open Set

variable {A B C D : Point}
variable {p₁ p₂ : Parallelogram}

-- Conditions
def constructed_on_AC (p₁ : Parallelogram) : Prop := 
  p₁.base = AC

def constructed_on_BC (p₂ : Parallelogram) : Prop := 
  p₂.base = BC

def intersect_at_D (p₁ p₂ : Parallelogram) (D : Point) : Prop := 
  ∃ D, (p₁.opposite_side ∩ p₂.opposite_side = {D})

-- Proof goal
theorem sum_areas_constructed_parallelograms_equal_area_on_AB_CD 
  (h₁ : constructed_on_AC p₁) 
  (h₂ : constructed_on_BC p₂) 
  (h₃ : intersect_at_D p₁ p₂ D) :
  p₁.area + p₂.area = (parallelogram_on_AB_CD A B C D).area :=
sorry

end sum_areas_constructed_parallelograms_equal_area_on_AB_CD_l205_205513


namespace simplify_expression_l205_205527

variable (x : ℝ)

theorem simplify_expression :
  (2 * x ^ 6 + 3 * x ^ 5 + x ^ 4 + x ^ 3 + 5) - (x ^ 6 + 4 * x ^ 5 + 2 * x ^ 4 - x ^ 3 + 7) = 
  x ^ 6 - x ^ 5 - x ^ 4 + 2 * x ^ 3 - 2 := by
  sorry

end simplify_expression_l205_205527


namespace units_digit_47_pow_47_l205_205621

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l205_205621


namespace find_x0_l205_205403

-- Definitions based on conditions
def f (x : ℝ) : ℝ := x * (2017 + Real.log x) * Real.exp 1
def f' (x : ℝ) : ℝ := 2018 + Real.log x

-- Statement to be proved
theorem find_x0 (x0 : ℝ) (h : f' x0 = 2018) : x0 = 1 := by
  sorry

end find_x0_l205_205403


namespace f_comp_f_0_eq_half_l205_205795

def f (x : ℝ) : ℝ :=
if x < 1 then (x + 1)^2 else 2^(x - 2)

theorem f_comp_f_0_eq_half : f (f 0) = 1 / 2 :=
by sorry

end f_comp_f_0_eq_half_l205_205795


namespace equilibrium_problems_l205_205728

-- Definition of equilibrium constant and catalyst relations

def q1 := False -- Any concentration of substances in equilibrium constant
def q2 := False -- Catalysts changing equilibrium constant
def q3 := False -- No shift if equilibrium constant doesn't change
def q4 := False -- ΔH > 0 if K decreases with increasing temperature
def q5 := True  -- Stoichiometric differences affecting equilibrium constants
def q6 := True  -- Equilibrium shift not necessarily changing equilibrium constant
def q7 := True  -- Extent of reaction indicated by both equilibrium constant and conversion rate

-- The theorem includes our problem statements

theorem equilibrium_problems :
  q1 = False ∧ q2 = False ∧ q3 = False ∧
  q4 = False ∧ q5 = True ∧ q6 = True ∧ q7 = True := by
  sorry

end equilibrium_problems_l205_205728


namespace units_digit_of_power_l205_205823

theorem units_digit_of_power (base : ℕ) (exp : ℕ) (units_base : ℕ) (units_exp_mod : ℕ) :
  (base % 10 = units_base) → (exp % 2 = units_exp_mod) → (units_base = 9) → (units_exp_mod = 0) →
  (base ^ exp % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l205_205823


namespace log_expression_equality_l205_205288

-- Definitions and the problem statement
theorem log_expression_equality : 
  (log 10 5 * (log 10 20 / (log 10 (sqrt 10))) + (log 10 (2 ^ sqrt 2)) ^ 2 + exp (log pi) = 2 + pi) :=
by 
  sorry 

end log_expression_equality_l205_205288


namespace sum_of_reciprocals_of_a_n_l205_205581

def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + n + 1

theorem sum_of_reciprocals_of_a_n (a : ℕ → ℕ) (h : seq a) : 
  (∑ i in Finset.range 100, 1 / (a (i + 1))) = 200 / 101 :=
sorry

end sum_of_reciprocals_of_a_n_l205_205581


namespace quadrilateral_midpoints_intersection_l205_205931

noncomputable def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem quadrilateral_midpoints_intersection (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) :
  let A := (a1, b1)
  let B := (a2, b2)
  let C := (a3, b3)
  let D := (a4, b4)
  let P := midpoint A B
  let Q := midpoint B C
  let R := midpoint C D
  let S := midpoint D A
  let M := midpoint A C
  let N := midpoint B D
  ∃ T : ℝ × ℝ, T = ( (a1 + a2 + a3 + a4) / 4, (b1 + b2 + b3 + b4) / 4 ) ∧
                T = midpoint P R ∧
                T = midpoint Q S ∧
                T = midpoint M N :=
by
  sorry

end quadrilateral_midpoints_intersection_l205_205931


namespace reciprocal_lcm_of_24_and_208_l205_205579

theorem reciprocal_lcm_of_24_and_208 :
  (1 / (Nat.lcm 24 208)) = (1 / 312) :=
by
  sorry

end reciprocal_lcm_of_24_and_208_l205_205579


namespace number_of_equilateral_triangles_l205_205945

noncomputable def count_equilateral_triangles : ℕ :=
  let lines := λ k : ℤ, {y = k, y = sqrt 2 * x + 2 * k, y = -sqrt 2 * x + 2 * k} 
  in
  count_triangles lines (λ (side_length : ℝ), side_length = 1 / sqrt 2)

theorem number_of_equilateral_triangles :
  count_equilateral_triangles = 1230 :=
sorry

end number_of_equilateral_triangles_l205_205945


namespace cannot_represent_same_function_A_C_D_l205_205266

/- Define the functions for each option -/

def f_A (x : ℝ) := real.sqrt (x + 1) * real.sqrt (x - 1)
def g_A (x : ℝ) := real.sqrt (x^2 - 1)

def f_B (x : ℝ) := x^2
def g_B (x : ℝ) := real.cbrt (x^6)

def f_C (x : ℝ) := (x^2 - 1) / (x - 1)
def g_C (x : ℝ) := x + 1

def f_D (x : ℝ) := real.sqrt (x^2)
def g_D (x : ℝ) := (real.sqrt x)^2

/- Prove that A, C, and D cannot represent the same function -/

theorem cannot_represent_same_function_A_C_D :
  (∃ x, f_A x ≠ g_A x) ∧ (∃ x, f_C x ≠ g_C x) ∧ (∃ x, f_D x ≠ g_D x) :=
by {
  sorry -- Step A (x > 1)
  sorry -- Step C (x != 1)
  sorry -- Step D (x < 0)
}

end cannot_represent_same_function_A_C_D_l205_205266


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205079

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205079


namespace find_function_expression_increasing_interval_axis_of_symmetry_g_l205_205799

noncomputable def func1 (x : ℝ) : ℝ := 5 * Real.sin(2 * x - (Real.pi / 6))

theorem find_function_expression :
  ∃ A ω φ, A > 0 ∧ ω > 0 ∧ Abs φ < (Real.pi / 2) ∧
  (A = 5 ∧ ω = 2 ∧ φ = -Real.pi / 6) ∧
  ∀ x, func1 x = 5 * Real.sin(2 * x - (Real.pi / 6)) := 
sorry

theorem increasing_interval (k : ℤ) : 
  ∀ x, func1 x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 5 * Real.sin(2 * x + (Real.pi / 6)) - 2

theorem axis_of_symmetry_g (k : ℤ) :
  ∀ x, g x = g (2 * x + (Real.pi / 6)) ↔ x = (Real.pi / 6) + (k * Real.pi / 2) := 
sorry

end find_function_expression_increasing_interval_axis_of_symmetry_g_l205_205799


namespace number_of_eggs_in_tray_l205_205053

-- Definitions
def egg_price : ℝ := 0.50
def tray_price : ℝ := 12
def savings_per_egg : ℝ := 0.10
def eggs_in_tray : ℝ := tray_price / (egg_price - savings_per_egg)

-- Theorem
theorem number_of_eggs_in_tray : eggs_in_tray = 30 := by
  -- skipping the proof part
  sorry

end number_of_eggs_in_tray_l205_205053


namespace adam_earnings_l205_205691

def lawns_to_mow : ℕ := 12
def lawns_forgotten : ℕ := 8
def earnings_per_lawn : ℕ := 9

theorem adam_earnings : (lawns_to_mow - lawns_forgotten) * earnings_per_lawn = 36 := by
  sorry

end adam_earnings_l205_205691


namespace kitten_probability_l205_205257

-- Define the side lengths of the triangle
def side_lengths : List ℝ := [5, 5, 6]

-- Define the area of the triangle using Heron's formula
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the event A and its complement A'
-- Event A: kitten is more than 2m away from all three vertices
-- Event A': kitten is within 2m of at least one vertex
noncomputable def event_A_complement_area (r : ℝ) : ℝ :=
  (1/2) * π * r^2

noncomputable def probability_event_A_complement (area_T : ℝ) (area_A_complement : ℝ) : ℝ :=
  area_A_complement / area_T

noncomputable def probability_event_A (p_A_complement : ℝ) : ℝ :=
  1 - p_A_complement

-- Proving the probability that the kitten is more than 2m away from all vertices is 1 - π/6
theorem kitten_probability : probability_event_A (probability_event_A_complement 12 (event_A_complement_area 2)) = 1 - π / 6 := by
  sorry

end kitten_probability_l205_205257


namespace presents_difference_l205_205522

theorem presents_difference :
  ∀ (num_siblings : ℕ)
    (march_birthdays may_birthdays june_birthdays
     october_birthdays november_birthdays december_birthdays aug_holiday christmas : ℕ),
  ∀ (total_first_half total_second_half presents_diff : ℕ),
  num_siblings = 10 →
  march_birthdays = 4 →
  may_birthdays = 1 →
  june_birthdays = 1 →
  october_birthdays = 1 →
  november_birthdays = 1 →
  december_birthdays = 2 →
  aug_holiday = num_siblings →
  christmas = num_siblings →
  total_first_half = march_birthdays + may_birthdays + june_birthdays →
  total_second_half = october_birthdays + november_birthdays + december_birthdays + aug_holiday + christmas →
  presents_diff = total_second_half - total_first_half →
  presents_diff = 18 :=
by
  intros num_siblings march_birthdays may_birthdays june_birthdays
        october_birthdays november_birthdays december_birthdays aug_holiday christmas
        total_first_half total_second_half presents_diff
  assume h_num_siblings h_march h_may h_june h_october h_november h_december h_aug h_christmas
         h_first_half h_second_half h_diff
  rw [h_march, h_may, h_june] at h_first_half
  rw [h_october, h_november, h_december, h_aug, h_christmas] at h_second_half
  sorry

end presents_difference_l205_205522


namespace number_of_trees_l205_205591

theorem number_of_trees (n : ℕ) (diff : ℕ) (count1 : ℕ) (count2 : ℕ) (timur1 : ℕ) (alexander1 : ℕ) (timur2 : ℕ) (alexander2 : ℕ) : 
  diff = alexander1 - timur1 ∧
  count1 = timur2 + (alexander2 - timur1) ∧
  n = count1 + diff →
  n = 118 :=
by
  sorry

end number_of_trees_l205_205591


namespace largest_two_digit_prime_factor_of_binomial_l205_205971

def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem largest_two_digit_prime_factor_of_binomial : 
  let n := binomial 210 105 in 
  ∃ p : ℕ, nat.prime p ∧ 10 ≤ p ∧ p < 100 ∧ 
           (∀ q : ℕ, nat.prime q ∧ 10 ≤ q ∧ q < 100 → q ≤ p) := 
  ∃ p, p = 67 ∧ nat.prime p ∧ 10 ≤ p ∧ p < 100 ∧ 
          (∀ q, nat.prime q ∧ 10 ≤ q ∧ q < 100 → q ≤ p) := 
sorry

end largest_two_digit_prime_factor_of_binomial_l205_205971


namespace problem1_problem2_l205_205791

noncomputable def f (x : ℝ) : ℝ := real.sin (x + π / 6) + real.cos x

theorem problem1 :
  (∀ x : ℝ, f x ≤ sqrt 3) ∧ (∃ k : ℤ, f (π/6 + 2 * k * π) = sqrt 3) :=
sorry

variables (a : ℝ)
axiom ha : a ∈ set.Ioo 0 (π/2)
axiom hfa : f (a + π/6) = 3 * sqrt 3 / 5

theorem problem2 : f (2 * a) = (24 * sqrt 3 - 21) / 50 :=
sorry

end problem1_problem2_l205_205791


namespace correct_calculation_l205_205979

variable (a : ℝ)

theorem correct_calculation (a : ℝ) : (2 * a)^2 / (4 * a) = a := by
  sorry

end correct_calculation_l205_205979


namespace least_positive_integer_divisible_by_four_primes_l205_205127

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205127


namespace vectors_triangle_ABC_is_right_angled_problem_statement_l205_205854

-- Define points in the complex plane
def A : ℂ := 1
def B : ℂ := - complex.i
def C : ℂ := -1 + 2 * complex.i

-- Conversion from complex number to 2D coordinates
def complex_to_coords (z : ℂ) : ℝ × ℝ := ⟨z.re, z.im⟩

-- Calculate vectors between points in 2D coordinates
def vector (P Q : ℂ) : ℝ × ℝ :=
  let ⟨x1, y1⟩ := complex_to_coords P
  let ⟨x2, y2⟩ := complex_to_coords Q
  (x2 - x1, y2 - y1)

-- Verify vectors
theorem vectors :
(vector A B = (-1, -1)) ∧
(vector A C = (-2, 2)) ∧
(vector B C = (-1, 3)) :=
by {
  -- Convert points to coordinates
  have hA: complex_to_coords A = (1, 0), from sorry,
  have hB: complex_to_coords B = (0, -1), from sorry,
  have hC: complex_to_coords C = (-1, 2), from sorry,
  -- Calculate the vectors
  iterate 3 {rw [vector]},
  iterate 3 {rw [complex_to_coords]},
  exact and.intro rfl (and.intro rfl rfl)
}

-- Check if vectors form a right-angled triangle at A
def is_right_angle (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem triangle_ABC_is_right_angled :
  is_right_angle (vector A B) (vector A C) :=
by {
  rw [is_right_angle, vector A B, vector A C],
  -- Calculate the dot product
  sorry
}

-- Final theorem combining the above results
theorem problem_statement :
(vectors A B C) ∧ (triangle_ABC_is_right_angled A B C) :=
by {
  split,
  exact vectors,
  exact triangle_ABC_is_right_angled
}

end vectors_triangle_ABC_is_right_angled_problem_statement_l205_205854


namespace sum_log_difference_l205_205712

theorem sum_log_difference :
  (∑ k in Finset.range 1500, k * (⌈Real.log k / Real.log 3⌉₊ - ⌊Real.log k / Real.log 3⌋₊)) = 1124657 :=
by
  /- The proof goes here -/
  sorry

end sum_log_difference_l205_205712


namespace radius_of_smaller_circle_l205_205228

theorem radius_of_smaller_circle (A1 : ℝ) (r1 r2 : ℝ) (h1 : π * r2^2 = 4 * A1)
    (h2 : r2 = 4) : r1 = 2 :=
by
  sorry

end radius_of_smaller_circle_l205_205228


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205141

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205141


namespace inequality_solution_set_l205_205944

variable {R : Type*} [LinearOrder R] [TopologicalSpace R] [OrderTopology R]
variable (f : R → R)

def decreasing (f : R → R) := ∀ ⦃x y : R⦄, x < y → f y ≤ f x

theorem inequality_solution_set (f_decreasing : decreasing f) (h : f 1 = 0) :
  {x : R | f (x - 1) < 0} = {x : R | x < 2} :=
by {
  sorry
}

end inequality_solution_set_l205_205944


namespace set_M_contains_3_l205_205878

theorem set_M_contains_3 :
  let U := {x : ℤ | x^2 - 6 * x < 0}
  let M := {x : ℤ | x ∈ U ∧ x ∉ {1, 2}}
  3 ∈ M :=
by {
  let U := {x : ℤ | x^2 - 6 * x < 0},
  have U_def : U = {1, 2, 3, 4, 5}, {
    sorry,
  },
  let M := {x : ℤ | x ∈ U ∧ x ∉ {1, 2}},
  have M_def : M = {3, 4, 5}, {
    sorry,
  },
  exact set.mem_singleton 3 M_def
}

end set_M_contains_3_l205_205878


namespace transformed_graph_l205_205593

def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := f (2 * x)
def h (x : ℝ) : ℝ := g (x - Real.pi / 6)

theorem transformed_graph :
  ∀ (x : ℝ), h(x) = Real.sin (2 * x - Real.pi / 3) :=
by
  -- Proof part is not required.
  sorry

end transformed_graph_l205_205593


namespace meteor_trail_is_a_line_l205_205223

-- We define a point moving through space.
def point_moves_to_form_a_line (P : Type) [point_space : add_comm_group P] [vector_space ℝ P] : Prop :=
  ∀ (p : P) (path : ℝ → P), (∃ t : ℝ, path t = p) →
    ∀ t₁ t₂ : ℝ, t₁ ≤ t₂ → ∃ q : set P, ∀ r : ℝ, t₁ ≤ r ∧ r ≤ t₂ → (path r ∈ q ∧ set.inj_on path q)

-- Prove that a point moving continuously forms a line.
theorem meteor_trail_is_a_line (P : Type) [point_space : add_comm_group P] [vector_space ℝ P]
  (move : ℝ → P) : point_moves_to_form_a_line P :=
by
  sorry

end meteor_trail_is_a_line_l205_205223


namespace find_n_value_l205_205759

theorem find_n_value : ∃ n : ℤ, 3^3 - 7 = 4^2 + n ∧ n = 4 :=
by
  use 4
  sorry

end find_n_value_l205_205759


namespace can_combine_with_sqrt3_l205_205264

theorem can_combine_with_sqrt3 
  (A : ℝ) (hA : A = real.sqrt 30)
  (B : ℝ) (hB : B = real.sqrt (1 / 2))
  (C : ℝ) (hC : C = real.sqrt 8)
  (D : ℝ) (hD : D = real.sqrt 27) :
  (∃ k : ℝ, D = k * real.sqrt 3) ∧ ¬ (∃ k : ℝ, A = k * real.sqrt 3) ∧ ¬ (∃ k : ℝ, B = k * real.sqrt 3) ∧ ¬ (∃ k : ℝ, C = k * real.sqrt 3) :=
by sorry

end can_combine_with_sqrt3_l205_205264


namespace lines_not_parallel_l205_205779

variables {m n : Type} [Line m] [Line n] {A : Type} [Point A] {α : Type} [Plane α]

-- Hypotheses
hypothesis h₁ : ¬(m ⊆ α)
hypothesis h₂ : n ⊆ α
hypothesis h₃ : A ∈ m
hypothesis h₄ : A ∈ α

-- Main theorem statement
theorem lines_not_parallel (h₁ : ¬(m ⊆ α)) (h₂ : n ⊆ α) (h₃ : A ∈ m) (h₄ : A ∈ α) : ¬ parallel m n :=
by
  sorry

end lines_not_parallel_l205_205779


namespace arithmetic_sequence_solution_l205_205450

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence : Prop :=
  ∀ n, a (n+1) = a n + d

-- The given condition a_3 + a_5 = 12 - a_7
def condition : Prop :=
  a 3 + a 5 = 12 - a 7

-- The proof statement
theorem arithmetic_sequence_solution 
  (h_arith : is_arithmetic_sequence a d) 
  (h_cond : condition a): a 1 + a 9 = 8 :=
sorry

end arithmetic_sequence_solution_l205_205450


namespace correct_rounded_result_l205_205736

-- Definition of rounding to the nearest hundred
def rounded_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 < 50 then n / 100 * 100 else (n / 100 + 1) * 100

-- Given conditions
def sum : ℕ := 68 + 57

-- The theorem to prove
theorem correct_rounded_result : rounded_to_nearest_hundred sum = 100 :=
by
  -- Proof skipped
  sorry

end correct_rounded_result_l205_205736


namespace average_age_decrease_l205_205029

-- Define the conditions as given in the problem
def original_strength : ℕ := 12
def new_students : ℕ := 12

def original_avg_age : ℕ := 40
def new_students_avg_age : ℕ := 32

def decrease_in_avg_age (O N : ℕ) (OA NA : ℕ) : ℕ :=
  let total_original_age := O * OA
  let total_new_students_age := N * NA
  let total_students := O + N
  let new_avg_age := (total_original_age + total_new_students_age) / total_students
  OA - new_avg_age

theorem average_age_decrease :
  decrease_in_avg_age original_strength new_students original_avg_age new_students_avg_age = 4 :=
sorry

end average_age_decrease_l205_205029


namespace probability_man_in_dark_l205_205645

theorem probability_man_in_dark (revolutions_per_minute : ℕ) (time_in_dark : ℕ) (total_time : ℕ) : 
  revolutions_per_minute = 3 → 
  time_in_dark = 10 → 
  total_time = 20 → 
  (time_in_dark : ℚ) / ↑total_time = 1 / 2 :=
by
  intros h_revolutions h_dark_time h_total_time
  rw [h_dark_time, h_total_time]
  norm_num
  sorry

end probability_man_in_dark_l205_205645


namespace XY_in_30_60_90_triangle_l205_205306

theorem XY_in_30_60_90_triangle : 
  ∀ (X Y Z : Type) (XY ZY : ℝ), 
  (angle X Y Z = 60 ∧ angle Z X Y = 90 ∧ angle X Z Y = 30) ∧ ZY = 15 → 
  XY = 5 * Real.sqrt 3 :=
by 
  intros X Y Z XY ZY h
  cases h with h1 h2
  cases h1 with h1a h1b
  cases h1b with h1b1 h1b2
  sorry

end XY_in_30_60_90_triangle_l205_205306


namespace parabola_relative_positions_l205_205291

def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + x + 3
def parabola3 (x : ℝ) : ℝ := x^2 + 2*x + 3

noncomputable def vertex_x (a b c : ℝ) : ℝ := -b / (2 * a)

theorem parabola_relative_positions :
  vertex_x 1 (-1) 3 < vertex_x 1 1 3 ∧ vertex_x 1 1 3 < vertex_x 1 2 3 :=
by {
  sorry
}

end parabola_relative_positions_l205_205291


namespace cyclic_quadrilateral_l205_205871

-- Define the geometry structures: points, triangles, circles, midpoints
-- Definitions based on given conditions:

axiom Point : Type
axiom Line : Type
axiom Circle : Type

axiom A B C D E F M : Point
axiom AD : Line
axiom omega1 omega2 : Circle
axiom triangle_ABC : triangle A B C

axiom is_midpoint (M D : Point) : Prop
axiom is_diameter (C : Point) (omega : Circle) : Prop
axiom intersects (L : Line) (omega : Circle) (P : Point) : Prop
axiom concyclic (P Q R S : Point) : Prop

-- Given conditions:
axiom midpoint_M : is_midpoint M D
axiom diameter_omega1 : is_diameter C omega1
axiom diameter_omega2 : is_diameter A omega2
axiom intersect_BM_omega1 : intersects BM omega1 E
axiom intersect_CM_omega2 : intersects CM omega2 F

-- The problem to prove:
theorem cyclic_quadrilateral : concyclic B E F C :=
by sorry

end cyclic_quadrilateral_l205_205871


namespace sum_of_possible_degrees_of_A_l205_205942

theorem sum_of_possible_degrees_of_A (angles : Fin 6 → ℤ) (arith_seq : ∃ α d : ℤ, ∀ i : Fin 6, angles i = α + i * d)
  (sum_of_int : ∑ i, angles i = 720) (nondegenerate : ∀ i j : Fin 6, i ≠ j → angles i ≠ angles j)
  (min_angle : angles 0 = min (angles 0) (min (angles 1) (min (angles 2) (min (angles 3) (min (angles 4) (angles 5)))))) :
  (∑ k in Finset.range 24, (120 - 5 * k)) = 1500 := by
  sorry

end sum_of_possible_degrees_of_A_l205_205942


namespace min_cells_to_mark_l205_205601

theorem min_cells_to_mark (G : fin 20 → fin 20 → Prop) :
  (∃ marked_cells : fin 400, 
   ∀ row : fin 20, ∃ i : fin 9, ∃ col : fin 12, G (row + i) col ∧ G (row + i) (col + 8) ∧
   ∀ col : fin 20, ∃ j : fin 9, ∃ row : fin 12, G col (row + j) ∧ G (col + 8) (row + j)) →
  32 := 
begin
  sorry
end

end min_cells_to_mark_l205_205601


namespace locus_of_rectangle_centers_is_well_defined_l205_205339

-- Definitions of the points on the plane
variables (A B C D : ℝ × ℝ)

theorem locus_of_rectangle_centers_is_well_defined
  (center_points : set (ℝ × ℝ)) :
  center_points = { O | O ∈ circle_midpoint (midpoint A C) ∨ O ∈ circle_midpoint (midpoint B D) ∨ O ∈ circle_midpoint (midpoint A D) ∨ O ∈ circle_midpoint (midpoint B C) } :=
sorry

end locus_of_rectangle_centers_is_well_defined_l205_205339


namespace speed_of_man_in_still_water_l205_205987

theorem speed_of_man_in_still_water
  (V_m V_s : ℝ)
  (cond1 : V_m + V_s = 5)
  (cond2 : V_m - V_s = 7) :
  V_m = 6 :=
by
  sorry

end speed_of_man_in_still_water_l205_205987


namespace equation_of_hyperbola_P_lies_on_fixed_line_l205_205361

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- The conditions provided in the problem
def focus := (-2 * Real.sqrt 5, 0)
def center := (0, 0)
def eccentricity : ℝ := Real.sqrt 5
def vertex1 := (-2, 0)
def vertex2 := (2, 0)
def passing_point := (-4, 0)

-- Finding the equation of the hyperbola
theorem equation_of_hyperbola : 
  ∀ (a b c : ℝ), 
  c = 2 * Real.sqrt 5 ∧ 
  eccentricity = Real.sqrt 5 ∧ 
  a = 2 ∧
  b = 4 → 
  hyperbola 2 4 :=
by
  sorry

-- Proving P lies on a fixed line
theorem P_lies_on_fixed_line :
  ∀ (m : ℝ), 
  let line : ℝ → ℝ := λ y, m * y - 4 in
  let intersection_M := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let intersection_N := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let P := (*) -- Intersection point of lines MA1 and NA2 -- (*) 
  ∃ (P : ℝ × ℝ), P.fst = -1 :=
by
  sorry

end equation_of_hyperbola_P_lies_on_fixed_line_l205_205361


namespace sum_and_count_l205_205990

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count (x y : ℕ) (hx : x = sum_of_integers 30 50) (hy : y = count_even_integers 30 50) :
  x + y = 851 :=
by
  -- proof goes here
  sorry

end sum_and_count_l205_205990


namespace value_of_expression_l205_205623

theorem value_of_expression :
  (43 + 15)^2 - (43^2 + 15^2) = 2 * 43 * 15 :=
by
  sorry

end value_of_expression_l205_205623


namespace largest_integer_solving_inequality_l205_205723

theorem largest_integer_solving_inequality :
  ∃ (x : ℤ), (7 - 5 * x > 22) ∧ ∀ (y : ℤ), (7 - 5 * y > 22) → x ≥ y ∧ x = -4 :=
by
  sorry

end largest_integer_solving_inequality_l205_205723


namespace simplify_sqrt_expr_l205_205922

/-- Simplify the given radical expression and prove its equivalence to the expected result. -/
theorem simplify_sqrt_expr :
  (Real.sqrt (5 * 3) * Real.sqrt ((3 ^ 4) * (5 ^ 2)) = 225 * Real.sqrt 15) := 
by
  sorry

end simplify_sqrt_expr_l205_205922


namespace circumcircle_radius_of_triangle_aks_l205_205889

theorem circumcircle_radius_of_triangle_aks
  (r₁ r₂ : ℝ)
  (ω₁ ω₂ : set Point)
  (T K A B S : Point)
  (circumcircle_radius : ℝ)
  (h₀ : is_circle ω₁ ∧ radius ω₁ = r₁) 
  (h₁ : is_circle ω₂ ∧ radius ω₂ = r₂)
  (h₂ : tangent_at_point ω₁ ω₂ T)
  (h₃ : K ∈ ω₁ ∧ K ≠ T)
  (h₄ : tangent_line_intersects_at_points ω₁ ω₂ K A B)
  (h₅ : midpoint_arc_ab ω₂ A B S ∧ T ∉ arc_ab ω₂ A B) :
  circumcircle_radius = sqrt (r₂ * (r₂ - r₁)) := 
sorry

end circumcircle_radius_of_triangle_aks_l205_205889


namespace constant_expression_l205_205765

-- Suppose x is a real number
variable {x : ℝ}

-- Define the expression sum
def expr_sum (x : ℝ) : ℝ :=
|3 * x - 1| + |4 * x - 1| + |5 * x - 1| + |6 * x - 1| + 
|7 * x - 1| + |8 * x - 1| + |9 * x - 1| + |10 * x - 1| + 
|11 * x - 1| + |12 * x - 1| + |13 * x - 1| + |14 * x - 1| + 
|15 * x - 1| + |16 * x - 1| + |17 * x - 1|

-- The Lean statement of the problem to be proven
theorem constant_expression : (∃ x : ℝ, expr_sum x = 5) :=
sorry

end constant_expression_l205_205765


namespace find_two_digit_number_l205_205197

-- A type synonym for digit
def Digit := {n : ℕ // n < 10}

-- Define the conditions
variable (X Y : Digit)
-- The product of the digits is 8
def product_of_digits : Prop := X.val * Y.val = 8

-- When 18 is added, digits are reversed
def digits_reversed : Prop := 10 * X.val + Y.val + 18 = 10 * Y.val + X.val

-- The question translated to Lean: Prove that the two-digit number is 24
theorem find_two_digit_number (h1 : product_of_digits X Y) (h2 : digits_reversed X Y) : 10 * X.val + Y.val = 24 :=
  sorry

end find_two_digit_number_l205_205197


namespace john_weekly_calories_l205_205897

-- Define the calorie calculation for each meal type
def breakfast_calories : ℝ := 500
def morning_snack_calories : ℝ := 150
def lunch_calories : ℝ := breakfast_calories + 0.25 * breakfast_calories
def afternoon_snack_calories : ℝ := lunch_calories - 0.30 * lunch_calories
def dinner_calories : ℝ := 2 * lunch_calories

-- Total calories for Friday
def friday_calories : ℝ := breakfast_calories + morning_snack_calories + lunch_calories + afternoon_snack_calories + dinner_calories

-- Additional treats on Saturday and Sunday
def dessert_calories : ℝ := 350
def energy_drink_calories : ℝ := 220

-- Total calories for each day
def saturday_calories : ℝ := friday_calories + dessert_calories
def sunday_calories : ℝ := friday_calories + 2 * energy_drink_calories
def weekday_calories : ℝ := friday_calories

-- Proof statement
theorem john_weekly_calories : 
  friday_calories = 2962.5 ∧ 
  saturday_calories = 3312.5 ∧ 
  sunday_calories = 3402.5 ∧ 
  weekday_calories = 2962.5 :=
by 
  -- proof expressions would go here
  sorry

end john_weekly_calories_l205_205897


namespace equation_of_hyperbola_P_lies_on_fixed_line_l205_205359

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- The conditions provided in the problem
def focus := (-2 * Real.sqrt 5, 0)
def center := (0, 0)
def eccentricity : ℝ := Real.sqrt 5
def vertex1 := (-2, 0)
def vertex2 := (2, 0)
def passing_point := (-4, 0)

-- Finding the equation of the hyperbola
theorem equation_of_hyperbola : 
  ∀ (a b c : ℝ), 
  c = 2 * Real.sqrt 5 ∧ 
  eccentricity = Real.sqrt 5 ∧ 
  a = 2 ∧
  b = 4 → 
  hyperbola 2 4 :=
by
  sorry

-- Proving P lies on a fixed line
theorem P_lies_on_fixed_line :
  ∀ (m : ℝ), 
  let line : ℝ → ℝ := λ y, m * y - 4 in
  let intersection_M := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let intersection_N := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let P := (*) -- Intersection point of lines MA1 and NA2 -- (*) 
  ∃ (P : ℝ × ℝ), P.fst = -1 :=
by
  sorry

end equation_of_hyperbola_P_lies_on_fixed_line_l205_205359


namespace math_problem_equivalent_l205_205346

noncomputable theory

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then -(x - 1) ^ 2 + 2 else sorry

def g (x : ℝ) : ℝ := f x - 2^(-x) - 1

def h (x : ℝ) : ℝ := f x / 2^x

theorem math_problem_equivalent :
  (∀ x ∈ [-2, 0], f x = -(x + 1)^2 + 2) ∧
  (∀ x1 x2, g x1 = 0 ∧ g x2 = 0 ∧ 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 → x1 + x2 ≥ 2) ∧
  (∃ m, (∀ x ∈ [4, 6], h x ≥ m) ∧ m = 1 / 64) ∧
  (f (Real.log 3 4) < f (Real.log 4 (5 / 16))) :=
by
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end math_problem_equivalent_l205_205346


namespace Nina_now_l205_205938

def Lisa_age (l m n : ℝ) := l + m + n = 36
def Nina_age (l n : ℝ) := n - 5 = 2 * l
def Mike_age (l m : ℝ) := m + 2 = (l + 2) / 2

theorem Nina_now (l m n : ℝ) (h1 : Lisa_age l m n) (h2 : Nina_age l n) (h3 : Mike_age l m) : n = 34.6 := by
  sorry

end Nina_now_l205_205938


namespace solution_for_a_if_fa_eq_a_l205_205767

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 2) / (x - 2)

theorem solution_for_a_if_fa_eq_a (a : ℝ) (h : f a = a) : a = -1 :=
sorry

end solution_for_a_if_fa_eq_a_l205_205767


namespace rope_length_l205_205246

-- Definitions from the conditions
def h := 48  -- horizontal distance to the wall, in cm
def d := 3   -- distance from the end of the rope to the floor, in cm

-- Main theorem statement
theorem rope_length (L : ℝ) (L ≥ 0) : 
  let leg1 := L - d in 
  let leg2 := h in 
  leg1^2 + leg2^2 = L^2 → L = 388.5 :=
sorry

end rope_length_l205_205246


namespace equation_of_hyperbola_point_P_fixed_line_l205_205367

-- Definitions based on conditions
def center_origin := (0, 0)
def left_focus := (-2 * real.sqrt 5, 0)
def eccentricity := real.sqrt 5
def left_vertex := (-2, 0)
def right_vertex := (2, 0)
def point_passed_through_line := (-4, 0)

-- The equation of the hyperbola that satisfies the given conditions.
theorem equation_of_hyperbola :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ 
    (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ 
      (2 * real.sqrt 5)^2 = a^2 + b^2) :=
by
  use [2, 4]
  split
  exact rfl
  split
  exact rfl
  sorry

-- Proving P lies on the fixed line x = -1 for given lines and vertices conditions
theorem point_P_fixed_line :
  ∀ m : ℝ, ∀ y1 y2 x1 x2 : ℝ, 
    x1 = m * y1 - 4 ∧ x2 = m * y2 - 4 ∧ 
    (y1 + y2 = (32 * m) / ((4 * m^2) - 1)) ∧
    (y1 * y2 = 48 / ((4 * m^2) - 1)) ∧ 
    (x1 + 2) ≠ 0 ∧ 
    (x2 - 2) ≠ 0 → 
    (P : ℝ × ℝ) ((P.1, P.2) = (-1, P.2) ∧ P.2 = (y1 * (x + 2) / x1 ≠ -2 ∧ y2 * (x - 2) / x2 ≠ 2)) := 
by
  intros
  sorry

end equation_of_hyperbola_point_P_fixed_line_l205_205367


namespace find_valid_n_l205_205049

def append_fives (n : ℕ) : ℕ := 1200 * 6^(10*n+2) + (5 * (6^0 + 6^1 + ... + 6^(10*n+1)))

def has_two_distinct_prime_factors (x : ℕ) : Prop :=
  let factors := factorization x
  factors.keys.to_finset.card = 2

theorem find_valid_n (n : ℕ) (x : ℕ) :
  x = append_fives n →
  has_two_distinct_prime_factors x :=
sorry

end find_valid_n_l205_205049


namespace lean_proof_problem_l205_205493

section

variable {R : Type*} [AddCommGroup R]

def is_odd_function (f : ℝ → R) : Prop :=
  ∀ x, f (-x) = -f x

theorem lean_proof_problem (f: ℝ → ℝ) (h_odd: is_odd_function f)
    (h_cond: f 3 + f (-2) = 2) : f 2 - f 3 = -2 :=
by
  sorry

end

end lean_proof_problem_l205_205493


namespace collinear_MNT_l205_205497

variables {A B P Q R S M N T : Type}
variables [add_comm_group A] [module ℝ A]

-- Ratios
variables {a b : ℝ} (h1 : ∃ k : ℝ, k ≠ 0 ∧ (R = k • A + (1 - k) • P) ∧ (S = k • B + (1 - k) • Q)) 
-- Points and their positional relationships
variables {e f : ℝ} (h2 : ∃ λ : ℝ, λ = e / (e + f) ∧
  (M = A + λ • (B - A)) ∧
  (N = (f / (e + f)) • P + (e / (e + f)) • Q) ∧
  (T = (f / (e + f)) • R + (e / (e + f)) • S))

theorem collinear_MNT (h1 : ∃ k : ℝ, k ≠ 0 ∧ (R = k • A + (1 - k) • P) ∧ (S = k • B + (1 - k) • Q))
  (h2 : ∃ λ : ℝ, (λ = e / (e + f)) ∧
    (M = A + λ • (B - A)) ∧
    (N = (f / (e + f)) • P + (e / (e + f)) • Q) ∧
    (T = (f / (e + f)) • R + (e / (e + f)) • S)) : 
  collinear A B P M N T := sorry

end collinear_MNT_l205_205497


namespace number_of_consistent_subsets_of_order_1_l205_205495

def configuration (A : set (ℕ × ℕ)) : set (ℕ × ℕ) :=
{p | ( ∃ i, p = (a i, b i) ∧ 1 ≤ i ∧ i ≤ 10 ) ∨
     ( ∃ i, p = (a i, a (i + 1)) ∧ 1 ≤ i ∧ i ≤ 9 ) ∨
     ( ∃ i, p = (b i, b (i + 1)) ∧ 1 ≤ i ∧ i ≤ 9 )}

def is_consistent_subset (C : set (ℕ × ℕ)) (s : set (ℕ × ℕ)) : Prop :=
∀ p ∈ s, p ∈ C

-- The main theorem stating the problem
theorem number_of_consistent_subsets_of_order_1 (C : set (ℕ × ℕ)) : 
  (∀ s : set (ℕ × ℕ), is_consistent_subset C s → s.card = 89) :=
sorry

end number_of_consistent_subsets_of_order_1_l205_205495


namespace final_weight_is_sixteen_l205_205869

def initial_weight : ℤ := 0
def weight_after_jellybeans : ℤ := initial_weight + 2
def weight_after_brownies : ℤ := weight_after_jellybeans * 3
def weight_after_more_jellybeans : ℤ := weight_after_brownies + 2
def final_weight : ℤ := weight_after_more_jellybeans * 2

theorem final_weight_is_sixteen : final_weight = 16 := by
  sorry

end final_weight_is_sixteen_l205_205869


namespace no_such_function_exists_l205_205732

noncomputable def f : ℕ → ℕ := sorry

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), (∀ n > 1, f n = f (f (n-1)) + f (f (n+1))) ∧ (∀ n, f n > 0) :=
sorry

end no_such_function_exists_l205_205732


namespace prob_neither_prime_nor_composite_l205_205830

theorem prob_neither_prime_nor_composite :
  (1 / 95 : ℚ) = 1 / 95 := by
  sorry

end prob_neither_prime_nor_composite_l205_205830


namespace sum_of_areas_equal_l205_205463

theorem sum_of_areas_equal (A B C D M : Point) (AM BM CM DM: ℝ)
  (h_AM: AM = dist A M) (h_BM: BM = dist B M) (h_CM: CM = dist C M) (h_DM: DM = dist D M)
  (h_AM2_CM2_eq_BM2_DM2: AM^2 + CM^2 = BM^2 + DM^2) :
  let S_A := pi * AM^2 / 4
      S_B := pi * BM^2 / 4
      S_C := pi * CM^2 / 4
      S_D := pi * DM^2 / 4
  in S_A + S_C = S_B + S_D :=
by {
  intro h_AMON S_A S_B S_C S_D,
  sorry
}

end sum_of_areas_equal_l205_205463


namespace tan_B_tan_C_eq_7_over_6_l205_205457

variables {A B C H D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace H] [MetricSpace D]

-- Define a triangle ABC with orthocenter H, and altitude BD from B
variables {triangle_ABC : Triangle A B C} {orthocenter_H : Orthocenter triangle_ABC H}

-- Altitude BD splits into segments HD = 4 and HB = 10
axiom altitude_BD : Segment B D
axiom HD_eq_4 : length (Segment H D) = 4
axiom HB_eq_10 : length (Segment H B) = 10

-- Define the tangent of angles B and C in the triangle
variable (tan_B tan_C : ℝ)

-- The goal is to show that the product of these tangents equals 7/6
theorem tan_B_tan_C_eq_7_over_6 :
  tan_B * tan_C = 7 / 6 :=
sorry

end tan_B_tan_C_eq_7_over_6_l205_205457


namespace clay_capacity_second_box_l205_205220

-- Define the dimensions and clay capacity of the first box
def height1 : ℕ := 4
def width1 : ℕ := 2
def length1 : ℕ := 3
def clay1 : ℕ := 24

-- Define the dimensions of the second box
def height2 : ℕ := 3 * height1
def width2 : ℕ := 2 * width1
def length2 : ℕ := length1

-- The volume relation
def volume_relation (height width length clay: ℕ) : ℕ :=
  height * width * length * clay

theorem clay_capacity_second_box (height1 width1 length1 clay1 : ℕ) (height2 width2 length2 : ℕ) :
  height1 = 4 →
  width1 = 2 →
  length1 = 3 →
  clay1 = 24 →
  height2 = 3 * height1 →
  width2 = 2 * width1 →
  length2 = length1 →
  volume_relation height2 width2 length2 1 = 6 * volume_relation height1 width1 length1 1 →
  volume_relation height2 width2 length2 clay1 / volume_relation height1 width1 length1 1 = 144 :=
by
  intros h1 w1 l1 c1 h2 w2 l2 vol_rel
  sorry

end clay_capacity_second_box_l205_205220


namespace dog_ate_cost_6_l205_205475

noncomputable def totalCost : ℝ := 4 + 2 + 0.5 + 2.5
noncomputable def costPerSlice : ℝ := totalCost / 6
noncomputable def slicesEatenByDog : ℕ := 6 - 2
noncomputable def costEatenByDog : ℝ := slicesEatenByDog * costPerSlice

theorem dog_ate_cost_6 : costEatenByDog = 6 := by
  sorry

end dog_ate_cost_6_l205_205475


namespace equation_of_hyperbola_point_P_fixed_line_l205_205370

-- Definitions based on conditions
def center_origin := (0, 0)
def left_focus := (-2 * real.sqrt 5, 0)
def eccentricity := real.sqrt 5
def left_vertex := (-2, 0)
def right_vertex := (2, 0)
def point_passed_through_line := (-4, 0)

-- The equation of the hyperbola that satisfies the given conditions.
theorem equation_of_hyperbola :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ 
    (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ 
      (2 * real.sqrt 5)^2 = a^2 + b^2) :=
by
  use [2, 4]
  split
  exact rfl
  split
  exact rfl
  sorry

-- Proving P lies on the fixed line x = -1 for given lines and vertices conditions
theorem point_P_fixed_line :
  ∀ m : ℝ, ∀ y1 y2 x1 x2 : ℝ, 
    x1 = m * y1 - 4 ∧ x2 = m * y2 - 4 ∧ 
    (y1 + y2 = (32 * m) / ((4 * m^2) - 1)) ∧
    (y1 * y2 = 48 / ((4 * m^2) - 1)) ∧ 
    (x1 + 2) ≠ 0 ∧ 
    (x2 - 2) ≠ 0 → 
    (P : ℝ × ℝ) ((P.1, P.2) = (-1, P.2) ∧ P.2 = (y1 * (x + 2) / x1 ≠ -2 ∧ y2 * (x - 2) / x2 ≠ 2)) := 
by
  intros
  sorry

end equation_of_hyperbola_point_P_fixed_line_l205_205370


namespace sum_floor_arithmetic_progression_l205_205714

theorem sum_floor_arithmetic_progression :
  ∑ k in Finset.range 102, Int.floor ((2 : ℝ) + 0.8 * k) = 4294.2 :=
by
  sorry

end sum_floor_arithmetic_progression_l205_205714


namespace cost_of_24_bananas_eq_11_oranges_l205_205273

variable (cost : Type) [LinearOrderedField cost]
variable (banana apple orange : cost)

-- Conditions
-- 4 bananas cost as much as 3 apples
def cost_equivalence_1 : cost := 4 * banana = 3 * apple

-- 8 apples cost as much as 5 oranges
def cost_equivalence_2 : cost := 8 * apple = 5 * orange

-- Problem statement
theorem cost_of_24_bananas_eq_11_oranges (cost : Type) [LinearOrderedField cost]
  (banana apple orange : cost)
  (h1 : 4 * banana = 3 * apple)
  (h2 : 8 * apple = 5 * orange) :
  24 * banana = 11 * orange :=
sorry

end cost_of_24_bananas_eq_11_oranges_l205_205273


namespace least_pos_int_div_by_four_distinct_primes_l205_205097

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205097


namespace double_prime_dates_2012_l205_205932

-- Definitions of prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definitions for leap year and month days
def is_leap_year (year : ℕ) : Prop := (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def days_in_february (year : ℕ) : ℕ := if is_leap_year year then 29 else 28

-- Condition for the prime months
def prime_months := {2, 3, 5, 7, 11}

-- Definition for the day being prime and the sum of the day and month is also prime
def is_double_prime_date (month day : ℕ) : Prop :=
  is_prime month ∧ month ∈ prime_months ∧ is_prime day ∧ is_prime (day + month)

-- Main Theorem
theorem double_prime_dates_2012 : 
  (Σ month in prime_months, Σ day in finset.range 32, if is_double_prime_date month day then 1 else 0) = 28 :=
by
  sorry

end double_prime_dates_2012_l205_205932


namespace function_crosses_asymptote_at_value_l205_205763

def g (x : ℝ) : ℝ := (3 * x^2 - 6 * x - 9) / (x^2 - 5 * x + 6)

theorem function_crosses_asymptote_at_value :
  g (9 / 5) = 3 := 
sorry

end function_crosses_asymptote_at_value_l205_205763


namespace percentage_of_360_is_165_6_l205_205186

theorem percentage_of_360_is_165_6 :
  (165.6 / 360) * 100 = 46 :=
by
  sorry

end percentage_of_360_is_165_6_l205_205186


namespace water_distribution_scheme_l205_205653

theorem water_distribution_scheme (a b c : ℚ) : 
  a + b + c = 1 ∧ 
  (∀ x : ℂ, ∃ n : ℕ, x^n = 1 → x = 1) ∧
  (∀ (x : ℂ), (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11 + x^12 + x^13 + x^14 + x^15 + x^16 + x^17 + x^18 + x^19 + x^20 + x^21 + x^22 = 0) → false) → 
  a = 0 ∧ b = 0 ∧ c = 1 :=
by
  sorry

end water_distribution_scheme_l205_205653


namespace solve_linear_system_l205_205927

open Matrix

-- Define a namespace
namespace LinearSystem

-- Define the system of linear equations in matrix form
def A : Matrix (Fin 4) (Fin 4) ℚ := ![
  ![-1, -2, -6, 3],
  ![2, 5, 14, -7],
  ![3, 7, 20, -10],
  ![0, -1, -2, 1]
]

def b : Vector ℚ 4 := ![-1, 3, 4, -1]

-- Define the parametric solution
def x (α β : ℚ) : Vector ℚ 4 := ![
  -1 - 2 * α + β,
  1 - 2 * α + β,
  α,
  β
]

-- The main theorem to prove
theorem solve_linear_system : 
  ∃ (α β : ℚ), A.mulVec (x α β) = b := 
by
  sorry 

end LinearSystem

end solve_linear_system_l205_205927


namespace borrowed_amount_correct_l205_205636

variables (monthly_payment : ℕ) (months : ℕ) (total_payment : ℕ) (borrowed_amount : ℕ)

def total_payment_calculation (monthly_payment : ℕ) (months : ℕ) : ℕ :=
  monthly_payment * months

theorem borrowed_amount_correct :
  monthly_payment = 15 →
  months = 11 →
  total_payment = total_payment_calculation monthly_payment months →
  total_payment = 110 * borrowed_amount / 100 →
  borrowed_amount = 150 :=
by
  intros h1 h2 h3 h4
  sorry

end borrowed_amount_correct_l205_205636


namespace acute_angles_equal_45_degrees_l205_205690

theorem acute_angles_equal_45_degrees 
  (α1 α2 α3 : ℝ) 
  (h1 : sin α1 = cos α2) 
  (h2 : sin α2 = cos α3) 
  (h3 : sin α3 = cos α1) 
  (hα1_pos : 0 < α1) (hα1_lt90 : α1 < π / 2)
  (hα2_pos : 0 < α2) (hα2_lt90 : α2 < π / 2)
  (hα3_pos : 0 < α3) (hα3_lt90 : α3 < π / 2) :
  α1 = π / 4 ∧ α2 = π / 4 ∧ α3 = π / 4 := 
sorry

end acute_angles_equal_45_degrees_l205_205690


namespace saeyoung_yen_value_l205_205704

-- Define the exchange rate
def exchange_rate : ℝ := 17.25

-- Define Saeyoung's total yuan
def total_yuan : ℝ := 1000 + 10

-- Define the total yen based on the exchange rate
def total_yen : ℝ := total_yuan * exchange_rate

-- State the theorem
theorem saeyoung_yen_value : total_yen = 17422.5 :=
by
  sorry

end saeyoung_yen_value_l205_205704


namespace range_of_a_l205_205802

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a ^ x

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 2 / 3) :=
by
  sorry

end range_of_a_l205_205802


namespace sweet_numbers_count_l205_205640

def Zor_rule (n : ℕ) : ℕ :=
if n ≤ 30 then 3 * n else n - 15

def is_sequence (start : ℕ) (seq : ℕ → ℕ) : Prop :=
seq 0 = start ∧ ∀ n, seq (n + 1) = Zor_rule (seq n)

def is_sweet (G : ℕ) : Prop :=
¬(∃ seq : ℕ → ℕ, is_sequence G seq ∧ ∃ n, seq n = 18)

def number_of_sweet_numbers : ℕ :=
(nat.filter is_sweet (list.fin_range 50)).length

theorem sweet_numbers_count :
  number_of_sweet_numbers = 31 :=
sorry

end sweet_numbers_count_l205_205640


namespace find_hyperbola_fixed_line_through_P_l205_205392

-- Define the conditions of the problem
def center_origin (C : Type) : Prop := 
  C.center = (0, 0)

def left_focus_at (C : Type) : Prop :=
  C.left_focus = (-2 * Real.sqrt 5, 0)

def eccentricity_sqrt_5 (C : Type) : Prop :=
  C.eccentricity = Real.sqrt 5

-- Define the specific hyperbola equation based on the conditions
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 16) = 1

-- State the results to prove
theorem find_hyperbola (C : Type) :
  center_origin C →
  left_focus_at C →
  eccentricity_sqrt_5 C →
  ∀ x y : ℝ, hyperbola_equation x y :=
sorry

-- Define the fixed line for point P
def fixed_line (P : Type) : Prop :=
  P.x = -1

-- State the results to prove for point P
theorem fixed_line_through_P (M N A1 A2 P : Type) (x y : ℝ) :
  (M.line_passing_through (Point.mk (-4, 0))).intersects_left_branch (N : Point) →
  M.in_second_quadrant →
  (A1.coords = (-2, 0)) ∧ (A2.coords = (2, 0)) →
  Point.mk x y = P →
  fixed_line P :=
sorry

end find_hyperbola_fixed_line_through_P_l205_205392


namespace S_lies_on_PQ_l205_205886

open Set

-- Define the square ABCD and its properties
variables {A B C D S P Q : Point}
variables (k k' : Circle)
variables [square : Square A B C D]

-- Define S as the intersection of the diagonals AC and BD
axiom S_is_intersection : S = intersection (line A C) (line B D)

-- Define the circles k and k'
axiom k_contains_A_C : k.contains A ∧ k.contains C
axiom k'_contains_B_D : k'.contains B ∧ k'.contains D

-- Define the intersection points P and Q of the circles
axiom intersection_points : P ≠ Q ∧ 
                            k.contains P ∧ k'.contains P ∧ 
                            k.contains Q ∧ k'.contains Q

-- Prove that S lies on line PQ
theorem S_lies_on_PQ : lies_on S (line P Q) :=
sorry

end S_lies_on_PQ_l205_205886


namespace find_functions_l205_205329

theorem find_functions (a b c : ℝ) (f g : ℝ → ℝ)
    (h_eqf : ∀ x, f x = 2 * x ^ 3 + a * x)
    (h_eqg : ∀ x, g x = b * x ^ 2 + c)
    (h_P_f : f 2 = 0)
    (h_P_g : g 2 = 0)
    (h_common_tangent : (deriv f 2 = deriv g 2))
    : f = (λ x, 2 * x ^ 3 - 8 * x) ∧ g = (λ x, 4 * x ^ 2 - 16) :=
by
  sorry

end find_functions_l205_205329


namespace num_vehicles_now_l205_205843

noncomputable def remaining_vehicles (initial_cars initial_motorcycles initial_buses : ℕ)
  (x y z q r s : ℝ) : ℕ :=
  let cars_left := (x / 100) * initial_cars
  let cars_entered := (q / 100) * cars_left
  let remaining_cars := initial_cars - Int.ofNat cars_left + Int.ofNat cars_entered
  
  let motorcycles_left := (y / 100) * initial_motorcycles
  let motorcycles_entered := (r / 100) * motorcycles_left
  let remaining_motorcycles := initial_motorcycles - Int.ofNat motorcycles_left + Int.ofNat motorcycles_entered
  
  let buses_left := (z / 100) * initial_buses
  let buses_entered := (s / 100) * buses_left
  let remaining_buses := initial_buses - Int.ofNat buses_left + Int.ofNat buses_entered
  
  remaining_cars + remaining_motorcycles + remaining_buses

theorem num_vehicles_now : remaining_vehicles 80 25 15 16.25 20 33.33 25 15 10 = 100 := by
  sorry

end num_vehicles_now_l205_205843


namespace log_of_sum_seq_equality_l205_205328

noncomputable theory
open Real

def f (x : ℝ) : ℝ :=
  (1 / 3) * x ^ 3 - 4 * x ^ 2 + x

def a_n_seq (n : ℕ) : ℤ :=
  sorry -- This depends on further exploration of the sequence definition

def extreme_points (x : ℕ) : Prop :=
  f x = x ∧ f (x + 1) < x ∧ f (x - 1) < x -- Assuming 'extreme points' means local max/min

theorem log_of_sum_seq_equality :
  (∀ n : ℕ, a_n_seq n + 2 = 2 * n + 1 - a_n_seq n) →
  (extreme_points 201 ∧ extreme_points 2014) →
  log 2 (a_n_seq 2000 + a_n_seq 2012 + a_n_seq 208 + a_n_seq 2030) = 4 :=
by {
  intro h_seq h_extreme,
  sorry
}

end log_of_sum_seq_equality_l205_205328


namespace length_DE_l205_205545

theorem length_DE (AB : ℝ) (h_base : AB = 15) (DE_parallel : ∀ x y z : Triangle, True) (area_ratio : ℝ) (h_area_ratio : area_ratio = 0.25) : 
  ∃ DE : ℝ, DE = 7.5 :=
by
  sorry

end length_DE_l205_205545


namespace tangerines_left_l205_205737

def total_tangerines : ℕ := 27
def tangerines_eaten : ℕ := 18

theorem tangerines_left : total_tangerines - tangerines_eaten = 9 := by
  sorry

end tangerines_left_l205_205737


namespace equation_of_hyperbola_point_P_on_fixed_line_l205_205373

-- Definition and conditions
def hyperbola_center_origin := (0, 0)
def hyperbola_left_focus := (-2 * Real.sqrt 5, 0)
def hyperbola_eccentricity := Real.sqrt 5
def point_on_line_through_origin := (-4, 0)
def point_M_in_second_quadrant (M : ℝ × ℝ) := M.1 < 0 ∧ M.2 > 0
def vertices_A1_A2 := ((-2, 0), (2, 0))

theorem equation_of_hyperbola (c a b : ℝ) (h_c : c = 2 * Real.sqrt 5)
  (h_e : hyperbola_eccentricity = Real.sqrt 5) (h_a : a = 2) (h_b : b = 4) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 16) = 1) := 
by
  intros x y h_x h_y
  have h : c^2 = a^2 + b^2 := sorry
  have h2 : hyperbola_eccentricity = c / a := sorry
  exact sorry

theorem point_P_on_fixed_line (M N P : ℝ × ℝ) (A1 A2 : ℝ × ℝ)
  (h_vertices : vertices_A1_A2 = (A1, A2))
  (h_M_in_2nd : point_M_in_second_quadrant M) 
  (h_P_intersects : ∀ t : ℝ, P = (t, (A1.2 * A2.1 - A1.1 * A2.2) / (A1.1 - A2.1))) :
  (P.1 = -1) := sorry

end equation_of_hyperbola_point_P_on_fixed_line_l205_205373


namespace klmn_parallelogram_center_l205_205899

open_locale big_operators

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

-- Definition of a parallelogram
structure parallelogram (A B C D : V) : Prop :=
(parallel : ∃ u v : V, u ≠ 0 ∧ v ≠ 0 ∧ B = A + u ∧ C = B + v ∧ D = A + v ∧ D = C + u)

-- Given a parallelogram ABCD
variables {A B C D M N K L O : V}
variables {r : ℝ} (hr : 0 < r ∧ r < 1)
variables (h_parallelogram : parallelogram A B C D)

-- Points M, N, K, L divide the sides in the ratio r : (1-r)
variables (hM : M = (1 - r) • A + r • B)
variables (hN : N = (1 - r) • B + r • C)
variables (hK : K = (1 - r) • C + r • D)
variables (hL : L = (1 - r) • D + r • A)

-- O is the center of the parallelogram ABCD
variables (hO1 : O = (1 / 2) • (A + C))
variables (hO2 : O = (1 / 2) • (B + D))

-- The statement that KLMN is a parallelogram and its center coincides with the center of ABCD
theorem klmn_parallelogram_center :
  parallelogram K L M N ∧ (O = (1 / 2) • (K + M)) ∧ (O = (1 / 2) • (L + N)) :=
sorry

end klmn_parallelogram_center_l205_205899


namespace y_value_when_x_is_zero_l205_205536

theorem y_value_when_x_is_zero :
  ∀ (t : ℚ), (0 = 3 - 2 * t) → (∃ y : ℚ, y = 5 * t + 6 ∧ y = 27 / 2) :=
by 
  intro t ht,
  use (5 * t + 6),
  split,
  { refl },
  { sorry }

end y_value_when_x_is_zero_l205_205536


namespace least_pos_int_div_by_four_distinct_primes_l205_205103

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205103


namespace least_pos_int_div_by_four_distinct_primes_l205_205095

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205095


namespace restore_trapezoid_l205_205511

variables (O A B C D K L M N : Point)
variables (OK EF l : Line)
variables [Trapezoid_ABCD : Trapezoid ABCD]

-- Conditions
axiom AD_parallel_BC : Parallel AD BC
axiom O_intersection_of_diagonals : Intersection (diagonal AC) (diagonal BD) = Some O
axiom OK_perpendicular_AD : Perpendicular OK AD
axiom EF_midline : Midline EF AD BC
axiom l_through_K_perpendicular_OK : Perpendicular l (LineThrough K O)

-- Define points L and N based on the problem statement
noncomputable def L : Point := Intersection l (LineThrough O M)
noncomputable def N : Point := SymmetricPoint L M

-- Theorem
theorem restore_trapezoid : RestorableTrapezoid OK EF :=
  sorry

end restore_trapezoid_l205_205511


namespace next_year_property_appears_l205_205959

def no_smaller_rearrangement (n: Nat) : Prop :=
  ∀ (l: List Nat), (l.permutations.map (λ p, p.foldl (λ acc d, acc * 10 + d) 0)).all (λ m, m >= n)

def next_year_with_property (current: Nat) : Nat :=
  if h : current = 2022 then 2022
  else if ∃ n, n > current ∧ no_smaller_rearrangement n then
    Classical.some (Classical.some_spec h)
  else current

theorem next_year_property_appears : next_year_with_property 2009 = 2022 := by
  sorry

end next_year_property_appears_l205_205959


namespace minimum_value_tangent_distance_l205_205300

theorem minimum_value_tangent_distance 
  (P Q N : ℝ × ℝ) (h1 : ∀ Q, (Q.1 - 3)^2 + (Q.2 - 4)^2 = 1 ∧ (P.1 - 3)^2 + (P.2 - 4)^2 = (P.1)^2 + (P.2)^2 + 1)
  (h2 : dist P Q = dist P ⟨0, 0⟩) :
  dist P Q = 12 / 5 := by
sorry

end minimum_value_tangent_distance_l205_205300


namespace smallest_pos_int_div_by_four_primes_l205_205114

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205114


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205158

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205158


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205085

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205085


namespace least_positive_integer_divisible_by_four_primes_l205_205124

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205124


namespace freddy_travel_time_l205_205735

/-
Conditions:
1. Eddy's travel time is 3 hours.
2. The distance from city A to city B is 600 km.
3. The distance from city A to city C is 460 km.
4. The ratio of Eddy's speed to Freddy's speed is 1.7391304347826086.
-/
def travel_time_eddy : ℝ := 3
def distance_AB : ℝ := 600
def distance_AC : ℝ := 460
def speed_ratio : ℝ := 1.7391304347826086

/-
Theorem:
Prove that Freddy's travel time from city A to city C is 4 hours.
-/
theorem freddy_travel_time :
  let speed_eddy := distance_AB / travel_time_eddy,
      speed_freddy := speed_eddy / speed_ratio,
      travel_time_freddy := distance_AC / speed_freddy
  in travel_time_freddy = 4 := by
  sorry

end freddy_travel_time_l205_205735


namespace calculate_retail_price_l205_205675

/-- Define the wholesale price of the machine. -/
def wholesale_price : ℝ := 90

/-- Define the profit rate as 20% of the wholesale price. -/
def profit_rate : ℝ := 0.20

/-- Define the discount rate as 10% of the retail price. -/
def discount_rate : ℝ := 0.10

/-- Calculate the profit based on the wholesale price. -/
def profit : ℝ := profit_rate * wholesale_price

/-- Calculate the selling price after the discount. -/
def selling_price (retail_price : ℝ) : ℝ := retail_price * (1 - discount_rate)

/-- Calculate the total selling price as the wholesale price plus profit. -/
def total_selling_price : ℝ := wholesale_price + profit

/-- State the theorem we need to prove. -/
theorem calculate_retail_price : ∃ R : ℝ, selling_price R = total_selling_price → R = 120 := by
  sorry

end calculate_retail_price_l205_205675


namespace train_passing_time_l205_205196

noncomputable def relative_speed_km_per_hr (v_train v_man : ℝ) : ℝ :=
  v_train + v_man

noncomputable def km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * (5 / 18)

theorem train_passing_time :
  let L := 550
  let v_train := 60
  let v_man := 6
  let v_rel := relative_speed_km_per_hr v_train v_man
  let v_rel_m_per_s := km_per_hr_to_m_per_s v_rel
  let t := L / v_rel_m_per_s
  t ≈ 30 :=
by
  -- Definitions of constants
  let L := 550
  let v_train := 60
  let v_man := 6
  let v_rel := relative_speed_km_per_hr v_train v_man
  let v_rel_m_per_s := km_per_hr_to_m_per_s v_rel
  let t := L / v_rel_m_per_s
  -- Given conditions
  have h1 : v_rel = 66 := by
    sorry  -- We can verify this directly from the provided calculation
  have h2 : v_rel_m_per_s = 18.333 (approximately) := by
    sorry  -- We can verify this from the conversion factor
  -- Main theorem
  show t ≈ 30, from
    sorry  -- Which can be computed as 550 / 18.333

end train_passing_time_l205_205196


namespace equivalent_proof_problem_l205_205489

noncomputable theory

open Complex

def omega := exp (2 * pi * I / 3)

lemma omega_properties : omega^3 = 1 ∧ omega ≠ 1 :=
begin
  split,
  {
    rw [pow_succ, pow_two, exp_mul_I_pi_div, exp_mul_I_pi_div],
    norm_num,
    use [0, rfl], -- omega^3 = 1
  },
  {
    exact Complex.exp_ne_one_of_ne_zero (ne_of_gt (by norm_num)),
  }
end

lemma transformed_expression : (omega - 2 * omega^2 + 2)^4 + (2 + 2 * omega - omega^2)^4 = -257 :=
by
  let e1 := (omega - 2 * omega^2 + 2),
  let e2 := (2 + 2 * omega - omega^2),
  calc
    (e1^4 + e2^4)
    = (_ : (ω - 2 * ω^2 + 2)^4 + (_ : (2 + 2 * ω - ω^2)^4)) := sorry
    = _ : by sorry
    = -257 := sorry

-- The equivalence proof problem as a Lean statement
theorem equivalent_proof_problem : (omega - 2 * omega^2 + 2)^4 + (2 + 2 * omega - omega^2)^4 = -257 :=
sorry

end equivalent_proof_problem_l205_205489


namespace total_pine_and_fir_trees_l205_205664

def total_trees : ℕ := 520

def pine_trees (total : ℕ) : ℕ := (1 / 3 : ℚ * total).natAbs

def fir_trees (total : ℕ) : ℕ := (0.25 * total).natAbs

theorem total_pine_and_fir_trees :
  pine_trees total_trees + fir_trees total_trees = 390 := by
  sorry

end total_pine_and_fir_trees_l205_205664


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205155

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205155


namespace Yoque_borrowed_150_l205_205638

noncomputable def Yoque_borrowed_amount (X : ℝ) : Prop :=
  1.10 * X = 11 * 15

theorem Yoque_borrowed_150 (X : ℝ) : Yoque_borrowed_amount X → X = 150 :=
by
  -- proof will be filled in
  sorry

end Yoque_borrowed_150_l205_205638


namespace find_a_b_find_k_l205_205793

noncomputable def f (a b x : ℝ) : ℝ := a * 4^x - a * 2^(x+1) + 1 - b

theorem find_a_b (a b : ℝ) :
  (∀ x ∈ set.Icc 1 2, f a b x ≤ 9) ∧ (∀ x ∈ set.Icc 1 2, f a b x ≥ 1) ∧ (∀ x ∈ set.Icc 1 2, ∃ c, is_max_on f {x ∈ set.Icc 1 2}) ∧ (∀ x ∈ set.Icc 1 2, ∃ m, is_min_on f {x ∈ set.Icc 1 2}) → 
  a = 1 ∧ b = 0 := sorry

theorem find_k (k : ℝ) :
  (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f 1 0 x - k * 4^x ≥ 0) → k ≤ 1 := sorry

end find_a_b_find_k_l205_205793


namespace length_of_chord_l205_205744

noncomputable def lineEquation (x y : ℝ) : Prop :=
  x + y + 1 = 0

noncomputable def circleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y - 5 = 0

theorem length_of_chord : 
  let center := (2 : ℝ, -1 : ℝ) in 
  let radius := Real.sqrt 10 in
  let d := Real.sqrt 2 in
  let chord_length := 2 * Real.sqrt (radius^2 - d^2) in
  chord_length = 4 * Real.sqrt 2 :=
sorry

end length_of_chord_l205_205744


namespace lucy_payment_proof_l205_205895

-- Prices and quantities of the fruits
def grape_price_kg := 74
def grape_kg := 6
def mango_price_kg := 59
def mango_kg := 9
def apple_price_kg := 45
def apple_kg := 4
def orange_price_kg := 32
def orange_kg := 12

-- Tax and discounts information
def grape_tax_rate := 0.10
def apple_tax_rate := 0.05
def promotion_discount := 5 -- dollars
def grape_apple_discount_rate := 0.07
def mango_orange_discount_rate := 0.05

-- Final computation of amount paid
noncomputable def final_amount_paid : ℝ :=
  let grape_cost := grape_price_kg * grape_kg
  let grape_tax := grape_cost * grape_tax_rate
  let grape_cost_with_tax := grape_cost + grape_tax
  let apple_cost := apple_price_kg * apple_kg
  let apple_tax := apple_cost * apple_tax_rate
  let apple_cost_with_tax := apple_cost + apple_tax
  let total_grape_apple_cost := grape_cost_with_tax + apple_cost_with_tax
  let grape_apple_discount := total_grape_apple_cost * grape_apple_discount_rate
  let final_grape_apple_cost := total_grape_apple_cost - grape_apple_discount
  let mango_cost := mango_price_kg * mango_kg
  let orange_cost := orange_price_kg * orange_kg
  let final_orange_cost := if orange_kg > 10 then orange_cost - promotion_discount else orange_cost
  let total_mango_orange_cost := mango_cost + final_orange_cost
  let mango_orange_discount := total_mango_orange_cost * mango_orange_discount_rate
  let final_mango_orange_cost := total_mango_orange_cost - mango_orange_discount
  final_grape_apple_cost + final_mango_orange_cost

theorem lucy_payment_proof : final_amount_paid = 1494.482 := by
  sorry

end lucy_payment_proof_l205_205895


namespace least_common_multiple_increments_l205_205599

theorem least_common_multiple_increments :
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  Nat.lcm (Nat.lcm (Nat.lcm a' b') c') d' = 8645 :=
by
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  sorry

end least_common_multiple_increments_l205_205599


namespace pyramid_surface_area_l205_205039

-- Definitions based on conditions from step a)
def pyramid := {cubes : ℕ // cubes = 30}
def cube_size := 1 -- Each cube measures 1m x 1m x 1m

-- Proof statement based on question and correct answer in step c)
theorem pyramid_surface_area (p : pyramid) : 
    total_surface_area p = 72 := 
sorry

end pyramid_surface_area_l205_205039


namespace alice_distance_from_start_l205_205239

def hexagon_side_length : ℝ := 3
def total_walk_distance : ℝ := 7

noncomputable def alice_final_position := 
  let p1 := (hexagon_side_length, 0) in
  let p2 := (hexagon_side_length / 2, hexagon_side_length * (Real.sqrt 3) / 2) in
  let final_x := p2.1 - 1 / 2 in
  let final_y := p2.2 + (Real.sqrt 3) / 2 in
  (final_x, final_y)

def distance_from_start (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem alice_distance_from_start :
  distance_from_start alice_final_position = Real.sqrt 13 :=
by sorry

end alice_distance_from_start_l205_205239


namespace value_of_a3_l205_205451

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem value_of_a3 (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 + a 4 = 20) :
  a 2 = 4 :=
sorry

end value_of_a3_l205_205451


namespace hyperbola_and_fixed_line_proof_l205_205387

-- Definitions based on conditions
def center_at_origin (C : Type) [Hyperbola C] : Prop :=
  C.center = (0, 0)

def left_focus (C : Type) [Hyperbola C] : Prop :=
  C.focus1 = (-2 * Real.sqrt 5, 0)

def eccentricity (C : Type) [Hyperbola C] : Prop :=
  C.eccentricity = Real.sqrt 5

def vertices (C : Type) [Hyperbola C] : Prop :=
  C.vertex1 = (-2, 0) ∧ C.vertex2 = (2, 0)

def passing_line_intersections (C : Type) (M N : Type) [Hyperbola C] : Prop :=
  ∃ l : Line, l ⨯ (-4, 0) ∧ l.intersects_hyperbola_left_branch C = {M, N}

def intersection (M N A1 A2 P : Type) [Hyperbola C] : Prop :=
  ∃ A1 A2, A1 = (-2, 0) ∧ A2 = (2, 0) ∧ (MA1.M ∩ NA2.N = P)

-- Proof problem statement
theorem hyperbola_and_fixed_line_proof (C : Type) [Hyperbola C] (M N P : Type)
  (h1 : center_at_origin C) (h2 : left_focus C) (h3 : eccentricity C)
  (h4 : passing_line_intersections C M N) (h5 : vertices C) (h6 : intersection M N C.vertex1 C.vertex2 P) : 
  (C.equation = (∀ x y, C.vertical_transverse_axis x y (x^2 / 4 - y^2 / 16)) ∧ 
   ∀ (x : ℝ), P.x = -1) :=
by sorry

end hyperbola_and_fixed_line_proof_l205_205387


namespace linear_combination_harmonic_l205_205022

-- Define the harmonic property for a function
def is_harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

-- The main statement to be proven in Lean
theorem linear_combination_harmonic
  (f g : ℤ × ℤ → ℝ) (a b : ℝ) (hf : is_harmonic f) (hg : is_harmonic g) :
  is_harmonic (fun p => a * f p + b * g p) :=
by
  sorry

end linear_combination_harmonic_l205_205022


namespace find_hyperbola_fixed_line_through_P_l205_205388

-- Define the conditions of the problem
def center_origin (C : Type) : Prop := 
  C.center = (0, 0)

def left_focus_at (C : Type) : Prop :=
  C.left_focus = (-2 * Real.sqrt 5, 0)

def eccentricity_sqrt_5 (C : Type) : Prop :=
  C.eccentricity = Real.sqrt 5

-- Define the specific hyperbola equation based on the conditions
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 16) = 1

-- State the results to prove
theorem find_hyperbola (C : Type) :
  center_origin C →
  left_focus_at C →
  eccentricity_sqrt_5 C →
  ∀ x y : ℝ, hyperbola_equation x y :=
sorry

-- Define the fixed line for point P
def fixed_line (P : Type) : Prop :=
  P.x = -1

-- State the results to prove for point P
theorem fixed_line_through_P (M N A1 A2 P : Type) (x y : ℝ) :
  (M.line_passing_through (Point.mk (-4, 0))).intersects_left_branch (N : Point) →
  M.in_second_quadrant →
  (A1.coords = (-2, 0)) ∧ (A2.coords = (2, 0)) →
  Point.mk x y = P →
  fixed_line P :=
sorry

end find_hyperbola_fixed_line_through_P_l205_205388


namespace correct_log_values_l205_205995

theorem correct_log_values (a b c : ℝ)
                          (log_027 : ℝ) (log_21 : ℝ) (log_1_5 : ℝ) (log_2_8 : ℝ)
                          (log_3 : ℝ) (log_5 : ℝ) (log_6 : ℝ) (log_7 : ℝ)
                          (log_8 : ℝ) (log_9 : ℝ) (log_14 : ℝ) :
  (log_3 = 2 * a - b) →
  (log_5 = a + c) →
  (log_6 = 1 + a - b - c) →
  (log_7 = 2 * (b + c)) →
  (log_9 = 4 * a - 2 * b) →
  (log_1_5 = 3 * a - b + c) →
  (log_14 = 1 - c + 2 * b) →
  (log_1_5 = 3 * a - b + c - 1) ∧ (log_7 = 2 * b + c) := sorry

end correct_log_values_l205_205995


namespace ratio_AXO_OYC_is_1_l205_205324

variables (DA AO OB BC CD OD AB DC : ℝ)
variables (P X Y A B C D O : Type)

noncomputable def midpoint (x y : ℝ) := (x + y) / 2

axiom rhombus : DA = AO ∧ AO = OB ∧ OB = BC ∧ BC = CD ∧ CD = OD ∧ AB = DC ∧ DA = 15 ∧ AB = 20
axiom point_mid_AC_eq_P (P : P) (A C : Type) : midpoint A C = P
axiom mid_da_eq_X (X : X) (DA : Type) : midpoint D A = X
axiom mid_bc_eq_Y (Y : Y) (BC : Type) : midpoint B C = Y

theorem ratio_AXO_OYC_is_1 (DA AO OB BC CD OD AB DC : ℝ) (P X Y A B C D O : Type) [H : rhombus]:
  (X ⊆ DA ∧ Y ⊆ BC ∧ AO = 15 ∧ PO = DA / 2 ∧ AX = 7.5 ∧ OX = 7.5) →
  AXO_OYC_ratio = 1 ∧ p + q = 2 :=
begin
  intros H1 H2 H3 H4 H5 H6,
  sorry,
end

end ratio_AXO_OYC_is_1_l205_205324


namespace least_pos_int_div_by_four_distinct_primes_l205_205163

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205163


namespace simplify_expression_l205_205921

theorem simplify_expression (x : ℝ) : (3 * x)^5 + (5 * x) * (x^4) - 7 * x^5 = 241 * x^5 := 
by
  sorry

end simplify_expression_l205_205921


namespace train_speed_l205_205686

theorem train_speed (train_length bridge_length : ℕ) (cross_time : ℕ) 
  (h_train_length : train_length = 170)
  (h_bridge_length : bridge_length = 205)
  (h_cross_time : cross_time = 30) :
  let total_distance := train_length + bridge_length in
  let speed_mps := total_distance / cross_time in
  let speed_kmph := speed_mps * 36 / 10 in
  speed_kmph = 45 :=
by 
  have h1 : total_distance = 375 := by 
    rw [h_train_length, h_bridge_length]
    rfl
  have h_speed_mps : speed_mps = 12.5 :=
    show speed_mps = 375 / 30 from by 
      rw [h1, h_cross_time]
      norm_num
  have h_speed_kmph : speed_kmph = 12.5 * 3.6 := by
    rw [h_speed_mps]
    norm_num
  exact h_speed_kmph

end train_speed_l205_205686


namespace units_digit_47_pow_47_l205_205605

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l205_205605


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205149

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205149


namespace frog_jumps_to_E_and_n_is_even_and_routes_count_l205_205218

-- Definitions based on the given conditions
def flower_pots : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7]
def start_pot : ℕ := 0
def end_pot : ℕ := 4
def adjacent_pots (n : ℕ) : List ℕ := [(n + 1) % 8, (n + 7) % 8]

-- Prove the statement.
theorem frog_jumps_to_E_and_n_is_even_and_routes_count (n m : ℕ)
    (h_n_even : n = 2 * m + 2) 
    (h_frog_jumps : n != 0) 
    (h_frog_ends_at_E : ∀ k, k < n → start_pot ≠ end_pot) :
    (n % 2 = 0) ∧ (∃ (m : ℕ), n = 2 * m + 2 ∧ 
        number_of_routes (start_pot ∈ flower_pots) (end_pot ∈ flower_pots) = 
        ((2 + Real.sqrt 2)^m - (2 - Real.sqrt 2)^m) / Real.sqrt 2) :=
by
  sorry

end frog_jumps_to_E_and_n_is_even_and_routes_count_l205_205218


namespace solve_for_y_l205_205622

variables (y : ℝ) (V : ℝ) (A : ℝ)

def cube_volume (y : ℝ) : ℝ := 3 * y
def cube_surface_area (y : ℝ) : ℝ := (3 * y^2) / 100

theorem solve_for_y (hV: V = cube_volume y) (hA: A = cube_surface_area y) : y = 600 :=
by
  sorry

end solve_for_y_l205_205622


namespace int_seq_proof_l205_205005

def is_integer_seq (a : ℤ) (a_seq : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, ∃ k : ℤ, a_seq n = k

theorem int_seq_proof (a : ℤ) (a_seq : ℕ → ℤ) 
  (h1 : a_seq 1 ∈ Int) (h2 : a_seq 2 ∈ Int)
  (h3 : (a_seq 1 * a_seq 2) ∣ (a_seq 1 ^ 2 + a_seq 2 ^ 2 + a)) 
  (h4 : ∀ n : ℕ, a_seq (n + 3) = (a_seq (n + 1) ^ 2 + a) / a_seq (n)) :
  is_integer_seq a a_seq :=
by
  sorry

end int_seq_proof_l205_205005


namespace min_value_fraction_l205_205327

variable (a b : ℝ)

theorem min_value_fraction (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 2 * b = 1) : 
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_fraction_l205_205327


namespace equation_of_hyperbola_point_P_on_fixed_line_l205_205376

-- Definition and conditions
def hyperbola_center_origin := (0, 0)
def hyperbola_left_focus := (-2 * Real.sqrt 5, 0)
def hyperbola_eccentricity := Real.sqrt 5
def point_on_line_through_origin := (-4, 0)
def point_M_in_second_quadrant (M : ℝ × ℝ) := M.1 < 0 ∧ M.2 > 0
def vertices_A1_A2 := ((-2, 0), (2, 0))

theorem equation_of_hyperbola (c a b : ℝ) (h_c : c = 2 * Real.sqrt 5)
  (h_e : hyperbola_eccentricity = Real.sqrt 5) (h_a : a = 2) (h_b : b = 4) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 16) = 1) := 
by
  intros x y h_x h_y
  have h : c^2 = a^2 + b^2 := sorry
  have h2 : hyperbola_eccentricity = c / a := sorry
  exact sorry

theorem point_P_on_fixed_line (M N P : ℝ × ℝ) (A1 A2 : ℝ × ℝ)
  (h_vertices : vertices_A1_A2 = (A1, A2))
  (h_M_in_2nd : point_M_in_second_quadrant M) 
  (h_P_intersects : ∀ t : ℝ, P = (t, (A1.2 * A2.1 - A1.1 * A2.2) / (A1.1 - A2.1))) :
  (P.1 = -1) := sorry

end equation_of_hyperbola_point_P_on_fixed_line_l205_205376


namespace hyperbola_center_origin_point_P_on_fixed_line_l205_205349

variables {x y : ℝ}

def hyperbola (h : (x^2 / 4) - (y^2 / 16) = 1) : Prop := true

theorem hyperbola_center_origin 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (left_focus : ∃ x y : ℝ, x = -2 * Real.sqrt 5 ∧ y = 0)
  (eccentricity_root5 : ∃ e : ℝ, e = Real.sqrt 5) :
  hyperbola ((x^2 / 4) - (y^2 / 16) = 1) := 
sorry

theorem point_P_on_fixed_line
  (line_through_point : ∃ x y : ℝ, x = -4 ∧ y = 0)
  (M_second_quadrant : ∃ Mx My : ℝ, Mx < 0 ∧ My > 0)
  (A1_vertex : ∃ x y : ℝ, x = -2 ∧ y = 0)
  (A2_vertex : ∃ x y : ℝ, x = 2 ∧ y = 0)
  (P_intersection : ∃ Px Py : ℝ, Px = -1) :
  Px = -1 :=
sorry

end hyperbola_center_origin_point_P_on_fixed_line_l205_205349


namespace problem_theorem_l205_205323
open Nat

def seq_a (a : ℕ → ℤ) (k : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n + k

def seq_b (a b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, b n = a (n + 1) - a n

def is_geometric (b : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = r * b n

def sum_seq (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in range n, seq i

def problem_statement (a b : ℕ → ℤ) (k : ℤ) : Prop :=
  seq_a a k ∧ seq_b a b ∧ b 1 ≠ 0 ∧
  is_geometric b 2 ∧
  let S_n := sum_seq a;
  let T_n := sum_seq b;
  S_n 6 ≠ T_n 4 ∧ S_n 5 = -9 ∧ k = 8

theorem problem_theorem (a b : ℕ → ℤ) (k : ℤ) : problem_statement a b k ↔
  (seq_a a k ∧ seq_b a b ∧ b 1 ≠ 0 → is_geometric b 2 ∧ S_n 6 ≠ T_n 4 ∧ S_n 5 = -9 ∧ k = 8) :=
sorry

end problem_theorem_l205_205323


namespace wheel_diameter_is_approx_18_l205_205688

noncomputable def wheel_diameter (revolutions : ℝ) (distance : ℝ) : ℝ :=
  let C := distance / revolutions
  C / Real.pi

theorem wheel_diameter_is_approx_18 :
  wheel_diameter 18.683651804670912 1056 ≈ 18 :=
by
  sorry

end wheel_diameter_is_approx_18_l205_205688


namespace sum_of_a_and_b_l205_205213

def otimes (x y : ℝ) : ℝ := x * (1 - y)

variable (a b : ℝ)

theorem sum_of_a_and_b :
  ({ x : ℝ | (x - a) * (1 - (x - b)) > 0 } = { x : ℝ | 2 < x ∧ x < 3 }) →
  a + b = 4 :=
by
  intro h
  have h_eq : ∀ x, (x - a) * ((1 : ℝ) - (x - b)) = (x - a) * (x - (b + 1)) := sorry
  have h_ineq : ∀ x, (x - a) * (x - (b + 1)) > 0 ↔ 2 < x ∧ x < 3 := sorry
  have h_set_eq : { x | (x - a) * ((1 : ℝ) - (x - b)) > 0 } = { x | 2 < x ∧ x < 3 } := sorry
  have h_roots_2_3 : (2 - a) * (2 - (b + 1)) = 0 ∧ (3 - a) * (3 - (b + 1)) = 0 := sorry
  have h_2_eq : 2 - a = 0 ∨ 2 - (b + 1) = 0 := sorry
  have h_3_eq : 3 - a = 0 ∨ 3 - (b + 1) = 0 := sorry
  have h_a_2 : a = 2 ∨ b + 1 = 2 := sorry
  have h_b_2 : b = 2 - 1 := sorry
  have h_a_3 : a = 3 ∨ b + 1 = 3 := sorry
  have h_b_3 : b = 3 - 1 := sorry
  sorry

end sum_of_a_and_b_l205_205213


namespace find_a_l205_205413

def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1| + |p.2| = a ∧ a > 0}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 * p.2| + 1 = |p.1| + |p.2|}

theorem find_a (a : ℝ) (hA : A a ∩ B = { (x, y) | True } ) :
  a = Real.sqrt 2 :=
sorry

end find_a_l205_205413


namespace astroid_length_theorem_l205_205277

noncomputable def astroid_length (a : ℝ) : ℝ :=
  6 * a

theorem astroid_length_theorem (a : ℝ) :
  ∃ (l : ℝ), l = astroid_length a ∧ l = 6 * a :=
by
  use astroid_length a
  split
  · rfl
  · rfl

end astroid_length_theorem_l205_205277


namespace percentage_increase_l205_205647

theorem percentage_increase (original new : ℝ) (h₁ : original = 50) (h₂ : new = 80) :
  ((new - original) / original) * 100 = 60 :=
by
  sorry

end percentage_increase_l205_205647


namespace parabola_equation_slope_condition_l205_205410

-- Definitions of the given conditions
def parabola (p : ℝ) (x y : ℝ) := x^2 = 2 * p * y
def line (k : ℝ) (x y : ℝ) := y = k * x + 2
def dot_product (A B : ℝ × ℝ) := A.1 * B.1 + A.2 * B.2

-- Given conditions
variables (p k : ℝ) (p_pos : p > 0)
variables (O A B : ℝ × ℝ)
variables (yA yB : ℝ)

-- Points A and B are intersections of the parabola and the line
axiom A_on_parabola : parabola p A.1 A.2 
axiom A_on_line : line k A.1 A.2
axiom B_on_parabola : parabola p B.1 B.2
axiom B_on_line : line k B.1 B.2

-- Dot product condition
axiom dot_product_condition : dot_product O A • dot_product O B = 2

-- Part (1) of the problem
theorem parabola_equation : 
  p = 1 / 2 → parabola (1 / 2) A.1 A.2 → parabola (1 / 2) B.1 B.2 → 
  ∃ y, A = (sqrt y, y) ∧ B = (-sqrt y, y) := 
  by sorry

-- Definitions of slopes k1 and k2
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

variables (C A B : ℝ × ℝ) -- C is the point (0, -2)
def C := (0, -2)
def k1 := slope C A
def k2 := slope C B

-- Part (2) of the problem
theorem slope_condition :
  p = 1 / 2 ∧ A_on_parabola ∧ A_on_line ∧ B_on_parabola ∧ B_on_line ∧ dot_product_condition →
  k1^2 + k2^2 - 2 * k^2 = 16 := 
  by sorry

end parabola_equation_slope_condition_l205_205410


namespace remainder_of_concatenated_natural_digits_l205_205984

theorem remainder_of_concatenated_natural_digits : 
  let digits := concat_nat_digits 198 in
  digits % 9 = 6 := 
by 
  sorry

end remainder_of_concatenated_natural_digits_l205_205984


namespace Xiaoming_excellent_score_l205_205848

namespace EnvironmentalProtectionCompetition

theorem Xiaoming_excellent_score (x : ℕ) (h_total_questions : 20 + x ≤ 20) : 5 * x - (20 - x) * 1 ≥ 85 → x ≥ 18 := 
by
  intro h_score
  have h_simplified : 6 * x ≥ 105 := by 
    linarith [h_score]

  have h_integer_part : x ≥ 18 := by
    linarith [h_simplified]

  exact h_integer_part
end


end Xiaoming_excellent_score_l205_205848


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205178

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205178


namespace frequency_count_of_fourth_group_l205_205692

noncomputable def frequency := ℝ
noncomputable def count := ℝ
noncomputable def total_count := ℝ

variables (N : total_count) (f1 f2 f3 : frequency) (n1 : count)

-- Conditions
def f1_is_0_1 : f1 = 0.1 := sorry
def f2_is_0_3 : f2 = 0.3 := sorry
def f3_is_0_4 : f3 = 0.4 := sorry
def n1_is_5 : n1 = 5 := sorry
def total_frequency_is_one : f1 + f2 + f3 + (1 - (f1 + f2 + f3)) = 1 := sorry
def f1_eq_n1_div_N : f1 = n1 / N := sorry

-- Theorem to prove
theorem frequency_count_of_fourth_group : (1 - (f1 + f2 + f3)) * N = 10 :=
by
  -- Using the provided conditions
  rw [f1_is_0_1, f2_is_0_3, f3_is_0_4, n1_is_5, f1_eq_n1_div_N]
  -- Simplify to show that the frequency count of the fourth group is 10
  sorry

end frequency_count_of_fourth_group_l205_205692


namespace total_cans_in_display_l205_205057

-- Defining the arithmetic sequence conditions
def arith_seq (a d : ℤ) : ℕ → ℤ
| 0       => a
| (n + 1) => arith_seq n + d

def num_terms (a d an : ℤ) : ℕ :=
  (a - an + d) / d

noncomputable def sum_arith_seq (a d an : ℤ) : ℤ :=
  let n := num_terms a d an
  n * (a + an) / 2

-- Given conditions
def bottom_layer_cans : ℤ := 35
def common_difference : ℤ := -4
def top_layer_cans : ℤ := 1

-- Theorem statement
theorem total_cans_in_display : sum_arith_seq bottom_layer_cans common_difference top_layer_cans + 1 = 172 := sorry

end total_cans_in_display_l205_205057


namespace bricks_in_wall_l205_205274

theorem bricks_in_wall (x : ℕ) : 
  let ben_rate := x / 12
  let jerry_rate := x / 8
  let combined_rate := ben_rate + jerry_rate
  let decreased_rate := combined_rate - 15
  let total_time := 6
  (decreased_rate * total_time = x) → x = 240 :=
by 
  intros
  let rate_common_denominator := (2 * x / 24 + 3 * x / 24)
  have h1: ben_rate = x / 12 := rfl
  have h2: jerry_rate = x / 8 := rfl
  have h3: combined_rate = rate_common_denominator := by
    rw [h1, h2]
    sorry
  have h4: decreased_rate = combined_rate - 15 := rfl
  have h5: x = decreased_rate * 6 := by
    rw [h4]
    sorry
  have final_eqn: 24 * x = 30 * x - 1440 := by
    rw [h5, h3]
    sorry
  have solved_x: x = 240 := by
    linarith
  exact solved_x
  sorry

end bricks_in_wall_l205_205274


namespace mod_inverse_28_mod_29_l205_205748

-- Define the question and conditions as Lean definitions
def mod_inverse (a m x : ℕ) : Prop := (a * x) % m = 1

theorem mod_inverse_28_mod_29 :
  ∃ (x : ℕ), x ≤ 28 ∧ mod_inverse 28 29 x :=
begin
  existsi 28,
  split,
  { norm_num }, -- checks that 28 ≤ 28
  { unfold mod_inverse, norm_num }, -- checks that 28 * 28 % 29 = 1
end

end mod_inverse_28_mod_29_l205_205748


namespace units_digit_47_pow_47_l205_205603

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l205_205603


namespace solution_set_lg2_l205_205784

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_deriv_lt : ∀ x : ℝ, deriv f x < 1

theorem solution_set_lg2 : { x : ℝ | f (Real.log x ^ 2) < Real.log x ^ 2 } = { x : ℝ | (1/10 : ℝ) < x ∧ x < 10 } :=
by
  sorry

end solution_set_lg2_l205_205784


namespace rabbit_jumps_before_dog_catches_l205_205231

/-- Prove that the number of additional jumps the rabbit can make before the dog catches up is 700,
    given the initial conditions:
      1. The rabbit has a 50-jump head start.
      2. The dog makes 5 jumps in the time the rabbit makes 6 jumps.
      3. The distance covered by 7 jumps of the dog equals the distance covered by 9 jumps of the rabbit. -/
theorem rabbit_jumps_before_dog_catches (h_head_start : ℕ) (h_time_ratio : ℚ) (h_distance_ratio : ℚ) : 
    h_head_start = 50 → h_time_ratio = 5/6 → h_distance_ratio = 7/9 → 
    ∃ (rabbit_additional_jumps : ℕ), rabbit_additional_jumps = 700 :=
by
  intro h_head_start_intro h_time_ratio_intro h_distance_ratio_intro
  have rabbit_additional_jumps := 700
  use rabbit_additional_jumps
  sorry

end rabbit_jumps_before_dog_catches_l205_205231


namespace radius_of_inscribed_semicircle_of_DEF_is_3_l205_205444

-- Define the right angled triangle DEF with DE = 15, EF = 8, and E as the right angle
def triangle_DEF : Type :=
  { DEF : Type // sorry }

-- Define that DE = 15 in this triangle
def DE : ℕ := 15

-- Define that EF = 8 in this triangle
def EF : ℕ := 8

-- Define that angle E is a right angle
def right_angle_E : Prop := sorry

-- Define the inradius of the right triangle DEF
def inradius (DEF : triangle_DEF) : ℝ := sorry

-- State the proof problem
theorem radius_of_inscribed_semicircle_of_DEF_is_3 (DEF : triangle_DEF) (DE_EQ_15 : DE = 15) (EF_EQ_8 : EF = 8) (RIGHT_ANGLE_E : right_angle_E) : 
    inradius DEF = 3 := sorry

end radius_of_inscribed_semicircle_of_DEF_is_3_l205_205444


namespace find_b8_l205_205560

noncomputable section

def increasing_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

axiom b_seq : ℕ → ℕ

axiom seq_inc : increasing_sequence b_seq

axiom b7_eq : b_seq 7 = 198

theorem find_b8 : b_seq 8 = 321 := by
  sorry

end find_b8_l205_205560


namespace find_number_l205_205756

theorem find_number (x : ℤ) (h : (85 + x) * 1 = 9637) : x = 9552 :=
by
  sorry

end find_number_l205_205756


namespace ratio_tin_copper_in_b_l205_205216

variable (L_a T_a T_b C_b : ℝ)

-- Conditions
axiom h1 : 170 + 250 = 420
axiom h2 : L_a / T_a = 1 / 3
axiom h3 : T_a + T_b = 221.25
axiom h4 : T_a + L_a = 170
axiom h5 : T_b + C_b = 250

-- Target
theorem ratio_tin_copper_in_b (h1 : 170 + 250 = 420) (h2 : L_a / T_a = 1 / 3)
  (h3 : T_a + T_b = 221.25) (h4 : T_a + L_a = 170) (h5 : T_b + C_b = 250) :
  T_b / C_b = 3 / 5 := by
  sorry

end ratio_tin_copper_in_b_l205_205216


namespace exists_bisecting_chord_l205_205334

-- Define a convex shape
structure ConvexShape (α : Type*) :=
  (is_convex : Prop)

-- Define a point A inside a convex shape
structure PointInConvexShape (α : Type*) :=
  (shape : ConvexShape α)
  (A : α)
  (A_inside_shape : A ∈ shape)

-- Define the property of having a chord through a point that is bisected by the point
def chord_bisected_by_point {α : Type*} (shape : ConvexShape α) (A : α) : Prop :=
  ∃ chord : set α, (∀ p ∈ chord, p ∈ shape) ∧ (∃ B C : α, B ≠ C ∧ (B ∈ chord ∨ C ∈ chord) ∧ (A = midpoint B C))

-- The main theorem statement
theorem exists_bisecting_chord {α : Type*} (shape : ConvexShape α) (A : α) (h : PointInConvexShape α) :
  chord_bisected_by_point shape A :=
sorry

end exists_bisecting_chord_l205_205334


namespace length_of_DE_l205_205543

variable (A B C X Y Z D E : Type)
variable [LinearOrderedField ℝ]

def base_length_ABC : ℝ := 15
def triangle_area_ratio : ℝ := 0.25

theorem length_of_DE (h1 : DE // BC ∥ BC) 
                    (h2 : triangle_area_ratio * (base_length_ABC ^ 2) = DE ^ 2)
                    : DE = 7.5 :=
sorry

end length_of_DE_l205_205543


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205089

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205089


namespace units_digit_47_pow_47_l205_205607

theorem units_digit_47_pow_47 :
  let cycle := [7, 9, 3, 1] in
  cycle.nth (47 % 4) = some 3 :=
by
  let cycle := [7, 9, 3, 1]
  have h : 47 % 4 = 3 := by norm_num
  rw h
  simp
  exact trivial

end units_digit_47_pow_47_l205_205607


namespace binomial_probability_4_l205_205766

noncomputable def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ := 
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem binomial_probability_4 (n : ℕ) (p : ℝ) (ξ : ℕ → ℝ)
  (H1 : (ξ 0) = (n*p))
  (H2 : (ξ 1) = (n*p*(1-p))) :
  binomial_pmf n 4 p = 10 / 243 :=
by {
  sorry 
}

end binomial_probability_4_l205_205766


namespace number_of_correct_sets_for_N_l205_205894

def setsNCondition (M : Set ℕ) (N : Set ℕ) : Prop :=
  M = {1, 2} ∧ M ∪ N = {1, 2, 3, 4}

theorem number_of_correct_sets_for_N : 
  (finset_univ (Set ℕ)).filter (λ N, setsNCondition {1, 2} N).card = 4 := by
  sorry

end number_of_correct_sets_for_N_l205_205894


namespace equation_of_hyperbola_point_P_fixed_line_l205_205365

-- Definitions based on conditions
def center_origin := (0, 0)
def left_focus := (-2 * real.sqrt 5, 0)
def eccentricity := real.sqrt 5
def left_vertex := (-2, 0)
def right_vertex := (2, 0)
def point_passed_through_line := (-4, 0)

-- The equation of the hyperbola that satisfies the given conditions.
theorem equation_of_hyperbola :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ 
    (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ 
      (2 * real.sqrt 5)^2 = a^2 + b^2) :=
by
  use [2, 4]
  split
  exact rfl
  split
  exact rfl
  sorry

-- Proving P lies on the fixed line x = -1 for given lines and vertices conditions
theorem point_P_fixed_line :
  ∀ m : ℝ, ∀ y1 y2 x1 x2 : ℝ, 
    x1 = m * y1 - 4 ∧ x2 = m * y2 - 4 ∧ 
    (y1 + y2 = (32 * m) / ((4 * m^2) - 1)) ∧
    (y1 * y2 = 48 / ((4 * m^2) - 1)) ∧ 
    (x1 + 2) ≠ 0 ∧ 
    (x2 - 2) ≠ 0 → 
    (P : ℝ × ℝ) ((P.1, P.2) = (-1, P.2) ∧ P.2 = (y1 * (x + 2) / x1 ≠ -2 ∧ y2 * (x - 2) / x2 ≠ 2)) := 
by
  intros
  sorry

end equation_of_hyperbola_point_P_fixed_line_l205_205365


namespace equation_of_hyperbola_point_P_on_fixed_line_l205_205378

-- Definition and conditions
def hyperbola_center_origin := (0, 0)
def hyperbola_left_focus := (-2 * Real.sqrt 5, 0)
def hyperbola_eccentricity := Real.sqrt 5
def point_on_line_through_origin := (-4, 0)
def point_M_in_second_quadrant (M : ℝ × ℝ) := M.1 < 0 ∧ M.2 > 0
def vertices_A1_A2 := ((-2, 0), (2, 0))

theorem equation_of_hyperbola (c a b : ℝ) (h_c : c = 2 * Real.sqrt 5)
  (h_e : hyperbola_eccentricity = Real.sqrt 5) (h_a : a = 2) (h_b : b = 4) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 16) = 1) := 
by
  intros x y h_x h_y
  have h : c^2 = a^2 + b^2 := sorry
  have h2 : hyperbola_eccentricity = c / a := sorry
  exact sorry

theorem point_P_on_fixed_line (M N P : ℝ × ℝ) (A1 A2 : ℝ × ℝ)
  (h_vertices : vertices_A1_A2 = (A1, A2))
  (h_M_in_2nd : point_M_in_second_quadrant M) 
  (h_P_intersects : ∀ t : ℝ, P = (t, (A1.2 * A2.1 - A1.1 * A2.2) / (A1.1 - A2.1))) :
  (P.1 = -1) := sorry

end equation_of_hyperbola_point_P_on_fixed_line_l205_205378


namespace sum_of_ages_l205_205472

variables (K T1 T2 : ℕ)

theorem sum_of_ages (h1 : K * T1 * T2 = 72) (h2 : T1 = T2) (h3 : T1 < K) : K + T1 + T2 = 14 :=
sorry

end sum_of_ages_l205_205472


namespace terry_wrong_problems_l205_205023

-- Definitions based on conditions
variable (R W : ℕ) -- number of right and wrong problems
variable (h1 : R + W = 25) -- condition 1: total problems
variable (h2 : 4 * R - W = 85) -- condition 2 and 3: scoring computation

-- Main theorem statement
theorem terry_wrong_problems : W = 3 :=
begin
  -- Proof will be filled in here
  sorry
end

end terry_wrong_problems_l205_205023


namespace sakshi_days_l205_205914

-- Define efficiency factor and days taken by Tanya
def efficiency_factor : ℝ := 1.25
def days_taken_by_tanya : ℝ := 16

-- Main theorem to prove Sakshi's days taken
theorem sakshi_days :
  let days_taken_by_sakshi := efficiency_factor * days_taken_by_tanya
  in days_taken_by_sakshi = 20 := 
by 
  -- Calculations that would be done within the proof
  sorry

end sakshi_days_l205_205914


namespace sum_diagonal_equals_fibonacci_l205_205908

noncomputable def binom (n k : ℕ) : ℕ :=
if k ≤ n then Nat.choose n k else 0

noncomputable def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

noncomputable def sum_diagonal (n : ℕ) : ℕ :=
(n+1).to_list.foldl (λ acc k => acc + binom (n-k) k) 0

theorem sum_diagonal_equals_fibonacci (n : ℕ) : sum_diagonal n = fib (n + 1) :=
sorry

end sum_diagonal_equals_fibonacci_l205_205908


namespace Sophie_l205_205531

-- Define the prices of each item
def price_cupcake : ℕ := 2
def price_doughnut : ℕ := 1
def price_apple_pie : ℕ := 2
def price_cookie : ℚ := 0.60

-- Define the quantities of each item
def qty_cupcake : ℕ := 5
def qty_doughnut : ℕ := 6
def qty_apple_pie : ℕ := 4
def qty_cookie : ℕ := 15

-- Define the total cost function for each item
def cost_cupcake := qty_cupcake * price_cupcake
def cost_doughnut := qty_doughnut * price_doughnut
def cost_apple_pie := qty_apple_pie * price_apple_pie
def cost_cookie := qty_cookie * price_cookie

-- Define total expenditure
def total_expenditure := cost_cupcake + cost_doughnut + cost_apple_pie + cost_cookie

-- Assertion of total expenditure
theorem Sophie's_total_expenditure : total_expenditure = 33 := by
  -- skipping proof
  sorry

end Sophie_l205_205531


namespace simplify_frac_op_l205_205832

-- Definition of the operation *
def frac_op (a b c d : ℚ) : ℚ := (a * c) * (d / (b + 1))

-- Proof problem stating the specific operation result
theorem simplify_frac_op :
  frac_op 5 11 9 4 = 15 :=
by
  sorry

end simplify_frac_op_l205_205832


namespace total_number_of_bricks_l205_205238

/-- Given bricks of volume 80 unit cubes and 42 unit cubes,
 and a box of volume 1540 unit cubes,
 prove the total number of bricks that can fill the box exactly is 24. -/
theorem total_number_of_bricks (x y : ℕ) (vol_a vol_b total_vol : ℕ)
  (vol_a_def : vol_a = 80)
  (vol_b_def : vol_b = 42)
  (total_vol_def : total_vol = 1540)
  (volume_filled : x * vol_a + y * vol_b = total_vol) :
  x + y = 24 :=
  sorry

end total_number_of_bricks_l205_205238


namespace fraction_female_attendees_on_time_l205_205703

theorem fraction_female_attendees_on_time (A : ℝ) (h1 : A > 0) :
  let males_fraction := 3/5
  let males_on_time := 7/8
  let not_on_time := 0.155
  let total_on_time_fraction := 1 - not_on_time
  let males := males_fraction * A
  let males_arrived_on_time := males_on_time * males
  let females := (1 - males_fraction) * A
  let females_arrived_on_time_fraction := (total_on_time_fraction * A - males_arrived_on_time) / females
  females_arrived_on_time_fraction = 4/5 :=
by
  sorry

end fraction_female_attendees_on_time_l205_205703


namespace equation_of_hyperbola_point_P_fixed_line_l205_205369

-- Definitions based on conditions
def center_origin := (0, 0)
def left_focus := (-2 * real.sqrt 5, 0)
def eccentricity := real.sqrt 5
def left_vertex := (-2, 0)
def right_vertex := (2, 0)
def point_passed_through_line := (-4, 0)

-- The equation of the hyperbola that satisfies the given conditions.
theorem equation_of_hyperbola :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ 
    (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ 
      (2 * real.sqrt 5)^2 = a^2 + b^2) :=
by
  use [2, 4]
  split
  exact rfl
  split
  exact rfl
  sorry

-- Proving P lies on the fixed line x = -1 for given lines and vertices conditions
theorem point_P_fixed_line :
  ∀ m : ℝ, ∀ y1 y2 x1 x2 : ℝ, 
    x1 = m * y1 - 4 ∧ x2 = m * y2 - 4 ∧ 
    (y1 + y2 = (32 * m) / ((4 * m^2) - 1)) ∧
    (y1 * y2 = 48 / ((4 * m^2) - 1)) ∧ 
    (x1 + 2) ≠ 0 ∧ 
    (x2 - 2) ≠ 0 → 
    (P : ℝ × ℝ) ((P.1, P.2) = (-1, P.2) ∧ P.2 = (y1 * (x + 2) / x1 ≠ -2 ∧ y2 * (x - 2) / x2 ≠ 2)) := 
by
  intros
  sorry

end equation_of_hyperbola_point_P_fixed_line_l205_205369


namespace base_conversion_l205_205031

theorem base_conversion (C D : ℕ) (hC : 0 ≤ C) (hC_lt : C < 8) (hD : 0 ≤ D) (hD_lt : D < 5) :
  (8 * C + D = 5 * D + C) → (8 * C + D = 0) :=
by 
  intro h
  sorry

end base_conversion_l205_205031


namespace area_quadrilateral_is_60_l205_205521

-- Definitions of the lengths of the quadrilateral sides and the ratio condition
def AB : ℝ := 8
def BC : ℝ := 5
def CD : ℝ := 17
def DA : ℝ := 10

-- Function representing the area of the quadrilateral ABCD
def area_ABCD (AB BC CD DA : ℝ) (ratio: ℝ) : ℝ :=
  -- Here we define the function to calculate the area, incorporating the given ratio
  sorry

-- The theorem to show that the area of quadrilateral ABCD is 60
theorem area_quadrilateral_is_60 : 
  area_ABCD AB BC CD DA (1/2) = 60 :=
by
  sorry

end area_quadrilateral_is_60_l205_205521


namespace collinear_points_b_value_l205_205727

theorem collinear_points_b_value (b : ℚ) :
  let P1 := (4, -6 : ℚ)
      P2 := (3 * b - 1, 5 : ℚ)
      P3 := (b + 4, 4 : ℚ)
  in collinear P1 P2 P3 → b = 50 / 19 :=
by
  -- Definitions of points
  let P1 := (4 : ℚ, -6 : ℚ)
  let P2 := (3 * b - 1 : ℚ, 5 : ℚ)
  let P3 := (b + 4 : ℚ, 4 : ℚ)
  
  -- Condition for collinearity in terms of slopes
  assume h_collinear : collinear P1 P2 P3

  -- Placeholder for the proof
  sorry

end collinear_points_b_value_l205_205727


namespace quadratic_inequality_solution_l205_205582

theorem quadratic_inequality_solution (a b : ℝ) (h1 : a ≠ 0) (h2 : -2 + 1/3 = -b / a) (h3 : -2 * (1/3) = -2 / a) : a + b = 8 :=
by
  have ha : a = 3 := sorry,
  have hb : b = 5 := sorry,
  rw [ha, hb]
  norm_num
  done

end quadratic_inequality_solution_l205_205582


namespace quadratic_no_real_roots_l205_205755

theorem quadratic_no_real_roots (m : ℝ) : (∀ x, x^2 - 2 * x + m ≠ 0) ↔ m > 1 := 
by sorry

end quadratic_no_real_roots_l205_205755


namespace farmer_tomatoes_l205_205661

theorem farmer_tomatoes (t p l : ℕ) (H1 : t = 97) (H2 : p = 83) : l = t - p → l = 14 :=
by {
  sorry
}

end farmer_tomatoes_l205_205661


namespace meteor_trail_is_a_line_l205_205224

-- We define a point moving through space.
def point_moves_to_form_a_line (P : Type) [point_space : add_comm_group P] [vector_space ℝ P] : Prop :=
  ∀ (p : P) (path : ℝ → P), (∃ t : ℝ, path t = p) →
    ∀ t₁ t₂ : ℝ, t₁ ≤ t₂ → ∃ q : set P, ∀ r : ℝ, t₁ ≤ r ∧ r ≤ t₂ → (path r ∈ q ∧ set.inj_on path q)

-- Prove that a point moving continuously forms a line.
theorem meteor_trail_is_a_line (P : Type) [point_space : add_comm_group P] [vector_space ℝ P]
  (move : ℝ → P) : point_moves_to_form_a_line P :=
by
  sorry

end meteor_trail_is_a_line_l205_205224


namespace math_proof_problem_l205_205344

noncomputable def question_1 (x y : ℝ) (a b : ℝ) : Prop :=
  (x = -1) ∧ (y = (real.sqrt 2) / 2) ∧
  (a^2 = 2) ∧ (b^2 = 1) ∧
  ((x^2 / (a^2)) + (y^2 / (b^2)) = 1)

noncomputable def question_2 (λ S : ℝ) : Prop :=
  (λ >= 2 / 3) ∧ (λ <= 3 / 4) ∧
  (S >= (real.sqrt 6) / 4) ∧ (S <= 2 / 3)

theorem math_proof_problem : 
  ∀ x y a b λ S, 
  question_1 x y a b → question_2 λ S → 
  ((x^2 / 2 + y^2 = 1) ∧ (S >= (real.sqrt 6) / 4) ∧ (S <= 2 / 3)) :=
by sorry

end math_proof_problem_l205_205344


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205082

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205082


namespace sakshi_days_l205_205912

-- Define efficiency factor and days taken by Tanya
def efficiency_factor : ℝ := 1.25
def days_taken_by_tanya : ℝ := 16

-- Main theorem to prove Sakshi's days taken
theorem sakshi_days :
  let days_taken_by_sakshi := efficiency_factor * days_taken_by_tanya
  in days_taken_by_sakshi = 20 := 
by 
  -- Calculations that would be done within the proof
  sorry

end sakshi_days_l205_205912


namespace theta_in_fourth_quadrant_l205_205430

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (∃ k : ℤ, θ = 2 * π * k + 7 * π / 4 ∨ θ = 2 * π * k + π / 4) ∧ θ = 2 * π * k + 7 * π / 4 :=
sorry

end theta_in_fourth_quadrant_l205_205430


namespace find_radius_l205_205251

noncomputable def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Prop := sorry

theorem find_radius (C1 : ℝ × ℝ × ℝ) (r1 : ℝ) (C2 : ℝ × ℝ × ℝ) (r : ℝ) :
  C1 = (3, 5, 0) →
  r1 = 2 →
  C2 = (0, 5, -8) →
  (sphere ((3, 5, -8) : ℝ × ℝ × ℝ) (2 * Real.sqrt 17)) →
  r = Real.sqrt 59 :=
by
  intros h1 h2 h3 h4
  sorry

end find_radius_l205_205251


namespace inverse_function_f_l205_205561

def f (x : ℝ) : ℝ := 2^(x + 1)

def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

theorem inverse_function_f (x : ℝ) (h : x > 0) : f_inv (f x) = x := 
by
  sorry

end inverse_function_f_l205_205561


namespace expr_eq_l205_205713

noncomputable def expr : ℝ :=
  (1 / 4) ^ (-1 / 2) - real.sqrt ((3 - real.pi) ^ 2) + real.logb 10 5 ^ 2 + real.logb 10 2 * real.logb 10 50

theorem expr_eq : expr = 6 - real.pi :=
by sorry

end expr_eq_l205_205713


namespace value_of_a_l205_205202

theorem value_of_a (a b k : ℝ) (h1 : a = k / b^2) (h2 : a = 40) (h3 : b = 12) (h4 : b = 24) : a = 10 := 
by
  sorry

end value_of_a_l205_205202


namespace sufficient_but_not_necessary_l205_205880

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > |b|) → (a^3 > b^3) ∧ ¬((a^3 > b^3) → (a > |b|)) :=
by
  sorry

end sufficient_but_not_necessary_l205_205880


namespace least_pos_int_div_by_four_distinct_primes_l205_205105

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205105


namespace units_digit_47_pow_47_l205_205620

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l205_205620


namespace equation_one_solutions_equation_two_solutions_l205_205530

section equation_one
variable (x : ℝ)

theorem equation_one_solutions (x1 x2 : ℝ) :
  (x1 = 2 + Real.sqrt 7) ∧ (x2 = 2 - Real.sqrt 7) → (x1 * x1 - 4 * x1 - 3 = 0) ∧ (x2 * x2 - 4 * x2 - 3 = 0) := by
  intro h
  cases h with hx1 hx2
  split
  · rw [hx1]
    sorry
  · rw [hx2]
    sorry
end equation_one

section equation_two
variable (x : ℝ)

theorem equation_two_solutions (x1 x2 : ℝ) :
  (x1 = 4) ∧ (x2 = 4 / 3) → ((x1 + 1) * (x1 + 1) = (2 * x1 - 3) * (2 * x1 - 3)) ∧ ((x2 + 1) * (x2 + 1) = (2 * x2 - 3) * (2 * x2 - 3)) := by
  intro h
  cases h with hx1 hx2
  split
  · rw [hx1]
    sorry
  · rw [hx2]
    sorry
end equation_two

end equation_one_solutions_equation_two_solutions_l205_205530


namespace expression_zero_iff_x_eq_three_l205_205297

theorem expression_zero_iff_x_eq_three (x : ℝ) :
  (4 * x - 8 ≠ 0) → ((x^2 - 6 * x + 9 = 0) ↔ (x = 3)) :=
by
  sorry

end expression_zero_iff_x_eq_three_l205_205297


namespace coloring_integers_l205_205537

theorem coloring_integers 
  (color : ℤ → ℕ) 
  (x y : ℤ) 
  (hx : x % 2 = 1) 
  (hy : y % 2 = 1) 
  (h_neq : |x| ≠ |y|) 
  (h_color_range : ∀ n : ℤ, color n < 4) :
  ∃ a b : ℤ, color a = color b ∧ (a - b = x ∨ a - b = y ∨ a - b = x + y ∨ a - b = x - y) :=
sorry

end coloring_integers_l205_205537


namespace smallest_pos_int_div_by_four_primes_l205_205111

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205111


namespace greatest_possible_gcd_l205_205318

theorem greatest_possible_gcd {n : ℕ} (hn : n > 0) : ∃ k, k = 12 ∧ ∀ m, gcd (6 * (n*(n+1) / 2)) (n-2) ≤ k :=
by
  sorry

end greatest_possible_gcd_l205_205318


namespace intersecting_parabolas_circle_radius_sq_l205_205051

theorem intersecting_parabolas_circle_radius_sq:
  (∀ (x y : ℝ), (y = (x + 1)^2 ∧ x + 4 = (y - 3)^2) → 
  ((x + 1/2)^2 + (y - 7/2)^2 = 13/2)) := sorry

end intersecting_parabolas_circle_radius_sq_l205_205051


namespace smallest_pos_int_div_by_four_primes_l205_205112

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205112


namespace negation_of_proposition_l205_205566

theorem negation_of_proposition (x : ℝ) : 
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := 
sorry

end negation_of_proposition_l205_205566


namespace remainders_not_unique_l205_205190

theorem remainders_not_unique (n : ℕ) (hpos : 0 < n) :
  ∃ i j : ℕ, i ≠ j ∧ (1 + a i % 2*n) = ( j + a j % 2*n) :=
  sorry

end remainders_not_unique_l205_205190


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205132

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205132


namespace P_not_on_curve_Q_on_curve_final_result_l205_205399

def curve (θ : ℝ) : ℝ := 2 * Real.cos (2 * θ)

def P : ℝ × ℝ := (0, Real.pi / 2)

def Q : ℝ × ℝ := (2, Real.pi)

theorem P_not_on_curve : ¬(P.1 = curve P.2) :=
by
  -- Formal proof would go here
  sorry

theorem Q_on_curve : (Q.1 = curve Q.2) :=
by
  -- Formal proof would go here
  sorry

theorem final_result : (¬(P.1 = curve P.2)) ∧ (Q.1 = curve Q.2) :=
by
  exact ⟨P_not_on_curve, Q_on_curve⟩

end P_not_on_curve_Q_on_curve_final_result_l205_205399


namespace total_money_shared_l205_205861

theorem total_money_shared (ratio_jonah ratio_kira ratio_liam kira_share : ℕ)
  (h_ratio : ratio_jonah = 2) (h_ratio2 : ratio_kira = 3) (h_ratio3 : ratio_liam = 8)
  (h_kira : kira_share = 45) :
  (ratio_jonah * (kira_share / ratio_kira) + kira_share + ratio_liam * (kira_share / ratio_kira)) = 195 := 
by
  sorry

end total_money_shared_l205_205861


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205148

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205148


namespace length_of_DE_l205_205542

variable (A B C X Y Z D E : Type)
variable [LinearOrderedField ℝ]

def base_length_ABC : ℝ := 15
def triangle_area_ratio : ℝ := 0.25

theorem length_of_DE (h1 : DE // BC ∥ BC) 
                    (h2 : triangle_area_ratio * (base_length_ABC ^ 2) = DE ^ 2)
                    : DE = 7.5 :=
sorry

end length_of_DE_l205_205542


namespace yanna_sandals_l205_205192

theorem yanna_sandals (shirts_cost: ℕ) (sandal_cost: ℕ) (total_money: ℕ) (change: ℕ) (num_shirts: ℕ)
  (h1: shirts_cost = 5)
  (h2: sandal_cost = 3)
  (h3: total_money = 100)
  (h4: change = 41)
  (h5: num_shirts = 10) : 
  ∃ num_sandals: ℕ, num_sandals = 3 :=
sorry

end yanna_sandals_l205_205192


namespace internal_diagonal_crosses_820_cubes_l205_205657

theorem internal_diagonal_crosses_820_cubes :
  let l := 200
  let w := 330
  let h := 360
  let g1 := Nat.gcd l w
  let g2 := Nat.gcd w h
  let g3 := Nat.gcd h l
  let g := Nat.gcd(g1, Nat.gcd(g2, g3))
  (l + w + h - g1 - g2 - g3 + g) = 820 :=
by {
  let l := 200
  let w := 330
  let h := 360
  let g1 := Nat.gcd l w
  let g2 := Nat.gcd w h
  let g3 := Nat.gcd h l
  let g := Nat.gcd(g1, Nat.gcd(g2, g3));
  sorry
}

end internal_diagonal_crosses_820_cubes_l205_205657


namespace percentage_waiting_for_parts_l205_205860

def totalComputers : ℕ := 20
def unfixableComputers : ℕ := (20 * 20) / 100
def fixedRightAway : ℕ := 8
def waitingForParts : ℕ := totalComputers - (unfixableComputers + fixedRightAway)

theorem percentage_waiting_for_parts : (waitingForParts : ℝ) / totalComputers * 100 = 40 := 
by 
  have : waitingForParts = 8 := sorry
  have : (8 / 20 : ℝ) * 100 = 40 := sorry
  exact sorry

end percentage_waiting_for_parts_l205_205860


namespace find_m_for_zero_of_function_l205_205834

def function_zero (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

theorem find_m_for_zero_of_function : ∀ (m : ℝ), function_zero (λ x, m + (1 / 3) ^ x) (-2) → m = -9 := by
  intros m h
  sorry

end find_m_for_zero_of_function_l205_205834


namespace measure_angle_Q_l205_205920

theorem measure_angle_Q
  (A B C D E Q : Type)
  (h1 : ∀ A B C D E : Type, regular_pentagon A B C D E)
  (h2 : ∃ Q : Type, extended_sides_meet A B D E Q) :
  measure_angle Q = 36 :=
sorry

end measure_angle_Q_l205_205920


namespace total_students_l205_205568

-- Definitions
variables (b g : ℕ) 

-- Conditions
def condition1 := b = 7 * g
def condition2 := b = g + 900

-- Theorem to prove
theorem total_students (h1 : condition1) (h2 : condition2) : b + g = 1200 :=
  sorry

end total_students_l205_205568


namespace part1_solution_l205_205401

variable (a : ℝ) (m : ℝ)

def f (x : ℝ) : ℝ := 2 * Real.log x - x^2 + a * x

theorem part1_solution (h : ∃ x1 x2 : ℝ, (1 / Real.exp 1 < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ x2 < Real.exp 1) ∧ (f x1 - a * x1 + m = 0 ∧ f x2 - a * x2 + m = 0)) :
  1 < m ∧ m ≤ 2 + 1 / (Real.exp 2) := sorry

end part1_solution_l205_205401


namespace smallest_n_b_n_equals_b₀_l205_205825

noncomputable def b₀ := Real.cos (Real.pi / 18) ^ 2

def b (n : ℕ) : ℝ :=
  Nat.recOn n b₀ (λ _ b_n, 4 * b_n * (1 - b_n))

theorem smallest_n_b_n_equals_b₀ : ∃ n : ℕ, n > 0 ∧ b n = b₀ ∧ ∀ m : ℕ, m > 0 ∧ b m = b₀ → n ≤ m :=
begin
  use 8,
  split, sorry, -- n > 0
  split, sorry, -- b 8 = b₀
  intros m hm hbm,
  sorry -- 8 ≤ m
end

end smallest_n_b_n_equals_b₀_l205_205825


namespace g_ln_1_over_2017_l205_205783

theorem g_ln_1_over_2017 (a : ℝ) (h_a_pos : 0 < a) (h_a_neq_1 : a ≠ 1) (f g : ℝ → ℝ)
  (h_f_add : ∀ m n : ℝ, f (m + n) = f m + f n - 1)
  (h_g : ∀ x : ℝ, g x = f x + a^x / (a^x + 1))
  (h_g_ln_2017 : g (Real.log 2017) = 2018) :
  g (Real.log (1 / 2017)) = -2015 :=
sorry

end g_ln_1_over_2017_l205_205783


namespace wendy_walked_l205_205596

theorem wendy_walked (x : ℝ) (h1 : 19.83 = x + 10.67) : x = 9.16 :=
sorry

end wendy_walked_l205_205596


namespace suff_condition_not_necc_condition_l205_205891

variable (x : ℝ)

def A : Prop := 0 < x ∧ x < 5
def B : Prop := |x - 2| < 3

theorem suff_condition : A x → B x := by
  sorry

theorem not_necc_condition : B x → ¬ A x := by
  sorry

end suff_condition_not_necc_condition_l205_205891


namespace calculate_S5_l205_205758

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def sum_of_digits_n_times (f : ℕ → ℕ) (n times : ℕ) : ℕ :=
  (List.range times).foldl (λ acc _, f acc) n

lemma sum_of_digits_mod_9 (n : ℕ) : sum_of_digits n % 9 = n % 9 :=
sorry

theorem calculate_S5 (n : ℕ) (h : n = 2018 ^ 2018 ^ 2018) : sum_of_digits_n_times sum_of_digits n 5 = 7 :=
sorry

end calculate_S5_l205_205758


namespace apex_angle_first_two_cones_l205_205966

theorem apex_angle_first_two_cones (A : Point) (cone1 cone2 cone3 cone4 : Cone) (α β : Real)
  (h_cone1_eq_cone2 : cone1.apex_angle = cone2.apex_angle)
  (h_cone3_apex_angle : cone3.apex_angle = 2 * arcsin (1/4))
  (h_cone4_apex_angle_half : cone1.apex_angle = 2 * α)
  (h_cones_touching : cones_touch_each_other_externally [cone1, cone2, cone3]
      ∧ cones_touch_each_other_internally cone4 [cone1, cone2, cone3]) :
  cone1.apex_angle = (π / 6) + arcsin (1 / 4) := 
  sorry

end apex_angle_first_two_cones_l205_205966


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205160

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205160


namespace refill_cost_calculation_l205_205816

variables (total_spent : ℕ) (refills : ℕ)

def one_refill_cost (total_spent refills : ℕ) : ℕ := total_spent / refills

theorem refill_cost_calculation (h1 : total_spent = 40) (h2 : refills = 4) :
  one_refill_cost total_spent refills = 10 :=
by
  sorry

end refill_cost_calculation_l205_205816


namespace floor_abs_square_sum_l205_205279

/-- Floor function -/
def floor (x : ℝ) : ℤ := Int.floor x

/-- Absolute value function -/
def abs (x : ℝ) : ℝ := Real.abs x

/-- Specific value to be used in the functions -/
def value : ℝ := -7.3

/-- Proof statement -/
theorem floor_abs_square_sum : floor (abs value) + (floor value)^2 = 71 :=
by
  sorry

end floor_abs_square_sum_l205_205279


namespace num_four_digit_snappy_numbers_divisible_by_25_l205_205670

def is_snappy (n : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by_25 (n : ℕ) : Prop :=
  let last_two_digits := n % 100
  last_two_digits = 0 ∨ last_two_digits = 25 ∨ last_two_digits = 50 ∨ last_two_digits = 75

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem num_four_digit_snappy_numbers_divisible_by_25 : 
  ∃ n, n = 3 ∧ (∀ x, is_four_digit x ∧ is_snappy x ∧ is_divisible_by_25 x ↔ x = 5225 ∨ x = 0550 ∨ x = 5775)
:=
sorry

end num_four_digit_snappy_numbers_divisible_by_25_l205_205670


namespace original_price_l205_205225

theorem original_price (x : ℝ) (h1 : 0.75 * x + 12 = x - 12) (h2 : 0.90 * x - 42 = x - 12) : x = 360 :=
by
  sorry

end original_price_l205_205225


namespace logarithmic_identity_unique_solution_l205_205193

theorem logarithmic_identity_unique_solution (x : ℝ) (hx : 0 < x ∧ x ≠ 1) :
    log 2 3 + 2 * log 4 x = x ^ (log 9 16 / log 3 x) → x = 16 / 3 :=
by
  sorry

end logarithmic_identity_unique_solution_l205_205193


namespace avg_of_first_5_numbers_equal_99_l205_205539

def avg_of_first_5 (S1 : ℕ) : ℕ := S1 / 5

theorem avg_of_first_5_numbers_equal_99
  (avg_9 : ℕ := 104) (avg_last_5 : ℕ := 100) (fifth_num : ℕ := 59)
  (sum_9 := 9 * avg_9) (sum_last_5 := 5 * avg_last_5) :
  avg_of_first_5 (sum_9 - sum_last_5 + fifth_num) = 99 :=
by
  sorry

end avg_of_first_5_numbers_equal_99_l205_205539


namespace smallest_pos_int_div_by_four_primes_l205_205108

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205108


namespace function_crosses_asymptote_at_value_l205_205762

def g (x : ℝ) : ℝ := (3 * x^2 - 6 * x - 9) / (x^2 - 5 * x + 6)

theorem function_crosses_asymptote_at_value :
  g (9 / 5) = 3 := 
sorry

end function_crosses_asymptote_at_value_l205_205762


namespace equation_of_hyperbola_P_lies_on_fixed_line_l205_205360

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- The conditions provided in the problem
def focus := (-2 * Real.sqrt 5, 0)
def center := (0, 0)
def eccentricity : ℝ := Real.sqrt 5
def vertex1 := (-2, 0)
def vertex2 := (2, 0)
def passing_point := (-4, 0)

-- Finding the equation of the hyperbola
theorem equation_of_hyperbola : 
  ∀ (a b c : ℝ), 
  c = 2 * Real.sqrt 5 ∧ 
  eccentricity = Real.sqrt 5 ∧ 
  a = 2 ∧
  b = 4 → 
  hyperbola 2 4 :=
by
  sorry

-- Proving P lies on a fixed line
theorem P_lies_on_fixed_line :
  ∀ (m : ℝ), 
  let line : ℝ → ℝ := λ y, m * y - 4 in
  let intersection_M := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let intersection_N := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let P := (*) -- Intersection point of lines MA1 and NA2 -- (*) 
  ∃ (P : ℝ × ℝ), P.fst = -1 :=
by
  sorry

end equation_of_hyperbola_P_lies_on_fixed_line_l205_205360


namespace reusable_bag_trips_correct_lowest_carbon_solution_l205_205504

open Real

-- Conditions definitions
def canvas_CO2 := 600 -- in pounds
def polyester_CO2 := 250 -- in pounds
def recycled_plastic_CO2 := 150 -- in pounds
def CO2_per_plastic_bag := 4 / 16 -- 4 ounces per bag, converted to pounds
def bags_per_trip := 8

-- Total CO2 per trip using plastic bags
def CO2_per_trip := CO2_per_plastic_bag * bags_per_trip

-- Proof of correct number of trips
theorem reusable_bag_trips_correct :
  canvas_CO2 / CO2_per_trip = 300 ∧
  polyester_CO2 / CO2_per_trip = 125 ∧
  recycled_plastic_CO2 / CO2_per_trip = 75 :=
by
  -- Here we would provide proofs for each part,
  -- ensuring we are fulfilling the conditions provided
  -- Skipping the proof with sorry
  sorry

-- Proof that recycled plastic bag is the lowest-carbon solution
theorem lowest_carbon_solution :
  min (canvas_CO2 / CO2_per_trip) (min (polyester_CO2 / CO2_per_trip) (recycled_plastic_CO2 / CO2_per_trip)) = recycled_plastic_CO2 / CO2_per_trip :=
by
  -- Here we would provide proofs for each part,
  -- ensuring we are fulfilling the conditions provided
  -- Skipping the proof with sorry
  sorry

end reusable_bag_trips_correct_lowest_carbon_solution_l205_205504


namespace cos_C_in_triangle_l205_205460

theorem cos_C_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_sinA : Real.sin A = 4/5) (h_cosB : Real.cos B = 3/5) :
  Real.cos C = 7/25 := 
sorry

end cos_C_in_triangle_l205_205460


namespace perpendicular_lines_slope_l205_205396

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, x + a * y = 1 - a ∧ (a - 2) * x + 3 * y + 2 = 0) → a = 1 / 2 := 
by 
  sorry

end perpendicular_lines_slope_l205_205396


namespace distance_midway_orbit_l205_205900

theorem distance_midway_orbit (perigee apogee : ℝ) (focal_distance : ℝ) : 
  perigee = 3 ∧ apogee = 10 ∧ focal_distance = 6.5 → 
  (∃ d, d = 6.5) :=
by 
  intros h
  use 6.5
  cases h with h1 h2
  cases h2 with h3 h4
  simp [h1, h3, h4]
  trivial

end distance_midway_orbit_l205_205900


namespace hyperbola_and_fixed_line_proof_l205_205383

-- Definitions based on conditions
def center_at_origin (C : Type) [Hyperbola C] : Prop :=
  C.center = (0, 0)

def left_focus (C : Type) [Hyperbola C] : Prop :=
  C.focus1 = (-2 * Real.sqrt 5, 0)

def eccentricity (C : Type) [Hyperbola C] : Prop :=
  C.eccentricity = Real.sqrt 5

def vertices (C : Type) [Hyperbola C] : Prop :=
  C.vertex1 = (-2, 0) ∧ C.vertex2 = (2, 0)

def passing_line_intersections (C : Type) (M N : Type) [Hyperbola C] : Prop :=
  ∃ l : Line, l ⨯ (-4, 0) ∧ l.intersects_hyperbola_left_branch C = {M, N}

def intersection (M N A1 A2 P : Type) [Hyperbola C] : Prop :=
  ∃ A1 A2, A1 = (-2, 0) ∧ A2 = (2, 0) ∧ (MA1.M ∩ NA2.N = P)

-- Proof problem statement
theorem hyperbola_and_fixed_line_proof (C : Type) [Hyperbola C] (M N P : Type)
  (h1 : center_at_origin C) (h2 : left_focus C) (h3 : eccentricity C)
  (h4 : passing_line_intersections C M N) (h5 : vertices C) (h6 : intersection M N C.vertex1 C.vertex2 P) : 
  (C.equation = (∀ x y, C.vertical_transverse_axis x y (x^2 / 4 - y^2 / 16)) ∧ 
   ∀ (x : ℝ), P.x = -1) :=
by sorry

end hyperbola_and_fixed_line_proof_l205_205383


namespace collinear_X_O_Y_l205_205910

/-- Quadrilateral ABCD is inscribed in circle Γ with center O.
    I is the incenter of triangle ABC, and J is the incenter of triangle ABD.
    Line IJ intersects segments AD, AC, BD, BC at points P, M, N, and Q respectively.
    The perpendicular from M to line AC intersects the perpendicular from N to line BD at point X.
    The perpendicular from P  to line AD intersects the perpendicular from Q to line BC at point Y.
    Prove that X, O, Y are collinear. -/
theorem collinear_X_O_Y
  (A B C D O I J P M N Q X Y: Point)
  (Γ : Circle)
  (h1: InscribedInCircle ABCD Γ)
  (h2: CenterOfCircle Γ O)
  (h3: IncenterOfTriangle I ABC)
  (h4: IncenterOfTriangle J ABD)
  (h5: IntersectsLine IJ SegmentAD P)
  (h6: IntersectsLine IJ SegmentAC M)
  (h7: IntersectsLine IJ SegmentBD N)
  (h8: IntersectsLine IJ SegmentBC Q)
  (h9: PerpendicularFromTo M AC X)
  (h10: PerpendicularFromTo N BD X)
  (h11: PerpendicularFromTo P AD Y)
  (h12: PerpendicularFromTo Q BC Y) :
  Collinear X O Y :=
sorry

end collinear_X_O_Y_l205_205910


namespace eldest_child_age_l205_205194

variable (y m e : ℕ)

-- conditions
def condition1 : Prop := m = y + 3
def condition2 : Prop := e = 3 * y
def condition3 : Prop := e = (y + m) + 2

-- theorem to prove
theorem eldest_child_age 
  (h1 : condition1 y m e)
  (h2 : condition2 y m e)
  (h3 : condition3 y m e) : 
  e = 15 :=
sorry

end eldest_child_age_l205_205194


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205176

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205176


namespace larger_integer_value_l205_205952

-- Define the conditions as Lean definitions
def quotient_condition (a b : ℕ) : Prop := a / b = 5 / 2
def product_condition (a b : ℕ) : Prop := a * b = 160
def larger_integer (a b : ℕ) : ℕ := if a > b then a else b

-- State the theorem with conditions and expected outcome
theorem larger_integer_value (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) :
  larger_integer a b = 20 :=
sorry -- Proof to be provided

end larger_integer_value_l205_205952


namespace solve_system_of_equations_l205_205019

theorem solve_system_of_equations : ∃ (x y : ℤ), 
  2 * x - 3 * y = -7 ∧ 5 * x + 4 * y = -6 ∧ x = -2 ∧ y = 1 :=
by
  use -2, 1
  split
  { sorry }
  split
  { sorry }
  split
  { rfl }
  { rfl }

end solve_system_of_equations_l205_205019


namespace least_positive_integer_divisible_by_four_primes_l205_205126

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205126


namespace smallest_pos_int_div_by_four_primes_l205_205110

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205110


namespace minimal_possible_neighbor_sums_l205_205994

theorem minimal_possible_neighbor_sums :
  ∀ (arrangement : list ℕ), 
    arrangement.perm (list.range 10) ∧ list.chain' (≠) arrangement →
    ∃ (neighbor_sums : set ℕ), neighbor_sums.card = 3 :=
by
  sorry

end minimal_possible_neighbor_sums_l205_205994


namespace exponentiation_comparison_l205_205981

theorem exponentiation_comparison :
  1.7 ^ 0.3 > 0.9 ^ 0.3 :=
by sorry

end exponentiation_comparison_l205_205981


namespace least_pos_int_div_by_four_distinct_primes_l205_205101

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205101


namespace find_a_l205_205807

def A : Set ℝ := {2, 3}
def B (a : ℝ) : Set ℝ := {1, 2, a}

theorem find_a (a : ℝ) : A ⊆ B a → a = 3 :=
by
  intro h
  sorry

end find_a_l205_205807


namespace completing_the_square_l205_205629

theorem completing_the_square (x : ℝ) : x^2 + 2 * x - 5 = 0 → (x + 1)^2 = 6 := by
  intro h
  -- Starting from h and following the steps outlined to complete the square.
  sorry

end completing_the_square_l205_205629


namespace integer_part_S_l205_205877

noncomputable def S : Real :=
  1 + (Finset.range 1000000).sum (λ n, 1 / Real.sqrt (n+1))

theorem integer_part_S : Int.floor S = 1998 :=
sorry

end integer_part_S_l205_205877


namespace smallest_k_l205_205930

-- Define the conditions
def is_lattice_polygon (F : Polygon) : Prop :=
  ∀ (v : Vertex), v ∈ F.vertices → lattice_vertex v

def sides_parallel (F : Polygon) : Prop :=
  ∀ (e : Edge), e ∈ F.edges → (parallel_to_x_axis e ∨ parallel_to_y_axis e)

def S (F : Polygon) : ℝ := area F
def P (F : Polygon) : ℝ := perimeter F

-- Main statement
theorem smallest_k (F : Polygon)
    (h1 : is_lattice_polygon F)
    (h2 : sides_parallel F) :
    ∃ k : ℝ, (∀ (F : Polygon), S F ≤ k * (P F) ^ 2) ∧ k = 1 / 16 := 
sorry

end smallest_k_l205_205930


namespace min_value_fraction_l205_205491

theorem min_value_fraction (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (h_sum : a + 3 * b = 1) :
  (∀ x y : ℝ, (0 < x) → (0 < y) → x + 3 * y = 1 → 16 ≤ 1 / x + 3 / y) :=
sorry

end min_value_fraction_l205_205491


namespace probability_between_9_l205_205659

noncomputable def weight_distribution : ℝ → ℝ :=
  pdf (Normal 10 0.1)

theorem probability_between_9.8_and_10.2 :
  ∫ x in 9.8..10.2, weight_distribution x = 0.9544 := 
sorry

end probability_between_9_l205_205659


namespace cumulative_decrease_correct_l205_205865

variables (S : ℝ)

-- Define the pay cuts as fractions of the original salary
def first_cut := S * 0.92
def second_cut := first_cut * 0.86
def third_cut := second_cut * 0.82
def fourth_cut := third_cut * 0.78
def fifth_cut := fourth_cut * 0.73

-- Define the total cumulative percentage decrease
noncomputable def cumulative_decrease : ℝ := 100 * (1 - fifth_cut / S)

-- Prove that the cumulative percentage decrease is approximately 56.07%
theorem cumulative_decrease_correct : abs (cumulative_decrease S - 56.07) < 0.01 :=
by sorry

end cumulative_decrease_correct_l205_205865


namespace height_of_model_tower_l205_205940

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem height_of_model_tower (h_city : ℝ) (v_city : ℝ) (v_model : ℝ) (h_model : ℝ) :
  h_city = 80 → v_city = 200000 → v_model = 0.05 →
  let volume_ratio := v_city / v_model in
  let scale_factor := volume_ratio^(1/3) in
  h_model = h_city / scale_factor →
  h_model = 0.5 :=
by
  intros h_city_eq v_city_eq v_model_eq volume_ratio scale_factor h_model_eq
  sorry 

end height_of_model_tower_l205_205940


namespace sum_of_three_numbers_is_71_point_5_l205_205835

noncomputable def sum_of_three_numbers (a b c : ℝ) : ℝ :=
a + b + c

theorem sum_of_three_numbers_is_71_point_5 (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 48) (h3 : c + a = 60) :
  sum_of_three_numbers a b c = 71.5 :=
by
  unfold sum_of_three_numbers
  sorry

end sum_of_three_numbers_is_71_point_5_l205_205835


namespace outfit_combinations_l205_205535

theorem outfit_combinations (shirts ties belts : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 6) (h_belts : belts = 4) :
  shirts * ties * belts = 192 :=
by
  rw [h_shirts, h_ties, h_belts]
  norm_num

end outfit_combinations_l205_205535


namespace selection_probability_correct_l205_205842

def percentage_women : ℝ := 0.55
def percentage_men : ℝ := 0.45

def women_below_35 : ℝ := 0.20
def women_35_to_50 : ℝ := 0.35
def women_above_50 : ℝ := 0.45

def men_below_35 : ℝ := 0.30
def men_35_to_50 : ℝ := 0.40
def men_above_50 : ℝ := 0.30

def women_below_35_lawyers : ℝ := 0.35
def women_below_35_doctors : ℝ := 0.45
def women_below_35_engineers : ℝ := 0.20

def women_35_to_50_lawyers : ℝ := 0.25
def women_35_to_50_doctors : ℝ := 0.50
def women_35_to_50_engineers : ℝ := 0.25

def women_above_50_lawyers : ℝ := 0.20
def women_above_50_doctors : ℝ := 0.30
def women_above_50_engineers : ℝ := 0.50

def men_below_35_lawyers : ℝ := 0.40
def men_below_35_doctors : ℝ := 0.30
def men_below_35_engineers : ℝ := 0.30

def men_35_to_50_lawyers : ℝ := 0.45
def men_35_to_50_doctors : ℝ := 0.25
def men_35_to_50_engineers : ℝ := 0.30

def men_above_50_lawyers : ℝ := 0.30
def men_above_50_doctors : ℝ := 0.40
def men_above_50_engineers : ℝ := 0.30

theorem selection_probability_correct :
  (percentage_women * women_below_35 * women_below_35_lawyers +
   percentage_men * men_above_50 * men_above_50_engineers +
   percentage_women * women_35_to_50 * women_35_to_50_doctors +
   percentage_men * men_35_to_50 * men_35_to_50_doctors) = 0.22025 :=
by
  sorry

end selection_probability_correct_l205_205842


namespace max_dot_product_l205_205449

-- Definitions of points and vectors
def A := (0 : ℝ, 2 : ℝ)
def B := (-2 : ℝ, 0 : ℝ)
def P (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)

-- Curve definition: P(α) lies on the curve x = sqrt(1 - y^2)
def on_curve (α : ℝ) : Prop := P α.1 = Real.sqrt (1 - (P α).2 ^ 2)

-- Vectors BA and BP
def vec_BA : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)
def vec_BP (α : ℝ) : ℝ × ℝ := ((P α).1 - B.1, (P α).2 - B.2)

-- Dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem stating the maximum value of the dot product
theorem max_dot_product : ∃ α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2),
  dot_product vec_BA (vec_BP α) = 4 + 2 * Real.sqrt 2 :=
sorry

end max_dot_product_l205_205449


namespace smallest_whole_number_larger_than_perimeter_l205_205976

theorem smallest_whole_number_larger_than_perimeter (s : ℝ) (h1 : 7 + 23 > s) (h2 : 7 + s > 23) (h3 : 23 + s > 7) : 
  60 = Int.ceil (7 + 23 + s - 1) :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l205_205976


namespace max_working_groups_l205_205585

theorem max_working_groups (teachers groups : ℕ) (memberships_per_teacher group_size : ℕ) 
  (h_teachers : teachers = 36) (h_memberships_per_teacher : memberships_per_teacher = 2)
  (h_group_size : group_size = 4) 
  (h_max_memberships : teachers * memberships_per_teacher = 72) :
  groups ≤ 18 :=
by
  sorry

end max_working_groups_l205_205585


namespace problem_sum_l205_205285

theorem problem_sum : 
  (∑' (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ b) (h₃ : b ≤ c), 1 / (2^a * 4^b * 8^c)) = 1 / 13671 :=
by
  sorry

end problem_sum_l205_205285


namespace find_p_q_l205_205951

theorem find_p_q (a b c : ℤ) (h_quad : ∀ x : ℝ, 6 * x^2 - 19 * x + 3 = 0) (q : ℤ) (p : ℤ) :
  let diff := |(19 + sqrt (b^2 - 4 * a * c)) / (2 * a) - (19 - sqrt (b^2 - 4 * a * c)) / (2 * a)|
  diff = (sqrt p) / q → p = 289 ∧ q = 6 → p + q = 295 := 
by
  sorry

end find_p_q_l205_205951


namespace addends_are_negative_l205_205978

theorem addends_are_negative (a b : ℤ) (h1 : a + b < a) (h2 : a + b < b) : a < 0 ∧ b < 0 := 
sorry

end addends_are_negative_l205_205978


namespace total_fish_bought_l205_205011

theorem total_fish_bought (gold_fish blue_fish : Nat) (h1 : gold_fish = 15) (h2 : blue_fish = 7) : gold_fish + blue_fish = 22 := by
  sorry

end total_fish_bought_l205_205011


namespace batsman_average_after_17th_inning_batsman_average_after_17th_is_eight_l205_205219

variable (A : ℕ) -- Assume the average score before the 17th inning is a natural number.

theorem batsman_average_after_17th_inning
  (h1 : 16 * A + 56 = 17 * (A + 3)) : A = 5 :=
by
  sorry

theorem batsman_average_after_17th_is_eight
  (h1 : 16 * A + 56 = 17 * (A + 3)) : (A + 3) = 8 :=
by
  have hA : A = 5 := batsman_average_after_17th_inning A h1
  rw [hA]
  rfl

end batsman_average_after_17th_inning_batsman_average_after_17th_is_eight_l205_205219


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205091

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205091


namespace area_of_figure_formed_by_parabola_line_and_x_axis_l205_205538

open Real

theorem area_of_figure_formed_by_parabola_line_and_x_axis :
  ∫ x in -1..0, (x^2 - x) = 5 / 6 :=
by
  sorry

end area_of_figure_formed_by_parabola_line_and_x_axis_l205_205538


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205134

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205134


namespace mod_product_invariant_l205_205006

theorem mod_product_invariant (d : ℤ) (a : ℕ → ℤ) (n : ℕ) (i : ℕ) (k : ℤ) :
  (∏ j in finset.range n, a j) % d = (∏ j in finset.range n, if j = i then a j + k * d else a j) % d :=
by sorry

end mod_product_invariant_l205_205006


namespace equilateral_triangle_to_square_l205_205730

noncomputable def side_length_square {a : ℝ} (h : a > 0) : ℝ :=
  (a * Real.sqrt 3) / 2

theorem equilateral_triangle_to_square (a : ℝ) (h : a > 0) :
  ∃ parts : List (Set (ℝ × ℝ)),
    (∀ part ∈ parts, True) ∧
    (∀ (s : ℝ), s = side_length_square h → 
        ∃ square : Set (ℝ × ℝ), square.shape = "square" ∧ 
        square.area = Set.union parts).area 
     sorry

end equilateral_triangle_to_square_l205_205730


namespace range_of_a_for_intersections_l205_205407

theorem range_of_a_for_intersections (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    (x₁^3 - 3 * x₁ = a) ∧ (x₂^3 - 3 * x₂ = a) ∧ (x₃^3 - 3 * x₃ = a)) ↔ 
  (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_for_intersections_l205_205407


namespace find_question_mark_l205_205656

theorem find_question_mark : 
  let x := 2 in
  (5568 / 87)^(1 / 3) + (72 * x)^(1 / 2) = (256)^(1 / 2) :=
by
  sorry

end find_question_mark_l205_205656


namespace train_length_l205_205687

theorem train_length (L : ℝ) 
  (h1 : (L / 20) = ((L + 1500) / 70)) : L = 600 := by
  sorry

end train_length_l205_205687


namespace programs_produce_same_result_l205_205698

-- Define Program A's computation
def programA_sum : ℕ := (List.range (1000 + 1)).sum -- Sum of numbers from 0 to 1000

-- Define Program B's computation
def programB_sum : ℕ := (List.range (1000 + 1)).reverse.sum -- Sum of numbers from 1000 down to 0

theorem programs_produce_same_result : programA_sum = programB_sum :=
  sorry

end programs_produce_same_result_l205_205698


namespace painting_count_l205_205453

def num_distinct_paintings : Nat := 4

theorem painting_count :
  ∃ n, n = 10 ∧ 
       (8 of 10 disks are yellow) ∧ 
       (1 of 10 disks is blue) ∧ 
       (1 of 10 disks is red) ∧ 
       (two paintings that can be obtained from one another by a rotation or a reflection of the entire decagon are considered the same) ∧ 
       (the blue and red disks must be adjacent) ∧
       (number of different paintings) = 4 :=
by
  use 4
  repeat { sorry }

end painting_count_l205_205453


namespace divisors_of_n_squared_l205_205882

open Nat

theorem divisors_of_n_squared (n : ℕ) (h1 : ∃ p q : ℕ, p ≠ q ∧ prime p ∧ prime q ∧ (n = p^3 ∨ n = p * q)) :
  ∃ k : ℕ, k = 7 ∨ k = 9 ∧ (number_of_divisors (n^2) = k) := by
  sorry

end divisors_of_n_squared_l205_205882


namespace mirror_area_proof_l205_205506

-- Definitions of conditions
def outer_width := 100
def outer_height := 70
def frame_width := 15
def mirror_width := outer_width - 2 * frame_width -- 100 - 2 * 15 = 70
def mirror_height := outer_height - 2 * frame_width -- 70 - 2 * 15 = 40

-- Statement of the proof problem
theorem mirror_area_proof : 
  (mirror_width * mirror_height) = 2800 := 
by
  sorry

end mirror_area_proof_l205_205506


namespace problem_1_problem_2_l205_205769

-- Define the sequence a_n
def seq_a : ℕ → ℕ
| 0       := 0  -- should never be used
| 1       := 1
| (n + 1) := 3 * seq_a n + 3^n - 1

-- Proof problem 1
theorem problem_1 (λ : ℝ) : (∀ n : ℕ, ∀ k : ℕ, (1 ≤ n + 1) → (1 ≤ k) → 
( (λ + seq_a (n + 1)) / (3^(n + 1)) - (λ + seq_a n) / (3^n) = (λ + seq_a k) / (3^k))) → λ = -1 / 2 := 
sorry

-- Define S_n based on the given sequence
def S : ℕ → ℝ
| 0     := 0
| (n+1) := (4 * (n + 1) - 2) / (3 * seq_a (n + 1) - (n + 1) - 1) + S n

-- Proof problem 2
theorem problem_2 (n : ℕ) : S n < 3 :=
sorry

end problem_1_problem_2_l205_205769


namespace probability_angle_AQB_is_obtuse_l205_205901

noncomputable def point := (ℝ × ℝ)

def A : point := (-1, 3)
def B : point := (5, -1)
def C : point := (2 * Real.pi + 2, -1)
def D : point := (2 * Real.pi + 2, 5)
def E : point := (-1, 5)

def is_interior_of_pentagon (Q : point) : Prop := sorry  -- Define this in terms of the vertices A, B, C, D, E.

def is_obtuse_angle (A B Q : point) : Prop := sorry  -- Define this to check if the angle AQB is obtuse.

-- The main theorem the equivalent problem statement.
theorem probability_angle_AQB_is_obtuse (Q : point) (H : is_interior_of_pentagon Q) : 
  probability (is_obtuse_angle A B Q | is_interior_of_pentagon Q) = 5 * Real.pi / (5 * Real.pi + 6) :=
sorry

end probability_angle_AQB_is_obtuse_l205_205901


namespace find_abcd_l205_205574

theorem find_abcd {abcd abcde M : ℕ} :
  (M > 0) ∧ (∃ e, M % 100000 = e ∧ M^2 % 100000 = e) ∧ (M // 10000 > 0) ∧ (M // 10000 < 10) →
  (abcd = M // 10) →
  abcd = 9687 :=
by
  sorry

end find_abcd_l205_205574


namespace find_four_digit_number_abcd_exists_l205_205575

theorem find_four_digit_number_abcd_exists (M : ℕ) (H1 : M > 0) (H2 : M % 10 ≠ 0) 
    (H3 : M % 100000 = M^2 % 100000) : ∃ abcd : ℕ, abcd = 2502 :=
by
  -- Proof is omitted
  sorry

end find_four_digit_number_abcd_exists_l205_205575


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205179

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205179


namespace highest_number_paper_l205_205831

theorem highest_number_paper (n : ℕ) (h : 1 / (n : ℝ) = 0.01020408163265306) : n = 98 :=
sorry

end highest_number_paper_l205_205831


namespace correct_option_is_pudong_l205_205555

/-- 
Given the conditions about the pattern of world city development (aggregation followed by radiation, radiating effect on surrounding areas, and forming a mutually influential and interdependent positive interactive relationship), determine which option best reflects this viewpoint.
-/
def city_development_pattern (A B C D : Prop) : Prop :=
  let condition1 := "the relationship between the central city and its surrounding areas first involves aggregation, followed by radiation",
      condition2 := "initially, resources are aggregated in the central city",
      condition3 := "then, the central city has a radiating effect on the surrounding areas, helping these areas to develop more quickly",
      condition4 := "finally, the central city and the surrounding small and medium-sized cities form a mutually influential and interdependent positive interactive relationship"
  in
  (A → "Promoting one benefits another" ∧ "economic development of Yangzhou and Yizhou during the late Tang Dynasty" ≠ "pattern of world city development") ∧
  (B → "The Thirteen Factories of Guangzhou" ∧ "Qing Dynasty's closed-door policy" ≠ "pattern of world city development") ∧
  (C → "Pudong, Shanghai" ∧ "decided to develop Pudong in 1992" = "pattern of world city development") ∧
  (D → "New Amsterdam" ∧ "development mainly refers to the development of the city itself" ≠ "pattern of world city development") ∧
  C

theorem correct_option_is_pudong : city_development_pattern A B C D := 
sorry

end correct_option_is_pudong_l205_205555


namespace find_solution_l205_205309

def satisfies_conditions (x y z : ℝ) :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solution (x y z : ℝ) :
  satisfies_conditions x y z →
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
by
  sorry

end find_solution_l205_205309


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205135

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205135


namespace Yoque_borrowed_150_l205_205639

noncomputable def Yoque_borrowed_amount (X : ℝ) : Prop :=
  1.10 * X = 11 * 15

theorem Yoque_borrowed_150 (X : ℝ) : Yoque_borrowed_amount X → X = 150 :=
by
  -- proof will be filled in
  sorry

end Yoque_borrowed_150_l205_205639


namespace correct_sum_l205_205726

def is_prime (n : ℕ) : Prop := sorry -- Assume definition is provided

def valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

def three_digit_prime_with_permutable_digits (p : ℕ) : Prop :=
  p > 100 ∧ p < 1000 ∧ is_prime p ∧ (∃ a b c : ℕ,
    p = 100 * a + 10 * b + c ∧ valid_digit a ∧ valid_digit c ∧ is_prime (100 * a + 10 * c + b))

def sum_of_valid_primes : ℕ :=
  Nat.sum (List.filter three_digit_prime_with_permutable_digits (List.range 899)) -- List.range 899 to cover 101 to 999
  
theorem correct_sum : sum_of_valid_primes = S :=
by sorry

end correct_sum_l205_205726


namespace find_n_l205_205782

theorem find_n (n : ℕ) (h1 : 0 < n) (h2 : (binomial n 1) + (binomial n 2) = 36) : n = 8 := 
by
  sorry

end find_n_l205_205782


namespace trapezoid_mn_correct_l205_205903

open EuclideanGeometry

noncomputable def trapezoid_mn (a b : ℝ) (M K A B C D N L : Point)
  (h1 : on_line AB M) (h2 : on_line AB K)
  (h3 : on_line CD N) (h4 : on_line CD L)
  (h5 : parallel MN AD)
  (h6 : area_ratio (trapezoid M B C N) (trapezoid M A D N) = 1 / 5)
  (h7 : distance B C = a)
  (h8 : distance A D = b) : ℝ :=
  MN

theorem trapezoid_mn_correct (a b : ℝ) (M K A B C D N L : Point)
  (h1 : on_line AB M) (h2 : on_line AB K)
  (h3 : on_line CD N) (h4 : on_line CD L)
  (h5 : parallel MN AD)
  (h6 : area_ratio (trapezoid M B C N) (trapezoid M A D N) = 1 / 5)
  (h7 : distance B C = a)
  (h8 : distance A D = b) :
  MN = sqrt ((5 * a ^ 2 + b ^ 2) / 6) :=
  sorry

end trapezoid_mn_correct_l205_205903


namespace magnitude_diff_of_unit_vectors_l205_205417

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem magnitude_diff_of_unit_vectors (a b : ℝ × ℝ) 
    (ha : magnitude a = 1) 
    (hb : magnitude b = 1) 
    (angle_ab : dot_product a b = 1 / 2) : 
    magnitude (a - b) = 1 := 
sorry

end magnitude_diff_of_unit_vectors_l205_205417


namespace find_selling_price_l205_205507

variable (SP CP : ℝ)

def original_selling_price (SP CP : ℝ) : Prop :=
  0.9 * SP = CP + 0.08 * CP

theorem find_selling_price (h1 : CP = 17500)
  (h2 : original_selling_price SP CP) : SP = 21000 :=
by
  sorry

end find_selling_price_l205_205507


namespace students_play_neither_l205_205840

theorem students_play_neither (total_students : ℕ) (football_players : ℕ) 
  (tennis_players : ℕ) (both_sports_players : ℕ) :
  total_students = 60 →
  football_players = 36 →
  tennis_players = 30 →
  both_sports_players = 22 →
  (total_students - (football_players + tennis_players - both_sports_players) = 16) :=
by
  intros h_total h_football h_tennis h_both
  rw [h_total, h_football, h_tennis, h_both]
  sorry

end students_play_neither_l205_205840


namespace Sn_divisible_by_mfact_Sn_not_divisible_by_mfact_mul_nplus1_l205_205905

def S (m n : ℕ) : ℤ :=
1 + ∑ k in Finset.range m, (-1)^(k+1) * ((n+(k+1)+1)! / (n! * (n+(k+1))))

theorem Sn_divisible_by_mfact (m n : ℕ) : m! ∣ S m n := 
sorry

theorem Sn_not_divisible_by_mfact_mul_nplus1 : ∃ m n : ℕ, ¬ (m! * (n+1)) ∣ S m n := 
sorry

end Sn_divisible_by_mfact_Sn_not_divisible_by_mfact_mul_nplus1_l205_205905


namespace floor_sufficient_but_not_necessary_l205_205828

theorem floor_sufficient_but_not_necessary {x y : ℝ} : 
  (∀ x y : ℝ, (⌊x⌋₊ = ⌊y⌋₊) → abs (x - y) < 1) ∧ 
  ¬ (∀ x y : ℝ, abs (x - y) < 1 → (⌊x⌋₊ = ⌊y⌋₊)) :=
by
  sorry

end floor_sufficient_but_not_necessary_l205_205828


namespace area_ratio_l205_205874

theorem area_ratio (ABCDE : ConvexPentagon) 
  (h1 : parallel AB DE)
  (h2 : parallel BC AE)
  (h3 : parallel BD CE)
  (angle_ABC : ∠ ABC = 100)
  (length_AB : length AB = 4)
  (length_BC : length BC = 6)
  (length_DE : length DE = 18) : 
  let ratio := area_ratio (triangle ABC) (triangle CDE) in
  let m := 127 in
  let n := 900 in
  ratio = m / n ∧ m.coprime n ∧ m + n = 1027 := 
sorry

end area_ratio_l205_205874


namespace correct_choices_l205_205441

/-- Definition of balls and events as per given conditions. --/
def num_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2

/-- Definition of events A and B. --/
def A : Prop := one_ball_drawn_is_red (first_draw)
def B : Prop := one_ball_drawn_is_red (second_draw)
def not_A : Prop := one_ball_drawn_is_white (first_draw)

/-- Probability of event A --/
axiom prob_A : ℙ(A) = 2/3

/-- Probability of event B given not A --/
axiom prob_notA_B : ℙ(B | not_A) = 4/5

/-- Theorem proving correct conclusions --/
theorem correct_choices :
  (ℙ(A) = 2/3) ∧ (ℙ(B | not_A) = 4/5) := by
  exact ⟨prob_A, prob_notA_B⟩

end correct_choices_l205_205441


namespace find_a_value_l205_205415

noncomputable def solve_for_a (A B : Set ℝ) (a : ℝ) : Prop :=
  A = {2 * a, 3} ∧
  B = {2, 3} ∧
  A ∪ B = {2, 3, 4} →
  a = 2

-- Lean statement for the problem
theorem find_a_value (A B : Set ℝ) (a : ℝ) : solve_for_a A B a :=
begin
  sorry,
end

end find_a_value_l205_205415


namespace repetend_of_5_over_13_l205_205311

theorem repetend_of_5_over_13 : (∃ r : ℕ, r = 384615) :=
by
  let d := 13
  let n := 5
  let r := 384615
  -- Definitions to use:
  -- d is denominator 13
  -- n is numerator 5
  -- r is the repetend 384615
  sorry

end repetend_of_5_over_13_l205_205311


namespace monotonicity_of_f_on_interval_l205_205290

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x - 2

theorem monotonicity_of_f_on_interval (a b : ℝ) (h1 : a = -3) (h2 : b = 0) :
  ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 2 → f x1 a b ≥ f x2 a b := 
by
  sorry

end monotonicity_of_f_on_interval_l205_205290


namespace find_hyperbola_fixed_line_through_P_l205_205391

-- Define the conditions of the problem
def center_origin (C : Type) : Prop := 
  C.center = (0, 0)

def left_focus_at (C : Type) : Prop :=
  C.left_focus = (-2 * Real.sqrt 5, 0)

def eccentricity_sqrt_5 (C : Type) : Prop :=
  C.eccentricity = Real.sqrt 5

-- Define the specific hyperbola equation based on the conditions
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 16) = 1

-- State the results to prove
theorem find_hyperbola (C : Type) :
  center_origin C →
  left_focus_at C →
  eccentricity_sqrt_5 C →
  ∀ x y : ℝ, hyperbola_equation x y :=
sorry

-- Define the fixed line for point P
def fixed_line (P : Type) : Prop :=
  P.x = -1

-- State the results to prove for point P
theorem fixed_line_through_P (M N A1 A2 P : Type) (x y : ℝ) :
  (M.line_passing_through (Point.mk (-4, 0))).intersects_left_branch (N : Point) →
  M.in_second_quadrant →
  (A1.coords = (-2, 0)) ∧ (A2.coords = (2, 0)) →
  Point.mk x y = P →
  fixed_line P :=
sorry

end find_hyperbola_fixed_line_through_P_l205_205391


namespace sequence_to_rational_l205_205870

open Nat

def is_least_prime_divisor (p n : ℕ) : Prop :=
  prime p ∧ p ∣ n ∧ ∀ q, prime q ∧ q ∣ n → p ≤ q

def sequence (a : ℕ → ℕ) :=
  a 1 > 0 ∧ a 2 > 0 ∧ ∀ n ≥ 2, is_least_prime_divisor (a (n + 1)) (a (n - 1) + a n)

def x_from_sequence (a : ℕ → ℕ) : ℝ :=
  let digits := [a 1, a 2, a 3, a 4] -- Extend as necessary to represent the sequence
  -- Concatenate digits to form a real number
  sorry

theorem sequence_to_rational (a : ℕ → ℕ):
  sequence a → ∃ x : ℚ, x_from_sequence a = x :=
by
  intros h
  sorry

end sequence_to_rational_l205_205870


namespace evaluate_expression_l205_205303

def cyclical_i (z : ℂ) : Prop := z^4 = 1

theorem evaluate_expression (i : ℂ) (h : cyclical_i i) : i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  sorry

end evaluate_expression_l205_205303


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205140

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205140


namespace max_shareholder_ownership_l205_205839

theorem max_shareholder_ownership (total_shareholders : ℕ)
  (threshold_shareholders : ℕ)
  (min_threshold_percentage : ℕ)
  (max_single_shareholder_percentage : ℕ) :
  total_shareholders = 100 → threshold_shareholders = 66 → min_threshold_percentage = 50 → max_single_shareholder_percentage = 25 → 
  (∀ (x : ℕ), (x ≤ total_shareholders) → ∀ (y : ℕ), (y ≤ threshold_shareholders) → ((x = total_shareholders) ∧ (y = threshold_shareholders) ∧ ((Σ a : fin threshold_shareholders, a.val) ≥ min_threshold_percentage)) → (x.val ≤ max_single_shareholder_percentage)) := 
by {
  rename x → h_estimations;
  intro_conditions;
  apply_correct_translations;
  verify_max_constraints;
  show_equivalent;
  sorry
}

end max_shareholder_ownership_l205_205839


namespace three_letter_sets_initials_count_l205_205818

theorem three_letter_sets_initials_count :
  let letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}
  in (letters.card = 10) -> (Finset.card (Finset.univ.image (λ x, x.val.toList)) = 720) :=
by
  intro
  sorry

end three_letter_sets_initials_count_l205_205818


namespace range_of_k_l205_205829

def f (x : ℝ) (k : ℝ) : ℝ :=
  if x ≤ 0 then x / (x - 2) + k * x^2 else Real.log x

theorem range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ f x k = 0 ∧ f y k = 0) ↔ 0 ≤ k :=
by
  sorry

end range_of_k_l205_205829


namespace distance_of_freely_falling_object_l205_205811

noncomputable def distance_traveled (g t₀ : ℝ) : ℝ :=
  ∫ x in 0..t₀, g * x

theorem distance_of_freely_falling_object (g t₀ : ℝ) :
  distance_traveled g t₀ = (1 / 2) * g * t₀^2 :=
by
  sorry

end distance_of_freely_falling_object_l205_205811


namespace least_pos_int_div_by_four_distinct_primes_l205_205098

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205098


namespace hyperbola_center_origin_point_P_on_fixed_line_l205_205348

variables {x y : ℝ}

def hyperbola (h : (x^2 / 4) - (y^2 / 16) = 1) : Prop := true

theorem hyperbola_center_origin 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (left_focus : ∃ x y : ℝ, x = -2 * Real.sqrt 5 ∧ y = 0)
  (eccentricity_root5 : ∃ e : ℝ, e = Real.sqrt 5) :
  hyperbola ((x^2 / 4) - (y^2 / 16) = 1) := 
sorry

theorem point_P_on_fixed_line
  (line_through_point : ∃ x y : ℝ, x = -4 ∧ y = 0)
  (M_second_quadrant : ∃ Mx My : ℝ, Mx < 0 ∧ My > 0)
  (A1_vertex : ∃ x y : ℝ, x = -2 ∧ y = 0)
  (A2_vertex : ∃ x y : ℝ, x = 2 ∧ y = 0)
  (P_intersection : ∃ Px Py : ℝ, Px = -1) :
  Px = -1 :=
sorry

end hyperbola_center_origin_point_P_on_fixed_line_l205_205348


namespace largest_triangle_perimeter_l205_205256

theorem largest_triangle_perimeter :
  ∀ (x : ℕ), 1 < x ∧ x < 15 → (7 + 8 + x = 29) :=
by
  intro x
  intro h
  sorry

end largest_triangle_perimeter_l205_205256


namespace magnitude_sum_l205_205419

variables (a b : ℝˣ) -- This is the Lean notation for vectors in ℝ^n
variables (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 2) (h3 : ‖a - b‖ = √3)

theorem magnitude_sum (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 2) (h3 : ‖a - b‖ = √3) :
  ‖a + b‖ = √7 :=
sorry -- Proof goes here

end magnitude_sum_l205_205419


namespace Benny_total_hours_l205_205706

def hours_per_day : ℕ := 7
def days_worked : ℕ := 14

theorem Benny_total_hours : hours_per_day * days_worked = 98 := by
  sorry

end Benny_total_hours_l205_205706


namespace tetrahedron_coloring_count_is_36_l205_205734

noncomputable def tetrahedron_coloring_ways : ℕ := 
  36

theorem tetrahedron_coloring_count_is_36 :
  ∃ (f : ℕ), f = tetrahedron_coloring_ways ∧ f = 36 :=
by {
  use 36,
  exact ⟨rfl, rfl⟩,
  sorry
}

end tetrahedron_coloring_count_is_36_l205_205734


namespace least_positive_integer_divisible_by_four_primes_l205_205118

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205118


namespace proof_f_of_fraction_l205_205999

def f (x : ℝ) : ℝ :=
if x < 1 then x
else x^3 - (1 / x) + 1

theorem proof_f_of_fraction :
  f (1 / f 2) = 2 / 17 :=
by
  have h1 : f 2 = 17 / 2 := sorry
  have h2 : 1 / (17 / 2) = 2 / 17 := sorry
  have h3 : (2 / 17) < 1 := sorry
  have h4 : f (2 / 17) = 2 / 17 := sorry
  exact h4

end proof_f_of_fraction_l205_205999


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205137

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205137


namespace repeated_root_quadratic_l205_205890

theorem repeated_root_quadratic (θ : ℝ) (h_acute : 0 < θ ∧ θ < π / 2) :
    (∃ x : ℝ, (x^2 + 4 * x * Real.cos θ + Real.cot θ) = 0 ∧ 
                 ∀ y : ℝ, (y^2 + 4 * y * Real.cos θ + Real.cot θ) = 0 -> y = x) ↔
      (θ = π / 12 ∨ θ = 5 * π / 12) :=
begin
  sorry
end

end repeated_root_quadratic_l205_205890


namespace evaluate_expression_l205_205304

def cyclical_i (z : ℂ) : Prop := z^4 = 1

theorem evaluate_expression (i : ℂ) (h : cyclical_i i) : i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  sorry

end evaluate_expression_l205_205304


namespace range_of_m_as_triangle_function_l205_205435

noncomputable def is_triangle_function (f : ℝ → ℝ) (A : Set ℝ) : Prop :=
  ∀ a b c ∈ A, 
    a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * Real.log x + m 

theorem range_of_m_as_triangle_function :
  (is_triangle_function (λ x, f x m) (Set.Icc (1 / Real.exp 2) Real.exp)) ↔
    m ∈ Set.Ioi ((Real.exp 2 + 2) / Real.exp) :=
by
  sorry

end range_of_m_as_triangle_function_l205_205435


namespace correct_statements_l205_205065

def data : List (ℕ × ℕ) := [(1, 19), (2, 32), (4, 44), (6, 40), (10, 52), (13, 53), (20, 54)]

def regression_slope (x_values y_values : List ℕ) : ℕ := 15 / 10

def average (values : List ℕ) : ℚ :=
  (values.foldl (· + ·) 0 : ℚ) / values.length

theorem correct_statements :
  let x_values := data.map Prod.fst
  let y_values := data.map Prod.snd
  x_values.range = 19 ∧
  let m := 42 - regression_slope x_values y_values * average x_values
  m = 30 :=
by
  sorry

end correct_statements_l205_205065


namespace trapezoid_side_length_l205_205681

theorem trapezoid_side_length (s : ℝ) (A : ℝ) (x : ℝ) (y : ℝ) :
  s = 1 ∧ A = 1 ∧ y = 1/2 ∧ (1/2) * ((x + y) * y) = 1/4 → x = 1/2 :=
by
  intro h
  rcases h with ⟨hs, hA, hy, harea⟩
  sorry

end trapezoid_side_length_l205_205681


namespace length_EF_l205_205456

-- Definition of trapezoid and given properties
structure Trapezoid where
  A B C D E F M N : Type
  AD BC : ℝ 
  cond1 : AD = 17
  cond2 : BC = 9
  M_is_midpoint_AC : True -- Assume we have this defined
  N_is_midpoint_BD : True -- Assume we have this defined
  E_on_AD : True -- Assume E is on AD
  F_on_BC : True -- Assume F is on BC
  M_N_midsegments_property : ∀ (MN : ℝ), MN = (AD - BC) / 2
  MENF_is_rectangle : True -- Assume rectangle property

-- Main theorem stating the length of segment EF
theorem length_EF (T : Trapezoid) : 
  let EF := (T.AD - T.BC) / 2
  EF = 4 :=
by
  suffices EF_base := 4
  sorry

end length_EF_l205_205456


namespace minjeong_walk_distance_l205_205896

noncomputable def park_side_length : ℕ := 40
noncomputable def square_sides : ℕ := 4

theorem minjeong_walk_distance (side_length : ℕ) (sides : ℕ) (h : side_length = park_side_length) (h2 : sides = square_sides) : 
  side_length * sides = 160 := by
  sorry

end minjeong_walk_distance_l205_205896


namespace students_left_on_bus_l205_205062

theorem students_left_on_bus (initial_students : ℕ) (students_got_off : ℕ) (final_students : ℕ) 
  (h1 : initial_students = 10) (h2 : students_got_off = 3) : final_students = 7 :=
begin
  sorry
end

end students_left_on_bus_l205_205062


namespace mice_boring_sum_l205_205024

theorem mice_boring_sum (n : ℕ) : 
  let big_mouse_boring := (finset.range n).sum (λ k, 2^k)
  let small_mouse_boring := (finset.range n).sum (λ k, (1/2)^k)
  let total_distance := big_mouse_boring + small_mouse_boring
  total_distance = 2^n - (1 / 2^(n-1)) + 1 :=
by 
  let big_mouse_boring := (finset.range n).sum (λ k, 2^k)
  let small_mouse_boring := (finset.range n).sum (λ k, (1/2)^k)
  let total_distance := big_mouse_boring + small_mouse_boring
  sorry

end mice_boring_sum_l205_205024


namespace C_should_pay_D_70_yuan_l205_205003

theorem C_should_pay_D_70_yuan
  (contributed_equal_amount : ∀ {x : ℝ}, ∀ A B C D : ℝ, x = A ∧ x = B ∧ x = C ∧ x = D)
  (identical_gifts : ∃ (g : ℝ), ∀ a b c d : ℝ, g = b - a ∧ g = c - a ∧ g = d - a)
  (additional_gifts : ∀ a₁ a₂ a₃ a₄ : ℝ, 
     let D := a₁ in 
     let A := a₁ + 3 in 
     let B := a₁ + 7 in 
     let C := a₁ + 14 in 
     ∃ b₁ b₂ b₃ b₄ : ℝ, b₁ = B - D ∧ b₂ = A - a₁ ∧ b₃ = B - a₁ ∧ b₄ = C - a₁)
  (B_paid_D : ∀ {y₁ y₂ : ℝ}, y₁ = 14 ∧ y₂ = 14)
  (B_not_paid_A : ∀ {p₁ p₂ : ℝ}, ∀ r : nat, p₁ = p₂ ∧ p₂ = r → p₁ = 0)
  : ∀ (C D : ℝ), ∃ (amount : ℝ), amount = 70 := by
  sorry

end C_should_pay_D_70_yuan_l205_205003


namespace select_committee_l205_205247

open Finset

variables {α : Type*} [Fintype α]

noncomputable def choose {n k : ℕ} : ℕ := n.choose k

-- Define the conditions
def num_ways_to_choose_3 (n : ℕ) : Prop := choose n 3 = 20

-- Define the question
def num_ways_to_choose_4 (n : ℕ) : ℕ := choose n 4

theorem select_committee (n : ℕ) (h : num_ways_to_choose_3 n) :
  num_ways_to_choose_4 n = 15 :=
begin
  sorry,
end

end select_committee_l205_205247


namespace exists_scientist_with_one_friend_l205_205702

variable {S : Type} [Fintype S] (F : S → S → Prop) [DecidableRel F]

theorem exists_scientist_with_one_friend
  (h_friend_symm : ∀ {a b : S}, F a b → F b a)
  (h_unique_friends : ∀ {a b : S}, (a ≠ b) → (card {c | F a c} = card {c | F b c} → F a b → False))
  : ∃ s : S, card {c : S | F s c} = 1 := 
sorry

end exists_scientist_with_one_friend_l205_205702


namespace cubic_sum_of_reciprocals_roots_l205_205787

theorem cubic_sum_of_reciprocals_roots :
  ∀ (a b c : ℝ),
  a ≠ b → b ≠ c → c ≠ a →
  0 < a ∧ a < 1 → 0 < b ∧ b < 1 → 0 < c ∧ c < 1 →
  (24 * a^3 - 38 * a^2 + 18 * a - 1 = 0) ∧
  (24 * b^3 - 38 * b^2 + 18 * b - 1 = 0) ∧
  (24 * c^3 - 38 * c^2 + 18 * c - 1 = 0) →
  ((1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 2 / 3) :=
by intros a b c neq_ab neq_bc neq_ca a_range b_range c_range roots_eqns
   sorry

end cubic_sum_of_reciprocals_roots_l205_205787


namespace number_of_primes_from_conditions_l205_205321

theorem  number_of_primes_from_conditions :
  ∀ (n : ℕ), n ≥ 2 → ¬ ∃ p : ℕ, p = 2 * (n ^ 3 + 1) ∧ nat.prime p :=
begin
  intros n hn,
  have h : 2 * (n ^ 3 + 1) = 2 * (n + 1) * (n ^ 2 - n + 1), sorry,
  simp [h],
  have hn_positive : n + 1 > 1 := nat.succ_lt_succ hn,
  have hn2 : n ^ 2 - n + 1 > 1, sorry,
  exact nat.not_prime_mul hn hn2,
end

end number_of_primes_from_conditions_l205_205321


namespace count_floor_eq_l205_205820

-- Define the floor function in Lean
def floor (a : ℚ) : ℤ := int.floor a

-- Define the problem statement
theorem count_floor_eq (positive_integers : set ℕ) :
  let k := λ (x : ℕ), floor (x / 20) in
  let j := λ (x : ℕ), floor (x / 17) in
  set.card {x : ℕ | x ∈ positive_integers ∧ k x = j x} = 57 :=
by
  sorry

end count_floor_eq_l205_205820


namespace reduce_sugar_consumption_l205_205989

theorem reduce_sugar_consumption (initial_price new_price : ℝ) (consumption_reduction_percentage : ℝ) :
  initial_price = 6 → 
  new_price = 7.50 → 
  consumption_reduction_percentage = 25 →
  reduce_consumption_by_percentage initial_price new_price = consumption_reduction_percentage := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end reduce_sugar_consumption_l205_205989


namespace sequence_sum_sum_sequence_sequence_sum_l205_205805

theorem sequence_sum :
  let a : ℕ → ℚ := λ n,
    if h : n = 0 then 1/2 else
      by {
        have ih : (⦃m} ∈ nat.succ m |().ih

  
    match n with
    | nat.succ n => 3 ) - 1 + 1/2 |()
--          ))
  sorry        
  (Sn : ℕ → ℚ) .

:= sorry
    else
      (succ_eq_add_one _).trans
  
  a, sum_eq_zero

  sum, nat_eq_not_eq
  1_n _ a_n_sum_sequence: _,_, sum_eq_sum_the_equal'_

 := match 2016 with  sum_the_assum := (, #

      

(S _ ) => nat_eq_trans.

theorem sum_sequence :
  let a : ℕ → ℚ := λ n,
    if h : n = 0 then 1/2 else 3 * (a (n - 1)) + 1
  ∀ n : ℕ, let S : ℕ → ℚ := λ k, ∑ i in finset.range k, a i in
    2016 = a (2016) :=  finite_sum_geometric2016 -2017'
    (2016 ): sum_eq := .:
 := let sum:
 := natural⟨ :=
math.
sum := { := ⦘
{sum_seq}[{ := 2016 = a_n
use_all_def


theorem sequence_sum :
  let a : ℕ → ℚ := λ n, if n = 0 then 1/2 else 3 ^ n - 1 - 1/2 in
  let S : ℕ → ℚ := λ n, ∑ i in finset.range (n + 1), if i = 0 then 1/2 else 3 ^ i - 1 - 1/2 in
  S 2016 =  (3^2016 - 2017) / 2 := sorry

end sequence_sum_sum_sequence_sequence_sum_l205_205805


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205181

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205181


namespace sakshi_work_days_l205_205915

theorem sakshi_work_days (tanya_days : ℕ) (efficiency_increase : ℕ) (efficiency_fraction : ℚ) : 
  efficiency_increase = 25 → tanya_days = 16 → efficiency_fraction = 1.25 → 
  let sakshi_days := tanya_days * efficiency_fraction in 
  sakshi_days = 20 := by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sakshi_work_days_l205_205915


namespace correct_statements_B_and_D_l205_205270

noncomputable def f1 (x : ℝ) : ℝ := |x| / x

def g (x : ℝ) : ℝ :=
if x ≥ 0 then 1 else -1

def statementB (f : ℝ → ℝ) : Prop :=
(∃ y, f 1 = y) → ∀ y, (f 1 = y →  ∃! x, x = 1)

noncomputable def f2 (x : ℝ) : ℝ := x^2 + 2 + 1 / (x^2 + 2)

def statementC : Prop :=
¬ ∃ (x : ℝ), f2 x = 2

noncomputable def f3 (x : ℝ) : ℝ := |x - 1| - |x|

def statementD : Prop :=
f3 (f3 (1 / 2)) = 1

theorem correct_statements_B_and_D :
  statementB f ∧ statementD :=
begin
  split,
  -- statement B: Assuming the function is arbitrary,
  -- proof: see problem analysis, skipping with sorry.
  sorry, -- Leaving out the proof for brevity.
  -- statement D: proof: see problem analysis, skipping with sorry.
  sorry  -- Leaving out the proof for brevity.
end

end correct_statements_B_and_D_l205_205270


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205151

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205151


namespace investment_amount_correct_l205_205993

noncomputable def investment_problem : Prop :=
  let initial_investment_rubles : ℝ := 10000
  let initial_exchange_rate : ℝ := 50
  let annual_return_rate : ℝ := 0.12
  let end_year_exchange_rate : ℝ := 80
  let currency_conversion_commission : ℝ := 0.05
  let broker_profit_commission_rate : ℝ := 0.3

  -- Computations
  let initial_investment_dollars := initial_investment_rubles / initial_exchange_rate
  let profit_dollars := initial_investment_dollars * annual_return_rate
  let total_dollars := initial_investment_dollars + profit_dollars
  let broker_commission_dollars := profit_dollars * broker_profit_commission_rate
  let post_commission_dollars := total_dollars - broker_commission_dollars
  let amount_in_rubles_before_conversion_commission := post_commission_dollars * end_year_exchange_rate
  let conversion_commission := amount_in_rubles_before_conversion_commission * currency_conversion_commission
  let final_amount_rubles := amount_in_rubles_before_conversion_commission - conversion_commission

  -- Proof goal
  final_amount_rubles = 16476.8

theorem investment_amount_correct : investment_problem := by {
  sorry
}

end investment_amount_correct_l205_205993


namespace mode_of_team_ages_is_18_l205_205027

theorem mode_of_team_ages_is_18 :
  let ages := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14] in
  mode ages = 18 :=
by
  sorry

end mode_of_team_ages_is_18_l205_205027


namespace solve_for_A_l205_205824

variable (x y : ℝ)

theorem solve_for_A (A : ℝ) : (2 * x - y) ^ 2 + A = (2 * x + y) ^ 2 → A = 8 * x * y :=
by
  intro h
  sorry

end solve_for_A_l205_205824


namespace collinear_X_Y_I_l205_205873

variables {A B C D I X Y : Type}

-- Given a tangential quadrilateral ABCD
def is_tangential_quadrilateral (ABCD : Type) : Prop := sorry

-- Given a circle ω is tangent to the four sides of ABCD with center I
def is_incircle (ω : Type) (ABCD : Type) (I : Type) : Prop := sorry

-- Let ω1 and ω2 be the circumcircles of triangles ACI and BID, respectively
def is_circumcircle (ω1 : Type) (A C I : Type) : Prop := sorry
def is_circumcircle2 (ω2 : Type) (B I D : Type) : Prop := sorry

-- X is the intersection of tangents to ω1 at A and C
def is_intersection_of_tangents (X : Type) (ω1 : Type) (A C : Type) : Prop := sorry

-- Y is the intersection of tangents to ω2 at B and D
def is_intersection_of_tangents2 (Y : Type) (ω2 : Type) (B D : Type) : Prop := sorry

-- Prove that points X, Y, and I are collinear
theorem collinear_X_Y_I (ABCD ω ω1 ω2 : Type) (A B C D I X Y : Type) :
  is_tangential_quadrilateral ABCD →
  is_incircle ω ABCD I →
  is_circumcircle ω1 A C I →
  is_circumcircle2 ω2 B I D →
  is_intersection_of_tangents X ω1 A C →
  is_intersection_of_tangents2 Y ω2 B D →
  collinear X Y I :=
sorry

end collinear_X_Y_I_l205_205873


namespace complex_number_solution_l205_205893

theorem complex_number_solution (z : ℂ) (h : (1 - complex.I) * z = 3 + complex.I) : 
  z = 1 + 2 * complex.I :=
sorry

end complex_number_solution_l205_205893


namespace possible_division_implies_infinite_ways_l205_205299

theorem possible_division_implies_infinite_ways :
  ∃ (division : (Set (Set ℝ)) → Prop), 
    (∀ part ∈ division, MeasureTheory.MeasureLebesgue.measure part = 1) 
    ∧ 
    ∃ (part ∈ division), 
      (∃ (p1 p2 p3 : ℝ × ℝ), 
         p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 
         ∧ dist p1 p2 > 1 ∧ dist p2 p3 > 1 ∧ dist p1 p3 > 1) 
    ∧ 
    Infinite (division) :=
sorry

end possible_division_implies_infinite_ways_l205_205299


namespace C0E_hex_to_dec_l205_205294

theorem C0E_hex_to_dec : 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  result = 3086 :=
by 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  sorry

end C0E_hex_to_dec_l205_205294


namespace impossible_to_equalize_candies_l205_205587

theorem impossible_to_equalize_candies :
  let initial_candies := [1, 2, 3, 4, 5, 6] in
  let total_candies := List.sum initial_candies in
  ∀ moves : ℕ,
    (∀ move : ℕ × ℕ, move.fst < move.snd) →
    let new_total_candies := total_candies + 2 * moves in
    (∀ k : ℕ, new_total_candies = 6 * k) → False :=
by
  sorry

end impossible_to_equalize_candies_l205_205587


namespace degree_of_f_plus_cg_l205_205293

noncomputable def c : ℚ := -5 / 9

def f : Polynomial ℚ := 1 - 12 * Polynomial.X + 3 * Polynomial.X^2 - 4 * Polynomial.X^3 + 5 * Polynomial.X^4

def g : Polynomial ℚ := 3 - 2 * Polynomial.X - 6 * Polynomial.X^3 + 9 * Polynomial.X^4

theorem degree_of_f_plus_cg (h : c = -5 / 9) : (f + c * g).degree = 3 := sorry

end degree_of_f_plus_cg_l205_205293


namespace intersection_of_A_and_B_l205_205806

open Set

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 9} :=
by
  sorry

end intersection_of_A_and_B_l205_205806


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205080

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205080


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205152

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205152


namespace inequality_sufficient_condition_l205_205058

theorem inequality_sufficient_condition (x : ℝ) (h : 1 < x ∧ x < 2) : 
  (x+1)/(x-1) > 2 :=
by
  sorry

end inequality_sufficient_condition_l205_205058


namespace num_of_permutations_2021_l205_205750

theorem num_of_permutations_2021 : 
  {l : List ℕ // l.length = 4 ∧ l.nodup ∧ l.permutations.count (λ p, p.head ≠ 0) = 9} :=
sorry

end num_of_permutations_2021_l205_205750


namespace units_digit_47_pow_47_l205_205611

theorem units_digit_47_pow_47 :
  let cycle := [7, 9, 3, 1] in
  cycle.nth (47 % 4) = some 3 :=
by
  let cycle := [7, 9, 3, 1]
  have h : 47 % 4 = 3 := by norm_num
  rw h
  simp
  exact trivial

end units_digit_47_pow_47_l205_205611


namespace subcommittees_with_at_least_one_teacher_l205_205580

theorem subcommittees_with_at_least_one_teacher
  (total_members teachers : ℕ)
  (total_members_eq : total_members = 12)
  (teachers_eq : teachers = 5)
  (subcommittee_size : ℕ)
  (subcommittee_size_eq : subcommittee_size = 5) :
  ∃ (n : ℕ), n = 771 :=
by
  sorry

end subcommittees_with_at_least_one_teacher_l205_205580


namespace equation_of_hyperbola_point_P_fixed_line_l205_205371

-- Definitions based on conditions
def center_origin := (0, 0)
def left_focus := (-2 * real.sqrt 5, 0)
def eccentricity := real.sqrt 5
def left_vertex := (-2, 0)
def right_vertex := (2, 0)
def point_passed_through_line := (-4, 0)

-- The equation of the hyperbola that satisfies the given conditions.
theorem equation_of_hyperbola :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ 
    (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ 
      (2 * real.sqrt 5)^2 = a^2 + b^2) :=
by
  use [2, 4]
  split
  exact rfl
  split
  exact rfl
  sorry

-- Proving P lies on the fixed line x = -1 for given lines and vertices conditions
theorem point_P_fixed_line :
  ∀ m : ℝ, ∀ y1 y2 x1 x2 : ℝ, 
    x1 = m * y1 - 4 ∧ x2 = m * y2 - 4 ∧ 
    (y1 + y2 = (32 * m) / ((4 * m^2) - 1)) ∧
    (y1 * y2 = 48 / ((4 * m^2) - 1)) ∧ 
    (x1 + 2) ≠ 0 ∧ 
    (x2 - 2) ≠ 0 → 
    (P : ℝ × ℝ) ((P.1, P.2) = (-1, P.2) ∧ P.2 = (y1 * (x + 2) / x1 ≠ -2 ∧ y2 * (x - 2) / x2 ≠ 2)) := 
by
  intros
  sorry

end equation_of_hyperbola_point_P_fixed_line_l205_205371


namespace intersecting_chords_theorem_l205_205589

theorem intersecting_chords_theorem
  (O : Type*) [euclidean_space O] 
  (A B C D P : O) 
  (h_diameter : ∃ c : circ O, is_diameter (c.center) A B)
  (h_chords_intersect : ∃ (c : circ O), chord c A C ∧ chord c B D ∧ (line A B ∩ line C D) = {P}) :
  dist A B ^ 2 = dist A C * dist A P + dist B D * dist B P :=
by
  sorry

end intersecting_chords_theorem_l205_205589


namespace train_speed_l205_205684

-- Train length (in meters)
def train_length : ℕ := 170

-- Bridge length (in meters)
def bridge_length : ℕ := 205

-- Time to cross bridge (in seconds)
def crossing_time : ℕ := 30

-- Function to calculate total distance
def total_distance (train : ℕ) (bridge : ℕ) : ℕ :=
  train + bridge

-- Function to calculate speed in m/s
def speed_m_s (distance : ℕ) (time : ℕ) : ℝ :=
  distance.to_real / time.to_real

-- Conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Function to convert speed to km/hr
def speed_km_hr (speed : ℝ) : ℝ :=
  speed * conversion_factor

-- Main theorem to prove
theorem train_speed : speed_km_hr (speed_m_s (total_distance train_length bridge_length) crossing_time) = 45 :=
by
  sorry

end train_speed_l205_205684


namespace units_digit_47_pow_47_l205_205609

theorem units_digit_47_pow_47 :
  let cycle := [7, 9, 3, 1] in
  cycle.nth (47 % 4) = some 3 :=
by
  let cycle := [7, 9, 3, 1]
  have h : 47 % 4 = 3 := by norm_num
  rw h
  simp
  exact trivial

end units_digit_47_pow_47_l205_205609


namespace midpoints_coincide_l205_205672

noncomputable theory
open_locale classical

variables {α : Type*} [euclidean_space α]

structure Orthocenter (P Q R : α) : α :=
(orthocenter : α) -- abstracting the orthocenter construction

def midpoint (A B : α) : α := (A + B) / 2

variables (A B C D : α)

def orthocenter_BCD : Orthocenter B C D := sorry
def orthocenter_ACD : Orthocenter A C D := sorry
def orthocenter_ABD : Orthocenter A B D := sorry
def orthocenter_ABC : Orthocenter A B C := sorry

def M_a : α := midpoint A (orthocenter_BCD A B C D).orthocenter
def M_b : α := midpoint B (orthocenter_ACD A B C D).orthocenter
def M_c : α := midpoint C (orthocenter_ABD A B C D).orthocenter
def M_d : α := midpoint D (orthocenter_ABC A B C D).orthocenter

theorem midpoints_coincide : M_a = M_b ∧ M_b = M_c ∧ M_c = M_d :=
by sorry

end midpoints_coincide_l205_205672


namespace units_digit_47_pow_47_l205_205604

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l205_205604


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205133

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205133


namespace simplify_sqrt_expression_l205_205925

theorem simplify_sqrt_expression :
  sqrt (5 * 3) * sqrt (3^4 * 5^2) = 15 * sqrt 15 :=
by sorry

end simplify_sqrt_expression_l205_205925


namespace systematic_sampling_method_l205_205227

def num_rows : ℕ := 50
def num_seats_per_row : ℕ := 30

def is_systematic_sampling (select_interval : ℕ) : Prop :=
  ∀ n, select_interval = n * num_seats_per_row + 8

theorem systematic_sampling_method :
  is_systematic_sampling 30 :=
by
  sorry

end systematic_sampling_method_l205_205227


namespace equation_of_hyperbola_P_lies_on_fixed_line_l205_205356

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- The conditions provided in the problem
def focus := (-2 * Real.sqrt 5, 0)
def center := (0, 0)
def eccentricity : ℝ := Real.sqrt 5
def vertex1 := (-2, 0)
def vertex2 := (2, 0)
def passing_point := (-4, 0)

-- Finding the equation of the hyperbola
theorem equation_of_hyperbola : 
  ∀ (a b c : ℝ), 
  c = 2 * Real.sqrt 5 ∧ 
  eccentricity = Real.sqrt 5 ∧ 
  a = 2 ∧
  b = 4 → 
  hyperbola 2 4 :=
by
  sorry

-- Proving P lies on a fixed line
theorem P_lies_on_fixed_line :
  ∀ (m : ℝ), 
  let line : ℝ → ℝ := λ y, m * y - 4 in
  let intersection_M := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let intersection_N := (x, y) ∈ set_of (λ (x y : ℝ), hyperbola 2 4 x y ∧ x = m * y - 4) in
  let P := (*) -- Intersection point of lines MA1 and NA2 -- (*) 
  ∃ (P : ℝ × ℝ), P.fst = -1 :=
by
  sorry

end equation_of_hyperbola_P_lies_on_fixed_line_l205_205356


namespace total_feet_in_garden_l205_205060

def dogs : ℕ := 6
def ducks : ℕ := 2
def cats : ℕ := 4
def birds : ℕ := 7
def insects : ℕ := 10

def feet_per_dog : ℕ := 4
def feet_per_duck : ℕ := 2
def feet_per_cat : ℕ := 4
def feet_per_bird : ℕ := 2
def feet_per_insect : ℕ := 6

theorem total_feet_in_garden :
  dogs * feet_per_dog + 
  ducks * feet_per_duck + 
  cats * feet_per_cat + 
  birds * feet_per_bird + 
  insects * feet_per_insect = 118 := by
  sorry

end total_feet_in_garden_l205_205060


namespace min_value_of_f_on_interval_l205_205565

noncomputable def f : ℝ → ℝ := λ x, x^2 + 2*x - 1

theorem min_value_of_f_on_interval : 
  ∃ x ∈ set.Icc (0 : ℝ) (3 : ℝ), 
  ∀ y ∈ set.Icc (0 : ℝ) (3 : ℝ), f y ≥ f x ∧ f x = -1 :=
by
  sorry

end min_value_of_f_on_interval_l205_205565


namespace mod_inverse_28_mod_29_l205_205745

theorem mod_inverse_28_mod_29 : ∃ a : ℕ, 28 * a % 29 = 1 ∧ 0 ≤ a ∧ a ≤ 28 ∧ a = 28 :=
by {
  use 28,
  simp,
  sorry
}

end mod_inverse_28_mod_29_l205_205745


namespace max_real_part_sum_l205_205885

def z (j : ℕ) : ℂ := 16 * (Complex.exp (Complex.I * (2 * Real.pi * j / 16)))

theorem max_real_part_sum :
  ∃ (w : Fin 16 → ℂ), (∀ j, w j = z j ∨ w j = -Complex.I * (z j)) ∧ 
    Real.Re (Finset.univ.sum (λ j: Fin 16, w j)) = 16 + 32 * Real.sqrt 2 :=
sorry

end max_real_part_sum_l205_205885


namespace find_b_l205_205428

theorem find_b (b : ℤ) (h : ∃ p : ℤ[X], 9 * X^2 + C b * X + 44 = (C 3 * X + 4) * p) : b = 45 :=
by sorry

end find_b_l205_205428


namespace midline_equation_l205_205836

theorem midline_equation (a b : ℝ) (K1 K2 : ℝ)
  (h1 : K1^2 = (a^2) / 4 + b^2)
  (h2 : K2^2 = a^2 + (b^2) / 4) :
  16 * K2^2 - 4 * K1^2 = 15 * a^2 :=
by
  sorry

end midline_equation_l205_205836


namespace simplify_equation_l205_205529

open Fraction

theorem simplify_equation (x : ℚ) : (1 - (x + 3) / 3 = x / 2) → (6 - 2 * x - 6 = 3 * x) :=
by
  intro h
  sorry

end simplify_equation_l205_205529


namespace maximize_volume_solution_l205_205245

noncomputable def maximize_volume_cutout_length 
  (length : ℝ) (width : ℝ) : ℝ :=
argmax (fun x => (length - 2 * x) * (width - 2 * x) * x) {x : ℝ | 0 < x ∧ x < length / 2 ∧ x < width / 2}

theorem maximize_volume_solution :
  maximize_volume_cutout_length 90 48 = 10 := 
by
  sorry

end maximize_volume_solution_l205_205245


namespace sin2x_half_l205_205426

theorem sin2x_half (x : ℝ) (h : sin x + cos x + tan x + cot x + sec x + csc x = 9) : 
  sin (2 * x) = 1 / 2 := 
sorry

end sin2x_half_l205_205426


namespace next_special_year_after_2009_l205_205961

def is_special_year (n : ℕ) : Prop :=
  ∃ d1 d2 d3 d4 : ℕ,
    (2000 ≤ n) ∧ (n < 10000) ∧
    (d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n) ∧
    (d1 ≠ 0) ∧
    ∀ (p q r s : ℕ),
    (p * 1000 + q * 100 + r * 10 + s < n) →
    (p ≠ d1 ∨ q ≠ d2 ∨ r ≠ d3 ∨ s ≠ d4)

theorem next_special_year_after_2009 : ∃ y : ℕ, is_special_year y ∧ y > 2009 ∧ y = 2022 :=
  sorry

end next_special_year_after_2009_l205_205961


namespace solve_linear_eq_l205_205056

theorem solve_linear_eq (x : ℝ) : (x + 1) / 3 = 0 → x = -1 := 
by 
  sorry

end solve_linear_eq_l205_205056


namespace girls_in_sample_l205_205033

theorem girls_in_sample (total_students boys sample_size : ℕ)
  (H1 : total_students = 54)
  (H2 : boys = 30)
  (H3 : sample_size = 18)
  : (24 / 54) * 18 = 8 := by
suffices girls_ratio : 24 / 54 = 4 / 9 by
suffices sample_girls : (4 / 9) * 18 = 8 by exact sample_girls
calc
  (4 / 9) * 18 = 72 / 9 := by sorry -- Simplify by cancellation
  ... = 8 := by norm_num -- Manual verification of division

end girls_in_sample_l205_205033


namespace sufficient_but_not_necessary_condition_l205_205583

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 / 8) → (∀ x : ℝ, x > 0 → 2 * x + a / x ≥ 1) :=
by sorry

end sufficient_but_not_necessary_condition_l205_205583


namespace equation_of_hyperbola_point_P_on_fixed_line_l205_205379

-- Definition and conditions
def hyperbola_center_origin := (0, 0)
def hyperbola_left_focus := (-2 * Real.sqrt 5, 0)
def hyperbola_eccentricity := Real.sqrt 5
def point_on_line_through_origin := (-4, 0)
def point_M_in_second_quadrant (M : ℝ × ℝ) := M.1 < 0 ∧ M.2 > 0
def vertices_A1_A2 := ((-2, 0), (2, 0))

theorem equation_of_hyperbola (c a b : ℝ) (h_c : c = 2 * Real.sqrt 5)
  (h_e : hyperbola_eccentricity = Real.sqrt 5) (h_a : a = 2) (h_b : b = 4) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 16) = 1) := 
by
  intros x y h_x h_y
  have h : c^2 = a^2 + b^2 := sorry
  have h2 : hyperbola_eccentricity = c / a := sorry
  exact sorry

theorem point_P_on_fixed_line (M N P : ℝ × ℝ) (A1 A2 : ℝ × ℝ)
  (h_vertices : vertices_A1_A2 = (A1, A2))
  (h_M_in_2nd : point_M_in_second_quadrant M) 
  (h_P_intersects : ∀ t : ℝ, P = (t, (A1.2 * A2.1 - A1.1 * A2.2) / (A1.1 - A2.1))) :
  (P.1 = -1) := sorry

end equation_of_hyperbola_point_P_on_fixed_line_l205_205379


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205173

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205173


namespace sally_seashell_profit_l205_205918

theorem sally_seashell_profit : 
  let monday_seashells := 30
  let tuesday_seashells := (1 / 2 : ℝ) * monday_seashells
  let total_seashells := monday_seashells + tuesday_seashells
  let total_money := 54
  in total_money / total_seashells = 1.2 :=
by
  -- Definitions
  let monday_seashells := 30
  let tuesday_seashells := (1 / 2 : ℝ) * monday_seashells
  let total_seashells := monday_seashells + tuesday_seashells
  let total_money := 54
  -- Proof goal
  show total_money / total_seashells = 1.2
  sorry

end sally_seashell_profit_l205_205918


namespace sakshi_work_days_l205_205916

theorem sakshi_work_days (tanya_days : ℕ) (efficiency_increase : ℕ) (efficiency_fraction : ℚ) : 
  efficiency_increase = 25 → tanya_days = 16 → efficiency_fraction = 1.25 → 
  let sakshi_days := tanya_days * efficiency_fraction in 
  sakshi_days = 20 := by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sakshi_work_days_l205_205916


namespace units_digit_47_pow_47_l205_205606

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l205_205606


namespace construct_convex_quadrilateral_l205_205720

noncomputable theory

variables (a b c d m : ℝ)

theorem construct_convex_quadrilateral :
  ∃ (A B C D : ℝ × ℝ), dist A B = a ∧ dist B C = b ∧ dist C D = c ∧ dist D A = d ∧ 
    dist ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ((C.1 + D.1) / 2, (C.2 + D.2) / 2) = m := 
sorry

end construct_convex_quadrilateral_l205_205720


namespace sqrt_two_non_repeating_decimal_l205_205567

theorem sqrt_two_non_repeating_decimal : ¬ (∃ p q : ℕ, q ≠ 0 ∧ nat.coprime p q ∧ (real.sqrt 2 = p / q)) → 
  ∀ r : ℝ, ∃ s t : ℕ, t ≠ 0 ∧ nat.coprime s t ∧ (r = s / t) → false :=
by sorry

end sqrt_two_non_repeating_decimal_l205_205567


namespace midpoint_x_of_parabola_l205_205875

open Real

theorem midpoint_x_of_parabola
  (F : Point) (A B : Point) (P : Point)
  (focus_F : F = (1/2, 0))
  (on_parabola_A : ∃ x₁, A = (x₁, sqrt (2 * x₁)))
  (on_parabola_B : ∃ x₂, B = (x₂, sqrt (2 * x₂)))
  (P_is_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (AF_plus_BF_eq_5 : abs (A.1 - 1 / 2) + abs (B.1 - 1 / 2) = 5) :
  P.1 = 2 :=
by
  sorry

end midpoint_x_of_parabola_l205_205875


namespace number_of_points_in_fourth_quadrant_is_two_l205_205016

def points : List (Int × Int) :=
  [(2, -3), (0, -1), (-2, 0), (2, 3), (-2, -3), (3, -2)]

def in_fourth_quadrant (p : Int × Int) : Bool :=
  p.1 > 0 ∧ p.2 < 0

def count_points_in_fourth_quadrant :=
  List.count
    (λ p => in_fourth_quadrant p)
    points

theorem number_of_points_in_fourth_quadrant_is_two :
  count_points_in_fourth_quadrant = 2 :=
by
  sorry

end number_of_points_in_fourth_quadrant_is_two_l205_205016


namespace hyperbola_center_origin_point_P_on_fixed_line_l205_205355

variables {x y : ℝ}

def hyperbola (h : (x^2 / 4) - (y^2 / 16) = 1) : Prop := true

theorem hyperbola_center_origin 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (left_focus : ∃ x y : ℝ, x = -2 * Real.sqrt 5 ∧ y = 0)
  (eccentricity_root5 : ∃ e : ℝ, e = Real.sqrt 5) :
  hyperbola ((x^2 / 4) - (y^2 / 16) = 1) := 
sorry

theorem point_P_on_fixed_line
  (line_through_point : ∃ x y : ℝ, x = -4 ∧ y = 0)
  (M_second_quadrant : ∃ Mx My : ℝ, Mx < 0 ∧ My > 0)
  (A1_vertex : ∃ x y : ℝ, x = -2 ∧ y = 0)
  (A2_vertex : ∃ x y : ℝ, x = 2 ∧ y = 0)
  (P_intersection : ∃ Px Py : ℝ, Px = -1) :
  Px = -1 :=
sorry

end hyperbola_center_origin_point_P_on_fixed_line_l205_205355


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205086

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205086


namespace find_multiple_l205_205307

theorem find_multiple (x k : ℕ) (hx : x > 0) (h_eq : x + 17 = k * (1/x)) (h_x : x = 3) : k = 60 :=
by
  sorry

end find_multiple_l205_205307


namespace length_of_DE_l205_205540

variable (A B C X Y Z D E : Type)
variable [LinearOrderedField ℝ]

def base_length_ABC : ℝ := 15
def triangle_area_ratio : ℝ := 0.25

theorem length_of_DE (h1 : DE // BC ∥ BC) 
                    (h2 : triangle_area_ratio * (base_length_ABC ^ 2) = DE ^ 2)
                    : DE = 7.5 :=
sorry

end length_of_DE_l205_205540


namespace cyclist_second_part_distance_l205_205230

variables (x : ℝ) -- Distance traveled in the second part of the trip
def time1 := 9 / 12
def time2 := x / 9
def avg_speed_total := 10.08
def total_distance := 9 + x
def total_time := time1 + time2

theorem cyclist_second_part_distance :
  avg_speed_total = total_distance / total_time → x = 12 := by
  intro h
  -- proof steps go here
  sorry

end cyclist_second_part_distance_l205_205230


namespace probability_leq_three_l205_205397

noncomputable def normal_pdf (μ σ : ℝ) : ℝ → ℝ :=
  λ x, (1 / (σ * Real.sqrt (2 * Real.pi)) * Real.exp (-((x - μ)^2) / (2 * σ^2)))

theorem probability_leq_three (ξ : ℝ → ℝ) (hξ : ∀ x, ξ = normal_pdf 4 6) (hprob : P(ξ ≤ 5) = 0.89) :
  P(ξ ≤ 3) = 0.11 :=
sorry

end probability_leq_three_l205_205397


namespace find_fraction_l205_205217

theorem find_fraction : (∃ (x y : ℚ), (377 / 13 / 29) * (x / y) / 2 = 0.125 ∧ x / y = 1 / 4) :=
begin
  sorry
end

end find_fraction_l205_205217


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205136

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205136


namespace find_radius_l205_205250

noncomputable def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Prop := sorry

theorem find_radius (C1 : ℝ × ℝ × ℝ) (r1 : ℝ) (C2 : ℝ × ℝ × ℝ) (r : ℝ) :
  C1 = (3, 5, 0) →
  r1 = 2 →
  C2 = (0, 5, -8) →
  (sphere ((3, 5, -8) : ℝ × ℝ × ℝ) (2 * Real.sqrt 17)) →
  r = Real.sqrt 59 :=
by
  intros h1 h2 h3 h4
  sorry

end find_radius_l205_205250


namespace inscribed_circle_radius_l205_205183

noncomputable def triangle_semiperimeter (DE DF EF : ℝ) : ℝ :=
  (DE + DF + EF) / 2

noncomputable def triangle_area (DE DF EF s : ℝ) : ℝ :=
  Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

theorem inscribed_circle_radius {DE DF EF : ℝ}
  (hDE : DE = 8)
  (hDF : DF = 10)
  (hEF : EF = 14) :
  let s := triangle_semiperimeter DE DF EF in
  let K := triangle_area DE DF EF s in
  let r := K / s in
  r = Real.sqrt 6 :=
by
  sorry

end inscribed_circle_radius_l205_205183


namespace units_digit_47_power_47_l205_205614

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l205_205614


namespace tangent_line_PQ_to_diameter_AB_l205_205815

-- Definitions of points and circles based on the conditions
variables {O₁ O₂ : Type*}
variables (A B C D P Q : Type*)

-- Hypothesis given in the conditions
variables (h₁ : Circles O₁) (h₂ : Circles O₂)
variables (h₃ : IntersectAtPoints O₁ O₂ A B)
variables (h₄ : LineThrough A intersects O₁ At C)
variables (h₅ : LineThrough A intersects O₂ At D)
variables (h₆ : Projection B TangentLineThrough C ToCircle O₁ At P)
variables (h₇ : Projection B TangentLineThrough D ToCircle O₂ At Q)

-- The statement to prove
theorem tangent_line_PQ_to_diameter_AB :
  TangentLine PQ (CircleWithDiameter AB) :=
by
  sorry

end tangent_line_PQ_to_diameter_AB_l205_205815


namespace final_weight_is_sixteen_l205_205868

def initial_weight : ℤ := 0
def weight_after_jellybeans : ℤ := initial_weight + 2
def weight_after_brownies : ℤ := weight_after_jellybeans * 3
def weight_after_more_jellybeans : ℤ := weight_after_brownies + 2
def final_weight : ℤ := weight_after_more_jellybeans * 2

theorem final_weight_is_sixteen : final_weight = 16 := by
  sorry

end final_weight_is_sixteen_l205_205868


namespace sphere_surface_area_l205_205563

theorem sphere_surface_area
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
  (vertices_on_sphere : ∀ x y z : ℝ, x^2 + y^2 + z^2 = a^2 + b^2 + c^2) :
  4 * π * (((sqrt (a^2 + b^2 + c^2)) / 2)^2) = 50 * π :=
by
  sorry

end sphere_surface_area_l205_205563


namespace concurrent_cevians_product_l205_205858

variables {D E F D' E' F' P : Type}
variables {DP PD' EP PE' FP PF' : ℝ}
variables {K_D K_E K_F [DEF] : ℝ}
variables [Nonzero : DP ≠ 0 ∧ PD' ≠ 0 ∧ EP ≠ 0 ∧ PE' ≠ 0 ∧ FP ≠ 0 ∧ PF' ≠ 0]

noncomputable def ratio_sum := DP / PD' + EP / PE' + FP / PF'

theorem concurrent_cevians_product (h1 : DP / PD' + EP / PE' + FP / PF' = 45) :
  DP / PD' * (EP / PE') * (FP / PF') = 47 := 
by
  sorry

end concurrent_cevians_product_l205_205858


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205180

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205180


namespace red_to_green_ratio_l205_205844

theorem red_to_green_ratio (total_flowers green_flowers blue_percentage yellow_flowers : ℕ)
  (h1 : total_flowers = 96)
  (h2 : green_flowers = 9)
  (h3 : blue_percentage = 50)
  (h4 : yellow_flowers = 12) :
  let blue_flowers := (blue_percentage * total_flowers) / 100
  let red_flowers := total_flowers - (green_flowers + blue_flowers + yellow_flowers)
  (red_flowers : ℚ) / green_flowers = 3 := 
by
  sorry

end red_to_green_ratio_l205_205844


namespace remainder_product_mod_5_l205_205955

theorem remainder_product_mod_5 
  (a b c : ℕ) 
  (ha : a % 5 = 1) 
  (hb : b % 5 = 2) 
  (hc : c % 5 = 3) : 
  (a * b * c) % 5 = 1 :=
by
  sorry

end remainder_product_mod_5_l205_205955


namespace infinite_pairs_part_a_l205_205646

noncomputable def prime_divisors (n : ℕ) : set ℕ :=
  {p | p.prime ∧ p ∣ n}

def conditions_a (a b : ℕ) : Prop :=
  prime_divisors a = prime_divisors b ∧ 
  prime_divisors (a + 1) = prime_divisors (b + 1) ∧ 
  a ≠ b

theorem infinite_pairs_part_a : ∃∞ (a b : ℕ), conditions_a a b := sorry

end infinite_pairs_part_a_l205_205646


namespace vertex_of_parabola_l205_205552

theorem vertex_of_parabola (x : ℝ) : 
  let y := 2 * (x - 3)^2 + 1 in 
  ∃ (h k : ℝ), y = 2 * (x - h)^2 + k ∧ h = 3 ∧ k = 1 :=
by
  sorry

end vertex_of_parabola_l205_205552


namespace different_purchasing_methods_l205_205229

noncomputable def purchasing_methods : Nat := 
  ∑ x in finset.range 8,      -- Sum over range (x = 0 to 7)
    if h : x ≥ 3 then           -- Condition x >= 3
      ∑ y in finset.range 8,    -- Sum over range (y = 0 to 7)
        if h₂ : y ≥ 2 ∧ 60 * x + 70 * y ≤ 500 then 1 else 0  -- Condition y >= 2 and 60x + 70y ≤ 500
    else 0

theorem different_purchasing_methods : purchasing_methods = 7 :=
by
  sorry

end different_purchasing_methods_l205_205229


namespace base_of_isosceles_triangle_l205_205201

theorem base_of_isosceles_triangle (p_eq_tri p_iso_tri : ℝ) (s_eq_tri : ℝ) (l_iso_tri b_iso_tri : ℝ)
  (h1 : p_eq_tri = 60)
  (h2 : 3 * s_eq_tri = p_eq_tri)
  (h3 : l_iso_tri = s_eq_tri)
  (h4 : p_iso_tri = 50)
  (h5 : 2 * l_iso_tri + b_iso_tri = p_iso_tri) :
  b_iso_tri = 10 :=
by
  -- Definitions based on the conditions
  have h6 : s_eq_tri = 20 := by linarith
  have h7 : l_iso_tri = 20 := by exact h3
  -- Simplifying the perimeter equation
  have h8 : 2 * 20 + b_iso_tri = 50 := by linarith
  have h9 : b_iso_tri = 10 := by linarith
  exact h9

end base_of_isosceles_triangle_l205_205201


namespace complementary_angle_of_60_l205_205035

theorem complementary_angle_of_60 (a : ℝ) : 
  (∀ (a b : ℝ), a + b = 180 → a = 60 → b = 120) := 
by
  sorry

end complementary_angle_of_60_l205_205035


namespace hyperbola_center_origin_point_P_on_fixed_line_l205_205352

variables {x y : ℝ}

def hyperbola (h : (x^2 / 4) - (y^2 / 16) = 1) : Prop := true

theorem hyperbola_center_origin 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (left_focus : ∃ x y : ℝ, x = -2 * Real.sqrt 5 ∧ y = 0)
  (eccentricity_root5 : ∃ e : ℝ, e = Real.sqrt 5) :
  hyperbola ((x^2 / 4) - (y^2 / 16) = 1) := 
sorry

theorem point_P_on_fixed_line
  (line_through_point : ∃ x y : ℝ, x = -4 ∧ y = 0)
  (M_second_quadrant : ∃ Mx My : ℝ, Mx < 0 ∧ My > 0)
  (A1_vertex : ∃ x y : ℝ, x = -2 ∧ y = 0)
  (A2_vertex : ∃ x y : ℝ, x = 2 ∧ y = 0)
  (P_intersection : ∃ Px Py : ℝ, Px = -1) :
  Px = -1 :=
sorry

end hyperbola_center_origin_point_P_on_fixed_line_l205_205352


namespace find_n_times_s_l205_205488

noncomputable def S : Set ℝ := { x : ℝ | x ≠ 0 }

variable (f : ℝ → ℝ)

axiom f_property1 : ∀ (x : ℝ), x ∈ S → f (1 / x) = 3 * x * f x
axiom f_property2 : ∀ (x y : ℝ), x ∈ S → y ∈ S → x + y ∈ S → f (1 / x) + f (1 / y) = 4 + f (1 / (x + y))

theorem find_n_times_s : 
  let n := {f_val : ℝ | f_val = f 1}.to_finset.card in 
  let s := {f_val : ℝ | f_val = f 1}.to_finset.sum id in 
  n * s = 5 := 
sorry

end find_n_times_s_l205_205488


namespace find_four_digit_number_abcd_exists_l205_205576

theorem find_four_digit_number_abcd_exists (M : ℕ) (H1 : M > 0) (H2 : M % 10 ≠ 0) 
    (H3 : M % 100000 = M^2 % 100000) : ∃ abcd : ℕ, abcd = 2502 :=
by
  -- Proof is omitted
  sorry

end find_four_digit_number_abcd_exists_l205_205576


namespace units_digit_47_pow_47_l205_205619

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l205_205619


namespace find_angle_complement_supplement_l205_205034

theorem find_angle_complement_supplement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end find_angle_complement_supplement_l205_205034


namespace solve_percentage_chromium_first_alloy_l205_205448

noncomputable def percentage_chromium_first_alloy (x : ℝ) : Prop :=
  let w1 := 15 -- weight of the first alloy
  let c2 := 10 -- percentage of chromium in the second alloy
  let w2 := 35 -- weight of the second alloy
  let w_total := 50 -- total weight of the new alloy formed by mixing
  let c_new := 10.6 -- percentage of chromium in the new alloy
  -- chromium percentage equation
  ((x / 100) * w1 + (c2 / 100) * w2) = (c_new / 100) * w_total

theorem solve_percentage_chromium_first_alloy : percentage_chromium_first_alloy 12 :=
  sorry -- proof goes here

end solve_percentage_chromium_first_alloy_l205_205448


namespace problem_1_problem_2_l205_205790

noncomputable def f (x k : ℝ) : ℝ := (2 * k * x) / (x * x + 6 * k)

theorem problem_1 (k m : ℝ) (hk : k > 0)
  (hsol : ∀ x, (f x k) > m ↔ x < -3 ∨ x > -2) :
  ∀ x, 5 * m * x ^ 2 + k * x + 3 > 0 ↔ -1 < x ∧ x < 3 / 2 :=
sorry

theorem problem_2 (k : ℝ) (hk : k > 0)
  (hsol : ∃ (x : ℝ), x > 3 ∧ (f x k) > 1) :
  k > 6 :=
sorry

end problem_1_problem_2_l205_205790


namespace least_pos_int_divisible_by_four_smallest_primes_l205_205076

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l205_205076


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205094

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205094


namespace new_person_weight_l205_205200

theorem new_person_weight (W : ℝ) (N : ℝ) (old_weight : ℝ) (average_increase : ℝ) (num_people : ℕ)
  (h1 : num_people = 8)
  (h2 : old_weight = 45)
  (h3 : average_increase = 6)
  (h4 : (W - old_weight + N) / num_people = W / num_people + average_increase) :
  N = 93 :=
by
  sorry

end new_person_weight_l205_205200


namespace Petya_workout_duration_l205_205002

theorem Petya_workout_duration :
  ∃ x : ℕ, (x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 135) ∧
            (x + 7 > x) ∧
            (x + 14 > x + 7) ∧
            (x + 21 > x + 14) ∧
            (x + 28 > x + 21) ∧
            x = 13 :=
by sorry

end Petya_workout_duration_l205_205002


namespace next_year_property_appears_l205_205960

def no_smaller_rearrangement (n: Nat) : Prop :=
  ∀ (l: List Nat), (l.permutations.map (λ p, p.foldl (λ acc d, acc * 10 + d) 0)).all (λ m, m >= n)

def next_year_with_property (current: Nat) : Nat :=
  if h : current = 2022 then 2022
  else if ∃ n, n > current ∧ no_smaller_rearrangement n then
    Classical.some (Classical.some_spec h)
  else current

theorem next_year_property_appears : next_year_with_property 2009 = 2022 := by
  sorry

end next_year_property_appears_l205_205960


namespace least_positive_integer_divisible_by_four_primes_l205_205122

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205122


namespace simplify_expression_l205_205626

theorem simplify_expression (a b : ℤ) (h_a : a = 43) (h_b : b = 15) :
  (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1290 := 
by
  -- We state the goal that needs to be proven:
  have h_simplified : (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2 * a * b := sorry
  -- Subsequently substituting values for a and b:
  rw [h_a, h_b] at h_simplified
  assumption

end simplify_expression_l205_205626


namespace line_segment_AB_length_l205_205347

noncomputable def length_AB (xA yA xB yB : ℝ) : ℝ :=
  Real.sqrt ((xA - xB)^2 + (yA - yB)^2)

theorem line_segment_AB_length :
  ∀ (xA yA xB yB : ℝ),
    (xA - yA = 0) →
    (xB + yB = 0) →
    (∃ k : ℝ, yA = k * (xA + 1) ∧ yB = k * (xB + 1)) →
    (-1 ≤ xA ∧ xA ≤ 0) →
    (xA + xB = 2 * k ∧ yA + yB = 2 * k) →
    length_AB xA yA xB yB = (4/3) * Real.sqrt 5 :=
by
  intros xA yA xB yB h1 h2 h3 h4 h5
  sorry

end line_segment_AB_length_l205_205347


namespace cos_C_in_triangle_l205_205458

theorem cos_C_in_triangle
  (A B C : ℝ)
  (sin_A : Real.sin A = 4 / 5)
  (cos_B : Real.cos B = 3 / 5) :
  Real.cos C = 7 / 25 :=
sorry

end cos_C_in_triangle_l205_205458


namespace simplify_expression_l205_205625

theorem simplify_expression (a b : ℤ) (h_a : a = 43) (h_b : b = 15) :
  (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1290 := 
by
  -- We state the goal that needs to be proven:
  have h_simplified : (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2 * a * b := sorry
  -- Subsequently substituting values for a and b:
  rw [h_a, h_b] at h_simplified
  assumption

end simplify_expression_l205_205625


namespace calculate_retail_price_l205_205676

/-- Define the wholesale price of the machine. -/
def wholesale_price : ℝ := 90

/-- Define the profit rate as 20% of the wholesale price. -/
def profit_rate : ℝ := 0.20

/-- Define the discount rate as 10% of the retail price. -/
def discount_rate : ℝ := 0.10

/-- Calculate the profit based on the wholesale price. -/
def profit : ℝ := profit_rate * wholesale_price

/-- Calculate the selling price after the discount. -/
def selling_price (retail_price : ℝ) : ℝ := retail_price * (1 - discount_rate)

/-- Calculate the total selling price as the wholesale price plus profit. -/
def total_selling_price : ℝ := wholesale_price + profit

/-- State the theorem we need to prove. -/
theorem calculate_retail_price : ∃ R : ℝ, selling_price R = total_selling_price → R = 120 := by
  sorry

end calculate_retail_price_l205_205676


namespace sarahs_score_l205_205523

theorem sarahs_score (g s : ℕ) (h1 : s = g + 60) (h2 : s + g = 260) : s = 160 :=
sorry

end sarahs_score_l205_205523


namespace tomatoes_grew_in_absence_l205_205446

def initial_tomatoes : ℕ := 36
def multiplier : ℕ := 100
def total_tomatoes_after_vacation : ℕ := initial_tomatoes * multiplier

theorem tomatoes_grew_in_absence : 
  total_tomatoes_after_vacation - initial_tomatoes = 3564 :=
by
  -- skipped proof with 'sorry'
  sorry

end tomatoes_grew_in_absence_l205_205446


namespace find_film_radius_l205_205501

noncomputable def liquid_volume : ℝ := 10 * 5 * 20

def film_thickness : ℝ := 0.2

def film_radius (r : ℝ) : Prop :=
  π * r^2 * film_thickness = liquid_volume

theorem find_film_radius (r : ℝ) (h : film_radius r) : r = Real.sqrt (5000 / π) :=
sorry

end find_film_radius_l205_205501


namespace fourth_sphere_radius_l205_205301

theorem fourth_sphere_radius (R r : ℝ) (h1 : R > 0)
  (h2 : ∀ (a b c d : ℝ × ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    dist a b = 2*R ∧ dist b c = 2*R ∧ dist c d = 2*R ∧ dist d a = R + r ∧
    dist a c = R + r ∧ dist b d = R + r) :
  r = 4*R/3 :=
  sorry

end fourth_sphere_radius_l205_205301


namespace ball_distribution_ways_l205_205822

theorem ball_distribution_ways :
  ∃ (ways : ℕ), ways = 10 ∧
    ∀ (balls boxes : ℕ), 
    balls = 6 ∧ boxes = 4 ∧ 
    (∀ (b : ℕ), b < boxes → b > 0) →
    ways = 10 :=
sorry

end ball_distribution_ways_l205_205822


namespace area_of_triangle_ABC_l205_205240

noncomputable def pointWithinTriangle (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop := sorry
noncomputable def parallelLinesThroughPoint (P : ℝ × ℝ) (A B C : ℝ × ℝ) : (List (ℝ × ℝ)) := sorry
noncomputable def triangleArea (a b c : ℕ) : ℝ := sorry

theorem area_of_triangle_ABC (P A B C : ℝ × ℝ)
  (hP : pointWithinTriangle P A B C)
  (t'1 t'2 t'3 : ℕ)
  (hArea_t'1 : t'1 = 16)
  (hArea_t'2 : t'2 = 25)
  (hArea_t'3 : t'3 = 64)
  : triangleArea A B C = 315 :=
by
  sorry

end area_of_triangle_ABC_l205_205240


namespace integral_sqrt_1_minus_x_squared_plus_x_l205_205715

theorem integral_sqrt_1_minus_x_squared_plus_x :
  ∫ x in -1..1, (Real.sqrt (1 - x^2) + x) = Real.pi / 2 := 
by
  sorry

end integral_sqrt_1_minus_x_squared_plus_x_l205_205715


namespace sum_of_exterior_angles_pentagon_l205_205630

/-- The sum of the exterior angles of any pentagon is 360 degrees. --/
theorem sum_of_exterior_angles_pentagon (p : ℕ) (hp : p = 5) : 
  ∑ i in Finset.range p, 360 = 360 :=
by
  sorry

end sum_of_exterior_angles_pentagon_l205_205630


namespace number_of_correct_statements_l205_205041

theorem number_of_correct_statements :
  (∀ m : ℝ, |m| + m = 0 → m < 0) = false ∧
  (∀ a b : ℝ, |a - b| = b - a → b > a) = false ∧
  (∀ a b : ℤ, a^5 + b^5 = 0 → a + b = 0) = true ∧
  (∀ a b : ℚ, a + b = 0 → a / b = -1) = false ∧
  (∀ a b c : ℚ, |a| / a + |b| / b + |c| / c = 1 → |abc| / abc = -1) = true →
  (2 : ℕ) :=
by sorry

end number_of_correct_statements_l205_205041


namespace largest_average_multiples_l205_205631

-- Define the sets of multiples and their bounds
def multiples (a : ℕ) (n : ℕ) : set ℕ := {m | m % a = 0 ∧ m ≤ n}

-- Define the conditions from the problem
def multiples2 := multiples 2 201
def multiples3 := multiples 3 201
def multiples4 := multiples 4 201
def multiples5 := multiples 5 201
def multiples7 := multiples 7 201

-- Function to calculate the average of a set of multiples
noncomputable def average (s : set ℕ) : ℚ :=
  if h : s.nonempty then
    (s.to_finset.sum id : ℚ) / s.to_finset.card
  else 0

-- Prove the set of multiples of 5 has the largest average
theorem largest_average_multiples :
  average multiples5 = 102.5 ∧
  average multiples5 ≥ average multiples2 ∧
  average multiples5 ≥ average multiples3 ∧
  average multiples5 ≥ average multiples4 ∧
  average multiples5 ≥ average multiples7 :=
by
  sorry

end largest_average_multiples_l205_205631


namespace count_of_two_digit_numbers_with_digits_sum_of_nine_l205_205821

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_sum_is_nine (n : ℕ) : Prop :=
  let (a, b) := (n / 10, n % 10) in
  a + b = 9

theorem count_of_two_digit_numbers_with_digits_sum_of_nine :
  {n : ℕ // is_two_digit_number n ∧ digits_sum_is_nine n}.to_finset.card = 8 :=
by sorry

end count_of_two_digit_numbers_with_digits_sum_of_nine_l205_205821


namespace isosceles_triangles_similar_l205_205632

theorem isosceles_triangles_similar (a b c : ℝ) (h2 : 0 < a ∧ 0 < b ∧ 0 < c)
(h3 : a = b) (h4 : b = 2 * c) : 
  ∀ (a2 b2 c2 : ℝ), 
  (0 < a2 ∧ 0 < b2 ∧ 0 < c2) → 
  (a2 = b2) → 
  (b2 = 2 * c2) → 
  similar_triangles (triangle.mk a b c) (triangle.mk a2 b2 c2) :=
sorry

end isosceles_triangles_similar_l205_205632


namespace t_shirt_cost_is_correct_l205_205508

structure ShoppingConditions (nat : Type) :=
  (total_money : nat)
  (money_left : nat)
  (cost_jumper : nat)
  (cost_heels : nat)
  (money_spent : nat := total_money - money_left) -- Derived condition

noncomputable def cost_tshirt (c : ShoppingConditions ℕ) : ℕ :=
  c.money_spent - c.cost_jumper - c.cost_heels

theorem t_shirt_cost_is_correct : 
  let c := ShoppingConditions.mk 26 8 9 5 
  in cost_tshirt c = 4 := 
by {
  let c := ShoppingConditions.mk 26 8 9 5,
  show cost_tshirt c = 4,
  sorry
}

end t_shirt_cost_is_correct_l205_205508


namespace cannot_transform_to_2011_l205_205671

theorem cannot_transform_to_2011 (a : ℕ) (h : Composite a) : 
  (∀ q : ℕ, ProperDivisor q a → q ≠ 2011) := by
  sorry

end cannot_transform_to_2011_l205_205671


namespace perpendicular_bisectors_triangle_l205_205935

open EuclideanGeometry

theorem perpendicular_bisectors_triangle
  {A B C D E K O : Point}
  (triangle_ABC : Triangle A B C)
  (is_bisector_CE : AngleBisector C E A B)
  (is_bisector_AD : AngleBisector A D B C)
  (intersect_O : LineIntersection (LineSegment A D) (LineSegment C E) O)
  (sym_AB_CE : SymmetricAbout (LineSegment C E) (LineSegment A B))
  (sym_BC_AD : SymmetricAbout (LineSegment A D) (LineSegment B C))
  (intersect_K : LineIntersection (SymmetricLine (LineSegment A B) (LineSegment C E)) (SymmetricLine (LineSegment B C) (LineSegment A D)) K) :
  Perpendicular (LineSegment K O) (LineSegment A C) := sorry

end perpendicular_bisectors_triangle_l205_205935


namespace largest_integer_in_list_l205_205234

theorem largest_integer_in_list : ∃ (l : List ℕ), 
  List.length l = 5 ∧ 
  (∀ x ∈ l, 0 < x) ∧ 
  (∃ n : ℕ, List.count 7 l = n ∧ n ≥ 2) ∧ 
  (l.nth (l.length / 2) = some 10) ∧ 
  (list_sum l = 60) ∧ 
  List.maximum l = some 25 :=
by 
  sorry

end largest_integer_in_list_l205_205234


namespace bill_initial_amount_l205_205699

/-- Suppose Ann has $777 and Bill gives Ann $167,
    after which they both have the same amount of money. 
    Prove that Bill initially had $1111. -/
theorem bill_initial_amount (A B : ℕ) (h₁ : A = 777) (h₂ : B - 167 = A + 167) : B = 1111 :=
by
  -- Proof goes here
  sorry

end bill_initial_amount_l205_205699


namespace selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l205_205934

-- Problem conditions
def cost_price : ℕ := 70
def max_price : ℕ := 99
def initial_price : ℕ := 110
def initial_sales : ℕ := 20
def price_drop_rate : ℕ := 1
def sales_increase_rate : ℕ := 2
def sales_increase_per_yuan : ℕ := 2
def profit_target : ℕ := 1200

-- Selling price for given sales volume
def selling_price_for_sales_volume (sales_volume : ℕ) : ℕ :=
  initial_price - (sales_volume - initial_sales) / sales_increase_per_yuan

-- Functional relationship between sales volume (y) and price (x)
def sales_volume_function (x : ℕ) : ℕ :=
  initial_sales + sales_increase_rate * (initial_price - x)

-- Profit for given price and resulting sales volume
def daily_profit (x : ℕ) : ℤ :=
  (x - cost_price) * (sales_volume_function x)

-- Part 1: Selling price for 30 items sold
theorem selling_price_30_items : selling_price_for_sales_volume 30 = 105 :=
by
  sorry

-- Part 2: Functional relationship between sales volume and selling price
theorem sales_volume_functional_relationship (x : ℕ) (hx : 70 ≤ x ∧ x ≤ 99) :
  sales_volume_function x = 240 - 2 * x :=
by
  sorry

-- Part 3: Selling price for a daily profit of 1200 yuan
theorem selling_price_for_1200_profit {x : ℕ} (hx : 70 ≤ x ∧ x ≤ 99) :
  daily_profit x = 1200 → x = 90 :=
by
  sorry

end selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l205_205934


namespace polygon_interior_angle_sum_l205_205036

theorem polygon_interior_angle_sum (n : ℕ) (h : 180 * (n - 2) = 2340) :
    180 * ((n + 4) - 2) = 3060 :=
begin
  sorry
end

end polygon_interior_angle_sum_l205_205036


namespace calculate_f_at_2_l205_205276

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_at_2 :
  (∀ x : ℝ, 25 * f (x / 1580) + (3 - Real.sqrt 34) * f (1580 / x) = 2017 * x) →
  f 2 = 265572 :=
by
  intro h
  sorry

end calculate_f_at_2_l205_205276


namespace train_speed_in_kmph_l205_205255

noncomputable def motorbike_speed : ℝ := 64
noncomputable def overtaking_time : ℝ := 40
noncomputable def train_length_meters : ℝ := 400.032

theorem train_speed_in_kmph :
  let train_length_km := train_length_meters / 1000
  let overtaking_time_hours := overtaking_time / 3600
  let relative_speed := train_length_km / overtaking_time_hours
  let train_speed := motorbike_speed + relative_speed
  train_speed = 100.00288 := by
  sorry

end train_speed_in_kmph_l205_205255


namespace min_value_y_l205_205313

noncomputable def y (x : ℝ) := x^4 - 4*x + 3

theorem min_value_y : ∃ x ∈ Set.Icc (-2 : ℝ) 3, y x = 0 ∧ ∀ x' ∈ Set.Icc (-2 : ℝ) 3, y x' ≥ 0 :=
by
  sorry

end min_value_y_l205_205313


namespace line_parallel_condition_l205_205210

theorem line_parallel_condition (a : ℝ) : (a = 2) ↔ (∀ x y : ℝ, (ax + 2 * y = 0 → x + y ≠ 1)) :=
by
  sorry

end line_parallel_condition_l205_205210


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205153

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205153


namespace find_hyperbola_fixed_line_through_P_l205_205389

-- Define the conditions of the problem
def center_origin (C : Type) : Prop := 
  C.center = (0, 0)

def left_focus_at (C : Type) : Prop :=
  C.left_focus = (-2 * Real.sqrt 5, 0)

def eccentricity_sqrt_5 (C : Type) : Prop :=
  C.eccentricity = Real.sqrt 5

-- Define the specific hyperbola equation based on the conditions
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 16) = 1

-- State the results to prove
theorem find_hyperbola (C : Type) :
  center_origin C →
  left_focus_at C →
  eccentricity_sqrt_5 C →
  ∀ x y : ℝ, hyperbola_equation x y :=
sorry

-- Define the fixed line for point P
def fixed_line (P : Type) : Prop :=
  P.x = -1

-- State the results to prove for point P
theorem fixed_line_through_P (M N A1 A2 P : Type) (x y : ℝ) :
  (M.line_passing_through (Point.mk (-4, 0))).intersects_left_branch (N : Point) →
  M.in_second_quadrant →
  (A1.coords = (-2, 0)) ∧ (A2.coords = (2, 0)) →
  Point.mk x y = P →
  fixed_line P :=
sorry

end find_hyperbola_fixed_line_through_P_l205_205389


namespace find_hyperbola_fixed_line_through_P_l205_205393

-- Define the conditions of the problem
def center_origin (C : Type) : Prop := 
  C.center = (0, 0)

def left_focus_at (C : Type) : Prop :=
  C.left_focus = (-2 * Real.sqrt 5, 0)

def eccentricity_sqrt_5 (C : Type) : Prop :=
  C.eccentricity = Real.sqrt 5

-- Define the specific hyperbola equation based on the conditions
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 16) = 1

-- State the results to prove
theorem find_hyperbola (C : Type) :
  center_origin C →
  left_focus_at C →
  eccentricity_sqrt_5 C →
  ∀ x y : ℝ, hyperbola_equation x y :=
sorry

-- Define the fixed line for point P
def fixed_line (P : Type) : Prop :=
  P.x = -1

-- State the results to prove for point P
theorem fixed_line_through_P (M N A1 A2 P : Type) (x y : ℝ) :
  (M.line_passing_through (Point.mk (-4, 0))).intersects_left_branch (N : Point) →
  M.in_second_quadrant →
  (A1.coords = (-2, 0)) ∧ (A2.coords = (2, 0)) →
  Point.mk x y = P →
  fixed_line P :=
sorry

end find_hyperbola_fixed_line_through_P_l205_205393


namespace OC_perpendicular_AB_OABC_coplanar_l205_205773

-- Definition of the given points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Given points O, A, B, C
def O := Point3D.mk 0 0 0
def A := Point3D.mk -2 2 -2
def B := Point3D.mk 1 4 -6
def C (x : ℝ) := Point3D.mk x (-8) 8

-- Define the vectors
def vector (p1 p2 : Point3D) : Point3D := 
  Point3D.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)

def dotProduct (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Proof problems
theorem OC_perpendicular_AB (x : ℝ) : dotProduct (vector O (C x)) (vector A B) = 0 → x = 16 := 
  by 
  sorry

theorem OABC_coplanar (x : ℝ) : ∃ λ μ : ℝ, 
  C x = ⟨λ * -2 + μ * 1, λ * 2 + μ * 4, λ * -2 + μ * -6⟩ → x = 8 :=
  by 
  sorry

end OC_perpendicular_AB_OABC_coplanar_l205_205773


namespace find_a_plus_b_l205_205043

theorem find_a_plus_b (a b : ℝ) 
  (h1 : ∀ x, x ∈ Ioo (-1 : ℝ) 2 → ((ax - 2) * (x + b)) > 0)
  (h2 : ∀ x, x ∉ Ioo (-1 : ℝ) 2 → ((ax - 2) * (x + b)) ≤ 0) :
  a + b = -4 := 
sorry

end find_a_plus_b_l205_205043


namespace least_pos_int_div_by_four_distinct_primes_l205_205170

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205170


namespace sum_of_squares_difference_l205_205709

theorem sum_of_squares_difference : 
  let even_squares_sum := ∑ n in Finset.range 100, (2 * (n + 1))^2
  let odd_squares_sum := ∑ n in Finset.range 100, (2 * (n + 1) - 1)^2 
  even_squares_sum - odd_squares_sum = 20000 :=
by
  sorry

end sum_of_squares_difference_l205_205709


namespace count_multiples_of_5_in_four_digit_range_l205_205422

noncomputable theory
open_locale big_operators

theorem count_multiples_of_5_in_four_digit_range : 
  let first := 1000 in
  let last := 9999 in
  let step := 5 in
  let count := (last - first) / step + 1 in
  count = 1800 :=
by
  sorry

end count_multiples_of_5_in_four_digit_range_l205_205422


namespace hyperbola_and_fixed_line_proof_l205_205381

-- Definitions based on conditions
def center_at_origin (C : Type) [Hyperbola C] : Prop :=
  C.center = (0, 0)

def left_focus (C : Type) [Hyperbola C] : Prop :=
  C.focus1 = (-2 * Real.sqrt 5, 0)

def eccentricity (C : Type) [Hyperbola C] : Prop :=
  C.eccentricity = Real.sqrt 5

def vertices (C : Type) [Hyperbola C] : Prop :=
  C.vertex1 = (-2, 0) ∧ C.vertex2 = (2, 0)

def passing_line_intersections (C : Type) (M N : Type) [Hyperbola C] : Prop :=
  ∃ l : Line, l ⨯ (-4, 0) ∧ l.intersects_hyperbola_left_branch C = {M, N}

def intersection (M N A1 A2 P : Type) [Hyperbola C] : Prop :=
  ∃ A1 A2, A1 = (-2, 0) ∧ A2 = (2, 0) ∧ (MA1.M ∩ NA2.N = P)

-- Proof problem statement
theorem hyperbola_and_fixed_line_proof (C : Type) [Hyperbola C] (M N P : Type)
  (h1 : center_at_origin C) (h2 : left_focus C) (h3 : eccentricity C)
  (h4 : passing_line_intersections C M N) (h5 : vertices C) (h6 : intersection M N C.vertex1 C.vertex2 P) : 
  (C.equation = (∀ x y, C.vertical_transverse_axis x y (x^2 / 4 - y^2 / 16)) ∧ 
   ∀ (x : ℝ), P.x = -1) :=
by sorry

end hyperbola_and_fixed_line_proof_l205_205381


namespace hexahedron_has_six_faces_l205_205819

-- Definition based on the condition
def is_hexahedron (P : Type) := 
  ∃ (f : P → ℕ), ∀ (x : P), f x = 6

-- Theorem statement based on the question and correct answer
theorem hexahedron_has_six_faces (P : Type) (h : is_hexahedron P) : 
  ∀ (x : P), ∃ (f : P → ℕ), f x = 6 :=
by 
  sorry

end hexahedron_has_six_faces_l205_205819


namespace find_c_l205_205719

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12 * x + 3 * x^2 - 4 * x^3 + 5 * x^4
def g (x : ℝ) : ℝ := 2 - 4 * x - 7 * x^3 + 8 * x^4

-- Define the condition that f(x) + c * g(x) has degree 2
def polynomial_degree (p : ℝ → ℝ) : ℕ := sorry -- Assume some mechanism to calculate polynomial degree

theorem find_c :
  ∃ (c : ℝ), c = -5 / 8 ∧ polynomial_degree (λ x, f x + c * g x) = 2 :=
sorry

end find_c_l205_205719


namespace expand_and_simplify_l205_205739

theorem expand_and_simplify (x : ℝ) : 
  -2 * (4 * x^3 - 5 * x^2 + 3 * x - 7) = -8 * x^3 + 10 * x^2 - 6 * x + 14 :=
sorry

end expand_and_simplify_l205_205739


namespace factor_x_minus_1_l205_205651

variable {R : Type*} [CommRing R]

-- Assuming polynomials over a commutative ring R
variables (P Q R S : R[X])

theorem factor_x_minus_1 (h : P.comp (X ^ 5) + X * Q.comp (X ^ 5) + (X ^ 2) * R.comp (X ^ 5) = (X ^ 4 + X ^ 3 + X ^ 2 + X + 1) * S) : 
  (X - 1) ∣ P :=
by
  sorry

end factor_x_minus_1_l205_205651


namespace smallest_pos_int_div_by_four_primes_l205_205115

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l205_205115


namespace president_and_committee_l205_205447

theorem president_and_committee (n : ℕ) (h : n = 8) : 
  let total_ways := (8 : ℕ) * ((7 : ℕ) * (6 : ℕ) / 2) in
  total_ways = 168 :=
by
  sorry

end president_and_committee_l205_205447


namespace determine_a_l205_205492

theorem determine_a (a : ℝ) (h : ∃ r : ℝ, (a / (1+1*I : ℂ) + (1+1*I : ℂ) / 2).im = 0) : a = 1 :=
sorry

end determine_a_l205_205492


namespace new_cost_after_decrease_l205_205259

theorem new_cost_after_decrease (C new_C : ℝ) (hC : C = 1100) (h_decrease : new_C = 0.76 * C) : new_C = 836 :=
-- To be proved based on the given conditions
sorry

end new_cost_after_decrease_l205_205259


namespace area_of_trajectory_area_of_figure_l205_205812

-- Define the points A and B
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk (-2) 0
def B := Point.mk 1 0

-- Define the condition that |PA| = 2|PB|
def satisfies_condition (P : Point) : Prop :=
  real.sqrt ((P.x + 2)^2 + P.y^2) = 2 * real.sqrt ((P.x - 1)^2 + P.y^2)

-- Define the trajectory of P as a circle centered at (2, 0) with a radius of 2
def trajectory : set Point := {P : Point | (P.x - 2)^2 + P.y^2 = 4}

-- Define the area of the figure enclosed by the trajectory
def figure_area : ℝ := real.pi * 2^2

theorem area_of_trajectory :
  ∀ (P : Point), satisfies_condition P ↔ P ∈ trajectory :=
by sorry

theorem area_of_figure :
  (∀ (P : Point), satisfies_condition P → P ∈ trajectory) →
  figure_area = 4 * real.pi :=
by sorry

end area_of_trajectory_area_of_figure_l205_205812


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205182

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205182


namespace total_diameter_of_three_circles_l205_205696

theorem total_diameter_of_three_circles (C1 C2 C3 π : ℝ) 
  (hC1 : C1 = 6.28) 
  (hC2 : C2 = 10.47) 
  (hC3 : C3 = 18.85) 
  (hπ : π = 3.14) : 
  (C1 / π) + (C2 / π) + (C3 / π) = 11.33 :=
by
  -- These enable us to use the relations given in the problem
  have D1 := C1 / π,
  have D2 := C2 / π,
  have D3 := C3 / π,
  
  -- Proving the main statement using the derived diameters
  calc
    D1 + D2 + D3 = 6.28 / 3.14 + 10.47 / 3.14 + 18.85 / 3.14 : by rw [D1, D2, D3]
             ... = 2 + 3.33 + 6                                : by norm_num
             ... = 11.33                                       : by norm_num

end total_diameter_of_three_circles_l205_205696


namespace sophie_total_spending_l205_205534

-- Definitions based on conditions
def num_cupcakes : ℕ := 5
def price_per_cupcake : ℝ := 2
def num_doughnuts : ℕ := 6
def price_per_doughnut : ℝ := 1
def num_slices_apple_pie : ℕ := 4
def price_per_slice_apple_pie : ℝ := 2
def num_cookies : ℕ := 15
def price_per_cookie : ℝ := 0.60

-- Total cost calculation
def total_cost : ℝ :=
  num_cupcakes * price_per_cupcake +
  num_doughnuts * price_per_doughnut +
  num_slices_apple_pie * price_per_slice_apple_pie +
  num_cookies * price_per_cookie

-- Theorem stating the total cost is 33
theorem sophie_total_spending : total_cost = 33 := by
  sorry

end sophie_total_spending_l205_205534


namespace number_of_sandwiches_l205_205524

-- Define the constants and assumptions

def soda_cost : ℤ := 1
def number_of_sodas : ℤ := 3
def cost_of_sodas : ℤ := number_of_sodas * soda_cost

def number_of_soups : ℤ := 2
def soup_cost : ℤ := cost_of_sodas
def cost_of_soups : ℤ := number_of_soups * soup_cost

def sandwich_cost : ℤ := 3 * soup_cost
def total_cost : ℤ := 18

-- The mathematical statement we want to prove
theorem number_of_sandwiches :
  ∃ n : ℤ, (n * sandwich_cost + cost_of_sodas + cost_of_soups = total_cost) ∧ n = 1 :=
by
  sorry

end number_of_sandwiches_l205_205524


namespace no_six_digit_number_divisible_by_30_l205_205969

/-
  The claim is to show that forming a six-digit integer with the digits 2, 3, 3, 6, 0, and 5
  will result in no number that is divisible by 30.
-/

theorem no_six_digit_number_divisible_by_30 :
  ¬ ∃ n : ℕ, (n ∈ { n | nat.digits 10 n = [2, 3, 3, 6, 0, 5].perm }) ∧ (30 ∣ n) :=
begin
  sorry
end

end no_six_digit_number_divisible_by_30_l205_205969


namespace geometric_sequence_a6_l205_205340

theorem geometric_sequence_a6 (a : ℕ → ℝ) 
  (h1 : a 4 * a 8 = 9) 
  (h2 : a 4 + a 8 = 8) 
  (geom_seq : ∀ n m, a (n + m) = a n * a m): 
  a 6 = 3 :=
by
  -- skipped proof
  sorry

end geometric_sequence_a6_l205_205340


namespace find_area_AGKIJEFB_l205_205968

noncomputable def hexagon_polygon_area : Real :=
  36 + Real.sqrt 6

theorem find_area_AGKIJEFB
  (hex1 hex2 : Hexagon)
  (A B C D E F G H I J K : Point)
  (h1 : hex1.A = A)
  (h2 : hex1.B = B)
  (h3 : hex1.C = C)
  (h4 : hex1.D = D)
  (h5 : hex1.E = E)
  (h6 : hex1.F = F)
  (h7 : hex2.G = G)
  (h8 : hex2.H = H)
  (h9 : hex2.I = I)
  (h10 : hex2.J = J)
  (h11 : hex2.E = E)
  (h12 : hex2.F = F)
  (h13 : hex1.area = 36)
  (h14 : hex2.area = 36)
  (h15 : K ∈ Line_through A B)
  (h16 : segment_ratio A K K B 1 2)
  (h17 : midpoint K G H) :
  area (Polygon.mk [A, G, K, I, J, E, F, B]) = hexagon_polygon_area := 
by 
  sorry  -- proof goes here

end find_area_AGKIJEFB_l205_205968


namespace least_pos_int_div_by_four_distinct_primes_l205_205100

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205100


namespace sphere_intersection_radius_l205_205248

theorem sphere_intersection_radius (r : ℝ) :
  (let center := (3 : ℝ, 5, -8),
       xy_center := (3, 5, 0),
       yz_center := (0, 5, -8),
       xy_radius := 2,
       sphere_radius := real.sqrt (2^2 + (xy_center.3 - center.3)^2) in
   (real.sqrt (r^2 + (yz_center.1 - center.1)^2) = sphere_radius)) :=
by
  let center := (3 : ℝ, 5, -8),
      xy_center := (3, 5, 0),
      yz_center := (0, 5, -8),
      xy_radius := 2,
      sphere_radius := real.sqrt (2^2 + (xy_center.3 - center.3)^2)
  use (real.sqrt 59)
  sorry

end sphere_intersection_radius_l205_205248


namespace impossible_to_place_numbers_l205_205464

noncomputable def problem_statement : Prop :=
∀ (table : list (list ℕ)),
  (∀ row ∈ table, row.length = 7) ∧   -- Each row has 7 columns
  table.length = 6 ∧                  -- The table has 6 rows
  (∀ i j, 1 ≤ i ∧ i < 6 → 1 ≤ j ∧ j ≤ 7 → table.nth_le i j + table.nth_le (i+1) j % 2 = 0) → -- Each 1x2 vertical rectangle sum is even
  false                               -- Conclusion: It's impossible (contradiction)

theorem impossible_to_place_numbers : problem_statement :=
sorry

end impossible_to_place_numbers_l205_205464


namespace find_y_90_l205_205879

def bowtie (a b : ℝ) : ℝ := a + (Real.sqrt (b + (Real.sqrt (b + ... -- definition needs a fix for an infinite nested sqrt
-- However, we can define as a fixed point:
def bowtie (a b : ℝ) : ℝ := a + (Real.sqrt (b + Real.sqrt b))

theorem find_y_90 (y : ℝ) (h : bowtie 5 y = 15) : y = 90 := by
  sorry

end find_y_90_l205_205879


namespace smallest_difference_of_9_digit_permutations_l205_205975

def is_permutation_of_1_to_9 (n : ℕ) : Prop :=
  ∃ l : List ℕ, l.perm [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ n = (l.foldl (λ acc x, acc * 10 + x) 0)

theorem smallest_difference_of_9_digit_permutations :
  ∀ x y : ℕ, is_permutation_of_1_to_9 x → is_permutation_of_1_to_9 y → x ≠ y → ∃ k, x - y = 9 * k :=
by
  sorry

end smallest_difference_of_9_digit_permutations_l205_205975


namespace dog_ate_cost_is_six_l205_205478

-- Definitions for the costs
def flour_cost : ℝ := 4
def sugar_cost : ℝ := 2
def butter_cost : ℝ := 2.5
def eggs_cost : ℝ := 0.5

-- Total cost calculation
def total_cost := flour_cost + sugar_cost + butter_cost + eggs_cost

-- Initial slices and remaining slices
def initial_slices : ℕ := 6
def eaten_slices : ℕ := 2
def remaining_slices := initial_slices - eaten_slices

-- The cost calculation of the amount the dog ate
def dog_ate_cost := (remaining_slices / initial_slices) * total_cost

-- Proof statement
theorem dog_ate_cost_is_six : dog_ate_cost = 6 :=
by
  sorry

end dog_ate_cost_is_six_l205_205478


namespace units_digit_47_pow_47_l205_205617

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l205_205617


namespace sum_smallest_and_largest_l205_205937

theorem sum_smallest_and_largest (x b : ℤ) (m : ℕ) (h : m % 2 = 0) :
  (x = (b * m + (2 * (1 + 2 + ... + (m-1))) / m)) → 
  (2 * x = b + (b + 2 * (m - 1))) :=
by sorry

end sum_smallest_and_largest_l205_205937


namespace carl_garden_area_l205_205711

theorem carl_garden_area (total_posts : ℕ) (post_interval : ℕ) (x_posts_on_shorter : ℕ) (y_posts_on_longer : ℕ)
  (h1 : total_posts = 26)
  (h2 : post_interval = 5)
  (h3 : y_posts_on_longer = 2 * x_posts_on_shorter)
  (h4 : 2 * x_posts_on_shorter + 2 * y_posts_on_longer - 4 = total_posts) :
  (x_posts_on_shorter - 1) * post_interval * (y_posts_on_longer - 1) * post_interval = 900 := 
by
  sorry

end carl_garden_area_l205_205711


namespace hamburgers_served_l205_205674

-- Definitions for the conditions
def hamburgers_made : ℕ := 9
def hamburgers_left_over : ℕ := 6

-- The main statement to prove
theorem hamburgers_served : hamburgers_made - hamburgers_left_over = 3 := by
  sorry

end hamburgers_served_l205_205674


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205130

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205130


namespace arithmetic_mean_sqrt2_plus1_sqrt2_minus1_l205_205936

theorem arithmetic_mean_sqrt2_plus1_sqrt2_minus1 : ( (sqrt 2 + 1) + (sqrt 2 - 1)) / 2 = sqrt 2 := 
by sorry

end arithmetic_mean_sqrt2_plus1_sqrt2_minus1_l205_205936


namespace vertices_on_circle_l205_205252

theorem vertices_on_circle 
  {n : ℕ} (pyramid : Type) [IsPyramid pyramid n]
  (sphere : Type) [IsInscribedSphere sphere pyramid]
  (rotate_lateral_faces : ∀ f : Face pyramid, Face plane) :
  (∃ c : Point, ∀ v : Vertex (rotate_lateral_faces f), distance c v = r) := sorry

end vertices_on_circle_l205_205252


namespace justin_more_pencils_l205_205864

variable (total_pencils justin_pencils sabrina_pencils twice_sabrina more_pencils : ℕ)

theorem justin_more_pencils :
  total_pencils = 50 →
  sabrina_pencils = 14 →
  justin_pencils = total_pencils - sabrina_pencils →
  twice_sabrina = 2 * sabrina_pencils →
  more_pencils = justin_pencils - twice_sabrina →
  more_pencils = 8 :=
by
  intros h_total h_sabrina h_justin h_twice h_more
  rw [h_total, h_sabrina] at *
  rw h_justin
  rw [←h_more, eq_comm]
  sorry

end justin_more_pencils_l205_205864


namespace square_difference_l205_205808

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x + 1) * (x - 1) = 9800 :=
by {
  sorry
}

end square_difference_l205_205808


namespace hyperbola_and_fixed_line_proof_l205_205382

-- Definitions based on conditions
def center_at_origin (C : Type) [Hyperbola C] : Prop :=
  C.center = (0, 0)

def left_focus (C : Type) [Hyperbola C] : Prop :=
  C.focus1 = (-2 * Real.sqrt 5, 0)

def eccentricity (C : Type) [Hyperbola C] : Prop :=
  C.eccentricity = Real.sqrt 5

def vertices (C : Type) [Hyperbola C] : Prop :=
  C.vertex1 = (-2, 0) ∧ C.vertex2 = (2, 0)

def passing_line_intersections (C : Type) (M N : Type) [Hyperbola C] : Prop :=
  ∃ l : Line, l ⨯ (-4, 0) ∧ l.intersects_hyperbola_left_branch C = {M, N}

def intersection (M N A1 A2 P : Type) [Hyperbola C] : Prop :=
  ∃ A1 A2, A1 = (-2, 0) ∧ A2 = (2, 0) ∧ (MA1.M ∩ NA2.N = P)

-- Proof problem statement
theorem hyperbola_and_fixed_line_proof (C : Type) [Hyperbola C] (M N P : Type)
  (h1 : center_at_origin C) (h2 : left_focus C) (h3 : eccentricity C)
  (h4 : passing_line_intersections C M N) (h5 : vertices C) (h6 : intersection M N C.vertex1 C.vertex2 P) : 
  (C.equation = (∀ x y, C.vertical_transverse_axis x y (x^2 / 4 - y^2 / 16)) ∧ 
   ∀ (x : ℝ), P.x = -1) :=
by sorry

end hyperbola_and_fixed_line_proof_l205_205382


namespace find_vertex_angle_of_identical_cones_l205_205967

def vertex_angle_cones (A : Point) (P : Plane) (C1 C2 C3 : Cone) : ℝ :=
  if C1.vertex = A ∧ C2.vertex = A ∧ C3.vertex = A ∧ C3.angle = π / 2 ∧ is_tangent_to_plane C1 P ∧ is_tangent_to_plane C2 P ∧ is_tangent_to_plane C3 P then
    2 * arctan (4 / 5)
  else
    0

theorem find_vertex_angle_of_identical_cones (A : Point) (P : Plane) (C1 C2 C3 : Cone) (h1 : C1.vertex = A) (h2 : C2.vertex = A) (h3 : C3.vertex = A) (h4 : C3.angle = π / 2) (h5: is_tangent_to_plane C1 P) (h6 : is_tangent_to_plane C2 P) (h7 : is_tangent_to_plane C3 P) : 
  vertex_angle_cones A P C1 C2 C3 = 2 * arctan (4 / 5) :=
sorry

end find_vertex_angle_of_identical_cones_l205_205967


namespace mean_median_mode_equal_x_l205_205045

theorem mean_median_mode_equal_x {x : ℕ} :
  let s := {12, 9, 11, 16, x}
  in (mean s = median s) ∧ (median s = mode s) → x = 12 :=
by
  sorry

end mean_median_mode_equal_x_l205_205045


namespace basis_vetors_correct_options_l205_205982

def is_basis (e1 e2 : ℝ × ℝ) : Prop :=
  e1 ≠ (0, 0) ∧ e2 ≠ (0, 0) ∧ e1.1 * e2.2 - e1.2 * e2.1 ≠ 0

def option_A : ℝ × ℝ := (0, 0)
def option_A' : ℝ × ℝ := (1, 2)

def option_B : ℝ × ℝ := (2, -1)
def option_B' : ℝ × ℝ := (1, 2)

def option_C : ℝ × ℝ := (-1, -2)
def option_C' : ℝ × ℝ := (1, 2)

def option_D : ℝ × ℝ := (1, 1)
def option_D' : ℝ × ℝ := (1, 2)

theorem basis_vetors_correct_options:
  ¬ is_basis option_A option_A' ∧ ¬ is_basis option_C option_C' ∧ 
  is_basis option_B option_B' ∧ is_basis option_D option_D' := 
by
  sorry

end basis_vetors_correct_options_l205_205982


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205144

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205144


namespace largest_x_value_l205_205018

noncomputable def quadratic_eq (x : ℝ) : Prop :=
  3 * (9 * x^2 + 15 * x + 20) = x * (9 * x - 60)

theorem largest_x_value (x : ℝ) :
  quadratic_eq x → x = - ((35 - Real.sqrt 745) / 12) ∨
  x = - ((35 + Real.sqrt 745) / 12) :=
by
  intro h
  sorry

end largest_x_value_l205_205018


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205128

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205128


namespace least_positive_integer_divisible_by_four_primes_l205_205121

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l205_205121


namespace units_digit_47_pow_47_l205_205602

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l205_205602


namespace convex_polygon_iff_m_eq_f_l205_205483

noncomputable def f (n : ℕ) [fact (n ≥ 4)] : ℕ := 2 * nat.choose n 4

theorem convex_polygon_iff_m_eq_f :
  ∀ (n : ℕ) (S : finset (euclidean_space ℝ 2)),
    (n ≥ 4) →
    (S.card = n) →
    (∀ P₁ P₂ P₃ ∈ S, ¬ collinear ℝ ({P₁, P₂, P₃}:set (euclidean_space ℝ 2))) →
    (∀ P₁ P₂ P₃ P₄ ∈ S, ¬ concyclic ℝ ({P₁, P₂, P₃, P₄} : set (euclidean_space ℝ 2))) →
    let m_S := ∑ t in S, (finset.card {circle | (circle : set (euclidean_space ℝ 2)).circumscribes (S \ {t})}) in
    (is_convex_polygon S ↔ m_S = f n) :=
begin
  intros,
  sorry,
end

end convex_polygon_iff_m_eq_f_l205_205483


namespace barycenter_squared_dist_relation_l205_205872

variables 
(Real : Type)
[vectSpace : innerProductSpace ℝ Real]
(a b c p : Real)
(g : Real := (2 • a + b + c) / 4)

theorem barycenter_squared_dist_relation :
  ∥p - a∥^2 + ∥p - b∥^2 + ∥p - c∥^2 = 
  3 * ∥p - g∥^2 + ∥g - a∥^2 + ∥g - b∥^2 + ∥g - c∥^2 :=
sorry

end barycenter_squared_dist_relation_l205_205872


namespace least_pos_int_div_by_four_distinct_primes_l205_205165

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205165


namespace ellipse_equation_and_ab_length_range_l205_205788

-- Definitions corresponding to the given conditions
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1
def foci_distance (a b : ℝ) : ℝ := 2 * real.sqrt (a^2 - b^2)
def valid_slope_range (k : ℝ) : Prop := 0 < k^2 ∧ k^2 < 1 / 2
def segment_length_range (ab_length : ℝ) : Prop := (3 * real.sqrt 2) / 2 < ab_length ∧ ab_length < 2 * real.sqrt 2

-- Proof to show the required ellipse and segment length properties
theorem ellipse_equation_and_ab_length_range
  (a : ℝ) (h_a_pos : 0 < a)
  (hf1f2_intersects : foci_distance a 1 = 2)
  (hx_range : ∀ (k : ℝ), valid_slope_range k → 
     ∃ (ab_length : ℝ), segment_length_range ab_length) :
  (ellipse real.sqrt 2 = λ x y, x^2 / 2 + y^2 = 1) ∧ (∃ (ab_length : ℝ), segment_length_range ab_length) :=
sorry

end ellipse_equation_and_ab_length_range_l205_205788


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205156

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205156


namespace ratio_of_division_l205_205338

-- Definitions
variable {Point : Type}

-- Given conditions
variable (A B C D : Point)
variable (M K N : Point) -- Points of intersection of medians of triangles ABC, ABD, BCD respectively
variable (BD_intersection : Point) -- Intersection point of the plane with edge BD

-- Define the plane passing through M, K, N
def plane_through_medians (M K N : Point) : Prop := sorry

-- Define the edge BD
def edge_BD (B D : Point) : Set Point := sorry

-- Denote the ratio calculation
def ratio_division (point P : Point) (B D : Point) : ℚ :=
  let p_BD := distance_between P B / distance_between B D in
  p_BD

-- The main theorem stating the required proof
theorem ratio_of_division (A B C D M K N BD_intersection : Point) :
  plane_through_medians M K N ∧ BD_intersection ∈ edge_BD B D →
  ratio_division BD_intersection B D = (1 / 3) ↔ ratio_division BD_intersection D B = (2 / 3) := sorry

end ratio_of_division_l205_205338


namespace find_hyperbola_fixed_line_through_P_l205_205390

-- Define the conditions of the problem
def center_origin (C : Type) : Prop := 
  C.center = (0, 0)

def left_focus_at (C : Type) : Prop :=
  C.left_focus = (-2 * Real.sqrt 5, 0)

def eccentricity_sqrt_5 (C : Type) : Prop :=
  C.eccentricity = Real.sqrt 5

-- Define the specific hyperbola equation based on the conditions
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 16) = 1

-- State the results to prove
theorem find_hyperbola (C : Type) :
  center_origin C →
  left_focus_at C →
  eccentricity_sqrt_5 C →
  ∀ x y : ℝ, hyperbola_equation x y :=
sorry

-- Define the fixed line for point P
def fixed_line (P : Type) : Prop :=
  P.x = -1

-- State the results to prove for point P
theorem fixed_line_through_P (M N A1 A2 P : Type) (x y : ℝ) :
  (M.line_passing_through (Point.mk (-4, 0))).intersects_left_branch (N : Point) →
  M.in_second_quadrant →
  (A1.coords = (-2, 0)) ∧ (A2.coords = (2, 0)) →
  Point.mk x y = P →
  fixed_line P :=
sorry

end find_hyperbola_fixed_line_through_P_l205_205390


namespace washing_machines_removed_per_box_l205_205845

theorem washing_machines_removed_per_box 
  (crates : ℕ) (boxes_per_crate : ℕ) (washing_machines_per_box : ℕ) 
  (total_removed : ℕ) (total_crates : ℕ) (total_boxes_per_crate : ℕ) 
  (total_washing_machines_per_box : ℕ) 
  (h1 : crates = total_crates) (h2 : boxes_per_crate = total_boxes_per_crate) 
  (h3 : washing_machines_per_box = total_washing_machines_per_box) 
  (h4 : total_removed = 60) (h5 : total_crates = 10) 
  (h6 : total_boxes_per_crate = 6) 
  (h7 : total_washing_machines_per_box = 4):
  total_removed / (total_crates * total_boxes_per_crate) = 1 :=
by
  sorry

end washing_machines_removed_per_box_l205_205845


namespace range_of_a_l205_205902

noncomputable def line_eq (a : ℝ) (x y : ℝ) : ℝ := 3 * x - 2 * y + a 

def pointA : ℝ × ℝ := (3, 1)
def pointB : ℝ × ℝ := (-4, 6)

theorem range_of_a :
  (line_eq a pointA.1 pointA.2) * (line_eq a pointB.1 pointB.2) < 0 ↔ -7 < a ∧ a < 24 := sorry

end range_of_a_l205_205902


namespace find_letters_with_dot_not_straight_l205_205838

theorem find_letters_with_dot_not_straight (D_inter_S S D_union_S : ℕ) (h1 : D_inter_S = 20)
(h2 : S = 46) (h3 : D_union_S = 76) :
  let D := D_union_S - S + D_inter_S in
  D - D_inter_S = 30 :=
by
  let D := D_union_S - S + D_inter_S
  have hD : D = 76 - 46 + 20 := by
    sorry
  show D - D_inter_S = 30 from sorry

end find_letters_with_dot_not_straight_l205_205838


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205139

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205139


namespace max_f_value_no_min_f_value_l205_205800

open Classical
open Real

noncomputable theory

def domain_M (x : ℝ) : Prop :=
  3 - 4 * x + x ^ 2 > 0

def f (x : ℝ) : ℝ :=
  2 ^ (x + 2) - 3 * 4 ^ x

theorem max_f_value :
  ∃ x, domain_M x ∧ f x = 4 / 3 :=
sorry

theorem no_min_f_value :
  ¬∃ m x, domain_M x ∧ f x = m ∧ ∀ y, domain_M y → f y ≥ m :=
sorry

end max_f_value_no_min_f_value_l205_205800


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205145

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205145


namespace units_digit_47_power_47_l205_205616

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l205_205616


namespace revenue_increase_l205_205188

variable (P Q : ℝ) -- Original price and quantity

def P_new : ℝ := 0.80 * P -- New price
def Q_new : ℝ := 1.60 * Q -- New quantity

def original_revenue : ℝ := P * Q -- Original revenue
def new_revenue : ℝ := P_new * Q_new -- New revenue

theorem revenue_increase (P Q : ℝ) :
  new_revenue P Q = 1.28 * original_revenue P Q :=
by
  -- skipping the actual proof
  sorry

end revenue_increase_l205_205188


namespace sum_odd_100_200_l205_205977

theorem sum_odd_100_200 : 
  let odd_numbers := (Finset.range 101).image (λ x, 2 * x + 101 - 1)
  in odd_numbers.sum id = 7500 :=
by
  have h1 :  odd_numbers = {101, 103, ..., 199} := sorry
  have h2 :  odd_numbers.sum id = 50 * (101 + 199) / 2 := sorry
  have h3 :  50 * (101 + 199) / 2 = 7500 := sorry
  exact h3.symm

end sum_odd_100_200_l205_205977


namespace complement_union_problem_l205_205500

noncomputable def U := set.univ
noncomputable def M : set ℝ := {x | abs x < 2}
noncomputable def N : set ℝ := {y | ∃ x, y = 2^x - 1}

theorem complement_union_problem :
  (U \ M) ∪ (U \ N) = (set.Iic (-1) ∪ set.Ici 2) := by
  sorry

end complement_union_problem_l205_205500


namespace quadrilateral_AD_length_l205_205849

variables (AB BC CD AD : ℝ)
variables (B C : ℝ)
variables (sin_C cos_B : ℝ)

noncomputable def length_of_AD (AB BC CD : ℝ) (B C : ℝ) (sin_C cos_B : ℝ) : ℝ :=
  let sin_B := math.cos(math.pi / 2 - B) in
  let cos_C := -sin_C in
  let E := sqrt ((AB + (BC * math.tan(B))) ^ 2 + (CD + (BC * math.tan(C))) ^ 2) in
  sqrt ((AB + E * sin_C) ^ 2 + (E + CD) ^ 2)

theorem quadrilateral_AD_length :
  (AB = 5) → (BC = 6) → (CD = 25) → (sin_C = 4 / 5) → (-cos_B = 4 / 5) → 90 < B ∧ B < 180 → 90 < C ∧ C < 180 → length_of_AD AB BC CD B C sin_C cos_B = 42.9 :=
by sorry

end quadrilateral_AD_length_l205_205849


namespace non_congruent_rectangles_count_l205_205673

theorem non_congruent_rectangles_count :
  (∃ (l w : ℕ), l + w = 50 ∧ l ≠ w) ∧
  (∀ (l w : ℕ), l + w = 50 ∧ l ≠ w → l > w) →
  (∃ (n : ℕ), n = 24) :=
by
  sorry

end non_congruent_rectangles_count_l205_205673


namespace marketing_firm_surveyed_households_l205_205237

theorem marketing_firm_surveyed_households :
  let neither := 80 in
  let only_E := 60 in
  let both := 40 in
  let only_B := 3 * both in
  neither + only_E + both + only_B = 300 := by
  let neither := 80
  let only_E := 60
  let both := 40
  let only_B := 3 * both
  show neither + only_E + both + only_B = 300 from sorry

end marketing_firm_surveyed_households_l205_205237


namespace find_n_l205_205046

def append_fives (n : ℕ) : String :=
  "1200" ++ String.replicate (10 * n + 2) '5'

def base_6_to_nat (s : String) : ℕ :=
  s.foldl (λ acc c, acc * 6 + (c.to_nat - '0'.to_nat)) 0

def x (n : ℕ) : ℕ :=
  base_6_to_nat (append_fives n)

def has_two_prime_factors (n : ℕ) : Prop :=
  (nat.prime_factors n).to_finset.card = 2

theorem find_n : ∀ n : ℕ, has_two_prime_factors (x n) ↔ n = 0 :=
by
  sorry

end find_n_l205_205046


namespace range_of_a_for_min_value_at_x_eq_1_l205_205794

noncomputable def f (a x : ℝ) : ℝ := a*x^3 + (a-1)*x^2 - x + 2

theorem range_of_a_for_min_value_at_x_eq_1 :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a 1 ≤ f a x) → a ≤ 3 / 5 :=
by
  sorry

end range_of_a_for_min_value_at_x_eq_1_l205_205794


namespace gdp_scientific_notation_l205_205025

theorem gdp_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 843000000 = a * 10 ^ n ∧ a = 8.43 ∧ n = 7 :=
by 
  use [8.43, 7]
  -- Further steps and proof verification will be done here.
  sorry

end gdp_scientific_notation_l205_205025


namespace total_presents_l205_205948

-- Definitions based on the problem conditions
def numChristmasPresents : ℕ := 60
def numBirthdayPresents : ℕ := numChristmasPresents / 2

-- Theorem statement
theorem total_presents : numChristmasPresents + numBirthdayPresents = 90 :=
by
  -- Proof is omitted
  sorry

end total_presents_l205_205948


namespace absolute_value_equation_solution_l205_205185

-- mathematical problem representation in Lean
theorem absolute_value_equation_solution (y : ℝ) (h : |y + 2| = |y - 3|) : y = 1 / 2 :=
sorry

end absolute_value_equation_solution_l205_205185


namespace length_DE_l205_205550

-- Definitions for conditions in the problem
variables (AB : ℝ) (DE : ℝ) (areaABC : ℝ)

-- Condition: AB is 15 cm
axiom length_AB : AB = 15

-- Condition: The area of triangle projected below the base is 25% of the area of triangle ABC
axiom area_ratio_condition : (1 / 4) * areaABC = (1 / 2)^2 * areaABC

-- The problem statement translated to Lean proof
theorem length_DE : DE = 7.5 :=
by
  -- Definitions and conditions
  have h1 : AB = 15 := length_AB
  have h2 : (1 / 2)^2 = 1 / 4 := by ring
  calc
    DE = (0.5) * AB :  sorry  -- proportional relationship since triangles are similar
    ... = 0.5 * 15   :  by rw [h1]
    ... = 7.5       :  by norm_num

end length_DE_l205_205550


namespace identify_counterfeit_coins_l205_205325

theorem identify_counterfeit_coins (m : ℕ) (coins : Finset ℕ)
  (h₁ : coins.card = 4^m)
  (h₂ : ∃ (G C : Finset ℕ), G.card = C.card ∧ G.card = 2^(2 * m) ∧ ∀ c ∈ C, c < ∀ g ∈ G, g) :
  ∃ weigh_method : list (Finset ℕ × Finset ℕ), 
    weigh_method.length ≤ 3^m ∧ 
    ∀ (w : Finset ℕ × Finset ℕ) in weigh_method,
    (∀ g₁ g₂ ∈ coins, g₁ ∈ w.1 → g₂ ∈ w.2),
    (∀ g₃ g₄ ∈ coins, g₃ = g₄ → g₃ ∈ w.1 → g₄ ∈ w.2) :=
sorry

end identify_counterfeit_coins_l205_205325


namespace amount_dog_ate_cost_l205_205479

-- Define the costs of each ingredient
def cost_flour : Real := 4
def cost_sugar : Real := 2
def cost_butter : Real := 2.5
def cost_eggs : Real := 0.5

-- Define the number of slices
def number_of_slices := 6

-- Define the number of slices eaten by Laura's mother
def slices_eaten_by_mother := 2

-- Calculate the total cost of the ingredients
def total_cost := cost_flour + cost_sugar + cost_butter + cost_eggs

-- Calculate the cost per slice
def cost_per_slice := total_cost / number_of_slices

-- Calculate the number of slices eaten by Kevin
def slices_eaten_by_kevin := number_of_slices - slices_eaten_by_mother

-- Define the total cost of slices eaten by Kevin
def cost_eaten_by_kevin := slices_eaten_by_kevin * cost_per_slice

-- The main statement to prove
theorem amount_dog_ate_cost :
  cost_eaten_by_kevin = 6 := by
    sorry

end amount_dog_ate_cost_l205_205479


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205142

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l205_205142


namespace redistribute_marbles_l205_205729

theorem redistribute_marbles :
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  (d + m + p + v) / n = 15 :=
by
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  sorry

end redistribute_marbles_l205_205729


namespace cos_C_in_triangle_l205_205461

theorem cos_C_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_sinA : Real.sin A = 4/5) (h_cosB : Real.cos B = 3/5) :
  Real.cos C = 7/25 := 
sorry

end cos_C_in_triangle_l205_205461


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205092

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205092


namespace no_extreme_points_l205_205569

noncomputable def f (x : ℝ) : ℝ := Real.sin x - x

theorem no_extreme_points : ∀ x ∈ Ioo (-Real.pi / 2) (Real.pi / 2), ∀ y ∈ Ioo (-Real.pi / 2) (Real.pi / 2), x ≠ y → f x = f y → false :=
by
  assume x hx y hy hxy hxy_eq
  have H1 : (f' x = Real.cos x - 1) := sorry
  have H2 : (Real.cos x - 1 ≤ 0) := sorry
  have H3 : (f' x ≤ 0) := sorry
  have H4 : (f' y ≤ 0) := sorry
  revert hx hy hxy hxy_eq H1 H2 H3 H4
  sorry

end no_extreme_points_l205_205569


namespace complex_coordinates_l205_205853

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number (1 + i)
def z1 : ℂ := 1 + i

-- Define the complex number i
def z2 : ℂ := i

-- The problem statement to be proven: the given complex number equals 1 - i
theorem complex_coordinates : (z1 / z2) = 1 - i :=
  sorry

end complex_coordinates_l205_205853


namespace least_positive_integer_divisible_by_four_smallest_primes_l205_205177

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l205_205177


namespace minimum_value_f_l205_205331

noncomputable def f (x : ℝ) : ℝ := ∫ t in 0..x, (2 * t - 4)

theorem minimum_value_f : (∀ x ∈ set.Icc (-1 : ℝ) 3, f x ≥ -4) ∧ (∃ x ∈ set.Icc (-1 : ℝ) 3, f x = -4) :=
by
  sorry

end minimum_value_f_l205_205331


namespace num_elements_in_S_l205_205881

noncomputable def f (x : ℝ) : ℝ := (2 * x + 6) / x
def f_seq : ℕ → (ℝ → ℝ)
| 1       := f
| (n + 1) := f ∘ (f_seq n)

def S : set ℝ := {x | ∃ n : ℕ, n > 0 ∧ (f_seq n) x = x}

theorem num_elements_in_S : fintype.card {x : ℝ | x ∈ S} = 2 :=
by { sorry }

end num_elements_in_S_l205_205881


namespace dice_arithmetic_progression_probability_l205_205588

theorem dice_arithmetic_progression_probability : 
  let total_outcomes := 6^3,
      favorable_outcomes := 4 * 3!,
      probability := favorable_outcomes / total_outcomes in
  probability = 1 / 9 :=
by
  sorry

end dice_arithmetic_progression_probability_l205_205588


namespace intersection_points_l205_205970

-- Define original line
def original_line (x : ℝ) : ℝ := 3 * x - 4

-- Define perpendicular line passing through (3, 2)
def perpendicular_line (x : ℝ) : ℝ := -(1 / 3) * x + (11 / 3)

-- Define parallel line passing through (0, -3)
def parallel_line (x : ℝ) : ℝ := 3 * x - 3

-- Prove intersection points
theorem intersection_points :
  (∃ x y : ℝ, original_line x = y ∧ perpendicular_line x = y ∧ x = 21 / 10 ∧ y = 23 / 10) ∧
  (∃ x y : ℝ, parallel_line x = y ∧ perpendicular_line x = y ∧ x = 9 / 5 ∧ y = 12 / 5) :=
by
  sorry

end intersection_points_l205_205970


namespace parallel_rays_converge_at_focus_l205_205904

theorem parallel_rays_converge_at_focus
  (P : Type) [parabola P]
  (X : P) (H : point)
  (F : point)
  (dir : line)
  (p_property: ∀ X, distance F X = distance X (project X dir))
  (tangent_angle_property: ∀ X, equal_angles (tangent X) (line_segment F X) (line_segment X (project X dir)))
  (reflection_property: ∀ ray line : line, parallel ray dir → incidence ray line F → reflects_through ray line F)
  : ∀ ray : line, parallel ray dir → reflects_converge ray F :=
begin
  intros ray h_parallel,
  apply reflection_property ray (parabola.reflect P ray) h_parallel,
  sorry
end

end parallel_rays_converge_at_focus_l205_205904


namespace number_of_special_integers_l205_205320

def tens_digit(n: Nat): Nat := (n / 10) % 10
def units_digit(n: Nat): Nat := n % 10

def special_set: Finset Nat := 
  (Finset.range 151).filter (λ n, 
    let nsq := n^2 in 
    (tens_digit nsq) % 2 = 1 ∧ (units_digit nsq) % 2 = 0)

theorem number_of_special_integers: 
  special_set.card = 30 :=
sorry

end number_of_special_integers_l205_205320


namespace Sophie_l205_205532

-- Define the prices of each item
def price_cupcake : ℕ := 2
def price_doughnut : ℕ := 1
def price_apple_pie : ℕ := 2
def price_cookie : ℚ := 0.60

-- Define the quantities of each item
def qty_cupcake : ℕ := 5
def qty_doughnut : ℕ := 6
def qty_apple_pie : ℕ := 4
def qty_cookie : ℕ := 15

-- Define the total cost function for each item
def cost_cupcake := qty_cupcake * price_cupcake
def cost_doughnut := qty_doughnut * price_doughnut
def cost_apple_pie := qty_apple_pie * price_apple_pie
def cost_cookie := qty_cookie * price_cookie

-- Define total expenditure
def total_expenditure := cost_cupcake + cost_doughnut + cost_apple_pie + cost_cookie

-- Assertion of total expenditure
theorem Sophie's_total_expenditure : total_expenditure = 33 := by
  -- skipping proof
  sorry

end Sophie_l205_205532


namespace least_pos_int_div_by_four_distinct_primes_l205_205104

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l205_205104


namespace problem_l205_205322

theorem problem (n : Nat) (a b : ℕ) :
  n = 7 → 
  a = 7 → 
  b = (-1) → 
  (∀ k, ∑ i in Finset.range (n + 1), (Nat.choose n i) * (a ^ (n - i)) * (b ^ i) = 2^n) ∧
  (∀ k, k ≠ 7 → Nat.choose n (k - 1) ≠ 49) ∧ 
  (∑ i in Finset.range (7 + 1), Nat.choose 7 i * 7 ^ (7 - i) * (-1) ^ i = 6^7) ∧
  (7 + 1 = 8) :=
by
  intros
  sorry

end problem_l205_205322


namespace unique_four_digit_numbers_from_2021_l205_205751

theorem unique_four_digit_numbers_from_2021 : 
  let digits := [2, 0, 2, 1] in 
  let unique_permutations := { p | multiset.erase_dup (multiset.of_list (list.permutations digits)).to_finset.count = 9 } in
  unique_permutations.to_list.count = 9 :=
sorry

end unique_four_digit_numbers_from_2021_l205_205751


namespace solution_set_of_f_lt_2_l205_205789

noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then log (2 - x) / log 2 else x^(1/3 : ℝ)

theorem solution_set_of_f_lt_2 :
  { x : ℝ | f x < 2 } = { x : ℝ | -2 < x ∧ x < 8 } :=
by
  sorry

end solution_set_of_f_lt_2_l205_205789


namespace bananas_shared_l205_205505

theorem bananas_shared (initial_bananas : ℕ) (bananas_left : ℕ) : 
  initial_bananas = 88 ∧ bananas_left = 84 → initial_bananas - bananas_left = 4 := 
by 
  intro h
  cases h with h1 h2
  rw [h1, h2]
  exact rfl

end bananas_shared_l205_205505


namespace total_path_length_travelled_by_B_l205_205243

-- Define the setting that appears as conditions in the problem
def radius : ℝ := 4 / Real.pi
def quarter_circle_circumference : ℝ := 2 * Real.pi * radius
def distance_quarter_circle_roll : ℝ := quarter_circle_circumference / 4
def straight_line_distance : ℝ := 2

-- Statement of the problem in Lean 4
theorem total_path_length_travelled_by_B :
  (2 * distance_quarter_circle_roll) + (2 * straight_line_distance) + (2 * distance_quarter_circle_roll) = 10 := 
by 
  -- Use the conditions directly in the proof
  have h1 : distance_quarter_circle_roll = 2 := 
    by simp [distance_quarter_circle_roll, quarter_circle_circumference, radius]; linarith
  
  calc 
    (2 * distance_quarter_circle_roll) + (2 * straight_line_distance) + (2 * distance_quarter_circle_roll)
      = (2 * 2) + (2 * 2) + (2 * 2) : by congr; assumption
    ... = 4 + 4 + 2 + 2                             : by ring
    ... = 10                                        : by ring

end total_path_length_travelled_by_B_l205_205243


namespace sum_of_values_l205_205042

open Real

-- The given piecewise function
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x + 2 else -x^2

-- Lean statement for the proof problem
theorem sum_of_values (a : ℝ) (h : f(f(a)) = 2) : a = 0 ∨ a = sqrt 2 ∧ (0 + sqrt 2 = sqrt 2) :=
by
  sorry

end sum_of_values_l205_205042


namespace radius_range_of_sector_l205_205578

theorem radius_range_of_sector (a : ℝ) (h : a > 0) :
  ∃ (R : ℝ), (a / (2 * (1 + π)) < R ∧ R < a / 2) :=
sorry

end radius_range_of_sector_l205_205578


namespace least_positive_integer_divisible_by_four_distinct_primes_l205_205090

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l205_205090


namespace polynomial_degree_2_exists_l205_205753

theorem polynomial_degree_2_exists :
  ∃ (P : ℝ[X]), degree P = 2 ∧ P.eval 0 = 1 ∧ P.eval 1 = 2 ∧ P.eval 2 = 5 ∧ P = X^2 + 1 := 
by
  sorry

end polynomial_degree_2_exists_l205_205753


namespace cos_theta_plus_pi_over_4_l205_205398

noncomputable def cos_sum (θ : ℝ) : ℝ :=
  let x := 3;
  let y := -4;
  let r := real.sqrt (x^2 + y^2);
  let cos_θ := x / r;
  let sin_θ := y / r;
  cos_θ * real.cos (real.pi / 4) - sin_θ * real.sin (real.pi / 4)

theorem cos_theta_plus_pi_over_4 :
  cos_sum (θ : ℝ) = (7 * real.sqrt 2 / 10) :=
sorry

end cos_theta_plus_pi_over_4_l205_205398


namespace series_sum_correct_l205_205278

def term (n : ℕ) : ℝ :=
  n * (1 - 1 / (n * n))

def series_sum (start end_ : ℕ) (term : ℕ → ℝ) : ℝ :=
  (Finset.range (end_ + 1)).filter (λ x, x ≥ start).sum term

theorem series_sum_correct :
  series_sum 2 15 term = 119 := 
sorry

end series_sum_correct_l205_205278


namespace grandpa_max_pieces_l205_205421

theorem grandpa_max_pieces (m n : ℕ) (h : (m - 3) * (n - 3) = 9) : m * n = 112 :=
sorry

end grandpa_max_pieces_l205_205421


namespace fraction_of_180_l205_205070

theorem fraction_of_180 : (1 / 2) * (1 / 3) * (1 / 6) * 180 = 5 := by
  sorry

end fraction_of_180_l205_205070


namespace find_range_of_a_l205_205406

noncomputable def e : ℝ := Real.exp 1

def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

def has_extreme_value_point_in_interval (a : ℝ) : Prop :=
∃ x : ℝ, x ∈ Ioo (-1) 1 ∧ deriv (λ x, f x a) x = 0

def has_exactly_one_integer_solution (a : ℝ) : Prop :=
∃ n : ℤ, (f n a < 0) ∧ ∀ m : ℤ, m ≠ n → f m a >= 0

def range_of_a (a : ℝ) : Prop :=
a ∈ Icc ((e^2 - 1) / (2 * e^2)) ((e-1) / e) ∨ a ∈ Ioo (e-1) e

theorem find_range_of_a (a : ℝ) :
  has_extreme_value_point_in_interval a →
  has_exactly_one_integer_solution a →
  range_of_a a :=
sorry

end find_range_of_a_l205_205406


namespace line_intersections_ellipse_l205_205271

noncomputable def ellipse_center := (0 : ℝ, 0 : ℝ)

def ellipse_passes_through_origin_point (p : ℝ × ℝ) : Prop :=
  p = (0, real.sqrt 3)

def ellipse_focus_right := (1 : ℝ, 0 : ℝ)

def ellipse_symmetric_about_coordinate_axes (p : ℝ × ℝ) : Prop :=
  ellipse (p)

def standard_ellipse_equation : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ 
  (λ x y, x^2 / a^2 + y^2 / b^2 = 1) 

def lines_through_points_are_parallel (A B C D: ℝ × ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, B = (A.1 + t, A.2 + k ^ 2) ∧ D = (C.1 + t, C.2 + k ^ 2)

theorem line_intersections_ellipse (A B C D: ℝ × ℝ):
  ∀ (a b : ℝ) , a > b  ∧ b > 0 ∧
  ellipse_passes_through_origin_point (0, real.sqrt 3) ∧
  ellipse_focus_right = (1, 0) ∧
  ellipse_symmetric_about_coordinate_axes (p : ℝ × ℝ)
  standard_ellipse_equation → 
  lines_through_points_are_parallel A B C D →
  (|AB|^2 = 4|CD|) :=
  sorry

end line_intersections_ellipse_l205_205271


namespace least_prime_in_sum_even_set_of_7_distinct_primes_l205_205498

noncomputable def is_prime (n : ℕ) : Prop := sorry -- Assume an implementation of prime numbers

theorem least_prime_in_sum_even_set_of_7_distinct_primes {q : Finset ℕ} 
  (hq_distinct : q.card = 7) 
  (hq_primes : ∀ n ∈ q, is_prime n) 
  (hq_sum_even : q.sum id % 2 = 0) :
  ∃ m ∈ q, m = 2 :=
by
  sorry

end least_prime_in_sum_even_set_of_7_distinct_primes_l205_205498


namespace least_pos_int_divisible_by_four_distinct_primes_l205_205150

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l205_205150


namespace find_product_xyz_l205_205052

-- Definitions for the given conditions
variables (x y z : ℕ) -- positive integers

-- Conditions
def condition1 : Prop := x + 2 * y = z
def condition2 : Prop := x^2 - 4 * y^2 + z^2 = 310

-- Theorem statement
theorem find_product_xyz (h1 : condition1 x y z) (h2 : condition2 x y z) : 
  x * y * z = 11935 ∨ x * y * z = 2015 :=
sorry

end find_product_xyz_l205_205052


namespace least_pos_int_div_by_four_distinct_primes_l205_205169

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l205_205169


namespace total_profit_calculation_l205_205281

noncomputable def candied_apple_profit (selling_price cost_price : ℝ) (quantity : ℕ) : ℝ :=
(selling_price - cost_price) * quantity

noncomputable def candied_grape_profit (selling_price cost_price : ℝ) (quantity : ℕ) : ℝ :=
(selling_price - cost_price) * quantity

noncomputable def candied_orange_profit (selling_price cost_price : ℝ) (quantity : ℕ) : ℝ :=
(selling_price - cost_price) * quantity

noncomputable def total_profit (apple_profit grape_profit orange_profit : ℝ) : ℝ :=
apple_profit + grape_profit + orange_profit

theorem total_profit_calculation :
  let apple_selling_price := 2
  let apple_cost_price := 1.2
  let num_apples := 15
  let grape_selling_price := 1.5
  let grape_cost_price := 0.9
  let num_grapes := 12
  let orange_selling_price := 2.5
  let orange_cost_price := 1.5
  let num_oranges := 10 in
  candied_apple_profit apple_selling_price apple_cost_price num_apples +
  candied_grape_profit grape_selling_price grape_cost_price num_grapes +
  candied_orange_profit orange_selling_price orange_cost_price num_oranges = 29.20 :=
by
  sorry

end total_profit_calculation_l205_205281
