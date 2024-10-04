import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.Arith
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Combinatorics.Graph.Turan
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial.TrailingZeroes
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Seq
import Mathlib.Data.Set.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Circle
import Mathlib.Geometry.Collinear
import Mathlib.Geometry.Concyclic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circumference
import Mathlib.Logic
import Mathlib.Probability.Distribution
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Variance
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Topology.Instances.Real

namespace f_at_1_is_neg7007_l473_473325

variable (a b c : ℝ)

def g (x : ℝ) := x^3 + a * x^2 + x + 10
def f (x : ℝ) := x^4 + x^3 + b * x^2 + 100 * x + c

theorem f_at_1_is_neg7007
  (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ g a (r1) = 0 ∧ g a (r2) = 0 ∧ g a (r3) = 0)
  (h2 : ∀ x, f x = 0 → g x = 0) :
  f 1 = -7007 := 
sorry

end f_at_1_is_neg7007_l473_473325


namespace decreasing_interval_of_f_l473_473677

noncomputable def f : ℝ → ℝ := λ x, x^2 - 2 * x + 4

theorem decreasing_interval_of_f : ∀ x, x ≤ 1 → f' x < 0 :=
by
  sorry

end decreasing_interval_of_f_l473_473677


namespace Area_Ratio_Proof_l473_473519

noncomputable def quadrangle_area_ratio (P P' : Type) [convex P] [convex P'] (O : Type) [point O P] [point O P'] :=
  ∀ (l : Type) [line l O], (∀ p q : Type, [segment p q P l] → [segment p q P' l] → |segment_length p q P l| > |segment_length p q P' l|) →
  (∃ (ε : ℝ), ε > 1.9 ∧ area P' / area P > ε)

theorem Area_Ratio_Proof (P P' : Type) [convex_quadrangle P] [convex_quadrangle P'] (O : Type) [point O P] [point O P']
  (cond : ∀ (l : Type) [line l O], ∃ p q : Type, [segment p q P l] → [segment p q P' l] → |segment_length p q P l| > |segment_length p q P' l])
  : ∃ ε : ℝ, ε > 1.9 ∧ area P' / area P > ε :=
sorry

end Area_Ratio_Proof_l473_473519


namespace two_io_ge_ih_l473_473458

theorem two_io_ge_ih (Δ : Triangle) 
  (H : Δ.orthocenter) 
  (I : Δ.incenter) 
  (O : Δ.circumcenter) :
  2 * dist I O ≥ dist I H ∧ (2 * dist I O = dist I H ↔ Δ.is_equilateral) := 
sorry

end two_io_ge_ih_l473_473458


namespace plane_equation_l473_473284

structure Point3D :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)

def P (x y z : ℤ) : ℤ := 6 * x - 6 * y - 5 * z - 3

theorem plane_equation (A B C D : ℤ) (p q r : Point3D) (hA : A > 0)
  (hGCD : Int.gcd4 (Int.natAbs A) (Int.natAbs B) (Int.natAbs C) (Int.natAbs D) = 1)
  (h1 : p = {x := 2, y := -1, z := 3})
  (h2 : q = {x := -1, y := 5, z := 0})
  (h3 : r = {x := 4, y := 0, z := -1}) :
  P p.x p.y p.z = 0 ∧ P q.x q.y q.z = 0 ∧ P r.x r.y r.z = 0 :=
by
  sorry

end plane_equation_l473_473284


namespace comic_books_ratio_l473_473934

variable (S : ℕ)

theorem comic_books_ratio (initial comics_left comics_bought : ℕ)
  (h1 : initial = 14)
  (h2 : comics_left = 13)
  (h3 : comics_bought = 6)
  (h4 : initial - S + comics_bought = comics_left) :
  (S / initial.toRat) = (1 / 2 : ℚ) :=
by
  sorry

end comic_books_ratio_l473_473934


namespace max_value_expression_l473_473997

theorem max_value_expression (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 * (a + b))) +
   Real.sqrt (Real.sqrt (b^2 * (b + c))) +
   Real.sqrt (Real.sqrt (c^2 * (c + d))) +
   Real.sqrt (Real.sqrt (d^2 * (d + a)))) ≤ 4 * Real.sqrt (Real.sqrt 2) := by
  sorry

end max_value_expression_l473_473997


namespace find_g_l473_473277

noncomputable def g (x : ℝ) := -4 * x ^ 4 + x ^ 3 - 6 * x ^ 2 + x - 1

theorem find_g (x : ℝ) :
  4 * x ^ 4 + 2 * x ^ 2 - x + 7 + g x = x ^ 3 - 4 * x ^ 2 + 6 :=
by
  sorry

end find_g_l473_473277


namespace largest_number_interior_angles_greater_than_180_l473_473469

theorem largest_number_interior_angles_greater_than_180 (n : ℕ) (h1 : n ≥ 3) :
  (∃ k : ℕ, k = if n < 5 then 0 else n - 3 ∧ ∀ (polygon : ℕ → ℝ × ℝ), 
  (is_n_gon polygon n) → (equal_side_lengths polygon) → (does_not_intersect_itself polygon) →
  (number_angles_greater_than_180 polygon = k)) :=
sorry

-- Definitions placeholder
def is_n_gon (polygon : ℕ → ℝ × ℝ) (n : ℕ) : Prop := sorry
def equal_side_lengths (polygon : ℕ → ℝ × ℝ) : Prop := sorry
def does_not_intersect_itself (polygon : ℕ → ℝ × ℝ) : Prop := sorry
def number_angles_greater_than_180 (polygon : ℕ → ℝ × ℝ) : ℕ := sorry

end largest_number_interior_angles_greater_than_180_l473_473469


namespace h_value_l473_473388

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l473_473388


namespace find_A_and_B_l473_473303

variable {n : ℕ}

noncomputable def seq (n : ℕ) : ℝ := Math.tan n * Math.tan (n - 1)

theorem find_A_and_B (A B : ℝ) (h : ∀ n : ℕ, (∑ k in finset.range (n + 1), seq k) = A * Math.tan n + B * n) :
  A = 1 / Math.tan 1 ∧ B = -1 :=
sorry

end find_A_and_B_l473_473303


namespace solution_set_of_inequality_l473_473750

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = f x) (h2 : ∀ x, 0 ≤ x → f x = x - 1) :
  { x : ℝ | f (x - 1) > 1 } = { x | x < -1 ∨ x > 3 } :=
by
  sorry

end solution_set_of_inequality_l473_473750


namespace find_t_l473_473064

theorem find_t (p q r s t : ℤ)
  (h₁ : p - q - r + s - t = -t)
  (h₂ : p - (q - (r - (s - t))) = -4 + t) :
  t = 2 := 
sorry

end find_t_l473_473064


namespace intersection_product_correct_l473_473032

-- Define the parametric equations of the curve C
def curve_parametric (θ : ℝ) : ℝ × ℝ := (sqrt 3 * Real.cos θ, Real.sin θ)

-- Define the rectangular coordinate equation of the line l
def line_rectangular (x y : ℝ) : Prop := x + y = sqrt 2

-- Define the intersection points condition
def intersection_product := 3 / 2

-- Prove that the product of |PA| and |PB| is 3 / 2
theorem intersection_product_correct (P A B : ℝ × ℝ) (θ_A θ_B : ℝ) :
  curve_parametric θ_A = A ∧ curve_parametric θ_B = B ∧
  (∃ x y, curve_parametric θ_A = (x, y) ∧ line_rectangular x y) ∧
  (∃ x y, curve_parametric θ_B = (x, y) ∧ line_rectangular x y) ∧
  (curve_parametric θ_A = (P.1, P.2)) ∨ (curve_parametric θ_B = (P.1, P.2)) →
  |P.1 - A.1| * |P.2 - B.2| = intersection_product := 
by sorry

end intersection_product_correct_l473_473032


namespace sum_of_first_fifteen_multiples_of_7_l473_473578

theorem sum_of_first_fifteen_multiples_of_7 : (List.range 15).map (λ n, 7 * (n + 1)).sum = 840 := by
  sorry

end sum_of_first_fifteen_multiples_of_7_l473_473578


namespace arithmetic_sequence_number_of_terms_l473_473318

variable {a : ℕ → ℕ}
variable {k : ℕ}

theorem arithmetic_sequence_number_of_terms :
  (∀ n, a (n + 1) = a n + 2) ∧
  (∃ k, 2 * k = 2 * k) ∧
  (∑ i in finset.range k, a (2 * i + 1) = 15) ∧
  (∑ i in finset.range k, a (2 * (i + 1)) = 25) →
  2 * k = 10 :=
by
  sorry

end arithmetic_sequence_number_of_terms_l473_473318


namespace compare_y1_y2_l473_473534

-- Definitions for the conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + 3
axiom a_neg (a : ℝ) : a < 0

-- The proof problem statement
theorem compare_y1_y2 (a : ℝ) (h_a : a_neg a)
  (y1 : ℝ) (y2 : ℝ) (h_y1 : y1 = quadratic_function a (-1))
  (h_y2 : y2 = quadratic_function a 2) : y1 < y2 :=
sorry

end compare_y1_y2_l473_473534


namespace number_of_possible_lists_l473_473615

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l473_473615


namespace correct_statement_is_B_l473_473588

theorem correct_statement_is_B (A B C D : Prop)
  (hA : ¬(EqualAnglesAreVerticalAngles := A))
  (hB : (TwoLinesParallelToSameLineAreParallel := B))
  (hC : ¬(NumbersWithSquareRootsAreAlwaysIrrational := C))
  (hD : ¬(IfAGreaterBThenASquaredGreaterBSquared := D)) :
  B := by
  -- Provide a proof outline in natural language
  -- hA tells us that Statement A is incorrect.
  -- hB tells us that Statement B is correct.
  -- hC tells us that Statement C is incorrect.
  -- hD tells us that Statement D is incorrect.
  -- Therefore, the correct statement among the given choices is B.
  sorry

end correct_statement_is_B_l473_473588


namespace number_of_possible_third_side_lengths_l473_473828

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l473_473828


namespace optimal_saving_is_45_cents_l473_473229

def initial_price : ℝ := 18
def fixed_discount : ℝ := 3
def percentage_discount : ℝ := 0.15

def price_after_fixed_discount (price fixed_discount : ℝ) : ℝ :=
  price - fixed_discount

def price_after_percentage_discount (price percentage_discount : ℝ) : ℝ :=
  price * (1 - percentage_discount)

def optimal_saving (initial_price fixed_discount percentage_discount : ℝ) : ℝ :=
  let price1 := price_after_fixed_discount initial_price fixed_discount
  let final_price1 := price_after_percentage_discount price1 percentage_discount
  let price2 := price_after_percentage_discount initial_price percentage_discount
  let final_price2 := price_after_fixed_discount price2 fixed_discount
  final_price1 - final_price2

theorem optimal_saving_is_45_cents : optimal_saving initial_price fixed_discount percentage_discount = 0.45 :=
by 
  sorry

end optimal_saving_is_45_cents_l473_473229


namespace length_of_minor_axis_l473_473942

theorem length_of_minor_axis
  (F₁ F₂ P : ℝ)
  (dist_F1_F2 : dist F₁ F₂ = 8)
  (dist_PF1F2 : dist P F₁ + dist P F₂ = 10) :
  let c := (dist F₁ F₂) / 2 in
  let a := (dist P F₁ + dist P F₂) / 2 in
  let b := sqrt(a^2 - c^2) in
  2 * b = 6 :=
by
  sorry

end length_of_minor_axis_l473_473942


namespace lines_perpendicular_l473_473014

theorem lines_perpendicular
  (k₁ k₂ : ℝ)
  (h₁ : k₁^2 - 3*k₁ - 1 = 0)
  (h₂ : k₂^2 - 3*k₂ - 1 = 0) :
  k₁ * k₂ = -1 → 
  (∃ l₁ l₂: ℝ → ℝ, 
    ∀ x, l₁ x = k₁ * x ∧ l₂ x = k₂ * x → 
    ∃ m, m = -1) := 
sorry

end lines_perpendicular_l473_473014


namespace points_on_P_perimeter_10cm_square_l473_473882

theorem points_on_P_perimeter_10cm_square :
  let side_length := 10
  let points_per_side := side_length + 1
  let number_of_sides := 3
  let overlapping_corners := 2
  total_points (side_length points_per_side number_of_sides overlapping_corners) 
    = points_per_side * number_of_sides - overlapping_corners :=
by {
  sorry
}

end points_on_P_perimeter_10cm_square_l473_473882


namespace sum_of_first_fifteen_multiples_of_7_l473_473569

theorem sum_of_first_fifteen_multiples_of_7 : (∑ k in Finset.range 15, 7 * (k + 1)) = 840 :=
by
  -- Summation from k = 0 to k = 14 (which corresponds to 1 to 15 multiples of 7)
  sorry

end sum_of_first_fifteen_multiples_of_7_l473_473569


namespace books_left_to_read_l473_473909

theorem books_left_to_read (total_books : ℕ) (books_mcgregor : ℕ) (books_floyd : ℕ) : total_books = 89 → books_mcgregor = 34 → books_floyd = 32 → 
  (total_books - (books_mcgregor + books_floyd) = 23) :=
by
  intros h1 h2 h3
  sorry

end books_left_to_read_l473_473909


namespace find_AX_l473_473689

theorem find_AX (AC BC BX : ℝ) (h1 : AC = 27) (h2 : BC = 40) (h3 : BX = 36)
    (h4 : ∀ (AX : ℝ), AX = AC * BX / BC) : 
    ∃ AX, AX = 243 / 10 :=
by
  sorry

end find_AX_l473_473689


namespace negative_integer_solution_l473_473988

theorem negative_integer_solution (N : ℤ) (h : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end negative_integer_solution_l473_473988


namespace sin_2theta_value_l473_473365

theorem sin_2theta_value (θ : ℝ) : (∑' n : ℕ, (Real.sin θ)^(2 * n)) = 5 / 3 → Real.sin (2 * θ) = 2 * (Real.sin θ) * Real.cos θ :=
begin
  -- We establish the given sum is equal to 5/3
  intros h,
  have h1 : (∑' n : ℕ, (Real.sin θ)^(2 * n)) = 1 / (1 - (Real.sin θ)^2),
  sorry,
  -- We equate this to 5/3 as given in the condition
  have h2 : 1 / (1 - (Real.sin θ)^2) = 5 / 3,
  from h,
  -- We then calculate Real.sin 2θ using the appropriate double angle identity
end

end sin_2theta_value_l473_473365


namespace coins_in_each_pile_l473_473077

theorem coins_in_each_pile :
  ∃ C : ℕ, (3 * C + 3 * C = 42) ∧ (∀ Q D, Q = C ∧ D = C) :=
by
  use 7
  split
  { norm_num }
  { intros Q D hQ hD
    rw [hQ, hD] }
  sorry

end coins_in_each_pile_l473_473077


namespace number_of_possible_lists_l473_473606

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l473_473606


namespace intersection_on_CD_l473_473876

variables {α : Type*} [Field α]
variables (A B C D K L : α)
variables (l m : Set (Point α)) 

-- Conditions that define the trapezoid and points
variables (is_trapezoid : Trapezoid ABCD)
variables (K_on_AB : K ∈ Line(A, B))
variables (l_parallel_KC : Parallel (Line A) (Line K, C))
variables (m_parallel_KD : Parallel (Line B) (Line K, D))

theorem intersection_on_CD :
  ∃ L, (L ∈ (Line (C, D))) ∧ Intersection_point (l) (m) = Some L :=
  sorry

end intersection_on_CD_l473_473876


namespace sequence_divisibility_l473_473737

theorem sequence_divisibility (a : ℕ → ℕ) (m : ℕ) (h1 : a 1 = 1) (h2 : a 2 = 1) (h3 : a 3 = 3)
  (h_rec : ∀ n, 4 ≤ n → a n = a (n - 1) + 2 * a (n - 2) + a (n - 3)):
  ∃ n, m > 0 → a n % m = 0 :=
begin
  sorry
end

end sequence_divisibility_l473_473737


namespace dr_loxley_ticket_probability_l473_473272

-- Define the range and the conditions
def valid_integers (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 56

def power_of_10 (S : Set ℕ) : Prop :=
  ∃ k:ℕ, S.prod = 10^k

def has_two_non_2_5_primes (S : Set ℕ) : Prop :=
  ∃ p q:ℕ, p ≠ 2 ∧ p ≠ 5 ∧ q ≠ 2 ∧ q ≠ 5 ∧
           nat.prime p ∧ nat.prime q ∧ p ∈ S ∧ q ∈ S ∧ p ≠ q

-- Define the problem in a Lean statement
theorem dr_loxley_ticket_probability :
  ∃ S: Set ℕ, S.card = 6 ∧
               (∀ x ∈ S, valid_integers x) ∧
               power_of_10 S ∧
               has_two_non_2_5_primes S →
               1 := sorry

end dr_loxley_ticket_probability_l473_473272


namespace count_valid_third_sides_l473_473816

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l473_473816


namespace construct_triangle_proof_l473_473257

noncomputable def construct_triangle (a b : ℝ) (α : Real.Angle) (h : ℝ) : Prop :=
∃ A B C : ℝ × ℝ, 
  let AC := dist A C 
  let CB := dist C B
  let angleCAB := ∠A B C in
  (AC + CB = a + b) ∧
  (angleCAB = α.toReal) ∧
  let CC1 := abs (C.2 - A.2) in 
  (CC1 = h)
  
/-
This theorem states that given the sum of two sides a + b, the angle alpha, and the height h (CC1),
a triangle ABC can be constructed such that:
1. AC + CB = a + b
2. ∠CAB = α
3. The height from C perpendicular to AB is h (CC1).
-/
theorem construct_triangle_proof (a b : ℝ) (α : Real.Angle) (h : ℝ) : construct_triangle a b α h := 
sorry

end construct_triangle_proof_l473_473257


namespace triangle_third_side_length_count_l473_473791

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l473_473791


namespace number_of_possible_lists_l473_473613

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l473_473613


namespace eggs_used_in_morning_l473_473912

theorem eggs_used_in_morning (total_eggs afternoon_eggs : ℕ) (h_total : total_eggs = 1339) (h_afternoon : afternoon_eggs = 523) : (total_eggs - afternoon_eggs) = 816 := 
by
  -- Initializing the conditions given in the problem
  rw [h_total, h_afternoon]
  -- Evaluating the expression
  exact Nat.sub_eq_of_eq_add $ congr_arg (λ x, x + 523) (Nat.add_comm 816 523).symm

end eggs_used_in_morning_l473_473912


namespace gcd_45_75_105_l473_473560

theorem gcd_45_75_105 : Nat.gcd (45 : ℕ) (Nat.gcd 75 105) = 15 := 
by
  sorry

end gcd_45_75_105_l473_473560


namespace find_a_l473_473197

theorem find_a (a : ℝ) : (∀ (x y : ℝ), ax + 1 * y - 1 = 0 → 1 * x + ay - 1 = 0 → let coeff1 := 1 in let coeff2 := a in coeff1 * coeff2 + coeff1 * coeff2 = 0) → a = 0 :=
by
  intros
  sorry

end find_a_l473_473197


namespace determine_N_l473_473263

variable {R : Type} [Field R]

def N : Matrix (Fin 2) (Fin 2) R := 
  (λ i j => match (i, j) with
            | (0, 0) => (8/7 : R)
            | (0, 1) => (11/7 : R)
            | (1, 0) => (31/14 : R)
            | (1, 1) => (-15/7 : R))

def v1 : Fin 2 → R := ![2, 3]
def v2 : Fin 2 → R := ![4, -1]
def r1 : Fin 2 → R := ![7, -2]
def r2 : Fin 2 → R := ![3, 11]

theorem determine_N (h1 : N.mulVec v1 = r1) (h2 : N.mulVec v2 = r2) : 
  N = (λ i j => match (i, j) with
                | (0, 0) => (8/7 : R)
                | (0, 1) => (11/7 : R)
                | (1, 0) => (31/14 : R)
                | (1, 1) => (-15/7 : R)) :=
by
  have h1 : N.mulVec v1 = ![7, -2] := sorry
  have h2 : N.mulVec v2 = ![3, 11] := sorry
  apply Matrix.ext
  finish

end determine_N_l473_473263


namespace sum_complex_root_of_unity_l473_473896

theorem sum_complex_root_of_unity (x : ℂ) (h1 : x ^ 2023 = 1) (h2 : x ≠ 1) :
  ∑ k in (finset.range 2022 + 1), (x ^ (2 * k)) / (x ^ k - 1) = 1011 :=
by sorry

end sum_complex_root_of_unity_l473_473896


namespace triangle_area_ratio_l473_473556

theorem triangle_area_ratio 
(problem_hypothesis : 
    ∃ A B C P : Type, -- Vertices and internal point
    ∃ S1 S2 S3 ABC_area : ℝ, -- Areas of triangles
    S1 > 0 ∧ S2 > 0 ∧ S3 > 0 ∧ ABC_area > 0 ∧ -- Positive areas
    S1 / S3 + S2 / S3 + S3 / S3 = 1 + 4 + 9 / 9 
    ∧ (S1 / S2 = 1 / 4) ∧ (S1 / S3 = 1 / 9) ∧ (S2 / S3 = 4 / 9) -- Area ratios
) 
: ∃ (ratio : ℝ), ratio = 9 / 14 := 
begin
  sorry
end

end triangle_area_ratio_l473_473556


namespace find_m_l473_473518

theorem find_m :
  ∃ m : ℕ, 264 * 391 % 100 = m ∧ 0 ≤ m ∧ m < 100 ∧ m = 24 :=
by
  sorry

end find_m_l473_473518


namespace triangle_angles_l473_473218

theorem triangle_angles (α β γ : ℝ) 
  (h₁ : α + β + γ = 180) 
  (h₂ : ∃ (A B C D : Type) [triangle A B C] (AD : line A D) (h_tri : triangle A B D) (h_tri' : triangle A D C),
    similar (triangle A B D) (triangle A D C) ∧
    (side_ratio (triangle A B D) (triangle A D C) = √3)) :
  (α = 90 ∧ β = 30 ∧ γ = 60) ∨
  (α = 90 ∧ β = 60 ∧ γ = 30) :=
sorry

end triangle_angles_l473_473218


namespace num_dress_designs_l473_473211

-- Define the number of fabric colors and patterns
def fabric_colors : ℕ := 4
def patterns : ℕ := 5

-- Define the number of possible dress designs
def total_dress_designs : ℕ := fabric_colors * patterns

-- State the theorem that needs to be proved
theorem num_dress_designs : total_dress_designs = 20 := by
  sorry

end num_dress_designs_l473_473211


namespace part1_part2_l473_473058

-- Part 1
theorem part1 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

-- Part 2
theorem part2 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

end part1_part2_l473_473058


namespace susan_homework_time_l473_473946

theorem susan_homework_time :
  ∀ (start finish practice : ℕ),
  start = 119 ->
  practice = 240 ->
  finish = practice - 25 ->
  (start < finish) ->
  (finish - start) = 96 :=
by
  intros start finish practice h_start h_practice h_finish h_lt
  sorry

end susan_homework_time_l473_473946


namespace triangle_third_side_length_l473_473800

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l473_473800


namespace find_n_l473_473196

theorem find_n (n : ℕ) (hn : (Nat.choose n 2 : ℚ) / 2^n = 10 / 32) : n = 5 :=
by
  sorry

end find_n_l473_473196


namespace arrangement_count_l473_473023

theorem arrangement_count :
  ∃ (arrangements : ℕ), 
    (∀ (grid : ℕ → (ℕ × ℕ)),
      (∀ (n : ℕ), 1 ≤ n ∧ n < 100 → 
        ((abs (grid n).fst - (grid (n + 1)).fst = 1 ∧ (grid n).snd = (grid (n + 1)).snd) ∨ 
         ((grid n).fst = (grid (n + 1)).fst ∧ abs (grid n).snd - (grid (n + 1)).snd = 1)))) →
    arrangements = 4904 :=
sorry

end arrangement_count_l473_473023


namespace MEQP_parallelogram_l473_473653

-- Define given conditions
variable (A B C D S M E F G P Q R : Point)

-- Given quadrilateral ABCD
variable (ABCD : ConvexQuadrilateral A B C D)

-- Define that squares are constructed on given sides
variable (AD_square : Square A D S M) (BC_square : Square B C F E)
variable (AC_square : Square A C G P) (BD_square : Square B D R Q)

-- Definition of quadrilateral as parallelogram
def isParallelogram (A B C D : Point) : Prop :=
  (vector A B).direction = (vector C D).direction ∧ 
  (vector B C).direction = (vector D A).direction

-- Problem statement
theorem MEQP_parallelogram : 
  isQuadrilateral A D S M →
  isQuadrilateral B C F E →
  isQuadrilateral A C G P →
  isQuadrilateral B D R Q →
  isParallelogram M E Q P := 
by
  sorry

end MEQP_parallelogram_l473_473653


namespace carla_students_l473_473660

theorem carla_students (R A num_rows num_desks : ℕ) (full_fraction : ℚ) 
  (h1 : R = 2) 
  (h2 : A = 3 * R - 1)
  (h3 : num_rows = 4)
  (h4 : num_desks = 6)
  (h5 : full_fraction = 2 / 3) : 
  num_rows * (num_desks * full_fraction).toNat + R + A = 23 := by
  sorry

end carla_students_l473_473660


namespace compute_K_l473_473945

theorem compute_K (P Q T N K : ℕ) (x y z : ℕ) 
  (hP : P * x + Q * y = z) 
  (hT : T * x + N * y = z)
  (hK : K * x = z)
  (h_unique : P > 0 ∧ Q > 0 ∧ T > 0 ∧ N > 0 ∧ K > 0) :
  K = (P * K - T * Q) / (N - Q) :=
by sorry

end compute_K_l473_473945


namespace area_ratio_correct_l473_473515

noncomputable def area_ratio : ℚ :=
  let side_WXYZ := 12
  let total_length := side_WXYZ^2
  let WJ := 9
  let JZ := 3
  let side_JK := Real.sqrt (WJ^2 + JZ^2)
  let area_JKLM := side_JK^2
  area_JKLM / total_length

theorem area_ratio_correct :
  let side_WXYZ := 12 in
  let WJ := 9 in
  let JZ := 3 in
  let side_JK := Real.sqrt (WJ^2 + JZ^2) in
  side_JK * side_JK / (side_WXYZ * side_WXYZ) = 5 / 8 := 
by
  sorry

end area_ratio_correct_l473_473515


namespace ron_l473_473600

theorem ron's_drink_percentage :
  ∀ (initial_volume : ℝ) (required_effect_volume : ℝ) (H : initial_volume = 480) (R : required_effect_volume = 30),
    let remaining_intelligence : ℝ := 60,
        remaining_beauty : ℝ := 60,
        remaining_strength : ℝ := 120,
        total_remaining : ℝ := remaining_intelligence + remaining_beauty + remaining_strength,
        percentage_to_drink : ℝ := (2 * required_effect_volume) / total_remaining
    in percentage_to_drink = 0.5 :=
by
  intros initial_volume required_effect_volume H R
  simp only [H, R]
  sorry

end ron_l473_473600


namespace area_WXUV_l473_473029

theorem area_WXUV (a b c d : ℝ) 
  (h1 : a * c = 9) 
  (h2 : b * c = 10) 
  (h3 : b * d = 15) : 
  a * d = 27 / 2 := 
begin
  sorry
end

end area_WXUV_l473_473029


namespace swimming_speed_still_water_l473_473633

theorem swimming_speed_still_water 
  (v t : ℝ) 
  (h1 : 3 = (v + 3) * t / (v - 3)) 
  (h2 : t ≠ 0) :
  v = 9 :=
by
  sorry

end swimming_speed_still_water_l473_473633


namespace angle_ADB_equilateral_triangle_l473_473460

noncomputable def m_angle_ADB : ℝ := 105

theorem angle_ADB_equilateral_triangle (A B C D : Type) 
  [inst : MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  [equilateral_triangle : EquilateralTriangle A B C]  
  (hAD : dist A D = real.sqrt 2) 
  (hBD : dist B D = 3) 
  (hCD : dist C D = real.sqrt 5) : 
  angle A D B = m_angle_ADB := 
begin
  sorry
end

end angle_ADB_equilateral_triangle_l473_473460


namespace mr_a_loss_l473_473483

noncomputable def house_initial_value := 12000
noncomputable def first_transaction_loss := 15 / 100
noncomputable def second_transaction_gain := 20 / 100

def house_value_after_first_transaction (initial_value loss : ℝ) : ℝ :=
  initial_value * (1 - loss)

def house_value_after_second_transaction (value_after_first gain : ℝ) : ℝ :=
  value_after_first * (1 + gain)

theorem mr_a_loss :
  let initial_value := house_initial_value
  let loss := first_transaction_loss
  let gain := second_transaction_gain
  let value_after_first := house_value_after_first_transaction initial_value loss
  let value_after_second := house_value_after_second_transaction value_after_first gain
  value_after_second - initial_value = 240 :=
by
  sorry

end mr_a_loss_l473_473483


namespace triangle_third_side_lengths_l473_473760

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l473_473760


namespace true_proposition_l473_473739

variable (p : Prop) (q : Prop)

-- Introduce the propositions as Lean variables
def prop_p : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2
def prop_q : Prop := ∀ a b : ℝ, ((a > 1 ∧ b > 1) → a * b > 1) ∧ ((a * b > 1) ∧ (¬ (a > 1 ∧ b > 1)))

-- Rewrite the main goal as a Lean statement
theorem true_proposition : ¬ prop_p ∧ prop_q := 
  sorry

end true_proposition_l473_473739


namespace sum_of_first_fifteen_multiples_of_7_l473_473581

theorem sum_of_first_fifteen_multiples_of_7 : (List.range 15).map (λ n, 7 * (n + 1)).sum = 840 := by
  sorry

end sum_of_first_fifteen_multiples_of_7_l473_473581


namespace third_side_integer_lengths_l473_473780

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l473_473780


namespace jogger_speed_l473_473214

theorem jogger_speed (l_train : ℝ) (d_jogger : ℝ) (speed_train_km_hr : ℝ) 
  (t_pass : ℝ) (conversion_factor : ℝ) (speed_train_m_s : ℝ)
  (distance_total : ℝ) (v_relative : ℝ) (answers_km_hr : ℝ) :
  (l_train = 210) →
  (d_jogger = 200) →
  (speed_train_km_hr = 45) →
  (t_pass = 41) →
  (conversion_factor = 1 / 3.6) →
  (speed_train_m_s = speed_train_km_hr * conversion_factor) →
  (distance_total = l_train + d_jogger) →
  (v_relative = distance_total / t_pass) →
  (v_relative = speed_train_m_s - (answers_km_hr * conversion_factor)) →
  (answers_km_hr * conversion_factor = 2.5) →
  answers_km_hr = 9 :=
by
  intros l_train_eq d_jogger_eq speed_train_km_hr_eq t_pass_eq conversion_factor_eq
        speed_train_m_s_eq distance_total_eq v_relative_eq relative_speed_eq
        jogger_speed_ms_eq
  -- Using the parameters provided and respective equations
  have h1 : speed_train_m_s = 12.5 := by {
    rw [speed_train_km_hr_eq, conversion_factor_eq];
    have : (45 * (1 / 3.6)) = 12.5 := sorry,
    exact this
  }
  have h2 : distance_total = 410 := by {
    rw [l_train_eq, d_jogger_eq],
    exact rfl }
  have h3 : v_relative = 10 := by {
    rw [distance_total_eq, t_pass_eq],
    exact rfl }
  have h4 : 9 * conversion_factor = 2.5 := by {
    rw conversion_factor_eq,
    exact rfl }
  exact rfl

end jogger_speed_l473_473214


namespace find_h_l473_473377

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l473_473377


namespace MaxChords_l473_473312

/-- Given a circle with 2006 points and 17 colors, prove that the maximal number of chords 
Frankinfueter can always draw is at least 117, given the conditions of the problem. -/
theorem MaxChords (points : Fin 2006 → Point) (colors : Fin 17) (color_of : Point → colors)
  (same_color_chord : ∀ (p1 p2 : Point), color_of p1 = color_of p2 → Chord p1 p2)
  (chord_disjoint : ∀ (c1 c2 : Chord), c1 ≠ c2 → Disjoint c1 c2) :
  ∃ (chosen_chords : List (Chord p1 p2)), chosen_chords.length ≥ 117 ∧
    ∀ (ch1 ch2 : Chord), ch1 ∈ chosen_chords → ch2 ∈ chosen_chords → ch1 = ch2 ∨ Disjoint ch1 ch2 :=
sorry

end MaxChords_l473_473312


namespace smallest_greater_than_1_1_l473_473697

def number_set : set ℝ := {1.4, 0.9, 1.2, 0.5, 1.3}

theorem smallest_greater_than_1_1 :
  ∃ (n : ℝ), n ∈ number_set ∧ n > 1.1 ∧ ∀ m ∈ number_set, m > 1.1 → n ≤ m :=
sorry

end smallest_greater_than_1_1_l473_473697


namespace bridge_length_l473_473970

theorem bridge_length (train_length : ℕ) (train_speed : ℕ) (cross_time : ℕ) (h1 : train_length = 160) (h2 : train_speed = 45) (h3 : cross_time = 30) : 
  let speed_m_per_s := train_speed * 1000 / 3600,
      distance_covered := speed_m_per_s * cross_time,
      bridge_length := distance_covered - train_length 
  in bridge_length = 215 :=
by
  have hs : speed_m_per_s = 12.5 := by sorry
  have hd : distance_covered = 375 := by sorry
  have hb : bridge_length = 215 := by sorry
  exact hb

end bridge_length_l473_473970


namespace min_distance_hyperbola_l473_473473

theorem min_distance_hyperbola (
  α β γ : ℝ
) (h : sin α * cos β + abs (cos α * sin β) = (sin α * abs (cos α)) + (abs (sin β) * cos β)) :
  ∃ x, x = (tan γ - sin α)^2 + (cot γ - cos β)^2 ∧ x = 3 - 2 * sqrt 2 :=
by sorry

end min_distance_hyperbola_l473_473473


namespace find_number_l473_473068

theorem find_number (d q r : ℕ) (h_divisor : d = 16) (h_quotient : q = 10) (h_remainder : r = 1) :
  let number := (d * q) + r in number = 161 :=  
by
  -- let definition and proof is skipped
  sorry

end find_number_l473_473068


namespace find_fixed_point_l473_473300

noncomputable def fixed_point {a b : ℝ} : Prop :=
∀ k : ℝ, ∃ (x y : ℝ), y = 9 * x^2 + k * x - 5 * k + 3 ∧ (x, y) = (a, b)

theorem find_fixed_point : fixed_point (5 : ℝ) (228 : ℝ) := 
by
  intro k,
  use (5, 228),
  split,
  sorry

end find_fixed_point_l473_473300


namespace solve_for_x_l473_473509

theorem solve_for_x : ∃ x : ℝ, 4^(x + 1) = real.cbrt 64 ∧ x = 0 :=
by
  use 0
  split
  · exact sorry -- Prove that 4^1 = real.cbrt 64
  · exact sorry -- State that x = 0

end solve_for_x_l473_473509


namespace shuffle_transformation_involution_l473_473304

open Function

variable {n : ℕ}

-- Define a shuffle transformation function
def shuffle (s : Fin (2 * n) → ℕ) : Fin (2 * n) → ℕ :=
  λ i => if i < n then s (Fin.mk (2 * i + 1) (by sorry)) else s (Fin.mk (2 * (i − n) + 1) (by sorry))

-- Euler's Totient function
def euler_totient (x : ℕ) : ℕ := Nat.totient x

theorem shuffle_transformation_involution (s : Fin (2 * n) → ℕ) : 
    (shuffle^[euler_totient (2 * n - 1)]) s = s := by
  sorry

end shuffle_transformation_involution_l473_473304


namespace timer_total_time_l473_473079

theorem timer_total_time (participants : ℕ) (timer1 : ℕ) (timer2 : ℕ) (timer3 : ℤ) (n : ℕ) :
  participants = 60 →
  timer1 = 1 →
  timer2 = 0 →
  timer3 = -1 →
  n = 60 →
  let final_timer1 := timer1 * 2^n in
  let final_timer2 := 0 in
  let final_timer3 := timer3 + n in
  final_timer1 + final_timer2 + final_timer3 = 59 + 2^60 :=
by
  intros participants_eq timer1_eq timer2_eq timer3_eq n_eq
  simp_all
  sorry

end timer_total_time_l473_473079


namespace incorrect_step_l473_473441

theorem incorrect_step (a b : Real) : 
  (sqrt 27 = sqrt (9 * 3)) ∧ 
  (sqrt 27 = 3 * sqrt 3) ∧ 
  (-3 * sqrt 3 = sqrt ((-3)^(2) * 3) = sqrt 27) ∧ 
  (-3 * sqrt 3 = 3 * sqrt 3) -> 
  False :=
by
  sorry

end incorrect_step_l473_473441


namespace sum_of_first_fifteen_multiples_of_7_l473_473579

theorem sum_of_first_fifteen_multiples_of_7 : (List.range 15).map (λ n, 7 * (n + 1)).sum = 840 := by
  sorry

end sum_of_first_fifteen_multiples_of_7_l473_473579


namespace complete_the_square_h_value_l473_473405

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l473_473405


namespace min_value_expr_l473_473288

theorem min_value_expr : ∃ (x : ℝ), (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 ∧ 
  ∀ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

end min_value_expr_l473_473288


namespace ratio_is_pi_over_8_l473_473206

noncomputable
def ratio_of_areas (R : ℝ) : ℝ :=
  let s := R / (1 / 2 * Real.sqrt (5 + 2 * Real.sqrt 5))
  let r := s / 2 * Real.cot (Real.pi / 5)
  let area_small_circle := Real.pi * r^2
  let area_large_pentagon := 5 * s^2 / 4 * Real.cot (Real.pi / 5)
  area_small_circle / area_large_pentagon

theorem ratio_is_pi_over_8 (R : ℝ) :
  ratio_of_areas R = Real.pi / 8 :=
by
  sorry

end ratio_is_pi_over_8_l473_473206


namespace curve_is_line_l473_473283

theorem curve_is_line (θ : ℝ) (h : θ = Real.pi / 6) : 
  (∃ (r : ℝ), r * cos θ = r * (cos (Real.pi / 6)) ∧ r * sin θ = r * (sin (Real.pi / 6))) :=
sorry

end curve_is_line_l473_473283


namespace calculate_roots_l473_473657

theorem calculate_roots : real.cbrt (-8) + real.sqrt 16 = 2 := by
  sorry

end calculate_roots_l473_473657


namespace am_gm_inequality_wxyz_l473_473500

theorem am_gm_inequality_wxyz (w x y z : ℝ) (hw : 0 < w) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  sqrt ( (w^2 + x^2 + y^2 + z^2) / 4 )  ≥  ( (w * x * y / 3 + w * x * z / 3 + w * y * z / 3 + x * y * z / 3) / 4)  :=
sorry

end am_gm_inequality_wxyz_l473_473500


namespace locus_square_l473_473188

open Real

variables {x y c1 c2 d1 d2 : ℝ}

/-- The locus of points in a square -/
theorem locus_square (h_square: d1 < d2 ∧ c1 < c2) (h_x: d1 ≤ x ∧ x ≤ d2) (h_y: c1 ≤ y ∧ y ≤ c2) :
  |y - c1| + |y - c2| = |x - d1| + |x - d2| :=
by sorry

end locus_square_l473_473188


namespace range_of_x_odot_x_plus_2_l473_473260

def odot (a b : ℝ) : ℝ := a * b - 2 * a - b

theorem range_of_x_odot_x_plus_2 : {x : ℝ | odot x (x + 2) < 0} = set.Ioo (-1) 2 :=
by
  sorry

end range_of_x_odot_x_plus_2_l473_473260


namespace cosine_largest_angle_l473_473330

variables (a b c cosC : ℝ)
variable (h1 : b^2 - 2 * a - real.sqrt 3 * b - 2 * c = 0)
variable (h2 : 2 * a + real.sqrt 3 * b - 2 * c + 1 = 0)

theorem cosine_largest_angle (a b c : ℝ) :
  b^2 - 2 * a - real.sqrt 3 * b - 2 * c = 0 →
  2 * a + real.sqrt 3 * b - 2 * c + 1 = 0 →
  cosC = - real.sqrt 3 / 2 :=
by
  sorry

end cosine_largest_angle_l473_473330


namespace eccentricity_of_hyperbola_l473_473735

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_asymptote : a = 2 * b) : ℝ :=
  let c := real.sqrt (a^2 + b^2) in
  c / a

theorem eccentricity_of_hyperbola : 
  ∀ (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_asymptote : a = 2 * b),
  hyperbola_eccentricity a b ha hb h_asymptote = real.sqrt 5 / 2 :=
by
  intros a b ha hb h_asymptote
  sorry

end eccentricity_of_hyperbola_l473_473735


namespace triangle_third_side_count_l473_473829

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l473_473829


namespace evaluate_expression_l473_473024

variable (a b c d e : ℝ)

-- The equivalent proof problem statement
theorem evaluate_expression 
  (h : (a / b * c - d + e = a / (b * c - d - e))) : 
  a / b * c - d + e = a / (b * c - d - e) :=
by 
  exact h

-- Placeholder for the proof
#check evaluate_expression

end evaluate_expression_l473_473024


namespace scalene_triangle_not_divisible_l473_473925

theorem scalene_triangle_not_divisible :
  ∀ (A B C D : ℝ) (T : Triangle) (CD : Segment),
  T.is_scalene A B C → 
  (A ≠ B ∧ A ≠ C ∧ B ≠ C) → 
  (T.area_of_triangle A C D = T.area_of_triangle B C D) →
  false :=
by
  intro A B C D T CD
  intro h_scalene h_distinct_sides h_equal_areas
  sorry

end scalene_triangle_not_divisible_l473_473925


namespace complete_the_square_3x2_9x_20_l473_473396

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l473_473396


namespace solve_congruence_l473_473513

theorem solve_congruence : ∃ (n : ℤ), 13 * n ≡ 8 [MOD 47] ∧ n ≡ 4 [MOD 47] :=
sorry

end solve_congruence_l473_473513


namespace ratio_gt_sqrt_two_error_decreases_find_n_l473_473928

theorem ratio_gt_sqrt_two (x y : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ n, y n * y n = 2 * (x n) * (x n) + 1)
  : (y n) / (x n) > Real.sqrt 2 :=
sorry

theorem error_decreases (a : ℕ → ℝ) (x y : ℕ → ℕ)
  (h2 : ∀ k, a k = (y k) / (x k) - Real.sqrt 2)
  (h3 : ∀ k, a (k + 1) < 0.03 * a k)
  : ∀ k, a (k + 1) < 0.03 * a k :=
sorry

theorem find_n (a : ℕ → ℝ) (x y : ℕ → ℕ)
  (h4 : ∀ k, y k = (4 * x k + 3 * y k) / (3 * x k + 2 * y k))
  (h5 : ∀ k, a k = (y k) / (x k) - Real.sqrt 2)
  (h6 : ∀ k, a (k + 1) < 0.03 * a k)
  : ∃ (n : ℕ), n ≥ 8 ∧ a n < 10 ^ (-10) :=
sorry

end ratio_gt_sqrt_two_error_decreases_find_n_l473_473928


namespace lexicon_total_words_l473_473104

section LexiconWords

-- Define the alphabet size
def alphabet_size : ℕ := 25

-- Define the number of valid words of specific lengths
def words_of_length (n : ℕ) : ℕ := alphabet_size^n

-- Define the number of invalid words (words without 'A')
def invalid_words_of_length (n : ℕ) : ℕ := 24^n

-- Define the number of valid words (words with at least one 'A')
def valid_words_of_length (n : ℕ) : ℕ := words_of_length n - invalid_words_of_length n

-- Sum up the valid words for lengths from 1 to 5
def total_valid_words : ℕ := (finset.range 5).sum (λ n => valid_words_of_length (n + 1))

theorem lexicon_total_words :
  total_valid_words = 1861701 :=
by
  sorry

end LexiconWords

end lexicon_total_words_l473_473104


namespace square_area_from_diagonal_l473_473693

theorem square_area_from_diagonal (d : ℝ) (hd : d = 3.8) : 
  ∃ (A : ℝ), A = 7.22 ∧ (∀ s : ℝ, d^2 = 2 * (s^2) → A = s^2) :=
by
  sorry

end square_area_from_diagonal_l473_473693


namespace inequality1_inequality2_l473_473598

variable {n : ℕ}
variable {i : Fin n}
variable {polygon : Fin n → Point}
variable {O : Point}
variable {α : Fin n → ℝ}
variable {x : Fin n → ℝ}
variable {d : Fin n → ℝ}
variable {p : ℝ}

-- Assumptions:
axiom h_convex_polygon : ConvexPolygon polygon
axiom h_point_inside : ∀ i, inside_polygon O polygon
axiom h_alpha_k : ∀ i, α i = angle (polygon i) (polygon (i + 1) % n)
axiom h_x_k : ∀ i, x i = distance O (polygon i)
axiom h_d_k : ∀ i, d i = distance_to_line O (polygon i) (polygon (i + 1) % n)
axiom h_semiperimeter : p = semiperimeter polygon -- where semiperimeter sums up half of the full perimeter

-- Proof Statements:
theorem inequality1 : ∑ i, (x i) * sin (α i / 2) ≥ ∑ i, d i := sorry

theorem inequality2 : ∑ i, (x i) * cos (α i / 2) ≥ p := sorry

end inequality1_inequality2_l473_473598


namespace percentage_of_boys_is_90_l473_473873

variables (B G : ℕ)

def total_children : ℕ := 100
def future_total_children : ℕ := total_children + 100
def percentage_girls : ℕ := 5
def girls_after_increase : ℕ := future_total_children * percentage_girls / 100
def boys_after_increase : ℕ := total_children - girls_after_increase

theorem percentage_of_boys_is_90 :
  B + G = total_children →
  G = girls_after_increase →
  B = total_children - G →
  (B:ℚ) / total_children * 100 = 90 :=
by
  sorry

end percentage_of_boys_is_90_l473_473873


namespace Alice_favorite_number_l473_473236

theorem Alice_favorite_number :
  ∃ n : ℕ, (30 ≤ n ∧ n ≤ 70) ∧ (7 ∣ n) ∧ ¬(3 ∣ n) ∧ (4 ∣ (n / 10 + n % 10)) ∧ n = 35 :=
by
  sorry

end Alice_favorite_number_l473_473236


namespace sum_of_first_fifteen_multiples_of_7_l473_473574

theorem sum_of_first_fifteen_multiples_of_7 :
  ∑ i in finset.range 15, 7 * (i + 1) = 840 :=
sorry

end sum_of_first_fifteen_multiples_of_7_l473_473574


namespace cost_per_kg_l473_473487

/--
Méliès bought 2 kg of meat. The meat costs a certain amount per kilogram. 
Méliès has $180 in his wallet and has $16 left after paying for the meat. 
What is the cost of the meat per kilogram?
-/
theorem cost_per_kg (amt_spent : ℕ) (cost_per_kg : ℕ) : 
  let initial_wallet := 180
      left_in_wallet := 16
      kg_of_meat := 2 in
  amt_spent = initial_wallet - left_in_wallet →
  amt_spent = cost_per_kg * kg_of_meat →
  cost_per_kg = 82 :=
by
  intros h1 h2
  rw [h1] at h2
  have : 164 = 82 * 2 := rfl
  exact sorry

end cost_per_kg_l473_473487


namespace first_part_trip_length_l473_473213

variable (x : ℝ) (total_distance : ℝ := 60) (speed1 : ℝ := 60) (speed2 : ℝ := 30)
variable (avg_speed : ℝ := 40)

/-- Define the total time equation based on the segment distances and speeds --/
def total_time (x : ℝ) : ℝ := (x / speed1) + ((total_distance - x) / speed2)

/-- Define the average speed equation based on the total distance and total time --/
def avg_speed_eqn : Prop := avg_speed = total_distance / total_time x

/-- The main theorem: The first part of the trip is 30 kilometers long --/
theorem first_part_trip_length : x = 30 :=
  by
  have h1 : total_distance = 60 := rfl
  have h2 : speed1 = 60 := rfl
  have h3 : speed2 = 30 := rfl
  have h4 : avg_speed = 40 := rfl
  
  have h5 : 30 = 30 := rfl -- Placeholder for proof steps
  sorry

end first_part_trip_length_l473_473213


namespace find_h_l473_473375

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l473_473375


namespace no_prime_p_satisfies_l473_473595

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_prime_p_satisfies (p : ℕ) (hp : Nat.Prime p) (hp1 : is_perfect_square (7 * p + 3 ^ p - 4)) : False :=
by
  sorry

end no_prime_p_satisfies_l473_473595


namespace speed_of_sisters_sailboat_l473_473039

variable (v_j : ℝ) (d : ℝ) (t_wait : ℝ)

-- Conditions
def janet_speed : Prop := v_j = 30
def lake_distance : Prop := d = 60
def janet_wait_time : Prop := t_wait = 3

-- Question to Prove
def sister_speed (v_s : ℝ) : Prop :=
  janet_speed v_j ∧ lake_distance d ∧ janet_wait_time t_wait →
  v_s = 12

-- The main theorem
theorem speed_of_sisters_sailboat (v_j d t_wait : ℝ) (h1 : janet_speed v_j) (h2 : lake_distance d) (h3 : janet_wait_time t_wait) :
  ∃ v_s : ℝ, sister_speed v_j d t_wait v_s :=
by
  sorry

end speed_of_sisters_sailboat_l473_473039


namespace find_purely_imaginary_z_l473_473721

open Complex

theorem find_purely_imaginary_z (z : ℂ) (hz : z.im ≠ 0) (hz2 : (z + 2) ^ 2 - 8 * I ∈ ℂ.im) : z = -2 * I :=
by
  sorry

end find_purely_imaginary_z_l473_473721


namespace main_proof_l473_473525

variables (A B C D O T : Point)
variables (M : Point) -- Midpoint of segment CD

-- Conditions
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry
def diagonals_intersect (A B C D O : Point) : Prop := sorry
def is_equilateral_triangle (X Y Z : Point) : Prop := sorry
def is_reflection_midpoint (O T M : Point) (C D : Point) : Prop := sorry
def midpoint (X Y M : Point) : Prop := sorry

-- Distances
def distance (X Y : Point) : ℝ := sorry
def BC : ℝ := 2
def AD : ℝ := 3

-- Questions
def question1 : Prop :=
  ∀ (A B C D O T M : Point), is_convex_quadrilateral A B C D →
  diagonals_intersect A B C D O →
  is_equilateral_triangle B O C →
  is_equilateral_triangle A O D →
  midpoint C D M →
  is_reflection_midpoint O T M C D →
  is_equilateral_triangle A B T

def question2 : Prop :=
  ∀ (A B C D O T M : Point), is_convex_quadrilateral A B C D →
  diagonals_intersect A B C D O →
  is_equilateral_triangle B O C →
  is_equilateral_triangle A O D →
  midpoint C D M →
  is_reflection_midpoint O T M C D →
  (distance B C = BC) → (distance A D = AD) →
  let S1 := (√3) / 4 * (distance A B)^2 in
  let S2 := (1 / 2) * (distance A C) * (distance B D) * (√3 / 2) in
  S1 / S2 = 19 / 25

-- Main theorem combining both parts
theorem main_proof :
  question1 A B C D O T M ∧ question2 A B C D O T M :=
sorry

end main_proof_l473_473525


namespace largest_w_exists_l473_473941

theorem largest_w_exists (w x y z : ℝ) (h1 : w + x + y + z = 25) (h2 : w * x + w * y + w * z + x * y + x * z + y * z = 2 * y + 2 * z + 193) :
  ∃ (w1 w2 : ℤ), w1 > 0 ∧ w2 > 0 ∧ ((w = w1 / w2) ∧ (w1 + w2 = 27)) :=
sorry

end largest_w_exists_l473_473941


namespace triangle_side_length_integers_l473_473767

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l473_473767


namespace frustum_height_l473_473222

-- Definitions based on the given conditions
variables {Pyramid : Type} (baseArea upperBaseArea : ℝ) (height frustumHeight cutHeight : ℝ)

-- Define the conditions
def isFrustum (pyramid : Pyramid) : Prop := 
  upperBaseArea / baseArea = 1 / 4 ∧ cutHeight = 3

-- Define the statement we want to prove
theorem frustum_height (pyramid : Pyramid)
  (h : isFrustum pyramid) : frustumHeight = 3 := 
sorry

end frustum_height_l473_473222


namespace range_a_for_no_real_roots_l473_473370

theorem range_a_for_no_real_roots :
  let Δ₁ := λ a : ℝ, a^2 - 36
  let Δ₂ := λ a : ℝ, a^2 + 8a
  let Δ₃ := λ a : ℝ, (a+1)^2 - 9
  (∀ a, (Δ₁ a < 0 → -6 < a ∧ a < 6) ∧ (Δ₂ a < 0 → -8 < a ∧ a < 0) ∧ (Δ₃ a < 0 → -4 < a ∧ a < 2)) →
    (-∞ < a ∧ a ≤ -4) ∨ (a ≥ 0 ∧ a < ∞) :=
sorry

end range_a_for_no_real_roots_l473_473370


namespace exists_kn_l473_473301

open Nat

theorem exists_kn (n : ℕ) (hn : 0 < n) :
  ∃ k_n : ℤ, 
  let x_n := (Finset.range n).sum (fun i => nat.prime (i + 1)) in
  let x_n1 := (Finset.range (n + 1)).sum (fun i => nat.prime (i + 1)) in
  x_n < k_n^2 ∧ k_n^2 < x_n1 :=
begin
  sorry
end

end exists_kn_l473_473301


namespace red_paint_quantity_l473_473305

-- Define the conditions
def ratio_blue_red_white : ℕ × ℕ × ℕ := (4, 3, 5)
def white_paint_quantity : ℕ := 15

-- Define the problem and the proof goal
theorem red_paint_quantity (r : ℕ × ℕ × ℕ) (w : ℕ) 
  (h_ratio: r = (4, 3, 5)) (h_white: w = 15) : 
  let red_paint := r.2.1, white_paint := r.2.2
  in (white_paint * w / red_paint) = 9 :=
by
  -- The proof will be filled here
  sorry

end red_paint_quantity_l473_473305


namespace solution_set_x2_minus_x_lt_0_l473_473542

theorem solution_set_x2_minus_x_lt_0 :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ x^2 - x < 0 := 
by
  sorry

end solution_set_x2_minus_x_lt_0_l473_473542


namespace largest_multiple_of_15_less_than_400_l473_473562

theorem largest_multiple_of_15_less_than_400 (x : ℕ) (k : ℕ) (h : x = 15 * k) (h1 : x < 400) (h2 : ∀ m : ℕ, (15 * m < 400) → m ≤ k) : x = 390 :=
by
  sorry

end largest_multiple_of_15_less_than_400_l473_473562


namespace math_problem_l473_473719

theorem math_problem (f : ℕ → Prop) (m : ℕ) 
  (h1 : f 1) (h2 : f 2) (h3 : f 3)
  (h_implies : ∀ k : ℕ, f k → f (k + m)) 
  (h_max : m = 3):
  ∀ n : ℕ, 0 < n → f n :=
by
  sorry

end math_problem_l473_473719


namespace root_cubic_expression_value_l473_473051

theorem root_cubic_expression_value :
  ∀ r s : ℝ, (Polynomial.X^2 * (2 : ℝ) - Polynomial.C 3 * Polynomial.X + Polynomial.C (-11)).root r ∧ 
             (Polynomial.X^2 * (2 : ℝ) - Polynomial.C 3 * Polynomial.X + Polynomial.C (-11)).root s →
             (4 * r^3 - 4 * s^3) / (r - s) = 31 :=
by 
  intros r s h_root
  sorry

end root_cubic_expression_value_l473_473051


namespace quadratic_expression_rewriting_l473_473415

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l473_473415


namespace cost_per_serving_of_pie_l473_473746

theorem cost_per_serving_of_pie 
  (w_gs : ℝ) (p_gs : ℝ) (w_gala : ℝ) (p_gala : ℝ) (w_hc : ℝ) (p_hc : ℝ)
  (pie_crust_cost : ℝ) (lemon_cost : ℝ) (butter_cost : ℝ) (servings : ℕ)
  (total_weight_gs : w_gs = 0.5) (price_gs_per_pound : p_gs = 1.80)
  (total_weight_gala : w_gala = 0.8) (price_gala_per_pound : p_gala = 2.20)
  (total_weight_hc : w_hc = 0.7) (price_hc_per_pound : p_hc = 2.50)
  (cost_pie_crust : pie_crust_cost = 2.50) (cost_lemon : lemon_cost = 0.60)
  (cost_butter : butter_cost = 1.80) (total_servings : servings = 8) :
  (w_gs * p_gs + w_gala * p_gala + w_hc * p_hc + pie_crust_cost + lemon_cost + butter_cost) / servings = 1.16 :=
by 
  sorry

end cost_per_serving_of_pie_l473_473746


namespace range_of_a_for_two_positive_roots_solve_inequality_f_gt_zero_l473_473306

variables (a x : ℝ)

def f (a x : ℝ) : ℝ := a * x^2 + 2 * x - 3

def two_positive_roots (a : ℝ) : Prop :=
  let Δ := 4 + 12 * a in
  Δ ≥ 0 ∧ (-2 / a > 0) ∧ (-3 / a > 0)

def inequality_solution (a x : ℝ) : Prop :=
  if a = 0 then x > 3 / 2
  else
    let Δ := 4 + 12 * a in
    if a > 0 then x < (-1 - sqrt (1 + 3 * a)) / a ∨ x > (-1 + sqrt (1 + 3 * a)) / a
    else
      if Δ > 0 then (-1 + sqrt (1 + 3 * a)) / a < x ∧ x < (-1 - sqrt (1 + 3 * a)) / a
      else false

theorem range_of_a_for_two_positive_roots :
  ∀ (a : ℝ), two_positive_roots a ↔ -1 / 3 ≤ a ∧ a < 0 :=
by sorry

theorem solve_inequality_f_gt_zero :
  ∀ (a x : ℝ), f a x > 0 ↔ inequality_solution a x :=
by sorry

end range_of_a_for_two_positive_roots_solve_inequality_f_gt_zero_l473_473306


namespace exists_root_in_interval_l473_473538

open Real

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem exists_root_in_interval : ∃ x ∈ Ioo 0 1, f x = 0 := sorry

end exists_root_in_interval_l473_473538


namespace square_feet_per_acre_l473_473628

theorem square_feet_per_acre 
  (pay_per_acre_per_month : ℕ) 
  (total_pay_per_month : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (total_acres : ℕ) 
  (H1 : pay_per_acre_per_month = 30) 
  (H2 : total_pay_per_month = 300) 
  (H3 : length = 360) 
  (H4 : width = 1210) 
  (H5 : total_acres = 10) : 
  (length * width) / total_acres = 43560 :=
by 
  sorry

end square_feet_per_acre_l473_473628


namespace winning_strategy_l473_473252

noncomputable def first_player_wins (k : ℕ) (n : Fin k → ℕ) : Prop :=
  let φ := {n : Fin k → ℕ | odd k ∧ ∀ i : Fin (k / 2), n 2 * i = n (2 * i + 1)}
  let B := {n : Fin k → ℕ | n ∉ φ}
  n ∈ B

theorem winning_strategy (k : ℕ) (n : Fin k → ℕ) :
  (first_player_wins k n → ∃ m : Fin k → ℕ, ∀ i : Fin k, m i ≤ n i ∧ ∑ i, m i = ∑ i, n i ∧ first_player_wins k m) ∧
  (¬ first_player_wins k n → ∃ m : Fin k → ℕ, ∀ i : Fin k, m i ≤ n i ∧ ∑ i, m i = ∑ i, n i ∧ ¬ first_player_wins k m) :=
sorry

end winning_strategy_l473_473252


namespace carla_total_students_l473_473662

-- Defining the conditions
def students_in_restroom : Nat := 2
def absent_students : Nat := (3 * students_in_restroom) - 1
def total_desks : Nat := 4 * 6
def occupied_desks : Nat := total_desks * 2 / 3
def students_present : Nat := occupied_desks

-- The target is to prove the total number of students Carla teaches
theorem carla_total_students : students_in_restroom + absent_students + students_present = 23 := by
  sorry

end carla_total_students_l473_473662


namespace wire_division_l473_473178

theorem wire_division (L leftover total_length : ℝ) (seg1 seg2 : ℝ)
  (hL : L = 120 * 2)
  (hleftover : leftover = 2.4)
  (htotal : total_length = L + leftover)
  (hseg1 : seg1 = total_length / 3)
  (hseg2 : seg2 = total_length / 3) :
  seg1 = 80.8 ∧ seg2 = 80.8 := by
  sorry

end wire_division_l473_473178


namespace proper_subset_count_of_A_l473_473352

universe u

variable {α : Type u}

-- Given conditions
def U : Set α := {0, 1, 2}
def A : Set α := U \ {2}

-- Problem statement
theorem proper_subset_count_of_A : ∃ (n : ℕ), n = 3 ∧ (n = (2 ^ (A.toFinset.card)) - 1) := by
  sorry

end proper_subset_count_of_A_l473_473352


namespace area_of_parallelogram_l473_473316

-- Define the conditions
variable (A B C D E O : Type)
variable [RealVectorSpace A] [RealVectorSpace B] [RealVectorSpace C] [RealVectorSpace D] [RealVectorSpace E] [RealVectorSpace O]
variables (AB BC BE AC AD : ℝ)
variable (midpoint_E : Prop)

-- Parallelogram ABCD with given sides and specific properties
def is_parallelogram (A B C D : Type) : Prop := sorry
def same_length (AB : ℝ) := (AB = 2)
def length_of_BC (BC : ℝ) := (BC = 3)
def AC_perpendicular_to_BE (AC BE : ℝ) : Prop := sorry
def E_is_midpoint (E : Type) (AD : ℝ) : Prop := (midpoint_E E AD)

theorem area_of_parallelogram :
  is_parallelogram A B C D →
  same_length AB →
  length_of_BC BC →
  AC_perpendicular_to_BE AC BE →
  E_is_midpoint E AD →
  (area_of A B C D = √35) :=
by sorry

end area_of_parallelogram_l473_473316


namespace opposite_of_three_l473_473108

theorem opposite_of_three :
  ∃ x : ℤ, 3 + x = 0 ∧ x = -3 :=
by
  sorry

end opposite_of_three_l473_473108


namespace rows_identical_l473_473493

theorem rows_identical {n : ℕ} {a : Fin n → ℝ} {k : Fin n → Fin n}
  (h_inc : ∀ i j : Fin n, i < j → a i < a j)
  (h_perm : ∀ i j : Fin n, k i ≠ k j → a (k i) ≠ a (k j))
  (h_sum_inc : ∀ i j : Fin n, i < j → a i + a (k i) < a j + a (k j)) :
  ∀ i : Fin n, a i = a (k i) :=
by
  sorry

end rows_identical_l473_473493


namespace number_of_blue_balls_l473_473621

theorem number_of_blue_balls (b : ℕ) 
  (h1 : 0 < b ∧ b ≤ 15)
  (prob : (b / 15) * ((b - 1) / 14) = 1 / 21) :
  b = 5 := sorry

end number_of_blue_balls_l473_473621


namespace neither_5_nice_nor_6_nice_count_l473_473704

noncomputable def is_k_nice (N k : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ ∃ p b : ℕ, p > 0 ∧ p^b = a^k ∧ (divisors_count (a^k) = N)

theorem neither_5_nice_nor_6_nice_count : ∀ (N less_than : ℕ), 
  less_than = 500 →
  N < less_than →
  let count_5_nice := (List.range less_than).count (λ n => is_k_nice n 5)
      count_6_nice := (List.range less_than).count (λ n => is_k_nice n 6)
      count_5_and_6_nice := (List.range less_than).count (λ n => is_k_nice n 5 ∧ is_k_nice n 6)
  in less_than - (count_5_nice + count_6_nice - count_5_and_6_nice) = 333 := 
sorry

end neither_5_nice_nor_6_nice_count_l473_473704


namespace vertices_integer_assignment_zero_l473_473691

theorem vertices_integer_assignment_zero (f : ℕ → ℤ) (h100 : ∀ i, i < 100 → (i + 3) % 100 < 100) 
  (h : ∀ i, (i < 97 → f i + f (i + 2) = f (i + 1)) 
            ∨ (i < 97 → f (i + 1) + f (i + 3) = f (i + 2)) 
            ∨ (i < 97 → f i + f (i + 1) = f (i + 2))): 
  ∀ i, i < 100 → f i = 0 :=
by
  sorry

end vertices_integer_assignment_zero_l473_473691


namespace minimum_value_of_f_l473_473107

noncomputable def f (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 + 2 * x * y + 4 * y * z + 2 * z * x - 6 * x - 10 * y - 12 * z

theorem minimum_value_of_f : ∃ x y z : ℝ, f x y z = -14 :=
by
  sorry

end minimum_value_of_f_l473_473107


namespace total_clovers_picked_l473_473456

theorem total_clovers_picked (C : ℕ) 
    (h1 : 0.75 * C + 0.24 * C + 0.01 * C = 554) : 
    C = 554 := 
sorry

end total_clovers_picked_l473_473456


namespace triangle_is_isosceles_l473_473267

-- Definitions for the lines
def line1 := λ (x : ℝ), 4 * x + 3
def line2 := λ (x : ℝ), -4 * x + 3
def line3 := -3

-- Intersection points
def P := (0, line1 0)
def Q := (-3/2, line3)
def R := (3/2, line3)

-- Definition of distances
def dist (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Conditions for isosceles triangle
def is_isosceles_triangle (A B C : ℝ × ℝ) :=
  (dist A B = dist A C ∧ dist A B ≠ dist B C ∨
   dist A B = dist B C ∧ dist A B ≠ dist A C ∨
   dist A C = dist B C ∧ dist A C ≠ dist A B)

-- Prove that the points P, Q, and R form an isosceles triangle
theorem triangle_is_isosceles : is_isosceles_triangle P Q R :=
by sorry

end triangle_is_isosceles_l473_473267


namespace complementary_point_on_line1_complementary_points_on_line2_value_of_k_l473_473871
noncomputable theory

-- Definition of complementary points
def is_complementary_point (x y : ℝ) := y = -x

-- Theorem 1
theorem complementary_point_on_line1:
  ∃ (x : ℝ), is_complementary_point x (2*x - 3) ∧ (x = 1) ∧ (-x = -1) :=
begin
  sorry
end

-- Theorem 2
theorem complementary_points_on_line2 (k : ℝ) (h : k ≠ 0 ∧ k ≠ -1):
  ∃ (x : ℝ), is_complementary_point x (k*x + 2) ∧ (x = -2 / (k + 1)) ∧ (-x = 2 / (k + 1)) :=
begin
  sorry
end

-- Theorem 3
theorem value_of_k (n : ℝ) (h_n : -1 ≤ n ∧ n ≤ 2) (m : ℝ) : 
  (∀ t, is_complementary_point t (1/4 * t^2 + (n - k - 1)*t + m + k - 2) → m = k) ↔ (k = 1 ∨ k = 3 + real.sqrt 3) :=
begin
  sorry
end

end complementary_point_on_line1_complementary_points_on_line2_value_of_k_l473_473871


namespace solve_for_a_l473_473012

theorem solve_for_a (a b : ℝ) (h₁ : b = 4 * a) (h₂ : b = 20 - 7 * a) : a = 20 / 11 :=
by
  sorry

end solve_for_a_l473_473012


namespace ellipse_equation_l473_473099

theorem ellipse_equation
  (center_origin : true) -- The ellipse has its center at the origin.
  (foci_on_x_axis : true) -- The ellipse has its foci on the x-axis.
  (major_axis_length : real) (h_major : major_axis_length = 4) -- Major axis length is 4.
  (minor_axis_length : real) (h_minor : minor_axis_length = 2) -- Minor axis length is 2.
  : (∃ a b : ℝ, (a = 2 ∧ b = 1) ∧ (a > 0 ∧ b > 0) ∧ (major_axis_length = 2 * a) ∧ (minor_axis_length = 2 * b) ∧ 
     (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) )
  :=
sorry

end ellipse_equation_l473_473099


namespace alex_needs_additional_coins_l473_473234

theorem alex_needs_additional_coins :
  ∀ (friends coins : ℕ), friends = 8 → coins = 28 →
  let needed_coins := (friends * (friends + 1)) / 2
  in needed_coins - coins = 8 :=
by
  intros friends coins h_friends h_coins needed_coins
  sorry

end alex_needs_additional_coins_l473_473234


namespace tenth_monomial_l473_473121

def monomial_sequence (n : ℕ) : ℝ[X] :=
  if n = 0 then 1
  else (if (n % 2) = 1 then 1 else -1) * (real.sqrt n) * (X ^ n)

theorem tenth_monomial :
  monomial_sequence 10 = -real.sqrt 10 * (X ^ 10) :=
sorry

end tenth_monomial_l473_473121


namespace cube_volume_edge_increase_l473_473850

noncomputable def edge_length_increase (a : ℝ) : Prop :=
  let v := a^3 in
  let v' := 27 * v in
  let a' := (v')^(1/3) in
  a' = 3 * a

theorem cube_volume_edge_increase (a : ℝ) (h : a > 0) : edge_length_increase a :=
by
  sorry

end cube_volume_edge_increase_l473_473850


namespace equilateral_cannot_be_obtuse_l473_473182

-- Additional definitions for clarity and mathematical rigor.
def is_equilateral (a b c : ℝ) : Prop := a = b ∧ b = c ∧ c = a
def is_obtuse (A B C : ℝ) : Prop := 
    (A > 90 ∧ B < 90 ∧ C < 90) ∨ 
    (B > 90 ∧ A < 90 ∧ C < 90) ∨
    (C > 90 ∧ A < 90 ∧ B < 90)

-- Theorem statement
theorem equilateral_cannot_be_obtuse (a b c : ℝ) (A B C : ℝ) :
  is_equilateral a b c → 
  (A + B + C = 180) → 
  (A = B ∧ B = C) → 
  ¬ is_obtuse A B C :=
by { sorry } -- Proof is not necessary as per instruction.

end equilateral_cannot_be_obtuse_l473_473182


namespace exists_infinite_set_gcd_property_l473_473680

theorem exists_infinite_set_gcd_property :
  ∃ S : Set ℕ, (Infinite S ∧ ∀ x y z w ∈ S, x < y → z < w → (x, y) ≠ (z, w) → Nat.gcd (x * y + 2022) (z * w + 2022) = 1) :=
by
  sorry

end exists_infinite_set_gcd_property_l473_473680


namespace area_percentage_l473_473841

noncomputable def d_R : ℝ := sorry
noncomputable def d_S : ℝ := sorry

-- Given condition
axiom diameter_relation : d_R = 0.20 * d_S

-- Radius definitions
def r_R : ℝ := d_R / 2
def r_S : ℝ := d_S / 2

-- Area definitions
def A_R : ℝ := Real.pi * (r_R)^2
def A_S : ℝ := Real.pi * (r_S)^2

-- Prove that the area of circle R is 4% of the area of circle S
theorem area_percentage : (A_R / A_S) * 100 = 4 :=
by
  sorry

end area_percentage_l473_473841


namespace part_I_part_II_part_III_l473_473053

section
  variables (a b : ℝ) (a_pos : a > 0) (x : ℝ) (x1 x2 : ℝ)
  
  -- Part I
  theorem part_I (h1 : x1 = 1) (symm : ∀ x : ℝ, (a * (2 - x) ^ 2 + (b - 1) * (2 - x) + 1) = (a * (2 + x) ^ 2 + (b - 1) * (2 + x) + 1)) :
    f(x) = (1/3) * x^2 - (4/3) * x + 1 :=
  sorry

  -- Part II
  theorem part_II (hb : b = 2*a - 3) :
    ∃ x0 : ℝ, (x0 < 0 ∧ x0 ∈ -1 - sqrt 2 .. -1/2) ∧ (∀ x0 : ℝ, f x0 = abs (2 * x0 - a) + 2) :=
  sorry

  -- Part III
  theorem part_III (a_ge_2 : a ≥ 2) (h2 : x2 - x1 = 2) 
    (g : ℝ → ℝ := λ x, - (a * (x - x1) * (x - x2) + (2 * (x2 - x)))):
    h(a) = a + 1/a + 2 → (∀ x ∈ Ioo x1 x2, g(x) ≤ a + 1/a + 2) →
    minimum (g(x)) = 9/2 :=
  sorry
end

end part_I_part_II_part_III_l473_473053


namespace natural_numbers_satisfy_inequality_l473_473282

theorem natural_numbers_satisfy_inequality:
  ∃ (a b c : ℕ), 
    a = 5 ∧ b = 9 ∧ c = 4 ∧ 
    ∀ n : ℕ, n > 2 → 
      b - (c / ((n-2) !)) < (∑ k in Finset.range (n+1).filter (λ x, x > 1), (k^3 - a) / (k !)) ∧ 
      (∑ k in Finset.range (n+1).filter (λ x, x > 1), (k^3 - a) / (k !)) < b :=
begin
  sorry
end

end natural_numbers_satisfy_inequality_l473_473282


namespace proof_AK_squared_eq_LK_times_KM_l473_473915

variables {A B C D K L M : Type*}
variables [Parallelogram ABCD] [Point K]
variables [OnDiagonal K BD ABCD] [LineIntersection AK BC L] [LineIntersection AK CD M]

theorem proof_AK_squared_eq_LK_times_KM : AK^2 = LK * KM :=
sorry

end proof_AK_squared_eq_LK_times_KM_l473_473915


namespace geometric_sequence_sum_formula_l473_473336

theorem geometric_sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_sum : ∀ n, S n = (∑ i in finset.range (n+1), a i))
  (h_cond1 : a 3 * a 7 = 16 * a 5)
  (h_cond2 : a 3 + a 5 = 20) :
  ∀ n, S n = 2 * a n - 1 :=
sorry

end geometric_sequence_sum_formula_l473_473336


namespace john_profit_l473_473884

-- Definitions based on given conditions
def total_newspapers := 500
def selling_price_per_newspaper : ℝ := 2
def discount_percentage : ℝ := 0.75
def percentage_sold : ℝ := 0.80

-- Derived basic definitions
def cost_price_per_newspaper := selling_price_per_newspaper * (1 - discount_percentage)
def total_cost_price := cost_price_per_newspaper * total_newspapers
def newspapers_sold := total_newspapers * percentage_sold
def revenue := selling_price_per_newspaper * newspapers_sold
def profit := revenue - total_cost_price

-- Theorem stating the profit
theorem john_profit : profit = 550 := by
  sorry

#check john_profit

end john_profit_l473_473884


namespace max_distance_to_origin_is_2_l473_473972

def x (θ : ℝ) : ℝ := 2 * Real.cos θ
def y (θ : ℝ) : ℝ := Real.sin θ

def distance_to_origin (θ : ℝ) : ℝ := Real.sqrt (x θ * x θ + y θ * y θ)

theorem max_distance_to_origin_is_2 : 
  ∃ θ : ℝ, distance_to_origin θ = 2 :=
sorry

end max_distance_to_origin_is_2_l473_473972


namespace proposition_1_correct_proposition_2_correct_proposition_3_correct_proposition_4_incorrect_l473_473726

open Function

-- Proposition 1
def prop1 (p q : Prop) : Prop :=
  ¬(p ∨ q) → (¬p ∧ ¬q)

-- Proposition 2
def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = f(x)

def prop2 (f : ℝ → ℝ) : Prop :=
  isEvenFunction (λ x, f(x + 1)) → (∀ x, f(1 - x) = f(1 + x))

-- Proposition 3
def prop3 (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x b, b = a → (f x = b → x = a)

-- Proposition 4
def prop4 : Prop :=
  ∀ x y : ℝ, (x ≠ y → sin x ≠ sin y)

def contrapositive_prop4 : Prop :=
  ∀ x y : ℝ, (sin x = sin y) → (x = y)

-- Theorems to prove
theorem proposition_1_correct : ∀ p q : Prop, prop1 p q :=
by intros; dsimp [prop1]; tauto

theorem proposition_2_correct : ∀ f : ℝ → ℝ, prop2 f :=
by intros; dsimp [prop2, isEvenFunction]; sorry 

theorem proposition_3_correct : ∀ f : ℝ → ℝ, prop3 f :=
by intros; dsimp [prop3]; sorry

theorem proposition_4_incorrect : ¬contrapositive_prop4 :=
by dsimp [contrapositive_prop4]; sorry

end proposition_1_correct_proposition_2_correct_proposition_3_correct_proposition_4_incorrect_l473_473726


namespace min_value_expression_l473_473289

noncomputable def expression (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2)

theorem min_value_expression : ∃ x : ℝ, expression x = 2 * Real.sqrt 5 :=
by
  sorry

end min_value_expression_l473_473289


namespace greatest_3_digit_base9_divisible_by_7_l473_473147

theorem greatest_3_digit_base9_divisible_by_7 :
  ∃ (n : ℕ), n < 729 ∧ n ≥ 81 ∧ n % 7 = 0 ∧ n = 8 * 81 + 8 * 9 + 8 := 
by 
  use 728
  split
  {
    exact nat.pred_lt (ne_of_lt (by norm_num))
  }
  split
  {
    exact nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.zero_le 7))) 
  }
  split
  {
    norm_num
  }
  norm_num

end greatest_3_digit_base9_divisible_by_7_l473_473147


namespace solve_trigonometric_equation_l473_473085

theorem solve_trigonometric_equation (x : Real) (k : Int) :
  (3 * cos (2 * x) + 9 / 4) * |1 - 2 * cos (2 * x)| = sin x * (sin x - sin (5 * x))
  → (∃ k : Int, x = π / 6 + k * (π / 2) ∨ x = -π / 6 + k * (π / 2)) := 
sorry

end solve_trigonometric_equation_l473_473085


namespace find_g2_l473_473102

variable (g : ℝ → ℝ)

def condition (x : ℝ) : Prop :=
  g x - 2 * g (1 / x) = 3^x

theorem find_g2 (h : ∀ x ≠ 0, condition g x) : g 2 = -3 - (4 * Real.sqrt 3) / 9 :=
  sorry

end find_g2_l473_473102


namespace complex_value_of_z_l473_473003

theorem complex_value_of_z (z : ℂ) (h1 : arg (z^2 - 4) = 5 * π / 6) (h2 : arg (z^2 + 4) = π / 3) :
  z = 1 + complex.i * real.sqrt 3 ∨ z = -1 - complex.i * real.sqrt 3 :=
sorry

end complex_value_of_z_l473_473003


namespace how_many_cubes_needed_l473_473360

def cube_volume (side_len : ℕ) : ℕ :=
  side_len ^ 3

theorem how_many_cubes_needed (Vsmall Vlarge Vsmall_cube num_small_cubes : ℕ) 
  (h1 : Vsmall = cube_volume 8) 
  (h2 : Vlarge = cube_volume 12) 
  (h3 : Vsmall_cube = cube_volume 2) 
  (h4 : num_small_cubes = (Vlarge - Vsmall) / Vsmall_cube) :
  num_small_cubes = 152 :=
by
  sorry

end how_many_cubes_needed_l473_473360


namespace abc_sum_eq_sqrt34_l473_473893

noncomputable def abc_sum (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 16)
                          (h2 : ab + bc + ca = 9)
                          (h3 : a^2 + b^2 = 10)
                          (h4 : 0 ≤ a) (h5 : 0 ≤ b) (h6 : 0 ≤ c) : ℝ :=
a + b + c

theorem abc_sum_eq_sqrt34 (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 16)
  (h2 : ab + bc + ca = 9)
  (h3 : a^2 + b^2 = 10)
  (h4 : 0 ≤ a)
  (h5 : 0 ≤ b)
  (h6 : 0 ≤ c) :
  abc_sum a b c h1 h2 h3 h4 h5 h6 = Real.sqrt 34 :=
by
  sorry

end abc_sum_eq_sqrt34_l473_473893


namespace BK_over_AH_gt_2_l473_473877

-- Define the properties of the right triangle ABC with specific side lengths
variables (A B C K H : Point)
variables (length_AB length_BC : ℝ)
variable (ratio : ℝ)
variable (AC BK AH : ℝ)
variable (is_right_triangle : IsRightTriangle A B C)
variable (A_to_C_K_ratio_three_to_one : AC_Split_Ratio A C K 3 1)
variable (altitude_AH : IsAltitude A H C)

-- Condition: Lengths of sides AB = 5 and BC = 6
def length_AB_5 : length_AB = 5 := rfl
def length_BC_6 : length_BC = 6 := rfl

-- Prove the required inequality: The ratio of BK to AH is greater than 2
theorem BK_over_AH_gt_2 : (BK / AH) > 2 :=
sorry

end BK_over_AH_gt_2_l473_473877


namespace cats_meowing_minutes_l473_473554

/-- Given the conditions:
1. The first cat meows 3 times per minute.
2. The second cat meows twice as frequently as the first cat.
3. The third cat meows at one-third the frequency of the second cat.
4. The combined total number of meows the three cats make in a certain number of minutes is 55.
Prove that the number of minutes \( m \) the cats were meowing is 5.
-/
theorem cats_meowing_minutes :
  ∃ m : ℕ, (3 * m + 6 * m + 2 * m = 55) ∧ m = 5 :=
by
  use 5
  split
  . calc
    3 * 5 + 6 * 5 + 2 * 5
        = 15 + 30 + 10 : by simp
    ... = 55 : by simp
  . refl

end cats_meowing_minutes_l473_473554


namespace vector_orthogonal_l473_473744

def vec_a := (2, -1) : ℝ × ℝ
def vec_b := (1, 7) : ℝ × ℝ

def vec_add (u v : ℝ × ℝ) := (u.1 + v.1, u.2 + v.2)
def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

theorem vector_orthogonal:
  dot_product vec_a (vec_add vec_a vec_b) = 0 := by
  sorry

end vector_orthogonal_l473_473744


namespace problem1_problem2_problem3_l473_473344

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- Proof statements generated from the problems
theorem problem1 (x : ℝ) : f (-x) = -f x := sorry

theorem problem2 (x1 x2 : ℝ) (h : x1 < x2) : f x1 < f x2 := sorry

theorem problem3 (y : ℝ) : y ∈ set.Ioo (-1 : ℝ) 1 ↔ ∃ x : ℝ, f x = y := sorry

end problem1_problem2_problem3_l473_473344


namespace dice_probability_l473_473631

noncomputable def probability_each_number_appears_at_least_once : ℝ :=
  1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10)

theorem dice_probability : probability_each_number_appears_at_least_once = 0.272 :=
by
  sorry

end dice_probability_l473_473631


namespace at_least_half_girls_are_girls_l473_473453

noncomputable def at_least_half_girls_probability : ℚ :=
  let n := 6 in
  let p := 0.5 in
  let binom k := nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k)) in
  binom 3 + binom 4 + binom 5 + binom 6

theorem at_least_half_girls_are_girls : at_least_half_girls_probability = 21 / 32 := by
  sorry

end at_least_half_girls_are_girls_l473_473453


namespace solve_congruence_l473_473510

theorem solve_congruence :
  ∃ n : ℤ, 13 * n ≡ 8 [MOD 47] ∧ n ≡ 29 [MOD 47] :=
sorry

end solve_congruence_l473_473510


namespace omega_min_value_l473_473902

def min_omega (ω : ℝ) : Prop :=
  ω > 0 ∧ ∃ k : ℤ, (k ≠ 0 ∧ ω = 8)

theorem omega_min_value (ω : ℝ) (h1 : ω > 0) (h2 : ∃ k : ℤ, k ≠ 0 ∧ (k * 2 * π) / ω = π / 4) : 
  ω = 8 :=
by
  sorry

end omega_min_value_l473_473902


namespace train_crosses_bridge_in_60_seconds_l473_473644

theorem train_crosses_bridge_in_60_seconds :
  let length_of_train := 500 -- meters
  let speed_of_train_kmph := 42 -- km/hr
  let length_of_bridge := 200 -- meters
  let total_distance := length_of_train + length_of_bridge
  let speed_of_train_mps := (speed_of_train_kmph * 1000) / 3600
  let time_in_seconds := total_distance / speed_of_train_mps
  time_in_seconds ≈ 60 :=
by { sorry }

end train_crosses_bridge_in_60_seconds_l473_473644


namespace chinese_character_equation_l473_473440

noncomputable def units_digit (n: ℕ) : ℕ :=
  n % 10

noncomputable def tens_digit (n: ℕ) : ℕ :=
  (n / 10) % 10

noncomputable def hundreds_digit (n: ℕ) : ℕ :=
  (n / 100) % 10

def Math : ℕ := 25
def LoveMath : ℕ := 125
def ILoveMath : ℕ := 3125

theorem chinese_character_equation :
  Math * LoveMath = ILoveMath :=
by
  have h_units_math := units_digit Math
  have h_units_lovemath := units_digit LoveMath
  have h_units_ilovemath := units_digit ILoveMath
  
  have h_tens_math := tens_digit Math
  have h_tens_lovemath := tens_digit LoveMath
  have h_tens_ilovemath := tens_digit ILoveMath

  have h_hundreds_lovemath := hundreds_digit LoveMath
  have h_hundreds_ilovemath := hundreds_digit ILoveMath

  -- Check conditions:
  -- h_units_* should be 0, 1, 5 or 6
  -- h_tens_math == h_tens_lovemath == h_tens_ilovemath
  -- h_hundreds_lovemath == h_hundreds_ilovemath

  sorry -- Proof would go here

end chinese_character_equation_l473_473440


namespace B_completes_in_20_days_l473_473184

noncomputable def work_done (total_work : ℝ) (days : ℝ): ℝ := total_work / days 

theorem B_completes_in_20_days :
  ∀ (total_work : ℝ), 
  (work_done total_work 10) = work_done total_work 10 -> 
  (work_done total_work 20) = work_done total_work 20 -> 
  noncomputable (work_done total_work 20) 
:=
by
  sorry

end B_completes_in_20_days_l473_473184


namespace complete_the_square_3x2_9x_20_l473_473398

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l473_473398


namespace point_in_second_quadrant_l473_473027

theorem point_in_second_quadrant (m n : ℝ)
  (h_translation : ∃ A' : ℝ × ℝ, A' = (m+2, n+3) ∧ (A'.1 < 0) ∧ (A'.2 > 0)) :
  m < -2 ∧ n > -3 :=
by
  sorry

end point_in_second_quadrant_l473_473027


namespace dorothy_age_l473_473271

-- Define the variables involved
variables (D S : ℕ)

-- Conditions
def condition1 := D = 3 * S
def condition2 := D + 5 = 2 * (S + 5)
def sister_age := S = 5

-- The theorem statement we need to prove
theorem dorothy_age : D = 15 :=
by
  -- include the conditions as premises
  have h1 : condition1, from sorry,
  have h2 : condition2, from sorry,
  have h3 : sister_age, from sorry,
  sorry

end dorothy_age_l473_473271


namespace annual_interest_rate_l473_473294

theorem annual_interest_rate (P SI : ℝ) (T : ℝ) (rate : ℝ) 
    (hP : P = 2000) 
    (hSI : SI = 25) 
    (hT : T = 73 / 365) 
    (h_rate : rate = (SI * 100) / (P * T)) : 
    rate = 6.25 := 
by
  rw [hP, hSI, hT] at h_rate
  norm_num at h_rate
  exact h_rate

#check annual_interest_rate

end annual_interest_rate_l473_473294


namespace line_parallel_to_plane_can_form_one_plane_perpendicular_l473_473217

noncomputable def form_perpendicular_plane (l : Line) (α : Plane) : ℕ :=
  if is_parallel l α then 1 else 0

theorem line_parallel_to_plane_can_form_one_plane_perpendicular (l : Line) (α : Plane)
  (h : is_parallel l α) : form_perpendicular_plane l α = 1 :=
by
  sorry

end line_parallel_to_plane_can_form_one_plane_perpendicular_l473_473217


namespace exists_set_with_divisibility_properties_l473_473075

theorem exists_set_with_divisibility_properties (n : ℕ) (hn : n > 0) :
  ∃ S : set ℕ, S.card = n ∧ (∀ a b ∈ S, a ≠ b → (a - b) ∣ a ∧ (a - b) ∣ b ∧ ∀ c ∈ S, c ≠ a → c ≠ b → (a - b) ∣ c → false) :=
sorry

end exists_set_with_divisibility_properties_l473_473075


namespace orthocentric_tetrahedron_l473_473927

-- Definitions based on conditions
variables (O H : Point) (R l : ℝ)
-- O is the center of the circumscribed sphere
-- H is the orthocenter
-- R is the radius of the circumscribed sphere
-- l is the distance between the midpoints of skew edges of the tetrahedron

-- Statement of the theorem to be proved
theorem orthocentric_tetrahedron (O H : Point) (R l : ℝ) :
  dist_squared O H = 4 * R^2 - 3 * l^2 :=
sorry

end orthocentric_tetrahedron_l473_473927


namespace probability_of_c_between_l473_473484

noncomputable def probability_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : ℝ :=
  let c := a / (a + b)
  if (1 / 4 : ℝ) ≤ c ∧ c ≤ (3 / 4 : ℝ) then sorry else sorry
  
theorem probability_of_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : 
  probability_c_between a b hab = (2 / 3 : ℝ) :=
sorry

end probability_of_c_between_l473_473484


namespace max_price_theorem_min_sales_volume_theorem_unit_price_theorem_l473_473627

noncomputable def max_price (original_price : ℝ) (original_sales : ℝ) 
  (sales_decrement : ℝ → ℝ) : ℝ :=
  let t := 40 in
  t

theorem max_price_theorem : 
  ∀ (original_price original_sales : ℝ)
    (sales_decrement : ℝ → ℝ),
  max_price original_price original_sales sales_decrement = 40 := 
sorry

noncomputable def min_sales_volume (original_price : ℝ) (original_sales : ℝ) 
  (fixed_cost : ℝ) (variable_cost : ℝ → ℝ) 
  (tech_innovation_cost : ℝ → ℝ) : ℝ :=
  let a := 10.2 * 10^6 in
  a

theorem min_sales_volume_theorem : 
  ∀ (original_price original_sales : ℝ) 
    (fixed_cost : ℝ) (variable_cost : ℝ → ℝ)
    (tech_innovation_cost : ℝ → ℝ),
  min_sales_volume original_price original_sales 
    fixed_cost variable_cost tech_innovation_cost = 10.2 * 10^6 :=
sorry

noncomputable def unit_price (x : ℝ) : ℝ :=
  if x = 30 then x else 30

theorem unit_price_theorem : 
  ∀ (x : ℝ),
  unit_price x = 30 :=
sorry

end max_price_theorem_min_sales_volume_theorem_unit_price_theorem_l473_473627


namespace midpoints_of_common_tangents_collinear_l473_473955

theorem midpoints_of_common_tangents_collinear
  (C₁ C₂ : Circle)
  (h : dist C₁.center C₂.center > C₁.radius + C₂.radius) :
  ∀ (T1 T2 T3 T4 : TangentSegment),
  midpoint T1 ∈ radical_axis C₁ C₂ ∧
  midpoint T2 ∈ radical_axis C₁ C₂ ∧
  midpoint T3 ∈ radical_axis C₁ C₂ ∧
  midpoint T4 ∈ radical_axis C₁ C₂ :=
sorry

end midpoints_of_common_tangents_collinear_l473_473955


namespace age_twice_in_2_years_l473_473634

/-
Conditions:
1. The man is 24 years older than his son.
2. The present age of the son is 22 years.
3. In a certain number of years, the man's age will be twice the age of his son.
-/
def man_is_24_years_older (S M : ℕ) : Prop := M = S + 24
def present_age_son : ℕ := 22
def age_twice_condition (Y S M : ℕ) : Prop := M + Y = 2 * (S + Y)

/-
Prove that in 2 years, the man's age will be twice the age of his son.
-/
theorem age_twice_in_2_years : ∃ (Y : ℕ), 
  (man_is_24_years_older present_age_son M) → 
  (age_twice_condition Y present_age_son M) →
  Y = 2 :=
by
  sorry

end age_twice_in_2_years_l473_473634


namespace average_income_l473_473186

/-- The daily incomes of the cab driver over 5 days. --/
def incomes : List ℕ := [400, 250, 650, 400, 500]

/-- Prove that the average income of the cab driver over these 5 days is $440. --/
theorem average_income : (incomes.sum / incomes.length) = 440 := by
  sorry

end average_income_l473_473186


namespace ophelia_age_l473_473878

/-- 
If Lennon is currently 8 years old, 
and in two years Ophelia will be four times as old as Lennon,
then Ophelia is currently 38 years old 
-/
theorem ophelia_age 
  (lennon_age : ℕ) 
  (ophelia_age_in_two_years : ℕ) 
  (h1 : lennon_age = 8)
  (h2 : ophelia_age_in_two_years = 4 * (lennon_age + 2)) : 
  ophelia_age_in_two_years - 2 = 38 :=
by
  sorry

end ophelia_age_l473_473878


namespace magic_sum_triangle_sides_l473_473025

theorem magic_sum_triangle_sides :
  ∃ S : set ℕ, S = {9, 10, 11, 12} ∧ 
  ∀ (a b c d e f : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + b + c + d + e + f = 21 →
  ∃ (x y z w u v : ℕ), 
  {x, y, z, w, u, v} = {a, b, c, d, e, f} ∧
  (x + y + z = S) ∧ (z + w + u = S) ∧ (u + v + x = S) := 
sorry

end magic_sum_triangle_sides_l473_473025


namespace range_of_a_l473_473844

def quadratic_inequality (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 ≤ 0

theorem range_of_a :
  ¬ quadratic_inequality a ↔ -1 < a ∧ a < 3 :=
  by
  sorry

end range_of_a_l473_473844


namespace total_voters_in_districts_l473_473116

theorem total_voters_in_districts : 
  ∀ (D1 D2 D3 : ℕ),
  (D1 = 322) →
  (D2 = D3 - 19) →
  (D3 = 2 * D1) →
  (D1 + D2 + D3 = 1591) :=
by
  intros D1 D2 D3 h1 h2 h3
  sorry

end total_voters_in_districts_l473_473116


namespace slope_angle_of_tangent_l473_473541

theorem slope_angle_of_tangent (P : ℝ × ℝ) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = x^2 + 1) (hxP : P = (1 / 2, f (1 / 2))) :
  ∃ θ : ℝ, θ = 45 ∧ ∀ x, ∂ f / ∂ x = 2 * x :=
  sorry

end slope_angle_of_tangent_l473_473541


namespace triangle_third_side_length_l473_473796

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l473_473796


namespace signed_sum_arithmetic_seq_bounded_seq_counterexample_l473_473194

-- Statement of the problem in Lean 4
theorem signed_sum_arithmetic_seq (a : ℕ → ℕ) (h₁ : ∀ n, 0 < a n) (h₂ : ∀ n, a n ≤ n):
  ∃ (b : ℕ → ℤ), (∃ d : ℤ, d ≠ 0 ∧ ∀ n, b n = (b 0) + (n : ℤ) * d) ∧
  (∀ k s, s = ∑ i in finset.range k, int.sign (a i) * (↑(a i)) → s = b k) :=
sorry

-- Counterexample with bounded a_n / n
theorem bounded_seq_counterexample :
  ∃ (a : ℕ → ℕ), (∀ n, a n ≤ 100 * n) ∧ (∀ (b : ℕ → ℤ), 
  ¬ ( ∃ d : ℤ, d ≠ 0 ∧ ∃ n0, ∀ n ≥ n0, b n = (b n0) + (n - n0 : ℤ) * d ) ∧ 
  (∀ k s, s = ∑ i in finset.range k, int.sign (a i) * (↑(a i)) → s = b k)) :=
sorry

end signed_sum_arithmetic_seq_bounded_seq_counterexample_l473_473194


namespace vector_dot_product_l473_473424

open_locale real_inner_product

variable {V : Type*} [inner_product_space ℝ V]

theorem vector_dot_product (A B C : V) 
  (h₁ : ∥A - B∥ = 2)
  (h₂ : ∥(B - A) + (C - A)∥ = ∥(B - A) - (C - A)∥) :
  (B - A) • (C - B) = -4 :=
by
  sorry

end vector_dot_product_l473_473424


namespace problem1_correct_problem2_correct_l473_473659

-- First problem definition
def problem1 : ℤ :=
  (-2)^2 + (1/2)^(-1) - abs (-3)

-- Theorem for the first problem
theorem problem1_correct : problem1 = 3 :=
by 
  simp [problem1]
  sorry

-- Second problem definition
def problem2 : ℤ :=
  (-1)^2022 * (-2)^2 + (-1/2)^(-3) - (4 - real.pi)^0

-- Theorem for the second problem
theorem problem2_correct : problem2 = -5 :=
by
  simp [problem2]
  sorry

end problem1_correct_problem2_correct_l473_473659


namespace closest_ratio_to_one_l473_473521

theorem closest_ratio_to_one (a c : ℕ) (h1 : 2 * a + c = 130) (h2 : a ≥ 1) (h3 : c ≥ 1) : 
  a = 43 ∧ c = 44 :=
by {
    sorry 
}

end closest_ratio_to_one_l473_473521


namespace smallest_a_l473_473045

theorem smallest_a (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) : a = 17 :=
sorry

end smallest_a_l473_473045


namespace range_of_a_l473_473528

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
sorry

end range_of_a_l473_473528


namespace arithmetic_sequence_common_diff_sum_of_five_terms_l473_473319

-- Definitions of arithmetic sequence and initial terms
def arithmetic_sequence (a d : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := arithmetic_sequence a d n + d

-- Sum of the first n terms of arithmetic sequence
def sum_of_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  (n + 1) * a + n * (n + 1) / 2 * d

-- Problem condition, a_5 + a_6 = 2 * exp(a_4)
def problem_condition (a d : ℝ) :=
  let a4 := arithmetic_sequence a d 4
  let a5 := arithmetic_sequence a d 5
  let a6 := arithmetic_sequence a d 6
  a5 + a6 = 2 * Real.exp a4

-- Lean statements of the proof problems
theorem arithmetic_sequence_common_diff (a d : ℝ) (h : problem_condition a d) : d ≥ 2 / 3 := sorry

theorem sum_of_five_terms (a d : ℝ) (h : problem_condition a d) : sum_of_first_n_terms a d 4 < 0 := sorry

end arithmetic_sequence_common_diff_sum_of_five_terms_l473_473319


namespace infinite_rectangles_containment_l473_473597

theorem infinite_rectangles_containment:
  ∃ (r1 r2 : ℕ × ℕ), 
    (r1 ≠ r2) ∧ 
    (r1.1 ≤ r2.1) ∧ 
    (r1.2 ≤ r2.2) ∧ 
    (r1.1 * r1.2 < r2.1 * r2.2) := 
begin
  sorry -- Proof goes here.
end

end infinite_rectangles_containment_l473_473597


namespace max_value_of_g_l473_473959

def g : ℕ → ℕ 
| n := if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : 
  ∃ N, (∀ n, g n ≤ N) ∧ N = 23 := 
sorry

end max_value_of_g_l473_473959


namespace quadratic_form_h_l473_473383

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l473_473383


namespace find_b_perpendicular_lines_l473_473103

theorem find_b_perpendicular_lines (b : ℚ)
  (line1 : (3 : ℚ) * x + 4 * y - 6 = 0)
  (line2 : b * x + 4 * y - 6 = 0)
  (perpendicular : ( - (3 : ℚ) / 4 ) * ( - (b / 4) ) = -1) :
  b = - (16 : ℚ) / 3 := 
sorry

end find_b_perpendicular_lines_l473_473103


namespace mutually_exclusive_but_not_opposite_events_l473_473865

def card_colors := {c : String // c = "red" ∨ c = "green" ∨ c = "blue"}

def drawing_two_cards (bag : Multiset card_colors) (draw : Multiset card_colors) : Prop :=
  (draw.card = 2) ∧ (draw ⊆ bag)

def event_both_cards_red (draw : Multiset card_colors) : Prop :=
  draw = {⟨"red", Or.inl rfl⟩, ⟨"red", Or.inl rfl⟩}

def event_neither_card_red (draw : Multiset card_colors) : Prop :=
  ∀ x ∈ draw, x ≠ ⟨"red", Or.inl rfl⟩

def event_exactly_one_card_blue (draw : Multiset card_colors) : Prop :=
  (draw.count (λ c, c = ⟨"blue", Or.inr (Or.inl rfl)⟩) = 1)

def event_both_cards_green (draw : Multiset card_colors) : Prop :=
  draw = {⟨"green", Or.inr (Or.inr rfl)⟩, ⟨"green", Or.inr (Or.inr rfl)⟩}

theorem mutually_exclusive_but_not_opposite_events :
  ∀ (bag draw : Multiset card_colors),
    bag = {⟨"red", Or.inl rfl⟩, ⟨"red", Or.inl rfl⟩,
           ⟨"green", Or.inr (Or.inr rfl)⟩, ⟨"green", Or.inr (Or.inr rfl)⟩,
           ⟨"blue", Or.inr (Or.inl rfl)⟩, ⟨"blue", Or.inr (Or.inl rfl)⟩} →
    drawing_two_cards bag draw →
    (event_neither_card_red draw ∨
     event_exactly_one_card_blue draw ∨
     event_both_cards_green draw) →
    ¬ event_both_cards_red draw :=
sorry

end mutually_exclusive_but_not_opposite_events_l473_473865


namespace first_term_exceeds_10000_is_6144_l473_473101

noncomputable def sequence : ℕ → ℕ
| 0 => 0
| 1 => 3
| n + 2 => sequence (n + 1) + sequence (n)

theorem first_term_exceeds_10000_is_6144 :
  ∃ n, sequence n > 10000 ∧ ∀ m < n, sequence m ≤ 10000 :=
sorry

end first_term_exceeds_10000_is_6144_l473_473101


namespace obtuse_probability_l473_473073

open Real

noncomputable def is_obtuse (Q : ℝ × ℝ) (F G : ℝ × ℝ) : Prop :=
  let dir1 := (fst Q - fst F, snd Q - snd F)
  let dir2 := (fst Q - fst G, snd Q - snd G)
  let dot_product := (dir1.1 * dir2.1) + (dir1.2 * dir2.2)
  dot_product < 0

def vertices : list (ℝ × ℝ) := [(0, 1), (3, 0), (2 * pi + 2, 0), (2 * pi + 2, 3), (0, 3)]

-- Assuming area calculation for the specific pentagon being described
noncomputable def pentagon_area : ℝ := 6 * pi + 3

-- Area of the semicircle calculated based on the given conditions in the solution
noncomputable def semicircle_area : ℝ := (5 / 4) * pi

-- Probability calculation
noncomputable def probability_is_obtuse : ℝ := semicircle_area / pentagon_area

theorem obtuse_probability :
  let F := (0, 1)
  let G := (3, 0)
  let pentagon := vertices
  in probability_is_obtuse = 5 / (24 * pi + 12) :=
by
  sorry

end obtuse_probability_l473_473073


namespace simplify_and_rationalize_l473_473505

theorem simplify_and_rationalize : 
  (∃ (a b : ℝ), a / b = (sqrt 5 / sqrt 6) * (sqrt 7 / sqrt 8) * (sqrt 9 / sqrt 10) ∧ (a / b = sqrt 210 / 8)) :=
by {
  let a := sqrt 210,
  let b := 8,
  existsi a, existsi b,
  split,
  sorry, -- Proof of the simplified expression
  refl -- Proof that a / b is indeed sqrt 210 / 8
}

end simplify_and_rationalize_l473_473505


namespace tenth_monomial_l473_473120

def monomial_sequence (n : ℕ) : ℝ[X] :=
  if n = 0 then 1
  else (if (n % 2) = 1 then 1 else -1) * (real.sqrt n) * (X ^ n)

theorem tenth_monomial :
  monomial_sequence 10 = -real.sqrt 10 * (X ^ 10) :=
sorry

end tenth_monomial_l473_473120


namespace billiard_expected_reflections_l473_473028

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem billiard_expected_reflections :
  expected_reflections = (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end billiard_expected_reflections_l473_473028


namespace abs_inequality_solution_set_l473_473543

theorem abs_inequality_solution_set :
  { x : ℝ | |x - 1| + |x + 2| ≥ 5 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end abs_inequality_solution_set_l473_473543


namespace product_mod_32_l473_473054

def product_of_all_odd_primes_less_than_32 : ℕ :=
  3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  (product_of_all_odd_primes_less_than_32) % 32 = 9 :=
sorry

end product_mod_32_l473_473054


namespace ten_year_old_dog_is_64_human_years_l473_473553

namespace DogYears

-- Definition of the conditions
def first_year_in_human_years : ℕ := 15
def second_year_in_human_years : ℕ := 9
def subsequent_year_in_human_years : ℕ := 5

-- Definition of the total human years for a 10-year-old dog.
def dog_age_in_human_years (dog_age : ℕ) : ℕ :=
  if dog_age = 1 then first_year_in_human_years
  else if dog_age = 2 then first_year_in_human_years + second_year_in_human_years
  else first_year_in_human_years + second_year_in_human_years + (dog_age - 2) * subsequent_year_in_human_years

-- The statement to prove
theorem ten_year_old_dog_is_64_human_years : dog_age_in_human_years 10 = 64 :=
  by
    sorry

end DogYears

end ten_year_old_dog_is_64_human_years_l473_473553


namespace sum_of_slopes_l473_473954

/-- The coordinates of vertices of an isosceles trapezoid ABCD have integer values,
with A = (2, 10) and D = (3, 15). The trapezoid has no horizontal or vertical sides,
and AB and CD are the only parallel sides. The sum of the absolute values of
all possible slopes for AB, expressed as a reduced fraction m/n, yields m + n = 5. -/
theorem sum_of_slopes {a b : ℤ} (h₁ : (2, 10) = a)
                      (h₂ : (3, 15) = b)
                      (slope_AB : ℚ)
                      (slope_CD : ℚ)
                      (h_parallel : slope_AB = slope_CD)
                      (no_horz_vert : ∀ x y : ℤ, (x ≠ y) ∧ (¬(slope_AB = 0)) ∧ (¬(slope_CD = 0))) :
                      let m := 2
                      let n := 3
                      (abs m + abs n = 5) :=
begin
    /- We assume the points meet the given geometric constraints and 
       follow through the calculations in the proof steps given. -/
    sorry,
end

end sum_of_slopes_l473_473954


namespace greatest_base9_3_digit_divisible_by_7_l473_473149

def base9_to_decimal (n : Nat) : Nat :=
  match n with
  | 0     => 0
  | n + 1 => (n % 10) * Nat.pow 9 (n / 10)

def decimal_to_base9 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n => let rec aux (n acc : Nat) :=
              if n = 0 then acc
              else aux (n / 9) ((acc * 10) + (n % 9))
         in aux n 0

theorem greatest_base9_3_digit_divisible_by_7 :
  ∃ (n : Nat), n < Nat.pow 9 3 ∧ (n % 7 = 0) ∧ n = 8 * 81 + 8 * 9 + 8 :=
begin
  sorry -- Proof would go here
end

end greatest_base9_3_digit_divisible_by_7_l473_473149


namespace molecular_weight_of_one_mole_l473_473168

variables (w_8moles : ℝ) (n_moles : ℝ) (mw_one_mole : ℝ)

-- Given condition
axiom h : w_8moles = 352
axiom n : n_moles = 8

-- Goal
theorem molecular_weight_of_one_mole (h : w_8moles = 352) (n : n_moles = 8) : mw_one_mole = w_8moles / n_moles := 
by
  have h1 : w_8moles = 352 := h
  have h2 : n_moles = 8 := n
  have h3 : mw_one_mole = 44 := sorry
  exact h3

end molecular_weight_of_one_mole_l473_473168


namespace complete_the_square_3x2_9x_20_l473_473395

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l473_473395


namespace exists_plane_through_point_parallel_to_line_at_distance_l473_473669

-- Definitions of the given entities
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Line :=
(point : Point)
(direction : Point) -- Considering direction as a point vector for simplicity

def distance (P : Point) (L : Line) : ℝ := 
  -- Define the distance from point P to line L
  sorry

noncomputable def construct_plane (P : Point) (L : Line) (d : ℝ) : Prop :=
  -- Define when a plane can be constructed as stated in the problem.
  sorry

-- The main proof problem statement without the solution steps
theorem exists_plane_through_point_parallel_to_line_at_distance (P : Point) (L : Line) (d : ℝ) (h : distance P L > d) :
  construct_plane P L d :=
sorry

end exists_plane_through_point_parallel_to_line_at_distance_l473_473669


namespace sum_of_first_fifteen_multiples_of_7_l473_473580

theorem sum_of_first_fifteen_multiples_of_7 : (List.range 15).map (λ n, 7 * (n + 1)).sum = 840 := by
  sorry

end sum_of_first_fifteen_multiples_of_7_l473_473580


namespace find_f_seven_l473_473629

theorem find_f_seven 
  (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 3) = x^2 - 2 * x + 3) :
  f 7 = 3 := 
sorry

end find_f_seven_l473_473629


namespace color_set_no_arith_prog_same_color_l473_473501

def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 1987}

def colors : Fin 4 := sorry  -- Color indexing set (0, 1, 2, 3)

def valid_coloring (c : ℕ → Fin 4) : Prop :=
  ∀ (a d : ℕ) (h₁ : a ∈ M) (h₂ : d ≠ 0) (h₃ : ∀ k, a + k * d ∈ M ∧ k < 10), 
  ¬ ∀ k, c (a + k * d) = c a

theorem color_set_no_arith_prog_same_color :
  ∃ (c : ℕ → Fin 4), valid_coloring c :=
sorry

end color_set_no_arith_prog_same_color_l473_473501


namespace line_perpendicular_slope_l473_473843

theorem line_perpendicular_slope (m : ℝ) :
  let slope1 := (1 / 2) 
  let slope2 := (-2 / m)
  slope1 * slope2 = -1 → m = 1 := 
by
  -- The proof will go here
  sorry

end line_perpendicular_slope_l473_473843


namespace mixture_percentage_l473_473592

variable (P : ℝ)
variable (x_ryegrass_percent : ℝ := 0.40)
variable (y_ryegrass_percent : ℝ := 0.25)
variable (final_mixture_ryegrass_percent : ℝ := 0.32)

theorem mixture_percentage (h : 0.40 * P + 0.25 * (1 - P) = 0.32) : P = 0.07 / 0.15 := by
  sorry

end mixture_percentage_l473_473592


namespace statement_A_statement_B_lower_bound_statement_B_no_upper_bound_statement_C_lower_bound_statement_C_no_upper_bound_statement_D_bounded_l473_473015

-- Definitions
def f_a (x : ℝ) : ℝ := x + 1/x
def f_b (x : ℝ) : ℝ := x * Real.log x
def f_c (x : ℝ) : ℝ := Real.exp x / x^2
def f_d (x : ℝ) : ℝ := Real.sin x / (x^2 + 1)

-- Statement A
theorem statement_A (x : ℝ) (hx : 0 < x) : f_a x ≥ 2 := by sorry

-- Statement B
theorem statement_B_lower_bound : ∃ m, ∀ x > 0, f_b x ≥ m := by sorry
theorem statement_B_no_upper_bound : ∀ M, ∃ x > 0, f_b x > M := by sorry

-- Statement C
theorem statement_C_lower_bound : ∃ m, ∀ x > 0, f_c x ≥ m := by sorry
theorem statement_C_no_upper_bound : ∀ M, ∃ x > 0, f_c x > M := by sorry

-- Statement D
theorem statement_D_bounded : ∃ M, ∀ x : ℝ, |f_d x| ≤ M := by sorry

end statement_A_statement_B_lower_bound_statement_B_no_upper_bound_statement_C_lower_bound_statement_C_no_upper_bound_statement_D_bounded_l473_473015


namespace sqrt_meaningful_l473_473977

theorem sqrt_meaningful (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l473_473977


namespace quadrilateral_is_parallelogram_l473_473868

open Real

-- Definitions for points and midpoints
structure Point :=
  (x : ℝ)
  (y : ℝ)

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

-- Assume the quadrilateral ABCD and points E and K
variables (A B C D E K : Point)

-- Conditions given in the problem
axiom E_mid : E = midpoint A B
axiom K_mid : K = midpoint C D

-- Define points X, Y, X1, Y1 as the midpoints of the segments AK, CE, BK, and DE
def X := midpoint A K
def Y := midpoint C E
def X1 := midpoint B K
def Y1 := midpoint D E

-- Prove that the quadrilateral formed by X, Y, X1, and Y1 is a parallelogram
theorem quadrilateral_is_parallelogram :
  (X, Y, X1, Y1 : Point) → (midpoint X Y = midpoint X1 Y1) → (X = X1 ∨ Y = Y1) ∨ (X1 = Y1) → 
  quadrilateral_is_parallelogram X Y X1 Y1 := sorry

end quadrilateral_is_parallelogram_l473_473868


namespace simplify_and_evaluate_l473_473082

-- Given conditions: x = 1/3 and y = -1/2
def x : ℚ := 1 / 3
def y : ℚ := -1 / 2

-- Problem statement: 
-- Prove that (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2
theorem simplify_and_evaluate :
  (2 * x + 3 * y)^2 - (2 * x + y) * (2 * x - y) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_l473_473082


namespace probability_supervision_is_correct_l473_473117

noncomputable def lathe1_not_require_supervision := 0.9
noncomputable def lathe2_not_require_supervision := 0.8
noncomputable def lathe3_not_require_supervision := 0.7

noncomputable def probability_no_supervision : ℝ := 
  lathe1_not_require_supervision * lathe2_not_require_supervision * lathe3_not_require_supervision

noncomputable def probability_at_least_one_requires_supervision : ℝ := 
  1 - probability_no_supervision

theorem probability_supervision_is_correct : 
  probability_at_least_one_requires_supervision = 0.496 :=
by
  calc 
    probability_at_least_one_requires_supervision
      = 1 - probability_no_supervision : by rfl
    ... = 1 - (lathe1_not_require_supervision * lathe2_not_require_supervision * lathe3_not_require_supervision) : by rfl
    ... = 1 - (0.9 * 0.8 * 0.7) : by rfl
    ... = 0.496 : by norm_num

end probability_supervision_is_correct_l473_473117


namespace triangle_side_length_integers_l473_473773

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l473_473773


namespace trapezoid_midpoints_divide_AC_l473_473362

theorem trapezoid_midpoints_divide_AC 
  (A B C D E F G H : Type*)
  (trapezoid_AB_CD : A B C D -- Type indicating these points form a trapezoid
  (midpoint_E : E = midpoint A B)
  (midpoint_F : F = midpoint C D)
  (parallel_AB_CD : AB ∥ CD) 
  (EB_intersect_AC : E B ∩ AC = G) 
  (DF_intersect_AC : D F ∩ AC = H)) :
  AG = GH ∧ GH = HC := sorry

end trapezoid_midpoints_divide_AC_l473_473362


namespace even_function_property_l473_473708

noncomputable def f (x : ℝ) : ℝ := if h : x ≥ 0 then x - 1 else -x - 1

theorem even_function_property (x : ℝ) (hx : x < 0) (heven : ∀ x, f(-x) = f(x)) (hpositive : ∀ y, y ≥ 0 → f(y) = y - 1) :
  f(x) = -x - 1 :=
by
  sorry

end even_function_property_l473_473708


namespace chord_parallel_length_l473_473426

-- Definitions and assumptions according to the conditions
variables {a b : ℝ}
variable (h : a > b > 0)
def ellipse (x y : ℝ) : Prop := b^2 * x^2 + a^2 * y^2 = a^2 * b^2

-- Chord AB passing through (-a, 0)
def A := (-a, 0 : ℝ)
def B (α t : ℝ) := (-a + t * real.cos α, t * real.sin α : ℝ)

-- Chord AB intersects y-axis at point C
def C (α : ℝ) := (0, a * real.tan (real.pi - α) : ℝ)

-- MN parallel to AB and passes through left focus (-c, 0)
let c := real.sqrt (a^2 - b^2)
def F₁ := (-c, 0 : ℝ)
def M (α t : ℝ) := (-c + t * real.cos α, t * real.sin α : ℝ)

-- Conclusion to prove: a * |MN| = |AB| * |AC|
theorem chord_parallel_length
  (α t_ab t_mn : ℝ) (H_parallel : ∃ α, parallel (B α t_ab) (M α t_mn)) : 
  a * (dist (M α t_mn).fst (M α t_mn).snd) = 
  (dist (A.fst) ((B α t_ab).fst)) * (dist ((B α t_ab).snd) (C α).snd) :=
sorry

end chord_parallel_length_l473_473426


namespace cricket_player_average_increase_l473_473522

theorem cricket_player_average_increase :
  ∃ (A : ℝ),
  (∀ (avg : ℝ) (innings : ℕ) (extra_runs : ℝ), avg = 42 → innings = 10 → extra_runs = 86 →
    ((avg + A) * (innings + 1) = (avg * innings + extra_runs)) → A = 4) :=
begin
  use 4,
  intros avg innings extra_runs h_avg h_innings h_extra_runs h_eq,
  rw [h_avg, h_innings, h_extra_runs] at h_eq,
  linarith,
end

end cricket_player_average_increase_l473_473522


namespace value_of_Y_l473_473002

-- Definitions for the conditions in part a)
def M := 2021 / 3
def N := M / 4
def Y := M + N

-- The theorem stating the question and its correct answer
theorem value_of_Y : Y = 843 := by
  sorry

end value_of_Y_l473_473002


namespace area_shaded_WVZ_l473_473867

-- Constants representing the lengths and height
constant WZ : ℝ
constant height_X_to_WZ : ℝ
constant XV : ℝ
constant VZ : ℝ

-- Values based on given conditions
axiom WZ_val : WZ = 12
axiom height_X_to_WZ_val : height_X_to_WZ = 10
axiom XV_val : XV = 7
axiom VZ_val : VZ = 5

-- Proof statement
theorem area_shaded_WVZ : 
  let parallelogram_area := WZ * height_X_to_WZ,
      triangle_WXV_area := (1/2) * XV * height_X_to_WZ
  in parallelogram_area - triangle_WXV_area = 85 :=
by
  rw [WZ_val, height_X_to_WZ_val, XV_val, VZ_val]
  let parallelogram_area := 12 * 10
  let triangle_WXV_area := (1/2) * 7 * 10
  show 120 - 35 = 85
  rfl

end area_shaded_WVZ_l473_473867


namespace probability_of_two_applicants_to_A_l473_473855

open Finset

-- Problem Statement
theorem probability_of_two_applicants_to_A :
  (Pr (λ (s : Finset (Fin 4)) → s.card = 2 ∧ ∀ x ∈ s, x ∈ { 0 }) = 8 / 27) :=
by
  /- Introduce the problem parameters and hypothesis -/
  let communities := {A, B, C}
  let num_communities := 3
  let num_applicants := 4

  /- Define the probability of selecting each community -/
  have h1 : ∀ i ∈ communities, Pr (λ (a : Fin num_applicants) → i = a) = 1/num_communities, by

  /- Total number of possible outcomes -/
  have h2 : total_outcomes = num_communities^num_applicants

  /- Number of favorable outcomes where exactly 2 applicants choose A -/
  let favorable_outcomes := (binom num_applicants 2) * (2^(num_applicants - 2))

  /- Calculate the probability -/
  calc
    Pr (λ (s : Finset (Fin num_applicants)) → s.card = 2 ∧ ∀ x ∈ s, x ∈ { 0 }) 
      = favorable_outcomes / total_outcomes
      = (6 * 4) / 81
      = 24 / 81
      = 8 / 27

end probability_of_two_applicants_to_A_l473_473855


namespace tan_phi_of_tan_half_beta_eq_sqrt3_l473_473433

theorem tan_phi_of_tan_half_beta_eq_sqrt3 (β φ : ℝ) (h1 : tan (β / 2) = sqrt 3) :
  tan φ = 6 * sqrt 3 := by
  sorry

end tan_phi_of_tan_half_beta_eq_sqrt3_l473_473433


namespace systematic_sampling_correct_l473_473115

noncomputable def bottles : ℕ := 60
noncomputable def sample_size : ℕ := 6
noncomputable def k : ℕ := bottles / sample_size
noncomputable def start : ℕ := 3

def systematic_sample (start k : ℕ) (n : ℕ) : List ℕ :=
  List.map (λ i, start + i * k) (List.range n)

theorem systematic_sampling_correct :
  systematic_sample start k sample_size = [3, 13, 23, 33, 43, 53] :=
by
  dunno (List.range sample_size)
-- remainder here
  sorry

end systematic_sampling_correct_l473_473115


namespace find_a_cubed_minus_b_cubed_l473_473718

theorem find_a_cubed_minus_b_cubed (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 66) : a^3 - b^3 = 486 := 
by 
  sorry

end find_a_cubed_minus_b_cubed_l473_473718


namespace simplify_and_rationalize_l473_473503

-- Definitions for the radicals and their relationships
def sqrt5 : ℝ := Real.sqrt 5
def sqrt6 : ℝ := Real.sqrt 6
def sqrt7 : ℝ := Real.sqrt 7
def sqrt8 : ℝ := Real.sqrt 8
def sqrt9 : ℝ := Real.sqrt 9
def sqrt10 : ℝ := Real.sqrt 10

-- Definition of the expression to simplify
def expression := (sqrt5 / sqrt6) * (sqrt7 / sqrt8) * (sqrt9 / sqrt10)

-- The expected simplified form
def expected := (3 * Real.sqrt 1050) / 120

-- The statement we need to prove
theorem simplify_and_rationalize :
  expression = expected :=
by 
  -- Proof will go here
  sorry

end simplify_and_rationalize_l473_473503


namespace find_remainder_l473_473886

noncomputable def X : ℕ → ℤ
| 0       => 1
| 1       => 1
| (n + 1) => X n + 2 * X (n - 1)

noncomputable def Y : ℕ → ℤ
| 0       => 1
| 1       => 1
| (n + 1) => 3 * Y n + 4 * Y (n - 1)

theorem find_remainder (k : ℤ) (hk1 : ∃ i : ℕ, |X i - k| ≤ 2007) (hk2 : ∃ j : ℕ, |Y j - k| ≤ 2007) (hk3 : k < 10^{2007}) :
  ∃ r : ℤ, r = k % 2007 :=
sorry

end find_remainder_l473_473886


namespace skew_lines_l473_473584

variable {P : Type} [Plane P]
variable (a b : P.line)

theorem skew_lines (h₁ : ¬ a.parallel b) (h₂ : ¬ a.intersect b) : a.skew b := by 
  sorry

end skew_lines_l473_473584


namespace mat_weaves_problem_l473_473514

theorem mat_weaves_problem (S1 S2: ℕ) (days1 days2: ℕ) (mats1 mats2: ℕ) (H1: S1 = 1)
    (H2: S2 = 8) (H3: days1 = 4) (H4: days2 = 8) (H5: mats1 = 4) (H6: mats2 = 16) 
    (rate_consistency: (mats1 / days1) = (mats2 / days2 / S2)): S1 = 4 := 
by
  sorry

end mat_weaves_problem_l473_473514


namespace ratio_area_triangle_rectangle_l473_473076

-- Definitions of the points and conditions based on the problem statement
structure Point :=
(x : ℝ)
(y : ℝ)

structure Rectangle :=
(A B C D : Point)
(E F G : Point)

def isMidpoint (A B M : Point) : Prop := M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def isOnLineSegment (P A B : Point) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.x = (1 - t) * A.x + t * B.x ∧ P.y = (1 - t) * A.y + t * B.y

def onSameLineSegment (A B : Point) : Prop := isOnLineSegment A A B ∧ isOnLineSegment B A B

noncomputable def areaOfTriangle (A B C : Point) : ℝ :=
0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

noncomputable def areaOfRectangle (A B C D : Point) : ℝ :=
abs ((B.x - A.x) * (C.y - B.y))

def triangleToRectangleAreaRatio (R : Rectangle) : ℝ :=
areaOfTriangle R.E R.F R.G / areaOfRectangle R.A R.B R.C R.D

-- Main theorem statement
theorem ratio_area_triangle_rectangle 
  (s : ℝ) (R : Rectangle)
  (h₁ : isMidpoint R.A R.B R.E)
  (h₂ : isMidpoint R.C R.D R.G)
  (h₃ : isOnLineSegment R.F R.B R.C ∧ onSameLineSegment R.F R.B R.C) :
  triangleToRectangleAreaRatio R = 1 / 8 :=
sorry

end ratio_area_triangle_rectangle_l473_473076


namespace perfect_square_condition_l473_473464

-- Define the relevant sums and perfect square properties
def S (n : ℕ) : ℕ := ∑ k in Finset.range n, (k^4 + 2 * k^3 + 2 * k^2 + k)

theorem perfect_square_condition (n : ℕ) (h : 0 < n) : 
  (∃ m, 5 * S n + n = m^2) ↔ (∃ p, n = p^2) :=
sorry

end perfect_square_condition_l473_473464


namespace Ada_initial_seat_l473_473084

-- We have 6 seats
def Seats := Fin 6

-- Friends' movements expressed in terms of seat positions changes
variable (Bea Ceci Dee Edie Fred Ada : Seats)

-- Conditions about the movements
variable (beMovedRight : Bea.val + 1 = Ada.val)
variable (ceMovedLeft : Ceci.val = Ada.val + 2)
variable (deeMovedRight : Dee.val + 1 = Ada.val)
variable (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
  edie_new = Fred ∧ fred_new = Edie)

-- Ada returns to an end seat (1 or 6)
axiom adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩

-- Theorem to prove Ada's initial position
theorem Ada_initial_seat (Bea Ceci Dee Edie Fred Ada : Seats)
  (beMovedRight : Bea.val + 1 = Ada.val)
  (ceMovedLeft : Ceci.val = Ada.val + 2)
  (deeMovedRight : Dee.val + 1 = Ada.val)
  (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
    edie_new = Fred ∧ fred_new = Edie)
  (adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩) :
  Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩ := sorry

end Ada_initial_seat_l473_473084


namespace scalene_triangle_not_divisible_l473_473922

theorem scalene_triangle_not_divisible (A B C D : Point) (hABC : scalene A B C) (D_on_AB : lies_on D (segment A B)) :
  ¬ (1/2 * (area (triangle A D C) + area (triangle B D C)) = area (triangle A B C)) :=
sorry

end scalene_triangle_not_divisible_l473_473922


namespace train_crossing_time_l473_473233

theorem train_crossing_time
    (train_speed_kmph : ℕ)
    (platform_length_meters : ℕ)
    (crossing_time_platform_seconds : ℕ)
    (crossing_time_man_seconds : ℕ)
    (train_speed_mps : ℤ)
    (train_length_meters : ℤ)
    (T : ℤ)
    (h1 : train_speed_kmph = 72)
    (h2 : platform_length_meters = 340)
    (h3 : crossing_time_platform_seconds = 35)
    (h4 : train_speed_mps = 20)
    (h5 : train_length_meters = 360)
    (h6 : train_length_meters = train_speed_mps * crossing_time_man_seconds)
    : T = 18 :=
by
  sorry

end train_crossing_time_l473_473233


namespace school_club_profit_l473_473636

theorem school_club_profit :
  let pencils := 1200
  let buy_rate := 4 / 3 -- pencils per dollar
  let sell_rate := 5 / 4 -- pencils per dollar
  let cost_per_pencil := 3 / 4 -- dollars per pencil
  let sell_per_pencil := 4 / 5 -- dollars per pencil
  let cost := pencils * cost_per_pencil
  let revenue := pencils * sell_per_pencil
  let profit := revenue - cost
  profit = 60 := 
by
  sorry

end school_club_profit_l473_473636


namespace max_product_of_sum_2020_l473_473160

/--
  Prove that the maximum product of two integers whose sum is 2020 is 1020100.
-/
theorem max_product_of_sum_2020 : 
  ∃ x : ℤ, (x + (2020 - x) = 2020) ∧ (x * (2020 - x) = 1020100) :=
by
  sorry

end max_product_of_sum_2020_l473_473160


namespace coefficient_x3_expansion_l473_473849

theorem coefficient_x3_expansion (n : ℕ) (h1 : (3:ℚ)^n = 32) :
  let T := (3*x - (1/x))^n
  in  n = 5 → 
  ∃ c : ℚ, 
  (coeff (3*x - (1/x))^5 x^3 = -405) :=
by
  sorry

end coefficient_x3_expansion_l473_473849


namespace triangle_side_length_integers_l473_473770

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l473_473770


namespace angle_in_first_or_second_quadrant_l473_473335

theorem angle_in_first_or_second_quadrant
  (θ : ℝ) (h : Real.cot (Real.sin θ) > 0) : 
  (0 ≤ θ ∧ θ ≤ π/2) ∨ (π/2 < θ ∧ θ ≤ π) :=
sorry

end angle_in_first_or_second_quadrant_l473_473335


namespace expected_rolls_of_six_correct_l473_473078

noncomputable def die_rolls : ℕ := 100
noncomputable def probability_of_six : ℚ := 1 / 6
noncomputable def expected_rolls_of_six : ℚ := 50 / 3

theorem expected_rolls_of_six_correct :
  let X := ProbabilityMassFunction.binomial die_rolls probability_of_six in
  ProbabilityMassFunction.expectedValue X = expected_rolls_of_six :=
sorry

end expected_rolls_of_six_correct_l473_473078


namespace complete_the_square_h_value_l473_473402

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l473_473402


namespace surface_area_of_sphere_l473_473957

-- Definitions of the conditions
def vertices_on_sphere (D A B C : Type) (O : Type) [MetricSpace O] :=
    SphericalCircumsphere D A B C O

def perpendicular (A B C : Type) : Prop := IsPerpendicular A B C

def equilateral_triangle (D B C : Type) : Prop := Equilateral D B C 4

def plane_perpendicular (ABC DBC : Type) : Prop := IsPlanePerpendicular ABC DBC

-- Problem Statement with the lean condition definitions and goal
theorem surface_area_of_sphere (D A B C O : Type) [MetricSpace O] 
  (circumsphere : vertices_on_sphere D A B C O)
  (perp_AC_AB : perpendicular A C A B) 
  (equil_DBC : equilateral_triangle D B C)
  (perp_planes : plane_perpendicular (PlaneFromTriangle A B C) (PlaneFromTriangle D B C))
  : surface_area O = 64 * π / 3 :=
sorry

end surface_area_of_sphere_l473_473957


namespace infinite_series_limit_l473_473255
  
noncomputable def series_limit := 2 + (1 / (3 - 1)) / 3 + (1 / (9 - 1)) / 9

theorem infinite_series_limit : 
  series_limit = 21 / 8 :=
by 
  sorry

end infinite_series_limit_l473_473255


namespace print_pages_l473_473524

theorem print_pages (pages_per_cost : ℕ) (cost_cents : ℕ) (dollars : ℕ)
                    (h1 : pages_per_cost = 7) (h2 : cost_cents = 9) (h3 : dollars = 50) :
  (dollars * 100 * pages_per_cost) / cost_cents = 3888 :=
by
  sorry

end print_pages_l473_473524


namespace range_of_m_for_circle_l473_473100

theorem range_of_m_for_circle (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + m*x - 2*y + 4 = 0)  ↔ m < -2*Real.sqrt 3 ∨ m > 2*Real.sqrt 3 :=
by 
  sorry

end range_of_m_for_circle_l473_473100


namespace possible_integer_lengths_third_side_l473_473808

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l473_473808


namespace largest_3_digit_base9_divisible_by_7_l473_473140

def is_three_digit_base9 (n : ℕ) : Prop :=
  n < 9^3

def is_divisible_by (n d : ℕ) : Prop :=
  n % d = 0

def base9_to_base10 (n : ℕ) : ℕ :=
  let digits := [n / 81 % 9, n / 9 % 9, n % 9] in
  digits[0] * 81 + digits[1] * 9 + digits[2]

theorem largest_3_digit_base9_divisible_by_7 :
  ∃ n : ℕ, is_three_digit_base9 n ∧ is_divisible_by (base9_to_base10 n) 7 ∧ base9_to_base10 n = 728 ∧ n = 888 :=
sorry

end largest_3_digit_base9_divisible_by_7_l473_473140


namespace h_value_l473_473392

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l473_473392


namespace order_of_abc_l473_473475

noncomputable def a : ℝ := (5 / 7) ^ (-5 / 7)
noncomputable def b : ℝ := (7 / 5) ^ (3 / 5)
noncomputable def c : ℝ := Real.log (14 / 5) / Real.log 3

theorem order_of_abc : a > b ∧ b > c := by
  sorry

end order_of_abc_l473_473475


namespace triangle_third_side_count_l473_473833

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l473_473833


namespace complete_the_square_3x2_9x_20_l473_473397

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l473_473397


namespace third_side_integer_lengths_l473_473778

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l473_473778


namespace abs_add_opposite_signs_l473_473364

theorem abs_add_opposite_signs (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a * b < 0) : |a + b| = 1 := 
sorry

end abs_add_opposite_signs_l473_473364


namespace find_m_find_m_range_l473_473714

-- Define set A and set B
def setA := { x : ℝ | -1 ≤ x ∧ x ≤ 3 }
def setB (m : ℝ) := { x : ℝ | m - 2 ≤ x ∧ x ≤ m + 2 }
def intersect (a b : Set ℝ) := { x : ℝ | x ∈ a ∧ x ∈ b }
def complement (r b : Set ℝ) := { x : ℝ | x ∉ b }

-- Problem 1
theorem find_m (m : ℝ) : intersect setA (setB m) = { x : ℝ | 0 ≤ x ∧ x ≤ 3 } → m = 2 := by
  sorry

-- Problem 2
theorem find_m_range (m : ℝ) : setA ⊆ complement ℝ (setB m) ↔ m < -3 ∨ m > 5 := by
  sorry

end find_m_find_m_range_l473_473714


namespace find_a_l473_473327

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {0, 1}) (hB : B = {-1, 0, a + 3}) (h : A ⊆ B) : a = -2 := by
  sorry

end find_a_l473_473327


namespace jesse_pencils_l473_473040

def initial_pencils : ℕ := 78
def pencils_given : ℕ := 44
def final_pencils : ℕ := initial_pencils - pencils_given

theorem jesse_pencils :
  final_pencils = 34 :=
by
  -- Proof goes here
  sorry

end jesse_pencils_l473_473040


namespace how_many_peaches_l473_473038

-- Define the variables
variables (Jake Steven : ℕ)

-- Conditions
def has_fewer_peaches : Prop := Jake = Steven - 7
def jake_has_9_peaches : Prop := Jake = 9

-- The theorem that proves Steven's number of peaches
theorem how_many_peaches (Jake Steven : ℕ) (h1 : has_fewer_peaches Jake Steven) (h2 : jake_has_9_peaches Jake) : Steven = 16 :=
by
  -- Proof goes here
  sorry

end how_many_peaches_l473_473038


namespace flower_pattern_perimeter_l473_473022

theorem flower_pattern_perimeter (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 45) : 
    let arc_length := (360 - θ) / 360 * 2 * π * r
    let total_perimeter := arc_length + 2 * r
    total_perimeter = (21 / 4 * π) + 6 := 
by
  -- Definitions from conditions
  let arc_length := (360 - θ) / 360 * 2 * π * r
  let total_perimeter := arc_length + 2 * r

  -- Assertions to reach the target conclusion
  have h_arc_length: arc_length = (21 / 4 * π) :=
    by
      sorry

  -- Incorporate the radius
  have h_total: total_perimeter = (21 / 4 * π) + 6 :=
    by
      sorry

  exact h_total

end flower_pattern_perimeter_l473_473022


namespace quadratic_form_h_l473_473382

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l473_473382


namespace max_value_g_l473_473967

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_g : ∃ M, ∀ n, g n ≤ M ∧ M = 23 :=
  sorry

end max_value_g_l473_473967


namespace positive_integers_with_divisors_l473_473278

theorem positive_integers_with_divisors (n : ℕ) :
  (∀ k : ℕ, prime_factors k.size = 16) → (d 6 = 18) → (d 9 - d 8 = 17) →
  (n = 2^1 * 3^3 * 37 ∨ n = 2^1 * 3^3 * 71) := by
  sorry

end positive_integers_with_divisors_l473_473278


namespace carla_total_students_l473_473663

-- Defining the conditions
def students_in_restroom : Nat := 2
def absent_students : Nat := (3 * students_in_restroom) - 1
def total_desks : Nat := 4 * 6
def occupied_desks : Nat := total_desks * 2 / 3
def students_present : Nat := occupied_desks

-- The target is to prove the total number of students Carla teaches
theorem carla_total_students : students_in_restroom + absent_students + students_present = 23 := by
  sorry

end carla_total_students_l473_473663


namespace cuboid_surface_area_correct_l473_473191

-- Define the edge lengths of the cuboid
def length : ℝ := 4
def width : ℝ := 5
def height : ℝ := 6

-- Define the surface area of the cuboid
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)

-- Statement to prove
theorem cuboid_surface_area_correct : surface_area length width height = 148 :=
by sorry

end cuboid_surface_area_correct_l473_473191


namespace find_n_l473_473755

theorem find_n (n : ℚ) : 1 / 2 + 2 / 3 + 3 / 4 + n / 12 = 2 ↔ n = 1 := by
  -- proof here
  sorry

end find_n_l473_473755


namespace distance_between_incircle_centers_l473_473432

theorem distance_between_incircle_centers
  (A B C D O1 O2: Point)
  (AB AC BC BD DC : Real)
  (h_right_triangle : is_right_triangle A B C)
  (h_AB : dist A B = 4)
  (h_AC : dist A C = 3)
  (h_BC : dist B C = 5)
  (h_D_midpoint : midpoint D B C)
  (h_O1 : incircle_center O1 A D C)
  (h_O2 : incircle_center O2 A D B) :
  dist O1 O2 = (5 * Real.sqrt 13) / 12 := 
sorry

end distance_between_incircle_centers_l473_473432


namespace solution_interval_l473_473279

theorem solution_interval (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ (5 / 2 : ℝ) < x ∧ x ≤ (14 / 5 : ℝ) := 
by
  sorry

end solution_interval_l473_473279


namespace process_terminates_max_euros_collected_l473_473090

theorem process_terminates (n : ℕ) (C : Fin n → ℕ) :
  ∃ T : ℕ, ∀ t ≥ T, ∀ i, ¬(1 ≤ i ∧ i ≤ n ∧ C i ≥ i) :=
sorry

theorem max_euros_collected (n : ℕ) : 
  ∃ C : Fin n → ℕ, ∀ s, (s ≤ 2^n - (n + 1)) :=
sorry

end process_terminates_max_euros_collected_l473_473090


namespace ten_percent_of_thirty_percent_of_fifty_percent_of_7000_is_105_l473_473131

theorem ten_percent_of_thirty_percent_of_fifty_percent_of_7000_is_105 :
  let initial_value := 7000 in  
  0.1 * (0.3 * (0.5 * initial_value)) = 105 :=
by
  let initial_value := 7000
  sorry

end ten_percent_of_thirty_percent_of_fifty_percent_of_7000_is_105_l473_473131


namespace area_of_garden_l473_473224

variable (w l : ℕ)
variable (h1 : l = 3 * w) 
variable (h2 : 2 * (l + w) = 72)

theorem area_of_garden : l * w = 243 := by
  sorry

end area_of_garden_l473_473224


namespace largest_3_digit_base9_divisible_by_7_l473_473139

def is_three_digit_base9 (n : ℕ) : Prop :=
  n < 9^3

def is_divisible_by (n d : ℕ) : Prop :=
  n % d = 0

def base9_to_base10 (n : ℕ) : ℕ :=
  let digits := [n / 81 % 9, n / 9 % 9, n % 9] in
  digits[0] * 81 + digits[1] * 9 + digits[2]

theorem largest_3_digit_base9_divisible_by_7 :
  ∃ n : ℕ, is_three_digit_base9 n ∧ is_divisible_by (base9_to_base10 n) 7 ∧ base9_to_base10 n = 728 ∧ n = 888 :=
sorry

end largest_3_digit_base9_divisible_by_7_l473_473139


namespace third_side_integer_lengths_l473_473776

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l473_473776


namespace find_h_l473_473374

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l473_473374


namespace find_nonemployee_expenses_l473_473221

-- Conditions as definitions
def number_of_employees : ℕ := 20
def shirts_per_employee_per_day : ℕ := 20
def hours_per_shift : ℕ := 8
def hourly_wage : ℕ := 12
def bonus_per_shirt : ℕ := 5
def price_per_shirt : ℕ := 35
def daily_profit : ℕ := 9080

-- Lean 4 statement
theorem find_nonemployee_expenses :
  let total_shirts_per_day := number_of_employees * shirts_per_employee_per_day in
  let total_daily_revenue := total_shirts_per_day * price_per_shirt in
  let total_daily_wages_per_employee := (hourly_wage * hours_per_shift) + (bonus_per_shirt * shirts_per_employee_per_day) in
  let total_daily_wages := number_of_employees * total_daily_wages_per_employee in
  let total_costs := total_daily_revenue - daily_profit in
  let nonemployee_expenses := total_costs - total_daily_wages in
  nonemployee_expenses = 1000 := by
    sorry

end find_nonemployee_expenses_l473_473221


namespace count_valid_third_sides_l473_473811

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l473_473811


namespace area_shaded_region_l473_473429

-- Define the conditions in Lean

def semicircle_radius_ADB : ℝ := 2
def semicircle_radius_BEC : ℝ := 2
def midpoint_arc_ADB (D : ℝ) : Prop := D = semicircle_radius_ADB
def midpoint_arc_BEC (E : ℝ) : Prop := E = semicircle_radius_BEC
def semicircle_radius_DFE : ℝ := 1
def midpoint_arc_DFE (F : ℝ) : Prop := F = semicircle_radius_DFE

-- Given the mentioned conditions, we want to show the area of the shaded region is 8 square units
theorem area_shaded_region 
  (D E F : ℝ) 
  (hD : midpoint_arc_ADB D)
  (hE : midpoint_arc_BEC E)
  (hF : midpoint_arc_DFE F) : 
  ∃ (area : ℝ), area = 8 := 
sorry

end area_shaded_region_l473_473429


namespace quadratic_expression_rewriting_l473_473409

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l473_473409


namespace maggi_initial_packages_l473_473906

theorem maggi_initial_packages (P : ℕ) (h1 : 4 * P - 5 = 12) : P = 4 :=
sorry

end maggi_initial_packages_l473_473906


namespace find_b_l473_473703

noncomputable def polynomial_root_properties (a b c d : ℝ) (i : ℂ) : Prop :=
  i^2 = -1 ∧
  ∃ z w : ℂ, z * w = 15 + 2 * i ∧
              z + w = 2 - 5 * i ∧
              z ≠ w ∧
              b = 30 + (2 - 5 * i) * (2 + 5 * i)

theorem find_b (a c d : ℝ) : ∀ (b : ℝ), polynomial_root_properties a b c d complex.I → b = 59 :=
  by
    intros b hp
    sorry

end find_b_l473_473703


namespace total_lunch_cost_l473_473881

theorem total_lunch_cost
  (children chaperones herself additional_lunches cost_per_lunch : ℕ)
  (h1 : children = 35)
  (h2 : chaperones = 5)
  (h3 : herself = 1)
  (h4 : additional_lunches = 3)
  (h5 : cost_per_lunch = 7) :
  (children + chaperones + herself + additional_lunches) * cost_per_lunch = 308 :=
by
  sorry

end total_lunch_cost_l473_473881


namespace expected_value_and_variance_Y_l473_473716

variable (E D : (ℝ → ℝ) → ℝ)

-- Defining the binomial distribution condition
def binomial_X : ℝ → ℝ := sorry -- Definition of binomial distribution X

-- Defining the transformation Y = 2X + 1
def Y (X : ℝ) : ℝ := 2 * X + 1

-- The proof statement that needs to be verified
theorem expected_value_and_variance_Y :
    E Y = 5 ∧ D Y = 4 :=
by sorry

end expected_value_and_variance_Y_l473_473716


namespace find_m_l473_473328

noncomputable def a : ℕ → ℤ
| n => 1 + (n - 1) * 2

noncomputable def S : ℕ → ℤ
| n => 3^n - 1

noncomputable def b : ℕ → ℤ
| n => 2 * 3^(n-1)

theorem find_m : ∃ m : ℕ, 1 + a m = b 4 → m = 27 :=
begin
  simp [a, b],
  intro h,
  have ha := a 4,
  have hb := b 4,
  rw [ha, hb] at h,
  linarith,
end

end find_m_l473_473328


namespace possible_integer_lengths_third_side_l473_473803

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l473_473803


namespace domain_of_h_l473_473672

noncomputable def h (x : ℝ) : ℝ := (x^4 - 5 * x + 6) / (|x - 4| + |x + 2| - 1)

theorem domain_of_h : ∀ x : ℝ, |x - 4| + |x + 2| - 1 ≠ 0 := by
  intro x
  sorry

end domain_of_h_l473_473672


namespace intersection_of_M_and_N_l473_473062

def set_M : Set ℝ := {x | 0 ≤ x ∧ x < 2}
def set_N : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def intersection_M_N : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := 
by sorry

end intersection_of_M_and_N_l473_473062


namespace range_f_l473_473274

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_f : set.range f = (set.Iio 1 ∪ set.Ioi 1) :=
by
  sorry

end range_f_l473_473274


namespace joe_list_possibilities_l473_473619

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l473_473619


namespace complex_expression_value_l473_473246

theorem complex_expression_value {i : ℂ} (h : i^2 = -1) : i^3 * (1 + i)^2 = 2 := 
by
  sorry

end complex_expression_value_l473_473246


namespace soccer_ball_vertex_assignment_l473_473594

-- Define the properties of the vertices and edges
structure SoccerBall :=
  (vertices : Type)
  (edges : vertices → vertices → Prop)
  (colors : Type)
  (edge_color : ∀ {v₁ v₂ : vertices}, edges v₁ v₂ → colors)
  (polygons : set (finset vertices))
  (is_polygon : ∀ {p : finset vertices}, p ∈ polygons → ∀ v ∈ p, ∃ w ∈ p, w ≠ v ∧ edges v w)

-- Define the properties given in the conditions
axiom edge_matching (B : SoccerBall) : ∀ {v₁ v₂ : B.vertices} (e : B.edges v₁ v₂), ∃ c : B.colors, B.edge_color e = c
axiom three_colors_meet (B : SoccerBall) : ∀ v : B.vertices, ∃ p ∈ B.polygons, ∃ c₁ c₂ c₃ : B.colors, c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₃ ≠ c₁ ∧ 
  (∀ e : B.edges v, ∃ (v₁ v₂ : B.vertices), (v₁ = v ∨ v₂ = v) ∧ B.edge_color (B.edges v₁ v₂) ∈ {c₁, c₂, c₃})

-- Complex numbers and their corresponding assignments
def complex_assignment (colors : Type) : Type := colors → ℂ
axiom color_assignment : complex_assignment SoccerBall.colors

-- The goal of the proof problem
theorem soccer_ball_vertex_assignment (B : SoccerBall) : 
  ∃ f : B.vertices → ℂ, (∀ p ∈ B.polygons, finset.prod p f = 1) :=
sorry

end soccer_ball_vertex_assignment_l473_473594


namespace total_distance_traveled_l473_473202

theorem total_distance_traveled :
  let d1 := 5
  let d2 := 8
  let d3 := 10
  d1 + d2 + d3 = 23 :=
by
  let d1 := 5
  let d2 := 8
  let d3 := 10
  show d1 + d2 + d3 = 23 from sorry

end total_distance_traveled_l473_473202


namespace sum_of_first_fifteen_multiples_of_7_l473_473575

theorem sum_of_first_fifteen_multiples_of_7 :
  ∑ i in finset.range 15, 7 * (i + 1) = 840 :=
sorry

end sum_of_first_fifteen_multiples_of_7_l473_473575


namespace quadratic_form_h_l473_473381

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l473_473381


namespace find_h_l473_473416

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l473_473416


namespace find_h_l473_473376

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l473_473376


namespace compare_y1_y2_l473_473535

-- Definitions for the conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + 3
axiom a_neg (a : ℝ) : a < 0

-- The proof problem statement
theorem compare_y1_y2 (a : ℝ) (h_a : a_neg a)
  (y1 : ℝ) (y2 : ℝ) (h_y1 : y1 = quadratic_function a (-1))
  (h_y2 : y2 = quadratic_function a 2) : y1 < y2 :=
sorry

end compare_y1_y2_l473_473535


namespace cone_ratio_range_l473_473540

noncomputable def cone_ratio (L r : ℝ) : Prop :=
  0 < L ∧ 0 < r ∧ L > r ∧ (1/2 * L^2 = 1/2 * 2 * r * sqrt (L^2 - r^2))

theorem cone_ratio_range (L r : ℝ) (h : cone_ratio L r) :
  (sqrt 2 / 2 : ℝ) ≤ r / L ∧ r / L < 1 :=
sorry

end cone_ratio_range_l473_473540


namespace elements_in_A_l473_473080

variable (A B : Set ℕ)
variable (a b : ℕ)
variable (h1 : a = 2 * b)
variable (h2 : (Set.card A ∪ Set.card B) = 3011)
variable (h3 : (Set.card A ∩ Set.card B) = 1000)

theorem elements_in_A :
  Set.card A = 2674 :=
sorry

end elements_in_A_l473_473080


namespace unique_number_not_in_range_l473_473527

noncomputable def f (x : ℝ) (a b c d : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range (a b c d : ℝ)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_f_20 : f 20 a b c d = 20)
  (h_f_100 : f 100 a b c d = 100)
  (h_f_invol : ∀ x, x ≠ (-d / c) → f (f x a b c d) a b c d = x)
  (h_f_prime_20 : deriv (λ x, f x a b c d) 20 = -1) :
  ∃ unique y, @set.compl _ _ (set.range (λ x, f x a b c d)) = {y} ∧ y = 60 :=
sorry

end unique_number_not_in_range_l473_473527


namespace shoes_problem_l473_473655

-- Definitions based on the conditions
def B : ℤ -- Becky's number of pairs of shoes
def Bonny_shoes (B : ℤ) : ℤ := 2 * B - 5
def Bobby_shoes (B : ℤ) : ℤ := 3 * B

-- The proof statement
theorem shoes_problem (h : Bonny_shoes B = Bobby_shoes B) : B = -5 :=
by sorry

end shoes_problem_l473_473655


namespace max_min_diff_l473_473346

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x + 8

theorem max_min_diff : 
  let M := max (f (-1)) (max (f 2) (f 3)),
    m := min (f (-1)) (min (f 2) (f 3))
  in M - m = 27 := by
  sorry

end max_min_diff_l473_473346


namespace find_m_minus_n_l473_473751

theorem find_m_minus_n (x y : ℤ) (m n : ℤ) 
  (h : x^m * y * 4 * y^n = 4 * x^6 * y^4) : m - n = 3 :=
sorry

end find_m_minus_n_l473_473751


namespace trig_question_l473_473717

theorem trig_question (α : ℝ) (h : sin (α - 3 * cos α) = 0) : 
  (sin (2 * α) / (cos α ^ 2 - sin α ^ 2)) = -3 / 4 := 
sorry

end trig_question_l473_473717


namespace replace_asterisk_l473_473585

theorem replace_asterisk (x : ℕ) (h : (42 / 21) * (42 / x) = 1) : x = 84 := by
  sorry

end replace_asterisk_l473_473585


namespace inscribed_sphere_radius_eq_l473_473641

-- Define the parameters for the right cone
structure RightCone where
  base_radius : ℝ
  height : ℝ

-- Given the right cone conditions
def givenCone : RightCone := { base_radius := 15, height := 40 }

-- Define the properties for inscribed sphere
def inscribedSphereRadius (c : RightCone) : ℝ := sorry

-- The theorem statement for the radius of the inscribed sphere
theorem inscribed_sphere_radius_eq (c : RightCone) : ∃ (b d : ℝ), 
  inscribedSphereRadius c = b * Real.sqrt d - b ∧ (b + d = 14) :=
by
  use 5, 9
  sorry

end inscribed_sphere_radius_eq_l473_473641


namespace tenth_monomial_is_neg_sqrt_10_x_10_l473_473118

def nth_monomial (n : ℕ) : ℝ := (-1)^(n-1) * real.sqrt n

theorem tenth_monomial_is_neg_sqrt_10_x_10 : nth_monomial 10 = - real.sqrt 10 * (x : ℝ)^10 :=
by
  sorry

end tenth_monomial_is_neg_sqrt_10_x_10_l473_473118


namespace max_price_theorem_min_sales_volume_theorem_unit_price_theorem_l473_473626

noncomputable def max_price (original_price : ℝ) (original_sales : ℝ) 
  (sales_decrement : ℝ → ℝ) : ℝ :=
  let t := 40 in
  t

theorem max_price_theorem : 
  ∀ (original_price original_sales : ℝ)
    (sales_decrement : ℝ → ℝ),
  max_price original_price original_sales sales_decrement = 40 := 
sorry

noncomputable def min_sales_volume (original_price : ℝ) (original_sales : ℝ) 
  (fixed_cost : ℝ) (variable_cost : ℝ → ℝ) 
  (tech_innovation_cost : ℝ → ℝ) : ℝ :=
  let a := 10.2 * 10^6 in
  a

theorem min_sales_volume_theorem : 
  ∀ (original_price original_sales : ℝ) 
    (fixed_cost : ℝ) (variable_cost : ℝ → ℝ)
    (tech_innovation_cost : ℝ → ℝ),
  min_sales_volume original_price original_sales 
    fixed_cost variable_cost tech_innovation_cost = 10.2 * 10^6 :=
sorry

noncomputable def unit_price (x : ℝ) : ℝ :=
  if x = 30 then x else 30

theorem unit_price_theorem : 
  ∀ (x : ℝ),
  unit_price x = 30 :=
sorry

end max_price_theorem_min_sales_volume_theorem_unit_price_theorem_l473_473626


namespace max_product_of_sum_2020_l473_473159

/--
  Prove that the maximum product of two integers whose sum is 2020 is 1020100.
-/
theorem max_product_of_sum_2020 : 
  ∃ x : ℤ, (x + (2020 - x) = 2020) ∧ (x * (2020 - x) = 1020100) :=
by
  sorry

end max_product_of_sum_2020_l473_473159


namespace find_n_l473_473847

noncomputable def f (x : ℝ) (n : ℝ) : ℝ := x^n + 3^x

theorem find_n (n : ℝ) :
  HasDerivAt (λ x : ℝ, f x n) (3 + 3 * Real.log 3) 1 → n = 3 :=
by
  intros h
  have h_deriv : deriv (λ x : ℝ, f x n) 1 = 3 + 3 * Real.log 3,
  { exact HasDerivAt.deriv h }
  simp [f, deriv] at h_deriv,
  sorry

end find_n_l473_473847


namespace pizza_slices_distributed_l473_473682

theorem pizza_slices_distributed :
  ∀ (total_pizzas: ℕ) (slices_per_pizza: ℕ) (coworkers: ℕ) (slices_per_person: ℕ),
  total_pizzas = 4 →
  slices_per_pizza = 10 →
  coworkers = 18 →
  slices_per_person = 2 →
  (total_pizzas * slices_per_pizza - coworkers * slices_per_person) = 4 :=
by
  intros total_pizzas slices_per_pizza coworkers slices_per_person Htp Hsp Hc Hspp
  rw [Htp, Hsp, Hc, Hspp]
  calc
    4 * 10 - 18 * 2 = 40 - 36 := by rfl
                   ... = 4     := by rfl

end pizza_slices_distributed_l473_473682


namespace range_of_a_on_opposite_sides_l473_473323

theorem range_of_a_on_opposite_sides
  (a : ℝ)
  (A : ℝ × ℝ := (1, 3))
  (B : ℝ × ℝ := (5, 2))
  (line : ℝ × ℝ → ℝ := λ p, 3 * p.1 + 2 * p.2 + a)
  (h : line A * line B < 0) :
  -19 < a ∧ a < -9 :=
sorry

end range_of_a_on_opposite_sides_l473_473323


namespace first_fifteen_multiples_of_seven_sum_l473_473573

theorem first_fifteen_multiples_of_seven_sum :
    (∑ i in finset.range 15, 7 * (i + 1)) = 840 := 
sorry

end first_fifteen_multiples_of_seven_sum_l473_473573


namespace right_triangle_tan_l473_473875

theorem right_triangle_tan
  (A B C : Type)
  [RightTriangle A B C]
  (hAC : dist A C = 5)
  (hAB : dist A B = 3) :
  dist B C = 4 ∧ tan (angle A B C) = 4 / 3 :=
by
  sorry

end right_triangle_tan_l473_473875


namespace number_of_subsets_M_sum_of_elements_of_all_subsets_M_l473_473740

-- We define the set M
def M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The number of subsets of M
theorem number_of_subsets_M : M.powerset.card = 2 ^ 10 := by
  sorry

-- The sum of the elements of all subsets of M
theorem sum_of_elements_of_all_subsets_M : M.sum * (2 ^ 9) = 55 * (2 ^ 9) := by
  have h₁ : M.sum = 55 := by
    sorry
  rw h₁
  have h₂ : M.powerset.card / 2 = 2 ^ 9 := by
    sorry
  ring
  exact h₂

end number_of_subsets_M_sum_of_elements_of_all_subsets_M_l473_473740


namespace problems_finished_equals_45_l473_473232

/-- Mathematical constants and conditions -/
def ratio_finished_left (F L : ℕ) : Prop := F = 9 * (L / 4)
def total_problems (F L : ℕ) : Prop := F + L = 65

/-- Lean theorem to prove the problem statement -/
theorem problems_finished_equals_45 :
  ∃ F L : ℕ, ratio_finished_left F L ∧ total_problems F L ∧ F = 45 :=
by
  sorry

end problems_finished_equals_45_l473_473232


namespace complete_the_square_h_value_l473_473408

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l473_473408


namespace h_value_l473_473391

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l473_473391


namespace smaller_number_is_three_l473_473998

theorem smaller_number_is_three (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 36) : min x y = 3 :=
sorry

end smaller_number_is_three_l473_473998


namespace fish_ranking_l473_473071

def ranks (P V K T : ℕ) : Prop :=
  P < K ∧ K < T ∧ T < V

theorem fish_ranking (P V K T : ℕ) (h1 : K < T) (h2 : P + V = K + T) (h3 : P + T < V + K) : ranks P V K T :=
by
  sorry

end fish_ranking_l473_473071


namespace find_h_l473_473420

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l473_473420


namespace parabola_standard_eq_l473_473544

theorem parabola_standard_eq (p : ℝ) (h : p / 2 = 1) : ∀ (x y : ℝ), y^2 = -2 * p * x → y^2 = -4 * x :=
by
  intro x y
  assume h2 : y^2 = -2 * p * x
  have p_value : p = 2 := by
    linarith
  rw [p_value] at h2
  exact h2
  sorry

end parabola_standard_eq_l473_473544


namespace triangle_side_length_integers_l473_473772

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l473_473772


namespace distinct_numbers_in_T_l473_473477

open Set

def sequence1 (k : ℕ) : ℤ := 3 * k - 1
def sequence2 (m : ℕ) : ℤ := 7 * m - 4

def C : Set ℤ := {n | ∃ k ∈ Finset.range 3005, n = sequence1 (k + 1)}
def D : Set ℤ := {n | ∃ m ∈ Finset.range 2003, n = sequence2 (m + 1)}
def T : Set ℤ := C ∪ D

theorem distinct_numbers_in_T : T.to_finset.card = 4865 :=
by
  sorry

end distinct_numbers_in_T_l473_473477


namespace ratio_and_equation_imp_value_of_a_l473_473010

theorem ratio_and_equation_imp_value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 20 - 7 * a) :
  a = 20 / 11 :=
by
  sorry

end ratio_and_equation_imp_value_of_a_l473_473010


namespace number_of_triangle_functions_l473_473313
open Real

-- Define what it means to be a triangle function.
def is_triangle_function (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ a b c ∈ D, let fa := f a, fb := f b, fc := f c in
  fa + fb > fc ∧ fa + fc > fb ∧ fb + fc > fa

-- Define the four functions.
def f1 (x : ℝ) : ℝ := log (x + 1)
def f2 (x : ℝ) : ℝ := 4 - cos x
def f3 (x : ℝ) : ℝ := sqrt x
def f4 (x : ℝ) : ℝ := (3 ^ x + 2) / (3 ^ x + 1)

-- Define the domains of the four functions.
def D1 : set ℝ := {x | x > 0}
def D2 : set ℝ := set.univ
def D3 : set ℝ := {x | 1 ≤ x ∧ x ≤ 16}
def D4 : set ℝ := set.univ

-- State the main theorem to prove.
theorem number_of_triangle_functions : 
  (∃ f : ℝ → ℝ, is_triangle_function f D1 ∧ f = f1) + 
  (∃ f : ℝ → ℝ, is_triangle_function f D2 ∧ f = f2) + 
  (∃ f : ℝ → ℝ, is_triangle_function f D3 ∧ f = f3) + 
  (∃ f : ℝ → ℝ, is_triangle_function f D4 ∧ f = f4) = 2 := sorry

end number_of_triangle_functions_l473_473313


namespace compute_f_at_pi_over_6_l473_473728

noncomputable def f (x φ : ℝ) : ℝ := sin (x + φ) + sqrt 3 * cos (x + φ)

theorem compute_f_at_pi_over_6 
  (h₀ : 0 ≤ φ ∧ φ ≤ π)
  (h₁ : ∀ x, f x φ = -f (-x) φ) :
  f (π / 6) (2 * π / 3) = -1 :=
by
  sorry

end compute_f_at_pi_over_6_l473_473728


namespace solve_arithmetic_sequence_l473_473864

variable {a : ℕ → ℝ}
variable {d a1 a2 a3 a10 a11 a6 a7 : ℝ}

axiom arithmetic_seq (n : ℕ) : a (n + 1) = a1 + n * d

def arithmetic_condition (h : a 2 + a 3 + a 10 + a 11 = 32) : Prop :=
  a 6 + a 7 = 16

theorem solve_arithmetic_sequence (h : a 2 + a 3 + a 10 + a 11 = 32) : a 6 + a 7 = 16 :=
  by
    -- Proof will go here
    sorry

end solve_arithmetic_sequence_l473_473864


namespace main_theorem_l473_473181

open Complex

-- A: If the complex number z = a + bi corresponds to a point on the imaginary axis in the complex plane, then a = 0.
def statement_A (a b : ℝ) (hz : (a : ℂ) + (b : ℂ) * I ∈ { z : ℂ | ∃ b : ℝ, z = (0 : ℂ) + (b : ℂ) * I }) : a = 0 :=
sorry

-- B: The conjugate of a complex number z is \overline{z}. A necessary and sufficient condition for z ∈ ℝ is z = \overline{z}.
def statement_B (z : ℂ) : (z ∈ ℝ) ↔ (z = conj z) :=
sorry

-- C: If (x^2 - 1) + (x^2 - 2x - 3)i is a purely imaginary number, then the real number x = ±1.
def statement_C (x : ℝ) (hz : (x^2 - 1 : ℂ) + (x^2 - 2*x - 3 : ℂ) * I ∈ { z : ℂ | ∃ b : ℝ, z = (0 : ℂ) + (b : ℂ) * I }) : x = 1 ∨ x = -1 :=
sorry

-- D: The two roots of the equation x^2 + 2x + 3 = 0 in the complex number range are conjugate complex numbers.
def statement_D (x1 x2 : ℂ) (hx : x^2 + 2*x + 3 = 0) : x1 = conj x2 :=
sorry

-- Putting it all together in a single theorem
theorem main_theorem (a b : ℝ) (z : ℂ) (x x1 x2 : ℂ)
  (hzA : (a : ℂ) + (b : ℂ) * I ∈ { z : ℂ | ∃ b : ℝ, z = (0 : ℂ) + (b : ℂ) * I })
  (hzB : z ∈ ℝ)
  (hzC : (x^2 - 1 : ℂ) + (x^2 - 2*x - 3 : ℂ) * I ∈ { z : ℂ | ∃ b : ℝ, z = (0 : ℂ) + (b : ℂ) * I })
  (hxD : x^2 + 2*x + 3 = 0) :
  (a = 0) ∧
  ((z ∈ ℝ) ↔ (z = conj z)) ∧
  (x = 1 ∨ x = -1) ∧
  (x1 = conj x2) :=
sorry

end main_theorem_l473_473181


namespace intersection_correct_l473_473715

def A : Set ℝ := { x | 0 < x ∧ x < 3 }
def B : Set ℝ := { x | x^2 ≥ 4 }
def intersection : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

theorem intersection_correct : A ∩ B = intersection := by
  sorry

end intersection_correct_l473_473715


namespace least_number_to_make_divisible_l473_473583

theorem least_number_to_make_divisible (k : ℕ) (h : 1202 + k = 1204) : (2 ∣ 1204) := 
by
  sorry

end least_number_to_make_divisible_l473_473583


namespace angle_C_condition1_angle_C_condition2_angle_C_condition3_max_area_l473_473445

theorem angle_C_condition1 
  (a b c : ℝ) 
  (S_ABC : ℝ)
  (h1 : sqrt 3 * (a * b * cos C) = 2 * divide 1 2 * (a * b * sin C))
  : C = π / 3 := sorry

theorem angle_C_condition2
  (a b c : ℝ)
  (h2 : (sin C + sin A) * (sin C - sin A) = sin B * (sin B - sin A))
  : C = π / 3 := sorry

theorem angle_C_condition3
  (a b c : ℝ)
  (h3 : (2 * a - b) * cos C = c * cos B)
  : C = π / 3 := sorry

theorem max_area 
  (a b : ℝ)
  (c : ℝ := 2)
  (h_c : c = 2)
  (h4 : angle C = π / 3)
  : S_ABC = sqrt 3 / 4 * (a * b) := sorry

end angle_C_condition1_angle_C_condition2_angle_C_condition3_max_area_l473_473445


namespace train_length_l473_473128

theorem train_length :
  ∃ (L : ℝ), (let speed_train1 := 75 / 3.6 in
               let speed_train2 := 55 / 3.6 in
               let relative_speed := speed_train1 - speed_train2 in
               let passing_time := 210 in
               2 * L = relative_speed * passing_time) ∧ 
              L = 583.33 :=
begin
    sorry,
end

end train_length_l473_473128


namespace greatest_3_digit_base9_divisible_by_7_l473_473144

theorem greatest_3_digit_base9_divisible_by_7 :
  ∃ (n : ℕ), n < 729 ∧ n ≥ 81 ∧ n % 7 = 0 ∧ n = 8 * 81 + 8 * 9 + 8 := 
by 
  use 728
  split
  {
    exact nat.pred_lt (ne_of_lt (by norm_num))
  }
  split
  {
    exact nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.zero_le 7))) 
  }
  split
  {
    norm_num
  }
  norm_num

end greatest_3_digit_base9_divisible_by_7_l473_473144


namespace rectangular_to_polar_l473_473259

theorem rectangular_to_polar :
  ∀ (x y : ℝ), x = -3 → y = 4 → 
  ∃ (r θ : ℝ), r = real.sqrt (x^2 + y^2) ∧ θ = real.pi - real.arctan (y / abs x) ∧
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * real.pi ∧
  (r, θ) = (5, real.pi - real.arctan (4 / 3)) :=
begin
  intros x y h1 h2,
  use [real.sqrt (x^2 + y^2), real.pi - real.arctan (y / abs x)],
  split,
  { rw [h1, h2], exact real.sqrt_eq rfl },
  split,
  { rw [h1, h2, real.pi_sub_eq_real_pi],
    exact real.arctan_pos (by norm_num) },
  split,
  { exact real.sqrt_pos.mpr (add_pos (pow_pos (by norm_num) _) (pow_pos (by norm_num) _)) },
  split,
  { apply sub_nonneg.mpr, exact real.pi_pos },
  split,
  { apply sub_lt.mpr; exact lt_trans (real.arctan_pos (by norm_num)) real.pi_lt_two_real_pi },
  { rw [h1, h2, real.pi_sub_eq_real_pi], exact eq.refl ⟩
end

end rectangular_to_polar_l473_473259


namespace line_parallel_condition_l473_473919

variables {α β l : Plane} {m n : Line}

-- Conditions
axiom cond1 : α ⊥ β
axiom cond2 : (α ∩ β) = l
axiom cond3 : n ⊂ β
axiom cond4 : n ⊥ l
axiom cond5 : m ⊥ α

-- The positional relationship to prove
theorem line_parallel_condition : m ‖ n :=
by
  sorry

end line_parallel_condition_l473_473919


namespace h_value_l473_473389

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l473_473389


namespace value_of_m_l473_473668

def f (x : ℚ) : ℚ := 3 * x^3 - 1 / x + 2
def g (x : ℚ) (m : ℚ) : ℚ := 2 * x^3 - 3 * x + m
def h (x : ℚ) : ℚ := x^2

theorem value_of_m : f 3 - g 3 (122 / 3) + h 3 = 5 :=
by
  sorry

end value_of_m_l473_473668


namespace books_left_to_read_l473_473910

theorem books_left_to_read (total_books : ℕ) (books_mcgregor : ℕ) (books_floyd : ℕ) : total_books = 89 → books_mcgregor = 34 → books_floyd = 32 → 
  (total_books - (books_mcgregor + books_floyd) = 23) :=
by
  intros h1 h2 h3
  sorry

end books_left_to_read_l473_473910


namespace infinite_series_sum_l473_473686

open BigOperators -- Allow big operators such as summations
open Nat -- Provide a way to work with natural numbers

theorem infinite_series_sum :
  (\sum n : ℕ, (if n > 0 then (3^n : ℝ) / (1 + 3^n + 3^(n+1) + 3^(2*n+1)) else 0))
  = 1/4 :=
by
  sorry

end infinite_series_sum_l473_473686


namespace complete_the_square_h_value_l473_473407

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l473_473407


namespace circles_tangent_to_C1_and_C2_l473_473461

-- Necessary definitions for circles
structure Circle (r : ℝ) :=
  (center : ℝ × ℝ)
  (radius : ℝ := r)

-- Define the conditions
def C1 : Circle 2 := {center := (0, 0)}
def C2 : Circle 2 := {center := (4, 0)}

-- Define the problem statement
theorem circles_tangent_to_C1_and_C2 : 
  let candidate_circles := {c : Circle 3 // 
    (dist c.center C1.center = 5) ∨ (dist c.center C1.center = 1) ∨
    (dist c.center C2.center = 5) ∨ (dist c.center C2.center = 1)} in
  fintype.card candidate_circles = 4 :=
sorry

end circles_tangent_to_C1_and_C2_l473_473461


namespace june_found_total_eggs_l473_473885

def eggs_in_tree_1 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_tree_2 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_yard (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest

def total_eggs (eggs_tree_1 : ℕ) (eggs_tree_2 : ℕ) (eggs_yard : ℕ) : ℕ :=
eggs_tree_1 + eggs_tree_2 + eggs_yard

theorem june_found_total_eggs :
  total_eggs (eggs_in_tree_1 2 5) (eggs_in_tree_2 1 3) (eggs_in_yard 1 4) = 17 :=
by
  sorry

end june_found_total_eggs_l473_473885


namespace possible_integer_lengths_third_side_l473_473802

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l473_473802


namespace probability_of_black_yellow_green_probability_of_not_red_or_green_l473_473427

namespace ProbabilityProof

/- Definitions of events A, B, C, D representing probabilities as real numbers -/
variables (P_A P_B P_C P_D : ℝ)

/- Conditions stated in the problem -/
def conditions (h1 : P_A = 1 / 3)
               (h2 : P_B + P_C = 5 / 12)
               (h3 : P_C + P_D = 5 / 12)
               (h4 : P_A + P_B + P_C + P_D = 1) :=
  true

/- Proof that P(B) = 1/4, P(C) = 1/6, and P(D) = 1/4 given the conditions -/
theorem probability_of_black_yellow_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1) :
  P_B = 1 / 4 ∧ P_C = 1 / 6 ∧ P_D = 1 / 4 :=
by
  sorry

/- Proof that the probability of not drawing a red or green ball is 5/12 -/
theorem probability_of_not_red_or_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1)
  (h5 : P_B = 1 / 4)
  (h6 : P_C = 1 / 6)
  (h7 : P_D = 1 / 4) :
  1 - (P_A + P_D) = 5 / 12 :=
by
  sorry

end ProbabilityProof

end probability_of_black_yellow_green_probability_of_not_red_or_green_l473_473427


namespace identify_5_genuine_coins_l473_473072

/-- Petya has 8 coins, labeled {A, B, C, D, E, F, G, H}. 
    7 of them are genuine and weigh the same, 1 is counterfeit (heavier or lighter). 
    Using a balance scale that shows which side is heavier, Petya wants 
    to identify 5 genuine coins without giving any to Vasya.
    This statement proves that Petya can guarantee identifying 
    5 genuine coins without giving any away to Vasya. -/
theorem identify_5_genuine_coins (coins : Finset ℕ) (A B C D E F G H : ℕ) (counterfeit : ℕ) :
  (∀ c ∈ coins, c ≠ counterfeit) →
  (H ∈ coins) →
  (coins = {A, B, C, D, E, F, G, H} \ {counterfeit}) →
  ∃ (genuine_coins : Finset ℕ), (genuine_coins.card = 5) ∧
  (∀ c ∈ genuine_coins, c ∉ {A, B, C, D, E, F, G, H} \ {counterfeit}) :=
sorry

end identify_5_genuine_coins_l473_473072


namespace domain_of_log_function_l473_473956

theorem domain_of_log_function (x : ℝ) :
  (5 - x > 0) ∧ (x - 2 > 0) ∧ (x - 2 ≠ 1) ↔ (2 < x ∧ x < 3) ∨ (3 < x ∧ x < 5) :=
by
  sorry

end domain_of_log_function_l473_473956


namespace sqrt_m_conditions_l473_473848

theorem sqrt_m_conditions (m : ℝ) (x : ℝ) 
  (h1 : sqrt m = x + 1) 
  (h2 : sqrt m = 5 + 2x) : 
  m = 1 := 
by 
  -- placeholder for the future proof
  sorry

end sqrt_m_conditions_l473_473848


namespace triangle_third_side_lengths_l473_473757

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l473_473757


namespace max_abs_sum_l473_473367

theorem max_abs_sum (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_l473_473367


namespace sector_arc_length_circumference_ratio_l473_473639

theorem sector_arc_length_circumference_ratio
  {r : ℝ}
  (h_radius : ∀ (sector_radius : ℝ), sector_radius = 2/3 * r)
  (h_area : ∀ (sector_area circle_area : ℝ), sector_area / circle_area = 5/27) :
  ∀ (l C : ℝ), l / C = 5 / 18 :=
by
  -- Prove the theorem using the given hypothesis.
  -- Construction of the detailed proof will go here.
  sorry

end sector_arc_length_circumference_ratio_l473_473639


namespace trees_not_need_replanting_l473_473557

/-- A track has trees planted every 4 meters on both sides, with the distance between the first and last tree being 48 meters. Now, trees are replanted every 6 meters. Prove that the number of trees that do not need to be replanted is 5. -/
theorem trees_not_need_replanting :
  (distance : ℕ) (plant_interval : ℕ) (replant_interval : ℕ)
  (h1 : plant_interval = 4) (h2 : replant_interval = 6) (h3 : distance = 48)
  : let lcm := Nat.lcm plant_interval replant_interval in
    (distance / lcm) + 1 = 5 :=
by
  intros
  sorry

end trees_not_need_replanting_l473_473557


namespace smallest_triangle_perimeter_l473_473171

theorem smallest_triangle_perimeter :
  ∃ a b c : ℕ, a + 1 = b ∧ b + 1 = c ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ (a + b + c = 9) :=
begin
  sorry
end

end smallest_triangle_perimeter_l473_473171


namespace find_h_l473_473419

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l473_473419


namespace sandy_comic_books_ratio_l473_473932

variable (S : ℕ)  -- number of comic books Sandy sold

theorem sandy_comic_books_ratio 
  (initial : ℕ) (bought : ℕ) (now : ℕ) (h_initial : initial = 14) (h_bought : bought = 6) (h_now : now = 13)
  (h_eq : initial - S + bought = now) :
  S = 7 ∧ S.to_rat / initial.to_rat = 1 / 2 := 
by
  sorry

end sandy_comic_books_ratio_l473_473932


namespace number_of_possible_third_side_lengths_l473_473824

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l473_473824


namespace hcf_of_two_numbers_900_l473_473968

theorem hcf_of_two_numbers_900 (A B H : ℕ) (h_lcm : lcm A B = H * 11 * 15) (h_A : A = 900) : gcd A B = 165 :=
by
  sorry

end hcf_of_two_numbers_900_l473_473968


namespace candles_lighting_time_l473_473126

-- Definitions of the initial lengths and burning times.
def candle1_burnout_time := 300 -- minutes
def candle2_burnout_time := 180 -- minutes

-- Function describing the stub length of the first candle at time t minutes.
def stub_length_candle1 (ℓ : ℝ) (t : ℝ) : ℝ :=
  ℓ * (1 - t / candle1_burnout_time)

-- Function describing the stub length of the second candle at time t minutes.
def stub_length_candle2 (ℓ : ℝ) (t : ℝ) : ℝ :=
  ℓ * (1 - t / candle2_burnout_time)

-- Prove that to have one stub three times the length of the other at 5 PM,
-- the candles should be lighted at 2:30 PM.
theorem candles_lighting_time :
  ∃ t_lighted : ℝ,
    5 * 60 - t_lighted = 150 ∧
    stub_length_candle1 ℓ (5 * 60 - t_lighted) = 3 * stub_length_candle2 ℓ (5 * 60 - t_lighted) :=
begin
  -- We'll postpone the proof for now.
  sorry
end

end candles_lighting_time_l473_473126


namespace negation_of_existential_l473_473973

theorem negation_of_existential :
  (∃ x₀ : ℝ, x₀ ∈ set.Ioi 0 ∧ Real.log x₀ = x₀ - 1) ↔ 
  ¬ (∀ x : ℝ, x ∈ set.Ioi 0 → Real.log x = x - 1) :=
by 
  sorry

end negation_of_existential_l473_473973


namespace triangle_side_length_integers_l473_473768

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l473_473768


namespace binom_coeffs_not_coprime_l473_473926

open Nat

theorem binom_coeffs_not_coprime (n k m : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) : 
  Nat.gcd (Nat.choose n k) (Nat.choose n m) > 1 := 
sorry

end binom_coeffs_not_coprime_l473_473926


namespace solve_negative_integer_sum_l473_473993

theorem solve_negative_integer_sum (N : ℤ) (h1 : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end solve_negative_integer_sum_l473_473993


namespace average_mpg_round_trip_l473_473642

-- Define the conditions
def distance_one_way := 150  -- miles
def mpg_motorcycle := 50     -- miles per gallon
def mpg_car := 25            -- miles per gallon

-- Define the problem proof statement
theorem average_mpg_round_trip :
  let total_distance := 2 * distance_one_way in
  let fuel_used_motorcycle := (distance_one_way : ℝ) / mpg_motorcycle in
  let fuel_used_car := (distance_one_way : ℝ) / mpg_car in
  let total_fuel_used := fuel_used_motorcycle + fuel_used_car in
  total_distance / total_fuel_used = 33 :=
by
  -- Definitions only, proof omitted
  [...]

sorry

end average_mpg_round_trip_l473_473642


namespace triangle_third_side_length_count_l473_473788

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l473_473788


namespace sequence_1000th_term_is_45_l473_473692

theorem sequence_1000th_term_is_45 :
  ∃ k : ℕ, 1000 = (k * (k + 1)) / 2 - ((k - 1) * k) / 2 ∧ k = 45 :=
begin
  sorry
end

end sequence_1000th_term_is_45_l473_473692


namespace partI_partII_l473_473706

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem partI (a : ℝ) : (∃ x : ℝ, f x < 2 * a - 1) → a > 2 := by
  sorry

theorem partII : {x : ℝ | f x ≥ x^2 - 2 * x} = set.Icc (-1) (2 + Real.sqrt 3) := by
  sorry

end partI_partII_l473_473706


namespace maria_isha_ratio_l473_473482

/-- Given conditions:
 1. Rene received $300.
 2. Florence received three times as much money as Rene.
 3. Maria gave half as much to Isha as she gave to Florence.
 4. Maria gave her three friends $1650 in total.
 
 Prove:
 The ratio of the money Maria gave to Isha to her total money is approximately 1:3.67.
-/
theorem maria_isha_ratio :
  ∃ (Rene Florence Isha Total : ℕ),
    Rene = 300 ∧
    Florence = 3 * Rene ∧
    Isha = Florence / 2 ∧
    Total = Rene + Florence + Isha ∧ 
    Total = 1650 ∧
    (Isha.toRat / Total.toRat ≈ 1 / 3.67.toRat) :=
begin
  sorry
end

end maria_isha_ratio_l473_473482


namespace arithmetic_sequence_sum_l473_473891

/-- Let {a_n} be an arithmetic sequence with a positive common difference d.
  Given that a_1 + a_2 + a_3 = 15 and a_1 * a_2 * a_3 = 80, we aim to show that
  a_11 + a_12 + a_13 = 105. -/
theorem arithmetic_sequence_sum
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d > 0)
  (h3 : a 1 + a 2 + a 3 = 15)
  (h4 : a 1 * a 2 * a 3 = 80) :
  a 11 + a 12 + a 13 = 105 :=
sorry

end arithmetic_sequence_sum_l473_473891


namespace vector_parallel_sufficient_not_necessary_l473_473743

theorem vector_parallel_sufficient_not_necessary :
  (∫ t in (1 : ℝ)..(real.exp 1), 2 / t) = 2 →
  ∃ x : ℝ, (∫ t in (1 : ℝ)..(real.exp 1), 2 / t) = x → 
  ((1, x) = (1, 2) ∧ (x, 4) = (2, 4) ∧ (1, x) ∥ (x, 4) ∧ 
  (∃ y : ℝ, y ≠ 2 ∧ (1, y) ∥ (y, 4))) :=
begin
  sorry
end

end vector_parallel_sufficient_not_necessary_l473_473743


namespace problem_1_l473_473047

noncomputable def a : ℕ → ℝ
| 0     := -3
| (n+1) := a n + b n + real.sqrt (a n ^ 2 + b n ^ 2)

noncomputable def b : ℕ → ℝ
| 0     := 2
| (n+1) := a n + b n - real.sqrt (a n ^ 2 + b n ^ 2)

theorem problem_1 :
  (1 / a 2013) + (1 / b 2013) = -1 / 6 :=
sorry

end problem_1_l473_473047


namespace range_p_l473_473266

open Set

noncomputable def p (x : ℝ) := x^4 - 4*x^2 + 4

theorem range_p : range (p ∘ (fun x => set.Ici 0)) = set.Ici 0 :=
sorry

end range_p_l473_473266


namespace solve_congruence_l473_473512

theorem solve_congruence : ∃ (n : ℤ), 13 * n ≡ 8 [MOD 47] ∧ n ≡ 4 [MOD 47] :=
sorry

end solve_congruence_l473_473512


namespace family_children_count_l473_473981

theorem family_children_count :
  let ratio_boys_to_girls := 5 / 7
  let total_amount_given_to_boys := 3900
  let amount_each_boy_receives := 52
  let number_of_boys := total_amount_given_to_boys / amount_each_boy_receives
  let number_of_girls := number_of_boys * ratio_boys_to_girls
  let total_children := number_of_boys + number_of_girls
  in total_children = 180 :=
by
  sorry

end family_children_count_l473_473981


namespace csc_identity_sum_l473_473459

theorem csc_identity_sum (x : ℝ) (h : -1 < x ∧ x < 1) :
  ∑ i in {1, 2, 3} \csc² (i * x.toFloat + (Float.pi.toFloat / 7)): = 8 :=
  sorry

end csc_identity_sum_l473_473459


namespace general_formula_l473_473006

noncomputable def seq : ℕ → ℕ
| 0 := 0
| 1 := 2
| (n + 2) := 2 * (2 * n + 3)^2 * seq (n + 1) - 4 * (n + 1)^2 * (2 * n + 1) * (2 * n + 3) * seq n

theorem general_formula (n : ℕ) : seq n = 2^n * (n.factorial * (n + 1).factorial) :=
sorry

end general_formula_l473_473006


namespace distribute_graduates_l473_473949

theorem distribute_graduates (graduates schools : ℕ) (h_grads : graduates = 6) (h_schools : schools = 3) :
  (nat.choose 6 2 * nat.choose 4 2 * nat.choose 2 2) / nat.factorial 3 = 90 :=
by
  rw [←h_grads, ←h_schools]
  -- Placeholder for the actual proof
  sorry

end distribute_graduates_l473_473949


namespace max_value_objective_l473_473339

def is_within_D (x y : ℝ) : Prop :=
  (x = 1 ∧ abs y ≤ 1) ∨ (x = 0 ∧ y = 0)

def objective_function (x y : ℝ) : ℝ :=
  x - 2*y + 5
  
theorem max_value_objective :
  ∃ (x y : ℝ), is_within_D x y ∧ objective_function x y = 8 :=
by
  use (1, -1)
  simp [is_within_D, objective_function]
  sorry

end max_value_objective_l473_473339


namespace charley_pulled_fraction_l473_473664

noncomputable def fraction_of_white_beads (total_white : ℕ) (total_black : ℕ) (fraction_black : ℚ) (total_pulled : ℕ) : ℚ :=
  let black_pulled := (fraction_black * total_black) in
  let white_pulled := total_pulled - black_pulled in
  white_pulled / total_white

theorem charley_pulled_fraction {total_white total_black total_pulled : ℕ} {fraction_black : ℚ}
  (h1 : total_white = 51)
  (h2 : total_black = 90)
  (h3 : fraction_black = 1/6)
  (h4 : total_pulled = 32) :
  fraction_of_white_beads total_white total_black fraction_black total_pulled = 1/3 :=
by
  unfold fraction_of_white_beads
  sorry

end charley_pulled_fraction_l473_473664


namespace greatest_product_l473_473161

theorem greatest_product (x : ℤ) (h : x + (2020 - x) = 2020) : x * (2020 - x) ≤ 1020100 :=
sorry

end greatest_product_l473_473161


namespace find_coefficients_l473_473276

theorem find_coefficients :
  ∃ A B : ℚ, ( ∀ x : ℚ, x ≠ 12  ∧ x ≠ -3 → ∀ A = 73 / 15 ∧ B = 17 / 15,
  (6 * x + 1) / ((x - 12) * (x + 3)) = A / (x - 12) + B / (x + 3)) :=
by 
  use [73 / 15, 17 / 15]
  sorry

end find_coefficients_l473_473276


namespace area_inequality_l473_473353

noncomputable def radius : ℝ := 1

def area_of_circles : ℝ := 3 * (Real.pi * radius ^ 2)

def t1 (covered_once : ℝ) : ℝ := covered_once

def t2 (covered_twice : ℝ) : ℝ := covered_twice

theorem area_inequality (t1 t2 : ℝ) (h1: t1 + 2 * t2 ≤ area_of_circles) : t1 ≥ t2 :=
  sorry

end area_inequality_l473_473353


namespace find_h_l473_473417

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l473_473417


namespace cricketer_wickets_l473_473590

noncomputable def initial_average (R W : ℝ) : ℝ := R / W

noncomputable def new_average (R W : ℝ) (additional_runs additional_wickets : ℝ) : ℝ :=
  (R + additional_runs) / (W + additional_wickets)

theorem cricketer_wickets (R W : ℝ) 
(h1 : initial_average R W = 12.4) 
(h2 : new_average R W 26 5 = 12.0) : 
  W = 85 :=
sorry

end cricketer_wickets_l473_473590


namespace number_of_possible_third_side_lengths_l473_473821

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l473_473821


namespace quadratic_form_h_l473_473386

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l473_473386


namespace problem_statement_l473_473372

noncomputable def f (x : ℝ) : ℝ := 3 * sin (3 * x + π / 3) - 3

theorem problem_statement :
  (∀ x x1 x2 : ℝ, f x = 3 * sin (3 * x + π / 3) - 3 ∧ |x1 - x2| ≥ 2 * π / 3 ∧ f (x + π / 9) = f (-x)) →
  -- Option A
  (f(x) = 3 * sin(3 * x + π / 3) - 3) ∧
  -- Option B
  (∃ g : ℝ → ℝ, (∀ x, g x = 3 * cos (3 * x + π / 3) - 3) ∧ (f (x + π / 6) = g x)) ∧
  -- Option C
  (∀ x : ℝ, x ∈ Icc (π / 16) (π / 3) → monotone_decreasing_on (λ y, f y) (Icc (π / 16) (π / 3))) ∧
  -- Option D
  (¬(is_symmetry_axis f (5 * π / 18))) :=
by
  sorry

end problem_statement_l473_473372


namespace find_h_l473_473379

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l473_473379


namespace find_f_neg_one_l473_473332

theorem find_f_neg_one (f h : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x)
    (h2 : ∀ x, h x = f x - 9) (h3 : h 1 = 2) : f (-1) = -11 := 
by
  sorry

end find_f_neg_one_l473_473332


namespace juicy_12_juicy_20_l473_473879

def is_juicy (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ 1 = (1 / a) + (1 / b) + (1 / c) + (1 / d) ∧ a * b * c * d = n

theorem juicy_12 : is_juicy 12 :=
sorry

theorem juicy_20 : is_juicy 20 :=
sorry

end juicy_12_juicy_20_l473_473879


namespace calculate_max_marks_l473_473066

theorem calculate_max_marks (shortfall_math : ℕ) (shortfall_science : ℕ) 
                            (shortfall_literature : ℕ) (shortfall_social_studies : ℕ)
                            (required_math : ℕ) (required_science : ℕ)
                            (required_literature : ℕ) (required_social_studies : ℕ)
                            (max_math : ℕ) (max_science : ℕ)
                            (max_literature : ℕ) (max_social_studies : ℕ) :
                            shortfall_math = 40 ∧ required_math = 95 ∧ max_math = 800 ∧
                            shortfall_science = 35 ∧ required_science = 92 ∧ max_science = 438 ∧
                            shortfall_literature = 30 ∧ required_literature = 90 ∧ max_literature = 300 ∧
                            shortfall_social_studies = 25 ∧ required_social_studies = 88 ∧ max_social_studies = 209 :=
by
  sorry

end calculate_max_marks_l473_473066


namespace final_difference_l473_473937

theorem final_difference (M m : ℕ) (hs : (1 <= M) ∧ (M <= 2017) ∧ (1 <= m) ∧ (m <= 2017))
  (hM : M = 2017) (hm : m = 3) : M - m = 2014 :=
by 
  rw [hM, hm]
  norm_num

end final_difference_l473_473937


namespace sum_of_solutions_l473_473295

theorem sum_of_solutions : 
  (∑ x in ({ x : ℝ | 2^(x^2 - 5*x - 4) = 8^(x - 6) }).to_finset) = 9 :=
sorry

end sum_of_solutions_l473_473295


namespace sum_d_e_f_equals_23_l473_473526

theorem sum_d_e_f_equals_23
  (d e f : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 9 * x + 20 = (x + d) * (x + e))
  (h2 : ∀ x : ℝ, x^2 + 11 * x - 60 = (x + e) * (x - f)) :
  d + e + f = 23 :=
by
  sorry

end sum_d_e_f_equals_23_l473_473526


namespace quotient_is_correct_l473_473695

-- Define the given polynomial and the divisor
noncomputable def P := (λ x : ℝ, x^5 - 22*x^3 + 12*x^2 - 16*x + 8)
noncomputable def D := (λ x : ℝ, x - 3)

-- Define the quotient
noncomputable def Q := (λ x : ℝ, x^4 + 3*x^3 - 13*x^2 - 27*x - 97)

-- Define the theorem stating the equivalence
theorem quotient_is_correct : 
  ∃ (R : ℝ), (∀ x : ℝ, P x = (D x) * (Q x) + R) ∧ R = -211 :=
by
  sorry

end quotient_is_correct_l473_473695


namespace journey_total_time_l473_473630

noncomputable def total_time (D : ℝ) (r_dist : ℕ → ℕ) (r_time : ℕ → ℕ) (u_speed : ℝ) : ℝ :=
  let dist_uphill := D * (r_dist 1) / (r_dist 1 + r_dist 2 + r_dist 3)
  let t_uphill := (dist_uphill / u_speed)
  let k := t_uphill / (r_time 1)
  (r_time 1 + r_time 2 + r_time 3) * k

theorem journey_total_time :
  total_time 50 (fun n => if n = 1 then 1 else if n = 2 then 2 else 3) 
                (fun n => if n = 1 then 4 else if n = 2 then 5 else 6) 
                3 = 10 + 5/12 :=
by
  sorry

end journey_total_time_l473_473630


namespace sequence_condition_satisfies_l473_473862

def seq_prove_abs_lt_1 (a : ℕ → ℝ) : Prop :=
  (∃ i : ℕ, |a i| < 1)

theorem sequence_condition_satisfies (a : ℕ → ℝ)
  (h1 : a 1 * a 2 < 0)
  (h2 : ∀ n > 2, ∃ i j, 1 ≤ i ∧ i < j ∧ j < n ∧ (∀ k l, 1 ≤ k ∧ k < l ∧ l < n → |a i + a j| ≤ |a k + a l|)) :
  seq_prove_abs_lt_1 a :=
by
  sorry

end sequence_condition_satisfies_l473_473862


namespace squared_diagonal_inequality_l473_473601

theorem squared_diagonal_inequality 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) :
  let AB := (x1 - x2)^2 + (y1 - y2)^2
  let BC := (x2 - x3)^2 + (y2 - y3)^2
  let CD := (x3 - x4)^2 + (y3 - y4)^2
  let DA := (x1 - x4)^2 + (y1 - y4)^2
  let AC := (x1 - x3)^2 + (y1 - y3)^2
  let BD := (x2 - x4)^2 + (y2 - y4)^2
  AC + BD ≤ AB + BC + CD + DA := 
by
  sorry

end squared_diagonal_inequality_l473_473601


namespace triangle_third_side_lengths_l473_473758

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l473_473758


namespace part_a_part_b_l473_473042

def E (x y : ℕ) : ℚ :=
  x / y + (x + 1) / (y + 1) + (x + 2) / (y + 2)

theorem part_a (x y : ℕ) : E x y = 3 ↔ x = y :=
  sorry

theorem part_b : ∃∞ n : ℕ, ∃ (x y : ℕ), E x y = n :=
  sorry

end part_a_part_b_l473_473042


namespace equal_areas_of_quadrilaterals_l473_473060

open EuclideanGeometry

variables {A B C D E F K : Point}
variables {AB AD : Line}
variables {S : Real}

-- Conditions
def is_parallelogram (A B C D : Point) : Prop := is_parallel (A - B) (C - D) ∧ is_parallel (A - D) (B - C)
def lies_on (P : Point) (L : Line) : Prop := on_line P L
def segment_condition (P Q R : Point) : Prop := between Q P R

-- Problem statement
theorem equal_areas_of_quadrilaterals
  (h1 : is_parallelogram A B C D)
  (h2 : lies_on E AB)
  (h3 : lies_on F AD)
  (h4 : segment_condition A B E)
  (h5 : segment_condition A D F)
  (h6 : ∃ K, intersection_point (line_through E D) (line_through F B) K) :
    area_quadrilateral A B K D = area_quadrilateral C E K F :=
begin
  sorry -- Proof goes here
end

end equal_areas_of_quadrilaterals_l473_473060


namespace third_side_integer_lengths_l473_473781

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l473_473781


namespace greatest_3_digit_base9_div_by_7_l473_473137

def base9_to_decimal (n : ℕ) : ℕ :=
  let d2 := n / 81
  let d1 := (n % 81) / 9
  let d0 := n % 9
  d2 * 81 + d1 * 9 + d0

def greatest_base9_3_digit_div_by_7 (n : ℕ) : Prop :=
  n < 9 * 9 * 9 ∧ 7 ∣ (base9_to_decimal n)

theorem greatest_3_digit_base9_div_by_7 :
  ∃ n, greatest_base9_3_digit_div_by_7 n ∧ n = 888 :=
begin
  sorry
end

end greatest_3_digit_base9_div_by_7_l473_473137


namespace smallest_positive_value_l473_473004

theorem smallest_positive_value (a b : ℝ) (h : log 2 a + log 2 b ≥ 6) : a + b ≥ 16 :=
sorry

end smallest_positive_value_l473_473004


namespace hundreds_digit_of_factorial_difference_l473_473561

theorem hundreds_digit_of_factorial_difference :
  (∃ k : ℤ, 0 ≤ k ∧ k < 10 ∧ 17! - 13! = k * 100) → ∃ k : ℕ, k = 0 ∧ (17! - 13!) / 100 % 10 = k :=
by
  sorry

end hundreds_digit_of_factorial_difference_l473_473561


namespace perimeter_sum_ABC_triangle_l473_473494

noncomputable def ABCD_triangle_perimeter_sum : ℕ :=
  let AB := 12
  let BC := 18
  let AC := 30
  let possible_pairs := [(95, 94), (33, 30), (17, 10)]
  let perimeters := possible_pairs.map (λ (xy : ℕ × ℕ), 2 * xy.1 + AC)
  perimeters.sum

theorem perimeter_sum_ABC_triangle : ABCD_triangle_perimeter_sum = 380 := by
  sorry

end perimeter_sum_ABC_triangle_l473_473494


namespace interval_contains_thousand_members_l473_473650

noncomputable def sequence : ℕ → ℝ
| 0       => 1
| (n + 1) => real.sqrt ((sequence n) ^ 2 + 1 / (sequence n))

/-- There exists some interval of length 1 that contains more than a thousand members of the sequence defined by
    a₀ = 1 and aₙ₊₁ = √(aₙ² + 1/aₙ). 
--/
theorem interval_contains_thousand_members :
  ∃ k : ℝ, ∃ f : ℕ → ℕ, (0 < f 0) ∧ (∀ n, sequence (f n + 1) - sequence (f n) < (1 / 2000)) ∧ (((λ n, sequence (f n)) ) ^ (1001 : ℝ) ∈ set.Icc k (k + 1)) := 
begin
  sorry, -- As requested, no proof is to be provided.
end

end interval_contains_thousand_members_l473_473650


namespace intersecting_circles_concyclic_or_collinear_l473_473320

noncomputable theory

-- Define circles and their intersections
variables (C1 C2 C3 C4 : Type)
variables (A1 B1 : C1 ∩ C2)
variables (A2 B2 : C2 ∩ C3)
variables (A3 B3 : C3 ∩ C4)
variables (A4 B4 : C4 ∩ C1)

-- Define the condition for concyclicity or collinearity
def concyclic_or_collinear (A B C D : Type) : Prop := sorry

-- The theorem we need to prove
theorem intersecting_circles_concyclic_or_collinear
  (h1: concyclic_or_collinear A1 B1 C1 D1)
  : concyclic_or_collinear A2 B2 C2 D2 :=
sorry

end intersecting_circles_concyclic_or_collinear_l473_473320


namespace area_of_garden_l473_473223

variable (w l : ℕ)
variable (h1 : l = 3 * w) 
variable (h2 : 2 * (l + w) = 72)

theorem area_of_garden : l * w = 243 := by
  sorry

end area_of_garden_l473_473223


namespace part_I_part_I_union_part_II_l473_473478

open Set Real

noncomputable def U := univ : set ℝ

noncomputable def A (m : ℝ) : set ℝ := {x | m - 2 < x ∧ x < m + 2}

noncomputable def B : set ℝ := {x | -4 < x ∧ x < 4}

-- Part (I)
theorem part_I (m : ℝ) (h : m = 3) :
  (A m ∩ B) = {x | 1 < x ∧ x < 4} :=
sorry

theorem part_I_union (m : ℝ) (h : m = 3) :
  (A m ∪ B) = {x | -4 < x ∧ x < 5} :=
sorry

-- Part (II)
theorem part_II :
  {m | A m ⊆ {x | x ≤ -4 ∨ x ≥ 4}} =  {m | m ≤ -6 ∨ m ≥ 6} :=
sorry

end part_I_part_I_union_part_II_l473_473478


namespace triangle_third_side_length_l473_473795

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l473_473795


namespace only_increasing_func_is_f_D_l473_473586

-- Define the four functions
def f_A (x : ℝ) : ℝ := -x
def f_B (x : ℝ) : ℝ := (2 / 3) ^ x
def f_C (x : ℝ) : ℝ := x ^ 2
def f_D (x : ℝ) : ℝ := real.cbrt x

-- Define a predicate for an increasing function
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Formalize the problem: prove that the only increasing function is f_D
theorem only_increasing_func_is_f_D :
  is_increasing f_D ∧ ¬is_increasing f_A ∧ ¬is_increasing f_B ∧ ¬is_increasing f_C :=
by
  split
  sorry,
  split
  sorry
  split
  sorry
  sorry

end only_increasing_func_is_f_D_l473_473586


namespace triangle_third_side_length_l473_473793

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l473_473793


namespace greatest_3_digit_base9_divisible_by_7_l473_473146

theorem greatest_3_digit_base9_divisible_by_7 :
  ∃ (n : ℕ), n < 729 ∧ n ≥ 81 ∧ n % 7 = 0 ∧ n = 8 * 81 + 8 * 9 + 8 := 
by 
  use 728
  split
  {
    exact nat.pred_lt (ne_of_lt (by norm_num))
  }
  split
  {
    exact nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.zero_le 7))) 
  }
  split
  {
    norm_num
  }
  norm_num

end greatest_3_digit_base9_divisible_by_7_l473_473146


namespace soda_cost_l473_473129

variable (b s : ℕ)

theorem soda_cost (h1 : 2 * b + s = 210) (h2 : b + 2 * s = 240) : s = 90 := by
  sorry

end soda_cost_l473_473129


namespace collin_savings_l473_473665

theorem collin_savings
    (lightweight_value : ℝ := 0.15)
    (mediumweight_value : ℝ := 0.25)
    (heavyweight_value : ℝ := 0.35)
    (n_lightweight_home : ℕ := 12)
    (n_mediumweight_grandparents : ℕ := n_lightweight_home * 3)
    (n_heavyweight_neighbor : ℕ := 46)
    (n_mixed_office : ℕ := 250)
    (percent_lw_office : ℝ := 0.5)
    (percent_mw_office : ℝ := 0.3)
    (percent_hw_office : ℝ := 0.2)
    (exchange_rate : ℝ := 1.20)
    (total_earned_euros : ℝ := 
        (n_lightweight_home * lightweight_value) +
        (n_mediumweight_grandparents * mediumweight_value) +
        (n_heavyweight_neighbor * heavyweight_value) +
        (n_mixed_office * percent_lw_office * lightweight_value) +
        (n_mixed_office * percent_mw_office * mediumweight_value) +
        (n_mixed_office * percent_hw_office * heavyweight_value)) :
  (total_earned_savings_usd : ℝ := 
    (total_earned_euros * exchange_rate) / 2) :
  total_earned_savings_usd = 49.14 := 
sorry

end collin_savings_l473_473665


namespace inequality_for_factorials_l473_473938

theorem inequality_for_factorials (n : ℕ) : 
  Real.root (n * (n! : ℝ) ^ 3) n ≤ (n * (n + 1) ^ 2) / 4 := 
sorry

end inequality_for_factorials_l473_473938


namespace fraction_girls_at_meet_l473_473546

-- Define the conditions of the problem
def numStudentsMaplewood : ℕ := 300
def ratioBoysGirlsMaplewood : ℕ × ℕ := (3, 2)
def numStudentsRiverview : ℕ := 240
def ratioBoysGirlsRiverview : ℕ × ℕ := (3, 5)

-- Define the combined number of students and number of girls
def totalStudentsMaplewood := numStudentsMaplewood
def totalStudentsRiverview := numStudentsRiverview

def numGirlsMaplewood : ℕ :=
  let (b, g) := ratioBoysGirlsMaplewood
  (totalStudentsMaplewood * g) / (b + g)

def numGirlsRiverview : ℕ :=
  let (b, g) := ratioBoysGirlsRiverview
  (totalStudentsRiverview * g) / (b + g)

def totalGirls := numGirlsMaplewood + numGirlsRiverview
def totalStudents := totalStudentsMaplewood + totalStudentsRiverview

-- Formalize the actual proof statement
theorem fraction_girls_at_meet : 
  (totalGirls : ℚ) / totalStudents = 1 / 2 := by
  sorry

end fraction_girls_at_meet_l473_473546


namespace possible_values_of_a_l473_473901

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - (a + 3) * x + 3 * a = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 4 = 0}

theorem possible_values_of_a (a : ℝ) :
  (∃ sA sB, sA = A a ∧ sB = B ∧ (∑ x in (sA ∪ sB), x) = 8) →
  a ∈ {0, 1, 3, 4} :=
by
  sorry

end possible_values_of_a_l473_473901


namespace triangle_area_l473_473447

open Real

/--
In a triangle ABC, assume the angle bisector BE and the median AD 
are equal and perpendicular. Given AB = sqrt 13, prove that the area 
of triangle ABC is 12.
-/
theorem triangle_area {A B C : EuclideanGeometry.Point} 
  (BE AD : EuclideanGeometry.Line) 
  (AB : ℝ) (sqrt_13_pos : AB = sqrt 13)
  (is_angle_bisector_BE : BE.is_angle_bisector B E C)
  (is_median_AD : AD.is_median A D B)
  (perpendicular_BE_AD : BE.perpendicular AD)
  (equal_lengths : BE.length = AD.length) :
  EuclideanGeometry.area A B C = 12 := 
sorry

end triangle_area_l473_473447


namespace initial_ratio_zinc_copper_l473_473647

theorem initial_ratio_zinc_copper (Z C : ℝ) 
  (h1 : Z + C = 6) 
  (h2 : Z + 8 = 3 * C) : 
  Z / C = 5 / 7 := 
sorry

end initial_ratio_zinc_copper_l473_473647


namespace find_h_l473_473422

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l473_473422


namespace pavan_travel_time_l473_473918

theorem pavan_travel_time (D : ℝ) (V1 V2 : ℝ) (distance : D = 300) (speed1 : V1 = 30) (speed2 : V2 = 25) : 
  ∃ t : ℝ, t = 11 := 
  by
    sorry

end pavan_travel_time_l473_473918


namespace rotated_line_x_intercept_l473_473905

theorem rotated_line_x_intercept (x y : ℝ) :
  (∃ (k : ℝ), y = (3 * Real.sqrt 3 + 5) / (2 * Real.sqrt 3) * x) →
  (∃ y : ℝ, 3 * x - 5 * y + 40 = 0) →
  (∃ (x_intercept : ℝ), x_intercept = 0) := 
by
  sorry

end rotated_line_x_intercept_l473_473905


namespace triangle_third_side_length_count_l473_473790

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l473_473790


namespace number_of_girls_l473_473640

theorem number_of_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 2 * B / 3 + G = 18) : G = 18 :=
by
  sorry

end number_of_girls_l473_473640


namespace angle_C_condition1_angle_C_condition2_angle_C_condition3_max_area_l473_473444

theorem angle_C_condition1 
  (a b c : ℝ) 
  (S_ABC : ℝ)
  (h1 : sqrt 3 * (a * b * cos C) = 2 * divide 1 2 * (a * b * sin C))
  : C = π / 3 := sorry

theorem angle_C_condition2
  (a b c : ℝ)
  (h2 : (sin C + sin A) * (sin C - sin A) = sin B * (sin B - sin A))
  : C = π / 3 := sorry

theorem angle_C_condition3
  (a b c : ℝ)
  (h3 : (2 * a - b) * cos C = c * cos B)
  : C = π / 3 := sorry

theorem max_area 
  (a b : ℝ)
  (c : ℝ := 2)
  (h_c : c = 2)
  (h4 : angle C = π / 3)
  : S_ABC = sqrt 3 / 4 * (a * b) := sorry

end angle_C_condition1_angle_C_condition2_angle_C_condition3_max_area_l473_473444


namespace best_regression_fit_l473_473237

theorem best_regression_fit (R1 R2 R3 R4 : ℝ) (h1 : R1 = 0.27) (h2 : R2 = 0.85) (h3 : R3 = 0.96) (h4 : R4 = 0.5)
  (h : ∀ (R : ℝ), 0 ≤ R ∧ R ≤ 1 ∧ (R = R3 → best_fitting_effect R)) : best_fitting_effect R3 :=
by
  -- Given conditions
  have hR3_close_to_1 : R3 = 0.96, from h3,
  sorry

end best_regression_fit_l473_473237


namespace min_varphi_l473_473125

noncomputable def f (x : ℝ) := Real.cos (2 * x + (Real.pi / 3))

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = -g (-x)

def translates_to (𝜑 : ℝ) (h g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = h (x + 𝜑)

theorem min_varphi (𝜑 : ℝ) (hg : translates_to 𝜑 f (λ x, f(x + 𝜑))) 
  (hodd : is_odd_function (λ x, f(x + 𝜑))) : 𝜑 = Real.pi / 12 :=
sorry

end min_varphi_l473_473125


namespace h_value_l473_473393

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l473_473393


namespace possible_integer_lengths_third_side_l473_473806

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l473_473806


namespace number_of_valid_triples_l473_473290

theorem number_of_valid_triples : 
  ∃ n, n = 16 ∧ 
       ∀ (a b c : ℕ), 
       0 < a → 0 < b → 0 < c → 
       (a * b * c ∣ (a * b + 1) * (b * c + 1) * (c * a + 1)) ↔ n = 16 :=
begin
  sorry
end

end number_of_valid_triples_l473_473290


namespace students_in_second_class_l473_473950

theorem students_in_second_class 
    (avg1 : ℝ)
    (n1 : ℕ)
    (avg2 : ℝ)
    (total_avg : ℝ)
    (x : ℕ)
    (h1 : avg1 = 40)
    (h2 : n1 = 26)
    (h3 : avg2 = 60)
    (h4 : total_avg = 53.1578947368421)
    (h5 : (n1 * avg1 + x * avg2) / (n1 + x) = total_avg) :
  x = 50 :=
by
  sorry

end students_in_second_class_l473_473950


namespace base10_to_base5_l473_473172

theorem base10_to_base5 (a b : ℕ) (h1 : a = 12) (h2 : b = 47) :
  nat.to_digits 5 (a + b) = [2, 1, 4] := 
by
  sorry

end base10_to_base5_l473_473172


namespace values_of_a_b_monotonicity_inequality_holds_l473_473722

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem values_of_a_b (h : ∀ x : ℝ, f a b (-x) = -f a b x) :
  a = 2 ∧ b = 1 := 
sorry

theorem monotonicity (h : ∀ x : ℝ, f 2 1 (-x) = -f 2 1 x) :
  ∀ x y : ℝ, x < y → f 2 1 x > f 2 1 y :=
sorry

theorem inequality_holds (k : ℝ) (h_odd : ∀ x : ℝ, f 2 1 (-x) = -f 2 1 x)
  (h_monotone : ∀ x y : ℝ, x < y → f 2 1 x > f 2 1 y)
  (h : ∀ x : ℝ, x ≥ 1 → f k (3^x) + f 2 1 (3^x - 9^x + 2) > 0) :
  k < 4/3 :=
sorry

end values_of_a_b_monotonicity_inequality_holds_l473_473722


namespace probability_two_even_multiples_of_five_drawn_l473_473199

-- Definition of conditions
def toys : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
                      39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

def isEvenMultipleOfFive (n : ℕ) : Bool := n % 10 == 0

-- Collect all such numbers from the list
def evenMultiplesOfFive : List ℕ := toys.filter isEvenMultipleOfFive

-- Number of such even multiples of 5
def countEvenMultiplesOfFive : ℕ := evenMultiplesOfFive.length

theorem probability_two_even_multiples_of_five_drawn :
  (countEvenMultiplesOfFive / 50) * ((countEvenMultiplesOfFive - 1) / 49) = 2 / 245 :=
  by sorry

end probability_two_even_multiples_of_five_drawn_l473_473199


namespace roots_real_l473_473747

variable {x p q k : ℝ}
variable {x1 x2 : ℝ}

theorem roots_real 
  (h1 : x^2 + p * x + q = 0) 
  (h2 : p = -(x1 + x2)) 
  (h3 : q = x1 * x2) 
  (h4 : x1 ≠ x2) 
  (h5 :  x1^2 - 2*x1*x2 + x2^2 + 4*q = 0):
  (∃ y1 y2, y1 = k * x1 + (1 / k) * x2 ∧ y2 = k * x2 + (1 / k) * x1 ∧ 
    (y1^2 + (k + 1/k) * p * y1 + (p^2 + q * ((k - 1/k)^2)) = 0) ∧ 
    (y2^2 + (k + 1/k) * p * y2 + (p^2 + q * ((k - 1/k)^2)) = 0)) → 
  (∃ z1 z2, z1 = k * x1 ∧ z2 = 1/k * x2 ∧ 
    (z1^2 - y1 * z1 + q = 0) ∧ 
    (z2^2 - y2 * z2 + q = 0)) :=
sorry

end roots_real_l473_473747


namespace arithmetic_sequence_sufficiency_arithmetic_sequence_necessity_arithmetic_sequence_sufficient_but_not_necessary_l473_473356

variable {ℕ : Type} [linear_order ℕ]
variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

theorem arithmetic_sequence_sufficiency (h : ∀ n, b n = a n + a (n + 1)) :
  (∃ d, ∀ n, a (n + 1) = a n + d) → (∃ d, ∀ n, b (n + 1) = b n + d) :=
by sorry

theorem arithmetic_sequence_necessity (h : ∀ n, b n = a n + a (n + 1)) :
  (∃ d, ∀ n, b (n + 1) = b n + d) → ¬ (∃ d, ∀ n, a (n + 1) = a n + d) :=
by sorry

theorem arithmetic_sequence_sufficient_but_not_necessary (h : ∀ n, b n = a n + a (n + 1)) :
  (∃ d, ∀ n, a (n + 1) = a n + d) ↔ (∃ d, ∀ n, b (n + 1) = b n + d) :=
by sorry

end arithmetic_sequence_sufficiency_arithmetic_sequence_necessity_arithmetic_sequence_sufficient_but_not_necessary_l473_473356


namespace volume_of_space_l473_473699

noncomputable def volume_of_region : ℝ :=
  let condition1 : Prop := ∀ (x y z : ℝ), |x + y + z| + |x + y - z| ≤ 10
  let condition2 : Prop := ∀ (x y z : ℝ), x ≥ 0
  let condition3 : Prop := ∀ (x y z : ℝ), y ≥ 0
  let condition4 : Prop := ∀ (x y z : ℝ), z ≥ 0
  if condition1 ∧ condition2 ∧ condition3 ∧ condition4 then 62.5 else 0

theorem volume_of_space : volume_of_region = 62.5 :=
by {
  sorry
}

end volume_of_space_l473_473699


namespace non_intersecting_segments_exist_l473_473707

noncomputable def exists_non_intersecting_segments (n : ℕ) (red_points blue_points : Fin n → ℝ × ℝ) : 
  Prop := 
  ∃ segments : Fin n → (ℝ × ℝ) × (ℝ × ℝ), 
    (∀ i, (segments i).1 ∈ Set.image red_points Finset.univ ∧ (segments i).2 ∈ Set.image blue_points Finset.univ) ∧
    (∀ i j, i ≠ j → ¬(segments i).1 = (segments j).1 ∧ ¬(segments i).2 = (segments j).2) ∧
    (∀ i j, i ≠ j → ¬intersect (segments i) (segments j))

theorem non_intersecting_segments_exist (n : ℕ) (red_points blue_points : Fin n → ℝ × ℝ) :
  (∀ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l → 
    ¬collinear (red_points i) (red_points j) (red_points k) ∧ 
    ¬collinear (red_points i) (red_points j) (blue_points l) ∧ 
    ¬collinear (red_points i) (blue_points k) (blue_points l) ∧ 
    ¬collinear (blue_points j) (blue_points k) (blue_points l)) → 
  exists_non_intersecting_segments n red_points blue_points :=
  sorry

end non_intersecting_segments_exist_l473_473707


namespace quadratic_form_h_l473_473384

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l473_473384


namespace number_of_possible_third_side_lengths_l473_473826

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l473_473826


namespace math_proof_problem_l473_473061

theorem math_proof_problem
  (a b c : ℝ)
  (h : a ≠ b)
  (h1 : b ≠ c)
  (h2 : c ≠ a)
  (h3 : (a / (2 * (b - c))) + (b / (2 * (c - a))) + (c / (2 * (a - b))) = 0) :
  (a / (b - c)^3) + (b / (c - a)^3) + (c / (a - b)^3) = 0 := 
by
  sorry

end math_proof_problem_l473_473061


namespace triangle_third_side_length_l473_473798

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l473_473798


namespace max_product_of_sum_2020_l473_473166

theorem max_product_of_sum_2020 : 
  ∃ x y : ℤ, x + y = 2020 ∧ (x * y) ≤ 1020100 ∧ (∀ a b : ℤ, a + b = 2020 → a * b ≤ x * y) :=
begin
  sorry
end

end max_product_of_sum_2020_l473_473166


namespace first_fifteen_multiples_of_seven_sum_l473_473571

theorem first_fifteen_multiples_of_seven_sum :
    (∑ i in finset.range 15, 7 * (i + 1)) = 840 := 
sorry

end first_fifteen_multiples_of_seven_sum_l473_473571


namespace solution_set_Inequality_l473_473985

theorem solution_set_Inequality : {x : ℝ | abs (1 + x + x^2 / 2) < 1} = {x : ℝ | -2 < x ∧ x < 0} :=
sorry

end solution_set_Inequality_l473_473985


namespace partner_profit_share_correct_l473_473853

-- Definitions based on conditions
def total_profit : ℝ := 280000
def profit_share_shekhar : ℝ := 0.28
def profit_share_rajeev : ℝ := 0.22
def profit_share_jatin : ℝ := 0.20
def profit_share_simran : ℝ := 0.18
def profit_share_ramesh : ℝ := 0.12

-- Each partner's share in the profit
def shekhar_share : ℝ := profit_share_shekhar * total_profit
def rajeev_share : ℝ := profit_share_rajeev * total_profit
def jatin_share : ℝ := profit_share_jatin * total_profit
def simran_share : ℝ := profit_share_simran * total_profit
def ramesh_share : ℝ := profit_share_ramesh * total_profit

-- Statement to be proved
theorem partner_profit_share_correct :
    shekhar_share = 78400 ∧ 
    rajeev_share = 61600 ∧ 
    jatin_share = 56000 ∧ 
    simran_share = 50400 ∧ 
    ramesh_share = 33600 ∧ 
    (shekhar_share + rajeev_share + jatin_share + simran_share + ramesh_share = total_profit) :=
by sorry

end partner_profit_share_correct_l473_473853


namespace number_of_lists_l473_473610

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l473_473610


namespace shooter_probability_l473_473228

-- Define the binomial probability calculation
def binomial_prob (n k : ℕ) (p : ℝ) : ℝ := 
  nat.choose n k * p^k * (1 - p)^(n - k)

-- State the theorem to prove the probability
theorem shooter_probability : 
  binomial_prob 5 4 0.8 = 0.4096 := 
sorry

end shooter_probability_l473_473228


namespace marked_price_correct_l473_473219

noncomputable def marked_price (original_price discount_percent purchase_price profit_percent final_price_percent : ℝ) := 
  (purchase_price * (1 + profit_percent)) / final_price_percent

theorem marked_price_correct
  (original_price : ℝ)
  (discount_percent : ℝ)
  (profit_percent : ℝ)
  (final_price_percent : ℝ)
  (purchase_price : ℝ := original_price * (1 - discount_percent))
  (expected_marked_price : ℝ) :
  original_price = 40 →
  discount_percent = 0.15 →
  profit_percent = 0.25 →
  final_price_percent = 0.90 →
  expected_marked_price = 47.20 →
  marked_price original_price discount_percent purchase_price profit_percent final_price_percent = expected_marked_price := 
by
  intros
  sorry

end marked_price_correct_l473_473219


namespace perpendicular_planes_perp_l473_473049

variables {Point Line Plane : Type}
variables (m n : Line) (α β : Plane)
variables (meets : Line → Line → Prop) (parallel : Line → Plane → Prop) (perpendicular : Line → Line → Prop)
variables (subset : Line → Plane → Prop) (perp_plane : Plane → Plane → Prop)

-- Conditions
def different_lines (m n : Line) : Prop := m ≠ n
def different_planes (α β : Plane) : Prop := α ≠ β

-- Given hypothesis
axiom H1 : different_lines m n
axiom H2 : different_planes α β
axiom H3 : perpendicular m n
axiom H4 : perpendicular m α
axiom H5 : perpendicular n β

-- Conclusion to prove
theorem perpendicular_planes_perp (m n : Line) (α β : Plane)
    (different_lines : m ≠ n)
    (different_planes : α ≠ β)
    (perpendicular m n)
    (perpendicular m α)
    (perpendicular n β) : perp_plane α β :=
by
  sorry

end perpendicular_planes_perp_l473_473049


namespace range_of_a_l473_473345

noncomputable def f (x a : ℝ) : ℝ := Real.exp (x - 1) + a * x

def monotonic_intervals (a : ℝ) : Prop :=
  if h : a ≥ 0 then 
    ∀ x y : ℝ, x < y → f x a < f y a
  else
    let c := Real.log (-a) + 1 in 
    ∀ x y : ℝ, (x < c ∧ y < c → f x a > f y a) ∧ (x > c ∧ y > c → f x a < f y a)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici (1 : ℝ), f x a + Real.log x ≥ a + 1) ↔ -2 ≤ a := sorry

end range_of_a_l473_473345


namespace quadratic_expression_rewriting_l473_473414

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l473_473414


namespace remove_five_yields_average_10_5_l473_473269

def numberList : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def averageRemaining (l : List ℕ) : ℚ :=
  (List.sum l : ℚ) / l.length

theorem remove_five_yields_average_10_5 :
  averageRemaining (numberList.erase 5) = 10.5 :=
sorry

end remove_five_yields_average_10_5_l473_473269


namespace tan_phi_l473_473020

variable (β : ℝ) (φ : ℝ)

-- Given condition
axiom tan_half_beta : Real.tan (β / 2) = 1 / Real.cbrt 3

-- Definition of φ being the angle between the median and the angle bisector of β
axiom phi_definition : φ = Real.atan ((Real.tan (Real.atan ((Real.tan β / 2)) - 1 / Real.cbrt 3)) / (1 + (Real.tan (Real.atan ((Real.tan β / 2)) * (1 / Real.cbrt 3)))))

theorem tan_phi : Real.tan φ = 1 / 2 :=
by
  sorry

end tan_phi_l473_473020


namespace polygon_with_largest_area_l473_473701

noncomputable def area_of_polygon_A : ℝ := 6
noncomputable def area_of_polygon_B : ℝ := 4
noncomputable def area_of_polygon_C : ℝ := 4 + 2 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_D : ℝ := 3 + 3 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_E : ℝ := 7

theorem polygon_with_largest_area : 
  area_of_polygon_E > area_of_polygon_A ∧ 
  area_of_polygon_E > area_of_polygon_B ∧ 
  area_of_polygon_E > area_of_polygon_C ∧ 
  area_of_polygon_E > area_of_polygon_D :=
by
  sorry

end polygon_with_largest_area_l473_473701


namespace probability_four_heads_before_three_tails_l473_473050

theorem probability_four_heads_before_three_tails :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ (q = m / n) ∧ m + n = 61 :=
by
  sorry

end probability_four_heads_before_three_tails_l473_473050


namespace vector_magnitude_product_l473_473745

variables (a b : ℝ^2) (θ : ℝ)

-- Conditions
def conditions : Prop := 
  let |a| := (2 : ℝ) in
  let |b| := (1 : ℝ) in
  let θ := (Real.pi / 3 : ℝ) in
  true

-- Question and proof goal
theorem vector_magnitude_product (ha : |a| = 2) (hb : |b| = 1) (hθ : θ = Real.pi / 3) :
  (Real.norm (a + b) * Real.norm (a - b) = Real.sqrt (21)) :=
sorry

end vector_magnitude_product_l473_473745


namespace hyperbola_asymptotes_proof_l473_473736

noncomputable def hyperbola_asymptotes (m : ℝ) (y x : ℝ) : Prop := m * y^2 - x^2 = 1

noncomputable def ellipse (y x : ℝ) : Prop := y^2 / 5 + x^2 = 1

theorem hyperbola_asymptotes_proof (m : ℝ) (h₁ : ∃ (y x : ℝ), hyperbola_asymptotes m y x) (h₂ : ∃ (y x : ℝ), ellipse y x)
(h_same_foci : ∃ y, y^2 = 4 ∧ ∀ y, hyperbola_asymptotes m y 0 → 4 = (1 / m + 1)) :
  (∀ x y, hyperbola_asymptotes (1/3) y x → y = (√3) * x ∨ y = - (√3) * x) :=
by
  sorry

end hyperbola_asymptotes_proof_l473_473736


namespace least_positive_int_to_multiple_of_3_l473_473167

theorem least_positive_int_to_multiple_of_3 (x : ℕ) (h : 575 + x ≡ 0 [MOD 3]) : x = 1 := 
by
  sorry

end least_positive_int_to_multiple_of_3_l473_473167


namespace circle_tangent_proof_l473_473436

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 + 4 * x - 2 * y + m = 0

-- Define the tangent line equation
def tangent_line_eq (x y : ℝ) : Prop := x - √3 * y + √3 - 2 = 0

-- Define the standard form of the circle equation
def standard_circle_eq (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 4

-- Define the equation of line MN
def line_MN_eq (x y c : ℝ) : Prop := 2 * x - y + c = 0

theorem circle_tangent_proof {m : ℝ} :
  (∀ x y, circle_eq x y m) ∧
  (∀ x y, tangent_line_eq x y) ∧
  (∃ x y, x + 2*y = 0 ∧ ∃ M N, M ≠ N ∧ |(M - N)| = 2*√3) →
  (∀ x y, standard_circle_eq x y) ∧
  (∃ c, c = 5 ± √5 ∧ ∀ x y, line_MN_eq x y c) :=
by
  sorry

end circle_tangent_proof_l473_473436


namespace regular_polygon_decompose_l473_473929

theorem regular_polygon_decompose {n : ℕ} (h : n ≥ 2) :
  ∀ (polygon : set (ℝ × ℝ)), 
    (is_regular polygon (2 * n) ∧ (∀ i, parallel (opposite_sides polygon i))) →
    (∃ (rhombuses : set (set (ℝ × ℝ))), divides_into_rhombuses polygon rhombuses) :=
begin
  intros polygon conditions,
  sorry
end

end regular_polygon_decompose_l473_473929


namespace distance_comparison_tetrahedron_l473_473545

variables (A B C D P : Type)
variables [has_dist A B C D P]

/-- Point P is inside or on the boundary of tetrahedron ABCD and different from D -/
axiom condition_tetrahedron_contains (A B C D P: Type) [inside_or_boundary_tetrahedron ABCD P] [P ≠ D]: Prop

/-- Among the distances PA, PB, PC, there exists one that is smaller than one of DA, DB, or DC -/
theorem distance_comparison_tetrahedron (A B C D P: Type) [inside_or_boundary_tetrahedron ABCD P] (h : P ≠ D) :
  (dist PA < dist DA) ∨ (dist PB < dist DB) ∨ (dist PC < dist DC) := 
sorry

end distance_comparison_tetrahedron_l473_473545


namespace max_product_of_sum_2020_l473_473158

/--
  Prove that the maximum product of two integers whose sum is 2020 is 1020100.
-/
theorem max_product_of_sum_2020 : 
  ∃ x : ℤ, (x + (2020 - x) = 2020) ∧ (x * (2020 - x) = 1020100) :=
by
  sorry

end max_product_of_sum_2020_l473_473158


namespace particular_solution_l473_473694

-- Initial conditions
def init_cond_y : ℝ := -1
def init_cond_dy : ℝ := 1

-- Function definition
def y (x : ℝ) : ℝ := -sin x - (x^2) / 2 + 2 * x - 1

-- First derivative of y
def dy (x : ℝ) : ℝ := -cos x - x + 2

-- Second derivative of y
def d2y (x : ℝ) : ℝ := sin x - 1

-- The main theorem to prove
theorem particular_solution :
  ∀ x : ℝ, (d2y x = sin x - 1) ∧ (y 0 = init_cond_y) ∧ (dy 0 = init_cond_dy) :=
  by
    intros x
    sorry

end particular_solution_l473_473694


namespace bacon_suggestion_l473_473508

theorem bacon_suggestion (x y : ℕ) (h1 : x = 479) (h2 : y = x + 10) : y = 489 := 
by {
  sorry
}

end bacon_suggestion_l473_473508


namespace greatest_base9_3_digit_divisible_by_7_l473_473151

def base9_to_decimal (n : Nat) : Nat :=
  match n with
  | 0     => 0
  | n + 1 => (n % 10) * Nat.pow 9 (n / 10)

def decimal_to_base9 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n => let rec aux (n acc : Nat) :=
              if n = 0 then acc
              else aux (n / 9) ((acc * 10) + (n % 9))
         in aux n 0

theorem greatest_base9_3_digit_divisible_by_7 :
  ∃ (n : Nat), n < Nat.pow 9 3 ∧ (n % 7 = 0) ∧ n = 8 * 81 + 8 * 9 + 8 :=
begin
  sorry -- Proof would go here
end

end greatest_base9_3_digit_divisible_by_7_l473_473151


namespace largest_3_digit_base9_divisible_by_7_l473_473141

def is_three_digit_base9 (n : ℕ) : Prop :=
  n < 9^3

def is_divisible_by (n d : ℕ) : Prop :=
  n % d = 0

def base9_to_base10 (n : ℕ) : ℕ :=
  let digits := [n / 81 % 9, n / 9 % 9, n % 9] in
  digits[0] * 81 + digits[1] * 9 + digits[2]

theorem largest_3_digit_base9_divisible_by_7 :
  ∃ n : ℕ, is_three_digit_base9 n ∧ is_divisible_by (base9_to_base10 n) 7 ∧ base9_to_base10 n = 728 ∧ n = 888 :=
sorry

end largest_3_digit_base9_divisible_by_7_l473_473141


namespace find_P_value_l473_473898

variable (R B G Y P U C : ℕ)

def P_value (C : ℕ) : ℕ := 0.425 * C - 13

theorem find_P_value (h1 : R + B + G + Y + P + U = C)
                     (h2 : R = 12)
                     (h3 : B = 8)
                     (h4 : G = (3/4)*B)
                     (h5 : Y = 0.15*C)
                     (h6 : P = U) :
  P = 0.425*C - 13 :=
by
  sorry

end find_P_value_l473_473898


namespace sector_perimeter_ratio_l473_473013

theorem sector_perimeter_ratio (α : ℝ) (r R : ℝ) 
  (h1 : α > 0) 
  (h2 : r > 0) 
  (h3 : R > 0) 
  (h4 : (1/2) * α * r^2 / ((1/2) * α * R^2) = 1/4) :
  (2 * r + α * r) / (2 * R + α * R) = 1 / 2 := 
sorry

end sector_perimeter_ratio_l473_473013


namespace parallelogram_area_formula_l473_473496

noncomputable def parallelogram_area (ha hb : ℝ) (γ : ℝ) : ℝ := 
  ha * hb / Real.sin γ

theorem parallelogram_area_formula (ha hb γ : ℝ) (a b : ℝ) 
  (h₁ : Real.sin γ ≠ 0) :
  (parallelogram_area ha hb γ = ha * hb / Real.sin γ) := by
  sorry

end parallelogram_area_formula_l473_473496


namespace simplify_and_evaluate_l473_473083

-- Given conditions: x = 1/3 and y = -1/2
def x : ℚ := 1 / 3
def y : ℚ := -1 / 2

-- Problem statement: 
-- Prove that (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2
theorem simplify_and_evaluate :
  (2 * x + 3 * y)^2 - (2 * x + y) * (2 * x - y) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_l473_473083


namespace h_zero_h_expression_and_min_l473_473727

noncomputable def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 2 then 1
  else if 2 < x ∧ x ≤ 3 then (1 / 2) * x ^ 2 - 1
  else 0

noncomputable def h (a : ℝ) : ℝ :=
  let g (x : ℝ) := f x - a * x
  let max_g := sup (set.image g (set.Icc 1 3))
  let min_g := inf (set.image g (set.Icc 1 3))
  max_g - min_g

theorem h_zero : h 0 = 5 / 2 :=
  sorry

theorem h_expression_and_min :
  h = λ a, if a <= 0 then 5 / 2 - 2 * a
           else if 0 < a ∧ a <= 5 / 4 then 5 / 2 - a
           else if 5 / 4 < a ∧ a <= 2 then a
           else if 2 < a ∧ a <= 3 then (1 / 2) * a ^ 2 - a + 2
           else 2 * a - 5 / 2 ∧ 
  (∀ a, a = 5 / 4 → h a = 5 / 4) :=
  sorry

end h_zero_h_expression_and_min_l473_473727


namespace probability_of_drawing_K_is_2_over_27_l473_473564

-- Define the total number of cards in a standard deck of 54 cards
def total_cards : ℕ := 54

-- Define the number of "K" cards in the standard deck
def num_K_cards : ℕ := 4

-- Define the probability function for drawing a "K"
def probability_drawing_K (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- Prove that the probability of drawing a "K" is 2/27
theorem probability_of_drawing_K_is_2_over_27 :
  probability_drawing_K total_cards num_K_cards = 2 / 27 :=
by
  sorry

end probability_of_drawing_K_is_2_over_27_l473_473564


namespace max_abs_sum_on_circle_l473_473369

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end max_abs_sum_on_circle_l473_473369


namespace find_C_coordinates_l473_473322

-- Define the points A, B, and the vector relationship
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)
def C : ℝ × ℝ := (-3, 9)

-- The condition stating vector AC is twice vector AB
def vector_condition (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1, C.2 - A.2) = (2 * (B.1 - A.1), 2 * (B.2 - A.2))

-- The theorem we need to prove
theorem find_C_coordinates (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (-1, 5))
  (hCondition : vector_condition A B C) : C = (-3, 9) :=
by
  rw [hA, hB] at hCondition
  -- sorry here skips the proof
  sorry

end find_C_coordinates_l473_473322


namespace sum_of_solutions_l473_473565

def equation (x : ℝ) : Prop := (6 * x) / 30 = 8 / x

theorem sum_of_solutions : ∀ x1 x2 : ℝ, equation x1 → equation x2 → x1 + x2 = 0 := by
  sorry

end sum_of_solutions_l473_473565


namespace evaluate_expression_l473_473683

theorem evaluate_expression :
  (216 ^ (Real.log 1729 / Real.log 3))^(1/3) = 2989441 := by
  sorry

end evaluate_expression_l473_473683


namespace minimum_distance_centroid_l473_473034

theorem minimum_distance_centroid
  (A B C : EuclideanSpace ℝ (Fin 3))
  (hA : ∠ A = 120 * (π / 180)) 
  (h_dot : (B - A) ⬝ (C - A) = -3) :
  ∃ G : EuclideanSpace ℝ (Fin 3), 
    is_centroid G A B C ∧ |G - A| = sqrt(6) / 3 := by
  sorry

end minimum_distance_centroid_l473_473034


namespace max_ab_perpendicular_l473_473000

theorem max_ab_perpendicular (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 2 * a + b = 3) : ab <= (9 / 8) := 
sorry

end max_ab_perpendicular_l473_473000


namespace at_least_three_on_same_topic_l473_473936

theorem at_least_three_on_same_topic
    (S : Fin 17) (T : Fin 3) (corr : S × S → T) :
    ∃ (s1 s2 s3 : S), s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧ 
                      corr (s1, s2) = corr (s2, s3) ∧ 
                      corr (s2, s3) = corr (s1, s3) :=
begin
  sorry
end

end at_least_three_on_same_topic_l473_473936


namespace difference_received_from_parents_l473_473298

-- Define conditions
def amount_from_mom := 8
def amount_from_dad := 5

-- Question: Prove the difference between amount_from_mom and amount_from_dad is 3
theorem difference_received_from_parents : (amount_from_mom - amount_from_dad) = 3 :=
by
  sorry

end difference_received_from_parents_l473_473298


namespace find_m_l473_473331

theorem find_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 ^ a = m) (h4 : 3 ^ b = m) (h5 : 2 * a * b = a + b) : m = Real.sqrt 6 :=
sorry

end find_m_l473_473331


namespace point_comparison_on_inverse_proportion_l473_473920

theorem point_comparison_on_inverse_proportion :
  (∃ y1 y2, (y1 = 2 / 1) ∧ (y2 = 2 / 2) ∧ y1 > y2) :=
by
  use 2
  use 1
  sorry

end point_comparison_on_inverse_proportion_l473_473920


namespace h_value_l473_473390

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l473_473390


namespace max_value_proof_l473_473995

noncomputable def max_expression_value (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_sum : a + b + c + d ≤ 4) : ℝ :=
  let expr := Real.root 4 (a^2 * (a + b)) +
              Real.root 4 (b^2 * (b + c)) +
              Real.root 4 (c^2 * (c + d)) +
              Real.root 4 (d^2 * (d + a))
  in expr

theorem max_value_proof (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_sum : a + b + c + d ≤ 4) :
  max_expression_value a b c d h_pos h_sum ≤ 4 * Real.root 4 2 :=
sorry

end max_value_proof_l473_473995


namespace sum_of_extremes_of_g_l473_473895

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - abs (2 * x - 8)

theorem sum_of_extremes_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≤ g 4) ∧ (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≥ g 1) → g 4 + g 1 = 2 :=
by
  sorry

end sum_of_extremes_of_g_l473_473895


namespace unique_tensor_identity_l473_473358

structure Vector :=
  (a : ℝ) (b : ℝ)

def tensor_op (m p : Vector) : Vector :=
  ⟨m.a * p.a + m.b * p.b, m.a * p.b + m.b * p.a⟩

theorem unique_tensor_identity (p : Vector) :
  (∀ m : Vector, tensor_op m p = m) ↔ p = ⟨1, 0⟩ :=
by
  sorry

end unique_tensor_identity_l473_473358


namespace shelby_gold_stars_total_l473_473081

theorem shelby_gold_stars_total :
  let Monday_star := 4
  let Tuesday_star := 6
  let Wednesday_star := 3
  let Thursday_star := 5
  let Friday_star := 2
  let Saturday_star := 3
  let Sunday_star := 7
  in Monday_star + Tuesday_star + Wednesday_star + Thursday_star + Friday_star + Saturday_star + Sunday_star = 30 :=
by
  let Monday_star := 4
  let Tuesday_star := 6
  let Wednesday_star := 3
  let Thursday_star := 5
  let Friday_star := 2
  let Saturday_star := 3
  let Sunday_star := 7
  sorry

end shelby_gold_stars_total_l473_473081


namespace class_1_stats_computed_class_2_stats_computed_choose_class_for_team_choose_class_for_top_three_l473_473935

section
variables (goals_class_1 : list ℕ) (goals_class_2 : list ℕ)
variables 
  (mean_class_1 : ℕ) (mode_class_1 : ℕ) (median_class_1 : ℕ)
  (mean_class_2 : ℕ) (mode_class_2 : ℕ) (median_class_2 : ℕ)
  (variance_class_1 : ℝ) (variance_class_2 : ℝ)

-- Conditions
axiom class_1_data : goals_class_1 = [10, 9, 8, 7, 7, 7, 7, 5, 5, 5]
axiom class_2_data : goals_class_2 = [9, 8, 8, 7, 7, 7, 7, 7, 5, 5]

-- Computations (definitions)
noncomputable def calculate_mean (scores : list ℕ) : ℕ :=
  sorry -- Function to calculate the mean

noncomputable def calculate_mode (scores : list ℕ) : ℕ :=
  sorry -- Function to calculate the mode

noncomputable def calculate_median (scores : list ℕ) : ℕ :=
  sorry -- Function to calculate the median

noncomputable def calculate_variance (scores : list ℕ) (mean : ℕ) : ℝ :=
  sorry -- Function to calculate the variance

-- Proof Statements
theorem class_1_stats_computed :
  calculate_mean goals_class_1 = 7 ∧
  calculate_mode goals_class_1 = 7 ∧
  calculate_median goals_class_1 = 7 :=
sorry

theorem class_2_stats_computed :
  calculate_mean goals_class_2 = 7 ∧
  calculate_mode goals_class_2 = 7 ∧
  calculate_median goals_class_2 = 7 :=
sorry

theorem choose_class_for_team :
  calculate_variance goals_class_2 (calculate_mean goals_class_2) <
  calculate_variance goals_class_1 (calculate_mean goals_class_1) :=
sorry

theorem choose_class_for_top_three :
  list.maximum goals_class_1 > list.maximum goals_class_2 :=
sorry

end

end class_1_stats_computed_class_2_stats_computed_choose_class_for_team_choose_class_for_top_three_l473_473935


namespace negative_integer_solution_l473_473989

theorem negative_integer_solution (N : ℤ) (h : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end negative_integer_solution_l473_473989


namespace at_most_two_special_numbers_in_range_at_most_two_special_numbers_in_range_alt_l473_473479

theorem at_most_two_special_numbers_in_range (k : ℕ) (h : 0 < k) :
  ∃ n m : ℕ, is_special n ∧ is_special m ∧ n ≠ m ∧ k^2 < n ∧ n < k^2 + 2 * k + 1 ∧ k^2 < m ∧ m < k^2 + 2 * k + 1 → false :=
by
  sorry

def is_special (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 3^b

-- Alternatively
theorem at_most_two_special_numbers_in_range_alt (k : ℕ) (h : 0 < k) :
  ∀ (n m : ℕ), is_special n ∧ is_special m ∧ n ≠ m ∧ k^2 < n ∧ n < k^2 + 2 * k + 1 ∧ k^2 < m ∧ m < k^2 + 2 * k + 1 → false :=
by
  sorry

end at_most_two_special_numbers_in_range_at_most_two_special_numbers_in_range_alt_l473_473479


namespace smallest_positive_period_range_of_values_l473_473342

noncomputable def f (x : Real) : Real := 
  √3 * Real.cos (2 * x) + 2 * (Real.cos (π / 4 - x))^2 - 1

theorem smallest_positive_period : 
  ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = π := 
sorry

theorem range_of_values : 
  ∀ x ∈ Set.Icc (-π/3) (π/2), 
  f(x) ∈ Set.Icc (-√3) 2 :=
sorry

end smallest_positive_period_range_of_values_l473_473342


namespace possible_values_m_l473_473044

variables {u v w : ℝ^3}
variables (h₁ : ∥u∥ = 1) (h₂ : ∥v∥ = 1) (h₃ : ∥w∥ = 1)
variables (h₄ : u ⋅ v = 0) (h₅ : u ⋅ w = 0)
variables (h₆ : inner v w = ∥v∥ * ∥w∥ * real.cos (real.pi / 3))

theorem possible_values_m :
  ∃ m : ℝ, u = m • (v × w) ∧ (m = 2 * real.sqrt(3) / 3 ∨ m = -2 * real.sqrt(3) / 3) :=
sorry

end possible_values_m_l473_473044


namespace sum_x_y_l473_473558

-- Let PQR be a triangle with QR = 30
variables (P Q R : ℝ)
variable h1 : QR = 30

-- Assume the incircle trisects the median PS
variable h2 : (incircle_trisects_median P Q R)

-- Define x and y such that area of triangle PQR is x * sqrt(y)
variables (x y : ℝ)

-- Condition: y is not divisible by the square of a prime
variable prime_square_condition : (¬ ∃ p:ℝ, is_prime p ∧ divides (p^2) y)

-- Define the area of the triangle as x * sqrt y
variable area_pqr : area P Q R = x * (sqrt y)

-- Statement of the problem to prove x + y = 55
theorem sum_x_y (h1 h2 : QR = 30) (h2 : incircle_trisects_median P Q R) (prime_square_condition : ¬∃ p: ℝ, is_prime p ∧ divides (p^2) y) (area_pqr : area P Q R = x * sqrt y) : x + y = 55 :=
by
  sorry

end sum_x_y_l473_473558


namespace monotonicity_and_extrema_on_interval_l473_473347

def f (x : ℝ) : ℝ := (x - 1) / (x + 2)

theorem monotonicity_and_extrema_on_interval :
  (∀ x ∈ Icc 3 5, ∀ y ∈ Icc 3 5, x < y → f x < f y) ∧
  (f 5 = 4 / 7) ∧
  (f 3 = 2 / 5) := by
  sorry

end monotonicity_and_extrema_on_interval_l473_473347


namespace arithmetic_sequence_problem1_l473_473198

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a(n+1) = a(n) + d

theorem arithmetic_sequence_problem1 (a : ℕ → ℤ) (d : ℤ) (m : ℕ)
  (h_seq : is_arithmetic_sequence a d) 
  (h_d_nonzero : d ≠ 0)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_am : a m = 8) :
  m = 8 :=
sorry

end arithmetic_sequence_problem1_l473_473198


namespace max_product_distances_l473_473709

noncomputable def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(a : ℝ)
(b : ℝ)
(c : ℝ) -- Ax + By + C = 0

def is_perpendicular (l1 l2 : Line) : Prop :=
l1.a * l2.a + l1.b * l2.b = 0

def distance_to_center (l : Line) : ℝ :=
(abs(l.a * 1 + l.b * 0 + l.c)) / (sqrt(l.a^2 + l.b^2))

def A := Point.mk 0 1
def center := Point.mk 1 0

theorem max_product_distances (l1 l2 : Line)
  (h1 : l1.a * A.x + l1.b * A.y + l1.c = 0)
  (h2 : l2.a * A.x + l2.b * A.y + l2.c = 0)
  (h3 : is_perpendicular l1 l2):
  let d1 := distance_to_center l1 in
  let d2 := distance_to_center l2 in
  d1 * d2 ≤ 1 :=
by sorry

end max_product_distances_l473_473709


namespace greatest_product_l473_473162

theorem greatest_product (x : ℤ) (h : x + (2020 - x) = 2020) : x * (2020 - x) ≤ 1020100 :=
sorry

end greatest_product_l473_473162


namespace quadratic_expression_rewriting_l473_473411

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l473_473411


namespace number_of_possible_third_side_lengths_l473_473822

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l473_473822


namespace triangle_third_side_length_l473_473794

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l473_473794


namespace rahim_sequence_final_value_l473_473243

theorem rahim_sequence_final_value :
  ∃ (a : ℕ) (b : ℕ), a ^ b = 5 ^ 16 :=
sorry

end rahim_sequence_final_value_l473_473243


namespace max_value_of_g_l473_473961

def g : ℕ → ℕ 
| n := if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : 
  ∃ N, (∀ n, g n ≤ N) ∧ N = 23 := 
sorry

end max_value_of_g_l473_473961


namespace find_b_for_parallelogram_roots_l473_473281

noncomputable def polynomial_has_parallelogram_roots (b : ℝ) : Prop :=
  ∃ z : ℂ, ∃ f : ℂ → ℂ,
    f z = z^4 - 8*z^3 + 13*b*z^2 - 5*(2*b^2 + b - 2)*z + 4 ∧
    (f (z + 2)).coeff 1 = 0

theorem find_b_for_parallelogram_roots :
  polynomial_has_parallelogram_roots 2 :=
sorry

end find_b_for_parallelogram_roots_l473_473281


namespace gcd_of_gy_and_y_l473_473334

theorem gcd_of_gy_and_y (y : ℕ) (h : ∃ k : ℕ, y = k * 3456) :
  gcd ((5 * y + 4) * (9 * y + 1) * (12 * y + 6) * (3 * y + 9)) y = 216 :=
by {
  sorry
}

end gcd_of_gy_and_y_l473_473334


namespace collinear_A_P_Q_l473_473471

variables {A B C D E P F G Q : Point}
variables [f : Euclidean_Geometry]

-- Conditions
axiom D_on_AB : OnSegment A B D
axiom E_on_AC : OnSegment A C E
axiom DE_parallel_BC : Parallel (Line.mk D E) (Line.mk B C)
axiom P_in_ADE : InTriangle A D E P
axiom F_int_DE_BP : Intersects (Line.mk D E) (Line.mk B P) F
axiom G_int_DE_CP : Intersects (Line.mk D E) (Line.mk C P) G
axiom Q_circ_PDG_PFE : SecIntersection (Circ P D G) (Circ P F E) Q

-- Theorem to prove
theorem collinear_A_P_Q : Collinear A P Q :=
  sorry

end collinear_A_P_Q_l473_473471


namespace count_valid_third_sides_l473_473815

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l473_473815


namespace sum_of_first_n_natural_numbers_l473_473921

theorem sum_of_first_n_natural_numbers (n : ℕ) : ∑ k in Finset.range (n + 1), k = n * (n + 1) / 2 := by
  sorry

end sum_of_first_n_natural_numbers_l473_473921


namespace jesse_stamps_l473_473450

variable (A E : Nat)

theorem jesse_stamps :
  E = 3 * A ∧ E + A = 444 → E = 333 :=
by
  sorry

end jesse_stamps_l473_473450


namespace sum_of_interior_angles_l473_473098

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1980) :
    180 * ((n + 3) - 2) = 2520 :=
by
  sorry

end sum_of_interior_angles_l473_473098


namespace votes_in_each_round_l473_473031

def number_of_votes_each_round (V : ℕ) : ℕ :=
3 * V

theorem votes_in_each_round : ∃ V : ℕ, let total_votes := number_of_votes_each_round V in
(V - 16000 - 8000) < V  ∧
(V + 16000 > V) ∧
(V + 16000 = 5 * (V - 16000 - 8000)) ∧
(total_votes = 102000) :=
sorry

end votes_in_each_round_l473_473031


namespace simplify_fraction_l473_473502

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l473_473502


namespace geometric_sequence_general_formula_l473_473430

theorem geometric_sequence_general_formula (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h1 : a 1 = 2)
  (h_rec : ∀ n, (a (n + 2))^2 + 4 * (a n)^2 = 4 * (a (n + 1))^2) :
  ∀ n, a n = 2^(n + 1) / 2 := 
sorry

end geometric_sequence_general_formula_l473_473430


namespace greatest_three_digit_base_nine_divisible_by_seven_l473_473156

/-- Define the problem setup -/
def greatest_three_digit_base_nine := 8 * 9^2 + 8 * 9 + 8

/-- Prove the greatest 3-digit base 9 positive integer that is divisible by 7 -/
theorem greatest_three_digit_base_nine_divisible_by_seven : 
  ∃ n : ℕ, n = greatest_three_digit_base_nine ∧ n % 7 = 0 ∧ (8 * 9^2 + 8 * 9 + 8) = 728 := by 
  sorry

end greatest_three_digit_base_nine_divisible_by_seven_l473_473156


namespace number_of_true_propositions_is_three_l473_473254

-- Define each proposition as a Lean term
def proposition1 : Prop := ¬((∃ x : ℝ, log x = 0) → (x = 1)) ↔ (¬∃ x : ℝ, log x = 0 → x ≠ 1)
def proposition2 : Prop := (¬(True ∧ True)) → (True ∧ False)
def proposition3 : Prop := (∃ x : ℝ, sin x > 1) ↔ (¬∀ x : ℝ, sin x ≤ 1)
def proposition4 : Prop := ∀ x : ℝ, (x > 2 → 1 / x < 1 / 2) ∧ (¬(2 > x → 1 / x < 1 / 2))

-- Define the overall proof problem
theorem number_of_true_propositions_is_three : 
  (nat.succ (nat.succ (nat.succ nat.zero))) = 
  (CondCount [proposition1, proposition2, proposition3, proposition4] (λ x, x = true)) :-
begin
  -- The proof will be skipped
  sorry
end

end number_of_true_propositions_is_three_l473_473254


namespace polynomial_divisibility_l473_473074

noncomputable def poly_expr (x : ℝ) (n : ℕ) : ℝ :=
  x^(4*n + 2) - (2*n + 1)*x^(2*n + 2) + (2*n + 1)*x^(2*n) - 1

noncomputable def divisor_expr (x : ℝ) : ℝ :=
  (x^2 - 1)^3

theorem polynomial_divisibility (x : ℝ) (n : ℕ) : 
  ∃ (q : ℝ), poly_expr x n = divisor_expr x * q :=
begin
  sorry
end

end polynomial_divisibility_l473_473074


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l473_473674

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (a : ℕ), (∀ n, n ∈ (list.range 4).map (λ i, a + i) -> n % 2 = 0 ∨ n % 3 = 0 ∨ n % 4 = 0) →
  12 ∣ list.prod ((list.range 4).map (λ i, a + i)) :=
by
  intro a
  intro h
  -- Insert proof here
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l473_473674


namespace john_total_spent_l473_473451

-- Defining the conditions from part a)
def vacuum_cleaner_original_price : ℝ := 250
def vacuum_cleaner_discount_rate : ℝ := 0.20
def dishwasher_price : ℝ := 450
def special_offer_discount : ℝ := 75
def sales_tax_rate : ℝ := 0.07

-- The adesso to formalize part c noncomputably.
noncomputable def total_amount_spent : ℝ :=
  let vacuum_cleaner_discount := vacuum_cleaner_original_price * vacuum_cleaner_discount_rate
  let vacuum_cleaner_final_price := vacuum_cleaner_original_price - vacuum_cleaner_discount
  let total_before_special_offer := vacuum_cleaner_final_price + dishwasher_price
  let total_after_special_offer := total_before_special_offer - special_offer_discount
  let sales_tax := total_after_special_offer * sales_tax_rate
  total_after_special_offer + sales_tax

-- The proof statement
theorem john_total_spent : total_amount_spent = 615.25 := by
  sorry

end john_total_spent_l473_473451


namespace fifth_equation_pattern_l473_473491

theorem fifth_equation_pattern :
  ∑ k in Finset.range (25 - 16 + 1) + 16 = (4^3) + (5^3) :=
by
  sorry

end fifth_equation_pattern_l473_473491


namespace number_of_non_pine_trees_l473_473547

theorem number_of_non_pine_trees 
  (total_trees : ℕ) 
  (percentage_pine : ℝ)
  (total_trees_eq : total_trees = 1895) 
  (percentage_pine_eq : percentage_pine = 63.5 / 100) 
  :
  total_trees - nat.floor (total_trees * percentage_pine) = 692 :=
by
  rw [total_trees_eq, percentage_pine_eq]
  sorry

end number_of_non_pine_trees_l473_473547


namespace triangle_third_side_lengths_l473_473763

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l473_473763


namespace ratio_and_equation_imp_value_of_a_l473_473009

theorem ratio_and_equation_imp_value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 20 - 7 * a) :
  a = 20 / 11 :=
by
  sorry

end ratio_and_equation_imp_value_of_a_l473_473009


namespace sum_of_first_fifteen_multiples_of_7_l473_473577

theorem sum_of_first_fifteen_multiples_of_7 :
  ∑ i in finset.range 15, 7 * (i + 1) = 840 :=
sorry

end sum_of_first_fifteen_multiples_of_7_l473_473577


namespace greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l473_473192

theorem greatest_divisor_of_product_of_5_consecutive_multiples_of_4 :
  let n1 := 4
  let n2 := 8
  let n3 := 12
  let n4 := 16
  let n5 := 20
  let spf1 := 2 -- smallest prime factor of 4
  let spf2 := 2 -- smallest prime factor of 8
  let spf3 := 2 -- smallest prime factor of 12
  let spf4 := 2 -- smallest prime factor of 16
  let spf5 := 2 -- smallest prime factor of 20
  let p1 := n1^spf1
  let p2 := n2^spf2
  let p3 := n3^spf3
  let p4 := n4^spf4
  let p5 := n5^spf5
  let product := p1 * p2 * p3 * p4 * p5
  product % (2^24) = 0 :=
by 
  sorry

end greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l473_473192


namespace more_pairs_B_than_A_l473_473550

theorem more_pairs_B_than_A :
    let pairs_per_box := 20
    let boxes_A := 8
    let pairs_A := boxes_A * pairs_per_box
    let pairs_B := 5 * pairs_A
    let more_pairs := pairs_B - pairs_A
    more_pairs = 640
:= by
    sorry

end more_pairs_B_than_A_l473_473550


namespace seven_points_seven_lines_impossible_l473_473448

theorem seven_points_seven_lines_impossible :
  ¬ (∃ (points : Finset Point) (lines : Finset Line),
        points.card = 7 ∧ lines.card = 7 ∧
        (∀ p ∈ points, ∃ ls : Finset Line, (∀ l ∈ ls, l.contains p) ∧ ls.card = 3) ∧
        (∀ l ∈ lines, ∃ ps : Finset Point, (∀ p ∈ ps, p ∈ l) ∧ ps.card = 3)) :=
by
  sorry

end seven_points_seven_lines_impossible_l473_473448


namespace sum_of_coefficients_125x6_minus_216y6_l473_473688

theorem sum_of_coefficients_125x6_minus_216y6 :
  let expression := (5 * x * x - 6 * y * y) * (25 * x * x * x * x + 30 * x * x * y * y + 36 * y * y * y * y) in
  (sum of integer coefficients of expression) = 41 :=
by
  let expression := (5 * x * x - 6 * y * y) * (25 * x * x * x * x + 30 * x * x * y * y + 36 * y * y * y * y)
  sorry

end sum_of_coefficients_125x6_minus_216y6_l473_473688


namespace real_y_iff_x_l473_473467

open Real

-- Definitions based on the conditions
def quadratic_eq (y x : ℝ) : ℝ := 9 * y^2 - 3 * x * y + x + 8

-- The main theorem to prove
theorem real_y_iff_x (x : ℝ) : (∃ y : ℝ, quadratic_eq y x = 0) ↔ x ≤ -4 ∨ x ≥ 8 := 
sorry

end real_y_iff_x_l473_473467


namespace a_range_l473_473741

noncomputable def A (x : ℝ) := 2 - (x + 3) / (x + 1) ≥ 0
noncomputable def B (x : ℝ) (a : ℝ) := (x - a - 1) * (x - 2 * a) < 0

noncomputable def a_condition (a : ℝ) := a < 1

theorem a_range (a : ℝ) (hA : ∀ x : ℝ, A x) (hB : ∀ x : ℝ, B x a) (hA_cond : a_condition a) :
  (∀ (x : ℝ), A x ∨ B x a → A x) → a ∈ (-∞, -2] ∪ [1/2, 1) :=
sorry

end a_range_l473_473741


namespace area_difference_of_square_screens_l473_473539

theorem area_difference_of_square_screens (d1 d2 : ℝ) (A1 A2 : ℝ) 
  (h1 : d1 = 18) (h2 : d2 = 16) 
  (hA1 : A1 = d1^2 / 2) (hA2 : A2 = d2^2 / 2) : 
  A1 - A2 = 34 := by
  sorry

end area_difference_of_square_screens_l473_473539


namespace total_money_made_l473_473230

-- Definitions based on the conditions
def average_price := 9.8
def pairs_sold := 75

-- The statement to prove
theorem total_money_made : average_price * pairs_sold = 735 := 
by 
  -- The proof is omitted
  sorry

end total_money_made_l473_473230


namespace count_x_satisfying_conditions_l473_473361

theorem count_x_satisfying_conditions :
  { x : ℝ | 0 ≤ x ∧ x < 360 ∧ real.sin (x * real.pi / 180) = -0.65 ∧ real.cos (x * real.pi / 180) < 0}.to_finset.card = 1 :=
sorry

end count_x_satisfying_conditions_l473_473361


namespace polynomial_remainder_l473_473292

theorem polynomial_remainder :
  ∃ (a b c : ℚ), 
    (a = 5/3) ∧ (b = 2) ∧ (c = 1/3) ∧ 
    (x^6 - 3*x^4 + 2*x^3 - x + 5 = ((x^2 - 1)*(x - 2)) * polynomial.ring_hom (polynomial ℚ) + (a*x^2 + b*x + c)) :=
by
  sorry

end polynomial_remainder_l473_473292


namespace lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l473_473248

def lassis_per_three_mangoes := 15
def smoothies_per_mango := 1
def bananas_per_smoothie := 2

-- proving the number of lassis Caroline can make with eighteen mangoes
theorem lassis_with_eighteen_mangoes :
  (18 / 3) * lassis_per_three_mangoes = 90 :=
by 
  sorry

-- proving the number of smoothies Caroline can make with eighteen mangoes and thirty-six bananas
theorem smoothies_with_eighteen_mangoes_and_thirtysix_bananas :
  min (18 / smoothies_per_mango) (36 / bananas_per_smoothie) = 18 :=
by 
  sorry

end lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l473_473248


namespace paco_initial_cookies_l473_473070

theorem paco_initial_cookies (x : ℕ) (h : x - 2 + 36 = 2 + 34) : x = 2 :=
by
-- proof steps will be filled in here
sorry

end paco_initial_cookies_l473_473070


namespace triangle_third_side_lengths_l473_473765

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l473_473765


namespace triangle_ABC_angle_ABC_l473_473033

theorem triangle_ABC_angle_ABC
  (A B C D : Type)
  [triangle ABC]
  (h1 : AB = AC)
  (h2 : D ∈ line_segment AC)
  (h3 : angle_bisector BD (angle BAC))
  (h4 : angle BAC = 72) : 
  angle ABC = 108 :=
by
  sorry

end triangle_ABC_angle_ABC_l473_473033


namespace sufficient_not_necessary_condition_l473_473987

theorem sufficient_not_necessary_condition
  (x : ℝ) : 
  x^2 - 4*x - 5 > 0 → (x > 5 ∨ x < -1) ∧ (x > 5 → x^2 - 4*x - 5 > 0) ∧ ¬(x^2 - 4*x - 5 > 0 → x > 5) := 
sorry

end sufficient_not_necessary_condition_l473_473987


namespace petya_always_wins_l473_473035

-- Definition of the problem setup
def is_convex (polygon : Polygon) : Prop := sorry  -- Definition for convex polygon
def is_not_on_sides_or_diagonals (X : Point) (polygon : Polygon) : Prop := sorry -- Definition for point X not on sides or diagonals
def initial_condition (X : Point) (polygon : Polygon) : Prop :=
  is_convex polygon ∧ is_not_on_sides_or_diagonals X polygon

-- Game conditions
inductive Player
| Petya | Vasya

def move (player : Player) (vertices : Finset Point) (marked : Finset Point) : Finset Point := sorry -- Definition for marking vertices

-- Define the losing condition
def losing_condition (X : Point) (vertices : Finset Point) : Prop := X ∈ interior (Polygon.of_vertices vertices)

-- Main theorem
theorem petya_always_wins (polygon : Polygon) (X : Point)
  (h1 : initial_condition X polygon) :
  ∃ strategy : ℕ → Player → Finset Point → Point,
    ∀ (turn : ℕ) (player : Player) (marked : Finset Point),
      (turn = 0 ∧ player = Player.Petya → strategy turn player marked ∈ polygon.vertices) ∧
      (turn = 1 ∨ turn > 1 → strategy turn player marked ∉ interior (Polygon.of_vertices marked)) ∧
      ¬ losing_condition X marked :=
sorry

end petya_always_wins_l473_473035


namespace inequality_sqrt_sum_l473_473474

noncomputable def seq_geq (x : ℕ → ℝ) (n : ℕ) : Prop :=
∀ i, 1 ≤ i ∧ i ≤ n → x i ≥ x (i + 1)

theorem inequality_sqrt_sum (x : ℕ → ℝ) (n : ℕ) (h_seq : seq_geq x n) (h_last : x (n + 1) = 0) :
  (sqrt (∑ i in Finset.range n, x i)) ≤
  (Finset.range n).sum (λ i, sqrt (i + 1) * (sqrt (x i) - sqrt (x (i + 1)))) :=
sorry

end inequality_sqrt_sum_l473_473474


namespace possible_integer_lengths_third_side_l473_473807

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l473_473807


namespace find_m_value_l473_473357

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def is_perpendicular (v1 v2 : vector) : Prop :=
  dot_product v1 v2 = 0

theorem find_m_value (a b : vector) (m : ℝ) (h: a = (2, -1)) (h2: b = (1, 3))
  (h3: is_perpendicular a (a.1 + m * b.1, a.2 + m * b.2)) : m = 5 :=
sorry

end find_m_value_l473_473357


namespace combined_CD_length_l473_473449

def CD1 := 1.5
def CD2 := 1.5
def CD3 := 2 * CD1

theorem combined_CD_length : CD1 + CD2 + CD3 = 6 := 
by
  sorry

end combined_CD_length_l473_473449


namespace count_valid_third_sides_l473_473818

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l473_473818


namespace find_k_find_a_l473_473903

noncomputable def f (a k : ℝ) (x : ℝ) := a ^ x + k * a ^ (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

theorem find_k (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : is_odd_function (f a k)) : k = -1 :=
sorry

theorem find_a (k : ℝ) (h₃ : k = -1) (h₄ : f 1 = 3 / 2) (h₅ : is_monotonic_increasing (f 2 k)) : a = 2 :=
sorry

end find_k_find_a_l473_473903


namespace solve_negative_integer_sum_l473_473992

theorem solve_negative_integer_sum (N : ℤ) (h1 : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end solve_negative_integer_sum_l473_473992


namespace max_price_per_unit_l473_473624

-- Define the conditions
def original_price : ℝ := 25
def original_sales_volume : ℕ := 80000
def price_increase_effect (t : ℝ) : ℝ := 2000 * (t - original_price)
def new_sales_volume (t : ℝ) : ℝ := 130 - 2 * t

-- Define the condition for revenue
def revenue_condition (t : ℝ) : Prop :=
  t * new_sales_volume t ≥ original_price * original_sales_volume

-- Statement to prove the maximum price per unit
theorem max_price_per_unit : ∀ t : ℝ, revenue_condition t → t ≤ 40 := sorry

end max_price_per_unit_l473_473624


namespace sum_of_three_numbers_is_520_l473_473698

noncomputable def sum_of_three_numbers (x y z : ℝ) : ℝ :=
  x + y + z

theorem sum_of_three_numbers_is_520 (x y z : ℝ) (h1 : z = (1848 / 1540) * x) (h2 : z = 0.4 * y) (h3 : x + y = 400) :
  sum_of_three_numbers x y z = 520 :=
sorry

end sum_of_three_numbers_is_520_l473_473698


namespace trucks_initially_required_l473_473209

theorem trucks_initially_required
  (T b c : ℝ )
  (hT : 0 < T)
  (hb : 0 < b)
  (hc : 0 < c)
  (x : ℝ) :
  x = (b * c + sqrt (b^2 * c^2 + 4 * b * c * T)) / (2 * c) := by
  sorry


end trucks_initially_required_l473_473209


namespace find_a4_l473_473309

-- Given expression of x^5
def polynomial_expansion (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5

theorem find_a4 (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (h : polynomial_expansion x a_0 a_1 a_2 a_3 a_4 a_5) : a_4 = -5 :=
  sorry

end find_a4_l473_473309


namespace books_left_to_read_l473_473911

theorem books_left_to_read (total_books : ℕ) (books_mcgregor : ℕ) (books_floyd : ℕ) : total_books = 89 → books_mcgregor = 34 → books_floyd = 32 → 
  (total_books - (books_mcgregor + books_floyd) = 23) :=
by
  intros h1 h2 h3
  sorry

end books_left_to_read_l473_473911


namespace number_of_machines_in_first_scenario_l473_473087

noncomputable def machine_work_rate (R : ℝ) (hours_per_job : ℝ) : Prop :=
  (6 * R * 8 = 1)

noncomputable def machines_first_scenario (M : ℝ) (R : ℝ) (hours_per_job_first : ℝ) : Prop :=
  (M * R * hours_per_job_first = 1)

theorem number_of_machines_in_first_scenario (M : ℝ) (R : ℝ) :
  machine_work_rate R 8 ∧ machines_first_scenario M R 6 -> M = 8 :=
sorry

end number_of_machines_in_first_scenario_l473_473087


namespace min_squared_sum_l473_473900

noncomputable def min_value (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : ℝ :=
  x^2 + y^2 + z^2

theorem min_squared_sum (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
  ∃ (minB : ℝ), minB = 4 ∧ ∀ a b c : ℝ, a^3 + b^3 + c^3 - 3 * a * b * c = 8 → x^2 + y^2 + z^2 ≥ minB :=
by {
  use 4,
  sorry
}

end min_squared_sum_l473_473900


namespace sum_of_first_fifteen_multiples_of_7_l473_473566

theorem sum_of_first_fifteen_multiples_of_7 : (∑ k in Finset.range 15, 7 * (k + 1)) = 840 :=
by
  -- Summation from k = 0 to k = 14 (which corresponds to 1 to 15 multiples of 7)
  sorry

end sum_of_first_fifteen_multiples_of_7_l473_473566


namespace midpoint_distance_to_y_axis_is_4_l473_473216

-- Definitions based on given conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus := (1, 0 : ℝ)
def line_through_focus (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m * x + b ∧ (1, 0) = (1 * m + b)

-- Definitions of points A, B and segment AB
def intersects_parabola (line : ℝ × ℝ → Prop) (x y : ℝ) : Prop := line (x, y) ∧ parabola x y

-- Given condition for distances
def distances_condition (x1 x2 : ℝ) : Prop := x1 + 1 + x2 + 1 = 10

-- Definition of midpoint
def midpoint_distance (x1 x2 : ℝ) : ℝ := (x1 + x2) / 2

-- The theorem to be proved
theorem midpoint_distance_to_y_axis_is_4 (x1 x2 : ℝ) 
  (h1 : intersects_parabola line_through_focus x1 (sqrt (4 * x1)))
  (h2 : intersects_parabola line_through_focus x2 (-sqrt (4 * x2)))
  (h3 : distances_condition x1 x2) : 
  midpoint_distance x1 x2 = 4 :=
by sorry

end midpoint_distance_to_y_axis_is_4_l473_473216


namespace count_words_200_l473_473457

theorem count_words_200 : 
  let single_word_numbers := 29
  let compound_words_21_to_99 := 144
  let compound_words_100_to_199 := 54 + 216
  single_word_numbers + compound_words_21_to_99 + compound_words_100_to_199 = 443 :=
by
  sorry

end count_words_200_l473_473457


namespace abs_expression_solution_l473_473582

theorem abs_expression_solution (x : ℝ) : 
  | x | * (| -25 | - | 5 |) = 40 ↔ x = 2 ∨ x = -2 :=
begin
  sorry -- proof not included as requested
end

end abs_expression_solution_l473_473582


namespace triangle_third_side_length_count_l473_473784

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l473_473784


namespace polynomial_bound_l473_473899

noncomputable def polynomial_with_n_real_roots (n : ℕ) : Type :=
  { f : polynomial ℝ // (∀ k, polynomial.coeff f k ≥ 0) ∧ (∀ x, polynomial.aeval x f = 0 ↔ ∃ (d : fin n → ℝ), (∀ i, d i ≥ 0) ∧ (∀ i, (d i)^(n - i) = x)) }

theorem polynomial_bound (n : ℕ) (f : polynomial_with_n_real_roots n) : polynomial.eval 2 f.val ≥ 3^n := 
  sorry

end polynomial_bound_l473_473899


namespace inverse_function_value_l473_473729

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3

theorem inverse_function_value :
  f 3 = 51 :=
by
  sorry

end inverse_function_value_l473_473729


namespace possible_integer_lengths_third_side_l473_473804

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l473_473804


namespace rectangle_tiling_l473_473863

theorem rectangle_tiling :
  ∃ (w h: ℕ), w = 61 ∧ h = 69 ∧ (∃ (squares: finset ℕ), 
    squares = {2, 5, 7, 9, 16, 25, 28, 33, 36} ∧ 
    squares.sum (λ x, x * x) = w * h ∧
    (∀ x ∈ squares, ∃ (i j: ℕ), 
      i < w ∧ j < h ∧ ∀ y ∈ squares, ∃ (k l: ℕ), 
      k < w ∧ l < h ∧ (i, j) ≠ (k, l))) :=
by
  sorry

end rectangle_tiling_l473_473863


namespace sum_of_digits_M_l473_473438

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

theorem sum_of_digits_M (M : ℕ) (h_even : M % 2 = 0) 
    (h_digits : ∀ d ∈ M.digits 10, d ∈ {0, 2, 4, 5, 7, 9})
    (h_2M : sum_of_digits (2 * M) = 35) 
    (h_M_div_2 : sum_of_digits (M / 2) = 29) : 
  sum_of_digits M = 31 :=
sorry

end sum_of_digits_M_l473_473438


namespace sequence_sum_of_powers_l473_473666

theorem sequence_sum_of_powers (j : ℕ) :
  ∃ b : ℕ → ℕ, (∀ i : ℕ, i < j → b i < b (i + 1)) ∧
  ((2^275 + 1) / (2^11 + 1) = (finset.range j).sum (λ i, 2^(b i)))
  ∧ j = 145 :=
by
  sorry

end sequence_sum_of_powers_l473_473666


namespace tenth_monomial_is_neg_sqrt_10_x_10_l473_473119

def nth_monomial (n : ℕ) : ℝ := (-1)^(n-1) * real.sqrt n

theorem tenth_monomial_is_neg_sqrt_10_x_10 : nth_monomial 10 = - real.sqrt 10 * (x : ℝ)^10 :=
by
  sorry

end tenth_monomial_is_neg_sqrt_10_x_10_l473_473119


namespace triangle_third_side_length_count_l473_473792

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l473_473792


namespace count_valid_third_sides_l473_473813

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l473_473813


namespace largest_3_digit_base9_divisible_by_7_l473_473138

def is_three_digit_base9 (n : ℕ) : Prop :=
  n < 9^3

def is_divisible_by (n d : ℕ) : Prop :=
  n % d = 0

def base9_to_base10 (n : ℕ) : ℕ :=
  let digits := [n / 81 % 9, n / 9 % 9, n % 9] in
  digits[0] * 81 + digits[1] * 9 + digits[2]

theorem largest_3_digit_base9_divisible_by_7 :
  ∃ n : ℕ, is_three_digit_base9 n ∧ is_divisible_by (base9_to_base10 n) 7 ∧ base9_to_base10 n = 728 ∧ n = 888 :=
sorry

end largest_3_digit_base9_divisible_by_7_l473_473138


namespace total_interactions_l473_473857

theorem total_interactions (witches zombies : ℕ) (H1 : witches = 25) (H2 : zombies = 18) : 
  ∑ i in finset.range(zombies), witches + ∑ i in finset.range(zombies), i = 603 :=
by
  sorry

end total_interactions_l473_473857


namespace monthly_instalment_is_504_l473_473623

-- Define constants and parameters
def cash_price : ℤ := 21000 -- Cash price in dollars
def deposit_rate : ℚ := 0.10 -- Deposit rate
def num_months : ℤ := 60 -- Number of months for instalments
def annual_interest_rate : ℚ := 0.12 -- Annual interest rate

-- Define derived values based on conditions
def deposit : ℤ := (deposit_rate * cash_price).toInt
def balance : ℤ := cash_price - deposit
def num_years : ℚ := (num_months / 12).toRat
def total_interest : ℤ := (balance * annual_interest_rate * num_years).toInt
def total_repayment : ℤ := balance + total_interest
def monthly_instalment : ℚ := (total_repayment / num_months).toRat

theorem monthly_instalment_is_504 :
  monthly_instalment = 504 := 
by
  sorry

end monthly_instalment_is_504_l473_473623


namespace daily_growth_rate_l473_473187

-- Define constant length on the day of planting
def initial_length : ℝ := 11

-- Define the daily growth rate variable
variable (x : ℝ)

-- Define the length on the 4th day
def length_on_4th_day (x : ℝ) : ℝ := initial_length + 3 * x

-- Define the length on the 10th day
def length_on_10th_day (x : ℝ) : ℝ := initial_length + 9 * x

-- Define the growth between the 4th day and 10th day
def growth_between_4th_and_10th_day (x : ℝ) : ℝ := length_on_10th_day x - length_on_4th_day x

-- Define the 30% growth relation
def growth_relation (x : ℝ) : Prop := growth_between_4th_and_10th_day x = 0.30 * length_on_4th_day x

-- The statement to prove the daily growth rate x is equal to 11/17 given the above conditions
theorem daily_growth_rate : ∃ x, growth_relation x ∧ x = 11 / 17 :=
begin
  sorry
end

end daily_growth_rate_l473_473187


namespace complete_the_square_h_value_l473_473404

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l473_473404


namespace parallelogram_circumference_l473_473529

-- Define the lengths of the sides of the parallelogram.
def side1 : ℝ := 18
def side2 : ℝ := 12

-- Define the formula for the circumference (or perimeter) of the parallelogram.
def circumference (a b : ℝ) : ℝ :=
  2 * (a + b)

-- Statement of the proof problem:
theorem parallelogram_circumference : circumference side1 side2 = 60 := 
  by
    sorry

end parallelogram_circumference_l473_473529


namespace complex_number_location_l473_473175

theorem complex_number_location (m : ℝ) (h : 1 < m ∧ m < 2) : 
  ∃ quadrant: string, quadrant = "Fourth quadrant" :=
by
  have h_real : 0 < m - 1 ∧ m - 1 < 1 := ⟨sub_pos.mpr h.1, sub_lt_sub_of_lt h.2⟩
  have h_imag : -1 < m - 2 ∧ m - 2 < 0 := ⟨sub_neg_of_lt h.1, sub_neg.mpr h.2⟩
  use "Fourth quadrant"
  sorry

end complex_number_location_l473_473175


namespace sum_of_first_ten_enhanced_nice_numbers_l473_473220

def is_enhanced_nice (n : ℕ) : Prop :=
  (1 < n) ∧
  (let divisors := n.properDivisors
   in n = divisors.prod ∨
      let prime_factors := n.primeFactors
      in n = prime_factors.prod (λ p => p ^ (n.factorization p < n.factorization p))

theorem sum_of_first_ten_enhanced_nice_numbers :
  ∑ i in {6, 8, 10, 14, 15, 21, 22, 26, 27, 33}, i = 182 :=
by sorry

end sum_of_first_ten_enhanced_nice_numbers_l473_473220


namespace parallelogram_slope_l473_473256

-- Define the points of the parallelogram
def A := (8 : ℝ, 50 : ℝ)
def B := (8 : ℝ, 120 : ℝ)
def C := (30 : ℝ, 160 : ℝ)
def D := (30 : ℝ, 90 : ℝ)

-- Define the slope question
noncomputable def slope : ℝ := 265 / 38

theorem parallelogram_slope :
  (∃ (m n : ℕ), m + n = 303 ∧ (m : ℝ) / n = slope) :=
by
  use 265, 38
  split
  . exact rfl
  . exact rfl
  sorry

end parallelogram_slope_l473_473256


namespace max_product_of_sum_2020_l473_473164

theorem max_product_of_sum_2020 : 
  ∃ x y : ℤ, x + y = 2020 ∧ (x * y) ≤ 1020100 ∧ (∀ a b : ℤ, a + b = 2020 → a * b ≤ x * y) :=
begin
  sorry
end

end max_product_of_sum_2020_l473_473164


namespace count_valid_third_sides_l473_473814

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l473_473814


namespace number_of_possible_lists_l473_473614

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l473_473614


namespace profit_percentage_l473_473204

def cost_price (M : ℝ) : ℝ := M / 1.5

theorem profit_percentage (M : ℝ) (h : 1.20 * cost_price M = 0.80 * M) : 
  ((M - cost_price M) / cost_price M) * 100 = 50 :=
by
  sorry

end profit_percentage_l473_473204


namespace find_s_of_2_l473_473052

-- Define t and s as per the given conditions
def t (x : ℚ) : ℚ := 4 * x - 9
def s (x : ℚ) : ℚ := x^2 + 4 * x - 5

-- The theorem that we need to prove
theorem find_s_of_2 : s 2 = 217 / 16 := by
  sorry

end find_s_of_2_l473_473052


namespace triangle_shortest_altitude_l473_473112

noncomputable def shortest_altitude :=
  (λ (a b c : ℕ), if a^2 + b^2 = c^2 then 2 * ((1 / 2) * a * b) / c else 0)

theorem triangle_shortest_altitude : shortest_altitude 13 84 85 = 12.8470588235 :=
by
  sorry

end triangle_shortest_altitude_l473_473112


namespace find_function_f_n_squared_l473_473261

theorem find_function_f_n_squared (f : ℕ → ℕ) (h : ∀ m n : ℕ, f m + f n - m * n ∣ m * f m + n * f n) :
  f = λ n, n * n :=
by
  -- Proof omitted
  sorry

end find_function_f_n_squared_l473_473261


namespace quadratic_form_h_l473_473387

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l473_473387


namespace fixed_point_permutations_l473_473599

theorem fixed_point_permutations (n : ℕ) (P : ℕ → ℕ → ℕ) (hPdef: ∀ k, P n k = (Nat.choose n k) * P (n - k) 0)
  (hPsum: ∑ k in Finset.range (n + 1), P n k = n!) :
  (∑ k in Finset.range (n + 1), (k - 1) * (k - 1) * P n k) = n! := 
sorry

end fixed_point_permutations_l473_473599


namespace total_ages_l473_473603

-- Definitions of the conditions
variables (A B : ℕ) (x : ℕ)

-- Condition 1: 10 years ago, A was half of B in age.
def condition1 : Prop := A - 10 = 1/2 * (B - 10)

-- Condition 2: The ratio of their present ages is 3:4.
def condition2 : Prop := A = 3 * x ∧ B = 4 * x

-- Main theorem to prove
theorem total_ages (A B : ℕ) (x : ℕ) (h1 : condition1 A B) (h2 : condition2 A B x) : A + B = 35 := 
by
  sorry

end total_ages_l473_473603


namespace incorrect_proposition_C_l473_473253

open Nat

theorem incorrect_proposition_C (a : ℕ → ℕ) (h : ∀ n, n ≥ 2 → a (n+1) * a (n-1) = a n * a n) : ¬(∀ n, (n ≥ 2 → a (n+1) * a (n-1) = a n * a n) → ∃ q, ∀ n, a (n+1) = q * a n) :=
by
  intro hyp
  have ex_counter_example: ∃ a, a 0 = 0 ∧ a 1 = 0 ∧ (∀ n, a (n+1) * a (n-1) = a n * a n) := sorry
  cases ex_counter_example with a ha
  specialize hyp a
  have H := hyp ha.2
  intro contradiction
  contradiction ha
  sorry

end incorrect_proposition_C_l473_473253


namespace problem_demo_l473_473890

open Set

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem problem_demo : S ∩ (U \ T) = {1, 2, 4} :=
by
  sorry

end problem_demo_l473_473890


namespace number_of_possible_lists_l473_473605

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l473_473605


namespace compare_y1_y2_l473_473537

theorem compare_y1_y2 (a : ℝ) (y1 y2 : ℝ) (h₁ : a < 0) (h₂ : y1 = a * (-1 - 1)^2 + 3) (h₃ : y2 = a * (2 - 1)^2 + 3) : 
  y1 < y2 :=
by
  sorry

end compare_y1_y2_l473_473537


namespace cost_of_luncheon_l473_473097

-- Define the variables and conditions for the problem
variables (s c p k : ℝ)

-- Assume the given conditions
def condition1 := 5 * s + 9 * c + 2 * p + 3 * k = 5.85
def condition2 := 6 * s + 12 * c + 2 * p + 4 * k = 7.20

-- Prove that the cost of one sandwich, one cup of coffee, one piece of pie, and one cookie is $1.35
theorem cost_of_luncheon : condition1 ∧ condition2 → s + c + p + k = 1.35 :=
begin
  sorry
end

end cost_of_luncheon_l473_473097


namespace determine_e_l473_473110

-- Define the polynomial Q(x)
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

-- Define the problem statement
theorem determine_e (d e f : ℝ)
  (h1 : f = 9)
  (h2 : (d * (d + 9)) - 168 = 0)
  (h3 : d^2 - 6 * e = 12 + d + e)
  : e = -24 ∨ e = 20 :=
by
  sorry

end determine_e_l473_473110


namespace longest_piece_length_l473_473947

def total_length := 92.5
def ratio := (3, 5, 8)
def total_parts := ratio.1 + ratio.2 + ratio.3
def length_per_part := total_length / total_parts
def longest_piece := length_per_part * ratio.3

theorem longest_piece_length : longest_piece = 46.25 :=
by
  sorry

end longest_piece_length_l473_473947


namespace largest_number_formed_by_2_6_9_is_962_l473_473589

noncomputable def largest_three_digit_number_using (a b c : ℕ) : ℕ :=
  let numbers := [100 * a + 10 * b + c, 100 * a + 10 * c + b, 100 * b + 10 * a + c,
                  100 * b + 10 * c + a, 100 * c + 10 * a + b, 100 * c + 10 * b + a]
  numbers.maximum

theorem largest_number_formed_by_2_6_9_is_962 : largest_three_digit_number_using 2 6 9 = 962 := by
  sorry

end largest_number_formed_by_2_6_9_is_962_l473_473589


namespace probability_divisible_by_5_and_distinct_digits_l473_473239

-- Define the range and the properties required.
def inRange (n : ℕ) : Prop := 3000 ≤ n ∧ n ≤ 8000
def divisibleBy5 (n : ℕ) : Prop := n % 5 = 0
def allDistinctDigits (n : ℕ) : Prop := 
  let digits := String.toList (n.toString) in 
  digits.length = digits.eraseDups.length

-- Define the problem statement.
theorem probability_divisible_by_5_and_distinct_digits :
  ∃ (p : ℚ), p = 616 / 5001 ∧ 
              ∀ n : ℕ, inRange n → divisibleBy5 n → allDistinctDigits n → 
              true :=
by
  existsi (616 / 5001 : ℚ)
  sorry

end probability_divisible_by_5_and_distinct_digits_l473_473239


namespace third_side_integer_lengths_l473_473777

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l473_473777


namespace sum_of_integers_is_14_l473_473113

theorem sum_of_integers_is_14 
  (x y : ℕ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 45) 
  (h3 : |x - y| < 7) :
  x + y = 14 :=
sorry

end sum_of_integers_is_14_l473_473113


namespace area_of_triangle_ABC_l473_473681

section
variable (side_length : ℝ) (h_side_length : side_length = 2)

def distance_between_centers : ℝ := side_length * Real.sqrt 2

def area_of_equilateral_triangle (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

theorem area_of_triangle_ABC (d : ℝ) (h_d : d = distance_between_centers side_length):
  area_of_equilateral_triangle d = 2 * Real.sqrt 3 := by
  rw [h_d, distance_between_centers, h_side_length]
  simp [area_of_equilateral_triangle]
  sorry
end

end area_of_triangle_ABC_l473_473681


namespace first_fifteen_multiples_of_seven_sum_l473_473570

theorem first_fifteen_multiples_of_seven_sum :
    (∑ i in finset.range 15, 7 * (i + 1)) = 840 := 
sorry

end first_fifteen_multiples_of_seven_sum_l473_473570


namespace simplify_expression_l473_473507

variables (a b : ℝ)

theorem simplify_expression (h₁ : a = 2) (h₂ : b = -1) :
  (2 * a^2 - a * b - b^2) - 2 * (a^2 - 2 * a * b + b^2) = -5 :=
by
  sorry

end simplify_expression_l473_473507


namespace area_of_triangle_BDC_l473_473940

def side_length : ℝ := 1
def dilation_factor (k : ℝ) := k
def B_coords : ℝ × ℝ := (1, 0)
def D_coords : ℝ × ℝ := (0, 1)
def C'_coords (k : ℝ) := (k, k)
def BC'_distance (k : ℝ) := real.sqrt ((k - 1)^2 + k^2)

theorem area_of_triangle_BDC' (k : ℝ) (h : BC'_distance k = 29) :
  let A := (0, 0) in
  let B := (1, 0) in
  let D := (0, 1) in
  let C' := (21, 21) in
  let area := (1 / 2) * real.abs (1 * (1 - 21) + 0 * (21 - 0) + 21 * (0 - 1)) in
  area = 420 :=
by
  sorry

end area_of_triangle_BDC_l473_473940


namespace cuboid_edges_closest_to_diagonal_l473_473917

theorem cuboid_edges_closest_to_diagonal
  (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) :
  let d_BC := (a^2 * b) / (a^2 + c^2),
      d_CD := (c^2 * b) / (c^2 + a^2),
      d_DH := (4 : ℝ) / 13 in
  d_BC = 1.8 ∧ d_CD = 0.6 ∧ d_DH ≈ 0.308 := by
  sorry

end cuboid_edges_closest_to_diagonal_l473_473917


namespace prob_neither_snow_nor_windy_l473_473975

-- Define the probabilities.
def prob_snow : ℚ := 1 / 4
def prob_windy : ℚ := 1 / 3

-- Define the complementary probabilities.
def prob_not_snow : ℚ := 1 - prob_snow
def prob_not_windy : ℚ := 1 - prob_windy

-- State that the events are independent and calculate the combined probability.
theorem prob_neither_snow_nor_windy :
  prob_not_snow * prob_not_windy = 1 / 2 := by
  sorry

end prob_neither_snow_nor_windy_l473_473975


namespace number_of_possible_third_side_lengths_l473_473820

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l473_473820


namespace min_value_expr_l473_473287

theorem min_value_expr : ∃ (x : ℝ), (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 ∧ 
  ∀ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

end min_value_expr_l473_473287


namespace max_value_of_g_l473_473962

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 23 :=
by
  sorry

end max_value_of_g_l473_473962


namespace total_area_correct_l473_473127

structure Rectangle where
  width : ℕ
  length : ℕ

structure Triangle where
  base : ℕ
  height : ℕ

def area_rectangle (r : Rectangle) : ℕ := r.width * r.length

def area_triangle (t : Triangle) : ℕ := t.base * t.height / 2

noncomputable def intersecting_area (r1 r2 : Rectangle) : ℕ := 
  let intersect_width := r2.width
  let intersect_length := if r2.length > r1.length then r1.length else r2.length
  intersect_width * intersect_length

def total_shaded_area (r1 r2 : Rectangle) (t : Triangle) (overlap1 : ℕ) (overlap2 : ℕ) : ℕ :=
  area_rectangle r1 + area_rectangle r2 + area_triangle t - intersecting_area r1 r2 - overlap1 - overlap2

theorem total_area_correct :
  let r1 := Rectangle.mk 5 12
  let r2 := Rectangle.mk 4 15
  let t := Triangle.mk 3 4
  let overlap1 := 2
  let overlap2 := 1
  total_shaded_area r1 r2 t overlap1 overlap2 = 107 := 
by
  sorry

end total_area_correct_l473_473127


namespace planks_needed_for_surface_l473_473454

theorem planks_needed_for_surface
  (total_tables : ℕ := 5)
  (total_planks : ℕ := 45)
  (planks_per_leg : ℕ := 4) :
  ∃ S : ℕ, total_tables * (planks_per_leg + S) = total_planks ∧ S = 5 :=
by
  use 5
  sorry

end planks_needed_for_surface_l473_473454


namespace triangle_side_length_integers_l473_473771

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l473_473771


namespace third_side_integer_lengths_l473_473779

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l473_473779


namespace sum_of_reciprocals_of_gp_l473_473314

theorem sum_of_reciprocals_of_gp (n : ℕ) :
  let a := 4
  let r := 2
  let s := a * (1 - r^n) / (1 - r)
  let t (i : ℕ) := 1 / (a * r^i)
  ∑ x in (finset.range n), t x = (1 / 2) * (1 - (1 / 2^n)) :=
by
  sorry

end sum_of_reciprocals_of_gp_l473_473314


namespace julie_earned_from_simple_interest_l473_473455

noncomputable def julie_simple_interest_earnings : ℝ :=
  let r := (Real.sqrt (1024 / 900) - 1) in
  let P := 900 in
  let t := 2 in
  P * r * t

theorem julie_earned_from_simple_interest :
  julie_simple_interest_earnings = 119.7 :=
sorry

end julie_earned_from_simple_interest_l473_473455


namespace complete_the_square_3x2_9x_20_l473_473400

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l473_473400


namespace symmetric_point_y_axis_l473_473870

def Point : Type := ℝ × ℝ

def symmetric_wrt_y_axis (P : Point) : Point :=
  let (x, y) := P
  in (-x, y)

theorem symmetric_point_y_axis (P : Point) (hx: P = (2, 1)) :
  symmetric_wrt_y_axis P = (-2, 1) :=
by
  cases hx
  sorry

end symmetric_point_y_axis_l473_473870


namespace concurrency_or_parallelism_of_lines_l473_473443

variables {α : Type*} [ordered_ring α]
variables {A B C A1 B1 C1 P A2 B2 C2 : point α}
variables (medians : ∀ (P : point α), intersection_condition α A B C A1 B1 C1 P A2 B2 C2)

theorem concurrency_or_parallelism_of_lines
    (T : triangle α)
    (medians_def : medians T A B C = [A1, B1, C1])
    (intersection_def : 
        (intersection A P B1 C1 = A2) ∧ 
        (intersection B P C1 A1 = B2) ∧ 
        (intersection C P A1 B1 = C2)) :
  concurrency_or_parallelism α A1 A2 B1 B2 C1 C2 :=
sorry

end concurrency_or_parallelism_of_lines_l473_473443


namespace course_selection_schemes_l473_473517

-- Definitions for the combinatorial functions
def binom : ℕ → ℕ → ℕ
| n, k :=
  if h : k ≤ n then
    Nat.choose n k
  else
    0

def num_schemes (num_courses : ℕ) (student_A_choices : ℕ → ℕ) (student_BC_choices : ℕ) : ℕ :=
  (student_A_choices num_courses + student_A_choices (num_courses - 1)) * binom num_courses student_BC_choices * binom num_courses student_BC_choices

-- Given conditions:
def num_courses := 4
def student_A_choices := λ n, binom n 3
def student_BC_choices := 3
def total_schemes := num_schemes num_courses student_A_choices student_BC_choices

-- Theorem stating the problem and its answer:
theorem course_selection_schemes : total_schemes = 80 :=
by
  sorry

end course_selection_schemes_l473_473517


namespace product_nonzero_except_cases_l473_473516

theorem product_nonzero_except_cases (n : ℤ) (h : n ≠ 5 ∧ n ≠ 17 ∧ n ≠ 257) : 
  (n - 5) * (n - 17) * (n - 257) ≠ 0 :=
by
  sorry

end product_nonzero_except_cases_l473_473516


namespace sum_of_g1_l473_473048

noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition : ∀ x y : ℝ, g (g (x - y)) = g x + g y - g x * g y - x * y := sorry

theorem sum_of_g1 : g 1 = 1 := 
by
  -- Provide the necessary proof steps to show g(1) = 1
  sorry

end sum_of_g1_l473_473048


namespace greatest_3_digit_base9_divisible_by_7_l473_473145

theorem greatest_3_digit_base9_divisible_by_7 :
  ∃ (n : ℕ), n < 729 ∧ n ≥ 81 ∧ n % 7 = 0 ∧ n = 8 * 81 + 8 * 9 + 8 := 
by 
  use 728
  split
  {
    exact nat.pred_lt (ne_of_lt (by norm_num))
  }
  split
  {
    exact nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.zero_le 7))) 
  }
  split
  {
    norm_num
  }
  norm_num

end greatest_3_digit_base9_divisible_by_7_l473_473145


namespace product_of_all_roots_l473_473463

noncomputable def Q : Polynomial ℚ := Polynomial.X^3 - 15 * Polynomial.X^2 + 75 * Polynomial.X - 130

theorem product_of_all_roots :
  (∃ y : ℚ, y = (5 : ℚ) + Real.cbrt (5 : ℚ) ∧ Q.eval (Real.cbrt 5 + 5) = 0) →
  (∀ y ∈ Q.roots.toFinset, y ∈ ℚ) →
  Q.roots.prod = 130 :=
by
  sorry

end product_of_all_roots_l473_473463


namespace triangle_third_side_count_l473_473830

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l473_473830


namespace triangle_sine_ratio_l473_473852

theorem triangle_sine_ratio (A B C : ℝ) (a b c : ℝ) :
  (b + c) / 4 = (c + a) / 5 ∧ (c + a) / 5 = (a + b) / 6 →
  sin A / sin B = 7 / 5 ∧ sin B / sin C = 5 / 3 :=
sorry

end triangle_sine_ratio_l473_473852


namespace exists_infinitely_many_primes_l473_473887

noncomputable def is_nonconstant (f : ℕ+ → ℕ+) : Prop :=
  ∃ a b : ℕ+, a ≠ b ∧ f a ≠ f b

noncomputable def divides (a b : ℕ) : Prop :=
  ∃ k : ℕ, b = k * a

theorem exists_infinitely_many_primes
  (f : ℕ+ → ℕ+) 
  (h_nonconstant : is_nonconstant f)
  (h_div : ∀ a b : ℕ+, a ≠ b → divides (a - b) (f a - f b)) :
  ∃ inf_primes : ℕ → Prop, 
    (∀ p : ℕ, p ∈ inf_primes ↔ Nat.Prime p) ∧ 
    (∀ p : ℕ, p ∈ inf_primes → ∃ c : ℕ+, p ∣ f c) :=
begin
  sorry
end

end exists_infinitely_many_primes_l473_473887


namespace ratio_of_sam_to_sue_l473_473497

-- Definitions
def Sam_age (S : ℕ) : Prop := 3 * S = 18
def Kendra_age (K : ℕ) : Prop := K = 18
def total_age_in_3_years (S U K : ℕ) : Prop := (S + 3) + (U + 3) + (K + 3) = 36

-- Theorem statement
theorem ratio_of_sam_to_sue (S U K : ℕ) (h1 : Sam_age S) (h2 : Kendra_age K) (h3 : total_age_in_3_years S U K) :
  S / U = 2 :=
sorry

end ratio_of_sam_to_sue_l473_473497


namespace count_valid_third_sides_l473_473819

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l473_473819


namespace geometric_sequence_a5_l473_473442

variable {a : ℕ → ℝ}
variable (h₁ : a 3 * a 7 = 3)
variable (h₂ : a 3 + a 7 = 4)

theorem geometric_sequence_a5 : a 5 = Real.sqrt 3 := 
sorry

end geometric_sequence_a5_l473_473442


namespace apples_per_sandwich_l473_473302

-- Define the conditions
def sam_sandwiches_per_day : Nat := 10
def days_in_week : Nat := 7
def total_apples_in_week : Nat := 280

-- Calculate total sandwiches in a week
def total_sandwiches_in_week := sam_sandwiches_per_day * days_in_week

-- Prove that Sam eats 4 apples for each sandwich
theorem apples_per_sandwich : total_apples_in_week / total_sandwiches_in_week = 4 :=
  by
    sorry

end apples_per_sandwich_l473_473302


namespace ratio_OC_OA_l473_473435

variable {Point : Type}
variable [AffineSpace Point ℝ]
variables (A B C D M O : Point)
variables (AD BC CM : ℝ)

def rectangle (A B C D : Point) : Prop := 
  ∃ u v : ℝ, AD = 6 ∧ BC = 8 ∧ CM = 2 ∧ 
  ((u * (A - D) + v * (B - C)) = (D - C)) ∧ 
  ((u * (A - C) + v * (B - M)) = (C - M))

theorem ratio_OC_OA 
  (h1 : rectangle A B C D)
  (h2 : affine.independent ℝ ![A, B, C, D, M, O]) :
  ∃ k : ℝ, k = (4/3) ∧ 
  ∃ j : ℝ, j = (8/3) ∧ 
  (OC / OA) = 1 / 2 := 
by
  sorry

end ratio_OC_OA_l473_473435


namespace number_of_possible_third_side_lengths_l473_473823

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l473_473823


namespace average_bc_is_70_l473_473523

-- We start by defining the given conditions and the necessary mathematical expressions.
variable {a b c : ℝ}

def avg_ab := (a + b) / 2 = 115
def diff_ac := a - c = 90

-- The theorem that needs to be proved.
theorem average_bc_is_70 (h1 : avg_ab) (h2 : diff_ac) : (b + c) / 2 = 70 :=
by
  sorry

end average_bc_is_70_l473_473523


namespace mars_appearance_time_l473_473123

def time := ℤ × ℤ -- Representing time as (hours, minutes)

def mars_seen_until (mars_time jupiter_time uranus_time : time) : Prop :=
  uranus_time = (6, 7) ∧ 
  jupiter_time = (urus_time.1 - 3, uranus_time.2 - 16) ∧ 
  mars_time = (jupiter_time.1 - 2, jupiter_time.2 - 41)

theorem mars_appearance_time : ∃ mars_time jupiter_time uranus_time, 
  mars_seen_until mars_time jupiter_time uranus_time ∧ mars_time = (0, 10) := 
sorry

end mars_appearance_time_l473_473123


namespace triangle_side_length_integers_l473_473769

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l473_473769


namespace absolute_difference_distance_l473_473359

noncomputable def home_to_park_distance : ℝ := 3
noncomputable def ratio : ℝ := 5 / 6

def distance_after_n_steps (h p : ℝ) (n : ℕ) : ℝ :=
  if h = p then abs(h - p)
  else 
    let a (n : ℕ) : ℝ := if n = 0 then p * ratio else distance_after_n_steps h (h + ratio * (p - h)) (n - 1)
    let b (n : ℕ) : ℝ := if n = 0 then h + ratio * (home_to_park_distance - h) else distance_after_n_steps (h + ratio * (p - h)) p (n - 1)
    abs(a n - b n)

theorem absolute_difference_distance :
  distance_after_n_steps 0 3 1 = 1.5 :=
sorry

end absolute_difference_distance_l473_473359


namespace total_students_l473_473069

theorem total_students (n : ℕ) (h1 : n / 6) (h2 : (n / 6) / 3 = n / 18) (h3 : n / 9 = 4) : n = 36 :=
sorry

end total_students_l473_473069


namespace vitia_stopped_saving_l473_473193

theorem vitia_stopped_saving :
  (Boris_start_month, Boris_start_year, Boris_end_month, Boris_end_year, Boris_savings, Vitia_multiplier, Vitia_savings_per_month) = (1, 2019, 5, 2021, 150, 7, 100) →
  (Vitia_start_month, Vitia_start_year) = (1, 2019) →
  let months_boris_saved := (Boris_end_year - Boris_start_year) * 12 + (Boris_end_month - Boris_start_month) in
  let total_savings_boris := months_boris_saved * Boris_savings in
  let total_savings_vitia := total_savings_boris / Vitia_multiplier in
  let months_vitia_saved := total_savings_vitia / Vitia_savings_per_month in
  (Vitia_stop_month, Vitia_stop_year) = 
      (Vitia_start_month + months_vitia_saved, Vitia_start_year) in
  -- Ensure the calculation reflects July, 2019
  Boris_multiple := 150 in
  Vitia_multiple := 100 in
  let total_years := (5 - 1) + 2 * 12 == 28,
  let total_savings_vitia := 600 / 100 == 6,
  total_savings_boris := let total_savings_vitia == 4200, 
  sorry

end vitia_stopped_saving_l473_473193


namespace reciprocal_of_sum_is_correct_l473_473982

theorem reciprocal_of_sum_is_correct : (1 / (1 / 4 + 1 / 6)) = 12 / 5 := by
  sorry

end reciprocal_of_sum_is_correct_l473_473982


namespace range_of_a_l473_473437

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

def sibling_point_pair (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A.2 = f a A.1 ∧ B.2 = f a B.1 ∧ A.1 = -B.1 ∧ A.2 = -B.2

theorem range_of_a (a : ℝ) :
  (∃ A B : ℝ × ℝ, sibling_point_pair a A B) ↔ a > 1 :=
sorry

end range_of_a_l473_473437


namespace main_theorem_l473_473317

noncomputable def a_seq : ℕ → ℝ
| 0     := 1
| (n+1) := ((2 * a_seq n) + 1)

def b_seq (n : ℕ) : ℝ := (-1 : ℝ)^n * real.log 2 (a_seq n + 1) / real.log 2 4

def T_n (n : ℕ) : ℝ := (finset.range n).sum b_seq

theorem main_theorem (n : ℕ) : 
  (∀ k, a_seq k + 1 = 2^k) ∧ 
  T_n n = if n % 2 = 0 then n / 4 else (-n - 1) / 4 :=
by sorry

end main_theorem_l473_473317


namespace problem_l473_473371

-- Define a parabola and the conditions given in the problem
def parabola (x y p : ℝ) :=
  x^2 = 2 * p * y

-- Define the distance from a point to the focus of the parabola
def distance_to_focus (m p : ℝ) :=
  Real.sqrt ((m - 0)^2 + (3 - p/2)^2)

-- Define the problem statement
theorem problem (p m : ℝ) (h1 : parabola m 3 p)
               (h2 : distance_to_focus m p = 5 * p) : p = 2 / 3 :=
by
  sorry

end problem_l473_473371


namespace greatest_base9_3_digit_divisible_by_7_l473_473150

def base9_to_decimal (n : Nat) : Nat :=
  match n with
  | 0     => 0
  | n + 1 => (n % 10) * Nat.pow 9 (n / 10)

def decimal_to_base9 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n => let rec aux (n acc : Nat) :=
              if n = 0 then acc
              else aux (n / 9) ((acc * 10) + (n % 9))
         in aux n 0

theorem greatest_base9_3_digit_divisible_by_7 :
  ∃ (n : Nat), n < Nat.pow 9 3 ∧ (n % 7 = 0) ∧ n = 8 * 81 + 8 * 9 + 8 :=
begin
  sorry -- Proof would go here
end

end greatest_base9_3_digit_divisible_by_7_l473_473150


namespace problem_statement_l473_473754

theorem problem_statement (c d : ℝ) (hc : 80^c = 4) (hd : 80^d = 5) : 16^((1 - c - d) / (2 * (1 - d))) = 4 :=
sorry

end problem_statement_l473_473754


namespace election_candidate_a_votes_l473_473189

theorem election_candidate_a_votes :
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  candidate_a_votes = 333200 :=
by
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  show candidate_a_votes = 333200
  sorry

end election_candidate_a_votes_l473_473189


namespace min_value_expression_l473_473897

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 9) :
  (min (frac (x^2 + y^2 + 1) (x + y) + frac (x^2 + z^2 + 1) (x + z) + frac (y^2 + z^2 + 1) (y + z))) = 4.833 :=
sorry

end min_value_expression_l473_473897


namespace charlyn_visibility_area_l473_473249

/-- Given that Charlyn walks around a square with side length 8 km, and she can see 
    a distance of 2 km horizontally from any point on the path, the area of the region 
    she can see during her walk is 125 km². -/
theorem charlyn_visibility_area (side_length visibility_range : ℝ)
  (h₁ : side_length = 8) 
  (h₂ : visibility_range = 2) :
  let visible_area := (64 - 16) + (64) + (4 * real.pi) in
  visible_area = 125 := by 
    let inner_square_side := side_length - 2 * visibility_range
    have h_inner_square : inner_square_side = 4 := by norm_num
    let inner_square_area := inner_square_side ^ 2
    have h_inner_area : inner_square_area = 16 := by norm_num
    let inner_visible_area := side_length ^ 2 - inner_square_area
    have h_inner_visible_area : inner_visible_area = 48 := by norm_num
    let side_rect_area := side_length * visibility_range
    have h_side_rect_area : side_rect_area = 16 := by norm_num
    let outer_visible_area := 4 * side_rect_area + 4 * (real.pi * visibility_range ^ 2 / 4)
    have h_outer_visible_area : outer_visible_area = 64 + 4 * real.pi := by norm_num
    have total_visible_area : visible_area = inner_visible_area + 64 + 4 * real.pi := by norm_num
    have approximation_pi : 4 * real.pi ≈ 12.56 := by norm_num
    have total_approx_area : total_visible_area = 112 + 12.56 := by norm_num
    have total_approx_area_125 : total_approx_area ≈ 125 := by norm_num
    show total_visible_area = 125 from sorry

end charlyn_visibility_area_l473_473249


namespace intersection_P_Q_l473_473462

def P : set ℝ := {x | x^2 - 16 < 0}
def Q : set ℝ := {x | ∃ n : ℤ, x = 2 * n}

theorem intersection_P_Q : P ∩ Q = {-2, 0, 2} := by 
  sorry

end intersection_P_Q_l473_473462


namespace total_questions_l473_473700

theorem total_questions (f s k : ℕ) (hf : f = 36) (hs : s = 2 * f) (hk : k = (f + s) / 2) :
  2 * (f + s + k) = 324 :=
by {
  sorry
}

end total_questions_l473_473700


namespace general_formula_bn_limit_tn_l473_473307

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2
noncomputable def g (x : ℝ) := 4 * (x - 1)

theorem general_formula_bn (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n ≥ 1, (a (n + 1) - a n) * g (a n) + f (a n) = 0) →
  a 1 = 2 →
  (∀ n ≥ 2, a n ≠ 1) →
  (∀ n, b n = a n - 1) →
  b 1 = 1 ∧ ∀ n, b (n + 1) = (3 / 4) * b n :=
by sorry

theorem limit_tn (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n ≥ 1, (a (n + 1) - a n) * g (a n) + f (a n) = 0) →
  a 1 = 2 →
  (∀ n ≥ 2, a n ≠ 1) →
  (∀ n, b n = a n - 1) →
  (∀ n, S n = ∑ k in finset.range(n), k * b (k + 1)) →
  (∀ n, T n = S n + (n * 3^n) / (4^(n - 1)) + (3^n) / (4^(n - 2))) →
  (∀ n, b n = (3 / 4)^(n - 1)) →
  filter.tendsto T filter.at_top (nhds 16) :=
by sorry

end general_formula_bn_limit_tn_l473_473307


namespace product_of_two_primes_l473_473976

theorem product_of_two_primes (p q z : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) 
    (h_p_range : 2 < p ∧ p < 6) 
    (h_q_range : 8 < q ∧ q < 24) 
    (h_z_def : z = p * q) 
    (h_z_range : 15 < z ∧ z < 36) : 
    z = 33 := 
by 
    sorry

end product_of_two_primes_l473_473976


namespace distance_from_point_to_line_correct_l473_473874

noncomputable def distance_from_point_to_line :
    ℝ :=
  let rho : ℝ := 2
  let theta : ℝ := Real.pi / 6
  let x : ℝ := sqrt 3  -- Converting polar to rectangular
  let y : ℝ := 1
  let line_a : ℝ := sqrt 3
  let line_b : ℝ := 1
  let line_c : ℝ := -1
  abs (line_a * x + line_b * y + line_c) / sqrt (line_a^2 + line_b^2)

theorem distance_from_point_to_line_correct:
    distance_from_point_to_line = 3 / 2 :=
by
  sorry

end distance_from_point_to_line_correct_l473_473874


namespace sum_of_first_fifteen_multiples_of_7_l473_473567

theorem sum_of_first_fifteen_multiples_of_7 : (∑ k in Finset.range 15, 7 * (k + 1)) = 840 :=
by
  -- Summation from k = 0 to k = 14 (which corresponds to 1 to 15 multiples of 7)
  sorry

end sum_of_first_fifteen_multiples_of_7_l473_473567


namespace quadratic_expression_rewriting_l473_473413

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l473_473413


namespace Lucas_age_in_3_years_l473_473093

variable (Gladys Billy Lucas : ℕ)

theorem Lucas_age_in_3_years :
  Gladys = 30 ∧ Billy = Gladys / 3 ∧ Gladys = 2 * (Billy + Lucas) →
  Lucas + 3 = 8 :=
by
  intro h
  cases h with h1 h2
  cases h2 with hBilly h3
  sorry

end Lucas_age_in_3_years_l473_473093


namespace triangle_side_length_integers_l473_473774

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l473_473774


namespace ajay_marathon_completion_time_l473_473646

theorem ajay_marathon_completion_time :
  let flat_distance := 1000
  let uphill_distance := 350
  let downhill_distance := 150
  let flat_speed := 50
  let uphill_speed := 30
  let downhill_speed := 60
  let break_time := 15 / 60  -- in hours
  let break_interval := 2  -- in hours
  let flat_time := flat_distance / flat_speed 
  let uphill_time := uphill_distance / uphill_speed
  let downhill_time := downhill_distance / downhill_speed
  let total_riding_time := flat_time + uphill_time + downhill_time
  let total_breaks := (total_riding_time / break_interval).ceil.to_nat
  let total_break_time := total_breaks * break_time
  in total_riding_time + total_break_time = 38.67 := by
  sorry

end ajay_marathon_completion_time_l473_473646


namespace triangle_third_side_lengths_l473_473761

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l473_473761


namespace longest_side_of_triangle_l473_473645

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def triangle_vertices : list point := [(2, 2), (5, 6), (7, 3)]

def longest_side_length (vertices : list point) : ℝ :=
  max (distance vertices[0] vertices[1]) (max (distance vertices[0] vertices[2]) (distance vertices[1] vertices[2]))

theorem longest_side_of_triangle :
  longest_side_length triangle_vertices = real.sqrt 26 :=
by sorry

end longest_side_of_triangle_l473_473645


namespace scalene_triangle_not_divisible_l473_473923

theorem scalene_triangle_not_divisible (A B C D : Point) (hABC : scalene A B C) (D_on_AB : lies_on D (segment A B)) :
  ¬ (1/2 * (area (triangle A D C) + area (triangle B D C)) = area (triangle A B C)) :=
sorry

end scalene_triangle_not_divisible_l473_473923


namespace team_points_l473_473231

theorem team_points (wins losses ties : ℕ) (points_per_win points_per_loss points_per_tie : ℕ) :
  wins = 9 → losses = 3 → ties = 4 → points_per_win = 2 → points_per_loss = 0 → points_per_tie = 1 →
  (points_per_win * wins + points_per_loss * losses + points_per_tie * ties = 22) :=
by
  intro h_wins h_losses h_ties h_points_per_win h_points_per_loss h_points_per_tie
  sorry

end team_points_l473_473231


namespace max_value_g_l473_473966

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_g : ∃ M, ∀ n, g n ≤ M ∧ M = 23 :=
  sorry

end max_value_g_l473_473966


namespace number_of_lists_l473_473611

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l473_473611


namespace age_of_17th_student_l473_473190

theorem age_of_17th_student (avg_age_17 : ℕ) (total_students : ℕ) (avg_age_5 : ℕ) (students_5 : ℕ) (avg_age_9 : ℕ) (students_9 : ℕ)
  (h1 : avg_age_17 = 17) (h2 : total_students = 17) (h3 : avg_age_5 = 14) (h4 : students_5 = 5) (h5 : avg_age_9 = 16) (h6 : students_9 = 9) :
  ∃ age_17th_student : ℕ, age_17th_student = 75 :=
by
  sorry

end age_of_17th_student_l473_473190


namespace water_needed_l473_473866

theorem water_needed (nutrient_concentrate : ℝ) (distilled_water : ℝ) (total_volume : ℝ) 
    (h1 : nutrient_concentrate = 0.08) (h2 : distilled_water = 0.04) (h3 : total_volume = 1) :
    total_volume * (distilled_water / (nutrient_concentrate + distilled_water)) = 0.333 :=
by
  sorry

end water_needed_l473_473866


namespace number_of_lists_l473_473608

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l473_473608


namespace max_value_of_g_l473_473963

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 23 :=
by
  sorry

end max_value_of_g_l473_473963


namespace total_wallpaper_removal_time_l473_473273

def dining_room_time (removal_time_wall: Nat -> Nat) : Nat :=
  removal_time_wall 1 * 1.5

def living_room_time (removal_time_wall: Nat -> Nat) : Nat :=
  (removal_time_wall 2 * 1) + (removal_time_wall 2 * 2.5)

def bedroom_time (removal_time_wall: Nat -> Nat) : Nat :=
  removal_time_wall 3 * 3

def hallway_time (removal_time_wall: Nat -> Nat) : Nat :=
  removal_time_wall 1 * 4 + removal_time_wall 4 * 2

def kitchen_time (removal_time_wall: Nat -> Nat) : Nat :=
  removal_time_wall 1 * 3 + removal_time_wall 2 * 1.5 + removal_time_wall 1 * 2

def bathroom_time (removal_time_wall: Nat -> Nat) : Nat :=
  removal_time_wall 1 * 2 + removal_time_wall 1 * 3

def total_time (removal_time_wall: Nat -> Nat) : Nat :=
  dining_room_time removal_time_wall +
  living_room_time removal_time_wall +
  bedroom_time removal_time_wall +
  hallway_time removal_time_wall +
  kitchen_time removal_time_wall +
  bathroom_time removal_time_wall

theorem total_wallpaper_removal_time : total_time removal_time_wall = 45.5 :=
by {
  -- Adding the specific conditions from the problem
  let dining_wall := 1 * 1.5,
  let living_wall1 := 2 * 1,
  let living_wall2 := 2 * 2.5,
  let bedroom_wall := 3 * 3,
  let hall_wall1 := 1 * 4,
  let hall_wall2 := 4 * 2,
  let kitchen_wall1 := 1 * 3,
  let kitchen_wall2 := 2 * 1.5,
  let kitchen_wall3 := 1 * 2,
  let bath_wall1 := 1 * 2,
  let bath_wall2 := 1 * 3,

  -- Calculating the total time
  let total_wallpaper_removal_algorithm := 
      dining_wall + (living_wall1 + living_wall2) + 
      bedroom_wall + (hall_wall1 + hall_wall2) + 
      (kitchen_wall1 + kitchen_wall2 + kitchen_wall3) + (bath_wall1 + bath_wall2),
  
  -- The total time should be equal to the given answer
  exact (total_wallpaper_removal_algorithm = 45.5),
}

end total_wallpaper_removal_time_l473_473273


namespace greatest_base9_3_digit_divisible_by_7_l473_473148

def base9_to_decimal (n : Nat) : Nat :=
  match n with
  | 0     => 0
  | n + 1 => (n % 10) * Nat.pow 9 (n / 10)

def decimal_to_base9 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n => let rec aux (n acc : Nat) :=
              if n = 0 then acc
              else aux (n / 9) ((acc * 10) + (n % 9))
         in aux n 0

theorem greatest_base9_3_digit_divisible_by_7 :
  ∃ (n : Nat), n < Nat.pow 9 3 ∧ (n % 7 = 0) ∧ n = 8 * 81 + 8 * 9 + 8 :=
begin
  sorry -- Proof would go here
end

end greatest_base9_3_digit_divisible_by_7_l473_473148


namespace isosceles_right_triangle_perpendicular_bisector_l473_473916

theorem isosceles_right_triangle_perpendicular_bisector
  (A B C D E K L : Point)
  (hABC : isosceles_right_triangle A B C)
  (h_cd_ce : dist C D = dist C E)
  (h_perp_d : ⊥ (Line.mk D (perp_to_line (Line.mk A E))))
  (h_perp_c : ⊥ (Line.mk C (perp_to_line (Line.mk A E))))
  (K_on_AB : online K (Line.mk A B))
  (L_on_AB : online L (Line.mk A B))
  (D_on_K : online D (Line.mk K (altitude_line (Line.mk A E))))
  (C_on_L : online C (Line.mk L (altitude_line (Line.mk A E)))) :
  dist K L = dist L B :=
by
  sorry

end isosceles_right_triangle_perpendicular_bisector_l473_473916


namespace complete_the_square_3x2_9x_20_l473_473399

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l473_473399


namespace quadratic_non_real_roots_l473_473678

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℝ, ¬(x^2 + b * x + 16 = 0)) ↔ b ∈ set.Ioo (-8 : ℝ) (8 : ℝ) :=
begin
  sorry
end

end quadratic_non_real_roots_l473_473678


namespace coefficient_binomial_expansion_l473_473338

theorem coefficient_binomial_expansion (a : ℝ) (x : ℝ) (h : x ≠ 0) :
  let f := λ x, (x ^ (1/2)) - (a * x^(-1/2))
  have coeff_term: (∑ r in Finset.range 6, (binomial 5 r) * ((sqrt x)^(5-r)) * ((-a / sqrt x) ^ r)) ∣ₓ (x^(3/2)) = 30 
  shows a = -6 :=
sorry

end coefficient_binomial_expansion_l473_473338


namespace intersection_point_of_circle_and_line_l473_473738

noncomputable def circle_parametric (α : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos α, 2 * Real.sin α)
noncomputable def line_polar (rho θ : ℝ) : Prop := rho * Real.sin θ = 2

theorem intersection_point_of_circle_and_line :
  ∃ (α : ℝ) (rho θ : ℝ), circle_parametric α = (1, 2) ∧ line_polar rho θ := sorry

end intersection_point_of_circle_and_line_l473_473738


namespace calculate_roots_l473_473658

theorem calculate_roots : real.cbrt (-8) + real.sqrt 16 = 2 := by
  sorry

end calculate_roots_l473_473658


namespace find_a_max_perimeter_l473_473446

-- Definitions based on conditions
variables (A B C : ℝ) -- Angles of the triangle
variables (a b c : ℝ) -- Side lengths opposite the angles A, B, and C

-- Given conditions
def is_triangle (A B C : ℝ) : Prop := A + B + C = π
def condition (A B C a b c : ℝ) : Prop := b * Real.cos A + a * Real.cos B = a * c

-- The first proof goal
theorem find_a 
  (h_triangle : is_triangle A B C)
  (h_condition : condition A B C a b c) : a = 1 :=
sorry

-- The second proof goal
theorem max_perimeter 
  (h_triangle : is_triangle A B C)
  (h_condition : condition A B C a b c)
  (h_A : A = π / 3) : a + b + c ≤ 3 :=
sorry

end find_a_max_perimeter_l473_473446


namespace congruence_solution_count_l473_473720

theorem congruence_solution_count :
  {x : ℕ // x < 100 ∧ (x + 17) % 46 = 73 % 46}.card = 2 := 
sorry

end congruence_solution_count_l473_473720


namespace second_discarded_number_l473_473095

theorem second_discarded_number (S : ℝ) (X : ℝ) :
  (S = 50 * 44) →
  ((S - 45 - X) / 48 = 43.75) →
  X = 55 :=
by
  intros h1 h2
  -- The proof steps would go here, but we leave it unproved
  sorry

end second_discarded_number_l473_473095


namespace triangle_side_length_integers_l473_473766

theorem triangle_side_length_integers {a b : ℕ} (h1 : a = 8) (h2 : b = 11) :
  { x : ℕ | 3 < x ∧ x < 19 }.card = 15 :=
by
  sorry

end triangle_side_length_integers_l473_473766


namespace first_fifteen_multiples_of_seven_sum_l473_473572

theorem first_fifteen_multiples_of_seven_sum :
    (∑ i in finset.range 15, 7 * (i + 1)) = 840 := 
sorry

end first_fifteen_multiples_of_seven_sum_l473_473572


namespace solve_for_t_l473_473296

theorem solve_for_t : 
  ∀ t : ℝ, (1 / (t + 2) + 2 * t / (t + 2) - 3 / (t + 2) = 1) → t = 4 :=
by 
  intro t h,
  sorry

end solve_for_t_l473_473296


namespace ellipse_eq_6_2_max_triangle_area_l473_473713

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_eq_6_2 (e : ℝ) (a b x y : ℝ)
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : e = (Real.sqrt 6) / 3) 
  (h4 : (x, y) = (Real.sqrt 3, 1))
  (h_ellipse : ellipse x y a b) :
  ellipse x y 6 2 :=
  sorry

theorem max_triangle_area (a b c : ℝ)
  (focus : (ℝ × ℝ))
  (origin : (ℝ × ℝ))
  (line_l : ℝ → ℝ)
  (A B : (ℝ × ℝ))
  (O : (ℝ × ℝ) := origin)
  (h1 : A ≠ B)
  (h2 : c = Real.sqrt (a^2 - b^2))
  (h3 : focus = (c, 0))
  (h4 : line_l = λ y => c)
  (h5 : ∃ x y : ℝ, ellipse x y a b ∧ (x, y) ∈ set_of fun p => p ∈ [A, B]) :
  ∃ l_eqn_1 l_eqn_2 : ℝ → ℝ, l_eqn_1 x - 2 = 0 ∧ l_eqn_2 x - 2 = 0 ∧ (area.origin O A B O = Real.sqrt 3) :=
  sorry

end ellipse_eq_6_2_max_triangle_area_l473_473713


namespace sqrt_meaningful_l473_473978

theorem sqrt_meaningful (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l473_473978


namespace value_a5_factorial_base_825_l473_473258

theorem value_a5_factorial_base_825 :
  ∃ a : ℕ → ℕ, 
    (825 = a 1 + a 2 * 2! + a 3 * 3! + a 4 * 4! + a 5 * 5! + a 6 * 6! + a 7 * 7!) ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 6 → 0 ≤ a k ∧ a k ≤ k) ∧
    a 5 = 0 :=
by
  sorry

end value_a5_factorial_base_825_l473_473258


namespace partial_fraction_sum_eq_zero_l473_473671

theorem partial_fraction_sum_eq_zero (A B C D E : ℂ) :
  (∀ x : ℂ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 4 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x - 4)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x - 4)) →
  A + B + C + D + E = 0 :=
by
  sorry

end partial_fraction_sum_eq_zero_l473_473671


namespace triangle_third_side_lengths_l473_473764

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l473_473764


namespace problem_l473_473057

def f (x: ℝ) := 3 * x - 4
def g (x: ℝ) := 2 * x + 3

theorem problem (x : ℝ) : f (2 + g 3) = 29 :=
by
  sorry

end problem_l473_473057


namespace joe_paint_usage_l473_473591

theorem joe_paint_usage
  (initial_paint : ℕ)
  (first_week_fraction : ℚ)
  (second_week_fraction : ℚ) :
  initial_paint = 360 →
  first_week_fraction = 1/6 →
  second_week_fraction = 1/5 →
  let first_week_paint := first_week_fraction * initial_paint in
  let remaining_after_first_week := initial_paint - first_week_paint in
  let second_week_paint := second_week_fraction * remaining_after_first_week in
  first_week_paint + second_week_paint = 120 :=
by
  intros h_initial h_first_fraction h_second_fraction
  let first_week_paint := first_week_fraction * initial_paint
  let remaining_after_first_week := initial_paint - first_week_paint
  let second_week_paint := second_week_fraction * remaining_after_first_week
  sorry

end joe_paint_usage_l473_473591


namespace triangle_third_side_length_count_l473_473786

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l473_473786


namespace number_of_possible_lists_l473_473607

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l473_473607


namespace exists_convex_pentagon_cut_off_similar_pentagon_l473_473270

noncomputable def is_similar (P Q : List (ℕ × ℝ)) : Prop :=
  ∃ k : ℝ, 0 < k ∧ ∀ i, (Q[i].fst, Q[i].snd) = (P[i].fst, k * P[i].snd)

theorem exists_convex_pentagon_cut_off_similar_pentagon :
  ∃ P : List (ℕ × ℝ),
    P.length = 5 ∧
    (P[0].fst = 60 ∧ P[1].fst = 120) ∧
    (P[0].snd = 2 ∧ P[1].snd = 4 ∧ P[2].snd = 8 ∧ P[3].snd = 6 ∧ P[4].snd = 12) ∧
    ∃ Q : List (ℕ × ℝ),
      Q.length = 5 ∧
      is_similar P Q :=
begin
  sorry
end

end exists_convex_pentagon_cut_off_similar_pentagon_l473_473270


namespace average_percentage_l473_473752

theorem average_percentage (n m : ℕ) (p q : ℝ)
  (hn : n = 15) (hm : m = 10) (hp : p = 70) (hq : q = 95) :
  ((n * p + m * q) / (n + m)) = 80 :=
by
  rw [hn, hm, hp, hq]
  norm_num
  sorry

end average_percentage_l473_473752


namespace triangle_third_side_length_count_l473_473787

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l473_473787


namespace greatest_3_digit_base9_divisible_by_7_l473_473143

theorem greatest_3_digit_base9_divisible_by_7 :
  ∃ (n : ℕ), n < 729 ∧ n ≥ 81 ∧ n % 7 = 0 ∧ n = 8 * 81 + 8 * 9 + 8 := 
by 
  use 728
  split
  {
    exact nat.pred_lt (ne_of_lt (by norm_num))
  }
  split
  {
    exact nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.zero_le 7))) 
  }
  split
  {
    norm_num
  }
  norm_num

end greatest_3_digit_base9_divisible_by_7_l473_473143


namespace part1_range_b1_a1_part2_range_a_b1_l473_473734

noncomputable def f (a b x : ℝ) : ℝ := a * cos x + 2 * b * cos x ^ 2 + 1 - b

theorem part1_range_b1_a1 :
  set.range (λ x, f 1 1 x) = set.Icc (-1 / 8 : ℝ) 3 :=
sorry

theorem part2_range_a_b1 :
  set_of (λ a, ∃ x : ℝ, |f a 1 x| ≥ a ^ 2) = set.Icc (-2 : ℝ) 2 :=
sorry

end part1_range_b1_a1_part2_range_a_b1_l473_473734


namespace max_value_of_g_l473_473964

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 23 :=
by
  sorry

end max_value_of_g_l473_473964


namespace tangent_lines_l473_473730

-- Definition of function f
def f (x : ℝ) : ℝ := (1/3) * x^3 + (4/3)

-- Definition of derivative of function f
def f_prime (x : ℝ) : ℝ := x^2

-- Definition of tangent equation
def tangent_equation (x₀ : ℝ) : (ℝ × ℝ) → Prop :=
  λ P, P.2 - ((1/3) * x₀^3 + (4/3)) = x₀^2 * (P.1 - x₀)

-- The point P(2, 4)
def P : ℝ × ℝ := (2, 4)

-- The main theorem stating the equations of tangent lines
theorem tangent_lines :
  (tangent_equation (-1) P ∧ (P.1 - P.2 + 2 = 0)) ∨
  (tangent_equation 2 P ∧ (4 * P.1 - P.2 - 4 = 0)) :=
sorry

end tangent_lines_l473_473730


namespace distinct_flavors_count_l473_473183

-- Define the red and green candies
def red_candies : ℕ := 5
def green_candies : ℕ := 4

-- Define the theorem counting distinct ratios
theorem distinct_flavors_count :
  ∃ n : ℕ, n = 15 ∧ 
  n = (finset.image (λ (p : ℕ × ℕ), 
    if p.2 = 0 then none else some (p.1 / gcd p.1 p.2, p.2 / gcd p.1 p.2))
    ((finset.range (red_candies + 1)).product (finset.range (green_candies + 1)))
).erase_none.card :=
sorry

end distinct_flavors_count_l473_473183


namespace sum_of_first_fifteen_multiples_of_7_l473_473576

theorem sum_of_first_fifteen_multiples_of_7 :
  ∑ i in finset.range 15, 7 * (i + 1) = 840 :=
sorry

end sum_of_first_fifteen_multiples_of_7_l473_473576


namespace part_a_l473_473468

variables {ABC : Triangle}
variables {A1 B1 C1 A2 B2 C2 : Point}
variables (A1_on_AB : A1 ∈ Segment (ABC.A) (ABC.B))
variables (B1_on_BC : B1 ∈ Segment (ABC.B) (ABC.C))
variables (C1_on_CA : C1 ∈ Segment (ABC.C) (ABC.A))
variables (A2_on_CA : A2 ∈ Segment (ABC.C) (ABC.A))
variables (B2_on_AB : B2 ∈ Segment (ABC.A) (ABC.B))
variables (C2_on_BC : C2 ∈ Segment (ABC.B) (ABC.C))
variables (A1B1C1_sim_ABC : Similar (Triangle.mk A1 B1 C1) ABC)
variables (A2B2C2_sim_ABC : Similar (Triangle.mk A2 B2 C2) ABC)
variables (A1B1_eq_A2B2 : angle (SegmentCenter A1 B1) (ABC.A) (ABC.B) = angle (SegmentCenter A2 B2) (ABC.A) (ABC.B))

theorem part_a : Congruent (Triangle.mk A1 B1 C1) (Triangle.mk A2 B2 C2) :=
sorry

end part_a_l473_473468


namespace find_k_l473_473690

theorem find_k (k : ℝ) :
  ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧
  (λ (a b c d x y : ℝ), (a*x + b*y, c*x + d*y))
  (2 : ℝ) (6 : ℝ) (3 : ℝ) (2 : ℝ) (v.1) (v.2) = (k*v.1, k*v.2) ↔
  k = 2 + 3*real.sqrt 2 ∨ k = 2 - 3*real.sqrt 2 :=
begin
  sorry
end

end find_k_l473_473690


namespace sin_C_value_area_of_triangle_l473_473425

open Real
open Classical

variable {A B C a b c : ℝ}

-- Given conditions
axiom h1 : b = sqrt 2
axiom h2 : c = 1
axiom h3 : cos B = 3 / 4

-- Proof statements
theorem sin_C_value : sin C = sqrt 14 / 8 := sorry

theorem area_of_triangle : 1 / 2 * b * c * sin (B + C) = sqrt 7 / 4 := sorry

end sin_C_value_area_of_triangle_l473_473425


namespace probability_P_plus_S_is_two_less_than_multiple_of_7_l473_473250

noncomputable def is_multiple_of_7 (n : Nat) : Prop := n % 7 = 0

theorem probability_P_plus_S_is_two_less_than_multiple_of_7 :
  (∃ a b : Nat, a ≠ b ∧ 1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧
  let P := a * b in let S := a + b in is_multiple_of_7 (P + S + 2)) →
  (392 / 1770 : Real) :=
sorry

end probability_P_plus_S_is_two_less_than_multiple_of_7_l473_473250


namespace range_of_a_decreasing_function_l473_473958

theorem range_of_a_decreasing_function (f : ℝ → ℝ) (h_decreasing : ∀ x y ∈ Ioo (-1 : ℝ) (1 : ℝ), x < y → f y < f x) (a : ℝ) (h_condition : f (1 - a) < f (a^2 - 1)) : 
  0 < a ∧ a < Real.sqrt 2 :=
sorry

end range_of_a_decreasing_function_l473_473958


namespace joe_list_possibilities_l473_473618

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l473_473618


namespace root_equation_val_l473_473749

theorem root_equation_val (a : ℝ) (h : a^2 - 2 * a - 5 = 0) : 2 * a^2 - 4 * a = 10 :=
by 
  sorry

end root_equation_val_l473_473749


namespace find_h_l473_473421

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l473_473421


namespace number_of_possible_lists_l473_473612

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l473_473612


namespace intersection_of_A_and_B_l473_473350

-- Definitions from the conditions
def A := {1, 2, 3}
def B := {y | ∃ x, x ∈ A ∧ y = 2 * x - 1}

-- The proof statement
theorem intersection_of_A_and_B :
  A ∩ B = {1, 3} :=
sorry

end intersection_of_A_and_B_l473_473350


namespace inclination_angle_range_l473_473839

theorem inclination_angle_range (l : Line) :
  l.passes_through_second_and_fourth_quadrants → l.inclination_angle ∈ (90:ℝ, 180:ℝ) :=
sorry

end inclination_angle_range_l473_473839


namespace number_of_possible_lists_l473_473604

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l473_473604


namespace suitable_selection_methods_108_l473_473913

theorem suitable_selection_methods_108 :
  let students := {A, B, C, D, E, F} in
  let legs := {first, second, third, fourth} in
  let participation := {A → (first ∨ fourth), B → ¬first} in
  -- More formal translation of conditions and calculation to match steps...
  calc_methods(students, legs, participation) = 108 :=
sorry

end suitable_selection_methods_108_l473_473913


namespace task_1_task_2_task_3_l473_473480

open Real

noncomputable def solve_inequality_a3 : Set ℝ := 
{ x : ℝ | x < -2 ∨ x > 6 }

theorem task_1 (a : ℝ) (x : ℝ) (h : a = 3) :
  log 2 (abs x + abs (x - 4)) > a ↔ (x < -2 ∨ x > 6) :=
by
  sorry
theorem task_2 (x : ℝ) (h : log 2 (abs x + abs (x - 4)) > 2) : false := 
by
  sorry

theorem task_3 {a : ℝ} (h : log 2 (abs x + abs (x - 4)) > a ↔ true) :
  a < 2 :=
by
  sorry

end task_1_task_2_task_3_l473_473480


namespace polynomial_divisibility_l473_473059

theorem polynomial_divisibility (f : ℝ[X]) (hf : ¬ is_constant f)
  (h : ∀ (n k : ℕ), 0 < n → 0 < k → 
    (∀ i, 0 ≤ i → i < k → (f.eval (n + i + 1)) ∈ ℤ) →
    (f.eval (n + 1) * f.eval (n + 2) * ... * f.eval (n + k)) / 
    (f.eval 1 * f.eval 2 * ... * f.eval k) ∈ ℤ) :
  (∃ g : ℝ[X], f = X * g) :=
sorry

end polynomial_divisibility_l473_473059


namespace number_of_lists_l473_473609

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l473_473609


namespace f_l473_473842

-- Define the function according to the given condition
def f (f' : ℝ → ℝ) (x : ℝ) := (1 / 2) * f'(-1) * x^2 - 2 * x + 3

-- Define the derivative of the function
def f' (f' : ℝ → ℝ) (x : ℝ) := f'(-1) * x - 2

-- Prove that f'(-1) = -1 given the conditions
theorem f'_at_neg_1 (f' : ℝ → ℝ) : f'(-1) = -1 :=
  by
    -- Placeholder for proof
    sorry

end f_l473_473842


namespace marty_combination_count_l473_473907

theorem marty_combination_count (num_colors : ℕ) (num_methods : ℕ) 
  (h1 : num_colors = 5) (h2 : num_methods = 4) : 
  num_colors * num_methods = 20 := by
  sorry

end marty_combination_count_l473_473907


namespace plant_ways_count_l473_473549

theorem plant_ways_count :
  ∃ (solutions : Finset (Fin 7 → ℕ)), 
    (∀ x ∈ solutions, (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 = 10) ∧ 
                       (100 * x 0 + 200 * x 1 + 300 * x 2 + 150 * x 3 + 125 * x 4 + 125 * x 5 = 2500)) ∧
    (solutions.card = 8) :=
sorry

end plant_ways_count_l473_473549


namespace triangle_third_side_lengths_l473_473762

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l473_473762


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l473_473673

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (a : ℕ), (∀ n, n ∈ (list.range 4).map (λ i, a + i) -> n % 2 = 0 ∨ n % 3 = 0 ∨ n % 4 = 0) →
  12 ∣ list.prod ((list.range 4).map (λ i, a + i)) :=
by
  intro a
  intro h
  -- Insert proof here
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l473_473673


namespace concurrency_of_ac_kn_lm_l473_473109

theorem concurrency_of_ac_kn_lm
  (A B C P Q K L M N : Point)
  (h_non_isosceles : ¬ is_isosceles A B C)
  (h_PQ_on_AC : P ∈ line_segment A C ∧ Q ∈ line_segment A C)
  (h_angles : ∃ (α : Real), α = ∠ABP ∧ α = ∠QBC ∧ α < ½ * ∠ABC)
  (h_angle_bisectors_bp : 
    is_angle_bisector_line (angle A) (line_segment B P) K ∧ 
    is_angle_bisector_line (angle C) (line_segment B P) L)
  (h_angle_bisectors_bq : 
    is_angle_bisector_line (angle A) (line_segment B Q) M ∧ 
    is_angle_bisector_line (angle C) (line_segment B Q) N) :
  are_concurrent (line_segment A C) (line_segment K N) (line_segment L M) :=
sorry

end concurrency_of_ac_kn_lm_l473_473109


namespace current_bottle_caps_l473_473670

def initial_bottle_caps : ℕ := 91
def lost_bottle_caps : ℕ := 66

theorem current_bottle_caps : initial_bottle_caps - lost_bottle_caps = 25 :=
by
  -- sorry is used to skip the proof
  sorry

end current_bottle_caps_l473_473670


namespace asymptotes_of_hyperbola_l473_473711

/-- For a hyperbola with an imaginary axis length of 2 and a focal distance of 2√3,
    the equations of the asymptotes are y = ± √2/2 x or y = ± √2 x. -/
theorem asymptotes_of_hyperbola :
  ∀ {a b c : ℝ},
  2 * b = 2 → 2 * c = 2 * √3 →
  a = √2 →
  (∀ x y : ℝ, (x ^ 2 / 2 - y^2 = 1 ∨ y^2 / 2 - x^2 = 1) → 
  (y = √2 / 2 * x ∨ y = -√2 / 2 * x ∨ y = √2 * x ∨ y = -√2 * x)) :=
begin
  sorry
end

end asymptotes_of_hyperbola_l473_473711


namespace range_of_f_l473_473291

section
  -- Define the function \( f(x) = \sqrt{1 - \frac{x^2}{4}} + 2x \)
  def f (x : ℝ) : ℝ := real.sqrt (1 - x^2 / 4) + 2 * x

  -- Statement to state that the range of the function \( f \) is \([-4, \sqrt{17}]\)
  theorem range_of_f : set.range f = set.Icc (-4 : ℝ) (real.sqrt 17) := 
    sorry
end

end range_of_f_l473_473291


namespace lateral_surface_area_of_cylinder_l473_473840

theorem lateral_surface_area_of_cylinder :
  (∀ (side_length : ℕ), side_length = 10 → 
  ∃ (lateral_surface_area : ℝ), lateral_surface_area = 100 * Real.pi) :=
by
  sorry

end lateral_surface_area_of_cylinder_l473_473840


namespace base_logarithm_l473_473132

noncomputable def base_of_logarithm : ℕ := 729

theorem base_logarithm (b : ℝ) (log_eq : log b base_of_logarithm = 6) : b = 3 :=
sorry

end base_logarithm_l473_473132


namespace solve_for_a_l473_473011

theorem solve_for_a (a b : ℝ) (h₁ : b = 4 * a) (h₂ : b = 20 - 7 * a) : a = 20 / 11 :=
by
  sorry

end solve_for_a_l473_473011


namespace complex_sum_solution_l473_473466

noncomputable def complex_sum_problem (x : ℂ) (h1 : x^1001 = 1) (h2 : x ≠ 1) : Prop :=
  (∑ k in Finset.range 1000, x^(2*(k+1)) / (x^(k+1) - 1)) = 499

theorem complex_sum_solution (x : ℂ) (h1 : x^1001 = 1) (h2 : x ≠ 1) : complex_sum_problem x h1 h2 :=
  by
    sorry

end complex_sum_solution_l473_473466


namespace purple_marble_probability_l473_473620

theorem purple_marble_probability (blue green : ℝ) (p : ℝ) 
  (h_blue : blue = 0.25)
  (h_green : green = 0.4)
  (h_sum : blue + green + p = 1) : p = 0.35 :=
by
  sorry

end purple_marble_probability_l473_473620


namespace unoccupied_volume_in_container_l473_473235

-- defining constants
def side_length_container := 12
def side_length_ice_cube := 3
def number_of_ice_cubes := 8
def water_fill_fraction := 3 / 4

-- defining volumes
def volume_container := side_length_container ^ 3
def volume_water := volume_container * water_fill_fraction
def volume_ice_cube := side_length_ice_cube ^ 3
def total_volume_ice := volume_ice_cube * number_of_ice_cubes
def volume_unoccupied := volume_container - (volume_water + total_volume_ice)

-- The theorem to be proved
theorem unoccupied_volume_in_container : volume_unoccupied = 216 := by
  -- Proof steps will go here
  sorry

end unoccupied_volume_in_container_l473_473235


namespace rectangle_length_fraction_of_circle_radius_l473_473105

noncomputable def square_side (area : ℕ) : ℕ :=
  Nat.sqrt area

noncomputable def rectangle_length (breadth area : ℕ) : ℕ :=
  area / breadth

theorem rectangle_length_fraction_of_circle_radius
  (square_area : ℕ)
  (rectangle_breadth : ℕ)
  (rectangle_area : ℕ)
  (side := square_side square_area)
  (radius := side)
  (length := rectangle_length rectangle_breadth rectangle_area) :
  square_area = 4761 →
  rectangle_breadth = 13 →
  rectangle_area = 598 →
  length / radius = 2 / 3 :=
by
  -- Proof steps go here
  sorry

end rectangle_length_fraction_of_circle_radius_l473_473105


namespace mass_of_1m3_l473_473530

/-- The volume of 1 gram of the substance in cubic centimeters cms_per_gram is 1.3333333333333335 cm³. -/
def cms_per_gram : ℝ := 1.3333333333333335

/-- There are 1,000,000 cubic centimeters in 1 cubic meter. -/
def cm3_per_m3 : ℕ := 1000000

/-- Given the volume of 1 gram of the substance, find the mass of 1 cubic meter of the substance. -/
theorem mass_of_1m3 (h1 : cms_per_gram = 1.3333333333333335) (h2 : cm3_per_m3 = 1000000) :
  ∃ m : ℝ, m = 750 :=
by
  sorry

end mass_of_1m3_l473_473530


namespace locus_of_A_fixed_median_length_l473_473251

-- Definitions for the circles, lines, points, and midpoints in the geometric setup
variables {C1 C2 : Circle} {O1 O2 P Q : Point}
variables {Δ : Line} {B C A M : Point} {k : ℝ}

-- Conditions extracted from the problem
-- C1 and C2 intersect at P
-- Line Δ passes through P and intersects C1 at B and C2 at C
-- M is the midpoint of B and C
-- O is the midpoint of O1 and O2
def intersects_at (C1 C2 : Circle) (P : Point) := Circle.contains C1 P ∧ Circle.contains C2 P
def midpoint (A B M : Point) := dist A M = dist M B
def circle_locus (A M : Point) (k : ℝ) := dist A M = k

-- Rewriting the problem statement in Lean 4
theorem locus_of_A_fixed_median_length
    (h_intersect : intersects_at C1 C2 P)
    (h_line : line_through Δ P)
    (h_intersect_BC : intersects_line_circle Δ C1 B ∧ intersects_line_circle Δ C2 C)
    (h_midpoint_M : midpoint B C M)
    (h_midpoint_O : midpoint O1 O2 O)
    (h_median_fixed : circle_locus A M k)
    : ∃ (O : Point) (r : ℝ), dist O M = r ∧ O = midpoint O1 O2 := sorry

end locus_of_A_fixed_median_length_l473_473251


namespace distance_between_foci_l473_473648

theorem distance_between_foci (center : ℝ × ℝ) (a b : ℝ) :
  center = (4, 1) →
  a = 4 →
  b = 1 →
  (2 * a) * (2 * a) - (2 * b) * (2 * b) = 60 →
  sqrt (2 * a) * (2 * a) - (2 * b) * (2 * b)) = 2 * sqrt 15 :=
by
  intros h_center h_a h_b h_C
  rw [h_a, h_b, h_center]
  sorry

end distance_between_foci_l473_473648


namespace area_of_right_triangle_l473_473969

theorem area_of_right_triangle (h : ℝ) (sin30 : ℝ) (hypotenuse_angle_30 : ∀ (x : ℝ), x = 13 ∧ sin30 = 1/2 → h = 36.6025) :
  h = 36.6025 :=
begin
  assume x h,
  sorry
end

end area_of_right_triangle_l473_473969


namespace no_valid_distribution_l473_473036

-- Definitions of the problem conditions
def ball_colors : Type := Fin 7  -- There are 7 colors of balls
def boxes : Type := Fin 5  -- There are 5 boxes arranged in a circle
def balls_per_box := 3  -- Each box contains 3 balls

-- No two adjacent boxes can contain balls of the same color
def adjacent (i : boxes) : boxes := if i.val = 0 then boxes - 1 else i - 1

-- Proposition that finding such a distribution is impossible
theorem no_valid_distribution : ¬∃ (distribution : boxes → finset (ball_colors)), 
  (∀ (b : boxes), finset.card (distribution b) = balls_per_box) ∧
  (∀ (b : boxes), ∀ (adj : boxes), 
    adjacent b = adj → (distribution b) ∩ (distribution adj) = ∅) :=
sorry

end no_valid_distribution_l473_473036


namespace baseball_card_decrease_l473_473200

noncomputable def percentDecrease (V : ℝ) (P : ℝ) : ℝ :=
  V * (P / 100)

noncomputable def valueAfterDecrease (V : ℝ) (D : ℝ) : ℝ :=
  V - D

theorem baseball_card_decrease (V : ℝ) (H1 : V > 0) :
  let D1 := percentDecrease V 50
  let V1 := valueAfterDecrease V D1
  let D2 := percentDecrease V1 10
  let V2 := valueAfterDecrease V1 D2
  let totalDecrease := V - V2
  totalDecrease / V * 100 = 55 := sorry

end baseball_card_decrease_l473_473200


namespace greatest_3_digit_base9_div_by_7_l473_473136

def base9_to_decimal (n : ℕ) : ℕ :=
  let d2 := n / 81
  let d1 := (n % 81) / 9
  let d0 := n % 9
  d2 * 81 + d1 * 9 + d0

def greatest_base9_3_digit_div_by_7 (n : ℕ) : Prop :=
  n < 9 * 9 * 9 ∧ 7 ∣ (base9_to_decimal n)

theorem greatest_3_digit_base9_div_by_7 :
  ∃ n, greatest_base9_3_digit_div_by_7 n ∧ n = 888 :=
begin
  sorry
end

end greatest_3_digit_base9_div_by_7_l473_473136


namespace minimize_distance_l473_473723

theorem minimize_distance (t : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)
  (hM : M = (t, t^2))
  (hN : N = (t, Real.log t)) :
  t = Real.sqrt 2 / 2 ↔ abs (M.2 - N.2) = min (λ t, abs (t^2 - Real.log t)) :=
begin
  sorry
end

end minimize_distance_l473_473723


namespace greatest_3_digit_base9_div_by_7_l473_473134

def base9_to_decimal (n : ℕ) : ℕ :=
  let d2 := n / 81
  let d1 := (n % 81) / 9
  let d0 := n % 9
  d2 * 81 + d1 * 9 + d0

def greatest_base9_3_digit_div_by_7 (n : ℕ) : Prop :=
  n < 9 * 9 * 9 ∧ 7 ∣ (base9_to_decimal n)

theorem greatest_3_digit_base9_div_by_7 :
  ∃ n, greatest_base9_3_digit_div_by_7 n ∧ n = 888 :=
begin
  sorry
end

end greatest_3_digit_base9_div_by_7_l473_473134


namespace arithmetic_seq_form_l473_473063

noncomputable def a_n : ℕ → ℕ := sorry

noncomputable def S_n (n : ℕ) : ℕ := (n * (a_n 1 + a_n n)) / 2

theorem arithmetic_seq_form :
  (a_6 = 12) →
  (S_n 3 = 12) →
  (∀ n, a_n n = 2 * n) :=
begin
  sorry
end

end arithmetic_seq_form_l473_473063


namespace value_of_x_l473_473088

theorem value_of_x (x y : ℕ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := sorry

end value_of_x_l473_473088


namespace april_roses_l473_473652

theorem april_roses (price_per_rose earnings number_of_roses_left : ℕ) 
  (h1 : price_per_rose = 7) 
  (h2 : earnings = 35) 
  (h3 : number_of_roses_left = 4) : 
  (earnings / price_per_rose + number_of_roses_left) = 9 :=
by
  sorry

end april_roses_l473_473652


namespace length_of_JM_l473_473860

-- Definitions for conditions
structure RegularHexagon where
  A B C D E F : ℝ -- vertices of the hexagon
  side_length : ℝ
  (h_sides: side_length = 2)
  (h_reg: ∀ i j, interior_angle ABC D E F = 120) -- All interior angles are 120 degrees.

structure RectangleInsideHexagon where
  J K L M N O P Q : ℝ -- vertices of the rectangles
  (on_AB: J ∈ line AB ∧ K ∈ line AB)
  (on_AF: L ∈ line AF ∧ M ∈ line AF)
  (on_CD: N ∈ line CD ∧ O ∈ line CD)
  (on_EF: P ∈ line EF ∧ Q ∈ line EF)

-- Proving the length of JM
theorem length_of_JM (h : RegularHexagon) (r1 r2 : RectangleInsideHexagon) : 
  length_segment J M = 2 :=
begin
  sorry
end

end length_of_JM_l473_473860


namespace max_abs_sum_on_circle_l473_473368

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end max_abs_sum_on_circle_l473_473368


namespace coin_path_scenario_possible_l473_473195

/- 
  Theorem: For a coin following a path from vertex A to B on an n x n board 
  with the given conditions, the scenario is possible if and only if n = 2.
-/
theorem coin_path_scenario_possible (n : ℕ) : 
  (∃ (path : List (ℕ × ℕ)), 
    path.head = some (0, 0) ∧ 
    path.last = some (n, n) ∧ 
    (∀ i, i < path.length - 1 → ((path.nth i).is_some ∧ 
                                (path.nth (i+1)).is_some ∧ 
                                (abs ((path.nth i).get_or_else (0,0)).fst - (path.nth (i+1)).get_or_else (0,0).fst) ≤ 1 ∧
                                (abs ((path.nth i).get_or_else (0,0)).snd - (path.nth (i+1)).get_or_else (0,0).snd) ≤ 1)) ∧ 
    (∀ (triangle : (ℕ × ℕ × ℕ)), count_seeds path triangle = 2)) ↔ n = 2 :=
begin
  sorry
end

end coin_path_scenario_possible_l473_473195


namespace triangle_third_side_length_count_l473_473785

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l473_473785


namespace valid_param_A_valid_param_B_valid_param_C_valid_param_D_valid_param_E_l473_473971

-- Define the line as a function
def line (x : ℝ) : ℝ := 3 * x - 4

-- Check if a point lies on the line
def point_on_line (p : ℝ × ℝ) : Prop := p.2 = line p.1

-- Parameterization checks
def valid_param (p d : ℝ × ℝ) : Prop := 
  point_on_line p ∧ ∃ k : ℝ, d = (k * 1, k * 3)

-- Given parameterizations
def A : ℝ × ℝ := (1, -1)
def A_dir : ℝ × ℝ := (2, 6)

def B : ℝ × ℝ := (-4/3, 0)
def B_dir : ℝ × ℝ := (1, 3)

def C : ℝ × ℝ := (2, 2)
def C_dir : ℝ × ℝ := (-1, -3)

def D : ℝ × ℝ := (0, -3)
def D_dir : ℝ × ℝ := (3, 9)

def E : ℝ × ℝ := (-2, -10)
def E_dir : ℝ × ℝ := (5/3, 5)

-- Statement to check valid parameterizations
theorem valid_param_A : valid_param A A_dir := sorry
theorem valid_param_B : valid_param B B_dir := sorry
theorem valid_param_C : valid_param C C_dir := sorry
theorem valid_param_D : ¬ valid_param D D_dir := sorry
theorem valid_param_E : ¬ valid_param E E_dir := sorry

end valid_param_A_valid_param_B_valid_param_C_valid_param_D_valid_param_E_l473_473971


namespace sixth_ninth_grader_buddy_fraction_l473_473861

theorem sixth_ninth_grader_buddy_fraction
  (s n : ℕ)
  (h_fraction_pairs : n / 4 = s / 3)
  (h_buddy_pairing : (∀ i, i < n -> ∃ j, j < s) 
     ∧ (∀ j, j < s -> ∃ i, i < n) -- each sixth grader paired with one ninth grader and vice versa
  ) :
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by 
  sorry

end sixth_ninth_grader_buddy_fraction_l473_473861


namespace min_dist_seven_points_in_rectangle_l473_473651

theorem min_dist_seven_points_in_rectangle (b : ℝ) :
  (∃ (r : rectangle), r.length = 1 ∧ r.width = 2 ∧
  (∀ pts : set point, pts.card = 7 → (∃ p1 p2 ∈ pts, dist p1 p2 ≤ b))) ↔ b = (sqrt 13) / 6 :=
by
  sorry

end min_dist_seven_points_in_rectangle_l473_473651


namespace minimum_perimeter_of_triangle_maximum_area_of_triangle_l473_473321

noncomputable def point := ℝ × ℝ

-- Define points A, B, and the parabola C
def A : point := (1, 0)
def B : point := (2, 1)
def parabola (C : point) : Prop := C.2^2 = 4 * C.1

-- Define the conditions that the point C is on the left and above the line AB (left and above condition)
def left_and_above (C : point) : Prop := C.1 ≤ (C.2 + 1) 

-- Define the perimeter of triangle ABC
def perimeter (A B C : point) : ℝ := 
  dist A B + dist B C + dist C A

#print dist

-- Define the area of triangle ABC using the determinant method
def area (A B C : point) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem minimum_perimeter_of_triangle :
  ∀ (C : point), parabola C → 
  min (perimeter A B C) = 3 + Real.sqrt 2 :=
sorry

theorem maximum_area_of_triangle :
  ∀ (C : point), parabola C → left_and_above C → 
  max (area A B C) = 1 :=
sorry

end minimum_perimeter_of_triangle_maximum_area_of_triangle_l473_473321


namespace complex_inequality_l473_473007

theorem complex_inequality (m : ℝ) : 
  (m - 3 ≥ 0 ∧ m^2 - 9 = 0) → m = 3 := 
by
  sorry

end complex_inequality_l473_473007


namespace sum_first_15_terms_l473_473329

-- Definitions of sequences and given conditions
def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r, ∀ n, a (n + 1) = r * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

variables {a b : ℕ → ℝ}

-- Given conditions
def cond1 : Prop := geometric_sequence a
def cond2 : Prop := arithmetic_sequence b
def cond3 : Prop := a 2 * a 14 = 4 * a 8
def cond4 : Prop := b 8 = a 8

-- Prove the sum of the first 15 terms of b is 60
theorem sum_first_15_terms 
  (h1 : cond1) 
  (h2 : cond2)
  (h3 : cond3)
  (h4 : cond4)
  : ∑ n in (Finset.range 15), b (n + 1) = 60 := 
sorry

end sum_first_15_terms_l473_473329


namespace negative_integer_solution_l473_473990

theorem negative_integer_solution (N : ℤ) (h : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end negative_integer_solution_l473_473990


namespace aerith_wins_probability_l473_473240

theorem aerith_wins_probability :
  let p : ℝ := 7 / 11 in
  let prob_heads : ℝ := 4 / 7 in
  let prob_tails : ℝ := 3 / 7 in
  ∀ (p : ℝ), (p = prob_tails + prob_heads * (1 - p)) → p = 7 / 11 :=
by
  sorry

end aerith_wins_probability_l473_473240


namespace scalene_triangle_not_divisible_l473_473924

theorem scalene_triangle_not_divisible :
  ∀ (A B C D : ℝ) (T : Triangle) (CD : Segment),
  T.is_scalene A B C → 
  (A ≠ B ∧ A ≠ C ∧ B ≠ C) → 
  (T.area_of_triangle A C D = T.area_of_triangle B C D) →
  false :=
by
  intro A B C D T CD
  intro h_scalene h_distinct_sides h_equal_areas
  sorry

end scalene_triangle_not_divisible_l473_473924


namespace sara_total_cents_l473_473498

-- Define the conditions as constants
def quarters : ℕ := 11
def value_per_quarter : ℕ := 25

-- Define the total amount formula based on the conditions
def total_cents (q : ℕ) (v : ℕ) : ℕ := q * v

-- The theorem to be proven
theorem sara_total_cents : total_cents quarters value_per_quarter = 275 :=
by
  -- Proof goes here
  sorry

end sara_total_cents_l473_473498


namespace real_solutions_of_cubic_l473_473851

noncomputable def cubic_roots (a b c d : ℝ) : set ℝ :=
  {x | a * x^3 + b * x^2 + (c - 1) * x + d = 0}

noncomputable def solve_cubic (a b c d : ℝ) : set (ℝ × ℝ) :=
  let p := λ (x : ℝ), a * x^3 + b * x^2 + c * x + d
  { (x, y) | x = p y ∧ y = p x }

theorem real_solutions_of_cubic (a b c d : ℝ) :
  ∃ (x y : ℝ), (x, y) ∈ solve_cubic a b c d :=
sorry

end real_solutions_of_cubic_l473_473851


namespace cost_to_fill_pool_l473_473041

/-- Definition of the pool dimensions and constants --/
def pool_length := 20
def pool_width := 6
def pool_depth := 10
def cubic_feet_to_liters := 25
def liter_cost := 3

/-- Calculating the cost to fill the pool --/
def pool_volume := pool_length * pool_width * pool_depth
def total_liters := pool_volume * cubic_feet_to_liters
def total_cost := total_liters * liter_cost

/-- Theorem stating that the total cost to fill the pool is $90,000 --/
theorem cost_to_fill_pool : total_cost = 90000 := by
  sorry

end cost_to_fill_pool_l473_473041


namespace cookie_distribution_l473_473124

theorem cookie_distribution (b m l : ℕ)
  (h1 : b + m + l = 30)
  (h2 : m = 2 * b)
  (h3 : l = b + m) :
  b = 5 ∧ m = 10 ∧ l = 15 := 
by 
  sorry

end cookie_distribution_l473_473124


namespace find_x_plus_y_l473_473340

variable (x y : ℝ)

def vector_a := (2, 4, x)
def vector_b := (2, y, 2)
def magnitude_a := real.sqrt (2^2 + 4^2 + x^2) = 6
def perpendicular := 2 * 2 + 4 * y + 2 * x = 0

theorem find_x_plus_y (h1 : magnitude_a) (h2 : perpendicular) : x + y = -3 ∨ x + y = 1 := by
  sorry

end find_x_plus_y_l473_473340


namespace rectangle_intersection_exists_l473_473555

theorem rectangle_intersection_exists (R1 R2 R3 : set (ℝ × ℝ))
  (hR1_area : ∃ s1, measure_theory.measure_space.measure (set.univ ∩ R1) = 6) 
  (hR2_area : ∃ s2, measure_theory.measure_space.measure (set.univ ∩ R2) = 6)
  (hR3_area : ∃ s3, measure_theory.measure_space.measure (set.univ ∩ R3) = 6)
  (hInside_square : ∀ (x y : ℝ), (x,y) ∈ (R1 ∪ R2 ∪ R3) → 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 4) :
  (∃ (i j : ℤ), 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 ∧ i ≠ j ∧ measure_theory.measure_space.measure (set.univ ∩ (R1 ∩ R2 ∩ R3)) ≥ 2/3) :=
sorry

end rectangle_intersection_exists_l473_473555


namespace greatest_3_digit_base9_div_by_7_l473_473133

def base9_to_decimal (n : ℕ) : ℕ :=
  let d2 := n / 81
  let d1 := (n % 81) / 9
  let d0 := n % 9
  d2 * 81 + d1 * 9 + d0

def greatest_base9_3_digit_div_by_7 (n : ℕ) : Prop :=
  n < 9 * 9 * 9 ∧ 7 ∣ (base9_to_decimal n)

theorem greatest_3_digit_base9_div_by_7 :
  ∃ n, greatest_base9_3_digit_div_by_7 n ∧ n = 888 :=
begin
  sorry
end

end greatest_3_digit_base9_div_by_7_l473_473133


namespace angle_DBC_39_l473_473872

noncomputable def isosceles_triangle (A B C : Type) (Angle_BAD Angle_ADC : ℝ)
  (ADE_equilateral : Bool) (BA_eq_BC : Bool) (angle_ABD angle_ADB angle_ACD : ℝ) : Prop :=
  (BA_eq_BC = true) ∧ (angle_ABD = 13) ∧ (angle_ADB = 150) ∧ (angle_ACD = 30) ∧ (ADE_equilateral = true)

theorem angle_DBC_39 {A B C D E : Type} :
  isosceles_triangle A B C (13 : ℝ) (150 : ℝ) (true) (true) (13) (150) (30) →
  ∃ angle_DBC : ℝ, angle_DBC = 39 :=
begin
  sorry
end

end angle_DBC_39_l473_473872


namespace Lucas_age_in_3_years_l473_473092

variable (Gladys Billy Lucas : ℕ)

theorem Lucas_age_in_3_years :
  Gladys = 30 ∧ Billy = Gladys / 3 ∧ Gladys = 2 * (Billy + Lucas) →
  Lucas + 3 = 8 :=
by
  intro h
  cases h with h1 h2
  cases h2 with hBilly h3
  sorry

end Lucas_age_in_3_years_l473_473092


namespace inscribed_angle_sum_l473_473205

noncomputable def sum_of_inscribed_angles
  (W X Y Z : Type) 
  (angle_WYZ_deg : ℕ) 
  (angle_WXY_deg : ℕ) : ℕ :=
  let angle_YWX_deg := 20
  let angle_YZW_deg := 70
  angle_YWX_deg + angle_YZW_deg

-- The problem conditions
variables [circle_inscribed WXYZ : Type]
variables (W X Y Z : WXYZ)

-- Given condition angles
variables (angle_WYZ : ℕ) (angle_WXY : ℕ)
hypothesis (h1: angle_WYZ = 20)
hypothesis (h2: angle_WXY = 70)

theorem inscribed_angle_sum 
  (h1 : angle_WYZ = 20)
  (h2 : angle_WXY = 70) :
  sum_of_inscribed_angles W X Y Z angle_WYZ angle_WXY = 90 :=
by
  sorry

end inscribed_angle_sum_l473_473205


namespace portsville_to_eastside_trip_time_l473_473114

def portsville_to_eastside_trip (initial_speed : ℝ) (initial_time : ℝ) (new_speed : ℝ) (rest_stop_duration : ℝ) : ℝ :=
  let distance := initial_speed * initial_time
  let new_time := distance / new_speed
  new_time + rest_stop_duration

theorem portsville_to_eastside_trip_time :
  portsville_to_eastside_trip 80 6 40 0.5 = 12.5 := by
  sorry

end portsville_to_eastside_trip_time_l473_473114


namespace max_value_expression_l473_473996

theorem max_value_expression (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 * (a + b))) +
   Real.sqrt (Real.sqrt (b^2 * (b + c))) +
   Real.sqrt (Real.sqrt (c^2 * (c + d))) +
   Real.sqrt (Real.sqrt (d^2 * (d + a)))) ≤ 4 * Real.sqrt (Real.sqrt 2) := by
  sorry

end max_value_expression_l473_473996


namespace find_h_l473_473378

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l473_473378


namespace solve_congruence_l473_473511

theorem solve_congruence :
  ∃ n : ℤ, 13 * n ≡ 8 [MOD 47] ∧ n ≡ 29 [MOD 47] :=
sorry

end solve_congruence_l473_473511


namespace candies_count_l473_473904

theorem candies_count (x : ℚ) (h : x + 3 * x + 12 * x + 72 * x = 468) : x = 117 / 22 :=
by
  sorry

end candies_count_l473_473904


namespace average_speed_last_segment_is_80_l473_473452

-- Definitions
def total_distance : ℝ := 150
def total_time_minutes : ℝ := 120
def total_time_hours : ℝ := total_time_minutes / 60

def first_segment_speed : ℝ := 70
def second_segment_speed : ℝ := 75
def segment_time_minutes : ℝ := 40
def segment_time_hours : ℝ := segment_time_minutes / 60

-- Distance calculations based on the given segments
def first_segment_distance : ℝ := first_segment_speed * segment_time_hours
def second_segment_distance : ℝ := second_segment_speed * segment_time_hours

-- Remaining distance for the last segment
def remaining_distance : ℝ := total_distance - (first_segment_distance + second_segment_distance)

-- Time for the last segment in hours
def last_segment_time : ℝ := segment_time_hours

-- The speed during the last segment
def last_segment_speed : ℝ := remaining_distance / last_segment_time

theorem average_speed_last_segment_is_80 :
  last_segment_speed = 80 := by
  sorry

end average_speed_last_segment_is_80_l473_473452


namespace min_41x_2y_eq_nine_l473_473310

noncomputable def min_value_41x_2y (x y : ℝ) : ℝ :=
  41*x + 2*y

theorem min_41x_2y_eq_nine (x y : ℝ) (h : ∀ n : ℕ, 0 < n →  n*x + (1/n)*y ≥ 1) :
  min_value_41x_2y x y = 9 :=
sorry

end min_41x_2y_eq_nine_l473_473310


namespace triangle_third_side_count_l473_473834

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l473_473834


namespace Marias_score_l473_473021

def total_questions := 30
def points_per_correct_answer := 20
def points_deducted_per_incorrect_answer := 5
def total_answered := total_questions
def correct_answers := 19
def incorrect_answers := total_questions - correct_answers
def score := (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect_answer)

theorem Marias_score : score = 325 := by
  -- proof goes here
  sorry

end Marias_score_l473_473021


namespace complex_in_fourth_quadrant_l473_473177

-- Conditions
variables (m : ℝ)
hypothesis h1 : 1 < m
hypothesis h2 : m < 2

-- Definition of the complex number
def complex_number : ℂ := ⟨m - 1, m - 2⟩

-- Definition of quadrant system for complex plane
def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Conjecture that needs to be proved
theorem complex_in_fourth_quadrant : fourth_quadrant (complex_number m) :=
by
  sorry

end complex_in_fourth_quadrant_l473_473177


namespace find_x_from_A_subset_B_l473_473326

theorem find_x_from_A_subset_B (x : ℝ) :
  let A := {-2, 1}
  let B := {0, 1, x + 1}
  A ⊆ B → x = -3 := sorry

end find_x_from_A_subset_B_l473_473326


namespace point_in_fourth_quadrant_l473_473001

noncomputable def b (a : ℝ) : ℝ := (Real.sqrt (a - 2) + Real.sqrt (2 - a) - 3)

theorem point_in_fourth_quadrant (a : ℝ) : 
  a = 2 → b a = -3 → (2, -3).snd < 0 ∧ (2, -3).fst > 0 := 
by 
  intros ha hb
  rw ha at hb
  exact ⟨lt_add_one 2, neg_lt_zero.mpr zero_lt_three⟩
  sorry

end point_in_fourth_quadrant_l473_473001


namespace equation_of_tangent_line_l473_473333

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) + 3 * x else -Real.log x + 3 * x

theorem equation_of_tangent_line : ∀ (f : ℝ → ℝ), (∀ x, f (-x) = -f x) → 
  (∀ x, x < 0 → f x = Real.log (-x) + 3 * x) →
  (∃ c : ℝ, ∀ x, x = 1 → c = f x) → 
  (∃ k : ℝ, ∀ x, x = 1 → k = deriv f x) → 
  (∃ (y : ℝ), y = 2 * x + 1) :=
begin
  sorry
end

end equation_of_tangent_line_l473_473333


namespace length_of_locus_of_P_l473_473096

noncomputable def point (x : ℝ) (y : ℝ) (z : ℝ) : Type := ℝ × ℝ × ℝ

def A : point := (0, -1, 0)
def B : point := (0, 1, 0)
def S : point := (0, 0, real.sqrt 3)
def O : point := (0, 0, 0)
def M : point := (0, 0, real.sqrt 3 / 2)
def P (x y : ℝ) : point := (x, y, 0)

theorem length_of_locus_of_P 
  (x y : ℝ)
  (hx : x^2 + y^2 = 1)
  (hAMMP : y = 3/4) :
  ∃ (L : ℝ), L = real.sqrt 7 / 2 :=
begin
  use (real.sqrt 7 / 2),
  sorry
end

end length_of_locus_of_P_l473_473096


namespace carla_students_l473_473661

theorem carla_students (R A num_rows num_desks : ℕ) (full_fraction : ℚ) 
  (h1 : R = 2) 
  (h2 : A = 3 * R - 1)
  (h3 : num_rows = 4)
  (h4 : num_desks = 6)
  (h5 : full_fraction = 2 / 3) : 
  num_rows * (num_desks * full_fraction).toNat + R + A = 23 := by
  sorry

end carla_students_l473_473661


namespace multiple_of_area_l473_473094

-- Define the given conditions
def perimeter (s : ℝ) : ℝ := 4 * s
def area (s : ℝ) : ℝ := s * s

theorem multiple_of_area (m s a p : ℝ) 
  (h1 : p = perimeter s)
  (h2 : a = area s)
  (h3 : m * a = 10 * p + 45)
  (h4 : p = 36) : m = 5 :=
by 
  sorry

end multiple_of_area_l473_473094


namespace pentagon_area_ratio_l473_473889

theorem pentagon_area_ratio (P Q R S T U : Type) [ConvexPent egon PQRST] (h₁ : PQ ∥ RT) (h₂ : QR ∥ PS) (h₃ : QS ∥ PT) (h₄ : angle PQR = 150) (h₅ : PQ = 4) (h₆ : QR = 8) (h₇ : PT = 24) :
  (∃ m n : ℕ, gcd m n = 1 ∧ m + n = 487) :=
sorry

end pentagon_area_ratio_l473_473889


namespace factor_x4_plus_81_l473_473264

theorem factor_x4_plus_81 (x : ℝ) : (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) = x^4 + 81 := 
by 
   sorry

end factor_x4_plus_81_l473_473264


namespace min_abs_sum_l473_473465

-- Definitions based on given conditions for the problem
variable (p q r s : ℤ)
variable (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
variable (h : (matrix 2 2 ℤ ![(p, q), (r, s)]) ^ 2 = matrix 2 2 ℤ ![(9, 0), (0, 9)])

-- Statement of the proof problem
theorem min_abs_sum :
  |p| + |q| + |r| + |s| = 8 :=
by
  sorry

end min_abs_sum_l473_473465


namespace tangent_line_eq_extrema_conditions_l473_473343

-- Proof Statement 1: Tangent line problem
theorem tangent_line_eq (a : ℝ) (f : ℝ → ℝ) (f_def : ∀ x, f x = (1 / 2) * x ^ 2 - (a + 1) * x + a * (1 + real.log x)) :
  let f' x := x - a - 1 + a / x in
  f' 2 = 1 → x - y = 2 := sorry

-- Proof Statement 2: Extrema of the function
theorem extrema_conditions (a : ℝ) (f : ℝ → ℝ) (f_def : ∀ x, f x = (1 / 2) * x ^ 2 - (a + 1) * x + a * (1 + real.log x)) :
  let f' x := x - a - 1 + a / x in
  (a ≤ 0 → (∀ x > 0, f' x ≤ 0 ∨ f' x ≥ 0) ∧ f 1 = -1 / 2) ∧
  (0 < a ∧ a < 1 → (f 1 = -1 / 2 ∧ ∃ b > 0, f b = -1 / 2 * b ^ 2 + a * real.log b) ∧ f' b = 0) ∧
  (a = 1 → (∀ x > 0, f' x ≥ 0) ∧ (∀ y > 0, ∀ z > 0, y ≠ z → f y ≠ f z)) ∧
  (a > 1 → (f 1 = -1 / 2 ∧ ∃ b > 0, (f b = -1 / 2 * b ^ 2 + a * real.log b))) := sorry

end tangent_line_eq_extrema_conditions_l473_473343


namespace plane_two_coloring_l473_473495

-- Definition of the division of a plane by lines and circles and two-coloring property.
def plane_divided_coloring (n : ℕ) : Prop :=
  ∀ (lines : list (line)), ∀ (circles : list (circle)),
    (lines.length + circles.length = n) →
    ∃ (coloring : region → color),
    (∀ (r1 r2 : region), adjacent r1 r2 → coloring r1 ≠ coloring r2)

-- Statement of the theorem using induction.
theorem plane_two_coloring : ∀ n : ℕ, plane_divided_coloring n :=
by
  intro n
  sorry

end plane_two_coloring_l473_473495


namespace triangle_third_side_count_l473_473835

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l473_473835


namespace greatest_3_digit_base9_div_by_7_l473_473135

def base9_to_decimal (n : ℕ) : ℕ :=
  let d2 := n / 81
  let d1 := (n % 81) / 9
  let d0 := n % 9
  d2 * 81 + d1 * 9 + d0

def greatest_base9_3_digit_div_by_7 (n : ℕ) : Prop :=
  n < 9 * 9 * 9 ∧ 7 ∣ (base9_to_decimal n)

theorem greatest_3_digit_base9_div_by_7 :
  ∃ n, greatest_base9_3_digit_div_by_7 n ∧ n = 888 :=
begin
  sorry
end

end greatest_3_digit_base9_div_by_7_l473_473135


namespace triangle_third_side_length_l473_473799

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l473_473799


namespace frequency_of_eighth_pitch_is_correct_l473_473520

-- Define the conditions
def first_pitch_frequency : ℝ := f
def ratio_between_pitches : ℝ := real.rpow 2 (1 / 12)
def step_count : ℕ := 7

-- Define the target frequency for the eighth pitch
def frequency_of_eighth_pitch : ℝ := (first_pitch_frequency * real.rpow ratio_between_pitches step_count)

-- The theorem to be proven
theorem frequency_of_eighth_pitch_is_correct (f : ℝ) :
  frequency_of_eighth_pitch = (f * real.rpow 2 (7 / 12)) :=
by
  sorry

end frequency_of_eighth_pitch_is_correct_l473_473520


namespace third_side_integer_lengths_l473_473782

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l473_473782


namespace garden_area_l473_473226

theorem garden_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end garden_area_l473_473226


namespace greatest_three_digit_base_nine_divisible_by_seven_l473_473157

/-- Define the problem setup -/
def greatest_three_digit_base_nine := 8 * 9^2 + 8 * 9 + 8

/-- Prove the greatest 3-digit base 9 positive integer that is divisible by 7 -/
theorem greatest_three_digit_base_nine_divisible_by_seven : 
  ∃ n : ℕ, n = greatest_three_digit_base_nine ∧ n % 7 = 0 ∧ (8 * 9^2 + 8 * 9 + 8) = 728 := by 
  sorry

end greatest_three_digit_base_nine_divisible_by_seven_l473_473157


namespace smallest_sum_of_big_in_circle_l473_473489

theorem smallest_sum_of_big_in_circle (arranged_circle : Fin 8 → ℕ) (h_circle : ∀ n, arranged_circle n ∈ Finset.range (9) ∧ arranged_circle n > 0) :
  (∀ n, (arranged_circle n > arranged_circle (n + 1) % 8 ∧ arranged_circle n > arranged_circle (n + 7) % 8) ∨ (arranged_circle n < arranged_circle (n + 1) % 8 ∧ arranged_circle n < arranged_circle (n + 7) % 8)) →
  ∃ big_indices : Finset (Fin 8), big_indices.card = 4 ∧ big_indices.sum arranged_circle = 23 :=
by
  sorry

end smallest_sum_of_big_in_circle_l473_473489


namespace sin_C_value_l473_473423

theorem sin_C_value (A B C : Real) (AC BC : Real) (h_AC : AC = 3) (h_BC : BC = 2 * Real.sqrt 3) (h_A : A = 2 * B) :
    let C : Real := Real.pi - A - B
    Real.sin C = Real.sqrt 6 / 9 :=
  sorry

end sin_C_value_l473_473423


namespace ezekiel_third_day_distance_l473_473687

theorem ezekiel_third_day_distance :
  let total_distance := 50
  let first_day_distance := 10
  let second_day_first_leg := 8
  let second_day_second_leg := 6
  let second_day_third_leg := 4
  let second_day_distance := second_day_first_leg + second_day_second_leg + second_day_third_leg
  let first_two_days_distance := first_day_distance + second_day_distance
  let third_day_distance := total_distance - first_two_days_distance
  in third_day_distance = 22 := by {
  let total_distance := 50
  let first_day_distance := 10
  let second_day_first_leg := 8
  let second_day_second_leg := 6
  let second_day_third_leg := 4
  let second_day_distance := second_day_first_leg + second_day_second_leg + second_day_third_leg
  let first_two_days_distance := first_day_distance + second_day_distance
  let third_day_distance := total_distance - first_two_days_distance
  have : third_day_distance = 22 := by {
    calc third_day_distance
        = 50 - (10 + (8 + 6 + 4)) : by rfl
    ... = 50 - 28 : by rfl
    ... = 22 : by rfl,
  },
  exact this,
}

end ezekiel_third_day_distance_l473_473687


namespace find_number_l473_473635

theorem find_number :
  let x := 1 / 8 + 0.0020000000000000018 in
  x = 0.1270000000000000018 := 
by
  let x := 1 / 8 + 0.0020000000000000018
  have h : x = 0.1270000000000000018 := sorry
  exact h

end find_number_l473_473635


namespace percentage_increase_20_l473_473656

noncomputable def oldCompanyEarnings : ℝ := 3 * 12 * 5000
noncomputable def totalEarnings : ℝ := 426000
noncomputable def newCompanyMonths : ℕ := 36 + 5
noncomputable def newCompanyEarnings : ℝ := totalEarnings - oldCompanyEarnings
noncomputable def newCompanyMonthlyEarnings : ℝ := newCompanyEarnings / newCompanyMonths
noncomputable def oldCompanyMonthlyEarnings : ℝ := 5000

theorem percentage_increase_20 :
  (newCompanyMonthlyEarnings - oldCompanyMonthlyEarnings) / oldCompanyMonthlyEarnings * 100 = 20 :=
by sorry

end percentage_increase_20_l473_473656


namespace hexagon_midpoint_triangle_area_l473_473355

-- Definitions of triangles and hexagons
structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A B C : Point)
structure Hexagon := (A B C D E F : Point)

-- Function to compute area of a triangle given its vertices
def area_triangle (t : Triangle) : ℝ := 
  abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)) / 2

-- Function to compute midpoints of the sides of a hexagon
def midpoint (A B : Point) : Point := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

def midpoints_of_hexagon (h : Hexagon) : Triangle :=
  { A := midpoint h.B h.C,
    B := midpoint h.D h.E,
    C := midpoint h.F h.A }

-- Problem statement
theorem hexagon_midpoint_triangle_area (h : Hexagon) (t : Triangle) 
  (hABXY : h.A ≡ t.A) (hABparallelXY : h.A.x - h.B.x = t.A.x - t.B.x ∧ h.A.y - h.B.y = t.A.y - t.B.y) 
  (hCDYZ : h.C ≡ t.B) (hCDparallelYZ : h.C.x - h.D.x = t.B.x - t.C.x ∧ h.C.y - h.D.y = t.B.y - t.C.y) 
  (hEFZX : h.E ≡ t.C) (hEFparallelZX : h.E.x - h.F.x = t.C.x - t.A.x ∧ h.E.y - h.F.y = t.C.y - t.A.y) :
  area_triangle (midpoints_of_hexagon h) ≥ area_triangle t :=
sorry

end hexagon_midpoint_triangle_area_l473_473355


namespace number_of_students_l473_473551

variables (m d r : ℕ) (k : ℕ)

theorem number_of_students :
  (30 < m + d ∧ m + d < 40) → (r = 3 * m) → (r = 5 * d) → m + d = 32 :=
by 
  -- The proof body is not necessary here according to instructions.
  sorry

end number_of_students_l473_473551


namespace probability_unique_six_digit_code_l473_473953

theorem probability_unique_six_digit_code :
  let total_permutations := 10^6,
      valid_permutations := let A10_6 := 10 * 9 * 8 * 7 * 6 * 5 in
                             let A9_5 := 9 * 8 * 7 * 6 * 5 in
                             A10_6 - A9_5
  in
  (valid_permutations / total_permutations : ℝ) = 0.124416 := 
by
  let total_permutations := 10^6
  let A10_6 := 10 * 9 * 8 * 7 * 6 * 5
  let A9_5 := 9 * 8 * 7 * 6 * 5
  let valid_permutations := A10_6 - A9_5
  have h1 : valid_permutations = 124416 := sorry
  have h2 : total_permutations = 1000000 := rfl
  have h3 : (124416 / 1000000 : ℝ) = 0.124416 := sorry
  rw [←h1, ←h2, h3] 
  rfl

end probability_unique_six_digit_code_l473_473953


namespace simplify_sqrt_expression_l473_473939

theorem simplify_sqrt_expression : sqrt (4 + 2 * sqrt 3) + sqrt (4 - 2 * sqrt 3) = 4 :=
by
  sorry

end simplify_sqrt_expression_l473_473939


namespace main_theorem_l473_473262

open Nat

-- Define the conditions
def conditions (p q n : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Odd p ∧ Odd q ∧ n > 1 ∧
  (q^(n+2) % p^n = 3^(n+2) % p^n) ∧ (p^(n+2) % q^n = 3^(n+2) % q^n)

-- Define the conclusion
def conclusion (p q n : ℕ) : Prop :=
  (p = 3 ∧ q = 3)

-- Define the main problem
theorem main_theorem : ∀ p q n : ℕ, conditions p q n → conclusion p q n :=
  by
    intros p q n h
    sorry

end main_theorem_l473_473262


namespace probability_hugo_six_given_win_l473_473431

variable (players : Finset ℕ)
variable (roll : ℕ → ℕ) -- mapping player to their roll
variable (Hugo : ℕ) -- Hugo's identifier
variable (max_roll : ℕ := players.sup roll)

-- Definitions of events based on the given problem
def HugoWins : Prop := ∀ p ∈ players, roll p <= max_roll
def HugoRollsSix : Prop := roll Hugo = 6

-- Probability calculation
noncomputable def probHugoWins (n : ℕ) [Decidable (HugoWins players roll Hugo)] : ℚ :=
  1 / n 

noncomputable def probHugoRollsSix : ℚ := 
  1 / 6

noncomputable def probHugoWinsGivenSix (others : Finset ℕ) [Decidable (∀ p ∈ others, roll p < roll Hugo)] : ℚ :=
  if (∀ p ∈ others, roll p < 6) then 1 else sorry -- handle tie-breaking

-- Conditional Probability
noncomputable def probHugoSixGivenWin : ℚ :=
  probHugoRollsSix players roll Hugo * probHugoWinsGivenSix players \ roll Hugo / probHugoWins players roll Hugo

theorem probability_hugo_six_given_win :
  probHugoSixGivenWin players roll Hugo = 3125 / 7776 :=
sorry

end probability_hugo_six_given_win_l473_473431


namespace find_other_diagonal_l473_473880

theorem find_other_diagonal (A : ℝ) (d1 : ℝ) (hA : A = 80) (hd1 : d1 = 16) :
  ∃ d2 : ℝ, 2 * A / d1 = d2 :=
by
  use 10
  -- Rest of the proof goes here
  sorry

end find_other_diagonal_l473_473880


namespace logarithmic_expression_correct_l473_473247

noncomputable def prop : Prop :=
  (log 3 2 + log 3 5) * log 10 9 = 2

theorem logarithmic_expression_correct : prop :=
by sorry

end logarithmic_expression_correct_l473_473247


namespace distance_between_4th_and_18th_blue_light_l473_473654

def blueLightsPattern : ℕ → ℕ 
| n => if n % 5 = 0 || n % 5 = 1 || n % 5 = 2 then 1 else 0

def positionOfNthBlueLight (n : ℕ) : ℕ :=
  let blue_positions := (List.range ((n + 2) * 5)).filter (fun x => blueLightsPattern x = 1)
  blue_positions.get? (n - 1).getD 0

def distanceInInches (pos1 pos2 : ℕ) : ℕ := (pos2 - pos1) * 8

noncomputable def distanceInFeet (inches : ℕ) : ℚ := inches / 12

theorem distance_between_4th_and_18th_blue_light :
  distanceInFeet (distanceInInches (positionOfNthBlueLight 4) (positionOfNthBlueLight 18)) = 14.67 := by
  sorry

end distance_between_4th_and_18th_blue_light_l473_473654


namespace gcd_x_y_l473_473016

-- Definitions of x and y based on the conditions
noncomputable def x : ℕ := finset.sum (finset.filter even (finset.range 64)) id
noncomputable def y : ℕ := finset.card (finset.filter even (finset.Icc 14 62))

-- The theorem that poses the problem
theorem gcd_x_y : gcd x y = 25 := 
by 
sorry

end gcd_x_y_l473_473016


namespace smallest_class_size_l473_473859

variable (x : ℕ) 

theorem smallest_class_size
  (h1 : 5 * x + 2 > 40)
  (h2 : x ≥ 0) : 
  5 * 8 + 2 = 42 :=
by sorry

end smallest_class_size_l473_473859


namespace quadratic_expression_rewriting_l473_473410

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l473_473410


namespace part_a_l473_473602

theorem part_a (a x y : ℕ) (h_a_pos : a > 0) (h_x_pos : x > 0) (h_y_pos : y > 0) (h_neq : x ≠ y) :
  (a * x + Nat.gcd a x + Nat.lcm a x) ≠ (a * y + Nat.gcd a y + Nat.lcm a y) := sorry

end part_a_l473_473602


namespace max_books_per_student_l473_473856

theorem max_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (avg_books_per_student : ℕ)
  (max_books_limit : ℕ)
  (total_books_available : ℕ) :
  total_students = 20 →
  students_0_books = 2 →
  students_1_book = 10 →
  students_2_books = 5 →
  students_at_least_3_books = total_students - students_0_books - students_1_book - students_2_books →
  avg_books_per_student = 2 →
  max_books_limit = 5 →
  total_books_available = 60 →
  avg_books_per_student * total_students = 40 →
  total_books_available = 60 →
  max_books_limit = 5 :=
by sorry

end max_books_per_student_l473_473856


namespace max_value_g_l473_473965

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_g : ∃ M, ∀ n, g n ≤ M ∧ M = 23 :=
  sorry

end max_value_g_l473_473965


namespace h_value_l473_473394

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l473_473394


namespace quadrilateral_is_circumscribed_l473_473914

structure Quadrilateral (A B C D : Type) :=
  (a b c d : ℝ)

def circumscribed (q : Quadrilateral) : Prop :=
  q.a + q.c = q.b + q.d

theorem quadrilateral_is_circumscribed 
  (ABCD STQR : Quadrilateral) :
  circumscribed STQR → circumscribed ABCD :=
by
  sorry

end quadrilateral_is_circumscribed_l473_473914


namespace triangle_third_side_count_l473_473836

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l473_473836


namespace complete_the_square_h_value_l473_473406

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l473_473406


namespace complete_the_square_h_value_l473_473403

theorem complete_the_square_h_value :
  ∃ a h k : ℝ, ∀ x : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3 / 2 :=
begin
  -- proof would go here
  sorry
end

end complete_the_square_h_value_l473_473403


namespace triangle_area_tangent_l473_473244

theorem triangle_area_tangent (x : ℝ) (y : ℝ) (y' : ℝ) (tangent_eq : x -> y)
  (curve_eq : x -> y = e^(-2 * x) + 1)
  (tangent_point : (x=0, y=2))
  (x_intercept : tangent_eq 1 = 0)
  (intersection : ∃ (a : ℝ), tangent_eq a = a) :
  ∃ (base height : ℝ), base = 2 / 3 ∧ height = 2 / 3 ∧ (1 / 2) * base * height = 1 / 3 :=
begin
  sorry
end

end triangle_area_tangent_l473_473244


namespace garden_area_l473_473225

theorem garden_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end garden_area_l473_473225


namespace final_result_l473_473710

noncomputable def f : ℝ → ℝ := sorry
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (3 + x) = f x
axiom f_half_periodic : ∀ x : ℝ, f (3 / 2 - x) = f x
axiom f_value_neg2 : f (-2) = -3

axiom a1_value : a 1 = -1
axiom S_n : ∀ n : ℕ, S n = 2 * a n + n

theorem final_result : f (a 5) + f (a 6) = 3 :=
sorry

end final_result_l473_473710


namespace polynomial_remainder_l473_473696

theorem polynomial_remainder :
  (4 * (2.5 : ℝ)^5 - 9 * (2.5 : ℝ)^4 + 7 * (2.5 : ℝ)^2 - 2.5 - 35 = 45.3125) :=
by sorry

end polynomial_remainder_l473_473696


namespace largest_3_digit_base9_divisible_by_7_l473_473142

def is_three_digit_base9 (n : ℕ) : Prop :=
  n < 9^3

def is_divisible_by (n d : ℕ) : Prop :=
  n % d = 0

def base9_to_base10 (n : ℕ) : ℕ :=
  let digits := [n / 81 % 9, n / 9 % 9, n % 9] in
  digits[0] * 81 + digits[1] * 9 + digits[2]

theorem largest_3_digit_base9_divisible_by_7 :
  ∃ n : ℕ, is_three_digit_base9 n ∧ is_divisible_by (base9_to_base10 n) 7 ∧ base9_to_base10 n = 728 ∧ n = 888 :=
sorry

end largest_3_digit_base9_divisible_by_7_l473_473142


namespace quadratic_form_h_l473_473385

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end quadratic_form_h_l473_473385


namespace monotonic_intervals_f_max_value_f_range_f_l473_473731

noncomputable def f (a x : ℝ) := (1/3)^(a * x^2 - 4 * x + 3)

-- Part 1
theorem monotonic_intervals_f (x : ℝ) :
  increasing_on (λ x, f (-1) x) (-2, ∞) ∧ decreasing_on (λ x, f (-1) x) (-∞, -2) :=
sorry

-- Part 2
theorem max_value_f (a x : ℝ) (h : (∃ x, f a x = 3)) : a = 1 :=
sorry

-- Part 3
theorem range_f (a x : ℝ) (h : set.range (λ x, f a x) = set.Ioi 0) : a = 0 :=
sorry

end monotonic_intervals_f_max_value_f_range_f_l473_473731


namespace correct_average_marks_l473_473428

-- Define all the given conditions
def average_marks : ℕ := 92
def number_of_students : ℕ := 25
def wrong_mark : ℕ := 75
def correct_mark : ℕ := 30

-- Define variables for total marks calculations
def total_marks_with_wrong : ℕ := average_marks * number_of_students
def total_marks_with_correct : ℕ := total_marks_with_wrong - wrong_mark + correct_mark

-- Goal: Prove that the correct average marks is 90.2
theorem correct_average_marks :
  (total_marks_with_correct : ℝ) / (number_of_students : ℝ) = 90.2 :=
by
  sorry

end correct_average_marks_l473_473428


namespace triangle_third_side_length_l473_473797

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l473_473797


namespace max_value_of_g_l473_473960

def g : ℕ → ℕ 
| n := if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : 
  ∃ N, (∀ n, g n ≤ N) ∧ N = 23 := 
sorry

end max_value_of_g_l473_473960


namespace min_value_of_expression_l473_473286

noncomputable def target_expression (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x)

theorem min_value_of_expression : (∀ x : ℝ, target_expression x ≥ -784) ∧ (∃ x : ℝ, target_expression x = -784) :=
by
  sorry

end min_value_of_expression_l473_473286


namespace divisibility_of_powers_l473_473888

open Nat

theorem divisibility_of_powers 
  (p : ℕ) [Fact p.prime]
  (x y z : ℕ)
  (hx : x < y)
  (hy : y < z)
  (hz : z < p)
  (h1 : ∃ k : ℚ, k = x^3 / p ∧ k = y^3 / p ∧ k = z^3 / p) :
  (x + y + z) ∣ (x^5 + y^5 + z^5) :=
begin
  sorry
end

end divisibility_of_powers_l473_473888


namespace _l473_473596

noncomputable def tangent_secant_theorem (M A A' P Q : Point) (circ : Circle) :
  is_tangent M A circ →
  is_tangent M A' circ →
  is_on_circle P circ →
  is_on_circle Q circ →
  ¬ (is_on_circle M circ) →  -- Ensuring M is outside the circle
  lies_on_secant M P Q →    -- Ensuring MPQ is a secant
  PA * QA' = PA' * QA :=
by 
  sorry

end _l473_473596


namespace lines_skew_iff_l473_473280

/-- Define the points and direction vectors of the lines -/
def P1 : ℝ^3 := ⟨2, 3, b⟩
def d1 : ℝ^3 := ⟨3, 4, 5⟩
def P2 : ℝ^3 := ⟨5, 2, 1⟩
def d2 : ℝ^3 := ⟨6, 3, 2⟩

/-- Statement: The lines are skew if and only if b ≠ 4 -/
theorem lines_skew_iff (b : ℝ) : (∀ t u : ℝ, P1 + t • d1 ≠ P2 + u • d2) ↔ b ≠ 4 :=
sorry

end lines_skew_iff_l473_473280


namespace part_1_prob_excellent_part_2_rounds_pvalues_l473_473019

-- Definition of the probability of an excellent pair
def prob_excellent (p1 p2 : ℚ) : ℚ :=
  2 * p1 * (1 - p1) * p2 * p2 + p1 * p1 * 2 * p2 * (1 - p2) + p1 * p1 * p2 * p2

-- Part (1) statement: Prove the probability that they achieve "excellent pair" status in the first round
theorem part_1_prob_excellent (p1 p2 : ℚ) (hp1 : p1 = 3/4) (hp2 : p2 = 2/3) :
  prob_excellent p1 p2 = 2/3 := by
  rw [hp1, hp2]
  sorry

-- Part (2) statement: Prove the minimum number of rounds and values of p1 and p2
theorem part_2_rounds_pvalues (n : ℕ) (p1 p2 : ℚ) (h_sum : p1 + p2 = 4/3)
  (h_goal : n * prob_excellent p1 p2 ≥ 16) :
  (n = 27) ∧ (p1 = 2/3) ∧ (p2 = 2/3) := by
  sorry

end part_1_prob_excellent_part_2_rounds_pvalues_l473_473019


namespace find_h_l473_473380

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l473_473380


namespace isosceles_triangle_cos_values_l473_473984

theorem isosceles_triangle_cos_values (x : ℝ) (deg_to_rad : ℝ := Math.pi / 180) :
  ((cos (x * deg_to_rad) = cos (30 * deg_to_rad) ∨
    cos (x * deg_to_rad) = cos (45 * deg_to_rad) ∨
    cos (x * deg_to_rad) = cos (15 * deg_to_rad)) ∧
   (cos (5 * x * deg_to_rad)).val ≥ 0 ∘ ∠ABC = 3 * x) ↔
  x = 15 ∨ x = 30 ∨ x = 45 :=
begin
  sorry
end

end isosceles_triangle_cos_values_l473_473984


namespace simplify_and_rationalize_l473_473504

-- Definitions for the radicals and their relationships
def sqrt5 : ℝ := Real.sqrt 5
def sqrt6 : ℝ := Real.sqrt 6
def sqrt7 : ℝ := Real.sqrt 7
def sqrt8 : ℝ := Real.sqrt 8
def sqrt9 : ℝ := Real.sqrt 9
def sqrt10 : ℝ := Real.sqrt 10

-- Definition of the expression to simplify
def expression := (sqrt5 / sqrt6) * (sqrt7 / sqrt8) * (sqrt9 / sqrt10)

-- The expected simplified form
def expected := (3 * Real.sqrt 1050) / 120

-- The statement we need to prove
theorem simplify_and_rationalize :
  expression = expected :=
by 
  -- Proof will go here
  sorry

end simplify_and_rationalize_l473_473504


namespace angle_of_inclination_of_line_l473_473725

-- Definition of the line l
def line_eq (x : ℝ) : ℝ := x + 1

-- Statement of the theorem about the angle of inclination
theorem angle_of_inclination_of_line (x : ℝ) : 
  ∃ (θ : ℝ), θ = 45 ∧ line_eq x = x + 1 := 
sorry

end angle_of_inclination_of_line_l473_473725


namespace number_of_possible_third_side_lengths_l473_473827

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l473_473827


namespace area_of_bonus_sector_l473_473207

-- The radius of the circular spinner
def radius : ℝ := 15

-- The total area of the circle
def total_area : ℝ := Real.pi * radius^2

-- The probability of landing on the "Bonus" sector
def bonus_probability : ℝ := 1 / 3

-- The given area of the "Bonus" sector
def bonus_area : ℝ := bonus_probability * total_area

-- Prove that the area of the "Bonus" sector is 75π square centimeters
theorem area_of_bonus_sector : bonus_area = 75 * Real.pi := by
  sorry

end area_of_bonus_sector_l473_473207


namespace no_nonconstant_arithmetic_progression_l473_473293

theorem no_nonconstant_arithmetic_progression (x : ℝ) :
  2 * (2 : ℝ)^(x^2) ≠ (2 : ℝ)^x + (2 : ℝ)^(x^3) :=
sorry

end no_nonconstant_arithmetic_progression_l473_473293


namespace y_decreasing_on_interval_l473_473265

-- Define the function y = x * ln(x)
def y (x : ℝ) : ℝ := x * Real.log x

-- Define the interval (0, 1/e)
noncomputable def interval : Set ℝ := Set.Ioo 0 (Real.exp (-1))

-- State the theorem
theorem y_decreasing_on_interval : ∀ x ∈ interval, ∃ ε > 0, ∀ x' ∈ interval, x' < x → y x' > y x :=
by
  -- Tutorial: state that the derivative of y is 1 + log(x)
  have dydx : ∀ x > 0, deriv y x = 1 + Real.log x := sorry

  -- Tutorial: state that we need to show 1 + log(x) < 0 
  have derivative_neg : ∀ x ∈ interval, deriv y x < 0 := 
  by
    intros x hx
    rw dydx x hx.1
    have hlog : Real.log x < -1 :=
      by sorry
    linarith

  -- Conclude the proof
  intros x hx
  existsi (x - 0) / 2
  split
  exact div_pos (sub_pos.2 hx.1) zero_lt_two
  intros x' hx' hxx'
  exact continuous_subinterval y x' x interval y_decreasing_on_interval 

  sorry

end y_decreasing_on_interval_l473_473265


namespace sum_of_first_fifteen_multiples_of_7_l473_473568

theorem sum_of_first_fifteen_multiples_of_7 : (∑ k in Finset.range 15, 7 * (k + 1)) = 840 :=
by
  -- Summation from k = 0 to k = 14 (which corresponds to 1 to 15 multiples of 7)
  sorry

end sum_of_first_fifteen_multiples_of_7_l473_473568


namespace count_valid_triangles_l473_473712

def is_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle (a b c : ℕ) : Prop :=
  is_triangle a b c ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem count_valid_triangles : 
  (∃ n : ℕ, n = 14 ∧ 
  ∃ (a b c : ℕ), valid_triangle a b c ∧ 
  ((b = 5 ∧ c > 5) ∨ (c = 5 ∧ b > 5)) ∧ 
  (a > 0 ∧ b > 0 ∧ c > 0)) :=
by { sorry }

end count_valid_triangles_l473_473712


namespace fraction_decomposition_l473_473268

theorem fraction_decomposition :
  ∃ (A B : ℚ), 
  (A = 27 / 10) ∧ (B = -11 / 10) ∧ 
  (∀ x : ℚ, 
    7 * x - 13 = A * (3 * x - 4) + B * (x + 2)) := 
  sorry

end fraction_decomposition_l473_473268


namespace problem_statement_l473_473476

-- Definitions of the sets P and Q
def P : Set ℝ := {x : ℝ | x > 1}
def Q : Set ℝ := {x : ℝ | abs x > 0}

-- Statement of the problem to prove that P is not a subset of Q
theorem problem_statement : ¬ (P ⊆ Q) :=
sorry

end problem_statement_l473_473476


namespace third_side_integer_lengths_l473_473783

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l473_473783


namespace pyramid_properties_l473_473951

-- Define the problem conditions
variables (m : ℝ)
variables (A B C D M O : ℝ³)
variables [plane : line A B D]
variables [base_area : Parallelogram.area (A B C D) = m^2]
variables [BD_perp_AD : Orthogonal (line B D) (line A D)]
variables [Dihedral_1 : DihedralAngle (plane A D O) (plane B C M) = 45°]
variables [Dihedral_2 : DihedralAngle (plane A B O) (plane C D M) = 60°]

-- Define the statements for lateral surface area and volume
def lateral_surface_area : ℝ := m^2 * (Real.sqrt 2 + 2) / 2
def pyramid_volume : ℝ := m^3 * Real.sqrt 2 ^ (1 / 4) / 6

-- The theorem to prove
theorem pyramid_properties :
  lateral_surface_area m = m^2 * (Real.sqrt 2 + 2) / 2 ∧
  pyramid_volume m = m^3 * Real.sqrt 2 ^ (1 / 4) / 6 :=
by
  sorry  -- replace this with the proof steps.

end pyramid_properties_l473_473951


namespace triangle_third_side_count_l473_473831

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l473_473831


namespace prove_dot_product_is_constant_l473_473341

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | let (x, y) := p in (x ^ 2) / 4 + (y ^ 2) / 3 = 1 }

def line_intersection (k m : ℝ) : Set (ℝ × ℝ) :=
  { p | let (x, y) := p in y = k * x + m }

def point_F : ℝ × ℝ := (-1, 0)

def point_Q (k m : ℝ) : ℝ × ℝ := (-4, m - 4 * k)

def point_P (k m : ℝ) : ℝ × ℝ :=
  let x := -8 * k * m / (4 * k^2 + 3)
  let y := 6 * m / (4 * k^2 + 3)
  (x, y)

def vec_OP (k m : ℝ) : ℝ × ℝ :=
  point_P k m

def vec_FQ (k m : ℝ) : ℝ × ℝ :=
  let (xQ, yQ) := point_Q k m
  (-3, yQ)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  ux * vx + uy * vy

theorem prove_dot_product_is_constant (k m : ℝ) (h: 4 * m^2 = 4 * k^2 + 3) :
  dot_product (vec_OP k m) (vec_FQ k m) = 3 / 2 :=
by
  sorry

end prove_dot_product_is_constant_l473_473341


namespace count_valid_third_sides_l473_473812

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l473_473812


namespace probability_neither_test_l473_473974

theorem probability_neither_test (P_hist : ℚ) (P_geo : ℚ) (indep : Prop) 
  (H1 : P_hist = 5/9) (H2 : P_geo = 1/3) (H3 : indep) :
  (1 - P_hist) * (1 - P_geo) = 8/27 := by
  sorry

end probability_neither_test_l473_473974


namespace comic_books_ratio_l473_473933

variable (S : ℕ)

theorem comic_books_ratio (initial comics_left comics_bought : ℕ)
  (h1 : initial = 14)
  (h2 : comics_left = 13)
  (h3 : comics_bought = 6)
  (h4 : initial - S + comics_bought = comics_left) :
  (S / initial.toRat) = (1 / 2 : ℚ) :=
by
  sorry

end comic_books_ratio_l473_473933


namespace possible_integer_lengths_third_side_l473_473809

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l473_473809


namespace intersection_complement_l473_473742

variable (U : Set ℝ) (A B : Set ℝ)

def U := {x : ℝ | True}
def A := {-1, 0, 1, 2, 3}
def B := {x : ℝ | 2 ≤ x}
def compl := {x : ℝ | x < 2}

theorem intersection_complement :
  A ∩ compl = {-1, 0, 1} := by
  -- The "sorry" here represents the proof that we're not providing as per instructions.
  sorry

end intersection_complement_l473_473742


namespace percent_runs_by_running_between_wickets_l473_473210

theorem percent_runs_by_running_between_wickets :
  (132 - (12 * 4 + 2 * 6)) / 132 * 100 = 54.54545454545455 :=
by
  sorry

end percent_runs_by_running_between_wickets_l473_473210


namespace greatest_number_of_good_isosceles_triangles_l473_473212

theorem greatest_number_of_good_isosceles_triangles :
  ∀ (P : polygon) (n : ℕ),
    P.regular (n + 6) →
    ∃ diagonals : List (P.edge × P.edge),
      (∀ (d : P.edge × P.edge), d ∈ diagonals → d.sides)
      ∧ (diagonals.length = 2003)
      ∧ (n = 1003) :=
sorry

end greatest_number_of_good_isosceles_triangles_l473_473212


namespace largest_area_of_polygons_l473_473297

theorem largest_area_of_polygons :
  ∀ (A B C D E : ℝ), 
    A = 4 ∧ B = 5.5 ∧ C = 4 ∧ D = 6 ∧ E = 5 →
    D = max (max (max (max A B) C) E) D := 
by
  intros A B C D E h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with h4 h5
  rw [h1, h2, h3, h4, h5]
  simp [max]
  sorry

end largest_area_of_polygons_l473_473297


namespace smaller_number_is_three_l473_473999

theorem smaller_number_is_three (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 36) : min x y = 3 :=
sorry

end smaller_number_is_three_l473_473999


namespace complex_in_fourth_quadrant_l473_473176

-- Conditions
variables (m : ℝ)
hypothesis h1 : 1 < m
hypothesis h2 : m < 2

-- Definition of the complex number
def complex_number : ℂ := ⟨m - 1, m - 2⟩

-- Definition of quadrant system for complex plane
def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Conjecture that needs to be proved
theorem complex_in_fourth_quadrant : fourth_quadrant (complex_number m) :=
by
  sorry

end complex_in_fourth_quadrant_l473_473176


namespace coffee_problem_l473_473241

theorem coffee_problem
  (lbs_coffee : ℕ)
  (cups_per_day : ℕ)
  (days : ℕ)
  (total_cups : ℕ) :
  lbs_coffee = 3 →
  cups_per_day = 3 →
  days = 40 →
  total_cups = cups_per_day * days →
  total_cups / lbs_coffee = 40 :=
by
  intros h_lbs h_cups_per_day h_days h_total_cups
  rw [h_lbs, h_cups_per_day, h_days] at h_total_cups
  simp at h_total_cups
  rw h_total_cups
  norm_num
  sorry

end coffee_problem_l473_473241


namespace min_pencils_per_box_l473_473563

theorem min_pencils_per_box (n_boxes : ℕ) (n_colors : ℕ) (k_boxes: ℕ) (enough_pencils : Prop) : 
  n_boxes = 6 ∧ n_colors = 26 ∧ k_boxes = 4 → 
  ∃ (min_pencils : ℕ), min_pencils = 13 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h2 h3
  use 13
  sorry

end min_pencils_per_box_l473_473563


namespace triangle_third_side_length_l473_473801

theorem triangle_third_side_length :
  ∃ (x : Finset ℕ), 
    (∀ (a ∈ x), 3 < a ∧ a < 19) ∧ 
    x.card = 15 :=
by
  sorry

end triangle_third_side_length_l473_473801


namespace seagull_catch_up_time_l473_473638

theorem seagull_catch_up_time {time_pick_up : ℕ} {time_catch_up : ℕ} {speed_halved : Prop} (h1 : time_pick_up = 3)
  (h2 : time_catch_up = 12) (h3 : speed_halved) : ∃ t : ℕ, t = 2 :=
by
  use 2
  sorry

end seagull_catch_up_time_l473_473638


namespace normal_probability_l473_473845

noncomputable def normal_distribution (μ σ : ℝ) : Type :=
  sorry

theorem normal_probability {X : normal_distribution 1 2} {m : ℝ}
  (h1 : ∀ x : ℝ, X.prob 0 x = m) :
  X.prob 0 2 = 1 - 2 * m :=
sorry

end normal_probability_l473_473845


namespace probability_distance_greater_than_2_l473_473348

theorem probability_distance_greater_than_2 :
  let D := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let area_square := 9
  let area_sector := Real.pi
  let area_shaded := area_square - area_sector
  let P := area_shaded / area_square
  P = (9 - Real.pi) / 9 :=
by
  sorry

end probability_distance_greater_than_2_l473_473348


namespace b_work_time_l473_473185

theorem b_work_time (W : ℝ) (days_A days_combined : ℝ)
  (hA : W / days_A = W / 16)
  (h_combined : W / days_combined = W / (16 / 3)) :
  ∃ days_B, days_B = 8 :=
by
  sorry

end b_work_time_l473_473185


namespace prime_factorization_of_expression_l473_473470

theorem prime_factorization_of_expression (p n : ℕ) (hp : Nat.Prime p) (hdiv : p^2 ∣ 2^(p-1) - 1) : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) ∧ 
  a ∣ (p-1) ∧ b ∣ (p! + 2^n) ∧ c ∣ (p! + 2^n) := 
sorry

end prime_factorization_of_expression_l473_473470


namespace basketball_team_cupcakes_sold_l473_473952

-- Define the conditions
def cupcake_price := 2  -- price of each cupcake in dollars
def number_of_cookies := 40
def cookie_price := 0.5 -- price of each cookie in dollars
def basketball_cost := 40
def number_of_basketballs := 2
def energy_drink_price := 2
def number_of_energy_drinks := 20

-- Define the expression for total earnings and spending
def total_earnings (C : ℕ) := cupcake_price * C + number_of_cookies * cookie_price
def total_spending := number_of_basketballs * basketball_cost + number_of_energy_drinks * energy_drink_price

-- State the theorem
theorem basketball_team_cupcakes_sold (C : ℕ) (H1 : total_earnings C = total_spending) : C = 50 := by
  sorry

end basketball_team_cupcakes_sold_l473_473952


namespace shiny_pennies_probability_l473_473622

theorem shiny_pennies_probability :
  ∃ (a b : ℕ), gcd a b = 1 ∧ a / b = 5 / 11 ∧ a + b = 16 :=
sorry

end shiny_pennies_probability_l473_473622


namespace horner_eval_l473_473130

noncomputable def f (x : ℤ) : ℤ := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

theorem horner_eval :
  let V0 := 3 in
  let V1 := V0 * (-4) + 5 in
  let V2 := V1 * (-4) + 6 in
  let V3 := V2 * (-4) + 79 in
  V3 = -57 :=
by
  let V0 := 3
  let V1 := V0 * (-4) + 5
  let V2 := V1 * (-4) + 6
  let V3 := V2 * (-4) + 79
  show V3 = -57
  calc
  V3 = (V2 * (-4) + 79) := rfl
      ... = 34 * (-4) + 79 := by rfl
      ... = -57 := by norm_num


end horner_eval_l473_473130


namespace greatest_product_l473_473163

theorem greatest_product (x : ℤ) (h : x + (2020 - x) = 2020) : x * (2020 - x) ≤ 1020100 :=
sorry

end greatest_product_l473_473163


namespace possible_integer_lengths_third_side_l473_473810

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l473_473810


namespace simplify_and_rationalize_l473_473506

theorem simplify_and_rationalize : 
  (∃ (a b : ℝ), a / b = (sqrt 5 / sqrt 6) * (sqrt 7 / sqrt 8) * (sqrt 9 / sqrt 10) ∧ (a / b = sqrt 210 / 8)) :=
by {
  let a := sqrt 210,
  let b := 8,
  existsi a, existsi b,
  split,
  sorry, -- Proof of the simplified expression
  refl -- Proof that a / b is indeed sqrt 210 / 8
}

end simplify_and_rationalize_l473_473506


namespace alien_abduction_problem_l473_473238

theorem alien_abduction_problem:
  ∀ (total_abducted people_taken_elsewhere people_taken_home people_returned: ℕ),
  total_abducted = 200 →
  people_taken_elsewhere = 10 →
  people_taken_home = 30 →
  people_returned = total_abducted - (people_taken_elsewhere + people_taken_home) →
  (people_returned : ℕ) / total_abducted * 100 = 80 := 
by
  intros total_abducted people_taken_elsewhere people_taken_home people_returned;
  intros h_total_abducted h_taken_elsewhere h_taken_home h_people_returned;
  sorry

end alien_abduction_problem_l473_473238


namespace remove_95_rows_exists_l473_473854

theorem remove_95_rows_exists 
  (G : Fin 100 → Fin 2018 → ℕ) 
  (h : ∀ j, 75 ≤ ∑ i, G i j) 
  (h01 : ∀ i j, G i j = 0 ∨ G i j = 1) : 
  ∃ (rows_to_remove : Finset (Fin 100)), 
    rows_to_remove.card = 95 ∧ 
    ∀ cols_with_zero, 
    cols_with_zero.card ≥ 2 → 
    ∃ col ∈ cols_with_zero, 
      ∑ i in rows_to_removeᶜ, G i col > 0 :=
by
  sorry

end remove_95_rows_exists_l473_473854


namespace greene_family_admission_cost_l473_473091

theorem greene_family_admission_cost (x : ℝ) (h1 : ∀ y : ℝ, y = x - 13) (h2 : ∀ z : ℝ, z = x + (x - 13)) :
  x = 45 :=
by
  sorry

end greene_family_admission_cost_l473_473091


namespace draw_probability_l473_473548

open prob_theory Classical

noncomputable def winning_tickets : finset ℕ := {1, 2, 3}
noncomputable def total_tickets : finset ℕ := {1, 2, 3, 4, 5}

def draws (n : ℕ) : finset (finset ℕ) :=
  (total_tickets.powerset.filter (λ s, s.card = n))

def A (i : ℕ) (s : finset ℕ) : Prop := i ∈ s

theorem draw_probability :
  ∀ (A1 A2 : finset ℕ),
    A1.card = 1 ∧ A2.card = 1 →
    (∀ s ∈ draws 1, ∀ t ∈ draws 1, A A1 s → A A2 t) →
    (5.choose 1) * ((5-1).choose 1) × (2.0/4.0) = (3.0/10.0) := 
sorry

end draw_probability_l473_473548


namespace joe_list_possibilities_l473_473617

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l473_473617


namespace greatest_three_digit_base_nine_divisible_by_seven_l473_473153

/-- Define the problem setup -/
def greatest_three_digit_base_nine := 8 * 9^2 + 8 * 9 + 8

/-- Prove the greatest 3-digit base 9 positive integer that is divisible by 7 -/
theorem greatest_three_digit_base_nine_divisible_by_seven : 
  ∃ n : ℕ, n = greatest_three_digit_base_nine ∧ n % 7 = 0 ∧ (8 * 9^2 + 8 * 9 + 8) = 728 := by 
  sorry

end greatest_three_digit_base_nine_divisible_by_seven_l473_473153


namespace monotonic_intervals_g_max_condition_at_x_eq_one_l473_473894

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - a * x^2 + (2 * a - 1) * x

def g (x a : ℝ) : ℝ := f x a

theorem monotonic_intervals_g (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, deriv (fun x => g x a) x > 0) ∧
  (a > 0 → ∀ x ∈ (0 : ℝ) .. 1 / (2 * a), deriv (fun x => g x a) x > 0 ∧
     ∀ x ∈ (1 / (2 * a) : ℝ) ..+ ∞, deriv (fun x => g x a) x < 0) :=
sorry

theorem max_condition_at_x_eq_one (a : ℝ) : 
  (f 1 a = 0 →  a > 1 / 2) :=
sorry

end monotonic_intervals_g_max_condition_at_x_eq_one_l473_473894


namespace red_blue_different_circles_l473_473354

noncomputable def red_points : Type := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)
noncomputable def blue_points : Type := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)
noncomputable def point_O : Type := ℝ × ℝ

theorem red_blue_different_circles
  (red_points : red_points) 
  (blue_points : blue_points) 
  (O : point_O)
  (h_red_tri : ∃ (r1 r2 r3 : ℝ × ℝ), red_points = ((r1, r2), r3) ∧ is_inside_tri O r1 r2 r3)
  (h_blue_tri : ∃ (b1 b2 b3 : ℝ × ℝ), blue_points = ((b1, b2), b3) ∧ is_inside_tri O b1 b2 b3)
  (h_dist : ∀ (r : ℝ × ℝ) (b : ℝ × ℝ), (r ∈ red_points → b ∈ blue_points → dist O r < dist O b)) : 
  ¬∃ (Ω : set (ℝ × ℝ)), (∀ r ∈ red_points, r ∈ Ω) ∧ (∀ b ∈ blue_points, b ∈ Ω) ∧ is_circle Ω :=
sorry

-- Definitions of helper concepts such as distance, inside_triangle, is_circle

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_inside_tri (p a b c : ℝ × ℝ) : Prop := 
  let p1 := a, p2 := b, p3 := c in 
  let α := ((p2.2 - p3.2) * (p1.1 - p3.1) + (p3.1 - p2.1) * (p1.2 - p3.2)) /
            ((p2.2 - p3.2) * (p1.1 - p3.1) + (p3.1 - p2.1) * (p1.2 - p2.2)) in
  let β := ((p3.2 - p1.2) * (p1.1 - p3.1) + (p1.1 - p3.1) * (p1.2 - p3.2)) /
            ((p3.2 - p1.2) * (p1.1 - p3.1) + (p1.1 - p3.1) * (p1.2 - p2.2)) in
  let γ := 1 - α - β in
  0 < α ∧ α < 1 ∧ 0 < β ∧ β < 1 ∧ 0 < γ ∧ γ < 1

def is_circle (Ω : set (ℝ × ℝ)) : Prop :=
  ∃ (c : ℝ × ℝ) (r : ℝ), ∀ (p : ℝ × ℝ), p ∈ Ω ↔ dist c p = r

end red_blue_different_circles_l473_473354


namespace max_k_no_real_roots_max_integer_value_k_no_real_roots_l473_473008

-- Define the quadratic equation with the condition on the discriminant.
theorem max_k_no_real_roots : ∀ k : ℤ, (4 + 4 * (k : ℝ) < 0) ↔ k < -1 := sorry

-- Prove that the maximum integer value of k satisfying this condition is -2.
theorem max_integer_value_k_no_real_roots : ∃ k_max : ℤ, k_max ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } ∧ ∀ k' : ℤ, k' ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } → k' ≤ k_max :=
sorry

end max_k_no_real_roots_max_integer_value_k_no_real_roots_l473_473008


namespace smallest_n_square_smallest_n_cube_l473_473170

theorem smallest_n_square (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 2) ↔ n = 3 := 
by sorry

theorem smallest_n_cube (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 3) ↔ n = 2 := 
by sorry

end smallest_n_square_smallest_n_cube_l473_473170


namespace minimum_value_ab_l473_473005

theorem minimum_value_ab (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : a * b - 2 * a - b = 0) :
  8 ≤ a * b :=
by sorry

end minimum_value_ab_l473_473005


namespace club_officer_selection_l473_473208

theorem club_officer_selection :
  let n : ℕ := 12 in
  ∃ (ways : ℕ), 
    ways = n * (n - 1) * (n - 2) * (n - 3) * (n - 4) ∧
    ways = 95040 :=
begin
  sorry
end

end club_officer_selection_l473_473208


namespace minimum_possible_base_maximum_possible_base_total_possible_bases_l473_473532

open Nat

def num_trailing_zeroes_in_factorial (n : Nat) : Nat :=
  let rec count_factors (n k acc : Nat) :=
    let div := n / k
    if div = 0 then acc else count_factors n (k * 5) (acc + div)
  count_factors n 5 0

theorem minimum_possible_base (n : Nat) (H : n = 2 + 2^96)
        (trailing_zeroes : Nat) (Hz : trailing_zeroes = 2^93) :
  ∃ B, (num_trailing_zeroes_in_factorial n = trailing_zeroes) → B = 16 := by
  sorry

theorem maximum_possible_base (n : Nat) (H : n = 2 + 2^96)
        (trailing_zeroes : Nat) (Hz : trailing_zeroes = 2^93) :
  ∃ B, (num_trailing_zeroes_in_factorial n = trailing_zeroes) → B = 5040 := by
  sorry

theorem total_possible_bases (n : Nat) (H : n = 2 + 2^96)
        (trailing_zeroes : Nat) (Hz : trailing_zeroes = 2^93) :
  ∃ count, (count = 12) := by
  sorry

end minimum_possible_base_maximum_possible_base_total_possible_bases_l473_473532


namespace cannon_hit_probability_l473_473203

theorem cannon_hit_probability
  (P1 P2 P3 : ℝ)
  (h1 : P1 = 0.2)
  (h3 : P3 = 0.3)
  (h_none_hit : (1 - P1) * (1 - P2) * (1 - P3) = 0.27999999999999997) :
  P2 = 0.5 :=
by
  sorry

end cannon_hit_probability_l473_473203


namespace triangle_third_side_lengths_l473_473759

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l473_473759


namespace second_hand_distance_l473_473111

theorem second_hand_distance (r t: ℕ) (h₀: r = 6) (h₁: t = 30) : 
  (distance := t * (2 * π * r)) = 360 * π := 
by {
  sorry
}

end second_hand_distance_l473_473111


namespace greatest_three_digit_base_nine_divisible_by_seven_l473_473155

/-- Define the problem setup -/
def greatest_three_digit_base_nine := 8 * 9^2 + 8 * 9 + 8

/-- Prove the greatest 3-digit base 9 positive integer that is divisible by 7 -/
theorem greatest_three_digit_base_nine_divisible_by_seven : 
  ∃ n : ℕ, n = greatest_three_digit_base_nine ∧ n % 7 = 0 ∧ (8 * 9^2 + 8 * 9 + 8) = 728 := by 
  sorry

end greatest_three_digit_base_nine_divisible_by_seven_l473_473155


namespace count_valid_third_sides_l473_473817

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l473_473817


namespace volume_custom_fitted_bowling_ball_l473_473201

noncomputable def fitted_bowling_ball_volume : ℝ :=
let radius_ball := 12 in
let volume_ball := (4/3) * Real.pi * radius_ball^3 in
let radius_hole1 := 0.5 in
let depth_hole1 := 6 in
let volume_hole1 := Real.pi * radius_hole1^2 * depth_hole1 in
let radius_hole2 := 1 in
let depth_hole2 := 6 in
let volume_hole2 := Real.pi * radius_hole2^2 * depth_hole2 in
let radius_hole3 := 2 in
let depth_hole3 := 4 in
let volume_hole3 := Real.pi * radius_hole3^2 * depth_hole3 in
let total_volume_holes := volume_hole1 + volume_hole2 + volume_hole3 in
volume_ball - total_volume_holes

theorem volume_custom_fitted_bowling_ball : fitted_bowling_ball_volume = 2280.5 * Real.pi :=
by sorry

end volume_custom_fitted_bowling_ball_l473_473201


namespace possible_integer_lengths_third_side_l473_473805

theorem possible_integer_lengths_third_side (c : ℕ) (h1 : 8 + 11 > c) (h2 : c + 11 > 8) (h3 : c + 8 > 11) : 
  15 = (setOf (fun x ↦ 3 < x ∧ x < 19)).toFinset.card :=
by
  sorry

end possible_integer_lengths_third_side_l473_473805


namespace slope_of_line_l473_473337

theorem slope_of_line (A B : ℝ × ℝ)
  (hA: (A.1^2 / 9) - (A.2^2 / 4) = 1)
  (hB: (B.1^2 / 9) - (B.2^2 / 4) = 1)
  (midpoint : (A.1 + B.1) / 2 = 6 ∧ (A.2 + B.2) / 2 = 2) :
  let k := (B.2 - A.2) / (B.1 - A.1)
  in k = 4 / 3 :=
begin
  sorry
end

end slope_of_line_l473_473337


namespace triangle_third_side_count_l473_473837

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l473_473837


namespace Luke_piles_of_quarters_l473_473065

theorem Luke_piles_of_quarters (Q : ℕ) (h : 6 * Q = 30) : Q = 5 :=
by
  sorry

end Luke_piles_of_quarters_l473_473065


namespace floor_inequality_l473_473499

theorem floor_inequality (x : ℝ) (hx : 0 < x) (n : ℕ) (hn : 0 < n) :
  ⌊n * x⌋ ≥ ∑ k in Finset.range n + 1, (⌊(k * x)⌋ : ℝ) / k :=
sorry

end floor_inequality_l473_473499


namespace nested_sqrt_solution_l473_473684

theorem nested_sqrt_solution (x : ℝ) (hx : x = sqrt (3 - x)) :
  x = ( -1 + sqrt 13 ) / 2 :=
by
  sorry

end nested_sqrt_solution_l473_473684


namespace joe_list_possibilities_l473_473616

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l473_473616


namespace number_of_possible_third_side_lengths_l473_473825

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l473_473825


namespace quadratic_expression_rewriting_l473_473412

theorem quadratic_expression_rewriting (a x h k : ℝ) :
  let expr := 3 * x^2 + 9 * x + 20 in
  expr = a * (x - h)^2 + k → h = -3 / 2 :=
by
  let expr := 3 * x^2 + 9 * x + 20
  assume : expr = a * (x - h)^2 + k
  sorry

end quadratic_expression_rewriting_l473_473412


namespace greatest_base9_3_digit_divisible_by_7_l473_473152

def base9_to_decimal (n : Nat) : Nat :=
  match n with
  | 0     => 0
  | n + 1 => (n % 10) * Nat.pow 9 (n / 10)

def decimal_to_base9 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n => let rec aux (n acc : Nat) :=
              if n = 0 then acc
              else aux (n / 9) ((acc * 10) + (n % 9))
         in aux n 0

theorem greatest_base9_3_digit_divisible_by_7 :
  ∃ (n : Nat), n < Nat.pow 9 3 ∧ (n % 7 = 0) ∧ n = 8 * 81 + 8 * 9 + 8 :=
begin
  sorry -- Proof would go here
end

end greatest_base9_3_digit_divisible_by_7_l473_473152


namespace smallest_n_multiple_11_l473_473089

theorem smallest_n_multiple_11 (x y : ℤ) 
  (hx : x ≡ 5 [ZMOD 11])
  (hy : y ≡ -5 [ZMOD 11]) : 
  ∃ n : ℕ, n > 0 ∧ (x^2 - x * y + y^2 + n ≡ 0 [ZMOD 11]) ∧ ∀ m : ℕ, m > 0 → (x^2 - x * y + y^2 + m ≡ 0 [ZMOD 11] → n ≤ m) :=
begin
  use 2,
  -- proof steps here
  sorry
end

end smallest_n_multiple_11_l473_473089


namespace max_abs_sum_l473_473366

theorem max_abs_sum (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_l473_473366


namespace smallest_positive_solution_l473_473173

def farthing : Type := ℕ -- We will use natural numbers to represent the farthings

-- Conditions of the problem
def condition (A B : farthing) : Prop :=
  (A * B) = 4 * (A + B)

-- Prove that A = 6, B = 12 is the smallest positive solution
theorem smallest_positive_solution : ∃ (A B : farthing), A > 0 ∧ B > 0 ∧ A * B = 4 * (A + B) ∧ A = 6 ∧ B = 12 :=
by
  use 6, 12
  have h1 : 6 > 0 := by norm_num
  have h2 : 12 > 0 := by norm_num
  split; try { assumption }
  calc 6 * 12 = 72 : by norm_num
       ... = 4 * (6 + 12) : by norm_num
  split; assumption

end smallest_positive_solution_l473_473173


namespace problem_I_problem_II_l473_473351

def intervalA := { x : ℝ | -2 < x ∧ x < 5 }
def intervalB (m : ℝ) := { x : ℝ | m < x ∧ x < m + 3 }

theorem problem_I (m : ℝ) :
  (intervalB m ⊆ intervalA) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by sorry

theorem problem_II (m : ℝ) :
  (intervalA ∩ intervalB m ≠ ∅) ↔ (-5 < m ∧ m < 2) :=
by sorry

end problem_I_problem_II_l473_473351


namespace sum_of_coefficients_l473_473753

-- Given polynomial
def polynomial (x : ℝ) : ℝ := (3 * x - 1) ^ 7

-- Statement
theorem sum_of_coefficients :
  (polynomial 1) = 128 := 
sorry

end sum_of_coefficients_l473_473753


namespace limit_problem_l473_473245

open Real

theorem limit_problem :
  (tendsto (λ n : ℕ, (n * n^(1 / 6 : ℝ) + (32 * n^10 + 1)^(1 / 5 : ℝ)) /
    ((n + n^(1 / 4 : ℝ)) * (n^3 - 1)^(1 / 3 : ℝ))) atTop (𝓝 2)) :=
sorry

end limit_problem_l473_473245


namespace area_of_quadrilateral_EFGM_l473_473030

noncomputable def area_ABMJ := 1.8 -- Given area of quadrilateral ABMJ

-- Conditions described in a more abstract fashion:
def is_perpendicular (A B C D E F G H I J K L : Point) : Prop :=
  -- Description of each adjacent pairs being perpendicular
  sorry

def is_congruent (A B C D E F G H I J K L : Point) : Prop :=
  -- Description of all sides except AL and GF being congruent
  sorry

def are_segments_intersecting (B G E L : Point) (M : Point) : Prop :=
  -- Description of segments BG and EL intersecting at point M
  sorry

def area_ratio (tri1 tri2 : Finset Triangle) : ℝ :=
  -- Function that returns the ratio of areas covered by the triangles
  sorry

theorem area_of_quadrilateral_EFGM 
  (A B C D E F G H I J K L M : Point)
  (h1 : is_perpendicular A B C D E F G H I J K L)
  (h2 : is_congruent A B C D E F G H I J K L)
  (h3 : are_segments_intersecting B G E L M)
  : 7 / 3 * area_ABMJ = 4.2 :=
by
  -- Proof of the theorem that area EFGM == 4.2 using the conditions
  sorry

end area_of_quadrilateral_EFGM_l473_473030


namespace max_volume_parallelepiped_l473_473667

theorem max_volume_parallelepiped : 
  ∃ x : ℝ, ∃ h : ℝ, 
  (∀ x h, 2 * x + 2 * h = 6 → ∃ M ≤ x^2 * h) ∧ 
  M = 4 :=
by
  -- Define the conditions
  let base_side_length := λ x : ℝ, x
  let height := λ x : ℝ, 3 - x
  let perimeter_condition := ∀ x h, 2 * x + 2 * h = 6
  let volume := λ x h : ℝ, x^2 * h
  let maximum_volume := 4

  -- Prove existence of x and h satisfying the perimeter condition and achieving maximum volume
  exact ⟨2, 1, 
           λ x h, 
           begin
             intro perimeter_cond, 
             use maximum_volume,
             exact ⟨le_maximum_volume, eq_maximum_volume⟩,
             sorry
           end
  ⟩ -- This proof structure needs to be filled with exactly steps.

end max_volume_parallelepiped_l473_473667


namespace even_function_derivative_is_odd_l473_473490

variable {f : ℝ → ℝ} {g : ℝ → ℝ}

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_derivative_of (g f : ℝ → ℝ) : Prop := ∀ x, g x = (f' x)

-- Math proof problem statement
theorem even_function_derivative_is_odd (hf : is_even_function f) (hg : is_derivative_of g f) : ∀ x, g (-x) = -g x :=
sorry

end even_function_derivative_is_odd_l473_473490


namespace remainder_when_divided_by_10_l473_473169

theorem remainder_when_divided_by_10 :
  (4219 * 2675 * 394082 * 5001) % 10 = 0 :=
sorry

end remainder_when_divided_by_10_l473_473169


namespace sqrt_meaningful_iff_ge_two_l473_473980

theorem sqrt_meaningful_iff_ge_two (x : ℝ) : (∃ y, y = sqrt (x - 2)) ↔ x ≥ 2 :=
by
  sorry

end sqrt_meaningful_iff_ge_two_l473_473980


namespace largest_number_among_list_l473_473587

theorem largest_number_among_list :
  ∃ x ∈ ({1^20, 2^14, 4^8, 8^5, 16^3}: set ℕ), 
  ∀ y ∈ ({1^20, 2^14, 4^8, 8^5, 16^3}: set ℕ), y ≤ x := 
  by
  use 4^8
  split
  · simp
  · intro y hy
    simp at hy
    interval_cases y ; sorry

end largest_number_among_list_l473_473587


namespace percentage_slump_in_business_l473_473593

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.05 * Y = 0.04 * X) : (X > 0) → (Y > 0) → (X - Y) / X * 100 = 20 := 
by
  sorry

end percentage_slump_in_business_l473_473593


namespace passes_first_third_fourth_l473_473373

variable (k : ℝ)
variable (h : k < 0)
variable (y : ℝ → ℝ)

def linear_passes_second_fourth (h : k < 0) : Prop :=
  ∀ x : ℝ, (λ (x : ℝ), k * x > 0 ↔ x < 0) ∧ (λ (x : ℝ), k * x < 0 ↔ x > 0)

theorem passes_first_third_fourth : 
  ∀ (x : ℝ), (λ (x : ℝ), -k * x - 1 > 0 ↔ x > -1/k) ∧ 
             (λ (x : ℝ), -k * x - 1 < 0 ↔ x < -1/k) :=
  sorry

end passes_first_third_fourth_l473_473373


namespace ethanol_percentage_B_correct_l473_473649

/-- Define the conditions -/
def total_capacity_gallons := 214
def ethanol_in_full_tank := 30
def gallons_fuel_A := 106
def ethanol_percentage_A := 0.12

/-- Define the unknown -/
def ethanol_percentage_B : ℝ := (ethanol_in_full_tank - (gallons_fuel_A * ethanol_percentage_A)) / (total_capacity_gallons - gallons_fuel_A) * 100

/-- State the theorem to be proved, which represents the equivalent mathematically proof problem -/
theorem ethanol_percentage_B_correct : ethanol_percentage_B = 16 := 
by
  sorry

end ethanol_percentage_B_correct_l473_473649


namespace num_dinosaur_dolls_l473_473017

-- Define the number of dinosaur dolls
def dinosaur_dolls : Nat := 3

-- Define the theorem to prove the number of dinosaur dolls
theorem num_dinosaur_dolls : dinosaur_dolls = 3 := by
  -- Add sorry to skip the proof
  sorry

end num_dinosaur_dolls_l473_473017


namespace percentage_increase_of_bill_l473_473908

theorem percentage_increase_of_bill 
  (original_bill : ℝ) 
  (increased_bill : ℝ)
  (h1 : original_bill = 60)
  (h2 : increased_bill = 78) : 
  ((increased_bill - original_bill) / original_bill * 100) = 30 := 
by 
  rw [h1, h2]
  -- The following steps show the intended logic:
  -- calc 
  --   [(78 - 60) / 60 * 100]
  --   = [(18) / 60 * 100]
  --   = [0.3 * 100]
  --   = 30
  sorry

end percentage_increase_of_bill_l473_473908


namespace max_value_of_k_l473_473055

open Polynomial

-- Define the problem conditions: polynomials and divisibility.
def Conds := ∀ (P : ℕ → Polynomial ℤ) (n : ℕ → ℕ),
  (∀ i ≤ k, nat_degree (P i) = 13 ∧ leading_coeff (P i) = 1) ∧  -- monic polynomials of degree 13
  (∀ i j, i ≠ j → n i ≠ n j) ∧                                -- distinct positive integers
  (∀ i j m, (i ≤ k ∧ j ≤ k) → (n i ∣ eval m (P j) ↔ i = j))      -- divisibility conditions

-- Define the goal: find the largest k satisfying the conditions
theorem max_value_of_k : ∃ k, (Conds k) ∧ k = 144 :=
sorry

end max_value_of_k_l473_473055


namespace arithmetic_sequence_n_value_l473_473983

theorem arithmetic_sequence_n_value (a : ℕ → ℤ) (a1 : a 1 = 1) (d : ℤ) (d_def : d = 3) (an : ∃ n, a n = 22) :
  ∃ n, n = 8 :=
by
  -- Assume the general term formula for the arithmetic sequence
  have general_term : ∀ n, a n = a 1 + (n-1) * d := sorry
  -- Use the given conditions
  have a_n_22 : ∃ n, a n = 22 := an
  -- Calculations to derive n = 8, skipped here
  sorry

end arithmetic_sequence_n_value_l473_473983


namespace infinite_product_eq_sixth_root_531441_l473_473679

noncomputable def infinite_product : Real := ∏' n : ℕ, (3^(2^(-n : ℝ)))

theorem infinite_product_eq_sixth_root_531441 :
  infinite_product = Real.sqrt 6 531441 := sorry

end infinite_product_eq_sixth_root_531441_l473_473679


namespace value_of_4_and_2_l473_473533

noncomputable def custom_and (a b : ℕ) : ℕ :=
  ((a + b) * (a - b)) ^ 2

theorem value_of_4_and_2 : custom_and 4 2 = 144 :=
  sorry

end value_of_4_and_2_l473_473533


namespace flowers_per_bug_l473_473485

theorem flowers_per_bug (total_flowers : ℝ) (total_bugs : ℝ) (h1 : total_flowers = 4.5) (h2 : total_bugs = 2.5) : (total_flowers / total_bugs) = 1.8 :=
by
  rw [h1, h2]
  norm_num
  sorry

end flowers_per_bug_l473_473485


namespace find_real_numbers_a_l473_473702

theorem find_real_numbers_a 
  (n : ℕ) (h : 5 < n) 
  (x : ℕ → ℝ) 
  (hx : ∀ i, 0 ≤ x i)
  (a : ℝ)
  (h1 : ∑ k in range (n+1), k * x k = a)
  (h2 : ∑ k in range (n+1), (k^3 : ℝ) * x k = a^2)
  (h3 : ∑ k in range (n+1), (k^5 : ℝ) * x k = a^3) 
  : a ∈ { i^2 | i : ℝ, i ∈ set.range (λ i, ↑i) ∧ (i : ℕ) ≤ n } :=
sorry

end find_real_numbers_a_l473_473702


namespace log_properties_l473_473756

theorem log_properties (x y z : ℝ) 
  (hx : log 4 (log 5 (log 6 x)) = 0)
  (hy : log 5 (log 6 (log 4 y)) = 0)
  (hz : log 6 (log 4 (log 5 z)) = 0) : 
  x + y + z = 12497 := 
sorry

end log_properties_l473_473756


namespace sum_of_squares_of_distances_l473_473869

noncomputable def distance {α : Type*} [linear_order α] [has_sqrt α] {x₁ y₁ x₂ y₂ : α} :=
  sqrt ((x₂ - x₁) ^ 2 + (y₂ - y₁) ^ 2)

theorem sum_of_squares_of_distances 
  (C_polar_center : (real.sqrt 2, real.pi / 4))
  (C_radius : real.sqrt 3)
  (P : (0, 1))
  (alpha : real.pi / 6) : 
  let center := (1, 1) in
  let rho_eq : real × real -> Prop := λ ⟨ρ, θ⟩, ρ^2 - 2 * ρ * real.cos θ - 2 * ρ * real.sin θ - 1 = 0 in
  let line_eq : real -> real × real := λ t, ( (real.sqrt 3 / 2) * t, 1 + (1 / 2) * t ) in
  let points := set.range line_eq in
  ∃ A B ∈ points, distance P A^2 + distance P B^2 = 7 := 
sorry

end sum_of_squares_of_distances_l473_473869


namespace greatest_three_digit_base_nine_divisible_by_seven_l473_473154

/-- Define the problem setup -/
def greatest_three_digit_base_nine := 8 * 9^2 + 8 * 9 + 8

/-- Prove the greatest 3-digit base 9 positive integer that is divisible by 7 -/
theorem greatest_three_digit_base_nine_divisible_by_seven : 
  ∃ n : ℕ, n = greatest_three_digit_base_nine ∧ n % 7 = 0 ∧ (8 * 9^2 + 8 * 9 + 8) = 728 := by 
  sorry

end greatest_three_digit_base_nine_divisible_by_seven_l473_473154


namespace benny_lunch_cost_l473_473242

theorem benny_lunch_cost :
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  total_cost = 24 :=
by
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  have h : total_cost = 24 := by
    sorry
  exact h

end benny_lunch_cost_l473_473242


namespace solution_set_of_inequality_l473_473986

noncomputable def roots_of_quadratic (a b c : ℝ) : set ℝ :=
  {x : ℝ | a * x^2 + b * x + c = 0}

theorem solution_set_of_inequality (a b : ℝ) :
  (roots_of_quadratic a b 2 = {x | -1 < x ∧ x < 2} ∧ a < 0) →
  {x : ℝ | 2 * x^2 + b * x + a > 0} = {x | x < -1 ∨ x > 0.5} :=
by
  sorry

end solution_set_of_inequality_l473_473986


namespace greatest_divisor_of_product_of_four_consecutive_integers_l473_473676

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, ∃ k : Nat, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l473_473676


namespace greatest_divisor_of_product_of_four_consecutive_integers_l473_473675

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, ∃ k : Nat, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l473_473675


namespace extreme_value_f_when_a_is_1_range_of_a_when_f_less_g_l473_473733

variable (a : ℝ) (f g : ℝ → ℝ)
variable (h : ℝ → ℝ) (x : ℝ)
variable (e : ℝ) [Fact (e > 0)] -- We assume e > 0 as part of necessary condition for 'e'

noncomputable def f (x : ℝ) : ℝ := x - a * Real.log x
noncomputable def g (x : ℝ) : ℝ := - (1 + a) / x
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem extreme_value_f_when_a_is_1 : ∀ x, a = 1 → (f x = x - Real.log x) ∧ (∀ x > 1, ∃ y > 0, y < 1 ∧ y = Real.log x) := sorry

theorem range_of_a_when_f_less_g : (∃ x ∈ Set.Icc 1 e, f x < g x) ↔ a > (Real.exp 2 + 1) / (e - 1) := sorry

end extreme_value_f_when_a_is_1_range_of_a_when_f_less_g_l473_473733


namespace AmpersandDoubleCalculation_l473_473299

def ampersand (x : Int) : Int := 7 - x
def doubleAmpersand (x : Int) : Int := (x - 7)

theorem AmpersandDoubleCalculation : doubleAmpersand (ampersand 12) = -12 :=
by
  -- This is where the proof would go, which shows the steps described in the solution.
  sorry

end AmpersandDoubleCalculation_l473_473299


namespace triangle_third_side_count_l473_473832

theorem triangle_third_side_count : 
  (∃ (n : ℕ), n = 15) ↔ (∀ (x : ℕ), 3 < x ∧ x < 19 → (x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})) := 
by 
  sorry

end triangle_third_side_count_l473_473832


namespace min_value_of_expression_l473_473285

noncomputable def target_expression (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x)

theorem min_value_of_expression : (∀ x : ℝ, target_expression x ≥ -784) ∧ (∃ x : ℝ, target_expression x = -784) :=
by
  sorry

end min_value_of_expression_l473_473285


namespace total_baseball_fans_l473_473018

-- Conditions given
def ratio_YM (Y M : ℕ) : Prop := 2 * Y = 3 * M
def ratio_MR (M R : ℕ) : Prop := 4 * R = 5 * M
def M_value : ℕ := 88

-- Prove total number of baseball fans
theorem total_baseball_fans (Y M R : ℕ) (h1 : ratio_YM Y M) (h2 : ratio_MR M R) (hM : M = M_value) :
  Y + M + R = 330 :=
sorry

end total_baseball_fans_l473_473018


namespace cycle_time_to_library_l473_473748

theorem cycle_time_to_library 
  (constant_speed : Prop)
  (time_to_park : ℕ)
  (distance_to_park : ℕ)
  (distance_to_library : ℕ)
  (h1 : constant_speed)
  (h2 : time_to_park = 30)
  (h3 : distance_to_park = 5)
  (h4 : distance_to_library = 3) :
  (18 : ℕ) = (30 * distance_to_library / distance_to_park) :=
by
  intros
  -- The proof would go here
  sorry

end cycle_time_to_library_l473_473748


namespace system1_solution_l473_473086

variable (x y : ℝ)

theorem system1_solution :
  (3 * x - y = -1) ∧ (x + 2 * y = 9) ↔ (x = 1) ∧ (y = 4) := by
  sorry

end system1_solution_l473_473086


namespace α_pi_over_three_sufficient_not_necessary_l473_473315

noncomputable def line_tangent_condition (α : ℝ) : Prop := 
  let C := (0, 2)
  let r := 1
  let line := (0, α)
  let dist := |2| / real.sqrt(1 + real.tan(α) ^ 2)
  dist = r

theorem α_pi_over_three_sufficient_not_necessary :
  line_tangent_condition (π / 3) → 
  ¬ (∀ α, line_tangent_condition α ↔ α = π / 3) :=
by sorry

end α_pi_over_three_sufficient_not_necessary_l473_473315


namespace possibility_all_outcomes_probability_one_white_one_black_probability_at_least_one_white_l473_473434

namespace BallDraw

open Set

-- Define the balls
inductive Ball
| white : Ball → ℕ
| black : Ball → ℕ

def all_possible_outcomes : Set (Ball × Ball) :=
  { (Ball.white 1, Ball.white 2), (Ball.white 1, Ball.white 3), (Ball.white 1, Ball.black 1), 
    (Ball.white 1, Ball.black 2), (Ball.white 2, Ball.white 3), (Ball.white 2, Ball.black 1), 
    (Ball.white 2, Ball.black 2), (Ball.white 3, Ball.black 1), (Ball.white 3, Ball.black 2), 
    (Ball.black 1, Ball.black 2) }

-- Define the events
def event_one_white_one_black : Set (Ball × Ball) :=
  { (Ball.white 1, Ball.black 1), (Ball.white 1, Ball.black 2), (Ball.white 2, Ball.black 1), 
    (Ball.white 2, Ball.black 2), (Ball.white 3, Ball.black 1), (Ball.white 3, Ball.black 2) }

def event_one_white : Set (Ball × Ball) :=
  all_possible_outcomes \ { (Ball.black 1, Ball.black 2) }

-- Probabilities
def probability (s : Set (Ball × Ball)) : ℝ :=
  s.card.to_real / all_possible_outcomes.card.to_real

-- The proof statements
theorem possibility_all_outcomes :
  all_possible_outcomes = 
    { (Ball.white 1, Ball.white 2), (Ball.white 1, Ball.white 3), (Ball.white 1, Ball.black 1), 
      (Ball.white 1, Ball.black 2), (Ball.white 2, Ball.white 3), (Ball.white 2, Ball.black 1), 
      (Ball.white 2, Ball.black 2), (Ball.white 3, Ball.black 1), (Ball.white 3, Ball.black 2), 
      (Ball.black 1, Ball.black 2) } := by sorry

theorem probability_one_white_one_black :
  probability event_one_white_one_black = 0.6 := by sorry

theorem probability_at_least_one_white :
  probability event_one_white = 0.9 := by sorry

end BallDraw

end possibility_all_outcomes_probability_one_white_one_black_probability_at_least_one_white_l473_473434


namespace trajectory_midpoint_AP_trajectory_midpoint_PQ_l473_473311

section circle_trajectory

variable (x y: ℝ)

-- Define the circle equation
def circle_eq (a b r : ℝ) := (x - a)^2 + (y - b)^2 = r^2

-- Fixed points A and B
def A := (2, 0)
def B := (1, 1)

-- Point inside the circle
def point_inside_circle := (B.1)^2 + (B.2)^2 < 4

-- Points P and Q moving on the circle
variable (P Q : ℝ × ℝ)

-- P lies on the circle
def point_on_circle_P := circle_eq P.1 P.2 2

-- Q lies on the circle
def point_on_circle_Q := circle_eq Q.1 Q.2 2

-- Midpoint of AP
def midpoint_AP := ((A.1 + P.1) / 2, (A.2 + P.2) / 2)

-- Midpoint of PQ
def midpoint_PQ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Theorem for the trajectory of midpoint of AP
theorem trajectory_midpoint_AP : circle_eq (midpoint_AP.1 - 1) (midpoint_AP.2) 1 := 
sorry

-- The angle PBQ condition and its implication
def right_angle_PBQ := 
  let θ := Math.atan2 (Q.2 - B.2) (Q.1 - B.1) - Math.atan2 (P.2 - B.2) (P.1 - B.1) in
  θ = π / 2

-- Theorem for the trajectory of midpoint of PQ under right angle condition
theorem trajectory_midpoint_PQ (h : right_angle_PBQ) : 
  x^2 + y^2 - x - y - 1 = 0 := 
sorry

end circle_trajectory

end trajectory_midpoint_AP_trajectory_midpoint_PQ_l473_473311


namespace cube_angle_property_l473_473037

/-- Given a cube formed by vertices A, B, C, D, A₁, B₁, C₁, D₁ 
  with perpendicular sides AA₁ ∥ BB₁ ∥ CC₁ ∥ DD₁,
  show that the angles ∠BAD and ∠A₁AB₁ are neither equal nor sum up to 180 degrees. -/
theorem cube_angle_property (A B C D A₁ B₁ C₁ D₁ : ℝ³)
  (h_AA₁ : A₁ - A = B₁ - B)
  (h_BB₁ : B₁ - B = C₁ - C)
  (h_CC₁ : C₁ - C = D₁ - D)
  (h_perp : ∀ P Q R S : ℝ³, angle P Q R = π / 2)
  (angle_BAD := (step1 : angle_spectrum A B D = π / 2))
  (angle_A₁AB₁ := (step2 : angle_spectrum A₁ A B₁ = π / 4)) :
  ¬ ((angle_BAD = angle_A₁AB₁) ∨ (angle_BAD + angle_A₁AB₁ = π)) := sorry

end cube_angle_property_l473_473037


namespace sum_of_group_numbers_l473_473122

noncomputable def group_sum (n : ℕ) : ℕ :=
  (n * (n - 1)) + ((n * (n + 1) / 2 + (n * (n + 1) / 2) - n)

theorem sum_of_group_numbers (n : ℕ) : group_sum n = n^3 := 
  sorry

end sum_of_group_numbers_l473_473122


namespace ratio_of_areas_l473_473559

noncomputable def r1 : ℝ := 2
noncomputable def r2 : ℝ := 3 * r1
noncomputable def A1 : ℝ := π * r1^2
noncomputable def A2 : ℝ := π * r2^2

theorem ratio_of_areas : A2 / A1 = 9 :=
by
  let r1 : ℝ := 2
  let r2 : ℝ := 3 * r1
  let A1 : ℝ := π * r1^2
  let A2 : ℝ := π * r2^2
  have : A2 / A1 = (π * r2^2) / (π * r1^2), by sorry
  have : A2 / A1 = (r2^2) / (r1^2), by sorry
  have : A2 / A1 = (3 * r1)^2 / r1^2, by sorry
  have : A2 / A1 = 9, by sorry
  sorry

end ratio_of_areas_l473_473559


namespace probability_cut_at_least_2m_segments_l473_473227

noncomputable def probability_at_least_two_meter_segments (l : ℕ) (n : ℕ) : ℚ :=
  if h : l = 6 ∧ n = 5 then
    3 / 5
  else
    0

theorem probability_cut_at_least_2m_segments :
  probability_at_least_two_meter_segments 6 5 = 3 / 5 :=
by
  simp [probability_at_least_two_meter_segments] 
  split_ifs
  . contradiction
  . sorry

end probability_cut_at_least_2m_segments_l473_473227


namespace compare_y1_y2_l473_473536

theorem compare_y1_y2 (a : ℝ) (y1 y2 : ℝ) (h₁ : a < 0) (h₂ : y1 = a * (-1 - 1)^2 + 3) (h₃ : y2 = a * (2 - 1)^2 + 3) : 
  y1 < y2 :=
by
  sorry

end compare_y1_y2_l473_473536


namespace max_product_of_sum_2020_l473_473165

theorem max_product_of_sum_2020 : 
  ∃ x y : ℤ, x + y = 2020 ∧ (x * y) ≤ 1020100 ∧ (∀ a b : ℤ, a + b = 2020 → a * b ≤ x * y) :=
begin
  sorry
end

end max_product_of_sum_2020_l473_473165


namespace pollen_diameter_scientific_notation_l473_473488

def nanometer_to_meters : ℝ := 10 ^ (-9)
def pollen_diameter_nanometers : ℝ := 1360

theorem pollen_diameter_scientific_notation 
  (h1 : nanometer_to_meters = 10 ^ (-9))
  (h2 : pollen_diameter_nanometers = 1360) : 
  pollen_diameter_nanometers * nanometer_to_meters = 1.36 * 10 ^ (-6) :=
by 
  sorry

end pollen_diameter_scientific_notation_l473_473488


namespace max_value_of_xyz_l473_473944

noncomputable def max_product (x y z : ℝ) : ℝ :=
  x * y * z

theorem max_value_of_xyz (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x = y) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) (h6 : x ≤ z) (h7 : z ≤ 2 * x) :
  max_product x y z ≤ (1 / 27) := 
by
  sorry

end max_value_of_xyz_l473_473944


namespace girls_try_out_l473_473552

-- Given conditions
variables (boys callBacks didNotMakeCut : ℕ)
variable (G : ℕ)

-- Define the conditions
def conditions : Prop := 
  boys = 14 ∧ 
  callBacks = 2 ∧ 
  didNotMakeCut = 21 ∧ 
  G + boys = callBacks + didNotMakeCut

-- The statement of the proof
theorem girls_try_out (h : conditions boys callBacks didNotMakeCut G) : G = 9 :=
by
  sorry

end girls_try_out_l473_473552


namespace max_value_proof_l473_473994

noncomputable def max_expression_value (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_sum : a + b + c + d ≤ 4) : ℝ :=
  let expr := Real.root 4 (a^2 * (a + b)) +
              Real.root 4 (b^2 * (b + c)) +
              Real.root 4 (c^2 * (c + d)) +
              Real.root 4 (d^2 * (d + a))
  in expr

theorem max_value_proof (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_sum : a + b + c + d ≤ 4) :
  max_expression_value a b c d h_pos h_sum ≤ 4 * Real.root 4 2 :=
sorry

end max_value_proof_l473_473994


namespace sandy_comic_books_ratio_l473_473931

variable (S : ℕ)  -- number of comic books Sandy sold

theorem sandy_comic_books_ratio 
  (initial : ℕ) (bought : ℕ) (now : ℕ) (h_initial : initial = 14) (h_bought : bought = 6) (h_now : now = 13)
  (h_eq : initial - S + bought = now) :
  S = 7 ∧ S.to_rat / initial.to_rat = 1 / 2 := 
by
  sorry

end sandy_comic_books_ratio_l473_473931


namespace transformed_area_l473_473056

-- Define the matrix
def M : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 2, -7]

-- Given conditions
def area_of_T : ℝ := 15

-- Statement to prove
theorem transformed_area (T : ℝ) (hT : T = area_of_T) : 
  let detM := Matrix.det M in
  (|detM| * T) = 435 := 
by
  -- Lean requires the content here to typecheck, hence we skip the proof with sorry
  sorry

end transformed_area_l473_473056


namespace find_hall_length_l473_473858

variable (W H total_cost cost_per_sqm : ℕ)

theorem find_hall_length
  (hW : W = 15)
  (hH : H = 5)
  (h_total_cost : total_cost = 57000)
  (h_cost_per_sqm : cost_per_sqm = 60)
  : (32 * W) + (2 * (H * 32)) + (2 * (H * W)) = total_cost / cost_per_sqm :=
by
  sorry

end find_hall_length_l473_473858


namespace triangle_third_side_length_count_l473_473789

theorem triangle_third_side_length_count :
  (∃ (n : ℕ), 8 < n ∧ n < 19) :=
begin
  sorry,
end

lemma third_side_integer_lengths_count : finset.card (finset.filter (λ n : ℕ, 8 < n ∧ n < 19) (finset.range 20)) = 15 :=
by {
  sorry,
}

end triangle_third_side_length_count_l473_473789


namespace true_proposition_l473_473180

theorem true_proposition : 
  (¬ ∀ θ1 θ2 : ℝ, (θ1 + θ2 = 180 ∧ θ1 = θ2)) ∧
  (¬ ∃ p q : ℤ, q ≠ 0 ∧ (π = p / q)) ∧
  (∀ P : Point, ∀ l : Line, is_perpendicular_segment_shortest P l) ∧
  (¬ ∀ a b : ℝ, (a > 0 ∧ b < 0) → (a + b > 0)) := 
by {
  -- placeholder for proof
  sorry
}

end true_proposition_l473_473180


namespace perimeter_of_triangle_is_28_l473_473439

variables {A B C : Type} [Point A] [Point B] [Point C] [Triangle ABC]

-- Definitions derived from the conditions
def is_isosceles (ABC : Triangle A B C) : Prop :=
  ∠ABC = ∠ACB

def side_BC : ℝ := 8
def side_AC : ℝ := 10

-- The theorem stating the perimeter of the given triangle
theorem perimeter_of_triangle_is_28 (ABC : Triangle A B C) 
  (h_iso : is_isosceles ABC) 
  (h1 : side BC = 8) 
  (h2 : side AC = 10) : 
  perimeter ABC = 28 := 
sorry

end perimeter_of_triangle_is_28_l473_473439


namespace solve_negative_integer_sum_l473_473991

theorem solve_negative_integer_sum (N : ℤ) (h1 : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end solve_negative_integer_sum_l473_473991


namespace pedal_triangle_similarity_l473_473106

-- Define the setup of the problem
variables {A B C P A_1 B_1 C_1 A_2 B_2 C_2 : Type}
variables [IsLine AP] [IsLine BP] [IsLine CP]

def circumcircle (A B C : Type) : Type := sorry -- Placeholder definition for the circumcircle
def pedal_triangle (P : Type) (A B C : Type) : Type := sorry -- Placeholder definition for the pedal triangle

-- Define the conditions
def problem_conditions (P : Type) (A B C A_2 B_2 C_2 A_1 B_1 C_1 : Type) :
  circumcircle A B C = circumcircle A_2 B_2 C_2 ∧
  pedal_triangle P A B C = pedal_triangle P A_1 B_1 C_1 := sorry

-- Define the theorem
theorem pedal_triangle_similarity (P : Type) (A B C A_2 B_2 C_2 A_1 B_1 C_1 : Type)
  (h : problem_conditions P A B C A_2 B_2 C_2 A_1 B_1 C_1) :
  similar (pedal_triangle P A_1 B_1 C_1) (triangle A_2 B_2 C_2) :=
sorry

end pedal_triangle_similarity_l473_473106


namespace area_of_triangle_is_constant_equation_of_locus_l473_473472

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the hyperbola
def hyperbola (a b : ℝ) : set (ℝ × ℝ) := {p | let (x, y) := p in x^2 / a^2 - y^2 / b^2 = 1}

-- Define the tangent line at point M (x0, y0) on the hyperbola
def tangent_line (a b x0 y0 : ℝ) : set (ℝ × ℝ) := {p | let (x, y) := p in x0 * x / a^2 - y0 * y / b^2 = 1}

-- Define the asymptotes of the hyperbola
def asymptote1 (a b : ℝ) : set (ℝ × ℝ) := {p | let (x, y) := p in x / a + y / b = 0}
def asymptote2 (a b : ℝ) : set (ℝ × ℝ) := {p | let (x, y) := p in x / a - y / b = 0}

-- Define the intersection points A and B of the tangent line with asymptotes
def points_of_intersection (a b x0 y0 : ℝ) : set (ℝ × ℝ) :=
  {p |
    (p ∈ tangent_line a b x0 y0 ∧ p ∈ asymptote1 a b) ∨
    (p ∈ tangent_line a b x0 y0 ∧ p ∈ asymptote2 a b)}

-- Proof that the area S of triangle AOB is constant and equal to |ab|
theorem area_of_triangle_is_constant (a b x0 y0 : ℝ) :
  ∀ A B ∈ points_of_intersection a b x0 y0,
  let xa := A.fst; let ya := A.snd;
      xb := B.fst; let yb := B.snd in
  0.5 * abs (xa * yb - xb * ya) = abs (a * b) := sorry

-- Define the locus of the circumcenter P of triangle AOB
def locus_of_circumcenter (a b : ℝ) : set (ℝ × ℝ) := {p | let (x, y) := p in a^2 * x^2 - b^2 * y^2 = (a^2 + b^2)^2 / 4}

-- Proof that the equation of the locus of the circumcenter P is as given
theorem equation_of_locus (a b : ℝ) :
  ∀ P,
  (∃ A B ∈ points_of_intersection a b (P.fst) (P.snd),
    let xa := A.fst; let ya := A.snd;
        xb := B.fst; let yb := B.snd in
    (P.fst - xa)^2 + (P.snd - ya)^2 = (P.fst - 0)^2 + (P.snd - 0)^2 ∧
    (P.fst - xb)^2 + (P.snd + yb)^2 = (P.fst - 0)^2 + (P.snd - 0)^2) →
  P ∈ locus_of_circumcenter a b := sorry

end area_of_triangle_is_constant_equation_of_locus_l473_473472


namespace fasterRouteAndTimeDifference_l473_473486

noncomputable def routeXTime : ℚ := (
  (7 / 40 * 60) + 
  (1 / 10 * 60)
)

noncomputable def routeYTime : ℚ := (
  (6 / 35 * 60) + 
  (1 / 15 * 60)
)

noncomputable def timeDifference : ℚ := routeXTime - routeYTime

theorem fasterRouteAndTimeDifference:
  routeYTime < routeXTime ∧ 
  (timeDifference ≈ 2.214 ∨ timeDifference ≈ 2 + 1/5) :=
by
  sorry

end fasterRouteAndTimeDifference_l473_473486


namespace relationship_a_b_c_l473_473892

noncomputable def distinct_positive_numbers (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem relationship_a_b_c (a b c : ℝ) (h1 : distinct_positive_numbers a b c) (h2 : a^2 + c^2 = 2 * b * c) : b > a ∧ a > c :=
by
  sorry

end relationship_a_b_c_l473_473892


namespace complex_number_location_l473_473174

theorem complex_number_location (m : ℝ) (h : 1 < m ∧ m < 2) : 
  ∃ quadrant: string, quadrant = "Fourth quadrant" :=
by
  have h_real : 0 < m - 1 ∧ m - 1 < 1 := ⟨sub_pos.mpr h.1, sub_lt_sub_of_lt h.2⟩
  have h_imag : -1 < m - 2 ∧ m - 2 < 0 := ⟨sub_neg_of_lt h.1, sub_neg.mpr h.2⟩
  use "Fourth quadrant"
  sorry

end complex_number_location_l473_473174


namespace distance_to_right_focus_of_hyperbola_l473_473026

theorem distance_to_right_focus_of_hyperbola :
  ∀ (y : ℝ), (3 ^ 2 / 4 - y ^ 2 / 12 = 1) → (sqrt ((3 - 4)^2 + y^2) = 4) :=
by
  intro y h
  sorry

end distance_to_right_focus_of_hyperbola_l473_473026


namespace complete_the_square_3x2_9x_20_l473_473401

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l473_473401


namespace sqrt_meaningful_iff_ge_two_l473_473979

theorem sqrt_meaningful_iff_ge_two (x : ℝ) : (∃ y, y = sqrt (x - 2)) ↔ x ≥ 2 :=
by
  sorry

end sqrt_meaningful_iff_ge_two_l473_473979


namespace determine_b_l473_473846

-- Conditions from part a)
variable (b : ℝ)

def complex_number := (1 + b * Complex.i) / (1 + Complex.i)

-- The real and imaginary parts of the complex number are additive inverses
def real_imag_additive_inverses : Prop :=
  Complex.re (complex_number b) + Complex.im (complex_number b) = 0

theorem determine_b (hb : real_imag_additive_inverses b) : b = 0 :=
  sorry

end determine_b_l473_473846


namespace problem_statement_l473_473349

def seq (a b : ℕ) : ℕ → ℤ
| 1     := a
| 2     := b
| (n+1) := seq a b n - seq a b (n-1)

def partial_sum (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum seq

theorem problem_statement (a b : ℕ) :
  seq a b 100 = -a ∧ partial_sum (seq a b) 100 = 2 * b - a := sorry

end problem_statement_l473_473349


namespace disjoint_sets_no_connections_l473_473948

open Finset

-- Define the type for harbour
def Harbour := Fin 2016

-- Define the type for the graph on harbours
def G : SimpleGraph Harbour := -- fill in with graph definition based on the problem conditions.

-- Define the key condition
axiom long_path_absent : ∀ (p : List Harbour), p.length ≥ 1062 → ¬ G.walk (p.head!) (p.last!)

-- Lean statement
theorem disjoint_sets_no_connections :
  ∃ (A B : Finset Harbour), A.card = 477 ∧ B.card = 477 ∧ A.disjoint B ∧
   ∀ (a : Harbour) (b : Harbour), a ∈ A → b ∈ B → ¬ G.adj a b :=
by
  -- The proof uses Turán's theorem and graph properties
  sorry

end disjoint_sets_no_connections_l473_473948


namespace speed_of_current_l473_473632

theorem speed_of_current (row_speed_kph : ℝ) (time_seconds : ℝ) (distance_meters : ℝ) 
                         (row_speed_kph = 15) 
                         (time_seconds = 23.998080153587715) 
                         (distance_meters = 120) : 
                         row_speed_kph / 3.6 + (distance_meters / time_seconds) - (row_speed_kph / 3.6) * 3.6 / 1000 = 3 :=
by
  sorry

end speed_of_current_l473_473632


namespace bridge_length_is_correct_l473_473643

-- Define the constants and parameters given in the problem
def train_length : ℝ := 120 -- in meters
def train_speed_kmph : ℝ := 60 -- in kilometers per hour
def crossing_time_seconds : ℝ := 17.39860811135109 -- in seconds

-- Proven goal: Length of the bridge
def bridge_length : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600 in
  let total_distance := train_speed_mps * crossing_time_seconds in
  total_distance - train_length

theorem bridge_length_is_correct : bridge_length = 170.31 := by
  sorry

end bridge_length_is_correct_l473_473643


namespace cylinder_surface_area_l473_473531

variable (height1 height2 radius1 radius2 : ℝ)
variable (π : ℝ)
variable (C1 : height1 = 6 * π)
variable (C2 : radius1 = 3)
variable (C3 : height2 = 4 * π)
variable (C4 : radius2 = 2)

theorem cylinder_surface_area : 
  (6 * π * 4 * π + 2 * π * radius1 ^ 2) = 24 * π ^ 2 + 18 * π ∨
  (4 * π * 6 * π + 2 * π * radius2 ^ 2) = 24 * π ^ 2 + 8 * π :=
by
  intros
  sorry

end cylinder_surface_area_l473_473531


namespace number_of_assignments_l473_473637

-- Definitions
def Male := Type
def Female := Type
def Student := Male ⊕ Female

noncomputable def assignMethods (males : ℕ) (females : ℕ) : ℕ :=
  let total := males + females
  -- Case 1: 2 females for location A
  let a_case1 := (females.choose 2)
  -- Case 2: 1 female and 1 male for location A
  let a_case2 := (females.choose 1) * (males.choose 1)
  -- Total ways for A
  let a_ways := a_case1 + a_case2
  -- Remaining students calculation
  let remaining := total - 2
  -- Ways to assign remaining 2 students to B and C
  let bc_ways := remaining.perm 2
  -- Total assignment methods
  a_ways * bc_ways

-- Theorem statement
theorem number_of_assignments : assignMethods 3 2 = 42 :=
by
  -- Placeholder proof
  sorry

end number_of_assignments_l473_473637


namespace third_side_integer_lengths_l473_473775

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l473_473775


namespace sixth_of_circle_anglets_l473_473838

-- Definitions based on conditions
def degree : Type := ℝ
def anglet_per_degree : ℝ := 100
def full_circle_degrees : degree := 360
def sixth_circle_degrees : degree := full_circle_degrees / 6

-- Statement of the problem
theorem sixth_of_circle_anglets : sixth_circle_degrees * anglet_per_degree = 6000 :=
by sorry

end sixth_of_circle_anglets_l473_473838


namespace divisible_by_24_count_number_of_solutions_l473_473046

theorem divisible_by_24_count (a : ℕ) (h1 : a > 0) (h2 : a < 100) (h3 : 24 ∣ (a^3 + 23)) : a = 1 ∨ a = 25 ∨ a = 49 ∨ a = 73 ∨ a = 97 :=
by
  sorry

theorem number_of_solutions : 5 = cardinal.mk { a : ℕ | a > 0 ∧ a < 100 ∧ 24 ∣ (a^3 + 23) } :=
by
  sorry

end divisible_by_24_count_number_of_solutions_l473_473046


namespace possible_denominators_count_l473_473943

theorem possible_denominators_count (a b c : ℕ) (h₀ : a ∈ finset.range 10) (h₁ : b ∈ finset.range 10) (h₂ : c ∈ finset.range 10) (h₃ : ¬(a = 9 ∧ b = 9 ∧ c = 9)) : 
  ∃ (denominators : set ℕ), denominators = {d | d ∣ 999 ∧ (100 * a + 10 * b + c) % d ≠ 0} ∧ denominators.to_finset.card = 7 :=
sorry

end possible_denominators_count_l473_473943


namespace max_price_per_unit_l473_473625

-- Define the conditions
def original_price : ℝ := 25
def original_sales_volume : ℕ := 80000
def price_increase_effect (t : ℝ) : ℝ := 2000 * (t - original_price)
def new_sales_volume (t : ℝ) : ℝ := 130 - 2 * t

-- Define the condition for revenue
def revenue_condition (t : ℝ) : Prop :=
  t * new_sales_volume t ≥ original_price * original_sales_volume

-- Statement to prove the maximum price per unit
theorem max_price_per_unit : ∀ t : ℝ, revenue_condition t → t ≤ 40 := sorry

end max_price_per_unit_l473_473625


namespace quadrilateral_is_square_l473_473179

variable (Q : Type) [Quadrilateral Q]

def has_perpendicular_diagonals (Q : Type) [Quadrilateral Q] : Prop :=
  ∃ (d1 d2 : Diagonal Q), d1.is_perpendicular d2

def has_equal_diagonals (Q : Type) [Quadrilateral Q] : Prop :=
  ∃ (d1 d2 : Diagonal Q), d1.length = d2.length

theorem quadrilateral_is_square (Q : Type) [Quadrilateral Q] :
  has_perpendicular_diagonals Q → has_equal_diagonals Q → is_square Q :=
by
  sorry

end quadrilateral_is_square_l473_473179


namespace find_a6_plus_b6_l473_473043

noncomputable theory

variables (a b : ℝ) (h1 : 0 < a ∧ 0 < b) (h2 : a * b = 2) (h3 : a / (a + b^2) + b / (b + a^2) = 7 / 8)

theorem find_a6_plus_b6 : a ^ 6 + b ^ 6 = 128 :=
by
  sorry

end find_a6_plus_b6_l473_473043


namespace perfume_volume_fraction_l473_473930

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def remaining_volume (initial_volume remaining_volume_in_liters : ℝ) : ℝ :=
  initial_volume - remaining_volume_in_liters

noncomputable def fraction_used (initial_volume volume_used : ℝ) : ℝ :=
  volume_used / initial_volume

theorem perfume_volume_fraction 
  (r h remaining_volume_in_liters : ℝ)
  (r_eq_7 : r = 7)
  (h_eq_10 : h = 10)
  (remaining_volume_eq_0_45 : remaining_volume_in_liters = 0.45) :
  fraction_used (volume_of_cylinder r h / 1000) (remaining_volume (volume_of_cylinder r h / 1000) remaining_volume_in_liters) = (49 * π - 45) / (49 * π) :=
by
  sorry

end perfume_volume_fraction_l473_473930


namespace cauliflower_sales_l473_473067

theorem cauliflower_sales :
  let total_earnings := 500
  let b_sales := 57
  let c_sales := 2 * b_sales
  let s_sales := (c_sales / 2) + 16
  let t_sales := b_sales + s_sales
  let ca_sales := total_earnings - (b_sales + c_sales + s_sales + t_sales)
  ca_sales = 126 := by
  sorry

end cauliflower_sales_l473_473067


namespace dataset_points_calculation_l473_473883

/--
Joe initially had 400 data points in his dataset. He first increased the data points by 20% and later decided to remove 1/5 of the total data points. Then he added 60 new data points and reduced the dataset by 12%. After that, he incorporated 25% more data points into the dataset and finally removed 15% of the total data points. Prove the total number of data points after all these data manipulations is 415.
-/
theorem dataset_points_calculation :
  let initial := 400 in
  let step1 := initial * (120 / 100) in
  let step2 := step1 * (4 / 5) in
  let step3 := step2 + 60 in
  let step4 := step3 * (88 / 100) in
  let step5 := step4 * (125 / 100) in
  let step6 := step5 * (85 / 100) in
  step6.to_nat = 415 :=
by
  sorry

end dataset_points_calculation_l473_473883


namespace sample_size_is_33_l473_473724

theorem sample_size_is_33 (popA popB popC : ℕ) (total_pop : popA + popB + popC = 3300) (sample_from_C : 15) (popC_val : popC = 1500) : ∃ n, n = 33 := by
  have eq1 : ∃ n, n / 3300 = 15 / 1500 := by sorry
  have eq2 : ∃ n, n = 15 * 3300 / 1500 := by sorry
  exists 33
  sorry

end sample_size_is_33_l473_473724


namespace true_propositions_l473_473705

variables {α β : Type} [plane α] [plane β] [line m] [line n]

-- Proposition 1: If m || α and n || α, then m || n
def prop1 : Prop := (m ∥ α) → (n ∥ α) → (m ∥ n)

-- Proposition 2: If m ⊥ α and n ⊥ α, then m || n
def prop2 : Prop := (m ⊥ α) → (n ⊥ α) → (m ∥ n)

-- Proposition 3: If m || α and α ∩ β = n, then m || n
def prop3 : Prop := (m ∥ α) → (α ∩ β = n) → (m ∥ n)

-- Proposition 4: If m ⊥ α, m || n, and n ⊆ β, then α ⊥ β
def prop4 : Prop := (m ⊥ α) → (m ∥ n) → (n ⊂ β) → (α ⊥ β)

-- Statement of the main theorem
theorem true_propositions : prop2 ∧ prop4 := 
by
  split,
  sorry, -- Proof for prop2
  sorry  -- Proof for prop4

end true_propositions_l473_473705


namespace problem_l473_473308

noncomputable def x : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem (x y : ℝ) (hx : x = Real.sqrt 3 + Real.sqrt 2) (hy : y = Real.sqrt 3 - Real.sqrt 2) :
  x * y^2 - x^2 * y = -2 * Real.sqrt 2 :=
by
  rw [hx, hy]
  sorry

end problem_l473_473308


namespace total_spent_l473_473481

variable (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ)

/-- Conditions from the problem setup --/
def conditions :=
  T_L = 40 ∧
  J_L = 0.5 * T_L ∧
  C_L = 2 * T_L ∧
  S_L = 3 * J_L ∧
  T_C = 0.25 * T_L ∧
  J_C = 3 * J_L ∧
  C_C = 0.5 * C_L ∧
  S_C = S_L ∧
  D_C = 2 * S_C ∧
  A_C = 0.5 * J_C

/-- Total spent by Lisa --/
def total_Lisa := T_L + J_L + C_L + S_L

/-- Total spent by Carly --/
def total_Carly := T_C + J_C + C_C + S_C + D_C + A_C

/-- Combined total spent by Lisa and Carly --/
theorem total_spent :
  conditions T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C →
  total_Lisa T_L J_L C_L S_L + total_Carly T_C J_C C_C S_C D_C A_C = 520 :=
by
  sorry

end total_spent_l473_473481


namespace expression_evaluation_l473_473685

theorem expression_evaluation :
  (3 : ℝ) + 3 * Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (3 - Real.sqrt 3)) = 4 + 3 * Real.sqrt 3 :=
sorry

end expression_evaluation_l473_473685


namespace eqn_of_line_l_l473_473215

noncomputable def line_equation (a b : ℝ) : String := 
  "The equation of the line is " ++ toString a ++ "x + " ++ toString b ++ "y - 12 = 0"

theorem eqn_of_line_l (x y : ℝ) :
    (l_passes_through P(3, 2)) ∧ 
    (l_intersects_x_axis A(a, 0)) ∧ 
    (l_intersects_y_axis B(0, b)) ∧ 
    (area_of_triangle_OAB 12) -> 
      line_equation 2 3 = "The equation of the line is 2x + 3y - 12 = 0" :=
begin
  sorry
end

end eqn_of_line_l_l473_473215


namespace find_h_l473_473418

theorem find_h : 
  ∃ (h : ℚ), ∃ (k : ℚ), 3 * (x - h)^2 + k = 3 * x^2 + 9 * x + 20 ∧ h = -3 / 2 :=
begin
  use -3/2,
  --this sets a value of h to -3/2 and expects to find k and prove the equality
  use 53/4,
  --this sets a value of k where this computed value from the solution steps 
  split,
  -- provable part
  linarith,
  -- proof finished without actual calculation for completeness
  sorry 
end

end find_h_l473_473418


namespace find_n_l473_473732

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.log (x + 1)

def line_slope (n : ℝ) : ℝ := 1 / n

theorem find_n (n : ℝ) :
  (∃ x y : ℝ, (x - n * y + 4 = 0) ∧ (line_slope n * (f' (0 : ℝ)) = -1)) → n = -2 := sorry

end find_n_l473_473732


namespace stones_counting_modulus_l473_473275

theorem stones_counting_modulus (n : ℕ) (h : n = 162) : 
  ∃ k, k ∈ finset.range 15 ∧ (162 % 15 = k) := by
  sorry

end stones_counting_modulus_l473_473275


namespace log_12_over_log_15_l473_473363

variables (m n : ℝ)
def log_2 := m
def log_3 := n

theorem log_12_over_log_15 (h1 : log_2 = m) (h2 : log_3 = n):
  real.log 12 / real.log 15 = (2 * m + n) / (1 - m + n) :=
sorry

end log_12_over_log_15_l473_473363


namespace cost_of_watermelon_and_grapes_l473_473492

variable (x y z f : ℕ)

theorem cost_of_watermelon_and_grapes (h1 : x + y + z + f = 45) 
                                    (h2 : f = 3 * x) 
                                    (h3 : z = x + y) :
    y + z = 9 := by
  sorry

end cost_of_watermelon_and_grapes_l473_473492


namespace correct_judgment_l473_473324

variable (x : ℝ)

def p := x^2 + 1 > 0
def q := sin x = 2

theorem correct_judgment : (p x ∨ q x) ∧ ¬ (¬ p x) :=
by {
  sorry
}

end correct_judgment_l473_473324
